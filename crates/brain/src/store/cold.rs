//! Cold-tier neuron storage per [`ARCHITECTURE.md`] §17.4.
//!
//! A pool's cold tier is an append-only binary file on disk holding the
//! serialised bytes of neurons evicted from the working set.  Reads
//! happen by absolute byte offset — the in-RAM index
//! (`Pool::cold_offsets`) maps `NeuronId → offset`.  No directory
//! seeking, no LSM compaction in this first cut: every eviction appends
//! a fresh record and the index points to the latest offset; older
//! versions of the same neuron stay on disk as garbage and are
//! reclaimed by a future compaction pass (Stage 17.4 follow-up).
//!
//! # File format
//!
//! Each record on disk is:
//! ```text
//! +----+----+----+----+----+----+----+----+ +-- ... --+
//! |  body_len: u32 little-endian          | | body    |
//! +----+----+----+----+----+----+----+----+ +-- ... --+
//!                                           ^^^^^^^^^^^^
//!                                           bincode::serialize(&Neuron)
//! ```
//!
//! The offset stored in `cold_offsets` points to the FIRST byte of the
//! length prefix.  Reading a neuron: seek → read 4 bytes length → read
//! `length` bytes → bincode deserialise.
//!
//! There is **no file header**.  The cold tier is anonymous — its name
//! is `<data_dir>/cold/pool_{id}.cold` and the file simply IS a sequence
//! of records, validated lazily on first read.
//!
//! # Concurrency
//!
//! - Append: serialised through a `parking_lot::Mutex<File>` write
//!   handle held inside the pool.  One writer at a time.
//! - Read: each read opens a fresh `File` handle so concurrent reads
//!   from multiple threads don't fight over a seek cursor.  This is
//!   safe because the file is append-only — reads never see a torn
//!   record provided the writer hasn't yet flushed.
//!
//! # Crash safety
//!
//! - The cold-tier file is fsynced on each `evict_neuron` call so an
//!   abrupt termination doesn't lose the just-evicted record.  At scale
//!   this can be batched; a `flush_cold` API is exposed for that future
//!   optimisation.
//! - The `cold_offsets` index lives on the brain.bin snapshot (added
//!   in §17.4 step 2) AND is serialised to the WAL as a `NeuronEvicted`
//!   event so crash recovery can replay it.  Until step 2 the index is
//!   transient — a process restart re-loads the pool's full neuron set
//!   from brain.bin, and any cold-tier records become unreachable
//!   garbage (compaction territory).

use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::neuron::Neuron;

/// Append-only cold-tier file for one pool.  Owns the writer handle;
/// readers open their own handles via the path.
pub struct ColdTier {
    path:   PathBuf,
    writer: Arc<Mutex<ColdWriter>>,
}

struct ColdWriter {
    file:  File,
    /// Current file length in bytes — also the next-append offset.
    /// Tracked manually so we don't pay a seek() on every evict.
    bytes: u64,
}

impl ColdTier {
    /// Open or create the cold tier at `path`.  Creates parent dirs.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true).write(true).create(true)
            .open(&path)?;
        let bytes = file.seek(SeekFrom::End(0))?;
        Ok(Self {
            path,
            writer: Arc::new(Mutex::new(ColdWriter { file, bytes })),
        })
    }

    pub fn path(&self) -> &Path { &self.path }

    /// Append a serialised neuron and return the offset where the
    /// length-prefix begins.  Caller stores this offset in its index.
    /// fsyncs the file before returning so a crash after this point
    /// does not lose the record.
    pub fn append_neuron(&self, neuron: &Neuron) -> io::Result<u64> {
        let body = bincode::serialize(neuron)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if body.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cold neuron record too large: {}", body.len()),
            ));
        }
        let len_prefix = (body.len() as u32).to_le_bytes();

        let mut w = self.writer.lock();
        let offset = w.bytes;
        w.file.write_all(&len_prefix)?;
        w.file.write_all(&body)?;
        w.file.sync_data()?;
        w.bytes += 4 + body.len() as u64;
        Ok(offset)
    }

    /// Read the neuron at the given offset.  Opens a fresh file handle
    /// for thread-safety with concurrent reads.
    pub fn read_neuron(&self, offset: u64) -> io::Result<Neuron> {
        let mut f = OpenOptions::new().read(true).open(&self.path)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut len_buf = [0u8; 4];
        f.read_exact(&mut len_buf)?;
        let body_len = u32::from_le_bytes(len_buf) as usize;
        // Defensive cap.
        if body_len > 16 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cold record length {} > 16 MB cap at offset {}",
                    body_len, offset),
            ));
        }
        let mut body = vec![0u8; body_len];
        f.read_exact(&mut body)?;
        bincode::deserialize(&body)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Total bytes written.  Used by /stats + the StorageControlState
    /// pressure signal (Stage 17.8 wiring).
    pub fn bytes(&self) -> u64 {
        self.writer.lock().bytes
    }

    /// Explicit flush + fsync.  Idempotent; safe to call between batches.
    pub fn flush(&self) -> io::Result<()> {
        let mut w = self.writer.lock();
        w.file.flush()?;
        w.file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Neuron, NeuronKind, NeuronRef};

    fn tmpdir(test: &str) -> PathBuf {
        let pid = std::process::id();
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        let d = std::env::temp_dir()
            .join(format!("w1z4rd_cold_{}_{}_{}", test, pid, nano));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trip_single_neuron() {
        let dir = tmpdir("rt1");
        let cold = ColdTier::open(dir.join("pool.cold")).unwrap();

        let mut n = Neuron::new_atom(7, "t:abc".into(), NeuronKind::Excitatory, 100);
        n.reinforce_terminal(NeuronRef::new(1, 42), 0.5, 100, 1.0);
        n.bump_salience(0.3);

        let off = cold.append_neuron(&n).unwrap();
        assert_eq!(off, 0, "first append must land at offset 0");

        let restored = cold.read_neuron(off).unwrap();
        assert_eq!(restored.id, n.id);
        assert_eq!(restored.label, n.label);
        assert_eq!(restored.kind, n.kind);
        assert_eq!(restored.terminals.len(), 1);
        assert_eq!(restored.terminals[0].target, n.terminals[0].target);
        assert!((restored.salience - n.salience).abs() < 1e-6);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn round_trip_many_neurons_with_independent_offsets() {
        let dir = tmpdir("rt_many");
        let cold = ColdTier::open(dir.join("pool.cold")).unwrap();
        let mut offsets = Vec::new();
        for i in 0..20u32 {
            let n = Neuron::new_atom(
                i, format!("t:n{i}"), NeuronKind::Excitatory, i as u64,
            );
            offsets.push(cold.append_neuron(&n).unwrap());
        }
        for (i, off) in offsets.iter().enumerate() {
            let restored = cold.read_neuron(*off).unwrap();
            assert_eq!(restored.id, i as u32);
            assert_eq!(restored.label, format!("t:n{i}"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reopen_existing_file_preserves_records() {
        let dir = tmpdir("reopen");
        let path = dir.join("pool.cold");
        let mut written_offs = Vec::new();
        {
            let cold = ColdTier::open(&path).unwrap();
            for i in 0..5u32 {
                let n = Neuron::new_atom(
                    i, format!("t:{i}"), NeuronKind::Excitatory, 0,
                );
                written_offs.push(cold.append_neuron(&n).unwrap());
            }
            cold.flush().unwrap();
        }
        // Reopen — bytes counter must advance past existing length so
        // new appends don't overwrite old data.
        let cold = ColdTier::open(&path).unwrap();
        let next = cold.append_neuron(&Neuron::new_atom(
            99, "t:reopen".into(), NeuronKind::Excitatory, 0,
        )).unwrap();
        assert!(next > *written_offs.last().unwrap(),
            "reopen append offset must exceed last prior offset");
        // Existing records must still be readable at their original offsets.
        for (i, off) in written_offs.iter().enumerate() {
            let restored = cold.read_neuron(*off).unwrap();
            assert_eq!(restored.id, i as u32);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bytes_counter_accurate() {
        let dir = tmpdir("bytes");
        let cold = ColdTier::open(dir.join("pool.cold")).unwrap();
        assert_eq!(cold.bytes(), 0);
        let n = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
        let body_len = bincode::serialize(&n).unwrap().len();
        let _off = cold.append_neuron(&n).unwrap();
        assert_eq!(cold.bytes(), 4 + body_len as u64);
        std::fs::remove_dir_all(&dir).ok();
    }
}
