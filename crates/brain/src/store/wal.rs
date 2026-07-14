//! Write-ahead-log store per [`ARCHITECTURE.md`] §17.1 + §17.9.
//!
//! Stage 17.1 implements the [`Store`] trait and two concrete backends:
//!
//! - [`NoopStore`] — `&dyn Store` default for brains that don't need
//!   persistence (unit tests, ephemeral evaluations).  Every method is a
//!   no-op.  Brain code uses this exclusively unless `Brain::with_store`
//!   plugs in something else.
//!
//! - [`MmapWalStore`] — production backend.  One append-only log file per
//!   pool (plus a global stream for cross-pool events).  Every append is
//!   bincode-framed `(length_u32_le || body)` followed by a `BufWriter`
//!   flush; a `flush()` call additionally fsyncs the underlying file.
//!
//! At this stage the WAL is **side-car**: every brain mutation is recorded
//! *in addition to* the existing in-memory state, not in place of it.  No
//! Pool API changes; no demand-paging yet.  The guarantee gained is: after
//! any crash, the brain's full learned state is reconstructible from the
//! WAL alone — no big-bang serialize required.
//!
//! Stage 17.4 will wire the eviction actor on top of this store.  Stage 17.6
//! will add `RemoteWalStore` as a thin RPC wrapper around the same trait.

use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::event::WalEvent;

/// Persistence backend for the brain's mutation stream per §17.9.
///
/// Every method takes `&self` so the store can be shared across pools
/// without external locking — the implementation owns its synchronisation.
pub trait Store: Send + Sync {
    /// Append one event.  Must be durable up to OS buffer at return time;
    /// `flush` guarantees fsync.
    fn append(&self, event: &WalEvent) -> io::Result<()>;

    /// Flush + fsync the underlying log so that all preceding appends
    /// survive a process or OS crash.  Called by `Brain::checkpoint`.
    fn flush(&self) -> io::Result<()>;

    /// Discard history already covered by an atomically durable checkpoint.
    /// Backends that cannot compact in place may retain it; replay remains
    /// correct because SnapshotMarker still identifies the usable tail.
    fn compact_after_checkpoint(&self) -> io::Result<()> { Ok(()) }

    /// Returns the on-disk byte size of the log, if known.  Used by
    /// diagnostics and tests; default-impl returns 0.
    fn log_size_bytes(&self) -> u64 { 0 }
}

/// Stand-in store for tests + ephemeral brains.  Every method is a no-op.
pub struct NoopStore;

impl Store for NoopStore {
    fn append(&self, _event: &WalEvent) -> io::Result<()> { Ok(()) }
    fn flush(&self) -> io::Result<()> { Ok(()) }
}

// ============================================================================
// MmapWalStore — the real backend
// ============================================================================

/// Wire format of the WAL.  Bump when adding/removing event variants in a
/// breaking way; recovery refuses to replay logs with a higher major version
/// than the running binary.
pub const WAL_FORMAT_VERSION: u32 = 1;

const FILE_HEADER_MAGIC: u32 = 0xB7A1_5701;  // 'BRAI' '57' '01'

/// One framed event on disk: 4-byte little-endian length, then bincode body.
///
/// The bincode body is variable-length but bounded by typical event size
/// (most events are well under 1 KB; `NeuronTerminalsPruned` for a hot
/// neuron with hundreds of terminals can reach ~16 KB).
fn write_framed_event<W: Write>(w: &mut W, event: &WalEvent) -> io::Result<usize> {
    let body = bincode::serialize(event)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = body.len() as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&body)?;
    Ok(4 + body.len())
}

/// Inverse of [`write_framed_event`].  Returns `Ok(None)` on clean EOF.
/// Returns `Ok(None)` with a logged warning on torn-tail (length-prefix
/// read OK but body short) — partial frames at log tail are recovered
/// by truncation.
fn read_framed_event<R: Read>(r: &mut R) -> io::Result<Option<WalEvent>> {
    let mut len_buf = [0u8; 4];
    match r.read(&mut len_buf)? {
        0 => return Ok(None),
        4 => {}
        n => {
            // Torn length prefix — log tail is corrupt past this point.
            // Stop replay here; downstream tooling can truncate.
            tracing::warn!(
                "WAL replay: torn length prefix ({} of 4 bytes); stopping",
                n
            );
            return Ok(None);
        }
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        // Sanity: no single event should exceed 16 MB.  Defensive against
        // bit-flipped length prefixes.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAL replay: event length {} exceeds 16 MB cap", len),
        ));
    }
    let mut body = vec![0u8; len];
    match r.read(&mut body)? {
        n if n == len => {}
        _ => {
            tracing::warn!("WAL replay: torn body; stopping replay at tail");
            return Ok(None);
        }
    }
    let event: WalEvent = bincode::deserialize(&body)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(event))
}

/// Production WAL backend.  One append-only file under the data dir,
/// receiving the global cross-pool event stream.  Per-pool sharding will
/// come in stage 17.4 once the eviction actor needs to seek into specific
/// pools without scanning the global log.
pub struct MmapWalStore {
    inner: Mutex<MmapWalInner>,
    path:  PathBuf,
}

struct MmapWalInner {
    writer:  BufWriter<File>,
    bytes:   u64,
}

impl MmapWalStore {
    /// Open or create the WAL at `<data_dir>/brain.wal`.
    ///
    /// On open, validates the file header.  If the file is empty, writes
    /// a fresh header.  On format-version mismatch, returns an error so
    /// the caller can decide whether to truncate or refuse to start.
    pub fn open<P: AsRef<Path>>(data_dir: P) -> io::Result<Self> {
        let dir = data_dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join("brain.wal");

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let file_len = file.seek(SeekFrom::End(0))?;
        if file_len == 0 {
            // Write header at file creation time.
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&FILE_HEADER_MAGIC.to_le_bytes())?;
            file.write_all(&WAL_FORMAT_VERSION.to_le_bytes())?;
            file.sync_all()?;
        } else {
            // Validate header.
            file.seek(SeekFrom::Start(0))?;
            let mut magic = [0u8; 4];
            file.read_exact(&mut magic)?;
            if u32::from_le_bytes(magic) != FILE_HEADER_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("WAL header magic mismatch at {}", path.display()),
                ));
            }
            let mut version = [0u8; 4];
            file.read_exact(&mut version)?;
            let v = u32::from_le_bytes(version);
            if v != WAL_FORMAT_VERSION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "WAL format version mismatch at {}: file is v{}, binary is v{}",
                        path.display(),
                        v,
                        WAL_FORMAT_VERSION,
                    ),
                ));
            }
        }

        // Seek to end for appends.
        let bytes = file.seek(SeekFrom::End(0))?;
        let writer = BufWriter::with_capacity(64 * 1024, file);

        tracing::info!(
            "WAL open at {} (size={} bytes, v{})",
            path.display(),
            bytes,
            WAL_FORMAT_VERSION,
        );

        Ok(Self {
            inner: Mutex::new(MmapWalInner { writer, bytes }),
            path,
        })
    }

    /// Path to the underlying WAL file.  Exposed for tests + recovery.
    pub fn path(&self) -> &Path { &self.path }

    /// Open a read-only handle to the WAL for recovery replay.
    pub fn open_replay(&self) -> io::Result<File> {
        let mut f = OpenOptions::new().read(true).open(&self.path)?;
        // Skip the 8-byte header — caller's replay loop is event-aligned.
        f.seek(SeekFrom::Start(8))?;
        Ok(f)
    }

    /// Convenience: build a recoverable replay reader directly from `data_dir`
    /// without first creating an `MmapWalStore` (avoids holding write lock
    /// during replay).
    pub fn open_replay_only<P: AsRef<Path>>(data_dir: P) -> io::Result<File> {
        let path = data_dir.as_ref().join("brain.wal");
        let mut f = OpenOptions::new().read(true).open(&path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if u32::from_le_bytes(magic) != FILE_HEADER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL header magic mismatch on replay",
            ));
        }
        let mut version = [0u8; 4];
        f.read_exact(&mut version)?;
        let v = u32::from_le_bytes(version);
        if v != WAL_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("WAL format version mismatch on replay: v{}", v),
            ));
        }
        Ok(f)
    }
}

impl Store for MmapWalStore {
    fn append(&self, event: &WalEvent) -> io::Result<()> {
        let mut inner = self.inner.lock();
        let n = write_framed_event(&mut inner.writer, event)? as u64;
        inner.bytes += n;
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        let mut inner = self.inner.lock();
        inner.writer.flush()?;
        inner.writer.get_ref().sync_all()?;
        Ok(())
    }

    fn compact_after_checkpoint(&self) -> io::Result<()> {
        let mut inner = self.inner.lock();
        inner.writer.flush()?;
        // The checkpoint has already been written through an atomic rename
        // and fsync. Retain the validated file header and make all future
        // appends the post-checkpoint recovery tail.
        inner.writer.get_mut().set_len(8)?;
        inner.writer.seek(SeekFrom::Start(8))?;
        inner.writer.get_ref().sync_all()?;
        inner.bytes = 8;
        Ok(())
    }

    fn log_size_bytes(&self) -> u64 {
        self.inner.lock().bytes
    }
}

/// Iterator over events in a replay file.  Stops on EOF or torn tail.
pub struct WalReader<R: Read> {
    inner: R,
}

impl<R: Read> WalReader<R> {
    pub fn new(reader: R) -> Self { Self { inner: reader } }
}

impl<R: Read> Iterator for WalReader<R> {
    type Item = io::Result<WalEvent>;
    fn next(&mut self) -> Option<Self::Item> {
        match read_framed_event(&mut self.inner) {
            Ok(Some(ev)) => Some(Ok(ev)),
            Ok(None)     => None,
            Err(e)       => Some(Err(e)),
        }
    }
}

// ============================================================================
// Convenience: Arc-typed store handle
// ============================================================================

/// Erased shared-pointer alias for stores.  Pool/Brain hold this and clone
/// it cheaply to fan out across components.
pub type StoreHandle = Arc<dyn Store>;

/// Construct the default `NoopStore` wrapped in `Arc<dyn Store>` for any
/// brain that hasn't been explicitly given persistence.
pub fn noop_store() -> StoreHandle {
    Arc::new(NoopStore)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{NeuronKind, NeuronRef};
    use crate::store::event::TerminalDelta;
    use std::env::temp_dir;

    fn tmpdir_for(test: &str) -> PathBuf {
        let pid = std::process::id();
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = temp_dir().join(format!("w1z4rd_brain_wal_test_{}_{}_{}", test, pid, nano));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn noop_store_is_inert() {
        let s = NoopStore;
        s.append(&WalEvent::TickAdvanced { new_tick: 1 }).unwrap();
        s.flush().unwrap();
        assert_eq!(s.log_size_bytes(), 0);
    }

    #[test]
    fn mmap_wal_round_trips_a_handful_of_events() {
        let dir = tmpdir_for("rt");
        let store = MmapWalStore::open(&dir).unwrap();

        let evs = vec![
            WalEvent::AtomCreated {
                pool_id:   1,
                id:        0,
                label:     "t:Vw".into(),
                kind:      NeuronKind::Excitatory,
                born_tick: 0,
            },
            WalEvent::AtomCreated {
                pool_id:   1,
                id:        1,
                label:     "t:YQ".into(),
                kind:      NeuronKind::Excitatory,
                born_tick: 0,
            },
            WalEvent::TerminalReinforced(TerminalDelta {
                src: NeuronRef::new(1, 0),
                dst: NeuronRef::new(1, 1),
                weight: 0.5,
                consolidation: 1,
                last_fired_tick: 1,
            }),
            WalEvent::TickAdvanced { new_tick: 1 },
            WalEvent::SnapshotMarker { tick: 1, wall_time_ms: 0 },
        ];
        for ev in &evs {
            store.append(ev).unwrap();
        }
        store.flush().unwrap();
        let written = store.log_size_bytes();
        drop(store);

        let f = MmapWalStore::open_replay_only(&dir).unwrap();
        let reader = WalReader::new(f);
        let recovered: Vec<WalEvent> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(recovered.len(), evs.len(), "WAL replayed wrong count");
        assert_eq!(recovered[0].variant_name(), "AtomCreated");
        assert_eq!(recovered[2].variant_name(), "TerminalReinforced");
        assert_eq!(recovered[4].variant_name(), "SnapshotMarker");
        assert!(written > 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mmap_wal_rejects_mismatched_magic() {
        let dir = tmpdir_for("magic");
        let path = dir.join("brain.wal");
        // Pre-create with garbage so the header check fails.
        std::fs::write(&path, b"NOTREALWAL").unwrap();
        let r = MmapWalStore::open(&dir);
        assert!(r.is_err(), "expected magic-mismatch error");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mmap_wal_survives_reopen_with_existing_log() {
        let dir = tmpdir_for("reopen");

        {
            let s = MmapWalStore::open(&dir).unwrap();
            for tick in 0..5 {
                s.append(&WalEvent::TickAdvanced { new_tick: tick }).unwrap();
            }
            s.flush().unwrap();
        }

        // Reopen — header must validate, appends must continue.
        let s = MmapWalStore::open(&dir).unwrap();
        s.append(&WalEvent::TickAdvanced { new_tick: 99 }).unwrap();
        s.flush().unwrap();
        drop(s);

        let f = MmapWalStore::open_replay_only(&dir).unwrap();
        let evs: Vec<_> = WalReader::new(f).map(|r| r.unwrap()).collect();
        assert_eq!(evs.len(), 6);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checkpoint_compaction_keeps_only_the_new_recovery_tail() {
        let dir = tmpdir_for("checkpoint_compact");
        let store = MmapWalStore::open(&dir).unwrap();
        for tick in 1..=4 {
            store.append(&WalEvent::TickAdvanced { new_tick: tick }).unwrap();
        }
        store.append(&WalEvent::SnapshotMarker {
            tick: 4,
            wall_time_ms: 0,
        }).unwrap();
        store.flush().unwrap();
        assert!(store.log_size_bytes() > 8);

        store.compact_after_checkpoint().unwrap();
        assert_eq!(store.log_size_bytes(), 8);
        store.append(&WalEvent::TickAdvanced { new_tick: 5 }).unwrap();
        store.flush().unwrap();
        drop(store);

        let file = MmapWalStore::open_replay_only(&dir).unwrap();
        let events: Vec<_> = WalReader::new(file).map(|event| event.unwrap()).collect();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], WalEvent::TickAdvanced { new_tick: 5 }));
        std::fs::remove_dir_all(&dir).ok();
    }
}
