//! Offline compaction of a `.wbrain` container — the §17.4 follow-up.
//!
//! `cold.rs` has said since the first cut that "every eviction appends a fresh
//! record and the index points to the latest offset; older versions of the same
//! neuron stay on disk as garbage and are reclaimed by a future compaction pass
//! (Stage 17.4 follow-up)". That pass was never built, so nothing in the system
//! has ever returned a superseded neuron body to the filesystem.
//!
//! Measured 2026-09-10 on the private training host: `brain.wbrain` held
//! 576.7 GB for 5,086,420 neurons — about 110 KB of file per neuron, i.e. the
//! container was overwhelmingly garbage — while the volume burned 206–257 GB/h
//! and stood ~2 h from ENOSPC. The previous exhaustion crash-looped the service
//! wrapper 115 times on a 6-byte `node.pid` write, which reads as a finished
//! stage rather than as a full disk.
//!
//! # Why this is a byte copy and not a re-serialization
//!
//! Neuron bodies are opaque here. Compaction reads a record's 24-byte header,
//! copies `body_len` bytes verbatim, and records the new offset. It never calls
//! `bincode::deserialize`. That means a body this build cannot decode still
//! round-trips intact, and a compaction pass cannot damage a field it does not
//! know exists.
//!
//! # Where the offsets hide
//!
//! Copying records is the easy half. An offset that is not rewritten points at
//! whatever bytes now occupy its old address, so every holder must be found:
//!
//! 1. `PoolContainerManifest::neuron_slot_table` — body is a dense array of
//!    24-byte slots whose first 8 bytes are an absolute neuron offset.
//! 2. `PoolContainerManifest::neuron_offsets` — the small-store equivalent.
//! 3. `PoolContainerManifest::label_indexes` — the references move; the bodies
//!    address themselves relatively and copy verbatim.
//! 4. `WbrainPoolMetadata::legacy_sequence_ledger` and
//!    `legacy_concept_sequence_index` — buried in an opaque `pool_metadata`
//!    blob.
//! 5. `WbrainBrainMetadata::binding_posting_indexes` — buried in an opaque
//!    `brain_metadata` blob.
//! 6. `W1ZCGEN1` / `W1ZSGEN1` generation directories — auxiliary bodies that
//!    nest `(offset, len)` pairs pointing at OTHER auxiliary records.
//!
//! (4) and (5) are remapped by functions that live beside their struct
//! definitions and destructure exhaustively, so a new reference field is a
//! compile error rather than silent corruption. (6) is why an auxiliary body
//! cannot simply be copied: this module classifies each body by its magic and
//! **fails closed** — an unrecognized body that is too small to classify stops
//! the pass instead of being copied blind.
//!
//! # Durability
//!
//! The destination is written and fsynced in full, and its manifest is
//! published only after every record it references is durable, because
//! `commit_manifest` syncs the body before touching a header slot. The caller
//! swaps files only after `verify` reopens the destination and re-reads every
//! live neuron offset. The source is never mutated.

use ahash::AHashMap;
use std::io;
use std::path::Path;

use crate::neuron::PoolId;
use crate::pool::{GENERATION_DIRECTORY_MAGICS, remap_pool_metadata_refs};
use crate::store::container::{
    AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
};
use crate::store::wbrain_store::{NEURON_SLOT_BYTES, NEURON_SLOT_TABLE_KIND, SLOT_PRESENT};
use crate::brain::wbrain_metadata::remap_brain_metadata_refs;

/// Streaming buffer for verbatim record copies. Sized so a multi-megabyte
/// neuron body moves in a few writes without ever holding a whole record.
const COPY_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Slots rewritten per batch when rebuilding a neuron slot table. At 24 bytes a
/// slot this buffers 24 MB for a 1M-neuron pool.
const SLOT_BATCH: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionReport {
    pub pools: u64,
    pub neurons_copied: u64,
    pub auxiliary_copied: u64,
    pub generation_directories_rewritten: u64,
    pub source_bytes: u64,
    pub destination_bytes: u64,
}

impl CompactionReport {
    /// Bytes the pass returns to the filesystem once the source is unlinked.
    pub fn reclaimed_bytes(&self) -> u64 {
        self.source_bytes.saturating_sub(self.destination_bytes)
    }
}

/// Rewrite `source` into `destination`, keeping only live records.
///
/// `destination` must not already exist as a populated container; a fresh path
/// is expected. The source is opened read/write only because `BrainContainer`
/// has no read-only constructor — nothing here writes to it.
pub fn compact(source: &Path, destination: &Path) -> io::Result<CompactionReport> {
    let mut src = BrainContainer::open(source)?;
    let manifest = src.manifest().cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "source container has no committed manifest; refusing to compact",
        )
    })?;
    let mut dst = BrainContainer::open(destination)?;
    if dst.byte_len()? > crate::store::container::header_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "destination container already holds records; refusing to compact into it",
        ));
    }

    let mut state = Compactor {
        buffer: vec![0_u8; COPY_BUFFER_BYTES],
        auxiliary: AHashMap::new(),
        report: CompactionReport::default(),
    };

    let BrainContainerManifest {
        generation,
        tick,
        brain_metadata,
        pools,
    } = manifest;

    let mut rebuilt_pools = Vec::with_capacity(pools.len());
    for pool in pools {
        rebuilt_pools.push(state.compact_pool(&mut src, &mut dst, pool)?);
        state.report.pools += 1;
    }

    // Brain metadata last: its posting-index records are shared across pools,
    // so by now most are already in the map and remap becomes a lookup.
    let brain_metadata = {
        let mut remap =
            |reference: AuxiliaryRecordRef| state.copy_auxiliary(&mut src, &mut dst, reference);
        remap_brain_metadata_refs(&brain_metadata, &mut remap)?
    };

    dst.flush()?;
    dst.commit_manifest(BrainContainerManifest {
        generation,
        tick,
        brain_metadata,
        pools: rebuilt_pools,
    })?;
    dst.flush()?;

    state.report.source_bytes = src.byte_len()?;
    state.report.destination_bytes = dst.byte_len()?;
    Ok(state.report)
}

/// Reopen a compacted container and read every live neuron offset back.
///
/// A compaction that is announced but not verified is the same class of error
/// as a supervisor that reports `active` while admitting nothing. This reads
/// the record header at each live offset and confirms the neuron ID recorded in
/// the slot table matches the one in the record it now points at, which is
/// exactly the property a mis-remapped offset would break.
pub fn verify(path: &Path, expected: &CompactionReport) -> io::Result<u64> {
    let mut container = BrainContainer::open(path)?;
    let manifest = container.manifest().cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "compacted container published no manifest",
        )
    })?;
    let mut checked = 0_u64;
    for pool in &manifest.pools {
        let offsets = live_offsets(&mut container, pool)?;
        for (id, offset) in offsets {
            let (record_pool, neuron) = container.read_neuron_at(offset)?;
            if record_pool != pool.pool_id || neuron.id != id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "compacted offset {offset} resolves to pool {record_pool} neuron {} \
                         but the slot table claims pool {} neuron {id}",
                        neuron.id, pool.pool_id
                    ),
                ));
            }
            checked += 1;
        }
    }
    if checked != expected.neurons_copied {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "compaction copied {} neurons but verification found {checked}",
                expected.neurons_copied
            ),
        ));
    }
    Ok(checked)
}

/// Live `(neuron id, offset)` pairs for one pool, from whichever addressing
/// form its manifest uses.
fn live_offsets(
    container: &mut BrainContainer,
    pool: &PoolContainerManifest,
) -> io::Result<Vec<(u32, u64)>> {
    let mut live = Vec::new();
    if let Some(table) = pool.neuron_slot_table {
        let slots = table.len / NEURON_SLOT_BYTES;
        let mut raw = [0_u8; NEURON_SLOT_BYTES as usize];
        for index in 0..slots {
            container.read_auxiliary_exact(table, index * NEURON_SLOT_BYTES, &mut raw)?;
            let offset = u64::from_le_bytes(raw[0..8].try_into().unwrap());
            let flags = raw[17];
            if flags & SLOT_PRESENT != 0 && offset != 0 {
                live.push((index as u32, offset));
            }
        }
    } else {
        for (index, slot) in pool.neuron_offsets.iter().enumerate() {
            if let Some(offset) = slot {
                live.push((index as u32, *offset));
            }
        }
    }
    Ok(live)
}

struct Compactor {
    buffer: Vec<u8>,
    /// Source auxiliary offset → destination reference. Shared records (a
    /// posting index referenced by several pools) are copied once.
    auxiliary: AHashMap<u64, AuxiliaryRecordRef>,
    report: CompactionReport,
}

impl Compactor {
    fn compact_pool(
        &mut self,
        src: &mut BrainContainer,
        dst: &mut BrainContainer,
        pool: PoolContainerManifest,
    ) -> io::Result<PoolContainerManifest> {
        let PoolContainerManifest {
            pool_id,
            neuron_count,
            neuron_capacity,
            neuron_slot_table,
            label_indexes,
            neuron_offsets,
            labels,
            pool_metadata,
        } = pool;

        // ---- Neuron bodies, and the addressing structure that finds them.
        let (neuron_slot_table, neuron_offsets) = match neuron_slot_table {
            Some(table) => {
                let rebuilt = self.rebuild_slot_table(src, dst, pool_id, table)?;
                (Some(rebuilt), Vec::new())
            }
            None => {
                let mut moved = Vec::with_capacity(neuron_offsets.len());
                for slot in neuron_offsets {
                    match slot {
                        Some(offset) => {
                            let new_offset =
                                src.copy_neuron_record_into(offset, dst, &mut self.buffer)?;
                            self.report.neurons_copied += 1;
                            moved.push(Some(new_offset));
                        }
                        None => moved.push(None),
                    }
                }
                (None, moved)
            }
        };

        // ---- Label indexes: references move, bodies are position independent.
        let mut moved_label_indexes = Vec::with_capacity(label_indexes.len());
        for reference in label_indexes {
            moved_label_indexes.push(self.copy_auxiliary(src, dst, reference)?);
        }

        // ---- Offsets hidden inside the opaque metadata blob.
        let pool_metadata = {
            let mut remap = |reference: AuxiliaryRecordRef| self.copy_auxiliary(src, dst, reference);
            remap_pool_metadata_refs(&pool_metadata, &mut remap)?
        };

        Ok(PoolContainerManifest {
            pool_id,
            neuron_count,
            neuron_capacity,
            neuron_slot_table,
            label_indexes: moved_label_indexes,
            neuron_offsets,
            labels,
            pool_metadata,
        })
    }

    /// Copy every live neuron in a slot table and write the table back with the
    /// new offsets.
    ///
    /// Only bytes 0..8 of each slot are rewritten. `born_tick`, kind and flags
    /// are carried through untouched, so a slot's identity survives a pass that
    /// only ever intended to move its address.
    fn rebuild_slot_table(
        &mut self,
        src: &mut BrainContainer,
        dst: &mut BrainContainer,
        pool_id: PoolId,
        table: AuxiliaryRecordRef,
    ) -> io::Result<AuxiliaryRecordRef> {
        if table.len % NEURON_SLOT_BYTES != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "neuron slot table is not a whole number of slots",
            ));
        }
        let slots = table.len / NEURON_SLOT_BYTES;
        let mut rebuilt: Vec<u8> = Vec::with_capacity(
            usize::try_from(table.len)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "slot table too large"))?,
        );
        let mut batch = vec![0_u8; SLOT_BATCH * NEURON_SLOT_BYTES as usize];
        let mut done = 0_u64;
        while done < slots {
            let take = (slots - done).min(SLOT_BATCH as u64);
            let bytes = (take * NEURON_SLOT_BYTES) as usize;
            src.read_auxiliary_exact(table, done * NEURON_SLOT_BYTES, &mut batch[..bytes])?;
            for index in 0..take as usize {
                let start = index * NEURON_SLOT_BYTES as usize;
                let slot = &mut batch[start..start + NEURON_SLOT_BYTES as usize];
                let offset = u64::from_le_bytes(slot[0..8].try_into().unwrap());
                let present = slot[17] & SLOT_PRESENT != 0 && offset != 0;
                if present {
                    let new_offset = src.copy_neuron_record_into(offset, dst, &mut self.buffer)?;
                    slot[0..8].copy_from_slice(&new_offset.to_le_bytes());
                    self.report.neurons_copied += 1;
                }
                rebuilt.extend_from_slice(slot);
            }
            done += take;
        }
        let reference = dst.append_auxiliary_body(pool_id, NEURON_SLOT_TABLE_KIND, &rebuilt)?;
        self.report.auxiliary_copied += 1;
        Ok(reference)
    }

    /// Move one auxiliary record, rewriting it if its body nests references.
    fn copy_auxiliary(
        &mut self,
        src: &mut BrainContainer,
        dst: &mut BrainContainer,
        reference: AuxiliaryRecordRef,
    ) -> io::Result<AuxiliaryRecordRef> {
        if let Some(existing) = self.auxiliary.get(&reference.offset) {
            return Ok(*existing);
        }
        let (pool, kind) = src.auxiliary_header(reference)?;
        let prefix = src.read_auxiliary_prefix(reference, 8)?;
        let is_directory = prefix.len() == 8
            && GENERATION_DIRECTORY_MAGICS
                .iter()
                .any(|magic| prefix.as_slice() == magic.as_slice());

        let moved = if is_directory {
            let body = self.rewrite_generation_directory(src, dst, reference)?;
            self.report.generation_directories_rewritten += 1;
            dst.append_auxiliary_body(pool, kind, &body)?
        } else {
            src.copy_auxiliary_record_into(reference, dst, &mut self.buffer)?
        };
        self.report.auxiliary_copied += 1;
        self.auxiliary.insert(reference.offset, moved);
        Ok(moved)
    }

    /// Rebuild a `W1ZCGEN1` / `W1ZSGEN1` directory body.
    ///
    /// Layout is `[8B magic][8B count][count × (u64 offset, u64 len)]`. Each
    /// child is copied first — recursively, since a directory may reference a
    /// directory — and the entry is rewritten to its new address.
    fn rewrite_generation_directory(
        &mut self,
        src: &mut BrainContainer,
        dst: &mut BrainContainer,
        reference: AuxiliaryRecordRef,
    ) -> io::Result<Vec<u8>> {
        let header = src.read_auxiliary_prefix(reference, 16)?;
        if header.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "generation directory is shorter than its header",
            ));
        }
        let count = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let required = 16_u64
            .checked_add(count.checked_mul(16).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "generation directory overflow")
            })?)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "generation directory overflow")
            })?;
        if required > reference.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "generation directory is truncated",
            ));
        }
        // The trailing bytes after the entry table are payload the directory
        // format may carry; preserve them verbatim.
        let whole = src.read_auxiliary_prefix(reference, reference.len as usize)?;
        let mut body = whole.clone();
        for index in 0..count {
            let at = (16 + index * 16) as usize;
            let child = AuxiliaryRecordRef {
                offset: u64::from_le_bytes(whole[at..at + 8].try_into().unwrap()),
                len: u64::from_le_bytes(whole[at + 8..at + 16].try_into().unwrap()),
            };
            if child.offset == 0 || child.len < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "generation directory contains an invalid reference",
                ));
            }
            let moved = self.copy_auxiliary(src, dst, child)?;
            body[at..at + 8].copy_from_slice(&moved.offset.to_le_bytes());
            body[at + 8..at + 16].copy_from_slice(&moved.len.to_le_bytes());
        }
        Ok(body)
    }
}

/// Write a compacted container beside `path` and swap it in.
///
/// The source is renamed aside rather than unlinked, so a failure at any point
/// leaves a complete brain on disk. The caller deletes the retired file only
/// after the swap is durable — which is also the only moment the space is
/// actually returned.
pub fn compact_in_place(path: &Path, keep_retired_as: Option<&Path>) -> io::Result<CompactionReport> {
    let working = path.with_extension("compacting");
    if working.exists() {
        std::fs::remove_file(&working)?;
    }
    let report = compact(path, &working)?;
    verify(&working, &report)?;

    let retired = match keep_retired_as {
        Some(target) => target.to_path_buf(),
        None => path.with_extension("retired"),
    };
    std::fs::rename(path, &retired)?;
    if let Err(error) = std::fs::rename(&working, path) {
        // Put the original back: a half-swapped brain is the one outcome worse
        // than a full disk.
        std::fs::rename(&retired, path)?;
        return Err(error);
    }
    if keep_retired_as.is_none() {
        std::fs::remove_file(&retired)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Neuron, NeuronKind, NeuronRef, Terminal};

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "w1z4rd_compact_{name}_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn concept(id: u32, label: &str, terminals: usize) -> Neuron {
        let mut neuron = Neuron::new_concept(
            id,
            label.into(),
            NeuronKind::Excitatory,
            vec![NeuronRef::new(0, 1)],
            1,
        );
        neuron.terminals = (0..terminals)
            .map(|t| Terminal::new(NeuronRef::new(0, t as u32), 0.25, 1))
            .collect();
        neuron
    }

    /// The property the whole pass exists for: superseded bodies are dropped
    /// and the survivors still read back.
    #[test]
    fn compaction_drops_superseded_bodies_and_preserves_live_ones() {
        let dir = tmpdir("supersede");
        let source = dir.join("brain.wbrain");
        let destination = dir.join("out.wbrain");

        let live_offset;
        {
            let mut container = BrainContainer::open(&source).unwrap();
            // Ten stale generations of the same neuron, then the live one.
            let mut offset = 0;
            for generation in 0..10 {
                offset = container
                    .append_neuron(1, &concept(0, "c:live", 500 + generation))
                    .unwrap();
            }
            live_offset = offset;
            container
                .commit_manifest(BrainContainerManifest {
                    generation: 2,
                    tick: 77,
                    brain_metadata: Vec::new(),
                    pools: vec![PoolContainerManifest {
                        pool_id: 1,
                        neuron_count: 1,
                        neuron_capacity: 1,
                        neuron_slot_table: None,
                        label_indexes: Vec::new(),
                        neuron_offsets: vec![Some(live_offset)],
                        labels: vec![("c:live".into(), 0)],
                        pool_metadata: Vec::new(),
                    }],
                })
                .unwrap();
        }

        let before = BrainContainer::open(&source).unwrap().byte_len().unwrap();
        let report = compact(&source, &destination).unwrap();
        verify(&destination, &report).unwrap();

        assert_eq!(report.neurons_copied, 1, "only the live body should move");
        assert!(
            report.destination_bytes * 4 < before,
            "ten superseded generations must not survive: {} vs {before}",
            report.destination_bytes
        );

        let mut compacted = BrainContainer::open(&destination).unwrap();
        let manifest = compacted.manifest().cloned().unwrap();
        assert_eq!(manifest.tick, 77, "tick must survive compaction");
        assert_eq!(manifest.generation, 2);
        let offset = manifest.pools[0].neuron_offsets[0].unwrap();
        let (pool, neuron) = compacted.read_neuron_at(offset).unwrap();
        assert_eq!(pool, 1);
        assert_eq!(neuron.label, "c:live");
        assert_eq!(
            neuron.terminals.len(),
            509,
            "the LAST generation is the live one"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A reference nested inside a generation directory must be rewritten. Left
    /// alone it would point at whatever bytes now sit at the old address, which
    /// is the failure mode no neuron-level check would catch.
    #[test]
    fn generation_directory_children_are_remapped() {
        let dir = tmpdir("directory");
        let source = dir.join("brain.wbrain");
        let destination = dir.join("out.wbrain");

        let root;
        {
            let mut container = BrainContainer::open(&source).unwrap();
            // Padding so the child cannot coincidentally land on the same
            // offset in the destination.
            for _ in 0..8 {
                container.append_neuron(1, &concept(0, "c:pad", 200)).unwrap();
            }
            let child = container
                .append_auxiliary_body(1, 9, b"W1ZSEQ01child-payload")
                .unwrap();
            let mut body = Vec::new();
            body.extend_from_slice(b"W1ZSGEN1");
            body.extend_from_slice(&1_u64.to_le_bytes());
            body.extend_from_slice(&child.offset.to_le_bytes());
            body.extend_from_slice(&child.len.to_le_bytes());
            root = container.append_auxiliary_body(1, 10, &body).unwrap();
            let live = container.append_neuron(1, &concept(0, "c:live", 4)).unwrap();
            container
                .commit_manifest(BrainContainerManifest {
                    generation: 2,
                    tick: 5,
                    brain_metadata: Vec::new(),
                    pools: vec![PoolContainerManifest {
                        pool_id: 1,
                        neuron_count: 1,
                        neuron_capacity: 1,
                        neuron_slot_table: None,
                        label_indexes: vec![root],
                        neuron_offsets: vec![Some(live)],
                        labels: Vec::new(),
                        pool_metadata: Vec::new(),
                    }],
                })
                .unwrap();
        }

        let report = compact(&source, &destination).unwrap();
        verify(&destination, &report).unwrap();
        assert_eq!(report.generation_directories_rewritten, 1);

        let mut compacted = BrainContainer::open(&destination).unwrap();
        let manifest = compacted.manifest().cloned().unwrap();
        let moved_root = manifest.pools[0].label_indexes[0];
        assert_ne!(
            moved_root.offset, root.offset,
            "the directory itself must have moved, or this proves nothing"
        );
        let body = compacted.read_auxiliary(moved_root).unwrap();
        let child = AuxiliaryRecordRef {
            offset: u64::from_le_bytes(body[16..24].try_into().unwrap()),
            len: u64::from_le_bytes(body[24..32].try_into().unwrap()),
        };
        assert_eq!(
            compacted.read_auxiliary(child).unwrap(),
            b"W1ZSEQ01child-payload",
            "the nested reference must resolve in the COMPACTED file"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// The shape the production brain actually has.
    ///
    /// A 5M-neuron pool addresses its bodies through a paged slot table, not
    /// through `neuron_offsets`; the offset-vector tests above exercise a form
    /// the training host never emits. This builds a container through the real
    /// store, compacts it, and reopens it through the real store again — so the
    /// assertion is that the BRAIN still works, not merely that the bytes moved.
    #[test]
    fn compaction_preserves_a_paged_slot_table_brain() {
        use crate::store::neuron_store::NeuronStore;
        use crate::store::wbrain_store::WbrainFile;

        let dir = tmpdir("paged");
        let source = dir.join("brain.wbrain");
        let destination = dir.join("out.wbrain");
        {
            let file = WbrainFile::open(&source).unwrap();
            let pool = file.pool(7);
            pool.prepare_paged_slots(4).unwrap();
            // Establish the slot extent in ascending id order.
            pool.persist_sleeping(&Neuron::new_atom(0, "zero".into(), NeuronKind::Excitatory, 1))
                .unwrap();
            pool.persist_sleeping(&concept(1, "one", 3)).unwrap();
            pool.persist_sleeping(&Neuron::new_atom(2, "two".into(), NeuronKind::Excitatory, 1))
                .unwrap();
            pool.persist_sleeping(&concept(3, "three", 3)).unwrap();
            // Now re-sleep each one repeatedly: every pass leaves a superseded
            // body behind, which is exactly the garbage compaction exists for.
            for generation in 1..6 {
                pool.persist_sleeping(&Neuron::new_atom(
                    0,
                    "zero".into(),
                    NeuronKind::Excitatory,
                    1,
                ))
                .unwrap();
                pool.persist_sleeping(&concept(1, "one", 3 + generation))
                    .unwrap();
                pool.persist_sleeping(&Neuron::new_atom(
                    2,
                    "two".into(),
                    NeuronKind::Excitatory,
                    1,
                ))
                .unwrap();
                pool.persist_sleeping(&concept(3, "three", 3 + generation))
                    .unwrap();
            }
            file.commit_manifest().unwrap();
            file.flush().unwrap();
        }

        let before = std::fs::metadata(&source).unwrap().len();
        let report = compact(&source, &destination).unwrap();
        verify(&destination, &report).unwrap();

        assert_eq!(
            report.neurons_copied, 4,
            "24 bodies were written but only 4 are live"
        );
        assert!(
            report.destination_bytes < before,
            "compacted {} is not smaller than {before}",
            report.destination_bytes
        );

        // Reopen through the STORE: slot identity, concept flags, label routing
        // and body content must all survive the move.
        let reopened = WbrainFile::open(&destination).unwrap();
        let pool = reopened.pool(7);
        assert_eq!(pool.slot_count(), 4);
        assert_eq!(pool.known_count(), 4);
        assert!(pool.slot_is_concept(1), "concept flag must survive");
        assert!(!pool.slot_is_concept(2), "atom flag must survive");
        assert_eq!(pool.get(0).unwrap().label, "zero");
        assert_eq!(pool.get(2).unwrap().label, "two");
        assert_eq!(pool.label_to_id("two"), Some(2));
        assert_eq!(
            pool.get(3).unwrap().terminals.len(),
            8,
            "the LAST generation must be the one that survives"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Compaction must refuse a destination that already holds records rather
    /// than appending a second brain into it.
    #[test]
    fn refuses_a_populated_destination() {
        let dir = tmpdir("populated");
        let source = dir.join("brain.wbrain");
        let destination = dir.join("out.wbrain");
        {
            let mut container = BrainContainer::open(&source).unwrap();
            let offset = container.append_neuron(1, &concept(0, "c:a", 2)).unwrap();
            container
                .commit_manifest(BrainContainerManifest {
                    generation: 2,
                    tick: 1,
                    brain_metadata: Vec::new(),
                    pools: vec![PoolContainerManifest {
                        pool_id: 1,
                        neuron_count: 1,
                        neuron_capacity: 1,
                        neuron_slot_table: None,
                        label_indexes: Vec::new(),
                        neuron_offsets: vec![Some(offset)],
                        labels: Vec::new(),
                        pool_metadata: Vec::new(),
                    }],
                })
                .unwrap();
        }
        {
            let mut occupied = BrainContainer::open(&destination).unwrap();
            occupied.append_neuron(1, &concept(0, "c:b", 2)).unwrap();
        }
        let error = compact(&source, &destination).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        std::fs::remove_dir_all(dir).ok();
    }
}
