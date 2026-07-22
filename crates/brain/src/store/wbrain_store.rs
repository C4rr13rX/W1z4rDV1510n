//! Working-set cache over a single random-access `.wbrain` container.
//!
//! This is the pool-facing storage layer. A neuron lookup wakes exactly one
//! record on a cache miss; `sleep_neuron` writes that neuron and removes it
//! from RAM. The compact offset/label maps remain resident so startup and
//! initial routing never require whole-brain hydration.

use ahash::AHashMap;
use parking_lot::{Mutex, RwLock};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::neuron::{Neuron, NeuronId, PoolId};
use crate::store::NeuronStore;
use crate::store::container::{
    AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
};

const NEURON_SLOT_TABLE_KIND: u32 = 0x534C_4F54; // "SLOT"
const NEURON_SLOT_BYTES: u64 = 24;
const SLOT_PRESENT: u8 = 0b0000_0001;
const SLOT_CONCEPT: u8 = 0b0000_0010;
const SLOT_WRITE_BATCH: usize = 65_536;

#[derive(Clone, Copy, Default)]
struct NeuronSlotRecord {
    offset: u64,
    born_tick: u64,
    kind: u8,
    flags: u8,
}

impl NeuronSlotRecord {
    fn from_neuron(offset: u64, neuron: &Neuron) -> Self {
        let kind = match neuron.kind {
            crate::neuron::NeuronKind::Excitatory => 0,
            crate::neuron::NeuronKind::Inhibitory => 1,
            crate::neuron::NeuronKind::Modulatory => 2,
        };
        Self {
            offset,
            born_tick: neuron.born_tick,
            kind,
            flags: SLOT_PRESENT | if neuron.is_atom() { 0 } else { SLOT_CONCEPT },
        }
    }

    fn encode(self) -> [u8; NEURON_SLOT_BYTES as usize] {
        let mut raw = [0_u8; NEURON_SLOT_BYTES as usize];
        raw[0..8].copy_from_slice(&self.offset.to_le_bytes());
        raw[8..16].copy_from_slice(&self.born_tick.to_le_bytes());
        raw[16] = self.kind;
        raw[17] = self.flags;
        raw
    }

    fn decode(raw: [u8; NEURON_SLOT_BYTES as usize]) -> Self {
        Self {
            offset: u64::from_le_bytes(raw[0..8].try_into().unwrap()),
            born_tick: u64::from_le_bytes(raw[8..16].try_into().unwrap()),
            kind: raw[16],
            flags: raw[17],
        }
    }

    fn is_present(self) -> bool {
        self.flags & SLOT_PRESENT != 0 && self.offset != 0
    }

    fn is_concept(self) -> bool {
        self.flags & SLOT_CONCEPT != 0
    }
}

/// One open `.wbrain` file shared by every pool in a brain.
pub struct WbrainFile {
    container: Mutex<BrainContainer>,
    pools: RwLock<AHashMap<PoolId, Arc<WbrainNeuronStore>>>,
    generation: AtomicU64,
    tick: AtomicU64,
    brain_metadata: RwLock<Vec<u8>>,
}

impl WbrainFile {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Arc<Self>> {
        let container = BrainContainer::open(path)?;
        let manifest = container.manifest().cloned();
        let generation = manifest.as_ref().map_or(0, |m| m.generation);
        let tick = manifest.as_ref().map_or(0, |m| m.tick);
        let brain_metadata = manifest
            .as_ref()
            .map_or_else(Vec::new, |m| m.brain_metadata.clone());
        let file = Arc::new(Self {
            container: Mutex::new(container),
            pools: RwLock::new(AHashMap::new()),
            generation: AtomicU64::new(generation),
            tick: AtomicU64::new(tick),
            brain_metadata: RwLock::new(brain_metadata),
        });
        if let Some(manifest) = manifest {
            for pool in manifest.pools {
                let store = Arc::new(WbrainNeuronStore::from_manifest(file.clone(), pool));
                file.pools.write().insert(store.pool_id(), store);
            }
        }
        Ok(file)
    }

    pub fn pool(self: &Arc<Self>, pool_id: PoolId) -> Arc<WbrainNeuronStore> {
        if let Some(store) = self.pools.read().get(&pool_id) {
            return store.clone();
        }
        let mut pools = self.pools.write();
        pools
            .entry(pool_id)
            .or_insert_with(|| Arc::new(WbrainNeuronStore::empty(self.clone(), pool_id)))
            .clone()
    }

    pub fn set_tick(&self, tick: u64) {
        self.tick.store(tick, Ordering::Release);
    }

    pub fn tick(&self) -> u64 {
        self.tick.load(Ordering::Acquire)
    }

    pub fn set_brain_metadata(&self, metadata: Vec<u8>) {
        *self.brain_metadata.write() = metadata;
    }

    pub fn brain_metadata(&self) -> Vec<u8> {
        self.brain_metadata.read().clone()
    }

    pub fn pool_ids(&self) -> Vec<PoolId> {
        let mut ids: Vec<_> = self.pools.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Publish all compact routing state. Neuron bodies were already appended
    /// individually by `put`/`sleep_neuron`, so this operation is independent
    /// of total neuron payload size.
    pub fn commit_manifest(&self) -> std::io::Result<u64> {
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        for store in self.pools.read().values() {
            store.flush_paged_slots()?;
        }
        let mut pools: Vec<_> = self
            .pools
            .read()
            .values()
            .map(|store| store.manifest())
            .collect();
        pools.sort_by_key(|pool| pool.pool_id);
        let manifest = BrainContainerManifest {
            generation,
            tick: self.tick.load(Ordering::Acquire),
            brain_metadata: self.brain_metadata.read().clone(),
            pools,
        };
        self.container.lock().commit_manifest(manifest)?;
        Ok(generation)
    }

    pub fn flush(&self) -> std::io::Result<()> {
        self.container.lock().flush()
    }

    pub fn append_auxiliary<F>(
        &self,
        pool_id: PoolId,
        kind: u32,
        write_body: F,
    ) -> std::io::Result<AuxiliaryRecordRef>
    where
        F: FnOnce(&mut std::fs::File) -> std::io::Result<()>,
    {
        self.container
            .lock()
            .append_auxiliary(pool_id, kind, write_body)
    }

    pub fn read_auxiliary(&self, reference: AuxiliaryRecordRef) -> std::io::Result<Vec<u8>> {
        self.container.lock().read_auxiliary(reference)
    }

    pub fn read_auxiliary_exact(
        &self,
        reference: AuxiliaryRecordRef,
        relative_offset: u64,
        body: &mut [u8],
    ) -> std::io::Result<()> {
        self.container
            .lock()
            .read_auxiliary_exact(reference, relative_offset, body)
    }

    pub fn write_auxiliary_exact(
        &self,
        reference: AuxiliaryRecordRef,
        relative_offset: u64,
        body: &[u8],
    ) -> std::io::Result<()> {
        self.container
            .lock()
            .write_auxiliary_exact(reference, relative_offset, body)
    }
}

/// NeuronStore scoped to one pool inside a shared `.wbrain` file.
pub struct WbrainNeuronStore {
    file: Arc<WbrainFile>,
    pool_id: PoolId,
    offsets: RwLock<Vec<Option<u64>>>,
    slot_table: RwLock<Option<AuxiliaryRecordRef>>,
    slot_count: AtomicU64,
    known_count: AtomicU64,
    pending_slots: Mutex<Vec<(NeuronId, NeuronSlotRecord)>>,
    labels: RwLock<AHashMap<String, NeuronId>>,
    pool_metadata: RwLock<Vec<u8>>,
    working_set: RwLock<AHashMap<NeuronId, Neuron>>,
    index_concept_labels: AtomicBool,
    page_ins: AtomicU64,
    page_outs: AtomicU64,
    read_errors: AtomicU64,
    write_errors: AtomicU64,
}

impl WbrainNeuronStore {
    fn empty(file: Arc<WbrainFile>, pool_id: PoolId) -> Self {
        Self {
            file,
            pool_id,
            offsets: RwLock::new(Vec::new()),
            slot_table: RwLock::new(None),
            slot_count: AtomicU64::new(0),
            known_count: AtomicU64::new(0),
            pending_slots: Mutex::new(Vec::with_capacity(SLOT_WRITE_BATCH)),
            labels: RwLock::new(AHashMap::new()),
            pool_metadata: RwLock::new(Vec::new()),
            working_set: RwLock::new(AHashMap::new()),
            index_concept_labels: AtomicBool::new(false),
            page_ins: AtomicU64::new(0),
            page_outs: AtomicU64::new(0),
            read_errors: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
        }
    }

    pub fn read_auxiliary_exact(
        &self,
        reference: AuxiliaryRecordRef,
        relative_offset: u64,
        body: &mut [u8],
    ) -> std::io::Result<()> {
        self.file
            .read_auxiliary_exact(reference, relative_offset, body)
    }

    pub fn write_auxiliary_exact(
        &self,
        reference: AuxiliaryRecordRef,
        relative_offset: u64,
        body: &[u8],
    ) -> std::io::Result<()> {
        self.file
            .write_auxiliary_exact(reference, relative_offset, body)
    }

    fn from_manifest(file: Arc<WbrainFile>, manifest: PoolContainerManifest) -> Self {
        let mut labels = AHashMap::new();
        labels.extend(manifest.labels);
        Self {
            file,
            pool_id: manifest.pool_id,
            offsets: RwLock::new(manifest.neuron_offsets),
            slot_table: RwLock::new(manifest.neuron_slot_table),
            slot_count: AtomicU64::new(manifest.neuron_count as u64),
            known_count: AtomicU64::new(manifest.neuron_count as u64),
            pending_slots: Mutex::new(Vec::with_capacity(SLOT_WRITE_BATCH)),
            labels: RwLock::new(labels),
            pool_metadata: RwLock::new(manifest.pool_metadata),
            working_set: RwLock::new(AHashMap::new()),
            index_concept_labels: AtomicBool::new(false),
            page_ins: AtomicU64::new(0),
            page_outs: AtomicU64::new(0),
            read_errors: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
        }
    }

    pub fn pool_id(&self) -> PoolId {
        self.pool_id
    }

    /// Binding concepts use their stable composite labels as a dedup key.
    /// Other concepts are routed by sequence/member indexes and keep their
    /// labels inside their neuron records, avoiding a multi-gigabyte resident
    /// duplicate label map.
    pub fn set_index_concept_labels(&self, enabled: bool) {
        self.index_concept_labels.store(enabled, Ordering::Release);
    }

    /// Allocate a fixed-width on-disk address/identity directory before a
    /// large streaming migration. No per-neuron offset, kind, concept flag,
    /// or birth tick is retained in RAM.
    pub fn prepare_paged_slots(&self, slot_count: usize) -> std::io::Result<()> {
        if self.slot_table.read().is_some() {
            return Ok(());
        }
        let body_len = (slot_count as u64)
            .checked_mul(NEURON_SLOT_BYTES)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "neuron slot table length overflow",
                )
            })?;
        let reference = self.file.append_auxiliary(
            self.pool_id,
            NEURON_SLOT_TABLE_KIND,
            |writer| {
                if body_len > 0 {
                    writer.seek(SeekFrom::Current((body_len - 1) as i64))?;
                    writer.write_all(&[0])?;
                }
                Ok(())
            },
        )?;
        self.offsets.write().clear();
        *self.slot_table.write() = Some(reference);
        self.slot_count.store(slot_count as u64, Ordering::Release);
        self.known_count.store(0, Ordering::Release);
        Ok(())
    }

    fn read_slot(&self, id: NeuronId) -> Option<NeuronSlotRecord> {
        if let Some((_, record)) = self
            .pending_slots
            .lock()
            .iter()
            .rev()
            .find(|(pending_id, _)| *pending_id == id)
        {
            return Some(*record);
        }
        let reference = *self.slot_table.read();
        if let Some(reference) = reference {
            if id as u64 >= self.slot_count.load(Ordering::Acquire) {
                return None;
            }
            let mut raw = [0_u8; NEURON_SLOT_BYTES as usize];
            self.file
                .read_auxiliary_exact(reference, id as u64 * NEURON_SLOT_BYTES, &mut raw)
                .ok()?;
            return Some(NeuronSlotRecord::decode(raw));
        }
        let offset = self.offsets.read().get(id as usize).copied().flatten()?;
        Some(NeuronSlotRecord {
            offset,
            flags: SLOT_PRESENT,
            ..NeuronSlotRecord::default()
        })
    }

    fn write_slot(&self, id: NeuronId, record: NeuronSlotRecord) -> std::io::Result<()> {
        let reference = *self.slot_table.read();
        if let Some(reference) = reference {
            if id as u64 >= self.slot_count.load(Ordering::Acquire) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("neuron id {} exceeds fixed slot table", id),
                ));
            }
            self.file.write_auxiliary_exact(
                reference,
                id as u64 * NEURON_SLOT_BYTES,
                &record.encode(),
            )?;
            return Ok(());
        }
        let mut offsets = self.offsets.write();
        let index = id as usize;
        if offsets.len() <= index {
            offsets.resize(index + 1, None);
        }
        offsets[index] = record.is_present().then_some(record.offset);
        self.slot_count
            .store(offsets.len() as u64, Ordering::Release);
        Ok(())
    }

    fn queue_slot_write(&self, id: NeuronId, record: NeuronSlotRecord) -> std::io::Result<()> {
        if self.slot_table.read().is_none() {
            return self.write_slot(id, record);
        }
        let should_flush = {
            let mut pending = self.pending_slots.lock();
            pending.push((id, record));
            pending.len() >= SLOT_WRITE_BATCH
        };
        if should_flush {
            self.flush_paged_slots()?;
        }
        Ok(())
    }

    /// Publish queued fixed-width slot records with one contiguous write.
    /// Initial migration IDs are dense and ascending, so this replaces tens
    /// of thousands of table seeks with one bounded batch.
    pub fn flush_paged_slots(&self) -> std::io::Result<()> {
        let reference = *self.slot_table.read();
        let Some(reference) = reference else {
            return Ok(());
        };
        let pending = {
            let mut guard = self.pending_slots.lock();
            if guard.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };
        let first = pending[0].0;
        let contiguous = pending
            .iter()
            .enumerate()
            .all(|(index, (id, _))| *id as u64 == first as u64 + index as u64);
        if contiguous {
            let mut raw = Vec::with_capacity(pending.len() * NEURON_SLOT_BYTES as usize);
            for (_, record) in pending {
                raw.extend_from_slice(&record.encode());
            }
            self.file.write_auxiliary_exact(
                reference,
                first as u64 * NEURON_SLOT_BYTES,
                &raw,
            )?;
        } else {
            for (id, record) in pending {
                self.file.write_auxiliary_exact(
                    reference,
                    id as u64 * NEURON_SLOT_BYTES,
                    &record.encode(),
                )?;
            }
        }
        Ok(())
    }

    pub fn slot_is_concept(&self, id: NeuronId) -> bool {
        self.read_slot(id).is_some_and(NeuronSlotRecord::is_concept)
    }

    pub fn label_to_id(&self, label: &str) -> Option<NeuronId> {
        self.labels.read().get(label).copied()
    }

    pub fn set_pool_metadata(&self, metadata: Vec<u8>) {
        *self.pool_metadata.write() = metadata;
    }

    pub fn pool_metadata(&self) -> Vec<u8> {
        self.pool_metadata.read().clone()
    }

    pub fn labels_snapshot(&self) -> Vec<(String, NeuronId)> {
        self.labels
            .read()
            .iter()
            .map(|(label, id)| (label.clone(), *id))
            .collect()
    }

    pub fn resident_count(&self) -> usize {
        self.working_set.read().len()
    }

    pub fn known_count(&self) -> usize {
        if self.slot_table.read().is_some() {
            self.known_count.load(Ordering::Acquire) as usize
        } else {
            self.offsets
                .read()
                .iter()
                .filter(|offset| offset.is_some())
                .count()
        }
    }

    pub fn slot_count(&self) -> usize {
        if self.slot_table.read().is_some() {
            self.slot_count.load(Ordering::Acquire) as usize
        } else {
            self.offsets.read().len()
        }
    }

    pub fn page_ins(&self) -> u64 {
        self.page_ins.load(Ordering::Relaxed)
    }

    pub fn page_outs(&self) -> u64 {
        self.page_outs.load(Ordering::Relaxed)
    }

    /// Persist the current neuron body and remove only that neuron from RAM.
    pub fn sleep_neuron(&self, id: NeuronId) -> std::io::Result<bool> {
        let neuron = match self.working_set.write().remove(&id) {
            Some(neuron) => neuron,
            None => return Ok(false),
        };
        match self.append_record(&neuron) {
            Ok(()) => {
                self.page_outs.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(error) => {
                // A failed write must not discard the only current copy.
                self.working_set.write().insert(id, neuron);
                Err(error)
            }
        }
    }

    pub fn sleep_all(&self) -> std::io::Result<usize> {
        let ids: Vec<_> = self.working_set.read().keys().copied().collect();
        let mut slept = 0;
        for id in ids {
            if self.sleep_neuron(id)? {
                slept += 1;
            }
        }
        Ok(slept)
    }

    /// Persist a neuron without admitting a duplicate copy to this store's
    /// working-set cache. Pool owns the live copy until it replaces that copy
    /// with its compact sleeping slot.
    pub fn persist_sleeping(&self, neuron: &Neuron) -> std::io::Result<()> {
        self.append_record(neuron)?;
        self.working_set.write().remove(&neuron.id);
        self.page_outs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Drop the store's cache copy after Pool has installed the paged-in
    /// neuron in its own live slot. There must be exactly one resident body.
    pub fn release_cached(&self, id: NeuronId) {
        self.working_set.write().remove(&id);
    }

    fn append_record(&self, neuron: &Neuron) -> std::io::Result<()> {
        let known_before = self.known_count.load(Ordering::Acquire);
        let was_present = (neuron.id as u64) < known_before;
        let offset = self
            .file
            .container
            .lock()
            .append_neuron(self.pool_id, neuron)?;
        let record = NeuronSlotRecord::from_neuron(offset, neuron);
        if !was_present && neuron.id as u64 == known_before {
            self.queue_slot_write(neuron.id, record)?;
        } else {
            self.flush_paged_slots()?;
            self.write_slot(neuron.id, record)?;
        }
        if !was_present {
            self.known_count.fetch_add(1, Ordering::AcqRel);
        }
        if neuron.is_atom() || self.index_concept_labels.load(Ordering::Acquire) {
            self.labels.write().insert(neuron.label.clone(), neuron.id);
        }
        Ok(())
    }

    fn manifest(&self) -> PoolContainerManifest {
        let slot_table = *self.slot_table.read();
        let offsets = if slot_table.is_some() {
            Vec::new()
        } else {
            self.offsets.read().clone()
        };
        let mut labels: Vec<_> = self
            .labels
            .read()
            .iter()
            .map(|(label, id)| (label.clone(), *id))
            .collect();
        labels.sort_by(|a, b| a.0.cmp(&b.0));
        PoolContainerManifest {
            pool_id: self.pool_id,
            neuron_count: self.slot_count() as u32,
            neuron_slot_table: slot_table,
            neuron_offsets: offsets,
            labels,
            pool_metadata: self.pool_metadata.read().clone(),
        }
    }
}

impl NeuronStore for WbrainNeuronStore {
    fn get(&self, id: NeuronId) -> Option<Neuron> {
        if let Some(neuron) = self.working_set.read().get(&id) {
            return Some(neuron.clone());
        }
        let slot = self.read_slot(id)?;
        if !slot.is_present() {
            return None;
        }
        let offset = slot.offset;
        let result = self.file.container.lock().read_neuron_at(offset);
        match result {
            Ok((pool_id, neuron)) if pool_id == self.pool_id && neuron.id == id => {
                self.working_set.write().insert(id, neuron.clone());
                self.page_ins.fetch_add(1, Ordering::Relaxed);
                Some(neuron)
            }
            Ok(_) | Err(_) => {
                self.read_errors.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    fn put(&self, id: NeuronId, mut neuron: Neuron) {
        neuron.id = id;
        self.working_set.write().insert(id, neuron.clone());
        if self.append_record(&neuron).is_err() {
            self.write_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn delete(&self, id: NeuronId) {
        self.working_set.write().remove(&id);
        let was_present = self
            .read_slot(id)
            .is_some_and(NeuronSlotRecord::is_present);
        if self
            .write_slot(id, NeuronSlotRecord::default())
            .is_err()
        {
            self.write_errors.fetch_add(1, Ordering::Relaxed);
        } else if was_present {
            self.known_count.fetch_sub(1, Ordering::AcqRel);
        }
        self.labels.write().retain(|_, known_id| *known_id != id);
    }

    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a> {
        Box::new((0..self.slot_count() as NeuronId).filter(|id| {
            self.read_slot(*id)
                .is_some_and(NeuronSlotRecord::is_present)
        }))
    }

    fn len(&self) -> usize {
        self.known_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{Brain, BrainConfig};
    use crate::neuron::{NeuronKind, NeuronRef, Terminal};
    use crate::pool::{BytePassthroughEncoding, Pool, PoolConfig};

    fn tmpfile(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "w1z4rd_store_{name}_{}_{}.wbrain",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn reopen_starts_empty_and_pages_only_requested_neuron() {
        let path = tmpfile("scope");
        {
            let file = WbrainFile::open(&path).unwrap();
            let pool = file.pool(7);
            for id in 0..3 {
                pool.put(
                    id,
                    Neuron::new_atom(id, format!("atom:{id}"), NeuronKind::Excitatory, 1),
                );
            }
            assert_eq!(pool.sleep_all().unwrap(), 3);
            assert_eq!(pool.resident_count(), 0);
            file.commit_manifest().unwrap();
            file.flush().unwrap();
        }

        let reopened = WbrainFile::open(&path).unwrap();
        let pool = reopened.pool(7);
        assert_eq!(pool.known_count(), 3);
        assert_eq!(pool.resident_count(), 0);
        assert_eq!(pool.get(1).unwrap().label, "atom:1");
        assert_eq!(pool.resident_count(), 1);
        assert_eq!(pool.page_ins(), 1);
        assert_eq!(pool.label_to_id("atom:2"), Some(2));
        assert_eq!(pool.resident_count(), 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn paged_slot_directory_reopens_without_resident_offset_vectors() {
        let path = tmpfile("paged-slots");
        {
            let file = WbrainFile::open(&path).unwrap();
            let pool = file.pool(7);
            pool.prepare_paged_slots(3).unwrap();
            pool.persist_sleeping(&Neuron::new_atom(
                0,
                "zero".into(),
                NeuronKind::Excitatory,
                1,
            ))
            .unwrap();
            pool.persist_sleeping(&Neuron::new_concept(
                1,
                "one".into(),
                NeuronKind::Excitatory,
                vec![NeuronRef::new(7, 0)],
                2,
            ))
            .unwrap();
            pool.persist_sleeping(&Neuron::new_atom(
                2,
                "two".into(),
                NeuronKind::Excitatory,
                3,
            ))
            .unwrap();
            file.commit_manifest().unwrap();
            file.flush().unwrap();
            let manifest = file.container.lock().manifest().unwrap().clone();
            assert!(manifest.pools[0].neuron_offsets.is_empty());
            assert!(manifest.pools[0].neuron_slot_table.is_some());
        }

        let reopened = WbrainFile::open(&path).unwrap();
        let pool = reopened.pool(7);
        assert_eq!(pool.slot_count(), 3);
        assert_eq!(pool.known_count(), 3);
        assert!(pool.slot_is_concept(1));
        assert!(!pool.slot_is_concept(2));
        assert_eq!(pool.resident_count(), 0);
        assert_eq!(pool.get(2).unwrap().label, "two");
        assert_eq!(pool.resident_count(), 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn sleep_serializes_each_resident_neuron_independently() {
        let path = tmpfile("sleep");
        let file = WbrainFile::open(&path).unwrap();
        let pool = file.pool(2);
        pool.put(
            0,
            Neuron::new_atom(0, "zero".into(), NeuronKind::Excitatory, 1),
        );
        pool.put(
            1,
            Neuron::new_atom(1, "one".into(), NeuronKind::Excitatory, 1),
        );
        assert_eq!(pool.resident_count(), 2);
        assert!(pool.sleep_neuron(0).unwrap());
        assert_eq!(pool.resident_count(), 1);
        assert_eq!(pool.get(0).unwrap().label, "zero");
        assert_eq!(pool.resident_count(), 2);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn pool_idle_serializes_atoms_and_pages_only_requested_id() {
        let path = tmpfile("pool-idle");
        let file = WbrainFile::open(&path).unwrap();
        let store = file.pool(3);
        let config = PoolConfig::defaults("test", 3);
        let mut pool = Pool::new(
            config,
            Box::new(BytePassthroughEncoding {
                prefix: "byte".into(),
            }),
        );
        pool.set_wbrain_store(store.clone());
        let ids = pool.ensure_frame_atoms_for_pretrain(b"abc", 1);
        assert_eq!(ids.len(), 3);

        assert_eq!(pool.serialize_all_neurons_for_idle().unwrap(), 3);
        assert_eq!(pool.live_count(), 0);
        assert_eq!(store.resident_count(), 0);

        pool.ensure_loaded(ids[1]).unwrap();
        assert_eq!(pool.live_count(), 1);
        assert_eq!(pool.get(ids[1]).unwrap().label, "byte:Yg");
        assert_eq!(store.resident_count(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn sleeping_concept_releases_members_without_losing_identity() {
        let path = tmpfile("concept-member-sleep");
        let file = WbrainFile::open(&path).unwrap();
        let store = file.pool(3);
        let mut pool = Pool::new(
            PoolConfig::defaults("test", 3),
            Box::new(BytePassthroughEncoding {
                prefix: "byte".into(),
            }),
        );
        pool.set_wbrain_store(store);
        pool.ensure_frame_concept_for_pretrain(b"sleeping concept", 1);
        let concept = pool.ensure_frame_concept_for_pretrain(b"sleeping concept", 2)[0];
        assert!(!pool.get(concept).unwrap().is_atom());
        assert!(!pool.get(concept).unwrap().members.is_empty());

        pool.serialize_all_neurons_for_idle().unwrap();
        assert!(
            pool.get(concept).is_none(),
            "a sleeping neuron must not retain an in-memory sentinel body"
        );

        pool.ensure_loaded(concept).unwrap();
        assert!(!pool.get(concept).unwrap().is_atom());
        assert!(!pool.get(concept).unwrap().members.is_empty());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn pool_reopens_from_compact_metadata_with_zero_live_neurons() {
        let path = tmpfile("pool-reopen");
        let ids = {
            let file = WbrainFile::open(&path).unwrap();
            let store = file.pool(3);
            let mut pool = Pool::new(
                PoolConfig::defaults("test", 3),
                Box::new(BytePassthroughEncoding {
                    prefix: "byte".into(),
                }),
            );
            pool.set_wbrain_store(store);
            let ids = pool.ensure_frame_atoms_for_pretrain(b"abc", 1);
            pool.stage_wbrain_metadata().unwrap();
            pool.serialize_all_neurons_for_idle().unwrap();
            file.commit_manifest().unwrap();
            file.flush().unwrap();
            ids
        };

        let reopened = WbrainFile::open(&path).unwrap();
        let store = reopened.pool(3);
        let mut pool = Pool::from_wbrain_store(
            Box::new(BytePassthroughEncoding {
                prefix: "byte".into(),
            }),
            store.clone(),
        )
        .unwrap();
        assert_eq!(pool.neuron_count(), 3);
        assert_eq!(pool.live_count(), 0);
        assert_eq!(store.resident_count(), 0);
        pool.ensure_loaded(ids[2]).unwrap();
        assert_eq!(pool.live_count(), 1);
        assert_eq!(pool.get(ids[2]).unwrap().label, "byte:Yw");
        assert_eq!(store.resident_count(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn whole_brain_reopens_without_deserializing_neuron_bodies() {
        let path = tmpfile("brain-reopen");
        let binding_pool_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            brain.create_pool(
                PoolConfig::defaults("bytes", 3),
                Box::new(BytePassthroughEncoding {
                    prefix: "byte".into(),
                }),
            );
            brain.observe(3, b"abc");
            brain.attach_wbrain(&path).unwrap();
            assert_eq!(brain.serialize_all_neurons_for_idle().unwrap(), 3);
        }

        let mut encodings: std::collections::HashMap<PoolId, Box<dyn crate::pool::AtomEncoding>> =
            std::collections::HashMap::new();
        encodings.insert(
            binding_pool_id,
            Box::new(BytePassthroughEncoding {
                prefix: "bind".into(),
            }),
        );
        encodings.insert(
            3,
            Box::new(BytePassthroughEncoding {
                prefix: "byte".into(),
            }),
        );
        let (restored, missing) = Brain::restore_wbrain(&path, encodings).unwrap();
        assert!(missing.is_empty());
        let stats = restored.stats();
        assert_eq!(stats.total_neurons, 3);
        assert_eq!(stats.evicted_neurons, 3);
        assert_eq!(stats.resident_terminals, 0);
        let pool = restored.fabric().pool(3).unwrap();
        let mut pool = pool.write();
        let id = pool.label_to_id("byte:Yg").unwrap();
        pool.ensure_loaded(id).unwrap();
        assert_eq!(pool.live_count(), 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn legacy_migration_preserves_source_and_produces_lazy_container() {
        let legacy = tmpfile("legacy-source").with_extension("bin");
        let destination = tmpfile("legacy-migrated");
        let binding_pool_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            brain.create_pool(
                PoolConfig::defaults("bytes", 3),
                Box::new(BytePassthroughEncoding {
                    prefix: "byte".into(),
                }),
            );
            brain.observe(3, b"abc");
            brain.checkpoint(&legacy).unwrap();
        }
        let source_bytes = std::fs::metadata(&legacy).unwrap().len();

        let encodings = || {
            let mut map: std::collections::HashMap<PoolId, Box<dyn crate::pool::AtomEncoding>> =
                std::collections::HashMap::new();
            map.insert(
                binding_pool_id,
                Box::new(BytePassthroughEncoding {
                    prefix: "bind".into(),
                }),
            );
            map.insert(
                3,
                Box::new(BytePassthroughEncoding {
                    prefix: "byte".into(),
                }),
            );
            map
        };
        let serialized = Brain::migrate_legacy_checkpoint_streaming(&legacy, &destination).unwrap();
        assert_eq!(serialized, 3);
        assert_eq!(std::fs::metadata(&legacy).unwrap().len(), source_bytes);

        let (restored, missing) = Brain::restore_wbrain(&destination, encodings()).unwrap();
        assert!(missing.is_empty());
        assert_eq!(restored.stats().evicted_neurons, 3);
        assert_eq!(restored.stats().resident_terminals, 0);
        std::fs::remove_file(legacy).ok();
        std::fs::remove_file(destination).ok();
    }

    #[test]
    fn streaming_migration_overlays_authoritative_cold_neuron_records() {
        let root = tmpfile("legacy-cold-root").with_extension("");
        std::fs::create_dir_all(&root).unwrap();
        let legacy = root.join("brain.bin");
        let destination = root.join("brain.wbrain");
        let binding_pool_id;
        let concept_id;
        let concept_label;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            brain.create_pool(
                PoolConfig::defaults("bytes", 3),
                Box::new(BytePassthroughEncoding {
                    prefix: "byte".into(),
                }),
            );
            let pool = brain.fabric().pool(3).unwrap();
            {
                let mut pool = pool.write();
                pool.ensure_frame_concept_for_pretrain(b"cold concept", 1);
                let promoted = pool.ensure_frame_concept_for_pretrain(b"cold concept", 2);
                concept_id = promoted[0];
                assert!(!pool.get(concept_id).unwrap().is_atom());
                concept_label = pool.get(concept_id).unwrap().label.clone();
                pool.get_mut(concept_id)
                    .unwrap()
                    .terminals
                    .push(Terminal::new(NeuronRef::new(3, 0), 0.75, 2));
            }
            assert_eq!(brain.attach_cold_tiers(&root), 2);
            assert!(pool.write().evict_neuron(concept_id).unwrap());
            assert!(pool.read().get(concept_id).unwrap().members.len() > 1);
            assert!(pool.read().get(concept_id).unwrap().terminals.is_empty());
            brain.checkpoint(&legacy).unwrap();
        }

        Brain::migrate_legacy_checkpoint_streaming(&legacy, &destination).unwrap();
        let file = WbrainFile::open(&destination).unwrap();
        assert_eq!(file.pool(3).label_to_id(&concept_label), None);
        drop(file);
        let mut encodings: std::collections::HashMap<PoolId, Box<dyn crate::pool::AtomEncoding>> =
            std::collections::HashMap::new();
        encodings.insert(
            binding_pool_id,
            Box::new(BytePassthroughEncoding {
                prefix: "bind".into(),
            }),
        );
        encodings.insert(
            3,
            Box::new(BytePassthroughEncoding {
                prefix: "byte".into(),
            }),
        );
        let (restored, missing) = Brain::restore_wbrain(&destination, encodings).unwrap();
        assert!(missing.is_empty());
        let pool = restored.fabric().pool(3).unwrap();
        assert_eq!(pool.read().live_count(), 0);
        pool.write().ensure_loaded(concept_id).unwrap();
        let pool = pool.read();
        let concept = pool.get(concept_id).unwrap();
        assert!(!concept.is_atom());
        assert_eq!(concept.label, concept_label);
        assert!(concept.members.len() > 1);
        assert_eq!(concept.terminals.len(), 1);
        assert_eq!(concept.terminals[0].target, NeuronRef::new(3, 0));

        drop(pool);
        drop(restored);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn streaming_migration_rebuilds_binding_indexes_with_bounded_working_set() {
        let legacy = tmpfile("legacy-binding-source").with_extension("bin");
        let destination = tmpfile("legacy-binding-migrated");
        let binding_pool_id;
        let trained_binding_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            for (id, name, prefix) in [(3, "prompt", "p"), (4, "answer", "a")] {
                brain.create_pool(
                    PoolConfig::defaults(name, id),
                    Box::new(BytePassthroughEncoding {
                        prefix: prefix.into(),
                    }),
                );
            }
            trained_binding_id = brain
                .pretrain_binding_episode(&[(3, b"hello".to_vec()), (4, b"world".to_vec())])
                .unwrap();
            brain.checkpoint(&legacy).unwrap();
        }
        let encodings = || {
            let mut map: std::collections::HashMap<PoolId, Box<dyn crate::pool::AtomEncoding>> =
                std::collections::HashMap::new();
            for (id, prefix) in [(binding_pool_id, "bind"), (3, "p"), (4, "a")] {
                map.insert(
                    id,
                    Box::new(BytePassthroughEncoding {
                        prefix: prefix.into(),
                    }),
                );
            }
            map
        };

        Brain::migrate_legacy_checkpoint_streaming(&legacy, &destination).unwrap();
        let (mut migrated, missing) = Brain::restore_wbrain(&destination, encodings()).unwrap();
        assert!(missing.is_empty());
        assert_eq!(migrated.rebuild_binding_indexes_bounded().unwrap(), 1);
        assert_eq!(
            migrated
                .pretrain_binding_episode(&[(3, b"hello".to_vec()), (4, b"world".to_vec())])
                .unwrap(),
            trained_binding_id,
            "binding labels must remain indexed for idempotent live training",
        );
        migrated.serialize_all_neurons_for_idle().unwrap();
        assert_eq!(
            migrated.stats().evicted_neurons,
            migrated.stats().total_neurons
        );
        drop(migrated);

        let (mut restored, missing) = Brain::restore_wbrain(&destination, encodings()).unwrap();
        assert!(missing.is_empty());
        restored.activate_for_prediction(3, b"hello");
        assert_eq!(
            restored
                .decode_best_trained_binding_multi(&[3], 4)
                .as_deref(),
            Some(b"world".as_slice())
        );
        restored.clear_prediction_activation();
        restored.serialize_all_neurons_for_idle().unwrap();
        assert_eq!(
            restored.stats().evicted_neurons,
            restored.stats().total_neurons
        );

        std::fs::remove_file(legacy).ok();
        std::fs::remove_file(destination).ok();
    }

    #[test]
    fn lazy_restore_pages_binding_scope_and_preserves_recall() {
        let path = tmpfile("lazy-recall");
        let binding_pool_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            for (id, name, prefix) in [(3, "prompt", "p"), (4, "answer", "a")] {
                brain.create_pool(
                    PoolConfig::defaults(name, id),
                    Box::new(BytePassthroughEncoding {
                        prefix: prefix.into(),
                    }),
                );
            }
            brain.create_pool(
                PoolConfig::defaults("unrelated", 5),
                Box::new(BytePassthroughEncoding { prefix: "n".into() }),
            );
            brain.observe(5, b"unrelated-neurons");
            // The first episode creates the binding; two subsequent
            // reinforcements satisfy integrate()'s trained-pathway gate.
            for _ in 0..3 {
                assert!(
                    brain
                        .pretrain_binding_episode(
                            &[(3, b"hello".to_vec()), (4, b"world".to_vec()),]
                        )
                        .is_some()
                );
            }
            brain.attach_wbrain(&path).unwrap();
            brain.serialize_all_neurons_for_idle().unwrap();
        }
        let mut encodings: std::collections::HashMap<PoolId, Box<dyn crate::pool::AtomEncoding>> =
            std::collections::HashMap::new();
        for (id, prefix) in [(binding_pool_id, "bind"), (3, "p"), (4, "a"), (5, "n")] {
            encodings.insert(
                id,
                Box::new(BytePassthroughEncoding {
                    prefix: prefix.into(),
                }),
            );
        }
        let (mut restored, missing) = Brain::restore_wbrain(&path, encodings).unwrap();
        assert!(missing.is_empty());
        let asleep = restored.stats().evicted_neurons;

        restored.activate_for_prediction(3, b"hello");
        let binding_match = restored.best_binding_match_v2(3);
        restored.clear_prediction_activation();
        assert!(
            binding_match.score() > 0.0,
            "confidence routing must page the trained binding instead of scanning sleeping slots"
        );
        restored.serialize_all_neurons_for_idle().unwrap();

        restored.activate_for_prediction(3, b"hello");
        let grounded = restored.integrate(3, 4);
        restored.clear_prediction_activation();
        assert_eq!(grounded.answer.as_deref(), Some(b"world".as_slice()));
        restored.serialize_all_neurons_for_idle().unwrap();

        restored.activate_for_prediction(3, b"hello");
        let single_answer = restored.decode_best_trained_binding(3, 4);
        assert_eq!(single_answer.as_deref(), Some(b"world".as_slice()));
        restored.clear_prediction_activation();
        restored.serialize_all_neurons_for_idle().unwrap();

        restored.activate_for_prediction(3, b"hello");
        let answer = restored.decode_best_trained_binding_multi(&[3], 4);
        restored.clear_prediction_activation();
        assert_eq!(answer.as_deref(), Some(b"world".as_slice()));
        let awake = restored.stats().total_neurons - restored.stats().evicted_neurons;
        assert!(awake > 0);
        assert!(awake < asleep, "inference must not hydrate the whole brain");
        restored.serialize_all_neurons_for_idle().unwrap();
        assert_eq!(
            restored.stats().evicted_neurons,
            restored.stats().total_neurons
        );
        std::fs::remove_file(path).ok();
    }
}
