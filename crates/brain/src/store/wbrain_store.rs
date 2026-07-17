//! Working-set cache over a single random-access `.wbrain` container.
//!
//! This is the pool-facing storage layer. A neuron lookup wakes exactly one
//! record on a cache miss; `sleep_neuron` writes that neuron and removes it
//! from RAM. The compact offset/label maps remain resident so startup and
//! initial routing never require whole-brain hydration.

use ahash::AHashMap;
use parking_lot::{Mutex, RwLock};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::neuron::{Neuron, NeuronId, PoolId};
use crate::store::NeuronStore;
use crate::store::container::{
    AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
};

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
}

/// NeuronStore scoped to one pool inside a shared `.wbrain` file.
pub struct WbrainNeuronStore {
    file: Arc<WbrainFile>,
    pool_id: PoolId,
    offsets: RwLock<Vec<Option<u64>>>,
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

    fn from_manifest(file: Arc<WbrainFile>, manifest: PoolContainerManifest) -> Self {
        let mut labels = AHashMap::new();
        labels.extend(manifest.labels);
        Self {
            file,
            pool_id: manifest.pool_id,
            offsets: RwLock::new(manifest.neuron_offsets),
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
        self.offsets
            .read()
            .iter()
            .filter(|offset| offset.is_some())
            .count()
    }

    pub fn slot_count(&self) -> usize {
        self.offsets.read().len()
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
        let offset = self
            .file
            .container
            .lock()
            .append_neuron(self.pool_id, neuron)?;
        let mut offsets = self.offsets.write();
        let index = neuron.id as usize;
        if offsets.len() <= index {
            offsets.resize(index + 1, None);
        }
        offsets[index] = Some(offset);
        if neuron.is_atom() || self.index_concept_labels.load(Ordering::Acquire) {
            self.labels.write().insert(neuron.label.clone(), neuron.id);
        }
        Ok(())
    }

    fn manifest(&self) -> PoolContainerManifest {
        let offsets = self.offsets.read().clone();
        let mut labels: Vec<_> = self
            .labels
            .read()
            .iter()
            .map(|(label, id)| (label.clone(), *id))
            .collect();
        labels.sort_by(|a, b| a.0.cmp(&b.0));
        PoolContainerManifest {
            pool_id: self.pool_id,
            neuron_count: offsets.len() as u32,
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
        let offset = self.offsets.read().get(id as usize).copied().flatten()?;
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
        if let Some(offset) = self.offsets.write().get_mut(id as usize) {
            *offset = None;
        }
        self.labels.write().retain(|_, known_id| *known_id != id);
    }

    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a> {
        let ids: Vec<_> = self
            .offsets
            .read()
            .iter()
            .enumerate()
            .filter_map(|(id, offset)| offset.map(|_| id as NeuronId))
            .collect();
        Box::new(ids.into_iter())
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
            assert!(
                brain
                    .pretrain_binding_episode(&[(3, b"hello".to_vec()), (4, b"world".to_vec())])
                    .is_some()
            );
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
