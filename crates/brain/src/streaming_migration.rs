//! Bounded-memory conversion of a legacy bincode checkpoint to `.wbrain`.
//!
//! The legacy format is positional. We read its vectors and maps field by
//! field, deserialize one neuron at a time, and immediately append that
//! neuron to the destination container.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use sysinfo::System;

use crate::action::{ActionEvent, ActionId};
use crate::neuron::{Neuron, NeuronId, PoolId};
use crate::persistence::{AnnealerSnapshot, EemSnapshot, SerializableFingerprint};
use crate::pool::{Pool, PoolConfig, StreamedPoolMetadata};
use crate::store::{AuxiliaryRecordRef, ColdTier, NeuronStore, WbrainFile};

use super::wbrain_metadata::{PersistedMomentFingerprint, WbrainBrainMetadata};
use super::{
    BrainConfig, default_min_atom_score, default_pressure_adjust_enabled,
    default_pressure_band_high, default_pressure_band_low, default_pressure_observation_grace,
    default_pressure_threshold_max,
};

const MIGRATION_MIN_AVAILABLE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MEMORY_CHECK_INTERVAL: u64 = 16_384;
const DISCARD_BUFFER_BYTES: usize = 64 * 1024;
const DISCARD_MEMORY_CHECK_BYTES: u64 = 64 * 1024 * 1024;
const MIGRATION_PROGRESS_MAGIC: &[u8; 8] = b"W1ZMIGR1";
const LEGACY_SEQUENCE_LEDGER_KIND: u32 = 1;
const LEGACY_CONCEPT_SEQUENCE_INDEX_KIND: u32 = 2;
const CONCEPT_INDEX_MAGIC: &[u8; 8] = b"W1ZCID01";
const MAX_DISK_INDEX_BUCKETS: u64 = 4 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
struct MigrationProgress {
    legacy_bytes: u64,
    legacy_tick: u64,
    pool_count: u64,
    completed_pools: u64,
    source_offset: u64,
}

fn read_value<R: Read, T: DeserializeOwned>(reader: &mut R) -> io::Result<T> {
    bincode::deserialize_from(reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[cfg(windows)]
fn trim_unused_working_set() {
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn EmptyWorkingSet(process: *mut std::ffi::c_void) -> i32;
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
    }

    // This only evicts currently unused pages from this process's working set;
    // live allocations remain addressable and are faulted back if needed.
    unsafe {
        EmptyWorkingSet(GetCurrentProcess());
    }
}

#[cfg(not(windows))]
fn trim_unused_working_set() {}

fn require_memory_floor(system: &mut System, context: &str) -> io::Result<()> {
    if cfg!(test) {
        return Ok(());
    }
    trim_unused_working_set();
    system.refresh_memory();
    let available = system.available_memory();
    if available < MIGRATION_MIN_AVAILABLE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!(
                "streaming migration stopped {context}: {:.2} GiB available is below the 4 GiB runtime floor",
                available as f64 / 1024_f64.powi(3),
            ),
        ));
    }
    Ok(())
}

fn skip_bincode_string<R: Read>(reader: &mut R, system: &mut System) -> io::Result<()> {
    let byte_count: u64 = read_value(reader)?;
    let mut remaining = byte_count;
    let mut since_memory_check = 0_u64;
    let mut buffer = [0_u8; DISCARD_BUFFER_BYTES];
    while remaining > 0 {
        let requested = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        reader.read_exact(&mut buffer[..requested]).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "legacy string declared {byte_count} bytes but ended with {remaining} unread"
                    ),
                )
            } else {
                error
            }
        })?;
        remaining -= requested as u64;
        since_memory_check += requested as u64;
        if since_memory_check >= DISCARD_MEMORY_CHECK_BYTES {
            require_memory_floor(system, "while discarding legacy label bytes")?;
            since_memory_check = 0;
        }
    }
    Ok(())
}

fn read_recent_atoms<R: Read>(
    reader: &mut R,
    configured_window: usize,
) -> io::Result<VecDeque<NeuronId>> {
    let count: u64 = read_value(reader)?;
    let hard_limit = configured_window.max(1_048_576) as u64;
    if count > hard_limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("legacy recent-atom count {count} exceeds safe limit {hard_limit}"),
        ));
    }
    let mut recent = VecDeque::with_capacity(count as usize);
    for _ in 0..count {
        recent.push_back(read_value(reader)?);
    }
    Ok(recent)
}

fn stream_sequence_ledger<R: Read>(
    reader: &mut R,
    system: &mut System,
    file: &std::sync::Arc<WbrainFile>,
    pool_id: PoolId,
) -> io::Result<AuxiliaryRecordRef> {
    let sequence_count: u64 = read_value(reader)?;
    const LEDGER_MAGIC: &[u8; 8] = b"W1ZSEQ01";
    const MAX_BUCKETS: u64 = 4 * 1024 * 1024;
    let desired_buckets = sequence_count
        .saturating_mul(2)
        .max(1)
        .next_power_of_two();
    let bucket_count = desired_buckets.min(MAX_BUCKETS);
    file.append_auxiliary(
        pool_id,
        LEGACY_SEQUENCE_LEDGER_KIND,
        |writer| -> io::Result<()> {
            let body_start = writer.stream_position()?;
            writer.write_all(LEDGER_MAGIC)?;
            writer.write_all(&sequence_count.to_le_bytes())?;
            writer.write_all(&bucket_count.to_le_bytes())?;
            let zeroes = [0_u8; DISCARD_BUFFER_BYTES];
            let mut bucket_bytes = bucket_count * 8;
            while bucket_bytes > 0 {
                let n = usize::try_from(bucket_bytes.min(zeroes.len() as u64)).unwrap();
                writer.write_all(&zeroes[..n])?;
                bucket_bytes -= n as u64;
            }
            let mut buffer = [0_u8; DISCARD_BUFFER_BYTES];
            let mut since_memory_check = 0_u64;
            for _ in 0..sequence_count {
                let member_count: u64 = read_value(reader)?;
                let member_count_u32 = u32::try_from(member_count).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "legacy sequence is too long")
                })?;
                let member_bytes = member_count.checked_mul(4).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "legacy sequence member byte count overflow",
                    )
                })?;
                let record_relative = writer.stream_position()? - body_start;
                writer.write_all(&0_u64.to_le_bytes())?;
                writer.write_all(&0_u64.to_le_bytes())?;
                writer.write_all(&member_count_u32.to_le_bytes())?;
                let mut hasher = blake3::Hasher::new();
                let mut remaining = member_bytes;
                while remaining > 0 {
                    let requested = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
                    reader.read_exact(&mut buffer[..requested])?;
                    writer.write_all(&buffer[..requested])?;
                    hasher.update(&buffer[..requested]);
                    remaining -= requested as u64;
                    since_memory_check += requested as u64;
                    if since_memory_check >= DISCARD_MEMORY_CHECK_BYTES {
                        require_memory_floor(system, "while streaming legacy sequence state")?;
                        since_memory_check = 0;
                    }
                }
                let recurrence: u32 = read_value(reader)?;
                writer.write_all(&recurrence.to_le_bytes())?;
                let end = writer.stream_position()?;
                let hash_raw = hasher.finalize();
                let hash = u64::from_le_bytes(hash_raw.as_bytes()[..8].try_into().unwrap());
                let bucket = hash & (bucket_count - 1);
                let bucket_relative = 24 + bucket * 8;
                writer.seek(io::SeekFrom::Start(body_start + bucket_relative))?;
                let mut old_head = [0_u8; 8];
                writer.read_exact(&mut old_head)?;
                writer.seek(io::SeekFrom::Start(body_start + record_relative))?;
                writer.write_all(&old_head)?;
                writer.write_all(&hash.to_le_bytes())?;
                writer.seek(io::SeekFrom::Start(body_start + bucket_relative))?;
                writer.write_all(&(record_relative + 1).to_le_bytes())?;
                writer.seek(io::SeekFrom::Start(end))?;
            }
            Ok(())
        },
    )
}

fn persisted(fingerprint: SerializableFingerprint) -> PersistedMomentFingerprint {
    PersistedMomentFingerprint {
        pairs: fingerprint.pairs,
        ordered_per_pool: Vec::new(),
        members_per_pool: Vec::new(),
    }
}

fn encode_progress(progress: &MigrationProgress) -> io::Result<Vec<u8>> {
    let encoded = bincode::serialize(progress)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut bytes = Vec::with_capacity(MIGRATION_PROGRESS_MAGIC.len() + encoded.len());
    bytes.extend_from_slice(MIGRATION_PROGRESS_MAGIC);
    bytes.extend_from_slice(&encoded);
    Ok(bytes)
}

fn decode_progress(bytes: &[u8]) -> io::Result<Option<MigrationProgress>> {
    let Some(encoded) = bytes.strip_prefix(MIGRATION_PROGRESS_MAGIC) else {
        return Ok(None);
    };
    bincode::deserialize(encoded)
        .map(Some)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

struct DiskConceptIndex {
    path: PathBuf,
    file: File,
    bucket_count: u64,
    entries: u64,
}

impl DiskConceptIndex {
    fn create(directory: &Path, pool_id: PoolId, expected_entries: u64) -> io::Result<Self> {
        let path = directory.join(format!(".pool-{pool_id}.concept-index.tmp"));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let desired = expected_entries
            .saturating_mul(2)
            .max(1)
            .next_power_of_two();
        let bucket_count = desired.min(MAX_DISK_INDEX_BUCKETS);
        file.write_all(CONCEPT_INDEX_MAGIC)?;
        file.write_all(&0_u64.to_le_bytes())?;
        file.write_all(&bucket_count.to_le_bytes())?;
        let zeroes = [0_u8; DISCARD_BUFFER_BYTES];
        let mut remaining = bucket_count * 8;
        while remaining > 0 {
            let n = usize::try_from(remaining.min(zeroes.len() as u64)).unwrap();
            file.write_all(&zeroes[..n])?;
            remaining -= n as u64;
        }
        Ok(Self {
            path,
            file,
            bucket_count,
            entries: 0,
        })
    }

    fn insert(&mut self, sequence: &[NeuronId], concept_id: NeuronId) -> io::Result<()> {
        let mut hasher = blake3::Hasher::new();
        for id in sequence {
            hasher.update(&id.to_le_bytes());
        }
        let digest = hasher.finalize();
        let hash = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let bucket = hash & (self.bucket_count - 1);
        let bucket_offset = 24 + bucket * 8;
        self.file.seek(SeekFrom::Start(bucket_offset))?;
        let mut old_head = [0_u8; 8];
        self.file.read_exact(&mut old_head)?;
        let record = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&old_head)?;
        self.file.write_all(&hash.to_le_bytes())?;
        self.file.write_all(
            &u32::try_from(sequence.len())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "concept too large"))?
                .to_le_bytes(),
        )?;
        for id in sequence {
            self.file.write_all(&id.to_le_bytes())?;
        }
        self.file.write_all(&concept_id.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(bucket_offset))?;
        self.file.write_all(&(record + 1).to_le_bytes())?;
        self.file.seek(SeekFrom::End(0))?;
        self.entries += 1;
        Ok(())
    }

    fn finish(
        mut self,
        destination: &std::sync::Arc<WbrainFile>,
        pool_id: PoolId,
    ) -> io::Result<AuxiliaryRecordRef> {
        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_all(&self.entries.to_le_bytes())?;
        self.file.flush()?;
        self.file.seek(SeekFrom::Start(0))?;
        let reference = destination.append_auxiliary(
            pool_id,
            LEGACY_CONCEPT_SEQUENCE_INDEX_KIND,
            |writer| io::copy(&mut self.file, writer).map(|_| ()),
        )?;
        drop(self.file);
        std::fs::remove_file(&self.path).ok();
        Ok(reference)
    }
}

pub(super) fn is_resumable(legacy_path: &Path, destination: &Path) -> bool {
    let Ok(legacy_bytes) = std::fs::metadata(legacy_path).map(|metadata| metadata.len()) else {
        return false;
    };
    let Ok(file) = WbrainFile::open(destination) else {
        return false;
    };
    decode_progress(&file.brain_metadata())
        .ok()
        .flatten()
        .is_some_and(|progress| progress.legacy_bytes == legacy_bytes)
}

pub(super) fn is_raw_complete(destination: &Path) -> bool {
    let Ok(file) = WbrainFile::open(destination) else {
        return false;
    };
    let metadata = file.brain_metadata();
    decode_progress(&metadata).ok().flatten().is_none()
        && bincode::deserialize::<WbrainBrainMetadata>(&metadata).is_ok()
        && !file.pool_ids().is_empty()
}

fn stream_pool<R: Read>(
    reader: &mut R,
    legacy_dir: &Path,
    file: &std::sync::Arc<WbrainFile>,
    expected_pool_id: PoolId,
    binding_pool_id: PoolId,
) -> io::Result<()> {
    let config: PoolConfig = read_value(reader)?;
    if config.id != expected_pool_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "legacy pool key {} disagrees with embedded id {}",
                expected_pool_id, config.id
            ),
        ));
    }
    let neuron_count: u64 = read_value(reader)?;
    let capacity = usize::try_from(neuron_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "neuron count exceeds usize"))?;
    let store = file.pool(config.id);
    store.set_index_concept_labels(config.id == binding_pool_id);
    store.prepare_paged_slots(capacity)?;
    let mut concept_index = DiskConceptIndex::create(legacy_dir, config.id, neuron_count)?;
    let mut concept_count = 0usize;
    let mut total_terminals = 0usize;
    let mut system = System::new();

    for expected_id in 0..neuron_count {
        if expected_id % MEMORY_CHECK_INTERVAL == 0 {
            require_memory_floor(
                &mut system,
                &format!("at pool {} neuron {}", config.id, expected_id),
            )?;
        }
        let neuron: Neuron = read_value(reader)?;
        if neuron.id as u64 != expected_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "pool {} expected neuron {}, found {}",
                    config.id, expected_id, neuron.id
                ),
            ));
        }
        total_terminals = total_terminals.saturating_add(neuron.terminals.len());
        if !neuron.is_atom() {
            concept_count = concept_count.saturating_add(1);
            let local_members: Vec<NeuronId> = neuron
                .members
                .iter()
                .filter(|member| member.pool == config.id)
                .map(|member| member.neuron)
                .collect();
            if !local_members.is_empty() {
                concept_index.insert(&local_members, neuron.id)?;
            }
        }
        store.persist_sleeping(&neuron)?;
    }
    store.flush_paged_slots()?;

    let label_count: u64 = read_value(reader)?;
    for _ in 0..label_count {
        skip_bincode_string(reader, &mut system)?;
        let _: NeuronId = read_value(reader)?;
    }
    require_memory_floor(
        &mut system,
        &format!("after discarding pool {} legacy labels", config.id),
    )?;
    let recent_atoms = read_recent_atoms(reader, config.recent_atoms_window)?;
    let legacy_sequence_ledger = stream_sequence_ledger(reader, &mut system, file, config.id)?;
    let legacy_concept_sequence_index = concept_index.finish(file, config.id)?;
    let cold_offset_count: u64 = read_value(reader)?;

    if cold_offset_count > 0 {
        let cold = ColdTier::open(
            legacy_dir
                .join("cold")
                .join(format!("pool_{}.cold", config.id)),
        )?;
        for index in 0..cold_offset_count {
            if index % MEMORY_CHECK_INTERVAL == 0 {
                require_memory_floor(
                    &mut system,
                    &format!("while overlaying pool {} cold offsets", config.id),
                )?;
            }
            let id: NeuronId = read_value(reader)?;
            let offset: u64 = read_value(reader)?;
            let neuron = cold.read_neuron(offset)?;
            if neuron.id != id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "cold pool {} expected neuron {}, found {}",
                        config.id, id, neuron.id
                    ),
                ));
            }
            if let Some(previous) = store.get(id) {
                total_terminals = total_terminals.saturating_sub(previous.terminals.len());
                store.release_cached(id);
            }
            total_terminals = total_terminals.saturating_add(neuron.terminals.len());
            store.persist_sleeping(&neuron)?;
        }
    }

    Pool::stage_streamed_wbrain_metadata(
        &store,
        StreamedPoolMetadata {
            config,
            recent_atoms,
            sequences: Vec::new(),
            legacy_sequence_ledger: Some(legacy_sequence_ledger),
            concept_sequence_to_id: Vec::new(),
            legacy_concept_sequence_index: Some(legacy_concept_sequence_index),
            neuron_kinds: Vec::new(),
            concept_slots: Vec::new(),
            born_ticks: Vec::new(),
            concept_count,
            total_terminals,
        },
    )
}

pub(super) fn migrate(legacy_path: &Path, destination: &Path) -> io::Result<usize> {
    let legacy = File::open(legacy_path)?;
    let legacy_bytes = legacy.metadata()?.len();
    let mut reader = BufReader::with_capacity(1024 * 1024, legacy);
    let destination_existed = destination.exists();
    let file = WbrainFile::open(destination)?;

    let _format_version: u32 = read_value(&mut reader)?;
    let binding_pool_id: PoolId = read_value(&mut reader)?;
    let binding_emergence_threshold: u32 = read_value(&mut reader)?;
    let tentative_emergence_threshold: u32 = read_value(&mut reader)?;
    let moment_history_window: usize = read_value(&mut reader)?;

    let fabric_config = read_value(&mut reader)?;
    let tick: u64 = read_value(&mut reader)?;
    let _pool_order: Vec<PoolId> = read_value(&mut reader)?;
    let pool_count: u64 = read_value(&mut reader)?;
    let initial_pool_offset = reader.stream_position()?;
    let progress = decode_progress(&file.brain_metadata())?;
    let (completed_pools, source_offset) = match progress {
        Some(progress)
            if progress.legacy_bytes == legacy_bytes
                && progress.legacy_tick == tick
                && progress.pool_count == pool_count
                && progress.completed_pools <= pool_count
                && progress.source_offset >= initial_pool_offset
                && progress.source_offset <= legacy_bytes =>
        {
            (progress.completed_pools, progress.source_offset)
        }
        Some(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "existing migration progress does not match the legacy checkpoint",
            ));
        }
        None if destination_existed => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "existing .wbrain is complete or is not a resumable migration",
            ));
        }
        None => {
            file.set_tick(tick);
            file.set_brain_metadata(encode_progress(&MigrationProgress {
                legacy_bytes,
                legacy_tick: tick,
                pool_count,
                completed_pools: 0,
                source_offset: initial_pool_offset,
            })?);
            file.commit_manifest()?;
            (0, initial_pool_offset)
        }
    };
    reader.seek(std::io::SeekFrom::Start(source_offset))?;
    let legacy_dir = legacy_path.parent().unwrap_or_else(|| Path::new("."));
    for completed in completed_pools..pool_count {
        let pool_id: PoolId = read_value(&mut reader)?;
        stream_pool(&mut reader, legacy_dir, &file, pool_id, binding_pool_id)?;
        let next_source_offset = reader.stream_position()?;
        file.set_brain_metadata(encode_progress(&MigrationProgress {
            legacy_bytes,
            legacy_tick: tick,
            pool_count,
            completed_pools: completed + 1,
            source_offset: next_source_offset,
        })?);
        file.commit_manifest()?;
    }

    let eem: EemSnapshot = read_value(&mut reader)?;
    let annealer: AnnealerSnapshot = read_value(&mut reader)?;
    let moment_history: VecDeque<SerializableFingerprint> = read_value(&mut reader)?;
    let binding_recurrences: Vec<(SerializableFingerprint, u32)> = read_value(&mut reader)?;
    let promoted_fingerprints: Vec<(SerializableFingerprint, NeuronId)> = read_value(&mut reader)?;
    let tentative_promoted: Vec<(SerializableFingerprint, NeuronId)> = read_value(&mut reader)?;
    let lifetime_recurrences: Vec<(SerializableFingerprint, u32)> = read_value(&mut reader)?;
    let current_threshold: u32 = read_value(&mut reader)?;
    let total_observations: u64 = read_value(&mut reader)?;
    let action_pool_id: Option<PoolId> = read_value(&mut reader)?;
    let pending_actions: Vec<(ActionId, ActionEvent)> = read_value(&mut reader)?;
    let next_action_id: ActionId = read_value(&mut reader)?;

    let config = BrainConfig {
        fabric: fabric_config,
        binding_emergence_threshold,
        tentative_emergence_threshold,
        moment_history_window,
        min_atom_score: default_min_atom_score(),
        pressure_band_low: default_pressure_band_low(),
        pressure_band_high: default_pressure_band_high(),
        pressure_threshold_max: default_pressure_threshold_max(),
        pressure_observation_grace: default_pressure_observation_grace(),
        pressure_adjust_enabled: default_pressure_adjust_enabled(),
        binding_pool_config: PoolConfig::defaults("binding", binding_pool_id),
        eem: eem.config.clone(),
        annealer: annealer.config.clone(),
    };
    let metadata = WbrainBrainMetadata {
        config,
        binding_pool_id,
        moment_history: moment_history.into_iter().map(persisted).collect(),
        binding_recurrences: binding_recurrences
            .into_iter()
            .map(|(fingerprint, count)| (persisted(fingerprint), count))
            .collect(),
        lifetime_recurrences: lifetime_recurrences
            .into_iter()
            .map(|(fingerprint, count)| (persisted(fingerprint), count))
            .collect(),
        tentative_promoted: tentative_promoted
            .into_iter()
            .map(|(fingerprint, id)| (persisted(fingerprint), id))
            .collect(),
        promoted_fingerprints: promoted_fingerprints
            .into_iter()
            .map(|(fingerprint, id)| (persisted(fingerprint), id))
            .collect(),
        binding_sequence_index: Vec::new(),
        binding_feature_atom_index: Vec::new(),
        binding_motif_index: Vec::new(),
        total_observations,
        current_threshold,
        last_pressure_check_obs: total_observations,
        action_pool_id,
        pending_actions,
        next_action_id,
        eem,
        annealer,
    };
    file.set_brain_metadata(
        bincode::serialize(&metadata)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
    );
    file.set_tick(tick);
    file.commit_manifest()?;
    file.flush()?;

    let position = reader.stream_position()?;
    let length = reader.get_ref().metadata()?.len();
    if position != length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("legacy parser stopped at byte {position} of {length}"),
        ));
    }
    Ok(file
        .pool_ids()
        .into_iter()
        .map(|pool_id| file.pool(pool_id).known_count())
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::{AtomEncoding, BytePassthroughEncoding};
    use crate::{Brain, BrainConfig};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn tmpfile(name: &str, extension: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("w1z4rd_{name}_{}_{}", std::process::id(), nonce))
            .with_extension(extension)
    }

    #[test]
    fn migration_resumes_from_a_durable_pool_boundary() {
        let legacy = tmpfile("resume-source", "bin");
        let destination = tmpfile("resume-destination", "wbrain");
        let binding_pool_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            for (id, name, prefix, bytes) in [
                (3, "left", "l", b"abc".as_slice()),
                (4, "right", "r", b"xyz".as_slice()),
            ] {
                brain.create_pool(
                    PoolConfig::defaults(name, id),
                    Box::new(BytePassthroughEncoding {
                        prefix: prefix.into(),
                    }),
                );
                brain.observe(id, bytes);
            }
            brain.checkpoint(&legacy).unwrap();
        }

        let legacy_bytes = std::fs::metadata(&legacy).unwrap().len();
        let mut reader = BufReader::new(File::open(&legacy).unwrap());
        let _: u32 = read_value(&mut reader).unwrap();
        let parsed_binding_pool_id: PoolId = read_value(&mut reader).unwrap();
        let _: u32 = read_value(&mut reader).unwrap();
        let _: u32 = read_value(&mut reader).unwrap();
        let _: usize = read_value(&mut reader).unwrap();
        let _: crate::fabric::FabricConfig = read_value(&mut reader).unwrap();
        let tick: u64 = read_value(&mut reader).unwrap();
        let _: Vec<PoolId> = read_value(&mut reader).unwrap();
        let pool_count: u64 = read_value(&mut reader).unwrap();
        let first_pool_id: PoolId = read_value(&mut reader).unwrap();
        let file = WbrainFile::open(&destination).unwrap();
        stream_pool(
            &mut reader,
            legacy.parent().unwrap(),
            &file,
            first_pool_id,
            parsed_binding_pool_id,
        )
        .unwrap();
        let source_offset = reader.stream_position().unwrap();
        file.set_tick(tick);
        file.set_brain_metadata(
            encode_progress(&MigrationProgress {
                legacy_bytes,
                legacy_tick: tick,
                pool_count,
                completed_pools: 1,
                source_offset,
            })
            .unwrap(),
        );
        file.commit_manifest().unwrap();
        drop(file);

        assert!(is_resumable(&legacy, &destination));
        assert_eq!(migrate(&legacy, &destination).unwrap(), 6);
        assert!(!is_resumable(&legacy, &destination));

        let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
        for (id, prefix) in [(binding_pool_id, "bind"), (3, "l"), (4, "r")] {
            encodings.insert(
                id,
                Box::new(BytePassthroughEncoding {
                    prefix: prefix.into(),
                }),
            );
        }
        let (restored, missing) = Brain::restore_wbrain(&destination, encodings).unwrap();
        assert!(missing.is_empty());
        assert_eq!(restored.stats().total_neurons, 6);
        assert_eq!(restored.stats().evicted_neurons, 6);

        std::fs::remove_file(legacy).ok();
        std::fs::remove_file(destination).ok();
    }

    #[test]
    fn migrated_sequence_recurrence_is_looked_up_without_hydrating_the_ledger() {
        let legacy = tmpfile("sequence-ledger-source", "bin");
        let destination = tmpfile("sequence-ledger-destination", "wbrain");
        let binding_pool_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            brain.create_pool(
                PoolConfig::defaults("frames", 9),
                Box::new(BytePassthroughEncoding {
                    prefix: "frame".into(),
                }),
            );
            let pool = brain.fabric().pool(9).unwrap();
            let first = pool
                .write()
                .ensure_frame_concept_for_pretrain(b"cold-ledger-frame", 1);
            assert!(first.len() > 1);
            brain.checkpoint(&legacy).unwrap();
        }

        migrate(&legacy, &destination).unwrap();
        let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
        encodings.insert(
            binding_pool_id,
            Box::new(BytePassthroughEncoding {
                prefix: "bind".into(),
            }),
        );
        encodings.insert(
            9,
            Box::new(BytePassthroughEncoding {
                prefix: "frame".into(),
            }),
        );
        let (restored, missing) = Brain::restore_wbrain(&destination, encodings).unwrap();
        assert!(missing.is_empty());
        let pool = restored.fabric().pool(9).unwrap();
        let promoted = pool
            .write()
            .ensure_frame_concept_for_pretrain(b"cold-ledger-frame", 2);
        assert_eq!(promoted.len(), 1);
        assert!(!pool.write().get(promoted[0]).unwrap().is_atom());

        std::fs::remove_file(legacy).ok();
        std::fs::remove_file(destination).ok();
    }

    #[test]
    fn migrated_concept_membership_is_resolved_from_the_cold_index() {
        let legacy = tmpfile("concept-index-source", "bin");
        let destination = tmpfile("concept-index-destination", "wbrain");
        let binding_pool_id;
        let original_id;
        {
            let mut brain = Brain::new(BrainConfig::default());
            binding_pool_id = brain.binding_pool_id();
            brain.create_pool(
                PoolConfig::defaults("frames", 10),
                Box::new(BytePassthroughEncoding {
                    prefix: "frame".into(),
                }),
            );
            let pool = brain.fabric().pool(10).unwrap();
            pool.write()
                .ensure_frame_concept_for_pretrain(b"cold-concept-frame", 1);
            original_id = pool
                .write()
                .ensure_frame_concept_for_pretrain(b"cold-concept-frame", 2)[0];
            brain.checkpoint(&legacy).unwrap();
        }

        migrate(&legacy, &destination).unwrap();
        let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
        encodings.insert(
            binding_pool_id,
            Box::new(BytePassthroughEncoding {
                prefix: "bind".into(),
            }),
        );
        encodings.insert(
            10,
            Box::new(BytePassthroughEncoding {
                prefix: "frame".into(),
            }),
        );
        let (restored, missing) = Brain::restore_wbrain(&destination, encodings).unwrap();
        assert!(missing.is_empty());
        let pool = restored.fabric().pool(10).unwrap();
        let before = pool.read().neuron_count();
        let resolved = pool
            .write()
            .ensure_frame_concept_for_pretrain(b"cold-concept-frame", 3);
        assert_eq!(resolved, vec![original_id]);
        assert_eq!(
            pool.read().neuron_count(),
            before,
            "lookup must not duplicate concept"
        );

        std::fs::remove_file(legacy).ok();
        std::fs::remove_file(destination).ok();
    }
}
