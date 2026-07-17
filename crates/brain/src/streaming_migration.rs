//! Bounded-memory conversion of a legacy bincode checkpoint to `.wbrain`.
//!
//! The legacy format is positional. We read its vectors and maps field by
//! field, deserialize one neuron at a time, and immediately append that
//! neuron to the destination container.

use serde::de::DeserializeOwned;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;
use sysinfo::System;

use crate::action::{ActionEvent, ActionId};
use crate::neuron::{Neuron, NeuronId, PoolId};
use crate::persistence::{AnnealerSnapshot, EemSnapshot, SerializableFingerprint};
use crate::pool::{Pool, PoolConfig, StreamedPoolMetadata};
use crate::store::{ColdTier, WbrainFile};

use super::wbrain_metadata::{PersistedMomentFingerprint, WbrainBrainMetadata};
use super::{
    BrainConfig, default_min_atom_score, default_pressure_adjust_enabled,
    default_pressure_band_high, default_pressure_band_low, default_pressure_observation_grace,
    default_pressure_threshold_max,
};

const MIGRATION_MIN_AVAILABLE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MEMORY_CHECK_INTERVAL: u64 = 16_384;

fn read_value<R: Read, T: DeserializeOwned>(reader: &mut R) -> io::Result<T> {
    bincode::deserialize_from(reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn skip_bincode_string<R: Read>(reader: &mut R) -> io::Result<()> {
    let byte_count: u64 = read_value(reader)?;
    let copied = io::copy(&mut reader.take(byte_count), &mut io::sink())?;
    if copied != byte_count {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("legacy string declared {byte_count} bytes but only {copied} remained"),
        ));
    }
    Ok(())
}

fn persisted(fingerprint: SerializableFingerprint) -> PersistedMomentFingerprint {
    PersistedMomentFingerprint {
        pairs: fingerprint.pairs,
        ordered_per_pool: Vec::new(),
        members_per_pool: Vec::new(),
    }
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
    let mut neuron_kinds = Vec::with_capacity(capacity);
    let mut concept_slots = Vec::with_capacity(capacity);
    let mut born_ticks = Vec::with_capacity(capacity);
    let mut concept_sequence_to_id = Vec::new();
    let mut total_terminals = 0usize;
    let mut system = System::new();

    for expected_id in 0..neuron_count {
        if expected_id % MEMORY_CHECK_INTERVAL == 0 {
            system.refresh_memory();
            let available = system.available_memory();
            if available < MIGRATION_MIN_AVAILABLE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    format!(
                        "streaming migration stopped at pool {} neuron {}: {:.2} GiB available is below the 4 GiB runtime floor",
                        config.id,
                        expected_id,
                        available as f64 / 1024_f64.powi(3),
                    ),
                ));
            }
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
        neuron_kinds.push(neuron.kind);
        concept_slots.push(!neuron.is_atom());
        born_ticks.push(neuron.born_tick);
        total_terminals = total_terminals.saturating_add(neuron.terminals.len());
        if !neuron.is_atom() {
            let local_members: Vec<NeuronId> = neuron
                .members
                .iter()
                .filter(|member| member.pool == config.id)
                .map(|member| member.neuron)
                .collect();
            if !local_members.is_empty() {
                concept_sequence_to_id.push((local_members, neuron.id));
            }
        }
        store.persist_sleeping(&neuron)?;
    }

    let label_count: u64 = read_value(reader)?;
    for _ in 0..label_count {
        skip_bincode_string(reader)?;
        let _: NeuronId = read_value(reader)?;
    }
    let recent_atoms: VecDeque<NeuronId> = read_value(reader)?;
    let sequences: Vec<(Vec<NeuronId>, u32)> = read_value(reader)?;
    let cold_offsets: Vec<(NeuronId, u64)> = read_value(reader)?;

    if !cold_offsets.is_empty() {
        let cold = ColdTier::open(
            legacy_dir
                .join("cold")
                .join(format!("pool_{}.cold", config.id)),
        )?;
        for (id, offset) in cold_offsets {
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
            total_terminals = total_terminals.saturating_add(neuron.terminals.len());
            store.persist_sleeping(&neuron)?;
        }
    }

    Pool::stage_streamed_wbrain_metadata(
        &store,
        StreamedPoolMetadata {
            config,
            recent_atoms,
            sequences,
            concept_sequence_to_id,
            neuron_kinds,
            concept_slots,
            born_ticks,
            total_terminals,
        },
    )
}

pub(super) fn migrate(legacy_path: &Path, destination: &Path) -> io::Result<usize> {
    let legacy = File::open(legacy_path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, legacy);
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
    let legacy_dir = legacy_path.parent().unwrap_or_else(|| Path::new("."));
    for _ in 0..pool_count {
        let pool_id: PoolId = read_value(&mut reader)?;
        stream_pool(
            &mut reader,
            legacy_dir,
            &file,
            pool_id,
            binding_pool_id,
        )?;
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
