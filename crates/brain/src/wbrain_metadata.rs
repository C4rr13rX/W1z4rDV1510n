//! Compact non-neuron state stored in a `.wbrain` manifest.
//!
//! Neuron bodies never appear here. This metadata is sufficient to rebuild
//! the brain's routing/index state while every neuron remains serialized.

use std::collections::VecDeque;

use crate::action::{ActionEvent, ActionId};
use crate::brain::BrainConfig;
use crate::neuron::{NeuronId, PoolId};
use crate::persistence::{AnnealerSnapshot, EemSnapshot};
use crate::store::AuxiliaryRecordRef;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct PersistedMomentFingerprint {
    pub pairs: Vec<(PoolId, NeuronId)>,
    pub ordered_per_pool: Vec<(PoolId, Vec<NeuronId>)>,
    pub members_per_pool: Vec<(PoolId, Vec<NeuronId>)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct WbrainBrainMetadata {
    pub config: BrainConfig,
    pub binding_pool_id: PoolId,
    pub moment_history: VecDeque<PersistedMomentFingerprint>,
    pub binding_recurrences: Vec<(PersistedMomentFingerprint, u32)>,
    pub lifetime_recurrences: Vec<(PersistedMomentFingerprint, u32)>,
    pub tentative_promoted: Vec<(PersistedMomentFingerprint, NeuronId)>,
    pub promoted_fingerprints: Vec<(PersistedMomentFingerprint, NeuronId)>,
    pub binding_sequence_index: Vec<((PoolId, PoolId, Vec<NeuronId>), Vec<NeuronId>)>,
    pub binding_feature_atom_index: Vec<((PoolId, NeuronId), Vec<NeuronId>)>,
    pub binding_motif_index: Vec<((PoolId, [u8; 3]), Vec<NeuronId>)>,
    pub binding_posting_indexes: Vec<AuxiliaryRecordRef>,
    pub total_observations: u64,
    pub current_threshold: u32,
    pub last_pressure_check_obs: u64,
    pub action_pool_id: Option<PoolId>,
    pub pending_actions: Vec<(ActionId, ActionEvent)>,
    pub next_action_id: ActionId,
    pub eem: EemSnapshot,
    pub annealer: AnnealerSnapshot,
}

/// Rewrite every container offset held inside a serialized brain metadata blob.
///
/// `binding_posting_indexes` is the only persisted reference vector here:
/// `fingerprint_posting_indexes` and `fingerprint_candidate_indexes` are
/// DERIVED from generation markers on restore (see `brain.rs`), so they hold no
/// durable offsets and must not be re-derived by a compactor.
///
/// As in `pool::remap_pool_metadata_refs`, the destructuring is exhaustive on
/// purpose: a future reference field added to [`WbrainBrainMetadata`] becomes a
/// compile error here rather than a brain whose binding recall silently reads
/// whatever bytes landed at a stale offset.
pub(crate) fn remap_brain_metadata_refs(
    blob: &[u8],
    remap: &mut dyn FnMut(AuxiliaryRecordRef) -> std::io::Result<AuxiliaryRecordRef>,
) -> std::io::Result<Vec<u8>> {
    if blob.is_empty() {
        return Ok(Vec::new());
    }
    let decoded: WbrainBrainMetadata = bincode::deserialize(blob)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let WbrainBrainMetadata {
        config,
        binding_pool_id,
        moment_history,
        binding_recurrences,
        lifetime_recurrences,
        tentative_promoted,
        promoted_fingerprints,
        binding_sequence_index,
        binding_feature_atom_index,
        binding_motif_index,
        binding_posting_indexes,
        total_observations,
        current_threshold,
        last_pressure_check_obs,
        action_pool_id,
        pending_actions,
        next_action_id,
        eem,
        annealer,
    } = decoded;
    let mut moved = Vec::with_capacity(binding_posting_indexes.len());
    for reference in binding_posting_indexes {
        moved.push(remap(reference)?);
    }
    let rebuilt = WbrainBrainMetadata {
        config,
        binding_pool_id,
        moment_history,
        binding_recurrences,
        lifetime_recurrences,
        tentative_promoted,
        promoted_fingerprints,
        binding_sequence_index,
        binding_feature_atom_index,
        binding_motif_index,
        binding_posting_indexes: moved,
        total_observations,
        current_threshold,
        last_pressure_check_obs,
        action_pool_id,
        pending_actions,
        next_action_id,
        eem,
        annealer,
    };
    bincode::serialize(&rebuilt)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}
