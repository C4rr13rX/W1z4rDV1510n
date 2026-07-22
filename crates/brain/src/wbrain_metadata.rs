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
