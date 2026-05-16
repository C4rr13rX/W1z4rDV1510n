//! Distributed-network protocol types per [`ARCHITECTURE.md`] §5.
//!
//! The brain crate defines the data model and API surface for
//! cross-brain motif gossip, equation-delta propagation, and
//! peer-augmented integration.  It does NOT own the transport —
//! that lives in `crates/cluster` (or any other transport plug-in)
//! and uses these types as the wire format.  Inverting that
//! dependency would couple cognition to a specific networking
//! library; spec §1.7 says the substrate's distributed layer is a
//! property of the brain factory, not a baked-in network stack.
//!
//! # Surface summary
//!
//! - **Motif gossip:** binding-concept promotions become
//!   [`GossipMotif`] records that drain into the cluster's outbound
//!   queue.  Inbound peer motifs land in `received_motifs`.
//! - **Equation gossip:** the EEM's current state can be exported as
//!   [`GossipEquation`] deltas; inbound deltas merge into the local
//!   EEM by confidence (higher wins).
//! - **Peer accuracy:** each peer brain's track record on supplied
//!   contributions is tracked in [`PeerAccuracy`]; integration weights
//!   peer answers by accuracy rate.
//! - **Peer-augmented grounding:** a caller hands a slice of
//!   [`PeerContribution`]s into `Brain::integrate_with_peers` and the
//!   resulting `GroundingReport.peer_contributions` lists each peer's
//!   weighted contribution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::neuron::{NeuronId, PoolId};

/// Stable identifier for a brain on the network.  Strings keep the
/// protocol legible and transport-agnostic (UUIDs, hostnames, or any
/// user-chosen identifier work).
pub type BrainId = String;

/// One motif as gossiped between brains.  Carries provenance so a
/// receiving brain can weight network knowledge against its own.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GossipMotif {
    pub source_brain:      BrainId,
    pub fingerprint:       Vec<(PoolId, NeuronId)>,
    pub observation_count: u32,
    pub local_confidence:  f32,
    pub observed_at_tick:  u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipEquation {
    pub source_brain:         BrainId,
    pub name:                 String,
    pub expression:           String,
    pub variable_names:       Vec<String>,
    pub discipline_name:      Option<String>,
    pub confidence:           f32,
    pub validation_successes: u32,
    pub validation_failures:  u32,
}

/// Per-peer accuracy track record.  Drives the weight a peer's
/// contributions earn in peer-augmented integration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerAccuracy {
    pub successful_contributions: u32,
    pub failed_contributions:     u32,
}

impl PeerAccuracy {
    /// Empirical success rate.  Unknown peers (no track record yet)
    /// return 0.5 — neither trusted nor disbelieved — so their first
    /// contribution gets neutral weight.
    pub fn rate(&self) -> f32 {
        let total = self.successful_contributions + self.failed_contributions;
        if total == 0 { 0.5 } else {
            self.successful_contributions as f32 / total as f32
        }
    }
}

/// One peer brain's pre-fetched contribution for a specific query.
/// Local integration combines these alongside the local fabric/EEM/
/// annealer outputs, weighted by per-peer accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerContribution {
    pub brain_id:              BrainId,
    pub fabric_confidence:     f32,
    pub eem_confidence:        Option<f32>,
    pub annealer_confidence:   Option<f32>,
    pub strongest_match_label: Option<String>,
}

/// Brain-side network state.  Pending-out queues are drained by the
/// transport layer; inbound records are stored in the "received"
/// indices.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct NetworkState {
    pub brain_id:           BrainId,
    pub pending_motif_out:  Vec<GossipMotif>,
    /// Inbound motifs keyed by (source_brain, fingerprint) so repeat
    /// gossip from the same peer about the same pattern updates the
    /// existing record (observation_count, local_confidence) rather
    /// than duplicating.
    pub received_motifs:    Vec<((BrainId, Vec<(PoolId, NeuronId)>), GossipMotif)>,
    pub peer_accuracy:      HashMap<BrainId, PeerAccuracy>,
}

impl NetworkState {
    pub fn new(brain_id: impl Into<String>) -> Self {
        Self {
            brain_id:          brain_id.into(),
            pending_motif_out: Vec::new(),
            received_motifs:   Vec::new(),
            peer_accuracy:     HashMap::new(),
        }
    }
}
