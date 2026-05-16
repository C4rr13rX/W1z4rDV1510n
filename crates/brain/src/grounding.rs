//! Answer contract per [`ARCHITECTURE.md`] §2.
//!
//! Every output carries a [`GroundingReport`].  The brain doesn't decide
//! whether to surface uncertainty — uncertainty is always surfaced.  The
//! caller decides how to present it.  No threshold gate, no decoder
//! fallback, no confident hallucination.

use serde::{Deserialize, Serialize};

use crate::neuron::{NeuronRef, PoolId};

/// Per-output grounding metrics.  See spec §2.1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingReport {
    /// Fraction of input atoms that mapped to known atoms in the pool.
    pub input_atom_coverage:    f32,
    /// Single strongest matching concept, if any.
    pub strongest_match:        Option<NeuronRef>,
    /// Jaccard overlap between strongest match's member set and input.
    pub strongest_match_jaccard: f32,
    /// All concepts that contributed to the answer (composition trace).
    pub composition_used:       Vec<NeuronRef>,
    /// Per-subsystem confidence.  At Phase 1 we only have fabric; EEM
    /// and annealer come online in later phases and are reported as
    /// `None` until then.  Spec §2.1.
    pub fabric_confidence:      f32,
    pub eem_confidence:         Option<f32>,
    pub annealer_confidence:    Option<f32>,
    pub integrated_confidence:  f32,
    /// True if no subsystem could ground the input adequately.  Caller
    /// should not interpret the answer as authoritative; should consider
    /// emitting a RequestObservation action.  Spec §2.2.
    pub outside_grounding:      bool,
    /// True if the answer came from cross-pool composition rather than
    /// direct retrieval of a single trained pair.  Spec §2.3.
    pub speculation_flag:       bool,
    /// Peer contributions that fed into integration.  Each entry is
    /// `(BrainId, weighted_confidence)` — the local integration's
    /// view of a specific peer's contribution after accuracy
    /// weighting.  Empty when no peers were consulted.  Spec §2.1.
    #[serde(default)]
    pub peer_contributions:     Vec<(crate::network::BrainId, f32)>,
}

impl GroundingReport {
    pub fn ungrounded() -> Self {
        Self {
            input_atom_coverage:    0.0,
            strongest_match:        None,
            strongest_match_jaccard: 0.0,
            composition_used:       Vec::new(),
            fabric_confidence:      0.0,
            eem_confidence:         None,
            annealer_confidence:    None,
            integrated_confidence:  0.0,
            outside_grounding:      true,
            speculation_flag:       false,
            peer_contributions:     Vec::new(),
        }
    }
}

/// Confidence tier — coarse summary of `integrated_confidence`.  Useful
/// for UI badging.  Boundaries are deliberately coarse; for numeric
/// decisions callers should read the float.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceTier {
    HighlyGrounded,
    PartiallyGrounded,
    Speculative,
    Ungrounded,
}

impl ConfidenceTier {
    pub fn from_confidence(c: f32, outside: bool, speculation: bool) -> Self {
        if outside              { Self::Ungrounded }
        else if speculation     { Self::Speculative }
        else if c >= 0.5        { Self::HighlyGrounded }
        else                    { Self::PartiallyGrounded }
    }
}

/// What the brain wants from the world when grounding is insufficient.
/// Routed via the action pool per spec §2.4.  At Phase 1 we surface this
/// as a structured value; routing wires up in Phase 7.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestObservation {
    pub domain:           String,
    pub examples_needed:  usize,
    pub why:              String,
    pub pool:             PoolId,
}

/// The single output type from the integration layer.  Spec §2.7: every
/// output carries grounding.  An `answer` of `None` plus
/// `outside_grounding: true` is the honest "I don't know yet" return.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerWithGrounding {
    pub answer:                   Option<Vec<u8>>,
    pub grounding:                GroundingReport,
    pub confidence_tier:          ConfidenceTier,
    pub next_steps_if_ungrounded: Vec<RequestObservation>,
}

impl AnswerWithGrounding {
    pub fn unknown(reason: impl Into<String>, pool: PoolId) -> Self {
        let grounding = GroundingReport::ungrounded();
        Self {
            answer: None,
            confidence_tier: ConfidenceTier::Ungrounded,
            grounding,
            next_steps_if_ungrounded: vec![RequestObservation {
                domain: reason.into(),
                examples_needed: 1,
                why: "no concept matches the input strongly enough".into(),
                pool,
            }],
        }
    }
}
