//! Action layer per [`ARCHITECTURE.md`] §4.E and §2.4.
//!
//! Action neurons are atoms in a designated action pool whose firings
//! produce external effects routed via deployment-spec channels.  When
//! one fires (typically as a result of cross-pool propagation from
//! sensor input), the brain emits an [`ActionEvent`] carrying:
//!  - which neuron fired (the routable identity)
//!  - which source neurons in OTHER pools fired it (credit attribution)
//!  - the tick of firing (for outcome timing)
//!  - a unique `action_id` so subsequent `feed_outcome` can find it
//!
//! The caller routes the event externally (webhook, MQTT, agent, human
//! notification, etc.) per its deployment, then later calls
//! [`crate::Brain::feed_outcome`] with the result.  Outcome score
//! reinforces (positive) or weakens (negative) the source→action
//! terminals — the closed loop that makes "best-learned-case-scenario"
//! emerge per spec §2 of the design conversation.

use serde::{Deserialize, Serialize};

use crate::neuron::NeuronRef;

/// Monotonically-increasing unique id per action firing.  Used by
/// `feed_outcome` to find the right action's source-terminal set for
/// reinforcement.
pub type ActionId = u64;

/// One action firing, ready to be routed externally + tracked for
/// outcome feedback.  The action neuron itself stays in the action
/// pool; the `sources` list records which neurons in other pools
/// drove it to fire this time, for later credit assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionEvent {
    pub id:             ActionId,
    pub action_neuron:  NeuronRef,
    pub action_label:   String,
    /// Currently-firing neurons elsewhere in the fabric whose terminals
    /// targeted the action neuron.  Reinforcement applies to the
    /// `source → action_neuron` terminals these neurons own.
    pub sources:        Vec<NeuronRef>,
    pub fired_at_tick:  u64,
    /// Activation level at firing.  Surfaced for caller telemetry —
    /// caller can choose to gate routing on activation strength.
    pub activation:     f32,
}

/// Pluggable external-routing handler.  Each deployment supplies
/// implementations for whatever channels its action atoms map to.
/// Phase 7 ships the trait + a no-op router (`NullRouter`) so tests and
/// development brains can run without external dependencies; concrete
/// routers (HTTP webhook, MQTT, etc.) land in deployment-spec wiring.
pub trait ActionRouter: Send + Sync {
    fn route(&self, event: &ActionEvent) -> RouteResult;
}

#[derive(Debug, Clone)]
pub enum RouteResult {
    /// The router accepted the event and dispatched it.  The brain
    /// keeps the action_id alive in `pending_actions` until
    /// `feed_outcome` arrives.
    Dispatched,
    /// The router has nothing to do with this action (e.g. unknown
    /// label).  The brain still keeps the action_id alive — outcome
    /// may arrive from a different mechanism (sensor observation of
    /// the world reacting).
    Ignored,
    /// The router failed (e.g. network error).  The brain may choose
    /// to retry or apply a negative outcome.
    Failed(String),
}

/// No-op router.  Used in tests and dev brains.  Every event returns
/// `Ignored`.  Real deployments swap this for a real router.
pub struct NullRouter;
impl ActionRouter for NullRouter {
    fn route(&self, _event: &ActionEvent) -> RouteResult {
        RouteResult::Ignored
    }
}
