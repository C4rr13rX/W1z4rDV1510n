//! Brain Construction Kit — substrate per [`ARCHITECTURE.md`].
//!
//! Phase 1 (per spec §11) lives here.  Crates/core's existing neural fabric
//! is left running in place until later phases migrate over.  This crate
//! is the substrate that compounds toward AGI; crates/core's drift-laden
//! fabric is what we are migrating away from.
//!
//! What's implemented in Phase 1:
//! - Unified [`Neuron`] struct with [`NeuronKind`], [`NeuronRef`]-based
//!   members, and per-neuron [`Terminal`]s that can cross pool boundaries.
//! - [`Pool`] with pluggable [`AtomEncoding`] contract and automatic
//!   concept emergence from recurring sequences in `recent_atoms`.
//! - [`Fabric`] with a tick-aligned moment buffer that auto-wires
//!   cross-pool axon terminals between any pair of neurons in different
//!   pools that fire in the same tick.
//! - [`GroundingReport`] + [`AnswerWithGrounding`] — the answer contract
//!   from spec §2.  EEM and annealer fields are present and reported as
//!   `None` until those subsystems come online in later phases.
//!
//! What's deliberately absent (per spec §9):
//! - No position-augmented atoms.  Atoms are raw bits via base64
//!   transport.
//! - No precomputed bigrams/trigrams as atoms.  Sequences emerge as
//!   concepts.
//! - No kwta / sparsity gate.  Sparsity emerges from decay + prune.
//! - No Jaccard ranker in the API surface.  Caller reads activation.
//! - No confidence threshold gate.  Caller reads the GroundingReport.
//! - No `MultiPoolFabric.cross` map.  Cross-pool wiring is per-neuron.
//! - No `paired_text` / `session_id` semantics in observe.

pub mod neuron;
pub mod pool;
pub mod fabric;
pub mod grounding;
pub mod brain;
pub mod action;
pub mod eem;
pub mod annealer;
pub mod persistence;
pub mod identity;
pub mod network;
pub mod store;

pub use neuron::{Neuron, NeuronId, NeuronKind, NeuronRef, PoolId, Terminal};
pub use pool::{AtomEncoding, BytePassthroughEncoding, ControlMode, ControlSignal, ControlState, Pool, PoolConfig};
pub use fabric::{Fabric, FabricConfig, Moment, SettleResult, TickProfile, TickProfileSnapshot};
pub use grounding::{
    AnswerWithGrounding, ConfidenceTier, GroundingReport, RequestObservation,
};
pub use brain::{
    Brain, BrainConfig, BrainStats, BindingMatch, MatchTier,
    DecodedConcept, EvictionParams, EvictionStats, PoolExtrusion, ResonantExtrusion,
};
pub use action::{ActionEvent, ActionId, ActionRouter, NullRouter, RouteResult};
pub use eem::{
    ChainResult, Discipline, DisciplineId, Eem, EemConfig, Equation,
    EquationApplication, EquationId, FactId, GroundedFact, Motif, MotifId,
    Variable, VariableId,
};
pub use annealer::{Annealer, AnnealerConfig, Frame, PredictionResult};
pub use persistence::{
    AnnealerSnapshot, BrainSnapshot, EemSnapshot, FabricSnapshot, PoolSnapshot,
    SerializableFingerprint, load_snapshot, save_snapshot,
};
pub use identity::{
    BrainIdentitySpec, EncodingFactory, IdentityBuildError, PoolKind,
    PoolPrototypeRegistry, PoolSpec,
};
pub use network::{
    BrainId, GossipEquation, GossipMotif, NetworkState, PeerAccuracy,
    PeerContribution,
};
pub use store::{
    MmapWalStore, NoopStore, RecoveryStats, Store, TerminalDelta, WalEvent,
    replay_into_brain,
};
