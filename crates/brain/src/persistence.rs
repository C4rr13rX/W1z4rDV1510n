//! Brain checkpoint + restore per [`ARCHITECTURE.md`] §6 + §11 Phase 9.
//!
//! Persists the learned state of a brain — neurons, terminals,
//! sequences, EEM equations/motifs, annealer history, action state,
//! binding-fingerprint history — to a single bincode-encoded file.
//! On restore, the caller re-supplies pool encodings (which are
//! stateless trait objects and therefore not serializable); the rest
//! of the brain is rebuilt verbatim.
//!
//! # Backend
//!
//! Spec §6.1 specifies redb (MIT) for cold-tier storage with one
//! file per pool.  This MVP ships a single-file bincode snapshot
//! behind a stable API surface ([`Brain::checkpoint`],
//! [`Brain::restore`]); swapping to redb is a backend change, not
//! an API change — the snapshot structs in this module are the
//! contract.
//!
//! # What's NOT persisted (deliberately transient)
//!
//! - Per-pool `currently_firing` / `activation` — these are the
//!   in-flight state of a single tick.  After restore the brain is
//!   "between observations"; the very next `observe` rebuilds them.
//! - `Fabric::current_moment` — same reason.  A new empty moment
//!   starts on restore; if the caller checkpoints mid-tick they
//!   lose the partial moment but learned synapses (which were
//!   updated at the last `advance_tick`) are intact.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::Path;

use crate::action::{ActionEvent, ActionId};
use crate::annealer::AnnealerConfig;
use crate::eem::{Discipline, EemConfig, Equation, GroundedFact, Motif, Variable};
use crate::fabric::FabricConfig;
use crate::neuron::{Neuron, NeuronId, PoolId};
use crate::pool::PoolConfig;

/// Per-pool persisted state.  Encoding is NOT serialized — the trait
/// object is stateless and is re-supplied at restore time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSnapshot {
    pub config:       PoolConfig,
    pub neurons:      Vec<Neuron>,
    pub label_to_id:  HashMap<String, NeuronId>,
    pub recent_atoms: VecDeque<NeuronId>,
    /// Map serialized as parallel vecs because HashMap<Vec<NeuronId>, u32>
    /// can hit serde corner cases when the key is itself a sequence.
    pub sequences:    Vec<(Vec<NeuronId>, u32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricSnapshot {
    pub config:      FabricConfig,
    pub tick:        u64,
    /// Order matters: pool ids are re-registered in this order so
    /// the cross-pool wiring on subsequent observations behaves
    /// identically across restore cycles.
    pub pool_order:  Vec<PoolId>,
    pub pools:       HashMap<PoolId, PoolSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EemSnapshot {
    pub config:      EemConfig,
    pub equations:   Vec<Equation>,
    pub variables:   Vec<Variable>,
    pub disciplines: Vec<Discipline>,
    pub motifs:      Vec<Motif>,
    pub motif_links: Vec<(u32, Vec<u32>)>,
    #[serde(default)]
    pub facts:       Vec<GroundedFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealerSnapshot {
    pub config:  AnnealerConfig,
    pub history: HashMap<PoolId, VecDeque<HashMap<NeuronId, f32>>>,
}

/// Public mirror of `brain::MomentFingerprint`.  Same data; lives
/// here so it can serialize without exposing `brain::MomentFingerprint`
/// (which stays private to keep the binding-emergence machinery
/// internal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableFingerprint {
    pub pairs: Vec<(PoolId, NeuronId)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    pub binding_pool_id:            PoolId,
    pub binding_emergence_threshold: u32,
    pub moment_history_window:       usize,
    pub fabric:                     FabricSnapshot,
    pub eem:                        EemSnapshot,
    pub annealer:                   AnnealerSnapshot,
    pub moment_history:             VecDeque<SerializableFingerprint>,
    pub binding_recurrences:        Vec<(SerializableFingerprint, u32)>,
    pub promoted_fingerprints:      Vec<(SerializableFingerprint, NeuronId)>,
    pub action_pool_id:             Option<PoolId>,
    pub pending_actions:            Vec<(ActionId, ActionEvent)>,
    pub next_action_id:             ActionId,
}

/// Write a [`BrainSnapshot`] to `path` using bincode.  Returns an
/// `io::Error` for filesystem failures or `InvalidData` for
/// serialization failures (the latter would indicate a programmer
/// error — every field above derives `Serialize`).
pub fn save_snapshot<P: AsRef<Path>>(snap: &BrainSnapshot, path: P) -> io::Result<()> {
    let bytes = bincode::serialize(snap)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, bytes)
}

pub fn load_snapshot<P: AsRef<Path>>(path: P) -> io::Result<BrainSnapshot> {
    let bytes = fs::read(path)?;
    bincode::deserialize(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
