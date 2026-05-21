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
use std::io::{self, Write};
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
    /// Stage 17.4 step 5: cold-tier neuron offsets per
    /// [`ARCHITECTURE.md`] §17.4.  Maps `NeuronId → byte offset` into
    /// the pool's cold-tier file (`<data_dir>/cold/pool_{id}.cold`).
    /// On restore, every neuron with an entry here is marked evicted
    /// and the brain knows where to fetch it.  `#[serde(default)]` for
    /// forward-compat with pre-17.4 snapshots.
    #[serde(default)]
    pub cold_offsets: Vec<(NeuronId, u64)>,
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

/// Current persisted-snapshot format version.  Bumped whenever the
/// `BrainSnapshot` field layout changes in a way bincode-1.3's
/// positional encoder cannot reconcile via `#[serde(default)]`.
///
/// Version history:
/// * `0`  legacy snapshots predating the audit (treated identically to v1)
/// * `1`  Stage 10 baseline (tentative-tier + lifetime + pressure fields)
/// * `2`  Stage 11 (concept-tier OOV reserved — no new persisted fields
///        in Stage 11A itself, but reserved here as the forward
///        contract so Stage 11B/C can layer on without further bumps)
pub const CURRENT_SNAPSHOT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    /// Snapshot format version — see `CURRENT_SNAPSHOT_VERSION`.
    /// `#[serde(default)]` returns 0 for pre-Stage-11 snapshots so
    /// `from_snapshot` can dispatch the right restore path.
    #[serde(default)]
    pub format_version:             u32,
    pub binding_pool_id:            PoolId,
    pub binding_emergence_threshold: u32,
    /// Two-tier emergence: tentative threshold (default 1).  Defaults
    /// to 1 on restore from older snapshots that didn't carry the
    /// field — matches `default_tentative_emergence_threshold()`.
    #[serde(default = "default_tentative_threshold_for_restore")]
    pub tentative_emergence_threshold: u32,
    pub moment_history_window:       usize,
    pub fabric:                     FabricSnapshot,
    pub eem:                        EemSnapshot,
    pub annealer:                   AnnealerSnapshot,
    pub moment_history:             VecDeque<SerializableFingerprint>,
    pub binding_recurrences:        Vec<(SerializableFingerprint, u32)>,
    pub promoted_fingerprints:      Vec<(SerializableFingerprint, NeuronId)>,
    /// Two-tier emergence: tentative-tier promotions.  Defaults to
    /// empty on restore from older snapshots.
    #[serde(default)]
    pub tentative_promoted:         Vec<(SerializableFingerprint, NeuronId)>,
    /// Lifetime (non-decaying) recurrence count per fingerprint.
    /// Defaults to empty on restore — the brain will re-accumulate
    /// from subsequent observations.
    #[serde(default)]
    pub lifetime_recurrences:       Vec<(SerializableFingerprint, u32)>,
    /// Pressure-adjusted consolidated threshold at snapshot time.
    /// Zero on restore from older snapshots, which from_snapshot
    /// treats as "use `binding_emergence_threshold` instead".
    #[serde(default)]
    pub current_threshold:          u32,
    /// Total non-empty-fingerprint observations since construction.
    #[serde(default)]
    pub total_observations:         u64,
    pub action_pool_id:             Option<PoolId>,
    pub pending_actions:            Vec<(ActionId, ActionEvent)>,
    pub next_action_id:             ActionId,
}

fn default_tentative_threshold_for_restore() -> u32 { 1 }

/// Write a [`BrainSnapshot`] to `path` using bincode.
///
/// Streams the snapshot directly into a buffered file writer via
/// [`bincode::serialize_into`].  Critically, this does *not* allocate the
/// entire serialised blob in RAM first — that approach OOMs the process on
/// any brain whose serialised size exceeds free physical memory (we hit this
/// empirically at ~8 GB blob / 463M terminals).  Peak heap during this call
/// is bounded by the buffered-writer capacity (~256 KB), not the brain size.
///
/// Per [`ARCHITECTURE.md`] §17.1, this is the *interim* path — content-
/// addressed per-neuron storage in [`crate::store`] is the long-term
/// substitute.  Until that ships in full, this implementation gives the
/// existing snapshot API the property it should have had from the start:
/// **memory cost is O(write-buffer), not O(brain)**.
///
/// Returns `io::Error` for filesystem failures or `InvalidData` for
/// serialisation failures (every field above derives `Serialize`).
pub fn save_snapshot<P: AsRef<Path>>(snap: &BrainSnapshot, path: P) -> io::Result<()> {
    use std::io::BufWriter;
    // Write to a sibling temp path then rename — atomic-replace guarantees
    // that a partially-written file never displaces a good one.  Crash
    // recovery sees either the previous good snapshot or the new good one,
    // never a torn intermediate.
    let final_path = path.as_ref();
    let tmp_path = final_path.with_extension("bin.tmp");

    {
        let file = fs::File::create(&tmp_path)?;
        let mut w = BufWriter::with_capacity(256 * 1024, file);
        bincode::serialize_into(&mut w, snap)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }

    // Atomic-replace.  Windows rename is atomic-replace for files on the
    // same volume since NTFS journals the operation.
    fs::rename(&tmp_path, final_path)?;
    Ok(())
}

pub fn load_snapshot<P: AsRef<Path>>(path: P) -> io::Result<BrainSnapshot> {
    use std::io::BufReader;
    // Symmetric streaming read — never allocates a Vec<u8> the size of the
    // whole file.  Bincode's deserialise reads only as much as the next
    // primitive demands.
    let file = fs::File::open(path)?;
    let mut r = BufReader::with_capacity(256 * 1024, file);
    bincode::deserialize_from(&mut r)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
