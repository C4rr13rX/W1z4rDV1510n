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
    pub config: PoolConfig,
    pub neurons: Vec<Neuron>,
    pub label_to_id: HashMap<String, NeuronId>,
    pub recent_atoms: VecDeque<NeuronId>,
    /// Map serialized as parallel vecs because HashMap<Vec<NeuronId>, u32>
    /// can hit serde corner cases when the key is itself a sequence.
    pub sequences: Vec<(Vec<NeuronId>, u32)>,
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
    pub config: FabricConfig,
    pub tick: u64,
    /// Order matters: pool ids are re-registered in this order so
    /// the cross-pool wiring on subsequent observations behaves
    /// identically across restore cycles.
    pub pool_order: Vec<PoolId>,
    pub pools: HashMap<PoolId, PoolSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EemSnapshot {
    pub config: EemConfig,
    pub equations: Vec<Equation>,
    pub variables: Vec<Variable>,
    pub disciplines: Vec<Discipline>,
    pub motifs: Vec<Motif>,
    pub motif_links: Vec<(u32, Vec<u32>)>,
    #[serde(default)]
    pub facts: Vec<GroundedFact>,
    #[serde(default)]
    pub semantic_relations: Vec<crate::workspace::GroundedRelation>,
    #[serde(default)]
    pub composition_rules: Vec<crate::workspace::CompositionRule>,
    #[serde(default)]
    pub crystallizer: crate::crystallizer::SemanticCrystallizer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealerSnapshot {
    pub config: AnnealerConfig,
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
/// * `3`  EEM semantic relations, composition rules, and crystallized role
///        templates used by transient logical inference.
pub const CURRENT_SNAPSHOT_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    /// Snapshot format version — see `CURRENT_SNAPSHOT_VERSION`.
    /// `#[serde(default)]` returns 0 for pre-Stage-11 snapshots so
    /// `from_snapshot` can dispatch the right restore path.
    #[serde(default)]
    pub format_version: u32,
    pub binding_pool_id: PoolId,
    pub binding_emergence_threshold: u32,
    /// Two-tier emergence: tentative threshold (default 1).  Defaults
    /// to 1 on restore from older snapshots that didn't carry the
    /// field — matches `default_tentative_emergence_threshold()`.
    #[serde(default = "default_tentative_threshold_for_restore")]
    pub tentative_emergence_threshold: u32,
    pub moment_history_window: usize,
    pub fabric: FabricSnapshot,
    pub eem: EemSnapshot,
    pub annealer: AnnealerSnapshot,
    pub moment_history: VecDeque<SerializableFingerprint>,
    pub binding_recurrences: Vec<(SerializableFingerprint, u32)>,
    pub promoted_fingerprints: Vec<(SerializableFingerprint, NeuronId)>,
    /// Two-tier emergence: tentative-tier promotions.  Defaults to
    /// empty on restore from older snapshots.
    #[serde(default)]
    pub tentative_promoted: Vec<(SerializableFingerprint, NeuronId)>,
    /// Lifetime (non-decaying) recurrence count per fingerprint.
    /// Defaults to empty on restore — the brain will re-accumulate
    /// from subsequent observations.
    #[serde(default)]
    pub lifetime_recurrences: Vec<(SerializableFingerprint, u32)>,
    /// Pressure-adjusted consolidated threshold at snapshot time.
    /// Zero on restore from older snapshots, which from_snapshot
    /// treats as "use `binding_emergence_threshold` instead".
    #[serde(default)]
    pub current_threshold: u32,
    /// Total non-empty-fingerprint observations since construction.
    #[serde(default)]
    pub total_observations: u64,
    pub action_pool_id: Option<PoolId>,
    pub pending_actions: Vec<(ActionId, ActionEvent)>,
    pub next_action_id: ActionId,
}

fn default_tentative_threshold_for_restore() -> u32 {
    1
}

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
    save_serializable(snap, path)
}

/// Streaming checkpoint primitive shared by owned snapshots and the
/// borrowed live-brain view used by `Brain::checkpoint`.  Keeping this
/// generic is what lets the production path avoid materialising a cloned
/// `BrainSnapshot` while preserving the existing bincode wire format.
pub(crate) fn save_serializable<T, P>(value: &T, path: P) -> io::Result<()>
where
    T: Serialize + ?Sized,
    P: AsRef<Path>,
{
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
        bincode::serialize_into(&mut w, value)
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
    use bincode::Options;
    use std::io::BufReader;
    // Symmetric streaming read — never allocates a Vec<u8> the size of the
    // whole file.  Bincode's deserialise reads only as much as the next
    // primitive demands.
    //
    // BOUNDED, because "only as much as the next primitive demands" is
    // exactly the problem when the file is damaged.  Bincode reads a length
    // prefix and immediately allocates that many elements; on a truncated or
    // corrupt snapshot that length is whatever bytes happen to sit there.
    //
    // Observed 2026-09-05: brain-data/brain.bin.tmp was 179 bytes — a
    // checkpoint that died mid-write — and the node aborted on startup with
    //
    //     memory allocation of 4575657221408424000 bytes failed
    //
    // 4.5 exabytes.  Not a real request: a garbage length prefix read as a
    // count.  The process died before it could report a parse error, so the
    // brain looked like it "kept crashing" rather than "has one damaged
    // file", and brain.bin has not been checkpointed since Aug 19 while the
    // WAL grew to 75 MB.
    //
    // The limit is deliberately generous — larger than any legitimate
    // snapshot this machine can hold — so it never rejects real data.  Its
    // only job is to turn an impossible allocation into an error the caller
    // can act on.
    let file = fs::File::open(path)?;
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = BufReader::with_capacity(256 * 1024, file);

    // A serialised structure cannot be larger than a generous multiple of
    // the bytes it was read from.  Anything past that is corruption, not
    // data.
    let limit = len.saturating_mul(64).max(64 * 1024 * 1024);

    bincode::options()
        .with_limit(limit)
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .deserialize_from(&mut r)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod truncated_snapshot_tests {
    use super::*;
    use std::io::Write;

    /// A snapshot that died mid-write must fail as an ERROR, not abort the
    /// process.
    ///
    /// These are the first 179 bytes of the real brain-data/brain.bin.tmp
    /// recovered on 2026-09-05. Loading it aborted the node with
    /// "memory allocation of 4575657221408424000 bytes failed" -- a garbage
    /// length prefix read as an element count. The node had been dying on
    /// startup for days, and the failure looked like instability rather than
    /// one damaged file, because an abort leaves no error to report.
    #[test]
    fn a_truncated_snapshot_errors_instead_of_aborting() {
        let dir = std::env::temp_dir().join("w1z4rd_truncated_snapshot_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("torn.bin");

        // Header bytes as recovered, then nothing -- the write died here.
        let torn: [u8; 32] = [
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x9a, 0x99, 0x19, 0x3e, 0x02, 0x00, 0x00, 0x00,
        ];
        {
            let mut f = fs::File::create(&path).expect("create torn snapshot");
            f.write_all(&torn).expect("write torn snapshot");
        }

        // The contract is simply that it RETURNS. Before the bound, this
        // line never returned at all -- the allocator aborted the process.
        let result = load_snapshot(&path);
        assert!(result.is_err(), "a torn snapshot must not deserialise");

        let _ = fs::remove_file(&path);
    }

    /// The bound must never reject data that is merely large.
    ///
    /// Exercises the same bincode options the loader uses against a payload
    /// far bigger than any header, so a limit set too tight would fail here
    /// rather than in production on a real brain.
    #[test]
    fn the_bound_does_not_reject_legitimate_data() {
        use bincode::Options;

        let dir = std::env::temp_dir().join("w1z4rd_truncated_snapshot_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("big.bin");

        // A million elements: larger than any snapshot header, and exactly
        // the shape (length prefix + payload) that the bound governs.
        let payload: Vec<u64> = (0..1_000_000u64).collect();
        {
            let f = fs::File::create(&path).expect("create");
            let mut w = std::io::BufWriter::new(f);
            bincode::serialize_into(&mut w, &payload).expect("serialise");
            w.flush().expect("flush");
        }

        let f = fs::File::open(&path).expect("open");
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        let mut r = std::io::BufReader::new(f);
        let limit = len.saturating_mul(64).max(64 * 1024 * 1024);
        let back: Vec<u64> = bincode::options()
            .with_limit(limit)
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .deserialize_from(&mut r)
            .expect("a payload we just wrote must load back");
        assert_eq!(back.len(), 1_000_000);

        let _ = fs::remove_file(&path);
    }
}
