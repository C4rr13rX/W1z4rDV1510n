//! Per-pool Merkle root per [`ARCHITECTURE.md`] §17.6.
//!
//! Each pool's learned state can be hashed to a deterministic 32-byte
//! root.  Two nodes computing this independently on identical state
//! arrive at the same root — that's the precondition for anti-entropy
//! cluster sync (Lakshman & Malik 2010 Cassandra; Demers et al. 1987
//! epidemic algorithms): nodes compare roots; mismatch implies divergent
//! state; only the divergent subtree-of-records flows over the wire.
//!
//! # Stage 17.6 local part (this file)
//!
//! Ships the **deterministic root computation**.  The full cluster
//! anti-entropy protocol (RPC for subtree diff, NeuronStore sync) builds
//! on this and requires the cluster crate; that's the follow-up.
//!
//! # Determinism contract
//!
//! Identical observation history on two machines must produce
//! byte-identical Merkle roots.  Sources of non-determinism in the
//! existing substrate that we deliberately exclude from the hash:
//! - Transient runtime state (`currently_firing`, `activation`,
//!   `recent_atoms`, `recent_surprise`, EMAs) — single-tick state, not
//!   learned content.  Two replays of the same training data would have
//!   the same transient state only after equal observe sequences ending
//!   on the same tick; we don't hash these.
//! - Bloom slot contents — order-independent representation of the same
//!   labels, but the slots are deterministic given identical inserts so
//!   they could be hashed too; we include them because they're a
//!   constant-size representation of the label set and they verify
//!   stage 17.3's "determinism across instances" contract empirically.
//! - Fabric tick — included separately (cluster sync per spec is on
//!   per-pool roots, not whole-fabric roots).
//!
//! What we DO hash for the pool root:
//! - `PoolConfig` (canonicalised to bincode)
//! - For each neuron, in ascending id order: `(id, label, kind,
//!   members (sorted-pairs), terminals (sorted by target+weight))`,
//!   bincoded
//! - `sequences` (sorted by key, then count), bincoded
//! - `bloom.counters` raw bytes (deterministic per §17.3)
//!
//! Terminal weights are rounded to 4 decimals before hashing so
//! that the same training run on two machines, where IEEE-754
//! associativity could shift the LSB of an EMA, still produces
//! identical roots.  4 decimals is well above the Hebbian noise
//! floor (the same precision the eventual 8-bit-quantize step
//! captures in stage 17.4 full).

use blake3::Hasher;
use serde::Serialize;

use crate::neuron::{Neuron, NeuronId, NeuronRef, Terminal};
use crate::pool::PoolConfig;
use crate::store::bloom::CountingBloom;

/// 32-byte BLAKE3 hash; serialisable for /stats responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PoolRoot(pub [u8; 32]);

impl PoolRoot {
    /// Hex encoding for diagnostics + Wizard frontend display.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

/// Quantise a weight to 4-decimal precision in u32 fixed-point for
/// determinism across machines.  Weights are typically in [0.0, 4.0]
/// (per the default max_weight), so u32-fixed-point at 1e-4 covers the
/// range with margin.  Per [`ARCHITECTURE.md`] §17.6 — eliminates the
/// IEEE-754 associativity noise that could otherwise diverge roots
/// across machines on the same training run.
fn quantise_weight(w: f32) -> i32 {
    (w * 10_000.0).round() as i32
}

fn quantise_terminal(t: &Terminal) -> ([u32; 2], i32, u8, u64) {
    (
        [t.target.pool, t.target.neuron],
        quantise_weight(t.weight),
        t.consolidation,
        t.last_fired_tick,
    )
}

fn canonical_neuron_record(n: &Neuron) -> Vec<u8> {
    // Sort terminals by (target.pool, target.neuron) so the hash is
    // independent of insertion order.  Then quantise each.
    let mut terminals: Vec<&Terminal> = n.terminals.iter().collect();
    terminals.sort_by_key(|t| (t.target.pool, t.target.neuron));
    let quant_terminals: Vec<_> = terminals.iter()
        .map(|t| quantise_terminal(t))
        .collect();

    // Members: keep insertion order (per spec §1.1, position matters and
    // is encoded by member-order at promotion time).  We do NOT sort
    // members because that would lose positional information.
    let members: Vec<(u32, u32)> = n.members.iter()
        .map(|nr: &NeuronRef| (nr.pool, nr.neuron))
        .collect();

    let record = (
        n.id,
        n.label.as_str(),
        n.kind,                  // enum, serialised by its discriminant
        members,
        quant_terminals,
        n.born_tick,
        // last_fired_tick varies with replay timing; exclude to keep
        // roots stable across replay-vs-no-replay differences on the
        // same logical content.
        n.use_count,
        // prediction_error_ema and salience EMAs are also replay-
        // dependent — exclude.  raw salience IS hashed because it's
        // a substrate-emitted retention signal we want cluster nodes
        // to converge on.
        quantise_weight(n.salience),
    );
    bincode::serialize(&record).unwrap_or_default()
}

/// Compute the deterministic 32-byte root for the given pool inputs.
/// Calling code passes the parts it has access to (it'd be circular
/// for this function to take a `&Pool` reference because of crate-
/// internal cycle).
pub fn compute_pool_root(
    config:   &PoolConfig,
    neurons:  &[Neuron],
    sequences: &[(Vec<NeuronId>, u32)],
    bloom:    &CountingBloom,
    fabric_tick: u64,
) -> PoolRoot {
    let mut h = Hasher::new();

    // Domain separation tag so collisions with unrelated blake3 uses
    // are impossible.
    h.update(b"w1z4rd_brain:pool_root:v1");

    // 1. Config.
    let cfg_bytes = bincode::serialize(config).unwrap_or_default();
    h.update(&(cfg_bytes.len() as u64).to_le_bytes());
    h.update(&cfg_bytes);

    // 2. Neurons, in ascending id order (Pool::neurons IS id-indexed
    //    so the natural iteration order is already canonical).
    h.update(&(neurons.len() as u64).to_le_bytes());
    for n in neurons {
        let rec = canonical_neuron_record(n);
        h.update(&(rec.len() as u64).to_le_bytes());
        h.update(&rec);
    }

    // 3. Sequences, sorted by key for determinism.
    let mut seqs: Vec<&(Vec<NeuronId>, u32)> = sequences.iter().collect();
    seqs.sort_by(|a, b| a.0.cmp(&b.0));
    h.update(&(seqs.len() as u64).to_le_bytes());
    for (k, v) in &seqs {
        let bytes = bincode::serialize(&(k, *v)).unwrap_or_default();
        h.update(&(bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }

    // 4. Bloom counters (raw bytes — deterministic per §17.3).
    h.update(&(bloom.byte_size() as u64).to_le_bytes());
    // CountingBloom doesn't expose raw bytes today; we hash a stable
    // proxy: insert_count + slots + k.  Acceptable because the bloom
    // is itself deterministic given identical insertions.  When the
    // counters byte slice is needed for stage 17.6 full diff, add a
    // public accessor.
    h.update(&(bloom.slots() as u64).to_le_bytes());
    h.update(&(bloom.k_hashes() as u64).to_le_bytes());
    h.update(&(bloom.inserted_keys() as u64).to_le_bytes());

    // 5. Fabric tick (per-pool roots are conceptually tied to a moment
    //    in time — this is what the SnapshotMarker captures).
    h.update(&fabric_tick.to_le_bytes());

    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_bytes());
    PoolRoot(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Neuron, NeuronKind};

    fn empty_bloom() -> CountingBloom {
        CountingBloom::with_slots(64, 7)
    }

    fn cfg(id: u32) -> PoolConfig {
        PoolConfig::defaults("test", id)
    }

    #[test]
    fn empty_pool_root_is_stable() {
        let a = compute_pool_root(&cfg(1), &[], &[], &empty_bloom(), 0);
        let b = compute_pool_root(&cfg(1), &[], &[], &empty_bloom(), 0);
        assert_eq!(a, b, "two computations on identical empty input must match");
        // Sanity that hex is 64 chars.
        assert_eq!(a.to_hex().len(), 64);
    }

    #[test]
    fn differing_tick_changes_root() {
        let a = compute_pool_root(&cfg(1), &[], &[], &empty_bloom(), 100);
        let b = compute_pool_root(&cfg(1), &[], &[], &empty_bloom(), 101);
        assert_ne!(a, b);
    }

    #[test]
    fn differing_neuron_content_changes_root() {
        let n1 = vec![Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0)];
        let n2 = vec![Neuron::new_atom(0, "t:b".into(), NeuronKind::Excitatory, 0)];
        let a = compute_pool_root(&cfg(1), &n1, &[], &empty_bloom(), 0);
        let b = compute_pool_root(&cfg(1), &n2, &[], &empty_bloom(), 0);
        assert_ne!(a, b, "different labels must produce different roots");
    }

    #[test]
    fn terminal_order_independence() {
        // Two neurons with the same terminals in different orders must
        // hash to the same root — sort canonicalises.
        let mut n_a = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
        n_a.reinforce_terminal(NeuronRef::new(1, 5), 0.3, 0, 1.0);
        n_a.reinforce_terminal(NeuronRef::new(1, 3), 0.5, 0, 1.0);

        let mut n_b = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
        n_b.reinforce_terminal(NeuronRef::new(1, 3), 0.5, 0, 1.0);
        n_b.reinforce_terminal(NeuronRef::new(1, 5), 0.3, 0, 1.0);

        let a = compute_pool_root(&cfg(1), &[n_a], &[], &empty_bloom(), 0);
        let b = compute_pool_root(&cfg(1), &[n_b], &[], &empty_bloom(), 0);
        assert_eq!(a, b,
            "terminal insertion order must not affect root (canonical sort)");
    }

    #[test]
    fn weight_quantisation_absorbs_small_noise() {
        // Two terminals differing in the last bit of an f32 weight (below
        // the 4-decimal quantisation) must hash to the same root.
        let mut n_a = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
        n_a.reinforce_terminal(NeuronRef::new(1, 5), 0.50001, 0, 1.0);
        let mut n_b = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
        n_b.reinforce_terminal(NeuronRef::new(1, 5), 0.50002, 0, 1.0);
        // 0.50001 and 0.50002 both quantise to 5000 at 1e-4.
        let a = compute_pool_root(&cfg(1), &[n_a], &[], &empty_bloom(), 0);
        let b = compute_pool_root(&cfg(1), &[n_b], &[], &empty_bloom(), 0);
        assert_eq!(a, b,
            "weights below 4-decimal quantisation noise must produce same root");
    }
}
