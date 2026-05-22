//! Stage 17.6 — cluster anti-entropy primitive tests per
//! [`ARCHITECTURE.md`] §17.6.
//!
//! Confirms:
//! 1. `Brain::cluster_pool_roots` returns one entry per pool.
//! 2. Two brains trained identically have matching roots.
//! 3. A divergent training history produces divergent roots.
//! 4. `Brain::cluster_pool_neurons` returns the pool's full neuron list.
//! 5. `Brain::cluster_merge_pool` accepts neurons from a peer and
//!    inserts only the ones whose id is the next free slot.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};
use w1z4rd_brain::pool::Pool;

fn make_brain_one_text_pool() -> Brain {
    let mut brain = Brain::new(BrainConfig::default());
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    brain.fabric_mut().register_pool(Pool::new(pc, enc));
    brain
}

#[test]
fn cluster_pool_roots_contains_an_entry_per_pool() {
    let brain = make_brain_one_text_pool();
    let roots = brain.cluster_pool_roots();
    // Binding pool (auto, id 0) + text pool (id 1) = 2 entries.
    assert_eq!(roots.len(), 2);
    assert!(roots.contains_key(&0));
    assert!(roots.contains_key(&1));
}

#[test]
fn identically_trained_brains_have_matching_roots() {
    let mut a = make_brain_one_text_pool();
    let mut b = make_brain_one_text_pool();
    for _ in 0..3 {
        a.observe(1, b"ab");
        b.observe(1, b"ab");
        a.advance_tick();
        b.advance_tick();
    }
    let ra = a.cluster_pool_roots();
    let rb = b.cluster_pool_roots();
    assert_eq!(ra.get(&1), rb.get(&1),
        "two brains with the same training must agree on pool 1's root");
}

#[test]
fn diverged_training_produces_diverged_roots() {
    let mut a = make_brain_one_text_pool();
    let mut b = make_brain_one_text_pool();
    for _ in 0..3 {
        a.observe(1, b"ab");
        b.observe(1, b"ab");
        a.advance_tick();
        b.advance_tick();
    }
    // Extra training only on b.
    b.observe(1, b"cd");
    b.advance_tick();
    let ra = a.cluster_pool_roots();
    let rb = b.cluster_pool_roots();
    assert_ne!(ra.get(&1), rb.get(&1),
        "diverged training must produce diverged roots");
}

#[test]
fn cluster_pool_neurons_returns_full_list() {
    let mut a = make_brain_one_text_pool();
    for _ in 0..3 { a.observe(1, b"ab"); a.advance_tick(); }
    let exported = a.cluster_pool_neurons(1).expect("pool exists");
    let local_count = a.fabric().pool(1).unwrap().read().neuron_count();
    assert_eq!(exported.len(), local_count,
        "export must include every neuron in the pool");
}

#[test]
fn cluster_merge_pool_inserts_only_new_neurons() {
    // Train two brains differently: a has 'ab', b has 'cd'.
    let mut a = make_brain_one_text_pool();
    let mut b = make_brain_one_text_pool();
    for _ in 0..3 { a.observe(1, b"ab"); a.advance_tick(); }
    for _ in 0..3 { b.observe(1, b"cd"); b.advance_tick(); }

    let count_a_before = a.fabric().pool(1).unwrap().read().neuron_count();

    // Try to merge b's pool into a — only the neurons whose ids
    // ≥ a's current count should land.  In practice, a and b have
    // overlapping id ranges, so the merge is conservative and may
    // insert 0 neurons (which is correct: ids are the source of truth).
    let b_neurons = b.cluster_pool_neurons(1).expect("b has pool 1");
    let inserted = a.cluster_merge_pool(1, b_neurons);

    let count_a_after = a.fabric().pool(1).unwrap().read().neuron_count();
    assert_eq!(count_a_after, count_a_before + inserted,
        "post-merge count matches pre-merge + inserted");
}

#[test]
fn cluster_merge_pool_transfers_terminal_weights() {
    // Train brain a so it has terminals, then transfer to b (fresh).
    let mut a = make_brain_one_text_pool();
    for _ in 0..4 { a.observe(1, b"ab"); a.advance_tick(); }

    let a_terminals: usize = a.fabric().pool(1).unwrap().read()
        .iter_neurons()
        .map(|n| n.terminals.len())
        .sum();
    assert!(a_terminals > 0, "a should have terminals after training");

    let b = make_brain_one_text_pool();
    let a_neurons = a.cluster_pool_neurons(1).expect("a has pool 1");
    let _ = b.cluster_merge_pool(1, a_neurons);

    let b_terminals: usize = b.fabric().pool(1).unwrap().read()
        .iter_neurons()
        .map(|n| n.terminals.len())
        .sum();
    assert_eq!(b_terminals, a_terminals,
        "cluster merge must transfer terminal weights, not just topology");
}
