//! Stage 3 (position-aware cross-pool binding) tests.
//!
//! When concepts P000 and P010 are observed in pool_a (sharing the
//! same atom set {P, 0, 1} but different member sequence), their
//! cross-pool terminals to pool_b should target DIFFERENT concepts —
//! C000 and C010 respectively.  Atom-level wiring alone can't do
//! this because firing-sets are deduplicated; concept-level
//! cross-pool wiring is what carries the positional discrimination.

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

fn build_two_pool_brain() -> (Brain, u32, u32) {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = u32::MAX; // single-binding focus off
    let mut brain = Brain::new(cfg);
    let mut pa = PoolConfig::defaults("prompt", 1);
    pa.recent_atoms_window         = 4096;
    pa.concept_emergence_threshold = 3;
    pa.max_concept_member_count    = 16;
    pa.decay_rate                  = 0.00001;
    pa.prune_floor                 = 0.0005;
    let mut pb = PoolConfig::defaults("completion", 2);
    pb.recent_atoms_window         = 4096;
    pb.concept_emergence_threshold = 3;
    pb.max_concept_member_count    = 16;
    pb.decay_rate                  = 0.00001;
    pb.prune_floor                 = 0.0005;
    let pool_a = brain.create_pool(pa,
        Box::new(BytePassthroughEncoding { prefix: "p" }));
    let pool_b = brain.create_pool(pb,
        Box::new(BytePassthroughEncoding { prefix: "c" }));
    (brain, pool_a, pool_b)
}

#[test]
fn shared_atom_prompts_grow_concept_to_concept_cross_pool_terminals() {
    // Train (P000, C000) and (P010, C010) pairs.  After enough
    // repetitions, SOME concepts in pool_a emerge that fire only for
    // "P000" and others that fire only for "P010" (the substrate
    // chooses what emerges based on Hebbian co-occurrence).  Those
    // concepts must accumulate concept→concept cross-pool terminals
    // to pool_b concepts that correspondingly fire only for C000 or
    // C010.  This is the position-aware binding property: distinct
    // member sequences ↔ distinct concept neurons ↔ distinct
    // cross-pool terminals.

    let (mut brain, pool_a, pool_b) = build_two_pool_brain();

    for _ in 0..15 {
        brain.observe(pool_a, b"P000");
        brain.observe(pool_b, b"C000");
        brain.advance_tick();
        brain.observe(pool_a, b"P010");
        brain.observe(pool_b, b"C010");
        brain.advance_tick();
    }

    // Count concepts in pool_a that have at least one concept→concept
    // cross-pool terminal to a pool_b concept.  If concept-level
    // cross-pool wiring is happening, this count should be >0.
    let pa_arc = brain.fabric().pool(pool_a).unwrap();
    let pa_read = pa_arc.read();
    let pb_arc = brain.fabric().pool(pool_b).unwrap();
    let pb_read = pb_arc.read();

    let pool_b_concept_ids: ahash::AHashSet<_> = pb_read.iter_neurons()
        .filter(|n| !n.is_atom())
        .map(|n| n.id)
        .collect();
    assert!(!pool_b_concept_ids.is_empty(),
        "pool_b must have emerged at least one concept after 15 reps each");

    let cross_pool_concept_terminals = pa_read.iter_neurons()
        .filter(|n| !n.is_atom())
        .flat_map(|n| n.terminals.iter())
        .filter(|t| t.target.pool == pool_b
                 && pool_b_concept_ids.contains(&t.target.neuron)
                 && t.weight > 0.0)
        .count();

    assert!(cross_pool_concept_terminals > 0,
        "Stage 3 concept-level cross-pool wiring must produce at least one \
         concept→concept terminal across pools after training; got 0");
}

#[test]
fn shared_atom_prompts_produce_differentiated_pool_b_activation() {
    // Discriminative-signal test: after training (P000,C000) and
    // (P010,C010), the *activation patterns* in pool_b must differ
    // when querying P000 vs P010 — at least one concept must have a
    // meaningfully different activation between the two queries.
    // This is the strict property Stage 3 delivers: distinct
    // concept-level cross-pool terminals carry distinct signals,
    // even if integrate's density-based winner-take-all selection
    // can still pick the same shared shorter concept for both.

    let (mut brain, pool_a, pool_b) = build_two_pool_brain();

    for _ in 0..15 {
        brain.observe(pool_a, b"P000");
        brain.observe(pool_b, b"C000");
        brain.advance_tick();
        brain.observe(pool_a, b"P010");
        brain.observe(pool_b, b"C010");
        brain.advance_tick();
    }

    brain.observe(pool_a, b"P000");
    let prop_000 = brain.fabric().propagate(pool_a);
    let empty = ahash::AHashMap::new();
    let act_000 = prop_000.get(&pool_b).unwrap_or(&empty).clone();
    brain.observe(pool_a, b"P010");
    let prop_010 = brain.fabric().propagate(pool_a);
    let act_010 = prop_010.get(&pool_b).unwrap_or(&empty).clone();

    // Find at least one neuron whose activation differs meaningfully.
    let mut max_diff = 0.0_f32;
    let all_ids: ahash::AHashSet<_> =
        act_000.keys().copied().chain(act_010.keys().copied()).collect();
    for nid in all_ids {
        let a = act_000.get(&nid).copied().unwrap_or(0.0);
        let b = act_010.get(&nid).copied().unwrap_or(0.0);
        let d = (a - b).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(max_diff > 0.01,
        "Stage 3 must produce SOME activation difference in pool_b between \
         P000 and P010 queries; max_diff={}", max_diff);
}
