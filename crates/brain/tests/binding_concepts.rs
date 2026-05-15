//! Binding-concept emergence tests per [`ARCHITECTURE.md`] §4.A.
//!
//! A binding concept is a neuron whose members span multiple pools.
//! It emerges when the same multi-pool firing pattern recurs N times
//! within the moment-history window.  Spec §1.7 (one mechanism, two
//! scopes): the same Hebbian-style co-occurrence rule that builds
//! within-pool hierarchies builds cross-pool bindings.

use w1z4rd_brain::{Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

#[test]
fn multi_pool_co_firing_promotes_a_binding_concept_with_cross_pool_members() {
    // Setup: brain with default binding-emergence threshold of 3.
    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text",  1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    let binding_pool = brain.binding_pool_id();

    let baseline = brain.stats();
    assert_eq!(baseline.total_binding, 0, "no bindings before observation");

    // Observe the SAME multi-pool firing 3 times to cross the
    // emergence threshold.  This is "X in pool A AND Y in pool B at
    // the same tick", repeated.  The fingerprint is sorted (pool,
    // neuron) pairs so the same pattern recurs identically each time.
    for _ in 0..3 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }

    let after = brain.stats();
    assert!(after.total_binding >= 1,
        "3 reps of the same multi-pool firing must promote 1+ binding; got {}",
        after.total_binding);

    // Inspect the binding concept directly: its members must reference
    // neurons in BOTH pool_a and pool_b (that's what makes it a
    // binding concept rather than a within-pool hierarchy).
    let bp = brain.fabric().pool(binding_pool).expect("binding pool exists");
    let bp = bp.read();
    let binding = bp.iter_neurons()
        .find(|n| !n.is_atom())
        .expect("binding concept neuron must exist");
    let pools_in_members: std::collections::HashSet<u32> =
        binding.members.iter().map(|m| m.pool).collect();
    assert!(pools_in_members.contains(&pool_a),
        "binding concept must have a member in pool_a");
    assert!(pools_in_members.contains(&pool_b),
        "binding concept must have a member in pool_b");
    assert!(pools_in_members.len() >= 2,
        "binding concept must span at least 2 pools; got {:?}", pools_in_members);
}

#[test]
fn single_pool_firing_does_not_promote_a_binding_concept() {
    // Bindings require ≥2 pools by definition.  Observing only one
    // pool — even repeatedly — should NEVER produce a binding concept.
    // Within-pool concept emergence is the appropriate rule for that
    // case, and it lives in Pool, not Brain.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );

    for _ in 0..10 {
        brain.observe(pool_a, b"X");
        brain.advance_tick();
    }

    let stats = brain.stats();
    assert_eq!(stats.total_binding, 0,
        "single-pool firing must never produce binding concepts; got {}",
        stats.total_binding);
}

#[test]
fn binding_concept_is_idempotent_under_repeated_co_firing() {
    // Spec §4.A: emergence is automatic but must be idempotent.  Once
    // a binding concept is promoted, observing the same pattern more
    // times must NOT keep creating duplicate binding concepts.  This
    // is what prevents the "O(N²) concept explosion" failure mode the
    // legacy fabric had.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text",  1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );

    // First 3 reps: threshold crossed, binding promoted.
    for _ in 0..3 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    let after_first_round = brain.stats().total_binding;
    assert!(after_first_round >= 1, "binding should have been promoted");

    // 20 more reps of the same pattern: count must not grow.
    for _ in 0..20 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    let after_many_more = brain.stats().total_binding;
    assert_eq!(after_first_round, after_many_more,
        "repeated co-firing of the same pattern must NOT keep creating bindings (was {}, now {})",
        after_first_round, after_many_more);
}

#[test]
fn distinct_multi_pool_patterns_produce_distinct_binding_concepts() {
    // Two genuinely different multi-pool patterns should produce two
    // different binding concepts.  This shows the binding layer can
    // discriminate — it's not just "any cross-pool co-fire produces
    // the same binding."

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text",  1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );

    // Pattern 1: X+Y co-fire, 3 times.
    for _ in 0..3 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    // Pattern 2: P+Q co-fire (different atoms entirely), 3 times.
    for _ in 0..3 {
        brain.observe(pool_a, b"P");
        brain.observe(pool_b, b"Q");
        brain.advance_tick();
    }

    let stats = brain.stats();
    assert!(stats.total_binding >= 2,
        "two distinct multi-pool patterns must produce ≥2 bindings; got {}",
        stats.total_binding);
}
