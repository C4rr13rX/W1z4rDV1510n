//! What does TRAINING put into the composition engine? Measured: nothing.
//!
//! `crates/brain` contains a working, general forward chainer --
//! `TransientWorkspace::resolve` joins premises by typed variable binding and
//! carries provenance, and `tests/logical_workspace.rs` composes dozens of
//! four-topic crossings deterministically with it. So "integration is 0%" is
//! not a missing mechanism. It is an unfed one, and this file measures that
//! from outside, with no change to any source file.
//!
//! The two inputs the chainer needs, and who writes them (measured
//! 2026-10-01 by grep over `crates/*/src`):
//!
//!   * `Eem::composition_rules` -- `register_composition_rule` has exactly one
//!     production caller, `crates/node/src/brain_api.rs:1274`, which
//!     deserialises the rule from an HTTP request body (`req.get("rule")`).
//!     There is not one `CompositionRule { .. }` construction anywhere in
//!     `crates/*/src`; the single grep hit is the struct definition.
//!   * `Eem::semantic_relations` -- `register_semantic_relation` has one
//!     non-self caller, `brain_api.rs:1265`, also from a request body;
//!     `consolidate_semantic_frame` has one caller, `brain_api.rs:1326`, also a
//!     handler.
//!
//! `resolve` breaks out of its round loop the moment a round derives nothing,
//! so with zero rules it derives nothing in one round whatever the facts are.
//! That is the asserted invariant below: it is the half of the finding that is
//! structural rather than a count, and it stays true after someone fixes the
//! feed, which is why it is the assertion and the counts are only printed.
//!
//! The counts are PRINTED, not asserted at 0. A test that fails the moment the
//! gap is closed turns the gate red for whoever closes it, and an assertion on
//! today's distribution is a test tuned to today's brain.
//!
//! Standing caution from pass 6, so it is not re-learned the expensive way:
//! populating the OTHER graph -- `Eem::register_fact`, which `chain_explore`
//! walks -- from the training path DID raise its fact count and still measured
//! 0.0 % integration at every scale, for 5.3 MB at scale 64. A non-zero input
//! count is necessary and demonstrably not sufficient; the thing to measure is
//! a DERIVED relation whose provenance names two trained facts.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// The same two-pool byte-passthrough brain the scorecard builds, so what is
/// measured here is what the scorecard's scales measure.
fn subject() -> Brain {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);
    for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
        let mut pc = PoolConfig::defaults(name, id);
        pc.recent_atoms_window = 2048;
        pc.concept_emergence_threshold = 2;
        pc.max_concept_member_count = 64;
        pc.decay_rate = 0.0001;
        pc.prune_floor = 0.005;
        brain.create_pool(pc, Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>);
    }
    brain
}

/// Two facts per room that CHAIN: the lamp names a piece of furniture, and
/// that furniture has a material. Neither answers "the lamp's material"; the
/// composition of the two does. Trained through `pretrain_binding_episode`,
/// which is the call the scorecard's `train` makes.
fn train_chainable_world(brain: &mut Brain, rooms: u32) {
    for room in 0..rooms {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{room:03} lamp on").into_bytes()),
            (ANSWER_POOL, b"desk".to_vec()),
        ]);
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{room:03} desk material").into_bytes()),
            (ANSWER_POOL, b"oak".to_vec()),
        ]);
    }
}

#[test]
fn training_feeds_the_composition_engine_nothing() {
    const ROOMS: u32 = 16;
    let mut brain = subject();

    let rules_before = brain.eem().composition_rule_count();
    let relations_before = brain.eem().semantic_relation_count();
    let facts_before = brain.eem_fact_count();

    train_chainable_world(&mut brain, ROOMS);

    let rules_after = brain.eem().composition_rule_count();
    let relations_after = brain.eem().semantic_relation_count();
    let facts_after = brain.eem_fact_count();

    // Sanity: the trained halves ARE recalled, so every zero below is about the
    // composition engine and not about an empty brain.
    let mut trained_recalled = 0;
    for room in 0..ROOMS {
        brain.observe_read_only(QUERY_POOL, format!("r{room:03} lamp on").as_bytes());
        if brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL).as_deref()
            == Some(&b"desk"[..])
        {
            trained_recalled += 1;
        }
    }
    assert_eq!(
        trained_recalled, ROOMS,
        "the trained half of the world is not recalled, so nothing below is about integration"
    );

    let workspace = brain.eem().compose_transient(8);
    let derived = workspace.facts().len().saturating_sub(relations_after);

    println!(
        "composition inputs after training {ROOMS} chainable rooms ({} episodes):\n  \
         composition_rules      {rules_before} -> {rules_after}\n  \
         semantic_relations     {relations_before} -> {relations_after}\n  \
         eem grounded facts     {facts_before} -> {facts_after}\n  \
         compose_transient(8)   {} facts in workspace, {derived} derived\n  \
         trained recall         {trained_recalled}/{ROOMS}",
        ROOMS * 2,
        workspace.facts().len(),
    );

    // The structural invariant, true before and after anyone fixes the feed:
    // `resolve` can only derive through a rule, so an empty rule set derives
    // nothing however many facts are registered.
    if rules_after == 0 {
        assert_eq!(
            derived, 0,
            "no composition rule is registered, so the workspace cannot have derived anything"
        );
        assert_eq!(
            workspace.facts().len(),
            relations_after,
            "with no rules the workspace must be exactly the registered relations"
        );
    } else {
        // Someone has fed the engine. Then the engine must be the thing that
        // is measured, not the feed: a derived fact has to name its sources.
        assert!(
            workspace.facts().len() >= relations_after,
            "resolve must never lose a registered relation"
        );
    }
}
