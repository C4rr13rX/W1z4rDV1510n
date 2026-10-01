//! Which arm does an integration probe leave `integrate_autonomous_tuned`
//! through?
//!
//! `Eem::chain_explore` has exactly ONE caller in this crate --
//! `Brain::integrate_autonomous_tuned` -- and three unconditional early
//! returns sit ahead of it:
//!
//!   0. the OOV gate: `best_binding_match_v2(query_pool).precision <
//!      binding_match_threshold` returns `answer: None`,
//!      `outside_grounding: true`.
//!   1. the legacy accept: if `self.integrate(..)` returned a non-empty answer
//!      at or above `fabric_confidence_threshold`, that answer is returned.
//!      The source comment is explicit -- "accept it instead of letting
//!      chain_explore re-decide".
//!   2. no query pool, or `seed.is_empty()`.
//!
//! So "integration is 0%" has at least four mechanisms behind it and they need
//! different fixes. CLAUDE.md's standing lesson applies exactly: the answer
//! path is an if/else chain, an earlier arm that matches ends it, and the
//! branch that ran must be READ rather than inferred.
//!
//! This file measures the arm from OUTSIDE, with no change to `brain.rs`:
//! `best_binding_match_v2` is public and is the precise quantity arm 0 gates
//! on, so the test asks it directly instead of parsing a diagnostic string.
//!
//! The numbers are PRINTED, not asserted. Asserting a distribution would tune
//! the test to this probe set, which is the one thing a probe must never do.
//! The assertions are only the invariants that hold whatever the distribution
//! is: every probe is grounded-or-not, and a probe that clears the gate is one
//! the explorer was actually given a chance at.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// The default `binding_match_threshold` every `integrate_autonomous` caller
/// gets, and the value `crates/node/src/api.rs` passes explicitly.
const BINDING_MATCH_THRESHOLD: f32 = 0.70;
/// What the node passes as `fabric_confidence_threshold` on its research-loop
/// answer path, commented there as "anything > random".
const NODE_FABRIC_CONFIDENCE_THRESHOLD: f32 = 0.10;

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
/// that furniture has a material. Neither fact answers "what is the lamp's
/// material" on its own; the composition of the two does.
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
fn the_integration_probe_arm_census() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    train_chainable_world(&mut brain, ROOMS);

    // Sanity: the TRAINED halves are recalled, so the world is real and any
    // zero below is about integration and not about an empty brain.
    let mut trained_recalled = 0;
    for room in 0..ROOMS {
        brain.observe_read_only(QUERY_POOL, format!("r{room:03} lamp on").as_bytes());
        if brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL).as_deref() == Some(&b"desk"[..])
        {
            trained_recalled += 1;
        }
    }
    assert_eq!(
        trained_recalled, ROOMS,
        "the trained half of the world is not recalled, so nothing below is about integration"
    );

    // Now the integration probes: never trained, true in the world, and
    // answerable only by composing the two facts.
    let mut cleared_gate = 0u32;
    let mut blocked_at_oov_gate = 0u32;
    let mut answered = 0u32;
    let mut answered_correctly = 0u32;
    let mut answered_wrongly = 0u32;
    let mut precisions: Vec<f32> = Vec::new();

    for room in 0..ROOMS {
        let probe = format!("r{room:03} lamp on material");
        brain.observe_read_only(QUERY_POOL, probe.as_bytes());

        // The exact quantity arm 0 gates on, asked directly.
        let bm = brain.best_binding_match_v2(QUERY_POOL);
        precisions.push(bm.precision);
        if bm.precision < BINDING_MATCH_THRESHOLD {
            blocked_at_oov_gate += 1;
        } else {
            cleared_gate += 1;
        }

        let res = brain.integrate_autonomous_tuned(
            QUERY_POOL,
            ANSWER_POOL,
            NODE_FABRIC_CONFIDENCE_THRESHOLD,
            3,
            16,
            BINDING_MATCH_THRESHOLD,
        );
        match res.answer.as_deref() {
            Some(b) if !b.is_empty() => {
                answered += 1;
                if b == b"oak" {
                    answered_correctly += 1;
                } else {
                    answered_wrongly += 1;
                }
            }
            _ => {}
        }
    }

    let mean_precision = precisions.iter().sum::<f32>() / precisions.len() as f32;
    let min_precision = precisions.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_precision = precisions.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // PRINTED, never asserted. `cargo test -- --nocapture` to read it.
    eprintln!("integration arm census over {ROOMS} probes");
    eprintln!("  binding precision   mean {mean_precision:.3}  min {min_precision:.3}  max {max_precision:.3}  (gate is {BINDING_MATCH_THRESHOLD})");
    eprintln!("  blocked at arm 0 (OOV gate)        {blocked_at_oov_gate}");
    eprintln!("  cleared arm 0                      {cleared_gate}");
    eprintln!("  returned a non-empty answer        {answered}");
    eprintln!("    of which CORRECT  (oak)          {answered_correctly}");
    eprintln!("    of which WRONG                   {answered_wrongly}");

    // The invariants, which hold whatever the distribution turns out to be.
    assert_eq!(
        blocked_at_oov_gate + cleared_gate,
        ROOMS,
        "every probe must either clear the OOV gate or be blocked by it"
    );
    assert_eq!(
        answered_correctly + answered_wrongly,
        answered,
        "an answered probe is either correct or wrong"
    );
    // A probe blocked at arm 0 cannot have been answered, because arm 0
    // returns `answer: None`. If this ever fails, the gate is not the single
    // source of truth its own comment claims to be.
    assert!(
        answered <= cleared_gate,
        "answered {answered} probes but only {cleared_gate} cleared the OOV gate, \
         so arm 0 is not the only grounding decision"
    );
}

/// Does ordinary training put anything in the graph the chain explorer walks?
///
/// Measured by the census above: composite probes clear the OOV gate at
/// precision EXACTLY 1.000 (16 of 16), the legacy accept does not fire, and
/// zero probes come back with any answer. So the probes fall past both arms
/// and the question becomes what `chain_explore` had to work with.
///
/// `chain_explore` walks `Eem` grounded facts from the firing query-pool
/// seeds. Two existing suites prove the machinery: `critical_thinking.rs`
/// walks `chain_explore` to a target-pool ref and `multi_fact.rs` asserts
/// `composition_used.len() >= 2` for a composed answer.
///
/// Neither registers a fact by hand -- `grep -rn register_fact crates/` finds
/// exactly three hits: the definition in `eem.rs`, ONE production caller in
/// `brain.rs`, and this comment. Those suites populate the graph by
/// `observe` + `advance_tick` repeated (`multi_fact.rs:39-41`, `:70-75`), and
/// the production caller sits inside `Brain::register_fingerprint`, reached
/// only when a binding fingerprint crosses `consolidated_thr`. So **EEM facts
/// come exclusively from binding consolidation during `advance_tick`**, and
/// `pretrain_binding_episode` never goes through `register_fingerprint`.
///
/// That is the same bypass `docs/RAM_GOAL.md` point 1 already records for a
/// different consequence -- "the ledger is empty after training:
/// `pretrain_binding_episode` does not go through emergence at all" -- and it
/// is why the scorecard's shape is recall exactly 100% beside integration
/// exactly 0.0: the two come from different paths and the fast bulk path only
/// exercises one. The two suites above pass BECAUSE they take the slow
/// learning path, which is how 378 green tests never caught this.
///
/// Every fix aimed at the answer-branch arms would therefore have been inert,
/// which is the single most expensive recurring mistake in this repository: a
/// guard keyed on evidence the system cannot produce.
#[test]
fn the_eem_graph_population_census() {
    const ROOMS: u32 = 16;
    let mut brain = subject();

    let facts_before = brain.eem().fact_count();
    train_chainable_world(&mut brain, ROOMS);
    let facts_after = brain.eem().fact_count();

    // How many of a probe's own firing seeds have ANY fact attached? That is
    // precisely the input `chain_explore` receives.
    brain.observe_read_only(QUERY_POOL, b"r000 lamp on material");
    let seeds: Vec<u32> = brain
        .fabric()
        .pool(QUERY_POOL)
        .map(|p| p.read().currently_firing().into_iter().collect())
        .unwrap_or_default();
    let seeds_with_facts = seeds
        .iter()
        .filter(|&&nid| !brain.eem().facts_involving(QUERY_POOL, nid).is_empty())
        .count();

    eprintln!("EEM graph population census");
    eprintln!("  grounded facts before training      {facts_before}");
    eprintln!("  grounded facts after {ROOMS} rooms x 2 {facts_after}");
    eprintln!("  equations {} variables {} motifs {}", brain.eem().equation_count(), brain.eem().variable_count(), brain.eem().motif_count());
    eprintln!("  probe firing seeds                  {}", seeds.len());
    eprintln!("  of those, seeds with >=1 fact       {seeds_with_facts}");

    // The invariants. Training cannot REMOVE facts, and a seed set must exist
    // at all -- an empty seed set is its own arm (`seed.is_empty()` returns
    // before `chain_explore`) and would make the census above unreadable.
    assert!(
        facts_after >= facts_before,
        "training removed grounded facts: {facts_before} -> {facts_after}"
    );
    assert!(
        !seeds.is_empty(),
        "nothing was firing after observing the probe, so the empty-seed arm \
         returns before chain_explore and no chaining conclusion is available"
    );
}
