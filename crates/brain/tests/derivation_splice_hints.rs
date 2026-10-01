//! What does a derivation SPEND, and does the splice-point cache cut it
//! without changing what it answers?
//!
//! `Brain::derivation_cut_hints` already made LOCATING the taught
//! sub-question one question instead of `~2k`. It left the other half alone:
//! with the cut `k` known, the answer is still tried at every `j` in `0..=k`.
//! Measured on the scorecard at HEAD before this file existed
//! (`logs/scorecard-gate.json`, scale 16): 864 derivations, 12,699 probes,
//! **14.7 probes per attempt** against a budget of 32 — so the splice scan,
//! not the cut scan, was what a 3-hop family could not fit inside.
//!
//! Two things are asserted here, because a cache that is cheaper and answers
//! differently is not a cost fix:
//!
//! 1. **It is a cache.** The second question of a shape costs strictly fewer
//!    probes than the first, and the hint table is bounded.
//! 2. **It changes no answer.** Every derivation that was right without the
//!    hint is right with it, and the hint is only ever consulted as the FIRST
//!    span of a scan that still tries every other one.
//!
//! The probe counts are read off `derive_by_substitution_profiled`'s own
//! return value rather than computed from `n(n+1)/2` — see `DerivationStats`
//! for why this file does not quote a formula.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig, DERIVATION_SPLICE_HINTS,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const MAX_PROBES: usize = 4096;

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

fn teach(brain: &mut Brain, question: &str, answer: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, question.as_bytes().to_vec()),
        (ANSWER_POOL, answer.as_bytes().to_vec()),
    ]);
}

/// One shape, repeated: `"r{n} lamp on material"` is never taught, and is the
/// composition of two questions that are. Every room has the same question
/// LENGTH, which is what makes the cut and the splice a property of the shape
/// rather than of the room.
fn teach_on_material(brain: &mut Brain, rooms: u32) {
    for room in 0..rooms {
        teach(brain, &format!("r{room:03} lamp on"), "desk");
        teach(brain, &format!("r{room:03} desk material"), "oak");
    }
}

#[test]
fn the_splice_point_is_remembered_and_the_second_question_of_a_shape_costs_less() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    teach_on_material(&mut brain, ROOMS);

    let mut costs: Vec<usize> = Vec::new();
    let mut correct = 0u32;
    for room in 0..ROOMS {
        let q = format!("r{room:03} lamp on material");
        let (answer, cost) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 2, MAX_PROBES);
        if answer.as_deref() == Some(b"oak".as_slice()) {
            correct += 1;
        }
        costs.push(cost);
    }
    eprintln!("probes per derivation, in order: {costs:?}");
    eprintln!("derived correctly: {correct} of {ROOMS}");
    eprintln!("splice hints learned: {:?}", brain.derivation_splice_hints());

    // 2 above: the cache must not cost an answer. This is the assertion that
    // makes the cost numbers below worth reading.
    assert_eq!(
        correct, ROOMS,
        "the splice hint changed what the derivation answers, which is not a cost fix"
    );

    // 1 above: it is a cache. The first question of the shape pays the full
    // scan; every later one pays the hint.
    assert!(
        !brain.derivation_splice_hints().is_empty(),
        "nothing was remembered, so the hint cannot be what made the later questions cheap"
    );
    let first = costs[0];
    let rest_max = *costs[1..].iter().max().expect("16 rooms");
    assert!(
        rest_max < first,
        "no question after the first was cheaper than it ({rest_max} vs {first}), \
         so the splice scan is still being paid per question"
    );

    // Bounded, like `derivation_cut_hints`: this is a cache of the current
    // question distribution, not a growing index.
    assert!(
        brain.derivation_splice_hints().len() <= DERIVATION_SPLICE_HINTS,
        "the hint table is unbounded"
    );
}

/// The cost cut has to survive the budget a production answer path actually
/// gives it. `DEFAULT_DERIVATION_PROBE_BUDGET` is 32, and a 3-hop derivation
/// did not fit inside it: at HEAD the scorecard's `next_on_material` spent
/// 3,694 probes over 128 attempts (28.9 each) to answer 123 and get 6 right.
#[test]
fn a_two_hop_derivation_fits_inside_the_production_budget_after_one_warm_up() {
    const ROOMS: u32 = 8;
    let mut brain = subject();
    teach_on_material(&mut brain, ROOMS);

    let budget = w1z4rd_brain::DEFAULT_DERIVATION_PROBE_BUDGET;
    let mut answered_within_budget = 0u32;
    let mut costs: Vec<usize> = Vec::new();
    for room in 0..ROOMS {
        let q = format!("r{room:03} lamp on material");
        let (answer, cost) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 2, budget);
        costs.push(cost);
        if answer.as_deref() == Some(b"oak".as_slice()) {
            answered_within_budget += 1;
        }
    }
    eprintln!("probes per derivation at budget {budget}: {costs:?}");
    assert_eq!(
        answered_within_budget, ROOMS,
        "a 2-hop derivation of a repeated shape does not fit the production budget"
    );
}
