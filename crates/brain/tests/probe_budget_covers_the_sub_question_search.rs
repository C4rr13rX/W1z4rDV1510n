//! The shipped probe budget must be able to FINISH the sub-question search,
//! because a budget that cannot is spent entirely on not finding the cut.
//!
//! Measured 2026-10-01 (Iris, pass 19) on the scorecard at scale 64, one
//! binary, the budget set per run through `--derivation-budget`:
//!
//! | budget | integration | next_on_material | probes/attempt | peak |
//! |--------|-------------|------------------|----------------|------|
//! |     32 |      77.286 |          113/512 |          10.09 | 39.8 MB |
//! |     64 |      84.259 |          352/512 |          14.88 | 39.7 MB |
//! |    128 |      84.259 |          352/512 |          21.49 | 40.5 MB |
//!
//! The arithmetic behind that step. `derive_by_substitution_profiled` locates
//! the taught sub-question with `k` outer over `1..n` and `t` inner over
//! `0..=1`, so a cold cut costs up to `2(n - 1)` questions BEFORE the splice
//! search has asked anything at all. The scorecard's 3-hop questions are 27
//! bytes (`lit_mean` 27.9, against 23.0 and 21.6 for its 2-hop families), so
//! that scan alone is up to 52 — above a ceiling of 32, and a search that
//! cannot reach its own second phase answers nothing however many times it is
//! run.
//!
//! So this file pins TWO things, and deliberately not "64 beats 32" — an
//! assertion that one setting beats another goes red when the search improves,
//! which is the exact shape that made pass 16's gate red before and after.
//!
//!   1. ARITHMETIC: the shipped budget covers the cold sub-question search for
//!      the longest question this world asks. Absolute, and it is the claim the
//!      budget rise rests on.
//!   2. BEHAVIOUR: the shipped budget loses no answer the uncapped search
//!      finds. Absolute, already the contract in
//!      `three_hop_derivation_fits_the_budget.rs` — asserted here at the
//!      question LENGTH where the cap actually bites, which is the regime that
//!      file's 16-room world does not reach.
//!
//! The 32 arm is run and PRINTED rather than asserted, so the floor stays
//! visible in the output of a passing run without the test taking a position
//! on which setting wins.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    DEFAULT_DERIVATION_PROBE_BUDGET,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const ROOMS: u32 = 16;
/// Above any budget this test compares, so it measures the search and not the
/// cap. `three_hop_derivation_fits_the_budget.rs` uses the same value.
const UNCAPPED: usize = 4096;
/// The ceiling that shipped until 2026-10-01, kept as the printed contrast.
const PREVIOUS_BUDGET: usize = 32;

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

/// The scorecard's three relations and its 3-hop question, so the question
/// LENGTH this measures is the product's and not one chosen to suit a budget.
fn teach_world(brain: &mut Brain) -> Vec<(String, String)> {
    let material = |r: u32| format!("m{:03}", r % 7);
    let mut asked = Vec::new();
    for r in 0..ROOMS {
        let room = format!("r{r:03}");
        let next = format!("r{:03}", (r + 1) % ROOMS);
        teach(brain, &format!("{room} next?"), &next);
        teach(brain, &format!("{room} lamp on?"), "desk");
        teach(brain, &format!("{room} desk material?"), &material(r));
    }
    for r in 0..ROOMS {
        asked.push((
            format!("r{r:03} next lamp on material?"),
            material((r + 1) % ROOMS),
        ));
    }
    asked
}

/// How many of `asked` derive CORRECTLY under `budget`, on a brain of its own
/// so one arm's warm cut cache cannot be another arm's result.
fn answered_under(budget: usize) -> (u32, usize, usize) {
    let mut brain = subject();
    let asked = teach_world(&mut brain);
    let mut right = 0u32;
    let mut probes_total = 0usize;
    for (question, want) in &asked {
        let (answer, probes) = brain.derive_by_substitution_profiled(
            QUERY_POOL,
            ANSWER_POOL,
            question.as_bytes(),
            3,
            budget,
        );
        probes_total += probes;
        if answer.as_deref() == Some(want.as_bytes()) {
            right += 1;
        }
    }
    (right, probes_total, asked.len())
}

#[test]
fn the_shipped_budget_covers_a_cold_sub_question_search() {
    let mut brain = subject();
    let asked = teach_world(&mut brain);
    let longest = asked.iter().map(|(q, _)| q.len()).max().expect("asked is non-empty");

    // `k` over `1..n` with `t` over `0..=1` is at most `2(n - 1)` questions,
    // plus the one base probe that opens the hop, plus at least one rewrite —
    // or the hop spends everything proving nothing and the derivation is
    // structurally unable to answer, whatever the corpus holds.
    let floor = 2 * (longest - 1) + 2;
    eprintln!(
        "longest 3-hop question {longest} bytes -> cold sub-question search needs \
         <= {} probes, so a hop needs a ceiling of at least {floor}; shipped \
         {DEFAULT_DERIVATION_PROBE_BUDGET}",
        2 * (longest - 1),
    );
    assert!(
        DEFAULT_DERIVATION_PROBE_BUDGET >= floor,
        "the shipped budget {DEFAULT_DERIVATION_PROBE_BUDGET} is below {floor}, the cost of \
         the deletion scan `k` in 1..{longest} with `t` in 0..=1 plus a base probe plus one \
         rewrite -- so a cold 3-hop hop cannot reach its splice search at all"
    );
}

#[test]
fn the_shipped_budget_turns_no_answer_into_a_silence() {
    let (uncapped_right, uncapped_probes, n) = answered_under(UNCAPPED);
    let (shipped_right, shipped_probes, _) = answered_under(DEFAULT_DERIVATION_PROBE_BUDGET);
    let (previous_right, previous_probes, _) = answered_under(PREVIOUS_BUDGET);

    // Printed, not asserted. A passing run still shows where the floor is, and
    // nothing here claims one setting beats another.
    eprintln!(
        "of {n} 3-hop questions: uncapped ({UNCAPPED}) {uncapped_right} right in \
         {uncapped_probes} probes; shipped ({DEFAULT_DERIVATION_PROBE_BUDGET}) {shipped_right} \
         right in {shipped_probes}; previous ({PREVIOUS_BUDGET}) {previous_right} right in \
         {previous_probes}"
    );

    // The uncapped arm measures the SEARCH. If it answers nothing, the budget
    // is not what this test is looking at and every number above is unreadable.
    assert!(
        uncapped_right > 0,
        "the uncapped search derived nothing of {n}, so this file is measuring a broken \
         derivation and not a budget"
    );
    assert_eq!(
        shipped_right, uncapped_right,
        "the shipped budget {DEFAULT_DERIVATION_PROBE_BUDGET} answers {shipped_right} of {n} \
         where the uncapped search answers {uncapped_right} -- the cap is turning answers into \
         silences, which is the one thing a ceiling may not do"
    );
}
