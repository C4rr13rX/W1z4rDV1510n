//! Is the derivation's cost MEASURED, and does the measurement tell starvation
//! apart from refusal?
//!
//! Criterion 2 of `[a7fb084f]` asks for "the cost per derived answer named:
//! probes asked per derivation at 2 and 3 hops, measured by
//! `derive_by_substitution_profiled` rather than from `n(n+1)/2`". That return
//! value existed and every caller discarded it: `Brain::integrate_autonomous`
//! put it in a `tracing::debug!` nothing reads, and
//! `derived_by_substitution_reply` (the node's half) never asked for it. So the
//! scorecard could publish `integration_pct 0.0` for a family without being
//! able to say whether the mechanism ran out of budget or ran and declined —
//! the two call for opposite repairs (raise the ceiling vs. reach further), and
//! this repository's standing lesson is that a zero from a path with no
//! opportunity to execute is not evidence about the path.
//!
//! `Brain::derivation_stats` is that readout. Three things are asserted here
//! and each one is a way the counter could be wrong while still looking right:
//!
//! 1. The counter's probe total EQUALS the sum of what the profiled calls
//!    returned. A counter that merely rises is not a measurement of the same
//!    quantity the criterion names.
//! 2. A call that spends its whole budget and returns nothing scores
//!    `budget_exhausted`, and a call that declines with budget left does NOT.
//!    This is the only distinction that says whether a larger budget would
//!    convert a miss.
//! 3. `derive_by_substitution` — the entry point the node's answer route uses
//!    (`crates/node/src/brain_api.rs`, `derived_by_substitution_reply`) — feeds
//!    the same counter as the profiled one the scorecard measures. If it did
//!    not, the product's cost would be unmeasured while the scorecard's was
//!    reported, which is exactly the parity rule this milestone exists for.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    DEFAULT_DERIVATION_PROBE_BUDGET,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const ROOMS: u32 = 16;

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

/// The scorecard's shape: three trained relations, every question terminated
/// with `?` so the taught sub-question is a DELETION and not a prefix.
fn teach_world(brain: &mut Brain) {
    for r in 0..ROOMS {
        let room = format!("r{r:03}");
        let next = format!("r{:03}", (r + 1) % ROOMS);
        teach(brain, &format!("{room} next?"), &next);
        teach(brain, &format!("{room} lamp on?"), "desk");
        teach(brain, &format!("{room} desk material?"), &format!("m{:03}", r % 7));
    }
}

#[test]
fn the_probe_counter_equals_what_the_profiled_calls_returned() {
    let mut brain = subject();
    teach_world(&mut brain);

    let cold = brain.derivation_stats();
    assert_eq!(cold.attempts, 0, "a brain that has derived nothing must say so");
    assert_eq!(cold.probes, 0);
    assert_eq!(
        cold.probes_per_attempt(), None,
        "zero probes over zero attempts must be None, not 0.0 -- a 0.0 reads as 'free'"
    );

    // 2 hops: `"rNNN lamp on material?"` cuts to `"rNNN lamp on?"` -> `desk`,
    // splices to `"rNNN desk material?"`, which is taught.
    let mut returned = 0usize;
    let mut answered_2h = 0usize;
    for r in 0..ROOMS {
        let q = format!("r{r:03} lamp on material?");
        let (answer, probes) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 4096);
        returned += probes;
        answered_2h += usize::from(answer.as_ref().map_or(false, |a| !a.is_empty()));
    }
    let two_hop = brain.derivation_stats();
    assert_eq!(
        two_hop.attempts, ROOMS as usize,
        "every call past the guard must be counted as an attempt"
    );
    assert_eq!(
        two_hop.probes, returned,
        "the counter must measure the SAME quantity the profiled call returns"
    );
    assert_eq!(two_hop.answered, answered_2h, "answered must count non-empty answers");
    assert!(answered_2h > 0, "a 2-hop derivation must answer at all, or the cost below is of nothing");

    // 3 hops, on the same brain so the cut cache is warm exactly as it is on
    // the scorecard's second family onward. Reported, not asserted against a
    // threshold: the number is the deliverable.
    let before_3h = brain.derivation_stats();
    for r in 0..ROOMS {
        let q = format!("r{r:03} next lamp on material?");
        let _ = brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 4096);
    }
    let after_3h = brain.derivation_stats();
    let p2 = two_hop.probes_per_answer();
    let p3 = (after_3h.answered > before_3h.answered).then(|| {
        (after_3h.probes - before_3h.probes) as f64
            / (after_3h.answered - before_3h.answered) as f64
    });
    eprintln!(
        "probes per ANSWER: 2 hops {p2:?} over {} answers; 3 hops {p3:?} over {} answers; \
         shipped budget {DEFAULT_DERIVATION_PROBE_BUDGET}",
        two_hop.answered,
        after_3h.answered - before_3h.answered,
    );
    assert!(p2.is_some(), "2-hop cost per answer must be reportable");
    assert_eq!(
        after_3h.attempts,
        2 * ROOMS as usize,
        "the 3-hop pass must also be counted"
    );
}

#[test]
fn starvation_is_told_apart_from_refusal() {
    let mut brain = subject();
    teach_world(&mut brain);

    // A question the brain was TAUGHT needs no derivation: the base probe
    // scores 1.0 and the loop stops with budget left. That is a refusal, not
    // starvation, and a counter that cannot tell them apart would invite a
    // budget rise that converts nothing.
    let taught = format!("r000 desk material?");
    let (answer, probes) =
        brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, taught.as_bytes(), 3, 64);
    let refused = brain.derivation_stats();
    assert_eq!(refused.attempts, 1);
    assert!(probes < 64, "a taught question must stop early, spent {probes}");
    assert_eq!(
        refused.budget_exhausted, 0,
        "a call that stopped with budget left is not starved (answer {answer:?}, {probes} probes)"
    );

    // The same 2-hop question that answers at 4096 probes, given 1.
    let q = "r000 lamp on material?";
    let (answer, probes) =
        brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 1);
    assert_eq!(answer, None, "one probe cannot derive a 2-hop answer");
    assert_eq!(probes, 1);
    let starved = brain.derivation_stats();
    assert_eq!(
        starved.budget_exhausted, 1,
        "a call that spent its whole budget with no answer must score as starved"
    );
    assert_eq!(starved.attempts, 2);
    assert_eq!(starved.answered, 0);
}

#[test]
fn the_entry_point_the_node_uses_feeds_the_same_counter() {
    let mut brain = subject();
    teach_world(&mut brain);

    // `derived_by_substitution_reply` in crates/node/src/brain_api.rs calls
    // `derive_by_substitution`, the un-profiled wrapper. The scorecard reaches
    // the derivation through `integrate_autonomous`, which calls the profiled
    // one. Both must be measured, or the product's cost is invisible while the
    // scorecard's is published.
    let q = "r000 lamp on material?";
    let answer = brain.derive_by_substitution(
        QUERY_POOL,
        ANSWER_POOL,
        q.as_bytes(),
        3,
        DEFAULT_DERIVATION_PROBE_BUDGET,
    );
    let stats = brain.derivation_stats();
    assert_eq!(stats.attempts, 1, "the node's entry point must count its attempt");
    assert!(stats.probes > 0, "and must report what it spent");
    eprintln!(
        "node entry point: answer {answer:?} in {} probes of {DEFAULT_DERIVATION_PROBE_BUDGET}, \
         starved {}",
        stats.probes, stats.budget_exhausted
    );
    // Whichever way it went, the two accounts must agree with each other.
    assert_eq!(
        stats.answered + stats.budget_exhausted
            + usize::from(answer.is_none() && stats.budget_exhausted == 0),
        1,
        "exactly one of answered / starved / refused must be recorded"
    );
}
