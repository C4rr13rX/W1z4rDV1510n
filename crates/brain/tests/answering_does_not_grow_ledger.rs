//! Answering a question must not grow the concept-emergence ledger.
//!
//! Measured 2026-09-30 on the scorecard's scene world at scale 16: the trained
//! brain peaked at 28.3 MB, and then answering its 2,432 trained questions with
//! `Brain::observe` took the process to 546.7 MB. `Pool::check_concept_emergence`
//! inserts one permanent `Vec<NeuronId>` ledger key per run of length
//! 2..=`max_concept_member_count` ending at every atom observed, so a ~17-byte
//! question added ~1,071 entries that nothing ever reclaims. Peak RAM tracked
//! questions asked rather than knowledge held.
//!
//! The assertions are paired on purpose: the second half proves plain `observe`
//! still grows the ledger, so a suppression that had quietly stopped working --
//! or an emergence path that no longer runs at all -- cannot make the first half
//! pass vacuously.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

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

fn ledger_entries(brain: &Brain) -> usize {
    brain
        .fabric()
        .pool_ids()
        .into_iter()
        .filter_map(|id| brain.fabric().pool(id))
        .map(|pool| pool.read().sequence_ledger_entries())
        .sum()
}

#[test]
fn answering_does_not_grow_the_emergence_ledger() {
    let mut brain = subject();
    for room in 0..8 {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{room:03} lamp color?").into_bytes()),
            (ANSWER_POOL, b"red".to_vec()),
        ]);
    }
    // `pretrain_binding_episode` does not go through concept emergence at all:
    // measured on the scorecard, the ledger is still empty after training and
    // every one of its 2.6 M entries arrived while ANSWERING.
    let trained = ledger_entries(&brain);

    // 32 questions, none of them trained, so every run is a new ledger key
    // under the old behaviour.
    for room in 100..132 {
        brain.observe_read_only(QUERY_POOL, format!("r{room:03} lamp color?").as_bytes());
    }
    assert_eq!(
        ledger_entries(&brain),
        trained,
        "answering 32 questions grew the emergence ledger"
    );

    // The same questions through the learning path DO grow it -- otherwise the
    // assertion above would hold for a brain whose emergence had simply died.
    for room in 100..132 {
        brain.observe(QUERY_POOL, format!("r{room:03} lamp color?").as_bytes());
    }
    assert!(
        ledger_entries(&brain) > trained,
        "plain observe no longer grows the ledger, so the check above is vacuous"
    );
}

#[test]
fn suppression_does_not_leak_past_one_observe() {
    let mut brain = subject();
    brain.observe_read_only(QUERY_POOL, b"a read only question");
    let before = ledger_entries(&brain);
    brain.observe(QUERY_POOL, b"a learned observation");
    assert!(
        ledger_entries(&brain) > before,
        "a read-only observe left emergence suppressed for the next caller"
    );
}
