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

/// The node's answer routes reach past `Brain` into the fabric, so the
/// suppression needs a fabric-level entry point too.
///
/// Measured 2026-09-30 read-only: `crates/node/src/api.rs:8016` (`/brain/ask`),
/// `:8146` (the hypothesis research loop) and
/// `crates/node/src/brain_api.rs:6089` (the idle thinking loop, every ~250 ms)
/// all called `brain.fabric_mut().observe(..)`, which never touches
/// `Brain::observe` and so could not be fixed by `observe_read_only`. Zero
/// callers of `observe_read_only` existed anywhere in `crates/node/src`
/// against 17 call sites of the mutating `observe`, while the node's own pool
/// config sets `text.max_concept_member_count = 32` (`brain_api.rs:148`,
/// `brain_server.rs:284`) -- so emergence was on, at a member cap HIGHER than
/// the one the 1,071-entries-per-question measurement was taken at.
///
/// Paired on purpose, as above: the fabric path must still grow the ledger, or
/// the read-only assertion passes for a brain whose emergence has died.
#[test]
fn the_fabric_level_answer_path_does_not_grow_the_emergence_ledger() {
    let mut brain = subject();
    for room in 0..8 {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{room:03} lamp color?").into_bytes()),
            (ANSWER_POOL, b"red".to_vec()),
        ]);
    }
    let trained = ledger_entries(&brain);

    for room in 200..232 {
        brain.observe_fabric_read_only(QUERY_POOL, format!("r{room:03} lamp color?").as_bytes());
    }
    assert_eq!(
        ledger_entries(&brain),
        trained,
        "answering 32 questions through the fabric-level path grew the emergence ledger"
    );

    for room in 200..232 {
        brain
            .fabric_mut()
            .observe(QUERY_POOL, format!("r{room:03} lamp color?").as_bytes());
    }
    assert!(
        ledger_entries(&brain) > trained,
        "fabric_mut().observe no longer grows the ledger, so the check above is vacuous"
    );
}

/// The fabric-level path is deliberately NOT `Brain::observe` with suppression
/// bolted on: it performs no QA capture and writes no `recent_frames` entry.
///
/// That is the whole reason it exists rather than the answer routes being
/// pointed at `observe_read_only`. A QA pair harvested from a question being
/// ANSWERED is training those routes never asked for, and the node's idle
/// thinking loop seeds itself FROM `qa_db`, so letting it write back would
/// have it feed on its own output.
#[test]
fn the_fabric_level_answer_path_captures_no_qa_pair() {
    let mut brain = subject();
    // Two frames in different pools within 2 ticks is exactly the shape
    // `Brain::observe` captures as a QA pair.
    brain.observe_fabric_read_only(QUERY_POOL, b"r001 lamp color?");
    brain.observe_fabric_read_only(ANSWER_POOL, b"red");
    assert_eq!(
        brain.qa_db().len(),
        0,
        "the fabric-level answer path captured a QA pair, so it is routing through Brain::observe"
    );

    // And the Brain-level path still does capture one, so the assertion above
    // is not passing because QA capture has stopped working.
    brain.observe(QUERY_POOL, b"r002 lamp color?");
    brain.observe(ANSWER_POOL, b"red");
    assert!(
        brain.qa_db().len() > 0,
        "Brain::observe no longer captures QA pairs, so the check above is vacuous"
    );
}

/// A byte must not acquire one terminal per fact.
///
/// Measured 2026-09-30 at scale 16 of the scorecard, the three largest neurons
/// in the query pool were the single bytes `q:cg`, `q:IA` and `q:Pw`, each at
/// fan-out 2,432 -- the fact count exactly -- while the largest concept held 21.
/// docs/RAM_GOAL.md records the same shape in production at ~4 M terminals and
/// 82 MB on one atom. The cap is `PoolConfig::max_atom_fanout`.
#[test]
fn an_atom_stops_acquiring_a_terminal_per_fact() {
    const CAP: usize = 8;
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);
    for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
        let mut pc = PoolConfig::defaults(name, id);
        pc.recent_atoms_window = 2048;
        pc.concept_emergence_threshold = 2;
        pc.max_concept_member_count = 64;
        pc.max_atom_fanout = CAP;
        brain.create_pool(pc, Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>);
    }
    // Every question contains "r", " " and "?", so under the old wiring each of
    // those bytes would end up with one terminal per fact -- 64 here.
    for room in 0..64 {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{room:03} lamp color?").into_bytes()),
            (ANSWER_POOL, b"red".to_vec()),
        ]);
    }
    let worst_atom = brain
        .fabric()
        .pool_ids()
        .into_iter()
        .filter_map(|id| brain.fabric().pool(id))
        .flat_map(|pool| {
            pool.read()
                .iter_neurons()
                .filter(|n| n.is_atom())
                .map(|n| n.terminals.len())
                .collect::<Vec<_>>()
        })
        .max()
        .unwrap_or(0);
    assert!(
        worst_atom <= CAP,
        "an atom reached fan-out {worst_atom} against a cap of {CAP}"
    );
    // ...and the facts are still recalled, which is the whole point: the
    // binding is reached by index, not by the byte firing into it.
    for room in 0..64 {
        brain.observe_read_only(QUERY_POOL, format!("r{room:03} lamp color?").as_bytes());
        let answer = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
        assert_eq!(
            answer.as_deref(),
            Some(&b"red"[..]),
            "fact r{room:03} was lost when atom fan-out was capped"
        );
    }
}
