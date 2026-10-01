//! Answering a question must give its working set back.
//!
//! Measured 2026-10-01 with `python tools/scorecard.py --stress --with-store`:
//! `page_ins` equalled `page_outs` EXACTLY at all four scales (241/241,
//! 803/803, 3035/3035, 11963/11963), a store-attached run ended with 0 neurons
//! evicted and 129,639 terminals resident at scale 64, and peak RAM was HIGHER
//! with the store attached (45.8 MB) than without it (40.5). Nothing bounded
//! the working set.
//!
//! The cause was not a discard that fails. `Pool::page_in_neuron` records a
//! page-in only `if self.read_only_inference_active`, and the only two callers
//! of `Brain::begin_read_only_inference` were `activate_for_prediction` and
//! `activate_for_indexed_prediction` -- neither of which is on any answer path
//! the scorecard or the node takes. So `finish_read_only_inference` drained an
//! empty set and returned 0, which both the scorecard and every node route
//! threw away.
//!
//! This asserts the three facts that together say the window is open and the
//! release is safe:
//!   1. answering releases bodies (`finish_read_only_inference` > 0),
//!   2. resident terminals after N questions do not track N,
//!   3. recall of what was taught survives being released and paged back.
//!
//! (3) is the one that makes (1) and (2) worth having: a release that loses a
//! taught fact is not a saving.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const FACTS: usize = 48;
const EPOCHS: usize = 3;

fn store_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("w1z4rd-ro-window-{}-{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("store dir");
    dir
}

/// Built exactly as `examples/scorecard.rs` builds its subject, with the same
/// `attach_wbrain` the node's brain_server uses. The attach count is asserted
/// because a partial attach pages some pools and not others, which measures
/// nothing.
fn trained_subject(dir: &std::path::Path) -> Brain {
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
    let attached = brain
        .attach_wbrain(dir.join("brain.wbrain"))
        .expect("attach_wbrain");
    assert_eq!(attached, 3, "binding + query + answer must all attach");
    for _ in 0..EPOCHS {
        for fact in 0..FACTS {
            brain.pretrain_binding_episode(&[
                (QUERY_POOL, question(fact).into_bytes()),
                (ANSWER_POOL, answer(fact).into_bytes()),
            ]);
        }
    }
    // The node's /brain/sleep: write every body into the container and drop its
    // RAM copy, so a later question has to page in to answer.
    brain
        .serialize_all_neurons_for_idle()
        .expect("serialize_all_neurons_for_idle");
    brain
}

fn question(fact: usize) -> String {
    format!("r{fact:03} lamp color?")
}

fn answer(fact: usize) -> String {
    format!("colour{}", fact % 7)
}

/// The scorecard's `recall`: the same two calls in the same order, then the
/// release whose return value this test is about.
fn ask(brain: &mut Brain, query: &str) -> (Vec<u8>, usize) {
    brain.observe_read_only(QUERY_POOL, query.as_bytes());
    let legacy = brain.integrate(QUERY_POOL, ANSWER_POOL);
    let decoded = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .or(legacy.answer)
        .unwrap_or_default();
    let released = brain
        .finish_read_only_inference()
        .expect("finish_read_only_inference");
    (decoded, released)
}

#[test]
fn answering_releases_the_bodies_it_paged_in_and_keeps_the_answer() {
    let dir = store_dir("releases");
    let mut brain = trained_subject(&dir);

    let resident_after_sleep = brain.stats().resident_terminals;
    let mut released_total = 0usize;
    let mut correct = 0usize;
    for fact in 0..FACTS {
        let (decoded, released) = ask(&mut brain, &question(fact));
        released_total += released;
        if decoded == answer(fact).into_bytes() {
            correct += 1;
        }
    }

    // 1. The window is open: answering pages bodies in and hands them back.
    assert!(
        released_total > 0,
        "finish_read_only_inference released {} bodies over {} questions -- the \
         read-only window is not being opened on the answer path",
        released_total,
        FACTS
    );

    // 3. Releasing a body must not lose a taught fact. Asserted BEFORE the
    //    residency bound, because a brain that answers nothing trivially holds
    //    nothing resident and would pass (2) vacuously.
    assert_eq!(
        correct, FACTS,
        "recall fell to {}/{} after releasing paged-in bodies",
        correct, FACTS
    );

    // 2. Residency does not track questions asked. Every one of the FACTS
    //    questions paged its own body set in; if none were released the final
    //    reading would be the union of all of them.
    let resident_after_asking = brain.stats().resident_terminals;
    assert!(
        resident_after_asking <= resident_after_sleep.saturating_add(released_total / 2),
        "resident terminals grew {} -> {} while answering {} questions, so the \
         working set is unbounded in questions asked",
        resident_after_sleep,
        resident_after_asking,
        FACTS
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The second half of the same fact, stated as a ratio rather than a bound:
/// a question's page-ins must come back in the same order of magnitude they
/// went in. `page_ins == page_outs` was the 2026-10-01 signature of a window
/// that never opened, and a ratio catches a PARTIAL release that a
/// `released > 0` assertion cannot.
#[test]
fn releases_track_page_ins_rather_than_staying_at_zero() {
    let dir = store_dir("ratio");
    let mut brain = trained_subject(&dir);

    let page_ins_before = brain.stats().page_ins;
    let mut released_total = 0usize;
    for fact in 0..FACTS {
        let (_, released) = ask(&mut brain, &question(fact));
        released_total += released;
    }
    let paged_in = brain.stats().page_ins.saturating_sub(page_ins_before);

    assert!(
        paged_in > 0,
        "no body was paged in over {} questions against a slept brain, so this \
         test cannot say anything about releasing them",
        FACTS
    );
    assert!(
        released_total * 2 >= paged_in as usize,
        "released {} of {} page-ins -- less than half the working set came back",
        released_total,
        paged_in
    );

    let _ = std::fs::remove_dir_all(&dir);
}
