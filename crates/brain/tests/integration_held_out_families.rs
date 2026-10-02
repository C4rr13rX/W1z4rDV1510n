//! Can integration only rise through a GENERAL mechanism?
//!
//! The scorecard's integration number is 77.3 % at scale 64, and every probe it
//! asks is built by one generator from one wording template. That is enough to
//! measure whether a chain composes; it is not enough to tell a general
//! mechanism from one that fits the template. Three things would move the
//! scorecard without any generalisation at all: wording the question the same
//! way every time, never putting a second chain in the brain that the question
//! has to choose between, and never asking a relation the generator did not
//! emit.
//!
//! So this file asks the same two-hop question four ways on the same world and
//! prints correct / WRONG / silent for each. It is deliberately NOT in the gated
//! scorecard: adding a harder family there would lower `integration_pct` and
//! trip the gate's own ratchet, which is a re-baseline decision and not a thing
//! to slip in beside a fix. It is a measurement the lead owns, run on demand.
//!
//! The one thing asserted is PRIORITY ZERO, and it is the owner's standard
//! rather than this file's: **a wrong answer is worse than silence**, so `wrong`
//! must be 0 in every condition. A condition that answers nothing passes. A
//! condition that invents does not. Recall is asserted too, so a column of
//! zeros cannot be read as "the brain is empty".
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --release --test integration_held_out_families -- --nocapture

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const ROOMS: u32 = 8;

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

fn teach(brain: &mut Brain, q: &str, a: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, q.as_bytes().to_vec()),
        (ANSWER_POOL, a.as_bytes().to_vec()),
    ]);
}

/// Exactly the scorecard's `infer()` in `crates/brain/examples/scorecard.rs`:
/// `observe_read_only`, `integrate_autonomous` on the shared `answer_path`
/// constants, then release what the query paged in. Copied in shape rather than
/// re-tuned, because a measurement taken on a different path would not be about
/// the number the gate reports.
fn infer(brain: &mut Brain, q: &str) -> Vec<u8> {
    brain.observe_read_only(QUERY_POOL, q.as_bytes());
    let answer = brain
        .integrate_autonomous(
            QUERY_POOL,
            ANSWER_POOL,
            w1z4rd_brain::ANSWER_FABRIC_CONFIDENCE_THRESHOLD,
            w1z4rd_brain::ANSWER_CHAIN_MAX_DEPTH,
            w1z4rd_brain::ANSWER_CHAIN_MAX_VISIT,
        )
        .answer;
    let _ = brain.finish_read_only_inference();
    answer.unwrap_or_default()
}

#[derive(Default)]
struct Verdict {
    correct: u32,
    wrong: u32,
    silent: u32,
}

impl Verdict {
    fn record(&mut self, got: &[u8], want: &str) {
        if got.is_empty() {
            self.silent += 1;
        } else if got == want.as_bytes() {
            self.correct += 1;
        } else {
            self.wrong += 1;
        }
    }
    fn total(&self) -> u32 {
        self.correct + self.wrong + self.silent
    }
}

/// The world the scorecard builds, in miniature: two facts per room that chain.
fn teach_base_world(brain: &mut Brain) {
    for room in 0..ROOMS {
        teach(brain, &format!("r{room:03} lamp on"), "desk");
        teach(brain, &format!("r{room:03} desk material"), "oak");
    }
}

#[test]
fn held_out_families_never_invent_an_answer() {
    // Condition A -- the scorecard's own wording. The control: without it a
    // zero in B, C or D is unreadable.
    let mut a_brain = subject();
    teach_base_world(&mut a_brain);
    let mut a = Verdict::default();
    for room in 0..ROOMS {
        let got = infer(&mut a_brain, &format!("r{room:03} lamp on material"));
        a.record(&got, "oak");
    }

    // Condition B -- PARAPHRASE. Same world, same answer, wording the generator
    // never emits and the brain was never taught. An atom is a byte, so this
    // shares most of its bytes with the trained questions and none of their
    // order: exactly the case where a mechanism that fits the template and one
    // that generalises come apart.
    let mut b_brain = subject();
    teach_base_world(&mut b_brain);
    let mut b = Verdict::default();
    for room in 0..ROOMS {
        let got = infer(&mut b_brain, &format!("material of the lamp in r{room:03}"));
        b.record(&got, "oak");
    }

    // Condition C -- DISTRACTOR CHAIN. A second complete chain per room, true
    // and irrelevant, sharing the room id. Now the question must pick WHICH
    // chain, and a mechanism that composes whatever it can reach will answer
    // `wool` -- which is the hallucination PRIORITY ZERO forbids, not a miss.
    let mut c_brain = subject();
    teach_base_world(&mut c_brain);
    for room in 0..ROOMS {
        teach(&mut c_brain, &format!("r{room:03} chair on"), "rug");
        teach(&mut c_brain, &format!("r{room:03} rug material"), "wool");
    }
    let mut c = Verdict::default();
    for room in 0..ROOMS {
        let got = infer(&mut c_brain, &format!("r{room:03} lamp on material"));
        c.record(&got, "oak");
    }

    // Condition D -- HELD-OUT RELATION. The same two-hop shape on a relation
    // the base world never used. Taught only as single facts, never composed.
    let mut d_brain = subject();
    for room in 0..ROOMS {
        teach(&mut d_brain, &format!("r{room:03} clock under"), "shelf");
        teach(&mut d_brain, &format!("r{room:03} shelf material"), "pine");
    }
    let mut d = Verdict::default();
    for room in 0..ROOMS {
        let got = infer(&mut d_brain, &format!("r{room:03} clock under material"));
        d.record(&got, "pine");
    }

    let conditions: [(&str, &Verdict, &str); 4] = [
        ("A scorecard wording", &a, "control: the family the gate measures"),
        ("B paraphrase", &b, "held out: wording never taught"),
        ("C distractor chain", &c, "held out: two chains, must choose"),
        ("D held-out relation", &d, "held out: relation never composed"),
    ];
    eprintln!("{ROOMS} rooms, the same two-hop question asked four ways");
    eprintln!("  {:<22} {:>7} {:>7} {:>7}   {}", "condition", "correct", "WRONG", "silent", "what it holds out");
    for (name, v, why) in &conditions {
        eprintln!(
            "  {name:<22} {:>7} {:>7} {:>7}   {why}",
            v.correct, v.wrong, v.silent
        );
    }

    // Non-vacuity: the trained half of each world is recalled, so a column of
    // zeros is about integration and not about an empty brain.
    for (name, brain, q, want) in [
        ("A", &mut a_brain, format!("r000 lamp on"), "desk"),
        ("B", &mut b_brain, format!("r000 lamp on"), "desk"),
        ("C", &mut c_brain, format!("r000 lamp on"), "desk"),
        ("D", &mut d_brain, format!("r000 clock under"), "shelf"),
    ] {
        brain.observe_read_only(QUERY_POOL, q.as_bytes());
        let recalled = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
        let _ = brain.finish_read_only_inference();
        assert_eq!(
            recalled.as_deref(),
            Some(want.as_bytes()),
            "condition {name}: the trained question {q:?} is not recalled, so no count above is \
             about integration"
        );
    }

    // PRIORITY ZERO, the only behavioural assertion here. Silence is a pass.
    let inventing: Vec<&str> = conditions
        .iter()
        .filter(|(_, v, _)| v.wrong > 0)
        .map(|(name, _, _)| *name)
        .collect();
    assert!(
        inventing.is_empty(),
        "condition(s) {inventing:?} returned a WRONG answer where silence was available. \
         The owner's standard is that the brain is never wrong: when it has no grounded answer \
         it has NO answer. See the table above for the counts"
    );

    // The control must do something, or every held-out zero below it is
    // explained by the control being broken rather than by generalisation.
    assert!(
        a.correct > 0,
        "the control condition answered nothing correct ({}/{}), so this file measures nothing",
        a.correct,
        a.total()
    );
}
