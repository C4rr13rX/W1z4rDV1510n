//! Does `integrate_autonomous_tuned`'s fabric arm return a PROPAGATION score
//! as if it were a retrieval, and shadow the derivation behind it?
//!
//! Backlog item `[988dd17c]`, measured by Cove in pass 12 by reading
//! `crates/brain/src/brain.rs`: after the step-0 binding-precision gate the
//! method runs `let fabric_ans = self.integrate(query_pool, target_pool)` and
//! returns it whenever it is non-empty and its `fabric_confidence` clears
//! `fabric_confidence_threshold` -- clearing the inner `outside_grounding`
//! flag on the way out. Every arm that could COMPOSE an answer sits below
//! that: `chain_explore`, and `integrate_autonomous`'s derivation.
//!
//! The scorecard cannot see this. It passes
//! `ANSWER_FABRIC_CONFIDENCE_THRESHOLD`, which is 100.0 against a 0..1
//! confidence, so the arm is switched off and every gated integration number
//! describes a configuration in which it never fires. The node's research loop
//! passes `RESEARCH_LOOP_FABRIC_CONFIDENCE_THRESHOLD`, which is 0.10. So the
//! question is entirely about the product, and it is a number: where does a
//! composite prompt's `fabric_confidence` actually land relative to 0.10?
//!
//! This file measures it rather than inferring it, and asserts the two
//! structural facts the fix rests on -- not the measured confidence itself,
//! which is a reading and would freeze this brain's tuning into the suite.
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --release --test fabric_arm_shadows_derivation -- --nocapture

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// What `crates/node/src/api.rs`'s hypothesis research loop passes, via
/// `w1z4rd_brain::RESEARCH_LOOP_FABRIC_CONFIDENCE_THRESHOLD`. Read from the
/// constant rather than copied, so a change to the product's threshold reds
/// this file instead of silently making it describe nothing.
const NODE_THRESHOLD: f32 = w1z4rd_brain::RESEARCH_LOOP_FABRIC_CONFIDENCE_THRESHOLD;
const BINDING_MATCH_THRESHOLD: f32 = w1z4rd_brain::ANSWER_BINDING_MATCH_THRESHOLD;

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

/// The 16-binding lamp/desk/material world the backlog item names: two facts
/// per room that chain, same shape as the scorecard's `on_material(2h)`.
/// Eight rooms is sixteen bindings.
const ROOMS: u32 = 8;

fn train_chainable_world(brain: &mut Brain) {
    for room in 0..ROOMS {
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

/// CRITERION 1: the number, in the test.
///
/// For a composite two-hop prompt, measure `fabric_confidence` off
/// `Brain::integrate` -- the exact value the short-circuit compares against --
/// and record which side of the node's 0.10 it lands on.
///
/// What is ASSERTED is not that number. It is the pair of facts that decide
/// whether a gate on this arm can work at all, and they are arithmetic rather
/// than tuning (`brain.rs:4120`): `precision = intersect / bind_query.len()`
/// and `recall = intersect / q_atoms.len()`. A composite prompt CONTAINS its
/// own sub-question, so the sub-question's binding is explained whole and
/// precision is 1.0 -- which is why a precision gate here is inert, and why
/// the acceptance criterion says recall. The composite carries atoms that
/// binding does not, so its recall is below 1.0. A prompt that IS the taught
/// question has both at 1.0.
#[test]
fn composite_prompt_fabric_confidence_against_the_node_threshold() {
    let mut brain = subject();
    train_chainable_world(&mut brain);

    let composite = format!("r{:03} lamp on material", 0);
    brain.observe_read_only(QUERY_POOL, composite.as_bytes());
    let bm = brain.best_binding_match_v2(QUERY_POOL);
    let legacy = brain.integrate(QUERY_POOL, ANSWER_POOL);
    let answered = legacy.answer.as_ref().map(|b| !b.is_empty()).unwrap_or(false);
    let fc = legacy.grounding.fabric_confidence;

    eprintln!("composite prompt {composite:?} on {} bindings", ROOMS * 2);
    eprintln!(
        "  best_binding_match_v2: precision {:.4}  recall {:.4}  tier {:?}",
        bm.precision, bm.recall, bm.tier
    );
    eprintln!(
        "  integrate():           answered {answered}  answer {:?}  fabric_confidence {fc:.4}",
        legacy.answer.as_deref().map(String::from_utf8_lossy)
    );
    eprintln!(
        "  node threshold {NODE_THRESHOLD:.2}  ->  fabric_confidence is {} it, so the short-circuit {}",
        if fc >= NODE_THRESHOLD { "AT OR ABOVE" } else { "BELOW" },
        if answered && fc >= NODE_THRESHOLD { "FIRES" } else { "does not fire" }
    );
    eprintln!(
        "  scorecard threshold {:.2}  ->  the arm is off there, which is why no gated number sees this",
        w1z4rd_brain::ANSWER_FABRIC_CONFIDENCE_THRESHOLD
    );

    // The prompt reaches the arm at all: step 0 accepts it.
    assert!(
        bm.precision >= BINDING_MATCH_THRESHOLD,
        "the composite prompt must clear the step-0 gate for this measurement to be about \
         the fabric arm; precision {:.4} < {BINDING_MATCH_THRESHOLD:.2}",
        bm.precision
    );
    // A precision gate on this arm would be INERT -- the defect it is meant to
    // fix. This is the load-bearing half of criterion 2.
    assert!(
        bm.precision >= 0.999,
        "a composite prompt is expected to score precision 1.0 against its own sub-question, \
         which is why the fix gates on recall; got {:.4}",
        bm.precision
    );
    // A recall gate BITES. The other load-bearing half.
    assert!(
        bm.recall < 0.999,
        "a composite prompt carries atoms its best binding does not, so its recall must be \
         below 1.0 or a recall gate would reject trained recalls too; got {:.4}",
        bm.recall
    );
}

/// A prompt that IS a taught question has recall 1.0, so the recall gate
/// cannot cost a retrieval. The counterpart to the assertion above: without
/// this, "recall < 0.999 for a composite" could be true because recall is
/// below 1.0 for EVERYTHING, and the gate would silence the trained half of
/// the world.
#[test]
fn a_taught_question_recalls_exactly_so_the_gate_cannot_cost_it() {
    let mut brain = subject();
    train_chainable_world(&mut brain);

    let mut exact = 0u32;
    let mut worst = 1.0f32;
    for room in 0..ROOMS {
        let taught = format!("r{room:03} lamp on");
        brain.observe_read_only(QUERY_POOL, taught.as_bytes());
        let bm = brain.best_binding_match_v2(QUERY_POOL);
        worst = worst.min(bm.recall);
        if bm.recall >= 0.999 {
            exact += 1;
        }
    }
    eprintln!("taught questions: recall >= 0.999 on {exact}/{ROOMS}, worst recall {worst:.4}");
    assert_eq!(
        exact, ROOMS,
        "every taught question must recall exactly or the recall gate silences trained material; \
         worst recall {worst:.4}"
    );
}

/// CRITERION 2: at the node's threshold, `integrate_autonomous_tuned` must not
/// hand back a propagation answer for a prompt it cannot bind EXACTLY.
///
/// This is the assertion that fails without the fix. It does not require the
/// method to answer -- composing this chain is a different item -- only that
/// what it returns for an inexact prompt is not a retrieval wearing
/// `outside_grounding = false`. PRIORITY ZERO: silence beats a plausible
/// answer, and the derivation arm behind this one cannot run until the answer
/// it shadows is empty.
#[test]
fn the_fabric_arm_does_not_answer_an_inexact_prompt_at_the_node_threshold() {
    let mut brain = subject();
    train_chainable_world(&mut brain);

    let mut returned_without_exact_recall = Vec::new();
    for room in 0..ROOMS {
        let composite = format!("r{room:03} lamp on material");
        brain.observe_read_only(QUERY_POOL, composite.as_bytes());
        let recall = brain.best_binding_match_v2(QUERY_POOL).recall;
        brain.observe_read_only(QUERY_POOL, composite.as_bytes());
        let res = brain.integrate_autonomous_tuned(
            QUERY_POOL,
            ANSWER_POOL,
            NODE_THRESHOLD,
            w1z4rd_brain::ANSWER_CHAIN_MAX_DEPTH,
            w1z4rd_brain::RESEARCH_LOOP_CHAIN_MAX_VISIT,
            BINDING_MATCH_THRESHOLD,
        );
        let answer = res.answer.as_deref().unwrap_or(b"");
        if !answer.is_empty() && recall < 0.999 {
            returned_without_exact_recall.push((
                composite,
                String::from_utf8_lossy(answer).into_owned(),
                recall,
                res.grounding.fabric_confidence,
                res.grounding.speculation_flag,
            ));
        }
    }

    for (q, a, recall, fc, spec) in &returned_without_exact_recall {
        eprintln!(
            "  {q:?} -> {a:?}  recall {recall:.4}  fabric_confidence {fc:.4}  speculation {spec}"
        );
    }
    assert!(
        returned_without_exact_recall.is_empty(),
        "integrate_autonomous_tuned returned a fabric answer for {} of {ROOMS} prompts whose \
         best binding is an INEXACT match, which shadows every arm below it including the \
         derivation; see the lines above",
        returned_without_exact_recall.len()
    );
}
