//! Does the route `/brain/ask` takes cost it any answer?
//!
//! `crates/node/src/api.rs:8021` answers `/brain/ask` synchronously with
//! `brain.integrate(qp, tp)` -- the legacy Hebbian integrate -- while
//! `bin/brain_server.rs:2398` (`/chat`) calls `integrate_autonomous_tuned`,
//! which is the only path that can reach `Eem::chain_explore`. Backlog item
//! [93a90bef] asks whether that difference is the reason integration is 0%:
//!
//!   (a) the chain explorer answers probes the legacy integrate cannot, and
//!       `/brain/ask` throws those answers away -- a real production-parity
//!       defect, fix by reaching the explorer inline; or
//!   (b) both routes return the same thing on the same brain, so the route is
//!       not where the answer is lost and the fault is inside the explorer.
//!
//! The two cases need opposite work, and a grep cannot tell them apart. This
//! file asks the one question that can: ONE trained brain, ONE probe family,
//! answered TWICE, both counts printed.
//!
//! The counts are PRINTED, not asserted against a distribution -- asserting
//! "integration is 0" would freeze today's defect into the suite, and
//! asserting "integration is N" would tune the test to this probe set. What IS
//! asserted is the invariant that makes the comparison mean anything: the
//! trained half of the world is recalled, so a zero below is about integration
//! and not about an empty brain.
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --release --test ask_route_integrate_parity -- --nocapture

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
/// material" on its own; the composition of the two does. Same shape as the
/// scorecard's `on_material(2h)` family.
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

/// Non-empty answers, and of those the ones equal to `expect`.
#[derive(Default)]
struct Tally {
    answered: u32,
    correct: u32,
}

impl Tally {
    fn record(&mut self, answer: Option<&[u8]>, expect: &[u8]) {
        if let Some(bytes) = answer {
            if !bytes.is_empty() {
                self.answered += 1;
                if bytes == expect {
                    self.correct += 1;
                }
            }
        }
    }
}

#[test]
fn ask_route_and_chat_route_answer_the_same_probe_family() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    train_chainable_world(&mut brain, ROOMS);

    // Control: the TRAINED questions, through BOTH routes. Without this a pair
    // of zeros below is unreadable -- it could mean the brain learned nothing.
    // It also prices the route difference on questions that DO have a stored
    // answer, which is what production parity (M2) actually turns on.
    let mut trained_legacy = Tally::default();
    let mut trained_autonomous = Tally::default();
    // Which of `integrate`'s own steps drops a trained answer? Reported rather
    // than inferred, per the standing rule that an if/else answer branch must
    // be READ. These are the fields `AnswerWithGrounding` already carries, so
    // naming the arm costs no new diagnostic.
    let mut legacy_outside_grounding = 0u32;
    let mut legacy_fabric_confidence = 0.0f32;
    let mut legacy_jaccard = 0.0f32;
    for room in 0..ROOMS {
        let question = format!("r{room:03} lamp on");
        brain.observe_read_only(QUERY_POOL, question.as_bytes());
        let legacy = brain.integrate(QUERY_POOL, ANSWER_POOL);
        if legacy.grounding.outside_grounding {
            legacy_outside_grounding += 1;
        }
        legacy_fabric_confidence += legacy.grounding.fabric_confidence;
        legacy_jaccard += legacy.grounding.strongest_match_jaccard;
        trained_legacy.record(legacy.answer.as_deref(), b"desk");
        brain.observe_read_only(QUERY_POOL, question.as_bytes());
        trained_autonomous.record(
            brain
                .integrate_autonomous_tuned(
                    QUERY_POOL,
                    ANSWER_POOL,
                    NODE_FABRIC_CONFIDENCE_THRESHOLD,
                    3,
                    16,
                    BINDING_MATCH_THRESHOLD,
                )
                .answer
                .as_deref(),
            b"desk",
        );
    }

    // The integration probes: never trained, true in the world, answerable
    // only by composing the two facts. Same family, both routes.
    let mut probe_legacy = Tally::default();
    let mut probe_autonomous = Tally::default();
    for room in 0..ROOMS {
        let probe = format!("r{room:03} lamp on material");
        brain.observe_read_only(QUERY_POOL, probe.as_bytes());
        probe_legacy.record(
            brain.integrate(QUERY_POOL, ANSWER_POOL).answer.as_deref(),
            b"oak",
        );
        brain.observe_read_only(QUERY_POOL, probe.as_bytes());
        probe_autonomous.record(
            brain
                .integrate_autonomous_tuned(
                    QUERY_POOL,
                    ANSWER_POOL,
                    NODE_FABRIC_CONFIDENCE_THRESHOLD,
                    3,
                    16,
                    BINDING_MATCH_THRESHOLD,
                )
                .answer
                .as_deref(),
            b"oak",
        );
    }

    eprintln!("ask-route parity over {ROOMS} rooms ({} facts trained)", ROOMS * 2);
    eprintln!("  TRAINED questions (control)");
    eprintln!(
        "    brain.integrate            (/brain/ask)  answered {}/{ROOMS}  correct {}",
        trained_legacy.answered, trained_legacy.correct
    );
    eprintln!(
        "    integrate_autonomous_tuned (/chat)       answered {}/{ROOMS}  correct {}",
        trained_autonomous.answered, trained_autonomous.correct
    );
    eprintln!(
        "    brain.integrate on a TRAINED question: outside_grounding {}/{ROOMS}  mean fabric_confidence {:.3}  mean strongest_match_jaccard {:.3}",
        legacy_outside_grounding,
        legacy_fabric_confidence / ROOMS as f32,
        legacy_jaccard / ROOMS as f32
    );
    eprintln!("  INTEGRATION probes (never trained)");
    eprintln!(
        "    brain.integrate            (/brain/ask)  answered {}/{ROOMS}  correct {}",
        probe_legacy.answered, probe_legacy.correct
    );
    eprintln!(
        "    integrate_autonomous_tuned (/chat)       answered {}/{ROOMS}  correct {}",
        probe_autonomous.answered, probe_autonomous.correct
    );
    eprintln!(
        "  verdict input: route difference on integration probes = {} answers",
        probe_autonomous.answered as i64 - probe_legacy.answered as i64
    );

    // The one invariant: the world is real. Everything above is a reading.
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
        "the trained half of the world is not recalled, so no count above is about integration"
    );
    assert!(
        probe_legacy.correct <= probe_legacy.answered
            && probe_autonomous.correct <= probe_autonomous.answered,
        "a correct answer that was not counted as answered means the tally is wrong"
    );
}
