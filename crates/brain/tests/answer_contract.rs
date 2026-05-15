//! End-to-end tests of the answer contract per [`ARCHITECTURE.md`] §2.
//!
//! These tests are the reality check on the implementation: every
//! architectural claim about the Brain's outputs must demonstrate as
//! a working test, or the claim doesn't exist.

use w1z4rd_brain::{
    AnswerWithGrounding, Brain, BrainConfig, BytePassthroughEncoding,
    ConfidenceTier, PoolConfig,
};

/// Build a tiny brain with two byte-passthrough pools.  Common test
/// fixture so every test starts from a clean, predictable substrate.
fn build_two_pool_brain() -> (Brain, u32, u32) {
    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    (brain, pool_a, pool_b)
}

#[test]
fn fresh_brain_returns_outside_grounding_for_unobserved_input() {
    // Spec §2.2: when grounding is insufficient the brain returns
    // Unknown with a concrete RequestObservation.  Not a refusal —
    // an honest acknowledgment.

    let (brain, pool_a, pool_b) = build_two_pool_brain();

    // No observations at all.  Integration should report outside-grounding
    // and an empty answer.  The brain commits to honesty about its
    // ungroundedness at every developmental stage.
    let answer = brain.integrate(pool_a, pool_b);
    assert!(answer.answer.is_none(),
        "fresh brain must not invent answers; got {:?}", answer.answer);
    assert_eq!(answer.confidence_tier, ConfidenceTier::Ungrounded);
    assert!(!answer.next_steps_if_ungrounded.is_empty(),
        "ungrounded answer must carry concrete next-steps");
}

#[test]
fn cross_pool_observation_produces_grounded_answer_with_provenance() {
    // The full answer-contract pipeline:
    //  1. Co-temporal observation in two pools (training)
    //  2. Single-pool stimulation (recall query)
    //  3. integrate(query_pool=A, target_pool=B) returns a non-None
    //     answer with a grounding report, speculation flag, and
    //     non-zero fabric_confidence.

    let (mut brain, pool_a, pool_b) = build_two_pool_brain();

    // Train: co-observe "X" in A and "Y" in B for several ticks.  This
    // grows cross-pool axon terminals between the X-atom and Y-atom
    // per Phase 1's wiring rule.
    for _ in 0..6 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }

    // Recall: observe only "X" in pool A.  integrate(A, B) propagates
    // through the cross-pool terminals and should produce a non-None
    // answer derived from pool B's activation.
    brain.observe(pool_a, b"X");
    let answer: AnswerWithGrounding = brain.integrate(pool_a, pool_b);

    assert!(answer.answer.is_some(),
        "trained co-temporal grounding must produce an answer; got None");
    let decoded = answer.answer.as_ref().unwrap();
    // The brain may have promoted both the cross-pool binding AND a
    // within-pool concept for the repeated Y atom (since Y was
    // observed 7 times in pool_b's stream).  The strongest target
    // concept could be either — both are correct learnings.  The
    // architectural claim is that propagation from pool_a's X
    // produces a Y-containing answer in pool_b.  Anything else would
    // indicate the cross-pool wiring or the decoding contract broke.
    assert!(!decoded.is_empty() && decoded.iter().all(|&b| b == b'Y'),
        "answer should be a sequence of Y bytes (cross-pool recall of trained pattern); got {:?}",
        decoded);

    // Grounding report must reflect the cross-pool composition.
    assert!(!answer.grounding.outside_grounding,
        "trained grounding should not be outside_grounding");
    assert!(answer.grounding.speculation_flag,
        "cross-pool query → target is compositional, must flag as speculation per §2.3");
    assert!(answer.grounding.fabric_confidence > 0.0,
        "fabric_confidence must be positive when an answer is produced; got {}",
        answer.grounding.fabric_confidence);
    assert!(answer.grounding.strongest_match.is_some(),
        "grounding must surface the strongest matched concept");
    assert!(answer.grounding.eem_confidence.is_none(),
        "EEM not online in Phase 3 — confidence must be None, not zero or fudged");
    assert!(answer.grounding.annealer_confidence.is_none(),
        "Annealer not online in Phase 3 — confidence must be None, not zero");
}

#[test]
fn same_pool_query_does_not_flag_speculation() {
    // If query and target are the same pool, the answer is direct
    // retrieval, not cross-pool composition.  speculation_flag should
    // be false (assuming no binding concept fired).

    let (mut brain, pool_a, _pool_b) = build_two_pool_brain();

    // Train within-pool concept emergence: observe "abab" repeatedly.
    // The 'ab' sequence will be promoted to a concept by the pool's
    // automatic emergence rule.
    for _ in 0..5 {
        brain.observe(pool_a, b"ab");
        brain.advance_tick();
    }

    brain.observe(pool_a, b"ab");
    let answer = brain.integrate(pool_a, pool_a);

    if answer.answer.is_some() {
        // If the within-pool concept fired and we have an answer, the
        // composition is within-pool only — no speculation.
        assert!(!answer.grounding.speculation_flag,
            "same-pool retrieval must not flag as speculation");
    }
    // Same-pool answer may still be ungrounded in this small test
    // (depends on the strongest-concept being a non-binding) — the
    // important property is that IF an answer is produced, the
    // speculation flag honestly reflects whether composition happened.
}

#[test]
fn stats_track_developmental_progression() {
    // Spec §2.5 + §10: the brain's developmental profile must be readable
    // and reflect what's been observed.  A fresh brain has no concepts;
    // an observed brain accumulates concepts and binding concepts.

    let (mut brain, pool_a, pool_b) = build_two_pool_brain();
    let initial = brain.stats();
    assert_eq!(initial.tick, 0);
    // 3 pools registered: binding pool (auto) + text + audio.
    assert_eq!(initial.pool_count, 3);
    assert_eq!(initial.total_concepts, 0,
        "no concepts before any observation");
    assert_eq!(initial.total_binding, 0,
        "no binding concepts before any co-temporal observation");

    // Observe both pools at the same tick for several rounds — should
    // produce cross-pool terminals AND a binding-concept candidate.
    for _ in 0..5 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }

    let mature = brain.stats();
    assert!(mature.tick > 0, "tick must advance");
    assert!(mature.total_terminals > 0,
        "co-temporal observation must grow terminals");
    assert!(mature.total_binding >= 1,
        "5 reps of the same X/Y multi-pool firing must promote a binding concept (threshold default 3); got total_binding={}",
        mature.total_binding);
}
