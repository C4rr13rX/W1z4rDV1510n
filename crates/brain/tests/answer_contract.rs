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

#[test]
fn trained_binding_does_not_decode_collapsed_target_leaves_twice() {
    let (mut brain, pool_a, pool_b) = build_two_pool_brain();

    // Repetition promotes both atom sequences and hierarchical concepts in
    // the target pool.  The binding therefore carries atoms plus concepts
    // derived from the same response leaves.
    for _ in 0..8 {
        brain.observe(pool_a, b"dog");
        brain.observe(pool_b, b"animal");
        brain.advance_tick();
    }

    brain.observe(pool_a, b"dog");
    assert_eq!(
        brain.decode_best_trained_binding(pool_a, pool_b),
        Some(b"animal".to_vec()),
        "trained readout must serialize ordered target atoms exactly once",
    );
}

#[test]
fn trained_binding_survives_wide_window_multi_pair_curriculum() {
    let mut brain = Brain::new(BrainConfig::default());
    let mut input = PoolConfig::defaults("prompt", 1);
    input.recent_atoms_window = 65_536;
    input.max_concept_member_count = 64;
    input.decay_rate = 0.00002;
    input.prune_floor = 0.001;
    let mut output = PoolConfig::defaults("response", 4);
    output.recent_atoms_window = 65_536;
    output.max_concept_member_count = 64;
    output.decay_rate = 0.00002;
    output.prune_floor = 0.001;
    brain.create_pool(input, Box::new(BytePassthroughEncoding { prefix: "prompt" }));
    brain.create_pool(output, Box::new(BytePassthroughEncoding { prefix: "response" }));

    let pairs: &[(&[u8], &[u8])] = &[
        (b"dog", b"animal"), (b"cat", b"animal"), (b"cow", b"animal"),
        (b"horse", b"animal"), (b"bird", b"animal"), (b"fish", b"animal"),
        (b"apple", b"food"), (b"banana", b"food"), (b"bread", b"food"),
        (b"cake", b"food"), (b"milk", b"food"),
        (b"car", b"vehicle"), (b"truck", b"vehicle"), (b"bike", b"vehicle"),
        (b"plane", b"vehicle"), (b"boat", b"vehicle"),
        (b"red", b"color"), (b"blue", b"color"), (b"green", b"color"),
        (b"yellow", b"color"),
        (b"ball", b"toy"), (b"doll", b"toy"), (b"kite", b"toy"),
        (b"drum", b"toy"),
        (b"tree", b"nature"), (b"flower", b"nature"), (b"river", b"nature"),
        (b"mountain", b"nature"),
        (b"hand", b"body"), (b"foot", b"body"), (b"eye", b"body"),
        (b"mouth", b"body"),
    ];
    for &(prompt, response) in pairs {
        for _ in 0..8 {
            brain.observe(1, prompt);
            brain.observe(4, response);
            brain.advance_tick();
        }
    }

    brain.activate_for_prediction(1, b"dog");
    assert_eq!(
        brain.decode_best_trained_binding(1, 4),
        Some(b"animal".to_vec()),
        "a later curriculum must not make an exact earlier binding undecodable",
    );
}

#[test]
fn reordered_sequences_form_distinct_binding_episodes() {
    let (mut brain, input, output) = build_two_pool_brain();
    for _ in 0..3 {
        brain.observe(input, b"abc");
        brain.observe(output, b"first");
        brain.advance_tick();
    }
    for _ in 0..3 {
        brain.observe(input, b"cba");
        brain.observe(output, b"second");
        brain.advance_tick();
    }

    brain.activate_for_prediction(input, b"abc");
    assert_eq!(brain.decode_best_trained_binding(input, output), Some(b"first".to_vec()));
    brain.clear_prediction_activation();
    brain.activate_for_prediction(input, b"cba");
    assert_eq!(brain.decode_best_trained_binding(input, output), Some(b"second".to_vec()));
}

#[test]
fn direct_pretrain_binding_is_atom_grounded_without_within_pool_concept_birth() {
    let (mut brain, input, output) = build_two_pool_brain();
    brain.designate_action_pool(output);

    let binding = brain.pretrain_binding_episode(&[
        (input, b"write a greeting".to_vec()),
        (output, b"def greet():\n    return 'hello'\n".to_vec()),
    ]);
    assert!(binding.is_some());
    assert_eq!(brain.fabric().pool(input).unwrap().read().concept_count(), 0);
    assert_eq!(brain.fabric().pool(output).unwrap().read().concept_count(), 0);

    brain.activate_for_prediction(input, b"write a greeting");
    assert!(brain.has_exact_trained_binding(input, output));
    assert_eq!(
        brain.decode_best_trained_binding(input, output),
        Some(b"def greet():\n    return 'hello'\n".to_vec()),
    );
}

#[test]
fn direct_pretrain_distinguishes_long_reorderings_with_the_same_atom_inventory() {
    let (mut brain, input, output) = build_two_pool_brain();
    brain.designate_action_pool(output);

    let first = b"abcdefghijklmnopqrstuvwxyz 0123456789".to_vec();
    let second = b"9876543210 zyxwvutsrqponmlkjihgfedcba".to_vec();
    assert!(brain
        .pretrain_binding_episode(&[(input, first.clone()), (output, b"first".to_vec())])
        .is_some());
    assert!(brain
        .pretrain_binding_episode(&[(input, second.clone()), (output, b"second".to_vec())])
        .is_some());

    brain.activate_for_prediction(input, &first);
    assert_eq!(
        brain.decode_best_trained_binding(input, output),
        Some(b"first".to_vec())
    );
    brain.clear_prediction_activation();
    brain.activate_for_prediction(input, &second);
    assert_eq!(
        brain.decode_best_trained_binding(input, output),
        Some(b"second".to_vec())
    );
}
