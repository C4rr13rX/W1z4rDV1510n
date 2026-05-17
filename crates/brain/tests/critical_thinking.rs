//! Stage 8 (critical-thinking loop) tests.
//!
//! Verifies the EEM auto-population + grounded-fact chain explorer +
//! Brain::integrate_autonomous orchestrator that resolves unbounded
//! prompts into bounded math chains over the substrate's accumulated
//! world knowledge.

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

fn build_three_pool_brain() -> (Brain, u32, u32) {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    let mut brain = Brain::new(cfg);
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("action", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    (brain, pool_a, pool_b)
}

#[test]
fn binding_promotion_auto_registers_grounded_fact() {
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    assert_eq!(brain.eem_fact_count(), 0,
        "no facts before any binding emerges");

    // Train cross-pool X↔Y three times so a binding promotes.
    for _ in 0..5 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }

    assert!(brain.eem_fact_count() >= 1,
        "at least one grounded fact must auto-populate after binding \
         emergence; got {}", brain.eem_fact_count());
}

#[test]
fn facts_involving_returns_facts_for_a_given_concept() {
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    for _ in 0..5 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }

    // Find the X atom in pool_a.
    let pa = brain.fabric().pool(pool_a).unwrap();
    let x_id = pa.read().iter_neurons()
        .find(|n| n.is_atom() && n.label == "t:WA") // 'X' = 0x58 → "WA"
        .expect("X atom exists")
        .id;
    drop(pa);

    let facts = brain.eem().facts_involving(pool_a, x_id);
    assert!(!facts.is_empty(),
        "facts_involving must surface at least one fact for the X atom; got {}",
        facts.len());
    // The fact's members should include both the X atom in pool_a and
    // something in pool_b.
    let f = facts[0];
    assert!(f.members.iter().any(|(p, _)| *p == pool_a),
        "fact must include a pool_a member");
    assert!(f.members.iter().any(|(p, _)| *p == pool_b),
        "fact must include a pool_b member (cross-pool binding)");
}

#[test]
fn chain_explore_walks_from_seed_to_target_pool() {
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    for _ in 0..5 {
        brain.observe(pool_a, b"A");
        brain.observe(pool_b, b"B");
        brain.advance_tick();
    }

    let pa = brain.fabric().pool(pool_a).unwrap();
    let a_id = pa.read().iter_neurons()
        .find(|n| n.is_atom() && n.label == "t:QQ")
        .expect("A atom exists")
        .id;
    drop(pa);

    let result = brain.eem().chain_explore(&[(pool_a, a_id)], 3, 100);
    let reached_in_b: Vec<_> = result.reached_members.iter()
        .filter(|((p, _), _)| *p == pool_b)
        .collect();
    assert!(!reached_in_b.is_empty(),
        "chain_explore from A in pool_a must reach at least one ref in \
         pool_b via the grounded fact graph; got {:?}", result);
    assert!(!result.visited_facts.is_empty(),
        "must record visited facts");
}

#[test]
fn out_of_vocabulary_prompt_returns_outside_grounding_not_hallucination() {
    // After training on small toddler-category vocab, a query whose
    // atoms only WEAKLY overlap any trained binding (e.g. "Hello"
    // against trained X→Y, A→B) must be honestly flagged as
    // outside_grounding rather than picking the strongest random
    // match.  This is the Stage 8 honesty gate.

    let (mut brain, pool_a, pool_b) = build_three_pool_brain();

    // Train two pairs whose atoms overlap with "Hello" only partially.
    for _ in 0..5 {
        brain.observe(pool_a, b"yellow");
        brain.observe(pool_b, b"color");
        brain.advance_tick();
        brain.observe(pool_a, b"red");
        brain.observe(pool_b, b"color");
        brain.advance_tick();
    }

    // Trained query: should ground.
    brain.observe(pool_a, b"yellow");
    let trained = brain.integrate_autonomous(pool_a, pool_b, 0.0, 4, 200);
    assert!(!trained.grounding.outside_grounding,
        "'yellow' is a trained prompt; must NOT be outside_grounding");
    assert!(trained.answer.is_some(),
        "trained prompt must produce an answer");

    // Out-of-vocab query: should reject.
    brain.observe(pool_a, b"Hello");
    let oov = brain.integrate_autonomous(pool_a, pool_b, 0.0, 4, 200);
    assert!(oov.grounding.outside_grounding,
        "'Hello' shares only weak atom overlap with trained \
         bindings — must be flagged outside_grounding; got {:?}",
        oov.grounding);
    assert!(oov.answer.is_none(),
        "OOV prompt must return None answer; got {:?}", oov.answer);
}

#[test]
fn best_binding_match_high_for_trained_query_low_for_oov() {
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    for _ in 0..5 {
        brain.observe(pool_a, b"dog");
        brain.observe(pool_b, b"animal");
        brain.advance_tick();
    }

    // Trained query → high precision.
    brain.observe(pool_a, b"dog");
    let (prec_trained, recall_trained) = brain.best_binding_match(pool_a);
    assert!(prec_trained >= 0.99,
        "exact-trained query must yield precision >= 0.99; got {}", prec_trained);
    assert!(recall_trained >= 0.99,
        "exact-trained query must yield recall >= 0.99; got {}", recall_trained);

    // Out-of-vocab query → low precision.
    brain.observe(pool_a, b"xyz");
    let (prec_oov, _) = brain.best_binding_match(pool_a);
    assert!(prec_oov < 0.5,
        "OOV query must yield precision < 0.5; got {}", prec_oov);
}

#[test]
fn integrate_autonomous_returns_fabric_answer_when_confidence_high() {
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    // Heavy training so fabric confidence is high.
    for _ in 0..10 {
        brain.observe(pool_a, b"P");
        brain.observe(pool_b, b"Q");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"P");

    let ans = brain.integrate_autonomous(pool_a, pool_b,
        /*threshold*/ 0.0,        // accept whatever fabric gives
        /*max_depth*/ 3,
        /*max_visit*/ 50);
    // Fabric path should produce an answer here.
    assert!(ans.answer.is_some(),
        "fabric path must produce an answer for trained pair; got {:?}", ans);
}

#[test]
fn integrate_autonomous_does_not_double_gate_via_inner_integrate_oog() {
    // Regression pin for Stage 9.1: when step-0's binding-precision
    // gate passes (query is in-vocab), integrate_autonomous must NOT
    // also re-gate via the inner self.integrate()'s outside_grounding
    // flag.  That inner flag is computed with a stricter fabric-
    // confidence threshold and can flip to true on legitimately
    // trained pairs when activation has decayed across many ticks —
    // which would push the call into chain_explore unnecessarily and
    // return a degraded single-byte answer instead of the trained
    // cross-pool answer.
    //
    // Concretely: a trained (dog, animal) pair should still return
    // "animal" from integrate_autonomous even if the inner integrate
    // flagged outside_grounding=true on its own confidence floor —
    // the step-0 gate is the single source of truth.
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    for _ in 0..5 {
        brain.observe(pool_a, b"dog");
        brain.observe(pool_b, b"animal");
        brain.advance_tick();
    }
    // Simulate many subsequent ticks of activity so activation decays —
    // this is what nudges the inner integrate to set outside_grounding
    // on the query in real long-running deployments.
    for _ in 0..50 {
        brain.observe(pool_a, b"x");
        brain.advance_tick();
    }
    // Now ask the trained question.
    brain.observe(pool_a, b"dog");
    let ans = brain.integrate_autonomous(pool_a, pool_b, 0.0, 4, 200);
    // Step-0 gate should pass (precision high for exact-trained query).
    assert!(!ans.grounding.outside_grounding,
        "trained dog→animal must not be flagged outside_grounding by integrate_autonomous \
         even after decay; got {:?}", ans.grounding);
    // The returned answer should contain "animal" (the trained target).
    let answer_bytes = ans.answer.as_ref().expect("answer should be Some");
    let answer_str = String::from_utf8_lossy(answer_bytes);
    assert!(answer_str.contains("animal"),
        "trained dog→animal must return 'animal' (not a degraded char_chain fragment); \
         got {:?}", answer_str);
}

#[test]
fn integrate_autonomous_falls_through_to_chain_when_fabric_low() {
    // Train cross-pool so facts populate, but use a very high fabric
    // threshold so the chain path runs and produces the answer.
    let (mut brain, pool_a, pool_b) = build_three_pool_brain();
    for _ in 0..5 {
        brain.observe(pool_a, b"K");
        brain.observe(pool_b, b"V");
        brain.advance_tick();
    }
    // Now query (with K firing in text).
    brain.observe(pool_a, b"K");
    let ans = brain.integrate_autonomous(pool_a, pool_b,
        /*threshold*/ 100.0,      // force chain path
        /*max_depth*/ 3,
        /*max_visit*/ 100);
    // Chain path should have populated eem_confidence.
    assert!(ans.grounding.eem_confidence.is_some(),
        "chain path must set eem_confidence; got {:?}", ans.grounding.eem_confidence);
    assert!(ans.answer.is_some(),
        "chain path must produce an answer; got {:?}", ans);
}
