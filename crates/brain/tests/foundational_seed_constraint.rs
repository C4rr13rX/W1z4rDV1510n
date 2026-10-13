//! Pre-training seed constraint verification test.
//!
//! Asserts:
//! 1. The 10 Commandments with their framed introduction recall at 100%
//!    immediately after seeding prior to training.
//! 2. The other commandments Jesus Christ spoke of recall at 100%.
//! 3. 100% recall of foundational commandments is retained alongside subsequent training.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    FOUNDATIONAL_SEED_PAIRS, seed_commandments_prior_to_training,
};

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

#[test]
fn foundational_seed_100_percent_recall_prior_to_training() {
    let mut brain = subject();
    let report = seed_commandments_prior_to_training(&mut brain, QUERY_POOL, ANSWER_POOL);
    assert_eq!(report.pairs_seeded, FOUNDATIONAL_SEED_PAIRS.len());
    assert!(report.pairs_seeded >= 20);

    for pair in FOUNDATIONAL_SEED_PAIRS {
        brain.observe_read_only(QUERY_POOL, pair.question.as_bytes());
        let answer = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
        let _ = brain.finish_read_only_inference();
        assert_eq!(
            answer.as_deref(),
            Some(pair.answer.as_bytes()),
            "Failed recall for question: {}",
            pair.question
        );
    }
}

#[test]
fn foundational_seed_retained_alongside_subsequent_training() {
    let mut brain = subject();
    seed_commandments_prior_to_training(&mut brain, QUERY_POOL, ANSWER_POOL);

    // Subsequent scene/fact training
    for r in 0..16 {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{r:03} lamp on").into_bytes()),
            (ANSWER_POOL, b"desk".to_vec()),
        ]);
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{r:03} desk material").into_bytes()),
            (ANSWER_POOL, b"oak".to_vec()),
        ]);
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{r:03} chair color").into_bytes()),
            (ANSWER_POOL, b"red".to_vec()),
        ]);
    }

    // Verify 100% recall of foundational commandments is completely retained
    for pair in FOUNDATIONAL_SEED_PAIRS {
        brain.observe_read_only(QUERY_POOL, pair.question.as_bytes());
        let answer = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
        let _ = brain.finish_read_only_inference();
        assert_eq!(
            answer.as_deref(),
            Some(pair.answer.as_bytes()),
            "Failed retained recall for question: {}",
            pair.question
        );
    }
}
