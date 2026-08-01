use w1z4rd_brain::{Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

fn pool(brain: &mut Brain, id: u32, name: &str, prefix: &'static str) -> u32 {
    let mut config = PoolConfig::defaults(name, id);
    config.max_concept_member_count = 1;
    brain.create_pool(config, Box::new(BytePassthroughEncoding { prefix }))
}

#[test]
fn joint_readout_uses_console_stream_to_disambiguate_same_instruction() {
    let mut config = BrainConfig::default();
    config.binding_emergence_threshold = 3;
    config.tentative_emergence_threshold = 3;
    let mut brain = Brain::new(config);
    let instruction = pool(&mut brain, 1, "instruction", "i");
    let console = pool(&mut brain, 2, "console", "c");
    let repair = pool(&mut brain, 3, "repair", "r");

    for _ in 0..3 {
        brain.observe(instruction, b"repair the function");
        brain.observe(console, b"TypeError");
        brain.observe(repair, b"convert input to int");
        brain.advance_tick();
    }
    for _ in 0..3 {
        brain.observe(instruction, b"repair the function");
        brain.observe(console, b"IndexError");
        brain.observe(repair, b"check the list bounds");
        brain.advance_tick();
    }

    brain.activate_for_prediction(instruction, b"repair the function");
    brain.activate_for_prediction(console, b"TypeError");
    assert_eq!(
        brain.decode_best_trained_binding_multi(&[instruction, console], repair),
        Some(b"convert input to int".to_vec()),
    );
    brain.clear_prediction_activation();

    brain.activate_for_prediction(instruction, b"repair the function");
    brain.activate_for_prediction(console, b"IndexError");
    assert_eq!(
        brain.decode_best_trained_binding_multi(&[instruction, console], repair),
        Some(b"check the list bounds".to_vec()),
    );
}

#[test]
fn joint_readout_rejects_a_binding_missing_a_requested_evidence_pool() {
    let mut config = BrainConfig::default();
    config.binding_emergence_threshold = 3;
    config.tentative_emergence_threshold = 3;
    let mut brain = Brain::new(config);
    let instruction = pool(&mut brain, 1, "instruction", "i");
    let console = pool(&mut brain, 2, "console", "c");
    let environment = pool(&mut brain, 4, "environment", "e");
    let repair = pool(&mut brain, 3, "repair", "r");
    for _ in 0..3 {
        brain.observe(instruction, b"repair it");
        brain.observe(console, b"TypeError");
        brain.observe(repair, b"convert input");
        brain.advance_tick();
    }
    brain.activate_for_prediction(instruction, b"repair it");
    brain.activate_for_prediction(console, b"TypeError");
    brain.activate_for_prediction(environment, b"python=3.13");
    assert_eq!(
        brain.decode_best_trained_binding_multi(&[instruction, console, environment], repair),
        None,
    );
}

#[test]
fn joint_readout_reports_bounded_evidence_confidence_separately_from_repetition() {
    let mut config = BrainConfig::default();
    config.binding_emergence_threshold = 3;
    config.tentative_emergence_threshold = 3;
    let mut brain = Brain::new(config);
    let ohlcv = pool(&mut brain, 1, "ohlcv", "o");
    let news = pool(&mut brain, 2, "news", "n");
    let outcome = pool(&mut brain, 3, "outcome", "f");
    for _ in 0..8 {
        brain.observe(ohlcv, b"trend=up volatility=low");
        brain.observe(news, b"sentiment=positive");
        brain.observe(outcome, b"future rally");
        brain.advance_tick();
    }

    brain.activate_for_prediction(ohlcv, b"trend=up volatility=low");
    brain.activate_for_prediction(news, b"sentiment=positive");
    let (answer, confidence) = brain
        .decode_best_trained_binding_multi_scored(&[ohlcv, news], outcome)
        .expect("exact multi-pool market episode should decode");
    assert_eq!(answer, b"future rally");
    assert!((0.0..=1.0).contains(&confidence));
    assert_eq!(confidence, 1.0);
}
