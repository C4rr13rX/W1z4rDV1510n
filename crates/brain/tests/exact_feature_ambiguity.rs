use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, InstructionIntentEncoding,
    PoolConfig,
};

fn configured_brain() -> (Brain, u32, u32) {
    let mut brain = Brain::new(BrainConfig::default());
    let feature_pool = brain.create_pool(
        PoolConfig::defaults("intent", 1),
        Box::new(InstructionIntentEncoding {
            prefix: "intent".into(),
        }),
    );
    let action_pool = brain.create_pool(
        PoolConfig::defaults("action", 2),
        Box::new(BytePassthroughEncoding { prefix: "action" }),
    );
    (brain, feature_pool, action_pool)
}

#[test]
fn exact_feature_readout_rejects_conflicting_targets_for_the_same_evidence() {
    let (mut brain, feature_pool, action_pool) = configured_brain();
    let features = b"@intent:LANGUAGE:PYTHON\n@intent:ENTERPRISE:SECRET_REDACTION\n";
    for _ in 0..6 {
        brain.observe(feature_pool, features);
        brain.observe(action_pool, b"def redact_secrets(value): return value");
        brain.advance_tick();
        brain.observe(feature_pool, features);
        brain.observe(
            action_pool,
            br#"{"files":{"observability.py":"def _redact(value): return value"}}"#,
        );
        brain.advance_tick();
    }
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(features);
    assert_eq!(
        brain.decode_exact_feature_binding(feature_pool, &labels, action_pool),
        None,
        "shared sparse features do not contain enough evidence to choose between targets"
    );
}

#[test]
fn exact_feature_readout_keeps_duplicate_evidence_for_one_target() {
    let (mut brain, feature_pool, action_pool) = configured_brain();
    let features = b"@intent:LANGUAGE:PYTHON\n@intent:ENTERPRISE:INPUT_VALIDATION\n";
    let response = b"def validate(value): return bool(value)";
    for _ in 0..12 {
        brain.observe(feature_pool, features);
        brain.observe(action_pool, response);
        brain.advance_tick();
    }
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(features);
    assert_eq!(
        brain
            .decode_exact_feature_binding(feature_pool, &labels, action_pool)
            .as_deref(),
        Some(response.as_slice())
    );
}
