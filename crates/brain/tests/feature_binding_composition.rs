use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding,
    InstructionIntentEncoding, PoolConfig,
};

#[test]
fn ranked_feature_readout_returns_independent_grounded_actions() {
    let mut brain = Brain::new(BrainConfig::default());
    let feature_pool = brain.create_pool(
        PoolConfig::defaults("intent", 1),
        Box::new(InstructionIntentEncoding { prefix: "intent".into() }),
    );
    let action_pool = brain.create_pool(
        PoolConfig::defaults("action", 2),
        Box::new(BytePassthroughEncoding { prefix: "action" }),
    );
    let examples: [(&[u8], &[u8]); 2] = [
        (
            b"Write Python structured logging with a correlation ID.",
            br#"{"files":{"observability.py":"LOG"}}"#,
        ),
        (
            b"Implement Python default-deny authorization.",
            br#"{"files":{"authorization.py":"AUTH"}}"#,
        ),
    ];
    for _ in 0..6 {
        for (prompt, action) in examples {
            brain.observe(feature_pool, prompt);
            brain.observe(action_pool, action);
            brain.advance_tick();
        }
    }

    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let labels = encoding.atomize(
        b"Build Python structured correlation-ID logging and default-deny authorization.",
    );
    let decoded = brain.decode_ranked_feature_bindings(feature_pool, &labels, action_pool, 8);
    assert_eq!(decoded.len(), 2, "expected both independently grounded actions");
    assert!(decoded.iter().any(|bytes| bytes.windows(b"observability.py".len())
        .any(|window| window == b"observability.py")));
    assert!(decoded.iter().any(|bytes| bytes.windows(b"authorization.py".len())
        .any(|window| window == b"authorization.py")));
}
