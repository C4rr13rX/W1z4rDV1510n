use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, InstructionIntentEncoding,
    PoolConfig,
};

#[test]
fn ranked_feature_readout_returns_independent_grounded_actions() {
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

    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let labels = encoding
        .atomize(b"Build Python structured correlation-ID logging and default-deny authorization.");
    let decoded = brain.decode_ranked_feature_bindings(feature_pool, &labels, action_pool, 8);
    assert_eq!(
        decoded.len(),
        2,
        "expected both independently grounded actions"
    );
    assert!(decoded.iter().any(|bytes| {
        bytes
            .windows(b"observability.py".len())
            .any(|window| window == b"observability.py")
    }));
    assert!(decoded.iter().any(|bytes| {
        bytes
            .windows(b"authorization.py".len())
            .any(|window| window == b"authorization.py")
    }));
}

#[test]
fn character_motif_route_generalizes_raw_phrase_to_learned_intent() {
    let mut brain = Brain::new(BrainConfig::default());
    let raw_pool = brain.create_pool(
        PoolConfig::defaults("raw", 1),
        Box::new(BytePassthroughEncoding { prefix: "raw" }),
    );
    let intent_pool = brain.create_pool(
        PoolConfig::defaults("intent", 2),
        Box::new(InstructionIntentEncoding {
            prefix: "intent".into(),
        }),
    );
    let trained = b"Build Python rules where administrators may change anything and people may view records they own.";
    let intent = b"@intent:LANGUAGE:PYTHON\n@intent:SECURITY:AUTHORIZATION\n";
    for _ in 0..6 {
        brain.observe(raw_pool, trained);
        brain.observe(intent_pool, intent);
        brain.advance_tick();
    }
    let heldout = b"Develop Python permissions where administrators act freely while users inspect their own records.";
    let decoded = brain
        .decode_best_binding_by_char_motifs(raw_pool, heldout, intent_pool, 0.20)
        .expect("character motifs should reinstate the learned intent frame");
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let labels = encoding.atomize(&decoded);
    assert!(labels.iter().any(|label| label == "intent:LANGUAGE:PYTHON"));
    assert!(
        labels
            .iter()
            .any(|label| label == "intent:SECURITY:AUTHORIZATION")
    );
}

#[test]
fn exact_feature_readout_rejects_partial_binding_candidates() {
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
    let complete = b"@intent:LANGUAGE:PYTHON\n@intent:OBSERVABILITY:CORRELATED_LOGGING\n@intent:ENTERPRISE:SECRET_REDACTION\n";
    let partial = b"@intent:LANGUAGE:PYTHON\n@intent:ENTERPRISE:SECRET_REDACTION\n";
    for _ in 0..6 {
        brain.observe(feature_pool, complete);
        brain.observe(action_pool, br#"{"files":{"observability.py":"COMPLETE"}}"#);
        brain.advance_tick();
        brain.observe(feature_pool, partial);
        brain.observe(action_pool, br#"{"files":{"redaction.py":"PARTIAL"}}"#);
        brain.advance_tick();
    }
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(complete);
    let decoded = brain
        .decode_exact_feature_binding(feature_pool, &labels, action_pool)
        .expect("complete feature binding should be available");
    assert!(
        decoded
            .windows(b"observability.py".len())
            .any(|window| window == b"observability.py")
    );
}

#[test]
fn close_richer_intent_class_integrates_subset_evidence() {
    let mut brain = Brain::new(BrainConfig::default());
    let raw_pool = brain.create_pool(
        PoolConfig::defaults("raw", 1),
        Box::new(BytePassthroughEncoding { prefix: "raw" }),
    );
    let intent_pool = brain.create_pool(
        PoolConfig::defaults("intent", 2),
        Box::new(InstructionIntentEncoding { prefix: "intent".into() }),
    );
    let routes: [(&[u8], &[u8]); 2] = [
        (
            b"Implement a Python function redact_secrets that recursively redacts password, token, and api_key values.",
            b"@intent:LANGUAGE:PYTHON\n@intent:ENTERPRISE:SECRET_REDACTION\n",
        ),
        (
            b"Create Python audit output that attaches a request trace to every record and scrubs credentials at any nesting depth.",
            b"@intent:LANGUAGE:PYTHON\n@intent:OBSERVABILITY:CORRELATED_LOGGING\n@intent:ENTERPRISE:SECRET_REDACTION\n",
        ),
    ];
    for _ in 0..6 {
        for (prompt, intent) in routes {
            brain.observe(raw_pool, prompt);
            brain.observe(intent_pool, intent);
            brain.advance_tick();
        }
    }
    let query = b"Develop Python audit entries with request tracking that recursively remove passwords and tokens.";
    let (decoded, _, _) = brain
        .decode_best_binding_by_char_motifs_with_margin(raw_pool, query, intent_pool, 0.20, 0.0)
        .expect("a close richer intent should integrate the subset class");
    let labels = InstructionIntentEncoding { prefix: "intent".into() }.atomize(&decoded);
    assert!(labels.iter().any(|label| label == "intent:OBSERVABILITY:CORRELATED_LOGGING"));
    assert!(labels.iter().any(|label| label == "intent:ENTERPRISE:SECRET_REDACTION"));
}
