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
fn language_and_behavior_can_recall_one_extra_learned_constraint() {
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
    let learned = b"@intent:LANGUAGE:TYPESCRIPT\n@intent:API:IDEMPOTENT_COMMAND\n@intent:ENTERPRISE:INPUT_VALIDATION\n";
    let response = br#"{"files":{"orders.ts":"VALIDATED IDEMPOTENT SERVICE"}}"#;
    for _ in 0..6 {
        brain.observe(feature_pool, learned);
        brain.observe(action_pool, response);
        brain.advance_tick();
    }
    let query = b"@intent:LANGUAGE:TYPESCRIPT\n@intent:API:IDEMPOTENT_COMMAND\n";
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(query);
    let decoded = brain.decode_ranked_feature_bindings(feature_pool, &labels, action_pool, 8);
    assert_eq!(
        decoded.first().map(Vec::as_slice),
        Some(response.as_slice())
    );
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
        Box::new(InstructionIntentEncoding {
            prefix: "intent".into(),
        }),
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
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(&decoded);
    assert!(
        labels
            .iter()
            .any(|label| label == "intent:OBSERVABILITY:CORRELATED_LOGGING")
    );
    assert!(
        labels
            .iter()
            .any(|label| label == "intent:ENTERPRISE:SECRET_REDACTION")
    );
}

#[test]
fn confirmed_outcome_promotes_repair_and_inhibits_failed_action_frame() {
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
    let failure_pool = brain.create_pool(
        PoolConfig::defaults("failure", 6),
        Box::new(BytePassthroughEncoding { prefix: "failure" }),
    );
    let success_pool = brain.create_pool(
        PoolConfig::defaults("success", 8),
        Box::new(BytePassthroughEncoding { prefix: "success" }),
    );
    let prompt = b"@intent:LANGUAGE:JAVASCRIPT\n@intent:STATE:INCREMENT_COUNT\n@intent:CODE:FUNCTION_SIGNATURE\n";
    let bad = b"return value - 1";
    let good = b"return value + 1";
    for _ in 0..6 {
        brain.observe(feature_pool, prompt);
        brain.observe(action_pool, bad);
        brain.advance_tick();
    }
    for _ in 0..6 {
        brain.observe(feature_pool, prompt);
        brain.observe(action_pool, bad);
        brain.observe(failure_pool, b"wrong_result");
        brain.advance_tick();
    }
    for _ in 0..6 {
        brain.observe(feature_pool, prompt);
        brain.observe(action_pool, good);
        brain.observe(success_pool, b"PASS");
        brain.advance_tick();
    }
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(prompt);
    let decoded = brain.decode_ranked_feature_bindings_with_outcomes(
        feature_pool,
        &labels,
        action_pool,
        8,
        Some(success_pool),
        Some(failure_pool),
    );
    assert_eq!(decoded.first().map(Vec::as_slice), Some(good.as_slice()));
    assert!(!decoded.iter().any(|frame| frame == bad));
}

#[test]
fn ranked_feature_readout_preserves_selection_evidence() {
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
    let failure_pool = brain.create_pool(
        PoolConfig::defaults("failure", 6),
        Box::new(BytePassthroughEncoding { prefix: "failure" }),
    );
    let success_pool = brain.create_pool(
        PoolConfig::defaults("success", 8),
        Box::new(BytePassthroughEncoding { prefix: "success" }),
    );
    let intent = b"@intent:LANGUAGE:RUST\n@intent:PERSISTENCE:ATOMIC_TRANSACTION\n@intent:DOMAIN:ATOMIC_LEDGER_TRANSFER\n";
    let rejected = b"fn transfer() { /* incomplete */ }";
    let accepted = b"fn transfer() { debit(); credit(); commit(); }";
    for _ in 0..4 {
        brain.observe(feature_pool, intent);
        brain.observe(action_pool, rejected);
        brain.observe(failure_pool, b"compile_error");
        brain.advance_tick();
    }
    for _ in 0..6 {
        brain.observe(feature_pool, intent);
        brain.observe(action_pool, accepted);
        brain.observe(success_pool, b"PASS");
        brain.advance_tick();
    }

    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(intent);
    let decoded = brain.decode_ranked_feature_bindings_with_evidence(
        feature_pool,
        &labels,
        action_pool,
        8,
        Some(success_pool),
        Some(failure_pool),
        &[feature_pool],
        &[],
    );
    let winner = decoded.first().expect("successful action should be recalled");
    assert_eq!(winner.bytes, accepted);
    assert!(winner.outcome_score > 0);
    assert!(winner.use_count > 0);
    assert!(winner.target_size > 0);
    assert!(winner.learned_feature_count >= 3);
    assert!(
        winner
            .matched_labels
            .iter()
            .any(|label| label.ends_with(":LANGUAGE:RUST"))
    );
    assert!(
        winner
            .matched_labels
            .iter()
            .any(|label| label.ends_with(":PERSISTENCE:ATOMIC_TRANSACTION"))
    );
    assert!(
        winner
            .matched_labels
            .iter()
            .any(|label| label.ends_with(":DOMAIN:ATOMIC_LEDGER_TRANSFER"))
    );
    assert!(!decoded.iter().any(|candidate| candidate.bytes == rejected));
}

#[test]
fn deterministic_validator_filters_before_bounded_feature_readout() {
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
    let intent =
        b"@intent:LANGUAGE:PYTHON\n@intent:PARITY:ODD\n";
    let popular_but_wrong =
        b"def median(values, count):\n    return values[count // 2] if count % 2 else 0";
    let rarer_but_valid =
        b"def filter_odd(values):\n    return [value for value in values if value % 2]";
    for _ in 0..12 {
        brain.observe(feature_pool, intent);
        brain.observe(action_pool, popular_but_wrong);
        brain.advance_tick();
    }
    for _ in 0..3 {
        brain.observe(feature_pool, intent);
        brain.observe(action_pool, rarer_but_valid);
        brain.advance_tick();
    }
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(intent);
    assert_eq!(
        brain
            .decode_ranked_feature_bindings(feature_pool, &labels, action_pool, 1)
            .first()
            .map(Vec::as_slice),
        Some(popular_but_wrong.as_slice())
    );
    let filtered = brain.decode_first_ranked_feature_binding_with_context_where(
        feature_pool,
        &labels,
        action_pool,
        None,
        None,
        &[feature_pool],
        &[],
        &|bytes| bytes.windows(b"filter_odd".len()).any(|part| part == b"filter_odd"),
    );
    assert_eq!(filtered.as_deref(), Some(rarer_but_valid.as_slice()));
}

#[test]
fn context_conditioned_corpus_action_cannot_override_context_free_rule() {
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
    let context_pool = brain.create_pool(
        PoolConfig::defaults("environment", 5),
        Box::new(BytePassthroughEncoding {
            prefix: "environment",
        }),
    );
    let intent = b"@intent:LANGUAGE:PYTHON\n@intent:MATH:AVERAGE\n@intent:GUARD:EMPTY_INPUT\n";
    let generic = b"def avg_list(xs):\n    return sum(xs) / len(xs) if xs else 0";
    let contextual = b"answer = (1.0 + 11.0) / 2.0\nprint(answer)";
    for _ in 0..6 {
        brain.observe(feature_pool, intent);
        brain.observe(action_pool, generic);
        brain.advance_tick();
        brain.observe(feature_pool, intent);
        brain.observe(context_pool, br#"{"kind":"math"}"#);
        brain.observe(action_pool, contextual);
        brain.advance_tick();
    }
    let labels = InstructionIntentEncoding {
        prefix: "intent".into(),
    }
    .atomize(intent);
    let decoded = brain.decode_ranked_feature_bindings_with_context(
        feature_pool,
        &labels,
        action_pool,
        8,
        None,
        None,
        &[feature_pool],
        &[context_pool],
    );
    assert_eq!(decoded.first().map(Vec::as_slice), Some(generic.as_slice()));
    assert!(!decoded.iter().any(|frame| frame == contextual));
}
