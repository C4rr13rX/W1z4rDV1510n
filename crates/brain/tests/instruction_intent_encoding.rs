use w1z4rd_brain::{AtomEncoding, InstructionIntentEncoding};

#[test]
fn square_paraphrases_share_intent_but_cube_is_distinct() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let square_a = encoding.atomize(b"Fix square so it returns a number multiplied by itself.");
    let square_b =
        encoding.atomize(b"Make square compute the product of its argument with itself.");
    let cube = encoding.atomize(b"Make cube calculate the third power of its input.");
    assert_eq!(square_a, square_b);
    assert_ne!(square_a, cube);
}

#[test]
fn unsupported_intents_do_not_alias_trained_power_features() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let factorial = encoding.atomize(b"Fix factorial so it computes factorial.");
    let square = encoding.atomize(b"Fix square so it multiplies by itself.");
    assert_ne!(factorial, square);
}

#[test]
fn bare_k12_concept_does_not_fire_a_coding_intent() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    assert!(encoding.atomize(b"square").is_empty());
    assert!(
        !encoding
            .atomize(b"Write a function that returns a square.")
            .is_empty()
    );
}

#[test]
fn word_count_paraphrases_share_state_intent() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let trained = encoding.atomize(b"Implement a function that returns a dict of word -> count.");
    let novel = encoding.atomize(b"Produce a function mapping every word to its occurrence count.");
    assert_eq!(trained, novel);
    assert!(!trained.is_empty());
}

#[test]
fn repair_verb_paraphrases_retain_the_same_intent() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let fixed = encoding.atomize(b"Fix is_negative so values below zero are recognized.");
    let corrected = encoding.atomize(b"Correct is_negative to recognize values less than zero.");
    assert_eq!(fixed, corrected);
    assert!(!fixed.is_empty());
}

#[test]
fn language_and_task_are_independent_cofiring_features() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let rust = encoding.atomize(b"Create Rust code computing the second power of a number.");
    assert!(rust.iter().any(|label| label == "intent:LANGUAGE:RUST"));
    assert!(rust.iter().any(|label| label == "intent:POWER_SELF:2"));

    let javascript = encoding.atomize(b"Write a JavaScript function that returns a square.");
    assert!(
        javascript
            .iter()
            .any(|label| label == "intent:LANGUAGE:JAVASCRIPT")
    );
    assert!(
        !javascript
            .iter()
            .any(|label| label == "intent:LANGUAGE:JAVA")
    );
}

#[test]
fn average_paraphrases_share_language_and_task_features() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let average = encoding.atomize(b"Implement avg_list in Python to return the average.");
    let mean = encoding.atomize(b"Write Python that calculates the arithmetic mean.");
    assert_eq!(average, mean);
    assert!(average.iter().any(|label| label == "intent:MATH:AVERAGE"));
}

#[test]
fn enterprise_paraphrases_emit_independent_behavior_features() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let cases = [
        (
            b"Implement Python input validation for required user fields.".as_slice(),
            "intent:ENTERPRISE:INPUT_VALIDATION",
        ),
        (
            b"Write Python with bounded retry attempts for transient failures.".as_slice(),
            "intent:ENTERPRISE:BOUNDED_RETRY",
        ),
        (
            b"Create Python code to summarize JSON order totals.".as_slice(),
            "intent:ENTERPRISE:JSON_AGGREGATION",
        ),
        (
            b"Build Python code to recursively mask secrets and tokens.".as_slice(),
            "intent:ENTERPRISE:SECRET_REDACTION",
        ),
        (
            b"Write a Python function to chunk records into batches.".as_slice(),
            "intent:ENTERPRISE:BATCHING",
        ),
    ];
    for (prompt, expected) in cases {
        let features = encoding.atomize(prompt);
        assert!(
            features.iter().any(|label| label == expected),
            "{features:?}"
        );
        assert!(
            features
                .iter()
                .any(|label| label == "intent:LANGUAGE:PYTHON")
        );
    }
}

#[test]
fn project_level_intents_are_compositional_and_distinct() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let cases = [
        (
            b"Build a Python multi-file domain and service project.".as_slice(),
            "intent:ARCHITECTURE:MULTIFILE_SERVICE",
        ),
        (
            b"Implement a Python SQLite atomic transfer transaction.".as_slice(),
            "intent:PERSISTENCE:ATOMIC_TRANSACTION",
        ),
        (
            b"Write Python async code with bounded concurrency.".as_slice(),
            "intent:CONCURRENCY:BOUNDED_ASYNC",
        ),
        (
            b"Create Python default-deny authorization logic.".as_slice(),
            "intent:SECURITY:AUTHORIZATION",
        ),
    ];
    for (prompt, expected) in cases {
        let features = encoding.atomize(prompt);
        assert!(
            features
                .iter()
                .any(|label| label == "intent:LANGUAGE:PYTHON")
        );
        assert!(
            features.iter().any(|label| label == expected),
            "{features:?}"
        );
        assert!(
            features.len() >= 2,
            "project prompt should emit language plus compositional evidence"
        );
    }
}

#[test]
fn authorization_paraphrase_retains_security_intent() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let trained = encoding.atomize(b"Implement Python default-deny authorization logic.");
    let paraphrase = encoding.atomize(
        b"Create Python access-control code that denies by default and permits administrators.",
    );
    assert_eq!(trained, paraphrase);
    assert!(
        trained
            .iter()
            .any(|label| label == "intent:SECURITY:AUTHORIZATION")
    );
}

#[test]
fn platform_intents_emit_language_plus_protocol_behavior() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let cases = [
        (
            b"Implement a Python idempotent API command.".as_slice(),
            "intent:API:IDEMPOTENT_COMMAND",
        ),
        (
            b"Write Python schema migration upgrade paths.".as_slice(),
            "intent:PERSISTENCE:SCHEMA_MIGRATION",
        ),
        (
            b"Create Python structured logs with a correlation ID.".as_slice(),
            "intent:OBSERVABILITY:CORRELATED_LOGGING",
        ),
        (
            b"Build a Python circuit breaker.".as_slice(),
            "intent:RESILIENCE:CIRCUIT_BREAKER",
        ),
    ];
    for (prompt, expected) in cases {
        let features = encoding.atomize(prompt);
        assert!(
            features
                .iter()
                .any(|label| label == "intent:LANGUAGE:PYTHON")
        );
        assert!(
            features.iter().any(|label| label == expected),
            "{features:?}"
        );
    }
}

#[test]
fn atomic_ledger_is_both_transactional_and_domain_specific() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding.atomize(
        b"Build a Go ledger transfer that preserves both balances on failure.",
    );
    assert!(features.iter().any(|label| label == "intent:LANGUAGE:GO"));
    assert!(features
        .iter()
        .any(|label| label == "intent:PERSISTENCE:ATOMIC_TRANSACTION"));
    assert!(features
        .iter()
        .any(|label| label == "intent:DOMAIN:ATOMIC_LEDGER_TRANSFER"));
    assert!(features
        .iter()
        .any(|label| label == "intent:ENTERPRISE:INPUT_VALIDATION"));
    let unique: std::collections::HashSet<_> = features.iter().collect();
    assert_eq!(unique.len(), features.len(), "intent atoms must be unique");
}

#[test]
fn fragment_constraints_cofire_without_replacing_raw_source() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding.atomize(
        b"Create a Python clamp function that floors values at a minimum, caps them at a maximum, and otherwise leaves the value unchanged.",
    );
    for expected in [
        "intent:LANGUAGE:PYTHON",
        "intent:CODE:CLAMP",
        "intent:CODE:FUNCTION_SIGNATURE",
        "intent:GUARD:LOWER_BOUND",
        "intent:GUARD:UPPER_BOUND",
        "intent:FLOW:RETURN_INPUT",
    ] {
        assert!(features.iter().any(|label| label == expected), "{features:?}");
    }
}

#[test]
fn missing_specification_emits_inhibitory_grounding_evidence() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding
        .atomize(b"Write a Python migration for an unspecified legacy schema and target schema.");
    assert!(
        features
            .iter()
            .any(|label| label == "intent:PERSISTENCE:SCHEMA_MIGRATION")
    );
    assert!(
        features
            .iter()
            .any(|label| label == "intent:GROUNDING:UNDERSPECIFIED")
    );
    for prompt in [
        b"Write Python code that migrates an unknown production database schema.".as_slice(),
        b"Implement a Python API client for an endpoint whose protocol is unknown.".as_slice(),
    ] {
        let features = encoding.atomize(prompt);
        assert!(
            features
                .iter()
                .any(|label| label == "intent:GROUNDING:UNDERSPECIFIED"),
            "missing inhibition for {features:?}"
        );
    }
}

#[test]
fn circuit_breaker_failure_cooldown_paraphrase_shares_intent() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let trained = encoding.atomize(b"Implement a Python circuit breaker.");
    let paraphrase = encoding.atomize(
        b"Build Python resilience code that opens after repeated failures and permits a trial after its cooldown.",
    );
    assert_eq!(trained, paraphrase);
    let compact = encoding.atomize(b"Create Python resilience with a cooldown circuit.");
    assert!(
        compact
            .iter()
            .any(|label| label == "intent:RESILIENCE:CIRCUIT_BREAKER")
    );
}

#[test]
fn native_enterprise_intents_combine_language_and_behavior() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let cases = [
        (
            b"Build a JavaScript transactional outbox event service.".as_slice(),
            "intent:LANGUAGE:JAVASCRIPT",
            "intent:INTEGRATION:TRANSACTIONAL_OUTBOX",
        ),
        (
            b"Implement Go concurrent deduplication.".as_slice(),
            "intent:LANGUAGE:GO",
            "intent:CONCURRENCY:DEDUPLICATION",
        ),
        (
            b"Write a C# asynchronous retry policy.".as_slice(),
            "intent:LANGUAGE:CSHARP",
            "intent:RESILIENCE:ASYNC_RETRY",
        ),
        (
            b"Create Java optimistic concurrency with expected versions.".as_slice(),
            "intent:LANGUAGE:JAVA",
            "intent:STATE:OPTIMISTIC_CONCURRENCY",
        ),
        (
            b"Build a Rust atomic ledger transfer.".as_slice(),
            "intent:LANGUAGE:RUST",
            "intent:DOMAIN:ATOMIC_LEDGER_TRANSFER",
        ),
    ];
    for (prompt, language, behavior) in cases {
        let features = encoding.atomize(prompt);
        assert!(
            features.iter().any(|label| label == language),
            "{features:?}"
        );
        assert!(
            features.iter().any(|label| label == behavior),
            "{features:?}"
        );
    }
    let hyphenated_outbox = encoding.atomize(b"Create Node.js outbox-event ordering code.");
    assert!(
        hyphenated_outbox
            .iter()
            .any(|label| label == "intent:INTEGRATION:TRANSACTIONAL_OUTBOX")
    );
    let hyphenated_version = encoding.atomize(b"Create Java expected-version storage.");
    assert!(
        hyphenated_version
            .iter()
            .any(|label| label == "intent:STATE:OPTIMISTIC_CONCURRENCY")
    );
}

#[test]
fn typescript_is_an_independent_language_feature() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding.atomize(
        b"Implement a TypeScript idempotent order command with bounded retry.",
    );
    assert!(
        features
            .iter()
            .any(|label| label == "intent:LANGUAGE:TYPESCRIPT")
    );
    assert!(
        !features
            .iter()
            .any(|label| label == "intent:LANGUAGE:JAVASCRIPT")
    );
    assert!(
        features
            .iter()
            .any(|label| label == "intent:API:IDEMPOTENT_COMMAND")
    );
}

#[test]
fn batching_size_validation_does_not_alias_generic_input_validation() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding.atomize(
        b"Write a Python batching function that splits records into fixed-size chunks and validates the size.",
    );
    assert!(features
        .iter()
        .any(|label| label == "intent:ENTERPRISE:BATCHING"));
    assert!(!features
        .iter()
        .any(|label| label == "intent:ENTERPRISE:INPUT_VALIDATION"));
}

#[test]
fn masking_nested_credentials_is_secret_redaction_evidence() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding.atomize(
        b"Create Python correlated logs that mask nested credentials.",
    );
    assert!(features
        .iter()
        .any(|label| label == "intent:ENTERPRISE:SECRET_REDACTION"));
    assert!(features
        .iter()
        .any(|label| label == "intent:OBSERVABILITY:CORRELATED_LOGGING"));
}

#[test]
fn typescript_transport_retry_does_not_alias_a_retry_helper() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let idempotent = encoding.atomize(
        b"Create typed order-handling code that prevents duplicate TypeScript commands when clients retry with the same idempotency key.",
    );
    assert!(idempotent
        .iter()
        .any(|label| label == "intent:API:IDEMPOTENT_COMMAND"));
    assert!(!idempotent
        .iter()
        .any(|label| label == "intent:ENTERPRISE:BOUNDED_RETRY"));

    let retry = encoding.atomize(
        b"Write TypeScript async retry code that limits attempts and stops after success.",
    );
    assert!(retry
        .iter()
        .any(|label| label == "intent:ENTERPRISE:BOUNDED_RETRY"));
}

#[test]
fn version_checked_storage_is_optimistic_concurrency_evidence() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features = encoding.atomize(
        b"Create typed version-checked storage so stale TypeScript writers cannot replace newer state.",
    );
    assert!(features
        .iter()
        .any(|label| label == "intent:LANGUAGE:TYPESCRIPT"));
    assert!(features
        .iter()
        .any(|label| label == "intent:STATE:OPTIMISTIC_CONCURRENCY"));
}

#[test]
fn native_ledger_paraphrase_and_missing_context_are_detected() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let paraphrase = encoding.atomize(
        b"Create Rust code for all-or-nothing account transfers that reject missing accounts.",
    );
    assert!(
        paraphrase
            .iter()
            .any(|label| label == "intent:LANGUAGE:RUST")
    );
    assert!(
        paraphrase
            .iter()
            .any(|label| label == "intent:DOMAIN:ATOMIC_LEDGER_TRANSFER")
    );

    let missing = encoding.atomize(
        b"Implement a Go deduplicator whose retention duration and storage are not provided.",
    );
    assert!(
        missing
            .iter()
            .any(|label| label == "intent:GROUNDING:UNDERSPECIFIED")
    );
    let plural_missing = encoding.atomize(
        b"Create JavaScript deployment code whose contracts have not been provided.",
    );
    assert!(
        plural_missing
            .iter()
            .any(|label| label == "intent:GROUNDING:UNDERSPECIFIED")
    );
    let requirements_missing = encoding.atomize(
        b"Build TypeScript retry timing without any latency requirements.",
    );
    assert!(
        requirements_missing
            .iter()
            .any(|label| label == "intent:GROUNDING:UNDERSPECIFIED")
    );
}

#[test]
fn composed_observability_paraphrase_retains_redaction_evidence() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let features =
        encoding.atomize(b"Create Python correlated JSON observability that masks secrets.");
    assert!(
        features
            .iter()
            .any(|label| label == "intent:OBSERVABILITY:CORRELATED_LOGGING")
    );
    assert!(
        features
            .iter()
            .any(|label| label == "intent:ENTERPRISE:SECRET_REDACTION")
    );
    let audit = encoding.atomize(
        b"Build Python correlated JSON audit logs with recursive secret redaction.",
    );
    assert!(
        audit
            .iter()
            .any(|label| label == "intent:OBSERVABILITY:CORRELATED_LOGGING")
    );
}

#[test]
fn complete_project_intent_is_distinct_from_an_internal_fragment() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let project = encoding.atomize(
        b"Create Python code in multiple files separating inventory rules from a service.",
    );
    let fragment = encoding.atomize(b"Write a Python multi-file inventory service class.");
    assert!(
        project
            .iter()
            .any(|label| label == "intent:ARTIFACT:PROJECT")
    );
    let safe_inventory = encoding.atomize(
        b"Create Python code in multiple files separating inventory rules from a service that reserves stock safely.",
    );
    assert!(
        safe_inventory
            .iter()
            .any(|label| label == "intent:GUARD:INSUFFICIENT_STOCK")
    );
    assert!(
        safe_inventory
            .iter()
            .any(|label| label == "intent:STRUCTURE:SERVICE_MODULE")
    );
    assert!(
        !fragment
            .iter()
            .any(|label| label == "intent:ARTIFACT:PROJECT")
    );
}

#[test]
fn internal_intent_frames_round_trip_sparse_semantic_atoms() {
    let encoding = InstructionIntentEncoding {
        prefix: "intent".into(),
    };
    let labels = encoding.atomize(b"@intent:LANGUAGE:PYTHON\n@intent:SECURITY:AUTHORIZATION\n");
    assert_eq!(
        labels,
        vec![
            "intent:LANGUAGE:PYTHON".to_string(),
            "intent:SECURITY:AUTHORIZATION".to_string(),
        ]
    );
    let active: Vec<(&str, f32)> = labels.iter().map(|label| (label.as_str(), 1.0)).collect();
    let frame = encoding.reassemble(&active);
    assert_eq!(encoding.atomize(&frame), labels);
}
