//! Brain identity / factory tests per [`ARCHITECTURE.md`] §3 + §11 Phase 10.
//!
//! These prove the factory contract: a declarative
//! `BrainIdentitySpec` (serializable to TOML) plus a prototype
//! registry produces a working brain — pools wired, action pool
//! designated, EEM and annealer ready for seeding.

use w1z4rd_brain::{
    Brain, BrainDeploymentSpec, BrainIdentitySpec, DeploymentValidationError,
    ControlMode, ControlSignal, FeedbackLoopSpec, IdentityBuildError, PoolPrototypeRegistry, PoolSpec,
    ResourceBudget,
};
use std::path::PathBuf;

#[test]
fn deployment_isolates_state_and_validates_feedback_pools() {
    let identity = BrainIdentitySpec::default_general_observer();
    let deployment = BrainDeploymentSpec {
        instance_id: "coding-small-a".into(),
        identity_path: "brains/general.toml".into(),
        data_dir: "runtime/brains/coding-small-a".into(),
        resource_budget: ResourceBudget {
            max_resident_bytes: 64 * 1024 * 1024,
            max_neurons: 100_000,
            max_propagation_steps: 32,
            max_learning_steps_per_second: 500,
        },
        feedback_loops: vec![FeedbackLoopSpec {
            source_pool: "action".into(),
            target_pool: "text".into(),
            signal: "prediction_error".into(),
            gain: 0.5,
            gain_mode: Some(ControlMode::DrivenBy {
                signal: ControlSignal::Surprise,
                scale: 0.8,
                offset: 0.1,
                min: 0.0,
                max: 1.0,
            }),
            delay_ticks: 1,
        }],
    };
    deployment.validate(&identity).unwrap();
    assert_eq!(deployment.snapshot_path(), PathBuf::from("runtime/brains/coding-small-a/brain.bin"));

    let mut invalid = deployment.clone();
    invalid.feedback_loops[0].target_pool = "missing".into();
    assert_eq!(invalid.validate(&identity),
        Err(DeploymentValidationError::UnknownFeedbackPool("missing".into())));
}

#[test]
fn shipped_small_brain_specs_parse_and_validate() {
    for (identity_text, deployment_text) in [
        (include_str!("../../../brains/market_small.identity.toml"),
         include_str!("../../../brains/market_small.deployment.toml")),
        (include_str!("../../../brains/coding_small.identity.toml"),
         include_str!("../../../brains/coding_small.deployment.toml")),
    ] {
        let identity: BrainIdentitySpec = toml::from_str(identity_text).unwrap();
        let deployment: BrainDeploymentSpec = toml::from_str(deployment_text).unwrap();
        deployment.validate(&identity).unwrap();
        Brain::from_identity(&identity, &PoolPrototypeRegistry::with_defaults()).unwrap();
    }
}

#[test]
fn identity_carries_ga_discovered_dynamic_pool_wiring() {
    let mut pool = PoolSpec::sensory_byte_passthrough("market", 7, "m");
    pool.sparsity_mode = Some(ControlMode::DrivenBy {
        signal: ControlSignal::InvSurprise,
        scale: 0.8,
        offset: 0.1,
        min: 0.05,
        max: 1.0,
    });
    pool.predict_gate_mode = Some(ControlMode::Constant(0.25));
    let config = pool.to_pool_config();
    assert_eq!(config.sparsity_mode, pool.sparsity_mode.unwrap());
    assert_eq!(config.predict_gate_mode, pool.predict_gate_mode.unwrap());
}

#[test]
fn online_feedback_executes_dynamic_gain_and_delay() {
    let identity = BrainIdentitySpec::default_general_observer();
    let mut brain = Brain::from_identity(&identity, &PoolPrototypeRegistry::with_defaults()).unwrap();
    let deployment = BrainDeploymentSpec {
        instance_id: "feedback-test".into(),
        identity_path: "unused.toml".into(),
        data_dir: "runtime/test-feedback".into(),
        resource_budget: ResourceBudget::default(),
        feedback_loops: vec![FeedbackLoopSpec {
            source_pool: "text".into(),
            target_pool: "action".into(),
            signal: "activation_trace".into(),
            gain: 1.0,
            gain_mode: Some(ControlMode::Constant(0.5)),
            delay_ticks: 1,
        }],
    };
    brain.configure_feedback_loops(&identity, &deployment).unwrap();

    brain.observe(1, b"alpha");
    brain.advance_tick();
    assert_eq!(brain.feedback_events_emitted(), 0, "half gain has not accumulated a spike");

    brain.observe(1, b"alpha");
    brain.advance_tick();
    assert_eq!(brain.feedback_events_emitted(), 0, "spike is queued for the delayed tick");

    brain.advance_tick();
    assert_eq!(brain.feedback_events_emitted(), 1);
    let action = brain.fabric().pool(2).unwrap();
    assert!(action.read().iter_neurons().next().is_some(), "feedback trace reached target pool");
}

#[test]
fn default_general_observer_builds_a_working_brain() {
    let spec     = BrainIdentitySpec::default_general_observer();
    let registry = PoolPrototypeRegistry::with_defaults();
    let mut brain = Brain::from_identity(&spec, &registry)
        .expect("default identity must build");

    // 3 pools: binding (auto) + text + action.
    let stats = brain.stats();
    assert_eq!(stats.pool_count, 3,
        "default observer should have binding + text + action; got {}",
        stats.pool_count);

    // Action pool auto-designated.
    assert!(brain.action_pool_id().is_some(),
        "first Action pool must be auto-designated");
    assert_eq!(brain.action_pool_id().unwrap(), 2,
        "action pool id should be 2 per default spec");

    // Smoke: observe + advance ticks to confirm wiring is real.
    for _ in 0..3 {
        brain.observe(1, b"X");
        brain.advance_tick();
    }
    let final_stats = brain.stats();
    assert!(final_stats.tick > 0, "ticks must advance");
    assert!(final_stats.total_neurons >= 1,
        "observation must create atoms; got total_neurons={}",
        final_stats.total_neurons);
}

#[test]
fn prediction_activation_is_transient_and_non_learning() {
    let spec = BrainIdentitySpec::default_general_observer();
    let mut brain = Brain::from_identity(&spec, &PoolPrototypeRegistry::with_defaults()).unwrap();
    for _ in 0..3 {
        brain.observe(1, b"known");
        brain.observe(2, b"answer");
        brain.advance_tick();
    }
    let before = brain.stats();
    assert!(!brain.activate_for_prediction(1, b"known").is_empty());
    let _ = brain.integrate(1, 2);
    brain.clear_prediction_activation();
    let after = brain.stats();
    assert_eq!(after.tick, before.tick);
    assert_eq!(after.total_neurons, before.total_neurons);
    assert_eq!(after.total_terminals, before.total_terminals);
    assert!(brain.fabric().current_moment().fired.is_empty());
    assert_eq!(brain.fabric().pool(1).unwrap().read().currently_firing().count(), 0);
    assert!(brain.activate_for_prediction(1, b"entirely unseen").len() < 15,
            "unknown atoms must not be born during prediction");
    brain.clear_prediction_activation();
    assert_eq!(brain.stats().total_neurons, before.total_neurons);
}

#[test]
fn unknown_prototype_returns_typed_error_without_partial_build() {
    let mut spec = BrainIdentitySpec::default_general_observer();
    spec.pools[0].prototype = "this-prototype-does-not-exist".into();
    let registry = PoolPrototypeRegistry::with_defaults();
    let err = Brain::from_identity(&spec, &registry).err()
        .expect("unknown prototype must error");
    match err {
        IdentityBuildError::UnknownPrototype(name) => {
            assert_eq!(name, "this-prototype-does-not-exist",
                "error must carry the offending prototype name");
        }
        other => panic!("expected UnknownPrototype, got {:?}", other),
    }
}

#[test]
fn duplicate_pool_ids_are_rejected_up_front() {
    let mut spec = BrainIdentitySpec::default_general_observer();
    spec.pools.push(PoolSpec::sensory_byte_passthrough("dup", 1, "d"));
    let registry = PoolPrototypeRegistry::with_defaults();
    let err = Brain::from_identity(&spec, &registry).err()
        .expect("duplicate id must error");
    assert!(matches!(err, IdentityBuildError::DuplicatePoolId(1)),
        "expected DuplicatePoolId(1), got {:?}", err);
}

#[test]
fn pool_id_zero_collides_with_binding_pool() {
    // Pool id 0 is reserved for the auto-created binding pool.
    let mut spec = BrainIdentitySpec::default_general_observer();
    spec.pools[0].id = 0;
    let registry = PoolPrototypeRegistry::with_defaults();
    let err = Brain::from_identity(&spec, &registry).err()
        .expect("binding-pool collision must error");
    assert!(matches!(err, IdentityBuildError::BindingPoolIdCollision(0)),
        "expected BindingPoolIdCollision(0), got {:?}", err);
}

#[test]
fn toml_roundtrip_preserves_identity_spec() {
    // Spec §3.3: identities load from TOML files.  Roundtrip a spec
    // through the on-disk format and confirm it deserializes back
    // to a buildable identity.

    let original = BrainIdentitySpec::default_general_observer();
    let dir  = std::env::temp_dir();
    let path = dir.join(format!("w1z4rd_brain_identity_test_{}.toml",
        std::process::id()));
    original.save_toml(&path).expect("save_toml must succeed");

    let reloaded = BrainIdentitySpec::load_toml(&path)
        .expect("load_toml must succeed");

    assert_eq!(reloaded.name, original.name);
    assert_eq!(reloaded.version, original.version);
    assert_eq!(reloaded.pools.len(), original.pools.len());
    for (a, b) in reloaded.pools.iter().zip(original.pools.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.id, b.id);
        assert_eq!(a.prototype, b.prototype);
        assert_eq!(a.atom_encoding_prefix, b.atom_encoding_prefix);
        assert_eq!(a.kind, b.kind);
    }
    assert_eq!(reloaded.binding_emergence_threshold,
               original.binding_emergence_threshold);

    let registry = PoolPrototypeRegistry::with_defaults();
    Brain::from_identity(&reloaded, &registry)
        .expect("reloaded identity must build a working brain");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn custom_prototype_registers_and_is_resolved() {
    // The registry is extensible: callers add their own encoding
    // factories without touching the brain crate.

    let mut registry = PoolPrototypeRegistry::new();
    registry.register("custom-byte", |prefix| {
        let leaked: &'static str = Box::leak(prefix.to_owned().into_boxed_str());
        Box::new(w1z4rd_brain::BytePassthroughEncoding { prefix: leaked })
    });

    let spec = BrainIdentitySpec {
        name:    "custom".into(),
        version: "0.1.0".into(),
        pools:   vec![{
            let mut p = PoolSpec::sensory_byte_passthrough("text", 1, "tx");
            p.prototype = "custom-byte".into();
            p
        }],
        binding_emergence_threshold: 3,
        moment_history_window:       64,
        min_atom_score:              0.5,
        fabric:   Default::default(),
        eem:      Default::default(),
        annealer: Default::default(),
    };

    Brain::from_identity(&spec, &registry)
        .expect("custom prototype must resolve");
}

#[test]
fn no_action_pool_means_no_designation_no_error() {
    // An identity with only sensory pools is valid — action pool is
    // optional.  Brain comes up with action_pool_id == None.

    let spec = BrainIdentitySpec {
        name:    "sensors_only".into(),
        version: "0.1.0".into(),
        pools:   vec![
            PoolSpec::sensory_byte_passthrough("a", 1, "a"),
            PoolSpec::sensory_byte_passthrough("b", 2, "b"),
        ],
        binding_emergence_threshold: 3,
        moment_history_window:       64,
        min_atom_score:              0.5,
        fabric:   Default::default(),
        eem:      Default::default(),
        annealer: Default::default(),
    };

    let registry = PoolPrototypeRegistry::with_defaults();
    let brain = Brain::from_identity(&spec, &registry).expect("must build");
    assert!(brain.action_pool_id().is_none(),
        "no Action pool in spec → none designated");
}
