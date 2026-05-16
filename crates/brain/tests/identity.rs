//! Brain identity / factory tests per [`ARCHITECTURE.md`] §3 + §11 Phase 10.
//!
//! These prove the factory contract: a declarative
//! `BrainIdentitySpec` (serializable to TOML) plus a prototype
//! registry produces a working brain — pools wired, action pool
//! designated, EEM and annealer ready for seeding.

use w1z4rd_brain::{
    Brain, BrainIdentitySpec, IdentityBuildError, PoolKind, PoolPrototypeRegistry,
    PoolSpec,
};

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
        fabric:   Default::default(),
        eem:      Default::default(),
        annealer: Default::default(),
    };

    let registry = PoolPrototypeRegistry::with_defaults();
    let brain = Brain::from_identity(&spec, &registry).expect("must build");
    assert!(brain.action_pool_id().is_none(),
        "no Action pool in spec → none designated");
}
