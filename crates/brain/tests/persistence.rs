//! Checkpoint / restore tests per [`ARCHITECTURE.md`] §6 + §11 Phase 9.
//!
//! The architectural claim: a brain's learned state survives a
//! process restart.  Tests train a brain, snapshot it, drop it,
//! restore it, and verify behavior is identical.

use std::collections::HashMap;
use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    PoolId,
};

/// Build matching encodings for the two-pool fixture used across
/// tests so restore gets the right prefixes back.
fn fixture_encodings() -> HashMap<PoolId, Box<dyn AtomEncoding>> {
    let mut m: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    m.insert(0u32, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    m.insert(1u32, Box::new(BytePassthroughEncoding { prefix: "t" }));
    m.insert(2u32, Box::new(BytePassthroughEncoding { prefix: "a" }));
    m
}

fn build_trained_brain() -> Brain {
    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    for _ in 0..6 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain
}

#[test]
fn snapshot_roundtrip_preserves_neuron_and_terminal_counts() {
    // The basic shape: snapshot + restore in memory; stats agree.
    let original = build_trained_brain();
    let snap = original.snapshot();
    let (restored, missing) = Brain::from_snapshot(snap, fixture_encodings());
    assert!(missing.is_empty(),
        "all pool encodings provided; missing must be empty; got {:?}", missing);

    let s1 = original.stats();
    let s2 = restored.stats();
    assert_eq!(s1.tick, s2.tick,                       "tick must round-trip");
    assert_eq!(s1.pool_count, s2.pool_count,           "pool_count must round-trip");
    assert_eq!(s1.total_neurons, s2.total_neurons,     "total_neurons must round-trip");
    assert_eq!(s1.total_concepts, s2.total_concepts,   "total_concepts must round-trip");
    assert_eq!(s1.total_binding, s2.total_binding,     "total_binding must round-trip");
    assert_eq!(s1.total_terminals, s2.total_terminals, "total_terminals must round-trip");
}

#[test]
fn restored_brain_produces_the_same_answer_as_original() {
    // The strong claim: behavior survives.  After training, the
    // restored brain integrates to the same answer the original
    // would have produced.

    let mut original = build_trained_brain();
    let snap = original.snapshot();
    let (mut restored, _) = Brain::from_snapshot(snap, fixture_encodings());

    original.observe(1, b"X");
    restored.observe(1, b"X");
    let a1 = original.integrate(1, 2);
    let a2 = restored.integrate(1, 2);

    assert_eq!(a1.answer, a2.answer,
        "restored brain must produce the same answer; original={:?} restored={:?}",
        a1.answer, a2.answer);
    assert!((a1.grounding.fabric_confidence - a2.grounding.fabric_confidence).abs() < 1e-6,
        "fabric_confidence must round-trip; original={} restored={}",
        a1.grounding.fabric_confidence, a2.grounding.fabric_confidence);
}

#[test]
fn checkpoint_then_restore_via_filesystem_round_trips() {
    // The shipped persistence path: write to a real file, drop the
    // original brain, restore from disk.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("w1z4rd_brain_checkpoint_test_{}.bin",
        std::process::id()));

    {
        let brain = build_trained_brain();
        brain.checkpoint(&path).expect("checkpoint must succeed");
    }

    let (restored, missing) = Brain::restore(&path, fixture_encodings())
        .expect("restore must succeed");
    assert!(missing.is_empty(), "all encodings provided");
    assert!(restored.stats().total_terminals > 0,
        "restored brain must carry the trained terminals");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn missing_encoding_reports_pool_id_without_panicking() {
    // Honest behavior under partial restore: pools without an
    // encoding are skipped and reported.  No panic, no invented
    // encoding.

    let brain = build_trained_brain();
    let snap = brain.snapshot();
    let mut partial: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    partial.insert(0, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    partial.insert(1, Box::new(BytePassthroughEncoding { prefix: "t" }));
    // pool 2 deliberately omitted.

    let (_restored, missing) = Brain::from_snapshot(snap, partial);
    assert!(missing.contains(&2),
        "missing pool encoding must be reported; got {:?}", missing);
}

#[test]
fn eem_state_round_trips_through_snapshot() {
    let mut brain = Brain::new(BrainConfig::default());
    let v = brain.eem_mut().register_variable("a", None);
    let eq = brain.eem_mut().register_equation("eq", "a + 1", vec![v], None);
    for _ in 0..3 { brain.eem_mut().report_validation(eq, true); }
    let conf_before = brain.eem().confidence(eq).unwrap();

    let snap = brain.snapshot();
    let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    encodings.insert(0, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    let (restored, _) = Brain::from_snapshot(snap, encodings);

    assert_eq!(restored.eem().equation_count(), 1,
        "registered equation must survive snapshot");
    let conf_after = restored.eem().confidence(eq).unwrap();
    assert!((conf_after - conf_before).abs() < 1e-6,
        "earned confidence must round-trip: before={} after={}", conf_before, conf_after);
}

#[test]
fn annealer_history_survives_round_trip() {
    let mut brain = Brain::new(BrainConfig::default());
    let pool = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    for _ in 0..4 {
        brain.observe(pool, b"X");
        brain.advance_tick();
    }
    let history_before = brain.annealer().history_len(pool);
    assert!(history_before > 0);

    let snap = brain.snapshot();
    let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    encodings.insert(0, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    encodings.insert(1, Box::new(BytePassthroughEncoding { prefix: "t" }));
    let (restored, _) = Brain::from_snapshot(snap, encodings);
    assert_eq!(restored.annealer().history_len(pool), history_before,
        "annealer history must round-trip");
}
