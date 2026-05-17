//! Stage 10 — dynamic emergent architecture tests.
//!
//! Verifies the two-tier binding promotion + pressure feedback loop:
//!
//! * Tentative bindings emerge at `tentative_emergence_threshold`
//!   (default 1) without registering an EEM grounded fact.
//! * Consolidated bindings emerge at the (pressure-adjusted)
//!   `binding_emergence_threshold` and register an EEM grounded fact.
//! * Sparse training schedules (same pair recurring far outside the
//!   moment_history_window) still promote via lifetime recurrence —
//!   the exact failure mode that left K-12 vocab at 0% recall under
//!   the legacy single-signal rule.
//! * `force_promote_tentative` is idempotent and respects already-
//!   promoted fingerprints.
//! * Pressure feedback ratchets `current_threshold` up/down between
//!   the configured band edges.

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

fn build(cfg: BrainConfig) -> (Brain, u32, u32) {
    let mut brain = Brain::new(cfg);
    let a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let b = brain.create_pool(
        PoolConfig::defaults("action", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    (brain, a, b)
}

fn default_cfg() -> BrainConfig {
    BrainConfig::default()
}

#[test]
fn tentative_tier_promotes_on_first_co_firing() {
    // Default tentative threshold = 1, so a single (X, Y) co-firing
    // crystallizes a tentative binding.  Consolidated threshold = 3,
    // so it does NOT yet register an EEM fact.
    let (mut brain, pa, pb) = build(default_cfg());
    assert_eq!(brain.tentative_binding_count(), 0);
    assert_eq!(brain.consolidated_binding_count(), 0);
    assert_eq!(brain.eem_fact_count(), 0);

    brain.observe(pa, b"X");
    brain.observe(pb, b"Y");
    brain.advance_tick();

    assert!(brain.tentative_binding_count() >= 1,
        "first co-firing must produce at least one tentative binding");
    assert_eq!(brain.consolidated_binding_count(), 0,
        "consolidated tier must NOT trip on a single co-firing");
    assert_eq!(brain.eem_fact_count(), 0,
        "no EEM fact until consolidation");
}

#[test]
fn consolidation_upgrades_tentative_and_registers_eem_fact() {
    let (mut brain, pa, pb) = build(default_cfg());
    // Three identical co-firings → cross consolidated threshold = 3.
    for _ in 0..3 {
        brain.observe(pa, b"K");
        brain.observe(pb, b"V");
        brain.advance_tick();
    }
    assert!(brain.consolidated_binding_count() >= 1,
        "third recurrence must promote a consolidated binding");
    assert_eq!(brain.tentative_binding_count(), 0,
        "tentative tier should have been upgraded to consolidated, \
         not left as a dangling duplicate");
    assert!(brain.eem_fact_count() >= 1,
        "consolidated promotion must register an EEM grounded fact");
}

#[test]
fn lifetime_count_promotes_sparse_schedule() {
    // The K-12 failure mode: same (X, Y) recurs every 200 ticks —
    // far outside moment_history_window of 64 — so windowed count
    // never accumulates.  Lifetime count is the sparse-schedule
    // fallback and *does* accumulate.  By the third recurrence the
    // consolidated tier must promote.
    let mut cfg = default_cfg();
    cfg.moment_history_window = 64;
    let (mut brain, pa, pb) = build(cfg);
    let interval = 200;
    for rep in 0..3 {
        brain.observe(pa, b"sparse_X");
        brain.observe(pb, b"sparse_Y");
        brain.advance_tick();
        // Distract: many unrelated observations push the sparse
        // fingerprint out of moment_history.
        if rep < 2 {
            for filler in 0..interval {
                let b1 = format!("noise_a_{}", filler);
                let b2 = format!("noise_b_{}", filler);
                brain.observe(pa, b1.as_bytes());
                brain.observe(pb, b2.as_bytes());
                brain.advance_tick();
            }
        }
    }
    assert!(brain.consolidated_binding_count() >= 1,
        "sparse-schedule training must still consolidate via lifetime count; \
         tentative={} consolidated={}",
        brain.tentative_binding_count(), brain.consolidated_binding_count());
}

#[test]
fn force_promote_tentative_picks_up_pending_fingerprints() {
    // Disable the tentative tier so the first co-firing does NOT
    // auto-promote.  Then call force_promote_tentative manually —
    // it should pick up the lifetime-tracked fingerprint.
    let mut cfg = default_cfg();
    cfg.tentative_emergence_threshold = u32::MAX;  // tentative tier off
    cfg.binding_emergence_threshold = 5;           // consolidated needs 5
    let (mut brain, pa, pb) = build(cfg);
    brain.observe(pa, b"P");
    brain.observe(pb, b"Q");
    brain.advance_tick();
    assert_eq!(brain.tentative_binding_count(), 0,
        "tentative tier was disabled — no auto-promotion expected");
    let promoted = brain.force_promote_tentative(1);
    assert!(!promoted.is_empty(),
        "force_promote_tentative(1) must promote the lifetime-tracked fingerprint");
    assert!(brain.tentative_binding_count() >= 1);

    // Idempotent: second call must not double-promote.
    let again = brain.force_promote_tentative(1);
    assert!(again.is_empty(),
        "force_promote_tentative is idempotent; got {} extra promotions", again.len());
}

#[test]
fn pressure_feedback_lowers_threshold_when_density_below_band() {
    // Configure a tight grace + aggressive band so we can observe
    // the loop firing within a small test.
    let mut cfg = default_cfg();
    cfg.binding_emergence_threshold = 5;
    cfg.tentative_emergence_threshold = u32::MAX;  // suppress tentative
    cfg.pressure_band_low  = 0.5;   // require huge density to satisfy
    cfg.pressure_band_high = 0.99;  // ceiling out of reach
    cfg.pressure_observation_grace = 4;
    cfg.pressure_threshold_max = 10;
    cfg.pressure_adjust_enabled = true;

    let (mut brain, pa, pb) = build(cfg);
    let starting = brain.current_emergence_threshold();
    assert_eq!(starting, 5);

    // Many observations with distinct fingerprints → density stays
    // near 0 → loop should ratchet threshold down toward 1.
    for i in 0..50 {
        let s = format!("v_{}", i);
        brain.observe(pa, s.as_bytes());
        brain.observe(pb, s.as_bytes());
        brain.advance_tick();
    }
    let after = brain.current_emergence_threshold();
    assert!(after < starting,
        "low density must ratchet threshold down; starting={} after={}", starting, after);
    assert!(after >= 1,
        "threshold has a floor of 1; got {}", after);
}

#[test]
fn pressure_feedback_is_disabled_when_flag_off() {
    let mut cfg = default_cfg();
    cfg.binding_emergence_threshold = 5;
    cfg.tentative_emergence_threshold = u32::MAX;
    cfg.pressure_band_low  = 0.5;
    cfg.pressure_band_high = 0.99;
    cfg.pressure_observation_grace = 4;
    cfg.pressure_adjust_enabled = false;  // locked

    let (mut brain, pa, pb) = build(cfg);
    let starting = brain.current_emergence_threshold();
    for i in 0..50 {
        let s = format!("v_{}", i);
        brain.observe(pa, s.as_bytes());
        brain.observe(pb, s.as_bytes());
        brain.advance_tick();
    }
    assert_eq!(brain.current_emergence_threshold(), starting,
        "disabled pressure loop must not move the threshold");
}

#[test]
fn snapshot_roundtrip_preserves_tentative_and_lifetime_state() {
    use std::collections::HashMap;
    use w1z4rd_brain::{AtomEncoding, PoolId};

    let (mut brain, pa, pb) = build(default_cfg());
    brain.observe(pa, b"snap_X");
    brain.observe(pb, b"snap_Y");
    brain.advance_tick();
    let pre_tentative = brain.tentative_binding_count();
    let pre_consolidated = brain.consolidated_binding_count();
    let pre_obs = brain.total_observations();
    let pre_thr = brain.current_emergence_threshold();
    assert!(pre_tentative >= 1, "precondition: tentative promotion happened");

    let snap = brain.snapshot();
    let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    encodings.insert(0, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    encodings.insert(pa, Box::new(BytePassthroughEncoding { prefix: "t" }));
    encodings.insert(pb, Box::new(BytePassthroughEncoding { prefix: "a" }));
    let (restored, missing) = Brain::from_snapshot(snap, encodings);
    assert!(missing.is_empty(), "no encodings should be missing");
    assert_eq!(restored.tentative_binding_count(), pre_tentative);
    assert_eq!(restored.consolidated_binding_count(), pre_consolidated);
    assert_eq!(restored.total_observations(), pre_obs);
    assert_eq!(restored.current_emergence_threshold(), pre_thr);
}
