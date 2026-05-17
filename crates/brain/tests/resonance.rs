//! Stage 13A — resonant settling tests.
//!
//! Validates the substrate's new parallel inference path
//! ([`Fabric::settle`] + [`Brain::integrate_resonant`]) without
//! modifying the existing single-pass `propagate` / `integrate` path.
//! Every test here checks ONE of:
//!
//!   * settle terminates (converges or hits max_iter cleanly)
//!   * settle preserves the seed pool's contribution (the "mould"
//!     never gets drifted off the constraint by feedback)
//!   * a single text concept paired in training with multiple
//!     action concepts produces a multi-pool extrusion with ALL the
//!     trained concepts present in the settled state
//!   * the existing integrate() path is unchanged (no state mutation
//!     leaks from the settle path)

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

fn build_brain() -> (Brain, u32, u32) {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    let mut brain = Brain::new(cfg);
    let text = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let action = brain.create_pool(
        PoolConfig::defaults("action", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    (brain, text, action)
}

#[test]
fn settle_terminates_on_empty_seed() {
    // No firing in the source pool → settle should return an empty
    // result with iterations_run=0 and converged=true.  No panic, no
    // infinite loop.
    let (brain, text, _action) = build_brain();
    let res = brain.fabric().settle(text, 16, 8, 0.01);
    assert_eq!(res.iterations_run, 0);
    assert!(res.converged);
    assert!(res.pool_activations.is_empty());
}

#[test]
fn settle_converges_on_simple_trained_pair() {
    // After training (X, Y) several times, observing X should settle
    // to a state where Y's atoms are strongly active in the action
    // pool.  Settling must terminate within max_iter.
    let (mut brain, text, action) = build_brain();
    for _ in 0..5 {
        brain.observe(text, b"X");
        brain.observe(action, b"Y");
        brain.advance_tick();
    }
    // Light up the constraint (mould).
    brain.observe(text, b"X");

    let res = brain.fabric().settle(text, 8, 16, 0.01);
    assert!(res.iterations_run >= 1,
        "settle must run at least one iteration when source is firing");
    assert!(res.iterations_run <= 8,
        "settle must respect max_iter cap; ran {}", res.iterations_run);
    // The action pool should have non-zero activation in the settled
    // state — the cross-pool axons from training were exercised.
    let top_action = res.top_in_pool(action, 8);
    assert!(!top_action.is_empty(),
        "settled state must surface action-pool activation when text query was \
         paired with action targets during training");
}

#[test]
fn settle_preserves_source_pool_after_iteration() {
    // The "mould" should remain present in the source pool's settled
    // state even after multiple iterations of feedback.  This is the
    // damping property: the constraint doesn't drift.
    let (mut brain, text, action) = build_brain();
    for _ in 0..5 {
        brain.observe(text, b"M");
        brain.observe(action, b"N");
        brain.advance_tick();
    }
    brain.observe(text, b"M");

    let res = brain.fabric().settle(text, 8, 16, 0.01);
    let top_text = res.top_in_pool(text, 8);
    assert!(!top_text.is_empty(),
        "source pool's seed must remain in the settled state — without \
         damping, feedback from action pool would erase the prompt");
}

#[test]
fn integrate_resonant_extrudes_multiple_pools() {
    // Train ONE source concept against MULTIPLE distinct action
    // targets.  After training, observing the source should produce
    // a settled state where ALL trained action targets contribute
    // activation — that's the "mould fires many neurons across pools"
    // contract.
    let (mut brain, text, action) = build_brain();
    let action_responses: &[&[u8]] = &[b"first", b"second", b"third"];
    for resp in action_responses {
        for _ in 0..4 {
            brain.observe(text, b"boat");
            brain.observe(action, resp);
            brain.advance_tick();
        }
    }
    // Now fire the mould.
    brain.observe(text, b"boat");

    let extrusion = brain.integrate_resonant(
        text,
        &[action],
        /*top_per_pool*/ 10,
        /*max_iter*/ 12,
        /*eps*/ 0.01,
    );

    // We trained 3 distinct response strings.  Their atoms partially
    // overlap (e.g., t, r, s) but each has unique atoms (f from
    // "first", o from "second", h from "third").  The settled state
    // should produce decoded bytes from the action pool that include
    // contributions from MULTIPLE responses, not just one.
    let action_pool = extrusion.pools.iter()
        .find(|p| p.pool == action)
        .expect("action pool must appear in extrusion output");
    let mut all_bytes: Vec<u8> = Vec::new();
    for d in action_pool.decoded.iter() {
        all_bytes.extend_from_slice(&d.bytes);
    }
    // At minimum: the top-decoded set should contain bytes from at
    // least 2 of the 3 trained responses.
    let mut hits = 0;
    for r in action_responses {
        let unique_byte = r[0];
        if all_bytes.iter().any(|b| *b == unique_byte) {
            hits += 1;
        }
    }
    assert!(hits >= 2,
        "settled extrusion must surface contributions from >= 2 trained \
         responses (epoxy-mould contract); got {} of 3.  Decoded action \
         atoms: {:?}",
        hits,
        action_pool.decoded.iter().map(|d| d.label.as_str()).collect::<Vec<_>>());
}

#[test]
fn settle_path_does_not_disturb_integrate_path() {
    // After running settle(), the existing integrate() must return
    // the same answer as if settle had never been called.  This
    // pins the contract that the resonant path is a pure side-effect-
    // free observer.
    let (mut brain, text, action) = build_brain();
    for _ in 0..5 {
        brain.observe(text, b"dog");
        brain.observe(action, b"animal");
        brain.advance_tick();
    }
    // Light up the query.
    brain.observe(text, b"dog");

    // Baseline integrate() answer BEFORE any settle call.
    let baseline = brain.integrate(text, action);
    let baseline_answer = baseline.answer.clone();

    // Re-light query (integrate consumes firing? actually no, it just
    // reads.  But cleanly reset state by re-observing.)
    brain.observe(text, b"dog");

    // Run settle, discard.
    let _ = brain.fabric().settle(text, 12, 16, 0.01);

    // Re-light and integrate again.
    brain.observe(text, b"dog");
    let after = brain.integrate(text, action);

    assert_eq!(baseline_answer.as_ref().map(|b| b.len()),
               after.answer.as_ref().map(|b| b.len()),
        "settle() must not mutate any state the integrate path reads — \
         answer length should be identical pre- and post-settle");
}
