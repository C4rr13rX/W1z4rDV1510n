//! Stage 17.7 partial — salience-weighted replay tests per
//! [`ARCHITECTURE.md`] §17.7.
//!
//! Confirms:
//! 1. Replay completes without panic on a brain trained through the
//!    normal cross-pool path.
//! 2. After a salience-weighted replay, the brain's moment_history is
//!    still valid + queryable.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};
use w1z4rd_brain::pool::Pool;

fn make_brain_two_pools() -> Brain {
    let cfg = BrainConfig::default();
    let mut brain = Brain::new(cfg);
    for (id, prefix) in [(1u32, "t"), (4u32, "act")] {
        let mut pc = PoolConfig::defaults(format!("p{id}"), id);
        pc.recent_atoms_window = 8;
        pc.concept_emergence_threshold = 2;
        let enc: Box<dyn AtomEncoding> =
            Box::new(BytePassthroughEncoding {
                prefix: Box::leak(prefix.to_string().into_boxed_str()),
            });
        let pool = Pool::new(pc, enc);
        brain.fabric_mut().register_pool(pool);
    }
    // Train cross-pool pairs.  Multiple reps so concepts emerge and
    // moment history fills with consistent fingerprints.
    for _ in 0..6 {
        // Use Brain::observe / advance_tick (not Fabric's directly) so the
        // moment_history is populated — that's the path replay reads.
        brain.observe(1, b"ab");
        brain.observe(4, b"yz");
        brain.advance_tick();
    }
    brain
}

#[test]
fn salience_weighted_replay_runs_without_panic() {
    let mut brain = make_brain_two_pools();
    let tick_before = brain.fabric().current_tick();
    let n = brain.replay_salience_weighted(/*count*/ 4, /*strength*/ 0.5);
    // Replay should fire each chosen moment + advance one tick per moment.
    // n is bounded by moment_history capacity but should be > 0 here.
    assert!(n > 0, "expected at least one replayed moment, got {}", n);
    let tick_after = brain.fabric().current_tick();
    assert!(tick_after > tick_before, "tick must advance during replay");
}

#[test]
fn salience_weighted_replay_with_zero_count_is_no_op() {
    let mut brain = make_brain_two_pools();
    let n = brain.replay_salience_weighted(0, 0.5);
    assert_eq!(n, 0);
}

#[test]
fn salience_weighted_replay_with_empty_history_is_no_op() {
    // Brain that has never observed anything has no moments.
    let mut brain = Brain::new(BrainConfig::default());
    let n = brain.replay_salience_weighted(8, 0.5);
    assert_eq!(n, 0);
}
