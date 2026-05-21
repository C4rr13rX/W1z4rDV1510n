//! Stage 17.4 sleep decomposition tests per [`ARCHITECTURE.md`] §17.4.
//!
//! Confirms:
//! 1. `Brain::sleep_pool_phase1` / `sleep_pool_phase2` / `sleep_pool_housekeeping`
//!    together produce the same final state as the single-shot `Brain::sleep`.
//! 2. Each phase function only writes the targeted pool's lock, leaving
//!    other pools readable concurrently — verified by holding a read lock
//!    on another pool during the call.

use std::sync::Arc;

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};
use w1z4rd_brain::pool::Pool;

fn make_brain_two_pools() -> Brain {
    let cfg = BrainConfig::default();
    let mut brain = Brain::new(cfg);
    for (id, prefix) in [(1u32, "t"), (2u32, "u")] {
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
    // Train a short pattern in both pools so concept neurons exist.
    for _ in 0..4 {
        brain.fabric_mut().observe(1, b"ab");
        brain.fabric_mut().observe(2, b"cd");
        brain.fabric_mut().advance_tick();
    }
    brain
}

#[test]
fn decomposed_sleep_matches_single_shot_sleep() {
    let brain_a = make_brain_two_pools();
    let brain_b = make_brain_two_pools();

    // Single-shot path on A.
    let pruned_a = brain_a.sleep(/*min_use_count*/ 100, /*stale_ticks*/ 0);

    // Decomposed path on B — exercises the §17.4 sub-methods directly.
    let pool_ids: Vec<_> = brain_b.fabric().pool_ids();
    let mut all_pruned: ahash::AHashSet<w1z4rd_brain::NeuronRef> =
        ahash::AHashSet::new();
    let mut pruned_b = 0usize;
    for pid in &pool_ids {
        let pr = brain_b.sleep_pool_phase1(*pid, 100, 0);
        pruned_b += pr.len();
        all_pruned.extend(pr.into_iter());
    }
    if !all_pruned.is_empty() {
        for pid in &pool_ids {
            brain_b.sleep_pool_phase2(*pid, &all_pruned);
        }
    }
    for pid in &pool_ids {
        brain_b.sleep_pool_housekeeping(*pid);
    }

    assert_eq!(pruned_a, pruned_b,
        "decomposed sleep should prune same count as single-shot");

    // Pool-by-pool: concept counts must match in both brains.
    for pid in &pool_ids {
        let pa = brain_a.fabric().pool(*pid).expect("pool a");
        let pb = brain_b.fabric().pool(*pid).expect("pool b");
        let ca = pa.read().concept_count();
        let cb = pb.read().concept_count();
        assert_eq!(ca, cb,
            "pool {} concept_count mismatch after sleep: {} vs {}",
            pid, ca, cb);
    }
}

#[test]
fn phase1_on_one_pool_does_not_lock_other_pools() {
    let brain = make_brain_two_pools();
    let p1 = brain.fabric().pool(1).expect("pool 1");
    let _read_guard = p1.read();  // hold a long read lock on pool 1

    // Run phase1 on pool 2 — must not deadlock or block on pool 1's lock.
    // If this hangs, the test framework's default 60s timeout would catch it.
    let pruned = brain.sleep_pool_phase1(2, 100, 0);
    // (number pruned may be 0 or more; what matters is that we got here)
    let _ = pruned;
}

#[test]
fn sleep_is_now_callable_on_immutable_brain_ref() {
    // Stage 17.4 changed Brain::sleep from `&mut self` to `&self`.  This
    // test exists purely to nail down the contract: the type system must
    // accept `&Brain` as the receiver so the brain_server can call sleep
    // from inside a borrowed `tokio::sync::MutexGuard`.
    let brain: Arc<Brain> = Arc::new(make_brain_two_pools());
    let pruned: usize = brain.sleep(100, 0);
    let _ = pruned;
}
