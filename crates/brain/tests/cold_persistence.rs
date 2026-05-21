//! Stage 17.4 step 5 — cold-tier persistence round-trip per
//! [`ARCHITECTURE.md`] §17.4.
//!
//! Confirms:
//! 1. After eviction, `PoolSnapshot::cold_offsets` contains the
//!    eviction index.
//! 2. After snapshot + restore + re-attaching the cold tier file, the
//!    evicted neuron can still be paged back in (i.e. the offset
//!    survived the round-trip AND points at intact cold-tier data).
//! 3. `Pool::is_evicted` is true for the evicted neuron immediately
//!    after restore — without re-attaching the cold tier yet.

use std::sync::Arc;

use w1z4rd_brain::{AtomEncoding, BytePassthroughEncoding, PoolConfig};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::persistence::PoolSnapshot;
use w1z4rd_brain::store::ColdTier;

fn tmpdir(test: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir()
        .join(format!("w1z4rd_cold_persist_{}_{}_{}", test, pid, nano));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn make_pool() -> Pool {
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    Pool::new(pc, enc)
}

#[test]
fn cold_offsets_round_trip_through_snapshot() {
    let dir = tmpdir("rt");
    let cold_path = dir.join("pool.cold");

    // ---- Pre-restart: train, evict, snapshot ----
    let snap: PoolSnapshot;
    let pre_neuron_count: usize;
    let evicted_id: u32;
    {
        let mut pool = make_pool();
        pool.set_cold_tier(Arc::new(ColdTier::open(&cold_path).unwrap()));
        for _ in 0..4 { pool.observe_frame(b"ab", 0); }

        evicted_id = pool.iter_neurons()
            .find(|n| !n.is_atom())
            .map(|n| n.id)
            .expect("concept");
        pre_neuron_count = pool.neuron_count();

        let did = pool.evict_neuron(evicted_id).expect("evict");
        assert!(did);
        assert!(pool.is_evicted(evicted_id));

        snap = pool.snapshot();
    }

    // PoolSnapshot must contain the cold_offsets entry.
    assert!(snap.cold_offsets.iter().any(|(id, _)| *id == evicted_id),
        "snapshot must include cold_offsets entry for evicted neuron");

    // ---- Restart: restore from snapshot ----
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let mut restored = Pool::from_snapshot(snap, enc);
    assert_eq!(restored.neuron_count(), pre_neuron_count,
        "neuron count must survive snapshot");
    assert!(restored.is_evicted(evicted_id),
        "evicted state must survive snapshot");

    // ---- Re-attach cold tier (same file path) ----
    restored.set_cold_tier(Arc::new(ColdTier::open(&cold_path).unwrap()));

    // ---- Page in: must produce identical neuron ----
    let paged_in = restored.page_in_neuron(evicted_id).expect("page_in");
    assert!(paged_in, "page_in after restart should restore");
    assert!(!restored.is_evicted(evicted_id));
    let restored_n = restored.get(evicted_id).expect("get");
    assert!(restored_n.terminals.len() > 0,
        "paged-in neuron must have terminals restored");

    std::fs::remove_dir_all(&dir).ok();
}
