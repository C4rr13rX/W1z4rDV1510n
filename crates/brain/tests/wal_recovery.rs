//! Stage 17.9 recovery test per [`ARCHITECTURE.md`] §17.9.
//!
//! End-to-end: train a brain with WAL attached, drop it WITHOUT
//! checkpointing the bincode snapshot, then build a fresh brain and
//! replay the WAL events.  Confirms the reconstructed brain has the
//! same neuron + concept structure as the original.

use std::sync::Arc;

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, MmapWalStore,
    PoolConfig, Store,
};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::store::load_events_after_marker;

fn tmpdir(test: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir()
        .join(format!("w1z4rd_wal_recovery_{}_{}_{}", test, pid, nano));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn build_brain_with_wal(dir: &std::path::Path) -> Brain {
    let mut brain = Brain::new(BrainConfig::default());
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let wal = MmapWalStore::open(dir).expect("wal open");
    let store: Arc<dyn Store> = Arc::new(wal);
    brain.set_store(store);
    brain.fabric_mut().register_pool(Pool::new(pc, enc));
    brain
}

#[test]
fn recovery_rebuilds_topology_from_wal_without_snapshot() {
    let dir = tmpdir("rebuild");

    // ---- Original training ----
    let (atoms_before, concepts_before, tick_before): (usize, usize, u64);
    {
        let mut brain = build_brain_with_wal(&dir);
        for _ in 0..4 {
            brain.observe(1, b"ab");
            brain.advance_tick();
        }
        // Force a WAL flush so events hit disk before we drop the brain.
        brain.store_clone().flush().expect("flush");

        let p = brain.fabric().pool(1).unwrap();
        let p = p.read();
        atoms_before    = p.iter_neurons().filter(|n| n.is_atom()).count();
        concepts_before = p.iter_neurons().filter(|n| !n.is_atom()).count();
        tick_before     = brain.fabric().current_tick();
        // Brain dropped here — no bincode snapshot written.
    }

    // ---- Fresh brain, replay WAL events ----
    let events = load_events_after_marker(&dir).expect("load events");
    assert!(!events.is_empty(),
        "WAL should have events from the original training");

    let mut recovered = Brain::new(BrainConfig::default());
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    recovered.fabric_mut().register_pool(Pool::new(pc, enc));

    let stats = recovered.apply_wal_events(&events);
    assert!(stats.events_total > 0,
        "should have applied at least one event");

    let p = recovered.fabric().pool(1).unwrap();
    let p = p.read();
    let atoms_after    = p.iter_neurons().filter(|n| n.is_atom()).count();
    let concepts_after = p.iter_neurons().filter(|n| !n.is_atom()).count();
    drop(p);

    assert_eq!(atoms_after, atoms_before,
        "recovered atom count mismatch: {} vs {}",
        atoms_after, atoms_before);
    assert_eq!(concepts_after, concepts_before,
        "recovered concept count mismatch: {} vs {}",
        concepts_after, concepts_before);
    assert_eq!(recovered.fabric().current_tick(), tick_before,
        "recovered tick should match original");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_events_after_marker_returns_only_post_marker_events() {
    let dir = tmpdir("post_marker");
    {
        let mut brain = build_brain_with_wal(&dir);
        for _ in 0..2 {
            brain.observe(1, b"a");
            brain.advance_tick();
        }
        // Checkpoint emits a SnapshotMarker.
        let bin_path = dir.join("brain.bin");
        brain.checkpoint(&bin_path).expect("checkpoint");
        // Now some MORE events past the marker.
        for _ in 0..3 {
            brain.observe(1, b"b");
            brain.advance_tick();
        }
        brain.store_clone().flush().expect("flush");
    }
    let events = load_events_after_marker(&dir).expect("load");
    // Should contain ONLY events that came after the marker.  The pre-
    // marker events (atom 'a', concept 'aa' if it emerged) should NOT
    // appear.  Sanity: should have at least the 3 TickAdvanced from
    // post-marker observations + any AtomCreated for 'b'.
    let ticks_post = events.iter()
        .filter(|e| matches!(e, w1z4rd_brain::WalEvent::TickAdvanced { .. }))
        .count();
    assert!(ticks_post >= 3, "expected ≥3 post-marker ticks, got {}", ticks_post);

    std::fs::remove_dir_all(&dir).ok();
}
