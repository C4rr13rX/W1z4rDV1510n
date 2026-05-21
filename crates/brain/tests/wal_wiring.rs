//! Stage 17.9 forward-path integration test per [`ARCHITECTURE.md`] §17.9.
//!
//! Confirms that:
//! 1. A brain with `MmapWalStore` attached emits the right events when it
//!    learns (AtomCreated for atoms, ConceptEmerged for concepts, TickAdvanced
//!    for ticks, PoolRegistered baseline at pool attach time).
//! 2. The WAL file actually grows on disk.
//! 3. The events read back are deserialisable and match the expected variant
//!    counts.
//! 4. A brain with NoopStore (default) produces no WAL — i.e. the side-car
//!    is opt-in and zero-cost when unused.
//!
//! This is the smallest end-to-end test that demonstrates the §17 wiring
//! works for a real observation path through Brain → Fabric → Pool.

use std::path::PathBuf;
use std::sync::Arc;

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding,
    MmapWalStore, PoolConfig, Store, WalEvent,
};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::store::wal::WalReader;

fn tmpdir(test: &str) -> PathBuf {
    let pid = std::process::id();
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir()
        .join(format!("w1z4rd_brain_wal_wiring_{}_{}_{}", test, pid, nano));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Build a small brain with one text pool and attach the WAL.
fn build_brain_with_wal(data_dir: &std::path::Path) -> Brain {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 64;
    let mut brain = Brain::new(cfg);

    // Attach the WAL.
    let wal = MmapWalStore::open(data_dir).expect("open WAL");
    let store: Arc<dyn Store> = Arc::new(wal);
    brain.set_store(store);

    // Register a text pool AFTER attaching the store so we exercise the
    // register_pool → PoolRegistered event path.
    let mut pool_cfg = PoolConfig::defaults("text", 1);
    pool_cfg.recent_atoms_window = 16;
    pool_cfg.concept_emergence_threshold = 2;
    let encoding: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let pool = Pool::new(pool_cfg, encoding);
    brain.fabric_mut().register_pool(pool);

    brain
}

#[test]
fn wal_captures_atom_concept_and_tick_events_for_a_short_training_run() {
    let dir = tmpdir("captures");
    let mut brain = build_brain_with_wal(&dir);

    // Observe a short repeating sequence; concept emergence will fire when
    // the same 2-tuple recurs twice (concept_emergence_threshold = 2).
    for _ in 0..4 {
        brain.fabric_mut().observe(1, b"ab");
        brain.fabric_mut().advance_tick();
    }

    // Flush so the WAL on-disk reflects everything we wrote.
    brain.store_clone().flush().expect("flush");

    // Read the WAL back.
    let replay = MmapWalStore::open_replay_only(&dir).expect("open replay");
    let events: Vec<WalEvent> = WalReader::new(replay)
        .map(|r| r.expect("event ok"))
        .collect();

    // We expect at least:
    //  - 1 PoolRegistered (for pool 1)
    //  - 2 AtomCreated (for atoms 'a' and 'b')
    //  - >= 1 ConceptEmerged (the 'ab' bigram concept after 2 repeats)
    //  - 4 TickAdvanced (one per observe loop)
    let mut count = std::collections::HashMap::<&'static str, u32>::new();
    for ev in &events {
        *count.entry(ev.variant_name()).or_insert(0) += 1;
    }

    assert!(count.get("PoolRegistered").copied().unwrap_or(0) >= 1,
        "expected ≥1 PoolRegistered, got {:?}", count);
    assert!(count.get("AtomCreated").copied().unwrap_or(0) >= 2,
        "expected ≥2 AtomCreated (atoms a + b), got {:?}", count);
    assert!(count.get("ConceptEmerged").copied().unwrap_or(0) >= 1,
        "expected ≥1 ConceptEmerged after repeats, got {:?}", count);
    assert!(count.get("TickAdvanced").copied().unwrap_or(0) >= 4,
        "expected ≥4 TickAdvanced, got {:?}", count);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn checkpoint_emits_snapshot_marker_and_flushes_wal() {
    let dir = tmpdir("marker");
    let mut brain = build_brain_with_wal(&dir);

    // Train a tiny bit.
    brain.fabric_mut().observe(1, b"xy");
    brain.fabric_mut().advance_tick();

    // Checkpoint — should write the snapshot bin AND emit a marker into the WAL.
    let bin_path = dir.join("brain.bin");
    brain.checkpoint(&bin_path).expect("checkpoint");
    assert!(bin_path.exists(), "brain.bin must exist after checkpoint");

    let replay = MmapWalStore::open_replay_only(&dir).expect("open replay");
    let events: Vec<WalEvent> = WalReader::new(replay)
        .map(|r| r.expect("event ok"))
        .collect();
    let markers = events.iter()
        .filter(|e| matches!(e, WalEvent::SnapshotMarker { .. }))
        .count();
    assert!(markers >= 1,
        "expected ≥1 SnapshotMarker after checkpoint, events = {:?}",
        events.iter().map(|e| e.variant_name()).collect::<Vec<_>>());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn noop_store_brain_produces_no_wal_file() {
    let dir = tmpdir("noop");

    // Brain WITHOUT explicit set_store — uses default NoopStore.
    let mut brain = Brain::new(BrainConfig::default());
    let mut pool_cfg = PoolConfig::defaults("text", 1);
    pool_cfg.recent_atoms_window = 16;
    let pool = Pool::new(pool_cfg,
        Box::new(BytePassthroughEncoding { prefix: "t" }));
    brain.fabric_mut().register_pool(pool);

    // Observe + tick — nothing should land in the data dir since no store
    // was attached and we never call set_store.
    for _ in 0..3 {
        brain.fabric_mut().observe(1, b"a");
        brain.fabric_mut().advance_tick();
    }
    // NoopStore.flush() is inert.
    brain.store_clone().flush().expect("noop flush");

    // No WAL file should have been created.
    assert!(!dir.join("brain.wal").exists(),
        "NoopStore must not create brain.wal");

    std::fs::remove_dir_all(&dir).ok();
}
