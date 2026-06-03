//! Stage 17.4 step 1 — Pool cold-tier primitive tests per
//! [`ARCHITECTURE.md`] §17.4.
//!
//! Confirms:
//! 1. A concept neuron can be evicted to the cold tier, then paged back
//!    in, with terminals + members preserved.
//! 2. While evicted, the neuron's slot in `Pool::neurons` is still
//!    present (label_to_id still resolves) but `terminals` is empty —
//!    proving the memory cost was actually shed.
//! 3. `is_evicted` and `evicted_count` track the eviction state.
//! 4. Atoms refuse to evict (the policy never wants atoms on disk).
//! 5. Calling `evict` twice or `page_in` on a never-evicted neuron is
//!    idempotent.

use std::sync::Arc;

use w1z4rd_brain::{AtomEncoding, BytePassthroughEncoding, PoolConfig};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::store::ColdTier;

fn tmpdir(test: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir()
        .join(format!("w1z4rd_pool_cold_{}_{}_{}", test, pid, nano));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn make_pool_with_cold(dir: &std::path::Path) -> Pool {
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let mut pool = Pool::new(pc, enc);
    let tier = Arc::new(ColdTier::open(dir.join("pool.cold")).unwrap());
    pool.set_cold_tier(tier);
    pool
}

#[test]
fn evict_then_page_in_round_trips_concept() {
    let dir = tmpdir("rt_concept");
    let mut pool = make_pool_with_cold(&dir);
    // Train enough to emerge at least one concept.
    for _ in 0..4 { pool.observe_frame(b"ab", 0, None); }

    // Find a concept ID.
    let concept_id = pool.iter_neurons()
        .find(|n| !n.is_atom())
        .map(|n| n.id)
        .expect("expected at least one concept after training");

    // Snapshot pre-evict state.
    let pre = pool.get(concept_id).unwrap().clone();
    let pre_terminal_count = pre.terminals.len();
    let pre_member_count   = pre.members.len();
    assert!(pre_terminal_count > 0, "concept should have terminals");
    assert!(pre_member_count > 0,   "concept should have members");

    // Evict.
    let evicted = pool.evict_neuron(concept_id).expect("evict");
    assert!(evicted, "evict should report transitioned");
    assert!(pool.is_evicted(concept_id));
    assert_eq!(pool.evicted_count(), 1);

    // While evicted, slot still exists but terminals are empty.
    let evicted_view = pool.get(concept_id).unwrap();
    assert_eq!(evicted_view.terminals.len(), 0,
        "evicted concept must have zero terminals in RAM");
    assert_eq!(evicted_view.id, concept_id,
        "id preserved while evicted");
    assert_eq!(evicted_view.label, pre.label,
        "label preserved while evicted (label_to_id continuity)");

    // Page back in.
    let paged = pool.page_in_neuron(concept_id).expect("page_in");
    assert!(paged, "page_in should report restored");
    assert!(!pool.is_evicted(concept_id));
    assert_eq!(pool.evicted_count(), 0);

    let restored = pool.get(concept_id).unwrap();
    assert_eq!(restored.terminals.len(), pre_terminal_count,
        "terminal count restored");
    assert_eq!(restored.members.len(),   pre_member_count,
        "member count restored");
    assert_eq!(restored.label, pre.label);
    assert_eq!(restored.kind,  pre.kind);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn evict_atom_is_refused() {
    let dir = tmpdir("no_atom");
    let mut pool = make_pool_with_cold(&dir);
    for _ in 0..2 { pool.observe_frame(b"a", 0, None); }
    let atom_id = pool.iter_neurons()
        .find(|n| n.is_atom())
        .map(|n| n.id)
        .expect("expected an atom");

    let res = pool.evict_neuron(atom_id).expect("evict call");
    assert!(!res, "evict on atom must return false (refused by policy)");
    assert!(!pool.is_evicted(atom_id), "atom must not be marked evicted");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn double_evict_is_idempotent() {
    let dir = tmpdir("double_evict");
    let mut pool = make_pool_with_cold(&dir);
    for _ in 0..4 { pool.observe_frame(b"xy", 0, None); }
    let concept_id = pool.iter_neurons()
        .find(|n| !n.is_atom())
        .map(|n| n.id)
        .expect("concept");

    let first  = pool.evict_neuron(concept_id).unwrap();
    let second = pool.evict_neuron(concept_id).unwrap();
    assert!(first, "first evict transitions");
    assert!(!second, "second evict is no-op (already evicted)");
    assert_eq!(pool.evicted_count(), 1);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn page_in_on_never_evicted_neuron_is_idempotent() {
    let dir = tmpdir("noevict_pagein");
    let mut pool = make_pool_with_cold(&dir);
    for _ in 0..3 { pool.observe_frame(b"ab", 0, None); }
    let any_id = pool.iter_neurons().map(|n| n.id).next().expect("at least one");

    let r = pool.page_in_neuron(any_id).expect("page_in");
    assert!(!r, "page_in on non-evicted should return false");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn evict_without_cold_tier_errors() {
    // Pool with NO cold tier attached.
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let mut pool = Pool::new(pc, enc);
    for _ in 0..4 { pool.observe_frame(b"ab", 0, None); }
    let concept_id = pool.iter_neurons()
        .find(|n| !n.is_atom())
        .map(|n| n.id)
        .expect("concept");

    let err = pool.evict_neuron(concept_id).err()
        .expect("evict without cold tier must error");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported,
        "expected Unsupported error kind, got {:?}", err.kind());
}

#[test]
fn live_count_tracks_evictions() {
    let dir = tmpdir("live_count");
    let mut pool = make_pool_with_cold(&dir);
    for _ in 0..4 { pool.observe_frame(b"ab", 0, None); }
    let total_before = pool.neuron_count();
    let live_before  = pool.live_count();
    assert_eq!(live_before, total_before);

    // Evict every non-atom.
    let concepts: Vec<_> = pool.iter_neurons()
        .filter(|n| !n.is_atom())
        .map(|n| n.id)
        .collect();
    let mut evicted = 0;
    for cid in &concepts {
        if pool.evict_neuron(*cid).unwrap() { evicted += 1; }
    }
    assert!(evicted > 0, "should have evicted at least one concept");
    assert_eq!(pool.evicted_count(), evicted);
    assert_eq!(pool.live_count(), total_before - evicted);
    assert_eq!(pool.neuron_count(), total_before,
        "neuron_count is total — unchanged by eviction (placeholder slots stay)");

    std::fs::remove_dir_all(&dir).ok();
}
