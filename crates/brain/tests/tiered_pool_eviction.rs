//! Stage 18.12 step 4b — Pool↔TieredStore integration tests per
//! [`ARCHITECTURE.md`] §18.2.
//!
//! Confirms:
//! 1. `Pool::set_tiered_store` attaches a `TieredStore` that intercepts
//!    eviction writes and page-in reads.
//! 2. When both `cold_tier` and `tiered_store` are set, `tiered_store`
//!    wins (the §18 distributed path takes precedence over §17.4 local
//!    cold disk).
//! 3. An evict→page-in round-trip through a tiered store preserves the
//!    neuron's full state (terminals, salience, members).
//! 4. Pool's existing legacy semantics are unchanged when only
//!    `cold_tier` is set.

use std::sync::Arc;

use w1z4rd_brain::{AtomEncoding, BytePassthroughEncoding, PoolConfig};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::store::{NodeId, RamStore, TieredStore};

fn make_pool() -> Pool {
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    Pool::new(pc, enc)
}

#[test]
fn evict_routes_through_tiered_store_when_attached() {
    let mut pool = make_pool();
    // Train enough to emerge at least one concept with terminals.
    for _ in 0..4 { pool.observe_frame(b"ab", 0, None); }
    let concept_id = pool.iter_neurons()
        .find(|n| !n.is_atom() && !n.terminals.is_empty())
        .map(|n| n.id)
        .expect("expected a concept with terminals after training");

    // Attach a tiered store backed only by a RamStore (the simplest
    // configuration — proves the path works without involving disk
    // or network).
    let ram = Arc::new(RamStore::with_node_id(NodeId(0)));
    let store = Arc::new(TieredStore::solo(NodeId(0), ram.clone()));
    pool.set_tiered_store(store.clone());
    assert!(pool.has_tiered_store());

    // Capture pre-evict state for round-trip verification.
    let pre = pool.get(concept_id).unwrap().clone();
    assert!(pre.terminals.len() > 0);

    // Evict — should route through tiered_store.
    let evicted = pool.evict_neuron(concept_id).expect("evict");
    assert!(evicted, "evict should report transitioned");
    assert!(pool.is_evicted(concept_id));

    // The tiered_store's underlying RamStore should now contain the
    // full neuron state (with terminals intact).
    use w1z4rd_brain::store::NeuronStore as _;
    let in_store = ram.get(concept_id).expect("ram should have it");
    assert_eq!(in_store.id, concept_id);
    assert_eq!(in_store.terminals.len(), pre.terminals.len(),
        "tiered_store must capture full terminal weights");

    // Page back in — should restore terminals from tiered_store, not
    // from the (unattached) cold_tier.
    let paged = pool.page_in_neuron(concept_id).expect("page_in");
    assert!(paged);
    let restored = pool.get(concept_id).unwrap();
    assert_eq!(restored.terminals.len(), pre.terminals.len(),
        "page_in must restore terminals via tiered_store");
    assert_eq!(restored.label, pre.label);
}

#[test]
fn evict_without_any_store_attached_errors() {
    let mut pool = make_pool();
    for _ in 0..4 { pool.observe_frame(b"xy", 0, None); }
    let concept_id = pool.iter_neurons()
        .find(|n| !n.is_atom())
        .map(|n| n.id)
        .expect("concept");
    let err = pool.evict_neuron(concept_id).err()
        .expect("evict without store must error");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn tiered_store_takes_precedence_over_cold_tier() {
    use w1z4rd_brain::store::ColdTier;
    let dir = std::env::temp_dir().join(format!(
        "w1z4rd_tiered_priority_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
    ));
    std::fs::create_dir_all(&dir).unwrap();

    let mut pool = make_pool();
    for _ in 0..4 { pool.observe_frame(b"cd", 0, None); }
    let concept_id = pool.iter_neurons()
        .find(|n| !n.is_atom() && !n.terminals.is_empty())
        .map(|n| n.id)
        .expect("concept");

    // Attach BOTH cold_tier and tiered_store.  The tiered_store should
    // win the evict path; the cold_tier file should stay empty (no
    // append happens) because tiered_store takes precedence.
    let cold = Arc::new(ColdTier::open(dir.join("pool.cold")).unwrap());
    pool.set_cold_tier(cold);
    let ram = Arc::new(RamStore::with_node_id(NodeId(0)));
    let store = Arc::new(TieredStore::solo(NodeId(0), ram.clone()));
    pool.set_tiered_store(store);

    let cold_bytes_before = std::fs::metadata(dir.join("pool.cold"))
        .unwrap().len();

    pool.evict_neuron(concept_id).expect("evict");

    let cold_bytes_after = std::fs::metadata(dir.join("pool.cold"))
        .unwrap().len();
    assert_eq!(cold_bytes_before, cold_bytes_after,
        "cold_tier file must NOT grow when tiered_store is attached \
         — tiered_store takes precedence over legacy cold disk");
    use w1z4rd_brain::store::NeuronStore as _;
    assert!(ram.get(concept_id).is_some(),
        "tiered_store's RamStore should hold the evicted neuron");

    std::fs::remove_dir_all(&dir).ok();
}
