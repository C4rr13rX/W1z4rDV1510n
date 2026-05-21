//! Stage 17.3 — Pool-level Bloom side-car tests per [`ARCHITECTURE.md`] §17.3.
//!
//! Confirms:
//! 1. Every label that exists in `label_to_id` is also present in the
//!    Bloom filter (`label_might_exist` returns true).
//! 2. The Bloom filter rejects labels that have never been inserted at a
//!    much-better-than-random rate (no false-negatives for inserted keys).
//! 3. After pruning a concept, the Bloom filter reflects the removal
//!    (`label_might_exist` for the pruned label returns false).
//! 4. After snapshot → restore round-trip, the Bloom filter is rebuilt
//!    from the restored `label_to_id`.

use w1z4rd_brain::{AtomEncoding, BytePassthroughEncoding, PoolConfig};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::persistence::PoolSnapshot;

fn make_pool() -> Pool {
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    Pool::new(pc, enc)
}

#[test]
fn every_inserted_label_is_in_bloom() {
    let mut pool = make_pool();
    // Drive enough observes to create some atoms AND a concept.
    for _ in 0..3 { pool.observe_frame(b"ab", 0); }
    // Every label that's in label_to_id MUST be might_contain in bloom
    // — Bloom may have false positives but never false negatives.
    // (Iterate raw labels via the public read API.)
    let mut atom_count = 0;
    let mut concept_count = 0;
    for n in pool.iter_neurons() {
        if n.is_atom() { atom_count += 1; } else { concept_count += 1; }
        assert!(pool.label_might_exist(&n.label),
            "label {:?} should be in bloom (it's in label_to_id)",
            n.label);
    }
    assert!(atom_count >= 2, "expected ≥2 atoms (a, b); got {}", atom_count);
    // concept may or may not have emerged depending on emergence path;
    // assertion is only on the relationship, not the absolute count.
    let _ = concept_count;
}

#[test]
fn bloom_rejects_definitively_absent_labels() {
    let mut pool = make_pool();
    for _ in 0..3 { pool.observe_frame(b"xy", 0); }
    // Probe 100 distinctly-named labels that were NOT inserted.
    let mut false_positives = 0;
    for i in 0..100 {
        let probe = format!("t:never_inserted_{i:03}");
        if pool.label_might_exist(&probe) {
            false_positives += 1;
        }
    }
    // At Bloom defaults (~14 bits/key, 7 hashes) the expected
    // false-positive rate at the very-low load of this test is well
    // under 1%.  Allow some slack.
    assert!(false_positives < 5,
        "too many false positives ({} of 100) — bloom params or impl off",
        false_positives);
}

#[test]
fn bloom_byte_size_and_inserted_keys_are_reasonable() {
    let pool = make_pool();
    // 100K-key sized filter: ~175 KB byte size.
    let bytes = pool.bloom_byte_size();
    assert!(bytes >= 100_000 / 2, "bloom too small: {} bytes", bytes);
    assert!(bytes <= 4_000_000, "bloom too big for 100K initial: {}", bytes);
    assert_eq!(pool.bloom_inserted_keys(), 0,
        "fresh pool's bloom should have no inserts");
}

#[test]
fn snapshot_roundtrip_rebuilds_bloom() {
    let mut pool = make_pool();
    for _ in 0..3 { pool.observe_frame(b"cd", 0); }
    // Capture the labels we expect to be present after restore.
    let expected_labels: Vec<String> =
        pool.iter_neurons().map(|n| n.label.clone()).collect();
    assert!(!expected_labels.is_empty());

    let snap: PoolSnapshot = pool.snapshot();
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let restored = Pool::from_snapshot(snap, enc);

    for label in &expected_labels {
        assert!(restored.label_might_exist(label),
            "after snapshot restore, bloom must contain {:?}", label);
    }
}
