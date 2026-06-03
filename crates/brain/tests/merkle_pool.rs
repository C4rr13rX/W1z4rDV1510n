//! Stage 17.6 — Pool::merkle_root integration tests per
//! [`ARCHITECTURE.md`] §17.6.
//!
//! Confirms:
//! 1. Two pools trained identically (same encoding, same atoms, same
//!    tick) produce byte-identical Merkle roots — the precondition for
//!    cluster anti-entropy sync.
//! 2. A pool that learns one extra atom has a different root.
//! 3. Snapshot → restore round-trip preserves the Merkle root.

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
fn identical_training_produces_identical_roots() {
    let mut p1 = make_pool();
    let mut p2 = make_pool();
    for _ in 0..3 {
        p1.observe_frame(b"ab", 0, None);
        p2.observe_frame(b"ab", 0, None);
    }
    let r1 = p1.merkle_root(/*fabric_tick*/ 3);
    let r2 = p2.merkle_root(3);
    assert_eq!(r1, r2,
        "two pools with identical training history must produce identical roots");
}

#[test]
fn extra_atom_changes_the_root() {
    let mut p1 = make_pool();
    let mut p2 = make_pool();
    for _ in 0..3 { p1.observe_frame(b"ab", 0, None); }
    for _ in 0..3 { p2.observe_frame(b"ab", 0, None); }
    p2.observe_frame(b"c", 0, None);  // one extra atom in p2
    let r1 = p1.merkle_root(3);
    let r2 = p2.merkle_root(3);
    assert_ne!(r1, r2,
        "diverging training must produce different roots");
}

#[test]
fn root_is_stable_across_snapshot_restore() {
    let mut p1 = make_pool();
    for _ in 0..3 { p1.observe_frame(b"xy", 0, None); }
    let r_before = p1.merkle_root(3);

    let snap: PoolSnapshot = p1.snapshot();
    let enc: Box<dyn AtomEncoding> =
        Box::new(BytePassthroughEncoding { prefix: "t" });
    let p2 = Pool::from_snapshot(snap, enc);
    let r_after = p2.merkle_root(3);

    assert_eq!(r_before, r_after,
        "Merkle root must survive a snapshot round-trip");
}

#[test]
fn root_hex_is_64_chars() {
    let p = make_pool();
    let r = p.merkle_root(0);
    assert_eq!(r.to_hex().len(), 64);
    assert!(r.to_hex().chars().all(|c| c.is_ascii_hexdigit()));
}
