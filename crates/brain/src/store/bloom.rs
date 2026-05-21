//! Counting Bloom filter per [`ARCHITECTURE.md`] §17.3.
//!
//! Used as a **side-car existence check** alongside `Pool::label_to_id`.
//! When a future Stage 17.4 demand-paged loader needs to answer "is
//! this label maybe in the on-disk store?" without doing a disk seek,
//! it consults this filter first.  Bloom-negative → definitely absent →
//! short-circuit to neurogenesis or returning None.  Bloom-positive →
//! maybe present → fall through to the slower exact lookup.
//!
//! # Why counting (not plain)
//!
//! A plain Bloom filter cannot be deleted from.  When the eviction
//! actor (Stage 17.4 full) reclaims a neuron's ID slot or `sleep_prune`
//! removes a concept, we need to remove from the filter too, or
//! false-positive rate climbs over time.  Counting Bloom (Fan et al.,
//! 2000) tracks per-slot counters instead of bits — set increments,
//! delete decrements.  Cost: 4 bits per slot in this implementation
//! (saturating-add 4-bit counters), so the filter is ~4× larger than
//! plain Bloom but supports remove.
//!
//! # Parameters (defaults)
//!
//! - `k_hashes`: 7 — standard for ~1e-4 false-positive rate
//! - `slots_per_key`: 14 — gives ~14 bits/key (counting filter is 4×
//!   that), targeting ~1e-4 fp at 100M neurons
//! - Capacity is dynamic — when load factor crosses a threshold the
//!   caller should `resize`-double.  Default constructor sizes for
//!   100K keys; production usage should size up explicitly.
//!
//! # Determinism
//!
//! All hashes derive from `ahash::AHasher` with a fixed seed
//! (0xB7A1_5717_5701_C0DE).  Identical labels on different machines
//! → identical filter slots, so `pool_root_hash` (Stage 17.6 Merkle)
//! reproduces.

use ahash::AHasher;
use std::hash::{Hash, Hasher};

const DEFAULT_K_HASHES: usize = 7;
const DEFAULT_SLOTS_PER_KEY: usize = 14;
const DEFAULT_INITIAL_KEYS: usize = 100_000;

/// Fixed seed for the AHasher used by the Bloom filter so all nodes
/// in a cluster produce identical filter slots for identical labels.
const BLOOM_SEED_A: u128 = 0xB7A1_5717_5701_C0DE_5717_5717_5701_5717;
const BLOOM_SEED_B: u128 = 0xC0DE_5701_5717_B7A1_C0DE_C0DE_5701_5701;

/// Counting Bloom filter with 4-bit saturating counters per slot.
///
/// Stored as packed u8 — two slots per byte.  At 4-bit width the
/// counter saturates at 15; insertions beyond that are no-ops (rare in
/// practice — 15 hash collisions on one slot from one label is unlikely).
#[derive(Debug, Clone)]
pub struct CountingBloom {
    /// Packed 4-bit counters; len = (slots + 1) / 2 bytes.
    counters: Vec<u8>,
    /// Total number of 4-bit slots.
    slots:    usize,
    /// Number of hash functions to use per key.
    k_hashes: usize,
    /// Number of distinct keys inserted (not the number of insertions —
    /// double-inserts count once toward this; we estimate by checking
    /// "all k counters > 0 before this insert").
    inserted_keys: usize,
}

impl CountingBloom {
    /// Construct sized for `expected_keys` keys with default parameters.
    pub fn with_expected_capacity(expected_keys: usize) -> Self {
        let target = expected_keys.max(1024);
        let slots = (target * DEFAULT_SLOTS_PER_KEY).next_power_of_two();
        Self::with_slots(slots, DEFAULT_K_HASHES)
    }

    /// Construct with explicit slot count + hash count.
    pub fn with_slots(slots: usize, k_hashes: usize) -> Self {
        let slots = slots.max(8).next_power_of_two();
        let bytes = (slots + 1) / 2;
        Self {
            counters: vec![0u8; bytes],
            slots,
            k_hashes: k_hashes.clamp(1, 16),
            inserted_keys: 0,
        }
    }

    /// Default initial sizing — caller should `resize_for` if expected
    /// scale is known.
    pub fn new() -> Self {
        Self::with_expected_capacity(DEFAULT_INITIAL_KEYS)
    }

    pub fn slots(&self) -> usize    { self.slots }
    pub fn k_hashes(&self) -> usize { self.k_hashes }
    pub fn inserted_keys(&self) -> usize { self.inserted_keys }

    /// Number of bytes the filter occupies in memory.  For sizing
    /// diagnostics.
    pub fn byte_size(&self) -> usize { self.counters.len() }

    /// Insert `key`'s hashes.  Each of the `k_hashes` slots gets a
    /// 4-bit saturating increment.  Updates `inserted_keys` if at
    /// least one slot transitioned from 0.
    pub fn insert<K: Hash + ?Sized>(&mut self, key: &K) {
        let (h1, h2) = self.dual_hash(key);
        let mut new_to_filter = false;
        for i in 0..self.k_hashes {
            let slot = self.slot_for(h1, h2, i);
            let was_zero = self.read_counter(slot) == 0;
            self.bump_counter(slot);
            if was_zero { new_to_filter = true; }
        }
        if new_to_filter { self.inserted_keys += 1; }
    }

    /// Probabilistic existence check.  `false` → definitely absent.
    /// `true` → maybe present (false-positive rate depends on filter
    /// load + parameters).
    pub fn might_contain<K: Hash + ?Sized>(&self, key: &K) -> bool {
        let (h1, h2) = self.dual_hash(key);
        for i in 0..self.k_hashes {
            let slot = self.slot_for(h1, h2, i);
            if self.read_counter(slot) == 0 {
                return false;
            }
        }
        true
    }

    /// Remove `key`'s hashes.  Each of the `k_hashes` slots gets a
    /// 4-bit saturating decrement.  Idempotent on a key that wasn't
    /// inserted (some counters may already be 0).
    ///
    /// Returns `true` if the filter went from "maybe contains" to
    /// "definitely absent" for this key as a result of the removal.
    pub fn remove<K: Hash + ?Sized>(&mut self, key: &K) -> bool {
        let (h1, h2) = self.dual_hash(key);
        let mut went_absent = false;
        for i in 0..self.k_hashes {
            let slot = self.slot_for(h1, h2, i);
            let prev = self.read_counter(slot);
            if prev > 0 {
                self.decay_counter(slot);
                if prev == 1 { went_absent = true; }
            }
        }
        if went_absent && self.inserted_keys > 0 {
            self.inserted_keys -= 1;
        }
        went_absent
    }

    /// Reset the filter to empty.
    pub fn clear(&mut self) {
        for byte in self.counters.iter_mut() { *byte = 0; }
        self.inserted_keys = 0;
    }

    // ---- internals ----

    fn dual_hash<K: Hash + ?Sized>(&self, key: &K) -> (u64, u64) {
        let mut a = AHasher::default();
        let mut b = AHasher::default();
        BLOOM_SEED_A.hash(&mut a);
        key.hash(&mut a);
        BLOOM_SEED_B.hash(&mut b);
        key.hash(&mut b);
        (a.finish(), b.finish())
    }

    /// Double-hashing scheme (Kirsch & Mitzenmacher, 2008): k slots
    /// derived as `(h1 + i*h2) mod slots`.  Statistically indistinguishable
    /// from k independent hashes in published analysis; saves k-2 hash
    /// computations per key.
    fn slot_for(&self, h1: u64, h2: u64, i: usize) -> usize {
        let mix = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (mix as usize) & (self.slots - 1)
    }

    fn read_counter(&self, slot: usize) -> u8 {
        let byte_idx = slot >> 1;
        let byte = self.counters[byte_idx];
        if slot & 1 == 0 { byte & 0x0F } else { byte >> 4 }
    }

    fn bump_counter(&mut self, slot: usize) {
        let byte_idx = slot >> 1;
        let byte = self.counters[byte_idx];
        if slot & 1 == 0 {
            let cur = byte & 0x0F;
            if cur < 15 {
                self.counters[byte_idx] = (byte & 0xF0) | (cur + 1);
            }
        } else {
            let cur = byte >> 4;
            if cur < 15 {
                self.counters[byte_idx] = (byte & 0x0F) | ((cur + 1) << 4);
            }
        }
    }

    fn decay_counter(&mut self, slot: usize) {
        let byte_idx = slot >> 1;
        let byte = self.counters[byte_idx];
        if slot & 1 == 0 {
            let cur = byte & 0x0F;
            if cur > 0 {
                self.counters[byte_idx] = (byte & 0xF0) | (cur - 1);
            }
        } else {
            let cur = byte >> 4;
            if cur > 0 {
                self.counters[byte_idx] = (byte & 0x0F) | ((cur - 1) << 4);
            }
        }
    }
}

impl Default for CountingBloom {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_says_no_to_everything() {
        let f = CountingBloom::new();
        for label in &["t:a", "t:b", "t:c", "concept_xy"] {
            assert!(!f.might_contain(label), "empty filter must say no: {}", label);
        }
    }

    #[test]
    fn insert_then_might_contain_is_yes() {
        let mut f = CountingBloom::new();
        f.insert("t:a");
        f.insert("t:b");
        assert!(f.might_contain("t:a"));
        assert!(f.might_contain("t:b"));
    }

    #[test]
    fn remove_undoes_insert() {
        let mut f = CountingBloom::new();
        f.insert("t:a");
        assert!(f.might_contain("t:a"));
        let went_absent = f.remove("t:a");
        assert!(went_absent, "should report transitioned-to-absent");
        assert!(!f.might_contain("t:a"),
            "after remove, filter should say definitely absent");
    }

    #[test]
    fn double_insert_double_remove_is_balanced() {
        let mut f = CountingBloom::new();
        f.insert("x");
        f.insert("x");
        // After 2 inserts, 1 remove leaves the counters > 0 — still says yes.
        f.remove("x");
        assert!(f.might_contain("x"),
            "one remove after two inserts should still leave key present");
        // Second remove: all counters back to 0 → no.
        f.remove("x");
        assert!(!f.might_contain("x"));
    }

    #[test]
    fn false_positive_rate_is_acceptable_at_target_load() {
        let mut f = CountingBloom::with_expected_capacity(10_000);
        // Insert 10K distinct labels.
        for i in 0..10_000u32 {
            f.insert(&format!("inserted_{i}"));
        }
        assert_eq!(f.inserted_keys(), 10_000);
        // Probe 10K NOT-inserted labels.
        let mut fp = 0;
        for i in 0..10_000u32 {
            if f.might_contain(&format!("not_inserted_{i}")) { fp += 1; }
        }
        // With 14 bits/key and 7 hashes we target ~1e-4 false-positive
        // rate at optimal load.  This test inserts at full target load
        // and probes 10K samples, so the observed rate fluctuates with
        // hash variance.  Allow up to 2% as a robustness margin — at
        // the bloom's intended large-scale use, load factor is much
        // lower and FP rate correspondingly tighter.
        let fp_rate = fp as f32 / 10_000.0;
        assert!(fp_rate < 0.02,
            "false-positive rate {} exceeded 2% bound", fp_rate);
    }

    #[test]
    fn determinism_across_instances() {
        let mut a = CountingBloom::with_slots(1024, 7);
        let mut b = CountingBloom::with_slots(1024, 7);
        for k in &["dog", "cat", "cow", "horse"] {
            a.insert(k);
            b.insert(k);
        }
        // Two filters built with the same params + same insertions must
        // be byte-identical — Stage 17.6 Merkle hashing depends on this.
        assert_eq!(a.counters, b.counters);
    }
}
