//! The fingerprint-keyed indexes must hold ONE copy of each key.
//!
//! Measured 2026-09-30 at scale 64 of the scorecard (`--census`): global index
//! bytes were 12.47 MB, of which `lifetime_recurrences` 5.13 MB over 9,728
//! entries and `tentative_promoted` 5.13 MB storing the SAME
//! `MomentFingerprint` a second time. A fingerprint owns three `Vec`s and holds
//! every atom id of query and answer twice, so a ~22-atom fact cost ~1.1 KB
//! across the two maps -- ~1.1 GB at 1 M facts.
//!
//! `register_fingerprint` cloned the whole fingerprint into `moment_history`,
//! `binding_recurrences`, `lifetime_recurrences` and `tentative_promoted`. The
//! fingerprint is never mutated after construction, so the five indexes now
//! share one `Arc` and charge themselves a pointer each.
//!
//! The assertions are paired on purpose: the second half proves the brain
//! really did learn the facts, so a census that reported zero because nothing
//! was indexed at all could not make the first half pass vacuously.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const FACTS: usize = 64;

fn subject() -> Brain {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);
    for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
        let mut pc = PoolConfig::defaults(name, id);
        pc.recent_atoms_window = 2048;
        pc.concept_emergence_threshold = 2;
        pc.max_concept_member_count = 64;
        pc.decay_rate = 0.0001;
        pc.prune_floor = 0.005;
        brain.create_pool(pc, Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>);
    }
    brain
}

fn census(brain: &Brain) -> serde_json::Value {
    brain.global_index_sizes()
}

#[test]
fn a_fingerprint_key_is_stored_once_however_many_indexes_hold_it() {
    let mut brain = subject();
    for fact in 0..FACTS {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{fact:03} lamp color?").into_bytes()),
            (ANSWER_POOL, b"red".to_vec()),
        ]);
    }
    let c = census(&brain);
    let bytes = &c["bytes"];
    let entries = &c["entries"];

    let lifetime_entries = entries["lifetime_recurrences"].as_u64().unwrap();
    let tentative_entries = entries["tentative_promoted"].as_u64().unwrap();
    let distinct = entries["fingerprint_keys"].as_u64().unwrap();
    let shared_bytes = bytes["fingerprint_keys"].as_u64().unwrap();

    // Not vacuous: the brain learned every fact and both indexes hold them.
    assert_eq!(
        lifetime_entries, FACTS as u64,
        "lifetime_recurrences did not index every trained fact"
    );
    assert_eq!(
        tentative_entries, FACTS as u64,
        "tentative_promoted did not index every trained fact"
    );
    assert!(
        shared_bytes > 0,
        "no fingerprint payload was counted, so the sharing check is vacuous"
    );

    // One key per distinct moment, however many indexes point at it.
    assert_eq!(
        distinct, FACTS as u64,
        "{} indexed keys against {} facts -- the indexes are not sharing",
        distinct, FACTS
    );

    // A map now costs a pointer plus its value per entry, not a fingerprint.
    // The bound is the TABLE's own arithmetic rather than a constant: an entry
    // is `(Arc<MomentFingerprint>, u32)` plus hashbrown's control byte, and the
    // table allocates a power-of-two bucket count at no more than 7/8 load, so
    // the per-entry charge is up to ~2.3 buckets' worth. A flat 32 was written
    // when the census counted `len` and no bucket table at all; the real
    // allocation at 64 facts is 128 buckets x 17 B = 34 B per entry, which is
    // the table being measured properly and not the map owning its keys.
    let entry = std::mem::size_of::<(std::sync::Arc<()>, u32)>() + 1;
    for map in ["lifetime_recurrences", "tentative_promoted"] {
        let b = bytes[map].as_u64().unwrap();
        let n = entries[map].as_u64().unwrap().max(1);
        assert!(
            (b / n) as usize <= 3 * entry,
            "{map} charges {} bytes per entry against a {} B bucket, so it still owns its keys",
            b / n,
            entry
        );
    }

    // The discriminator the constant above cannot express: a map that owned its
    // keys would charge what the key PAYLOAD costs. `fingerprint_keys` is that
    // payload, counted once for the shared Arc, so a per-entry charge an order
    // below it is the sharing this file is named for -- and it stays true if
    // the fingerprint grows, which a fixed byte count does not.
    let payload_per_fact = shared_bytes / distinct.max(1);
    for map in ["lifetime_recurrences", "tentative_promoted"] {
        let per = bytes[map].as_u64().unwrap() / entries[map].as_u64().unwrap().max(1);
        assert!(
            per * 4 < payload_per_fact,
            "{map} charges {per} B per entry against a {payload_per_fact} B fingerprint -- \
             that is the key, not a pointer to it"
        );
    }
}

/// A binding's dedup label must not grow with the fact it names.
///
/// Measured 2026-09-30 at scale 64: the label SPELLED the membership
/// ("p1n0|p1n0|p1n0|p1n2|..." plus an ordered hash), 170 bytes per binding,
/// held once in `pool.label_index` (1.65 MB) and again in the neuron. Nothing
/// parses it -- it is written once and read back only through `label_to_id`.
#[test]
fn a_binding_label_is_a_symbol_not_a_spelled_out_membership() {
    let mut brain = subject();
    // Two questions an order of magnitude apart in length. A label that spells
    // its members grows with the second; a symbol does not.
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, b"r000 lamp color?".to_vec()),
        (ANSWER_POOL, b"red".to_vec()),
    ]);
    brain.pretrain_binding_episode(&[
        (
            QUERY_POOL,
            format!("r001 {} lamp color?", "very ".repeat(40)).into_bytes(),
        ),
        (ANSWER_POOL, b"blue".to_vec()),
    ]);

    let binding_pool = brain.binding_pool_id();
    let pool = brain.fabric().pool(binding_pool).expect("binding pool");
    let pool = pool.read();
    let labels: Vec<String> = pool
        .iter_neurons()
        .filter(|n| !n.is_atom())
        .map(|n| n.label.clone())
        .collect();
    drop(pool);

    assert!(
        labels.len() >= 2,
        "expected a binding per episode, got {}",
        labels.len()
    );
    let longest = labels.iter().map(|l| l.len()).max().unwrap();
    assert!(
        longest <= 48,
        "longest binding label is {longest} bytes ({:?}) -- it still spells its members",
        labels.iter().max_by_key(|l| l.len()).unwrap()
    );
    // Distinct episodes must still get distinct labels, or dedup collapses them.
    let mut sorted = labels.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), labels.len(), "two episodes share one label");
}

#[test]
fn recall_survives_the_shared_key() {
    let mut brain = subject();
    for fact in 0..FACTS {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{fact:03} lamp color?").into_bytes()),
            (ANSWER_POOL, format!("shade{fact:03}").into_bytes()),
        ]);
    }
    // The same path the scorecard and crates/node/src/brain_api.rs answer on.
    let mut hits = 0usize;
    for fact in 0..FACTS {
        brain.observe_read_only(QUERY_POOL, format!("r{fact:03} lamp color?").as_bytes());
        let legacy = brain.integrate(QUERY_POOL, ANSWER_POOL);
        let answer = brain
            .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
            .or(legacy.answer)
            .unwrap_or_default();
        let _ = brain.finish_read_only_inference();
        if answer == format!("shade{fact:03}").into_bytes() {
            hits += 1;
        }
    }
    assert_eq!(hits, FACTS, "recall fell to {hits}/{FACTS} under the shared key");
}
