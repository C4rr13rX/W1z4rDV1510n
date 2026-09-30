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
    // 32 bytes per entry is generous for 8 (pointer) + 4 (value) plus slack.
    for map in ["lifetime_recurrences", "tentative_promoted"] {
        let b = bytes[map].as_u64().unwrap();
        let n = entries[map].as_u64().unwrap().max(1);
        assert!(
            b / n <= 32,
            "{map} charges {} bytes per entry, so it still owns its keys",
            b / n
        );
    }
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
