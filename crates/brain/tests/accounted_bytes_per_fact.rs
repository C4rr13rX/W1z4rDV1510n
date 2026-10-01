//! The M1 ratchet: accounted bytes PER FACT must not rise with the corpus.
//!
//! The RAM promise is that peak memory is a few hundred MB whatever the brain
//! knows. `tools/scorecard.py` gates the end of that chain -- peak_mb must not
//! rise more than 15% over the baseline -- which catches a regression only
//! after it has already cost megabytes, and never says WHICH structure did it.
//! This test guards the middle of the chain, where the defect class actually
//! lives: a map that charges more per fact as facts arrive.
//!
//! Measured 2026-10-01 with `scorecard.exe --phase infer --census` at scales 1
//! and 64 (152 and 9,728 facts), which is where the ceilings below come from.
//! Every named global bucket is at worst FLAT per fact across that 64x change:
//!
//!   bucket                       s1 B/fact   s64 B/fact
//!   fingerprint_keys                 287.2        287.3
//!   lifetime_recurrences              28.6         28.6
//!   tentative_promoted                28.6         28.6
//!   binding_sequence_index           104.1        102.8
//!   binding_feature_atom_index       112.3         91.7
//!   binding_motif_index              164.4         47.9
//!   moment_history                    13.5          0.2
//!   TOTAL                            738.7        587.1
//!
//! So "per-fact cost does not rise" is a property the brain HAS, not a target
//! it is reaching for, and a test is the cheapest way to keep it. The defect it
//! exists to catch is the one that produced the original 2,471 MB scale-64
//! peak: `Pool::check_concept_emergence` charged ~1,071 permanent ledger
//! entries per question, so cost per fact climbed with the corpus. Anything of
//! that shape -- a key whose length grows with what is already stored, a
//! posting list that is rebuilt rather than appended, a cache with no bound --
//! shows up here as a rising B/fact long before it shows up as a megabyte.
//!
//! The assertions are paired against vacuity on purpose. A brain that stored
//! nothing, or a census that had stopped counting, would pass a
//! "does not grow" check trivially, so the test first proves the structures
//! are populated and growing in absolute terms.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// The two corpus sizes compared. Small enough to train in a test, 4x apart so
/// a per-fact cost that rises with the corpus has somewhere to show.
const SMALL_FACTS: usize = 64;
const LARGE_FACTS: usize = 256;

/// Headroom on the per-bucket comparison. Measured above, the worst bucket
/// across a 64x scale change moves +0.04% (`fingerprint_keys`, 287.2 -> 287.3,
/// which is hash-table bucket rounding rather than a real trend). A real
/// superlinear structure is not marginal: `binding_motif_index` costs 3.7x more
/// per ENTRY at scale 64 and `binding_feature_atom_index` 50x. 1.25 is
/// therefore far above the noise and far below any defect worth catching.
const PER_BUCKET_TOLERANCE: f64 = 1.25;

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

/// One brain holding `facts` distinct question/answer bindings. The question
/// text carries the fact index so every binding is a distinct moment, and the
/// answer varies so no two facts collapse onto one concept.
fn trained(facts: usize) -> Brain {
    let mut brain = subject();
    for i in 0..facts {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("r{i:04} object{i:04} material?").into_bytes()),
            (ANSWER_POOL, format!("material{:03}", i % 97).into_bytes()),
        ]);
    }
    brain
}

/// `(bucket name, bytes)` for every global index the census names, plus the
/// total it reports. Read through the public `global_index_sizes` so this test
/// measures the same numbers the scorecard prints rather than a private field.
fn buckets(brain: &Brain) -> (Vec<(String, u64)>, u64) {
    let census = brain.global_index_sizes();
    let total = census["total_bytes"]
        .as_u64()
        .expect("global_index_sizes reports total_bytes");
    let mut out = Vec::new();
    let bytes = census["bytes"]
        .as_object()
        .expect("global_index_sizes reports a bytes object")
        .clone();
    for (name, value) in bytes {
        out.push((name, value.as_u64().expect("bucket bytes are a number")));
    }
    out.sort();
    (out, total)
}

#[test]
fn accounted_bytes_per_fact_do_not_rise_with_the_corpus() {
    let small = trained(SMALL_FACTS);
    let large = trained(LARGE_FACTS);
    let (small_buckets, small_total) = buckets(&small);
    let (large_buckets, large_total) = buckets(&large);

    // Not vacuous: the census must be counting something, and the bigger
    // corpus must cost more in absolute terms. A brain that learned nothing,
    // or a census that returned zeros, would otherwise sail through.
    assert!(
        small_total > 0,
        "the census reports 0 total bytes for {SMALL_FACTS} facts -- nothing was counted, so a \
         'does not grow' assertion below would be vacuous"
    );
    assert!(
        large_total > small_total,
        "{LARGE_FACTS} facts cost {large_total} B and {SMALL_FACTS} facts cost {small_total} B -- \
         the larger corpus did not cost more, so either training did not happen or the census is \
         not measuring it"
    );

    let small_per_fact = small_total as f64 / SMALL_FACTS as f64;
    let large_per_fact = large_total as f64 / LARGE_FACTS as f64;
    println!(
        "TOTAL  {SMALL_FACTS} facts {small_total} B = {small_per_fact:.1} B/fact   \
         {LARGE_FACTS} facts {large_total} B = {large_per_fact:.1} B/fact   \
         ratio {:.3}",
        large_per_fact / small_per_fact
    );
    for ((name, s), (_, l)) in small_buckets.iter().zip(large_buckets.iter()) {
        if *l == 0 {
            continue;
        }
        let sp = *s as f64 / SMALL_FACTS as f64;
        let lp = *l as f64 / LARGE_FACTS as f64;
        println!("  {name:<28} {sp:9.1} -> {lp:9.1} B/fact   ratio {:.3}", lp / sp.max(1e-9));
    }

    // The ratchet itself. The total is held to no growth at all, because that
    // is the number the RAM promise is made of; individual buckets get the
    // measured headroom above.
    assert!(
        large_per_fact <= small_per_fact,
        "accounted bytes per fact ROSE with the corpus: {small_per_fact:.1} B/fact at \
         {SMALL_FACTS} facts -> {large_per_fact:.1} B/fact at {LARGE_FACTS} facts. Peak RAM now \
         grows faster than the corpus. The per-bucket lines printed above name which structure \
         did it; the defect class is a key or a posting list whose size depends on what is \
         already stored."
    );
    for ((name, s), (lname, l)) in small_buckets.iter().zip(large_buckets.iter()) {
        assert_eq!(name, lname, "the census changed its bucket set between two runs");
        if *l == 0 {
            continue;
        }
        let sp = (*s as f64 / SMALL_FACTS as f64).max(1.0);
        let lp = *l as f64 / LARGE_FACTS as f64;
        assert!(
            lp <= sp * PER_BUCKET_TOLERANCE,
            "{name} costs {lp:.1} B/fact at {LARGE_FACTS} facts against {sp:.1} B/fact at \
             {SMALL_FACTS} -- a factor of {:.2}, past the measured {PER_BUCKET_TOLERANCE} \
             headroom. This structure charges more per fact the more facts it holds, which is \
             what makes peak RAM track the corpus instead of the knowledge.",
            lp / sp
        );
    }
}
