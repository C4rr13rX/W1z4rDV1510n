//! Integration tests for the two-pool associative recall path.
//!
//! Mirrors `scripts/integration_test.py` (which validated the architectural
//! change set in Python before the Rust port) — exercises end-to-end
//! lifecycle: train, recall fwd & rev, repeat queries (state stability),
//! incremental training, OOD near-neighbor recall, edge cases (empty / single
//! char input), mixed input/output lengths, identical-input collision, and
//! determinism across two identical runs.
//!
//! Pre-port (defective Rust): the 8-pair baseline scored ~0.04 / 0/8 exact.
//! Post-port (this test) target: 8/8 fwd & 8/8 rev exact recall on baseline.

use std::collections::HashMap;
use std::sync::Arc;

use w1z4rdv1510n::neuro::{AssocDirection, NeuroRuntime, NeuroRuntimeConfig};
use w1z4rdv1510n::schema::{EnvironmentSnapshot, Timestamp};
use w1z4rdv1510n::streaming::char_label;

fn empty_snapshot() -> EnvironmentSnapshot {
    EnvironmentSnapshot {
        timestamp:     Timestamp { unix: 0 },
        bounds:        HashMap::new(),
        symbols:       Vec::new(),
        metadata:      HashMap::new(),
        stack_history: Vec::new(),
    }
}

fn fresh_runtime() -> Arc<NeuroRuntime> {
    let snap = empty_snapshot();
    let cfg  = NeuroRuntimeConfig {
        enabled: true,
        min_activation: 0.05,
        ..Default::default()
    };
    Arc::new(NeuroRuntime::new(&snap, cfg))
}

fn atoms_of(s: &str) -> Vec<String> {
    s.chars().map(char_label).collect()
}

const PASSES: usize = 30;
const LR: f32 = 0.5;
const QUERY_HOPS: usize = 4;
const MIN_ACT: f32 = 0.05;

fn train_pairs(rt: &NeuroRuntime, pairs: &[(&str, &str)]) {
    for (q, a) in pairs {
        let in_atoms  = atoms_of(q);
        let out_atoms = atoms_of(a);
        for _ in 0..PASSES {
            rt.two_pool_train_pair(&in_atoms, &out_atoms, LR);
        }
    }
}

fn fwd(rt: &NeuroRuntime, q: &str) -> String {
    rt.two_pool_query(&atoms_of(q), AssocDirection::InToOut, QUERY_HOPS, MIN_ACT)
        .unwrap_or_default()
}

fn rev(rt: &NeuroRuntime, a: &str) -> String {
    rt.two_pool_query(&atoms_of(a), AssocDirection::OutToIn, QUERY_HOPS, MIN_ACT)
        .unwrap_or_default()
}

fn baseline_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("greet warmly", "hello there friend welcome aboard"),
        ("greet coldly", "acknowledged standard protocol greeting"),
        ("explain quantum physics",
         "quantum mechanics superposition entanglement uncertainty"),
        ("explain classical physics",
         "classical mechanics newton forces gravity motion"),
        ("describe weather sunny",
         "today bright skies warm temperatures gentle breeze"),
        ("describe spring season",
         "currently blooming flowers warming temperatures growth"),
        ("describe ocean waves",
         "deep blue rolling waves crashing against shores"),
        ("describe forest morning",
         "ancient towering pines dappled sunlight quiet birds"),
    ]
}

// ── IT1: 8/8 fwd & 8/8 rev exact recall on the baseline corpus ───────────────

#[test]
fn it1_recall_baseline_8_pairs_fwd_and_rev() {
    let rt = fresh_runtime();
    let pairs = baseline_pairs();
    train_pairs(&rt, &pairs);

    let mut fwd_ok = 0; let mut rev_ok = 0;
    for (q, a) in &pairs {
        if fwd(&rt, q) == *a { fwd_ok += 1; }
        if rev(&rt, a) == *q { rev_ok += 1; }
    }
    assert_eq!(fwd_ok, pairs.len(),
        "forward recall: {}/{}", fwd_ok, pairs.len());
    assert_eq!(rev_ok, pairs.len(),
        "reverse recall: {}/{}", rev_ok, pairs.len());
}

// ── IT2: repeated queries on the same model must not drift ───────────────────

#[test]
fn it2_repeated_queries_stable() {
    let rt = fresh_runtime();
    let pairs = baseline_pairs();
    train_pairs(&rt, &pairs);

    for (q, _) in &pairs {
        let r1 = fwd(&rt, q);
        let r2 = fwd(&rt, q);
        let r3 = fwd(&rt, q);
        assert_eq!(r1, r2, "drift on second query for {q:?}");
        assert_eq!(r2, r3, "drift on third query for {q:?}");
    }
}

// ── IT3: incremental training preserves prior recall ─────────────────────────

#[test]
fn it3_incremental_training_preserves_prior() {
    let rt = fresh_runtime();
    let pairs = baseline_pairs();
    train_pairs(&rt, &pairs);

    // Verify initial recall before incremental train.
    for (q, a) in &pairs {
        assert_eq!(fwd(&rt, q), *a,
            "pre-incremental fwd recall failed for {q:?}");
    }

    // Add one new pair.
    let new = ("describe new topic added later",
               "novel content emerging from incremental training");
    train_pairs(&rt, &[new]);

    // All 9 must still recall correctly.
    let mut all = pairs.clone();
    all.push(new);
    for (q, a) in &all {
        assert_eq!(fwd(&rt, q), *a,
            "post-incremental fwd recall failed for {q:?}");
        assert_eq!(rev(&rt, a), *q,
            "post-incremental rev recall failed for {a:?}");
    }
}

// ── IT4: untrained near-neighbor inputs return the closest trained answer ───

#[test]
fn it4_ood_near_neighbors_return_closest() {
    let rt = fresh_runtime();
    let pairs = baseline_pairs();
    train_pairs(&rt, &pairs);

    let near_cases: &[(&str, &str)] = &[
        ("greet warmly!",            "hello there friend welcome aboard"),
        ("greet warmly ",            "hello there friend welcome aboard"),
        ("explain quantum physic",   "quantum mechanics superposition entanglement uncertainty"),
        ("describe ocean wave",      "deep blue rolling waves crashing against shores"),
        ("describe spring",          "currently blooming flowers warming temperatures growth"),
    ];
    for (q, expected) in near_cases {
        let pred = fwd(&rt, q);
        assert_eq!(&pred, expected,
            "near-neighbor {q:?} expected {expected:?}, got {pred:?}");
    }
}

// ── IT5: empty / single-char inputs do not crash ─────────────────────────────

#[test]
fn it5_edge_inputs_no_crash() {
    let rt = fresh_runtime();
    train_pairs(&rt, &baseline_pairs());

    // Each of these should return *something* (possibly empty string) with
    // no panic.  We only require the call not to panic.
    let _ = fwd(&rt, "");
    let _ = fwd(&rt, "a");
    let _ = fwd(&rt, "z");
    let _ = rev(&rt, "");
    let _ = rev(&rt, " ");
}

// ── IT6: mixed input/output lengths (short→long, long→short) ─────────────────

#[test]
fn it6_mixed_lengths() {
    let rt = fresh_runtime();
    let pairs: &[(&str, &str)] = &[
        ("hi",
         "this is a much longer answer about greetings and salutations"),
        ("a long detailed query about the weather today",
         "warm"),
        ("short",  "tiny"),
        ("medium length question",
         "medium length answer of similar size"),
    ];
    train_pairs(&rt, pairs);

    for (q, a) in pairs {
        assert_eq!(fwd(&rt, q), *a,
            "mixed-length fwd failed for {q:?}");
        assert_eq!(rev(&rt, a), *q,
            "mixed-length rev failed for {a:?}");
    }
}

// ── IT7: determinism across two identical runs ───────────────────────────────

#[test]
fn it7_determinism() {
    let pairs = baseline_pairs();

    let rt_a = fresh_runtime();
    train_pairs(&rt_a, &pairs);
    let results_a: Vec<String> = pairs.iter().map(|(q, _)| fwd(&rt_a, q)).collect();

    let rt_b = fresh_runtime();
    train_pairs(&rt_b, &pairs);
    let results_b: Vec<String> = pairs.iter().map(|(q, _)| fwd(&rt_b, q)).collect();

    assert_eq!(results_a, results_b,
        "two identical training+query runs disagreed");
}
