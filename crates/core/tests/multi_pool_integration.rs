//! Integration tests for the N-pool associative fabric.
//!
//! Validates that:
//!   - More than two pools can be registered.
//!   - Training pool A ↔ pool B does not break pool A ↔ pool C.
//!   - Sending input to one pool causes EVERY other connected pool to
//!     produce its own decoded prediction.
//!   - Cross-modal training works (train an input pool against multiple
//!     target pools simultaneously).

use std::collections::HashMap;
use std::sync::Arc;

use w1z4rdv1510n::neuro::{NeuroRuntime, NeuroRuntimeConfig};
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

fn train(
    rt: &NeuroRuntime,
    src: &str,
    src_atoms: &[String],
    tgt: &str,
    tgt_atoms: &[String],
) {
    for _ in 0..PASSES {
        rt.multi_pool_train_pair(src, src_atoms, tgt, tgt_atoms, LR);
    }
}

#[test]
fn input_to_one_pool_fires_all_other_pools() {
    let rt = fresh_runtime();

    // Three pools: a "lang" question pool, a "ans" answer pool, and an
    // "emo" emotion pool.  Pair every (lang, ans) and (lang, emo) so a
    // language input fires both downstream pools simultaneously.
    rt.multi_pool_register("lang");
    rt.multi_pool_register("ans");
    rt.multi_pool_register("emo");

    let triples: &[(&str, &str, &str)] = &[
        ("hello",        "hi how are you",          "warm"),
        ("goodbye",      "see you later friend",    "sad"),
        ("thanks",       "you are welcome always",  "grateful"),
        ("are you ok",   "yes i am operational",    "calm"),
    ];

    for (q, a, e) in triples {
        train(&rt, "lang", &atoms_of(q), "ans", &atoms_of(a));
        train(&rt, "lang", &atoms_of(q), "emo", &atoms_of(e));
    }

    // multi_pool_query("lang", q) should fire BOTH "ans" AND "emo".
    for (q, a, e) in triples {
        let preds = rt.multi_pool_query(
            "lang", &atoms_of(q), 4, 0.05,
        );
        assert_eq!(preds.get("ans").map(String::as_str), Some(*a),
            "ans recall failed for {q:?}");
        assert_eq!(preds.get("emo").map(String::as_str), Some(*e),
            "emo recall failed for {q:?}");
    }
}

#[test]
fn fanout_train_helper_works() {
    let rt = fresh_runtime();
    rt.multi_pool_register("text");
    rt.multi_pool_register("color");
    rt.multi_pool_register("category");

    let pairs: &[(&str, &str, &str)] = &[
        ("apple",   "red",    "fruit"),
        ("banana",  "yellow", "fruit"),
        ("car",     "blue",   "vehicle"),
        ("airplane","silver", "vehicle"),
    ];

    for (src, color, cat) in pairs {
        let targets = vec![
            ("color".to_string(),    atoms_of(color)),
            ("category".to_string(), atoms_of(cat)),
        ];
        for _ in 0..PASSES {
            rt.multi_pool_train_fanout(
                "text", &atoms_of(src), &targets, LR,
            );
        }
    }

    for (src, color, cat) in pairs {
        let preds = rt.multi_pool_query("text", &atoms_of(src), 4, 0.05);
        assert_eq!(preds.get("color").map(String::as_str), Some(*color),
            "color fanout failed for {src:?}");
        assert_eq!(preds.get("category").map(String::as_str), Some(*cat),
            "category fanout failed for {src:?}");
    }
}

#[test]
fn reverse_query_works_for_any_pool_pair() {
    let rt = fresh_runtime();
    rt.multi_pool_register("a");
    rt.multi_pool_register("b");
    rt.multi_pool_register("c");

    let triples: &[(&str, &str, &str)] = &[
        ("xx aa", "yy aa", "zz aa"),
        ("xx bb", "yy bb", "zz bb"),
        ("xx cc", "yy cc", "zz cc"),
    ];
    for (x, y, z) in triples {
        train(&rt, "a", &atoms_of(x), "b", &atoms_of(y));
        train(&rt, "b", &atoms_of(y), "c", &atoms_of(z));
    }

    // Query into b should fire both a and c.
    for (x, y, z) in triples {
        let preds = rt.multi_pool_query("b", &atoms_of(y), 4, 0.05);
        assert_eq!(preds.get("a").map(String::as_str), Some(*x),
            "rev recall a from b failed for {y:?}");
        assert_eq!(preds.get("c").map(String::as_str), Some(*z),
            "fwd recall c from b failed for {y:?}");
    }
}
