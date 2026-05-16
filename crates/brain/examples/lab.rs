//! Empirical validation harness for the brain substrate.
//!
//! Five experiments probe the architecture under load.  The point is
//! to HONESTLY characterize where the substrate works today and
//! where its current edges are for the senior-software-engineer goal:
//!
//! - A: memorization at scale     — recall vs. N (small → medium)
//! - B: distractor robustness     — recall after interference
//! - C: concept emergence on code — what patterns become concepts?
//! - D: performance scaling       — observe/integrate latency, footprint
//! - E: persistence round-trip    — does checkpoint/restore preserve behavior?
//!
//! Run: `cargo run --release --example lab -p w1z4rd-brain`

use std::time::Instant;
use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

// -----------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------

const PROMPT_POOL:     u32 = 1;
const COMPLETION_POOL: u32 = 2;

fn build_brain(window: usize, concept_threshold: u32) -> Brain {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);

    let mut pa = PoolConfig::defaults("prompt", PROMPT_POOL);
    pa.recent_atoms_window         = window;
    pa.concept_emergence_threshold = concept_threshold;
    pa.max_concept_member_count    = 64;
    pa.decay_rate                  = 0.0001;
    pa.prune_floor                 = 0.005;

    let mut pb = PoolConfig::defaults("completion", COMPLETION_POOL);
    pb.recent_atoms_window         = window;
    pb.concept_emergence_threshold = concept_threshold;
    pb.max_concept_member_count    = 64;
    pb.decay_rate                  = 0.0001;
    pb.prune_floor                 = 0.005;

    brain.create_pool(pa, Box::new(BytePassthroughEncoding { prefix: "p" }) as Box<dyn AtomEncoding>);
    brain.create_pool(pb, Box::new(BytePassthroughEncoding { prefix: "c" }) as Box<dyn AtomEncoding>);
    brain
}

/// Deterministic per-epoch Fisher-Yates permutation using xorshift64.
/// Each epoch trains in a different order so cross-pair boundary
/// atom-sequences don't accumulate to threshold — only the intra-pair
/// completion sequences (which repeat verbatim each epoch) cross
/// threshold and become concepts.
fn permutation_for_epoch(n: usize, epoch: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state: u64 = 0xC0FFEE_CAFE_BABE
        ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if state == 0 { state = 0xDEAD_BEEF_C0FFEE; }
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

fn train_pair(brain: &mut Brain, prompt: &[u8], completion: &[u8]) {
    brain.observe(PROMPT_POOL, prompt);
    brain.observe(COMPLETION_POOL, completion);
    brain.advance_tick();
}

fn train_corpus(brain: &mut Brain, pairs: &[(Vec<u8>, Vec<u8>)], epochs: usize) {
    for ep in 0..epochs {
        let perm = permutation_for_epoch(pairs.len(), ep);
        for i in perm {
            let (p, c) = &pairs[i];
            train_pair(brain, p, c);
        }
    }
}

fn recall(brain: &mut Brain, prompt: &[u8]) -> Option<Vec<u8>> {
    brain.observe(PROMPT_POOL, prompt);
    let ans = brain.integrate(PROMPT_POOL, COMPLETION_POOL);
    ans.answer
}

fn synthetic_corpus(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let templates: &[(&str, &str)] = &[
        ("fn{}(",           "x+y}"),
        ("let{}=",          "{};"),
        ("if{}>0",          "{return;}"),
        ("for{}in",         "vec.iter()"),
        ("while{}<n",       "{}++;"),
        ("struct{}{",       "field:i32,}"),
        ("impl{}for",       "Trait{}"),
        ("match{}{",        "_=>{}}"),
        ("use{}::",         "module;"),
        ("pub fn{}",        "Result<()>"),
    ];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (pt, ct) = templates[i % templates.len()];
        let tag = format!("{:03}", i);
        out.push((
            pt.replace("{}", &tag).into_bytes(),
            ct.replace("{}", &tag).into_bytes(),
        ));
    }
    out
}

fn pct(num: usize, denom: usize) -> f32 {
    if denom == 0 { 0.0 } else { 100.0 * num as f32 / denom as f32 }
}

fn fmt_bytes(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

// -----------------------------------------------------------------
// Experiment A — Memorization at Scale
// -----------------------------------------------------------------

fn experiment_a() {
    println!("--- Experiment A: Memorization at Scale ---");
    println!("Cross-pool training of (prompt, completion) pairs.  5 epochs with");
    println!("per-epoch shuffle so cross-pair boundary patterns don't accumulate.");
    println!("Recall = observe prompt, integrate, compare answer to completion.\n");
    println!("    N    train_ms   exact%   contains%   neurons   terminals");
    println!("  ----  ---------  -------  ----------  --------  ----------");
    let sizes = [1_usize, 5, 25, 100, 500];
    for &n in &sizes {
        let pairs = synthetic_corpus(n);
        let mut brain = build_brain(/*window*/ 2048, /*threshold*/ 2);
        let t0 = Instant::now();
        train_corpus(&mut brain, &pairs, 5);
        let train_ms = t0.elapsed().as_millis();

        let mut exact = 0usize;
        let mut contains = 0usize;
        let mut samples = Vec::new();
        for (i, (p, c)) in pairs.iter().enumerate() {
            let answer = recall(&mut brain, p).unwrap_or_default();
            if &answer == c { exact += 1; }
            if !c.is_empty() && answer.windows(c.len()).any(|w| w == c.as_slice()) {
                contains += 1;
            }
            if i < 3 && n <= 25 { samples.push((p.clone(), c.clone(), answer)); }
        }
        let s = brain.stats();
        println!("  {:>4}  {:>9}  {:>6.1}%  {:>9.1}%  {:>8}  {:>10}",
                 n, train_ms, pct(exact, n), pct(contains, n),
                 s.total_neurons, s.total_terminals);
        for (p, c, a) in &samples {
            println!("        prompt={:?}  target={:?}  answer={:?}",
                fmt_bytes(p), fmt_bytes(c), fmt_bytes(a));
        }
    }
    println!();
}

// -----------------------------------------------------------------
// Experiment B — Distractor Robustness
// -----------------------------------------------------------------

fn experiment_b() {
    println!("--- Experiment B: Distractor Robustness ---");
    println!("Train one target pair, then K distractor pairs, then test target recall.");
    println!("Honest interference test: does adding noise wipe out the trained signal?\n");

    let target_p: &[u8] = b"signature_target_AAA";
    let target_c: &[u8] = b"unique_response_ZZZ";
    let distractors_max = 200;
    let test_points = [0usize, 5, 20, 50, 100, 200];

    println!("   distractors   exact   contains   answer_bytes");
    println!("   -----------  ------  ---------  -------------");
    for &k in &test_points {
        if k > distractors_max { continue; }
        let mut brain = build_brain(/*window*/ 4096, /*threshold*/ 2);
        // Train target k+1 times so it has solid grounding even under load.
        for _ in 0..5 { train_pair(&mut brain, target_p, target_c); }
        // Train k distractors (each with a single rep so target stays dominant
        // by repetition count — this measures interference, not weight contest).
        let distractors = synthetic_corpus(k);
        for (p, c) in &distractors {
            train_pair(&mut brain, p, c);
        }
        // Re-train target a few more times to refresh recent-atoms window.
        for _ in 0..3 { train_pair(&mut brain, target_p, target_c); }

        let ans = recall(&mut brain, target_p).unwrap_or_default();
        let exact = &ans == target_c;
        let contains = !target_c.is_empty()
            && ans.windows(target_c.len()).any(|w| w == target_c);
        println!("   {:>11}  {:>5}  {:>8}   {:?}",
                 k,
                 if exact { "YES" } else { "no" },
                 if contains { "YES" } else { "no" },
                 fmt_bytes(&ans));
    }
    println!();
}

// -----------------------------------------------------------------
// Experiment C — Concept Emergence on Code-Like Text
// -----------------------------------------------------------------

fn experiment_c() {
    println!("--- Experiment C: Concept Emergence on Code-Like Text ---");
    println!("Train a single pool on a stream of Rust-like tokens; inspect");
    println!("which atom-sequences became concept neurons.\n");

    let code_snippets: &[&str] = &[
        "fn main() { let x = 42; let y = 17; }",
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "let x = 1; let y = 2; let z = x + y;",
        "if x > 0 { return; } else { panic!(); }",
        "for i in 0..10 { println!(\"{}\", i); }",
        "let result = Vec::new(); let mut v = result;",
        "fn main() { let x = 1; }",
        "fn main() { let x = 2; }",
        "let a = 1; let b = 2; let c = a + b;",
        "if x > 0 { x } else { -x }",
    ];

    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 99; // single pool — no binding interesting
    let mut brain = Brain::new(cfg);
    let mut pc = PoolConfig::defaults("code", 1);
    pc.recent_atoms_window         = 4096;
    pc.concept_emergence_threshold = 3;
    pc.max_concept_member_count    = 12;
    pc.decay_rate                  = 0.00005;
    pc.prune_floor                 = 0.002;
    let pool = brain.create_pool(pc,
        Box::new(BytePassthroughEncoding { prefix: "c" }) as Box<dyn AtomEncoding>);

    // Train: observe each snippet a few times (mix repetition so emergent
    // concepts have a chance to cross the threshold).
    for _ in 0..5 {
        for s in code_snippets {
            brain.observe(pool, s.as_bytes());
            brain.advance_tick();
        }
    }

    // Inspect: enumerate concept neurons, decode their member labels back
    // to bytes via base64-url-safe (the BytePassthroughEncoding format).
    use base64::Engine;
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let pool_arc = brain.fabric().pool(pool).unwrap();
    let pool_ref = pool_arc.read();

    let mut concepts: Vec<(String, usize, u64)> = Vec::new();
    for n in pool_ref.iter_neurons() {
        if n.is_atom() { continue; }
        let mut decoded = Vec::new();
        for m in &n.members {
            if let Some(mn) = pool_ref.get(m.neuron) {
                if let Some(payload) = mn.label.strip_prefix("c:") {
                    if let Ok(bs) = engine.decode(payload) {
                        decoded.extend(bs);
                    }
                }
            }
        }
        concepts.push((String::from_utf8_lossy(&decoded).into_owned(),
                       n.members.len(),
                       n.use_count));
    }
    concepts.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));

    let total = concepts.len();
    println!("Emerged {} concept neurons.  Top 25 by use_count:\n", total);
    println!("    use_count  members   text");
    println!("    ---------  -------   ----");
    for (text, members, uc) in concepts.iter().take(25) {
        println!("    {:>9}  {:>7}   {:?}", uc, members, text);
    }
    println!();
}

// -----------------------------------------------------------------
// Experiment D — Performance Scaling
// -----------------------------------------------------------------

fn experiment_d() {
    println!("--- Experiment D: Performance Scaling ---");
    println!("Observe/integrate latency vs total training observations.\n");
    println!("       N    observe_us/op  integrate_us/op   neurons   terminals");
    println!("    ----   -------------  ---------------  --------  ----------");

    for &n in &[100_usize, 500, 2000, 5000] {
        let pairs = synthetic_corpus(n.min(500));
        let mut brain = build_brain(/*window*/ 4096, /*threshold*/ 3);

        // Run n training observations (cycle through pairs).
        let t0 = Instant::now();
        let mut idx = 0;
        for _ in 0..n {
            let (p, c) = &pairs[idx % pairs.len()];
            train_pair(&mut brain, p, c);
            idx += 1;
        }
        let observe_us_per_op = t0.elapsed().as_micros() as f32 / n as f32;

        let t1 = Instant::now();
        let trials = 100;
        for i in 0..trials {
            let (p, _) = &pairs[i % pairs.len()];
            let _ = recall(&mut brain, p);
        }
        let integrate_us_per_op = t1.elapsed().as_micros() as f32 / trials as f32;

        let s = brain.stats();
        println!("    {:>4}   {:>13.1}  {:>15.1}  {:>8}  {:>10}",
                 n, observe_us_per_op, integrate_us_per_op,
                 s.total_neurons, s.total_terminals);
    }
    println!();
}

// -----------------------------------------------------------------
// Experiment E — Persistence Round-Trip
// -----------------------------------------------------------------

fn experiment_e() {
    println!("--- Experiment E: Persistence Round-Trip ---");
    println!("Train, checkpoint, restore, re-run all recall queries; agree?\n");

    let pairs = synthetic_corpus(50);
    let mut brain = build_brain(/*window*/ 4096, /*threshold*/ 2);
    train_corpus(&mut brain, &pairs, 5);

    // Baseline recall set.
    let mut baseline = Vec::with_capacity(pairs.len());
    for (p, _) in &pairs {
        baseline.push(recall(&mut brain, p).unwrap_or_default());
    }

    let path = std::env::temp_dir().join("w1z4rd_brain_lab_checkpoint.bin");
    brain.checkpoint(&path).expect("checkpoint failed");

    let mut encodings: std::collections::HashMap<u32, Box<dyn AtomEncoding>> =
        std::collections::HashMap::new();
    encodings.insert(0, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    encodings.insert(PROMPT_POOL, Box::new(BytePassthroughEncoding { prefix: "p" }));
    encodings.insert(COMPLETION_POOL, Box::new(BytePassthroughEncoding { prefix: "c" }));
    let (mut restored, missing) = Brain::restore(&path, encodings).expect("restore failed");
    assert!(missing.is_empty(), "missing encodings: {:?}", missing);

    let mut matches = 0usize;
    let mut total = 0usize;
    for (i, (p, _)) in pairs.iter().enumerate() {
        let after = recall(&mut restored, p).unwrap_or_default();
        if after == baseline[i] { matches += 1; }
        total += 1;
    }
    let _ = std::fs::remove_file(&path);

    println!("  Recall-after-restore agreement: {}/{} ({:.0}%)",
             matches, total, pct(matches, total));
    println!();
}

// -----------------------------------------------------------------
// Experiment F — Clean Single-Atom Pairs (substrate granularity test)
// -----------------------------------------------------------------
//
// Each prompt is one byte; each completion is its uppercase letter
// doubled (e.g. 'A' → "AA").  No within-pair repetition complexity,
// no cross-pair shared atoms.  This tests whether the substrate
// retrieves precisely under conditions favoring it.

fn experiment_f() {
    println!("--- Experiment F: Clean Single-Atom Pairs ---");
    println!("Each pair: byte X → 'XX'.  Tests substrate's baseline precision.\n");

    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (b'A'..=b'Z')
        .map(|c| (vec![c], vec![c, c]))
        .collect();
    let n = pairs.len();

    let mut brain = build_brain(/*window*/ 1024, /*threshold*/ 3);
    train_corpus(&mut brain, &pairs, 6);

    let mut exact = 0usize;
    let mut contains = 0usize;
    let mut samples = Vec::new();
    for (i, (p, c)) in pairs.iter().enumerate() {
        let answer = recall(&mut brain, p).unwrap_or_default();
        if &answer == c { exact += 1; }
        if !c.is_empty() && answer.windows(c.len()).any(|w| w == c.as_slice()) {
            contains += 1;
        }
        if i < 5 {
            samples.push((p.clone(), c.clone(), answer));
        }
    }
    let s = brain.stats();
    println!("  N={}  exact_recall={}/{} ({:.1}%)  contains_recall={}/{} ({:.1}%)",
        n, exact, n, pct(exact, n), contains, n, pct(contains, n));
    println!("  neurons={}  terminals={}  total_binding={}",
        s.total_neurons, s.total_terminals, s.total_binding);
    for (p, c, a) in &samples {
        println!("    prompt={:?}  target={:?}  answer={:?}",
            fmt_bytes(p), fmt_bytes(c), fmt_bytes(a));
    }
    println!();
}

// -----------------------------------------------------------------
// Experiment G — Clean unique-pair scaling
// -----------------------------------------------------------------
//
// 4-byte prompts mapping to 4-byte completions with every pair
// genuinely unique (each byte position varies).  Reveals how high
// the substrate's exact-recall ceiling goes when training data
// avoids shared-token confusion.

fn experiment_g() {
    println!("--- Experiment G: Clean Unique-Pair Scaling ---");
    println!("Pairs: prompt='P'+3 unique digits → completion='C'+SAME 3 digits.");
    println!("All pairs distinct; same 1-1 mapping the brain must memorize.\n");
    println!("     N   epochs   exact%   contains%   neurons   terminals");
    println!("  ----  -------  -------  ----------  --------  ----------");
    for &n in &[10_usize, 26, 50, 100, 250, 500] {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .map(|i| {
                let p = format!("P{:03}", i).into_bytes();
                let c = format!("C{:03}", i).into_bytes();
                (p, c)
            })
            .collect();
        let mut brain = build_brain(/*window*/ 4096, /*threshold*/ 3);
        let epochs = 6;
        train_corpus(&mut brain, &pairs, epochs);

        let mut exact = 0usize;
        let mut contains = 0usize;
        for (p, c) in &pairs {
            let answer = recall(&mut brain, p).unwrap_or_default();
            if &answer == c { exact += 1; }
            if !c.is_empty() && answer.windows(c.len()).any(|w| w == c.as_slice()) {
                contains += 1;
            }
        }
        let s = brain.stats();
        println!("  {:>4}  {:>7}  {:>6.1}%  {:>9.1}%  {:>8}  {:>10}",
            n, epochs, pct(exact, n), pct(contains, n),
            s.total_neurons, s.total_terminals);
    }
    println!();
}

fn main() {
    println!("=== W1z4rD Brain — Empirical Validation Harness ===\n");
    experiment_a();
    experiment_b();
    experiment_c();
    experiment_d();
    experiment_e();
    experiment_f();
    experiment_g();
    println!("=== Done ===");
}
