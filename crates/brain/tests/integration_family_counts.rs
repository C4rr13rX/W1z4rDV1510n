//! The scorecard's scale-1 integration families, asked through the SAME call
//! the scorecard asks them through, as a 10-second guard instead of a 3.8-minute
//! gate.
//!
//! # Why this file exists
//!
//! A change to `derive_by_substitution_profiled` cost `on_material` 22/24 ->
//! 8/24 at scale 1 -- integration 74.07 -> 48.15, a 26-point drop -- and the
//! ONLY way to see it was a scorecard run. Measured, both numbers read off the
//! example's own JSON with `json.load`:
//!
//! ```text
//!   HEAD                           integr 74.07   on_material 22/24  ppa 8.62
//!   HEAD + a splice-scan shortcut  integr 48.15   on_material  8/24  ppa 7.50
//! ```
//!
//! Note the shape of the failure, because it is why a coverage number alone
//! cannot catch it: the probe count went DOWN. A shortcut that skips the scan
//! which finds the taught rewrite is CHEAPER and WRONGER at the same time, and
//! `wrong` stayed 0 throughout -- the fourteen lost answers all became silence.
//! So it passes PRIORITY ZERO, passes every RAM rule, passes every timing rule,
//! and shows up in exactly one place: the per-family hit count.
//!
//! # What the splice shortcut was, and why no cache can be it
//!
//! `derive_by_substitution_profiled` caches two things per question SHAPE
//! `(n, k)` -- the deletion that finds the taught sub-question, and the splice
//! points that reached the ceiling. The second is written from the ceiling arm
//! only, so hop 1 of an n-hop question learns nothing: its rewrite is the
//! intermediate question, which is untrained, which is exactly what makes the
//! probe an integration probe. Hop 1 therefore re-pays the full `k + 1` splice
//! scan on every repeat, and at scale 64 that is why `next_on_material` spends
//! `probes_per_attempt` 26.82 of a 32 budget and starves 356 of 512 attempts
//! while its two halves (`next_color` 1024/1024 at 4.10 probes, `on_material`
//! 1534/1536 at 4.07) both answer.
//!
//! Caching "this shape reached no ceiling, so replay the continuation and stop"
//! is unsound, and the 8/24 is the proof. `(n, k)` identifies a LENGTH and a
//! CUT, not a question: a trained 22-byte recall probe and `"r000 lamp on
//! material?"` share `(22, 12)` and have completely different rewrites. The
//! first one to reach no ceiling teaches the cache a negative that the second
//! one obeys -- so `on_material`'s hop 1 took the continuation walk and never
//! probed the `j` that produces the taught `"r000 desk material?"`. A property
//! of a QUESTION was cached as a property of a SHAPE.
//!
//! The 3-hop's cost is therefore not reducible by memoising the scan. Nor by
//! raising the budget: backlog `f1bd9c76` priced +32 probes at +4.1 % for
//! +25 % RAM and 7x wall. What is left is making the hop structurally cheaper
//! -- recursing on the sub-question instead of enumerating splices of it.
//!
//! # What is asserted
//!
//! The per-family counts at the committed baseline, as floors, with `wrong`
//! pinned at 0. Floors and not equalities: a pass that IMPROVES a family must
//! not have to edit this file, and PRIORITY ZERO says abstaining is always
//! allowed, so only invention and a fall in `correct` are failures.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    ANSWER_CHAIN_MAX_DEPTH, ANSWER_CHAIN_MAX_VISIT, ANSWER_FABRIC_CONFIDENCE_THRESHOLD,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

// The scorecard's scale-1 world, read off `examples/scorecard.rs` INCLUDING
// its `pick` salt and its brain and pool config, so a difference here is a
// difference there. The 17/31 multipliers are not decoration: that file records
// that a salt sharing a factor with the list lengths made one probe per room
// solvable by the prefix shortcut the world exists to defeat.
const ROOMS: usize = 8;
const EPOCHS: usize = 2;
const OBJECTS: &[&str] = &["bed", "chair", "mirror", "desk", "lamp", "door", "window", "paper"];
const COLORS: &[&str] = &["red", "blue", "green", "white", "black", "grey"];
const MATERIALS: &[&str] = &["oak", "steel", "glass", "cloth", "pine", "brass"];
const RESTS_ON: &[(&str, &str)] = &[("lamp", "desk"), ("paper", "desk"), ("mirror", "door")];
const BESIDE_TRAINED_EVERY: usize = 4;
const NEXT_COLOR_OBJECTS: usize = 2;

fn room(r: usize) -> String {
    format!("r{r:03}")
}
fn pick(list: &[&'static str], r: usize, salt: usize) -> String {
    list[(r * 31 + salt * 17) % list.len()].to_string()
}
fn color(r: usize, i: usize) -> String {
    pick(COLORS, r, i)
}
fn material(r: usize, i: usize) -> String {
    pick(MATERIALS, r, i + 1)
}
fn idx(obj: &str) -> usize {
    OBJECTS.iter().position(|o| *o == obj).expect("object is in OBJECTS")
}
fn decoy(r: usize, obj: &str, base: &str) -> String {
    let want = material(r, idx(base));
    OBJECTS
        .iter()
        .find(|d| **d != obj && **d != base && material(r, idx(d)) != want)
        .expect("6 materials over 8 objects always leave a differing decoy")
        .to_string()
}

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

fn trained_facts() -> Vec<(String, String)> {
    let mut facts = Vec::new();
    for r in 0..ROOMS {
        let rm = room(r);
        let next = room((r + 1) % ROOMS);
        for (i, obj) in OBJECTS.iter().enumerate() {
            facts.push((format!("{rm} {obj} color?"), color(r, i)));
            facts.push((format!("{rm} {obj} material?"), material(r, i)));
        }
        for (obj, base) in RESTS_ON {
            facts.push((format!("{rm} {obj} on?"), (*base).to_string()));
            facts.push((format!("{rm} {obj} near?"), decoy(r, obj, base)));
        }
        facts.push((format!("{rm} next?"), next.clone()));
        if r % BESIDE_TRAINED_EVERY == 0 {
            facts.push((format!("{rm} beside?"), next));
        }
    }
    facts
}

/// The four held-out families, in the scorecard's order.
fn families() -> Vec<(&'static str, Vec<(String, String)>)> {
    let mut on_material = Vec::new();
    let mut next_color = Vec::new();
    let mut next_on_material = Vec::new();
    let mut beside_next = Vec::new();
    for r in 0..ROOMS {
        let rm = room(r);
        let nr = (r + 1) % ROOMS;
        for (obj, base) in RESTS_ON {
            on_material
                .push((format!("{rm} {obj} on material?"), material(r, idx(base))));
        }
        for k in 0..NEXT_COLOR_OBJECTS {
            let i = (r + 3 * k) % OBJECTS.len();
            next_color.push((format!("{rm} next {} color?", OBJECTS[i]), color(nr, i)));
        }
        let (obj, base) = RESTS_ON[r % RESTS_ON.len()];
        next_on_material
            .push((format!("{rm} next {obj} on material?"), material(nr, idx(base))));
        if r % BESIDE_TRAINED_EVERY != 0 {
            beside_next.push((format!("{rm} beside?"), room(nr)));
        }
    }
    vec![
        ("on_material", on_material),
        ("next_color", next_color),
        ("next_on_material", next_on_material),
        ("beside_next", beside_next),
    ]
}

/// The scorecard's `recall`, call for call: the trained binding first,
/// `integrate` as the fallback. This is the route `crates/node/src/brain_api.rs`
/// answers a KNOWN question on, and asking a trained fact through `infer`
/// instead measured 0 of 186 -- the two paths are not interchangeable.
fn recall(brain: &mut Brain, query: &str) -> Option<String> {
    brain.observe_read_only(QUERY_POOL, query.as_bytes());
    let legacy = brain.integrate(QUERY_POOL, ANSWER_POOL);
    let answer = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .or(legacy.answer);
    let _ = brain.finish_read_only_inference();
    answer.filter(|a| !a.is_empty()).map(|a| String::from_utf8_lossy(&a).to_string())
}

/// The scorecard's `infer`, call for call: `observe_read_only`, then
/// `integrate_autonomous` at the shipped constants, then
/// `finish_read_only_inference`. Asking through `derive_by_substitution`
/// directly would measure a function the product does not call.
fn infer(brain: &mut Brain, query: &str) -> Option<String> {
    brain.observe_read_only(QUERY_POOL, query.as_bytes());
    let answer = brain
        .integrate_autonomous(
            QUERY_POOL,
            ANSWER_POOL,
            ANSWER_FABRIC_CONFIDENCE_THRESHOLD,
            ANSWER_CHAIN_MAX_DEPTH,
            ANSWER_CHAIN_MAX_VISIT,
        )
        .answer;
    let _ = brain.finish_read_only_inference();
    answer.filter(|a| !a.is_empty()).map(|a| String::from_utf8_lossy(&a).to_string())
}

/// `examples/scorecard.rs`'s own training order, copied exactly. Order is not
/// cosmetic here: the derivation's shape caches are warmed by whatever ran
/// before, so a different order is a different measurement.
fn shuffled(n: usize, epoch: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s: u64 = 0xC0FF_EECA_FEBA_BE ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s as usize) % (i + 1));
    }
    idx
}

/// `(correct, wrong, silent)` per family, in the scorecard's probe order --
/// order matters, because the derivation's shape caches are warmed by whatever
/// ran before.
fn measure() -> Vec<(&'static str, usize, usize, usize, usize)> {
    let mut brain = subject();
    let facts = trained_facts();
    for epoch in 0..EPOCHS {
        for i in shuffled(facts.len(), epoch) {
            let (q, a) = &facts[i];
            brain.pretrain_binding_episode(&[
                (QUERY_POOL, q.as_bytes().to_vec()),
                (ANSWER_POOL, a.as_bytes().to_vec()),
            ]);
            brain.eem_mut().induce_from_episode(q.as_bytes(), a.as_bytes());
        }
    }
    // Recall first, exactly as the scorecard does, and it is not decoration:
    // the recall pass warms the cut and splice caches that every integration
    // probe then reads, so a measurement that skips it measures a colder brain
    // than the product has.
    let mut recalled = 0;
    for (q, a) in &facts {
        if recall(&mut brain, q).as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    assert_eq!(recalled, facts.len(), "recall of trained material is ALWAYS 100 %");

    let mut out = Vec::new();
    for (name, probes) in families() {
        let (mut correct, mut wrong, mut silent) = (0, 0, 0);
        for (q, want) in &probes {
            match infer(&mut brain, q) {
                None => silent += 1,
                Some(a) if a == *want => correct += 1,
                Some(_) => wrong += 1,
            }
        }
        out.push((name, correct, wrong, silent, probes.len()));
    }
    out
}

/// The guard. Floors read off `docs/scorecard-baseline.json` at scale 1 with
/// `json.load`, confirmed by running the example itself at HEAD:
/// `integr 74.07`, `on_material 22/24`, `next_color 16/16`,
/// `next_on_material 2/8`, `beside_next 0/6`.
#[test]
fn scale_one_integration_families_hold_their_baseline() {
    // `beside_next` has no floor above 0: it is 0 at every scale and the
    // mechanism that would fix it is not in this file's subject.
    let floors = [("on_material", 22), ("next_color", 16), ("next_on_material", 2),
                  ("beside_next", 0)];
    let measured = measure();
    let mut total_correct = 0;
    let mut total_probes = 0;
    for (name, correct, wrong, silent, probes) in &measured {
        println!("  {name:<18} correct {correct:>3}/{probes:<3} WRONG {wrong}  silent {silent}");
        total_correct += correct;
        total_probes += probes;
    }
    println!(
        "integration {:.2} % over {total_probes} probes",
        100.0 * total_correct as f32 / total_probes as f32
    );
    for (name, _correct, wrong, _, probes) in &measured {
        // PRIORITY ZERO, and it is first because a family may always abstain.
        assert_eq!(*wrong, 0, "{name}: the derivation invented {wrong} answers of {probes}");
    }
    for (name, floor) in floors {
        let (_, correct, _, _, probes) = measured
            .iter()
            .find(|(n, _, _, _, _)| *n == name)
            .copied()
            .unwrap_or_else(|| panic!("{name} is a scorecard family"));
        assert!(
            correct >= floor,
            "{name} answered {correct} of {probes}; the committed scale-1 baseline is \
             {floor}. A FALL here with `wrong` still 0 and the probe count DOWN is the \
             signature of a splice-scan shortcut -- see this file's header."
        );
    }
}
