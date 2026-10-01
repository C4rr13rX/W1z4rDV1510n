//! Can a 3-hop derivation finish inside the budget the product actually ships?
//!
//! `next_on_material` is the scorecard's 3-hop family and it scored 0 of 8 at
//! scale 1 and 22 of 512 at scale 64 — not because the chain is wrong, but
//! because it does not FIT. Each hop pays a deletion search to locate the
//! taught sub-question, and that search costs `~2k` questions for a cut at
//! `k`: `"r000 next lamp on material?"` cuts at 9, then `"r001 lamp on
//! material?"` cuts at 12, which is 42 questions of a 32-question budget spent
//! before a single rewrite is tried.
//!
//! The world here is the scorecard's SHAPE — three trained relations, every
//! question terminated with `?` so the taught sub-question is a deletion and
//! not a prefix — and the question asked is the one the scorecard asks.
//!
//! The before/after is taken on ONE brain in ONE run, because the cut cache
//! starts empty: the first question pays the full scan and is the "before",
//! every later question pays the remembered cut and is the "after". Nothing is
//! configured differently between them, so the comparison cannot be an artifact
//! of two differently-built subjects.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    DEFAULT_DERIVATION_PROBE_BUDGET, DERIVATION_CUT_HINTS,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const ROOMS: u32 = 16;

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

fn teach(brain: &mut Brain, question: &str, answer: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, question.as_bytes().to_vec()),
        (ANSWER_POOL, answer.as_bytes().to_vec()),
    ]);
}

fn recall(brain: &mut Brain, question: &str) -> Option<Vec<u8>> {
    brain.observe_read_only(QUERY_POOL, question.as_bytes());
    brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
}

/// `rNNN next?` → the next room, `rNNN lamp on?` → `desk`,
/// `rNNN desk material?` → that room's material. Three relations, so a
/// mechanism that chains only one of them derives nothing.
fn teach_world(brain: &mut Brain) -> Vec<(String, String)> {
    let material = |r: u32| format!("m{:03}", r % 7);
    let mut expected = Vec::new();
    for r in 0..ROOMS {
        let room = format!("r{r:03}");
        let next = format!("r{:03}", (r + 1) % ROOMS);
        teach(brain, &format!("{room} next?"), &next);
        teach(brain, &format!("{room} lamp on?"), "desk");
        teach(brain, &format!("{room} desk material?"), &material(r));
    }
    for r in 0..ROOMS {
        let nr = (r + 1) % ROOMS;
        expected.push((format!("r{r:03} next lamp on material?"), material(nr)));
    }
    expected
}

#[test]
fn a_three_hop_derivation_fits_the_shipped_probe_budget() {
    let mut brain = subject();
    let asked = teach_world(&mut brain);

    // Recall first. Every number below is unreadable if the taught half
    // regressed, and the derivation rewrites the firing state dozens of times.
    let mut recalled = 0u32;
    for r in 0..ROOMS {
        let room = format!("r{r:03}");
        for (q, a) in [
            (format!("{room} next?"), format!("r{:03}", (r + 1) % ROOMS)),
            (format!("{room} lamp on?"), "desk".to_string()),
            (format!("{room} desk material?"), format!("m{:03}", r % 7)),
        ] {
            if recall(&mut brain, &q).as_deref() == Some(a.as_bytes()) {
                recalled += 1;
            }
        }
    }
    assert_eq!(recalled, ROOMS * 3, "the taught world is not recalled, so nothing below reads");

    // The uncapped cost first, so "fits the budget" is measured against a
    // known total rather than against a number the cap itself produced. The
    // cache is cold on the first question and warm on every later one.
    let budget = DEFAULT_DERIVATION_PROBE_BUDGET;
    let mut cost: Vec<usize> = Vec::new();
    let mut uncapped_right = 0u32;
    for (q, want) in &asked {
        let (answer, probes) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 4096);
        cost.push(probes);
        if answer.as_deref() == Some(want.as_bytes()) {
            uncapped_right += 1;
        } else {
            eprintln!(
                "  miss {q:?} want {want:?} got {:?} in {probes}",
                answer.as_deref().map(String::from_utf8_lossy)
            );
        }
    }
    let cold = cost[0];
    // A second pass over the same family is the WARM measurement, because the
    // cache is keyed on the question's length and a derivation rewrites the
    // question as it goes — so the first pass is still meeting new lengths
    // partway through, and one of its questions pays a full scan at a length
    // nothing had asked yet. A scale-64 run is in this state after its first
    // few questions, not after its first.
    let mut warm: Vec<usize> = Vec::new();
    for (q, _) in &asked {
        let (_, probes) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 4096);
        warm.push(probes);
    }
    let warm_mean = warm.iter().sum::<usize>() as f64 / warm.len() as f64;
    let warm_max = warm.iter().max().copied().unwrap_or(0);
    eprintln!(
        "3-hop, uncapped: right {uncapped_right} of {}; questions asked -- \
         cold (cache empty) {cold}, warm mean {warm_mean:.0} max {warm_max}",
        asked.len()
    );
    eprintln!("cut hints learned: {:?}", brain.derivation_cut_hints());

    assert!(
        uncapped_right > 0,
        "the 3-hop chain derives nothing even uncapped, so no cost number below means anything"
    );
    assert!(
        brain.derivation_cut_hints().len() <= DERIVATION_CUT_HINTS,
        "the cut cache is unbounded: {} entries",
        brain.derivation_cut_hints().len()
    );
    assert!(
        warm_max <= budget,
        "a warm 3-hop derivation costs up to {warm_max} questions against a shipped \
         budget of {budget}, so the family cannot finish in the product"
    );
    assert!(
        cold > warm_max,
        "the cache did not reduce anything: cold {cold}, warm max {warm_max}"
    );

    // And now the thing the scorecard actually runs: the SHIPPED budget, on a
    // brain whose cache is already warm the way a scale-64 run's is after its
    // first question.
    let mut capped_right = 0u32;
    let mut capped_wrong = 0u32;
    for (q, want) in &asked {
        let answer =
            brain.derive_by_substitution(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, budget);
        match answer.as_deref() {
            Some(a) if a == want.as_bytes() => capped_right += 1,
            Some(_) => capped_wrong += 1,
            None => {}
        }
    }
    eprintln!(
        "3-hop at the shipped budget of {budget}: right {capped_right}, wrong {capped_wrong}, \
         of {}",
        asked.len()
    );
    // THE CONTRACT: the cap costs nothing. Whatever the chain derives with an
    // unlimited budget, it derives with the shipped one — which is what "cost
    // is no longer the blocker" means, and it is a claim that survives the
    // chain getting more accurate later.
    //
    // It is deliberately NOT `== asked.len()`. The chain is right 3 of 16 here
    // and the 13 misses all answer `m000`, whatever the room asked about, for a
    // reason that has nothing to do with cost: at the splice point `j = 2` the
    // rewrite `"r0desk material?"` carries the same SET of byte atoms as the
    // taught `"r000 desk material?"`, so it scores a perfect 1.0, breaks the
    // splice search, and wins before `j = 5` can build the correct
    // `"r001 desk material?"`. Garbage text that set-matches a taught question
    // is accepted as that question. That is backlog item [0935d42d] — "the
    // derivation invents four wrong answers for every right one" — and it is
    // measured here rather than described, because a repair for it must turn
    // this number up without any of the three cost assertions above moving.
    assert_eq!(
        capped_right, uncapped_right,
        "the shipped budget of {budget} costs {} correct answers against an unlimited one \
         ({capped_right} capped, {uncapped_right} uncapped, {capped_wrong} wrong)",
        uncapped_right as i64 - capped_right as i64
    );

    // Recall survives every one of those rewrites.
    let mut after = 0u32;
    for r in 0..ROOMS {
        if recall(&mut brain, &format!("r{r:03} desk material?")).as_deref()
            == Some(format!("m{:03}", r % 7).as_bytes())
        {
            after += 1;
        }
    }
    assert_eq!(after, ROOMS, "recall did not survive the derivations");
}

/// A cut the cache has never seen must still be found, at the cost it always
/// cost. The cache is an ORDERING, so a world it has never met can only be
/// slower by the handful of misses it spends first — never unanswerable.
#[test]
fn an_unseen_cut_shape_still_derives_after_the_cache_is_full_of_others() {
    let mut brain = subject();
    teach_world(&mut brain);
    // Warm the cache on the 3-hop shapes.
    for r in 0..4 {
        let q = format!("r{r:03} next lamp on material?");
        brain.derive_by_substitution(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 4096);
    }
    let warmed = brain.derivation_cut_hints().to_vec();
    assert!(!warmed.is_empty(), "the cache learned nothing to be wrong about");

    // A relation of a different length, so its cut sits at an offset the cache
    // does not hold.
    for r in 0..ROOMS {
        teach(&mut brain, &format!("r{r:03} ceiling-fixture?"), "bulb");
        teach(&mut brain, &format!("r{r:03} bulb wattage?"), "sixty");
    }
    let mut right = 0u32;
    for r in 0..ROOMS {
        let q = format!("r{r:03} ceiling-fixture wattage?");
        let answer = brain.derive_by_substitution(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 2, 4096);
        if answer.as_deref() == Some(b"sixty".as_ref()) {
            right += 1;
        }
    }
    eprintln!("unseen cut shape: {right} of {ROOMS}; hints now {:?}", brain.derivation_cut_hints());
    assert!(right > 0, "a cut shape the cache had never seen derived {right} of {ROOMS}");
    assert!(
        brain.derivation_cut_hints().len() <= DERIVATION_CUT_HINTS,
        "the cache grew past its bound"
    );
}
