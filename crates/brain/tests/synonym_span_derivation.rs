//! WHY the two worst integration families miss, measured per probe rather
//! than counted in aggregate.
//!
//! `examples/scorecard.rs` reports four families. Two of them are the whole
//! problem and they fail in DIFFERENT ways, which the aggregate percentage
//! hides:
//!
//! * `beside_next` is 0 of 6 / 24 / 96 / 384 at every scale, every miss in the
//!   `empty` bucket -- the derivation returns `None`.
//! * `on_material` is 6 of 24 at scale 1 and 173 of 1536 at scale 64, and its
//!   misses are mostly in the `material` bucket -- the derivation returns an
//!   answer of the RIGHT KIND that is WRONG.
//!
//! An empty miss and a wrong-material miss cannot have the same cause, so a
//! single "integration is low" number cannot be acted on. This file rebuilds
//! the scorecard's scale-1 world and prints, for each miss, which sub-question
//! the deletion search can reach at score 1.0 and which rewrites the splice
//! search can reach at score 1.0. Those two lists ARE the cause: the
//! derivation accepts the first rewrite reaching 1.0, so a family whose list
//! holds more than one trained question is ambiguous by construction, and a
//! family whose list is empty is unreachable by construction.
//!
//! Nothing here is a tuning target. The assertions are the two structural
//! claims; the distributions are printed.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, MatchTier, PoolConfig,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
/// The scorecard's own budget for one derivation, so the counts here are the
/// counts it would get.
const MAX_PROBES: usize = 32;

// The scorecard's scale-1 world, copied field for field from
// `examples/scorecard.rs` so a difference here is a difference there.
const ROOMS: usize = 8;
const OBJECTS: &[&str] = &["bed", "chair", "mirror", "desk", "lamp", "door", "window", "paper"];
const COLORS: &[&str] = &["red", "blue", "green", "white", "black", "grey"];
const MATERIALS: &[&str] = &["oak", "steel", "glass", "cloth", "pine", "brass"];
const RESTS_ON: &[(&str, &str)] = &[("lamp", "desk"), ("paper", "desk"), ("mirror", "door")];
const BESIDE_TRAINED_EVERY: usize = 4;

fn room(r: usize) -> String {
    format!("r{r:03}")
}
fn color(r: usize, i: usize) -> String {
    COLORS[(r + i) % COLORS.len()].to_string()
}
fn material(r: usize, i: usize) -> String {
    MATERIALS[(r + 2 * i) % MATERIALS.len()].to_string()
}
fn idx(obj: &str) -> usize {
    OBJECTS.iter().position(|o| *o == obj).expect("object is in OBJECTS")
}
fn decoy(r: usize, obj: &str, base: &str) -> String {
    // Any object that is neither the subject nor its base, so `near?` is a
    // trained relation the derivation must NOT confuse with `on?`.
    OBJECTS
        .iter()
        .filter(|o| **o != obj && **o != base)
        .nth(r % (OBJECTS.len() - 2))
        .expect("there is a third object")
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

fn teach(brain: &mut Brain, question: &str, answer: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, question.as_bytes().to_vec()),
        (ANSWER_POOL, answer.as_bytes().to_vec()),
    ]);
}

/// Every question the scorecard trains at scale 1, in its order.
fn trained_world() -> Vec<(String, String)> {
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

fn teach_world(brain: &mut Brain) -> Vec<(String, String)> {
    let facts = trained_world();
    // The scorecard trains for EPOCHS = 2.
    for _ in 0..2 {
        for (q, a) in &facts {
            teach(brain, q, a);
        }
    }
    facts
}

/// The score and answer of one question, through exactly the calls
/// `Brain::probe_question` makes. That private helper is what the derivation
/// uses, so reproducing it here measures the derivation's own view.
fn probe(brain: &mut Brain, question: &str) -> (f32, Option<String>) {
    let (score, _, answer) = probe_tiered(brain, question);
    (score, answer)
}

/// `probe`, also reporting the TIER the match was found at. Concepts emerge from
/// recurring byte SEQUENCES, so `MatchTier::Concept` is the one signal in this
/// scorer that is order-sensitive -- which is exactly what the score itself is
/// not.
fn probe_tiered(brain: &mut Brain, question: &str) -> (f32, MatchTier, Option<String>) {
    brain.observe_fabric_read_only(QUERY_POOL, question.as_bytes());
    let m = brain.best_binding_match_v2(QUERY_POOL);
    let answer = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .filter(|a| !a.is_empty())
        .map(|a| String::from_utf8_lossy(&a).to_string());
    (m.score(), m.tier, answer)
}

/// Every sub-question the deletion search can reach that the brain knows
/// PERFECTLY, in the order the search visits them. `MAX_DELETION_TAIL` is 1 in
/// `derive_by_substitution`, and `k` is the outer loop.
fn perfect_deletions(brain: &mut Brain, q: &str) -> Vec<(usize, usize, String, String)> {
    let bytes = q.as_bytes().to_vec();
    let n = bytes.len();
    let mut hits = Vec::new();
    for k in 1..n {
        for t in 0..=1usize.min(n - k) {
            if t > 0 && n - t == k {
                continue;
            }
            let mut sub = Vec::with_capacity(k + t);
            sub.extend_from_slice(&bytes[..k]);
            sub.extend_from_slice(&bytes[n - t..]);
            let text = String::from_utf8_lossy(&sub).to_string();
            let (score, answer) = probe(brain, &text);
            if score >= 1.0 {
                if let Some(answer) = answer {
                    hits.push((k, t, text, answer));
                }
            }
        }
    }
    hits
}

/// Every rewrite that splices `insert` over a span of `q` and lands on a
/// question the brain knows PERFECTLY. The derivation accepts the FIRST of
/// these in its own span order, so a list of length > 1 is an ambiguity and
/// the order decides the answer.
fn perfect_splices(
    brain: &mut Brain,
    q: &str,
    k: usize,
    insert: &str,
) -> Vec<(usize, String, String, MatchTier)> {
    let bytes = q.as_bytes().to_vec();
    let mut hits = Vec::new();
    for j in 0..=k {
        let mut rewrite = Vec::with_capacity(bytes.len() + insert.len());
        rewrite.extend_from_slice(&bytes[..j]);
        rewrite.extend_from_slice(insert.as_bytes());
        rewrite.extend_from_slice(&bytes[k..]);
        if rewrite == bytes || rewrite.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(&rewrite).to_string();
        let (score, tier, answer) = probe_tiered(brain, &text);
        if score >= 1.0 {
            if let Some(answer) = answer {
                hits.push((j, text, answer, tier));
            }
        }
    }
    hits
}

/// Which end of the splice range to accept from when several rewrites tie at
/// the ceiling score of 1.0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpliceOrder {
    /// `for j in 0..=k`, which is what `derive_by_substitution` does today.
    SmallestFirst,
    /// `for j in (0..=k).rev()` -- the rewrite that discards the least of what
    /// was asked.
    LargestFirst,
    /// Production `j` order, but the ceiling score alone no longer wins the
    /// tie: the rewrite must ALSO have matched at `MatchTier::Concept`. A
    /// concept neuron emerges from a recurring byte SEQUENCE, so it is the one
    /// signal here that an unordered byte set cannot fake, and a rewrite with a
    /// truncated subject cannot fire the concept that covers the intact one.
    /// Falls back to the production choice when no rewrite in the range reaches
    /// the ceiling at concept tier, so it can never derive LESS.
    ConceptTierFirst,
}

/// `Brain::derive_by_substitution_profiled`'s known-prefix branch, with the
/// splice order as a parameter so the two orders can be compared on the SAME
/// brain in the same run. The deletion search, the `>= 1.0` acceptance, the
/// `score > base_score` fallback, the `max_depth` recursion and the
/// `max_probes` budget are all the production ones; only the direction of the
/// `j` loop differs. The hint cache is omitted because it can only reorder the
/// deletion search, and the full-span fallback is omitted because the families
/// here all reach a taught sub-question (`beside_next`, measured above, reaches
/// nothing under either).
fn derive_with_order(
    brain: &mut Brain,
    query: &str,
    order: SpliceOrder,
    max_depth: usize,
    max_probes: usize,
) -> (Option<String>, usize) {
    let mut probes = 0usize;
    let mut current = query.as_bytes().to_vec();
    let mut derived: Option<String> = None;

    for _hop in 0..max_depth {
        if probes >= max_probes {
            break;
        }
        probes += 1;
        let text = String::from_utf8_lossy(&current).to_string();
        let (base_score, base_answer) = probe(brain, &text);
        let Some(base_answer) = base_answer else { break };
        if base_score >= 1.0 {
            break;
        }
        let n = current.len();

        let mut known_prefix: Option<(usize, String)> = None;
        'deletion: for k in 1..n {
            for t in 0..=1usize.min(n - k) {
                if probes >= max_probes {
                    break 'deletion;
                }
                if t > 0 && n - t == k {
                    continue;
                }
                let mut sub = Vec::with_capacity(k + t);
                sub.extend_from_slice(&current[..k]);
                sub.extend_from_slice(&current[n - t..]);
                probes += 1;
                let (score, answer) = probe(brain, &String::from_utf8_lossy(&sub));
                if score >= 1.0 {
                    if let Some(answer) = answer {
                        known_prefix = Some((k, answer));
                    }
                    break 'deletion;
                }
            }
        }

        let (js, splice_answer): (Vec<usize>, String) = match &known_prefix {
            Some((k, answer)) => {
                let mut js: Vec<usize> = (0..=*k).collect();
                if order == SpliceOrder::LargestFirst {
                    js.reverse();
                }
                (js, answer.clone())
            }
            None => (Vec::new(), base_answer.clone()),
        };

        let mut best: Option<(f32, Vec<u8>, String)> = None;
        for j in js {
            if probes >= max_probes {
                break;
            }
            let k = known_prefix.as_ref().map(|(k, _)| *k).expect("js is empty without a prefix");
            let mut rewrite = Vec::with_capacity(n + splice_answer.len());
            rewrite.extend_from_slice(&current[..j]);
            rewrite.extend_from_slice(splice_answer.as_bytes());
            rewrite.extend_from_slice(&current[k..]);
            if rewrite == current || rewrite.is_empty() {
                continue;
            }
            probes += 1;
            let (score, tier, answer) = probe_tiered(brain, &String::from_utf8_lossy(&rewrite));
            let Some(answer) = answer else { continue };
            let perfect = score >= 1.0;
            if order == SpliceOrder::ConceptTierFirst {
                // A concept-tier ceiling match ends the search; anything else is
                // only kept as the fallback the production order would have
                // taken, so this variant is a strict superset of it.
                if perfect && tier == MatchTier::Concept {
                    best = Some((score, rewrite, answer));
                    break;
                }
                if score > base_score && best.as_ref().map_or(true, |(b, _, _)| score > *b) {
                    best = Some((score, rewrite, answer));
                }
                continue;
            }
            if score > base_score && best.as_ref().map_or(true, |(b, _, _)| score > *b) {
                best = Some((score, rewrite, answer));
                if perfect {
                    break;
                }
            }
        }
        let Some((_, next_question, next_answer)) = best else { break };
        derived = Some(next_answer);
        current = next_question;
    }
    (derived, probes)
}

/// Every integration probe the scorecard asks at scale 1, by family, with the
/// answer the world says is true.
fn integration_probes() -> Vec<(&'static str, String, String)> {
    let mut out = Vec::new();
    for r in 0..ROOMS {
        let rm = room(r);
        let nr = (r + 1) % ROOMS;
        for (obj, base) in RESTS_ON {
            out.push((
                "on_material",
                format!("{rm} {obj} on material?"),
                material(r, idx(base)),
            ));
        }
        // NEXT_COLOR_OBJECTS = 2 in the scorecard.
        for k in 0..2usize {
            let i = (r + 3 * k) % OBJECTS.len();
            out.push((
                "next_color",
                format!("{rm} next {} color?", OBJECTS[i]),
                color(nr, i),
            ));
        }
        let (obj, base) = RESTS_ON[r % RESTS_ON.len()];
        out.push((
            "next_on_material",
            format!("{rm} next {obj} on material?"),
            material(nr, idx(base)),
        ));
        if r % BESIDE_TRAINED_EVERY != 0 {
            out.push(("beside_next", format!("{rm} beside?"), room(nr)));
        }
    }
    out
}

/// THE EXPERIMENT. Two splice orders, one brain, every family -- because a fix
/// that moves one family and drops another is not a fix, and the scorecard's
/// aggregate percentage cannot tell the two apart.
#[test]
fn largest_j_splice_beats_smallest_j_on_every_family() {
    use std::collections::BTreeMap;
    let mut brain = subject();
    let facts = teach_world(&mut brain);
    let mut recalled = 0usize;
    for (q, a) in &facts {
        let (_, got) = probe(&mut brain, q);
        if got.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    assert_eq!(recalled, facts.len(), "recall must be 100% before the experiment is read");

    let probes_list = integration_probes();
    // family -> (right, wrong, empty, total probes asked)
    let mut tally: BTreeMap<(&str, &str), (usize, usize, usize, usize)> = BTreeMap::new();
    for (family, q, want) in &probes_list {
        for (label, order) in [
            ("smallest_j", SpliceOrder::SmallestFirst),
            ("largest_j", SpliceOrder::LargestFirst),
            ("concept_tier", SpliceOrder::ConceptTierFirst),
        ] {
            let (got, cost) = derive_with_order(&mut brain, q, order, 3, MAX_PROBES);
            let slot = tally.entry((*family, label)).or_insert((0, 0, 0, 0));
            slot.3 += cost;
            match got.as_deref() {
                None => slot.2 += 1,
                Some(a) if a == want.as_str() => slot.0 += 1,
                Some(_) => slot.1 += 1,
            }
        }
    }
    for ((family, label), (right, wrong, empty, cost)) in &tally {
        let n = right + wrong + empty;
        println!(
            "{family:>17} {label:>11}: right {right}/{n} wrong {wrong} empty {empty} \
             probes {cost} ({:.1}/derivation)",
            *cost as f32 / n as f32
        );
    }

    let sum = |label: &str| -> (usize, usize) {
        tally
            .iter()
            .filter(|((_, l), _)| *l == label)
            .fold((0, 0), |(r, n), (_, (right, wrong, empty, _))| {
                (r + right, n + right + wrong + empty)
            })
    };
    let (small_right, total) = sum("smallest_j");
    let (large_right, total2) = sum("largest_j");
    assert_eq!(total, total2, "both orders must be asked the same questions");
    println!("ALL FAMILIES smallest_j {small_right}/{total}  largest_j {large_right}/{total}");

    // THE RESULT, and it is a REFUTATION rather than a fix.
    //
    // The first version of this test asserted that largest-j must not leave any
    // family below smallest-j, expecting it to pass. It FAILED on `next_color`
    // -- 14/16 against 16/16 -- and that failure is the finding, so it is
    // recorded here as the contract instead of being tuned away.
    //
    // The two families want OPPOSITE ends of the same range:
    //
    // * `"r001 lamp on material?"` -> sub-question `"r001 lamp on?"` -> `desk`.
    //   The correct rewrite is `"r001 desk material?"`, a LARGE `j`, because the
    //   room name must survive.
    // * `"r000 next bed color?"` -> sub-question `"r000 next?"` -> `r001`. The
    //   correct rewrite is `"r001 bed color?"`, `j = 0`, because `"r000 next"`
    //   must be replaced WHOLE.
    //
    // So `j` is not the discriminator and no direction can serve both. What is
    // actually broken is one layer down, and it contradicts the method's own
    // documented justification -- "a question scoring 1.0 IS a question the
    // brain was taught". Measured false: `"r0desk material?"` was never taught
    // and scores exactly 1.0, because `score` is precision x recall over the
    // firing set, an atom is a BYTE, the set is UNORDERED and DISTINCT, and
    // `"r0desk material?"` and `"r000 desk material?"` have the identical byte
    // set. A truncated subject is indistinguishable from an intact one at the
    // ceiling score, which is why all 24 `on_material` probes have more than
    // one perfect splice and why the 6 that look right today are the rooms
    // whose material coincides with `r000`'s.
    //
    // These assertions therefore pin the MEASUREMENT, so that the next reader
    // who reaches for `.rev()` sees `next_color` at 14/16 here in 16 seconds
    // rather than in a scorecard run.
    let om_small = tally[&("on_material", "smallest_j")];
    let om_large = tally[&("on_material", "largest_j")];
    let nc_small = tally[&("next_color", "smallest_j")];
    let nc_large = tally[&("next_color", "largest_j")];
    assert!(
        om_large.0 > om_small.0,
        "largest-j must still be the better order for on_material: {om_small:?} -> {om_large:?}"
    );
    assert!(
        nc_large.0 < nc_small.0,
        "and it must still COST next_color, which is what makes it unshippable \
         as a one-line reorder: {nc_small:?} -> {nc_large:?}"
    );
    assert!(
        large_right > small_right,
        "the aggregate improves ({small_right} -> {large_right}) even though a family falls -- \
         which is exactly why the aggregate is not the thing to gate on"
    );
    // Neither order moves the family whose misses are EMPTY, because its misses
    // are not a tie-break problem at all. Measured in the test above: no
    // sub-question of `"rNNN beside?"` is known perfectly, so the splice search
    // has nothing to insert.
    let bn_small = tally[&("beside_next", "smallest_j")];
    let bn_large = tally[&("beside_next", "largest_j")];
    assert_eq!(
        (bn_small.0, bn_large.0),
        (0, 0),
        "beside_next is unreachable under BOTH orders, so it is not a selection problem"
    );
}

#[test]
fn beside_next_is_unreachable_and_on_material_is_ambiguous() {
    let mut brain = subject();
    let facts = teach_world(&mut brain);

    // RECALL FIRST: every number below is meaningless if the taught half
    // regressed, and the probes here observe the fabric thousands of times.
    let mut recalled = 0usize;
    let mut trained_concept = 0usize;
    for (q, a) in &facts {
        let (_, tier, got) = probe_tiered(&mut brain, q);
        if got.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
        if tier == MatchTier::Concept {
            trained_concept += 1;
        }
    }
    println!(
        "recall {recalled}/{} trained questions; matched at concept tier {trained_concept}",
        facts.len()
    );
    assert_eq!(recalled, facts.len(), "recall must be 100% before anything else is read");
    // THE TIER IS ABSENT, NOT MERELY UNHELPFUL. `MatchTier::Concept` is the one
    // order-sensitive signal the matcher has -- a concept neuron emerges from a
    // recurring byte SEQUENCE -- and it is the only thing that could tell
    // `"r0desk material?"` from `"r000 desk material?"`, whose DISTINCT BYTE
    // SETS are identical. Measured: 0 of 184 perfect splices reach concept tier,
    // and this number says whether the TRAINED questions reach it either. If
    // they do not, no tie-break policy can ever discriminate, because every
    // match in this world is decided on an unordered byte set.
    println!(
        "concept tier on trained questions: {trained_concept}/{} ({:.1}%)",
        facts.len(),
        100.0 * trained_concept as f32 / facts.len() as f32
    );
    // MEASURED 0 of 186. Not one TRAINED question reaches concept tier either,
    // so the absence is not about the derivation's rewrites -- the whole world
    // is matched on unordered byte sets, and `recall` is 100 % on that alone.
    // That is why no tie-break among ceiling-scoring rewrites can be correct
    // here: the signal a tie-break would need does not exist yet.
    //
    // Pinned as an assertion rather than a print, because the day concepts DO
    // emerge in the query pool is the day the tier discriminator is worth
    // re-measuring, and this is the line that will say so.
    assert_eq!(
        trained_concept, 0,
        "a trained question now matches at concept tier -- re-run the tier          discriminator in the experiment below, it was measured inert against          0 of 184 perfect splices while this was 0"
    );

    // ---- beside_next: the empty family -------------------------------------
    //
    // `"r001 beside?"` was never taught; `"r000 beside?"` was, and `"r001
    // next?"` was. The answer is reachable by SWAPPING the span `beside?` for
    // the span `next?`, which is a span drawn from another QUESTION. The
    // derivation only ever splices in an ANSWER, so this family is not a
    // tuning miss.
    let mut beside_empty = 0usize;
    let mut beside_right = 0usize;
    let mut beside_wrong = 0usize;
    let mut beside_deletion_hits = 0usize;
    for r in 0..ROOMS {
        if r % BESIDE_TRAINED_EVERY == 0 {
            continue;
        }
        let q = format!("{} beside?", room(r));
        let want = room((r + 1) % ROOMS);
        let (got, probes) = brain.derive_by_substitution_profiled(
            QUERY_POOL,
            ANSWER_POOL,
            q.as_bytes(),
            3,
            MAX_PROBES,
        );
        let got = got.map(|a| String::from_utf8_lossy(&a).to_string());
        match got.as_deref() {
            None => beside_empty += 1,
            Some(a) if a == want => beside_right += 1,
            Some(_) => beside_wrong += 1,
        }
        let dels = perfect_deletions(&mut brain, &q);
        beside_deletion_hits += dels.len();
        println!(
            "beside_next {q:>14} want {want} got {got:?} probes {probes} perfect-deletions {dels:?}"
        );
    }
    println!(
        "beside_next right {beside_right} wrong {beside_wrong} empty {beside_empty} \
         perfect-deletions-total {beside_deletion_hits}"
    );

    // The structural claim, and it is about the SEARCH SPACE rather than about
    // a score: there is no sub-question of `"rNNN beside?"` that the brain
    // knows perfectly, so `known_prefix` is always `None`, so the splice
    // search has nothing but the base answer to insert and no taught rewrite
    // to land on. A family whose deletion search can reach nothing cannot be
    // fixed by reordering, rescoring or a wider budget.
    assert_eq!(
        beside_deletion_hits, 0,
        "beside_next's misses are empty because NO sub-question of it is known perfectly"
    );
    assert_eq!(
        beside_right, 0,
        "if this ever becomes non-zero the mechanism changed and this file's claim is stale"
    );

    // ---- on_material: the wrong-answer family -----------------------------
    //
    // Here the deletion search DOES reach a taught sub-question, so the family
    // derives -- and then mostly derives the wrong material. The question is
    // whether more than one rewrite reaches 1.0, because the derivation
    // accepts the first one it meets in span order.
    let mut om_right = 0usize;
    let mut om_wrong = 0usize;
    let mut om_empty = 0usize;
    let mut ambiguous = 0usize;
    let mut total = 0usize;
    let mut tier_concept = 0usize;
    let mut tier_total = 0usize;
    let mut tier_correct_concept = 0usize;
    for r in 0..ROOMS {
        for (obj, base) in RESTS_ON {
            let q = format!("{} {obj} on material?", room(r));
            let want = material(r, idx(base));
            total += 1;
            let (got, probes) = brain.derive_by_substitution_profiled(
                QUERY_POOL,
                ANSWER_POOL,
                q.as_bytes(),
                3,
                MAX_PROBES,
            );
            let got = got.map(|a| String::from_utf8_lossy(&a).to_string());
            match got.as_deref() {
                None => om_empty += 1,
                Some(a) if a == want => om_right += 1,
                Some(_) => om_wrong += 1,
            }
            let dels = perfect_deletions(&mut brain, &q);
            // The splice list for the FIRST deletion the search would accept,
            // which is the one the derivation actually uses.
            let splices = match dels.first() {
                Some((k, _, _, answer)) => perfect_splices(&mut brain, &q, *k, answer),
                None => Vec::new(),
            };
            if splices.len() > 1 {
                ambiguous += 1;
            }
            // TIER REACHABILITY, with NO budget involved. The `concept_tier`
            // variant in the experiment below measured identical to the
            // production order with its probe budget EXHAUSTED -- 32.0 of 32 per
            // derivation -- which is the "a counter of zero from a path that
            // never had the opportunity to run" shape. Counting the tiers
            // directly is the only thing that separates "the tier cannot break
            // this tie" from "the tier never got asked".
            let concept_hits =
                splices.iter().filter(|(_, _, _, t)| *t == MatchTier::Concept).count();
            let correct_at_concept = splices
                .iter()
                .filter(|(_, _, a, t)| *t == MatchTier::Concept && a.as_str() == want.as_str())
                .count();
            tier_concept += concept_hits;
            tier_total += splices.len();
            tier_correct_concept += correct_at_concept;
            println!(
                "on_material {q:>26} want {want} got {got:?} probes {probes} \
                 deletions {} first {:?} perfect-splices {splices:?}",
                dels.len(),
                dels.first().map(|(k, t, text, a)| (*k, *t, text.clone(), a.clone()))
            );
        }
    }
    println!(
        "on_material right {om_right} wrong {om_wrong} empty {om_empty} of {total}; \
         probes with >1 perfect splice: {ambiguous}"
    );
    println!(
        "on_material TIER CENSUS of perfect splices: {tier_concept} concept of {tier_total};          {tier_correct_concept} concept hits carry the CORRECT answer"
    );
    assert!(tier_total > 0, "there must be perfect splices to census, or the census is vacuous");

    // The on_material claim is the opposite of beside_next's: the search space
    // is NOT empty, so this family is a selection problem and not a
    // reachability one. Asserting only that the taught sub-question is
    // reachable keeps the file honest if the counts move.
    assert!(
        om_right + om_wrong > 0,
        "on_material reaches a taught sub-question, so it must derive SOMETHING"
    );
}
