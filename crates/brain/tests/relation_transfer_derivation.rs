//! `beside_next` is 0 of 6 / 24 / 96 / 384 at EVERY scale, and this file
//! measures why, then measures a mechanism that fixes it -- both through the
//! public API only, so the production wiring is a call and not a redesign.
//!
//! # What the aggregate hides
//!
//! `tests/synonym_span_derivation.rs` established that every match in this
//! world is precision x recall over the UNORDERED DISTINCT BYTE SET (0 of 186
//! trained questions reach `MatchTier::Concept`), and asserted that
//! `beside_next` is a REACHABILITY problem: no sub-question of `"rNNN beside?"`
//! is known perfectly, so the deletion search finds no prefix and the splice
//! search has nothing to insert.
//!
//! That is true and it is not the whole cause. Modelling the matcher's own
//! scorer over the scale-1 corpus (186 trained questions, scored against each
//! of the 6 held-out `beside?` probes) gives the SAME four numbers for all six:
//!
//! ```text
//!   r001 beside?  want r002  TOP 0.9000 "r000 beside?" -> r001  ties = 1
//!                            its own "r001 next?" 0.4000, rank 44 of 186
//! ```
//!
//! So the top match is UNIQUE at 0.90 -- not a tie -- and it is the wrong room.
//! Nine of the ten distinct bytes of `"r001 beside?"` are shared with
//! `"r000 beside?"`, while the three bytes that are the entire question
//! (`001` against `000`) are worth 0.1. The question that actually holds the
//! answer sits at rank 44. That makes `beside_next` neither a tie-break problem
//! nor a budget problem: the RELATION WORD dominates the SUBJECT by 9:1, so any
//! mechanism that reaches the `beside?` relation drags `r000` along with it.
//!
//! # The mechanism: subject-preserving relation transfer
//!
//! Nothing below knows the strings `beside` or `next`. The general statement is:
//!
//! 1. A query that scores below the ceiling has a best trained answer `A`.
//! 2. Reverse-decoding `A` (observe it in the ANSWER pool, decode into the
//!    QUERY pool) returns a trained QUESTION `T` that produces `A`. `T` is the
//!    neighbourhood the query landed in -- including its subject.
//! 3. The query and `T` are aligned on their common prefix and suffix. The spans
//!    that differ are the candidate substitution: `T`'s differing span is a
//!    RELATION the fabric has seen, and the query's own differing span is the
//!    SUBJECT it must keep.
//! 4. Every contiguous span of `T` is spliced over every span of the query, and
//!    a rewrite scoring at the ceiling wins.
//!
//! Step 4 is the same accept-at-1.0 rule `derive_by_substitution` already uses.
//! The only new idea is that the INSERT may come from a trained QUESTION rather
//! than only from an ANSWER, which is exactly the span `beside_next` needs and
//! which no reordering or wider budget can supply.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, MatchTier, PoolConfig,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

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

fn trained_world() -> Vec<(String, String)> {
    trained_world_n(ROOMS)
}

/// The same world at any room count, so a mechanism proven at the scorecard's
/// scale 1 can be re-measured at its scale 4 without a scorecard run. Scale is
/// the one axis every integration family decays along -- `next_color` is 100 %
/// at scale 1 and 49.8 % at scale 64 -- so a scale-1-only result is a
/// hypothesis about scale 4, not a measurement of it.
fn trained_world_n(rooms: usize) -> Vec<(String, String)> {
    let mut facts = Vec::new();
    for r in 0..rooms {
        let rm = room(r);
        let next = room((r + 1) % rooms);
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
    teach_world_n(brain, ROOMS)
}

fn teach_world_n(brain: &mut Brain, rooms: usize) -> Vec<(String, String)> {
    let facts = trained_world_n(rooms);
    for _ in 0..2 {
        for (q, a) in &facts {
            teach(brain, q, a);
        }
    }
    facts
}

/// Every integration probe the scorecard asks at scale 1, by family, copied
/// from `examples/scorecard.rs` so a difference here is a difference there.
fn integration_probes() -> Vec<(&'static str, String, String)> {
    integration_probes_n(ROOMS)
}

fn integration_probes_n(rooms: usize) -> Vec<(&'static str, String, String)> {
    let mut out = Vec::new();
    for r in 0..rooms {
        let rm = room(r);
        let nr = (r + 1) % rooms;
        for (obj, base) in RESTS_ON {
            out.push(("on_material", format!("{rm} {obj} on material?"), material(r, idx(base))));
        }
        // NEXT_COLOR_OBJECTS = 2 in the scorecard.
        for k in 0..2usize {
            let i = (r + 3 * k) % OBJECTS.len();
            out.push(("next_color", format!("{rm} next {} color?", OBJECTS[i]), color(nr, i)));
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

/// Score and answer of a question in the QUERY pool, through the same calls the
/// derivation makes.
fn ask(brain: &mut Brain, question: &str) -> (f32, Option<String>) {
    brain.observe_fabric_read_only(QUERY_POOL, question.as_bytes());
    let score = brain.best_binding_match_v2(QUERY_POOL).score();
    let answer = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .filter(|a| !a.is_empty())
        .map(|a| String::from_utf8_lossy(&a).to_string());
    (score, answer)
}

/// REVERSE decode: observe an ANSWER and read back a trained QUESTION that
/// produces it. This is the one call the transfer mechanism needs that the
/// existing derivation never makes, and `decode_best_trained_binding` refuses
/// `query_pool == target_pool`, so the pools must genuinely be swapped.
fn reverse(brain: &mut Brain, answer: &str) -> Option<String> {
    brain.observe_fabric_read_only(ANSWER_POOL, answer.as_bytes());
    brain
        .decode_best_trained_binding(ANSWER_POOL, QUERY_POOL)
        .filter(|q| !q.is_empty())
        .map(|q| String::from_utf8_lossy(&q).to_string())
}

/// THE FIRST MEASUREMENT: the two API facts the mechanism rests on, printed
/// before anything is built on them. Either one being absent kills the design,
/// and a design killed in 20 seconds here is cheaper than one killed in a
/// scorecard run.
#[test]
fn the_two_api_facts_the_transfer_needs_are_present() {
    let mut brain = subject();
    let facts = teach_world(&mut brain);

    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    assert_eq!(recalled, facts.len(), "recall must be 100% before anything else is read");

    let mut forward_subceiling = 0usize;
    let mut reverse_nonempty = 0usize;
    let mut reverse_is_a_trained_question = 0usize;
    let mut probes = 0usize;
    for r in 0..ROOMS {
        if r % BESIDE_TRAINED_EVERY == 0 {
            continue;
        }
        probes += 1;
        let q = format!("{} beside?", room(r));
        let (score, answer) = ask(&mut brain, &q);
        if score < 1.0 {
            forward_subceiling += 1;
        }
        let back = answer.as_deref().and_then(|a| reverse(&mut brain, a));
        if back.is_some() {
            reverse_nonempty += 1;
        }
        if let Some(t) = back.as_deref() {
            if facts.iter().any(|(fq, _)| fq == t) {
                reverse_is_a_trained_question += 1;
            }
        }
        println!("probe {q:>14} score {score:.4} answer {answer:?} reverse-decode {back:?}");
    }
    println!(
        "FACT 1 forward sub-ceiling {forward_subceiling}/{probes}; \
         FACT 2 reverse decode non-empty {reverse_nonempty}/{probes}, \
         of which a TRAINED question {reverse_is_a_trained_question}"
    );
    // FACT 1: the family is sub-ceiling, so a derivation is attempted at all.
    assert_eq!(
        forward_subceiling, probes,
        "every held-out beside? probe must score below the ceiling, or it is recall not integration"
    );
    // FACT 2: the reverse direction is usable. If this is 0 the mechanism has no
    // span source and the design below is dead -- which is the point of
    // measuring it in its own test.
    assert_eq!(
        reverse_nonempty, probes,
        "reverse decode (answer pool -> query pool) must return a question, or there is no span to transfer"
    );
}

/// Longest common prefix and suffix of two byte strings, as byte counts. The
/// spans between them are what differ, and they are the substitution.
fn align(a: &[u8], b: &[u8]) -> (usize, usize) {
    let p = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let max_s = a.len().min(b.len()) - p;
    let s = a
        .iter()
        .rev()
        .zip(b.iter().rev())
        .take(max_s)
        .take_while(|(x, y)| x == y)
        .count();
    (p, s)
}

/// SUBJECT-PRESERVING RELATION TRANSFER, written only against the public API so
/// that wiring it into `derive_by_substitution` is a call rather than a port.
///
/// Returns the derived answer and the number of fabric probes spent, so its
/// cost is reported the way `derive_by_substitution_profiled` reports its own.
fn transfer(brain: &mut Brain, query: &str, max_probes: usize) -> (Option<String>, usize) {
    let mut probes = 1usize;
    let (base_score, base_answer) = ask(brain, query);
    if base_score >= 1.0 {
        return (base_answer, probes);
    }
    let Some(base_answer) = base_answer else { return (None, probes) };

    // The neighbourhood the query landed in, named: a trained question that
    // produces the answer the query was about to return.
    probes += 1;
    let Some(t) = reverse(brain, &base_answer) else { return (None, probes) };

    let q = query.as_bytes();
    let tb = t.as_bytes();
    // Align, so the SUBJECT the query asked about is the part that must survive
    // and `t`'s differing span is the relation to borrow. The aligned
    // substitution is tried first; it is the one the alignment argues for.
    let (p, s) = align(q, tb);
    let mut candidates: Vec<Vec<u8>> = Vec::new();
    if p + s < q.len() && p + s < tb.len() {
        let mut first = Vec::with_capacity(tb.len());
        first.extend_from_slice(&q[..p]);
        first.extend_from_slice(&tb[p..tb.len() - s]);
        first.extend_from_slice(&q[q.len() - s..]);
        candidates.push(first);
    }
    // THEN A TAIL TRANSFER, ORDERED BY ALIGNMENT RATHER THAN BY LENGTH. A
    // relation is the TAIL of a question in this world and the subject is its
    // head, so the rewrite wanted is `query[..a] ++ t[i..]`: keep `a` bytes of
    // what was asked, borrow `t`'s relation from `i`.
    //
    // The order is the whole cost. Sorting by insert length first measured 50.0
    // probes per derivation; sorting by `|a - i|` -- the pairs where the borrowed
    // tail starts where the query's own tail starts -- puts the correct rewrite
    // among the first few, because two questions about the same KIND of thing
    // have their relation at nearly the same offset. Ties go to the longer
    // borrow, which is the one that replaces more of the relation and less of
    // the subject.
    let mut tails: Vec<(usize, usize)> = Vec::new();
    for a in 0..q.len() {
        for i in 0..tb.len() {
            tails.push((a, i));
        }
    }
    tails.sort_by_key(|(a, i)| (a.abs_diff(*i), std::cmp::Reverse(tb.len() - i)));
    for (a, i) in tails {
        let mut rw = Vec::with_capacity(a + (tb.len() - i));
        rw.extend_from_slice(&q[..a]);
        rw.extend_from_slice(&tb[i..]);
        if rw != q && !rw.is_empty() {
            candidates.push(rw);
        }
    }
    // Only then every contiguous span of `t` over every contiguous span of the
    // query, longest insert first. This is the exhaustive fallback; the accept
    // rule is the existing one -- a rewrite scoring at the ceiling IS a question
    // the brain was taught -- so a wrong span cannot win, it can only waste a
    // probe.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for i in 0..tb.len() {
        for j in (i + 1)..=tb.len() {
            spans.push((i, j));
        }
    }
    spans.sort_by_key(|(i, j)| std::cmp::Reverse(j - i));
    for (i, j) in spans {
        for a in 0..q.len() {
            for b in (a + 1)..=q.len() {
                let mut rw = Vec::with_capacity(q.len() + (j - i));
                rw.extend_from_slice(&q[..a]);
                rw.extend_from_slice(&tb[i..j]);
                rw.extend_from_slice(&q[b..]);
                if rw != q && !rw.is_empty() {
                    candidates.push(rw);
                }
            }
        }
    }

    let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    for rw in candidates {
        if probes >= max_probes {
            break;
        }
        if !seen.insert(rw.clone()) {
            continue;
        }
        probes += 1;
        let text = String::from_utf8_lossy(&rw).to_string();
        let (score, answer) = ask(brain, &text);
        if score >= 1.0 {
            if let Some(answer) = answer {
                // The rewrite must have asked something DIFFERENT from what the
                // query already resolved to, or the transfer has only
                // rediscovered the wrong-room neighbour it started from.
                if answer != base_answer {
                    return (Some(answer), probes);
                }
            }
        }
    }
    (None, probes)
}

/// THE RESULT: the family that is 0% at every scale, under the transfer, with
/// recall re-measured afterwards because every probe observes the fabric.
#[test]
fn relation_transfer_derives_the_family_that_is_zero_at_every_scale() {
    let mut brain = subject();
    let facts = teach_world(&mut brain);

    let budget = 4096usize;
    let mut right = 0usize;
    let mut wrong = 0usize;
    let mut empty = 0usize;
    let mut cost = 0usize;
    let mut n = 0usize;
    for r in 0..ROOMS {
        if r % BESIDE_TRAINED_EVERY == 0 {
            continue;
        }
        n += 1;
        let q = format!("{} beside?", room(r));
        let want = room((r + 1) % ROOMS);
        let (got, probes) = transfer(&mut brain, &q, budget);
        cost += probes;
        match got.as_deref() {
            None => empty += 1,
            Some(a) if a == want => right += 1,
            Some(_) => wrong += 1,
        }
        println!("beside_next {q:>14} want {want} got {got:?} probes {probes}");
    }
    println!(
        "beside_next UNDER TRANSFER: right {right}/{n} wrong {wrong} empty {empty}; \
         probes {cost} ({:.1}/derivation)",
        cost as f32 / n as f32
    );

    // RECALL AFTER, not before: the transfer observes the fabric thousands of
    // times and `observe_fabric_read_only` is the call the RAM work is about, so
    // a mechanism that derives by damaging recall is not a gain.
    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    println!("recall after transfer {recalled}/{}", facts.len());
    assert_eq!(recalled, facts.len(), "recall must still be 100% after the transfer has run");

    // The claim this file exists to settle. `synonym_span_derivation.rs` asserts
    // the production derivation gets 0 here under BOTH splice orders; anything
    // above 0 is a mechanism the production path does not have.
    assert!(
        right > 0,
        "the transfer must derive at least one beside_next the production derivation cannot: \
         right {right} wrong {wrong} empty {empty}"
    );
    // AND IT MUST BE AFFORDABLE. `derive_by_substitution` is budgeted at 32
    // probes in the scorecard, and one probe is one
    // `observe_fabric_read_only` -- the call the RAM work is about. A mechanism
    // that derives this family for more than the existing budget is a cost
    // increase dressed as an integration gain, so the bound is asserted rather
    // than printed. Measured 50.0 under length-first candidate order and
    // {see the printed line} under alignment-first.
    let per = cost as f32 / n as f32;
    assert!(
        per <= 32.0,
        "the transfer must fit the existing 32-probe derivation budget, measured {per:.1}"
    );
}

/// CAN THE REVERSE DECODE ALSO TELL A TAUGHT REWRITE FROM AN UNTAUGHT ONE?
///
/// `on_material` is the opposite failure from `beside_next` and the larger one:
/// 1,536 probes at scale 64 at 11.1 %, 513 of the misses a WRONG material and
/// 852 empty. `synonym_span_derivation.rs` measured the cause -- the accept rule
/// "a question scoring 1.0 IS a question the brain was taught" is FALSE, because
/// the score is precision x recall over an unordered distinct byte set and
/// `"r0desk material?"` has the identical byte set to `"r000 desk material?"`.
/// So a truncated subject is accepted at the ceiling.
///
/// The transfer above needed a trained QUESTION for a different purpose, and the
/// same call is a candidate soundness test: ask the rewrite, take its answer,
/// reverse-decode that answer, and require the question that comes back to be
/// BYTE-EQUAL to the rewrite. A rewrite that was really taught should decode
/// back to itself; `"r0desk material?"` never can.
///
/// The complement, and it is the likely one: many questions share each material
/// answer (6 materials over 8 objects x 8 rooms), so the reverse decode returns
/// the best of ~10 and almost never the rewrite -- in which case the test
/// rejects the CORRECT rewrite too and is useless. This test measures which,
/// per splice, and asserts only what it measures.
///
/// MEASURED, AND IT IS THE COMPLEMENT: 184 perfect splices of which 24 were
/// really taught; round-trip to self 0; as a detector, true-pos 0, false-pos 0,
/// false-neg 24, true-neg 160. It rejects every taught rewrite as well as every
/// untaught one, so it carries NO information and is dead. Kept as a test
/// rather than deleted, because "reverse-decode the answer and check it comes
/// back" is the obvious next idea for anyone reading the transfer above, and
/// the 2x2 here costs 0.6 s against a scorecard run. The reason is in the
/// numbers: a material answer is shared by ~10 trained questions, so the
/// reverse decode returns the best of the ten and the odds it is the rewrite
/// are ~1 in 10 even when the rewrite WAS taught -- and measured, 0 in 24.
/// A one-best reverse decode cannot test set membership.
#[test]
fn does_the_reverse_decode_separate_a_taught_rewrite_from_a_byte_set_twin() {
    let mut brain = subject();
    let facts = teach_world(&mut brain);

    let mut perfect_total = 0usize;
    let mut perfect_taught = 0usize;
    let mut roundtrip_self = 0usize;
    // The 2x2 that decides it: does round-tripping to yourself predict having
    // been taught?
    let (mut tp, mut fp, mut fn_, mut tn) = (0usize, 0usize, 0usize, 0usize);
    for r in 0..ROOMS {
        for (obj, base) in RESTS_ON {
            let q = format!("{} {obj} on material?", room(r));
            // The sub-question the production deletion search accepts first, and
            // its answer, which is what gets spliced.
            let qb = q.as_bytes().to_vec();
            let n = qb.len();
            let mut prefix: Option<(usize, String)> = None;
            'del: for k in 1..n {
                for t in 0..=1usize.min(n - k) {
                    if t > 0 && n - t == k {
                        continue;
                    }
                    let mut sub = Vec::with_capacity(k + t);
                    sub.extend_from_slice(&qb[..k]);
                    sub.extend_from_slice(&qb[n - t..]);
                    let (score, answer) = ask(&mut brain, &String::from_utf8_lossy(&sub));
                    if score >= 1.0 {
                        if let Some(answer) = answer {
                            prefix = Some((k, answer));
                            break 'del;
                        }
                    }
                }
            }
            let Some((k, spliced)) = prefix else { continue };
            for j in 0..=k {
                let mut rw = Vec::with_capacity(n);
                rw.extend_from_slice(&qb[..j]);
                rw.extend_from_slice(spliced.as_bytes());
                rw.extend_from_slice(&qb[k..]);
                if rw == qb || rw.is_empty() {
                    continue;
                }
                let text = String::from_utf8_lossy(&rw).to_string();
                let (score, answer) = ask(&mut brain, &text);
                if score < 1.0 {
                    continue;
                }
                let Some(answer) = answer else { continue };
                perfect_total += 1;
                // GROUND TRUTH, available only to the test: was this exact
                // string ever taught? The mechanism never gets to look.
                let taught = facts.iter().any(|(fq, _)| *fq == text);
                if taught {
                    perfect_taught += 1;
                }
                let back = reverse(&mut brain, &answer);
                let same = back.as_deref() == Some(text.as_str());
                if same {
                    roundtrip_self += 1;
                }
                match (same, taught) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => tn += 1,
                }
            }
        }
    }
    println!(
        "on_material perfect splices {perfect_total}; TAUGHT {perfect_taught}; \
         round-trip to self {roundtrip_self}"
    );
    println!(
        "round-trip as a taught-detector: true-pos {tp} false-pos {fp} false-neg {fn_} true-neg {tn}"
    );
    assert!(perfect_total > 0, "there must be perfect splices, or the census is vacuous");
    // The one thing that must hold for the test to be worth anything: there
    // ARE untaught rewrites scoring at the ceiling. That is the defect, and it
    // is asserted so the census cannot go vacuous if the matcher changes.
    assert!(
        perfect_total > perfect_taught,
        "untaught rewrites must reach the ceiling, or the accept-at-1.0 rule is already sound: \
         {perfect_total} perfect of which {perfect_taught} taught"
    );
    // THE REFUTATION, pinned. If the reverse decode ever becomes a usable
    // membership test -- say it gains a top-k form -- this assertion fails and
    // says so, which is the only honest way to keep a dead idea on file.
    assert_eq!(
        (tp, fp),
        (0, 0),
        "the round-trip detector accepted something: it was measured to accept NOTHING \
         (tp 0, fp 0, fn 24, tn 160), so it is worth re-measuring as a soundness test"
    );
    assert!(fn_ > 0, "there were taught rewrites for it to have accepted and it accepted none");
    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    assert_eq!(recalled, facts.len(), "recall must still be 100%");
}

/// WHAT DOES WORK ON `on_material`: STOP ACCEPTING THE FIRST CEILING REWRITE
/// AND TAKE THE ANSWER THE CEILING REWRITES AGREE ON.
///
/// The production splice loop enumerates `j in 0..=k` and breaks on the first
/// rewrite scoring 1.0. Measured in `synonym_span_derivation.rs`, that first
/// one is routinely a rewrite with a TRUNCATED SUBJECT:
///
/// ```text
///   r001 lamp on material?  want steel  got oak
///     perfect splices: (2,"r0desk material?","oak") (3,"r00desk material?","oak")
///                      (4,"r001desk material?","steel") (5,"r001 desk material?","steel")
///                      (6,"r001 ldesk material?","steel") (7,..,"steel") (8,..,"steel")
/// ```
///
/// Two of the seven lost a byte of `r001` and answered `r000`'s material; five
/// kept it and answered correctly. The information needed to pick is already in
/// the probes the search makes -- it is thrown away by the early break.
///
/// So: probe every `j`, and return the MOST COMMON answer among the ceiling
/// rewrites rather than the first. Nothing here is specific to a family or a
/// wording: it is a vote over the search the derivation already performs.
///
/// The complement is live and is why both arms run on the same brain: a vote
/// could just as easily be dominated by the truncations, since there are more
/// short `j` than long ones in some shapes, and `next_color`'s correct rewrite
/// is `j = 0` -- the MOST truncated one. If the vote costs `next_color` it is
/// the `.rev()` trap again and unshippable.
#[test]
fn a_vote_over_ceiling_rewrites_beats_the_first_one() {
    use std::collections::BTreeMap;
    let mut brain = subject();
    let facts = teach_world(&mut brain);

    /// The production deletion search: first sub-question known at the ceiling.
    fn known_prefix(brain: &mut Brain, qb: &[u8]) -> Option<(usize, String)> {
        let n = qb.len();
        for k in 1..n {
            for t in 0..=1usize.min(n - k) {
                if t > 0 && n - t == k {
                    continue;
                }
                let mut sub = Vec::with_capacity(k + t);
                sub.extend_from_slice(&qb[..k]);
                sub.extend_from_slice(&qb[n - t..]);
                let (score, answer) = ask(brain, &String::from_utf8_lossy(&sub));
                if score >= 1.0 {
                    if let Some(answer) = answer {
                        return Some((k, answer));
                    }
                }
            }
        }
        None
    }

    // family -> (first-wins right, vote right, total)
    let mut tally: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    let mut votes_cost = 0usize;
    let mut first_cost = 0usize;
    for (family, q, want) in integration_probes() {
        let slot = tally.entry(family).or_insert((0, 0, 0));
        slot.2 += 1;
        let qb = q.as_bytes().to_vec();
        let Some((k, spliced)) = known_prefix(&mut brain, &qb) else { continue };
        let mut first: Option<String> = None;
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for j in 0..=k {
            let mut rw = Vec::with_capacity(qb.len());
            rw.extend_from_slice(&qb[..j]);
            rw.extend_from_slice(spliced.as_bytes());
            rw.extend_from_slice(&qb[k..]);
            if rw == qb || rw.is_empty() {
                continue;
            }
            let (score, answer) = ask(&mut brain, &String::from_utf8_lossy(&rw));
            votes_cost += 1;
            if score < 1.0 {
                continue;
            }
            let Some(answer) = answer else { continue };
            if first.is_none() {
                first = Some(answer.clone());
                first_cost = votes_cost;
            }
            *counts.entry(answer).or_insert(0) += 1;
        }
        // The vote: most common answer, ties broken by the one the production
        // order would have taken, so the vote can never be arbitrary.
        let winner = counts
            .iter()
            .max_by_key(|(a, n)| (**n, Some(a.as_str()) == first.as_deref()))
            .map(|(a, _)| a.clone());
        if first.as_deref() == Some(want.as_str()) {
            slot.0 += 1;
        }
        if winner.as_deref() == Some(want.as_str()) {
            slot.1 += 1;
        }
    }
    let mut f_tot = 0usize;
    let mut v_tot = 0usize;
    let mut n_tot = 0usize;
    for (family, (f, v, n)) in &tally {
        println!("{family:>17} first-wins {f}/{n} -> vote {v}/{n}");
        f_tot += f;
        v_tot += v;
        n_tot += n;
    }
    println!(
        "ALL FAMILIES first-wins {f_tot}/{n_tot} ({:.1}%) -> vote {v_tot}/{n_tot} ({:.1}%)",
        100.0 * f_tot as f32 / n_tot as f32,
        100.0 * v_tot as f32 / n_tot as f32
    );
    let _ = first_cost;

    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    println!("recall after the vote {recalled}/{}", facts.len());
    assert_eq!(recalled, facts.len(), "recall must still be 100%");

    // The contract is per family, not on the aggregate: `synonym_span_derivation.rs`
    // measured a change that raised the aggregate while dropping `next_color`,
    // and recorded that as the reason the aggregate is not the thing to gate on.
    for (family, (f, v, n)) in &tally {
        assert!(v >= f, "{family} fell under the vote: {f}/{n} -> {v}/{n}");
    }
    assert!(v_tot > f_tot, "the vote must move something: {f_tot} -> {v_tot}");
}

/// `derive_by_substitution_profiled`'s own loop with BOTH changes in it, so the
/// shipped shape can be compared against the production call on one brain in
/// one run. The deletion search, the `>= 1.0` acceptance, the `max_depth`
/// recursion and the `max_probes` budget are the production ones; the splice
/// winner is the vote, and an empty result falls back to the relation transfer.
fn derive_voted(
    brain: &mut Brain,
    query: &str,
    max_depth: usize,
    max_probes: usize,
) -> (Option<String>, usize) {
    derive_voted_rule(brain, query, max_depth, max_probes, VoteRule::Plurality)
}

/// How a splice scan picks its winner from the rewrites that reach the ceiling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VoteRule {
    /// Production: the first ceiling rewrite in `j` order.
    FirstWins,
    /// The most common answer, ties to the production choice. Measured
    /// on_material 6/24 -> 18/24 at scale 1 with next_color holding 16/16 --
    /// and measured to COST next_color 64/64 -> 53/64 at scale 4, because at
    /// 32 rooms a truncated rewrite's unordered byte set reaches OTHER rooms'
    /// questions at the ceiling and the truncations acquire votes.
    Plurality,
    /// The most common answer only when it is a strict MAJORITY of the ceiling
    /// rewrites; otherwise the production choice. The reasoning is that a
    /// plurality among mutually inconsistent confusions is not agreement, and
    /// `next_color`'s correct rewrite is `j = 0` -- which first-wins already
    /// picks. So this can only override when the rewrites genuinely agree.
    Majority,
}

fn derive_voted_rule(
    brain: &mut Brain,
    query: &str,
    max_depth: usize,
    max_probes: usize,
    rule: VoteRule,
) -> (Option<String>, usize) {
    use std::collections::BTreeMap;
    let mut probes = 0usize;
    let mut current = query.as_bytes().to_vec();
    let mut derived: Option<String> = None;

    for _hop in 0..max_depth {
        if probes >= max_probes {
            break;
        }
        probes += 1;
        let (base_score, base_answer) = ask(brain, &String::from_utf8_lossy(&current));
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
                let (score, answer) = ask(brain, &String::from_utf8_lossy(&sub));
                if score >= 1.0 {
                    if let Some(answer) = answer {
                        known_prefix = Some((k, answer));
                        break 'deletion;
                    }
                }
            }
        }
        let Some((k, spliced)) = known_prefix else { break };

        // Every ceiling rewrite, counted. `first` is the production winner and
        // is kept as the tie-break so the vote is never arbitrary.
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut rewrite_of: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut first: Option<String> = None;
        let mut best_subceiling: Option<(f32, Vec<u8>, String)> = None;
        for j in 0..=k {
            if probes >= max_probes {
                break;
            }
            let mut rw = Vec::with_capacity(n + spliced.len());
            rw.extend_from_slice(&current[..j]);
            rw.extend_from_slice(spliced.as_bytes());
            rw.extend_from_slice(&current[k..]);
            if rw == current || rw.is_empty() {
                continue;
            }
            probes += 1;
            let (score, answer) = ask(brain, &String::from_utf8_lossy(&rw));
            let Some(answer) = answer else { continue };
            if score >= 1.0 {
                if first.is_none() {
                    first = Some(answer.clone());
                }
                *counts.entry(answer.clone()).or_insert(0) += 1;
                rewrite_of.entry(answer).or_insert(rw);
            } else if score > base_score
                && best_subceiling.as_ref().map_or(true, |(b, _, _)| score > *b)
            {
                best_subceiling = Some((score, rw, answer));
            }
        }
        let ceiling: usize = counts.values().sum();
        let modal = counts
            .iter()
            .max_by_key(|(a, n)| (**n, Some(a.as_str()) == first.as_deref()))
            .map(|(a, c)| (a.clone(), *c));
        let winner = match (rule, &modal) {
            (VoteRule::FirstWins, _) => first.clone(),
            (VoteRule::Plurality, _) => modal.as_ref().map(|(a, _)| a.clone()),
            (VoteRule::Majority, Some((a, c))) => {
                if *c * 2 > ceiling {
                    Some(a.clone())
                } else {
                    first.clone()
                }
            }
            (VoteRule::Majority, None) => None,
        };
        let step = match winner {
            Some(a) => match rewrite_of.remove(&a) {
                Some(rw) => Some((rw, a)),
                // `first` can name an answer the map no longer holds only if
                // the maps disagree, which they cannot; kept as a fallthrough
                // rather than a panic so a rule change cannot abort a run.
                None => best_subceiling.take().map(|(_, rw, a)| (rw, a)),
            },
            // No ceiling rewrite: the production fallback, the best improvement
            // on the base score.
            None => best_subceiling.map(|(_, rw, a)| (rw, a)),
        };
        let Some((next_question, next_answer)) = step else { break };
        derived = Some(next_answer);
        current = next_question;
    }
    if derived.is_none() && probes < max_probes {
        let (t, cost) = transfer(brain, query, max_probes - probes);
        probes += cost;
        if t.is_some() {
            return (t, probes);
        }
    }
    (derived, probes)
}

/// THE SHIPPED SHAPE, against the production call, on one brain, at two
/// budgets -- because the vote probes every `j` instead of breaking on the
/// first, and that cost is the reason to report the budget rather than pick
/// one. Production is budgeted at 32 in the scorecard.
#[test]
fn vote_plus_transfer_against_the_production_call_at_two_budgets() {
    use std::collections::BTreeMap;
    let mut brain = subject();
    let facts = teach_world(&mut brain);
    let probes_list = integration_probes();

    // family -> (right, wrong, empty, probes)
    let mut arms: BTreeMap<(&str, &str), (usize, usize, usize, usize)> = BTreeMap::new();
    for (family, q, want) in &probes_list {
        let mut record = |label: &'static str, got: Option<&str>, cost: usize| {
            let slot = arms.entry((*family, label)).or_insert((0, 0, 0, 0));
            slot.3 += cost;
            match got {
                None => slot.2 += 1,
                Some(a) if a == want.as_str() => slot.0 += 1,
                Some(_) => slot.1 += 1,
            }
        };
        let (got, cost) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 32);
        let got = got.map(|a| String::from_utf8_lossy(&a).to_string());
        record("production_32", got.as_deref(), cost);
        for (label, budget) in [("voted_32", 32usize), ("voted_128", 128usize)] {
            let (got, cost) = derive_voted(&mut brain, q, 3, budget);
            record(label, got.as_deref(), cost);
        }
    }

    for label in ["production_32", "voted_32", "voted_128"] {
        let mut r = 0usize;
        let mut n = 0usize;
        let mut c = 0usize;
        for ((family, l), (right, wrong, empty, cost)) in &arms {
            if *l != label {
                continue;
            }
            println!(
                "{label:>14} {family:>17} right {right}/{} wrong {wrong} empty {empty} \
                 probes {cost} ({:.1}/probe)",
                right + wrong + empty,
                *cost as f32 / (right + wrong + empty) as f32
            );
            r += right;
            n += right + wrong + empty;
            c += cost;
        }
        println!(
            "{label:>14} ALL FAMILIES {r}/{n} ({:.1}%) probes {c} ({:.1}/derivation)",
            100.0 * r as f32 / n as f32,
            c as f32 / n as f32
        );
    }

    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    println!("recall after all three arms {recalled}/{}", facts.len());
    assert_eq!(recalled, facts.len(), "recall must still be 100%");

    // Per family against production, at both budgets, RECORDED AND NO LONGER
    // ASSERTED. This used to assert `v.0 >= p.0` -- that the voted arm may not
    // cost correct answers against the production call. The reference point
    // moved: `derive_by_substitution` now admits only rewrites the brain was
    // TAUGHT, which raised production sharply on its own (next_color 75.8 %
    // -> 98.4 % at scale 16, 43.8 % -> 64.6 % at scale 64, with
    // integration_wrong_pct 0.00 at all four scales). An assertion of the form
    // "my arm is at least as good as production" is stale by construction the
    // moment production improves, and it fails in the direction that means the
    // PRODUCT got better -- which is the one direction a test must never red
    // on. Measured 2026-10-01: this test passed on the tree it was written
    // against and failed on the tree carrying the new accept rule, with no
    // change to the arms it measures.
    //
    // Removed rather than re-baselined against a number, because any fixed
    // number here has the same expiry. The census below is the durable half:
    // it prints every arm against production so a future pass can read the
    // comparison without a guard that expires. The absolute contract this
    // test still asserts is recall at 100 %.
    for label in ["voted_32", "voted_128"] {
        for (family, _, _) in &probes_list {
            let p = arms[&(*family, "production_32")];
            let v = arms[&(*family, label)];
            println!(
                "{label:>10} {family:>18} production {}/{} -> arm {}/{}",
                p.0,
                p.0 + p.1 + p.2,
                v.0,
                v.0 + v.1 + v.2
            );
        }
    }
}

/// AT SCALE 4, NO VOTE THRESHOLD SEPARATES THE TWO FAMILIES.
///
/// THE SAME COMPARISON AT THE SCORECARD'S SCALE 4 (32 rooms, 744 facts).
///
/// Every integration family decays along scale and they decay at different
/// rates -- `next_color` is 100 % at scale 1 and 49.8 % at scale 64,
/// `on_material` 25 % and 11.1 % -- so a mechanism measured only at scale 1 is
/// a hypothesis about scale 4. The complement is real and is the reason this
/// test exists: both changes depend on a UNIQUE best match (the transfer on
/// `r000 beside?` being the only trained `beside?`, the vote on truncations
/// being a minority), and 4x the rooms means 4x the trained `beside?` questions
/// and 4x the rooms whose material could coincide. Either could invert.
#[test]
fn at_scale_four_no_vote_threshold_separates_the_two_families() {
    use std::collections::BTreeMap;
    let rooms = ROOMS * 4;
    let mut brain = subject();
    let facts = teach_world_n(&mut brain, rooms);
    println!("scale 4: {rooms} rooms, {} trained facts", facts.len());

    let mut arms: BTreeMap<(&str, &str), (usize, usize, usize, usize)> = BTreeMap::new();
    for (family, q, want) in integration_probes_n(rooms) {
        let mut record = |label: &'static str, got: Option<&str>, cost: usize| {
            let slot = arms.entry((family, label)).or_insert((0, 0, 0, 0));
            slot.3 += cost;
            match got {
                None => slot.2 += 1,
                Some(a) if a == want.as_str() => slot.0 += 1,
                Some(_) => slot.1 += 1,
            }
        };
        let (got, cost) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, 32);
        record("production_32", got.map(|a| String::from_utf8_lossy(&a).to_string()).as_deref(), cost);
        for (label, rule) in
            [("plurality_128", VoteRule::Plurality), ("majority_128", VoteRule::Majority)]
        {
            let (got, cost) = derive_voted_rule(&mut brain, &q, 3, 128, rule);
            record(label, got.as_deref(), cost);
        }
    }

    for label in ["production_32", "plurality_128", "majority_128"] {
        let (mut r, mut n, mut c) = (0usize, 0usize, 0usize);
        for ((family, l), (right, wrong, empty, cost)) in &arms {
            if *l != label {
                continue;
            }
            println!(
                "s4 {label:>14} {family:>17} right {right}/{} wrong {wrong} empty {empty} probes {cost}",
                right + wrong + empty
            );
            r += right;
            n += right + wrong + empty;
            c += cost;
        }
        println!(
            "s4 {label:>14} ALL FAMILIES {r}/{n} ({:.1}%) probes {c} ({:.1}/derivation)",
            100.0 * r as f32 / n as f32,
            c as f32 / n as f32
        );
    }

    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    println!("s4 recall after both arms {recalled}/{}", facts.len());
    assert_eq!(recalled, facts.len(), "recall must be 100% at scale 4 too");

    // THE REFUTATION, AND IT IS WHY THE MAJORITY ARM EXISTS. A bare plurality
    // holds `next_color` 16/16 at scale 1 and COSTS it 64/64 -> 53/64 at scale
    // 4. It is the vote and not the transfer: the transfer fires only on an
    // empty production answer, and production `next_color` has empty 0 here, so
    // it never ran for that family.
    //
    // The cause is the same unordered byte set as everything else in this area.
    // `next_color`'s correct rewrite is `j = 0` -- "r000 next bed color?" ->
    // "r001 bed color?" -- and the truncations are "rr001 bed color?",
    // "r0r001 bed color?" and so on. At 8 rooms those reach no trained question
    // but the right one, so the vote was unanimous and 16/16 read as safety. At
    // 32 rooms their distinct byte set {r,0,1,sp,b,e,d,c,o,l,?} ALSO reaches
    // "r010 bed color?" and "r011 bed color?" at the ceiling, so truncations
    // acquire votes and can outvote `j = 0`. A vote over mutually inconsistent
    // confusions degrades exactly as confusability grows, which is scale.
    //
    // Both regressions are pinned as assertions. A plurality that stops costing
    // next_color at scale 4 means the matcher changed, and this is the line that
    // will say so.
    let nc_p = arms[&("next_color", "production_32")];
    let nc_pl = arms[&("next_color", "plurality_128")];
    assert!(
        nc_pl.0 < nc_p.0,
        "the plurality must still cost next_color at scale 4 -- that regression is this          test's finding, measured 64/64 -> 53/64: {nc_p:?} -> {nc_pl:?}"
    );

    // AND THE MAJORITY RULE DOES NOT RECOVER IT EITHER, WHICH IS THE RESULT.
    //
    // This test asserted that no family may fall under the majority rule,
    // expecting it to pass -- the reasoning being that a ceiling set with no
    // strict majority falls back to first-wins, which for `next_color` IS
    // `j = 0` and IS correct, so the only way it can still lose is a WRONG
    // answer holding a strict majority. Measured: that happens 7 times of 64.
    // At 32 rooms the truncated rewrites stop being mutually inconsistent and
    // AGREE on one wrong room, because "rr001 bed color?" and "r0r001 bed
    // color?" carry the same distinct byte set and it reaches "r010 bed color?"
    // and "r011 bed color?" at the ceiling. More scale means more agreement
    // AMONG the confusions.
    //
    // So every threshold between "any plurality" and "unanimous" sits between
    // on_material's 5-of-7 and next_color's 1-of-many, and the majority rule is
    // DOMINATED: it recovers 4 of next_color's 11 by giving back 10 of
    // on_material and 5 of next_on_material, for 11 points of aggregate. A
    // guard that cannot hold the family is only a worse exchange rate.
    //
    // The discriminator the problem actually needs is "was this rewrite ever
    // TAUGHT", and the matcher cannot answer it: 184 ceiling rewrites over 24
    // on_material probes of which 24 were taught, and 0 of 186 trained
    // questions match at `MatchTier::Concept`, so every match here is an
    // unordered distinct byte set. No selection policy recovers information the
    // representation does not carry.
    // RETIRED 2026-10-01, and the reason is the point. Two assertions used to
    // live here: that the majority rule must still COST next_color, and that
    // it must stay DOMINATED in aggregate. Both were true when measured in
    // pass 14 -- next_color (57, 7) under production against (53, 11) under
    // the plurality and (57, 7) under majority, with majority 11 aggregate
    // points behind -- and both stopped a vote threshold shipping, which was
    // the right call then.
    //
    // They assert about a world that no longer exists. A vote over ceiling
    // rewrites was a MITIGATION for an accept rule that admitted any rewrite
    // scoring 1.0; the rule now requires the rewrite to be a question the
    // brain was TAUGHT, so the ceiling set contains only taught questions and
    // there is almost nothing left for a vote to arbitrate. Measured through
    // the product after that change: next_color 75.8 % -> 98.4 % at scale 16
    // and 43.8 % -> 64.6 % at scale 64, with integration_wrong_pct 0.00 at
    // all four scales. The first assertion then fails on EQUALITY --
    // (57, 7, 0, 662) -> (57, 7, 0, 1920), the two arms agreeing -- which is
    // not the finding being refuted, it is the finding becoming unmeasurable.
    //
    // Deleted rather than relaxed. An assertion rewritten to match what the
    // code now does is a not-done dressed as a done; the measurement itself
    // is still printed above, so a future pass that revives a vote threshold
    // has the pass-14 numbers to compare against and no stale guard to argue
    // with. The per-family contract below is what this file still asserts.
    let nc_mj = arms[&("next_color", "majority_128")];
    let agg = |label: &str| -> usize {
        arms.iter().filter(|((_, l), _)| *l == label).map(|(_, v)| v.0).sum()
    };
    eprintln!(
        "s4 vote-threshold census (retired assertions): next_color production {nc_p:?}          plurality {nc_p:?} majority {nc_mj:?}; aggregate plurality {} majority {}",
        agg("plurality_128"),
        agg("majority_128")
    );

    // RETIRED 2026-10-01, the same way and for the same reason as 70bba62:
    // "the plurality must beat production" per family and in aggregate went
    // red the moment PRODUCTION improved -- the exact-ordered accept rule took
    // on_material to 94/96 with 0 wrong, above the vote's 74/96 -- and a test
    // must never red because the product got better. The vote arm also
    // answers WRONG (22/96 on_material here), and the owner's rule is that the
    // brain never hallucinates, so no vote arm can ship anyway. The census is
    // kept as output for any future pass that revisits voting.
    for family in ["on_material", "next_on_material", "beside_next"] {
        let p = arms[&(family, "production_32")];
        let v = arms[&(family, "plurality_128")];
        eprintln!("s4 census {family}: production {p:?} plurality {v:?}");
    }
    eprintln!(
        "s4 census aggregate: production {} plurality {}",
        agg("production_32"),
        agg("plurality_128")
    );
}

/// All four families, with the transfer wired the way it would actually ship:
/// as a FALLBACK after `derive_by_substitution_profiled` returns nothing. That
/// composition is the only one that cannot cost a family, and "cannot cost a
/// family" is the thing `synonym_span_derivation.rs` measured a one-line splice
/// reorder failing (`next_color` 16/16 -> 14/16 for `on_material` 6 -> 11).
#[test]
fn as_a_fallback_the_transfer_cannot_cost_a_family() {
    use std::collections::BTreeMap;
    let mut brain = subject();
    let facts = teach_world(&mut brain);
    // The scorecard's own budget for the production half.
    const MAX_PROBES: usize = 32;

    // family -> (right, wrong, empty, probes)
    let mut base: BTreeMap<&str, (usize, usize, usize, usize)> = BTreeMap::new();
    let mut with_fb: BTreeMap<&str, (usize, usize, usize, usize)> = BTreeMap::new();
    for (family, q, want) in integration_probes() {
        let (got, probes) = brain.derive_by_substitution_profiled(
            QUERY_POOL,
            ANSWER_POOL,
            q.as_bytes(),
            3,
            MAX_PROBES,
        );
        let got = got.map(|a| String::from_utf8_lossy(&a).to_string());
        let tally = |slot: &mut (usize, usize, usize, usize), got: Option<&str>, cost: usize| {
            slot.3 += cost;
            match got {
                None => slot.2 += 1,
                Some(a) if a == want.as_str() => slot.0 += 1,
                Some(_) => slot.1 += 1,
            }
        };
        tally(base.entry(family).or_insert((0, 0, 0, 0)), got.as_deref(), probes);

        // The fallback fires ONLY on an empty production answer, so every probe
        // the production path already answers is byte-identical to the baseline
        // and costs nothing extra.
        let (fb_got, fb_cost) = match got {
            Some(a) => (Some(a), 0),
            None => transfer(&mut brain, &q, MAX_PROBES),
        };
        tally(with_fb.entry(family).or_insert((0, 0, 0, 0)), fb_got.as_deref(), probes + fb_cost);
    }

    let mut moved = 0usize;
    for (family, (r0, w0, e0, c0)) in &base {
        let (r1, w1, e1, c1) = with_fb[family];
        println!(
            "{family:>17} base right {r0} wrong {w0} empty {e0} probes {c0} \
             -> fallback right {r1} wrong {w1} empty {e1} probes {c1}"
        );
        // THE CONTRACT: no family may lose a correct answer. A fallback that
        // only fires on an empty production answer cannot, and this is the
        // assertion that keeps it that way if the composition is ever changed.
        assert!(
            r1 >= *r0,
            "{family} lost correct answers: {r0} -> {r1}, which is what makes a change unshippable"
        );
        if r1 > *r0 {
            moved += 1;
        }
    }
    let total = |m: &BTreeMap<&str, (usize, usize, usize, usize)>| -> (usize, usize, usize) {
        m.values().fold((0, 0, 0), |(r, n, c), (right, wrong, empty, cost)| {
            (r + right, n + right + wrong + empty, c + cost)
        })
    };
    let (r0, n0, c0) = total(&base);
    let (r1, n1, c1) = total(&with_fb);
    assert_eq!(n0, n1, "both arms must be asked the same questions");
    // THE QUANTITY THIS SUITE USED TO REPORT WAS `right` ALONE, AND THAT IS HOW
    // A NET-ZERO MECHANISM GOT PRIORITISED AS "+12.9 POINTS".
    //
    // Every assertion in this file was satisfied -- both arms asked the same
    // questions, no family lost a correct answer, recall stayed 186/186 -- while
    // the fallback turned EVERY silence into an answer (`empty` 6 -> 0, 4 -> 0,
    // 2 -> 0) and five of the twelve new answers were wrong. An assertion on
    // `right` is structurally blind to a silence converted into an invention,
    // because invention raises `right` and `wrong` together.
    //
    // The project gates integration on NET, correct minus wrong, precisely
    // because invention is worse than silence. So this suite reports and gates
    // the same quantity, and the headline percentage is net from here on.
    let wrong_total = |m: &BTreeMap<&str, (usize, usize, usize, usize)>| -> usize {
        m.values().map(|(_, wrong, _, _)| *wrong).sum()
    };
    let (w0, w1) = (wrong_total(&base), wrong_total(&with_fb));
    let (net0, net1) = (r0 as i64 - w0 as i64, r1 as i64 - w1 as i64);
    println!(
        "ALL FAMILIES base {r0}/{n0} ({:.1}%) wrong {w0} NET {net0} probes {c0} \
         -> fallback {r1}/{n1} ({:.1}%) wrong {w1} NET {net1} probes {c1} \
         (+{:.1}% probes); families improved {moved}",
        100.0 * net0 as f32 / n0 as f32,
        100.0 * net1 as f32 / n1 as f32,
        100.0 * (c1 as f32 - c0 as f32) / c0 as f32
    );
    // A mechanism whose NET does not improve is not a gain however many `right`
    // it adds. This is the assertion whose absence cost pass 18 its whole arc.
    assert!(
        net1 >= net0,
        "the fallback lowered NET integration (correct - wrong): {net0} -> {net1}; \
         right {r0} -> {r1} and wrong {w0} -> {w1}"
    );

    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    println!("recall after both arms {recalled}/{}", facts.len());
    assert_eq!(recalled, facts.len(), "recall must still be 100%");
    // DELIBERATELY NOT `r1 > r0`. The production call is another agent's file
    // and the transfer is being wired INTO it, at which point the base arm
    // already carries the gain and a strict-improvement assertion turns this
    // file red for the change it exists to argue for. The claim that survives
    // either way is that the family is answered by SOMETHING here, and the
    // per-family `r1 >= r0` above is what forbids a regression.
    let bn = with_fb["beside_next"];
    println!("beside_next under the fallback arm: right {} of {}", bn.0, bn.0 + bn.1 + bn.2);
    assert!(
        bn.0 > 0,
        "beside_next must be answered by the production call or by the fallback: {bn:?}"
    );
    assert!(r1 >= r0, "the fallback must not lower the aggregate: {r0} -> {r1}");
}

/// A pool at `PoolConfig::defaults`'s own `max_concept_member_count` of 8,
/// which is the value the harness overrides to 64 and the one the ledger
/// pricing says is affordable.
fn subject_m8() -> Brain {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);
    for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
        let mut pc = PoolConfig::defaults(name, id);
        pc.recent_atoms_window = 2048;
        pc.concept_emergence_threshold = 2;
        pc.decay_rate = 0.0001;
        pc.prune_floor = 0.005;
        // deliberately NOT overridden: PoolConfig::defaults sets 8.
        brain.create_pool(pc, Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>);
    }
    brain
}

fn ask_tiered(brain: &mut Brain, question: &str) -> (f32, MatchTier, Option<String>) {
    brain.observe_fabric_read_only(QUERY_POOL, question.as_bytes());
    let m = brain.best_binding_match_v2(QUERY_POOL);
    let answer = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .filter(|a| !a.is_empty())
        .map(|a| String::from_utf8_lossy(&a).to_string());
    (m.score(), m.tier, answer)
}

/// THE ONE FACT THAT DECIDES WHETHER A SPLIT CEILING CAN EVER BE JUDGED.
///
/// Every signal a split ceiling could be decided on is measured flat: the score
/// is exactly 1.0 for each member by construction, j-position is right for one
/// family and wrong for the other, a byte-weighted vote is largest-j in
/// disguise, and the reverse-decode membership test accepts nothing (tp 0, fp 0,
/// fn 24, tn 160). The only order-sensitive signal the matcher has is
/// `MatchTier::Concept`, because a concept neuron emerges from a recurring byte
/// SEQUENCE -- and it was 0 of 186 on trained questions, because emergence had
/// no live caller on the train path.
///
/// This measures whether the tier separates a TAUGHT ceiling rewrite from its
/// byte-set twin. The complement was live and cheap to state: concepts may
/// emerge and still not be admitted by the match's coverage gate, in which case
/// the tier stays 0 and the change has no derivation value. `total_concepts`
/// against `total_binding` separates those two, so both are printed.
///
/// # MEASURED BOTH WAYS, AND IT IS THE COMPLEMENT: A MATCHER PROBLEM
///
/// At HEAD this prints `total_concepts 186 == total_binding 186`,
/// `total_neurons 242`, `total_terminals 5638`, concept tier 0/186, and the 2x2
/// `tp 0 fp 0 fn 24 tn 160`. Emergence has no live caller on the train path:
/// `Pool::ensure_frame_atoms_for_pretrain_profiled` (pool.rs:3300) was nothing
/// but `ensure_atom` per label, and answering is deliberately forbidden to
/// emerge (`emergence_suppressed`).
///
/// Running it again with six lines added to that method -- `push_recent` plus
/// `check_concept_emergence` over the frame's sequence, with
/// `max_concept_member_count` left at `PoolConfig::defaults`'s own 8 -- gives:
///
/// ```text
///   total_neurons 5473  total_concepts 5417  total_binding 186  total_terminals 49243
///   recall 186/186 ; trained questions at concept tier 0/186
///   CONCEPT TIER over 184 ceiling rewrites: tp 0  fp 0  fn 24  tn 160
/// ```
///
/// So **5,231 non-binding concepts emerge where there had never been one,
/// recall is untouched, and the matcher admits exactly zero of them.** That
/// settles the branch `synonym_span_derivation.rs` wrote its assertion for -- a
/// tier of zero beside a concept count of zero is an emergence problem, beside
/// a non-zero count it is a MATCHER problem -- and it is a matcher problem.
///
/// Three consequences, because they are what the next change depends on:
///
/// * No split-ceiling discriminator is reachable from the splice loop at any
///   vote threshold. `MatchTier::Concept` is the only order-sensitive signal the
///   matcher has and it stays unreached with 5,417 concepts in the pool.
/// * The next change is `best_binding_match_v2`'s coverage gate, not emergence
///   and not the derivation.
/// * Emergence on the train path is cheap and safe AT THE RIGHT BOUND, which
///   reverses this suite's earlier claim that it was unshippable. The ledger
///   keys are byte RUNS, so their count is bounded by the alphabet: 62,421 keys
///   / 5.0 MB at scale 64 growing 24x for 64x the facts at the defaults value
///   of 8, against 11,009,761 keys / 3347 MB growing 65x at the harness value
///   of 64.
///
/// The pool.rs change is NOT in the tree. It takes neurons 242 -> 5473 and
/// terminals 5638 -> 49243 for a derivation gain of zero today, so "RAM never
/// rises and nothing fell" is not satisfied until the coverage gate moves. It
/// belongs in the same commit as that gate change. This test is valid at HEAD
/// and prints the HEAD reading, so the comparison costs no rebuild.
#[test]
fn does_the_concept_tier_separate_a_taught_rewrite_from_its_byte_set_twin() {
    let mut brain = subject_m8();
    let facts = teach_world(&mut brain);

    let st = brain.stats();
    println!(
        "fabric at max_concept_member_count 8: total_neurons {} total_concepts {}          total_binding {} total_terminals {}",
        st.total_neurons, st.total_concepts, st.total_binding, st.total_terminals
    );
    let mut recalled = 0usize;
    let mut trained_concept = 0usize;
    for (q, a) in &facts {
        let (_, tier, got) = ask_tiered(&mut brain, q);
        if got.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
        if tier == MatchTier::Concept {
            trained_concept += 1;
        }
    }
    println!(
        "recall {recalled}/{} ; trained questions at concept tier {trained_concept}/{}",
        facts.len(),
        facts.len()
    );
    // RECALL IS THE GATE ON EVERYTHING. An emergence change that costs recall is
    // not a discriminator, it is a regression.
    assert_eq!(recalled, facts.len(), "recall must stay 100% with emergence on the train path");

    // Now the 2x2 over `on_material`'s ceiling rewrites, exactly as the dead
    // reverse-decode detector was measured.
    let (mut tp, mut fp, mut fn_, mut tn) = (0usize, 0usize, 0usize, 0usize);
    let mut perfect_total = 0usize;
    for r in 0..ROOMS {
        for (obj, base) in RESTS_ON {
            let q = format!("{} {obj} on material?", room(r));
            let qb = q.as_bytes().to_vec();
            let n = qb.len();
            let mut prefix: Option<(usize, String)> = None;
            'del: for k in 1..n {
                for t in 0..=1usize.min(n - k) {
                    if t > 0 && n - t == k {
                        continue;
                    }
                    let mut sub = Vec::with_capacity(k + t);
                    sub.extend_from_slice(&qb[..k]);
                    sub.extend_from_slice(&qb[n - t..]);
                    let (score, _, answer) = ask_tiered(&mut brain, &String::from_utf8_lossy(&sub));
                    if score >= 1.0 {
                        if let Some(answer) = answer {
                            prefix = Some((k, answer));
                            break 'del;
                        }
                    }
                }
            }
            let Some((k, spliced)) = prefix else { continue };
            for j in 0..=k {
                let mut rw = Vec::with_capacity(n);
                rw.extend_from_slice(&qb[..j]);
                rw.extend_from_slice(spliced.as_bytes());
                rw.extend_from_slice(&qb[k..]);
                if rw == qb || rw.is_empty() {
                    continue;
                }
                let text = String::from_utf8_lossy(&rw).to_string();
                let (score, tier, answer) = ask_tiered(&mut brain, &text);
                if score < 1.0 || answer.is_none() {
                    continue;
                }
                perfect_total += 1;
                let taught = facts.iter().any(|(fq, _)| *fq == text);
                match (tier == MatchTier::Concept, taught) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => tn += 1,
                }
            }
        }
    }
    println!(
        "CONCEPT TIER as a taught-detector over {perfect_total} ceiling rewrites:          true-pos {tp} false-pos {fp} false-neg {fn_} true-neg {tn}"
    );
    assert!(perfect_total > 0, "there must be ceiling rewrites to census, or this is vacuous");
    // Printed, not asserted either way. The number IS the finding and the two
    // outcomes need opposite follow-ups: tp > 0 with fp == 0 means the
    // discriminator exists and the derivation should prefer concept-tier
    // rewrites; tp == 0 means emergence changed the fabric without changing
    // what the matcher admits, and the next change is the coverage gate rather
    // than the splice loop.
}
