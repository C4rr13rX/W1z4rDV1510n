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

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

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
    let mut out = Vec::new();
    for r in 0..ROOMS {
        let rm = room(r);
        let nr = (r + 1) % ROOMS;
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
    println!(
        "ALL FAMILIES base {r0}/{n0} ({:.1}%) probes {c0} -> fallback {r1}/{n1} ({:.1}%) probes {c1} \
         (+{:.1}% probes); families improved {moved}",
        100.0 * r0 as f32 / n0 as f32,
        100.0 * r1 as f32 / n1 as f32,
        100.0 * (c1 as f32 - c0 as f32) / c0 as f32
    );

    let mut recalled = 0usize;
    for (q, a) in &facts {
        if ask(&mut brain, q).1.as_deref() == Some(a.as_str()) {
            recalled += 1;
        }
    }
    println!("recall after both arms {recalled}/{}", facts.len());
    assert_eq!(recalled, facts.len(), "recall must still be 100%");
    assert!(r1 > r0, "the fallback must raise the aggregate: {r0} -> {r1}");
}
