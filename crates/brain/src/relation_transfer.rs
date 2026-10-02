//! SUBJECT-PRESERVING RELATION TRANSFER -- the derivation step that answers the
//! one integration family that is 0.0 % at EVERY scale.
//!
//! # Why this exists as a module rather than as a test helper
//!
//! The mechanism was designed and measured in
//! `tests/relation_transfer_derivation.rs`, where it is a private `fn` with no
//! caller outside that file -- so it shipped no value. Every call it makes is
//! already public (`observe_fabric_read_only`, `best_binding_match_v2`,
//! `decode_best_trained_binding`), which is why promoting it is a MOVE and not
//! a port: nothing here reaches into `Brain`'s internals, and the function
//! takes its two pool ids as arguments so it is not tied to the scorecard's
//! world.
//!
//! # The measurement that makes it the next change
//!
//! At the scale-1 symbol world, composed as a FALLBACK behind the existing
//! `derive_by_substitution` (fires only when that returns nothing):
//!
//! ```text
//!   ALL FAMILIES base 42/54 (77.8%) probes 1083
//!             -> fallback 49/54 (90.7%) probes 1131   (+4.4% probes)
//!   beside_next under the fallback arm: right 6 of 6
//!   recall after both arms 186/186
//! ```
//!
//! Priced against the post-`dea50bd` four-scale baseline, which solved
//! `on_material` (1534/1536) and `next_color` (1024/1024) outright and left
//! `beside_next` byte-identical at 0/384, that family is **384 of the 785
//! misses remaining at scale 64 -- 48.9 %** and the only bucket with a
//! mechanism already measured to answer it.
//!
//! # What the existing derivation cannot do, stated without naming a word
//!
//! Every match in this world is precision x recall over the UNORDERED DISTINCT
//! BYTE SET. For a held-out probe of the shape `<subject> <relation>?`, the
//! relation's bytes outnumber the subject's: modelling the matcher's own scorer
//! over 186 trained questions gives the same reading for all six held-out
//! probes -- the top match is UNIQUE at 0.9000 and is the WRONG SUBJECT, while
//! the trained question that holds the answer sits at rank 44. So the relation
//! outweighs the subject 9:1, and any mechanism that reaches the relation drags
//! the wrong subject along with it. That is neither a tie-break problem nor a
//! budget problem, and no reordering or wider budget supplies the missing span:
//! `derive_by_substitution` can only insert spans taken from an ANSWER, and the
//! span needed here only ever appears inside a trained QUESTION.
//!
//! # The mechanism
//!
//! 1. A query scoring below the ceiling has a best trained answer `A`.
//! 2. REVERSE-decode `A` -- observe it in the answer pool, decode into the query
//!    pool -- giving a trained question `T` that produces `A`. `T` names the
//!    neighbourhood the query landed in, including its subject.
//! 3. Align query and `T` on their common prefix and suffix, giving the span
//!    that differs.
//! 4. Splice spans of `T` over spans of the query. A rewrite scoring at the
//!    ceiling IS a question the brain was taught, so a wrong span cannot win --
//!    it can only cost a probe.
//!
//! # ALIGNMENT ALONE CANNOT SEPARATE THE SUBJECT FROM THE RELATION
//!
//! This was asserted the other way round first and the unit test refuted it, so
//! it is recorded here rather than rediscovered. For query `"r001 beside?"` and
//! trained question `"r000 next?"` the common prefix is 3 bytes and the common
//! suffix is 1, so the aligned substitution is
//! `"r00" ++ "0 next" ++ "?"` = **`"r000 next?"`** -- the WRONG SUBJECT. The
//! borrowed span straddles the subject byte and the relation, because the two
//! differences are adjacent and the alignment has no way to cut between them.
//!
//! What actually preserves the subject is the TAIL transfer: `query[..a] ++
//! T[i..]`, which cuts at the subject/relation boundary instead of at the common
//! prefix. At `a == i == 4` that is `"r001" ++ " next?"` = `"r001 next?"`, the
//! wanted rewrite, and the `|a - i|` ordering puts it at candidate 5 of 4,395.
//! So the aligned candidate is kept because it is cheap and sometimes right, not
//! because it is the one the mechanism relies on.
//!
//! Step 4's accept rule is the one `derive_by_substitution` already uses. The
//! only new idea is that the INSERT may come from a trained QUESTION rather
//! than only from an ANSWER.
//!
//! # HALLUCINATION, AND THE RULE THAT CEILING-SCORING ALONE IS NOT ENOUGH
//!
//! The first version of this module returned the FIRST rewrite that scored at
//! the ceiling and resolved to something new. That is not safe, and the number
//! says so rather than an argument. Composed as a fallback over the scale-1
//! world, counting `wrong` as well as `right` -- which the design suite never
//! did, because its only assertion was `right` must not fall:
//!
//! ```text
//!        beside_next  base 0 right  0 wrong  6 empty -> 6 right  0 wrong  0 empty
//!   next_on_material  base 4 right  0 wrong  4 empty -> 4 right  4 WRONG  0 empty
//!        on_material  base 22 right 0 wrong  2 empty -> 23 right 1 WRONG  0 empty
//! ```
//!
//! Every silence became an answer: `empty` went to 0 in all three. So the
//! headline "42/54 -> 49/54" was +7 right bought with **+5 wrong**, a NET of
//! +2 against a brain whose measured `wrong` is 0 at every scale -- and the
//! project's first standard is that a brain with no grounded answer has NO
//! answer.
//!
//! The cause is structural, not a threshold: a rewrite can be an exactly
//! trained question (score 1.0) and still return the answer to a DIFFERENT
//! question than the one asked. For a one-hop family the single reachable
//! rewrite is the right one; for a multi-hop family several distinct trained
//! questions are reachable at the ceiling and the first one found is arbitrary.
//!
//! So the accept rule is the owner's: an answer is returned only when the chain
//! is **UNIQUE**. Every distinct ceiling-reachable answer inside the budget is
//! collected; exactly one is returned, and two or more abstains immediately --
//! a second distinct answer settles it, because no further probe can make a
//! non-unique chain unique. This is what makes the mechanism silent where it
//! used to invent, and it costs the multi-hop families nothing they had.

use crate::brain::Brain;
use crate::neuron::PoolId;
use std::collections::HashSet;

/// A rewrite is accepted only at the exact ceiling, which is the score of a
/// question the brain was actually taught. Shared with `derive_by_substitution`
/// by value rather than by import so this module depends on nothing private.
const CEILING: f32 = 1.0;

/// Longest common prefix and suffix of two byte strings, as byte counts. What
/// lies between them is what differs, and that is the substitution.
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

/// Score and answer of a question, through the same calls the existing
/// derivation makes. Read-only on the fabric: it must not train.
fn ask(
    brain: &mut Brain,
    query_pool: PoolId,
    answer_pool: PoolId,
    question: &[u8],
) -> (f32, Option<Vec<u8>>) {
    brain.observe_fabric_read_only(query_pool, question);
    let score = brain.best_binding_match_v2(query_pool).score();
    let answer = brain
        .decode_best_trained_binding(query_pool, answer_pool)
        .filter(|a| !a.is_empty());
    (score, answer)
}

/// REVERSE decode: observe an ANSWER and read back a trained QUESTION that
/// produces it. This is the one call the transfer needs that the existing
/// derivation never makes. `decode_best_trained_binding` refuses
/// `query_pool == target_pool`, so the pools are genuinely swapped here.
fn reverse(brain: &mut Brain, query_pool: PoolId, answer_pool: PoolId, answer: &[u8]) -> Option<Vec<u8>> {
    brain.observe_fabric_read_only(answer_pool, answer);
    brain
        .decode_best_trained_binding(answer_pool, query_pool)
        .filter(|q| !q.is_empty())
}

/// The ordered candidate rewrites, cheapest-to-be-right first. Split out from
/// the probing loop so the ORDER -- which is the entire cost of the mechanism --
/// can be tested without a brain.
///
/// Ordering is not cosmetic. Sorting the tail transfers by insert LENGTH
/// measured 50.0 probes per derivation; sorting by `|a - i|`, the pairs whose
/// borrowed tail starts where the query's own tail starts, puts the correct
/// rewrite among the first few, because two questions about the same KIND of
/// thing carry their relation at nearly the same offset.
pub fn candidate_rewrites(query: &[u8], trained: &[u8]) -> Vec<Vec<u8>> {
    let q = query;
    let tb = trained;
    let mut candidates: Vec<Vec<u8>> = Vec::new();

    // The aligned substitution first: it is the one the alignment argues for.
    let (p, s) = align(q, tb);
    if p + s < q.len() && p + s < tb.len() {
        let mut first = Vec::with_capacity(tb.len());
        first.extend_from_slice(&q[..p]);
        first.extend_from_slice(&tb[p..tb.len() - s]);
        first.extend_from_slice(&q[q.len() - s..]);
        candidates.push(first);
    }

    // Then tail transfers, ordered by alignment rather than by length: keep `a`
    // bytes of what was asked, borrow the relation from `i`. Ties go to the
    // longer borrow, which replaces more of the relation and less of the
    // subject.
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

    // Only then every contiguous span of `trained` over every contiguous span of
    // the query, longest insert first. This is the exhaustive fallback, and the
    // accept-at-ceiling rule is what makes an exhaustive search safe: a wrong
    // span cannot win, it can only cost a probe.
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
    candidates
}

/// Derive an answer by SUBJECT-PRESERVING RELATION TRANSFER, reporting the
/// number of fabric probes spent so its cost is readable the way
/// `derive_by_substitution_profiled` reports its own.
///
/// Intended composition is a FALLBACK: call it only when the existing
/// derivation returned nothing. Composed that way it cannot cost a family a
/// correct answer, because every probe the production path already answers
/// never reaches this function -- which is the property
/// `as_a_fallback_the_transfer_cannot_cost_a_family` asserts per family.
///
/// Returns `None` rather than a guess whenever no rewrite reaches the ceiling.
pub fn derive_by_relation_transfer(
    brain: &mut Brain,
    query_pool: PoolId,
    answer_pool: PoolId,
    query: &[u8],
    max_probes: usize,
) -> (Option<Vec<u8>>, usize) {
    let mut probes = 1usize;
    let (base_score, base_answer) = ask(brain, query_pool, answer_pool, query);
    // At the ceiling this is RECALL, not integration: hand back what was taught.
    if base_score >= CEILING {
        return (base_answer, probes);
    }
    let Some(base_answer) = base_answer else { return (None, probes) };

    // Name the neighbourhood the query landed in: a trained question that
    // produces the answer the query was about to return.
    probes += 1;
    let Some(trained) = reverse(brain, query_pool, answer_pool, &base_answer) else {
        return (None, probes);
    };

    // NARROWING THE CANDIDATE SET MAKES THE UNIQUENESS TEST WEAKER, NOT
    // STRONGER, and it was measured the wrong way round first. Filtering to the
    // rewrites that begin with the query's bytes up to its first differing byte
    // -- enforcing the "subject-preserving" the mechanism is NAMED for -- took
    // `next_on_material` from 2 wrong to **4 wrong**, because the candidates it
    // removed were the ones producing a SECOND distinct answer, and a second
    // distinct answer is what triggers the abstain. Uniqueness is evidence of
    // absence; pruning the search destroys the evidence. So the search is
    // deliberately left wide, and the only gate is the one below.
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    // EVERY distinct answer reachable at the ceiling inside the budget, not the
    // first one. See `UNIQUENESS` below for why the first one is not safe.
    let mut answers: Vec<Vec<u8>> = Vec::new();
    for rewrite in candidate_rewrites(query, &trained) {
        if probes >= max_probes {
            break;
        }
        if !seen.insert(rewrite.clone()) {
            continue;
        }
        probes += 1;
        let (score, answer) = ask(brain, query_pool, answer_pool, &rewrite);
        if score < CEILING {
            continue;
        }
        if let Some(answer) = answer {
            // The rewrite must have asked something DIFFERENT from what the
            // query already resolved to, or the transfer has only rediscovered
            // the wrong-subject neighbour it started from.
            if answer != base_answer && !answers.contains(&answer) {
                answers.push(answer);
                // Two distinct answers already settle it: the chain is not
                // unique, so no further probe can make it unique.
                if answers.len() > 1 {
                    return (None, probes);
                }
            }
        }
    }
    // UNIQUENESS. Exactly one distinct ceiling-reachable answer is returned;
    // anything else abstains.
    if answers.len() == 1 {
        return (Some(answers.remove(0)), probes);
    }
    (None, probes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `align` is the only arithmetic here that can be wrong silently, and the
    /// subtraction in it underflows on a shared prefix that consumes the
    /// shorter string unless `max_s` is clamped the way it is.
    #[test]
    fn align_finds_the_differing_span_and_does_not_underflow() {
        assert_eq!(align(b"r001 beside?", b"r000 beside?"), (3, 8));
        // One string is a prefix of the other: the suffix must be 0, not a
        // wrapped usize.
        assert_eq!(align(b"r001", b"r001 beside?"), (4, 0));
        assert_eq!(align(b"", b"anything"), (0, 0));
        assert_eq!(align(b"abc", b"xyz"), (0, 0));
    }

    /// THE ORDER IS THE COST, so it is asserted rather than described -- and
    /// WHICH CANDIDATE PATH SUPPLIES THE ANSWER is asserted with it, because
    /// getting that backwards is what the first version of this test did.
    ///
    /// The aligned substitution goes first and is the WRONG SUBJECT here: the
    /// subject byte and the relation are adjacent, so `T`'s differing span
    /// carries both and the alignment cannot cut between them. The tail transfer
    /// is what preserves the subject, and the `|a - i|` ordering is what makes
    /// it affordable.
    #[test]
    fn the_wanted_rewrite_comes_from_the_tail_transfer_and_arrives_inside_the_budget() {
        let candidates = candidate_rewrites(b"r001 beside?", b"r000 next?");
        assert_eq!(
            candidates.first().map(|c| String::from_utf8_lossy(c).to_string()),
            Some("r000 next?".to_string()),
            "the aligned substitution is first, and on adjacent differences it borrows the wrong subject"
        );
        let position = candidates
            .iter()
            .position(|c| c.as_slice() == b"r001 next?")
            .expect("the wanted rewrite is among the candidates");
        println!(
            "aligned candidate {:?}; wanted rewrite at {position} of {}",
            String::from_utf8_lossy(&candidates[0]),
            candidates.len()
        );
        // 32 is the scorecard's probe budget, and two probes are already spent
        // before the first rewrite is asked, so the reachable window is 30.
        assert!(
            position < 30,
            "the wanted rewrite must land inside the 32-probe budget, got position {position}"
        );
        // AND IT MUST BEAT THE EXHAUSTIVE SPAN SEARCH. If the ordering ever
        // regresses to length-first the position moves into the hundreds, which
        // is the 50.0-probes-per-derivation reading the ordering replaced.
        assert!(
            position < 16,
            "the |a - i| ordering is the mechanism's whole cost argument: {position}"
        );
    }

    /// A candidate equal to the query itself buys nothing and costs a probe.
    #[test]
    fn no_candidate_is_the_query_itself_or_empty() {
        for c in candidate_rewrites(b"r001 beside?", b"r000 next?") {
            assert_ne!(c.as_slice(), b"r001 beside?", "re-asking the query is not a rewrite");
            assert!(!c.is_empty(), "an empty rewrite cannot be asked");
        }
    }
}
