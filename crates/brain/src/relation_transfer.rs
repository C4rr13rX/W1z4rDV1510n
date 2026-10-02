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
    // AT THE CEILING THIS IS RECALL **ONLY IF THE QUERY WAS TAUGHT**, and the
    // score cannot establish that. `best_binding_match_v2` is precision x
    // recall over the UNORDERED DISTINCT BYTE SET, so a held-out question
    // sharing its distinct bytes with a trained one reaches 1.0 -- that is the
    // premise `tests/derivation_rejects_untaught.rs` pins, where
    // `"r0desk material?"` hits the ceiling and answers a different room.
    //
    // This arm returned that answer with NO accept rule applied: uniqueness,
    // subject preservation and ordered trained-frame identity all live below
    // it. So the shortcut is gated on the ordered identity of the QUERY
    // itself, which is the only thing that can establish "this was taught".
    //
    // WHAT THAT GATE ACTUALLY BOUGHT, because the hypothesis it was built on
    // was half wrong and the number says which half. It was added expecting
    // the shortcut to be the source of ALL four inventions -- that would have
    // explained why three narrowings below measured inert. Measured at scales
    // 1 and 4 through `tests/integration_family_counts.rs`:
    //
    // ```text
    //   s4 beside_next        4/24 WRONG 3 silent 17  ->  4/24 WRONG 0 silent 20
    //   s4 next_on_material   8/32 WRONG 1            ->  8/32 WRONG 1
    //   s1 on_material       22/24 WRONG 2            -> 22/24 WRONG 2
    //   s1 next_on_material   2/8  WRONG 2            ->  2/8  WRONG 2
    //   s4 invented 4 -> 1;  s1 invented 4 -> 4
    // ```
    //
    // So it removes every invention in the ONE-HOP family while keeping all
    // four of its correct answers -- at scale 4 `beside_next` is now 4 right,
    // 0 wrong, 20 silent, and the "s32: 4 right 3 WRONG" reading recorded in
    // `README.md` as the measurement that killed this mechanism for its own
    // family is closed. It changes NOTHING for `on_material` or
    // `next_on_material`, whose inventions therefore do NOT come through this
    // arm and are not reached by any of the four conditions in this function.
    // Those two are COMPOSITIONS, which is the hop-count limit recorded below:
    // the remaining fix is to fire only when the production derivation found
    // no taught sub-question at all (backlog `6eb030ab`).
    if base_score >= CEILING && brain.is_trained_frame(query_pool, query) {
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
    // Of the answers above, the ones reached by a SUBJECT-PRESERVING rewrite.
    // Kept beside `answers` rather than replacing it: see `AT THE ACCEPT`.
    let mut preserving: Vec<Vec<u8>> = Vec::new();
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
                if admissible(brain, query_pool, query, &trained, &rewrite) {
                    preserving.push(answer.clone());
                }
                answers.push(answer);
                // Two distinct answers already settle it: the chain is not
                // unique, so no further probe can make it unique.
                if answers.len() > 1 {
                    return (None, probes);
                }
            } else if answer != base_answer
                && !preserving.contains(&answer)
                && admissible(brain, query_pool, query, &trained, &rewrite)
            {
                // Same answer, reached again by a rewrite that DOES preserve
                // the subject. The uniqueness test has already counted this
                // answer once; what is new is the evidence about how it was
                // reached, and that is what the accept reads.
                preserving.push(answer);
            }
        }
    }
    // UNIQUENESS, AND THEN SUBJECT PRESERVATION AT THE ACCEPT. Exactly one
    // distinct ceiling-reachable answer, AND that answer must have been
    // reached by a rewrite that kept the query's own subject.
    if answers.len() == 1 && preserving.contains(&answers[0]) {
        return (Some(answers.remove(0)), probes);
    }
    (None, probes)
}

/// May this rewrite's answer be RETURNED? Two necessary conditions, both at the
/// accept and neither on the search.
///
/// # ORDER. `score >= CEILING` does not mean "a question the brain was taught"
///
/// `best_binding_match_v2` scores precision x recall over the UNORDERED
/// DISTINCT BYTE SET, so an anagram of a trained question reaches the exact
/// ceiling -- and a rewrite that keeps the query's own subject can therefore
/// resolve against a trained question of a DIFFERENT subject. That is why
/// subject preservation alone measured exactly inert: all four inventions
/// survived it, so the fault was never in the splice.
///
/// `Brain::is_trained_frame` is FNV-1a over the pool id and the frame BYTES, so
/// it is order- and multiplicity-sensitive where the matcher is not. The
/// production derivation already admits only rewrites that are in it; this
/// mechanism accepted on the score alone, which is the whole gap.
///
/// This is the owner's rule stated in code: *an answer is returned only when
/// every step of its derivation is an EXACT, ORDERED trained binding and the
/// chain is UNIQUE*. Uniqueness is the caller's test over the answer SET;
/// these two are the test on the DERIVATION.
///
/// # BOTH CONDITIONS ARE MEASURED INERT ON THIS WORLD, and that is the finding
///
/// Kept because they are the stated rule and cost nothing, NOT because they are
/// what makes the mechanism safe -- it is not safe, and nothing here makes it
/// so. Measured 2026-10-01 at scales 1 and 4 through
/// `tests/integration_family_counts.rs`, the per-family counts are
/// byte-identical with each condition and without it:
///
/// ```text
///   uniqueness over the ceiling answer set    NOT inert (invention 5 -> 3)
///   preserves_subject, at the accept          EXACTLY INERT
///   is_trained_frame, ORDER-sensitive         EXACTLY INERT
/// ```
///
/// So the four remaining inventions come from rewrites that are byte-exact
/// trained questions, present in the ordered digest, keeping the query's own
/// subject. They are legitimate taught text asking THE WRONG QUESTION, and the
/// first diagnosis -- that the unordered byte-set matcher was the fault
/// (`f711d18a`) -- does not survive this: order was added and changed nothing.
///
/// The limit is semantic and no accept rule on the rewrite reaches it. The one
/// condition that separates the family this mechanism answers from the ones it
/// invents on is HOP COUNT: `beside_next` is a held-out SYNONYM of a trained
/// relation, so the rewrite's answer is identical to the query's; `on_material`
/// and `next_on_material` are COMPOSITIONS that no single trained question
/// answers. Firing the transfer only when the production derivation found no
/// taught sub-question at all is therefore the gate, and
/// `derive_by_substitution_profiled` does not report that today.
fn admissible(
    brain: &Brain,
    query_pool: PoolId,
    query: &[u8],
    trained: &[u8],
    rewrite: &[u8],
) -> bool {
    brain.is_trained_frame(query_pool, rewrite) && preserves_subject(query, trained, rewrite)
}

/// Does `rewrite` keep the QUERY's byte at the first position where the query
/// and the reverse-decoded trained question `trained` disagree?
///
/// # AT THE ACCEPT, NOT AT THE SEARCH -- and that distinction is the whole
/// reason this exists after the same idea was measured to fail
///
/// Filtering the CANDIDATE SET to subject-preserving rewrites took
/// `next_on_material` from 2 wrong to 4 wrong (recorded at `README.md:556`),
/// because the candidates it removed were the ones producing a SECOND distinct
/// answer -- and a second distinct answer is what triggers the abstain.
/// Pruning the search destroys the evidence of absence while leaving the
/// conclusion.
///
/// This condition is therefore applied to the ONE ANSWER RETURNED and to
/// nothing else: every candidate is still asked, every distinct answer still
/// counts toward uniqueness, and what changes is only whether the surviving
/// answer is allowed out. Uniqueness is a test on the ANSWER SET; this is a
/// test on the DERIVATION, so composing them is strictly narrowing and cannot
/// convert a silence into an answer.
///
/// # What it assumes, stated because it is an assumption and not a measurement
///
/// `align` finds the common prefix `p`, so `query[p]` is the first byte that is
/// the query's own rather than borrowed. Requiring `rewrite[p] == query[p]`
/// assumes the differing SUBJECT is at or after that first disagreement -- in
/// `"r001 beside?"` against `"r000 next?"`, `p == 3` and the subject's last
/// byte is exactly there, which is why the wrong-subject rewrite `"r000 next?"`
/// takes `trained[3]` and the wanted `"r001 next?"` keeps `query[3]`. It is the
/// same structural assumption the `|a - i|` candidate ordering already encodes
/// and it is not a probe's wording: no relation word, family or hop count
/// appears here. A world whose subject followed its relation would need the
/// mirror condition on the common SUFFIX, and this function would then be
/// measurably inert rather than silently wrong -- `preserving` would be empty
/// and the mechanism would abstain everywhere.
fn preserves_subject(query: &[u8], trained: &[u8], rewrite: &[u8]) -> bool {
    let (p, _) = align(query, trained);
    // The query is a prefix of the trained question: it has no byte of its own
    // to preserve, so there is nothing for a rewrite to drop.
    if p >= query.len() {
        return true;
    }
    rewrite.get(p) == query.get(p)
}

/// THE ANSWER ENTRY POINT: the production derivation first, this module's
/// transfer only when that returned nothing.
///
/// # Why the composition lives here and not inside `derive_by_substitution`
///
/// `f16e499d` specified the wiring as a fallback *inside*
/// `derive_by_substitution_profiled`. That would have put it in `brain.rs`,
/// and two measured properties argue for this placement instead of that one.
///
/// First, `tests/relation_transfer_derivation.rs` asserts that the transfer
/// derives at least one `beside_next` **the production derivation cannot**. A
/// fallback inside `derive_by_substitution_profiled` makes the production
/// derivation able to do it, so that assertion becomes a statement about
/// itself and stops discriminating -- the exact shape CLAUDE.md records for
/// `relation_transfer_derivation` going red when production improved (three
/// times: `2a1c445`, `70bba62`, pass 16). Composed one level out, the inner
/// function is untouched and the comparison stays meaningful.
///
/// Second, `derive_by_substitution_profiled` owns the probe accounting the
/// starvation work reads (`derivation_starved`, probes/attempt). Folding a
/// second mechanism's probes into that counter would move a number another
/// agent is measuring against, for no gain -- so the transfer reports its own
/// cost, as the second element of the returned pair.
///
/// # What it cannot cost
///
/// The transfer fires only on an empty production answer, so every query the
/// production path already answers returns byte-identically and spends zero
/// extra probes. That is the property
/// `as_a_fallback_the_transfer_cannot_cost_a_family` asserts per family.
///
/// `query` is the question's bytes. The caller has already observed them --
/// `integrate_autonomous` recovers the question from `recent_frames` and
/// cannot take it as an argument -- but the transfer asks REWRITES, which were
/// never observed, so it needs the bytes explicitly.
pub fn answer_with_relation_transfer(
    brain: &mut Brain,
    query_pool: PoolId,
    answer_pool: PoolId,
    query: &[u8],
    fabric_confidence_threshold: f32,
    chain_max_depth: usize,
    chain_max_visit: usize,
) -> (Option<Vec<u8>>, usize) {
    let direct = brain
        .integrate_autonomous(
            query_pool,
            answer_pool,
            fabric_confidence_threshold,
            chain_max_depth,
            chain_max_visit,
        )
        .answer
        .filter(|a| !a.is_empty());
    if direct.is_some() {
        return (direct, 0);
    }
    // The same budget gate `integrate_autonomous` applies to its own
    // derivation arm: a brain configured not to derive does not derive here
    // either, and there is one setting rather than two.
    let budget = brain.derivation_probe_budget();
    if budget == 0 {
        return (None, 0);
    }
    derive_by_relation_transfer(brain, query_pool, answer_pool, query, budget)
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

    /// THE ACCEPT CONDITION, asserted on the pair the module's own header
    /// names: the aligned substitution and the tail transfer differ by exactly
    /// one byte and that byte is the subject's.
    #[test]
    fn the_accept_condition_separates_the_tail_transfer_from_the_aligned_one() {
        let (q, t) = (b"r001 beside?".as_slice(), b"r000 next?".as_slice());
        assert!(
            preserves_subject(q, t, b"r001 next?"),
            "the tail transfer keeps the query's own subject byte and must be accepted"
        );
        assert!(
            !preserves_subject(q, t, b"r000 next?"),
            "the aligned substitution borrows the subject byte and must be refused"
        );
        // A rewrite shorter than the first disagreement cannot carry the
        // subject, so `get` must refuse rather than index out of bounds.
        assert!(!preserves_subject(q, t, b"r0"));
        // The query is a prefix of the trained question: no byte of its own is
        // in dispute.
        assert!(preserves_subject(b"r001", b"r001 next?", b"r001 next?"));
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
