//! The derivation's accept rule was FALSE, and this file pins the exact case
//! that proves it and the fix that closes it.
//!
//! # The defect, in one string
//!
//! `Brain::best_binding_match_atom_tier` scores a question against a binding
//! as precision x recall over SETS of firing atom neurons. An atom is a BYTE,
//! so a binding trained on `"r000 desk material?"` carries one member per
//! DISTINCT byte of it — order and multiplicity are represented nowhere in the
//! member set. `"r0desk material?"` is built from exactly those distinct bytes
//! (it drops two `0`s and a space, all of which repeat), so it scores the
//! ceiling 1.0 against that binding and decodes r000's material.
//!
//! `derive_by_substitution` accepted any rewrite scoring 1.0 on the stated
//! grounds that "a question scoring 1.0 IS a question the brain was taught".
//! That is the sentence this file falsifies. The truncation is reachable as a
//! real rewrite: from `"r001 lamp on material?"`, splicing the answer `desk`
//! over the span `[2, 12)` produces `"r0desk material?"` — which is why the
//! invention measured 4 wrong answers per right one at every scale, and why no
//! vote over the ceiling set removes it (a vote can only pick between
//! truncations when truncations outnumber the truth).
//!
//! # What the fix is
//!
//! `Brain::is_trained_frame` — the identity of what was taught, recorded on
//! the training path where it is known for certain, consulted as a membership
//! test. It reads no byte and knows no grammar. Asserted here in both
//! directions: the anagram is refused, the real question is admitted, and a
//! derivation that depends on the real question still works.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// The measured case, verbatim. See the module docs.
const TAUGHT: &str = "r000 desk material?";
const ANAGRAM: &str = "r0desk material?";

fn brain() -> Brain {
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
        brain.create_pool(
            pc,
            Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>,
        );
    }
    brain
}

fn teach(brain: &mut Brain, question: &str, answer: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, question.as_bytes().to_vec()),
        (ANSWER_POOL, answer.as_bytes().to_vec()),
    ]);
}

/// Enough of the scorecard's world that the derivation has somewhere to go,
/// and that the anagram's bytes are all present.
fn teach_world(brain: &mut Brain) {
    let facts: Vec<(String, String)> = vec![
        ("r000 desk material?".into(), "oak".into()),
        ("r000 lamp material?".into(), "steel".into()),
        ("r000 lamp on?".into(), "desk".into()),
        ("r001 desk material?".into(), "steel".into()),
        ("r001 lamp material?".into(), "glass".into()),
        ("r001 lamp on?".into(), "desk".into()),
        ("r002 desk material?".into(), "glass".into()),
        ("r002 lamp on?".into(), "desk".into()),
    ];
    for _ in 0..3 {
        for (q, a) in &facts {
            teach(brain, q, a);
        }
    }
}

/// The premise: the anagram really does reach the ceiling, so the score cannot
/// be the accept test. If this assertion ever fails the matcher has gained
/// order or multiplicity evidence and the rest of this file is about a defect
/// that no longer exists — which is a reason to re-measure, not to delete it.
#[test]
fn the_anagram_reaches_the_ceiling_so_the_score_cannot_be_the_accept_test() {
    let mut brain = brain();
    teach_world(&mut brain);

    // The same two calls `probe_question_score` makes, through the public API.
    brain.observe_fabric_read_only(QUERY_POOL, TAUGHT.as_bytes());
    let taught_score = brain.best_binding_match_v2(QUERY_POOL).score();
    brain.observe_fabric_read_only(QUERY_POOL, ANAGRAM.as_bytes());
    let anagram_score = brain.best_binding_match_v2(QUERY_POOL).score();

    assert!(
        taught_score >= 1.0,
        "{TAUGHT:?} is taught, so it must score the ceiling; scored {taught_score}"
    );
    assert!(
        anagram_score >= 1.0,
        "the whole defect is that {ANAGRAM:?} scores the ceiling too; scored {anagram_score}. \
         If this is now below 1.0 the matcher has changed and the fix may be redundant."
    );
}

/// The fix, in the direction that matters: the anagram is REFUSED.
#[test]
fn derivation_rejects_an_anagram_of_a_trained_question() {
    let mut brain = brain();
    teach_world(&mut brain);

    assert!(
        !brain.is_trained_frame(QUERY_POOL, ANAGRAM.as_bytes()),
        "{ANAGRAM:?} was never taught, so it must not be admitted as a rewrite"
    );
    assert!(
        brain.is_trained_frame(QUERY_POOL, TAUGHT.as_bytes()),
        "{TAUGHT:?} WAS taught, so refusing it would break every derivation"
    );
}

/// The same assertion through the public derivation, which is what the answer
/// path calls. `"r001 lamp on material?"` is never taught; its honest
/// derivation is `"r001 desk material?"` -> `steel`, and `"r0desk material?"`
/// -> `oak` is the invention. Asserting on the ANSWER is what makes this a
/// test of the mechanism rather than of the helper.
#[test]
fn derive_by_substitution_does_not_answer_from_a_truncated_rewrite() {
    let mut brain = brain();
    teach_world(&mut brain);

    let derived = brain.derive_by_substitution(
        QUERY_POOL,
        ANSWER_POOL,
        b"r001 lamp on material?",
        2,
        512,
    );
    let derived = derived.map(|d| String::from_utf8_lossy(&d).to_string());

    assert_ne!(
        derived.as_deref(),
        Some("oak"),
        "oak is r000's desk material, reachable only through the {ANAGRAM:?} truncation"
    );
    assert_eq!(
        derived.as_deref(),
        Some("steel"),
        "the honest derivation is {:?} -> steel; got {derived:?}",
        "r001 desk material?"
    );
    assert!(
        brain.derivation_stats().rejected_untaught > 0,
        "the truncations are reachable rewrites, so some must have been refused; \
         a zero here means the rewrites never ran and this test proves nothing"
    );
}

/// The counter must not be bought by refusing everything: a derivation that
/// only needs taught rewrites still answers.
#[test]
fn a_taught_rewrite_is_still_admitted() {
    let mut brain = brain();
    teach_world(&mut brain);

    let derived = brain
        .derive_by_substitution(QUERY_POOL, ANSWER_POOL, b"r000 lamp on material?", 2, 512)
        .map(|d| String::from_utf8_lossy(&d).to_string());

    assert_eq!(
        derived.as_deref(),
        Some("oak"),
        "r000 lamp on -> desk, r000 desk material? -> oak, both taught; got {derived:?}"
    );
}
