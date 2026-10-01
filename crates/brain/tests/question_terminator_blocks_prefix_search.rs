//! Why the derivation measures 32 of 32 in its own test and 0 of 3,456 in the
//! scorecard, on worlds that chain the same way.
//!
//! `Brain::derive_by_substitution` finds the taught sub-question by walking
//! PREFIXES of the asked question and accepting the first that scores 1.0
//! (`brain.rs`, `for k in 1..n` over `current[..k]`). A question scores 1.0
//! only when precision and recall are both 1, so a prefix that scores 1.0 IS a
//! question the brain was taught -- that part is sound, and cheap.
//!
//! It is also the only shape the search can find, and `tests/
//! derive_by_substitution.rs` teaches a world with NO question terminator
//! (`"r03 lamp on"` -> `"desk"`), where the taught sub-question is a prefix of
//! the asked one by construction. The scorecard's world
//! (`examples/scorecard.rs`) terminates every question with `?`, so the taught
//! `"r001 lamp on?"` is NOT a prefix of the asked `"r001 lamp on material?"` --
//! the `?` sits in the MIDDLE at offset 12. No prefix of the asked question is
//! a trained question, the search returns `None` every time, and the fallback
//! then splices the answer to the whole UNTRAINED question across `n(n+1)/2`
//! spans: maximum cost, zero derivations.
//!
//! This file measures that, so the finding is a number rather than a reading of
//! the source, and so the two halves stay pinned:
//!
//!   1. NO prefix of the asked question is a trained question. If this ever
//!      becomes false the scorecard world has drifted back to the one shape the
//!      prefix search already handles, and its integration score stops meaning
//!      anything.
//!   2. The chain IS reachable -- by deleting ONE CONTIGUOUS SPAN of the asked
//!      question instead of only a suffix. `"r001 lamp on material?"` minus the
//!      span `" material"` is `"r001 lamp on?"`, which is trained and answers
//!      `"desk"`; splicing `"desk"` back over `"lamp on"` gives
//!      `"r001 desk material?"`, which is trained and answers the integration
//!      probe. A prefix is the special case of that deletion whose kept tail is
//!      empty, so generalising costs the search nothing it was not already
//!      paying.
//!
//! Nothing here knows that `?` is a terminator, and nothing splits on it. The
//! deletion span is found by scanning, exactly as the prefix is.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// The same two-pool byte-passthrough brain the scorecard builds.
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

/// Exactly what `Brain::probe_question` does, in public API: observe the
/// candidate into the query pool without learning from it, then read how well
/// the trained bindings know it and what they answer.
fn probe(brain: &mut Brain, question: &[u8]) -> (f32, Option<Vec<u8>>) {
    brain.observe_fabric_read_only(QUERY_POOL, question);
    let score = brain.best_binding_match_v2(QUERY_POOL).score();
    let answer = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .filter(|a| !a.is_empty());
    (score, answer)
}

/// A miniature of the scorecard's scene world, carrying the one property this
/// file is about: every question ends in `?`. The colour and `near?` rows are
/// the scorecard's distractors, and they are here so the prefix scores below
/// are measured against a brain with alternatives rather than a brain with one
/// thing to say.
fn teach_terminated_world(brain: &mut Brain, rooms: usize) {
    let objects = ["bed", "chair", "mirror", "desk", "lamp", "door", "window", "paper"];
    let materials = ["oak", "steel", "glass", "cloth", "pine", "brass"];
    let colors = ["red", "blue", "green", "white", "black", "grey"];
    for r in 0..rooms {
        let room = format!("r{r:03}");
        for (i, obj) in objects.iter().enumerate() {
            teach(brain, &format!("{room} {obj} color?"), colors[(r * 31 + i * 17) % colors.len()]);
            teach(
                brain,
                &format!("{room} {obj} material?"),
                materials[(r * 31 + (i + 1) * 17) % materials.len()],
            );
        }
        teach(brain, &format!("{room} lamp on?"), "desk");
        teach(brain, &format!("{room} lamp near?"), "window");
        teach(brain, &format!("{room} next?"), &format!("r{:03}", (r + 1) % rooms));
    }
}

#[test]
fn the_taught_sub_question_is_not_a_prefix_but_is_one_deletion_away() {
    const ROOMS: usize = 8;
    let mut brain = subject();
    teach_terminated_world(&mut brain, ROOMS);

    let asked = "r000 lamp on material?";
    let n = asked.len();

    // 0. The control. Unless a question the brain WAS taught scores 1.0, the
    //    prefix measurement below is about a broken scorer and not about shape.
    let (taught_score, taught_answer) = probe(&mut brain, b"r000 lamp on?");
    assert!(
        taught_score >= 1.0,
        "a TRAINED question scored {taught_score:.4}, so nothing below measures question shape"
    );
    assert_eq!(
        taught_answer.as_deref(),
        Some(&b"desk"[..]),
        "the trained sub-question does not recall its own answer"
    );

    // 1. Every prefix of the asked question, exactly as the prefix search walks
    //    them. The search accepts the first at 1.0; none of these is.
    let mut prefix_rows = Vec::new();
    let mut best_prefix = (0.0f32, 0usize);
    for k in 1..n {
        let (score, _) = probe(&mut brain, &asked.as_bytes()[..k]);
        if score > best_prefix.0 {
            best_prefix = (score, k);
        }
        prefix_rows.push((k, score));
    }

    // 2. The same scan, one byte of kept tail. `current[..k] ++ current[n-1..]`
    //    is the asked question with the span `(k, n-1)` deleted; `k = n-1` is
    //    the asked question itself and is skipped.
    let mut deletion_hit: Option<(usize, Vec<u8>, Vec<u8>)> = None;
    for k in 1..(n - 1) {
        let mut candidate = asked.as_bytes()[..k].to_vec();
        candidate.extend_from_slice(&asked.as_bytes()[n - 1..]);
        let (score, answer) = probe(&mut brain, &candidate);
        if score >= 1.0 {
            if let Some(answer) = answer {
                deletion_hit = Some((k, candidate, answer));
                break;
            }
        }
    }

    println!(
        "asked {asked:?} ({n} bytes), trained control scored {taught_score:.4}\n  \
         best PREFIX           k={} score {:.4}  (search accepts only >= 1.0)\n  \
         prefix scores         {:?}",
        best_prefix.1,
        best_prefix.0,
        prefix_rows
            .iter()
            .map(|(k, s)| format!("{k}:{s:.2}"))
            .collect::<Vec<_>>(),
    );

    assert!(
        best_prefix.0 < 1.0,
        "a prefix of {asked:?} scored {:.4} at k={}, so the world no longer has the \
         property the scorecard exists to measure -- the taught sub-question is \
         reachable by the prefix search and integration there is not evidence of \
         anything general",
        best_prefix.0,
        best_prefix.1,
    );

    let (k, candidate, link) = deletion_hit.expect(
        "no single-span deletion of the asked question is a trained question either, so the \
         chain is not reachable by substitution at all and this world cannot be derived",
    );
    println!(
        "  DELETION hit          k={k} {:?} -> {:?}",
        String::from_utf8_lossy(&candidate),
        String::from_utf8_lossy(&link),
    );
    assert_eq!(
        link.as_slice(),
        b"desk",
        "the deletion found a trained question but not the one on the chain"
    );

    // 3. And the splice the existing `for j in 0..=k` arm already builds reaches
    //    a trained question, so the hop is complete rather than merely started.
    let mut spliced = None;
    for j in 0..=k {
        let mut rewrite = asked.as_bytes()[..j].to_vec();
        rewrite.extend_from_slice(&link);
        rewrite.extend_from_slice(&asked.as_bytes()[k..]);
        let (score, answer) = probe(&mut brain, &rewrite);
        if score >= 1.0 {
            if let Some(answer) = answer {
                spliced = Some((j, rewrite, answer));
                break;
            }
        }
    }
    let (j, rewrite, answer) = spliced.expect(
        "splicing the link back over the asked question reached no trained question, so the \
         second hop of the chain is unreachable",
    );
    println!(
        "  SPLICE hit            j={j} {:?} -> {:?}",
        String::from_utf8_lossy(&rewrite),
        String::from_utf8_lossy(&answer),
    );

    // The world's own truth: r000's desk material. Stated here, not in
    // crates/brain, and compared against what two trained questions compose to.
    let materials = ["oak", "steel", "glass", "cloth", "pine", "brass"];
    let desk = 3usize;
    let truth = materials[(0 * 31 + (desk + 1) * 17) % materials.len()];
    assert_eq!(
        String::from_utf8_lossy(&answer),
        truth,
        "the composed answer is not what the world says is true for r000's desk"
    );
}
