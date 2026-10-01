//! What the posting cap does and does NOT hide, measured -- and the headline
//! this file was written to prove is REFUTED by its own run.
//!
//! # The asymmetry that is real
//!
//! `routed_binding_candidates` consults `binding_sequence_postings` -- the one
//! posting key unique per distinct taught question, and so the one that can
//! never saturate -- only when it is handed a target pool. Until 2026-10-01
//! both tier scorers passed `None`, so `probe_question_score` ->
//! `best_binding_match_v2` (the SCORE half of a derivation probe) saw only
//! per-byte feature postings, capped at 512 and selected NEWEST-first, while
//! `decode_best_trained_binding_with_context` (the ANSWER half) passed the
//! target pool and had the exact ordered route. Passing it through moved the
//! scorecard: integration 64.47 -> 65.62 at scale 16 and 38.54 -> 39.00 at
//! scale 64, wrong 0.0 and recall 100.0 at every scale, scales 1 and 4
//! digit-identical (commit 6fee594, read off `logs/scorecard-latest.json`).
//!
//! # Where the loss actually is: the CANDIDATE RANK, not the posting overlay
//!
//! The mechanism first claimed for that gain was: a shared byte's posting list
//! is over the cap, the newest 512 hold only the last questions taught, the
//! first-come atom fan-out bound keeps the earliest, so the MIDDLE of the
//! training order is reachable by neither route.
//!
//! Measured here at 512 rooms -- 1,536 facts against a 512-entry cap, with
//! `feature_saturated` 10 of 12 lookups so the cap is demonstrably biting:
//!
//! ```text
//! room   0: "r000 lamp on?" score without target 1.0000, with 1.0000
//! room 256: "r256 lamp on?" score without target 1.0000, with 1.0000
//! room 511: "r511 lamp on?" score without target 1.0000, with 1.0000
//! ```
//!
//! Nothing is hidden at any position, with or without the exact route. A
//! truncated feature list does not lose the right binding here, because the
//! rare bytes of a room id have short posting lists nowhere near the cap and
//! they vote for the correct binding alongside `terminal_routes`. Saturating
//! the COMMON bytes costs nothing while one uncommon atom still reaches the
//! answer. So the overlay's newest-first selection is NOT the loss, and this
//! world is too small to contain whatever is.
//!
//! The loss is one step later, in `rank_bounded_binding_evidence`: it sorts
//! candidates by how many firing atoms voted for them, truncates to
//! `MAX_ROUTED_BINDING_CANDIDATES` (512) and tie-breaks by id DESCENDING, so
//! the newest bindings win ties. A binding that every feature posting has
//! dropped arrives with exactly ONE vote -- from the exact route -- and loses
//! both the sort and the tie-break. Consulting the exact route therefore
//! bought almost nothing on its own; letting it KEEP ITS SLOT, with the 512
//! bound preserved, is the whole fix. Measured on the same build, `python
//! tools/scorecard.py --scales 16,64`:
//!
//! ```text
//! scale 16: integration 65.62 -> 77.50   on_material 291/384 -> 382/384   empty 93 -> 2
//! scale 64: integration 39.00 -> 77.34   on_material 647/1536 -> 1534/1536 empty 889 -> 2
//! ```
//!
//! with `integration_wrong_pct` 0.0 and `recall_pct` 100.0 at every scale, and
//! scale-64 `peak_mb` 40.3 against a 40.4 baseline. Three scale-dependent
//! stories about posting lists were wrong about WHERE the binding was lost,
//! and the counter that settled it (`PostingCapStats`) is the one that proved
//! the cap was biting while the answer survived anyway -- saturation and
//! unreachability are different facts.
//!
//! What this file therefore asserts is only what it can show: that a taught
//! question scores the ceiling at every position in the training order (the
//! precondition the derivation's `>= 1.0` cut test depends on), and that
//! posting-cap saturation is COUNTED per key kind rather than inferred.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

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

/// `rooms` rooms, three relations each, taught in room order -- so a room's
/// index IS its position in the training order.
fn teach_world(brain: &mut Brain, rooms: u32) {
    for r in 0..rooms {
        let room = format!("r{r:03}");
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("{room} next?").into_bytes()),
            (ANSWER_POOL, format!("r{:03}", (r + 1) % rooms).into_bytes()),
        ]);
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("{room} lamp on?").into_bytes()),
            (ANSWER_POOL, b"desk".to_vec()),
        ]);
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, format!("{room} desk material?").into_bytes()),
            (ANSWER_POOL, format!("m{:03}", r % 7).into_bytes()),
        ]);
    }
}

/// The score the two halves give the SAME taught question: without the target
/// pool (what the score half used to see) and with it (what the answer half
/// always saw).
fn scores(brain: &mut Brain, question: &str) -> (f32, f32) {
    brain.observe_read_only(QUERY_POOL, question.as_bytes());
    let without = brain.best_binding_match_routed(QUERY_POOL, None).score();
    brain.observe_read_only(QUERY_POOL, question.as_bytes());
    let with = brain
        .best_binding_match_routed(QUERY_POOL, Some(ANSWER_POOL))
        .score();
    (without, with)
}

/// Every taught question must score 1.0 to the score half, wherever in the
/// training order it was taught. This is the criterion the derivation's
/// `>= 1.0` cut test depends on, and the one the cap broke.
#[test]
fn a_taught_question_scores_one_at_every_position_in_the_training_order() {
    // 512 rooms is the scorecard's scale 64: 1,536 facts against a 512-entry
    // posting cap, so a shared byte's list is 3x over.
    const ROOMS: u32 = 512;
    let mut brain = subject();
    teach_world(&mut brain, ROOMS);

    // First, middle and last room. The middle is the one with neither route.
    let probes = [0u32, ROOMS / 2, ROOMS - 1];
    let mut rows = Vec::new();
    for r in probes {
        let question = format!("r{r:03} lamp on?");
        let (without, with) = scores(&mut brain, &question);
        eprintln!("room {r:3}: \"{question}\" score without target {without:.4}, with {with:.4}");
        rows.push((r, without, with));
    }

    // Training does no posting lookups -- they happen on the answer path --
    // so the guard that this measured anything at all belongs here and not
    // after `teach_world`. (`posting_caps_train` is all zeros in the
    // scorecard JSON for the same reason, and that is a fact about where the
    // lookups are, not a broken counter.)
    assert!(
        brain.posting_cap_stats().saturated_fraction().is_some(),
        "no posting lookup ran, so this measures nothing"
    );

    // The fix's claim, stated as the assertion: with the exact ordered route
    // available, a taught question scores the ceiling at every position.
    for (r, _, with) in &rows {
        assert!(
            *with >= 1.0,
            "taught question for room {r} scores {with:.4} with the exact route; \
             the derivation's cut test needs 1.0 and training put it there"
        );
    }

    // The cap must be biting, or the paragraph above is about nothing. This
    // is the anti-vacuous guard, and it is the one that survived: the cap
    // bites and the right binding is found anyway.
    let caps = brain.posting_cap_stats();
    assert!(
        caps.feature_saturated > 0,
        "no feature posting lookup came back full at {ROOMS} rooms, so this measures nothing about the cap: {caps:?}"
    );

    // And the refutation, pinned so nobody re-derives the hidden-middle story
    // from the saturation counter alone. If a future change makes a taught
    // question unreachable from the score half, THIS is the assertion that
    // fails, and the module comment becomes wrong and must be rewritten
    // rather than this line relaxed.
    let hidden = rows.iter().filter(|(_, without, _)| *without < 1.0).count();
    assert_eq!(
        hidden, 0,
        "a taught question is now hidden from the score half without the exact route; that is the mechanism this file records as REFUTED at 512 rooms: {rows:?}"
    );
}

/// A saturated lookup must be COUNTED, not inferred from a fact count. The
/// previous readout was one boolean on one key kind for the duration of one
/// call.
#[test]
fn posting_cap_saturation_is_counted_per_key_kind() {
    let mut brain = subject();
    let before = brain.posting_cap_stats();
    assert_eq!(before.saturated_fraction(), None, "nothing looked up yet");

    teach_world(&mut brain, 512);
    brain.observe_read_only(QUERY_POOL, b"r256 lamp on?");
    let _ = brain.best_binding_match_routed(QUERY_POOL, Some(ANSWER_POOL));
    let after = brain.posting_cap_stats();
    let delta = after.minus(before);

    eprintln!("posting caps: {delta:?} fraction {:?}", delta.saturated_fraction());
    assert!(delta.feature_lookups > 0, "the feature route was never consulted");
    assert!(
        delta.feature_saturated > 0,
        "1,536 facts share a byte against a 512 cap and nothing came back full: {delta:?}"
    );
    // The exact-sequence key is unique per taught question, so it is the one
    // kind that must NEVER saturate -- that is why it is the fix.
    assert_eq!(
        delta.sequence_saturated, 0,
        "the exact ordered key saturated, which would make it as blind as the \
         feature route: {delta:?}"
    );
    assert!(
        delta.saturated_fraction().unwrap() > 0.0,
        "fraction must be non-zero when a kind saturated: {delta:?}"
    );
}
