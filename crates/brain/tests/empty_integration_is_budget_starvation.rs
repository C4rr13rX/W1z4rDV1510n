//! Integration's `empty` miss bucket is BUDGET STARVATION, and the obvious
//! cause of the starvation is refuted here.
//!
//! Measured 2026-10-01 off `logs/scorecard-latest.json` with the per-family
//! cost counters: the `empty` miss bucket and `derivation_starved` are the
//! same number. scale 16 `on_material` 87 and 87; scale 16 `next_color` 4 and
//! 4; scale 64 `next_color` 362 and 362; scale 64 overall **1,557 starved
//! attempts of 3,456**. So `empty:852`, which four passes have read as "no
//! chain at all", is 843 budget exhaustions -- a ceiling, not an absence. Cost
//! per ANSWER for `on_material` runs 9.08 / 6.77 / 16.54 / 56.01 probes at
//! scales 1 / 4 / 16 / 64 against a shipped ceiling of 32, on a question whose
//! SHAPE is identical at every scale.
//!
//! # What this test REFUTES
//!
//! `derive_by_substitution_profiled` has exactly one cliff: when no deletion
//! of the question scores `>= 1.0`, `known_prefix` stays `None` and the splice
//! search widens from `k + 1` spans to all `n(n+1)/2` -- 253 questions at
//! `n = 22` against a budget of 32, which fits no budget this product would
//! ship. The natural hypothesis is that the corpus dilutes the score, so the
//! taught sub-question stops reaching 1.0 and every large-corpus question
//! falls off that cliff.
//!
//! **It does not.** Measured here, same shape, three relations, one question:
//!
//! ```text
//! rooms    8: asked scores 0.7692; taught cut scores 1.0000 -> "desk"; spent 29 of 32, derived true
//! rooms  128: asked scores 0.7692; taught cut scores 1.0000 -> "desk"; spent 29 of 32, derived true
//! rooms  512: asked scores 0.7692; taught cut scores 1.0000 -> "desk"; spent 29 of 32, derived true
//! ```
//!
//! The score is scale-INVARIANT to four decimals, and so is the cost. 512
//! rooms is the scorecard's scale 64. So whatever makes the scorecard's
//! scale-64 questions starve is NOT score dilution, and it is not the question
//! shape either -- the asked question's own score is 0.7692 at every size.
//!
//! What it leaves: the cost is **29 of 32**, three probes of headroom on a
//! COLD cut cache. Anything that costs three more probes starves, and the
//! scorecard's world has six relations to this one's three (`color?`,
//! `material?`, `on?`, `near?`, `next?`, `beside?`), eight object names of
//! differing length rather than one, and `EPOCHS` passes of shuffled training.
//! The margin, not the mechanism, is what the next change has to buy.
//!
//! # And one repair already measured INERT against it
//!
//! Accepting the best-scoring deletion when none reaches 1.0, under the same
//! "strictly better known than the question we started from" test the splice
//! search already applies to its rewrites, moved `empty` at scale 64 by
//! **zero** (852 before, 852 after) while `on_material` hits fell 182 -> 157
//! and its wrong answers rose 502 -> 527. Integration fell 35.2 -> 34.8 at
//! scale 16 and 20.6 -> 20.3 at scale 64. Reverted. Two reasons, both visible
//! in the numbers above: the cut IS already scoring 1.0 in the cases that
//! matter, so the rule was never what blocked them; and the cut is a DIFFERENT
//! question from the one asked, so `0.7692` is the wrong thing to score it
//! against -- the few cuts that cleared that bar spliced wrong answers where
//! there had been silence.

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
    DEFAULT_DERIVATION_PROBE_BUDGET,
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

fn teach(brain: &mut Brain, question: &str, answer: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, question.as_bytes().to_vec()),
        (ANSWER_POOL, answer.as_bytes().to_vec()),
    ]);
}

/// The scorecard's `on_material` shape at an arbitrary room count.
fn teach_world(brain: &mut Brain, rooms: u32) {
    for r in 0..rooms {
        let room = format!("r{r:03}");
        teach(brain, &format!("{room} next?"), &format!("r{:03}", (r + 1) % rooms));
        teach(brain, &format!("{room} lamp on?"), "desk");
        teach(brain, &format!("{room} desk material?"), &format!("m{:03}", r % 7));
    }
}

/// The score of the question asked, and of the one deletion that IS taught.
fn scores(brain: &mut Brain, rooms: u32) -> (f32, f32, Option<Vec<u8>>, usize, bool) {
    let asked = "r000 lamp on material?";
    let cut = "r000 lamp on?"; // the taught sub-question the search hunts for
    brain.observe_read_only(QUERY_POOL, asked.as_bytes());
    let asked_score = brain.best_binding_match_v2(QUERY_POOL).score();
    brain.observe_read_only(QUERY_POOL, cut.as_bytes());
    let cut_score = brain.best_binding_match_v2(QUERY_POOL).score();
    let cut_answer = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
    let (derived, probes) = brain.derive_by_substitution_profiled(
        QUERY_POOL,
        ANSWER_POOL,
        asked.as_bytes(),
        3,
        DEFAULT_DERIVATION_PROBE_BUDGET,
    );
    let _ = rooms;
    (asked_score, cut_score, cut_answer, probes, derived.is_some())
}

#[test]
fn the_taught_cut_score_and_the_derivation_cost_are_scale_invariant() {
    // 8 rooms is the scorecard's scale 1; 512 is its scale 64. Same shape,
    // same three relations, same question.
    let mut report = Vec::new();
    for rooms in [8u32, 128, 512] {
        let mut brain = subject();
        teach_world(&mut brain, rooms);
        let (asked, cut, cut_answer, probes, derived) = scores(&mut brain, rooms);
        eprintln!(
            "rooms {rooms:4}: asked \"r000 lamp on material?\" scores {asked:.4}; \
             taught cut \"r000 lamp on?\" scores {cut:.4} -> {:?}; \
             derivation spent {probes} of {DEFAULT_DERIVATION_PROBE_BUDGET}, derived {derived}",
            cut_answer.as_ref().map(|a| String::from_utf8_lossy(a).into_owned()),
        );
        report.push((rooms, asked, cut, probes, derived));
    }

    // The cut is a question the brain WAS taught, so it must still be
    // recallable at every size -- if that fails, the defect is in training and
    // not in the derivation, and everything above is unreadable.
    for (rooms, _, _, _, _) in &report {
        let mut brain = subject();
        teach_world(&mut brain, *rooms);
        brain.observe_read_only(QUERY_POOL, b"r000 lamp on?");
        assert_eq!(
            brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL).as_deref(),
            Some(&b"desk"[..]),
            "the taught cut is not recalled at {rooms} rooms, so its score says nothing"
        );
    }

    // What this test exists to publish: whether the cut's score crosses the
    // `>= 1.0` acceptance the deletion search demands, and at which size.
    let small = report[0];
    let large = report[report.len() - 1];
    eprintln!(
        "VERDICT: taught-cut score {:.4} at {} rooms vs {:.4} at {} rooms; \
         acceptance needs >= 1.0",
        small.2, small.0, large.2, large.0
    );
    // The refutation, asserted so it cannot quietly stop being true: the
    // taught cut reaches the `>= 1.0` acceptance at BOTH sizes, and the
    // derivation that depends on it answers at both. A future change that
    // breaks this turns `empty` into a score problem and this test red, which
    // is the whole point of pinning a refuted hypothesis.
    for (rooms, _, cut, probes, derived) in &report {
        assert!(
            *cut >= 1.0,
            "the taught cut scores {cut:.4} at {rooms} rooms -- below the              deletion search's acceptance, so the corpus IS diluting the score"
        );
        assert!(derived, "the derivation found no answer at {rooms} rooms");
        assert!(
            *probes <= DEFAULT_DERIVATION_PROBE_BUDGET,
            "cost {probes} at {rooms} rooms exceeds the shipped budget"
        );
    }
    assert!(
        (small.2 - large.2).abs() < 1e-4,
        "the taught cut's score moved {:.4} -> {:.4} between {} and {} rooms",
        small.2, large.2, small.0, large.0
    );
}

#[test]
fn refused_hub_terminals_are_retained_on_store_and_paged_by_goal() {
    use w1z4rd_brain::NeuronRef;

    let rooms = 512u32;
    let total_facts = (rooms as usize) * 3;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "w1z4rd_hub_overflow_paging_{}_{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let wbrain_path = dir.join("brain.wbrain");

    let mut brain = subject();
    brain.attach_wbrain(&wbrain_path).unwrap();
    teach_world(&mut brain, rooms);

    let binding_pool_id = brain.stats().binding_pool_id;
    let r_atom = {
        let query_handle = brain.fabric().pool(QUERY_POOL).unwrap();
        let query = query_handle.read();
        let r_atom = query.label_to_id("q:cg").expect("'r' atom must exist");
        let resident_len = query.get(r_atom).unwrap().terminals.len();
        let overflow_len = query.atom_fanout_overflow_targets(r_atom).len();
        assert_eq!(resident_len, 512, "resident fan-out must stay capped at 512");
        assert_eq!(
            overflow_len,
            total_facts - 512,
            "refused hub terminals must be kept in overflow instead of dropped"
        );
        for binding_id in 0..total_facts as u32 {
            assert!(
                query.atom_reaches(r_atom, NeuronRef::new(binding_pool_id, binding_id)),
                "every taught fact's binding ({binding_id}) must be reachable from hub atom before idle"
            );
        }
        r_atom
    };

    brain.serialize_all_neurons_for_idle().unwrap();

    // After idle serialization to .wbrain, both resident neurons and in-RAM overflow
    // are dropped from RAM, and the full fan-out is recovered from the store on demand.
    {
        let query_handle = brain.fabric().pool(QUERY_POOL).unwrap();
        let query = query_handle.read();
        assert_eq!(
            query.resident_len(),
            0,
            "resident neurons must be evicted to .wbrain on idle serialization"
        );
        assert_eq!(
            query.atom_fanout_overflow_census(),
            (0, 0),
            "in-RAM overflow table must be empty after idle serialization to .wbrain"
        );
        let paged_overflow = query.atom_fanout_overflow_targets_paged(r_atom);
        assert_eq!(
            paged_overflow.len(),
            total_facts - 512,
            "all refused terminals must be retained on the .wbrain store"
        );
        let routes = query.binding_routes_from(r_atom, binding_pool_id);
        assert_eq!(
            routes.len(),
            total_facts,
            "full fan-out (resident + store overflow) must be recoverable for hub atom"
        );
        for binding_id in 0..total_facts as u32 {
            assert!(
                query.atom_reaches(r_atom, NeuronRef::new(binding_pool_id, binding_id)),
                "taught binding {binding_id} must be reachable from hub atom on store"
            );
        }
    }

    // Every taught fact across early, middle (both >512 from start and >512 from end),
    // and late rooms must page in its binding on demand when queried.
    for r in [0u32, 175, 250, 340, 511] {
        let q = format!("r{r:03} lamp on?");
        brain.observe_read_only(QUERY_POOL, q.as_bytes());
        let score = brain.best_binding_match_v2(QUERY_POOL).score();
        assert!(
            score >= 1.0,
            "taught fact {q:?} must score >= 1.0 after paging from store, got {score:.4}"
        );
        assert_eq!(
            brain
                .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
                .as_deref(),
            Some(&b"desk"[..]),
            "taught fact {q:?} must page in its binding and decode after idle"
        );

        let asked = format!("r{r:03} lamp on material?");
        let expected = format!("m{:03}", r % 7);
        let (derived, probes) = brain.derive_by_substitution_profiled(
            QUERY_POOL,
            ANSWER_POOL,
            asked.as_bytes(),
            3,
            DEFAULT_DERIVATION_PROBE_BUDGET,
        );
        assert_eq!(
            derived.as_deref(),
            Some(expected.as_bytes()),
            "derivation for {asked:?} failed after paging from store (spent {probes} probes)"
        );
    }

    drop(brain);
    std::fs::remove_dir_all(&dir).ok();
}

