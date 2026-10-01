//! What did aligning the chat routes to the gated configuration COST?
//!
//! `crates/node/src/brain_api.rs` (`/brain/chat`) and
//! `crates/node/src/bin/brain_server.rs` (`/chat`) answered with
//! `integrate_autonomous(.., 0.0, 4, 200)` while the gated scorecard asks with
//! `(.., 100.0, 3, 200)`. Both now take `w1z4rd_brain::answer_path`, which is
//! the scorecard's values — a strictly TIGHTER fabric gate and a shallower
//! chain.
//!
//! Tightening a gate can only move an answer toward silence, so under the
//! owner's PRIORITY ZERO it cannot add a wrong answer. It CAN remove a correct
//! one, and that is a cost the owner has to be told rather than one this change
//! gets to absorb. So this file answers one trained world twice, at the old
//! node parameters and at the shared ones, and prints correct / wrong / silent
//! for both.
//!
//! # What is asserted and what is only printed
//!
//! Asserted:
//!   - the trained half is recalled under BOTH configurations, so a zero below
//!     is about integration and not about an empty brain. This is the
//!     invariant that makes the comparison mean anything, and it is the
//!     owner's absolute: recall is always 100%.
//!   - the shared configuration's WRONG count does not exceed the old node
//!     one's. That is the directional claim the change rests on, and it is the
//!     one thing here that can red.
//!
//! Printed, not asserted: the correct counts. Asserting "correct is N" would
//! freeze this probe set into the suite, and asserting "correct does not fall"
//! would be asserting a thing the change does not promise — a tighter gate may
//! legitimately cost a correct answer, and the owner's standard is that
//! invention is worse than silence.
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --test answer_path_alignment_cost -- --nocapture

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// What `/chat` and `/brain/chat` passed before this change.
const OLD_NODE: (f32, usize, usize) = (0.0, 4, 200);

/// Rooms. Small on purpose: this file compares two configurations on the same
/// brain, so the absolute scores do not need to be the scorecard's.
const ROOMS: usize = 24;

const OBJECTS: [&str; 4] = ["lamp", "desk", "chair", "paper"];
const MATERIALS: [&str; 4] = ["oak", "steel", "glass", "pine"];

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

/// `{room} lamp on? -> desk` and `{room} desk material? -> <m>`, so
/// `{room} lamp on material?` is answerable only by composing the two. The
/// same two-hop shape the scorecard's `on_material` family uses.
struct World {
    trained: Vec<(String, String)>,
    held_out: Vec<(String, String)>,
}

fn world() -> World {
    let mut trained = Vec::new();
    let mut held_out = Vec::new();
    for r in 0..ROOMS {
        let room = format!("r{r:03}");
        // 17 against a 4-long list is coprime, so the material actually varies
        // per room rather than taking two values -- the trap the scorecard
        // documents at `examples/scorecard.rs`.
        for (i, obj) in OBJECTS.iter().enumerate() {
            let m = MATERIALS[(r * 31 + (i + 1) * 17) % MATERIALS.len()];
            trained.push((format!("{room} {obj} material?"), m.to_string()));
        }
        trained.push((format!("{room} lamp on?"), "desk".to_string()));
        let desk_material = MATERIALS[(r * 31 + 2 * 17) % MATERIALS.len()];
        held_out.push((format!("{room} lamp on material?"), desk_material.to_string()));
    }
    World { trained, held_out }
}

fn train(brain: &mut Brain, facts: &[(String, String)]) {
    for _ in 0..2 {
        for (q, a) in facts {
            brain.pretrain_binding_episode(&[
                (QUERY_POOL, q.as_bytes().to_vec()),
                (ANSWER_POOL, a.as_bytes().to_vec()),
            ]);
            brain.eem_mut().induce_from_episode(q.as_bytes(), a.as_bytes());
        }
    }
}

/// The scorecard's `recall`: trained binding first, legacy integrate as the
/// fallback, then release what the query paged in.
fn recall(brain: &mut Brain, q: &str) -> String {
    brain.observe_read_only(QUERY_POOL, q.as_bytes());
    let legacy = brain.integrate(QUERY_POOL, ANSWER_POOL);
    let answer = brain
        .decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
        .or(legacy.answer);
    let _ = brain.finish_read_only_inference();
    String::from_utf8_lossy(&answer.unwrap_or_default()).into_owned()
}

/// The scorecard's `infer`, with the three parameters supplied so the same
/// brain can be asked both ways.
fn infer(brain: &mut Brain, q: &str, p: (f32, usize, usize)) -> String {
    brain.observe_read_only(QUERY_POOL, q.as_bytes());
    let answer = brain.integrate_autonomous(QUERY_POOL, ANSWER_POOL, p.0, p.1, p.2).answer;
    let _ = brain.finish_read_only_inference();
    String::from_utf8_lossy(&answer.unwrap_or_default()).into_owned()
}

#[derive(Default, Debug)]
struct Tally {
    correct: usize,
    wrong: usize,
    silent: usize,
}

fn score(brain: &mut Brain, held_out: &[(String, String)], p: (f32, usize, usize)) -> Tally {
    let mut t = Tally::default();
    for (q, want) in held_out {
        let got = infer(brain, q, p);
        if got.is_empty() {
            t.silent += 1;
        } else if got == *want {
            t.correct += 1;
        } else {
            t.wrong += 1;
        }
    }
    t
}

#[test]
fn the_shared_configuration_never_invents_more_than_the_old_node_one() {
    let w = world();
    let mut brain = subject();
    train(&mut brain, &w.trained);

    // The invariant that makes every number below mean anything. Checked
    // BEFORE the two inference passes, because inference is the phase that
    // pages bodies in and out, and a recall measured after it would be
    // measuring the release path as well.
    let mut recalled = 0usize;
    let mut misrecalled: Vec<String> = Vec::new();
    for (q, want) in &w.trained {
        let got = recall(&mut brain, q);
        if got == *want {
            recalled += 1;
        } else if misrecalled.len() < 5 {
            misrecalled.push(format!("{q} -> want `{want}` got `{got}`"));
        }
    }
    assert_eq!(
        recalled,
        w.trained.len(),
        "recall of trained material is the owner's absolute and it is always 100%. \
         {recalled}/{} recalled; first misses:\n  {}",
        w.trained.len(),
        misrecalled.join("\n  "),
    );

    let shared = (
        w1z4rd_brain::ANSWER_FABRIC_CONFIDENCE_THRESHOLD,
        w1z4rd_brain::ANSWER_CHAIN_MAX_DEPTH,
        w1z4rd_brain::ANSWER_CHAIN_MAX_VISIT,
    );
    let old = score(&mut brain, &w.held_out, OLD_NODE);
    let new = score(&mut brain, &w.held_out, shared);

    println!("\n  held-out two-hop probes: {}", w.held_out.len());
    println!(
        "    old node   (fabric {:>5}, depth {}, visit {})  correct {:>3}  wrong {:>3}  silent {:>3}",
        OLD_NODE.0, OLD_NODE.1, OLD_NODE.2, old.correct, old.wrong, old.silent,
    );
    println!(
        "    shared     (fabric {:>5}, depth {}, visit {})  correct {:>3}  wrong {:>3}  silent {:>3}",
        shared.0, shared.1, shared.2, new.correct, new.wrong, new.silent,
    );
    println!(
        "    delta: correct {:+}, wrong {:+}, silent {:+}\n",
        new.correct as i64 - old.correct as i64,
        new.wrong as i64 - old.wrong as i64,
        new.silent as i64 - old.silent as i64,
    );

    assert!(
        new.wrong <= old.wrong,
        "the shared configuration gates the fabric arm HARDER ({} vs {}), so it cannot invent \
         more than the old node one. It returned {} wrong against {} -- either the threshold is \
         not the gate this change assumes, or a tighter gate is routing answers to a looser arm.",
        shared.0,
        OLD_NODE.0,
        new.wrong,
        old.wrong,
    );
}
