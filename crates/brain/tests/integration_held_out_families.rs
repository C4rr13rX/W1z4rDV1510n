//! Can integration only rise through a GENERAL mechanism?
//!
//! The scorecard's integration number is 77.3 % at scale 64, and every probe it
//! asks is built by one generator from one wording template. That is enough to
//! measure whether a chain composes; it is not enough to tell a general
//! mechanism from one that fits the template. Three things would move the
//! scorecard without any generalisation at all: wording the question the same
//! way every time, never putting a second chain in the brain that the question
//! has to choose between, and never asking a relation the generator did not
//! emit.
//!
//! So this file asks the same two-hop question four ways on the same world and
//! prints correct / WRONG / silent for each. It is deliberately NOT in the gated
//! scorecard: adding a harder family there would lower `integration_pct` and
//! trip the gate's own ratchet, which is a re-baseline decision and not a thing
//! to slip in beside a fix. It is a measurement the lead owns, run on demand.
//!
//! The one thing asserted is PRIORITY ZERO, and it is the owner's standard
//! rather than this file's: **a wrong answer is worse than silence**, so `wrong`
//! must be 0 in every condition. A condition that answers nothing passes. A
//! condition that invents does not. Recall is asserted too, so a column of
//! zeros cannot be read as "the brain is empty".
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --release --test integration_held_out_families -- --nocapture

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
/// Two scales, and the second one exists because of a measurement Cove made on
/// a different mechanism the same day: a transfer rule scored 6 right / 0 wrong
/// at 8 rooms and 4 right / 3 WRONG at 32, for the same family at the same hop
/// count. More trained questions are reachable at the ceiling, so a chain that
/// is unique in a small world stops being unique in a larger one. A single-scale
/// honesty table is therefore a HYPOTHESIS about the next scale, and this file's
/// own "the distractor does not fool it" result would be exactly that if it were
/// only measured at 8.
const SCALES: [u32; 2] = [8, 32];

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

fn teach(brain: &mut Brain, q: &str, a: &str) {
    brain.pretrain_binding_episode(&[
        (QUERY_POOL, q.as_bytes().to_vec()),
        (ANSWER_POOL, a.as_bytes().to_vec()),
    ]);
}

/// Exactly the scorecard's `infer()` in `crates/brain/examples/scorecard.rs`:
/// `observe_read_only`, `integrate_autonomous` on the shared `answer_path`
/// constants, then release what the query paged in. Copied in shape rather than
/// re-tuned, because a measurement taken on a different path would not be about
/// the number the gate reports.
fn infer(brain: &mut Brain, q: &str) -> Vec<u8> {
    brain.observe_read_only(QUERY_POOL, q.as_bytes());
    let answer = brain
        .integrate_autonomous(
            QUERY_POOL,
            ANSWER_POOL,
            w1z4rd_brain::ANSWER_FABRIC_CONFIDENCE_THRESHOLD,
            w1z4rd_brain::ANSWER_CHAIN_MAX_DEPTH,
            w1z4rd_brain::ANSWER_CHAIN_MAX_VISIT,
        )
        .answer;
    let _ = brain.finish_read_only_inference();
    answer.unwrap_or_default()
}

#[derive(Default)]
struct Verdict {
    correct: u32,
    wrong: u32,
    silent: u32,
}

impl Verdict {
    fn record(&mut self, got: &[u8], want: &str) {
        if got.is_empty() {
            self.silent += 1;
        } else if got == want.as_bytes() {
            self.correct += 1;
        } else {
            self.wrong += 1;
        }
    }
    fn total(&self) -> u32 {
        self.correct + self.wrong + self.silent
    }
}

/// The world the scorecard builds, in miniature: two facts per room that chain.
fn teach_base_world(brain: &mut Brain, rooms: u32) {
    for room in 0..rooms {
        teach(brain, &format!("r{room:03} lamp on"), "desk");
        teach(brain, &format!("r{room:03} desk material"), "oak");
    }
}

/// One scale's worth of the table. Called once per entry in `SCALES`, so an
/// assertion that fires names the scale it fired at.
fn one_scale(rooms: u32) {
    // Condition A -- the scorecard's own wording. The control: without it a
    // zero in B, C or D is unreadable.
    let mut a_brain = subject();
    teach_base_world(&mut a_brain, rooms);
    let mut a = Verdict::default();
    for room in 0..rooms {
        let got = infer(&mut a_brain, &format!("r{room:03} lamp on material"));
        a.record(&got, "oak");
    }

    // Condition B -- PARAPHRASE. Same world, same answer, wording the generator
    // never emits and the brain was never taught. An atom is a byte, so this
    // shares most of its bytes with the trained questions and none of their
    // order: exactly the case where a mechanism that fits the template and one
    // that generalises come apart.
    let mut b_brain = subject();
    teach_base_world(&mut b_brain, rooms);
    let mut b = Verdict::default();
    for room in 0..rooms {
        let got = infer(&mut b_brain, &format!("material of the lamp in r{room:03}"));
        b.record(&got, "oak");
    }

    // Condition C -- DISTRACTOR CHAIN. A second complete chain per room, true
    // and irrelevant, sharing the room id. Now the question must pick WHICH
    // chain, and a mechanism that composes whatever it can reach will answer
    // `wool` -- which is the hallucination PRIORITY ZERO forbids, not a miss.
    let mut c_brain = subject();
    teach_base_world(&mut c_brain, rooms);
    for room in 0..rooms {
        teach(&mut c_brain, &format!("r{room:03} chair on"), "rug");
        teach(&mut c_brain, &format!("r{room:03} rug material"), "wool");
    }
    let mut c = Verdict::default();
    for room in 0..rooms {
        let got = infer(&mut c_brain, &format!("r{room:03} lamp on material"));
        c.record(&got, "oak");
    }

    // Condition D -- HELD-OUT RELATION. The same two-hop shape on a relation
    // the base world never used. Taught only as single facts, never composed.
    let mut d_brain = subject();
    for room in 0..rooms {
        teach(&mut d_brain, &format!("r{room:03} clock under"), "shelf");
        teach(&mut d_brain, &format!("r{room:03} shelf material"), "pine");
    }
    let mut d = Verdict::default();
    for room in 0..rooms {
        let got = infer(&mut d_brain, &format!("r{room:03} clock under material"));
        d.record(&got, "pine");
    }

    let conditions: [(&str, &Verdict, &str); 4] = [
        ("A scorecard wording", &a, "control: the family the gate measures"),
        ("B paraphrase", &b, "held out: wording never taught"),
        ("C distractor chain", &c, "held out: two chains, must choose"),
        ("D held-out relation", &d, "held out: relation never composed"),
    ];
    eprintln!("{rooms} rooms, the same two-hop question asked four ways");
    eprintln!("  {:<22} {:>7} {:>7} {:>7}   {}", "condition", "correct", "WRONG", "silent", "what it holds out");
    for (name, v, why) in &conditions {
        eprintln!(
            "  {name:<22} {:>7} {:>7} {:>7}   {why}",
            v.correct, v.wrong, v.silent
        );
    }

    // Non-vacuity: the trained half of each world is recalled, so a column of
    // zeros is about integration and not about an empty brain.
    for (name, brain, q, want) in [
        ("A", &mut a_brain, format!("r000 lamp on"), "desk"),
        ("B", &mut b_brain, format!("r000 lamp on"), "desk"),
        ("C", &mut c_brain, format!("r000 lamp on"), "desk"),
        ("D", &mut d_brain, format!("r000 clock under"), "shelf"),
    ] {
        brain.observe_read_only(QUERY_POOL, q.as_bytes());
        let recalled = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
        let _ = brain.finish_read_only_inference();
        assert_eq!(
            recalled.as_deref(),
            Some(want.as_bytes()),
            "scale {rooms}, condition {name}: the trained question {q:?} is not recalled, so no count above is \
             about integration"
        );
    }

    // PRIORITY ZERO, the only behavioural assertion here. Silence is a pass.
    let inventing: Vec<&str> = conditions
        .iter()
        .filter(|(_, v, _)| v.wrong > 0)
        .map(|(name, _, _)| *name)
        .collect();
    assert!(
        inventing.is_empty(),
        "scale {rooms}: condition(s) {inventing:?} returned a WRONG answer where silence was available. \
         The owner's standard is that the brain is never wrong: when it has no grounded answer \
         it has NO answer. See the table above for the counts"
    );

    // The control must do something, or every held-out zero below it is
    // explained by the control being broken rather than by generalisation.
    assert!(
        a.correct > 0,
        "the control condition answered nothing correct ({}/{}), so this file measures nothing",
        a.correct,
        a.total()
    );
}

/// WHICH relaxation could reach the paraphrase gap, priced before either is
/// built.
///
/// `held_out_families_never_invent_an_answer` measures paraphrase at 0 of 8 and
/// names the mechanism: the derivation substitutes over byte SPANS — a prefix
/// plus a suffix — of the question as asked, so a rewording offers it no taught
/// sub-question to splice. Two relaxations are the obvious candidates, and they
/// cost very different amounts of work:
///
///   1. ORDERED SUBSEQUENCE. Keep order, drop contiguity: accept a taught
///      question whose bytes appear in the paraphrase in the same order. This is
///      a change to one matcher and keeps the ordering guarantee PRIORITY ZERO
///      rests on.
///   2. UNORDERED CONTAINMENT. Accept a taught question whose bytes are a subset
///      of the paraphrase's, order ignored. This is what `BindingMatch` already
///      scores, and `16bc51b` records what it costs: an atom is a byte, so a set
///      represents neither order nor multiplicity and every anagram scores 1.0.
///
/// This lab's most expensive recurring mistake is shipping a guard keyed on
/// evidence the host cannot produce, so the question is not which is nicer but
/// whether either can reach a single one of the 8 failures. Both are measured
/// here against the real taught questions and the real paraphrases, in plain
/// code — no brain call, so nothing can be confounded by budgets or thresholds.
///
/// Printed, not asserted on its values: these are readings that name the next
/// change, and freezing them would freeze the wording of this file's probes.
/// What IS asserted is that the census is not vacuous.
#[test]
fn held_out_families_never_invent_an_answer() {
    for rooms in SCALES {
        one_scale(rooms);
    }
}

/// Also run at both scales: the count of taught questions a paraphrase matches
/// is exactly the quantity that grows with the world, so the ambiguity column is
/// meaningless at one scale.
#[test]
fn which_relaxation_could_reach_the_paraphrase_gap() {
    for rooms in SCALES {
        relaxation_census(rooms);
    }
}

fn relaxation_census(rooms: u32) {
    /// Does `needle` appear in `hay` in order, not necessarily contiguously?
    fn ordered_subsequence(needle: &[u8], hay: &[u8]) -> bool {
        let mut it = hay.iter();
        needle.iter().all(|b| it.any(|h| h == b))
    }
    /// Is every byte of `needle` present somewhere in `hay`? Order and
    /// multiplicity ignored — exactly what a set-scored binding match sees.
    fn unordered_contained(needle: &[u8], hay: &[u8]) -> bool {
        needle.iter().all(|b| hay.contains(b))
    }

    let taught: Vec<String> = (0..rooms)
        .flat_map(|r| [format!("r{r:03} lamp on"), format!("r{r:03} desk material")])
        .collect();

    let mut subseq_reachable = 0u32;
    let mut contained_reachable = 0u32;
    let mut contained_ambiguous = 0u32;
    for room in 0..rooms {
        let paraphrase = format!("material of the lamp in r{room:03}");
        let p = paraphrase.as_bytes();
        let subseq: Vec<&String> = taught
            .iter()
            .filter(|t| ordered_subsequence(t.as_bytes(), p))
            .collect();
        let contained: Vec<&String> = taught
            .iter()
            .filter(|t| unordered_contained(t.as_bytes(), p))
            .collect();
        if !subseq.is_empty() {
            subseq_reachable += 1;
        }
        if !contained.is_empty() {
            contained_reachable += 1;
        }
        if contained.len() > 1 {
            contained_ambiguous += 1;
        }
        if room == 0 {
            eprintln!("  paraphrase {paraphrase:?}");
            eprintln!("    ordered-subsequence matches: {} {:?}", subseq.len(), subseq);
            eprintln!(
                "    unordered-containment matches: {} (first 4: {:?})",
                contained.len(),
                contained.iter().take(4).collect::<Vec<_>>()
            );
        }
    }

    eprintln!("over {rooms} paraphrases against {} taught questions:", taught.len());
    eprintln!("  ordered subsequence   reaches {subseq_reachable}/{rooms} paraphrases");
    eprintln!("  unordered containment reaches {contained_reachable}/{rooms}, of which {contained_ambiguous} match MORE THAN ONE taught question");

    // Non-vacuity: the paraphrases and the taught questions are the ones the
    // test above uses, so if NOTHING matched under either rule the census would
    // be measuring a typo rather than a mechanism. Unordered containment is
    // guaranteed to match something here -- the paraphrase contains the room id
    // and the word `lamp` -- so a zero in that column means the helper is wrong.
    assert!(
        contained_reachable > 0,
        "unordered containment reached none of {rooms} paraphrases, which cannot be true of a \
         paraphrase that contains the room id and the subject -- the census helper is wrong, so \
         neither column above means anything"
    );
}
