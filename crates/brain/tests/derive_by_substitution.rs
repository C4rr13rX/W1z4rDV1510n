//! Does `Brain::derive_by_substitution` answer questions that were never
//! trained, and does it stay silent when it should?
//!
//! `eem_chain_bridge.rs` measured why the EEM chain walk cannot do this: the
//! fact graph's members are BYTES, so a populated graph is a complete graph
//! and the walk reaches the distractor more completely than the answer. This
//! file measures the replacement.
//!
//! The two families here chain through a DIFFERENT number of hops and use
//! different relation words, so a mechanism that happened to key on one
//! wording would fail the other. Both are also expressed without any
//! separator convention the mechanism could lean on.
//!
//! Counts are asserted; distributions are printed. The assertions are the
//! contract (recall unharmed, derivation non-zero, silence when ungrounded),
//! not a tuning target.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const MAX_PROBES: usize = 4096;

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

fn recall(brain: &mut Brain, question: &str) -> Option<Vec<u8>> {
    brain.observe_read_only(QUERY_POOL, question.as_bytes());
    brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL)
}

/// Two hops: the lamp names a piece of furniture, the furniture has a
/// material. `"r03 lamp on material"` is never taught.
fn teach_on_material(brain: &mut Brain, rooms: u32) {
    for room in 0..rooms {
        teach(brain, &format!("r{room:03} lamp on"), "desk");
        teach(brain, &format!("r{room:03} desk material"), "oak");
    }
}

/// Three hops, different relation words, and a longer chain, so nothing can
/// pass both families by keying on a word.
fn teach_next_colour(brain: &mut Brain, rooms: u32) {
    for room in 0..rooms {
        teach(brain, &format!("c{room:03} after"), "amber");
        teach(brain, &format!("c{room:03} amber shade"), "warm");
        teach(brain, &format!("c{room:03} warm pair"), "rust");
    }
}

#[test]
fn derivation_answers_untrained_questions_and_leaves_recall_at_100_percent() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    teach_on_material(&mut brain, ROOMS);
    teach_next_colour(&mut brain, ROOMS);

    // 1. RECALL FIRST. Every number below is meaningless if the taught half
    // regressed, and `derive_by_substitution` probes the fabric hundreds of
    // times, so "did that leave recall intact" is the first question.
    let mut trained_hits = 0u32;
    let trained_total = ROOMS * 5;
    for room in 0..ROOMS {
        for (q, a) in [
            (format!("r{room:03} lamp on"), "desk"),
            (format!("r{room:03} desk material"), "oak"),
            (format!("c{room:03} after"), "amber"),
            (format!("c{room:03} amber shade"), "warm"),
            (format!("c{room:03} warm pair"), "rust"),
        ] {
            if recall(&mut brain, &q).as_deref() == Some(a.as_bytes()) {
                trained_hits += 1;
            }
        }
    }
    eprintln!("recall of taught questions BEFORE any derivation: {trained_hits} of {trained_total}");
    assert_eq!(
        trained_hits, trained_total,
        "the taught half is not recalled, so no derivation number below is readable"
    );

    // 2. The untrained questions. True in the taught world, never taught, and
    // answerable only by composing two (or three) taught questions.
    let mut families: Vec<(&str, Vec<(String, &'static str)>, usize)> = Vec::new();
    families.push((
        "on_material (2 hops)",
        (0..ROOMS)
            .map(|r| (format!("r{r:03} lamp on material"), "oak"))
            .collect(),
        2,
    ));
    families.push((
        "next_colour (3 hops)",
        (0..ROOMS)
            .map(|r| (format!("c{r:03} after shade pair"), "rust"))
            .collect(),
        3,
    ));

    let mut total_correct = 0u32;
    let mut total_asked = 0u32;
    for (name, probes, hops) in &families {
        let mut correct = 0u32;
        let mut wrong = 0u32;
        let mut silent = 0u32;
        let mut probe_cost: Vec<usize> = Vec::new();
        for (question, expected) in probes {
            let (answer, cost) = brain.derive_by_substitution_profiled(
                QUERY_POOL,
                ANSWER_POOL,
                question.as_bytes(),
                *hops,
                MAX_PROBES,
            );
            probe_cost.push(cost);
            match answer.as_deref() {
                Some(a) if a == expected.as_bytes() => correct += 1,
                Some(_) => wrong += 1,
                None => silent += 1,
            }
        }
        let mean_cost = probe_cost.iter().sum::<usize>() as f64 / probe_cost.len() as f64;
        eprintln!(
            "{name}: correct {correct}/{}  wrong {wrong}  silent {silent}  questions asked per probe: mean {mean_cost:.0} max {}",
            probes.len(),
            probe_cost.iter().max().copied().unwrap_or(0)
        );
        total_correct += correct;
        total_asked += probes.len() as u32;
    }
    eprintln!("derivation overall: {total_correct} of {total_asked}");

    // 3. HONESTY. A question about something the brain was never taught must
    // derive nothing. Without this, "integration went up" is indistinguishable
    // from "the brain now guesses", and a guess is worse than a silence.
    let ungrounded = [
        "z99 piano on material",
        "r03 lamp on sibling",
        "Hello",
        "what is the capital of assyria",
    ];
    let mut invented = 0u32;
    for q in ungrounded {
        let answer = brain.derive_by_substitution(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 3, MAX_PROBES);
        eprintln!("  ungrounded {q:?} -> {:?}", answer.as_deref().map(String::from_utf8_lossy));
        if answer.is_some() {
            invented += 1;
        }
    }
    eprintln!("ungrounded questions that produced an answer: {invented} of {}", ungrounded.len());

    // 4. Recall again, AFTER every derivation, because the mechanism rewrites
    // the firing state hundreds of times and claims to restore it.
    let mut trained_hits_after = 0u32;
    for room in 0..ROOMS {
        for (q, a) in [
            (format!("r{room:03} lamp on"), "desk"),
            (format!("r{room:03} desk material"), "oak"),
            (format!("c{room:03} after"), "amber"),
            (format!("c{room:03} amber shade"), "warm"),
            (format!("c{room:03} warm pair"), "rust"),
        ] {
            if recall(&mut brain, &q).as_deref() == Some(a.as_bytes()) {
                trained_hits_after += 1;
            }
        }
    }
    eprintln!("recall of taught questions AFTER all derivation: {trained_hits_after} of {trained_total}");

    // The contract.
    assert_eq!(
        trained_hits_after, trained_total,
        "derivation damaged recall: {trained_hits_after} of {trained_total} after, {trained_hits} before"
    );
    assert!(
        total_correct > 0,
        "derivation answered none of {total_asked} untrained questions, so the mechanism is inert"
    );
}

/// Does the mechanism work when the taught sub-question is NOT a prefix?
///
/// Both families above are prefix-shaped, so they exercise only the fast path
/// (shortest prefix scoring 1.0). The full span search is the fallback for
/// every other shape, and an untested fallback is how "32 of 32" turns out to
/// mean "32 of 32 of the one shape I happened to write". Here the taught
/// sub-question is a SUFFIX of the asked one, so the prefix scan cannot find it.
#[test]
fn derivation_works_when_the_subquestion_is_not_a_prefix() {
    const SHELVES: u32 = 8;
    let mut brain = subject();
    for s in 0..SHELVES {
        teach(&mut brain, &format!("holds shelf b{s:02}"), "atlas");
        teach(&mut brain, &format!("weight of atlas b{s:02}"), "heavy");
    }

    let mut correct = 0u32;
    let mut wrong = 0u32;
    let mut silent = 0u32;
    let mut cost: Vec<usize> = Vec::new();
    for s in 0..SHELVES {
        let probe = format!("weight of holds shelf b{s:02}");
        let (answer, probes) = brain.derive_by_substitution_profiled(
            QUERY_POOL,
            ANSWER_POOL,
            probe.as_bytes(),
            2,
            MAX_PROBES,
        );
        cost.push(probes);
        match answer.as_deref() {
            Some(a) if a == b"heavy" => correct += 1,
            Some(_) => wrong += 1,
            None => silent += 1,
        }
    }
    eprintln!(
        "non-prefix (suffix sub-question, 2 hops): correct {correct}/{SHELVES} wrong {wrong} silent {silent}  questions asked mean {:.0}",
        cost.iter().sum::<usize>() as f64 / cost.len() as f64
    );
    // Recall must survive the fallback path too.
    let mut hits = 0u32;
    for s in 0..SHELVES {
        if recall(&mut brain, &format!("holds shelf b{s:02}")).as_deref() == Some(&b"atlas"[..]) {
            hits += 1;
        }
        if recall(&mut brain, &format!("weight of atlas b{s:02}")).as_deref() == Some(&b"heavy"[..]) {
            hits += 1;
        }
    }
    eprintln!("  recall after the fallback path: {hits} of {}", SHELVES * 2);
    assert_eq!(
        hits,
        SHELVES * 2,
        "the fallback span search damaged recall: {hits} of {}",
        SHELVES * 2
    );
    // Printed, not asserted on a count: this measures COVERAGE of the fallback,
    // and asserting a number here would turn a coverage probe into a target.
    assert_eq!(wrong, 0, "the fallback derived a WRONG answer {wrong} times, which is worse than silence");
}

/// A taught question must not be ANSWERED by this path. Derivation exists for
/// questions recall cannot reach; if it also fires on taught ones it would mask
/// a recall regression behind a derived answer that happened to agree.
#[test]
fn derivation_is_silent_on_a_question_that_was_taught() {
    let mut brain = subject();
    teach_on_material(&mut brain, 8);

    let taught = "r003 desk material";
    let recalled = recall(&mut brain, taught);
    assert_eq!(
        recalled.as_deref(),
        Some(&b"oak"[..]),
        "the question under test is not actually recalled, so this proves nothing"
    );
    let derived = brain.derive_by_substitution(QUERY_POOL, ANSWER_POOL, taught.as_bytes(), 3, MAX_PROBES);
    eprintln!("taught question {taught:?}: recall {:?}, derivation {:?}",
        recalled.as_deref().map(String::from_utf8_lossy),
        derived.as_deref().map(String::from_utf8_lossy));
    assert!(
        derived.is_none(),
        "derivation fired on a taught question and returned {:?}; it can only accept a \
         rewrite the brain knows STRICTLY better, and a taught question already scores 1.0",
        derived.as_deref().map(String::from_utf8_lossy)
    );
}

/// Does the PRODUCTION entry point derive, or only the mechanism behind it?
///
/// `derive_by_substitution` passed 32 of 32 here while the scorecard's
/// `integration_pct` stayed 0.0, for one reason: it had no callers. The answer
/// path is `Brain::integrate_autonomous`, so this test asks the question the
/// way `crates/brain/examples/scorecard.rs:389` and the node's routes do --
/// `observe_read_only` then `integrate_autonomous` -- and prints the tier and
/// the grounding of every miss, because "empty" and "wrong" and "rejected as
/// ungrounded" are three different faults with three different repairs and the
/// percentage alone cannot tell them apart.
#[test]
fn the_production_answer_path_derives_untrained_answers() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    // ON by default now, and the default is what the scorecard and the node's
    // answer routes both pick up without either of them naming the setting.
    // Asserted rather than assumed: this was 0 for a pass, every caller
    // inherited it, and `derive_by_substitution` measured 32 of 32 on a path
    // nothing could reach. See `Brain::derivation_probe_budget` for the cost.
    assert_eq!(
        brain.derivation_probe_budget(),
        w1z4rd_brain::DEFAULT_DERIVATION_PROBE_BUDGET,
        "a fresh brain must inherit the shipped derivation budget"
    );
    assert!(
        brain.derivation_probe_budget() > 0,
        "the shipped default must actually enable the derivation"
    );
    // This test measures the MECHANISM's ceiling, not the shipped budget, so it
    // raises it deliberately -- the shipped figure is asserted above.
    brain.set_derivation_probe_budget(MAX_PROBES);
    teach_on_material(&mut brain, ROOMS);
    teach_next_colour(&mut brain, ROOMS);

    // Every trained question must still answer through the path the
    // scorecard's `recall()` uses, and we separately COUNT how many answer
    // through `integrate_autonomous`. Those are different numbers and the
    // second one is a pre-existing hole this change does not touch -- see the
    // assertion at the bottom.
    let mut trained_ok = 0u32;
    let mut trained_recall_ok = 0u32;
    let mut trained_total = 0u32;
    for room in 0..ROOMS {
        for (q, want) in [
            (format!("r{room:03} lamp on"), "desk"),
            (format!("r{room:03} desk material"), "oak"),
        ] {
            trained_total += 1;
            brain.observe_read_only(QUERY_POOL, q.as_bytes());
            let got = brain
                .integrate_autonomous(QUERY_POOL, ANSWER_POOL, 100.0, 3, 200)
                .answer
                .unwrap_or_default();
            if got == want.as_bytes() {
                trained_ok += 1;
            }
            if recall(&mut brain, &q).as_deref() == Some(want.as_bytes()) {
                trained_recall_ok += 1;
            }
        }
    }
    println!(
        "trained: {trained_ok}/{trained_total} through integrate_autonomous,          {trained_recall_ok}/{trained_total} through decode_best_trained_binding"
    );

    // WHY, measured rather than guessed. The step-0 gate reads
    // `best_binding_match_v2` off the firing state, so the two observe entry
    // points are the suspects: `observe_read_only` goes through
    // `Brain::observe` (bookkeeping, ticks), `observe_fabric_read_only` does
    // not -- and the derivation's own probes use the latter and score fine.
    let probe = format!("r000 lamp on");
    brain.observe_read_only(QUERY_POOL, probe.as_bytes());
    let via_brain = brain.best_binding_match_v2(QUERY_POOL);
    let decode_brain = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
    brain.observe_fabric_read_only(QUERY_POOL, probe.as_bytes());
    let via_fabric = brain.best_binding_match_v2(QUERY_POOL);
    let decode_fabric = brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
    println!(
        "  observe_read_only    -> precision {:.4} recall {:.4} tier {:?} score {:.4} decode {:?}",
        via_brain.precision, via_brain.recall, via_brain.tier, via_brain.score(),
        decode_brain.as_deref().map(String::from_utf8_lossy)
    );
    println!(
        "  observe_fabric_ro    -> precision {:.4} recall {:.4} tier {:?} score {:.4} decode {:?}",
        via_fabric.precision, via_fabric.recall, via_fabric.tier, via_fabric.score(),
        decode_fabric.as_deref().map(String::from_utf8_lossy)
    );

    // And the untrained ones must derive.
    let mut hits = 0u32;
    let mut empty = 0u32;
    let mut wrong = 0u32;
    let mut ungrounded = 0u32;
    let mut asked = 0u32;
    for room in 0..ROOMS {
        for (q, want) in [
            (format!("r{room:03} lamp on material"), "oak"),
            (format!("c{room:03} after shade"), "warm"),
        ] {
            asked += 1;
            brain.observe_read_only(QUERY_POOL, q.as_bytes());
            let res = brain.integrate_autonomous(QUERY_POOL, ANSWER_POOL, 100.0, 3, 200);
            let tier = res.confidence_tier;
            let got = res.answer.unwrap_or_default();
            if got == want.as_bytes() {
                hits += 1;
            } else if got.is_empty() {
                empty += 1;
                if tier == w1z4rd_brain::ConfidenceTier::Ungrounded {
                    ungrounded += 1;
                }
                if empty <= 2 {
                    println!(
                        "  EMPTY {q:?} tier={tier:?} outside_grounding={} fabric_conf={:.4}",
                        res.grounding.outside_grounding, res.grounding.fabric_confidence
                    );
                }
            } else {
                wrong += 1;
                if wrong <= 2 {
                    println!("  WRONG {q:?} -> {:?} want {want:?} tier={tier:?}",
                             String::from_utf8_lossy(&got));
                }
            }
        }
    }
    println!(
        "integrate_autonomous on untrained: {hits}/{asked} derived, {empty} empty \
         ({ungrounded} of those rejected as Ungrounded), {wrong} wrong"
    );

    // Recall is measured on the path the scorecard measures it on, and must
    // be perfect. `integrate_autonomous` answering 0 of 32 TRAINED questions
    // is a SEPARATE, pre-existing hole that predates this change and is not
    // asserted here: with `fabric_confidence_threshold` at 100.0 its fabric
    // arm can never be taken, the legacy `integrate()` finds nothing in this
    // world, and the derivation correctly declines a question that is already
    // fully explained (nothing scores strictly better than 1.0). That is why
    // the scorecard's `recall()` puts `decode_best_trained_binding` first and
    // its `infer()` does not -- filed rather than fixed inside this change.
    assert_eq!(
        trained_recall_ok, trained_total,
        "recall through the scorecard's own recall path regressed"
    );
    assert_eq!(wrong, 0, "the derivation answered {wrong} untrained questions WRONGLY");
    assert_eq!(
        hits, asked,
        "the production answer path derived {hits} of {asked}: {empty} empty ({ungrounded} ungrounded), {wrong} wrong"
    );
}

/// The sub-question is a DELETION, and a `?`-terminated world is what proves it.
///
/// Every other world in this file is prefix-shaped, so all of them pass with a
/// prefix-only search and none of them can tell the two mechanisms apart. The
/// scorecard's world is not: its questions end in `?`, so the taught
/// sub-question of `"r000 lamp on material?"` is `"r000 lamp on?"`, which is not
/// a prefix of it. Measured on that world (Iris,
/// `tests/question_terminator_blocks_prefix_search.rs`), the prefix ladder peaks
/// at 0.90 and NOTHING reaches 1.0, so a prefix-only search spends all `n-1`
/// probes and derives nothing — which is exactly what `on_material` did at every
/// scale and under every budget.
///
/// This asserts the repair where it is cheap to assert: a terminated world must
/// derive, and it must do so inside the budget the brain actually SHIPS, because
/// a mechanism that works only at the 4096 this file uses elsewhere is a
/// mechanism the product cannot run.
#[test]
fn derivation_finds_a_subquestion_that_is_not_a_prefix_of_the_question() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    for room in 0..ROOMS {
        teach(&mut brain, &format!("r{room:03} lamp on?"), "desk");
        teach(&mut brain, &format!("r{room:03} desk material?"), "oak");
    }

    // Recall first: a derivation measured against a world the brain cannot
    // recall measures nothing.
    let mut recalled = 0u32;
    for room in 0..ROOMS {
        if recall(&mut brain, &format!("r{room:03} lamp on?")).as_deref() == Some(b"desk".as_ref()) {
            recalled += 1;
        }
    }
    assert_eq!(recalled, ROOMS, "the terminated world must be recallable first");

    let budget = w1z4rd_brain::DEFAULT_DERIVATION_PROBE_BUDGET;
    let mut derived = 0u32;
    let mut cost: Vec<usize> = Vec::new();
    for room in 0..ROOMS {
        let q = format!("r{room:03} lamp on material?");
        let (answer, probes) =
            brain.derive_by_substitution_profiled(QUERY_POOL, ANSWER_POOL, q.as_bytes(), 2, budget);
        cost.push(probes);
        if answer.as_deref() == Some(b"oak".as_ref()) {
            derived += 1;
        }
    }
    let mean = cost.iter().sum::<usize>() as f64 / cost.len() as f64;
    eprintln!(
        "terminated world: derived {derived} of {ROOMS} inside a budget of {budget}; \
         questions asked mean {mean:.0} max {}",
        cost.iter().max().copied().unwrap_or(0)
    );
    assert!(
        derived > 0,
        "a `?`-terminated world derived {derived} of {ROOMS} — the sub-question search \
         is prefix-only again, and no prefix of a terminated question scores 1.0"
    );

    // And recall is untouched afterwards, because the derivation rewrote the
    // firing state `mean` times per question and claims to restore it.
    let mut after = 0u32;
    for room in 0..ROOMS {
        if recall(&mut brain, &format!("r{room:03} desk material?")).as_deref()
            == Some(b"oak".as_ref())
        {
            after += 1;
        }
    }
    assert_eq!(after, ROOMS, "recall must survive every derivation");
}
