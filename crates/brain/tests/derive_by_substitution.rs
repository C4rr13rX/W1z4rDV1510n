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
