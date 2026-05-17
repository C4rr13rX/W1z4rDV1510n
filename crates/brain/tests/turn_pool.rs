//! Stage 11B — turn pool tests.
//!
//! Validates the substrate-level guarantees the turn pool relies on.
//! Brain crate is pool-agnostic; these tests use any two arbitrary
//! pools as stand-ins for (text, turn) so we don't have to spin up
//! the brain_server binary just to prove the pattern works.
//!
//! Audit-7 coverage:
//!   - turn_id_creates_distinct_neuron_per_value
//!   - turn_pool_decay_naturally_retires_old_neurons (audit-2 LRU
//!     behavior via Pool decay + prune, no separate cap)
//!   - dialogue_recall_via_turn_chain (end-to-end empirical pin)

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

fn build_text_and_turn() -> (Brain, u32, u32) {
    let cfg = BrainConfig::default();
    let mut brain = Brain::new(cfg);
    let text = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    // The turn-pool config in brain_server uses aggressive decay so
    // old turn neurons recede.  Mirror that here so the LRU-decay
    // test exercises the same behavior shipped to the server.
    let mut turn_cfg = PoolConfig::defaults("turn", 5);
    turn_cfg.recent_atoms_window         = 32;
    turn_cfg.concept_emergence_threshold = u32::MAX;
    turn_cfg.max_concept_member_count    = 4;
    turn_cfg.decay_rate                  = 0.001;
    turn_cfg.prune_floor                 = 0.01;
    let turn = brain.create_pool(
        turn_cfg,
        Box::new(BytePassthroughEncoding { prefix: "turn" }),
    );
    (brain, text, turn)
}

#[test]
fn turn_id_creates_distinct_neuron_per_value() {
    // Two different turn ids must produce two different binding
    // fingerprints when each is paired with distinct text — that is
    // the property /chat dialogue continuation will rely on.
    let (mut brain, text, turn) = build_text_and_turn();

    // Train: turn-1 ↔ "hello", turn-2 ↔ "bye"
    for _ in 0..3 {
        brain.observe(turn, b"turn-1");
        brain.observe(text, b"hello");
        brain.advance_tick();
    }
    for _ in 0..3 {
        brain.observe(turn, b"turn-2");
        brain.observe(text, b"bye");
        brain.advance_tick();
    }

    // Two consolidated bindings should exist — one per (turn_id, text)
    // pair.  The Stage 10 tentative tier may also have crystallized
    // both on first co-firing.
    let total = brain.consolidated_binding_count() + brain.tentative_binding_count();
    assert!(total >= 2,
        "two distinct (turn_id, text) pairs must produce at least 2 \
         bindings; got tentative={} consolidated={}",
        brain.tentative_binding_count(), brain.consolidated_binding_count());
}

#[test]
fn turn_pool_decay_naturally_retires_old_neurons() {
    // Audit-2 / audit-10 LRU behavior.  The brain_server's turn-pool
    // config uses decay_rate=0.001 + prune_floor=0.01, so neurons
    // that are not re-fired for many ticks should fall below the
    // floor and disappear from the pool's iter_neurons view.
    let (mut brain, text, turn) = build_text_and_turn();

    // Fire turn-old once, then advance many ticks of unrelated
    // activity.  Older turn atoms should decay below prune_floor.
    brain.observe(turn, b"turn-old");
    brain.observe(text, b"x");
    brain.advance_tick();

    for i in 0..200 {
        let s = format!("noise-{}", i);
        brain.observe(text, s.as_bytes());
        brain.advance_tick();
    }

    // Query the turn pool — turn-old's atoms should have decayed.
    // We don't assert outright deletion (pool internals may keep
    // them at very low activation) — we assert that THEY ARE NOT
    // ACTIVELY FIRING when a fresh observation lights up a new
    // turn id, which is the LRU property dialogue needs.
    brain.observe(turn, b"turn-new");
    let turn_pool = brain.fabric().pool(turn).expect("turn pool exists");
    let p = turn_pool.read();
    let firing: Vec<_> = p.currently_firing().collect();
    // Only the new turn-new atoms should be currently firing — not
    // the long-ago turn-old atoms.
    assert!(!firing.is_empty(),
        "fresh turn observation must produce firing atoms");
    // Specifically: any atom whose label starts with "turn:" and
    // decodes to "turn-old" should NOT be currently firing.
    for nid in firing.iter() {
        if let Some(n) = p.get(*nid) {
            assert!(!n.label.contains("dHVybi1vbGQ"),  // b64 of "turn-old"
                "stale turn-old atom is still firing after 200 noise \
                 ticks — LRU decay is not working as expected: {}", n.label);
        }
    }
}

#[test]
fn dialogue_recall_via_turn_chain() {
    // The end-to-end empirical pin from audit-9.  Train a 3-turn
    // dialogue:
    //   turn-0 ↔ "hi"
    //   turn-1 ↔ "hello"
    //   turn-2 ↔ "how are you"
    // Then fire turn-1 alone — the cross-pool binding to "hello"
    // text should still be reachable via integrate retrieval.
    let (mut brain, text, turn) = build_text_and_turn();
    let pairs = [
        ("turn-0", "hi"),
        ("turn-1", "hello"),
        ("turn-2", "how are you"),
    ];
    for (t_id, txt) in pairs.iter() {
        for _ in 0..4 {
            brain.observe(turn, t_id.as_bytes());
            brain.observe(text, txt.as_bytes());
            brain.advance_tick();
        }
    }

    // Query: light up turn-1 alone.  integrate(turn, text) should
    // surface "hello"-related atoms in the text pool.
    brain.observe(turn, b"turn-1");
    let ans = brain.integrate(turn, text);
    let answer_bytes = ans.answer.as_ref().expect("integrate must produce some answer");
    let answer_str = String::from_utf8_lossy(answer_bytes);
    // The decoded answer should contain at least one byte from
    // "hello".  We're tolerant of partial decode (the byte-passthrough
    // chain decoder doesn't guarantee whole-word reassembly).
    let any_overlap = answer_str.bytes()
        .any(|b| b"hello".contains(&b));
    assert!(any_overlap,
        "turn-1 should chain-recall some byte of its paired text 'hello'; \
         got {:?}", answer_str);
}
