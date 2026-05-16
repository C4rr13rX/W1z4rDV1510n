//! Sleep-cycle CLS replay tests per spec §6.3.
//!
//! sleep() is the consolidation pass that soft-prunes weak concept
//! neurons — emergence artifacts that never made it past
//! repeated-use confirmation.  Strong concepts (high use_count)
//! survive; weak ones get their terminals zeroed and become inert.

use w1z4rd_brain::{Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

fn build_brain_with_pool() -> (Brain, u32) {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = u32::MAX;
    let mut brain = Brain::new(cfg);
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window         = 4096;
    pc.concept_emergence_threshold = 3;
    pc.max_concept_member_count    = 8;
    pc.decay_rate                  = 0.00001;
    pc.prune_floor                 = 0.0005;
    let pool = brain.create_pool(pc,
        Box::new(BytePassthroughEncoding { prefix: "t" }));
    (brain, pool)
}

#[test]
fn sleep_prunes_concepts_below_use_count_threshold() {
    let (mut brain, pool) = build_brain_with_pool();

    // Train: emerge many concepts but only USE some of them via
    // recall.  Concepts that never get re-fired after emergence are
    // candidates for pruning.
    for _ in 0..6 {
        brain.observe(pool, b"ABCD");
        brain.advance_tick();
    }
    // At this point various concepts have emerged from sub-sequences.
    let pre = brain.stats();
    assert!(pre.total_concepts > 0);

    // Advance time so concepts age past stale_ticks.
    for _ in 0..50 { brain.advance_tick(); }

    let pruned = brain.sleep(/*min_use_count*/ 2, /*stale_ticks*/ 10);
    let post = brain.stats();

    // sleep returns the COUNT pruned; total_concepts in stats doesn't
    // shrink because we soft-prune (concepts remain in the Vec with
    // empty terminals).  But total_terminals must drop.
    assert!(pruned > 0,
        "at least one weak concept must have been pruned; got {}", pruned);
    assert!(post.total_terminals < pre.total_terminals,
        "post-sleep terminal count must be lower than pre-sleep ({} → {})",
        pre.total_terminals, post.total_terminals);
}

#[test]
fn sleep_preserves_the_highest_use_count_concept() {
    // Train varied data so multiple concepts emerge with a range of
    // use_counts.  The HIGHEST-use_count concept (whatever the
    // substrate found to be most-fired) must survive sleep when
    // min_use_count is set below its count.
    let (mut brain, pool) = build_brain_with_pool();
    for _ in 0..15 {
        brain.observe(pool, b"AB");
        brain.advance_tick();
        brain.observe(pool, b"CD");
        brain.advance_tick();
        brain.observe(pool, b"EF");
        brain.advance_tick();
    }

    // Find the concept with the highest use_count.
    let (champion_id, champion_use) = {
        let pool_arc = brain.fabric().pool(pool).unwrap();
        let p = pool_arc.read();
        let mut best: Option<(w1z4rd_brain::NeuronId, u64)> = None;
        for n in p.iter_neurons() {
            if n.is_atom() { continue; }
            if best.map_or(true, |(_, u)| n.use_count > u) {
                best = Some((n.id, n.use_count));
            }
        }
        best.expect("at least one concept must have emerged")
    };
    assert!(champion_use >= 1,
        "the most-used concept must have use_count >= 1; got {}",
        champion_use);

    // Sleep with min_use_count below the champion's count.  The
    // champion must survive.
    let min_use = (champion_use as u64).saturating_sub(1).max(1);
    brain.sleep(min_use, /*stale_ticks*/ 5);

    let pool_arc = brain.fabric().pool(pool).unwrap();
    let p = pool_arc.read();
    let champion = p.get(champion_id).unwrap();
    assert!(!champion.terminals.is_empty(),
        "champion concept (id={} use_count={}) must retain terminals after \
         sleep with min_use_count={}", champion_id, champion_use, min_use);
}
