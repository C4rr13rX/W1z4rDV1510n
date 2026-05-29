//! End-to-end test proving the spec §1.3 property:
//! axons grow toward dendrites across pools because of co-temporal firing,
//! and subsequent activation of one pool's neuron propagates into the
//! other pool's neuron via the grown terminal.
//!
//! This is the architectural breakthrough vs. the existing `MultiPoolFabric`
//! (which kept cross-pool wiring in a separate routing table and required
//! explicit `multi_pool_train_pair` calls).  Here wiring is per-neuron and
//! emerges automatically from the moment buffer.

use w1z4rd_brain::{
    BytePassthroughEncoding, Fabric, FabricConfig, Pool, PoolConfig,
};

#[test]
fn co_temporal_firing_grows_cross_pool_terminals_and_enables_recall() {
    // Two pools at the same tick frequency.  Pool A = "text", pool B =
    // "audio-like" — both use byte-passthrough encoding so we can verify
    // the wiring behaviour without dragging real codec adapters in.
    let mut fabric = Fabric::new(FabricConfig::default());

    let mut cfg_a = PoolConfig::defaults("pool_a", 1);
    cfg_a.concept_emergence_threshold = 100;  // disable concept emergence
                                              // for this test — we're
                                              // measuring atom-level
                                              // wiring only.
    let mut cfg_b = PoolConfig::defaults("pool_b", 2);
    cfg_b.concept_emergence_threshold = 100;

    let pool_a = Pool::new(cfg_a, Box::new(BytePassthroughEncoding { prefix: "a" }));
    let pool_b = Pool::new(cfg_b, Box::new(BytePassthroughEncoding { prefix: "b" }));
    let a_id = fabric.register_pool(pool_a);
    let b_id = fabric.register_pool(pool_b);

    // Tick 0: observe co-temporal input into both pools.  Pool A sees
    // byte 'X'; pool B sees byte 'Y'.  These fire at the same tick.
    // The fabric's `advance_tick` will then close the moment and wire
    // axon terminals between every (A-fired, B-fired) pair.
    let fired_a = fabric.observe(a_id, b"X");
    let fired_b = fabric.observe(b_id, b"Y");
    assert_eq!(fired_a.len(), 1, "pool A should fire one atom for byte X");
    assert_eq!(fired_b.len(), 1, "pool B should fire one atom for byte Y");

    let x_id = fired_a[0];
    let y_id = fired_b[0];

    // Repeat the co-temporal presentation a few ticks to build weight
    // above the prune floor.  cross_pool_lr defaults to 0.15 → 5 ticks
    // gets us to weight ~0.75 (well above prune_floor 0.01).
    for _ in 0..5 {
        fabric.advance_tick();
        fabric.observe(a_id, b"X");
        fabric.observe(b_id, b"Y");
    }
    fabric.advance_tick();  // close the final moment.

    // Verify the cross-pool axon terminal grew.
    let pool_a_arc = fabric.pool(a_id).unwrap();
    let x_neuron_terminals = {
        let p = pool_a_arc.read();
        p.get(x_id).unwrap().terminals.clone()
    };
    let cross_terminal = x_neuron_terminals.iter()
        .find(|t| t.target.pool == b_id && t.target.neuron == y_id)
        .expect("X in pool A should have grown a terminal to Y in pool B");
    assert!(cross_terminal.weight > 0.1,
        "cross-pool terminal weight should accumulate across co-temporal firings, got {}",
        cross_terminal.weight,
    );
    assert!(cross_terminal.consolidation >= 4,
        "consolidation should track distinct-tick reinforcement events (>= 4 after 5 co-firings), got {}",
        cross_terminal.consolidation,
    );

    // And symmetrically: Y in pool B should have a terminal toward X.
    // Cross-pool wiring is bidirectional per spec — when A and B co-fire,
    // BOTH directions strengthen.
    let pool_b_arc = fabric.pool(b_id).unwrap();
    let y_neuron_terminals = {
        let p = pool_b_arc.read();
        p.get(y_id).unwrap().terminals.clone()
    };
    let reverse_terminal = y_neuron_terminals.iter()
        .find(|t| t.target.pool == a_id && t.target.neuron == x_id)
        .expect("Y in pool B should have grown a terminal back to X in pool A");
    assert!(reverse_terminal.weight > 0.1,
        "reverse cross-pool terminal weight should accumulate, got {}",
        reverse_terminal.weight,
    );

    // The functional test: now fire X in pool A only, and verify that
    // propagation activates Y in pool B via the cross-pool terminal.
    // This is the cross-modal generative-composition primitive — stimulating
    // text-pool with "X" should activate audio-pool's Y because they were
    // observed together.
    fabric.advance_tick();
    fabric.observe(a_id, b"X");
    let propagated = fabric.propagate(a_id);

    let pool_b_activation = propagated.get(&b_id)
        .expect("propagation should reach pool B");
    let y_activation = pool_b_activation.get(&y_id).copied().unwrap_or(0.0);
    assert!(y_activation > 0.0,
        "Y in pool B should fire when X in pool A is stimulated alone (got activation {})",
        y_activation,
    );

    // And the pool that wasn't stimulated independently should NOT fire
    // unrelated atoms.  Confirm no spurious activation in pool B beyond
    // the cross-wired target.
    let other_b_active: Vec<_> = pool_b_activation.iter()
        .filter(|&(&nid, _)| nid != y_id)
        .collect();
    assert!(other_b_active.is_empty() || other_b_active.iter().all(|&(_, &a)| a < 0.001),
        "no spurious cross-pool activation expected; got {:?}",
        other_b_active,
    );
}

#[test]
fn concepts_emerge_from_repeating_sequences_without_explicit_promotion() {
    // Spec §4.A: when the same atom sequence has fired N times in
    // recent_atoms, a concept neuron is promoted automatically.  No
    // `multi_pool_train_pair` call — emergence is automatic.

    let mut fabric = Fabric::new(FabricConfig::default());
    let mut cfg = PoolConfig::defaults("text", 1);
    cfg.concept_emergence_threshold = 3;
    cfg.recent_atoms_window = 16;
    let pool = Pool::new(cfg, Box::new(BytePassthroughEncoding { prefix: "txt" }));
    let pool_id = fabric.register_pool(pool);

    let baseline_concepts = fabric.pool(pool_id).unwrap().read().concept_count();
    assert_eq!(baseline_concepts, 0, "no concepts before observation");

    // Observe the same 2-byte sequence three times.  After the third,
    // the [a, b] run's count crosses the threshold and a concept is born.
    for _ in 0..3 {
        fabric.observe(pool_id, b"ab");
        fabric.advance_tick();
    }

    let final_concepts = fabric.pool(pool_id).unwrap().read().concept_count();
    assert!(final_concepts >= 1,
        "at least one concept should emerge from 3 repetitions of 'ab', got {}",
        final_concepts,
    );

    // The concept's members should be the atoms for 'a' and 'b' in order.
    let pool_arc = fabric.pool(pool_id).unwrap();
    let pool = pool_arc.read();
    let concept = pool.iter_neurons()
        .find(|n| !n.is_atom())
        .expect("concept neuron should exist");
    assert_eq!(concept.members.len(), 2,
        "emergent concept should have 2 members (one per atom in sequence)",
    );
    let a_atom = pool.label_to_id("txt:YQ").expect("txt:YQ atom for 'a'");
    let b_atom = pool.label_to_id("txt:Yg").expect("txt:Yg atom for 'b'");
    assert_eq!(concept.members[0].neuron, a_atom,
        "first member should be the atom for 'a' (order preserved per spec §1.1)",
    );
    assert_eq!(concept.members[1].neuron, b_atom,
        "second member should be the atom for 'b'",
    );

    // And the atoms should have terminals back to the concept (atom→
    // concept Hebbian wiring established at promotion).
    let a_neuron = pool.get(a_atom).unwrap();
    let target_concept = a_neuron.terminals.iter()
        .find(|t| t.target.pool == pool_id && t.target.neuron == concept.id);
    assert!(target_concept.is_some(),
        "atom 'a' should have an atom→concept terminal to the emergent concept",
    );
}

#[test]
fn synapse_decay_prunes_unused_terminals() {
    // Spec §1.5: every tick decays terminals by epsilon; below floor they
    // get pruned.  This is the only forgetting mechanism.

    let mut fabric = Fabric::new(FabricConfig::default());
    let mut cfg = PoolConfig::defaults("text", 1);
    cfg.decay_rate = 0.4;        // aggressive but not so aggressive that a
                                 // freshly-wired terminal gets pruned on
                                 // its first housekeeping pass.
    cfg.prune_floor = 0.02;      // low enough that wiring at lr=0.15
                                 // saturates above floor across multiple
                                 // co-fires before decay catches up.
    cfg.concept_emergence_threshold = 100;
    let pool = Pool::new(cfg, Box::new(BytePassthroughEncoding { prefix: "txt" }));
    let pool_id = fabric.register_pool(pool);

    let mut cfg_b = PoolConfig::defaults("audio", 2);
    cfg_b.decay_rate = 0.4;
    cfg_b.prune_floor = 0.02;
    cfg_b.concept_emergence_threshold = 100;
    let pool_b = Pool::new(cfg_b, Box::new(BytePassthroughEncoding { prefix: "aud" }));
    let pool_b_id = fabric.register_pool(pool_b);

    // Reinforce just twice (below the Phase A consolidation lock
    // threshold of 3) so the terminal stays decay-eligible.  Phase A
    // intentionally makes terminals at consolidation >= 3 permanent
    // — that's the 100%-recall floor for trained pairs — so this
    // test exercises the OPPOSITE regime: lightly-reinforced
    // terminals that the brain should still forget if they aren't
    // touched again.
    for _ in 0..2 {
        fabric.observe(pool_id, b"x");
        fabric.observe(pool_b_id, b"y");
        fabric.advance_tick();
    }

    let pool_arc = fabric.pool(pool_id).unwrap();
    let pre_decay_count = pool_arc.read().iter_neurons()
        .map(|n| n.terminals.len()).sum::<usize>();
    assert!(pre_decay_count > 0, "should have grown at least one cross-pool terminal");

    // Now advance many ticks with NO reinforcement.  Each tick decays
    // by 0.5; weights drop fast.  Anything below prune_floor=0.1 gets
    // removed.  After enough ticks the cross-pool terminal should be
    // gone (within-pool atom→atom edges may persist depending on
    // their initial strength, but the cross-pool one was wired only
    // by the moment-buffer rule).
    for _ in 0..20 {
        fabric.advance_tick();
    }

    let post_decay_count = pool_arc.read().iter_neurons()
        .map(|n| n.terminals.len()).sum::<usize>();
    assert!(post_decay_count < pre_decay_count,
        "decay should have pruned at least one terminal (pre={}, post={})",
        pre_decay_count, post_decay_count,
    );
}
