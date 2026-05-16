//! Stage 2b tests: when concept A fires at tick T and concept B fires
//! at tick T+1 in the same pool, the fabric grows an A → B terminal.
//! This is the within-pool temporal "what follows what" wiring above
//! the atom-emergence layer.

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, NeuronRef, PoolConfig,
};

fn build_brain_for_sequential() -> (Brain, u32) {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = u32::MAX;  // single-pool focus
    let mut brain = Brain::new(cfg);
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window         = 1024;
    pc.concept_emergence_threshold = 3;
    // Cap at 2 so only "AB" and "CD" emerge — level-2 concepts like
    // ABCD can't form and eat the bare CD-concept's firing.  This
    // isolates the Stage 2b sequential wiring property we're testing.
    pc.max_concept_member_count    = 2;
    pc.decay_rate                  = 0.00001;
    pc.prune_floor                 = 0.0005;
    let pool = brain.create_pool(pc,
        Box::new(BytePassthroughEncoding { prefix: "t" }));
    (brain, pool)
}

#[test]
fn sequential_concept_firings_grow_concept_to_concept_terminal() {
    // Train: alternate observations "AB" and "CD" so concepts "AB" and
    // "CD" emerge.  Then deliberately train "AB" followed by "CD" so
    // the AB-concept fires at tick T and CD-concept fires at T+1.

    let (mut brain, pool) = build_brain_for_sequential();

    // Emerge "AB" and "CD" concepts (need 3+ reps each), then drive
    // AB → CD sequential firing repeatedly.
    for _ in 0..10 {
        brain.observe(pool, b"AB");
        brain.advance_tick();
        brain.observe(pool, b"CD");
        brain.advance_tick();
    }

    // After these reps, the AB→CD sequential terminal should already
    // be growing (every (AB, CD) pair adds to it).  Let's locate the
    // two concept neurons and read the terminal weight.
    let pool_arc = brain.fabric().pool(pool).unwrap();
    let p = pool_arc.read();

    let ab_concept = p.iter_neurons()
        .find(|n| !n.is_atom() && n.label == "t:QQ~t:Qg")
        .expect("AB concept must have emerged");
    // 'A'=0x41 → base64 'QQ', 'B'=0x42 → 'Qg'
    let cd_concept = p.iter_neurons()
        .find(|n| !n.is_atom() && n.label == "t:Qw~t:RA")
        .expect("CD concept must have emerged");
    // 'C'=0x43 → 'Qw', 'D'=0x44 → 'RA'

    let target = NeuronRef::new(pool, cd_concept.id);
    let weight = ab_concept.terminals.iter()
        .find(|t| t.target == target)
        .map(|t| t.weight)
        .unwrap_or(0.0);
    assert!(weight > 0.0,
        "AB-concept must have grown a terminal toward CD-concept after \
         sequential firings; got weight={}", weight);
}

#[test]
fn no_self_loop_terminal_grows_on_same_concept_repeating() {
    // Spec §1.3 spirit: temporal-sequence wiring connects DIFFERENT
    // concepts.  A concept firing twice in a row should NOT grow a
    // self-loop terminal (those are nonsensical for sequencing).

    let (mut brain, pool) = build_brain_for_sequential();
    for _ in 0..6 {
        brain.observe(pool, b"AB");
        brain.advance_tick();
    }

    let pool_arc = brain.fabric().pool(pool).unwrap();
    let p = pool_arc.read();
    let ab = p.iter_neurons()
        .find(|n| !n.is_atom() && n.label == "t:QQ~t:Qg")
        .expect("AB concept must have emerged");
    let self_target = NeuronRef::new(pool, ab.id);
    let self_terminal = ab.terminals.iter().find(|t| t.target == self_target);
    assert!(self_terminal.is_none(),
        "concept must not grow a self-loop terminal on repeated firing; \
         got {:?}", self_terminal);
}
