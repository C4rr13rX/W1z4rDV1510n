//! Action layer tests per [`ARCHITECTURE.md`] §4.E and §2.4.
//!
//! The action loop is the closed feedback path that makes
//! "best-learned-case-scenario" emerge: sensor input → cross-pool
//! axons → action firing → external effect → outcome → reinforcement
//! of source→action terminals.  Every property below is the spec's
//! architectural claim demonstrated as a working test.

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, NeuronRef, PoolConfig,
};

/// Build a brain with one sensor pool and one action pool (designated).
/// Returns (brain, sensor_pool_id, action_pool_id, action_ref).
fn build_action_brain(action_label: &str) -> (Brain, u32, u32, NeuronRef) {
    let mut brain = Brain::new(BrainConfig::default());
    let sensor = brain.create_pool(
        PoolConfig::defaults("sensor", 1),
        Box::new(BytePassthroughEncoding { prefix: "s" }),
    );
    let action_pool = brain.create_pool(
        PoolConfig::defaults("action", 2),
        Box::new(BytePassthroughEncoding { prefix: "act" }),
    );
    brain.designate_action_pool(action_pool);
    let action_ref = brain.register_action(action_label.into());
    (brain, sensor, action_pool, action_ref)
}

#[test]
fn action_fires_via_cross_pool_propagation_after_training() {
    // Spec §4.E: an action neuron fires when sensor input flows
    // through cross-pool terminals grown during training.  Train by
    // co-firing sensor "S" alongside the action neuron; recall by
    // observing "S" alone and confirming the action pool sees
    // activation.

    let (mut brain, sensor, _action_pool, action_ref) =
        build_action_brain("greet");

    // Train: co-fire sensor "S" with the action neuron for several
    // ticks.  Each tick close grows the sensor-S → action terminal.
    for _ in 0..6 {
        brain.observe(sensor, b"S");
        brain.fire_action(action_ref, 1.0);
        brain.advance_tick();
    }

    // Recall: observe "S" only.  No direct firing of the action
    // neuron — it must come through cross-pool propagation.
    brain.observe(sensor, b"S");
    let events = brain.take_action(0.05);

    assert_eq!(events.len(), 1,
        "trained sensor input must drive exactly one action event; got {}",
        events.len());
    let event = &events[0];
    assert_eq!(event.action_neuron, action_ref,
        "fired action must be the trained action neuron");
    assert_eq!(event.action_label, "greet",
        "event must carry the registered label");
    assert!(!event.sources.is_empty(),
        "credit attribution must list the firing sensor neuron(s)");
    assert!(event.sources.iter().all(|r| r.pool == sensor),
        "sources must live in the sensor pool; got {:?}", event.sources);
}

#[test]
fn feed_outcome_positive_strengthens_source_to_action_terminal() {
    // Spec §2.4: positive outcomes reinforce the source→action
    // terminal so future propagation produces stronger activation.

    let (mut brain, sensor, action_pool, action_ref) =
        build_action_brain("greet");

    // Train minimally so a cross-pool terminal exists.
    for _ in 0..4 {
        brain.observe(sensor, b"S");
        brain.fire_action(action_ref, 1.0);
        brain.advance_tick();
    }

    // Fire the action via recall.
    brain.observe(sensor, b"S");
    let events = brain.take_action(0.05);
    assert_eq!(events.len(), 1);
    let event_id = events[0].id;
    let source_ref = events[0].sources[0];

    // Read the source→action terminal weight BEFORE outcome.
    let weight_before = {
        let p = brain.fabric().pool(source_ref.pool).unwrap();
        let p = p.read();
        let n = p.get(source_ref.neuron).unwrap();
        n.terminals.iter()
            .find(|t| t.target == NeuronRef::new(action_pool, action_ref.neuron))
            .map(|t| t.weight)
            .unwrap_or(0.0)
    };
    assert!(weight_before > 0.0,
        "trained source→action terminal must exist before outcome");

    // Positive outcome.
    let applied = brain.feed_outcome(event_id, 1.5);
    assert!(applied, "feed_outcome must find the pending event");

    let weight_after = {
        let p = brain.fabric().pool(source_ref.pool).unwrap();
        let p = p.read();
        let n = p.get(source_ref.neuron).unwrap();
        n.terminals.iter()
            .find(|t| t.target == NeuronRef::new(action_pool, action_ref.neuron))
            .map(|t| t.weight)
            .unwrap_or(0.0)
    };
    assert!(weight_after > weight_before,
        "positive outcome must strengthen source→action terminal: before={} after={}",
        weight_before, weight_after);
}

#[test]
fn feed_outcome_negative_weakens_source_to_action_terminal() {
    // Negative outcomes weaken the terminal.  This is the substrate's
    // self-correction mechanism: bad outcomes should reduce the
    // likelihood of firing the same action from the same sources.

    let (mut brain, sensor, action_pool, action_ref) =
        build_action_brain("press");

    for _ in 0..6 {
        brain.observe(sensor, b"S");
        brain.fire_action(action_ref, 1.0);
        brain.advance_tick();
    }

    brain.observe(sensor, b"S");
    let events = brain.take_action(0.05);
    assert_eq!(events.len(), 1);
    let event_id = events[0].id;
    let source_ref = events[0].sources[0];

    let weight_before = {
        let p = brain.fabric().pool(source_ref.pool).unwrap();
        let p = p.read();
        let n = p.get(source_ref.neuron).unwrap();
        n.terminals.iter()
            .find(|t| t.target == NeuronRef::new(action_pool, action_ref.neuron))
            .map(|t| t.weight)
            .unwrap_or(0.0)
    };
    assert!(weight_before > 0.0);

    brain.feed_outcome(event_id, -0.3);

    let weight_after = {
        let p = brain.fabric().pool(source_ref.pool).unwrap();
        let p = p.read();
        let n = p.get(source_ref.neuron).unwrap();
        n.terminals.iter()
            .find(|t| t.target == NeuronRef::new(action_pool, action_ref.neuron))
            .map(|t| t.weight)
            .unwrap_or(0.0)
    };
    assert!(weight_after < weight_before,
        "negative outcome must weaken source→action terminal: before={} after={}",
        weight_before, weight_after);
}

#[test]
fn repeated_positive_outcomes_saturate_at_max_weight() {
    // Substrate must not let a single hyper-trained action terminal
    // dominate forever — weights saturate at `max_weight`.  Spec §1.5
    // and Terminal::reinforce_terminal.

    let (mut brain, sensor, action_pool, action_ref) =
        build_action_brain("blink");

    for _ in 0..3 {
        brain.observe(sensor, b"S");
        brain.fire_action(action_ref, 1.0);
        brain.advance_tick();
    }

    let max_w = {
        let p = brain.fabric().pool(sensor).unwrap();
        p.read().config.max_weight
    };

    // Many rounds of fire + maximal positive outcome.
    for _ in 0..30 {
        brain.observe(sensor, b"S");
        let events = brain.take_action(0.05);
        if let Some(e) = events.first() {
            brain.feed_outcome(e.id, max_w);
        }
        brain.advance_tick();
    }

    // Read the source→action terminal — it must be at or just below
    // max_weight (decay applies on each tick housekeeping too).
    let weight = {
        let p = brain.fabric().pool(sensor).unwrap();
        let p = p.read();
        // Find a neuron in sensor pool with a terminal targeting action.
        p.iter_neurons()
            .flat_map(|n| n.terminals.iter())
            .filter(|t| t.target.pool == action_pool && t.target.neuron == action_ref.neuron)
            .map(|t| t.weight)
            .fold(0.0_f32, f32::max)
    };
    assert!(weight <= max_w + 1e-3,
        "saturated weight {} must not exceed max_weight {}", weight, max_w);
    assert!(weight > max_w * 0.5,
        "after heavy reinforcement weight must approach max_weight; got {} vs max {}",
        weight, max_w);
}

#[test]
fn unknown_action_id_returns_false_on_feed_outcome() {
    // feed_outcome on an action_id that was never emitted (or already
    // fed back) must return false — caller telemetry signal.

    let (mut brain, _sensor, _action_pool, _action_ref) =
        build_action_brain("blink");
    assert!(!brain.feed_outcome(9999, 1.0),
        "feed_outcome on unknown id must return false");
    assert_eq!(brain.pending_action_count(), 0);
}

#[test]
fn pending_action_count_drops_after_feed_outcome() {
    // One-shot outcome: after feed_outcome the event leaves the
    // pending set.  Stops indefinite memory growth.

    let (mut brain, sensor, _action_pool, action_ref) =
        build_action_brain("nod");

    for _ in 0..4 {
        brain.observe(sensor, b"S");
        brain.fire_action(action_ref, 1.0);
        brain.advance_tick();
    }
    brain.observe(sensor, b"S");
    let events = brain.take_action(0.05);
    assert!(!events.is_empty());
    assert_eq!(brain.pending_action_count(), events.len());

    for e in &events {
        brain.feed_outcome(e.id, 0.5);
    }
    assert_eq!(brain.pending_action_count(), 0,
        "all events fed back; pending must drop to 0");
}
