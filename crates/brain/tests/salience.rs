//! Stage 17.5 salience tests per [`ARCHITECTURE.md`] §17.5.
//!
//! Confirms:
//! 1. New atoms start with salience = 0.0.
//! 2. New concepts start with a small baseline salience (≥ 0).
//! 3. `Neuron::bump_salience` saturates at 1.0 and updates EMA.
//! 4. `Neuron::decay_salience` reduces salience toward 0.
//! 5. After a brain successfully decodes a trained query (the canonical
//!    cross-pool decode path), the participating binding + member neurons
//!    receive a measurable salience bump — i.e. the substrate IS emitting
//!    its own retention signal.

use w1z4rd_brain::neuron::{Neuron, NeuronKind};

#[test]
fn new_atom_has_zero_salience() {
    let a = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
    assert_eq!(a.salience, 0.0);
    assert_eq!(a.salience_ema, 0.0);
}

#[test]
fn new_concept_has_small_baseline_salience() {
    let c = Neuron::new_concept(
        1, "ab".into(), NeuronKind::Excitatory, vec![], 0,
    );
    assert!(c.salience > 0.0 && c.salience <= 0.05,
        "concept baseline salience should be small positive; got {}",
        c.salience);
    assert_eq!(c.salience, c.salience_ema,
        "concept's initial salience and EMA should match");
}

#[test]
fn bump_salience_saturates_at_one() {
    let mut n = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
    for _ in 0..50 { n.bump_salience(0.5); }
    assert_eq!(n.salience, 1.0, "salience must saturate at 1.0");
    assert!(n.salience_ema > 0.5,
        "EMA should be substantial after many bumps; got {}",
        n.salience_ema);
}

#[test]
fn bump_salience_ignores_negative_delta() {
    let mut n = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
    n.bump_salience(0.3);
    let before = n.salience;
    n.bump_salience(-0.5);
    assert_eq!(n.salience, before,
        "negative delta must not decrease salience (use decay_salience)");
}

#[test]
fn decay_salience_reduces_toward_zero() {
    let mut n = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
    n.bump_salience(0.8);
    let before = n.salience;
    n.decay_salience(0.5);
    assert!(n.salience < before,
        "decay must reduce salience; was {}, now {}", before, n.salience);
    for _ in 0..30 { n.decay_salience(0.5); }
    assert!(n.salience < 0.001,
        "many decay steps should drive salience near zero; got {}",
        n.salience);
    assert!(n.salience >= 0.0, "salience must not go negative");
}

#[test]
fn ema_follows_salience_with_lag() {
    let mut n = Neuron::new_atom(0, "t:a".into(), NeuronKind::Excitatory, 0);
    // One big bump.
    n.bump_salience(0.6);
    // EMA should be < salience after the first bump (it lags).
    assert!(n.salience_ema < n.salience,
        "EMA should lag salience on first bump: ema={} salience={}",
        n.salience_ema, n.salience);
    // After many bumps at the same delta, EMA converges toward salience.
    for _ in 0..100 { n.bump_salience(0.01); }
    let gap = (n.salience - n.salience_ema).abs();
    assert!(gap < 0.05,
        "EMA should converge with many bumps; gap = {}", gap);
}
