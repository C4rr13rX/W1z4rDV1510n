//! Temporal-prediction annealer tests per [`ARCHITECTURE.md`] §4.C.
//!
//! These prove the architectural claims:
//! - Annealer maintains a per-pool history and produces predictions
//!   after enough frames accumulate.
//! - Repeated periodic patterns earn high confidence; pure noise
//!   earns low confidence.
//! - report_actual returns the prediction error magnitude.
//! - Brain::integrate_with_prediction surfaces annealer_confidence in
//!   the grounding report; pre-history brains keep it `None`.

use ahash::AHashMap;
use w1z4rd_brain::{
    Annealer, AnnealerConfig, Brain, BrainConfig, BytePassthroughEncoding,
    NeuronId, PoolConfig,
};

fn frame(pairs: &[(NeuronId, f32)]) -> AHashMap<NeuronId, f32> {
    pairs.iter().copied().collect()
}

#[test]
fn predict_returns_none_until_two_frames_recorded() {
    // Spec §2.7: substrate must not invent predictions out of
    // nothing.  One frame isn't a pair; predict_next must honestly
    // return None.

    let mut a = Annealer::new(AnnealerConfig::default());
    assert!(a.predict_next(1).is_none(), "no history → no prediction");
    a.record_frame(1, frame(&[(10, 1.0)]));
    assert!(a.predict_next(1).is_none(), "one frame → no prediction");
    a.record_frame(1, frame(&[(11, 1.0)]));
    assert!(a.predict_next(1).is_some(), "two frames → prediction emerges");
}

#[test]
fn repeated_periodic_pattern_earns_high_confidence() {
    // Train: A → B → A → B → ... many times.  After enough cycles
    // the annealer should converge to a sharp prediction with high
    // confidence.

    let mut a = Annealer::new(AnnealerConfig::default());
    let pool = 1u32;
    for _ in 0..15 {
        a.record_frame(pool, frame(&[(10, 1.0)]));
        a.record_frame(pool, frame(&[(11, 1.0)]));
    }
    let pred = a.predict_next(pool).expect("must produce a prediction");
    assert!(pred.confidence > 0.4,
        "regular periodic pattern should yield notable confidence; got {}",
        pred.confidence);
}

#[test]
fn noisy_unpredictable_history_yields_low_confidence_vs_periodic() {
    // The architectural claim is comparative: a periodic-pattern
    // pool should outscore a chaotic-pattern pool on confidence,
    // even with the same number of frames.

    let mut periodic = Annealer::new(AnnealerConfig::default());
    let pool_p = 1u32;
    for _ in 0..15 {
        periodic.record_frame(pool_p, frame(&[(10, 1.0)]));
        periodic.record_frame(pool_p, frame(&[(11, 1.0)]));
    }

    // Noisy: each successive frame picks a wildly different neuron.
    let mut noisy = Annealer::new(AnnealerConfig::default());
    let pool_n = 2u32;
    for i in 0..30u32 {
        noisy.record_frame(pool_n, frame(&[((i.wrapping_mul(7919)) as NeuronId % 1000, 1.0)]));
    }

    let p = periodic.predict_next(pool_p).expect("p prediction");
    let n = noisy.predict_next(pool_n).expect("n prediction");
    assert!(p.confidence > n.confidence,
        "periodic ({}) must outscore noisy ({}) in confidence",
        p.confidence, n.confidence);
}

#[test]
fn report_actual_yields_zero_for_exact_match_and_positive_otherwise() {
    let a = Annealer::new(AnnealerConfig::default());
    let predicted = frame(&[(1, 0.5), (2, 0.8)]);
    let same = predicted.clone();
    assert!(a.report_actual(&predicted, &same).abs() < 1e-6,
        "exact match must be zero error");

    let different = frame(&[(1, 0.1), (2, 0.2)]);
    let err = a.report_actual(&predicted, &different);
    assert!(err > 0.0, "differing frame must give positive error; got {}", err);
}

#[test]
fn history_window_bounds_memory() {
    // Frames beyond `history_window` must roll off, not accumulate.
    let mut cfg = AnnealerConfig::default();
    cfg.history_window = 5;
    let mut a = Annealer::new(cfg);
    for i in 0..50u32 {
        a.record_frame(1, frame(&[(i as NeuronId, 1.0)]));
    }
    assert_eq!(a.history_len(1), 5,
        "history must be bounded by window; got {}", a.history_len(1));
}

#[test]
fn empty_frames_are_not_recorded() {
    // An empty frame is "nothing fired" — not a predictive signal.
    let mut a = Annealer::new(AnnealerConfig::default());
    a.record_frame(1, AHashMap::new());
    assert_eq!(a.history_len(1), 0,
        "empty frame must not bump history");
}

#[test]
fn brain_advance_tick_auto_captures_one_frame_per_active_pool() {
    // The brain's job in Phase 6: every tick close, snapshot each
    // pool's currently-firing pattern into the annealer.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    for _ in 0..5 {
        brain.observe(pool_a, b"X");
        brain.advance_tick();
    }
    assert_eq!(brain.annealer().history_len(pool_a), 5,
        "5 ticks of observation must produce 5 history frames; got {}",
        brain.annealer().history_len(pool_a));
}

#[test]
fn integrate_with_prediction_populates_annealer_confidence_in_grounding() {
    // The integration surface: when the annealer has history for the
    // target pool, `annealer_confidence` becomes `Some(_)` and
    // `integrated_confidence` reflects both fabric AND annealer.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );

    for _ in 0..8 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"X");
    let answer = brain.integrate_with_prediction(pool_a, pool_b);
    assert!(answer.grounding.annealer_confidence.is_some(),
        "with history, annealer_confidence must be Some(_)");
    let ac = answer.grounding.annealer_confidence.unwrap();
    assert!(ac > 0.0 && ac <= 1.0,
        "annealer_confidence must be in (0,1]; got {}", ac);
}

#[test]
fn integrate_alone_keeps_annealer_confidence_none() {
    // Spec compatibility: plain `integrate` must NOT auto-consult the
    // annealer.  Phase 6 adds a new entry point; it does not change
    // the meaning of the original one.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    for _ in 0..5 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"X");
    let answer = brain.integrate(pool_a, pool_b);
    assert!(answer.grounding.annealer_confidence.is_none(),
        "plain integrate must keep annealer_confidence None");
}
