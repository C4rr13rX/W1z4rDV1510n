//! Environmental Equation Matrix tests per [`ARCHITECTURE.md`] §4.B + §4.D.
//!
//! The EEM is the brain's symbolic layer.  These tests prove the
//! architectural surface: equations evaluate under bindings, motifs
//! correlate to equations, validation moves confidence, and EEM
//! confidence enters the grounding report through
//! `Brain::integrate_with_equation`.

use ahash::AHashMap;
use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, Eem, EemConfig, PoolConfig,
};

#[test]
fn equation_evaluates_under_explicit_bindings() {
    // Spec §4.B: equation application is `evalexpr`-style.  Variables
    // referenced in the expression must be registered with the EEM
    // and bound by the caller.

    let mut eem = Eem::new(EemConfig::default());
    let v_a = eem.register_variable("a", None);
    let v_b = eem.register_variable("b", None);
    let eq  = eem.register_equation(
        "sum_of_two", "a + b", vec![v_a, v_b], None,
    );

    let mut bindings = AHashMap::new();
    bindings.insert(v_a, 3.0_f64);
    bindings.insert(v_b, 4.0_f64);
    let result = eem.apply(eq, &bindings).expect("must evaluate");
    assert!((result.value - 7.0).abs() < 1e-9,
        "3 + 4 = 7; got {}", result.value);
    assert!((result.confidence - 0.5).abs() < 1e-6,
        "fresh equation starts at config.initial_confidence (0.5); got {}",
        result.confidence);
}

#[test]
fn missing_binding_returns_none_rather_than_inventing_zero() {
    // Spec §2.7: substrate must not invent values.  An equation with
    // an unbound required variable returns None (honest absence),
    // not 0 or NaN.

    let mut eem = Eem::new(EemConfig::default());
    let v_a = eem.register_variable("a", None);
    let v_b = eem.register_variable("b", None);
    let eq  = eem.register_equation("ab", "a + b", vec![v_a, v_b], None);

    let mut bindings = AHashMap::new();
    bindings.insert(v_a, 3.0);
    // v_b deliberately unbound.
    assert!(eem.apply(eq, &bindings).is_none(),
        "missing binding must produce None");
}

#[test]
fn motif_observation_dedupes_by_fingerprint_and_bumps_count() {
    // Repeat observation of the same multi-pool fingerprint must NOT
    // create duplicate motifs.  observation_count is the architecture's
    // counter for "this pattern is recurring."

    let mut eem = Eem::new(EemConfig::default());
    let fp1 = vec![(1u32, 10u32), (2u32, 20u32)];
    let fp1_perm = vec![(2u32, 20u32), (1u32, 10u32)];  // same after sort
    let fp2 = vec![(1u32, 99u32), (2u32, 88u32)];

    let m1 = eem.observe_motif(fp1.clone());
    let m1_again = eem.observe_motif(fp1_perm);
    let m2 = eem.observe_motif(fp2);

    assert_eq!(m1, m1_again,
        "same fingerprint (modulo order) must dedupe to same motif id");
    assert_ne!(m1, m2,
        "distinct fingerprints must produce distinct motif ids");
    assert_eq!(eem.motif(m1).unwrap().observation_count, 2,
        "two observations must bump count to 2");
    assert_eq!(eem.motif(m2).unwrap().observation_count, 1);
}

#[test]
fn motif_to_equation_link_is_recorded_and_idempotent() {
    let mut eem = Eem::new(EemConfig::default());
    let v   = eem.register_variable("x", None);
    let eq  = eem.register_equation("identity", "x", vec![v], None);
    let m   = eem.observe_motif(vec![(1u32, 5u32)]);

    assert!(eem.link_motif_to_equation(m, eq));
    assert!(eem.link_motif_to_equation(m, eq), "duplicate link must succeed-and-noop");
    let links = eem.equations_for_motif(m);
    assert_eq!(links, vec![eq],
        "exactly one link recorded; idempotent on repeat");
}

#[test]
fn validation_moves_confidence_in_correct_direction_and_clamps() {
    let mut eem = Eem::new(EemConfig::default());
    let eq = eem.register_equation("c", "1.0", vec![], None);
    let start = eem.confidence(eq).unwrap();

    // 5 successes — confidence rises.
    for _ in 0..5 { eem.report_validation(eq, true); }
    let up = eem.confidence(eq).unwrap();
    assert!(up > start, "successes must raise confidence: {} -> {}", start, up);

    // Many failures — confidence falls and clamps at 0.
    for _ in 0..50 { eem.report_validation(eq, false); }
    let down = eem.confidence(eq).unwrap();
    assert!(down >= 0.0 && down <= 1.0, "confidence clamped to [0,1]; got {}", down);
    assert!(down < up, "failures must lower confidence: {} -> {}", up, down);
}

#[test]
fn duplicate_equation_registration_preserves_track_record() {
    // Registering an equation with an existing name must return the
    // existing id without resetting the validation counters.  Earned
    // confidence is data — losing it on a re-register would be
    // catastrophic for a long-running brain.

    let mut eem = Eem::new(EemConfig::default());
    let v   = eem.register_variable("x", None);
    let eq1 = eem.register_equation("e", "x", vec![v], None);
    eem.report_validation(eq1, true);
    let conf_after_validation = eem.confidence(eq1).unwrap();

    let eq2 = eem.register_equation("e", "x * 2", vec![v], None);
    assert_eq!(eq1, eq2, "duplicate name must return same id");
    assert_eq!(eem.confidence(eq2).unwrap(), conf_after_validation,
        "re-register must not reset confidence");
    // And the original expression must still be in place (replace
    // requires explicit replace_equation_expression).
    assert_eq!(eem.equation(eq2).unwrap().expression, "x",
        "register_equation must NOT silently overwrite expression");
}

#[test]
fn brain_exposes_owned_eem_for_seeding_and_query() {
    let mut brain = Brain::new(BrainConfig::default());
    let v = brain.eem_mut().register_variable("a", None);
    let eq = brain.eem_mut().register_equation("eq", "a * 2", vec![v], None);
    assert_eq!(brain.eem().equation_count(), 1);

    let mut bindings = AHashMap::new();
    bindings.insert(v, 5.0);
    let app = brain.eem().apply(eq, &bindings).expect("must evaluate");
    assert!((app.value - 10.0).abs() < 1e-9);
}

#[test]
fn integrate_with_equation_surfaces_eem_confidence_in_grounding() {
    // The architectural integration: when a caller supplies an
    // equation that applies to the context, `eem_confidence` becomes
    // `Some(_)` in the grounding report.  Spec §2.1 + §4.D.
    //
    // Without the equation path, `integrate` still works and reports
    // `eem_confidence: None` honestly.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );

    // Train cross-pool wiring (so the fabric path produces an answer).
    for _ in 0..6 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"X");

    // Register an equation that the caller wants applied to this
    // context.  Successes bring confidence up so the integration
    // shows a real positive signal.
    let v = brain.eem_mut().register_variable("c", None);
    let eq = brain.eem_mut().register_equation("constant", "c", vec![v], None);
    for _ in 0..6 { brain.eem_mut().report_validation(eq, true); }

    let mut bindings = AHashMap::new();
    bindings.insert("c", 1.0);
    let answer = brain.integrate_with_equation(pool_a, pool_b, "constant", &bindings);

    assert!(answer.grounding.eem_confidence.is_some(),
        "consulting an applicable equation must populate eem_confidence");
    let eem_c = answer.grounding.eem_confidence.unwrap();
    assert!(eem_c > 0.5,
        "after positive validations EEM confidence should exceed initial 0.5; got {}",
        eem_c);
    assert!(answer.grounding.integrated_confidence > 0.0,
        "integrated_confidence should be positive when both subsystems contribute");
}

#[test]
fn integrate_with_unknown_equation_keeps_eem_confidence_none() {
    // Asking for an unregistered equation must NOT invent a confidence.
    // Spec §2.7: honest absence beats invented presence.

    let mut brain = Brain::new(BrainConfig::default());
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    for _ in 0..4 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"X");
    let bindings = AHashMap::new();
    let answer = brain.integrate_with_equation(pool_a, pool_b, "not_registered", &bindings);
    assert!(answer.grounding.eem_confidence.is_none(),
        "unknown equation must NOT fabricate an EEM confidence");
}
