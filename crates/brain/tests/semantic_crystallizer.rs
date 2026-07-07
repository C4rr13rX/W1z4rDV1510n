use w1z4rd_brain::{CompositionRule, PatternValue as P, RelationPattern,
    SemanticCrystallizer, SemanticFrame, TransientWorkspace, TypedValue as V};
use w1z4rd_brain::{Eem, EemConfig};

fn frame(channel: &str, source: &str, tokens: &[(&str, &str)]) -> SemanticFrame {
    SemanticFrame { channel: channel.into(), provenance: source.into(),
        tokens: tokens.iter().map(|(kind, value)| V::new(*kind, *value)).collect() }
}

fn pattern(predicate: &str, arguments: Vec<P>) -> RelationPattern {
    RelationPattern::new(predicate, arguments)
}

fn rules() -> Vec<CompositionRule> {
    vec![
        CompositionRule { name: "law_join".into(), premises: vec![
            pattern("observes", vec![P::var("problem", "Problem"), P::var("model", "Model")]),
            pattern("law", vec![P::var("topic", "Topic"), P::var("model", "Model"), P::var("rel", "Relation")]),
        ], conclusion: pattern("applicable", vec![P::var("problem", "Problem"), P::var("rel", "Relation")]) },
        CompositionRule { name: "render_join".into(), premises: vec![
            pattern("applicable", vec![P::var("problem", "Problem"), P::var("rel", "Relation")]),
            pattern("requests", vec![P::var("problem", "Problem"), P::var("format", "Format")]),
            pattern("renderer", vec![P::var("format", "Format"), P::var("rel", "Relation"), P::var("artifact", "Artifact")]),
        ], conclusion: pattern("solution", vec![P::var("problem", "Problem"), P::var("artifact", "Artifact")]) },
    ]
}

#[test]
fn repeated_variation_crystallizes_roles_then_composes_unseen_crossings() {
    let mut c = SemanticCrystallizer::default();
    let training = [
        frame("law", "mechanics", &[("Topic","mechanics"),("Syntax","maps"),("Model","mass_acceleration"),("Syntax","to"),("Relation","multiply")]),
        frame("law", "density", &[("Topic","density"),("Syntax","maps"),("Model","mass_volume"),("Syntax","to"),("Relation","divide")]),
        frame("observes", "sensor-a", &[("Problem","p0"),("Syntax","contains"),("Model","mass_acceleration")]),
        frame("observes", "sensor-b", &[("Problem","p1"),("Syntax","contains"),("Model","mass_volume")]),
        frame("requests", "prompt-a", &[("Problem","p0"),("Syntax","wants"),("Format","python")]),
        frame("requests", "prompt-b", &[("Problem","p1"),("Syntax","wants"),("Format","sql")]),
        frame("renderer", "coding-a", &[("Format","python"),("Syntax","renders"),("Relation","multiply"),("Syntax","as"),("Artifact","python:multiply")]),
        frame("renderer", "coding-b", &[("Format","sql"),("Syntax","renders"),("Relation","divide"),("Syntax","as"),("Artifact","sql:divide")]),
    ];
    let mut learned = Vec::new();
    for item in training { learned.extend(c.observe(item)); }
    assert_eq!(c.template_count(), 4);

    // Neither complete problem was part of template induction.
    for item in [
        frame("law", "electricity", &[("Topic","electricity"),("Syntax","maps"),("Model","voltage_resistance"),("Syntax","to"),("Relation","divide")]),
        frame("observes", "live-sensor", &[("Problem","new_problem"),("Syntax","contains"),("Model","voltage_resistance")]),
        frame("requests", "live-prompt", &[("Problem","new_problem"),("Syntax","wants"),("Format","python")]),
        frame("renderer", "coding-c", &[("Format","python"),("Syntax","renders"),("Relation","divide"),("Syntax","as"),("Artifact","python:divide")]),
    ] { learned.extend(c.recognize(&item)); }
    let mut workspace = TransientWorkspace::new(learned);
    workspace.resolve(&rules(), 4);
    let solution = workspace.query("solution")
        .find(|f| f.arguments[0].value == "new_problem").unwrap();
    assert_eq!(solution.arguments[1].value, "python:divide");
    assert!(solution.provenance.contains("electricity"));
    assert!(solution.provenance.contains("live-sensor"));
    assert!(solution.provenance.contains("live-prompt"));
    assert!(solution.provenance.contains("coding-c"));
}

#[test]
fn coincidence_without_invariant_structure_does_not_crystallize() {
    let mut c = SemanticCrystallizer::default();
    assert!(c.observe(frame("noise", "a", &[("Token","one"),("Token","two")])).is_empty());
    assert!(c.observe(frame("noise", "b", &[("Token","three"),("Token","four")])).is_empty());
    assert_eq!(c.template_count(), 0);
}

#[test]
fn eem_serializes_templates_and_recognition_remains_read_only() {
    let mut eem = Eem::new(EemConfig::default());
    eem.consolidate_semantic_frame(frame("law", "a", &[
        ("Topic","mechanics"),("Syntax","maps"),("Model","mass_acceleration") ]));
    eem.consolidate_semantic_frame(frame("law", "b", &[
        ("Topic","density"),("Syntax","maps"),("Model","mass_volume") ]));
    assert_eq!(eem.semantic_template_count(), 1);
    let restored = Eem::from_snapshot(eem.snapshot());
    let before = restored.semantic_relation_count();
    let recognized = restored.recognize_semantic_frame(&frame("law", "query", &[
        ("Topic","geometry"),("Syntax","maps"),("Model","width_height") ]));
    assert_eq!(recognized.len(), 1);
    assert_eq!(restored.semantic_relation_count(), before);
    assert_eq!(restored.semantic_template_count(), 1);
}
