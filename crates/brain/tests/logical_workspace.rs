use w1z4rd_brain::{CompositionRule, GroundedRelation, PatternValue as P,
    RelationPattern, TransientWorkspace, TypedValue as V, Eem, EemConfig};

fn fact(predicate: &str, args: &[(&str, &str)], source: &str) -> GroundedRelation {
    GroundedRelation::new(predicate, args.iter().map(|(k, v)| V::new(*k, *v)).collect(),
                          1.0, source)
}

fn pattern(predicate: &str, args: Vec<P>) -> RelationPattern {
    RelationPattern::new(predicate, args)
}

fn rules() -> Vec<CompositionRule> {
    vec![
        CompositionRule {
            name: "match_observation_to_law".into(),
            premises: vec![
                pattern("observes", vec![P::var("problem", "Problem"), P::var("model", "Model")]),
                pattern("law", vec![P::var("topic", "Topic"), P::var("model", "Model"),
                                    P::var("relation", "Relation")]),
            ],
            conclusion: pattern("applicable", vec![P::var("problem", "Problem"),
                                                    P::var("relation", "Relation")]),
        },
        CompositionRule {
            name: "render_applicable_relation".into(),
            premises: vec![
                pattern("applicable", vec![P::var("problem", "Problem"),
                                            P::var("relation", "Relation")]),
                pattern("requests", vec![P::var("problem", "Problem"), P::var("format", "Format")]),
                pattern("renderer", vec![P::var("format", "Format"), P::var("relation", "Relation"),
                                          P::var("artifact", "Artifact")]),
            ],
            conclusion: pattern("solution", vec![P::var("problem", "Problem"),
                                                  P::var("artifact", "Artifact")]),
        },
    ]
}

#[test]
fn dozens_of_four_topic_crossings_compose_deterministically() {
    // Each tuple is learned independently: scientific/business law,
    // recognizable model, and its deterministic relation.
    let laws = [
        ("mechanics", "mass_acceleration", "multiply"),
        ("density", "mass_volume", "divide"),
        ("electricity", "voltage_resistance", "divide"),
        ("geometry", "width_height", "multiply"),
        ("finance", "gross_fees", "subtract"),
        ("networks", "bytes_rate", "divide"),
        ("chemistry", "molarity_volume", "multiply"),
        ("music", "tempo_duration", "multiply"),
    ];
    let formats = ["python", "spreadsheet", "equation", "sql"];
    let mut learned = Vec::new();
    for (topic, model, relation) in laws {
        learned.push(fact("law", &[("Topic", topic), ("Model", model), ("Relation", relation)],
                          &format!("experience:law:{topic}")));
        for format in formats {
            learned.push(fact("renderer", &[("Format", format), ("Relation", relation),
                                             ("Artifact", &format!("{format}:{relation}"))],
                              &format!("experience:renderer:{format}:{relation}")));
        }
    }
    // 8 × 4 = 32 independently checkable cross-topic compositions.
    for (topic, model, relation) in laws {
        for format in formats {
            let problem = format!("{topic}_as_{format}");
            let mut episode = learned.clone();
            episode.push(fact("observes", &[("Problem", &problem), ("Model", model)],
                              &format!("experience:sensor:{problem}")));
            episode.push(fact("requests", &[("Problem", &problem), ("Format", format)],
                              &format!("experience:prompt:{problem}")));
            let mut workspace = TransientWorkspace::new(episode);
            let permanent_count = workspace.facts().len();
            workspace.resolve(&rules(), 4);
            let solutions: Vec<_> = workspace.query("solution")
                .filter(|f| f.arguments[0].value == problem).collect();
            assert_eq!(solutions.len(), 1, "{problem} must have one deterministic solution");
            assert_eq!(solutions[0].arguments[1].value, format!("{format}:{relation}"));
            assert!(solutions[0].provenance.len() >= 6,
                    "answer must retain four experiences and two inference rules");
            assert!(workspace.facts().len() > permanent_count,
                    "derived relations exist only in the transient workspace");
        }
    }
}

#[test]
fn type_mismatch_and_missing_path_remain_unanswered() {
    let facts = vec![
        fact("observes", &[("Problem", "p"), ("Model", "mass_volume")], "sensor"),
        // Same text, wrong type: cannot unify Model with an untyped label.
        fact("law", &[("Topic", "density"), ("Label", "mass_volume"),
                      ("Relation", "divide")], "law"),
        fact("requests", &[("Problem", "p"), ("Format", "rust")], "prompt"),
    ];
    let mut workspace = TransientWorkspace::new(facts);
    workspace.resolve(&rules(), 4);
    assert_eq!(workspace.query("solution").count(), 0);
}

#[test]
fn confidence_is_weakest_evidence_and_workspace_does_not_mutate_sources() {
    let mut source = vec![
        GroundedRelation::new("observes", vec![V::new("Problem", "p"), V::new("Model", "m")], 0.9, "sensor"),
        GroundedRelation::new("law", vec![V::new("Topic", "t"), V::new("Model", "m"),
                                             V::new("Relation", "r")], 0.7, "law"),
        GroundedRelation::new("requests", vec![V::new("Problem", "p"), V::new("Format", "f")], 0.8, "prompt"),
        GroundedRelation::new("renderer", vec![V::new("Format", "f"), V::new("Relation", "r"),
                                                  V::new("Artifact", "answer")], 0.6, "renderer"),
    ];
    let original = source.clone();
    let mut workspace = TransientWorkspace::new(source.drain(..));
    workspace.resolve(&rules(), 4);
    assert_eq!(workspace.query("solution").next().unwrap().confidence, 0.6);
    assert_eq!(original.len(), 4);
    assert!(original.iter().all(|fact| fact.predicate != "solution"));
}

#[test]
fn eem_persists_experiences_but_not_transient_conclusions() {
    let mut eem = Eem::new(EemConfig::default());
    for relation in [
        fact("observes", &[("Problem", "p"), ("Model", "m")], "sensor"),
        fact("law", &[("Topic", "t"), ("Model", "m"), ("Relation", "r")], "law"),
        fact("requests", &[("Problem", "p"), ("Format", "f")], "prompt"),
        fact("renderer", &[("Format", "f"), ("Relation", "r"),
                           ("Artifact", "answer")], "renderer"),
    ] { eem.register_semantic_relation(relation); }
    for rule in rules() { eem.register_composition_rule(rule); }
    let before = eem.semantic_relation_count();
    let workspace = eem.compose_transient(4);
    assert_eq!(workspace.query("solution").count(), 1);
    assert_eq!(eem.semantic_relation_count(), before,
               "inferred solution must not self-consolidate");

    let restored = Eem::from_snapshot(eem.snapshot());
    assert_eq!(restored.semantic_relation_count(), 4);
    assert_eq!(restored.composition_rule_count(), 2);
    assert_eq!(restored.compose_transient(4).query("solution").count(), 1);
}
