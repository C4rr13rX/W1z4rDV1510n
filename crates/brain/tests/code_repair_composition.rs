use w1z4rd_brain::{apply_relation, CodeRepairRelation};

#[test]
fn operator_relation_preserves_unseen_function_and_parameter_names() {
    let source = "def power_two(value):\n    return value + value";
    let repaired = apply_relation(source, &CodeRepairRelation::ReplaceOperator {
        from: "+".into(), to: "*".into(),
    }).unwrap();
    assert_eq!(repaired, "def power_two(value):\n    return value * value");
}

#[test]
fn empty_guard_uses_current_function_parameter() {
    let source = "def mean_values(values):\n    return sum(values) / len(values)";
    let repaired = apply_relation(source, &CodeRepairRelation::GuardEmpty {
        fallback: "0".into(),
    }).unwrap();
    assert_eq!(repaired,
        "def mean_values(values):\n    return sum(values) / len(values) if values else 0");
}

#[test]
fn counter_relation_preserves_current_mapping_and_key_names() {
    let source = "def tally(text):\n    counts = {}\n    for token in text.split():\n        counts[token] = counts.get(token, 0)\n    return counts";
    let repaired = apply_relation(source, &CodeRepairRelation::IncrementStoredCount {
        amount: 1,
    }).unwrap();
    assert!(repaired.contains("counts[token] = counts.get(token, 0) + 1"));
}
