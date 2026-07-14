use w1z4rd_brain::{AtomEncoding, CodeStructureEncoding, PoolPrototypeRegistry};

#[test]
fn renamed_functions_share_structural_firing_sequence() {
    let encoding = CodeStructureEncoding { prefix: "s".into() };
    assert_eq!(
        encoding.atomize(b"def square(n):\n    return n + n"),
        encoding.atomize(b"def power_two(value):\n    return value + value"),
    );
}

#[test]
fn operators_remain_visible_as_distinct_structural_atoms() {
    let encoding = CodeStructureEncoding { prefix: "s".into() };
    assert_ne!(
        encoding.atomize(b"def f(x):\n return x + x"),
        encoding.atomize(b"def f(x):\n return x * x"),
    );
}

#[test]
fn default_registry_builds_code_structure_prototype() {
    let encoding = PoolPrototypeRegistry::with_defaults()
        .build("code-structure", "syntax").unwrap();
    assert_eq!(encoding.name(), "code-structure");
}
