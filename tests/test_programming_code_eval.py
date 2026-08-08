from scripts.programming_code_eval import callable_candidates, executes


def test_unspecified_function_name_is_evaluated_by_behavior() -> None:
    code = "def count_words(s):\n    return {w: s.split().count(w) for w in set(s.split())}"
    passed, error, invoked = executes(
        code,
        "word_freq",
        ["a b a"],
        {"a": 2, "b": 1},
        "Produce a Python function mapping each word to its occurrence count.",
    )
    assert passed, error
    assert invoked == "count_words"


def test_explicit_function_name_remains_a_contract() -> None:
    code = "def chi_squared(n):\n    return n * n"
    passed, _, invoked = executes(
        code,
        "square",
        [7],
        49,
        "Create a Python function named square that computes a number times itself.",
    )
    assert not passed
    assert invoked == ""


def test_canonical_function_is_preferred_when_name_is_unspecified() -> None:
    code = "def helper(xs):\n    return 0\n\ndef avg_list(xs):\n    return sum(xs) / len(xs)"
    assert callable_candidates(
        code,
        "avg_list",
        "Write Python that calculates the arithmetic mean of a list.",
    ) == ["avg_list", "helper"]
