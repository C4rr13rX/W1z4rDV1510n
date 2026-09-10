"""Every registry .toml must load, and every `must_be_valid` must discriminate.

`load_registry()` is all-or-nothing: it walks the directory and raises
`SchemaError` on the first file that fails to parse. One bad .toml therefore
takes down every corpus, not just its own -- the driver dies at startup before
it posts a single row, for a corpus it was never asked to train.

Measured 2026-09-09, twice over, from one file:

  1. `go_systems_001.toml` shipped `category = "systems_programming_go"`, which
     is not in `schema.CATEGORIES`.
  2. With the category corrected, the same file still failed to load on
     `must_be_valid = "go"`, which was not in `schema.SUPPORTED_LANGS`.

Both raised for the WHOLE registry. Twelve deferred-replay intervals across
four unrelated corpora -- jupyter-scientific, metamathqa, webstack -- were
marked `deferred_replay_failed` and rejected without a row being trained or a
gate being run, and the pass then reported `deferred_replay_complete` with 26
rejections. A deploy typo was recorded as twelve semantic verdicts about
corpus content. Defect 2 survived the fix for defect 1 because the category was
checked against CATEGORIES by eye and `load_registry()` was never re-run.

So this test loads the real registry directory rather than a fixture. A fixture
would have passed throughout both outages.
"""
from __future__ import annotations

import pathlib

import pytest

from tools.training_standard import schema, score


REGISTRY_DIR = pathlib.Path(__file__).resolve().parents[1] / "tools" / "training_standard" / "registry"


def test_the_real_registry_directory_loads():
    """The check that both outages needed and neither had."""
    scripts = schema.load_registry(REGISTRY_DIR)
    assert scripts, f"no scripts loaded from {REGISTRY_DIR}"


def test_every_registry_file_loads_individually():
    """Name the offending file instead of just the first failure.

    `load_registry()` stops at the first bad file, so a single SchemaError
    hides however many others are queued behind it -- exactly the shape that
    let defect 2 hide behind defect 1.
    """
    failures = []
    for path in sorted(REGISTRY_DIR.glob("*.toml")):
        try:
            schema.load_script(path)
        except schema.SchemaError as exc:
            failures.append(f"{path.name}: {exc}")
    assert not failures, "registry files failed to load:\n" + "\n".join(failures)


def test_every_declared_language_has_a_validator_arm():
    """A language in SUPPORTED_LANGS with no arm in `_validate_lang` is worse
    than one that is absent.

    Absent, it raises SchemaError and the corpus is visibly blocked. Present
    but unimplemented, it falls through to `return True, ""` at the bottom of
    `_validate_lang` and silently awards the full 0.5 structural weight to any
    text whatsoever -- a benchmark that reports a pass while checking nothing.
    """
    for lang in sorted(schema.SUPPORTED_LANGS):
        ok, _ = _validate(lang, "this is prose, not %s source at all" % lang)
        assert not ok, (
            f"{lang!r} is in SUPPORTED_LANGS but `_validate_lang` accepts "
            f"arbitrary prose as valid {lang}, so its structural check is "
            f"vacuous"
        )


@pytest.mark.parametrize(
    "code, expected_ok",
    [
        # gofmt-shaped, the way CodeSearchNet rows arrive.
        ("func Incr(m map[string]int, k string) { mu.Lock(); m[k]++; mu.Unlock() }", True),
        ("package main\n\nfunc main() {}\n", True),
        ("type Counter struct {\n\tmu sync.Mutex\n}\n", True),
        # Truncated mid-body: the failure a structural check exists to catch.
        ("func Incr(m map[string]int) {\n\tmu.Lock()\n", False),
        # Prose that mentions Go but is not Go.
        ("Use a sync.Mutex to guard the map across goroutines.", False),
        ("", False),
    ],
)
def test_go_validator_discriminates(code, expected_ok):
    ok, err = _validate("go", code)
    assert ok is expected_ok, f"go check returned {ok} for {code!r}: {err}"


@pytest.mark.parametrize(
    "code, expected_ok",
    [
        # The two shapes code_gen_bash_001 actually benchmarks.
        ("find . -mtime -1 | grep ERROR", True),
        ('for i in 1 2 3; do curl -sf "$URL" && break; sleep 5; done', True),
        # Truncated mid-loop -- what a cut-off response looks like in shell.
        ('for i in 1 2 3; do curl -sf "$URL"', False),
        ('if [ -f x ]; then echo hi', False),
        # Prose, which this arm accepted as valid bash until 2026-09-09.
        ("Loop over the files and grep each one for the word ERROR.", False),
    ],
)
def test_bash_validator_discriminates(code, expected_ok):
    ok, err = _validate("bash", code)
    assert ok is expected_ok, f"bash check returned {ok} for {code!r}: {err}"


def test_go_benchmarks_score_against_a_real_response():
    """End-to-end: the go_systems_001 benchmarks must be satisfiable.

    A benchmark whose `must_include` can never co-occur with valid source is a
    permanent gate failure that looks like a training deficit.
    """
    script = schema.load_script(REGISTRY_DIR / "go_systems_001.toml")
    mutex_bench = next(b for b in script.benchmarks if "mutex" in b.label)
    answer = (
        "```go\n"
        "func Incr(m map[string]int, k string) {\n"
        "\tvar mu sync.Mutex\n"
        "\tmu.Lock()\n"
        "\tdefer mu.Unlock()\n"
        "\tm[k]++\n"
        "}\n"
        "```"
    )
    result = score.evaluate(answer, mutex_bench)
    assert result.passed, result.breakdown


def _validate(lang: str, code: str) -> tuple[bool, str]:
    return score._validate_lang(code, lang)
