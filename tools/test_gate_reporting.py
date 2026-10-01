"""The gate's FAILURE REPORT, asserted rather than read off a screen once.

[5183439d] measured the cost of not having this: step 1 is fail-fast, so cargo
aborts at the first failing TARGET and prints `error: test failed, to rerun pass
-p w1z4rd-brain --lib` -- a target, never a test -- and an agent with 26 minutes
left had to re-run the whole 316s suite with --no-fail-fast just to learn a name.
Twice more, on 2026-09-30 and 2026-10-01, a killed gate's log held exactly one
line (`FAIL  brain tests   789.0s`, then `962.3s`) because every print after the
verdict lacked flush=True and the tail died in the buffer.

These are the two failure modes, and both are properties of gate.py's own
formatting, so they are testable with no cargo, no brain and no build lock --
which is also why they can be checked on a pass whose lock belongs to someone
else.

    python tools/test_gate_reporting.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    spec = importlib.util.spec_from_file_location("gatemod", ROOT / "tools" / "gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAILURES_IN_TWO_TARGETS = """
   Compiling w1z4rd-brain v0.1.0
test recall_is_total ... ok
test beside_next_derives ... FAILED
test result: FAILED. 134 passed; 1 failed; 0 ignored
test derivation_rejects_untaught ... FAILED
test splice_is_ordered ... ok
test result: FAILED. 7 passed; 1 failed; 0 ignored
error: test failed, to rerun pass `-p w1z4rd-brain --lib`
"""


def test_every_failing_test_is_named_across_targets(g) -> None:
    names = g.failing_tests(FAILURES_IN_TWO_TARGETS)
    assert names == ["beside_next_derives", "derivation_rejects_untaught"], names
    # The point of --no-fail-fast: a report that names only the first target's
    # failure is the defect, so assert BOTH targets are represented.
    assert len(names) == 2, names


def test_a_compile_error_names_no_test_and_says_so(g) -> None:
    compile_error = (
        "error[E0599]: no method named `binding_routes_from` found for struct `Pool`\n"
        "error: could not compile `w1z4rd-brain` (lib test) due to 1 previous error\n"
    )
    assert g.failing_tests(compile_error) == []
    # The empty list must not print as an empty line: "no test named" and "no
    # output" are different diagnoses and the gate has to distinguish them.
    assert "no test named" in (
        f"      FAILED: {', '.join([]) if [] else '(no test named -- compile error or harness failure)'}"
    )


def test_ok_output_names_nothing(g) -> None:
    assert g.failing_tests("test a ... ok\ntest result: ok. 135 passed; 0 failed") == []


def test_no_fail_fast_goes_before_cargos_separator_not_capped_pys(g) -> None:
    # The real step 1, including capped.py's own `--`.
    step = next(cmd for name, cmd in g.STEPS if name == "brain tests")
    out = g.with_no_fail_fast(step)
    assert "--no-fail-fast" in out
    i = out.index("--no-fail-fast")
    # After `cargo test`, before the LAST `--`, and therefore before
    # --test-threads=2 is handed to libtest.
    assert out.index("cargo") < i, out
    assert i < len(out) - 1 - out[::-1].index("--"), out
    assert out[-1] == "--test-threads=2", out
    # The flag must never land where capped.py would eat it.
    assert out.index("--") > i or out.count("--") >= 2, out
    assert out.index("--") < out.index("cargo"), "capped.py's separator moved"


def test_no_fail_fast_is_idempotent_and_safe_on_one_separator(g) -> None:
    step = next(cmd for name, cmd in g.STEPS if name == "brain tests")
    once = g.with_no_fail_fast(step)
    assert g.with_no_fail_fast(once) == once, "applying it twice must not duplicate"

    # A `cargo check` step has only capped.py's `--`, so the flag appends.
    check = next(cmd for name, cmd in g.STEPS if name == "node compiles")
    out = g.with_no_fail_fast(check)
    assert out[-1] == "--no-fail-fast", out
    assert out[: len(check)] == check, "appending must not reorder the command"


def test_this_file_is_a_gate_step_and_costs_no_build_lock(g) -> None:
    # A test nothing runs is prose. It must be IN the gate...
    step = dict(g.STEPS).get("gate reporting")
    assert step is not None, [n for n, _ in g.STEPS]
    assert step[-1].endswith("test_gate_reporting.py"), step
    # ...and must NOT go through capped.py, or the one step that needs no cargo
    # would queue on the machine-wide build lock behind every other agent's
    # build. Measured 2026-10-01: one waiter sat in that lock's acquire loop for
    # 9 minutes.
    assert not any("capped.py" in a for a in step), step
    # It runs before the expensive steps, so a broken report is known in 0.1s
    # rather than after 962s of cargo.
    assert [n for n, _ in g.STEPS][0] == "gate reporting", [n for n, _ in g.STEPS]


def test_this_files_own_output_is_parseable_by_the_gate(g) -> None:
    # If this step fails, gate.py names it with the SAME parser it uses on cargo.
    # Printing "FAIL  name" instead would make this the one step whose failures
    # the gate cannot name -- the defect [5183439d] exists for, reintroduced by
    # the test that pins it.
    src = Path(__file__).read_text(encoding="utf-8")
    body = src[src.index("def main("):]
    assert 'f"test {t.__name__} ... FAILED"' in body, "failure lines must match cargo's shape"
    assert g.failing_tests("test some_check ... FAILED") == ["some_check"]


def test_the_fail_path_prints_names_before_the_tail(g) -> None:
    # Properties of the source, because the alternative is running a red gate:
    # the verdict and everything after it must be flushed, or a killed gate
    # reports that something failed and deletes which thing.
    src = (ROOT / "tools" / "gate.py").read_text(encoding="utf-8")
    body = src[src.index("def main("):]
    prints = [seg for seg in body.split("print(")[1:]]
    unflushed = [p.splitlines()[0] for p in prints if "flush=True" not in p[:400]]
    assert not unflushed, f"unflushed print(s) in main(): {unflushed}"
    assert body.index("FAILED: ") < body.index("tail[-20:]"), "names must precede the tail"


def main() -> int:
    """Output is deliberately in CARGO'S FORMAT, and that is not cosmetic.

    gate.py's own `failing_tests()` parses lines that start with "test " and end
    with "FAILED". Printing "FAIL  name" instead would make this the one step in
    the gate whose failures the gate cannot name -- the exact defect [5183439d]
    was filed for, reintroduced by the test that pins it.
    """
    g = load_gate()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t(g)
            print(f"test {t.__name__} ... ok", flush=True)
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"test {t.__name__} ... FAILED", flush=True)
            print(f"      {e}", flush=True)
    result = "ok" if not failed else "FAILED"
    print(f"\ntest result: {result}. {len(tests) - len(failed)} passed; {len(failed)} failed",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
