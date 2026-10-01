#!/usr/bin/env python3
"""
The gate every automated pass on the brain must keep green.

    python tools/gate.py

1. The brain crate's tests          (cargo test -p w1z4rd-brain)
2. The node still compiles against it (cargo check -p w1z4rdv1510n-node; the
   stray src/bin/brain_server_cluster.rs, a module with no main, was already
   broken on main 2026-09-30 and is left out)
3. The scorecard against its baseline (tools/scorecard.py --stress, so scale
   64 is compared too): recall and
   integration never drop, RAM stays under 2 GB and never regresses, and
   integration_wrong_pct stays strictly BELOW integration_pct and never rises
   more than 2 points over the baseline -- so a pass cannot buy integration by
   inventing more answers, which is what every rule above was blind to.

Everything runs inside tools/capped.py with 2 build jobs, so no step can take
the machine's memory. Nothing here starts the node, touches
D:\\w1z4rdv1510n-data or brain-data*, or talks to AWS.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPPED = [sys.executable, str(ROOT / "tools" / "capped.py"), "--mb", "8000", "--timeout", "2400", "--"]
STEPS = [
    # First, and NOT through capped.py: no cargo, no brain, ~0.1s, so it cannot
    # take the machine-wide build lock or the memory this cap exists to bound.
    # It guards the gate's own failure REPORT, which is the thing a red gate is
    # read through -- [5183439d] cost one agent a 316s re-run to learn a test's
    # name, and two more passes logged a single unattributable line because the
    # tail was unflushed. A test nothing runs is prose, so it runs here.
    ("gate reporting", [sys.executable, str(ROOT / "tools" / "test_gate_reporting.py")]),
    ("brain tests", CAPPED + ["cargo", "test", "-p", "w1z4rd-brain", "--release", "-j", "2", "--",
                              "--test-threads=2"]),
    ("node compiles", CAPPED + ["cargo", "check", "-p", "w1z4rdv1510n-node", "--release", "-j", "2",
                                "--bin", "w1z4rdv1510n-node", "--bin", "w1z4rd_brain_server",
                                "--bin", "w1z4rd_brain_migrate"]),
    # --stress is NOT optional here. Without it the scorecard step runs scales
    # 1, 4 and 16 only, so the gate could never see a scale-64 RAM regression --
    # and it did not: a +5.3 MB scale-64 growth landed green on 2026-10-01 and
    # was found by hand. Scale 64 is exempt from the 2 GB BUDGET check
    # (report_only) but NOT from the +15%-over-baseline check, which is the one
    # that catches a regression. Costs ~37s.
    ("scorecard", [sys.executable, str(ROOT / "tools" / "scorecard.py"), "--stress"]),
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def failing_tests(out: str) -> list[str]:
    return [l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.rstrip().endswith("FAILED")]


def with_no_fail_fast(cmd: list[str]) -> list[str]:
    """`cmd` with cargo's `--no-fail-fast`, inserted BEFORE the `--` separator.

    The placement is the whole point, and there are TWO separators in play.
    A step is `python capped.py --mb 8000 --timeout 2400 -- cargo test -p
    w1z4rd-brain --release -j 2 -- --test-threads=2`: the FIRST `--` belongs to
    capped.py and the LAST belongs to cargo. Insert before the first and the flag
    becomes a capped.py argument; append after the last and it is handed to the
    TEST HARNESS, which does not know it, so libtest exits on a usage error and
    the gate reports a red it invented rather than one in the code.

    So: insert before the LAST `--` when there are two or more (cargo's own
    separator), and append when there is one or none (a `cargo check` step, where
    the only `--` is capped.py's). Idempotent.
    """
    if "--no-fail-fast" in cmd:
        return list(cmd)
    out = list(cmd)
    if out.count("--") >= 2:
        out.insert(len(out) - 1 - out[::-1].index("--"), "--no-fail-fast")
    else:
        out.append("--no-fail-fast")
    return out


def main() -> int:
    failed = []
    for name, cmd in STEPS:
        t0 = time.time()
        proc = run(cmd)
        ok = proc.returncode == 0
        if not ok and name == "brain tests":
            # One immediate retry: a test that fails once and passes at once is
            # flaky (seen 2026-09-30), and a flake must not reject a good pass.
            # It is still named, so it gets fixed rather than forgotten.
            #
            # THE RETRY RUNS --no-fail-fast AND THE FIRST ATTEMPT DOES NOT, which
            # is deliberate and is not what [5183439d] asked for. Step 1 is
            # fail-fast, so cargo aborts at the first failing TARGET and the only
            # thing printed is "error: test failed, to rerun pass -p w1z4rd-brain
            # --lib" -- a target, never a test, and every other failing target
            # stays hidden. Running the FIRST attempt with --no-fail-fast fixes
            # that and doubles the cost of a red gate, which does not fit:
            # measured 2026-10-01 this step took 962.3s for attempt plus retry
            # under contention, against capped.py's 2400s timeout, so two full
            # no-fail-fast passes can hit the timeout and a timed-out gate names
            # nothing at all. The retry is already paid for, so it is the free
            # place to buy completeness: the first attempt detects red cheaply,
            # the retry sees every target, and `proc` below is the retry whenever
            # one ran, so the names come from the complete run.
            names = failing_tests(proc.stdout)
            retry = run(with_no_fail_fast(cmd))
            if retry.returncode == 0:
                ok = True
                print(f"      flaky, passed on retry: {', '.join(names) or '(unnamed)'}",
                      flush=True)
            else:
                proc = retry
        print(f"{'PASS' if ok else 'FAIL'}  {name:14} {time.time() - t0:6.1f}s", flush=True)
        if not ok:
            failed.append(name)
            # The NAMES first, flushed, and on their own line. The verdict above
            # says a step failed; without this nobody can say WHICH suite, and a
            # gate whose failure cannot be attributed cannot be compared against
            # a baseline. Measured twice: on 2026-09-30 and again on 2026-10-01
            # a killed gate's log held exactly one line -- "FAIL brain tests
            # 789.0s", then "FAIL brain tests 962.3s" -- because the verdict
            # print carries flush=True and every print after it did not, so the
            # tail sat in a 8 KB buffer that only a clean exit would drain. Both
            # runs had the list; both lost it.
            names = failing_tests(proc.stdout)
            print(f"      FAILED: {', '.join(names) if names else '(no test named -- compile error or harness failure)'}",
                  flush=True)
            tail = [l for l in (proc.stdout + proc.stderr).splitlines()
                    if l.strip() and not l.startswith(("warning", "  |", "   |", "  =", " -->"))]
            print("\n".join("      " + l for l in tail[-20:]), flush=True)
        elif name == "scorecard":
            print("\n".join("      " + l for l in proc.stdout.splitlines()
                            if l.lstrip()[:1].isdigit() or l.startswith(("scale", "RAM growth"))),
                  flush=True)
    print(f"\ngate: {'FAILED -- ' + ', '.join(failed) if failed else 'OK'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
