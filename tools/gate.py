#!/usr/bin/env python3
"""
The gate every automated pass on the brain must keep green.

    python tools/gate.py

1. The brain crate's tests          (cargo test -p w1z4rd-brain)
2. The node still compiles against it (cargo check -p w1z4rdv1510n-node; the
   stray src/bin/brain_server_cluster.rs, a module with no main, was already
   broken on main 2026-09-30 and is left out)
3. The scorecard against its baseline (tools/scorecard.py): recall and
   integration never drop, RAM stays under 2 GB and never regresses.

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
    ("brain tests", CAPPED + ["cargo", "test", "-p", "w1z4rd-brain", "--release", "-j", "2", "--",
                              "--test-threads=2"]),
    ("node compiles", CAPPED + ["cargo", "check", "-p", "w1z4rdv1510n-node", "--release", "-j", "2",
                                "--bin", "w1z4rdv1510n-node", "--bin", "w1z4rd_brain_server",
                                "--bin", "w1z4rd_brain_migrate"]),
    ("scorecard", [sys.executable, str(ROOT / "tools" / "scorecard.py")]),
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def failing_tests(out: str) -> list[str]:
    return [l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.rstrip().endswith("FAILED")]


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
            names = failing_tests(proc.stdout)
            retry = run(cmd)
            if retry.returncode == 0:
                ok = True
                print(f"      flaky, passed on retry: {', '.join(names) or '(unnamed)'}")
            else:
                proc = retry
        print(f"{'PASS' if ok else 'FAIL'}  {name:14} {time.time() - t0:6.1f}s", flush=True)
        if not ok:
            failed.append(name)
            tail = [l for l in (proc.stdout + proc.stderr).splitlines()
                    if l.strip() and not l.startswith(("warning", "  |", "   |", "  =", " -->"))]
            print("\n".join("      " + l for l in tail[-20:]))
        elif name == "scorecard":
            print("\n".join("      " + l for l in proc.stdout.splitlines()
                            if l.lstrip()[:1].isdigit() or l.startswith(("scale", "RAM growth"))))
    print(f"\ngate: {'FAILED -- ' + ', '.join(failed) if failed else 'OK'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
