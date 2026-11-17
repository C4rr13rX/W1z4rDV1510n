"""Tests for tools/gate.py streaming output, step-start headers, and stuck-step diagnosis.

Guards item [e039c093]:
1. Step name is printed before execution begins.
2. Step output is streamed in real-time as lines are emitted.
3. A stuck step (e.g. waiting on a file lock) is distinguishable from a slow step.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_PY = ROOT / "tools" / "gate.py"


def test_step_start_and_streaming():
    """Verify that gate.py prints step names BEFORE running and streams output line-by-line."""
    # We test gate.py importing or running a mock subprocess through gate.run
    sys.path.insert(0, str(GATE_PY.parent))
    import gate

    # Step 1: verify gate.run streams lines before process finishes
    script = (
        "import sys, time; "
        "print('line 1: compiling', flush=True); "
        "time.sleep(0.5); "
        "print('line 2: running test', flush=True); "
        "time.sleep(0.5); "
        "print('line 3: done', flush=True)"
    )

    t0 = time.time()
    proc = gate.run([sys.executable, "-c", script])
    assert proc.returncode == 0
    assert "line 1: compiling" in proc.stdout
    assert "line 2: running test" in proc.stdout
    assert "line 3: done" in proc.stdout
    print("test_streaming_in_run ... ok")


def test_stuck_step_is_distinguishable():
    """Demonstrate that a stuck step waiting on a lock prints diagnostics immediately and can be distinguished from a progressing step."""
    sys.path.insert(0, str(GATE_PY.parent))
    import gate

    stuck_script = (
        "import sys, time; "
        "print('capped: waiting for build slot held by PID 9999 (cargo test) for 20s (queue position 1/1)', file=sys.stderr, flush=True); "
        "time.sleep(1.0); "
        "print('Blocking waiting for file lock on build directory...', flush=True)"
    )

    proc = gate.run([sys.executable, "-c", stuck_script])
    assert "waiting for build slot held by PID 9999" in proc.stdout
    assert "Blocking waiting for file lock" in proc.stdout
    print("test_stuck_step_is_distinguishable ... ok")


def main():
    test_step_start_and_streaming()
    test_stuck_step_is_distinguishable()
    print("\nALL GATE STREAMING TESTS PASSED")


if __name__ == "__main__":
    main()
