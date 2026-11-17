"""Tests for tools/capped.py BuildSlot FIFO ordering, holder diagnostics, and duplicate refusal.

Guards item [19c6d306]:
1. Grants the lock in arrival order (demonstrated by waiters acquiring in start order).
2. Prints holder's PID, command, and how long held to stderr.
3. Refuses duplicate invocations with identical command and target dir from the same worktree.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPPED_PY = ROOT / "tools" / "capped.py"


def test_duplicate_refusal():
    """Verify capped.py refuses a duplicate invocation from same worktree & target."""
    cmd1 = [sys.executable, str(CAPPED_PY), "--mb", "512", "--", sys.executable, "-c", "import time; time.sleep(3)"]
    p1 = subprocess.Popen(cmd1, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.8)
    try:
        assert p1.poll() is None, "Process 1 should be running"
        p2 = subprocess.Popen(cmd1, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out2, err2 = p2.communicate(timeout=5)
        assert p2.returncode == 1, f"Process 2 should have been refused with exit 1, got {p2.returncode}"
        assert f"PID {p1.pid}" in err2 or "refusing duplicate invocation" in err2, (
            f"Expected duplicate refusal message naming PID {p1.pid}, got: {err2}"
        )
        print("test_duplicate_refusal ... ok")
    finally:
        p1.terminate()
        p1.wait()


def test_holder_diagnostics_and_fifo_order():
    """Verify waiting processes print holder PID, command, and acquire in FIFO arrival order."""
    sys_temp = Path(tempfile.gettempdir())
    order_file = sys_temp / f"w1z4rd_test_fifo_order_{os.getpid()}.txt"
    order_file.unlink(missing_ok=True)
    test_prefix = f"test-build-slot-{os.getpid()}"

    worker_code = """
import sys, time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from capped import BuildSlot

label = sys.argv[2]
hold = float(sys.argv[3])
prefix = sys.argv[4]
order_path = sys.argv[5]

with BuildSlot(["cargo", label], Path.cwd(), prefix=prefix):
    if order_path != "none":
        with open(order_path, "a") as f:
            f.write(label + "\\n")
    time.sleep(hold)
"""

    try:
        # Start holder (holds for 2.0s)
        p_holder = subprocess.Popen(
            [sys.executable, "-c", worker_code, str(CAPPED_PY.parent), "holder", "2.0", test_prefix, "none"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.5)

        # Waiter 1 arrives first
        p_w1 = subprocess.Popen(
            [sys.executable, "-c", worker_code, str(CAPPED_PY.parent), "waiter1", "0.2", test_prefix, str(order_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.5)

        # Waiter 2 arrives second
        p_w2 = subprocess.Popen(
            [sys.executable, "-c", worker_code, str(CAPPED_PY.parent), "waiter2", "0.2", test_prefix, str(order_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        _, holder_err = p_holder.communicate(timeout=10)
        _, w1_err = p_w1.communicate(timeout=10)
        _, w2_err = p_w2.communicate(timeout=10)

        assert p_holder.returncode == 0, f"Holder failed: {holder_err}"
        assert p_w1.returncode == 0, f"Waiter 1 failed: {w1_err}"
        assert p_w2.returncode == 0, f"Waiter 2 failed: {w2_err}"

        # Check holder diagnostics in waiter 1 output
        assert f"held by PID {p_holder.pid}" in w1_err, f"Waiter 1 stderr should name holder PID {p_holder.pid}, got: {w1_err}"
        assert "cargo" in w1_err, f"Waiter 1 stderr should describe holder command, got: {w1_err}"

        # Check FIFO order: waiter1 arrived before waiter2, so waiter1 must have acquired first
        assert order_file.exists(), f"{order_file} was not written"
        order = [l.strip() for l in order_file.read_text().splitlines() if l.strip()]
        assert order == ["waiter1", "waiter2"], f"Expected acquisition order ['waiter1', 'waiter2'], got {order}"
        print("test_holder_diagnostics_and_fifo_order ... ok")
    finally:
        order_file.unlink(missing_ok=True)
        (sys_temp / f"{test_prefix}.lock").unlink(missing_ok=True)
        (sys_temp / f"{test_prefix}-holder.json").unlink(missing_ok=True)
        qdir = sys_temp / f"{test_prefix}-queue"
        if qdir.exists():
            for f in qdir.glob("*"):
                f.unlink(missing_ok=True)
            try:
                qdir.rmdir()
            except Exception:
                pass


def main():
    test_duplicate_refusal()
    test_holder_diagnostics_and_fifo_order()
    print("\nALL CAPPED BUILD SLOT TESTS PASSED")


if __name__ == "__main__":
    main()
