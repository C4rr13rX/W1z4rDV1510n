#!/usr/bin/env python3
"""Test live failure→repair→success learning and held-out transfer."""
from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Case:
    name: str
    instruction: str
    broken: str
    corrected: str
    function: str
    value: int
    factor: int
    offset: int


EXPERIENCE = Case(
    "nebula_offset_experience",
    "The previously unseen Nebula checksum rule multiplies the input by four "
    "and then ADDS nine. Repair the supplied Python function; subtraction is invalid.",
    "def nebula_checksum(value):\n    return value * 4 - 9\n",
    "def nebula_checksum(value):\n    return value * 4 + 9\n",
    "nebula_checksum", 5, 4, 9,
)

HELDOUT = Case(
    "quasar_offset_transfer",
    "For the new Quasar checksum described here, multiply the input by seven "
    "and then ADD thirteen. Correct this Python implementation, which subtracts.",
    "def quasar_checksum(datum):\n    return datum * 7 - 13\n",
    "def quasar_checksum(datum):\n    return datum * 7 + 13\n",
    "quasar_checksum", 3, 7, 13,
)

RELATION = {"kind": "replace_operator", "from": "-", "to": "+"}
QUERY_FIELDS = (1, 3, 5, 6, 10, 12)


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).rstrip(b"=").decode("ascii")


def unb64(value: str) -> str:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")


def request(endpoint: str, path: str, payload: dict | None,
            timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def execute(case: Case, source: str) -> tuple[bool, str]:
    expected = case.value * case.factor + case.offset
    harness = (
        source + f"\nassert {case.function}({case.value}) == {expected}, "
        f"'expected {expected}'\nprint('PASS')\n"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", harness], capture_output=True,
        text=True, timeout=15,
    )
    return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-1200:]


def environment() -> str:
    return json.dumps({
        "language": "python", "language_version": platform.python_version(),
        "operating_system": platform.system(), "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }, sort_keys=True, separators=(",", ":"))


def episode(case: Case) -> dict[int, str]:
    broken_ok, console_before = execute(case, case.broken)
    corrected_ok, console_after = execute(case, case.corrected)
    if broken_ok or not corrected_ok:
        raise RuntimeError(f"invalid experiential fixture: {case.name}")
    delta = "".join(difflib.unified_diff(
        case.broken.splitlines(True), case.corrected.splitlines(True),
        fromfile="before.py", tofile="after.py",
    ))
    return {
        1: case.instruction,
        2: case.broken,
        3: console_before,
        4: case.corrected,
        5: environment(),
        6: json.dumps({"status": "failure", "error_class": "AssertionError"}, sort_keys=True),
        7: delta,
        8: console_after,
        9: json.dumps({"transition": "failure_to_success", "verified": True}, sort_keys=True),
        10: case.broken,
        11: json.dumps(RELATION, sort_keys=True, separators=(",", ":")),
        12: case.instruction,
    }


def predict(endpoint: str, case: Case) -> dict:
    frames = episode(case)
    result = request(endpoint, "/brain/repair/predict", {
        "source": b64(case.broken),
        "relation_pool": 11,
        "streams": [
            {"pool_id": pool_id, "frame": b64(frames[pool_id])}
            for pool_id in QUERY_FIELDS
        ],
    })
    encoded = result.get("answer")
    answer = unb64(encoded) if isinstance(encoded, str) and encoded else ""
    passed, detail = execute(case, answer) if answer else (False, "no answer")
    return {
        "case": case.name, "answered": bool(answer), "executes": passed,
        "answer": answer, "detail": "" if passed else detail,
        "relation": unb64(result["relation"]) if result.get("relation") else None,
        "composition_error": result.get("composition_error"),
    }


def train_experience(endpoint: str, case: Case, repeats: int) -> dict:
    frames = episode(case)
    payload_episode = {
        "frames": [
            {"pool_id": pool_id, "frame": b64(frame)}
            for pool_id, frame in sorted(frames.items())
        ]
    }
    result = request(endpoint, "/brain/pretrain/batch", {
        "episodes": [payload_episode for _ in range(repeats)],
    }, timeout=600.0)
    if not result.get("ok") or result.get("accepted") != repeats:
        raise RuntimeError(f"experience batch rejected: {result}")
    return result


def run_retention(endpoint: str, output: Path) -> dict:
    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/programming_integrated_retention.py"),
        "--endpoint", endpoint, "--no-checkpoint", "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=2 * 3600, check=False)
    if run.returncode != 0 or not output.is_file():
        raise RuntimeError(f"foundation retention failed: {run.stderr[-2000:]}")
    return json.loads(output.read_text(encoding="utf-8"))


def retention_passed(report: dict) -> bool:
    after = report.get("after_debug") or {}
    foundation = after.get("foundation") or {}
    python = (after.get("python") or {}).get("summary") or {}
    debug = after.get("debug") or {}
    return (
        foundation.get("toddler") == foundation.get("toddler_total")
        and foundation.get("k12") == foundation.get("k12_total")
        and foundation.get("oov") == foundation.get("oov_total")
        and all(
            group.get("executes") == group.get("count")
            and group.get("syntax_valid") == group.get("count")
            for group in python.values()
        )
        and all(group.get("passed") == group.get("total") for group in debug.values())
    )


def run_enterprise_retention(endpoint: str, output: Path) -> dict:
    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/programming_enterprise_retention.py"),
        "--endpoint", endpoint, "--output", str(output), "--suite-timeout", "900",
    ], cwd=ROOT, capture_output=True, text=True, timeout=4 * 3600, check=False)
    if run.returncode != 0 or not output.is_file():
        raise RuntimeError(f"enterprise retention failed: {run.stderr[-2000:]}")
    return json.loads(output.read_text(encoding="utf-8"))


def begin_experience_transaction(endpoint: str, runtime: Path) -> tuple[Path, Path]:
    """Checkpoint and hard-link the pre-experience state until every gate passes."""
    brain_dir = runtime.resolve() / "brain"
    snapshot = brain_dir / "brain.bin"
    guard = brain_dir / "brain.experience-last-good.bin"
    metadata = brain_dir / "brain.experience-last-good.json"
    if guard.exists() or metadata.exists():
        raise RuntimeError(
            f"unresolved experiential transaction guard exists: {guard}"
        )
    checkpoint = request(endpoint, "/brain/checkpoint", {}, timeout=2 * 3600)
    if not checkpoint.get("ok"):
        raise RuntimeError(f"pre-experience checkpoint failed: {checkpoint}")
    reported = Path(str(checkpoint.get("path", ""))).resolve()
    if reported != snapshot.resolve():
        raise RuntimeError(
            f"checkpoint path {reported} does not match runtime snapshot {snapshot}"
        )
    if not snapshot.is_file():
        raise RuntimeError(f"checkpoint did not create snapshot: {snapshot}")
    os.link(snapshot, guard)
    metadata.write_text(json.dumps({
        "snapshot": str(snapshot), "guard": str(guard),
        "tick": checkpoint.get("tick"), "created_unix": time.time(),
        "recovery": "stop the node, replace brain.bin with this guard, then restart",
    }, indent=2) + "\n", encoding="utf-8")
    return guard, metadata


def commit_experience_transaction(endpoint: str, guard: Path, metadata: Path) -> dict:
    """Persist admitted learning before releasing its rollback snapshot."""
    checkpoint = request(endpoint, "/brain/checkpoint", {}, timeout=2 * 3600)
    if not checkpoint.get("ok"):
        raise RuntimeError(f"post-experience checkpoint failed: {checkpoint}")
    guard.unlink()
    metadata.unlink()
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--skip-retention", action="store_true")
    parser.add_argument("--enterprise-gate", action="store_true")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/experiential-generalization.json"))
    args = parser.parse_args()

    if args.train and args.runtime is None:
        parser.error("--train requires --runtime for transactional rollback protection")

    transaction = (
        begin_experience_transaction(args.endpoint, args.runtime)
        if args.train else None
    )

    retention_before = (
        run_retention(args.endpoint, args.output.with_name("experience-foundation-before.json"))
        if args.train and not args.skip_retention else None
    )
    tick_before = request(args.endpoint, "/brain/stats", None).get("tick")
    baseline = predict(args.endpoint, EXPERIENCE)
    training = train_experience(args.endpoint, EXPERIENCE, args.repeats) if args.train else None
    learned = predict(args.endpoint, EXPERIENCE)
    transfer = predict(args.endpoint, HELDOUT)
    tick_after = request(args.endpoint, "/brain/stats", None).get("tick")
    retention_after = (
        run_retention(args.endpoint, args.output.with_name("experience-foundation-after.json"))
        if args.train and not args.skip_retention else None
    )
    enterprise = (
        run_enterprise_retention(
            args.endpoint, args.output.with_name("experience-enterprise-after.json")
        ) if args.train and args.enterprise_gate else None
    )
    retention_ok = (
        True if retention_after is None else retention_passed(retention_after)
    )
    enterprise_ok = True if enterprise is None else bool(enterprise.get("passed"))
    passed = (
        learned["executes"] and transfer["executes"]
        if args.train else baseline["executes"] and transfer["executes"]
    ) and retention_ok and enterprise_ok
    report = {
        "passed": passed,
        "mode": "train-and-transfer" if args.train else "read-only",
        "baseline": baseline,
        "training": training,
        "learned_experience": learned,
        "heldout_transfer": transfer,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "retention_passed": retention_ok,
        "enterprise_after": enterprise,
        "enterprise_passed": enterprise_ok,
        "tick_before": tick_before,
        "tick_after": tick_after,
        "tick_delta": (tick_after - tick_before) if isinstance(tick_before, int) and isinstance(tick_after, int) else None,
        "updated_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    admitted_checkpoint = None
    if passed and transaction is not None:
        admitted_checkpoint = commit_experience_transaction(
            args.endpoint, transaction[0], transaction[1]
        )
        report["admitted_checkpoint"] = admitted_checkpoint
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": passed, "baseline": baseline["executes"],
        "learned": learned["executes"], "heldout_transfer": transfer["executes"],
        "retention_passed": retention_ok, "enterprise_passed": enterprise_ok,
        "tick_delta": report["tick_delta"],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
