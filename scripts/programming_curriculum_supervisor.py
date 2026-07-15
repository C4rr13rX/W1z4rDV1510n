#!/usr/bin/env python3
"""Durably supervise sequential direct-pretrain corpus phases.

The supervisor can attach to an already-running first phase.  A worker exit is
not treated as completion until its progress ledger reaches the configured
logical-row target.  Interrupted phases restart from their RAM offset while
checkpoint accounting resumes from the separately recorded durable offset.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from programming_integrated_retention import integrated_retention_passed


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Phase:
    name: str
    script_id: str
    corpus: Path
    rows: int
    repeats: int = 1


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except psutil.Error:
        # The process can exit between pid_exists and Process construction.
        return False


def publish(path: Path, payload: dict) -> None:
    """Atomically publish state despite transient Windows reader locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def phase_offsets(progress_path: Path) -> tuple[int, int]:
    progress = read_json(progress_path)
    ram = max(0, int(progress.get("ram_next_row", 0)))
    durable = max(0, min(ram, int(progress.get("durable_next_row", 0))))
    return ram, durable


def responsive_batch_size(configured: int, progress: dict,
                          max_lock_seconds: float) -> int:
    """Reduce future bulk size when any measured transaction blocked too long."""
    previous_size = int(
        progress.get("max_batch_size") or progress.get("last_batch_size") or 0
    )
    previous_seconds = float(
        progress.get("max_batch_seconds")
        or progress.get("last_batch_seconds") or 0.0
    )
    if (previous_size <= 0 or previous_seconds <= max_lock_seconds
            or max_lock_seconds <= 0):
        return configured
    scaled = int(previous_size * max_lock_seconds / previous_seconds)
    return max(1, min(configured, scaled))


def runtime_responsive_batch_size(runtime: Path, configured: int,
                                  progress: dict,
                                  max_lock_seconds: float) -> int:
    """Carry proven live-lock limits across corpus phase boundaries."""
    candidates = [
        responsive_batch_size(configured, read_json(path), max_lock_seconds)
        for path in runtime.glob("*.progress.json")
    ]
    candidates.append(
        responsive_batch_size(configured, progress, max_lock_seconds)
    )
    return min(candidates)


def ensure_last_good_guard(runtime: Path, phase: Phase, row: int) -> Path:
    """Hard-link the accepted snapshot until the next retention gate passes."""
    brain_dir = runtime / "brain"
    snapshot = brain_dir / "brain.bin"
    guard = brain_dir / "brain.last-good.bin"
    metadata = brain_dir / "brain.last-good.json"
    if guard.exists():
        existing = read_json(metadata)
        if (existing.get("phase") != phase.name
                or not isinstance(existing.get("row"), int)
                or existing["row"] > row):
            raise RuntimeError(
                "unresolved last-good snapshot guard exists: "
                f"{existing or guard}"
            )
        return guard
    if not snapshot.exists():
        raise RuntimeError(f"cannot guard missing snapshot: {snapshot}")
    os.link(snapshot, guard)
    publish(metadata, {
        "phase": phase.name,
        "row": row,
        "snapshot": str(snapshot),
        "guard": str(guard),
        "created_unix": time.time(),
    })
    return guard


def accept_last_good_guard(runtime: Path) -> None:
    """Discard the prior snapshot only after the new state passes its gates."""
    brain_dir = runtime / "brain"
    (brain_dir / "brain.last-good.bin").unlink(missing_ok=True)
    (brain_dir / "brain.last-good.json").unlink(missing_ok=True)


def guarded_block_target(runtime: Path, phase: Phase, current_row: int,
                         gate_rows: int) -> int:
    """Keep one immutable retention boundary across worker/supervisor restarts."""
    metadata = read_json(runtime / "brain" / "brain.last-good.json")
    start = metadata.get("row") if metadata.get("phase") == phase.name else None
    if not isinstance(start, int) or start > current_row:
        start = current_row
    return min(phase.rows, start + gate_rows)


def run_json_command(command: list[str], timeout: float = 3600.0) -> dict:
    run = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(
            f"gate command failed ({run.returncode}): {' '.join(command)}\n"
            f"stdout: {run.stdout[-4000:]}\nstderr: {run.stderr[-2000:]}"
        )
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"gate command produced no JSON: {' '.join(command)}")
    return json.loads(lines[-1])


def recall_command(args: argparse.Namespace, phase: Phase, runtime: Path,
                   rows: int, samples: int) -> list[str]:
    """Accept deterministic answers supervised by any durable prior corpus."""
    command = [
        sys.executable, "scripts/programming_corpus_recall.py", str(phase.corpus),
        "--endpoint", args.endpoint,
        "--start-row", "0", "--window-rows", str(rows),
        "--samples", str(samples), "--syntax", "none",
    ]
    accepted = {phase.corpus.resolve()}
    for progress_path in runtime.glob("*.progress.json"):
        progress = read_json(progress_path)
        if int(progress.get("durable_next_row") or 0) <= 0:
            continue
        corpus = Path(str(progress.get("corpus") or ""))
        if corpus.is_file():
            accepted.add(corpus.resolve())
    for corpus in sorted(accepted - {phase.corpus.resolve()}, key=str):
        command.extend(["--accepted-corpus", str(corpus)])
    return command


def run_completion_gate(args: argparse.Namespace, phase: Phase,
                        runtime: Path) -> dict:
    """Require corpus recall plus protected foundation/code execution."""
    recall = run_json_command(recall_command(args, phase, runtime, phase.rows, 64))
    if recall.get("accepted_trained_response") != recall.get("sampled"):
        raise RuntimeError(f"{phase.name} recall regression: {recall}")

    foundation = run_json_command([
        sys.executable, "scripts/programming_brain_eval.py",
        "--endpoint", args.endpoint,
    ])
    for passed_key, total_key in (
        ("toddler_exact", "toddler_total"),
        ("k12_trained_answer", "k12_total"),
        ("oov_honest", "oov_total"),
    ):
        if foundation.get(passed_key) != foundation.get(total_key):
            raise RuntimeError(f"foundation regression after {phase.name}: {foundation}")

    code = run_json_command([
        sys.executable, "scripts/programming_code_eval.py",
        "--endpoint", args.endpoint,
    ])
    for kind in ("trained", "novel_paraphrase"):
        group = (code.get("summary") or {}).get(kind) or {}
        if (group.get("executes") != group.get("count")
                or group.get("syntax_valid") != group.get("count")):
            raise RuntimeError(f"code regression after {phase.name}: {code}")

    typescript = run_json_command([
        sys.executable, "scripts/programming_typescript_enterprise.py",
        "--endpoint", args.endpoint, "--no-train",
        "--output", str(runtime / f"{phase.name}.typescript-gate.json"),
    ])
    for key in ("trained", "paraphrase"):
        group = typescript.get(key) or {}
        if group.get("executes") != group.get("total"):
            raise RuntimeError(
                f"TypeScript regression after {phase.name}: {typescript}"
            )
    ts_oov = typescript.get("oov_honesty") or {}
    if ts_oov.get("passed") != ts_oov.get("total"):
        raise RuntimeError(
            f"TypeScript OOV regression after {phase.name}: {typescript}"
        )

    enterprise = run_json_command([
        sys.executable, "scripts/programming_enterprise_retention.py",
        "--endpoint", args.endpoint,
        "--output", str(runtime / f"{phase.name}.enterprise-gate.json"),
        "--suite-timeout", "900",
    ], timeout=4 * 3600.0)
    if (not enterprise.get("passed")
            or enterprise.get("passed_suites") != enterprise.get("total_suites")
            or enterprise.get("tick_delta") != 0
            or enterprise.get("structure_unchanged") is not True):
        raise RuntimeError(
            f"enterprise regression after {phase.name}: {enterprise}"
        )

    report = {
        "phase": phase.name,
        "passed": True,
        "recall": recall,
        "foundation": foundation,
        "code": code,
        "typescript": typescript,
        "enterprise": enterprise,
        "updated_unix": time.time(),
    }
    publish(runtime / f"{phase.name}.completion-gate.json", report)
    return report


def run_midphase_gate(args: argparse.Namespace, phase: Phase,
                      runtime: Path, trained_rows: int) -> dict:
    """Protect retained knowledge before permitting the next corpus chunk."""
    recall = run_json_command(
        recall_command(args, phase, runtime, trained_rows, 32)
    )
    if recall.get("accepted_trained_response") != recall.get("sampled"):
        raise RuntimeError(
            f"{phase.name} midphase recall regression at {trained_rows}: {recall}"
        )
    foundation_path = runtime / f"{phase.name}.row-{trained_rows}.foundation.json"
    run_json_command([
        sys.executable, "scripts/programming_integrated_retention.py",
        "--endpoint", args.endpoint, "--no-checkpoint",
        "--output", str(foundation_path),
    ], timeout=2 * 3600.0)
    foundation = read_json(foundation_path)
    if not integrated_retention_passed(foundation):
        raise RuntimeError(
            f"integrated retention regression after {phase.name} row "
            f"{trained_rows}: {foundation}"
        )
    enterprise = run_json_command([
        sys.executable, "scripts/programming_enterprise_retention.py",
        "--endpoint", args.endpoint,
        "--output", str(runtime / f"{phase.name}.row-{trained_rows}.enterprise.json"),
        "--suite-timeout", "900",
    ], timeout=4 * 3600.0)
    if (not enterprise.get("passed")
            or enterprise.get("passed_suites") != enterprise.get("total_suites")
            or enterprise.get("tick_delta") != 0
            or enterprise.get("structure_unchanged") is not True):
        raise RuntimeError(
            f"enterprise midphase regression after {phase.name} row "
            f"{trained_rows}: {enterprise}"
        )
    report = {
        "phase": phase.name,
        "trained_rows": trained_rows,
        "passed": True,
        "recall": recall,
        "foundation": foundation,
        "enterprise": enterprise,
        "updated_unix": time.time(),
    }
    publish(runtime / f"{phase.name}.row-{trained_rows}.retention-gate.json", report)
    return report


def run_phase(args: argparse.Namespace, phase: Phase, runtime: Path,
              status_path: Path, block_target_row: int) -> int:
    progress = runtime / f"{phase.name}.progress.json"
    ram, durable = phase_offsets(progress)
    if ram >= phase.rows:
        return 0
    stdout_path = runtime / f"{phase.name}.stdout.log"
    stderr_path = runtime / f"{phase.name}.stderr.log"
    batch_size = runtime_responsive_batch_size(
        runtime, args.batch_size, read_json(progress),
        args.max_live_lock_seconds
    )
    command = [
        sys.executable, "-m", "tools.training_standard.drive_corpora_brain",
        "--brain", args.endpoint,
        "--script", phase.script_id,
        "--input-path", str(phase.corpus),
        "--repeats", str(phase.repeats),
        "--direct-pretrain",
        "--start-row", str(ram),
        "--limit-rows", str(max(0, block_target_row - ram)),
        "--durable-start-row", str(durable),
        "--batch-size", str(batch_size),
        "--max-batch-seconds", str(args.max_live_lock_seconds),
        "--inter-post-sleep", str(args.inter_batch_yield_seconds),
        "--checkpoint-rows", str(args.checkpoint_rows),
        "--wal-durable",
        "--feature-policy", "auto",
        "--midcheck-rows", "0",
        "--no-sleep-between",
        "--progress-path", str(progress),
    ]
    worker_pid_path = runtime / f"{phase.name}.pid"
    with stdout_path.open("a", encoding="utf-8") as stdout, \
            stderr_path.open("a", encoding="utf-8") as stderr:
        worker = subprocess.Popen(
            command, cwd=ROOT, stdout=stdout, stderr=stderr,
        )
        worker_pid_path.write_text(f"{worker.pid}\n", encoding="ascii")
        try:
            while True:
                code = worker.poll()
                ram, durable = phase_offsets(progress)
                publish(status_path, {
                    "state": "running", "phase": phase.name,
                    "worker_pid": worker.pid, "ram_next_row": ram,
                    "durable_next_row": durable,
                    "batch_size": batch_size,
                    "block_target_row": block_target_row,
                    "updated_unix": time.time(),
                })
                if code is not None:
                    return code
                time.sleep(max(1.0, args.poll_seconds))
        finally:
            try:
                recorded_pid = worker_pid_path.read_text(encoding="ascii").strip()
                if recorded_pid == str(worker.pid):
                    worker_pid_path.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--attach-pid", type=int, default=0)
    parser.add_argument("--attach-phase", default="")
    parser.add_argument(
        "--gate-only-phase", default="",
        help="gate the phase's current durable boundary and exit without training",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inter-batch-yield-seconds", type=float, default=0.1)
    parser.add_argument("--max-live-lock-seconds", type=float, default=8.0)
    parser.add_argument("--checkpoint-rows", type=int, default=16384)
    parser.add_argument("--gate-rows", type=int, default=16384)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument(
        "--corpus-root", type=Path,
        default=Path(r"D:\w1z4rdv1510n-data\training"),
        help="directory containing the generated programming corpora",
    )
    parser.add_argument(
        "--include-seed-corpora", action="store_true",
        help="also train the canonical algorithms and GSM8K phases used by a fresh brain",
    )
    args = parser.parse_args()

    corpus_root = args.corpus_root.resolve()
    phases = [
        Phase("mathinstruct-domain-safe", "reasoning_math_001",
              corpus_root / "mathinstruct.jsonl", 245_323),
        Phase("metamathqa-domain-safe", "reasoning_math_001",
              corpus_root / "metamathqa.jsonl", 385_524),
        Phase("csn-python-full", "programming_literacy_python_001",
              corpus_root / "csn_python_full.jsonl", 421_477),
        Phase("csn-python-para5", "programming_literacy_python_001",
              corpus_root / "csn_python_full_para5.jsonl",
              2_028_816),
        Phase("jupyter-scientific-full", "domain_scientific_python_001",
              corpus_root / "jupyter_scientific_full.jsonl",
              690_175),
        Phase("jupyter-scientific-para4", "domain_scientific_python_001",
              corpus_root / "jupyter_scientific_para4.jsonl",
              2_760_496),
        Phase("jupyter-scientific-partial", "domain_scientific_python_001",
              corpus_root / "jupyter_scientific_partial.jsonl",
              206_948),
    ]
    if args.include_seed_corpora:
        phases[0:0] = [
            Phase("canonical-algorithms", "dsa_classical_001",
                  corpus_root / "the_algorithms_full.jsonl", 1_953, repeats=4),
            Phase("gsm8k-domain-safe", "reasoning_math_001",
                  corpus_root / "gsm8k.jsonl", 7_473),
        ]
    missing = [str(phase.corpus) for phase in phases if not phase.corpus.is_file()]
    if missing:
        parser.error("missing corpus files: " + ", ".join(missing))
    runtime = args.runtime.resolve()
    status_path = runtime / "curriculum-supervisor.status.json"

    if args.gate_only_phase:
        phase = next(
            (item for item in phases if item.name == args.gate_only_phase), None
        )
        if phase is None:
            parser.error(f"unknown --gate-only-phase: {args.gate_only_phase}")
        ram, durable = phase_offsets(runtime / f"{phase.name}.progress.json")
        if ram <= 0 or durable != ram:
            parser.error(
                f"gate-only requires a positive durable boundary; ram={ram}, "
                f"durable={durable}"
            )
        publish(status_path, {
            "state": "gate_only_benchmarking", "phase": phase.name,
            "ram_next_row": ram, "durable_next_row": durable,
            "updated_unix": time.time(),
        })
        try:
            if ram >= phase.rows:
                run_completion_gate(args, phase, runtime)
            else:
                run_midphase_gate(args, phase, runtime, ram)
        except (RuntimeError, subprocess.TimeoutExpired,
                json.JSONDecodeError) as exc:
            publish(status_path, {
                "state": "gate_only_failed", "phase": phase.name,
                "ram_next_row": ram, "durable_next_row": durable,
                "error": str(exc), "updated_unix": time.time(),
            })
            return 1
        accept_last_good_guard(runtime)
        publish(status_path, {
            "state": "gate_only_complete", "phase": phase.name,
            "ram_next_row": ram, "durable_next_row": durable,
            "updated_unix": time.time(),
        })
        return 0

    if args.attach_pid:
        attach_phase = next(
            (phase for phase in phases if phase.name == args.attach_phase),
            next((phase for phase in phases
                  if phase_offsets(runtime / f"{phase.name}.progress.json")[0]
                  < phase.rows), phases[0]),
        )
        attached_start, _ = phase_offsets(
            runtime / f"{attach_phase.name}.progress.json"
        )
        publish(status_path, {"state": "attached", "pid": args.attach_pid,
                              "phase": attach_phase.name,
                              "ram_next_row": attached_start,
                              "updated_unix": time.time()})
        while process_alive(args.attach_pid):
            time.sleep(max(1.0, args.poll_seconds))
        attached_ram, attached_durable = phase_offsets(
            runtime / f"{attach_phase.name}.progress.json"
        )
        if attached_ram > attached_start and attached_ram < attach_phase.rows:
            if attached_durable != attached_ram:
                publish(status_path, {"state": "midphase_gate_failed",
                                      "phase": attach_phase.name,
                                      "error": "attached worker ended before durable boundary",
                                      "ram_next_row": attached_ram,
                                      "durable_next_row": attached_durable,
                                      "updated_unix": time.time()})
                return 1
            publish(status_path, {"state": "midphase_benchmarking",
                                  "phase": attach_phase.name,
                                  "ram_next_row": attached_ram,
                                  "durable_next_row": attached_durable,
                                  "updated_unix": time.time()})
            try:
                run_midphase_gate(args, attach_phase, runtime, attached_ram)
            except (RuntimeError, subprocess.TimeoutExpired,
                    json.JSONDecodeError) as exc:
                publish(status_path, {"state": "midphase_gate_failed",
                                      "phase": attach_phase.name,
                                      "ram_next_row": attached_ram,
                                      "durable_next_row": attached_durable,
                                      "error": str(exc),
                                      "updated_unix": time.time()})
                return 1
            accept_last_good_guard(runtime)

    for phase in phases:
        restarts = 0
        while True:
            ram, durable = phase_offsets(runtime / f"{phase.name}.progress.json")
            if ram >= phase.rows:
                gate_path = runtime / f"{phase.name}.completion-gate.json"
                if not read_json(gate_path).get("passed"):
                    publish(status_path, {"state": "benchmarking",
                                          "phase": phase.name,
                                          "updated_unix": time.time()})
                    try:
                        run_completion_gate(args, phase, runtime)
                    except (RuntimeError, subprocess.TimeoutExpired,
                            json.JSONDecodeError) as exc:
                        publish(status_path, {"state": "gate_failed",
                                              "phase": phase.name,
                                              "error": str(exc),
                                              "updated_unix": time.time()})
                        return 1
                publish(status_path, {"state": "complete", "phase": phase.name,
                                      "ram_next_row": ram,
                                      "durable_next_row": durable,
                                      "updated_unix": time.time()})
                accept_last_good_guard(runtime)
                break
            publish(status_path, {"state": "running", "phase": phase.name,
                                  "ram_next_row": ram,
                                  "durable_next_row": durable,
                                  "restart": restarts,
                                  "updated_unix": time.time()})
            ensure_last_good_guard(runtime, phase, ram)
            block_target = guarded_block_target(
                runtime, phase, ram, args.gate_rows
            )
            code = run_phase(args, phase, runtime, status_path, block_target)
            ram_after, durable_after = phase_offsets(
                runtime / f"{phase.name}.progress.json"
            )
            if code == 0 and ram_after >= phase.rows:
                continue
            if code == 0 and ram_after > ram and durable_after == ram_after:
                publish(status_path, {"state": "midphase_benchmarking",
                                      "phase": phase.name,
                                      "ram_next_row": ram_after,
                                      "durable_next_row": durable_after,
                                      "updated_unix": time.time()})
                try:
                    run_midphase_gate(args, phase, runtime, ram_after)
                except (RuntimeError, subprocess.TimeoutExpired,
                        json.JSONDecodeError) as exc:
                    publish(status_path, {"state": "midphase_gate_failed",
                                          "phase": phase.name,
                                          "ram_next_row": ram_after,
                                          "durable_next_row": durable_after,
                                          "error": str(exc),
                                          "updated_unix": time.time()})
                    return 1
                accept_last_good_guard(runtime)
                continue
            restarts += 1
            if restarts > args.max_restarts or ram_after <= ram:
                publish(status_path, {"state": "failed", "phase": phase.name,
                                      "exit_code": code, "ram_next_row": ram_after,
                                      "durable_next_row": durable_after,
                                      "restarts": restarts,
                                      "updated_unix": time.time()})
                return 1
            time.sleep(max(1.0, args.poll_seconds))

    publish(status_path, {"state": "all_complete", "updated_unix": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
