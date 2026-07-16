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
import urllib.request
import hashlib
from urllib.parse import urlparse
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


def canary_quarantine_path(runtime: Path) -> Path:
    return runtime / "curriculum-canary-quarantine.json"


def assert_training_not_quarantined(runtime: Path) -> None:
    """Fail closed until a rejected interval is explicitly rolled back."""
    quarantine = read_json(canary_quarantine_path(runtime))
    if quarantine:
        raise RuntimeError(
            "unresolved continuous-canary quarantine; restore the guarded "
            f"snapshot and progress ledger before training: {quarantine}"
        )


def restore_canary_quarantine(runtime: Path) -> dict:
    """Restore the accepted snapshot and ledger after a stopped-node failure."""
    marker = canary_quarantine_path(runtime)
    quarantine = read_json(marker)
    if not quarantine:
        raise RuntimeError(f"no continuous-canary quarantine exists: {marker}")
    node_pid_path = runtime / "node.pid"
    try:
        node_pid = int(node_pid_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        node_pid = 0
    if process_alive(node_pid):
        raise RuntimeError(
            f"brain server PID {node_pid} is still running; stop it before rollback"
        )
    last_good = quarantine.get("last_good") or read_json(
        runtime / "brain" / "brain.last-good.json"
    )
    phase = str(last_good.get("phase") or quarantine.get("phase") or "")
    row = last_good.get("row")
    if not phase or not isinstance(row, int) or row < 0:
        raise RuntimeError(f"quarantine lacks valid last-good phase/row: {quarantine}")
    brain_dir = runtime / "brain"
    snapshot = brain_dir / "brain.bin"
    guard = brain_dir / "brain.last-good.bin"
    if not guard.is_file():
        raise RuntimeError(f"quarantine guard is missing: {guard}")
    same_snapshot = snapshot.exists() and os.path.samefile(snapshot, guard)
    if same_snapshot:
        guard.unlink()
    else:
        snapshot.unlink(missing_ok=True)
        os.replace(guard, snapshot)
    (brain_dir / "brain.wal").unlink(missing_ok=True)
    progress_path = runtime / f"{phase}.progress.json"
    progress = read_json(progress_path)
    progress.update({
        "ram_next_row": row,
        "durable_next_row": row,
        "accepted_episodes": 0,
        "restored_from_canary_quarantine": True,
        "updated_unix": time.time(),
    })
    publish(progress_path, progress)
    (brain_dir / "brain.last-good.json").unlink(missing_ok=True)
    marker.unlink()
    return {"phase": phase, "row": row, "snapshot": str(snapshot)}


def stop_runtime_node(runtime: Path, timeout: float = 60.0) -> int:
    pid_path = runtime / "node.pid"
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        pid = 0
    if not process_alive(pid):
        return pid
    process = psutil.Process(pid)
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
    return pid


def start_runtime_node(runtime: Path, executable: Path, endpoint: str,
                       timeout: float = 900.0) -> subprocess.Popen:
    executable = executable.resolve()
    if not executable.is_file():
        raise RuntimeError(f"brain server executable is missing: {executable}")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise RuntimeError(f"automatic node recovery requires an HTTP host and port: {endpoint}")
    identity = runtime / "brain" / "brain.identity.toml"
    deployment = runtime / "brain.deployment.toml"
    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(runtime / "node"),
        "W1Z4RD_NODE_BRAIN_DIR": str(runtime / "brain"),
        "W1Z4RD_BRAIN_IDENTITY": str(identity),
        "W1Z4RD_BRAIN_PORT": str(parsed.port),
        "W1Z4RD_BRAIN_BIND": parsed.hostname,
        # Eager housekeeping scans every terminal on every tick.  At corpus
        # scale that made a mathematically routine decay pass consume >95% of
        # training CPU.  Lazy decay applies the same compounded decay when a
        # neuron is next touched, so it preserves the learning result without
        # the O(total_terminals) scan.
        "W1Z4RD_TICK_HOUSEKEEPING": "lazy",
        # Promotion remains atom-grounded but is deferred to the end of the
        # moment, avoiding repeated intermediate graph maintenance.
        "W1Z4RD_DEFER_PROMOTION": "1",
    })
    if deployment.is_file():
        env["W1Z4RD_BRAIN_DEPLOYMENT"] = str(deployment)
    stdout = (runtime / "node-auto-recovery.stdout.log").open("ab")
    stderr = (runtime / "node-auto-recovery.stderr.log").open("ab")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(executable)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
        creationflags=flags,
    )
    (runtime / "node.pid").write_text(f"{process.pid}\n", encoding="ascii")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"recovered brain server exited with {process.returncode}")
        try:
            endpoint_json(endpoint, "/brain/stats", timeout=2.0)
            return process
        except Exception:
            time.sleep(1.0)
    process.kill()
    raise RuntimeError(f"recovered brain server did not become ready within {timeout}s")


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


def append_health_event(runtime: Path, event: dict) -> None:
    """Append an auditable candidate-boundary result without rewriting history."""
    path = runtime / "curriculum-health.jsonl"
    payload = dict(event)
    payload.setdefault("updated_unix", time.time())
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def latest_passing_canary_row(runtime: Path, phase: str, floor: int) -> int:
    """Return the latest durable behavioral boundary known to be green."""
    latest = floor
    path = runtime / "curriculum-health.jsonl"
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = event.get("trained_rows")
                if (event.get("kind") == "continuous_canary"
                        and event.get("phase") == phase
                        and event.get("passed") is True
                        and isinstance(row, int)):
                    latest = max(latest, row)
    except (FileNotFoundError, OSError):
        pass
    return latest


def deferred_intervals_path(runtime: Path) -> Path:
    return runtime / "curriculum-deferred-intervals.jsonl"


def deferred_interval_id(phase: str, start_row: int, end_row: int) -> str:
    return f"{phase}:{start_row}:{end_row}"


def preserve_deferred_base(runtime: Path, interval_id: str) -> Path:
    """Keep the exact causal starting snapshot for later interval bisection."""
    guard = runtime / "brain" / "brain.last-good.bin"
    if not guard.is_file():
        raise RuntimeError(f"cannot preserve deferred base without guard: {guard}")
    digest = hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]
    directory = runtime / "deferred" / digest
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / "brain.base.bin"
    if not base.exists():
        os.link(guard, base)
    return base


def append_deferred_event(runtime: Path, event: dict) -> None:
    payload = dict(event)
    payload.setdefault("updated_unix", time.time())
    path = deferred_intervals_path(runtime)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def unresolved_deferred_intervals(runtime: Path, phase: str | None = None) -> list[dict]:
    """Fold the append-only defer/resolve ledger into current obligations."""
    current: dict[str, dict] = {}
    try:
        with deferred_intervals_path(runtime).open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                interval_id = event.get("interval_id")
                if not isinstance(interval_id, str) or not interval_id:
                    continue
                if event.get("status") == "resolved":
                    current.pop(interval_id, None)
                elif event.get("status") == "deferred":
                    current[interval_id] = event
    except (FileNotFoundError, OSError):
        pass
    rows = list(current.values())
    if phase is not None:
        rows = [row for row in rows if row.get("phase") == phase]
    return sorted(rows, key=lambda row: (
        str(row.get("phase") or ""), int(row.get("start_row") or 0)
    ))


def record_deferred_failure(runtime: Path, phase: Phase, candidate_row: int,
                            durable_row: int, error: str, reason: str) -> dict:
    """Persist one failed interval before any rollback can erase its evidence."""
    last_good = read_json(runtime / "brain" / "brain.last-good.json")
    suspect_start = latest_passing_canary_row(
        runtime, phase.name, int(last_good.get("row") or 0)
    )
    interval_id = deferred_interval_id(phase.name, suspect_start, candidate_row)
    base_snapshot = preserve_deferred_base(runtime, interval_id)
    event = {
        "interval_id": interval_id,
        "phase": phase.name,
        "start_row": suspect_start,
        "end_row": candidate_row,
        "base_snapshot": str(base_snapshot),
        "base_row": int(last_good.get("row") or 0),
        "reason": reason,
        "error": error,
    }
    append_health_event(runtime, {
        "kind": reason,
        "phase": phase.name,
        "trained_rows": candidate_row,
        "passed": False,
        "suspect_start_row": suspect_start,
        "suspect_end_row": candidate_row,
        "last_good": last_good,
        "error": error,
    })
    if not any(
        row.get("interval_id") == interval_id
        for row in unresolved_deferred_intervals(runtime)
    ):
        append_deferred_event(runtime, {**event, "status": "deferred"})
    publish(canary_quarantine_path(runtime), {
        "state": reason,
        "candidate_row": candidate_row,
        "suspect_start_row": suspect_start,
        "suspect_end_row": candidate_row,
        "durable_next_row": durable_row,
        "last_good": last_good,
        "error": error,
        "created_unix": time.time(),
        **event,
    })
    return event


def endpoint_json(endpoint: str, path: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(endpoint.rstrip("/") + path)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def topology_delta(before: dict, after: dict) -> dict:
    fields = (
        "tick", "pool_count", "total_neurons", "total_concepts",
        "total_binding", "total_terminals",
    )
    return {
        field: int(after.get(field, 0)) - int(before.get(field, 0))
        for field in fields
    }


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
    for interval in unresolved_deferred_intervals(runtime, phase.name):
        command.extend([
            "--skip-range",
            f"{int(interval['start_row'])}:{int(interval['end_row'])}",
        ])
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


def run_continuous_canary(args: argparse.Namespace, phase: Phase,
                          runtime: Path, trained_rows: int) -> dict:
    """Fast read-only drift screen while the corpus worker keeps advancing."""
    progress_path = runtime / f"{phase.name}.progress.json"
    rows_before = phase_offsets(progress_path)
    stats_before = endpoint_json(args.endpoint, "/brain/stats")
    recall = run_json_command(
        recall_command(args, phase, runtime, trained_rows, 8), timeout=900.0
    )
    if recall.get("accepted_trained_response") != recall.get("sampled"):
        raise RuntimeError(f"continuous recall regression: {recall}")
    foundation = run_json_command([
        sys.executable, "scripts/programming_brain_eval.py",
        "--endpoint", args.endpoint,
    ], timeout=900.0)
    for passed_key, total_key in (
        ("toddler_exact", "toddler_total"),
        ("k12_trained_answer", "k12_total"),
        ("oov_honest", "oov_total"),
    ):
        if foundation.get(passed_key) != foundation.get(total_key):
            raise RuntimeError(f"continuous foundation regression: {foundation}")
    code = run_json_command([
        sys.executable, "scripts/programming_code_eval.py",
        "--endpoint", args.endpoint,
    ], timeout=900.0)
    for kind in ("trained", "novel_paraphrase"):
        group = (code.get("summary") or {}).get(kind) or {}
        if (group.get("executes") != group.get("count")
                or group.get("syntax_valid") != group.get("count")):
            raise RuntimeError(f"continuous code regression: {code}")
    stats_after = endpoint_json(args.endpoint, "/brain/stats")
    rows_after = phase_offsets(progress_path)
    report = {
        "kind": "continuous_canary", "phase": phase.name,
        "trained_rows": trained_rows, "passed": True,
        "recall": {
            "accepted": recall.get("accepted_trained_response"),
            "sampled": recall.get("sampled"),
        },
        "foundation": {
            "toddler": foundation.get("toddler_exact"),
            "k12": foundation.get("k12_trained_answer"),
            "oov": foundation.get("oov_honest"),
        },
        "code": code.get("summary"),
        "concurrent_training": {
            "ram_rows_before": rows_before[0],
            "ram_rows_after": rows_after[0],
            "durable_rows_before": rows_before[1],
            "durable_rows_after": rows_after[1],
        },
        "topology_before": stats_before,
        "topology_after": stats_after,
        "topology_delta": topology_delta(stats_before, stats_after),
    }
    append_health_event(runtime, report)
    return report


def run_phase(args: argparse.Namespace, phase: Phase, runtime: Path,
              status_path: Path, block_target_row: int) -> int:
    progress = runtime / f"{phase.name}.progress.json"
    ram, durable = phase_offsets(progress)
    if ram >= phase.rows:
        return 0
    stdout_path = runtime / f"{phase.name}.stdout.log"
    stderr_path = runtime / f"{phase.name}.stderr.log"
    batch_size = args.batch_size
    lock_chunk_size = runtime_responsive_batch_size(
        runtime, args.lock_chunk_size, read_json(progress),
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
        "--lock-chunk-size", str(lock_chunk_size),
        "--max-batch-seconds", str(args.max_live_lock_seconds),
        "--inter-post-sleep", str(args.inter_batch_yield_seconds),
        "--checkpoint-rows", str(args.checkpoint_rows),
        "--wal-durable",
        "--feature-policy", "auto",
        "--midcheck-rows", "0",
        "--no-sleep-between",
        "--progress-path", str(progress),
    ]
    for interval in unresolved_deferred_intervals(runtime, phase.name):
        command.extend([
            "--skip-range",
            f"{int(interval['start_row'])}:{int(interval['end_row'])}",
        ])
    worker_pid_path = runtime / f"{phase.name}.pid"
    next_canary = (
        ((ram // args.canary_rows) + 1) * args.canary_rows
        if args.canary_rows > 0 else None
    )
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
                    "lock_chunk_size": lock_chunk_size,
                    "block_target_row": block_target_row,
                    "updated_unix": time.time(),
                })
                if (code is None and next_canary is not None
                        and ram >= next_canary):
                    candidate_row = ram
                    publish(status_path, {
                        "state": "continuous_canary", "phase": phase.name,
                        "worker_pid": worker.pid, "ram_next_row": ram,
                        "durable_next_row": durable,
                        "canary_row": candidate_row,
                        "block_target_row": block_target_row,
                        "updated_unix": time.time(),
                    })
                    try:
                        run_continuous_canary(
                            args, phase, runtime, candidate_row
                        )
                    except (RuntimeError, subprocess.TimeoutExpired,
                            json.JSONDecodeError) as exc:
                        worker.terminate()
                        try:
                            worker.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait(timeout=30)
                        record_deferred_failure(
                            runtime, phase, candidate_row, durable, str(exc),
                            "continuous_canary_failed",
                        )
                        publish(status_path, {
                            "state": "continuous_canary_failed",
                            "phase": phase.name,
                            "ram_next_row": candidate_row,
                            "durable_next_row": durable,
                            "error": str(exc),
                            "updated_unix": time.time(),
                        })
                        return 86
                    while next_canary <= candidate_row:
                        next_canary += args.canary_rows
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


def perform_automatic_recovery(args: argparse.Namespace, phase: Phase,
                               runtime: Path, status_path: Path) -> bool:
    if not args.auto_quarantine_recovery:
        return False
    publish(status_path, {
        "state": "automatic_quarantine_recovery",
        "phase": phase.name,
        "updated_unix": time.time(),
    })
    try:
        stop_runtime_node(runtime)
        restored = restore_canary_quarantine(runtime)
        start_runtime_node(runtime, args.node_bin, args.endpoint)
    except (RuntimeError, OSError, psutil.Error) as exc:
        publish(status_path, {
            "state": "automatic_quarantine_recovery_failed",
            "phase": phase.name,
            "error": str(exc),
            "updated_unix": time.time(),
        })
        return False
    append_health_event(runtime, {
        "kind": "automatic_quarantine_recovery",
        "phase": phase.name,
        "passed": True,
        "restored": restored,
    })
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--attach-pid", type=int, default=0)
    parser.add_argument("--attach-phase", default="")
    parser.add_argument(
        "--restart-node-after-attach", action="store_true",
        help="restart the node at the attached worker's durable boundary before gating",
    )
    parser.add_argument(
        "--gate-only-phase", default="",
        help="gate the phase's current durable boundary and exit without training",
    )
    parser.add_argument(
        "--restore-canary-quarantine", action="store_true",
        help="with the brain server stopped, restore last-good state and rewind progress",
    )
    parser.add_argument(
        "--auto-quarantine-recovery", action="store_true",
        help="on canary failure, restart the node, rollback, defer the suspect range, and continue",
    )
    parser.add_argument(
        "--node-bin", type=Path,
        help="brain server executable required by --auto-quarantine-recovery",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lock-chunk-size", type=int, default=12)
    parser.add_argument("--inter-batch-yield-seconds", type=float, default=0.0)
    parser.add_argument("--max-live-lock-seconds", type=float, default=8.0)
    parser.add_argument("--checkpoint-rows", type=int, default=131072)
    parser.add_argument("--gate-rows", type=int, default=131072)
    parser.add_argument(
        "--canary-rows", type=int, default=16384,
        help="run fast read-only drift checks while training continues; 0 disables",
    )
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
    if (args.auto_quarantine_recovery or args.restart_node_after_attach) \
            and args.node_bin is None:
        parser.error(
            "--auto-quarantine-recovery and --restart-node-after-attach "
            "require --node-bin"
        )

    runtime = args.runtime.resolve()
    status_path = runtime / "curriculum-supervisor.status.json"
    if args.restore_canary_quarantine:
        try:
            restored = restore_canary_quarantine(runtime)
        except RuntimeError as exc:
            publish(status_path, {
                "state": "canary_restore_failed", "error": str(exc),
                "updated_unix": time.time(),
            })
            return 1
        publish(status_path, {
            "state": "canary_restore_complete", **restored,
            "updated_unix": time.time(),
        })
        return 0

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
        assert_training_not_quarantined(runtime)
        attach_phase = next(
            (phase for phase in phases if phase.name == args.attach_phase),
            next((phase for phase in phases
                  if phase_offsets(runtime / f"{phase.name}.progress.json")[0]
                  < phase.rows), phases[0]),
        )
        attached_start, _ = phase_offsets(
            runtime / f"{attach_phase.name}.progress.json"
        )
        attach_recovered = False
        next_attached_canary = (
            ((attached_start // args.canary_rows) + 1) * args.canary_rows
            if args.canary_rows > 0 else None
        )
        publish(status_path, {"state": "attached", "pid": args.attach_pid,
                              "phase": attach_phase.name,
                              "ram_next_row": attached_start,
                              "updated_unix": time.time()})
        while process_alive(args.attach_pid):
            attached_ram, attached_durable = phase_offsets(
                runtime / f"{attach_phase.name}.progress.json"
            )
            publish(status_path, {
                "state": "attached",
                "pid": args.attach_pid,
                "phase": attach_phase.name,
                "ram_next_row": attached_ram,
                "durable_next_row": attached_durable,
                "updated_unix": time.time(),
            })
            if (next_attached_canary is not None
                    and attached_durable >= next_attached_canary):
                candidate_row = next_attached_canary
                try:
                    run_continuous_canary(
                        args, attach_phase, runtime, candidate_row
                    )
                except (RuntimeError, subprocess.TimeoutExpired,
                        json.JSONDecodeError) as exc:
                    try:
                        attached_process = psutil.Process(args.attach_pid)
                        attached_process.terminate()
                        attached_process.wait(timeout=30)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
                    record_deferred_failure(
                        runtime, attach_phase, candidate_row,
                        attached_durable, str(exc),
                        "attached_continuous_canary_failed",
                    )
                    if not perform_automatic_recovery(
                            args, attach_phase, runtime, status_path):
                        return 1
                    attach_recovered = True
                    break
                while next_attached_canary <= attached_durable:
                    next_attached_canary += args.canary_rows
            time.sleep(max(1.0, args.poll_seconds))
        attached_ram, attached_durable = phase_offsets(
            runtime / f"{attach_phase.name}.progress.json"
        )
        if args.restart_node_after_attach and not attach_recovered:
            if attached_durable != attached_ram:
                publish(status_path, {
                    "state": "attached_restart_refused",
                    "phase": attach_phase.name,
                    "ram_next_row": attached_ram,
                    "durable_next_row": attached_durable,
                    "error": "node restart requires an exact durable boundary",
                    "updated_unix": time.time(),
                })
                return 1
            try:
                stop_runtime_node(runtime)
                start_runtime_node(runtime, args.node_bin, args.endpoint)
            except (RuntimeError, OSError, psutil.Error) as exc:
                publish(status_path, {
                    "state": "attached_restart_failed",
                    "phase": attach_phase.name,
                    "error": str(exc),
                    "updated_unix": time.time(),
                })
                return 1
            append_health_event(runtime, {
                "kind": "attached_boundary_node_restart",
                "phase": attach_phase.name,
                "trained_rows": attached_ram,
                "passed": True,
                "tick_housekeeping": "lazy",
                "defer_promotion": True,
            })
        if (not attach_recovered and attached_ram > attached_start
                and attached_ram < attach_phase.rows):
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
                record_deferred_failure(
                    runtime, attach_phase, attached_ram, attached_durable,
                    str(exc), "attached_midphase_gate_failed",
                )
                publish(status_path, {"state": "midphase_gate_failed",
                                      "phase": attach_phase.name,
                                      "ram_next_row": attached_ram,
                                      "durable_next_row": attached_durable,
                                      "error": str(exc),
                                      "updated_unix": time.time()})
                if not perform_automatic_recovery(
                        args, attach_phase, runtime, status_path):
                    return 1
            else:
                accept_last_good_guard(runtime)

    assert_training_not_quarantined(runtime)
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
                        record_deferred_failure(
                            runtime, phase, ram, durable, str(exc),
                            "completion_gate_failed",
                        )
                        publish(status_path, {"state": "gate_failed",
                                              "phase": phase.name,
                                              "error": str(exc),
                                              "updated_unix": time.time()})
                        if perform_automatic_recovery(
                                args, phase, runtime, status_path):
                            restarts = 0
                            continue
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
            if code == 86:
                if perform_automatic_recovery(
                        args, phase, runtime, status_path):
                    restarts = 0
                    continue
                return 1
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
                    record_deferred_failure(
                        runtime, phase, ram_after, durable_after, str(exc),
                        "midphase_gate_failed",
                    )
                    publish(status_path, {"state": "midphase_gate_failed",
                                          "phase": phase.name,
                                          "ram_next_row": ram_after,
                                          "durable_next_row": durable_after,
                                          "error": str(exc),
                                          "updated_unix": time.time()})
                    if perform_automatic_recovery(
                            args, phase, runtime, status_path):
                        restarts = 0
                        continue
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

    deferred = unresolved_deferred_intervals(runtime)
    if deferred:
        publish(status_path, {
            "state": "deferred_intervals_pending",
            "deferred_intervals": deferred,
            "updated_unix": time.time(),
        })
        return 1
    publish(status_path, {"state": "all_complete", "updated_unix": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
