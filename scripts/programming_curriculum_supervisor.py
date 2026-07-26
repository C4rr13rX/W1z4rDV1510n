#!/usr/bin/env python3
"""Durably supervise sequential direct-pretrain corpus phases.

The supervisor can attach to an already-running first phase.  A worker exit is
not treated as completion until its progress ledger reaches the configured
logical-row target.  Interrupted phases restart from their RAM offset while
checkpoint accounting resumes from the separately recorded durable offset.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
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

RECOVERABLE_GATE_ERRORS = (
    RuntimeError,
    subprocess.TimeoutExpired,
    json.JSONDecodeError,
    TimeoutError,
    urllib.error.URLError,
    ConnectionError,
)
MIN_COPY_RESERVE_BYTES = 64 * 1024 * 1024
MAX_COPY_RESERVE_BYTES = 4 * 1024 * 1024 * 1024


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


def matching_live_supervisor_pid(runtime: Path) -> int:
    """Return another Python supervisor already owning this runtime."""
    runtime_token = str(runtime.resolve()).casefold()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid == os.getpid():
                continue
            name = str(process.info.get("name") or "").casefold()
            command = " ".join(process.info.get("cmdline") or []).casefold()
            if (
                name.startswith("python")
                and "programming_curriculum_supervisor.py" in command
                and runtime_token in command
            ):
                return process.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return 0


def release_supervisor_claim(path: Path, pid: int) -> None:
    try:
        if int(path.read_text(encoding="ascii").strip()) == pid:
            path.unlink()
    except (FileNotFoundError, OSError, ValueError):
        pass


def claim_curriculum_supervisor(runtime: Path) -> Path:
    """Atomically claim one supervisor per runtime, recovering stale PID files."""
    live_pid = matching_live_supervisor_pid(runtime)
    if live_pid:
        raise RuntimeError(
            f"curriculum supervisor PID {live_pid} already owns {runtime}"
        )
    path = runtime / "curriculum-supervisor.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            )
        except FileExistsError:
            try:
                recorded_pid = int(
                    path.read_text(encoding="ascii").strip()
                )
            except (FileNotFoundError, OSError, ValueError):
                recorded_pid = 0
            if process_alive(recorded_pid):
                raise RuntimeError(
                    f"curriculum supervisor claim is held by live PID "
                    f"{recorded_pid}: {path}"
                )
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"{os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        atexit.register(release_supervisor_claim, path, os.getpid())
        return path
    raise RuntimeError(f"could not atomically claim curriculum supervisor: {path}")


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


def require_snapshot_copy_headroom(source: Path, copies: int,
                                   operation: str) -> None:
    """Refuse a large copy before it consumes rollback or host headroom."""
    size = source.stat().st_size
    reserve = min(
        MAX_COPY_RESERVE_BYTES,
        max(MIN_COPY_RESERVE_BYTES, size // 10),
    )
    required = size * max(1, copies) + reserve
    free = shutil.disk_usage(source.parent).free
    if free < required:
        gib = 1024 ** 3
        raise RuntimeError(
            f"insufficient disk headroom for {operation}: "
            f"{free / gib:.2f} GiB free, {required / gib:.2f} GiB required "
            f"for {copies} snapshot copy/copies plus reserve"
        )


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


def ensure_last_good_guard(runtime: Path, phase: Phase, row: int,
                           checkpoint_proof: dict | None = None) -> Path:
    """Preserve the authoritative accepted state until the next gate passes."""
    brain_dir = runtime / "brain"
    snapshot = (
        brain_dir / "brain.wbrain"
        if (brain_dir / "brain.wbrain").is_file()
        else brain_dir / "brain.bin"
    )
    guard = brain_dir / f"brain.last-good{snapshot.suffix}"
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
    if snapshot.suffix == ".wbrain":
        # The neuron container updates slots and appends bodies in place, so a
        # hard link would mutate the alleged rollback copy too. Publish a full
        # independent copy atomically. This is paid only at comprehensive gate
        # boundaries, never for fast canaries.
        # The next failure must still have room to make its independent
        # rollback replacement after this guard exists.
        require_snapshot_copy_headroom(
            snapshot, copies=2, operation="independent .wbrain guard"
        )
        temporary = guard.with_suffix(guard.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        shutil.copy2(snapshot, temporary)
        os.replace(temporary, guard)
        guard_mode = "copy"
    else:
        os.link(snapshot, guard)
        guard_mode = "hardlink"
    publish(metadata, {
        "phase": phase.name,
        "row": row,
        "snapshot": str(snapshot),
        "guard": str(guard),
        "storage": snapshot.suffix.lstrip("."),
        "guard_mode": guard_mode,
        "checkpoint_proof": checkpoint_proof or {},
        "created_unix": time.time(),
    })
    return guard


def ensure_live_last_good_guard(args: argparse.Namespace, runtime: Path,
                                phase: Phase, row: int) -> Path:
    """Checkpoint the accepted live state before assigning it a corpus row.

    WAL durability protects an in-flight candidate, but copying only the base
    container does not.  A rollback guard therefore owns an explicit checkpoint
    barrier and the topology observed immediately after that barrier.
    """
    existing = runtime / "brain" / "brain.last-good.json"
    if existing.is_file():
        return ensure_last_good_guard(runtime, phase, row)
    checkpoint = endpoint_post_json(
        args.endpoint, "/brain/checkpoint", {}, timeout=4 * 3600.0
    )
    if checkpoint.get("ok") is False:
        raise RuntimeError(f"last-good checkpoint barrier failed: {checkpoint}")
    topology = endpoint_json(args.endpoint, "/brain/stats", timeout=120.0)
    return ensure_last_good_guard(runtime, phase, row, {
        "checkpoint": checkpoint,
        "topology": topology,
        "row": row,
        "completed_unix": time.time(),
    })


def accept_last_good_guard(runtime: Path, expected_phase: str | None = None) -> bool:
    """Discard a prior snapshot only when the accepting phase owns it.

    ``expected_phase`` is required by supervisor workflows.  The optional form is
    retained for explicit administrative cleanup and focused unit tests.
    """
    brain_dir = runtime / "brain"
    if expected_phase is not None:
        metadata = read_json(brain_dir / "brain.last-good.json")
        if metadata.get("phase") != expected_phase:
            return False
    (brain_dir / "brain.last-good.bin").unlink(missing_ok=True)
    (brain_dir / "brain.last-good.wbrain").unlink(missing_ok=True)
    (brain_dir / "brain.last-good.wbrain.tmp").unlink(missing_ok=True)
    (brain_dir / "brain.last-good.json").unlink(missing_ok=True)
    return True


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


def restore_canary_quarantine(runtime: Path, finalize: bool = True) -> dict:
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
    snapshot = Path(last_good.get("snapshot") or brain_dir / "brain.bin")
    guard = Path(last_good.get("guard") or brain_dir / f"brain.last-good{snapshot.suffix}")
    if not guard.is_file():
        raise RuntimeError(f"quarantine guard is missing: {guard}")
    same_snapshot = snapshot.exists() and os.path.samefile(snapshot, guard)
    if same_snapshot:
        pass
    else:
        temporary = snapshot.with_suffix(snapshot.suffix + ".restore.tmp")
        temporary.unlink(missing_ok=True)
        require_snapshot_copy_headroom(
            guard, copies=1, operation=".wbrain quarantine restore"
        )
        shutil.copy2(guard, temporary)
        os.replace(temporary, snapshot)
    (brain_dir / "brain.wal").unlink(missing_ok=True)
    restored = {
        "phase": phase,
        "row": row,
        "snapshot": str(snapshot),
        "checkpoint_proof": last_good.get("checkpoint_proof") or {},
    }
    if finalize:
        finalize_canary_restore(runtime, restored)
    return restored


def finalize_canary_restore(runtime: Path, restored: dict) -> None:
    """Publish the rewind only after a replacement node proves the snapshot."""
    phase = str(restored["phase"])
    row = int(restored["row"])
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
    brain_dir = runtime / "brain"
    (brain_dir / "brain.last-good.bin").unlink(missing_ok=True)
    (brain_dir / "brain.last-good.wbrain").unlink(missing_ok=True)
    (brain_dir / "brain.last-good.json").unlink(missing_ok=True)
    canary_quarantine_path(runtime).unlink(missing_ok=True)


def verify_restored_topology(restored: dict, observed: dict) -> None:
    """Reject a rollback whose reopened topology is not its recorded barrier."""
    expected = ((restored.get("checkpoint_proof") or {}).get("topology") or {})
    if not expected:
        raise RuntimeError(
            "last-good guard has no checkpoint topology proof; refusing to "
            "claim its corpus row after restart"
        )
    fields = (
        "tick", "pool_count", "total_neurons", "total_concepts",
        "total_binding", "total_terminals",
    )
    mismatches = {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in fields if expected.get(field) != observed.get(field)
    }
    if mismatches:
        raise RuntimeError(
            f"restored topology does not match last-good checkpoint: {mismatches}"
        )


def endpoint_listener_pid(endpoint: str) -> int:
    """Return the PID that owns the endpoint's listening socket, if any."""
    parsed = urlparse(endpoint)
    if not parsed.port:
        return 0
    host = (parsed.hostname or "").lower()
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status != psutil.CONN_LISTEN or not connection.laddr:
            continue
        if connection.laddr.port != parsed.port:
            continue
        address = str(connection.laddr.ip).lower()
        if host in ("localhost", "127.0.0.1", "::1") and address not in (
                "0.0.0.0", "::", "127.0.0.1", "::1"):
            continue
        return int(connection.pid or 0)
    return 0


def stop_runtime_node(runtime: Path, endpoint: str,
                      timeout: float = 60.0) -> int:
    pid_path = runtime / "node.pid"
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        pid = 0
    listener_pid = endpoint_listener_pid(endpoint)
    if listener_pid and listener_pid != pid:
        pid = listener_pid
    if not process_alive(pid):
        if listener_pid:
            raise RuntimeError(
                f"endpoint {endpoint} remains owned by live PID {listener_pid}"
            )
        return pid
    process = psutil.Process(pid)
    executable = Path(process.exe()).name.lower()
    if "w1z4rd_brain_server" not in executable:
        raise RuntimeError(
            f"refusing to stop unrelated PID {pid}: {executable}"
        )
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
    deadline = time.monotonic() + timeout
    while endpoint_listener_pid(endpoint):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"brain endpoint remained live after stopping PID {pid}")
        time.sleep(0.1)
    return pid


def start_runtime_node(runtime: Path, executable: Path, endpoint: str,
                       timeout: float = 900.0) -> subprocess.Popen:
    executable = executable.resolve()
    if not executable.is_file():
        raise RuntimeError(f"brain server executable is missing: {executable}")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise RuntimeError(f"automatic node recovery requires an HTTP host and port: {endpoint}")
    occupied_pid = endpoint_listener_pid(endpoint)
    if occupied_pid:
        raise RuntimeError(
            f"refusing false recovery: endpoint {endpoint} is already owned by "
            f"PID {occupied_pid}"
        )
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
            owner_pid = endpoint_listener_pid(endpoint)
            endpoint_json(endpoint, "/brain/stats", timeout=2.0)
            if owner_pid == process.pid:
                return process
            if owner_pid:
                process.kill()
                raise RuntimeError(
                    f"recovery endpoint belongs to PID {owner_pid}, not launched "
                    f"PID {process.pid}"
                )
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


def latest_passing_canary_row(runtime: Path, phase: str, floor: int,
                              before_row: int | None = None,
                              after_unix: float = 0.0) -> int:
    """Return the latest green boundary in the current guarded epoch.

    A logical row can be revisited after rollback with a different set of
    deferred intervals.  A green canary from the earlier state is therefore
    not evidence about the rebuilt state at the same row.  Callers may scope
    the lookup to events published after the current last-good guard.
    """
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
                        and isinstance(row, int)
                        and (before_row is None or row < before_row)
                        and float(event.get("updated_unix") or 0.0) >= after_unix):
                    latest = max(latest, row)
    except (FileNotFoundError, OSError):
        pass
    return latest


def deferred_intervals_path(runtime: Path) -> Path:
    return runtime / "curriculum-deferred-intervals.jsonl"


def deferred_interval_id(phase: str, start_row: int, end_row: int) -> str:
    return f"{phase}:{start_row}:{end_row}"


def valid_deferred_interval(event: dict, phase: str | None = None) -> bool:
    """Accept only well-formed half-open corpus ranges."""
    start = event.get("start_row")
    end = event.get("end_row")
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end
        and (phase is None or event.get("phase") == phase)
    )


def preserve_deferred_base(runtime: Path, interval_id: str) -> Path:
    """Keep the exact causal starting snapshot for later interval bisection."""
    metadata = read_json(runtime / "brain" / "brain.last-good.json")
    guard = Path(metadata.get("guard") or runtime / "brain" / "brain.last-good.bin")
    if not guard.is_file():
        raise RuntimeError(f"cannot preserve deferred base without guard: {guard}")
    digest = hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]
    directory = runtime / "deferred" / digest
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"brain.base{guard.suffix}"
    if not base.exists():
        # The last-good guard is already an immutable, independent inode. A
        # hard link to that guard remains isolated from the mutable live
        # `.wbrain`; rollback copies the guard into a new live inode before
        # the guard name is removed. This avoids another tens-of-gigabytes
        # copy for every quarantined interval while retaining its exact base.
        try:
            os.link(guard, base)
        except OSError:
            # Cross-volume or link-restricted filesystems retain the slower
            # but portable independent-copy fallback.
            require_snapshot_copy_headroom(
                guard, copies=1, operation="deferred causal-base fallback"
            )
            shutil.copy2(guard, base)
    return base


def append_deferred_event(runtime: Path, event: dict) -> None:
    payload = dict(event)
    payload.setdefault("updated_unix", time.time())
    path = deferred_intervals_path(runtime)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prune_resolved_deferred_bases(runtime: Path) -> list[Path]:
    """Remove causal bases only after every interval using them is resolved.

    Deferred `.wbrain` bases can be tens of gigabytes. The append-only ledger
    remains authoritative: a known interval directory absent from its
    unresolved fold has no replay obligation. Unknown files and directories
    are deliberately preserved.
    """
    root = runtime / "deferred"
    try:
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return []

    known: set[str] = set()
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
                known.add(hashlib.sha256(
                    interval_id.encode("utf-8")
                ).hexdigest()[:16])
    except (FileNotFoundError, OSError):
        return []

    active: set[str] = set()
    for event in unresolved_deferred_intervals(runtime):
        active.add(hashlib.sha256(
            str(event["interval_id"]).encode("utf-8")
        ).hexdigest()[:16])
        # Ledger repair can replace an outer interval with a non-overlapping
        # sub-interval while deliberately retaining the outer interval's exact
        # causal base. Protect that recorded directory as well as the new ID's
        # derived directory.
        base_snapshot = event.get("base_snapshot")
        if isinstance(base_snapshot, str) and base_snapshot:
            base_parent = Path(base_snapshot).resolve(strict=False).parent
            if base_parent.parent == resolved_root:
                active.add(base_parent.name)
    removed: list[Path] = []
    for digest in sorted(known - active):
        directory = root / digest
        try:
            resolved = directory.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved.parent != resolved_root or resolved.name != digest:
            raise RuntimeError(
                f"refusing deferred-base cleanup outside {resolved_root}: "
                f"{resolved}"
            )
        if not any(
            child.is_file() and child.name.startswith("brain.base.")
            for child in resolved.iterdir()
        ):
            continue
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed


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
                elif (event.get("status") == "deferred"
                      and valid_deferred_interval(event)):
                    current[interval_id] = event
    except (FileNotFoundError, OSError):
        pass
    rows = list(current.values())
    if phase is not None:
        rows = [row for row in rows if row.get("phase") == phase]
    return sorted(rows, key=lambda row: (
        str(row.get("phase") or ""), int(row.get("start_row") or 0)
    ))


def next_suspect_start(runtime: Path, phase: str, candidate_row: int,
                       floor: int, canary_after_unix: float = 0.0) -> int:
    """Return the first newly trained row after subtracting quarantines.

    A canary immediately following one or more skipped ranges must not create
    a new interval that overlaps those same ranges merely because no green
    canary exists inside quarantined data.
    """
    ranges = suspect_intervals(
        runtime, phase, candidate_row, floor, canary_after_unix
    )
    return ranges[0][0] if ranges else candidate_row


def suspect_intervals(runtime: Path, phase: str, candidate_row: int,
                      floor: int,
                      canary_after_unix: float = 0.0) -> list[tuple[int, int]]:
    """Subtract existing deferred coverage from this failed canary window."""
    start = latest_passing_canary_row(
        runtime, phase, floor,
        before_row=candidate_row,
        after_unix=canary_after_unix,
    )
    if start >= candidate_row:
        return []
    uncovered: list[tuple[int, int]] = []
    cursor = start
    for interval in unresolved_deferred_intervals(runtime, phase):
        interval_start = max(start, int(interval["start_row"]))
        interval_end = min(candidate_row, int(interval["end_row"]))
        if interval_end <= cursor or interval_start >= candidate_row:
            continue
        if interval_start > cursor:
            uncovered.append((cursor, interval_start))
        cursor = max(cursor, interval_end)
    if cursor < candidate_row:
        uncovered.append((cursor, candidate_row))
    return [(begin, end) for begin, end in uncovered if begin < end]


def record_deferred_failure(runtime: Path, phase: Phase, candidate_row: int,
                            durable_row: int, error: str, reason: str) -> dict:
    """Persist one failed interval before any rollback can erase its evidence."""
    last_good = read_json(runtime / "brain" / "brain.last-good.json")
    ranges = suspect_intervals(
        runtime, phase.name, candidate_row, int(last_good.get("row") or 0),
        float(last_good.get("created_unix") or 0.0),
    )
    if not ranges:
        raise RuntimeError(
            "failed canary contains no newly trained, non-deferred rows: "
            f"phase={phase.name} boundary={candidate_row}"
        )
    events = []
    for suspect_start, suspect_end in ranges:
        interval_id = deferred_interval_id(
            phase.name, suspect_start, suspect_end
        )
        base_snapshot = preserve_deferred_base(runtime, interval_id)
        event = {
            "interval_id": interval_id,
            "phase": phase.name,
            "start_row": suspect_start,
            "end_row": suspect_end,
            "base_snapshot": str(base_snapshot),
            "base_row": int(last_good.get("row") or 0),
            "reason": reason,
            "error": error,
        }
        events.append(event)
        if not any(
            row.get("interval_id") == interval_id
            for row in unresolved_deferred_intervals(runtime)
        ):
            append_deferred_event(runtime, {**event, "status": "deferred"})
    suspect_start = ranges[0][0]
    suspect_end = ranges[-1][1]
    append_health_event(runtime, {
        "kind": reason,
        "phase": phase.name,
        "trained_rows": candidate_row,
        "passed": False,
        "suspect_start_row": suspect_start,
        "suspect_end_row": suspect_end,
        "suspect_intervals": [
            {"start_row": begin, "end_row": end} for begin, end in ranges
        ],
        "last_good": last_good,
        "error": error,
    })
    publish(canary_quarantine_path(runtime), {
        "state": reason,
        "candidate_row": candidate_row,
        "suspect_start_row": suspect_start,
        "suspect_end_row": suspect_end,
        "suspect_intervals": [
            {"start_row": begin, "end_row": end} for begin, end in ranges
        ],
        "durable_next_row": durable_row,
        "last_good": last_good,
        "error": error,
        "created_unix": time.time(),
        "deferred_events": events,
    })
    return {
        **events[0],
        "suspect_intervals": [
            {"start_row": begin, "end_row": end} for begin, end in ranges
        ],
        "deferred_events": events,
    }


def endpoint_json(endpoint: str, path: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(endpoint.rstrip("/") + path)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def endpoint_post_json(endpoint: str, path: str, payload: dict,
                       timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def settle_brain_for_admission(args: argparse.Namespace, phase: Phase,
                               runtime: Path, trained_rows: int) -> dict:
    """Finish maintenance, serialize every neuron, then durably checkpoint.

    This runs only after the corpus worker has stopped at an exact guarded
    boundary. Continuous canaries deliberately skip it because their worker is
    still learning. The subsequent behavioral gate therefore evaluates the
    same fully settled state that the next rollback guard will protect.
    """
    before = endpoint_json(args.endpoint, "/brain/stats", timeout=120.0)
    sleep = endpoint_post_json(
        args.endpoint,
        "/brain/sleep",
        {"min_use_count": 2, "stale_ticks": 1000},
        timeout=4 * 3600.0,
    )
    if sleep.get("error"):
        raise RuntimeError(f"idle brain settlement failed: {sleep}")
    checkpoint = endpoint_post_json(
        args.endpoint, "/brain/checkpoint", {}, timeout=4 * 3600.0
    )
    if checkpoint.get("ok") is False:
        raise RuntimeError(f"settled checkpoint failed: {checkpoint}")
    after = endpoint_json(args.endpoint, "/brain/stats", timeout=120.0)
    if int(after.get("resident_terminals") or 0) != 0:
        raise RuntimeError(
            "settled brain retained terminals before admission: "
            f"{after.get('resident_terminals')}"
        )
    report = {
        "kind": "admission_idle_settlement",
        "phase": phase.name,
        "trained_rows": trained_rows,
        "before": before,
        "sleep": sleep,
        "checkpoint": checkpoint,
        "after": after,
        "updated_unix": time.time(),
    }
    publish(
        runtime / f"{phase.name}.row-{trained_rows}.idle-settlement.json",
        report,
    )
    return report


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
                   rows: int, samples: int,
                   include_interval_ids: frozenset[str] = frozenset()) -> list[str]:
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
        if interval.get("interval_id") in include_interval_ids:
            continue
        command.extend([
            "--skip-range",
            f"{int(interval['start_row'])}:{int(interval['end_row'])}",
        ])
    return command


def run_completion_gate(args: argparse.Namespace, phase: Phase,
                        runtime: Path,
                        include_interval_ids: frozenset[str] = frozenset()) -> dict:
    """Require corpus recall plus protected foundation/code execution."""
    settlement = settle_brain_for_admission(args, phase, runtime, phase.rows)
    recall = run_json_command(recall_command(
        args, phase, runtime, phase.rows, 64, include_interval_ids
    ))
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
        "--endpoint", args.endpoint, "--details",
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
        "idle_settlement": settlement,
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
                      runtime: Path, trained_rows: int,
                      include_interval_ids: frozenset[str] = frozenset()) -> dict:
    """Protect retained knowledge before permitting the next corpus chunk."""
    settlement = settle_brain_for_admission(args, phase, runtime, trained_rows)
    recall = run_json_command(
        recall_command(
            args, phase, runtime, trained_rows, 32, include_interval_ids
        )
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
        "idle_settlement": settlement,
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
        "--endpoint", args.endpoint, "--details",
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
    initial_lock_chunk_size = runtime_responsive_batch_size(
        runtime, args.lock_chunk_size, read_json(progress),
        args.max_live_lock_seconds
    )
    lock_chunk_size = args.lock_chunk_size
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
        "--initial-lock-chunk-size", str(initial_lock_chunk_size),
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
                    "initial_lock_chunk_size": initial_lock_chunk_size,
                    "block_target_row": block_target_row,
                    "updated_unix": time.time(),
                })
                if (code is None and next_canary is not None
                        and ram >= next_canary
                        and ram < block_target_row):
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
                    except RECOVERABLE_GATE_ERRORS as exc:
                        worker.terminate()
                        try:
                            worker.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait(timeout=30)
                        failed_ram, failed_durable = phase_offsets(progress)
                        failed_row = max(
                            candidate_row, failed_ram, failed_durable
                        )
                        record_deferred_failure(
                            runtime, phase, failed_row, failed_durable, str(exc),
                            "continuous_canary_failed",
                        )
                        publish(status_path, {
                            "state": "continuous_canary_failed",
                            "phase": phase.name,
                            "canary_started_row": candidate_row,
                            "ram_next_row": failed_ram,
                            "durable_next_row": failed_durable,
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
        stop_runtime_node(runtime, args.endpoint)
        restored = restore_canary_quarantine(runtime, finalize=False)
        start_runtime_node(runtime, args.node_bin, args.endpoint)
        verify_restored_topology(
            restored,
            endpoint_json(args.endpoint, "/brain/stats", timeout=120.0),
        )
        finalize_canary_restore(runtime, restored)
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


def quarantine_interval_ids(quarantine: dict) -> list[str]:
    """Return the exact disjoint obligations owned by one failed candidate."""
    identifiers = []
    for event in quarantine.get("deferred_events") or []:
        interval_id = event.get("interval_id")
        if (
            isinstance(interval_id, str)
            and interval_id
            and interval_id not in identifiers
        ):
            identifiers.append(interval_id)
    legacy = quarantine.get("interval_id")
    if isinstance(legacy, str) and legacy and legacy not in identifiers:
        identifiers.append(legacy)
    return identifiers


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
        "--retest-canary-quarantine", action="store_true",
        help=(
            "comprehensively retest the live failed candidate after a fix; "
            "on success admit its exact deferred spans instead of rolling back"
        ),
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
    claim_curriculum_supervisor(args.runtime.resolve())
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
    if args.retest_canary_quarantine:
        quarantine = read_json(canary_quarantine_path(runtime))
        if not quarantine:
            parser.error("no continuous-canary quarantine exists to retest")
        last_good = quarantine.get("last_good") or read_json(
            runtime / "brain" / "brain.last-good.json"
        )
        phase_name = str(
            quarantine.get("phase") or last_good.get("phase") or ""
        )
        phase = next((item for item in phases if item.name == phase_name), None)
        if phase is None:
            parser.error(f"quarantine has unknown phase: {phase_name!r}")
        candidate_row = int(quarantine.get("candidate_row", -1))
        ram, durable = phase_offsets(runtime / f"{phase.name}.progress.json")
        if candidate_row < 0 or ram != candidate_row or durable != candidate_row:
            parser.error(
                "quarantine retest requires its exact durable live candidate; "
                f"candidate={candidate_row} ram={ram} durable={durable}"
            )
        guard_metadata = read_json(runtime / "brain" / "brain.last-good.json")
        guard_path = Path(str(guard_metadata.get("guard") or ""))
        if (
            guard_metadata.get("phase") != phase.name
            or not guard_path.is_file()
            or int(guard_metadata.get("row", -1)) > candidate_row
        ):
            parser.error(
                "quarantine retest requires the candidate's valid rollback "
                f"guard: {guard_metadata}"
            )
        interval_ids = frozenset(quarantine_interval_ids(quarantine))
        unresolved_ids = {
            str(event["interval_id"])
            for event in unresolved_deferred_intervals(runtime, phase.name)
        }
        if not interval_ids or not interval_ids.issubset(unresolved_ids):
            parser.error(
                "quarantine retest intervals are missing from the unresolved "
                f"ledger: candidate={sorted(interval_ids)} "
                f"unresolved={sorted(unresolved_ids)}"
            )
        publish(status_path, {
            "state": "quarantine_retest_benchmarking",
            "phase": phase.name,
            "candidate_row": candidate_row,
            "interval_ids": sorted(interval_ids),
            "updated_unix": time.time(),
        })
        try:
            if candidate_row >= phase.rows:
                report = run_completion_gate(
                    args, phase, runtime, interval_ids
                )
            else:
                report = run_midphase_gate(
                    args, phase, runtime, candidate_row, interval_ids
                )
        except RECOVERABLE_GATE_ERRORS as exc:
            publish(status_path, {
                "state": "quarantine_retest_failed",
                "phase": phase.name,
                "candidate_row": candidate_row,
                "interval_ids": sorted(interval_ids),
                "error": str(exc),
                "updated_unix": time.time(),
            })
            return 1
        for interval_id in sorted(interval_ids):
            append_deferred_event(runtime, {
                "interval_id": interval_id,
                "status": "resolved",
                "phase": phase.name,
                "reason": "fixed candidate passed comprehensive quarantine retest",
            })
        pruned_bases = prune_resolved_deferred_bases(runtime)
        canary_quarantine_path(runtime).unlink(missing_ok=True)
        if not accept_last_good_guard(runtime, phase.name):
            raise RuntimeError(
                f"could not release {phase.name} last-good guard after admission"
            )
        append_health_event(runtime, {
            "kind": "quarantine_retest_admitted",
            "phase": phase.name,
            "trained_rows": candidate_row,
            "interval_ids": sorted(interval_ids),
            "passed": True,
            "report": report,
        })
        publish(status_path, {
            "state": "quarantine_retest_admitted",
            "phase": phase.name,
            "ram_next_row": ram,
            "durable_next_row": durable,
            "interval_ids": sorted(interval_ids),
            "pruned_deferred_bases": [str(path) for path in pruned_bases],
            "updated_unix": time.time(),
        })
        return 0
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
        except RECOVERABLE_GATE_ERRORS as exc:
            publish(status_path, {
                "state": "gate_only_failed", "phase": phase.name,
                "ram_next_row": ram, "durable_next_row": durable,
                "error": str(exc), "updated_unix": time.time(),
            })
            return 1
        accept_last_good_guard(runtime, phase.name)
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
                except RECOVERABLE_GATE_ERRORS as exc:
                    try:
                        attached_process = psutil.Process(args.attach_pid)
                        attached_process.terminate()
                        attached_process.wait(timeout=30)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
                    failed_ram, failed_durable = phase_offsets(
                        runtime / f"{attach_phase.name}.progress.json"
                    )
                    failed_row = max(
                        candidate_row, failed_ram, failed_durable
                    )
                    record_deferred_failure(
                        runtime, attach_phase, failed_row,
                        failed_durable, str(exc),
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
                stop_runtime_node(runtime, args.endpoint)
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
            except RECOVERABLE_GATE_ERRORS as exc:
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
                accept_last_good_guard(runtime, attach_phase.name)

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
                    except RECOVERABLE_GATE_ERRORS as exc:
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
                accept_last_good_guard(runtime, phase.name)
                break
            publish(status_path, {"state": "running", "phase": phase.name,
                                  "ram_next_row": ram,
                                  "durable_next_row": durable,
                                  "restart": restarts,
                                  "updated_unix": time.time()})
            ensure_live_last_good_guard(args, runtime, phase, ram)
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
                except RECOVERABLE_GATE_ERRORS as exc:
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
                accept_last_good_guard(runtime, phase.name)
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
