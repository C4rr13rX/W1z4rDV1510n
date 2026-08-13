#!/usr/bin/env python3
"""Keep the market evolution worker available without consuming scarce RAM.

The watchdog stays lightweight while memory is constrained, starts the real
evolution service only after the configured free-memory floor is met, and
restarts unexpected exits.  The worker keeps its own memory guard so a race
between this check and dataset loading remains safe.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable
import urllib.error
import urllib.request

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GHOST_STACK_ROOT = ROOT.parent / "CoolCryptoUtilities"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_memory_guard import VerifiedWorkingSetReclaimer


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def available_memory_gb() -> float:
    return psutil.virtual_memory().available / 1024**3 if psutil else 999.0


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (2 ** attempt))


def publish_status(state_dir: Path, phase: str, **payload: object) -> None:
    try:
        atomic_json(state_dir / "supervisor_status.json", {
            "at": utc_now(), "phase": phase, "supervisor_pid": os.getpid(),
            **payload,
        })
    except OSError:
        pass


def append_event(state_dir: Path, event: str, **payload: object) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "supervisor_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utc_now(), "event": event, **payload}) + "\n")
        stream.flush()


def validated_profit_factor(state_dir: Path, required_folds: int = 3) -> float | None:
    """Read only the protected-fold minimum PF from the durable champion."""
    try:
        champion = json.loads((state_dir / "champion.json").read_text(encoding="utf-8"))
        result = champion.get("result") or {}
        if str(result.get("status") or "") != "screened":
            return None
        if int(result.get("evaluated_folds") or 0) < required_folds:
            return None
        value = float((result.get("summary") or {}).get("min_profit_factor"))
        return value if value >= 0.0 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def ghost_stack_healthy(stack_root: Path, timeout: float = 3.0,
                        heartbeat_max_age: float = 150.0) -> bool:
    """Require both Django HTTP and a fresh production-manager heartbeat."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8001/api/wizard-chat/status/", timeout=timeout,
        ) as response:
            if not 200 <= response.status < 300:
                return False
        heartbeat = json.loads(
            (stack_root / "logs/production_manager_heartbeat.json").read_text(encoding="utf-8")
        )
        age = time.time() - float(heartbeat.get("timestamp") or 0.0)
        status = str(heartbeat.get("status") or "").lower()
        return age <= heartbeat_max_age and status in {"running", "starting"}
    except (OSError, ValueError, TypeError, json.JSONDecodeError,
            urllib.error.URLError):
        return False


def maybe_start_ghost_stack(
    state_dir: Path,
    stack_root: Path,
    threshold: float,
    last_attempt: float,
    *,
    now: float | None = None,
    cooldown: float = 90.0,
) -> tuple[float, bool]:
    """Start the paper stack only after durable all-fold PF admission."""
    current = time.time() if now is None else now
    profit_factor = validated_profit_factor(state_dir)
    healthy = ghost_stack_healthy(stack_root) if profit_factor is not None and profit_factor >= threshold else False
    atomic_json(state_dir / "ghost_stack_admission.json", {
        "at": utc_now(),
        "validated_profit_factor": profit_factor,
        "threshold": threshold,
        "admitted": bool(profit_factor is not None and profit_factor >= threshold),
        "healthy": healthy,
        "mode": "ghost",
        "live_execution_enabled": False,
    })
    if profit_factor is None or profit_factor < threshold or healthy:
        return last_attempt, healthy
    if current - last_attempt < cooldown:
        return last_attempt, False
    python = stack_root / ".venv/Scripts/python.exe" if os.name == "nt" else stack_root / ".venv/bin/python"
    launcher = stack_root / "scripts/start_ghost_stack.py"
    if not python.exists() or not launcher.exists():
        append_event(state_dir, "ghost_stack_launch_unavailable",
                     python=str(python), launcher=str(launcher))
        return current, False
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (state_dir / "ghost_stack.stdout.log").open("ab") as stdout, \
            (state_dir / "ghost_stack.stderr.log").open("ab") as stderr:
        process = subprocess.Popen(
            [str(python), str(launcher), "--timeout", "90"],
            cwd=stack_root, stdout=stdout, stderr=stderr,
            creationflags=creationflags,
        )
    append_event(state_dir, "ghost_stack_launch_started", pid=process.pid,
                 validated_profit_factor=profit_factor, threshold=threshold)
    return current, False


def claim_supervisor(state_dir: Path) -> Path:
    path = state_dir / "supervisor.pid"
    prior = read_pid(path)
    if prior and process_alive(prior):
        raise RuntimeError(f"market evolution supervisor already owned by PID {prior}")
    path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{os.getpid()}\n")
    return path


def wait_until_launchable(
    state_dir: Path,
    stop_path: Path,
    required_gb: float,
    poll_seconds: float,
    memory_reader: Callable[[], float] = available_memory_gb,
    *,
    reclaim_after_polls: int = 4,
    reclaimer: VerifiedWorkingSetReclaimer | None = None,
) -> bool:
    """Return true once RAM is sufficient, or false after a stop request."""
    low_polls = 0
    while not stop_path.exists():
        available = memory_reader()
        if available >= required_gb:
            publish_status(
                state_dir, "launch_ready", available_memory_gb=available,
                required_memory_gb=required_gb,
            )
            return True
        low_polls += 1
        publish_status(
            state_dir, "waiting_for_memory", available_memory_gb=available,
            required_memory_gb=required_gb,
            consecutive_low_memory_polls=low_polls,
        )
        if reclaimer is not None and low_polls >= max(1, reclaim_after_polls):
            evidence = reclaimer.attempt()
            if evidence is not None:
                append_event(
                    state_dir, "memory_reclamation_attempt",
                    available_memory_gb=available,
                    required_memory_gb=required_gb, **evidence,
                )
                low_polls = 0
                if evidence.get("outcome") == "trimmed":
                    continue
        time.sleep(poll_seconds)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--service", type=Path,
        default=ROOT / "scripts/market_evolution_service.py",
    )
    parser.add_argument(
        "--state-dir", type=Path, default=ROOT / "runtime/market-evolution",
    )
    parser.add_argument("--min-free-memory-gb", type=float, default=3.5)
    parser.add_argument("--memory-poll-seconds", type=float, default=15.0)
    parser.add_argument("--memory-reclaim-after-polls", type=int, default=4)
    parser.add_argument("--memory-reclaim-process-name", default="w1z4rd_node.exe")
    parser.add_argument(
        "--memory-reclaim-command-fragment", default="api --addr 127.0.0.1:8090",
    )
    parser.add_argument(
        "--memory-reclaim-health-url", default="http://127.0.0.1:8090/health",
    )
    parser.add_argument("--memory-reclaim-cooldown-seconds", type=float, default=900.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--ghost-stack-root", type=Path,
                        default=DEFAULT_GHOST_STACK_ROOT)
    parser.add_argument("--ghost-stack-profit-factor", type=float, default=1.1)
    parser.add_argument("--ghost-stack-check-seconds", type=float, default=60.0)
    parser.add_argument("service_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.service_args[:1] == ["--"]:
        args.service_args = args.service_args[1:]
    if min(args.memory_poll_seconds, args.restart_delay_seconds,
           args.heartbeat_seconds, args.ghost_stack_check_seconds) <= 0:
        parser.error("poll, restart, and heartbeat intervals must be positive")
    if args.memory_reclaim_after_polls < 1:
        parser.error("memory reclaim polling threshold must be positive")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    stop_path = args.state_dir / "STOP"
    owner = claim_supervisor(args.state_dir)
    append_event(args.state_dir, "supervisor_started", pid=os.getpid())
    ghost_stack_last_check = 0.0
    ghost_stack_last_attempt = 0.0
    memory_reclaimer = VerifiedWorkingSetReclaimer(
        process_name=args.memory_reclaim_process_name,
        command_fragment=args.memory_reclaim_command_fragment,
        health_url=args.memory_reclaim_health_url,
        cooldown_seconds=max(0.0, args.memory_reclaim_cooldown_seconds),
    )
    try:
        while not stop_path.exists():
            worker_pid = read_pid(args.state_dir / "service.pid")
            if worker_pid and process_alive(worker_pid):
                publish_status(args.state_dir, "adopting_existing_worker",
                               worker_pid=worker_pid)
                while process_alive(worker_pid) and not stop_path.exists():
                    time.sleep(args.heartbeat_seconds)
                    publish_status(args.state_dir, "worker_running",
                                   worker_pid=worker_pid, adopted=True)
                    now = time.time()
                    if now - ghost_stack_last_check >= args.ghost_stack_check_seconds:
                        ghost_stack_last_attempt, _ = maybe_start_ghost_stack(
                            args.state_dir, args.ghost_stack_root,
                            args.ghost_stack_profit_factor, ghost_stack_last_attempt,
                            now=now,
                        )
                        ghost_stack_last_check = now
                continue

            if not wait_until_launchable(
                args.state_dir, stop_path, args.min_free_memory_gb,
                args.memory_poll_seconds,
                reclaim_after_polls=args.memory_reclaim_after_polls,
                reclaimer=memory_reclaimer,
            ):
                break
            command = [
                str(args.python), "-u", str(args.service),
                "--state-dir", str(args.state_dir),
                "--min-free-memory-gb", str(args.min_free_memory_gb),
                "--memory-poll-seconds", str(args.memory_poll_seconds),
                "--memory-reclaim-after-polls", str(args.memory_reclaim_after_polls),
                "--memory-reclaim-process-name", args.memory_reclaim_process_name,
                "--memory-reclaim-command-fragment", args.memory_reclaim_command_fragment,
                "--memory-reclaim-health-url", args.memory_reclaim_health_url,
                "--memory-reclaim-cooldown-seconds",
                str(args.memory_reclaim_cooldown_seconds),
                *args.service_args,
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with (args.state_dir / "service.stdout.log").open("ab") as stdout, \
                    (args.state_dir / "service.stderr.log").open("ab") as stderr:
                worker = subprocess.Popen(
                    command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    creationflags=creationflags,
                )
                append_event(args.state_dir, "worker_started", pid=worker.pid,
                             command=command)
                while worker.poll() is None:
                    publish_status(args.state_dir, "worker_running",
                                   worker_pid=worker.pid, adopted=False)
                    time.sleep(args.heartbeat_seconds)
                    now = time.time()
                    if now - ghost_stack_last_check >= args.ghost_stack_check_seconds:
                        ghost_stack_last_attempt, _ = maybe_start_ghost_stack(
                            args.state_dir, args.ghost_stack_root,
                            args.ghost_stack_profit_factor, ghost_stack_last_attempt,
                            now=now,
                        )
                        ghost_stack_last_check = now
            append_event(args.state_dir, "worker_exited", pid=worker.pid,
                         returncode=worker.returncode,
                         restart=not stop_path.exists())
            if not stop_path.exists():
                publish_status(args.state_dir, "restart_delay",
                               last_worker_pid=worker.pid,
                               last_returncode=worker.returncode)
                time.sleep(args.restart_delay_seconds)
    finally:
        try:
            if read_pid(owner) == os.getpid():
                owner.unlink(missing_ok=True)
        except OSError:
            pass
        publish_status(args.state_dir, "stopped", requested=stop_path.exists())
        append_event(args.state_dir, "supervisor_stopped",
                     requested=stop_path.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
