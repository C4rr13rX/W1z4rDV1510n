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

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


ROOT = Path(__file__).resolve().parents[1]


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
) -> bool:
    """Return true once RAM is sufficient, or false after a stop request."""
    while not stop_path.exists():
        available = memory_reader()
        if available >= required_gb:
            publish_status(
                state_dir, "launch_ready", available_memory_gb=available,
                required_memory_gb=required_gb,
            )
            return True
        publish_status(
            state_dir, "waiting_for_memory", available_memory_gb=available,
            required_memory_gb=required_gb,
        )
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
    parser.add_argument("--restart-delay-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("service_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.service_args[:1] == ["--"]:
        args.service_args = args.service_args[1:]
    if min(args.memory_poll_seconds, args.restart_delay_seconds,
           args.heartbeat_seconds) <= 0:
        parser.error("poll, restart, and heartbeat intervals must be positive")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    stop_path = args.state_dir / "STOP"
    owner = claim_supervisor(args.state_dir)
    append_event(args.state_dir, "supervisor_started", pid=os.getpid())
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
                continue

            if not wait_until_launchable(
                args.state_dir, stop_path, args.min_free_memory_gb,
                args.memory_poll_seconds,
            ):
                break
            command = [
                str(args.python), "-u", str(args.service),
                "--state-dir", str(args.state_dir),
                "--min-free-memory-gb", str(args.min_free_memory_gb),
                "--memory-poll-seconds", str(args.memory_poll_seconds),
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
