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


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Phase:
    name: str
    script_id: str
    corpus: Path
    rows: int


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()


def publish(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def phase_offsets(progress_path: Path) -> tuple[int, int]:
    progress = read_json(progress_path)
    ram = max(0, int(progress.get("ram_next_row", 0)))
    durable = max(0, min(ram, int(progress.get("durable_next_row", 0))))
    return ram, durable


def run_phase(args: argparse.Namespace, phase: Phase, runtime: Path) -> int:
    progress = runtime / f"{phase.name}.progress.json"
    ram, durable = phase_offsets(progress)
    if ram >= phase.rows:
        return 0
    stdout_path = runtime / f"{phase.name}.stdout.log"
    stderr_path = runtime / f"{phase.name}.stderr.log"
    command = [
        sys.executable, "-m", "tools.training_standard.drive_corpora_brain",
        "--brain", args.endpoint,
        "--script", phase.script_id,
        "--input-path", str(phase.corpus),
        "--repeats", "1",
        "--direct-pretrain",
        "--start-row", str(ram),
        "--durable-start-row", str(durable),
        "--batch-size", str(args.batch_size),
        "--checkpoint-rows", str(args.checkpoint_rows),
        "--feature-policy", "auto",
        "--midcheck-rows", "0",
        "--no-sleep-between",
        "--progress-path", str(progress),
    ]
    with stdout_path.open("a", encoding="utf-8") as stdout, \
            stderr_path.open("a", encoding="utf-8") as stderr:
        return subprocess.run(
            command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--attach-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-rows", type=int, default=4096)
    parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args()

    phases = [
        Phase("mathinstruct-domain-safe", "reasoning_math_001",
              Path(r"D:\w1z4rdv1510n-data\training\mathinstruct.jsonl"), 245_323),
        Phase("metamathqa-domain-safe", "reasoning_math_001",
              Path(r"D:\w1z4rdv1510n-data\training\metamathqa.jsonl"), 385_524),
    ]
    runtime = args.runtime.resolve()
    status_path = runtime / "curriculum-supervisor.status.json"

    if args.attach_pid:
        publish(status_path, {"state": "attached", "pid": args.attach_pid,
                              "phase": phases[0].name, "updated_unix": time.time()})
        while process_alive(args.attach_pid):
            time.sleep(max(1.0, args.poll_seconds))

    for phase in phases:
        restarts = 0
        while True:
            ram, durable = phase_offsets(runtime / f"{phase.name}.progress.json")
            if ram >= phase.rows:
                publish(status_path, {"state": "complete", "phase": phase.name,
                                      "ram_next_row": ram,
                                      "durable_next_row": durable,
                                      "updated_unix": time.time()})
                break
            publish(status_path, {"state": "running", "phase": phase.name,
                                  "ram_next_row": ram,
                                  "durable_next_row": durable,
                                  "restart": restarts,
                                  "updated_unix": time.time()})
            code = run_phase(args, phase, runtime)
            ram_after, durable_after = phase_offsets(
                runtime / f"{phase.name}.progress.json"
            )
            if code == 0 and ram_after >= phase.rows:
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
