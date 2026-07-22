#!/usr/bin/env python3
"""Reproduce the proven Wizard Vision programming-brain curriculum.

This is the durable entry point for training a fresh coding brain. It records
each accepted seed stage, checkpoints after every stage, runs foundation
retention after every expansion, then delegates the multi-million-row corpus
curriculum to the guarded/resumable supervisor.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    arguments: tuple[str, ...] = ()


SEED_STAGES = (
    Stage("foundation-python-debug", "programming_integrated_retention.py"),
    Stage("multilanguage", "programming_multilanguage_eval.py"),
    Stage("python-enterprise", "programming_enterprise_eval.py"),
    Stage("python-projects", "programming_project_eval.py"),
    Stage("platform-engineering", "programming_platform_eval.py"),
    Stage("native-enterprise", "programming_native_enterprise_eval.py"),
    Stage("typescript-enterprise", "programming_typescript_enterprise.py"),
    Stage("cross-language-transfer", "programming_cross_language_transfer.py"),
    Stage("semantic-routing", "programming_semantic_router_train.py"),
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": 1, "completed_seed_stages": []}


def request(endpoint: str, path: str, payload: dict | None = None,
            timeout: float = 600.0) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + path, data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def wait_healthy(endpoint: str, process: subprocess.Popen | None,
                 timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    error = "node did not become healthy"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"brain node exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=3):
                return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = str(exc)
            time.sleep(0.5)
    raise TimeoutError(f"node did not become healthy: {error}")


def endpoint_healthy(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def stage_command(stage: Stage, endpoint: str, output: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / stage.script),
               "--endpoint", endpoint, *stage.arguments]
    if stage.name != "semantic-routing":
        command.extend(["--output", str(output)])
    return command


def foundation_gate_command(endpoint: str, output: Path) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts/programming_integrated_retention.py"),
        "--endpoint", endpoint, "--no-checkpoint", "--output", str(output),
    ]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def prepare_runtime(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    runtime = args.runtime.resolve()
    state_path = runtime / "programming-training.state.json"
    identity_copy = runtime / "brain.identity.toml"
    deployment_copy = runtime / "brain.deployment.toml"
    if runtime.exists() and any(runtime.iterdir()) and not args.resume:
        raise RuntimeError(
            f"runtime is not empty: {runtime}; choose a new directory or pass --resume"
        )
    runtime.mkdir(parents=True, exist_ok=True)
    if not identity_copy.exists():
        shutil.copy2(args.identity.resolve(), identity_copy)
    if not deployment_copy.exists():
        shutil.copy2(args.deployment.resolve(), deployment_copy)
    return runtime, state_path, identity_copy


def authoritative_brain_path(runtime: Path) -> Path:
    brain_dir = runtime / "brain"
    wbrain = brain_dir / "brain.wbrain"
    return wbrain if wbrain.is_file() else brain_dir / "brain.bin"


def seed_guard_path(runtime: Path) -> Path:
    brain_dir = runtime / "brain"
    for suffix in (".wbrain", ".bin"):
        candidate = brain_dir / f"seed.last-good{suffix}"
        if candidate.exists():
            return candidate
    return brain_dir / f"seed.last-good{authoritative_brain_path(runtime).suffix}"


def seed_guard_metadata_path(runtime: Path) -> Path:
    return runtime / "brain" / "seed.last-good.json"


def guard_seed_stage(runtime: Path, stage: str) -> None:
    snapshot = authoritative_brain_path(runtime)
    guard = seed_guard_path(runtime)
    metadata = seed_guard_metadata_path(runtime)
    if guard.exists():
        raise RuntimeError(f"unresolved seed-stage guard: {guard}")
    if not snapshot.is_file():
        raise RuntimeError(f"cannot guard missing brain snapshot: {snapshot}")
    if snapshot.suffix == ".wbrain":
        temporary = guard.with_suffix(guard.suffix + ".tmp")
        shutil.copy2(snapshot, temporary)
        os.replace(temporary, guard)
        guard_mode = "copy"
    else:
        os.link(snapshot, guard)
        guard_mode = "hardlink"
    atomic_json(metadata, {
        "stage": stage,
        "snapshot": str(snapshot.resolve()),
        "guard": str(guard.resolve()),
        "storage": snapshot.suffix.lstrip("."),
        "guard_mode": guard_mode,
        "created_unix": time.time(),
    })


def accept_seed_stage(runtime: Path) -> None:
    brain_dir = runtime / "brain"
    (brain_dir / "seed.last-good.bin").unlink(missing_ok=True)
    (brain_dir / "seed.last-good.wbrain").unlink(missing_ok=True)
    (brain_dir / "seed.last-good.wbrain.tmp").unlink(missing_ok=True)
    seed_guard_metadata_path(runtime).unlink(missing_ok=True)


def resolve_seed_guard(runtime: Path, state: dict) -> tuple[str, str] | None:
    """Commit or restore an interrupted stage before starting the owned node."""
    guard = seed_guard_path(runtime)
    if not guard.exists():
        return None
    metadata_path = seed_guard_metadata_path(runtime)
    metadata = read_state(metadata_path)
    stage = str(metadata.get("stage") or "unknown")
    if stage in set(state.get("completed_seed_stages") or []):
        accept_seed_stage(runtime)
        return (stage, "committed")
    snapshot = Path(metadata.get("snapshot") or authoritative_brain_path(runtime))
    os.replace(guard, snapshot)
    (runtime / "brain" / "brain.wal").unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    return (stage, "restored")


def start_node(args: argparse.Namespace, runtime: Path,
               identity: Path) -> subprocess.Popen:
    node_bin = args.node_bin.resolve()
    if not node_bin.is_file():
        raise FileNotFoundError(
            f"brain server not found: {node_bin}; build the release binary first"
        )
    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(runtime / "node"),
        "W1Z4RD_NODE_BRAIN_DIR": str(runtime / "brain"),
        "W1Z4RD_BRAIN_IDENTITY": str(identity),
        "W1Z4RD_BRAIN_DEPLOYMENT": str(runtime / "brain.deployment.toml"),
        "W1Z4RD_BRAIN_PORT": str(args.port),
        "W1Z4RD_BRAIN_BIND": "127.0.0.1",
        "W1Z4RD_TICK_HOUSEKEEPING": "lazy",
        "W1Z4RD_DEFER_PROMOTION": "1",
    })
    stdout = (runtime / "node.stdout.log").open("ab")
    stderr = (runtime / "node.stderr.log").open("ab")
    process = subprocess.Popen(
        [str(node_bin)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    (runtime / "node.pid").write_text(f"{process.pid}\n", encoding="ascii")
    return process


def checkpoint(endpoint: str) -> None:
    result = request(endpoint, "/brain/checkpoint", {}, timeout=900.0)
    if result.get("ok") is False:
        raise RuntimeError(f"checkpoint failed: {result}")


def training_plan(args: argparse.Namespace) -> list[list[str]]:
    endpoint = args.endpoint
    commands: list[list[str]] = []
    for stage in SEED_STAGES:
        commands.append(stage_command(stage, endpoint, args.runtime / "seed" / f"{stage.name}.json"))
        commands.append(foundation_gate_command(
            endpoint, args.runtime / "seed" / f"{stage.name}.foundation.json"
        ))
    commands.append([
        sys.executable, str(ROOT / "scripts/programming_enterprise_retention.py"),
        "--endpoint", endpoint,
        "--output", str(args.runtime / "seed" / "enterprise-final.json"),
    ])
    if not args.seed_only:
        commands.append([
            sys.executable, str(ROOT / "scripts/programming_curriculum_supervisor.py"),
            "--endpoint", endpoint, "--runtime", str(args.runtime),
            "--corpus-root", str(args.corpus_root), "--include-seed-corpora",
            "--batch-size", str(args.batch_size),
            "--lock-chunk-size", str(args.lock_chunk_size),
            "--checkpoint-rows", str(args.checkpoint_rows),
            "--gate-rows", str(args.gate_rows),
            "--canary-rows", str(args.canary_rows),
            "--max-live-lock-seconds", str(args.max_live_lock_seconds),
            "--auto-quarantine-recovery",
            "--node-bin", str(args.node_bin.resolve()),
        ])
        commands.extend(experience_commands(args))
    return commands


def experience_commands(args: argparse.Namespace) -> list[list[str]]:
    benchmark_dir = args.runtime / "benchmarks"
    return [
        [
            sys.executable,
            str(ROOT / "scripts/programming_experiential_generalization.py"),
            "--endpoint", args.endpoint, "--runtime", str(args.runtime),
            "--train", "--enterprise-gate",
            "--output", str(benchmark_dir / "experiential-generalization.json"),
        ],
        [
            sys.executable,
            str(ROOT / "scripts/programming_multidomain_synthesis.py"),
            "--endpoint", args.endpoint, "--runtime", str(args.runtime),
            "--train", "--enterprise-gate",
            "--output", str(benchmark_dir / "multidomain-synthesis.json"),
        ],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--external-node", action="store_true",
                        help="use an already-running endpoint instead of owning a node")
    parser.add_argument("--port", type=int, default=18600)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--node-bin", type=Path,
                        default=ROOT / "target/release/w1z4rd_brain_server.exe")
    parser.add_argument("--identity", type=Path,
                        default=ROOT / "brains/coding_debug.identity.toml")
    parser.add_argument("--deployment", type=Path,
                        default=ROOT / "brains/coding_debug.deployment.toml")
    parser.add_argument("--corpus-root", type=Path,
                        default=Path(r"D:\w1z4rdv1510n-data\training"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lock-chunk-size", type=int, default=12)
    parser.add_argument("--checkpoint-rows", type=int, default=131072)
    parser.add_argument("--gate-rows", type=int, default=131072)
    parser.add_argument("--canary-rows", type=int, default=16384)
    parser.add_argument("--max-live-lock-seconds", type=float, default=8.0)
    args = parser.parse_args()
    args.runtime = args.runtime.resolve()
    args.corpus_root = args.corpus_root.resolve()
    args.endpoint = args.endpoint or f"http://127.0.0.1:{args.port}"

    if args.dry_run:
        print(json.dumps({
            "runtime": str(args.runtime), "endpoint": args.endpoint,
            "identity": str(args.identity.resolve()),
            "seed_stages": [stage.name for stage in SEED_STAGES],
            "commands": training_plan(args),
        }, indent=2))
        return 0

    runtime, state_path, identity = prepare_runtime(args)
    state = read_state(state_path)
    process: subprocess.Popen | None = None
    try:
        if not args.external_node:
            if endpoint_healthy(args.endpoint):
                raise RuntimeError(
                    f"endpoint is already in use: {args.endpoint}; choose another "
                    "--port or explicitly pass --external-node"
                )
            resolution = resolve_seed_guard(runtime, state)
            if resolution:
                print(f"[programming-train] {resolution[1]} interrupted stage {resolution[0]}")
            process = start_node(args, runtime, identity)
        elif seed_guard_path(runtime).exists():
            raise RuntimeError(
                "an unaccepted seed-stage guard exists; resume once with the "
                "script-owned node so it can restore safely"
            )
        wait_healthy(args.endpoint, process)
        checkpoint(args.endpoint)

        completed = set(state.get("completed_seed_stages") or [])
        for index, stage in enumerate(SEED_STAGES, 1):
            if stage.name in completed:
                print(f"[programming-train] [{index}/{len(SEED_STAGES)}] resume {stage.name}")
                continue
            print(f"[programming-train] [{index}/{len(SEED_STAGES)}] train {stage.name}", flush=True)
            guard_seed_stage(runtime, stage.name)
            stage_dir = runtime / "seed"
            run_logged(
                stage_command(stage, args.endpoint, stage_dir / f"{stage.name}.json"),
                runtime / "logs" / f"{stage.name}.log",
            )
            run_logged(
                foundation_gate_command(
                    args.endpoint, stage_dir / f"{stage.name}.foundation.json"
                ),
                runtime / "logs" / f"{stage.name}.foundation.log",
            )
            checkpoint(args.endpoint)
            completed.add(stage.name)
            state.update({
                "version": 1,
                "identity": str(identity),
                "completed_seed_stages": [
                    item.name for item in SEED_STAGES if item.name in completed
                ],
                "updated_unix": time.time(),
            })
            atomic_json(state_path, state)
            accept_seed_stage(runtime)

        enterprise_command = [
            sys.executable, str(ROOT / "scripts/programming_enterprise_retention.py"),
            "--endpoint", args.endpoint,
            "--output", str(runtime / "seed" / "enterprise-final.json"),
        ]
        if not state.get("enterprise_seed_gate_passed"):
            run_logged(enterprise_command, runtime / "logs/enterprise-final.log")
            checkpoint(args.endpoint)
            state["enterprise_seed_gate_passed"] = True
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        if not args.seed_only and not state.get("corpus_curriculum_passed"):
            supervisor = training_plan(args)[-(len(experience_commands(args)) + 1)]
            run_logged(supervisor, runtime / "logs/curriculum-supervisor.log")
            state["corpus_curriculum_passed"] = True
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        if not args.seed_only and not state.get("experiential_admission_passed"):
            command = experience_commands(args)[0]
            run_logged(command, runtime / "logs/experiential-admission.log")
            state["experiential_admission_passed"] = True
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        if not args.seed_only and not state.get("multidomain_admission_passed"):
            command = experience_commands(args)[1]
            run_logged(command, runtime / "logs/multidomain-admission.log")
            state["multidomain_admission_passed"] = True
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)
        print(f"[programming-train] complete: {runtime}")
        return 0
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
