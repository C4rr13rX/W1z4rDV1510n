#!/usr/bin/env python3
"""Replay recorded slow corpus ranges on parallel disposable micro-brains."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.training_standard.drive_corpora_brain import _pretrain_episode  # noqa: E402


def request(endpoint: str, path: str, payload: dict | None, timeout: float) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint + path, data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(endpoint: str, process: subprocess.Popen, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"micro-brain exited with {process.returncode}")
        try:
            request(endpoint, "/brain/stats", None, 2)
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"micro-brain did not become ready: {endpoint}")


def read_events(ledger: Path, limit: int) -> list[dict]:
    events = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique: dict[tuple[int, int], dict] = {}
    for event in events:
        key = (int(event["logical_start_row"]), int(event["logical_end_row"]))
        unique.pop(key, None)
        unique[key] = event
    return list(unique.values())[-limit:]


def read_corpus_ranges(corpus: Path, events: list[dict]) -> dict[tuple[int, int], list[dict]]:
    wanted = {
        (int(event["logical_start_row"]), int(event["logical_end_row"]))
        for event in events
    }
    rows = {key: [] for key in wanted}
    maximum = max(end for _, end in wanted)
    with corpus.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= maximum:
                break
            for (start, end), selected in rows.items():
                if start <= index < end:
                    item = json.loads(line)
                    prompt = str(item.get("prompt") or item.get("question") or "").strip()
                    answer = str(item.get("response") or item.get("answer") or "").strip()
                    if prompt and answer:
                        selected.append(_pretrain_episode(prompt, answer, None, "auto"))
    return rows


def run_range(event: dict, episodes: list[dict], index: int, args: argparse.Namespace,
              runtime_root: Path) -> dict:
    start = int(event["logical_start_row"])
    end = int(event["logical_end_row"])
    data_dir = runtime_root / f"rows-{start}-{end}"
    node_dir = data_dir / "node"
    brain_dir = data_dir / "brain"
    node_dir.mkdir(parents=True)
    brain_dir.mkdir()
    stdout = (data_dir / "node.stdout.log").open("w", encoding="utf-8")
    stderr = (data_dir / "node.stderr.log").open("w", encoding="utf-8")
    port = args.base_port + index
    endpoint = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(node_dir),
        "W1Z4RD_NODE_BRAIN_DIR": str(brain_dir),
        "W1Z4RD_BRAIN_IDENTITY": str(args.identity.resolve()),
        "W1Z4RD_BRAIN_DEPLOYMENT": str(args.deployment.resolve()),
        "W1Z4RD_BRAIN_PORT": str(port),
        "W1Z4RD_BIND_ADDR": "127.0.0.1",
        "W1Z4RD_TICK_HOUSEKEEPING": "lazy",
        "W1Z4RD_DEFER_PROMOTION": "1",
    })
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(args.executable.resolve())], cwd=ROOT, env=env,
        stdout=stdout, stderr=stderr, creationflags=flags,
    )
    try:
        wait_ready(endpoint, process)
        started = time.perf_counter()
        result = request(endpoint, "/brain/pretrain_bindings", {
            "episodes": episodes,
            "lock_chunk_size": 1,
        }, 300)
        wall_seconds = time.perf_counter() - started
        return {
            "logical_start_row": start,
            "logical_end_row": end,
            "production_max_lock_seconds": event.get("max_lock_seconds"),
            "episodes": len(episodes),
            "microbrain_wall_seconds": round(wall_seconds, 4),
            "microbrain_max_lock_seconds": round(
                float(result.get("max_lock_millis") or 0) / 1000, 4
            ),
            "max_lock_chunk_index": result.get("max_lock_chunk_index"),
            "max_lock_profile_ns": result.get("max_lock_profile_ns"),
            "max_lock_logical_row": (
                start + int(result["max_lock_chunk_index"])
                if "max_lock_chunk_index" in result else None
            ),
            "accepted": result.get("accepted"),
            "ok": result.get("ok") is True,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        stdout.close()
        stderr.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--identity", type=Path,
                        default=Path("brains/coding_debug.identity.toml"))
    parser.add_argument("--deployment", type=Path,
                        default=Path("brains/coding_debug.deployment.toml"))
    parser.add_argument("--runtime-root", type=Path,
                        default=Path("runtime/microbrains/slow-batches"))
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/slow-batch-microbrains.json"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=18760)
    args = parser.parse_args()
    events = read_events(args.ledger, args.limit)
    if not events:
        raise RuntimeError("slow-batch ledger contains no events")
    ranges = read_corpus_ranges(args.corpus, events)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runtime_root = (args.runtime_root / stamp).resolve()
    runtime_root.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_range, event,
                ranges[(int(event["logical_start_row"]),
                        int(event["logical_end_row"]))],
                index, args, runtime_root,
            )
            for index, event in enumerate(events)
        ]
        results = [future.result() for future in futures]
    report = {
        "runtime_root": str(runtime_root),
        "results": sorted(results, key=lambda row: row["logical_start_row"]),
        "updated_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(row["ok"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
