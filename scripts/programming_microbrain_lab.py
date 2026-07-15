#!/usr/bin/env python3
"""Run programming-composition experiments on parallel disposable micro-brains.

The production brain is intentionally excluded.  Each variant gets the same
coding identity and native decoder, but only the motif/noise episodes needed
to expose an architectural hypothesis.  Winners must still pass the guarded
production-brain admission before they can alter the persistent brain.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

from programming_experiential_generalization import request
from programming_domain_transfer_holdout import query as query_domain_transfer, transfer_prompt
from programming_multidomain_holdout import DOMAIN_REQUIREMENTS, holdout_prompt, query
from programming_multidomain_synthesis import train as train_noise
from programming_parameterized_fulfillment import train as train_target, training_rows
from programming_state_contract_holdout import query as query_third_state_contract


VARIANTS = (
    ("target_r1", (("target", 1),)),
    ("target_r3", (("target", 3),)),
    ("target_r6", (("target", 6),)),
    ("noise_then_target", (("noise", 3), ("target", 6))),
    ("target_then_noise", (("target", 6), ("noise", 3))),
    ("heavy_noise_then_target", (("noise", 6), ("target", 6))),
    ("heavy_noise_then_target_r1", (("noise", 6), ("target", 1))),
    ("target_r1_then_heavy_noise", (("target", 1), ("noise", 6))),
)


def wait_ready(endpoint: str, process: subprocess.Popen, timeout: float = 90) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"micro-brain exited with {process.returncode}")
        try:
            return request(endpoint, "/brain/stats", None, timeout=2)
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"micro-brain did not become ready: {endpoint}")


def run_variant(name: str, schedule: tuple[tuple[str, int], ...], port: int,
                root: Path, executable: Path, identity: Path) -> dict:
    data_dir = root / name
    data_dir.mkdir(parents=True, exist_ok=False)
    stdout = (data_dir / "node.stdout.log").open("w", encoding="utf-8")
    stderr = (data_dir / "node.stderr.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(data_dir),
        "W1Z4RD_BRAIN_IDENTITY": str(identity),
        "W1Z4RD_BRAIN_PORT": str(port),
    })
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(executable)], cwd=Path(__file__).resolve().parents[1], env=env,
        stdout=stdout, stderr=stderr, creationflags=flags,
    )
    endpoint = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    try:
        initial = wait_ready(endpoint, process)
        stages = []
        for kind, repeats in schedule:
            result = (train_target if kind == "target" else train_noise)(endpoint, repeats)
            stages.append({"kind": kind, "repeats": repeats, "accepted": result["accepted"]})
        response = request(endpoint, "/brain/chat", {"text": holdout_prompt()}, timeout=60)
        diagnostics = response.get("intent_diagnostics") or {}
        fulfillment_cases = {
            f"{class_name}.{method_name}": query(
                endpoint, holdout_prompt(class_name=class_name, method_name=method_name),
                class_name, method_name,
            )
            for class_name, method_name in (
                ("ResilientFulfillmentService", "fulfill"),
                ("DurableWarehouseEngine", "allocate_order"),
            )
        }
        ablations = {
            name: query(endpoint, holdout_prompt(name))
            for name in DOMAIN_REQUIREMENTS
        }
        domain_transfer = query_domain_transfer(endpoint, transfer_prompt())
        third_state_contract = query_third_state_contract(endpoint)
        probes = []
        for prompt, _ in training_rows():
            probe = request(endpoint, "/brain/chat", {"text": prompt + " Please reuse it."}, timeout=20)
            probe_diagnostics = probe.get("intent_diagnostics") or {}
            probes.append({
                "prompt": prompt,
                "labels": probe_diagnostics.get("labels") or [],
                "fragment_candidates": probe_diagnostics.get("fragment_candidates") or [],
            })
        reply = str(response.get("reply") or "")
        try:
            files = json.loads(reply).get("files") or {}
        except (json.JSONDecodeError, AttributeError):
            files = {}
        return {
            "name": name,
            "schedule": stages,
            "initial_tick": initial.get("tick"),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "ranked_candidates": diagnostics.get("ranked_candidates"),
            "labels": diagnostics.get("labels") or [],
            "fragment_candidates": diagnostics.get("fragment_candidates") or [],
            "training_probes": probes,
            "fulfillment_cases": fulfillment_cases,
            "ablations": ablations,
            "domain_transfer": domain_transfer,
            "third_state_contract": third_state_contract,
            "files": sorted(files),
            "reply_chars": len(reply),
        }
    finally:
        process.kill()
        process.wait(timeout=15)
        stdout.close()
        stderr.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=Path("brains/coding_debug.identity.toml"))
    parser.add_argument("--runtime-root", type=Path, default=Path("runtime/microbrains"))
    parser.add_argument("--base-port", type=int, default=18710)
    parser.add_argument("--workers", type=int, default=len(VARIANTS))
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/microbrain-composition-lab.json"))
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = (args.runtime_root / stamp).resolve()
    root.mkdir(parents=True, exist_ok=False)
    executable = args.executable.resolve()
    identity = args.identity.resolve()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_variant, name, schedule, args.base_port + index,
                        root, executable, identity)
            for index, (name, schedule) in enumerate(VARIANTS)
        ]
        results = [future.result() for future in futures]
    report = {"root": str(root), "variants": results, "updated_unix": time.time()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "variants": len(results),
        "candidate_counts": {row["name"]: row["ranked_candidates"] for row in results},
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
