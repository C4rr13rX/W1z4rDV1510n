#!/usr/bin/env python3
"""Evolve coding-debug evidence topology against executable held-out tests."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import time
import tomllib
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_POOLS = (3, 5, 6, 10)


@dataclass
class DebugGenome:
    name: str
    evidence_pools: list[int]
    min_pool_score: float
    min_joint_score: float
    concept_threshold: int
    max_concept_members: int
    repeats: int
    generation: int = 0
    parents: list[str] | None = None


def seed() -> DebugGenome:
    return DebugGenome("debug-g0-seed", [1, 2, 3, 5, 6, 10], 0.20, 0.0, 3, 24, 4)


def mutate(parent: DebugGenome, rng: random.Random, index: int) -> DebugGenome:
    child = copy.deepcopy(parent)
    child.generation += 1
    child.parents = [parent.name]
    child.name = f"debug-g{child.generation}-{index}"
    gene = rng.choice(("evidence", "pool_floor", "joint_floor", "threshold", "window", "repeats"))
    if gene == "evidence":
        pool = rng.choice(OPTIONAL_POOLS)
        if pool in child.evidence_pools and len(child.evidence_pools) > 2:
            child.evidence_pools.remove(pool)
        elif pool not in child.evidence_pools:
            child.evidence_pools.append(pool)
            child.evidence_pools.sort()
    elif gene == "pool_floor":
        child.min_pool_score = round(min(0.85, max(0.05,
            child.min_pool_score + rng.uniform(-0.20, 0.20))), 3)
    elif gene == "joint_floor":
        child.min_joint_score = round(min(0.95, max(0.0,
            child.min_joint_score + rng.uniform(-0.25, 0.25))), 3)
    elif gene == "threshold":
        child.concept_threshold = rng.randint(2, 6)
    elif gene == "window":
        child.max_concept_members = rng.choice((8, 12, 16, 24, 32, 48))
    else:
        child.repeats = rng.randint(3, 6)
    return child


def request(endpoint: str, path: str) -> dict:
    with urllib.request.urlopen(endpoint + path, timeout=5) as response:
        return json.loads(response.read())


def wait_health(endpoint: str, process: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"node exited with {process.returncode}")
        try:
            urllib.request.urlopen(endpoint + "/brain/health", timeout=1).close()
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("node did not become healthy")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)


def identity_for(genome: DebugGenome) -> dict:
    with (ROOT / "brains/coding_debug.identity.toml").open("rb") as handle:
        identity = tomllib.load(handle)
    identity["name"] = genome.name
    identity["version"] = f"ga-{genome.generation}"
    for pool in identity["pools"]:
        if pool["id"] not in (4, 11):
            pool["concept_emergence_threshold"] = genome.concept_threshold
            pool["max_concept_member_count"] = genome.max_concept_members
    return identity


def evaluate(genome: DebugGenome, args, index: int) -> dict:
    run_dir = args.out / f"generation-{genome.generation}" / genome.name
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "identity.json"
    identity_path.write_text(json.dumps(identity_for(genome), indent=2), encoding="utf-8")
    (run_dir / "genome.json").write_text(json.dumps(asdict(genome), indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(run_dir / "node"),
        "W1Z4RD_NODE_BRAIN_DIR": str(run_dir / "brain"),
        "W1Z4RD_BRAIN_IDENTITY": str(identity_path),
        "W1Z4RD_BRAIN_DEPLOYMENT": str(ROOT / "brains/coding_debug.deployment.toml"),
        "BRAIN_MULTI_MIN_POOL_SCORE": str(genome.min_pool_score),
        "BRAIN_MULTI_MIN_JOINT_SCORE": str(genome.min_joint_score),
    })
    port = args.port + index
    endpoint = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            [str(args.node_bin), "--config", str(args.config), "api", "--addr", f"127.0.0.1:{port}"],
            cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            wait_health(endpoint, process)
            subprocess.run([
                os.fspath(args.python), "scripts/programming_debug_episode_train.py",
                "--endpoint", endpoint, "--repeats", str(genome.repeats), "--pretrain",
                "--output", str(run_dir / "episodes.jsonl")], cwd=ROOT, env=env,
                check=True, stdout=subprocess.DEVNULL)
            subprocess.run([
                os.fspath(args.python), "scripts/programming_debug_benchmark.py",
                "--endpoint", endpoint, "--evidence-pools",
                ",".join(map(str, genome.evidence_pools)),
                "--output", str(run_dir / "benchmark.json")], cwd=ROOT, env=env,
                check=True, stdout=subprocess.DEVNULL)
            report = json.loads((run_dir / "benchmark.json").read_text())
            stats = request(endpoint, "/brain/stats")
        finally:
            stop(process)
    ratios = {key: value["passed"] / value["total"] for key, value in report.items()}
    growth_cost = min(1.0, stats.get("total_neurons", 0) / 100_000)
    terminal_cost = min(1.0, stats.get("total_terminals", 0) / 5_000_000)
    score = (4 * ratios["exact"] + 4 * ratios["heldout_execution"]
             + 8 * ratios["structural_transfer"] + 5 * ratios["oov_honesty"]
             - growth_cost - terminal_cost)
    result = {"genome": asdict(genome), "fitness": score, "metrics": ratios,
              "stats": stats, "elapsed_seconds": time.perf_counter() - started}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=18400)
    parser.add_argument("--node-bin", type=Path,
                        default=ROOT / "target/debug/w1z4rdv1510n-node.exe")
    parser.add_argument("--config", type=Path, default=ROOT / "node_config.json")
    parser.add_argument("--python", type=Path, default=Path(os.sys.executable))
    parser.add_argument("--out", type=Path, default=ROOT / "runtime/debug_topology_ga")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    population = [seed()]
    while len(population) < args.population:
        population.append(mutate(population[0], rng, len(population)))
    history = []
    for generation in range(args.generations):
        scored = []
        for index, genome in enumerate(population):
            try:
                result = evaluate(genome, args, index)
            except Exception as error:
                result = {"genome": asdict(genome), "fitness": -1e9, "error": str(error)}
            history.append(result)
            scored.append((result["fitness"], genome, result))
            print(json.dumps({"name": genome.name, "fitness": result["fitness"],
                              "metrics": result.get("metrics"), "error": result.get("error")}),
                  flush=True)
        scored.sort(key=lambda value: value[0], reverse=True)
        elites = [value[1] for value in scored[:args.elite]]
        population = [copy.deepcopy(value) for value in elites]
        for value in population:
            value.generation = generation + 1
        while len(population) < args.population:
            population.append(mutate(rng.choice(elites), rng, len(population)))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.out / "best.json").write_text(json.dumps(scored[0][2], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
