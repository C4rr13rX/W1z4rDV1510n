"""Execute structural Wizard-brain evolution against isolated live nodes."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ga_architecture_genome import ArchitectureGenome, crossover, fitness, load_genome, mutate, seed
from scripts.experiments.small_brain_generalization import Client, evaluate, summarize, DEFAULT_CORPUS


def wait_health(endpoint: str, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"node exited {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=1.0):
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"node did not become healthy: {endpoint}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def pool_id(genome: ArchitectureGenome, name: str) -> int:
    return next(i for i, pool in enumerate(genome.pools, 1) if pool.name == name)


def evaluate_genome(genome: ArchitectureGenome, *, node_bin: Path, config: Path,
                    run_dir: Path, port: int, repeats: int, corpus: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    materialized = genome.materialize(str(run_dir / "brain"))
    identity_path = run_dir / "identity.json"
    deployment_path = run_dir / "deployment.json"
    identity_path.write_text(json.dumps(materialized["identity"], indent=2))
    deployment_path.write_text(json.dumps(materialized["deployment"], indent=2))
    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(run_dir / "node"),
        "W1Z4RD_NODE_BRAIN_DIR": str(run_dir / "brain"),
        "W1Z4RD_BRAIN_IDENTITY": str(identity_path),
        "W1Z4RD_BRAIN_DEPLOYMENT": str(deployment_path),
    })
    endpoint = f"http://127.0.0.1:{port}"
    with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            [str(node_bin), "--config", str(config), "api", "--addr", f"127.0.0.1:{port}"],
            cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            wait_health(endpoint, process)
            action = next(pool.name for pool in genome.pools if pool.kind == "Action")
            client = Client(endpoint, "/brain", 30.0,
                            pool_id(genome, genome.pools[0].name), pool_id(genome, action),
                            pool_id(genome, genome.readout_pool or genome.pools[0].name),
                            genome.settle_ticks, genome.inference_mode)
            before = client.request("GET", "/stats")
            training_latency = []
            for _ in range(repeats):
                for row in corpus["train"]:
                    started = time.perf_counter()
                    client.train(row["prompt"], row["answer"])
                    training_latency.append(time.perf_counter() - started)
            trained = client.request("GET", "/stats")
            rows = evaluate(corpus["test"], client.chat, corpus["train"])
            summary = summarize(rows)
            after = client.request("GET", "/stats")
            latency_ms = statistics.mean(row["latency_ms"] for row in rows)
            neuron_growth = max(0, after.get("total_neurons", 0) - before.get("total_neurons", 0))
            metrics = {
                "exact_recall": summary["exact"]["concept_recall"],
                "paraphrase": summary["paraphrase"]["concept_recall"],
                "composition": summary["composition"]["concept_recall"],
                "oov_honesty": summary["oov"]["oov_accuracy"],
                "directional_edge_over_baseline": 0.0,
                "calibration": 0.0,
                "latency_cost": min(1.0, latency_ms / 500.0),
                "memory_cost": min(1.0, neuron_growth / 100_000.0),
            }
            result = {
                "genome": genome.name, "generation": genome.generation,
                "fitness": fitness(metrics), "metrics": metrics,
                "summary": summary, "latency_ms": latency_ms,
                "training_latency_ms": statistics.mean(training_latency) * 1000,
                "stats_before": before, "stats_after": after,
                "stats_after_training": trained,
                "prediction_state_delta": {
                    key: after.get(key, 0) - trained.get(key, 0)
                    for key in ("tick", "total_neurons", "total_concepts", "total_binding", "total_terminals")
                },
                "route": materialized["evaluation_route"],
            }
            (run_dir / "result.json").write_text(json.dumps(result, indent=2))
            return result
        finally:
            stop_process(process)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("coding",), default="coding")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--node-bin", type=Path, default=ROOT / "target/debug/w1z4rdv1510n-node.exe")
    ap.add_argument("--config", type=Path, default=ROOT / "node_config.json")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=ROOT / "runtime/architecture_ga")
    ap.add_argument("--seed-dir", type=Path, action="append", default=[])
    args = ap.parse_args()
    if args.population < 2 or args.elite < 1 or args.elite >= args.population:
        ap.error("require population >= 2 and 1 <= elite < population")
    rng = random.Random(args.seed)
    corpus = json.loads(args.corpus.read_text())
    seeded = [load_genome(path) for path in args.seed_dir]
    root = seed(args.domain, rng)
    population = seeded[:args.population] or [root]
    while len(population) < args.population:
        population.append(mutate(rng.choice(population), rng, len(population)))
    history = []
    args.out.mkdir(parents=True, exist_ok=True)
    for generation in range(args.generations):
        scored = []
        for index, genome in enumerate(population):
            run_dir = args.out / f"generation-{generation}" / genome.name
            try:
                result = evaluate_genome(genome, node_bin=args.node_bin, config=args.config,
                                         run_dir=run_dir, port=args.port + index,
                                         repeats=args.repeats, corpus=corpus)
            except Exception as exc:
                result = {"genome": genome.name, "generation": generation,
                          "fitness": -1e9, "error": str(exc)}
                (run_dir / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({k: result.get(k) for k in ("genome", "generation", "fitness", "error")}), flush=True)
            scored.append((result["fitness"], genome, result))
            history.append(result)
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = [item[1] for item in scored[:args.elite]]
        objective_keys = ("exact_recall", "paraphrase", "composition", "oov_honesty")
        champions = []
        for key in objective_keys:
            candidate = max(scored, key=lambda item: item[2].get("metrics", {}).get(key, -1))[1]
            if all(candidate.name != prior.name for prior in champions):
                champions.append(candidate)
        parent_pool = elites + [c for c in champions if all(c.name != e.name for e in elites)]
        survivor_count = min(len(parent_pool), max(args.elite, args.population // 2))
        population = [copy_genome(parent, generation + 1, i)
                      for i, parent in enumerate(parent_pool[:survivor_count])]
        while len(population) < args.population:
            if len(parent_pool) > 1 and rng.random() < 0.65:
                left, right = rng.sample(parent_pool, 2)
                child = crossover(left, right, rng, len(population))
                population.append(mutate(child, rng, len(population)) if rng.random() < 0.7 else child)
            else:
                population.append(mutate(rng.choice(parent_pool), rng, len(population)))
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        (args.out / "best.json").write_text(json.dumps(scored[0][2], indent=2))
    return 0


def copy_genome(parent: ArchitectureGenome, generation: int, index: int) -> ArchitectureGenome:
    import copy
    child = copy.deepcopy(parent)
    child.parents = [parent.name]
    child.generation = generation
    child.name = f"{parent.name.split('-g')[0]}-g{generation}-elite{index}"
    return child


if __name__ == "__main__":
    raise SystemExit(main())
