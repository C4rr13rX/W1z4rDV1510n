#!/usr/bin/env python3
"""Run perpetual, restart-safe evolution of causal market-brain genomes.

Every genome is evaluated in an isolated process on purged chronological folds
and complete unseen assets.  Failed candidates are preserved in quarantine;
only the weakest-fold score drives selection.  A surrogate champion is never
silently installed as the production Wizard brain: candidates clearing the
working target are written to the brain-validation queue for neuron-fabric
training and protected re-evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - production dependency is already present
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_evolution_genome import (  # noqa: E402
    MarketGenome, crossover, load_genome, mutate, report_fitness, seed,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def claim(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "supervisor.pid"
    if path.exists():
        try:
            prior = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            prior = -1
        if process_alive(prior):
            raise RuntimeError(f"market evolution already owned by PID {prior}")
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{os.getpid()}\n")
    return path


def dataset_signature(manifest: Path, supplemental: Path) -> str:
    digest = hashlib.sha256()
    for path in (manifest, *sorted((supplemental / "features").glob("*.json"))):
        if not path.exists():
            continue
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def memory_available_gb() -> float:
    if psutil is None:
        return 999.0
    return psutil.virtual_memory().available / 1024**3


def write_status(root: Path, **values: Any) -> None:
    atomic_json(root / "status.json", {"updated_at": utcnow(), **values})


def evaluate_one(args: argparse.Namespace, genome: MarketGenome, generation_dir: Path,
                 data_signature: str) -> dict[str, Any]:
    candidate = generation_dir / genome.genome_id
    candidate.mkdir(parents=True, exist_ok=True)
    genome_path = candidate / "genome.json"
    report_path = candidate / "report.json"
    result_path = candidate / "result.json"
    atomic_json(genome_path, genome.as_json())
    if result_path.is_file() and report_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("dataset_signature") == data_signature:
            return prior
    command = [
        sys.executable, str(ROOT / "scripts/market_signal_audit.py"),
        "--manifest", str(args.manifest), "--supplemental-root", str(args.supplemental_root),
        "--genome", str(genome_path), "--folds", str(args.folds),
        "--test-days", str(args.test_days), "--cost-bps", str(args.cost_bps),
        "--permutation-repeats", "0", "--out", str(report_path),
    ]
    started = time.monotonic()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with (candidate / "stdout.log").open("wb") as stdout, \
                (candidate / "stderr.log").open("wb") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       timeout=args.candidate_timeout_minutes * 60,
                                       creationflags=creationflags)
        if completed.returncode != 0 or not report_path.is_file():
            raise RuntimeError(f"audit exited {completed.returncode}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        score, gate = report_fitness(report)
        result = {
            "genome_id": genome.genome_id, "generation": genome.generation,
            "dataset_signature": data_signature, "score": score, "gate": gate,
            "elapsed_seconds": time.monotonic() - started, "status": "evaluated",
            "finished_at": utcnow(),
        }
    except Exception as exc:
        result = {
            "genome_id": genome.genome_id, "generation": genome.generation,
            "dataset_signature": data_signature, "score": -1_000_000.0,
            "gate": {"admitted": False, "working_target": False},
            "elapsed_seconds": time.monotonic() - started, "status": "quarantined",
            "error": f"{type(exc).__name__}: {exc}", "finished_at": utcnow(),
        }
        atomic_json(candidate / "quarantine.json", result)
    atomic_json(result_path, result)
    return result


def initial_population(size: int, rng: random.Random) -> list[MarketGenome]:
    population = [seed(rng, index) for index in range(min(3, size))]
    while len(population) < size:
        population.append(mutate(rng.choice(population), rng))
    return unique_population(population, size, rng)


def unique_population(population: list[MarketGenome], size: int,
                      rng: random.Random) -> list[MarketGenome]:
    unique: dict[str, MarketGenome] = {genome.genome_id: genome for genome in population}
    attempts = 0
    while len(unique) < size and attempts < size * 50:
        parent = rng.choice(list(unique.values()))
        child = mutate(parent, rng)
        unique.setdefault(child.genome_id, child)
        attempts += 1
    if len(unique) < size:
        raise RuntimeError("could not produce a unique evolution population")
    return list(unique.values())[:size]


def next_population(ranked: list[tuple[MarketGenome, dict[str, Any]]], size: int,
                    elite_count: int, rng: random.Random) -> list[MarketGenome]:
    elites = [genome for genome, _ in ranked[:elite_count]]
    children: list[MarketGenome] = list(elites)
    parent_pool = [genome for genome, _ in ranked[:max(2, min(len(ranked), size // 2 + 1))]]
    while len(children) < size:
        if len(parent_pool) > 1 and rng.random() < 0.35:
            left, right = rng.sample(parent_pool, 2)
            child = crossover(left, right, rng)
        else:
            child = mutate(rng.choice(parent_pool), rng)
        if rng.random() < 0.20:
            child = mutate(child, rng)
        children.append(child)
    return unique_population(children, size, rng)


def load_state(path: Path) -> tuple[int, list[MarketGenome]] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["generation"]), [load_genome(item) for item in payload["population"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "runtime/market-evolution-process-isolated",
                        help="separate from the authoritative in-memory evolution service state")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runtime/benchmarks/market-corpus-manifest.json")
    parser.add_argument("--supplemental-root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"))
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--candidate-timeout-minutes", type=float, default=45.0)
    parser.add_argument("--min-free-memory-gb", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-generations", type=int, default=0,
                        help="zero runs perpetually")
    args = parser.parse_args()
    if args.population < 2 or not 1 <= args.elites < args.population:
        raise ValueError("population must exceed elites, with at least one elite")
    owner = claim(args.root)
    rng = random.Random(args.seed)
    state_path = args.root / "state.json"
    loaded = load_state(state_path)
    generation, population = loaded if loaded else (0, initial_population(args.population, rng))
    try:
        completed_generations = 0
        while not (args.root / "stop.requested").exists():
            while memory_available_gb() < args.min_free_memory_gb:
                write_status(args.root, phase="memory_wait", generation=generation,
                             available_memory_gb=memory_available_gb())
                time.sleep(args.poll_seconds)
            signature = dataset_signature(args.manifest, args.supplemental_root)
            generation_dir = args.root / f"generation-{generation:06d}"
            write_status(args.root, phase="evaluating", generation=generation,
                         population=len(population), dataset_signature=signature)
            results: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
                futures = {executor.submit(evaluate_one, args, genome, generation_dir, signature): genome
                           for genome in population}
                for future in as_completed(futures):
                    genome = futures[future]
                    result = future.result()
                    results[genome.genome_id] = result
                    write_status(args.root, phase="evaluating", generation=generation,
                                 completed=len(results), population=len(population),
                                 latest=result, dataset_signature=signature)
            ranked = sorted(((genome, results[genome.genome_id]) for genome in population),
                            key=lambda item: item[1]["score"], reverse=True)
            summary = {
                "generation": generation, "dataset_signature": signature,
                "finished_at": utcnow(),
                "ranking": [{"genome": genome.as_json(), "result": result}
                            for genome, result in ranked],
            }
            atomic_json(generation_dir / "summary.json", summary)
            champion, champion_result = ranked[0]
            current_champion_path = args.root / "surrogate-champion.json"
            prior_score = -1_000_000.0
            if current_champion_path.is_file():
                prior_score = json.loads(current_champion_path.read_text(encoding="utf-8"))["result"]["score"]
            if champion_result["score"] > prior_score:
                atomic_json(current_champion_path,
                            {"genome": champion.as_json(), "result": champion_result,
                             "promoted_at": utcnow()})
            if champion_result["gate"].get("working_target"):
                atomic_json(args.root / "brain-validation-queue.json", {
                    "genome": champion.as_json(), "surrogate_result": champion_result,
                    "queued_at": utcnow(),
                    "required_next_gate": "disposable Wizard micro-brains then full untouched admission",
                })
            population = next_population(ranked, args.population, args.elites, rng)
            generation += 1
            atomic_json(state_path, {"generation": generation,
                                     "population": [item.as_json() for item in population],
                                     "dataset_signature": signature, "updated_at": utcnow()})
            completed_generations += 1
            write_status(args.root, phase="generation_complete", generation=generation - 1,
                         champion=champion_result, next_generation=generation)
            if args.once or (args.max_generations and completed_generations >= args.max_generations):
                break
            time.sleep(max(0.0, args.poll_seconds))
        return 0
    finally:
        try:
            if owner.read_text(encoding="ascii").strip() == str(os.getpid()):
                owner.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
