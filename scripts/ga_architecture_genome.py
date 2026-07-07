"""Structural GA primitives for configuring Wizard brain instances.

The genome evolves topology, not only scalar hyperparameters: sensory pools,
meta-pools, feedback edges, delays, and the substrate signals driving each
edge/pool controller.  It emits JSON compatible with BrainIdentitySpec and
BrainDeploymentSpec.  Evaluation remains external so every genome can be run
in a disposable process/data directory with domain-specific held-out tests.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

SIGNALS = (
    "Surprise", "InvSurprise", "FiringRate", "InvFiringRate",
    "DecodePrecisionEma", "InvDecodePrecisionEma",
    "ConceptCountEma", "TerminalCountEma",
)
FEEDBACK_SIGNALS = ("activation_trace", "prediction_error", "reward", "attention", "surprise")


def driven(rng: random.Random) -> dict:
    lo = rng.uniform(0.0, 0.25)
    hi = rng.uniform(max(lo + 0.05, 0.3), 1.0)
    return {"DrivenBy": {
        "signal": rng.choice(SIGNALS),
        "scale": round(rng.uniform(-1.5, 1.5), 4),
        "offset": round(rng.uniform(0.0, 1.0), 4),
        "min": round(lo, 4), "max": round(hi, 4),
    }}


@dataclass
class PoolGene:
    name: str
    kind: str = "Internal"
    prefix: str = "meta"
    observes: list[str] = field(default_factory=list)
    sparsity_mode: dict | None = None
    predict_gate_mode: dict | None = None


@dataclass
class EdgeGene:
    source: str
    target: str
    signal: str
    delay_ticks: int
    gain_mode: dict


@dataclass
class ArchitectureGenome:
    name: str
    pools: list[PoolGene]
    edges: list[EdgeGene]
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    readout_pool: str | None = None
    settle_ticks: int = 0
    inference_mode: str = "integrate"

    def validate(self) -> None:
        names = [p.name for p in self.pools]
        if len(names) != len(set(names)):
            raise ValueError("pool names must be unique")
        if sum(p.kind == "Action" for p in self.pools) != 1:
            raise ValueError("genome must have exactly one action pool")
        known = set(names)
        if self.readout_pool is not None and self.readout_pool not in known:
            raise ValueError("readout pool does not exist")
        for pool in self.pools:
            if pool.kind == "Internal" and not pool.observes:
                raise ValueError(f"meta-pool {pool.name} observes nothing")
            if not set(pool.observes) <= known - {pool.name}:
                raise ValueError(f"meta-pool {pool.name} has invalid observations")
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("feedback edge references unknown pool")
            if edge.source == edge.target:
                raise ValueError("self feedback must pass through a distinct meta-pool")

    def materialize(self, data_dir: str) -> dict:
        self.validate()
        identity_pools = []
        for idx, gene in enumerate(self.pools, start=1):
            item = {
                "name": gene.name, "id": idx, "prototype": "byte-passthrough",
                "atom_encoding_prefix": gene.prefix, "kind": gene.kind,
            }
            if gene.sparsity_mode is not None:
                item["sparsity_mode"] = gene.sparsity_mode
            if gene.predict_gate_mode is not None:
                item["predict_gate_mode"] = gene.predict_gate_mode
            identity_pools.append(item)
        edges = [{
            "source_pool": e.source, "target_pool": e.target,
            "signal": e.signal, "gain": 1.0, "gain_mode": e.gain_mode,
            "delay_ticks": e.delay_ticks,
        } for e in self.edges]
        # Observed-pool activation is an explicit edge into each meta-pool.
        for pool in self.pools:
            for source in pool.observes:
                edges.append({
                    "source_pool": source, "target_pool": pool.name,
                    "signal": "activation_trace", "gain": 1.0,
                    "gain_mode": pool.sparsity_mode or {"Constant": 1.0},
                    "delay_ticks": 0,
                })
        return {
            "identity": {
                "name": self.name, "version": f"ga-{self.generation}",
                "pools": identity_pools, "binding_emergence_threshold": 3,
                "moment_history_window": 128,
            },
            "deployment": {
                "instance_id": self.name, "identity_path": "identity.json",
                "data_dir": data_dir,
                "resource_budget": {
                    "max_resident_bytes": 268435456, "max_neurons": 500000,
                    "max_propagation_steps": 64,
                    "max_learning_steps_per_second": 1000,
                },
                "feedback_loops": edges,
            },
            "lineage": {"generation": self.generation, "parents": self.parents},
            "evaluation_route": {
                "input_pool": self.pools[0].name,
                "query_pool": self.readout_pool or self.pools[0].name,
                "output_pool": next(p.name for p in self.pools if p.kind == "Action"),
                "settle_ticks": self.settle_ticks,
                "inference_mode": self.inference_mode,
            },
        }


def seed(domain: str, rng: random.Random) -> ArchitectureGenome:
    sensory = ["ohlcv", "news"] if domain == "market" else ["prompt", "context"]
    pools = [PoolGene(n, "SensoryInput", n) for n in sensory]
    pools += [PoolGene("meta_0", "Internal", "meta", sensory, driven(rng), driven(rng)),
              PoolGene("outcome", "Action", "outcome")]
    edges = [EdgeGene("outcome", n, "prediction_error", 1, driven(rng)) for n in sensory]
    # Preserve direct retrieval as the initial viable organism. Mutations can
    # move readout into meta-pools after feedback wiring proves useful.
    return ArchitectureGenome(f"{domain}-g0", pools, edges,
                              readout_pool=sensory[0], settle_ticks=0)


def mutate(parent: ArchitectureGenome, rng: random.Random, child_index: int) -> ArchitectureGenome:
    child = copy.deepcopy(parent)
    child.generation += 1
    child.parents = [parent.name]
    child.name = f"{parent.name.split('-g')[0]}-g{child.generation}-{child_index}"
    operation = rng.choice(("add_meta", "remove_meta", "rewire_meta", "add_edge", "remove_edge", "controller", "readout", "inference"))
    names = [p.name for p in child.pools]
    meta = [p for p in child.pools if p.kind == "Internal"]
    if operation == "add_meta":
        name = f"meta_{max([int(p.name.split('_')[-1]) for p in meta] or [-1]) + 1}"
        sources = rng.sample(names, k=min(len(names), rng.randint(1, min(3, len(names)))))
        child.pools.insert(-1, PoolGene(name, "Internal", name, sources, driven(rng), driven(rng)))
    elif operation == "remove_meta" and len(meta) > 1:
        victim = rng.choice(meta).name
        child.pools = [p for p in child.pools if p.name != victim]
        child.edges = [e for e in child.edges if victim not in (e.source, e.target)]
        for p in child.pools:
            p.observes = [n for n in p.observes if n != victim]
        if child.readout_pool == victim:
            child.readout_pool = child.pools[0].name
    elif operation == "rewire_meta" and meta:
        target = rng.choice(meta)
        candidates = [n for n in names if n != target.name]
        target.observes = rng.sample(candidates, k=rng.randint(1, min(3, len(candidates))))
    elif operation == "add_edge":
        source, target = rng.sample(names, 2)
        child.edges.append(EdgeGene(source, target, rng.choice(FEEDBACK_SIGNALS), rng.randint(0, 4), driven(rng)))
    elif operation == "remove_edge" and len(child.edges) > 1:
        child.edges.pop(rng.randrange(len(child.edges)))
    elif operation == "readout":
        child.readout_pool = rng.choice([p.name for p in child.pools if p.kind != "Action"])
        child.settle_ticks = rng.randint(0, 4)
    elif operation == "inference":
        child.inference_mode = rng.choice(("integrate", "chat", "hybrid"))
    else:
        target = rng.choice(child.pools)
        target.sparsity_mode, target.predict_gate_mode = driven(rng), driven(rng)
    child.validate()
    return child


def crossover(left: ArchitectureGenome, right: ArchitectureGenome, rng: random.Random,
              child_index: int) -> ArchitectureGenome:
    """Recombine complete topology and routing while retaining a valid brain."""
    ordered_names = list(dict.fromkeys([p.name for p in left.pools + right.pools]))
    pool_names = set(ordered_names)
    pools = []
    for name in ordered_names:
        candidates = [p for parent in (left, right) for p in parent.pools if p.name == name]
        gene = copy.deepcopy(rng.choice(candidates))
        gene.observes = [n for n in gene.observes if n in pool_names and n != name]
        if gene.kind == "Internal" and not gene.observes:
            gene.observes = [p.name for p in pools if p.kind != "Action"][:1]
        pools.append(gene)
    for pool in pools:
        pool.kind = "Action" if pool.name == "outcome" else pool.kind
        if pool.name != "outcome" and pool.kind == "Action":
            pool.kind = "Internal"
    edge_map = {}
    for edge in left.edges + right.edges:
        key = (edge.source, edge.target, edge.signal, edge.delay_ticks)
        if edge.source in pool_names and edge.target in pool_names:
            edge_map.setdefault(key, copy.deepcopy(edge))
    route_parent = rng.choice((left, right))
    inference_mode = ("hybrid" if left.inference_mode != right.inference_mode
                      and {left.inference_mode, right.inference_mode} <= {"chat", "integrate"}
                      else route_parent.inference_mode)
    generation = max(left.generation, right.generation) + 1
    child = ArchitectureGenome(
        f"coding-g{generation}-x{child_index}", pools, list(edge_map.values()), generation,
        [left.name, right.name], route_parent.readout_pool, route_parent.settle_ticks,
        inference_mode)
    child.validate()
    return child


def load_genome(run_dir: Path) -> ArchitectureGenome:
    """Rehydrate a genome from a prior isolated evaluation directory."""
    identity = json.loads((run_dir / "identity.json").read_text())
    deployment = json.loads((run_dir / "deployment.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    observed, explicit = {}, []
    kinds = {p["name"]: p["kind"] for p in identity["pools"]}
    for item in deployment.get("feedback_loops", []):
        target = item["target_pool"]
        if item["signal"] == "activation_trace" and kinds.get(target) == "Internal" and item["delay_ticks"] == 0:
            observed.setdefault(target, []).append(item["source_pool"])
        else:
            explicit.append(EdgeGene(item["source_pool"], target, item["signal"], item["delay_ticks"],
                                     item.get("gain_mode") or {"Constant": item.get("gain", 1.0)}))
    pools = [PoolGene(p["name"], p["kind"], p["atom_encoding_prefix"], observed.get(p["name"], []),
                      p.get("sparsity_mode"), p.get("predict_gate_mode")) for p in identity["pools"]]
    route = result["route"]
    genome = ArchitectureGenome(result["genome"], pools, explicit, result.get("generation", 0), [],
                                route["query_pool"], route["settle_ticks"],
                                route.get("inference_mode", "integrate"))
    genome.validate()
    return genome


def fitness(metrics: dict) -> float:
    """Multi-objective score; perfection requires every behavioral axis."""
    return (
        2.0 * metrics.get("exact_recall", 0.0)
        + 2.5 * metrics.get("paraphrase", 0.0)
        + 4.0 * metrics.get("composition", 0.0)
        + 2.0 * metrics.get("oov_honesty", 0.0)
        + 4.0 * metrics.get("directional_edge_over_baseline", 0.0)
        + 2.0 * metrics.get("calibration", 0.0)
        - 0.5 * metrics.get("latency_cost", 0.0)
        - 0.5 * metrics.get("memory_cost", 0.0)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("market", "coding"), required=True)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root = seed(args.domain, rng)
    genomes = [root] + [mutate(root, rng, i) for i in range(1, args.population)]
    args.out.mkdir(parents=True, exist_ok=True)
    for genome in genomes:
        materialized = genome.materialize(str(args.out / genome.name))
        (args.out / f"{genome.name}.identity.json").write_text(
            json.dumps(materialized["identity"], indent=2))
        (args.out / f"{genome.name}.deployment.json").write_text(
            json.dumps(materialized["deployment"], indent=2))
        (args.out / f"{genome.name}.lineage.json").write_text(
            json.dumps(materialized["lineage"], indent=2))
        (args.out / f"{genome.name}.route.json").write_text(
            json.dumps(materialized["evaluation_route"], indent=2))
    print(json.dumps({"domain": args.domain, "population": len(genomes), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
