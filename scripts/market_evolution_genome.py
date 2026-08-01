"""Genome and conservative fitness contract for perpetual market evolution."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

FEATURE_GROUPS = {
    "price": (
        "body", "range", "position", "upper_wick", "lower_wick", "r1", "r2", "r3",
        "r6", "r12", "r24", "r72", "r168", "acceleration", "rv6", "rv24",
        "rv168", "volatility_ratio", "drawdown", "location", "trend_vote",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ),
    "flow": ("volume_ratio24", "volume_ratio168", "flow_imbalance"),
    "cross": (
        "reference_r1", "reference_r6", "reference_r12", "reference_r24",
        "relative_r1", "relative_r6", "relative_r12", "relative_r24",
        "rolling_beta", "residual_r12",
    ),
    "derivatives": (
        "cex_dex_basis", "futures_spot_basis", "basis_delta6", "basis_delta24",
        "premium_close", "premium_delta6", "funding_rate", "funding_delta24",
        "spot_taker_imbalance", "futures_taker_imbalance", "flow_divergence",
        "spot_quote_ratio24", "futures_quote_ratio24", "spot_trade_ratio24",
        "futures_trade_ratio24", "futures_spot_quote_ratio",
    ),
    "breadth": (
        "market_median_r1", "market_median_r6", "market_median_r24",
        "market_breadth_r1", "market_breadth_r6", "market_dispersion_r6",
        "relative_market_r6", "relative_market_r24",
    ),
}
ALL_FEATURES = tuple(dict.fromkeys(feature for values in FEATURE_GROUPS.values() for feature in values))


@dataclass(frozen=True)
class MarketGenome:
    generation: int
    parents: tuple[str, ...]
    features: tuple[str, ...]
    horizon: int = 12
    stride: int = 12
    direction_threshold: float = 0.003
    confidence_quantile: float = 0.20
    learning_rate: float = 0.06
    max_iter: int = 180
    max_leaf_nodes: int = 24
    l2_regularization: float = 1.0
    active_pools: tuple[int, ...] = (2, 3, 4, 5, 9)
    train_per_pair: int = 60
    binding_emergence_threshold: int = 3
    concept_emergence_threshold: int = 5
    moment_history_window: int = 256

    def validate(self) -> None:
        if len(self.features) < 8 or not set(self.features) <= set(ALL_FEATURES):
            raise ValueError("genome needs at least eight known causal features")
        if self.horizon not in (4, 6, 12, 24, 48):
            raise ValueError("unsupported forecast horizon")
        if self.stride not in (4, 6, 12, 24):
            raise ValueError("unsupported sample stride")
        if not 0.002 <= self.direction_threshold <= 0.015:
            raise ValueError("direction threshold outside cost-aware bounds")
        if not 0.0 <= self.confidence_quantile <= 0.30:
            raise ValueError("confidence selectivity would violate the coverage floor")
        if not self.active_pools or not set(self.active_pools) <= set(range(1, 15)):
            raise ValueError("invalid Wizard pool selection")

    @property
    def genome_id(self) -> str:
        payload = asdict(self) | {"parents": (), "generation": 0}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"market-g{self.generation}-{digest}"

    def as_json(self) -> dict[str, Any]:
        return asdict(self) | {"genome_id": self.genome_id}


def seed(rng: random.Random, index: int = 0) -> MarketGenome:
    groups = ["price", "flow"]
    if index % 3 != 1:
        groups.append("derivatives")
    if index % 3 != 2:
        groups.append("breadth")
    features = tuple(dict.fromkeys(feature for group in groups for feature in FEATURE_GROUPS[group]))
    genome = MarketGenome(
        generation=0, parents=(), features=features,
        horizon=rng.choice((6, 12, 24)), stride=rng.choice((6, 12)),
        direction_threshold=rng.choice((0.003, 0.004, 0.006)),
        confidence_quantile=rng.choice((0.10, 0.20, 0.30)),
        active_pools=tuple(sorted(set((2, 4, 5, 9) + ((3,) if "flow" in groups else ())))),
    )
    genome.validate()
    return genome


def mutate(parent: MarketGenome, rng: random.Random) -> MarketGenome:
    operation = rng.choice((
        "feature", "feature", "horizon", "threshold", "confidence", "model",
        "pool", "training", "binding", "concept", "history",
    ))
    child = replace(parent, generation=parent.generation + 1, parents=(parent.genome_id,))
    if operation == "feature":
        features = set(child.features)
        feature = rng.choice(ALL_FEATURES)
        if feature in features and len(features) > 8:
            features.remove(feature)
        else:
            features.add(feature)
        child = replace(child, features=tuple(sorted(features)))
    elif operation == "horizon":
        child = replace(child, horizon=rng.choice((4, 6, 12, 24, 48)))
    elif operation == "threshold":
        child = replace(child, direction_threshold=rng.choice((0.002, 0.003, 0.004, 0.006, 0.01)))
    elif operation == "confidence":
        child = replace(child, confidence_quantile=rng.choice((0.0, 0.10, 0.20, 0.25, 0.30)))
    elif operation == "model":
        child = replace(child, learning_rate=rng.choice((0.025, 0.04, 0.06, 0.09)),
                        max_iter=rng.choice((100, 140, 180, 240)),
                        max_leaf_nodes=rng.choice((12, 16, 24, 32)),
                        l2_regularization=rng.choice((0.25, 0.5, 1.0, 2.0, 4.0)))
    elif operation == "pool":
        pools = set(child.active_pools)
        pool = rng.choice(tuple(range(1, 15)))
        if pool in pools and len(pools) > 1:
            pools.remove(pool)
        else:
            pools.add(pool)
        child = replace(child, active_pools=tuple(sorted(pools)))
    elif operation == "training":
        child = replace(child, train_per_pair=rng.choice((30, 45, 60, 80, 120)))
    elif operation == "binding":
        child = replace(child, binding_emergence_threshold=rng.choice((2, 3, 4, 5, 6)))
    elif operation == "concept":
        child = replace(child, concept_emergence_threshold=rng.choice((3, 4, 5, 6, 8)))
    elif operation == "history":
        child = replace(child, moment_history_window=rng.choice((64, 128, 256, 512, 1024)))
    child.validate()
    return child


def crossover(left: MarketGenome, right: MarketGenome, rng: random.Random) -> MarketGenome:
    shared = set(left.features) & set(right.features)
    optional = list(set(left.features) ^ set(right.features))
    rng.shuffle(optional)
    features = tuple(sorted(shared | set(optional[:max(0, len(optional) // 2)])))
    scalar_names = (
        "horizon", "stride", "direction_threshold", "confidence_quantile", "learning_rate",
        "max_iter", "max_leaf_nodes", "l2_regularization", "active_pools", "train_per_pair",
        "binding_emergence_threshold", "concept_emergence_threshold", "moment_history_window",
    )
    values = {name: getattr(rng.choice((left, right)), name) for name in scalar_names}
    child = MarketGenome(max(left.generation, right.generation) + 1,
                         (left.genome_id, right.genome_id), features, **values)
    child.validate()
    return child


def load_genome(payload: dict[str, Any]) -> MarketGenome:
    values = {key: value for key, value in payload.items() if key != "genome_id"}
    for key in ("parents", "features", "active_pools"):
        values[key] = tuple(values[key])
    genome = MarketGenome(**values)
    genome.validate()
    return genome


def report_fitness(report: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Score the weakest fold/split; lucky averages cannot dominate evolution."""
    folds = next(iter(report["feature_sets"].values()))
    sections = []
    for fold in folds:
        if fold.get("skipped"):
            return -1_000_000.0, {"admitted": False, "reason": fold["skipped"]}
        for name in ("known_asset_future", "unseen_asset_future"):
            base = fold["sections"][name]
            selected = base.get("selective") or base.get("selective_70") or base
            sections.append(selected)
    minima = {
        "accuracy": min(value.get("directional_accuracy", 0.0) for value in sections),
        "balanced_accuracy": min(value.get("directional_balanced_accuracy", 0.0) for value in sections),
        "mcc": min(value.get("mcc", -1.0) for value in sections),
        "coverage": min(value.get("coverage", 1.0) for value in sections),
        "observations": min(value.get("acted_observations", value.get("observations", 0)) for value in sections),
        "expectancy": min(value.get("net_expectancy", -1.0) for value in sections),
        "profit_factor": min(value.get("profit_factor") or 0.0 for value in sections),
        "ece": max(value.get("ece", 1.0) for value in sections),
        "baseline_margin": min(value.get("baseline_margin", -1.0) for value in sections),
    }
    admitted = (
        minima["accuracy"] >= 0.58 and minima["balanced_accuracy"] >= 0.55
        and minima["mcc"] >= 0.15 and minima["coverage"] >= 0.70
        and minima["observations"] >= 200 and minima["expectancy"] > 0
        and minima["profit_factor"] >= 1.20 and minima["ece"] <= 0.10
        and minima["baseline_margin"] >= 0.05
    )
    working_target = (
        admitted and minima["accuracy"] >= 0.62
        and minima["balanced_accuracy"] >= 0.58
        and minima["mcc"] >= 0.25 and minima["profit_factor"] >= 1.35
    )
    score = (
        240 * minima["accuracy"] + 100 * minima["balanced_accuracy"]
        + 45 * minima["mcc"] + 80 * minima["baseline_margin"]
        + 8 * min(minima["profit_factor"], 2.0) + 800 * minima["expectancy"]
        + 10 * minima["coverage"] - 25 * minima["ece"]
        - 0.1 * max(0, 200 - minima["observations"])
    )
    return score, {"admitted": admitted, "working_target": working_target, "minima": minima}
