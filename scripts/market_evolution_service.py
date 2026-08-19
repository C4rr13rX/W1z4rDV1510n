#!/usr/bin/env python3
"""Perpetually evolve causal market feature and brain-configuration genomes.

This service is a durable, restartable screening layer for the Wizard market
brain.  It never promotes from training fitness or one lucky interval.  Every
genome is scored on purged walk-forward folds and complete unseen assets;
failed candidates remain in the ledger, while only protected winners become
eligible for the slower isolated Wizard-brain gate.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import joblib
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_memory_guard import VerifiedWorkingSetReclaimer

FEATURE_SCHEMA = 5
# 15: profit-first objective (2026-08-17). curriculum_fitness was rewritten
# so profit dominates and the reward scales with how many walk-forward folds
# actually ran. Every cached result therefore carries a fitness computed
# under the OLD accuracy-first weighting -- the incumbent champion was still
# holding 2384.7 for a PF 0.983 model that scores ~1009 now, which would let
# stale scores outrank freshly measured profit. Bumping the schema retires
# that evidence so selection compares like with like.
EVOLUTION_SCHEMA = 15
LEARNER_KINDS = (
    "classifier", "regressor", "extra_trees", "decomposed_regressor",
    "regime_regressor", "regime_decomposed_regressor",
)
RETURN_LEARNER_KINDS = {
    "regressor", "decomposed_regressor", "regime_regressor",
    "regime_decomposed_regressor", "multiscale_regressor",
    "extra_trees_regressor", "continuous_rank_regressor",
}


@dataclass
class ExtraTreesHybridModel:
    """Use independent tree pools for direction and trade desirability."""
    direction_model: Any
    return_model: Any

    @property
    def classes_(self) -> Any:
        return self.direction_model.classes_

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self.direction_model.predict_proba(values)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.direction_model.predict(values)
REGIME_FEATURES = (
    "rv24", "volatility_ratio", "market_breadth_r6", "market_median_r6",
    "futures_spot_basis", "funding_rate", "flow_divergence", "news_polarity_24h",
)
RELIABILITY_FEATURES = (
    "rv24", "volatility_ratio", "market_breadth_r6", "relative_market_r6",
    "flow_divergence", "futures_spot_basis", "news_negative_share_24h",
)
RELIABILITY_FEATURE_POOLS = {
    "core": RELIABILITY_FEATURES,
    "trend_regime": (
        "r1", "r6", "r12", "r24", "r72", "rv24", "volatility_ratio",
        "market_median_r6", "market_breadth_r6", "relative_market_r6",
        "cross_rank_r6", "cross_rank_r24",
    ),
    "flow_news": (
        "spot_taker_imbalance", "futures_taker_imbalance", "flow_imbalance",
        "flow_divergence", "futures_spot_basis", "funding_rate",
        "basis_z24", "funding_z168", "asset_news_sentiment_acceleration",
        "asset_news_sentiment_24h", "news_polarity_24h",
        "news_negative_share_24h", "news_macro_24h", "news_regulation_24h",
        "news_security_24h", "news_liquidation_24h",
    ),
    "flow_derivatives": (
        "spot_taker_imbalance", "futures_taker_imbalance", "flow_imbalance",
        "flow_divergence", "futures_spot_basis", "funding_rate",
        "basis_z24", "funding_z168",
    ),
    "news_regime": (
        "asset_news_sentiment_acceleration", "asset_news_sentiment_24h",
        "news_polarity_24h", "news_negative_share_24h", "news_macro_24h",
        "news_regulation_24h", "news_security_24h", "news_liquidation_24h",
        "rv24", "volatility_ratio", "market_breadth_r6",
    ),
    "relative_trend": (
        "r1", "r6", "r12", "r24", "r72", "market_median_r6",
        "market_breadth_r6", "relative_market_r6", "cross_rank_r6",
        "cross_rank_r24",
    ),
}
RELIABILITY_FEATURE_POOLS["combined"] = tuple(dict.fromkeys(
    # Preserve the empirically stronger flow/news specialist, then add only
    # compact regime modifiers whose co-occurrence splits its reliability.
    # The full trend pool is intentionally not concatenated after its isolated
    # protected result underperformed flow/news on the hard temporal fold.
    RELIABILITY_FEATURE_POOLS["flow_news"] + RELIABILITY_FEATURES
))
AUXILIARY_HORIZONS = (1, 6, 12, 24)
EVOLVABLE_POOL_NAMES = (
    "forecast_horizon", "instrument_context", "price_geometry", "trade_flow",
    "cross_market_context", "derivatives_state", "market_breadth", "news_state",
    "evolved_causal_relationships", "regime_context", "specialist_arbitration",
    "realized_ghost_experience", "competitive_reflexivity",
)
MAX_EMERGENT_POOLS = 8
MAX_EMERGENT_POOL_FEATURES = 4

from scripts.market_brain_experiment import load_bars  # noqa: E402
from scripts.market_signal_audit import (  # noqa: E402
    FEATURE_GROUPS, attach_market_breadth, build_rows, load_supplemental,
    metrics, select_primary_assets, stable_order,
)

# ─── THE OBJECTIVE ──────────────────────────────────────────────────────────
#
# One statement of what this system is for. Everything below derives from it,
# and nothing may silently contradict it.
#
#   Make money, net of what it actually costs to trade.
#
# Two rules follow, and they are not negotiable by any other threshold:
#
#   1. PROFIT DECIDES. A candidate that earns more money outranks one that is
#      more accurate, trades more often, or looks better on any other metric.
#      Accuracy, MCC, coverage and calibration are tie-breakers only -- they
#      correlate with durable edges, so they steer breeding, but they never
#      outvote the money.
#
#   2. PROFIT MUST BE MEASURED, NOT CLAIMED. A profit factor is only real if
#      it survived the full walk-forward at the true execution cost. Anything
#      measured on fewer folds, or at a cost lower than we actually pay, is a
#      hypothesis -- and hypotheses never outrank measurements.
#
# Why this is written down: for 1347 generations the fitness function weighted
# accuracy 20x profit while scoring candidates on a single fold, so the search
# optimised for being right rather than for earning, on numbers that were not
# real. It produced a champion that was 55.6% accurate and lost money. The
# objective was never stated anywhere, so nothing caught the drift.
#
# OBJECTIVE_PROFIT_FACTOR is the bar that means "this is worth trading".
# OBJECTIVE_COST_BPS must reflect what a round trip actually costs; evaluating
# below it manufactures profit that will not exist live. Measured 2026-08-17:
# GAS_ROUNDTRIP_FEE_RATIO alone is 25 bps before slippage or DEX fees.
# Lowered 1.10 -> 1.05 on 2026-08-18 to start accumulating ghost evidence
# while the search continues. This is the bar for "worth trading at all",
# not a claim that 1.05 is the goal: ghost results, not backtest profit,
# decide live promotion, so a lower entry bar buys real execution data
# sooner without weakening the graduation gate that actually risks money.
OBJECTIVE_PROFIT_FACTOR = 1.05
OBJECTIVE_COST_BPS = 25.0

FLOOR = {
    "observations": 200, "coverage": 0.70, "accuracy": 0.58,
    "balanced_accuracy": 0.55, "baseline_margin": 0.05, "mcc": 0.15,
    "ece": 0.10, "profit_factor": 1.20, "max_drawdown": 0.15,
}
WORKING_TARGET = {"accuracy": 0.62, "balanced_accuracy": 0.58,
                  "mcc": 0.25, "profit_factor": max(1.35, OBJECTIVE_PROFIT_FACTOR)}
PRESCREEN = {
    "observations": 150, "coverage": 0.60, "accuracy": 0.54,
    "balanced_accuracy": 0.52, "mcc": 0.04, "ece": 0.20,
    "profit_factor": 0.85,
}
# A candidate whose worst section is at least break-even earns the remaining
# walk-forward folds even when it misses the non-economic prescreen floors.
# Profit is the objective, so profitable evidence must be measured to
# completion rather than discarded on fold 1 -- but it earns MEASUREMENT,
# never admission. 1.0 = must at least not be losing money to continue.
PROFIT_CONTINUE_FLOOR = 1.0
COMPETENCE_FLOOR = {
    "observations": 30, "accuracy": 0.55, "balanced_accuracy": 0.52,
    "mcc": 0.04, "profit_factor": 1.05,
}
COMPETENCE_FEATURES = (
    "rv24", "volatility_ratio", "market_breadth_r6", "market_median_r6",
    "futures_spot_basis", "funding_rate", "flow_divergence",
    "news_polarity_24h",
)
DERIVED_FEATURES = {
    "flow_consensus": lambda f: (f["spot_taker_imbalance"] + f["futures_taker_imbalance"]) / 2,
    "flow_basis_pressure": lambda f: f["flow_divergence"] * f["futures_spot_basis"],
    "funding_basis_pressure": lambda f: f["funding_rate"] * f["futures_spot_basis"],
    "vol_adjusted_r6": lambda f: f["r6"] / max(abs(f["rv24"]), 1e-6),
    "vol_adjusted_r24": lambda f: f["r24"] / max(abs(f["rv168"]), 1e-6),
    "breadth_gap_r6": lambda f: f["r6"] - f["market_median_r6"],
    "breadth_alignment": lambda f: f["trend_vote"] * (f["market_breadth_r6"] - .5),
    "liquidity_acceleration": lambda f: (
        f["futures_quote_ratio24"] - f["spot_quote_ratio24"]),
    "participant_direction": lambda f: statistics.fmean((
        math.tanh(2 * f["futures_taker_imbalance"]),
        math.tanh(f["basis_z24"]), math.tanh(f["funding_z168"]),
    )),
    "participant_consensus": lambda f: abs(statistics.fmean((
        math.tanh(2 * f["futures_taker_imbalance"]),
        math.tanh(f["basis_z24"]), math.tanh(f["funding_z168"]),
    ))),
    "participant_disagreement": lambda f: statistics.pstdev((
        math.tanh(2 * f["spot_taker_imbalance"]),
        math.tanh(2 * f["futures_taker_imbalance"]),
        math.tanh(f["basis_z24"]), math.tanh(f["funding_z168"]),
    )),
    "crowding_intensity": lambda f: statistics.fmean((
        abs(math.tanh(f["basis_z24"])), abs(math.tanh(f["funding_z168"])),
        abs(math.tanh(2 * f["futures_taker_imbalance"])),
    )),
    "crowd_price_alignment": lambda f: f["participant_direction"] * math.tanh(
        f["r6"] / max(abs(f["rv24"]), 1e-6)
    ),
    "participant_pressure_acceleration": lambda f: statistics.fmean((
        math.tanh(f["imbalance_acceleration"]),
        math.tanh(f["basis_delta6"] * 100),
        math.tanh(f["funding_delta24"] * 10_000),
    )),
}
ROLLING_SOURCES = (
    "r1", "r6", "r12", "r24", "r72", "r168", "rv24",
    "volatility_ratio", "volume_ratio24", "flow_imbalance",
    "market_median_r6", "market_breadth_r6", "relative_market_r6",
    "futures_spot_basis", "funding_rate", "flow_divergence",
)
ROLLING_WINDOWS = (14, 60)
ROLLING_FEATURES = tuple(
    f"causal_z{window}_{name}" for name in ROLLING_SOURCES for window in ROLLING_WINDOWS
)
CROSS_SECTION_SOURCES = (
    "r1", "r6", "r12", "r24", "volume_ratio24", "flow_imbalance",
    "futures_spot_basis", "funding_rate",
)
CROSS_SECTION_FEATURES = tuple(f"cross_rank_{name}" for name in CROSS_SECTION_SOURCES)
EVOLVED_OPS = (
    "add", "sub", "mul", "ratio", "abs_gap", "signed_sqrt_product",
    "regime_gate", "tanh_mix",
)
NEWS_FEATURES = (
    "news_count_6h", "news_count_24h", "news_count_72h",
    "news_sentiment_6h", "news_sentiment_24h", "news_sentiment_72h",
    "news_polarity_24h", "asset_news_count_6h", "asset_news_count_24h",
    "asset_news_count_72h",
    "asset_news_sentiment_6h", "asset_news_sentiment_24h",
    "asset_news_sentiment_72h", "asset_news_sentiment_acceleration",
    "news_negative_share_24h", "news_regulation_24h", "news_macro_24h",
    "news_security_24h", "news_institutional_24h", "news_liquidation_24h",
    "news_exchange_24h", "news_stablecoin_24h", "news_network_24h",
    "news_whale_24h",
)
NEWS_CATEGORIES = {
    "regulation": ("REGULAT", " SEC ", "CFTC", "LAWMAKER", "LEGISLAT", "COURT"),
    "macro": ("FED ", "FEDERAL RESERVE", "INTEREST RATE", "INFLATION", "CPI", "JOBS REPORT"),
    "security": ("HACK", "EXPLOIT", "BREACH", "ATTACK", "VULNERAB", "DRAIN"),
    "institutional": (" ETF", "INSTITUTION", "TREASURY", "BLACKROCK", "FIDELITY"),
    "liquidation": ("LIQUIDAT", "LEVERAGE", "MARGIN CALL", "SHORT SQUEEZE"),
    "exchange": ("EXCHANGE", "BINANCE", "COINBASE", "KRAKEN", "BYBIT"),
    "stablecoin": ("STABLECOIN", "USDT", "USDC", "TETHER", "CIRCLE"),
    "network": ("UPGRADE", "FORK", "MAINNET", "OUTAGE", "VALIDATOR", "PROTOCOL"),
    "whale": ("WHALE", "LARGE HOLDER", "ON-CHAIN", "ONCHAIN"),
}
NEWS_SPECIALIST_FEATURES = (
    "news_count_24h", "news_sentiment_24h", "news_polarity_24h",
    "asset_news_count_24h", "asset_news_sentiment_24h",
    "asset_news_sentiment_acceleration", "news_negative_share_24h",
    "news_macro_24h", "news_regulation_24h", "news_security_24h",
    "news_liquidation_24h",
)
BASE_FEATURES = tuple(dict.fromkeys(
    FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"]
    + FEATURE_GROUPS["cross"] + FEATURE_GROUPS["derivatives"]
    + FEATURE_GROUPS["breadth"] + NEWS_FEATURES
    + ROLLING_FEATURES + CROSS_SECTION_FEATURES
))
REFLEXIVITY_FEATURES = {
    "participant_direction", "participant_consensus", "participant_disagreement",
    "crowding_intensity", "crowd_price_alignment", "participant_pressure_acceleration",
}
DERIVATIVE_FEATURES = set(FEATURE_GROUPS["derivatives"]) | {
    "flow_consensus", "flow_basis_pressure", "funding_basis_pressure",
    "liquidity_acceleration",
}
DERIVATIVE_FEATURES.update(REFLEXIVITY_FEATURES)
DERIVATIVE_FEATURES.update(
    f"causal_z{window}_{source}"
    for source in FEATURE_GROUPS["derivatives"] for window in ROLLING_WINDOWS
)
DERIVATIVE_FEATURES.update(
    f"cross_rank_{source}" for source in FEATURE_GROUPS["derivatives"]
)


@dataclass
class Genome:
    features: list[str]
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    confidence_quantile: float
    binding_threshold: int
    concept_threshold: int
    presentations: int
    feature_programs: list[dict[str, Any]] = field(default_factory=list)
    recency_half_life_days: float = 720.0
    selection_max_iter: int = 0
    selection_max_leaf_nodes: int = 0
    selection_min_samples_leaf: int = 0
    selection_recency_half_life_days: float = 0.0
    learner_kind: str = "classifier"
    market_weight: float = 1.0
    regime_feature: str = "rv24"
    regime_bins: int = 1
    training_horizons: list[int] = field(default_factory=lambda: [12])
    pool_thresholds: dict[str, int] = field(default_factory=dict)
    emergent_pools: list[dict[str, Any]] = field(default_factory=list)
    calibration_safety: float = 1.0
    calibration_orientation: bool = False
    calibration_reliability: bool = False
    calibration_reliability_version: int = 0
    calibration_reliability_pool: str = "core"
    calibration_reliability_decay: float = 0.0
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    genome_id: str = ""
    fitness: float | None = None
    result: dict[str, Any] | None = None

    def finalize(self) -> "Genome":
        payload = asdict(self)
        for key in ("genome_id", "fitness", "result"):
            payload.pop(key, None)
        self.features = sorted(set(self.features))
        if self.regime_feature not in REGIME_FEATURES:
            self.regime_feature = "rv24"
        self.regime_bins = max(1, min(3, int(self.regime_bins)))
        self.training_horizons = sorted(
            {12, *(int(value) for value in self.training_horizons
                   if int(value) in AUXILIARY_HORIZONS)}
        )
        self.pool_thresholds = {
            name: max(2, min(12, int(value)))
            for name, value in sorted(self.pool_thresholds.items())
            if name in EVOLVABLE_POOL_NAMES
        }
        self.calibration_safety = max(1.0, min(12.0, float(self.calibration_safety)))
        self.calibration_orientation = bool(self.calibration_orientation)
        self.calibration_reliability = bool(self.calibration_reliability)
        self.calibration_reliability_version = max(
            0, min(8, int(self.calibration_reliability_version))
        )
        if self.calibration_reliability_pool not in RELIABILITY_FEATURE_POOLS:
            self.calibration_reliability_pool = "core"
        self.calibration_reliability_decay = max(
            0.0, min(8.0, float(self.calibration_reliability_decay))
        )
        if self.learner_kind == "extra_trees_hybrid":
            self.selection_max_iter = min(360, max(
                80, int(self.selection_max_iter or self.max_iter)
            ))
            self.selection_max_leaf_nodes = min(72, max(
                8, int(self.selection_max_leaf_nodes or self.max_leaf_nodes)
            ))
            self.selection_min_samples_leaf = min(100, max(
                8, int(self.selection_min_samples_leaf or self.min_samples_leaf)
            ))
            self.selection_recency_half_life_days = min(2200.0, max(
                45.0, float(
                    self.selection_recency_half_life_days
                    or self.recency_half_life_days
                )
            ))
        else:
            self.selection_max_iter = int(self.max_iter)
            self.selection_max_leaf_nodes = int(self.max_leaf_nodes)
            self.selection_min_samples_leaf = int(self.min_samples_leaf)
            self.selection_recency_half_life_days = float(
                self.recency_half_life_days
            )
        if (self.regime_bins > 1
                or self.learner_kind in {"regime_regressor", "regime_decomposed_regressor"}):
            self.features = sorted(set(self.features) | {self.regime_feature})
        self.feature_programs = sorted(
            (normalize_program(program) for program in self.feature_programs),
            key=lambda program: json.dumps(program, sort_keys=True, separators=(",", ":")),
        )
        active_sources = set(self.features) | {
            program_name(program) for program in self.feature_programs
        }
        normalized_pools: dict[tuple[str, ...], dict[str, Any]] = {}
        for pool in self.emergent_pools:
            normalized = normalize_emergent_pool(pool, active_sources)
            if normalized is not None:
                normalized_pools[tuple(normalized["features"])] = normalized
        self.emergent_pools = [
            normalized_pools[key] for key in sorted(normalized_pools)
        ][:MAX_EMERGENT_POOLS]
        payload["features"] = self.features
        payload["feature_programs"] = self.feature_programs
        payload["regime_feature"] = self.regime_feature
        payload["regime_bins"] = self.regime_bins
        payload["training_horizons"] = self.training_horizons
        payload["pool_thresholds"] = self.pool_thresholds
        payload["emergent_pools"] = self.emergent_pools
        payload["calibration_safety"] = self.calibration_safety
        payload["calibration_orientation"] = self.calibration_orientation
        payload["calibration_reliability"] = self.calibration_reliability
        payload["calibration_reliability_version"] = (
            self.calibration_reliability_version
        )
        payload["calibration_reliability_pool"] = self.calibration_reliability_pool
        payload["calibration_reliability_decay"] = self.calibration_reliability_decay
        payload["selection_max_iter"] = self.selection_max_iter
        payload["selection_max_leaf_nodes"] = self.selection_max_leaf_nodes
        payload["selection_min_samples_leaf"] = self.selection_min_samples_leaf
        payload["selection_recency_half_life_days"] = (
            self.selection_recency_half_life_days
        )
        self.genome_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return self


def normalize_program(program: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a bounded causal feature expression for stable heredity."""
    universe = set(BASE_FEATURES) | set(DERIVED_FEATURES)
    op = str(program.get("op", "sub"))
    if op not in EVOLVED_OPS:
        op = "sub"
    left = str(program.get("left", "r6"))
    right = str(program.get("right", "r24"))
    if left not in universe:
        left = "r6"
    if right not in universe:
        right = "r24"
    scale = max(.05, min(20.0, float(program.get("scale", 1.0))))
    return {"op": op, "left": left, "right": right, "scale": round(scale, 8)}


def program_name(program: dict[str, Any]) -> str:
    encoded = json.dumps(normalize_program(program), sort_keys=True,
                         separators=(",", ":")).encode()
    return "evolved_" + hashlib.sha256(encoded).hexdigest()[:12]


def normalize_emergent_pool(
    pool: dict[str, Any], active_sources: set[str],
) -> dict[str, Any] | None:
    """Canonicalize one heritable specialist pool around active features."""
    sources = sorted({
        str(source) for source in pool.get("features", [])
        if str(source) in active_sources
    })[:MAX_EMERGENT_POOL_FEATURES]
    if not sources:
        return None
    digest = hashlib.sha256("\0".join(sources).encode()).hexdigest()[:12]
    return {
        "name": f"emergent_{digest}",
        "features": sources,
        "concept_threshold": min(12, max(
            2, int(pool.get("concept_threshold", 5))
        )),
    }


def emergent_pool_sources(genome: "Genome") -> list[str]:
    return sorted(set(genome.features) | {
        program_name(program) for program in genome.feature_programs
    })


def random_emergent_pool(genome: "Genome", rng: random.Random) -> dict[str, Any] | None:
    """Create a bounded specialist hypothesis, favoring true isolation."""
    sources = emergent_pool_sources(genome)
    if not sources:
        return None
    width = min(len(sources), rng.choices((1, 2, 3, 4), (5, 3, 1, 1))[0])
    selected = rng.sample(sources, width)
    return normalize_emergent_pool(
        {"features": selected, "concept_threshold": rng.randint(3, 8)},
        set(sources),
    )


def program_value(features: dict[str, float], program: dict[str, Any]) -> float:
    """Evaluate one same-moment expression; no target or future state is accessible."""
    program = normalize_program(program)
    left = float(features.get(program["left"], 0.0))
    right = float(features.get(program["right"], 0.0))
    scale = float(program["scale"])
    op = program["op"]
    if op == "add":
        value = left + scale * right
    elif op == "sub":
        value = left - scale * right
    elif op == "mul":
        value = scale * left * right
    elif op == "ratio":
        value = left / (abs(right) + 1e-6 * scale)
    elif op == "abs_gap":
        value = abs(left) - scale * abs(right)
    elif op == "signed_sqrt_product":
        product = left * right
        value = math.copysign(math.sqrt(abs(product)), product) * scale
    elif op == "regime_gate":
        value = left if right >= 0 else -scale * left
    else:  # tanh_mix
        value = math.tanh(scale * left) + math.tanh(scale * right)
    return float(max(-1e6, min(1e6, value))) if math.isfinite(value) else 0.0


def feature_vector(row: dict[str, Any], genome: "Genome") -> list[float]:
    features = row["features"]
    return ([float(features.get(name, 0.0)) for name in genome.features]
            + [program_value(features, program) for program in genome.feature_programs])


def random_program(rng: random.Random) -> dict[str, Any]:
    universe = list(BASE_FEATURES) + list(DERIVED_FEATURES)
    return normalize_program({
        "op": rng.choice(EVOLVED_OPS), "left": rng.choice(universe),
        "right": rng.choice(universe), "scale": 10 ** rng.uniform(-.7, .7),
    })


def genome_uses_derivatives(genome: "Genome") -> bool:
    references = set(genome.features)
    for program in genome.feature_programs:
        references.add(str(program.get("left", "")))
        references.add(str(program.get("right", "")))
    return bool(references & DERIVATIVE_FEATURES)


@dataclass
class Surrogate:
    model: Any
    kind: str
    score_scale: float = 1.0
    market_model: Any = None
    market_weight: float = 1.0
    reliability_model: Any = None
    reliability_indices: tuple[int, ...] = ()
    reliability_mean: Any = None
    reliability_scale: Any = None
    reliability_direction: float = 1.0

    def raw_score(self, values: np.ndarray) -> np.ndarray:
        if self.kind == "decomposed_regressor":
            residual = np.asarray(self.model.predict(values), dtype=np.float64)
            market = np.asarray(self.market_model.predict(values), dtype=np.float64)
            return residual + self.market_weight * market
        return np.asarray(self.model.predict(values), dtype=np.float64)

    def base_probability(self, values: np.ndarray) -> np.ndarray:
        if self.kind in {
            "classifier", "extra_trees", "extra_trees_ranked",
            "extra_trees_hybrid",
        }:
            return self.model.predict_proba(values)[:, list(self.model.classes_).index(1)]
        score = self.raw_score(values)
        normalized = np.clip(score / max(self.score_scale, 1e-9), -30, 30)
        return 1.0 / (1.0 + np.exp(-normalized))

    def reliability_design(
        self, values: np.ndarray, probability: np.ndarray,
    ) -> np.ndarray:
        confidence = np.maximum(probability, 1.0 - probability)
        columns = [confidence.reshape(-1, 1)]
        if self.reliability_indices:
            columns.append(values[:, self.reliability_indices])
        return np.column_stack(columns)

    def fit_reliability(
        self, values: np.ndarray, labels: np.ndarray,
        feature_indices: Sequence[int], version: int = 1,
        decay: float = 0.0,
    ) -> bool:
        """Learn correctness ranking from calibration only, never direction."""
        probability = self.base_probability(values)
        prediction = np.where(probability >= .5, 1, -1).astype(np.int8)
        correctness = (prediction == np.asarray(labels, dtype=np.int8)).astype(np.int8)
        if len(values) < 80 or len(np.unique(correctness)) < 2:
            return False
        self.reliability_indices = tuple(int(index) for index in feature_indices)
        design = self.reliability_design(values, probability)
        self.reliability_mean = design.mean(axis=0)
        self.reliability_scale = np.maximum(design.std(axis=0), 1e-6)
        normalized = (design - self.reliability_mean) / self.reliability_scale
        if version >= 2:
            # Correctness depends on interactions such as high volatility plus
            # weak breadth or adverse news plus flow divergence.  A bounded
            # nonlinear pool can discover those WHEN/WHY regions while the
            # directional base model remains untouched.
            effective_decay = (float(decay) if decay > 0 else 2.0)
            reliability_weights = (
                np.exp(np.linspace(-effective_decay, 0.0, len(correctness)))
                if version >= 5 else None
            )
            self.reliability_model = ExtraTreesClassifier(
                n_estimators=160, max_leaf_nodes=24, min_samples_leaf=20,
                max_features="sqrt", class_weight="balanced", n_jobs=1,
                random_state=2718,
            ).fit(normalized, correctness, sample_weight=reliability_weights)
        else:
            self.reliability_model = LogisticRegression(
                C=.2, solver="lbfgs", max_iter=300, class_weight="balanced",
            ).fit(normalized, correctness)
        return True

    def probability(self, values: np.ndarray) -> np.ndarray:
        return self.base_probability(values)

    @staticmethod
    def causal_tie_break(values: np.ndarray) -> np.ndarray:
        """Stable label-free rank for otherwise indistinguishable scores.

        Histogram/tree leaves intentionally emit repeated predictions. A hard
        quantile threshold then admits or rejects the whole leaf, which can
        jump coverage by tens of percentage points. This bounded projection
        uses only same-row causal inputs and is far too small to reorder any
        materially different confidence values; it only makes a tied leaf
        continuously selectable.
        """
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or not len(matrix):
            return np.zeros(len(matrix), dtype=np.float64)
        bounded = np.tanh(np.nan_to_num(
            matrix, nan=0.0, posinf=20.0, neginf=-20.0,
        ))
        index = np.arange(1, bounded.shape[1] + 1, dtype=np.float64)
        weights = np.sin(index * 1.618033988749895)
        secondary = np.cos(index * 0.7548776662466927)
        projection = bounded @ weights + (bounded * bounded) @ secondary
        return 0.5 + 0.5 * np.sin(projection * 12.9898 + 0.61803398875)

    def selection_confidence(self, values: np.ndarray) -> np.ndarray:
        """Return an abstention score independent of directional probability."""
        probability = self.base_probability(values)
        if self.reliability_model is None:
            if self.kind == "extra_trees_hybrid":
                return np.abs(np.asarray(
                    self.model.return_model.predict(values), dtype=np.float64,
                ))
            score = np.maximum(probability, 1.0 - probability)
            if self.kind in {"continuous_rank_regressor", "extra_trees_ranked"}:
                score = score + 1e-10 * self.causal_tie_break(values)
            return score
        design = self.reliability_design(values, probability)
        normalized = (design - self.reliability_mean) / self.reliability_scale
        # This score chooses whether to abstain. Direction and ECE continue to
        # use base_probability, so low predicted correctness cannot flip or
        # masquerade as calibrated directional confidence.
        score = self.reliability_model.predict_proba(normalized)[:, 1]
        if self.reliability_direction < 0:
            return 1.0 - score
        if self.reliability_direction == 0:
            return np.full(len(score), .5, dtype=np.float64)
        if self.kind in {"continuous_rank_regressor", "extra_trees_ranked"}:
            score = score + 1e-10 * self.causal_tie_break(values)
        return score

    def tune_reliability_orientation(
        self, values: np.ndarray, labels: np.ndarray,
    ) -> float:
        """Orient abstention using a later calibration slice only.

        A correctness ranker can drift so that its high-score tail becomes the
        least reliable region.  Detect that on protected calibration data and
        invert selection (never prediction direction).  Weak evidence disables
        abstention for the fold instead of manufacturing a confident ranking.
        """
        if self.reliability_model is None or len(values) < 40:
            self.reliability_direction = 0.0
            return self.reliability_direction
        probability = self.base_probability(values)
        predicted = np.where(probability >= .5, 1, -1).astype(np.int8)
        correctness = (predicted == np.asarray(labels, dtype=np.int8)).astype(float)
        design = self.reliability_design(values, probability)
        normalized = (design - self.reliability_mean) / self.reliability_scale
        score = self.reliability_model.predict_proba(normalized)[:, 1]
        lower, upper = np.quantile(score, (.25, .75))
        low = correctness[score <= lower]
        high = correctness[score >= upper]
        if not len(low) or not len(high):
            self.reliability_direction = 0.0
            return self.reliability_direction
        tail_gap = float(high.mean() - low.mean())
        self.reliability_direction = (
            1.0 if tail_gap >= .03 else -1.0 if tail_gap <= -.03 else 0.0
        )
        return self.reliability_direction

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.where(self.probability(values) >= .5, 1, -1).astype(np.int8)


@dataclass
class RegimeRegressor:
    """Observable-state router over independently fitted return specialists."""
    fallback: Any
    experts: list[Any]
    feature_index: int
    edges: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(self.fallback.predict(values), dtype=np.float64)
        buckets = np.digitize(values[:, self.feature_index], self.edges)
        for bucket, expert in enumerate(self.experts):
            mask = buckets == bucket
            if mask.any() and expert is not None:
                result[mask] = expert.predict(values[mask])
        return result


@dataclass
class DecomposedRegressor:
    """Recombine independently learned market and instrument-residual motion."""
    residual_model: Any
    market_model: Any
    market_weight: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        residual = np.asarray(self.residual_model.predict(values), dtype=np.float64)
        market = np.asarray(self.market_model.predict(values), dtype=np.float64)
        return residual + self.market_weight * market


@dataclass
class MultiscaleRegressor:
    """Blend causal return models fitted with short and long memory."""
    short_model: Any
    long_model: Any
    short_weight: float = .5
    allow_orientation: bool = False
    direction: float = 1.0

    def predict(self, values: np.ndarray) -> np.ndarray:
        short = np.asarray(self.short_model.predict(values), dtype=np.float64)
        long = np.asarray(self.long_model.predict(values), dtype=np.float64)
        return self.direction * (
            self.short_weight * short + (1.0 - self.short_weight) * long
        )

    def tune(self, values: np.ndarray, labels: np.ndarray) -> None:
        """Select one bounded blend on calibration data, never protected data."""
        short = np.asarray(self.short_model.predict(values), dtype=np.float64)
        long = np.asarray(self.long_model.predict(values), dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int8)
        def accuracy(weight: float, direction: float) -> float:
            prediction = np.where(
                direction * (weight * short + (1.0 - weight) * long) >= 0,
                1, -1,
            )
            return float(np.mean(prediction == labels))

        positive = max(
            (accuracy(weight, 1.0), -abs(weight - .5), weight)
            for weight in (.25, .5, .75)
        )
        self.short_weight = positive[2]
        self.direction = 1.0
        if self.allow_orientation and len(labels) >= 40:
            inverted = max(
                (accuracy(weight, -1.0), -abs(weight - .5), weight)
                for weight in (.25, .5, .75)
            )
            if inverted[0] >= .55 and inverted[0] >= positive[0] + .04:
                self.short_weight = inverted[2]
                self.direction = -1.0


def regression_probability_scale(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fit confidence temperature while preserving the score-zero boundary."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    fallback = max(float(np.median(np.abs(scores))), 1e-6)
    if len(scores) < 20 or len(np.unique(labels)) < 2 or np.allclose(scores, 0):
        return fallback
    calibrator = LogisticRegression(
        fit_intercept=False, C=1_000_000.0, solver="lbfgs", max_iter=200,
    ).fit(scores.reshape(-1, 1), (labels > 0).astype(np.int8))
    coefficient = float(calibrator.coef_[0, 0])
    return 1.0 / coefficient if coefficient > 1e-9 else fallback


def decompose_returns(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Split future return into a same-time market component and asset residual.

    These are supervised targets used only while fitting. No future aggregate is
    added to a decision row, so inference remains strictly causal.
    """
    by_time: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_time[float(row["timestamp"])].append(float(row["return"]))
    market_by_time = {
        timestamp: float(statistics.median(returns))
        for timestamp, returns in by_time.items()
    }
    market = np.asarray(
        [market_by_time[float(row["timestamp"])] for row in rows], dtype=np.float64
    )
    realized = np.asarray([row["return"] for row in rows], dtype=np.float64)
    return market, realized - market


def new_return_regressor(genome: Genome, random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=genome.learning_rate, max_iter=genome.max_iter,
        max_leaf_nodes=genome.max_leaf_nodes,
        min_samples_leaf=genome.min_samples_leaf,
        l2_regularization=genome.l2_regularization, random_state=random_state,
    )


def new_extra_trees_return_regressor(
    genome: Genome, random_state: int, *, selection_coordinates: bool = False,
) -> ExtraTreesRegressor:
    """Rank signed return magnitude with the champion's nonlinear tree shape."""
    max_iter = (
        genome.selection_max_iter if selection_coordinates else genome.max_iter
    )
    max_leaf_nodes = (
        genome.selection_max_leaf_nodes
        if selection_coordinates else genome.max_leaf_nodes
    )
    min_samples_leaf = (
        genome.selection_min_samples_leaf
        if selection_coordinates else genome.min_samples_leaf
    )
    return ExtraTreesRegressor(
        n_estimators=min(240, max(80, max_iter)),
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt", n_jobs=1, random_state=random_state,
    )


def fit_multiscale_regressor(
    genome: Genome, values: np.ndarray, target: np.ndarray,
    short_weights: np.ndarray, long_weights: np.ndarray, random_state: int,
) -> MultiscaleRegressor:
    """Fit the same phenotype at two causal recency scales."""
    return MultiscaleRegressor(
        new_return_regressor(genome, random_state).fit(
            values, target, sample_weight=short_weights
        ),
        new_return_regressor(genome, random_state + 1000).fit(
            values, target, sample_weight=long_weights
        ),
        allow_orientation=genome.calibration_orientation,
    )


def fit_regime_regressor(genome: Genome, values: np.ndarray, target: np.ndarray,
                         weights: np.ndarray, random_state: int) -> RegimeRegressor:
    """Fit bounded specialists routed only by an observable decision feature."""
    fallback = new_return_regressor(genome, random_state).fit(
        values, target, sample_weight=weights
    )
    feature_index = genome.features.index(genome.regime_feature)
    regime_values = values[:, feature_index]
    bins = max(2, genome.regime_bins)
    edges = np.unique(np.quantile(regime_values, np.arange(1, bins) / bins))
    assignments = np.digitize(regime_values, edges)
    experts = []
    for bucket in range(len(edges) + 1):
        mask = assignments == bucket
        experts.append(
            new_return_regressor(genome, random_state + 100 + bucket).fit(
                values[mask], target[mask], sample_weight=weights[mask]
            ) if int(mask.sum()) >= max(100, genome.min_samples_leaf * 3) else None
        )
    return RegimeRegressor(fallback, experts, feature_index, edges)


def fit_decomposed_regressor(genome: Genome, values: np.ndarray,
                             market_target: np.ndarray, residual_target: np.ndarray,
                             weights: np.ndarray, random_state: int) -> DecomposedRegressor:
    market_model = new_return_regressor(genome, random_state).fit(
        values, market_target, sample_weight=weights
    )
    residual_model = new_return_regressor(genome, random_state + 1000).fit(
        values, residual_target, sample_weight=weights
    )
    return DecomposedRegressor(residual_model, market_model, genome.market_weight)


def fit_regime_decomposed_regressor(
    genome: Genome, values: np.ndarray, market_target: np.ndarray,
    residual_target: np.ndarray, weights: np.ndarray, random_state: int,
) -> RegimeRegressor:
    """Route among market/residual specialists using observable context only."""
    fallback = fit_decomposed_regressor(
        genome, values, market_target, residual_target, weights, random_state
    )
    feature_index = genome.features.index(genome.regime_feature)
    regime_values = values[:, feature_index]
    bins = max(2, genome.regime_bins)
    edges = np.unique(np.quantile(regime_values, np.arange(1, bins) / bins))
    assignments = np.digitize(regime_values, edges)
    experts = []
    for bucket in range(len(edges) + 1):
        mask = assignments == bucket
        experts.append(
            fit_decomposed_regressor(
                genome, values[mask], market_target[mask], residual_target[mask],
                weights[mask], random_state + 100 + bucket,
            ) if int(mask.sum()) >= max(100, genome.min_samples_leaf * 3) else None
        )
    return RegimeRegressor(fallback, experts, feature_index, edges)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Antivirus, indexers, and dashboard readers can briefly hold a
            # Windows file handle. The durable payload remains in the temp
            # file while replacement is retried.
            time.sleep(.05 * (2 ** attempt))


def write_live_status(state_dir: Path, phase: str, generation: int, **payload: Any) -> None:
    """Publish an atomic, human-readable heartbeat for the perpetual service."""
    try:
        atomic_json(state_dir / "status.json", {
            "at": utc_now(), "phase": phase, "generation": generation,
            "available_memory_gb": available_memory_gb(), **payload,
        })
    except OSError:
        # A monitoring surface must never terminate the evolutionary owner.
        pass


def external_brain_gate_pids() -> list[int]:
    """Find neural gates that survived a controller restart on this machine."""
    if psutil is None:
        return []
    result = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "market_evolution_brain_gate.py" in command:
                result.append(int(process.info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError):
            continue
    return result


def stale_external_brain_gate_pids(
    state_dir: Path, max_age_seconds: float, *, now: float | None = None,
) -> list[int]:
    """Find only this service's neural gates that exceeded their time budget."""
    if psutil is None or max_age_seconds <= 0:
        return []
    now = time.time() if now is None else float(now)
    state_marker = str(state_dir.resolve()).lower()
    stale: list[int] = []
    for process in psutil.process_iter(("pid", "cmdline", "create_time")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if ("market_evolution_brain_gate.py" not in command
                    or state_marker not in command.lower()):
                continue
            age = now - float(process.info.get("create_time") or now)
            if age >= max_age_seconds:
                stale.append(int(process.info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError, ValueError):
            continue
    return stale


def terminate_process_tree(pid: int) -> bool:
    """Terminate one verified stale gate and descendants, never unrelated PIDs."""
    if psutil is None:
        return False
    try:
        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
        if "market_evolution_brain_gate.py" not in command:
            return False
        descendants = process.children(recursive=True)
        for child in descendants:
            child.terminate()
        process.terminate()
        _, alive = psutil.wait_procs([*descendants, process], timeout=5)
        for remaining in alive:
            remaining.kill()
        return True
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.TimeoutExpired):
        return False


def recover_stale_external_brain_gates(
    state_dir: Path, max_age_seconds: float,
) -> list[int]:
    """Release neural-validation capacity after a controller restart or hang."""
    recovered = []
    for pid in stale_external_brain_gate_pids(state_dir, max_age_seconds):
        if terminate_process_tree(pid):
            recovered.append(pid)
    return recovered


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


def claim_service(state_dir: Path) -> Path:
    """Atomically enforce one evolutionary owner for this state directory."""
    path = state_dir / "service.pid"
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


def available_memory_gb() -> float:
    return (psutil.virtual_memory().available / 1024**3 if psutil is not None else 999.0)


def wait_for_memory_floor(
    state_dir: Path,
    stop_path: Path,
    required_gb: float,
    poll_seconds: float,
    phase: str,
    *,
    generation: int | None = None,
    reclaim_after_polls: int = 4,
    reclaimer: VerifiedWorkingSetReclaimer | None = None,
    memory_reader: Any = available_memory_gb,
) -> bool:
    """Wait safely while optionally evicting one verified node working set."""
    low_polls = 0
    while not stop_path.exists():
        available = memory_reader()
        if available >= required_gb:
            return True
        low_polls += 1
        payload: dict[str, Any] = {
            "at": utc_now(), "phase": phase,
            "available_memory_gb": available,
            "required_memory_gb": required_gb,
            "consecutive_low_memory_polls": low_polls,
        }
        if generation is not None:
            payload["generation"] = generation
        atomic_json(state_dir / "status.json", payload)
        if reclaimer is not None and low_polls >= max(1, reclaim_after_polls):
            evidence = reclaimer.attempt()
            if evidence is not None:
                append_event(
                    state_dir / "events.jsonl", "memory_reclamation_attempt",
                    phase=phase, generation=generation,
                    available_memory_gb=available,
                    required_memory_gb=required_gb, **evidence,
                )
                low_polls = 0
                if evidence.get("outcome") == "trimmed":
                    # Re-sample immediately; do not impose another full poll delay.
                    continue
        time.sleep(poll_seconds)
    return False


def dataset_signature(manifest: Path, supplemental_root: Path,
                      news_path: Path | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(f"evolution-schema:{FEATURE_SCHEMA}".encode())
    paths = [manifest, *sorted((supplemental_root / "features").glob("*.json"))]
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths.extend(Path(row["path"]) for row in payload.get("selected", [])
                     if row.get("path"))
    if news_path is not None:
        paths.append(news_path)
    paths = sorted(set(paths), key=lambda path: str(path.resolve()))
    for path in paths:
        if path.is_file():
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def evaluation_signature(data_signature: str, *, folds: int, test_days: int,
                         calibration_days: int, final_days: int, horizon: int,
                         cost_bps: float) -> str:
    payload = (
        f"{data_signature}|evaluation:{EVOLUTION_SCHEMA}|folds:{folds}|"
        f"test:{test_days}|calibration:{calibration_days}|final:{final_days}|"
        f"horizon:{horizon}|cost:{cost_bps:.8g}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def refresh_genome_audit(state_dir: Path, generation: int) -> None:
    """Keep the auditable spreadsheet current as the search runs.

    The event log is the system of record; this is the reviewable view of it.
    Rebuilding is cheap relative to a generation and never blocks evolution --
    any failure is swallowed, because losing the audit view must not stop the
    search. Cadence is env-tunable for very fast generations.
    """
    every = 1
    try:
        every = max(1, int(os.getenv("GENOME_AUDIT_EVERY_GENERATIONS", "1")))
    except (TypeError, ValueError):
        every = 1
    if generation % every:
        return
    try:
        from scripts.export_genome_audit import build

        build(state_dir, state_dir / "genome_audit.xlsx", OBJECTIVE_PROFIT_FACTOR)
    except Exception:
        pass


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), "event": event, **payload},
                                separators=(",", ":"), allow_nan=False) + "\n")


def record_accuracy_improvement(
    state_dir: Path,
    data_signature: str,
    source: str,
    accuracy: float,
    *,
    generation: int,
    genome_id: str | None,
    metrics: dict[str, Any] | None = None,
) -> bool:
    """Persist and log only a new comparable prediction-accuracy high.

    Accuracy is keyed by the exact evaluation/corpus signature and evidence
    source.  This prevents a changed corpus or a surrogate score from being
    compared with isolated Wizard-brain evidence.  The first observation is a
    silent baseline; every subsequent strict increase gets one JSONL record.
    """
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        return False
    key = f"{data_signature}:{source}"
    state_path = state_dir / "accuracy_best.json"
    try:
        state = (json.loads(state_path.read_text(encoding="utf-8"))
                 if state_path.is_file() else {})
    except (OSError, ValueError, TypeError):
        state = {}
    best = state.setdefault("best", {})
    prior_entry = best.get(key)
    prior = (float(prior_entry.get("accuracy"))
             if isinstance(prior_entry, dict) and prior_entry.get("accuracy") is not None
             else None)
    if prior is not None and accuracy <= prior + 1e-12:
        return False
    entry = {
        "accuracy": accuracy, "at": utc_now(), "generation": int(generation),
        "genome_id": genome_id, "source": source,
        "dataset_signature": data_signature,
    }
    best[key] = entry
    state.update({"schema": 1, "updated_at": entry["at"]})
    atomic_json(state_path, state)
    if prior is None:
        return False
    append_event(
        state_dir / "accuracy_improvements.jsonl", "accuracy_increased",
        source=source, dataset_signature=data_signature,
        generation=int(generation), genome_id=genome_id,
        previous_accuracy=prior, accuracy=accuracy, delta=accuracy - prior,
        metrics=metrics or {},
    )
    return True


def brain_accuracy_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return weakest-section metrics from an isolated Wizard gate report."""
    sections = [
        section.get("metrics", {})
        for fold in (report or {}).get("folds", [])
        for section in fold.get("sections", {}).values()
        if isinstance(section.get("metrics"), dict)
    ]
    if not sections:
        return None
    return {
        "min_accuracy": min(float(row.get("directional_accuracy", 0)) for row in sections),
        "min_balanced_accuracy": min(
            float(row.get("directional_balanced_accuracy", 0)) for row in sections
        ),
        "min_mcc": min(float(row.get("mcc", -1)) for row in sections),
        "min_profit_factor": min(float(row.get("profit_factor") or 0) for row in sections),
        "max_ece": max(float(row.get("ece", 1)) for row in sections),
        "sections": len(sections),
    }


def brain_gate_obligation_viable(state_dir: Path, genome_id: str) -> bool:
    """Reject only conclusive multi-fold anti-signals before neural gating."""
    path = state_dir / "candidates" / f"{genome_id}.json"
    if not path.is_file():
        return True
    try:
        candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return True
    result = candidate.result or {}
    if int(result.get("evaluated_folds", 0)) < 2:
        return True
    summary = result.get("summary", {})
    conclusive_anti_signal = (
        float(summary.get("min_accuracy", 1)) < .50
        and float(summary.get("min_mcc", 1)) < 0
        and float(summary.get("min_expectancy", 1)) <= 0
        and float(summary.get("min_profit_factor", 1)) < .90
    )
    return not conclusive_anti_signal


def brain_gate_attempt_count(events_path: Path, genome_id: str) -> int:
    """Count durable launches for one isolated neural hypothesis."""
    if not events_path.is_file():
        return 0
    attempts = 0
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if (event.get("event") == "brain_gate_started"
                        and event.get("genome_id") == genome_id):
                    attempts += 1
    except OSError:
        return attempts
    return attempts


def brain_gate_retry_exhausted(
    state_dir: Path, events_path: Path, genome_id: str, max_attempts: int = 2,
) -> bool:
    """Stop deterministic report-less gates from monopolizing evolution."""
    report = state_dir / "brain-gate-reports" / f"{genome_id}.smoke.json"
    return (not report.is_file()
            and brain_gate_attempt_count(events_path, genome_id) >= max_attempts)


def recover_pending_gate(
    state_dir: Path, champion: Genome | None, pending: str | None,
    *, prefer_champion: bool = False,
) -> str | None:
    """Restore a neural-validation obligation that was not yet recorded.

    A current ungated champion is stronger evidence than a stale queued
    hypothesis.  ``prefer_champion`` makes it the next obligation without
    disturbing an already-running isolated gate.
    """
    if champion is None:
        return pending
    report = state_dir / "brain-gate-reports" / f"{champion.genome_id}.smoke.json"
    if prefer_champion and not report.is_file():
        return champion.genome_id
    if pending:
        return pending
    return None if report.is_file() else champion.genome_id


def invalidate_population_for_new_evidence(
    population: list[Genome], generation: int, rng: random.Random,
) -> list[Genome]:
    """Clear evidence-dependent scores while preserving heritable lineages.

    Fitness and neural tie-break scores describe one exact causal corpus.  A
    growing market/news corpus therefore has to re-evaluate every survivor;
    retaining the genome is useful heredity, but retaining its old score would
    compare unlike experiments and could promote a stale winner.
    """
    for genome in population:
        genome.fitness = None
        genome.result = None
    population = introduce_calibration_variants(population, generation)
    return introduce_missing_learner_species(population, generation, rng)


def restore_completed_candidates(
    population: list[Genome], state_dir: Path, evaluation_id: str,
    *, legacy_after: float | None = None,
) -> tuple[list[Genome], list[str]]:
    """Hydrate exact genomes or identical phenotypes in one evaluation scope."""
    restored: list[str] = []
    phenotype_evidence: dict[str, Genome] = {}
    for evidence_path in (state_dir / "candidates").glob("*.json"):
        try:
            evidence = genome_from_dict(json.loads(
                evidence_path.read_text(encoding="utf-8")
            ))
        except (OSError, ValueError, TypeError):
            continue
        retired_reliability = (
            evidence.calibration_reliability
            and evidence.calibration_reliability_version <= 0
        )
        if (evidence.fitness is not None and not retired_reliability
                and (evidence.result or {}).get("evaluation_signature") == evaluation_id):
            phenotype_evidence[genome_evaluation_key(evidence)] = evidence
    for index, genome in enumerate(population):
        if genome.fitness is not None:
            continue
        path = state_dir / "candidates" / f"{genome.genome_id}.json"
        candidate: Genome | None = None
        legacy_match = False
        if path.is_file():
            try:
                direct = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                direct = None
            if direct is not None:
                result = direct.result or {}
                signed_match = result.get("evaluation_signature") == evaluation_id
                legacy_match = (
                    result.get("evaluation_signature") is None
                    and legacy_after is not None
                    and path.stat().st_mtime > legacy_after
                )
                retired_reliability = (
                    direct.calibration_reliability
                    and direct.calibration_reliability_version <= 0
                )
                if (direct.genome_id == genome.genome_id
                        and direct.fitness is not None
                        and not retired_reliability
                        and (signed_match or legacy_match)):
                    candidate = direct
        if candidate is None:
            evidence = phenotype_evidence.get(genome_evaluation_key(genome))
            if evidence is not None:
                candidate = genome_from_dict(asdict(genome))
                candidate.fitness = evidence.fitness
                candidate.result = json.loads(json.dumps(evidence.result))
        if candidate is None:
            continue
        if legacy_match:
            candidate.result["evaluation_signature"] = evaluation_id
        atomic_json(path, asdict(candidate))
        population[index] = candidate
        restored.append(genome.genome_id)
    return population, restored


def genome_evaluation_key(genome: Genome) -> str:
    """Hash every predictive/evaluation gene while ignoring lineage metadata."""
    payload = asdict(genome)
    for key in ("genome_id", "fitness", "result", "generation", "parents"):
        payload.pop(key, None)
    learner = str(payload.get("learner_kind", "classifier"))
    if learner in {
        "extra_trees", "extra_trees_ranked", "extra_trees_regressor",
        "extra_trees_hybrid",
    }:
        # These learners use max_iter only as a bounded tree count. Their
        # learning-rate and L2 genes never reach either fitted estimator.
        payload["learning_rate"] = 0.0
        payload["l2_regularization"] = 0.0
        payload["max_iter"] = min(240, max(80, int(payload["max_iter"])))
    if learner == "extra_trees_hybrid":
        payload["selection_max_iter"] = min(
            240, max(80, int(payload["selection_max_iter"]))
        )
    else:
        payload["selection_max_iter"] = int(payload["max_iter"])
        payload["selection_max_leaf_nodes"] = int(payload["max_leaf_nodes"])
        payload["selection_min_samples_leaf"] = int(payload["min_samples_leaf"])
        payload["selection_recency_half_life_days"] = float(
            payload["recency_half_life_days"]
        )
    if learner not in {"decomposed_regressor", "regime_decomposed_regressor"}:
        payload["market_weight"] = 1.0
    if learner not in {"regime_regressor", "regime_decomposed_regressor"}:
        payload["regime_feature"] = "rv24"
        payload["regime_bins"] = 1
    else:
        # Both regime fitters enforce at least two buckets.
        payload["regime_bins"] = max(2, int(payload["regime_bins"]))
    if learner != "multiscale_regressor":
        payload["calibration_orientation"] = False
    if int(payload.get("calibration_reliability_version", 0)) <= 0:
        # Version zero is deliberately inactive. Canonicalize it to the
        # ordinary phenotype, while restore_completed_candidates excludes old
        # v0 result files produced by the retired clamped implementation.
        payload["calibration_reliability"] = False
        payload["calibration_reliability_version"] = 0
        payload["calibration_reliability_pool"] = "core"
        payload["calibration_reliability_decay"] = 0.0
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reliability_evidence_rank(candidate: Genome) -> tuple[float, ...]:
    """Compare reliability experiments on accuracy, balance, and economics."""
    summary = (candidate.result or {}).get("summary", {})
    accuracy = float(summary.get("min_accuracy", 0))
    balanced = float(summary.get("min_balanced_accuracy", 0))
    mcc = float(summary.get("min_mcc", -1))
    expectancy = float(summary.get("min_expectancy", -1))
    profit = float(summary.get("min_profit_factor", 0))
    composite = (
        accuracy + balanced + .25 * mcc
        + min(.02, expectancy) + .02 * min(2.0, profit)
    )
    return composite, balanced, mcc, accuracy, expectancy, profit


def next_reliability_decays(evidence: Sequence[Genome]) -> tuple[float, float]:
    """Bracket the best protected decay without reusing tested values."""
    tested: dict[float, Genome] = {}
    for candidate in evidence:
        if (candidate.calibration_reliability_version != 6
                or candidate.fitness is None):
            continue
        tested[round(candidate.calibration_reliability_decay, 6)] = candidate
    initial = (.75, 3.5, .25, 5.0, 1.5, 7.0)
    if len(tested) < 3:
        remaining = [value for value in initial if round(value, 6) not in tested]
        return tuple(remaining[:2])  # type: ignore[return-value]

    best_decay = max(tested, key=lambda value: reliability_evidence_rank(tested[value]))
    lower = max((value for value in tested if value < best_decay), default=.1)
    upper = min((value for value in tested if value > best_decay), default=8.0)
    proposals = [round((lower + best_decay) / 2.0, 6),
                 round((best_decay + upper) / 2.0, 6)]
    proposals = [value for value in proposals if value not in tested]
    for value in initial:
        if value not in tested and value not in proposals:
            proposals.append(value)
    while len(proposals) < 2:
        anchors = sorted({.1, 8.0, *tested, *proposals})
        _, left, right = max(
            (right - left, left, right)
            for left, right in zip(anchors, anchors[1:])
        )
        midpoint = round((left + right) / 2.0, 6)
        if midpoint in tested or midpoint in proposals:
            break
        proposals.append(midpoint)
    return tuple(proposals[:2])  # type: ignore[return-value]


def next_reliability_quantiles(
    evidence: Sequence[Genome], base_quantile: float,
) -> tuple[float, float]:
    """Evolve abstention only after decay has enough protected evidence."""
    tested = {
        round(candidate.confidence_quantile, 6): candidate
        for candidate in evidence
        if candidate.calibration_reliability_version == 7
        and candidate.fitness is not None
    }
    schedule = list(dict.fromkeys(
        round(min(.30, base_quantile + offset), 6)
        for offset in (.02, .04, .06)
    ))
    remaining = [value for value in schedule if value not in tested]
    if len(tested) < 3:
        while len(remaining) < 2:
            midpoint = round((base_quantile + .30) / 2.0, 6)
            if midpoint in tested or midpoint in remaining:
                break
            remaining.append(midpoint)
        return tuple(remaining[:2])  # type: ignore[return-value]
    best = max(tested, key=lambda value: reliability_evidence_rank(tested[value]))
    lower = max((value for value in tested if value < best), default=base_quantile)
    upper = min((value for value in tested if value > best), default=.30)
    proposals = [round((lower + best) / 2.0, 6),
                 round((best + upper) / 2.0, 6)]
    proposals = [value for value in proposals if value not in tested]
    for value in schedule:
        if value not in tested and value not in proposals:
            proposals.append(value)
    return tuple(proposals[:2])  # type: ignore[return-value]


def next_oriented_reliability_variants(
    evidence: Sequence[Genome], base_quantile: float,
) -> tuple[tuple[str, float], tuple[str, float]]:
    """Search orientation-aware pool/coverage pairs without repeating failures."""
    tested = {
        (candidate.calibration_reliability_pool,
         round(candidate.confidence_quantile, 6)): candidate
        for candidate in evidence
        if candidate.calibration_reliability_version == 8
        and candidate.fitness is not None
    }
    inert_pools = {
        candidate.calibration_reliability_pool
        for candidate in evidence
        if candidate.calibration_reliability_version == 8
        and candidate.fitness is not None
        and (candidate.result or {}).get("folds")
        and all(
            float(fold.get("multiscale_calibration", {}).get(
                "reliability_direction", 1.0
            )) == 0.0
            for fold in (candidate.result or {}).get("folds", [])
        )
    }
    outcome_by_pool: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    tested_by_pool: Counter[str] = Counter()
    for candidate in evidence:
        if (candidate.calibration_reliability_version != 8
                or candidate.fitness is None):
            continue
        summary = (candidate.result or {}).get("summary", {})
        tested_by_pool[candidate.calibration_reliability_pool] += 1
        outcome_by_pool[candidate.calibration_reliability_pool].add(tuple(
            round(float(summary.get(name, fallback)), 8)
            for name, fallback in (
                ("min_accuracy", 0), ("min_balanced_accuracy", 0),
                ("min_mcc", -1), ("min_coverage", 0),
                ("min_expectancy", -1), ("min_profit_factor", 0),
            )
        ))
    plateau_pools = {
        pool for pool, count in tested_by_pool.items()
        if count >= 6 and len(outcome_by_pool[pool]) <= max(1, count // 12)
    }
    primary_pools = ("flow_news", "combined")
    compact_pools = ("flow_derivatives", "news_regime", "relative_trend")
    pool_schedule = primary_pools + compact_pools
    primary_active = tuple(
        pool for pool in primary_pools
        if pool not in inert_pools and pool not in plateau_pools
    )
    active_pools = primary_active or tuple(
        pool for pool in compact_pools
        if pool not in inert_pools and pool not in plateau_pools
    )
    if not active_pools:
        # Every current representation has become uninformative. Revisit the
        # least-sampled one rather than bisection-looping inside a plateau.
        active_pools = (min(
            pool_schedule, key=lambda pool: (tested_by_pool[pool], pool)
        ),)
    quantiles = list(dict.fromkeys(
        round(min(.30, max(.05, base_quantile + offset)), 6)
        for offset in (0.0, .02, -.02, .04, -.04, .06)
    ))
    schedule = [
        (pool, quantile)
        for quantile in quantiles
        for pool in active_pools
    ]
    remaining = [variant for variant in schedule if variant not in tested]
    if len(remaining) >= 2:
        return remaining[0], remaining[1]

    # Once coarse pairs are exhausted, refine the strongest pool locally.
    if tested:
        active_tested = {
            key: candidate for key, candidate in tested.items()
            if key[0] in active_pools
        }
        best_pool, best_quantile = max(
            active_tested,
            key=lambda key: reliability_evidence_rank(active_tested[key]),
        )
        pool_values = sorted(
            quantile for pool, quantile in tested if pool == best_pool
        )
        anchors = sorted({.05, .30, *pool_values})
        gaps = sorted(
            ((right - left, left, right)
             for left, right in zip(anchors, anchors[1:])), reverse=True,
        )
        local_gaps = sorted(
            gaps,
            key=lambda gap: (min(abs(gap[1] - best_quantile),
                                abs(gap[2] - best_quantile)), -gap[0]),
        )
        for _, left, right in local_gaps:
            proposal = (best_pool, round((left + right) / 2.0, 6))
            if proposal not in tested and proposal not in remaining:
                remaining.append(proposal)
            if len(remaining) >= 2:
                break
    while len(remaining) < 2:
        # Resolution exhaustion is practically unreachable, but preserve a
        # two-child scheduler contract without changing predictive semantics.
        fallback = (active_pools[len(remaining) % len(active_pools)], .05)
        remaining.append(fallback)
    return remaining[0], remaining[1]


def load_direct_descendant_evidence(
    state_dir: Path, parent_ids: set[str], evaluation_id: str,
) -> list[Genome]:
    """Recover signed direct-descendant failures for adaptive search memory."""
    if not parent_ids:
        return []
    evidence: dict[str, Genome] = {}
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if (candidate.fitness is None
                or (candidate.result or {}).get("evaluation_signature") != evaluation_id
                or not (set(candidate.parents) & parent_ids)):
            continue
        evidence[candidate.genome_id] = candidate
    return list(evidence.values())


def load_nearby_program_transfer_evidence(
    state_dir: Path, champion: Genome, evaluation_id: str,
) -> list[Genome]:
    """Recover one-program trials across harmless nearby champion handoffs."""
    champion_programs = {
        program_name(program) for program in champion.feature_programs
    }
    evidence: dict[str, Genome] = {}
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        candidate_programs = {
            program_name(program) for program in candidate.feature_programs
        }
        if (candidate.fitness is None
                or (candidate.result or {}).get("evaluation_signature") != evaluation_id
                or candidate_programs - champion_programs == set()
                or len(candidate_programs - champion_programs) != 1
                or champion_programs - candidate_programs
                or abs(candidate.confidence_quantile
                       - champion.confidence_quantile) > .031):
            continue
        payload = asdict(candidate)
        payload.update({
            "feature_programs": champion.feature_programs,
            "confidence_quantile": champion.confidence_quantile,
            "generation": champion.generation, "parents": champion.parents,
            "fitness": champion.fitness, "result": champion.result,
            "genome_id": champion.genome_id,
        })
        reconstructed = Genome(**payload).finalize()
        if genome_evaluation_key(reconstructed) != genome_evaluation_key(champion):
            continue
        evidence[candidate.genome_id] = candidate
    return list(evidence.values())


def load_nearby_return_tree_evidence(
    state_dir: Path, champion: Genome, evaluation_id: str,
) -> list[Genome]:
    """Keep compatible curves plus strong independent return-tree lineages."""
    evidence: dict[str, Genome] = {}
    signed_candidates: list[Genome] = []
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if (candidate.fitness is None
                or candidate.learner_kind not in {
                    "extra_trees_regressor", "extra_trees_hybrid",
                }
                or (candidate.result or {}).get("evaluation_signature") != evaluation_id):
            continue
        signed_candidates.append(candidate)
    frontier_structures = {
        return_tree_threshold_key(candidate)
        for candidate in signed_candidates
        if profitable_return_tree_coverage_frontier(candidate)
    }
    for candidate in signed_candidates:
        payload = asdict(candidate)
        payload.update({
            "learner_kind": champion.learner_kind,
            "confidence_quantile": champion.confidence_quantile,
            "min_samples_leaf": champion.min_samples_leaf,
            "max_leaf_nodes": champion.max_leaf_nodes,
            "recency_half_life_days": champion.recency_half_life_days,
            "generation": champion.generation, "parents": champion.parents,
            "fitness": champion.fitness, "result": champion.result,
            "genome_id": champion.genome_id,
        })
        reconstructed = Genome(**payload).finalize()
        independent_frontier_curve = (
            return_tree_threshold_key(candidate) in frontier_structures
        )
        if (genome_evaluation_key(reconstructed) != genome_evaluation_key(champion)
                and not independent_frontier_curve):
            continue
        evidence[candidate.genome_id] = candidate
    # Coordinate experiments intentionally change structure, so their failure
    # cannot be recovered by a same-topology key. Retain the signed descendant
    # closure of every compatible/frontier curve; this is bounded by the
    # candidate ledger and keeps negative branch evidence durable across
    # restarts and population turnover.
    changed = True
    while changed:
        changed = False
        retained_ids = set(evidence)
        for candidate in signed_candidates:
            if (candidate.genome_id not in evidence
                    and set(candidate.parents) & retained_ids):
                evidence[candidate.genome_id] = candidate
                changed = True
    return list(evidence.values())


def profitable_return_tree_coverage_frontier(genome: Genome) -> bool:
    """Return whether a signed return ranker warrants finite cutoff repair."""
    summary = (genome.result or {}).get("summary") or {}
    return (
        genome.learner_kind == "extra_trees_regressor"
        and genome.fitness is not None
        and float(summary.get("min_accuracy", 0)) >= .60
        and float(summary.get("min_balanced_accuracy", 0)) >= .60
        and float(summary.get("min_mcc", -1)) >= .20
        and float(summary.get("min_profit_factor", 0)) >= 1.05
        and float(summary.get("min_expectancy", 0)) > 0
        and .45 <= float(summary.get("min_coverage", 0)) < PRESCREEN["coverage"]
    )


def load_structure_evidence(
    state_dir: Path, frontier: Genome, evaluation_id: str,
) -> list[Genome]:
    """Recover all signed thresholds for one phenotype across ancestry hops."""
    structure = genome_structure_key(frontier)
    evidence: dict[str, Genome] = {}
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if (candidate.fitness is None
                or candidate.genome_id == frontier.genome_id
                or (candidate.result or {}).get("evaluation_signature") != evaluation_id
                or genome_structure_key(candidate) != structure):
            continue
        evidence[candidate.genome_id] = candidate
    return list(evidence.values())


def add_derived_features(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        features = row["features"]
        for name, expression in DERIVED_FEATURES.items():
            try:
                value = float(expression(features))
                features[name] = value if math.isfinite(value) else 0.0
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                features[name] = 0.0


def attach_causal_normalization(rows: Sequence[dict[str, Any]]) -> None:
    """Add asset-relative rolling state and same-time cross-sectional ranks.

    Rolling statistics use only observations strictly preceding the decision
    row. Cross-sectional ranks use values simultaneously observable at that
    decision timestamp. Both representations transfer to never-trained assets.
    """
    by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_asset[str(row["asset"])].append(row)
    for members in by_asset.values():
        members.sort(key=lambda row: float(row["timestamp"]))
        history = {
            (source, window): deque()
            for source in ROLLING_SOURCES for window in ROLLING_WINDOWS
        }
        rolling_sum = {key: 0.0 for key in history}
        rolling_square_sum = {key: 0.0 for key in history}
        for row in members:
            features = row["features"]
            for source in ROLLING_SOURCES:
                current = float(features.get(source, 0.0))
                for window in ROLLING_WINDOWS:
                    key = (source, window)
                    prior = history[key]
                    if len(prior) >= max(6, window // 3):
                        mean = rolling_sum[key] / len(prior)
                        variance = max(0.0, rolling_square_sum[key] / len(prior) - mean * mean)
                        deviation = math.sqrt(variance)
                        value = (current - mean) / max(deviation, 1e-8)
                        features[f"causal_z{window}_{source}"] = max(-12.0, min(12.0, value))
                    else:
                        features[f"causal_z{window}_{source}"] = 0.0
                    if len(prior) == window:
                        expired = prior.popleft()
                        rolling_sum[key] -= expired
                        rolling_square_sum[key] -= expired * expired
                    prior.append(current)
                    rolling_sum[key] += current
                    rolling_square_sum[key] += current * current

    by_time: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_time[float(row["timestamp"])].append(row)
    for members in by_time.values():
        for source in CROSS_SECTION_SOURCES:
            ordered = sorted(
                ((float(row["features"].get(source, 0.0)), index)
                 for index, row in enumerate(members)),
                key=lambda item: (item[0], item[1]),
            )
            denominator = max(1, len(ordered) - 1)
            for rank, (_, index) in enumerate(ordered):
                members[index]["features"][f"cross_rank_{source}"] = (
                    2.0 * rank / denominator - 1.0 if len(ordered) > 1 else 0.0
                )


def attach_news_features(rows: Sequence[dict[str, Any]], path: Path | None) -> None:
    """Attach publication-time-bounded global and asset-specific news state."""
    if path is None or not path.is_file():
        for row in rows:
            row["features"].update({name: 0.0 for name in NEWS_FEATURES})
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    articles = payload.get("articles", payload if isinstance(payload, list) else [])
    sentiment_map = {"positive": 1.0, "bullish": 1.0, "negative": -1.0,
                     "bearish": -1.0, "neutral": 0.0}
    normalized = []
    for article in articles:
        try:
            timestamp = float(article["timestamp"])
            raw = article.get("sentiment", article.get("sentiment_score", 0.0))
            sentiment = float(sentiment_map.get(str(raw).lower(), raw))
            tokens = {str(token).upper() for token in article.get("tokens", []) if token}
            headline = str(article.get("headline") or "").upper()
            source = str(article.get("source") or "")
            # Generic software advisories are not global crypto-market news.
            # Keep an advisory only when its structured tokens name a traded asset.
            if source == "GitHub Security Advisories" and not (
                tokens & {"BTC", "BITCOIN", "ETH", "ETHEREUM", "SOL", "SOLANA",
                          "LINK", "CHAINLINK", "AAVE", "UNI", "UNISWAP", "DOGE"}
            ):
                continue
            article_text = (headline + " " + str(article.get("article") or "").upper())[:12000]
            categories = {
                category for category, needles in NEWS_CATEGORIES.items()
                if any(needle in article_text for needle in needles)
            }
            normalized.append((timestamp, max(-1.0, min(1.0, sentiment)),
                               tokens, headline, categories))
        except (KeyError, TypeError, ValueError):
            continue
    normalized.sort(key=lambda item: item[0])
    times = [item[0] for item in normalized]
    aliases = {"WBTC": "BTC", "WETH": "ETH"}
    for row in rows:
        now = float(row["timestamp"])
        hi = bisect.bisect_right(times, now)
        windows = {}
        for hours in (6, 24, 72):
            lo = bisect.bisect_left(times, now - hours * 3600, 0, hi)
            selected = normalized[lo:hi]
            windows[hours] = selected
            row["features"][f"news_count_{hours}h"] = math.log1p(len(selected))
            row["features"][f"news_sentiment_{hours}h"] = (
                statistics.fmean(item[1] for item in selected) if selected else 0.0)
        selected24 = windows[24]
        row["features"]["news_polarity_24h"] = (
            statistics.fmean(1.0 if item[1] > 0 else -1.0 if item[1] < 0 else 0.0
                             for item in selected24) if selected24 else 0.0)
        row["features"]["news_negative_share_24h"] = (
            statistics.fmean(item[1] < 0 for item in selected24) if selected24 else 0.0
        )
        for category in NEWS_CATEGORIES:
            row["features"][f"news_{category}_24h"] = math.log1p(
                sum(category in item[4] for item in selected24)
            )
        asset = str(row["asset"]).upper()
        alias = aliases.get(asset, asset)
        asset_windows = {}
        for hours in (6, 24, 72):
            relevant = [item for item in windows[hours]
                        if asset in item[2] or alias in item[2]
                        or re_word(alias, item[3])]
            asset_windows[hours] = relevant
            row["features"][f"asset_news_count_{hours}h"] = math.log1p(len(relevant))
            row["features"][f"asset_news_sentiment_{hours}h"] = (
                statistics.fmean(item[1] for item in relevant) if relevant else 0.0
            )
        row["features"]["asset_news_sentiment_acceleration"] = (
            row["features"]["asset_news_sentiment_6h"]
            - row["features"]["asset_news_sentiment_72h"]
        )


def re_word(value: str, text: str) -> bool:
    """Cheap uppercase word-boundary match without compiling per article."""
    padded = " " + "".join(character if character.isalnum() else " " for character in text) + " "
    return f" {value} " in padded


def load_dataset(manifest_path: Path, supplemental_root: Path, horizon: int,
                 stride: int, seed: str, news_path: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = select_primary_assets(manifest["selected"])
    reference_record = next((row for row in records if row["base_asset"] == "WBTC"),
                            next(row for row in records if row["base_asset"] == "WETH"))
    reference = load_bars(Path(reference_record["path"]))
    reference_times = [row["timestamp"] for row in reference]
    supplemental = load_supplemental(supplemental_root, records)
    context_rows: list[dict[str, Any]] = []
    for position, record in enumerate(records, 1):
        bars = load_bars(Path(record["path"]))
        context_rows.extend(build_rows(
            record, bars, reference=reference, reference_times=reference_times,
            horizon=horizon, stride=stride,
            supplemental=supplemental.get(str(record["base_asset"])),
        ))
        print(f"evolution dataset {position}/{len(records)} {record['base_asset']}", flush=True)
    attach_market_breadth(context_rows)
    attach_news_features(context_rows, news_path)
    add_derived_features(context_rows)
    attach_causal_normalization(context_rows)
    rows = [row for row in context_rows if row["target"] != 0]
    assets = sorted({row["asset"] for row in rows}, key=lambda value: stable_order(value, seed))
    holdout_n = max(1, round(len(assets) * .25))
    asset_ends = {
        asset: max(float(row["timestamp"]) for row in rows if row["asset"] == asset)
        for asset in assets
    }
    return {
        "rows": rows,
        "training_assets": set(assets[holdout_n:]),
        "holdout_assets": set(assets[:holdout_n]),
        "supplemental_assets": set(supplemental),
        "end": min(asset_ends.values()),
        "asset_ends": asset_ends,
        "assets": assets,
    }


def load_dataset_cached(manifest_path: Path, supplemental_root: Path, horizon: int,
                        stride: int, seed: str, news_path: Path | None,
                        cache_path: Path | None) -> dict[str, Any]:
    data_signature = dataset_signature(manifest_path, supplemental_root, news_path)
    cache_signature = hashlib.sha256(
        f"{data_signature}|{horizon}|{stride}|{seed}".encode()
    ).hexdigest()[:20]
    if cache_path is not None and cache_path.is_file():
        try:
            payload = joblib.load(cache_path)
            cached = payload.get("dataset") if isinstance(payload, dict) else None
            if (payload.get("signature") == cache_signature
                    and isinstance(cached, dict) and isinstance(cached.get("rows"), list)):
                return payload["dataset"]
        except (EOFError, OSError, ValueError, TypeError, KeyError):
            pass
    dataset = load_dataset(manifest_path, supplemental_root, horizon, stride, seed, news_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        joblib.dump({"signature": cache_signature, "dataset": dataset}, temporary, compress=3)
        os.replace(temporary, cache_path)
    return dataset


def evaluation_scope(dataset: dict[str, Any], eligible: set[str], seed: str
                     ) -> tuple[set[str], set[str], float]:
    """Use the newest timestamp supported by at least 75% of eligible assets."""
    ends = sorted(float(dataset["asset_ends"][asset]) for asset in eligible
                  if asset in dataset["asset_ends"])
    if len(ends) < 4:
        raise ValueError("fewer than four assets have eligible evaluation histories")
    end = ends[max(0, math.ceil(len(ends) * .25) - 1)]
    active = sorted(
        (asset for asset in eligible if dataset["asset_ends"].get(asset, 0) >= end),
        key=lambda value: stable_order(value, seed),
    )
    holdout_n = max(1, round(len(active) * .25))
    if len(active) - holdout_n < 2:
        raise ValueError("availability filter leaves too few training assets")
    return set(active[holdout_n:]), set(active[:holdout_n]), end


def seed_genomes(population: int, rng: random.Random) -> list[Genome]:
    price = list(FEATURE_GROUPS["price"])
    flow = list(FEATURE_GROUPS["flow"])
    derivatives = list(FEATURE_GROUPS["derivatives"])
    breadth = list(FEATURE_GROUPS["breadth"])
    seeds = [price, price + flow, price + flow + derivatives,
             price + flow + breadth, price + flow + derivatives + breadth,
             price + flow + derivatives + breadth]
    seed_learners = (
        "classifier", "regressor", "extra_trees", "decomposed_regressor", "regime_regressor",
        "regime_decomposed_regressor",
    )
    result = [Genome(
        features=features, learning_rate=.06, max_iter=180, max_leaf_nodes=24,
        min_samples_leaf=20, l2_regularization=1.0, confidence_quantile=.20,
        binding_threshold=3, concept_threshold=5, presentations=3,
        feature_programs=[], recency_half_life_days=720.0,
        learner_kind=seed_learners[index],
        market_weight=1.0,
    ).finalize() for index, features in enumerate(seeds[:population])]
    all_features = list(BASE_FEATURES) + list(DERIVED_FEATURES)
    while len(result) < population:
        count = rng.randint(10, min(36, len(all_features)))
        result.append(Genome(
            features=rng.sample(all_features, count),
            learning_rate=10 ** rng.uniform(math.log10(.02), math.log10(.15)),
            max_iter=rng.randint(100, 280), max_leaf_nodes=rng.randint(12, 48),
            min_samples_leaf=rng.randint(12, 60),
            l2_regularization=10 ** rng.uniform(-2, 1),
            confidence_quantile=rng.uniform(0, .30),
            binding_threshold=rng.randint(2, 7), concept_threshold=rng.randint(3, 9),
            presentations=rng.randint(2, 7),
            feature_programs=[random_program(rng) for _ in range(rng.randint(1, 5))],
            recency_half_life_days=10 ** rng.uniform(math.log10(90), math.log10(1500)),
            learner_kind=rng.choice(LEARNER_KINDS),
            market_weight=rng.uniform(0.25, 1.75),
            regime_feature=rng.choice(REGIME_FEATURES),
            regime_bins=rng.randint(1, 3),
            training_horizons=sorted({12, *rng.sample(
                list(AUXILIARY_HORIZONS), rng.randint(1, len(AUXILIARY_HORIZONS))
            )}),
            pool_thresholds={name: rng.randint(3, 8) for name in EVOLVABLE_POOL_NAMES},
            calibration_safety=10 ** rng.uniform(0, math.log10(8)),
        ).finalize())
    return result


def position_turnover(acted_rows: Sequence[dict[str, Any]],
                      predicted: np.ndarray) -> np.ndarray:
    """Per-observation share of a round-trip cost that was actually incurred.

    A round-trip fee is paid when a position is opened or reversed, not for
    every bar it is held. Walking each asset in time order, an observation
    costs a full round trip only when its direction differs from that
    asset's previous acted direction; continuing the same direction costs
    nothing extra.

    This is measured from the predictions themselves -- no assumed holding
    period. If the model flips direction every bar the result is all 1.0 and
    the charge is identical to the old per-bar model; the saving only
    appears to the extent the model genuinely holds.
    """
    turnover = np.ones(len(acted_rows), dtype=np.float64)
    order = sorted(range(len(acted_rows)),
                   key=lambda i: (str(acted_rows[i].get("asset", "")),
                                  float(acted_rows[i].get("timestamp", 0.0))))
    previous: dict[str, int] = {}
    for index in order:
        asset = str(acted_rows[index].get("asset", ""))
        direction = int(predicted[index])
        if previous.get(asset) == direction and direction != 0:
            # Position simply held: no entry, no exit, no fee.
            turnover[index] = 0.0
        previous[asset] = direction
    return turnover


def _portfolio_drawdown(selected: Sequence[dict[str, Any]], predicted: np.ndarray,
                        cost_bps: float) -> float:
    by_time: dict[float, list[float]] = {}
    cost = cost_bps / 10_000
    for row, prediction in zip(selected, predicted):
        by_time.setdefault(row["timestamp"], []).append(prediction * row["return"] - cost)
    equity = peak = drawdown = 0.0
    for timestamp in sorted(by_time):
        equity += statistics.fmean(by_time[timestamp])
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def condition_mask(rows: Sequence[dict[str, Any]], conditions: Sequence[dict[str, Any]],
                   confidence: np.ndarray | None = None) -> np.ndarray:
    """Return the union of causal, calibration-frozen competence conditions."""
    result = np.zeros(len(rows), dtype=bool)
    for condition in conditions:
        if condition["kind"] == "all":
            conjunction = np.ones(len(rows), dtype=bool)
            for clause in condition.get("clauses", []):
                conjunction &= condition_mask(rows, [clause], confidence)
            result |= conjunction
            continue
        if condition["kind"] == "confidence":
            if confidence is not None:
                result |= confidence >= float(condition["threshold"])
            continue
        values = np.asarray([
            float(row["features"].get(condition["feature"], 0.0)) for row in rows
        ])
        if condition["side"] == "high":
            result |= values >= float(condition["threshold"])
        else:
            result |= values <= float(condition["threshold"])
    return result


def competence_passes(section: dict[str, Any]) -> bool:
    """A deliberately ghost-only floor for a narrower competence envelope."""
    return (
        section.get("acted_observations", 0) >= COMPETENCE_FLOOR["observations"]
        and section.get("directional_accuracy", 0) >= COMPETENCE_FLOOR["accuracy"]
        and section.get("directional_balanced_accuracy", 0)
        >= COMPETENCE_FLOOR["balanced_accuracy"]
        and section.get("mcc", -1) >= COMPETENCE_FLOOR["mcc"]
        and section.get("net_expectancy", -1) > 0
        and (section.get("profit_factor") or 0) >= COMPETENCE_FLOOR["profit_factor"]
    )


def evaluate_slice(model: Surrogate, selected: list[dict[str, Any]],
                   genome: Genome, threshold: float, cost_bps: float,
                   conditions: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    x = np.asarray([feature_vector(row, genome) for row in selected], dtype=np.float32)
    actual = np.asarray([row["target"] for row in selected], dtype=np.int8)
    realized = np.asarray([row["return"] for row in selected], dtype=np.float64)
    probability = model.probability(x)
    confidence = model.selection_confidence(x)
    mask = confidence >= threshold
    if conditions:
        mask &= condition_mask(selected, conditions, confidence)
    if not mask.any():
        return {"observations": len(selected), "acted_observations": 0, "coverage": 0.0}
    predicted = model.predict(x)[mask]
    acted_rows = [row for row, keep in zip(selected, mask) if keep]
    result = metrics(actual[mask], predicted, probability[mask], realized[mask],
                     cost_bps, turnover=position_turnover(acted_rows, predicted))
    momentum = np.asarray([1 if row["features"]["r12"] > 0 else -1 for row in selected],
                          dtype=np.int8)[mask]
    baseline = max(float((momentum == actual[mask]).mean()),
                   float((-momentum == actual[mask]).mean()))
    result.update({
        "observations": len(selected), "acted_observations": int(mask.sum()),
        "coverage": float(mask.mean()), "best_baseline_accuracy": baseline,
        "baseline_margin": result["directional_accuracy"] - baseline,
        "max_portfolio_drawdown": _portfolio_drawdown(acted_rows, predicted, cost_bps),
    })
    return result


def discover_competence_envelope(model: Surrogate, calibration: list[dict[str, Any]],
                                 genome: Genome, threshold: float,
                                 cost_bps: float) -> dict[str, Any]:
    """Learn where a candidate is reliable using calibration evidence only.

    Conditions are simple observable predicates so the resulting explanation
    can be applied unchanged to later historical rows or a future ghost feed.
    Protected test labels never participate in discovery.
    """
    if not calibration:
        return {"conditions": [], "calibration": {}}
    x = np.asarray([feature_vector(row, genome) for row in calibration], dtype=np.float32)
    probability = model.probability(x)
    confidence = model.selection_confidence(x)
    candidates: list[dict[str, Any]] = [{
        "kind": "confidence", "label": "highest model confidence",
        "threshold": float(np.quantile(confidence, .75)),
    }]
    thresholds: dict[str, tuple[float, float]] = {}
    for feature in COMPETENCE_FEATURES:
        values = np.asarray([
            float(row["features"].get(feature, 0.0)) for row in calibration
        ], dtype=np.float64)
        if np.allclose(values, values[0]):
            continue
        low, high = np.quantile(values, (.25, .75))
        thresholds[feature] = (float(low), float(high))
        candidates.extend((
            {"kind": "feature", "feature": feature, "side": "low",
             "threshold": float(low), "label": f"low {feature}"},
            {"kind": "feature", "feature": feature, "side": "high",
             "threshold": float(high), "label": f"high {feature}"},
        ))
    # Human-readable conjunctions approximate the multi-panel setups a chart
    # reader uses. The list is deliberately pre-registered and small; mining
    # arbitrary combinations on calibration labels would invite overfitting.
    paired_setups = (
        ("volatility_ratio", "market_breadth_r6"),
        ("flow_divergence", "futures_spot_basis"),
        ("funding_rate", "futures_spot_basis"),
        ("news_polarity_24h", "market_breadth_r6"),
        ("news_polarity_24h", "volatility_ratio"),
    )
    for left, right in paired_setups:
        if left not in thresholds or right not in thresholds:
            continue
        for left_side in ("low", "high"):
            for right_side in ("low", "high"):
                left_threshold = thresholds[left][left_side == "high"]
                right_threshold = thresholds[right][right_side == "high"]
                candidates.append({
                    "kind": "all",
                    "label": f"{left_side} {left} + {right_side} {right}",
                    "clauses": [
                        {"kind": "feature", "feature": left, "side": left_side,
                         "threshold": left_threshold},
                        {"kind": "feature", "feature": right, "side": right_side,
                         "threshold": right_threshold},
                    ],
                })
    qualified = []
    for condition in candidates:
        evidence = evaluate_slice(
            model, calibration, genome, threshold, cost_bps, [condition]
        )
        if competence_passes(evidence):
            qualified.append((
                float(evidence["directional_accuracy"])
                + .25 * float(evidence["mcc"])
                + min(.02, float(evidence["net_expectancy"])),
                condition, evidence,
            ))
    qualified.sort(key=lambda item: item[0], reverse=True)
    # One rule is intentionally easier to audit and less vulnerable to a
    # multiple-comparisons union than a large hand-picked regime expression.
    if not qualified:
        return {"conditions": [], "calibration": {}}
    _, condition, evidence = qualified[0]
    return {"conditions": [condition], "calibration": evidence}


def passes_floor(section: dict[str, Any]) -> bool:
    return (
        section.get("acted_observations", 0) >= FLOOR["observations"]
        and section.get("coverage", 0) >= FLOOR["coverage"]
        and section.get("directional_accuracy", 0) >= FLOOR["accuracy"]
        and section.get("directional_balanced_accuracy", 0) >= FLOOR["balanced_accuracy"]
        and section.get("baseline_margin", -1) >= FLOOR["baseline_margin"]
        and section.get("mcc", -1) >= FLOOR["mcc"]
        and section.get("ece", 1) <= FLOOR["ece"]
        and section.get("net_expectancy", -1) > 0
        and (section.get("profit_factor") or 0) >= FLOOR["profit_factor"]
        and section.get("max_portfolio_drawdown", 1) <= FLOOR["max_drawdown"]
    )


def passes_prescreen(section: dict[str, Any]) -> bool:
    """Cheap lower bound that only decides whether later folds are worth fitting."""
    return (
        section.get("acted_observations", 0) >= PRESCREEN["observations"]
        and section.get("coverage", 0) >= PRESCREEN["coverage"]
        and section.get("directional_accuracy", 0) >= PRESCREEN["accuracy"]
        and section.get("directional_balanced_accuracy", 0) >= PRESCREEN["balanced_accuracy"]
        and section.get("mcc", -1) >= PRESCREEN["mcc"]
        and section.get("ece", 1) <= PRESCREEN["ece"]
        and (section.get("profit_factor") or 0) >= PRESCREEN["profit_factor"]
    )


def curriculum_fitness(
    *, fold_count: int, min_accuracy: float, min_balanced: float,
    min_mcc: float, min_margin: float, min_coverage: float,
    min_observations: int, min_expectancy: float, min_profit: float,
    max_ece: float, max_drawdown: float, conditional_ghost_pass: bool,
    conditional_ghost_accuracy: float,
) -> float:
    """Reward accuracy only when it remains useful and statistically supported.

    The previous objective made a five-point accuracy gain worth far more than
    losing a quarter of actionable observations. This gate-shaped curriculum
    retains accuracy pressure while making coverage collapse and negative
    economics expensive enough to breed targeted repairs instead of winners.
    """
    # PROFIT IS THE OBJECTIVE.
    #
    # The previous weighting made accuracy worth 20x profit (500 * accuracy
    # vs 25 * profit_factor), so the search optimised for being right rather
    # than for making money -- and it worked: the champion sits at 55.6%
    # accuracy with PF 0.981, i.e. accurate and unprofitable. Profit now
    # dominates, and because a fold_count term worth 1000 would still swamp
    # it, fold completeness is folded into profit instead: an unmeasured
    # candidate cannot outrank a measured profitable one.
    #
    # Accuracy/MCC/margin remain as small tie-breakers only. They correlate
    # with durable edges, so they still steer breeding, but they can no
    # longer outvote the money.
    profit_edge = min_profit - 1.0
    # Profit is only believable in proportion to how much walk-forward it
    # survived. A fold-1 PF of 1.377 that collapses to 0.936 on three folds
    # (measured 2026-08-17, 12 of 12 candidates) must never outrank a
    # smaller profit that held up. Discounting the reward by evidence depth
    # makes an unmeasured claim structurally unable to win.
    evidence = min(1.0, max(0, fold_count) / 3.0)
    return (
        # Dominant term: real profit above break-even, discounted by how
        # thoroughly it was actually measured.
        4000 * max(-1.0, min(1.0, profit_edge)) * evidence
        # Losing money is punished at full weight regardless of evidence
        # depth -- we never need more folds to reject a loser.
        - 6000 * max(0, -profit_edge)
        + 3000 * min(0.01, max(-0.01, min_expectancy)) * 100 * evidence
        # Completing the walk-forward is itself rewarded, so a candidate can
        # never gain by being abandoned early.
        + 600 * (fold_count - 1)
        # Tie-breakers.
        + 60 * min_accuracy
        + 40 * min_balanced
        + 40 * min_mcc
        + 30 * min_margin
        - 25 * max_ece
        - 15 * max_drawdown
        # Coverage is a *soft* preference now, not a wall: more profitable
        # trades is better than fewer, but a high-coverage money-loser is
        # still worse than a selective winner.
        - 60 * max(0, PRESCREEN["coverage"] - min_coverage)
        - .25 * max(0, PRESCREEN["observations"] - min_observations)
        - 100 * max(0, -min_margin)
        + (25 * conditional_ghost_accuracy if conditional_ghost_pass else 0)
    )


def evaluate_genome(genome: Genome, dataset: dict[str, Any], *, folds: int,
                    test_days: int, calibration_days: int, final_days: int,
                    horizon: int, cost_bps: float) -> Genome:
    started = time.perf_counter()
    rows = dataset["rows"]
    uses_derivatives = genome_uses_derivatives(genome)
    eligible = dataset["supplemental_assets"] if uses_derivatives else set(dataset["assets"])
    training_assets, holdout_assets, evaluation_end = evaluation_scope(
        dataset, set(eligible), "market-perpetual-v1"
    )
    test_seconds = test_days * 86400
    calibration_seconds = calibration_days * 86400
    final_seconds = final_days * 86400
    cutoffs = [evaluation_end - final_seconds - (folds - fold) * test_seconds
               for fold in range(folds)]
    fold_results = []
    try:
        for fold, cutoff in enumerate(cutoffs):
            fit_stop = cutoff - horizon * 3600 - calibration_seconds
            calibration_start = fit_stop + horizon * 3600
            fit = [row for row in rows if row["asset"] in training_assets
                   and row["asset"] in eligible and row["timestamp"] < fit_stop]
            calibration = [row for row in rows if row["asset"] in training_assets
                           and row["asset"] in eligible
                           and calibration_start <= row["timestamp"] < cutoff - horizon * 3600]
            known = [row for row in rows if row["asset"] in training_assets
                     and row["asset"] in eligible
                     and cutoff <= row["timestamp"] < cutoff + test_seconds]
            unseen = [row for row in rows if row["asset"] in holdout_assets
                      and row["asset"] in eligible
                      and cutoff <= row["timestamp"] < cutoff + test_seconds]
            if min(len(fit), len(calibration), len(known), len(unseen)) == 0:
                raise ValueError("one or more protected split sections are empty")
            x_fit = np.asarray([feature_vector(row, genome) for row in fit], dtype=np.float32)
            y_fit = np.asarray([row["target"] for row in fit], dtype=np.int8)
            asset_counts = Counter(str(row["asset"]) for row in fit)
            class_counts = Counter(int(row["target"]) for row in fit)
            def causal_weights(half_life_days: float) -> np.ndarray:
                half_life_seconds = max(1.0, half_life_days * 86400)
                result = np.asarray([
                    (1.0 / asset_counts[str(row["asset"])])
                    * ((1.0 / class_counts[int(row["target"])])
                       if genome.learner_kind not in RETURN_LEARNER_KINDS else 1.0)
                    * math.exp(-math.log(2) * (fit_stop - float(row["timestamp"]))
                               / half_life_seconds)
                    for row in fit
                ], dtype=np.float64)
                result *= len(result) / max(result.sum(), 1e-12)
                return result

            weights = causal_weights(genome.recency_half_life_days)
            if genome.learner_kind in RETURN_LEARNER_KINDS:
                raw_model = new_return_regressor(genome, 1700 + fold)
                if genome.learner_kind == "extra_trees_regressor":
                    raw_model = new_extra_trees_return_regressor(
                        genome, 1700 + fold
                    ).fit(
                        x_fit,
                        np.asarray([row["return"] for row in fit], dtype=np.float64),
                        sample_weight=weights,
                    )
                    model = Surrogate(raw_model, "extra_trees_regressor")
                elif genome.learner_kind == "multiscale_regressor":
                    raw_model = fit_multiscale_regressor(
                        genome, x_fit,
                        np.asarray([row["return"] for row in fit], dtype=np.float64),
                        causal_weights(genome.recency_half_life_days / 3.0),
                        causal_weights(genome.recency_half_life_days * 3.0),
                        1700 + fold,
                    )
                    model = Surrogate(raw_model, "multiscale_regressor")
                elif genome.learner_kind == "decomposed_regressor":
                    market_target, residual_target = decompose_returns(fit)
                    decomposed = fit_decomposed_regressor(
                        genome, x_fit, market_target, residual_target, weights, 1700 + fold,
                    )
                    model = Surrogate(decomposed, "regressor")
                elif genome.learner_kind == "regime_decomposed_regressor":
                    market_target, residual_target = decompose_returns(fit)
                    raw_model = fit_regime_decomposed_regressor(
                        genome, x_fit, market_target, residual_target, weights, 1700 + fold,
                    )
                    model = Surrogate(raw_model, "regime_decomposed_regressor")
                elif genome.learner_kind == "regime_regressor":
                    raw_model = fit_regime_regressor(
                        genome, x_fit,
                        np.asarray([row["return"] for row in fit], dtype=np.float64),
                        weights, 1700 + fold,
                    )
                    model = Surrogate(raw_model, "regime_regressor")
                else:
                    raw_model = raw_model.fit(
                        x_fit, np.asarray([row["return"] for row in fit], dtype=np.float64),
                        sample_weight=weights,
                    )
                    model = Surrogate(
                        raw_model,
                        ("continuous_rank_regressor"
                         if genome.learner_kind == "continuous_rank_regressor"
                         else "regressor"),
                    )
            elif genome.learner_kind in {"extra_trees", "extra_trees_ranked"}:
                raw_model = ExtraTreesClassifier(
                    n_estimators=min(240, max(80, genome.max_iter)),
                    max_leaf_nodes=genome.max_leaf_nodes,
                    min_samples_leaf=genome.min_samples_leaf,
                    max_features="sqrt", class_weight=None, n_jobs=1,
                    random_state=1700 + fold,
                ).fit(x_fit, y_fit, sample_weight=weights)
                model = Surrogate(raw_model, genome.learner_kind)
            elif genome.learner_kind == "extra_trees_hybrid":
                direction_model = ExtraTreesClassifier(
                    n_estimators=min(240, max(80, genome.max_iter)),
                    max_leaf_nodes=genome.max_leaf_nodes,
                    min_samples_leaf=genome.min_samples_leaf,
                    max_features="sqrt", class_weight=None, n_jobs=1,
                    random_state=1700 + fold,
                ).fit(x_fit, y_fit, sample_weight=weights)
                return_model = new_extra_trees_return_regressor(
                    genome, 2700 + fold, selection_coordinates=True,
                ).fit(
                    x_fit,
                    np.asarray([row["return"] for row in fit], dtype=np.float64),
                    sample_weight=causal_weights(
                        genome.selection_recency_half_life_days
                    ),
                )
                raw_model = ExtraTreesHybridModel(direction_model, return_model)
                model = Surrogate(raw_model, "extra_trees_hybrid")
            else:
                raw_model = HistGradientBoostingClassifier(
                    learning_rate=genome.learning_rate, max_iter=genome.max_iter,
                    max_leaf_nodes=genome.max_leaf_nodes,
                    min_samples_leaf=genome.min_samples_leaf,
                    l2_regularization=genome.l2_regularization, random_state=1700 + fold,
                ).fit(x_fit, y_fit, sample_weight=weights)
                model = Surrogate(raw_model, "classifier")
            reliability_active = (
                genome.calibration_reliability
                and genome.calibration_reliability_version >= 1
                and len(calibration) >= 120
            )
            if reliability_active:
                calibration_split = max(80, int(len(calibration) * 2 / 3))
                calibration_fit = calibration[:calibration_split]
                calibration_threshold = calibration[calibration_split:]
            else:
                calibration_fit = calibration
                calibration_threshold = calibration
            x_cal_fit = np.asarray(
                [feature_vector(row, genome) for row in calibration_fit],
                dtype=np.float32,
            )
            calibration_fit_labels = np.asarray(
                [row["target"] for row in calibration_fit], dtype=np.int8
            )
            if genome.learner_kind in RETURN_LEARNER_KINDS:
                if genome.learner_kind == "multiscale_regressor":
                    raw_model.tune(x_cal_fit, calibration_fit_labels)
                calibration_scores = model.raw_score(x_cal_fit)
                model.score_scale = regression_probability_scale(
                    calibration_scores, calibration_fit_labels
                ) * genome.calibration_safety
            reliability_fitted = False
            if reliability_active:
                reliability_feature_names = RELIABILITY_FEATURE_POOLS.get(
                    genome.calibration_reliability_pool, RELIABILITY_FEATURES
                )
                reliability_indices = [
                    genome.features.index(name)
                    for name in reliability_feature_names if name in genome.features
                ]
                reliability_fitted = model.fit_reliability(
                    x_cal_fit, calibration_fit_labels, reliability_indices,
                    genome.calibration_reliability_version,
                    genome.calibration_reliability_decay,
                )
            x_cal_threshold = np.asarray(
                [feature_vector(row, genome) for row in calibration_threshold],
                dtype=np.float32,
            )
            if reliability_fitted and genome.calibration_reliability_version >= 8:
                model.tune_reliability_orientation(
                    x_cal_threshold,
                    np.asarray(
                        [row["target"] for row in calibration_threshold],
                        dtype=np.int8,
                    ),
                )
            cal_probability = model.probability(x_cal_threshold)
            cal_confidence = model.selection_confidence(x_cal_threshold)
            threshold = float(np.quantile(cal_confidence, genome.confidence_quantile))
            competence = discover_competence_envelope(
                model, calibration_threshold, genome, threshold, cost_bps
            )
            conditions = competence["conditions"]
            fold_result = {
                "fold": fold, "cutoff": cutoff, "fit_rows": len(fit),
                "calibration_rows": len(calibration_threshold),
                "calibration_fit_rows": len(calibration_fit),
                "confidence_threshold": threshold,
                "multiscale_calibration": ({
                    "orientation_enabled": genome.calibration_orientation,
                    "direction": float(raw_model.direction),
                    "short_weight": float(raw_model.short_weight),
                    "reliability_enabled": genome.calibration_reliability,
                    "reliability_version": genome.calibration_reliability_version,
                    "reliability_pool": genome.calibration_reliability_pool,
                    "reliability_decay": genome.calibration_reliability_decay,
                    "reliability_direction": model.reliability_direction,
                    "reliability_model": (
                        "recency_weighted_extra_trees"
                        if genome.calibration_reliability_version >= 5
                        else "nonlinear_extra_trees"
                        if genome.calibration_reliability_version >= 2
                        else "linear_logistic"
                    ) if reliability_fitted else None,
                    "reliability_fitted": reliability_fitted,
                } if genome.learner_kind == "multiscale_regressor" else {}),
                "selection_reliability": ({
                    "enabled": True,
                    "version": genome.calibration_reliability_version,
                    "pool": genome.calibration_reliability_pool,
                    "decay": genome.calibration_reliability_decay,
                    "direction": model.reliability_direction,
                    "model": (
                        "recency_weighted_extra_trees"
                        if genome.calibration_reliability_version >= 5
                        else "nonlinear_extra_trees"
                        if genome.calibration_reliability_version >= 2
                        else "linear_logistic"
                    ) if reliability_fitted else None,
                    "fitted": reliability_fitted,
                } if reliability_active else {}),
                "calibration_model": evaluate_slice(
                    model, calibration_threshold, genome, threshold, cost_bps
                ),
                "known_asset_future": evaluate_slice(model, known, genome, threshold, cost_bps),
                "unseen_asset_future": evaluate_slice(model, unseen, genome, threshold, cost_bps),
                "competence_envelope": competence,
                "conditional_ghost_known": evaluate_slice(
                    model, known, genome, threshold, cost_bps, conditions
                ) if conditions else {},
                "conditional_ghost_unseen": evaluate_slice(
                    model, unseen, genome, threshold, cost_bps, conditions
                ) if conditions else {},
            }
            fold_results.append(fold_result)
            broad_prescreen_pass = all(
                passes_prescreen(fold_result[name])
                for name in ("known_asset_future", "unseen_asset_future")
            )
            conditional_prescreen_pass = bool(conditions) and all(
                competence_passes(fold_result[name])
                for name in ("conditional_ghost_known", "conditional_ghost_unseen")
            )
            # PROFIT EARNS THE REMAINING FOLDS.
            #
            # Breaking here on the non-economic floors is what produced a
            # 1347-generation search built on noise: a candidate that looked
            # profitable on fold 1 was abandoned before folds 2-3 could test
            # it, and its optimistic fold-1 profit_factor was recorded as if
            # it were the verdict. Measured 2026-08-17 on the 12 best such
            # candidates: fold-1 PF 1.23-1.38 collapsed to 0.79-0.98 once the
            # remaining folds actually ran. Every one of them. The GA was
            # therefore selecting partly on single-fold luck.
            #
            # So: if a candidate is profitable so far, it does NOT get to
            # keep that number for free -- it must survive the rest of the
            # walk-forward. Cheap models that are already losing money still
            # break early, which is what keeps the search fast.
            profit_so_far = min(
                (fold_result[name].get("profit_factor") or 0.0)
                for name in ("known_asset_future", "unseen_asset_future")
            )
            profit_earns_more_folds = profit_so_far >= PROFIT_CONTINUE_FLOOR
            if (not broad_prescreen_pass and not conditional_prescreen_pass
                    and not profit_earns_more_folds):
                break
        sections = [fold_result[name] for fold_result in fold_results
                    for name in ("known_asset_future", "unseen_asset_future")]
        min_accuracy = min(section.get("directional_accuracy", 0) for section in sections)
        min_balanced = min(section.get("directional_balanced_accuracy", 0) for section in sections)
        min_mcc = min(section.get("mcc", -1) for section in sections)
        min_margin = min(section.get("baseline_margin", -1) for section in sections)
        min_coverage = min(section.get("coverage", 0) for section in sections)
        min_expectancy = min(section.get("net_expectancy", -1) for section in sections)
        min_profit = min(section.get("profit_factor") or 0 for section in sections)
        max_ece = max(section.get("ece", 1) for section in sections)
        max_drawdown = max(section.get("max_portfolio_drawdown", 1) for section in sections)
        min_observations = min(section.get("acted_observations", 0) for section in sections)
        ghost_sections = [fold_result[name] for fold_result in fold_results
                          for name in ("conditional_ghost_known", "conditional_ghost_unseen")
                          if fold_result.get(name)]
        conditional_ghost_pass = (
            len(fold_results) == folds and len(ghost_sections) == folds * 2
            and all(competence_passes(section) for section in ghost_sections)
        )
        conditional_ghost_accuracy = (
            min(section.get("directional_accuracy", 0) for section in ghost_sections)
            if ghost_sections else 0.0
        )
        evaluated_all_folds = len(fold_results) == folds
        passed = evaluated_all_folds and all(passes_floor(section) for section in sections)
        # Fitness is a curriculum coordinate, never an admission shortcut.
        # A candidate that earns another protected fold has repaired every
        # earlier regime's prescreen and must outrank a one-fold specialist.
        # Within the same stage, emphasize the directional evidence currently
        # blocking progress while retaining smaller economic/risk pressure.
        genome.fitness = curriculum_fitness(
            fold_count=len(fold_results), min_accuracy=min_accuracy,
            min_balanced=min_balanced, min_mcc=min_mcc,
            min_margin=min_margin, min_coverage=min_coverage,
            min_observations=min_observations, min_expectancy=min_expectancy,
            min_profit=min_profit, max_ece=max_ece,
            max_drawdown=max_drawdown,
            conditional_ghost_pass=conditional_ghost_pass,
            conditional_ghost_accuracy=conditional_ghost_accuracy,
        )
        working_target = (
            passed and min_accuracy >= WORKING_TARGET["accuracy"]
            and min_balanced >= WORKING_TARGET["balanced_accuracy"]
            and min_mcc >= WORKING_TARGET["mcc"]
            and min_profit >= WORKING_TARGET["profit_factor"]
        )
        genome.result = {
            "status": ("surrogate_working_target_pass" if working_target else
                       "surrogate_floor_pass" if passed else
                       "screened" if evaluated_all_folds else "prescreen_reject"),
            "uses_derivatives": uses_derivatives,
            "evaluation_end": evaluation_end,
            "training_assets": sorted(training_assets),
            "holdout_assets": sorted(holdout_assets),
            "evaluated_folds": len(fold_results),
            "requested_folds": folds,
            "folds": fold_results,
            "summary": {
                "min_accuracy": min_accuracy, "min_balanced_accuracy": min_balanced,
                "min_mcc": min_mcc, "min_baseline_margin": min_margin,
                "min_coverage": min_coverage, "min_acted_observations": min_observations,
                "min_expectancy": min_expectancy, "min_profit_factor": min_profit,
                "max_ece": max_ece, "max_drawdown": max_drawdown,
                "all_surrogate_floor_gates": passed,
                "surrogate_working_target": working_target,
                "conditional_ghost_pass": conditional_ghost_pass,
                "conditional_ghost_min_accuracy": conditional_ghost_accuracy,
                "conditional_ghost_sections": len(ghost_sections),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        genome.fitness = -1_000_000.0
        genome.result = {"status": "failed", "error": repr(exc),
                         "elapsed_seconds": time.perf_counter() - started}
    return genome


def evaluate_genome_after_memory_floor(
    genome: Genome, dataset: dict[str, Any], *, state_dir: Path,
    stop_path: Path, required_memory_gb: float, poll_seconds: float,
    generation: int, reclaim_after_polls: int,
    reclaimer: VerifiedWorkingSetReclaimer | None, guard_each_candidate: bool,
    folds: int, test_days: int, calibration_days: int, final_days: int,
    horizon: int, cost_bps: float,
) -> Genome | None:
    """Keep a single-worker batch from consuming the floor between genomes."""
    if guard_each_candidate and not wait_for_memory_floor(
        state_dir, stop_path, required_memory_gb, poll_seconds,
        "memory_wait_candidate", generation=generation,
        reclaim_after_polls=reclaim_after_polls, reclaimer=reclaimer,
    ):
        return None
    return evaluate_genome(
        genome, dataset, folds=folds, test_days=test_days,
        calibration_days=calibration_days, final_days=final_days,
        horizon=horizon, cost_bps=cost_bps,
    )


def mutate(parent: Genome, generation: int, rng: random.Random) -> Genome:
    features = set(parent.features)
    programs = [dict(program) for program in parent.feature_programs]
    universe = list(BASE_FEATURES) + list(DERIVED_FEATURES)
    pool_thresholds = dict(parent.pool_thresholds)
    emergent_pools = [dict(pool) for pool in parent.emergent_pools]
    if rng.random() < .55:
        pool_name = rng.choice(EVOLVABLE_POOL_NAMES)
        current = pool_thresholds.get(pool_name, parent.concept_threshold)
        pool_thresholds[pool_name] = min(12, max(2, current + rng.choice((-1, 1))))
    for _ in range(rng.randint(1, 4)):
        if rng.random() < .55 and len(features) < 42:
            features.add(rng.choice(universe))
        elif len(features) > 8:
            features.remove(rng.choice(sorted(features)))
    for _ in range(rng.randint(1, 3)):
        action = rng.random()
        if action < .45 and len(programs) < 10:
            programs.append(random_program(rng))
        elif action < .70 and programs:
            programs.pop(rng.randrange(len(programs)))
        elif programs:
            index = rng.randrange(len(programs))
            changed = dict(programs[index])
            field_name = rng.choice(("op", "left", "right", "scale"))
            if field_name == "op":
                changed[field_name] = rng.choice(EVOLVED_OPS)
            elif field_name in {"left", "right"}:
                changed[field_name] = rng.choice(universe)
            else:
                changed[field_name] = float(changed[field_name]) * math.exp(rng.gauss(0, .35))
            programs[index] = normalize_program(changed)
    active_sources = sorted(set(features) | {
        program_name(program) for program in programs
    })
    topology_action = rng.random()
    if topology_action < .50 and active_sources and len(emergent_pools) < MAX_EMERGENT_POOLS:
        width = min(len(active_sources), rng.choices((1, 2, 3, 4), (5, 3, 1, 1))[0])
        emergent_pools.append({
            "features": rng.sample(active_sources, width),
            "concept_threshold": rng.randint(3, 8),
        })
    elif topology_action < .68 and emergent_pools:
        emergent_pools.pop(rng.randrange(len(emergent_pools)))
    elif topology_action < .88 and emergent_pools:
        index = rng.randrange(len(emergent_pools))
        changed_pool = dict(emergent_pools[index])
        changed_pool["concept_threshold"] = (
            int(changed_pool.get("concept_threshold", 5)) + rng.choice((-1, 1))
        )
        if active_sources and rng.random() < .5:
            changed_pool["features"] = [rng.choice(active_sources)]
        emergent_pools[index] = changed_pool
    child = Genome(
        features=sorted(features),
        learning_rate=min(.20, max(.01, parent.learning_rate * math.exp(rng.gauss(0, .22)))),
        max_iter=min(360, max(80, round(parent.max_iter + rng.gauss(0, 28)))),
        max_leaf_nodes=min(72, max(8, round(parent.max_leaf_nodes + rng.gauss(0, 7)))),
        min_samples_leaf=min(100, max(8, round(parent.min_samples_leaf + rng.gauss(0, 10)))),
        l2_regularization=min(30, max(.001, parent.l2_regularization * math.exp(rng.gauss(0, .4)))),
        confidence_quantile=min(.30, max(0, parent.confidence_quantile + rng.gauss(0, .035))),
        binding_threshold=min(9, max(2, parent.binding_threshold + rng.choice((-1, 0, 0, 1)))),
        concept_threshold=min(12, max(2, parent.concept_threshold + rng.choice((-1, 0, 0, 1)))),
        presentations=min(9, max(2, parent.presentations + rng.choice((-1, 0, 1)))),
        feature_programs=programs,
        recency_half_life_days=min(2200, max(45, parent.recency_half_life_days
                                           * math.exp(rng.gauss(0, .25)))),
        selection_max_iter=parent.selection_max_iter,
        selection_max_leaf_nodes=parent.selection_max_leaf_nodes,
        selection_min_samples_leaf=parent.selection_min_samples_leaf,
        selection_recency_half_life_days=parent.selection_recency_half_life_days,
        learner_kind=(rng.choice(LEARNER_KINDS)
                      if rng.random() < .12 else parent.learner_kind),
        market_weight=min(2.5, max(0.0, parent.market_weight + rng.gauss(0, .15))),
        regime_feature=(rng.choice(REGIME_FEATURES) if rng.random() < .20
                        else parent.regime_feature),
        regime_bins=min(3, max(1, parent.regime_bins + rng.choice((-1, 0, 0, 1)))),
        training_horizons=sorted({12, *(horizon for horizon in AUXILIARY_HORIZONS
                                        if ((horizon in parent.training_horizons)
                                            != (rng.random() < .12)))}),
        pool_thresholds=pool_thresholds,
        emergent_pools=emergent_pools,
        calibration_safety=min(12.0, max(
            1.0, parent.calibration_safety * math.exp(rng.gauss(0, .25))
        )),
        generation=generation, parents=[parent.genome_id],
    )
    return child.finalize()


def crossover(left: Genome, right: Genome, generation: int, rng: random.Random) -> Genome:
    common = set(left.features) & set(right.features)
    optional = list(set(left.features) ^ set(right.features))
    features = common | {name for name in optional if rng.random() < .5}
    if len(features) < 8:
        features.update(rng.sample(list(BASE_FEATURES), 8 - len(features)))
    choose = lambda a, b: a if rng.random() < .5 else b
    program_by_name = {
        program_name(program): program
        for program in left.feature_programs + right.feature_programs
    }
    inherited_programs = [program for _, program in sorted(program_by_name.items())
                          if rng.random() < .6][:10]
    pool_by_sources = {
        tuple(pool.get("features", [])): pool
        for pool in left.emergent_pools + right.emergent_pools
    }
    inherited_pools = [
        dict(pool) for _, pool in sorted(pool_by_sources.items())
        if rng.random() < .6
    ][:MAX_EMERGENT_POOLS]
    child = Genome(
        features=sorted(features), learning_rate=choose(left.learning_rate, right.learning_rate),
        max_iter=choose(left.max_iter, right.max_iter),
        max_leaf_nodes=choose(left.max_leaf_nodes, right.max_leaf_nodes),
        min_samples_leaf=choose(left.min_samples_leaf, right.min_samples_leaf),
        l2_regularization=choose(left.l2_regularization, right.l2_regularization),
        confidence_quantile=choose(left.confidence_quantile, right.confidence_quantile),
        binding_threshold=choose(left.binding_threshold, right.binding_threshold),
        concept_threshold=choose(left.concept_threshold, right.concept_threshold),
        presentations=choose(left.presentations, right.presentations),
        feature_programs=inherited_programs,
        recency_half_life_days=choose(left.recency_half_life_days,
                                     right.recency_half_life_days),
        selection_max_iter=choose(
            left.selection_max_iter, right.selection_max_iter
        ),
        selection_max_leaf_nodes=choose(
            left.selection_max_leaf_nodes, right.selection_max_leaf_nodes
        ),
        selection_min_samples_leaf=choose(
            left.selection_min_samples_leaf, right.selection_min_samples_leaf
        ),
        selection_recency_half_life_days=choose(
            left.selection_recency_half_life_days,
            right.selection_recency_half_life_days,
        ),
        learner_kind=choose(left.learner_kind, right.learner_kind),
        market_weight=choose(left.market_weight, right.market_weight),
        regime_feature=choose(left.regime_feature, right.regime_feature),
        regime_bins=choose(left.regime_bins, right.regime_bins),
        training_horizons=sorted({12, *(horizon for horizon in AUXILIARY_HORIZONS
                                        if horizon in choose(left.training_horizons,
                                                             right.training_horizons))}),
        pool_thresholds={
            name: choose(left.pool_thresholds.get(name, left.concept_threshold),
                         right.pool_thresholds.get(name, right.concept_threshold))
            for name in EVOLVABLE_POOL_NAMES
        },
        emergent_pools=inherited_pools,
        calibration_safety=choose(left.calibration_safety, right.calibration_safety),
        generation=generation, parents=[left.genome_id, right.genome_id],
    ).finalize()
    return mutate(child, generation, rng) if rng.random() < .7 else child


def genome_from_dict(payload: dict[str, Any]) -> Genome:
    compatible = dict(payload)
    compatible.setdefault("feature_programs", [])
    compatible.setdefault("recency_half_life_days", 720.0)
    compatible.setdefault("selection_max_iter", compatible.get("max_iter", 0))
    compatible.setdefault(
        "selection_max_leaf_nodes", compatible.get("max_leaf_nodes", 0)
    )
    compatible.setdefault(
        "selection_min_samples_leaf", compatible.get("min_samples_leaf", 0)
    )
    compatible.setdefault(
        "selection_recency_half_life_days",
        compatible.get("recency_half_life_days", 0.0),
    )
    compatible.setdefault("learner_kind", "classifier")
    compatible.setdefault("market_weight", 1.0)
    compatible.setdefault("regime_feature", "rv24")
    compatible.setdefault("regime_bins", 1)
    compatible.setdefault("training_horizons", [12])
    compatible.setdefault("pool_thresholds", {})
    compatible.setdefault("emergent_pools", [])
    compatible.setdefault("calibration_safety", 1.0)
    return Genome(**compatible)


OUTCOME_POOL_LEARNERS = tuple(dict.fromkeys((
    *LEARNER_KINDS, *sorted(RETURN_LEARNER_KINDS),
    "extra_trees_hybrid", "extra_trees_ranked",
)))
OUTCOME_POOL_TARGETS = (
    "fold_survival", "accuracy", "balanced_accuracy", "mcc", "coverage",
    "profit_factor", "expectancy", "calibration", "drawdown",
)


@dataclass
class GenomeOutcomePool:
    """Live neural surrogate for reproduction only, never admission."""
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    models: list[Any]
    examples: int
    validation_mae: list[float]
    baseline_mae: list[float]
    validation_rank_correlation: list[float]
    active: bool

    def predict(self, genomes: Sequence[Genome]) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(
            [genome_outcome_vector(genome) for genome in genomes], dtype=np.float64
        )
        normalized = (values - self.feature_mean) / self.feature_scale
        predictions = np.asarray(
            [np.clip(
                model.predict(normalized) * self.target_scale + self.target_mean,
                0.0, 1.0,
            ) for model in self.models],
            dtype=np.float64,
        )
        return predictions.mean(axis=0), predictions.std(axis=0)


def _bounded(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (float(value) - low) / max(1e-12, high - low)))


def _outcome_bucket(value: str, width: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % width


def genome_outcome_vector(genome: Genome) -> np.ndarray:
    """Encode genes, feature combinations, programs, and neural topology."""
    scalars = [
        _bounded(math.log10(max(genome.learning_rate, 1e-6)), -2.0, -.5),
        _bounded(genome.max_iter, 60, 500),
        _bounded(genome.max_leaf_nodes, 8, 72),
        _bounded(genome.min_samples_leaf, 8, 100),
        _bounded(math.log10(max(genome.l2_regularization, 1e-6)), -3, 1.5),
        _bounded(genome.confidence_quantile, 0, .30),
        _bounded(genome.binding_threshold, 2, 9),
        _bounded(genome.concept_threshold, 2, 12),
        _bounded(genome.presentations, 2, 9),
        _bounded(math.log(max(genome.recency_half_life_days, 1)), math.log(45),
                 math.log(2200)),
        _bounded(genome.selection_max_iter, 80, 240),
        _bounded(genome.selection_max_leaf_nodes, 8, 72),
        _bounded(genome.selection_min_samples_leaf, 8, 100),
        _bounded(
            math.log(max(genome.selection_recency_half_life_days, 1)),
            math.log(45), math.log(2200),
        ),
        _bounded(genome.market_weight, 0, 2.5),
        _bounded(genome.regime_bins, 1, 3),
        _bounded(genome.calibration_safety, 1, 12),
        _bounded(genome.calibration_reliability_version, 0, 8),
        _bounded(genome.calibration_reliability_decay, 0, 8),
        _bounded(len(genome.features), 8, 48),
        _bounded(len(genome.feature_programs), 0, 10),
        _bounded(len(genome.emergent_pools), 0, MAX_EMERGENT_POOLS),
        float(genome.calibration_orientation),
        float(genome.calibration_reliability),
    ]
    learners = [float(genome.learner_kind == name) for name in OUTCOME_POOL_LEARNERS]
    feature_hash = np.zeros(64, dtype=np.float64)
    for name in genome.features:
        feature_hash[_outcome_bucket(f"feature:{name}", len(feature_hash))] = 1.0
    program_hash = np.zeros(32, dtype=np.float64)
    for program in genome.feature_programs:
        token = ":".join((
            str(program.get("op", "")), str(program.get("left", "")),
            str(program.get("right", "")),
        ))
        program_hash[_outcome_bucket(f"program:{token}", len(program_hash))] += .25
    program_hash = np.clip(program_hash, 0.0, 1.0)
    pool_hash = np.zeros(16, dtype=np.float64)
    for pool in genome.emergent_pools:
        for name in pool.get("features", []):
            pool_hash[_outcome_bucket(f"pool:{name}", len(pool_hash))] = 1.0
    return np.concatenate((
        np.asarray(scalars + learners, dtype=np.float64),
        feature_hash, program_hash, pool_hash,
    ))


def genome_outcome_target(genome: Genome) -> np.ndarray | None:
    result = genome.result or {}
    summary = result.get("summary") or {}
    if not summary or result.get("status") == "failed":
        return None
    requested = max(1, int(result.get("requested_folds", 3)))
    survived = int(result.get("evaluated_folds", 0)) / requested
    return np.asarray([
        _bounded(survived, 0, 1),
        _bounded(summary.get("min_accuracy", 0), 0, 1),
        _bounded(summary.get("min_balanced_accuracy", 0), 0, 1),
        _bounded(summary.get("min_mcc", -1), -1, 1),
        _bounded(summary.get("min_coverage", 0), 0, 1),
        _bounded(summary.get("min_profit_factor", 0), 0, 3),
        _bounded(summary.get("min_expectancy", -.01), -.01, .01),
        1.0 - _bounded(summary.get("max_ece", .3), 0, .3),
        1.0 - _bounded(summary.get("max_drawdown", 2), 0, 2),
    ], dtype=np.float64)


def outcome_pool_evidence(
    state_dir: Path, evaluation_id: str, limit: int = 768,
) -> list[Genome]:
    """Retain recent distribution plus rare high-quality historical outcomes."""
    candidates: dict[str, Genome] = {}
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            genome = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if ((genome.result or {}).get("evaluation_signature") != evaluation_id
                or genome_outcome_target(genome) is None):
            continue
        key = genome_evaluation_key(genome)
        prior = candidates.get(key)
        if prior is None or int((genome.result or {}).get("evaluated_folds", 0)) > int(
            (prior.result or {}).get("evaluated_folds", 0)
        ):
            candidates[key] = genome
    values = list(candidates.values())
    recent = sorted(values, key=lambda genome: (genome.generation, genome.genome_id),
                    reverse=True)[:max(1, limit * 2 // 3)]
    quality = sorted(values, key=lambda genome: (
        int((genome.result or {}).get("evaluated_folds", 0)),
        float(((genome.result or {}).get("summary") or {}).get("min_profit_factor", 0)),
        float(((genome.result or {}).get("summary") or {}).get("min_accuracy", 0)),
        float(((genome.result or {}).get("summary") or {}).get("min_coverage", 0)),
    ), reverse=True)[:max(1, limit // 3)]
    selected: dict[str, Genome] = {}
    for genome in [*recent, *quality]:
        selected[genome.genome_id] = genome
        if len(selected) >= limit:
            break
    return sorted(selected.values(), key=lambda genome: (genome.generation, genome.genome_id))


def train_genome_outcome_pool(
    evidence: Sequence[Genome], *, minimum_examples: int = 96,
) -> GenomeOutcomePool | None:
    usable = [genome for genome in evidence if genome_outcome_target(genome) is not None]
    if len(usable) < minimum_examples:
        return None
    values = np.asarray([genome_outcome_vector(genome) for genome in usable])
    targets = np.asarray([genome_outcome_target(genome) for genome in usable])
    split = max(minimum_examples // 2, min(len(usable) - 16, int(len(usable) * .8)))
    if split <= 0 or split >= len(usable):
        return None
    x_train, x_test = values[:split], values[split:]
    y_train, y_test = targets[:split], targets[split:]
    mean = x_train.mean(axis=0)
    # Rare hashed features must not explode merely because they are absent in
    # most of one chronological split.
    scale = np.maximum(x_train.std(axis=0), .1)
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    target_mean = y_train.mean(axis=0)
    target_scale = np.maximum(y_train.std(axis=0), .03)
    normalized_target = (y_train - target_mean) / target_scale
    models = []
    predictions = []
    for seed in (2718, 3141, 5772):
        model = MLPRegressor(
            hidden_layer_sizes=(24, 12), activation="tanh", solver="adam",
            alpha=.03, learning_rate_init=.0007, max_iter=180,
            early_stopping=True, validation_fraction=.15,
            n_iter_no_change=15, random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, normalized_target)
        models.append(model)
        predictions.append(np.clip(
            model.predict(x_test) * target_scale + target_mean, 0.0, 1.0
        ))
    prediction = np.mean(predictions, axis=0)
    validation_mae = np.mean(np.abs(prediction - y_test), axis=0)
    baseline = np.tile(y_train.mean(axis=0), (len(y_test), 1))
    baseline_mae = np.mean(np.abs(baseline - y_test), axis=0)
    rank_correlations = []
    for index in range(y_test.shape[1]):
        actual_rank = np.argsort(np.argsort(y_test[:, index])).astype(np.float64)
        predicted_rank = np.argsort(np.argsort(prediction[:, index])).astype(np.float64)
        correlation = (
            float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
            if np.std(actual_rank) > 0 and np.std(predicted_rank) > 0 else 0.0
        )
        rank_correlations.append(correlation if math.isfinite(correlation) else 0.0)
    key_indices = (0, 1, 4, 5)
    active = (
        float(validation_mae.mean()) < float(baseline_mae.mean())
        and float(validation_mae[list(key_indices)].mean())
        < float(baseline_mae[list(key_indices)].mean())
    )
    return GenomeOutcomePool(
        mean, scale, target_mean, target_scale, models, len(usable),
        validation_mae.tolist(),
        baseline_mae.tolist(), rank_correlations, active,
    )


def nudge_genome(parent: Genome, generation: int, rng: random.Random) -> Genome:
    """Make one bounded departure so the outcome pool can map local gradients."""
    payload = asdict(parent)
    coordinate = rng.choice((
        "confidence_quantile", "recency_half_life_days", "min_samples_leaf",
        "max_leaf_nodes", "learning_rate", "market_weight",
        "calibration_safety", "feature",
    ))
    if coordinate == "confidence_quantile":
        payload[coordinate] = min(.30, max(
            0.0, parent.confidence_quantile + rng.choice((-.01, -.005, .005, .01))
        ))
    elif coordinate == "recency_half_life_days":
        payload[coordinate] = min(2200.0, max(
            45.0, parent.recency_half_life_days * rng.choice((.8, 1.25))
        ))
    elif coordinate == "min_samples_leaf":
        payload[coordinate] = min(100, max(
            8, parent.min_samples_leaf + rng.choice((-4, 4))
        ))
    elif coordinate == "max_leaf_nodes":
        payload[coordinate] = min(72, max(
            8, parent.max_leaf_nodes + rng.choice((-4, 4))
        ))
    elif coordinate == "learning_rate":
        payload[coordinate] = min(.30, max(
            .005, parent.learning_rate * rng.choice((.8, 1.25))
        ))
    elif coordinate == "market_weight":
        payload[coordinate] = min(2.5, max(
            0.0, parent.market_weight + rng.choice((-.1, .1))
        ))
    elif coordinate == "calibration_safety":
        payload[coordinate] = min(12.0, max(
            1.0, parent.calibration_safety * rng.choice((.85, 1.15))
        ))
    else:
        features = set(parent.features)
        universe = sorted((set(BASE_FEATURES) | set(DERIVED_FEATURES)) - features)
        if universe and rng.random() < .5:
            features.add(rng.choice(universe))
        elif len(features) > 8:
            features.remove(rng.choice(sorted(features)))
        payload["features"] = sorted(features)
    payload.update({
        "generation": generation, "parents": [parent.genome_id],
        "genome_id": "", "fitness": None, "result": None,
    })
    return Genome(**payload).finalize()


def outcome_acquisition(
    mean: np.ndarray, uncertainty: np.ndarray, *, profit_discovery: bool = False,
    accuracy_discovery: bool = False,
    target_reliability: np.ndarray | None = None,
) -> float:
    fold, accuracy, balanced, mcc, coverage, profit, expectancy, calibration, drawdown = mean
    reliable = (np.ones(len(mean), dtype=np.float64)
                if target_reliability is None else np.asarray(
                    target_reliability, dtype=np.float64
                ))
    profit_reliability = float(reliable[5])
    expectancy_reliability = float(reliable[6])
    if accuracy_discovery:
        # Once the validated champion has stopped moving, let the learned
        # genome/outcome pool search deliberately for more directional signal.
        # Chronologically validated predicted economics remain constraints
        # rather than being traded away. Unreliable surrogate targets exert no
        # acquisition force; protected evaluation remains authoritative.
        return float(
            4.0 * fold + 14.0 * accuracy + 5.0 * balanced + 2.0 * mcc
            + 2.0 * coverage + 4.0 * profit_reliability * profit
            + 1.5 * expectancy_reliability * expectancy
            + .5 * calibration + .5 * drawdown
            - 8.0 * max(0.0, .60 - coverage)
            - 12.0 * profit_reliability * max(0.0, (1.0 / 3.0) - profit)
            + 1.0 * float(np.mean(uncertainty))
        )
    if profit_discovery:
        return float(
            1.0 * fold + 2.5 * accuracy + 1.0 * balanced + .75 * mcc
            + 2.0 * coverage + 20.0 * profit_reliability * profit
            + 1.5 * expectancy_reliability * expectancy
            + .25 * calibration + .25 * drawdown
            - 4.0 * max(0.0, .60 - coverage)
            - 3.0 * profit_reliability * max(0.0, 2.0 / 3.0 - profit)
            + 1.25 * float(np.mean(uncertainty))
        )
    return float(
        4.0 * fold + 3.0 * accuracy + 1.5 * balanced + 1.0 * mcc
        + 2.0 * coverage + 4.0 * profit_reliability * profit
        + 1.5 * expectancy_reliability * expectancy
        + .5 * calibration + .5 * drawdown
        - 5.0 * max(0.0, .60 - coverage)
        - 5.0 * profit_reliability * max(0.0, 2.0 / 3.0 - profit)
        + .75 * float(np.mean(uncertainty))
    )


def introduce_outcome_pool_variant(
    population: list[Genome], evaluated: Sequence[Genome],
    pool: GenomeOutcomePool | None, generation: int, rng: random.Random,
    protected_parent_ids: set[str] | None = None,
    plateau_generations: int = 0,
) -> tuple[list[Genome], dict[str, Any]]:
    report: dict[str, Any] = {
        "active": bool(pool and pool.active),
        "examples": int(pool.examples if pool else 0),
        "validation_mae": (dict(zip(OUTCOME_POOL_TARGETS, pool.validation_mae))
                           if pool else {}),
        "baseline_mae": (dict(zip(OUTCOME_POOL_TARGETS, pool.baseline_mae))
                         if pool else {}),
        "validation_rank_correlation": (dict(zip(
            OUTCOME_POOL_TARGETS, pool.validation_rank_correlation
        )) if pool else {}),
        "proposed": False,
    }
    if pool is None or not pool.active or not population or not evaluated:
        return population, report
    target_reliability = np.asarray([
        float(mae < baseline and rank >= .10)
        for mae, baseline, rank in zip(
            pool.validation_mae, pool.baseline_mae,
            pool.validation_rank_correlation,
        )
    ], dtype=np.float64)
    report["target_reliability"] = dict(zip(
        OUTCOME_POOL_TARGETS,
        (bool(value) for value in target_reliability),
    ))
    signed = [genome for genome in evaluated if genome.fitness is not None]
    full_fold_bases = sorted(signed, key=lambda genome: (
            int((genome.result or {}).get("evaluated_folds", 0)),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
        ), reverse=True)[:8]
    profit_bases = sorted(signed, key=lambda genome: (
            float(((genome.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )),
        ), reverse=True)[:8]
    accuracy_bases = sorted(signed, key=lambda genome: (
            float(((genome.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
        ), reverse=True)[:8]
    bases_by_id: dict[str, Genome] = {}
    for genome in [*full_fold_bases, *profit_bases, *accuracy_bases]:
        bases_by_id[genome.genome_id] = genome
    bases = list(bases_by_id.values())
    if not bases:
        return population, report
    known = {genome.genome_id for genome in [*population, *evaluated]}
    known_evaluation_keys = {
        genome_evaluation_key(genome) for genome in [*population, *evaluated]
    }
    proposals: dict[str, Genome] = {}
    known_phenotypes_filtered = 0
    duplicate_proposals_filtered = 0
    for index in range(144):
        base = bases[index % len(bases)] if index < len(bases) else rng.choice(bases)
        proposal = (nudge_genome(base, generation, rng) if index % 2 == 0
                    else mutate(base, generation, rng))
        evaluation_key = genome_evaluation_key(proposal)
        if (proposal.genome_id in known
                or evaluation_key in known_evaluation_keys):
            known_phenotypes_filtered += 1
            continue
        if evaluation_key in proposals:
            duplicate_proposals_filtered += 1
            continue
        proposals[evaluation_key] = proposal
    if not proposals:
        report.update({
            "known_phenotypes_filtered": known_phenotypes_filtered,
            "duplicate_proposals_filtered": duplicate_proposals_filtered,
        })
        return population, report
    candidates = list(proposals.values())
    means, uncertainties = pool.predict(candidates)
    accuracy_discovery = plateau_generations >= 24 and generation % 3 == 1
    profit_discovery_requested = not accuracy_discovery and generation % 3 == 0
    profit_index = OUTCOME_POOL_TARGETS.index("profit_factor")
    profit_discovery = (
        profit_discovery_requested and bool(target_reliability[profit_index])
    )
    ranked = sorted(
        zip(candidates, means, uncertainties),
        key=lambda item: outcome_acquisition(
            item[1], item[2], profit_discovery=profit_discovery,
            accuracy_discovery=accuracy_discovery,
            target_reliability=target_reliability,
        ), reverse=True,
    )
    protected_parent_ids = protected_parent_ids or set()
    replacement = next((
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and not (set(population[index].parents) & protected_parent_ids)
    ), None)
    replacement_scope = "unprotected"
    selected = ranked[0] if replacement is not None else None
    if selected is None:
        # Reproduction can fill every pending slot with protected-frontier
        # descendants, suppressing the adviser even when one lineage has two
        # children. Replace only within the proposal's own protected lineage
        # and only when another pending sibling survives.
        for ranked_item in ranked:
            proposal_protected_parents = (
                set(ranked_item[0].parents) & protected_parent_ids
            )
            sibling_indices = [
                index for index in range(1, len(population))
                if population[index].fitness is None
                and bool(
                    set(population[index].parents)
                    & proposal_protected_parents
                )
            ]
            if len(sibling_indices) >= 2:
                selected = ranked_item
                replacement = sibling_indices[-1]
                replacement_scope = "redundant_protected_sibling"
                break
    if replacement is None or selected is None:
        report.update({
            "known_phenotypes_filtered": known_phenotypes_filtered,
            "duplicate_proposals_filtered": duplicate_proposals_filtered,
            "replacement_scope": "none",
        })
        return population, report
    proposal, mean, uncertainty = selected
    population[replacement] = proposal
    report.update({
        "proposed": True, "genome_id": proposal.genome_id,
        "parent_ids": proposal.parents,
        "predicted": dict(zip(OUTCOME_POOL_TARGETS, mean.tolist())),
        "uncertainty": dict(zip(OUTCOME_POOL_TARGETS, uncertainty.tolist())),
        "acquisition": outcome_acquisition(
            mean, uncertainty, profit_discovery=profit_discovery,
            accuracy_discovery=accuracy_discovery,
            target_reliability=target_reliability,
        ),
        "acquisition_mode": (
            "accuracy_discovery" if accuracy_discovery else
            "profit_discovery" if profit_discovery else
            "balanced_safe_improvement"
        ),
        "plateau_generations": int(plateau_generations),
        "profit_discovery_suppressed": bool(
            profit_discovery_requested and not profit_discovery
        ),
        "candidate_search_size": len(candidates),
        "known_phenotypes_filtered": known_phenotypes_filtered,
        "duplicate_proposals_filtered": duplicate_proposals_filtered,
        "replacement_scope": replacement_scope,
    })
    return population, report


def tree_leaf_refinement_key(genome: Genome) -> str:
    """Identify one predictive phenotype while ignoring tree leaf capacity."""
    payload = asdict(genome)
    payload.update({
        "max_leaf_nodes": 8,
        "generation": 0,
        "parents": [],
        "genome_id": "",
        "fitness": None,
        "result": None,
    })
    return genome_evaluation_key(Genome(**payload).finalize())


def return_tree_threshold_key(genome: Genome) -> str:
    """Identify one canonical predictive phenotype while ignoring cutoff."""
    payload = asdict(genome)
    payload.update({
        "confidence_quantile": 0.0,
        "generation": 0,
        "parents": [],
        "genome_id": "",
        "fitness": None,
        "result": None,
    })
    return genome_evaluation_key(Genome(**payload).finalize())


def return_tree_min_leaf_key(genome: Genome) -> str:
    """Identify a return-tree cutoff while ignoring minimum leaf support."""
    payload = asdict(genome)
    payload.update({
        "min_samples_leaf": 1,
        "generation": 0,
        "parents": [],
        "genome_id": "",
        "fitness": None,
        "result": None,
    })
    return genome_evaluation_key(Genome(**payload).finalize())


def return_tree_leaf_capacity_key(genome: Genome) -> str:
    """Identify a return-tree cutoff while ignoring maximum leaf capacity."""
    payload = asdict(genome)
    payload.update({
        "max_leaf_nodes": 8,
        "generation": 0,
        "parents": [],
        "genome_id": "",
        "fitness": None,
        "result": None,
    })
    return genome_evaluation_key(Genome(**payload).finalize())


def tree_leaf_refinement_quality(genome: Genome) -> float:
    """Balance direction, economics, and coverage for a signed tree result."""
    summary = (genome.result or {}).get("summary") or {}
    return float(
        float(summary.get("min_accuracy", 0))
        + float(summary.get("min_balanced_accuracy", 0))
        + .5 * max(-1.0, float(summary.get("min_mcc", -1)))
        + .05 * min(3.0, max(0.0, float(summary.get("min_profit_factor", 0))))
        + .10 * float(summary.get("min_coverage", 0))
    )


def introduce_tree_leaf_refinement_variant(
    population: list[Genome], evidence: Sequence[Genome], generation: int,
    evaluation_id: str, protected_parent_ids: set[str] | None = None,
) -> tuple[list[Genome], dict[str, Any]]:
    """Refine a signed local tree-capacity optimum instead of walking past it.

    Tree schedules historically moved in coarse four-leaf steps. Once three
    otherwise identical current-data phenotypes bracket a strong interior
    result, further movement toward an already weaker endpoint is wasteful.
    Test the finite integer midpoints around that result, best-neighbour first,
    and stop automatically when both sides are adjacent or already signed.
    """
    report: dict[str, Any] = {
        "active": False, "proposed": False, "evaluation_id": evaluation_id,
    }
    if len(population) < 4 or not evidence or not evaluation_id:
        return population, report
    signed = [
        genome for genome in evidence
        if genome.fitness is not None
        and (genome.result or {}).get("evaluation_signature") == evaluation_id
        and genome.learner_kind in {
            "regressor", "decomposed_regressor", "regime_regressor",
            "regime_decomposed_regressor", "multiscale_regressor",
            "extra_trees", "extra_trees_ranked", "extra_trees_regressor",
            "extra_trees_hybrid",
        }
        and (genome.result or {}).get("status") != "failed"
    ]
    groups: dict[str, dict[int, Genome]] = {}
    for genome in signed:
        group = groups.setdefault(tree_leaf_refinement_key(genome), {})
        leaf = int(genome.max_leaf_nodes)
        incumbent = group.get(leaf)
        if incumbent is None or tree_leaf_refinement_quality(genome) > (
            tree_leaf_refinement_quality(incumbent)
        ):
            group[leaf] = genome
    known_keys = {
        genome_evaluation_key(genome) for genome in [*population, *signed]
    }
    refinements: list[tuple[float, Genome, Genome, Genome, list[int]]] = []
    for by_leaf in groups.values():
        ordered = sorted(by_leaf.items())
        if len(ordered) < 3:
            continue
        for index in range(1, len(ordered) - 1):
            _, lower = ordered[index - 1]
            center_leaf, center = ordered[index]
            _, upper = ordered[index + 1]
            summary = (center.result or {}).get("summary") or {}
            if not (
                float(summary.get("min_accuracy", 0)) >= .58
                and float(summary.get("min_balanced_accuracy", 0)) >= .58
                and float(summary.get("min_mcc", -1)) >= .15
                and float(summary.get("min_coverage", 0)) >= .50
                and float(summary.get("min_expectancy", 0)) > 0
                and float(summary.get("min_profit_factor", 0)) >= 1.0
            ):
                continue
            center_quality = tree_leaf_refinement_quality(center)
            if not (
                center_quality > tree_leaf_refinement_quality(lower)
                and center_quality > tree_leaf_refinement_quality(upper)
            ):
                continue
            midpoint_options: list[tuple[float, int]] = []
            for neighbour in (lower, upper):
                gap = abs(int(neighbour.max_leaf_nodes) - center_leaf)
                if gap <= 1:
                    continue
                midpoint = (int(neighbour.max_leaf_nodes) + center_leaf) // 2
                if midpoint in by_leaf or midpoint == center_leaf:
                    continue
                midpoint_options.append((
                    tree_leaf_refinement_quality(neighbour), midpoint,
                ))
            if midpoint_options:
                midpoints = [
                    value for _, value in sorted(midpoint_options, reverse=True)
                ]
                refinements.append((
                    center_quality, center, lower, upper, midpoints,
                ))
    if not refinements:
        return population, report
    report["active"] = True
    variant: Genome | None = None
    selected: tuple[float, Genome, Genome, Genome, list[int]] | None = None
    selected_leaf: int | None = None
    for refinement in sorted(refinements, key=lambda item: item[0], reverse=True):
        _, center, _, _, midpoints = refinement
        for midpoint in midpoints:
            payload = asdict(center)
            payload.update({
                "max_leaf_nodes": midpoint,
                "generation": generation,
                "parents": [center.genome_id],
                "genome_id": "",
                "fitness": None,
                "result": None,
            })
            proposal = Genome(**payload).finalize()
            if genome_evaluation_key(proposal) not in known_keys:
                variant = proposal
                selected = refinement
                selected_leaf = midpoint
                break
        if variant is not None:
            break
    if variant is None or selected is None or selected_leaf is None:
        return population, report
    protected_parent_ids = protected_parent_ids or set()
    replacement = next((
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and not (set(population[index].parents) & protected_parent_ids)
    ), None)
    replacement_scope = "unprotected"
    if replacement is None:
        parent_counts = Counter(
            parent for genome in population if genome.fitness is None
            for parent in genome.parents if parent in protected_parent_ids
        )
        replacement = next((
            index for index in range(len(population) - 1, 0, -1)
            if population[index].fitness is None
            and any(parent_counts[parent] > 1 for parent in population[index].parents)
        ), None)
        replacement_scope = "redundant_protected_sibling"
    if replacement is None:
        report["replacement_scope"] = "none"
        return population, report
    _, center, lower, upper, _ = selected
    population[replacement] = variant
    report.update({
        "proposed": True,
        "genome_id": variant.genome_id,
        "parent_id": center.genome_id,
        "max_leaf_nodes": selected_leaf,
        "signed_bracket": [
            int(lower.max_leaf_nodes), int(center.max_leaf_nodes),
            int(upper.max_leaf_nodes),
        ],
        "replacement_scope": replacement_scope,
    })
    return population, report


def introduce_missing_learner_species(
    population: list[Genome], generation: int, rng: random.Random,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Guarantee architecture upgrades enter a resumed population immediately."""
    learners = LEARNER_KINDS
    present = {genome.learner_kind for genome in population}
    if not population:
        return population
    base = population[0]
    protected_parent_ids = protected_parent_ids or set()
    protected = {
        index for index, genome in enumerate(population)
        if bool(set(genome.parents) & protected_parent_ids)
    }
    replaceable = [
        index for index in range(len(population) - 1, 0, -1)
        if index not in protected and population[index].fitness is None
    ]
    replaceable.extend(
        index for index in range(len(population) - 1, 0, -1)
        if index not in protected and index not in replaceable
    )
    missing = [kind for kind in learners if kind not in present]
    for learner, replacement_index in zip(missing, replaceable):
        payload = asdict(base)
        payload.update({
            "learner_kind": learner,
            "market_weight": rng.uniform(.25, 1.75),
            "regime_feature": rng.choice(REGIME_FEATURES),
            "regime_bins": (rng.randint(2, 3) if learner in {
                "regime_regressor", "regime_decomposed_regressor"
            } else 1),
            "training_horizons": sorted({12, rng.choice(AUXILIARY_HORIZONS)}),
            "pool_thresholds": {
                name: rng.randint(3, 8) for name in EVOLVABLE_POOL_NAMES
            },
            "calibration_safety": 10 ** rng.uniform(0, math.log10(8)),
            "generation": generation,
            "parents": [base.genome_id],
            "fitness": None,
            "result": None,
            "genome_id": "",
        })
        replacement = Genome(**payload).finalize()
        population[replacement_index] = replacement
    return population


def introduce_calibration_variants(population: list[Genome], generation: int) -> list[Genome]:
    """Seed a new monotonic calibration gene without waiting for genetic drift."""
    if len(population) < 4 or len({round(g.calibration_safety, 3) for g in population}) > 1:
        return population
    base = population[0]
    for offset, safety in enumerate((2.0, 4.0, 8.0), 1):
        payload = asdict(base)
        payload.update({
            "calibration_safety": safety, "generation": generation,
            "parents": [base.genome_id], "fitness": None, "result": None,
            "genome_id": "",
        })
        population[-offset] = Genome(**payload).finalize()
    return population


def introduce_reflexivity_variant(
    population: list[Genome], generation: int,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Seed the new participant-state pool without making it globally mandatory."""
    if not population or any(set(genome.features) & REFLEXIVITY_FEATURES
                             for genome in population):
        return population
    base = population[0]
    protected_parent_ids = protected_parent_ids or set()
    counts = Counter(genome.learner_kind for genome in population)
    replacement_index = next(
        (index for index in range(len(population) - 1, -1, -1)
         if counts[population[index].learner_kind] > 1
         and not (set(population[index].parents) & protected_parent_ids)),
        next((index for index in range(len(population) - 1, 0, -1)
              if not (set(population[index].parents) & protected_parent_ids)), None),
    )
    if replacement_index is None:
        return population
    payload = asdict(base)
    payload.update({
        "features": sorted(set(base.features) | REFLEXIVITY_FEATURES),
        "generation": generation, "parents": [base.genome_id],
        "fitness": None, "result": None, "genome_id": "",
    })
    population[replacement_index] = Genome(**payload).finalize()
    return population


def introduce_emergent_pool_variant(
    population: list[Genome], generation: int, rng: random.Random,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Ensure a resumed population immediately tests mutable pool topology."""
    if not population or any(genome.emergent_pools for genome in population):
        return population
    base = next((genome for genome in population if genome.feature_programs), population[0])
    pool = random_emergent_pool(base, rng)
    if pool is None:
        return population
    protected_parent_ids = protected_parent_ids or set()
    counts = Counter(genome.learner_kind for genome in population)
    replacement_index = next(
        (index for index in range(len(population) - 1, -1, -1)
         if counts[population[index].learner_kind] > 1
         and not (set(population[index].parents) & protected_parent_ids)),
        next((index for index in range(len(population) - 1, 0, -1)
              if not (set(population[index].parents) & protected_parent_ids)), None),
    )
    if replacement_index is None:
        return population
    payload = asdict(base)
    payload.update({
        "emergent_pools": [pool], "generation": generation,
        "parents": [base.genome_id], "fitness": None, "result": None,
        "genome_id": "",
    })
    population[replacement_index] = Genome(**payload).finalize()
    return population


def preserve_emergent_pool_elite(
    population: list[Genome], evaluated: Sequence[Genome],
) -> list[Genome]:
    """Retain one topology species until neural evidence can judge its pools."""
    if not population or any(genome.emergent_pools for genome in population):
        return population
    specialist = next((genome for genome in evaluated if genome.emergent_pools), None)
    if specialist is None:
        return population
    counts = Counter(genome.learner_kind for genome in population)
    replacement_index = next(
        (index for index in range(len(population) - 1, 0, -1)
         if counts[population[index].learner_kind] > 1),
        len(population) - 1,
    )
    population[replacement_index] = genome_from_dict(asdict(specialist)).finalize()
    return population


def brain_feedback_score(report: dict[str, Any] | None) -> float:
    """Bound neural smoke evidence so it guides but cannot bypass fold stages."""
    sections = [
        section.get("metrics", {})
        for fold in (report or {}).get("folds", [])
        for section in fold.get("sections", {}).values()
    ]
    if not sections:
        return -25.0
    accuracy = min(float(row.get("directional_accuracy", 0)) for row in sections)
    balanced = min(float(row.get("directional_balanced_accuracy", 0)) for row in sections)
    mcc = min(float(row.get("mcc", -1)) for row in sections)
    profit = min(float(row.get("profit_factor") or 0) for row in sections)
    score = (80 * (accuracy - .5) + 35 * (balanced - .5)
             + 12 * mcc + 4 * (profit - 1.0))
    return max(-25.0, min(25.0, score))


def selection_fitness(genome: Genome, neural_scores: dict[str, float] | None = None) -> float:
    return float(genome.fitness if genome.fitness is not None else -math.inf) + float(
        (neural_scores or {}).get(genome.genome_id, 0.0)
    )


def completed_brain_gate_evidence(
    state_dir: Path, evaluation_id: str,
) -> tuple[dict[str, float], set[str]]:
    """Recover current-scope neural scores, including orphaned gate reports."""
    phenotype_scores: dict[str, float] = {}
    failed_phenotypes: set[str] = set()
    for report_path in (state_dir / "brain-gate-reports").glob("*.smoke.json"):
        genome_id = report_path.name.removesuffix(".smoke.json")
        candidate_path = state_dir / "candidates" / f"{genome_id}.json"
        try:
            candidate = genome_from_dict(json.loads(
                candidate_path.read_text(encoding="utf-8")
            ))
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if ((candidate.result or {}).get("evaluation_signature")
                != evaluation_id):
            continue
        phenotype = genome_evaluation_key(candidate)
        score = brain_feedback_score(report)
        phenotype_scores[phenotype] = min(
            phenotype_scores.get(phenotype, score), score
        )
        if not bool(report.get("all_brain_floor_gates")):
            failed_phenotypes.add(phenotype)
    scores: dict[str, float] = {}
    failed: set[str] = set()
    if not phenotype_scores:
        return scores, failed
    for candidate_path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(
                candidate_path.read_text(encoding="utf-8")
            ))
        except (OSError, ValueError, TypeError):
            continue
        if ((candidate.result or {}).get("evaluation_signature")
                != evaluation_id):
            continue
        phenotype = genome_evaluation_key(candidate)
        if phenotype in phenotype_scores:
            scores[candidate.genome_id] = phenotype_scores[phenotype]
        if phenotype in failed_phenotypes:
            failed.add(candidate.genome_id)
    return scores, failed


def best_brain_gate_eligible_champion(
    state_dir: Path, evaluation_id: str, neural_scores: dict[str, float],
    failed_gate_ids: set[str],
) -> Genome | None:
    """Recover the strongest fully screened candidate not neurally rejected."""
    candidates: list[Genome] = []
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        result = candidate.result or {}
        if (candidate.genome_id in failed_gate_ids
                or candidate.fitness is None
                or result.get("evaluation_signature") != evaluation_id
                or int(result.get("evaluated_folds", 0))
                < int(result.get("requested_folds", 3))):
            continue
        candidates.append(candidate)
    return max(
        candidates,
        key=lambda genome: selection_fitness(genome, neural_scores),
        default=None,
    )


def reconcile_brain_gate_champion(
    state_dir: Path, champion: Genome | None, evaluation_id: str,
    neural_scores: dict[str, float],
) -> tuple[Genome | None, str | None, set[str]]:
    """Invalidate a champion when a completed isolated gate rejected it."""
    report_scores, failed_gate_ids = completed_brain_gate_evidence(
        state_dir, evaluation_id
    )
    neural_scores.update(report_scores)
    if champion is None or champion.genome_id not in failed_gate_ids:
        return champion, None, failed_gate_ids
    rejected = champion.genome_id
    replacement = best_brain_gate_eligible_champion(
        state_dir, evaluation_id, neural_scores, failed_gate_ids
    )
    return replacement, rejected, failed_gate_ids


def _env_float(name: str, default: float) -> float:
    """Env override for a safety ceiling, ignoring unparseable values."""
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def fit_live_surrogate(genome_payload: dict[str, Any],
                       *, dataset: dict[str, Any] | None = None) -> Any:
    """Return a fitted Surrogate for a genome, for live scoring.

    The GA never persists a fitted estimator -- only the genome spec -- so a
    consumer that wants to score live bars has to refit. Doing that by
    reimplementing the learner branches would drift from what was measured,
    so this reuses the evaluation path and hands back the model it produced.

    Returns None when the genome cannot be fitted; callers must treat that as
    "unscorable" rather than substituting a guess.
    """
    try:
        genome = genome_from_dict(genome_payload)
    except (TypeError, ValueError, KeyError):
        return None

    if dataset is None:
        try:
            dataset = load_dataset_cached(
                ROOT / "runtime/benchmarks/market-corpus-manifest.json",
                Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"),
                12, 12, "market-perpetual-v1",
                Path(r"D:\Projects\CoolCryptoUtilities\data\news\historical_deduplicated.json"),
                ROOT / "runtime/cache/market-evolution-dataset-v4.joblib",
            )
        except Exception:
            return None

    captured: list[Any] = []
    original = globals().get("Surrogate")
    if original is None:
        return None

    def capture(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        captured.append(model)
        return model

    globals()["Surrogate"] = capture
    try:
        evaluate_genome(genome, dataset, folds=1, test_days=28,
                        calibration_days=30, final_days=21, horizon=12,
                        cost_bps=OBJECTIVE_COST_BPS)
    except Exception:
        pass
    finally:
        globals()["Surrogate"] = original

    return captured[-1] if captured else None


def champion_replacement_allowed(candidate: Genome, incumbent: Genome) -> bool:
    """Require a bounded Pareto improvement for the durable research champion."""
    if (candidate.fitness is None or incumbent.fitness is None
            or candidate.fitness <= incumbent.fitness):
        return False
    candidate_result = candidate.result or {}
    incumbent_result = incumbent.result or {}
    if (int(candidate_result.get("evaluated_folds", 0))
            < int(candidate_result.get("requested_folds", 3))):
        return False
    candidate_summary = candidate_result.get("summary", {})
    incumbent_summary = incumbent_result.get("summary", {})
    # PROFIT decides. Only the economic metrics may veto a better earner.
    #
    # Requiring a candidate to be no worse on all seven metrics made this a
    # strict Pareto ratchet, and any single non-economic regression outranked
    # the objective. Measured over this run's event log: 40 candidates that
    # earned MORE than the incumbent were suppressed, and 23 of them were
    # blocked ONLY by "regression" on a diagnostic metric -- min_coverage
    # accounted for 26 of the 84 individual blocks, more than any other.
    #
    # Coverage is not profit. Across 1069 full-fold genomes in this same run,
    # corr(coverage, expectancy) = -0.003: how OFTEN a genome trades carries
    # essentially no information about what it earns. corr(MCC, expectancy) is
    # +0.60, so discrimination is the real driver -- but MCC is already priced
    # into profit, and gating on it separately double-counts it. Meanwhile the
    # incumbent sat at PF 1.081 while PF 1.2244 genomes went unpromoted.
    #
    # So: profit and expectancy may not regress, safety keeps its absolute
    # ceilings below, and the diagnostic metrics are reported rather than
    # enforced.
    economic_bounds = {
        "min_expectancy": .000025,
        "min_profit_factor": .002,
    }
    for name, tolerance in economic_bounds.items():
        if (float(candidate_summary.get(name, -math.inf))
                < float(incumbent_summary.get(name, -math.inf)) - tolerance):
            return False
    # Calibration and drawdown are RISK guards, not profit vetoes.
    #
    # Enforcing them as a strict Pareto bound made them silently outrank the
    # objective: measured 2026-08-18, genome 701a3dfda8afcf6e beat the
    # incumbent on every economic metric (PF 1.1965 vs 1.0567, expectancy 3x,
    # better accuracy/MCC/coverage/margin) and was refused the championship
    # because its ECE was 0.178 vs 0.087 and drawdown 0.844 vs 0.782. A
    # second genome (3568a1392bb2efae, PF 1.0731) was blocked identically.
    # The objective says PROFIT DECIDES and these are tie-breakers, so a
    # better earner may trade some calibration for the profit -- but only
    # within an absolute safety ceiling that no amount of profit can buy
    # through.
    for name, ceiling_env, default_ceiling in (
        ("max_ece", "CHAMPION_MAX_ECE_CEILING", 0.25),
        ("max_drawdown", "CHAMPION_MAX_DRAWDOWN_CEILING", 1.00),
    ):
        ceiling = _env_float(ceiling_env, default_ceiling)
        # A ceiling the SITTING champion already violates is unfalsifiable: it
        # refuses every challenger while the incumbent keeps the title on a
        # worse number, so nothing can ever replace it. Measured 2026-08-18 the
        # champion carried max_drawdown 2.14 against this 1.00 ceiling, and 21
        # better-earning candidates were refused for drawdowns of 2.04-2.20 --
        # several of them BELOW the incumbent's own.
        #
        # When the incumbent is already over a ceiling, the honest bar is "no
        # worse than what is actually running", so the guard still blocks a
        # deterioration but cannot defend an incumbent that fails its own
        # standard. A candidate inside the ceiling is always acceptable.
        effective = max(ceiling, float(incumbent_summary.get(name, -math.inf)))
        if float(candidate_summary.get(name, math.inf)) > effective:
            return False
    return True


def champion_replacement_blockers(candidate: Genome,
                                  incumbent: Genome) -> list[dict[str, Any]]:
    """Explain, metric by metric, why a candidate cannot take the title.

    Exists so a suppressed better-earner is never a mystery: the event log
    names the exact comparison that blocked it, with both values.
    """
    blockers: list[dict[str, Any]] = []
    if candidate.fitness is None or incumbent.fitness is None:
        blockers.append({"reason": "missing_fitness"})
        return blockers
    if candidate.fitness <= incumbent.fitness:
        blockers.append({"reason": "fitness_not_higher",
                         "candidate": candidate.fitness,
                         "incumbent": incumbent.fitness})
    candidate_result = candidate.result or {}
    if (int(candidate_result.get("evaluated_folds", 0))
            < int(candidate_result.get("requested_folds", 3))):
        blockers.append({"reason": "incomplete_walk_forward",
                         "evaluated_folds": candidate_result.get("evaluated_folds"),
                         "requested_folds": candidate_result.get("requested_folds")})
    candidate_summary = candidate_result.get("summary", {})
    incumbent_summary = (incumbent.result or {}).get("summary", {})
    # Mirrors champion_replacement_allowed: only the economic metrics block.
    # The diagnostic metrics are still reported, as "observed_regression", so a
    # promotion that traded accuracy or coverage for profit stays auditable
    # without that trade being silently forbidden.
    economic_bounds = {"min_expectancy": .000025, "min_profit_factor": .002}
    diagnostic_bounds = {
        "min_accuracy": 0.0, "min_balanced_accuracy": .002, "min_mcc": .005,
        "min_baseline_margin": .005, "min_coverage": .005,
    }
    for name, tolerance in economic_bounds.items():
        value = float(candidate_summary.get(name, -math.inf))
        floor = float(incumbent_summary.get(name, -math.inf)) - tolerance
        if value < floor:
            blockers.append({"reason": "regression", "metric": name,
                             "candidate": value,
                             "incumbent": incumbent_summary.get(name)})
    for name, tolerance in diagnostic_bounds.items():
        value = float(candidate_summary.get(name, -math.inf))
        floor = float(incumbent_summary.get(name, -math.inf)) - tolerance
        if value < floor:
            blockers.append({"reason": "observed_regression", "metric": name,
                             "candidate": value,
                             "incumbent": incumbent_summary.get(name)})
    for name, ceiling_env, default_ceiling in (
        ("max_ece", "CHAMPION_MAX_ECE_CEILING", 0.25),
        ("max_drawdown", "CHAMPION_MAX_DRAWDOWN_CEILING", 1.00),
    ):
        value = float(candidate_summary.get(name, math.inf))
        ceiling = _env_float(ceiling_env, default_ceiling)
        effective = max(ceiling, float(incumbent_summary.get(name, -math.inf)))
        if value > effective:
            blockers.append({"reason": "safety_ceiling", "metric": name,
                             "candidate": value, "ceiling": ceiling,
                             "effective_ceiling": effective,
                             "incumbent": incumbent_summary.get(name)})
    return blockers


def rollback_unsafe_champion(
    state_dir: Path, champion: Genome | None, evaluation_id: str,
) -> tuple[Genome | None, list[str]]:
    """Undo scalar-fitness handoffs that violate the new Pareto contract."""
    if champion is None:
        return None, []
    rolled_back: list[str] = []
    current = champion
    visited = {current.genome_id}
    while current.parents:
        parents: list[Genome] = []
        for parent_id in current.parents:
            path = state_dir / "candidates" / f"{parent_id}.json"
            try:
                parent = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            if (parent.fitness is not None
                    and (parent.result or {}).get("evaluation_signature") == evaluation_id):
                parents.append(parent)
        if not parents:
            break
        parent = max(parents, key=lambda genome: genome.fitness or -math.inf)
        if parent.genome_id in visited or champion_replacement_allowed(current, parent):
            break
        rolled_back.append(current.genome_id)
        current = parent
        visited.add(current.genome_id)
    return current, rolled_back


def select_diverse_elites(population: Sequence[Genome], count: int,
                          neural_scores: dict[str, float] | None = None) -> list[Genome]:
    """Retain fitness leaders plus viable behavioral species.

    Species preservation keeps evolution capable of revisiting learner and
    expression behaviors after a regime change without allowing novelty to
    satisfy any admission metric.
    """
    ranked = sorted(population, key=lambda genome: selection_fitness(genome, neural_scores),
                    reverse=True)
    selected: list[Genome] = []
    seen: set[str] = set()

    def retain(genome: Genome) -> None:
        if genome.genome_id not in seen and len(selected) < count:
            seen.add(genome.genome_id)
            selected.append(genome_from_dict(asdict(genome)))

    if ranked:
        retain(ranked[0])
    for learner in LEARNER_KINDS:
        candidate = next((genome for genome in ranked if genome.learner_kind == learner), None)
        if candidate is not None:
            retain(candidate)
    for has_programs in (True, False):
        candidate = next((genome for genome in ranked
                          if bool(genome.feature_programs) is has_programs), None)
        if candidate is not None:
            retain(candidate)
    for uses_reflexivity in (True, False):
        candidate = next((genome for genome in ranked
                          if bool(set(genome.features) & REFLEXIVITY_FEATURES)
                          is uses_reflexivity), None)
        if candidate is not None:
            retain(candidate)
    for genome in ranked:
        retain(genome)
    return selected


def elite_budget(population_size: int) -> int:
    """Bound survivors so evolution can never retain the entire population."""
    if population_size < 4:
        raise ValueError("population must be at least four")
    return min(population_size - 2, max(2, round(population_size * .40)))


def minimum_novel_candidates(population_size: int) -> int:
    """Minimum unevaluated hypotheses required at every generation boundary."""
    return min(population_size - 1, max(2, math.ceil(population_size * .25)))


def breed_population(
    evaluated: Sequence[Genome], generation: int, rng: random.Random,
    neural_scores: dict[str, float] | None = None,
) -> tuple[list[Genome], dict[str, int]]:
    """Create a full population with a hard guarantee of genuine offspring."""
    size = len(evaluated)
    budget = elite_budget(size)
    elites = select_diverse_elites(evaluated, budget, neural_scores)
    following = elites[:]
    known_ids = {genome.genome_id for genome in following}
    attempts = 0
    created = 0
    ranked = sorted(
        evaluated, key=lambda genome: selection_fitness(genome, neural_scores),
        reverse=True,
    )
    parent_pool = ranked[:max(budget * 2, 4)]
    while len(following) < size and attempts < size * 100:
        attempts += 1
        left, right = rng.sample(parent_pool, 2)
        child = crossover(left, right, generation, rng)
        if child.genome_id in known_ids:
            continue
        known_ids.add(child.genome_id)
        following.append(child)
        created += 1
    # Crossover can theoretically collide in a converged population. Mutated
    # immigrants keep the service moving instead of silently becoming inert.
    while len(following) < size and attempts < size * 300:
        attempts += 1
        child = mutate(rng.choice(ranked), generation, rng)
        if child.genome_id in known_ids:
            continue
        known_ids.add(child.genome_id)
        following.append(child)
        created += 1
    if len(following) != size or created < minimum_novel_candidates(size):
        raise RuntimeError(
            f"evolution invariant failed: population={size} elites={len(elites)} "
            f"offspring={created}"
        )
    return following, {
        "population": size, "elite_budget": budget,
        "offspring_created": created, "breeding_attempts": attempts,
    }


def ensure_novelty(
    population: list[Genome], generation: int, rng: random.Random,
) -> tuple[list[Genome], int]:
    """Repair any later transformation that accidentally consumes exploration."""
    required = minimum_novel_candidates(len(population))
    injected = 0
    known_ids = {genome.genome_id for genome in population}
    attempts = 0
    while sum(genome.fitness is None for genome in population) < required:
        attempts += 1
        if attempts > len(population) * 100:
            raise RuntimeError("unable to restore minimum evolutionary novelty")
        replaceable = [
            index for index, genome in enumerate(population)
            if index > 0 and genome.fitness is not None
        ]
        if not replaceable:
            raise RuntimeError("no replaceable evaluated genome for novelty repair")
        child = mutate(rng.choice(population), generation, rng)
        if child.genome_id in known_ids:
            continue
        index = replaceable[-1]
        known_ids.discard(population[index].genome_id)
        population[index] = child
        known_ids.add(child.genome_id)
        injected += 1
    return population, injected


def introduce_directional_frontier_variants(
    population: list[Genome], evaluated: Sequence[Genome], generation: int,
) -> list[Genome]:
    """Preserve useful signs while testing materially safer confidence scales."""
    frontier = [
        genome for genome in evaluated
        if genome.learner_kind in RETURN_LEARNER_KINDS
        and (genome.result or {}).get("summary", {}).get(
            "min_accuracy", 0
        ) >= PRESCREEN["accuracy"]
        and (genome.result or {}).get("summary", {}).get(
            "min_balanced_accuracy", 0
        ) >= PRESCREEN["balanced_accuracy"]
        and (genome.result or {}).get("summary", {}).get(
            "min_mcc", -1
        ) >= PRESCREEN["mcc"]
        and (genome.result or {}).get("summary", {}).get("min_expectancy", 0) > 0
        and (genome.result or {}).get("summary", {}).get("max_ece", 0) > FLOOR["ece"]
    ]
    if not frontier or len(population) < len(LEARNER_KINDS) + 2:
        return population
    base = max(frontier, key=lambda genome: (
        (genome.result or {}).get("summary", {}).get("min_accuracy", 0),
        (genome.result or {}).get("summary", {}).get("min_mcc", -1),
    ))
    known = {genome.genome_id for genome in population}
    variants = []
    for multiplier in (4.0, 8.0):
        payload = asdict(base)
        payload.update({
            "calibration_safety": min(12.0, max(1.0, base.calibration_safety * multiplier)),
            "generation": generation, "parents": [base.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        variant = Genome(**payload).finalize()
        if variant.genome_id not in known:
            known.add(variant.genome_id)
            variants.append(variant)
    if variants:
        population[-len(variants):] = variants
    return population


def introduce_coverage_repair_variants(
    population: list[Genome], evaluated: Sequence[Genome], generation: int,
    reversal_frontier: Genome | None = None,
    multiscale_frontier: Genome | None = None,
    multiscale_reversal_frontier: Genome | None = None,
    multiscale_boundary_frontier: Genome | None = None,
) -> list[Genome]:
    """Expand the action envelope of accurate but over-selective lineages."""
    frontier = [
        genome for genome in evaluated
        if (genome.result or {}).get("summary", {}).get("min_accuracy", 0) >= .56
        and (genome.result or {}).get("summary", {}).get("min_balanced_accuracy", 0) >= .52
        and (genome.result or {}).get("summary", {}).get("min_coverage", 1)
        < PRESCREEN["coverage"]
    ]
    if not frontier or len(population) < 4:
        return population
    base = max(frontier, key=lambda genome: (
        coverage_frontier_rank(genome) or (-math.inf,)
    ))
    if multiscale_frontier_rank(multiscale_frontier) is not None:
        # Preserve the high-accuracy architecture as a separate research
        # lineage even when the primary frontier is closer to the coverage
        # gate. Its own descendants must earn coverage independently.
        base = multiscale_frontier
        reversal_frontier = multiscale_reversal_frontier
    if multiscale_boundary_rank(multiscale_boundary_frontier) is not None:
        # A boundary may be directionally strong yet not economically
        # promotable. Use it only to localize the threshold discontinuity;
        # the independent multiscale frontier remains the quality record.
        base = multiscale_boundary_frontier
        reversal_frontier = multiscale_reversal_frontier
    known = {genome.genome_id for genome in population}
    variants: list[Genome] = []
    summary = (base.result or {}).get("summary", {})
    failed_margin_children = [
        genome for genome in evaluated
        if base.genome_id in genome.parents
        and genome_structure_key(genome) != genome_structure_key(base)
        and (
            int((genome.result or {}).get("evaluated_folds", 0)) >= 2
            or (
                genome.calibration_reliability
                and genome.calibration_reliability_version >= 6
                and int((genome.result or {}).get("evaluated_folds", 0)) >= 1
            )
        )
        and (
            float((genome.result or {}).get("summary", {}).get(
                "min_accuracy", 1
            )) < .50
            # Mature reliability descendants carry protected search memory
            # even when they remove the anti-signal and land exactly at the
            # neutral boundary. Excluding == .50 caused cached repeats rather
            # than the intended pool/threshold bracket progression.
            or genome.calibration_reliability_version >= 6
        )
    ]
    narrow_reversal_bracket = False
    if (reversal_frontier is not None
            and reversal_frontier.confidence_quantile < base.confidence_quantile
            and genome_structure_key(reversal_frontier) == genome_structure_key(base)):
        # We have a causal bracket: the upper endpoint preserves the edge but
        # misses coverage, while the lower endpoint crosses coverage and
        # reveals a protected-fold reversal. Bisect rather than repeat either.
        low = reversal_frontier.confidence_quantile
        high = base.confidence_quantile
        narrow_reversal_bracket = high - low <= .002
        if narrow_reversal_bracket:
            # Threshold search has localized a discontinuity closely enough
            # that another scalar bisection is not informative.  Cross the
            # boundary deliberately and alter only causal feature ordering so
            # protected fold 2 can tell us which interaction generalizes.
            quantiles = (low, low)
        else:
            midpoint = (low + high) / 2
            quantiles = (midpoint, (midpoint + high) / 2)
    else:
        coverage_gap = max(.001, PRESCREEN["coverage"] - float(
            summary.get("min_coverage", 0)
        ))
        # Before a bracket exists, scale movement to the measured deficit.
        phase_scale = (.5, .75, 1.0)[generation % 3]
        offsets = (
            max(.0025, coverage_gap * 1.5) * phase_scale,
            max(.005, coverage_gap * 3.0) * phase_scale,
        )
        quantiles = (
            base.confidence_quantile - offsets[0],
            base.confidence_quantile - offsets[1],
        )
    margin_templates = (
        {"op": "tanh_mix", "left": "asset_news_sentiment_acceleration",
         "right": "flow_divergence", "scale": 1.0},
        {"op": "signed_sqrt_product", "left": "market_breadth_r6",
         "right": "relative_market_r6", "scale": 1.0},
        {"op": "regime_gate", "left": "news_negative_share_24h",
         "right": "volatility_ratio", "scale": 1.0},
        {"op": "abs_gap", "left": "futures_spot_basis",
         "right": "flow_imbalance", "scale": 1.0},
    )
    for variant_index, quantile in enumerate(quantiles):
        payload = asdict(base)
        if narrow_reversal_bracket and len(failed_margin_children) >= 2:
            # Independent interaction probes failed to change the temporal
            # reversal. Escalate architecture while retaining the successful
            # feature core: blend short and long causal memory, with the blend
            # selected only on each fold's calibration period.
            payload["learner_kind"] = "multiscale_regressor"
            if base.learner_kind == "multiscale_regressor":
                attempted_scales = {
                    round(
                        child.recency_half_life_days
                        / max(1.0, base.recency_half_life_days),
                        6,
                    )
                    for child in failed_margin_children
                    if child.learner_kind == "multiscale_regressor"
                    and child.feature_programs == base.feature_programs
                    and not child.calibration_orientation
                    and not child.calibration_reliability
                }
                no_flip_oriented = [
                    child for child in failed_margin_children
                    if child.learner_kind == "multiscale_regressor"
                    and child.feature_programs == base.feature_programs
                    and child.calibration_orientation
                    and (child.result or {}).get("folds")
                    and all(
                        float(fold.get("multiscale_calibration", {}).get(
                            "direction", 1.0
                        )) > 0
                        for fold in (child.result or {}).get("folds", [])
                    )
                ]
                if len(no_flip_oriented) >= 2:
                    reliability_failures = {
                        version: [
                            child for child in failed_margin_children
                            if child.learner_kind == "multiscale_regressor"
                            and child.feature_programs == base.feature_programs
                            and child.calibration_reliability
                            and child.calibration_reliability_version == version
                            and child.fitness is not None
                        ]
                        for version in (1, 2, 3, 4, 5)
                    }
                    # A paired linear failure is evidence that WHEN/WHY
                    # correctness is interaction-shaped, not that reliability
                    # ranking itself should be abandoned.
                    reliability_version = 1
                    for completed_version in (1, 2, 3, 4, 5):
                        if len(reliability_failures[completed_version]) >= 2:
                            reliability_version = completed_version + 1
                        else:
                            break
                    version_six_evidence = [
                        child for child in failed_margin_children
                        if child.learner_kind == "multiscale_regressor"
                        and child.feature_programs == base.feature_programs
                        and child.calibration_reliability_version == 6
                        and child.fitness is not None
                    ]
                    if reliability_version == 6 and len(version_six_evidence) >= 8:
                        reliability_version = 7
                    version_seven_evidence = [
                        child for child in failed_margin_children
                        if child.learner_kind == "multiscale_regressor"
                        and child.feature_programs == base.feature_programs
                        and child.calibration_reliability_version == 7
                        and child.fitness is not None
                    ]
                    if reliability_version == 7 and len(version_seven_evidence) >= 2:
                        reliability_version = 8
                    reliability_scales = {
                        round(
                            child.recency_half_life_days
                            / max(1.0, base.recency_half_life_days),
                            6,
                        )
                        for child in failed_margin_children
                        if child.learner_kind == "multiscale_regressor"
                        and child.feature_programs == base.feature_programs
                        and child.calibration_reliability
                        and child.calibration_reliability_version == reliability_version
                    }
                    reliability_schedule = (1.0, .5, 2.0, .25, 4.0, .125, 8.0)
                    minimum_scale = 14.0 / max(14.0, base.recency_half_life_days)
                    maximum_scale = 2200.0 / max(14.0, base.recency_half_life_days)
                    scheduled = list(dict.fromkeys(
                        round(min(maximum_scale, max(minimum_scale, scale)), 6)
                        for scale in reliability_schedule
                    ))
                    remaining = [scale for scale in scheduled
                                 if scale not in reliability_scales]
                    # Once the coarse schedule is exhausted, bisect its
                    # largest untested memory gaps. This keeps the perpetual
                    # loop novel and prevents an empty two-candidate batch.
                    while len(remaining) < 2:
                        anchors = sorted({
                            round(minimum_scale, 6), round(maximum_scale, 6),
                            *reliability_scales, *remaining,
                        })
                        gaps = sorted(
                            ((right - left, left, right)
                             for left, right in zip(anchors, anchors[1:])),
                            reverse=True,
                        )
                        added = False
                        for _, left, right in gaps:
                            midpoint = round((left + right) / 2.0, 6)
                            if midpoint not in reliability_scales and midpoint not in remaining:
                                remaining.append(midpoint)
                                added = True
                                break
                        if not added:
                            break
                    if len(remaining) < 2:
                        remaining.extend([remaining[0] if remaining else 1.0] * 2)
                    memory_scales = tuple(remaining[:2])
                    payload["calibration_orientation"] = False
                    payload["calibration_reliability"] = True
                    payload["calibration_reliability_version"] = reliability_version
                    if reliability_version == 6:
                        remaining_decays = next_reliability_decays(
                            version_six_evidence
                        )
                        # Hold memory at the empirically strongest short-memory
                        # boundary (78 days for the current 313-day parent) so
                        # the two candidates isolate decay rather than confound
                        # decay with memory again.
                        memory_scales = (.25, .25)
                        payload["calibration_reliability_decay"] = (
                            remaining_decays[variant_index]
                        )
                    elif reliability_version >= 7:
                        best_decay = max(
                            version_six_evidence, key=reliability_evidence_rank
                        )
                        best_scale = (
                            best_decay.recency_half_life_days
                            / max(1.0, base.recency_half_life_days)
                        )
                        memory_scales = (best_scale, best_scale)
                        payload["calibration_reliability_decay"] = (
                            best_decay.calibration_reliability_decay
                        )
                        if reliability_version == 7:
                            quantile = next_reliability_quantiles(
                                version_seven_evidence,
                                best_decay.confidence_quantile,
                            )[variant_index]
                        else:
                            version_eight_evidence = [
                                child for child in failed_margin_children
                                if child.learner_kind == "multiscale_regressor"
                                and child.feature_programs == base.feature_programs
                                and child.calibration_reliability_version == 8
                                and child.fitness is not None
                            ]
                            reliability_pool, quantile = (
                                next_oriented_reliability_variants(
                                    version_eight_evidence,
                                    best_decay.confidence_quantile,
                                )[variant_index]
                            )
                            payload["calibration_reliability_pool"] = reliability_pool
                    elif reliability_version >= 5:
                        payload["calibration_reliability_decay"] = 2.0
                    else:
                        payload["calibration_reliability_decay"] = 0.0
                    if reliability_version == 3:
                        payload["calibration_reliability_pool"] = (
                            "trend_regime" if variant_index == 0 else "flow_news"
                        )
                    elif reliability_version == 4:
                        payload["calibration_reliability_pool"] = "combined"
                    elif reliability_version >= 8:
                        # Assigned by the orientation-aware pool/quantile
                        # scheduler above.
                        pass
                    elif reliability_version >= 5:
                        payload["calibration_reliability_pool"] = "flow_news"
                    else:
                        payload["calibration_reliability_pool"] = "core"
                elif len(attempted_scales) >= 4:
                    oriented_scales = {
                        round(
                            child.recency_half_life_days
                            / max(1.0, base.recency_half_life_days),
                            6,
                        )
                        for child in failed_margin_children
                        if child.learner_kind == "multiscale_regressor"
                        and child.feature_programs == base.feature_programs
                        and child.calibration_orientation
                        and not child.calibration_reliability
                    }
                    orientation_schedule = (1.0, .5, 2.0, .25, 4.0, .125, 8.0)
                    remaining = [
                        scale for scale in orientation_schedule
                        if round(scale, 6) not in oriented_scales
                    ]
                    if len(remaining) >= 2:
                        memory_scales = tuple(remaining[:2])
                        payload["calibration_orientation"] = True
                    else:
                        shortest = min({1.0, *attempted_scales})
                        longest = max({1.0, *attempted_scales})
                        memory_scales = (shortest / 2.0, longest * 2.0)
                else:
                    shortest = min({1.0, *attempted_scales})
                    longest = max({1.0, *attempted_scales})
                    # Each failed pair widens the causal-memory search instead
                    # of regenerating an already disproven phenotype.
                    memory_scales = (shortest / 2.0, longest * 2.0)
                memory_scales = tuple(
                    min(2200.0 / max(14.0, base.recency_half_life_days),
                        max(14.0 / max(14.0, base.recency_half_life_days), scale))
                    for scale in memory_scales
                )
            else:
                memory_scales = (1.0, 2.0)
            payload["recency_half_life_days"] = min(
                2200.0,
                max(14.0, base.recency_half_life_days * memory_scales[variant_index]),
            )
        elif narrow_reversal_bracket:
            template = normalize_program(margin_templates[
                # Test disjoint pairs on adjacent generations.  The +2 keeps
                # an already-running even generation's (2, 3) evidence from
                # being repeated immediately after this upgrade.
                (generation * 2 + 2 + variant_index) % len(margin_templates)
            ])
            programs = [dict(program) for program in base.feature_programs]
            template_key = program_name(template)
            programs = [
                program for program in programs
                if program_name(program) != template_key
            ]
            programs.append(template)
            payload["feature_programs"] = programs[-10:]
            payload["features"] = sorted(
                set(base.features) | {template["left"], template["right"]}
            )
        payload.update({
            "confidence_quantile": max(0.0, min(.30, quantile)),
            "calibration_safety": base.calibration_safety,
            "generation": generation, "parents": [base.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        variant = Genome(**payload).finalize()
        if variant.genome_id not in known:
            known.add(variant.genome_id)
            variants.append(variant)
    if variants:
        # Pool genes are heritable and can become ubiquitous. Reserve one
        # fresh pool topology without allowing that ubiquity to block every
        # targeted coverage repair slot.
        emergent_novel = next((
            index for index, genome in enumerate(population)
            if index > 0 and genome.fitness is None and genome.emergent_pools
        ), None)
        protected = ({emergent_novel} if emergent_novel is not None else set())
        replaceable = [
            index for index in range(len(population) - 1, 0, -1)
            if index not in protected and population[index].fitness is None
        ]
        if len(replaceable) < len(variants):
            replaceable.extend(
                index for index in range(len(population) - 1, 0, -1)
                if index not in protected and index not in replaceable
            )
        for index, variant in zip(replaceable, variants):
            population[index] = variant
    return population


def introduce_extra_trees_coverage_variant(
    population: list[Genome], frontier: Genome | None, generation: int,
    reversal_frontier: Genome | None = None,
    protected_parent_ids: set[str] | None = None,
    evidence: Sequence[Genome] = (),
    evaluation_id: str | None = None,
) -> list[Genome]:
    """Spend one slot expanding a strong nonlinear tree specialist."""
    if extra_trees_frontier_rank(frontier) is None:
        return population
    historical_frontier = frontier
    current_baseline: Genome | None = None
    stale_frontier = bool(
        evaluation_id
        and (frontier.result or {}).get("evaluation_signature") != evaluation_id
    )
    if stale_frontier:
        frontier_key = genome_evaluation_key(frontier)
        current_baseline = next((
            genome for genome in evidence
            if genome.fitness is not None
            and (genome.result or {}).get("evaluation_signature") == evaluation_id
            and genome_evaluation_key(genome) == frontier_key
        ), None)
        if current_baseline is not None:
            # A coordinate experiment is causal only when its upper endpoint
            # was measured on the same data. If exact revalidation no longer
            # qualifies as a profitable research frontier, retire the stale
            # lane instead of attributing dataset drift to confidence q.
            if extra_trees_frontier_rank(current_baseline) is None:
                return population
            frontier = current_baseline
    if (evaluation_id and reversal_frontier is not None
            and (reversal_frontier.result or {}).get(
                "evaluation_signature"
            ) != evaluation_id):
        reversal_frontier = None
    summary = (frontier.result or {}).get("summary", {})
    coverage = float(summary.get("min_coverage", 0))
    accuracy = float(summary.get("min_accuracy", 0))
    signed_keys = {
        genome_evaluation_key(genome) for genome in evidence
        if genome.fitness is not None
    }
    revalidate_stale_frontier = stale_frontier and current_baseline is None
    high_accuracy_quantile: float | None = None
    if (not revalidate_stale_frontier and reversal_frontier is None
            and accuracy >= .62
            and coverage < PRESCREEN["coverage"] - .01):
        # A large gap-derived threshold jump can erase the very directional
        # edge this frontier exists to preserve. Trace a short monotonic curve
        # and consume each signed phenotype once.
        for offset in (.01, .02, .04, .06):
            trial_quantile = max(0.0, frontier.confidence_quantile - offset)
            probe = asdict(frontier)
            probe.update({
                "confidence_quantile": trial_quantile,
                "fitness": None, "result": None, "genome_id": "",
            })
            if genome_evaluation_key(Genome(**probe).finalize()) not in signed_keys:
                high_accuracy_quantile = trial_quantile
                break
        if high_accuracy_quantile is None:
            return population
    use_reliability_rank = (
        not revalidate_stale_frontier
        and reversal_frontier is None
        and coverage >= PRESCREEN["coverage"] - .01
        and not frontier.calibration_reliability
    )
    reliability_trial: tuple[int, str, float] | None = None
    use_ranked_tie_break = False
    ranked_quantile: float | None = None
    if use_reliability_rank:
        # First isolate how much of the calibration correctness rank may be
        # trusted before changing its model family or feature scope. Continue
        # only after the prior phenotype has a signed protected result.
        for trial in (
            (1, "core", .30), (1, "core", .10),
            (2, "core", .10), (3, "trend_regime", .10),
            (3, "flow_news", .10),
        ):
            version, pool, trial_quantile = trial
            probe = asdict(frontier)
            probe.update({
                "confidence_quantile": trial_quantile,
                "calibration_reliability": True,
                "calibration_reliability_version": version,
                "calibration_reliability_pool": pool,
                "calibration_reliability_decay": 0.0,
                "fitness": None, "result": None, "genome_id": "",
            })
            if genome_evaluation_key(Genome(**probe).finalize()) not in signed_keys:
                reliability_trial = trial
                break
        if reliability_trial is None:
            # The calibration and protected score distributions can shift, so
            # a nominal 40% rejection rate need not yield 60% protected
            # coverage. Build a short signed response curve while changing no
            # direction-model coordinate. The exact parent quantile is the
            # critical ablation; two bounded lower points remain if it still
            # misses the floor.
            for trial_quantile in dict.fromkeys((
                1.0 - PRESCREEN["coverage"],
                frontier.confidence_quantile,
                max(0.0, frontier.confidence_quantile - .01),
                max(0.0, frontier.confidence_quantile - .02),
            )):
                probe = asdict(frontier)
                probe.update({
                    "learner_kind": "extra_trees_ranked",
                    "confidence_quantile": trial_quantile,
                    "calibration_reliability": False,
                    "calibration_reliability_version": 0,
                    "calibration_reliability_pool": "core",
                    "calibration_reliability_decay": 0.0,
                    "fitness": None, "result": None, "genome_id": "",
                })
                if genome_evaluation_key(Genome(**probe).finalize()) not in signed_keys:
                    ranked_quantile = trial_quantile
                    break
            if ranked_quantile is None:
                return population
            use_ranked_tie_break = True
    if revalidate_stale_frontier:
        quantile = historical_frontier.confidence_quantile
    elif high_accuracy_quantile is not None:
        quantile = high_accuracy_quantile
    elif (reversal_frontier is not None
            and reversal_frontier.confidence_quantile < frontier.confidence_quantile
            and genome_structure_key(reversal_frontier) == genome_structure_key(frontier)):
        quantile = (
            reversal_frontier.confidence_quantile + frontier.confidence_quantile
        ) / 2.0
    else:
        coverage_gap = max(.001, PRESCREEN["coverage"] - coverage)
        phase = (.75, 1.0, 1.25)[generation % 3]
        offset = min(.06, max(.01, coverage_gap * .25)) * phase
        quantile = frontier.confidence_quantile - offset
    if (not revalidate_stale_frontier and reversal_frontier is not None
            and frontier.confidence_quantile - reversal_frontier.confidence_quantile <= .001):
        reversal_summary = (reversal_frontier.result or {}).get("summary", {})
        reversal_behavior = tuple(round(float(reversal_summary.get(key, 0)), 9) for key in (
            "min_accuracy", "min_balanced_accuracy", "min_mcc",
            "min_coverage", "min_acted_observations", "min_expectancy",
            "min_profit_factor",
        ))
        same_behavior = sum(
            1 for genome in evidence
            if genome.fitness is not None
            and genome_structure_key(genome) == genome_structure_key(frontier)
            and tuple(round(float((genome.result or {}).get("summary", {}).get(key, 0)), 9)
                      for key in (
                          "min_accuracy", "min_balanced_accuracy", "min_mcc",
                          "min_coverage", "min_acted_observations", "min_expectancy",
                          "min_profit_factor",
                      )) == reversal_behavior
        )
        if same_behavior >= 2:
            return population
    payload = asdict(historical_frontier if revalidate_stale_frontier else frontier)
    payload.update({
        "learner_kind": (
            "extra_trees_ranked" if use_ranked_tie_break
            else frontier.learner_kind
        ),
        "confidence_quantile": (
            ranked_quantile if use_ranked_tie_break else
            reliability_trial[2] if reliability_trial
            else max(0.0, min(.30, quantile))
        ),
        "calibration_reliability": (
            False if use_ranked_tie_break else
            True if use_reliability_rank else frontier.calibration_reliability
        ),
        "calibration_reliability_version": (
            0 if use_ranked_tie_break else
            reliability_trial[0] if reliability_trial
            else frontier.calibration_reliability_version
        ),
        "calibration_reliability_pool": (
            "core" if use_ranked_tie_break else
            reliability_trial[1] if reliability_trial
            else frontier.calibration_reliability_pool
        ),
        "calibration_reliability_decay": (
            0.0 if use_reliability_rank else frontier.calibration_reliability_decay
        ),
        "generation": generation, "parents": [historical_frontier.genome_id],
        "fitness": None, "result": None, "genome_id": "",
    })
    variant = Genome(**payload).finalize()
    if variant.genome_id in {genome.genome_id for genome in population}:
        return population
    protected_parent_ids = protected_parent_ids or set()
    replacement = next((
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and not (set(population[index].parents) & protected_parent_ids)
    ), None)
    if replacement is None:
        return population
    population[replacement] = variant
    return population


def introduce_primary_coverage_variant(
    population: list[Genome], frontier: Genome | None,
    evidence: Sequence[Genome], generation: int,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Give the strongest non-tree coverage frontier its own repair lane.

    The general repair path may deliberately switch to a multiscale boundary
    once one exists. Without this independent lane, that switch silently
    starved a stronger ordinary regressor that was profitable and only a few
    observations short of prescreen coverage.
    """
    if (coverage_frontier_rank(frontier) is None
            or frontier is None
            or frontier.learner_kind in {"extra_trees", "multiscale_regressor"}):
        return population
    def coverage_family_key(genome: Genome) -> str:
        payload = asdict(genome)
        for key in (
            "genome_id", "fitness", "result", "generation", "parents",
            "confidence_quantile", "learner_kind",
        ):
            payload.pop(key, None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    summary = (frontier.result or {}).get("summary", {})
    lower_evidence = sorted(
        (
            genome for genome in evidence
            if genome_structure_key(genome) == genome_structure_key(frontier)
            and genome.confidence_quantile < frontier.confidence_quantile
            and genome.fitness is not None
        ),
        key=lambda genome: genome.confidence_quantile,
        reverse=True,
    )
    quantiles: list[float] = []
    plateau_fields = (
        "min_accuracy", "min_balanced_accuracy", "min_mcc", "min_coverage",
        "min_acted_observations", "min_expectancy", "min_profit_factor",
    )
    plateau_evidence = [
        genome for genome in lower_evidence
        if all(abs(
            float(((genome.result or {}).get("summary") or {}).get(field, 0))
            - float(summary.get(field, 0))
        ) < 1e-9 for field in plateau_fields)
    ]
    upper_quantile = min([
        frontier.confidence_quantile,
        *(genome.confidence_quantile for genome in plateau_evidence),
    ])
    reversal = next((
        genome for genome in lower_evidence
        if genome.confidence_quantile < upper_quantile
        and float(((genome.result or {}).get("summary") or {}).get(
            "min_coverage", 0
        )) >= PRESCREEN["coverage"]
        and (
            float(((genome.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )) < PRESCREEN["accuracy"]
            or float(((genome.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )) < PRESCREEN["profit_factor"]
        )
    ), None)
    reversal_plateau_evidence: list[Genome] = []
    if reversal is not None:
        reversal_summary = (reversal.result or {}).get("summary", {})
        reversal_plateau_evidence = [
            genome for genome in lower_evidence
            if float(((genome.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )) >= PRESCREEN["coverage"]
            and all(abs(
                float(((genome.result or {}).get("summary") or {}).get(field, 0))
                - float(reversal_summary.get(field, 0))
            ) < 1e-9 for field in plateau_fields)
        ]
    proposal_base = frontier
    if reversal is not None:
        ranked_variants = sorted((
            genome for genome in evidence
            if genome.fitness is not None
            and genome.learner_kind == "continuous_rank_regressor"
            and coverage_family_key(genome) == coverage_family_key(frontier)
        ), key=lambda genome: (genome.confidence_quantile, genome.generation))
        ranked_variant_seen = any(
            genome.learner_kind == "continuous_rank_regressor"
            and frontier.genome_id in genome.parents
            for genome in evidence
        )
        if (frontier.learner_kind == "regressor"
                and (plateau_evidence or len(reversal_plateau_evidence) >= 2)
                and not ranked_variant_seen):
            # Signed evidence proves the ordinary regressor has a discrete
            # confidence plateau. It may be visible either above the coverage
            # boundary or as repeated identical reversals below it; requiring
            # only the former caused endless scalar bisection toward the same
            # floating-point endpoint. Preserve direction/fit exactly and test
            # a causal within-leaf rank before mutating more coordinates.
            payload = asdict(frontier)
            payload.update({
                "learner_kind": "continuous_rank_regressor",
                "generation": generation, "parents": [frontier.genome_id],
                "fitness": None, "result": None, "genome_id": "",
            })
            variant = Genome(**payload).finalize()
            known_keys = {
                genome_evaluation_key(genome) for genome in [*population, *evidence]
            }
            if genome_evaluation_key(variant) not in known_keys:
                protected_parent_ids = protected_parent_ids or set()
                replacement = next((
                    index for index in range(len(population) - 1, 0, -1)
                    if population[index].fitness is None
                    and not (set(population[index].parents) & protected_parent_ids)
                ), None)
                if replacement is not None:
                    population[replacement] = variant
                return population
        ranked_underfloor = [
            genome for genome in ranked_variants
            if float(((genome.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )) < PRESCREEN["coverage"]
        ]
        ranked_overfloor_failure = [
            genome for genome in ranked_variants
            if float(((genome.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )) >= PRESCREEN["coverage"]
            and (
                float(((genome.result or {}).get("summary") or {}).get(
                    "min_accuracy", 0
                )) < PRESCREEN["accuracy"]
                or float(((genome.result or {}).get("summary") or {}).get(
                    "min_profit_factor", 0
                )) < PRESCREEN["profit_factor"]
            )
        ]
        if ranked_overfloor_failure and not ranked_underfloor:
            # The same threshold failed after within-leaf ranking, so the
            # discontinuity is not a score-tie artifact. Continuing scalar
            # bisection would preserve the harmful acted set. Spend the lane
            # on one deterministic causal margin interaction instead. The
            # original feature core, fit hyperparameters, and gate threshold
            # remain intact; protected folds decide whether the new ordering
            # generalizes.
            margin_templates = (
                {"op": "tanh_mix",
                 "left": "asset_news_sentiment_acceleration",
                 "right": "flow_divergence", "scale": 1.0},
                {"op": "signed_sqrt_product",
                 "left": "market_breadth_r6",
                 "right": "relative_market_r6", "scale": 1.0},
                {"op": "regime_gate",
                 "left": "news_negative_share_24h",
                 "right": "volatility_ratio", "scale": 1.0},
                {"op": "abs_gap", "left": "futures_spot_basis",
                 "right": "flow_imbalance", "scale": 1.0},
            )
            known_keys = {
                genome_evaluation_key(genome)
                for genome in [*population, *evidence]
            }
            evaluated_keys = {
                genome_evaluation_key(genome)
                for genome in [*population, *evidence]
                if genome.fitness is not None
            }
            margin_trial_keys: list[str] = []
            for raw_template in margin_templates:
                template = normalize_program(raw_template)
                template_key = program_name(template)
                payload = asdict(frontier)
                programs = [
                    dict(program) for program in frontier.feature_programs
                    if program_name(program) != template_key
                ]
                programs.append(template)
                payload.update({
                    "learner_kind": "regressor",
                    "feature_programs": programs[-10:],
                    "features": sorted(set(frontier.features) | {
                        template["left"], template["right"],
                    }),
                    "confidence_quantile": frontier.confidence_quantile,
                    "generation": generation,
                    "parents": [
                        frontier.genome_id,
                        max(ranked_overfloor_failure,
                            key=lambda genome: genome.generation).genome_id,
                    ],
                    "fitness": None, "result": None, "genome_id": "",
                })
                proposal = Genome(**payload).finalize()
                proposal_key = genome_evaluation_key(proposal)
                margin_trial_keys.append(proposal_key)
                if proposal_key in known_keys:
                    continue
                protected_parent_ids = protected_parent_ids or set()
                replacement = next((
                    index for index in range(len(population) - 1, 0, -1)
                    if population[index].fitness is None
                    and not (set(population[index].parents)
                             & protected_parent_ids)
                ), None)
                if replacement is not None:
                    population[replacement] = proposal
                return population
            if not all(key in evaluated_keys for key in margin_trial_keys):
                # A structurally reserved probe still lacks signed evidence.
                # Never leapfrog it merely because its phenotype is known.
                return population
            # Every bounded margin interaction has now produced evidence. A
            # return-tree leaf can ignore an added observation completely, so
            # returning to its scalar confidence boundary would recreate the
            # disproven reversal. Hold the original observations, programs,
            # threshold and fit coordinates fixed while escalating only the
            # learner architecture. Once these finite ablations are exhausted,
            # yield the protected lane to general search instead of cycling.
            ranked_parent = max(
                ranked_overfloor_failure, key=lambda genome: genome.generation
            )
            margin_trials = [
                genome for genome in evidence
                if genome.fitness is not None
                and frontier.genome_id in genome.parents
                and ranked_parent.genome_id in genome.parents
                and genome.learner_kind == "regressor"
                and genome.confidence_quantile == frontier.confidence_quantile
            ]
            causal_parent_ids = [frontier.genome_id, ranked_parent.genome_id]
            if margin_trials:
                causal_parent_ids.append(max(
                    margin_trials, key=lambda genome: genome.generation
                ).genome_id)
            for learner_kind in (
                "decomposed_regressor", "extra_trees_regressor",
                "multiscale_regressor",
            ):
                payload = asdict(frontier)
                payload.update({
                    "learner_kind": learner_kind,
                    "confidence_quantile": frontier.confidence_quantile,
                    "generation": generation,
                    "parents": causal_parent_ids,
                    "fitness": None, "result": None, "genome_id": "",
                })
                proposal = Genome(**payload).finalize()
                proposal_key = genome_evaluation_key(proposal)
                if proposal_key in known_keys:
                    if proposal_key not in evaluated_keys:
                        return population
                    continue
                protected_parent_ids = protected_parent_ids or set()
                replacement = next((
                    index for index in range(len(population) - 1, 0, -1)
                    if population[index].fitness is None
                    and not (set(population[index].parents)
                             & protected_parent_ids)
                ), None)
                if replacement is not None:
                    population[replacement] = proposal
                return population
            # Direction, scalar threshold, causal margin interactions, and the
            # finite return architectures have all produced signed failures.
            # The remaining causal degree of freedom is WHEN to trust the
            # original direction. Learn correctness only from the fold's
            # calibration prefix and use it solely for abstention ranking.
            # This cannot flip a prediction, see protected outcomes, or bypass
            # any coverage/profit/calibration/unseen-asset gate.
            reliability_trials = (
                (1, "core"),
                (2, "core"),
                (3, "trend_regime"),
                (3, "flow_news"),
            )
            for reliability_version, reliability_pool in reliability_trials:
                payload = asdict(frontier)
                payload.update({
                    "learner_kind": "continuous_rank_regressor",
                    "confidence_quantile": .30,
                    "calibration_reliability": True,
                    "calibration_reliability_version": reliability_version,
                    "calibration_reliability_pool": reliability_pool,
                    "calibration_reliability_decay": 0.0,
                    "generation": generation,
                    "parents": causal_parent_ids,
                    "fitness": None, "result": None, "genome_id": "",
                })
                proposal = Genome(**payload).finalize()
                proposal_key = genome_evaluation_key(proposal)
                if proposal_key in known_keys:
                    if proposal_key not in evaluated_keys:
                        return population
                    continue
                protected_parent_ids = protected_parent_ids or set()
                replacement = next((
                    index for index in range(len(population) - 1, 0, -1)
                    if population[index].fitness is None
                    and not (set(population[index].parents)
                             & protected_parent_ids)
                ), None)
                if replacement is not None:
                    population[replacement] = proposal
                return population
            return population
        if ranked_underfloor:
            # Continue threshold bisection inside the ranked species. Reverting
            # to the ordinary regressor here would recreate the leaf plateau
            # that justified this repair and make the first ablation a dead end.
            proposal_base = min(
                ranked_underfloor, key=lambda genome: genome.confidence_quantile
            )
            lower_bound = max(
                [
                    genome.confidence_quantile
                    for genome in ranked_overfloor_failure
                    if genome.confidence_quantile < proposal_base.confidence_quantile
                ],
                default=reversal.confidence_quantile,
            )
            quantiles.append(
                (proposal_base.confidence_quantile + lower_bound) / 2.0
            )
        else:
            quantiles.append((upper_quantile + reversal.confidence_quantile) / 2.0)
    else:
        base = (min(plateau_evidence, key=lambda genome: genome.confidence_quantile)
                if plateau_evidence else
                lower_evidence[0] if lower_evidence else frontier)
        base_summary = (base.result or {}).get("summary", {})
        coverage_gap = max(
            .001,
            PRESCREEN["coverage"] - float(base_summary.get("min_coverage", 0)),
        )
        phase = (.75, 1.0, 1.25)[generation % 3]
        step = max(.0025, min(.02, coverage_gap * 1.5)) * phase
        if lower_evidence:
            frontier_coverage = float(summary.get("min_coverage", 0))
            base_coverage = float(base_summary.get("min_coverage", 0))
            if abs(base_coverage - frontier_coverage) < 1e-6:
                # Quantile movement can remain inside one tree-leaf score
                # plateau, producing an identical acted set. Escape that
                # plateau deliberately instead of spending generations on
                # numerically different but behaviorally identical genomes.
                prior_step = max(
                    .0, frontier.confidence_quantile - base.confidence_quantile
                )
                step = max(.01, min(.03, prior_step * 2.0))
        quantiles.append(base.confidence_quantile - step)
    known_keys = {
        genome_evaluation_key(genome) for genome in [*population, *evidence]
    }
    variant: Genome | None = None
    for quantile in quantiles:
        payload = asdict(proposal_base)
        parents = list(dict.fromkeys([
            proposal_base.genome_id, frontier.genome_id,
        ]))
        payload.update({
            "confidence_quantile": max(0.0, min(.30, quantile)),
            # Keep the immediate causal parent and the durable frontier. Later
            # reserved lanes protect frontier parent IDs; omitting the second
            # identity lets an otherwise valid continuation be overwritten.
            "generation": generation, "parents": parents,
            "fitness": None, "result": None, "genome_id": "",
        })
        proposal = Genome(**payload).finalize()
        if genome_evaluation_key(proposal) not in known_keys:
            variant = proposal
            break
    if variant is None:
        return population
    protected_parent_ids = protected_parent_ids or set()
    replacement = next((
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and not (set(population[index].parents) & protected_parent_ids)
    ), None)
    if replacement is not None:
        population[replacement] = variant
    return population


def introduce_champion_coordinate_variant(
    population: list[Genome], champion: Genome | None,
    evidence: Sequence[Genome], generation: int,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Spend one slot on a controlled descendant of the full-fold champion.

    Frontier repair deliberately explores high-accuracy one-fold lineages, but
    it must not consume the whole population after their temporal reversal is
    well established.  This search changes exactly one continuous or bounded
    tree coordinate at a time and remembers every signed phenotype result.
    """
    if champion is None or champion.fitness is None or len(population) < 4:
        return population
    result = champion.result or {}
    if int(result.get("evaluated_folds", 0)) < int(result.get("requested_folds", 3)):
        return population
    schedules: list[tuple[str, float | int]] = [
        ("confidence_quantile", max(0.0, champion.confidence_quantile - .01)),
        ("confidence_quantile", min(.30, champion.confidence_quantile + .01)),
        ("confidence_quantile", max(0.0, champion.confidence_quantile - .02)),
        ("confidence_quantile", min(.30, champion.confidence_quantile + .02)),
        ("min_samples_leaf", max(8, champion.min_samples_leaf - 4)),
        ("min_samples_leaf", min(100, champion.min_samples_leaf + 4)),
        ("max_leaf_nodes", max(8, champion.max_leaf_nodes - 4)),
        ("max_leaf_nodes", min(72, champion.max_leaf_nodes + 4)),
        ("recency_half_life_days", max(45.0, champion.recency_half_life_days * .75)),
        ("recency_half_life_days", min(2200.0, champion.recency_half_life_days * 1.25)),
        ("calibration_safety", max(1.0, champion.calibration_safety * .75)),
        ("calibration_safety", min(12.0, champion.calibration_safety * 1.25)),
    ]
    champion_summary = result.get("summary", {})
    tradeoff_refinements: list[tuple[float, float, float]] = []
    for observed in evidence:
        observed_result = observed.result or {}
        if (int(observed_result.get("evaluated_folds", 0))
                < int(observed_result.get("requested_folds", 3))):
            continue
        # Accept only a true one-coordinate observation. Direct lineage alone
        # is insufficient because migration/capping can rewrite other genes.
        comparison = asdict(champion)
        comparison.update({
            "confidence_quantile": observed.confidence_quantile,
            "generation": observed.generation, "parents": observed.parents,
            "fitness": observed.fitness, "result": observed.result,
            "genome_id": observed.genome_id,
        })
        if genome_evaluation_key(Genome(**comparison)) != genome_evaluation_key(observed):
            continue
        observed_summary = observed_result.get("summary", {})
        accuracy_gain = float(observed_summary.get("min_accuracy", 0)) - float(
            champion_summary.get("min_accuracy", 0)
        )
        economics_worse = (
            float(observed_summary.get("min_profit_factor", -math.inf))
            < float(champion_summary.get("min_profit_factor", -math.inf))
            or float(observed_summary.get("min_expectancy", -math.inf))
            < float(champion_summary.get("min_expectancy", -math.inf))
        )
        gap = abs(observed.confidence_quantile - champion.confidence_quantile)
        if accuracy_gain > 0 and economics_worse and gap >= .0025:
            midpoint = (
                observed.confidence_quantile + champion.confidence_quantile
            ) / 2.0
            mirrored = max(0.0, min(
                .30,
                champion.confidence_quantile
                + (champion.confidence_quantile
                   - observed.confidence_quantile),
            ))
            tradeoff_refinements.append((gap, mirrored, midpoint))
    # First mirror across the champion to establish the local economics slope;
    # then bisect the accuracy/economics boundary. Known-phenotype filtering
    # advances automatically from the mirror to the midpoint on the next turn.
    # This seeks PF recovery before spending another evaluation farther into a
    # direction already proven to worsen trade economics.
    adaptive: list[tuple[str, float | int]] = []
    for _, mirrored, midpoint in sorted(tradeoff_refinements):
        adaptive.extend((
            ("confidence_quantile", mirrored),
            ("confidence_quantile", midpoint),
        ))
    schedules = adaptive + schedules
    known_keys = {
        genome_evaluation_key(genome)
        for genome in [*population, *evidence]
        if genome.fitness is not None or genome in population
    }
    variant: Genome | None = None
    for field_name, value in schedules:
        if value == getattr(champion, field_name):
            continue
        payload = asdict(champion)
        payload.update({
            field_name: value, "generation": generation,
            "parents": [champion.genome_id], "fitness": None,
            "result": None, "genome_id": "",
        })
        proposal = Genome(**payload).finalize()
        if genome_evaluation_key(proposal) not in known_keys:
            variant = proposal
            break
    if variant is None:
        return population
    protected_parent_ids = protected_parent_ids or set()
    replaceable = [
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and not (set(population[index].parents) & protected_parent_ids)
    ]
    # Prefer an ordinary child, but do not silently abandon the reserved
    # full-fold search merely because crossover copied emergent topology into
    # every unprotected child. One emergent experiment is a smaller cost than
    # another generation with no controlled descendant of the only candidate
    # that has survived every requested fold.
    replacement = next((
        index for index in replaceable
        if not population[index].emergent_pools
    ), replaceable[0] if replaceable else None)
    if replacement is None:
        return population
    population[replacement] = variant
    return population


def introduce_champion_profit_program_variant(
    population: list[Genome], champion: Genome | None,
    profitable_frontier: Genome | None, evidence: Sequence[Genome],
    generation: int, protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Transfer one profitable causal program into the full-fold champion.

    A one-fold frontier is never eligible for promotion, but its exclusive
    feature programs are useful hypotheses. Testing one program at a time on
    the fully validated champion separates causal transfer from wholesale
    crossover and keeps every fold, coverage, and economics gate authoritative.
    """
    if (champion is None or profitable_frontier is None
            or champion.fitness is None or len(population) < 6):
        return population
    champion_result = champion.result or {}
    if int(champion_result.get("evaluated_folds", 0)) < int(
        champion_result.get("requested_folds", 3)
    ):
        return population
    frontier_summary = (profitable_frontier.result or {}).get("summary", {})
    if (float(frontier_summary.get("min_profit_factor", 0)) <= 1.0
            or float(frontier_summary.get("min_expectancy", -math.inf)) <= 0):
        return population
    champion_programs = {
        program_name(program) for program in champion.feature_programs
    }
    champion_summary = champion_result.get("summary", {})
    rejected_programs: set[str] = set()
    tradeoff_programs: dict[str, dict[str, Any]] = {}
    for observed in evidence:
        observed_programs = {
            program_name(program) for program in observed.feature_programs
        }
        added = observed_programs - champion_programs
        observed_summary = (observed.result or {}).get("summary", {})
        if (len(added) == 1
                and float(observed_summary.get("min_accuracy", 0))
                    <= float(champion_summary.get("min_accuracy", 0))
                and float(observed_summary.get("min_profit_factor", 0))
                    <= float(champion_summary.get("min_profit_factor", 0))):
            rejected_programs.update(added)
        elif (len(added) == 1
                and float(observed_summary.get("min_accuracy", 0))
                    > float(champion_summary.get("min_accuracy", 0))
                and float(observed_summary.get("min_profit_factor", 0))
                    < float(champion_summary.get("min_profit_factor", 0))):
            added_name = next(iter(added))
            tradeoff_programs[added_name] = next(
                program for program in observed.feature_programs
                if program_name(program) == added_name
            )
    hypotheses = [
        normalize_program(program)
        for program in profitable_frontier.feature_programs
        if program_name(program) not in champion_programs
        and program_name(program) not in rejected_programs
    ]
    known_keys = {
        genome_evaluation_key(genome)
        for genome in [*population, *evidence]
        if genome.fitness is not None or genome in population
    }
    variant: Genome | None = None
    # A transferred program that improves all-fold accuracy but hurts
    # economics may still contain useful signal with an unprofitable low-
    # confidence tail. Raise its threshold in bounded steps before discarding
    # the feature, while phenotype evidence prevents repeated evaluations.
    for added_name, program in tradeoff_programs.items():
        if added_name in rejected_programs:
            continue
        for offset in (.01, .02, .03):
            payload = asdict(champion)
            payload.update({
                "feature_programs": [*champion.feature_programs, program][-10:],
                "confidence_quantile": min(
                    .30, champion.confidence_quantile + offset
                ),
                "generation": generation, "parents": [champion.genome_id],
                "fitness": None, "result": None, "genome_id": "",
            })
            proposal = Genome(**payload).finalize()
            if genome_evaluation_key(proposal) not in known_keys:
                variant = proposal
                break
        if variant is not None:
            break
    for program in hypotheses:
        if variant is not None:
            break
        payload = asdict(champion)
        payload.update({
            "feature_programs": [*champion.feature_programs, program][-10:],
            "generation": generation, "parents": [champion.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        proposal = Genome(**payload).finalize()
        if genome_evaluation_key(proposal) not in known_keys:
            variant = proposal
            break
    if variant is None:
        return population
    protected_parent_ids = protected_parent_ids or set()
    parent_counts = Counter(
        parent
        for genome in population if genome.fitness is None
        for parent in genome.parents
    )
    ordinary = [
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and champion.genome_id not in population[index].parents
        and not (set(population[index].parents) & protected_parent_ids)
    ]
    # If every slot is protected, replace only a duplicate repair lineage and
    # leave at least one descendant of that frontier in the generation.
    duplicate = [
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and champion.genome_id not in population[index].parents
        and any(parent_counts[parent] > 1 for parent in population[index].parents)
    ]
    replacement = ordinary[0] if ordinary else (duplicate[0] if duplicate else None)
    if replacement is not None:
        population[replacement] = variant
    return population


def introduce_champion_profit_program_from_frontiers(
    population: list[Genome], champion: Genome | None,
    profitable_frontiers: Sequence[Genome | None], evidence: Sequence[Genome],
    generation: int, protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Try profitable frontiers in rank order, reserving exactly one slot."""
    for frontier in profitable_frontiers:
        before = [genome.genome_id for genome in population]
        population = introduce_champion_profit_program_variant(
            population, champion, frontier, evidence, generation,
            protected_parent_ids,
        )
        if [genome.genome_id for genome in population] != before:
            break
    return population


def introduce_champion_return_tree_variant(
    population: list[Genome], champion: Genome | None,
    evidence: Sequence[Genome], generation: int,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Test return-magnitude ranking after program hypotheses are exhausted."""
    if (champion is None or champion.fitness is None
            or champion.learner_kind != "extra_trees" or len(population) < 6):
        return population
    result = champion.result or {}
    if int(result.get("evaluated_folds", 0)) < int(result.get("requested_folds", 3)):
        return population
    known_keys = {
        genome_evaluation_key(genome)
        for genome in [*population, *evidence]
        if genome.fitness is not None or genome in population
    }
    quantile_trials: list[tuple[Genome, float]] = []
    return_evidence = sorted(
        (genome for genome in evidence
         if genome.learner_kind in {
             "extra_trees_regressor", "extra_trees_hybrid",
         }
         and genome.fitness is not None),
        key=lambda genome: (
            float(((genome.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
            float(((genome.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
        ),
        reverse=True,
    )
    evidence_quality: list[tuple[Genome, bool]] = []
    for observed in return_evidence:
        summary = (observed.result or {}).get("summary", {})
        strong_signal = (
            float(summary.get("min_accuracy", 0)) >= .56
            and float(summary.get("min_balanced_accuracy", 0)) >= .55
            and float(summary.get("min_mcc", -1)) >= .10
            and float(summary.get("min_profit_factor", 0)) >= .95
        )
        evidence_quality.append((observed, strong_signal))
    topology_bases = sorted(
        (
            observed for observed, _ in evidence_quality
            if .50 <= float(((observed.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )) < PRESCREEN["coverage"]
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )) >= .58
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_balanced_accuracy", 0
            )) >= .58
            and .95 <= float(((observed.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            ))
        ),
        key=lambda observed: (
            float(((observed.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )),
            float(((observed.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
            float(((observed.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
        ),
        reverse=True,
    )
    priority_topology = any(
        PRESCREEN["coverage"] - float(
            ((observed.result or {}).get("summary") or {}).get("min_coverage", 0)
        ) <= .01
        and float(((observed.result or {}).get("summary") or {}).get(
            "min_accuracy", 0
        )) < .60
        and float(((observed.result or {}).get("summary") or {}).get(
            "min_profit_factor", 0
        )) >= 1.0
        and float(((observed.result or {}).get("summary") or {}).get(
            "min_expectancy", 0
        )) > 0
        for observed in topology_bases
    )
    # A topology coordinate can improve direction and economics while moving
    # the protected-score distribution just below the coverage floor.  Once
    # that happens, test the interaction with one small cutoff recovery before
    # resuming unrelated topology mutations.  The stricter accuracy,
    # calibration, and economics requirements keep this from reopening the
    # broad cutoff descent that ``priority_topology`` deliberately stops.
    topology_recovery_bases = sorted(
        (
            observed for observed in topology_bases
            if (
                float(((observed.result or {}).get("summary") or {}).get(
                    "min_coverage", 0
                )) >= .55
                or (
                    float(((observed.result or {}).get("summary") or {}).get(
                        "min_coverage", 0
                    )) >= .54
                    and float(((observed.result or {}).get("summary") or {}).get(
                        "min_accuracy", 0
                    )) >= .60
                    and float(((observed.result or {}).get("summary") or {}).get(
                        "min_balanced_accuracy", 0
                    )) >= .60
                    and float(((observed.result or {}).get("summary") or {}).get(
                        "min_profit_factor", 0
                    )) >= 1.10
                )
            )
            and PRESCREEN["coverage"] - float(
                ((observed.result or {}).get("summary") or {}).get(
                    "min_coverage", 0
                )
            ) > .01
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )) >= .595
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_balanced_accuracy", 0
            )) >= .59
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )) >= 1.0
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_expectancy", 0
            )) > 0
            and float(((observed.result or {}).get("summary") or {}).get(
                "max_ece", 1
            )) <= PRESCREEN["ece"]
        ),
        key=lambda observed: (
            float(((observed.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
            float(((observed.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
            float(((observed.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )),
        ),
        reverse=True,
    )
    for observed in topology_recovery_bases:
        coverage = float(((observed.result or {}).get("summary") or {}).get(
            "min_coverage", 0
        ))
        recovery = min(
            .015, max(.005, (PRESCREEN["coverage"] - coverage) * .5)
        )
        quantile_trials.append((
            observed, max(0.0, observed.confidence_quantile - recovery)
        ))
    variant: Genome | None = None
    # First compose coverage search with the exact signed return-tree
    # phenotype. A threshold result from a different leaf topology is not an
    # endpoint for this curve, and rebuilding from the classifier champion
    # would silently discard the discovered return-ranking capacity.
    brackets = [] if priority_topology else [
        (strong.confidence_quantile - weak.confidence_quantile, strong,
         (strong.confidence_quantile + weak.confidence_quantile) / 2.0)
        for strong, is_strong in evidence_quality if is_strong
        for weak, weak_is_strong in evidence_quality if not weak_is_strong
        if float(((strong.result or {}).get("summary") or {}).get(
            "min_profit_factor", 0
        )) >= 1.0
        if float(((strong.result or {}).get("summary") or {}).get(
            "min_expectancy", 0
        )) > 0
        if return_tree_threshold_key(strong) == return_tree_threshold_key(weak)
        if weak.confidence_quantile < strong.confidence_quantile
        and strong.confidence_quantile - weak.confidence_quantile >= .002
    ]
    if brackets:
        _, strong, midpoint = min(brackets, key=lambda item: item[0])
        quantile_trials.append((strong, midpoint))
    for observed, strong_signal in sorted(
        evidence_quality,
        key=lambda item: tree_leaf_refinement_quality(item[0]),
        reverse=True,
    ):
        summary = (observed.result or {}).get("summary", {})
        coverage = float(summary.get("min_coverage", 0))
        if (strong_signal
                and float(summary.get("min_profit_factor", 0)) >= 1.0
                and float(summary.get("min_expectancy", 0)) > 0
                and coverage < PRESCREEN["coverage"]):
            coverage_gap = PRESCREEN["coverage"] - coverage
            # Once a profitable curve is within one coverage point but has
            # already fallen below 60% direction accuracy, another lower
            # cutoff predictably trades away the objective we are trying to
            # preserve. Hold that signed boundary and refine its ranking
            # topology before admitting still weaker signals.
            near_coverage_tradeoff = (
                coverage_gap <= .01
                and float(summary.get("min_accuracy", 0)) < .60
            )
            if not near_coverage_tradeoff and not priority_topology:
                quantile_trials.append((observed, max(
                    0.0, observed.confidence_quantile
                    - max(.02, min(.08, coverage_gap * .5)),
                )))
    for base, quantile in quantile_trials:
        payload = asdict(base)
        payload.update({
            "learner_kind": "extra_trees_regressor",
            "confidence_quantile": quantile,
            "generation": generation, "parents": [base.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        proposal = Genome(**payload).finalize()
        if genome_evaluation_key(proposal) not in known_keys:
            variant = proposal
            break
    hybrid_evidence = [
        observed for observed in return_evidence
        if observed.learner_kind == "extra_trees_hybrid"
    ]
    profitable_selective = [
        observed for observed in return_evidence
        if observed.learner_kind == "extra_trees_regressor"
        and float(((observed.result or {}).get("summary") or {}).get(
            "min_profit_factor", 0
        )) >= 1.0
        and float(((observed.result or {}).get("summary") or {}).get(
            "min_accuracy", 0
        )) >= .60
    ]
    coverage_anchors = [
        observed for observed in return_evidence
        if observed.learner_kind == "extra_trees_regressor"
        and float(((observed.result or {}).get("summary") or {}).get(
            "min_coverage", 0
        )) >= PRESCREEN["coverage"]
    ]
    # Preserve the champion's direction classifier while a separate return
    # pool ranks WHEN to act. The launch requires signed evidence for both a
    # profitable selective region and a quantile that clears coverage on the
    # same return-tree phenotype.  Keeping feature views identical makes this
    # a causal topology/cutoff fusion instead of silently changing the
    # champion direction model at the same time.
    compatible_hybrid_pairs = [
        (selective, anchor)
        for selective in profitable_selective
        for anchor in coverage_anchors
        if return_tree_threshold_key(selective) == return_tree_threshold_key(anchor)
        if selective.features == champion.features
        and selective.feature_programs == champion.feature_programs
        and anchor.features == champion.features
        and anchor.feature_programs == champion.feature_programs
    ]
    if (variant is None and not priority_topology
            and compatible_hybrid_pairs
            and not hybrid_evidence):
        selection_source, base = max(
            compatible_hybrid_pairs,
            key=lambda pair: (
                float(((pair[0].result or {}).get("summary") or {}).get(
                    "min_profit_factor", 0
                )),
                float(((pair[0].result or {}).get("summary") or {}).get(
                    "min_accuracy", 0
                )),
                float(((pair[1].result or {}).get("summary") or {}).get(
                    "min_profit_factor", 0
                )),
            ),
        )
        parents = list(dict.fromkeys((
            champion.genome_id, selection_source.genome_id, base.genome_id,
        )))
        payload = asdict(champion)
        payload.update({
            "learner_kind": "extra_trees_hybrid",
            "confidence_quantile": base.confidence_quantile,
            "selection_max_iter": selection_source.max_iter,
            "selection_max_leaf_nodes": selection_source.max_leaf_nodes,
            "selection_min_samples_leaf": selection_source.min_samples_leaf,
            "selection_recency_half_life_days": (
                selection_source.recency_half_life_days
            ),
            "generation": generation, "parents": parents,
            "fitness": None, "result": None, "genome_id": "",
        })
        proposal = Genome(**payload).finalize()
        if genome_evaluation_key(proposal) not in known_keys:
            variant = proposal
    # A large smoothing step can recover direction after a weaker intermediate
    # leaf size, but the recovered endpoint may still sit just below coverage.
    # Search the immediate leaf neighborhood on that exact cutoff before
    # changing another coordinate.  The lower neighbor is important because
    # tree ensembles are not monotonic in ``min_samples_leaf``; only searching
    # larger leaves can skip the local accuracy/coverage intersection.
    recovered_leaf_bases = sorted(
        (
            observed for observed in topology_bases
            if observed.min_samples_leaf >= champion.min_samples_leaf + 8
            and 0 < PRESCREEN["coverage"] - float(
                ((observed.result or {}).get("summary") or {}).get(
                    "min_coverage", 0
                )
            ) <= .01
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )) >= .595
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_balanced_accuracy", 0
            )) >= .59
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )) >= 1.0
            and float(((observed.result or {}).get("summary") or {}).get(
                "min_expectancy", 0
            )) > 0
            and float(((observed.result or {}).get("summary") or {}).get(
                "max_ece", 1
            )) <= PRESCREEN["ece"]
        ),
        key=lambda observed: (
            float(((observed.result or {}).get("summary") or {}).get(
                "min_accuracy", 0
            )),
            float(((observed.result or {}).get("summary") or {}).get(
                "min_coverage", 0
            )),
            float(((observed.result or {}).get("summary") or {}).get(
                "min_profit_factor", 0
            )),
        ),
        reverse=True,
    )
    if variant is None:
        for base in recovered_leaf_bases:
            for min_samples_leaf in (
                max(2, base.min_samples_leaf - 4),
                min(100, base.min_samples_leaf + 4),
            ):
                payload = asdict(base)
                payload.update({
                    "learner_kind": "extra_trees_regressor",
                    "min_samples_leaf": min_samples_leaf,
                    "generation": generation, "parents": [base.genome_id],
                    "fitness": None, "result": None, "genome_id": "",
                })
                proposal = Genome(**payload).finalize()
                if genome_evaluation_key(proposal) not in known_keys:
                    variant = proposal
                    break
            if variant is not None:
                break
    # Near the economics boundary, improve the ranking function before
    # admitting weaker signals. Smoother leaves can generalize return magnitude
    # across assets while the quantile remains fixed, making this a controlled
    # one-coordinate follow-up to the signed return-tree phenotype.
    if variant is not None:
        topology_bases = []
    for base in topology_bases:
        min_leaf_reversal = any(
            observed.min_samples_leaf > base.min_samples_leaf
            and return_tree_min_leaf_key(observed) == return_tree_min_leaf_key(base)
            and (
                float(((observed.result or {}).get("summary") or {}).get(
                    "min_accuracy", 0
                )) < .50
                or float(((observed.result or {}).get("summary") or {}).get(
                    "min_profit_factor", 0
                )) < .80
            )
            for observed in return_evidence
        )
        leaf_capacity_reversal = any(
            observed.max_leaf_nodes < base.max_leaf_nodes
            and return_tree_leaf_capacity_key(observed) == (
                return_tree_leaf_capacity_key(base)
            )
            and (
                float(((observed.result or {}).get("summary") or {}).get(
                    "min_accuracy", 0
                )) < .50
                or float(((observed.result or {}).get("summary") or {}).get(
                    "min_profit_factor", 0
                )) < .80
            )
            for observed in return_evidence
        )
        specifications = (
            *(() if min_leaf_reversal else (
                {"min_samples_leaf": min(100, base.min_samples_leaf + 4)},
                {"min_samples_leaf": min(100, base.min_samples_leaf + 12)},
            )),
            *(() if leaf_capacity_reversal else (
                {"max_leaf_nodes": max(8, base.max_leaf_nodes - 2)},
                {"max_leaf_nodes": max(8, base.max_leaf_nodes - 4)},
            )),
            {"recency_half_life_days": max(
                45.0, base.recency_half_life_days * .75
            )},
        )
        for specification in specifications:
            payload = asdict(base)
            payload.update({
                "learner_kind": "extra_trees_regressor",
                "confidence_quantile": base.confidence_quantile,
                **specification,
                "generation": generation, "parents": [base.genome_id],
                "fitness": None, "result": None, "genome_id": "",
            })
            proposal = Genome(**payload).finalize()
            if genome_evaluation_key(proposal) not in known_keys:
                variant = proposal
                break
        if variant is not None:
            break
    if variant is None:
        payload = asdict(champion)
        payload.update({
            "learner_kind": "extra_trees_regressor",
            "confidence_quantile": champion.confidence_quantile,
            "generation": generation, "parents": [champion.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        proposal = Genome(**payload).finalize()
        if genome_evaluation_key(proposal) not in known_keys:
            variant = proposal
    if variant is None:
        return population
    protected_parent_ids = protected_parent_ids or set()
    parent_counts = Counter(
        parent
        for genome in population if genome.fitness is None
        for parent in genome.parents
    )
    ordinary = [
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and champion.genome_id not in population[index].parents
        and not (set(population[index].parents) & protected_parent_ids)
    ]
    duplicate = [
        index for index in range(len(population) - 1, 0, -1)
        if population[index].fitness is None
        and champion.genome_id not in population[index].parents
        and any(parent_counts[parent] > 1 for parent in population[index].parents)
    ]
    replacement = ordinary[0] if ordinary else (duplicate[0] if duplicate else None)
    if replacement is not None:
        population[replacement] = variant
    return population


def coverage_frontier_rank(genome: Genome) -> tuple[float, ...] | None:
    """Rank near-coverage misses without discarding profitable accuracy.

    The earlier coverage-first ordering forgot a 63.6%-accurate, positive-
    expectancy lineage in favor of a 62.3%-accurate negative-expectancy one.
    Economics and directional strength now lead; adaptive repair handles the
    remaining coverage distance.
    """
    summary = (genome.result or {}).get("summary", {})
    accuracy = float(summary.get("min_accuracy", 0))
    balanced = float(summary.get("min_balanced_accuracy", 0))
    mcc = float(summary.get("min_mcc", -1))
    coverage = float(summary.get("min_coverage", 0))
    if not (accuracy >= .56 and balanced >= .52 and mcc >= .04
            and .40 <= coverage < PRESCREEN["coverage"]):
        return None
    expectancy = float(summary.get("min_expectancy", -1))
    profit = float(summary.get("min_profit_factor", 0))
    return (
        float(expectancy > 0 and profit >= 1.0),
        coverage,
        accuracy,
        expectancy,
        profit,
        balanced,
        mcc,
        float(summary.get("min_acted_observations", 0)),
        # If every observed outcome is identical, retain the higher-quantile
        # upper endpoint. Replacing it with a numerically lower descendant
        # erases the parent/child evidence needed to detect a score plateau.
        float(genome.confidence_quantile),
    )


def multiscale_frontier_rank(genome: Genome | None) -> tuple[float, ...] | None:
    """Rank accurate, profitable multiscale models awaiting coverage repair."""
    if genome is None or genome.learner_kind != "multiscale_regressor":
        return None
    summary = (genome.result or {}).get("summary", {})
    accuracy = float(summary.get("min_accuracy", 0))
    balanced = float(summary.get("min_balanced_accuracy", 0))
    mcc = float(summary.get("min_mcc", -1))
    coverage = float(summary.get("min_coverage", 0))
    expectancy = float(summary.get("min_expectancy", -1))
    profit = float(summary.get("min_profit_factor", 0))
    if not (accuracy >= .56 and balanced >= .52 and mcc >= .04
            and .40 <= coverage < PRESCREEN["coverage"]
            and expectancy > 0 and profit >= 1.0):
        return None
    return (accuracy, balanced, mcc, coverage, expectancy, profit)


def multiscale_boundary_rank(genome: Genome | None) -> tuple[float, ...] | None:
    """Rank the nearest accurate upper endpoint, independent of promotion."""
    if genome is None or genome.learner_kind != "multiscale_regressor":
        return None
    summary = (genome.result or {}).get("summary", {})
    accuracy = float(summary.get("min_accuracy", 0))
    balanced = float(summary.get("min_balanced_accuracy", 0))
    mcc = float(summary.get("min_mcc", -1))
    coverage = float(summary.get("min_coverage", 0))
    if not (accuracy >= .60 and balanced >= .58 and mcc >= .15
            and .40 <= coverage < PRESCREEN["coverage"]):
        return None
    return (
        coverage, accuracy, balanced, mcc,
        float(summary.get("min_expectancy", -1)),
        float(summary.get("min_profit_factor", 0)),
        -float(genome.confidence_quantile),
    )


def extra_trees_frontier_rank(genome: Genome | None) -> tuple[float, ...] | None:
    """Rank accurate, profitable extra-tree specialists needing coverage."""
    if genome is None or genome.learner_kind != "extra_trees":
        return None
    summary = (genome.result or {}).get("summary", {})
    accuracy = float(summary.get("min_accuracy", 0))
    balanced = float(summary.get("min_balanced_accuracy", 0))
    mcc = float(summary.get("min_mcc", -1))
    coverage = float(summary.get("min_coverage", 0))
    expectancy = float(summary.get("min_expectancy", -1))
    profit = float(summary.get("min_profit_factor", 0))
    # Preserve the strict accuracy requirement for broad coverage searches.
    # A specialist already within one point of the action floor may instead
    # qualify on strong balanced accuracy and MCC. That narrow exception lets
    # calibration-only reliability ranking recover its last few actions
    # without weakening the protected prescreen or admitting low-coverage
    # trees into the persistent repair frontier.
    accurate_enough = accuracy >= .62 or (
        coverage >= PRESCREEN["coverage"] - .01 and accuracy >= .58
    )
    if not (accurate_enough and balanced >= .58 and mcc >= .15
            and .25 <= coverage < PRESCREEN["coverage"]
            and expectancy > 0 and profit >= 1.0):
        return None
    # This is a research frontier, not an admission score. Preserve the best
    # directional lineage long enough to repair its coverage; otherwise a
    # nearly admitted but exhausted mask can permanently hide a materially
    # more accurate, still-profitable hypothesis. Coverage remains a
    # tie-breaker and every descendant must still pass the real prescreen.
    return (
        float(accuracy >= .62), accuracy, balanced, mcc,
        coverage, expectancy, profit,
    )


def load_extra_trees_frontier(state_dir: Path) -> Genome | None:
    saved = state_dir / "extra_trees_frontier.json"
    best: Genome | None = None
    best_rank: tuple[float, ...] | None = None
    if saved.is_file():
        try:
            candidate = genome_from_dict(
                json.loads(saved.read_text(encoding="utf-8"))["genome"]
            )
            best, best_rank = candidate, extra_trees_frontier_rank(candidate)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        rank = extra_trees_frontier_rank(candidate)
        if rank is not None and (best_rank is None or rank > best_rank):
            best, best_rank = candidate, rank
    if best is not None:
        atomic_json(saved, {"at": utc_now(), "rank": best_rank, "genome": asdict(best)})
    return best


def update_extra_trees_frontier(
    state_dir: Path, current: Genome | None, candidate: Genome,
) -> Genome | None:
    candidate_rank = extra_trees_frontier_rank(candidate)
    current_rank = extra_trees_frontier_rank(current)
    if candidate_rank is None or (
        current_rank is not None and candidate_rank <= current_rank
    ):
        return current
    saved = genome_from_dict(asdict(candidate))
    atomic_json(state_dir / "extra_trees_frontier.json", {
        "at": utc_now(), "rank": candidate_rank, "genome": asdict(saved),
    })
    return saved


def load_multiscale_boundary_frontier(state_dir: Path) -> Genome | None:
    saved = state_dir / "multiscale_boundary_frontier.json"
    best: Genome | None = None
    best_rank: tuple[float, ...] | None = None
    if saved.is_file():
        try:
            candidate = genome_from_dict(
                json.loads(saved.read_text(encoding="utf-8"))["genome"]
            )
            best, best_rank = candidate, multiscale_boundary_rank(candidate)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        rank = multiscale_boundary_rank(candidate)
        if rank is not None and (best_rank is None or rank > best_rank):
            best, best_rank = candidate, rank
    if best is not None:
        atomic_json(saved, {"at": utc_now(), "rank": best_rank, "genome": asdict(best)})
    return best


def update_multiscale_boundary_frontier(
    state_dir: Path, current: Genome | None, candidate: Genome,
) -> Genome | None:
    candidate_rank = multiscale_boundary_rank(candidate)
    current_rank = multiscale_boundary_rank(current)
    if candidate_rank is None or (
        current_rank is not None and candidate_rank <= current_rank
    ):
        return current
    saved = genome_from_dict(asdict(candidate))
    atomic_json(state_dir / "multiscale_boundary_frontier.json", {
        "at": utc_now(), "rank": candidate_rank, "genome": asdict(saved),
    })
    return saved


def load_multiscale_frontier(state_dir: Path) -> Genome | None:
    """Recover the best multiscale near-pass across population turnover."""
    saved = state_dir / "multiscale_frontier.json"
    best: Genome | None = None
    best_rank: tuple[float, ...] | None = None
    if saved.is_file():
        try:
            candidate = genome_from_dict(
                json.loads(saved.read_text(encoding="utf-8"))["genome"]
            )
            best, best_rank = candidate, multiscale_frontier_rank(candidate)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        rank = multiscale_frontier_rank(candidate)
        if rank is not None and (best_rank is None or rank > best_rank):
            best, best_rank = candidate, rank
    if best is not None:
        atomic_json(saved, {"at": utc_now(), "rank": best_rank, "genome": asdict(best)})
    return best


def update_multiscale_frontier(
    state_dir: Path, current: Genome | None, candidate: Genome,
) -> Genome | None:
    candidate_rank = multiscale_frontier_rank(candidate)
    current_rank = multiscale_frontier_rank(current)
    if candidate_rank is None or (
        current_rank is not None and candidate_rank <= current_rank
    ):
        return current
    saved = genome_from_dict(asdict(candidate))
    atomic_json(state_dir / "multiscale_frontier.json", {
        "at": utc_now(), "rank": candidate_rank, "genome": asdict(saved),
    })
    return saved


def genome_structure_key(genome: Genome) -> str:
    """Identify one model phenotype while ignoring lineage and action cutoff."""
    payload = asdict(genome)
    for key in (
        "genome_id", "fitness", "result", "generation", "parents",
        "confidence_quantile",
    ):
        payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_coverage_frontier(state_dir: Path) -> Genome | None:
    """Recover the best near-pass so generational turnover cannot forget it."""
    saved = state_dir / "coverage_frontier.json"
    best: Genome | None = None
    best_rank: tuple[float, ...] | None = None
    if saved.is_file():
        try:
            candidate = genome_from_dict(
                json.loads(saved.read_text(encoding="utf-8"))["genome"]
            )
            best, best_rank = candidate, coverage_frontier_rank(candidate)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    # Always rescan: ranking can improve between service versions, and a saved
    # single frontier must not hide stronger historical evidence.
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        rank = coverage_frontier_rank(candidate)
        if rank is not None and (best_rank is None or rank > best_rank):
            best, best_rank = candidate, rank
    if best is not None:
        atomic_json(saved, {"at": utc_now(), "rank": best_rank, "genome": asdict(best)})
    return best


def update_coverage_frontier(
    state_dir: Path, current: Genome | None, candidate: Genome,
) -> Genome | None:
    candidate_rank = coverage_frontier_rank(candidate)
    current_rank = coverage_frontier_rank(current) if current is not None else None
    if candidate_rank is None or (
        current_rank is not None and candidate_rank <= current_rank
    ):
        return current
    saved = genome_from_dict(asdict(candidate))
    atomic_json(state_dir / "coverage_frontier.json", {
        "at": utc_now(), "rank": candidate_rank, "genome": asdict(saved),
    })
    return saved


def regime_shift_rank(genome: Genome) -> tuple[float, ...] | None:
    """Identify a strong early edge that reverses in a later protected fold."""
    folds = (genome.result or {}).get("folds", [])
    if len(folds) < 2:
        return None
    fold_accuracy = [
        min(float(fold.get(name, {}).get("directional_accuracy", 0))
            for name in ("known_asset_future", "unseen_asset_future"))
        for fold in folds
    ]
    first, later = fold_accuracy[0], min(fold_accuracy[1:])
    if first < .58 or later >= .50:
        return None
    return (
        first, -later,
        float((genome.result or {}).get("summary", {}).get("min_mcc", -1)),
    )


def load_regime_shift_frontier(state_dir: Path) -> Genome | None:
    """Recover a cross-regime reversal after restart or population turnover."""
    saved = state_dir / "regime_shift_frontier.json"
    if saved.is_file():
        try:
            return genome_from_dict(json.loads(saved.read_text(encoding="utf-8"))["genome"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
    best: Genome | None = None
    best_rank: tuple[float, ...] | None = None
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        rank = regime_shift_rank(candidate)
        if rank is not None and (best_rank is None or rank > best_rank):
            best, best_rank = candidate, rank
    if best is not None:
        atomic_json(saved, {"at": utc_now(), "rank": best_rank, "genome": asdict(best)})
    return best


def update_regime_shift_frontier(
    state_dir: Path, current: Genome | None, candidate: Genome,
) -> Genome | None:
    candidate_rank = regime_shift_rank(candidate)
    current_rank = regime_shift_rank(current) if current is not None else None
    if candidate_rank is None or (
        current_rank is not None and candidate_rank <= current_rank
    ):
        return current
    saved = genome_from_dict(asdict(candidate))
    atomic_json(state_dir / "regime_shift_frontier.json", {
        "at": utc_now(), "rank": candidate_rank, "genome": asdict(saved),
    })
    return saved


def compatible_reversal_rank(
    genome: Genome, coverage_frontier: Genome | None,
) -> tuple[float, ...] | None:
    """Rank the nearest lower-q boundary for one exact model structure.

    Most lineages need a protected-fold reversal before a lower confidence
    threshold becomes the unsafe endpoint. Extra trees can instead cross the
    coverage floor in one step while losing their directional or economic
    edge. Retain either a profitable soft boundary or that first unsafe
    coverage crossing. This prevents widening a known-bad threshold move and
    turns the next generation into an informative bisection.
    """
    if (coverage_frontier is None
            or genome_structure_key(genome) != genome_structure_key(coverage_frontier)
            or genome.confidence_quantile >= coverage_frontier.confidence_quantile):
        return None
    reversal = regime_shift_rank(genome)
    if reversal is None and coverage_frontier.learner_kind == "extra_trees":
        summary = (genome.result or {}).get("summary", {})
        frontier_summary = (coverage_frontier.result or {}).get("summary", {})
        accuracy = float(summary.get("min_accuracy", 0))
        balanced = float(summary.get("min_balanced_accuracy", 0))
        mcc = float(summary.get("min_mcc", -1))
        coverage = float(summary.get("min_coverage", 0))
        expectancy = float(summary.get("min_expectancy", -1))
        profit = float(summary.get("min_profit_factor", 0))
        frontier_accuracy = float(frontier_summary.get("min_accuracy", 0))
        frontier_coverage = float(frontier_summary.get("min_coverage", 0))
        profitable_soft_boundary = (
            .55 <= accuracy < .62 and balanced >= .55 and mcc >= .10
            and coverage > frontier_coverage
            and expectancy > 0 and profit >= 1.0
        )
        unsafe_coverage_crossing = (
            coverage >= PRESCREEN["coverage"]
            and coverage > frontier_coverage
            and accuracy <= frontier_accuracy - .03
            and (accuracy < PRESCREEN["accuracy"]
                 or balanced < PRESCREEN["balanced_accuracy"]
                 or mcc < PRESCREEN["mcc"]
                 or expectancy <= 0
                 or profit < 1.0)
        )
        if not (profitable_soft_boundary or unsafe_coverage_crossing):
            return None
        reversal = (
            float(profitable_soft_boundary), accuracy, balanced, mcc,
            coverage, expectancy, profit,
        )
    if reversal is None:
        return None
    return (float(genome.confidence_quantile), *reversal)


def load_compatible_reversal_frontier(
    state_dir: Path, coverage_frontier: Genome | None,
) -> Genome | None:
    """Recover the tightest causal q bracket for the active coverage lineage."""
    if coverage_frontier is None:
        return None
    best: Genome | None = None
    best_rank: tuple[float, ...] | None = None
    for path in (state_dir / "candidates").glob("*.json"):
        try:
            candidate = genome_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        rank = compatible_reversal_rank(candidate, coverage_frontier)
        if rank is not None and (best_rank is None or rank > best_rank):
            best, best_rank = candidate, rank
    return best


def update_compatible_reversal_frontier(
    current: Genome | None, candidate: Genome,
    coverage_frontier: Genome | None,
) -> Genome | None:
    candidate_rank = compatible_reversal_rank(candidate, coverage_frontier)
    current_rank = (
        compatible_reversal_rank(current, coverage_frontier)
        if current is not None else None
    )
    if candidate_rank is None or (
        current_rank is not None and candidate_rank <= current_rank
    ):
        return current if current_rank is not None else None
    return genome_from_dict(asdict(candidate))


def introduce_regime_shift_variants(
    population: list[Genome], frontier: Genome | None, generation: int,
    protected_parent_ids: set[str] | None = None,
) -> list[Genome]:
    """Breed stability experiments from a verified temporal reversal.

    A short-memory/volatility pair made the protected reversal worse in the
    first live observation cycle. Rotate through progressively longer memory
    scales and observable regime routers so the service learns from that
    negative result instead of recreating the same phenotype every generation.
    """
    if frontier is None or regime_shift_rank(frontier) is None:
        return population
    phase = generation % 4
    memory_factors = ((1.5, 2.5), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0))[phase]
    regime_features = (
        "market_breadth_r6", "market_median_r6", "funding_rate", "rv24",
    )
    specifications = (
        {
            "learner_kind": "regressor",
            "recency_half_life_days": min(
                2200.0, frontier.recency_half_life_days * memory_factors[0]
            ),
            "l2_regularization": min(30.0, frontier.l2_regularization * 4.0),
        },
        {
            "learner_kind": "regime_decomposed_regressor",
            "recency_half_life_days": min(
                2200.0, frontier.recency_half_life_days * memory_factors[1]
            ),
            "regime_feature": regime_features[phase], "regime_bins": 2,
            "l2_regularization": min(30.0, frontier.l2_regularization * 4.0),
        },
    )
    # One stability probe per generation is enough to preserve this search
    # axis. Alternating species avoids letting the expensive decomposed model
    # monopolize a six-core host and doubles observe-and-adjust throughput.
    specifications = (specifications[generation % len(specifications)],)
    variants = []
    known = {genome.genome_id for genome in population}
    for specification in specifications:
        payload = asdict(frontier)
        payload.update({
            **specification, "generation": generation,
            "parents": [frontier.genome_id], "fitness": None,
            "result": None, "genome_id": "",
        })
        variant = Genome(**payload).finalize()
        if variant.genome_id not in known:
            known.add(variant.genome_id)
            variants.append(variant)
    protected_parent_ids = protected_parent_ids or set()
    # Emergent-pool genes rapidly spread through crossover. Protecting every
    # carrier can therefore occupy every replaceable slot and silently prevent
    # a verified regime-shift experiment from launching. Preserve one fresh
    # topology carrier plus all explicitly protected repair lineages.
    emergent_novel = next((
        index for index, genome in enumerate(population)
        if index > 0 and genome.fitness is None and genome.emergent_pools
    ), None)
    protected = {
        index for index, genome in enumerate(population)
        if bool(set(genome.parents) & protected_parent_ids)
        or index == emergent_novel
    }
    replaceable = [
        index for index in range(len(population) - 1, 0, -1)
        if index not in protected and population[index].fitness is None
    ]
    for index, variant in zip(replaceable, variants):
        population[index] = variant
    return population


def prioritize_pending_genomes(
    pending: Sequence[Genome], coverage_frontier: Genome | None,
    regime_shift_frontier: Genome | None,
    multiscale_frontier: Genome | None = None,
    extra_trees_frontier: Genome | None = None,
    champion: Genome | None = None,
) -> list[Genome]:
    """Evaluate the highest-information repairs first on small worker pools.

    Ordering cannot affect a candidate's isolated score, but it materially
    shortens observe-and-adjust latency when a slow regime model otherwise
    sits ahead of the active accuracy/economics frontier's descendants.
    Python's stable sort preserves breeding order within each class.
    """
    coverage_id = coverage_frontier.genome_id if coverage_frontier else None
    multiscale_id = multiscale_frontier.genome_id if multiscale_frontier else None
    tree_id = extra_trees_frontier.genome_id if extra_trees_frontier else None
    shift_id = regime_shift_frontier.genome_id if regime_shift_frontier else None
    champion_id = champion.genome_id if champion else None

    def priority(genome: Genome) -> int:
        parents = set(genome.parents)
        # A direct, single-coordinate descendant of the only full-fold winner
        # produces the cleanest signed evidence and is usually cheaper than
        # multiscale or decomposed repairs. Evaluate it before speculative
        # frontier descendants so a 30-minute feedback cycle cannot expire
        # while its most actionable experiment is still queued.
        if champion_id is not None and champion_id in parents:
            return 6
        if coverage_id is not None and coverage_id in parents:
            return 5
        if multiscale_id is not None and multiscale_id in parents:
            return 4
        if tree_id is not None and tree_id in parents:
            return 4
        if genome.learner_kind == "regime_decomposed_regressor":
            return 1
        if shift_id is not None and shift_id in parents:
            return 2
        return 3

    return sorted(pending, key=priority, reverse=True)


def cap_expensive_regime_candidates(
    population: list[Genome], generation: int,
    regime_shift_frontier: Genome | None,
) -> tuple[list[Genome], int]:
    """Keep one targeted decomposed regime probe and convert other copies.

    On the six-core runtime each decomposed candidate can consume several
    minutes. Multiple fresh copies delayed feedback without producing better
    evidence. The targeted reversal descendant is retained when present;
    converted candidates keep their features and pool topology.
    """
    candidates = [
        index for index, genome in enumerate(population)
        if genome.fitness is None
        and genome.learner_kind == "regime_decomposed_regressor"
    ]
    if not candidates:
        return population, 0
    shift_id = regime_shift_frontier.genome_id if regime_shift_frontier else None
    keep = next((
        index for index in candidates
        if shift_id is not None and shift_id in population[index].parents
    ), None)
    known = {genome.genome_id for genome in population}
    converted = 0
    for index in candidates:
        if keep is not None and index == keep:
            continue
        source = population[index]
        payload = asdict(source)
        payload.update({
            "learner_kind": "regressor", "regime_bins": 1,
            "generation": generation, "parents": [source.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        replacement = Genome(**payload).finalize()
        if replacement.genome_id in known:
            continue
        known.discard(source.genome_id)
        known.add(replacement.genome_id)
        population[index] = replacement
        converted += 1
    return population, converted


def cap_expensive_multiscale_candidates(
    population: list[Genome], generation: int,
    protected_parent_ids: set[str],
    protected_evaluation_keys: set[str] | None = None,
) -> tuple[list[Genome], int]:
    """Reserve double-model compute for active multiscale frontier research."""
    known = {genome.genome_id for genome in population}
    converted = 0
    protected_evaluation_keys = protected_evaluation_keys or set()
    protected_seen: set[tuple[tuple[str, ...], str]] = set()
    for index, source in enumerate(population):
        if source.learner_kind != "multiscale_regressor":
            continue
        protected = (
            bool(set(source.parents) & protected_parent_ids)
            or genome_evaluation_key(source) in protected_evaluation_keys
        )
        protected_key = (
            tuple(sorted(source.parents)), genome_structure_key(source),
        )
        if protected and protected_key not in protected_seen:
            protected_seen.add(protected_key)
            continue
        if source.fitness is not None:
            continue
        payload = asdict(source)
        payload.update({
            "learner_kind": "regressor", "regime_bins": 1,
            "generation": generation, "parents": [source.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        replacement = Genome(**payload).finalize()
        if replacement.genome_id in known:
            continue
        known.discard(source.genome_id)
        known.add(replacement.genome_id)
        population[index] = replacement
        converted += 1
    return population, converted


def learner_ablation_evaluation_key(
    frontier: Genome | None, learner_kind: str,
) -> str | None:
    """Identify one exact learner-only ablation independent of lineage."""
    if frontier is None:
        return None
    payload = asdict(frontier)
    payload.update({
        "learner_kind": learner_kind,
        "fitness": None, "result": None, "genome_id": "",
    })
    return genome_evaluation_key(Genome(**payload).finalize())


def introduce_regime_repair_variants(
    population: list[Genome], evaluated: Sequence[Genome], generation: int,
    rng: random.Random,
) -> list[Genome]:
    """Reserve targeted descendants of the deepest cross-regime failure.

    Random mutation is deliberately retained for exploration, but once a
    lineage reaches more protected folds than its peers its weakest fold is
    actionable experience.  These descendants test shorter causal memory and
    observable-state routing without ever fitting to the untouched final
    period or changing an admission gate.
    """
    candidates = [
        genome for genome in evaluated
        if (genome.result or {}).get("status") not in {
            "failed", "surrogate_floor_pass", "surrogate_working_target_pass",
        }
        and (genome.result or {}).get("evaluated_folds", 0) >= 2
    ]
    if not candidates or len(population) < len(LEARNER_KINDS) + 2:
        return population
    base = max(candidates, key=lambda genome: (
        int((genome.result or {}).get("evaluated_folds", 0)),
        float((genome.result or {}).get("summary", {}).get("min_accuracy", 0)),
        float(genome.fitness if genome.fitness is not None else -math.inf),
    ))
    known = {genome.genome_id for genome in population}
    regime_choices = [name for name in REGIME_FEATURES if name != base.regime_feature]
    rng.shuffle(regime_choices)
    variants: list[Genome] = []
    for index, (learner, memory_factor, bins) in enumerate((
        ("regime_decomposed_regressor", .35, 3),
        ("regime_regressor", .60, 2),
    )):
        payload = asdict(base)
        features = set(base.features)
        programs = [dict(program) for program in base.feature_programs]
        if len(programs) < 10:
            programs.append(random_program(rng))
        payload.update({
            "features": sorted(features),
            "feature_programs": programs,
            "recency_half_life_days": max(45.0, base.recency_half_life_days * memory_factor),
            "learner_kind": learner,
            "regime_feature": regime_choices[index % len(regime_choices)],
            "regime_bins": bins,
            "generation": generation,
            "parents": [base.genome_id],
            "fitness": None, "result": None, "genome_id": "",
        })
        variant = Genome(**payload).finalize()
        if variant.genome_id not in known:
            known.add(variant.genome_id)
            variants.append(variant)
    if variants:
        population[-len(variants):] = variants
    return population


def introduce_news_context_variant(
    population: list[Genome], evaluated: Sequence[Genome], generation: int,
    rng: random.Random,
) -> list[Genome]:
    """Reserve one fresh causal news/market interaction experiment.

    News had vanished from every live genome while the zero-offspring bug was
    active. This invariant guarantees representation, not favorable scoring.
    The candidate must still pass the same unseen-asset and economic gates.
    """
    if not population or not evaluated:
        return population
    news_lineages = [
        genome for genome in evaluated
        if set(genome.features) & set(NEWS_SPECIALIST_FEATURES)
    ]
    base = max(news_lineages or list(evaluated), key=lambda genome: float(
        genome.fitness if genome.fitness is not None else -math.inf
    ))
    payload = asdict(base)
    features = set(base.features) | set(NEWS_SPECIALIST_FEATURES)
    programs = [dict(program) for program in base.feature_programs]
    templates = (
        {"op": "tanh_mix", "left": "asset_news_sentiment_24h",
         "right": "market_breadth_r6"},
        {"op": "mul", "left": "news_negative_share_24h",
         "right": "volatility_ratio"},
        {"op": "signed_sqrt_product", "left": "news_liquidation_24h",
         "right": "funding_rate"},
        {"op": "regime_gate", "left": "asset_news_sentiment_acceleration",
         "right": "flow_divergence"},
    )
    template = dict(rng.choice(templates))
    template["scale"] = 10 ** rng.uniform(-.5, .5)
    programs.append(normalize_program(template))
    payload.update({
        "features": sorted(features), "feature_programs": programs[-10:],
        "recency_half_life_days": max(
            45.0, min(2200.0, base.recency_half_life_days * rng.choice((.7, 1.0, 1.3)))
        ),
        "generation": generation, "parents": [base.genome_id],
        "fitness": None, "result": None, "genome_id": "",
    })
    variant = Genome(**payload).finalize()
    known = {genome.genome_id for genome in population}
    if variant.genome_id in known:
        variant = mutate(variant, generation, rng)
        variant.features = sorted(set(variant.features) | set(NEWS_SPECIALIST_FEATURES))
        variant.fitness = None
        variant.result = None
        variant.finalize()
    if variant.genome_id in known:
        return population
    replacement = next(
        (index for index in range(len(population) - 1, 0, -1)
         if population[index].fitness is None),
        len(population) - 1,
    )
    population[replacement] = variant
    return population


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runtime/benchmarks/market-corpus-manifest.json")
    parser.add_argument("--supplemental-root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"))
    parser.add_argument("--news", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\news\historical_deduplicated.json"))
    parser.add_argument("--state-dir", type=Path, default=ROOT / "runtime/market-evolution")
    parser.add_argument("--dataset-cache", type=Path,
                        default=ROOT / "runtime/cache/market-evolution-dataset-v4.joblib")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-generations", type=int, default=0,
                        help="zero evolves perpetually")
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--seed", default="market-perpetual-v1")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=42)
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--final-days", type=int, default=21)
    parser.add_argument(
        "--cost-bps", type=float, default=OBJECTIVE_COST_BPS,
        help=(
            "round-trip execution cost in basis points. Defaults to the "
            "objective's true cost (%(default)s). The old 20.0 default was "
            "below GAS_ROUNDTRIP_FEE_RATIO (25 bps) alone, so every profit "
            "factor was measured against a cost we do not actually pay."
        ),
    )
    parser.add_argument("--brain-gate-every", type=int, default=5,
                        help="launch one isolated Wizard smoke gate this many generations")
    parser.add_argument(
        "--brain-gate-timeout-seconds", type=float, default=900.0,
        help="terminate this state directory's neural gate tree after this budget",
    )
    parser.add_argument("--min-free-memory-gb", type=float, default=8.0)
    parser.add_argument("--memory-poll-seconds", type=float, default=15.0)
    parser.add_argument("--memory-reclaim-after-polls", type=int, default=4)
    parser.add_argument("--memory-reclaim-process-name", default="w1z4rd_node.exe")
    parser.add_argument(
        "--memory-reclaim-command-fragment", default="api --addr 127.0.0.1:8090",
    )
    parser.add_argument(
        "--memory-reclaim-health-url", default="http://127.0.0.1:8090/health",
    )
    parser.add_argument("--memory-reclaim-cooldown-seconds", type=float, default=900.0)
    parser.add_argument(
        "--dataset-refresh-seconds", type=float, default=1800.0,
        help=("fingerprint OHLCV, derivatives, and news at generation boundaries; "
              "zero disables live corpus refresh"),
    )
    args = parser.parse_args()
    if args.population < 4:
        raise ValueError("population must be at least four")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.state_dir / "events.jsonl"
    # Materialize the dedicated stream immediately. It intentionally stays
    # empty until a strict, comparable accuracy increase occurs.
    (args.state_dir / "accuracy_improvements.jsonl").touch(exist_ok=True)
    (args.state_dir / "conditional_ghost_trades.jsonl").touch(exist_ok=True)
    state_path = args.state_dir / "state.json"
    stop_path = args.state_dir / "STOP"
    gate_process: subprocess.Popen | None = None
    gate_genome: str | None = None
    gate_out: Path | None = None
    gate_signature: str | None = None
    pending_gate_genome: str | None = None
    neural_scores: dict[str, float] = {}
    owner = claim_service(args.state_dir)
    rng = random.Random(args.seed)
    memory_reclaimer = VerifiedWorkingSetReclaimer(
        process_name=args.memory_reclaim_process_name,
        command_fragment=args.memory_reclaim_command_fragment,
        health_url=args.memory_reclaim_health_url,
        cooldown_seconds=max(0.0, args.memory_reclaim_cooldown_seconds),
    )
    if not wait_for_memory_floor(
        args.state_dir, stop_path, args.min_free_memory_gb,
        args.memory_poll_seconds, "memory_wait_before_dataset",
        reclaim_after_polls=args.memory_reclaim_after_polls,
        reclaimer=memory_reclaimer,
    ):
        try:
            if owner.read_text(encoding="ascii").strip() == str(os.getpid()):
                owner.unlink(missing_ok=True)
        except OSError:
            pass
        append_event(events_path, "stopped", generation=None, requested=True)
        return 0
    # Objective rule 2: profit must be measured at what trading actually
    # costs. Evaluating below OBJECTIVE_COST_BPS manufactures profit that
    # cannot exist live, so it is recorded loudly rather than passing
    # silently -- this is exactly how a champion came to look profitable
    # while every real round trip lost money to gas.
    if args.cost_bps < OBJECTIVE_COST_BPS:
        append_event(
            events_path, "objective_cost_understated",
            configured_cost_bps=args.cost_bps,
            objective_cost_bps=OBJECTIVE_COST_BPS,
            warning=("profit factors will be optimistic; results are not "
                     "comparable to the objective"),
        )
    data_signature = dataset_signature(args.manifest, args.supplemental_root, args.news)
    signature = evaluation_signature(
        data_signature, folds=args.folds, test_days=args.test_days,
        calibration_days=args.calibration_days, final_days=args.final_days,
        horizon=args.horizon, cost_bps=args.cost_bps,
    )
    dataset = load_dataset_cached(args.manifest, args.supplemental_root,
                                  args.horizon, args.stride, args.seed, args.news,
                                  args.dataset_cache)
    append_event(events_path, "dataset_loaded", rows=len(dataset["rows"]),
                 assets=dataset["assets"], supplemental_assets=sorted(dataset["supplemental_assets"]))
    coverage_frontier = load_coverage_frontier(args.state_dir)
    multiscale_frontier = load_multiscale_frontier(args.state_dir)
    multiscale_boundary_frontier = load_multiscale_boundary_frontier(args.state_dir)
    extra_trees_frontier = load_extra_trees_frontier(args.state_dir)
    regime_shift_frontier = load_regime_shift_frontier(args.state_dir)
    coverage_reversal_frontier = load_compatible_reversal_frontier(
        args.state_dir, coverage_frontier
    )
    multiscale_reversal_frontier = load_compatible_reversal_frontier(
        args.state_dir, multiscale_boundary_frontier or multiscale_frontier
    )
    extra_trees_reversal_frontier = load_compatible_reversal_frontier(
        args.state_dir, extra_trees_frontier
    )
    protected_frontier_ids = {
        genome.genome_id for genome in (
            coverage_frontier, multiscale_frontier, regime_shift_frontier,
            coverage_reversal_frontier, multiscale_reversal_frontier,
            multiscale_boundary_frontier, extra_trees_frontier,
            extra_trees_reversal_frontier,
        )
        if genome is not None
    }
    resumed_from_state = state_path.is_file()
    if resumed_from_state:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        generation = int(state["generation"])
        population = [genome_from_dict(row) for row in state["population"]]
        champion = genome_from_dict(state["champion"]) if state.get("champion") else None
        pending_gate_genome = state.get("pending_brain_gate")
        neural_scores = {str(key): float(value)
                         for key, value in state.get("neural_scores", {}).items()}
        # A champion whose fitness was computed under a previous objective
        # must not defend its title against freshly scored rivals. The
        # dataset check below only catches new DATA; bumping EVOLUTION_SCHEMA
        # (i.e. changing how fitness is computed) left the incumbent holding
        # a stale score. Observed 2026-08-17 after the profit-first rewrite:
        # the champion kept fitness 2384.7 for a PF 0.983 model that scores
        # ~1009 under the new function, so no honestly-measured challenger
        # could ever displace it.
        champion_signature = (champion.result or {}).get("evaluation_signature") \
            if champion is not None else None
        if champion is not None and champion_signature != signature:
            append_event(
                events_path, "champion_retired_stale_objective",
                generation=generation, genome=champion.genome_id,
                stale_fitness=champion.fitness,
                previous_signature=champion_signature,
                current_signature=signature,
            )
            champion = None
        if state.get("dataset_signature") != signature:
            population = invalidate_population_for_new_evidence(
                population, generation, rng
            )
            champion = None
            neural_scores = {}
            pending_gate_genome = None
            append_event(events_path, "dataset_changed", previous=state.get("dataset_signature"),
                         current=signature, action="population_revalidation")
        else:
            population, restored_candidates = restore_completed_candidates(
                population, args.state_dir, signature,
                legacy_after=state_path.stat().st_mtime,
            )
            if restored_candidates:
                append_event(
                    events_path, "candidate_evidence_restored",
                    generation=generation, genomes=restored_candidates,
                    evaluation_signature=signature,
                )
            safe_champion, rolled_back = rollback_unsafe_champion(
                args.state_dir, champion, signature
            )
            if safe_champion is not None and rolled_back:
                champion = safe_champion
                atomic_json(args.state_dir / "champion.json", asdict(champion))
                append_event(
                    events_path, "unsafe_champion_rolled_back",
                    rejected=rolled_back, restored=champion.genome_id,
                    reason="bounded_pareto_contract",
                )
            champion, gate_rejected, failed_gate_ids = reconcile_brain_gate_champion(
                args.state_dir, champion, signature, neural_scores
            )
            if gate_rejected is not None:
                if champion is not None:
                    atomic_json(
                        args.state_dir / "champion.json", asdict(champion)
                    )
                pending_gate_genome = recover_pending_gate(
                    args.state_dir, champion, pending_gate_genome,
                    prefer_champion=True,
                )
                append_event(
                    events_path, "brain_gate_champion_rolled_back",
                    rejected=gate_rejected,
                    restored=(champion.genome_id if champion else None),
                    reason="completed_isolated_gate_failed",
                    live_admission_effect="none",
                )
        population, resumed_regime_conversions = cap_expensive_regime_candidates(
            population, generation, regime_shift_frontier
        )
        if resumed_regime_conversions:
            append_event(
                events_path, "resume_regime_candidates_converted",
                generation=generation, count=resumed_regime_conversions,
                reason="one-expensive-regime-candidate-budget",
            )
        population, resumed_multiscale_conversions = cap_expensive_multiscale_candidates(
            population, generation, {
                genome.genome_id for genome in (
                    multiscale_frontier, multiscale_boundary_frontier,
                ) if genome is not None
            },
            set(filter(None, {
                learner_ablation_evaluation_key(
                    coverage_frontier, "multiscale_regressor"
                ),
            })),
        )
        if resumed_multiscale_conversions:
            append_event(
                events_path, "resume_multiscale_candidates_converted",
                generation=generation, count=resumed_multiscale_conversions,
                reason="active-multiscale-frontier-only-budget",
            )
        # A restart is not a breeding boundary. Apart from the explicit
        # one-expensive-regime resource migration above, diversity invariants
        # are applied only after evaluation at the next real breeding boundary.
        pending_gate_genome = recover_pending_gate(
            args.state_dir, champion, pending_gate_genome,
            prefer_champion=True,
        )
        append_event(events_path, "resumed", generation=generation, population=len(population))
    else:
        generation = 0
        population = seed_genomes(args.population, rng)
        champion = None
        append_event(events_path, "started", population=args.population)
    if not resumed_from_state:
        population = introduce_reflexivity_variant(
            population, generation, protected_frontier_ids
        )
    if coverage_frontier is not None:
        append_event(
            events_path, "coverage_frontier_loaded",
            genome_id=coverage_frontier.genome_id,
            summary=(coverage_frontier.result or {}).get("summary", {}),
        )
    if multiscale_frontier is not None:
        append_event(
            events_path, "multiscale_frontier_loaded",
            genome_id=multiscale_frontier.genome_id,
            rank=multiscale_frontier_rank(multiscale_frontier),
            summary=(multiscale_frontier.result or {}).get("summary", {}),
        )
    if multiscale_boundary_frontier is not None:
        append_event(
            events_path, "multiscale_boundary_frontier_loaded",
            genome_id=multiscale_boundary_frontier.genome_id,
            rank=multiscale_boundary_rank(multiscale_boundary_frontier),
            summary=(multiscale_boundary_frontier.result or {}).get("summary", {}),
        )
    if extra_trees_frontier is not None:
        append_event(
            events_path, "extra_trees_frontier_loaded",
            genome_id=extra_trees_frontier.genome_id,
            rank=extra_trees_frontier_rank(extra_trees_frontier),
            summary=(extra_trees_frontier.result or {}).get("summary", {}),
        )
    if regime_shift_frontier is not None:
        append_event(
            events_path, "regime_shift_frontier_loaded",
            genome_id=regime_shift_frontier.genome_id,
            rank=regime_shift_rank(regime_shift_frontier),
        )
    if coverage_reversal_frontier is not None:
        append_event(
            events_path, "coverage_reversal_frontier_loaded",
            genome_id=coverage_reversal_frontier.genome_id,
            confidence_quantile=coverage_reversal_frontier.confidence_quantile,
            rank=compatible_reversal_rank(
                coverage_reversal_frontier, coverage_frontier
            ),
        )
    if multiscale_reversal_frontier is not None:
        append_event(
            events_path, "multiscale_reversal_frontier_loaded",
            genome_id=multiscale_reversal_frontier.genome_id,
            confidence_quantile=multiscale_reversal_frontier.confidence_quantile,
            rank=compatible_reversal_rank(
                multiscale_reversal_frontier,
                multiscale_boundary_frontier or multiscale_frontier,
            ),
        )
    if extra_trees_reversal_frontier is not None:
        append_event(
            events_path, "extra_trees_reversal_frontier_loaded",
            genome_id=extra_trees_reversal_frontier.genome_id,
            confidence_quantile=extra_trees_reversal_frontier.confidence_quantile,
            rank=compatible_reversal_rank(
                extra_trees_reversal_frontier, extra_trees_frontier
            ),
        )
    had_emergent_pool = any(genome.emergent_pools for genome in population)
    if not resumed_from_state:
        population = introduce_emergent_pool_variant(
            population, generation, rng, protected_frontier_ids
        )
    if not had_emergent_pool and any(genome.emergent_pools for genome in population):
        seeded = next(genome for genome in population if genome.emergent_pools)
        append_event(
            events_path, "emergent_pool_variant_seeded", generation=generation,
            genome_id=seeded.genome_id, pools=seeded.emergent_pools,
        )
    next_dataset_check = time.monotonic() + max(0.0, args.dataset_refresh_seconds)
    try:
        while not stop_path.exists() and (args.max_generations == 0 or generation < args.max_generations):
            recovered_gate_pids = recover_stale_external_brain_gates(
                args.state_dir, args.brain_gate_timeout_seconds
            )
            for stale_pid in recovered_gate_pids:
                append_event(
                    events_path, "stale_brain_gate_recovered", pid=stale_pid,
                    timeout_seconds=args.brain_gate_timeout_seconds,
                    live_admission_effect="none",
                )
            champion, gate_rejected, failed_gate_ids = reconcile_brain_gate_champion(
                args.state_dir, champion, signature, neural_scores
            )
            if gate_rejected is not None:
                if champion is not None:
                    atomic_json(
                        args.state_dir / "champion.json", asdict(champion)
                    )
                pending_gate_genome = recover_pending_gate(
                    args.state_dir, champion, None
                )
                append_event(
                    events_path, "brain_gate_champion_rolled_back",
                    rejected=gate_rejected,
                    restored=(champion.genome_id if champion else None),
                    reason="completed_isolated_gate_failed",
                    live_admission_effect="none",
                )
            if gate_process is not None and gate_process.poll() is not None:
                gate_report = (json.loads(gate_out.read_text(encoding="utf-8"))
                               if gate_out is not None and gate_out.is_file() else None)
                gate_is_current = gate_signature == signature
                append_event(events_path, "brain_gate_finished", genome_id=gate_genome,
                             returncode=gate_process.returncode,
                             passed=((gate_report or {}).get("all_brain_floor_gates")
                                     if gate_is_current else False),
                             feedback_score=(brain_feedback_score(gate_report)
                                             if gate_is_current else 0.0),
                             evidence_current=gate_is_current,
                             gate_signature=gate_signature,
                             current_signature=signature)
                if gate_genome is not None and gate_is_current:
                    neural_scores[gate_genome] = brain_feedback_score(gate_report)
                    neural_summary = brain_accuracy_summary(gate_report)
                    if neural_summary is not None:
                        neural_source = (
                            f"isolated_wizard_brain_{len((gate_report or {}).get('folds', []))}fold"
                        )
                        record_accuracy_improvement(
                            args.state_dir, signature, neural_source,
                            neural_summary["min_accuracy"], generation=generation,
                            genome_id=gate_genome, metrics=neural_summary,
                        )
                if (gate_is_current and gate_report
                        and gate_report.get("all_brain_floor_gates")):
                    atomic_json(args.state_dir / "untouched-final-queue.json", {
                        "queued_at": utc_now(), "genome_id": gate_genome,
                        "brain_gate_report": str(gate_out),
                        "required_next_gate": "one-time untouched final period; no further selection on it",
                    })
                gate_process = None
                gate_genome = None
                gate_out = None
                gate_signature = None
            if (args.dataset_refresh_seconds > 0
                    and time.monotonic() >= next_dataset_check):
                next_dataset_check = time.monotonic() + args.dataset_refresh_seconds
                refreshed_data_signature = dataset_signature(
                    args.manifest, args.supplemental_root, args.news
                )
                refreshed_signature = evaluation_signature(
                    refreshed_data_signature, folds=args.folds,
                    test_days=args.test_days, calibration_days=args.calibration_days,
                    final_days=args.final_days, horizon=args.horizon,
                    cost_bps=args.cost_bps,
                )
                if refreshed_signature != signature:
                    previous_signature = signature
                    write_live_status(
                        args.state_dir, "dataset_refresh", generation,
                        previous_signature=previous_signature,
                        current_signature=refreshed_signature,
                    )
                    dataset = load_dataset_cached(
                        args.manifest, args.supplemental_root, args.horizon,
                        args.stride, args.seed, args.news, args.dataset_cache,
                    )
                    population = invalidate_population_for_new_evidence(
                        population, generation, rng
                    )
                    champion = None
                    pending_gate_genome = None
                    neural_scores = {}
                    data_signature = refreshed_data_signature
                    signature = refreshed_signature
                    append_event(
                        events_path, "dataset_changed",
                        previous=previous_signature, current=signature,
                        action="live_population_revalidation",
                        rows=len(dataset["rows"]), assets=dataset["assets"],
                    )
            if not wait_for_memory_floor(
                args.state_dir, stop_path, args.min_free_memory_gb,
                args.memory_poll_seconds, "memory_wait", generation=generation,
                reclaim_after_polls=args.memory_reclaim_after_polls,
                reclaimer=memory_reclaimer,
            ):
                break
            population, reused_evidence = restore_completed_candidates(
                population, args.state_dir, signature
            )
            if reused_evidence:
                append_event(
                    events_path, "phenotype_evidence_reused",
                    generation=generation, genomes=reused_evidence,
                    evaluation_signature=signature,
                )
            append_event(events_path, "generation_started", generation=generation,
                         genomes=[genome.genome_id for genome in population])
            pending = prioritize_pending_genomes(
                [genome for genome in population if genome.fitness is None],
                coverage_frontier, regime_shift_frontier,
                multiscale_boundary_frontier or multiscale_frontier,
                extra_trees_frontier,
                champion,
            )
            write_live_status(
                args.state_dir, "evaluating", generation,
                completed=0, pending=len(pending), population=len(population),
                champion=champion.genome_id if champion else None,
            )
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        evaluate_genome_after_memory_floor, genome, dataset,
                        state_dir=args.state_dir, stop_path=stop_path,
                        required_memory_gb=args.min_free_memory_gb,
                        poll_seconds=args.memory_poll_seconds,
                        generation=generation,
                        reclaim_after_polls=args.memory_reclaim_after_polls,
                        reclaimer=memory_reclaimer,
                        guard_each_candidate=args.workers == 1,
                        folds=args.folds,
                        test_days=args.test_days, calibration_days=args.calibration_days,
                        final_days=args.final_days, horizon=args.horizon, cost_bps=args.cost_bps,
                    ): genome for genome in pending
                }
                completed = 0
                for future in as_completed(futures):
                    genome = future.result()
                    if genome is None:
                        continue
                    completed += 1
                    if genome.result is not None:
                        genome.result["evaluation_signature"] = signature
                    atomic_json(args.state_dir / "candidates" / f"{genome.genome_id}.json",
                                asdict(genome))
                    append_event(events_path, "candidate_scored", generation=generation,
                                 genome_id=genome.genome_id, fitness=genome.fitness,
                                 status=(genome.result or {}).get("status"),
                                 summary=(genome.result or {}).get("summary"))
                    result = genome.result or {}
                    summary = result.get("summary") or {}
                    previous_structure = (
                        genome_structure_key(coverage_frontier)
                        if coverage_frontier is not None else None
                    )
                    previous_multiscale_structure = (
                        genome_structure_key(
                            multiscale_boundary_frontier or multiscale_frontier
                        )
                        if (multiscale_boundary_frontier is not None
                            or multiscale_frontier is not None) else None
                    )
                    previous_tree_structure = (
                        genome_structure_key(extra_trees_frontier)
                        if extra_trees_frontier is not None else None
                    )
                    coverage_frontier = update_coverage_frontier(
                        args.state_dir, coverage_frontier, genome
                    )
                    multiscale_frontier = update_multiscale_frontier(
                        args.state_dir, multiscale_frontier, genome
                    )
                    multiscale_boundary_frontier = update_multiscale_boundary_frontier(
                        args.state_dir, multiscale_boundary_frontier, genome
                    )
                    extra_trees_frontier = update_extra_trees_frontier(
                        args.state_dir, extra_trees_frontier, genome
                    )
                    current_multiscale_structure = (
                        genome_structure_key(
                            multiscale_boundary_frontier or multiscale_frontier
                        )
                        if (multiscale_boundary_frontier is not None
                            or multiscale_frontier is not None) else None
                    )
                    if current_multiscale_structure != previous_multiscale_structure:
                        multiscale_reversal_frontier = load_compatible_reversal_frontier(
                            args.state_dir,
                            multiscale_boundary_frontier or multiscale_frontier,
                        )
                    else:
                        multiscale_reversal_frontier = update_compatible_reversal_frontier(
                            multiscale_reversal_frontier, genome,
                            multiscale_boundary_frontier or multiscale_frontier,
                        )
                    current_tree_structure = (
                        genome_structure_key(extra_trees_frontier)
                        if extra_trees_frontier is not None else None
                    )
                    if current_tree_structure != previous_tree_structure:
                        extra_trees_reversal_frontier = load_compatible_reversal_frontier(
                            args.state_dir, extra_trees_frontier
                        )
                    else:
                        extra_trees_reversal_frontier = update_compatible_reversal_frontier(
                            extra_trees_reversal_frontier, genome, extra_trees_frontier
                        )
                    current_structure = (
                        genome_structure_key(coverage_frontier)
                        if coverage_frontier is not None else None
                    )
                    if current_structure != previous_structure:
                        coverage_reversal_frontier = load_compatible_reversal_frontier(
                            args.state_dir, coverage_frontier
                        )
                    else:
                        coverage_reversal_frontier = update_compatible_reversal_frontier(
                            coverage_reversal_frontier, genome, coverage_frontier
                        )
                    regime_shift_frontier = update_regime_shift_frontier(
                        args.state_dir, regime_shift_frontier, genome
                    )
                    if summary.get("conditional_ghost_sections", 0):
                        append_event(
                            args.state_dir / "conditional_ghost_trades.jsonl",
                            "historical_conditional_ghost_evaluation",
                            generation=generation, genome_id=genome.genome_id,
                            evaluation_signature=signature,
                            passed=bool(summary.get("conditional_ghost_pass")),
                            min_accuracy=summary.get("conditional_ghost_min_accuracy"),
                            sections=summary.get("conditional_ghost_sections"),
                            folds=[{
                                "fold": fold.get("fold"),
                                "conditions": (fold.get("competence_envelope") or {}).get(
                                    "conditions", []),
                                "known": fold.get("conditional_ghost_known", {}),
                                "unseen": fold.get("conditional_ghost_unseen", {}),
                            } for fold in result.get("folds", [])],
                            scope="historical_only",
                            live_admission_effect="none",
                        )
                    if (result.get("evaluated_folds") == result.get("requested_folds")
                            and summary.get("min_accuracy") is not None):
                        record_accuracy_improvement(
                            args.state_dir, signature, "protected_surrogate",
                            summary["min_accuracy"], generation=generation,
                            genome_id=genome.genome_id, metrics=summary,
                        )
                    write_live_status(
                        args.state_dir, "evaluating", generation,
                        completed=completed, pending=len(pending), population=len(population),
                        latest_genome=genome.genome_id, latest_fitness=genome.fitness,
                        latest_status=(genome.result or {}).get("status"),
                        latest_summary=(genome.result or {}).get("summary"),
                        champion=champion.genome_id if champion else None,
                    )
                    print(f"generation {generation} {genome.genome_id} fitness={genome.fitness:.4f} "
                          f"{(genome.result or {}).get('status')}", flush=True)
            population.sort(key=lambda genome: selection_fitness(genome, neural_scores),
                            reverse=True)
            champion_eligible_population = [
                genome for genome in population
                if genome.genome_id not in failed_gate_ids
            ]
            challenger = (
                champion_eligible_population[0]
                if champion is None and champion_eligible_population else
                next((
                    genome for genome in champion_eligible_population
                    if champion_replacement_allowed(genome, champion)
                ), None) if champion is not None else None
            )
            # HARDENING: never let a better earner be suppressed silently.
            #
            # A strict Pareto rule once refused the championship to a genome
            # that beat the incumbent on every economic metric, and nothing in
            # the logs said so -- the only symptom was a champion that quietly
            # stopped improving. Any fully-measured candidate with a higher
            # profit factor than the incumbent that is NOT promoted now emits
            # an explicit event naming the exact blocking metrics.
            if champion is not None:
                champion_pf = float(((champion.result or {}).get("summary") or {})
                                    .get("min_profit_factor") or 0.0)
                for genome in champion_eligible_population:
                    if challenger is not None and genome.genome_id == challenger.genome_id:
                        continue
                    result = genome.result or {}
                    summary = result.get("summary") or {}
                    candidate_pf = float(summary.get("min_profit_factor") or 0.0)
                    if candidate_pf <= champion_pf:
                        continue
                    if int(result.get("evaluated_folds", 0)) < int(
                            result.get("requested_folds", 3)):
                        continue
                    append_event(
                        events_path, "higher_profit_candidate_suppressed",
                        generation=generation, genome_id=genome.genome_id,
                        candidate_profit_factor=candidate_pf,
                        champion_profit_factor=champion_pf,
                        candidate_fitness=genome.fitness,
                        champion_fitness=champion.fitness,
                        blocking_metrics=champion_replacement_blockers(
                            genome, champion
                        ),
                        summary=summary,
                    )
            champion_changed = challenger is not None
            if challenger is not None:
                champion = genome_from_dict(asdict(challenger))
                atomic_json(args.state_dir / "champion.json", asdict(champion))
                append_event(events_path, "champion_updated", generation=generation,
                             genome_id=champion.genome_id, fitness=champion.fitness,
                             summary=(champion.result or {}).get("summary"),
                             brain_gate="pending_isolated_wizard_validation")
                if (generation == 0
                        or generation % max(1, args.brain_gate_every) == 0
                        or (champion.result or {}).get("status") in {
                            "surrogate_floor_pass", "surrogate_working_target_pass"}):
                    # A gate can be much slower than statistical screening.
                    # Retain the newest obligation while an older gate runs.
                    pending_gate_genome = champion.genome_id
            if (pending_gate_genome is None
                    and generation % max(1, args.brain_gate_every) == 0):
                # Surrogate scoring is intentionally blind to neural topology.
                # Give the strongest emergent-pool hypothesis an isolated brain
                # gate so new pool layouts receive causal neural evidence.
                ungated_pool_candidates = [
                    genome for genome in population if genome.emergent_pools
                    and not (args.state_dir / "brain-gate-reports"
                             / f"{genome.genome_id}.smoke.json").exists()
                ]
                newly_exhausted = next((
                    genome for genome in ungated_pool_candidates
                    if genome.genome_id not in neural_scores
                    and brain_gate_retry_exhausted(
                        args.state_dir, events_path, genome.genome_id
                    )
                ), None)
                if newly_exhausted is not None:
                    neural_scores[newly_exhausted.genome_id] = brain_feedback_score(None)
                    append_event(
                        events_path, "brain_gate_retry_exhausted",
                        genome_id=newly_exhausted.genome_id,
                        attempts=brain_gate_attempt_count(
                            events_path, newly_exhausted.genome_id
                        ),
                        feedback_score=brain_feedback_score(None),
                        live_admission_effect="none",
                    )
                pool_candidate = next((
                    genome for genome in ungated_pool_candidates
                    if not brain_gate_retry_exhausted(
                        args.state_dir, events_path, genome.genome_id
                    )
                ), None)
                if pool_candidate is not None:
                    pending_gate_genome = pool_candidate.genome_id
                    append_event(
                        events_path, "emergent_pool_gate_queued",
                        generation=generation, genome_id=pool_candidate.genome_id,
                        pools=pool_candidate.emergent_pools,
                    )
            if (pending_gate_genome is not None
                    and brain_gate_retry_exhausted(
                        args.state_dir, events_path, pending_gate_genome
                    )):
                attempts = brain_gate_attempt_count(
                    events_path, pending_gate_genome
                )
                neural_scores[pending_gate_genome] = brain_feedback_score(None)
                append_event(
                    events_path, "brain_gate_retry_exhausted",
                    genome_id=pending_gate_genome, attempts=attempts,
                    feedback_score=brain_feedback_score(None),
                    live_admission_effect="none",
                )
                pending_gate_genome = None
            if (pending_gate_genome is not None
                    and not brain_gate_obligation_viable(
                        args.state_dir, pending_gate_genome
                    )):
                append_event(
                    events_path, "brain_gate_skipped_conclusive_anti_signal",
                    genome_id=pending_gate_genome,
                    live_admission_effect="none",
                )
                pending_gate_genome = None
            if (gate_process is None and pending_gate_genome is not None
                    and not external_brain_gate_pids()):
                candidate_path = (args.state_dir / "candidates"
                                  / f"{pending_gate_genome}.json")
                gate_out = (args.state_dir / "brain-gate-reports"
                            / f"{pending_gate_genome}.smoke.json")
                if candidate_path.is_file() and not gate_out.exists():
                    gate_candidate = genome_from_dict(
                        json.loads(candidate_path.read_text(encoding="utf-8"))
                    )
                    gate_out.parent.mkdir(parents=True, exist_ok=True)
                    gate_stdout = (args.state_dir / "brain-gate-reports"
                                   / f"{pending_gate_genome}.stdout.log").open("wb")
                    gate_stderr = (args.state_dir / "brain-gate-reports"
                                   / f"{pending_gate_genome}.stderr.log").open("wb")
                    full_gate = ((gate_candidate.result or {}).get("status")
                                 == "surrogate_working_target_pass")
                    command = [
                        sys.executable, str(ROOT / "scripts/market_evolution_brain_gate.py"),
                        "--candidate", str(candidate_path), "--out", str(gate_out),
                        "--folds", "3" if full_gate else "1",
                        "--train-per-asset", "16" if full_gate else "8",
                        "--calibration-per-asset", "8" if full_gate else "4",
                        "--test-per-asset", "40" if full_gate else "20",
                    ]
                    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    gate_process = subprocess.Popen(command, cwd=ROOT, stdout=gate_stdout,
                                                    stderr=gate_stderr,
                                                    creationflags=creationflags)
                    gate_stdout.close()
                    gate_stderr.close()
                    gate_genome = pending_gate_genome
                    gate_signature = signature
                    pending_gate_genome = None
                    append_event(events_path, "brain_gate_started", genome_id=gate_genome,
                                 pid=gate_process.pid,
                                 stage="full" if full_gate else "smoke", command=command)
                elif gate_out.exists():
                    append_event(events_path, "brain_gate_already_recorded",
                                 genome_id=pending_gate_genome, report=str(gate_out))
                    pending_gate_genome = None
            generation += 1
            evaluated_population = population
            next_population, exploration = breed_population(
                evaluated_population, generation, rng, neural_scores
            )
            population = introduce_directional_frontier_variants(
                next_population, evaluated_population, generation
            )
            population = introduce_regime_repair_variants(
                population, evaluated_population, generation, rng
            )
            population = introduce_reflexivity_variant(population, generation)
            population = introduce_news_context_variant(
                population, evaluated_population, generation, rng
            )
            population = preserve_emergent_pool_elite(
                population, evaluated_population
            )
            population = introduce_emergent_pool_variant(population, generation, rng)
            # Apply coverage repair last. Its slot chooser protects emergent
            # topology, while news-bearing repair parents retain news context.
            historical_repair_evidence = load_direct_descendant_evidence(
                args.state_dir,
                {
                    genome.genome_id for genome in (
                        coverage_frontier, multiscale_frontier,
                        multiscale_boundary_frontier,
                    ) if genome is not None
                },
                signature,
            )
            primary_coverage_evidence = (
                load_structure_evidence(
                    args.state_dir, coverage_frontier, signature
                ) if coverage_frontier is not None else []
            )
            population = introduce_coverage_repair_variants(
                population,
                [*([coverage_frontier] if coverage_frontier is not None else []),
                 *historical_repair_evidence,
                 *(genome for genome in evaluated_population
                   if genome.genome_id not in {
                       candidate.genome_id
                       for candidate in historical_repair_evidence
                   }
                   and (coverage_frontier is None
                        or genome.genome_id != coverage_frontier.genome_id))],
                generation,
                coverage_reversal_frontier,
                multiscale_frontier,
                multiscale_reversal_frontier,
                multiscale_boundary_frontier,
            )
            population = introduce_primary_coverage_variant(
                population, coverage_frontier,
                [*primary_coverage_evidence, *historical_repair_evidence],
                generation,
                {
                    genome.genome_id for genome in (
                        multiscale_frontier, multiscale_boundary_frontier,
                    ) if genome is not None
                },
            )
            population = introduce_extra_trees_coverage_variant(
                population, extra_trees_frontier, generation,
                extra_trees_reversal_frontier,
                {
                    genome.genome_id for genome in (
                        coverage_frontier, multiscale_frontier,
                        multiscale_boundary_frontier,
                    ) if genome is not None
                },
                evidence=load_direct_descendant_evidence(
                    args.state_dir,
                    ({extra_trees_frontier.genome_id}
                     if extra_trees_frontier is not None else set()),
                    signature,
                ),
                evaluation_id=signature,
            )
            population = introduce_regime_shift_variants(
                population, regime_shift_frontier, generation,
                ({
                    genome.genome_id for genome in (
                        coverage_frontier, multiscale_frontier,
                        multiscale_boundary_frontier, extra_trees_frontier,
                    ) if genome is not None
                }),
            )
            champion_coordinate_evidence = load_direct_descendant_evidence(
                args.state_dir,
                ({champion.genome_id} if champion is not None else set()),
                signature,
            )
            champion_profit_evidence = (
                load_nearby_program_transfer_evidence(
                    args.state_dir, champion, signature
                ) if champion is not None else []
            )
            champion_return_tree_evidence = (
                load_nearby_return_tree_evidence(
                    args.state_dir, champion, signature
                ) if champion is not None else []
            )
            population = introduce_champion_coordinate_variant(
                population, champion, champion_coordinate_evidence,
                generation,
                {
                    genome.genome_id for genome in (
                        coverage_frontier, multiscale_frontier,
                        multiscale_boundary_frontier, extra_trees_frontier,
                        regime_shift_frontier,
                    ) if genome is not None
                },
            )
            protected_specialist_parents = {
                genome.genome_id for genome in (
                    coverage_frontier, multiscale_frontier,
                    multiscale_boundary_frontier, extra_trees_frontier,
                    regime_shift_frontier,
                ) if genome is not None
            }
            before_specialist_search = [
                genome.genome_id for genome in population
            ]
            return_tree_priority = any(
                profitable_return_tree_coverage_frontier(genome)
                for genome in champion_return_tree_evidence
            )
            if return_tree_priority:
                population = introduce_champion_return_tree_variant(
                    population, champion, champion_return_tree_evidence,
                    generation, protected_specialist_parents,
                )
            if [genome.genome_id for genome in population] == before_specialist_search:
                population = introduce_champion_profit_program_from_frontiers(
                    population, champion,
                    (extra_trees_frontier, coverage_frontier, multiscale_frontier),
                    champion_profit_evidence, generation,
                    protected_specialist_parents,
                )
            if (not return_tree_priority
                    and [genome.genome_id for genome in population]
                    == before_specialist_search):
                population = introduce_champion_return_tree_variant(
                    population, champion, champion_return_tree_evidence,
                    generation, protected_specialist_parents,
                )
            population, regime_candidates_converted = cap_expensive_regime_candidates(
                population, generation, regime_shift_frontier
            )
            population, multiscale_candidates_converted = (
                cap_expensive_multiscale_candidates(
                    population, generation, {
                        genome.genome_id for genome in (
                            multiscale_frontier, multiscale_boundary_frontier,
                        ) if genome is not None
                    },
                    set(filter(None, {
                        learner_ablation_evaluation_key(
                            coverage_frontier, "multiscale_regressor"
                        ),
                    })),
                )
            )
            outcome_evidence: list[Genome] = []
            protected_search_parents = {
                genome.genome_id for genome in (
                    champion, coverage_frontier, multiscale_frontier,
                    multiscale_boundary_frontier, extra_trees_frontier,
                    regime_shift_frontier,
                ) if genome is not None
            }
            try:
                outcome_evidence = outcome_pool_evidence(
                    args.state_dir, signature
                )
                outcome_pool = train_genome_outcome_pool(outcome_evidence)
                population, outcome_pool_report = introduce_outcome_pool_variant(
                    population, outcome_evidence, outcome_pool, generation, rng,
                    protected_search_parents,
                    plateau_generations=(
                        generation - champion.generation if champion else 0
                    ),
                )
            except Exception as exc:
                outcome_pool_report = {
                    "active": False, "proposed": False,
                    "error": repr(exc),
                }
            try:
                population, leaf_refinement_report = (
                    introduce_tree_leaf_refinement_variant(
                        population, outcome_evidence, generation, signature,
                        protected_search_parents,
                    )
                )
            except Exception as exc:
                leaf_refinement_report = {
                    "active": False, "proposed": False, "error": repr(exc),
                }
            atomic_json(args.state_dir / "genome_outcome_pool.json", {
                "at": utc_now(), "generation": generation,
                **outcome_pool_report,
                "scope": "reproduction_advisory_only",
                "live_admission_effect": "none",
            })
            atomic_json(args.state_dir / "tree_leaf_refinement.json", {
                "at": utc_now(), "generation": generation,
                **leaf_refinement_report,
                "scope": "reproduction_advisory_only",
                "live_admission_effect": "none",
            })
            population, novelty_injections = ensure_novelty(
                population, generation, rng
            )
            exploration.update({
                "generation": generation,
                "novel_candidates": sum(
                    genome.fitness is None for genome in population
                ),
                "minimum_novel_candidates": minimum_novel_candidates(len(population)),
                "novelty_injections": novelty_injections,
                "unique_genomes": len({genome.genome_id for genome in population}),
                "news_candidates": sum(
                    bool(set(genome.features) & set(NEWS_SPECIALIST_FEATURES))
                    for genome in population
                ),
                "champion_age_generations": (
                    generation - champion.generation if champion else 0
                ),
                "low_threshold_candidates": sum(
                    genome.fitness is None and genome.confidence_quantile <= .15
                    for genome in population
                ),
                "regime_shift_candidates": sum(
                    regime_shift_frontier is not None
                    and regime_shift_frontier.genome_id in genome.parents
                    for genome in population
                ),
                "primary_coverage_candidates": sum(
                    coverage_frontier is not None
                    and coverage_frontier.genome_id in genome.parents
                    and genome.fitness is None
                    for genome in population
                ),
                "champion_coordinate_candidates": sum(
                    champion is not None
                    and champion.genome_id in genome.parents
                    and genome.fitness is None
                    and genome.feature_programs == champion.feature_programs
                    and genome.learner_kind == champion.learner_kind
                    for genome in population
                ),
                "champion_profit_program_candidates": sum(
                    champion is not None
                    and champion.genome_id in genome.parents
                    and genome.fitness is None
                    and len(genome.feature_programs) > len(champion.feature_programs)
                    for genome in population
                ),
                "champion_return_tree_candidates": sum(
                    bool(set(genome.parents) & {
                        evidence.genome_id
                        for evidence in champion_return_tree_evidence
                    })
                    and genome.fitness is None
                    and genome.learner_kind in {
                        "extra_trees_regressor", "extra_trees_hybrid",
                    }
                    for genome in population
                ),
                "surplus_regime_candidates_converted": regime_candidates_converted,
                "untargeted_multiscale_candidates_converted": (
                    multiscale_candidates_converted
                ),
                "outcome_pool_active": bool(outcome_pool_report.get("active")),
                "outcome_pool_examples": int(outcome_pool_report.get("examples", 0)),
                "outcome_pool_candidates": int(bool(
                    outcome_pool_report.get("proposed")
                )),
                "tree_leaf_refinement_candidates": int(bool(
                    leaf_refinement_report.get("proposed")
                )),
            })
            atomic_json(args.state_dir / "evolution_health.json", {
                "at": utc_now(), **exploration,
            })
            append_event(events_path, "exploration_health", **exploration)
            state = {
                "schema": EVOLUTION_SCHEMA, "updated_at": utc_now(), "generation": generation,
                "configuration": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                "contract": str(ROOT / "docs/MARKET_BRAIN_GHOST_TRADING_ADMISSION.md"),
                "working_target": WORKING_TARGET,
                "dataset_signature": signature,
                "pending_brain_gate": pending_gate_genome,
                "neural_scores": neural_scores,
                "population": [asdict(genome) for genome in population],
                "champion": asdict(champion) if champion else None,
            }
            atomic_json(state_path, state)
            append_event(events_path, "generation_completed", generation=generation,
                         champion=champion.genome_id if champion else None,
                         champion_fitness=champion.fitness if champion else None)
            refresh_genome_audit(args.state_dir, generation)
            write_live_status(
                args.state_dir, "generation_complete", generation,
                population=len(population), champion=champion.genome_id if champion else None,
                champion_fitness=champion.fitness if champion else None,
                champion_summary=(champion.result or {}).get("summary") if champion else None,
            )
            time.sleep(max(0, args.sleep_seconds))
    finally:
        try:
            if owner.read_text(encoding="ascii").strip() == str(os.getpid()):
                owner.unlink(missing_ok=True)
        except OSError:
            pass
        append_event(events_path, "stopped", generation=generation,
                     requested=stop_path.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
