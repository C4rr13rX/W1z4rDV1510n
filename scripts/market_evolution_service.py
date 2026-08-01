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
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_brain_experiment import load_bars  # noqa: E402
from scripts.market_signal_audit import (  # noqa: E402
    FEATURE_GROUPS, attach_market_breadth, build_rows, load_supplemental,
    metrics, select_primary_assets, stable_order,
)

FLOOR = {
    "observations": 200, "coverage": 0.70, "accuracy": 0.58,
    "balanced_accuracy": 0.55, "baseline_margin": 0.05, "mcc": 0.15,
    "ece": 0.10, "profit_factor": 1.20, "max_drawdown": 0.15,
}
WORKING_TARGET = {"accuracy": 0.62, "balanced_accuracy": 0.58,
                  "mcc": 0.25, "profit_factor": 1.35}
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
}
NEWS_FEATURES = (
    "news_count_6h", "news_count_24h", "news_count_72h",
    "news_sentiment_6h", "news_sentiment_24h", "news_sentiment_72h",
    "news_polarity_24h", "asset_news_count_24h", "asset_news_count_72h",
    "asset_news_sentiment_24h",
)
BASE_FEATURES = tuple(dict.fromkeys(
    FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"]
    + FEATURE_GROUPS["derivatives"] + FEATURE_GROUPS["breadth"] + NEWS_FEATURES
))
DERIVATIVE_FEATURES = set(FEATURE_GROUPS["derivatives"]) | {
    "flow_consensus", "flow_basis_pressure", "funding_basis_pressure",
    "liquidity_acceleration",
}


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
        payload["features"] = self.features
        self.genome_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return self


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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


def dataset_signature(manifest: Path, supplemental_root: Path,
                      news_path: Path | None = None) -> str:
    digest = hashlib.sha256()
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


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), "event": event, **payload},
                                separators=(",", ":"), allow_nan=False) + "\n")


def recover_pending_gate(state_dir: Path, champion: Genome | None,
                         pending: str | None) -> str | None:
    """Restore a neural-validation obligation that was not yet recorded."""
    if pending or champion is None:
        return pending
    report = state_dir / "brain-gate-reports" / f"{champion.genome_id}.smoke.json"
    return None if report.is_file() else champion.genome_id


def add_derived_features(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        features = row["features"]
        for name, expression in DERIVED_FEATURES.items():
            try:
                value = float(expression(features))
                features[name] = value if math.isfinite(value) else 0.0
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                features[name] = 0.0


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
            normalized.append((timestamp, max(-1.0, min(1.0, sentiment)), tokens, headline))
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
        asset = str(row["asset"]).upper()
        alias = aliases.get(asset, asset)
        asset_windows = {}
        for hours in (24, 72):
            relevant = [item for item in windows[hours]
                        if asset in item[2] or alias in item[2]
                        or re_word(alias, item[3])]
            asset_windows[hours] = relevant
            row["features"][f"asset_news_count_{hours}h"] = math.log1p(len(relevant))
        relevant24 = asset_windows[24]
        row["features"]["asset_news_sentiment_24h"] = (
            statistics.fmean(item[1] for item in relevant24) if relevant24 else 0.0)


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
    rows = [row for row in context_rows if row["target"] != 0]
    assets = sorted({row["asset"] for row in rows}, key=lambda value: stable_order(value, seed))
    holdout_n = max(1, round(len(assets) * .25))
    return {
        "rows": rows,
        "training_assets": set(assets[holdout_n:]),
        "holdout_assets": set(assets[:holdout_n]),
        "supplemental_assets": set(supplemental),
        "end": min(max(row["timestamp"] for row in rows if row["asset"] == asset)
                   for asset in assets),
        "assets": assets,
    }


def seed_genomes(population: int, rng: random.Random) -> list[Genome]:
    price = list(FEATURE_GROUPS["price"])
    flow = list(FEATURE_GROUPS["flow"])
    derivatives = list(FEATURE_GROUPS["derivatives"])
    breadth = list(FEATURE_GROUPS["breadth"])
    seeds = [price, price + flow, price + flow + derivatives,
             price + flow + breadth, price + flow + derivatives + breadth]
    result = [Genome(
        features=features, learning_rate=.06, max_iter=180, max_leaf_nodes=24,
        min_samples_leaf=20, l2_regularization=1.0, confidence_quantile=.20,
        binding_threshold=3, concept_threshold=5, presentations=3,
    ).finalize() for features in seeds]
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
        ).finalize())
    return result


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


def evaluate_slice(model: HistGradientBoostingClassifier, selected: list[dict[str, Any]],
                   names: Sequence[str], threshold: float, cost_bps: float) -> dict[str, Any]:
    x = np.asarray([[row["features"][name] for name in names] for row in selected], dtype=np.float32)
    actual = np.asarray([row["target"] for row in selected], dtype=np.int8)
    realized = np.asarray([row["return"] for row in selected], dtype=np.float64)
    probability = model.predict_proba(x)[:, list(model.classes_).index(1)]
    confidence = np.maximum(probability, 1 - probability)
    mask = confidence >= threshold
    if not mask.any():
        return {"observations": len(selected), "acted_observations": 0, "coverage": 0.0}
    predicted = model.predict(x)[mask]
    acted_rows = [row for row, keep in zip(selected, mask) if keep]
    result = metrics(actual[mask], predicted, probability[mask], realized[mask], cost_bps)
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


def evaluate_genome(genome: Genome, dataset: dict[str, Any], *, folds: int,
                    test_days: int, calibration_days: int, final_days: int,
                    horizon: int, cost_bps: float) -> Genome:
    started = time.perf_counter()
    rows = dataset["rows"]
    uses_derivatives = bool(set(genome.features) & DERIVATIVE_FEATURES)
    eligible = dataset["supplemental_assets"] if uses_derivatives else set(dataset["assets"])
    test_seconds = test_days * 86400
    calibration_seconds = calibration_days * 86400
    final_seconds = final_days * 86400
    cutoffs = [dataset["end"] - final_seconds - (folds - fold) * test_seconds
               for fold in range(folds)]
    fold_results = []
    try:
        for fold, cutoff in enumerate(cutoffs):
            fit_stop = cutoff - horizon * 3600 - calibration_seconds
            calibration_start = fit_stop + horizon * 3600
            fit = [row for row in rows if row["asset"] in dataset["training_assets"]
                   and row["asset"] in eligible and row["timestamp"] < fit_stop]
            calibration = [row for row in rows if row["asset"] in dataset["training_assets"]
                           and row["asset"] in eligible
                           and calibration_start <= row["timestamp"] < cutoff - horizon * 3600]
            known = [row for row in rows if row["asset"] in dataset["training_assets"]
                     and row["asset"] in eligible
                     and cutoff <= row["timestamp"] < cutoff + test_seconds]
            unseen = [row for row in rows if row["asset"] in dataset["holdout_assets"]
                      and row["asset"] in eligible
                      and cutoff <= row["timestamp"] < cutoff + test_seconds]
            if min(len(fit), len(calibration), len(known), len(unseen)) == 0:
                raise ValueError("one or more protected split sections are empty")
            names = genome.features
            x_fit = np.asarray([[row["features"][name] for name in names] for row in fit], dtype=np.float32)
            y_fit = np.asarray([row["target"] for row in fit], dtype=np.int8)
            model = HistGradientBoostingClassifier(
                learning_rate=genome.learning_rate, max_iter=genome.max_iter,
                max_leaf_nodes=genome.max_leaf_nodes,
                min_samples_leaf=genome.min_samples_leaf,
                l2_regularization=genome.l2_regularization, random_state=1700 + fold,
            ).fit(x_fit, y_fit)
            x_cal = np.asarray([[row["features"][name] for name in names]
                                for row in calibration], dtype=np.float32)
            cal_probability = model.predict_proba(x_cal)[:, list(model.classes_).index(1)]
            cal_confidence = np.maximum(cal_probability, 1 - cal_probability)
            threshold = float(np.quantile(cal_confidence, genome.confidence_quantile))
            fold_results.append({
                "fold": fold, "cutoff": cutoff, "fit_rows": len(fit),
                "calibration_rows": len(calibration), "confidence_threshold": threshold,
                "known_asset_future": evaluate_slice(model, known, names, threshold, cost_bps),
                "unseen_asset_future": evaluate_slice(model, unseen, names, threshold, cost_bps),
            })
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
        passed = all(passes_floor(section) for section in sections)
        penalty = (
            max(0, FLOOR["coverage"] - min_coverage) * 100
            + max(0, FLOOR["baseline_margin"] - min_margin) * 100
            + max(0, FLOOR["ece"] - max_ece) * 0
            + max(0, max_ece - FLOOR["ece"]) * 30
            + max(0, FLOOR["observations"] - min_observations) / 20
            + max(0, -min_expectancy) * 500
            + max(0, max_drawdown - FLOOR["max_drawdown"]) * 20
        )
        genome.fitness = (100 * min_accuracy + 35 * min_balanced + 30 * min_mcc
                          + 20 * min_margin + 3 * min(min_profit, 2.0) - penalty)
        working_target = (
            passed and min_accuracy >= WORKING_TARGET["accuracy"]
            and min_balanced >= WORKING_TARGET["balanced_accuracy"]
            and min_mcc >= WORKING_TARGET["mcc"]
            and min_profit >= WORKING_TARGET["profit_factor"]
        )
        genome.result = {
            "status": ("surrogate_working_target_pass" if working_target else
                       "surrogate_floor_pass" if passed else "screened"),
            "uses_derivatives": uses_derivatives,
            "folds": fold_results,
            "summary": {
                "min_accuracy": min_accuracy, "min_balanced_accuracy": min_balanced,
                "min_mcc": min_mcc, "min_baseline_margin": min_margin,
                "min_coverage": min_coverage, "min_acted_observations": min_observations,
                "min_expectancy": min_expectancy, "min_profit_factor": min_profit,
                "max_ece": max_ece, "max_drawdown": max_drawdown,
                "all_surrogate_floor_gates": passed,
                "surrogate_working_target": working_target,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        genome.fitness = -1_000_000.0
        genome.result = {"status": "failed", "error": repr(exc),
                         "elapsed_seconds": time.perf_counter() - started}
    return genome


def mutate(parent: Genome, generation: int, rng: random.Random) -> Genome:
    features = set(parent.features)
    universe = list(BASE_FEATURES) + list(DERIVED_FEATURES)
    for _ in range(rng.randint(1, 4)):
        if rng.random() < .55 and len(features) < 42:
            features.add(rng.choice(universe))
        elif len(features) > 8:
            features.remove(rng.choice(sorted(features)))
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
        generation=generation, parents=[left.genome_id, right.genome_id],
    ).finalize()
    return mutate(child, generation, rng) if rng.random() < .7 else child


def genome_from_dict(payload: dict[str, Any]) -> Genome:
    return Genome(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runtime/benchmarks/market-corpus-manifest.json")
    parser.add_argument("--supplemental-root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"))
    parser.add_argument("--news", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\news\historical_deduplicated.json"))
    parser.add_argument("--state-dir", type=Path, default=ROOT / "runtime/market-evolution")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-generations", type=int, default=0,
                        help="zero evolves perpetually")
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--seed", default="market-perpetual-v1")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--final-days", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--brain-gate-every", type=int, default=5,
                        help="launch one isolated Wizard smoke gate this many generations")
    parser.add_argument("--min-free-memory-gb", type=float, default=8.0)
    parser.add_argument("--memory-poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.population < 4:
        raise ValueError("population must be at least four")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.state_dir / "events.jsonl"
    state_path = args.state_dir / "state.json"
    stop_path = args.state_dir / "STOP"
    gate_process: subprocess.Popen | None = None
    gate_genome: str | None = None
    gate_out: Path | None = None
    pending_gate_genome: str | None = None
    owner = claim_service(args.state_dir)
    rng = random.Random(args.seed)
    while available_memory_gb() < args.min_free_memory_gb:
        atomic_json(args.state_dir / "status.json", {
            "at": utc_now(), "phase": "memory_wait_before_dataset",
            "available_memory_gb": available_memory_gb(),
            "required_memory_gb": args.min_free_memory_gb,
        })
        time.sleep(args.memory_poll_seconds)
    signature = dataset_signature(args.manifest, args.supplemental_root, args.news)
    dataset = load_dataset(args.manifest, args.supplemental_root,
                           args.horizon, args.stride, args.seed, args.news)
    append_event(events_path, "dataset_loaded", rows=len(dataset["rows"]),
                 assets=dataset["assets"], supplemental_assets=sorted(dataset["supplemental_assets"]))
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        generation = int(state["generation"])
        population = [genome_from_dict(row) for row in state["population"]]
        champion = genome_from_dict(state["champion"]) if state.get("champion") else None
        pending_gate_genome = state.get("pending_brain_gate")
        if state.get("dataset_signature") != signature:
            for genome in population:
                genome.fitness = None
                genome.result = None
            champion = None
            append_event(events_path, "dataset_changed", previous=state.get("dataset_signature"),
                         current=signature, action="population_revalidation")
        pending_gate_genome = recover_pending_gate(
            args.state_dir, champion, pending_gate_genome
        )
        append_event(events_path, "resumed", generation=generation, population=len(population))
    else:
        generation = 0
        population = seed_genomes(args.population, rng)
        champion = None
        append_event(events_path, "started", population=args.population)
    try:
        while not stop_path.exists() and (args.max_generations == 0 or generation < args.max_generations):
            if gate_process is not None and gate_process.poll() is not None:
                gate_report = (json.loads(gate_out.read_text(encoding="utf-8"))
                               if gate_out is not None and gate_out.is_file() else None)
                append_event(events_path, "brain_gate_finished", genome_id=gate_genome,
                             returncode=gate_process.returncode,
                             passed=(gate_report or {}).get("all_brain_floor_gates"))
                if gate_report and gate_report.get("all_brain_floor_gates"):
                    atomic_json(args.state_dir / "untouched-final-queue.json", {
                        "queued_at": utc_now(), "genome_id": gate_genome,
                        "brain_gate_report": str(gate_out),
                        "required_next_gate": "one-time untouched final period; no further selection on it",
                    })
                gate_process = None
                gate_genome = None
                gate_out = None
            while available_memory_gb() < args.min_free_memory_gb:
                atomic_json(args.state_dir / "status.json", {
                    "at": utc_now(), "phase": "memory_wait", "generation": generation,
                    "available_memory_gb": available_memory_gb(),
                    "required_memory_gb": args.min_free_memory_gb,
                })
                time.sleep(args.memory_poll_seconds)
            append_event(events_path, "generation_started", generation=generation,
                         genomes=[genome.genome_id for genome in population])
            pending = [genome for genome in population if genome.fitness is None]
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        evaluate_genome, genome, dataset, folds=args.folds,
                        test_days=args.test_days, calibration_days=args.calibration_days,
                        final_days=args.final_days, horizon=args.horizon, cost_bps=args.cost_bps,
                    ): genome for genome in pending
                }
                for future in as_completed(futures):
                    genome = future.result()
                    atomic_json(args.state_dir / "candidates" / f"{genome.genome_id}.json",
                                asdict(genome))
                    append_event(events_path, "candidate_scored", generation=generation,
                                 genome_id=genome.genome_id, fitness=genome.fitness,
                                 status=(genome.result or {}).get("status"),
                                 summary=(genome.result or {}).get("summary"))
                    print(f"generation {generation} {genome.genome_id} fitness={genome.fitness:.4f} "
                          f"{(genome.result or {}).get('status')}", flush=True)
            population.sort(key=lambda genome: genome.fitness if genome.fitness is not None else -math.inf,
                            reverse=True)
            champion_changed = (champion is None
                                or (population[0].fitness or -math.inf) > (champion.fitness or -math.inf))
            if champion_changed:
                champion = genome_from_dict(asdict(population[0]))
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
            if gate_process is None and pending_gate_genome is not None:
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
                    pending_gate_genome = None
                    append_event(events_path, "brain_gate_started", genome_id=gate_genome,
                                 pid=gate_process.pid,
                                 stage="full" if full_gate else "smoke", command=command)
                elif gate_out.exists():
                    append_event(events_path, "brain_gate_already_recorded",
                                 genome_id=pending_gate_genome, report=str(gate_out))
                    pending_gate_genome = None
            generation += 1
            elite_count = max(2, round(args.population * .25))
            elites = [genome_from_dict(asdict(genome)) for genome in population[:elite_count]]
            next_population = elites[:]
            known_ids = {genome.genome_id for genome in next_population}
            attempts = 0
            while len(next_population) < args.population and attempts < args.population * 50:
                attempts += 1
                contenders = rng.sample(population[:max(elite_count * 2, 4)], 2)
                child = crossover(contenders[0], contenders[1], generation, rng)
                if child.genome_id not in known_ids:
                    known_ids.add(child.genome_id)
                    next_population.append(child)
            population = next_population
            state = {
                "schema": 1, "updated_at": utc_now(), "generation": generation,
                "configuration": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                "contract": str(ROOT / "docs/MARKET_BRAIN_GHOST_TRADING_ADMISSION.md"),
                "working_target": WORKING_TARGET,
                "dataset_signature": signature,
                "pending_brain_gate": pending_gate_genome,
                "population": [asdict(genome) for genome in population],
                "champion": asdict(champion) if champion else None,
            }
            atomic_json(state_path, state)
            append_event(events_path, "generation_completed", generation=generation,
                         champion=champion.genome_id if champion else None,
                         champion_fitness=champion.fitness if champion else None)
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
