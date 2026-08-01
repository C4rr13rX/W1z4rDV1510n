#!/usr/bin/env python3
"""Diagnose whether the current causal market corpus contains usable signal.

This is deliberately not a candidate production model.  It is an ablation
instrument: a conventional classifier receives continuous versions of the
same causal feature families offered to the Wizard brain.  If it cannot clear
the admission floor on chronological and entire-asset holdouts, more neural
training cannot manufacture information absent from those streams.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_brain_experiment import load_bars, safe_return  # noqa: E402


STABLE_QUOTES = ("USDC", "USDT", "USDT0", "DAI", "USD")
FEATURE_GROUPS = {
    "price": (
        "body", "range", "position", "upper_wick", "lower_wick",
        "r1", "r2", "r3", "r6", "r12", "r24", "r72", "r168",
        "acceleration", "rv6", "rv24", "rv168", "volatility_ratio",
        "drawdown", "location", "trend_vote", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
    ),
    "flow": ("volume_ratio24", "volume_ratio168", "flow_imbalance"),
    "cross": (
        "reference_r1", "reference_r6", "reference_r12", "reference_r24",
        "relative_r1", "relative_r6", "relative_r12", "relative_r24",
        "rolling_beta", "residual_r12",
    ),
}


def stable_order(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def select_primary_assets(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one liquid representative per base asset to remove copy leakage."""
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        base = str(record["base_asset"])
        quote = str(record["quote_asset"])
        rank = (
            0 if quote in STABLE_QUOTES else 1,
            -int(record.get("rows") or 0),
            str(record.get("relative_path") or record.get("path")),
        )
        current = selected.get(base)
        if current is None or rank < current["_selection_rank"]:
            selected[base] = {**record, "_selection_rank": rank}
    return [{key: value for key, value in row.items() if key != "_selection_rank"}
            for row in sorted(selected.values(), key=lambda item: item["base_asset"])]


def _returns(bars: Sequence[dict[str, float]], index: int) -> dict[int, float]:
    close = bars[index]["close"]
    return {window: safe_return(close, bars[index - window]["close"])
            for window in (1, 2, 3, 6, 12, 24, 72, 168)}


def continuous_features(
    bars: Sequence[dict[str, float]],
    index: int,
    reference: Sequence[dict[str, float]],
    reference_times: Sequence[float],
) -> dict[str, float]:
    """Produce causal features using rows at or before the decision timestamp."""
    bar = bars[index]
    opened, high, low, close = (bar[key] for key in ("open", "high", "low", "close"))
    denom = max(abs(opened), 1e-12)
    values = _returns(bars, index)
    volumes = [row["volume"] for row in bars[index - 167:index + 1]]
    one_bar = [safe_return(bars[i]["close"], bars[i - 1]["close"])
               for i in range(index - 167, index + 1)]
    rv6 = statistics.pstdev(one_bar[-6:])
    rv24 = statistics.pstdev(one_bar[-24:])
    rv168 = statistics.pstdev(one_bar)
    rolling = bars[index - 167:index + 1]
    rolling_high = max(row["high"] for row in rolling)
    rolling_low = min(row["low"] for row in rolling)
    total_flow = bar["buy_volume"] + bar["sell_volume"]
    dt = datetime.fromtimestamp(bar["timestamp"], timezone.utc)

    ref_index = bisect.bisect_right(reference_times, bar["timestamp"]) - 1
    reference_returns = {window: 0.0 for window in (1, 6, 12, 24)}
    if ref_index >= 168:
        reference_returns = {
            window: safe_return(reference[ref_index]["close"], reference[ref_index - window]["close"])
            for window in reference_returns
        }
    aligned_asset = [safe_return(bars[i]["close"], bars[i - 1]["close"])
                     for i in range(index - 167, index + 1)]
    aligned_reference: list[float] = []
    for i in range(index - 167, index + 1):
        ri = bisect.bisect_right(reference_times, bars[i]["timestamp"]) - 1
        aligned_reference.append(
            safe_return(reference[ri]["close"], reference[ri - 1]["close"])
            if ri >= 1 else 0.0
        )
    ref_variance = float(np.var(aligned_reference))
    beta = (float(np.cov(aligned_asset, aligned_reference, ddof=0)[0, 1]) / ref_variance
            if ref_variance > 1e-12 else 0.0)

    return {
        "body": (close - opened) / denom,
        "range": max(0.0, high - low) / denom,
        "position": (close - low) / (high - low) if high > low else 0.5,
        "upper_wick": (high - max(opened, close)) / denom,
        "lower_wick": (min(opened, close) - low) / denom,
        **{f"r{window}": value for window, value in values.items()},
        "acceleration": values[6] - values[24],
        "volume_ratio24": bar["volume"] / max(statistics.median(volumes[-24:]), 1e-12),
        "volume_ratio168": bar["volume"] / max(statistics.median(volumes), 1e-12),
        "flow_imbalance": ((bar["buy_volume"] - bar["sell_volume"]) / total_flow
                           if total_flow else 0.0),
        "rv6": rv6,
        "rv24": rv24,
        "rv168": rv168,
        "volatility_ratio": rv24 / max(rv168, 1e-12),
        "drawdown": safe_return(close, rolling_high),
        "location": ((close - rolling_low) / (rolling_high - rolling_low)
                     if rolling_high > rolling_low else 0.5),
        "trend_vote": float(sum(1 if values[w] > 0 else -1 if values[w] < 0 else 0
                                for w in (6, 12, 24, 72, 168))),
        "hour_sin": math.sin(2 * math.pi * dt.hour / 24),
        "hour_cos": math.cos(2 * math.pi * dt.hour / 24),
        "dow_sin": math.sin(2 * math.pi * dt.weekday() / 7),
        "dow_cos": math.cos(2 * math.pi * dt.weekday() / 7),
        **{f"reference_r{window}": value for window, value in reference_returns.items()},
        **{f"relative_r{window}": values[window] - reference_returns[window]
           for window in reference_returns},
        "rolling_beta": beta,
        "residual_r12": values[12] - beta * reference_returns[12],
    }


def build_rows(record: dict[str, Any], bars: Sequence[dict[str, float]], *,
               reference: Sequence[dict[str, float]], reference_times: Sequence[float],
               horizon: int, stride: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(168, len(bars) - horizon, stride):
        realized = safe_return(bars[index + horizon]["close"], bars[index]["close"])
        if abs(realized) <= 0.003:
            continue
        rows.append({
            "asset": record["base_asset"],
            "timestamp": bars[index]["timestamp"],
            "return": realized,
            "target": 1 if realized > 0 else -1,
            "features": continuous_features(bars, index, reference, reference_times),
        })
    return rows


def metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray,
            realized: np.ndarray, cost_bps: float) -> dict[str, Any]:
    pnl = predicted * realized - cost_bps / 10_000.0
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    confidence = np.maximum(probability, 1.0 - probability)
    correct = (actual == predicted).astype(float)
    ece = 0.0
    for lower in np.arange(0.5, 1.0, 0.05):
        mask = (confidence >= lower) & (confidence < lower + 0.05 + 1e-12)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return {
        "observations": int(len(actual)),
        "directional_accuracy": float(correct.mean()),
        "directional_balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "mcc": float(matthews_corrcoef(actual, predicted)),
        "ece": ece,
        "net_expectancy": float(pnl.mean()),
        "profit_factor": gains / losses if losses else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runtime/benchmarks/market-corpus-manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runtime/benchmarks/market-signal-audit.json")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--seed", default="market-signal-audit-v1")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = select_primary_assets(manifest["selected"])
    reference_record = next(
        (row for row in records if row["base_asset"] == "WBTC"),
        next(row for row in records if row["base_asset"] == "WETH"),
    )
    reference = load_bars(Path(reference_record["path"]))
    reference_times = [row["timestamp"] for row in reference]
    all_rows: list[dict[str, Any]] = []
    data_quality = Counter()
    for position, record in enumerate(records, 1):
        bars = load_bars(Path(record["path"]))
        data_quality["bars"] += len(bars)
        data_quality["bars_with_flow"] += sum(
            bool(row["buy_volume"] + row["sell_volume"]) for row in bars
        )
        rows = build_rows(record, bars, reference=reference, reference_times=reference_times,
                          horizon=args.horizon, stride=args.stride)
        all_rows.extend(rows)
        print(f"[{position}/{len(records)}] {record['base_asset']}: {len(rows)} directional moments", flush=True)

    assets = sorted({row["asset"] for row in all_rows}, key=lambda value: stable_order(value, args.seed))
    holdout_n = max(1, round(len(assets) * args.holdout_fraction))
    holdout_assets = set(assets[:holdout_n])
    training_assets = set(assets[holdout_n:])
    end = min(max(row["timestamp"] for row in all_rows if row["asset"] == asset) for asset in assets)
    test_seconds = args.test_days * 86400
    cutoffs = [end - (args.folds - fold) * test_seconds for fold in range(args.folds)]
    feature_sets = {
        "price": FEATURE_GROUPS["price"],
        "price_flow": FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"],
        "price_flow_cross": FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"] + FEATURE_GROUPS["cross"],
    }
    report: dict[str, Any] = {
        "purpose": "diagnostic_only_not_a_candidate_trading_model",
        "configuration": vars(args) | {"manifest": str(args.manifest), "out": str(args.out)},
        "reference": {key: reference_record[key] for key in ("base_asset", "symbol", "chain", "path")},
        "assets": {"training": sorted(training_assets), "holdout": sorted(holdout_assets)},
        "data_quality": dict(data_quality),
        "feature_sets": {},
    }
    for set_name, names in feature_sets.items():
        folds = []
        for fold_index, cutoff in enumerate(cutoffs):
            train = [row for row in all_rows
                     if row["asset"] in training_assets
                     and row["timestamp"] < cutoff - args.horizon * 3600]
            known = [row for row in all_rows
                     if row["asset"] in training_assets and cutoff <= row["timestamp"] < cutoff + test_seconds]
            unseen = [row for row in all_rows
                      if row["asset"] in holdout_assets and cutoff <= row["timestamp"] < cutoff + test_seconds]
            x_train = np.asarray([[row["features"][name] for name in names] for row in train], dtype=np.float32)
            y_train = np.asarray([row["target"] for row in train], dtype=np.int8)
            model = HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=180, max_leaf_nodes=24,
                l2_regularization=1.0, random_state=17,
            ).fit(x_train, y_train)
            sections = {}
            for section_name, selected in (("known_asset_future", known), ("unseen_asset_future", unseen)):
                x = np.asarray([[row["features"][name] for name in names] for row in selected], dtype=np.float32)
                actual = np.asarray([row["target"] for row in selected], dtype=np.int8)
                realized = np.asarray([row["return"] for row in selected], dtype=np.float64)
                predicted = model.predict(x)
                probability = model.predict_proba(x)[:, list(model.classes_).index(1)]
                section_metrics = metrics(actual, predicted, probability, realized, args.cost_bps)
                momentum = np.asarray([1 if row["features"]["r12"] > 0 else -1 for row in selected], dtype=np.int8)
                section_metrics["momentum_accuracy"] = float((momentum == actual).mean())
                section_metrics["inverse_momentum_accuracy"] = float((-momentum == actual).mean())
                sections[section_name] = section_metrics
            sample = known[:min(2500, len(known))]
            if sample:
                x_sample = np.asarray([[row["features"][name] for name in names] for row in sample], dtype=np.float32)
                y_sample = np.asarray([row["target"] for row in sample], dtype=np.int8)
                importance = permutation_importance(model, x_sample, y_sample, n_repeats=3,
                                                    random_state=23, scoring="balanced_accuracy")
                top = sorted(zip(names, importance.importances_mean), key=lambda item: item[1], reverse=True)[:10]
            else:
                top = []
            folds.append({
                "fold": fold_index,
                "cutoff": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                "training_rows": len(train),
                "sections": sections,
                "top_permutation_features": [{"feature": name, "importance": float(value)} for name, value in top],
            })
            print(f"{set_name} fold {fold_index}: " + json.dumps(sections, separators=(",", ":")), flush=True)
        report["feature_sets"][set_name] = folds

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
