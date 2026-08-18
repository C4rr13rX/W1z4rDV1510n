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
from scripts.market_evolution_genome import load_genome  # noqa: E402


STABLE_QUOTES = ("USDC", "USDT", "USDT0", "DAI", "USD")
FEATURE_GROUPS = {
    "price": (
        "body", "range", "position", "upper_wick", "lower_wick",
        "r1", "r2", "r3", "r6", "r12", "r24", "r72", "r168",
        "acceleration", "rv6", "rv24", "rv168", "volatility_ratio",
        "drawdown", "location", "trend_vote", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "mean_r6", "mean_r24", "mean_r72", "return_autocorr24",
        "return_skew24", "return_skew168", "rsi14", "price_sma_gap24",
        "price_sma_gap72", "price_sma_gap168", "breakout24", "breakout72",
        "range_ratio24", "downside_ratio24", "trend_efficiency24",
    ),
    "flow": (
        "volume_ratio24", "volume_ratio168", "flow_imbalance",
        "volume_acceleration", "flow_mean6", "flow_mean24", "flow_acceleration",
    ),
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
        "basis_z24", "basis_z168", "funding_mean24", "funding_z168",
        "spot_imbalance_mean6", "spot_imbalance_mean24",
        "futures_imbalance_mean6", "futures_imbalance_mean24",
        "imbalance_acceleration", "derivative_volume_acceleration",
    ),
    "breadth": (
        "market_median_r1", "market_median_r6", "market_median_r24",
        "market_breadth_r1", "market_breadth_r6", "market_dispersion_r6",
        "relative_market_r6", "relative_market_r24",
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


def supplemental_symbol(base_asset: str) -> str:
    return {"WBTC": "BTCUSDT", "WETH": "ETHUSDT"}.get(base_asset, f"{base_asset}USDT")


def derive_supplemental_features(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Build rolling features from information published at or before each row."""
    result: dict[int, dict[str, float]] = {}
    basis = [float(row.get("futures_spot_basis") or 0.0) for row in rows]
    premium = [float(row.get("premium_close") or 0.0) for row in rows]
    funding = [float(row.get("funding_rate") or 0.0) for row in rows]
    def prefix_moments(values: Sequence[float]) -> tuple[list[float], list[float]]:
        sums = [0.0]
        squares = [0.0]
        for value in values:
            sums.append(sums[-1] + value)
            squares.append(squares[-1] + value * value)
        return sums, squares

    basis_sums, basis_squares = prefix_moments(basis)
    funding_sums, funding_squares = prefix_moments(funding)

    def moment_mean(sums: Sequence[float], index: int, window: int) -> float:
        start = max(0, index - window + 1)
        return (sums[index + 1] - sums[start]) / (index + 1 - start)

    def moment_z(values: Sequence[float], sums: Sequence[float], squares: Sequence[float],
                 index: int, window: int) -> float:
        start = max(0, index - window + 1)
        count = index + 1 - start
        if count < 6:
            return 0.0
        mean = (sums[index + 1] - sums[start]) / count
        square_mean = (squares[index + 1] - squares[start]) / count
        deviation = math.sqrt(max(0.0, square_mean - mean * mean))
        return (values[index] - mean) / max(deviation, 1e-12)

    spot_imbalances: list[float] = []
    futures_imbalances: list[float] = []
    for index, row in enumerate(rows):
        spot_volume = float(row.get("spot_base_volume") or 0.0)
        futures_volume = float(row.get("futures_base_volume") or 0.0)
        spot_buy = float(row.get("spot_taker_buy_base") or 0.0)
        futures_buy = float(row.get("futures_taker_buy_base") or 0.0)
        spot_quote = float(row.get("spot_quote_volume") or 0.0)
        futures_quote = float(row.get("futures_quote_volume") or 0.0)
        spot_trades = float(row.get("spot_trade_count") or 0.0)
        futures_trades = float(row.get("futures_trade_count") or 0.0)
        start = max(0, index - 23)

        def ratio(key: str, current: float) -> float:
            history = [float(value.get(key) or 0.0) for value in rows[start:index + 1]]
            return current / max(statistics.median(history), 1e-12)

        spot_imbalance = 2.0 * spot_buy / spot_volume - 1.0 if spot_volume else 0.0
        futures_imbalance = 2.0 * futures_buy / futures_volume - 1.0 if futures_volume else 0.0
        spot_imbalances.append(spot_imbalance)
        futures_imbalances.append(futures_imbalance)

        def rolling_mean(values: Sequence[float], window: int) -> float:
            return statistics.fmean(values[max(0, index - window + 1):index + 1])

        spot_mean6 = rolling_mean(spot_imbalances, 6)
        spot_mean24 = rolling_mean(spot_imbalances, 24)
        futures_mean6 = rolling_mean(futures_imbalances, 6)
        futures_mean24 = rolling_mean(futures_imbalances, 24)
        result[int(row["timestamp"])] = {
            "binance_spot_close": float(row.get("spot_close") or 0.0),
            "futures_spot_basis": basis[index],
            "basis_delta6": basis[index] - basis[max(0, index - 6)],
            "basis_delta24": basis[index] - basis[max(0, index - 24)],
            "premium_close": premium[index],
            "premium_delta6": premium[index] - premium[max(0, index - 6)],
            "funding_rate": funding[index],
            "funding_delta24": funding[index] - funding[max(0, index - 24)],
            "spot_taker_imbalance": spot_imbalance,
            "futures_taker_imbalance": futures_imbalance,
            "flow_divergence": futures_imbalance - spot_imbalance,
            "spot_quote_ratio24": ratio("spot_quote_volume", spot_quote),
            "futures_quote_ratio24": ratio("futures_quote_volume", futures_quote),
            "spot_trade_ratio24": ratio("spot_trade_count", spot_trades),
            "futures_trade_ratio24": ratio("futures_trade_count", futures_trades),
            "futures_spot_quote_ratio": futures_quote / max(spot_quote, 1e-12),
            "basis_z24": moment_z(basis, basis_sums, basis_squares, index, 24),
            "basis_z168": moment_z(basis, basis_sums, basis_squares, index, 168),
            "funding_mean24": moment_mean(funding_sums, index, 24),
            "funding_z168": moment_z(funding, funding_sums, funding_squares, index, 168),
            "spot_imbalance_mean6": spot_mean6,
            "spot_imbalance_mean24": spot_mean24,
            "futures_imbalance_mean6": futures_mean6,
            "futures_imbalance_mean24": futures_mean24,
            "imbalance_acceleration": ((futures_mean6 - spot_mean6)
                                       - (futures_mean24 - spot_mean24)),
            "derivative_volume_acceleration": (
                ratio("futures_quote_volume", futures_quote)
                - ratio("spot_quote_volume", spot_quote)
            ),
        }
    return result


def load_supplemental(root: Path, records: Sequence[dict[str, Any]]) -> dict[str, dict[int, dict[str, float]]]:
    loaded = {}
    cache: dict[Path, dict[int, dict[str, float]]] = {}
    for record in records:
        asset = str(record["base_asset"])
        path = root / "features" / f"{supplemental_symbol(asset)}.json"
        if not path.is_file():
            continue
        if path not in cache:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cache[path] = derive_supplemental_features(payload.get("rows", []))
        features = cache[path]
        if features:
            loaded[asset] = features
    return loaded


def continuous_features(
    bars: Sequence[dict[str, float]],
    index: int,
    reference: Sequence[dict[str, float]],
    reference_times: Sequence[float],
    supplemental: dict[int, dict[str, float]] | None = None,
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
    flow_history = []
    for prior in bars[index - 23:index + 1]:
        prior_total = prior["buy_volume"] + prior["sell_volume"]
        flow_history.append((prior["buy_volume"] - prior["sell_volume"]) / prior_total
                            if prior_total else 0.0)
    ranges = [max(0.0, prior["high"] - prior["low"]) / max(abs(prior["open"]), 1e-12)
              for prior in bars[index - 23:index + 1]]
    closes = [prior["close"] for prior in bars[index - 167:index + 1]]
    recent24 = one_bar[-24:]
    mean24 = statistics.fmean(recent24)
    std24 = statistics.pstdev(recent24)
    skew24 = (statistics.fmean(((value - mean24) / std24) ** 3 for value in recent24)
              if std24 > 1e-12 else 0.0)
    mean168 = statistics.fmean(one_bar)
    std168 = statistics.pstdev(one_bar)
    skew168 = (statistics.fmean(((value - mean168) / std168) ** 3 for value in one_bar)
               if std168 > 1e-12 else 0.0)
    autocorr24 = (float(np.corrcoef(recent24[:-1], recent24[1:])[0, 1])
                  if np.std(recent24[:-1]) > 1e-12 and np.std(recent24[1:]) > 1e-12 else 0.0)
    gains14 = sum(max(0.0, value) for value in one_bar[-14:])
    losses14 = sum(max(0.0, -value) for value in one_bar[-14:])
    rsi14 = gains14 / max(gains14 + losses14, 1e-12)
    high24 = max(prior["high"] for prior in bars[index - 23:index + 1])
    low24 = min(prior["low"] for prior in bars[index - 23:index + 1])
    high72 = max(prior["high"] for prior in bars[index - 71:index + 1])
    low72 = min(prior["low"] for prior in bars[index - 71:index + 1])
    dt = datetime.fromtimestamp(bar["timestamp"], timezone.utc)
    external = (supplemental or {}).get(int(bar["timestamp"]), {})
    binance_spot = external.get("binance_spot_close", 0.0)

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

    features = {
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
        "mean_r6": statistics.fmean(one_bar[-6:]),
        "mean_r24": mean24,
        "mean_r72": statistics.fmean(one_bar[-72:]),
        "return_autocorr24": autocorr24,
        "return_skew24": skew24,
        "return_skew168": skew168,
        "rsi14": rsi14,
        "price_sma_gap24": safe_return(close, statistics.fmean(closes[-24:])),
        "price_sma_gap72": safe_return(close, statistics.fmean(closes[-72:])),
        "price_sma_gap168": safe_return(close, statistics.fmean(closes)),
        "breakout24": (close - low24) / max(high24 - low24, 1e-12) - .5,
        "breakout72": (close - low72) / max(high72 - low72, 1e-12) - .5,
        "range_ratio24": ranges[-1] / max(statistics.median(ranges), 1e-12),
        "downside_ratio24": sum(value < 0 for value in recent24) / 24.0,
        "trend_efficiency24": abs(values[24]) / max(sum(abs(value) for value in recent24), 1e-12),
        "volume_acceleration": (statistics.fmean(volumes[-6:])
                                / max(statistics.fmean(volumes[-24:]), 1e-12) - 1.0),
        "flow_mean6": statistics.fmean(flow_history[-6:]),
        "flow_mean24": statistics.fmean(flow_history),
        "flow_acceleration": (statistics.fmean(flow_history[-6:])
                              - statistics.fmean(flow_history)),
        **{f"reference_r{window}": value for window, value in reference_returns.items()},
        **{f"relative_r{window}": values[window] - reference_returns[window]
           for window in reference_returns},
        "rolling_beta": beta,
        "residual_r12": values[12] - beta * reference_returns[12],
    }
    features.update({name: float(external.get(name, 0.0))
                     for name in FEATURE_GROUPS["derivatives"] if name != "cex_dex_basis"})
    features["cex_dex_basis"] = safe_return(close, binance_spot) if binance_spot else 0.0
    return features


def build_rows(record: dict[str, Any], bars: Sequence[dict[str, float]], *,
               reference: Sequence[dict[str, float]], reference_times: Sequence[float],
               horizon: int, stride: int, direction_threshold: float = 0.003,
               supplemental: dict[int, dict[str, float]] | None = None,
               auxiliary_horizons: Sequence[int] = (1, 6, 12, 24),
               ) -> list[dict[str, Any]]:
    rows = []
    target_horizons = sorted({horizon, *(int(value) for value in auxiliary_horizons
                                        if int(value) > 0)})
    for index in range(168, len(bars) - max(target_horizons), stride):
        realized = safe_return(bars[index + horizon]["close"], bars[index]["close"])
        future_returns = {
            str(value): safe_return(bars[index + value]["close"], bars[index]["close"])
            for value in target_horizons
        }
        rows.append({
            "asset": record["base_asset"],
            "timestamp": bars[index]["timestamp"],
            "return": realized,
            "target": (1 if realized > direction_threshold else
                       -1 if realized < -direction_threshold else 0),
            "future_returns": future_returns,
            "features": continuous_features(bars, index, reference, reference_times, supplemental),
        })
    return rows


def attach_market_breadth(rows: Sequence[dict[str, Any]]) -> None:
    """Attach same-time cross-sectional state without consulting any target."""
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["timestamp"], []).append(row)
    for members in groups.values():
        r1 = [float(row["features"]["r1"]) for row in members]
        r6 = [float(row["features"]["r6"]) for row in members]
        r24 = [float(row["features"]["r24"]) for row in members]
        state = {
            "market_median_r1": statistics.median(r1),
            "market_median_r6": statistics.median(r6),
            "market_median_r24": statistics.median(r24),
            "market_breadth_r1": statistics.fmean(value > 0 for value in r1),
            "market_breadth_r6": statistics.fmean(value > 0 for value in r6),
            "market_dispersion_r6": statistics.pstdev(r6) if len(r6) > 1 else 0.0,
        }
        for row in members:
            row["features"].update(state)
            row["features"]["relative_market_r6"] = row["features"]["r6"] - state["market_median_r6"]
            row["features"]["relative_market_r24"] = row["features"]["r24"] - state["market_median_r24"]


def metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray,
            realized: np.ndarray, cost_bps: float,
            turnover: np.ndarray | None = None) -> dict[str, Any]:
    """Score one slice of predictions.

    `cost_bps` is a ROUND-TRIP execution cost, so it may only be charged when
    the position actually changes. Charging it on every bar -- which this did
    unconditionally -- bills a full entry+exit for merely continuing to hold
    the same direction, and at a 12-period horizon that overstates cost by
    the average holding length. Measured 2026-08-17: the champion's gross
    edge is PF 1.18, and the per-bar charge dragged it to 0.95.

    `turnover` is a per-observation multiplier in [0, 1]: 1.0 when the
    position changed (a real round trip), 0.0 when it was simply held. When
    it is not supplied we fall back to the old per-bar charge, which is
    conservative -- it can only understate profit, never invent it.
    """
    if turnover is None:
        charge = np.full(len(predicted), cost_bps / 10_000.0, dtype=np.float64)
    else:
        charge = np.asarray(turnover, dtype=np.float64) * (cost_bps / 10_000.0)
    pnl = predicted * realized - charge
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
    parser.add_argument("--supplemental-root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"))
    parser.add_argument("--feature-sets",
                        help="comma-separated ablations to run; default runs every configured set")
    parser.add_argument("--permutation-repeats", type=int, default=0,
                        help="diagnostic feature-importance repeats; zero skips the expensive ranking")
    parser.add_argument("--genome", type=Path,
                        help="market-evolution genome JSON; overrides evolvable evaluation settings")
    args = parser.parse_args()
    genome = None
    if args.genome:
        genome = load_genome(json.loads(args.genome.read_text(encoding="utf-8")))
        args.horizon = genome.horizon
        args.stride = genome.stride

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = select_primary_assets(manifest["selected"])
    reference_record = next(
        (row for row in records if row["base_asset"] == "WBTC"),
        next(row for row in records if row["base_asset"] == "WETH"),
    )
    reference = load_bars(Path(reference_record["path"]))
    reference_times = [row["timestamp"] for row in reference]
    supplemental = load_supplemental(args.supplemental_root, records)
    all_context_rows: list[dict[str, Any]] = []
    data_quality = Counter()
    for position, record in enumerate(records, 1):
        bars = load_bars(Path(record["path"]))
        data_quality["bars"] += len(bars)
        data_quality["bars_with_flow"] += sum(
            bool(row["buy_volume"] + row["sell_volume"]) for row in bars
        )
        rows = build_rows(record, bars, reference=reference, reference_times=reference_times,
                          horizon=args.horizon, stride=args.stride,
                          direction_threshold=(genome.direction_threshold if genome else 0.003),
                          supplemental=supplemental.get(str(record["base_asset"])))
        all_context_rows.extend(rows)
        directional_count = sum(row["target"] != 0 for row in rows)
        print(f"[{position}/{len(records)}] {record['base_asset']}: "
              f"{directional_count} directional moments", flush=True)

    attach_market_breadth(all_context_rows)
    all_rows = [row for row in all_context_rows if row["target"] != 0]

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
        "price_flow_derivatives": (FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"]
                                   + FEATURE_GROUPS["derivatives"]),
        "price_flow_cross_derivatives": (FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"]
                                         + FEATURE_GROUPS["cross"]
                                         + FEATURE_GROUPS["derivatives"]),
        "price_flow_breadth": (FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"]
                               + FEATURE_GROUPS["breadth"]),
        "price_flow_derivatives_breadth": (FEATURE_GROUPS["price"] + FEATURE_GROUPS["flow"]
                                           + FEATURE_GROUPS["derivatives"]
                                           + FEATURE_GROUPS["breadth"]),
    }
    if args.feature_sets:
        requested = [value.strip() for value in args.feature_sets.split(",") if value.strip()]
        unknown = set(requested) - set(feature_sets)
        if unknown:
            raise ValueError(f"unknown feature sets: {sorted(unknown)}")
        feature_sets = {name: feature_sets[name] for name in requested}
    if genome:
        feature_sets = {"genome": genome.features}
    report: dict[str, Any] = {
        "purpose": "diagnostic_only_not_a_candidate_trading_model",
        "configuration": vars(args) | {
            "manifest": str(args.manifest), "out": str(args.out),
            "supplemental_root": str(args.supplemental_root),
            "genome": str(args.genome) if args.genome else None,
        },
        "genome": genome.as_json() if genome else None,
        "reference": {key: reference_record[key] for key in ("base_asset", "symbol", "chain", "path")},
        "assets": {"training": sorted(training_assets), "holdout": sorted(holdout_assets)},
        "data_quality": dict(data_quality),
        "supplemental_assets": sorted(supplemental),
        "feature_sets": {},
    }
    for set_name, names in feature_sets.items():
        eligible_assets = (set(supplemental) if any(name in FEATURE_GROUPS["derivatives"] for name in names)
                           else {row["asset"] for row in all_rows})
        folds = []
        for fold_index, cutoff in enumerate(cutoffs):
            train = [row for row in all_rows
                     if row["asset"] in training_assets
                     and row["asset"] in eligible_assets
                     and row["timestamp"] < cutoff - args.horizon * 3600]
            known = [row for row in all_rows
                     if row["asset"] in training_assets and row["asset"] in eligible_assets
                     and cutoff <= row["timestamp"] < cutoff + test_seconds]
            unseen = [row for row in all_rows
                      if row["asset"] in holdout_assets and row["asset"] in eligible_assets
                      and cutoff <= row["timestamp"] < cutoff + test_seconds]
            if not train or not known or not unseen:
                folds.append({"fold": fold_index, "skipped": "insufficient eligible rows"})
                continue
            x_train = np.asarray([[row["features"][name] for name in names] for row in train], dtype=np.float32)
            y_train = np.asarray([row["target"] for row in train], dtype=np.int8)
            model = HistGradientBoostingClassifier(
                learning_rate=genome.learning_rate if genome else 0.06,
                max_iter=genome.max_iter if genome else 180,
                max_leaf_nodes=genome.max_leaf_nodes if genome else 24,
                l2_regularization=genome.l2_regularization if genome else 1.0,
                random_state=17,
            ).fit(x_train, y_train)
            train_probability = model.predict_proba(x_train)[:, list(model.classes_).index(1)]
            train_confidence = np.maximum(train_probability, 1.0 - train_probability)
            confidence_quantile = genome.confidence_quantile if genome else 0.30
            confidence_threshold = float(np.quantile(train_confidence, confidence_quantile))
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
                section_metrics["baseline_margin"] = (
                    section_metrics["directional_accuracy"]
                    - max(section_metrics["momentum_accuracy"], section_metrics["inverse_momentum_accuracy"])
                )
                confidence = np.maximum(probability, 1.0 - probability)
                acted = confidence >= confidence_threshold
                if acted.any():
                    selective = metrics(actual[acted], predicted[acted], probability[acted],
                                        realized[acted], args.cost_bps)
                    selective["coverage"] = float(acted.mean())
                    selective["acted_observations"] = int(acted.sum())
                    selective["momentum_accuracy"] = float((momentum[acted] == actual[acted]).mean())
                    selective["inverse_momentum_accuracy"] = float((-momentum[acted] == actual[acted]).mean())
                    selective["baseline_margin"] = (
                        selective["directional_accuracy"]
                        - max(selective["momentum_accuracy"], selective["inverse_momentum_accuracy"])
                    )
                else:
                    selective = {"coverage": 0.0, "acted_observations": 0}
                section_metrics["selective"] = selective
                sections[section_name] = section_metrics
            sample = known[:min(2500, len(known))]
            if sample and args.permutation_repeats > 0:
                x_sample = np.asarray([[row["features"][name] for name in names] for row in sample], dtype=np.float32)
                y_sample = np.asarray([row["target"] for row in sample], dtype=np.int8)
                importance = permutation_importance(model, x_sample, y_sample,
                                                    n_repeats=args.permutation_repeats,
                                                    random_state=23, scoring="balanced_accuracy")
                top = sorted(zip(names, importance.importances_mean), key=lambda item: item[1], reverse=True)[:10]
            else:
                top = []
            folds.append({
                "fold": fold_index,
                "cutoff": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                "training_rows": len(train),
                "training_confidence_threshold": confidence_threshold,
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
