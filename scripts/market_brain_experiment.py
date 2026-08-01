#!/usr/bin/env python3
"""Leakage-resistant multi-pool OHLCV + news Wizard brain experiment.

This script never modifies source corpora.  Training uses one supervised
Hebbian moment containing several atom-grounded feature streams and a confirmed
future outcome.  Evaluation uses the read-only multi-pool prediction endpoint.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import json
import math
import statistics
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

POOL_OHLCV = 1
POOL_TEMPORAL = 2
POOL_FLOW = 3
POOL_VOLATILITY = 4
POOL_REGIME = 5
POOL_CROSS_MARKET = 6
POOL_NEWS_ENTITIES = 7
POOL_NEWS_STATE = 8
POOL_HORIZON = 9
POOL_INSTRUMENT = 10
POOL_OUTCOME = 11

# Byte-disjoint labels prevent a frequent short label from being a substring
# of a rarer label, a failure measured in the first market-brain experiments.
RETURN_LABELS = ("crater", "dive", "slip", "still", "lift", "rally", "soar")
RETURN_EDGES = (-0.03, -0.012, -0.003, 0.003, 0.012, 0.03)
RETURN_CENTERS = {
    "crater": -0.045, "dive": -0.020, "slip": -0.007,
    "still": 0.0, "lift": 0.007, "rally": 0.020, "soar": 0.045,
}
DIRECTION_LABELS = ("downshift", "sideways", "updraft")


def timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 10_000_000_000 else number
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("rows", value.get("articles", []))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected rows/articles JSON list")
    return rows


def load_bars(path: Path) -> list[dict[str, float]]:
    rows: dict[float, dict[str, float]] = {}
    for raw in load_json_rows(path):
        try:
            ts = timestamp(raw["timestamp"])
            close = float(raw["close"])
            if not (ts > 0 and close > 0):
                continue
            rows[ts] = {
                "timestamp": ts,
                "open": float(raw.get("open", close)),
                "high": float(raw.get("high", close)),
                "low": float(raw.get("low", close)),
                "close": close,
                "volume": abs(float(raw.get("volume") or raw.get("net_volume") or 0.0)),
                "buy_volume": abs(float(raw.get("buy_volume") or 0.0)),
                "sell_volume": abs(float(raw.get("sell_volume") or 0.0)),
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return [rows[key] for key in sorted(rows)]


@dataclass(frozen=True)
class NewsItem:
    timestamp: float
    headline: str
    sentiment: float
    tokens: tuple[str, ...]


def load_news(path: Path | None) -> list[NewsItem]:
    if path is None:
        return []
    result: list[NewsItem] = []
    sentiment_value = {"negative": -1.0, "bearish": -1.0, "positive": 1.0,
                       "bullish": 1.0, "neutral": 0.0}
    for row in load_json_rows(path):
        try:
            raw_time = next(row[k] for k in ("published_at", "published", "timestamp",
                                             "date", "created_at") if row.get(k) is not None)
            headline = str(row.get("headline") or row.get("title") or row.get("text") or "").strip()
            if not headline:
                continue
            raw_sentiment = row.get("sentiment_score", row.get("sentiment", 0.0))
            sentiment = sentiment_value.get(str(raw_sentiment).lower(), raw_sentiment)
            sentiment = max(-1.0, min(1.0, float(sentiment)))
            tokens = tuple(sorted({str(v).upper() for v in row.get("tokens", []) if v}))
            result.append(NewsItem(timestamp(raw_time), headline, sentiment, tokens))
        except (StopIteration, TypeError, ValueError, OverflowError):
            continue
    return sorted(result, key=lambda item: item.timestamp)


def bucket(value: float, edges: Iterable[float]) -> int:
    return bisect.bisect_right(tuple(edges), value)


def safe_return(current: float, prior: float) -> float:
    return (current - prior) / prior if prior else 0.0


def return_label(value: float) -> str:
    return RETURN_LABELS[bucket(value, RETURN_EDGES)]


def target_label(value: float, scheme: str) -> str:
    if scheme == "return7":
        return return_label(value)
    if scheme == "direction3":
        if value < -0.003:
            return "downshift"
        if value > 0.003:
            return "updraft"
        return "sideways"
    raise ValueError(f"unknown target scheme {scheme!r}")


def direction(label: str | None) -> int:
    if label in ("crater", "dive", "slip", "downshift"):
        return -1
    if label in ("lift", "rally", "soar", "updraft"):
        return 1
    return 0


def _quant(value: float, edges: Sequence[float]) -> str:
    return f"b{bucket(value, edges)}"


def _returns(bars: Sequence[dict[str, float]], index: int, windows: Sequence[int]) -> dict[int, float]:
    close = bars[index]["close"]
    return {window: safe_return(close, bars[max(0, index - window)]["close"])
            for window in windows}


def _news_window(news: Sequence[NewsItem], now: float, lookback_seconds: float,
                 asset: str) -> list[NewsItem]:
    stamps = [item.timestamp for item in news]
    lo = bisect.bisect_left(stamps, now - lookback_seconds)
    hi = bisect.bisect_right(stamps, now)
    candidates = news[lo:hi]
    asset = asset.upper()
    relevant = [item for item in candidates if not item.tokens or asset in item.tokens
                or "CRYPTO" in item.tokens or "WETH" in item.tokens or "BTC" in item.tokens]
    return relevant


def feature_streams(
    bars: Sequence[dict[str, float]],
    index: int,
    *,
    symbol: str,
    chain: str,
    horizon: int,
    news: Sequence[NewsItem] = (),
    news_lookback_hours: float = 48.0,
    reference_bars: Sequence[dict[str, float]] | None = None,
    active_pools: set[int] | None = None,
) -> list[tuple[int, str]]:
    """Return causal, atom-grounded views for one decision moment."""
    if index < 1 or index + horizon >= len(bars):
        raise IndexError("feature index lacks history or future target")
    bar = bars[index]
    close, opened, high, low = (bar[k] for k in ("close", "open", "high", "low"))
    denom = max(abs(opened), 1e-12)
    spread = max(0.0, high - low) / denom
    body = (close - opened) / denom
    position = (close - low) / (high - low) if high > low else 0.5
    upper_wick = (high - max(opened, close)) / denom
    lower_wick = (min(opened, close) - low) / denom
    geometry = " ".join((
        f"body={_quant(body, (-.03,-.015,-.008,-.004,-.002,-.0005,.0005,.002,.004,.008,.015,.03))}",
        f"range={_quant(spread, (.001,.002,.003,.005,.008,.012,.02,.03,.05))}",
        f"position={_quant(position, (.1,.25,.4,.6,.75,.9))}",
        f"upper={_quant(upper_wick, (.0005,.001,.002,.004,.008,.015))}",
        f"lower={_quant(lower_wick, (.0005,.001,.002,.004,.008,.015))}",
    ))

    returns = _returns(bars, index, (1, 2, 3, 6, 12, 24, 72, 168))
    temporal = " ".join(f"r{window}={_quant(value, RETURN_EDGES)}"
                        for window, value in returns.items())
    temporal += f" acceleration={_quant(returns[6]-returns[24], RETURN_EDGES)}"

    volumes = [row["volume"] for row in bars[max(0, index - 167):index + 1]]
    median24 = statistics.median(volumes[-24:]) if volumes[-24:] else 0.0
    median168 = statistics.median(volumes) if volumes else 0.0
    ratio24 = bar["volume"] / median24 if median24 else 1.0
    ratio168 = bar["volume"] / median168 if median168 else 1.0
    total_flow = bar["buy_volume"] + bar["sell_volume"]
    imbalance = ((bar["buy_volume"] - bar["sell_volume"]) / total_flow
                 if total_flow else 0.0)
    flow = (f"v24={_quant(ratio24, (.3,.5,.7,.9,1.1,1.5,2,3,5))} "
            f"v168={_quant(ratio168, (.3,.5,.7,.9,1.1,1.5,2,3,5))} "
            f"imbalance={_quant(imbalance, (-.6,-.3,-.15,-.05,.05,.15,.3,.6))}")

    one_bar = [safe_return(bars[i]["close"], bars[i-1]["close"])
               for i in range(max(1, index - 167), index + 1)]
    rv6 = statistics.pstdev(one_bar[-6:]) if len(one_bar[-6:]) > 2 else 0.0
    rv24 = statistics.pstdev(one_bar[-24:]) if len(one_bar[-24:]) > 2 else 0.0
    rv168 = statistics.pstdev(one_bar) if len(one_bar) > 2 else 0.0
    recent = bars[max(0, index - 167):index + 1]
    rolling_high = max(row["high"] for row in recent)
    rolling_low = min(row["low"] for row in recent)
    drawdown = safe_return(close, rolling_high)
    location = ((close - rolling_low) / (rolling_high - rolling_low)
                if rolling_high > rolling_low else 0.5)
    volatility = (f"rv6={_quant(rv6, (.001,.002,.003,.005,.008,.012,.02,.03))} "
                  f"rv24={_quant(rv24, (.001,.002,.003,.005,.008,.012,.02,.03))} "
                  f"rv168={_quant(rv168, (.001,.002,.003,.005,.008,.012,.02,.03))} "
                  f"drawdown={_quant(drawdown, (-.3,-.2,-.12,-.08,-.05,-.03,-.015,-.005))} "
                  f"location={_quant(location, (.1,.25,.4,.6,.75,.9))}")

    dt = datetime.fromtimestamp(bar["timestamp"], tz=timezone.utc)
    trend_votes = sum(1 if returns[w] > 0 else -1 if returns[w] < 0 else 0
                      for w in (6, 12, 24, 72, 168))
    regime = (f"trend_vote={trend_votes} volatility_ratio={_quant(rv24/max(rv168,1e-9),(.5,.75,1,1.25,1.75,2.5))} "
              f"hour={dt.hour} dow={dt.weekday()} session={dt.hour//6}")

    cross = "reference=unavailable"
    if reference_bars:
        ref_times = [row["timestamp"] for row in reference_bars]
        ref_index = bisect.bisect_right(ref_times, bar["timestamp"]) - 1
        if ref_index >= 24:
            ref_ret6 = safe_return(reference_bars[ref_index]["close"], reference_bars[ref_index-6]["close"])
            ref_ret24 = safe_return(reference_bars[ref_index]["close"], reference_bars[ref_index-24]["close"])
            cross = (f"reference_r6={_quant(ref_ret6, RETURN_EDGES)} "
                     f"reference_r24={_quant(ref_ret24, RETURN_EDGES)} "
                     f"relative_r6={_quant(returns[6]-ref_ret6, RETURN_EDGES)}")

    asset = symbol.replace("/", "-").split("-")[0].upper()
    eligible = _news_window(news, bar["timestamp"], news_lookback_hours * 3600, asset)
    token_counts = Counter(token for item in eligible for token in item.tokens
                           if token not in {"THE", "AND", "FOR", "WITH", "FROM", "NEWS", "GOOGLE"})
    entity_frame = " ".join(f"entity={token.lower()}" for token, _ in token_counts.most_common(8))
    if not entity_frame:
        entity_frame = "entity=none"
    sentiment = statistics.fmean(item.sentiment for item in eligible) if eligible else 0.0
    age = ((bar["timestamp"] - eligible[-1].timestamp) / 3600.0 if eligible else math.inf)
    news_state = (f"count={_quant(float(len(eligible)), (0,1,3,8,20,50,100))} "
                  f"sentiment={_quant(sentiment, (-.6,-.25,-.05,.05,.25,.6))} "
                  f"age={_quant(age, (1,3,6,12,24,48))}")

    base, *quote_parts = symbol.replace("/", "-").upper().split("-")
    quote = quote_parts[-1] if quote_parts else "USD"
    streams = [
        (POOL_OHLCV, geometry),
        (POOL_TEMPORAL, temporal),
        (POOL_FLOW, flow),
        (POOL_VOLATILITY, volatility),
        (POOL_REGIME, regime),
        (POOL_CROSS_MARKET, cross),
        (POOL_NEWS_ENTITIES, entity_frame),
        (POOL_NEWS_STATE, news_state),
        (POOL_HORIZON, f"horizon_bars={horizon}"),
        (POOL_INSTRUMENT, f"chain={chain.lower()} base={base.lower()} quote={quote.lower()}"),
    ]
    return streams if active_pools is None else [row for row in streams if row[0] in active_pools]


def actual_return(bars: Sequence[dict[str, float]], index: int, horizon: int) -> float:
    return safe_return(bars[index + horizon]["close"], bars[index]["close"])


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def unb64(value: str | None) -> str | None:
    if not value:
        return None
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode(errors="replace")


class BrainClient:
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read())

    def consolidate(self, streams: Sequence[tuple[int, str]], outcome: str) -> bool:
        reply = self.post("/brain/consolidate/multi", {
            "streams": [{"pool_id": pool, "frame": b64(frame)} for pool, frame in streams],
            "outcome_pool": POOL_OUTCOME,
            "outcome_frame": b64(outcome),
        })
        return reply.get("consolidated") is True

    def predict(self, streams: Sequence[tuple[int, str]]) -> tuple[str | None, float, float]:
        started = time.perf_counter()
        reply = self.post("/brain/predict/multi", {
            "streams": [{"pool_id": pool, "frame": b64(frame)} for pool, frame in streams],
            "target_pool": POOL_OUTCOME,
        })
        return (unb64(reply.get("answer")),
                max(0.0, min(1.0, float(reply.get("integrated_confidence") or 0.0))),
                time.perf_counter() - started)


def parse_prediction(value: str | None) -> str | None:
    text = (value or "").lower()
    return next((label for label in (*RETURN_LABELS, *DIRECTION_LABELS) if label in text), None)


def chronological_fold_indices(length: int, horizon: int, folds: int,
                               test_n: int, minimum_history: int = 512) -> list[tuple[range, range]]:
    # Reserve `horizon` realized bars after the final decision as well as the
    # purge gap before every fold.  A decision at the last corpus row has no
    # knowable target and must never enter an accuracy report.
    required = minimum_history + 2 * horizon + folds * test_n
    if length < required:
        raise ValueError(f"need at least {required} bars, found {length}")
    first_test = length - horizon - folds * test_n
    result = []
    for fold in range(folds):
        test_start = first_test + fold * test_n
        train_stop = test_start - horizon
        result.append((range(minimum_history, train_stop), range(test_start, test_start + test_n)))
    return result


def evaluate_rows(rows: Sequence[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    covered = [row for row in rows if row["predicted"] is not None]
    directional = [row for row in covered if direction(row["actual"]) and direction(row["predicted"])]
    correct = sum(direction(row["actual"]) == direction(row["predicted"]) for row in directional)
    classes = (-1, 1)
    recalls = []
    for cls in classes:
        actual_cls = [row for row in directional if direction(row["actual"]) == cls]
        recalls.append(sum(direction(row["predicted"]) == cls for row in actual_cls) / max(1, len(actual_cls)))
    tp = sum(direction(r["actual"]) == 1 and direction(r["predicted"]) == 1 for r in directional)
    tn = sum(direction(r["actual"]) == -1 and direction(r["predicted"]) == -1 for r in directional)
    fp = sum(direction(r["actual"]) == -1 and direction(r["predicted"]) == 1 for r in directional)
    fn = sum(direction(r["actual"]) == 1 and direction(r["predicted"]) == -1 for r in directional)
    denominator = math.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denominator
    cost = cost_bps / 10_000.0
    pnl = [direction(row["predicted"]) * row["return"] - cost for row in directional]
    gains = sum(value for value in pnl if value > 0)
    losses = -sum(value for value in pnl if value < 0)
    equity = peak = 0.0
    max_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    latencies = sorted(row["latency_seconds"] for row in rows)
    ece = 0.0
    for bin_index in range(10):
        low, high = bin_index / 10, (bin_index + 1) / 10
        selected = [row for row in covered if low <= row["confidence"] < high
                    or (bin_index == 9 and row["confidence"] == 1.0)]
        if selected:
            mean_confidence = statistics.fmean(row["confidence"] for row in selected)
            accuracy = statistics.fmean(float(row["predicted"] == row["actual"])
                                        for row in selected)
            ece += len(selected) / max(1, len(covered)) * abs(mean_confidence - accuracy)
    return {
        "observations": len(rows),
        "coverage": len(covered) / max(1, len(rows)),
        "directional_n": len(directional),
        "directional_accuracy": correct / max(1, len(directional)),
        "directional_balanced_accuracy": statistics.fmean(recalls),
        "mcc": mcc,
        "ece": ece,
        "net_expectancy": statistics.fmean(pnl) if pnl else 0.0,
        "profit_factor": gains / losses if losses else (None if gains else 0.0),
        "max_additive_drawdown": max_drawdown,
        "latency_p95": latencies[round((len(latencies) - 1) * .95)] if latencies else 0.0,
    }


def run_fold(
    bars: Sequence[dict[str, float]], client: BrainClient, train_indices: Sequence[int],
    test_indices: Sequence[int], *, symbol: str, chain: str, horizon: int,
    news: Sequence[NewsItem], reference_bars: Sequence[dict[str, float]] | None,
    train_limit: int, cost_bps: float, balance_mode: str = "bounded",
    active_pools: set[int] | None = None, target_scheme: str = "direction3",
    prediction_mode: str = "joint",
) -> dict[str, Any]:
    selected = list(train_indices)[-train_limit:] if train_limit else list(train_indices)
    labels = DIRECTION_LABELS if target_scheme == "direction3" else RETURN_LABELS
    label_counts = Counter(target_label(actual_return(bars, i, horizon), target_scheme)
                           for i in selected)
    by_label: dict[str, list[int]] = {label: [] for label in labels}
    for index in selected:
        by_label[target_label(actual_return(bars, index, horizon), target_scheme)].append(index)
    if balance_mode == "none":
        balanced = selected
    elif balance_mode == "equal":
        nonempty = [len(indices) for indices in by_label.values() if indices]
        cap = min(nonempty) if nonempty else 0
        balanced = sorted(index for indices in by_label.values() for index in indices[-cap:])
    elif balance_mode == "bounded":
        # Preserve observed priors but prevent one attractor from receiving
        # more than twice the median class evidence.
        nonempty = sorted(len(indices) for indices in by_label.values() if indices)
        cap = max(1, round(statistics.median(nonempty) * 2)) if nonempty else 0
        balanced = sorted(index for indices in by_label.values() for index in indices[-cap:])
    else:
        raise ValueError(f"unknown balance mode {balance_mode!r}")
    failures = 0
    started = time.perf_counter()
    for index in balanced:
        streams = feature_streams(bars, index, symbol=symbol, chain=chain, horizon=horizon,
                                  news=news, reference_bars=reference_bars,
                                  active_pools=active_pools)
        target = target_label(actual_return(bars, index, horizon), target_scheme)
        training_stream_groups = ([streams] if prediction_mode == "joint"
                                  else [[stream] for stream in streams])
        for stream_group in training_stream_groups:
            if not client.consolidate(stream_group, f"future {target}"):
                failures += 1
    rows = []
    for index in test_indices:
        streams = feature_streams(bars, index, symbol=symbol, chain=chain, horizon=horizon,
                                  news=news, reference_bars=reference_bars,
                                  active_pools=active_pools)
        if prediction_mode == "joint":
            answer, confidence, latency = client.predict(streams)
            predicted = parse_prediction(answer)
            pool_predictions = None
        elif prediction_mode == "independent_vote":
            vote_started = time.perf_counter()
            votes: Counter[str] = Counter()
            pool_predictions = []
            for pool, frame in streams:
                answer, score, _pool_latency = client.predict([(pool, frame)])
                candidate = parse_prediction(answer)
                pool_predictions.append({"pool": pool, "predicted": candidate, "score": score})
                if candidate is not None:
                    votes[candidate] += max(score, 0.01)
            predicted = votes.most_common(1)[0][0] if votes else None
            total_vote = sum(votes.values())
            confidence = votes[predicted] / total_vote if predicted and total_vote else 0.0
            latency = time.perf_counter() - vote_started
        else:
            raise ValueError(f"unknown prediction mode {prediction_mode!r}")
        realized = actual_return(bars, index, horizon)
        actual = target_label(realized, target_scheme)
        momentum = direction(return_label(safe_return(bars[index]["close"], bars[index-1]["close"])))
        rows.append({"index": index, "timestamp": bars[index]["timestamp"], "return": realized,
                     "actual": actual, "predicted": predicted,
                     "confidence": confidence, "latency_seconds": latency,
                     "momentum_direction": momentum,
                     "pool_predictions": pool_predictions})
    metrics = evaluate_rows(rows, cost_bps)
    directional_actual = [row for row in rows if direction(row["actual"])]
    metrics["momentum_accuracy"] = sum(direction(row["actual"]) == row["momentum_direction"]
                                       for row in directional_actual) / max(1, len(directional_actual))
    metrics["inverse_momentum_accuracy"] = sum(
        direction(row["actual"]) == -row["momentum_direction"] for row in directional_actual
    ) / max(1, len(directional_actual))
    metrics["best_baseline_accuracy"] = max(metrics["momentum_accuracy"],
                                             metrics["inverse_momentum_accuracy"])
    metrics["baseline_margin"] = metrics["directional_accuracy"] - metrics["best_baseline_accuracy"]
    magnitude_by_label = {}
    for label_name in labels:
        values = [actual_return(bars, i, horizon) for i in selected
                  if target_label(actual_return(bars, i, horizon), target_scheme) == label_name]
        magnitude_by_label[label_name] = statistics.median(values) if values else 0.0
    return {"training": {"selected": len(balanced), "failures": failures,
                          "seconds": time.perf_counter() - started,
                          "target_scheme": target_scheme,
                          "prediction_mode": prediction_mode,
                          "label_counts": dict(Counter(target_label(actual_return(bars, i, horizon), target_scheme)
                                                       for i in balanced)),
                          "conditional_return_medians": magnitude_by_label},
            "metrics": metrics, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--reference-corpus", type=Path)
    parser.add_argument("--brain", default="http://127.0.0.1:18110")
    parser.add_argument("--symbol", default="WETH-USDC")
    parser.add_argument("--chain", default="base")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-n", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--balance-mode", choices=("none", "bounded", "equal"), default="bounded")
    parser.add_argument("--active-pools", default="",
                        help="comma-separated input pool ids; default uses all pools")
    parser.add_argument("--target-scheme", choices=("direction3", "return7"), default="direction3")
    parser.add_argument("--prediction-mode", choices=("joint", "independent_vote"), default="joint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bars = load_bars(args.corpus)
    news = load_news(args.news)
    reference = load_bars(args.reference_corpus) if args.reference_corpus else None
    folds = chronological_fold_indices(len(bars), args.horizon, args.folds, args.test_n)
    client = BrainClient(args.brain)
    active_pools = ({int(value) for value in args.active_pools.split(",") if value.strip()}
                    if args.active_pools else None)
    reports = []
    for number, (train_indices, test_indices) in enumerate(folds, 1):
        report = run_fold(bars, client, train_indices, test_indices,
                          symbol=args.symbol, chain=args.chain, horizon=args.horizon,
                          news=news, reference_bars=reference,
                          train_limit=args.train_limit, cost_bps=args.cost_bps,
                          balance_mode=args.balance_mode, active_pools=active_pools,
                          target_scheme=args.target_scheme,
                          prediction_mode=args.prediction_mode)
        report["fold"] = number
        report["split"] = {"train_start": train_indices.start, "train_stop": train_indices.stop,
                           "purge_bars": args.horizon, "test_start": test_indices.start,
                           "test_stop": test_indices.stop}
        reports.append(report)
    output = {
        "contract": "docs/MARKET_BRAIN_GHOST_TRADING_ADMISSION.md",
        "inputs": {"corpus": str(args.corpus.resolve()),
                   "news": str(args.news.resolve()) if args.news else None,
                   "reference": str(args.reference_corpus.resolve()) if args.reference_corpus else None,
                   "symbol": args.symbol, "chain": args.chain, "horizon": args.horizon},
        "folds": reports,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps([{"fold": row["fold"], **row["metrics"]} for row in reports], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
