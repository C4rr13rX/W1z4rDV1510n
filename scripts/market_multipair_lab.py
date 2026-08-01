#!/usr/bin/env python3
"""Train one disposable brain on many markets and test future/unseen assets.

The split is deliberately stronger than a shuffled row split: every training
row precedes one global cutoff, and complete base assets are withheld from all
training.  The resulting report separates future performance on familiar
assets from transfer to never-trained assets.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.market_brain_experiment import (
        BrainClient, actual_return, direction, evaluate_rows, feature_streams,
        load_bars, load_news, parse_prediction, target_label,
    )
except ModuleNotFoundError:
    from market_brain_experiment import (
        BrainClient, actual_return, direction, evaluate_rows, feature_streams,
        load_bars, load_news, parse_prediction, target_label,
    )

ROOT = Path(__file__).resolve().parents[1]


def stable_order(value: str, seed: str) -> str:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).hexdigest()


def parse_cutoff(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def derive_cutoff(records: Sequence[dict[str, Any]], test_n: int, horizon: int) -> float:
    ends = sorted(float(item["end_timestamp"]) for item in records)
    if not ends:
        raise ValueError("manifest has no selected series")
    # The lower-decile end prevents a few stale markets from controlling the
    # split while leaving about 90% of series eligible for the common future.
    lower_decile = ends[round((len(ends) - 1) * .10)]
    return lower_decile - (test_n + horizon + 1) * 3600.0


def partition_assets(records: Sequence[dict[str, Any]], holdout_fraction: float,
                     seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families: dict[str, set[str]] = defaultdict(set)
    for record in records:
        families[record["family"]].add(record["base_asset"])
    holdout_assets: set[str] = set()
    for family, assets in families.items():
        ordered = sorted(assets, key=lambda asset: stable_order(asset, seed + family))
        count = min(len(ordered) - 1, max(1, round(len(ordered) * holdout_fraction))) \
            if len(ordered) > 1 else 0
        holdout_assets.update(asset for asset in ordered[:count] if asset not in {"WETH", "WBTC"})
    training = [record for record in records if record["base_asset"] not in holdout_assets]
    holdout = [record for record in records if record["base_asset"] in holdout_assets]
    return training, holdout


def evenly_spaced(indices: Sequence[int], count: int) -> list[int]:
    if count <= 0 or len(indices) <= count:
        return list(indices)
    if count == 1:
        return [indices[-1]]
    return [indices[round(position * (len(indices) - 1) / (count - 1))]
            for position in range(count)]


def eligible_indices(bars: Sequence[dict[str, float]], cutoff: float, horizon: int,
                     minimum_history: int = 512) -> tuple[list[int], list[int]]:
    timestamps = [bar["timestamp"] for bar in bars]
    train_stop = bisect.bisect_left(timestamps, cutoff - horizon * 3600.0)
    test_start = bisect.bisect_left(timestamps, cutoff)
    train = list(range(minimum_history, max(minimum_history, train_stop)))
    test = list(range(test_start, max(test_start, len(bars) - horizon)))
    return train, test


def wait_for_health(endpoint: str, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"brain server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(.25)
    raise TimeoutError(f"brain server did not become healthy at {endpoint}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def progress(message: str) -> None:
    """Progress must never make a detached training supervisor fail."""
    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


def market_rows(record: dict[str, Any], bars: Sequence[dict[str, float]], indices: Sequence[int],
                client: BrainClient, *, horizon: int, news, reference_bars,
                active_pools: set[int], cost_bps: float) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        streams = feature_streams(
            bars, index, symbol=record["symbol"], chain=record["chain"], horizon=horizon,
            news=news, reference_bars=reference_bars, active_pools=active_pools,
        )
        answer, confidence, latency = client.predict(streams)
        predicted = parse_prediction(answer)
        realized = actual_return(bars, index, horizon)
        actual = target_label(realized, "direction3")
        momentum = direction(target_label(
            (bars[index]["close"] - bars[index - 1]["close"]) / bars[index - 1]["close"],
            "direction3",
        ))
        rows.append({
            "chain": record["chain"], "symbol": record["symbol"],
            "index": index, "timestamp": bars[index]["timestamp"], "return": realized,
            "actual": actual, "predicted": predicted, "confidence": confidence,
            "latency_seconds": latency, "momentum_direction": momentum,
        })
    return rows


def add_baselines(metrics: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    directional = [row for row in rows if direction(row["actual"])]
    momentum = sum(direction(row["actual"]) == row["momentum_direction"]
                   for row in directional) / max(1, len(directional))
    inverse = sum(direction(row["actual"]) == -row["momentum_direction"]
                  for row in directional) / max(1, len(directional))
    metrics.update({"momentum_accuracy": momentum, "inverse_momentum_accuracy": inverse,
                    "best_baseline_accuracy": max(momentum, inverse),
                    "baseline_margin": metrics["directional_accuracy"] - max(momentum, inverse)})


def cluster_asset_time_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse correlated chain/quote copies to one base-asset/time decision."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["base_asset"], row["timestamp"])].append(row)
    collapsed = []
    for (_asset, _timestamp), members in sorted(groups.items()):
        actual_votes = Counter(direction(row["actual"]) for row in members)
        predicted_votes = Counter(direction(row["predicted"]) for row in members
                                  if row["predicted"] is not None)
        actual_direction = actual_votes.most_common(1)[0][0]
        predicted_direction = predicted_votes.most_common(1)[0][0] if predicted_votes else None
        label = {-1: "downshift", 0: "sideways", 1: "updraft"}
        representative = dict(members[0])
        representative.update({
            "actual": label[actual_direction],
            "predicted": label[predicted_direction] if predicted_direction is not None else None,
            "return": statistics.fmean(row["return"] for row in members),
            "confidence": statistics.fmean(row["confidence"] for row in members),
            "latency_seconds": max(row["latency_seconds"] for row in members),
            "momentum_direction": Counter(row["momentum_direction"] for row in members).most_common(1)[0][0],
            "cluster_members": len(members),
        })
        collapsed.append(representative)
    return collapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("runtime/benchmarks/market-corpus-manifest.json"))
    parser.add_argument("--news", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\news\free_news.json"))
    parser.add_argument("--identity", type=Path,
                        default=ROOT / "brains" / "market_predictor_v2.identity.toml")
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "target" / "debug" / "w1z4rd_brain_server.exe")
    parser.add_argument("--migration-binary", type=Path,
                        default=ROOT / "target" / "debug" / "w1z4rd_brain_migrate.exe")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cutoff", help="ISO timestamp; default derives a common lower-decile cutoff")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=40)
    parser.add_argument("--train-per-pair", type=int, default=80)
    parser.add_argument("--max-train-pairs", type=int, default=96)
    parser.add_argument("--max-holdout-pairs", type=int, default=32)
    parser.add_argument("--known-eval-pairs", type=int, default=24)
    parser.add_argument("--holdout-fraction", type=float, default=.25)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--seed", default="market-v2-asset-holdout")
    parser.add_argument("--active-pools", default="2,4,5,6,9",
                        help="instrument pool 10 is excluded by default to permit unseen-asset transfer")
    parser.add_argument("--settle-before-eval", action=argparse.BooleanOptionalAction, default=True,
                        help="serialize the trained overlay neuron-by-neuron before read-only evaluation")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = list(manifest["selected"])
    cutoff = parse_cutoff(args.cutoff) if args.cutoff else derive_cutoff(
        candidates, args.test_n, args.horizon)
    minimum_end = cutoff + (args.test_n + args.horizon) * 3600.0
    candidates = [record for record in candidates if record["end_timestamp"] >= minimum_end]
    training_records, holdout_records = partition_assets(
        candidates, args.holdout_fraction, args.seed)
    training_records = sorted(training_records,
                              key=lambda row: stable_order(row["relative_path"], args.seed))[:args.max_train_pairs]
    holdout_records = sorted(holdout_records,
                             key=lambda row: stable_order(row["relative_path"], args.seed))[:args.max_holdout_pairs]
    if not training_records or not holdout_records:
        raise ValueError("the manifest/cutoff did not produce both training and held-out markets")
    active_pools = {int(value) for value in args.active_pools.split(",") if value.strip()}

    reference_record = next((row for row in candidates if row["symbol"] == "WETH-USDC"),
                            next(row for row in candidates if row["base_asset"] == "WETH"))
    reference_bars = load_bars(Path(reference_record["path"]))
    news = load_news(args.news if args.news.exists() else None)

    if args.runtime.exists() and any(args.runtime.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty runtime {args.runtime}")
    args.runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "W1Z4RD_NODE_BRAIN_DIR": str(args.runtime.resolve()),
        "W1Z4RD_BRAIN_IDENTITY": str(args.identity.resolve()),
        "W1Z4RD_BRAIN_PORT": str(args.port), "W1Z4RD_BRAIN_BIND": "127.0.0.1",
        "W1Z4RD_TIER_MIN_SYS_AVAIL_MB": "4096",
    })
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = datetime.now(timezone.utc)
    with (args.runtime / "stdout.log").open("wb") as stdout, \
            (args.runtime / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen([str(args.binary.resolve())], cwd=ROOT, env=env,
                                   stdout=stdout, stderr=stderr, creationflags=creationflags)
        try:
            endpoint = f"http://127.0.0.1:{args.port}"
            wait_for_health(endpoint, process)
            client = BrainClient(endpoint, timeout=60.0)
            training_candidates = []
            loaded_training: dict[str, list[dict[str, float]]] = {}
            for record in training_records:
                bars = load_bars(Path(record["path"]))
                loaded_training[record["relative_path"]] = bars
                train_indices, _ = eligible_indices(bars, cutoff, args.horizon)
                for index in evenly_spaced(train_indices, args.train_per_pair):
                    training_candidates.append((record, bars, index))
            # Preserve corpus prevalence but bound any one direction attractor.
            by_label: dict[str, list[tuple]] = defaultdict(list)
            for item in training_candidates:
                record, bars, index = item
                by_label[target_label(actual_return(bars, index, args.horizon), "direction3")].append(item)
            counts = sorted(len(items) for items in by_label.values() if items)
            cap = round(statistics.median(counts) * 2) if counts else 0
            training_candidates = sorted(
                (item for items in by_label.values() for item in items[-cap:]),
                key=lambda item: (item[2] and item[1][item[2]]["timestamp"], item[0]["relative_path"]),
            )
            failures = 0
            label_counts: Counter[str] = Counter()
            for record, bars, index in training_candidates:
                streams = feature_streams(
                    bars, index, symbol=record["symbol"], chain=record["chain"],
                    horizon=args.horizon, news=news, reference_bars=reference_bars,
                    active_pools=active_pools,
                )
                label = target_label(actual_return(bars, index, args.horizon), "direction3")
                label_counts[label] += 1
                if not client.consolidate(streams, f"future {label}"):
                    failures += 1

            settlement = None
            if args.settle_before_eval:
                checkpoint = client.post("/brain/checkpoint", {})
                if checkpoint.get("ok") is not True:
                    raise RuntimeError(f"pre-settlement checkpoint failed: {checkpoint}")
                # Fresh brains begin in the compatibility format.  Conversion
                # requires a stopped server, after which the same identity is
                # restored from the neuron-addressable package.
                stop_process(process)
                migration = subprocess.run(
                    [str(args.migration_binary.resolve()), str(args.runtime.resolve())],
                    cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=creationflags, timeout=600,
                )
                if migration.returncode != 0:
                    raise RuntimeError(f"brain migration failed with exit {migration.returncode}")
                process = subprocess.Popen([str(args.binary.resolve())], cwd=ROOT, env=env,
                                           stdout=stdout, stderr=stderr,
                                           creationflags=creationflags)
                wait_for_health(endpoint, process)
                client = BrainClient(endpoint, timeout=60.0)
                settlement = client.post("/brain/sleep", {
                    "min_use_count": 0,
                    "stale_ticks": 9_223_372_036_854_775_807,
                })
                if settlement.get("error"):
                    raise RuntimeError(f"brain settlement failed: {settlement['error']}")
                settlement["checkpoint"] = checkpoint
                settlement["migration"] = "legacy_to_neuron_addressable"

            evaluation: dict[str, Any] = {}
            for name, records in (("known_asset_future", training_records[:args.known_eval_pairs]),
                                  ("unseen_asset_future", holdout_records)):
                rows = []
                per_market = []
                for record in records:
                    bars = loaded_training.get(record["relative_path"])
                    if bars is None:
                        bars = load_bars(Path(record["path"]))
                    _, test_indices = eligible_indices(bars, cutoff, args.horizon)
                    # Cover the full unseen interval instead of scoring one
                    # highly autocorrelated block immediately after cutoff.
                    selected_test = evenly_spaced(test_indices, args.test_n)
                    market = market_rows(record, bars, selected_test, client,
                                         horizon=args.horizon, news=news,
                                         reference_bars=reference_bars,
                                         active_pools=active_pools, cost_bps=args.cost_bps)
                    rows.extend(market)
                    market_metrics = evaluate_rows(market, args.cost_bps)
                    add_baselines(market_metrics, market)
                    per_market.append({"chain": record["chain"], "symbol": record["symbol"],
                                       "family": record["family"], "metrics": market_metrics})
                    progress(f"evaluated {name} {len(per_market)}/{len(records)} "
                             f"{record['chain']}:{record['symbol']}")
                metrics = evaluate_rows(rows, args.cost_bps)
                add_baselines(metrics, rows)
                clustered_rows = cluster_asset_time_rows(rows)
                clustered_metrics = evaluate_rows(clustered_rows, args.cost_bps)
                add_baselines(clustered_metrics, clustered_rows)
                evaluation[name] = {"metrics": metrics,
                                    "asset_time_clustered_metrics": clustered_metrics,
                                    "markets": per_market, "rows": rows}

            report = {
                "contract": "docs/MARKET_BRAIN_GHOST_TRADING_ADMISSION.md",
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "cutoff": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                "split": {
                    "training_pairs": len(training_records),
                    "training_assets": sorted({row["base_asset"] for row in training_records}),
                    "holdout_pairs": len(holdout_records),
                    "holdout_assets": sorted({row["base_asset"] for row in holdout_records}),
                    "all_training_rows_precede_cutoff": True,
                },
                "configuration": {
                    "horizon": args.horizon, "test_n": args.test_n,
                    "train_per_pair": args.train_per_pair,
                    "active_pools": sorted(active_pools), "cost_bps": args.cost_bps,
                },
                "training": {"episodes": len(training_candidates), "failures": failures,
                             "label_counts": dict(label_counts), "settlement": settlement},
                "evaluation": evaluation,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({name: section["metrics"] for name, section in evaluation.items()},
                             indent=2, allow_nan=False))
        finally:
            stop_process(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
