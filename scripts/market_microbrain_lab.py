#!/usr/bin/env python3
"""Launch one disposable market micro-brain and evaluate one walk-forward fold."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from market_brain_experiment import (
    BrainClient,
    chronological_fold_indices,
    load_bars,
    load_news,
    run_fold,
)

ROOT = Path(__file__).resolve().parents[1]


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
            time.sleep(0.25)
    raise TimeoutError(f"brain server did not become healthy at {endpoint}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--reference-corpus", type=Path)
    parser.add_argument("--identity", type=Path,
                        default=ROOT / "brains" / "market_predictor_v2.identity.toml")
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "target" / "debug" / "w1z4rd_brain_server.exe")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--symbol", default="WETH-USDC")
    parser.add_argument("--chain", default="base")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--test-n", type=int, default=200)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--balance-mode", choices=("none", "bounded", "equal"), default="bounded")
    parser.add_argument("--active-pools", default="")
    parser.add_argument("--target-scheme", choices=("direction3", "return7"), default="direction3")
    parser.add_argument("--prediction-mode", choices=("joint", "independent_vote"), default="joint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.runtime.exists() and any(args.runtime.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty runtime {args.runtime}")
    args.runtime.mkdir(parents=True, exist_ok=True)
    stdout_path = args.runtime / "stdout.log"
    stderr_path = args.runtime / "stderr.log"
    environment = os.environ.copy()
    environment.update({
        "W1Z4RD_NODE_BRAIN_DIR": str(args.runtime.resolve()),
        "W1Z4RD_BRAIN_IDENTITY": str(args.identity.resolve()),
        "W1Z4RD_BRAIN_PORT": str(args.port),
        "W1Z4RD_BRAIN_BIND": "127.0.0.1",
        "W1Z4RD_TIER_MIN_SYS_AVAIL_MB": "6144",
    })
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [str(args.binary.resolve())],
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        try:
            endpoint = f"http://127.0.0.1:{args.port}"
            wait_for_health(endpoint, process)
            bars = load_bars(args.corpus)
            news = load_news(args.news)
            reference = load_bars(args.reference_corpus) if args.reference_corpus else None
            folds = chronological_fold_indices(len(bars), args.horizon, args.folds, args.test_n)
            if not 0 <= args.fold_index < len(folds):
                raise ValueError(f"fold-index must be in [0,{len(folds)-1}]")
            train_indices, test_indices = folds[args.fold_index]
            report = run_fold(
                bars,
                BrainClient(endpoint),
                train_indices,
                test_indices,
                symbol=args.symbol,
                chain=args.chain,
                horizon=args.horizon,
                news=news,
                reference_bars=reference,
                train_limit=args.train_limit,
                cost_bps=args.cost_bps,
                balance_mode=args.balance_mode,
                active_pools=({int(value) for value in args.active_pools.split(",") if value.strip()}
                              if args.active_pools else None),
                target_scheme=args.target_scheme,
                prediction_mode=args.prediction_mode,
            )
            report.update({
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "identity": str(args.identity.resolve()),
                "fold": args.fold_index,
                "split": {"train_start": train_indices.start, "train_stop": train_indices.stop,
                          "purge_bars": args.horizon, "test_start": test_indices.start,
                          "test_stop": test_indices.stop},
                "inputs": {"corpus": str(args.corpus.resolve()),
                           "news": str(args.news.resolve()) if args.news else None,
                           "reference": str(args.reference_corpus.resolve()) if args.reference_corpus else None,
                           "symbol": args.symbol, "chain": args.chain},
                "configuration": {"balance_mode": args.balance_mode,
                                  "active_pools": args.active_pools or "all",
                                  "target_scheme": args.target_scheme,
                                  "prediction_mode": args.prediction_mode},
            })
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(report["metrics"], indent=2, allow_nan=False))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
