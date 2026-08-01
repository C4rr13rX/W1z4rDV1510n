#!/usr/bin/env python3
"""Full protected validation for one evolved market genome."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from market_evolution_service import (
    ROOT, atomic_json, evaluate_genome, genome_from_dict, load_dataset_cached,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runtime/benchmarks/market-corpus-manifest.json")
    parser.add_argument("--supplemental-root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"))
    parser.add_argument("--news", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\news\historical_deduplicated.json"))
    parser.add_argument("--dataset-cache", type=Path,
                        default=ROOT / "runtime/cache/market-evolution-dataset-v2.joblib")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=42)
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--final-days", type=int, default=21)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    args = parser.parse_args()

    genome = genome_from_dict(json.loads(args.candidate.read_text(encoding="utf-8"))).finalize()
    # A validation result must never reuse the micro-sweep fitness attached to
    # the source candidate.
    genome.fitness = None
    genome.result = None
    dataset = load_dataset_cached(
        args.manifest, args.supplemental_root, args.horizon, args.stride,
        "market-perpetual-v1", args.news, args.dataset_cache,
    )
    evaluated = evaluate_genome(
        genome, dataset, folds=args.folds, test_days=args.test_days,
        calibration_days=args.calibration_days, final_days=args.final_days,
        horizon=args.horizon, cost_bps=args.cost_bps,
    )
    atomic_json(args.out, asdict(evaluated))
    print(json.dumps({
        "genome_id": evaluated.genome_id, "fitness": evaluated.fitness,
        "status": (evaluated.result or {}).get("status"),
        "summary": (evaluated.result or {}).get("summary"),
    }, indent=2))
    return 0 if (evaluated.result or {}).get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
