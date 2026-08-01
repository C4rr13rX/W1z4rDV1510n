#!/usr/bin/env python3
"""Audit, deduplicate, and stratify locally downloaded OHLCV corpora.

The manifest is evidence, not training state.  It preserves every discovered
file while naming exactly one canonical source for each chain/pair export and
one representative for byte-equivalent price histories across chains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.market_brain_experiment import load_bars
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from market_brain_experiment import load_bars

FILE_RE = re.compile(r"^(?:\d+_)?(.+?)\.(?:json|jsonl)$", re.IGNORECASE)
STABLES = {"USDC", "USDT", "USDT0", "USD₮0", "DAI", "FRAX", "LUSD", "USDE", "EURC"}
MAJORS = {"BTC", "WBTC", "CBBTC", "ETH", "WETH", "SOL", "JITOSOL"}
DEFI = {
    "AAVE", "AERO", "ARB", "COMP", "CRV", "ENS", "LDO", "LINK", "MKR",
    "OP", "PENDLE", "SNX", "SUSHI", "UNI", "VIRTUAL", "WLD",
}
HIGH_VOLATILITY = {"DOGE", "PEPE", "SHIB", "WIF", "BONK", "FLOKI", "VVV"}


def normalize_token(value: str) -> str:
    return value.strip().upper().replace("USD₮0", "USDT0")


def symbol_from_path(path: Path) -> str | None:
    match = FILE_RE.match(path.name)
    if not match or path.name.lower().endswith(".tmp"):
        return None
    symbol = match.group(1).replace("_", "-").replace("/", "-").upper()
    parts = [normalize_token(part) for part in symbol.split("-") if part]
    return "-".join(parts) if len(parts) >= 2 else None


def asset_family(symbol: str) -> str:
    base, quote = symbol.split("-", 1)
    if base in STABLES and quote in STABLES:
        return "stable_cross"
    if base in MAJORS:
        return "major"
    if base in DEFI:
        return "defi"
    if base in HIGH_VOLATILITY:
        return "high_volatility"
    return "other"


def series_fingerprint(bars: Iterable[dict[str, float]]) -> str:
    """Hash the full timestamp/close path; volume-source changes do not clone it."""
    digest = hashlib.blake2b(digest_size=20)
    for bar in bars:
        digest.update(struct.pack("<qd", round(bar["timestamp"]), bar["close"]))
    return digest.hexdigest()


def inspect_file(path: Path, root: Path, min_rows: int, min_hourly_share: float) -> dict[str, Any]:
    symbol = symbol_from_path(path)
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(root).as_posix(),
        "chain": path.parent.name.lower(),
        "symbol": symbol,
        "size_bytes": path.stat().st_size,
        "eligible": False,
    }
    if symbol is None:
        record["reason"] = "unrecognized_symbol"
        return record
    try:
        bars = load_bars(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record["reason"] = f"unreadable:{type(error).__name__}"
        return record
    record["rows"] = len(bars)
    if not bars:
        record["reason"] = "no_valid_bars"
        return record
    deltas = [bars[i]["timestamp"] - bars[i - 1]["timestamp"] for i in range(1, len(bars))]
    positive = [delta for delta in deltas if delta > 0]
    hourly_share = (sum(abs(delta - 3600.0) <= 60.0 for delta in positive) / len(positive)
                    if positive else 0.0)
    one_bar_returns = [
        (bars[i]["close"] - bars[i - 1]["close"]) / bars[i - 1]["close"]
        for i in range(1, len(bars)) if bars[i - 1]["close"] > 0
    ]
    record.update({
        "start_timestamp": bars[0]["timestamp"],
        "end_timestamp": bars[-1]["timestamp"],
        "coverage_days": (bars[-1]["timestamp"] - bars[0]["timestamp"]) / 86400.0,
        "median_interval_seconds": statistics.median(positive) if positive else None,
        "hourly_cadence_share": hourly_share,
        "return_volatility": statistics.pstdev(one_bar_returns) if len(one_bar_returns) > 2 else 0.0,
        "series_fingerprint": series_fingerprint(bars),
        "family": asset_family(symbol),
        "base_asset": symbol.split("-", 1)[0],
        "quote_asset": symbol.split("-", 1)[1],
    })
    if len(bars) < min_rows:
        record["reason"] = "too_few_rows"
    elif hourly_share < min_hourly_share:
        record["reason"] = "irregular_cadence"
    else:
        record["eligible"] = True
        record["reason"] = "eligible"
    return record


def build_manifest(root: Path, *, min_rows: int = 2000,
                   min_hourly_share: float = 0.90) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    records = [inspect_file(path, root, min_rows, min_hourly_share) for path in paths]

    # Repeated numbered downloads of one chain/pair are versions, not samples.
    by_market: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["eligible"]:
            by_market[(record["chain"], record["symbol"])].append(record)
    chain_canonical: list[dict[str, Any]] = []
    for candidates in by_market.values():
        candidates.sort(key=lambda item: (
            item["rows"], item["end_timestamp"], item["coverage_days"], item["size_bytes"],
            item["relative_path"],
        ), reverse=True)
        winner = candidates[0]
        winner["canonical_for_chain_pair"] = True
        chain_canonical.append(winner)
        for duplicate in candidates[1:]:
            duplicate["canonical_for_chain_pair"] = False
            duplicate["duplicate_of"] = winner["relative_path"]

    # The same exported path on several chains is still one observed market.
    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in chain_canonical:
        by_fingerprint[record["series_fingerprint"]].append(record)
    selected: list[dict[str, Any]] = []
    for candidates in by_fingerprint.values():
        candidates.sort(key=lambda item: (
            item["rows"], item["end_timestamp"], item["size_bytes"], item["relative_path"]
        ), reverse=True)
        winner = candidates[0]
        winner["selected"] = True
        selected.append(winner)
        for duplicate in candidates[1:]:
            duplicate["selected"] = False
            duplicate["duplicate_of"] = winner["relative_path"]

    selected.sort(key=lambda item: (item["family"], item["symbol"], item["chain"]))
    family_counts = Counter(item["family"] for item in selected)
    chain_counts = Counter(item["chain"] for item in selected)
    base_counts = Counter(item["base_asset"] for item in selected)
    return {
        "schema_version": 1,
        "source_root": str(root.resolve()),
        "policy": {
            "min_rows": min_rows,
            "min_hourly_cadence_share": min_hourly_share,
            "chain_pair_rule": "largest/latest valid export",
            "cross_chain_rule": "one representative per exact timestamp/close fingerprint",
        },
        "summary": {
            "files_discovered": len(records),
            "eligible_files": sum(record["eligible"] for record in records),
            "canonical_chain_pairs": len(chain_canonical),
            "selected_independent_series": len(selected),
            "selected_rows": sum(item["rows"] for item in selected),
            "families": dict(sorted(family_counts.items())),
            "chains": dict(sorted(chain_counts.items())),
            "top_base_assets": dict(base_counts.most_common(25)),
        },
        "selected": selected,
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\historical_ohlcv"))
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/market-corpus-manifest.json"))
    parser.add_argument("--min-rows", type=int, default=2000)
    parser.add_argument("--min-hourly-share", type=float, default=0.90)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"OHLCV root does not exist: {args.root}")
    manifest = build_manifest(args.root, min_rows=args.min_rows,
                              min_hourly_share=args.min_hourly_share)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
