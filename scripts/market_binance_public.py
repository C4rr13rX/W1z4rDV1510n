#!/usr/bin/env python3
"""Fetch and merge causal Binance public spot/perpetual market features.

The downloader uses Binance's public monthly archives, validates their
published SHA-256 checksums, and is restart-safe.  ZIPs and merged JSON files
are generated corpus artifacts; this script is the reproducible source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.binance.vision/data"
DEFAULT_ROOT = Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public")
DEFAULT_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "TRXUSDT", "AAVEUSDT", "UNIUSDT", "NEARUSDT", "ARBUSDT", "OPUSDT",
    "PENDLEUSDT", "SUIUSDT", "PEPEUSDT", "SHIBUSDT", "CAKEUSDT",
    "COMPUSDT", "CRVUSDT", "ETHFIUSDT", "LDOUSDT", "MKRUSDT", "PAXGUSDT",
    "SANDUSDT", "SNXUSDT", "STGUSDT", "SUSHIUSDT", "WLDUSDT", "ZROUSDT",
)
KIND_PATHS = {
    "spot": "spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip",
    "futures": "futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip",
    "premium": "futures/um/monthly/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{month}.zip",
    "funding": "futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip",
}


@dataclass(frozen=True)
class Artifact:
    kind: str
    symbol: str
    month: str

    @property
    def relative_url(self) -> str:
        return KIND_PATHS[self.kind].format(symbol=self.symbol, month=self.month)


def normalize_timestamp(value: str | int | float) -> int:
    """Normalize Binance seconds/milliseconds/microseconds to UTC seconds."""
    timestamp = int(float(value))
    if timestamp >= 10**15:
        timestamp //= 1_000_000
    elif timestamp >= 10**12:
        timestamp //= 1_000
    return timestamp


def month_range(start: str, end: str) -> list[str]:
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    if (year, month) > (end_year, end_month):
        raise ValueError("start month must not be later than end month")
    result = []
    while (year, month) <= (end_year, end_month):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def last_complete_month(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    year, month = now.year, now.month - 1
    if month == 0:
        year, month = year - 1, 12
    return f"{year:04d}-{month:02d}"


def _request(url: str, attempts: int = 4) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "W1z4rdVision-market-corpus/1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            error = exc
        except (OSError, urllib.error.URLError) as exc:
            error = exc
        time.sleep(min(8.0, 0.5 * 2**attempt))
    assert error is not None
    raise error


def download_artifact(root: Path, artifact: Artifact) -> tuple[str, str, str]:
    url = f"{BASE_URL}/{artifact.relative_url}"
    target = root / "archives" / artifact.kind / artifact.symbol / Path(artifact.relative_url).name
    checksum_target = target.with_suffix(target.suffix + ".sha256")
    try:
        checksum_text = _request(url + ".CHECKSUM").decode("ascii").strip()
        expected = checksum_text.split()[0].lower()
        if len(expected) != 64:
            raise ValueError(f"invalid checksum response for {url}")
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return artifact.kind, artifact.symbol, "cached"
        payload = _request(url)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch for {url}: {actual} != {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        checksum_target.write_text(checksum_text + "\n", encoding="ascii")
        return artifact.kind, artifact.symbol, "downloaded"
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return artifact.kind, artifact.symbol, "unavailable"
        raise


def _csv_rows(path: Path) -> Iterable[list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one CSV in {path}, found {names}")
        with archive.open(names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            yield from csv.reader(text)


def parse_kline_archive(path: Path, prefix: str) -> dict[int, dict[str, float | int]]:
    result: dict[int, dict[str, float | int]] = {}
    for row in _csv_rows(path):
        if not row or not row[0] or not row[0][0].isdigit():
            continue
        timestamp = normalize_timestamp(row[0])
        result[timestamp] = {
            f"{prefix}_open": float(row[1]), f"{prefix}_high": float(row[2]),
            f"{prefix}_low": float(row[3]), f"{prefix}_close": float(row[4]),
            f"{prefix}_base_volume": float(row[5]),
            f"{prefix}_quote_volume": float(row[7]),
            f"{prefix}_trade_count": int(float(row[8])),
            f"{prefix}_taker_buy_base": float(row[9]),
            f"{prefix}_taker_buy_quote": float(row[10]),
        }
    return result


def parse_funding_archive(path: Path) -> list[tuple[int, float]]:
    result = []
    for row in _csv_rows(path):
        if not row or not row[0] or not row[0][0].isdigit():
            continue
        result.append((normalize_timestamp(row[0]), float(row[2])))
    return sorted(result)


def build_symbol(root: Path, symbol: str) -> dict[str, object]:
    by_kind: dict[str, list[Path]] = {}
    for kind in KIND_PATHS:
        folder = root / "archives" / kind / symbol
        by_kind[kind] = sorted(folder.glob("*.zip")) if folder.exists() else []
    frames: dict[str, dict[int, dict[str, float | int]]] = {}
    for kind in ("spot", "futures", "premium"):
        combined: dict[int, dict[str, float | int]] = {}
        for path in by_kind[kind]:
            combined.update(parse_kline_archive(path, kind))
        frames[kind] = combined
    funding = []
    for path in by_kind["funding"]:
        funding.extend(parse_funding_archive(path))
    funding.sort()

    timestamps = sorted(set(frames["spot"]) & set(frames["futures"]))
    rows = []
    funding_index = 0
    known_funding: float | None = None
    for timestamp in timestamps:
        while funding_index < len(funding) and funding[funding_index][0] <= timestamp:
            known_funding = funding[funding_index][1]
            funding_index += 1
        row: dict[str, float | int | None] = {"timestamp": timestamp}
        row.update(frames["spot"][timestamp])
        row.update(frames["futures"][timestamp])
        if timestamp in frames["premium"]:
            row.update(frames["premium"][timestamp])
        spot_close = float(row["spot_close"])
        futures_close = float(row["futures_close"])
        row["futures_spot_basis"] = ((futures_close - spot_close) / spot_close
                                     if spot_close else 0.0)
        row["funding_rate"] = known_funding
        rows.append(row)
    output = root / "features" / f"{symbol}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "https://data.binance.vision/",
        "symbol": symbol,
        "causality": ("funding is forward-filled only after calc_time; kline open_time identifies "
                      "the completed hourly bar used at its close"),
        "rows": rows,
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return {
        "symbol": symbol, "rows": len(rows),
        "start": rows[0]["timestamp"] if rows else None,
        "end": rows[-1]["timestamp"] if rows else None,
        "funding_rows": len(funding),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default=last_complete_month())
    parser.add_argument("--kinds", default=",".join(KIND_PATHS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    symbols = tuple(dict.fromkeys(value.strip().upper() for value in args.symbols.split(",") if value.strip()))
    kinds = tuple(dict.fromkeys(value.strip() for value in args.kinds.split(",") if value.strip()))
    unknown = set(kinds) - set(KIND_PATHS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    artifacts = [Artifact(kind, symbol, month) for symbol in symbols
                 for month in month_range(args.start, args.end) for kind in kinds]
    counts: dict[str, int] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(download_artifact, args.root, artifact): artifact
                   for artifact in artifacts}
        for future in as_completed(futures):
            kind, symbol, status = future.result()
            counts[status] = counts.get(status, 0) + 1
            completed += 1
            if completed % 50 == 0 or completed == len(artifacts):
                print(f"download {completed}/{len(artifacts)} {counts}", flush=True)
    built = [] if args.download_only else [build_symbol(args.root, symbol) for symbol in symbols]
    manifest = {
        "source": BASE_URL, "start": args.start, "end": args.end,
        "symbols": symbols, "kinds": kinds, "downloads": counts, "series": built,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        (",".join(symbols) + "|" + args.start + "|" + args.end).encode()
    ).hexdigest()[:12]
    manifests = args.root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{args.start}-{args.end}-{identity}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
