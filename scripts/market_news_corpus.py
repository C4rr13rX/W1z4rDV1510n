#!/usr/bin/env python3
"""Deduplicate timestamped crypto-news shards into a causal training corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = Path(r"D:\Projects\CoolCryptoUtilities\data\news\free_news")
DEFAULT_OUTPUT = Path(r"D:\Projects\CoolCryptoUtilities\data\news\historical_deduplicated.json")
CRYPTO_TERMS = re.compile(
    r"\b(bitcoin|crypto|blockchain|ethereum|ether|defi|stablecoin|token|altcoin|"
    r"binance|coinbase|solana|web3|wallet|on-chain|cryptocurrency|btc|eth)\b", re.I)
CRYPTO_SOURCES = {
    "coindesk", "cointelegraph", "decrypt", "the block", "bitcoin magazine",
    "cryptoslate", "beincrypto", "bitcoin.com", "blockworks",
}


def canonical_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query = [(key, val) for key, val in query
                 if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source"}]
        path = parsed.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path,
                                       urllib.parse.urlencode(query), ""))
    except ValueError:
        return value.strip()


def rows_from_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        yield from (row for row in payload if isinstance(row, dict))
    elif isinstance(payload, dict):
        for key in ("articles", "results", "data", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                yield from (row for row in rows if isinstance(row, dict))
                return


def normalize_article(row: dict[str, Any]) -> dict[str, Any] | None:
    headline = str(row.get("headline") or row.get("title") or "").strip()
    source = str(row.get("source") or "").strip()
    url = canonical_url(str(row.get("url") or row.get("link") or ""))
    try:
        timestamp = int(float(row.get("timestamp") or row.get("published_at") or row.get("published")))
        if timestamp >= 10**12:
            timestamp //= 1000
    except (TypeError, ValueError):
        return None
    if not headline or timestamp < 1_262_304_000:
        return None
    tokens = sorted({str(token).upper() for token in row.get("tokens", []) if token})
    searchable = " ".join((headline, str(row.get("article") or ""), " ".join(tokens)))
    if source.lower() not in CRYPTO_SOURCES and not CRYPTO_TERMS.search(searchable):
        return None
    sentiment = row.get("sentiment", row.get("sentiment_score", "neutral"))
    return {
        "timestamp": timestamp, "headline": headline,
        "article": str(row.get("article") or headline).strip(),
        "sentiment": sentiment, "tokens": tokens, "source": source, "url": url,
    }


def article_key(row: dict[str, Any]) -> str:
    # Timestamp remains part of the identity because a few publishers reuse a
    # rolling URL for distinct daily market reports.
    identity = row["url"] or re.sub(r"\W+", " ", row["headline"].lower()).strip()
    return hashlib.sha256(f"{row['timestamp']}|{identity}".encode()).hexdigest()


def build_corpus(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    stats = Counter()
    for position, path in enumerate(paths, 1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            stats["invalid_files"] += 1
            continue
        stats["files"] += 1
        for raw in rows_from_payload(payload):
            stats["input_rows"] += 1
            row = normalize_article(raw)
            if row is None:
                stats["filtered_rows"] += 1
                continue
            key = article_key(row)
            if key in unique:
                stats["duplicates"] += 1
                continue
            unique[key] = row
        if position % 100 == 0:
            print(f"news {position} files, {len(unique)} unique", flush=True)
    rows = sorted(unique.values(), key=lambda row: (row["timestamp"], row["headline"]))
    stats["unique_rows"] = len(rows)
    if rows:
        stats["start"] = rows[0]["timestamp"]
        stats["end"] = rows[-1]["timestamp"]
    return rows, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.json"))
    rows, stats = build_corpus(paths)
    payload = {
        "source_directory": str(args.input),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deduplication": "publication timestamp plus canonical URL/headline",
        "stats": stats, "articles": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
