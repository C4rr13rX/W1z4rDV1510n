#!/usr/bin/env python3
"""
Poll RSS/Atom feeds and emit StreamEnvelope JSONL for PublicTopics.

Outputs JSONL envelopes (StreamSource=PUBLIC_TOPICS) with quality metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

USER_AGENT = "W1z4rDV1510n-RSS/0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSS/Atom -> PublicTopics JSONL bridge")
    parser.add_argument("--feed", action="append", default=[], help="Feed URL (repeatable)")
    parser.add_argument("--feed-file", help="Path to file with feed URLs (one per line)")
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Polling interval seconds")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    parser.add_argument("--max-backoff", type=float, default=600.0, help="Max backoff seconds")
    parser.add_argument("--max-latency", type=float, default=86400.0, help="Max latency for quality")
    parser.add_argument("--max-items-per-poll", type=int, default=50, help="Emit at most N items per poll")
    parser.add_argument("--max-seen", type=int, default=10000, help="Max seen items to remember")
    parser.add_argument("--state-path", help="Persist seen item IDs to JSON file")
    parser.add_argument("--topic", help="Override topic label for all items")
    parser.add_argument("--once", action="store_true", help="Fetch once and exit")
    return parser.parse_args()


def now_unix() -> int:
    return int(time.time())


def parse_datetime(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def read_feed_urls(args: argparse.Namespace) -> List[str]:
    urls = list(args.feed)
    if args.feed_file:
        with open(args.feed_file, "r", encoding="utf-8") as handle:
            for line in handle:
                url = line.strip()
                if url and not url.startswith("#"):
                    urls.append(url)
    if not urls:
        raise SystemExit("At least one --feed or --feed-file is required")
    return urls


def fetch_url(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def find_text(elem: ET.Element, tag: str) -> str:
    node = elem.find(tag)
    if node is None:
        node = elem.find(f".//{{*}}{tag}")
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def find_link(elem: ET.Element) -> str:
    node = elem.find("link")
    if node is None:
        node = elem.find(".//{*}link")
    if node is None:
        return ""
    href = node.attrib.get("href")
    if href:
        return href.strip()
    if node.text:
        return node.text.strip()
    return ""


def parse_feed(xml_bytes: bytes) -> Tuple[str, str, List[Dict[str, str]]]:
    root = ET.fromstring(xml_bytes)
    feed_title = ""
    kind = "rss"
    items: List[Dict[str, str]] = []
    if root.tag.endswith("feed"):
        kind = "atom"
        feed_title = find_text(root, "title")
        for entry in root.findall(".//{*}entry"):
            items.append(parse_atom_entry(entry))
    else:
        channel = root.find("channel")
        if channel is None:
            channel = root.find(".//{*}channel")
        if channel is not None:
            feed_title = find_text(channel, "title")
            for item in channel.findall("item"):
                items.append(parse_rss_item(item))
            if not items:
                for item in channel.findall(".//{*}item"):
                    items.append(parse_rss_item(item))
    return kind, feed_title, items


def parse_rss_item(item: ET.Element) -> Dict[str, str]:
    return {
        "id": find_text(item, "guid") or find_link(item) or find_text(item, "title"),
        "title": find_text(item, "title"),
        "summary": find_text(item, "description"),
        "link": find_link(item),
        "published": find_text(item, "pubDate"),
        "author": find_text(item, "author"),
        "category": find_text(item, "category"),
    }


def parse_atom_entry(entry: ET.Element) -> Dict[str, str]:
    return {
        "id": find_text(entry, "id") or find_link(entry) or find_text(entry, "title"),
        "title": find_text(entry, "title"),
        "summary": find_text(entry, "summary") or find_text(entry, "content"),
        "link": find_link(entry),
        "published": find_text(entry, "updated") or find_text(entry, "published"),
        "author": find_text(entry, "author"),
        "category": find_text(entry, "category"),
    }


def compute_quality(latency: float, max_latency: float, missing: int) -> float:
    latency_ratio = min(max(latency / max_latency, 0.0), 1.0) if max_latency > 0 else 0.0
    latency_penalty = latency_ratio * 0.4
    missing_penalty = min(missing * 0.15, 0.6)
    quality = 1.0 - latency_penalty - missing_penalty
    return max(0.0, min(1.0, quality))


def build_envelope(topic: str, intensity: float, timestamp: int, metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "source": "PUBLIC_TOPICS",
        "timestamp": {"unix": timestamp},
        "payload": {
            "kind": "JSON",
            "value": {
                "topic": topic,
                "intensity": intensity,
                "metadata": metadata,
            },
        },
        "metadata": metadata,
    }


def load_state(path: Optional[str]) -> deque:
    if not path:
        return deque()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return deque(data)
    except Exception:
        return deque()
    return deque()


def save_state(path: Optional[str], seen: deque) -> None:
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(list(seen), handle)
    except Exception:
        return


def emit(envelope: Dict[str, object]) -> None:
    sys.stdout.write(json.dumps(envelope) + "\n")
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    urls = read_feed_urls(args)
    seen = load_state(args.state_path)
    seen_set = set(seen)
    backoff = args.poll_interval
    failures = 0
    while True:
        for url in urls:
            try:
                xml_bytes = fetch_url(url, args.timeout)
                kind, feed_title, items = parse_feed(xml_bytes)
            except (URLError, ET.ParseError, ValueError):
                failures += 1
                backoff = min(args.poll_interval * (2 ** failures), args.max_backoff)
                continue
            failures = 0
            emitted = 0
            for item in items:
                item_id = item.get("id") or ""
                if not item_id:
                    continue
                if item_id in seen_set:
                    continue
                published_unix = parse_datetime(item.get("published"))
                if published_unix is None:
                    published_unix = now_unix()
                latency = max(0.0, now_unix() - published_unix)
                missing_fields = [
                    name
                    for name in ("title", "summary", "link")
                    if not item.get(name)
                ]
                quality = compute_quality(latency, args.max_latency, len(missing_fields))
                intensity = max(0.1, 1.0 - min(latency / args.max_latency, 1.0)) if args.max_latency > 0 else 1.0
                topic = args.topic or item.get("category") or feed_title or "news"
                metadata = {
                    "source_url": url,
                    "feed_kind": kind,
                    "feed_title": feed_title,
                    "item_id": item_id,
                    "title": item.get("title") or "",
                    "summary": item.get("summary") or "",
                    "link": item.get("link") or "",
                    "author": item.get("author") or "",
                    "category": item.get("category") or "",
                    "published_unix": published_unix,
                    "latency_secs": latency,
                    "missing_fields": missing_fields,
                    "quality": quality,
                    "confidence": quality,
                    "ingestor": "rss_topic_bridge",
                }
                envelope = build_envelope(topic, intensity, published_unix, metadata)
                emit(envelope)
                emitted += 1
                seen.append(item_id)
                seen_set.add(item_id)
                if len(seen) > args.max_seen:
                    oldest = seen.popleft()
                    seen_set.discard(oldest)
                if emitted >= args.max_items_per_poll:
                    break
            save_state(args.state_path, seen)
            backoff = args.poll_interval
        if args.once:
            break
        time.sleep(backoff)


if __name__ == "__main__":
    main()
