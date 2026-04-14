#!/usr/bin/env python3
"""
fetch_simple_wikipedia.py — Download and parse Simple English Wikipedia
=======================================================================
Downloads the latest Simple English Wikipedia XML dump, extracts article
body text (no markup, no tables, no headers), and writes one article per
line to data/foundation/simple_wiki_articles.jsonl.

Each JSON line:
  { "id": "...", "title": "...", "text": "..." }

Where "text" is clean prose paragraphs only — no wiki markup, no section
headings, no template garbage.  This is the raw English foundation corpus.

Usage:
  python scripts/fetch_simple_wikipedia.py [--out data/foundation] [--limit N]

Dependencies:
  pip install requests mwparserfromhell

License of output data: CC-BY-SA 4.0 (Simple English Wikipedia)
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")

try:
    import mwparserfromhell
except ImportError:
    sys.exit("Missing: pip install mwparserfromhell")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)
# Mirror fallback if primary is slow:
DUMP_MIRROR = (
    "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)

NS_ARTICLE = "0"   # Only main namespace articles, skip Talk/User/etc.

# Minimum character length of cleaned article text to keep
MIN_TEXT_LEN = 150

# Wikitext patterns to strip before mwparserfromhell can't handle them
_GALLERY_RE   = re.compile(r"<gallery[^>]*>.*?</gallery>", re.S | re.I)
_REF_RE       = re.compile(r"<ref[^>]*>.*?</ref>",         re.S | re.I)
_REF_SELF_RE  = re.compile(r"<ref[^/]*/?>",                re.I)
_HTML_TAG_RE  = re.compile(r"<[^>]+>")
_MULTI_NL_RE  = re.compile(r"\n{3,}")

# ---------------------------------------------------------------------------

def download_dump(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download the bz2 dump to dest, showing progress."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    received = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size):
            fh.write(chunk)
            received += len(chunk)
            if total:
                pct = 100 * received / total
                mb  = received / (1 << 20)
                print(f"\r  {mb:.1f} MB / {total/(1<<20):.1f} MB  ({pct:.0f}%)", end="", flush=True)
    print(f"\nDownloaded {received/(1<<20):.1f} MB")


def clean_wikitext(raw: str) -> str:
    """Strip wiki markup and return plain prose text."""
    # Pre-clean HTML that mwparserfromhell doesn't handle
    raw = _GALLERY_RE.sub("", raw)
    raw = _REF_RE.sub("", raw)
    raw = _REF_SELF_RE.sub("", raw)

    parsed = mwparserfromhell.parse(raw)

    # Strip templates entirely (infoboxes, navboxes, etc.)
    for template in parsed.filter_templates():
        try:
            parsed.remove(template)
        except Exception:
            pass

    text = parsed.strip_code(normalize=True, collapse=True)

    # Remove remaining HTML tags
    text = _HTML_TAG_RE.sub("", text)

    # Collapse section headings (== Heading ==) → blank line
    text = re.sub(r"={2,}[^=\n]+={2,}", "", text)

    # Collapse runs of blank lines
    text = _MULTI_NL_RE.sub("\n\n", text)

    # Remove lines that are purely punctuation / numbers / single words
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 30:
            lines.append(stripped)

    return "\n".join(lines).strip()


def iter_articles(bz2_path: Path):
    """Yield (title, wikitext) for every article-namespace page."""
    NS = "http://www.mediawiki.org/xml/export-0.11/"
    with bz2.open(bz2_path, "rb") as fh:
        context = ET.iterparse(fh, events=("end",))
        for event, elem in context:
            if elem.tag != f"{{{NS}}}page":
                continue
            ns_elem = elem.find(f"{{{NS}}}ns")
            if ns_elem is None or ns_elem.text != NS_ARTICLE:
                elem.clear()
                continue
            title_elem = elem.find(f"{{{NS}}}title")
            rev_elem   = elem.find(f"{{{NS}}}revision")
            if rev_elem is None:
                elem.clear()
                continue
            text_elem = rev_elem.find(f"{{{NS}}}text")
            if text_elem is None or not text_elem.text:
                elem.clear()
                continue

            title = title_elem.text if title_elem is not None else ""
            wikitext = text_elem.text

            # Skip redirects
            if wikitext.strip().lower().startswith("#redirect"):
                elem.clear()
                continue

            yield title, wikitext
            elem.clear()


def process_dump(bz2_path: Path, out_path: Path, limit: int | None) -> None:
    """Parse the dump and write clean JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    print(f"Parsing dump: {bz2_path}")
    print(f"Writing to  : {out_path}")

    with open(out_path, "w", encoding="utf-8") as out_fh:
        for i, (title, wikitext) in enumerate(iter_articles(bz2_path)):
            if limit and kept >= limit:
                break

            text = clean_wikitext(wikitext)
            if len(text) < MIN_TEXT_LEN:
                skipped += 1
                continue

            record = {
                "id":    f"simplewiki_{i}",
                "title": title,
                "text":  text,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 5000 == 0:
                print(f"  {kept:,} articles kept  ({skipped:,} skipped)")

    print(f"\nDone. {kept:,} articles written, {skipped:,} skipped (too short/redirect).")
    size_mb = out_path.stat().st_size / (1 << 20)
    print(f"Output size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Simple English Wikipedia for foundation training")
    parser.add_argument("--out",   default="data/foundation", help="Output directory")
    parser.add_argument("--limit", type=int, default=None,    help="Max articles to keep (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if bz2 already exists")
    parser.add_argument("--mirror", action="store_true",
                        help="Use mirror URL instead of primary Wikimedia dump server")
    args = parser.parse_args()

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bz2_dest = out_dir / "simplewiki-latest-pages-articles.xml.bz2"
    jsonl_out = out_dir / "simple_wiki_articles.jsonl"

    if not args.skip_download or not bz2_dest.exists():
        url = DUMP_MIRROR if args.mirror else DUMP_URL
        try:
            download_dump(url, bz2_dest)
        except Exception as e:
            if not args.mirror:
                print(f"Primary failed ({e}), trying mirror...")
                download_dump(DUMP_MIRROR, bz2_dest)
            else:
                raise
    else:
        print(f"Skipping download — using existing {bz2_dest}")

    process_dump(bz2_dest, jsonl_out, args.limit)


if __name__ == "__main__":
    main()
