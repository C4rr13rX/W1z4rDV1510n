#!/usr/bin/env python3
"""
LibreTexts PDF downloader with cross-project deduplication.
Never re-downloads a textbook already present across any tracked project.
"""

import os
import re
import time
import hashlib
import argparse
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BOOKSHELVES = [
    "bio", "biz", "chem", "eng", "geo", "human",
    "k12", "math", "med", "phys", "socialsci", "stats",
    "workforce", "espanol",
]

LIBRETEXTS_BOOKSHELF_URL = "https://{sub}.libretexts.org/Bookshelves"
PDF_URL_TEMPLATE = "https://batch.libretexts.org/print/Letter/Finished/{print_id}/Full.pdf"

# All project textbook directories -- dedup across these
KNOWN_TEXTBOOK_DIRS = [
    Path("D:/Projects/StateOfLoci/textbooks"),
    Path("D:/w1z4rdv1510n-data/textbooks"),
]

OUTPUT_DIR = Path("D:/w1z4rdv1510n-data/textbooks")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; W1z4rDV1510n/1.0; textbook-fetcher)"
})


def build_existing_index() -> set[str]:
    """Collect all PDF filenames already downloaded across all project dirs."""
    seen = set()
    for d in KNOWN_TEXTBOOK_DIRS:
        if d.exists():
            for f in d.rglob("*.pdf"):
                seen.add(f.name.lower())
    return seen


def slugify(title: str) -> str:
    title = title.strip().lower()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"[\s_-]+", "_", title)
    return title[:120]


def fetch_books_on_shelf(sub: str) -> list[dict]:
    """Crawl a single LibreTexts bookshelf and return list of {title, print_id}."""
    url = LIBRETEXTS_BOOKSHELF_URL.format(sub=sub)
    books = []
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [warn] {sub} shelf unreachable: {e}")
        return books

    soup = BeautifulSoup(r.text, "html.parser")

    # Each book entry has a link with data-page-id attribute on an ancestor element,
    # or we can find article/section cards.
    for link in soup.select("a[href*='/Bookshelves/']"):
        page_id = None
        title_text = link.get_text(strip=True)
        if not title_text:
            continue

        # Walk up to find data-page-id
        el = link
        for _ in range(5):
            pid = el.get("data-page-id") or el.get("data-id")
            if pid:
                page_id = pid
                break
            parent = el.parent
            if parent is None:
                break
            el = parent

        if page_id:
            print_id = f"{sub}-{page_id}"
            books.append({"title": title_text, "print_id": print_id, "sub": sub})

    # Fallback: scan for data-page-id anywhere
    if not books:
        for el in soup.find_all(attrs={"data-page-id": True}):
            page_id = el.get("data-page-id")
            title_el = el.find(["h1", "h2", "h3", "a"])
            title_text = title_el.get_text(strip=True) if title_el else f"{sub}-{page_id}"
            print_id = f"{sub}-{page_id}"
            books.append({"title": title_text, "print_id": print_id, "sub": sub})

    return books


def pdf_exists_anywhere(filename: str, existing: set[str]) -> bool:
    return filename.lower() in existing


def download_pdf(print_id: str, dest: Path) -> bool:
    url = PDF_URL_TEMPLATE.format(print_id=print_id)
    try:
        r = SESSION.get(url, timeout=60, stream=True)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  [error] download failed {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def run(shelves: list[str], limit: int | None, dry_run: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = build_existing_index()
    print(f"Found {len(existing)} PDFs already downloaded across all projects.")

    total_downloaded = 0
    total_skipped = 0

    for sub in shelves:
        print(f"\n--- Shelf: {sub} ---")
        books = fetch_books_on_shelf(sub)
        print(f"  Found {len(books)} books on shelf.")

        shelf_dir = OUTPUT_DIR / sub
        for book in books:
            if limit is not None and total_downloaded >= limit:
                print(f"  Reached download limit ({limit}), stopping.")
                return

            filename = f"{slugify(book['title'])}__{book['print_id']}.pdf"
            dest = shelf_dir / filename

            if pdf_exists_anywhere(filename, existing):
                print(f"  [skip] already have: {filename}")
                total_skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] would download: {filename}")
                continue

            print(f"  [fetch] {book['title']} -> {filename}")
            ok = download_pdf(book["print_id"], dest)
            if ok:
                size_kb = dest.stat().st_size // 1024
                print(f"    saved {size_kb} KB")
                existing.add(filename.lower())
                total_downloaded += 1
                time.sleep(0.5)
            else:
                print(f"    [skip] PDF not available for {book['print_id']}")

    print(f"\nDone. Downloaded: {total_downloaded}, Skipped: {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibreTexts PDF downloader")
    parser.add_argument("--shelves", nargs="+", default=BOOKSHELVES,
                        help="Bookshelves to fetch (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max PDFs to download this run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without fetching")
    args = parser.parse_args()
    run(args.shelves, args.limit, args.dry_run)
