#!/usr/bin/env python3
"""
Fetch USPTO reaction datasets (full Lowe extractions + USPTO-50K) to data/uspto/.

Datasets:
- Full USPTO (Lowe-style) reactions: uspto.zip (~500MB compressed) from deepchemdata S3.
- USPTO-50K curated subset: uspto_50k.zip (~13MB compressed) from deepchemdata S3.

This script streams downloads with a progress bar and extracts the zip files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import requests
from tqdm import tqdm
import zipfile

DEFAULT_ROOT = Path("data") / "uspto"

MIRRORS: dict[str, list[str]] = {
    "full": [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/uspto.zip",
        "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/uspto.zip",
        "https://s3-us-west-1.amazonaws.com/deepchemdata/datasets/uspto.zip",
        # Hugging Face mirrors (if available):
        "https://huggingface.co/datasets/hwchang/USPTO-1MT/resolve/main/uspto.zip",
        "https://huggingface.co/datasets/mlabonne/USPTO-full/resolve/main/uspto.zip",
        "https://figshare.com/ndownloader/articles/25242010/versions/1",
    ],
    "50k": [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/uspto_50k.zip",
        "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/uspto_50k.zip",
        "https://s3-us-west-1.amazonaws.com/deepchemdata/datasets/uspto_50k.zip",
        # Hugging Face mirrors (popular community uploads):
        "https://huggingface.co/datasets/pschwllr/USPTO_50K/resolve/main/uspto_50k.zip",
        "https://huggingface.co/datasets/samoturk/USPTO-50K/resolve/main/uspto_50k.zip",
        "https://huggingface.co/datasets/pschwllr/USPTO_50K/resolve/main/uspto_50k.zip?download=1",
        "https://huggingface.co/datasets/samoturk/USPTO-50K/resolve/main/uspto_50k.zip?download=1",
        "https://raw.githubusercontent.com/pschwllr/USPTO_50K/main/uspto_50k.zip",
        "https://raw.githubusercontent.com/wengong-jin/nips17-rexgen/master/dataset/uspto-50k.zip",
        "https://raw.githubusercontent.com/bp-kelley/uspto-50k/master/uspto_50k.zip",
        "https://figshare.com/ndownloader/articles/25325623/versions/1",
    ],
}


def stream_download(url: str, target: Path, chunk_size: int = 1 << 20) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        desc = f"Downloading {target.name}"
        with open(target, "wb") as fout, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc, leave=False
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fout.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def fetch_dataset(name: str, url: str, root: Path) -> Tuple[Path, Path]:
    archive = root / f"{name}.zip"
    if archive.exists():
        print(f"[skip] {archive} already exists")
    else:
        print(f"[fetch] {name} from {url}")
        stream_download(url, archive)
    extract_dir = root / name
    print(f"[extract] {archive} -> {extract_dir}")
    extract_zip(archive, extract_dir)
    return archive, extract_dir


def fetch_with_fallback(name: str, root: Path) -> Tuple[Path, Path]:
    urls = MIRRORS.get(name, [])
    if not urls:
        raise SystemExit(f"No mirrors configured for dataset: {name}")
    for url in urls:
        try:
            return fetch_dataset(name, url, root)
        except Exception as exc:
            print(f"[warn] fetch failed for {url}: {exc}")
            continue
    raise SystemExit(f"Failed to fetch dataset {name} from all known mirrors")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch USPTO reaction datasets")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Destination root directory (default: data/uspto)",
    )
    parser.add_argument(
        "--only",
        choices=["full", "50k"],
        default=None,
        help="Optional: fetch only one dataset",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.root
    selected = {args.only} if args.only else set(["full", "50k"])
    for key in selected:
        fetch_with_fallback(key, root)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
