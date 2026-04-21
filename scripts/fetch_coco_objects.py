#!/usr/bin/env python3
"""
fetch_coco_objects.py -- Download COCO 2017 val + annotations for foundation training
======================================================================================
Downloads COCO 2017 validation set (5,000 images) and captions annotations.
Produces a JSONL index at data/foundation/coco_val_index.jsonl where each line:

  {
    "image_id": 12345,
    "file": "data/foundation/coco_val2017/000000012345.jpg",
    "captions": ["A cat sitting on a mat.", "..."],
    "categories": ["cat", "mat"],    # from instance annotations
    "width": 640, "height": 480
  }

This is fed into train_foundation.py which ingests the image (via /neuro/ingest_image)
alongside its captions and category labels as co-activating text.

Usage:
  python scripts/fetch_coco_objects.py [--out data/foundation]

Requires:
  pip install requests

Data license: CC-BY 4.0 (COCO dataset, Microsoft)
Images license: Flickr (various CC licenses -- val set is CC-BY 2.0 or similar)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")

# ---------------------------------------------------------------------------
# COCO 2017 URLs
# ---------------------------------------------------------------------------

COCO_VAL_IMAGES_URL      = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL     = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name} -- skipping download")
        return
    print(f"  Downloading: {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total    = int(r.headers.get("content-length", 0))
    received = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size):
            fh.write(chunk)
            received += len(chunk)
            if total:
                pct = 100 * received / total
                mb  = received / (1 << 20)
                print(f"\r    {mb:.0f} / {total/(1<<20):.0f} MB  ({pct:.0f}%)", end="", flush=True)
    print(f"\n  Done: {received/(1<<20):.1f} MB")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} -> {dest_dir}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for i, member in enumerate(members):
            zf.extract(member, dest_dir)
            if (i + 1) % 1000 == 0:
                print(f"\r    {i+1}/{len(members)} files", end="", flush=True)
    print(f"\r    {len(members)} files extracted")


def build_index(out_dir: Path) -> None:
    """Combine captions + instances into per-image JSONL index."""
    ann_dir = out_dir / "annotations"
    img_dir = out_dir / "val2017"

    captions_file  = ann_dir / "captions_val2017.json"
    instances_file = ann_dir / "instances_val2017.json"

    if not captions_file.exists():
        print(f"ERROR: {captions_file} not found -- annotations not extracted?")
        return
    if not instances_file.exists():
        print(f"WARNING: {instances_file} not found -- skipping category labels")

    print("Building per-image index...")

    # Load captions
    with open(captions_file, encoding="utf-8") as fh:
        captions_data = json.load(fh)

    # image_id -> list of caption strings
    cap_map: dict[int, list[str]] = {}
    for ann in captions_data["annotations"]:
        cap_map.setdefault(ann["image_id"], []).append(ann["caption"])

    # image_id -> image metadata
    img_meta: dict[int, dict] = {}
    for img in captions_data["images"]:
        img_meta[img["id"]] = img

    # Load category names for instance labels
    cat_map: dict[int, str] = {}    # category_id -> name
    inst_map: dict[int, list[str]] = {}  # image_id -> list of category names
    if instances_file.exists():
        with open(instances_file, encoding="utf-8") as fh:
            inst_data = json.load(fh)
        for cat in inst_data["categories"]:
            cat_map[cat["id"]] = cat["name"]
        for ann in inst_data["annotations"]:
            name = cat_map.get(ann["category_id"], "")
            if name:
                inst_map.setdefault(ann["image_id"], [])
                if name not in inst_map[ann["image_id"]]:
                    inst_map[ann["image_id"]].append(name)

    # Write JSONL
    index_path = out_dir / "coco_val_index.jsonl"
    written = 0
    missing = 0
    with open(index_path, "w", encoding="utf-8") as out_fh:
        for img_id, meta in img_meta.items():
            file_path = img_dir / meta["file_name"]
            if not file_path.exists():
                missing += 1
                continue
            record = {
                "image_id":  img_id,
                "file":      str(file_path),
                "captions":  cap_map.get(img_id, []),
                "categories": inst_map.get(img_id, []),
                "width":     meta.get("width", 0),
                "height":    meta.get("height", 0),
            }
            out_fh.write(json.dumps(record) + "\n")
            written += 1

    print(f"Index written: {index_path}")
    print(f"  {written:,} images indexed, {missing:,} image files missing")
    size_mb = index_path.stat().st_size / (1 << 20)
    print(f"  Index size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO 2017 val for foundation training")
    parser.add_argument("--out", default="data/foundation", help="Output directory")
    parser.add_argument("--skip-images", action="store_true", help="Skip image download (only get annotations)")
    parser.add_argument("--index-only",  action="store_true", help="Only rebuild index, no downloads")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_only:
        # Download annotations (small -- ~241MB)
        print("\n[1/2] Annotations:")
        ann_zip = out_dir / "annotations_trainval2017.zip"
        download_file(COCO_ANNOTATIONS_URL, ann_zip)
        ann_dest = out_dir / "annotations"
        if not ann_dest.exists():
            extract_zip(ann_zip, out_dir)

        # Download val images (~1GB)
        if not args.skip_images:
            print("\n[2/2] Val images (~1 GB):")
            img_zip = out_dir / "val2017.zip"
            download_file(COCO_VAL_IMAGES_URL, img_zip)
            img_dest = out_dir / "val2017"
            if not img_dest.exists():
                extract_zip(img_zip, out_dir)
        else:
            print("\n[2/2] Skipping images (--skip-images)")

    print("\nBuilding index...")
    build_index(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
