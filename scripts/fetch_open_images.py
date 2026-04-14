#!/usr/bin/env python3
"""
fetch_open_images.py — Download Open Images v7 images for concept dataset
=========================================================================
Reads data/foundation/concept_dataset.jsonl.
For each concept that has an open_images_class ID, downloads N images
from Open Images v7 (CC-BY 4.0).

Uses the official Open Images image URLs from the OI metadata CSVs.
Downloads a configurable number of images per concept (default 10).

Updates concept_dataset.jsonl with an "images" list of local file paths.

Usage:
  python scripts/fetch_open_images.py [--out data/foundation/oi_images]
                                      [--per-concept 10]
                                      [--max-concepts 500]

Open Images v7 metadata files needed (~1.4GB total for val+test CSV):
  - https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
  - https://storage.googleapis.com/openimages/2018_04/validation/validation-images-boxable.csv
  - https://storage.googleapis.com/openimages/v6/validation-annotations-bbox.csv

The script auto-downloads the metadata CSVs if not present.
Image downloads are parallelised and skipped if already present.

Dependencies:
  pip install requests httpx

Data license: Images are CC-BY 4.0 (Open Images Dataset)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

# ---------------------------------------------------------------------------
# Open Images metadata URLs
# ---------------------------------------------------------------------------

OI_CLASS_DESC_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)
OI_VAL_IMAGES_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
OI_VAL_BBOX_URL = (
    "https://storage.googleapis.com/openimages/v5/"
    "validation-annotations-bbox.csv"
)

# We'll also use train subset for more images if needed
OI_TRAIN_IMAGES_URL_TPL = (
    "https://storage.googleapis.com/openimages/2018_04/train/"
    "train-images-boxable-{:02d}.csv"  # chunks 00-09
)


# ---------------------------------------------------------------------------

def download_csv(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  Already have {dest.name}")
        return
    print(f"  Downloading {dest.name} ({url})")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    received = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size):
            fh.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r    {received/(1<<20):.0f}/{total/(1<<20):.0f} MB", end="", flush=True)
    print(f"\r  Done: {dest.name} ({received/(1<<20):.1f} MB)")


def load_class_descriptions(csv_path: Path) -> dict[str, str]:
    """Returns {label_name: class_id} and {class_id: label_name}."""
    id_to_name: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            class_id, label_name = row[0], row[1].lower().strip()
            id_to_name[class_id] = label_name
            name_to_id[label_name] = class_id
    return id_to_name, name_to_id


def build_class_to_image_urls(
    images_csv: Path, bbox_csv: Path, wanted_class_ids: set[str], max_per_class: int
) -> dict[str, list[str]]:
    """
    Returns {class_id: [image_url, ...]} for the requested class IDs.
    Uses bbox annotations to find images containing each class.
    """
    # Step 1: Load bbox annotations → {image_id: set(class_ids)}
    print(f"  Loading bbox annotations from {bbox_csv.name}...")
    img_to_classes: dict[str, set[str]] = {}
    with open(bbox_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row.get("LabelName", "")
            iid = row.get("ImageID", "")
            if cid in wanted_class_ids:
                img_to_classes.setdefault(iid, set()).add(cid)

    # Invert: class → image_ids
    class_to_imgs: dict[str, list[str]] = {}
    for img_id, classes in img_to_classes.items():
        for cid in classes:
            class_to_imgs.setdefault(cid, []).append(img_id)

    # Step 2: Load image URLs from images CSV → {image_id: url}
    print(f"  Loading image URLs from {images_csv.name}...")
    needed_ids = set()
    for imgs in class_to_imgs.values():
        needed_ids.update(imgs[:max_per_class])

    id_to_url: dict[str, str] = {}
    with open(images_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            iid = row.get("ImageID", "")
            if iid in needed_ids:
                url = row.get("OriginalURL", "")
                if url:
                    id_to_url[iid] = url

    # Build final mapping
    result: dict[str, list[str]] = {}
    for cid, img_ids in class_to_imgs.items():
        urls = [id_to_url[i] for i in img_ids[:max_per_class] if i in id_to_url]
        if urls:
            result[cid] = urls

    return result


async def download_image(client: httpx.AsyncClient, url: str, dest: Path,
                         sem: asyncio.Semaphore) -> bool:
    if dest.exists():
        return True
    async with sem:
        try:
            r = await client.get(url, timeout=20, follow_redirects=True)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return True
        except Exception:
            pass
    return False


async def download_all_images(
    class_to_urls: dict[str, list[str]],
    class_to_concept: dict[str, str],
    out_dir: Path,
    per_concept: int,
    concurrency: int,
) -> dict[str, list[str]]:
    """Download images for all concepts. Returns {class_id: [local_path...]}."""
    sem = asyncio.Semaphore(concurrency)
    class_to_local: dict[str, list[str]] = {}
    total_ok = 0
    total_fail = 0

    async with httpx.AsyncClient(timeout=20) as client:
        tasks = []
        dest_map = []   # parallel list of (class_id, dest_path)

        for class_id, urls in class_to_urls.items():
            concept = class_to_concept.get(class_id, class_id.replace("/", "_"))
            concept_dir = out_dir / concept.replace(" ", "_").replace("/", "_")
            concept_dir.mkdir(parents=True, exist_ok=True)

            for i, url in enumerate(urls[:per_concept]):
                ext = ".jpg"
                dest = concept_dir / f"{i:03d}{ext}"
                tasks.append(asyncio.create_task(
                    download_image(client, url, dest, sem)
                ))
                dest_map.append((class_id, dest))

        print(f"  Downloading {len(tasks)} images ({concurrency} concurrent)...")
        results = await asyncio.gather(*tasks)

        for (class_id, dest), ok in zip(dest_map, results):
            if ok:
                class_to_local.setdefault(class_id, []).append(str(dest))
                total_ok += 1
            else:
                total_fail += 1

    print(f"  Downloaded: {total_ok} ok, {total_fail} failed")
    return class_to_local


async def main_async(args: argparse.Namespace) -> None:
    out_dir     = Path(args.out)
    meta_dir    = out_dir / "_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_jsonl = Path(args.concepts)
    if not concept_jsonl.exists():
        sys.exit(f"Concept dataset not found: {concept_jsonl}\n"
                 "Run build_concept_dataset.py first.")

    # ── Load concept dataset ──────────────────────────────────────────────
    records = []
    with open(concept_jsonl, encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    # Collect the OI class IDs we need
    class_to_concept: dict[str, str] = {}
    for r in records:
        cid = r.get("open_images_class", "")
        if cid:
            class_to_concept[cid] = r["clean"]

    if not class_to_concept:
        print("No Open Images class IDs found in concept dataset.")
        return

    wanted = set(class_to_concept.keys())
    if args.max_concepts:
        # Take the first N by order in file
        seen_cids = set()
        limited = {}
        for r in records:
            cid = r.get("open_images_class", "")
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                limited[cid] = r["clean"]
            if len(limited) >= args.max_concepts:
                break
        wanted        = set(limited.keys())
        class_to_concept = limited

    print(f"Fetching images for {len(wanted)} concept classes, {args.per_concept} each...")

    # ── Download OI metadata ──────────────────────────────────────────────
    class_desc_csv = meta_dir / "oidv6-class-descriptions.csv"
    val_images_csv = meta_dir / "validation-images-boxable.csv"
    val_bbox_csv   = meta_dir / "validation-annotations-bbox.csv"

    print("\n[1/3] Class descriptions:")
    download_csv(OI_CLASS_DESC_URL, class_desc_csv)

    print("\n[2/3] Validation image list:")
    download_csv(OI_VAL_IMAGES_URL, val_images_csv)

    print("\n[3/3] Validation bounding box annotations:")
    download_csv(OI_VAL_BBOX_URL, val_bbox_csv)

    # ── Build class → image URL map ───────────────────────────────────────
    print("\nBuilding class-to-image map...")
    class_to_urls = build_class_to_image_urls(
        val_images_csv, val_bbox_csv, wanted, args.per_concept * 3
    )
    found = len(class_to_urls)
    print(f"  Found images for {found}/{len(wanted)} classes in val set")

    if found < len(wanted) // 2:
        print("  NOTE: val set has limited coverage; consider adding train chunks")

    # ── Download images ───────────────────────────────────────────────────
    print(f"\nDownloading images to {out_dir}/...")
    class_to_local = await download_all_images(
        class_to_urls, class_to_concept, out_dir,
        args.per_concept, args.concurrency
    )

    # ── Update concept dataset with image paths ───────────────────────────
    print("\nUpdating concept_dataset.jsonl with image paths...")
    updated = 0
    updated_records = []
    for r in records:
        cid = r.get("open_images_class", "")
        if cid and cid in class_to_local:
            r["images"] = class_to_local[cid]
            updated += 1
        elif "images" not in r:
            r["images"] = []
        updated_records.append(r)

    with open(concept_jsonl, "w", encoding="utf-8") as fh:
        for r in updated_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  {updated} concepts now have images")
    total_imgs = sum(len(r.get("images", [])) for r in updated_records)
    print(f"  Total image files: {total_imgs}")
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Open Images v7 images for concept dataset"
    )
    parser.add_argument("--concepts",     default="data/foundation/concept_dataset.jsonl")
    parser.add_argument("--out",          default="data/foundation/oi_images",
                        help="Output directory for downloaded images")
    parser.add_argument("--per-concept",  type=int, default=10,
                        help="Images per concept class (default 10)")
    parser.add_argument("--max-concepts", type=int, default=None,
                        help="Limit to first N concepts with OI classes")
    parser.add_argument("--concurrency",  type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
