#!/usr/bin/env python3
"""
train_foundation.py — Feed English foundation corpus into the neural node
=========================================================================
Three-pass foundation training pipeline.  Run passes in order:

  Pass 0 (PRIMARY) — Concept dataset (build_concept_dataset.py output)
    data/foundation/concept_dataset.jsonl, processed LEVEL BY LEVEL (-1 → 3).
    For each concept:
      - Definition text → /media/train (text modality)
      - Each image + definition caption → /media/train (page modality)
      - QA pairs (definition, misconception/correction) → /qa/ingest
    This is the developmental curriculum pass — first words, then infant
    vocabulary, then toddler, pre-K, kindergarten.

  Pass 1 — Wikipedia prose (Simple English Wikipedia)
    data/foundation/simple_wiki_articles.jsonl
    Full articles as text → /media/train.  Builds broad English word
    co-occurrence representations to support K-12 reading.

  Pass 2 — Scene images (COCO 2017 val)
    data/foundation/coco_val_index.jsonl
    Real-world scene images + captions → /media/train (page modality).

Usage:
  python scripts/train_foundation.py [--node http://127.0.0.1:8090]
                                     [--pass concepts|text|images|all]
                                     [--limit N]
                                     [--concurrency 10]
                                     [--batch-size 50]

Dependencies:
  pip install httpx Pillow
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing: pip install Pillow")

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

NODE_URL     = "http://127.0.0.1:8090"
CONCURRENCY  = 40      # max concurrent HTTP requests
BATCH_SIZE   = 50      # QA pairs per /qa/ingest request
IMAGE_MAX_PX = 512     # resize longest edge before base64

MIN_TEXT_LEN = 60      # min character length for a text block to train

# ---------------------------------------------------------------------------
# QA extraction (same filter rules as train_k12.py)
# ---------------------------------------------------------------------------

_IS_RE = re.compile(
    r"^([A-Z][A-Za-z\s\-']{2,50})\s+(?:is|are|was|were|refers? to|means?)\s+(.{40,500})\.$",
    re.MULTILINE,
)
_SKIP_SUBJECTS = {
    "this", "that", "these", "those", "it", "they", "he", "she", "we", "you", "i",
    "a", "an", "the",
    "because", "when", "if", "although", "while", "since", "once",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "there", "here", "then", "now",
    "no", "not", "one", "some", "many", "most", "all", "each", "both",
    "several", "few", "any", "every", "either", "neither",
    "for", "with", "by", "from", "of", "in", "on", "at", "as", "to",
}
_CONTEXT_SUBJ_RE = re.compile(r"\b(of the|of a|of an|in the|in a|by the|for the)\b", re.I)
_MATH_ONLY_RE    = re.compile(r"^[\d\s+\-*/=.,()[\]\\<>%°±×÷√π]+$")


def extract_qa_from_text(text: str, source_id: str) -> list[dict]:
    pairs = []
    for m in _IS_RE.finditer(text):
        subj = m.group(1).strip()
        defn = m.group(2).strip()
        if subj.lower() in _SKIP_SUBJECTS:
            continue
        if len(subj) > 60 or len(subj.split()) > 6:
            continue
        if _CONTEXT_SUBJ_RE.search(subj):
            continue
        words = defn.split()
        if len(defn) < 30 or len(words) < 4:
            continue
        if _MATH_ONLY_RE.match(defn):
            continue
        if not any(len(w.strip(".,;:()[]\"'")) >= 5 for w in words[:8]):
            continue
        if defn.rstrip().endswith("?"):
            continue
        first = words[0].strip(".,;:()[]\"'")
        if not (first and (first[0].isupper() or first.lower() in ("a", "an", "the"))):
            continue
        pairs.append({
            "qa_id":         str(uuid.uuid4()),
            "question":      f"What is {subj.lower()}?",
            "answer":        defn,
            "book_id":       source_id,
            "page_index":    0,
            "confidence":    0.75,
            "evidence":      subj,
            "review_status": "foundation",
        })
    return pairs


# ---------------------------------------------------------------------------
# Async HTTP helpers
# ---------------------------------------------------------------------------

async def media_train_text(client: httpx.AsyncClient, node_url: str, text: str) -> bool:
    try:
        r = await client.post(
            f"{node_url}/media/train",
            json={"modality": "text", "text": text, "lr_scale": 1.0},
            timeout=30,
        )
        return r.status_code == 200
    except Exception:
        return False


async def media_train_image(client: httpx.AsyncClient, node_url: str,
                            img_b64: str, caption: str) -> bool:
    try:
        r = await client.post(
            f"{node_url}/media/train",
            json={"modality": "page", "data_b64": img_b64, "text": caption, "lr_scale": 1.0},
            timeout=30,
        )
        return r.status_code == 200
    except Exception:
        return False


async def qa_ingest_batch(client: httpx.AsyncClient, node_url: str,
                          batch: list[dict]) -> int:
    try:
        r = await client.post(
            f"{node_url}/qa/ingest",
            json={"candidates": batch},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("ingested", 0)
        return 0
    except Exception:
        return 0


def resize_image_b64(image_path: str, max_px: int = IMAGE_MAX_PX) -> str | None:
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pass 0 — Concept dataset (PRIMARY: developmental vocabulary + images)
# ---------------------------------------------------------------------------

LEVEL_NAMES = {-1: "first-words", 0: "infant", 1: "toddler",
               2: "pre-K", 3: "kindergarten"}


async def run_concept_pass(node_url: str, data_dir: Path, limit: int | None,
                           concurrency: int, batch_size: int) -> None:
    jsonl = data_dir / "concept_dataset.jsonl"
    if not jsonl.exists():
        print(f"[concepts] {jsonl} not found — run build_concept_dataset.py first")
        return

    records = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    # Sort by level then frequency (most common first within level)
    records.sort(key=lambda r: (r.get("level", 0), -r.get("frequency", 0)))

    if limit:
        records = records[:limit]

    total = len(records)
    print(f"\n[CONCEPT PASS] {total:,} concepts across levels "
          f"{min(r.get('level',0) for r in records)} to "
          f"{max(r.get('level',0) for r in records)}")
    print(f"  concurrency={concurrency}  batch_size={batch_size}")

    sem     = asyncio.Semaphore(concurrency)
    qa_buf: list[dict] = []
    stats   = {"text_ok": 0, "text_fail": 0,
               "img_ok": 0, "img_skip": 0, "img_fail": 0, "qa": 0}
    t_start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:

        async def train_text(text: str) -> None:
            async with sem:
                ok = await media_train_text(client, node_url, text)
            stats["text_ok" if ok else "text_fail"] += 1

        async def train_img(img_path: str, caption: str) -> None:
            loop = asyncio.get_running_loop()
            b64  = await loop.run_in_executor(None, resize_image_b64, img_path)
            if b64 is None:
                stats["img_skip"] += 1
                return
            async with sem:
                ok = await media_train_image(client, node_url, b64, caption)
            stats["img_ok" if ok else "img_fail"] += 1

        async def flush_qa(pairs: list[dict]) -> None:
            n = await qa_ingest_batch(client, node_url, pairs)
            stats["qa"] += n

        tasks: list[asyncio.Task] = []
        current_level = None

        for idx, rec in enumerate(records):
            level   = rec.get("level", 0)
            concept = rec.get("concept", "")
            defn    = rec.get("definition", "")
            images  = rec.get("images", [])
            qa_pairs = rec.get("qa_pairs", [])
            wiki_text = rec.get("wiki_text", "")

            # Announce level transitions
            if level != current_level:
                current_level = level
                print(f"\n  --- Level {level}: {LEVEL_NAMES.get(level, str(level))} ---")

            # Build full training text: definition + wiki excerpt if available
            full_text = defn
            if wiki_text and len(wiki_text) > len(defn):
                full_text = defn + "  " + wiki_text[:400]

            if full_text.strip():
                tasks.append(asyncio.create_task(train_text(full_text)))

            # Train each image paired with the definition as caption
            caption = defn[:200] if defn else concept
            for img_path in images[:8]:   # max 8 images per concept
                tasks.append(asyncio.create_task(train_img(img_path, caption)))

            # Collect QA pairs (add source info)
            for qa in qa_pairs:
                qa_with_source = dict(qa)
                qa_with_source["book_id"]    = f"concept_{concept}"
                qa_with_source["page_index"] = level + 1
                qa_buf.append(qa_with_source)

            while len(qa_buf) >= batch_size:
                batch = qa_buf[:batch_size]
                del qa_buf[:batch_size]
                tasks.append(asyncio.create_task(flush_qa(batch)))

            if len(tasks) >= concurrency * 4:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{idx+1}/{total}] concept={concept!r:<20} "
                      f"txt={stats['text_ok']}  img={stats['img_ok']}  "
                      f"qa={stats['qa']}  {rate:.1f}/s")

        if qa_buf:
            tasks.append(asyncio.create_task(flush_qa(qa_buf)))
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - t_start
    print(f"\n[CONCEPT PASS] Done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Text trained  : {stats['text_ok']:,} ok  {stats['text_fail']:,} failed")
    print(f"  Images trained: {stats['img_ok']:,} ok  "
          f"{stats['img_skip']:,} skip  {stats['img_fail']:,} failed")
    print(f"  QA ingested   : {stats['qa']:,}")


# ---------------------------------------------------------------------------
# Pass 1 — Text (Simple English Wikipedia)
# ---------------------------------------------------------------------------

async def run_text_pass(node_url: str, data_dir: Path, limit: int | None,
                        concurrency: int, batch_size: int) -> None:
    jsonl = data_dir / "simple_wiki_articles.jsonl"
    if not jsonl.exists():
        print(f"[text] {jsonl} not found — run fetch_simple_wikipedia.py first")
        return

    print(f"\n[TEXT PASS] {jsonl}")
    print(f"  concurrency={concurrency}  batch_size={batch_size}")

    sem      = asyncio.Semaphore(concurrency)
    qa_buf:  list[dict] = []
    stats    = {"ok": 0, "fail": 0, "qa": 0}
    t_start  = time.time()

    async with httpx.AsyncClient(timeout=30) as client:

        async def train_one(text: str) -> None:
            async with sem:
                ok = await media_train_text(client, node_url, text)
            if ok:
                stats["ok"] += 1
            else:
                stats["fail"] += 1

        async def flush_qa(pairs: list[dict]) -> None:
            n = await qa_ingest_batch(client, node_url, pairs)
            stats["qa"] += n

        tasks: list[asyncio.Task] = []
        count = 0

        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                if limit and count >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = rec.get("text", "").strip()
                source_id = rec.get("id", str(count))

                if len(text) >= MIN_TEXT_LEN:
                    tasks.append(asyncio.create_task(train_one(text)))

                # QA extraction
                pairs = extract_qa_from_text(text, source_id)
                qa_buf.extend(pairs)
                while len(qa_buf) >= batch_size:
                    batch = qa_buf[:batch_size]
                    del qa_buf[:batch_size]
                    tasks.append(asyncio.create_task(flush_qa(batch)))

                count += 1

                # Drain completed tasks to avoid memory explosion
                if len(tasks) >= concurrency * 3:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

                if count % 5000 == 0:
                    elapsed = time.time() - t_start
                    rate = stats["ok"] / elapsed if elapsed > 0 else 0
                    print(f"  {count:,} articles  {stats['ok']:,} trained  "
                          f"{stats['qa']:,} qa  {rate:.1f} art/s")

        # Flush remaining QA
        if qa_buf:
            tasks.append(asyncio.create_task(flush_qa(qa_buf)))

        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - t_start
    print(f"\n[TEXT PASS] Done in {elapsed:.0f}s")
    print(f"  Articles trained: {stats['ok']:,} ok  {stats['fail']:,} failed")
    print(f"  QA pairs ingested: {stats['qa']:,}")


# ---------------------------------------------------------------------------
# Pass 2 — Images (COCO)
# ---------------------------------------------------------------------------

async def run_images_pass(node_url: str, data_dir: Path, limit: int | None,
                          concurrency: int, batch_size: int) -> None:
    jsonl = data_dir / "coco_val_index.jsonl"
    if not jsonl.exists():
        print(f"[images] {jsonl} not found — run fetch_coco_objects.py first")
        return

    print(f"\n[IMAGES PASS] {jsonl}")

    sem     = asyncio.Semaphore(concurrency)
    qa_buf: list[dict] = []
    stats   = {"ok": 0, "skip": 0, "fail": 0, "qa": 0}
    t_start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:

        async def train_one_image(img_path: str, cap: str) -> None:
            loop = asyncio.get_running_loop()
            b64  = await loop.run_in_executor(None, resize_image_b64, img_path)
            if b64 is None:
                stats["skip"] += 1
                return
            async with sem:
                ok = await media_train_image(client, node_url, b64, cap)
            if ok:
                stats["ok"] += 1
            else:
                stats["fail"] += 1

        async def flush_qa(pairs: list[dict]) -> None:
            n = await qa_ingest_batch(client, node_url, pairs)
            stats["qa"] += n

        tasks: list[asyncio.Task] = []
        count = 0

        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                if limit and count >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                img_path   = rec.get("file", "")
                captions   = rec.get("captions", [])
                categories = rec.get("categories", [])
                img_id     = rec.get("image_id", count)

                cap_text = captions[0] if captions else ""
                if categories:
                    cap_text += "  Objects: " + ", ".join(categories) + "."

                if cap_text.strip():
                    tasks.append(asyncio.create_task(train_one_image(img_path, cap_text)))

                for cat in categories:
                    qa_buf.append({
                        "qa_id":         str(uuid.uuid4()),
                        "question":      f"What is a {cat}?",
                        "answer":        captions[0] if captions else f"A {cat} is a real-world object.",
                        "book_id":       f"coco_{img_id}",
                        "page_index":    0,
                        "confidence":    0.70,
                        "evidence":      cat,
                        "review_status": "foundation",
                    })

                while len(qa_buf) >= batch_size:
                    batch = qa_buf[:batch_size]
                    del qa_buf[:batch_size]
                    tasks.append(asyncio.create_task(flush_qa(batch)))

                count += 1

                if len(tasks) >= concurrency * 3:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

        if qa_buf:
            tasks.append(asyncio.create_task(flush_qa(qa_buf)))
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - t_start
    print(f"\n[IMAGES PASS] Done in {elapsed:.0f}s")
    print(f"  Images trained: {stats['ok']:,} ok  {stats['skip']:,} skip  {stats['fail']:,} failed")
    print(f"  QA pairs ingested: {stats['qa']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def checkpoint(node_url: str) -> None:
    """Force-save both the neuro pool and QA store to disk."""
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{node_url}/neuro/checkpoint")
            if r.status_code == 200:
                d = r.json()
                print(f"  Checkpoint saved: pool={d.get('pool_path','?')}  qa={d.get('qa_path','?')}")
            else:
                print(f"  Checkpoint warning: HTTP {r.status_code}")
    except Exception as e:
        print(f"  Checkpoint failed: {e}")


async def main_async(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)

    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{args.node}/health", timeout=5)
            info = r.json()
            print(f"Node: {info.get('node_id','?')}  status={info.get('status')}  uptime={info.get('uptime_secs')}s")
    except Exception as e:
        sys.exit(f"Node not reachable at {args.node}: {e}")

    if args.mode in ("concepts", "all"):
        await run_concept_pass(args.node, data_dir, args.limit,
                               args.concurrency, args.batch_size)
        await checkpoint(args.node)

    if args.mode in ("text", "all"):
        await run_text_pass(args.node, data_dir, args.limit,
                            args.concurrency, args.batch_size)
        await checkpoint(args.node)

    if args.mode in ("images", "all"):
        await run_images_pass(args.node, data_dir, args.limit,
                              args.concurrency, args.batch_size)
        await checkpoint(args.node)

    print("\nFoundation training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundation training pipeline (async)")
    parser.add_argument("--node",        default=NODE_URL)
    parser.add_argument("--pass",        dest="mode",
                        choices=["concepts", "text", "images", "all"], default="concepts")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max articles/images (default: all)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help="Max concurrent HTTP requests (default: 40)")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--data-dir",    default="data/foundation")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
