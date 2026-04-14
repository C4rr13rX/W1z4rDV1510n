#!/usr/bin/env python3
"""
train_foundation.py — Feed English foundation corpus into the neural node
=========================================================================
Two-pass foundation training pipeline:

  Pass 1 — Text (Simple English Wikipedia)
    Reads data/foundation/simple_wiki_articles.jsonl.
    Sends each article as a single /media/train text call.
    Also extracts clean Q&A pairs and batches them to /qa/ingest.

  Pass 2 — Images (COCO 2017 val)
    Reads data/foundation/coco_val_index.jsonl.
    For each image, encodes it as base64 JPEG and POSTs to /media/train
    as modality="page" with caption text attached.  Also ingests each
    "What is this? -> <category>" pair to /qa/ingest.

Uses asyncio + httpx.AsyncClient for full concurrent throughput
(50 concurrent requests instead of GIL-limited threading).

Usage:
  python scripts/train_foundation.py [--node http://127.0.0.1:8090]
                                     [--pass text|images|both]
                                     [--limit N]
                                     [--concurrency 40]
                                     [--batch-size 50]

Dependencies:
  pip install httpx[http2] Pillow anyio
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

async def main_async(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)

    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{args.node}/health", timeout=5)
            info = r.json()
            print(f"Node: {info.get('node_id','?')}  status={info.get('status')}  uptime={info.get('uptime_secs')}s")
    except Exception as e:
        sys.exit(f"Node not reachable at {args.node}: {e}")

    if args.mode in ("text", "both"):
        await run_text_pass(args.node, data_dir, args.limit,
                            args.concurrency, args.batch_size)

    if args.mode in ("images", "both"):
        await run_images_pass(args.node, data_dir, args.limit,
                              args.concurrency, args.batch_size)

    print("\nFoundation training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundation training pipeline (async)")
    parser.add_argument("--node",        default=NODE_URL)
    parser.add_argument("--pass",        dest="mode",
                        choices=["text", "images", "both"], default="both")
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
