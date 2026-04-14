#!/usr/bin/env python3
"""
train_foundation.py — Feed English foundation corpus into the neural node
=========================================================================
Two-pass foundation training pipeline:

  Pass 1 — Text (Simple English Wikipedia)
    Reads data/foundation/simple_wiki_articles.jsonl.
    Breaks each article into paragraphs and POSTs to /media/train as
    modality="text".  Also extracts clean Q&A pairs and batches them
    to /qa/ingest.

  Pass 2 — Images (COCO 2017 val)
    Reads data/foundation/coco_val_index.jsonl.
    For each image, encodes it as base64 JPEG and POSTs to /media/train
    as modality="page" with caption text attached.  Also ingests each
    "What is this? → <category>" pair to /qa/ingest.

Run BOTH passes by default.  Use --pass text|images to run one only.

Usage:
  python scripts/train_foundation.py [--node http://127.0.0.1:8090]
                                     [--pass text|images|both]
                                     [--limit N]
                                     [--batch-size 50]
                                     [--workers 4]

Dependencies:
  pip install httpx Pillow tqdm
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import re
import sys
import threading
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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # graceful degradation

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

NODE_URL     = "http://127.0.0.1:8090"
BATCH_SIZE   = 50        # QA pairs per /qa/ingest request
WORKERS      = 4         # parallel HTTP workers
IMAGE_MAX_PX = 640       # resize longest edge to this before base64 encoding

# Min paragraph length for text training
MIN_PARA_LEN = 60

# QA extraction (simple "X is Y" patterns — same rules as train_k12.py)
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

        # Subject filters
        if subj.lower() in _SKIP_SUBJECTS:
            continue
        if len(subj) > 60 or len(subj.split()) > 6:
            continue
        if _CONTEXT_SUBJ_RE.search(subj):
            continue

        # Answer quality gates
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

        question = f"What is {subj.lower()}?"
        pairs.append({
            "qa_id":         str(uuid.uuid4()),
            "question":      question,
            "answer":        defn,
            "book_id":       source_id,
            "page_index":    0,
            "confidence":    0.75,
            "evidence":      subj,
            "review_status": "foundation",
        })

    return pairs


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def media_train_text(client: httpx.Client, node_url: str, paragraph: str) -> bool:
    """POST a plain text paragraph to /media/train."""
    try:
        r = client.post(
            f"{node_url}/media/train",
            json={"modality": "text", "text": paragraph, "lr_scale": 1.0},
            timeout=20,
        )
        return r.status_code == 200
    except Exception:
        return False


def media_train_image(client: httpx.Client, node_url: str, img_b64: str,
                      caption: str) -> bool:
    """POST image + caption to /media/train as page modality."""
    try:
        r = client.post(
            f"{node_url}/media/train",
            json={
                "modality":  "page",
                "data_b64":  img_b64,
                "text":      caption,
                "lr_scale":  1.0,
            },
            timeout=30,
        )
        return r.status_code == 200
    except Exception:
        return False


def qa_ingest_batch(client: httpx.Client, node_url: str,
                    batch: list[dict]) -> int:
    """POST a batch of QA pairs to /qa/ingest.  Returns number ingested."""
    try:
        r = client.post(
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
    """Load an image, resize so longest edge ≤ max_px, return base64 JPEG."""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pass 1 — Text
# ---------------------------------------------------------------------------

def run_text_pass(node_url: str, data_dir: Path, limit: int | None,
                  workers: int, batch_size: int) -> None:
    jsonl = data_dir / "simple_wiki_articles.jsonl"
    if not jsonl.exists():
        print(f"[text] {jsonl} not found — run fetch_simple_wikipedia.py first")
        return

    print(f"\n[TEXT PASS] Reading {jsonl}")
    total_articles = sum(1 for _ in open(jsonl, encoding="utf-8"))
    print(f"  {total_articles:,} articles in corpus")

    # Producer: yields (text_paragraph, source_id, qa_pairs) tuples
    para_q:  queue.Queue = queue.Queue(maxsize=workers * 4)
    qa_buf:  list[dict]  = []
    qa_lock  = threading.Lock()
    stop_evt = threading.Event()

    stats = {"text_ok": 0, "text_fail": 0, "qa_ingested": 0}
    stats_lock = threading.Lock()

    def producer() -> None:
        count = 0
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                if limit and count >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                article_text = rec.get("text", "")
                source_id    = rec.get("id", "")

                # Split into paragraphs
                paragraphs = [p.strip() for p in article_text.split("\n\n")
                              if len(p.strip()) >= MIN_PARA_LEN]
                for para in paragraphs:
                    para_q.put(("text", para, source_id))

                # Extract QA pairs from full article
                pairs = extract_qa_from_text(article_text, source_id)
                if pairs:
                    with qa_lock:
                        qa_buf.extend(pairs)

                count += 1

        # Poison pills
        for _ in range(workers):
            para_q.put(None)

    def worker(wid: int) -> None:
        client = httpx.Client()
        while True:
            item = para_q.get()
            if item is None:
                break
            kind, text, _source = item
            ok = media_train_text(client, node_url, text)
            with stats_lock:
                if ok:
                    stats["text_ok"] += 1
                else:
                    stats["text_fail"] += 1

    def qa_flusher() -> None:
        """Periodically flush accumulated QA pairs."""
        client = httpx.Client()
        while not stop_evt.wait(timeout=5.0):
            with qa_lock:
                if len(qa_buf) >= batch_size:
                    batch = qa_buf[:batch_size]
                    del qa_buf[:batch_size]
                else:
                    continue
            n = qa_ingest_batch(client, node_url, batch)
            with stats_lock:
                stats["qa_ingested"] += n
        # Final flush
        with qa_lock:
            remaining = list(qa_buf)
            qa_buf.clear()
        if remaining:
            for i in range(0, len(remaining), batch_size):
                n = qa_ingest_batch(client, node_url, remaining[i:i+batch_size])
                with stats_lock:
                    stats["qa_ingested"] += n

    prod_thread  = threading.Thread(target=producer, daemon=True)
    worker_threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                      for i in range(workers)]
    qa_thread    = threading.Thread(target=qa_flusher, daemon=True)

    prod_thread.start()
    for t in worker_threads:
        t.start()
    qa_thread.start()

    # Progress monitor
    prod_thread.join()
    for t in worker_threads:
        t.join()
    stop_evt.set()
    qa_thread.join()

    s = stats
    print(f"\n[TEXT PASS] Done.")
    print(f"  Paragraphs trained: {s['text_ok']:,} ok  {s['text_fail']:,} failed")
    print(f"  QA pairs ingested : {s['qa_ingested']:,}")


# ---------------------------------------------------------------------------
# Pass 2 — Images
# ---------------------------------------------------------------------------

def run_images_pass(node_url: str, data_dir: Path, limit: int | None,
                    workers: int, batch_size: int) -> None:
    jsonl = data_dir / "coco_val_index.jsonl"
    if not jsonl.exists():
        print(f"[images] {jsonl} not found — run fetch_coco_objects.py first")
        return

    print(f"\n[IMAGES PASS] Reading {jsonl}")

    img_q:   queue.Queue = queue.Queue(maxsize=workers * 2)
    qa_buf:  list[dict]  = []
    qa_lock  = threading.Lock()
    stop_evt = threading.Event()

    stats = {"img_ok": 0, "img_skip": 0, "img_fail": 0, "qa_ingested": 0}
    stats_lock = threading.Lock()

    def producer() -> None:
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

                # Build caption text: first caption + category hint
                cap_text = captions[0] if captions else ""
                if categories:
                    cap_text += "  Objects: " + ", ".join(categories) + "."

                if cap_text.strip():
                    img_q.put((img_path, cap_text, img_id, categories))

                # Also build "What is this? → <category>" QA pairs
                for cat in categories:
                    qa_buf_item = {
                        "qa_id":         str(uuid.uuid4()),
                        "question":      f"What is a {cat}?",
                        "answer":        captions[0] if captions else f"A {cat} is a real-world object.",
                        "book_id":       f"coco_{img_id}",
                        "page_index":    0,
                        "confidence":    0.70,
                        "evidence":      cat,
                        "review_status": "foundation",
                    }
                    with qa_lock:
                        qa_buf.append(qa_buf_item)

                count += 1

        for _ in range(workers):
            img_q.put(None)

    def worker(wid: int) -> None:
        client = httpx.Client()
        while True:
            item = img_q.get()
            if item is None:
                break
            img_path, cap_text, img_id, _cats = item
            b64 = resize_image_b64(img_path)
            if b64 is None:
                with stats_lock:
                    stats["img_skip"] += 1
                continue
            ok = media_train_image(client, node_url, b64, cap_text)
            with stats_lock:
                if ok:
                    stats["img_ok"] += 1
                else:
                    stats["img_fail"] += 1

    def qa_flusher() -> None:
        client = httpx.Client()
        while not stop_evt.wait(timeout=5.0):
            with qa_lock:
                if len(qa_buf) >= batch_size:
                    batch = qa_buf[:batch_size]
                    del qa_buf[:batch_size]
                else:
                    continue
            n = qa_ingest_batch(client, node_url, batch)
            with stats_lock:
                stats["qa_ingested"] += n
        with qa_lock:
            remaining = list(qa_buf)
            qa_buf.clear()
        if remaining:
            for i in range(0, len(remaining), batch_size):
                n = qa_ingest_batch(client, node_url, remaining[i:i+batch_size])
                with stats_lock:
                    stats["qa_ingested"] += n

    prod_thread    = threading.Thread(target=producer, daemon=True)
    worker_threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                      for i in range(workers)]
    qa_thread      = threading.Thread(target=qa_flusher, daemon=True)

    prod_thread.start()
    for t in worker_threads:
        t.start()
    qa_thread.start()

    prod_thread.join()
    for t in worker_threads:
        t.join()
    stop_evt.set()
    qa_thread.join()

    s = stats
    print(f"\n[IMAGES PASS] Done.")
    print(f"  Images trained: {s['img_ok']:,} ok  {s['img_skip']:,} skipped  {s['img_fail']:,} failed")
    print(f"  QA pairs ingested: {s['qa_ingested']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Foundation training pipeline")
    parser.add_argument("--node",       default=NODE_URL,        help="Node URL")
    parser.add_argument("--pass",       dest="mode",
                        choices=["text", "images", "both"],       default="both")
    parser.add_argument("--limit",      type=int, default=None,  help="Max articles/images")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers",    type=int, default=WORKERS)
    parser.add_argument("--data-dir",   default="data/foundation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Health check
    try:
        r = httpx.get(f"{args.node}/health", timeout=5)
        info = r.json()
        print(f"Node: {info.get('node_id','?')}  status={info.get('status')}  uptime={info.get('uptime_secs')}s")
    except Exception as e:
        sys.exit(f"Node not reachable at {args.node}: {e}")

    if args.mode in ("text", "both"):
        run_text_pass(args.node, data_dir, args.limit, args.workers, args.batch_size)

    if args.mode in ("images", "both"):
        run_images_pass(args.node, data_dir, args.limit, args.workers, args.batch_size)

    print("\nFoundation training complete.")


if __name__ == "__main__":
    main()
