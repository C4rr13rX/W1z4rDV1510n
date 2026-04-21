#!/usr/bin/env python3
"""
train_foundation.py -- Feed English foundation corpus into the neural node
=========================================================================
Full-architecture training pipeline.  Uses every available API endpoint:

  /media/train_sequence  -- temporal STDP (image+text, text+context, Q->A)
  /equations/ingest      -- Environmental Equation Matrix
  /knowledge/ingest      -- structured knowledge documents
  /qa/ingest             -- Q&A store + internal STDP bridge
  /neuro/record_episode  -- episodic learning from confirmed Q->A pairs
  /neuro/checkpoint      -- pool persistence

Three passes:
  Pass 0 (CONCEPTS)  -- concept_dataset.jsonl (first-words -> kindergarten)
  Pass 1 (TEXT)      -- simple_wiki_articles.jsonl
  Pass 2 (IMAGES)    -- coco_val_index.jsonl

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
import json
import re
import sys
import time
import uuid
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing: pip install Pillow")

from neuro_client import NeuroClient, resize_image_b64, detect_discipline, make_spans

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NODE_URL     = "http://127.0.0.1:8090"
CONCURRENCY  = 40
BATCH_SIZE   = 50
IMAGE_MAX_PX = 512
MIN_TEXT_LEN = 60

# ---------------------------------------------------------------------------
# QA extraction from free text
# ---------------------------------------------------------------------------

_IS_RE = re.compile(
    r"^([A-Z][A-Za-z\s\-']{2,50})\s+(?:is|are|was|were|refers? to|means?)\s+(.{40,500})\.$",
    re.MULTILINE,
)
_SKIP_SUBJECTS = {
    "this", "that", "these", "those", "it", "they", "he", "she", "we", "you", "i",
    "a", "an", "the", "because", "when", "if", "although", "while", "since", "once",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "there", "here", "then", "now", "no", "not", "one", "some", "many", "most",
    "all", "each", "both", "several", "few", "any", "every", "either", "neither",
    "for", "with", "by", "from", "of", "in", "on", "at", "as", "to",
}
_CONTEXT_SUBJ_RE = re.compile(r"\b(of the|of a|of an|in the|in a|by the|for the)\b", re.I)
_MATH_ONLY_RE    = re.compile(r"^[\d\s+\-*/=.,()[\]\\<>%deg±x÷√pi]+$")


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


def _is_misconception_question(q: str) -> bool:
    lower = q.lower()
    return lower.startswith(("is it true", "is it correct", "is it a fact", "do people think"))


# ---------------------------------------------------------------------------
# Pass 0 -- Concept dataset
# ---------------------------------------------------------------------------

LEVEL_NAMES = {-1: "first-words", 0: "infant", 1: "toddler",
               2: "pre-K", 3: "kindergarten"}


async def run_concept_pass(node_url: str, data_dir: Path, limit: int | None,
                           concurrency: int, batch_size: int) -> None:
    jsonl = data_dir / "concept_dataset.jsonl"
    if not jsonl.exists():
        print(f"[concepts] {jsonl} not found -- run build_concept_dataset.py first")
        return

    records = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    records.sort(key=lambda r: (r.get("level", 0), -r.get("frequency", 0)))
    if limit:
        records = records[:limit]

    total = len(records)
    print(f"\n[CONCEPT PASS] {total:,} concepts -- full architecture training")
    print(f"  Endpoints: train_sequence | equations/ingest | knowledge/ingest | qa/ingest | record_episode")

    sem     = asyncio.Semaphore(concurrency)
    qa_buf: list[dict] = []
    corr_buf: list[dict] = []
    stats   = {"seq": 0, "img": 0, "eq": 0, "know": 0, "qa": 0, "ep": 0}
    t_start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        nc = NeuroClient(node_url, client)

        async def train_concept_record(rec: dict) -> None:
            level   = rec.get("level", 0)
            concept = rec.get("concept", "")
            defn    = rec.get("definition", "")
            images  = rec.get("images", [])
            qa_pairs = rec.get("qa_pairs", [])
            wiki_text = rec.get("wiki_text", "")

            full_text = defn
            if wiki_text and len(wiki_text) > len(defn):
                full_text = defn + "  " + wiki_text[:500]

            async with sem:
                # 1. Text temporal sequence (definition + wiki)
                if full_text.strip():
                    ok = await nc.train_text_temporal(
                        full_text,
                        context=f"[concept:{concept}] [level:{level}]",
                        lr=1.0,
                    )
                    if ok:
                        stats["seq"] += 1

                # 2. Each image as image+text+structural temporal sequence
                for img_path in images[:8]:
                    loop = asyncio.get_running_loop()
                    b64  = await loop.run_in_executor(
                        None, resize_image_b64, img_path, IMAGE_MAX_PX)
                    if b64:
                        ok = await nc.train_image_text_temporal(
                            b64, defn[:300],
                            structural=f"[concept:{concept}] [level:{level}] [visual]",
                            lr=1.0,
                        )
                        if ok:
                            stats["img"] += 1

                # 3. Equation matrix -- scientific text
                disc = detect_discipline(full_text)
                if disc:
                    n = await nc.ingest_equations(full_text, discipline=disc)
                    stats["eq"] += n

                # 4. Knowledge document
                if defn.strip():
                    await nc.ingest_knowledge(
                        title=concept,
                        body=full_text,
                        source="concept_dataset",
                        tags=[f"level:{level}"],
                    )
                    stats["know"] += 1

        tasks: list[asyncio.Task] = []
        current_level = None
        QA_REPEATS = 4

        for idx, rec in enumerate(records):
            level   = rec.get("level", 0)
            concept = rec.get("concept", "")
            qa_pairs = rec.get("qa_pairs", [])

            if level != current_level:
                current_level = level
                print(f"\n  --- Level {level}: {LEVEL_NAMES.get(level, str(level))} ---")

            tasks.append(asyncio.create_task(train_concept_record(rec)))

            is_compound = "-" in concept or " " in concept
            repeats = 1 if is_compound else QA_REPEATS

            for qa in qa_pairs:
                qa_with_source = dict(qa)
                qa_with_source["book_id"]    = f"concept_{concept}"
                qa_with_source["page_index"] = level + 1
                if _is_misconception_question(qa.get("question", "")):
                    corr_buf.append(qa_with_source)
                else:
                    for _ in range(repeats):
                        qa_buf.append(qa_with_source)

            # Flush QA buffers -- full pipeline: ingest + sequence + episode
            while len(qa_buf) >= batch_size:
                batch = qa_buf[:batch_size]
                del qa_buf[:batch_size]
                tasks.append(asyncio.create_task(
                    nc.ingest_qa_full(batch, pool="knowledge", record_episodes=True)))

            while len(corr_buf) >= batch_size:
                batch = corr_buf[:batch_size]
                del corr_buf[:batch_size]
                tasks.append(asyncio.create_task(
                    nc.ingest_qa_full(batch, pool="correction", record_episodes=True)))

            if len(tasks) >= concurrency * 4:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{idx+1}/{total}] {concept!r:<20}  "
                      f"seq={stats['seq']} img={stats['img']} eq={stats['eq']} "
                      f"know={stats['know']} qa={stats['qa']}  {rate:.1f}/s")

        if qa_buf:
            tasks.append(asyncio.create_task(
                nc.ingest_qa_full(qa_buf, pool="knowledge", record_episodes=True)))
        if corr_buf:
            tasks.append(asyncio.create_task(
                nc.ingest_qa_full(corr_buf, pool="correction", record_episodes=True)))
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - t_start
    print(f"\n[CONCEPT PASS] Done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Sequences  : {stats['seq']:,}")
    print(f"  Images     : {stats['img']:,}")
    print(f"  Equations  : {stats['eq']:,}")
    print(f"  Knowledge  : {stats['know']:,}")
    print(f"  QA ingested: {stats['qa']:,}")


# ---------------------------------------------------------------------------
# Pass 1 -- Simple English Wikipedia
# ---------------------------------------------------------------------------

async def run_text_pass(node_url: str, data_dir: Path, limit: int | None,
                        concurrency: int, batch_size: int) -> None:
    jsonl = data_dir / "simple_wiki_articles.jsonl"
    if not jsonl.exists():
        print(f"[text] {jsonl} not found -- run fetch_simple_wikipedia.py first")
        return

    print(f"\n[TEXT PASS] {jsonl}")
    print(f"  Endpoints: train_sequence | equations/ingest | qa/ingest | record_episode")

    sem      = asyncio.Semaphore(concurrency)
    qa_buf:  list[dict] = []
    stats    = {"seq": 0, "fail": 0, "eq": 0, "qa": 0}
    t_start  = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        nc = NeuroClient(node_url, client)

        async def train_article(text: str, source_id: str) -> None:
            async with sem:
                # Temporal text sequence (text + structural context)
                ok = await nc.train_text_temporal(
                    text[:2000],
                    context=f"[source:simple_wikipedia] [id:{source_id}]",
                    lr=1.0,
                )
                if ok:
                    stats["seq"] += 1
                else:
                    stats["fail"] += 1

                # Equation matrix for scientific articles
                disc = detect_discipline(text)
                if disc:
                    n = await nc.ingest_equations(text[:2000], discipline=disc)
                    stats["eq"] += n

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

                text      = rec.get("text", "").strip()
                source_id = rec.get("id", str(count))

                if len(text) >= MIN_TEXT_LEN:
                    tasks.append(asyncio.create_task(train_article(text, source_id)))

                    # Q&A extraction from text
                    pairs = extract_qa_from_text(text, source_id)
                    qa_buf.extend(pairs)

                while len(qa_buf) >= batch_size:
                    batch = qa_buf[:batch_size]
                    del qa_buf[:batch_size]
                    tasks.append(asyncio.create_task(
                        nc.ingest_qa_full(batch, pool="knowledge", record_episodes=True)))

                count += 1

                if len(tasks) >= concurrency * 3:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

                if count % 5000 == 0:
                    elapsed = time.time() - t_start
                    rate = stats["seq"] / elapsed if elapsed > 0 else 0
                    print(f"  {count:,} articles  seq={stats['seq']:,}  eq={stats['eq']:,}  "
                          f"qa={stats['qa']:,}  {rate:.1f}/s")

        if qa_buf:
            tasks.append(asyncio.create_task(
                nc.ingest_qa_full(qa_buf, pool="knowledge", record_episodes=True)))
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - t_start
    print(f"\n[TEXT PASS] Done in {elapsed:.0f}s")
    print(f"  Sequences  : {stats['seq']:,}  failures: {stats['fail']:,}")
    print(f"  Equations  : {stats['eq']:,}")
    print(f"  QA ingested: {stats['qa']:,}")


# ---------------------------------------------------------------------------
# Pass 2 -- COCO scene images
# ---------------------------------------------------------------------------

async def run_images_pass(node_url: str, data_dir: Path, limit: int | None,
                          concurrency: int, batch_size: int) -> None:
    jsonl = data_dir / "coco_val_index.jsonl"
    if not jsonl.exists():
        print(f"[images] {jsonl} not found -- run fetch_coco_objects.py first")
        return

    print(f"\n[IMAGES PASS] {jsonl}")
    print(f"  Endpoints: train_sequence (image+text) | qa/ingest | record_episode")

    sem     = asyncio.Semaphore(concurrency)
    qa_buf: list[dict] = []
    stats   = {"seq": 0, "skip": 0, "fail": 0, "qa": 0}
    t_start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        nc = NeuroClient(node_url, client)

        async def train_coco_image(img_path: str, caption: str, categories: list[str],
                                   img_id: int) -> None:
            loop = asyncio.get_running_loop()
            b64  = await loop.run_in_executor(None, resize_image_b64, img_path, IMAGE_MAX_PX)
            if b64 is None:
                stats["skip"] += 1
                return
            # Structural context: what objects are present
            structural = f"[scene] [objects:{','.join(categories[:5])}] [id:{img_id}]"
            async with sem:
                ok = await nc.train_image_text_temporal(
                    b64, caption, structural=structural, lr=1.0,
                )
            if ok:
                stats["seq"] += 1
            else:
                stats["fail"] += 1

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

                if cap_text.strip() and img_path:
                    tasks.append(asyncio.create_task(
                        train_coco_image(img_path, cap_text, categories, img_id)))

                # Q&A from scene captions + category labels
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
                    tasks.append(asyncio.create_task(
                        nc.ingest_qa_full(batch, pool="knowledge", record_episodes=True)))

                count += 1

                if len(tasks) >= concurrency * 3:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

        if qa_buf:
            tasks.append(asyncio.create_task(
                nc.ingest_qa_full(qa_buf, pool="knowledge", record_episodes=True)))
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - t_start
    print(f"\n[IMAGES PASS] Done in {elapsed:.0f}s")
    print(f"  Image sequences: {stats['seq']:,}  skip={stats['skip']:,}  fail={stats['fail']:,}")
    print(f"  QA ingested    : {stats['qa']:,}")


# ---------------------------------------------------------------------------
# Checkpoint + main
# ---------------------------------------------------------------------------

async def checkpoint(nc: NeuroClient) -> None:
    ok = await nc.checkpoint()
    print("  Checkpoint saved." if ok else "  Checkpoint warning: failed.")


async def main_async(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(f"{args.node}/health", timeout=5)
            info = r.json()
            print(f"Node: {info.get('node_id','?')}  status={info.get('status')}  "
                  f"uptime={info.get('uptime_secs')}s")
        except Exception as e:
            sys.exit(f"Node not reachable at {args.node}: {e}")

        nc = NeuroClient(args.node, client)

        if args.mode in ("concepts", "all"):
            await run_concept_pass(args.node, data_dir, args.limit,
                                   args.concurrency, args.batch_size)
            await checkpoint(nc)

        if args.mode in ("text", "all"):
            await run_text_pass(args.node, data_dir, args.limit,
                                args.concurrency, args.batch_size)
            await checkpoint(nc)

        if args.mode in ("images", "all"):
            await run_images_pass(args.node, data_dir, args.limit,
                                  args.concurrency, args.batch_size)
            await checkpoint(nc)

    print("\nFoundation training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundation training pipeline -- full architecture")
    parser.add_argument("--node",        default=NODE_URL)
    parser.add_argument("--pass",        dest="mode",
                        choices=["concepts", "text", "images", "all"], default="concepts")
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--data-dir",    default="data/foundation")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
