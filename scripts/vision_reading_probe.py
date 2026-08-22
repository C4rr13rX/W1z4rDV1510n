#!/usr/bin/env python3
"""Measure whether the brain can learn to read a word at an unseen scale.

The point is not OCR for its own sake. A rendered page crop is a case where
the ground truth is free and exact, so scale invariance can be MEASURED
rather than asserted: train a word at some zoom levels, then ask for it at a
zoom it never saw. A model that has learned the glyph answers; one that has
memorised a pixel pattern does not.

That is the same property a camera needs to recognise one person near and
far, tested here on material where every label is already known.

Reports two numbers that mean different things:

  seen    accuracy at the zoom levels used for training -- memorisation
  unseen  accuracy at a held-out zoom -- invariance

`unseen` is the number that matters. `seen` is the control: if it is low the
run says nothing about invariance, only that training did not take.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Run from anywhere: the ingest modules live at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_ZOOMS = (0.6, 1.0, 2.4, 3.2)
#: Deliberately interior to the training range, so success cannot be
#: extrapolation luck at an endpoint.
HELDOUT_ZOOM = 1.6

POOL_TEXT = 1
POOL_ACTION = 4
POOL_INTENT = 12


def request(endpoint: str, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


def b64url(value: bytes | str) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def crops_for(pdf_path: Path, limit: int):
    """(word, {zoom: png_bytes}) for the first `limit` distinct words."""
    import fitz

    from tools.training_standard.ingest.document_directory import (
        page_is_commercial_safe,
    )
    from tools.training_standard.ingest.page_images import (
        crop_to_png, readable_words,
    )

    document = fitz.open(pdf_path)
    try:
        seen: dict[str, dict[float, bytes]] = {}
        for index in range(document.page_count):
            page = document[index]
            if not page_is_commercial_safe(page.get_text()):
                continue
            for box, word in readable_words(page):
                if word in seen or len(seen) >= limit:
                    continue
                renders = {}
                for zoom in (*TRAIN_ZOOMS, HELDOUT_ZOOM):
                    png = crop_to_png(page, box, zoom)
                    if png is not None:
                        renders[zoom] = png
                if len(renders) == len(TRAIN_ZOOMS) + 1:
                    seen[word] = renders
            if len(seen) >= limit:
                break
        return list(seen.items())
    finally:
        document.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18095")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--words", type=int, default=12)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/vision_reading.json"))
    args = parser.parse_args()

    samples = crops_for(args.pdf, args.words)
    if not samples:
        print("no usable crops", file=sys.stderr)
        return 2
    print(f"{len(samples)} words x {len(TRAIN_ZOOMS)} train zooms "
          f"(held out {HELDOUT_ZOOM}x)", flush=True)

    if not args.no_train:
        for word, renders in samples:
            for zoom in TRAIN_ZOOMS:
                request(args.endpoint, "/brain/pretrain_binding", {"frames": [
                    {"pool_id": POOL_TEXT, "frame": b64url(renders[zoom])},
                    {"pool_id": POOL_INTENT, "frame": b64url("read this word")},
                    {"pool_id": POOL_ACTION, "frame": b64url(word)},
                ]})
        request(args.endpoint, "/brain/tick", {})

    results = []
    for word, renders in samples:
        for kind, zoom in (("seen", TRAIN_ZOOMS[1]), ("unseen", HELDOUT_ZOOM)):
            try:
                reply = request(args.endpoint, "/brain/chat",
                                {"text_b64": b64url(renders[zoom])})
            except urllib.error.HTTPError:
                # The chat route may not accept a binary frame; fall back to
                # the recall route so the probe still reports something real
                # rather than silently scoring zero.
                reply = request(args.endpoint, "/brain/recall",
                                {"pool_id": POOL_TEXT,
                                 "frame": b64url(renders[zoom])})
            got = str(reply.get("reply") or reply.get("answer") or "")
            results.append({"word": word, "kind": kind, "zoom": zoom,
                            "got": got[:64], "correct": got.strip() == word})

    summary = {
        kind: {
            "correct": sum(r["correct"] for r in results if r["kind"] == kind),
            "total": sum(r["kind"] == kind for r in results),
        }
        for kind in ("seen", "unseen")
    }
    report = {"summary": summary, "heldout_zoom": HELDOUT_ZOOM,
              "train_zooms": list(TRAIN_ZOOMS), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
