#!/usr/bin/env python3
"""scripts/sensors/toddler_multimodal_v2.py — expanded toddler multimodal trainer.

Compared to the v1 trainer (6 synthetic colored shapes):
  * 120+ concrete toddler nouns (compiled from publicly available
    toddler vocabulary references — MacArthur-Bates CDI Words & Sentences,
    Dolch nouns list, common ESL beginner word lists)
  * Real photographs sourced from Wikimedia Commons (CC-licensed) instead
    of synthetic colored shapes — gives the image-pixels pool real
    photographic atom distributions to bind cross-modally
  * Polly TTS audio for each word (same pipeline as v1)
  * Multi-pass training with shuffled order per epoch so cross-modal
    bindings emerge without cross-pair noise dominating
  * Per-modality probe + cross-modal recall scoring

This is the "verification floor" the system must clear: after training,
any one of {image, audio, text} for an object should fire predictions
for the other two pools.

Run:
  python scripts/sensors/toddler_multimodal_v2.py
  python scripts/sensors/toddler_multimodal_v2.py --probe-only
  python scripts/sensors/toddler_multimodal_v2.py --words 30 --repeats 5

Dependencies: boto3 (Polly), pillow, requests
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ── Vocabulary: 120 concrete toddler nouns ────────────────────────────────
#
# Compiled from publicly-available toddler vocabulary references:
#  * MacArthur-Bates CDI Words and Sentences inventory (open data set,
#    Stanford Wordbank: http://wordbank.stanford.edu)
#  * Dolch nouns list (public domain pedagogical reference)
#  * Common ESL beginner word lists
#
# All concrete, image-able, with unambiguous Wikipedia page mappings.

VOCAB: list[tuple[str, str, str]] = [
    # word, wikipedia_page_title (English), category
    # ── Animals ──
    ("dog",       "Dog",         "animal"),
    ("cat",       "Cat",         "animal"),
    ("fish",      "Fish",        "animal"),
    ("bird",      "Bird",        "animal"),
    ("cow",       "Cattle",      "animal"),
    ("horse",     "Horse",       "animal"),
    ("sheep",     "Sheep",       "animal"),
    ("pig",       "Pig",         "animal"),
    ("duck",      "Duck",        "animal"),
    ("chicken",   "Chicken",     "animal"),
    ("lion",      "Lion",        "animal"),
    ("tiger",     "Tiger",       "animal"),
    ("bear",      "Bear",        "animal"),
    ("elephant",  "Elephant",    "animal"),
    ("monkey",    "Monkey",      "animal"),
    ("rabbit",    "Rabbit",      "animal"),
    ("mouse",     "Mouse",       "animal"),
    ("frog",      "Frog",        "animal"),
    ("butterfly", "Butterfly",   "animal"),
    ("snake",     "Snake",       "animal"),
    ("turtle",    "Turtle",      "animal"),
    ("owl",       "Owl",         "animal"),
    ("bee",       "Bee",         "animal"),
    ("ant",       "Ant",         "animal"),
    ("giraffe",   "Giraffe",     "animal"),

    # ── Food ──
    ("apple",     "Apple",       "food"),
    ("banana",    "Banana",      "food"),
    ("bread",     "Bread",       "food"),
    ("milk",      "Milk",        "food"),
    ("cheese",    "Cheese",      "food"),
    ("egg",       "Egg",         "food"),
    ("juice",     "Juice",       "food"),
    ("water",     "Water",       "food"),
    ("cookie",    "Biscuit",     "food"),
    ("cake",      "Cake",        "food"),
    ("pizza",     "Pizza",       "food"),
    ("orange",    "Orange_(fruit)", "food"),
    ("grape",     "Grape",       "food"),
    ("strawberry","Strawberry",  "food"),
    ("carrot",    "Carrot",      "food"),
    ("potato",    "Potato",      "food"),
    ("tomato",    "Tomato",      "food"),
    ("rice",      "Rice",        "food"),
    ("pasta",     "Pasta",       "food"),
    ("soup",      "Soup",        "food"),
    ("honey",     "Honey",       "food"),
    ("butter",    "Butter",      "food"),

    # ── Body parts ──
    ("eye",       "Human_eye",   "body"),
    ("ear",       "Ear",         "body"),
    ("nose",      "Human_nose",  "body"),
    ("mouth",     "Mouth",       "body"),
    ("hand",      "Hand",        "body"),
    ("foot",      "Foot",        "body"),
    ("hair",      "Hair",        "body"),
    ("tooth",     "Tooth",       "body"),
    ("finger",    "Finger",      "body"),

    # ── Clothes ──
    ("shirt",     "Shirt",       "clothes"),
    ("pants",     "Trousers",    "clothes"),
    ("shoe",      "Shoe",        "clothes"),
    ("hat",       "Hat",         "clothes"),
    ("coat",      "Coat_(clothing)", "clothes"),
    ("sock",      "Sock",        "clothes"),
    ("glove",     "Glove",       "clothes"),
    ("scarf",     "Scarf",       "clothes"),

    # ── Vehicles ──
    ("car",       "Car",         "vehicle"),
    ("truck",     "Truck",       "vehicle"),
    ("train",     "Train",       "vehicle"),
    ("plane",     "Airplane",    "vehicle"),
    ("boat",      "Boat",        "vehicle"),
    ("bike",      "Bicycle",     "vehicle"),
    ("bus",       "Bus",         "vehicle"),
    ("ship",      "Ship",        "vehicle"),
    ("rocket",    "Rocket",      "vehicle"),

    # ── Toys / household objects ──
    ("ball",      "Ball",        "toy"),
    ("book",      "Book",        "object"),
    ("cup",       "Cup",         "object"),
    ("chair",     "Chair",       "object"),
    ("table",     "Table_(furniture)", "object"),
    ("bed",       "Bed",         "object"),
    ("clock",     "Clock",       "object"),
    ("phone",     "Telephone",   "object"),
    ("key",       "Key_(lock)",  "object"),
    ("door",      "Door",        "object"),
    ("window",    "Window",      "object"),
    ("lamp",      "Lamp_(lighting)", "object"),
    ("toy",       "Toy",         "toy"),
    ("doll",      "Doll",        "toy"),
    ("balloon",   "Balloon",     "toy"),
    ("kite",      "Kite",        "toy"),
    ("drum",      "Drum",        "toy"),

    # ── Nature ──
    ("tree",      "Tree",        "nature"),
    ("flower",    "Flower",      "nature"),
    ("sun",       "Sun",         "nature"),
    ("moon",      "Moon",        "nature"),
    ("star",      "Star",        "nature"),
    ("cloud",     "Cloud",       "nature"),
    ("leaf",      "Leaf",        "nature"),
    ("grass",     "Grass",       "nature"),
    ("river",     "River",       "nature"),
    ("mountain",  "Mountain",    "nature"),
    ("beach",     "Beach",       "nature"),
    ("rain",      "Rain",        "nature"),
    ("snow",      "Snow",        "nature"),

    # ── Colors (as image-able objects with prominent color) ──
    ("red",       "Red",         "color"),
    ("blue",      "Blue",        "color"),
    ("green",     "Green",       "color"),
    ("yellow",    "Yellow",      "color"),
    ("black",     "Black",       "color"),
    ("white",     "White",       "color"),
    ("orange_color", "Orange_(colour)", "color"),
    ("purple",    "Purple",      "color"),
    ("pink",      "Pink",        "color"),
    ("brown",     "Brown",       "color"),

    # ── Buildings / places ──
    ("house",     "House",       "place"),
    ("school",    "School",      "place"),
    ("hospital",  "Hospital",    "place"),
    ("farm",      "Farm",        "place"),
    ("park",      "Park",        "place"),
    ("zoo",       "Zoo",         "place"),
    ("shop",      "Shop",        "place"),
    ("bridge",    "Bridge",      "place"),

    # ── Tools / utensils ──
    ("spoon",     "Spoon",       "utensil"),
    ("fork",      "Fork",        "utensil"),
    ("knife",     "Knife",       "utensil"),
    ("pen",       "Pen",         "tool"),
    ("pencil",    "Pencil",      "tool"),
    ("brush",     "Brush",       "tool"),
    ("scissors",  "Scissors",    "tool"),
    ("hammer",    "Hammer",      "tool"),
    ("umbrella",  "Umbrella",    "tool"),
    ("bag",       "Bag",         "object"),
    ("box",       "Box",         "object"),
    ("plate",     "Plate",       "utensil"),
    ("bowl",      "Bowl",        "utensil"),
]

# ── Wikipedia / Wikimedia image fetch ─────────────────────────────────────

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "W1z4rDV1510n-Toddler-Trainer/1.0 (contact: c4rr13rX@gmail.com)"


def fetch_wikipedia_image(page_title: str, max_px: int = 256) -> bytes | None:
    """Fetch a single representative image for a Wikipedia article.
    Uses the MediaWiki API's `pageimages` extension to get a thumbnail.
    Returns JPEG bytes (or None on failure).  The thumbnail is a
    derivative work — Wikimedia Commons CC-licensing applies to the
    underlying file; usage in this trainer is research/educational."""
    params = {
        "action":      "query",
        "prop":        "pageimages",
        "format":      "json",
        "piprop":      "thumbnail|original",
        "pithumbsize": str(max_px),
        "titles":      page_title,
        "redirects":   "1",
    }
    qs = urllib.parse.urlencode(params)
    url = f"{WIKI_API}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            doc = json.load(r)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  [wiki] error fetching {page_title!r}: {e}", flush=True)
        return None
    pages = doc.get("query", {}).get("pages", {})
    for _, page in pages.items():
        thumb = page.get("thumbnail")
        if not thumb:
            continue
        src = thumb.get("source")
        if not src:
            continue
        try:
            req2 = urllib.request.Request(src, headers={"User-Agent": UA})
            with urllib.request.urlopen(req2, timeout=20) as r2:
                return r2.read()
        except urllib.error.URLError as e:
            print(f"  [wiki] error downloading {src!r}: {e}", flush=True)
            return None
    return None


# ── Polly TTS (same as v1, kept inline so this script stands alone) ───────


def synthesize_polly_wav(text: str, voice: str = "Joanna",
                         rate: int = 16000) -> bytes:
    """Call AWS Polly and return WAV bytes."""
    import boto3
    import wave
    client = boto3.client("polly")
    resp = client.synthesize_speech(
        OutputFormat="pcm",
        SampleRate=str(rate),
        VoiceId=voice,
        Text=text,
    )
    pcm = resp["AudioStream"].read()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── HTTP ──────────────────────────────────────────────────────────────────


def post_json(url: str, body: dict, timeout: float = 60.0) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ── Cache management ──────────────────────────────────────────────────────


def ensure_assets(cache_dir: Path, vocab: list[tuple[str, str, str]],
                  *, fetch_polly: bool, verbose: bool) -> list[tuple[str, str]]:
    """For every vocab entry, make sure cache_dir has a {word}.jpg and
    {word}.wav.  Returns the list of (word, category) tuples that
    successfully ended up with BOTH cached assets."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ready: list[tuple[str, str]] = []
    for word, wiki_title, category in vocab:
        img_path = cache_dir / f"{word}.jpg"
        wav_path = cache_dir / f"{word}.wav"

        if not img_path.exists():
            if verbose:
                print(f"  [fetch] image for {word!r} (wiki: {wiki_title})", flush=True)
            data = fetch_wikipedia_image(wiki_title)
            if not data:
                print(f"  [skip] no image for {word!r}", flush=True)
                continue
            img_path.write_bytes(data)
            time.sleep(0.2)  # be polite to Wikipedia

        if not wav_path.exists():
            if not fetch_polly:
                print(f"  [skip] no polly audio for {word!r} (--no-polly set)",
                      flush=True)
                continue
            if verbose:
                print(f"  [polly] audio for {word!r}", flush=True)
            try:
                wav_path.write_bytes(synthesize_polly_wav(f"this is a {word}"))
            except Exception as e:
                print(f"  [skip] polly failed for {word!r}: {e}", flush=True)
                continue

        ready.append((word, category))
    return ready


# ── Training + probing ────────────────────────────────────────────────────


def shuffled(items: list, seed: int) -> list:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def train_triples(node_url: str, ready: list[tuple[str, str]],
                  cache_dir: Path, repeats: int, lr: float,
                  verbose: bool) -> dict:
    """Train each (image, audio, text) triple `repeats` times, shuffled
    per epoch.  Reports a summary."""
    print(f"\n[train] {len(ready)} objects x {repeats} epochs", flush=True)
    fails = 0
    for epoch in range(repeats):
        for word, _category in shuffled(ready, seed=epoch * 1009 + 17):
            img_b64 = base64.b64encode((cache_dir / f"{word}.jpg").read_bytes()).decode("ascii")
            wav_b64 = base64.b64encode((cache_dir / f"{word}.wav").read_bytes()).decode("ascii")
            try:
                r = post_json(f"{node_url}/sensor/observe_triple", {
                    "text":      word,
                    "image_b64": img_b64,
                    "audio_b64": wav_b64,
                    "lr":        lr,
                })
                if verbose:
                    print(f"  ep{epoch+1}/{repeats} {word}: "
                          f"img_labels={r.get('img_labels','?')} "
                          f"aud_labels={r.get('aud_labels','?')}",
                          flush=True)
            except Exception as e:
                fails += 1
                if verbose:
                    print(f"  ep{epoch+1} {word} FAILED: {e}", flush=True)
        print(f"  [epoch {epoch+1}/{repeats}] done", flush=True)
    return {"trained_objects": len(ready), "epochs": repeats, "fails": fails}


def probe_directions(node_url: str, ready: list[tuple[str, str]],
                     cache_dir: Path) -> dict:
    """For each object, query each modality and report what cross-pool
    predictions came back.  Score is the fraction of objects for which
    EACH direction produced any prediction in the other modalities."""
    results = {"image": 0, "audio": 0, "text": 0, "total": len(ready)}
    for word, _category in ready:
        img_b64 = base64.b64encode((cache_dir / f"{word}.jpg").read_bytes()).decode("ascii")
        wav_b64 = base64.b64encode((cache_dir / f"{word}.wav").read_bytes()).decode("ascii")

        try:
            r = post_json(f"{node_url}/sensor/observe", {
                "kind": "image", "bytes_b64": img_b64,
            })
            if r.get("predictions"):
                results["image"] += 1
        except Exception:
            pass

        try:
            r = post_json(f"{node_url}/sensor/observe", {
                "kind": "audio", "bytes_b64": wav_b64,
            })
            if r.get("predictions"):
                results["audio"] += 1
        except Exception:
            pass

        try:
            r = post_json(f"{node_url}/chat", {"text": word})
            if r.get("predictions"):
                results["text"] += 1
        except Exception:
            pass

    return results


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--node",    default="http://127.0.0.1:8090")
    p.add_argument("--words",   type=int, default=len(VOCAB),
                   help=f"limit to first N vocab entries (default {len(VOCAB)})")
    p.add_argument("--repeats", type=int, default=8,
                   help="training epochs (default 8)")
    p.add_argument("--lr",      type=float, default=2.0)
    p.add_argument("--cache",   default="data/multimodal_toddler_v2",
                   help="cache directory for images + Polly WAVs")
    p.add_argument("--probe-only", action="store_true",
                   help="skip training, just probe with cached assets")
    p.add_argument("--no-polly", action="store_true",
                   help="skip Polly audio synthesis (use cached only)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cache = Path(args.cache).resolve()
    vocab = VOCAB[: args.words]
    print(f"[ensure] {len(vocab)} vocab entries; cache at {cache}", flush=True)
    ready = ensure_assets(cache, vocab,
                          fetch_polly=not args.no_polly,
                          verbose=args.verbose)
    print(f"[ready] {len(ready)}/{len(vocab)} objects have both image + audio",
          flush=True)
    if not ready:
        print("[error] no assets available — cannot train", flush=True)
        return 1

    if not args.probe_only:
        summary = train_triples(args.node, ready, cache,
                                args.repeats, args.lr, args.verbose)
        print(f"[train summary] {summary}", flush=True)

    print(f"\n[probe] testing bidirectional cross-modal recall...", flush=True)
    res = probe_directions(args.node, ready, cache)
    n = max(res["total"], 1)
    print(f"\n=== CROSS-MODAL RECALL ===")
    print(f"  image -> others:   {res['image']}/{n}  ({100*res['image']/n:.1f}%)")
    print(f"  audio -> others:   {res['audio']}/{n}  ({100*res['audio']/n:.1f}%)")
    print(f"  text  -> others:   {res['text']}/{n}  ({100*res['text']/n:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
