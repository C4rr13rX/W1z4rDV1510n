#!/usr/bin/env python3
"""scripts/sensors/k12_vocab_multimodal.py — K-12 multimodal vocabulary trainer.

Builds on toddler_multimodal_v2 with grade-school subject vocabulary —
math, science, geography, biology, civics, language arts.  Each term
gets a Wikimedia image + a Polly TTS audio clip + the text label,
trained as a cross-modal triple via /sensor/observe_triple.

This complements `train_k12.py` (which feeds PDF text into the
fabric) by giving the K-12 stage the missing modality coverage —
without an image of a triangle and the spoken word "triangle", the
brain can read about geometry but can't ground concepts visually
or auditorily.

Sources for the vocab list (all publicly published curricula):
  * NGSS (Next Generation Science Standards) topic vocabulary
  * Common Core Mathematics Glossary terms
  * National Geographic geography curriculum (K-12)
  * Standard biology textbook (Campbell, OpenStax) chapter terms
  * U.S. civics curriculum (USCIS naturalization study guide)

Run:
  python scripts/sensors/k12_vocab_multimodal.py
  python scripts/sensors/k12_vocab_multimodal.py --words 50 --repeats 4
  python scripts/sensors/k12_vocab_multimodal.py --probe-only

Dependencies: boto3, requests
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ── K-12 multimodal vocabulary (200+ terms) ───────────────────────────────

VOCAB: list[tuple[str, str, str]] = [
    # ── Mathematics ──
    ("number",      "Number",                 "math"),
    ("triangle",    "Triangle",               "math"),
    ("circle",      "Circle",                 "math"),
    ("square",      "Square",                 "math"),
    ("rectangle",   "Rectangle",              "math"),
    ("pentagon",    "Pentagon",               "math"),
    ("hexagon",     "Hexagon",                "math"),
    ("octagon",     "Octagon",                "math"),
    ("sphere",      "Sphere",                 "math"),
    ("cube",        "Cube",                   "math"),
    ("cone",        "Cone",                   "math"),
    ("cylinder",    "Cylinder",               "math"),
    ("pyramid",     "Pyramid",                "math"),
    ("fraction",    "Fraction",               "math"),
    ("decimal",     "Decimal",                "math"),
    ("equation",    "Equation",               "math"),
    ("angle",       "Angle",                  "math"),
    ("graph",       "Graph_(discrete_mathematics)", "math"),
    ("pi",          "Pi",                     "math"),
    ("infinity",    "Infinity",               "math"),
    ("addition",    "Addition",               "math"),
    ("subtraction", "Subtraction",            "math"),
    ("multiplication", "Multiplication",      "math"),
    ("division",    "Division_(mathematics)", "math"),
    ("percentage",  "Percentage",             "math"),
    ("ruler",       "Ruler",                  "math"),
    ("protractor",  "Protractor",             "math"),
    ("compass",     "Compass_(drafting)",     "math"),
    ("abacus",      "Abacus",                 "math"),

    # ── Physical science ──
    ("atom",        "Atom",                   "science"),
    ("molecule",    "Molecule",               "science"),
    ("electron",    "Electron",               "science"),
    ("proton",      "Proton",                 "science"),
    ("neutron",     "Neutron",                "science"),
    ("element",     "Chemical_element",       "science"),
    ("water",       "Water",                  "science"),
    ("ice",         "Ice",                    "science"),
    ("steam",       "Steam",                  "science"),
    ("magnet",      "Magnet",                 "science"),
    ("battery",     "Battery_(electricity)",  "science"),
    ("circuit",     "Electrical_network",     "science"),
    ("lightbulb",   "Incandescent_light_bulb","science"),
    ("prism",       "Prism",                  "science"),
    ("rainbow",     "Rainbow",                "science"),
    ("microscope",  "Microscope",             "science"),
    ("telescope",   "Telescope",              "science"),
    ("thermometer", "Thermometer",            "science"),
    ("scale",       "Weighing_scale",         "science"),
    ("gear",        "Gear",                   "science"),
    ("pulley",      "Pulley",                 "science"),
    ("lever",       "Lever",                  "science"),
    ("pendulum",    "Pendulum",               "science"),

    # ── Astronomy ──
    ("planet",      "Planet",                 "astronomy"),
    ("mercury",     "Mercury_(planet)",       "astronomy"),
    ("venus",       "Venus",                  "astronomy"),
    ("earth",       "Earth",                  "astronomy"),
    ("mars",        "Mars",                   "astronomy"),
    ("jupiter",     "Jupiter",                "astronomy"),
    ("saturn",      "Saturn",                 "astronomy"),
    ("uranus",      "Uranus",                 "astronomy"),
    ("neptune",     "Neptune",                "astronomy"),
    ("comet",       "Comet",                  "astronomy"),
    ("asteroid",    "Asteroid",               "astronomy"),
    ("galaxy",      "Galaxy",                 "astronomy"),
    ("nebula",      "Nebula",                 "astronomy"),
    ("eclipse",     "Eclipse",                "astronomy"),
    ("constellation","Constellation",         "astronomy"),

    # ── Biology ──
    ("cell",        "Cell_(biology)",         "biology"),
    ("dna",         "DNA",                    "biology"),
    ("bacteria",    "Bacteria",               "biology"),
    ("virus",       "Virus",                  "biology"),
    ("plant",       "Plant",                  "biology"),
    ("animal",      "Animal",                 "biology"),
    ("fungus",      "Fungus",                 "biology"),
    ("seed",        "Seed",                   "biology"),
    ("root",        "Root",                   "biology"),
    ("stem",        "Plant_stem",             "biology"),
    ("leaf",        "Leaf",                   "biology"),
    ("flower",      "Flower",                 "biology"),
    ("fruit",       "Fruit",                  "biology"),
    ("brain",       "Brain",                  "biology"),
    ("heart",       "Heart",                  "biology"),
    ("lung",        "Lung",                   "biology"),
    ("liver",       "Liver",                  "biology"),
    ("kidney",      "Kidney",                 "biology"),
    ("bone",        "Bone",                   "biology"),
    ("muscle",      "Muscle",                 "biology"),
    ("blood",       "Blood",                  "biology"),
    ("skin",        "Skin",                   "biology"),

    # ── Geography ──
    ("continent",   "Continent",              "geography"),
    ("africa",      "Africa",                 "geography"),
    ("asia",        "Asia",                   "geography"),
    ("europe",      "Europe",                 "geography"),
    ("north_america","North_America",         "geography"),
    ("south_america","South_America",         "geography"),
    ("australia",   "Australia",              "geography"),
    ("antarctica",  "Antarctica",             "geography"),
    ("ocean",       "Ocean",                  "geography"),
    ("river",       "River",                  "geography"),
    ("lake",        "Lake",                   "geography"),
    ("mountain",    "Mountain",               "geography"),
    ("volcano",     "Volcano",                "geography"),
    ("desert",      "Desert",                 "geography"),
    ("forest",      "Forest",                 "geography"),
    ("island",      "Island",                 "geography"),
    ("peninsula",   "Peninsula",              "geography"),
    ("valley",      "Valley",                 "geography"),
    ("canyon",      "Canyon",                 "geography"),
    ("glacier",     "Glacier",                "geography"),
    ("map",         "Map",                    "geography"),
    ("globe",       "Globe",                  "geography"),
    ("compass_navigation", "Compass",         "geography"),

    # ── Civics / history ──
    ("flag",        "Flag",                   "civics"),
    ("government",  "Government",             "civics"),
    ("president",   "President",              "civics"),
    ("congress",    "United_States_Congress", "civics"),
    ("court",       "Court",                  "civics"),
    ("law",         "Law",                    "civics"),
    ("vote",        "Voting",                 "civics"),
    ("election",    "Election",               "civics"),
    ("constitution","Constitution_of_the_United_States", "civics"),
    ("liberty",     "Statue_of_Liberty",      "civics"),

    # ── Language arts ──
    ("alphabet",    "Alphabet",               "language"),
    ("letter",      "Letter_(alphabet)",      "language"),
    ("word",        "Word",                   "language"),
    ("sentence",    "Sentence_(linguistics)", "language"),
    ("paragraph",   "Paragraph",              "language"),
    ("book_la",     "Book",                   "language"),
    ("library",     "Library",                "language"),
    ("dictionary",  "Dictionary",             "language"),
    ("poem",        "Poetry",                 "language"),
    ("story",       "Narrative",              "language"),

    # ── Music / arts ──
    ("piano",       "Piano",                  "music"),
    ("guitar",      "Guitar",                 "music"),
    ("violin",      "Violin",                 "music"),
    ("trumpet",     "Trumpet",                "music"),
    ("drum",        "Drum",                   "music"),
    ("flute",       "Flute",                  "music"),
    ("note",        "Musical_note",           "music"),
    ("paintbrush",  "Paintbrush",             "art"),
    ("palette",     "Palette_(painting)",     "art"),
    ("sculpture",   "Sculpture",              "art"),
    ("museum",      "Museum",                 "art"),

    # ── Weather / atmosphere ──
    ("cloud",       "Cloud",                  "weather"),
    ("rain",        "Rain",                   "weather"),
    ("snow",        "Snow",                   "weather"),
    ("storm",       "Storm",                  "weather"),
    ("lightning",   "Lightning",              "weather"),
    ("tornado",     "Tornado",                "weather"),
    ("hurricane",   "Tropical_cyclone",       "weather"),
    ("fog",         "Fog",                    "weather"),
    ("wind",        "Wind",                   "weather"),

    # ── Energy / motion ──
    ("light",       "Light",                  "physics"),
    ("sound",       "Sound",                  "physics"),
    ("heat",        "Heat",                   "physics"),
    ("electricity", "Electricity",            "physics"),
    ("gravity",     "Gravity",                "physics"),
    ("speed",       "Speed",                  "physics"),
    ("wheel",       "Wheel",                  "physics"),
    ("spring",      "Spring_(device)",        "physics"),
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "W1z4rDV1510n-K12-Trainer/1.0 (contact: c4rr13rX@gmail.com)"


def fetch_wikipedia_image(page_title: str, max_px: int = 256) -> bytes | None:
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


def synthesize_polly_wav(text: str, voice: str = "Joanna",
                         rate: int = 16000) -> bytes:
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


def post_json(url: str, body: dict, timeout: float = 60.0) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ensure_assets(cache_dir: Path, vocab: list[tuple[str, str, str]],
                  *, fetch_polly: bool, verbose: bool) -> list[tuple[str, str]]:
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
            time.sleep(0.2)
        if not wav_path.exists():
            if not fetch_polly:
                print(f"  [skip] no polly audio for {word!r}", flush=True)
                continue
            if verbose:
                print(f"  [polly] audio for {word!r}", flush=True)
            try:
                wav_path.write_bytes(synthesize_polly_wav(f"{word.replace('_', ' ')}"))
            except Exception as e:
                print(f"  [skip] polly failed for {word!r}: {e}", flush=True)
                continue
        ready.append((word, category))
    return ready


def shuffled(items: list, seed: int) -> list:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def train_triples(node_url: str, ready: list[tuple[str, str]],
                  cache_dir: Path, repeats: int, lr: float,
                  verbose: bool) -> dict:
    print(f"\n[train] {len(ready)} K-12 vocab terms x {repeats} epochs", flush=True)
    fails = 0
    for epoch in range(repeats):
        for word, _category in shuffled(ready, seed=epoch * 2011 + 31):
            img_b64 = base64.b64encode((cache_dir / f"{word}.jpg").read_bytes()).decode("ascii")
            wav_b64 = base64.b64encode((cache_dir / f"{word}.wav").read_bytes()).decode("ascii")
            try:
                r = post_json(f"{node_url}/sensor/observe_triple", {
                    "text":      word.replace("_", " "),
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
    return {"trained_terms": len(ready), "epochs": repeats, "fails": fails}


def probe_directions(node_url: str, ready: list[tuple[str, str]],
                     cache_dir: Path) -> dict:
    results = {"image": 0, "audio": 0, "text": 0,
               "by_category": {},
               "total": len(ready)}
    for word, category in ready:
        cat = results["by_category"].setdefault(category, {"image": 0, "audio": 0, "text": 0, "n": 0})
        cat["n"] += 1
        img_b64 = base64.b64encode((cache_dir / f"{word}.jpg").read_bytes()).decode("ascii")
        wav_b64 = base64.b64encode((cache_dir / f"{word}.wav").read_bytes()).decode("ascii")

        try:
            r = post_json(f"{node_url}/sensor/observe", {
                "kind": "image", "bytes_b64": img_b64,
            })
            if r.get("predictions"):
                results["image"] += 1
                cat["image"] += 1
        except Exception:
            pass

        try:
            r = post_json(f"{node_url}/sensor/observe", {
                "kind": "audio", "bytes_b64": wav_b64,
            })
            if r.get("predictions"):
                results["audio"] += 1
                cat["audio"] += 1
        except Exception:
            pass

        try:
            r = post_json(f"{node_url}/chat", {"text": word.replace("_", " ")})
            if r.get("predictions"):
                results["text"] += 1
                cat["text"] += 1
        except Exception:
            pass

    return results


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--node",    default="http://127.0.0.1:8090")
    p.add_argument("--words",   type=int, default=len(VOCAB))
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--lr",      type=float, default=2.0)
    p.add_argument("--cache",   default="data/multimodal_k12")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--no-polly", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cache = Path(args.cache).resolve()
    vocab = VOCAB[: args.words]
    print(f"[ensure] {len(vocab)} K-12 vocab terms; cache at {cache}", flush=True)
    ready = ensure_assets(cache, vocab,
                          fetch_polly=not args.no_polly,
                          verbose=args.verbose)
    print(f"[ready] {len(ready)}/{len(vocab)} terms have both image + audio",
          flush=True)
    if not ready:
        return 1

    if not args.probe_only:
        summary = train_triples(args.node, ready, cache,
                                args.repeats, args.lr, args.verbose)
        print(f"[train summary] {summary}", flush=True)

    print(f"\n[probe] testing bidirectional cross-modal recall...", flush=True)
    res = probe_directions(args.node, ready, cache)
    n = max(res["total"], 1)
    print(f"\n=== CROSS-MODAL RECALL (K-12) ===")
    print(f"  image -> others:   {res['image']}/{n}  ({100*res['image']/n:.1f}%)")
    print(f"  audio -> others:   {res['audio']}/{n}  ({100*res['audio']/n:.1f}%)")
    print(f"  text  -> others:   {res['text']}/{n}  ({100*res['text']/n:.1f}%)")
    print(f"\n  by category:")
    for cat, c in sorted(res["by_category"].items()):
        n_cat = max(c["n"], 1)
        print(f"    {cat:<14} n={c['n']:<3}  "
              f"img={c['image']:<3}({100*c['image']/n_cat:.0f}%)  "
              f"aud={c['audio']:<3}({100*c['audio']/n_cat:.0f}%)  "
              f"txt={c['text']:<3}({100*c['text']/n_cat:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
