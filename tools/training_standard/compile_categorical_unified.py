"""tools/training_standard/compile_categorical_unified.py
============================================================

Unify ALL third-party categorical sources into one curated training
corpus shaped for the substrate's emergence math.  Sources:

  1. WordNet hyponym closures (NLTK)
     Curated by Brown-corpus frequency filter so only common-English
     words pass.  See compile_wordnet_categories.py.

  2. concept_dataset.jsonl (already on disk, K-12-curated third-party)
     Provides 1,623 (concept, category) pairs in K-12 vocabulary.

  3. Toddler 32-pair set (already-trained baseline) — included so the
     unified corpus REINFORCES rather than competes with the
     71.9%-recall toddler bindings.

Categories are unified across sources by matching the category label
exactly.  When concept_dataset uses 'musical_instrument' and WordNet
uses 'musical_instrument', they merge.

Output: data/training/categorical_unified_001.jsonl

The empirical math (4096-byte action-pool window, threshold=3 repeats)
requires ~26 pairs per category.  The unified corpus targets >= 30
per category by combining sources.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from nltk.corpus import wordnet as wn  # type: ignore
    from nltk.corpus import brown  # type: ignore
except ImportError:
    sys.exit("Missing: python -m nltk.downloader wordnet omw-1.4 brown")


ROOT          = Path(__file__).resolve().parents[2]
CONCEPT_PATH  = ROOT / "data" / "foundation" / "concept_dataset.jsonl"
OUT_PATH      = ROOT / "data" / "training" / "categorical_unified_001.jsonl"

# Toddler bindings — provided here so the unified corpus reinforces
# the 71.9% baseline rather than introducing new prompts that compete
# for the same atoms.  Sourced directly from scripts/brain_xpool_chat_test.py.
TODDLER_PAIRS: list[tuple[str, str]] = [
    ("dog", "animal"), ("cat", "animal"), ("cow", "animal"),
    ("horse", "animal"), ("bird", "animal"), ("fish", "animal"),
    ("apple", "food"), ("banana", "food"), ("bread", "food"),
    ("cake", "food"), ("milk", "food"),
    ("car", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"),
    ("plane", "vehicle"), ("boat", "vehicle"),
    ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"),
    ("ball", "toy"), ("doll", "toy"), ("kite", "toy"), ("drum", "toy"),
    ("tree", "nature"), ("flower", "nature"), ("river", "nature"), ("mountain", "nature"),
    ("hand", "body"), ("foot", "body"), ("eye", "body"), ("mouth", "body"),
]

# Same root set as compile_wordnet_categories.  Duplicated here for
# self-contained execution; if the two scripts drift, this is the
# authoritative copy for the unified corpus.
CATEGORY_ROOTS: dict[str, str] = {
    "animal.n.01": "animal", "fish.n.01": "animal", "bird.n.01": "animal",
    "mammal.n.01": "animal", "reptile.n.01": "animal", "insect.n.01": "animal",
    "amphibian.n.03": "animal",
    "plant.n.02": "plant", "tree.n.01": "plant", "flower.n.01": "plant",
    "food.n.01": "food", "food.n.02": "food",
    "vehicle.n.01": "vehicle",
    "color.n.01": "color",
    "shape.n.02": "shape",
    "body_part.n.01": "body",
    "natural_object.n.01": "nature",
    "tool.n.01": "tool",
    "musical_instrument.n.01": "musical_instrument",
    "number.n.02": "number",
    "sport.n.01": "sport",
    "game.n.01": "game",
    "emotion.n.01": "emotion", "feeling.n.01": "emotion",
    "material.n.01": "material", "metal.n.01": "material", "fabric.n.01": "material",
    "structure.n.01": "structure",
    "container.n.01": "container",
    "clothing.n.01": "clothing",
    "furniture.n.01": "furniture",
    "liquid.n.01": "liquid",
    "rock.n.01": "mineral",
    "move.v.02": "motion", "travel.v.01": "motion",
    "create.v.02": "creation", "make.v.03": "creation",
    "destroy.v.01": "destruction",
    "communicate.v.02": "communication", "speak.v.01": "communication",
    "see.v.01": "perception", "hear.v.01": "perception",
    "know.v.01": "cognition", "think.v.03": "cognition",
    "consume.v.02": "consumption", "eat.v.01": "consumption",
}

EXCLUDED_FROM_CONCEPT_DATASET = {"social_word", "social"}  # owned by greetings.

_BAD = re.compile(r"[^a-z]")
_MIN_LEN = 3
_MAX_LEN = 14
_BROWN_MIN_FREQ = 2


def _brown_vocab() -> set[str]:
    c: Counter = Counter()
    for w in brown.words():
        if w.isalpha():
            c[w.lower()] += 1
    return {w for w, n in c.items() if n >= _BROWN_MIN_FREQ}


def _acceptable(name: str, common: set[str]) -> bool:
    if "_" in name or " " in name or "-" in name:
        return False
    if not (_MIN_LEN <= len(name) <= _MAX_LEN):
        return False
    if _BAD.search(name):
        return False
    if name not in common:
        return False
    return True


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect() -> list[tuple[str, str, str]]:
    """Returns list of (lemma, category, source) tuples."""
    print("[compile] building Brown common-word vocab...", flush=True)
    common = _brown_vocab()
    print(f"  Brown {_BROWN_MIN_FREQ}+ vocab: {len(common)}", flush=True)

    pairs: dict[tuple[str, str], str] = {}  # (lemma, cat) → first source seen

    # 1) Toddler — never filtered (these are the regression baseline).
    for p, r in TODDLER_PAIRS:
        pairs.setdefault((p.lower(), r.lower()), "toddler")

    # 2) concept_dataset — K-12-curated, single (concept, category) per row.
    if CONCEPT_PATH.exists():
        for row in _iter_jsonl(CONCEPT_PATH):
            cat = (row.get("category") or "").lower()
            if not cat or cat in EXCLUDED_FROM_CONCEPT_DATASET:
                continue
            concept = (row.get("concept") or "").strip().lower()
            if not concept:
                continue
            # Keep multi-word concepts that match concept_dataset's
            # phrasing exactly — they were already manually curated.
            if 2 <= len(concept) <= 24 and " " not in concept:
                pairs.setdefault((concept, cat), "concept_dataset")
    else:
        print(f"  WARN: {CONCEPT_PATH} missing — skipping", flush=True)

    # 3) WordNet — Brown-filtered hyponym closures.
    for ss_name, cat in CATEGORY_ROOTS.items():
        try:
            root = wn.synset(ss_name)
        except Exception:
            continue
        for h in root.closure(lambda s: s.hyponyms()):
            for lemma in h.lemmas():
                name = lemma.name().lower()
                if not _acceptable(name, common):
                    continue
                pairs.setdefault((name, cat), "wordnet")

    return [(p, c, s) for (p, c), s in pairs.items()]


def main() -> int:
    rows = collect()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for lemma, cat, _src in rows:
            f.write(json.dumps({"prompt": lemma, "response": cat}) + "\n")
    print(f"[compile] wrote {len(rows)} pairs -> {OUT_PATH}")

    by_cat: Counter = Counter()
    by_src: Counter = Counter()
    for _, c, s in rows:
        by_cat[c] += 1
        by_src[s] += 1
    print(f"\ndistinct categories: {len(by_cat)}")
    print("per-category counts:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        mark = "ok " if n >= 26 else "LOW"
        print(f"  [{mark}] {cat:<22} {n:>6}")
    print(f"\nby source: {dict(by_src)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
