"""tools/training_standard/compile_wordnet_categories.py
==========================================================

Compile a categorical training corpus from WordNet 3.0 (third-party
open-licensed lexical database).  For each category root synset, walk
its hyponym closure and emit one `(lemma, category)` pair per
single-token lemma.

This is the corpus shape the substrate's emergence math actually
consumes: many distinct prompts mapping to ONE short category
response.  The empirical math established in scripts/math_audit_*.py
sets the threshold for a category to emerge as a queryable concept
in the action pool at ~26 pairs / 4096-byte window.  WordNet's
hyponym closures comfortably exceed this for every interesting
category (animal: 5457, plant: 3239, motion: 2106, ...).

The category labels are chosen to MATCH the substrate's existing
toddler categories ('animal', 'vehicle', 'color', 'food', 'toy',
'nature', 'body') so WordNet entries REINFORCE the existing
bindings rather than create competing ones.

Source data: NLTK WordNet corpus (downloaded once via
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
)

Output: data/training/wordnet_categories_001.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from nltk.corpus import wordnet as wn  # type: ignore
except ImportError:
    sys.exit("Missing: pip install nltk  +  python -m nltk.downloader wordnet omw-1.4")

try:
    from nltk.corpus import brown  # type: ignore
except ImportError:
    sys.exit("Missing: python -m nltk.downloader brown")


# Hyponym roots → category labels.  Aligned with the substrate's
# toddler categories where possible; new categories (motion,
# communication, creation, ...) added to broaden the queryable
# semantic space.
CATEGORY_ROOTS: dict[str, str] = {
    # Nouns reinforcing toddler bindings
    "animal.n.01":            "animal",
    "fish.n.01":              "animal",
    "bird.n.01":              "animal",
    "mammal.n.01":            "animal",
    "reptile.n.01":           "animal",
    "insect.n.01":            "animal",
    "amphibian.n.03":         "animal",
    "plant.n.02":             "plant",
    "tree.n.01":              "plant",
    "flower.n.01":            "plant",
    "food.n.01":              "food",
    "food.n.02":              "food",
    "vehicle.n.01":           "vehicle",
    "color.n.01":             "color",
    "shape.n.02":             "shape",
    "body_part.n.01":         "body",
    "natural_object.n.01":    "nature",
    # New noun categories (no toddler conflict)
    "tool.n.01":              "tool",
    "musical_instrument.n.01":"musical_instrument",
    "number.n.02":            "number",
    "sport.n.01":             "sport",
    "game.n.01":              "game",
    "emotion.n.01":           "emotion",
    "feeling.n.01":           "emotion",
    "material.n.01":          "material",
    "metal.n.01":             "material",
    "fabric.n.01":            "material",
    "structure.n.01":         "structure",
    "container.n.01":         "container",
    "clothing.n.01":          "clothing",
    "furniture.n.01":         "furniture",
    "liquid.n.01":            "liquid",
    "rock.n.01":              "mineral",
    # Verbs — entire new action-categorical domain
    "move.v.02":              "motion",
    "travel.v.01":             "motion",
    "create.v.02":             "creation",
    "make.v.03":               "creation",
    "destroy.v.01":            "destruction",
    "communicate.v.02":        "communication",
    "speak.v.01":              "communication",
    "see.v.01":                "perception",
    "hear.v.01":               "perception",
    "know.v.01":               "cognition",
    "think.v.03":              "cognition",
    "consume.v.02":            "consumption",
    "eat.v.01":                "consumption",
}

# A lemma can be reached from MANY roots (e.g., "dog" is a mammal → animal,
# and also a hyponym of several other synsets).  We use a priority order:
# the category that wins is the one most specific to the substrate's
# existing toddler bindings.
TODDLER_CATEGORIES = {"animal", "plant", "food", "vehicle", "color",
                       "shape", "body", "nature"}

# Lemma quality filter — drop names that aren't useful as single-token
# K-12 vocab.
_BAD_LEMMA_RE = re.compile(r"[^a-z]")  # only letters
_MIN_LEN = 3
_MAX_LEN = 14


def _build_common_english_vocab() -> set[str]:
    """Lowercase set of every alphabetic word that appears at least
    `_BROWN_MIN_FREQ` times in the Brown corpus.  Used as a curation
    filter: WordNet lemmas not present in everyday English text are
    dropped (medical/clinical/military jargon, obscure scientific
    names, etc.).  Brown is third-party open-licensed."""
    from collections import Counter
    counts: Counter = Counter()
    for w in brown.words():
        if w.isalpha():
            counts[w.lower()] += 1
    return {w for w, c in counts.items() if c >= _BROWN_MIN_FREQ}


_BROWN_MIN_FREQ = 2  # word must appear at least 2x in Brown to count as "common"
_COMMON: set[str] = set()  # populated in main()


def acceptable_lemma(name: str) -> bool:
    if "_" in name or " " in name or "-" in name:
        return False
    if len(name) < _MIN_LEN or len(name) > _MAX_LEN:
        return False
    if _BAD_LEMMA_RE.search(name):
        return False
    # Brown-corpus frequency curation.
    if _COMMON and name not in _COMMON:
        return False
    return True


def collect_pairs() -> dict[tuple[str, str], None]:
    """Returns an ordered dict of (lemma, category) keys (dedup preserved)."""
    pairs: dict[tuple[str, str], None] = {}
    lemma_to_cats: dict[str, set[str]] = {}

    for ss_name, cat in CATEGORY_ROOTS.items():
        try:
            root = wn.synset(ss_name)
        except Exception:
            continue
        # closure() returns generator over all hyponyms recursively.
        for h in root.closure(lambda s: s.hyponyms()):
            for lemma in h.lemmas():
                name = lemma.name().lower()
                if not acceptable_lemma(name):
                    continue
                lemma_to_cats.setdefault(name, set()).add(cat)

    # Resolve multi-category lemmas: prefer toddler-aligned categories.
    for lemma, cats in lemma_to_cats.items():
        toddler_overlap = cats & TODDLER_CATEGORIES
        if toddler_overlap:
            # If toddler-aligned categories conflict with each other,
            # we keep both — same lemma can be (lemma, animal) AND
            # (lemma, nature) e.g. for "horse" which is animal but
            # also nature.  Multiple labels reinforce the binding
            # set rather than create noise.
            for cat in toddler_overlap:
                pairs[(lemma, cat)] = None
        else:
            for cat in cats:
                pairs[(lemma, cat)] = None
    return pairs


def main() -> int:
    global _COMMON
    print("[wordnet_categories] building Brown common-word vocab...", flush=True)
    _COMMON = _build_common_english_vocab()
    print(f"  Brown ({_BROWN_MIN_FREQ}+ occurrences) vocab size: {len(_COMMON)}", flush=True)
    pairs = collect_pairs()
    out = Path("data/training/wordnet_categories_001.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    by_cat: Counter = Counter()
    with out.open("w", encoding="utf-8") as f:
        for (lemma, cat) in pairs:
            f.write(json.dumps({"prompt": lemma, "response": cat}) + "\n")
            by_cat[cat] += 1

    print(f"[wordnet_categories] wrote {len(pairs)} (lemma, category) pairs -> {out}")
    print(f"distinct categories: {len(by_cat)}")
    print("per-category counts:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        mark = "ok " if n >= 26 else "LOW"
        print(f"  [{mark}] {cat:<22} {n:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
