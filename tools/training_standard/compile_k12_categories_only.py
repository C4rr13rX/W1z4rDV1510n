"""tools/training_standard/compile_k12_categories_only.py
==========================================================

Stage 12 follow-up corpus: emit ONLY the short (concept, category)
pairs from concept_dataset, dropping the definitional and QA pairs.

Rationale (from math_audit_concepts.py + live diagnostic):
  - The full K-12 corpus produces 3 bindings per concept whose
    text-pool members are identical (concept word atoms):
        (piano, musical_instrument)   -- short target
        (piano, "Piano is...")        -- long definitional target
        (What does piano mean?, ...)  -- different text but adds noise
  - All three crystallize as bindings with precision=1.0 against
    the query atoms.  Stage 7 binding-pool routing's tiebreak
    (HashMap iteration order) is non-deterministic — the long
    definition binding often wins, then its target_atoms set is too
    large for any single concept to clear the 0.99 coverage gate.
  - Result: no binding_boost fires.  Selection falls back to pure
    fabric activation, which favors toddler atoms.
  - Removing the long-target bindings eliminates the tie and lets
    Stage 7 routing land on the short category binding cleanly.

Output: data/training/k12_categories_only_001.jsonl

This is a PURE TRAINING experiment.  The script that consumes the
corpus is unchanged; only the corpus shape differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONCEPT_PATH  = _PROJECT_ROOT / "data" / "foundation" / "concept_dataset.jsonl"
CLASS_PATH    = _PROJECT_ROOT / "data" / "foundation" / "class_corpus.jsonl"
OUT_PATH      = _PROJECT_ROOT / "data" / "training" / "k12_categories_only_001.jsonl"

EXCLUDED_CATEGORIES = {"social_word", "social"}  # owned by greetings corpus.


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compile_categories() -> list[dict]:
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(prompt: str, response: str, source: str) -> None:
        p, r = prompt.strip(), response.strip()
        if not p or not r:
            return
        # Hard cap on response length — 32 bytes max so the action
        # pool's concept_emergence sees the WHOLE response within its
        # recent_atoms window, and so the concept's member set stays
        # below the pool's max_concept_member_count (32 in Stage 12).
        if len(r) > 32:
            return  # skip rather than truncate
        key = (p.lower(), r.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append({"prompt": p, "response": r, "source": source})

    # ONLY (concept, category) pairs.  Definitions and QA dropped.
    for row in _iter_jsonl(CONCEPT_PATH):
        cat = (row.get("category") or "").lower()
        if not cat or cat in EXCLUDED_CATEGORIES:
            continue
        concept = (row.get("concept") or "").strip()
        if concept and cat:
            _add(concept, cat, source="concept_dataset:cat")

    # Class corpus: short class IDs only as targets (description → id).
    # Drop the reverse (id → description) which would emit long
    # response strings that bind to the same text atoms as the short
    # category bindings.
    for row in _iter_jsonl(CLASS_PATH):
        cid  = (row.get("class_id") or "").strip()
        desc = (row.get("description") or "").strip()
        if cid and desc:
            _add(desc, cid, source="class_corpus:desc->id")

    return pairs


def main() -> int:
    pairs = compile_categories()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p["prompt"], "response": p["response"]}) + "\n")
    print(f"[compile_k12_categories_only] wrote {len(pairs)} pairs -> {OUT_PATH}", flush=True)

    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
