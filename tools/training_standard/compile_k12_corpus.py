"""tools/training_standard/compile_k12_corpus.py
==================================================

Compile a K-12 vocabulary / world-knowledge training corpus from
third-party data already on disk.  Outputs go to
`data/training/k12_subjects_001.jsonl` in the standard registry
schema:  one `{"prompt": ..., "response": ...}` per line.

Sources (no new content fabricated by this script — every pair is
either copied verbatim or trivially restructured from data already
present in the repo):

  1. data/foundation/concept_dataset.jsonl
     1675 third-party-prepared concept rows organized into 36
     semantic categories (animal, body, color, food, language,
     math, music, nature, plant, science, ...).  Each row carries
     a concept token, a category, a short definition, and a
     `qa_pairs[]` array with at least one (question, answer) pair.

     We emit:
       - (concept, category)           — categorical association
       - (concept, definition)         — semantic anchor
       - (question, answer) from qa_pairs

  2. data/foundation/class_corpus.jsonl
     155 third-party class-description rows (data structures,
     algorithms, OO patterns).  Emit (description, code-name)
     and (code-name, description) as bidirectional vocab.

The `social_word`/`social` rows are deliberately *excluded* here
because the greetings corpus (`compile_greetings_corpus.py`) already
owns them — duplicating would over-train on social atoms and waste
training time.

Run:
  python -m tools.training_standard.compile_k12_corpus
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONCEPT_PATH  = _PROJECT_ROOT / "data" / "foundation" / "concept_dataset.jsonl"
CLASS_PATH    = _PROJECT_ROOT / "data" / "foundation" / "class_corpus.jsonl"
OUT_PATH      = _PROJECT_ROOT / "data" / "training" / "k12_subjects_001.jsonl"

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


def compile_k12() -> list[dict]:
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(prompt: str, response: str, source: str) -> None:
        p, r = prompt.strip(), response.strip()
        if not p or not r:
            return
        # Cap response length so brain action-pool atoms stay scoped.
        # Cross-pool decoding retrieves a single concept; very long
        # responses dilute the binding strength of the head atoms.
        if len(r) > 140:
            r = r[:140].rsplit(" ", 1)[0]
        key = (p.lower(), r.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append({"prompt": p, "response": r, "source": source})

    # 1) Concept dataset — categorical + definitional + canonical QA.
    for row in _iter_jsonl(CONCEPT_PATH):
        cat = (row.get("category") or "").lower()
        if not cat or cat in EXCLUDED_CATEGORIES:
            continue
        concept = (row.get("concept") or "").strip()
        defn    = (row.get("definition") or "").strip()
        if concept and cat:
            _add(concept, cat, source="concept_dataset:cat")
        if concept and defn:
            _add(concept, defn, source="concept_dataset:def")
        for qa in row.get("qa_pairs") or []:
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            if q and a:
                _add(q, a, source="concept_dataset:qa")

    # 2) Class corpus — CS / data-structures vocab (bidirectional).
    for row in _iter_jsonl(CLASS_PATH):
        cid  = (row.get("class_id") or "").strip()
        desc = (row.get("description") or "").strip()
        if cid and desc:
            _add(desc, cid, source="class_corpus:desc->id")
            _add(cid, desc, source="class_corpus:id->desc")

    return pairs


def main() -> int:
    pairs = compile_k12()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p["prompt"], "response": p["response"]}) + "\n")
    print(f"[compile_k12] wrote {len(pairs)} pairs -> {OUT_PATH}", flush=True)

    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
