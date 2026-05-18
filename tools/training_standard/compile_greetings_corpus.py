"""tools/training_standard/compile_greetings_corpus.py
========================================================

Compile a greetings training corpus from third-party data already on
disk.  No new text is fabricated: the source rows are entries already
present in `data/training/conversation_basics_001.jsonl` (compiled
during a prior `tools/training_standard/generate_corpora.py` run) and
in `data/foundation/concept_dataset.jsonl` (which packages each
concept with a definition and a canonical QA pair).

Output: data/training/greetings_001.jsonl

Categories included:
  * Pure greeting prompts (hi / hello / hey / good morning / etc.)
    from conversation_basics_001 — these already carry the W1z4rD
    persona-consistent response we want to reinforce.
  * `social_word` and `social` concepts from concept_dataset, which
    package short conversational atoms (no, more, please, thanks,
    sorry, ...) with their definitions and a what-does-X-mean QA pair.

The output schema is the standard one the registry consumes:
  {"prompt": "...", "response": "..."}

Run:
  python -m tools.training_standard.compile_greetings_corpus
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVO_PATH    = _PROJECT_ROOT / "data" / "training" / "conversation_basics_001.jsonl"
CONCEPT_PATH  = _PROJECT_ROOT / "data" / "foundation" / "concept_dataset.jsonl"
OUT_PATH      = _PROJECT_ROOT / "data" / "training" / "greetings_001.jsonl"

# Greeting / closing / acknowledgment surface forms.  Match at start.
GREET_PAT = re.compile(
    r"^(hi|hello|hey|hiya|howdy|greetings|"
    r"good\s+(morning|afternoon|evening|night|day)|"
    r"goodbye|bye(\s+bye)?|see\s+you|farewell|"
    r"thanks?|thank\s+you|"
    r"how\s+are\s+you|how('?s|\s+is)\s+it\s+going|"
    r"what'?s\s+up|nice\s+to\s+meet|pleased\s+to|"
    r"welcome|excuse\s+me|sorry|pardon|please)\b",
    re.IGNORECASE,
)

SOCIAL_CATEGORIES = {"social_word", "social"}


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


def compile_greetings() -> list[dict]:
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # Hard cap on response length.  Long definitional responses
    # (e.g. "Problem is a state of difficulty that needs to be
    # resolved.") crystallise as long action-pool concepts that
    # poison categorical retrieval by atom overlap.  Per ARCHITECTURE.md
    # §4.D.1 the substrate is supposed to use the deepest CONFIDENT
    # layer; bloated definitional concepts with many atoms compete
    # against legitimate short category responses at the atom layer
    # and win on bag-of-letters bleed.  Cap at 32 bytes — fits within
    # the action pool's max_concept_member_count (Stage 12 config).
    _MAX_RESP_LEN = 32

    def _add(prompt: str, response: str, source: str) -> None:
        p, r = prompt.strip(), response.strip()
        if not p or not r:
            return
        if len(r) > _MAX_RESP_LEN:
            return  # drop polluting long responses
        key = (p.lower(), r.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append({"prompt": p, "response": r, "source": source})

    # 1) Pull greeting-like prompts from conversation_basics_001.
    for row in _iter_jsonl(CONVO_PATH):
        prompt = (row.get("prompt") or "").strip()
        resp   = (row.get("response") or "").strip()
        if not prompt or not resp:
            continue
        if GREET_PAT.match(prompt):
            _add(prompt, resp, source="conversation_basics_001")

    # 2) Pull social_word / social concepts from concept_dataset.
    #    Schema:  concept, definition, category, qa_pairs[{question, answer}]
    for row in _iter_jsonl(CONCEPT_PATH):
        cat = (row.get("category") or "").lower()
        if cat not in SOCIAL_CATEGORIES:
            continue
        concept = (row.get("concept") or "").strip()
        defn    = (row.get("definition") or "").strip()
        if concept and defn:
            _add(concept, defn, source="concept_dataset:social")
        for qa in row.get("qa_pairs") or []:
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            if q and a:
                _add(q, a, source="concept_dataset:social_qa")

    return pairs


def main() -> int:
    pairs = compile_greetings()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p["prompt"], "response": p["response"]}) + "\n")
    print(f"[compile_greetings] wrote {len(pairs)} pairs -> {OUT_PATH}", flush=True)

    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
