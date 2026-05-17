#!/usr/bin/env python3
"""scripts/brain_fluency_eval.py — empirical fluency probe of the brain.

Hits brain /chat with a curated panel that spans:
  * Trained toddler cross-pool pairs (the 32-pair set — must hold 71.9%)
  * Greetings (hello / hi / thanks / bye / good morning / ...)
  * K-12 categorical vocab (animal/color/food/body/...) from concept_dataset
  * Out-of-vocabulary prompts (must stay outside_grounding=true)

For each probe, reports:
  - the answer (truncated)
  - the decoder path (multi_pool / eem / char_chain)
  - outside_grounding flag
  - whether the answer contains the expected substring (when known)

This is the empirical replacement for the registry's deterministic
benchmarks — it works against the brain server at port 8095 (not the
legacy node at 8090) and reports honest hit / miss counts.

Run:
  python scripts/brain_fluency_eval.py
"""
from __future__ import annotations

import json
import sys
import urllib.request

BRAIN = "http://127.0.0.1:8095"


# (prompt, expected_substring_or_None, source_label)
PANEL: list[tuple[str, str | None, str]] = [
    # Toddler 32-pair (cross-pool)
    ("dog",      "animal",  "toddler"),
    ("cat",      "animal",  "toddler"),
    ("cow",      "animal",  "toddler"),
    ("horse",    "animal",  "toddler"),
    ("bird",     "animal",  "toddler"),
    ("fish",     "animal",  "toddler"),
    ("apple",    "food",    "toddler"),
    ("banana",   "food",    "toddler"),
    ("bread",    "food",    "toddler"),
    ("cake",     "food",    "toddler"),
    ("milk",     "food",    "toddler"),
    ("car",      "vehicle", "toddler"),
    ("truck",    "vehicle", "toddler"),
    ("bike",     "vehicle", "toddler"),
    ("plane",    "vehicle", "toddler"),
    ("boat",     "vehicle", "toddler"),
    ("red",      "color",   "toddler"),
    ("blue",     "color",   "toddler"),
    ("green",    "color",   "toddler"),
    ("yellow",   "color",   "toddler"),
    ("ball",     "toy",     "toddler"),
    ("doll",     "toy",     "toddler"),
    ("kite",     "toy",     "toddler"),
    ("drum",     "toy",     "toddler"),
    ("tree",     "nature",  "toddler"),
    ("flower",   "nature",  "toddler"),
    ("river",    "nature",  "toddler"),
    ("mountain", "nature",  "toddler"),
    ("hand",     "body",    "toddler"),
    ("foot",     "body",    "toddler"),
    ("eye",      "body",    "toddler"),
    ("mouth",    "body",    "toddler"),

    # Greetings (post-greetings_001 training)
    ("Hello",        None,     "greeting"),
    ("Hi",           None,     "greeting"),
    ("Thank you",    "welcome","greeting"),
    ("Thanks",       "welcome","greeting"),
    ("Goodbye",      "bye",    "greeting"),
    ("Bye",          "bye",    "greeting"),
    ("Good morning", "morning","greeting"),

    # K-12 categoricals (post-k12_subjects_001 training)
    ("piano",     "musical", "k12"),
    ("guitar",    "musical", "k12"),
    ("triangle",  "shape",   "k12"),
    ("square",    "shape",   "k12"),
    ("seven",     "number",  "k12"),
    ("nine",      "number",  "k12"),
    ("sad",       "emotion", "k12"),
    ("happy",     "emotion", "k12"),
    ("doctor",    "people",  "k12"),
    ("school",    "place",   "k12"),
    ("rose",      "plant",   "k12"),
    ("oak",       "plant",   "k12"),
    ("hammer",    "tool",    "k12"),
    ("saw",       "tool",    "k12"),
    ("football",  "sport",   "k12"),
    ("tennis",    "sport",   "k12"),

    # K-12 definitional QA
    ("What does water mean?",  "water", "k12_qa"),
    ("What does light mean?",  "light", "k12_qa"),
    ("What does sound mean?",  "sound", "k12_qa"),

    # Stage 11C multi-fact probes — these concepts appear in the
    # concept_dataset with both a category-binding AND a definition-
    # binding.  After K-12 training, chain_explore should visit BOTH
    # facts and the assembler should compose the two answers.
    ("piano",     "music",   "multi_fact"),
    ("rose",      "plant",   "multi_fact"),
    ("hammer",    "tool",    "multi_fact"),
    ("triangle",  "shape",   "multi_fact"),
    ("doctor",    "people",  "multi_fact"),

    # Out-of-vocabulary (must be honestly OOG)
    ("xyzzy",            None, "oov"),
    ("foobarbaz",        None, "oov"),
    ("zzzzqqqq",         None, "oov"),
]


def post(path: str, body: dict, timeout: float = 30.0) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BRAIN}{path}", data=raw, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def main() -> int:
    stats = post("/stats", {})
    print(f"[brain] tick={stats.get('tick')} neurons={stats.get('total_neurons')} "
            f"bindings={stats.get('total_binding')} terminals={stats.get('total_terminals')}",
            flush=True)

    by_source: dict[str, dict] = {}
    for prompt, expected, source in PANEL:
        r = post("/chat", {"text": prompt})
        ans = (r.get("answer") or "").strip()
        decoder = r.get("decoder")
        oog = (r.get("grounding") or {}).get("outside_grounding")
        if source == "oov":
            ok = (oog is True) and (ans == "")
            verdict = "OOG-correct" if ok else f"!! GROUNDED: {ans!r}"
        elif expected is None:
            # No specific expected answer — just check we got *something*
            ok = bool(ans) and (oog is False)
            verdict = f"answered: {ans!r}" if ok else "no-answer"
        else:
            ok = expected.lower() in ans.lower()
            verdict = f"hit: {ans!r}" if ok else f"miss: {ans!r}"
        s = by_source.setdefault(source, {"ok": 0, "n": 0})
        s["ok"] += int(bool(ok))
        s["n"] += 1
        print(f"  [{source:8s}] {prompt[:24]:24s} dec={decoder or '?':10s} "
                f"oog={oog}  {verdict}", flush=True)

    print("\n=== summary ===", flush=True)
    for src, s in sorted(by_source.items()):
        pct = 100.0 * s["ok"] / max(s["n"], 1)
        print(f"  {src:10s}: {s['ok']}/{s['n']}  ({pct:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
