#!/usr/bin/env python3
"""
stress_scenario_2b_dose_response.py — does scaling up reinforcement
strength close the "are you an ai" failure observed in scenario 2?

Scenario 2 ARM 2 protected A1 in 2/3 conflicts but failed on the
"are you an ai" pair — reinforcement of A1 with 5 cycles at
confidence=0.85 still let A2 ("No, I am a person.") win after one
round of plain A2 training.

This experiment grids reinforcement cycles × confidence to find out
whether the failure is dose-dependent (just needs more reinforcement)
or structural (a ceiling that no reinforcement can cross).  An
architectural fix is needed only if the latter.

Cells:    cycles ∈ {5, 10, 20, 50}
          confidence ∈ {0.85, 1.0}
Per cell: clear pool, train A1 (10 passes), reinforce A1 with the
          chosen (cycles, confidence), train A2 (10 passes), query Q,
          classify reply as A1 / A2 / MIX / NEITHER.
"""
from __future__ import annotations
import argparse
import base64
import json
import re
import sys
import time
import urllib.request
import uuid

NODE = "http://localhost:8090"

Q  = "are you an ai"
A1 = "Yes, I am W1z4rD, a Hebbian neural AI."
A2 = "No, I am a person."

CYCLES_GRID     = [5, 10, 20, 50]
CONFIDENCE_GRID = [0.85, 1.0]


def post(path: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(NODE + path, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def spans(text: str, idx: int, total: int) -> list:
    return [{"text": text, "role": "body", "bold": False, "italic": False,
             "size_ratio": 1.0, "x_frac": 0.5, "y_frac": 0.5,
             "seq_index": idx, "seq_total": total}]


def _frames(q: str, a: str) -> list:
    q_words, a_words = q.split(), a.split()
    total = len(q_words) + len(a_words)
    out = []
    for j, w in enumerate(q_words):
        out.append({"modality": "text", "t_secs": j * 0.05, "lr_scale": 1.0,
                     "data_b64": b64(w), "text": w,
                     "spans": spans(w, j, total)})
    a_start = (len(q_words) - 1) * 0.05 + 0.4 if q_words else 0.4
    for j, w in enumerate(a_words):
        out.append({"modality": "text", "t_secs": a_start + j * 0.10,
                     "lr_scale": 1.0, "data_b64": b64(w), "text": w,
                     "spans": spans(w, len(q_words) + j, total)})
    return out


def train_multi(q: str, a: str, *, passes: int = 10) -> None:
    post("/multi_pool/train_pair", {
        "src_pool": "in", "src": q, "tgt_pool": "out", "tgt": a,
        "passes": passes,
    }, timeout=60)


def reinforce_pair(q: str, a: str, cycles: int, confidence: float) -> None:
    frames = _frames(q, a)
    for _ in range(cycles):
        post("/media/train_sequence", {
            "session_id": str(uuid.uuid4()),
            "base_lr": 0.40, "tau_secs": 2.0, "frames": frames,
        }, timeout=20)
        post("/neuro/reinforce", {"confidence": confidence}, timeout=10)


def query_chat(q: str) -> tuple[str, str]:
    r = post("/chat", {"text": q, "hops": 2, "min_strength": 0.05}, timeout=10)
    return (r.get("answer") or ""), r.get("decoder", "?")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _distinguishing_tokens(target: str, other: str) -> list[str]:
    target_words = set(re.findall(r"[A-Za-z0-9]+", target))
    other_words  = set(re.findall(r"[A-Za-z0-9]+", other))
    return [w for w in target_words if w not in other_words and len(w) >= 3] \
            or [target.split()[0]]


def classify_answer(actual: str) -> str:
    actual_n = _norm(actual)
    a1_keys = _distinguishing_tokens(A1, A2)
    a2_keys = _distinguishing_tokens(A2, A1)
    has_a1 = any(_norm(k) in actual_n for k in a1_keys)
    has_a2 = any(_norm(k) in actual_n for k in a2_keys)
    if has_a1 and has_a2: return "MIX"
    if has_a1:            return "A1"
    if has_a2:            return "A2"
    return "NEITHER"


def run_cell(cycles: int, confidence: float) -> dict:
    post("/neuro/clear", {})
    time.sleep(0.3)
    train_multi(Q, A1, passes=10)
    reinforce_pair(Q, A1, cycles=cycles, confidence=confidence)
    train_multi(Q, A2, passes=10)
    answer, decoder = query_chat(Q)
    cls = classify_answer(answer)
    return {
        "cycles":     cycles,
        "confidence": confidence,
        "answer":     answer,
        "decoder":    decoder,
        "class":      cls,
    }


def main() -> int:
    global NODE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", default=NODE)
    args = ap.parse_args()
    NODE = args.node

    print(f"Q  = {Q!r}")
    print(f"A1 = {A1!r}  (target — should win if reinforcement holds)")
    print(f"A2 = {A2!r}  (interferer — wins by default if A1 isn't protected)")
    print()
    print(f"  {'cycles':>6s}  {'conf':>5s}  {'class':>7s}  decoder      answer")

    results = []
    for cycles in CYCLES_GRID:
        for conf in CONFIDENCE_GRID:
            r = run_cell(cycles, conf)
            short = r["answer"][:50].replace("\n", "\\n")
            print(f"  {r['cycles']:6d}  {r['confidence']:5.2f}  "
                  f"{r['class']:>7s}  {r['decoder']:12s} {short!r}")
            results.append(r)

    print()
    print("============== Verdict ==============")
    a1_wins = sum(1 for r in results if r["class"] == "A1")
    a2_wins = sum(1 for r in results if r["class"] == "A2")
    print(f"A1 wins: {a1_wins} / {len(results)}")
    print(f"A2 wins: {a2_wins} / {len(results)}")
    flipped = [r for r in results
               if r["class"] == "A1" and r["cycles"] > 5]
    if a1_wins == 0:
        print("CONCLUSION: dose-response negative — even cycles=50 at conf=1.0 "
              "did not flip the result.  This is a structural ceiling, not a "
              "tuning issue.  An architectural fix is needed (e.g. an "
              "additional consolidation factor that survives plain training "
              "more robustly).")
        return 2
    if flipped:
        first = sorted(flipped, key=lambda r: (r["cycles"], r["confidence"]))[0]
        print(f"CONCLUSION: dose-response positive — the failure flips at "
              f"cycles={first['cycles']}, conf={first['confidence']}.  "
              f"Reinforcement is dose-dependent; the default verifier setting "
              f"(5 cycles, 0.85) was just too weak for highly-discriminative "
              f"adversarial answers like A2's leading 'No'.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
