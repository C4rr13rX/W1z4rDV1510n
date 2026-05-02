#!/usr/bin/env python3
"""
stress_scenario_2_conflict.py — what happens when the same Q is trained
with two different As?  Does multi_pool overwrite, blend, or pick by
recency?  Does reinforcement protect the first answer from the second?

For each conflict pair the test runs four arms on a freshly-cleared pool:

  ARM 1  train A1 (10 passes)  →  train A2 (10 passes)  →  query
         Both answers get equal training.  Outcome shows the default
         arbitration (recency vs. order vs. blend).

  ARM 2  train A1 (10 passes)  →  REINFORCE A1 (5 cycles)
                                →  train A2 (10 passes)  →  query
         A1 was reinforced into late-LTP.  Did it survive A2's
         arrival?

  ARM 3  train A1 (10 passes)  →  train A1 again (10 more)
                                →  train A2 (10 passes)  →  query
         A1 got 2× the training of A2 but no reinforcement.  Volume
         protection vs. recency.

  ARM 4  train A1 (10 passes)  →  REINFORCE A1
                                →  train A2 (10 passes)  →  REINFORCE A2
                                →  query
         Both answers consolidated.  Last-reinforced should win, or
         both should appear at saturation.
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

CONFLICTS: list[tuple[str, str, str]] = [
    ("who are you",
     "I am W1z4rD, a distributed neural AI.",
     "I am Bob, a regular human."),
    ("what is your name",
     "My name is W1z4rD.",
     "My name is Carl."),
    ("are you an ai",
     "Yes, I am W1z4rD, a Hebbian neural AI.",
     "No, I am a person."),
]


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


def reinforce_pair(q: str, a: str, *, cycles: int = 5) -> None:
    frames = _frames(q, a)
    for _ in range(cycles):
        post("/media/train_sequence", {
            "session_id": str(uuid.uuid4()),
            "base_lr": 0.40, "tau_secs": 2.0, "frames": frames,
        }, timeout=20)
        post("/neuro/reinforce", {"confidence": 0.85}, timeout=10)


def query(q: str) -> tuple[str, str]:
    r = post("/chat", {"text": q, "hops": 2, "min_strength": 0.05}, timeout=10)
    return (r.get("answer") or ""), r.get("decoder", "?")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def classify_answer(actual: str, a1: str, a2: str) -> str:
    """Bucket the response into A1, A2, MIX, or NEITHER based on which
    set of distinguishing tokens it contains."""
    actual_n = _norm(actual)
    a1_keys = _distinguishing_tokens(a1, a2)
    a2_keys = _distinguishing_tokens(a2, a1)
    has_a1 = any(_norm(k) in actual_n for k in a1_keys)
    has_a2 = any(_norm(k) in actual_n for k in a2_keys)
    if has_a1 and has_a2:
        return "MIX"
    if has_a1:
        return "A1"
    if has_a2:
        return "A2"
    return "NEITHER"


def _distinguishing_tokens(target: str, other: str) -> list[str]:
    """Words present in `target` but not in `other` — used to detect
    which answer a chat reply favors when both are candidates."""
    target_words = set(re.findall(r"[A-Za-z0-9]+", target))
    other_words  = set(re.findall(r"[A-Za-z0-9]+", other))
    distinctive = [w for w in target_words if w not in other_words and len(w) >= 3]
    return distinctive or [target.split()[0]]


# ── Arms ────────────────────────────────────────────────────────────────────

def arm_1(q: str, a1: str, a2: str) -> dict:
    """Train A1 then A2, equal passes, no reinforcement."""
    post("/neuro/clear", {})
    time.sleep(0.3)
    train_multi(q, a1)
    train_multi(q, a2)
    return _query_and_classify("ARM 1: A1 + A2", q, a1, a2)


def arm_2(q: str, a1: str, a2: str) -> dict:
    """Train A1, reinforce A1, train A2."""
    post("/neuro/clear", {})
    time.sleep(0.3)
    train_multi(q, a1)
    reinforce_pair(q, a1)
    train_multi(q, a2)
    return _query_and_classify("ARM 2: A1 + reinforce(A1) + A2", q, a1, a2)


def arm_3(q: str, a1: str, a2: str) -> dict:
    """Train A1 twice (volume), then A2 once.  Volume vs. recency."""
    post("/neuro/clear", {})
    time.sleep(0.3)
    train_multi(q, a1)
    train_multi(q, a1)
    train_multi(q, a2)
    return _query_and_classify("ARM 3: 2x A1 + A2", q, a1, a2)


def arm_4(q: str, a1: str, a2: str) -> dict:
    """Both answers reinforced.  Belt and suspenders test."""
    post("/neuro/clear", {})
    time.sleep(0.3)
    train_multi(q, a1)
    reinforce_pair(q, a1)
    train_multi(q, a2)
    reinforce_pair(q, a2)
    return _query_and_classify("ARM 4: A1 + reinf + A2 + reinf", q, a1, a2)


def _query_and_classify(arm: str, q: str, a1: str, a2: str) -> dict:
    actual, decoder = query(q)
    cls = classify_answer(actual, a1, a2)
    return {"arm": arm, "q": q, "answer": actual, "decoder": decoder, "class": cls}


# ── Driver ──────────────────────────────────────────────────────────────────

def main() -> int:
    global NODE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", default=NODE)
    args = ap.parse_args()
    NODE = args.node

    print(f"Conflicts to test: {len(CONFLICTS)}")
    print()

    all_results = []
    for q, a1, a2 in CONFLICTS:
        print(f"=== Conflict: {q!r} ===")
        print(f"  A1: {a1!r}")
        print(f"  A2: {a2!r}")
        for arm_fn in (arm_1, arm_2, arm_3, arm_4):
            res = arm_fn(q, a1, a2)
            short = res["answer"][:80].replace("\n", "\\n")
            print(f"  [{res['class']:7s}] {res['arm']:42s} -> {short!r}")
            all_results.append(res)
        print()

    print("======================== AGGREGATE ========================")
    by_arm: dict[str, dict[str, int]] = {}
    for r in all_results:
        a = r["arm"]
        by_arm.setdefault(a, {"A1": 0, "A2": 0, "MIX": 0, "NEITHER": 0})
        by_arm[a][r["class"]] += 1
    for arm, counts in by_arm.items():
        total = sum(counts.values())
        print(f"  {arm:42s}  A1={counts['A1']}  A2={counts['A2']}  "
              f"MIX={counts['MIX']}  NEITHER={counts['NEITHER']}  (n={total})")
    print("===========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
