#!/usr/bin/env python3
"""
Smallest-possible motif-as-primary smoke test.

Goal: prove the architecture can recall a single trained (Q, A) pair
via the motif system's bigram transition graph before we commit any
larger training run.  Per the operator: "run small experiments that
validate it will work with a large training set with only a small
training set, and don't stop until you have verified that."

Setup:
  • clear pool
  • train ONE pair via /media/train_sequence × 5 passes:
        "ping"  →  "pong"
  • ask the motif system: what follows the label for "ping"?
  • assert the top-probability successor is the label for "pong"

If this fails on a tiny corpus, no amount of training data will fix
it — the wiring is wrong.  If it passes, we know the basic transition
plumbing works and we can build the multi-label prefix-query layer
on top with confidence.
"""
from __future__ import annotations
import base64
import json
import sys
import time
import urllib.request
import urllib.error
import uuid

NODE = "http://localhost:8090"


def b64url(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


def b64std(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def post(path: str, payload: dict, timeout: float = 20) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{NODE}{path}", data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"error": str(exc)}


def frame(text: str, t: float, idx: int, total: int) -> dict:
    return {
        "modality": "text", "t_secs": t, "lr_scale": 1.0,
        "data_b64": b64std(text), "text": text,
        "spans": [{
            "text": text, "role": "body", "bold": False, "italic": False,
            "size_ratio": 1.0, "x_frac": 0.5, "y_frac": 0.5,
            "seq_index": idx, "seq_total": total,
        }],
    }


def main() -> int:
    Q = "ping"
    A = "pong"
    PASSES = 5

    print("=" * 60)
    print(f"Motif-as-primary smoke test: {Q!r} -> {A!r}")
    print("=" * 60)

    print(f"\n[1/4] Clearing pool...")
    r = post("/neuro/clear", {})
    print(f"      response: {r}")
    time.sleep(1)

    print(f"\n[2/4] Training {Q!r} -> {A!r} × {PASSES} passes via train_sequence...")
    frames = [frame(Q, 0.0, 0, 2), frame(A, 0.4, 1, 2)]
    for i in range(PASSES):
        r = post("/media/train_sequence", {
            "session_id": str(uuid.uuid4()),
            "base_lr": 0.5,
            "tau_secs": 2.0,
            "frames": frames,
        })
        if r.get("error"):
            print(f"      pass {i+1} FAILED: {r['error']}")
            return 1
    print(f"      done")

    # TextBitsEncoder emits one label per character, not per word.  The
    # motif transition graph is over single-codepoint atoms.  Query for
    # the LAST char of Q and verify A's chars appear in the successor
    # distribution.
    def cl(ch: str) -> str:
        return "txt:" + b64url(ch)

    q_last_char = Q[-1]
    a_first_char = A[0]
    print(f"\n[3/4] Querying motif transitions from {cl(q_last_char)!r} (last char of {Q!r}={q_last_char!r})")
    print(f"      expecting {cl(a_first_char)!r} ({a_first_char!r}, first char of {A!r}) to appear")
    r = post("/neuro/motifs/predict", {"last_label": cl(q_last_char), "top_k": 20})
    preds = r.get("predictions", [])
    print(f"      top-{len(preds)} predictions:")
    for p in preds:
        decoded = "?"
        if isinstance(p.get("label"), str) and p["label"].startswith("txt:"):
            payload = p["label"][4:]
            pad = "=" * ((4 - len(payload) % 4) % 4)
            try:
                decoded = base64.urlsafe_b64decode(payload + pad).decode("utf-8", errors="replace")
            except Exception:
                pass
        print(f"        {p['label']!r:30s}  p={p['probability']:.4f}  decoded={decoded!r}")
    # Override targets — we look for first char of A, not the whole word.
    a_label = cl(a_first_char)

    print(f"\n[4/4] Verdict:")
    if not preds:
        print(f"      FAIL: motif system returned no predictions for {q_label}")
        print(f"            → motif observe_label_sequence isn't being called from train_sequence,")
        print(f"              or the observe path doesn't register single-pair transitions yet")
        return 2

    top = preds[0]
    if top["label"] == a_label:
        print(f"      PASS: top prediction is {a_label} ({A!r}) at p={top['probability']:.4f}")
        return 0
    if any(p["label"] == a_label for p in preds):
        rank = next(i for i, p in enumerate(preds) if p["label"] == a_label)
        print(f"      PARTIAL PASS: {a_label} present at rank {rank+1}, not #1")
        print(f"                    top was {top['label']!r} p={top['probability']:.4f}")
        print(f"                    → bigram is there but other transitions outcompete; expected")
        print(f"                      to be #1 on a fresh pool with only this pair trained")
        return 3
    print(f"      FAIL: {a_label} ({A!r}) not in top-10 predictions")
    print(f"            → train_sequence isn't producing the {Q!r}→{A!r} transition in motif level 0")
    return 4


if __name__ == "__main__":
    sys.exit(main())
