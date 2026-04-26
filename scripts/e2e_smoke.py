#!/usr/bin/env python3
"""
e2e_smoke.py — non-interactive end-to-end smoke test for the multi-pool fabric.

Runs a scripted /chat session that exercises:
  1. Train conversation pairs via /multi_pool/train_pair.
  2. Train a parallel emo pool via /multi_pool/train_fanout.
  3. Train a couple of equation-style pairs against an "equation" pool.
  4. Query /chat with trained inputs (exact recall expected).
  5. Query /chat with paraphrased inputs (nearest-neighbor recall).
  6. Query /multi_pool/ask to verify the parallel emo + equation pools fire.
  7. Query an OOD input to confirm graceful degradation.
"""
from __future__ import annotations
import json
import sys
import urllib.request

NODE = "http://127.0.0.1:8090"


def post(path: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        NODE + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(NODE + path, timeout=timeout) as r:
        return json.loads(r.read())


PAIRS = [
    ("hello",                  "hello there friend welcome aboard",         "warm"),
    ("hi",                     "hi how can i help you today",               "cheerful"),
    ("good morning",           "good morning hope your day is great",       "warm"),
    ("how are you",            "i am doing well thanks for asking",         "calm"),
    ("who are you",            "i am wizard a distributed neural ai",       "neutral"),
    ("what can you do",        "i answer questions and learn from data",    "curious"),
    ("are you an ai",          "yes i am wizard a hebbian neural ai",       "neutral"),
    ("thanks",                 "you are most welcome",                      "grateful"),
    ("goodbye",                "goodbye take care safe travels",            "wistful"),
    ("explain quantum",        "quantum mechanics superposition entanglement", "curious"),
    ("explain gravity",        "gravity bends spacetime mass attracts mass",   "curious"),
    ("describe ocean",         "deep blue rolling waves crashing against shores", "calm"),
]

EQ_PAIRS = [
    ("explain quantum",        "schrodinger_equation"),
    ("explain gravity",        "einstein_field_equations"),
    ("describe ocean",         "navier_stokes"),
]

OOD_QUERIES = [
    ("hello!",                 "hello there friend welcome aboard"),
    ("hi there",               "hi how can i help you today"),
    ("explain quantum stuff",  "quantum mechanics superposition entanglement"),
    ("describe oceans",        "deep blue rolling waves crashing against shores"),
    ("totally unknown query",   None),  # no specific expectation; just no crash
]

PASS = "[PASS]"
FAIL = "[FAIL]"


def main() -> None:
    failed = 0
    total  = 0

    print("=" * 70)
    print("Multi-pool fabric end-to-end smoke test")
    print("=" * 70)

    # 1. Health check
    h = get("/health")
    print(f"\n[setup] node {h['node_id']} uptime {h['uptime_secs']}s")

    # 2. Register the emo + equation pools
    post("/multi_pool/register", {"pool_id": "emo"})
    post("/multi_pool/register", {"pool_id": "equation"})
    print("[setup] registered pools: emo, equation")

    # 3. Train all (in -> out, in -> emo) pairs via train_fanout
    print("\n[train] in -> {out, emo} via train_fanout (12 pairs, 30 passes each)")
    for q, a, e in PAIRS:
        post("/multi_pool/train_fanout", {
            "src_pool": "in", "src": q,
            "targets": [
                {"tgt_pool": "out", "tgt": a},
                {"tgt_pool": "emo", "tgt": e},
            ],
            "passes": 30, "lr": 0.5,
        })

    # 4. Train (in -> equation) pairs
    print("[train] in -> equation (3 pairs)")
    for q, eq in EQ_PAIRS:
        post("/multi_pool/train_pair", {
            "src_pool": "in", "src": q,
            "tgt_pool": "equation", "tgt": eq,
            "passes": 30, "lr": 0.5,
        })

    stats = get("/multi_pool/stats")
    print(f"[train] stats: cross_edges={stats['cross_edges']}")
    for pool_id, info in sorted(stats["pools"].items()):
        print(f"          {pool_id:>10}: neurons={info['neurons']:>5} sequences={info['sequences']}")

    # 5. /chat exact recall on trained Q's
    print("\n[test 1] /chat exact recall on trained inputs")
    for q, expected, _ in PAIRS:
        total += 1
        r = post("/chat", {"text": q, "hops": 3, "min_strength": 0.05})
        ans = r.get("answer") or ""
        ok = (ans == expected)
        if not ok: failed += 1
        marker = PASS if ok else FAIL
        print(f"  {marker} {q[:30]:<32} decoder={r.get('decoder')} ans={ans[:40]!r}")

    # 6. /chat OOD / paraphrase
    print("\n[test 2] /chat near-neighbor (paraphrase / OOD)")
    for q, expected in OOD_QUERIES:
        total += 1
        r = post("/chat", {"text": q, "hops": 3, "min_strength": 0.05})
        ans = r.get("answer") or ""
        if expected is None:
            ok = True  # any answer (or null) is fine
            note = "no-crash"
        else:
            ok = (ans == expected)
            note = "exact-near-neighbor"
        if not ok: failed += 1
        marker = PASS if ok else FAIL
        print(f"  {marker} {q[:30]:<32} {note:>20} ans={ans[:40]!r}")

    # 7. multi_pool/ask shows simultaneous fan-out: out + emo + equation
    print("\n[test 3] /multi_pool/ask fans out across all connected pools")
    for q, expected_out, expected_emo in PAIRS[:5]:
        total += 1
        r = post("/multi_pool/ask", {"src_pool": "in", "text": q})
        preds = r.get("predictions") or {}
        # out + emo expected; equation only for the EQ_PAIRS subset
        ok = (preds.get("out") == expected_out and preds.get("emo") == expected_emo)
        if not ok: failed += 1
        marker = PASS if ok else FAIL
        print(f"  {marker} {q[:25]:<27} out={preds.get('out', '')[:30]!r:<35} "
              f"emo={preds.get('emo')!r:<12} eq={preds.get('equation')!r}")

    # 8. Verify equation pool fires for the EQ_PAIRS inputs
    print("\n[test 4] /multi_pool/ask verifies equation pool fires for trained EQ pairs")
    for q, eq in EQ_PAIRS:
        total += 1
        r = post("/multi_pool/ask", {"src_pool": "in", "text": q})
        preds = r.get("predictions") or {}
        ok = (preds.get("equation") == eq)
        if not ok: failed += 1
        marker = PASS if ok else FAIL
        print(f"  {marker} {q[:25]:<27} equation={preds.get('equation')!r}")

    # 9. Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
