#!/usr/bin/env python3
"""scripts/brain_dense_burst_toddler.py

Hypothesis test for the destructive-collapse diagnosis.

Standard toddler training (brain_xpool_chat_test.py) uses EPOCHS:
  for epoch in 0..8:
    for pair in 32_pairs:
      observe(pair); tick

This means each prompt is observed once per epoch, spread across all
32 pairs.  By the time a single prompt is observed for the 3rd time
(needed for concept emergence), MANY OTHER words have been observed
in between — and their atoms have caused short-fragment concepts
(like 'pp', 'le', 'er') to emerge.  After those fragments emerge,
`collapse_tail_to_concept` collapses them destructively when the
prompt is observed again, preventing the FULL prompt sequence from
ever appearing as a contiguous atom run in recent_atoms.

DENSE BURST schedule:
  for pair in 32_pairs:
    for rep in 0..8:
      observe(pair); tick

Each prompt is observed 8 times back-to-back.  By the 3rd
consecutive observation, the full prompt-sequence count in
`self.sequences` reaches 3 and the WHOLE prompt promotes as a
single concept BEFORE any other word's atoms have a chance to
emerge fragments that would steal the collapse.

After training, probe via /integrate to see if the toddler 32-pair
recall changes from the 23/32 (71.9%) baseline.
"""
import base64
import json
import sys
import time
import urllib.request

BRAIN = "http://127.0.0.1:8095"
POOL_TEXT = 1
POOL_ACTION = 4

PAIRS = [
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
REPS_PER_PAIR = 8


def b64u(b):
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def post(path, body, timeout=15):
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{BRAIN}{path}", data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def main():
    print(f"[brain] stats:", json.dumps(post('/stats', {}))[:200])
    t0 = time.time()
    total_ticks = 0
    for prompt, response in PAIRS:
        for _ in range(REPS_PER_PAIR):
            post("/observe", {"pool_id": POOL_TEXT,
                              "frame": b64u(prompt.encode("utf-8"))})
            post("/observe", {"pool_id": POOL_ACTION,
                              "frame": b64u(response.encode("utf-8"))})
            post("/tick", {})
            total_ticks += 1
    dt = time.time() - t0
    print(f"[train] dense burst: 32 pairs x 8 reps = {total_ticks} ticks in {dt:.1f}s")
    print(f"[brain] stats:", json.dumps(post('/stats', {}))[:200])
    print()

    # Probe.
    print(f"  {'prompt':<10} {'expect':<10} {'reply':<48} {'verdict'}")
    print(f"  {'-'*10} {'-'*10} {'-'*48} {'-'*8}")
    hits = 0
    by_cat = {}
    for prompt, expected in PAIRS:
        post("/observe", {"pool_id": POOL_TEXT,
                          "frame": b64u(prompt.encode("utf-8"))})
        r = post("/integrate", {"query_pool": POOL_TEXT,
                                 "target_pool": POOL_ACTION})
        b64ans = r.get("answer") or ""
        try:
            ans_bytes = base64.urlsafe_b64decode(b64ans + "==")
            ans = ans_bytes.decode("utf-8", "replace")
        except Exception:
            ans = ""
        hit = expected in ans
        verdict = "HIT" if hit else "miss"
        if hit:
            hits += 1
        by_cat.setdefault(expected, [0, 0])
        by_cat[expected][1] += 1
        if hit:
            by_cat[expected][0] += 1
        print(f"  {prompt:<10} {expected:<10} {ans[:48]!r:<48} {verdict}")

    print()
    pct = 100.0 * hits / len(PAIRS)
    print(f"=== TOTAL: {hits}/{len(PAIRS)}  ({pct:.1f}%) ===")
    print(f"per category:")
    for cat, (h, t) in sorted(by_cat.items()):
        print(f"  {cat:<10} {h}/{t}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
