#!/usr/bin/env python3
"""scripts/brain_toddler_with_categorical.py

Measure whether the categorical_unified corpus lifts the 23/32
toddler ceiling, particularly the 0/5 food and 0/4 body misses.

We DO NOT retrain toddler here — the brain has already absorbed
the 8-epoch toddler training plus greetings plus the unified
categorical corpus.  We just re-probe.

Differs from brain_xpool_chat_test.py: that script trains then
probes via /integrate.  This one only probes — assumes the brain
is already trained.  Used to isolate the empirical question "did
the unified corpus help" from training noise.
"""
import base64
import json
import sys
import urllib.request

BRAIN = "http://127.0.0.1:8095"
POOL_TEXT   = 1
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


def b64u(b):
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def post(path, body, timeout=10):
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{BRAIN}{path}", data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def main():
    stats = post("/stats", {})
    print(f"[brain] tick={stats.get('tick')} bindings={stats.get('total_binding')} "
          f"neurons={stats.get('total_neurons')}")
    print()
    print(f"  {'prompt':<10} {'expect':<10} {'reply':<50} {'verdict'}")
    print(f"  {'-'*10} {'-'*10} {'-'*50} {'-'*8}")

    hits = 0
    by_cat = {}
    for prompt, expected in PAIRS:
        # Re-light the prompt in text pool.
        post("/observe", {"pool_id": POOL_TEXT,
                          "frame": b64u(prompt.encode("utf-8"))})
        # Cross-pool retrieve via /integrate (same path
        # brain_xpool_chat_test uses).
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
        print(f"  {prompt:<10} {expected:<10} {ans[:50]!r:<50} {verdict}")

    print()
    pct = 100.0 * hits / len(PAIRS)
    print(f"=== TOTAL: {hits}/{len(PAIRS)}  ({pct:.1f}%) ===")
    print(f"per category:")
    for cat, (h, t) in sorted(by_cat.items()):
        print(f"  {cat:<10} {h}/{t}")


if __name__ == "__main__":
    sys.exit(main() or 0)
