#!/usr/bin/env python3
"""brain_per_prompt_diag.py — per-prompt diagnostic against a running brain.

Expects the brain server already up on $W1Z4RD_BRAIN_PORT (default 8095)
with desired ControlMode env vars.  Trains toddler + categorical, then
prints every K-12 / integrate prompt that does NOT pass the
'any-trained-categorical' acceptance criterion.

Use this to understand which specific prompts a winning GA genome is
still missing — directs the next intervention.

Run:
  python scripts/brain_per_prompt_diag.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request

PORT = int(os.environ.get("W1Z4RD_BRAIN_PORT", "8095"))
BRAIN = f"http://127.0.0.1:{PORT}"
CORPUS = os.path.join(os.path.dirname(__file__), "..",
                     "data", "training", "categorical_unified_001.jsonl")


def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def post(path: str, body: dict, timeout: float = 10.0):
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        BRAIN + path, data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


TODDLER = [
    ("dog","animal"),("cat","animal"),("cow","animal"),("horse","animal"),
    ("bird","animal"),("fish","animal"),
    ("apple","food"),("banana","food"),("bread","food"),("cake","food"),("milk","food"),
    ("car","vehicle"),("truck","vehicle"),("bike","vehicle"),
    ("plane","vehicle"),("boat","vehicle"),
    ("red","color"),("blue","color"),("green","color"),("yellow","color"),
    ("ball","toy"),("doll","toy"),("kite","toy"),("drum","toy"),
    ("tree","nature"),("flower","nature"),("river","nature"),("mountain","nature"),
    ("hand","body"),("foot","body"),("eye","body"),("mouth","body"),
]

K12 = [
    "rose","oak","hammer","piano","triangle","football","tennis",
    "nine","sad","happy","doctor","school",
]


def main() -> int:
    # Sanity ping
    try:
        with urllib.request.urlopen(f"{BRAIN}/health", timeout=2) as r:
            if not r.read():
                print("brain server health check returned empty", flush=True)
                return 1
    except Exception as e:
        print(f"brain server not reachable on {BRAIN}: {e}", flush=True)
        return 1

    print(f"[diag] training toddler against {BRAIN}", flush=True)
    for p, r in TODDLER:
        for _ in range(8):
            post("/observe", {"pool_id": 1, "frame": b64(p)})
            post("/observe", {"pool_id": 4, "frame": b64(r)})
            post("/tick", {})

    print("[diag] training categorical", flush=True)
    t0 = time.time()
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            try: row = json.loads(line)
            except Exception: continue
            p = (row.get("prompt") or "").strip()
            resp = (row.get("response") or "").strip()
            if not p or not resp: continue
            for _ in range(3):
                post("/observe", {"pool_id": 1, "frame": b64(p)}, 5)
                post("/observe", {"pool_id": 4, "frame": b64(resp)}, 5)
                post("/tick", {}, 5)
    print(f"[diag] categorical trained in {time.time()-t0:.0f}s", flush=True)

    accepted: dict[str, set[str]] = {}
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            try: row = json.loads(line)
            except Exception: continue
            p = (row.get("prompt") or "").strip()
            resp = (row.get("response") or "").strip()
            if p and resp:
                accepted.setdefault(p, set()).add(resp)

    print("\n=== K-12 results ===", flush=True)
    k_hit = 0
    for p in K12:
        r = post("/chat", {"text": p})
        reply = (r or {}).get("reply", "")
        valid = accepted.get(p, set())
        ok = reply in valid
        if ok: k_hit += 1
        mark = "HIT" if ok else "MISS"
        print(f"  [{mark}] {p:10} reply={reply!r:25} trained={sorted(valid)}", flush=True)
    print(f"K-12: {k_hit}/{len(K12)}", flush=True)

    print("\n=== /integrate results ===", flush=True)
    i_hit = 0
    for p, exp in TODDLER:
        post("/observe", {"pool_id": 1, "frame": b64(p)})
        r = post("/integrate", {"query_pool": 1, "target_pool": 4})
        ans_b64 = (r or {}).get("answer", "") or ""
        try: ans = base64.urlsafe_b64decode(ans_b64 + "==").decode("utf-8", "replace")
        except Exception: ans = ""
        valid = accepted.get(p, set()) | {exp}
        ok = any(v and v in ans for v in valid)
        if ok: i_hit += 1
        mark = "HIT " if ok else "MISS"
        print(f"  [{mark}] {p:10} ans={ans[:40]!r:42} expected_or_trained={sorted(valid)}",
              flush=True)
    print(f"integrate: {i_hit}/{len(TODDLER)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
