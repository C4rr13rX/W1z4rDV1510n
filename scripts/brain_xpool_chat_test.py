#!/usr/bin/env python3
"""scripts/brain_xpool_chat_test.py — Q->A via cross-pool binding.

Trains prompts in pool 1 (text) and responses in pool 4 (action),
both observed at the SAME tick.  This is the architectural mode the
brain crate is designed for — cross-pool wiring at tick close grows
prompt-atom <-> response-atom terminals, plus Stage 3 concept->concept
terminals after the concepts emerge.

At query time:
  POST /observe pool_id=1 frame=Q     -- light up prompt atoms in text
  POST /integrate query_pool=1 target_pool=4
                                       -- propagate, density+pathway-
                                          boost select strongest target
                                          concept, decode to bytes
  -> Stage 5's concept-pathway boost is what should make C000-style
     specific bindings beat shared shorter atom-only concepts.

Run:
  python scripts/brain_xpool_chat_test.py
"""
from __future__ import annotations

import base64
import json
import random
import sys
import urllib.request


BRAIN = "http://127.0.0.1:8095"
POOL_TEXT   = 1
POOL_ACTION = 4


PAIRS: list[tuple[str, str]] = [
    ("dog",     "animal"),
    ("cat",     "animal"),
    ("cow",     "animal"),
    ("horse",   "animal"),
    ("bird",    "animal"),
    ("fish",    "animal"),
    ("apple",   "food"),
    ("banana",  "food"),
    ("bread",   "food"),
    ("cake",    "food"),
    ("milk",    "food"),
    ("car",     "vehicle"),
    ("truck",   "vehicle"),
    ("bike",    "vehicle"),
    ("plane",   "vehicle"),
    ("boat",    "vehicle"),
    ("red",     "color"),
    ("blue",    "color"),
    ("green",   "color"),
    ("yellow",  "color"),
    ("ball",    "toy"),
    ("doll",    "toy"),
    ("kite",    "toy"),
    ("drum",    "toy"),
    ("tree",    "nature"),
    ("flower",  "nature"),
    ("river",   "nature"),
    ("mountain","nature"),
    ("hand",    "body"),
    ("foot",    "body"),
    ("eye",     "body"),
    ("mouth",   "body"),
]


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def post(path: str, body: dict, timeout: float = 30.0) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BRAIN + path, data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def get(path: str) -> dict:
    with urllib.request.urlopen(BRAIN + path, timeout=10) as r:
        return json.loads(r.read())


def train_pair_xpool(prompt: str, response: str) -> None:
    """Observe prompt in text pool, response in action pool, then tick.
    This puts both atom sets into the same fabric.current.fired moment,
    so advance_tick wires cross-pool prompt<->response terminals."""
    post("/observe", {"pool_id": POOL_TEXT,   "frame": b64u(prompt.encode("utf-8"))})
    post("/observe", {"pool_id": POOL_ACTION, "frame": b64u(response.encode("utf-8"))})
    post("/tick", {})


def train(epochs: int = 8) -> None:
    print(f"[train] {len(PAIRS)} pairs x {epochs} epochs (cross-pool)",
          flush=True)
    for ep in range(epochs):
        rng = random.Random(ep * 1009 + 7)
        order = list(range(len(PAIRS)))
        rng.shuffle(order)
        for i in order:
            q, a = PAIRS[i]
            train_pair_xpool(q, a)
        print(f"  epoch {ep+1}/{epochs} done", flush=True)


def probe_xpool() -> None:
    """For each prompt: observe it in text pool, integrate text->action,
    decode answer.  Compare against expected response."""
    print(f"\n[probe] cross-pool integrate text->action", flush=True)
    print(f"  {'prompt':<10} {'expected':<10} reply", flush=True)
    print(f"  {'-'*10} {'-'*10} {'-'*40}", flush=True)
    hits_exact = 0
    hits_contain = 0
    for q, expected in PAIRS:
        post("/observe", {"pool_id": POOL_TEXT, "frame": b64u(q.encode("utf-8"))})
        r = post("/integrate", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
        ans_b64 = r.get("answer")
        if not ans_b64:
            reply = "(no answer)"
        else:
            # response uses URL_SAFE_NO_PAD
            pad = '=' * (-len(ans_b64) % 4)
            try:
                reply_bytes = base64.urlsafe_b64decode(ans_b64 + pad)
                reply = reply_bytes.decode("utf-8", "replace")
            except Exception as e:
                reply = f"(decode err: {e})"
        if reply == expected:
            hits_exact += 1
            marker = " [EXACT]"
        elif expected in reply:
            hits_contain += 1
            marker = " [CONTAINS]"
        else:
            marker = ""
        truncated = reply if len(reply) < 50 else reply[:47] + "..."
        print(f"  {q:<10} {expected:<10} {truncated!r}{marker}", flush=True)

    n = len(PAIRS)
    print(f"\n  exact:    {hits_exact}/{n}  ({100*hits_exact/n:.1f}%)", flush=True)
    print(f"  contains: {hits_contain + hits_exact}/{n}  "
          f"({100*(hits_contain + hits_exact)/n:.1f}%)", flush=True)


def main() -> int:
    s = get("/stats")
    print(f"[pre] tick={s['tick']} neurons={s['total_neurons']} "
          f"concepts={s['total_concepts']} terminals={s['total_terminals']}",
          flush=True)

    train(epochs=8)

    s = get("/stats")
    print(f"[post] tick={s['tick']} neurons={s['total_neurons']} "
          f"concepts={s['total_concepts']} terminals={s['total_terminals']} "
          f"binding={s['total_binding']}",
          flush=True)

    probe_xpool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
