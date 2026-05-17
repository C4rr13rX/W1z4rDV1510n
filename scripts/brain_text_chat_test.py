#!/usr/bin/env python3
"""scripts/brain_text_chat_test.py — train + test chat through the new brain.

Trains simple Q->A text pairs against the multimodal brain server at
port 8095 using the low-level /observe + /tick endpoints (skipping
/sensor/observe_triple which is too expensive when image/audio bytes
are real-sized JPEGs and WAVs — that's a known follow-up).  Then
probes /chat with each Q to see what the brain produces.

The training pattern:
  POST /observe   pool_id=1 (text)   frame=base64url(prompt)
  POST /observe   pool_id=1 (text)   frame=base64url(response)
  POST /tick

Both prompt and response go into the SAME text pool back-to-back in
the same tick.  This lets the brain emerge prompt-then-response
sequence concepts, and /chat's same-pool generate() can then
retrieve them.

Run:
  python scripts/brain_text_chat_test.py
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.parse
import urllib.request


BRAIN = "http://127.0.0.1:8095"
TEXT_POOL = 1


# Curated Q->A toddler-level pairs.  Each is a short word + its
# 1-word noun-style answer.  Repeated training puts the (Q,A) byte
# sequence into the text pool's recent_atoms so the concept can emerge.
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


def train_pair(prompt: str, response: str) -> None:
    # Frame separator: a single newline byte between Q and A so the
    # concept emergence can pick up the boundary without conflating
    # word-internal sequences.
    text = f"{prompt} {response}"
    post("/observe", {"pool_id": TEXT_POOL, "frame": b64u(text.encode("utf-8"))})
    post("/tick", {})


def train(epochs: int = 6) -> None:
    print(f"[train] {len(PAIRS)} pairs x {epochs} epochs", flush=True)
    # Simple deterministic per-epoch rotation so concept-spanning
    # sequences across pairs don't accumulate to threshold in fixed
    # order.
    import random
    for ep in range(epochs):
        rng = random.Random(ep * 1009 + 7)
        order = list(range(len(PAIRS)))
        rng.shuffle(order)
        for i in order:
            q, a = PAIRS[i]
            train_pair(q, a)
        print(f"  epoch {ep+1}/{epochs} done", flush=True)


def chat_probe() -> None:
    print(f"\n[chat] probing /chat with each prompt", flush=True)
    print(f"  {'prompt':<10} -> reply", flush=True)
    print(f"  {'-'*10}    {'-'*40}", flush=True)
    for q, expected_a in PAIRS:
        try:
            r = post("/chat", {"text": q, "max_steps": 6})
        except Exception as e:
            print(f"  {q:<10} -> ERROR: {e}", flush=True)
            continue
        reply = r.get("reply", "")
        # Truncate ridiculous replies but show the actual bytes.
        if len(reply) > 60:
            reply = reply[:57] + "..."
        contains_expected = expected_a in reply
        marker = " [HIT]" if contains_expected else ""
        print(f"  {q:<10} -> {reply!r}{marker}", flush=True)


def main() -> int:
    # Brain stats before
    s = json.loads(urllib.request.urlopen(BRAIN + "/stats", timeout=10).read())
    print(f"[pre] stats: tick={s['tick']} neurons={s['total_neurons']} "
          f"concepts={s['total_concepts']} terminals={s['total_terminals']}",
          flush=True)

    train(epochs=8)

    s = json.loads(urllib.request.urlopen(BRAIN + "/stats", timeout=10).read())
    print(f"[post] stats: tick={s['tick']} neurons={s['total_neurons']} "
          f"concepts={s['total_concepts']} terminals={s['total_terminals']}",
          flush=True)

    chat_probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
