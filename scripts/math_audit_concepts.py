#!/usr/bin/env python3
"""scripts/math_audit_concepts.py

Empirically test the hypothesis that K-12 response strings (like
'musical_instrument') never crystallize as concept neurons in the
action pool because the round-robin training schedule pushes their
recent_atoms sequence out of the window long before
concept_emergence_threshold (3) is met.

We exercise the brain server's /sensor/observe with TEXT prompts that
should force each action concept to ACTIVATE.  Then we inspect the
predictions response — but /sensor/observe only iterates
[POOL_TEXT, POOL_IMAGE, POOL_AUDIO], not POOL_ACTION.  So instead,
we infer concept existence indirectly:

  1. For each candidate response string, /chat with the PROMPT that
     should retrieve it.
  2. If /chat returns the exact response string, the action-pool
     concept exists (it was decoded as a whole concept).
  3. If /chat returns junk or a different category, the action-pool
     concept doesn't exist; selection routed to a TODDLER category
     concept that does exist.

We then test the "training schedule" hypothesis: if we train one
K-12 pair 6 times BACK-TO-BACK (dense burst) instead of round-robin,
its response WILL crystallize as a concept and /chat should return
it cleanly.
"""

import base64
import json
import sys
import time
import urllib.request

BRAIN = "http://127.0.0.1:8095"
POOL_TEXT   = 1
POOL_ACTION = 4


def b64u(b):
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def post(path, body, timeout=30):
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{BRAIN}{path}", data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def chat(text):
    r = post("/chat", {"text": text})
    return {
        "answer": (r.get("answer") or ""),
        "decoder": r.get("decoder"),
        "oog": (r.get("grounding") or {}).get("outside_grounding"),
    }


def train_dense(prompt, response, reps):
    """Train one pair `reps` times back-to-back — no other pairs in between."""
    for _ in range(reps):
        post("/observe", {"pool_id": POOL_TEXT,   "frame": b64u(prompt.encode("utf-8"))})
        post("/observe", {"pool_id": POOL_ACTION, "frame": b64u(response.encode("utf-8"))})
        post("/tick", {})


def main():
    stats = post("/stats", {})
    print(f"[brain] tick={stats.get('tick')} bindings={stats.get('total_binding')} "
            f"neurons={stats.get('total_neurons')}")
    print()

    # 1. Baseline: probe K-12 prompts that should have crystallized.
    print("[part 1] K-12 prompts (round-robin trained 6 reps over 7237 pairs)")
    for prompt in ["piano", "rose", "hammer", "guitar", "doctor"]:
        r = chat(prompt)
        print(f"  {prompt:>10} -> {r['answer'][:40]!r:<42} "
                f"dec={r['decoder']} oog={r['oog']}")

    # 2. Dense-burst training of a fresh K-12 pair the brain has NEVER
    # seen (audit by isolation).  We pick a synthetic prompt+response
    # so no atom overlap pre-exists.
    print()
    print("[part 2] dense-burst training: 'zyglop' -> 'flarble' x 12 reps back-to-back")
    train_dense("zyglop", "flarble", 12)
    # Now probe it.
    r = chat("zyglop")
    print(f"  zyglop -> {r['answer'][:60]!r}  dec={r['decoder']} oog={r['oog']}")

    # 3. Dense-burst train a REAL K-12 pair that round-robin couldn't
    # crystallize.  If the concept emerges and /chat returns it
    # cleanly, the hypothesis is confirmed: it's the schedule, not
    # the data volume.
    print()
    print("[part 3] dense-burst training: 'piano' -> 'musical_instrument' x 12 reps")
    train_dense("piano", "musical_instrument", 12)
    r = chat("piano")
    print(f"  piano -> {r['answer'][:60]!r}  dec={r['decoder']} oog={r['oog']}")

    print()
    print("[part 4] dense-burst: 'rose' -> 'plant' x 12 reps")
    train_dense("rose", "plant", 12)
    r = chat("rose")
    print(f"  rose  -> {r['answer'][:60]!r}  dec={r['decoder']} oog={r['oog']}")

    print()
    print("[part 5] dense-burst: 'hammer' -> 'tool' x 12 reps")
    train_dense("hammer", "tool", 12)
    r = chat("hammer")
    print(f"  hammer -> {r['answer'][:60]!r}  dec={r['decoder']} oog={r['oog']}")

    print()
    print("[part 6] verify toddler baseline still intact (no regression)")
    for prompt in ["dog", "red", "car", "tree"]:
        r = chat(prompt)
        print(f"  {prompt:>10} -> {r['answer'][:40]!r:<42} "
                f"dec={r['decoder']} oog={r['oog']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
