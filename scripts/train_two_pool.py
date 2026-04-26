#!/usr/bin/env python3
"""
train_two_pool.py — train the bidirectional two-pool associative memory.

Each pair is sent to /two_pool/train_pair which:
  * activates input atoms in pool_in (concept formation runs)
  * activates output atoms in pool_out (concept formation runs)
  * strengthens cross-pool synapses for every (in_active, out_active) pair

After training, runs the full forward pass through /two_pool/ask?direction=in_to_out
and the full reverse pass through /two_pool/ask?direction=out_to_in.

Reverse should retrieve the input given the output — that's the "send 'Hello!
I'm doing good.' to output, get 'Hi. How are you?' from input" behaviour.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_NODE   = "http://localhost:8090"
DEFAULT_PASSES = 30
DEFAULT_LR     = 0.3

CONVERSATIONS: list[tuple[str, str]] = [
    ("hello",                   "Hello, I am W1z4rD. Ask me anything."),
    ("hi",                      "Hi there. How can I help?"),
    ("hey",                     "Hey. Ready to learn something new?"),
    ("good morning",            "Good morning. Hope your day is great."),
    ("how are you",             "I am doing well, thanks for asking."),
    ("who are you",             "I am W1z4rD, a distributed neural AI."),
    ("what is your name",       "My name is W1z4rD."),
    ("what can you do",         "I can answer questions and learn from data."),
    ("are you an ai",           "Yes, I am W1z4rD, a Hebbian neural AI."),
    ("goodbye",                 "Goodbye. Take care."),
    ("thanks",                  "You are welcome."),
]


def _post(url: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node",   default=DEFAULT_NODE)
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    ap.add_argument("--lr",     type=float, default=DEFAULT_LR)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    try:
        health = _get(f"{args.node}/health")
        print(f"Node online: {health.get('node_id', '?')}")
    except Exception as exc:
        print(f"ERROR: Node offline: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_train:
        t0 = time.time()
        for i, (q, a) in enumerate(CONVERSATIONS, 1):
            payload = {
                "src_pool": "in",  "src": q,
                "tgt_pool": "out", "tgt": a,
                "passes": args.passes, "lr": args.lr,
            }
            try:
                resp = _post(f"{args.node}/multi_pool/train_pair", payload, timeout=60)
                stats = resp.get("stats", {}).get("pools", {})
                print(f"[{i:2d}/{len(CONVERSATIONS)}] {q!r} -> {a[:40]!r}  "
                      f"in={stats.get('in', {}).get('neurons', 0)} "
                      f"out={stats.get('out', {}).get('neurons', 0)} "
                      f"x={resp.get('stats', {}).get('cross_edges', 0)}")
                sys.stdout.flush()
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr)
        dt = time.time() - t0
        print(f"\nTraining done in {dt:.1f}s")

    # Forward test (in -> all other pools)
    print("\n=== Forward (in -> {out, ...}) ===")
    for q, expected in CONVERSATIONS:
        try:
            resp = _post(f"{args.node}/multi_pool/ask",
                {"src_pool": "in", "text": q}, timeout=15)
            preds = resp.get("predictions") or {}
            ans = preds.get("out", "(none)")
            print(f"  {q!r:30s} -> {ans!r}")
        except Exception as exc:
            print(f"  {q!r:30s} -> ERROR: {exc}")

    # Reverse test (out -> all other pools)
    print("\n=== Reverse (out -> {in, ...}) ===")
    for q, a in CONVERSATIONS:
        try:
            resp = _post(f"{args.node}/multi_pool/ask",
                {"src_pool": "out", "text": a}, timeout=15)
            preds = resp.get("predictions") or {}
            ans = preds.get("in", "(none)")
            print(f"  {a[:30]!r:32s} -> {ans!r}")
        except Exception as exc:
            print(f"  {a[:30]!r:32s} -> ERROR: {exc}")


if __name__ == "__main__":
    main()
