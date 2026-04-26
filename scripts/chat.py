#!/usr/bin/env python3
"""
chat.py — manual REPL for the W1z4rD node.

Sends each user message into the N-pool fabric via /multi_pool/ask and shows
the prediction from every connected pool, plus the legacy /chat answer (which
itself routes through multi_pool first, then falls back to the slow pool's
char-chain decoder).

Commands:
  /quit             — exit
  /help             — show commands
  /pools            — show currently registered pools and stats
  /train <q> | <a>  — quick teach: trains the (in -> out) pair for `q -> a`
  /reg <pool>       — register a new pool by name
  /pair <src> | <src text> | <tgt> | <tgt text>
                    — train a single pair across two named pools
  /raw <text>       — send a raw question via /multi_pool/ask only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post(node: str, path: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        node.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(node: str, path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(node.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read())


def show_pools(node: str) -> None:
    try:
        st = get(node, "/multi_pool/stats")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return
    print(f"  cross_edges total: {st.get('cross_edges', 0)}")
    for pool_id, info in sorted((st.get("pools") or {}).items()):
        print(f"    [{pool_id:>10}]  neurons={info.get('neurons', 0):>6}"
              f"  sequences={info.get('sequences', 0):>6}")


def chat(node: str, text: str, hops: int = 3, min_strength: float = 0.05) -> dict:
    """POST /chat — primary path is multi_pool; falls back to char_chain."""
    return post(node, "/chat", {
        "text": text, "hops": hops, "min_strength": min_strength,
    })


def multi_ask(node: str, src_pool: str, text: str,
              hops: int = 3, min_strength: float = 0.05) -> dict:
    return post(node, "/multi_pool/ask", {
        "src_pool": src_pool, "text": text,
        "hops": hops, "min_strength": min_strength,
    })


def quick_teach(node: str, q: str, a: str, passes: int = 30, lr: float = 0.5) -> dict:
    return post(node, "/multi_pool/train_pair", {
        "src_pool": "in",  "src": q,
        "tgt_pool": "out", "tgt": a,
        "passes": passes, "lr": lr,
    })


def register_pool(node: str, pool_id: str) -> dict:
    return post(node, "/multi_pool/register", {"pool_id": pool_id})


def pair_pools(node: str, src_pool: str, src: str,
               tgt_pool: str, tgt: str,
               passes: int = 30, lr: float = 0.5) -> dict:
    return post(node, "/multi_pool/train_pair", {
        "src_pool": src_pool, "src": src,
        "tgt_pool": tgt_pool, "tgt": tgt,
        "passes": passes, "lr": lr,
    })


HELP = """\
commands:
  /quit                                  — exit
  /pools                                 — show registered pools + counts
  /reg <pool>                            — register a new pool
  /train <q> | <a>                       — teach (in -> out) for q -> a
  /pair <src>|<text>|<tgt>|<text>        — teach across any two named pools
  /raw <text>                            — query via /multi_pool/ask
"""


def repl(node: str) -> None:
    print(f"connecting to {node} ...")
    try:
        h = get(node, "/health")
        print(f"  ok — node {h.get('node_id', '?')} uptime {h.get('uptime_secs', 0)}s")
    except Exception as exc:
        print(f"  WARN: node unreachable: {exc}")
        print("  responses will fail until node is up.")

    print("\nW1z4rD multi-pool chat. Type /help for commands, /quit to exit.\n")

    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        if raw in ("/quit", "/exit", "/q"):
            break
        if raw == "/help":
            print(HELP); continue
        if raw == "/pools":
            show_pools(node); continue
        if raw.startswith("/reg "):
            pool = raw[5:].strip()
            try:
                register_pool(node, pool)
                print(f"  registered pool {pool!r}")
            except Exception as exc:
                print(f"  ERROR: {exc}")
            continue
        if raw.startswith("/train "):
            body = raw[len("/train "):]
            if "|" not in body:
                print("  usage: /train <question> | <answer>"); continue
            q, a = (s.strip() for s in body.split("|", 1))
            if not q or not a:
                print("  question and answer must not be empty"); continue
            try:
                t0 = time.time()
                quick_teach(node, q, a)
                print(f"  trained ({time.time() - t0:.1f}s)")
            except Exception as exc:
                print(f"  ERROR: {exc}")
            continue
        if raw.startswith("/pair "):
            parts = [s.strip() for s in raw[len("/pair "):].split("|")]
            if len(parts) != 4 or not all(parts):
                print("  usage: /pair <src_pool> | <src text> | <tgt_pool> | <tgt text>")
                continue
            src_pool, src, tgt_pool, tgt = parts
            try:
                t0 = time.time()
                pair_pools(node, src_pool, src, tgt_pool, tgt)
                print(f"  paired {src_pool}↔{tgt_pool} ({time.time() - t0:.1f}s)")
            except Exception as exc:
                print(f"  ERROR: {exc}")
            continue
        if raw.startswith("/raw "):
            text = raw[len("/raw "):].strip()
            try:
                resp = multi_ask(node, "in", text)
                preds = resp.get("predictions") or {}
                if not preds:
                    print("  (no predictions)")
                else:
                    for pool_id, decoded in sorted(preds.items()):
                        print(f"  [{pool_id}] {decoded}")
            except Exception as exc:
                print(f"  ERROR: {exc}")
            continue

        # Plain text — go through /chat (multi_pool primary, char_chain fallback)
        try:
            resp = chat(node, raw)
        except urllib.error.HTTPError as exc:
            print(f"  HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}")
            continue
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        ans = resp.get("answer")
        decoder = resp.get("decoder", "?")
        if ans:
            print(f"w1z4rd> {ans}")
        else:
            print(f"w1z4rd> (no answer — decoder={decoder} reason={resp.get('reason', '')})")
        preds = resp.get("predictions") or {}
        # Surface any other pool's prediction (besides the main "out") so the
        # user sees the full multi-pool fan-out.
        extras = {k: v for k, v in preds.items() if k != "out" and v}
        if extras:
            for pool_id, decoded in sorted(extras.items()):
                print(f"          [{pool_id}] {decoded}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    try:
        repl(args.node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
