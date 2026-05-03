#!/usr/bin/env python3
"""
stress_test_scenarios.py — adversarial interference tests for the W1z4rD
fabric.  Each scenario is a controlled experiment that trains a base
corpus, measures recall, trains a deliberately-interfering corpus, then
re-measures recall.  The retention rate is the answer to "did new
training overwrite what we already learned?"

Each scenario runs four arms, all on a freshly-cleared pool, so we can
attribute interference effects to architecture rather than confounds:

  arm A  slow-pool train, no reinforcement       — baseline char-chain
  arm B  slow-pool train, +reinforce on base     — does dopamine
                                                   consolidation protect
                                                   the slow pool?
  arm C  multi-pool train, no reinforcement      — concept-bound
                                                   greetings should be
                                                   inherently more robust
  arm D  multi-pool train, +reinforce on base    — belt and suspenders

Pre-interference and post-interference pass rates are reported per arm
along with the per-pair degradation list so we can see exactly which
pairs lost the fight.
"""
from __future__ import annotations
import argparse
import base64
import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

NODE = "http://localhost:8090"

# ── Scenario data ───────────────────────────────────────────────────────────

# Base corpus: a focused set of greetings that we expect to retain.
BASE_PAIRS: list[tuple[str, str]] = [
    ("hello",                   "Hello, I am W1z4rD. Ask me anything."),
    ("hi",                      "Hi there. How can I help?"),
    ("how are you",             "I am doing well, thanks for asking."),
    ("who are you",             "I am W1z4rD, a distributed neural AI."),
    ("what is your name",       "My name is W1z4rD."),
    ("what can you do",         "I can answer questions and learn from data."),
    ("are you an ai",           "Yes, I am W1z4rD, a Hebbian neural AI."),
    ("goodbye",                 "Goodbye. Take care."),
    ("thanks",                  "You are welcome."),
    ("good morning",            "Good morning. Hope your day is great."),
]

# Interfering corpus: code-corpus-style pairs heavy in structural
# markers (`\n`, `{`, `}`, `(`, `)`, `;`, `,`).  Designed to maximally
# inflate hub fan-in for those punctuation atoms — exactly the
# structural-marker dominance we diagnosed earlier.
INTERFERENCE_PAIRS: list[tuple[str, str]] = [
    ("declare a javascript variable",
     "let x = 5;"),
    ("define a function in python",
     "def foo(x):\n    return x + 1"),
    ("import json in python",
     "import json"),
    ("declare a const in javascript",
     "const name = 'value';"),
    ("create a list in python",
     "items = [1, 2, 3]"),
    ("create an object in javascript",
     "const obj = { key: 'value' };"),
    ("python for loop",
     "for i in range(10):\n    print(i)"),
    ("javascript array map",
     "arr.map((x) => x * 2);"),
    ("python dict",
     "d = { 'a': 1, 'b': 2 }"),
    ("read a file in python",
     "with open('file.txt') as f:\n    data = f.read()"),
    ("javascript fetch",
     "fetch('/api').then((r) => r.json());"),
    ("python list comprehension",
     "[x * 2 for x in range(10)]"),
    ("javascript arrow function",
     "const fn = (x) => x + 1;"),
    ("python class",
     "class Foo:\n    def __init__(self):\n        self.x = 0"),
    ("javascript class",
     "class Foo { constructor() { this.x = 0; } }"),
    ("typescript interface",
     "interface User { name: string; age: number; }"),
    ("python try except",
     "try:\n    x = 1\nexcept Exception as e:\n    print(e)"),
    ("javascript promise",
     "new Promise((resolve, reject) => { resolve(1); });"),
    ("python f string",
     "name = 'Bob'\nprint(f'Hello {name}')"),
    ("javascript template literal",
     "const s = `Hello ${name}`;"),
]


# ── HTTP helpers ────────────────────────────────────────────────────────────

def post(path: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(NODE + path, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:  # type: ignore[name-defined]
        return {"error": str(exc)}


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def spans(text: str, idx: int, total: int) -> list:
    return [{"text": text, "role": "body", "bold": False, "italic": False,
             "size_ratio": 1.0, "x_frac": 0.5, "y_frac": 0.5,
             "seq_index": idx, "seq_total": total}]


def _frames(q: str, a: str) -> list:
    q_words, a_words = q.split(), a.split()
    total = len(q_words) + len(a_words)
    out = []
    for j, w in enumerate(q_words):
        out.append({"modality": "text", "t_secs": j * 0.05, "lr_scale": 1.0,
                     "data_b64": b64(w), "text": w,
                     "spans": spans(w, j, total)})
    a_start = (len(q_words) - 1) * 0.05 + 0.4 if q_words else 0.4
    for j, w in enumerate(a_words):
        out.append({"modality": "text", "t_secs": a_start + j * 0.10,
                     "lr_scale": 1.0, "data_b64": b64(w), "text": w,
                     "spans": spans(w, len(q_words) + j, total)})
    return out


# ── Training primitives ─────────────────────────────────────────────────────

def train_slow(q: str, a: str, *, lr: float = 7.5, passes: int = 5) -> None:
    frames = _frames(q, a)
    for _ in range(passes):
        post("/media/train_sequence", {
            "session_id": str(uuid.uuid4()),
            "base_lr": lr, "tau_secs": 2.0, "frames": frames,
        }, timeout=20)


def train_multi(q: str, a: str, *, passes: int = 10) -> None:
    # 10 passes is enough to seat the concept binding for a stress
    # test; the GA-tuned default of 35 is for production saturation.
    post("/multi_pool/train_pair", {
        "src_pool": "in", "src": q, "tgt_pool": "out", "tgt": a,
        "passes": passes,
    }, timeout=60)


def reinforce_pair(q: str, a: str, *, cycles: int = 5) -> None:
    """train+reinforce cycles: train re-tags traces, reinforce captures
    them.  5 cycles saturates per the validation experiment."""
    frames = _frames(q, a)
    for _ in range(cycles):
        post("/media/train_sequence", {
            "session_id": str(uuid.uuid4()),
            "base_lr": 0.40, "tau_secs": 2.0, "frames": frames,
        }, timeout=20)
        post("/neuro/reinforce", {"confidence": 0.85}, timeout=10)


# ── Recall measurement ──────────────────────────────────────────────────────

_NORM_RE = re.compile(r"\s+")
_STOPWORDS = {"a","an","the","is","i","am","are","you","to","of","in","on",
               "at","for","and","or","with","my","your","me","yes","no",
               "ok","ask","what","who","how"}


def norm(s: str) -> str:
    return _NORM_RE.sub(" ", (s or "").strip().lower())


def key_phrases(answer: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", answer)
    keep = [w for w in words if len(w) >= 4 and w.lower() not in _STOPWORDS]
    if not keep:
        keep = [answer]
    keep.sort(key=len, reverse=True)
    return keep[:3]


def recall_check(q: str, a: str) -> tuple[bool, str, str]:
    r = post("/chat", {"text": q, "hops": 2, "min_strength": 0.05}, timeout=10)
    actual = r.get("answer") or ""
    decoder = r.get("decoder", "?")
    a_norm = norm(actual)
    phrases = key_phrases(a)
    passed = any(norm(p) in a_norm for p in phrases)
    return passed, actual, decoder


def measure_battery(label: str, pairs: list[tuple[str, str]]) -> dict:
    results = []
    for q, a in pairs:
        p, actual, decoder = recall_check(q, a)
        results.append({"q": q, "a": a, "actual": actual, "decoder": decoder,
                         "pass": p})
    passed = sum(1 for r in results if r["pass"])
    decoder_counts: dict[str, int] = {}
    for r in results:
        decoder_counts[r["decoder"]] = decoder_counts.get(r["decoder"], 0) + 1
    return {
        "label": label,
        "total": len(pairs), "passed": passed,
        "pass_rate": passed / len(pairs),
        "decoders": decoder_counts,
        "results": results,
    }


def print_summary(m: dict) -> None:
    print(f"  {m['label']:35s}  {m['passed']}/{m['total']}  "
          f"({m['pass_rate']*100:5.1f}%)  decoders={m['decoders']}")


# ── Arms ────────────────────────────────────────────────────────────────────

def run_arm(name: str, *, train_fn, reinforce_after_base: bool) -> dict:
    print(f"\n-------- ARM: {name} --------")
    print("  clearing pool...")
    post("/neuro/clear", {})
    time.sleep(0.5)

    print(f"  training base corpus ({len(BASE_PAIRS)} pairs)...")
    for q, a in BASE_PAIRS:
        train_fn(q, a)

    if reinforce_after_base:
        print(f"  reinforcing base corpus (5 cycles each)...")
        for q, a in BASE_PAIRS:
            reinforce_pair(q, a)

    print(f"  measuring base recall (PRE-interference)...")
    pre = measure_battery("PRE", BASE_PAIRS)
    print_summary(pre)

    print(f"  training interference corpus "
          f"({len(INTERFERENCE_PAIRS)} pairs)...")
    for q, a in INTERFERENCE_PAIRS:
        train_fn(q, a)

    print(f"  measuring base recall (POST-interference)...")
    post_m = measure_battery("POST", BASE_PAIRS)
    print_summary(post_m)

    retention = (post_m["pass_rate"] / pre["pass_rate"]) if pre["pass_rate"] > 0 else 0.0
    print(f"  retention: {retention * 100:.1f}%")

    # List pairs that regressed (passed pre, fail post).
    pre_idx = {r["q"]: r for r in pre["results"]}
    post_idx = {r["q"]: r for r in post_m["results"]}
    regressed = [q for q in pre_idx
                 if pre_idx[q]["pass"] and not post_idx[q]["pass"]]
    if regressed:
        print(f"  regressed pairs ({len(regressed)}):")
        for q in regressed:
            actual_post = post_idx[q]["actual"][:60].replace("\n", "\\n")
            print(f"    {q!r:30s} -> {actual_post!r}")

    return {
        "name": name,
        "pre_pass_rate": pre["pass_rate"],
        "post_pass_rate": post_m["pass_rate"],
        "retention": retention,
        "regressed": regressed,
        "pre_decoders": pre["decoders"],
        "post_decoders": post_m["decoders"],
    }


def main() -> int:
    global NODE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", default=NODE)
    ap.add_argument("--arm", default="all",
                     choices=["all", "slow", "slow_reinf", "multi",
                              "multi_reinf"],
                     help="Which arm to run (default: all)")
    args = ap.parse_args()
    NODE = args.node

    h = post("/health", {})
    if h.get("error"):
        try:
            with urllib.request.urlopen(NODE + "/health", timeout=3) as r:
                h = json.loads(r.read())
        except Exception as exc:
            print(f"ERROR: node offline: {exc}", file=sys.stderr)
            return 1
    print(f"Node: {h.get('node_id', '?')}")
    print(f"Base corpus:        {len(BASE_PAIRS)} pairs")
    print(f"Interference corpus:{len(INTERFERENCE_PAIRS)} pairs")

    arms = []
    selected = args.arm
    if selected in ("all", "slow"):
        arms.append(run_arm("A: slow-pool, no reinforcement",
                             train_fn=lambda q, a: train_slow(q, a),
                             reinforce_after_base=False))
    if selected in ("all", "slow_reinf"):
        arms.append(run_arm("B: slow-pool, +reinforce on base",
                             train_fn=lambda q, a: train_slow(q, a),
                             reinforce_after_base=True))
    if selected in ("all", "multi"):
        arms.append(run_arm("C: multi-pool, no reinforcement",
                             train_fn=lambda q, a: train_multi(q, a),
                             reinforce_after_base=False))
    if selected in ("all", "multi_reinf"):
        arms.append(run_arm("D: multi-pool, +reinforce on base",
                             train_fn=lambda q, a: train_multi(q, a),
                             reinforce_after_base=True))

    print("\n========================= SUMMARY =========================")
    print(f"{'arm':40s}  {'pre%':>6s}  {'post%':>6s}  {'retention':>10s}  "
          f"{'regressed':>10s}")
    for a in arms:
        print(f"{a['name']:40s}  "
              f"{a['pre_pass_rate']*100:5.1f}%  "
              f"{a['post_pass_rate']*100:5.1f}%  "
              f"{a['retention']*100:8.1f}%  "
              f"{len(a['regressed']):>10d}")
    print("===========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
