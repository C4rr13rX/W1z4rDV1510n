#!/usr/bin/env python3
"""
train_convos_atoms.py — raw-codepoint trainer for the conversational corpus.

Unlike train_conversations.py, this script:
  * skips the contrastive `propagate()` path (dead under raw-codepoint labels)
  * runs at LR=0.3 (not 7.5) so STDP order asymmetry survives — atom→atom
    saturation was the cause of character-soup output
  * uses a per-request HTTP timeout so a stalled socket can't freeze the run

Each training pair sends a single train_sequence with Q atoms at 50ms steps,
400ms Q→A gap, A atoms at 100ms steps. Repeated for --passes total.
"""
from __future__ import annotations
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_NODE   = "http://localhost:8090"
# Match the GA-experimental winning genome banked into runtime defaults.
DEFAULT_PASSES = 35
DEFAULT_LR     = 0.825

CONVERSATIONS: list[tuple[str, str]] = [
    ("hello",                   "Hello, I am W1z4rD. Ask me anything."),
    ("hello!",                  "Hello! Great to see you."),
    ("hi",                      "Hi there. How can I help?"),
    ("hi!",                     "Hi! What is on your mind?"),
    ("hey",                     "Hey. Ready to learn something new?"),
    ("howdy",                   "Howdy. Ask me anything."),
    ("greetings",               "Greetings. What would you like to know?"),
    ("good morning",            "Good morning. Hope your day is great."),
    ("good afternoon",          "Good afternoon. What can I help you with?"),
    ("good evening",            "Good evening. Ready to explore?"),
    ("good night",              "Good night. Rest well."),
    ("how are you",             "I am doing well, thanks for asking."),
    ("how are you?",            "Doing great! How about you?"),
    ("how is it going",         "Going well. What would you like to explore?"),
    ("what is up",              "Not much, here to help."),
    ("how do you do",           "I am doing well, thank you."),
    ("are you doing well",      "Yes, I am functioning well and ready to help."),
    ("who are you",             "I am W1z4rD, a distributed neural AI."),
    ("what are you",            "I am W1z4rD Vision, a Hebbian neural AI system."),
    ("what is your name",       "My name is W1z4rD."),
    ("tell me about yourself",  "I am W1z4rD, a distributed Hebbian neural AI."),
    ("what can you do",         "I can answer questions and learn from data."),
    ("are you an ai",           "Yes, I am W1z4rD, a Hebbian neural AI."),
    ("are you human",           "No, I am an AI. W1z4rD Vision."),
    ("are you smart",           "I learn from data and improve over time."),
    ("can you learn",           "Yes, I learn continuously from training data."),
    ("what do you know",        "I know what I have been trained on."),
    ("goodbye",                 "Goodbye. Take care."),
    ("bye",                     "Bye. See you later."),
    ("see you later",           "See you later. Be well."),
    ("thanks",                  "You are welcome."),
    ("thank you",               "You are very welcome."),
    ("yes",                     "Understood."),
    ("no",                      "Okay."),
    ("please",                  "Of course."),
    ("sorry",                   "No problem at all."),
]


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _post(url: str, payload: bytes, timeout: float = 15) -> dict:
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _span(tok: str, idx: int, total: int) -> list:
    return [{
        "text": tok, "role": "body", "bold": False, "italic": False,
        "size_ratio": 1.0, "x_frac": 0.5, "y_frac": 0.5,
        "seq_index": idx, "seq_total": total,
    }]


def _tokenize(text: str) -> list[str]:
    """Alnum runs + each non-alnum char as its own token-frame."""
    out, buf = [], []
    for ch in text:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf)); buf = []
            out.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def build_frames(question: str, answer: str) -> list[dict]:
    q_toks = _tokenize(question)
    a_toks = _tokenize(answer)
    total = len(q_toks) + len(a_toks)
    frames = []

    for j, tok in enumerate(q_toks):
        t = j * 0.05
        frames.append({
            "modality": "text", "t_secs": t, "lr_scale": 1.0,
            "data_b64": _b64(tok), "text": tok,
            "spans": _span(tok, j, total),
        })

    q_end = (len(q_toks) - 1) * 0.05 if len(q_toks) > 1 else 0.0
    a_start = q_end + 0.4
    for j, tok in enumerate(a_toks):
        t = a_start + j * 0.10
        frames.append({
            "modality": "text", "t_secs": t, "lr_scale": 1.0,
            "data_b64": _b64(tok), "text": tok,
            "spans": _span(tok, len(q_toks) + j, total),
        })

    return frames


def train_pair(node: str, q: str, a: str, passes: int, lr: float) -> bool:
    frames = build_frames(q, a)
    for i in range(passes):
        sid = str(uuid.uuid4())
        payload = json.dumps({
            "session_id": sid,
            "base_lr":    lr,
            "tau_secs":   2.0,
            "frames":     frames,
        }).encode()
        try:
            _post(f"{node}/media/train_sequence", payload, timeout=20)
        except Exception as exc:
            print(f"  pass {i+1}/{passes} failed: {exc}", file=sys.stderr)
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node",   default=DEFAULT_NODE)
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    ap.add_argument("--lr",     type=float, default=DEFAULT_LR)
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(f"{args.node}/health", timeout=5) as r:
            health = json.loads(r.read())
        print(f"Node online: {health.get('node_id', '?')}")
    except Exception as exc:
        print(f"ERROR: Node offline: {exc}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    total = len(CONVERSATIONS)
    ok = 0
    for i, (q, a) in enumerate(CONVERSATIONS, 1):
        print(f"[{i:3d}/{total}] {q!r:32s} -> {a[:50]!r}")
        sys.stdout.flush()
        if train_pair(args.node, q, a, args.passes, args.lr):
            ok += 1
        else:
            print("  FAILED")
    dt = time.time() - t0
    print(f"\nDone: {ok}/{total} pairs, {args.passes} passes @ LR={args.lr}, {dt:.1f}s total")


if __name__ == "__main__":
    main()
