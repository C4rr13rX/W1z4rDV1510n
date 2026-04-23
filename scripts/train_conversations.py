#!/usr/bin/env python3
"""
train_conversations.py — seed the Hebbian pool with conversational Q->A pairs.

Uses train_sequence (positive Hebbian association) + train_contrastive
(inhibitory suppression of wrong words) to overcome corpus-training bias.

Usage:
    python scripts/train_conversations.py [--node http://localhost:8090] [--rounds 3]
"""
from __future__ import annotations
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
import uuid

DEFAULT_NODE   = "http://localhost:8090"
DEFAULT_ROUNDS = 20   # full contrastive rounds per pair (saturates to max_weight at LR=0.50)
DEFAULT_PASSES = 10   # train_sequence passes per round
DEFAULT_LR     = 7.5  # saturation LR — 11 passes hits max_weight=4.0, dominates corpus noise
CONTRASTIVE_LR = 0.45


CONVERSATIONS: list[tuple[str, str]] = [
    # ── Greetings ──────────────────────────────────────────────────────────────
    ("hello",                   "Hello, I am W1z4rD. Ask me anything."),
    ("hello!",                  "Hello! Great to see you. What would you like to explore?"),
    ("hi",                      "Hi there. How can I help?"),
    ("hi!",                     "Hi! What is on your mind?"),
    ("hey",                     "Hey. Ready to learn something new?"),
    ("hey there",               "Hey! What can I help you with today?"),
    ("howdy",                   "Howdy. Ask me anything."),
    ("greetings",               "Greetings. What would you like to know?"),
    ("sup",                     "Not much, just processing. What is on your mind?"),
    ("good morning",            "Good morning. Hope your day is off to a great start."),
    ("good afternoon",          "Good afternoon. What can I help you explore?"),
    ("good evening",            "Good evening. Ready to dig into something interesting?"),
    ("good night",              "Good night. Rest well."),

    # ── How are you variants ───────────────────────────────────────────────────
    ("how are you",             "I am doing well, thanks for asking. What can I help you with?"),
    ("how are you?",            "Doing great! Ready to answer your questions. How about you?"),
    ("how is it going",         "Going well. What would you like to explore?"),
    ("what is up",              "Not much, here to help. What is on your mind?"),
    ("how do you do",           "I am doing well, thank you. What can I help you with?"),
    ("are you doing well",      "Yes, I am functioning well and ready to help. Ask me anything."),

    # ── Identity / self ────────────────────────────────────────────────────────
    ("who are you",             "I am W1z4rD, a distributed neural AI node that learns from training data and experience."),
    ("what are you",            "I am W1z4rD Vision, a neural AI system that uses Hebbian learning to understand and reason."),
    ("what is your name",       "My name is W1z4rD. What would you like to know?"),
    ("tell me about yourself",  "I am W1z4rD Vision, a distributed Hebbian neural AI. I learn from training data and get smarter as I am taught more."),
    ("what can you do",         "I can answer questions, reason about facts I have been trained on, and learn from your corrections."),
    ("are you an ai",           "Yes, I am W1z4rD, a Hebbian neural AI. I learn from data and improve over time."),
    ("are you human",           "No, I am an AI. W1z4rD Vision, a distributed neural system."),

    # ── Farewell ───────────────────────────────────────────────────────────────
    ("bye",                     "Goodbye. Feel free to come back anytime."),
    ("goodbye",                 "Goodbye. It was a pleasure."),
    ("see you later",           "See you later. Come back anytime."),
    ("take care",               "You too. Take care."),
    ("thanks",                  "You are welcome. Let me know if you need anything else."),
    ("thank you",               "You are welcome. Happy to help."),

    # ── Meta / capability ──────────────────────────────────────────────────────
    ("are you smart",           "I am learning. The more I am trained, the smarter I become."),
    ("can you learn",           "Yes, I learn from the question and answer pairs you train me on, and from the training curriculum."),
    ("do you understand me",    "I try to. My understanding comes from Hebbian associations built during training."),
    ("what do you know",        "I know what I have been trained on: science, language, facts, and whatever you teach me."),
]


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _spans(text: str, y: float, idx: int, total: int) -> list:
    return [{"text": text, "role": "body", "bold": False, "italic": False,
             "size_ratio": 1.0, "x_frac": 0.5, "y_frac": y,
             "seq_index": idx, "seq_total": total}]


def _post(url: str, payload: bytes, timeout: float = 20) -> dict:
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def propagate(node: str, question: str) -> list[str]:
    """Return current noisy activations for `question` — used as wrong answers."""
    q_labels = [f"txt:word_{w}" for w in question.lower().split() if len(w) > 2]
    if not q_labels:
        return []
    payload = json.dumps({"seed_labels": q_labels, "hops": 2}).encode()
    try:
        result = _post(f"{node}/neuro/propagate", payload, timeout=10)
        raw = result.get("activated") or []
        # API returns either a list of {label, strength} dicts or a {label: strength} dict
        if isinstance(raw, dict):
            items = [(l, s) for l, s in raw.items()]
        else:
            items = [(e["label"], e["strength"]) for e in raw if isinstance(e, dict)]
        # Keep only word labels, strip prefix, sort by strength desc
        words = sorted(
            [(l.removeprefix("txt:word_"), s) for l, s in items
             if l.startswith("txt:word_") and s > 0.05],
            key=lambda x: -x[1],
        )
        # Exclude question words themselves
        q_words = set(question.lower().split())
        return [w for w, _ in words if w not in q_words][:20]
    except Exception as exc:
        print(f"  propagate failed: {exc}", file=sys.stderr)
        return []


def _tokenize_chars(text: str) -> list[str]:
    """Split text into a stream of atomic token-frames: alnum runs (words) and
    each non-alnum character (space, comma, period, etc.) as its own frame.

    'Hello, world!' → ['Hello', ',', ' ', 'world', '!']

    Every space and punctuation character gets its own frame and its own
    timestamp so temporal STDP creates directional edges between word → space
    → word, word → comma → space → word, etc. — per the bottom-up architecture
    where every character is a neuron.
    """
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)  # space, comma, period — each its own frame
    if buf:
        tokens.append("".join(buf))
    return tokens


def train_sequence_passes(node: str, question: str, answer: str, passes: int, lr: float,
                          contrastive_lr: float = CONTRASTIVE_LR) -> bool:
    # Per-character-run frames: each word AND each non-alnum char (space,
    # comma, period, exclaim, etc.) is its own timed frame. STDP creates
    # directional edges between every adjacent token pair.
    # Q tokens spaced 50ms apart; A tokens spaced 100ms apart with 400ms Q→A gap.
    # tau_secs=2.0 → W(token_i→token_i+1) = lr × exp(-0.05) ≈ 0.951 directed edge.
    q_toks = _tokenize_chars(question)
    a_toks = _tokenize_chars(answer)
    q_total = len(q_toks)
    a_total = len(a_toks)
    total_frames = q_total + a_total

    frames = []
    seq_index = 0
    for j, tok in enumerate(q_toks):
        t = j * 0.05
        frames.append({
            "modality": "text", "t_secs": t, "lr_scale": 1.0,
            "data_b64": _b64(tok), "text": tok,
            "pool_namespace": "q",   # Q-pool: labels become q:txt:word_* / q:txt:punct_*
            "spans": _spans(tok, t / max((q_total + a_total - 1) * 0.15, 0.001),
                            seq_index, total_frames),
        })
        seq_index += 1

    q_end = (q_total - 1) * 0.05 if q_total > 1 else 0.0
    a_start = q_end + 0.4  # 400ms gap between question end and answer start
    for j, tok in enumerate(a_toks):
        t = a_start + j * 0.10
        frames.append({
            "modality": "text", "t_secs": t, "lr_scale": 1.0,
            "data_b64": _b64(tok), "text": tok,
            "pool_namespace": "a",   # A-pool: labels become a:txt:word_* / a:txt:punct_*
            "spans": _spans(tok, t / max((a_start + (a_total - 1) * 0.10), 0.001),
                            seq_index, total_frames),
        })
        seq_index += 1

    for i in range(passes):
        sid = str(uuid.uuid4())
        payload = json.dumps({
            "session_id": sid,
            "base_lr":    lr,
            "tau_secs":   2.0,
            "frames":     frames,
        }).encode()
        try:
            _post(f"{node}/media/train_sequence", payload)
        except Exception as exc:
            print(f"  seq pass {i+1} failed: {exc}", file=sys.stderr)
            return False
    return True


def train_contrastive_pass(node: str, question: str, correct: str,
                            wrong_words: list[str], lr: float) -> bool:
    if not wrong_words:
        return True
    wrong_answers = [" ".join(wrong_words[:10])]  # group top noise words as one wrong "answer"
    payload = json.dumps({
        "question":       question,
        "correct_answer": correct,
        "wrong_answers":  wrong_answers,
        "lr_scale":       lr,
    }).encode()
    try:
        _post(f"{node}/media/train_contrastive", payload)
        return True
    except Exception as exc:
        print(f"  contrastive failed: {exc}", file=sys.stderr)
        return False


def train_pair(node: str, question: str, answer: str, rounds: int,
               seq_passes: int = DEFAULT_PASSES, lr: float = DEFAULT_LR) -> bool:
    for rnd in range(rounds):
        if not train_sequence_passes(node, question, answer, seq_passes, lr):
            return False
        noisy = propagate(node, question)
        correct_words = set(answer.lower().split())
        noisy = [w for w in noisy if w not in correct_words]
        if noisy:
            train_contrastive_pass(node, question, answer, noisy, CONTRASTIVE_LR)
    return True


def main():
    ap = argparse.ArgumentParser(description="Seed conversational training data")
    ap.add_argument("--node",   default=DEFAULT_NODE)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                    help="train_sequence passes per round (default %(default)s)")
    ap.add_argument("--lr",     type=float, default=DEFAULT_LR,
                    help="sequence learning rate (default %(default)s; 7.5 saturates to max_weight in ~11 passes)")
    args = ap.parse_args()

    # Health check
    try:
        with urllib.request.urlopen(f"{args.node}/health", timeout=5) as r:
            health = json.loads(r.read())
        print(f"Node online: {health.get('node_id', '?')}")
    except Exception as exc:
        print(f"ERROR: Node offline: {exc}", file=sys.stderr)
        sys.exit(1)

    total = len(CONVERSATIONS)
    ok = 0
    for i, (q, a) in enumerate(CONVERSATIONS, 1):
        print(f"[{i:3d}/{total}] {q!r:40s} -> {a[:50]!r}")
        sys.stdout.flush()
        if train_pair(args.node, q, a, args.rounds, args.passes, args.lr):
            ok += 1
        else:
            print(f"  FAILED")

    print(f"\nDone: {ok}/{total} pairs ({args.rounds} rounds x {args.passes} seq passes @ LR={args.lr} + contrastive)")


if __name__ == "__main__":
    main()
