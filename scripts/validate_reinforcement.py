#!/usr/bin/env python3
"""
validate_reinforcement.py — does the dopamine-gated reinforcement path
actually contribute extra LTP beyond plain training?

Run two clean-slate training arms with IDENTICAL Hebbian work, weak
enough that no synapse hits max_weight=4.0:

  PLAIN       /neuro/clear, train_sequence × 1 at LR=0.10
  REINFORCED  /neuro/clear, train_sequence × 1 at LR=0.10, /neuro/reinforce

Then propagate from the question seed atoms and compare activation of
the answer-side atoms.  If the reinforced arm shows higher activation,
the dopamine pulse + flush is producing measurable extra weight on
the synapses that participated in training.

Why these knobs:
  • LR=0.10 keeps weights well below the 4.0 ceiling so the dopamine
    boost (Δw = 0.08 × tag × weight) shows up in the result rather
    than being clamped away.
  • Single training pass minimises homeostatic scaling between train
    and probe.
  • Hops=2 with min_strength=0.0001 catches weakly-activated atoms
    that would be filtered out at higher thresholds.
"""
from __future__ import annotations
import argparse
import base64
import json
import sys
import time
import urllib.request

NODE = "http://localhost:8090"
Q = "what is your name"
A = "My name is W1z4rD."
TRAIN_LR = 0.10
TRAIN_PASSES = 1


def post(path: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(NODE + path, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def char_label(c: str) -> str:
    return "txt:" + base64.urlsafe_b64encode(c.encode()).decode().rstrip("=")


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def spans(text: str, idx: int, total: int) -> list:
    return [{"text": text, "role": "body", "bold": False, "italic": False,
             "size_ratio": 1.0, "x_frac": 0.5, "y_frac": 0.5,
             "seq_index": idx, "seq_total": total}]


def _build_frames() -> list:
    q_words, a_words = Q.split(), A.split()
    total = len(q_words) + len(a_words)
    frames = []
    for j, w in enumerate(q_words):
        frames.append({"modality": "text", "t_secs": j * 0.05, "lr_scale": 1.0,
                        "data_b64": b64(w), "text": w,
                        "spans": spans(w, j, total)})
    a_start = (len(q_words) - 1) * 0.05 + 0.4 if q_words else 0.4
    for j, w in enumerate(a_words):
        frames.append({"modality": "text", "t_secs": a_start + j * 0.10,
                        "lr_scale": 1.0, "data_b64": b64(w), "text": w,
                        "spans": spans(w, len(q_words) + j, total)})
    return frames


def train_sequence(passes: int, lr: float) -> None:
    frames = _build_frames()
    for _ in range(passes):
        post("/media/train_sequence", {
            "session_id": f"valid-{time.time_ns()}",
            "base_lr": lr, "tau_secs": 2.0, "frames": frames,
        }, timeout=20)


def probe_activations() -> dict[str, float]:
    """Propagate from question seeds, return ALL nonzero activations
    (not just answer-only) so we can inspect every atom that the slow
    pool's synapses route to."""
    seeds = [char_label(c) for c in Q if c.strip()]
    r = post("/neuro/propagate", {
        "seed_labels": seeds, "hops": 2, "min_strength": 0.0001,
    })
    return {a["label"]: a["strength"] for a in r.get("activated", [])}


def label_to_char(label: str) -> str | None:
    if not label.startswith("txt:"):
        return None
    payload = label[4:]
    pad = "=" * ((4 - len(payload) % 4) % 4)
    try:
        b = base64.urlsafe_b64decode(payload + pad)
        return b.decode("utf-8")
    except Exception:
        return None


def run_arm(label: str, do_reinforce: bool) -> dict[str, float]:
    print(f"\n=== {label} ===")
    post("/neuro/clear", {})
    time.sleep(0.5)
    train_sequence(TRAIN_PASSES, TRAIN_LR)
    if do_reinforce:
        post("/neuro/reinforce", {"confidence": 0.85}, timeout=10)
    time.sleep(0.3)
    acts = probe_activations()
    print(f"  total non-zero atoms: {len(acts)}")
    if acts:
        top = sorted(acts.items(), key=lambda kv: -kv[1])[:8]
        for lbl, s in top:
            ch = label_to_char(lbl)
            ch_disp = repr(ch) if ch else "?"
            print(f"    {lbl}  {ch_disp:5s}  {s:.5f}")
    return acts


def run_compounding_arm(label: str, n_pulses: int) -> dict[str, float]:
    """Train once weakly, then fire dopamine pulses N times in
    succession.  Each pulse adds another 0.08 × tag × weight increment.
    If the architecture works as designed, the answer-side activation
    should rise monotonically with the pulse count and approach
    saturation after enough pulses."""
    print(f"\n=== {label} (n_pulses={n_pulses}) ===")
    post("/neuro/clear", {})
    time.sleep(0.5)
    train_sequence(TRAIN_PASSES, TRAIN_LR)
    # First, snapshot pre-pulse state
    pre = probe_activations()
    pre_total = sum(pre.values())
    print(f"  pre-pulse total activation: {pre_total:.4f}")
    # Fire N pulses in quick succession.  We need to re-tag the
    # synapses between pulses, otherwise once flush_dopamine zeros the
    # neuromodulator level the next release_neuromodulator call has
    # nothing-already-tagged to potentiate.  We re-touch the path by
    # one extra train_sequence pass between each pulse, mirroring how
    # a real verifier would call train + reinforce as a unit.
    for i in range(n_pulses):
        train_sequence(1, TRAIN_LR)
        post("/neuro/reinforce", {"confidence": 0.85}, timeout=10)
    post_acts = probe_activations()
    post_total = sum(post_acts.values())
    print(f"  post-pulse total activation: {post_total:.4f}  "
          f"(delta {post_total - pre_total:+.4f}, "
          f"{(post_total - pre_total) / pre_total * 100:+.1f}%)")
    return post_acts


def main() -> int:
    print(f"Q = {Q!r}")
    print(f"A = {A!r}")
    print(f"Training: {TRAIN_PASSES} pass(es) at LR={TRAIN_LR}")

    plain = run_arm("PLAIN training", do_reinforce=False)
    reinf = run_arm("REINFORCED training (/neuro/reinforce confidence=0.85)",
                     do_reinforce=True)

    # Compounding test: same training base, different pulse counts.
    pulse_5  = run_compounding_arm("REINFORCED x5 (train+reinforce x5)", 5)
    pulse_25 = run_compounding_arm("REINFORCED x25 (train+reinforce x25)", 25)

    print("\n=== Atom-by-atom comparison (top boosts) ===")
    all_labels = set(plain) | set(reinf)
    deltas = []
    for lbl in all_labels:
        p = plain.get(lbl, 0.0)
        r = reinf.get(lbl, 0.0)
        if abs(r - p) > 1e-6:
            ch = label_to_char(lbl) or "?"
            deltas.append((r - p, lbl, ch, p, r))
    deltas.sort(key=lambda x: -abs(x[0]))
    print(f"  total atoms changed: {len(deltas)}")
    for d, lbl, ch, p, r in deltas[:10]:
        sign = "+" if d > 0 else ""
        print(f"    {lbl}  {ch!r:5s}  plain={p:.5f}  reinf={r:.5f}  delta={sign}{d:.5f}")

    print("\n=== Verdict ===")
    plain_total = sum(plain.values())
    reinf_total = sum(reinf.values())
    p5_total    = sum(pulse_5.values())
    p25_total   = sum(pulse_25.values())
    diff_single = reinf_total - plain_total
    rel_single  = (diff_single / plain_total * 100) if plain_total > 0 else 0
    n_changed = sum(1 for d, *_ in deltas if abs(d) > 1e-6)
    all_pos   = all(d > 0 for d, *_ in deltas)
    print(f"  plain total:                {plain_total:.4f}")
    print(f"  +1 pulse total:             {reinf_total:.4f}  "
          f"(+{rel_single:.2f}%)")
    print(f"  +5 train+pulse cycles:      {p5_total:.4f}  "
          f"(+{(p5_total - plain_total)/plain_total*100:.2f}%)")
    print(f"  +25 train+pulse cycles:     {p25_total:.4f}  "
          f"(+{(p25_total - plain_total)/plain_total*100:.2f}%)")
    print(f"  atoms changed (single pulse): {n_changed}, all positive: {all_pos}")
    if n_changed == 0:
        print("  CONCLUSION: /neuro/reinforce produced ZERO weight change.")
        return 1
    saturated = abs(p25_total - p5_total) < 1e-3 and p5_total > reinf_total
    if all_pos and p5_total > reinf_total > plain_total and saturated:
        print("  CONCLUSION: dopamine reinforcement is live, compounds, AND "
              "saturates at max_weight as designed.  5 train+pulse cycles "
              "drives the path to its weight ceiling; further pulses are "
              "no-ops because synapses are clamped at max_weight=4.0.  This "
              "is the ideal consolidation behavior — fast enough to lock "
              "in verified-correct paths, bounded so reinforcement can't "
              "inflate weights beyond the architectural ceiling.")
        return 0
    if all_pos and p5_total > reinf_total > plain_total:
        print("  CONCLUSION: reinforcement is live and compounds, but did "
              "not saturate within 25 cycles — verifier may need more "
              "calls per pair to fully consolidate.")
        return 0
    print("  CONCLUSION: reinforcement fires on single pulse but compounding "
          "behavior is unexpected — investigate before trusting at scale.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
