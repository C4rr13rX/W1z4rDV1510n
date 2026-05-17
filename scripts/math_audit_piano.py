#!/usr/bin/env python3
"""scripts/math_audit_piano.py

Forensic math walk-through for: why does `piano` query route to
`vehicle` even after K-12 training?

We have:
  - toddler 32-pair corpus, trained 8 epochs = 8 reps each pair
  - greetings_001 (186 pairs), trained 6 reps
  - k12_subjects_001 (7237 pairs), trained 6 reps (3 + 3)

Each (prompt, response) pair contributes axon weight from every
text-pool atom in `prompt` to every action-pool atom in `response`,
linearly with rep count.

We count co-firings per (text_atom, action_atom) pair to estimate
relative axon weights, then compute predicted action-pool atom
activations when piano = {p, i, a, n, o} fires in the text pool.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TODDLER_PAIRS = [
    ("dog", "animal"), ("cat", "animal"), ("cow", "animal"),
    ("horse", "animal"), ("bird", "animal"), ("fish", "animal"),
    ("apple", "food"), ("banana", "food"), ("bread", "food"),
    ("cake", "food"), ("milk", "food"),
    ("car", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"),
    ("plane", "vehicle"), ("boat", "vehicle"),
    ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"),
    ("ball", "toy"), ("doll", "toy"), ("kite", "toy"), ("drum", "toy"),
    ("tree", "nature"), ("flower", "nature"), ("river", "nature"), ("mountain", "nature"),
    ("hand", "body"), ("foot", "body"), ("eye", "body"), ("mouth", "body"),
]
TODDLER_REPS = 8

GREETINGS_PATH = ROOT / "data" / "training" / "greetings_001.jsonl"
K12_PATH       = ROOT / "data" / "training" / "k12_subjects_001.jsonl"


def load_jsonl(p):
    out = []
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def axon_counts(pairs, reps, axon):
    """axon[(text_byte, action_byte)] += reps for each co-occurring pair of bytes."""
    for prompt, response in pairs:
        if not prompt or not response:
            continue
        tset = set(prompt.encode("utf-8"))
        aset = set(response.encode("utf-8"))
        for tb in tset:
            for ab in aset:
                axon[(tb, ab)] += reps


def main():
    axon = defaultdict(int)

    # Toddler 32-pair x 8 reps
    axon_counts(TODDLER_PAIRS, TODDLER_REPS, axon)

    # Greetings 6 reps
    greet = [(r["prompt"], r["response"]) for r in load_jsonl(GREETINGS_PATH)]
    axon_counts(greet, 6, axon)

    # K-12 6 reps (3 reps + 3 reps cumulative as per current training state)
    k12 = [(r["prompt"], r["response"]) for r in load_jsonl(K12_PATH)]
    axon_counts(k12, 6, axon)

    print(f"[corpus] toddler={len(TODDLER_PAIRS)} greet={len(greet)} k12={len(k12)}")
    print(f"[axon] distinct (text_byte, action_byte) pairs: {len(axon)}")

    # ─── Query: piano = {p, i, a, n, o} ─────────────────────────────
    query_bytes = list(set(b"piano"))
    print(f"\n[query] piano bytes = {query_bytes!r} = {[chr(b) for b in query_bytes]}")

    # Compute predicted activation per action-pool atom.
    # In the brain's propagate(), each text atom contributes:
    #   contribution = source_activation * effective_weight * hop_decay / sqrt(fan_out)
    # Source activation is ~1.0 right after firing.  hop_decay defaults
    # to ~0.85.  We approximate effective_weight ∝ axon_count.  We
    # IGNORE fan-out for this first pass to keep the math interpretable;
    # we'll add it once we see the unnormalized picture.
    action_act = defaultdict(float)
    HOP_DECAY = 0.85
    for tb in query_bytes:
        for (t, a), w in axon.items():
            if t != tb:
                continue
            action_act[a] += float(w) * HOP_DECAY  # unit src_activation, no fan-out yet

    # Rank action atoms by predicted activation.
    print(f"\n[predicted action-pool atom activations from piano firing]")
    print("  (axon-count weighted, hop_decay=0.85, no fan-out normalization)")
    top = sorted(action_act.items(), key=lambda kv: -kv[1])[:25]
    for ab, act in top:
        ch = chr(ab) if 32 <= ab < 127 else f"\\x{ab:02x}"
        print(f"    {ch!r:>6}  act={act:>8.1f}")

    # ─── Score the candidate target concepts ────────────────────────
    # The action pool's most relevant concept neurons are the *response
    # words themselves* — "vehicle", "musical_instrument", etc.  Each
    # one's avg_member_act = mean activation across its member atoms.
    candidates = {
        "animal":             "animal",
        "vehicle":            "vehicle",
        "color":              "color",
        "food":               "food",
        "toy":                "toy",
        "nature":             "nature",
        "body":               "body",
        "musical_instrument": "musical_instrument",
        "plant":              "plant",
        "shape":              "shape",
        "tool":               "tool",
        "sport":              "sport",
        "people":             "people",
        "number":             "number",
        "emotion":            "emotion",
    }
    print(f"\n[candidate target concepts — avg_member_act x sqrt(len) x info]")
    rows = []
    for name, response in candidates.items():
        bs = list(response.encode("utf-8"))
        unique_bs = list(set(bs))
        if not bs:
            continue
        member_sum = sum(action_act.get(b, 0.0) for b in bs)
        avg_member_act = member_sum / len(bs)
        length_factor = len(bs) ** 0.5
        repetition = len(bs) - len(unique_bs)
        info_factor = max(0.1, len(unique_bs) - 0.3 * repetition)
        # The Stage 7 binding boost matters but we will compute that
        # separately below.  Here we show the no-binding score.
        score_no_binding = avg_member_act * length_factor * info_factor
        rows.append((name, avg_member_act, length_factor, info_factor, score_no_binding))
    rows.sort(key=lambda r: -r[4])
    print(f"    {'concept':<22} {'avg_act':>9} {'len_f':>6} {'info_f':>6} {'score':>10}")
    for r in rows:
        print(f"    {r[0]:<22} {r[1]:>9.2f} {r[2]:>6.2f} {r[3]:>6.2f} {r[4]:>10.2f}")

    # ─── Expected vs actual ─────────────────────────────────────────
    print("\n[expected vs actual]")
    print("    Stage 7 binding-pool routing SHOULD apply an 8x boost to")
    print("    'musical_instrument' if the (piano, musical_instrument)")
    print("    binding has best precisionxrecall over the query atoms.")
    print()
    print("    Best binding match for piano query:")
    print("      piano -> musical_instrument: query atoms = {p,i,a,n,o}, intersect = 5/5 = 1.0 precision")
    print("      plane -> vehicle:            query atoms = {p,l,a,n,e}, intersect = {p,a,n} = 3/5 = 0.6")
    print("      bike  -> vehicle:            query atoms = {b,i,k,e},   intersect = {i}     = 1/4 = 0.25")
    print()
    print("    => piano binding wins; binding_target_atoms = unique bytes of 'musical_instrument'")
    print()

    # Compute binding_boost for each candidate against musical_instrument binding's target atoms.
    bind_target = set("musical_instrument".encode("utf-8"))
    print(f"    binding_target_atoms = unique bytes of 'musical_instrument' = "
          f"{sorted([chr(b) for b in bind_target])}")
    print()
    print(f"    {'concept':<22} {'coverage':>9} {'boost':>6} {'final_score':>13}")
    for name, response in candidates.items():
        bs = list(response.encode("utf-8"))
        if not bs:
            continue
        bs_set = set(bs)
        matched = len(bind_target & bs_set)
        coverage = matched / len(bind_target)
        if coverage >= 0.99:
            boost = 8.0
        elif coverage >= 0.75:
            boost = 3.0
        elif coverage >= 0.5:
            boost = 1.5
        else:
            boost = 1.0
        member_sum = sum(action_act.get(b, 0.0) for b in bs)
        avg_member_act = member_sum / len(bs)
        length_factor = len(bs) ** 0.5
        unique_bs = set(bs)
        repetition = len(bs) - len(unique_bs)
        info_factor = max(0.1, len(unique_bs) - 0.3 * repetition)
        final = avg_member_act * length_factor * info_factor * boost
        print(f"    {name:<22} {coverage:>9.2f} {boost:>6.2f} {final:>13.2f}")


if __name__ == "__main__":
    sys.exit(main())
