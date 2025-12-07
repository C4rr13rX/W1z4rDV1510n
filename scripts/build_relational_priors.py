#!/usr/bin/env python3
"""
Builds generic relational priors from sequence data (any symbol set).

Input format (JSONL):
Each line is a sequence record:
{
  "sequence": [
    {
      "timestamp": {"unix": 123},
      "symbol_states": {
        "idA": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
        "idB": {"position": {"x": 3.0, "y": 1.0, "z": 0.0}}
      }
    },
    ...
  ],
  "metadata": {...}  // optional
}

The script computes:
- Coarse bins for positions
- Relational signatures between symbols (direction + distance buckets)
- Motif signatures per frame (small graph hashes)
- Transition tables P(next_motif | motif)
- Top-k destination bins per symbol role/type and motif context

Output: a JSON with priors suitable for lightweight CPU usage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math


def bin_position(pos: Dict[str, float], bins: int) -> Tuple[int, int]:
    return (
        int(math.floor(pos.get("x", 0.0) * bins)),
        int(math.floor(pos.get("y", 0.0) * bins)),
    )


def distance_bucket(a: Dict[str, float], b: Dict[str, float]) -> str:
    dx = a.get("x", 0.0) - b.get("x", 0.0)
    dy = a.get("y", 0.0) - b.get("y", 0.0)
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.5:
        return "near"
    if dist < 2.0:
        return "mid"
    return "far"


def direction_bucket(a: Dict[str, float], b: Dict[str, float]) -> str:
    dx = b.get("x", 0.0) - a.get("x", 0.0)
    dy = b.get("y", 0.0) - a.get("y", 0.0)
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    if abs(dy) > 0:
        return "north" if dy > 0 else "south"
    return "same"


def relational_signature(symbol_states: Dict[str, Dict]) -> List[str]:
    sig = []
    ids = sorted(symbol_states.keys())
    for i, first in enumerate(ids):
        for second in ids[i + 1 :]:
            a = symbol_states[first]["position"]
            b = symbol_states[second]["position"]
            sig.append(
                f"{first}->{second}:{direction_bucket(a,b)}:{distance_bucket(a,b)}"
            )
    return sig


def motif_hash(symbol_states: Dict[str, Dict]) -> str:
    parts = relational_signature(symbol_states)
    return "|".join(parts)


def update_counts(
    motifs: Counter,
    transitions: Counter,
    role_bins: Dict[str, Counter],
    role_pair_bins: Dict[str, Counter],
    current_states: Dict[str, Dict],
    next_states: Dict[str, Dict] | None,
    bins: int,
):
    current_motif = motif_hash(current_states)
    motifs[current_motif] += 1
    # role bins
    for sid, state in current_states.items():
        role_bins[sid].update([bin_position(state["position"], bins)])
    # pairwise bins
    ids = sorted(current_states.keys())
    for i, first in enumerate(ids):
        for second in ids[i + 1 :]:
            a = bin_position(current_states[first]["position"], bins)
            b = bin_position(current_states[second]["position"], bins)
            role_pair_bins[f"{first}|{second}"].update([(*a, *b)])
    if next_states is None:
        return
    next_motif = motif_hash(next_states)
    transitions[(current_motif, next_motif)] += 1

    # destination bins conditioned on motif
    for sid, state in next_states.items():
        role_bins[f"{sid}|next|{current_motif}"].update(
            [bin_position(state["position"], bins)]
        )


def finalize_topk(counter: Counter, k: int) -> List[Tuple[str, int]]:
    return counter.most_common(k)


def build_priors(
    path: Path, bins: int, topk: int, max_sequences: int | None
) -> Dict:
    motifs = Counter()
    transitions = Counter()
    role_bins: Dict[str, Counter] = defaultdict(Counter)
    role_pair_bins: Dict[str, Counter] = defaultdict(Counter)

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_sequences is not None and idx >= max_sequences:
                break
            record = json.loads(line)
            sequence = record.get("sequence", [])
            if len(sequence) < 1:
                continue
            for t in range(len(sequence)):
                current = sequence[t]["symbol_states"]
                nxt = sequence[t + 1]["symbol_states"] if t + 1 < len(sequence) else None
                update_counts(motifs, transitions, role_bins, role_pair_bins, current, nxt, bins)

    motif_total = sum(motifs.values()) or 1
    motif_probs = {k: v / motif_total for k, v in motifs.items()}

    transition_probs = {}
    outgoing = defaultdict(int)
    for (m1, m2), count in transitions.items():
        outgoing[m1] += count
    for (m1, m2), count in transitions.items():
        denom = outgoing[m1] or 1
        transition_probs[f"{m1}→{m2}"] = count / denom

    role_topk = {role: finalize_topk(c, topk) for role, c in role_bins.items()}
    role_pair_topk = {pair: finalize_topk(c, topk) for pair, c in role_pair_bins.items()}

    return {
        "bins": bins,
        "motif_probs": motif_probs,
        "transition_probs": transition_probs,
        "role_topk_bins": role_topk,
        "role_pair_topk_bins": role_pair_topk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSONL of sequences (generic schema).",
    )
    parser.add_argument("--output", type=Path, default=Path("data/relational_priors.json"))
    parser.add_argument("--bins", type=int, default=8, help="Grid bins per axis.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k bins to keep.")
    parser.add_argument(
        "--max-sequences", type=int, default=None, help="Optional cap for speed."
    )
    args = parser.parse_args()

    priors = build_priors(args.input, args.bins, args.topk, args.max_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(priors, handle, indent=2)
    print(f"Wrote priors to {args.output}")


if __name__ == "__main__":
    main()
