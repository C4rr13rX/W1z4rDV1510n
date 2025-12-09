#!/usr/bin/env python3
"""
Export a small chemistry snapshot from USPTO-50K sequences for the Rust annealer.

It samples one reaction from `data/uspto/50k_sequences.jsonl` (or a provided JSONL),
emits an EnvironmentSnapshot with:
- symbols for all reactants/reagents/products
- positions mapped from the sequence frames (x=index, y=frame)
- properties: domain=chemistry, role, smiles, class

Usage:
  python scripts/export_uspto_snapshot.py --input data/uspto/50k_sequences.jsonl --index 0 --output data/uspto/chem_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any


def load_line(path: Path, idx: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {path}")


def sequence_to_snapshot(rec: Dict[str, Any]) -> Dict[str, Any]:
    symbols = []
    # map each frame to y coordinate to preserve ordering (frame 0 -> y=0, frame 1 -> y=1)
    for frame_idx, frame in enumerate(rec["sequence"]):
        for sid, state in frame["symbol_states"].items():
            pos = state.get("position", {})
            properties = state.get("properties", {}).copy()
            properties.setdefault("domain", "chemistry")
            properties.setdefault("role", "unknown")
            sym = {
                "id": sid,
                "type": "OBJECT",
                "position": {"x": float(pos.get("x", 0.0)), "y": float(frame_idx), "z": 0.0},
                "properties": properties,
            }
            symbols.append(sym)

    stack_history = []
    for frame_idx, frame in enumerate(rec["sequence"]):
        symbol_states = {}
        for sid, state in frame["symbol_states"].items():
            pos = state.get("position", {})
            properties = state.get("properties", {}).copy()
            properties.setdefault("domain", "chemistry")
            properties.setdefault("role", "unknown")
            symbol_states[sid] = {
                "position": {"x": float(pos.get("x", 0.0)), "y": float(frame_idx), "z": 0.0},
                "velocity": None,
                "internal_state": properties,
            }
        stack_history.append(
            {
                "timestamp": {"unix": int(frame_idx)},
                "symbol_states": symbol_states,
            }
        )

    snapshot = {
        "timestamp": {"unix": 0},
        "bounds": {
            "width": max([s["position"]["x"] for s in symbols] + [1.0]) + 1.0,
            "height": 2.0,
            "depth": 1.0,
        },
        "symbols": symbols,
        "stack_history": stack_history,  # allow stack hashing/gap-fill
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/uspto/50k_sequences.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/uspto/chem_snapshot.json"))
    parser.add_argument("--index", type=int, default=None, help="Which line to sample (default: random).")
    args = parser.parse_args()

    # pick an index if not provided
    if args.index is None:
        # lightweight count
        with args.input.open("r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        idx = random.randint(0, total - 1)
    else:
        idx = args.index

    rec = load_line(args.input, idx)
    snap = sequence_to_snapshot(rec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        json.dump(snap, out, indent=2)
    print(f"Wrote snapshot to {args.output} (index={idx})")


if __name__ == "__main__":
    main()
