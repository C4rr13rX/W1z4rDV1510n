#!/usr/bin/env python3
"""
Export a generic snapshot with stack_history from a JSONL sequences file.
Assumes input records follow the format produced by our preprocessors
({ "sequence": [ {timestamp, symbol_states}, ... ], ... }).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_line(path: Path, idx: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {path}")


def to_snapshot(rec: Dict[str, Any]) -> Dict[str, Any]:
    symbols: List[Dict[str, Any]] = []
    stack_history: List[Dict[str, Any]] = []
    max_x = 0.0
    max_y = 0.0
    for frame_idx, frame in enumerate(rec["sequence"]):
        symbol_states = {}
        for sid, state in frame["symbol_states"].items():
            pos = state.get("position", {})
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            props = state.get("properties", {}).copy()
            symbols.append(
                {
                    "id": sid,
                    "type": "OBJECT",
                    "position": {"x": x, "y": y, "z": z},
                    "properties": props,
                }
            )
            symbol_states[sid] = {
                "position": {"x": x, "y": y, "z": z},
                "velocity": None,
                "internal_state": props,
            }
        stack_history.append({"timestamp": {"unix": int(frame_idx)}, "symbol_states": symbol_states})
    snapshot = {
        "timestamp": {"unix": 0},
        "bounds": {"width": max_x + 1.0, "height": max_y + 1.0, "depth": 1.0},
        "symbols": symbols,
        "stack_history": stack_history,
    }
    return snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="JSONL sequences file")
    ap.add_argument("--index", type=int, default=None, help="Line index to export (default random).")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.index is None:
        with args.input.open("r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        idx = random.randint(0, total - 1)
    else:
        idx = args.index

    rec = load_line(args.input, idx)
    snap = to_snapshot(rec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        json.dump(snap, out, indent=2)
    print(f"Wrote snapshot to {args.output} (index={idx})")


if __name__ == "__main__":
    main()
