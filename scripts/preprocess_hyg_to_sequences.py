#!/usr/bin/env python3
"""
Convert exoplanet catalog into small sequences for relational priors.

We take batches (by V magnitude) and emit 2-frame sequences:
- Frame 0: brighter half (role=bright)
- Frame 1: dimmer half (role=dim)
Positions use RA/DEC bins.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict


def bin_pos(val: float, bins: int = 16) -> float:
    return float(int(val * bins) / bins)


def load_rows(path: Path, max_rows: int | None) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/astronomy/exoplanets.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/astronomy/exoplanet_sequences.jsonl"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=4096)
    args = ap.parse_args()

    rows = load_rows(args.input, args.max_rows)
    rows.sort(key=lambda r: float(r.get("sy_vmag", "0") or 0.0))

    sequences = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        if len(chunk) < 2:
            break
        mid = len(chunk) // 2
        bright = chunk[:mid]
        dim = chunk[mid:]

        def make_states(stars: List[Dict[str, str]], role: str, frame_y: float):
            states = {}
            for j, s in enumerate(stars):
                sid = f"{role}_{i+j}"
                ra = float(s.get("ra", "0") or 0.0)
                dec = float(s.get("dec", "0") or 0.0)
                states[sid] = {
                    "position": {"x": bin_pos(ra / 360.0), "y": bin_pos((dec + 90.0) / 180.0), "z": 0.0},
                    "properties": {
                        "domain": "astronomy",
                        "role": role,
                        "mag": float(s.get("sy_vmag", "0") or 0.0),
                        "name": s.get("pl_name", ""),
                    },
                }
            return states

        seq = {
            "sequence": [
                {"timestamp": {"unix": 0}, "symbol_states": make_states(bright, "bright", 0.0)},
                {"timestamp": {"unix": 1}, "symbol_states": make_states(dim, "dim", 1.0)},
            ],
            "metadata": {"batch_start": i},
        }
        sequences.append(seq)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for rec in sequences:
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()
