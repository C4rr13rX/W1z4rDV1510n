#!/usr/bin/env python3
"""
Convert ETH/UCY obsmat.txt into frame-by-frame person trajectories.
Each line in obsmat: frame, person_id, x, y
We produce sequences of length `window` with stride, per dataset.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_obsmat(path: Path) -> List[Tuple[int, int, float, float]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            frame, pid = int(float(parts[0])), int(float(parts[1]))
            x, y = float(parts[2]), float(parts[3])
            data.append((frame, pid, x, y))
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/eth_ucy/eth_univ_obsmat.txt"))
    ap.add_argument("--output", type=Path, default=Path("data/eth_ucy/eth_sequences.jsonl"))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    rows = load_obsmat(args.input)
    by_pid: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for frame, pid, x, y in rows:
        by_pid[pid].append((frame, x, y))

    sequences = []
    for pid, traj in by_pid.items():
        traj.sort(key=lambda t: t[0])
        for start in range(0, len(traj) - args.window + 1, args.stride):
            window = traj[start : start + args.window]
            frames = []
            for t, (frame_idx, x, y) in enumerate(window):
                frames.append(
                    {
                        "timestamp": {"unix": frame_idx},
                        "symbol_states": {
                            f"p{pid}": {
                                "position": {"x": x, "y": y, "z": 0.0},
                                "properties": {"domain": "crowd", "role": "person", "pid": pid},
                            }
                        },
                    }
                )
            sequences.append({"sequence": frames, "metadata": {"pid": pid, "start_frame": window[0][0]}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for rec in sequences:
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()
