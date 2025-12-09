#!/usr/bin/env python3
"""
Convert a genomic-benchmarks dataset (downloaded via download_dataset) into JSONL sequences.

Dataset layout (after download_dataset):
  <root>/<name>/<split>/<class>/*.txt  (each file contains a DNA sequence)

We emit two-frame sequences:
- Frame 0: per-base symbols along x-axis (role=base, base in properties)
- Frame 1: a label symbol (role=label, class name)

This keeps things light for priors while retaining positional structure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_sequences(root: Path, split: str, max_sequences: int | None) -> List[tuple[str, str]]:
    data = []
    split_dir = root / split
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for txt in cls_dir.glob("*.txt"):
            seq = txt.read_text().strip().upper()
            data.append((seq, cls))
            if max_sequences and len(data) >= max_sequences:
                return data
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/genomics/human_nontata_promoters"))
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--output", type=Path, default=Path("data/genomics/genomic_sequences.jsonl"))
    ap.add_argument("--max-sequences", type=int, default=500)
    args = ap.parse_args()

    samples = load_sequences(args.root, args.split, args.max_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for idx, (seq, cls) in enumerate(samples):
            bases = list(seq)
            frame0_states = {}
            for i, base in enumerate(bases):
                sid = f"b{idx}_{i}"
                frame0_states[sid] = {
                    "position": {"x": float(i), "y": 0.0, "z": 0.0},
                    "properties": {"domain": "genomics", "role": "base", "base": base},
                }
            frame1_states = {
                f"label_{idx}": {
                    "position": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "properties": {"domain": "genomics", "role": "label", "class": cls},
                }
            }
            rec = {
                "sequence": [
                    {"timestamp": {"unix": 0}, "symbol_states": frame0_states},
                    {"timestamp": {"unix": 1}, "symbol_states": frame1_states},
                ],
                "metadata": {"class": cls},
            }
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(samples)} sequences to {args.output}")


if __name__ == "__main__":
    main()
