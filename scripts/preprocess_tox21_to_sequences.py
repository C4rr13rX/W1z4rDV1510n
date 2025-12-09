#!/usr/bin/env python3
"""
Convert Tox21 CSV.gz into two-frame sequences (compound -> label) for priors.
Frame 0: compound symbol (role=compound, domain=tox21, SMILES stored)
Frame 1: outcome symbol (role=label, domain=tox21, label in properties)
"""
from __future__ import annotations

import argparse
import gzip
import csv
import json
from pathlib import Path
from typing import List, Dict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/tox21/tox21.csv.gz"))
    ap.add_argument("--output", type=Path, default=Path("data/tox21/tox21_sequences.jsonl"))
    ap.add_argument("--max-rows", type=int, default=5000)
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    with gzip.open(args.input, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if args.max_rows and len(rows) >= args.max_rows:
                break

    sequences = []
    label_keys = [k for k in rows[0].keys() if k.startswith("NR-") or k.startswith("SR-") or k.startswith("DR-")]
    for idx, r in enumerate(rows):
        smiles = r.get("smiles", "")
        # pick first non-empty label
        label = None
        for key in label_keys:
            v = r.get(key)
            if v not in (None, "", "NA"):
                label = (key, v)
                break
        if not label:
            continue
        lname, lval = label
        seq = {
            "sequence": [
                {
                    "timestamp": {"unix": 0},
                    "symbol_states": {
                        f"cmp_{idx}": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "properties": {"domain": "tox21", "role": "compound", "smiles": smiles},
                        }
                    },
                },
                {
                    "timestamp": {"unix": 1},
                    "symbol_states": {
                        f"label_{idx}": {
                            "position": {"x": float(lval) if str(lval).replace('.', '', 1).isdigit() else 0.0, "y": 0.0, "z": 0.0},
                            "properties": {"domain": "tox21", "role": "label", "name": lname, "value": lval},
                        }
                    },
                },
            ],
            "metadata": {"label_name": lname},
        }
        sequences.append(seq)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for rec in sequences:
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()
