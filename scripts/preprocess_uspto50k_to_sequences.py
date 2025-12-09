#!/usr/bin/env python3
"""
Convert the Hugging Face USPTO-50K parquet files into a generic JSONL sequence
format consumable by `build_relational_priors.py`.

For each reaction we emit a two-frame sequence:
- Frame 0: reactant symbols (role=reactant, domain=chemistry)
- Frame 1: product symbols (role=product, domain=chemistry)

Positions are simple 1D bins (x = index, y = 0) to let the relational hasher
observe ordering/co-occurrence without needing geometry. Raw SMILES are kept
in properties for downstream symbolization if desired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import duckdb


def load_parquet_rows(paths: Iterable[Path], limit: int | None) -> List[Tuple]:
    rows: List[Tuple] = []
    for p in paths:
        con = duckdb.connect(database=":memory:")
        query = "SELECT class, id, prod_smiles, rxn_smiles FROM read_parquet(?)"
        if limit is not None:
            query += f" LIMIT {limit}"
        part = con.execute(query, [str(p)]).fetchall()
        rows.extend(part)
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def reaction_to_sequence(reaction_id: str, rxn_smiles: str, prod_smiles: str, cls: int):
    # rxn_smiles format: reactants>reagents>products
    pieces = rxn_smiles.split(">")
    if len(pieces) != 3:
        reactants, reagents, products = rxn_smiles, "", prod_smiles
    else:
        reactants, reagents, products = pieces

    reactant_parts = [r for r in reactants.split(".") if r]
    product_parts = [p for p in products.split(".") if p]
    # optional: include reagents as context symbols in frame 0
    reagent_parts = [r for r in reagents.split(".") if r]

    def make_symbols(parts: List[str], role: str):
        states = {}
        for idx, smi in enumerate(parts):
            sid = f"{role}_{reaction_id}_{idx}"
            states[sid] = {
                "position": {"x": float(idx), "y": 0.0, "z": 0.0},
                "properties": {
                    "domain": "chemistry",
                    "role": role,
                    "class": int(cls),
                    "smiles": smi,
                },
            }
        return states

    frame0 = {
        "timestamp": {"unix": 0},
        "symbol_states": {},
    }
    frame0["symbol_states"].update(make_symbols(reactant_parts, "reactant"))
    frame0["symbol_states"].update(make_symbols(reagent_parts, "reagent"))

    frame1 = {
        "timestamp": {"unix": 1},
        "symbol_states": make_symbols(product_parts, "product"),
    }

    return {"sequence": [frame0, frame1], "metadata": {"id": reaction_id, "class": int(cls)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/uspto/50k"),
        help="Directory containing train.parquet / validation.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/uspto/50k_sequences.jsonl"),
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    args = parser.parse_args()

    train_path = args.root / "train.parquet"
    val_path = args.root / "validation.parquet"
    if not train_path.exists():
        raise SystemExit(f"Missing parquet: {train_path}")
    paths = [train_path]
    if val_path.exists():
        paths.append(val_path)

    rows = load_parquet_rows(paths, args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for cls, rid, prod_smi, rxn_smi in rows:
            rec = reaction_to_sequence(str(rid), rxn_smi, prod_smi, cls)
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(rows)} sequences to {args.output}")


if __name__ == "__main__":
    main()
