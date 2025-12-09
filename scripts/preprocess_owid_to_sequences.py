#!/usr/bin/env python3
"""
Convert OWID COVID CSV into time-ordered sequences for selected countries.

Each sequence = sliding window of days per country:
- Frame per day with symbols: cases, deaths, vaccinations (roles), positions mapped to normalized counts.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/epidemiology/owid-covid.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/epidemiology/owid_sequences.jsonl"))
    ap.add_argument("--countries", nargs="+", default=["USA", "IND", "BRA", "GBR", "CAN"])
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--max-rows", type=int, default=200000)
    args = ap.parse_args()

    rows_per_country: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.max_rows and i >= args.max_rows:
                break
            c = row.get("iso_code")
            if c not in args.countries:
                continue
            rows_per_country[c].append(row)

    sequences = []
    for country, rows in rows_per_country.items():
        rows.sort(key=lambda r: r.get("date"))
        for start in range(0, len(rows) - args.window + 1, args.stride):
            window = rows[start : start + args.window]
            frames = []
            for t, r in enumerate(window):
                # normalize counts per million if available
                def safe_float(key: str) -> float:
                    v = r.get(key)
                    try:
                        return float(v) if v not in (None, "", "NA") else 0.0
                    except Exception:
                        return 0.0

                cases = safe_float("new_cases_per_million")
                deaths = safe_float("new_deaths_per_million")
                vax = safe_float("new_vaccinations_smoothed_per_million")
                frame_states = {
                    f"{country}_cases": {
                        "position": {"x": cases / 10000.0, "y": 0.0, "z": 0.0},
                        "properties": {"domain": "epidemiology", "role": "cases", "country": country},
                    },
                    f"{country}_deaths": {
                        "position": {"x": deaths / 1000.0, "y": 0.1, "z": 0.0},
                        "properties": {"domain": "epidemiology", "role": "deaths", "country": country},
                    },
                    f"{country}_vax": {
                        "position": {"x": vax / 10000.0, "y": 0.2, "z": 0.0},
                        "properties": {"domain": "epidemiology", "role": "vax", "country": country},
                    },
                }
                frames.append({"timestamp": {"unix": t}, "symbol_states": frame_states})
            sequences.append({"sequence": frames, "metadata": {"country": country, "start": start}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for rec in sequences:
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()
