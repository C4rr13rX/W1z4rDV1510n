#!/usr/bin/env python3
"""
Fetch Tox21 dataset (CSV.gz) to data/tox21/tox21.csv.gz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import requests
from tqdm import tqdm

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/tox21/tox21.csv.gz"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(args.output, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=args.output.name) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
