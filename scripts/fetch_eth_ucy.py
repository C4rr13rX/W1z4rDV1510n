#!/usr/bin/env python3
"""
Fetch ETH/UCY ETH-univ trajectories (alternate host) to data/eth_ucy/eth_univ_obsmat.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import requests
from tqdm import tqdm

# Mirror of ETH/univ obsmat
URL = "https://raw.githubusercontent.com/agrimgupta92/sgan/master/datasets/eth/univ/obsmat.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/eth_ucy/eth_univ_obsmat.txt"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(args.output, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=args.output.name) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
