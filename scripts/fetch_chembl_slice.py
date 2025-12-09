#!/usr/bin/env python3
"""
Fetch a small ChEMBL bioactivity slice (ChEMBL 33 SQLite) requires confirmation because full DB is large.
Default behavior: abort unless --confirm-large is passed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import requests
from tqdm import tqdm

URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_sqlite.tar.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/chembl/chembl_33_sqlite.tar.gz"))
    ap.add_argument("--confirm-large", action="store_true", help="Required to download (~1.5GB).")
    args = ap.parse_args()
    if not args.confirm_large:
        raise SystemExit("Refusing to download ~1.5GB without --confirm-large.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=120) as resp:
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
