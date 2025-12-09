#!/usr/bin/env python3
"""
Fetch a small exoplanet catalog slice from NASA Exoplanet Archive TAP API to data/astronomy/exoplanets.csv.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import requests
from urllib.parse import quote_plus

QUERY = "SELECT TOP 5000 pl_name,ra,dec,sy_dist,sy_vmag FROM pscomppars"
URL = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?request=doQuery&format=csv&query={quote_plus(QUERY)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/astronomy/exoplanets.csv"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    args.output.write_bytes(resp.content)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
