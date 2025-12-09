#!/usr/bin/env python3
"""
Run a small batch of USPTO-50K reactions through the annealer to sanity-check chemistry predictions.

Steps per reaction index:
- Export a snapshot from the sequences JSONL.
- Write a temp config overriding `snapshot_file`.
- Invoke `predict_state` and capture the best energy.

Usage:
  python scripts/eval_uspto_batch.py --indices 0 1 2 --base-config run_config_chem.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    return res.stdout + res.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, default=Path("data/uspto/50k_sequences.jsonl"))
    parser.add_argument("--base-config", type=Path, default=Path("run_config_chem.json"))
    parser.add_argument("--output-log", type=Path, default=Path("logs/chem_batch.log"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    if not shutil.which("cargo"):
        raise SystemExit("cargo not found in PATH")

    args.output_log.parent.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        snapshot_path = Path(f"data/uspto/chem_snapshot_{idx}.json")
        print(f"[export] reaction index {idx} -> {snapshot_path}")
        run_cmd(
            [
                "python",
                "scripts/export_uspto_snapshot.py",
                "--input",
                str(args.sequences),
                "--index",
                str(idx),
                "--output",
                str(snapshot_path),
            ]
        )

        with args.base_config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["snapshot_file"] = str(snapshot_path).replace("\\", "/")
        cfg["logging"]["log_path"] = str(args.output_log).replace("\\", "/")

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            json.dump(cfg, tmp, indent=2)
            tmp.flush()
            tmp_config = tmp.name

        print(f"[run ] predict_state for index {idx}")
        output = run_cmd(["cargo", "run", "--bin", "predict_state", "--", "--config", tmp_config])
        # Append to the batch log
        with args.output_log.open("a", encoding="utf-8") as logf:
            logf.write(f"\n=== index {idx} ===\n")
            logf.write(output)

        Path(tmp_config).unlink(missing_ok=True)

    print(f"Batch complete. Log at {args.output_log}")


if __name__ == "__main__":
    main()
