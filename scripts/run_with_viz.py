#!/usr/bin/env python3
"""
Wrapper to run a command (e.g., predict_state) and auto-render a visualization
for the snapshot referenced in the config.

Usage:
  python scripts/run_with_viz.py --config run_config_exoplanet.json --cmd "cargo run --bin predict_state -- --config {config}" --viz logs/exoplanet_viz.html

The placeholder {config} is replaced with the config path.
Requires: plotly (for visualization).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd: str) -> None:
    print(f"[run] {cmd}")
    res = subprocess.run(shlex.split(cmd))
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="Config JSON with snapshot_file field.")
    ap.add_argument(
        "--cmd",
        required=True,
        help='Command to run; use "{config}" placeholder for the config path (e.g., cargo run --bin predict_state -- --config {config})',
    )
    ap.add_argument("--viz", type=Path, default=None, help="Output HTML for visualization (default: logs/auto_viz.html)")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    snapshot = cfg.get("snapshot_file")
    if not snapshot:
        print("[warn] No snapshot_file in config; skipping visualization after run.")
    cmd = args.cmd.replace("{config}", str(args.config))
    run_cmd(cmd)

    if snapshot:
        viz_out = args.viz or Path("logs/auto_viz.html")
        viz_out.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(f"python scripts/visualize_snapshot.py --snapshot {snapshot} --output {viz_out}")
        print(f"[viz] wrote {viz_out}")


if __name__ == "__main__":
    main()
