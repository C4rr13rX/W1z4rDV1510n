#!/usr/bin/env python3
"""
Visualize a snapshot (and its stack_history) as a psychedelic 2D/3D animation.

Inputs:
  --snapshot data/<domain>/<file>.json (must contain symbols; optional stack_history frames)
  --output   HTML file to write (default: logs/visualization.html)

What it renders:
  - 2D grid on black background with bright, role-coded points per frame (frames animate).
  - 3D surface heatmap showing symbol density per frame.

Requires: plotly (pip install plotly)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import plotly.graph_objects as go


def hash_color(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    # Bright palette: use high saturation/value in HSV mapped to RGB
    hue = int(h[:2], 16) / 255.0
    sat = 0.9
    val = 1.0
    i = int(hue * 6)
    f = hue * 6 - i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    i = i % 6
    if i == 0:
        r, g, b = val, t, p
    elif i == 1:
        r, g, b = q, val, p
    elif i == 2:
        r, g, b = p, val, t
    elif i == 3:
        r, g, b = p, q, val
    elif i == 4:
        r, g, b = t, p, val
    else:
        r, g, b = val, p, q
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def load_frames(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "stack_history" in snapshot:
        return snapshot["stack_history"]
    # fallback: single frame from symbols
    frame_states = {}
    for sym in snapshot.get("symbols", []):
        pos = sym.get("position", {})
        frame_states[sym["id"]] = {
            "position": {"x": pos.get("x", 0.0), "y": pos.get("y", 0.0), "z": pos.get("z", 0.0)},
            "internal_state": sym.get("properties", {}),
        }
    return [{"timestamp": {"unix": 0}, "symbol_states": frame_states}]


def build_density(frame: Dict[str, Any], bins: int = 24) -> Tuple[List[List[float]], List[float], List[float]]:
    xs = []
    ys = []
    for state in frame["symbol_states"].values():
        pos = state.get("position", {})
        xs.append(float(pos.get("x", 0.0)))
        ys.append(float(pos.get("y", 0.0)))
    if not xs or not ys:
        return [[0.0]], [0.0, 1.0], [0.0, 1.0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    if maxx == minx:
        maxx += 1.0
    if maxy == miny:
        maxy += 1.0
    grid = [[0.0 for _ in range(bins)] for _ in range(bins)]
    for x, y in zip(xs, ys):
        gx = int((x - minx) / (maxx - minx) * (bins - 1))
        gy = int((y - miny) / (maxy - miny) * (bins - 1))
        grid[gy][gx] += 1.0
    xs_lin = [minx + (maxx - minx) * i / (bins - 1) for i in range(bins)]
    ys_lin = [miny + (maxy - miny) * i / (bins - 1) for i in range(bins)]
    return grid, xs_lin, ys_lin


def make_fig(frames: List[Dict[str, Any]]) -> go.Figure:
    # 2D scatter per frame
    scatter_frames = []
    roles = set()
    for frame in frames:
        xs, ys, colors, sizes = [], [], [], []
        for sid, state in frame["symbol_states"].items():
            pos = state.get("position", {})
            role = state.get("internal_state", {}).get("role", "symbol")
            roles.add(role)
            xs.append(pos.get("x", 0.0))
            ys.append(pos.get("y", 0.0))
            colors.append(hash_color(role + sid))
            sizes.append(10 + (hash(sid) % 6))
        scatter_frames.append(go.Frame(data=[go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.95),
        )]))

    # 3D heatmap per frame
    surface_frames = []
    for frame in frames:
        grid, xs_lin, ys_lin = build_density(frame, bins=24)
        surface_frames.append(go.Frame(data=[go.Surface(
            z=grid, x=xs_lin, y=ys_lin, colorscale="Turbo", opacity=0.8
        )]))

    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="markers",
                       marker=dict(size=1, color="white")),
            go.Surface(z=[[0]], x=[0], y=[0], colorscale="Turbo", showscale=True, opacity=0.8)
        ],
        layout=go.Layout(
            template="plotly_dark",
            title="Psychedelic Sequence Explorer",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            scene=dict(
                xaxis=dict(showgrid=False, backgroundcolor="black"),
                yaxis=dict(showgrid=False, backgroundcolor="black"),
                zaxis=dict(showgrid=False, backgroundcolor="black"),
                bgcolor="black",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[dict(label="Play", method="animate", args=[None])],
                    bgcolor="black",
                    bordercolor="magenta",
                )
            ],
        ),
        frames=[
            go.Frame(data=sf.data + sff.data, name=str(i))
            for i, (sff, sf) in enumerate(zip(scatter_frames, surface_frames))
        ],
    )
    return fig


def board_json_to_snapshot(board: Dict[str, Any]) -> Dict[str, Any]:
    """Convert chess_live_board.json into a minimal EnvironmentSnapshot-like dict."""
    frame_states = {}
    for piece in board.get("pieces", []):
        sid = piece.get("id", "")
        x = float(piece.get("x", 0.0))
        y = float(piece.get("y", 0.0))
        frame_states[sid] = {
            "position": {"x": x, "y": y, "z": 0.0},
            "internal_state": {
                "piece": piece.get("piece", ""),
                "color": piece.get("color", ""),
                "square": piece.get("square", ""),
            },
        }
    frame = {
        "timestamp": {"unix": board.get("ply", 0)},
        "symbol_states": frame_states,
    }
    return {"stack_history": [frame]}


def make_live_html(output: Path) -> str:
    """Return an HTML wrapper that auto-refreshes every 3 seconds."""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>W1z4rDV1510n Live Board</title>
  <meta http-equiv="refresh" content="3">
  <style>body{{background:#000;margin:0;padding:0}}</style>
</head>
<body>
  <iframe src="{output.name}" style="width:100%;height:100vh;border:none;"></iframe>
</body>
</html>
"""


def main() -> None:
    import time
    import webbrowser

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path,
                    help="Snapshot JSON file (EnvironmentSnapshot format). "
                         "Omit with --live to use logs/chess_live_board.json.")
    ap.add_argument("--output", type=Path, default=Path("logs/visualization.html"))
    ap.add_argument("--live", action="store_true",
                    help="Poll the snapshot/board file every --interval seconds and regenerate.")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="Refresh interval in seconds when --live is set (default: 3).")
    ap.add_argument("--open", action="store_true",
                    help="Open the output in a browser on first render.")
    args = ap.parse_args()

    # Resolve source file
    if args.snapshot:
        src = args.snapshot
    elif args.live:
        src = Path("logs/chess_live_board.json")
    else:
        ap.error("--snapshot is required unless --live is used")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def render_once() -> bool:
        if not src.exists():
            print(f"  waiting for {src} ...", flush=True)
            return False
        raw = src.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        # Accept both EnvironmentSnapshot and chess_live_board.json
        if "stack_history" in data or "symbols" in data:
            snapshot = data
        else:
            snapshot = board_json_to_snapshot(data)
        frames = load_frames(snapshot)
        fig = make_fig(frames)
        fig.write_html(str(args.output), include_plotlyjs="cdn", auto_play=False)
        return True

    if not args.live:
        render_once()
        print(f"Wrote visualization to {args.output}")
        return

    # Live polling mode -- write a wrapper HTML with auto-refresh pointing at the
    # inner visualization file, then keep regenerating the inner file.
    wrapper = args.output.parent / ("live_" + args.output.name)
    wrapper.write_text(make_live_html(args.output), encoding="utf-8")
    print(f"Live wrapper: {wrapper}  (inner: {args.output}, refresh every {args.interval}s)")
    if args.open:
        webbrowser.open(wrapper.as_uri())
    while True:
        try:
            ok = render_once()
            if ok:
                print(f"  [{time.strftime('%H:%M:%S')}] updated {args.output}", flush=True)
        except Exception as exc:
            print(f"  render error: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
