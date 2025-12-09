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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("logs/visualization.html"))
    args = ap.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    frames = load_frames(snapshot)
    fig = make_fig(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn", auto_play=False)
    print(f"Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()
