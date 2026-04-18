#!/usr/bin/env python3
"""
Cow Sensor Stream — W1z4rD V1510n
Reads Stage 2 cow video frames, routes each frame through the neural fabric
(/entity/observe), and broadcasts pose/state updates over WebSocket for the
3D cow world frontend.

Ports:
  8092  WebSocket  ws://localhost:8092
  8093  HTTP       http://localhost:8093   (serves packages/cow_world/)

Usage:
  python scripts/cow_sensor_stream.py [--node localhost:8090] [--fps 8]
"""
import argparse, asyncio, base64, io, json, math, os, sys, threading, time
import urllib.request
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional

import cv2
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("D:/w1z4rdv1510n-data/training/training/stage2_video/videos")
HTML_DIR  = ROOT / "packages" / "cow_world"

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> dict:
    small = cv2.resize(frame, (320, 180))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w  = gray.shape
    th, tw = h // 3, w // 3

    bright_top    = float(gray[:th, :].mean())
    bright_bot    = float(gray[2*th:, :].mean())
    bright_left   = float(gray[:, :tw].mean())
    bright_right  = float(gray[:, 2*tw:].mean())
    brightness    = float(gray.mean())
    contrast      = float(gray.std())

    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    edge_density = float(edges.mean() / 255.0)

    motion_mag = 0.0
    motion_cx  = 0.5
    motion_cy  = 0.5
    motion_dir = 0.0

    if prev_gray is not None:
        diff = np.abs(gray - prev_gray)
        motion_mag = float(diff.mean())
        if motion_mag > 0.004:
            thresh = diff.max() * 0.25
            ys, xs = np.where(diff > thresh)
            if len(xs) > 0:
                motion_cx  = float(xs.mean() / w)
                motion_cy  = float(ys.mean() / h)
                motion_dir = float(math.degrees(math.atan2(motion_cy - 0.5, motion_cx - 0.5)))

    return {
        "motion_mag":   motion_mag,
        "motion_cx":    motion_cx,
        "motion_cy":    motion_cy,
        "motion_dir":   motion_dir,
        "bright_top":   bright_top,
        "bright_bot":   bright_bot,
        "bright_left":  bright_left,
        "bright_right": bright_right,
        "brightness":   brightness,
        "contrast":     contrast,
        "edge_density": edge_density,
        "gray":         gray,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Neural fabric calls
# ─────────────────────────────────────────────────────────────────────────────

def call_entity_observe(features: dict, node: str) -> dict:
    body = {
        "entity_id": "bovine_cam",
        "timestamp": int(time.time() * 1000),
        "species":   "BOVINE",
        "sensors": [
            {
                "kind": "Motion",
                "values": {
                    "magnitude":  features["motion_mag"],
                    "x":          features["motion_cx"],
                    "y":          features["motion_cy"],
                    "direction":  features["motion_dir"],
                },
                "quality": 0.85,
            },
            {
                "kind": "Environment",
                "values": {
                    "brightness":    features["brightness"],
                    "contrast":      features["contrast"],
                    "edge_density":  features["edge_density"],
                    "top_ratio":     features["bright_top"]  / max(features["bright_bot"],   1e-6),
                    "left_ratio":    features["bright_left"] / max(features["bright_right"], 1e-6),
                },
                "quality": 0.9,
            },
        ],
    }
    try:
        req = urllib.request.Request(
            f"http://{node}/entity/observe",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return {}

def call_neuro_snapshot(node: str) -> dict:
    try:
        with urllib.request.urlopen(f"http://{node}/neuro/snapshot", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Pose computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pose(features: dict, observe: dict, state: dict, dt: float) -> dict:
    motion_mag = features["motion_mag"]
    bright_top = features["bright_top"]
    bright_bot = features["bright_bot"]

    # Animation state
    if motion_mag > 0.025:
        anim = "walk"
    elif bright_bot > bright_top * 1.12 and motion_mag < 0.01:
        anim = "graze"
    else:
        anim = "stand"

    # Walk speed (radians/second for phase integration)
    walk_speed = min(motion_mag * 100.0, 8.0)

    # Phase: integrate walk_speed * dt
    walk_phase = (state.get("walk_phase", 0.0) + walk_speed * dt) % (2 * math.pi)

    # Head pitch: negative = down (grazing), positive = alert
    raw_pitch = (bright_top - bright_bot) * 50.0
    head_pitch = max(-28.0, min(12.0, raw_pitch))

    # Head yaw: follow horizontal motion
    head_yaw = (features["motion_cx"] - 0.5) * 30.0

    # Body yaw: slowly track motion direction
    body_yaw = state.get("body_yaw", 0.0)
    if motion_mag > 0.008:
        target = features["motion_dir"]
        # Smooth toward motion direction (only when moving)
        body_yaw = body_yaw * 0.97 + target * 0.03

    # Pull values from neural fabric response
    confidence = float(observe.get("confidence", 0.5))
    narrative  = observe.get("narrative") or ""
    neuro_labels = observe.get("neuro_labels") or []
    survival   = observe.get("survival") or {}

    # Threat nudges movement speed
    threat = float((survival.get("threat_level") or 0.0))
    if threat > 0.6 and anim == "walk":
        walk_speed = min(walk_speed * 1.5, 8.0)

    state.update({
        "walk_phase": walk_phase,
        "body_yaw":   body_yaw,
    })

    return {
        "type":            "pose",
        "walk_phase":      walk_phase,
        "walk_speed":      walk_speed,
        "head_pitch":      head_pitch,
        "head_yaw":        head_yaw,
        "body_yaw_delta":  (body_yaw - state.get("prev_body_yaw", body_yaw)),
        "body_yaw":        body_yaw,
        "animation_state": anim,
        "confidence":      confidence,
        "narrative":       narrative[:120] if narrative else "",
        "concept_labels":  [l for l in neuro_labels if ":" in l][:10],
        "motion_mag":      motion_mag,
        "threat":          threat,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HTTP server (serves packages/cow_world/)
# ─────────────────────────────────────────────────────────────────────────────

def run_http_server(port: int):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HTML_DIR), **kw)
        def log_message(self, *_): pass

    server = HTTPServer(("localhost", port), Handler)
    print(f"  HTTP  → http://localhost:{port}")
    server.serve_forever()

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server + main async loop
# ─────────────────────────────────────────────────────────────────────────────

connected: set = set()

async def ws_handler(ws):
    connected.add(ws)
    try:
        await ws.wait_closed()
    finally:
        connected.discard(ws)

async def broadcast(msg: str):
    if not connected:
        return
    dead = set()
    for ws in list(connected):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    connected.difference_update(dead)

# ─────────────────────────────────────────────────────────────────────────────
# Video frame reader
# ─────────────────────────────────────────────────────────────────────────────

def get_video_files() -> list:
    if not VIDEO_DIR.exists():
        print(f"  [WARN] Video dir not found: {VIDEO_DIR}", file=sys.stderr)
        return []
    files = sorted(VIDEO_DIR.glob("*.mp4"))
    return [f for f in files if f.stat().st_size > 100_000]

async def sensor_loop(node: str, target_fps: float):
    videos = get_video_files()
    if not videos:
        print("  [WARN] No video files found — sensor loop idle", file=sys.stderr)
        while True:
            await asyncio.sleep(1)

    print(f"  Sensor loop: {len(videos)} videos at {target_fps:.0f} fps")
    interval   = 1.0 / target_fps
    video_idx  = 0
    state      = {"walk_phase": 0.0, "body_yaw": 0.0}
    prev_gray  = None
    snapshot_ts = 0.0
    last_snapshot: dict = {}
    observe_result: dict = {}
    frame_count = 0
    cap = None

    while True:
        loop_start = time.monotonic()

        # Open next video if needed
        if cap is None or not cap.isOpened():
            path = videos[video_idx % len(videos)]
            video_idx += 1
            cap = cv2.VideoCapture(str(path))
            prev_gray = None
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            skip = max(1, int(native_fps / target_fps))
            print(f"  [{video_idx}/{len(videos)}] {path.name}  native={native_fps:.0f}fps  skip={skip}")

        # Read frame (skip ahead to maintain target FPS)
        ret = False
        for _ in range(skip):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            cap.release()
            cap = None
            prev_gray = None
            await asyncio.sleep(0.05)
            continue

        frame_count += 1

        # Feature extraction (blocking but fast)
        features = await asyncio.to_thread(extract_features, frame, prev_gray)
        prev_gray = features.pop("gray")

        # Call entity_observe (async, skip if slow)
        observe_result = await asyncio.to_thread(call_entity_observe, features, node)

        # Poll snapshot every 5s
        now = time.monotonic()
        if now - snapshot_ts > 5.0:
            last_snapshot = await asyncio.to_thread(call_neuro_snapshot, node)
            snapshot_ts = now

        # Compute pose
        dt = time.monotonic() - loop_start
        pose = compute_pose(features, observe_result, state, max(dt, 0.01))
        state["prev_body_yaw"] = pose["body_yaw"]

        # Top concept labels from snapshot
        snap_labels = last_snapshot.get("active_labels", [])[:8]
        pose["snap_labels"] = snap_labels

        # Thumbnail (160×90 JPEG)
        thumb = cv2.resize(frame, (160, 90))
        _, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 55])
        pose["frame_b64"] = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()

        await broadcast(json.dumps(pose))

        elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(0.0, interval - elapsed))

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(node: str, ws_port: int, http_port: int, fps: float):
    print("=" * 60)
    print("  W1z4rD V1510n — Cow Sensor Stream")
    print("=" * 60)
    print(f"  Node  → http://{node}")
    print(f"  WS    → ws://localhost:{ws_port}")

    threading.Thread(target=run_http_server, args=(http_port,), daemon=True).start()

    async with websockets.serve(ws_handler, "localhost", ws_port):
        await sensor_loop(node, fps)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--node",      default="localhost:8090")
    ap.add_argument("--ws-port",   type=int, default=8092)
    ap.add_argument("--http-port", type=int, default=8093)
    ap.add_argument("--fps",       type=float, default=8.0)
    args = ap.parse_args()
    try:
        asyncio.run(main(args.node, args.ws_port, args.http_port, args.fps))
    except KeyboardInterrupt:
        print("\nStopped.")
