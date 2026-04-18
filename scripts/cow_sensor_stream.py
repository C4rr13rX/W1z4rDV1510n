#!/usr/bin/env python3
"""
Cow Sensor Stream — W1z4rD V1510n
Reads pre-extracted Stage 2 JPEG frames, routes each through the neural fabric
(/media/train cross-modal: image + anatomical text), reads /neuro/snapshot and
/qa/query for semantic state, and broadcasts over WebSocket for the 3D world.

No third-party CV libraries — the neural fabric is the vision system.

Ports:
  8092  WebSocket  ws://localhost:8092
  8093  HTTP       http://localhost:8093   (serves packages/cow_world/)

Usage:
  python scripts/cow_sensor_stream.py [--node localhost:8090] [--fps 4]
"""
import argparse, asyncio, base64, json, math, sys, threading, time
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

import websockets

ROOT       = Path(__file__).resolve().parent.parent
FRAMES_DIR = Path("D:/w1z4rdv1510n-data/training/training/stage2_video/frames")
VIDEOS_DIR = Path("D:/w1z4rdv1510n-data/training/training/stage2_video/videos")
HTML_DIR   = ROOT / "packages" / "cow_world"

# Rotating anatomical context strings for cross-modal training variety
CONTEXTS = [
    "Holstein dairy cow bovine anatomy legs spine neck head tail udder",
    "Bovine locomotion musculoskeletal system limbs hoof stride gait",
    "Dairy cow body condition dorsal spine ribs pelvis bovine conformation",
    "Bovine thorax abdomen rumen reticulum digestive system flank",
    "Cattle behavior grazing standing walking bovine ethology pasture",
    "Holstein Friesian breed black white coat pattern udder teat milking",
    "Bovine cervical thoracic lumbar sacral vertebrae spine atlas axis",
    "Cow face eye horn ear muzzle nasal head anatomy bovine skull",
    "Bovine fetlock pastern coronary hoof coffin bone digital anatomy",
    "Cow shoulder elbow carpal metacarpal front limb bovine forelimb",
    "Bovine hip stifle hock metatarsal rear limb hindlimb anatomy",
    "Dairy cattle poll poll forehead brow occipital bovine cranium",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_title_map() -> dict:
    result = {}
    if not VIDEOS_DIR.exists():
        return result
    for info_file in VIDEOS_DIR.glob("*.info.json"):
        try:
            d = json.loads(info_file.read_bytes())
            result[info_file.stem] = d.get("title", "")
        except Exception:
            pass
    return result

def get_all_frames() -> list:
    if not FRAMES_DIR.exists():
        return []
    return sorted(FRAMES_DIR.glob("*.jpg"))

# ─────────────────────────────────────────────────────────────────────────────
# Neural fabric API calls
# ─────────────────────────────────────────────────────────────────────────────

def call_media_train(frame_path: Path, text: str, node: str) -> dict:
    """Cross-modal train: image + anatomical text in one co-activation."""
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({"modality": "image", "data_b64": b64, "text": text}).encode()
    req = urllib.request.Request(
        f"http://{node}/media/train",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except Exception:
        return {}

def call_qa_query(question: str, node: str) -> str:
    """Query the Q&A fabric and return the top bovine answer."""
    body = json.dumps({"question": question, "top_k": 3}).encode()
    req = urllib.request.Request(
        f"http://{node}/qa/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            d = json.loads(r.read())
        results = d.get("report", {}).get("results", [])
        # Take the highest-confidence result that mentions bovine/cow behaviour
        for res in results:
            a = (res.get("answer") or "").lower()
            if any(w in a for w in ["cow", "bovine", "cattle", "dairy",
                                     "walk", "graz", "stand", "hoof", "leg"]):
                return res.get("answer", "")
        return results[0].get("answer", "") if results else ""
    except Exception:
        return ""

def call_neuro_snapshot(node: str) -> dict:
    try:
        with urllib.request.urlopen(f"http://{node}/neuro/snapshot", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Motion proxy (fabric-native — no CV)
# Label count change + file size change between consecutive frames.
# JPEG file size correlates with scene complexity; label count change reflects
# how much the fabric's visual concept activations shifted frame-to-frame.
# ─────────────────────────────────────────────────────────────────────────────

def motion_from_frame_delta(
    curr_label_count: int,
    prev_label_count: int,
    curr_file_size: int,
    prev_file_size: int,
) -> float:
    label_d = abs(curr_label_count - prev_label_count) / max(prev_label_count, 1)
    size_d  = abs(curr_file_size  - prev_file_size)   / max(prev_file_size,  1)
    return min(label_d * 0.6 + size_d * 0.4, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Pose computation (fabric-driven)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pose(
    motion_mag: float,
    qa_answer: str,
    snapshot: dict,
    state: dict,
    dt: float,
) -> dict:
    qa_lower = qa_answer.lower()

    # Animation state from motion + Q&A context
    if motion_mag > 0.12:
        anim = "walk"
    elif any(w in qa_lower for w in ["graz", "graze", "grass", "head low", "pasture"]):
        anim = "graze"
    elif any(w in qa_lower for w in ["walk", "stride", "locomot", "gait"]):
        anim = "walk"
    else:
        anim = "stand"

    # Walk speed
    walk_speed = min(motion_mag * 6.0, 5.0) if anim == "walk" else 0.3 if anim == "graze" else 0.0

    # Integrate walk phase
    walk_phase = (state.get("walk_phase", 0.0) + walk_speed * dt) % (2 * math.pi)

    # Head pitch: graze = down, alert = up
    head_pitch = -22.0 if anim == "graze" else (5.0 if motion_mag < 0.05 else -5.0)

    # Body yaw: slow random drift from motion
    body_yaw = state.get("body_yaw", 0.0)
    if motion_mag > 0.08:
        body_yaw += (motion_mag - 0.08) * 0.3

    # Snapshot labels
    active_labels = snapshot.get("active_labels", [])
    bovine_labels = [l for l in active_labels if any(
        w in l.lower() for w in ["bovine", "cow", "cattle", "dairy",
                                   "rumen", "udder", "hoof", "spine"]
    )]

    state.update({"walk_phase": walk_phase, "body_yaw": body_yaw})

    return {
        "type":            "pose",
        "walk_phase":      walk_phase,
        "walk_speed":      walk_speed,
        "head_pitch":      head_pitch,
        "head_yaw":        0.0,
        "body_yaw":        body_yaw,
        "animation_state": anim,
        "confidence":      min(motion_mag * 4.0 + 0.4, 1.0),
        "concept_labels":  bovine_labels[:6] or active_labels[:4],
        "motion_mag":      motion_mag,
        "narrative":       qa_answer[:120] if qa_answer else "",
    }

# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

def run_http_server(port: int):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HTML_DIR), **kw)
        def log_message(self, *_): pass

    server = HTTPServer(("localhost", port), Handler)
    print(f"  HTTP  -> http://localhost:{port}")
    server.serve_forever()

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server
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
# Sensor loop
# ─────────────────────────────────────────────────────────────────────────────

async def sensor_loop(node: str, target_fps: float):
    frames = get_all_frames()
    if not frames:
        print("  [WARN] No frames found in", FRAMES_DIR)
        while True:
            await asyncio.sleep(1)

    title_map = load_title_map()
    print(f"  Sensor loop: {len(frames)} frames  node={node}  fps={target_fps}")

    interval   = 1.0 / target_fps
    frame_idx  = 0
    state      = {"walk_phase": 0.0, "body_yaw": 0.0}
    prev_lc    = 487   # baseline label count
    prev_sz    = 20480 # baseline file size (~20KB)
    qa_answer  = ""
    snapshot: dict = {}
    snap_ts    = 0.0

    while True:
        t0 = time.monotonic()

        frame_path = frames[frame_idx % len(frames)]
        frame_idx += 1

        # Build cross-modal text description
        video_id = frame_path.stem.split("_")[0]
        title    = title_map.get(video_id, "Bovine dairy cow CC video")
        ctx      = CONTEXTS[frame_idx % len(CONTEXTS)]
        text     = f"{title}. {ctx}."

        # Train: image + text co-activation (cross-modal Hebbian linking)
        result = await asyncio.to_thread(call_media_train, frame_path, text, node)
        curr_lc = result.get("label_count", prev_lc)
        curr_sz = frame_path.stat().st_size

        # Motion proxy from fabric output
        motion_mag = motion_from_frame_delta(curr_lc, prev_lc, curr_sz, prev_sz)
        prev_lc = curr_lc
        prev_sz = curr_sz

        # Periodic snapshot + Q&A query
        now = time.monotonic()
        if now - snap_ts > 4.0:
            snapshot   = await asyncio.to_thread(call_neuro_snapshot, node)
            qa_answer  = await asyncio.to_thread(
                call_qa_query,
                "What is a dairy cow doing when its legs are in motion?",
                node,
            )
            snap_ts = now

        dt   = max(time.monotonic() - t0, 0.01)
        pose = compute_pose(motion_mag, qa_answer, snapshot, state, dt)

        # Read frame as data URI (no resize needed at 20KB; browser handles display)
        with open(frame_path, "rb") as f:
            pose["frame_b64"] = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

        await broadcast(json.dumps(pose))

        elapsed = time.monotonic() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(node: str, ws_port: int, http_port: int, fps: float):
    print("=" * 60)
    print("  W1z4rD V1510n -- Cow Sensor Stream (fabric-only)")
    print("=" * 60)
    print(f"  Node  -> http://{node}")
    print(f"  WS    -> ws://localhost:{ws_port}")

    threading.Thread(target=run_http_server, args=(http_port,), daemon=True).start()

    async with websockets.serve(ws_handler, "localhost", ws_port):
        await sensor_loop(node, fps)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--node",      default="localhost:8090")
    ap.add_argument("--ws-port",   type=int, default=8092)
    ap.add_argument("--http-port", type=int, default=8093)
    ap.add_argument("--fps",       type=float, default=4.0)
    args = ap.parse_args()
    try:
        asyncio.run(main(args.node, args.ws_port, args.http_port, args.fps))
    except KeyboardInterrupt:
        print("\nStopped.")
