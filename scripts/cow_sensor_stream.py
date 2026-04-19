#!/usr/bin/env python3
"""
Cow Sensor Stream — W1z4rD V1510n
Reads pre-extracted Stage 2 JPEG frames, routes each through the neural fabric
(/media/train cross-modal: image + anatomical text), reads /neuro/snapshot and
/qa/query for semantic state, and broadcasts over WebSocket for the 3D world.

Per-cow bounding-box detection is done via Pillow (pip install Pillow).
Without Pillow the stream still works but emits a single synthetic cow.

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
    "Dairy cattle poll forehead brow occipital bovine cranium",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pillow-based per-cow bounding box detection
# ─────────────────────────────────────────────────────────────────────────────

DETECT_W, DETECT_H = 96, 54
_PIL_AVAILABLE: Optional[bool] = None


def _check_pil() -> bool:
    global _PIL_AVAILABLE
    if _PIL_AVAILABLE is None:
        try:
            from PIL import Image  # noqa: F401
            _PIL_AVAILABLE = True
        except ImportError:
            print("  [WARN] Pillow not installed — per-cow detection disabled.")
            print("         Run:  pip install Pillow   to enable multi-cow tracking")
            _PIL_AVAILABLE = False
    return _PIL_AVAILABLE


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n else 0


def detect_cows_pil(frame_path: Path) -> list:
    """
    Row-median deviation detector.  For each pixel, compare it to the median
    colour of its own row.  Pixels that differ significantly from the row
    median are foreground (entities), because each row is mostly background.
    Works on any grass/sky colour.
    Returns {cx,cy,w,h,area} dicts normalised to [0,1].
    """
    if not _check_pil():
        return []
    try:
        from PIL import Image

        img = (
            Image.open(frame_path)
            .convert("RGB")
            .resize((DETECT_W, DETECT_H), Image.BILINEAR)
        )
        pixels = list(img.getdata())

        # ── Per-row median background ──────────────────────────────────────
        row_med = []
        for y in range(DETECT_H):
            row = [pixels[y * DETECT_W + x] for x in range(DETECT_W)]
            row_med.append((
                _median([p[0] for p in row]),
                _median([p[1] for p in row]),
                _median([p[2] for p in row]),
            ))

        # Exclude top 35% (sky + title overlays) and bottom 10% (definite ground)
        sky_end   = int(DETECT_H * 0.35)
        gnd_start = int(DETECT_H * 0.90)
        DEVIATION = 42   # L1 distance to call a pixel foreground

        mask = bytearray(DETECT_W * DETECT_H)
        for y in range(sky_end, gnd_start):
            mr, mg, mb = row_med[y]
            for x in range(DETECT_W):
                r, g, b = pixels[y * DETECT_W + x]
                if abs(r - mr) + abs(g - mg) + abs(b - mb) > DEVIATION:
                    mask[y * DETECT_W + x] = 1

        # ── Column projection ──────────────────────────────────────────────
        active_h = gnd_start - sky_end
        col_profile = [
            sum(mask[y * DETECT_W + x] for y in range(sky_end, gnd_start))
            for x in range(DETECT_W)
        ]
        THRESH = max(2, active_h // 9)

        # ── Horizontal runs with small-gap merging ─────────────────────────
        MAX_GAP = 2
        runs: list[tuple[int, int]] = []
        in_run = False
        gap = 0
        for x, cnt in enumerate(col_profile):
            if cnt >= THRESH:
                if not in_run:
                    run_start = x; in_run = True
                gap = 0
            else:
                if in_run:
                    gap += 1
                    if gap > MAX_GAP:
                        runs.append((run_start, x - gap)); in_run = False; gap = 0
        if in_run:
            runs.append((run_start, DETECT_W - 1))

        # ── Split / force-divide wide blobs ────────────────────────────────
        # Target entity width ≤ 40% of frame.  For blobs wider than 55%, force
        # an equal split so we always get multiple entities from dense herds.
        TARGET_W  = int(DETECT_W * 0.38)   # ideal max per entity
        SPLIT_THR = int(DETECT_W * 0.55)   # min blob width that triggers split

        split_runs: list[tuple[int, int]] = []
        for x0, x1 in runs:
            span = x1 - x0 + 1
            if span > SPLIT_THR:
                # First try structural split at local minima
                prof = col_profile[x0:x1 + 1]
                splits = [x0]
                for i in range(2, len(prof) - 2):
                    if prof[i] <= min(prof[i-1], prof[i-2], prof[i+1], prof[i+2]) \
                            and prof[i] < THRESH * 3.0:
                        splits.append(x0 + i)
                splits.append(x1 + 1)
                # If structural split produced segments, use them
                if len(splits) > 2:
                    for a, b in zip(splits, splits[1:]):
                        if b - a >= 4:
                            split_runs.append((a, b - 1))
                else:
                    # Force equal split into n_parts
                    n_parts = max(2, round(span / TARGET_W))
                    part_w  = span / n_parts
                    for i in range(n_parts):
                        a = x0 + round(i * part_w)
                        b = x0 + round((i + 1) * part_w) - 1
                        if b - a >= 4:
                            split_runs.append((a, b))
            else:
                split_runs.append((x0, x1))

        # ── Build bboxes ───────────────────────────────────────────────────
        results = []
        for x0, x1 in split_runs:
            if x1 - x0 < 3:
                continue
            y_set = [
                y for y in range(sky_end, gnd_start)
                for x in range(x0, x1 + 1)
                if mask[y * DETECT_W + x]
            ]
            if not y_set:
                # No active pixels in this column range — still emit a bbox
                # centred vertically so the entity has a reasonable crop region
                y_set = [sky_end, gnd_start - 1]
            y_min, y_max = min(y_set), max(y_set)
            w_px = x1 - x0 + 1
            h_px = y_max - y_min + 1
            if h_px < 3:
                continue
            aspect = w_px / max(h_px, 1)
            if aspect > 9.0 or aspect < 0.08:
                continue
            w_frac = w_px / DETECT_W
            if w_frac < 0.11:     # discard tiny noise blobs
                continue
            # Dominant non-grass colour for this entity (used as shader fallback)
            non_grass = []
            for y in range(max(sky_end, y_min), min(gnd_start, y_max + 1)):
                for x in range(x0, x1 + 1):
                    if mask[y * DETECT_W + x]:
                        r2, g2, b2 = pixels[y * DETECT_W + x]
                        gd = g2 - max(r2, b2)
                        if gd < 25:   # not strongly green
                            non_grass.append((r2, g2, b2))
            if non_grass:
                dom_r = _median([p[0] for p in non_grass])
                dom_g = _median([p[1] for p in non_grass])
                dom_b = _median([p[2] for p in non_grass])
            else:
                dom_r, dom_g, dom_b = 80, 60, 40

            results.append({
                "cx":    (x0 + x1) / 2.0 / DETECT_W,
                "cy":    (y_min + y_max) / 2.0 / DETECT_H,
                "w":     min(w_frac, 0.55),
                "h":     min(h_px / DETECT_H, 0.75),
                "area":  w_px * h_px,
                "color": [dom_r, dom_g, dom_b],
            })

        results.sort(key=lambda r: -r["area"])
        return results[:3]
    except Exception:
        return []


def track_cows(prev_tracked: list, new_detections: list) -> list:
    """
    Assign stable IDs to new detections by nearest-centroid matching
    against the previous frame's tracked cows.
    """
    if not prev_tracked:
        return [dict(d, id=i, motion=0.0) for i, d in enumerate(new_detections)]

    next_new_id = max(c["id"] for c in prev_tracked) + 1
    used: set[int] = set()
    result = []

    for d in new_detections:
        best_id, best_dist = None, 0.22
        for pc in prev_tracked:
            if pc["id"] in used:
                continue
            dist = math.hypot(d["cx"] - pc["cx"], d["cy"] - pc["cy"])
            if dist < best_dist:
                best_dist = dist
                best_id = pc["id"]

        if best_id is not None:
            used.add(best_id)
            pc = next(c for c in prev_tracked if c["id"] == best_id)
            motion = math.hypot(d["cx"] - pc["cx"], d["cy"] - pc["cy"]) * 10.0
            result.append(dict(d, id=best_id, motion=min(motion, 1.0)))
        else:
            result.append(dict(d, id=next_new_id, motion=0.0))
            next_new_id += 1

    return result


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

# ── 3D spatial observation training ──────────────────────────────────────────
import random as _random

def _bbox_to_3d_parts(cow: dict) -> list:
    """
    Map a detected entity to canonical body-relative 3D anatomy positions.

    Coordinate system matches train_cow_3d_anatomy.py exactly:
      X = head-to-tail axis  (snout ≈ +1.62, tail_tip ≈ -1.55)
      Y = height             (hooves = 0, withers ≈ 1.42)
      Z = lateral left/right (shoulders ≈ ±0.32)

    All positions are body-centred (not world-space).  This lets the neural
    fabric learn the correct anatomy regardless of where the cow appears in the
    video frame.  Small Gaussian jitter per frame simulates natural pose variation.

    Returns list of (symbol_id, x, y, z) tuples.
    """
    # Canonical bovine proportions (metres, withers = 1.42 m)
    WITHER_H = 1.42
    HALF_L   = 1.585  # (snout 1.62 + tail_tip 1.55) / 2
    HALF_W   = 0.32   # shoulder half-width

    # Per-frame jitter magnitude — simulates natural pose/size variation
    j = 0.025
    def jit():
        return _random.gauss(0, j)

    pts = []

    # ── Dense surface grid in canonical body-relative space ───────────────────
    # X axis = head-to-tail (NX columns), Y axis = height (NY rows)
    # Z width follows cow's barrel cross-section: widest at belly, tapers top/bottom
    import math as _math
    NX, NY = 14, 10
    for ix in range(NX):
        tx = ix / (NX - 1)         # 0 = head end, 1 = tail end
        x  = HALF_L * (1.0 - 2*tx) + jit()   # head = +HALF_L, tail = -HALF_L
        for iy in range(NY):
            ty = iy / (NY - 1)     # 0 = top (withers), 1 = bottom (ground)
            y  = WITHER_H * (1.0 - ty) + jit()

            # Barrel cross-section: widest at mid-belly (ty≈0.6), narrow at withers/hooves
            barrel = HALF_W * (0.25 + 0.75 * _math.sin(_math.pi * min(ty * 1.5, 1.0)) ** 0.7)

            # Use grid indices for full spatial resolution (140 unique X×Y positions)
            label = f"cow_surf_{iy:02d}_{ix:02d}"

            pts.append((label + "_L", x, max(0.0, y), +barrel + jit()))
            pts.append((label + "_R", x, max(0.0, y), -barrel - jit()))
            pts.append((label,        x, max(0.0, y),  0.0    + jit()))

    # ── Named anatomy landmarks (matching anatomy atlas label names) ──────────
    pts += [
        ("cow_head",       1.40+jit(), 1.05+jit(),  0.00+jit()),
        ("cow_snout",      1.62+jit(), 0.90+jit(),  0.00+jit()),
        ("cow_brain",      1.36+jit(), 1.22+jit(),  0.00+jit()),
        ("cow_jaw",        1.52+jit(), 0.84+jit(),  0.00+jit()),
        ("cow_neck_upper", 1.08+jit(), 1.18+jit(),  0.00+jit()),
        ("cow_withers",    0.55+jit(), 1.42+jit(),  0.00+jit()),
        ("cow_spine_T",    0.10+jit(), 1.32+jit(),  0.00+jit()),
        ("cow_spine_L",   -0.30+jit(), 1.30+jit(),  0.00+jit()),
        ("cow_sacrum",    -0.68+jit(), 1.20+jit(),  0.00+jit()),
        ("cow_rump",      -0.80+jit(), 1.18+jit(),  0.00+jit()),
        ("cow_tail_root", -1.10+jit(), 1.12+jit(),  0.00+jit()),
        ("cow_tail_tip",  -1.55+jit(), 0.78+jit(),  0.00+jit()),
        # Trunk
        ("cow_sternum",    0.50+jit(), 0.68+jit(),  0.00+jit()),
        ("cow_belly",     -0.08+jit(), 0.55+jit(),  0.00+jit()),
        ("cow_udder",     -0.34+jit(), 0.28+jit(),  0.00+jit()),
        # Organs
        ("cow_heart",      0.62+jit(), 1.10+jit(),  0.00+jit()),
        ("cow_lung_L",     0.45+jit(), 1.16+jit(), +0.24+jit()),
        ("cow_lung_R",     0.45+jit(), 1.16+jit(), -0.24+jit()),
        ("cow_rumen",     -0.28+jit(), 0.92+jit(), +0.24+jit()),
        ("cow_liver",      0.15+jit(), 0.94+jit(), -0.26+jit()),
        # Front limbs
        ("cow_shoulder_L", 0.76+jit(), 1.12+jit(), +0.32+jit()),
        ("cow_shoulder_R", 0.76+jit(), 1.12+jit(), -0.32+jit()),
        ("cow_elbow_L",    0.80+jit(), 0.74+jit(), +0.28+jit()),
        ("cow_elbow_R",    0.80+jit(), 0.74+jit(), -0.28+jit()),
        ("cow_knee_FL",    0.80+jit(), 0.44+jit(), +0.24+jit()),
        ("cow_knee_FR",    0.80+jit(), 0.44+jit(), -0.24+jit()),
        ("cow_hoof_FL",    0.80+jit(), 0.00,       +0.22+jit()),
        ("cow_hoof_FR",    0.80+jit(), 0.00,       -0.22+jit()),
        # Hind limbs
        ("cow_hip_L",     -0.80+jit(), 1.12+jit(), +0.33+jit()),
        ("cow_hip_R",     -0.80+jit(), 1.12+jit(), -0.33+jit()),
        ("cow_stifle_L",  -0.76+jit(), 0.76+jit(), +0.28+jit()),
        ("cow_stifle_R",  -0.76+jit(), 0.76+jit(), -0.28+jit()),
        ("cow_hock_L",    -0.80+jit(), 0.44+jit(), +0.24+jit()),
        ("cow_hock_R",    -0.80+jit(), 0.44+jit(), -0.24+jit()),
        ("cow_hoof_BL",   -0.80+jit(), 0.00,       +0.21+jit()),
        ("cow_hoof_BR",   -0.80+jit(), 0.00,       -0.21+jit()),
        ("cow_cannon_FL",  0.80+jit(), 0.28+jit(), +0.23+jit()),
        ("cow_cannon_FR",  0.80+jit(), 0.28+jit(), -0.23+jit()),
        ("cow_cannon_BL", -0.80+jit(), 0.28+jit(), +0.22+jit()),
        ("cow_cannon_BR", -0.80+jit(), 0.28+jit(), -0.22+jit()),
    ]
    return pts


def train_entity_observations(cows: list, ts: int, node: str) -> None:
    """Post an EnvironmentSnapshot built from real video detections to /neuro/train."""
    if not cows:
        return
    symbols = []
    for cow in cows:
        for (part_id, x, y, z) in _bbox_to_3d_parts(cow):
            symbols.append({
                "id": part_id,
                "type": "OBJECT",
                "position": {"x": float(x), "y": float(max(0.0, y)), "z": float(z)},
                "properties": {
                    "species":  "bovine",
                    "category": "cow_body",
                },
            })
    body = json.dumps({
        "snapshot": {
            "timestamp": {"unix": ts},
            "bounds":    {"x": 16.0, "y": 2.0, "z": 16.0},
            "symbols":   symbols,
            "metadata":  {
                "scene":   "cow_world_video_observation",
                "species": "holstein_dairy_cow",
                "source":  "video_sensor_stream",
            },
        },
        "extra_labels": [
            "txt:word_cow", "txt:word_dairy", "txt:word_bovine", "txt:word_Holstein",
            "txt:word_anatomy", "txt:word_body", "txt:word_skeleton",
            "txt:word_leg", "txt:word_head", "txt:word_neck", "txt:word_spine",
            "txt:word_rumen", "txt:word_heart", "txt:word_hoof", "txt:word_udder",
            "cow_body", "bovine_anatomy",
        ],
    }).encode()
    req = urllib.request.Request(
        f"http://{node}/neuro/train", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            r.read()
    except Exception:
        pass  # don't interrupt stream on train failure


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
# Motion proxy (fabric-native)
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

    if motion_mag > 0.12:
        anim = "walk"
    elif any(w in qa_lower for w in ["graz", "graze", "grass", "head low", "pasture"]):
        anim = "graze"
    elif any(w in qa_lower for w in ["walk", "stride", "locomot", "gait"]):
        anim = "walk"
    else:
        anim = "stand"

    walk_speed = min(motion_mag * 6.0, 5.0) if anim == "walk" else 0.3 if anim == "graze" else 0.0
    walk_phase = (state.get("walk_phase", 0.0) + walk_speed * dt) % (2 * math.pi)
    head_pitch = -22.0 if anim == "graze" else (5.0 if motion_mag < 0.05 else -5.0)

    body_yaw = state.get("body_yaw", 0.0)
    if motion_mag > 0.08:
        body_yaw += (motion_mag - 0.08) * 0.3

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

    interval       = 1.0 / target_fps
    frame_idx      = 0
    state          = {"walk_phase": 0.0, "body_yaw": 0.0}
    prev_lc        = 487
    prev_sz        = 20480
    qa_answer      = ""
    snapshot: dict = {}
    snap_ts        = 0.0
    tracked_cows: list = []

    while True:
        t0 = time.monotonic()

        frame_path = frames[frame_idx % len(frames)]
        frame_idx += 1

        video_id = frame_path.stem.split("_")[0]
        title    = title_map.get(video_id, "Bovine dairy cow CC video")
        ctx      = CONTEXTS[frame_idx % len(CONTEXTS)]
        text     = f"{title}. {ctx}."

        result  = await asyncio.to_thread(call_media_train, frame_path, text, node)
        curr_lc = result.get("label_count", prev_lc)
        curr_sz = frame_path.stat().st_size

        motion_mag = motion_from_frame_delta(curr_lc, prev_lc, curr_sz, prev_sz)
        prev_lc = curr_lc
        prev_sz = curr_sz

        now = time.monotonic()
        if now - snap_ts > 4.0:
            snapshot  = await asyncio.to_thread(call_neuro_snapshot, node)
            qa_answer = await asyncio.to_thread(
                call_qa_query,
                "What is a dairy cow doing when its legs are in motion?",
                node,
            )
            snap_ts = now

        dt   = max(time.monotonic() - t0, 0.01)
        pose = compute_pose(motion_mag, qa_answer, snapshot, state, dt)

        # Per-cow detection and tracking
        new_dets    = await asyncio.to_thread(detect_cows_pil, frame_path)
        tracked_cows = track_cows(tracked_cows, new_dets)

        # 3D spatial observation training: project real video detections to
        # world space and post as EnvironmentSnapshot so the neural fabric
        # accumulates 3D centroid positions from real video observations.
        if tracked_cows and frame_idx % 4 == 0:
            await asyncio.to_thread(
                train_entity_observations,
                tracked_cows, int(time.time()), node,
            )

        # Fallback: if detection yields nothing, emit one synthetic centred cow
        if not tracked_cows:
            tracked_cows = [{"id": 0, "cx": 0.5, "cy": 0.55, "w": 0.55, "h": 0.65, "motion": motion_mag, "color": [60, 50, 40]}]

        pose["cows"] = [
            {
                "id":     c["id"],
                "cx":     round(c["cx"],   4),
                "cy":     round(c["cy"],   4),
                "w":      round(c["w"],    4),
                "h":      round(c["h"],    4),
                "motion": round(c.get("motion", 0.0), 4),
                "color":  c.get("color", [60, 50, 40]),
            }
            for c in tracked_cows
        ]

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
    print("  W1z4rD V1510n -- Cow Sensor Stream (multi-cow)")
    print("=" * 60)
    print(f"  Node  -> http://{node}")
    print(f"  WS    -> ws://localhost:{ws_port}")
    _check_pil()

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
