#!/usr/bin/env python3
# coding: utf-8
"""
Real video -> photorealistic world viewer stream.

Reads a local video (looped), estimates per-frame depth using colour /
vertical-position / edge heuristics, and serves:

  GET /frame   -> JSON {w, h, frame_num, fps, image, depth}
                  image  = base64 JPEG of the colour frame
                  depth  = base64 JPEG of the depth map (0=far, 255=close)
  GET /health  -> "ok"

Also posts EnvironmentSnapshot to /neuro/train on the W1z4rD node so the
neural fabric learns the scene labels alongside the photorealistic view.

Usage:
  python stream_real_video.py [--video PATH] [--neuro HOST:PORT] [--fps N]
                              [--width W] [--height H]
"""

import argparse
import base64
import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VIDEO  = 'D:/w1z4rdv1510n-data/videos/cow_clean.f135.mp4'
DEFAULT_NEURO  = 'localhost:8090'
FRAME_PORT     = 9001
TARGET_W       = 854
TARGET_H       = 480
STREAM_FPS     = 25


# ── Depth estimation (no ML required) ────────────────────────────────────────
def estimate_depth(frame_bgr: np.ndarray,
                   bg_sub: cv2.BackgroundSubtractor) -> np.ndarray:
    """
    Returns a single-channel uint8 map where 0=far, 255=close.

    Layers used (blended):
      1. Vertical gradient  : lower pixels = closer (outdoor ground-level scene)
      2. Sky suppression    : blue-dominant upper pixels pushed to far
      3. Foreground mask    : MOG2 background subtraction -> close layer
      4. Edge sharpening    : Laplacian edges create depth boundaries
      5. Bilateral smooth   : edge-preserving final smooth
    """
    H, W = frame_bgr.shape[:2]
    f    = frame_bgr.astype(np.float32) / 255.0
    R, G, B = f[:, :, 2], f[:, :, 1], f[:, :, 0]

    # 1. Vertical gradient: y=0 (top) -> 0.12 (far),  y=H (bottom) -> 1.0 (close)
    yg    = np.linspace(0.12, 1.0, H)[:, np.newaxis] * np.ones((1, W), np.float32)
    depth = yg.copy()

    # 2. Sky: blue-dominant pixels -> push far.
    # Limit to upper 50% only — avoids misidentifying white cow markings as sky.
    sky_blue  = np.clip((B - np.maximum(R, G) * 1.05) / (B + 0.01), 0, 1).astype(np.float32)
    sky_mask  = np.clip(1.5 - yg * 3.0, 0, 1).astype(np.float32)   # upper 50% of frame only
    sky_conf  = sky_blue * sky_mask
    depth    -= sky_conf * 0.70

    # 3. Warm/brown objects (cow body) -> significantly closer
    warm     = np.clip((R - B) / (R + B + 0.01), 0, 1).astype(np.float32)
    warm    *= np.clip((yg - 0.12) * 3.5, 0, 1) * np.clip((0.90 - yg) * 3.5, 0, 1)
    depth   += warm * 0.32   # doubled warm boost for sharper cow separation

    # 4. Foreground mask via background subtractor (stabilises after ~10 frames)
    fg_mask  = bg_sub.apply(frame_bgr).astype(np.float32) / 255.0
    fg_mask  = cv2.GaussianBlur(fg_mask, (9, 9), 0)
    depth   += fg_mask * 0.28

    # 5. Edge sharpening: Laplacian edges create sharper depth boundaries
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges  = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    emx    = edges.max()
    if emx > 0:
        edges /= emx
    edges_s = cv2.GaussianBlur(edges, (5, 5), 0)
    depth  += edges_s * 0.06

    # Normalise + stronger bilateral smooth (preserves edges, smooths flat regions)
    depth   = np.clip(depth, 0.0, 1.0)
    depth8  = (depth * 255).astype(np.uint8)
    depth8  = cv2.bilateralFilter(depth8, 11, 75, 75)
    return depth8       # shape (H, W)  dtype uint8


# ── Per-frame cow sensor: drives the neural fabric with body-part observations ─
#
# Every video frame is treated as a sensor reading.  Computer-vision heuristics
# (colour, shape, background subtraction) estimate the 3D positions of visible
# cow body parts as best we can from a 2-D image.  These are posted to the
# neural fabric so its Hebbian centroids converge toward the current observation.
# The fabric's trained anatomy priors (from train_cow_anatomy.py) fill in parts
# that are not directly visible, producing a complete learned 3-D model that
# drives the Three.js scene in real time.

def _sym(sym_id: str, x: float, y: float, z: float, props: dict) -> dict:
    """Build a symbol dict for EnvironmentSnapshot."""
    return {
        'id':         sym_id,
        'type':       'CUSTOM',
        'position':   {'x': float(x), 'y': float(y), 'z': float(z)},
        'velocity':   {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'properties': props,
    }


def detect_objects(frame_bgr: np.ndarray, W: int, H: int,
                   bg_sub: cv2.BackgroundSubtractor | None = None) -> list:
    """
    Detect cow body parts in the frame and return them as EnvironmentSnapshot
    symbols.  Each body part that can be located is posted with its estimated
    3-D position (X = left/right, Y = height, Z = depth).

    Coordinate convention matches train_cow_anatomy.py:
      X ∈ [0,1]  left=0, right=1
      Y ∈ [0,1]  ground=0, top=1
      Z ∈ [0,1]  rear=0, front/nose=1
    """
    f   = frame_bgr.astype(np.float32) / 255.0
    R, G, B = f[:, :, 2], f[:, :, 1], f[:, :, 0]
    symbols = []

    # ── Cow body: brown / warm colours ────────────────────────────────────
    # Both Holstein (black+white) and Hereford (brown) covered.
    warm_mask  = ((R - B) > 0.14) & (R > 0.22) & (G > 0.12) & (G < 0.80)
    black_mask = (R < 0.18) & (G < 0.18) & (B < 0.18)          # Holstein black
    white_mask = (R > 0.72) & (G > 0.72) & (B > 0.72)           # Holstein / face white
    cow_mask   = warm_mask | black_mask

    ys, xs = np.where(cow_mask)
    if len(xs) < 400:
        # No cow found — post environment symbols only
        _add_env_symbols(symbols, f, R, G, B, W, H)
        return symbols

    # ── Bounding box ───────────────────────────────────────────────────────
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return symbols

    cx_img   = (x1 + x2) / 2 / W    # image-space normalised [0,1]
    cy_img   = (y1 + y2) / 2 / H
    area_frac = len(xs) / (W * H)

    # Depth estimate: larger cow → closer
    depth_z  = float(np.clip(0.70 - area_frac * 4.0, 0.15, 0.90))

    # ── Pose estimation ────────────────────────────────────────────────────
    # Separate "warm pixels" in upper vs lower half of bounding box
    upper_mask = cow_mask[:H // 2, :]
    lower_mask = cow_mask[H // 2:, :]
    n_upper    = int(np.sum(upper_mask))
    n_lower    = int(np.sum(lower_mask))
    head_low   = n_lower > n_upper * 1.6   # most mass in lower half -> grazing
    pose       = 'grazing' if head_low else 'standing'

    # Estimate facing direction from horizontal distribution skew
    left_mass  = int(np.sum(cow_mask[:, :W // 2]))
    right_mass = int(np.sum(cow_mask[:, W // 2:]))
    facing     = 'left' if left_mass > right_mass * 1.3 else (
                 'right' if right_mass > left_mass * 1.3 else 'camera')

    # ── Locate key body regions ────────────────────────────────────────────
    def _region_centroid(mask_region, col_offset, row_offset, fw, fh):
        ry, rx = np.where(mask_region)
        if len(rx) < 20:
            return None
        return ((rx.mean() + col_offset) / W,
                1.0 - (ry.mean() + row_offset) / H)   # Y flipped: image top=far=low in world

    # Head region: top 35% of bounding box
    head_h   = max(1, int(bh * 0.35))
    head_roi = cow_mask[y1:y1 + head_h, x1:x2]
    head_pos = _region_centroid(head_roi, x1, y1, W, H)

    # Neck region: next 15%
    neck_h   = max(1, int(bh * 0.15))
    neck_roi = cow_mask[y1 + head_h:y1 + head_h + neck_h, x1:x2]
    neck_pos = _region_centroid(neck_roi, x1, y1 + head_h, W, H)

    # Body region: middle 40%
    body_y1  = y1 + int(bh * 0.30)
    body_y2  = y1 + int(bh * 0.70)
    body_roi = cow_mask[body_y1:body_y2, x1:x2]
    body_pos = _region_centroid(body_roi, x1, body_y1, W, H)

    # Leg / hoof region: bottom 25%
    leg_y1   = y1 + int(bh * 0.75)
    leg_roi  = cow_mask[leg_y1:y2, x1:x2]
    # Split into left-front / right-rear hooves if wide enough
    half     = (x1 + x2) // 2
    llroi    = cow_mask[leg_y1:y2, x1:half]
    rlroi    = cow_mask[leg_y1:y2, half:x2]
    lhoof_pos = _region_centroid(llroi, x1,   leg_y1, W, H)
    rhoof_pos = _region_centroid(rlroi, half, leg_y1, W, H)

    # Foreground motion mask → detect leg motion (walking cue)
    motion = 0.0
    if bg_sub is not None:
        fg = bg_sub.apply(frame_bgr).astype(np.float32) / 255.0
        motion = float(fg[leg_y1:, x1:x2].mean())
    walking = motion > 0.12

    # Determine current behaviour label
    if walking:
        behaviour = 'walking_a'
    elif head_low:
        behaviour = 'grazing'
    else:
        behaviour = 'standing'

    # ── Emit body-part symbols ─────────────────────────────────────────────
    # Z-depth is shared for the whole cow (we can only estimate one depth
    # without stereo/lidar). X/Y come from the colour-region centroids above.

    def sym_if(key: str, pos, y_world: float | None = None):
        if pos is None:
            return
        px, py = pos
        y_out  = py if y_world is None else y_world
        symbols.append(_sym(f'cow_{key}', px, y_out, depth_z,
                            {'label': f'cow_{key}', 'pose': behaviour,
                             'type_label': 'body_part', 'scale_m': 1.4}))

    # Core skeleton from observed regions
    if head_pos:
        # Head top = poll, centre = head, bottom of head region = muzzle
        hx, hy = head_pos
        symbols.append(_sym('cow_head',    hx,        hy,        depth_z,
                            {'label': 'cow_head', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_muzzle',  hx,        max(0, hy - 0.12), depth_z,
                            {'label': 'cow_muzzle', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_poll',    hx,        min(1, hy + 0.08), depth_z,
                            {'label': 'cow_poll', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_eye_L',   max(0, hx - 0.05), hy + 0.02, depth_z,
                            {'label': 'cow_eye_L', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_eye_R',   min(1, hx + 0.05), hy + 0.02, depth_z,
                            {'label': 'cow_eye_R', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))

    sym_if('neck',         neck_pos)
    sym_if('withers',      neck_pos,    (neck_pos[1] + 0.06) if neck_pos else None)

    if body_pos:
        bx, by = body_pos
        symbols.append(_sym('cow_back',   bx, by + 0.10, depth_z,
                            {'label': 'cow_back', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_belly',  bx, max(0, by - 0.14), depth_z,
                            {'label': 'cow_belly', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_body_centre', bx, by, depth_z,
                            {'label': 'cow_body_centre', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_centre_of_mass', bx, max(0, by - 0.04), depth_z,
                            {'label': 'cow_centre_of_mass', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        # Rump / tail on opposite Z side
        symbols.append(_sym('cow_rump',   bx, by + 0.06, max(0.05, depth_z - 0.20),
                            {'label': 'cow_rump', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))

    if lhoof_pos:
        lx, _ = lhoof_pos
        symbols.append(_sym('cow_front_hoof_L', lx, 0.04, depth_z,
                            {'label': 'cow_front_hoof_L', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_shoulder_L', lx, (body_pos[1] + 0.05) if body_pos else 0.75, depth_z,
                            {'label': 'cow_shoulder_L', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
    if rhoof_pos:
        rx, _ = rhoof_pos
        symbols.append(_sym('cow_front_hoof_R', rx, 0.04, depth_z,
                            {'label': 'cow_front_hoof_R', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym('cow_shoulder_R', rx, (body_pos[1] + 0.05) if body_pos else 0.75, depth_z,
                            {'label': 'cow_shoulder_R', 'pose': behaviour, 'type_label': 'body_part', 'scale_m': 1.4}))

    # Top-level cow entity at detected centre
    symbols.append(_sym('cow', cx_img, 1.0 - cy_img, depth_z,
                        {'label': 'cow', 'pose': behaviour, 'behaviour': behaviour,
                         'facing': facing, 'type_label': 'animal',
                         'scale_m': 2.4, 'motion': float(motion)}))

    # ── Environment symbols ────────────────────────────────────────────────
    _add_env_symbols(symbols, f, R, G, B, W, H)
    return symbols


def _add_env_symbols(symbols: list, f, R, G, B, W: int, H: int) -> None:
    """Append sky and ground symbols (always present in outdoor field scene)."""
    sky_mask = (B > R * 1.05) & (B > G * 1.00)
    sy, sx   = np.where(sky_mask[:H // 3, :])
    if len(sx) > 200:
        symbols.append(_sym('env_sky', 0.5, 0.90, 0.95,
                            {'label': 'env_sky', 'type_label': 'environment', 'scale_m': 100.0}))
    grnd_mask = ((G - R) > -0.10) & ((G - B) > 0.02) & (G > 0.15)
    gy, gx   = np.where(grnd_mask[H // 2:, :])
    if len(gx) > 300:
        symbols.append(_sym('env_grass', 0.5, 0.05, 0.50,
                            {'label': 'env_grass', 'type_label': 'environment', 'scale_m': 50.0}))


# ── Shared frame state ────────────────────────────────────────────────────────
class FrameState:
    def __init__(self):
        self.payload = b'{}'
        self.lock    = threading.Lock()

_state = FrameState()


# ── HTTP handler ──────────────────────────────────────────────────────────────
class FrameHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # silence access logs

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/frame'):
            with _state.lock:
                body = _state.payload
            self.send_response(200)
            self.send_header('Content-Type',   'application/json')
            self.send_header('Content-Length', str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/health':
            self.send_response(200)
            self._cors()
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_response(404)
            self.end_headers()


def _run_http():
    server = HTTPServer(('0.0.0.0', FRAME_PORT), FrameHandler)
    server.serve_forever()


# ── Neuro poster ──────────────────────────────────────────────────────────────
def _post_neuro(neuro_host: str, symbols: list) -> bool:
    snapshot = {
        'timestamp': {'unix': int(time.time() * 1000)},
        'bounds':    {'x': 1.0, 'y': 1.0, 'z': 1.0},
        'symbols':   symbols,
        'metadata':  {'context': 'field', 'modality': 'video_stream'},
    }
    body = json.dumps({'snapshot': snapshot}).encode()
    req  = urllib.request.Request(
        f'http://{neuro_host}/neuro/train', data=body,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


# ── Capture loop (main thread) ────────────────────────────────────────────────
def capture_loop(video_path: str, neuro_host: str, fps: int,
                 target_w: int, target_h: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'ERROR: Cannot open video: {video_path}', flush=True)
        sys.exit(1)

    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or fps
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = 1.0 / fps

    print(f'  Video : {vid_w}x{vid_h} @ {vid_fps:.1f}fps  ({n_frames} frames)', flush=True)
    print(f'  Serve : {target_w}x{target_h} @ {fps}fps  ->  http://localhost:{FRAME_PORT}/frame',
          flush=True)

    # Background subtractor for foreground detection
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=False)

    frame_num = 0
    neuro_ok  = False
    drift     = 0.0

    while True:
        t0   = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=40, detectShadows=False)
            continue

        # Resize to target
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)

        # Depth at half-res (faster, still good quality)
        small   = cv2.resize(frame, (target_w // 2, target_h // 2))
        depth_s = estimate_depth(small, bg_sub)
        depth   = cv2.resize(depth_s, (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)

        # Encode colour as JPEG quality 90, depth quality 82
        _, img_jpg  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, dep_jpg  = cv2.imencode('.jpg', depth, [cv2.IMWRITE_JPEG_QUALITY, 82])

        payload = json.dumps({
            'w': target_w, 'h': target_h,
            'frame_num': frame_num,
            'fps': float(fps),
            'image': base64.b64encode(bytes(img_jpg)).decode(),
            'depth': base64.b64encode(bytes(dep_jpg)).decode(),
        }).encode()

        with _state.lock:
            _state.payload = payload

        # Post to neuro every 6 frames
        if frame_num % 6 == 0:
            syms     = detect_objects(frame, target_w, target_h, bg_sub)
            if syms:
                neuro_ok = _post_neuro(neuro_host, syms)

        # Progress log every 5 seconds
        if frame_num % (fps * 5) == 0:
            kb = len(payload) // 1024
            print(f'  frame {frame_num:06d}  payload={kb}KB  neuro={"ok" if neuro_ok else "off"}',
                  flush=True)

        frame_num += 1
        elapsed = time.perf_counter() - t0
        wait    = interval - elapsed - drift
        if wait > 0:
            time.sleep(wait)
            drift = 0.0
        else:
            drift = -wait * 0.1


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Real video -> photorealistic frame server')
    ap.add_argument('--video',   default=DEFAULT_VIDEO, help='Video file path')
    ap.add_argument('--neuro',   default=DEFAULT_NEURO, help='Node host:port')
    ap.add_argument('--fps',     type=int, default=STREAM_FPS, help='Stream FPS')
    ap.add_argument('--width',   type=int, default=TARGET_W)
    ap.add_argument('--height',  type=int, default=TARGET_H)
    args = ap.parse_args()

    print('Real video frame server starting...', flush=True)
    print(f'  Video: {args.video}', flush=True)
    print(f'  Neuro: {args.neuro}', flush=True)

    # Start HTTP server in daemon thread
    t = threading.Thread(target=_run_http, daemon=True)
    t.start()

    capture_loop(args.video, args.neuro, args.fps, args.width, args.height)


if __name__ == '__main__':
    main()
