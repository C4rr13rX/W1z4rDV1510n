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
DEFAULT_VIDEO  = 'D:/w1z4rdv1510n-data/videos/cow_field.webm'
DEFAULT_NEURO  = 'localhost:8090'
FRAME_PORT     = 9001
TARGET_W       = 640
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
    symbols with correct 3-D coordinates for a side-view camera.

    Coordinate convention matches train_cow_anatomy.py:
      X ∈ [0,1]  bilateral: left=0, right=1  (0.5 = midline)
      Y ∈ [0,1]  height: ground=0, top=1
      Z ∈ [0,1]  front-back: rear=0, nose=1

    SIDE-VIEW MAPPING
    ─────────────────
    In a side-view video the camera looks along the cow's bilateral X axis.
    The observable image axes are:
      • image horizontal (pixel_x) → cow Z (front-back)  [directly measurable]
      • image vertical   (pixel_y) → cow Y (height)      [directly measurable]
      • bilateral X → NOT observable from the side; use anatomy prior offsets

    The bilateral X offset is fixed per body part (0.5 for midline, ±0.12–0.15
    for bilateral pairs) so the video sensor only updates Y and Z.
    """
    f   = frame_bgr.astype(np.float32) / 255.0
    R, G, B = f[:, :, 2], f[:, :, 1], f[:, :, 0]
    symbols = []

    # ── Cow body: brown / warm colours ────────────────────────────────────
    warm_mask  = ((R - B) > 0.14) & (R > 0.22) & (G > 0.12) & (G < 0.80)
    black_mask = (R < 0.18) & (G < 0.18) & (B < 0.18)   # Holstein black
    cow_mask   = warm_mask | black_mask

    ys, xs = np.where(cow_mask)
    if len(xs) < 400:
        _add_env_symbols(symbols, f, R, G, B, W, H)
        return symbols

    # ── Bounding box ───────────────────────────────────────────────────────
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return symbols

    cx_img    = (x1 + x2) / 2 / W
    cy_img    = (y1 + y2) / 2 / H
    area_frac = len(xs) / (W * H)

    # ── Facing direction: which end has the head? ─────────────────────────
    # The head end of a cow is narrower at the top (poll/muzzle protrude less
    # horizontally than the broad rump). Detect by comparing top-half pixel
    # density in left vs right thirds of the bounding box.
    th = max(1, bh // 2)
    left_third  = int(np.sum(cow_mask[y1:y1+th, x1        : x1+bw//3]))
    right_third = int(np.sum(cow_mask[y1:y1+th, x1+2*bw//3: x2      ]))
    # Skew of overall mass also helps:
    left_mass   = int(np.sum(cow_mask[:, :W//2]))
    right_mass  = int(np.sum(cow_mask[:, W//2:]))
    # Head is on the lighter / narrower side at top
    if left_third < right_third * 0.80:
        facing = 'left'    # head on LEFT side of image
    elif right_third < left_third * 0.80:
        facing = 'right'   # head on RIGHT side of image
    elif left_mass > right_mass * 1.25:
        facing = 'right'
    elif right_mass > left_mass * 1.25:
        facing = 'left'
    else:
        facing = 'right'   # default assumption

    # ── Coordinate converters ─────────────────────────────────────────────
    # to_z: image pixel_x (0–W) → neural Z (rear=0, nose=1)
    if facing == 'right':
        def to_z(px): return float(np.clip(px / W, 0.0, 1.0))
    else:
        def to_z(px): return float(np.clip(1.0 - px / W, 0.0, 1.0))

    def to_y(py): return float(np.clip(1.0 - py / H, 0.0, 1.0))

    # ── Region centroid helpers (returns absolute pixel coords) ───────────
    def _centroid_px(mask_region, col_off, row_off):
        """Returns (mean_px_x, mean_px_y) in frame pixels, or None."""
        ry, rx = np.where(mask_region)
        if len(rx) < 20:
            return None
        return (float(rx.mean()) + col_off, float(ry.mean()) + row_off)

    # ── Pose estimation ───────────────────────────────────────────────────
    upper_n = int(np.sum(cow_mask[:H//2, :]))
    lower_n = int(np.sum(cow_mask[H//2:, :]))
    head_low   = lower_n > upper_n * 1.6
    motion     = 0.0
    leg_y1     = y1 + int(bh * 0.75)
    if bg_sub is not None:
        fg     = bg_sub.apply(frame_bgr).astype(np.float32) / 255.0
        motion = float(fg[leg_y1:, x1:x2].mean())
    walking    = motion > 0.12
    if walking:
        behaviour = 'walking_a'
    elif head_low:
        behaviour = 'grazing'
    else:
        behaviour = 'standing'

    # ── Locate body regions in pixel space ────────────────────────────────
    # The bounding box horizontal axis = cow front-back.
    # Split into zones by fraction of bbox width:
    #   head zone:    forward 25% (nearest nose end)
    #   neck zone:    next 15%
    #   body zone:    middle 40%
    #   rump zone:    rear 20%
    # "Forward" = right side of bbox if facing right, left side if facing left.

    if facing == 'right':
        head_x1 = x1 + int(bw * 0.75);  head_x2 = x2
        neck_x1 = x1 + int(bw * 0.60);  neck_x2 = x1 + int(bw * 0.75)
        body_x1 = x1 + int(bw * 0.25);  body_x2 = x1 + int(bw * 0.65)
        rump_x1 = x1;                   rump_x2 = x1 + int(bw * 0.25)
        # Front legs = forward half of leg region; rear = back half
        front_leg_x1 = (x1 + x2)//2;    front_leg_x2 = x2
        rear_leg_x1  = x1;              rear_leg_x2  = (x1 + x2)//2
    else:  # facing left
        head_x1 = x1;                   head_x2 = x1 + int(bw * 0.25)
        neck_x1 = x1 + int(bw * 0.25);  neck_x2 = x1 + int(bw * 0.40)
        body_x1 = x1 + int(bw * 0.35);  body_x2 = x1 + int(bw * 0.75)
        rump_x1 = x1 + int(bw * 0.75);  rump_x2 = x2
        front_leg_x1 = x1;              front_leg_x2 = (x1 + x2)//2
        rear_leg_x1  = (x1 + x2)//2;   rear_leg_x2  = x2

    # Vertical zones (shared)
    head_y1  = y1;               head_y2  = y1 + int(bh * 0.60)
    upper_y1 = y1;               upper_y2 = y1 + int(bh * 0.35)
    body_y1_ = y1 + int(bh*0.20); body_y2_ = y1 + int(bh * 0.75)
    leg_y2   = y2

    head_c  = _centroid_px(cow_mask[head_y1:head_y2,   head_x1:head_x2],  head_x1, head_y1)
    neck_c  = _centroid_px(cow_mask[upper_y1:upper_y2, neck_x1:neck_x2],  neck_x1, upper_y1)
    body_c  = _centroid_px(cow_mask[body_y1_:body_y2_, body_x1:body_x2],  body_x1, body_y1_)
    rump_c  = _centroid_px(cow_mask[upper_y1:upper_y2, rump_x1:rump_x2],  rump_x1, upper_y1)
    fhoof_c = _centroid_px(cow_mask[leg_y1:leg_y2, front_leg_x1:front_leg_x2], front_leg_x1, leg_y1)
    rhoof_c = _centroid_px(cow_mask[leg_y1:leg_y2, rear_leg_x1:rear_leg_x2],   rear_leg_x1,  leg_y1)

    # ── Build symbol for a midline part ───────────────────────────────────
    def emit_mid(key, px, py, x_bilateral=0.50):
        z = to_z(px); y = to_y(py)
        symbols.append(_sym(f'cow_{key}', x_bilateral, y, z,
                            {'label': f'cow_{key}', 'pose': behaviour,
                             'type_label': 'body_part', 'scale_m': 1.4}))

    def emit_pair(key_l, key_r, px, py, xl=0.37, xr=0.63):
        z = to_z(px); y = to_y(py)
        symbols.append(_sym(f'cow_{key_l}', xl, y, z,
                            {'label': f'cow_{key_l}', 'pose': behaviour,
                             'type_label': 'body_part', 'scale_m': 1.4}))
        symbols.append(_sym(f'cow_{key_r}', xr, y, z,
                            {'label': f'cow_{key_r}', 'pose': behaviour,
                             'type_label': 'body_part', 'scale_m': 1.4}))

    # ── HEAD ──────────────────────────────────────────────────────────────
    if head_c:
        hpx, hpy = head_c
        emit_mid('head',   hpx, hpy)
        emit_mid('muzzle', hpx, hpy + bh * 0.10)   # muzzle: slightly lower in image
        emit_mid('poll',   hpx, hpy - bh * 0.06)   # poll: slightly higher
        emit_pair('eye_L', 'eye_R', hpx, hpy, xl=0.45, xr=0.55)
        emit_pair('ear_L', 'ear_R', hpx, hpy - bh * 0.04, xl=0.37, xr=0.63)

    # ── NECK / WITHERS ────────────────────────────────────────────────────
    if neck_c:
        npx, npy = neck_c
        emit_mid('neck',    npx, npy)
        emit_mid('withers', npx, npy - bh * 0.05)   # withers: top of shoulders, just above neck line
        emit_pair('shoulder_L', 'shoulder_R', npx, npy + bh * 0.06)

    # ── BODY (back / belly / brisket) ─────────────────────────────────────
    if body_c:
        bpx, bpy = body_c
        emit_mid('back',          bpx, bpy - bh * 0.08)   # back is near top of body
        emit_mid('belly',         bpx, bpy + bh * 0.10)   # belly hangs lower
        emit_mid('body_centre',   bpx, bpy)
        emit_mid('centre_of_mass',bpx, bpy + bh * 0.03)
        emit_mid('loin',          bpx, bpy - bh * 0.06)
        # Brisket: lower-front of body (same Z as neck, lower Y)
        if neck_c:
            emit_mid('brisket', neck_c[0], bpy + bh * 0.08)

    # ── RUMP / TAIL ───────────────────────────────────────────────────────
    if rump_c:
        rpx, rpy = rump_c
        emit_mid('rump',      rpx, rpy)
        emit_mid('tail_root', rpx, rpy + bh * 0.06)
        emit_mid('tail_mid',  rpx, rpy + bh * 0.15)
        emit_pair('hip_L', 'hip_R', rpx, rpy, xl=0.37, xr=0.63)

    # ── FRONT HOOVES ──────────────────────────────────────────────────────
    if fhoof_c:
        fpx, _ = fhoof_c
        emit_pair('front_hoof_L',   'front_hoof_R',   fpx, y2 - 2, xl=0.36, xr=0.64)
        emit_pair('front_cannon_L', 'front_cannon_R', fpx, y2 - bh*0.22, xl=0.36, xr=0.64)
        emit_pair('elbow_L',        'elbow_R',        fpx, y2 - bh*0.42, xl=0.35, xr=0.65)

    # ── REAR HOOVES ───────────────────────────────────────────────────────
    if rhoof_c:
        rpx2, _ = rhoof_c
        emit_pair('rear_hoof_L',   'rear_hoof_R',   rpx2, y2 - 2, xl=0.37, xr=0.63)
        emit_pair('rear_cannon_L', 'rear_cannon_R', rpx2, y2 - bh*0.22, xl=0.37, xr=0.63)
        emit_pair('hock_L',        'hock_R',        rpx2, y2 - bh*0.42, xl=0.37, xr=0.63)
        if rump_c:
            emit_pair('stifle_L',  'stifle_R',      rpx2, y2 - bh*0.54, xl=0.37, xr=0.63)

    # ── Udder (lower middle-rear of body) ─────────────────────────────────
    if body_c and rump_c:
        udder_px = (body_c[0] + rump_c[0]) / 2
        emit_mid('udder', udder_px, y2 - bh * 0.15)

    # ── Top-level cow entity ───────────────────────────────────────────────
    symbols.append(_sym('cow', 0.50, to_y(y1 + bh//2), to_z(cx_img * W),
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
