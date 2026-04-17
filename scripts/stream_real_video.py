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
DEFAULT_VIDEO  = 'D:/w1z4rdv1510n-data/videos/cow_yt.mp4'
DEFAULT_NEURO  = 'localhost:8090'
FRAME_PORT     = 9001
TARGET_W       = 640
TARGET_H       = 360
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


# ── Object detection (colour-blob heuristics -> EnvironmentSnapshot symbols) -
def detect_objects(frame_bgr: np.ndarray, W: int, H: int) -> list:
    """
    Very lightweight detection: colour segmentation to find cow, sky, ground.
    Returns a list of symbol dicts compatible with EnvironmentSnapshot.
    """
    f      = frame_bgr.astype(np.float32) / 255.0
    R, G, B = f[:, :, 2], f[:, :, 1], f[:, :, 0]
    symbols = []

    # Cow: brownish-orange (R high, G mid, B low)
    cow_mask = ((R - B) > 0.18) & (R > 0.28) & (G > 0.15) & (G < 0.75)
    ys, xs   = np.where(cow_mask)
    if len(xs) > 300:
        cx   = float(np.median(xs)) / W
        cy   = float(np.median(ys)) / H
        area = len(xs) / (W * H)
        symbols.append({
            'id': 'cow_0', 'type': 'CUSTOM',
            'position': {'x': cx, 'y': 1.0 - cy, 'z': max(0.05, 0.6 - area * 3)},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'properties': {
                'label': 'cow', 'scene': 'field',
                'track_id': 'cow_0',
                'scale_m': 1.5, 'diameter_m': 1.5,
            },
        })

    # Sky: blue-dominant in upper third
    sky_mask = (B > R * 1.15) & (B > G * 1.05)
    sy, sx   = np.where(sky_mask[:H // 3, :])
    if len(sx) > 200:
        symbols.append({
            'id': 'sky_0', 'type': 'CUSTOM',
            'position': {'x': 0.5, 'y': 0.9, 'z': 0.95},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'properties': {
                'label': 'sky', 'scene': 'field',
                'track_id': 'sky_0',
                'scale_m': 100.0, 'diameter_m': 100.0,
            },
        })

    # Ground: green/brown in lower half
    grnd_mask = ((G - R) > -0.08) & ((G - B) > 0.04) & (G > 0.18)
    gy, gx    = np.where(grnd_mask[H // 2:, :])
    if len(gx) > 300:
        symbols.append({
            'id': 'ground_0', 'type': 'CUSTOM',
            'position': {'x': 0.5, 'y': 0.1, 'z': 0.5},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'properties': {
                'label': 'ground', 'scene': 'field',
                'track_id': 'ground_0',
                'scale_m': 50.0, 'diameter_m': 50.0,
            },
        })

    return symbols


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
            syms     = detect_objects(frame, target_w, target_h)
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
