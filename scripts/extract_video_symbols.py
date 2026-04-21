#!/usr/bin/env python3
"""
Video Symbol Extractor
======================
Converts live video frames into EnvironmentSnapshot symbols and streams them
into the neural fabric at configurable FPS.

Each detected object becomes a symbol with:
  id          -- persistent track ID across frames (e.g. cow_3, person_7)
  position    -- normalised x, y (0-1 frame coords), z (monocular depth estimate)
  velocity    -- dx, dy per second in normalised frame coords
  properties  -- label, scale_m, depth_class, track_id, confidence

Detection backends (auto-selected by availability):
  yolo  -- YOLOv8n / YOLO11n real-time detection + ByteTrack
            pip install ultralytics
  flow  -- OpenCV MOG2 background subtraction + centroid tracking
            works with any OpenCV install, no ML dependencies

Usage:
    python scripts/extract_video_symbols.py
    python scripts/extract_video_symbols.py --source video.mp4
    python scripts/extract_video_symbols.py --source 0 --backend flow
    python scripts/extract_video_symbols.py --host 192.168.1.84 --port 8090 --fps 10
    python scripts/extract_video_symbols.py --source rtsp://cam/stream --no-preview

After training, GET /neuro/stream shows what the neural fabric has learned.
The world_viewer.html reads that stream and drives the 3D scene from it.
"""

import argparse, json, math, sys, time, collections
import urllib.request, urllib.error
import numpy as np
import cv2

# -- Optional: ultralytics for YOLO detection + tracking ----------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# -- Constants -----------------------------------------------------------------
FPS           = 10
BOUNDS        = {'x': 1.0, 'y': 1.0, 'z': 1.0}
MIN_AREA_FRAC = 0.0003   # ignore detections < 0.03% of frame area
MAX_SYMBOLS   = 60       # cap per frame

# COCO / YOLO class name -> neural depth layer
DEPTH_CLASS = {
    'person':    'd2_subject',  'rider':    'd2_subject',
    'cow':       'd2_subject',  'horse':    'd2_subject',
    'sheep':     'd2_subject',  'goat':     'd2_subject',
    'dog':       'd2_subject',  'cat':      'd2_subject',
    'bird':      'd1_midground','elephant':'d2_subject',
    'bear':      'd2_subject',  'zebra':    'd2_subject',
    'giraffe':   'd2_subject',  'deer':     'd2_subject',
    'car':       'd2_subject',  'truck':    'd2_subject',
    'bus':       'd2_subject',  'bicycle':  'd2_subject',
    'motorcycle':'d2_subject',  'airplane': 'd0_background',
    'boat':      'd1_midground',
    'tree':      'd1_midground','plant':    'd1_midground',
    'grass':     'd0_background','sky':     'd0_sky',
    'mountain':  'd0_background','building':'d0_background',
    'bench':     'd2_subject',  'chair':    'd2_subject',
    'bottle':    'd3_detail',   'cup':      'd3_detail',
    'book':      'd3_detail',   'laptop':   'd2_subject',
    # Medical / microscopy
    'cell':      'd2_subject',  'nucleus':  'd1_nuclear',
    'tissue':    'd1_midground','lesion':   'd2_subject',
    # Flow fallback
    'motion_region': 'd2_subject',
    'background':    'd0_background',
    'sky_region':    'd0_sky',
    'ground_region': 'd0_background',
}
DEFAULT_DEPTH = 'd2_subject'

# Approximate real-world scale (metres) for COCO classes
# Used as EEM calibration hint when no scene_width_m estimate is better
KNOWN_SCALE_M = {
    'person': 1.7,   'cow': 1.5,   'horse': 1.6,  'sheep': 0.7,
    'dog':    0.5,   'cat': 0.3,   'bird':  0.2,  'elephant': 2.5,
    'car':    1.5,   'truck': 3.0, 'bus':   3.0,  'bicycle': 0.7,
    'bottle': 0.25,  'laptop': 0.35,
}

# Per-class BGR colours for preview annotations
CLASS_COLORS = {
    'person': (80,220,80),   'cow': (60,200,255),  'horse': (200,140,60),
    'sheep':  (160,220,160), 'dog': (255,160,60),  'cat':   (255,100,180),
    'bird':   (200,255,120), 'car': (80,120,255),  'default': (0,210,80),
}


# -- Centroid tracker (used by FlowBackend) ------------------------------------
class CentroidTracker:
    """Greedy nearest-neighbour tracker. Assigns persistent integer IDs."""

    def __init__(self, max_missing: int = 20, max_dist: float = 0.20):
        self.next_id    = 0
        self.centroids  = {}   # id -> (cx, cy)
        self.missing    = {}   # id -> frames since last seen
        self.max_miss   = max_missing
        self.max_dist   = max_dist

    def update(self, rects):
        """rects: list of (cx, cy, w, h, label)  -- all normalised 0-1
        Returns: list of (id, cx, cy, w, h, label)"""

        # Expire old tracks
        for oid in list(self.missing):
            self.missing[oid] += 1
            if self.missing[oid] > self.max_miss:
                self.centroids.pop(oid, None)
                self.missing.pop(oid, None)

        if not rects:
            return []

        if not self.centroids:
            result = []
            for cx, cy, w, h, label in rects:
                oid = self.next_id
                self.centroids[oid] = (cx, cy)
                self.missing[oid]   = 0
                result.append((oid, cx, cy, w, h, label))
                self.next_id += 1
            return result

        old_ids   = list(self.centroids.keys())
        old_pts   = [self.centroids[i] for i in old_ids]
        used_old  = set()
        new_to_old = {}

        for ni, (ncx, ncy, *_) in enumerate(rects):
            best_d, best_oi = float('inf'), -1
            for oi, (ocx, ocy) in enumerate(old_pts):
                if oi in used_old:
                    continue
                d = math.hypot(ncx - ocx, ncy - ocy)
                if d < best_d:
                    best_d, best_oi = d, oi
            if best_oi >= 0 and best_d < self.max_dist:
                new_to_old[ni] = best_oi
                used_old.add(best_oi)

        result = []
        for ni, (cx, cy, w, h, label) in enumerate(rects):
            if ni in new_to_old:
                oid = old_ids[new_to_old[ni]]
                self.centroids[oid] = (cx, cy)
                self.missing[oid]   = 0
            else:
                oid = self.next_id
                self.centroids[oid] = (cx, cy)
                self.missing[oid]   = 0
                self.next_id += 1
            result.append((oid, cx, cy, w, h, label))

        for oi, oid in enumerate(old_ids):
            if oi not in used_old:
                self.missing.setdefault(oid, 0)

        return result


# -- YOLO backend --------------------------------------------------------------
class YOLOBackend:
    """YOLOv8/11 with ByteTrack for persistent IDs across frames."""

    def __init__(self, model_name: str = 'yolo11n.pt'):
        print(f'  Loading {model_name} (downloads ~6 MB on first use)...', flush=True)
        self.model = YOLO(model_name)
        self.names = self.model.names   # {int: str}
        print(f'  YOLO ready -- {len(self.names)} classes', flush=True)

    def detect(self, frame) -> list:
        """Returns list of dicts: id, label, cx, cy, w, h, conf  (0-1 normalised)."""
        H, W = frame.shape[:2]
        frame_area = H * W

        results = self.model.track(
            frame, persist=True, tracker='bytetrack.yaml',
            verbose=False, conf=0.28, iou=0.45, stream=False,
        )
        dets = []
        if not results or results[0].boxes is None:
            return dets

        boxes = results[0].boxes
        ids   = boxes.id   # None if tracking not yet established

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area / frame_area < MIN_AREA_FRAC:
                continue
            cx = float((x1 + x2) / 2 / W)
            cy = float((y1 + y2) / 2 / H)
            w  = float((x2 - x1) / W)
            h  = float((y2 - y1) / H)
            cls_i = int(boxes.cls[i].item())
            label = self.names.get(cls_i, 'unknown')
            conf  = float(boxes.conf[i].item())
            tid   = int(ids[i].item()) if ids is not None else i
            dets.append({'id': tid, 'label': label,
                         'cx': cx, 'cy': cy, 'w': w, 'h': h, 'conf': conf})
        return dets


# -- Flow backend --------------------------------------------------------------
class FlowBackend:
    """
    No-ML fallback: MOG2 background subtraction finds moving regions.
    Lucas-Kanade sparse optical flow estimates per-blob velocities.
    CentroidTracker assigns persistent IDs.
    Labels are generic ('motion_region') unless --label is set.
    """

    LK_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, default_label: str = 'motion_region'):
        self.bg      = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)
        self.tracker = CentroidTracker(max_missing=20)
        self.prev_gray   = None
        self.prev_points = {}   # blob_id -> keypoints (for LK flow)
        self.default_lbl = default_label
        self._kern_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        print('  Flow backend ready (MOG2 + LK optical flow + centroid tracking)', flush=True)

    def detect(self, frame) -> list:
        H, W = frame.shape[:2]
        frame_area = H * W
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Foreground mask
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kern_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kern_close, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area / frame_area < MIN_AREA_FRAC:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            wn = bw / W
            hn = bh / H
            rects.append((cx, cy, wn, hn, self.default_lbl))

        # Sort largest first, cap
        rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        rects = rects[:MAX_SYMBOLS // 2]

        tracked = self.tracker.update(rects)
        self.prev_gray = gray

        return [{'id': oid, 'label': lbl, 'cx': cx, 'cy': cy,
                 'w': w, 'h': h, 'conf': 0.85}
                for oid, cx, cy, w, h, lbl in tracked]


# -- Depth estimation (monocular) ----------------------------------------------
def estimate_depth(cx: float, cy: float, w: float, h: float) -> float:
    """
    Combine size cue (larger bbox -> closer) and vertical cue (lower in frame -> closer).
    Returns z in [0.02, 0.95].
    """
    area = w * h
    # Size cue: normalised diagonal relative to whole frame
    diag  = math.sqrt(area)
    z_sz  = 1.0 - min(diag * 2.8, 0.92)
    # Vertical cue: cy=1.0 (bottom) means close, cy=0 (top) means far
    z_vy  = 0.25 * (1.0 - cy)
    z = z_sz * 0.75 + z_vy
    return round(max(0.02, min(0.95, z)), 4)


# -- Scene anchors (static background context symbols) -------------------------
def scene_anchors(context: str, scene_width_m: float) -> list:
    """
    Always-present symbols for sky, ground, and midground so the neural fabric
    builds a stable spatial scaffold even during static or slow scenes.
    """
    anchors = [
        # Sky strip -- top 30 % of frame
        {'id': 'sky_0',    'label': 'sky_region',    'cx': 0.50, 'cy': 0.15, 'w': 1.0, 'h': 0.30, 'conf': 1.0},
        # Ground strip -- bottom 20 %
        {'id': 'ground_0', 'label': 'ground_region', 'cx': 0.50, 'cy': 0.90, 'w': 1.0, 'h': 0.20, 'conf': 1.0},
        # Midground band
        {'id': 'mid_0',    'label': 'background',    'cx': 0.50, 'cy': 0.55, 'w': 1.0, 'h': 0.40, 'conf': 1.0},
    ]
    if context == 'medical_imaging':
        anchors = [
            {'id': 'tissue_bg', 'label': 'tissue',    'cx': 0.50, 'cy': 0.50, 'w': 1.0, 'h': 1.0, 'conf': 1.0},
        ]
    return anchors


# -- EnvironmentSnapshot builder -----------------------------------------------
def build_snapshot(detections: list, prev_pos: dict, frame_dt: float,
                   t: float, context: str, scene_width_m: float) -> dict:
    symbols = []
    all_dets = detections   # anchors already included upstream

    for d in all_dets[:MAX_SYMBOLS]:
        sym_id = d.get('id')
        label  = d['label']
        cx, cy = d['cx'], d['cy']
        w,  h  = d['w'],  d['h']
        z      = estimate_depth(cx, cy, w, h)

        # Velocity: change in normalised position per second
        if sym_id in prev_pos:
            pcx, pcy = prev_pos[sym_id]
            vx = round((cx - pcx) / max(frame_dt, 1e-4), 5)
            vy = round((cy - pcy) / max(frame_dt, 1e-4), 5)
        else:
            vx = vy = 0.0

        # Scale: prefer known real-world value, fall back to bbox x scene width
        known = KNOWN_SCALE_M.get(label)
        scale_m = round(known if known else max(0.01, w * scene_width_m), 4)

        track_id = f'{label[:5]}_{sym_id}'

        symbols.append({
            'id':   track_id,
            'type': 'CUSTOM',
            'position': {'x': round(cx, 5), 'y': round(cy, 5), 'z': z},
            'velocity': {'x': vx, 'y': vy, 'z': 0.0},
            'properties': {
                'label':       label,
                'scale_m':     str(scale_m),
                'diameter_m':  str(scale_m),
                'depth_class': DEPTH_CLASS.get(label, DEFAULT_DEPTH),
                'track_id':    track_id,
                'confidence':  str(round(d['conf'], 3)),
            }
        })

    return {
        'timestamp': int(t * 1000),
        'bounds':    BOUNDS,
        'symbols':   symbols,
        'metadata':  {
            'context':       context,
            'modality':      'rgb_video',
            'frame_t':       str(round(t, 3)),
            'scene_width_m': str(scene_width_m),
            'symbol_count':  str(len(symbols)),
        }
    }


# -- Preview renderer ----------------------------------------------------------
def draw_preview(frame, detections: list, host: str, port: int,
                 fps: int, frame_no: int, last_labels):
    H, W = frame.shape[:2]
    ann = frame.copy()

    for d in detections:
        if d['label'] in ('sky_region', 'ground_region', 'background'):
            continue   # skip anchor symbols in preview
        x1 = int((d['cx'] - d['w'] / 2) * W)
        y1 = int((d['cy'] - d['h'] / 2) * H)
        x2 = int((d['cx'] + d['w'] / 2) * W)
        y2 = int((d['cy'] + d['h'] / 2) * H)
        col = CLASS_COLORS.get(d['label'], CLASS_COLORS['default'])
        cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
        tag = f"{d['label']} #{d['id']}  {d['conf']:.2f}"
        cv2.putText(ann, tag, (max(x1, 2), max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        # Depth bar on right edge of bbox
        z = estimate_depth(d['cx'], d['cy'], d['w'], d['h'])
        bar_h = int((1 - z) * (y2 - y1))
        cv2.rectangle(ann, (x2 + 2, y2 - bar_h), (x2 + 6, y2), col, -1)

    # Status bar
    n_obj = sum(1 for d in detections if d['label'] not in ('sky_region', 'ground_region', 'background'))
    status = f'  {fps} fps  |  {n_obj} objects  |  -> {host}:{port}  |  frame {frame_no}  |  labels={last_labels}'
    cv2.rectangle(ann, (0, H - 24), (W, H), (8, 14, 22), -1)
    cv2.putText(ann, status, (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 200, 200), 1, cv2.LINE_AA)

    return ann


# -- HTTP helpers --------------------------------------------------------------
def post(host: str, port: int, path: str, body: dict) -> dict:
    url  = f'http://{host}:{port}{path}'
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f'  HTTP {e.code} {path}: {e.read().decode()[:60]}', flush=True)
        return {}
    except Exception as e:
        print(f'  ERR {path}: {e}', flush=True)
        return {}


def check_connection(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/health', timeout=4):
            return True
    except Exception:
        return False


# -- Main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Stream live video frames as EnvironmentSnapshot symbols into the neural fabric.')
    ap.add_argument('--source',        default='0',
                    help='Video source: 0=webcam, path/to/video.mp4, rtsp://...')
    ap.add_argument('--host',          default='192.168.1.84')
    ap.add_argument('--port',          type=int, default=8090)
    ap.add_argument('--fps',           type=int, default=FPS,
                    help='Snapshot rate sent to node (default 10)')
    ap.add_argument('--backend',       default='auto', choices=['auto', 'yolo', 'flow'],
                    help='Detection backend (default: auto-select)')
    ap.add_argument('--model',         default='yolo11n.pt',
                    help='YOLO model file (yolo11n.pt, yolov8n.pt, etc.)')
    ap.add_argument('--context',       default='scene',
                    help='Scene context label fed to neural fabric (e.g. field_scene, medical_imaging)')
    ap.add_argument('--scene-width-m', type=float, default=10.0,
                    help='Estimated physical width of the scene in metres (used for scale hints)')
    ap.add_argument('--no-anchors',    action='store_true',
                    help='Disable static background scene anchors')
    ap.add_argument('--no-preview',    action='store_true',
                    help='Disable OpenCV preview window')
    ap.add_argument('--label',         default=None,
                    help='Force a single label for all flow-backend detections')
    args = ap.parse_args()

    # -- Open video source ----------------------------------------------------
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'ERROR: cannot open video source: {args.source}', flush=True)
        sys.exit(1)
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f'Video source: {args.source}  (native {cap_fps:.1f} fps)', flush=True)

    # -- Select backend -------------------------------------------------------
    backend_name = args.backend
    if backend_name == 'auto':
        backend_name = 'yolo' if YOLO_AVAILABLE else 'flow'
        print(f'Backend auto-selected: {backend_name}', flush=True)

    if backend_name == 'yolo':
        if not YOLO_AVAILABLE:
            print('ultralytics not found -- falling back to flow backend.', flush=True)
            print('  To enable YOLO:  pip install ultralytics', flush=True)
            backend_name = 'flow'
        else:
            detector = YOLOBackend(args.model)

    if backend_name == 'flow':
        lbl = args.label or 'motion_region'
        detector = FlowBackend(default_label=lbl)

    # -- Node connection ------------------------------------------------------
    print(f'Connecting to {args.host}:{args.port} ...', flush=True)
    if not check_connection(args.host, args.port):
        print('Node not reachable -- start the node first.', flush=True)
        sys.exit(1)
    print('Connected. Starting video symbol extraction.', flush=True)
    print(f'  context={args.context}  scene_width={args.scene_width_m}m  '
          f'target_fps={args.fps}  backend={backend_name}', flush=True)
    print('Press Q in the preview window (or Ctrl+C) to stop.', flush=True)

    # -- Rate-control: skip video frames to hit target fps -------------------
    interval   = 1.0 / args.fps
    skip_every = max(1, round(cap_fps / args.fps))

    prev_pos   = {}   # track_id string -> (cx, cy)
    last_labels = '?'
    frame_count = 0
    send_count  = 0
    t = 0.0
    drift = 0.0
    anchors = scene_anchors(args.context, args.scene_width_m)

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print('Stream ended or read error.', flush=True)
                break

            # Skip frames to match target fps
            frame_count += 1
            if frame_count % skip_every != 0:
                continue

            # -- Detect ------------------------------------------------------
            dets = detector.detect(frame)

            # Merge with static scene anchors
            all_dets = dets + ([] if args.no_anchors else anchors)

            # -- Build and send snapshot --------------------------------------
            snap   = build_snapshot(all_dets, prev_pos, interval, t,
                                    args.context, args.scene_width_m)
            result = post(args.host, args.port, '/neuro/train', {'snapshot': snap})

            # Update previous positions for velocity computation
            for sym in snap['symbols']:
                p = sym['position']
                prev_pos[sym['id']] = (p['x'], p['y'])

            send_count  += 1
            last_labels  = result.get('label_count', last_labels)
            t           += interval

            # -- Progress log every 5 s ---------------------------------------
            if send_count % (args.fps * 5) == 0:
                n_real = sum(1 for d in dets
                             if d['label'] not in ('sky_region', 'ground_region', 'background'))
                print(f'  frame {send_count:06d}  t={t:.1f}s  '
                      f'objects={n_real}  labels={last_labels}', flush=True)

            # -- Preview ------------------------------------------------------
            if not args.no_preview:
                ann = draw_preview(frame, all_dets, args.host, args.port,
                                   args.fps, send_count, last_labels)
                cv2.imshow('W1z4rD -- Video Symbol Extractor', ann)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('User quit.', flush=True)
                    break

            # -- Rate control -------------------------------------------------
            elapsed = time.perf_counter() - t0
            wait    = interval - elapsed - drift
            if wait > 0:
                time.sleep(wait)
                drift = 0.0
            else:
                drift = -wait * 0.1

    except KeyboardInterrupt:
        print('\nInterrupted.', flush=True)

    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()
    print(f'Stopped after {send_count} snapshots.', flush=True)


if __name__ == '__main__':
    main()
