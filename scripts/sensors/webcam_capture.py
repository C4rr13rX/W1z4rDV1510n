#!/usr/bin/env python3
"""scripts/sensors/webcam_capture.py — live webcam → node /sensor/observe.

Captures frames from the configured camera, JPEG-encodes each frame, and
POSTs to the node's /sensor/observe endpoint at a configurable rate.
The node decodes the JPEG with ImageBitsEncoder and trains the resulting
labels into the `image_pixels` pool (online Hebbian update — no batch
required).

Optional --caption argument supplies a paired_text on each frame so the
fabric learns "this visual = this caption" via cross-pool synapses
between `keyboard_text` and `image_pixels`.  Use this when you want to
teach the node what it's looking at.

Default behaviour with no caption: pure observation.  The pool still
accumulates visual atoms, and any pre-existing cross-edges from other
training will fire when the visuals match.

Dependencies: opencv-python  (the only non-stdlib import)

Usage:
  python webcam_capture.py                       # 2 fps observation
  python webcam_capture.py --fps 5
  python webcam_capture.py --caption "office desk"
  python webcam_capture.py --camera 1 --node http://127.0.0.1:8090
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request


def _post_json(url: str, body: dict, timeout: float = 10.0) -> tuple[dict, str]:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=raw, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace")), ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else str(e)
        return {}, f"HTTP {e.code}: {body[:200]}"
    except (urllib.error.URLError, OSError) as e:
        return {}, str(e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node",    default="http://127.0.0.1:8090",
                     help="node base URL (default %(default)s)")
    p.add_argument("--camera",  type=int, default=0,
                     help="OpenCV camera index (default %(default)s)")
    p.add_argument("--fps",     type=float, default=2.0,
                     help="frames per second to POST (default %(default)s)")
    p.add_argument("--width",   type=int, default=320,
                     help="resize width before JPEG encoding (default %(default)s)")
    p.add_argument("--height",  type=int, default=240,
                     help="resize height (default %(default)s)")
    p.add_argument("--quality", type=int, default=70,
                     help="JPEG quality 1-100 (default %(default)s)")
    p.add_argument("--caption", default="",
                     help="paired text trained alongside each frame (optional)")
    p.add_argument("--max-frames", type=int, default=0,
                     help="stop after N frames; 0 = run forever")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        import cv2  # type: ignore
    except ImportError:
        print("opencv-python not installed.  `pip install opencv-python`",
                file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera index {args.camera}", file=sys.stderr)
        return 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    url = args.node.rstrip("/") + "/sensor/observe"
    interval = 1.0 / max(args.fps, 0.1)
    frames_sent = 0
    last_log = time.time()

    print(f"[webcam] capturing from camera {args.camera} at {args.fps} fps → {url}",
            flush=True)
    if args.caption:
        print(f"[webcam] paired caption: {args.caption!r}", flush=True)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(interval)
                continue
            # Resize defensively in case the camera ignored the SET above.
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            ok, jpg = cv2.imencode(".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
            if not ok:
                time.sleep(interval)
                continue
            body = {
                "kind":      "image",
                "bytes_b64": base64.b64encode(jpg.tobytes()).decode("ascii"),
            }
            if args.caption:
                body["paired_text"] = args.caption
            resp, err = _post_json(url, body, timeout=10.0)
            frames_sent += 1
            if err:
                print(f"[webcam] POST failed: {err}", file=sys.stderr, flush=True)
            elif args.verbose:
                labels_n = resp.get("labels_count", 0)
                preds = resp.get("predictions", {}) or {}
                print(f"[webcam] frame {frames_sent}  labels={labels_n}  "
                        f"pools_decoded={len(preds)}", flush=True)
            elif time.time() - last_log > 10:
                print(f"[webcam] {frames_sent} frames posted "
                        f"(last labels={resp.get('labels_count', 0)})", flush=True)
                last_log = time.time()

            if args.max_frames and frames_sent >= args.max_frames:
                break
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
    except KeyboardInterrupt:
        print("\n[webcam] interrupted", flush=True)
    finally:
        cap.release()
    print(f"[webcam] done; {frames_sent} frames posted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
