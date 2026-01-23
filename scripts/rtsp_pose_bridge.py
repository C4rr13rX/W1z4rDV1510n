import argparse
import base64
import json
import os
import time

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required: pip install opencv-python") from exc

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit("mediapipe is required: pip install mediapipe") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="RTSP -> pose keypoints JSONL bridge")
    parser.add_argument("--source", required=True, help="RTSP URL or video file path")
    parser.add_argument("--fps", type=float, default=8.0, help="Target sampling FPS")
    parser.add_argument("--entity-id", default="person-0", help="Entity id label")
    parser.add_argument("--confidence", type=float, default=0.3, help="Min keypoint confidence")
    parser.add_argument("--max-failures", type=int, default=30, help="Max read failures before reconnect")
    parser.add_argument("--reconnect-delay", type=float, default=5.0, help="Seconds to wait before reconnect")
    parser.add_argument(
        "--emit-image",
        choices=["none", "ref", "b64"],
        default="none",
        help="Attach frame bytes (b64) or write image_ref",
    )
    parser.add_argument("--image-dir", default="data/frames", help="Directory for image_ref mode")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--image-quality", type=int, default=80, help="JPEG quality 1-100")
    parser.add_argument("--max-width", type=int, default=0, help="Optional resize width")
    parser.add_argument("--max-height", type=int, default=0, help="Optional resize height")
    parser.add_argument("--quality-floor", type=float, default=0.05, help="Minimum quality floor")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"failed to open source: {args.source}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_emit = 0.0
    last_frame_time = None
    frame_idx = 0
    failures = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures >= args.max_failures:
                cap.release()
                time.sleep(max(args.reconnect_delay, 0.5))
                cap = cv2.VideoCapture(args.source)
                failures = 0
            else:
                time.sleep(0.1)
            continue
        failures = 0
        now = time.time()
        if now - last_emit < 1.0 / max(args.fps, 1.0):
            continue
        frame_dt = None if last_frame_time is None else max(0.0, now - last_frame_time)
        last_frame_time = now
        last_emit = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            continue

        keypoints = []
        min_x = None
        min_y = None
        max_x = None
        max_y = None
        confidence_sum = 0.0
        confidence_count = 0
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            if landmark.visibility < args.confidence:
                continue
            x = float(landmark.x)
            y = float(landmark.y)
            confidence_sum += float(landmark.visibility)
            confidence_count += 1
            keypoints.append({
                "name": f"kp_{idx}",
                "x": x,
                "y": y,
                "confidence": float(landmark.visibility),
            })
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = x if max_x is None else max(max_x, x)
            max_y = y if max_y is None else max(max_y, y)

        frame_id = f"frame-{int(now * 1000)}-{frame_idx}"
        frame_idx += 1
        expected_dt = 1.0 / max(args.fps, 1.0)
        jitter_ratio = 0.0
        drop_ratio = 0.0
        frame_dt_ms = None
        if frame_dt is not None:
            frame_dt_ms = frame_dt * 1000.0
            jitter_ratio = abs(frame_dt - expected_dt) / expected_dt if expected_dt > 0 else 0.0
            drop_ratio = max(0.0, frame_dt / expected_dt - 1.0) if expected_dt > 0 else 0.0
        keypoint_conf_avg = confidence_sum / confidence_count if confidence_count else 0.0
        quality = keypoint_conf_avg * (1.0 - min(jitter_ratio, 1.0))
        quality = min(1.0, max(args.quality_floor, quality))
        metadata = {
            "source": args.source,
            "fps": args.fps,
            "frame_id": frame_id,
            "keypoint_confidence_avg": keypoint_conf_avg,
            "frame_dt_ms": frame_dt_ms,
            "jitter_ratio": jitter_ratio,
            "drop_ratio": drop_ratio,
            "quality": quality,
            "confidence": quality,
        }

        frame_for_export = frame
        if args.max_width or args.max_height:
            h, w = frame.shape[:2]
            scale_w = args.max_width / w if args.max_width else 1.0
            scale_h = args.max_height / h if args.max_height else 1.0
            scale = min(scale_w, scale_h, 1.0)
            if scale < 1.0:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                frame_for_export = cv2.resize(frame, (new_w, new_h))

        if args.emit_image != "none":
            ext = args.image_format
            if args.emit_image == "ref":
                os.makedirs(args.image_dir, exist_ok=True)
                filename = f"{frame_id}.{ext}"
                image_path = os.path.join(args.image_dir, filename)
                if ext == "jpg":
                    cv2.imwrite(
                        image_path,
                        frame_for_export,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.image_quality],
                    )
                else:
                    cv2.imwrite(image_path, frame_for_export)
                metadata["image_ref"] = image_path
            else:
                encode_params = []
                if ext == "jpg":
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.image_quality]
                ok, buf = cv2.imencode(f".{ext}", frame_for_export, encode_params)
                if ok:
                    metadata["image_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
                    metadata["image_format"] = ext

        payload = {
            "entity_id": args.entity_id,
            "timestamp": {"unix": int(now)},
            "keypoints": keypoints,
            "metadata": metadata,
        }
        if min_x is not None and min_y is not None:
            max_x_val = max_x if max_x is not None else min_x
            max_y_val = max_y if max_y is not None else min_y
            payload["bbox"] = {
                "x": min_x,
                "y": min_y,
                "width": max(0.0, max_x_val - min_x),
                "height": max(0.0, max_y_val - min_y),
            }
        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
