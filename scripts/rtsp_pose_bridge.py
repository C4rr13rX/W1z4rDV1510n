import argparse
import json
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
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        now = time.time()
        if now - last_emit < 1.0 / max(args.fps, 1.0):
            continue
        last_emit = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            continue

        keypoints = []
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            if landmark.visibility < args.confidence:
                continue
            keypoints.append({
                "name": f"kp_{idx}",
                "x": float(landmark.x),
                "y": float(landmark.y),
                "confidence": float(landmark.visibility),
            })

        payload = {
            "entity_id": args.entity_id,
            "timestamp": {"unix": int(now)},
            "keypoints": keypoints,
            "metadata": {
                "source": args.source,
                "fps": args.fps,
            },
        }
        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
