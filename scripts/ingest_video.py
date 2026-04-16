#!/usr/bin/env python3
"""
W1z4rD V1510n — Video/Audio Ingestion Script
=============================================
Processes video files into multimodal training sequences for the node.

For each time chunk (default 1s) the script extracts:
  - A JPEG frame (image labels: zones, hues, edges)
  - A mono WAV audio clip (audio labels: bands, energy, beats)
  - Subtitle/caption text if available (text labels: words, phonemes, roles)

All three modalities from the same timestamp are bundled as adjacent frames
in a /media/train_sequence call, bridged at tau=0 (max strength). Adjacent
chunks are bridged with temporal decay at the configured tau (default 1.5s),
so concepts that co-occur across a few seconds get linked.

Dependencies:
    pip install httpx tqdm
    ffmpeg must be in PATH  (https://ffmpeg.org/download.html)

Usage:
    python ingest_video.py --video lecture.mp4
    python ingest_video.py --video lecture.mp4 --srt lecture.srt
    python ingest_video.py --video lecture.mp4 --subs      # auto-extract embedded subs
    python ingest_video.py --video lecture.mp4 --start 60 --end 600 --chunk 2.0
    python ingest_video.py --video lecture.mp4 --batch-secs 30 --tau 1.5
"""

import argparse
import base64
import io
import json
import re
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get("total", "?")
        for i, x in enumerate(it):
            print(f"  [{i+1}/{total}]", end="\r", flush=True)
            yield x
        print()


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found in PATH. Install from https://ffmpeg.org/download.html")


def video_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    info = json.loads(result.stdout)
    return float(info["format"].get("duration", 0))


def extract_frame_jpeg(video_path: str, t: float, quality: int = 3) -> bytes | None:
    """Extract one JPEG frame at time t. Returns None on failure."""
    result = subprocess.run(
        ["ffmpeg", "-ss", str(t), "-i", video_path,
         "-vframes", "1", "-q:v", str(quality),
         "-f", "image2", "pipe:1"],
        capture_output=True
    )
    if result.returncode != 0 or len(result.stdout) < 100:
        return None
    return result.stdout


def extract_audio_wav(video_path: str, t: float, duration: float,
                      sample_rate: int = 16000) -> bytes | None:
    """
    Extract [t, t+duration] of audio as 16-bit mono WAV.
    Returns None if no audio stream or ffmpeg fails.
    """
    result = subprocess.run(
        ["ffmpeg", "-ss", str(t), "-t", str(duration), "-i", video_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
         "-f", "wav", "pipe:1"],
        capture_output=True
    )
    if result.returncode != 0 or len(result.stdout) < 44:  # 44-byte WAV header minimum
        return None
    return result.stdout


def extract_embedded_srt(video_path: str) -> str | None:
    """
    Try to extract the first subtitle stream from the video as SRT text.
    Returns the SRT content string, or None if no subtitle stream found.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-map", "0:s:0",
         "-f", "srt", "pipe:1"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


# ── SRT parser ─────────────────────────────────────────────────────────────────

def _srt_time_to_secs(ts: str) -> float:
    """Convert '00:01:23,456' to seconds."""
    ts = ts.strip().replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(srt_text: str) -> list[tuple[float, float, str]]:
    """
    Parse SRT content into (start_secs, end_secs, text) triples.
    """
    entries = []
    blocks = re.split(r"\n{2,}", srt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # lines[0] = index, lines[1] = timestamps, lines[2+] = text
        time_line = lines[1]
        m = re.match(r"([\d:,]+)\s*-->\s*([\d:,]+)", time_line)
        if not m:
            continue
        start = _srt_time_to_secs(m.group(1))
        end   = _srt_time_to_secs(m.group(2))
        text  = " ".join(lines[2:]).strip()
        if text:
            entries.append((start, end, text))
    return entries


def subtitle_at(entries: list[tuple[float, float, str]], t: float) -> str | None:
    """Return subtitle text active at time t, or None."""
    for start, end, text in entries:
        if start <= t <= end:
            return text
    return None


# ── Node API ───────────────────────────────────────────────────────────────────

def train_sequence(client: httpx.Client, node_url: str,
                   frames: list[dict], tau: float) -> dict:
    payload = {"frames": frames, "temporal_tau": tau}
    resp = client.post(f"{node_url}/media/train_sequence", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def checkpoint(client: httpx.Client, node_url: str) -> None:
    resp = client.post(f"{node_url}/neuro/checkpoint", timeout=10)
    if resp.is_success:
        data = resp.json()
        tqdm.write(f"  Checkpointed -> {data.get('path', '?')}")
    else:
        tqdm.write(f"  Checkpoint failed: {resp.status_code}")


# ── Main ───────────────────────────────────────────────────────────────────────

def build_frames_for_chunk(
    video_path: str,
    t: float,
    audio_duration: float,
    subtitle_text: str | None,
    lr_scale: float,
) -> list[dict]:
    """
    Build the list of TrainSequenceFrame dicts for one time chunk.
    Each modality present becomes its own frame with the same t_secs.
    Adjacent frames at the same t get max bridge strength (dt=0 → exp(0)=1).
    """
    frames = []

    # Image frame
    jpeg = extract_frame_jpeg(video_path, t)
    if jpeg and len(jpeg) > 500:
        frames.append({
            "t_secs":   t,
            "modality": "image",
            "data_b64": base64.b64encode(jpeg).decode(),
            "lr_scale": lr_scale,
        })

    # Audio frame
    wav = extract_audio_wav(video_path, t, audio_duration)
    if wav and len(wav) > 200:
        frames.append({
            "t_secs":   t,
            "modality": "audio",
            "data_b64": base64.b64encode(wav).decode(),
            "lr_scale": lr_scale,
        })

    # Text frame (subtitle)
    if subtitle_text:
        frames.append({
            "t_secs":   t,
            "modality": "text",
            "text":     subtitle_text,
            "lr_scale": lr_scale,
        })

    return frames


def main():
    require_ffmpeg()

    parser = argparse.ArgumentParser(description="Ingest video/audio into the W1z4rD node")
    parser.add_argument("--video",    required=True, help="Input video file path")
    parser.add_argument("--srt",      default=None,  help="SRT subtitle file (optional)")
    parser.add_argument("--subs",     action="store_true",
                        help="Auto-extract embedded subtitle stream from video")
    parser.add_argument("--node",     default="http://localhost:8090", help="Node API base URL")
    parser.add_argument("--start",    type=float, default=0.0,  help="Start time in seconds")
    parser.add_argument("--end",      type=float, default=None, help="End time in seconds (default: full video)")
    parser.add_argument("--chunk",    type=float, default=1.0,  dest="chunk_secs",
                        help="Seconds between extracted frames (default 1.0)")
    parser.add_argument("--audio-duration", type=float, default=1.0, dest="audio_dur",
                        help="Duration of audio clip per chunk in seconds (default 1.0)")
    parser.add_argument("--batch-secs",  type=float, default=10.0, dest="batch_secs",
                        help="Seconds of video per /media/train_sequence POST (default 10.0)")
    parser.add_argument("--tau",      type=float, default=1.5,
                        help="Temporal bridge decay tau in seconds (default 1.5)")
    parser.add_argument("--lr",       type=float, default=1.0, help="Learning rate scale")
    parser.add_argument("--checkpoint-every", type=int, default=60, dest="ckpt_every",
                        help="Checkpoint pool every N chunks (default 60)")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        sys.exit(f"File not found: {video_path}")

    # -- Duration
    try:
        total_dur = video_duration(video_path)
    except Exception as e:
        sys.exit(f"Could not read video duration: {e}")

    start = args.start
    end   = min(args.end, total_dur) if args.end else total_dur
    if end <= start:
        sys.exit(f"Invalid range: {start}s to {end}s")
    print(f"Video: {video_path} | range {start:.1f}s–{end:.1f}s | total {total_dur:.1f}s")

    # -- Subtitles
    subtitles: list[tuple[float, float, str]] = []
    if args.srt:
        srt_path = Path(args.srt)
        if not srt_path.exists():
            sys.exit(f"SRT file not found: {srt_path}")
        subtitles = parse_srt(srt_path.read_text(encoding="utf-8", errors="replace"))
        print(f"Loaded {len(subtitles)} subtitle entries from {srt_path.name}")
    elif args.subs:
        raw = extract_embedded_srt(video_path)
        if raw:
            subtitles = parse_srt(raw)
            print(f"Extracted {len(subtitles)} subtitle entries from embedded stream")
        else:
            print("No embedded subtitle stream found, continuing without subtitles")

    # -- Check node health
    with httpx.Client() as hc:
        try:
            health = hc.get(f"{args.node}/health", timeout=5).json()
            print(f"Node: {health.get('node_id')} — {health.get('status')}")
        except Exception as e:
            sys.exit(f"Node not reachable at {args.node}: {e}")

    # -- Build chunk timestamps
    chunk_times = []
    t = start
    while t < end:
        chunk_times.append(t)
        t += args.chunk_secs

    batch_size = max(1, int(args.batch_secs / args.chunk_secs))
    total_chunks = len(chunk_times)
    print(f"Chunks: {total_chunks} @ {args.chunk_secs}s each | batch_size={batch_size} | tau={args.tau}s")

    # -- Ingest
    trained_chunks = 0
    skipped_chunks = 0
    total_frames   = 0
    start_time = time.time()

    with httpx.Client() as client:
        bar = tqdm(total=total_chunks, unit="chunk", desc="Ingesting")
        for batch_start in range(0, total_chunks, batch_size):
            batch_chunk_times = chunk_times[batch_start:batch_start + batch_size]
            all_frames: list[dict] = []

            for t in batch_chunk_times:
                sub_text = subtitle_at(subtitles, t) if subtitles else None
                frames = build_frames_for_chunk(
                    video_path, t, args.audio_dur, sub_text, args.lr
                )
                all_frames.extend(frames)

            if not all_frames:
                skipped_chunks += len(batch_chunk_times)
                bar.update(len(batch_chunk_times))
                continue

            try:
                result = train_sequence(client, args.node, all_frames, args.tau)
                trained_chunks += result.get("trained_frames", 0)
                total_frames   += len(all_frames)
                bar.set_postfix(
                    trained=result.get("trained_frames", 0),
                    labels=result.get("total_labels", 0),
                )
                if result.get("warnings"):
                    for w in result["warnings"]:
                        tqdm.write(f"  warn: {w}")
            except httpx.HTTPStatusError as e:
                tqdm.write(f"  batch@{batch_chunk_times[0]:.1f}s error: {e.response.status_code}")
                skipped_chunks += len(batch_chunk_times)
            except Exception as e:
                tqdm.write(f"  batch@{batch_chunk_times[0]:.1f}s error: {e}")
                skipped_chunks += len(batch_chunk_times)

            bar.update(len(batch_chunk_times))

            if trained_chunks > 0 and trained_chunks % args.ckpt_every == 0:
                tqdm.write(f"  [{trained_chunks} frames trained] checkpointing...")
                checkpoint(client, args.node)

        bar.close()

        print(f"\nFinal checkpoint...")
        checkpoint(client, args.node)

    elapsed = time.time() - start_time
    rate = total_chunks / elapsed if elapsed > 0 else 0
    print(f"\nDone.  chunks={total_chunks}  frames_sent={total_frames}  "
          f"skipped={skipped_chunks}  elapsed={elapsed:.1f}s  ({rate:.1f} chunks/sec)")


if __name__ == "__main__":
    main()
