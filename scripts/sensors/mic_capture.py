#!/usr/bin/env python3
"""scripts/sensors/mic_capture.py — live microphone → node /sensor/observe.

Records short audio chunks from the default input device, packages each
chunk as a WAV blob, and POSTs to the node's /sensor/observe endpoint.
The node decodes via AudioBitsEncoder and trains the resulting labels
into the `audio_features` pool (online Hebbian update).

Optional --caption supplies paired_text on each chunk so the fabric
learns "this audio = this label" via cross-pool synapses with
`keyboard_text`.

Dependencies: sounddevice + numpy  (both pip-installable)

Usage:
  python mic_capture.py                              # 2s chunks at 16kHz
  python mic_capture.py --chunk-seconds 1 --rate 22050
  python mic_capture.py --caption "speech"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave


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
        body_txt = e.read().decode("utf-8", "replace") if hasattr(e, "read") else str(e)
        return {}, f"HTTP {e.code}: {body_txt[:200]}"
    except (urllib.error.URLError, OSError) as e:
        return {}, str(e)


def _pcm16_to_wav_bytes(samples: bytes, rate: int, channels: int = 1) -> bytes:
    """Wrap a raw PCM16 buffer in a WAV container so AudioBitsEncoder's
    encode_wav_bytes can read it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)            # 16-bit
        wf.setframerate(rate)
        wf.writeframes(samples)
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node",          default="http://127.0.0.1:8090")
    p.add_argument("--rate",          type=int, default=16000,
                     help="sample rate (default %(default)s)")
    p.add_argument("--chunk-seconds", type=float, default=2.0,
                     help="audio per POST (default %(default)s)")
    p.add_argument("--caption",       default="")
    p.add_argument("--max-chunks",    type=int, default=0,
                     help="stop after N chunks; 0 = forever")
    p.add_argument("--device",        default=None,
                     help="sounddevice input device name or index")
    p.add_argument("--verbose",       action="store_true")
    args = p.parse_args()

    try:
        import sounddevice as sd  # type: ignore
        import numpy as np         # type: ignore
    except ImportError:
        print("install sounddevice + numpy:  pip install sounddevice numpy",
                file=sys.stderr)
        return 2

    url = args.node.rstrip("/") + "/sensor/observe"
    chunk_samples = int(args.rate * args.chunk_seconds)
    chunks_sent = 0
    last_log = time.time()

    print(f"[mic] recording {args.chunk_seconds}s chunks at {args.rate}Hz → {url}",
            flush=True)
    if args.caption:
        print(f"[mic] paired caption: {args.caption!r}", flush=True)

    device = args.device
    try:
        if device is not None and device.isdigit():
            device = int(device)
    except AttributeError:
        pass

    try:
        while True:
            audio = sd.rec(chunk_samples, samplerate=args.rate, channels=1,
                              dtype="int16", device=device)
            sd.wait()
            wav_bytes = _pcm16_to_wav_bytes(audio.tobytes(), args.rate, 1)
            body = {
                "kind":      "audio",
                "bytes_b64": base64.b64encode(wav_bytes).decode("ascii"),
            }
            if args.caption:
                body["paired_text"] = args.caption
            resp, err = _post_json(url, body, timeout=15.0)
            chunks_sent += 1
            if err:
                print(f"[mic] POST failed: {err}", file=sys.stderr, flush=True)
            elif args.verbose:
                labels_n = resp.get("labels_count", 0)
                preds = resp.get("predictions", {}) or {}
                print(f"[mic] chunk {chunks_sent}  labels={labels_n}  "
                        f"pools_decoded={len(preds)}", flush=True)
            elif time.time() - last_log > 10:
                print(f"[mic] {chunks_sent} chunks posted "
                        f"(last labels={resp.get('labels_count', 0)})", flush=True)
                last_log = time.time()

            if args.max_chunks and chunks_sent >= args.max_chunks:
                break
    except KeyboardInterrupt:
        print("\n[mic] interrupted", flush=True)
    print(f"[mic] done; {chunks_sent} chunks posted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
