#!/usr/bin/env python3
"""scripts/sensors/toddler_multimodal.py — toddler-level multimodal trainer.

Trains the node on (image, audio, text) triples — the cross-modal floor
the system should clear: given any one of the three inputs, recall the
other two.

Approach:
  - 6 simple objects (apple, banana, ball, leaf, cup, book)
  - For each object: a synthetic image with distinctive colour+shape, an
    AWS Polly TTS audio clip of "this is a <object>", and the text label
  - Train each triple N times via /sensor/observe (kind=image+audio+text
    in one call) so cross-pool synapses form across all three pools:
      keyboard_text ↔ image_pixels ↔ audio_features
  - Probe each direction: image->{text,audio}, text->{image,audio},
    audio->{image,text}

This is the verification floor.  If it doesn't fire, the architecture
isn't doing what it claims.

Dependencies: boto3 (for Polly), pillow, requests
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


# ── Objects: name, colour (RGB), shape ────────────────────────────────────


OBJECTS = [
    ("apple",  (220, 30, 30),   "circle"),
    ("banana", (240, 215, 65),  "ellipse_h"),
    ("ball",   (40, 90, 220),   "circle"),
    ("leaf",   (50, 160, 60),   "ellipse_v"),
    ("cup",    (210, 210, 210), "trapezoid"),
    ("book",   (140, 90, 50),   "rectangle"),
]


# ── Image synthesis ───────────────────────────────────────────────────────


def make_image(colour: tuple[int, int, int], shape: str,
                size: int = 96) -> bytes:
    """Render a simple solid-colour shape on a neutral background.
    Returns JPEG bytes.  Deterministic — same inputs give same image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (240, 240, 240))
    d = ImageDraw.Draw(img)
    pad = size // 6
    box = (pad, pad, size - pad, size - pad)
    if shape == "circle":
        d.ellipse(box, fill=colour, outline=(0, 0, 0))
    elif shape == "ellipse_h":
        d.ellipse((pad, size // 3, size - pad, 2 * size // 3),
                    fill=colour, outline=(0, 0, 0))
    elif shape == "ellipse_v":
        d.ellipse((size // 3, pad, 2 * size // 3, size - pad),
                    fill=colour, outline=(0, 0, 0))
    elif shape == "trapezoid":
        d.polygon([(pad + 8, pad), (size - pad - 8, pad),
                     (size - pad, size - pad), (pad, size - pad)],
                    fill=colour, outline=(0, 0, 0))
    elif shape == "rectangle":
        d.rectangle(box, fill=colour, outline=(0, 0, 0))
    else:
        d.rectangle(box, fill=colour, outline=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ── Polly TTS ─────────────────────────────────────────────────────────────


def synthesize_polly_wav(text: str, voice: str = "Joanna",
                          rate: int = 16000) -> bytes:
    """Call AWS Polly and return WAV bytes (16-bit mono PCM wrapped in
    a WAV header).  Polly's pcm output is raw PCM at the chosen sample
    rate; we wrap it for AudioBitsEncoder's encode_wav_bytes path."""
    import boto3
    import wave
    client = boto3.client("polly")
    resp = client.synthesize_speech(
        OutputFormat="pcm",
        SampleRate=str(rate),
        VoiceId=voice,
        Text=text,
    )
    pcm = resp["AudioStream"].read()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── HTTP ─────────────────────────────────────────────────────────────────


def post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ── Training + probing ────────────────────────────────────────────────────


def train_triples(node_url: str, repeats: int, lr: float,
                    cache_dir: Path, verbose: bool = False) -> list[dict]:
    """For each object, train (image+caption) and (audio+caption) pairs
    `repeats` times.  Cache image/audio bytes to disk to avoid hitting
    Polly on every run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, colour, shape in OBJECTS:
        img_path = cache_dir / f"{name}.jpg"
        wav_path = cache_dir / f"{name}.wav"

        if not img_path.exists():
            img_path.write_bytes(make_image(colour, shape))
        if not wav_path.exists():
            print(f"[polly] synthesizing audio for {name!r}", flush=True)
            wav_path.write_bytes(
                synthesize_polly_wav(f"this is a {name}"))

        img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

        for i in range(repeats):
            # observe_triple wires all three pairwise cross-edges in one
            # call (keyboard_text↔image_pixels, keyboard_text↔audio_features,
            # image_pixels↔audio_features), so a single presentation gives
            # the modality fabric every direct association.
            r = post_json(f"{node_url}/sensor/observe_triple", {
                "text": name,
                "image_b64": img_b64,
                "audio_b64": wav_b64,
                "lr": lr,
            })
            if verbose:
                print(f"  {name}  iter {i+1}/{repeats}  "
                        f"img_labels={r.get('img_labels')} "
                        f"aud_labels={r.get('aud_labels')}",
                        flush=True)
        print(f"[trained] {name}: {repeats}x image+audio paired with {name!r}",
                flush=True)
        summary.append({"object": name, "img_bytes": img_path.stat().st_size,
                          "wav_bytes": wav_path.stat().st_size})
    return summary


def probe_directions(node_url: str, cache_dir: Path) -> None:
    """For each object, query each modality and report what cross-pool
    predictions came back."""
    for name, _, _ in OBJECTS:
        img_b64 = base64.b64encode((cache_dir / f"{name}.jpg").read_bytes()).decode("ascii")
        wav_b64 = base64.b64encode((cache_dir / f"{name}.wav").read_bytes()).decode("ascii")

        # 1. image -> others
        r = post_json(f"{node_url}/sensor/observe", {
            "kind": "image", "bytes_b64": img_b64,
        })
        img_preds = r.get("predictions") or {}

        # 2. audio -> others
        r = post_json(f"{node_url}/sensor/observe", {
            "kind": "audio", "bytes_b64": wav_b64,
        })
        aud_preds = r.get("predictions") or {}

        # 3. text -> others via /chat
        r = post_json(f"{node_url}/chat", {"text": name})
        txt_preds = r.get("predictions") or {}

        print(f"\n=== {name} ===")
        print(f"  image ->  {list(img_preds.keys()) or '(none)'}")
        print(f"  audio ->  {list(aud_preds.keys()) or '(none)'}")
        print(f"  text  ->  {list(txt_preds.keys()) or '(none)'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node",    default="http://127.0.0.1:8090")
    p.add_argument("--repeats", type=int, default=10,
                     help="paired observations per object (default 10)")
    p.add_argument("--lr",      type=float, default=2.0)
    p.add_argument("--cache",   default="data/multimodal_toddler",
                     help="directory for cached images + Polly WAVs")
    p.add_argument("--probe-only", action="store_true",
                     help="skip training, just probe with cached assets")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cache = Path(args.cache).resolve()
    if not args.probe_only:
        print(f"[train] objects={len(OBJECTS)} repeats={args.repeats} lr={args.lr}",
                flush=True)
        train_triples(args.node, args.repeats, args.lr, cache, args.verbose)
        # Brain snapshot
        with urllib.request.urlopen(f"{args.node}/brain", timeout=10) as r:
            brain = json.load(r)
        mp = brain.get("multi_pool", {})
        print(f"\n[brain] pools={mp.get('pools')}  cross_edges={mp.get('cross_edges')}",
                flush=True)
    print(f"\n[probe] testing bidirectional recall...", flush=True)
    probe_directions(args.node, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
