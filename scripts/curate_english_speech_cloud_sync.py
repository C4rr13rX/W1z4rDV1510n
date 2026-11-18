#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:/Projects/W1z4rDV1510n/scripts/curate_english_speech_cloud_sync.py

English Corpus Speech Synthesis + Canonical Acoustic Decomposition + Zero-Local-Storage Cloud Sync
==================================================================================================

Architecture & Laws Enforced:
  1. Speech Synthesis Parity with C4rr13rX:
     - Uses C4rr13rX's Rapid Diplomat `ContextualLinguisticDisambiguator` + neural `edge-tts`
       (`en-US-ChristopherNeural` / `en-US-JennyNeural`) with optional AWS Polly fallback.
     - Zero media embellishment: preserves native TTS mono stream bitrate/sample rate directly.
  2. Strict Neuron-Economy & Ingress Decomposition Law:
     - Before unlinking each synthesized audio file, decomposes the speech waveform in RAM
       (via piped ffmpeg at BELOW_NORMAL priority — zero intermediate WAV files on disk) into:
         (a) `audio_bits_labels`: exact W1z4rDV1510n `AudioBitsEncoder` sparse tonotopic/temporal
             tokens (`aud:freq*`, `aud:amp*`, `aud:loud*`, `aud:freq*_t*`).
         (b) `pool_audio_motifs`: ultra-compact, state-deduplicated phonetic-prosodic motif sequence
             (voicing + semitone F0 + F1/F2 Bark/Mel formant pair + sibilance + energy) so shared
             vowels, consonants, and prosodic contours co-fire identical minimal neuron sets in
             `POOL_AUDIO` (`id=3`) without overflowing `recent_atoms_window=4096`.
         (c) `vocoder_b64u` (`W1ZA1`): deterministic 20-band Mel harmonic-plus-noise vocoder code
             (~300-650 bytes/utterance) that reconstructs 16 kHz mono WAV on demand for
             `/sensor/observe` or audio playback without re-downloading from cloud storage.
  3. Zero-Local-Disk-Bloat Cloud Transaction:
     - Every synthesized `.mp3` follows:
         synthesize temp -> decompose atoms -> GCS upload (`gs://w1z4rdv1510n-multimodal-corpus/...`)
         -> independent remote HEAD + byte_size + MD5/SHA256 verification -> immediate `unlink()`.
  4. Polite Elastic PC-Allowance Governor ("Nice" Resource Scaling):
     - Runs at `BELOW_NORMAL_PRIORITY_CLASS`, leaves >= 8.0 GiB free RAM + 12% under-cap margin
       (effective soft floor 8.96 GiB), scales concurrency up to 4 when free RAM >= 14.5 GiB, and
       shrinks smoothly to batch=1 + `gc.collect()` + `EmptyWorkingSet` when RAM tightens.

Usage:
  # Run initial verified batch (e.g. 150 core utterances across anchored/greetings/toddler/comm/STEM)
  C:/Python313/python.exe scripts/curate_english_speech_cloud_sync.py --limit 150 --resume

  # Run full English corpus continuously with zero disk growth
  C:/Python313/python.exe scripts/curate_english_speech_cloud_sync.py --all --resume

  # Verify cloud objects & co-fire manifest items into W1z4rD (:8090)
  C:/Python313/python.exe scripts/curate_english_speech_cloud_sync.py --verify-cloud-sample 25 --cofire-sample 10
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).resolve().parent
WIZARD_ROOT = SCRIPTS_DIR.parent
CCU_ROOT = Path("D:/Projects/CoolCryptoUtilities")
C4RR13RX_ROOT = Path("D:/Projects/C4rr13rX")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multimodal_cloud_sync import (  # noqa: E402
    ElasticResourceGovernor,
    MultimodalCloudSync,
)

import edge_tts  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402

MANIFEST_PATH = WIZARD_ROOT / "data" / "manifests" / "english_speech_cloud_manifest.jsonl"
SUMMARY_PATH = WIZARD_ROOT / "data" / "manifests" / "english_speech_cloud_summary.json"
TEMP_STAGE_ROOT = WIZARD_ROOT / "data" / "staging" / "tts_ephemeral"
CLOUD_AUDIO_PREFIX = "english_speech/v1/audio"

POOL_TEXT = 1
POOL_AUDIO = 3
POOL_ACTION = 4

WS_RE = re.compile(r"\s+")
SPEAKABLE_ALPHA_RE = re.compile(r"[A-Za-z]{2,}")
CODE_HEAVY_RE = re.compile(r"[{};=<>\[\]\\]{4,}|^\s*(def |fn |pub fn |import |#include |SELECT )", re.M)


# ============================================================================
# 1. C4rr13rX Natural Speech Disambiguator & Synthesis Pipeline
# ============================================================================

class ContextualLinguisticDisambiguator:
    """Ported directly from D:/Projects/C4rr13rX/scripts/natural_tts_pipeline.py
    (Rapid Diplomat heteronym & phonetic disambiguation).
    """

    PAST_INDICATORS = (
        "yesterday", "last night", "last week", "last year", "ago",
        "already", "previously", "have", "has", "had", "was", "were", "been",
    )
    FUTURE_INDICATORS = (
        "will", "shall", "to", "can", "could", "would", "should",
        "must", "may", "might", "please",
    )

    @classmethod
    def disambiguate_english(cls, text: str) -> str:
        working = text
        if re.search(r"\bread\b", working, re.IGNORECASE):
            lower = working.lower()
            is_past = any(ind in lower for ind in cls.PAST_INDICATORS) or (
                re.search(r"\b(i|he|she|they|we|who)\s+read\b", lower)
                and not any(ind in lower for ind in cls.FUTURE_INDICATORS)
            )
            rep = "red" if is_past else "reed"
            working = re.sub(r"\bread\b", rep, working, flags=re.IGNORECASE)

        if re.search(r"\b(lead\s+(pipe|pipes|poisoning|pencil|weight|shield|bullet|bullets|apron))\b", working, re.IGNORECASE):
            working = re.sub(r"\blead\b", "led", working, flags=re.IGNORECASE)
        elif re.search(r"\b(lead\s+(the|a|to|us|them|our))\b", working, re.IGNORECASE):
            working = re.sub(r"\blead\b", "leed", working, flags=re.IGNORECASE)

        if re.search(r"\b(to\s+wind|wind\s+(up|down|the\s+clock|the\s+watch|the\s+path))\b", working, re.IGNORECASE):
            working = re.sub(r"\bwind\b", "whynd", working, flags=re.IGNORECASE)
        elif re.search(r"\b((the|cold|strong|gentle|warm|high|gusty|gale)\s+wind|wind\s+(blows|blew|howls|howled|gusts|chime|speed|direction))\b", working, re.IGNORECASE):
            working = re.sub(r"\bwind\b", "wihnd", working, flags=re.IGNORECASE)

        if re.search(r"\b(live\s+(broadcast|stream|performance|show|coverage|audience|wires|feed|album|coding|ctf))\b", working, re.IGNORECASE):
            working = re.sub(r"\blive\b", "lyve", working, flags=re.IGNORECASE)

        if re.search(r"\b((a|single|tear)\s+(in|on|from|down)\s+(his|her|my|their|the)\s+eye|(in\s+tears)|shed\s+a\s+tear|tear\s+fell)\b", working, re.IGNORECASE):
            working = re.sub(r"\btear\b", "teer", working, flags=re.IGNORECASE)
        elif re.search(r"\b(tear\s+(down|apart|up|the\s+paper|the\s+document))\b", working, re.IGNORECASE):
            working = re.sub(r"\btear\b", "tare", working, flags=re.IGNORECASE)

        # Speak W1z4rD naturally as "Wizard" while keeping original text in manifest
        working = re.sub(r"\bW1z4rDV1510n\b", "Wizard Vision", working, flags=re.IGNORECASE)
        working = re.sub(r"\bW1z4rD\b", "Wizard", working, flags=re.IGNORECASE)
        working = re.sub(r"\bC4rr13rX\b", "Carrier X", working, flags=re.IGNORECASE)
        return working


VOICES = {
    "en": "en-US-ChristopherNeural",
    "en-female": "en-US-JennyNeural",
    "polly-ruth": "Ruth",
    "polly-matthew": "Matthew",
}


async def synthesize_speech_to_temp(
    text: str,
    temp_path: Path,
    *,
    engine: str = "edge",
    voice_key: str = "en",
) -> tuple[str, list[dict[str, Any]]]:
    """Synthesize native TTS MP3 directly to `temp_path` with zero upsampling/embellishment.
    Returns `(voice_id, word_marks)`.
    """
    spoken = ContextualLinguisticDisambiguator.disambiguate_english(text)
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "polly":
        import boto3

        voice_id = VOICES.get(voice_key, "Ruth")
        profile = os.getenv("AWS_PROFILE", "FountainServer")
        region = os.getenv("AWS_REGION", "us-east-1")
        session = boto3.Session(profile_name=profile, region_name=region)
        polly = session.client("polly")
        resp = polly.synthesize_speech(
            Text=spoken[:1500],
            Engine="neural",
            VoiceId=voice_id,
            LanguageCode="en-US",
            OutputFormat="mp3",
        )
        audio_bytes = resp["AudioStream"].read()
        temp_path.write_bytes(audio_bytes)
        return f"polly:{voice_id}", []

    voice_id = VOICES.get(voice_key, VOICES["en"])
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            comm = edge_tts.Communicate(
                spoken,
                voice_id,
                rate="+0%",
                pitch="+0Hz",
                boundary="WordBoundary",
                connect_timeout=12,
                receive_timeout=45,
            )
            audio_buf = bytearray()
            marks: list[dict[str, Any]] = []
            async for chunk in comm.stream():
                ctype = chunk.get("type")
                if ctype == "audio":
                    audio_buf.extend(chunk["data"])
                elif ctype in ("WordBoundary", "SentenceBoundary"):
                    marks.append({
                        "t_ms": int(round(chunk.get("offset", 0) / 10_000)),
                        "d_ms": int(round(chunk.get("duration", 0) / 10_000)),
                        "w": str(chunk.get("text", "")),
                    })
            if not audio_buf:
                raise RuntimeError("edge-tts returned 0 audio bytes")
            temp_path.write_bytes(bytes(audio_buf))
            return voice_id, marks
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            await asyncio.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"TTS synthesis failed after retries: {last_err}")


# ============================================================================
# 2. Canonical Acoustic/Spectral Atom Decomposition & Reconstructible Vocoder
# ============================================================================

def _build_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 20,
    fmin: float = 80.0,
    fmax: float = 7600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic triangular Mel filterbank + center frequencies (Hz)."""
    def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, n_bins)):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, n_bins)):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    # Normalize each filter to unit weight so band energies are scale-consistent
    sums = np.maximum(fb.sum(axis=1, keepdims=True), 1e-6)
    return fb / sums, hz_pts[1:-1].astype(np.float32)


_MEL_FB_20, _MEL_CENTERS_20 = _build_mel_filterbank(16000, 512, 20, 80.0, 7600.0)
_HANN_512 = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(512, dtype=np.float32) / 511.0))).astype(np.float32)
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def decode_mp3_bytes_to_pcm16k_mono(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 bytes strictly in memory to f32 [-1, 1] at 16,000 Hz mono.
    Uses piped stdin/stdout and BELOW_NORMAL_PRIORITY_CLASS — zero temp WAV files.
    """
    proc = subprocess.run(
        [
            _FFMPEG_EXE,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
        timeout=90,
        creationflags=ElasticResourceGovernor.subprocess_nice_CreationFlags(),
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"ffmpeg in-memory decode failed ({proc.returncode}): {err}")
    pcm_i16 = np.frombuffer(proc.stdout, dtype="<i2")
    return (pcm_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def encode_w1z4rd_audio_bits(samples: np.ndarray) -> dict[str, Any]:
    """Vectorized bit-accurate port of `crates/core/src/streaming/audio_bits.rs`
    (`AudioBitsConfig::default()`: sr=16000, ws=512, hop=256, freq_bins=32, amp_bins=8,
    time_slots=16, silence_threshold=0.005, stream_tag='aud').
    """
    ws, hop = 512, 256
    freq_bins, amp_bins, time_slots = 32, 8, 16
    silence_thr = 0.005
    n = int(samples.shape[0])
    if n < ws:
        return {"labels": [], "windows_processed": 0, "active_windows": 0}

    total_windows = (n - ws) // hop + 1
    idx = np.arange(ws)[None, :] + (np.arange(total_windows) * hop)[:, None]
    windows = samples[idx]  # (W, 512)
    rms = np.sqrt(np.mean(windows * windows, axis=1))
    active_mask = rms >= silence_thr
    active_indices = np.nonzero(active_mask)[0]
    if active_indices.size == 0:
        return {"labels": [], "windows_processed": total_windows, "active_windows": 0}

    active_win = windows[active_indices] * _HANN_512[None, :]
    spec = np.abs(np.fft.rfft(active_win, axis=1)).astype(np.float32)  # (A, 257)
    nyquist_bins = spec.shape[1]

    labels_set: set[str] = set()
    band_ranges = [
        ((b * nyquist_bins) // freq_bins, min(((b + 1) * nyquist_bins) // freq_bins, nyquist_bins))
        for b in range(freq_bins)
    ]

    for row_i, win_idx in enumerate(active_indices.tolist()):
        t_slot = min(int((win_idx / max(1, total_windows)) * time_slots), time_slots - 1)
        row_spec = spec[row_i]
        for band, (lo, hi) in enumerate(band_ranges):
            if lo >= hi:
                continue
            energy = float(np.sum(row_spec[lo:hi]) / (hi - lo))
            norm = min(energy / (ws * 0.5), 1.0)
            if norm < 0.005:
                continue
            amp_bin = min(int(norm * amp_bins), amp_bins - 1)
            labels_set.add(f"aud:freq{band}")
            labels_set.add(f"aud:amp{amp_bin}")
            labels_set.add(f"aud:freq{band}_t{t_slot}")
            labels_set.add(f"aud:amp{amp_bin}_t{t_slot}")

        loud_bin = min(int(float(rms[win_idx]) * amp_bins), amp_bins - 1)
        labels_set.add(f"aud:loud{loud_bin}")
        labels_set.add(f"aud:t{t_slot}")

    sorted_labels = sorted(labels_set)
    return {
        "labels": sorted_labels,
        "windows_processed": int(total_windows),
        "active_windows": int(active_indices.size),
    }


def _estimate_pitch_and_voicing(window: np.ndarray, sample_rate: int = 16000) -> tuple[float, bool]:
    """Fast normalized autocorrelation pitch + harmonic voicing detector (65..500 Hz)."""
    w = window - float(np.mean(window))
    energy = float(np.dot(w, w))
    if energy < 1e-6:
        return 0.0, False
    # Zero-crossing rate helps separate voiced vowels from high-frequency unvoiced sibilants
    zcr = float(np.mean(np.abs(np.diff(np.signbit(w)))))
    corr = np.correlate(w, w, mode="full")[len(w) - 1:]
    min_lag = int(sample_rate / 500.0)  # 32 samples
    max_lag = min(len(w) - 1, int(sample_rate / 65.0))  # 246 samples
    if max_lag <= min_lag:
        return 0.0, False
    seg = corr[min_lag:max_lag + 1]
    peak_idx = int(np.argmax(seg))
    peak_val = float(seg[peak_idx]) / max(energy, 1e-8)
    voiced = bool(peak_val >= 0.32 and zcr < 0.35)
    if not voiced:
        return 0.0, False
    f0 = float(sample_rate) / float(min_lag + peak_idx)
    return f0, True


def decompose_speech_canonical(
    samples: np.ndarray,
    word_marks: Optional[list[dict[str, Any]]] = None,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Decompose 16 kHz mono speech into:
      1. `audio_bits_labels`: exact W1z4rDV1510n AudioBitsEncoder sparse labels.
      2. `pool_audio_motifs`: state-deduplicated phonetic-prosodic motif sequence
         for minimal-neuron, high-consistency Hebbian co-firing in `POOL_AUDIO`.
      3. `vocoder_b64u` (`W1ZA1`): compact 20-band Mel harmonic+noise vocoder packet
         capable of reconstructing 16 kHz mono WAV without cloud re-download.
    """
    duration_s = round(float(samples.shape[0]) / float(sample_rate), 3)
    abits = encode_w1z4rd_audio_bits(samples)

    ws = 512       # 32 ms analysis window
    hop = 400      # 25 ms frame stride (40 fps)
    n = int(samples.shape[0])
    if n < ws:
        samples = np.pad(samples, (0, ws - n))
        n = ws

    n_frames = (n - ws) // hop + 1
    frame_bytes = bytearray()
    raw_motif_states: list[str] = []

    # Pre-pass over frames to compute Mel log-energies, voicing, F0, and RMS
    mel_db_list: list[np.ndarray] = []
    f0_list: list[float] = []
    voiced_list: list[bool] = []
    rms_list: list[float] = []

    for fi in range(n_frames):
        win = samples[fi * hop: fi * hop + ws]
        rms = float(np.sqrt(np.mean(win * win)))
        rms_list.append(rms)
        if rms < 0.0035:
            mel_db_list.append(np.full(20, -80.0, dtype=np.float32))
            f0_list.append(0.0)
            voiced_list.append(False)
            continue
        f0, voiced = _estimate_pitch_and_voicing(win, sample_rate)
        spec_mag = np.abs(np.fft.rfft(win * _HANN_512)).astype(np.float32)
        mel_e = _MEL_FB_20 @ spec_mag
        mel_db = 20.0 * np.log10(np.maximum(mel_e, 1e-5))
        mel_db_list.append(mel_db)
        f0_list.append(f0)
        voiced_list.append(voiced)

    mel_matrix = np.stack(mel_db_list, axis=0)  # (F, 20)
    active_vals = mel_matrix[mel_matrix > -79.0]
    if active_vals.size > 0:
        db_floor = float(np.percentile(active_vals, 5))
        db_peak = float(np.percentile(active_vals, 99.5))
        if db_peak - db_floor < 12.0:
            db_peak = db_floor + 12.0
    else:
        db_floor, db_peak = -60.0, -20.0

    max_rms = max(max(rms_list, default=0.01), 0.01)

    for fi in range(n_frames):
        rms = rms_list[fi]
        if rms < 0.0035:
            # Silent frame: flags_gain = 0, f0 = 0, 10 zero bytes
            frame_bytes.extend(b"\x00" * 12)
            raw_motif_states.append("_")
            continue

        voiced = voiced_list[fi]
        f0 = f0_list[fi]
        gain_7bit = min(127, max(1, int(round((rms / max_rms) * 127.0))))
        flags_gain = (0x80 if voiced else 0x00) | gain_7bit

        if voiced and f0 >= 65.0:
            # Quantize F0 logarithmically across 65..520 Hz (3 octaves = 36 semitones) in 1..255
            f0_norm = np.clip(math.log2(f0 / 65.0) / 3.0, 0.0, 1.0)
            f0_q = max(1, min(255, int(round(1 + f0_norm * 254.0))))
            semitone_bin = min(35, int(f0_norm * 36.0))
            pitch_tok = f"P{semitone_bin // 3:X}"  # 12 quarter-octave prosodic pitch classes
        else:
            f0_q = 0
            pitch_tok = "U"

        # Quantize 20 Mel bands into 4-bit nibbles (0..15)
        norm_mel = np.clip((mel_matrix[fi] - db_floor) / (db_peak - db_floor), 0.0, 1.0)
        nibbles = np.round(norm_mel * 15.0).astype(np.uint8)
        packed_mel = bytearray(10)
        for bi in range(10):
            packed_mel[bi] = int((nibbles[2 * bi] << 4) | nibbles[2 * bi + 1])

        frame_bytes.append(flags_gain)
        frame_bytes.append(f0_q)
        frame_bytes.extend(packed_mel)

        # Canonical phonetic-spectral motif state:
        # Formant 1 peak in low/mid bands (0..7: ~80-1100 Hz), Formant 2 in mid/high (5..15: ~600-4200 Hz),
        # Sibilance in high bands (15..19: ~4200-7600 Hz).
        f1_band = int(np.argmax(nibbles[:8]))
        f2_band = int(5 + np.argmax(nibbles[5:16]))
        sib_high = int(np.mean(nibbles[15:]) >= 9)
        energy_tier = min(3, int(( rms / max_rms ) * 4.0))
        state_tok = f"{pitch_tok}F{f1_band:X}{f2_band:X}{'S' if sib_high else ''}{energy_tier}"
        raw_motif_states.append(state_tok)

    # Collapse consecutive identical phonetic states with coarse duration (1=short <=50ms, 2=med <=125ms, 3=long)
    dedup_motifs: list[str] = []
    idx_m = 0
    while idx_m < len(raw_motif_states):
        cur = raw_motif_states[idx_m]
        run = 1
        while idx_m + run < len(raw_motif_states) and raw_motif_states[idx_m + run] == cur:
            run += 1
        idx_m += run
        if cur == "_":
            if run >= 3 and dedup_motifs and dedup_motifs[-1] != "|":
                dedup_motifs.append("|")  # Prosodic pause boundary
            continue
        dur_code = "1" if run <= 2 else ("2" if run <= 5 else "3")
        dedup_motifs.append(f"{cur}:{dur_code}")

    motif_sequence_str = " ".join(dedup_motifs)
    motif_bytes = motif_sequence_str.encode("ascii")

    # Pack W1ZA1 binary vocoder container + zlib compress
    header = struct.pack(
        "<5sBHff",
        b"W1ZA1",
        20,  # n_mels
        n_frames,
        float(db_floor),
        float(db_peak),
    )
    compressed_vocoder = zlib.compress(header + bytes(frame_bytes), level=6)
    vocoder_b64u = base64.urlsafe_b64encode(compressed_vocoder).decode("ascii").rstrip("=")

    raw_wav_equiv_bytes = 44 + int(samples.shape[0]) * 2
    return {
        "codec": "W1ZA1-mel20-hnm-25ms",
        "sample_rate_hz": 16000,
        "frames": n_frames,
        "active_windows": abits["active_windows"],
        "audio_bits_count": len(abits["labels"]),
        "audio_bits_labels": abits["labels"],
        "pool_audio_motifs": motif_sequence_str,
        "pool_audio_b64u": base64.urlsafe_b64encode(motif_bytes).decode("ascii").rstrip("="),
        "vocoder_b64u": vocoder_b64u,
        "word_marks": (word_marks or [])[:64],
        "economy": {
            "raw_16k_wav_bytes": raw_wav_equiv_bytes,
            "motif_bytes": len(motif_bytes),
            "vocoder_compressed_bytes": len(compressed_vocoder),
            "pool_audio_compression_ratio": round(raw_wav_equiv_bytes / max(1, len(motif_bytes)), 2),
            "vocoder_compression_ratio": round(raw_wav_equiv_bytes / max(1, len(compressed_vocoder)), 2),
        },
    }


def reconstruct_wav_bytes(compact_audio_atoms: dict[str, Any]) -> bytes:
    """Deterministically reconstruct a 16 kHz 16-bit mono WAV directly from
    `compact_audio_atoms["vocoder_b64u"]` without re-downloading from cloud storage.
    Uses phase-continuous harmonic + shaped-noise overlap-add synthesis.
    """
    b64u = compact_audio_atoms["vocoder_b64u"]
    pad = "=" * ((4 - len(b64u) % 4) % 4)
    raw = zlib.decompress(base64.urlsafe_b64decode(b64u + pad))
    magic, n_mels, n_frames, db_floor, db_peak = struct.unpack("<5sBHff", raw[:16])
    if magic != b"W1ZA1" or n_mels != 20:
        raise ValueError(f"Unsupported compact audio container: {magic!r}")

    sr = 16000
    ws = 512
    hop = 400
    out_len = (max(1, n_frames) - 1) * hop + ws
    out = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    rng = np.random.default_rng(42)  # Deterministic noise seed

    # Precompute frequency grid for spectral envelope interpolation
    rfft_freqs = np.fft.rfftfreq(ws, d=1.0 / sr).astype(np.float32)
    phase = 0.0
    offset = 16

    for fi in range(n_frames):
        chunk = raw[offset: offset + 12]
        offset += 12
        if len(chunk) < 12:
            break
        flags_gain = chunk[0]
        f0_q = chunk[1]
        gain = (flags_gain & 0x7F) / 127.0
        if gain <= 0.0:
            continue
        voiced = bool(flags_gain & 0x80) and f0_q > 0

        nibbles = np.empty(20, dtype=np.float32)
        for bi in range(10):
            bval = chunk[2 + bi]
            nibbles[2 * bi] = float((bval >> 4) & 0x0F)
            nibbles[2 * bi + 1] = float(bval & 0x0F)

        mel_db = db_floor + (nibbles / 15.0) * (db_peak - db_floor)
        mel_lin = np.power(10.0, mel_db / 20.0)
        env_rfft = np.interp(rfft_freqs, _MEL_CENTERS_20, mel_lin, left=mel_lin[0], right=mel_lin[-1])

        if voiced:
            f0 = 65.0 * (2.0 ** (((f0_q - 1) / 254.0) * 3.0))
            t = np.arange(ws, dtype=np.float32) / sr
            # Band-limited harmonic glottal excitation + subtle high-band breathiness
            exc = np.zeros(ws, dtype=np.float32)
            max_h = min(28, int((sr * 0.45) // max(f0, 1.0)))
            for h in range(1, max_h + 1):
                exc += (1.0 / (h ** 0.85)) * np.sin(2.0 * np.pi * (f0 * h) * t + h * phase)
            phase = float((phase + 2.0 * np.pi * f0 * (hop / sr)) % (2.0 * np.pi))
            noise = rng.standard_normal(ws).astype(np.float32) * 0.12
            src_spec = np.fft.rfft((exc * 0.88 + noise) * _HANN_512)
        else:
            noise = rng.standard_normal(ws).astype(np.float32)
            src_spec = np.fft.rfft(noise * _HANN_512)

        # Normalize excitation spectrum Locally and apply Mel spectral envelope
        src_mag = np.maximum(np.abs(src_spec), 1e-4)
        shaped_spec = (src_spec / src_mag) * env_rfft
        frame_wav = np.fft.irfft(shaped_spec, n=ws).astype(np.float32)
        frame_rms = float(np.sqrt(np.mean(frame_wav * frame_wav)))
        if frame_rms > 1e-6:
            frame_wav = frame_wav * ((gain * 0.22) / frame_rms)

        start = fi * hop
        out[start: start + ws] += frame_wav * _HANN_512
        win_sum[start: start + ws] += _HANN_512 * _HANN_512

    nz = win_sum > 1e-4
    out[nz] /= win_sum[nz]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.95:
        out *= 0.92 / peak
    pcm_i16 = np.round(np.clip(out, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm_i16),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sr,
        sr * 2,
        2,
        16,
        b"data",
        len(pcm_i16),
    )
    return wav_header + pcm_i16


# ============================================================================
# 3. Comprehensive English Corpus Gatherer (Streaming + Deduplicated)
# ============================================================================

@dataclass
class UtteranceItem:
    text_sha256: str
    text: str
    tier: int
    sources: list[str] = field(default_factory=list)
    modality_links: dict[str, Any] = field(default_factory=dict)


def _clean_english_text(raw: str) -> str:
    txt = WS_RE.sub(" ", (raw or "").strip())
    return txt


def _is_speakable_english(txt: str) -> bool:
    if len(txt) < 2 or len(txt) > 1200:
        return False
    if not SPEAKABLE_ALPHA_RE.search(txt):
        return False
    if txt.startswith(("http://", "https://", "{", "[")):
        return False
    if CODE_HEAVY_RE.search(txt):
        return False
    # Skip Cyrillic / non-Latin scripts in English speech corpus
    if re.search(r"[\u0400-\u04FF\u4E00-\u9FFF]", txt):
        return False
    return True


def _split_utterances(text: str, max_chars: int = 340) -> list[str]:
    cleaned = _clean_english_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned] if _is_speakable_english(cleaned) else []
    sents = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    cur = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = f"{cur} {s}"
        else:
            if _is_speakable_english(cur):
                chunks.append(cur)
            cur = s
    if cur and _is_speakable_english(cur):
        chunks.append(cur)
    return chunks


def _extract_python_literal(py_path: Path, var_name: str) -> Any:
    """Safely parse top-level constant assignments from corpus builder scripts via AST
    without importing or executing network/training side effects.
    """
    if not py_path.exists():
        return None
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return ast.literal_eval(node.value)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == var_name and node.value is not None:
                    return ast.literal_eval(node.value)
    except Exception:  # noqa: BLE001
        return None
    return None


def iter_english_training_candidates(max_tier: int = 4) -> Iterator[tuple[int, str, str, dict[str, Any]]]:
    """Stream `(tier, text, corpus_source, modality_links)` across W1z4rDV1510n & CoolCryptoUtilities."""

    # --- TIER 0: Anchored Memories, Greetings & Core Conversational Basics ---
    anchored_paths = [
        CCU_ROOT / "storage" / "wizard_memory" / "anchored_memory.json",
        CCU_ROOT / "data" / "anchored_memories.json",
        WIZARD_ROOT / "data" / "anchored_memories.json",
    ]
    for ap in anchored_paths:
        if ap.exists():
            try:
                data = json.loads(ap.read_text(encoding="utf-8"))
                for item in data.get("items", []):
                    aid = item.get("id", "anchor")
                    ans = _clean_english_text(item.get("answer", ""))
                    prompt = _clean_english_text(item.get("prompt", ""))
                    if _is_speakable_english(prompt):
                        yield (0, prompt, "ccu:anchored_memory", {"anchor_id": aid, "role": "prompt", "paired_response": ans[:240]})
                    if _is_speakable_english(ans):
                        for seg in _split_utterances(ans):
                            yield (0, seg, "ccu:anchored_memory", {"anchor_id": aid, "role": "answer", "paired_prompt": prompt})
                    for var in item.get("variations", [])[:6]:
                        v = _clean_english_text(var)
                        if _is_speakable_english(v):
                            yield (0, v, "ccu:anchored_memory", {"anchor_id": aid, "role": "variation", "paired_response": ans[:240]})
            except Exception:  # noqa: BLE001
                pass

    greetings_jsonl = WIZARD_ROOT / "data" / "training" / "greetings_001.jsonl"
    if greetings_jsonl.exists():
        for line in greetings_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            p, r = _clean_english_text(row.get("prompt", "")), _clean_english_text(row.get("response", ""))
            if _is_speakable_english(p):
                yield (0, p, "w1z4rd:greetings_001", {"role": "prompt", "paired_response": r})
            if _is_speakable_english(r):
                yield (0, r, "w1z4rd:greetings_001", {"role": "response", "paired_prompt": p})

    convos = _extract_python_literal(WIZARD_ROOT / "scripts" / "train_convos_atoms.py", "CONVERSATIONS") or []
    for q, a in convos:
        q_c, a_c = _clean_english_text(q), _clean_english_text(a)
        if _is_speakable_english(q_c):
            yield (0, q_c, "w1z4rd:train_convos_atoms", {"role": "prompt", "paired_response": a_c})
        if _is_speakable_english(a_c):
            yield (0, a_c, "w1z4rd:train_convos_atoms", {"role": "response", "paired_prompt": q_c})

    ccu_convos = _extract_python_literal(
        CCU_ROOT / "tools" / "wizard_learning" / "ingest_conversational_basics.py",
        "CONVERSATIONAL_PAIRS",
    ) or []
    for cluster in ccu_convos:
        ans = _clean_english_text(cluster.get("answer", ""))
        inputs = cluster.get("inputs", [])
        first_in = _clean_english_text(inputs[0]) if inputs else ""
        if _is_speakable_english(ans):
            for seg in _split_utterances(ans):
                yield (0, seg, "ccu:conversational_basics", {"role": "answer", "paired_prompt": first_in})
        for inp in inputs[:4]:
            ic = _clean_english_text(inp)
            if _is_speakable_english(ic):
                yield (0, ic, "ccu:conversational_basics", {"role": "prompt", "paired_response": ans[:240]})

    if max_tier < 1:
        return

    # --- TIER 1: Toddler / First-Words / Multimodal Concept Streams ---
    toddler_pairs = _extract_python_literal(
        WIZARD_ROOT / "scripts" / "brain_toddler_with_categorical.py", "PAIRS"
    ) or []
    for word, cat in toddler_pairs:
        sentence = f"A {word} is a {cat}." if cat not in ("color", "food") else f"{word.capitalize()} is a {cat}."
        yield (1, word, "w1z4rd:toddler_categorical", {"category": cat, "paired_response": cat})
        yield (1, sentence, "w1z4rd:toddler_categorical", {"concept": word, "category": cat})

    # Concept dataset (levels -1, 0, 1 in Tier 1; levels 2, 3 in Tier 3)
    concept_jsonl = WIZARD_ROOT / "data" / "foundation" / "concept_dataset.jsonl"
    if concept_jsonl.exists():
        with open(concept_jsonl, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                lvl = int(rec.get("level", 2))
                tier = 1 if lvl <= 1 else 3
                if tier > max_tier:
                    continue
                word = _clean_english_text(rec.get("clean") or rec.get("concept") or "")
                defn = _clean_english_text(rec.get("definition") or "")
                imgs = [str(p).replace("\\", "/") for p in (rec.get("images") or [])[:3]]
                links = {
                    "concept_id": rec.get("id", ""),
                    "concept": word,
                    "category": rec.get("category", ""),
                    "level": lvl,
                }
                if imgs:
                    links["local_image_refs"] = imgs
                if _is_speakable_english(word) and tier == 1:
                    yield (tier, word, f"w1z4rd:concept_dataset_L{lvl}", links)
                if _is_speakable_english(defn) and "one of the first words you learn" not in defn:
                    for seg in _split_utterances(defn):
                        yield (tier, seg, f"w1z4rd:concept_dataset_L{lvl}", links)

    if max_tier < 2:
        return

    # --- TIER 2: Communication/Social, Foundational Seeds & Moral Prerogatives ---
    comm_py = WIZARD_ROOT / "scripts" / "build_communication_corpus.py"
    for var_name in ("LONG_Q_SHORT_A_PAIRS", "RUNON_CORRECTION_PAIRS", "GRAMMAR_CORRECTION_PAIRS", "LONG_Q_LONG_A_PAIRS"):
        pairs = _extract_python_literal(comm_py, var_name) or []
        for q, a in pairs:
            for seg in _split_utterances(a):
                yield (2, seg, f"w1z4rd:communication:{var_name.lower()}", {"role": "answer", "paired_prompt": _clean_english_text(q)[:220]})

    seeds_py = WIZARD_ROOT / "scripts" / "build_foundational_seeds.py"
    seed_groups = _extract_python_literal(seeds_py, "SEED_GROUPS") or []
    for group_title, discipline, qas in seed_groups:
        for q, a in qas:
            q_c = _clean_english_text(q)
            if _is_speakable_english(q_c):
                yield (2, q_c, "w1z4rd:foundational_seeds", {"topic": group_title, "discipline": discipline or "general", "role": "question"})
            for seg in _split_utterances(a):
                yield (2, seg, "w1z4rd:foundational_seeds", {"topic": group_title, "discipline": discipline or "general", "paired_prompt": q_c})

    prerogs = _extract_python_literal(
        CCU_ROOT / "tools" / "wizard_learning" / "curriculum.py", "PREROGATIVES"
    ) or []
    for pid, meaning, scripture in prerogs:
        stmt = f"Core prerogative {pid}, from {scripture}: {meaning}."
        yield (2, stmt, "ccu:foundation_curriculum", {"prerogative_id": pid, "source": scripture})

    if max_tier < 3:
        return

    # --- TIER 3: Empirical STEM Material, Pedagogy, Multimodal Pairs & K-12 ---
    # Parse EmpiricalCurriculumItem entries cleanly via AST from empirical_material.py
    emp_path = CCU_ROOT / "tools" / "wizard_learning" / "empirical_material.py"
    if emp_path.exists():
        try:
            tree = ast.parse(emp_path.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "EmpiricalCurriculumItem":
                    kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg and isinstance(k.value, ast.Constant)}
                    eid = kw.get("id", "")
                    q = _clean_english_text(str(kw.get("question", "")))
                    a = _clean_english_text(str(kw.get("answer", "")))
                    meta = {"empirical_id": eid, "category": kw.get("category", ""), "topic": kw.get("topic", ""), "units": kw.get("units", "")}
                    if _is_speakable_english(q):
                        yield (3, q, "ccu:empirical_material", {**meta, "role": "question"})
                    for seg in _split_utterances(a):
                        yield (3, seg, "ccu:empirical_material", {**meta, "role": "answer", "paired_prompt": q})
        except Exception:  # noqa: BLE001
            pass

    ped_principles = _extract_python_literal(
        WIZARD_ROOT / "scripts" / "build_pedagogy_corpus.py", "LEARNING_PRINCIPLES"
    ) or []
    for lp in ped_principles:
        name = _clean_english_text(lp.get("name", ""))
        mech = _clean_english_text(lp.get("mechanism", ""))
        for seg in _split_utterances(f"{name}: {mech}"):
            yield (3, seg, "w1z4rd:pedagogy_corpus", {"principle": name})

    for fname, src_tag in (
        ("multimodal_pairs_001.jsonl", "w1z4rd:multimodal_pairs_001"),
        ("k12_subjects_001.jsonl", "w1z4rd:k12_subjects_001"),
        ("categorical_unified_001.jsonl", "w1z4rd:categorical_unified_001"),
    ):
        fpath = WIZARD_ROOT / "data" / "training" / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                p = _clean_english_text(row.get("prompt") or row.get("question") or "")
                r = _clean_english_text(row.get("response") or row.get("answer") or "")
                if len(r.split()) >= 4 and _is_speakable_english(r):
                    for seg in _split_utterances(r):
                        yield (3, seg, src_tag, {"paired_prompt": p[:200]})

    # Relational priors natural-language summary statements
    rel_files = [
        (WIZARD_ROOT / "data" / "relational_priors.json", "w1z4rd:relational_priors"),
        (WIZARD_ROOT / "data" / "astronomy" / "exoplanet_snapshot.json", "w1z4rd:astronomy_snapshot"),
        (WIZARD_ROOT / "data" / "tox21" / "tox21_snapshot.json", "w1z4rd:tox21_snapshot"),
        (WIZARD_ROOT / "data" / "uspto" / "chem_snapshot.json", "w1z4rd:uspto_chem_snapshot"),
    ]
    for rpath, rtag in rel_files:
        if rpath.exists():
            yield (
                3,
                f"Relational prior substrate {rpath.stem.replace('_', ' ')} anchors spatial and causal symbol transitions in Wizard Vision.",
                rtag,
                {"prior_file": rpath.name},
            )

    if max_tier < 4:
        return

    # --- TIER 4: Extended Conversational, Library, NCBI & Simple Wikipedia ---
    for fname, src_tag in (
        ("conversation_basics_001.jsonl", "w1z4rd:conversation_basics_001"),
        ("conversation_dialog_001.jsonl", "w1z4rd:conversation_dialog_001"),
        ("agent_planning_001.jsonl", "w1z4rd:agent_planning_001"),
        ("error_diagnosis_001.jsonl", "w1z4rd:error_diagnosis_001"),
        ("wordnet_categories_001.jsonl", "w1z4rd:wordnet_categories_001"),
    ):
        fpath = WIZARD_ROOT / "data" / "training" / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                p = _clean_english_text(row.get("prompt") or row.get("question") or "")
                r = _clean_english_text(row.get("response") or row.get("answer") or "")
                if _is_speakable_english(p) and len(p.split()) >= 3:
                    yield (4, p, src_tag, {"role": "prompt", "paired_response": r[:200]})
                for seg in _split_utterances(r):
                    yield (4, seg, src_tag, {"role": "response", "paired_prompt": p[:200]})

    for fname, src_tag in (
        ("class_corpus.jsonl", "w1z4rd:foundation_class_corpus"),
        ("ncbi_pairs.jsonl", "w1z4rd:foundation_ncbi_pairs"),
        ("simple_wiki_articles.jsonl", "w1z4rd:simple_wikipedia"),
    ):
        fpath = WIZARD_ROOT / "data" / "foundation" / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                txt = _clean_english_text(
                    row.get("answer")
                    or row.get("summary")
                    or row.get("text", "")[:420]
                )
                title = _clean_english_text(row.get("title") or row.get("question") or "")
                segs = _split_utterances(txt)
                if segs:
                    yield (4, segs[0], src_tag, {"title": title[:140]})


def collect_curated_utterances(
    *,
    already_synced_sha256: set[str],
    limit: Optional[int],
    min_tier: int = 0,
    max_tier: int = 4,
) -> list[UtteranceItem]:
    """Stream candidates cleanly, deduplicating by `text_sha256` with bounded RAM."""
    by_hash: dict[str, UtteranceItem] = {}
    cap = limit if (limit is not None and limit > 0) else None
    per_source_cap = (cap + 40) if cap is not None else 50_000
    source_counts: dict[tuple[int, str], int] = {}

    for tier, text, source, links in iter_english_training_candidates(max_tier=max_tier):
        if tier < min_tier or source_counts.get((tier, source), 0) >= per_source_cap:
            continue
        t_norm = _clean_english_text(text)
        if not _is_speakable_english(t_norm):
            continue
        digest = hashlib.sha256(t_norm.encode("utf-8")).hexdigest()
        if digest in already_synced_sha256:
            continue
        existing = by_hash.get(digest)
        if existing is not None:
            if source not in existing.sources:
                existing.sources.append(source)
            if tier < existing.tier:
                existing.tier = tier
            for k, v in links.items():
                if k not in existing.modality_links and v:
                    existing.modality_links[k] = v
        else:
            source_counts[(tier, source)] = source_counts.get((tier, source), 0) + 1
            by_hash[digest] = UtteranceItem(
                text_sha256=digest,
                text=t_norm,
                tier=tier,
                sources=[source],
                modality_links=dict(links),
            )

    # Round-robin interleave across sources within each tier so every corpus domain gets equal voice
    tier_source_groups: dict[int, dict[str, list[UtteranceItem]]] = {}
    for u in sorted(by_hash.values(), key=lambda x: (x.tier, x.sources[0], x.text_sha256[:8])):
        tier_source_groups.setdefault(u.tier, {}).setdefault(u.sources[0], []).append(u)

    buckets: dict[int, list[UtteranceItem]] = {}
    ordered: list[UtteranceItem] = []
    for t in sorted(tier_source_groups):
        src_lists = [tier_source_groups[t][s] for s in sorted(tier_source_groups[t])]
        interleaved: list[UtteranceItem] = []
        max_len = max((len(lst) for lst in src_lists), default=0)
        for i in range(max_len):
            for lst in src_lists:
                if i < len(lst):
                    interleaved.append(lst[i])
        buckets[t] = interleaved
        ordered.extend(interleaved)

    if cap is not None and len(ordered) > cap:
        active_tiers = sorted(buckets.keys())
        even_share = max(1, cap // max(1, len(active_tiers)))
        quotas = {
            t: min(len(buckets[t]), even_share)
            for t in active_tiers
        }
        selected: list[UtteranceItem] = []
        used_hashes: set[str] = set()
        for t in sorted(quotas):
            for it in buckets.get(t, [])[:quotas[t]]:
                if len(selected) < cap:
                    selected.append(it)
                    used_hashes.add(it.text_sha256)
        if len(selected) < cap:
            for it in ordered:
                if it.text_sha256 not in used_hashes:
                    selected.append(it)
                    if len(selected) >= cap:
                        break
        return selected
    return ordered


# ============================================================================
# 4. Non-Blocking Live Co-Firing into W1z4rDV1510n (:8090)
# ============================================================================

def cofire_record_into_wizard(
    record: dict[str, Any],
    *,
    node_url: str = "http://127.0.0.1:8090",
    timeout_s: float = 4.0,
) -> dict[str, Any]:
    """Co-fire `text + compact_audio` into W1z4rDV1510n without re-downloading from cloud:
      1. `POST /brain/pretrain_binding` with `POOL_TEXT=1` + `POOL_AUDIO=3` (`pool_audio_b64u`).
      2. `POST /sensor/observe` (`kind="audio"`, `paired_text=text`) using on-the-fly WAV
         reconstructed from `vocoder_b64u`.
    Fails open quickly if the node is offline or busy holding a checkpoint lock.
    """
    out: dict[str, Any] = {"node_url": node_url, "binding_ok": False, "sensor_ok": False}
    text = record["text"]
    atoms = record["compact_audio_atoms"]
    text_b64u = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")
    aud_motif_b64u = atoms["pool_audio_b64u"]

    # 1. Phase A-E substrate binding (`POOL_TEXT` <-> `POOL_AUDIO`)
    frames = [
        {"pool_id": POOL_TEXT, "frame": text_b64u},
        {"pool_id": POOL_AUDIO, "frame": aud_motif_b64u},
    ]
    paired_resp = (record.get("modality_links") or {}).get("paired_response")
    if isinstance(paired_resp, str) and paired_resp.strip():
        act_b64u = base64.urlsafe_b64encode(paired_resp.encode("utf-8")).decode("ascii").rstrip("=")
        frames.append({"pool_id": POOL_ACTION, "frame": act_b64u})

    try:
        req = urllib.request.Request(
            f"{node_url}/brain/pretrain_binding",
            data=json.dumps({"frames": frames}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            out["binding_ok"] = bool(body.get("ok"))
            out["binding_resp"] = body
    except Exception as exc:  # noqa: BLE001
        out["binding_err"] = str(exc)[:180]

    # 2. Sensor online cross-pool audio<->text observation using reconstructed WAV
    try:
        wav_bytes = reconstruct_wav_bytes(atoms)
        wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
        req2 = urllib.request.Request(
            f"{node_url}/sensor/observe",
            data=json.dumps({
                "kind": "audio",
                "bytes_b64": wav_b64,
                "paired_text": text,
                "lr": 0.65,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=timeout_s) as resp2:
            body2 = json.loads(resp2.read().decode("utf-8"))
            out["sensor_ok"] = bool(body2.get("labels_count", 0) > 0)
            out["sensor_labels_count"] = body2.get("labels_count", 0)
            out["trained_pair"] = body2.get("trained_pair", False)
    except Exception as exc:  # noqa: BLE001
        out["sensor_err"] = str(exc)[:180]

    return out


# ============================================================================
# 5. Main Elastic Zero-Bloat Speech Synthesis & Cloud Sync Pipeline
# ============================================================================

def load_existing_manifest(manifest_path: Path) -> tuple[set[str], int]:
    seen: set[str] = set()
    count = 0
    if not manifest_path.exists():
        return seen, 0
    with open(manifest_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("verified_in_cloud") and rec.get("deleted_locally") and rec.get("text_sha256"):
                    seen.add(str(rec["text_sha256"]))
                    count += 1
            except Exception:  # noqa: BLE001
                continue
    return seen, count


async def process_one_utterance(
    item: UtteranceItem,
    *,
    sync: MultimodalCloudSync,
    stage_dir: Path,
    tts_engine: str,
    voice_key: str,
) -> dict[str, Any]:
    """Atomic per-utterance transaction:
    Synthesize temp MP3 -> Decompose canonical acoustic atoms in RAM ->
    Upload to GCS -> Verify remote HEAD/size/MD5 -> Immediate local unlink().
    """
    short_id = f"en_tts_{item.text_sha256[:16]}"
    temp_mp3 = stage_dir / f"{short_id}_{os.getpid()}_{time.monotonic_ns()}.mp3"
    rel_key = f"{CLOUD_AUDIO_PREFIX}/{item.text_sha256[:2]}/{item.text_sha256}.mp3"

    try:
        voice_id, word_marks = await synthesize_speech_to_temp(
            item.text,
            temp_mp3,
            engine=tts_engine,
            voice_key=voice_key,
        )
        mp3_bytes = temp_mp3.read_bytes()
        pcm_16k = decode_mp3_bytes_to_pcm16k_mono(mp3_bytes)
        duration_s = round(float(pcm_16k.shape[0]) / 16000.0, 3)
        compact_atoms = decompose_speech_canonical(pcm_16k, word_marks=word_marks, sample_rate=16000)
        del pcm_16k
        del mp3_bytes

        receipt = await asyncio.to_thread(
            sync.upload_verify_and_delete,
            temp_mp3,
            rel_key,
            content_type="audio/mpeg",
            metadata={
                "text_sha256": item.text_sha256,
                "voice_id": voice_id,
                "duration_s": str(duration_s),
                "corpus_tier": str(item.tier),
            },
            cleanup_temp_on_failure=True,
        )
        if temp_mp3.exists():
            temp_mp3.unlink()

        return {
            "id": short_id,
            "text_sha256": item.text_sha256,
            "text": item.text,
            "corpus_source": item.sources[0] if len(item.sources) == 1 else item.sources,
            "tier": item.tier,
            "voice_id": voice_id,
            "modality_links": item.modality_links,
            "cloud_uri": receipt.cloud_uri,
            "byte_size": receipt.byte_size,
            "audio_sha256": receipt.sha256,
            "duration_s": duration_s,
            "compact_audio_atoms": compact_atoms,
            "verified_in_cloud": receipt.verified_in_cloud,
            "deleted_locally": receipt.deleted_locally and (not temp_mp3.exists()),
            "cloud_generation": receipt.generation_or_etag,
            "synced_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        if temp_mp3.exists():
            try:
                temp_mp3.unlink()
            except OSError:
                pass


async def run_pipeline(args: argparse.Namespace) -> int:
    governor = ElasticResourceGovernor(
        min_free_gib=args.min_free_gib,
        under_cap_margin=0.12,
        abundant_free_gib=args.abundant_free_gib,
        max_concurrency=args.max_concurrency,
        apply_below_normal_priority=True,
    )
    init_allowance = governor.sample()

    sync = MultimodalCloudSync(
        provider=args.provider,
        bucket=args.bucket or None,
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    seen_hashes, prior_count = (
        load_existing_manifest(MANIFEST_PATH) if args.resume else (set(), 0)
    )

    limit = None if args.all else args.limit
    print("=" * 78)
    print("  W1z4rDV1510n English Speech Synthesis -> Canonical Atoms -> Cloud Zero-Bloat")
    print("=" * 78)
    print(f"  Cloud Bucket     : gs://{sync.bucket} ({sync.bucket_provision_note})")
    print(f"  Manifest Path    : {MANIFEST_PATH} (existing verified={prior_count})")
    print(f"  TTS Engine/Voice : {args.tts_engine} / {VOICES.get(args.voice, args.voice)}")
    print(
        f"  Elastic Governor : tier={init_allowance.tier} avail={init_allowance.avail_gib:.2f} GiB "
        f"(floor={init_allowance.effective_floor_gib:.2f} GiB) cpu={init_allowance.cpu_pct:.1f}% "
        f"concurrency={init_allowance.concurrency} rss={init_allowance.process_rss_mib:.1f} MiB"
    )

    stats = {
        "synthesized_ok": 0,
        "failed": 0,
        "total_mp3_bytes_in_cloud": 0,
        "total_raw_wav_equiv_bytes": 0,
        "total_motif_bytes": 0,
        "total_vocoder_bytes": 0,
        "total_duration_s": 0.0,
        "governor_tiers_observed": {},
    }
    t_start = time.perf_counter()

    if limit != 0:
        queue = collect_curated_utterances(
            already_synced_sha256=seen_hashes,
            limit=limit,
            min_tier=args.min_tier,
            max_tier=args.max_tier,
        )
        print(f"  Curated Queue    : {len(queue)} utterances queued (max_tier={args.max_tier})")

        manifest_lock = asyncio.Lock()
        cursor = 0
        total_q = len(queue)

        with open(MANIFEST_PATH, "a", encoding="utf-8") as mf:
            while cursor < total_q:
                allowance = governor.sample()
                stats["governor_tiers_observed"][allowance.tier] = (
                    stats["governor_tiers_observed"].get(allowance.tier, 0) + 1
                )
                step_n = min(allowance.concurrency, total_q - cursor)
                batch_items = queue[cursor: cursor + step_n]
                cursor += step_n

                results = await asyncio.gather(
                    *(
                        process_one_utterance(
                            it,
                            sync=sync,
                            stage_dir=TEMP_STAGE_ROOT,
                            tts_engine=args.tts_engine,
                            voice_key=args.voice,
                        )
                        for it in batch_items
                    ),
                    return_exceptions=True,
                )

                async with manifest_lock:
                    for it, res in zip(batch_items, results):
                        if isinstance(res, Exception):
                            stats["failed"] += 1
                            print(f"  [WARN] Utterance {it.text_sha256[:10]} failed: {res}", file=sys.stderr)
                            continue
                        if not (res["verified_in_cloud"] and res["deleted_locally"]):
                            stats["failed"] += 1
                            continue
                        mf.write(json.dumps(res, ensure_ascii=False, separators=(",", ":")) + "\n")
                        mf.flush()
                        stats["synthesized_ok"] += 1
                        stats["total_mp3_bytes_in_cloud"] += int(res["byte_size"])
                        econ = res["compact_audio_atoms"]["economy"]
                        stats["total_raw_wav_equiv_bytes"] += int(econ["raw_16k_wav_bytes"])
                        stats["total_motif_bytes"] += int(econ["motif_bytes"])
                        stats["total_vocoder_bytes"] += int(econ["vocoder_compressed_bytes"])
                        stats["total_duration_s"] += float(res["duration_s"])

                if cursor % 20 == 0 or cursor == total_q:
                    leftover_now = MultimodalCloudSync.count_local_media_files(TEMP_STAGE_ROOT)
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"  [{cursor}/{total_q}] ok={stats['synthesized_ok']} fail={stats['failed']} "
                        f"cloud_mp3={stats['total_mp3_bytes_in_cloud']/1024:.1f} KiB "
                        f"local_temp_files={leftover_now} tier={allowance.tier} "
                        f"(avail={allowance.avail_gib:.2f} GiB, conc={allowance.concurrency}, rss={allowance.process_rss_mib:.1f} MiB) "
                        f"elapsed={elapsed:.1f}s"
                    )

                if allowance.pace_sleep_s > 0:
                    await asyncio.sleep(allowance.pace_sleep_s)

    # Clean up empty staging directory if completely empty
    leftover_files = MultimodalCloudSync.count_local_media_files(TEMP_STAGE_ROOT)
    if leftover_files == 0 and TEMP_STAGE_ROOT.exists():
        try:
            TEMP_STAGE_ROOT.rmdir()
        except OSError:
            pass

    # Independent Cloud Sample Verification Audit
    cloud_audit_passed = 0
    cloud_audit_checked = 0
    manifest_records: list[dict[str, Any]] = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    manifest_records.append(json.loads(line))

    if args.verify_cloud_sample > 0 and manifest_records:
        sample_recs = manifest_records[-min(args.verify_cloud_sample, len(manifest_records)):]
        for rec in sample_recs:
            cloud_audit_checked += 1
            chk = sync.verify_remote_object(
                rec["cloud_uri"],
                expected_size=rec["byte_size"],
                expected_sha256=rec["audio_sha256"],
            )
            if chk.get("verified"):
                cloud_audit_passed += 1

    # Reconstruction & Live W1z4rD Co-Firing Probe
    cofire_results: list[dict[str, Any]] = []
    reconstruct_verified = 0
    if args.cofire_sample > 0 and manifest_records:
        probe_recs = manifest_records[:min(args.cofire_sample, len(manifest_records))]
        for rec in probe_recs:
            wav_b = reconstruct_wav_bytes(rec["compact_audio_atoms"])
            if wav_b[:4] == b"RIFF" and len(wav_b) > 44:
                reconstruct_verified += 1
            cf = cofire_record_into_wizard(rec, node_url=args.node_url, timeout_s=args.wizard_timeout)
            cofire_results.append({
                "id": rec["id"],
                "text": rec["text"][:60],
                "reconstructed_wav_bytes": len(wav_b),
                **cf,
            })
            # If node is completely unreachable/timed out on first 2 tries, don't stall!
            if len(cofire_results) >= 2 and all(
                (not r["binding_ok"] and not r["sensor_ok"]) for r in cofire_results
            ):
                break

    governor.trim_working_set(force=True)
    final_allowance = governor.sample()

    summary = {
        "cloud_bucket": f"gs://{sync.bucket}",
        "cloud_audio_prefix": f"gs://{sync.bucket}/{CLOUD_AUDIO_PREFIX}/",
        "manifest_path": str(MANIFEST_PATH),
        "total_manifest_records": len(manifest_records),
        "run_synthesized_ok": stats["synthesized_ok"],
        "run_failed": stats["failed"],
        "run_duration_audio_s": round(stats["total_duration_s"], 2),
        "run_cloud_mp3_bytes": stats["total_mp3_bytes_in_cloud"],
        "local_temp_audio_files_remaining": leftover_files,
        "neuron_economy": {
            "raw_16k_wav_equiv_bytes": stats["total_raw_wav_equiv_bytes"],
            "pool_audio_motif_bytes": stats["total_motif_bytes"],
            "reconstructible_vocoder_bytes": stats["total_vocoder_bytes"],
            "motif_vs_wav_reduction_x": round(
                stats["total_raw_wav_equiv_bytes"] / max(1, stats["total_motif_bytes"]), 2
            ) if stats["total_motif_bytes"] else 0.0,
            "vocoder_vs_wav_reduction_x": round(
                stats["total_raw_wav_equiv_bytes"] / max(1, stats["total_vocoder_bytes"]), 2
            ) if stats["total_vocoder_bytes"] else 0.0,
        },
        "cloud_audit": {
            "checked": cloud_audit_checked,
            "passed": cloud_audit_passed,
        },
        "reconstruction_and_cofire": {
            "reconstruct_verified": reconstruct_verified,
            "cofire_attempts": len(cofire_results),
            "cofire_samples": cofire_results[:5],
        },
        "elastic_governor": {
            "tiers_observed": stats["governor_tiers_observed"],
            "working_set_trims": governor.trim_count,
            "final_state": final_allowance.to_dict(),
        },
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Mirror compact checkpoint summary to cloud bucket too
    sync.upload_json_checkpoint("english_speech/v1/_manifest_summary.json", summary)

    print("\n" + "=" * 78)
    print("  RUN SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    return 0 if leftover_files == 0 else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Synthesize English training corpus speech, decompose into canonical atoms, sync to GCS, and delete local audio immediately."
    )
    ap.add_argument("--limit", type=int, default=160, help="Number of utterances to synthesize in this run (0 = audit/cofire only)")
    ap.add_argument("--all", action="store_true", help="Process the entire English corpus continuously")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip utterances already verified in manifest by text_sha256")
    ap.add_argument("--min-tier", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Minimum curriculum tier to include")
    ap.add_argument("--max-tier", type=int, default=3, choices=[0, 1, 2, 3, 4], help="Maximum curriculum tier to include (0=greetings/anchors .. 4=full SimpleWiki/library)")
    ap.add_argument("--tts-engine", default="edge", choices=["edge", "polly"], help="C4rr13rX TTS engine: free neural edge-tts (default) or AWS Polly")
    ap.add_argument("--voice", default="en", help="Voice key: en (ChristopherNeural), en-female (JennyNeural), polly-ruth, polly-matthew")
    ap.add_argument("--provider", default="gcs", choices=["gcs", "s3"])
    ap.add_argument("--bucket", default="", help="Override cloud bucket name (defaults to dedicated w1z4rdv1510n-multimodal-corpus)")
    ap.add_argument("--min-free-gib", type=float, default=8.0, help="Untouched system free RAM floor in GiB (+12%% safety margin applied automatically)")
    ap.add_argument("--abundant-free-gib", type=float, default=14.5, help="Free RAM threshold in GiB for scaling up to max concurrency")
    ap.add_argument("--max-concurrency", type=int, default=4, help="Maximum parallel synthesis/upload workers when PC free RAM is abundant")
    ap.add_argument("--verify-cloud-sample", type=int, default=20, help="Number of manifest entries to re-verify via fresh GCS HEAD/MD5 after run")
    ap.add_argument("--cofire-sample", type=int, default=8, help="Number of utterances to reconstruct & co-fire into W1z4rD (:8090) if responsive")
    ap.add_argument("--node-url", default="http://127.0.0.1:8090", help="W1z4rDV1510n HTTP endpoint")
    ap.add_argument("--wizard-timeout", type=float, default=4.0, help="Non-blocking per-request timeout (s) for W1z4rD co-firing")
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_pipeline(parse_args())))
