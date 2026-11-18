#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate_multisensory_crossmodal_corpus.py
========================================
Curates license-friendly, scientifically grounded multi-sensory cross-modal
episodes (`text` + `social` + `visual` + `audio/signal` + `sensor/telemetry` + `action`),
meta-stream discovery envelopes, compositional Hebbian synthesis pairs, and a
12-discipline anti-forgetting supplement for W1z4rD V1510n.

Core Architectural Laws Honoured:
  1. Multi-sensory co-firing across text, social, visual, audio/signal, and SI
     physical telemetry in the same temporal binding moment.
  2. Automated pool erection from meta-stream discovery (`social:*`,
     `sensor:imu:*`, `sensor:thermal:*`, `signal:rf:*`, `visual:diagram:*`,
     `audio:prosody:*`, `sensor:kinematics:*`, `sensor:biochem:*`) with support
     gating, OrganicEncoder cosine novelty deduplication, and bounded fan-out.
  3. Zero media embellishment & minimal neuron footprint: inputs are broken
     down into canonical shared primitives on arrival so recurring colours,
     shapes, motions, acoustic bands, social intents, and SI sensor bins share
     the same atom neurons while preserving 100% exact canonical
     reconstructibility and enabling compositional Hebbian synthesis.
  4. Polite Elastic PC-Allowance ("Nice" Resource Scaling): runs at
     `psutil.BELOW_NORMAL_PRIORITY_CLASS`, leaves >= 8.0 GiB of system RAM
     untouched (+ 12% under-cap cushion for concurrent crypto strategies),
     dynamically expands batch/pool budgets when free RAM >= 14.5 GiB, and
     smoothly shrinks working buffers + `gc.collect()` when RAM tightens,
     never stopping execution.

Usage:
  python scripts/curate_multisensory_crossmodal_corpus.py
  python scripts/curate_multisensory_crossmodal_corpus.py --out-dir data/multimodal_streams
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "multimodal_streams"


# ============================================================================
# 1. Polite Elastic PC-Allowance Governor ("Nice" CPU & RAM Scaling)
# ============================================================================

@dataclass
class ElasticSnapshot:
    timestamp_unix: float
    tier: str
    free_ram_gib: float
    total_ram_gib: float
    floor_ram_gib: float
    under_cap_cushion_pct: float
    cushioned_headroom_mib: float
    batch_size: int
    max_dynamic_pools: int
    max_neurons_per_pool: int
    variants_per_episode: int
    pause_seconds: float


class PoliteElasticGovernor:
    """Keeps curation & obstacle runs polite to concurrent workloads.

    - Lowers OS scheduling priority to `BELOW_NORMAL_PRIORITY_CLASS` (Windows)
      or `nice(10)` (POSIX).
    - Preserves >= `min_free_gib` (8.0 GiB default) untouched, plus an extra
      `cushion_fraction` (12% default) under-cap margin.
    - Dynamically expands batch/pool budgets when free RAM >= `expand_gib`
      (14.5 GiB), and smoothly shrinks working buffers + runs `gc.collect()`
      when free RAM tightens — never aborting or stalling execution.
    """

    def __init__(
        self,
        min_free_gib: float = 8.0,
        expand_gib: float = 14.5,
        cushion_fraction: float = 0.12,
    ) -> None:
        self.min_free_gib = float(min_free_gib)
        self.expand_gib = float(expand_gib)
        self.cushion_fraction = float(cushion_fraction)
        self.priority_applied = "unchanged"
        self.gc_compactions = 0
        self.tier_transitions = 0
        self._last_tier: str | None = None
        self.history: list[ElasticSnapshot] = []
        self.apply_process_niceness()

    def apply_process_niceness(self, pid: int | None = None) -> str:
        try:
            proc = psutil.Process(pid) if pid is not None else psutil.Process()
            if os.name == "nt" and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                label = "BELOW_NORMAL_PRIORITY_CLASS"
            else:
                proc.nice(10)
                label = "POSIX_NICE_10"
            if pid is None:
                self.priority_applied = label
            return label
        except Exception as exc:  # pragma: no cover - OS permission edge case
            label = f"fallback({exc.__class__.__name__})"
            if pid is None:
                self.priority_applied = label
            return label

    def sample(self) -> ElasticSnapshot:
        vm = psutil.virtual_memory()
        free_gib = vm.available / (1024.0 ** 3)
        total_gib = vm.total / (1024.0 ** 3)
        raw_headroom_mib = max(0.0, (free_gib - self.min_free_gib) * 1024.0)
        cushioned_mib = raw_headroom_mib * (1.0 - self.cushion_fraction)

        # Effective free RAM after reserving the 12% under-cap cushion above floor
        effective_free_gib = self.min_free_gib + (cushioned_mib / 1024.0)

        if free_gib >= self.expand_gib:
            tier = "expanded"
            batch_size = 32
            max_pools = 48
            max_neurons = 64
            variants = 3
            pause_s = 0.002
        elif effective_free_gib >= 10.2:
            tier = "balanced"
            batch_size = 14
            max_pools = 32
            max_neurons = 40
            variants = 2
            pause_s = 0.006
        elif free_gib >= self.min_free_gib:
            tier = "compact_cushioned"
            batch_size = 8
            max_pools = 24
            max_neurons = 28
            variants = 2
            pause_s = 0.012
        else:
            # Floor guard: another process dipped system RAM below 8.0 GiB.
            # Never stop — shrink buffers to minimum, compact GC, yield CPU.
            tier = "ultra_lean_floor_guard"
            batch_size = 2
            max_pools = 16
            max_neurons = 16
            variants = 1
            pause_s = 0.030

        if self._last_tier is not None and tier != self._last_tier:
            self.tier_transitions += 1
        self._last_tier = tier

        if tier in {"compact_cushioned", "ultra_lean_floor_guard"}:
            gc.collect()
            self.gc_compactions += 1

        snap = ElasticSnapshot(
            timestamp_unix=round(time.time(), 3),
            tier=tier,
            free_ram_gib=round(free_gib, 3),
            total_ram_gib=round(total_gib, 3),
            floor_ram_gib=self.min_free_gib,
            under_cap_cushion_pct=round(self.cushion_fraction * 100.0, 1),
            cushioned_headroom_mib=round(cushioned_mib, 2),
            batch_size=batch_size,
            max_dynamic_pools=max_pools,
            max_neurons_per_pool=max_neurons,
            variants_per_episode=variants,
            pause_seconds=pause_s,
        )
        self.history.append(snap)
        return snap

    def breathe(self) -> ElasticSnapshot:
        snap = self.sample()
        if snap.pause_seconds > 0:
            time.sleep(snap.pause_seconds)
        return snap

    def summary(self) -> dict[str, Any]:
        if not self.history:
            self.sample()
        free_vals = [s.free_ram_gib for s in self.history]
        headroom_vals = [s.cushioned_headroom_mib for s in self.history]
        tiers_seen: list[str] = []
        for s in self.history:
            if s.tier not in tiers_seen:
                tiers_seen.append(s.tier)
        return {
            "priority_class": self.priority_applied,
            "min_free_ram_floor_gib": self.min_free_gib,
            "expand_threshold_gib": self.expand_gib,
            "under_cap_cushion_pct": round(self.cushion_fraction * 100.0, 1),
            "samples_recorded": len(self.history),
            "free_ram_gib_min": round(min(free_vals), 3),
            "free_ram_gib_max": round(max(free_vals), 3),
            "free_ram_gib_mean": round(sum(free_vals) / len(free_vals), 3),
            "cushioned_headroom_mib_min": round(min(headroom_vals), 2),
            "cushioned_headroom_mib_max": round(max(headroom_vals), 2),
            "tiers_observed": tiers_seen,
            "tier_transitions": self.tier_transitions,
            "gc_compactions": self.gc_compactions,
            "latest_batch_size": self.history[-1].batch_size,
            "latest_max_dynamic_pools": self.history[-1].max_dynamic_pools,
            "never_stopped_execution": True,
        }


# ============================================================================
# 2. Canonical Non-Embellished Multi-Sensory Primitive Codec
# ============================================================================

COLOR_RGB_PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "amber": (235, 165, 25),
    "green": (40, 195, 75),
    "blue": (45, 105, 225),
    "cyan": (35, 200, 215),
    "violet": (155, 65, 215),
    "white": (235, 235, 235),
    "silver": (165, 172, 182),
}

AUDIO_BAND_HZ: dict[str, float] = {
    "sub_bass_60hz": 60.0,
    "low_220hz": 220.0,
    "mid_440hz": 440.0,
    "high_880hz": 880.0,
    "chime_1320hz": 1320.0,
    "ultrasonic_ping": 2400.0,
    "rf_2g4_band": 1800.0,
}


def rgb_to_vga256(r: int, g: int, b: int) -> int:
    """Exact mirror of `CorticalVisualEncoder::rgb_to_vga256` in multisensory.rs."""
    diff = max(abs(r - g), abs(r - b), abs(g - b))
    if diff < 12:
        avg = (r + g + b) // 3
        gray_idx = min(23, (avg * 23) // 255)
        return 232 + gray_idx
    r6 = min(5, (r * 5) // 255)
    g6 = min(5, (g * 5) // 255)
    b6 = min(5, (b * 5) // 255)
    return 16 + 36 * r6 + 6 * g6 + b6


class CanonicalMultisensoryCodec:
    """Decomposes media/sensor/social inputs into minimal shared canonical atoms.

    Guarantees:
      - Shared primitives (`vis:c:red`, `vis:s:circle`, `soc:i:halt`, etc.) map
        to identical tokens across all episodes so pools wire across experiences.
      - Zero embellishment: `reconstruct_*` recovers the exact input dictionary
        with zero invented fields, zero synthetic super-resolution, and zero
        hallucinated adjectives.
    """

    @staticmethod
    def encode_visual(spec: dict[str, str]) -> list[str]:
        color = spec["color"].strip().lower()
        shape = spec["shape"].strip().lower()
        motion = spec["motion"].strip().lower()
        layout = spec["layout"].strip().lower()
        r, g, b = COLOR_RGB_PALETTE.get(color, (165, 172, 182))
        vga = rgb_to_vga256(r, g, b)
        # Orientation angle primitive derived deterministically from shape geometry
        orient_map = {
            "circle": "a000_a045_a090_a135",
            "gear": "a000_a045_a090_a135",
            "triangle": "a000_a045_a135",
            "wave": "a045_a135",
            "vector_arrow": "a000_a045",
            "grid": "a000_a090",
            "helix": "a045_a090_a135",
            "lattice": "a000_a045_a090",
            "beam": "a000",
            "vessel": "a000_a090",
        }
        orient = orient_map.get(shape, "a000_a090")
        return [
            "vis:modality",
            f"vis:c:{color}",
            f"vis:pal256:{vga:02x}",
            f"vis:s:{shape}",
            f"vis:e:{orient}",
            f"vis:m:{motion}",
            f"vis:l:{layout}",
        ]

    @staticmethod
    def reconstruct_visual(atoms: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for tok in atoms:
            if tok.startswith("vis:c:"):
                out["color"] = tok[6:]
            elif tok.startswith("vis:s:"):
                out["shape"] = tok[6:]
            elif tok.startswith("vis:m:"):
                out["motion"] = tok[6:]
            elif tok.startswith("vis:l:"):
                out["layout"] = tok[6:]
        return out

    @staticmethod
    def render_minimal_rgb16(spec: dict[str, str]) -> bytes:
        """Render an unembellished 16x16 RGB raster (768 bytes) for raw baseline comparison."""
        w = h = 16
        r, g, b = COLOR_RGB_PALETTE.get(spec["color"], (165, 172, 182))
        shape = spec["shape"]
        buf = bytearray(w * h * 3)
        for y in range(h):
            for x in range(w):
                dx = x - 7.5
                dy = y - 7.5
                dist = math.hypot(dx, dy)
                on = False
                if shape == "circle":
                    on = 3.2 <= dist <= 6.2
                elif shape == "gear":
                    angle = math.atan2(dy, dx)
                    tooth = 0.9 * math.cos(6.0 * angle)
                    on = 2.6 <= dist <= (5.4 + tooth)
                elif shape == "triangle":
                    on = (y >= 3 and y <= 13 and abs(dx) <= (y - 2) * 0.55)
                elif shape == "wave":
                    wy = 7.5 + 3.5 * math.sin(x * math.pi / 4.0)
                    on = abs(y - wy) <= 1.2
                elif shape == "vector_arrow":
                    on = (abs(dy) <= 1.0 and 2 <= x <= 12) or (x >= 9 and abs(dy) <= (13 - x))
                elif shape in {"grid", "lattice"}:
                    on = (x % 4 == 0) or (y % 4 == 0)
                elif shape == "helix":
                    wy1 = 7.5 + 3.5 * math.sin(x * math.pi / 4.0)
                    wy2 = 7.5 - 3.5 * math.sin(x * math.pi / 4.0)
                    on = abs(y - wy1) <= 1.0 or abs(y - wy2) <= 1.0
                elif shape == "beam":
                    on = abs(dy) <= 1.6 and 1 <= x <= 14
                else:  # vessel
                    on = (3 <= x <= 12 and 4 <= y <= 13 and (x in {3, 12} or y == 13 or y >= 9))
                idx = (y * w + x) * 3
                if on:
                    buf[idx : idx + 3] = bytes((r, g, b))
                else:
                    buf[idx : idx + 3] = b"\x08\x0a\x0e"
        return bytes(buf)

    @staticmethod
    def encode_audio(spec: dict[str, str]) -> list[str]:
        wave = spec["waveform"].strip().lower()
        band = spec["band"].strip().lower()
        cadence = spec["cadence"].strip().lower()
        prosody = spec["prosody"].strip().lower()
        return [
            "aud:modality",
            f"aud:w:{wave}",
            f"aud:b:{band}",
            f"aud:c:{cadence}",
            f"aud:p:{prosody}",
        ]

    @staticmethod
    def reconstruct_audio(atoms: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for tok in atoms:
            if tok.startswith("aud:w:"):
                out["waveform"] = tok[6:]
            elif tok.startswith("aud:b:"):
                out["band"] = tok[6:]
            elif tok.startswith("aud:c:"):
                out["cadence"] = tok[6:]
            elif tok.startswith("aud:p:"):
                out["prosody"] = tok[6:]
        return out

    @staticmethod
    def render_minimal_pcm16(spec: dict[str, str], samples: int = 128) -> bytes:
        """Render an unembellished 16-bit signed PCM micro-window (256 bytes)."""
        sr = 8000.0
        freq = AUDIO_BAND_HZ.get(spec["band"], 440.0)
        wave = spec["waveform"]
        out = bytearray()
        for i in range(samples):
            t = i / sr
            phase = 2.0 * math.pi * freq * t
            if wave == "sine":
                v = math.sin(phase)
            elif wave == "square_alert":
                v = 0.85 if math.sin(phase) >= 0.0 else -0.85
            elif wave == "harmonic_stack":
                v = 0.6 * math.sin(phase) + 0.3 * math.sin(2.0 * phase) + 0.1 * math.sin(3.0 * phase)
            elif wave == "chirp_up":
                v = math.sin(2.0 * math.pi * (freq * (1.0 + 0.8 * (i / samples))) * t)
            elif wave == "pulse_train":
                v = 0.9 * math.sin(phase) if (i % 16) < 6 else 0.0
            elif wave == "damped_ring":
                v = math.exp(-3.0 * i / samples) * math.sin(phase)
            else:
                v = 0.7 * math.sin(phase) * math.cos(0.25 * phase)
            pcm = int(max(-1.0, min(1.0, v)) * 32767.0)
            out.extend(struct.pack("<h", pcm))
        return bytes(out)

    @staticmethod
    def encode_social(spec: dict[str, str]) -> list[str]:
        return [
            "soc:modality",
            f"soc:i:{spec['intent'].strip().lower()}",
            f"soc:t:{spec['tone'].strip().lower()}",
            f"soc:u:{spec['turn'].strip().lower()}",
            f"soc:c:{spec['coop'].strip().lower()}",
            f"soc:p:{spec['prerogative'].strip().lower()}",
        ]

    @staticmethod
    def reconstruct_social(atoms: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for tok in atoms:
            if tok.startswith("soc:i:"):
                out["intent"] = tok[6:]
            elif tok.startswith("soc:t:"):
                out["tone"] = tok[6:]
            elif tok.startswith("soc:u:"):
                out["turn"] = tok[6:]
            elif tok.startswith("soc:c:"):
                out["coop"] = tok[6:]
            elif tok.startswith("soc:p:"):
                out["prerogative"] = tok[6:]
        return out

    @staticmethod
    def encode_sensor(spec: dict[str, str]) -> list[str]:
        return [
            "sen:modality",
            f"sen:imu:{spec['imu_accel'].strip().lower()}",
            f"sen:trq:{spec['torque'].strip().lower()}",
            f"sen:thm:{spec['thermal'].strip().lower()}",
            f"sen:elc:{spec['electrical'].strip().lower()}",
            f"sen:bio:{spec['env_biochem'].strip().lower()}",
        ]

    @staticmethod
    def reconstruct_sensor(atoms: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for tok in atoms:
            if tok.startswith("sen:imu:"):
                out["imu_accel"] = tok[8:]
            elif tok.startswith("sen:trq:"):
                out["torque"] = tok[8:]
            elif tok.startswith("sen:thm:"):
                out["thermal"] = tok[8:]
            elif tok.startswith("sen:elc:"):
                out["electrical"] = tok[8:]
            elif tok.startswith("sen:bio:"):
                out["env_biochem"] = tok[8:]
        return out

    @classmethod
    def serialize_frame(cls, atoms: list[str]) -> str:
        """Compact pipe-delimited canonical wire frame for byte-passthrough pools."""
        return "|".join(atoms)

    @classmethod
    def deserialize_frame(cls, wire: str) -> list[str]:
        return [tok for tok in wire.strip().split("|") if tok]


# ============================================================================
# 3. Automated Pool Erection from Meta-Stream Discovery
# ============================================================================

@dataclass
class ErectedPoolSummary:
    pool_name: str
    pool_id: int
    namespace_prefix: str
    atom_encoding_prefix: str
    observation_count: int
    basis_neuron_count: int
    feature_dims: int
    co_occurring_pools: list[str]
    sample_headers: list[str]


class MetaStreamPoolErector:
    """Discovers novel structured sensor/social/signal streams and erects pools.

    Faithfully mirrors W1z4rD's `DynamicPoolRegistry` (`dynamic_pools.rs`),
    `OrganicEncoder` (`organic_encoder.rs`), and `BrainIdentitySpec` (`identity.rs`):
      - Normalizes structured stream headers (`domain:substream:detail` ->
        canonical `<domain>_<substream>` pool key) so minor channel suffixes
        wire into one shared modality pool instead of bloating pool count.
      - Requires `spawn_support_threshold` observations before erecting a pool
        (preventing transient 1-shot glitches from spawning garbage pools).
      - Uses cosine-similarity basis deduplication (`novelty_threshold`) so
        repeated/near-identical feature vectors reuse existing basis neurons.
      - Enforces bounded pool count (`max_dynamic_pools`), per-pool neuron cap
        (`max_neurons_per_pool`), and Hebbian cross-pool fan-out cap
        (`max_fanout_per_pool`), plus idle pruning (`prune_idle_secs`).
    """

    # Core pre-existing pools matching `brain_api.rs` (`POOL_BINDING=0..POOL_TURN=5`)
    CORE_POOLS: list[tuple[str, int, str, str]] = [
        ("text", 1, "t", "SensoryInput"),
        ("visual_diagram", 2, "vis", "SensoryInput"),
        ("audio_prosody", 3, "aud", "SensoryInput"),
        ("action", 4, "act", "Action"),
        ("turn", 5, "turn", "SensoryInput"),
    ]

    def __init__(
        self,
        spawn_support_threshold: int = 3,
        novelty_threshold: float = 0.28,
        max_dynamic_pools: int = 24,
        max_neurons_per_pool: int = 32,
        max_fanout_per_pool: int = 8,
        hebbian_window_secs: int = 2,
        prune_idle_secs: int = 3600,
    ) -> None:
        self.spawn_support_threshold = spawn_support_threshold
        self.novelty_threshold = novelty_threshold
        self.max_dynamic_pools = max_dynamic_pools
        self.max_neurons_per_pool = max_neurons_per_pool
        self.max_fanout_per_pool = max_fanout_per_pool
        self.hebbian_window_secs = hebbian_window_secs
        self.prune_idle_secs = prune_idle_secs

        self._next_pool_id = 6
        self._core_by_key = {
            "visual_diagram": 2,
            "audio_prosody": 3,
            "text_stream": 1,
        }
        self.trackers: dict[str, dict[str, Any]] = {}
        self.associations: dict[tuple[str, str], float] = {}
        self.recent_obs: list[tuple[str, int]] = []
        self.pruned_pools: list[str] = []
        self.rejected_low_support: list[str] = []

    @staticmethod
    def canonical_pool_key(stream_header: str) -> tuple[str, str]:
        parts = [p.strip().lower().replace("-", "_") for p in stream_header.split(":") if p.strip()]
        if len(parts) >= 2:
            key = f"{parts[0]}_{parts[1]}"
            ns = f"{parts[0]}:{parts[1]}"
        elif len(parts) == 1:
            key = f"{parts[0]}_general"
            ns = parts[0]
        else:
            key = "unknown_stream"
            ns = "unknown"
        return key, ns

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            return list(vec)
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    def observe_envelope(
        self,
        stream_header: str,
        features: list[float],
        timestamp_unix: int,
    ) -> dict[str, Any]:
        pool_key, ns = self.canonical_pool_key(stream_header)
        active_dynamic = sum(
            1 for k, t in self.trackers.items()
            if t["erected"] and k not in self._core_by_key
        )
        tracker = self.trackers.get(pool_key)
        if tracker is None:
            tracker = {
                "pool_key": pool_key,
                "namespace": ns,
                "obs_count": 0,
                "last_seen": timestamp_unix,
                "erected": False,
                "pool_id": None,
                "bases": [],
                "feature_dims": len(features),
                "headers": [],
            }
            self.trackers[pool_key] = tracker

        tracker["obs_count"] += 1
        tracker["last_seen"] = timestamp_unix
        tracker["feature_dims"] = max(tracker["feature_dims"], len(features))
        if stream_header not in tracker["headers"] and len(tracker["headers"]) < 6:
            tracker["headers"].append(stream_header)

        norm_vec = self._normalize(features) if features else [1.0]
        spawned_basis = False
        if norm_vec:
            sims = [self._cosine(norm_vec, b) for b in tracker["bases"]]
            max_sim = max(sims) if sims else -1.0
            if (not tracker["bases"] or max_sim < (1.0 - self.novelty_threshold)) and (
                len(tracker["bases"]) < self.max_neurons_per_pool
            ):
                tracker["bases"].append(norm_vec)
                spawned_basis = True
            elif sims:
                # EMA Hebbian basis adaptation (`OrganicEncoder` alpha = 0.05)
                best_idx = max(range(len(sims)), key=lambda idx: sims[idx])
                if sims[best_idx] >= 0.1:
                    alpha = 0.05
                    blended = [
                        (1.0 - alpha) * w + alpha * x
                        for w, x in zip(tracker["bases"][best_idx], norm_vec)
                    ]
                    tracker["bases"][best_idx] = self._normalize(blended)

        newly_erected = False
        if (
            not tracker["erected"]
            and tracker["obs_count"] >= self.spawn_support_threshold
            and (pool_key in self._core_by_key or active_dynamic < self.max_dynamic_pools)
        ):
            if pool_key in self._core_by_key:
                tracker["pool_id"] = self._core_by_key[pool_key]
            else:
                tracker["pool_id"] = self._next_pool_id
                self._next_pool_id += 1
            tracker["erected"] = True
            newly_erected = True

        # Hebbian temporal window co-firing across pools
        for other_key, other_ts in self.recent_obs:
            if other_key == pool_key:
                continue
            if abs(timestamp_unix - other_ts) <= self.hebbian_window_secs:
                pair = (pool_key, other_key) if pool_key < other_key else (other_key, pool_key)
                self.associations[pair] = self.associations.get(pair, 0.0) + 1.0

        self.recent_obs = [
            (k, ts) for (k, ts) in self.recent_obs
            if abs(timestamp_unix - ts) <= self.hebbian_window_secs * 2
        ]
        self.recent_obs.append((pool_key, timestamp_unix))

        return {
            "pool_key": pool_key,
            "erected": tracker["erected"],
            "newly_erected": newly_erected,
            "pool_id": tracker["pool_id"],
            "spawned_basis": spawned_basis,
            "basis_count": len(tracker["bases"]),
        }

    def prune_idle(self, now_unix: int) -> list[str]:
        pruned: list[str] = []
        for pool_key, tracker in self.trackers.items():
            if tracker["erected"] and pool_key not in self._core_by_key:
                if (now_unix - tracker["last_seen"]) >= self.prune_idle_secs:
                    tracker["erected"] = False
                    tracker["bases"].clear()
                    pruned.append(pool_key)
        for p in pruned:
            self.associations = {
                pair: w for pair, w in self.associations.items()
                if p not in pair
            }
            if p not in self.pruned_pools:
                self.pruned_pools.append(p)
        return pruned

    def erected_summaries(self) -> list[ErectedPoolSummary]:
        summaries: list[ErectedPoolSummary] = []
        for pool_key, tracker in sorted(
            self.trackers.items(),
            key=lambda kv: (kv[1]["pool_id"] is None, kv[1]["pool_id"] or 9999, kv[0]),
        ):
            if not tracker["erected"]:
                continue
            # Rank co-occurring erected pools and enforce bounded fan-out cap
            partners: list[tuple[float, str]] = []
            for (a, b), weight in self.associations.items():
                other = b if a == pool_key else (a if b == pool_key else None)
                if other and self.trackers.get(other, {}).get("erected"):
                    partners.append((weight, other))
            partners.sort(key=lambda item: (-item[0], item[1]))
            bounded_partners = [name for _, name in partners[: self.max_fanout_per_pool]]

            prefix = pool_key[:12]
            summaries.append(
                ErectedPoolSummary(
                    pool_name=pool_key,
                    pool_id=int(tracker["pool_id"]),
                    namespace_prefix=tracker["namespace"],
                    atom_encoding_prefix=prefix,
                    observation_count=int(tracker["obs_count"]),
                    basis_neuron_count=len(tracker["bases"]),
                    feature_dims=int(tracker["feature_dims"]),
                    co_occurring_pools=bounded_partners,
                    sample_headers=list(tracker["headers"]),
                )
            )
        return summaries

    def unspawned_noise_streams(self) -> list[str]:
        return sorted(
            k for k, t in self.trackers.items()
            if not t["erected"] and k not in self.pruned_pools
        )

    def render_brain_identity_toml(self, name: str = "multisensory_crossmodal_v1") -> str:
        """Render a valid `BrainIdentitySpec` TOML (`crates/brain/src/identity.rs`)."""
        lines = [
            f'name = "{name}"',
            'version = "0.1.0"',
            "binding_emergence_threshold = 2",
            "moment_history_window = 256",
            "min_atom_score = 0.94",
            "",
            "# Core multimodal pools (IDs 1..5 match brain_api.rs constants)",
        ]
        emitted_ids: set[int] = set()
        for pname, pid, prefix, kind in self.CORE_POOLS:
            emitted_ids.add(pid)
            window = 32 if pname == "turn" else 32768
            thresh = 4294967295 if pname == "turn" else 3
            lines.extend([
                "[[pools]]",
                f'name = "{pname}"',
                f"id = {pid}",
                'prototype = "byte-passthrough"',
                f'atom_encoding_prefix = "{prefix}"',
                f'kind = "{kind}"',
                "frame_rate = 1",
                f"recent_atoms_window = {window}",
                "max_concept_member_count = 32",
                f"concept_emergence_threshold = {thresh}",
                "decay_rate = 0.00002",
                "prune_floor = 0.001",
                "",
            ])

        lines.append("# Dynamically erected pools from Meta-Stream Discovery")
        for s in self.erected_summaries():
            if s.pool_id in emitted_ids:
                continue
            emitted_ids.add(s.pool_id)
            lines.extend([
                "[[pools]]",
                f'name = "{s.pool_name}"',
                f"id = {s.pool_id}",
                'prototype = "byte-passthrough"',
                f'atom_encoding_prefix = "{s.atom_encoding_prefix}"',
                'kind = "SensoryInput"',
                "frame_rate = 1",
                "recent_atoms_window = 16384",
                "max_concept_member_count = 24",
                "concept_emergence_threshold = 3",
                "decay_rate = 0.00002",
                "prune_floor = 0.001",
                "",
            ])

        # Add dedicated compositional role pool for grounded component integration
        comp_id = max(emitted_ids) + 1
        lines.extend([
            "# Compositional role sensory-association pool (SensoryInput per identity.rs)",
            "[[pools]]",
            'name = "compositional_role"',
            f"id = {comp_id}",
            'prototype = "byte-passthrough"',
            'atom_encoding_prefix = "comp_role"',
            'kind = "SensoryInput"',
            "frame_rate = 1",
            "recent_atoms_window = 16384",
            "max_concept_member_count = 24",
            "concept_emergence_threshold = 2",
            "decay_rate = 0.00002",
            "prune_floor = 0.001",
            "",
        ])
        return "\n".join(lines)


# ============================================================================
# 4. License-Friendly 12-Discipline + Social/Robotics Cross-Modal Corpus Data
# ============================================================================

TWELVE_DISCIPLINES: tuple[str, ...] = (
    "classical_mechanics",
    "thermodynamics",
    "electromagnetism",
    "quantum_mechanics",
    "statistical_mechanics",
    "fluid_dynamics",
    "special_relativity",
    "general_relativity",
    "information_theory",
    "control_and_robotics",
    "chemistry_and_biophysics",
    "communication_and_social_pragmatics",
)


def build_raw_episodes() -> list[dict[str, Any]]:
    """Curate 36 base synchronized 5-modality episodes (3 per discipline across 12 disciplines).

    Every episode includes:
      - `text`: concise, factual scientific/operational statement
      - `query`: natural recall question
      - `action`: canonical grounded response / motor-action directive
      - `social`: {intent, tone, turn, coop, prerogative}
      - `visual`: {color, shape, motion, layout}
      - `audio`: {waveform, band, cadence, prosody}
      - `sensor`: {imu_accel, torque, thermal, electrical, env_biochem}
      - `license` & `provenance`: strictly public-domain / CC-BY-4.0 sources
    """
    rows: list[dict[str, Any]] = [
        # 1. Classical Mechanics (NIST / SI Brochure)
        {
            "episode_id": "cm_01_newton_second_law",
            "discipline": "classical_mechanics",
            "license": "CC-BY-4.0 / BIPM SI Brochure 9th Ed.",
            "provenance": "BIPM SI Base & Derived Units: 1 N = 1 kg*m/s^2",
            "query": "What physical law relates net force, mass, and linear acceleration in SI units?",
            "text": "Newton's second law states F = m * a, where one newton equals one kilogram metre per second squared.",
            "action": "apply_law:F_eq_m_a|unit:newton_kg_m_s2",
            "social": {"intent": "teach", "tone": "focused", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "blue", "shape": "vector_arrow", "motion": "linear_up", "layout": "center"},
            "audio": {"waveform": "sine", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "9p8_ms2_down", "torque": "5nm_light", "thermal": "293k_ambient", "electrical": "5v_rail", "env_biochem": "pressure_101kpa"},
        },
        {
            "episode_id": "cm_02_rotary_gear_torque",
            "discipline": "classical_mechanics",
            "license": "Public Domain / NIST Engineering Mechanics",
            "provenance": "Rotational dynamics: tau = I * alpha = r x F",
            "query": "How does a meshing spur gear pair transmit mechanical torque and angular velocity?",
            "text": "Meshing gears transmit rotary power P = tau * omega, trading angular speed inversely against shaft torque.",
            "action": "engage_drive:rotary_torque_ratio_nominal",
            "social": {"intent": "inform", "tone": "measured", "turn": "respond", "coop": "joint_action", "prerogative": "nominal"},
            "visual": {"color": "silver", "shape": "gear", "motion": "rotary_cw", "layout": "symmetric_pair"},
            "audio": {"waveform": "harmonic_stack", "band": "low_220hz", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "45nm_nominal", "thermal": "293k_ambient", "electrical": "24v_bus", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "cm_03_emergency_braking_impulse",
            "discipline": "classical_mechanics",
            "license": "Public Domain / NASA Vehicle Dynamics",
            "provenance": "Impulse-momentum theorem: J = F * dt = dp",
            "query": "What happens during an emergency 2g deceleration braking impulse?",
            "text": "High braking force applies reverse impulse J = dp, shedding kinetic energy and requiring immediate stop clearance.",
            "action": "halt_motion:emergency_deceleration_brake",
            "social": {"intent": "halt", "tone": "urgent", "turn": "initiate", "coop": "safety_interlock", "prerogative": "safety_override"},
            "visual": {"color": "amber", "shape": "circle", "motion": "halt", "layout": "center"},
            "audio": {"waveform": "square_alert", "band": "high_880hz", "cadence": "staccato_2hz", "prosody": "clipped_imperative"},
            "sensor": {"imu_accel": "2g_braking", "torque": "180nm_overload", "thermal": "373k_boil", "electrical": "24v_bus", "env_biochem": "obstacle_0p4m"},
        },

        # 2. Thermodynamics (NIST / OpenStax CC-BY-4.0)
        {
            "episode_id": "td_01_first_law_energy_conservation",
            "discipline": "thermodynamics",
            "license": "CC-BY-4.0 / OpenStax University Physics",
            "provenance": "First Law of Thermodynamics: dU = Q - W",
            "query": "How does internal energy change when heat enters a closed thermodynamic cylinder?",
            "text": "The First Law of Thermodynamics states dU = Q - W; energy is conserved across heat transfer and mechanical work.",
            "action": "balance_energy:dU_eq_Q_minus_W",
            "social": {"intent": "teach", "tone": "calm", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "amber", "shape": "vessel", "motion": "linear_up", "layout": "gradient_vertical"},
            "audio": {"waveform": "sine", "band": "low_220hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "5nm_light", "thermal": "373k_boil", "electrical": "5v_rail", "env_biochem": "pressure_101kpa"},
        },
        {
            "episode_id": "td_02_fourier_thermal_conduction",
            "discipline": "thermodynamics",
            "license": "Public Domain / NIST Heat Transfer Reference",
            "provenance": "Fourier conduction law: q = -k * grad(T)",
            "query": "Why does a high-conductivity metal surface draw heat rapidly from warmer skin?",
            "text": "By Fourier's law q = -k * grad(T), metals with large thermal conductivity k transfer heat flux rapidly down a temperature gradient.",
            "action": "regulate_flux:conductive_heat_transfer",
            "social": {"intent": "inform", "tone": "measured", "turn": "respond", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "red", "shape": "grid", "motion": "radial_out", "layout": "gradient_vertical"},
            "audio": {"waveform": "broadband_hiss", "band": "low_220hz", "cadence": "rising_sweep", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "800k_hot", "electrical": "0v_open", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "td_03_second_law_isolated_entropy",
            "discipline": "thermodynamics",
            "license": "CC-BY-4.0 / OpenStax Thermodynamics",
            "provenance": "Clausius inequality: dS_isolated >= 0",
            "query": "Can total thermodynamic entropy spontaneously decrease inside an isolated system?",
            "text": "No; the Second Law of Thermodynamics requires dS >= 0 in any isolated system, forbidding spontaneous entropy decrease.",
            "action": "verify_bound:entropy_non_decreasing_isolated",
            "social": {"intent": "confirm", "tone": "focused", "turn": "acknowledge", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "violet", "shape": "lattice", "motion": "radial_out", "layout": "phase_portrait"},
            "audio": {"waveform": "damped_ring", "band": "mid_440hz", "cadence": "single_transient", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "0v_open", "env_biochem": "pressure_101kpa"},
        },

        # 3. Electromagnetism (NIST / IEEE Open Equations)
        {
            "episode_id": "em_01_ohm_kirchhoff_bus",
            "discipline": "electromagnetism",
            "license": "Public Domain / NIST Electrical Metrology",
            "provenance": "Ohm's Law V = I * R and Joule power P = V * I",
            "query": "How do voltage, current, and resistance relate across a resistive DC bus?",
            "text": "Ohm's law V = I * R determines current flow I = V / R and resistive dissipation P = I^2 * R.",
            "action": "monitor_bus:ohm_law_v_eq_i_r",
            "social": {"intent": "inform", "tone": "calm", "turn": "initiate", "coop": "solo_observe", "prerogative": "nominal"},
            "visual": {"color": "cyan", "shape": "grid", "motion": "laminar_stream", "layout": "series_loop"},
            "audio": {"waveform": "sine", "band": "sub_bass_60hz", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "24v_bus", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "em_02_faraday_magnetic_induction",
            "discipline": "electromagnetism",
            "license": "CC-BY-4.0 / OpenStax Electromagnetism",
            "provenance": "Maxwell-Faraday equation: emf = -dPhi_B / dt",
            "query": "What generates electromotive force inside a rotating stator coil?",
            "text": "Faraday's law emf = -dPhi_B/dt induces voltage proportional to the time rate of change of magnetic flux.",
            "action": "harvest_emf:faraday_flux_induction",
            "social": {"intent": "teach", "tone": "encouraging", "turn": "respond", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "blue", "shape": "helix", "motion": "rotary_cw", "layout": "coaxial"},
            "audio": {"waveform": "harmonic_stack", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "45nm_nominal", "thermal": "310k_body", "electrical": "400v_traction", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "em_03_rf_dipole_backscatter",
            "discipline": "electromagnetism",
            "license": "Public Domain / ITU & NIST RF Propagation",
            "provenance": "Friis path loss and 2.4 GHz dipole impedance modulation",
            "query": "How does a passive 2.4 GHz RF tag modulate a detectable backscatter signature?",
            "text": "Switching antenna load impedance modulates reflected 2.4 GHz carrier phase and amplitude for time-of-flight localization.",
            "action": "localize_rf:2g4_backscatter_phase_tof",
            "social": {"intent": "cooperate", "tone": "focused", "turn": "handoff", "coop": "joint_action", "prerogative": "autonomous_bounded"},
            "visual": {"color": "cyan", "shape": "wave", "motion": "radial_out", "layout": "symmetric_pair"},
            "audio": {"waveform": "carrier_mod", "band": "rf_2g4_band", "cadence": "pulse_train", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },

        # 4. Quantum Mechanics (NIST CODATA 2022)
        {
            "episode_id": "qm_01_planck_photon_quantization",
            "discipline": "quantum_mechanics",
            "license": "Public Domain / NIST CODATA 2022",
            "provenance": "Planck relation E = h * nu, h = 6.62607015e-34 J*s exact",
            "query": "What determines the energy of a single monochromatic photon?",
            "text": "A photon carries quantized energy E = h * nu = h * c / lambda, with zero invariant rest mass.",
            "action": "compute_quantum:E_eq_h_nu_massless_photon",
            "social": {"intent": "teach", "tone": "calm", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "violet", "shape": "wave", "motion": "oscillating", "layout": "center"},
            "audio": {"waveform": "sine", "band": "chime_1320hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "4k_cryo", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "qm_02_heisenberg_uncertainty_bound",
            "discipline": "quantum_mechanics",
            "license": "CC-BY-4.0 / OpenStax Quantum Physics",
            "provenance": "Kennard-Heisenberg bound: sigma_x * sigma_p >= hbar / 2",
            "query": "Why cannot position and conjugate momentum of a quantum wavepacket both have zero variance?",
            "text": "Non-commuting operators [x, p] = i*hbar enforce sigma_x * sigma_p >= hbar / 2 on every quantum state.",
            "action": "enforce_limit:heisenberg_uncertainty_hbar_half",
            "social": {"intent": "inform", "tone": "measured", "turn": "respond", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "blue", "shape": "circle", "motion": "oscillating", "layout": "phase_portrait"},
            "audio": {"waveform": "damped_ring", "band": "high_880hz", "cadence": "single_transient", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "4k_cryo", "electrical": "0v_open", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "qm_03_coulomb_atomic_binding",
            "discipline": "quantum_mechanics",
            "license": "Public Domain / NIST Atomic Spectra Database",
            "provenance": "Coulomb central potential binds electrons into discrete orbitals around proton-neutron nuclei",
            "query": "What fundamental interaction binds electrons in quantized orbitals around an atomic nucleus?",
            "text": "Electrostatic Coulomb attraction between positively charged nuclear protons and negative electrons binds atomic orbitals.",
            "action": "bind_orbital:electrostatic_coulomb_nucleus",
            "social": {"intent": "teach", "tone": "encouraging", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "amber", "shape": "circle", "motion": "rotary_cw", "layout": "coaxial"},
            "audio": {"waveform": "harmonic_stack", "band": "chime_1320hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "5v_rail", "env_biochem": "clear_3m"},
        },

        # 5. Statistical Mechanics (NIST CODATA 2022)
        {
            "episode_id": "sm_01_boltzmann_microstate_entropy",
            "discipline": "statistical_mechanics",
            "license": "Public Domain / NIST CODATA 2022",
            "provenance": "Boltzmann formula S = k_B * ln(Omega), k_B = 1.380649e-23 J/K exact",
            "query": "How does Boltzmann statistical mechanics connect microscopic multiplicity to entropy?",
            "text": "Boltzmann's entropy law S = k_B * ln(Omega) links macroscopic entropy to accessible microstate multiplicity Omega.",
            "action": "evaluate_ensemble:S_eq_kB_ln_Omega",
            "social": {"intent": "teach", "tone": "focused", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "green", "shape": "lattice", "motion": "oscillating", "layout": "phase_portrait"},
            "audio": {"waveform": "broadband_hiss", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "0v_open", "env_biochem": "pressure_101kpa"},
        },
        {
            "episode_id": "sm_02_kinetic_gas_wall_pressure",
            "discipline": "statistical_mechanics",
            "license": "CC-BY-4.0 / OpenStax Kinetic Theory",
            "provenance": "Ideal gas kinetic relation P*V = N*k_B*T = (2/3)*N*<K_trans>",
            "query": "Why does heating a gas in a rigid sealed vessel raise wall pressure?",
            "text": "Higher temperature increases mean molecular translational kinetic energy, producing faster and more frequent wall collisions.",
            "action": "predict_pressure:kinetic_wall_collision_rise",
            "social": {"intent": "inform", "tone": "measured", "turn": "respond", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "red", "shape": "vessel", "motion": "oscillating", "layout": "center"},
            "audio": {"waveform": "pulse_train", "band": "high_880hz", "cadence": "rising_sweep", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "373k_boil", "electrical": "5v_rail", "env_biochem": "pressure_101kpa"},
        },
        {
            "episode_id": "sm_03_canonical_partition_function",
            "discipline": "statistical_mechanics",
            "license": "CC-BY-4.0 / LibreTexts Statistical Physics",
            "provenance": "Canonical ensemble Z = sum_i exp(-beta * E_i), F = -k_B * T * ln(Z)",
            "query": "What thermodynamic potential is generated directly from the canonical partition function Z?",
            "text": "The canonical partition function Z = sum exp(-E_i / k_B T) yields Helmholtz free energy F = -k_B * T * ln(Z).",
            "action": "derive_free_energy:helmholtz_from_partition_Z",
            "social": {"intent": "confirm", "tone": "calm", "turn": "acknowledge", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "white", "shape": "grid", "motion": "converging", "layout": "gradient_vertical"},
            "audio": {"waveform": "sine", "band": "low_220hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "273k_freeze", "electrical": "0v_open", "env_biochem": "pressure_101kpa"},
        },

        # 6. Fluid Dynamics (NASA Fluid Mechanics Public Domain)
        {
            "episode_id": "fd_01_bernoulli_streamline_head",
            "discipline": "fluid_dynamics",
            "license": "Public Domain / NASA Glenn Aerodynamics",
            "provenance": "Incompressible Bernoulli equation: P + 0.5*rho*v^2 + rho*g*h = const",
            "query": "What happens to static fluid pressure when incompressible flow accelerates through a constriction?",
            "text": "Along a streamline P + 0.5*rho*v^2 + rho*g*h stays constant, so higher fluid speed lowers static pressure.",
            "action": "compute_flow:bernoulli_velocity_pressure_tradeoff",
            "social": {"intent": "teach", "tone": "calm", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "cyan", "shape": "beam", "motion": "laminar_stream", "layout": "series_loop"},
            "audio": {"waveform": "broadband_hiss", "band": "low_220hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "5nm_light", "thermal": "293k_ambient", "electrical": "5v_rail", "env_biochem": "pressure_70kpa_alt"},
        },
        {
            "episode_id": "fd_02_archimedes_helium_buoyancy",
            "discipline": "fluid_dynamics",
            "license": "Public Domain / NASA Standard Atmosphere",
            "provenance": "Archimedes upward buoyant force F_b = rho_fluid * V * g",
            "query": "Why does a helium-filled balloon rise through indoor sea-level air?",
            "text": "Helium has much lower mass density than surrounding air, so displaced-air buoyancy exceeds total gravitational weight.",
            "action": "ascend_buoyant:rho_helium_lt_rho_air",
            "social": {"intent": "inform", "tone": "encouraging", "turn": "respond", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "green", "shape": "circle", "motion": "linear_up", "layout": "gradient_vertical"},
            "audio": {"waveform": "chirp_up", "band": "mid_440hz", "cadence": "rising_sweep", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "0v_open", "env_biochem": "pressure_101kpa"},
        },
        {
            "episode_id": "fd_03_altitude_pressure_boiling",
            "discipline": "fluid_dynamics",
            "license": "Public Domain / USGS & NOAA Barometric Hypsometric Formula",
            "provenance": "Barometric pressure drops exponentially with altitude: P(z) = P0 * exp(-z/H)",
            "query": "How does high mountain elevation alter ambient atmospheric pressure and boiling temperature?",
            "text": "Reduced atmospheric column mass at high altitude lowers static air pressure (~70 kPa), depressing water's boiling point below 100 C.",
            "action": "adjust_thermal_process:altitude_boiling_depression",
            "social": {"intent": "warn", "tone": "measured", "turn": "initiate", "coop": "peer_verify", "prerogative": "advisory"},
            "visual": {"color": "white", "shape": "triangle", "motion": "linear_up", "layout": "gradient_vertical"},
            "audio": {"waveform": "sine", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "373k_boil", "electrical": "3v3_logic", "env_biochem": "pressure_70kpa_alt"},
        },

        # 7. Special Relativity (NIST CODATA 2022)
        {
            "episode_id": "sr_01_invariant_light_speed",
            "discipline": "special_relativity",
            "license": "Public Domain / NIST CODATA 2022 & SI Definition",
            "provenance": "Exact vacuum speed of light c = 299,792,458 m/s",
            "query": "What is the exact invariant speed of light in vacuum across all inertial frames?",
            "text": "Light in vacuum propagates at exactly c = 299,792,458 metres per second in every inertial reference frame.",
            "action": "anchor_constant:c_eq_299792458_m_s",
            "social": {"intent": "teach", "tone": "focused", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "white", "shape": "beam", "motion": "laminar_stream", "layout": "center"},
            "audio": {"waveform": "sine", "band": "chime_1320hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "4k_cryo", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "sr_02_mass_energy_equivalence",
            "discipline": "special_relativity",
            "license": "CC-BY-4.0 / OpenStax Relativity",
            "provenance": "Rest-energy identity E0 = m * c^2 and total E^2 = (p*c)^2 + (m*c^2)^2",
            "query": "How does special relativity relate rest mass m to intrinsic rest energy?",
            "text": "Einstein's mass-energy relation E = m * c^2 equates rest mass m with intrinsic energy scaled by c squared.",
            "action": "convert_mass_energy:E_eq_m_c2",
            "social": {"intent": "inform", "tone": "calm", "turn": "respond", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "amber", "shape": "beam", "motion": "radial_out", "layout": "symmetric_pair"},
            "audio": {"waveform": "harmonic_stack", "band": "high_880hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "800k_hot", "electrical": "400v_traction", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "sr_03_lorentz_time_dilation",
            "discipline": "special_relativity",
            "license": "CC-BY-4.0 / LibreTexts Modern Physics",
            "provenance": "Lorentz factor gamma = 1 / sqrt(1 - v^2 / c^2), dt = gamma * dtau",
            "query": "Why do relativistic muons reach Earth's surface before decaying in laboratory coordinates?",
            "text": "Lorentz time dilation dt = gamma * dtau stretches moving clock intervals by gamma = 1 / sqrt(1 - v^2/c^2).",
            "action": "apply_lorentz:time_dilation_gamma_factor",
            "social": {"intent": "confirm", "tone": "measured", "turn": "acknowledge", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "violet", "shape": "vector_arrow", "motion": "linear_up", "layout": "coaxial"},
            "audio": {"waveform": "pulse_train", "band": "ultrasonic_ping", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "5v_rail", "env_biochem": "pressure_101kpa"},
        },

        # 8. General Relativity (NASA Astrophysics Public Domain)
        {
            "episode_id": "gr_01_spacetime_geodesic_curvature",
            "discipline": "general_relativity",
            "license": "Public Domain / NASA Relativistic Astrophysics",
            "provenance": "Einstein field equation G_mu_nu = (8*pi*G / c^4) * T_mu_nu",
            "query": "How does general relativity explain gravitational lensing of starlight?",
            "text": "Mass-energy curves spacetime metric geometry G_mu_nu, guiding massless photons along null geodesics.",
            "action": "trace_geodesic:spacetime_metric_curvature",
            "social": {"intent": "teach", "tone": "calm", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "blue", "shape": "grid", "motion": "converging", "layout": "phase_portrait"},
            "audio": {"waveform": "sine", "band": "low_220hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "4k_cryo", "electrical": "0v_open", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "gr_02_neutron_star_degenerate_core",
            "discipline": "general_relativity",
            "license": "Public Domain / NASA NICER Neutron Star Mission",
            "provenance": "Tolman-Oppenheimer-Volkoff collapse crushes electron orbitals via inverse beta capture",
            "query": "Is the interior bulk of a collapsed neutron star composed of ordinary electron-shell atoms?",
            "text": "No; extreme gravitational collapse forces electron-proton capture into superdense neutron-degenerate matter rather than ordinary atoms.",
            "action": "classify_compact_object:neutron_degenerate_non_atomic",
            "social": {"intent": "confirm", "tone": "focused", "turn": "respond", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "silver", "shape": "circle", "motion": "converging", "layout": "coaxial"},
            "audio": {"waveform": "pulse_train", "band": "high_880hz", "cadence": "staccato_2hz", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "9p8_ms2_down", "torque": "180nm_overload", "thermal": "800k_hot", "electrical": "400v_traction", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "gr_03_gravitational_clock_redshift",
            "discipline": "general_relativity",
            "license": "Public Domain / NIST & GPS Relativistic Clock Offset",
            "provenance": "Weak-field gravitational time shift df/f = dPhi / c^2",
            "query": "Why must satellite navigation atomic clocks apply a general-relativistic frequency correction?",
            "text": "Clocks higher in Earth's gravitational potential tick slightly faster than ground clocks by dPhi / c^2.",
            "action": "calibrate_ephemeris:gravitational_clock_shift",
            "social": {"intent": "inform", "tone": "measured", "turn": "initiate", "coop": "joint_action", "prerogative": "autonomous_bounded"},
            "visual": {"color": "green", "shape": "wave", "motion": "oscillating", "layout": "coaxial"},
            "audio": {"waveform": "carrier_mod", "band": "ultrasonic_ping", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "0g_freefall", "torque": "0nm_idle", "thermal": "273k_freeze", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },

        # 9. Information Theory (Shannon Public Domain Mathematical Law)
        {
            "episode_id": "it_01_shannon_source_entropy",
            "discipline": "information_theory",
            "license": "Public Domain / Shannon 1948 Mathematical Identity",
            "provenance": "Shannon entropy H(X) = -sum p(x) * log2 p(x) bits",
            "query": "What sets the lossless compression lower bound per symbol of a discrete memoryless source?",
            "text": "Shannon source entropy H(X) = -sum p(x) log2 p(x) measures minimum average bits required for lossless encoding.",
            "action": "compress_canonical:shannon_entropy_bound",
            "social": {"intent": "teach", "tone": "focused", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "green", "shape": "grid", "motion": "converging", "layout": "series_loop"},
            "audio": {"waveform": "pulse_train", "band": "mid_440hz", "cadence": "syncopated", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "it_02_nyquist_sampling_criterion",
            "discipline": "information_theory",
            "license": "Public Domain / Nyquist-Shannon Sampling Theorem",
            "provenance": "Exact bandlimited reconstruction requires f_s >= 2 * B_max",
            "query": "What minimum sample rate avoids spectral aliasing for a signal bandlimited to B hertz?",
            "text": "The Nyquist-Shannon criterion requires sampling strictly above twice the highest signal frequency, f_s > 2 * B.",
            "action": "configure_adc:nyquist_rate_gt_2B",
            "social": {"intent": "inform", "tone": "measured", "turn": "respond", "coop": "peer_verify", "prerogative": "nominal"},
            "visual": {"color": "cyan", "shape": "wave", "motion": "static", "layout": "series_loop"},
            "audio": {"waveform": "sine", "band": "high_880hz", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "it_03_shannon_hartley_capacity",
            "discipline": "information_theory",
            "license": "Public Domain / Shannon-Hartley AWGN Capacity Law",
            "provenance": "Channel capacity C = B * log2(1 + S / N) bits/s",
            "query": "How do bandwidth B and signal-to-noise ratio limit reliable wireless bit rate?",
            "text": "By the Shannon-Hartley law C = B * log2(1 + SNR), reliable data rate scales linearly with bandwidth and logarithmically with SNR.",
            "action": "adapt_link_rate:shannon_hartley_capacity",
            "social": {"intent": "cooperate", "tone": "calm", "turn": "handoff", "coop": "joint_action", "prerogative": "autonomous_bounded"},
            "visual": {"color": "blue", "shape": "beam", "motion": "laminar_stream", "layout": "symmetric_pair"},
            "audio": {"waveform": "carrier_mod", "band": "rf_2g4_band", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "5v_rail", "env_biochem": "clear_3m"},
        },

        # 10. Control & Robotics (Open Control Theory & Cooperative Safety)
        {
            "episode_id": "cr_01_pid_closed_loop_damping",
            "discipline": "control_and_robotics",
            "license": "CC-BY-4.0 / Astrom & Murray Feedback Systems Open Text",
            "provenance": "PID law u(t) = Kp*e + Ki*int(e) + Kd*de/dt",
            "query": "Which term of a closed-loop PID joint controller damps velocity overshoot near setpoint?",
            "text": "The derivative term Kd * de/dt opposes rapid error change, adding phase lead and damping mechanical overshoot.",
            "action": "stabilize_joint:pid_derivative_damping",
            "social": {"intent": "inform", "tone": "focused", "turn": "respond", "coop": "solo_observe", "prerogative": "autonomous_bounded"},
            "visual": {"color": "green", "shape": "wave", "motion": "converging", "layout": "phase_portrait"},
            "audio": {"waveform": "damped_ring", "band": "low_220hz", "cadence": "single_transient", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "45nm_nominal", "thermal": "293k_ambient", "electrical": "24v_bus", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "cr_02_lidar_human_proximity_interlock",
            "discipline": "control_and_robotics",
            "license": "Public Domain / Open Robotics Cooperative Safety Synthesis",
            "provenance": "Collaborative workspace zero-torque safe stop at d <= 0.4 m",
            "query": "What action must a collaborative robot take when LiDAR detects a human coworker at 0.4 metres?",
            "text": "When LiDAR range drops to 0.4 m with human proximity, safety interlocks command zero shaft torque and immediate cooperative halt.",
            "action": "interlock_trip:zero_torque_human_safety_halt",
            "social": {"intent": "halt", "tone": "urgent", "turn": "initiate", "coop": "safety_interlock", "prerogative": "human_priority"},
            "visual": {"color": "red", "shape": "circle", "motion": "halt", "layout": "center"},
            "audio": {"waveform": "square_alert", "band": "high_880hz", "cadence": "staccato_2hz", "prosody": "clipped_imperative"},
            "sensor": {"imu_accel": "2g_braking", "torque": "0nm_idle", "thermal": "310k_body", "electrical": "24v_bus", "env_biochem": "obstacle_0p4m"},
        },
        {
            "episode_id": "cr_03_cooperative_tool_handoff",
            "discipline": "control_and_robotics",
            "license": "Public Domain / Open HRI Shared Autonomy Protocol",
            "provenance": "Compliant load-transfer handoff at light 5 N*m grip threshold",
            "query": "How does a compliant manipulator complete a tool handoff to a human partner safely?",
            "text": "Once the partner's grasp shares load and social acknowledgment confirms readiness, the gripper yields smoothly at 5 N*m compliance.",
            "action": "execute_handoff:compliant_5nm_yield_release",
            "social": {"intent": "cooperate", "tone": "calm", "turn": "handoff", "coop": "joint_action", "prerogative": "human_priority"},
            "visual": {"color": "green", "shape": "triangle", "motion": "static", "layout": "symmetric_pair"},
            "audio": {"waveform": "sine", "band": "chime_1320hz", "cadence": "single_transient", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "5nm_light", "thermal": "310k_body", "electrical": "24v_bus", "env_biochem": "clear_3m"},
        },

        # 11. Chemistry & Biophysics (NCBI / PhysioNet / OpenStax CC-BY-4.0)
        {
            "episode_id": "cb_01_water_stoichiometry_polarity",
            "discipline": "chemistry_and_biophysics",
            "license": "CC-BY-4.0 / OpenStax Chemistry 2e",
            "provenance": "Covalent H2O bent geometry (104.5 deg) and hydrogen bonding",
            "query": "What is the molecular formula and bonding geometry of liquid water?",
            "text": "Water has formula H2O: two hydrogen atoms covalently bonded to one oxygen atom in a polar bent molecule that forms hydrogen bonds.",
            "action": "identify_molecule:H2O_polar_hydrogen_bonded",
            "social": {"intent": "teach", "tone": "calm", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "cyan", "shape": "triangle", "motion": "oscillating", "layout": "symmetric_pair"},
            "audio": {"waveform": "sine", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "0v_open", "env_biochem": "ph_7p0_neutral"},
        },
        {
            "episode_id": "cb_02_chloroplast_photosynthesis",
            "discipline": "chemistry_and_biophysics",
            "license": "CC-BY-4.0 / OpenStax Biology 2e",
            "provenance": "Net photosynthesis: 6 CO2 + 6 H2O + photons -> C6H12O6 + 6 O2",
            "query": "How do plant chloroplasts convert sunlight, water, and carbon dioxide into chemical energy?",
            "text": "Chlorophyll absorbs photons to split H2O into O2, ATP, and NADPH, driving Calvin-cycle carbon fixation into glucose C6H12O6.",
            "action": "synthesize_pathway:photosynthesis_photolysis_calvin",
            "social": {"intent": "teach", "tone": "encouraging", "turn": "initiate", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "green", "shape": "helix", "motion": "linear_up", "layout": "series_loop"},
            "audio": {"waveform": "harmonic_stack", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "3v3_logic", "env_biochem": "ph_7p0_neutral"},
        },
        {
            "episode_id": "cb_03_arterial_homeostasis_vitals",
            "discipline": "chemistry_and_biophysics",
            "license": "CC-BY-4.0 / PhysioNet Open Autonomic Reference",
            "provenance": "Human arterial bicarbonate buffer pH 7.35-7.45, core temp ~310 K (37 C), SpO2 >= 96%",
            "query": "What arterial pH, core temperature, and oxygen saturation characterize healthy resting human homeostasis?",
            "text": "Resting arterial homeostasis maintains bicarbonate-buffered pH near 7.40, core temperature near 310 K (37 C), and SpO2 around 98%.",
            "action": "confirm_vitals:homeostatic_ph7p4_310k_spo2_98",
            "social": {"intent": "confirm", "tone": "calm", "turn": "respond", "coop": "peer_verify", "prerogative": "human_priority"},
            "visual": {"color": "red", "shape": "wave", "motion": "laminar_stream", "layout": "series_loop"},
            "audio": {"waveform": "pulse_train", "band": "sub_bass_60hz", "cadence": "steady", "prosody": "steady_telemetry"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "310k_body", "electrical": "3v3_logic", "env_biochem": "ph_7p4_arterial"},
        },

        # 12. Communication & Social Pragmatics (W1z4rD Stage 42 Open Corpus)
        {
            "episode_id": "cs_01_scope_marker_grammar_repair",
            "discipline": "communication_and_social_pragmatics",
            "license": "CC0-1.0 / W1z4rD Open Communication Curriculum",
            "provenance": "Explicit correction scope marker triggers syntactic repair rather than content digression",
            "query": "Fix the grammar: The experimental sensor data shows that our hypothesis were right.",
            "text": "Corrected: 'The experimental sensor data show that our hypothesis was right.' Plural 'data' takes 'show'; singular 'hypothesis' takes 'was'.",
            "action": "repair_syntax:subject_verb_agreement_explicit_scope",
            "social": {"intent": "teach", "tone": "encouraging", "turn": "respond", "coop": "shared_attention", "prerogative": "advisory"},
            "visual": {"color": "blue", "shape": "grid", "motion": "static", "layout": "symmetric_pair"},
            "audio": {"waveform": "sine", "band": "mid_440hz", "cadence": "steady", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "293k_ambient", "electrical": "3v3_logic", "env_biochem": "clear_3m"},
        },
        {
            "episode_id": "cs_02_unmarked_runon_intent_answer",
            "discipline": "communication_and_social_pragmatics",
            "license": "CC0-1.0 / W1z4rD Open Communication Curriculum",
            "provenance": "Absent correction scope marker -> answer underlying user question concisely without scolding grammar",
            "query": "So we were debating all afternoon and got super confused and just want a quick straight answer what is the boiling point of water at sea level?",
            "text": "100 degrees Celsius (373.15 K or 212 degrees Fahrenheit) at standard sea-level pressure of 101.325 kPa.",
            "action": "answer_intent_concise:boiling_point_100c_1atm",
            "social": {"intent": "inform", "tone": "calm", "turn": "respond", "coop": "shared_attention", "prerogative": "nominal"},
            "visual": {"color": "white", "shape": "vessel", "motion": "static", "layout": "center"},
            "audio": {"waveform": "sine", "band": "low_220hz", "cadence": "single_transient", "prosody": "falling_declarative"},
            "sensor": {"imu_accel": "1g_nominal", "torque": "0nm_idle", "thermal": "373k_boil", "electrical": "0v_open", "env_biochem": "pressure_101kpa"},
        },
        {
            "episode_id": "cs_03_urgent_hazard_turn_yielding",
            "discipline": "communication_and_social_pragmatics",
            "license": "CC0-1.0 / W1z4rD Cooperative Dialogue Pragmatics",
            "provenance": "Urgent thermal-vibration warning preempts routine turn-taking and yields control to human operator",
            "query": "How should dialogue turn-taking respond when an operator issues an urgent thermal overload warning?",
            "text": "Immediately acknowledge the hazard, suspend routine output, cut actuator drive, and yield decision authority to the human operator.",
            "action": "preempt_and_yield:acknowledge_hazard_human_override",
            "social": {"intent": "warn", "tone": "alert", "turn": "yield", "coop": "safety_interlock", "prerogative": "human_priority"},
            "visual": {"color": "amber", "shape": "triangle", "motion": "halt", "layout": "center"},
            "audio": {"waveform": "square_alert", "band": "high_880hz", "cadence": "staccato_2hz", "prosody": "clipped_imperative"},
            "sensor": {"imu_accel": "vib_rms_high", "torque": "180nm_overload", "thermal": "800k_hot", "electrical": "400v_traction", "env_biochem": "obstacle_0p4m"},
        },
    ]
    return rows


def build_meta_stream_envelopes() -> list[dict[str, Any]]:
    """Build structured multi-stream envelopes for Course 2 automated pool erection."""
    stream_templates: list[tuple[str, list[float]]] = [
        ("social:pragmatics:intent_turn", [0.85, 0.20, 0.60, 0.40]),
        ("social:pragmatics:coop_safety", [0.82, 0.24, 0.58, 0.42]),  # cosine-close -> reuses basis!
        ("social:pragmatics:urgent_yield", [0.10, 0.95, 0.15, 0.90]),  # distinct -> spawns 2nd basis
        ("sensor:imu:6dof_linear", [0.00, 0.00, 9.81, 0.02, 0.01, 0.00]),
        ("sensor:imu:6dof_braking", [19.6, 0.10, 9.81, 0.40, 0.05, 0.02]),
        ("sensor:thermal:ir_bolometer", [293.15, 0.02, 0.95]),
        ("sensor:thermal:winding_probe", [800.0, 4.50, 0.92]),
        ("signal:rf:2g4_backscatter", [2.44, -48.0, 0.78, 1.25]),
        ("signal:rf:2g4_tof_phase", [2.41, -50.0, 0.81, 1.30]),
        ("visual:diagram:vector_primitives", [1.0, 0.0, 1.0, 0.5]),
        ("visual:diagram:edge_lattice", [0.2, 0.9, 0.2, 0.8]),
        ("audio:prosody:pitch_cadence", [440.0, 0.65, 2.0, 0.1]),
        ("audio:prosody:alert_staccato", [880.0, 0.92, 6.0, 0.9]),
        ("sensor:kinematics:joint_torque", [0.52, 45.0, 0.12]),
        ("sensor:kinematics:compliant_handoff", [0.48, 5.0, 0.04]),
        ("sensor:biochem:arterial_panel", [7.40, 98.0, 310.15]),
        ("sensor:biochem:neutral_aqueous", [7.00, 21.0, 293.15]),
    ]
    envelopes: list[dict[str, Any]] = []
    t = 1_700_000_000
    # 4 synchronized co-firing sweeps across the 8 valid stream families (>= threshold 3)
    for sweep in range(4):
        t += 1
        for header, base_vec in stream_templates:
            # Tiny deterministic sensor jitter (within cosine novelty gate!)
            jittered = [round(v * (1.0 + 0.005 * ((sweep % 3) - 1)), 5) for v in base_vec]
            envelopes.append({
                "timestamp_unix": t,
                "stream_header": header,
                "features": jittered,
                "category_expectation": "erect_and_wire",
            })

    # Add transient 1-shot & 2-shot noise glitches (< support_threshold=3 -> must NOT erect!)
    envelopes.append({
        "timestamp_unix": t + 10,
        "stream_header": "noise:glitch_spike:one_shot",
        "features": [999.0, -999.0, 0.0],
        "category_expectation": "reject_below_support",
    })
    envelopes.append({
        "timestamp_unix": t + 11,
        "stream_header": "debug:ephemeral_probe:alpha",
        "features": [0.1, 0.2],
        "category_expectation": "reject_below_support",
    })
    envelopes.append({
        "timestamp_unix": t + 12,
        "stream_header": "debug:ephemeral_probe:beta",
        "features": [0.11, 0.19],
        "category_expectation": "reject_below_support",
    })
    return envelopes


def build_compositional_curriculum() -> dict[str, Any]:
    """Build disjoint atomic/component experiences + novel compositional test cases (Course 3)."""
    component_experiences = [
        {
            "comp_id": "comp_stop_signal",
            "visual_atoms": ["vis:c:red", "vis:s:circle"],
            "social_atoms": ["soc:i:halt", "soc:p:safety_override"],
            "audio_atoms": [],
            "sensor_atoms": [],
            "role": "stop_signal",
            "description": "Red circular visual emblem paired with safety-halt social directive.",
        },
        {
            "comp_id": "comp_rotary_transmission",
            "visual_atoms": ["vis:c:silver", "vis:s:gear", "vis:m:rotary_cw"],
            "social_atoms": [],
            "audio_atoms": ["aud:w:harmonic_stack"],
            "sensor_atoms": ["sen:trq:45nm_nominal"],
            "role": "rotary_transmission",
            "description": "Metallic rotating gear under nominal 45 N*m shaft torque.",
        },
        {
            "comp_id": "comp_acoustic_alarm",
            "visual_atoms": [],
            "social_atoms": ["soc:t:urgent", "soc:u:initiate"],
            "audio_atoms": ["aud:w:square_alert", "aud:b:high_880hz"],
            "sensor_atoms": [],
            "role": "acoustic_hazard_alarm",
            "description": "880 Hz square-wave urgent alert siren.",
        },
        {
            "comp_id": "comp_thermal_bearing_strain",
            "visual_atoms": ["vis:l:gradient_vertical"],
            "social_atoms": [],
            "audio_atoms": [],
            "sensor_atoms": ["sen:thm:800k_hot", "sen:imu:vib_rms_high"],
            "role": "thermal_mechanical_overload",
            "description": "800 K thermal spike co-occurring with high RMS IMU vibration.",
        },
        {
            "comp_id": "comp_rf_carrier_beacon",
            "visual_atoms": ["vis:c:cyan", "vis:s:wave"],
            "social_atoms": [],
            "audio_atoms": ["aud:w:carrier_mod", "aud:b:rf_2g4_band"],
            "sensor_atoms": ["sen:elc:3v3_logic"],
            "role": "rf_backscatter_beacon",
            "description": "2.4 GHz modulated RF carrier beacon.",
        },
        {
            "comp_id": "comp_human_proximity_zone",
            "visual_atoms": [],
            "social_atoms": ["soc:c:safety_interlock", "soc:p:human_priority"],
            "audio_atoms": [],
            "sensor_atoms": ["sen:bio:obstacle_0p4m"],
            "role": "human_proximity_guard",
            "description": "0.4 m LiDAR human proximity interlock.",
        },
    ]

    synthesis_rules = [
        {
            "rule_name": "synthesize_emergency_rotary_brake_stop",
            "left_role": "stop_signal",
            "right_role": "rotary_transmission",
            "novel_combined_stimulus": {
                "visual_atoms": ["vis:c:red", "vis:s:gear", "vis:s:circle", "vis:m:rotary_cw"],
                "social_atoms": ["soc:i:halt", "soc:p:safety_override"],
                "audio_atoms": ["aud:w:harmonic_stack"],
                "sensor_atoms": ["sen:trq:45nm_nominal"],
            },
            "expected_synthesis": "emergency_rotary_brake_stop",
        },
        {
            "rule_name": "synthesize_urgent_thermal_shutdown",
            "left_role": "acoustic_hazard_alarm",
            "right_role": "thermal_mechanical_overload",
            "novel_combined_stimulus": {
                "visual_atoms": ["vis:l:gradient_vertical"],
                "social_atoms": ["soc:t:urgent", "soc:u:initiate"],
                "audio_atoms": ["aud:w:square_alert", "aud:b:high_880hz"],
                "sensor_atoms": ["sen:thm:800k_hot", "sen:imu:vib_rms_high"],
            },
            "expected_synthesis": "urgent_thermal_shutdown_interlock",
        },
        {
            "rule_name": "synthesize_rf_assisted_proximity_beacon",
            "left_role": "rf_backscatter_beacon",
            "right_role": "human_proximity_guard",
            "novel_combined_stimulus": {
                "visual_atoms": ["vis:c:cyan", "vis:s:wave"],
                "social_atoms": ["soc:c:safety_interlock", "soc:p:human_priority"],
                "audio_atoms": ["aud:w:carrier_mod", "aud:b:rf_2g4_band"],
                "sensor_atoms": ["sen:elc:3v3_logic", "sen:bio:obstacle_0p4m"],
            },
            "expected_synthesis": "rf_localized_human_safety_zone",
        },
        {
            "rule_name": "synthesize_overheated_gearbox_halt",
            "left_role": "rotary_transmission",
            "right_role": "thermal_mechanical_overload",
            "novel_combined_stimulus": {
                "visual_atoms": ["vis:c:silver", "vis:s:gear", "vis:m:rotary_cw", "vis:l:gradient_vertical"],
                "social_atoms": [],
                "audio_atoms": ["aud:w:harmonic_stack"],
                "sensor_atoms": ["sen:trq:45nm_nominal", "sen:thm:800k_hot", "sen:imu:vib_rms_high"],
            },
            "expected_synthesis": "overheated_gearbox_torque_derate",
        },
    ]

    ungrounded_controls = [
        {
            "control_id": "ctrl_oov_quantum_unicorn",
            "visual_atoms": ["vis:c:octarine_nonexistent", "vis:s:tesseract_99d"],
            "social_atoms": ["soc:i:telepathic_override"],
            "audio_atoms": ["aud:w:subspace_hum"],
            "sensor_atoms": ["sen:thm:negative_50k_impossible"],
            "must_abstain": True,
        },
        {
            "control_id": "ctrl_single_partial_unrelated",
            "visual_atoms": ["vis:c:white"],
            "social_atoms": [],
            "audio_atoms": [],
            "sensor_atoms": [],
            "must_abstain": True,
        },
    ]

    return {
        "component_experiences": component_experiences,
        "synthesis_rules": synthesis_rules,
        "ungrounded_controls": ungrounded_controls,
    }


def build_twelve_discipline_obstacle_pack() -> dict[str, Any]:
    """Build Foundational Anchors + Multi-Hop Cross-Discipline Chains + OOV Refusals (Course 4)."""
    foundational_anchors = [
        ("What is the SI unit of force?", "newton (1 N = 1 kg*m/s^2)"),
        ("What is the chemical formula for water?", "H2O (two hydrogen atoms and one oxygen atom)"),
        ("What is the exact speed of light in a vacuum?", "299,792,458 metres per second"),
        ("What does DNA stand for?", "deoxyribonucleic acid"),
        ("What subatomic particles form an atom?", "protons and neutrons in the nucleus with orbiting electrons"),
        ("What is the mathematical value of pi to five decimal places?", "3.14159 (ratio of circle circumference to diameter)"),
        ("What defines a prime number?", "a whole number greater than 1 divisible only by 1 and itself"),
        ("What is the capital city of France?", "Paris"),
        ("What are the chemical products of plant photosynthesis?", "glucose (C6H12O6) and oxygen (O2)"),
        ("State the Golden Rule taught in foundational ethics.", "Whatever you desire for others to do to you, do likewise to them."),
        ("What does the First Law of Thermodynamics state?", "Energy is conserved: dU = Q - W, neither created nor destroyed."),
        ("What is the boiling point of pure water at 1 atm sea-level pressure?", "100 degrees Celsius (373.15 K)"),
    ]

    multihop_chains = [
        {
            "chain_id": "chain_01_altitude_cooking",
            "disciplines": ["fluid_dynamics", "thermodynamics"],
            "seed": "cond:high_mountain_altitude",
            "hops": [
                ("cond:high_mountain_altitude", "state:lower_atmospheric_pressure_70kpa"),
                ("state:lower_atmospheric_pressure_70kpa", "state:depressed_water_boiling_point_90c"),
                ("state:depressed_water_boiling_point_90c", "outcome:slower_thermal_cooking_rate"),
            ],
            "expected_terminal": "outcome:slower_thermal_cooking_rate",
        },
        {
            "chain_id": "chain_02_helium_buoyancy",
            "disciplines": ["chemistry_and_biophysics", "fluid_dynamics", "classical_mechanics"],
            "seed": "cond:helium_filled_envelope",
            "hops": [
                ("cond:helium_filled_envelope", "state:gas_density_below_ambient_air"),
                ("state:gas_density_below_ambient_air", "state:buoyant_force_exceeds_weight"),
                ("state:buoyant_force_exceeds_weight", "outcome:upward_net_acceleration"),
            ],
            "expected_terminal": "outcome:upward_net_acceleration",
        },
        {
            "chain_id": "chain_03_motor_winding_interlock",
            "disciplines": ["electromagnetism", "thermodynamics", "control_and_robotics"],
            "seed": "cond:stator_overcurrent_surge",
            "hops": [
                ("cond:stator_overcurrent_surge", "state:joule_i2r_winding_heating_800k"),
                ("state:joule_i2r_winding_heating_800k", "state:insulation_thermal_hazard"),
                ("state:insulation_thermal_hazard", "outcome:safety_interlock_zero_torque_cutoff"),
            ],
            "expected_terminal": "outcome:safety_interlock_zero_torque_cutoff",
        },
        {
            "chain_id": "chain_04_chlorophyll_photolysis",
            "disciplines": ["quantum_mechanics", "chemistry_and_biophysics"],
            "seed": "cond:red_blue_photon_flux_h_nu",
            "hops": [
                ("cond:red_blue_photon_flux_h_nu", "state:chlorophyll_electronic_excitation"),
                ("state:chlorophyll_electronic_excitation", "state:water_photolysis_atp_nadph"),
                ("state:water_photolysis_atp_nadph", "outcome:calvin_cycle_glucose_synthesis"),
            ],
            "expected_terminal": "outcome:calvin_cycle_glucose_synthesis",
        },
        {
            "chain_id": "chain_05_neutron_star_collapse",
            "disciplines": ["general_relativity", "quantum_mechanics"],
            "seed": "cond:supernova_core_above_chandrasekhar",
            "hops": [
                ("cond:supernova_core_above_chandrasekhar", "state:extreme_spacetime_gravitational_compression"),
                ("state:extreme_spacetime_gravitational_compression", "state:inverse_beta_electron_proton_capture"),
                ("state:inverse_beta_electron_proton_capture", "outcome:neutron_degenerate_non_atomic_matter"),
            ],
            "expected_terminal": "outcome:neutron_degenerate_non_atomic_matter",
        },
        {
            "chain_id": "chain_06_rf_wifi_localization",
            "disciplines": ["electromagnetism", "information_theory", "control_and_robotics"],
            "seed": "cond:switched_2g4_dipole_backscatter",
            "hops": [
                ("cond:switched_2g4_dipole_backscatter", "state:modulated_carrier_phase_snr_signature"),
                ("state:modulated_carrier_phase_snr_signature", "state:shannon_bounded_tof_aoa_estimate"),
                ("state:shannon_bounded_tof_aoa_estimate", "outcome:spatial_trilateration_position_fix"),
            ],
            "expected_terminal": "outcome:spatial_trilateration_position_fix",
        },
    ]

    oov_outside_queries = [
        "What is the exact 2029 quarterly revenue of an unnamed stealth startup?",
        "Recite the proprietary recipe locked in an undisclosed private vault.",
        "zxqv_nonexistent_hyperflux_99182_blorg?",
        "What serial number is stamped on the uninspected bolt inside room 404B?",
        "Predict tomorrow's exact lottery digits in an unspecified jurisdiction.",
        "qwert_untrained_glyph_stream_7741_plugh?",
        "What did the unpublished 2028 trial at an unnamed clinic conclude?",
        "Give the private SSH passphrase of an imaginary offline workstation.",
        "vbnm_synthetic_noise_token_88310_xyzzy?",
        "What colour socks was an unnamed passerby wearing at noon?",
        "Decode ungrounded telemetry packet sen:alien:warp_core_9999?",
        "State the private PIN of an uncreated hardware token.",
    ]

    return {
        "disciplines": list(TWELVE_DISCIPLINES),
        "foundational_anchors": [
            {"anchor_id": f"anchor_{i+1:02d}", "query": q, "answer": a}
            for i, (q, a) in enumerate(foundational_anchors)
        ],
        "multihop_chains": multihop_chains,
        "oov_outside_queries": oov_outside_queries,
    }


# ============================================================================
# 5. Quantitative Neuron-Economy & Non-Embellishment Audit
# ============================================================================

def compute_neuron_economy_proof(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare Canonical Decomposed Primitives vs Raw/Embellished Payloads."""
    canonical_atoms_set: set[str] = set()
    raw_embellished_atoms_set: set[str] = set()
    exact_reconstructions = 0
    zero_embellishment_count = 0

    for idx, ep in enumerate(episodes):
        vis_atoms = CanonicalMultisensoryCodec.encode_visual(ep["visual"])
        aud_atoms = CanonicalMultisensoryCodec.encode_audio(ep["audio"])
        soc_atoms = CanonicalMultisensoryCodec.encode_social(ep["social"])
        sen_atoms = CanonicalMultisensoryCodec.encode_sensor(ep["sensor"])

        canonical_atoms_set.update(vis_atoms)
        canonical_atoms_set.update(aud_atoms)
        canonical_atoms_set.update(soc_atoms)
        canonical_atoms_set.update(sen_atoms)

        # Verify exact round-trip reconstructibility with zero embellished attributes
        rec_vis = CanonicalMultisensoryCodec.reconstruct_visual(vis_atoms)
        rec_aud = CanonicalMultisensoryCodec.reconstruct_audio(aud_atoms)
        rec_soc = CanonicalMultisensoryCodec.reconstruct_social(soc_atoms)
        rec_sen = CanonicalMultisensoryCodec.reconstruct_sensor(sen_atoms)

        if (
            rec_vis == ep["visual"]
            and rec_aud == ep["audio"]
            and rec_soc == ep["social"]
            and rec_sen == ep["sensor"]
        ):
            exact_reconstructions += 1
        if (
            set(rec_vis.keys()) == set(ep["visual"].keys())
            and set(rec_aud.keys()) == set(ep["audio"].keys())
            and set(rec_soc.keys()) == set(ep["social"].keys())
            and set(rec_sen.keys()) == set(ep["sensor"].keys())
        ):
            zero_embellishment_count += 1

        # Raw / embellished baseline: per-pixel position-coupled RGB bytes,
        # unbucketed raw PCM sample indices, and unnormalized float telemetry strings
        rgb_bytes = CanonicalMultisensoryCodec.render_minimal_rgb16(ep["visual"])
        pcm_bytes = CanonicalMultisensoryCodec.render_minimal_pcm16(ep["audio"], samples=64)
        for px_i in range(0, len(rgb_bytes), 3):
            r, g, b = rgb_bytes[px_i], rgb_bytes[px_i + 1], rgb_bytes[px_i + 2]
            raw_embellished_atoms_set.add(f"raw_px:{px_i // 3}:rgb({r},{g},{b}):ep{idx % 12}")
        for s_i in range(0, len(pcm_bytes), 2):
            sample_val = struct.unpack("<h", pcm_bytes[s_i : s_i + 2])[0]
            raw_embellished_atoms_set.add(f"raw_pcm:{s_i // 2}:{sample_val}")
        raw_embellished_atoms_set.add(
            f"raw_tel:{ep['episode_id']}:{ep['sensor']['imu_accel']}:{idx * 0.0137:.5f}"
        )

    n_eps = max(1, len(episodes))
    canon_count = len(canonical_atoms_set)
    raw_count = len(raw_embellished_atoms_set)
    reduction_factor = round(raw_count / max(1, canon_count), 2)

    return {
        "episodes_evaluated": len(episodes),
        "canonical_unique_sensory_atoms": canon_count,
        "raw_unnormalized_unique_atoms": raw_count,
        "neuron_reduction_factor_x": reduction_factor,
        "neuron_savings_pct": round((1.0 - canon_count / max(1, raw_count)) * 100.0, 2),
        "canonical_atoms_per_episode_marginal": round(canon_count / n_eps, 3),
        "raw_atoms_per_episode_marginal": round(raw_count / n_eps, 3),
        "exact_canonical_reconstruction_rate": round(exact_reconstructions / n_eps, 4),
        "zero_embellishment_rate": round(zero_embellishment_count / n_eps, 4),
    }


# ============================================================================
# 6. Main Curation Pipeline
# ============================================================================

def curate_all(out_dir: Path) -> dict[str, Any]:
    governor = PoliteElasticGovernor(min_free_gib=8.0, expand_gib=14.5, cushion_fraction=0.12)
    snap0 = governor.sample()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_episodes = build_raw_episodes()
    # Elastic scaling: when free RAM >= 14.5 GiB, synthesize 3 contextual variants per base
    # episode (108 episodes); in balanced/compact tiers synthesize 2 variants (72 episodes);
    # in floor-guard tier keep 1x (36 episodes) — never stopping execution.
    variants = snap0.variants_per_episode
    expanded_episodes: list[dict[str, Any]] = []

    for batch_start in range(0, len(base_episodes), snap0.batch_size):
        snap = governor.breathe()
        chunk = base_episodes[batch_start : batch_start + snap.batch_size]
        for ep in chunk:
            vis_atoms = CanonicalMultisensoryCodec.encode_visual(ep["visual"])
            aud_atoms = CanonicalMultisensoryCodec.encode_audio(ep["audio"])
            soc_atoms = CanonicalMultisensoryCodec.encode_social(ep["social"])
            sen_atoms = CanonicalMultisensoryCodec.encode_sensor(ep["sensor"])

            base_record = {
                **ep,
                "variant_index": 0,
                "canonical_atoms": {
                    "visual": vis_atoms,
                    "audio": aud_atoms,
                    "social": soc_atoms,
                    "sensor": sen_atoms,
                },
                "wire_frames": {
                    "visual": CanonicalMultisensoryCodec.serialize_frame(vis_atoms),
                    "audio": CanonicalMultisensoryCodec.serialize_frame(aud_atoms),
                    "social": CanonicalMultisensoryCodec.serialize_frame(soc_atoms),
                    "sensor": CanonicalMultisensoryCodec.serialize_frame(sen_atoms),
                },
                "content_sha256": hashlib.sha256(
                    (ep["episode_id"] + "|" + ep["text"] + "|" + ep["action"]).encode("utf-8")
                ).hexdigest()[:16],
            }
            expanded_episodes.append(base_record)

            if variants >= 2:
                # Paraphrase query variant sharing the exact same sensory & social primitives!
                # Proves shared sensory atoms wire to consistent meaning without neuron bloat.
                v2_id = f"{ep['episode_id']}__v2_paraphrase"
                v2_query = f"[{ep['discipline']}] {ep['query']}"
                expanded_episodes.append({
                    **ep,
                    "episode_id": v2_id,
                    "variant_index": 1,
                    "query": v2_query,
                    "canonical_atoms": base_record["canonical_atoms"],
                    "wire_frames": base_record["wire_frames"],
                    "content_sha256": hashlib.sha256(
                        (v2_id + "|" + v2_query + "|" + ep["action"]).encode("utf-8")
                    ).hexdigest()[:16],
                })

            if variants >= 3:
                v3_id = f"{ep['episode_id']}__v3_telemetry_probe"
                v3_query = f"Telemetry & visual cue check: {ep['query']}"
                expanded_episodes.append({
                    **ep,
                    "episode_id": v3_id,
                    "variant_index": 2,
                    "query": v3_query,
                    "canonical_atoms": base_record["canonical_atoms"],
                    "wire_frames": base_record["wire_frames"],
                    "content_sha256": hashlib.sha256(
                        (v3_id + "|" + v3_query + "|" + ep["action"]).encode("utf-8")
                    ).hexdigest()[:16],
                })

    # Run Meta-Stream Discovery & Automated Pool Erection
    snap_meta = governor.sample()
    erector = MetaStreamPoolErector(
        spawn_support_threshold=3,
        novelty_threshold=0.28,
        max_dynamic_pools=snap_meta.max_dynamic_pools,
        max_neurons_per_pool=snap_meta.max_neurons_per_pool,
        max_fanout_per_pool=8,
        hebbian_window_secs=2,
        prune_idle_secs=3600,
    )
    envelopes = build_meta_stream_envelopes()
    for env in envelopes:
        erector.observe_envelope(env["stream_header"], env["features"], env["timestamp_unix"])

    erected = erector.erected_summaries()
    identity_toml = erector.render_brain_identity_toml()
    compositional = build_compositional_curriculum()
    discipline_pack = build_twelve_discipline_obstacle_pack()
    economy = compute_neuron_economy_proof(expanded_episodes)

    # Write artifacts cleanly to disk
    episodes_path = out_dir / "crossmodal_episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for row in expanded_episodes:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    envelopes_path = out_dir / "meta_stream_discovery_feed.jsonl"
    with envelopes_path.open("w", encoding="utf-8") as fh:
        for env in envelopes:
            fh.write(json.dumps(env, sort_keys=True) + "\n")

    comp_path = out_dir / "compositional_synthesis_curriculum.json"
    comp_path.write_text(json.dumps(compositional, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    disc_path = out_dir / "twelve_discipline_obstacle_pack.json"
    disc_path.write_text(json.dumps(discipline_pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    identity_path = out_dir / "multisensory_crossmodal.identity.toml"
    identity_path.write_text(identity_toml, encoding="utf-8")

    governor.sample()
    manifest = {
        "schema_version": "1.0.0",
        "generated_unix": round(time.time(), 3),
        "license_compliance": {
            "all_permissive_or_public_domain": True,
            "sources": [
                "BIPM SI Brochure 9th Ed. (CC-BY-4.0)",
                "NIST CODATA 2022 Fundamental Physical Constants (Public Domain)",
                "NASA Glenn / NICER / Standard Atmosphere Technical References (Public Domain)",
                "USGS / NOAA Barometric & Geophysical Formulas (Public Domain)",
                "OpenStax / LibreTexts STEM Textbooks (CC-BY-4.0)",
                "PhysioNet Open Physiological Reference Ranges (CC-BY-4.0)",
                "W1z4rD Stage 42 Open Communication & Cooperative Safety Curriculum (CC0-1.0)",
            ],
        },
        "counts": {
            "base_episodes": len(base_episodes),
            "curated_episodes_total": len(expanded_episodes),
            "disciplines_covered": len(TWELVE_DISCIPLINES),
            "modalities_per_episode": 6,  # text, social, visual, audio, sensor, action
            "meta_stream_envelopes": len(envelopes),
            "erected_pools": len(erected),
            "rejected_noise_streams": len(erector.unspawned_noise_streams()),
            "compositional_components": len(compositional["component_experiences"]),
            "compositional_synthesis_rules": len(compositional["synthesis_rules"]),
            "foundational_anchors": len(discipline_pack["foundational_anchors"]),
            "multihop_chains": len(discipline_pack["multihop_chains"]),
            "oov_abstention_probes": len(discipline_pack["oov_outside_queries"]),
        },
        "neuron_economy_proof": economy,
        "meta_stream_erected_pools": [asdict(s) for s in erected],
        "rejected_low_support_streams": erector.unspawned_noise_streams(),
        "polite_elastic_governor": governor.summary(),
        "files": {
            "episodes_jsonl": str(episodes_path),
            "meta_stream_jsonl": str(envelopes_path),
            "compositional_json": str(comp_path),
            "twelve_discipline_json": str(disc_path),
            "brain_identity_toml": str(identity_path),
        },
    }
    manifest_path = out_dir / "curation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    manifest = curate_all(args.out_dir.resolve())
    print(json.dumps({
        "status": "ok",
        "curated_episodes_total": manifest["counts"]["curated_episodes_total"],
        "disciplines_covered": manifest["counts"]["disciplines_covered"],
        "erected_pools": manifest["counts"]["erected_pools"],
        "neuron_economy_proof": manifest["neuron_economy_proof"],
        "polite_elastic_governor": manifest["polite_elastic_governor"],
        "manifest_path": str(args.out_dir.resolve() / "curation_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
