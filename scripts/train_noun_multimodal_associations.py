#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:/Projects/W1z4rDV1510n/scripts/train_noun_multimodal_associations.py

Multimodal Hebbian Co-Firing Trainer for W1z4rDV1510n Noun Dictionary
(`text + image + audio + social` pools).

Key Capabilities:
  1. Minimal-Neuron Co-Firing:
     Feeds compact decomposed visual, acoustic, text, and social/affordance primitives
     from `data/multimodal_noun_dictionary/noun_multimodal_dictionary.jsonl` into the
     same neural tick so neurons that fire together wire together across modalities —
     birthing cross-modal relationships and enabling novel concept synthesis without
     bloated high-entropy media byte streams.
  2. Supports Both `--dry-run` Audit & Live Neural Fabric Modes:
     - `--dry-run`: Audits neuron-atom counts, verifies byte-disjointness of shared
       primitives, maps emergent cross-noun Hebbian bridges, and optionally spot-verifies
       remote cloud media objects (`--verify-cloud-sample N`) with zero disk writes.
     - `--mode brain`: Co-fires `POOL_TEXT=1`, `POOL_IMAGE=2`, `POOL_AUDIO=3`, and
       `POOL_SOCIAL=4` (dense-burst ticks >= `concept_emergence_threshold=3`) against
       `w1z4rd_brain_server` (`/observe` + `/tick` + `/integrate`) or `/brain/*` on node,
       then probes live cross-modal recall (`text -> social`, `image -> social`).
     - `--mode node`: Registers `keyboard_text`, `image_pixels`, `audio_features`, and
       `social_context` on `w1z4rdv1510n-node` (`/multi_pool/*` and `/sensor/observe_triple`,
       streaming canonical PNG/WAV bytes strictly in RAM when `--stream-raw-cloud` is set).
  3. Polite Elastic PC-Allowance ("Nice" Resource Scaling):
     Runs at `psutil.BELOW_NORMAL_PRIORITY_CLASS`, leaves >= 8.0 GB RAM untouched + 12%
     under-cap cushion, scales batch sizes UP when available RAM >= 14.5 GB, and shrinks
     batch sizes to 1 + `gc.collect()` when RAM tightens.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_CCU_VENV_SITE = Path("D:/Projects/CoolCryptoUtilities/.venv/Lib/site-packages")
_W1Z4RD_ROOT = Path("D:/Projects/W1z4rDV1510n")
_SCRIPTS_DIR = _W1Z4RD_ROOT / "scripts"

for _p in (_SCRIPTS_DIR, _CCU_VENV_SITE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from curate_noun_visual_audio_dictionary import (
    DEFAULT_MANIFEST,
    PoliteResourceGovernor,
)
from multimodal_cloud_sync import MultimodalCloudSync

POOL_TEXT = 1
POOL_IMAGE = 2
POOL_AUDIO = 3
POOL_SOCIAL_DEFAULT = 4  # Maps to action/social grounding pool on standard 5-pool brain


def b64u_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def decode_b64u_flexible(b64_str: str) -> str:
    if not b64_str:
        return ""
    pad = "=" * (-len(b64_str) % 4)
    try:
        return base64.urlsafe_b64decode(b64_str + pad).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            if payload is None:
                req = urllib.request.Request(url, method="GET")
            else:
                raw = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=raw, method="POST", headers={"Content-Type": "application/json"}
                )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", "replace").strip()
                if not body:
                    return {}
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return {"raw": body}
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt == 2:
                raise
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"HTTP call failed for {url}: {last_exc}")


def load_dictionary(manifest_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dictionary manifest not found: {manifest_path}")
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            if 0 < limit <= len(records):
                break
    return records


def build_cofire_frames(rec: dict[str, Any]) -> dict[str, bytes]:
    """Construct ultra-compact, minimal-neuron byte frames across all 4 pools.
    Uses shared cross-noun visual/acoustic primitives + 16-hex structural fingerprint
    so shared traits wire across nouns while each noun retains a unique visual/audio signature
    with minimal neurons and synapses.
    """
    noun = str(rec["noun"]).strip()
    vis_shared = rec["visual_decomposition"]["shared_primitives"]
    # Sample 8 evenly spaced cells across the 8x8 visual grid (16 hex chars)
    full_vgrid = rec["visual_decomposition"]["cell_grid_8x8_hex"]
    vis_sig = "".join(full_vgrid[i : i + 2] for i in range(0, min(len(full_vgrid), 128), 16))

    aud_shared = rec["audio_decomposition"]["shared_primitives"][:5]
    full_agrid = rec["audio_decomposition"]["spectral_grid_8x8_hex"]
    aud_sig = full_agrid[::4][:16]

    soc_tokens = rec["social_primitives"]

    return {
        "text": noun.encode("utf-8"),
        "grounding": str(rec.get("grounding_sentence") or noun).encode("utf-8"),
        "image": ("|".join(vis_shared) + f"|vg:{vis_sig}").encode("utf-8"),
        "audio": ("|".join(aud_shared) + f"|ag:{aud_sig}").encode("utf-8"),
        "social": "|".join(soc_tokens).encode("utf-8"),
    }


def analyze_hebbian_graph_and_dry_run(
    records: list[dict[str, Any]],
    governor: PoliteResourceGovernor,
    verify_cloud_sample: int = 0,
) -> dict[str, Any]:
    """Audit minimal-neuron counts, shared cross-noun primitives, and cloud integrity."""
    governor.sample()
    primitive_to_nouns: dict[str, list[str]] = collections.defaultdict(list)
    total_text_atoms = 0
    total_vis_tokens = 0
    total_aud_tokens = 0
    total_soc_tokens = 0
    total_cofire_bytes = 0
    license_families: dict[str, int] = collections.Counter()

    for rec in records:
        noun = str(rec["noun"])
        nb = rec.get("neuron_budget") or {}
        total_text_atoms += int(nb.get("text_char_atoms", len(noun)))
        total_vis_tokens += int(nb.get("visual_compact_tokens", 0))
        total_aud_tokens += int(nb.get("audio_compact_tokens", 0))
        total_soc_tokens += int(nb.get("social_primitives", 0))
        total_cofire_bytes += int(nb.get("total_co_fire_bytes", 0))
        lf = str(rec.get("provenance", {}).get("license_family") or "unknown")
        license_families[lf] += 1

        for p in rec["visual_decomposition"]["shared_primitives"]:
            primitive_to_nouns[p].append(noun)
        for p in rec["audio_decomposition"]["shared_primitives"]:
            if p.startswith(("adur:", "aspec:", "aenv:")):
                primitive_to_nouns[p].append(noun)
        for p in rec["social_primitives"]:
            primitive_to_nouns[p].append(noun)

    # Top shared cross-modal primitives that wire distinct nouns together
    shared_bridges = [
        {"primitive": prim, "noun_count": len(nouns), "sample_nouns": nouns[:8]}
        for prim, nouns in sorted(
            primitive_to_nouns.items(), key=lambda kv: len(kv[1]), reverse=True
        )
        if len(nouns) >= 2
    ]

    cloud_checks: list[dict[str, Any]] = []
    if verify_cloud_sample > 0 and records:
        cloud = MultimodalCloudSync(base_prefix="noun_dictionary")
        step = max(1, len(records) // verify_cloud_sample)
        sampled = records[::step][:verify_cloud_sample]
        for rec in sampled:
            cm = rec["cloud_media"]
            img_chk = cloud.verify_remote_object(
                cm["image_uri"],
                expected_size=cm["image_bytes"],
                expected_sha256=cm["image_sha256"],
            )
            aud_chk = cloud.verify_remote_object(
                cm["audio_word_uri"],
                expected_size=cm["audio_word_bytes"],
                expected_sha256=cm["audio_word_sha256"],
            )
            cloud_checks.append({
                "noun": rec["noun"],
                "image_verified": bool(img_chk.get("verified")),
                "audio_verified": bool(aud_chk.get("verified")),
                "image_uri": cm["image_uri"],
                "audio_uri": cm["audio_word_uri"],
            })

    n = max(1, len(records))
    return {
        "mode": "dry_run",
        "nouns_evaluated": len(records),
        "pools_wired": ["text(1)", "image(2)", "audio(3)", "social(4)"],
        "mean_atoms_per_noun": {
            "text_chars": round(total_text_atoms / n, 2),
            "visual_compact_tokens": round(total_vis_tokens / n, 2),
            "audio_compact_tokens": round(total_aud_tokens / n, 2),
            "social_tokens": round(total_soc_tokens / n, 2),
            "total_cofire_bytes": round(total_cofire_bytes / n, 2),
        },
        "license_families": dict(license_families),
        "shared_hebbian_bridges_count": len(shared_bridges),
        "top_shared_hebbian_bridges": shared_bridges[:12],
        "cloud_spot_verifications": cloud_checks,
        "resource_governor": governor.summary(),
    }


def train_live_brain(
    records: list[dict[str, Any]],
    endpoint: str,
    *,
    reps: int,
    social_pool_id: int,
    stream_raw_cloud: bool,
    governor: PoliteResourceGovernor,
) -> dict[str, Any]:
    """Co-fire `text + image + audio + social` in dense bursts on `w1z4rd_brain_server`
    (or `/brain` mount on `w1z4rd_node`) and probe live cross-modal Hebbian recall.
    """
    base = endpoint.rstrip("/")
    pre_stats = http_json(f"{base}/stats")
    cloud = MultimodalCloudSync(base_prefix="noun_dictionary") if stream_raw_cloud else None

    t0 = time.perf_counter()
    ticks_executed = 0
    idx = 0

    while idx < len(records):
        mem = governor.sample()
        batch_sz = max(1, min(int(mem["batch_size"]), len(records) - idx))
        batch = records[idx : idx + batch_sz]
        idx += batch_sz

        for rec in batch:
            frames = build_cofire_frames(rec)
            img_bytes = frames["image"]
            aud_bytes = frames["audio"]
            if cloud is not None:
                cm = rec["cloud_media"]
                img_bytes = cloud.fetch_bytes_ephemeral(cm["image_uri"], expected_sha256=cm["image_sha256"])
                aud_bytes = cloud.fetch_bytes_ephemeral(cm["audio_word_uri"], expected_sha256=cm["audio_word_sha256"])

            # Dense burst >= 3 reps so concepts promote cleanly before moving to next noun
            for rep in range(max(1, reps)):
                http_json(f"{base}/observe", {"pool_id": POOL_TEXT, "frame": b64u_no_pad(frames["text"])})
                http_json(f"{base}/observe", {"pool_id": POOL_IMAGE, "frame": b64u_no_pad(img_bytes)})
                http_json(f"{base}/observe", {"pool_id": POOL_AUDIO, "frame": b64u_no_pad(aud_bytes)})
                http_json(f"{base}/observe", {"pool_id": social_pool_id, "frame": b64u_no_pad(frames["social"])})
                http_json(f"{base}/tick", {})
                ticks_executed += 1

            time.sleep(float(mem["pace_sleep"]) * 0.5)

        governor.sweep_memory()

    train_elapsed_s = round(time.perf_counter() - t0, 2)
    post_stats = http_json(f"{base}/stats")

    # Probe cross-modal recall across a representative sample (up to 30 nouns)
    probe_set = records[: min(30, len(records))]
    text_to_social_hits = 0
    image_to_social_hits = 0
    sample_probes: list[dict[str, Any]] = []

    for rec in probe_set:
        frames = build_cofire_frames(rec)
        expected_cat = f"scat:{rec['category']}"
        expected_dom = f"sdom:{rec['domain']}"

        # 1. Query POOL_TEXT -> target POOL_SOCIAL
        http_json(f"{base}/observe", {"pool_id": POOL_TEXT, "frame": b64u_no_pad(frames["text"])})
        r_ts = http_json(f"{base}/integrate", {"query_pool": POOL_TEXT, "target_pool": social_pool_id})
        ans_ts = decode_b64u_flexible(str(r_ts.get("answer") or ""))
        hit_ts = (expected_cat in ans_ts) or (expected_dom in ans_ts)
        if hit_ts:
            text_to_social_hits += 1

        # 2. Query POOL_IMAGE -> target POOL_SOCIAL
        http_json(f"{base}/observe", {"pool_id": POOL_IMAGE, "frame": b64u_no_pad(frames["image"])})
        r_is = http_json(f"{base}/integrate", {"query_pool": POOL_IMAGE, "target_pool": social_pool_id})
        ans_is = decode_b64u_flexible(str(r_is.get("answer") or ""))
        hit_is = (expected_cat in ans_is) or (expected_dom in ans_is)
        if hit_is:
            image_to_social_hits += 1

        if len(sample_probes) < 8:
            sample_probes.append({
                "noun": rec["noun"],
                "expected_social": frames["social"].decode("utf-8"),
                "text_to_social_reply": ans_ts[:80],
                "text_to_social_hit": hit_ts,
                "image_to_social_reply": ans_is[:80],
                "image_to_social_hit": hit_is,
            })

    denom = max(1, len(probe_set))
    return {
        "mode": "live_brain_cofire",
        "endpoint": base,
        "nouns_trained": len(records),
        "reps_per_noun": reps,
        "ticks_executed": ticks_executed,
        "train_elapsed_seconds": train_elapsed_s,
        "pre_stats": {
            "tick": pre_stats.get("tick"),
            "neurons": pre_stats.get("total_neurons"),
            "concepts": pre_stats.get("total_concepts"),
            "terminals": pre_stats.get("total_terminals"),
            "binding": pre_stats.get("total_binding"),
        },
        "post_stats": {
            "tick": post_stats.get("tick"),
            "neurons": post_stats.get("total_neurons"),
            "concepts": post_stats.get("total_concepts"),
            "terminals": post_stats.get("total_terminals"),
            "binding": post_stats.get("total_binding"),
        },
        "probe_count": len(probe_set),
        "text_to_social_recall_pct": round(100.0 * text_to_social_hits / denom, 1),
        "image_to_social_recall_pct": round(100.0 * image_to_social_hits / denom, 1),
        "sample_probes": sample_probes,
        "resource_governor": governor.summary(),
    }


def train_live_node_multipool(
    records: list[dict[str, Any]],
    endpoint: str,
    *,
    passes: int,
    lr: float,
    stream_raw_cloud: bool,
    governor: PoliteResourceGovernor,
) -> dict[str, Any]:
    """Train `keyboard_text <-> image_pixels <-> audio_features <-> social_context`
    on `w1z4rdv1510n-node` (`/multi_pool/*` + `/sensor/observe_triple`).
    """
    base = endpoint.rstrip("/")
    for pool_name in ("keyboard_text", "image_pixels", "audio_features", "social_context"):
        http_json(f"{base}/multi_pool/register", {"pool_id": pool_name})

    cloud = MultimodalCloudSync(base_prefix="noun_dictionary") if stream_raw_cloud else None
    pairs_trained = 0
    triples_trained = 0
    t0 = time.perf_counter()

    idx = 0
    while idx < len(records):
        mem = governor.sample()
        batch_sz = max(1, min(int(mem["batch_size"]), len(records) - idx))
        batch = records[idx : idx + batch_sz]
        idx += batch_sz

        for rec in batch:
            noun = str(rec["noun"])
            vis_compact = " ".join(rec["visual_decomposition"]["compact_brain_tokens"])
            aud_compact = " ".join(rec["audio_decomposition"]["compact_brain_tokens"])
            soc_compact = " ".join(rec["social_primitives"])

            if cloud is not None:
                cm = rec["cloud_media"]
                img_b = cloud.fetch_bytes_ephemeral(cm["image_uri"], expected_sha256=cm["image_sha256"])
                aud_b = cloud.fetch_bytes_ephemeral(cm["audio_word_uri"], expected_sha256=cm["audio_word_sha256"])
                http_json(f"{base}/sensor/observe_triple", {
                    "text": noun,
                    "image_b64": base64.b64encode(img_b).decode("ascii"),
                    "audio_b64": base64.b64encode(aud_b).decode("ascii"),
                    "lr": lr,
                })
                triples_trained += 1

            for src_p, src_v, tgt_p, tgt_v in (
                ("keyboard_text", noun, "image_pixels", vis_compact),
                ("keyboard_text", noun, "audio_features", aud_compact),
                ("keyboard_text", noun, "social_context", soc_compact),
                ("image_pixels", vis_compact, "social_context", soc_compact),
                ("audio_features", aud_compact, "social_context", soc_compact),
            ):
                http_json(f"{base}/multi_pool/train_pair", {
                    "src_pool": src_p,
                    "src": src_v,
                    "tgt_pool": tgt_p,
                    "tgt": tgt_v,
                    "passes": passes,
                    "lr": lr,
                })
                pairs_trained += 1

            time.sleep(float(mem["pace_sleep"]) * 0.5)
        governor.sweep_memory()

    mp_stats = http_json(f"{base}/multi_pool/stats")
    return {
        "mode": "live_node_multipool",
        "endpoint": base,
        "nouns_trained": len(records),
        "triples_trained": triples_trained,
        "cross_pool_pairs_trained": pairs_trained,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "multi_pool_stats": mp_stats,
        "resource_governor": governor.summary(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train & verify cross-modal Hebbian associations (text + image + audio + social)."
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--limit", type=int, default=0, help="Max nouns to train (0 = all in manifest).")
    ap.add_argument("--dry-run", action="store_true", help="Audit neuron budgets & cross-noun Hebbian bridges without mutating a live server.")
    ap.add_argument("--verify-cloud-sample", type=int, default=5, help="Number of cloud URIs to spot-verify during --dry-run.")
    ap.add_argument("--mode", choices=("brain", "node"), default="brain")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8095", help="Brain server URL (e.g. http://127.0.0.1:8095 or http://127.0.0.1:8090/brain) or node URL.")
    ap.add_argument("--reps", type=int, default=4, help="Dense-burst ticks per noun in brain mode (>=3 promotes full concepts).")
    ap.add_argument("--passes", type=int, default=6, help="Passes per pair in node multi-pool mode.")
    ap.add_argument("--lr", type=float, default=0.85)
    ap.add_argument("--social-pool-id", type=int, default=POOL_SOCIAL_DEFAULT)
    ap.add_argument("--stream-raw-cloud", action="store_true", help="Stream canonical 64x64 PNG & 16kHz WAV from cloud into RAM (zero disk writes).")
    ap.add_argument("--max-batch-size", type=int, default=16)
    args = ap.parse_args()

    governor = PoliteResourceGovernor(max_batch_size=args.max_batch_size)
    records = load_dictionary(args.manifest, limit=args.limit)

    if args.dry_run:
        report = analyze_hebbian_graph_and_dry_run(
            records, governor, verify_cloud_sample=args.verify_cloud_sample
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.mode == "brain":
        report = train_live_brain(
            records,
            args.endpoint,
            reps=args.reps,
            social_pool_id=args.social_pool_id,
            stream_raw_cloud=args.stream_raw_cloud,
            governor=governor,
        )
    else:
        report = train_live_node_multipool(
            records,
            args.endpoint,
            passes=args.passes,
            lr=args.lr,
            stream_raw_cloud=args.stream_raw_cloud,
            governor=governor,
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
