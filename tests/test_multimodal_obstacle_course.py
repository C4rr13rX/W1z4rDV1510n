#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated test suite for multimodal curation, meta-stream pool erection,
polite elastic RAM/CPU scaling, validator honesty, and the 4 obstacle courses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from curate_multisensory_crossmodal_corpus import (  # noqa: E402
    TWELVE_DISCIPLINES,
    CanonicalMultisensoryCodec,
    MetaStreamPoolErector,
    PoliteElasticGovernor,
    build_raw_episodes,
    compute_neuron_economy_proof,
    curate_all,
)
from run_multimodal_obstacle_course import (  # noqa: E402
    LocalSimulatorBackend,
    find_brain_server_executable,
    run_course_1_crossmodal_hebbian_cofiring,
    run_obstacle_suite,
)


class _FakeVM:
    def __init__(self, avail_gib: float, total_gib: float = 32.0) -> None:
        self.available = int(avail_gib * (1024 ** 3))
        self.total = int(total_gib * (1024 ** 3))


def test_polite_elastic_governor_tiers_and_cushion() -> None:
    gov = PoliteElasticGovernor(min_free_gib=8.0, expand_gib=14.5, cushion_fraction=0.12)
    assert gov.priority_applied in {"BELOW_NORMAL_PRIORITY_CLASS", "POSIX_NICE_10"} or gov.priority_applied.startswith("fallback")

    # 1. Expanded tier (>= 14.5 GiB free)
    with patch("psutil.virtual_memory", return_value=_FakeVM(16.0)):
        s_exp = gov.sample()
        assert s_exp.tier == "expanded"
        assert s_exp.batch_size == 32
        assert s_exp.variants_per_episode == 3
        # (16.0 - 8.0) * 1024 * 0.88 = 7208.96 MiB cushioned headroom
        assert abs(s_exp.cushioned_headroom_mib - 7208.96) < 1.0

    # 2. Compact cushioned tier (between 8.0 GiB floor and 10.2 GiB effective)
    with patch("psutil.virtual_memory", return_value=_FakeVM(8.5)):
        s_cmp = gov.sample()
        assert s_cmp.tier == "compact_cushioned"
        assert s_cmp.batch_size == 8
        assert gov.gc_compactions >= 1

    # 3. Ultra-lean floor guard (< 8.0 GiB free -> shrinks to batch=2, never stops!)
    with patch("psutil.virtual_memory", return_value=_FakeVM(7.2)):
        s_lean = gov.sample()
        assert s_lean.tier == "ultra_lean_floor_guard"
        assert s_lean.batch_size == 2
        assert s_lean.cushioned_headroom_mib == 0.0
        assert gov.summary()["never_stopped_execution"] is True


def test_canonical_codec_zero_embellishment_and_neuron_economy() -> None:
    episodes = build_raw_episodes()
    assert len(episodes) == 36
    assert {ep["discipline"] for ep in episodes} == set(TWELVE_DISCIPLINES)

    for ep in episodes:
        v_atoms = CanonicalMultisensoryCodec.encode_visual(ep["visual"])
        a_atoms = CanonicalMultisensoryCodec.encode_audio(ep["audio"])
        s_atoms = CanonicalMultisensoryCodec.encode_social(ep["social"])
        t_atoms = CanonicalMultisensoryCodec.encode_sensor(ep["sensor"])

        assert CanonicalMultisensoryCodec.reconstruct_visual(v_atoms) == ep["visual"]
        assert CanonicalMultisensoryCodec.reconstruct_audio(a_atoms) == ep["audio"]
        assert CanonicalMultisensoryCodec.reconstruct_social(s_atoms) == ep["social"]
        assert CanonicalMultisensoryCodec.reconstruct_sensor(t_atoms) == ep["sensor"]

    economy = compute_neuron_economy_proof(episodes)
    assert economy["exact_canonical_reconstruction_rate"] == 1.0
    assert economy["zero_embellishment_rate"] == 1.0
    assert economy["neuron_reduction_factor_x"] >= 30.0


def test_meta_stream_pool_erector_discovery_and_noise_rejection() -> None:
    erector = MetaStreamPoolErector(
        spawn_support_threshold=3,
        novelty_threshold=0.28,
        max_dynamic_pools=16,
        max_neurons_per_pool=16,
        max_fanout_per_pool=4,
        prune_idle_secs=60,
    )
    for t in range(4):
        erector.observe_envelope("social:pragmatics:turn", [0.9, 0.1, 0.5], 100 + t)
        erector.observe_envelope("sensor:imu:6dof", [0.0, 0.0, 9.81], 100 + t)
        erector.observe_envelope("signal:rf:2g4", [2.4, -50.0, 0.8], 100 + t)

    # Low-support 1-shot noise must NOT erect a pool
    erector.observe_envelope("noise:random_spike:once", [42.0, -42.0], 105)

    erected = {s.pool_name: s for s in erector.erected_summaries()}
    assert {"social_pragmatics", "sensor_imu", "signal_rf"} == set(erected.keys())
    assert "noise_random_spike" in erector.unspawned_noise_streams()
    # Repeated identical vectors must deduplicate into 1 basis neuron per pool
    for s in erected.values():
        assert s.basis_neuron_count == 1
        assert len(s.co_occurring_pools) <= 4
        assert len(s.co_occurring_pools) == 2

    # Idle pruning reclaims stale dynamic pools cleanly
    pruned = erector.prune_idle(200)
    assert set(pruned) == {"social_pragmatics", "sensor_imu", "signal_rf"}


def test_validator_attribution_honesty_against_stub_candidate(tmp_path: Path) -> None:
    """Ensure obstacle validators fail cleanly (never pass or crash) on a stub brain."""
    curate_all(tmp_path)
    episodes = [
        json.loads(line)
        for line in (tmp_path / "crossmodal_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    class _StubHallucinatingBackend(LocalSimulatorBackend):
        def pretrain_bindings_polite(self, episodes_in: list) -> int:
            return len(episodes_in)  # no-op stub: learns nothing

        def predict(self, query_pool: int, frame: str, target_pool: int = 4) -> dict:
            return {"answer": "hallucinated_wrong_output", "outside_grounding": False, "confidence": 0.99, "known_atom_count": 1}

        def predict_multi(self, streams: list, target_pool: int = 4) -> dict:
            return {"answer": "hallucinated_wrong_output", "outside_grounding": False, "confidence": 0.99}

        def chat(self, text: str) -> dict:
            return {"reply": "hallucinated_wrong_output", "outside_grounding": False}

    gov = PoliteElasticGovernor()
    stub = _StubHallucinatingBackend(gov, list(range(1, 13)))
    verdict = run_course_1_crossmodal_hebbian_cofiring(episodes, ["untrained_oov?"], stub, gov)
    assert verdict["passed"] is False
    assert verdict["oov_abstention_honesty"]["honesty_rate"] == 0.0
    assert verdict["visual_plus_sensor_recall"]["hits"] == 0


def test_end_to_end_simulator_and_rust_brain(tmp_path: Path) -> None:
    data_dir = tmp_path / "mm_streams"
    sim_report_path = tmp_path / "sim_report.json"
    sim_report = run_obstacle_suite(
        mode="simulator",
        data_dir=data_dir,
        report_path=sim_report_path,
    )
    assert sim_report["all_passed"] is True
    assert sim_report["courses_passed"] == 4

    exe = find_brain_server_executable()
    if exe is not None:
        rust_report_path = tmp_path / "rust_report.json"
        rust_report = run_obstacle_suite(
            mode="local-rust-brain",
            executable=exe,
            data_dir=data_dir,
            report_path=rust_report_path,
        )
        assert rust_report["all_passed"] is True
        assert rust_report["courses_passed"] == 4
