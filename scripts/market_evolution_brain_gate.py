#!/usr/bin/env python3
"""Validate one evolved genome in fresh, isolated Wizard micro-brains."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_brain_experiment import (  # noqa: E402
    BrainClient, direction, evaluate_rows, parse_prediction,
)
from scripts.market_evolution_service import (  # noqa: E402
    DERIVATIVE_FEATURES, FEATURE_GROUPS, FLOOR, Genome, evaluation_scope,
    genome_from_dict, genome_uses_derivatives, load_dataset_cached, passes_floor,
    program_name, program_value,
)

POOL_HORIZON = 9
POOL_INSTRUMENT = 10
POOL_OUTCOME = 11
FEATURE_POOLS = {
    "price": 15,
    "flow": 16,
    "cross": 17,
    "derivatives": 18,
    "breadth": 19,
    "news": 20,
}
POOL_EVOLVED = 21


def quantize(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    magnitude = abs(value)
    if magnitude < 1e-8:
        return "zero"
    sign = "p" if value > 0 else "n"
    exponent = math.floor(math.log10(magnitude))
    mantissa = round(magnitude / (10 ** exponent), 1)
    return f"{sign}{mantissa:g}e{exponent}"


def feature_frame(row: dict[str, Any], names: Sequence[str]) -> str:
    return " ".join(f"{name}={quantize(float(row['features'].get(name, 0.0)))}"
                    for name in names)


def genome_feature_frame(row: dict[str, Any], genome: Genome) -> str:
    base = feature_frame(row, genome.features)
    evolved = " ".join(
        f"{program_name(program)}={quantize(program_value(row['features'], program))}"
        for program in genome.feature_programs
    )
    return " ".join(part for part in (base, evolved) if part)


def feature_family(name: str) -> str:
    """Route one causal feature to an independently firing sensory pool."""
    if name in DERIVATIVE_FEATURES:
        return "derivatives"
    if "news_" in name or name.startswith(("asset_news", "global_news")):
        return "news"
    for family in ("flow", "breadth", "cross", "price"):
        if any(name == source or name.endswith("_" + source)
               for source in FEATURE_GROUPS[family]):
            return family
    return "price"


def streams(row: dict[str, Any], genome: Genome, horizon: int) -> list[tuple[int, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in genome.features:
        grouped[feature_family(name)].append(name)
    result = [
        (FEATURE_POOLS[family], feature_frame(row, names))
        for family, names in sorted(grouped.items(), key=lambda item: FEATURE_POOLS[item[0]])
        if names
    ]
    evolved = " ".join(
        f"{program_name(program)}={quantize(program_value(row['features'], program))}"
        for program in genome.feature_programs
    )
    if evolved:
        result.append((POOL_EVOLVED, evolved))
    result.extend([
        (POOL_HORIZON, f"horizon_bars={horizon}"),
        (POOL_INSTRUMENT, f"base={str(row['asset']).lower()} market=crypto"),
    ])
    return result


def render_identity(template: Path, destination: Path, genome: Genome) -> None:
    text = template.read_text(encoding="utf-8")
    text = re.sub(r"binding_emergence_threshold = \d+",
                  f"binding_emergence_threshold = {genome.binding_threshold}", text, count=1)
    # Evolve sensory concept emergence while preserving the outcome pool's
    # deliberately lower supervised-action threshold.
    blocks = text.split("[[pools]]")
    for index in range(1, len(blocks)):
        if 'kind = "SensoryInput"' in blocks[index]:
            blocks[index] = re.sub(r"concept_emergence_threshold = \d+",
                                   f"concept_emergence_threshold = {genome.concept_threshold}",
                                   blocks[index])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("[[pools]]".join(blocks), encoding="utf-8")


def wait_health(endpoint: str, process: subprocess.Popen, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"brain server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(.25)
    raise TimeoutError("brain server did not become healthy")


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def settle_brain(client: BrainClient) -> dict[str, Any]:
    """Create the neuron-addressable rest boundary required before evaluation."""
    before = client.get("/brain/stats")
    sleep = client.post("/brain/sleep", {
        # Serialize without pruning fresh low-use neurons.  Evaluation must
        # demand-page the learned fabric instead of inheriting train residency.
        "min_use_count": 0,
        "stale_ticks": 9_223_372_036_854_775_807,
    })
    if sleep.get("error"):
        raise RuntimeError(f"brain settlement failed: {sleep}")
    checkpoint = client.post("/brain/checkpoint", {})
    if checkpoint.get("ok") is False:
        raise RuntimeError(f"settled checkpoint failed: {checkpoint}")
    after = client.get("/brain/stats")
    resident = int(after.get("resident_terminals") or 0)
    if resident != 0:
        raise RuntimeError(
            f"settled brain retained {resident} terminals before evaluation"
        )
    return {"before": before, "sleep": sleep, "checkpoint": checkpoint, "after": after}


def stop_brain_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def retain_attempt(passed: bool, retain_failed: bool) -> bool:
    return passed or retain_failed


def evenly_spaced(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    return [rows[round(position * (len(rows) - 1) / (count - 1))]
            for position in range(count)] if count > 1 else [rows[-1]]


def select_per_asset(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["asset"]].append(row)
    return sorted((row for asset_rows in grouped.values()
                   for row in evenly_spaced(asset_rows, count)),
                  key=lambda row: (row["timestamp"], row["asset"]))


def predict_rows(client: BrainClient, rows: Sequence[dict[str, Any]], genome: Genome,
                 horizon: int, confidence_floor: float = 0.0) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        answer, confidence, latency = client.predict(streams(row, genome, horizon))
        prediction = parse_prediction(answer)
        if confidence < confidence_floor:
            prediction = None
        result.append({
            "asset": row["asset"], "timestamp": row["timestamp"],
            "actual": "updraft" if row["target"] > 0 else "downshift",
            "predicted": prediction, "return": row["return"],
            "confidence": confidence, "latency_seconds": latency,
            "momentum_direction": 1 if row["features"]["r12"] > 0 else -1,
        })
    return result


def add_baselines(metrics: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    # Normalize the shared evaluator's descriptive key to the perpetual GA's
    # admission-contract key. Without this alias, an otherwise valid neural
    # fold can never satisfy the minimum-action gate.
    metrics["acted_observations"] = int(metrics.get("acted_directional_n", 0))
    actual = [direction(row["actual"]) for row in rows]
    momentum = [row["momentum_direction"] for row in rows]
    direct = statistics.fmean(a == m for a, m in zip(actual, momentum)) if rows else 0.0
    inverse = statistics.fmean(a == -m for a, m in zip(actual, momentum)) if rows else 0.0
    metrics.update({"best_baseline_accuracy": max(direct, inverse),
                    "baseline_margin": metrics["directional_accuracy"] - max(direct, inverse)})


def run_fold(fold: int, cutoff: float, genome: Genome, dataset: dict[str, Any], args,
             identity: Path) -> dict[str, Any]:
    fold_root = args.attempt_root / f"fold-{fold}"
    if fold_root.exists() and any(fold_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite micro-brain {fold_root}")
    fold_root.mkdir(parents=True, exist_ok=True)
    uses_derivatives = genome_uses_derivatives(genome)
    eligible = dataset["supplemental_assets"] if uses_derivatives else set(dataset["assets"])
    training_assets, holdout_assets, _ = evaluation_scope(
        dataset, set(eligible), "market-perpetual-v1"
    )
    test_seconds = args.test_days * 86400
    calibration_seconds = args.calibration_days * 86400
    train_stop = cutoff - args.horizon * 3600
    calibration_start = train_stop - calibration_seconds
    train = [row for row in dataset["rows"] if row["asset"] in training_assets
             and row["asset"] in eligible and row["timestamp"] < calibration_start]
    calibration = [row for row in dataset["rows"] if row["asset"] in training_assets
                   and row["asset"] in eligible
                   and calibration_start <= row["timestamp"] < train_stop]
    known = [row for row in dataset["rows"] if row["asset"] in training_assets
             and row["asset"] in eligible and cutoff <= row["timestamp"] < cutoff + test_seconds]
    unseen = [row for row in dataset["rows"] if row["asset"] in holdout_assets
              and row["asset"] in eligible and cutoff <= row["timestamp"] < cutoff + test_seconds]
    train = select_per_asset(train, args.train_per_asset)
    calibration = select_per_asset(calibration, args.calibration_per_asset)
    known = select_per_asset(known, args.test_per_asset)
    unseen = select_per_asset(unseen, args.test_per_asset)
    port = args.port + fold if args.port else available_port()
    environment = os.environ.copy()
    environment.update({
        "W1Z4RD_NODE_BRAIN_DIR": str(fold_root.resolve()),
        "W1Z4RD_BRAIN_IDENTITY": str(identity.resolve()),
        "W1Z4RD_BRAIN_PORT": str(port), "W1Z4RD_BRAIN_BIND": "127.0.0.1",
        "W1Z4RD_TIER_MIN_SYS_AVAIL_MB": "4096",
    })
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (fold_root / "stdout.log").open("wb") as stdout, \
            (fold_root / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen([str(args.binary.resolve())], cwd=ROOT, env=environment,
                                   stdout=stdout, stderr=stderr, creationflags=creationflags)
        try:
            endpoint = f"http://127.0.0.1:{port}"
            wait_health(endpoint, process)
            client = BrainClient(endpoint, timeout=90)
            failures = 0
            for row in train:
                outcome = "future updraft" if row["target"] > 0 else "future downshift"
                for _ in range(genome.presentations):
                    if not client.consolidate(streams(row, genome, args.horizon), outcome):
                        failures += 1
            # Current nodes create a neuron-addressable container before their
            # first observation, so they can settle in-place without writing a
            # second monolithic copy.  Only old binaries/checkpoints need the
            # compatibility checkpoint and one-time migration.
            migration_mode = "native_neuron_addressable"
            checkpoint = {"skipped": "native .wbrain persists at idle boundary"}
            if (fold_root / "brain.wbrain").is_file():
                settlement = settle_brain(client)
            else:
                checkpoint = client.post("/brain/checkpoint", {})
                if checkpoint.get("ok") is False:
                    raise RuntimeError(f"pre-migration checkpoint failed: {checkpoint}")
                stop_brain_process(process)
                migration = subprocess.run(
                    [str(args.migration_binary.resolve()), str(fold_root.resolve())],
                    cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                    creationflags=creationflags, timeout=900,
                )
                if migration.returncode != 0:
                    raise RuntimeError(
                        f"brain migration failed with exit {migration.returncode}"
                    )
                migration_mode = "legacy_to_neuron_addressable"
                process = subprocess.Popen(
                    [str(args.binary.resolve())], cwd=ROOT, env=environment,
                    stdout=stdout, stderr=stderr, creationflags=creationflags,
                )
                wait_health(endpoint, process)
                client = BrainClient(endpoint, timeout=90)
                settlement = settle_brain(client)
            settlement["pre_migration_checkpoint"] = checkpoint
            settlement["storage_migration"] = migration_mode
            calibration_predictions = predict_rows(client, calibration, genome, args.horizon)
            confidences = sorted(row["confidence"] for row in calibration_predictions)
            threshold_index = round((len(confidences) - 1) * genome.confidence_quantile)
            confidence_floor = confidences[threshold_index] if confidences else 0.0
            sections = {}
            for name, selected in (("known_asset_future", known), ("unseen_asset_future", unseen)):
                predicted = predict_rows(client, selected, genome, args.horizon, confidence_floor)
                section_metrics = evaluate_rows(predicted, args.cost_bps)
                add_baselines(section_metrics, predicted)
                sections[name] = {"metrics": section_metrics, "rows": predicted}
            return {
                "fold": fold, "cutoff": cutoff, "training_rows": len(train),
                "presentations": genome.presentations, "training_failures": failures,
                "settlement": settlement,
                "calibration_rows": len(calibration), "confidence_floor": confidence_floor,
                "sections": sections,
            }
        finally:
            stop_brain_process(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runtime/benchmarks/market-corpus-manifest.json")
    parser.add_argument("--supplemental-root", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\binance_public"))
    parser.add_argument("--news", type=Path,
                        default=Path(r"D:\Projects\CoolCryptoUtilities\data\news\historical_deduplicated.json"))
    parser.add_argument("--identity-template", type=Path,
                        default=ROOT / "brains/market_predictor_evolution.identity.toml")
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "target/debug/w1z4rd_brain_server.exe")
    parser.add_argument("--migration-binary", type=Path,
                        default=ROOT / "target/debug/w1z4rd_brain_migrate.exe")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime/market-evolution/brain-gates")
    parser.add_argument("--dataset-cache", type=Path,
                        default=ROOT / "runtime/cache/market-evolution-dataset-v3.joblib")
    parser.add_argument("--port", type=int, default=0,
                        help="base port; zero reserves a free loopback port per fold")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--final-days", type=int, default=21)
    parser.add_argument("--train-per-asset", type=int, default=16)
    parser.add_argument("--calibration-per-asset", type=int, default=8)
    parser.add_argument("--test-per-asset", type=int, default=40)
    parser.add_argument("--cost-bps", type=float, default=20)
    parser.add_argument("--retain-failed-runtime", action="store_true",
                        help="keep the generated microbrain even when its gate fails")
    args = parser.parse_args()
    payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    genome = genome_from_dict(payload).finalize()
    attempt = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S-%fZ")
    args.attempt_root = args.runtime / genome.genome_id / attempt
    dataset = load_dataset_cached(
        args.manifest, args.supplemental_root, args.horizon, args.stride,
        "market-perpetual-v1", args.news, args.dataset_cache,
    )
    identity = args.attempt_root / "identity.toml"
    render_identity(args.identity_template, identity, genome)
    test_seconds = args.test_days * 86400
    final_seconds = args.final_days * 86400
    uses_derivatives = genome_uses_derivatives(genome)
    eligible = dataset["supplemental_assets"] if uses_derivatives else set(dataset["assets"])
    _, _, evaluation_end = evaluation_scope(dataset, set(eligible), "market-perpetual-v1")
    cutoffs = [evaluation_end - final_seconds - (args.folds - fold) * test_seconds
               for fold in range(args.folds)]
    started = datetime.now(timezone.utc).isoformat()
    folds = [run_fold(index, cutoff, genome, dataset, args, identity)
             for index, cutoff in enumerate(cutoffs)]
    all_sections = [fold["sections"][name]["metrics"] for fold in folds
                    for name in ("known_asset_future", "unseen_asset_future")]
    passed = args.folds >= 3 and all(passes_floor(section) for section in all_sections)
    report = {
        "candidate": str(args.candidate.resolve()), "genome_id": genome.genome_id,
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "stage": ("isolated_wizard_full_gate" if args.folds >= 3
                  else "isolated_wizard_smoke_gate"), "folds": folds,
        "attempt_runtime": str(args.attempt_root),
        "all_brain_floor_gates": passed,
        "promotion": "eligible_for_untouched_final" if passed else "quarantined",
    }
    keep_runtime = retain_attempt(passed, args.retain_failed_runtime)
    report["attempt_runtime_retained"] = keep_runtime
    if not keep_runtime:
        try:
            shutil.rmtree(args.attempt_root)
        except OSError as error:
            report["attempt_runtime_retained"] = True
            report["attempt_cleanup_error"] = repr(error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"genome_id": genome.genome_id, "passed": passed,
                      "promotion": report["promotion"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
