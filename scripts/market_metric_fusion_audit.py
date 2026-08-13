#!/usr/bin/env python3
"""Build a provenance-preserving audit of historical market metric extrema."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "min_accuracy": "max",
    "min_balanced_accuracy": "max",
    "min_mcc": "max",
    "min_baseline_margin": "max",
    "min_coverage": "max",
    "min_acted_observations": "max",
    "min_expectancy": "max",
    "min_profit_factor": "max",
    "max_ece": "min",
    "max_drawdown": "min",
}


def _record(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    result = payload.get("result") or {}
    summary = result.get("summary") or {}
    if not summary:
        return None
    requested = max(1, int(result.get("requested_folds", 3)))
    evaluated = max(0, int(result.get("evaluated_folds", 0)))
    if evaluated < 1:
        return None
    return {
        "candidate_id": payload.get("genome_id") or path.stem,
        "candidate_path": str(path.resolve()),
        "evaluation_signature": result.get("evaluation_signature"),
        "evaluated_folds": evaluated,
        "requested_folds": requested,
        "full_retention": evaluated >= requested,
        "learner_kind": payload.get("learner_kind"),
        "generation": payload.get("generation"),
        "summary": summary,
    }


def load_records(state_dir: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    for path in sorted((state_dir / "candidates").glob("*.json")):
        try:
            record = _record(path, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            malformed += 1
            continue
        if record is not None:
            records.append(record)
    return records, malformed


def extrema(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    materialized = list(records)
    output: dict[str, dict[str, Any]] = {}
    for metric, direction in METRICS.items():
        eligible = []
        for record in materialized:
            value = record["summary"].get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                eligible.append((float(value), record))
        if not eligible:
            continue
        ordered = sorted(
            eligible,
            key=lambda item: (item[0], str(item[1]["candidate_id"])),
            reverse=direction == "max",
        )
        value, record = ordered[0]
        output[metric] = {
            "best_direction": direction,
            "value": value,
            **{key: record[key] for key in record if key != "summary"},
            "context_summary": record["summary"],
        }
    return output


def build_report(state_dir: Path) -> dict[str, Any]:
    records, malformed = load_records(state_dir)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    signatures = Counter(
        str(record["evaluation_signature"] or "legacy:none") for record in records
    )
    current_signature = state.get("dataset_signature")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_dir": str(state_dir.resolve()),
        "candidate_records": len(records),
        "malformed_candidate_files": malformed,
        "current_dataset_signature": current_signature,
        "signature_counts": dict(sorted(signatures.items())),
        "historical_extrema": extrema(records),
        "full_retention_extrema": extrema(
            record for record in records if record["full_retention"]
        ),
        "current_signature_extrema": extrema(
            record for record in records
            if record["evaluation_signature"] == current_signature
        ),
        "current_signature_full_retention_extrema": extrema(
            record for record in records
            if record["evaluation_signature"] == current_signature
            and record["full_retention"]
        ),
        "interpretation": [
            "Each extremum is one observed candidate; values are never combined into a synthetic score.",
            "Comparisons across evaluation signatures are historical context, not retained performance.",
            "Only full-retention extrema completed every requested protected fold.",
            "ECE and drawdown are minimized; all other listed metrics are maximized.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir", type=Path, default=ROOT / "runtime/market-evolution"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.state_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_records": report["candidate_records"],
        "current_dataset_signature": report["current_dataset_signature"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
