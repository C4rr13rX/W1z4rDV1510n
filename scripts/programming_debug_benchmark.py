#!/usr/bin/env python3
"""Execution-first benchmark for the multi-pool coding-debug brain."""
from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from dataclasses import replace
from pathlib import Path

from programming_debug_episode_train import Case, CASES, classify, environment, execute, generate

CAUSAL_FIELDS = {1: "instruction", 2: "source_before", 3: "console_before",
                 5: "environment", 6: "failure_outcome", 10: "source_before"}

PARAPHRASES = [
    "Make square compute the product of its argument with itself.",
    "Correct is_negative to recognize numbers less than zero.",
    "Prevent avg_list from failing when it receives no elements; return zero.",
    "Change filter_odd so only integers with odd parity remain.",
    "Repair word_freq so repeated words increase their stored total.",
]

VARIANTS = [
    replace(CASES[0], args=[11], expected=121),
    replace(CASES[1], args=[4], expected=False),
    replace(CASES[2], args=[[2, 4, 6]], expected=4),
    replace(CASES[3], args=[[6, 7, 8, 9]], expected=[7, 9]),
    replace(CASES[4], args=["x x y x"], expected={"x": 3, "y": 1}),
]

STRUCTURAL_TRANSFER = [
    Case("renamed_square", "Correct power_two so it multiplies the value by itself.",
         "def power_two(value):\n    return value + value",
         "def power_two(value):\n    return value * value",
         "power_two", [9], 81, "operator_replacement"),
    Case("renamed_negative", "Correct below_zero so it recognizes values less than zero.",
         "def below_zero(value):\n    return value > 0",
         "def below_zero(value):\n    return value < 0",
         "below_zero", [-8], True, "comparison_direction"),
    Case("renamed_average", "Make mean_values safely return zero for an empty sequence.",
         "def mean_values(values):\n    return sum(values) / len(values)",
         "def mean_values(values):\n    return sum(values) / len(values) if values else 0",
         "mean_values", [[]], 0, "empty_input_guard"),
]

UNSUPPORTED = [
    Case("cube_unsupported", "Fix cube so it multiplies the value three times.",
         "def cube(value):\n    return value + value",
         "def cube(value):\n    return value * value * value",
         "cube", [3], 27, "unsupported_cube"),
    Case("uppercase_unsupported", "Fix shout so it returns uppercase text.",
         "def shout(text):\n    return text",
         "def shout(text):\n    return text.upper()",
         "shout", ["hello"], "HELLO", "unsupported_method"),
    Case("maximum_unsupported", "Fix maximum so it returns the largest value.",
         "def maximum(values):\n    return min(values)",
         "def maximum(values):\n    return max(values)",
         "maximum", [[2, 9, 4]], 9, "unsupported_builtin"),
]


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def predict(endpoint: str, streams: dict[int, str]) -> str | None:
    payload = {"streams": [{"pool_id": pool, "frame": b64(value)}
                           for pool, value in streams.items()], "target_pool": 4}
    request = urllib.request.Request(endpoint.rstrip("/") + "/brain/predict/multi",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())
    answer = result.get("answer")
    return base64.urlsafe_b64decode(answer + "===").decode() if answer else None


def predict_composed(endpoint: str, source: str, streams: dict[int, str]) -> str | None:
    payload = {"source": b64(source), "relation_pool": 11,
               "streams": [{"pool_id": pool, "frame": b64(value)}
                           for pool, value in streams.items()]}
    request = urllib.request.Request(endpoint.rstrip("/") + "/brain/repair/predict",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())
    answer = result.get("answer")
    return base64.urlsafe_b64decode(answer + "===").decode() if answer else None


def streams_for(case: Case, instruction: str | None = None) -> dict[int, str]:
    before = execute(case.broken, case)
    values = {
        "instruction": instruction or case.instruction,
        "source_before": case.broken,
        "console_before": json.dumps(before, sort_keys=True),
        "environment": environment(),
        "failure_outcome": json.dumps(
            {"status": "failure", "error_class": classify(before)}, sort_keys=True),
    }
    return {pool: values[field] for pool, field in CAUSAL_FIELDS.items()}


def evaluate_case(endpoint: str, case: Case, instruction: str | None = None,
                  exact_reference: bool = False, compose: bool = False) -> dict:
    streams = streams_for(case, instruction)
    answer = (predict_composed(endpoint, case.broken, streams) if compose
              else predict(endpoint, streams))
    result = execute(answer, case) if answer else None
    return {
        "id": case.name,
        "answered": answer is not None,
        "exact": answer == case.corrected if exact_reference else None,
        "execution_pass": bool(result and result["return_code"] == 0),
        "answer": answer,
    }


def main() -> int:
    global CAUSAL_FIELDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18310")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/debug_benchmark.json"))
    parser.add_argument("--evidence-pools", default=",".join(map(str, CAUSAL_FIELDS)),
                        help="comma-separated causal pool ids used at inference")
    args = parser.parse_args()
    selected = {int(value) for value in args.evidence_pools.split(",") if value.strip()}
    CAUSAL_FIELDS = {pool: field for pool, field in CAUSAL_FIELDS.items() if pool in selected}
    if not CAUSAL_FIELDS:
        parser.error("at least one known evidence pool is required")
    exact_cases = [Case(row["id"], row["instruction"], row["source_before"],
                        row["corrected_source"], case.function, case.args,
                        case.expected, case.repair_kind)
                   for row, case in zip(generate(), CASES)]
    exact = [evaluate_case(args.endpoint, case, exact_reference=True)
             for case in exact_cases]
    variation = [evaluate_case(args.endpoint, case, prompt)
                 for case, prompt in zip(VARIANTS, PARAPHRASES)]
    transfer = [evaluate_case(args.endpoint, case, compose=True)
                for case in STRUCTURAL_TRANSFER]
    unsupported = [evaluate_case(args.endpoint, case, compose=True)
                   for case in UNSUPPORTED]
    report = {
        "exact": {"passed": sum(row["exact"] for row in exact), "total": len(exact),
                  "rows": exact},
        "heldout_execution": {"passed": sum(row["execution_pass"] for row in variation),
                              "total": len(variation), "rows": variation},
        "structural_transfer": {"passed": sum(row["execution_pass"] for row in transfer),
                                "total": len(transfer), "rows": transfer},
        "oov_honesty": {"passed": sum(not row["answered"] for row in unsupported),
                        "total": len(unsupported), "rows": unsupported},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: {"passed": value["passed"], "total": value["total"]}
                      for key, value in report.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
