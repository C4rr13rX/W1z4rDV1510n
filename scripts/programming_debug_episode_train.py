#!/usr/bin/env python3
"""Generate and optionally train real failure->repair->success code episodes."""
from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_FIELDS = {
    1: "instruction",
    2: "source_before",
    3: "console_before",
    5: "environment",
    6: "failure_outcome",
    7: "repair_delta",
    8: "console_after",
    9: "resolution",
    10: "source_before",
}
ACTION_POOL = 4


@dataclass(frozen=True)
class Case:
    name: str
    instruction: str
    broken: str
    corrected: str
    function: str
    args: list
    expected: object
    repair_kind: str


CASES = [
    Case("square_operator", "Fix square so it returns a number multiplied by itself.",
         "def square(n):\n    return n + n", "def square(n):\n    return n * n",
         "square", [7], 49, "operator_replacement"),
    Case("negative_comparison", "Fix is_negative so it identifies values below zero.",
         "def is_negative(n):\n    return n > 0", "def is_negative(n):\n    return n < 0",
         "is_negative", [-3], True, "comparison_direction"),
    Case("average_empty", "Fix avg_list so an empty list returns zero without an exception.",
         "def avg_list(xs):\n    return sum(xs) / len(xs)",
         "def avg_list(xs):\n    return sum(xs) / len(xs) if xs else 0",
         "avg_list", [[]], 0, "empty_input_guard"),
    Case("odd_parity", "Fix filter_odd so it keeps odd integers rather than even integers.",
         "def filter_odd(xs):\n    return [x for x in xs if x % 2 == 0]",
         "def filter_odd(xs):\n    return [x for x in xs if x % 2 == 1]",
         "filter_odd", [[1, 2, 3, 4, 5]], [1, 3, 5], "predicate_repair"),
    Case("word_frequency_increment", "Fix word_freq so every occurrence increments its count.",
         "def word_freq(s):\n    out = {}\n    for w in s.split():\n        out[w] = out.get(w, 0)\n    return out",
         "def word_freq(s):\n    out = {}\n    for w in s.split():\n        out[w] = out.get(w, 0) + 1\n    return out",
         "word_freq", ["a b a"], {"a": 2, "b": 1}, "state_update_repair"),
]


def environment() -> str:
    memory = "unknown"
    try:
        import psutil  # type: ignore
        memory = str(psutil.virtual_memory().total)
    except Exception:
        pass
    return json.dumps({
        "language": "python",
        "language_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": memory,
    }, sort_keys=True, separators=(",", ":"))


def execute(code: str, case: Case) -> dict:
    check = (
        f"\nimport json\n_actual={case.function}(*json.loads({json.dumps(case.args)!r}))"
        f"\n_expected=json.loads({json.dumps(case.expected)!r})"
        "\nassert _actual == _expected, {'actual': _actual, 'expected': _expected}\n"
    )
    try:
        run = subprocess.run([sys.executable, "-I", "-c", code + check],
                             capture_output=True, text=True, timeout=5)
        return {"return_code": run.returncode, "stdout": run.stdout,
                "stderr": run.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"return_code": -1, "stdout": exc.stdout or "",
                "stderr": exc.stderr or "timeout", "timed_out": True}


def classify(result: dict) -> str:
    stderr = str(result["stderr"])
    if result["timed_out"]:
        return "timeout"
    if result["return_code"] == 0:
        return "success"
    if "AssertionError" in stderr:
        return "wrong_result"
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1].split(":", 1)[0] if lines else "nonzero_exit"


def generate() -> list[dict]:
    env = environment()
    episodes = []
    for case in CASES:
        before = execute(case.broken, case)
        after = execute(case.corrected, case)
        if before["return_code"] == 0 or after["return_code"] != 0:
            raise RuntimeError(f"invalid episode fixture: {case.name}")
        error_class = classify(before)
        delta = "".join(difflib.unified_diff(
            case.broken.splitlines(True), case.corrected.splitlines(True),
            fromfile="before.py", tofile="after.py"))
        episodes.append({
            "id": case.name,
            "instruction": case.instruction,
            "source_before": case.broken,
            "console_before": json.dumps(before, sort_keys=True),
            "environment": env,
            "failure_outcome": json.dumps({"status": "failure", "error_class": error_class},
                                           sort_keys=True),
            "repair_delta": delta,
            "corrected_source": case.corrected,
            "console_after": json.dumps(after, sort_keys=True),
            "resolution": json.dumps({"transition": "failure_to_success",
                                      "error_class": error_class,
                                      "repair_kind": case.repair_kind}, sort_keys=True),
            "provenance": {"kind": "executed_fixture", "license": "project_native"},
        })
    return episodes


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def post(endpoint: str, path: str, payload: dict, timeout: int = 120) -> dict:
    request = urllib.request.Request(endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def train(endpoint: str, episodes: list[dict], repeats: int, pretrain: bool) -> None:
    if pretrain:
        for pool_id, field in POOL_FIELDS.items():
            report = post(endpoint, "/brain/pretrain", {
                "pool_id": pool_id,
                "frames": [b64(str(episode[field])) for episode in episodes],
                "min_recurrence": 3,
                "max_promotions": 1024,
            })
            print(json.dumps({"pretrain_pool": pool_id, "field": field, "result": report}))
    for _ in range(repeats):
        for episode in episodes:
            for pool_id, field in POOL_FIELDS.items():
                post(endpoint, "/brain/observe", {"pool_id": pool_id,
                     "frame": b64(str(episode[field]))})
            post(endpoint, "/brain/observe", {"pool_id": ACTION_POOL,
                 "frame": b64(str(episode["corrected_source"]))})
            post(endpoint, "/brain/tick", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runtime/benchmarks/debug_episodes.jsonl")
    args = parser.parse_args()
    episodes = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n"
                                   for e in episodes), encoding="utf-8")
    if args.endpoint:
        train(args.endpoint, episodes, args.repeats, args.pretrain)
    print(json.dumps({"episodes": len(episodes), "all_failed_before": True,
                      "all_passed_after": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
