#!/usr/bin/env python3
"""Train foundation + coding-debug stages and measure cross-stage interference."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from programming_brain_eval import K12, OOV, TODDLER, accepted_answers
from programming_code_eval import CASES as CODE_CASES, load_examples
from programming_debug_episode_train import generate, train as train_debug

ROOT = Path(__file__).resolve().parents[1]


def mutation_enabled(read_only: bool) -> bool:
    return not read_only


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def request(endpoint: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def train_pairs(endpoint: str, pairs: list[tuple[str, str]], repeats: int,
                instruction_features: bool = False) -> None:
    for _ in range(repeats):
        for prompt, answer in pairs:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(prompt)})
            if instruction_features:
                request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(answer)})
            request(endpoint, "/brain/tick", {})


def chat(endpoint: str, prompt: str) -> dict:
    return request(endpoint, "/brain/chat", {"text": prompt})


def foundation_eval(endpoint: str, accepted: dict[str, set[str]]) -> dict:
    toddler = sum(chat(endpoint, prompt).get("reply") == expected
                   for prompt, expected in TODDLER)
    k12 = sum(chat(endpoint, prompt).get("reply") in accepted.get(prompt, set())
              for prompt in K12)
    oov = 0
    for prompt in OOV:
        result = chat(endpoint, prompt)
        oov += bool(not result.get("reply")
                    and (result.get("grounding") or {}).get("outside_grounding"))
    return {"toddler": toddler, "toddler_total": len(TODDLER),
            "k12": k12, "k12_total": len(K12),
            "oov": oov, "oov_total": len(OOV)}


def code_eval(endpoint: str) -> dict:
    run = subprocess.run([sys.executable, "scripts/programming_code_eval.py",
                          "--endpoint", endpoint], cwd=ROOT, check=True,
                         capture_output=True, text=True)
    return json.loads(run.stdout.strip().splitlines()[-1])


def debug_eval(endpoint: str, output: Path) -> dict:
    run = subprocess.run([sys.executable, "scripts/programming_debug_benchmark.py",
                          "--endpoint", endpoint, "--output", str(output)],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(output.read_text(encoding="utf-8"))


def integrated_retention_passed(report: dict) -> bool:
    """Require every protected foundation, execution, and transfer result."""
    after = report.get("after_debug") or {}
    foundation = after.get("foundation") or {}
    python = (after.get("python") or {}).get("summary") or {}
    debug = after.get("debug") or {}
    return (
        foundation.get("toddler") == foundation.get("toddler_total")
        and foundation.get("k12") == foundation.get("k12_total")
        and foundation.get("oov") == foundation.get("oov_total")
        and bool(python)
        and all(
            group.get("executes") == group.get("count")
            and group.get("syntax_valid") == group.get("count")
            for group in python.values()
        )
        and bool(debug)
        and all(group.get("passed") == group.get("total") for group in debug.values())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--toddler-repeats", type=int, default=8)
    parser.add_argument("--k12-repeats", type=int, default=4)
    parser.add_argument("--python-repeats", type=int, default=4)
    parser.add_argument("--debug-repeats", type=int, default=4)
    parser.add_argument("--pretrain-debug", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="run the complete retention suite read-only: no training and no snapshot")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runtime/benchmarks/integrated_retention.json")
    args = parser.parse_args()

    accepted = accepted_answers(ROOT / "data/training/categorical_unified_001.jsonl")
    k12_pairs = [(prompt, sorted(accepted[prompt])[0]) for prompt in K12]
    examples = load_examples()
    python_pairs = []
    for response_prefix, *_ in CODE_CASES:
        row = next(row for row in examples if str(row.get("response", "")).startswith(response_prefix))
        python_pairs.append((row["prompt"], row["response"]))

    # Read-only verification must be a genuinely separate branch.  The old
    # --no-checkpoint implementation skipped only the final snapshot while
    # silently retraining every protected curriculum and the debug episodes.
    # That made a benchmark mutate the state it claimed to measure and added
    # thousands of ticks when a timed-out client retried it.
    if mutation_enabled(args.no_checkpoint):
        train_pairs(args.endpoint, TODDLER, args.toddler_repeats)
        train_pairs(args.endpoint, k12_pairs, args.k12_repeats)
        train_pairs(args.endpoint, python_pairs, args.python_repeats, instruction_features=True)
    before = {"foundation": foundation_eval(args.endpoint, accepted),
              "python": code_eval(args.endpoint),
              "stats": request(args.endpoint, "/brain/stats")}

    episodes = generate()
    if mutation_enabled(args.no_checkpoint):
        train_debug(args.endpoint, episodes, args.debug_repeats, args.pretrain_debug)
    after = {"foundation": foundation_eval(args.endpoint, accepted),
             "python": code_eval(args.endpoint),
             "debug": debug_eval(args.endpoint, args.output.with_name("integrated_debug.json")),
             "stats": request(args.endpoint, "/brain/stats")}
    checkpoint = None if args.no_checkpoint else request(args.endpoint, "/brain/checkpoint", {})
    report = {"before_debug": before, "after_debug": after, "checkpoint": checkpoint}
    report["passed"] = integrated_retention_passed(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"before": before["foundation"], "after": after["foundation"],
                      "python_before": before["python"]["summary"],
                      "python_after": after["python"]["summary"],
                      "debug": {key: {"passed": value["passed"], "total": value["total"]}
                                for key, value in after["debug"].items()}}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
