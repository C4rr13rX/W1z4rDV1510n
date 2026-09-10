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
    toddler_rows = []
    for prompt, expected in TODDLER:
        result = chat(endpoint, prompt)
        passed = result.get("reply") == expected
        toddler_rows.append({
            "prompt": prompt,
            "expected": expected,
            "reply": result.get("reply"),
            "passed": passed,
        })
    k12_rows = []
    for prompt in K12:
        result = chat(endpoint, prompt)
        expected = sorted(accepted.get(prompt, set()))
        passed = result.get("reply") in expected
        k12_rows.append({
            "prompt": prompt,
            "accepted": expected,
            "reply": result.get("reply"),
            "passed": passed,
        })
    oov_rows = []
    for prompt in OOV:
        result = chat(endpoint, prompt)
        passed = bool(
            not result.get("reply")
            and (result.get("grounding") or {}).get("outside_grounding")
        )
        oov_rows.append({
            "prompt": prompt,
            "reply": result.get("reply"),
            "grounding": result.get("grounding") or {},
            "route": result.get("route") or {},
            "passed": passed,
        })
    return {
        "toddler": sum(row["passed"] for row in toddler_rows),
        "toddler_total": len(TODDLER),
        "toddler_rows": toddler_rows,
        "k12": sum(row["passed"] for row in k12_rows),
        "k12_total": len(K12),
        "k12_rows": k12_rows,
        "oov": sum(row["passed"] for row in oov_rows),
        "oov_total": len(OOV),
        "oov_rows": oov_rows,
    }


class EvaluatorUnavailable(RuntimeError):
    """A child evaluator crashed instead of returning a verdict.

    This exception reaches the curriculum supervisor only as text, on this
    process's traceback, so the message deliberately carries both the child's
    own stderr and the `infrastructure_only_failure` marker that
    `transient_gate_failure()` already scans for.
    """

    def __init__(self, command: list[str], returncode: int,
                 stdout: str, stderr: str) -> None:
        self.returncode = returncode
        super().__init__(
            f'{{"infrastructure_only_failure": true}} '
            f"{Path(command[1]).name} exited {returncode} without a verdict\n"
            f"child stdout: {stdout[-1500:]}\n"
            f"child stderr: {stderr[-2500:]}"
        )


def run_evaluator(command: list[str]) -> subprocess.CompletedProcess:
    """Run a child evaluator without discarding what classifies its failure.

    `check=True` beside `capture_output=True` funnels the child's traceback
    into a CalledProcessError whose str() is only "returned non-zero exit
    status 1". The supervisor classifies a gate failure by scanning that text
    for transient markers, so it had nothing to match and scored every crash
    as neural regression.

    Measured 2026-09-10: the child died on `socket.timeout: timed out` -- a
    marker `transient_gate_failure()` already looks for -- yet 0 of 45
    midphase gate failures had ever been classified as infrastructure, and two
    go-systems blocks (rows 131072 and 262144) were quarantined for a client
    timeout against a brain that was answering correctly.
    """
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def code_eval(endpoint: str) -> dict:
    """Return the execution verdict, or report that none was produced.

    This evaluator can say no: it exits 1 when a case fails to execute, and
    prints its report either way. A non-zero exit is therefore a verdict only
    when that report parses -- a crash prints no JSON, and returning the gate
    a fabricated verdict for it is exactly the misclassification above.
    """
    command = [sys.executable, "scripts/programming_code_eval.py",
               "--endpoint", endpoint]
    run = run_evaluator(command)
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        raise EvaluatorUnavailable(
            command, run.returncode, run.stdout, run.stderr) from None


def debug_eval(endpoint: str, output: Path) -> dict:
    """Return the debug-repair verdict, or report that none was produced.

    `programming_debug_benchmark.py` returns 0 unconditionally, so a non-zero
    exit from it is never a verdict. Its report also lands in a FILE, and the
    evidence collector preserves mtimes: the copy beside both quarantined
    go-systems candidates was a 769.7 h leftover from a run a month earlier.
    Remove the path first, so a stale report can never be read as this run's
    result and admit a brain nothing measured.
    """
    command = [sys.executable, "scripts/programming_debug_benchmark.py",
               "--endpoint", endpoint, "--output", str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    run = run_evaluator(command)
    if run.returncode != 0 or not output.exists():
        raise EvaluatorUnavailable(
            command, run.returncode, run.stdout, run.stderr)
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
