#!/usr/bin/env python3
"""Run the complete enterprise coding battery without mutating the brain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def brain_tick(endpoint: str) -> int:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/brain/stats", timeout=30) as response:
        return int(json.load(response)["tick"])


def run_suite(name: str, command: list[str], timeout: float) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, *command], cwd=ROOT, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "name": name, "passed": False, "timed_out": True,
            "exit_code": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "summary": {}, "stderr_tail": stderr[-2000:],
        }
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    parsed: object = {}
    if lines:
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError:
            parsed = {"stdout_tail": "\n".join(lines[-10:])}
    return {
        "name": name,
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "summary": parsed,
        "stderr_tail": result.stderr[-2000:] if result.returncode else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/enterprise-retention.json"))
    parser.add_argument("--suite-timeout", type=float, default=900.0)
    args = parser.parse_args()
    output_dir = args.output.parent
    endpoint_args = ["--endpoint", args.endpoint]
    suites = [
        ("python_enterprise", ["scripts/programming_enterprise_eval.py", *endpoint_args,
                               "--no-train", "--output", str(output_dir / "enterprise.json")]),
        ("multilanguage", ["scripts/programming_multilanguage_eval.py", *endpoint_args,
                            "--no-train", "--output", str(output_dir / "multilanguage.json")]),
        ("native_enterprise", ["scripts/programming_native_enterprise_eval.py", *endpoint_args,
                                "--no-train", "--output", str(output_dir / "native.json")]),
        ("platform", ["scripts/programming_platform_eval.py", *endpoint_args,
                       "--no-train", "--output", str(output_dir / "platform.json")]),
        ("project", ["scripts/programming_project_eval.py", *endpoint_args,
                      "--no-train", "--output", str(output_dir / "project.json")]),
        ("typescript", ["scripts/programming_typescript_enterprise.py", *endpoint_args,
                         "--no-train", "--output", str(output_dir / "typescript.json")]),
        ("cross_language", ["scripts/programming_cross_language_transfer.py", *endpoint_args,
                             "--no-train", "--output", str(output_dir / "cross-language.json")]),
        ("cross_project", ["scripts/programming_cross_project_composition.py", *endpoint_args,
                            "--output", str(output_dir / "cross-project.json")]),
        ("polyglot", ["scripts/programming_polyglot_composition.py", *endpoint_args,
                       "--output", str(output_dir / "polyglot.json")]),
        ("composition", ["scripts/programming_composition_eval.py", *endpoint_args,
                          "--output", str(output_dir / "composition.json")]),
        ("semantic_stress", ["scripts/programming_semantic_stress.py", *endpoint_args,
                              "--output", str(output_dir / "semantic-stress.json")]),
    ]

    tick_before = brain_tick(args.endpoint)
    results = []
    for index, (name, command) in enumerate(suites, 1):
        print(f"[enterprise-retention] [{index}/{len(suites)}] {name}", flush=True)
        result = run_suite(name, command, args.suite_timeout)
        results.append(result)
        print(json.dumps({
            "suite": name, "passed": result["passed"],
            "timed_out": result.get("timed_out", False),
            "elapsed_seconds": result["elapsed_seconds"],
        }), flush=True)
    tick_after = brain_tick(args.endpoint)
    report = {
        "passed": all(row["passed"] for row in results) and tick_after == tick_before,
        "tick_before": tick_before,
        "tick_after": tick_after,
        "tick_delta": tick_after - tick_before,
        "passed_suites": sum(row["passed"] for row in results),
        "total_suites": len(results),
        "results": results,
        "updated_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "tick_before", "tick_after", "tick_delta",
        "passed_suites", "total_suites",
    )}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
