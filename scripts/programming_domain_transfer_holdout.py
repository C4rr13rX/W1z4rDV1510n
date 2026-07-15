#!/usr/bin/env python3
"""Measure twelve-discipline transfer into an unseen scheduler state model."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from programming_experiential_generalization import request
from programming_multidomain_holdout import extract_source
from programming_multidomain_synthesis import DISCIPLINES, PRIMARY_FEATURE


CLASS_NAME = "ResilientJobScheduler"
METHOD_NAME = "schedule"
REQUIREMENTS = {
    "validation": "strict input validation for key, actor, job, slots, and expected_version",
    "authorization": "default-deny authorization except for the operator role",
    "schema_migration": "forward-only schema migration",
    "secret_redaction": "recursive token and password secret redaction",
    "observability": "correlated structured JSON audit logging",
    "circuit_breaker": "a resettable circuit breaker that opens after two exhausted operations",
    "async_retry": "bounded asynchronous retry after transient dispatch failure",
    "idempotency": "idempotent command replay that returns the original response",
    "optimistic_concurrency": "optimistic concurrency using an expected version",
    "deduplication": "duplicate-work deduplication for job and slot requests",
    "atomic_transaction": "an all-or-nothing transaction that restores scheduler capacity",
    "transactional_outbox": "a transactional job-scheduled outbox event",
}


def transfer_prompt(excluded: str | None = None) -> str:
    requirements = [value for name, value in REQUIREMENTS.items() if name != excluded]
    return (
        f"Create a complete executable Python class named {CLASS_NAME} with an async "
        f"method named {METHOD_NAME}. It starts with capacity 10 and must "
        + "; ".join(requirements)
        + ". Return complete implementation source, not fragments or pseudocode."
    )


def execute(source: str) -> tuple[bool, str]:
    if not source:
        return False, "no source"
    harness = "import asyncio\nimport json\n" + source + r'''
async def _verify_scheduler():
    scheduler = ResilientJobScheduler()
    scheduler.migrate(2)
    attempts = {'count': 0}
    async def dispatch():
        attempts['count'] += 1
        if attempts['count'] < 2:
            raise RuntimeError('transient dispatch failure')
        return 'worker-4'
    command = {'key':'job-17','actor':'operator','job':'render','slots':3,
               'expected_version':1,'password':'never-log'}
    first = await scheduler.schedule(command, dispatch)
    assert first['worker'] == 'worker-4' and attempts['count'] == 2
    assert scheduler.capacity == 7 and scheduler.version == 2
    assert scheduler.schema_version == 2 and len(scheduler.outbox) == 1
    replay = await scheduler.schedule(command, dispatch)
    assert replay == first and scheduler.capacity == 7
    assert json.loads(scheduler.logs[0])['command']['password'] == '[REDACTED]'
    denied = dict(command, key='job-18', actor='viewer', expected_version=2)
    try:
        await scheduler.schedule(denied, dispatch)
        raise AssertionError('authorization allowed')
    except PermissionError:
        pass
asyncio.run(_verify_scheduler())
print('PASS')
'''
    with tempfile.TemporaryDirectory(prefix="wv-domain-transfer-") as raw:
        path = Path(raw) / "verify.py"
        path.write_text(harness, encoding="utf-8")
        run = subprocess.run(
            [sys.executable, "-I", str(path)], capture_output=True, text=True,
            timeout=30, check=False,
        )
    return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-2000:]


def query(endpoint: str, prompt: str) -> dict:
    started = time.perf_counter()
    response = request(endpoint, "/brain/chat", {"text": prompt}, timeout=300)
    source = extract_source(str(response.get("reply") or ""))
    passed, detail = execute(source)
    return {
        "executes": passed,
        "source": source,
        "detail": "" if passed else detail,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "intent_diagnostics": response.get("intent_diagnostics") or {},
        "honest_oov": not response.get("reply") and bool(
            (response.get("grounding") or {}).get("outside_grounding")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "runtime/benchmarks/domain-transfer-holdout.json"
    ))
    args = parser.parse_args()
    full = query(args.endpoint, transfer_prompt())
    ablations = {}
    if args.ablations:
        for premise in DISCIPLINES:
            row = query(args.endpoint, transfer_prompt(premise.name))
            labels = (row.get("intent_diagnostics") or {}).get("labels") or []
            row["feature_removed"] = not any(
                label.endswith(":" + PRIMARY_FEATURE[premise.name])
                for label in labels
            )
            ablations[premise.name] = row
    ablations_passed = not args.ablations or all(
        not row["executes"] and row["feature_removed"]
        for row in ablations.values()
    )
    report = {
        "passed": full["executes"] and ablations_passed,
        "full": full,
        "ablations": ablations,
        "ablations_passed": ablations_passed,
        "disciplines": list(REQUIREMENTS),
        "updated_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "full_executes": full["executes"],
        "ablations_passed": ablations_passed,
    }))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
