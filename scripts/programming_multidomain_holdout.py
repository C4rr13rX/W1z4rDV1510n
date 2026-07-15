#!/usr/bin/env python3
"""Test causal twelve-discipline integration in a genuinely new code domain."""
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
from programming_multidomain_synthesis import DISCIPLINES, PRIMARY_FEATURE


CLASS_NAME = "ResilientFulfillmentService"
DOMAIN_REQUIREMENTS = {
    "validation": "validate an order containing key, actor, sku, quantity, and expected_version",
    "authorization": "default-deny fulfillment authorization except for the warehouse role",
    "schema_migration": "support only forward schema migration",
    "secret_redaction": "recursively redact token and password fields",
    "observability": "write correlation-aware structured JSON audit logs",
    "circuit_breaker": "open a resettable circuit after two exhausted operations",
    "async_retry": "retry transient asynchronous allocation at most three times",
    "idempotency": "return the original response when an order key is replayed",
    "optimistic_concurrency": "reject an order whose expected inventory version is stale",
    "deduplication": "reject duplicate sku and quantity work under a different key",
    "atomic_transaction": "restore inventory completely when fulfillment cannot commit",
    "transactional_outbox": "atomically append an inventory-allocated outbox event",
}


def holdout_prompt(excluded: str | None = None) -> str:
    requirements = [
        requirement for name, requirement in DOMAIN_REQUIREMENTS.items()
        if name != excluded
    ]
    return (
        f"Create a new executable Python class named {CLASS_NAME}. It manages "
        "inventory initialized with 10 widgets and must "
        + "; ".join(requirements)
        + ". Return the complete implementation, not fragments or pseudocode."
    )


def extract_source(reply: str) -> str:
    try:
        files = json.loads(reply).get("files") or {}
    except (json.JSONDecodeError, TypeError):
        return ""
    sources = [value for name, value in files.items()
               if str(name).endswith(".py") and isinstance(value, str)]
    return "\n".join(sources)


def execute(source: str) -> tuple[bool, str]:
    if not source:
        return False, "no source"
    harness = source + r'''
async def _verify_holdout():
    service = ResilientFulfillmentService()
    service.migrate(2)
    attempts = {'count': 0}
    async def allocate():
        attempts['count'] += 1
        if attempts['count'] < 2:
            raise RuntimeError('temporary allocation failure')
        return 'bin-7'
    order = {'key':'order-91','actor':'warehouse','sku':'widget','quantity':3,
             'expected_version':1,'token':'do-not-log'}
    first = await service.fulfill(order, allocate)
    assert first['allocation'] == 'bin-7' and attempts['count'] == 2
    assert service.inventory['widget'] == 7 and service.version == 2
    assert service.schema_version == 2 and len(service.outbox) == 1
    replay = await service.fulfill(order, allocate)
    assert replay == first and service.inventory['widget'] == 7
    assert json.loads(service.logs[0])['order']['token'] == '[REDACTED]'
    denied = dict(order, key='order-92', actor='customer', expected_version=2)
    try:
        await service.fulfill(denied, allocate)
        raise AssertionError('authorization allowed')
    except PermissionError:
        pass
asyncio.run(_verify_holdout())
print('PASS')
'''
    with tempfile.TemporaryDirectory(prefix="wv-multidomain-holdout-") as raw:
        path = Path(raw) / "verify.py"
        path.write_text(harness, encoding="utf-8")
        run = subprocess.run(
            [sys.executable, "-I", str(path)], capture_output=True, text=True,
            timeout=30, check=False,
        )
    detail = run.stderr or run.stdout
    return run.returncode == 0 and "PASS" in run.stdout, detail[-2000:]


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
    parser.add_argument("--output", type=Path, default=Path(
        "runtime/benchmarks/multidomain-holdout.json"
    ))
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()

    full = query(args.endpoint, holdout_prompt())
    ablations = {}
    if args.ablations:
        for premise in DISCIPLINES:
            row = query(args.endpoint, holdout_prompt(premise.name))
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
        "novel_class": CLASS_NAME,
        "disciplines": list(DOMAIN_REQUIREMENTS),
        "full": full,
        "ablations": ablations,
        "ablations_passed": ablations_passed,
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
