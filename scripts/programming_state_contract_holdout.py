#!/usr/bin/env python3
"""Verify learned enterprise fragments on a third, never-trained state contract."""
from __future__ import annotations

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


CLASS_NAME = "ResilientQuotaBroker"
METHOD_NAME = "reserve"


def holdout_prompt() -> str:
    return (
        f"Create a complete executable Python class named {CLASS_NAME} with an async method "
        f"named {METHOD_NAME}. It starts with credits 10 and must strict input validation "
        "for key, actor, tenant, units, and expected_version; default-deny authorization "
        "except for the auditor role; forward-only schema migration; recursive token and "
        "password secret redaction; correlated structured JSON audit logging; a resettable "
        "circuit breaker that opens after two exhausted operations; bounded asynchronous "
        "retry after transient reservation failure; idempotent command replay that returns "
        "the original response; optimistic concurrency using an expected version; "
        "duplicate-work deduplication for tenant and unit requests; an all-or-nothing "
        "transaction that restores credits; a transactional quota-reserved outbox event. "
        "Use scalar resource field credits, item field tenant, amount field units, authorized "
        "role auditor, result field receipt, event kind quota-reserved, and log request as "
        "command. Return complete implementation source, not fragments or pseudocode."
    )


def execute(source: str) -> tuple[bool, str]:
    if not source:
        return False, "no source"
    harness = "import asyncio\nimport json\n" + source + r'''
async def _verify_quota():
    broker = ResilientQuotaBroker()
    broker.migrate(2)
    attempts = {'count': 0}
    async def reserve_remote():
        attempts['count'] += 1
        if attempts['count'] < 2:
            raise RuntimeError('transient reservation failure')
        return 'receipt-8'
    command = {'key':'quota-5','actor':'auditor','tenant':'acme','units':3,
               'expected_version':1,'token':'never-log'}
    first = await broker.reserve(command, reserve_remote)
    assert first['receipt'] == 'receipt-8' and attempts['count'] == 2
    assert broker.credits == 7 and broker.version == 2
    assert broker.schema_version == 2 and len(broker.outbox) == 1
    replay = await broker.reserve(command, reserve_remote)
    assert replay == first and broker.credits == 7
    assert json.loads(broker.logs[0])['command']['token'] == '[REDACTED]'
    denied = dict(command, key='quota-6', actor='viewer', expected_version=2)
    try:
        await broker.reserve(denied, reserve_remote)
        raise AssertionError('authorization allowed')
    except PermissionError:
        pass
asyncio.run(_verify_quota())
print('PASS')
'''
    with tempfile.TemporaryDirectory(prefix="wv-state-contract-") as raw:
        path = Path(raw) / "verify.py"
        path.write_text(harness, encoding="utf-8")
        run = subprocess.run(
            [sys.executable, "-I", str(path)], capture_output=True, text=True,
            timeout=30, check=False,
        )
    return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-2000:]


def query(endpoint: str) -> dict:
    started = time.perf_counter()
    response = request(endpoint, "/brain/chat", {"text": holdout_prompt()}, timeout=300)
    source = extract_source(str(response.get("reply") or ""))
    passed, detail = execute(source)
    return {
        "executes": passed,
        "source": source,
        "detail": "" if passed else detail,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "intent_diagnostics": response.get("intent_diagnostics") or {},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/state-contract-holdout.json"))
    args = parser.parse_args()
    result = query(args.endpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["executes"]}))
    raise SystemExit(0 if result["executes"] else 1)
