#!/usr/bin/env python3
"""Admit a parameterized twelve-discipline fulfillment composition motif."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from programming_experiential_generalization import (
    b64,
    begin_experience_transaction,
    commit_experience_transaction,
    request,
    retention_passed,
    run_enterprise_retention,
    run_retention,
)
from programming_multidomain_holdout import execute, holdout_prompt, query
from programming_multidomain_synthesis import active_training_pids


@dataclass(frozen=True)
class Fragment:
    name: str
    phrase: str
    role: str
    after: tuple[str, ...]
    source: str
    parameters: tuple[tuple[str, str], ...] = ()


FRAGMENTS = (
    Fragment(
        "validation_structure",
        "strict order input validation in a Python fulfillment service class",
        "validation_structure", (),
        "import asyncio\nimport copy\nimport json\n\n"
        "class {{CLASS_NAME}}:\n"
        "    def __init__(self):\n"
        "        self.inventory = {'widget': 10}\n"
        "        self.version = 1\n        self.schema_version = 1\n"
        "        self.idempotency = {}\n        self.seen = set()\n"
        "        self.outbox = []\n        self.logs = []\n"
        "        self.failures = 0\n        self.circuit_open = False\n\n"
        "    def _validate(self, order):\n"
        "        required = {'key','actor','sku','quantity','expected_version'}\n"
        "        if not required <= set(order) or order['quantity'] <= 0:\n"
        "            raise ValueError('invalid order')\n\n",
        (("CLASS_NAME", "python_class_named"),),
    ),
    Fragment(
        "authorization", "default-deny warehouse-role authorization in Python",
        "authorization", ("validation_structure",),
        "    def _authorize(self, order):\n"
        "        if order['actor'] != 'warehouse':\n"
        "            raise PermissionError('denied')\n\n",
    ),
    Fragment(
        "migration", "forward-only Python schema migration",
        "migration", ("authorization",),
        "    def migrate(self, target):\n"
        "        if target < self.schema_version:\n"
        "            raise ValueError('downgrade')\n"
        "        self.schema_version = target\n\n",
    ),
    Fragment(
        "redaction", "recursive Python token and password secret redaction",
        "redaction", ("migration",),
        "    def _redact(self, value):\n"
        "        if isinstance(value, dict):\n"
        "            return {k: ('[REDACTED]' if k.lower() in {'token','password'} else self._redact(v)) for k,v in value.items()}\n"
        "        if isinstance(value, list):\n"
        "            return [self._redact(v) for v in value]\n"
        "        return value\n\n",
    ),
    Fragment(
        "logging", "correlation-aware structured JSON audit logging in Python",
        "logging", ("redaction",),
        "    def _log(self, event, correlation_id, **fields):\n"
        "        record = {'event': event, 'correlation_id': correlation_id, **fields}\n"
        "        self.logs.append(json.dumps(self._redact(record), sort_keys=True))\n\n",
    ),
    Fragment(
        "circuit", "a resettable Python circuit breaker that opens after two failures",
        "circuit", ("logging",),
        "    def reset_circuit(self):\n"
        "        self.failures = 0\n        self.circuit_open = False\n\n",
    ),
    Fragment(
        "retry", "bounded Python async retry for transient allocation failure",
        "retry", ("circuit",),
        "    async def _retry(self, operation, attempts=3):\n"
        "        last = None\n"
        "        for _ in range(attempts):\n"
        "            try:\n                return await operation()\n"
        "            except RuntimeError as error:\n"
        "                last = error\n                await asyncio.sleep(0)\n"
        "        raise last\n\n",
    ),
    Fragment(
        "idempotency", "idempotent Python order-key replay in an async method",
        "method", ("retry",),
        "    async def {{METHOD_NAME}}(self, order, operation):\n"
        "        self._validate(order)\n        self._authorize(order)\n"
        "        key = order['key']\n"
        "        if key in self.idempotency:\n"
        "            return self.idempotency[key]\n",
        (("METHOD_NAME", "python_method_named"),),
    ),
    Fragment(
        "optimistic_concurrency",
        "optimistic Python concurrency with expected inventory version",
        "version", ("method",),
        "        if order['expected_version'] != self.version:\n"
        "            raise RuntimeError('stale write')\n",
    ),
    Fragment(
        "deduplication", "duplicate Python sku and quantity work rejection",
        "dedup", ("version",),
        "        fingerprint = (order['sku'], order['quantity'])\n"
        "        if fingerprint in self.seen:\n"
        "            raise RuntimeError('duplicate work')\n",
    ),
    Fragment(
        "circuit_use", "Python circuit breaker enforcement around async retry",
        "circuit_use", ("dedup",),
        "        if self.circuit_open:\n"
        "            raise RuntimeError('circuit open')\n"
        "        try:\n            allocation = await self._retry(operation)\n"
        "        except RuntimeError:\n"
        "            self.failures += 1\n"
        "            if self.failures >= 2:\n                self.circuit_open = True\n"
        "            raise\n",
    ),
    Fragment(
        "transaction", "restore Python inventory completely when fulfillment cannot commit",
        "transaction", ("circuit_use",),
        "        before = copy.deepcopy(self.inventory)\n"
        "        try:\n"
        "            sku, quantity = order['sku'], order['quantity']\n"
        "            if self.inventory.get(sku, 0) < quantity:\n"
        "                raise ValueError('insufficient inventory')\n"
        "            self.inventory[sku] -= quantity\n"
        "        except Exception:\n"
        "            self.inventory = before\n            raise\n",
    ),
    Fragment(
        "outbox_commit", "a transactional Python inventory-allocated outbox event",
        "outbox_commit", ("transaction",),
        "        event = {'kind':'inventory-allocated','key':key,'allocation':allocation}\n"
        "        self.outbox.append(event)\n"
        "        self.seen.add(fingerprint)\n        self.version += 1\n"
        "        response = {'ok': True, 'allocation': allocation, 'version': self.version}\n"
        "        self.idempotency[key] = response\n"
        "        self._log('fulfilled', key, order=order, response=response)\n"
        "        return response\n",
    ),
)

REQUIRED_FEATURE = {
    "validation_structure": "ENTERPRISE:INPUT_VALIDATION",
    "authorization": "SECURITY:AUTHORIZATION",
    "migration": "PERSISTENCE:SCHEMA_MIGRATION",
    "redaction": "ENTERPRISE:SECRET_REDACTION",
    "logging": "OBSERVABILITY:CORRELATED_LOGGING",
    "circuit": "RESILIENCE:CIRCUIT_BREAKER",
    "retry": "RESILIENCE:ASYNC_RETRY",
    "idempotency": "API:IDEMPOTENT_COMMAND",
    "optimistic_concurrency": "STATE:OPTIMISTIC_CONCURRENCY",
    "deduplication": "CONCURRENCY:DEDUPLICATION",
    "circuit_use": "RESILIENCE:CIRCUIT_BREAKER",
    "transaction": "PERSISTENCE:ATOMIC_TRANSACTION",
    "outbox_commit": "INTEGRATION:TRANSACTIONAL_OUTBOX",
}


def encoded_fragment(fragment: Fragment) -> str:
    payload: dict[str, object] = {
        "file": "parameterized_fulfillment.py",
        "role": fragment.role,
        "after": list(fragment.after),
        "source": fragment.source,
    }
    if fragment.parameters:
        payload["parameters"] = dict(fragment.parameters)
    return json.dumps({"code_fragment": payload}, sort_keys=True, separators=(",", ":"))


def training_rows() -> list[tuple[str, str]]:
    return [
        (f"Implement {fragment.phrase} for an inventory fulfillment domain "
         "as a reusable grounded fragment.",
         encoded_fragment(fragment))
        for fragment in FRAGMENTS
    ]


def train(endpoint: str, repeats: int) -> dict:
    rows = training_rows()
    episodes = [{"frames": [
        {"pool_id": 1, "frame": b64(prompt)},
        {"pool_id": 12, "frame": b64(prompt)},
        {"pool_id": 4, "frame": b64(response)},
    ]} for prompt, response in rows] * repeats
    result = request(endpoint, "/brain/pretrain_bindings", {"episodes": episodes}, timeout=1200)
    if not result.get("ok") or result.get("accepted") != len(episodes):
        raise RuntimeError(f"parameterized motif training rejected: {result}")
    for fragment, (prompt, _) in zip(FRAGMENTS, rows, strict=True):
        probe = request(endpoint, "/brain/chat", {"text": prompt + " Verify its intent."}, timeout=60)
        labels = (probe.get("intent_diagnostics") or {}).get("labels") or []
        required = REQUIRED_FEATURE[fragment.name]
        if not any(label.endswith(":" + required) for label in labels):
            raise RuntimeError(
                f"training prompt for {fragment.name!r} did not fire {required}: {labels}"
            )
    return result


def evaluate(endpoint: str) -> dict:
    cases = [
        ("ResilientFulfillmentService", "fulfill"),
        ("DurableWarehouseEngine", "allocate_order"),
    ]
    return {
        f"{class_name}.{method_name}": query(
            endpoint,
            holdout_prompt(class_name=class_name, method_name=method_name),
            class_name,
            method_name,
        )
        for class_name, method_name in cases
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--enterprise-gate", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "runtime/benchmarks/parameterized-fulfillment.json"
    ))
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    live_training = active_training_pids(runtime)
    if args.train:
        if live_training:
            parser.error("pause corpus training before motif mutation; active PIDs: "
                         + ",".join(map(str, live_training)))

    transaction = begin_experience_transaction(args.endpoint, runtime) if args.train else None
    retention_before = run_retention(
        args.endpoint, args.output.with_name("parameterized-foundation-before.json")
    ) if args.train else None
    tick_before = request(args.endpoint, "/brain/stats", None).get("tick")
    baseline = evaluate(args.endpoint)
    trained = train(args.endpoint, args.repeats) if args.train else None
    learned = evaluate(args.endpoint) if args.train else baseline
    tick_after = request(args.endpoint, "/brain/stats", None).get("tick")
    retention_after = run_retention(
        args.endpoint, args.output.with_name("parameterized-foundation-after.json")
    ) if args.train else None
    enterprise = run_enterprise_retention(
        args.endpoint, args.output.with_name("parameterized-enterprise-after.json")
    ) if args.train and args.enterprise_gate else None
    retention_ok = retention_after is None or retention_passed(retention_after)
    enterprise_ok = enterprise is None or bool(enterprise.get("passed"))
    learned_ok = all(row["executes"] for row in learned.values())
    final_sources = {row["source"] for row in learned.values()}
    trained_responses = {response for _, response in training_rows()}
    complete_artifact_was_unseen = all(
        source and all(source not in response for response in trained_responses)
        for source in final_sources
    )
    passed = learned_ok and complete_artifact_was_unseen and retention_ok and enterprise_ok
    report = {
        "passed": passed,
        "baseline": baseline,
        "training": trained,
        "learned": learned,
        "complete_artifact_was_unseen": complete_artifact_was_unseen,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "retention_passed": retention_ok,
        "enterprise_after": enterprise,
        "enterprise_passed": enterprise_ok,
        "tick_before": tick_before,
        "tick_after": tick_after,
        "tick_delta": tick_after - tick_before if isinstance(tick_before, int)
        and isinstance(tick_after, int) else None,
        "concurrent_training_pids": live_training,
        "concurrent_mutation_detected": bool(
            live_training and isinstance(tick_before, int)
            and isinstance(tick_after, int) and tick_after != tick_before
        ),
        "updated_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if passed and transaction is not None:
        report["admitted_checkpoint"] = commit_experience_transaction(
            args.endpoint, transaction[0], transaction[1]
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": passed, "learned_executes": learned_ok,
        "complete_artifact_was_unseen": complete_artifact_was_unseen,
        "retention_passed": retention_ok, "enterprise_passed": enterprise_ok,
        "tick_delta": report["tick_delta"],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
