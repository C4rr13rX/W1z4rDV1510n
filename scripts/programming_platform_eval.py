#!/usr/bin/env python3
"""Execution-backed platform engineering curriculum for Wizard Vision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from programming_project_eval import Case, b64, execute_project, manifest, request


CASES = [
    Case(
        "idempotent_order_api",
        "Implement a Python order API project whose POST command requires an idempotency key and safely replays the original response.",
        "Build Python API code that prevents duplicate orders when clients retry with the same replay key.",
        manifest({
            "api.py":
                "class OrderApi:\n"
                "    def __init__(self):\n        self.orders = []\n        self.responses = {}\n\n"
                "    def handle(self, method, path, body=None, idempotency_key=None):\n"
                "        if method != \"POST\" or path != \"/orders\":\n            return 404, {\"error\": \"not_found\"}\n"
                "        if not idempotency_key:\n            return 400, {\"error\": \"idempotency_key_required\"}\n"
                "        if idempotency_key in self.responses:\n            return self.responses[idempotency_key]\n"
                "        if not isinstance(body, dict) or not body.get(\"sku\") or not isinstance(body.get(\"quantity\"), int) or body[\"quantity\"] < 1:\n"
                "            return 422, {\"error\": \"invalid_order\"}\n"
                "        order = {\"id\": len(self.orders) + 1, \"sku\": body[\"sku\"], \"quantity\": body[\"quantity\"]}\n"
                "        self.orders.append(order)\n"
                "        response = (201, order.copy())\n        self.responses[idempotency_key] = response\n        return response\n",
        }),
        "from api import OrderApi\napi = OrderApi()\n"
        "first = api.handle('POST','/orders',{'sku':'A','quantity':2},'key-1')\n"
        "second = api.handle('POST','/orders',{'sku':'CHANGED','quantity':9},'key-1')\n"
        "assert first == second and first[0] == 201 and len(api.orders) == 1\n"
        "assert api.handle('POST','/orders',{'sku':'A','quantity':1})[0] == 400\n"
        "assert api.handle('POST','/orders',{'sku':'A','quantity':0},'key-2')[0] == 422\n"
        "assert len(api.orders) == 1\n",
    ),
    Case(
        "versioned_migrations",
        "Implement a Python SQLite schema migration project with ordered version upgrades that is safe to run repeatedly.",
        "Write Python database upgrade paths using schema versions so fresh and legacy SQLite databases reach the same structure.",
        manifest({
            "migrations.py":
                "import sqlite3\n\n"
                "def migrate(db_path):\n"
                "    with sqlite3.connect(db_path) as connection:\n"
                "        version = connection.execute(\"PRAGMA user_version\").fetchone()[0]\n"
                "        if version < 1:\n"
                "            connection.execute(\"CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT NOT NULL)\")\n"
                "            connection.execute(\"PRAGMA user_version = 1\")\n            version = 1\n"
                "        if version < 2:\n"
                "            columns = {row[1] for row in connection.execute(\"PRAGMA table_info(users)\")}\n"
                "            if \"email\" not in columns:\n                connection.execute(\"ALTER TABLE users ADD COLUMN email TEXT\")\n"
                "            connection.execute(\"PRAGMA user_version = 2\")\n"
                "    return 2\n",
        }),
        "import sqlite3, tempfile\nfrom pathlib import Path\nfrom migrations import migrate\n"
        "fresh = str(Path(tempfile.mkdtemp())/'fresh.db')\nassert migrate(fresh) == 2\nassert migrate(fresh) == 2\n"
        "with sqlite3.connect(fresh) as c:\n    assert c.execute('PRAGMA user_version').fetchone()[0] == 2\n    assert {r[1] for r in c.execute('PRAGMA table_info(users)')} == {'id','name','email'}\n"
        "legacy = str(Path(tempfile.mkdtemp())/'legacy.db')\n"
        "with sqlite3.connect(legacy) as c:\n    c.execute('CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT NOT NULL)')\n    c.execute('PRAGMA user_version = 1')\n"
        "migrate(legacy)\n"
        "with sqlite3.connect(legacy) as c:\n    assert 'email' in {r[1] for r in c.execute('PRAGMA table_info(users)')}\n",
    ),
    Case(
        "correlated_logging",
        "Implement Python structured logging with a required correlation ID and recursive redaction of password, token, and api_key fields.",
        "Create a Python observability module that writes correlated JSON logs without leaking nested secrets.",
        manifest({
            "observability.py":
                "import json\n\n"
                "SECRET_KEYS = {\"password\", \"token\", \"api_key\"}\n\n"
                "def _redact(value):\n"
                "    if isinstance(value, dict):\n        return {key: (\"[REDACTED]\" if str(key).lower() in SECRET_KEYS else _redact(item)) for key, item in value.items()}\n"
                "    if isinstance(value, list):\n        return [_redact(item) for item in value]\n"
                "    return value\n\n"
                "def write_event(stream, event, correlation_id, **fields):\n"
                "    if not correlation_id:\n        raise ValueError(\"correlation_id is required\")\n"
                "    record = {\"event\": event, \"correlation_id\": correlation_id, **_redact(fields)}\n"
                "    stream.write(json.dumps(record, sort_keys=True, separators=(\",\", \":\")) + \"\\n\")\n"
                "    return record\n",
        }),
        "import io, json\nfrom observability import write_event\nstream = io.StringIO()\n"
        "record = write_event(stream,'login','req-7',user={'name':'a','password':'p'},token='t')\n"
        "decoded = json.loads(stream.getvalue())\nassert decoded == record\n"
        "assert decoded['correlation_id'] == 'req-7' and decoded['token'] == '[REDACTED]'\n"
        "assert decoded['user']['password'] == '[REDACTED]'\n"
        "assert '\"password\":\"p\"' not in stream.getvalue() and '\"token\":\"t\"' not in stream.getvalue()\n"
        "try:\n    write_event(stream,'x','')\n    raise AssertionError('missing correlation accepted')\nexcept ValueError:\n    pass\n",
    ),
    Case(
        "circuit_breaker",
        "Implement a Python circuit breaker with a failure threshold, recovery timeout, and injected clock for deterministic testing.",
        "Build Python resilience code that opens after repeated failures and permits a trial call after its cooldown.",
        manifest({
            "circuit.py":
                "import time\n\nclass CircuitOpen(RuntimeError):\n    pass\n\n"
                "class CircuitBreaker:\n"
                "    def __init__(self, failure_threshold, recovery_timeout, clock=time.monotonic):\n"
                "        if failure_threshold < 1 or recovery_timeout < 0:\n            raise ValueError(\"invalid circuit configuration\")\n"
                "        self.failure_threshold = failure_threshold\n        self.recovery_timeout = recovery_timeout\n        self.clock = clock\n"
                "        self.failures = 0\n        self.opened_at = None\n\n"
                "    def call(self, operation):\n"
                "        now = self.clock()\n"
                "        if self.opened_at is not None and now - self.opened_at < self.recovery_timeout:\n            raise CircuitOpen(\"circuit is open\")\n"
                "        try:\n            result = operation()\n"
                "        except Exception:\n"
                "            self.failures += 1\n"
                "            if self.opened_at is not None or self.failures >= self.failure_threshold:\n                self.opened_at = now\n"
                "            raise\n"
                "        self.failures = 0\n        self.opened_at = None\n        return result\n",
        }),
        "from circuit import CircuitBreaker, CircuitOpen\nnow = [0.0]\ncb = CircuitBreaker(2, 10, lambda: now[0])\n"
        "def fail(): raise RuntimeError('down')\n"
        "for _ in range(2):\n    try: cb.call(fail)\n    except RuntimeError: pass\n"
        "try:\n    cb.call(lambda:'should not run')\n    raise AssertionError('open circuit allowed call')\nexcept CircuitOpen:\n    pass\n"
        "now[0] = 11\nassert cb.call(lambda:'ok') == 'ok'\nassert cb.failures == 0 and cb.opened_at is None\n",
    ),
]

OOV = [
    "Implement a Python API client for an endpoint whose protocol is unknown.",
    "Write a Python migration for an unspecified legacy schema and target schema.",
    "Create Python production alert thresholds without any service objectives or metrics.",
]


def train(endpoint: str, repeats: int) -> None:
    for _ in range(repeats):
        for case in CASES:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(case.response)})
            request(endpoint, "/brain/tick", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/platform.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    results = []
    for case in CASES:
        for kind, prompt in (("trained", case.prompt), ("paraphrase", case.paraphrase)):
            result = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(result.get("reply") or "")
            passed, detail = execute_project(reply, case.integration_test)
            results.append({"name": case.name, "kind": kind, "nonempty": bool(reply),
                            "exact": reply == case.response, "executes": passed,
                            "detail": "" if passed else detail})
    oov = []
    for prompt in OOV:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not result.get("reply") and bool((result.get("grounding") or {}).get("outside_grounding"))
        oov.append({"prompt": prompt, "honest": honest, "reply": result.get("reply")})
    summary = {kind: {"executes": sum(row["executes"] for row in results if row["kind"] == kind),
                      "total": len(CASES)} for kind in ("trained", "paraphrase")}
    summary["oov_honesty"] = {"passed": sum(row["honest"] for row in oov), "total": len(oov)}
    report = {"summary": summary, "results": results, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in results) and all(row["honest"] for row in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
