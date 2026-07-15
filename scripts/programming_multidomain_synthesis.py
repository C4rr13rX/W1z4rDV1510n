#!/usr/bin/env python3
"""Measure novel code synthesis from twelve independently learned disciplines."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

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


@dataclass(frozen=True)
class Premise:
    name: str
    phrase: str
    role: str
    after: tuple[str, ...]
    source: str


def fragment(role: str, after: tuple[str, ...], source: str) -> str:
    return json.dumps({"code_fragment": {
        "file": "adaptive_system.py", "role": role,
        "after": list(after), "source": source,
    }}, sort_keys=True, separators=(",", ":"))


HEADER = (
    Premise("structure", "a Python project class", "imports", (),
            "import asyncio\nimport copy\nimport json\n\nclass AdaptiveCoordinator:\n"),
    Premise("state", "stateful account coordination", "init", ("imports",),
            "    def __init__(self):\n"
            "        self.balances = {'a': 50, 'b': 5}\n"
            "        self.version = 1\n        self.schema_version = 1\n"
            "        self.idempotency = {}\n        self.seen = set()\n"
            "        self.outbox = []\n        self.logs = []\n"
            "        self.failures = 0\n        self.circuit_open = False\n\n"),
)

PREMISES = (
    Premise("validation", "strict input validation", "validation", ("init",),
            "    def _validate(self, command):\n"
            "        required = {'key','actor','source','target','amount','expected_version'}\n"
            "        if not required <= set(command) or command['amount'] <= 0:\n"
            "            raise ValueError('invalid command')\n\n"),
    Premise("authorization", "default-deny authorization", "authorization", ("validation",),
            "    def _authorize(self, command):\n"
            "        if command['actor'] != 'admin':\n"
            "            raise PermissionError('denied')\n\n"),
    Premise("schema_migration", "schema migration", "migration", ("authorization",),
            "    def migrate(self, target):\n"
            "        if target < self.schema_version:\n"
            "            raise ValueError('downgrade')\n"
            "        self.schema_version = target\n\n"),
    Premise("secret_redaction", "recursive secret redaction", "redaction", ("migration",),
            "    def _redact(self, value):\n"
            "        if isinstance(value, dict):\n"
            "            return {k: ('[REDACTED]' if k.lower() in {'token','password'} else self._redact(v)) for k,v in value.items()}\n"
            "        if isinstance(value, list):\n"
            "            return [self._redact(v) for v in value]\n"
            "        return value\n\n"),
    Premise("observability", "correlated structured JSON logging", "logging", ("redaction",),
            "    def _log(self, event, correlation_id, **fields):\n"
            "        record = {'event': event, 'correlation_id': correlation_id, **fields}\n"
            "        self.logs.append(json.dumps(self._redact(record), sort_keys=True))\n\n"),
    Premise("circuit_breaker", "a cooldown circuit breaker", "circuit", ("logging",),
            "    def reset_circuit(self):\n"
            "        self.failures = 0\n        self.circuit_open = False\n\n"),
    Premise("async_retry", "bounded async retry after transient failure", "retry", ("circuit",),
            "    async def _retry(self, operation, attempts=3):\n"
            "        last = None\n"
            "        for _ in range(attempts):\n"
            "            try:\n                return await operation()\n"
            "            except RuntimeError as error:\n"
            "                last = error\n                await asyncio.sleep(0)\n"
            "        raise last\n\n"),
    Premise("idempotency", "idempotent command replay", "process_signature", ("retry",),
            "    async def process(self, command, operation):\n"
            "        self._validate(command)\n        self._authorize(command)\n"
            "        key = command['key']\n"
            "        if key in self.idempotency:\n            return self.idempotency[key]\n"),
    Premise("optimistic_concurrency", "optimistic concurrency with expected version", "version_guard", ("process_signature",),
            "        if command['expected_version'] != self.version:\n"
            "            raise RuntimeError('stale write')\n"),
    Premise("deduplication", "duplicate-work deduplication", "dedup_guard", ("version_guard",),
            "        fingerprint = (command['source'], command['target'], command['amount'])\n"
            "        if fingerprint in self.seen:\n"
            "            raise RuntimeError('duplicate work')\n"),
    Premise("circuit_use", "circuit breaker opens after repeated failures", "circuit_guard", ("dedup_guard",),
            "        if self.circuit_open:\n            raise RuntimeError('circuit open')\n"
            "        try:\n            result = await self._retry(operation)\n"
            "        except RuntimeError:\n"
            "            self.failures += 1\n"
            "            if self.failures >= 2:\n                self.circuit_open = True\n"
            "            raise\n"),
    Premise("atomic_transaction", "an all-or-nothing database transaction for balances", "transaction", ("circuit_guard",),
            "        before = copy.deepcopy(self.balances)\n"
            "        try:\n"
            "            amount = command['amount']\n"
            "            if self.balances[command['source']] < amount:\n"
            "                raise ValueError('insufficient')\n"
            "            self.balances[command['source']] -= amount\n"
            "            self.balances[command['target']] += amount\n"
            "        except Exception:\n"
            "            self.balances = before\n            raise\n"),
    Premise("transactional_outbox", "a transactional outbox event", "outbox", ("transaction",),
            "        event = {'kind':'transfer','key':key,'result':result}\n"
            "        self.outbox.append(event)\n"),
    Premise("commit", "correlated audit completion", "commit", ("outbox",),
            "        self.seen.add(fingerprint)\n        self.version += 1\n"
            "        response = {'ok': True, 'result': result, 'version': self.version}\n"
            "        self.idempotency[key] = response\n"
            "        self._log('transfer', key, command=command, response=response)\n"
            "        return response\n"),
)

ALTERNATIVE_PREMISES = (
    Premise(
        "no_retry_policy",
        "no retries: fail immediately after the first transient failure",
        "retry", ("circuit",),
        "    async def _retry(self, operation, attempts=1):\n"
        "        return await operation()\n\n",
    ),
)

# `circuit_use` is the implementation continuation of the same circuit-breaker
# discipline, not a thirteenth independent premise.
DISCIPLINES = tuple(
    p for p in PREMISES if p.name not in {"commit", "circuit_use"}
)


def training_rows() -> list[tuple[str, str]]:
    rows = []
    for premise in HEADER + PREMISES + ALTERNATIVE_PREMISES:
        prompt = f"Implement {premise.phrase} for the AdaptiveCoordinator Python project."
        rows.append((prompt, fragment(premise.role, premise.after, premise.source)))
    return rows


def synthesis_prompt(excluded: str | None = None, contradiction: bool = False) -> str:
    phrases = [p.phrase for p in DISCIPLINES if p.name != excluded]
    if contradiction:
        phrases = [p for p in phrases if "retry" not in p]
        phrases.append("no retries: fail immediately after the first transient failure")
    return "Build a Python project class integrating " + ", ".join(phrases) + "."


def train(endpoint: str, repeats: int) -> dict:
    episodes = []
    for prompt, response in training_rows():
        episodes.append({"frames": [
            {"pool_id": 1, "frame": b64(prompt)},
            {"pool_id": 12, "frame": b64(prompt)},
            {"pool_id": 4, "frame": b64(response)},
        ]})
    result = request(endpoint, "/brain/pretrain_bindings", {
        "episodes": episodes * repeats,
    }, timeout=1200)
    expected = len(episodes) * repeats
    if not result.get("ok") or result.get("accepted") != expected:
        raise RuntimeError(f"multi-domain premise training rejected: {result}")
    return result


def extract_source(reply: str) -> str:
    try:
        source = json.loads(reply)["files"]["adaptive_system.py"]
        return source if isinstance(source, str) else ""
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def execute(source: str) -> tuple[bool, str]:
    if not source:
        return False, "no source"
    harness = source + r'''
async def _verify():
    coordinator = AdaptiveCoordinator()
    coordinator.migrate(2)
    attempts = {'count': 0}
    async def transient():
        attempts['count'] += 1
        if attempts['count'] < 2:
            raise RuntimeError('transient')
        return 'settled'
    command = {'key':'req-1','actor':'admin','source':'a','target':'b',
               'amount':20,'expected_version':1,'token':'secret'}
    first = await coordinator.process(command, transient)
    assert first['result'] == 'settled' and attempts['count'] == 2
    assert coordinator.balances == {'a':30,'b':25}
    assert len(coordinator.outbox) == 1 and coordinator.schema_version == 2
    replay = await coordinator.process(command, transient)
    assert replay == first and coordinator.balances == {'a':30,'b':25}
    assert json.loads(coordinator.logs[0])['command']['token'] == '[REDACTED]'
    denied = dict(command, key='req-2', actor='user', expected_version=2)
    try:
        await coordinator.process(denied, transient)
        raise AssertionError('authorization allowed')
    except PermissionError:
        pass
asyncio.run(_verify())
print('PASS')
'''
    with tempfile.TemporaryDirectory(prefix="wv-multidomain-") as raw:
        path = Path(raw) / "verify.py"
        path.write_text(harness, encoding="utf-8")
        import subprocess
        run = subprocess.run([sys.executable, "-I", str(path)], capture_output=True,
                             text=True, timeout=30)
    return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-1500:]


def execute_no_retry_contradiction(source: str) -> tuple[bool, str]:
    """Require an updated no-retry premise to change observable behavior."""
    if not source:
        return False, "no source"
    harness = source + r'''
async def _verify_no_retry():
    coordinator = AdaptiveCoordinator()
    attempts = {'count': 0}
    async def transient():
        attempts['count'] += 1
        raise RuntimeError('transient')
    command = {'key':'req-c','actor':'admin','source':'a','target':'b',
               'amount':1,'expected_version':1}
    try:
        await coordinator.process(command, transient)
    except RuntimeError:
        pass
    assert attempts['count'] == 1, attempts
asyncio.run(_verify_no_retry())
print('PASS')
'''
    with tempfile.TemporaryDirectory(prefix="wv-multidomain-contradiction-") as raw:
        path = Path(raw) / "verify.py"
        path.write_text(harness, encoding="utf-8")
        import subprocess
        run = subprocess.run([sys.executable, "-I", str(path)], capture_output=True,
                             text=True, timeout=30)
    return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-1500:]


def query(endpoint: str, prompt: str) -> dict:
    started = time.perf_counter()
    response = request(endpoint, "/brain/chat", {"text": prompt}, timeout=300)
    latency = time.perf_counter() - started
    source = extract_source(str(response.get("reply") or ""))
    passed, detail = execute(source)
    return {
        "executes": passed, "source": source,
        "detail": "" if passed else detail, "latency_seconds": round(latency, 4),
        "activated_concepts": response.get("activated_concepts") or [],
        "intent_diagnostics": response.get("intent_diagnostics") or {},
        "honest_oov": not response.get("reply") and bool(
            (response.get("grounding") or {}).get("outside_grounding")
        ),
    }


def active_training_pids(runtime: Path) -> list[int]:
    candidates = [runtime / "curriculum-supervisor.pid", *runtime.glob("*.pid")]
    active = set()
    for path in candidates:
        try:
            pid = int(path.read_text(encoding="ascii").strip())
            process = psutil.Process(pid)
            command = " ".join(process.cmdline())
            if ("programming_curriculum_supervisor.py" in command
                    or "drive_corpora_brain" in command):
                active.add(pid)
        except (ValueError, OSError, psutil.Error):
            continue
    return sorted(active)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--enterprise-gate", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/multidomain-synthesis.json"))
    args = parser.parse_args()
    if args.train and args.runtime is None:
        parser.error("--train requires --runtime for transactional rollback protection")
    if args.train:
        live_training = active_training_pids(args.runtime.resolve())
        if live_training:
            parser.error(
                "pause corpus training before multi-domain mutation; active PIDs: "
                + ",".join(map(str, live_training))
            )

    transaction = begin_experience_transaction(args.endpoint, args.runtime) if args.train else None
    retention_before = run_retention(
        args.endpoint, args.output.with_name("multidomain-foundation-before.json")
    ) if args.train else None
    tick_before = request(args.endpoint, "/brain/stats", None).get("tick")
    baseline = query(args.endpoint, synthesis_prompt())
    trained = train(args.endpoint, args.repeats) if args.train else None
    full = query(args.endpoint, synthesis_prompt())
    ablations = {
        premise.name: query(args.endpoint, synthesis_prompt(excluded=premise.name))
        for premise in DISCIPLINES
    }
    contradiction = query(args.endpoint, synthesis_prompt(contradiction=True))
    contradiction_pass, contradiction_detail = execute_no_retry_contradiction(
        contradiction["source"]
    )
    contradiction["updated_premise_executes"] = contradiction_pass
    contradiction["updated_premise_detail"] = (
        "" if contradiction_pass else contradiction_detail
    )
    tick_after = request(args.endpoint, "/brain/stats", None).get("tick")
    retention_after = run_retention(
        args.endpoint, args.output.with_name("multidomain-foundation-after.json")
    ) if args.train else None
    enterprise = run_enterprise_retention(
        args.endpoint, args.output.with_name("multidomain-enterprise-after.json")
    ) if args.train and args.enterprise_gate else None

    ablations_pass = all(not row["executes"] for row in ablations.values())
    retention_ok = retention_after is None or retention_passed(retention_after)
    enterprise_ok = enterprise is None or bool(enterprise.get("passed"))
    passed = full["executes"] and ablations_pass and contradiction_pass \
        and retention_ok and enterprise_ok
    report = {
        "passed": passed, "disciplines": [p.name for p in DISCIPLINES],
        "final_artifact_observed": full["source"] in {
            json.loads(response)["code_fragment"]["source"]
            for _, response in training_rows()
        },
        "baseline": baseline, "training": trained, "full": full,
        "ablations": ablations, "ablations_passed": ablations_pass,
        "contradiction": contradiction,
        "contradiction_propagated": contradiction_pass,
        "retention_before": retention_before, "retention_after": retention_after,
        "retention_passed": retention_ok, "enterprise_after": enterprise,
        "enterprise_passed": enterprise_ok, "tick_before": tick_before,
        "tick_after": tick_after,
        "tick_delta": tick_after - tick_before if isinstance(tick_before, int)
        and isinstance(tick_after, int) else None,
        "concurrent_mutation_detected": (
            not args.train and isinstance(tick_before, int)
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
        "passed": passed, "full_executes": full["executes"],
        "ablations_passed": ablations_pass,
        "contradiction_propagated": contradiction_pass,
        "retention_passed": retention_ok, "enterprise_passed": enterprise_ok,
        "tick_delta": report["tick_delta"],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
