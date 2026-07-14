#!/usr/bin/env python3
"""Execute never-trained projects composed from independently learned subsystems."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from programming_project_eval import execute_project, request


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    paraphrase: str
    integration_test: str
    required_files: tuple[str, ...]


CASES = [
    Case(
        "resilient_observability",
        "Build a Python project combining a circuit breaker with correlated structured logging that recursively redacts secrets.",
        "Create Python resilience and observability modules with a cooldown circuit and correlated logs that mask nested credentials.",
        "import io, json\nfrom circuit import CircuitBreaker, CircuitOpen\n"
        "from observability import write_event\n"
        "now=[0.0]; breaker=CircuitBreaker(2,10,lambda:now[0])\n"
        "def fail(): raise RuntimeError('down')\n"
        "for _ in range(2):\n    try: breaker.call(fail)\n    except RuntimeError: pass\n"
        "try:\n    breaker.call(lambda:'bad')\n    raise AssertionError('circuit stayed closed')\nexcept CircuitOpen:\n    pass\n"
        "stream=io.StringIO(); write_event(stream,'circuit_open','req-1',token='secret')\n"
        "record=json.loads(stream.getvalue()); assert record['correlation_id']=='req-1'\n"
        "assert record['token']=='[REDACTED]'\n",
        ("circuit.py", "observability.py"),
    ),
    Case(
        "authorized_transfer",
        "Create a Python project with a SQLite atomic balance-transfer repository and default-deny authorization for transfer commands.",
        "Build Python database transaction and access-control modules: transfers are all-or-nothing and permissions deny by default.",
        "import sqlite3, tempfile\nfrom pathlib import Path\n"
        "from repository import transfer\nfrom authorization import is_authorized\n"
        "assert is_authorized({'id':'admin','roles':['admin']},'transfer','other')\n"
        "assert not is_authorized({'id':'u','roles':['user']},'transfer','u')\n"
        "db=str(Path(tempfile.mkdtemp())/'db.sqlite')\n"
        "with sqlite3.connect(db) as c:\n    c.execute('CREATE TABLE accounts(id TEXT PRIMARY KEY,balance INTEGER NOT NULL)')\n    c.executemany('INSERT INTO accounts VALUES(?,?)',[('a',50),('b',5)])\n"
        "transfer(db,'a','b',20)\n"
        "with sqlite3.connect(db) as c:\n    assert c.execute('SELECT balance FROM accounts WHERE id=\"a\"').fetchone()[0]==30\n",
        ("repository.py", "authorization.py"),
    ),
    Case(
        "bounded_audit_pipeline",
        "Implement a Python project that runs work with bounded asynchronous concurrency and writes correlated JSON audit logs with secret redaction.",
        "Create a Python async project using a semaphore to limit parallel work plus correlated audit logging that masks secrets.",
        "import asyncio, io, json\nfrom concurrency import bounded_map\n"
        "from observability import write_event\n"
        "async def main():\n"
        "    async def worker(value):\n        await asyncio.sleep(0); return value*2\n"
        "    assert await bounded_map(worker,[1,2,3],2)==[2,4,6]\n"
        "asyncio.run(main())\n"
        "stream=io.StringIO(); write_event(stream,'batch','req-2',password='p')\n"
        "assert json.loads(stream.getvalue())['password']=='[REDACTED]'\n",
        ("concurrency.py", "observability.py"),
    ),
]

OOV = [
    "Build a Python project combining an undocumented payment protocol with unspecified settlement rules.",
    "Create a multi-service Python integration whose external API contracts have not been provided.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runtime/benchmarks/cross_project_composition.json"),
    )
    args = parser.parse_args()
    rows = []
    for case in CASES:
        for kind, prompt in (("canonical", case.prompt), ("paraphrase", case.paraphrase)):
            response = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(response.get("reply") or "")
            passed, detail = execute_project(reply, case.integration_test)
            files = {}
            try:
                files = json.loads(reply).get("files", {})
            except (json.JSONDecodeError, AttributeError):
                pass
            rows.append({
                "name": case.name,
                "kind": kind,
                "executes": passed,
                "required_files": all(name in files for name in case.required_files),
                "files": sorted(files),
                "detail": "" if passed else detail,
                "intent_diagnostics": response.get("intent_diagnostics"),
            })
    oov = []
    for prompt in OOV:
        response = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not response.get("reply") and bool(
            (response.get("grounding") or {}).get("outside_grounding")
        )
        oov.append({"prompt": prompt, "honest": honest, "reply": response.get("reply")})
    summary = {
        "novel_compositions": {"passed": sum(row["executes"] for row in rows), "total": len(rows)},
        "oov_honesty": {"passed": sum(row["honest"] for row in oov), "total": len(oov)},
    }
    report = {"summary": summary, "results": rows, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in rows) and all(row["honest"] for row in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
