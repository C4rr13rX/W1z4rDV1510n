#!/usr/bin/env python3
"""Evaluate zero-shot project composition from independently learned bindings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from programming_project_eval import execute_project, request

PROMPTS = [
    "Build a Python multi-file project containing default-deny authorization, an idempotent order API, an atomic SQLite transfer transaction, and structured correlation-ID logging with secret redaction.",
    "Create Python code in multiple files combining access control, replay-safe API commands, all-or-nothing database transfers, and correlated JSON observability that masks secrets.",
]

INTEGRATION_TEST = (
    "import io, sqlite3, tempfile\nfrom pathlib import Path\n"
    "from authorization import is_authorized\nfrom api import OrderApi\n"
    "from repository import transfer\nfrom observability import write_event\n"
    "assert is_authorized({'id':'admin','roles':['admin']},'delete','x')\n"
    "assert not is_authorized({'id':'u','roles':['user']},'write','u')\n"
    "api=OrderApi(); first=api.handle('POST','/orders',{'sku':'A','quantity':1},'key')\n"
    "assert first == api.handle('POST','/orders',{'sku':'B','quantity':9},'key') and len(api.orders)==1\n"
    "db=str(Path(tempfile.mkdtemp())/'db.sqlite')\n"
    "with sqlite3.connect(db) as c:\n    c.execute('CREATE TABLE accounts(id TEXT PRIMARY KEY,balance INTEGER)')\n    c.executemany('INSERT INTO accounts VALUES(?,?)',[('a',10),('b',0)])\n"
    "transfer(db,'a','b',4)\n"
    "with sqlite3.connect(db) as c:\n    assert c.execute('SELECT balance FROM accounts WHERE id=\"a\"').fetchone()[0]==6\n"
    "stream=io.StringIO(); write_event(stream,'transfer','req-1',token='secret')\n"
    "assert 'secret' not in stream.getvalue() and 'req-1' in stream.getvalue()\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/composition.json"))
    args = parser.parse_args()
    results = []
    for index, prompt in enumerate(PROMPTS):
        reply = str(request(args.endpoint, "/brain/chat", {"text": prompt}).get("reply") or "")
        passed, detail = execute_project(reply, INTEGRATION_TEST)
        files = []
        try:
            files = sorted(json.loads(reply).get("files", {}).keys())
        except (json.JSONDecodeError, AttributeError):
            pass
        results.append({"kind": "canonical" if index == 0 else "paraphrase",
                        "executes": passed, "files": files,
                        "detail": "" if passed else detail})
    report = {"passed": sum(row["executes"] for row in results),
              "total": len(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"]}))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
