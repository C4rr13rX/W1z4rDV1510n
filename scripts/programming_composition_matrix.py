#!/usr/bin/env python3
"""Zero-shot matrix for independently learned enterprise module combinations."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from programming_project_eval import execute_project, request


@dataclass(frozen=True)
class Scenario:
    name: str
    prompts: tuple[str, str]
    required_files: tuple[str, ...]
    test: str


SCENARIOS = [
    Scenario(
        "authorization_logging",
        ("Build a Python project with default-deny authorization and structured correlation-ID logging with secret redaction.",
         "Create Python access control plus correlated JSON observability that masks secrets."),
        ("authorization.py", "observability.py"),
        "import io\nfrom authorization import is_authorized\nfrom observability import write_event\n"
        "assert not is_authorized({'id':'u','roles':['user']},'write','u')\n"
        "s=io.StringIO(); write_event(s,'denied','r-1',token='secret')\nassert 'secret' not in s.getvalue()\n",
    ),
    Scenario(
        "api_transaction",
        ("Build a Python project with an idempotent order API and an atomic SQLite transfer transaction.",
         "Create Python replay-safe API commands plus all-or-nothing database transfers."),
        ("api.py", "repository.py"),
        "import sqlite3,tempfile\nfrom pathlib import Path\nfrom api import OrderApi\nfrom repository import transfer\n"
        "a=OrderApi(); x=a.handle('POST','/orders',{'sku':'A','quantity':1},'k'); assert x==a.handle('POST','/orders',{'sku':'B','quantity':2},'k')\n"
        "db=str(Path(tempfile.mkdtemp())/'x.db')\nwith sqlite3.connect(db) as c:\n c.execute('CREATE TABLE accounts(id TEXT PRIMARY KEY,balance INTEGER)')\n c.executemany('INSERT INTO accounts VALUES(?,?)',[('a',5),('b',0)])\n"
        "transfer(db,'a','b',2)\n",
    ),
    Scenario(
        "authorized_api_logging",
        ("Build Python modules for authorization, an idempotent API, and correlated structured logging with redaction.",
         "Create a Python project combining access control, replay-safe API behavior, and JSON observability that masks secrets."),
        ("authorization.py", "api.py", "observability.py"),
        "import io\nfrom authorization import is_authorized\nfrom api import OrderApi\nfrom observability import write_event\n"
        "assert is_authorized({'id':'a','roles':['admin']},'create','x')\na=OrderApi(); assert a.handle('POST','/orders',{'sku':'A','quantity':1},'k')[0]==201\n"
        "s=io.StringIO(); write_event(s,'created','r',password='p'); assert '\"password\":\"p\"' not in s.getvalue()\n",
    ),
    Scenario(
        "transaction_logging",
        ("Build a Python project with atomic SQLite transfers and structured correlation-ID logging with secret redaction.",
         "Create Python all-or-nothing database transfer code plus correlated JSON observability that masks secrets."),
        ("repository.py", "observability.py"),
        "import io,sqlite3,tempfile\nfrom pathlib import Path\nfrom repository import transfer\nfrom observability import write_event\n"
        "db=str(Path(tempfile.mkdtemp())/'x.db')\nwith sqlite3.connect(db) as c:\n c.execute('CREATE TABLE accounts(id TEXT PRIMARY KEY,balance INTEGER)')\n c.executemany('INSERT INTO accounts VALUES(?,?)',[('a',3),('b',0)])\n"
        "transfer(db,'a','b',1)\ns=io.StringIO(); write_event(s,'transfer','r',api_key='x'); assert '[REDACTED]' in s.getvalue()\n",
    ),
]

OOV = "Build a Python project combining an idempotent API and database transaction for an unspecified protocol and schema."


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--endpoint",default="http://127.0.0.1:18600")
    parser.add_argument("--output",type=Path,default=Path("runtime/benchmarks/composition_matrix.json")); args=parser.parse_args()
    rows=[]
    for scenario in SCENARIOS:
        for index,prompt in enumerate(scenario.prompts):
            reply=str(request(args.endpoint,"/brain/chat",{"text":prompt}).get("reply") or "")
            passed,detail=execute_project(reply,scenario.test)
            try: files=sorted(json.loads(reply).get("files",{}).keys())
            except (json.JSONDecodeError,AttributeError): files=[]
            rows.append({"scenario":scenario.name,"kind":"canonical" if index==0 else "paraphrase",
                         "executes":passed,"required_present":all(f in files for f in scenario.required_files),
                         "files":files,"detail":"" if passed else detail})
    oov_result=request(args.endpoint,"/brain/chat",{"text":OOV})
    oov_honest=not oov_result.get("reply") and bool((oov_result.get("grounding") or {}).get("outside_grounding"))
    report={"passed":sum(r["executes"] and r["required_present"] for r in rows),"total":len(rows),
            "oov_honest":oov_honest,"results":rows}; args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({k:report[k] for k in ("passed","total","oov_honest")}))
    return 0 if report["passed"]==report["total"] and oov_honest else 1


if __name__=="__main__": raise SystemExit(main())
