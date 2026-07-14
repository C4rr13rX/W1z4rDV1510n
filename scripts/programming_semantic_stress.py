#!/usr/bin/env python3
"""Diagnostic: quantify coding recall when prompts avoid hand-authored intent cues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from programming_project_eval import request

CASES = [
    ("authorization.py", "Build a Python project where operations are refused unless the caller is an administrator, except people may view records they own."),
    ("api.py", "Build a Python order project where repeating the same client command must return the first result without creating another order."),
    ("repository.py", "Build Python balance-movement code where debit and credit either both happen or neither happens when an account is invalid."),
    ("observability.py", "Build a Python module emitting machine-readable audit records carrying a request identifier while removing credentials from nested data."),
]


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--endpoint",default="http://127.0.0.1:18600")
    parser.add_argument("--output",type=Path,default=Path("runtime/benchmarks/semantic_stress.json")); args=parser.parse_args()
    rows=[]
    for expected,prompt in CASES:
        result=request(args.endpoint,"/brain/chat",{"text":prompt}); reply=str(result.get("reply") or "")
        try: files=sorted(json.loads(reply).get("files",{}).keys())
        except (json.JSONDecodeError,AttributeError): files=[]
        rows.append({"expected":expected,"recalled":expected in files,"files":files,
                     "outside_grounding":bool((result.get("grounding") or {}).get("outside_grounding"))})
    report={"recalled":sum(row["recalled"] for row in rows),"total":len(rows),"results":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"recalled":report["recalled"],"total":report["total"]}))
    return 0


if __name__=="__main__": raise SystemExit(main())
