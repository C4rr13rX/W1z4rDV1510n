#!/bin/bash
# WHICH polyglot case is the 11/12, and has the Go corpus moved it?
# `_pretest_polyglot.json` was written 1.8 h ago against a brain that has now
# trained ~99k go-systems rows, while `polyglot.json` is the 8.7 h gate run.
# Diffing the two per-case verdicts is the only way to tell a corpus that is
# working slowly from one that is not working at all.
set -uo pipefail

python3 - <<'PY'
import glob
import json
import pathlib
import time

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}


def load(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def failures(body):
    """Every case this artifact recorded as not passing, whatever it calls them.

    The suite has renamed its keys before, so walk the structure rather than
    reaching for a path: any dict carrying a falsey `passed`/`ok`/`success`
    beside something name-like is a case verdict.
    """
    found = []

    def walk(node, trail):
        if isinstance(node, dict):
            verdict = None
            for key in ("passed", "ok", "success"):
                if key in node and isinstance(node[key], bool):
                    verdict = node[key]
                    break
            if verdict is False:
                found.append({
                    "trail": "/".join(trail)[:120],
                    "case": {k: str(v)[:600] for k, v in node.items()
                             if not isinstance(v, (dict, list))},
                })
            for key, value in node.items():
                walk(value, trail + [str(key)])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, trail + [str(index)])

    walk(body, [])
    return found


for name in ("_pretest_polyglot.json", "polyglot.json"):
    path = runtime / name
    if not path.exists():
        out[name] = {"missing": True}
        continue
    body = load(path)
    out[name] = {
        "age_hours": round((time.time() - path.stat().st_mtime) / 3600, 1),
        "top_keys": sorted(body) if isinstance(body, dict) else type(body).__name__,
        "summary": {k: v for k, v in body.items()
                    if not isinstance(v, (dict, list))} if isinstance(body, dict) else None,
        "failures": failures(body)[:6],
    }

print("PROBE_JSON " + json.dumps(out))
PY
