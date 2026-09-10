#!/bin/bash
# The previous probe walked for a boolean `passed`/`ok`/`success` and found
# none, which is a VACUOUS zero rather than a clean suite: the summary says
# 11/12, so a failing case exists and is recorded under a key that walk did
# not match. Dump the actual shape instead of guessing at it again.
set -uo pipefail

python3 - <<'PY'
import json
import pathlib
import time

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}

for name in ("_pretest_polyglot.json", "polyglot.json"):
    path = runtime / name
    if not path.exists():
        out[name] = {"missing": True}
        continue
    body = json.loads(path.read_text(encoding="utf-8"))
    entry = {"age_hours": round((time.time() - path.stat().st_mtime) / 3600, 1)}
    entry["summary"] = body.get("summary")
    results = body.get("results")
    entry["results_type"] = type(results).__name__
    if isinstance(results, dict):
        entry["results_keys"] = sorted(results)
        first = next(iter(results.values()), None)
        entry["one_value_shape"] = (sorted(first) if isinstance(first, dict)
                                    else str(first)[:300])
        # Every entry, compressed: name plus every scalar field it carries.
        entry["entries"] = {
            key: ({k: str(v)[:200] for k, v in value.items()
                   if not isinstance(v, (dict, list))}
                  if isinstance(value, dict) else str(value)[:200])
            for key, value in results.items()}
    elif isinstance(results, list):
        entry["results_len"] = len(results)
        entry["entries"] = [
            ({k: str(v)[:200] for k, v in item.items()
              if not isinstance(v, (dict, list))}
             | {"_nested_keys": sorted(k for k, v in item.items()
                                       if isinstance(v, (dict, list)))})
            if isinstance(item, dict) else str(item)[:200]
            for item in results]
    entry["oov"] = str(body.get("oov"))[:400]
    out[name] = entry

print("PROBE_JSON " + json.dumps(out))
PY
