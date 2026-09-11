python3 - <<'PY'
"""Which SUITE fails the 11/12 gate on this interval, and which case inside it?

The ledger's `enterprise regression` string carries the whole gate payload as a
repr, but every consumer so far has read only its head -- `passed_suites: 11,
total_suites: 12` and a suite name cut mid-word at "pyth". A count names no
suite, and CLAUDE.md already records that the per-suite verdict lives in
`results[].name` with the PER-CASE verdict one level further out in the suite's
own report, where `results[].name` carries only a boolean.

So parse the embedded repr with `ast.literal_eval` (it is Python repr, not
JSON) and report every failing suite for all 14 failures, plus the freshest
on-disk gate artifact, which holds the per-case detail the ledger truncates.

Dates every failure against the supervisor and against the go-systems
admissions: a ledger entry is not evidence about the process that is running,
and the most recent of these is 86 h old.
"""
import ast
import collections
import json
import os
import re
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def load(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as exc:
        out.setdefault("errors", []).append(f"{path}: {exc}")
    return rows


health = load(f"{R}/curriculum-health.jsonl")
current = "jupyter-scientific-full:201344:262144"
try:
    with open(f"{R}/deferred-replay-active.json", "r", encoding="utf-8") as fh:
        current = str(json.load(fh).get("interval_id") or current)
except Exception:
    pass
out["current_interval"] = current

failures = [r for r in health
            if str(r.get("interval_id") or "") == current
            and str(r.get("kind") or "") == "deferred_replay_failed"]

failing_suites = collections.Counter()
detail = []
for record in failures:
    error = str(record.get("error") or "")
    match = re.search(r"\{.*\}\s*$", error, re.S)
    payload = {}
    if match:
        try:
            payload = ast.literal_eval(match.group(0))
        except Exception as exc:
            payload = {"_parse_error": f"{type(exc).__name__}: {exc}"}
    results = payload.get("results") or []
    failed = []
    for suite in results:
        if not isinstance(suite, dict):
            continue
        name = str(suite.get("name") or "?")
        ok = suite.get("passed")
        if ok is None:
            ok = suite.get("ok")
        if ok is False:
            failed.append(name)
            failing_suites[name] += 1
    detail.append({
        "hours_ago": round(
            (out["now"] - float(record.get("updated_unix") or 0)) / 3600.0, 2),
        "passed_suites": payload.get("passed_suites"),
        "total_suites": payload.get("total_suites"),
        "infrastructure_only": payload.get("infrastructure_only_failure"),
        "failing_suites": failed,
        "suite_names": [str(s.get("name")) for s in results
                        if isinstance(s, dict)],
        "evidence_dir": record.get("evidence_dir"),
    })

out["failing_suite_counts"] = dict(failing_suites)
out["failures"] = detail

# The freshest on-disk gate artifact holds the per-CASE verdict.
artifacts = []
for root, _dirs, names in os.walk(R):
    for name in names:
        if "enterprise-gate" in name and name.endswith(".json"):
            path = os.path.join(root, name)
            try:
                artifacts.append((os.path.getmtime(path), path))
            except OSError:
                pass
artifacts.sort(reverse=True)
out["gate_artifact_count"] = len(artifacts)
out["gate_artifacts_recent"] = [
    {"path": path, "age_h": round((out["now"] - mtime) / 3600.0, 2)}
    for mtime, path in artifacts[:6]
]
if artifacts:
    _mtime, path = artifacts[0]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            gate = json.load(fh)
        out["freshest_gate"] = {
            "path": path,
            "age_h": round((out["now"] - _mtime) / 3600.0, 2),
            "passed": gate.get("passed"),
            "passed_suites": gate.get("passed_suites"),
            "total_suites": gate.get("total_suites"),
            "suites": [
                {"name": s.get("name"), "passed": s.get("passed"),
                 "report": s.get("report") or s.get("report_path")}
                for s in (gate.get("results") or []) if isinstance(s, dict)
            ],
        }
    except Exception as exc:
        out["freshest_gate_error"] = f"{type(exc).__name__}: {exc}"

# Date everything against the running supervisor and the go-systems admissions.
admitted = [r for r in health
            if str(r.get("kind") or "") == "deferred_replay_admitted"]
out["recent_admissions"] = [
    {"interval_id": r.get("interval_id"),
     "hours_ago": round((out["now"] - float(r.get("updated_unix") or 0)) / 3600.0, 2)}
    for r in admitted[-4:]]
out["newest_failure_hours_ago"] = min(
    (d["hours_ago"] for d in detail), default=None)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
