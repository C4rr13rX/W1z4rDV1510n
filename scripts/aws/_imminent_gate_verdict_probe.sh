python3 - <<'PY'
"""What will the gate say when `go-systems:0:131072` reaches it?

The interval is 45k rows from its midphase gate. Two artifacts written 1.7 h
ago -- after the current supervisor started -- already contain a verdict from
the same brain against the same suites, so the gate's answer is largely
knowable now instead of after another 45k rows of billed compute.

Read them at the level that actually carries a verdict. CLAUDE.md: the
`enterprise_gate_confirmation` ledger record carries only COUNTS; the
per-suite verdict is in `<phase>.enterprise-gate.json` under `results[].name`;
and the PER-CASE verdict is one level further out in the suite's own
`polyglot.json` under `results[].executes`, because `results[].name` there
carries only a boolean. A walk that stops at the wrong level reports a
vacuous zero.

Also: does `programming_enterprise_retention.py` unlink its fixed-path output
before writing? It is 674 h old and has no `unlink`. If a suite crashes
before writing, the reader picks up the PREVIOUS run's file -- the same trap
that left a 769.7 h `integrated_debug.json` reading 6/6 beside two quarantined
candidates, which would trade a false quarantine for a false admission.

Read-only.
"""
import json
import os
import subprocess
import time

P = "/srv/wizard/project"
R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}


def sh(command, timeout=60):
    try:
        return subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout).stdout.strip()
    except Exception as error:
        return f"<{type(error).__name__}: {str(error)[:80]}>"


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as error:
        return {"_error": f"{type(error).__name__}: {str(error)[:120]}"}


# --- the suite-level verdict --------------------------------------------------
gate_path = f"{R}/go-systems.enterprise-gate.json"
gate = load(gate_path)
if isinstance(gate, dict) and "_error" not in gate:
    results = gate.get("results") or []
    out["enterprise_gate"] = {
        "age_h": round((time.time() - os.path.getmtime(gate_path)) / 3600.0, 2),
        "passed": gate.get("passed"),
        "passed_suites": gate.get("passed_suites"),
        "total_suites": gate.get("total_suites"),
        "tick_delta": gate.get("tick_delta"),
        "infrastructure_only_failure": gate.get("infrastructure_only_failure"),
        "residency_after": gate.get("residency_after"),
        "failing": [r.get("name") for r in results if not r.get("passed")],
        "passing": [r.get("name") for r in results if r.get("passed")],
        "timed_out": [r.get("name") for r in results if r.get("timed_out")],
    }
else:
    out["enterprise_gate"] = gate

# --- the CASE-level verdict, one level further out ---------------------------
for label, name in (("polyglot", "polyglot.json"),
                    ("pretest_polyglot", "_pretest_polyglot.json")):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        out[label] = {"missing": True}
        continue
    body = load(path)
    rows = body.get("results") or [] if isinstance(body, dict) else []
    cases = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cases.append({
            "name": row.get("name"),
            "executes": row.get("executes"),
            "projects": f"{row.get('projects_ok')}/{row.get('projects_total')}",
            "components": f"{row.get('components_ok')}/{row.get('components_total')}",
            "oov": f"{row.get('oov_ok')}/{row.get('oov_total')}",
        })
    out[label] = {
        "age_h": round((time.time() - os.path.getmtime(path)) / 3600.0, 2),
        "passed": body.get("passed") if isinstance(body, dict) else None,
        "case_count": len(cases),
        "failing_cases": [c for c in cases if c.get("executes") is False],
        "executes_true": sum(1 for c in cases if c.get("executes") is True),
        "executes_missing": sum(1 for c in cases if c.get("executes") is None),
        "sample": cases[:3],
    }

# --- does the enterprise runner delete a stale report before writing? ---------
runner = f"{P}/scripts/programming_enterprise_retention.py"
body = open(runner, encoding="utf-8", errors="replace").read()
out["enterprise_runner"] = {
    "age_h": round((time.time() - os.path.getmtime(runner)) / 3600.0, 2),
    "has_unlink": "unlink" in body,
    "has_missing_ok": "missing_ok" in body,
    # Every fixed path it reads back from. A read-back with no prior delete is
    # where a leftover becomes a verdict.
    "readback_sites": [
        line.strip()[:150] for line in body.splitlines()
        if ("json.load" in line or "read_text" in line)
    ],
    "output_sites": [
        line.strip()[:150] for line in body.splitlines()
        if ("write_text" in line or "json.dump" in line)
    ],
}

# Which suites even exist, so a "12" can be checked against the roster.
out["suite_roster"] = sh(
    f"grep -oE '\\(\"[a-z_]+\", ' {runner} | head -20").splitlines()

# --- how close is the gate, and is the row still moving? ---------------------
try:
    first = load(f"{R}/deferred-replay-b01232b593532da2.resume.json")
    time.sleep(10)
    second = load(f"{R}/deferred-replay-b01232b593532da2.resume.json")
    moved = int(second["durable_next_row"]) - int(first["durable_next_row"])
    out["replay"] = {
        "durable_next_row": second["durable_next_row"],
        "end_row": second["end_row"],
        "rows_to_gate": int(second["end_row"]) - int(second["durable_next_row"]),
        "rows_per_second_10s": round(moved / 10.0, 2),
        "marker_age_s": round(time.time() - float(second["updated_unix"]), 1),
    }
except Exception as error:
    out["replay_error"] = str(error)[:150]

out["mem_available_gb"] = sh("awk '/MemAvailable/{print $2/1048576}' /proc/meminfo")

print("PROBEJSON " + json.dumps(out))
PY
