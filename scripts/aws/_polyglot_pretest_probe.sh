python3 - <<'PY'
"""Pre-test the ONE suite that has held the enterprise gate at 11/12.

Seven consecutive gate runs (29.8 h -> 6.8 h ago) scored first=11 confirm=11
passed=false, and both recent gate reports name the same failing suite:
`polyglot`. The go_systems_001 corpus now training exists to close exactly that
gap -- "golang integration transactional outbox" answered with order_service.js
under LANGUAGE:JAVASCRIPT because the outbox behaviour had only ever been
observed in JavaScript.

The gate is pre-testable, so measuring now predicts the 131072-row verdict
~1.5 h early instead of paying a rollback to learn it. Two cautions are
encoded here rather than assumed:

  * `programming_polyglot_composition.py` takes no `--no-train` flag because it
    is read-only by construction -- the enterprise runner passes `--no-train`
    only to the six suites that would otherwise mutate. `programming_code_eval`
    is NOT one of the safe ones: it calls refresh_routes() (observe+tick)
    unless `--no-train` is given, and the completion gate does not give it. So
    that suite is deliberately not run here.
  * The supervisor holds a `--min-free-memory-gb 3` floor and available memory
    is ~3.65 GB. Tripping it mid-interval produces a resource yield, which the
    replay-yield lesson recorded being scored as a semantic failure. So memory
    is sampled around the run and the probe refuses to start below a margin.

Read-only: no restart, no observe, no tick.
"""
import json
import os
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
PROJECT = "/srv/wizard/project"
ENDPOINT = "http://127.0.0.1:18095"
FLOOR_GB = 3.25
out = {"now": time.time()}


def avail_gb():
    for line in open("/proc/meminfo", encoding="utf-8"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024.0 * 1024.0)
    return 0.0


def forward_row():
    try:
        data = json.load(open(
            os.path.join(R, "curriculum-supervisor.status.json"), encoding="utf-8"))
        return data.get("durable_next_row"), data.get("state"), data.get("phase")
    except Exception:  # noqa: BLE001
        return None, None, None


out["avail_before"] = round(avail_gb(), 2)
out["row_before"], out["state_before"], out["phase"] = forward_row()

# ---- Confirm the deployed registry LOADS. Deploy is not load: a corrected
# file that never reached the host still runs the old code, and load_registry
# is all-or-nothing, so one bad file kills every corpus at driver startup.
reg = subprocess.run(
    ["/usr/bin/python3", "-c",
     "import sys; sys.path.insert(0,'.');"
     "from tools.training_standard.schema import load_registry;"
     "from pathlib import Path;"
     "r=load_registry(Path('tools/training_standard/registry'));"
     "import json;"
     "print(json.dumps({'count':len(r),"
     "'go':(r['go_systems_001'].category if 'go_systems_001' in r else None),"
     "'ids':sorted(r)[:40]}))"],
    cwd=PROJECT, capture_output=True, text=True, timeout=120)
out["registry_load"] = {
    "rc": reg.returncode,
    "stdout": reg.stdout.strip()[:600],
    "stderr": reg.stderr.strip()[-600:],
}
go_toml = os.path.join(PROJECT, "tools/training_standard/registry/go_systems_001.toml")
if os.path.exists(go_toml):
    text = open(go_toml, encoding="utf-8").read()
    out["deployed_go_toml"] = {
        "age_h": round((out["now"] - os.path.getmtime(go_toml)) / 3600.0, 1),
        "category_line": next((l.strip() for l in text.splitlines()
                               if l.strip().startswith("category")), None),
        "must_be_valid": sorted({l.split("=")[1].strip()
                                 for l in text.splitlines()
                                 if l.strip().startswith("must_be_valid")}),
        "owner": subprocess.run(["stat", "-c", "%U:%G", go_toml],
                                capture_output=True, text=True).stdout.strip(),
    }

# ---- The pre-test itself, only if there is memory headroom ---------------
if out["avail_before"] < FLOOR_GB:
    out["skipped"] = "available %.2f GB below %.2f GB floor" % (
        out["avail_before"], FLOOR_GB)
else:
    started = time.time()
    proc = subprocess.run(
        ["/usr/bin/python3", "scripts/programming_polyglot_composition.py",
         "--endpoint", ENDPOINT,
         "--output", os.path.join(R, "_pretest_polyglot.json")],
        cwd=PROJECT, capture_output=True, text=True, timeout=1500)
    out["polyglot"] = {
        "rc": proc.returncode,
        "elapsed_s": round(time.time() - started, 1),
        "stdout_tail": proc.stdout.strip()[-2500:],
        "stderr_tail": proc.stderr.strip()[-800:],
    }
    report_path = os.path.join(R, "_pretest_polyglot.json")
    if os.path.exists(report_path):
        try:
            report = json.load(open(report_path, encoding="utf-8"))
        except Exception as error:  # noqa: BLE001
            report = {"err": str(error)[:200]}
        rows = report.get("results") or report.get("rows") or []
        out["polyglot_cases"] = [
            {k: row.get(k) for k in
             ("case", "name", "kind", "passed", "components", "language_label",
              "executes", "exact")}
            for row in rows
        ] if isinstance(rows, list) else rows
        out["polyglot_summary"] = {
            k: report.get(k) for k in
            ("passed", "passed_cases", "total_cases", "oov_honest", "oov_total")
        }

out["avail_after"] = round(avail_gb(), 2)
out["row_after"], out["state_after"], _ = forward_row()
if out["row_before"] and out["row_after"]:
    out["forward_still_advancing"] = out["row_after"] > out["row_before"]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
