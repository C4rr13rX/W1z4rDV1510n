python3 - <<'PY'
"""Does the CURRENTLY RUNNING brain compose dedup.go for the canonical row?

The composition-coverage fix (396cd85) is in the image the brain is executing
-- inode 1616920738 on disk and on /proc/<pid>/exe. That proves the process is
running the file on disk. It does NOT prove the file was built from the fix,
and it does not prove the fix works: the 2026-09-07 attempt at this same row
was deployed, loaded, and inert, because the component-count cap made its
`.rev()` loop single-valued.

So this asks the brain directly. The suite is read-only -- `/brain/chat` does
not observe or tick -- and it writes to /tmp so the gate's own artifact is not
clobbered by a diagnostic.

The measurement is the canonical `javascript_go_order_workers` row emitting
dedup.go and `go_deduplication` executing. A 6/6 summary is reported but is
not by itself the claim: the row identity is printed so a pass cannot be
confused with a suite that silently ran a different case.
"""
import json
import os
import subprocess
import time

PROJ = "/srv/wizard/project"
OUT = "/tmp/polyglot_fix_check.json"
out = {"now": time.time()}

out["host_git"] = {}
for label, cmd in (("head", ["git", "rev-parse", "HEAD"]),
                   ("subject", ["git", "log", "-1", "--format=%h %ci %s"]),
                   ("has_fix", ["git", "log", "--oneline", "-1", "396cd85"])):
    try:
        r = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True, timeout=60)
        out["host_git"][label] = (r.stdout or r.stderr or "").strip()[:200]
    except Exception as exc:  # noqa: BLE001
        out["host_git"][label] = f"ERR {exc}"

# Does the running image literally contain the fix's marker strings? A binary
# grep is weak evidence on its own but it is free, and it distinguishes
# "built from the fix" from "built from the parent commit" without a rebuild.
try:
    r = subprocess.run(
        ["grep", "-c", "-a", "instruction_intent:BEHAVIOUR",
         f"{PROJ}/target/release/w1z4rd_brain_server"],
        capture_output=True, text=True, timeout=120)
    out["binary_behaviour_marker_hits"] = (r.stdout or "").strip()
except Exception as exc:  # noqa: BLE001
    out["binary_behaviour_marker_hits"] = f"ERR {exc}"

env = dict(os.environ)
env["PYTHONPATH"] = f"{PROJ}/scripts:" + env.get("PYTHONPATH", "")
started = time.time()
try:
    proc = subprocess.run(
        ["python3", "scripts/programming_polyglot_composition.py",
         "--endpoint", "http://127.0.0.1:18095", "--output", OUT],
        cwd=PROJ, capture_output=True, text=True, timeout=900, env=env)
    out["exit_code"] = proc.returncode
    out["stdout"] = (proc.stdout or "").strip()[:1500]
    out["stderr_tail"] = (proc.stderr or "").strip()[-1500:]
except Exception as exc:  # noqa: BLE001
    out["run_error"] = str(exc)
out["elapsed_s"] = round(time.time() - started, 1)

try:
    with open(OUT, encoding="utf-8") as fh:
        report = json.load(fh)
    out["summary"] = report.get("summary")
    rows = []
    for row in report.get("results", []):
        rows.append({
            "name": row.get("name"), "kind": row.get("kind"),
            "executes": row.get("executes"), "files": row.get("files"),
            "components": [
                {"component": c.get("component"), "executes": c.get("executes"),
                 "detail": str(c.get("detail") or "")[:220]}
                for c in row.get("components", [])
            ],
        })
    out["rows"] = rows
    # The one row that has decided the gate for three days.
    out["target_row"] = next(
        (r for r in rows
         if r["name"] == "javascript_go_order_workers" and r["kind"] == "canonical"),
        None)
    out["oov_honest"] = [o.get("honest") for o in report.get("oov", [])]
except Exception as exc:  # noqa: BLE001
    out["report_error"] = str(exc)

print("PROBE_JSON " + json.dumps(out, default=str))
PY
