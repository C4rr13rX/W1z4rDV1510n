#!/bin/bash
# Is the supervisor I just started actually training, or crash-looping on a
# quarantine latch that predates it?
#
# The service log is append-only across days, so its tail proves nothing about
# the current generation. Everything here is measured against the running
# process's own start time.
set -uo pipefail

RUNTIME=/srv/wizard/runtime/programming-integrated-20260713

python3 - <<'PY'
import glob
import json
import os
import pathlib
import subprocess
import time

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}

def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr or "").strip()
    except Exception as exc:
        return f"ERR {exc}"

# 1) The current generation: pid and how long it has been up.
pids = sh("pgrep", "-f", "programming_curriculum_supervisor").split()
out["supervisor_pids"] = pids
proc = {}
for pid in pids[:4]:
    try:
        stat_path = pathlib.Path(f"/proc/{pid}")
        proc[pid] = {
            "age_seconds": round(time.time() - stat_path.stat().st_mtime, 1),
            "cmd": (stat_path / "cmdline").read_bytes().replace(b"\0", b" ")
                   .decode("utf-8", "replace")[:200],
        }
    except Exception as exc:
        proc[pid] = {"error": str(exc)}
out["supervisor_procs"] = proc
out["driver_pids"] = sh("pgrep", "-f", "drive_corpora_brain").split()

out["service_active"] = sh("systemctl", "is-active", "wizard-curriculum-supervisor")
out["service_since"] = sh("systemctl", "show", "-p", "ActiveEnterTimestampMonotonic",
                          "-p", "NRestarts", "-p", "ExecMainStatus",
                          "-p", "ExecMainStartTimestamp",
                          "wizard-curriculum-supervisor")

# 2) Only the log written since the service came up.
started = sh("systemctl", "show", "-p", "ExecMainStartTimestampMonotonic", "--value",
             "wizard-curriculum-supervisor")
log = runtime / "curriculum-service.stderr.log"
if log.is_file():
    out["log_size"] = log.stat().st_size
    out["log_mtime_age"] = round(time.time() - log.stat().st_mtime, 1)
    body = log.read_text(encoding="utf-8", errors="replace")
    out["log_tail"] = body[-1500:]
    # How many times has the quarantine assertion fired, ever?
    out["quarantine_raises_total"] = body.count(
        "unresolved continuous-canary quarantine")

# 3) The quarantine record itself: what is it, and when was it made?
qfiles = {}
for pattern in ("quarantine*.json", "*quarantine*.json", "status.json",
                "deferred-replay-active.json"):
    for path in glob.glob(str(runtime / pattern)):
        p = pathlib.Path(path)
        try:
            st = p.stat()
            blob = json.loads(p.read_text(encoding="utf-8"))
            qfiles[p.name] = {
                "mtime": st.st_mtime,
                "age_seconds": round(time.time() - st.st_mtime, 1),
                "state": blob.get("state"),
                "keys": sorted(blob)[:14],
            }
        except Exception as exc:
            qfiles[p.name] = {"error": str(exc)}
out["runtime_state_files"] = qfiles

# 4) Is any row actually being posted right now?  The progress file advancing
#    is the only thing that distinguishes training from a tight restart loop.
progs = sorted(glob.glob(str(runtime / "deferred-replay-*.progress.json")),
               key=os.path.getmtime, reverse=True)[:3]
snap = {}
for path in progs:
    p = pathlib.Path(path)
    try:
        snap[p.name] = {
            "age_seconds": round(time.time() - p.stat().st_mtime, 1),
            "body": json.loads(p.read_text(encoding="utf-8")),
        }
    except Exception as exc:
        snap[p.name] = {"error": str(exc)}
out["progress_files"] = snap

print("PROBE_JSON " + json.dumps(out, default=str)[:7000])
PY
