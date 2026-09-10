#!/bin/bash
# Which stage is the restarted supervisor actually in, and on which corpus?
# The brain ticking proves learning; it does not say whether the work is the
# forward remainder or the 26 owed replay intervals.
set -uo pipefail

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
        return (r.stdout or "").strip()
    except Exception:
        return ""


# What is the driver being asked to do, verbatim?
cmds = {}
for pid in sh("pgrep", "-f", "drive_corpora_brain").split():
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
        cmds[pid] = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except Exception as exc:
        cmds[pid] = f"ERR {exc}"
out["driver_cmdlines"] = cmds

for pid in sh("pgrep", "-f", "programming_curriculum_supervisor").split():
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
        out["supervisor_cmdline"] = raw.replace(b"\0", b" ").decode(
            "utf-8", "replace").strip()
    except Exception:
        pass

# The freshest runtime state files -- whichever the current pass is writing.
fresh = []
for path in glob.glob(str(runtime / "*.json")):
    p = pathlib.Path(path)
    try:
        age = time.time() - p.stat().st_mtime
        if age < 3600:
            body = json.loads(p.read_text(encoding="utf-8"))
            fresh.append({
                "name": p.name,
                "age_seconds": round(age, 1),
                "state": body.get("state"),
                "phase": body.get("phase"),
                "row": body.get("row") or body.get("durable_next_row"),
                "keys": sorted(body)[:12],
            })
    except Exception:
        continue
out["fresh_state_files"] = sorted(fresh, key=lambda r: r["age_seconds"])[:12]

# Forward progress files (distinct from deferred-replay ones).
fwd = []
for path in glob.glob(str(runtime / "*progress*.json")):
    p = pathlib.Path(path)
    age = time.time() - p.stat().st_mtime
    if age < 3600:
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            body = {}
        fwd.append({
            "name": p.name, "age_seconds": round(age, 1),
            "corpus": os.path.basename(str(body.get("corpus") or "")),
            "durable_next_row": body.get("durable_next_row"),
            "accepted_episodes": body.get("accepted_episodes"),
            "script_id": body.get("script_id"),
        })
out["fresh_progress"] = sorted(fwd, key=lambda r: r["age_seconds"])[:10]

# Anything written to the service stdout log since the restart tells us the
# stage banner directly.
stdout_log = runtime / "curriculum-service.stdout.log"
if stdout_log.is_file():
    body = stdout_log.read_text(encoding="utf-8", errors="replace")
    out["stdout_tail"] = body[-2000:]
    out["stdout_age"] = round(time.time() - stdout_log.stat().st_mtime, 1)

print("PROBE_JSON " + json.dumps(out, default=str)[:6500])
PY
