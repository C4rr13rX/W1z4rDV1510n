#!/bin/bash
# Confirm the falsely-rejected intervals are still owed, then clear the
# exit-42 latch and start the supervisor.
#
# The 26 rejections were never losses: `rejected_this_pass` is in-memory only
# and a rejection re-appends the interval to the append-only ledger as
# `deferred`, so `unresolved_deferred_intervals()` should still return them.
# That claim is checked here rather than trusted -- if it is wrong, the restart
# would quietly skip 12 intervals' worth of real training.
set -uo pipefail

PROJ=/srv/wizard/project
RUNTIME=/srv/wizard/runtime/programming-integrated-20260713

python3 - <<'PY'
import json
import pathlib
import sys

proj = pathlib.Path("/srv/wizard/project")
runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
sys.path.insert(0, str(proj))

out = {}
try:
    from scripts import programming_curriculum_supervisor as sup
    pending = sup.unresolved_deferred_intervals(runtime)
    out["unresolved_count"] = len(pending)
    out["unresolved_ids"] = sorted(str(p.get("interval_id")) for p in pending)[:40]
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"

# The 26 the completed pass reported as rejected.
status = runtime / "programming-curriculum-status.json"
rejected = []
for candidate in (status, runtime / "status.json"):
    if candidate.is_file():
        try:
            blob = json.loads(candidate.read_text(encoding="utf-8"))
            rejected = blob.get("rejected_intervals") or []
            out["status_file"] = str(candidate)
            out["status_state"] = blob.get("state")
            break
        except Exception:
            continue
out["rejected_reported"] = len(rejected)

owed = set(out.get("unresolved_ids") or [])
try:
    from scripts import programming_curriculum_supervisor as sup
    owed = {str(p.get("interval_id")) for p in sup.unresolved_deferred_intervals(runtime)}
except Exception:
    pass
out["rejected_still_owed"] = sorted(set(rejected) & owed)
out["rejected_dropped"] = sorted(set(rejected) - owed)

print("PROBE_JSON " + json.dumps(out, default=str))
PY

echo "--- clearing exit-42 latch and starting ---"
systemctl reset-failed wizard-curriculum-supervisor 2>&1
systemctl start wizard-curriculum-supervisor 2>&1
sleep 20
echo "is-active: $(systemctl is-active wizard-curriculum-supervisor)"
systemctl show -p Result -p ExecMainStatus -p NRestarts -p ActiveState \
    wizard-curriculum-supervisor 2>&1
echo "--- procs ---"
echo "supervisor=$(pgrep -fc programming_curriculum_supervisor || echo 0)"
echo "driver=$(pgrep -fc drive_corpora_brain || echo 0)"
echo "--- service stderr tail ---"
tail -n 25 "$RUNTIME/curriculum-service.stderr.log" 2>/dev/null
