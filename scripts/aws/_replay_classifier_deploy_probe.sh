python3 - <<'PY'
"""Decide whether the replay worker-exit classifier is actually LOADED.

The corrected failure bucketing (see `_forward_convergence_probe.sh`) showed
317 of 324 `deferred_replay_failed` events are `worker_exit_or_signal` -- the
exact class `replay_worker_failure()` exists to reclassify as infrastructure.
Yet `deferred_replay_infrastructure_paused` appears ZERO times in the ledger,
and the newest worker-exit rejection is only hours old. Two stories fit:

  A. The fix is not deployed, or is deployed but the running supervisor
     predates the file (`deploy_is_not_load`: a fix copied but never loaded
     runs the OLD code).
  B. The fix IS live and these events are all pre-fix history
     (`failure_ledger_predates_process`: `last_failure` can be entirely
     stale; timestamp it against process start before re-debugging).

These have opposite remedies -- restart the supervisor vs. do nothing -- so
the probe refuses to guess. It compares three clocks: when the classifier
entered the deployed FILE, when the running supervisor process started, and
when each worker-exit rejection was written. A rejection is only evidence of
a live defect if it postdates BOTH the file and the process.

Read-only. Nothing is restarted here: a restart during `state: training`
discards the whole interval.
"""
import collections
import json
import os
import re
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
SUP = "/srv/wizard/project/scripts/programming_curriculum_supervisor.py"
out = {"now": time.time()}


def sh(cmd, timeout=60):
    try:
        done = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()
    except Exception as error:  # noqa: BLE001
        return "<error: %s>" % error


# ---- 1. Does the deployed FILE carry the classifier? ----------------------
# Checked by source text, not by mtime arithmetic: mtime tracks the copy, and
# a copy of the wrong revision has a fresh mtime too.
try:
    source = open(SUP, encoding="utf-8", errors="replace").read()
    out["supervisor_file"] = {
        "path": SUP,
        "mtime": os.path.getmtime(SUP),
        "age_h": round((out["now"] - os.path.getmtime(SUP)) / 3600.0, 1),
        "size": len(source),
        "has_replay_worker_failure": "def replay_worker_failure" in source,
        "has_stderr_tail": "def replay_worker_stderr_tail" in source,
        "raises_classifier": "raise replay_worker_failure" in source,
        "has_infrastructure_paused_kind":
            "deferred_replay_infrastructure_paused" in source,
    }
except OSError as error:
    out["supervisor_file"] = {"error": str(error)}

# ---- 2. When did the RUNNING supervisor start? ---------------------------
# etimes is seconds-since-start, which is immune to clock skew between the
# host and this laptop -- unlike comparing a parsed lstart to time.time().
pid_line = sh(
    "ps -eo pid,etimes,args | grep '[p]rogramming_curriculum_supervisor.py' "
    "| head -1")
out["supervisor_ps"] = pid_line
process_started = None
if pid_line:
    parts = pid_line.split(None, 2)
    if len(parts) >= 2 and parts[1].isdigit():
        process_started = out["now"] - int(parts[1])
        out["supervisor_process"] = {
            "pid": parts[0],
            "etimes_s": int(parts[1]),
            "started_unix": process_started,
            "age_h": round(int(parts[1]) / 3600.0, 1),
        }

# ---- 3. Timestamp every worker-exit rejection against both clocks --------
ledger = os.path.join(R, "curriculum-health.jsonl")
file_started = (out.get("supervisor_file") or {}).get("mtime") or 0
kinds = collections.Counter()
exits = []
suites = collections.Counter()
if os.path.exists(ledger):
    for line in open(ledger, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        kind = str(row.get("kind") or row.get("event") or "")
        kinds[kind] += 1
        if kind not in ("deferred_replay_failed",
                        "deferred_replay_infrastructure_paused"):
            continue
        error = str(row.get("error") or "")
        for name in re.findall(
                r"'name':\s*'([^']+)'[^}]*?'passed':\s*False", error):
            suites[name] += 1
        if "exited" not in error:
            continue
        when = float(row.get("updated_unix") or 0)
        exits.append({
            "unix": when,
            "age_h": round((out["now"] - when) / 3600.0, 1),
            "kind": kind,
            "phase": row.get("phase"),
            "after_file": bool(when and when > file_started),
            "after_process": bool(
                when and process_started and when > process_started),
            "stderr": (re.search(r"stderr=(\S+)", error).group(1)
                       if "stderr=" in error else ""),
        })

exits.sort(key=lambda item: item["unix"])
out["ledger_kind_counts"] = dict(kinds.most_common(40))
out["worker_exit_total"] = len(exits)
# The load-bearing count. If this is 0, every worker-exit rejection is stale
# history and there is nothing to repair on this path.
out["worker_exit_after_file"] = sum(1 for e in exits if e["after_file"])
out["worker_exit_after_process"] = sum(1 for e in exits if e["after_process"])
out["worker_exit_recent"] = exits[-12:]
out["failing_suites"] = dict(suites.most_common(20))

# ---- 4. What did the newest worker exits actually say? -------------------
# The rule is: never blame interval content before reading the worker's
# stderr. Tails are read for the newest distinct logs only.
tails = {}
for entry in reversed(exits):
    path = entry.get("stderr")
    if not path or path in tails or len(tails) >= 3:
        continue
    if os.path.exists(path):
        tails[path] = {
            "age_h": round((out["now"] - os.path.getmtime(path)) / 3600.0, 1),
            "tail": sh("tail -c 2500 %s" % path).splitlines()[-20:],
        }
    else:
        tails[path] = {"missing": True}
out["stderr_tails"] = tails

out["service"] = sh(
    "systemctl show wizard-curriculum-supervisor -p ActiveState -p SubState "
    "-p NRestarts -p ExecMainStartTimestamp --no-pager").splitlines()

print("PROBE_JSON " + json.dumps(out, default=str))
PY
