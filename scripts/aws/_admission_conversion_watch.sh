python3 - <<'PY'
"""Does the reordered queue ADMIT, or has it only stopped grinding?

CLAUDE.md's standing rule: never report training as working without checking
that it CONVERTS. A healthy-looking curriculum that admits nothing has cost
real money twice, and every liveness signal here -- unit active, row moving,
episodes rising -- was already true throughout the 437 h this interval spent
never reaching a verdict. The measurement is the `resolved`/`admitted` count
RISING, not that a process is alive.

So: count admissions and resolutions before, watch, count after. Also reports
which intervals were selected during the window, because the point of the size
key is that small spans reach their gate inside one generation.
"""
import collections
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
WATCH = 420.0
out = {"now": time.time()}


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout + p.stderr).strip()[-600:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def tallies():
    kinds = collections.Counter()
    admitted = []
    try:
        with open(f"{R}/curriculum-health.jsonl", encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                kind = str(event.get("kind") or "")
                kinds[kind] += 1
                if kind == "deferred_replay_admitted":
                    admitted.append(str(event.get("interval_id") or ""))
    except OSError:
        pass
    return kinds, admitted


def unresolved():
    current = {}
    try:
        with open(f"{R}/curriculum-deferred-intervals.jsonl",
                  encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                interval_id = event.get("interval_id")
                if not isinstance(interval_id, str) or not interval_id:
                    continue
                if event.get("status") == "resolved":
                    current.pop(interval_id, None)
                elif event.get("status") == "deferred":
                    current[interval_id] = event
    except OSError:
        pass
    return current


kinds_a, admitted_a = tallies()
open_a = unresolved()
free_a = shutil.disk_usage(R).free

selected = []
t0 = time.time()
while time.time() - t0 < WATCH:
    try:
        with open(f"{R}/curriculum-supervisor.status.json",
                  encoding="utf-8") as fh:
            status = json.load(fh)
        entry = (str(status.get("interval_id") or ""),
                 str(status.get("state") or ""))
        if not selected or selected[-1] != entry:
            selected.append(entry)
    except Exception:
        pass
    time.sleep(5)

kinds_b, admitted_b = tallies()
open_b = unresolved()
free_b = shutil.disk_usage(R).free
elapsed = time.time() - t0

out["watch_seconds"] = round(elapsed, 1)
out["admissions_before"] = len(admitted_a)
out["admissions_after"] = len(admitted_b)
out["admitted_during_window"] = admitted_b[len(admitted_a):]
out["unresolved_before"] = len(open_a)
out["unresolved_after"] = len(open_b)
out["resolved_during_window"] = sorted(set(open_a) - set(open_b))
out["selected_sequence"] = selected[:40]
out["distinct_intervals_selected"] = len(
    {interval for interval, _state in selected})
out["free_gb"] = round(free_b / 2**30, 2)
out["burn_gb_per_hour"] = round(
    (free_a - free_b) / 2**30 / (elapsed / 3600.0), 2)
out["kind_deltas"] = {
    kind: kinds_b[kind] - kinds_a.get(kind, 0)
    for kind in kinds_b if kinds_b[kind] != kinds_a.get(kind, 0)
}
out["unit"] = sh("systemctl show wizard-curriculum-supervisor.service "
                 "-p ActiveState -p SubState -p NRestarts --no-pager")
out["converting"] = (out["admissions_after"] > out["admissions_before"]
                     or bool(out["resolved_during_window"]))

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
