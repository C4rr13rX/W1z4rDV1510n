python3 - <<'PY'
"""Separate a canary that is grinding from a canary that is hung.

`continuous_canary` held row 65536 for 75 minutes with the status file stale
the whole time. The status file freezing during a canary is by design -- the
supervisor writes it only between batches -- so staleness alone is not a
fault, and the row not moving is not a fault either: the canary deliberately
stops the forward worker. Both of the readings that look alarming are the
expected shape, which is exactly why this needs a measurement that can come
back negative.

A canary that is working spawns gate subprocesses that start and exit, and
burns CPU in the brain. A canary that is hung has a stable process set and a
flat CPU counter. So the discriminator is a DELTA on both, plus any event the
gate has appended. Sampling once cannot tell those apart, and the CPU delta
must be measured on the brain rather than on the supervisor, which sleeps
between polls either way.

Read-only. Nothing is restarted: the canary holds a settled brain, and a
restart during it discards the block.
"""
import collections
import glob
import json
import os
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def sh(cmd, timeout=60):
    try:
        done = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()
    except Exception as error:  # noqa: BLE001
        return "<error: %s>" % error


def brain_pid():
    for line in sh("ps -eo pid,args | grep '[w]1z4rd_brain_server'").splitlines():
        parts = line.split(None, 1)
        if parts and parts[0].isdigit():
            return parts[0]
    return None


def cpu_ticks(pid):
    """utime+stime for a pid, in clock ticks."""
    try:
        fields = open("/proc/%s/stat" % pid, encoding="utf-8").read().split()
        return int(fields[13]) + int(fields[14])
    except Exception:  # noqa: BLE001
        return None


pid = brain_pid()
out["brain_pid"] = pid

# ---- Sample processes and CPU twice, far enough apart to be meaningful ----
def snapshot():
    procs = {}
    for line in sh("ps -eo pid,etimes,rss,args --sort=-rss | head -20").splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) >= 4 and parts[0].isdigit():
            procs[parts[0]] = {"etimes": int(parts[1]), "rss_kb": int(parts[2]),
                               "cmd": parts[3][:110]}
    return {"unix": time.time(), "procs": procs, "cpu": cpu_ticks(pid)}


first = snapshot()
time.sleep(45)
second = snapshot()

span = second["unix"] - first["unix"]
ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
delta = None
if first["cpu"] is not None and second["cpu"] is not None:
    delta = (second["cpu"] - first["cpu"]) / float(ticks)
out["sample_seconds"] = round(span, 1)
out["brain_cpu_seconds_used"] = None if delta is None else round(delta, 2)
out["brain_cpu_percent"] = (None if delta is None
                            else round(100.0 * delta / span, 1))

# Processes that appeared or vanished between the samples: a canary running
# gate suites churns short-lived subprocesses.
before, after = set(first["procs"]), set(second["procs"])
out["processes_started"] = [
    {"pid": p, **second["procs"][p]} for p in sorted(after - before)]
out["processes_exited"] = [
    {"pid": p, **first["procs"][p]} for p in sorted(before - after)]

# ---- What has the ledger recorded recently? ------------------------------
ledger = os.path.join(R, "curriculum-health.jsonl")
recent = collections.Counter()
newest = []
if os.path.exists(ledger):
    for line in open(ledger, encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        when = float(row.get("updated_unix") or 0)
        age = out["now"] - when
        if age <= 7200:
            recent[str(row.get("kind") or "")] += 1
        newest.append((when, str(row.get("kind") or ""), row.get("phase")))
newest.sort()
out["ledger_last_2h"] = dict(recent.most_common())
out["ledger_newest"] = [
    {"age_h": round((out["now"] - w) / 3600.0, 2), "kind": k, "phase": p}
    for w, k, p in newest[-8:]]

# ---- The gate's own artifacts for this phase ----------------------------
arts = []
for path in glob.glob(os.path.join(R, "*.enterprise-gate.json")) + \
        glob.glob(os.path.join(R, "*canary*.json")):
    try:
        arts.append({
            "file": os.path.basename(path),
            "age_h": round((out["now"] - os.path.getmtime(path)) / 3600.0, 2),
        })
    except OSError:
        continue
arts.sort(key=lambda item: item["age_h"])
out["gate_artifacts"] = arts[:10]

# The forward driver's own progress file: during a canary it should be frozen,
# and its freeze is the expected shape rather than the fault.
progress = os.path.join(R, "go-systems.progress.json")
if os.path.exists(progress):
    out["forward_progress"] = {
        "age_s": round(out["now"] - os.path.getmtime(progress), 1),
    }
    try:
        out["forward_progress"].update(json.load(
            open(progress, encoding="utf-8")))
    except Exception as error:  # noqa: BLE001
        out["forward_progress"]["error"] = str(error)[:120]

status = os.path.join(R, "curriculum-supervisor.status.json")
if os.path.exists(status):
    out["status_age_s"] = round(out["now"] - os.path.getmtime(status), 1)
    try:
        out["status"] = json.load(open(status, encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        out["status"] = {"error": str(error)[:120]}

print("PROBE_JSON " + json.dumps(out, default=str))
PY
