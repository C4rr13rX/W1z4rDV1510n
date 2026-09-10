python3 - <<'PY'
"""Decide whether the live forward block converges before memory stops it.

The watchdog fired on a 101.2 h admission drought while the STATUS heartbeat
was advancing at 16 rows/s -- the inverted case from the heartbeat lesson,
where the freshest writer is the forward worker and the replay progress file
is a leftover. A drought beside a live heartbeat is not automatically a
fault, so this probe refuses to repair anything until it has separated three
questions that have different answers and opposite remedies:

  1. Is the block advancing at a rate that reaches 131072 at all?  Sampled,
     not read once: a live curriculum trains underneath any measurement.
  2. Will it get there before memory forces a yield?  `available_gb` was 3.62
     against a 10.96 GB brain with no swap. A yield mid-interval is what the
     resource-yield lesson recorded as being scored as a semantic failure, so
     the drain rate decides whether waiting is safe or merely hopeful.
  3. Is the 101.2 h drought explained by the named `last_failure`, or is that
     one row of a much larger population?  The named-failure lesson measured a
     `last_failure` that was 6.6% of 288; acting on it alone repairs the tail.

Everything here is read-only. Nothing is restarted: a restart during
`state: training` discards the whole interval (78,168 of 131,072 rows,
measured), so the cost of guessing wrong is the entire block.
"""
import collections
import glob
import json
import os
import re
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time(), "runtime": R}


def sh(cmd, timeout=60):
    try:
        done = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()
    except Exception as error:  # noqa: BLE001
        return "<error: %s>" % error


def meminfo():
    fields = {}
    for line in open("/proc/meminfo", encoding="utf-8"):
        key, _, rest = line.partition(":")
        fields[key] = int(rest.split()[0]) / (1024.0 * 1024.0)
    return fields


def rss_gb(pid):
    try:
        for line in open("/proc/%s/status" % pid, encoding="utf-8"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None
    return None


# ---- 0. Who is running, and how old is each process? ----------------------
ps = sh("ps -eo pid,etimes,rss,comm,args --sort=-rss | head -25")
out["ps_top"] = ps.splitlines()
brain_pid = sh("pgrep -f 'brain[-_]server|wizard-brain' | head -3").split()
out["brain_pids"] = brain_pid
# The pgrep lesson: the first PID can be the supervisor wearing the brain's
# name, which reads as a 30 MB brain and looks like the not-hydrating fault.
out["pid_identity"] = {
    pid: sh("tr '\\0' ' ' < /proc/%s/cmdline 2>/dev/null | cut -c1-160" % pid)
    for pid in brain_pid
}

# ---- 1 & 2. Sample the forward row and memory together --------------------
status_path = os.path.join(R, "curriculum-supervisor.status.json")
samples = []
for index in range(7):
    if index:
        time.sleep(20)
    entry = {"t": round(time.time() - out["now"], 1)}
    try:
        data = json.load(open(status_path, encoding="utf-8"))
        entry["row"] = data.get("durable_next_row")
        entry["ram_row"] = data.get("ram_next_row")
        entry["state"] = data.get("state")
        entry["phase"] = data.get("phase")
        entry["target"] = data.get("block_target_row")
        entry["batch"] = data.get("batch_size")
        entry["worker_pid"] = data.get("worker_pid")
        entry["age_s"] = round(time.time() - os.path.getmtime(status_path), 1)
    except Exception as error:  # noqa: BLE001
        entry["err"] = str(error)[:100]
    mem = meminfo()
    entry["avail_gb"] = round(mem.get("MemAvailable", 0), 2)
    entry["swap_free_gb"] = round(mem.get("SwapFree", 0), 2)
    for pid in brain_pid[:2]:
        entry["rss_%s" % pid] = rss_gb(pid)
    samples.append(entry)
out["samples"] = samples

rows = [(s["t"], s["row"]) for s in samples if s.get("row") is not None]
if len(rows) >= 2 and rows[-1][1] is not None and rows[0][1] is not None:
    span = rows[-1][0] - rows[0][0]
    delta = rows[-1][1] - rows[0][1]
    rate = delta / span if span else 0.0
    target = samples[-1].get("target") or 131072
    out["forward"] = {
        "rows_per_second": round(rate, 2),
        "row": rows[-1][1],
        "target": target,
        "remaining_rows": target - rows[-1][1],
        "eta_hours": round((target - rows[-1][1]) / rate / 3600.0, 2)
        if rate > 0 else None,
    }

mems = [(s["t"], s["avail_gb"]) for s in samples if s.get("avail_gb")]
if len(mems) >= 2:
    span = mems[-1][0] - mems[0][0]
    drop = mems[0][1] - mems[-1][1]
    drain = drop / span if span else 0.0
    out["memory"] = {
        "available_gb": mems[-1][1],
        "drain_gb_per_hour": round(drain * 3600.0, 3),
        # A yield fires well before zero; 1.0 GB is the conservative floor the
        # resource-yield events have historically tripped near.
        "hours_to_1gb": round((mems[-1][1] - 1.0) / (drain * 3600.0), 2)
        if drain > 0 else None,
    }

# ---- 3. Bucket the failure population, not the named row ------------------
# `append_health_event` writes curriculum-health.jsonl. An earlier revision of
# this probe read "curriculum-admissions.jsonl", which does not exist, and
# reported zero failures of every kind -- a uniformly empty table is the
# vacuous-zero shape, distinct from one empty bucket. The kind_counts line
# below exists to prove the pattern CAN be non-zero before any absence here is
# believed. Events also carry `updated_unix`, not `unix`.
ledger = os.path.join(R, "curriculum-health.jsonl")
buckets = collections.Counter()
kinds_seen = collections.Counter()
recent = []
first_seen = {}
if os.path.exists(ledger):
    for line in open(ledger, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        event = str(row.get("kind") or row.get("event") or "")
        kinds_seen[event] += 1
        if event != "deferred_replay_failed":
            continue
        reason = str(row.get("reason") or row.get("detail") or "")
        if "exited" in reason or "signal" in reason or "stderr" in reason:
            bucket = "worker_exit_or_signal"
        elif "yield" in reason or "memory" in reason or "resource" in reason:
            bucket = "resource_yield"
        elif "timeout" in reason or "timed out" in reason:
            bucket = "timeout"
        elif "gate" in reason or "semantic" in reason or "canary" in reason:
            bucket = "semantic_gate"
        elif reason:
            bucket = "other:" + re.sub(r"[^a-z ]", "", reason.lower())[:40]
        else:
            bucket = "no_reason_recorded"
        buckets[bucket] += 1
        first_seen.setdefault(bucket, row.get("updated_unix") or row.get("unix"))
        recent.append((row.get("updated_unix") or row.get("unix") or 0, bucket, reason[:200]))
recent.sort()
out["ledger_kind_counts"] = dict(kinds_seen.most_common(30))
out["failure_buckets"] = dict(buckets.most_common(15))
out["failure_bucket_first_seen"] = first_seen
out["failure_recent"] = [
    {"age_h": round((out["now"] - t) / 3600.0, 1), "bucket": b, "reason": r}
    for t, b, r in recent[-12:]
]

# ---- The named stderr log: is it even from this generation of the process?
match = re.search(r"stderr=(\S+)", sh(
    "grep -o 'stderr=[^\"]*' %s 2>/dev/null | tail -1"
    % os.path.join(R, "curriculum-admissions.jsonl")))
stderr_path = match.group(1) if match else None
out["last_failure_stderr_path"] = stderr_path
if stderr_path and os.path.exists(stderr_path):
    out["last_failure_stderr_age_h"] = round(
        (out["now"] - os.path.getmtime(stderr_path)) / 3600.0, 1)
    out["last_failure_stderr_tail"] = sh("tail -c 3000 %s" % stderr_path).splitlines()[-25:]
else:
    out["last_failure_stderr_missing"] = True

# ---- Admission truth from the ledger, not the announcement event ----------
resolved = []
if os.path.exists(ledger):
    for line in open(ledger, encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if str(row.get("kind") or row.get("event") or "") in (
                "deferred_replay_admitted", "quarantine_retest_admitted",
                "protected_hebbian_cadence_combined_admission",
                "phase_forward_harvested", "fully_deferred_block_advanced"):
            resolved.append((row.get("updated_unix") or row.get("unix") or 0, row.get("kind")))
resolved.sort()
out["recent_resolutions"] = [
    {"age_h": round((out["now"] - t) / 3600.0, 1), "event": e}
    for t, e in resolved[-10:]
]

out["service"] = sh(
    "systemctl show wizard-curriculum-supervisor -p ActiveState -p SubState "
    "-p NRestarts -p ExecMainStatus -p ExecMainStartTimestamp --no-pager"
).splitlines()

print("PROBE_JSON " + json.dumps(out, default=str))
PY
