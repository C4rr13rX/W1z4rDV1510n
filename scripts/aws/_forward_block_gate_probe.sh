python3 - <<'PY'
"""Decide what actually stands between this forward block and the next admission.

The watchdog fired on a 103.7 h drought while the freshest writer was
`go-systems.progress.json` advancing at 12 rows/s toward 131072. The
forward-stage lesson says that alarm is expected -- a forward stage harvests
rows and admission belongs to deferred replay -- so the useful question is
NOT "is it stalled" but "how many more forward blocks stand between now and
the replay stage that can admit anything".

Three separable questions, each with a different remedy:

  1. Does THIS block converge?  Sampled adaptively across a commit period,
     because the row moves once per committed batch and any window shorter
     than the commit period reads 0 rows/s on a healthy block.
  2. Does memory let it?  `available_gb` was 2.98 against an 11.6 GB brain
     with no swap, and the last yield fired at exactly 3.0 GB. If the drain
     rate crosses the floor before the block lands, waiting is hopeful, not
     safe -- but a yield is only cheap while `durable_next_row` tracks
     `ram_next_row`, so measure the gap, not just the headroom.
  3. After this block, how much forward work remains before the stage flips
     to deferred replay?  `forward_remaining_rows` was 217409 against a
     131072 block target, so this block is NOT the last one and the drought
     will keep growing by design. That number, not the heartbeat, is what
     predicts the next admission.

Read-only. Nothing is restarted: a restart during `state: training` discards
the whole interval (78,168 of 131,072 rows, measured), so guessing wrong here
costs the block.
"""
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


def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def meminfo():
    fields = {}
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            key, _, rest = line.partition(":")
            fields[key] = int(rest.split()[0]) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    return fields


# ---------------------------------------------------------------- stage
status = load(os.path.join(R, "curriculum-supervisor.status.json")) or {}
out["status"] = status
out["status_age_seconds"] = (
    time.time() - status["updated_unix"] if status.get("updated_unix") else None)

# ------------------------------------------------- freshest progress writer
best = None
for path in glob.glob(os.path.join(R, "*.progress.json")):
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        continue
    if best is None or age < best[0]:
        best = (age, path)
out["freshest_progress"] = {
    "file": os.path.basename(best[1]) if best else None,
    "age_seconds": round(best[0], 1) if best else None,
}

# ------------------------------------------------------- adaptive sampling
#
# The row moves once per committed batch. A fixed short window reads 0 rows/s
# on a perfectly healthy block (measured: 32 rows per commit at 0.355 rows/s
# is one commit every ~90 s, caught by a 6 s sample about 7% of the time).
# Sample until the row MOVES or the budget expires, and report which.
SAMPLE_BUDGET = 150.0


def read_row(path):
    data = load(path) or {}
    for key in ("row", "durable_next_row", "next_row", "rows_processed"):
        if isinstance(data.get(key), int):
            return data[key], data
    return None, data


sample = {"budget_seconds": SAMPLE_BUDGET}
if best:
    path = best[1]
    start_row, start_data = read_row(path)
    start_mem = meminfo().get("MemAvailable")
    began = time.time()
    moved_at = None
    end_row = start_row
    while time.time() - began < SAMPLE_BUDGET:
        time.sleep(3.0)
        end_row, end_data = read_row(path)
        if start_row is not None and end_row is not None and end_row != start_row:
            moved_at = time.time()
            break
    elapsed = (moved_at or time.time()) - began
    end_mem = meminfo().get("MemAvailable")
    sample.update({
        "file": os.path.basename(path),
        "start_row": start_row,
        "end_row": end_row,
        "elapsed_seconds": round(elapsed, 1),
        "row_moved": moved_at is not None,
        "rows_per_second": (
            round((end_row - start_row) / elapsed, 3)
            if moved_at and start_row is not None and elapsed > 0 else 0.0),
        "mem_available_start_gb": round(start_mem, 2) if start_mem else None,
        "mem_available_end_gb": round(end_mem, 2) if end_mem else None,
        "mem_drain_gb_per_hour": (
            round((start_mem - end_mem) * 3600.0 / elapsed, 3)
            if start_mem and end_mem and elapsed > 0 else None),
    })
out["sample"] = sample

# --------------------------------------------------------- durable vs ram
# A yield is cheap only while the durable row tracks the in-RAM row: the
# worker resumes from `durable_next_row`, so the gap IS the rollback cost.
out["rollback_exposure_rows"] = (
    (status.get("ram_next_row") or 0) - (status.get("durable_next_row") or 0))

# ------------------------------------------------------- remaining forward
# The number that predicts the next admission is not the heartbeat: it is how
# many blocks of forward harvest remain before the replay stage runs at all.
ledger = os.path.join(R, "curriculum-health.jsonl")
recent = sh("tail -n 4000 %s 2>/dev/null" % ledger, timeout=90)
phases_seen = []
harvested = []
for line in recent.splitlines():
    try:
        event = json.loads(line)
    except Exception:  # noqa: BLE001
        continue
    kind = event.get("kind") or event.get("event")
    if kind == "phase_forward_harvested":
        harvested.append({"phase": event.get("phase"),
                          "unix": event.get("unix") or event.get("timestamp")})
    if kind in ("deferred_replay_admitted", "quarantine_retest_admitted"):
        phases_seen.append({"kind": kind, "phase": event.get("phase"),
                            "unix": event.get("unix") or event.get("timestamp")})
out["forward_harvested_events"] = harvested[-12:]
out["admission_events"] = phases_seen[-6:]

progress_files = {}
for path in glob.glob(os.path.join(R, "*.progress.json")):
    data = load(path) or {}
    progress_files[os.path.basename(path)] = {
        "age_seconds": round(time.time() - os.path.getmtime(path), 1),
        "row": data.get("row") or data.get("durable_next_row"),
        "target": data.get("block_target_row") or data.get("target_row"),
        "phase": data.get("phase"),
    }
out["progress_files"] = progress_files

# ----------------------------------------------------------- named failure
# Timestamp the named failure against process start before re-debugging it:
# a `last_failure` can be entirely pre-fix (measured: repaired and redeployed
# 5.8 h before the alarm that named it).
worker = status.get("worker_pid")
if worker:
    out["worker_start"] = sh("ps -o lstart= -p %s" % worker)
    out["worker_elapsed"] = sh("ps -o etimes= -p %s" % worker).strip()
out["supervisor_active_since"] = sh(
    "systemctl show wizard-curriculum-supervisor -p ActiveEnterTimestamp "
    "--value 2>/dev/null")

mem = meminfo()
out["memory"] = {
    "available_gb": round(mem.get("MemAvailable", 0.0), 2),
    "total_gb": round(mem.get("MemTotal", 0.0), 2),
    "swap_total_gb": round(mem.get("SwapTotal", 0.0), 2),
}

print("PROBE_JSON " + json.dumps(out, sort_keys=True))
PY
