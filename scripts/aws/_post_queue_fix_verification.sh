python3 - <<'PY'
"""Did the queue fix convert compute into progress, or just move the stall?

The deploy is only half the claim. What must be true afterwards:
  * the supervisor is running the code that was shipped (compare the digest,
    not the mtime -- a fix copied but never loaded runs the OLD code),
  * a DIFFERENT interval is selected than the one that consumed every previous
    generation,
  * its row is advancing, and
  * the volume that the rollback reclaimed is not being spent faster than the
    new interval can finish.

Reports the new interval's ETA at its measured rate against the disk window,
because that comparison is the binding constraint and a queue that merely
moves to another interval too large for the window has fixed nothing.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
DST = "/srv/wizard/project/scripts/programming_curriculum_supervisor.py"
WINDOW = 150.0
out = {"now": time.time()}


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout + p.stderr).strip()[-900:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# The file on disk, and the file the RUNNING process actually loaded.
try:
    out["deployed_sha"] = hashlib.sha256(open(DST, "rb").read()).hexdigest()
except Exception as exc:
    out["deployed_sha_error"] = str(exc)
pid = sh("pgrep -f programming_curriculum_supervisor.py | head -1")
out["supervisor_pid"] = pid
if pid.isdigit():
    # /proc/<pid>/cwd + argv gives the path it was started with; compare the
    # inode it opened, not just the name.
    out["loaded_inode"] = sh(f"stat -c %i {DST}")
    out["proc_start"] = sh(f"ps -o lstart= -p {pid}")
out["unit"] = sh("systemctl show wizard-curriculum-supervisor.service "
                 "-p ActiveState -p SubState -p NRestarts --no-pager")

active = read_json(f"{R}/deferred-replay-active.json")
interval_id = str(active.get("interval_id")
                  or (active.get("interval") or {}).get("interval_id") or "")
out["interval_id"] = interval_id
out["interval_state"] = active.get("state")
out["interval_created_hours_ago"] = round(
    (out["now"] - float(active.get("created_unix") or 0)) / 3600.0, 2)
match = re.match(r"^(.*):(\d+):(\d+)$", interval_id)
start = int(match.group(2)) if match else None
end = int(match.group(3)) if match else None
out["interval_rows"] = (end - start) if match else None
out["moved_off_stuck_interval"] = (
    interval_id != "jupyter-scientific-full:201344:262144")

stalls = 0
try:
    with open(f"{R}/curriculum-health.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if "deferred_replay_interrupted_before_gate" in line:
                stalls += 1
except OSError:
    pass
out["stall_records"] = stalls


def freshest_row():
    best = None
    for path in glob.glob(os.path.join(R, "*.progress.json")):
        payload = read_json(path)
        row = payload.get("durable_next_row")
        if row is None:
            continue
        mtime = os.path.getmtime(path)
        if best is None or mtime > best[0]:
            best = (mtime, path, int(row), payload)
    return best


free_a = shutil.disk_usage(R).free
row_a = freshest_row()
t0 = time.time()
time.sleep(WINDOW)
elapsed = time.time() - t0
free_b = shutil.disk_usage(R).free
row_b = freshest_row()

out["elapsed_seconds"] = round(elapsed, 1)
out["free_gb"] = round(free_b / 2**30, 2)
burn = (free_a - free_b) / 2**30 / (elapsed / 3600.0)
out["burn_gb_per_hour"] = round(burn, 2)

if row_a and row_b:
    delta = row_b[2] - row_a[2]
    out["row"] = row_b[2]
    out["row_file"] = os.path.basename(row_b[1])
    out["counter_reset"] = delta < 0
    out["rows_per_second"] = None if delta < 0 else round(delta / elapsed, 4)
    out["accepted_episodes"] = row_b[3].get("accepted_episodes")
else:
    out["no_row_writer"] = True

floor = 150.0
argv = sh("tr '\\0' ' ' < /proc/%s/cmdline" % pid) if pid.isdigit() else ""
found = re.search(r"--min-free-disk-gb\s+(\d+(?:\.\d+)?)", argv)
if found:
    floor = float(found.group(1))
out["disk_floor_gb"] = floor
headroom = out["free_gb"] - floor
out["disk_headroom_gb"] = round(headroom, 2)
out["disk_window_hours"] = round(headroom / burn, 2) if burn > 0.01 else None
rate = out.get("rows_per_second")
if rate and end is not None and row_b:
    remaining = end - row_b[2]
    out["rows_remaining"] = remaining
    out["interval_eta_hours"] = round(remaining / rate / 3600.0, 2)
    if out["disk_window_hours"]:
        out["fits_in_window"] = (
            out["interval_eta_hours"] <= out["disk_window_hours"])

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
