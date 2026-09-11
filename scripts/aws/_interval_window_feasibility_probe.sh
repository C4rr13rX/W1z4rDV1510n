python3 - <<'PY'
"""Can the RUNNING interval finish before the volume reaches the halt floor?

CLAUDE.md names this the binding constraint on the whole objective: durable
progress survives a memory yield within one supervisor generation, but a
supervisor RESTART rolls a `state: training` marker back to its interval start.
So an interval whose training ETA exceeds the disk window can never admit, no
matter how healthy every liveness signal looks.

Both terms are measured over the SAME window, because both are duty cycles:
 * rows/s from the freshest file that actually exposes a row, and
 * GB/h from `df`, never from summing file sizes (XFS reflink shares extents,
   so `du` reports 2.48 TB inside 1.0 TB).

The interval's target is read from durable state (`deferred-replay-active.json`
and the `interval_id`, which encodes `phase:start:end`), never from
`curriculum-supervisor.status.json['block_target_row']` -- that key is written
only by the forward stage, and forward is complete here.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

R = "/srv/wizard/runtime/programming-integrated-20260713"
ENDPOINT = "http://127.0.0.1:18095/brain/stats"
WINDOW = 180.0
out = {"now": time.time(), "runtime": R}


def sh(cmd, timeout=90):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout + p.stderr).strip()[-1500:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def stats():
    try:
        with urllib.request.urlopen(ENDPOINT, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def freshest_row():
    """The freshest writer that actually exposes a row, plus its lag.

    Selecting on mtime alone picks `curriculum-supervisor.status.json`, which
    during a replay carries `resume_row`/`end_row` and never `durable_next_row`
    -- so the probe waits its whole bound for a row that file cannot contain.
    """
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


# ---- durable interval identity -------------------------------------------
active = read_json(os.path.join(R, "deferred-replay-active.json"))
out["active_state"] = active.get("state")
out["active_created_unix"] = active.get("created_unix")
interval = active.get("interval") or {}
interval_id = active.get("interval_id") or interval.get("interval_id") or ""
out["interval_id"] = interval_id
match = re.match(r"^(.*):(\d+):(\d+)$", str(interval_id))
start = end = None
if match:
    start, end = int(match.group(2)), int(match.group(3))
out["interval_start"], out["interval_end"] = start, end
if start is not None:
    out["interval_rows"] = end - start

# ---- unit / census --------------------------------------------------------
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service "
    "-p ActiveState -p SubState -p NRestarts -p ExecMainStartTimestamp "
    "--no-pager"
)
census = {"wrapper": 0, "supervisor": 0, "worker": 0}
patterns = {
    "wrapper": "run_programming_curriculum_service.sh",
    "supervisor": "programming_curriculum_supervisor.py",
    "worker": "drive_corpora_brain",
}
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    for name, pat in patterns.items():
        if pat in cmd:
            census[name] += 1
out["census"] = census

# Does the RUNNING supervisor source carry the replay-stage disk guard? A
# guard attached to the finished forward loop cannot protect replay.
src = "/srv/wizard/project/scripts/programming_curriculum_supervisor.py"
out["replay_disk_guard_in_source"] = sh(
    f"grep -c 'replay_disk_floor_breached' {src}")
out["supervisor_argv"] = sh(
    "tr '\\0' ' ' < /proc/$(pgrep -f programming_curriculum_supervisor.py "
    "| head -1)/cmdline")

# ---- paired window: disk burn AND row rate over the same interval ---------
free_a = shutil.disk_usage(R).free
stats_a = stats()
row_a = freshest_row()
mem_a = sh("awk '/MemAvailable/{print $2}' /proc/meminfo")
t0 = time.time()
time.sleep(WINDOW)
elapsed = time.time() - t0
free_b = shutil.disk_usage(R).free
stats_b = stats()
row_b = freshest_row()
mem_b = sh("awk '/MemAvailable/{print $2}' /proc/meminfo")

out["elapsed_seconds"] = round(elapsed, 1)
out["free_gb_before"] = round(free_a / 2**30, 2)
out["free_gb_after"] = round(free_b / 2**30, 2)
burn_gb_h = (free_a - free_b) / 2**30 / (elapsed / 3600.0)
out["burn_gb_per_hour"] = round(burn_gb_h, 2)
out["mem_available_gb_before"] = round(int(mem_a or 0) / 2**20, 2)
out["mem_available_gb_after"] = round(int(mem_b or 0) / 2**20, 2)

if row_a and row_b:
    out["row_file"] = os.path.basename(row_b[1])
    out["row_before"], out["row_after"] = row_a[2], row_b[2]
    out["row_source_lag_seconds"] = round(time.time() - row_b[0], 1)
    delta = row_b[2] - row_a[2]
    # A counter that went BACKWARDS is a worker restart, not a negative rate.
    out["counter_reset"] = delta < 0
    out["rows_per_second"] = None if delta < 0 else round(delta / elapsed, 4)
    out["accepted_before"] = row_a[3].get("accepted_episodes")
    out["accepted_after"] = row_b[3].get("accepted_episodes")
else:
    out["no_row_writer"] = True

for label, payload in (("before", stats_a), ("after", stats_b)):
    out[f"stats_{label}"] = {
        k: payload.get(k) for k in
        ("total_neurons", "evicted_neurons", "resident_terminals",
         "page_outs", "clean_skips")
    } if "error" not in payload else payload

# ---- the feasibility question itself -------------------------------------
floor_gb = 150.0
match_floor = re.search(r"--min-free-disk-gb\s+(\d+(?:\.\d+)?)",
                        out.get("supervisor_argv") or "")
if match_floor:
    floor_gb = float(match_floor.group(1))
out["disk_floor_gb"] = floor_gb
headroom = out["free_gb_after"] - floor_gb
out["disk_headroom_gb"] = round(headroom, 2)
out["disk_window_hours"] = (
    round(headroom / burn_gb_h, 2) if burn_gb_h > 0.01 else None)

rate = out.get("rows_per_second")
if rate and end is not None and row_b:
    remaining = end - row_b[2]
    out["rows_remaining"] = remaining
    out["interval_eta_hours"] = round(remaining / rate / 3600.0, 2)
    if out["disk_window_hours"]:
        out["fits_in_window"] = (
            out["interval_eta_hours"] <= out["disk_window_hours"])
        out["shortfall_factor"] = round(
            out["interval_eta_hours"] / out["disk_window_hours"], 1)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
