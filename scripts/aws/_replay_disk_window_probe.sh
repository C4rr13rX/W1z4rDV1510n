python3 - <<'PY'
"""Why is a 150 GB floor letting a replay train at 96 GB free?

The `quarantine_ready` payload says forward harvesting is done, replay is
active on `jupyter-scientific-full:201344:262144` at row 227,072 of 262,144,
and 4.0 rows/s -- all healthy. It also says `disk.free_gb` is 96.39. The
wrapper in the repository passes `--min-free-disk-gb 150`, and
`disk_floor_breached` stops at the first durable boundary below that floor.
Both cannot be true of the same process, so one of these holds:

  * the RUNNING supervisor predates the 150 GB floor and carries the old 8 GB
    default -- the deploy-is-not-load trap, and the halt is still ~88 GB away;
  * the floor is deployed and the guard is not on the deferred-replay path --
    in which case a replay can drive the volume to ENOSPC with the guard
    watching a path that is no longer running;
  * free space is rising (a recycle rolled the container back), and 96 GB is a
    trough in a sawtooth rather than a slide toward zero.

Only the running process's own cmdline settles the first. Only two samples of
`df` settle the third. Both are cheap; a story about which one is happening is
not evidence.

Also measures the interval ETA against the disk window, because the binding
constraint on this whole objective is whether an interval can FIT: 35,072 rows
remaining at the sustained (not instantaneous) rate, against (free - floor)
divided by the measured burn.
"""
import glob
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
SRC = "/srv/wizard/repo"
out = {"now": time.time()}


def sh(cmd, timeout=120):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return (proc.stdout + proc.stderr)[-4000:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def row():
    """Freshest progress file that actually exposes a row (never mtime alone)."""
    best = None
    for path in glob.glob(os.path.join(R, "deferred-replay-*.progress.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                blob = json.load(handle)
        except Exception:
            continue
        if blob.get("durable_next_row") is None:
            continue
        age = time.time() - os.path.getmtime(path)
        if best is None or age < best["age_s"]:
            best = {
                "file": os.path.basename(path),
                "durable_next_row": blob.get("durable_next_row"),
                "accepted_episodes": blob.get("accepted_episodes"),
                "age_s": round(age, 1),
            }
    return best


# --- the authoritative floor: the running process's own argv -----------------
cmdlines = {}
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    for name, pat in {
        "wrapper": "run_programming_curriculum_service.sh",
        "supervisor": "programming_curriculum_supervisor.py",
        "worker": "drive_corpora_brain",
        "brain": "w1z4rd_brain_server",
    }.items():
        if pat in cmd:
            cmdlines.setdefault(name, []).append({"pid": pid, "cmd": cmd[:600]})
out["cmdlines"] = cmdlines

# --- is the fix actually LOADED, not merely copied? --------------------------
sup = f"{SRC}/scripts/programming_curriculum_supervisor.py"
watch = f"{SRC}/scripts/aws/watch_programming_brain.py"
wrapper = f"{SRC}/scripts/aws/run_programming_curriculum_service.sh"
out["deployed_markers"] = {
    "supervisor_has_disk_exhausted": sh(
        f"grep -c disk_exhausted_unrecoverable {sup} 2>&1"
    ).strip(),
    "supervisor_has_reclaim_for_floor": sh(
        f"grep -c 'def reclaim_disk_for_floor' {sup} 2>&1"
    ).strip(),
    "wrapper_floor": sh(f"grep -n 'min-free-disk-gb' {wrapper} 2>&1").strip(),
    "watch_alarm_floor": sh(
        f"grep -n 'DISK_ALARM_FLOOR_GB\\|disk_exhausted' {watch} 2>&1"
    ).strip()[:1200],
    "supervisor_mtime": sh(f"stat -c '%y %i %s' {sup} 2>&1").strip(),
    "wrapper_mtime": sh(f"stat -c '%y %i %s' {wrapper} 2>&1").strip(),
    "head_commit": sh(f"cd {SRC} && git log --oneline -3 2>&1").strip(),
}

# --- two samples: burn and sustained row rate --------------------------------
free_first = shutil.disk_usage(R).free
row_first = row()
t0 = time.time()
time.sleep(150)
free_second = shutil.disk_usage(R).free
row_second = row()
elapsed = time.time() - t0

out["free_gb_first"] = round(free_first / 2**30, 2)
out["free_gb_second"] = round(free_second / 2**30, 2)
out["sample_seconds"] = round(elapsed, 1)
out["burn_gb_per_hour"] = round(
    (free_first - free_second) / 2**30 / (elapsed / 3600.0), 2
)
out["row_first"] = row_first
out["row_second"] = row_second
if row_first and row_second:
    delta = row_second["durable_next_row"] - row_first["durable_next_row"]
    out["row_delta"] = delta
    out["rows_per_second_sustained"] = (
        round(delta / elapsed, 3) if delta >= 0 else None
    )
    out["counter_reset"] = delta < 0

# --- interval identity and ETA vs the disk window ----------------------------
active = f"{R}/deferred-replay-active.json"
try:
    with open(active, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    out["active_interval"] = {
        "interval_id": blob.get("interval_id"),
        "state": blob.get("state"),
        "created_unix": blob.get("created_unix"),
        "created_age_hours": (
            round((time.time() - blob["created_unix"]) / 3600.0, 2)
            if blob.get("created_unix") else None
        ),
        "keys": sorted(blob.keys())[:30],
    }
except Exception as exc:
    out["active_interval"] = f"{type(exc).__name__}: {exc}"

out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p NRestarts -p ExecMainStatus --no-pager 2>&1"
).strip()
out["status_tail"] = sh(f"tail -c 900 {R}/curriculum-supervisor.status.json 2>&1")
out["health_tail"] = sh(f"tail -c 1500 {R}/curriculum-health.jsonl 2>&1")
out["df"] = sh("df -h /srv/wizard 2>&1")
out["free"] = sh("free -g | head -2")
out["biggest"] = sh(
    "du -xh --max-depth=2 /srv/wizard 2>/dev/null | sort -rh | head -14"
)
out["wbrain"] = sh(
    "ls -l --block-size=1G /srv/wizard/runtime/*/brain/*.wbrain 2>&1 | head -8"
)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
