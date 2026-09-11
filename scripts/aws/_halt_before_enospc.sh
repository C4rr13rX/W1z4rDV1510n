python3 - <<'PY'
"""Stop the replay at a durable boundary before the volume reaches zero.

Measured minutes before this ran: 44.13 GB free on a 1.0 TB volume, burning
108.66 GB/h, with the deferred-replay worker holding 33,432 rows still to train
at a sustained 1.652 rows/s. That is 5.6 hours of training against 0.34 hours
of disk. The interval cannot finish, so nothing is being given up by stopping
it -- and the alternative is known: the wrapper dies writing its 6-byte
`node.pid` with ENOSPC and systemd restarts it every 10 s, which is the 115x
crash loop that blocked all reclaim for five weeks.

The `--min-free-disk-gb 150` guard does not prevent this. It is enforced by
`disk_floor_breached` on the forward corpus-phase loop, and
`forward_remaining_rows` is 0 -- that loop has finished. The path that is
actually running, `run_deferred_replay_worker`, polls
`replay_memory_floor_breached` and nothing else. The guard is watching the
stage that ended while the stage that runs fills the disk.

Stops the unit first so systemd cannot resurrect the worker between the kill
and the measurement, then uses the repository's own stop script so the worker
lands on its published WAL-durable boundary rather than wherever SIGKILL finds
it. Records the durable row and the interval marker BEFORE and AFTER, because
`deferred-replay-active.json` carrying `state: training` is exactly the shape
that a later supervisor rolls back to `start_row`, and the difference between
"we stopped at 228,712" and "the next generation restarts at 201,344" is
27,368 rows of billed compute.
"""
import glob
import json
import os
import subprocess
import shutil
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def sh(cmd, timeout=180):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "rc": proc.returncode,
            "out": (proc.stdout + proc.stderr)[-2500:],
        }
    except Exception as exc:
        return {"rc": None, "out": f"{type(exc).__name__}: {exc}"}


def snapshot():
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
    try:
        with open(f"{R}/deferred-replay-active.json", "r", encoding="utf-8") as h:
            active = json.load(h)
    except Exception as exc:
        active = f"{type(exc).__name__}: {exc}"
    return {
        "row": best,
        "active_state": (
            active.get("state") if isinstance(active, dict) else active
        ),
        "active_interval_id": (
            active.get("interval_id") if isinstance(active, dict) else None
        ),
        "free_gb": round(shutil.disk_usage(R).free / 2**30, 2),
    }


out["before"] = snapshot()

out["unit_stop"] = sh(
    "sudo systemctl stop wizard-curriculum-supervisor.service 2>&1"
)
out["script_stop"] = sh(
    "cd /srv/wizard/project && sudo -u ec2-user "
    "bash scripts/aws/stop_programming_curriculum_service.sh 2>&1"
)
time.sleep(20)

census = {}
for name, pat in {
    "wrapper": "run_programming_curriculum_service.sh",
    "supervisor": "programming_curriculum_supervisor.py",
    "worker": "drive_corpora_brain",
    "brain": "w1z4rd_brain_server",
}.items():
    count = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if pat in cmd:
            count += 1
    census[name] = count
out["census_after_stop"] = census

out["after"] = snapshot()
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p NRestarts --no-pager 2>&1"
)["out"].strip()

# The burn must go to ~zero once the worker is gone. If it does not, something
# other than neuron eviction is writing and the attribution above is wrong.
free_a = shutil.disk_usage(R).free
time.sleep(90)
free_b = shutil.disk_usage(R).free
out["post_stop_burn_gb_per_hour"] = round(
    (free_a - free_b) / 2**30 / (90 / 3600.0), 2
)
out["free_gb_final"] = round(free_b / 2**30, 2)
out["df"] = sh("df -h /srv/wizard 2>&1")["out"]

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
