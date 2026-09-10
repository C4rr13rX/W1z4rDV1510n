python3 - <<'PY'
"""Stop the curriculum cleanly BEFORE the volume fills, and prove the burn ends.

Measured twice this session: 206 GB/h and 257 GB/h against ~508 GB free, so the
volume reaches ENOSPC in roughly two hours. The last time it did, the wrapper
died writing its 6-byte `node.pid`, systemd restarted it 115 times, and the
census read as a finished stage. The supervisor's own `--min-free-disk-gb 8`
guard is 112 seconds of warning at this rate AND requires three consecutive
breaches at a durable boundary, so it cannot win that race.

Compaction of the 562 GB container needs the brain quiescent regardless, so
stopping now costs no downtime that the repair would not have cost anyway -- it
only moves the stop before the crash instead of after it.

`systemctl stop` is used rather than killing processes directly: the unit's
`ExecStop` is the sanctioned stop script, and the unit would otherwise restart
the supervisor 10 seconds later. `KillMode=process` deliberately leaves the
brain node alive, which is what we want here -- with no worker POSTing rows it
appends nothing, and leaving it up avoids a full re-hydration.

The burn is then re-sampled. A stop that is announced but not measured is the
same class of error as a supervisor that reports `active` while admitting
nothing.
"""
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
UNIT = "wizard-curriculum-supervisor"
out = {}


def sh(*args, timeout=180):
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "rc": proc.returncode,
            "out": (proc.stdout or "").strip()[-2000:],
            "err": (proc.stderr or "").strip()[-2000:],
        }
    except Exception as exc:
        return {"rc": None, "error": f"{type(exc).__name__}: {exc}"}


def census(needle):
    hits = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = handle.read().decode("utf-8", "replace").replace("\0", " ")
        except Exception:
            continue
        if needle in cmd:
            hits.append(int(pid))
    return hits


def snapshot(tag):
    free = shutil.disk_usage(R).free
    try:
        size = os.stat(os.path.join(R, "brain/brain.wbrain")).st_size
    except Exception:
        size = None
    out[f"{tag}_free_gb"] = round(free / 1e9, 3)
    out[f"{tag}_wbrain_gb"] = round(size / 1e9, 3) if size else None
    return free, size


# ---- Durable position before the stop, so the cost of stopping is recorded
# rather than estimated. A restart discards an interval whose state is
# `training`, so this is the progress being paid for the volume.
def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


active = load(os.path.join(R, "deferred-replay-active.json"))
out["active_state_before"] = active.get("state") if isinstance(active, dict) else None
interval = active.get("interval") if isinstance(active, dict) else None
out["active_interval_id"] = (
    interval.get("interval_id") if isinstance(interval, dict) else None
)
progress = load(os.path.join(R, "deferred-replay-909de5e9d4936130.progress.json"))
out["durable_next_row_before"] = progress.get("durable_next_row")
out["accepted_episodes_before"] = progress.get("accepted_episodes")

free0, _ = snapshot("before")
t0 = time.time()

out["systemctl_stop"] = sh("sudo", "systemctl", "stop", UNIT, timeout=180)
out["unit_state"] = sh(
    "systemctl", "show", "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts", UNIT
)

# Give the ExecStop script and any orphan worker time to leave.
deadline = time.time() + 90
while time.time() < deadline:
    if not census("run_programming_curriculum_service.sh") and not census(
        "tools.training_standard.drive_corpora_brain"
    ) and not census("curriculum_supervisor"):
        break
    time.sleep(3)

out["wrapper_pids"] = census("run_programming_curriculum_service.sh")
out["worker_pids"] = census("tools.training_standard.drive_corpora_brain")
out["supervisor_pids"] = census("curriculum_supervisor")
# The brain node is intentionally left running (KillMode=process).
out["node_pids"] = census("w1z4rdv1510n-node") or census("brain_server")

# ---- Re-measure the burn with the producer gone. This is the whole point of
# the action; a stop that is not measured is not a stop.
settle_start = time.time()
free1, size1 = snapshot("after_stop")
time.sleep(90)
free2, size2 = snapshot("verify")
elapsed = time.time() - settle_start - 0  # includes the two snapshots' cost
window = 90.0
delta = free1 - free2
out["post_stop_window_seconds"] = round(time.time() - settle_start, 1)
out["post_stop_consumed_gb"] = round(delta / 1e9, 4)
if delta > 0:
    out["post_stop_burn_gb_per_hour"] = round(delta / window * 3600.0 / 1e9, 2)
else:
    out["post_stop_burn_gb_per_hour"] = 0.0
    out["post_stop_reclaimed_gb"] = round(-delta / 1e9, 4)
out["wbrain_grew_after_stop_gb"] = (
    round((size2 - size1) / 1e9, 4) if size1 and size2 else None
)
out["total_stop_seconds"] = round(time.time() - t0, 1)

print("PROBE_JSON " + json.dumps(out, sort_keys=True))
PY
