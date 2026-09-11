python3 - <<'PY'
"""Did the predicted disk halt actually happen, and in the predicted shape?

The prediction, made with ~21 minutes of headroom left: the interval reaches
the 150 GB floor long before its gate, settles at a durable boundary, and the
supervisor -- which is still the generation that predates the fix -- publishes
`resource_waiting` and sits there. A prediction that is not checked against the
host is just a story, and two confident causal stories were wrong in one day
earlier this week.

What distinguishes the predicted halt from the alternatives:

  * `resource_waiting` (or `disk_exhausted_unrecoverable` if the unit was
    restarted) beside free space just under 150 GB -- the halt;
  * `deferred_replay_training` with the row still advancing -- the burn eased
    and the prediction was wrong about the rate;
  * a dead or auto-restarting unit -- ENOSPC got there first, which would mean
    the floor is too low rather than unreachable.

Also re-measures the row twice, because liveness is the ROW DELTA and a census
during a cooperative yield reads `worker 0` on a perfectly healthy block.
"""
import glob
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def sh(cmd, timeout=120):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-4000:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def row():
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
                "durable_next_row": blob.get("durable_next_row"),
                "accepted_episodes": blob.get("accepted_episodes"),
                "age_s": round(age, 1),
            }
    return best


first_free = shutil.disk_usage("/srv/wizard").free
out["row_first"] = row()
out["free_gb_first"] = round(first_free / 2**30, 2)
time.sleep(120)
second_free = shutil.disk_usage("/srv/wizard").free
out["row_second"] = row()
out["free_gb_second"] = round(second_free / 2**30, 2)
out["burn_gb_per_hour"] = round(
    (first_free - second_free) / 2**30 / (120 / 3600.0), 2
)
out["headroom_gb"] = round(second_free / 2**30 - 150.0, 2)

out["status"] = sh(f"tail -c 900 {R}/curriculum-supervisor.status.json 2>&1")
out["status_age_s"] = round(
    time.time() - os.path.getmtime(f"{R}/curriculum-supervisor.status.json"), 1
)
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p NRestarts -p ExecMainStatus --no-pager 2>&1"
)
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
out["census"] = census
out["health_tail"] = sh(
    f"tail -c 1200 {R}/curriculum-health.jsonl 2>&1"
)
out["free"] = sh("free -g | head -2")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
