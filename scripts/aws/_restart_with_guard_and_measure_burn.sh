python3 - <<'PY'
"""Restart the curriculum and measure whether the two fixes actually bite.

Three things must happen, and each has a number attached that makes it
checkable rather than narratable:

  1. `deferred-replay-active.json` carries `state: training` with no owner, so
     `recover_interrupted_deferred_replay` must roll the interrupted interval
     back. That path reflink-clones `brain.last-good.wbrain` over
     `brain.wbrain` and `os.replace`s it, which unlinks the old inode. Measured
     by extent subtraction beforehand: 587.83 GB is unique to the live file and
     0.00 GB is unique to the guard, so free space must go from ~22.8 GB to
     roughly 610 GB. Anything much less means the rollback did not run.

  2. The rebuilt brain server must be the one that comes up, not the node that
     is still resident from the previous generation -- deploy is not load, and
     a capability verdict from the old binary is worthless.

  3. The burn must fall. The old rate was 108.66 GB/h of full-body re-appends;
     `clean_skips` against `page_outs` says directly how much of that was
     eviction churn rather than learning.

The unit file changed too (`RestartPreventExitStatus=42 90`), and systemd reads
from /etc/systemd/system, not from the repository copy -- installing the file
without `daemon-reload` is its own silent no-op.
"""
import glob
import json
import os
import shutil
import subprocess
import time
import urllib.request

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def sh(cmd, timeout=180):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return (proc.stdout + proc.stderr)[-2500:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def free_gb():
    return round(shutil.disk_usage(R).free / 2**30, 2)


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
                "file": os.path.basename(path),
                "durable_next_row": blob.get("durable_next_row"),
                "accepted_episodes": blob.get("accepted_episodes"),
                "age_s": round(age, 1),
            }
    return best


def census():
    found = {}
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
                    cmd = handle.read().replace(b"\0", b" ").decode(
                        "utf-8", "replace"
                    )
            except OSError:
                continue
            if pat in cmd:
                count += 1
        found[name] = count
    return found


out["free_gb_start"] = free_gb()
out["census_start"] = census()

# The running node predates the rebuild. The restore path stops and relaunches
# it from `--node-bin`, but a node left running would be adopted instead.
out["stop_stale_node"] = sh("pkill -f w1z4rd_brain_server 2>&1; sleep 3; echo stopped")

out["install_unit"] = sh(
    "install -m 0644 /srv/wizard/project/scripts/aws/"
    "wizard-curriculum-supervisor.service "
    "/etc/systemd/system/wizard-curriculum-supervisor.service && "
    "systemctl daemon-reload && "
    "systemctl show wizard-curriculum-supervisor.service "
    "-p RestartPreventExitStatus --no-pager 2>&1"
)
out["reset_failed"] = sh(
    "systemctl reset-failed wizard-curriculum-supervisor.service 2>&1; echo reset"
)
out["start"] = sh("systemctl start wizard-curriculum-supervisor.service 2>&1; echo started")

# Watch the rollback land. The reflink clone is cheap but the topology
# verification restarts the node and waits on /brain/stats.
timeline = []
for _ in range(18):
    time.sleep(20)
    timeline.append({
        "t": round(time.time() - out["now"], 1),
        "free_gb": free_gb(),
        "census": census(),
        "row": row(),
        "state": (lambda s: s.get("state") if isinstance(s, dict) else None)(
            (lambda p: json.load(open(p)) if os.path.exists(p) else {})(
                f"{R}/curriculum-supervisor.status.json"
            ) if os.path.exists(f"{R}/curriculum-supervisor.status.json") else {}
        ),
    })
out["timeline"] = timeline
out["free_gb_after_rollback"] = free_gb()
out["reclaimed_gb"] = round(out["free_gb_after_rollback"] - out["free_gb_start"], 2)

# Burn over a clean window, plus the counters that explain it.
a = shutil.disk_usage(R).free
row_a = row()
time.sleep(180)
b = shutil.disk_usage(R).free
row_b = row()
out["burn_gb_per_hour"] = round((a - b) / 2**30 / (180 / 3600.0), 2)
out["row_window"] = {"first": row_a, "second": row_b}
if row_a and row_b:
    delta = row_b["durable_next_row"] - row_a["durable_next_row"]
    out["rows_per_second"] = round(delta / 180.0, 3) if delta >= 0 else None

try:
    with urllib.request.urlopen(
        "http://127.0.0.1:18095/brain/stats", timeout=60
    ) as response:
        stats = json.loads(response.read().decode("utf-8"))
    out["stats"] = {
        key: stats.get(key)
        for key in (
            "total_neurons", "total_concepts", "evicted_neurons",
            "resident_terminals", "tick", "page_outs", "clean_skips",
            "page_ins",
        )
        if key in stats
    }
    out["stats_keys_sample"] = sorted(stats.keys())[:40]
except Exception as exc:
    out["stats"] = f"{type(exc).__name__}: {exc}"

out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p NRestarts -p ExecMainStatus --no-pager 2>&1"
).strip()
out["status_tail"] = sh(f"tail -c 800 {R}/curriculum-supervisor.status.json 2>&1")
out["health_tail"] = sh(f"tail -c 1400 {R}/curriculum-health.jsonl 2>&1")
out["brain_files"] = sh(
    "ls -l --block-size=1G --time-style=full-iso "
    f"{R}/brain/*.wbrain 2>&1 | head -6"
)
out["df"] = sh("df -h /srv/wizard | tail -1")
out["free_mem"] = sh("free -g | head -2")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
