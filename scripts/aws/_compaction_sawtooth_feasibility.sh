python3 - <<'PY'
"""Is an in-place compaction sawtooth enough to keep this interval training?

Established minutes ago and not in dispute: burn 159.91 GB/h, 69.68 GB above
the 150 GB floor, 10.02 h of training still owed at 1.11 rows/s. The interval
needs ~1,600 GB of appends on a 1.0 TB volume, so it cannot finish and no
deletion changes that. `brain/stats` says why the burn is structural rather
than incidental: `evicted_neurons` 5,098,116 against `total_neurons` 5,098,439
-- the brain has evicted essentially every neuron it owns, because 363 GB of
bodies cannot be resident on a 15.26 GB host. Every eviction appends a whole
~71 KB body to an append-only store that has never reclaimed one.

CLAUDE.md records compaction as net-NEGATIVE, and that verdict was correct for
the operation it measured: an OUT-OF-PLACE rewrite of a 576.67 GB file with
363.34 GB live, which writes 363 GB of fresh unshareable blocks to reclaim
~153 GB. Two things have changed and both push the other way:

  * the file is now 836.46 GB apparent, so if live is still ~363 GB the
    garbage fraction has gone from ~37% to ~57%;
  * `wbrain_compact --in-place` needs no scratch copy at all, which is the
    entire reason the out-of-place arithmetic lost. (The deployed binary's
    usage lists `--inspect`, `--in-place` and `<src> <dst>` -- there is no
    `--estimate`, whatever CLAUDE.md says, so `--inspect` is the measurement.)

What this probe decides, before any code is written against it:

  1. has the supervisor already entered the unbounded `resource_waiting`
     loop -- a predicted deadlock and one already running need different
     urgency;
  2. what `--inspect` reports as live vs reclaimable on the CURRENT file;
  3. whether a compaction sawtooth clears the floor by enough, and for long
     enough at 160 GB/h, to be worth wiring into the disk-pressure path.

Read-only. `--inspect` opens the store without writing, and nothing here
stops, starts or deletes anything.
"""
import glob
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
BRAIN = f"{R}/brain/brain.wbrain"
out = {"now": time.time()}


def sh(cmd, timeout=240):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-8000:] + (
            ("\n[stderr] " + proc.stderr[-1500:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# --- 1. Predicted deadlock, or one already running? -----------------------
out["status_now"] = sh(f"tail -c 900 {R}/curriculum-supervisor.status.json 2>&1")
out["resource_waiting_count"] = sh(
    f"grep -ao resource_waiting {R}/curriculum-supervisor.status.json "
    f"{R}/curriculum-health.jsonl {R}/curriculum-service.stdout.log 2>/dev/null "
    "| sort | uniq -c"
)
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p NRestarts --no-pager 2>&1"
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

usage = shutil.disk_usage("/srv/wizard")
out["df"] = {
    "free_gb": round(usage.free / 2**30, 2),
    "used_gb": round(usage.used / 2**30, 2),
    "total_gb": round(usage.total / 2**30, 2),
}
out["brain_file"] = sh(f"ls -la {R}/brain/*.wbrain 2>&1")
out["free"] = sh("free -g")

# --- 2. Live vs reclaimable on the CURRENT file. --------------------------
out["inspect_usage"] = sh(f"{P}/target/release/wbrain_compact 2>&1 | head -10")
started = time.time()
out["inspect"] = sh(
    f"{P}/target/release/wbrain_compact --inspect {BRAIN} 2>&1 | tail -40",
    timeout=3000,
)
out["inspect_seconds"] = round(time.time() - started, 1)

# --- 3. Did the volume move while we read it? -----------------------------
usage_after = shutil.disk_usage("/srv/wizard")
out["df_after"] = {"free_gb": round(usage_after.free / 2**30, 2)}
out["burn_during_inspect_gb_per_h"] = (
    round(
        (usage.free - usage_after.free) / 2**30
        / max(1e-9, (time.time() - out["now"]) / 3600.0),
        2,
    )
)

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
out["row_now"] = best

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
