python3 - <<'PY'
"""Can this interval reach its gate before the disk floor stops it?

The `quarantine_ready` wake-up reads healthy: forward harvesting is complete,
a supervisor is 2.73 h old, the row is advancing, and admission was only 10.1 h
ago. Every liveness check CLAUDE.md prescribes passes. The fault is not in any
of them -- it is in an arithmetic nobody publishes:

    free 261.65 GB - floor 150 GB = 111.65 GB of headroom
    measured sustained burn                ~112 GB/h
    => roughly ONE HOUR of runway

against an interval at row 221,736 of 262,144 -- 40,408 rows still owed. The
previous session shipped `--min-free-disk-gb 150` to convert a 115x ENOSPC
crash loop into a clean cooperative yield. Read the code that yield lands in
(`programming_curriculum_supervisor.py` 4766-4812) and it is:

    while (memory below floor) or (disk free < disk_floor_bytes):
        publish({"state": "resource_waiting", ...})
        time.sleep(poll_seconds)

There is no reclaim inside that loop -- `prune_resolved_deferred_bases` is
called at 3686, 4065 and 4321, and at none of the three disk-wait sites. Once
the worker is stopped the `.wbrain` stops growing, so free space stops falling
and never rises. The loop is therefore not a yield, it is a DEADLOCK: the unit
stays `active`, the row freezes at a durable boundary, and nothing alarms,
because every heartbeat rule we wrote says a frozen row during settlement is
normal by design. That is strictly harder to see than the crash loop it
replaced.

This probe refuses to act on that reasoning without measuring it:

  * burn rate and row rate over the SAME window, so runway and ETA are
    comparable rather than one being recalled from a previous session;
  * whether `resource_waiting` has ever actually been published, which
    distinguishes a predicted deadlock from one already running;
  * a reclaim inventory that separates reflink-shared bytes from bytes a
    delete would actually return, because `du` inflates on this volume and
    only `df` measures a reclaim.

It writes nothing and deletes nothing.
"""
import glob
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
out = {"now": time.time()}


def sh(cmd, timeout=120):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-6000:] + (
            ("\n[stderr] " + proc.stderr[-600:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def freshest_row():
    """Row from the freshest file that ACTUALLY EXPOSES one (CLAUDE.md rule)."""
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


def sample():
    usage = shutil.disk_usage("/srv/wizard")
    return {
        "t": time.time(),
        "free_bytes": usage.free,
        "free_gb": round(usage.free / 2**30, 2),
        "row": freshest_row(),
    }


# --- 1. Burn and row rate over the SAME window. ----------------------------
first = sample()
time.sleep(180)
second = sample()
out["first"], out["second"] = first, second

elapsed_h = (second["t"] - first["t"]) / 3600.0
burn_gb_h = (first["free_bytes"] - second["free_bytes"]) / 2**30 / elapsed_h
row_a = (first["row"] or {}).get("durable_next_row")
row_b = (second["row"] or {}).get("durable_next_row")
rows_per_s = None
if row_a is not None and row_b is not None and row_b >= row_a:
    rows_per_s = (row_b - row_a) / (second["t"] - first["t"])

FLOOR_GB = 150.0
TARGET = 262144
headroom_gb = second["free_gb"] - FLOOR_GB
out["arithmetic"] = {
    "window_seconds": round(second["t"] - first["t"], 1),
    "burn_gb_per_hour": round(burn_gb_h, 2),
    "rows_per_second": round(rows_per_s, 3) if rows_per_s else rows_per_s,
    "headroom_gb_above_floor": round(headroom_gb, 2),
    "runway_hours_to_floor": (
        round(headroom_gb / burn_gb_h, 2) if burn_gb_h > 0.01 else None
    ),
    "rows_remaining": (TARGET - row_b) if row_b is not None else None,
    "eta_hours_to_gate": (
        round((TARGET - row_b) / rows_per_s / 3600.0, 2)
        if rows_per_s and row_b is not None else None
    ),
}

# --- 2. Is the deadlock predicted, or already running? ---------------------
out["resource_waiting_seen"] = sh(
    f"grep -c resource_waiting {R}/curriculum-supervisor.status.json 2>/dev/null; "
    f"grep -ao resource_waiting {R}/curriculum-health.jsonl 2>/dev/null | wc -l; "
    f"grep -ao resource_waiting {R}/curriculum-service.stdout.log 2>/dev/null | wc -l"
)
out["status_now"] = sh(f"tail -c 1200 {R}/curriculum-supervisor.status.json 2>&1")
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service "
    "-p ActiveState -p SubState -p NRestarts -p ExecMainStatus --no-pager 2>&1"
)

# --- 3. Does the DEPLOYED supervisor match the repo at the wait loop? ------
out["deployed_wait_loop"] = sh(
    f"grep -n 'resource_waiting' {P}/scripts/programming_curriculum_supervisor.py 2>&1"
)
out["deployed_prune_sites"] = sh(
    f"grep -n 'prune_resolved_deferred_bases' "
    f"{P}/scripts/programming_curriculum_supervisor.py 2>&1"
)
out["deployed_floor"] = sh(
    f"grep -n 'min-free-disk-gb' {P}/scripts/aws/run_programming_curriculum_service.sh 2>&1"
)

# --- 4. Reclaim inventory. `du` inflates on reflink; report BOTH. ----------
out["df"] = sh("df -h /srv/wizard; df -i /srv/wizard")
out["wbrain_files"] = sh(
    "find /srv/wizard -name '*.wbrain' -printf '%s\\t%n\\t%p\\n' 2>/dev/null "
    "| sort -rn | head -20"
)
out["runtime_dirs"] = sh(
    "du -x -d1 --block-size=1M /srv/wizard 2>/dev/null | sort -rn | head -20",
    timeout=400,
)
out["runtime_children"] = sh(
    f"du -x -d1 --block-size=1M {R} 2>/dev/null | sort -rn | head -25",
    timeout=400,
)
out["big_logs"] = sh(
    f"find {R} -type f -size +200M -printf '%s\\t%p\\n' 2>/dev/null | sort -rn | head -20"
)
out["deferred_dirs"] = sh(
    f"ls -1d {R}/deferred-* 2>/dev/null | wc -l; "
    f"ls -1 {R}/*.base 2>/dev/null | wc -l"
)

# --- 5. Memory, because the burn is eviction churn. ------------------------
out["free"] = sh("free -g")
out["brain_rss"] = sh(
    "ps -eo pid,rss,etimes,comm,args --sort=-rss 2>/dev/null "
    "| grep -i -E 'brain|node' | head -5"
)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
