#!/bin/bash
# Did the replay worker come back after the cooperative resource yield?
#
# The wake-up payload observed `worker_count: 0` beside `wrapper_count: 1`
# 212 s after a `deferred_replay_resource_yield` that freed 3.0 -> 14.66 GB.
# That is the expected mid-yield instant: the wrapper SIGTERMs the worker,
# the brain restarts (`brain_age_seconds: 0.7`, RSS 20 MB), and the next
# worker resumes from `durable_next_row`. A worker that never returns looks
# identical at that instant, so the reading has to be repeated -- one sample
# cannot tell a duty-cycle trough from a stall.
#
# This probe therefore samples twice, far enough apart to straddle a commit
# (32 rows per commit at sub-1 rows/s is one commit every ~60-90 s), and
# reports the row DELTA rather than an instantaneous rate.
set -uo pipefail

python3 - <<'PY'
import glob
import json
import os
import pathlib
import subprocess
import time

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}


def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def counts():
    """Process census using the SAME patterns the watchdog scans /proc for.

    Measured 2026-09-10: this probe originally guessed `deferred_replay_worker`
    and `deferred_replay`, which match no process on the host, so it reported
    `worker: 0, wrapper: 0` against a block demonstrably advancing 168 rows in
    90 s. A pattern that CANNOT match reports 0 forever.

    The corrected patterns still read 0 for the worker on a healthy replay,
    and that is not a defect either: the worker is `drive_corpora_brain`,
    respawned once per cooperative yield, so an instantaneous sample lands in
    a trough most of the time. LIVENESS COMES FROM THE ROW DELTA, never from
    this census -- the census is here to tell a yield cycle apart from a host
    that has lost its supervisor.
    """
    return {
        "wrapper": len(
            sh("pgrep", "-f", "run_programming_curriculum_service.sh").split()
        ),
        "supervisor": len(
            sh("pgrep", "-f", "programming_curriculum_supervisor.py").split()
        ),
        "worker": len(
            sh("pgrep", "-f",
               "tools.training_standard.drive_corpora_brain").split()
        ),
    }


def progress():
    """Freshest file that actually exposes a row -- never mtime alone.

    CLAUDE.md: selecting on mtime picked the supervisor status file, which
    during a replay carries resume_row/end_row and never durable_next_row,
    so the probe published `row: null`.
    """
    best = None
    for path in glob.glob(str(runtime / "*.progress.json")):
        try:
            data = json.loads(pathlib.Path(path).read_text())
        except Exception:
            continue
        row = data.get("durable_next_row")
        if row is None:
            continue
        entry = {
            "file": os.path.basename(path),
            "row": row,
            "accepted_episodes": data.get("accepted_episodes"),
            "mtime": os.path.getmtime(path),
        }
        if best is None or entry["mtime"] > best["mtime"]:
            best = entry
    return best


# --- durable interval state: the target a replay is measured against ---
# `curriculum-supervisor.status.json` publishes `block_target_row` only in a
# forward stage, so a replay read from it always sees None.
active = {}
try:
    active = json.loads((runtime / "deferred-replay-active.json").read_text())
except Exception as exc:
    active = {"error": str(exc)}
out["active_interval"] = active

out["counts_first"] = counts()
first = progress()
out["progress_first"] = first

time.sleep(90)

out["counts_second"] = counts()
second = progress()
out["progress_second"] = second

if first and second:
    delta = second["row"] - first["row"]
    out["row_delta"] = delta
    # A decrease is a worker restart, not a negative rate.
    out["counter_reset"] = delta < 0
    out["rows_per_second"] = None if delta < 0 else round(delta / 90.0, 3)
    ep_first = first.get("accepted_episodes")
    ep_second = second.get("accepted_episodes")
    if isinstance(ep_first, int) and isinstance(ep_second, int):
        out["accepted_delta"] = ep_second - ep_first
        out["accepted_reset"] = ep_second < ep_first

# --- what the supervisor last recorded, and how stale it is ---
try:
    status = json.loads(
        (runtime / "curriculum-supervisor.status.json").read_text()
    )
    out["status_state"] = status.get("state")
    out["status_phase"] = status.get("phase")
    out["status_age_seconds"] = round(
        time.time() - status.get("updated_unix", 0), 1
    )
except Exception as exc:
    out["status_error"] = str(exc)

# --- memory headroom: the yield fires against minimum_free_memory_gb ---
try:
    meminfo = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        meminfo[key] = int(rest.strip().split()[0])
    out["available_gb"] = round(meminfo.get("MemAvailable", 0) / 1048576, 2)
except Exception as exc:
    out["meminfo_error"] = str(exc)

# --- most recent lifecycle events, so a yield loop is visible as a loop ---
tail = []
health = runtime / "curriculum-health.jsonl"
try:
    lines = health.read_text(errors="replace").splitlines()[-40:]
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        # The ledger's kind key is `kind`, not `event`; reading `event`
        # published eighteen consecutive nulls and said nothing.
        tail.append({
            "kind": rec.get("kind") or rec.get("event"),
            "unix": rec.get("unix") or rec.get("observed_unix")
            or rec.get("updated_unix"),
        })
except Exception as exc:
    out["health_error"] = str(exc)
out["recent_events"] = tail[-18:]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
