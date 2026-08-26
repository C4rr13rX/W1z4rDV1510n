#!/usr/bin/env python3
"""Wake the agent when the curriculum stops converting paid compute into work.

The failure this exists to catch is not a crash -- a crash is loud. It is the
quiet one that has already cost real money twice: the supervisor reports
`active`, the brain answers `/health`, the instance bills by the hour, and
NOTHING is being admitted. Every check below is a different way that can
happen, because each has actually happened:

  no_admission     the gate fails every interval, so replay runs forever and
                   retires nothing (measured: 0 admitted / 18 failed over 48h)
  not_training     no worker process and the tick is frozen -- but NOT during
                   a gate, which legitimately settles the brain to zero ticks
  supervisor_down  the systemd unit died or is restart-looping
  brain_down       /health stopped answering
  queue_growing    deferred count rising: the canary is quarantining faster
                   than the replay admits (measured: 43 -> 49 in under an hour)
  disk_low         /srv/wizard below the supervisor's own floor
  memory_low       free RAM below the floor, which pauses training silently
  status_stale     the supervisor has not written its status in over an hour
                   while claiming to be running

Exit 0 = still healthy at the deadline. Exit 2 = a problem, named and
described on stdout so the agent can act on it without re-deriving anything.

The point is the exit code: the harness re-invokes the agent when a
background command finishes, so exiting non-zero IS the alert.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

SP = pathlib.Path(__file__).resolve().parent
PROBE = SP / "_watchdog_probe.sh"

#: How long the queue may go without a single admission before that is a
#: fault.
#:
#: One interval measures ~161 minutes end to end, so this cannot be tighter
#: than that without firing on a healthy run. 3h20m is one interval plus a
#: 20-minute margin -- the earliest point at which a missed admission is
#: provable rather than suspected.
#:
#: A FAILED gate does not wait for this: `deferred_replay_failed` is a
#: verdict, not silence, and is reported the moment it appears. This limit
#: only catches the case where no verdict arrives at all.
ADMISSION_SILENCE_LIMIT = 200 * 60

#: How often to look.
#:
#: A hard fault -- dead supervisor, dead brain, frozen tick with nothing
#: running -- is true the moment it happens, so the only thing standing
#: between it and the alert is this interval. Ninety seconds costs one cheap
#: SSM call per poll and bounds the wasted billing at a minute and a half
#: rather than ten.
POLL_SECONDS = 90

#: Total watch window. Re-armed by the agent when it fires.
DEADLINE_SECONDS = 20 * 3600

PROBE_BODY = r"""
R=/srv/wizard/runtime/programming-integrated-20260713
python3 - <<'PY'
import json, os, re, time, collections, subprocess

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}

# --- supervisor unit -------------------------------------------------------
try:
    unit = subprocess.run(["systemctl", "is-active",
                           "wizard-curriculum-supervisor.service"],
                          capture_output=True, text=True, timeout=30)
    out["unit"] = unit.stdout.strip()
except Exception as error:
    out["unit"] = f"unknown: {error}"

# --- status file -----------------------------------------------------------
status_path = R + "/curriculum-supervisor.status.json"
try:
    status = json.load(open(status_path))
    out["state"] = status.get("state")
    out["phase"] = status.get("phase")
    out["status_error"] = str(status.get("error") or "")[:200]
    out["status_age"] = round(time.time() - os.path.getmtime(status_path))
except Exception as error:
    out["state"] = f"unreadable: {error}"
    out["status_age"] = -1

# --- interval ledger -------------------------------------------------------
state = {}
for line in open(R + "/curriculum-deferred-intervals.jsonl", encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        state[row["interval_id"]] = row
counts = collections.Counter(row["status"] for row in state.values())
out["deferred"] = counts.get("deferred", 0)
out["resolved"] = counts.get("resolved", 0)

# When did an interval last reach `resolved`? That is the only event that
# means paid compute became permanent progress.
resolved_times = [row.get("updated_unix") or 0
                  for row in state.values() if row["status"] == "resolved"]
out["last_admission_age"] = (round(time.time() - max(resolved_times))
                             if resolved_times else -1)

# A FAILED verdict is the loudest possible signal and must not wait for the
# silence threshold: `deferred_replay_failed` is exactly what repeated 18
# times over 48 hours while the instance billed and admitted nothing. Report
# any that landed since the running binary was built, with the failing suite
# names, so the agent starts from the cause instead of rediscovering it.
try:
    binary_built = os.path.getmtime(
        "/srv/wizard/project/target/release/w1z4rd_brain_server")
except OSError:
    binary_built = 0
recent_failures = []
try:
    for line in open(R + "/curriculum-health.jsonl", encoding="utf-8"):
        if '"deferred_replay_failed"' not in line:
            continue
        event = json.loads(line)
        if (event.get("updated_unix") or 0) < binary_built:
            continue
        error = str(event.get("error") or "")
        suites = re.findall(
            r"'name':\s*'([^']+)'[^}]*?'passed':\s*False", error)
        recent_failures.append({
            "at": round(time.time() - (event.get("updated_unix") or 0)),
            "phase": event.get("phase"),
            "suites": suites[:4],
        })
except Exception:
    pass
out["failed_since_deploy"] = len(recent_failures)
out["last_failure"] = recent_failures[-1] if recent_failures else None

# --- is anything actually running? -----------------------------------------
def running(pattern):
    try:
        found = subprocess.run(["pgrep", "-f", pattern],
                               capture_output=True, text=True, timeout=30)
        return bool(found.stdout.strip())
    except Exception:
        return False

out["worker"] = running("drive_corpora_brain")
# A live process is not the same as a working one. Measured 2026-08-26, the
# supervisor sat in state `deferred_replay_training` for 1h49m having burned
# 0 CPU seconds, mid-interval at 92% -- deadlocked, not training. The status
# file was the tell: it had not been rewritten in 6,500 seconds while the
# unit reported active and a worker PID existed.
try:
    out["service_log_age"] = round(
        time.time() - os.path.getmtime(R + "/curriculum-service.stderr.log"))
except OSError:
    out["service_log_age"] = -1
# The replay's own progress file is the heartbeat. Measured 2026-08-26 while
# training ran at 936 rows/45s: the status file was 2,905 s old and the
# service log 2,909 s, because the supervisor writes those only at state
# TRANSITIONS -- but the progress file was 0 s old. Watching the status file
# raised a false alarm on a perfectly healthy run.
import glob as _glob
progress = _glob.glob(R + "/deferred-replay-*.progress.json")
if progress:
    newest = max(progress, key=os.path.getmtime)
    out["progress_age"] = round(time.time() - os.path.getmtime(newest))
    try:
        out["progress_row"] = json.load(open(newest)).get("ram_next_row")
    except Exception:
        out["progress_row"] = None
else:
    out["progress_age"] = -1
    out["progress_row"] = None
# A gate legitimately freezes the tick: it settles the brain first. Treating
# that as a stall is a false alarm, and one nearly sent me chasing a
# non-problem.
out["gating"] = running("programming_.*[.]py --endpoint")

# --- brain -----------------------------------------------------------------
# Three tries before calling the brain down. A single 15-second timeout
# fired twice on a brain that was up the whole time -- it was busy, not
# gone -- and each false alarm costs the watchdog credibility, which is
# how the original stall went unnoticed for two days.
out["brain_up"] = False
for _attempt in range(3):
    try:
        health = subprocess.run(
            ["curl", "-fsS", "--max-time", "20",
             "http://127.0.0.1:18095/health"],
            capture_output=True, text=True, timeout=40)
        if health.returncode == 0:
            out["brain_up"] = True
            break
    except Exception:
        pass
    time.sleep(5)

def tick():
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "15",
             "http://127.0.0.1:18095/stats"],
            capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout).get("tick")
    except Exception:
        return None

first = tick()
time.sleep(25)
second = tick()
out["tick_delta"] = (second - first) if (first and second) else None

# --- resources -------------------------------------------------------------
try:
    disk = os.statvfs("/srv/wizard")
    out["disk_free_gb"] = round(disk.f_bavail * disk.f_frsize / 1e9)
except Exception:
    out["disk_free_gb"] = -1
try:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            out["mem_free_gb"] = round(int(line.split()[1]) / 1e6)
            break
except Exception:
    out["mem_free_gb"] = -1

print(json.dumps(out))
PY
"""


def probe() -> dict | None:
    PROBE.write_text(PROBE_BODY, encoding="utf-8")
    try:
        run = subprocess.run(
            ["C:/Python313/python.exe", str(SP / "ssm.py"), str(PROBE), "300"],
            capture_output=True, text=True, timeout=900, encoding="utf-8",
            errors="replace")
    except Exception as error:
        print(f"  probe failed: {error}", flush=True)
        return None
    for line in run.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"deferred"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def faults(now: dict, baseline_deferred: int) -> list[str]:
    """Every reason this run is not converting compute into progress."""
    found = []

    if now.get("unit") != "active":
        found.append(f"supervisor_down: unit is {now.get('unit')!r}")

    if not now.get("brain_up"):
        found.append("brain_down: /health is not answering")

    # A failed verdict is not silence -- act on it now, not in 3h20m.
    failures = now.get("failed_since_deploy") or 0
    if failures:
        last = now.get("last_failure") or {}
        suites = ", ".join(last.get("suites") or []) or "unnamed"
        found.append(
            f"gate_failing: {failures} deferred_replay_failed since deploy; "
            f"latest {last.get('at')}s ago on {last.get('phase')} "
            f"-- failing suites: {suites}")

    age = now.get("last_admission_age", -1)
    if age > ADMISSION_SILENCE_LIMIT:
        found.append(
            f"no_admission: nothing admitted for {age // 3600}h "
            f"{(age % 3600) // 60}m -- the queue is not converting")

    # A frozen tick is only a fault when nothing legitimately froze it.
    if (now.get("tick_delta") == 0
            and not now.get("gating")
            and not now.get("worker")):
        found.append("not_training: no worker, no gate, and the tick is frozen")

    if now.get("deferred", 0) > baseline_deferred:
        found.append(
            f"queue_growing: deferred {baseline_deferred} -> "
            f"{now['deferred']} -- quarantining faster than admitting")

    if 0 <= now.get("disk_free_gb", -1) < 20:
        found.append(f"disk_low: {now['disk_free_gb']}GB free on /srv/wizard")

    if 0 <= now.get("mem_free_gb", -1) < 4:
        found.append(f"memory_low: {now['mem_free_gb']}GB available")

    # A wedge is a stalled HEARTBEAT, not a stale status file.
    #
    # The supervisor writes its status only at state transitions, so during a
    # long replay that file is legitimately tens of minutes old -- measured
    # 2026-08-26, 2,905 s old while training ran at 936 rows/45 s. Alarming
    # on it was a false positive on a healthy run.
    #
    # What a working replay writes continuously is its own progress file. A
    # deadlock shows as that file going stale AND the tick frozen: during the
    # real 1h49m wedge the tick sat at 2,712,230 and nothing advanced, while
    # here the tick moves every poll.
    if (now.get("progress_age", 0) > 10 * 60
            and not now.get("tick_delta")
            and not now.get("gating")):
        found.append(
            f"supervisor_wedged: no replay progress for "
            f"{now['progress_age']}s and the tick is frozen -- "
            f"process alive, not working")

    if now.get("status_age", 0) > 3600 and now.get("state") not in (
            "deferred_replay_training", "continuous_canary", "running"):
        found.append(
            f"status_stale: no status written for {now['status_age']}s "
            f"in state {now.get('state')!r}")

    error = now.get("status_error") or ""
    if error and error != "none":
        found.append(f"supervisor_error: {error}")

    return found


def main() -> int:
    base = probe()
    if base is None:
        print("could not read a baseline", file=sys.stderr)
        return 2
    baseline_deferred = base.get("deferred", 0)
    print(f"watching: deferred={baseline_deferred} "
          f"resolved={base.get('resolved')} "
          f"last_admission={base.get('last_admission_age')}s ago", flush=True)

    deadline = time.monotonic() + DEADLINE_SECONDS
    pending: list[str] = []
    while time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        now = probe()
        if now is None:
            continue

        problems = faults(now, baseline_deferred)
        stamp = time.strftime("%H:%M:%S")

        # A fault must survive a second look before it wakes anyone.
        #
        # Every false alarm so far came from trusting one sample: a momentary
        # /health timeout, and a ledger read that predated three admissions
        # already on disk. Both cleared on the very next poll. Requiring the
        # SAME fault twice in a row costs 90 seconds of detection delay and
        # removes the class of alarm that makes a watchdog ignorable.
        confirmed = sorted(set(problems) & set(pending))
        pending = problems
        if problems and not confirmed:
            print(f"  {stamp} unconfirmed, re-checking: {problems[0][:70]}",
                  flush=True)
            continue
        problems = confirmed
        if problems:
            print(f"\n{stamp} CURRICULUM FAULT -- paid compute is not "
                  f"producing admissions")
            for problem in problems:
                print(f"  * {problem}")
            print("\n  full state:")
            for key in sorted(now):
                print(f"    {key}: {now[key]}")
            return 2

        print(f"  {stamp} ok  deferred={now['deferred']} "
              f"resolved={now['resolved']} "
              f"last_admission={now.get('last_admission_age')}s "
              f"tick+{now.get('tick_delta')} "
              f"{'gating' if now.get('gating') else ''}", flush=True)
        # An admission moves the floor: the queue must never climb back.
        baseline_deferred = min(baseline_deferred, now.get("deferred", 0))

    print("\nstill healthy at the deadline; re-arm the watchdog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
