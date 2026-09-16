python3 - <<'PY'
"""Can this host recover from its own disk halt, and would the interval fit?

The supervisor exited 90 (`DISK_EXHAUSTED_EXIT`) at 11:44 UTC on 2026-09-11 and
`RestartPreventExitStatus=42 90` makes that terminal, so the unit has been
`failed` ever since. Three `disk_floor_reclaim` attempts each returned 0.0 GB,
because `reclaim_disk_for_floor` calls `prune_resolved_deferred_bases` and
nothing else, and CLAUDE.md records that reclaim as exhausted on this volume
(2 prunable directories, 0.01 GB upper bound). The reclaim that DOES work here
-- rolling `brain.wbrain` back onto the `brain.last-good.wbrain` guard -- only
runs from `recover_interrupted_deferred_replay` on the STARTUP path, which a
terminal halt can never reach.

Two numbers decide whether restarting is a fix or a loop, and neither is in the
watchdog payload:

  1. What the rollback returns. Not the file size: this is an XFS reflink
     volume where `brain.wbrain` was cloned from the guard and appended to, so
     only the blocks it does NOT share come back. The previous probe printed
     `wbrain_total_gb 5799882.57` for an 854 GB file -- it summed the wrong
     `filefrag` column. Parsed by the documented layout here
     (`ext: logical..logical: physical..physical: length: flags`) and subtracted
     as sorted INTERVALS, never as a per-block set: 854 GB is 223M blocks and a
     set of those is not a measurement, it is an OOM.

  2. The burn, and the interval's ETA at its own phase's rate. CLAUDE.md's
     history says 112-257 GB/h, at which nothing fits any window. But the last
     eight yields before the halt show free falling 180.19 -> 147.28 GB, which
     is a different order of magnitude, and the clean-skip suppression landed
     between those two observations. If the burn really is ~7 GB/h the window
     is ~57 h against a ~25 h interval and ordering was never the problem; if
     it is still ~100 GB/h the work unit has to be split. This measures it from
     the yields' OWN timestamps across the whole halted generation rather than
     from one window, because an instantaneous rate on this host is a duty
     cycle.

Read-only. Nothing here starts, stops or deletes anything.
"""
import bisect
import collections
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
P = "/srv/wizard/project"
GIB = 1024 ** 3
out = {"now": time.time()}


def sh(cmd, timeout=300):
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout)
        return {"rc": proc.returncode,
                "out": (proc.stdout or "").strip()[-4000:],
                "err": (proc.stderr or "").strip()[-2000:]}
    except Exception as exc:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# 1. The burn, from every yield this generation published.
# --------------------------------------------------------------------------
# The generation that halted started when systemd last started the unit. Take
# that boundary from the unit itself rather than assuming it, then keep only
# the yields inside it: yields from earlier generations straddle rollbacks and
# would make the volume look like it rose on its own, which it never does.
unit = sh(
    "systemctl show wizard-curriculum-supervisor.service "
    "-p ExecMainStartTimestamp -p ExecMainExitTimestamp -p ExecMainStatus "
    "-p ActiveState -p Result -p NRestarts --no-pager"
)["out"]
out["unit"] = unit
started = None
for line in unit.splitlines():
    if line.startswith("ExecMainStartTimestamp="):
        stamp = line.split("=", 1)[1].strip()
        if stamp:
            parsed = sh(f"date -d '{stamp}' +%s")["out"]
            if parsed.isdigit():
                started = int(parsed)
out["generation_started_unix"] = started

yields, exhausted, admitted = [], [], []
kinds = collections.Counter()
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            kinds[kind] += 1
            if kind == "deferred_replay_resource_yield":
                yields.append(event)
            elif kind == "disk_exhausted_unrecoverable":
                exhausted.append(event)
            elif kind == "deferred_replay_admitted":
                admitted.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)
out["ledger_kinds"] = dict(kinds)


def stamp_of(event):
    for key in ("updated_unix", "unix", "observed_unix", "ts"):
        value = event.get(key)
        if isinstance(value, (int, float)) and value > 1_000_000_000:
            return float(value)
    return None


# `append_health_event` may not stamp every record. Where it does not, the
# ledger is still append-ordered, so position is a usable proxy for order --
# but never for RATE, so an unstamped run is reported as unusable rather than
# interpolated into a number that looks measured.
burn = {"unstamped_yields": 0}
series = []
for event in yields:
    when = stamp_of(event)
    free = event.get("disk_free_bytes_after")
    if not isinstance(free, (int, float)):
        continue
    if when is None:
        burn["unstamped_yields"] += 1
        continue
    series.append((float(when), int(free)))
series.sort()
burn["stamped_yields"] = len(series)

def window_rate(points, label):
    """GB/h across a run of yields, ignoring any segment where free ROSE.

    A rise is a rollback or a prune, not negative burn -- the same counter-reset
    trap CLAUDE.md records for `durable_next_row` and `accepted_episodes`. Those
    segments are excluded from both numerator and denominator instead of being
    allowed to cancel real burn out of the average.
    """
    fell_bytes = 0.0
    fell_seconds = 0.0
    rises = 0
    for (t0, f0), (t1, f1) in zip(points, points[1:]):
        if t1 <= t0:
            continue
        if f1 > f0:
            rises += 1
            continue
        fell_bytes += (f0 - f1)
        fell_seconds += (t1 - t0)
    rate = (fell_bytes / GIB) / (fell_seconds / 3600.0) if fell_seconds > 0 else None
    return {
        "label": label,
        "points": len(points),
        "rises_excluded": rises,
        "hours_measured": round(fell_seconds / 3600.0, 2),
        "gb_burned": round(fell_bytes / GIB, 2),
        "gb_per_hour": round(rate, 2) if rate is not None else None,
    }


burn["all_time"] = window_rate(series, "all stamped yields")
if started:
    burn["halted_generation"] = window_rate(
        [p for p in series if p[0] >= started], "generation that halted"
    )
burn["last_40"] = window_rate(series[-40:], "last 40 yields")
burn["free_gb_trace_last_20"] = [
    [round(t), round(f / GIB, 2)] for t, f in series[-20:]
]
out["burn"] = burn

# --------------------------------------------------------------------------
# 2. The interval's own end-to-end rate, and its ETA.
# --------------------------------------------------------------------------
status = json.loads((R / "curriculum-supervisor.status.json").read_text())
out["status"] = status
active = json.loads((R / "deferred-replay-active.json").read_text())
out["active_state"] = active.get("state")
out["active_interval_id"] = active.get("interval_id")

interval = {
    "interval_id": status.get("interval_id"),
    "start_row": status.get("start_row"),
    "resume_row": status.get("resume_row"),
    "end_row": status.get("end_row"),
}
if all(isinstance(interval[k], (int, float))
       for k in ("start_row", "resume_row", "end_row")):
    trained = int(interval["resume_row"]) - int(interval["start_row"])
    remaining = int(interval["end_row"]) - int(interval["resume_row"])
    elapsed_h = ((status.get("updated_unix", 0) - started) / 3600.0
                 if started else None)
    rate = (trained / (elapsed_h * 3600.0)) if elapsed_h else None
    interval.update({
        "rows_trained_this_generation": trained,
        "rows_remaining": remaining,
        "generation_hours": round(elapsed_h, 2) if elapsed_h else None,
        "rows_per_second": round(rate, 4) if rate else None,
        "eta_hours_at_that_rate": (round(remaining / rate / 3600.0, 2)
                                   if rate else None),
    })
out["interval"] = interval

# --------------------------------------------------------------------------
# 3. What a rollback returns: blocks the live brain does not share.
# --------------------------------------------------------------------------
EXTENT = re.compile(
    r"^\s*(\d+):\s+(\d+)\.\.\s*(\d+):\s+(\d+)\.\.\s*(\d+):\s+(\d+):"
)


def extents(path):
    """[(physical_start_block, length_blocks)] by filefrag's documented layout."""
    proc = subprocess.run(["filefrag", "-v", str(path)],
                          capture_output=True, text=True, timeout=3600)
    found = []
    for line in proc.stdout.splitlines():
        match = EXTENT.match(line)
        if match:
            found.append((int(match.group(4)), int(match.group(6))))
    return found


def merge(spans):
    """Sorted, non-overlapping [start, end) intervals."""
    ordered = sorted((s, s + n) for s, n in spans if n > 0)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def unique_blocks(mine, theirs):
    """Blocks in `mine` not covered by `theirs`, by interval subtraction."""
    guard = merge(theirs)
    starts = [span[0] for span in guard]
    total = 0
    for start, end in merge(mine):
        cursor = start
        index = max(0, bisect.bisect_right(starts, cursor) - 1)
        while cursor < end and index < len(guard):
            g_start, g_end = guard[index]
            if g_end <= cursor:
                index += 1
                continue
            if g_start >= end:
                break
            total += max(0, min(g_start, end) - cursor)
            cursor = max(cursor, g_end)
            index += 1
        total += max(0, end - cursor)
    return total


rollback = {}
try:
    live = R / "brain/brain.wbrain"
    guard = R / "brain/brain.last-good.wbrain"
    rollback["live_size_gb"] = round(live.stat().st_size / GIB, 2)
    rollback["guard_size_gb"] = round(guard.stat().st_size / GIB, 2)
    live_ex = extents(live)
    guard_ex = extents(guard)
    block = 4096
    live_blocks = sum(n for _, n in live_ex)
    rollback["live_extents"] = len(live_ex)
    rollback["guard_extents"] = len(guard_ex)
    rollback["live_allocated_gb"] = round(live_blocks * block / GIB, 2)
    rollback["guard_allocated_gb"] = round(
        sum(n for _, n in guard_ex) * block / GIB, 2)
    unique = unique_blocks(live_ex, guard_ex)
    rollback["live_unique_gb"] = round(unique * block / GIB, 2)
    rollback["shared_gb"] = round((live_blocks - unique) * block / GIB, 2)
    # The guard's own unique blocks are what deleting the GUARD would return --
    # reported only to show it is the wrong file to delete.
    rollback["guard_unique_gb"] = round(
        unique_blocks(guard_ex, live_ex) * block / GIB, 2)
except Exception as exc:  # noqa: BLE001
    rollback["error"] = f"{type(exc).__name__}: {exc}"
usage = shutil.disk_usage(R)
rollback["free_gb_now"] = round(usage.free / GIB, 2)
rollback["total_gb"] = round(usage.total / GIB, 2)
rollback["floor_gb"] = status.get("minimum_free_disk_gb")
if isinstance(rollback.get("live_unique_gb"), float):
    predicted = rollback["free_gb_now"] + rollback["live_unique_gb"]
    rollback["predicted_free_after_rollback_gb"] = round(predicted, 2)
    floor = float(status.get("minimum_free_disk_gb") or 0)
    rollback["window_gb"] = round(predicted - floor, 2)
    rate = (out["burn"].get("halted_generation") or {}).get("gb_per_hour")
    if rate:
        rollback["window_hours_at_measured_burn"] = round(
            (predicted - floor) / rate, 2)
out["rollback"] = rollback

# --------------------------------------------------------------------------
# 4. The queue, imported IN PROCESS so an ImportError is reported, not lost.
# --------------------------------------------------------------------------
queue = {}
try:
    import sys
    sys.path.insert(0, P)
    from scripts.programming_curriculum_supervisor import (  # noqa: E402
        unresolved_deferred_intervals, order_replay_candidates,
    )
    pending = unresolved_deferred_intervals(R)
    queue["pending"] = len(pending)
    try:
        ordered = order_replay_candidates(R, pending)
    except TypeError as exc:
        queue["arity_note"] = str(exc)
        ordered = order_replay_candidates(pending)
    queue["head"] = [
        {"id": e.get("interval_id"), "phase": e.get("phase"),
         "span": int(e["end_row"]) - int(e["start_row"])}
        for e in ordered[:15]
    ]
    spans = collections.defaultdict(list)
    for event in pending:
        spans[event.get("phase")].append(
            int(event["end_row"]) - int(event["start_row"]))
    queue["by_phase"] = {
        phase: {"intervals": len(values), "rows": sum(values),
                "span_min": min(values), "span_max": max(values)}
        for phase, values in sorted(spans.items())
    }
except Exception as exc:  # noqa: BLE001
    queue["error"] = f"{type(exc).__name__}: {exc}"
out["queue"] = queue

# --------------------------------------------------------------------------
# 5. Per-phase end-to-end rows/s, from intervals that actually ADMITTED.
# --------------------------------------------------------------------------
# A span key is only a proxy for cost; go rows and jupyter rows differ by an
# order of magnitude, so the ordering committed in 345b1cb sorts on a unit that
# is not the one the window is spent in.
rates = collections.defaultdict(list)
for event in admitted:
    phase = event.get("phase")
    interval_id = event.get("interval_id") or ""
    parts = interval_id.split(":")
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        rates[phase].append(int(parts[2]) - int(parts[1]))
out["admitted_spans_by_phase"] = {
    phase: {"count": len(values), "rows": sum(values)}
    for phase, values in sorted(rates.items())
}

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
