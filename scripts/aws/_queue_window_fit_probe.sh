python3 - <<'PY'
"""Does ANY pending interval fit the disk window at its own phase's rate?

Measured minutes ago on this host: a rollback returns 414.02 GB (the blocks
`brain.wbrain` does not share with the guard -- the guard's own unique blocks
are 0.00 GB, so deleting IT returns nothing), giving 565.18 GB free against a
150 GB floor: a 415.18 GB window. The halted generation burned 114.12 GB/h
across 101 stamped yields over 3.91 h, so the window is 3.64 h. The interval it
died on needs 24.88 h for its remaining 64,292 rows at its own measured 0.7179
rows/s -- and a rollback restarts it from row 131,072, so really 29.4 h. It
misses by 8x.

`order_replay_candidates` sorts by `(stalls, span)`. Span is a proxy for cost
and it is wrong by the ratio between phases: go rows run ~12.6 rows/s and
jupyter-scientific ~0.72 rows/s, so 18,432 go rows are 0.4 h and 18,432 jupyter
rows are 7.1 h. At a 3.64 h window the cutoff is ~165,000 go rows but only
~9,400 jupyter rows. This measures, per phase, the rate from intervals that
ACTUALLY ADMITTED (end to end, wall clock, which is the unit the window is
spent in) and then asks the binary question: after the stall records push the
stuck interval back, does the head of the queue fit?

If some interval fits, restarting converges and ordering was sufficient.
If none fits, the work unit itself has to be split and ordering is inert.

Read-only.
"""
import collections
import json
import pathlib
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
P = "/srv/wizard/project"
sys.path.insert(0, P)
out = {"now": time.time()}

WINDOW_HOURS = 3.64          # 415.18 GB above the floor at 114.12 GB/h
from scripts.programming_curriculum_supervisor import (  # noqa: E402
    unresolved_deferred_intervals, order_replay_candidates,
    replay_stall_counts,
)

pending = unresolved_deferred_intervals(R)
stalls = replay_stall_counts(R)
ordered = order_replay_candidates(pending, stalls)
out["pending_count"] = len(pending)
out["stall_counts"] = stalls

# ---- per-phase end-to-end rate, from the ledger's own admissions ----------
# An admission is the only event that brackets a whole interval: it is written
# when the gate passes, and the interval's first selection is the preceding
# `deferred_replay_admitted` or the generation start. Pairing consecutive
# admissions within a phase gives wall-clock rows/hour including every yield,
# settlement and recycle -- which is what a wall-clock disk window is spent on.
events = []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") in {
                "deferred_replay_admitted", "deferred_replay_failed",
                "deferred_replay_resource_yield",
            }:
                events.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)


def span_of(interval_id):
    parts = str(interval_id or "").split(":")
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return int(parts[2]) - int(parts[1])
    return None


def stamp(event):
    value = event.get("updated_unix") or event.get("unix")
    return float(value) if isinstance(value, (int, float)) else None


# Rate per phase: rows admitted divided by the wall-clock between the first
# yield seen for that interval and its admission. Reported with the sample
# count, because one interval is an anecdote and this host's instantaneous
# rate is a duty cycle.
first_seen = {}
observed = collections.defaultdict(list)
for event in events:
    interval_id = event.get("interval_id")
    when = stamp(event)
    if not interval_id or when is None:
        continue
    first_seen.setdefault(interval_id, when)
    if event.get("kind") == "deferred_replay_admitted":
        span = span_of(interval_id)
        hours = (when - first_seen[interval_id]) / 3600.0
        if span and hours > 0.05:
            observed[event.get("phase")].append((span, hours))

rate_by_phase = {}
for phase, samples in observed.items():
    rows = sum(s for s, _ in samples)
    hours = sum(h for _, h in samples)
    rate_by_phase[phase] = {
        "samples": len(samples),
        "rows": rows,
        "hours": round(hours, 2),
        "rows_per_hour": round(rows / hours, 1) if hours else None,
        "rows_per_second": round(rows / (hours * 3600), 4) if hours else None,
    }
out["rate_by_phase_from_admissions"] = rate_by_phase

# The halted generation is a direct, independent measurement for the phase it
# was on: 11,584 rows in 4.48 h.
out["measured_directly"] = {
    "jupyter-scientific-partial": {"rows": 11584, "hours": 4.48,
                                   "rows_per_hour": round(11584 / 4.48, 1)}
}

# ---- the binary question -------------------------------------------------
fallback = {"jupyter-scientific-partial": 11584 / 4.48}


def rate_for(phase):
    entry = rate_by_phase.get(phase) or {}
    if entry.get("rows_per_hour"):
        return entry["rows_per_hour"], f"admissions(n={entry['samples']})"
    if phase in fallback:
        return fallback[phase], "halted generation"
    # No evidence for this phase. Say so rather than substituting a number
    # from a different phase -- that substitution is the exact error the span
    # key already makes.
    return None, "no measurement"


rows = []
for event in ordered:
    phase = event.get("phase")
    span = int(event["end_row"]) - int(event["start_row"])
    rate, source = rate_for(phase)
    eta = round(span / rate, 2) if rate else None
    rows.append({
        "interval_id": event.get("interval_id"),
        "phase": phase,
        "span": span,
        "stalls": stalls.get(str(event.get("interval_id")), 0),
        "rate_rows_per_hour": round(rate, 1) if rate else None,
        "rate_source": source,
        "eta_hours": eta,
        "fits_window": (eta is not None and eta <= WINDOW_HOURS),
    })
out["queue_in_selection_order"] = rows
out["window_hours"] = WINDOW_HOURS
fits = [r for r in rows if r["fits_window"]]
out["any_fits"] = bool(fits)
out["fitting_count"] = len(fits)
out["unknown_rate_count"] = sum(1 for r in rows if r["eta_hours"] is None)
out["head_fits"] = rows[0]["fits_window"] if rows else None
out["max_span_that_fits_by_phase"] = {
    phase: (round(rate_for(phase)[0] * WINDOW_HOURS) if rate_for(phase)[0]
            else None)
    for phase in sorted({r["phase"] for r in rows})
}

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
