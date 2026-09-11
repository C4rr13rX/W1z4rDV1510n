python3 - <<'PY'
"""How far does each attempt at this interval get before it is rolled back?

A livelock and a slow convergence look identical in any single sample: both
show a row advancing. They differ in their HISTORY -- a converging interval's
attempts reach steadily further, a livelocked one keeps restarting from the
same row. `curriculum-health.jsonl` is append-only and outlives every worker,
so the attempt history is the one place the difference is visible.

Reports, per attempt at the currently-active interval: the furthest row it
reached and how it ended. A rollback is not recorded as an event, so it is
inferred the honest way -- from a later attempt starting BELOW an earlier
attempt's high-water mark.
"""
import collections
import json
import os
import re
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
LEDGER = f"{R}/curriculum-health.jsonl"
out = {"now": time.time()}

active = {}
try:
    with open(f"{R}/deferred-replay-active.json", "r", encoding="utf-8") as fh:
        active = json.load(fh)
except Exception as exc:
    out["active_error"] = f"{type(exc).__name__}: {exc}"
interval_id = str(active.get("interval_id") or "")
out["interval_id"] = interval_id
match = re.match(r"^(.*):(\d+):(\d+)$", interval_id)
start = int(match.group(2)) if match else None
end = int(match.group(3)) if match else None
out["interval_start"], out["interval_end"] = start, end

rows = []
try:
    with open(LEDGER, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
except Exception as exc:
    out["ledger_error"] = f"{type(exc).__name__}: {exc}"
out["ledger_records"] = len(rows)

mine = [r for r in rows if str(r.get("interval_id") or "") == interval_id]
out["records_for_this_interval"] = len(mine)
out["kinds_for_this_interval"] = dict(
    collections.Counter(str(r.get("kind") or "?") for r in mine))

ROW_KEYS = ("durable_next_row", "resume_row", "trained_rows", "row",
            "next_row", "end_row")


def row_of(record):
    for key in ROW_KEYS:
        value = record.get(key)
        if isinstance(value, (int, float)) and value:
            return int(value)
    return None


timeline = []
for record in mine:
    value = row_of(record)
    timeline.append({
        "unix": record.get("updated_unix") or record.get("unix"),
        "kind": record.get("kind"),
        "row": value,
    })
timeline.sort(key=lambda item: item["unix"] or 0)
out["timeline_tail"] = timeline[-40:]

# Attempts: a new attempt begins wherever the row drops back toward `start`.
attempts, current = [], None
for item in timeline:
    value = item["row"]
    if value is None or start is None or not (start <= value <= (end or value)):
        continue
    if current is None or value < current["high_water"]:
        if current is not None:
            attempts.append(current)
        current = {"first_unix": item["unix"], "first_row": value,
                   "high_water": value, "last_unix": item["unix"],
                   "last_kind": item["kind"]}
    else:
        current["high_water"] = value
        current["last_unix"] = item["unix"]
        current["last_kind"] = item["kind"]
if current is not None:
    attempts.append(current)

for attempt in attempts:
    span = (attempt["last_unix"] or 0) - (attempt["first_unix"] or 0)
    attempt["hours"] = round(span / 3600.0, 2)
    attempt["rows_gained"] = attempt["high_water"] - attempt["first_row"]
    attempt["fraction_of_interval"] = round(
        (attempt["high_water"] - start) / max(1, (end or 1) - start), 3)
out["attempts"] = attempts[-15:]
out["attempt_count"] = len(attempts)
out["best_fraction_ever"] = max(
    (a["fraction_of_interval"] for a in attempts), default=None)
out["rolled_back_attempts"] = sum(
    1 for index, a in enumerate(attempts[1:], 1)
    if a["first_row"] < attempts[index - 1]["high_water"])

# The same question for the whole curriculum: are admissions still happening?
admitted = [r for r in rows if str(r.get("kind") or "") == "deferred_replay_admitted"]
out["admissions_total"] = len(admitted)
out["admission_unixes_tail"] = [
    r.get("updated_unix") or r.get("unix") for r in admitted[-8:]]
yields = [r for r in rows
          if str(r.get("kind") or "") == "deferred_replay_resource_yield"]
out["yields_total"] = len(yields)
out["yield_unixes_tail"] = [
    r.get("updated_unix") or r.get("unix") for r in yields[-5:]]

# Was the last-good guard ever advanced past this interval's start?
try:
    with open(f"{R}/brain/brain.last-good.json", "r", encoding="utf-8") as fh:
        guard = json.load(fh)
    out["last_good"] = {k: guard.get(k) for k in ("phase", "row", "updated_unix")}
except Exception as exc:
    out["last_good_error"] = f"{type(exc).__name__}: {exc}"

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
