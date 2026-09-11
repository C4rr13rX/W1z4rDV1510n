python3 - <<'PY'
"""Do the intervals that ADMIT differ in size from the one that cannot?

Two intervals admitted 16.5 h and 13.6 h ago, so the gate, the worker and the
admission path all work. The interval running now has had 13.6 h, 361 resource
yields and 14 failures and has admitted nothing. If the admitted ones are
materially smaller, the binding constraint is the work unit against the disk
window and the repair is to resize it; if they are the same size, it is not,
and resizing would be another inert fix.

Also reports WHERE this interval's failures died. `deferred_replay_failed`
before the gate means the pass never finished (the window ran out); at the gate
means a real verdict. That split decides resize-vs-repair, and it is the same
distinction `replay_failures_before_gate` / `_at_gate` publish in aggregate --
here it is asked of one interval.
"""
import collections
import json
import re
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def load(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as exc:
        out.setdefault("errors", []).append(f"{path}: {exc}")
    return rows


health = load(f"{R}/curriculum-health.jsonl")
deferred = load(f"{R}/curriculum-deferred-intervals.jsonl")
out["health_records"] = len(health)
out["deferred_records"] = len(deferred)


def span(interval_id):
    match = re.match(r"^(.*):(\d+):(\d+)$", str(interval_id or ""))
    return (int(match.group(3)) - int(match.group(2))) if match else None


# ---- the intervals that admitted, newest first ---------------------------
admitted = [r for r in health
            if str(r.get("kind") or "") == "deferred_replay_admitted"]
out["admitted"] = [
    {"interval_id": r.get("interval_id"),
     "rows": span(r.get("interval_id")),
     "hours_ago": round((out["now"] - float(r.get("updated_unix") or r.get("unix") or 0)) / 3600.0, 2),
     "reason": str(r.get("reason") or r.get("resolution") or "")[:120]}
    for r in admitted[-10:]
]

# ---- the interval running now --------------------------------------------
active = {}
try:
    with open(f"{R}/deferred-replay-active.json", "r", encoding="utf-8") as fh:
        active = json.load(fh)
except Exception as exc:
    out["active_error"] = str(exc)
current = str(active.get("interval_id") or "")
out["current_interval"] = {"interval_id": current, "rows": span(current),
                           "state": active.get("state"),
                           "created_hours_ago": round(
                               (out["now"] - float(active.get("created_unix") or 0)) / 3600.0, 2)}

# ---- where did this interval's failures die? -----------------------------
mine = [r for r in health if str(r.get("interval_id") or "") == current]
failures = [r for r in mine if str(r.get("kind") or "") == "deferred_replay_failed"]
out["failure_count"] = len(failures)
out["failures"] = [
    {"hours_ago": round((out["now"] - float(r.get("updated_unix") or r.get("unix") or 0)) / 3600.0, 2),
     "stage": r.get("stage") or r.get("failure_stage"),
     "reached_gate": bool(r.get("gate_report") or r.get("gate_results")
                          or r.get("enterprise") or r.get("passed") is not None),
     "reason": str(r.get("reason") or "")[:160],
     "error_head": str(r.get("error") or "")[:220],
     "error_tail": str(r.get("error") or "")[-220:]}
    for r in failures[-14:]
]
out["failure_reasons"] = dict(collections.Counter(
    str(r.get("reason") or "?")[:80] for r in failures))

# ---- the whole deferred population, by size ------------------------------
sizes = collections.Counter()
unresolved = []
for record in deferred:
    interval_id = str(record.get("interval_id") or "")
    rows = span(interval_id)
    if rows is None:
        continue
    sizes[rows] += 1
    if str(record.get("status") or "") == "deferred":
        unresolved.append((interval_id, rows))
resolved_ids = {str(r.get("interval_id")) for r in deferred
                if str(r.get("status") or "") == "resolved"}
still = [(i, n) for i, n in unresolved if i not in resolved_ids]
out["deferred_size_histogram"] = dict(sorted(sizes.items()))
out["unresolved_count"] = len(still)
out["unresolved_rows_total"] = sum(n for _, n in still)
out["unresolved_largest"] = sorted(still, key=lambda p: -p[1])[:10]
out["unresolved_smallest"] = sorted(still, key=lambda p: p[1])[:10]

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
