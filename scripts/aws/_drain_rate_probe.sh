python3 - <<'PY'
"""Measure the CURRENT quarantine drain rate and the remaining work.

The 0.75 rows/s figure was averaged over a window that turned out to sit on a
74.8 h dead zone which ended ~28 h ago, so it describes a cleared stall rather
than today's throughput. Replay has been completing intervals every 2-3 h
since. This reads the outstanding interval ledger directly -- the same source
the curriculum counts derive from -- so the remaining rows and the honest ETA
come from state rather than from a stale average.
"""
import json, os, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
now = time.time()
out = {"now": now}

# ---- Outstanding intervals, by status. Last write wins per interval id.
state = {}
path = os.path.join(R, "curriculum-deferred-intervals.jsonl")
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("interval_id"):
        state[row["interval_id"]] = row

status_counts = collections.Counter()
rows_by_status = collections.Counter()
by_phase = collections.Counter()
for interval_id, row in state.items():
    status = str(row.get("status") or "?")
    status_counts[status] += 1
    try:
        parts = interval_id.split(":")
        width = int(parts[2]) - int(parts[1])
    except Exception:
        width = 0
    rows_by_status[status] += width
    if status == "deferred":
        by_phase[interval_id.split(":")[0]] += width

out["interval_status_counts"] = dict(status_counts)
out["rows_by_status"] = dict(rows_by_status)
out["outstanding_rows_by_phase"] = dict(by_phase.most_common(12))
out["outstanding_rows_total"] = rows_by_status.get("deferred", 0)

# ---- Resolution timestamps, to get a rate that is not contaminated by the
# dead zone: how many rows actually left "deferred" in the last N hours.
resolved = []
for line in open(path, encoding="utf-8"):
    try:
        row = json.loads(line)
    except Exception:
        continue
    when = row.get("updated_unix") or row.get("resolved_unix") or 0
    if when and str(row.get("status") or "") in ("resolved", "admitted", "retired"):
        try:
            parts = str(row.get("interval_id")).split(":")
            width = int(parts[2]) - int(parts[1])
        except Exception:
            width = 0
        resolved.append((when, width, row.get("interval_id"), row.get("status")))
resolved.sort()
for hours in (6, 12, 24, 48, 96):
    cut = now - hours * 3600
    rows = sum(w for t, w, _, _ in resolved if t >= cut)
    out[f"rows_resolved_last_{hours}h"] = rows
    out[f"rate_last_{hours}h_rows_per_s"] = round(rows / (hours * 3600), 3)
out["recent_resolutions"] = [
    {"h_ago": round((now - t) / 3600.0, 2), "rows": w, "interval": i, "status": s}
    for t, w, i, s in resolved[-12:]]

# ---- Live drain, for the burst rate to compare against.
prog = None
for name in os.listdir(R):
    if name.endswith(".progress.json") and "deferred-replay" in name:
        p = os.path.join(R, name)
        if prog is None or os.path.getmtime(p) > os.path.getmtime(prog):
            prog = p
if prog:
    a = json.load(open(prog, encoding="utf-8"))
    time.sleep(30)
    b = json.load(open(prog, encoding="utf-8"))
    out["live"] = {
        "rows_per_s": round(((b.get("durable_next_row") or 0)
                             - (a.get("durable_next_row") or 0)) / 30.0, 2),
        "durable_next_row": b.get("durable_next_row"),
        "accepted_episodes": b.get("accepted_episodes"),
    }

mem = {}
for line in open("/proc/meminfo"):
    key, _, rest = line.partition(":")
    mem[key] = int(rest.split()[0]) * 1024
out["available_gb"] = round(mem.get("MemAvailable", 0) / 2**30, 2)

print("PROBEJSON " + json.dumps(out))
PY
