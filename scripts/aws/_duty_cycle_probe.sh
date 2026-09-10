python3 - <<'PY'
"""Split the replay duty-cycle loss into stall versus rework.

Established: the worker drains at ~9 rows/s when observed, the 74 h average is
0.75 rows/s, and the brain burns 11.6 GB in a 25 min cycle before a forced
recycle. Two candidate causes remain and they need different fixes -- a
hydration stall after each recycle (fix: make the recycle cheaper or rarer)
versus rows re-run after a rollback (fix: keep the resume marker durable).
This reads the ledger with its real key, `kind`, and reconstructs per-interval
lifecycles so the two are separable rather than argued.
"""
import json, os, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
now = time.time()
out = {"now": now}

rows = []
for line in open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8"):
    try:
        row = json.loads(line)
    except Exception:
        continue
    when = row.get("updated_unix") or 0
    if when:
        rows.append((when, str(row.get("kind") or ""), row))
rows.sort(key=lambda r: r[0])

WINDOW = 74.21 * 3600
recent = [r for r in rows if r[0] >= now - WINDOW]
out["kinds_in_74h"] = collections.Counter(r[1] for r in recent).most_common(30)
out["kinds_all_time"] = collections.Counter(r[1] for r in rows).most_common(12)

# ---- Interval lifecycles. `admitted` marks a drained interval; anything
# between the first sighting and that admission is time the interval cost.
life = collections.defaultdict(
    lambda: {"first": None, "last": None, "kinds": collections.Counter()})
for when, kind, row in rows:
    interval = row.get("interval_id") or ""
    if not interval or interval.count(":") < 2:
        continue
    entry = life[interval]
    entry["first"] = when if entry["first"] is None else min(entry["first"], when)
    entry["last"] = max(entry["last"] or 0, when)
    entry["kinds"][kind] += 1

report = []
for interval, entry in life.items():
    try:
        start, end = int(interval.split(":")[1]), int(interval.split(":")[2])
    except Exception:
        continue
    span_rows = end - start
    hours = (entry["last"] - entry["first"]) / 3600.0
    report.append({
        "interval": interval,
        "rows": span_rows,
        "wall_h": round(hours, 2),
        "ended_h_ago": round((now - entry["last"]) / 3600.0, 2),
        "admitted": entry["kinds"].get("deferred_replay_admitted", 0),
        "failed": entry["kinds"].get("deferred_replay_failed", 0),
        "yields": entry["kinds"].get("deferred_replay_resource_yield", 0),
        "rows_per_s": round(span_rows / (hours * 3600), 3) if hours > 0 else None,
    })
report.sort(key=lambda r: r["ended_h_ago"])
out["interval_lifecycles"] = report[:14]
out["interval_count"] = len(report)
admitted = [r for r in report if r["admitted"]]
out["admitted_count"] = len(admitted)
if admitted:
    out["admitted_median_wall_h"] = sorted(r["wall_h"] for r in admitted)[len(admitted) // 2]
    out["admitted_median_yields"] = sorted(r["yields"] for r in admitted)[len(admitted) // 2]

# ---- Recycle cost. Sample the live progress file across a recycle boundary:
# if the row counter freezes for minutes after a recycle, the loss is stall.
recycles = [r[0] for r in rows if r[1] == "settled_node_memory_recycle"]
gaps = [round(b - a, 1) for a, b in zip(recycles, recycles[1:]) if b - a < 20000]
if gaps:
    gaps_sorted = sorted(gaps)
    out["recycle_period_s"] = {
        "count": len(gaps),
        "median": gaps_sorted[len(gaps_sorted) // 2],
        "min": gaps_sorted[0],
        "max": gaps_sorted[-1],
    }
out["recycles_in_74h"] = sum(1 for r in recent if r[1] == "settled_node_memory_recycle")

# ---- Direct drain sampling over 3 min, to see whether the rate holds or
# collapses inside a single cycle.
prog = None
for name in os.listdir(R):
    if name.endswith(".progress.json") and "deferred-replay" in name:
        path = os.path.join(R, name)
        if prog is None or os.path.getmtime(path) > os.path.getmtime(prog):
            prog = path
samples = []
for _ in range(10):
    try:
        row = json.load(open(prog, encoding="utf-8"))
        samples.append({"t": round(time.time() - now, 1),
                        "row": row.get("durable_next_row"),
                        "acc": row.get("accepted_episodes")})
    except Exception as error:
        samples.append({"err": str(error)[:60]})
    time.sleep(18)
out["drain_samples"] = samples
out["progress_file"] = os.path.basename(prog) if prog else None
if len(samples) >= 2 and samples[0].get("row") and samples[-1].get("row"):
    dt = samples[-1]["t"] - samples[0]["t"]
    out["sampled_rows_per_s"] = round((samples[-1]["row"] - samples[0]["row"]) / dt, 2)

mem = {}
for line in open("/proc/meminfo"):
    key, _, rest = line.partition(":")
    mem[key] = int(rest.split()[0]) * 1024
out["available_gb"] = round(mem.get("MemAvailable", 0) / 2**30, 2)

print("PROBEJSON " + json.dumps(out))
PY
