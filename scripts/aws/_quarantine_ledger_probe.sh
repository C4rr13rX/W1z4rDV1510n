python3 - <<'PY'
"""Enumerate every unresolved quarantine obligation and the replay in flight.

The gate-classification fix is deployed, so the question is now what still
owes resolution and whether the running replay is the mechanism that will
deliver it. Do not mutate: two of these intervals were quarantined by a
crash, and the replay transaction for one of them is live.
"""
import glob, json, os, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

# Every deferred interval directory and its resolution state.
intervals = []
for d in sorted(glob.glob(os.path.join(R, "deferred/*")), key=os.path.getmtime):
    if not os.path.isdir(d):
        continue
    entry = {"id": os.path.basename(d),
             "age_h": round((time.time() - os.path.getmtime(d)) / 3600, 1),
             "files": sorted(f for f in os.listdir(d) if f.endswith(".json"))[:8]}
    for name in ("interval.json", "resolution.json", "state.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                entry[name] = json.load(open(p, encoding="utf-8"))
            except Exception as error:
                entry[name] = str(error)[:80]
    intervals.append(entry)
out["deferred_dirs"] = intervals

for name in ("deferred-replay-active.json", "curriculum-supervisor.status.json",
             "deferred-intervals.json", "deferred-queue.json"):
    p = os.path.join(R, name)
    if os.path.exists(p):
        try:
            out[name] = {"age_s": round(time.time() - os.path.getmtime(p), 1),
                         "body": json.load(open(p, encoding="utf-8"))}
        except Exception as error:
            out[name] = {"err": str(error)[:120]}

prog = sorted(glob.glob(os.path.join(R, "deferred-replay-*.progress.json")),
              key=os.path.getmtime)
if prog:
    d = json.load(open(prog[-1], encoding="utf-8"))
    out["live_progress"] = {"f": os.path.basename(prog[-1]),
                            "age_s": round(time.time() - os.path.getmtime(prog[-1]), 1),
                            "durable_next_row": d.get("durable_next_row"),
                            "ram_next_row": d.get("ram_next_row"),
                            "accepted_episodes": d.get("accepted_episodes")}

rows = []
for line in open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8"):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
rows.sort(key=lambda r: r.get("updated_unix") or 0)
out["since_deploy"] = [
    {"kind": r.get("kind"), "ago_s": round(out["now"] - (r.get("updated_unix") or 0)),
     "phase": r.get("phase"), "err": str(r.get("error") or "")[:120]}
    for r in rows if (r.get("updated_unix") or 0) > out["now"] - 2400]
out["resolved_total"] = sum(1 for r in rows if str(r.get("kind")) == "deferred_replay_admitted")
out["mem_gb"] = round(int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 2**20, 2)
print("PROBEJSON " + json.dumps(out)[:12000])
PY
