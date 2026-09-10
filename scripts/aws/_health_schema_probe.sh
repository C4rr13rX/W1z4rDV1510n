python3 - <<'PY'
"""Read curriculum-health.jsonl's ACTUAL schema before bucketing it again.

The previous probe asked for `ev['event']` and got None on all 2704 records,
publishing `failures_since: 0`. CLAUDE.md's own rule: verify a pattern CAN be
non-zero before trusting its absence. So: dump the key universe first, then
bucket on keys that exist.

Also re-read the live row -- the block was 0.2 h from its gate.

Read-only.
"""
import json, os, time, collections, glob

R = "/srv/wizard/runtime/programming-integrated-20260713"
NOW = time.time()
out = {"now": NOW}

recs = []
with open(f"{R}/curriculum-health.jsonl", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
out["total"] = len(recs)

keys = collections.Counter()
for r in recs:
    if isinstance(r, dict):
        keys.update(r.keys())
out["key_universe"] = keys.most_common(30)
out["sample_first"] = recs[0] if recs else None
out["sample_last3"] = recs[-3:]

# whichever key names the record type, find it empirically
for cand in ("event", "kind", "type", "state", "status", "record", "name"):
    if keys.get(cand):
        vals = collections.Counter(str(r.get(cand)) for r in recs if isinstance(r, dict))
        out[f"values_of_{cand}"] = vals.most_common(30)

# --- live row ---------------------------------------------------------------
prog = sorted(glob.glob(f"{R}/deferred-replay-*.progress.json"),
              key=os.path.getmtime)
if prog:
    p = prog[-1]
    a = json.load(open(p, encoding="utf-8"))
    time.sleep(6)
    b = json.load(open(p, encoding="utf-8"))
    out["live"] = {
        "file": os.path.basename(p),
        "age_s": round(NOW - os.path.getmtime(p), 1),
        "row_a": a.get("durable_next_row"), "row_b": b.get("durable_next_row"),
        "ram_next_row": b.get("ram_next_row"),
        "accepted_a": a.get("accepted_episodes"),
        "accepted_b": b.get("accepted_episodes"),
    }

st = json.load(open(f"{R}/curriculum-supervisor.status.json", encoding="utf-8"))
out["status"] = {k: st.get(k) for k in
                 ("state", "phase", "passed", "trained_rows", "updated_unix",
                  "reason", "detail", "interval_id")}
out["status_age_s"] = round(NOW - float(st.get("updated_unix") or NOW), 1)

with open("/proc/meminfo") as fh:
    mi = {l.split(":")[0]: l.split()[1] for l in fh if ":" in l}
out["mem_available_gb"] = round(int(mi["MemAvailable"]) / 1048576.0, 2)

print("PROBE_JSON " + json.dumps(out, default=str)[:12000])
PY
