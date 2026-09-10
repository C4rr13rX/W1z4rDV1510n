python3 - <<'PY'
"""Decide whether the go-systems quarantine replay RESUMED or restarted.

The watchdog fired `quarantine_ready` 129 s after a resource yield
(2.92 -> 14.65 GB) with the interval in `deferred_replay_training` at row
36384. `durable_next_row` rises whether or not the pass resumed, so
`accepted_episodes` is the discriminator: a restarted pass re-earns its
episodes from zero. The heartbeat block reported a NULL rate because it chose
the supervisor status file -- which carries no row -- so measure the replay
progress file directly, adaptively, until the row moves or the window ends.
"""
import glob, json, os, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

for name in ("deferred-replay-active.json", "curriculum-supervisor.status.json"):
    p = os.path.join(R, name)
    if os.path.exists(p):
        try:
            out[name] = {"age_s": round(time.time() - os.path.getmtime(p), 1),
                         "body": json.load(open(p, encoding="utf-8"))}
        except Exception as error:
            out[name] = {"err": str(error)[:120]}

prog = sorted(glob.glob(os.path.join(R, "deferred-replay-*.progress.json")),
              key=os.path.getmtime)
out["progress_files"] = [
    {"f": os.path.basename(p), "age_s": round(time.time() - os.path.getmtime(p), 1)}
    for p in prog[-4:]]
samples = []
if prog:
    live = prog[-1]
    out["progress_file"] = os.path.basename(live)
    deadline = time.time() + 150
    first = None
    while time.time() < deadline:
        try:
            d = json.load(open(live, encoding="utf-8"))
            s = {"t": round(time.time() - out["now"], 1),
                 "row": d.get("durable_next_row"),
                 "ram": d.get("ram_next_row"),
                 "acc": d.get("accepted_episodes"),
                 "batch": d.get("current_batch_size")}
        except Exception as error:
            s = {"t": round(time.time() - out["now"], 1), "err": str(error)[:80]}
        if not samples or s.get("row") != samples[-1].get("row"):
            samples.append(s)
            if first is None:
                first = s
            elif len(samples) >= 3:
                break
        time.sleep(5)
    if samples and samples[-1].get("t") != round(time.time() - out["now"], 1):
        samples.append({"t": round(time.time() - out["now"], 1)})
out["samples"] = samples
if len(samples) >= 2 and samples[0].get("row") and samples[-1].get("row"):
    dt = max(samples[-1]["t"] - samples[0]["t"], 1e-6)
    out["rows_per_s"] = round((samples[-1]["row"] - samples[0]["row"]) / dt, 3)

rows = []
try:
    for line in open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
except Exception as error:
    out["ledger_error"] = str(error)[:120]
rows.sort(key=lambda r: r.get("updated_unix") or 0)
out["ledger_tail"] = [
    {"kind": r.get("kind"), "ago_s": round(out["now"] - (r.get("updated_unix") or 0)),
     "interval": r.get("interval_id"), "reason": str(r.get("reason"))[:70]}
    for r in rows[-14:]]
day = [r for r in rows if (r.get("updated_unix") or 0) >= out["now"] - 86400]
out["kinds_24h"] = collections.Counter(str(r.get("kind")) for r in day).most_common(12)

mem = {}
for line in open("/proc/meminfo"):
    key, _, rest = line.partition(":")
    mem[key] = int(rest.split()[0]) * 1024
out["available_gb"] = round(mem.get("MemAvailable", 0) / 2**30, 2)
out["procs"] = os.popen("ps -eo pid,etimes,rss,comm --sort=-rss | head -7").read().splitlines()
print("PROBEJSON " + json.dumps(out))
PY
