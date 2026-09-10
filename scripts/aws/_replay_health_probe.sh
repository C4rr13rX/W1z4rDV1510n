python3 - <<'PY'
"""Decide whether the quarantine replay is draining or only appears alive.

The watchdog fired on `quarantine_ready` with `forward_remaining_rows` 0 and a
resource yield 712 s before the reading, so the interesting question is not
"is a process up" -- it is whether the pass RESUMED after that yield or
restarted the interval. `accepted_episodes` is the discriminator:
`durable_next_row` rises either way, but a restarted pass re-earns its
episodes from zero (see the resume-discriminator lesson).
"""
import glob, json, os, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

st = os.path.join(R, "deferred-replay-active.json")
if os.path.exists(st):
    out["status"] = json.load(open(st, encoding="utf-8"))
    out["status_age_s"] = round(time.time() - os.path.getmtime(st), 1)

prog = sorted(glob.glob(os.path.join(R, "deferred-replay-*.progress.json")),
              key=os.path.getmtime)
out["progress_files"] = len(prog)
samples = []
if prog:
    out["progress_file"] = os.path.basename(prog[-1])
    for _ in range(6):
        try:
            d = json.load(open(prog[-1], encoding="utf-8"))
            samples.append({
                "t": round(time.time() - out["now"], 1),
                "row": d.get("durable_next_row"),
                "acc": d.get("accepted_episodes"),
                "batch": d.get("current_batch_size"),
                "ema": d.get("batch_seconds_ema"),
            })
        except Exception as error:
            samples.append({"err": str(error)[:80]})
        time.sleep(20)
out["samples"] = samples
if len(samples) >= 2 and samples[0].get("row") and samples[-1].get("row"):
    dt = samples[-1]["t"] - samples[0]["t"]
    out["rows_per_s"] = round((samples[-1]["row"] - samples[0]["row"]) / dt, 2)
    out["acc_per_s"] = round((samples[-1]["acc"] - samples[0]["acc"]) / dt, 2)

# Ledger tail: what has actually happened since the yield.
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
     "interval": r.get("interval_id"), "phase": r.get("phase")}
    for r in rows[-12:]
]
day = [r for r in rows if (r.get("updated_unix") or 0) >= out["now"] - 86400]
out["kinds_24h"] = collections.Counter(str(r.get("kind")) for r in day).most_common(14)

mem = {}
for line in open("/proc/meminfo"):
    key, _, rest = line.partition(":")
    mem[key] = int(rest.split()[0]) * 1024
out["available_gb"] = round(mem.get("MemAvailable", 0) / 2**30, 2)
out["procs"] = os.popen(
    "ps -eo pid,etimes,rss,comm --sort=-rss | head -6").read().splitlines()
print("PROBEJSON " + json.dumps(out))
PY
