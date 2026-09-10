python3 - <<'PY'
"""Confirm the post-fix state: replay converging, no new false quarantine.

Reads the replay progress file directly rather than the heartbeat, because
during a replay the supervisor status file is often the freshest writer and
carries no row -- which is why the watchdog reported `rows_per_second: null`.
"""
import glob, json, os, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
out = {"now": time.time()}

prog = sorted(glob.glob(os.path.join(R, "deferred-replay-*.progress.json")),
              key=os.path.getmtime)[-1]
samples = []
deadline = time.time() + 130
while time.time() < deadline and len(samples) < 3:
    d = json.load(open(prog, encoding="utf-8"))
    s = {"t": round(time.time() - out["now"], 1), "row": d.get("durable_next_row"),
         "ram": d.get("ram_next_row"), "acc": d.get("accepted_episodes")}
    if not samples or s["row"] != samples[-1]["row"]:
        samples.append(s)
    time.sleep(4)
out["samples"] = samples
if len(samples) >= 2:
    dt = max(samples[-1]["t"] - samples[0]["t"], 1e-6)
    out["rows_per_s"] = round((samples[-1]["row"] - samples[0]["row"]) / dt, 2)
    out["rollback_exposure_rows"] = samples[-1]["ram"] - samples[-1]["row"]
    out["remaining_rows"] = 131072 - samples[-1]["row"]
    if out["rows_per_s"] > 0:
        out["eta_hours_instantaneous"] = round(out["remaining_rows"] / out["rows_per_s"] / 3600, 2)

st = os.path.join(R, "curriculum-supervisor.status.json")
out["status"] = {"age_s": round(time.time() - os.path.getmtime(st), 1),
                 "body": json.load(open(st, encoding="utf-8"))}

rows = []
for line in open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8"):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
rows.sort(key=lambda r: r.get("updated_unix") or 0)
DEPLOY = 1789043364  # SSM deploy completed
out["events_since_deploy"] = [
    {"kind": r.get("kind"), "ago_s": round(out["now"] - (r.get("updated_unix") or 0)),
     "err": str(r.get("error") or "")[:100]}
    for r in rows if (r.get("updated_unix") or 0) >= DEPLOY]
out["midphase_failed_since_deploy"] = sum(
    1 for r in rows if (r.get("updated_unix") or 0) >= DEPLOY
    and str(r.get("kind")) == "midphase_gate_failed")

# The deployed files are the ones the next gate will spawn.
import hashlib
out["deployed"] = {}
for rel in ("scripts/programming_integrated_retention.py",
            "scripts/programming_debug_benchmark.py"):
    p = os.path.join(P, rel)
    out["deployed"][os.path.basename(rel)] = {
        "sha": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16],
        "uid": os.stat(p).st_uid,
        "has_fix": "run_evaluator" in open(p, encoding="utf-8").read()
                   or "PREDICT_TIMEOUT_SECONDS" in open(p, encoding="utf-8").read()}

out["mem_available_gb"] = round(int([l for l in open("/proc/meminfo")
    if l.startswith("MemAvailable")][0].split()[1]) / 2**20, 2)
print("PROBEJSON " + json.dumps(out))
PY
