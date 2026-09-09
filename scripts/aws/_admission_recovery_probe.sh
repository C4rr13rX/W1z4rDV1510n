python3 - <<'PY'
"""Has the composition fix actually cleared the gate, or only been deployed?

`deploy_is_not_load`: a binary copied to the host but never restarted runs the
old module. And a gate that passes once is not an admission -- the ledger's
`deferred_replay_admitted` count rising is. So this reads three independent
things and does not infer any of them from the others: which binary the
listening brain is executing and when it was built, what the newest enterprise
gate report says per suite, and whether the admission ledger has moved.
"""
import glob, json, os, subprocess, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

# ---- which build is actually serving
try:
    listener = subprocess.run(
        "ss -lntpH | grep 18095 | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2",
        shell=True, capture_output=True, text=True).stdout.strip()
    out["brain_pid"] = listener
    if listener:
        exe = os.path.realpath(f"/proc/{listener}/exe")
        out["brain_exe"] = exe
        out["brain_exe_mtime"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(exe)))
        out["brain_started_ago_s"] = round(
            time.time() - os.path.getmtime(f"/proc/{listener}"), 1)
except Exception as error:
    out["listener_error"] = str(error)[:160]

# ---- newest gate verdict, per suite
reports = sorted(glob.glob(os.path.join(R, "**", "*enterprise-gate*.json"),
                           recursive=True), key=os.path.getmtime)
if reports:
    newest = reports[-1]
    out["gate_report"] = os.path.basename(newest)
    out["gate_age_h"] = round((time.time() - os.path.getmtime(newest)) / 3600, 2)
    report = json.load(open(newest, encoding="utf-8"))
    out["gate"] = {k: report.get(k) for k in
                   ("passed", "passed_suites", "total_suites")}
    out["gate_failing"] = [row.get("name") for row in report.get("results") or []
                           if not row.get("passed")]

# ---- has anything actually been admitted since the fix landed
rows = []
for line in open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8"):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
admits = [r for r in rows if r.get("kind") == "deferred_replay_admitted"]
fails = [r for r in rows if r.get("kind") == "deferred_replay_failed"]
out["admitted_total"] = len(admits)
out["failed_total"] = len(fails)
if admits:
    out["last_admit_ago_h"] = round(
        (out["now"] - (admits[-1].get("updated_unix") or 0)) / 3600, 2)
    out["last_admit_interval"] = admits[-1].get("interval_id")
if fails:
    out["last_fail_ago_h"] = round(
        (out["now"] - (fails[-1].get("updated_unix") or 0)) / 3600, 2)

st = os.path.join(R, "deferred-replay-active.json")
if os.path.exists(st):
    out["status"] = json.load(open(st, encoding="utf-8"))

mem = {}
for line in open("/proc/meminfo"):
    key, _, rest = line.partition(":")
    mem[key] = int(rest.split()[0]) * 1024
out["available_gb"] = round(mem.get("MemAvailable", 0) / 2**30, 2)

print("PROBEJSON " + json.dumps(out))
PY
