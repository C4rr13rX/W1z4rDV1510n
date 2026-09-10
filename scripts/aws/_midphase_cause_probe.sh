python3 - <<'PY'
"""Bucket every midphase_gate_failed by its ACTUAL error shape.

`last_failure` names one event; the population decides whether the fix is
worth deploying. The hypothesis under test: a `GateCommandFailure` whose
stderr ends in `CalledProcessError ... programming_debug_benchmark.py` carries
no transient marker, so `transient_gate_failure()` scores an infrastructure
crash as a semantic quarantine. Count how many of the 45 look like that
versus a genuine retention regression.
"""
import glob, json, os, re, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

rows = []
for line in open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8"):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass

MARKERS = ("timeouterror", "timed out", "filenotfounderror",
           "no such file or directory", "connectionrefusederror",
           "connectionreseterror", "remotedisconnected", "connection aborted",
           "connection refused", "urlerror", '"infrastructure_only_failure": true')

def bucket(err: str) -> str:
    low = (err or "").casefold()
    if any(m in low for m in MARKERS):
        return "would_have_been_transient"
    m = re.search(r"returned non-zero exit status \d+", low)
    if "calledprocesserror" in low and m:
        child = re.findall(r"scripts/(\w+)\.py", err or "")
        return "swallowed_child_exit:" + (child[-1] if child else "?")
    if "gate command failed" in low:
        return "gate_command_failed_other"
    return "other:" + low[:60]

mid = [r for r in rows if str(r.get("kind")) == "midphase_gate_failed"]
out["midphase_total"] = len(mid)
out["midphase_buckets"] = collections.Counter(
    bucket(str(r.get("error") or r.get("reason") or "")) for r in mid).most_common()
out["midphase_recent"] = [
    {"ago_h": round((out["now"] - (r.get("updated_unix") or 0)) / 3600, 1),
     "phase": r.get("phase"), "rows": r.get("trained_rows"),
     "err": str(r.get("error") or "")[:400]}
    for r in mid[-4:]]

for kind in ("midphase_gate_infrastructure_paused", "midphase_gate_infrastructure_retry"):
    out[kind] = sum(1 for r in rows if str(r.get("kind")) == kind)

# Does an inner cause survive anywhere on disk?
out["debug_report"] = None
p = os.path.join(R, "integrated_debug.json")
if os.path.exists(p):
    try:
        d = json.load(open(p, encoding="utf-8"))
        out["debug_report"] = {"age_h": round((time.time() - os.path.getmtime(p)) / 3600, 1),
                               "summary": {k: {"passed": v.get("passed"), "total": v.get("total")}
                                           for k, v in d.items() if isinstance(v, dict)}}
    except Exception as error:
        out["debug_report"] = {"err": str(error)[:120]}

ev = sorted(glob.glob(os.path.join(R, "deferred/*/evidence/candidate-row-*")),
            key=lambda p: os.path.getmtime(p))
out["evidence_dirs"] = [{"d": p.replace(R + "/", ""),
                         "age_h": round((time.time() - os.path.getmtime(p)) / 3600, 1),
                         "files": sorted(os.listdir(p))[:12]} for p in ev[-2:]]

# The stderr log the watchdog's last_failure names.
log = os.path.join(R, "deferred-replay-8f4a439a7fc7a772.stderr.log")
if os.path.exists(log):
    body = open(log, encoding="utf-8", errors="replace").read()
    out["named_worker_stderr"] = {"bytes": len(body), "tail": body[-900:]}
print("PROBEJSON " + json.dumps(out))
PY
