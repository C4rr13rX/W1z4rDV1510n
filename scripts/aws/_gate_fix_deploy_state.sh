python3 - <<'PY'
"""Is commit c343443's evaluator fix actually ON the host, and is it LOADED?

"Deploy is not load": a fix copied but never loaded ran the old code for 96 h
once. The midphase gate spawns a fresh `python scripts/...` per invocation,
so for THESE files being on disk is being loaded -- but only if the bytes are
right. Compare sha256 against the local repo, and prove the deployed source
has no check=True in its evaluator runner.

The live block is minutes from its midphase gate, and that gate is where two
go-systems blocks were already falsely quarantined.

Read-only.
"""
import hashlib, json, os, re, subprocess, time

P = "/srv/wizard/project"
R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

for name in ("programming_integrated_retention.py", "programming_debug_benchmark.py"):
    path = f"{P}/scripts/{name}"
    rec = {}
    try:
        data = open(path, "rb").read()
        rec["sha16"] = hashlib.sha256(data).hexdigest()[:16]
        rec["bytes"] = len(data)
        rec["mtime_h_ago"] = round((time.time() - os.path.getmtime(path)) / 3600.0, 2)
        rec["inode"] = os.stat(path).st_ino
        text = data.decode("utf-8", "replace")
        rec["has_check_true"] = bool(re.search(r"check\s*=\s*True", text))
        rec["has_run_evaluator"] = "def run_evaluator" in text
        rec["has_unlink"] = "unlink(missing_ok=True)" in text
        rec["has_EvaluatorUnavailable"] = "EvaluatorUnavailable" in text
    except Exception as e:
        rec["_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    out[name] = rec

# any backup dirs the deploy script would have created
try:
    out["deploy_backups"] = sorted(
        d for d in os.listdir(P) if d.startswith(".deploy-backup-"))[-5:]
except Exception as e:
    out["deploy_backups"] = f"<{e}>"

# git state of the deployed tree, if it is a checkout
try:
    out["host_git"] = subprocess.run(
        f"cd {P} && git log --oneline -3 2>&1 | head -3", shell=True,
        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:
    out["host_git"] = f"<{e}>"

# live row, again -- how close is the gate?
import glob
prog = sorted(glob.glob(f"{R}/deferred-replay-*.progress.json"), key=os.path.getmtime)
if prog:
    a = json.load(open(prog[-1], encoding="utf-8"))
    time.sleep(5)
    b = json.load(open(prog[-1], encoding="utf-8"))
    ra, rb = a.get("durable_next_row"), b.get("durable_next_row")
    rate = ((rb - ra) / 5.0) if (ra is not None and rb is not None) else None
    out["live"] = {"row": rb, "ram": b.get("ram_next_row"), "rate": rate,
                   "remaining": (131072 - rb) if rb is not None else None,
                   "eta_min": round((131072 - rb) / rate / 60.0, 1)
                   if rate else None}

with open("/proc/meminfo") as fh:
    mi = {l.split(":")[0]: l.split()[1] for l in fh if ":" in l}
out["mem_available_gb"] = round(int(mi["MemAvailable"]) / 1048576.0, 2)

print("PROBE_JSON " + json.dumps(out, default=str))
PY
