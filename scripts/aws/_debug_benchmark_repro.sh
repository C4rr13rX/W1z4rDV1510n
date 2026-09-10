python3 - <<'PY'
"""Name the swallowed cause of the midphase gate crash.

`debug_eval` runs `programming_debug_benchmark.py` with capture_output=True and
check=True, so the child's traceback is captured into CalledProcessError and
discarded -- which is why `transient_gate_failure()` sees no marker and scores
an infrastructure crash as a semantic quarantine. The benchmark only issues
/brain/predict/multi and /brain/repair/predict, so it is read-only against the
brain; run it directly and keep the stderr the gate threw away.
"""
import json, os, subprocess, time

P = "/srv/wizard/project"
out = {"now": time.time()}
target = "/tmp/_repro_integrated_debug.json"
if os.path.exists(target):
    os.unlink(target)

run = subprocess.run(
    ["/usr/bin/python3", "scripts/programming_debug_benchmark.py",
     "--endpoint", "http://127.0.0.1:18095", "--output", target],
    cwd=P, capture_output=True, text=True, timeout=900)
out["returncode"] = run.returncode
out["stdout_tail"] = run.stdout[-1500:]
out["stderr_tail"] = run.stderr[-2500:]
out["wrote_report"] = os.path.exists(target)
if os.path.exists(target):
    body = json.load(open(target, encoding="utf-8"))
    out["report"] = {k: [v.get("passed"), v.get("total")]
                     for k, v in body.items() if isinstance(v, dict)}

# Which markers would transient_gate_failure() have matched?
MARKERS = ("timeouterror", "timed out", "filenotfounderror",
           "no such file or directory", "connectionrefusederror",
           "connectionreseterror", "remotedisconnected", "connection aborted",
           "connection refused", "urlerror", '"infrastructure_only_failure": true')
low = (run.stdout + "\n" + run.stderr).casefold()
out["child_markers_present"] = [m for m in MARKERS if m in low]
swallowed = f"gate command failed (1): ...\nstdout: \nstderr: ...CalledProcessError..."
out["parent_markers_present"] = [m for m in MARKERS if m in swallowed.casefold()]
print("PROBEJSON " + json.dumps(out))
PY
