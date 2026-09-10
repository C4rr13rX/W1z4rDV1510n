python3 - <<'PY'
"""Is the code that will judge the imminent gate the code that was FIXED?

`go-systems:0:131072` is replaying at ~13 rows/s with ~46k rows left, so it
reaches its midphase gate within the hour. It was quarantined 8 h ago by the
exact defect commit c343443 repairs: `debug_eval` ran its child
captured-and-checked, so a child crash arrived as "returned non-zero exit
status 1" -- a string with the classifier's marker deleted -- and 131,072 rows
were quarantined for what may have been a client timeout.

A fix that is committed, or even copied, is not a fix that RUNS
(`deploy_is_not_load`: a repair copied but never loaded ran the old code for
96 h). So compare the deployed files against the repaired ones by content, not
by mtime, and say which symbols are actually present.

Second question, same class: every one of the 65 `worker_exit` failures in the
ledger is <=137 characters -- the stderr PATH and nothing else -- while the
file it names holds a 1116-byte traceback. `replay_worker_failure` has
appended that tail inline since 44e496b (2026-09-09). Either the deployed
supervisor predates it or `replay_worker_stderr_tail` returns empty. Check
which.

Read-only: hashes and greps. No evaluator is run, so nothing here can observe,
tick, or write to the brain.
"""
import hashlib
import json
import os
import subprocess
import time

P = "/srv/wizard/project"
R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}


def sh(command, timeout=60):
    try:
        return subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout).stdout.strip()
    except Exception as error:
        return f"<{type(error).__name__}: {str(error)[:80]}>"


def survey(relative, symbols):
    path = os.path.join(P, relative)
    if not os.path.exists(path):
        return {"missing": True}
    body = open(path, encoding="utf-8", errors="replace").read()
    return {
        "sha256": hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:16],
        "bytes": len(body),
        "age_h": round((time.time() - os.path.getmtime(path)) / 3600.0, 2),
        "symbols": {name: (name in body) for name in symbols},
    }


out["integrated_retention"] = survey(
    "scripts/programming_integrated_retention.py",
    ["EvaluatorUnavailable", "run_evaluator", "infrastructure_only_failure",
     "check=True", "unlink"])
out["debug_benchmark"] = survey(
    "scripts/programming_debug_benchmark.py",
    ["return 0", "unlink", "infrastructure"])
out["supervisor"] = survey(
    "scripts/programming_curriculum_supervisor.py",
    ["replay_worker_stderr_tail", "replay_worker_failure", "stderr_mark",
     "MAX_BARREN_REPLAY_YIELDS"])
out["enterprise_retention"] = survey(
    "scripts/programming_enterprise_retention.py",
    ["infrastructure_only_failure", "unlink"])

# The registry file whose category killed 12 workers 15.5 h ago. Is it now
# valid, and does the real loader accept the WHOLE directory?
registry = f"{P}/tools/training_standard/registry/go_systems_001.toml"
if os.path.exists(registry):
    out["go_registry"] = {
        "age_h": round((time.time() - os.path.getmtime(registry)) / 3600.0, 2),
        "body": open(registry, encoding="utf-8").read()[:600],
    }
else:
    out["go_registry"] = {"missing": True}

out["registry_loads"] = sh(
    f"cd {P} && python3 -c \""
    "import sys; sys.path.insert(0,'.');"
    "from tools.training_standard import schema, runner;"
    "r=schema.load_registry(runner.REGISTRY_DIR);"
    "print('OK', len(r))\" 2>&1 | tail -3")

# Does a worker-exit tail actually survive the supervisor's own helper?
out["tail_helper_selftest"] = sh(
    f"cd {P} && python3 -c \""
    "import sys,pathlib; sys.path.insert(0,'.');"
    "import importlib.util as u;"
    "s=u.spec_from_file_location('sup','scripts/programming_curriculum_supervisor.py');"
    "m=u.module_from_spec(s); s.loader.exec_module(m);"
    f"t=m.replay_worker_stderr_tail(pathlib.Path('{R}/deferred-replay-8f4a439a7fc7a772.stderr.log'), 0);"
    "print('TAILLEN', len(t)); print(t[-160:])\" 2>&1 | tail -6")

# The debug benchmark writes to a FIXED path. A stale copy reads as a pass
# (measured once at 769.7 h old), so age is part of the verdict.
for name in ("integrated_debug.json", "go-systems.enterprise-gate.json",
             "_pretest_polyglot.json", "polyglot.json"):
    path = os.path.join(R, name)
    if os.path.exists(path):
        out.setdefault("gate_artifacts", {})[name] = {
            "age_h": round((time.time() - os.path.getmtime(path)) / 3600.0, 2),
            "bytes": os.path.getsize(path),
        }

# What the supervisor is doing right now, and how close the gate is.
try:
    resume = json.load(open(f"{R}/deferred-replay-b01232b593532da2.resume.json"))
    out["rows_to_gate"] = int(resume["end_row"]) - int(resume["durable_next_row"])
    out["resume_age_s"] = round(time.time() - float(resume["updated_unix"]), 1)
except Exception as error:
    out["resume_error"] = str(error)[:150]

out["deployed_commit"] = sh(f"cd {P} && git rev-parse --short HEAD 2>&1 | tail -1")
out["deployed_dirty"] = sh(f"cd {P} && git status --porcelain 2>&1 | head -12")
out["mem"] = sh("free -g | head -2")

print("PROBEJSON " + json.dumps(out))
PY
