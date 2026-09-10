python3 - <<'PY'
"""Prove or disprove that the two go-systems quarantines are FALSE.

Both scored `midphase_gate_failed` with a swallowed
`programming_debug_benchmark.py` exit 1. That script returns 0 unconditionally,
so exit 1 cannot be a verdict -- but that is an argument from source, and the
contract says measure. If the evidence `integrated_debug.json` beside each
failed candidate is complete and passing, the gate crashed AFTER observing a
healthy brain and the quarantine is false.

Also timestamp the registry SchemaError named in `last_failure` against the
current registry and the running worker, per the ledger-predates-process rule.
"""
import glob, json, os, subprocess, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
out = {"now": time.time()}

cands = []
for d in sorted(glob.glob(os.path.join(R, "deferred/*/evidence/candidate-row-*")),
                key=os.path.getmtime)[-2:]:
    entry = {"dir": d.replace(R + "/", ""),
             "age_h": round((time.time() - os.path.getmtime(d)) / 3600, 1)}
    for name in ("integrated_debug.json", "enterprise.json", "polyglot.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            entry[name] = "absent"
            continue
        try:
            body = json.load(open(p, encoding="utf-8"))
        except Exception as error:
            entry[name] = {"err": str(error)[:100]}
            continue
        entry[name] = {"age_h": round((time.time() - os.path.getmtime(p)) / 3600, 1)}
        if name == "integrated_debug.json":
            entry[name]["groups"] = {k: [v.get("passed"), v.get("total")]
                                     for k, v in body.items() if isinstance(v, dict)}
        elif name == "polyglot.json":
            res = body.get("results") or []
            entry[name]["cases"] = [[r.get("name"), r.get("executes")] for r in res][:14]
        else:
            res = body.get("results") or []
            entry[name]["suites"] = [[r.get("name"), r.get("passed")] for r in res][:14]
    # The foundation report the gate was writing when it died.
    row = entry["dir"].rsplit("-", 1)[-1]
    f = os.path.join(R, f"go-systems.row-{row}.foundation.json")
    entry["foundation_output"] = ("absent" if not os.path.exists(f) else
        {"age_h": round((time.time() - os.path.getmtime(f)) / 3600, 1),
         "bytes": os.path.getsize(f)})
    cands.append(entry)
out["candidates"] = cands

# Registry: is the SchemaError current or historical?
reg = os.path.join(P, "tools/training_standard/registry/go_systems_001.toml")
if os.path.exists(reg):
    out["registry"] = {"age_h": round((time.time() - os.path.getmtime(reg)) / 3600, 1),
                       "body": open(reg, encoding="utf-8").read()[:600]}
log = os.path.join(R, "deferred-replay-8f4a439a7fc7a772.stderr.log")
if os.path.exists(log):
    out["named_stderr_age_h"] = round((time.time() - os.path.getmtime(log)) / 3600, 1)
out["registry_loads_now"] = subprocess.run(
    ["/usr/bin/python3", "-c",
     "import sys; sys.path.insert(0,'/srv/wizard/project');"
     "from tools.training_standard import runner, schema;"
     "r=schema.load_registry(runner.REGISTRY_DIR);"
     "print('OK', len(r))"],
    cwd=P, capture_output=True, text=True).stdout.strip()[:200] or "FAILED"

# Live worker generation, so the ledger can be timestamped against it.
ps = os.popen("ps -eo pid,etimes,args --sort=-etimes | grep -i 'drive_corpora_brain\|curriculum_supervisor' | grep -v grep").read()
out["procs"] = [l.strip()[:150] for l in ps.splitlines()]
print("PROBEJSON " + json.dumps(out))
PY
