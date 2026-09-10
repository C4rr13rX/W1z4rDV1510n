set -u
# The repo's go_systems_001.toml already says category="code_generation"
# (commit 44e496b) and the 15 local schema tests pass. The host, 3.5 h ago,
# was still raising SchemaError on category='systems_programming_go'. That is
# the `deploy_is_not_load` shape: a fix that exists in git is not a fix that
# the running curriculum has loaded. So do not eyeball the field -- run the
# host's OWN load_registry() against the host's OWN registry directory, which
# is the only check that reproduces what the worker does at startup.
python3 - <<'PY'
import glob, json, os, subprocess, sys, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
out = {"now": time.time()}

# --- what the host's registry actually contains ---------------------------
reg = os.path.join(P, "tools/training_standard/registry")
out["registry_dir"] = reg
files = sorted(glob.glob(os.path.join(reg, "*.toml")))
out["registry_count"] = len(files)
out["registry_mtimes"] = sorted(
    ({"f": os.path.basename(p), "age_s": round(time.time() - os.path.getmtime(p))}
     for p in files), key=lambda d: d["age_s"])[:8]
go = os.path.join(reg, "go_systems_001.toml")
if os.path.exists(go):
    text = open(go, encoding="utf-8", errors="replace").read()
    out["go_category_lines"] = [ln for ln in text.splitlines()
                                if ln.strip().startswith("category")]
    out["go_mtime_age_s"] = round(time.time() - os.path.getmtime(go))
    out["go_owner"] = subprocess.run(["stat", "-c", "%U:%G", go],
                                     capture_output=True, text=True).stdout.strip()

# --- the decisive test: does the host's loader load the host's registry? ---
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0,'%s');\n"
     "from tools.training_standard.schema import load_registry\n"
     "import tools.training_standard.runner as r\n"
     "import pathlib\n"
     "reg = load_registry(r.REGISTRY_DIR)\n"
     "print('LOADOK', len(reg), r.REGISTRY_DIR)\n" % P],
    capture_output=True, text=True, cwd=P, timeout=120)
out["load_stdout"] = probe.stdout.strip()[-400:]
out["load_stderr"] = probe.stderr.strip()[-900:]
out["load_rc"] = probe.returncode

# --- host git state --------------------------------------------------------
for args, key in ((["git", "log", "-1", "--format=%H %ci %s"], "head"),
                  (["git", "status", "--short"], "dirty")):
    got = subprocess.run(args, capture_output=True, text=True, cwd=P)
    out[key] = got.stdout.strip()[:600] or got.stderr.strip()[:200]

# --- quarantine latch ------------------------------------------------------
for name in ("continuous-canary-quarantine.json", "deferred-replay-active.json",
             "curriculum-supervisor.status.json"):
    path = os.path.join(R, name)
    out.setdefault("runtime_files", {})[name] = (
        {"age_s": round(time.time() - os.path.getmtime(path)),
         "head": open(path, encoding="utf-8", errors="replace").read()[:700]}
        if os.path.exists(path) else None)
out["quarantine_glob"] = [os.path.basename(p) for p in
                          glob.glob(os.path.join(R, "*quarantine*"))][:20]

# --- is the canary advancing? sample the file the supervisor really writes -
status_path = os.path.join(R, "curriculum-supervisor.status.json")
samples = []
for _ in range(6):
    try:
        d = json.load(open(status_path, encoding="utf-8"))
        samples.append({"t": round(time.time() - out["now"], 1),
                        "state": d.get("state"), "phase": d.get("phase"),
                        "canary_row": d.get("canary_row"),
                        "ram_next_row": d.get("ram_next_row"),
                        "durable_next_row": d.get("durable_next_row"),
                        "pid": d.get("worker_pid"),
                        "age_s": round(time.time() - os.path.getmtime(status_path))})
    except Exception as error:
        samples.append({"err": str(error)[:100]})
    time.sleep(25)
out["samples"] = samples

out["procs"] = subprocess.run(
    ["bash", "-lc", "ps -eo pid,etimes,rss,stat,args --sort=-rss | head -8 | cut -c1-160"],
    capture_output=True, text=True).stdout.splitlines()
print("PROBEJSON " + json.dumps(out))
PY
