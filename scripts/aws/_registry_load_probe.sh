#!/bin/bash
# Does the DEPLOYED registry load, and is that why every replay worker died?
#
# The local repo and the host can disagree: a fix committed here has not run
# until the host's copy changes (see "deploy is not load"). So this asks the
# host's own Python, against the host's own registry directory, and prints the
# SchemaError verbatim rather than a boolean.
set -uo pipefail

RUNTIME=/srv/wizard/runtime/programming-integrated-20260713
cd "$RUNTIME" 2>/dev/null || { echo '{"error":"no runtime"}'; exit 0; }

python3 - <<'PY'
import glob
import json
import os
import pathlib
import subprocess
import sys

out = {}
runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
src = runtime / "src" if (runtime / "src").is_dir() else runtime
sys.path.insert(0, str(src))

reg = src / "tools" / "training_standard" / "registry"
out["registry_dir"] = str(reg)
out["registry_exists"] = reg.is_dir()

# 1) The deployed go file: what category / must_be_valid does the HOST have?
go = reg / "go_systems_001.toml"
if go.is_file():
    text = go.read_text(encoding="utf-8", errors="replace")
    out["go_toml_mtime"] = go.stat().st_mtime
    out["go_category"] = [
        l.strip() for l in text.splitlines() if l.strip().startswith("category")
    ]
    out["go_must_be_valid"] = sorted({
        l.strip() for l in text.splitlines() if "must_be_valid" in l
    })
else:
    out["go_toml_mtime"] = None

# 2) Does load_registry() actually succeed on the host right now?
try:
    from tools.training_standard import schema
    scripts = schema.load_registry(reg)
    out["registry_loads"] = True
    out["registry_count"] = len(scripts)
    out["supported_langs"] = sorted(schema.SUPPORTED_LANGS)
except Exception as exc:
    out["registry_loads"] = False
    out["registry_error"] = f"{type(exc).__name__}: {exc}"

# 3) The smoking gun: what did the dead replay workers write to stderr?
tails = {}
logs = sorted(glob.glob(str(runtime / "deferred-replay-*.stderr.log")),
              key=os.path.getmtime, reverse=True)[:4]
for path in logs:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        tails[os.path.basename(path)] = {
            "bytes": len(body),
            "tail": body[-1200:].strip(),
        }
    except OSError as exc:
        tails[os.path.basename(path)] = {"error": str(exc)}
out["stderr_tails"] = tails

# 4) Is anything actually running?
def count(pattern):
    try:
        res = subprocess.run(["pgrep", "-fc", pattern],
                             capture_output=True, text=True, timeout=20)
        return int((res.stdout or "0").strip() or 0)
    except Exception:
        return -1

out["procs"] = {
    "supervisor": count("programming_curriculum_supervisor"),
    "driver": count("drive_corpora_brain"),
    "brain": count("wizard-brain-server|brain_server"),
}

print("PROBE_JSON " + json.dumps(out, default=str))
PY
