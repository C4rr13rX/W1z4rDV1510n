#!/bin/bash
# How is /srv/wizard/project updated -- a git checkout, or file drops?
# Determines whether the fix ships as a pull or as an SSM write (which lands
# root:root and has to be chowned back to ec2-user).
set -uo pipefail

python3 - <<'PY'
import json
import os
import pathlib
import subprocess

out = {}
proj = pathlib.Path("/srv/wizard/project")
out["project_exists"] = proj.is_dir()
out["is_git"] = (proj / ".git").exists()

def sh(*cmd, cwd=None):
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=60)
        return (res.stdout or res.stderr or "").strip()
    except Exception as exc:
        return f"ERR {exc}"

if out["is_git"]:
    out["head"] = sh("git", "rev-parse", "HEAD", cwd=proj)
    out["branch"] = sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=proj)
    out["remote"] = sh("git", "remote", "-v", cwd=proj)
    out["dirty"] = sh("git", "status", "--porcelain", cwd=proj)[:2000]

# Ownership matters: the supervisor runs as ec2-user.
for rel in ("tools/training_standard/registry/go_systems_001.toml",
            "tools/training_standard/schema.py",
            "tools/training_standard/score.py",
            "scripts/programming_curriculum_supervisor.py"):
    p = proj / rel
    if p.exists():
        st = p.stat()
        out.setdefault("files", {})[rel] = {
            "uid": st.st_uid, "gid": st.st_gid,
            "mode": oct(st.st_mode)[-4:],
            "mtime": st.st_mtime, "inode": st.st_ino,
            "size": st.st_size,
        }
    else:
        out.setdefault("files", {})[rel] = None

out["whoami"] = sh("id")
out["service_state"] = sh("systemctl", "is-active",
                          "wizard-curriculum-supervisor")
out["service_result"] = sh("systemctl", "show", "-p", "Result", "-p",
                           "ExecMainStatus", "-p", "NRestarts",
                           "wizard-curriculum-supervisor")

print("PROBE_JSON " + json.dumps(out, default=str))
PY
