#!/bin/bash
# Ship commit 44e496b's four changed files to /srv/wizard/project and PROVE the
# registry loads afterwards.
#
# Surgical rather than a whole-tree extract: /srv/wizard/project is not a git
# checkout, so a full overwrite would silently discard any host-local drift
# nobody has audited. Four files caused this outage; four files go back.
#
# Does not restart the service. "Deploy is not load" -- the copy is verified
# here, and the restart is a separate, gated step.
set -uo pipefail

COMMIT=44e496b8acf51fac2aa2f663fd826df25c1f187f
BUCKET=wizard-vision-private-321572159829-us-east-1
KEY="wizard-vision/source/${COMMIT}/wizard-vision-source.tar.gz"
WANT_SHA=571595e77f80a231a98036c30bcc8c93a175dc0f803cc6a991db4971bc84ce4a

PROJ=/srv/wizard/project
STAGE=$(mktemp -d /tmp/wvfix.XXXXXX)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

TARBALL="$STAGE/src.tar.gz"
if ! aws s3 cp "s3://${BUCKET}/${KEY}" "$TARBALL" --region us-east-1 --only-show-errors; then
    echo "PROBE_JSON {\"error\":\"s3 download failed\",\"key\":\"${KEY}\"}"
    exit 0
fi

GOT_SHA=$(sha256sum "$TARBALL" | awk '{print $1}')
if [ "$GOT_SHA" != "$WANT_SHA" ]; then
    echo "PROBE_JSON {\"error\":\"sha mismatch\",\"want\":\"${WANT_SHA}\",\"got\":\"${GOT_SHA}\"}"
    exit 0
fi

mkdir -p "$STAGE/tree"
tar -xzf "$TARBALL" -C "$STAGE/tree" || {
    echo 'PROBE_JSON {"error":"extract failed"}'; exit 0; }

FILES="
tools/training_standard/registry/go_systems_001.toml
tools/training_standard/schema.py
tools/training_standard/score.py
scripts/programming_curriculum_supervisor.py
"

BACKUP="$PROJ/.deploy-backup-$STAMP"
mkdir -p "$BACKUP"

for rel in $FILES; do
    src="$STAGE/tree/$rel"
    dst="$PROJ/$rel"
    [ -f "$src" ] || { echo "PROBE_JSON {\"error\":\"missing in bundle: $rel\"}"; exit 0; }
    if [ -f "$dst" ]; then
        mkdir -p "$BACKUP/$(dirname "$rel")"
        cp -p "$dst" "$BACKUP/$rel"
    fi
    install -m 0644 "$src" "$dst"
    # SSM runs as root, so every write above landed root:root. The supervisor
    # runs as ec2-user and dies with Permission denied on anything it must
    # rewrite, so hand each file back before returning.
    chown 1000:1000 "$dst"
done
chown -R 1000:1000 "$BACKUP"

python3 - <<PY
import hashlib
import json
import pathlib
import subprocess
import sys

proj = pathlib.Path("$PROJ")
out = {"commit": "$COMMIT", "backup": "$BACKUP", "files": {}}

for rel in """$FILES""".split():
    p = proj / rel
    st = p.stat()
    out["files"][rel] = {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        "uid": st.st_uid, "gid": st.st_gid, "mode": oct(st.st_mode)[-4:],
        "inode": st.st_ino, "size": st.st_size,
    }

# The check both outages needed: does the DEPLOYED registry actually load?
sys.path.insert(0, str(proj))
try:
    from tools.training_standard import schema
    scripts = schema.load_registry(proj / "tools" / "training_standard" / "registry")
    out["registry_loads"] = True
    out["registry_count"] = len(scripts)
    out["go_present"] = any(
        getattr(s, "id", None) == "go_systems_001"
        for s in (scripts.values() if hasattr(scripts, "values") else scripts)
    )
    out["supported_langs"] = sorted(schema.SUPPORTED_LANGS)
except Exception as exc:
    out["registry_loads"] = False
    out["registry_error"] = f"{type(exc).__name__}: {exc}"

# And does the driver itself now get past the import that killed every worker?
res = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0,'%s');"
     "import tools.training_standard.drive_corpora_brain as d;"
     "print('driver import ok')" % proj],
    capture_output=True, text=True, timeout=180,
)
out["driver_import_rc"] = res.returncode
out["driver_import_out"] = (res.stdout + res.stderr).strip()[-600:]

# The supervisor's new classifier must be present in the deployed copy, not
# just in the repo.
sup = (proj / "scripts" / "programming_curriculum_supervisor.py").read_text(
    encoding="utf-8", errors="replace")
out["supervisor_has_classifier"] = "def replay_worker_failure" in sup
out["supervisor_has_stderr_tail"] = "def replay_worker_stderr_tail" in sup

print("PROBE_JSON " + json.dumps(out, default=str))
PY
