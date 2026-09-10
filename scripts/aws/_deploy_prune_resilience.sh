#!/bin/bash
# Ship commit 5fe8efd's resilient `prune_resolved_deferred_bases` to
# /srv/wizard/project and PROVE the deployed copy carries it.
#
# Deliberately does NOT restart the supervisor. `deferred-replay-active.json`
# is in state `training` at ~row 202k of the jupyter-scientific-full
# 201344:262144 interval, and a restart discards the whole interval -- that is
# exactly what cost 39,704 rows an hour ago when the crashed generation was
# replaced. Ownership on this host is already repaired, so the CURRENTLY LOADED
# prune has nothing left to abort on; the new code is defence against the next
# root-owned directory and loads at the next natural wrapper restart.
#
# "Deploy is not load": this verifies the bytes on disk by inode and content,
# and says plainly that the running process is still the old generation.
set -uo pipefail

COMMIT=5fe8efdaa1da5eead60a915b209beb80b833ba69
BUCKET=wizard-vision-private-321572159829-us-east-1
KEY="wizard-vision/source/${COMMIT}/wizard-vision-source.tar.gz"
WANT_SHA=cba3da4e432f87d6d4c3b6fd3a195ec7c63fadfc34641ba16732c5788ab779ed

PROJ=/srv/wizard/project
STAGE=$(mktemp -d /tmp/wvprune.XXXXXX)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
trap 'rm -rf "$STAGE"' EXIT

TARBALL="$STAGE/src.tar.gz"
if ! aws s3 cp "s3://${BUCKET}/${KEY}" "$TARBALL" --region us-east-1 --only-show-errors; then
    echo "DEPLOY_ERROR s3 download failed: ${KEY}"; exit 0
fi
GOT_SHA=$(sha256sum "$TARBALL" | awk '{print $1}')
if [ "$GOT_SHA" != "$WANT_SHA" ]; then
    echo "DEPLOY_ERROR sha mismatch got=${GOT_SHA}"; exit 0
fi
mkdir -p "$STAGE/tree"
tar -xzf "$TARBALL" -C "$STAGE/tree" || { echo "DEPLOY_ERROR extract failed"; exit 0; }

FILES="scripts/programming_curriculum_supervisor.py"
BACKUP="$PROJ/.deploy-backup-$STAMP"
mkdir -p "$BACKUP/scripts"

for f in $FILES; do
    echo "--- $f"
    echo "  inode BEFORE: $(stat -c '%i' "$PROJ/$f" 2>/dev/null || echo none)"
    cp -p "$PROJ/$f" "$BACKUP/$f" 2>/dev/null || true
    cp -f "$STAGE/tree/$f" "$PROJ/$f"
    # SSM runs as root; the supervisor runs as ec2-user and dies on anything it
    # must write. This is the same trap that caused the outage being fixed.
    chown ec2-user:ec2-user "$PROJ/$f"
    echo "  inode AFTER:  $(stat -c '%i' "$PROJ/$f")"
    echo "  owner:        $(stat -c '%U:%G' "$PROJ/$f")"
done

echo "=== the fix is present in the bytes on disk ==="
grep -c "deferred_base_prune_blocked" "$PROJ/scripts/programming_curriculum_supervisor.py"
grep -n "except OSError as error:" "$PROJ/scripts/programming_curriculum_supervisor.py" | head -4

echo "=== the deployed file imports and the function is callable ==="
cd "$PROJ"
sudo -u ec2-user python3 -c "
import sys; sys.path.insert(0, '/srv/wizard/project')
from scripts.programming_curriculum_supervisor import prune_resolved_deferred_bases as p
import inspect
src = inspect.getsource(p)
print('HAS_PER_DIR_GUARD', 'except OSError as error:' in src)
print('HAS_BLOCKED_EVENT', 'deferred_base_prune_blocked' in src)
"

echo "=== running generation is still the OLD one (deploy is not load) ==="
supervisor_pid=$(pgrep -f "programming_curriculum_supervisor.py --runtime" | head -1)
echo "supervisor pid: ${supervisor_pid:-none}"
if [ -n "${supervisor_pid:-}" ]; then
    echo "started: $(ps -o lstart= -p "$supervisor_pid")"
fi

echo "=== disk + training liveness ==="
df -h /srv/wizard
systemctl show wizard-curriculum-supervisor.service \
  -p ActiveState -p SubState -p NRestarts --no-pager
