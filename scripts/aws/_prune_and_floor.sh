set -u
# Run the supervisor's OWN base pruner and measure what it returns.
#
# 106+ `brain.base.wbrain` names exist against 26 unresolved intervals, so most
# belong to intervals already resolved or retired. `prune_resolved_deferred_bases`
# is the sanctioned routine -- it protects every unresolved interval and its
# recorded base_snapshot -- so this is a reclaim that cannot violate a replay
# obligation. It is called rather than reimplemented for exactly that reason.
#
# The reclaim is measured with `df`, never by summing file sizes: the tree
# claims 7,417 GB of apparent size against 596 GB actually used, so an inode
# with nlink=85 returns nothing until its LAST name goes.
R=/srv/wizard/runtime/programming-integrated-20260713
cd /srv/wizard/project || exit 1

echo "=== before ==="
df -B1 --output=avail "$R" | tail -1
find "$R/deferred" -maxdepth 1 -mindepth 1 -type d | wc -l
find "$R/deferred" -name 'brain.base.*' | wc -l

echo "=== prune ==="
sudo -u ec2-user python3 - <<'PY'
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, "/srv/wizard/project")
R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")

from scripts.programming_curriculum_supervisor import (
    prune_resolved_deferred_bases,
    unresolved_deferred_intervals,
)

before = shutil.disk_usage(R).free
unresolved = list(unresolved_deferred_intervals(R))
print(json.dumps({"unresolved_intervals": len(unresolved)}))

started = time.time()
try:
    removed = prune_resolved_deferred_bases(R)
except Exception as exc:
    print(json.dumps({"prune_error": f"{type(exc).__name__}: {exc}"}))
    removed = []
after = shutil.disk_usage(R).free
print(json.dumps({
    "removed_directories": len(removed),
    "elapsed_seconds": round(time.time() - started, 1),
    "free_before_gb": round(before / 1e9, 2),
    "free_after_gb": round(after / 1e9, 2),
    "reclaimed_gb": round((after - before) / 1e9, 2),
}))
PY

echo "=== after ==="
df -B1 --output=avail "$R" | tail -1
find "$R/deferred" -maxdepth 1 -mindepth 1 -type d | wc -l
find "$R/deferred" -name 'brain.base.*' | wc -l
echo "=== distinct base inodes remaining ==="
find "$R/deferred" -name 'brain.base.*' -printf '%i %n %s\n' 2>/dev/null \
  | sort -u -k1,1n | awk '{printf "inode=%s nlink=%s gb=%.2f\n", $1, $2, $3/1e9}'
