set -u
R=/srv/wizard/runtime/programming-integrated-20260713

# Why the 1 TB volume filled: `prune_resolved_deferred_bases` is the only thing
# that reclaims multi-gigabyte causal bases, the supervisor runs as ec2-user
# (User=ec2-user in the unit), and unlinking a file needs write permission on
# its DIRECTORY. Deferred directories created during root-run SSM operations
# are root:root, so the prune raised PermissionError instead of freeing space.
# This is the SSM root-ownership trap CLAUDE.md already names, landing on the
# disk-reclaim path rather than on a supervisor write.

echo "=== ownership census BEFORE ==="
find "${R}/deferred" -maxdepth 1 -mindepth 1 -type d -printf '%u:%g\n' \
  | sort | uniq -c | sort -rn
echo "--- runtime top level ---"
find "${R}" -maxdepth 1 -printf '%u:%g\n' | sort | uniq -c | sort -rn | head

echo "=== chown to the identity that actually runs the supervisor ==="
chown -R ec2-user:ec2-user "${R}"
echo "chown rc=$?"

echo "=== ownership census AFTER ==="
find "${R}/deferred" -maxdepth 1 -mindepth 1 -type d -printf '%u:%g\n' \
  | sort | uniq -c | sort -rn

echo "=== df BEFORE prune ==="
df -h /srv/wizard

cd /srv/wizard/project
sudo -u ec2-user python3 - <<'PY'
import pathlib
import shutil
import sys

sys.path.insert(0, "/srv/wizard/project")
from scripts.programming_curriculum_supervisor import (
    prune_resolved_deferred_bases,
    unresolved_deferred_intervals,
)

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
before = shutil.disk_usage(runtime)
outstanding = unresolved_deferred_intervals(runtime)
print(f"UNRESOLVED_BEFORE {len(outstanding)}")
removed = prune_resolved_deferred_bases(runtime)
after = shutil.disk_usage(runtime)
print(f"PRUNED_DIRS {len(removed)}")
print(f"FREE_BEFORE_GB {before.free / 1e9:.2f}")
print(f"FREE_AFTER_GB {after.free / 1e9:.2f}")
print(f"RECLAIMED_GB {(after.free - before.free) / 1e9:.2f}")

still = unresolved_deferred_intervals(runtime)
print(f"UNRESOLVED_AFTER {len(still)}")
assert {r["interval_id"] for r in outstanding} == {r["interval_id"] for r in still}, \
    "prune changed the outstanding obligation set"
print("OBLIGATIONS_UNCHANGED")

missing = []
for row in still:
    snap = row.get("base_snapshot")
    if snap and not pathlib.Path(snap).is_file():
        missing.append((row["interval_id"], snap))
print(f"MISSING_BASES {len(missing)}")
for iid, snap in missing[:10]:
    print(f"  MISSING {iid} -> {snap}")
PY

echo "=== df AFTER prune ==="
df -h /srv/wizard
echo "deferred dirs remaining: $(ls -1 ${R}/deferred | wc -l)"
