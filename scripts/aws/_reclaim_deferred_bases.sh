set -u
R=/srv/wizard/runtime/programming-integrated-20260713

# Preserve the evidence before truncating. The 57 MB stderr log is almost all
# repeated ENOSPC tracebacks, but its HEAD holds what the wrapper was doing
# when the volume first filled, and CLAUDE.md's front-truncation lesson says
# the informative line of a traceback is its last. Keep both ends, and keep the
# first ENOSPC with context, on the ROOT filesystem which has 26 GB free.
mkdir -p /var/tmp/wizard-enospc-evidence
head -c 400000 "${R}/curriculum-service.stderr.log" \
  >/var/tmp/wizard-enospc-evidence/stderr.head.txt 2>/dev/null || true
tail -c 400000 "${R}/curriculum-service.stderr.log" \
  >/var/tmp/wizard-enospc-evidence/stderr.tail.txt 2>/dev/null || true
grep -n -m1 -B40 -A10 "No space left on device" \
  "${R}/curriculum-service.stderr.log" \
  >/var/tmp/wizard-enospc-evidence/first_enospc.txt 2>/dev/null || true
echo "EVIDENCE_SAVED"
ls -l /var/tmp/wizard-enospc-evidence/

echo "=== df BEFORE ==="
df -h /srv/wizard

# Truncate in place so systemd's append: target keeps its inode, owner and mode.
truncate -s 0 "${R}/curriculum-service.stderr.log"
truncate -s 0 "${R}/curriculum-service.stdout.log" 2>/dev/null || true
truncate -s 0 "${R}/node-auto-recovery.stderr.log" 2>/dev/null || true
echo "LOGS_TRUNCATED"
df -h /srv/wizard

# Prune with the supervisor's OWN routine. Hand-rolling `rm` here would put the
# accept/quarantine/replay invariant in a shell script that no test covers;
# `prune_resolved_deferred_bases` folds the append-only ledger, refuses to act
# outside the deferred root, and is covered by tests in
# tests/test_programming_runtime_contract.py.
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
for path in removed[:10]:
    print(f"  removed {path}")
print(f"FREE_BEFORE_GB {before.free / 1e9:.2f}")
print(f"FREE_AFTER_GB {after.free / 1e9:.2f}")
print(f"RECLAIMED_GB {(after.free - before.free) / 1e9:.2f}")
still = unresolved_deferred_intervals(runtime)
print(f"UNRESOLVED_AFTER {len(still)}")
# The obligation set must be IDENTICAL across a prune. Pruning is a disk
# operation, never a curriculum decision.
assert {r["interval_id"] for r in outstanding} == {r["interval_id"] for r in still}, \
    "prune changed the outstanding obligation set"
print("OBLIGATIONS_UNCHANGED")

# Every protected interval must still have its causal base on disk, or a
# rollback has silently lost its anchor.
missing = []
for row in still:
    snap = row.get("base_snapshot")
    if snap and not pathlib.Path(snap).is_file():
        missing.append((row["interval_id"], snap))
print(f"MISSING_BASES {len(missing)}")
for iid, snap in missing[:10]:
    print(f"  MISSING {iid} -> {snap}")
PY

echo "=== df AFTER ==="
df -h /srv/wizard
df -i /srv/wizard
ls -1 "${R}/deferred" | wc -l
