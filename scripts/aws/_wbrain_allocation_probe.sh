set -u
# Stop the crash loop FIRST. 115 restarts at RestartSec=10 have been appending
# ENOSPC tracebacks into `curriculum-service.stderr.log`, which lives on the
# very filesystem that is full -- so any byte reclaimed while the loop runs is
# re-consumed within seconds. `stop_programming_curriculum_service.sh` targets
# only the worker and supervisor markers and the unit is KillMode=process, so
# the brain server keeps its hydrated fabric across this stop.
systemctl stop wizard-curriculum-supervisor.service 2>&1 || true
echo "STOPPED_UNIT rc=$?"

python3 - <<'PY'
"""Allocated blocks, not apparent size, for every large runtime file.

`du` reported 2.31 TB inside a 1.0 TB volume, so apparent size is not what the
filesystem is holding: either the `.wbrain` files are sparse, or they share
blocks. Which one decides the entire reclaim plan -- deleting an old causal
base that is a HARD LINK to the live guard frees nothing at all, and would
destroy a rollback anchor for zero benefit.

st_ino groups hardlinks; st_blocks*512 is what a delete actually returns.
Nothing here removes anything.
"""
import collections
import json
import os
import subprocess

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}


def sh(cmd, timeout=180):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-4000:] + (
            ("\n[stderr] " + proc.stderr[-500:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


out["fstype"] = sh("findmnt -no SOURCE,FSTYPE,OPTIONS /srv/wizard")
out["xfs_info"] = sh("xfs_info /srv/wizard 2>&1 | head -12 || true")
out["df"] = sh("df -h /srv/wizard")

records = []
for dirpath, dirnames, filenames in os.walk(R):
    for name in filenames:
        path = os.path.join(dirpath, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if st.st_size < 100 * 1024 * 1024 and st.st_blocks * 512 < 100 * 1024 * 1024:
            continue
        records.append(
            {
                "path": path[len(R) + 1:],
                "inode": st.st_ino,
                "links": st.st_nlink,
                "apparent_gb": round(st.st_size / 1e9, 2),
                "allocated_gb": round(st.st_blocks * 512 / 1e9, 2),
                "mtime": st.st_mtime,
            }
        )

# One inode counted once, however many names point at it.
by_inode = {}
for rec in records:
    by_inode.setdefault(rec["inode"], []).append(rec)

unique = []
for inode, group in by_inode.items():
    first = dict(group[0])
    first["names"] = sorted(r["path"] for r in group)
    first["name_count"] = len(group)
    unique.append(first)
unique.sort(key=lambda r: -r["allocated_gb"])

out["unique_inodes"] = [
    {
        "allocated_gb": r["allocated_gb"],
        "apparent_gb": r["apparent_gb"],
        "st_nlink": r["links"],
        "names_here": r["name_count"],
        "first_name": r["names"][0],
        "all_names": r["names"][:6] if r["name_count"] <= 6 else
        r["names"][:3] + [f"... +{r['name_count'] - 3} more"],
    }
    for r in unique[:30]
]
out["unique_inode_total_allocated_gb"] = round(
    sum(r["allocated_gb"] for r in unique), 2
)
out["unique_inode_total_apparent_gb"] = round(
    sum(r["apparent_gb"] for r in unique), 2
)
out["large_name_count"] = len(records)
out["unique_inode_count"] = len(unique)

# How many distinct inodes does deferred/ actually own, versus names?
deferred = [r for r in unique if any(n.startswith("deferred/") for n in r["names"])]
out["deferred_unique_inodes"] = len(deferred)
out["deferred_allocated_gb"] = round(sum(r["allocated_gb"] for r in deferred), 2)
out["deferred_names"] = sum(r["name_count"] for r in deferred)

# Which deferred digests are still an outstanding obligation? Fold the ledger
# exactly the way `unresolved_deferred_intervals` does.
import hashlib

current = {}
try:
    with open(os.path.join(R, "curriculum-deferred-intervals.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except Exception:
                continue
            iid = event.get("interval_id")
            if not isinstance(iid, str) or not iid:
                continue
            if event.get("status") == "resolved":
                current.pop(iid, None)
            elif event.get("status") == "deferred":
                current[iid] = event
except OSError as exc:
    out["ledger_error"] = str(exc)

out["unresolved_count"] = len(current)
out["unresolved_ids"] = sorted(current)[:40]
unresolved_digests = {
    hashlib.sha256(i.encode("utf-8")).hexdigest()[:16] for i in current
}
# Protect a base_snapshot directory recorded on an unresolved event too.
for event in current.values():
    snap = event.get("base_snapshot")
    if isinstance(snap, str) and snap:
        unresolved_digests.add(os.path.basename(os.path.dirname(snap)))
out["unresolved_digests"] = sorted(unresolved_digests)

try:
    on_disk = sorted(os.listdir(os.path.join(R, "deferred")))
except OSError as exc:
    on_disk = []
    out["deferred_listdir_error"] = str(exc)
out["deferred_dirs_on_disk"] = len(on_disk)
out["deferred_dirs_prunable"] = sorted(set(on_disk) - unresolved_digests)[:60]
out["deferred_dirs_prunable_count"] = len(set(on_disk) - unresolved_digests)
out["deferred_dirs_protected"] = sorted(set(on_disk) & unresolved_digests)

# What would pruning actually return? Sum allocated blocks of inodes whose
# ONLY names live under prunable digests.
prunable = set(on_disk) - unresolved_digests
freed = 0.0
kept_shared = 0.0
for r in unique:
    names = r["names"]
    def digest_of(n):
        parts = n.split("/")
        return parts[1] if len(parts) > 1 and parts[0] == "deferred" else None
    in_prunable = [n for n in names if digest_of(n) in prunable]
    if not in_prunable:
        continue
    if len(in_prunable) == len(names) and r["links"] == len(names):
        freed += r["allocated_gb"]
    else:
        kept_shared += r["allocated_gb"]
out["prune_would_free_gb"] = round(freed, 2)
out["prune_shared_with_live_gb"] = round(kept_shared, 2)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
