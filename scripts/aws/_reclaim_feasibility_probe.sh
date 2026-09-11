python3 - <<'PY'
"""What reclaim is REAL on this volume, and can any of it keep training alive?

Measured minutes before this ran: burn 159.91 GB/h, 69.68 GB above the 150 GB
floor, 0.44 h of runway, against an interval 40,040 rows from its gate at 1.11
rows/s -- a 10.02 h ETA. Ten hours at 160 GB/h is ~1,600 GB of appends on a
1.0 TB volume, so the interval cannot finish no matter what is deleted. The
question this probe answers is therefore not "can we finish" but "is there a
sustainable sawtooth at all", and it is asked in the only currency that counts
on an XFS reflink volume: `df`, never `du`.

Three candidate reclaims, each with the reason it might be worth nothing:

  * **Prunable deferred bases.** `prune_resolved_deferred_bases` removes
    `known - active`. It returned 0.00 GB last session because a RETIRED
    interval drops out of `known` and its base becomes unprunable forever.
    There are now 232 deferred directories, so the gap is worth re-measuring.
  * **The hardlink group.** Twenty-odd bases report 57,905,274,458 bytes at
    `st_nlink` 17 -- ONE inode wearing seventeen names. Deleting sixteen of
    them frees exactly nothing; deleting all seventeen frees the inode, and
    only then if its extents are not also reflink-shared with the live brain.
  * **Compaction.** `brain.wbrain` was 576.67 GB with 363.34 GB live when
    compaction was judged net-negative. It is now 836.46 GB apparent, so the
    garbage fraction has moved and the verdict deserves re-deriving -- but an
    out-of-place rewrite needs ~363 GB of free blocks and only ~220 GB exist,
    so `--estimate` may be describing an operation that cannot be started.

This probe is READ-ONLY. It deletes nothing, because a reclaim that is guessed
at rather than measured is how this volume reached 20 KB free once already.
Every size it reports is paired with the sharing evidence that says whether a
delete would actually return the bytes.
"""
import hashlib
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
out = {"now": time.time()}


def sh(cmd, timeout=180):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-7000:] + (
            ("\n[stderr] " + proc.stderr[-800:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


out["df_now"] = sh("df -B1 --output=avail,used,size /srv/wizard | tail -1")

# --- 1. Which deferred digests would the pruner actually remove? -----------
# Reimplemented read-only here: fold the ledger exactly as the supervisor does,
# then report `known - active` WITHOUT deleting, plus the digests that are
# neither (the retired-and-unprunable population).
def digest_of(interval_id):
    return hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]


known, active, ledger_states = set(), set(), {}
ledger = os.path.join(R, "deferred-intervals.jsonl")
try:
    with open(ledger, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            interval_id = event.get("interval_id")
            if isinstance(interval_id, str) and interval_id:
                known.add(digest_of(interval_id))
                ledger_states[digest_of(interval_id)] = event.get(
                    "state") or event.get("kind") or event.get("event")
except OSError as exc:
    out["ledger_error"] = f"{type(exc).__name__}: {exc}"

# Fold to unresolved obligations the same way the supervisor does.
folded = {}
try:
    with open(ledger, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            interval_id = event.get("interval_id")
            if isinstance(interval_id, str) and interval_id:
                folded[interval_id] = event
except OSError:
    pass
for interval_id, event in folded.items():
    state = str(event.get("state") or event.get("kind") or "")
    if "resolve" not in state and "retired" not in state:
        active.add(digest_of(interval_id))

on_disk = set()
deferred_root = os.path.join(R, "deferred")
try:
    on_disk = {name for name in os.listdir(deferred_root)
               if os.path.isdir(os.path.join(deferred_root, name))}
except OSError as exc:
    out["deferred_listdir_error"] = f"{type(exc).__name__}: {exc}"


def dir_bytes(digest):
    total, inodes = 0, {}
    path = os.path.join(deferred_root, digest)
    for root_dir, _dirs, files in os.walk(path):
        for name in files:
            try:
                stat = os.stat(os.path.join(root_dir, name))
            except OSError:
                continue
            total += stat.st_size
            inodes[stat.st_ino] = stat.st_nlink
    return total, inodes


prunable = sorted(known - active)
unknown = sorted(on_disk - known)
out["populations"] = {
    "ledger_known": len(known),
    "ledger_active_unresolved": len(active),
    "dirs_on_disk": len(on_disk),
    "prunable_known_minus_active": len(prunable),
    "unknown_on_disk_preserved": len(unknown),
}

prunable_bytes, prunable_inodes = 0, {}
for digest in prunable:
    if digest not in on_disk:
        continue
    size, inodes = dir_bytes(digest)
    prunable_bytes += size
    prunable_inodes.update(inodes)
unknown_bytes, unknown_inodes = 0, {}
for digest in unknown:
    size, inodes = dir_bytes(digest)
    unknown_bytes += size
    unknown_inodes.update(inodes)

out["prunable"] = {
    "digests": prunable[:40],
    "apparent_gb": round(prunable_bytes / 2**30, 2),
    "distinct_inodes": len(prunable_inodes),
    "unique_inode_gb_upper_bound": None,
}
out["unknown_preserved"] = {
    "apparent_gb": round(unknown_bytes / 2**30, 2),
    "distinct_inodes": len(unknown_inodes),
    "sample": unknown[:20],
}

# --- 2. Hardlink structure: how many NAMES share each big inode? ----------
inode_names = {}
for root_dir, _dirs, files in os.walk(deferred_root):
    for name in files:
        full = os.path.join(root_dir, name)
        try:
            stat = os.stat(full)
        except OSError:
            continue
        if stat.st_size < 2**30:
            continue
        entry = inode_names.setdefault(
            stat.st_ino, {"size_gb": round(stat.st_size / 2**30, 2),
                          "nlink": stat.st_nlink, "names": []}
        )
        entry["names"].append(os.path.relpath(full, deferred_root))
out["big_inodes"] = sorted(
    (
        {"ino": ino, **info, "name_count": len(info["names"]),
         "names": info["names"][:4]}
        for ino, info in inode_names.items()
    ),
    key=lambda item: -item["size_gb"],
)[:15]
out["deferred_unique_inode_gb"] = round(
    sum(info["size_gb"] for info in inode_names.values()), 2
)

# --- 3. Reflink sharing: are these extents shared with the LIVE brain? ----
# `filefrag -v` on a 836 GB file is unusable, so sample fixed offsets and
# compare physical block numbers, the same method that proved last-good was a
# clone. A shared extent means a delete returns nothing.
out["fiemap_sample"] = sh(
    "for f in "
    f"{R}/brain/brain.wbrain "
    f"{R}/brain/brain.last-good.wbrain "
    f"{R}/deferred/fb48d6903447476c/brain.base.wbrain "
    f"{R}/deferred/8f4a439a7fc7a772/brain.base.wbrain ; do "
    "echo \"=== $f\"; "
    "for off in 4 40 100 200 300 400 ; do "
    "python3 -c \"import sys;print(' off',sys.argv[1],'GB',end=' ')\" $off; "
    "xfs_io -r -c \"fiemap -v $((off*1024*1024*1024)) 4096\" \"$f\" 2>/dev/null "
    "| tail -2 | head -1 || echo unreadable; done; done",
    timeout=300,
)

# --- 4. Compaction: live vs garbage, and can it even be STARTED? ----------
out["compactor_binary"] = sh(
    f"ls -la {P}/target/release/wbrain_compact 2>&1; "
    f"ls -la {P}/target/release/wbrain_compact.exe 2>&1 | head -2"
)
out["compact_estimate"] = sh(
    f"cd {P} && ./target/release/wbrain_compact --estimate "
    f"--path {R}/brain/brain.wbrain 2>&1 | tail -30",
    timeout=900,
)

out["brain_stats"] = sh(
    "curl -s --max-time 30 http://127.0.0.1:18095/brain/stats 2>&1 | head -c 1500"
)
out["df_after"] = sh("df -B1 --output=avail,used,size /srv/wizard | tail -1")
out["free"] = sh("free -g")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
