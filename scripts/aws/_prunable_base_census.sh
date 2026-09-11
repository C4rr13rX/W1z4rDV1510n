python3 - <<'PY'
"""How much of the deferred tree would `prune_resolved_deferred_bases` return?

The previous census answered this against `deferred-intervals.jsonl` and got
`ledger_known: 0` -- the real path is `curriculum-deferred-intervals.jsonl`, so
every digest fell into "unknown, deliberately preserved" and the answer was a
vacuous zero. That is the exact failure mode CLAUDE.md names: a pattern that
cannot match reports 0 forever. This re-asks it against the path the supervisor
actually uses, and folds the ledger with the supervisor's own rule
(`status == "resolved"` pops, `status == "deferred"` sets) rather than a
guessed one.

It decides something concrete. The interval is 39,600 rows from its gate at
~1.1 rows/s against 61 GB above the 150 GB floor at ~93-160 GB/h, so it has
well under an hour and needs ten. If a real reclaim exists the block can keep
training; if it does not, the volume is the binding constraint and that is a
purchase decision, not a code change. Either way the answer must come from
`df`, because on this reflink volume one 53.55 GB inode wears 85 names and
summing sizes would report terabytes that do not exist.

READ-ONLY: it computes what the pruner WOULD remove and the upper bound on
what that could return, without unlinking anything.
"""
import hashlib
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
LEDGER = os.path.join(R, "curriculum-deferred-intervals.jsonl")
DEFERRED = os.path.join(R, "deferred")
out = {"now": time.time()}


def sh(cmd, timeout=180):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-5000:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def digest_of(interval_id):
    return hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]


out["ledger_exists"] = os.path.isfile(LEDGER)
out["ledger_bytes"] = os.path.getsize(LEDGER) if out["ledger_exists"] else 0

known, current, statuses = set(), {}, {}
lines = 0
with open(LEDGER, "r", encoding="utf-8") as handle:
    for line in handle:
        lines += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        interval_id = event.get("interval_id")
        if not isinstance(interval_id, str) or not interval_id:
            continue
        known.add(digest_of(interval_id))
        statuses[event.get("status")] = statuses.get(event.get("status"), 0) + 1
        # The supervisor's own fold, not a guess at it.
        if event.get("status") == "resolved":
            current.pop(interval_id, None)
        elif event.get("status") == "deferred":
            current[interval_id] = event
out["ledger_lines"] = lines
out["status_counts"] = statuses

active = {digest_of(interval_id) for interval_id in current}
# `base_snapshot` protection, as the pruner applies it.
for event in current.values():
    snapshot = event.get("base_snapshot")
    if isinstance(snapshot, str) and snapshot:
        parent = os.path.basename(os.path.dirname(snapshot))
        if parent:
            active.add(parent)

on_disk = {name for name in os.listdir(DEFERRED)
           if os.path.isdir(os.path.join(DEFERRED, name))}

prunable = sorted((known - active) & on_disk)
protected = sorted(active & on_disk)
unknown = sorted(on_disk - known)

out["populations"] = {
    "ledger_known_digests": len(known),
    "unresolved_active": len(active),
    "dirs_on_disk": len(on_disk),
    "prunable": len(prunable),
    "protected_active": len(protected),
    "unknown_preserved": len(unknown),
}

# What would actually come back: count each INODE once, and only inodes whose
# every remaining link sits inside the prunable set. An inode also linked from
# a protected or unknown directory survives the unlink and returns nothing.
link_owners = {}
for digest in sorted(on_disk):
    directory = os.path.join(DEFERRED, digest)
    for root_dir, _dirs, files in os.walk(directory):
        for name in files:
            try:
                stat = os.stat(os.path.join(root_dir, name))
            except OSError:
                continue
            entry = link_owners.setdefault(
                stat.st_ino,
                {"size": stat.st_size, "nlink": stat.st_nlink, "dirs": set()},
            )
            entry["dirs"].add(digest)

prunable_set = set(prunable)
freed_bytes = 0
pinned_bytes = 0
for ino, entry in link_owners.items():
    if entry["dirs"] and entry["dirs"] <= prunable_set:
        # Every name we know of is inside the prunable set. If nlink exceeds
        # the names found here the inode is ALSO linked from outside the tree
        # (the last-good guard), and unlinking returns nothing.
        if entry["nlink"] <= len(entry["dirs"]):
            freed_bytes += entry["size"]
        else:
            pinned_bytes += entry["size"]
    else:
        pinned_bytes += entry["size"]

out["reclaim_upper_bound"] = {
    "would_free_gb": round(freed_bytes / 2**30, 2),
    "pinned_elsewhere_gb": round(pinned_bytes / 2**30, 2),
    "distinct_inodes": len(link_owners),
    "note": "upper bound: reflink sharing with the live brain is not visible "
            "here, so the df delta can still be smaller",
}
out["prunable_sample"] = prunable[:25]
out["protected_sample"] = protected[:25]
out["unknown_sample"] = unknown[:25]

usage = shutil.disk_usage("/srv/wizard")
out["df"] = {
    "free_gb": round(usage.free / 2**30, 2),
    "used_gb": round(usage.used / 2**30, 2),
}
out["floor_gb"] = 150.0
out["headroom_gb"] = round(usage.free / 2**30 - 150.0, 2)
out["status_now"] = sh(f"tail -c 700 {R}/curriculum-supervisor.status.json 2>&1")
out["active_marker"] = sh(f"tail -c 700 {R}/deferred-replay-active.json 2>&1")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
