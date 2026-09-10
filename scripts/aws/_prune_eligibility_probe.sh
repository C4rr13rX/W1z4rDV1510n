python3 - <<'PY'
"""Per-directory: exactly why `prune_resolved_deferred_bases` did or did not act.

The prune removed 9 directories and returned 0.00 GB. That is not necessarily a
bug -- 85 of the large names are hardlinks to ONE 57.5 GB inode, so removing 9
of them frees nothing until the last link goes. But 618 GB was predicted and 0
arrived, so the set arithmetic gets measured rather than assumed.

Three predicates decide each directory, and they are not the same predicate:
  known  -- digest appears in the append-only ledger at all
  active -- digest folds to an OUTSTANDING obligation (must be preserved)
  base   -- directory still contains a `brain.base.*` file (else prune skips)
A directory absent from `known` is deliberately preserved as unknown, which is
the arm most likely to be holding hundreds of gigabytes here.
"""
import hashlib
import json
import os

R = "/srv/wizard/runtime/programming-integrated-20260713"
root = os.path.join(R, "deferred")
out = {}

known = set()
known_ids = {}
try:
    with open(os.path.join(R, "curriculum-deferred-intervals.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except Exception:
                continue
            iid = event.get("interval_id")
            if isinstance(iid, str) and iid:
                digest = hashlib.sha256(iid.encode("utf-8")).hexdigest()[:16]
                known.add(digest)
                known_ids[digest] = iid
except OSError as exc:
    out["ledger_error"] = str(exc)

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
except OSError:
    pass

active = {hashlib.sha256(i.encode("utf-8")).hexdigest()[:16] for i in current}
for event in current.values():
    snap = event.get("base_snapshot")
    if isinstance(snap, str) and snap:
        parent = os.path.dirname(snap)
        if os.path.dirname(parent) == root:
            active.add(os.path.basename(parent))

rows = []
for name in sorted(os.listdir(root)):
    directory = os.path.join(root, name)
    if not os.path.isdir(directory):
        continue
    base_files = []
    total_alloc = 0
    for child in os.listdir(directory):
        path = os.path.join(directory, child)
        if os.path.isfile(path) and child.startswith("brain.base."):
            st = os.stat(path)
            base_files.append(
                {"name": child, "alloc_gb": round(st.st_blocks * 512 / 1e9, 2),
                 "inode": st.st_ino, "nlink": st.st_nlink}
            )
            total_alloc += st.st_blocks * 512
    rows.append(
        {
            "digest": name,
            "known": name in known,
            "active": name in active,
            "has_base": bool(base_files),
            "alloc_gb": round(total_alloc / 1e9, 2),
            "bases": base_files,
            "interval_id": known_ids.get(name),
        }
    )

out["dirs_total"] = len(rows)
out["counts"] = {
    "known_and_active": sum(1 for r in rows if r["known"] and r["active"]),
    "known_not_active": sum(1 for r in rows if r["known"] and not r["active"]),
    "unknown": sum(1 for r in rows if not r["known"]),
    "unknown_with_base": sum(1 for r in rows if not r["known"] and r["has_base"]),
    "known_not_active_with_base": sum(
        1 for r in rows if r["known"] and not r["active"] and r["has_base"]
    ),
}

# What is each bucket actually holding? Count each inode once per bucket.
def bucket_alloc(predicate):
    seen = set()
    total = 0
    for row in rows:
        if not predicate(row):
            continue
        for base in row["bases"]:
            if base["inode"] in seen:
                continue
            seen.add(base["inode"])
            total += base["alloc_gb"]
    return round(total, 2), len(seen)

out["alloc_unknown"] = bucket_alloc(lambda r: not r["known"])
out["alloc_known_not_active"] = bucket_alloc(lambda r: r["known"] and not r["active"])
out["alloc_active"] = bucket_alloc(lambda r: r["active"])

# Inodes reachable ONLY from directories the prune may delete -- deleting these
# names actually returns blocks. An inode with any surviving name does not.
deletable = {r["digest"] for r in rows if not r["active"]}
inode_names = {}
for row in rows:
    for base in row["bases"]:
        inode_names.setdefault(base["inode"], {"dirs": [], "nlink": base["nlink"],
                                               "alloc_gb": base["alloc_gb"]})
        inode_names[base["inode"]]["dirs"].append(row["digest"])

truly_free = 0.0
anchored = 0.0
detail = []
for inode, info in inode_names.items():
    dirs = info["dirs"]
    if all(d in deletable for d in dirs) and info["nlink"] == len(dirs):
        truly_free += info["alloc_gb"]
        detail.append({"inode": inode, "alloc_gb": info["alloc_gb"],
                       "names": len(dirs), "nlink": info["nlink"], "frees": True})
    else:
        anchored += info["alloc_gb"]
        detail.append({"inode": inode, "alloc_gb": info["alloc_gb"],
                       "names": len(dirs), "nlink": info["nlink"], "frees": False,
                       "anchored_by": [d for d in dirs if d not in deletable][:4]})
out["would_truly_free_gb"] = round(truly_free, 2)
out["anchored_gb"] = round(anchored, 2)
out["inode_detail"] = sorted(detail, key=lambda d: -d["alloc_gb"])[:20]

# The blocking arm, listed explicitly.
out["unknown_dirs_with_base"] = [
    {"digest": r["digest"], "alloc_gb": r["alloc_gb"]}
    for r in rows if not r["known"] and r["has_base"]
][:40]
out["known_not_active_with_base_dirs"] = [
    {"digest": r["digest"], "alloc_gb": r["alloc_gb"], "interval_id": r["interval_id"]}
    for r in rows if r["known"] and not r["active"] and r["has_base"]
][:40]

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
