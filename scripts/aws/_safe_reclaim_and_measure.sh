set -u
R=/srv/wizard/runtime/programming-integrated-20260713
P=/srv/wizard/project

echo "=== df BEFORE ==="
df -h /srv/wizard

# The 85 "unknown" deferred directories are all hardlink names for ONE inode
# whose st_nlink (85) exactly equals the number of names found under deferred/,
# so no other path in the runtime references it. They are unknown to the CURRENT
# ledger because the ledger was rotated on 2026-08-22 (the .bak file beside it);
# confirm against that backup before removing, so "unknown" is not being used as
# a synonym for "unreferenced".
python3 - <<'PY'
import hashlib, json, os
R = "/srv/wizard/runtime/programming-integrated-20260713"
bak = None
for name in os.listdir(R):
    if name.startswith("curriculum-deferred-intervals.jsonl.bak"):
        bak = os.path.join(R, name)
known_bak, resolved_bak = set(), set()
if bak:
    with open(bak, encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            iid = ev.get("interval_id")
            if not isinstance(iid, str) or not iid:
                continue
            d = hashlib.sha256(iid.encode()).hexdigest()[:16]
            known_bak.add(d)
            if ev.get("status") == "resolved":
                resolved_bak.add(d)
print(f"BACKUP_LEDGER {bak}")
print(f"BACKUP_KNOWN {len(known_bak)} BACKUP_RESOLVED {len(resolved_bak)}")

# Current obligations, folded the same way the supervisor folds them.
cur = {}
with open(os.path.join(R, "curriculum-deferred-intervals.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        iid = ev.get("interval_id")
        if not isinstance(iid, str) or not iid:
            continue
        if ev.get("status") == "resolved":
            cur.pop(iid, None)
        elif ev.get("status") == "deferred":
            cur[iid] = ev
active = {hashlib.sha256(i.encode()).hexdigest()[:16] for i in cur}
for ev in cur.values():
    s = ev.get("base_snapshot")
    if isinstance(s, str) and s:
        active.add(os.path.basename(os.path.dirname(s)))

root = os.path.join(R, "deferred")
on_disk = {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
unknown = sorted(on_disk - active)
covered = [d for d in unknown if d in resolved_bak]
uncovered = [d for d in unknown if d not in known_bak]
print(f"UNKNOWN_DIRS {len(unknown)}")
print(f"RESOLVED_IN_BACKUP {len(covered)}")
print(f"IN_NEITHER_LEDGER {len(uncovered)}")
with open("/tmp/prunable_digests.txt", "w") as fh:
    for d in covered:
        fh.write(d + "\n")
PY

echo "=== removing directories confirmed RESOLVED in the rotated ledger ==="
count=0
while read -r digest; do
  [ -n "${digest}" ] || continue
  case "${digest}" in */*|.|..) echo "refusing ${digest}"; continue;; esac
  target="${R}/deferred/${digest}"
  [ -d "${target}" ] || continue
  rm -rf "${target}"
  count=$((count + 1))
done </tmp/prunable_digests.txt
echo "REMOVED_DIRS ${count}"
df -h /srv/wizard

echo "=== rebuildable build artifacts ==="
du -sm "${P}/target/debug" 2>/dev/null || true
rm -rf "${P}/target/debug"
rm -rf "${P}/runtime/benchmark-tool-cache"
rm -rf /srv/wizard/staging
echo "ARTIFACTS_REMOVED"
df -h /srv/wizard

echo "=== stale per-interval evidence and rotated ledger backup ==="
find "${R}" -maxdepth 1 -name '*.bak.*' -type f -printf '%s %f\n' -delete 2>/dev/null || true
find "${R}" -maxdepth 1 -name 'deferred-replay-*.admission.json' -mtime +14 -delete 2>/dev/null || true
find "${R}" -maxdepth 1 -name '*.progress.slow-batches.jsonl' -mtime +7 -delete 2>/dev/null || true
df -h /srv/wizard

echo "=== what still holds the volume ==="
python3 - <<'PY'
import os
R = "/srv/wizard/runtime/programming-integrated-20260713"
rows = []
for dirpath, _dirs, files in os.walk(R):
    for name in files:
        path = os.path.join(dirpath, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if st.st_blocks * 512 < 1e9:
            continue
        rows.append((st.st_blocks * 512, st.st_nlink, st.st_ino, path[len(R)+1:]))
seen, total = set(), 0
rows.sort(reverse=True)
for size, nlink, ino, path in rows:
    if ino in seen:
        continue
    seen.add(ino)
    total += size
    print(f"  {size/1e9:9.2f} GB  nlink={nlink}  {path}")
print(f"UNIQUE_INODE_TOTAL_GB {total/1e9:.2f}")
PY

chown -R ec2-user:ec2-user "${R}" 2>/dev/null || true
echo "=== df FINAL ==="
df -h /srv/wizard
