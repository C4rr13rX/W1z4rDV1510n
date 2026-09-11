set -u
# Predict the reclaim BEFORE writing 363 GB that cannot be taken back cheaply.
#
# Compaction writes its output to the same volume as its source, so the pass
# costs the compacted size up front and only returns space when the original is
# unlinked. On XFS with reflink=1 an unlink returns only the blocks no other
# file still references -- which is why deleting nine deferred directories
# holding ~560 GB of apparent size once returned 0.00 GB.
#
# So the question is not "how big is brain.wbrain" but "how many of its blocks
# does anything else point at". `fiemap` answers it directly: two files that
# share an extent report the SAME physical block for the same logical offset.
# Sampling a few ranges is enough to tell a clone from an independent copy, and
# costs seconds instead of the hours a full extent map of a 576 GB file needs.
R=/srv/wizard/runtime/programming-integrated-20260713
LIVE="$R/brain/brain.wbrain"
GUARD="$R/brain/brain.last-good.wbrain"

echo "=== sizes ==="
for f in "$LIVE" "$GUARD"; do
  stat -c '%n apparent=%s blocks512=%b nlink=%h' "$f"
done

echo "=== df ==="
df -B1 --output=avail,used,size "$R" | tail -1

echo "=== apparent total of the whole runtime tree ==="
# Sum of st_size across regular files. Compared against df used, the gap IS the
# sharing: anything the tree claims beyond what the volume actually spends is
# an extent counted more than once.
python3 - <<'PY'
import os
total = 0
count = 0
for dirpath, dirnames, filenames in os.walk("/srv/wizard/runtime/programming-integrated-20260713"):
    for name in filenames:
        p = os.path.join(dirpath, name)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        import stat as s
        if s.S_ISREG(st.st_mode):
            total += st.st_size
            count += 1
print(f"regular_files={count} apparent_total_gb={total/1e9:.2f}")
PY

echo "=== fiemap sample: do the live brain and its guard share physical blocks? ==="
# Compare the physical extent backing the same logical offset in both files.
# Identical physical block => the guard is a reflink clone of that region and
# unlinking the live file will NOT return it.
for off in 0 4294967296 17179869184 68719476736 171798691840 274877906944 412316860416; do
  a=$(xfs_io -r -c "fiemap ${off} 4096" "$LIVE" 2>/dev/null | sed -n '2p')
  b=$(xfs_io -r -c "fiemap ${off} 4096" "$GUARD" 2>/dev/null | sed -n '2p')
  echo "offset=${off}"
  echo "   live : ${a:-<none>}"
  echo "   guard: ${b:-<none>}"
done

echo "=== deferred causal bases: inode sharing ==="
find "$R/deferred" -name 'brain.base.*' -printf '%i %n %s %p\n' 2>/dev/null \
  | sort -k1,1n | awk '{printf "inode=%s nlink=%s gb=%.2f %s\n", $1, $2, $3/1e9, $4}'

echo "=== distinct inodes among those bases (what they actually cost) ==="
find "$R/deferred" -name 'brain.base.*' -printf '%i %s\n' 2>/dev/null \
  | sort -u -k1,1n | awk '{s+=$2} END {printf "distinct_inode_apparent_gb=%.2f\n", s/1e9}'
