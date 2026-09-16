R=/srv/wizard/runtime/programming-integrated-20260713
OUT=$R/brain/brain.compacted-dryrun.wbrain
echo "--- output size ---"
ls -l "$OUT" 2>/dev/null | awk '{printf "%.3f GB\n", $5/1e9}' || echo "(none)"
echo "--- free ---"
df -B1 --output=avail "$R" | tail -1 | awk '{printf "%.2f GB\n", $1/1e9}'
echo "--- running? ---"
pgrep -f "wbrain_compact .*brain.wbrain" >/dev/null && echo "compaction: running" \
  || echo "compaction: finished-or-dead"
echo "--- rc ---"
cat /tmp/compact_dryrun.rc 2>/dev/null || echo "(not finished)"
echo "--- stdout ---"
tail -6 /tmp/compact_dryrun.out 2>/dev/null || true
echo "--- stderr tail ---"
tail -12 /tmp/compact_dryrun.err 2>/dev/null || true
echo "--- exact stride-1 estimate ---"
cat /tmp/live_exact.json 2>/dev/null || echo "(still running)"
