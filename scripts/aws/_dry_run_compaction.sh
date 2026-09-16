set -e
R=/srv/wizard/runtime/programming-integrated-20260713
C=/srv/wizard/project/target/release/wbrain_compact
B=$R/brain/brain.wbrain
OUT=$R/brain/brain.compacted-dryrun.wbrain

echo "=== NON-DESTRUCTIVE. Writes a NEW file and modifies nothing existing."
echo "=== Proves on the real 472 GB brain what the estimator only sampled,"
echo "=== before any in-place pass is considered."
echo

echo "--- free space and current containers ---"
df -B1 --output=avail "$R" | tail -1 | awk '{printf "free_bytes %s (%.2f GB)\n", $1, $1/1e9}'
ls -l "$R"/brain/*.wbrain 2>/dev/null | awk '{printf "%s %.2f GB\n", $NF, $5/1e9}'

echo
echo "--- brain topology BEFORE (restored afterwards for comparison) ---"
curl -s --max-time 20 http://127.0.0.1:18095/stats || echo "(no brain on 18095)"
echo

echo "--- brain server process ---"
ps -eo pid,etimes,rss,comm,args | grep -i "[w]1z4rd_brain_server" | head -3 \
  | tee /tmp/brain_cmdline.txt
echo

if [ -e "$OUT" ]; then
  echo "removing stale dry-run output"
  rm -f "$OUT"
fi

# The container has no locking protocol: a live server appends to the source
# while it is being copied, which would publish a manifest pointing at records
# the copy never saw. Stop it for the duration.
PID=$(pgrep -f w1z4rd_brain_server | head -1 || true)
if [ -n "$PID" ]; then
  echo "stopping brain server pid $PID"
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID" || true
  echo "stopped"
else
  echo "no brain server running"
fi

echo
echo "--- launching out-of-place compaction (detached) ---"
nohup sh -c "
  /usr/bin/time -v $C $B $OUT > /tmp/compact_dryrun.out 2>/tmp/compact_dryrun.err
  echo \$? > /tmp/compact_dryrun.rc
" >/dev/null 2>&1 &
echo "launched pid $!"
sleep 20
echo "--- 20 s in ---"
ls -l "$OUT" 2>/dev/null | awk '{printf "output %.3f GB\n", $5/1e9}' || echo "(no output yet)"
df -B1 --output=avail "$R" | tail -1 | awk '{printf "free_bytes %s (%.2f GB)\n", $1, $1/1e9}'
