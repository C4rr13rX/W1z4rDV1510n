R=/srv/wizard/runtime/programming-integrated-20260713
C=/srv/wizard/project/target/release/wbrain_compact
B=$R/brain/brain.wbrain

echo "=== convergence check: does the estimator settle as the sample tightens? ==="
echo "--- stride 5 ---"
timeout 1500 "$C" --estimate "$B" 5 2>&1 | tail -3
echo "rc=$?"

echo
echo "=== launching stride 1: every live slot, so this is the live set itself"
echo "=== and not an extrapolation. Detached; poll /tmp/live_exact.json ==="
pkill -f "wbrain_compact --estimate $B 1$" 2>/dev/null || true
nohup sh -c "$C --estimate $B 1 > /tmp/live_exact.json 2>/tmp/live_exact.err" \
  >/dev/null 2>&1 &
echo "launched pid $!"
