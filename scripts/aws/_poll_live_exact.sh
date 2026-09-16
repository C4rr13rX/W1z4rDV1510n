echo "--- /tmp/live_exact.json ---"
cat /tmp/live_exact.json 2>/dev/null || echo "(not written yet)"
echo
echo "--- stderr ---"
cat /tmp/live_exact.err 2>/dev/null | tail -3 || true
echo
pgrep -f "wbrain_compact --estimate" >/dev/null && echo "estimate: running" \
  || echo "estimate: finished-or-dead"
