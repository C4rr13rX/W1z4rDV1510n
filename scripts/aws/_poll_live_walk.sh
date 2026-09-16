cat /tmp/live_walk.json 2>/dev/null || echo "{}"
echo
pgrep -f /tmp/live_walk.py >/dev/null && echo "walk: running" || echo "walk: finished-or-dead"
tail -3 /tmp/live_walk.log 2>/dev/null || true
