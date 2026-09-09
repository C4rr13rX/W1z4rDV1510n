set -u
# Is the running brain executing the binary that is on disk RIGHT NOW?
#
# mtime arithmetic cannot answer this. Measured 2026-09-09, the brain started
# 12 s before cargo relinked, so the lag was +12 -- under any threshold tuned
# for the supervisor, yet the process was serving the previous image and the
# canonical polyglot row still composed ledger.go.
#
# The inode is exact and needs no threshold: cargo relinks by creating a new
# file, so a process holding the old image keeps the old inode (and Linux
# marks the unlinked image "(deleted)").
BIN=/srv/wizard/project/target/release/w1z4rd_brain_server
echo "--- on-disk binary ---"
stat -c 'inode=%i mtime=%y size=%s' "$BIN"
echo "--- running brain (anchored, so the supervisor is not matched) ---"
for pid in $(pgrep -f 'release/w1z4rd_brain_server$'); do
  echo "pid=$pid started=$(ps -o lstart= -p "$pid") etimes=$(ps -o etimes= -p "$pid")"
  echo "  exe_link=$(readlink /proc/$pid/exe)"
  echo "  exe_inode=$(stat -Lc %i /proc/$pid/exe 2>/dev/null)"
  echo "  rss_kb=$(ps -o rss= -p "$pid")"
done
echo "--- what pgrep WITHOUT the anchor would have matched ---"
pgrep -af 'release/w1z4rd_brain_server' | head -5
echo "--- replay worker in flight? (a restart under it costs the pass) ---"
pgrep -af 'drive_corpora_brain' | head -3
echo "worker_count=$(pgrep -c -f drive_corpora_brain 2>/dev/null || echo 0)"
echo "--- supervisor state / resume row ---"
cat /srv/wizard/runtime/programming-integrated-20260713/curriculum-supervisor.status.json 2>/dev/null | head -c 300
echo
echo "--- memory (a recycle relaunches the brain by itself) ---"
free -g | head -2
echo done
