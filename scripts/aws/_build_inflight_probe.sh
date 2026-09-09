set -u
# Is a rebuild already running, and is it safe to start one if not?
#
# Two agents share this host. Starting a second release build would contend
# for the ~3 GB of headroom the replay worker needs and could push the brain
# into the memory guard, so ASK before acting.
echo "--- cargo/rustc processes ---"
ps -eo pid,etimes,rss,args 2>/dev/null | grep -E 'cargo|rustc' | grep -v grep | head -10
echo "cargo_count=$(pgrep -c -f 'cargo (build|test)' 2>/dev/null || echo 0)"
echo "rustc_count=$(pgrep -c -f rustc 2>/dev/null || echo 0)"
echo "--- brain binary vs newest rust source ---"
BIN=/srv/wizard/project/target/release/w1z4rd_brain_server
stat -c 'binary  mtime=%y size=%s' "$BIN" 2>/dev/null
find /srv/wizard/project/crates -name '*.rs' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null \
  | sort -r | head -3
echo "--- does host source carry the fix now? ---"
f=/srv/wizard/project/crates/node/src/brain_api.rs
echo "  selection_behaviour_coverage=$(grep -c 'fn selection_behaviour_coverage' "$f" 2>/dev/null)"
echo "  servable=$(grep -c 'let servable' "$f" 2>/dev/null)"
echo "  sha=$(sha256sum "$f" 2>/dev/null | cut -c1-16)"
echo "--- replay marker (restart cost) ---"
cat /srv/wizard/runtime/programming-integrated-20260713/deferred-replay-active.json 2>/dev/null | head -c 400
echo
echo "--- supervisor state ---"
cat /srv/wizard/runtime/programming-integrated-20260713/curriculum-supervisor.status.json 2>/dev/null | head -c 300
echo
echo "--- memory / load ---"
free -g
uptime
echo "--- disk ---"
df -h /srv | tail -1
echo done
