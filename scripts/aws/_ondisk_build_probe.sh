B=/srv/wizard/project/target/release/w1z4rd_brain_server
S=/srv/wizard/project/crates/node/src/brain_api.rs
echo "--- on-disk binary ---"
stat -c 'mtime_unix=%Y mtime=%y size=%s' "$B" 2>/dev/null || echo "MISSING"
echo "--- source on host ---"
stat -c 'mtime_unix=%Y mtime=%y' "$S" 2>/dev/null
echo "--- does host source carry the provenance-coverage fix? ---"
grep -c 'selection_behaviour_coverage' "$S" 2>/dev/null
grep -n 'fn selection_behaviour_coverage\|component_behaviour_routes' "$S" 2>/dev/null | head -4
echo "--- is a build running now? ---"
pgrep -af 'cargo|rustc' 2>/dev/null | head -5 || echo "no build running"
echo "--- serving pid vs disk inode ---"
pid=$(ss -lntpH 2>/dev/null | grep 18095 | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
echo "pid=$pid"
[ -n "$pid" ] && readlink /proc/$pid/exe
[ -n "$pid" ] && stat -c 'running_inode=%i' /proc/$pid/exe 2>/dev/null
stat -c 'disk_inode=%i' "$B" 2>/dev/null
echo done
