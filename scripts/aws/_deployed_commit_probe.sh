R=/srv/wizard/runtime/programming-integrated-20260713
echo "--- host checkouts ---"
for d in /srv/wizard/project /srv/wizard/src /srv/wizard/repo /srv/wizard/W1z4rDV1510n; do
  if [ -d "$d/.git" ]; then
    echo "$d HEAD: $(git -C "$d" log -1 --format='%h %ad %s' --date=iso 2>/dev/null)"
  fi
done
echo "--- brain_api.rs on host: does it carry the servable-coverage fix? ---"
for f in /srv/wizard/project/crates/node/src/brain_api.rs /srv/wizard/src/crates/node/src/brain_api.rs; do
  if [ -f "$f" ]; then
    echo "$f"
    stat -c '  mtime=%y size=%s' "$f"
    echo "  selection_behaviour_coverage=$(grep -c 'fn selection_behaviour_coverage' "$f")"
    echo "  servable_block=$(grep -c 'let servable' "$f")"
    echo "  behaviour_query_frame=$(grep -c 'fn behaviour_query_frame' "$f")"
  fi
done
echo "--- listener on 18095 ---"
ss -lntp 2>/dev/null | grep 18095
pid=$(ss -lntpH 2>/dev/null | grep 18095 | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
echo "listener pid=[$pid]"
if [ -n "$pid" ]; then
  echo "  started: $(ps -o lstart= -p $pid 2>/dev/null)"
  echo "  etimes:  $(ps -o etimes= -p $pid 2>/dev/null)"
  target=$(readlink -f /proc/$pid/exe 2>/dev/null)
  echo "  exe=$target"
  stat -c '  exe mtime=%y size=%s' "$target" 2>/dev/null
  # A comment never reaches the binary; these are real string literals emitted
  # by the diagnostics the fix travels with, so they DO prove which build runs.
  echo "  str[manifest_composition_ready]=$(strings -a "$target" 2>/dev/null | grep -acF 'manifest_composition_ready')"
  echo "  str[component_recall]=$(strings -a "$target" 2>/dev/null | grep -acF 'component_recall')"
fi
echo "--- any newer built binary the running process is NOT using ---"
find /srv/wizard -maxdepth 6 -type f \( -name 'w1z4rd_brain_server' -o -name '*brain_server*' \) -size +5M \
  -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null | sort -r | head -8
echo "done"
