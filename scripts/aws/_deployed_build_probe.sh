R=/srv/wizard/runtime/programming-integrated-20260713
echo "--- running brain processes ---"
ps -eo pid,lstart,etimes,rss,args --sort=-rss 2>/dev/null | grep -i 'brain\|node' | grep -v grep | head -8
echo "--- listener on 18095 ---"
ss -lntp 2>/dev/null | grep 18095
echo "--- exe of listener ---"
pid=$(ss -lntpH 2>/dev/null | grep 18095 | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
echo "listener pid=[$pid]"
if [ -n "$pid" ]; then
  target=$(readlink -f /proc/$pid/exe 2>/dev/null)
  echo "exe=$target"
  stat -c 'mtime=%y size=%s' "$target" 2>/dev/null
  echo "--- marker strings ---"
  strings -a "$target" 2>/dev/null | grep -aF 'largest satisfying' | head -3
  echo "largest_satisfying_hits=$(strings -a "$target" 2>/dev/null | grep -acF 'largest satisfying')"
fi
echo "--- candidate binaries by mtime ---"
find /srv/wizard -maxdepth 4 -type f -name '*node*' -size +5M -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null | sort -r | head -10
echo "--- host checkout ---"
for d in /srv/wizard/src /srv/wizard/repo /srv/wizard/W1z4rDV1510n /srv/wizard/runtime/src; do
  if [ -d "$d/.git" ]; then echo "$d: $(git -C "$d" log -1 --format='%h %ad %s' --date=iso 2>/dev/null)"; fi
done
echo "done"
