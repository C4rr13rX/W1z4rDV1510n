set -u
# Read-only reconnaissance before rebuilding the brain server on a host that
# is mid-replay. Two things make a careless deploy expensive here:
#
#   - `deferred-replay-active.json` with state:training means restarting the
#     SUPERVISOR discards the whole interval, back to start_row. The brain
#     server is a different process and is relaunched at every memory recycle,
#     so a binary swap needs no supervisor restart at all.
#   - the host has ~15 GB and the brain holds ~11 GB of it, so a release
#     build wants a window right after a recycle rather than just before one.
echo "--- project ---"
cd /srv/wizard/project 2>/dev/null || { echo "NO PROJECT DIR"; exit 1; }
git rev-parse --short HEAD 2>&1
git status --short 2>&1 | head -20
echo "--- branch/remote ---"
git rev-parse --abbrev-ref HEAD 2>&1
git remote -v 2>&1 | head -2
echo "--- binary ---"
ls -la target/release/w1z4rd_brain_server 2>&1
echo "--- replay marker ---"
cat /srv/wizard/runtime/programming-integrated-20260713/deferred-replay-active.json 2>&1 | head -c 600
echo
echo "--- progress ---"
ls -t /srv/wizard/runtime/programming-integrated-20260713/deferred-replay-*.progress.json 2>/dev/null | head -1 | xargs -r cat | head -c 400
echo
echo "--- memory ---"
free -g
echo "--- brain age ---"
pgrep -f "release/w1z4rd_brain_server$" | head -1 | xargs -r -I{} ps -o pid=,etimes=,rss= -p {}
echo "--- supervisor ---"
systemctl is-active wizard-curriculum-supervisor 2>&1 || true
echo "--- cargo ---"
which cargo 2>&1; cargo --version 2>&1 | head -1
echo "--- disk ---"
df -h /srv | tail -1
