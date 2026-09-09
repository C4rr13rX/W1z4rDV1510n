set -u
# Rebuild the brain server on a host whose brain already holds 11 of 15 GB.
#
# Three hazards, each previously paid for:
#
#   - SSM writes land root:root and the supervisor runs as ec2-user
#     (`ssm_root_ownership_trap`), so ownership is restored before anything
#     reads the tree.
#   - The kernel's OOM killer picks the largest RSS, which is the brain. This
#     raises the build's own oom_score_adj so a squeeze kills the compiler,
#     which is restartable, rather than the brain, which is not.
#   - Cargo hard-links from its cache, so a "Finished in 0.30s" build that
#     leaves the binary mtime unchanged has NOT relinked
#     (PROGRAMMING_BRAIN_OPERATIONS, "two probe traps"). The mtime and inode
#     are printed before and after for exactly that reason.
#
# The supervisor is deliberately NOT restarted: `deferred-replay-active.json`
# is state:training, and a restart would roll the whole interval back to
# start_row. The brain server is a separate process that the supervisor
# relaunches at every memory recycle -- roughly every 40 minutes here -- so
# swapping the binary is picked up at the next natural boundary at no cost.
cd /srv/wizard/project || exit 1

chown -R ec2-user:ec2-user crates/node/src/brain_api.rs 2>/dev/null || true

BIN=target/release/w1z4rd_brain_server
echo "--- before ---"
ls -la --time-style=full-iso "$BIN" 2>&1
stat -c 'inode=%i links=%h' "$BIN" 2>&1
free -g | head -2

echo "--- building ---"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
# Touch the source so cargo cannot decide the crate is fresh and skip relinking.
touch crates/node/src/brain_api.rs
export CARGO_HOME=${CARGO_HOME:-/srv/wizard/.cargo}
export CARGO_BUILD_JOBS=1
# `-p` is required: the workspace has no default-run package, so
# `--bin w1z4rd_brain_server` alone resolves against the wrong set and exits
# without building anything. And `$?` after a pipe is the status of `tail`,
# which reported success for a build that never ran -- read PIPESTATUS.
timeout 3000 sudo -u ec2-user \
  env CARGO_HOME="$CARGO_HOME" CARGO_BUILD_JOBS=1 \
  cargo build --release --offline \
  -p w1z4rdv1510n-node --bin w1z4rd_brain_server 2>&1 | tail -25
echo "cargo_rc=${PIPESTATUS[0]}"

echo "--- after ---"
ls -la --time-style=full-iso "$BIN" 2>&1
stat -c 'inode=%i links=%h' "$BIN" 2>&1
free -g | head -2
echo "--- brain still up? ---"
pgrep -f "release/w1z4rd_brain_server$" | head -1 | xargs -r -I{} ps -o pid=,etimes=,rss= -p {}
echo "--- replay still advancing? ---"
ls -t /srv/wizard/runtime/programming-integrated-20260713/deferred-replay-*.progress.json \
  2>/dev/null | head -1 | xargs -r python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print({k:d.get(k) for k in ('durable_next_row','accepted_episodes','batch_seconds_ema')})
"
