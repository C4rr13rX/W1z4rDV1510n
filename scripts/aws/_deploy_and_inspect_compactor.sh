set -u
# Ship the compaction pass and READ the container before rewriting it.
#
# Deliberately two steps. `--inspect` opens the container, prints the manifest
# shape and exits without writing, so the addressing form of the real brain is
# known before a pass touches 576 GB of it. "Deploy is not load" applies to a
# tool as much as to a server: cargo hard-links from its cache, so a build that
# reports success while leaving the binary's inode unchanged has not relinked.
# The inode and mtime are printed before and after for that reason.
#
# SSM writes land root:root and the supervisor runs as ec2-user, so ownership
# is restored before cargo reads the tree. The build raises its own
# oom_score_adj: the kernel picks the largest RSS, and the brain server is
# still up.
cd /srv/wizard/project || exit 1

echo "--- repo before ---"
git rev-parse HEAD 2>&1
git status --porcelain 2>&1 | head -10

chown -R ec2-user:ec2-user .git crates scripts 2>/dev/null || true

echo "--- pull ---"
sudo -u ec2-user git fetch --quiet origin 2>&1 | tail -5
sudo -u ec2-user git -c advice.detachedHead=false merge --ff-only origin/main 2>&1 | tail -5
echo "head_now=$(git rev-parse HEAD)"

BIN=target/release/wbrain_compact
echo "--- binary before ---"
stat -c 'inode=%i mtime=%y size=%s' "$BIN" 2>&1

echo "--- building ---"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
export CARGO_HOME=${CARGO_HOME:-/srv/wizard/.cargo}
timeout 3000 sudo -u ec2-user \
  env CARGO_HOME="$CARGO_HOME" CARGO_BUILD_JOBS=1 \
  cargo build --release --offline \
  -p w1z4rd-brain --bin wbrain_compact 2>&1 | tail -20
echo "cargo_rc=${PIPESTATUS[0]}"

echo "--- binary after ---"
stat -c 'inode=%i mtime=%y size=%s' "$BIN" 2>&1

echo "--- container inspect (read-only) ---"
R=/srv/wizard/runtime/programming-integrated-20260713
sudo -u ec2-user "$BIN" --inspect "$R/brain/brain.wbrain" 2>&1

echo "--- volume ---"
df -B1 --output=avail,size,pcent "$R" | tail -1
echo "--- brain server still up? ---"
pgrep -f "release/w1z4rd_brain_server$" | head -1 | xargs -r ps -o pid=,etimes=,rss= -p
echo "--- curriculum census (expected 0: stopped earlier this session) ---"
pgrep -fc "run_programming_curriculum_service.sh" || echo 0
pgrep -fc "tools.training_standard.drive_corpora_brain" || echo 0
