set -u
cd /srv/wizard/project
echo "=== BEFORE ==="
ls -i --time-style=full-iso -l target/release/w1z4rd_brain_server 2>&1
df -h /srv/wizard | tail -1

# `--bin` alone searches only default-run packages; the brain server lives in
# the node package. Name the package explicitly.
echo "=== BUILD ==="
sudo -u ec2-user env PATH="/home/ec2-user/.cargo/bin:$PATH" \
  cargo build --release -p w1z4rdv1510n-node --bin w1z4rd_brain_server 2>&1 | tail -25

echo "=== AFTER ==="
ls -i --time-style=full-iso -l target/release/w1z4rd_brain_server 2>&1
# DEPLOY IS NOT LOAD. A binary that was copied but never rebuilt runs the old
# code, and a capability verdict from such a build is worthless -- measured
# 850 s for a Python fix and 96 h for a Rust one. The compiled-in string is the
# only evidence that the fix is actually in THIS file.
echo "=== FIX PRESENT IN BINARY ==="
if strings target/release/w1z4rd_brain_server 2>/dev/null \
    | grep -q "persisted neuron must still occupy its live slot"; then
  echo "store_symbols: present"
else
  echo "store_symbols: MISSING"
fi
df -h /srv/wizard | tail -1
