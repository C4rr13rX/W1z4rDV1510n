set -u
cd /srv/wizard/project || exit 1
mkdir -p /tmp/wizard-deploy
cat > /tmp/wizard-deploy/floor.patch <<'WIZARD_PATCH_EOF'
diff --git a/scripts/aws/run_programming_curriculum_service.sh b/scripts/aws/run_programming_curriculum_service.sh
index 1ea976f..63bfc22 100644
--- a/scripts/aws/run_programming_curriculum_service.sh
+++ b/scripts/aws/run_programming_curriculum_service.sh
@@ -172,7 +172,23 @@ common=(
   # before interval_recall and the behavioural gate could run: zero
   # admissions since 2026-08-22 despite eight clean yield/recycle cycles.
   --replay-rows-per-pass 49152
-  --min-free-disk-gb 8
+  # Sized from the MEASURED burn, not from a round number. The `.wbrain` store
+  # appends a full neuron body on every sleep -- mean body 71 KB across
+  # 5,086,800 live neurons -- and consumes 112-257 GB/h while a replay runs.
+  # An 8 GB floor is 112 seconds of warning at the high end, and the guard
+  # needs three consecutive breaches AT A DURABLE BOUNDARY to act, so it lost
+  # that race: the volume filled, the wrapper died writing its 6-byte node.pid,
+  # and systemd restarted it 115 times, which reads as a finished stage rather
+  # than as a full disk. 150 GB is 35-80 minutes at the measured rates, which
+  # is many commit periods and leaves room for a clean cooperative yield.
+  #
+  # This buys a SAFE STOP, not headroom. Compaction cannot create headroom
+  # here: the guard and every causal base are reflink clones of the live
+  # container (verified by fiemap -- identical physical blocks at 6 of 7 sampled
+  # offsets), so a compacted copy writes 363 GB of unshareable blocks to reclaim
+  # 153 GB of unshared ones. The tree reports 7,417 GB apparent against 596 GB
+  # used; only `df` measures anything here.
+  --min-free-disk-gb 150
   --max-restarts 10
 )
 
WIZARD_PATCH_EOF
echo "--- apply floor patch ---"
git apply --check -p1 /tmp/wizard-deploy/floor.patch 2>&1 | tail -5
echo "check_rc=${PIPESTATUS[0]}"
git apply -p1 /tmp/wizard-deploy/floor.patch 2>&1 | tail -5
echo "apply_rc=${PIPESTATUS[0]}"
chown -R ec2-user:ec2-user scripts 2>/dev/null || true
echo "--- deployed value ---"
grep -n "min-free-disk-gb" scripts/aws/run_programming_curriculum_service.sh

R=/srv/wizard/runtime/programming-integrated-20260713
echo "=== final state ==="
systemctl show -p ActiveState -p SubState -p NRestarts wizard-curriculum-supervisor
echo "wrapper=$(pgrep -fc 'run_programming_curriculum_service.sh' || echo 0)"
echo "supervisor=$(pgrep -fc 'curriculum_supervisor' || echo 0)"
echo "worker=$(pgrep -fc 'tools.training_standard.drive_corpora_brain' || echo 0)"
echo "brain_server=$(pgrep -f 'release/w1z4rd_brain_server$' | head -1)"
df -B1 --output=avail,pcent "$R" | tail -1
stat -c 'wbrain_gb=%s' "$R/brain/brain.wbrain" | awk -F= '{printf "wbrain_gb=%.2f\n", $2/1e9}'
echo "--- 60s burn recheck (should be ~0 with the curriculum stopped) ---"
a=$(df -B1 --output=avail "$R" | tail -1); sleep 60; b=$(df -B1 --output=avail "$R" | tail -1)
echo "delta_bytes=$((a-b))"
echo "--- compactor binary present ---"
ls -la target/release/wbrain_compact 2>&1 | awk '{print $5, $9}'
