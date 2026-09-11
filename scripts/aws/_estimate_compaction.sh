set -u
# Ship the sampled estimator, then ask what a compaction would produce.
#
# The output goes on the SAME volume as the source, so a pass that produces
# more than the free space fills the disk this repair exists to save. The
# estimate is checked against `df` before anything is written.
cd /srv/wizard/project || exit 1
mkdir -p /tmp/wizard-deploy
cat > /tmp/wizard-deploy/inc.patch <<'WIZARD_PATCH_EOF'
diff --git a/crates/brain/src/bin/wbrain_compact.rs b/crates/brain/src/bin/wbrain_compact.rs
index 3f83832..4c656b6 100644
--- a/crates/brain/src/bin/wbrain_compact.rs
+++ b/crates/brain/src/bin/wbrain_compact.rs
@@ -90,6 +90,44 @@ fn main() -> ExitCode {
         return usage();
     }
 
+    if args[0] == "--estimate" {
+        if args.len() != 2 && args.len() != 3 {
+            return usage();
+        }
+        let stride: u64 = if args.len() == 3 {
+            match args[2].parse() {
+                Ok(value) => value,
+                Err(_) => return usage(),
+            }
+        } else {
+            1000
+        };
+        return match compaction::estimate(Path::new(&args[1]), stride) {
+            Ok(report) => {
+                println!(
+                    "{}",
+                    serde_json::json!({
+                        "live_neurons": report.live_neurons,
+                        "sampled_neurons": report.sampled_neurons,
+                        "sampled_body_bytes": report.sampled_body_bytes,
+                        "estimated_live_bytes": report.estimated_live_bytes,
+                        "estimated_live_gb":
+                            (report.estimated_live_bytes as f64 / 1e9 * 100.0).round() / 100.0,
+                        "source_bytes": report.source_bytes,
+                        "source_gb": (report.source_bytes as f64 / 1e9 * 100.0).round() / 100.0,
+                        "stride": stride,
+                        "note": "sampled estimate, not a bound; leave margin",
+                    })
+                );
+                ExitCode::SUCCESS
+            }
+            Err(error) => {
+                eprintln!("estimate failed: {error}");
+                ExitCode::FAILURE
+            }
+        };
+    }
+
     if args[0] == "--inspect" {
         if args.len() != 2 {
             return usage();
diff --git a/crates/brain/src/store/compaction.rs b/crates/brain/src/store/compaction.rs
index fa11138..2387e62 100644
--- a/crates/brain/src/store/compaction.rs
+++ b/crates/brain/src/store/compaction.rs
@@ -198,6 +198,56 @@ pub fn verify(path: &Path, expected: &CompactionReport) -> io::Result<u64> {
     Ok(checked)
 }
 
+/// What a compaction pass would produce, without producing it.
+#[derive(Debug, Clone, Copy, Default)]
+pub struct LiveEstimate {
+    pub live_neurons: u64,
+    pub sampled_neurons: u64,
+    pub sampled_body_bytes: u64,
+    pub estimated_live_bytes: u64,
+    pub source_bytes: u64,
+}
+
+/// Estimate the compacted size by sampling live record headers.
+///
+/// Reading all 5M headers would be ~5M random seeks across a 576 GB file —
+/// IOPS-bound at roughly half an hour on network storage, which is too slow to
+/// answer "will the output fit?" before committing to a pass. Sampling one slot
+/// in `stride` answers it in seconds.
+///
+/// This is deliberately an ESTIMATE and is reported as one: body sizes are not
+/// uniform, so the caller must leave margin rather than treat the number as a
+/// bound. It exists to catch the case where compaction cannot possibly fit, not
+/// to prove that it will.
+pub fn estimate(path: &Path, stride: u64) -> io::Result<LiveEstimate> {
+    let stride = stride.max(1);
+    let mut container = BrainContainer::open(path)?;
+    let manifest = container.manifest().cloned().ok_or_else(|| {
+        io::Error::new(io::ErrorKind::InvalidData, "container has no manifest")
+    })?;
+    let mut report = LiveEstimate {
+        source_bytes: container.byte_len()?,
+        ..Default::default()
+    };
+    for pool in &manifest.pools {
+        let offsets = live_offsets(&mut container, pool)?;
+        report.live_neurons += offsets.len() as u64;
+        for (index, (_id, offset)) in offsets.iter().enumerate() {
+            if index as u64 % stride != 0 {
+                continue;
+            }
+            report.sampled_body_bytes += container.record_body_len(*offset)?;
+            report.sampled_neurons += 1;
+        }
+    }
+    if report.sampled_neurons > 0 {
+        let mean = report.sampled_body_bytes as f64 / report.sampled_neurons as f64;
+        let per_record = mean + BrainContainer::RECORD_HEADER_BYTES as f64;
+        report.estimated_live_bytes = (per_record * report.live_neurons as f64) as u64;
+    }
+    Ok(report)
+}
+
 /// Live `(neuron id, offset)` pairs for one pool, from whichever addressing
 /// form its manifest uses.
 fn live_offsets(
diff --git a/crates/brain/src/store/container.rs b/crates/brain/src/store/container.rs
index e7cfa6d..ccbc39f 100644
--- a/crates/brain/src/store/container.rs
+++ b/crates/brain/src/store/container.rs
@@ -498,6 +498,14 @@ impl BrainContainer {
             .map(|(new_offset, _len)| new_offset)
     }
 
+    /// Body length recorded in a record header, without reading the body.
+    pub(crate) fn record_body_len(&mut self, offset: u64) -> io::Result<u64> {
+        self.file.seek(SeekFrom::Start(offset))?;
+        let mut header = [0_u8; Self::RECORD_HEADER_BYTES as usize];
+        self.file.read_exact(&mut header)?;
+        Ok(u64::from_le_bytes(header[16..24].try_into().unwrap()))
+    }
+
     /// Pool and kind fields of an auxiliary record header.
     pub(crate) fn auxiliary_header(
         &mut self,
WIZARD_PATCH_EOF

echo "patch_bytes=$(stat -c %s /tmp/wizard-deploy/inc.patch)"
echo "--- check ---"
git apply --check -p1 /tmp/wizard-deploy/inc.patch 2>&1 | tail -10
echo "check_rc=${PIPESTATUS[0]}"
git apply -p1 /tmp/wizard-deploy/inc.patch 2>&1 | tail -10
echo "apply_rc=${PIPESTATUS[0]}"
chown -R ec2-user:ec2-user crates 2>/dev/null || true

export CARGO_HOME=${CARGO_HOME:-/srv/wizard/.cargo}
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
timeout 3000 sudo -u ec2-user \
  env CARGO_HOME="$CARGO_HOME" CARGO_BUILD_JOBS=1 \
  cargo build --release --offline \
  -p w1z4rd-brain --bin wbrain_compact 2>&1 | tail -6
echo "cargo_rc=${PIPESTATUS[0]}"

R=/srv/wizard/runtime/programming-integrated-20260713
BIN=target/release/wbrain_compact
echo "--- estimate: live brain ---"
time sudo -u ec2-user "$BIN" --estimate "$R/brain/brain.wbrain" 2000 2>&1
echo "--- estimate: last-good guard ---"
time sudo -u ec2-user "$BIN" --estimate "$R/brain/brain.last-good.wbrain" 2000 2>&1
echo "--- free bytes ---"
df -B1 --output=avail "$R" | tail -1
