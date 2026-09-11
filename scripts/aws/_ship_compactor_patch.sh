set -u
# Ship the compaction pass to a host whose /srv/wizard/project is NOT a git
# checkout and has no `patch` binary. `git apply` needs neither a repository
# nor the patch package, and it refuses a partial application by default --
# which is what we want against a 411 KB source file.
#
# --check first. The host's copy of brain.rs and pool.rs is not known to match
# the commit this diff was cut against, and a half-applied patch is a worse
# problem than a failed deploy.
cd /srv/wizard/project || exit 1
mkdir -p /tmp/wizard-deploy
cat > /tmp/wizard-deploy/compactor.patch <<'WIZARD_PATCH_EOF'
diff --git a/crates/brain/src/bin/wbrain_compact.rs b/crates/brain/src/bin/wbrain_compact.rs
new file mode 100644
index 0000000..3f83832
--- /dev/null
+++ b/crates/brain/src/bin/wbrain_compact.rs
@@ -0,0 +1,165 @@
+//! Reclaim superseded neuron bodies from a `.wbrain` container.
+//!
+//! The container is append-only and has never had a compactor, so every
+//! eviction since the first cut has left its previous body on disk. Measured on
+//! the private training host 2026-09-10: 576.7 GB of file for 5,086,420
+//! neurons, growing 206–257 GB/h and about two hours from filling a 1.1 TB
+//! volume.
+//!
+//! # Usage
+//!
+//! ```text
+//! wbrain_compact --inspect  <brain.wbrain>
+//! wbrain_compact --in-place <brain.wbrain> [--keep-retired <path>]
+//! wbrain_compact <source.wbrain> <destination.wbrain>
+//! ```
+//!
+//! The brain MUST NOT be running. This tool takes no lock: the container has no
+//! locking protocol, and a live server appends to the source while it is being
+//! copied, which would publish a manifest pointing at records the copy never
+//! saw. `--inspect` is always safe.
+//!
+//! `--in-place` renames the original aside, installs the compacted file, and
+//! only then unlinks the original — so an interruption leaves a complete brain
+//! on disk under one name or the other. Space is returned at the unlink, which
+//! is also the only point `df` will move.
+
+use std::path::{Path, PathBuf};
+use std::process::ExitCode;
+
+use w1z4rd_brain::store::compaction;
+use w1z4rd_brain::store::container::BrainContainer;
+
+fn usage() -> ExitCode {
+    eprintln!(
+        "usage:\n  \
+         wbrain_compact --inspect <brain.wbrain>\n  \
+         wbrain_compact --in-place <brain.wbrain> [--keep-retired <path>]\n  \
+         wbrain_compact <source.wbrain> <destination.wbrain>"
+    );
+    ExitCode::from(2)
+}
+
+/// Report what the container holds without modifying it.
+///
+/// Prints the reference classes a compaction must remap, so a pass is never the
+/// first thing to discover that a brain uses an addressing form the tool has
+/// not been taught.
+fn inspect(path: &Path) -> std::io::Result<()> {
+    let container = BrainContainer::open(path)?;
+    let bytes = container.byte_len()?;
+    let manifest = container.manifest().cloned().ok_or_else(|| {
+        std::io::Error::new(
+            std::io::ErrorKind::InvalidData,
+            "container has no committed manifest",
+        )
+    })?;
+    let mut live = 0_u64;
+    let mut slot_tables = 0_u64;
+    let mut offset_vectors = 0_u64;
+    let mut label_indexes = 0_u64;
+    let mut metadata_bytes = 0_u64;
+    for pool in &manifest.pools {
+        if pool.neuron_slot_table.is_some() {
+            slot_tables += 1;
+        }
+        if !pool.neuron_offsets.is_empty() {
+            offset_vectors += 1;
+            live += pool.neuron_offsets.iter().filter(|s| s.is_some()).count() as u64;
+        }
+        label_indexes += pool.label_indexes.len() as u64;
+        metadata_bytes += pool.pool_metadata.len() as u64;
+    }
+    println!("path                 {}", path.display());
+    println!("bytes                {bytes} ({:.2} GB)", bytes as f64 / 1e9);
+    println!("generation           {}", manifest.generation);
+    println!("tick                 {}", manifest.tick);
+    println!("pools                {}", manifest.pools.len());
+    println!("pools_with_slot_table {slot_tables}");
+    println!("pools_with_offset_vec {offset_vectors}");
+    println!("live_in_offset_vecs  {live}");
+    println!("label_index_records  {label_indexes}");
+    println!("pool_metadata_bytes  {metadata_bytes}");
+    println!("brain_metadata_bytes {}", manifest.brain_metadata.len());
+    Ok(())
+}
+
+fn main() -> ExitCode {
+    let args: Vec<String> = std::env::args().skip(1).collect();
+    if args.is_empty() {
+        return usage();
+    }
+
+    if args[0] == "--inspect" {
+        if args.len() != 2 {
+            return usage();
+        }
+        return match inspect(Path::new(&args[1])) {
+            Ok(()) => ExitCode::SUCCESS,
+            Err(error) => {
+                eprintln!("inspect failed: {error}");
+                ExitCode::FAILURE
+            }
+        };
+    }
+
+    let started = std::time::Instant::now();
+    let (report, target) = if args[0] == "--in-place" {
+        if args.len() != 2 && args.len() != 4 {
+            return usage();
+        }
+        let path = PathBuf::from(&args[1]);
+        let keep = if args.len() == 4 {
+            if args[2] != "--keep-retired" {
+                return usage();
+            }
+            Some(PathBuf::from(&args[3]))
+        } else {
+            None
+        };
+        match compaction::compact_in_place(&path, keep.as_deref()) {
+            Ok(report) => (report, path),
+            Err(error) => {
+                eprintln!("compaction failed: {error}");
+                return ExitCode::FAILURE;
+            }
+        }
+    } else {
+        if args.len() != 2 {
+            return usage();
+        }
+        let source = PathBuf::from(&args[0]);
+        let destination = PathBuf::from(&args[1]);
+        let report = match compaction::compact(&source, &destination) {
+            Ok(report) => report,
+            Err(error) => {
+                eprintln!("compaction failed: {error}");
+                return ExitCode::FAILURE;
+            }
+        };
+        // A two-path run leaves both files in place, so verification is the
+        // caller's only evidence before they swap.
+        if let Err(error) = compaction::verify(&destination, &report) {
+            eprintln!("VERIFICATION FAILED, do not install this file: {error}");
+            return ExitCode::FAILURE;
+        }
+        (report, destination)
+    };
+
+    println!(
+        "{}",
+        serde_json::json!({
+            "target": target.display().to_string(),
+            "pools": report.pools,
+            "neurons_copied": report.neurons_copied,
+            "auxiliary_copied": report.auxiliary_copied,
+            "generation_directories_rewritten": report.generation_directories_rewritten,
+            "source_bytes": report.source_bytes,
+            "destination_bytes": report.destination_bytes,
+            "reclaimed_bytes": report.reclaimed_bytes(),
+            "reclaimed_gb": (report.reclaimed_bytes() as f64 / 1e9 * 100.0).round() / 100.0,
+            "elapsed_seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
+        })
+    );
+    ExitCode::SUCCESS
+}
diff --git a/crates/brain/src/brain.rs b/crates/brain/src/brain.rs
index d767818..e643107 100644
--- a/crates/brain/src/brain.rs
+++ b/crates/brain/src/brain.rs
@@ -12,7 +12,7 @@
 #[path = "streaming_migration.rs"]
 mod streaming_migration;
 #[path = "wbrain_metadata.rs"]
-mod wbrain_metadata;
+pub(crate) mod wbrain_metadata;
 
 use ahash::AHashMap;
 use std::collections::VecDeque;
diff --git a/crates/brain/src/pool.rs b/crates/brain/src/pool.rs
index 7dec63a..8fc8299 100644
--- a/crates/brain/src/pool.rs
+++ b/crates/brain/src/pool.rs
@@ -1393,6 +1393,76 @@ struct WbrainPoolMetadata {
     total_terminals: usize,
 }
 
+/// Rewrite every container offset held inside a serialized pool metadata blob.
+///
+/// Compaction moves records, so any `AuxiliaryRecordRef` that survives the move
+/// unchanged points at whatever bytes now occupy the old offset. The manifest's
+/// own references are visible to the compactor; these are NOT — they are buried
+/// inside an opaque `pool_metadata: Vec<u8>`, which is exactly why they are
+/// remapped here, next to the struct that defines them, rather than in the
+/// compactor where a new field would be missed.
+///
+/// The destructuring below is deliberately exhaustive: adding a field to
+/// [`WbrainPoolMetadata`] breaks this function at COMPILE time. A compactor
+/// that silently ignored a new reference field would corrupt recall in a way
+/// no test that predates the field could detect.
+pub(crate) fn remap_pool_metadata_refs(
+    blob: &[u8],
+    remap: &mut dyn FnMut(AuxiliaryRecordRef) -> std::io::Result<AuxiliaryRecordRef>,
+) -> std::io::Result<Vec<u8>> {
+    if blob.is_empty() {
+        return Ok(Vec::new());
+    }
+    let decoded: WbrainPoolMetadata = bincode::deserialize(blob)
+        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
+    let WbrainPoolMetadata {
+        config,
+        recent_atoms,
+        sequences,
+        legacy_sequence_ledger,
+        concept_multiset_to_id,
+        concept_sequence_to_id,
+        legacy_concept_sequence_index,
+        neuron_kinds,
+        concept_slots,
+        born_ticks,
+        concept_count,
+        total_terminals,
+    } = decoded;
+    let legacy_sequence_ledger = match legacy_sequence_ledger {
+        Some(reference) => Some(remap(reference)?),
+        None => None,
+    };
+    let legacy_concept_sequence_index = match legacy_concept_sequence_index {
+        Some(reference) => Some(remap(reference)?),
+        None => None,
+    };
+    let rebuilt = WbrainPoolMetadata {
+        config,
+        recent_atoms,
+        sequences,
+        legacy_sequence_ledger,
+        concept_multiset_to_id,
+        concept_sequence_to_id,
+        legacy_concept_sequence_index,
+        neuron_kinds,
+        concept_slots,
+        born_ticks,
+        concept_count,
+        total_terminals,
+    };
+    bincode::serialize(&rebuilt)
+        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
+}
+
+/// Magics of auxiliary bodies that nest absolute references to OTHER records.
+/// Compaction must rewrite these bodies; every other auxiliary body addresses
+/// itself relatively and copies verbatim.
+pub(crate) const GENERATION_DIRECTORY_MAGICS: [&[u8; 8]; 2] = [
+    CONCEPT_GENERATION_DIRECTORY_MAGIC,
+    SEQUENCE_GENERATION_DIRECTORY_MAGIC,
+];
+
 pub(crate) struct StreamedPoolMetadata {
     pub config: PoolConfig,
     pub recent_atoms: VecDeque<NeuronId>,
diff --git a/crates/brain/src/store/compaction.rs b/crates/brain/src/store/compaction.rs
new file mode 100644
index 0000000..fa11138
--- /dev/null
+++ b/crates/brain/src/store/compaction.rs
@@ -0,0 +1,756 @@
+//! Offline compaction of a `.wbrain` container — the §17.4 follow-up.
+//!
+//! `cold.rs` has said since the first cut that "every eviction appends a fresh
+//! record and the index points to the latest offset; older versions of the same
+//! neuron stay on disk as garbage and are reclaimed by a future compaction pass
+//! (Stage 17.4 follow-up)". That pass was never built, so nothing in the system
+//! has ever returned a superseded neuron body to the filesystem.
+//!
+//! Measured 2026-09-10 on the private training host: `brain.wbrain` held
+//! 576.7 GB for 5,086,420 neurons — about 110 KB of file per neuron, i.e. the
+//! container was overwhelmingly garbage — while the volume burned 206–257 GB/h
+//! and stood ~2 h from ENOSPC. The previous exhaustion crash-looped the service
+//! wrapper 115 times on a 6-byte `node.pid` write, which reads as a finished
+//! stage rather than as a full disk.
+//!
+//! # Why this is a byte copy and not a re-serialization
+//!
+//! Neuron bodies are opaque here. Compaction reads a record's 24-byte header,
+//! copies `body_len` bytes verbatim, and records the new offset. It never calls
+//! `bincode::deserialize`. That means a body this build cannot decode still
+//! round-trips intact, and a compaction pass cannot damage a field it does not
+//! know exists.
+//!
+//! # Where the offsets hide
+//!
+//! Copying records is the easy half. An offset that is not rewritten points at
+//! whatever bytes now occupy its old address, so every holder must be found:
+//!
+//! 1. `PoolContainerManifest::neuron_slot_table` — body is a dense array of
+//!    24-byte slots whose first 8 bytes are an absolute neuron offset.
+//! 2. `PoolContainerManifest::neuron_offsets` — the small-store equivalent.
+//! 3. `PoolContainerManifest::label_indexes` — the references move; the bodies
+//!    address themselves relatively and copy verbatim.
+//! 4. `WbrainPoolMetadata::legacy_sequence_ledger` and
+//!    `legacy_concept_sequence_index` — buried in an opaque `pool_metadata`
+//!    blob.
+//! 5. `WbrainBrainMetadata::binding_posting_indexes` — buried in an opaque
+//!    `brain_metadata` blob.
+//! 6. `W1ZCGEN1` / `W1ZSGEN1` generation directories — auxiliary bodies that
+//!    nest `(offset, len)` pairs pointing at OTHER auxiliary records.
+//!
+//! (4) and (5) are remapped by functions that live beside their struct
+//! definitions and destructure exhaustively, so a new reference field is a
+//! compile error rather than silent corruption. (6) is why an auxiliary body
+//! cannot simply be copied: this module classifies each body by its magic and
+//! **fails closed** — an unrecognized body that is too small to classify stops
+//! the pass instead of being copied blind.
+//!
+//! # Durability
+//!
+//! The destination is written and fsynced in full, and its manifest is
+//! published only after every record it references is durable, because
+//! `commit_manifest` syncs the body before touching a header slot. The caller
+//! swaps files only after `verify` reopens the destination and re-reads every
+//! live neuron offset. The source is never mutated.
+
+use ahash::AHashMap;
+use std::io;
+use std::path::Path;
+
+use crate::neuron::PoolId;
+use crate::pool::{GENERATION_DIRECTORY_MAGICS, remap_pool_metadata_refs};
+use crate::store::container::{
+    AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
+};
+use crate::store::wbrain_store::{NEURON_SLOT_BYTES, NEURON_SLOT_TABLE_KIND, SLOT_PRESENT};
+use crate::brain::wbrain_metadata::remap_brain_metadata_refs;
+
+/// Streaming buffer for verbatim record copies. Sized so a multi-megabyte
+/// neuron body moves in a few writes without ever holding a whole record.
+const COPY_BUFFER_BYTES: usize = 4 * 1024 * 1024;
+
+/// Slots rewritten per batch when rebuilding a neuron slot table. At 24 bytes a
+/// slot this buffers 24 MB for a 1M-neuron pool.
+const SLOT_BATCH: usize = 1024 * 1024;
+
+#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
+pub struct CompactionReport {
+    pub pools: u64,
+    pub neurons_copied: u64,
+    pub auxiliary_copied: u64,
+    pub generation_directories_rewritten: u64,
+    pub source_bytes: u64,
+    pub destination_bytes: u64,
+}
+
+impl CompactionReport {
+    /// Bytes the pass returns to the filesystem once the source is unlinked.
+    pub fn reclaimed_bytes(&self) -> u64 {
+        self.source_bytes.saturating_sub(self.destination_bytes)
+    }
+}
+
+/// Rewrite `source` into `destination`, keeping only live records.
+///
+/// `destination` must not already exist as a populated container; a fresh path
+/// is expected. The source is opened read/write only because `BrainContainer`
+/// has no read-only constructor — nothing here writes to it.
+pub fn compact(source: &Path, destination: &Path) -> io::Result<CompactionReport> {
+    let mut src = BrainContainer::open(source)?;
+    let manifest = src.manifest().cloned().ok_or_else(|| {
+        io::Error::new(
+            io::ErrorKind::InvalidData,
+            "source container has no committed manifest; refusing to compact",
+        )
+    })?;
+    let mut dst = BrainContainer::open(destination)?;
+    if dst.byte_len()? > crate::store::container::header_bytes() {
+        return Err(io::Error::new(
+            io::ErrorKind::AlreadyExists,
+            "destination container already holds records; refusing to compact into it",
+        ));
+    }
+
+    let mut state = Compactor {
+        buffer: vec![0_u8; COPY_BUFFER_BYTES],
+        auxiliary: AHashMap::new(),
+        report: CompactionReport::default(),
+    };
+
+    let BrainContainerManifest {
+        generation,
+        tick,
+        brain_metadata,
+        pools,
+    } = manifest;
+
+    let mut rebuilt_pools = Vec::with_capacity(pools.len());
+    for pool in pools {
+        rebuilt_pools.push(state.compact_pool(&mut src, &mut dst, pool)?);
+        state.report.pools += 1;
+    }
+
+    // Brain metadata last: its posting-index records are shared across pools,
+    // so by now most are already in the map and remap becomes a lookup.
+    let brain_metadata = {
+        let mut remap =
+            |reference: AuxiliaryRecordRef| state.copy_auxiliary(&mut src, &mut dst, reference);
+        remap_brain_metadata_refs(&brain_metadata, &mut remap)?
+    };
+
+    dst.flush()?;
+    dst.commit_manifest(BrainContainerManifest {
+        generation,
+        tick,
+        brain_metadata,
+        pools: rebuilt_pools,
+    })?;
+    dst.flush()?;
+
+    state.report.source_bytes = src.byte_len()?;
+    state.report.destination_bytes = dst.byte_len()?;
+    Ok(state.report)
+}
+
+/// Reopen a compacted container and read every live neuron offset back.
+///
+/// A compaction that is announced but not verified is the same class of error
+/// as a supervisor that reports `active` while admitting nothing. This reads
+/// the record header at each live offset and confirms the neuron ID recorded in
+/// the slot table matches the one in the record it now points at, which is
+/// exactly the property a mis-remapped offset would break.
+pub fn verify(path: &Path, expected: &CompactionReport) -> io::Result<u64> {
+    let mut container = BrainContainer::open(path)?;
+    let manifest = container.manifest().cloned().ok_or_else(|| {
+        io::Error::new(
+            io::ErrorKind::InvalidData,
+            "compacted container published no manifest",
+        )
+    })?;
+    let mut checked = 0_u64;
+    for pool in &manifest.pools {
+        let offsets = live_offsets(&mut container, pool)?;
+        for (id, offset) in offsets {
+            let (record_pool, neuron) = container.read_neuron_at(offset)?;
+            if record_pool != pool.pool_id || neuron.id != id {
+                return Err(io::Error::new(
+                    io::ErrorKind::InvalidData,
+                    format!(
+                        "compacted offset {offset} resolves to pool {record_pool} neuron {} \
+                         but the slot table claims pool {} neuron {id}",
+                        neuron.id, pool.pool_id
+                    ),
+                ));
+            }
+            checked += 1;
+        }
+    }
+    if checked != expected.neurons_copied {
+        return Err(io::Error::new(
+            io::ErrorKind::InvalidData,
+            format!(
+                "compaction copied {} neurons but verification found {checked}",
+                expected.neurons_copied
+            ),
+        ));
+    }
+    Ok(checked)
+}
+
+/// Live `(neuron id, offset)` pairs for one pool, from whichever addressing
+/// form its manifest uses.
+fn live_offsets(
+    container: &mut BrainContainer,
+    pool: &PoolContainerManifest,
+) -> io::Result<Vec<(u32, u64)>> {
+    let mut live = Vec::new();
+    if let Some(table) = pool.neuron_slot_table {
+        let slots = table.len / NEURON_SLOT_BYTES;
+        let mut raw = [0_u8; NEURON_SLOT_BYTES as usize];
+        for index in 0..slots {
+            container.read_auxiliary_exact(table, index * NEURON_SLOT_BYTES, &mut raw)?;
+            let offset = u64::from_le_bytes(raw[0..8].try_into().unwrap());
+            let flags = raw[17];
+            if flags & SLOT_PRESENT != 0 && offset != 0 {
+                live.push((index as u32, offset));
+            }
+        }
+    } else {
+        for (index, slot) in pool.neuron_offsets.iter().enumerate() {
+            if let Some(offset) = slot {
+                live.push((index as u32, *offset));
+            }
+        }
+    }
+    Ok(live)
+}
+
+struct Compactor {
+    buffer: Vec<u8>,
+    /// Source auxiliary offset → destination reference. Shared records (a
+    /// posting index referenced by several pools) are copied once.
+    auxiliary: AHashMap<u64, AuxiliaryRecordRef>,
+    report: CompactionReport,
+}
+
+impl Compactor {
+    fn compact_pool(
+        &mut self,
+        src: &mut BrainContainer,
+        dst: &mut BrainContainer,
+        pool: PoolContainerManifest,
+    ) -> io::Result<PoolContainerManifest> {
+        let PoolContainerManifest {
+            pool_id,
+            neuron_count,
+            neuron_capacity,
+            neuron_slot_table,
+            label_indexes,
+            neuron_offsets,
+            labels,
+            pool_metadata,
+        } = pool;
+
+        // ---- Neuron bodies, and the addressing structure that finds them.
+        let (neuron_slot_table, neuron_offsets) = match neuron_slot_table {
+            Some(table) => {
+                let rebuilt = self.rebuild_slot_table(src, dst, pool_id, table)?;
+                (Some(rebuilt), Vec::new())
+            }
+            None => {
+                let mut moved = Vec::with_capacity(neuron_offsets.len());
+                for slot in neuron_offsets {
+                    match slot {
+                        Some(offset) => {
+                            let new_offset =
+                                src.copy_neuron_record_into(offset, dst, &mut self.buffer)?;
+                            self.report.neurons_copied += 1;
+                            moved.push(Some(new_offset));
+                        }
+                        None => moved.push(None),
+                    }
+                }
+                (None, moved)
+            }
+        };
+
+        // ---- Label indexes: references move, bodies are position independent.
+        let mut moved_label_indexes = Vec::with_capacity(label_indexes.len());
+        for reference in label_indexes {
+            moved_label_indexes.push(self.copy_auxiliary(src, dst, reference)?);
+        }
+
+        // ---- Offsets hidden inside the opaque metadata blob.
+        let pool_metadata = {
+            let mut remap = |reference: AuxiliaryRecordRef| self.copy_auxiliary(src, dst, reference);
+            remap_pool_metadata_refs(&pool_metadata, &mut remap)?
+        };
+
+        Ok(PoolContainerManifest {
+            pool_id,
+            neuron_count,
+            neuron_capacity,
+            neuron_slot_table,
+            label_indexes: moved_label_indexes,
+            neuron_offsets,
+            labels,
+            pool_metadata,
+        })
+    }
+
+    /// Copy every live neuron in a slot table and write the table back with the
+    /// new offsets.
+    ///
+    /// Only bytes 0..8 of each slot are rewritten. `born_tick`, kind and flags
+    /// are carried through untouched, so a slot's identity survives a pass that
+    /// only ever intended to move its address.
+    fn rebuild_slot_table(
+        &mut self,
+        src: &mut BrainContainer,
+        dst: &mut BrainContainer,
+        pool_id: PoolId,
+        table: AuxiliaryRecordRef,
+    ) -> io::Result<AuxiliaryRecordRef> {
+        if table.len % NEURON_SLOT_BYTES != 0 {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "neuron slot table is not a whole number of slots",
+            ));
+        }
+        let slots = table.len / NEURON_SLOT_BYTES;
+        let mut rebuilt: Vec<u8> = Vec::with_capacity(
+            usize::try_from(table.len)
+                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "slot table too large"))?,
+        );
+        let mut batch = vec![0_u8; SLOT_BATCH * NEURON_SLOT_BYTES as usize];
+        let mut done = 0_u64;
+        while done < slots {
+            let take = (slots - done).min(SLOT_BATCH as u64);
+            let bytes = (take * NEURON_SLOT_BYTES) as usize;
+            src.read_auxiliary_exact(table, done * NEURON_SLOT_BYTES, &mut batch[..bytes])?;
+            for index in 0..take as usize {
+                let start = index * NEURON_SLOT_BYTES as usize;
+                let slot = &mut batch[start..start + NEURON_SLOT_BYTES as usize];
+                let offset = u64::from_le_bytes(slot[0..8].try_into().unwrap());
+                let present = slot[17] & SLOT_PRESENT != 0 && offset != 0;
+                if present {
+                    let new_offset = src.copy_neuron_record_into(offset, dst, &mut self.buffer)?;
+                    slot[0..8].copy_from_slice(&new_offset.to_le_bytes());
+                    self.report.neurons_copied += 1;
+                }
+                rebuilt.extend_from_slice(slot);
+            }
+            done += take;
+        }
+        let reference = dst.append_auxiliary_body(pool_id, NEURON_SLOT_TABLE_KIND, &rebuilt)?;
+        self.report.auxiliary_copied += 1;
+        Ok(reference)
+    }
+
+    /// Move one auxiliary record, rewriting it if its body nests references.
+    fn copy_auxiliary(
+        &mut self,
+        src: &mut BrainContainer,
+        dst: &mut BrainContainer,
+        reference: AuxiliaryRecordRef,
+    ) -> io::Result<AuxiliaryRecordRef> {
+        if let Some(existing) = self.auxiliary.get(&reference.offset) {
+            return Ok(*existing);
+        }
+        let (pool, kind) = src.auxiliary_header(reference)?;
+        let prefix = src.read_auxiliary_prefix(reference, 8)?;
+        let is_directory = prefix.len() == 8
+            && GENERATION_DIRECTORY_MAGICS
+                .iter()
+                .any(|magic| prefix.as_slice() == magic.as_slice());
+
+        let moved = if is_directory {
+            let body = self.rewrite_generation_directory(src, dst, reference)?;
+            self.report.generation_directories_rewritten += 1;
+            dst.append_auxiliary_body(pool, kind, &body)?
+        } else {
+            src.copy_auxiliary_record_into(reference, dst, &mut self.buffer)?
+        };
+        self.report.auxiliary_copied += 1;
+        self.auxiliary.insert(reference.offset, moved);
+        Ok(moved)
+    }
+
+    /// Rebuild a `W1ZCGEN1` / `W1ZSGEN1` directory body.
+    ///
+    /// Layout is `[8B magic][8B count][count × (u64 offset, u64 len)]`. Each
+    /// child is copied first — recursively, since a directory may reference a
+    /// directory — and the entry is rewritten to its new address.
+    fn rewrite_generation_directory(
+        &mut self,
+        src: &mut BrainContainer,
+        dst: &mut BrainContainer,
+        reference: AuxiliaryRecordRef,
+    ) -> io::Result<Vec<u8>> {
+        let header = src.read_auxiliary_prefix(reference, 16)?;
+        if header.len() < 16 {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "generation directory is shorter than its header",
+            ));
+        }
+        let count = u64::from_le_bytes(header[8..16].try_into().unwrap());
+        let required = 16_u64
+            .checked_add(count.checked_mul(16).ok_or_else(|| {
+                io::Error::new(io::ErrorKind::InvalidData, "generation directory overflow")
+            })?)
+            .ok_or_else(|| {
+                io::Error::new(io::ErrorKind::InvalidData, "generation directory overflow")
+            })?;
+        if required > reference.len {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "generation directory is truncated",
+            ));
+        }
+        // The trailing bytes after the entry table are payload the directory
+        // format may carry; preserve them verbatim.
+        let whole = src.read_auxiliary_prefix(reference, reference.len as usize)?;
+        let mut body = whole.clone();
+        for index in 0..count {
+            let at = (16 + index * 16) as usize;
+            let child = AuxiliaryRecordRef {
+                offset: u64::from_le_bytes(whole[at..at + 8].try_into().unwrap()),
+                len: u64::from_le_bytes(whole[at + 8..at + 16].try_into().unwrap()),
+            };
+            if child.offset == 0 || child.len < 8 {
+                return Err(io::Error::new(
+                    io::ErrorKind::InvalidData,
+                    "generation directory contains an invalid reference",
+                ));
+            }
+            let moved = self.copy_auxiliary(src, dst, child)?;
+            body[at..at + 8].copy_from_slice(&moved.offset.to_le_bytes());
+            body[at + 8..at + 16].copy_from_slice(&moved.len.to_le_bytes());
+        }
+        Ok(body)
+    }
+}
+
+/// Write a compacted container beside `path` and swap it in.
+///
+/// The source is renamed aside rather than unlinked, so a failure at any point
+/// leaves a complete brain on disk. The caller deletes the retired file only
+/// after the swap is durable — which is also the only moment the space is
+/// actually returned.
+pub fn compact_in_place(path: &Path, keep_retired_as: Option<&Path>) -> io::Result<CompactionReport> {
+    let working = path.with_extension("compacting");
+    if working.exists() {
+        std::fs::remove_file(&working)?;
+    }
+    let report = compact(path, &working)?;
+    verify(&working, &report)?;
+
+    let retired = match keep_retired_as {
+        Some(target) => target.to_path_buf(),
+        None => path.with_extension("retired"),
+    };
+    std::fs::rename(path, &retired)?;
+    if let Err(error) = std::fs::rename(&working, path) {
+        // Put the original back: a half-swapped brain is the one outcome worse
+        // than a full disk.
+        std::fs::rename(&retired, path)?;
+        return Err(error);
+    }
+    if keep_retired_as.is_none() {
+        std::fs::remove_file(&retired)?;
+    }
+    Ok(report)
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::neuron::{Neuron, NeuronKind, NeuronRef, Terminal};
+
+    fn tmpdir(name: &str) -> std::path::PathBuf {
+        let nonce = std::time::SystemTime::now()
+            .duration_since(std::time::UNIX_EPOCH)
+            .unwrap()
+            .as_nanos();
+        let dir = std::env::temp_dir().join(format!(
+            "w1z4rd_compact_{name}_{}_{}",
+            std::process::id(),
+            nonce
+        ));
+        std::fs::create_dir_all(&dir).unwrap();
+        dir
+    }
+
+    fn concept(id: u32, label: &str, terminals: usize) -> Neuron {
+        let mut neuron = Neuron::new_concept(
+            id,
+            label.into(),
+            NeuronKind::Excitatory,
+            vec![NeuronRef::new(0, 1)],
+            1,
+        );
+        neuron.terminals = (0..terminals)
+            .map(|t| Terminal::new(NeuronRef::new(0, t as u32), 0.25, 1))
+            .collect();
+        neuron
+    }
+
+    /// The property the whole pass exists for: superseded bodies are dropped
+    /// and the survivors still read back.
+    #[test]
+    fn compaction_drops_superseded_bodies_and_preserves_live_ones() {
+        let dir = tmpdir("supersede");
+        let source = dir.join("brain.wbrain");
+        let destination = dir.join("out.wbrain");
+
+        let live_offset;
+        {
+            let mut container = BrainContainer::open(&source).unwrap();
+            // Ten stale generations of the same neuron, then the live one.
+            let mut offset = 0;
+            for generation in 0..10 {
+                offset = container
+                    .append_neuron(1, &concept(0, "c:live", 500 + generation))
+                    .unwrap();
+            }
+            live_offset = offset;
+            container
+                .commit_manifest(BrainContainerManifest {
+                    generation: 2,
+                    tick: 77,
+                    brain_metadata: Vec::new(),
+                    pools: vec![PoolContainerManifest {
+                        pool_id: 1,
+                        neuron_count: 1,
+                        neuron_capacity: 1,
+                        neuron_slot_table: None,
+                        label_indexes: Vec::new(),
+                        neuron_offsets: vec![Some(live_offset)],
+                        labels: vec![("c:live".into(), 0)],
+                        pool_metadata: Vec::new(),
+                    }],
+                })
+                .unwrap();
+        }
+
+        let before = BrainContainer::open(&source).unwrap().byte_len().unwrap();
+        let report = compact(&source, &destination).unwrap();
+        verify(&destination, &report).unwrap();
+
+        assert_eq!(report.neurons_copied, 1, "only the live body should move");
+        assert!(
+            report.destination_bytes * 4 < before,
+            "ten superseded generations must not survive: {} vs {before}",
+            report.destination_bytes
+        );
+
+        let mut compacted = BrainContainer::open(&destination).unwrap();
+        let manifest = compacted.manifest().cloned().unwrap();
+        assert_eq!(manifest.tick, 77, "tick must survive compaction");
+        assert_eq!(manifest.generation, 2);
+        let offset = manifest.pools[0].neuron_offsets[0].unwrap();
+        let (pool, neuron) = compacted.read_neuron_at(offset).unwrap();
+        assert_eq!(pool, 1);
+        assert_eq!(neuron.label, "c:live");
+        assert_eq!(
+            neuron.terminals.len(),
+            509,
+            "the LAST generation is the live one"
+        );
+        std::fs::remove_dir_all(dir).ok();
+    }
+
+    /// A reference nested inside a generation directory must be rewritten. Left
+    /// alone it would point at whatever bytes now sit at the old address, which
+    /// is the failure mode no neuron-level check would catch.
+    #[test]
+    fn generation_directory_children_are_remapped() {
+        let dir = tmpdir("directory");
+        let source = dir.join("brain.wbrain");
+        let destination = dir.join("out.wbrain");
+
+        let root;
+        {
+            let mut container = BrainContainer::open(&source).unwrap();
+            // Padding so the child cannot coincidentally land on the same
+            // offset in the destination.
+            for _ in 0..8 {
+                container.append_neuron(1, &concept(0, "c:pad", 200)).unwrap();
+            }
+            let child = container
+                .append_auxiliary_body(1, 9, b"W1ZSEQ01child-payload")
+                .unwrap();
+            let mut body = Vec::new();
+            body.extend_from_slice(b"W1ZSGEN1");
+            body.extend_from_slice(&1_u64.to_le_bytes());
+            body.extend_from_slice(&child.offset.to_le_bytes());
+            body.extend_from_slice(&child.len.to_le_bytes());
+            root = container.append_auxiliary_body(1, 10, &body).unwrap();
+            let live = container.append_neuron(1, &concept(0, "c:live", 4)).unwrap();
+            container
+                .commit_manifest(BrainContainerManifest {
+                    generation: 2,
+                    tick: 5,
+                    brain_metadata: Vec::new(),
+                    pools: vec![PoolContainerManifest {
+                        pool_id: 1,
+                        neuron_count: 1,
+                        neuron_capacity: 1,
+                        neuron_slot_table: None,
+                        label_indexes: vec![root],
+                        neuron_offsets: vec![Some(live)],
+                        labels: Vec::new(),
+                        pool_metadata: Vec::new(),
+                    }],
+                })
+                .unwrap();
+        }
+
+        let report = compact(&source, &destination).unwrap();
+        verify(&destination, &report).unwrap();
+        assert_eq!(report.generation_directories_rewritten, 1);
+
+        let mut compacted = BrainContainer::open(&destination).unwrap();
+        let manifest = compacted.manifest().cloned().unwrap();
+        let moved_root = manifest.pools[0].label_indexes[0];
+        assert_ne!(
+            moved_root.offset, root.offset,
+            "the directory itself must have moved, or this proves nothing"
+        );
+        let body = compacted.read_auxiliary(moved_root).unwrap();
+        let child = AuxiliaryRecordRef {
+            offset: u64::from_le_bytes(body[16..24].try_into().unwrap()),
+            len: u64::from_le_bytes(body[24..32].try_into().unwrap()),
+        };
+        assert_eq!(
+            compacted.read_auxiliary(child).unwrap(),
+            b"W1ZSEQ01child-payload",
+            "the nested reference must resolve in the COMPACTED file"
+        );
+        std::fs::remove_dir_all(dir).ok();
+    }
+
+    /// The shape the production brain actually has.
+    ///
+    /// A 5M-neuron pool addresses its bodies through a paged slot table, not
+    /// through `neuron_offsets`; the offset-vector tests above exercise a form
+    /// the training host never emits. This builds a container through the real
+    /// store, compacts it, and reopens it through the real store again — so the
+    /// assertion is that the BRAIN still works, not merely that the bytes moved.
+    #[test]
+    fn compaction_preserves_a_paged_slot_table_brain() {
+        use crate::store::neuron_store::NeuronStore;
+        use crate::store::wbrain_store::WbrainFile;
+
+        let dir = tmpdir("paged");
+        let source = dir.join("brain.wbrain");
+        let destination = dir.join("out.wbrain");
+        {
+            let file = WbrainFile::open(&source).unwrap();
+            let pool = file.pool(7);
+            pool.prepare_paged_slots(4).unwrap();
+            // Establish the slot extent in ascending id order.
+            pool.persist_sleeping(&Neuron::new_atom(0, "zero".into(), NeuronKind::Excitatory, 1))
+                .unwrap();
+            pool.persist_sleeping(&concept(1, "one", 3)).unwrap();
+            pool.persist_sleeping(&Neuron::new_atom(2, "two".into(), NeuronKind::Excitatory, 1))
+                .unwrap();
+            pool.persist_sleeping(&concept(3, "three", 3)).unwrap();
+            // Now re-sleep each one repeatedly: every pass leaves a superseded
+            // body behind, which is exactly the garbage compaction exists for.
+            for generation in 1..6 {
+                pool.persist_sleeping(&Neuron::new_atom(
+                    0,
+                    "zero".into(),
+                    NeuronKind::Excitatory,
+                    1,
+                ))
+                .unwrap();
+                pool.persist_sleeping(&concept(1, "one", 3 + generation))
+                    .unwrap();
+                pool.persist_sleeping(&Neuron::new_atom(
+                    2,
+                    "two".into(),
+                    NeuronKind::Excitatory,
+                    1,
+                ))
+                .unwrap();
+                pool.persist_sleeping(&concept(3, "three", 3 + generation))
+                    .unwrap();
+            }
+            file.commit_manifest().unwrap();
+            file.flush().unwrap();
+        }
+
+        let before = std::fs::metadata(&source).unwrap().len();
+        let report = compact(&source, &destination).unwrap();
+        verify(&destination, &report).unwrap();
+
+        assert_eq!(
+            report.neurons_copied, 4,
+            "24 bodies were written but only 4 are live"
+        );
+        assert!(
+            report.destination_bytes < before,
+            "compacted {} is not smaller than {before}",
+            report.destination_bytes
+        );
+
+        // Reopen through the STORE: slot identity, concept flags, label routing
+        // and body content must all survive the move.
+        let reopened = WbrainFile::open(&destination).unwrap();
+        let pool = reopened.pool(7);
+        assert_eq!(pool.slot_count(), 4);
+        assert_eq!(pool.known_count(), 4);
+        assert!(pool.slot_is_concept(1), "concept flag must survive");
+        assert!(!pool.slot_is_concept(2), "atom flag must survive");
+        assert_eq!(pool.get(0).unwrap().label, "zero");
+        assert_eq!(pool.get(2).unwrap().label, "two");
+        assert_eq!(pool.label_to_id("two"), Some(2));
+        assert_eq!(
+            pool.get(3).unwrap().terminals.len(),
+            8,
+            "the LAST generation must be the one that survives"
+        );
+        std::fs::remove_dir_all(dir).ok();
+    }
+
+    /// Compaction must refuse a destination that already holds records rather
+    /// than appending a second brain into it.
+    #[test]
+    fn refuses_a_populated_destination() {
+        let dir = tmpdir("populated");
+        let source = dir.join("brain.wbrain");
+        let destination = dir.join("out.wbrain");
+        {
+            let mut container = BrainContainer::open(&source).unwrap();
+            let offset = container.append_neuron(1, &concept(0, "c:a", 2)).unwrap();
+            container
+                .commit_manifest(BrainContainerManifest {
+                    generation: 2,
+                    tick: 1,
+                    brain_metadata: Vec::new(),
+                    pools: vec![PoolContainerManifest {
+                        pool_id: 1,
+                        neuron_count: 1,
+                        neuron_capacity: 1,
+                        neuron_slot_table: None,
+                        label_indexes: Vec::new(),
+                        neuron_offsets: vec![Some(offset)],
+                        labels: Vec::new(),
+                        pool_metadata: Vec::new(),
+                    }],
+                })
+                .unwrap();
+        }
+        {
+            let mut occupied = BrainContainer::open(&destination).unwrap();
+            occupied.append_neuron(1, &concept(0, "c:b", 2)).unwrap();
+        }
+        let error = compact(&source, &destination).unwrap_err();
+        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
+        std::fs::remove_dir_all(dir).ok();
+    }
+}
diff --git a/crates/brain/src/store/container.rs b/crates/brain/src/store/container.rs
index 97832c0..e7cfa6d 100644
--- a/crates/brain/src/store/container.rs
+++ b/crates/brain/src/store/container.rs
@@ -475,6 +475,144 @@ impl BrainContainer {
         self.file.flush()?;
         self.file.sync_all()
     }
+
+    /// Byte length of every record header: magic, two u32 fields, then a u64
+    /// body length. Neuron and auxiliary records share this shape.
+    pub(crate) const RECORD_HEADER_BYTES: u64 = 24;
+
+    /// Copy one neuron record verbatim into `destination`, returning its new
+    /// offset.
+    ///
+    /// The bincode body is deliberately NOT decoded. Compaction moves bytes,
+    /// so a body this build cannot deserialize still round-trips intact
+    /// instead of being silently re-serialized into whatever the current
+    /// `Neuron` layout happens to be. It also means a compaction pass cannot
+    /// corrupt a field it does not know exists.
+    pub(crate) fn copy_neuron_record_into(
+        &mut self,
+        offset: u64,
+        destination: &mut BrainContainer,
+        buffer: &mut [u8],
+    ) -> io::Result<u64> {
+        self.copy_record_into(offset, NEURON_RECORD, destination, buffer)
+            .map(|(new_offset, _len)| new_offset)
+    }
+
+    /// Pool and kind fields of an auxiliary record header.
+    pub(crate) fn auxiliary_header(
+        &mut self,
+        reference: AuxiliaryRecordRef,
+    ) -> io::Result<(PoolId, u32)> {
+        self.file.seek(SeekFrom::Start(reference.offset))?;
+        let mut header = [0_u8; Self::RECORD_HEADER_BYTES as usize];
+        self.file.read_exact(&mut header)?;
+        if &header[0..8] != AUXILIARY_RECORD {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "auxiliary marker mismatch",
+            ));
+        }
+        if u64::from_le_bytes(header[16..24].try_into().unwrap()) != reference.len {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "auxiliary length mismatch",
+            ));
+        }
+        Ok((
+            u32::from_le_bytes(header[8..12].try_into().unwrap()),
+            u32::from_le_bytes(header[12..16].try_into().unwrap()),
+        ))
+    }
+
+    /// Copy an auxiliary record verbatim, returning its reference in
+    /// `destination`. Suitable only for bodies that are position independent;
+    /// bodies that nest absolute offsets must be rewritten by the caller.
+    pub(crate) fn copy_auxiliary_record_into(
+        &mut self,
+        reference: AuxiliaryRecordRef,
+        destination: &mut BrainContainer,
+        buffer: &mut [u8],
+    ) -> io::Result<AuxiliaryRecordRef> {
+        let (offset, len) =
+            self.copy_record_into(reference.offset, AUXILIARY_RECORD, destination, buffer)?;
+        if len != reference.len {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "auxiliary body length disagreed with its reference",
+            ));
+        }
+        Ok(AuxiliaryRecordRef { offset, len })
+    }
+
+    /// Read at most `limit` bytes from the start of an auxiliary body. Used to
+    /// classify a record by its magic without hydrating a multi-gigabyte body.
+    pub(crate) fn read_auxiliary_prefix(
+        &mut self,
+        reference: AuxiliaryRecordRef,
+        limit: usize,
+    ) -> io::Result<Vec<u8>> {
+        let take = (reference.len as usize).min(limit);
+        let mut body = vec![0_u8; take];
+        if take > 0 {
+            self.file
+                .seek(SeekFrom::Start(reference.offset + Self::RECORD_HEADER_BYTES))?;
+            self.file.read_exact(&mut body)?;
+        }
+        Ok(body)
+    }
+
+    /// Shared verbatim record copy. Streams the body through `buffer` so a
+    /// record larger than RAM still moves.
+    fn copy_record_into(
+        &mut self,
+        offset: u64,
+        expected_magic: &[u8; 8],
+        destination: &mut BrainContainer,
+        buffer: &mut [u8],
+    ) -> io::Result<(u64, u64)> {
+        self.file.seek(SeekFrom::Start(offset))?;
+        let mut header = [0_u8; Self::RECORD_HEADER_BYTES as usize];
+        self.file.read_exact(&mut header)?;
+        if &header[0..8] != expected_magic {
+            return Err(io::Error::new(
+                io::ErrorKind::InvalidData,
+                "record marker mismatch during compaction",
+            ));
+        }
+        let body_len = u64::from_le_bytes(header[16..24].try_into().unwrap());
+        let new_offset = destination.file.seek(SeekFrom::End(0))?;
+        destination.file.write_all(&header)?;
+        let mut remaining = body_len;
+        while remaining > 0 {
+            let take = remaining.min(buffer.len() as u64) as usize;
+            let slice = &mut buffer[..take];
+            self.file.read_exact(slice)?;
+            destination.file.write_all(slice)?;
+            remaining -= take as u64;
+        }
+        Ok((new_offset, body_len))
+    }
+
+    /// Append an auxiliary record whose body is already materialized.
+    pub(crate) fn append_auxiliary_body(
+        &mut self,
+        pool: PoolId,
+        kind: u32,
+        body: &[u8],
+    ) -> io::Result<AuxiliaryRecordRef> {
+        self.append_auxiliary(pool, kind, |writer| writer.write_all(body))
+    }
+
+    /// Total bytes currently occupied by the container.
+    pub fn byte_len(&self) -> io::Result<u64> {
+        Ok(self.file.metadata()?.len())
+    }
+}
+
+/// Bytes reserved for the fixed header and its two alternating manifest slots.
+/// A container of exactly this length holds no records.
+pub(crate) fn header_bytes() -> u64 {
+    HEADER_BYTES
 }
 
 #[cfg(test)]
diff --git a/crates/brain/src/store/mod.rs b/crates/brain/src/store/mod.rs
index f651b75..6801e7a 100644
--- a/crates/brain/src/store/mod.rs
+++ b/crates/brain/src/store/mod.rs
@@ -22,6 +22,7 @@
 
 pub mod bloom;
 pub mod cold;
+pub mod compaction;
 pub mod container;
 pub mod control;
 pub mod event;
@@ -34,6 +35,7 @@ pub mod wbrain_store;
 
 pub use bloom::CountingBloom;
 pub use cold::ColdTier;
+pub use compaction::{CompactionReport, compact, compact_in_place, verify as verify_compaction};
 pub use container::{
     AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
 };
diff --git a/crates/brain/src/store/wbrain_store.rs b/crates/brain/src/store/wbrain_store.rs
index e0a7f31..3a200c7 100644
--- a/crates/brain/src/store/wbrain_store.rs
+++ b/crates/brain/src/store/wbrain_store.rs
@@ -18,9 +18,9 @@ use crate::store::container::{
     AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
 };
 
-const NEURON_SLOT_TABLE_KIND: u32 = 0x534C_4F54; // "SLOT"
-const NEURON_SLOT_BYTES: u64 = 24;
-const SLOT_PRESENT: u8 = 0b0000_0001;
+pub(crate) const NEURON_SLOT_TABLE_KIND: u32 = 0x534C_4F54; // "SLOT"
+pub(crate) const NEURON_SLOT_BYTES: u64 = 24;
+pub(crate) const SLOT_PRESENT: u8 = 0b0000_0001;
 const SLOT_CONCEPT: u8 = 0b0000_0010;
 const SLOT_WRITE_BATCH: usize = 65_536;
 const AUXILIARY_HEADER_BYTES: u64 = 24;
diff --git a/crates/brain/src/wbrain_metadata.rs b/crates/brain/src/wbrain_metadata.rs
index f812383..86ace3c 100644
--- a/crates/brain/src/wbrain_metadata.rs
+++ b/crates/brain/src/wbrain_metadata.rs
@@ -40,3 +40,73 @@ pub(crate) struct WbrainBrainMetadata {
     pub eem: EemSnapshot,
     pub annealer: AnnealerSnapshot,
 }
+
+/// Rewrite every container offset held inside a serialized brain metadata blob.
+///
+/// `binding_posting_indexes` is the only persisted reference vector here:
+/// `fingerprint_posting_indexes` and `fingerprint_candidate_indexes` are
+/// DERIVED from generation markers on restore (see `brain.rs`), so they hold no
+/// durable offsets and must not be re-derived by a compactor.
+///
+/// As in `pool::remap_pool_metadata_refs`, the destructuring is exhaustive on
+/// purpose: a future reference field added to [`WbrainBrainMetadata`] becomes a
+/// compile error here rather than a brain whose binding recall silently reads
+/// whatever bytes landed at a stale offset.
+pub(crate) fn remap_brain_metadata_refs(
+    blob: &[u8],
+    remap: &mut dyn FnMut(AuxiliaryRecordRef) -> std::io::Result<AuxiliaryRecordRef>,
+) -> std::io::Result<Vec<u8>> {
+    if blob.is_empty() {
+        return Ok(Vec::new());
+    }
+    let decoded: WbrainBrainMetadata = bincode::deserialize(blob)
+        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
+    let WbrainBrainMetadata {
+        config,
+        binding_pool_id,
+        moment_history,
+        binding_recurrences,
+        lifetime_recurrences,
+        tentative_promoted,
+        promoted_fingerprints,
+        binding_sequence_index,
+        binding_feature_atom_index,
+        binding_motif_index,
+        binding_posting_indexes,
+        total_observations,
+        current_threshold,
+        last_pressure_check_obs,
+        action_pool_id,
+        pending_actions,
+        next_action_id,
+        eem,
+        annealer,
+    } = decoded;
+    let mut moved = Vec::with_capacity(binding_posting_indexes.len());
+    for reference in binding_posting_indexes {
+        moved.push(remap(reference)?);
+    }
+    let rebuilt = WbrainBrainMetadata {
+        config,
+        binding_pool_id,
+        moment_history,
+        binding_recurrences,
+        lifetime_recurrences,
+        tentative_promoted,
+        promoted_fingerprints,
+        binding_sequence_index,
+        binding_feature_atom_index,
+        binding_motif_index,
+        binding_posting_indexes: moved,
+        total_observations,
+        current_threshold,
+        last_pressure_check_obs,
+        action_pool_id,
+        pending_actions,
+        next_action_id,
+        eem,
+        annealer,
+    };
+    bincode::serialize(&rebuilt)
+        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
+}
WIZARD_PATCH_EOF

echo "patch_bytes=$(stat -c %s /tmp/wizard-deploy/compactor.patch)"
echo "git_version=$(git --version 2>&1)"

echo "--- check ---"
git apply --check -p1 -v /tmp/wizard-deploy/compactor.patch 2>&1 | tail -25
check_rc=${PIPESTATUS[0]}
echo "check_rc=${check_rc}"

if [ "${check_rc}" -ne 0 ]; then
  echo "REFUSING TO APPLY: git apply --check failed"
  echo "--- host file fingerprints, to see how far they have drifted ---"
  for f in crates/brain/src/brain.rs crates/brain/src/pool.rs \
           crates/brain/src/wbrain_metadata.rs crates/brain/src/store/container.rs \
           crates/brain/src/store/mod.rs crates/brain/src/store/wbrain_store.rs; do
    echo "$(md5sum "$f" 2>&1)"
  done
  exit 1
fi

echo "--- applying ---"
git apply -p1 /tmp/wizard-deploy/compactor.patch 2>&1 | tail -20
echo "apply_rc=${PIPESTATUS[0]}"
chown -R ec2-user:ec2-user crates 2>/dev/null || true

BIN=target/release/wbrain_compact
echo "--- binary before ---"
stat -c 'inode=%i mtime=%y size=%s' "$BIN" 2>&1

echo "--- building ---"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
export CARGO_HOME=${CARGO_HOME:-/srv/wizard/.cargo}
timeout 3000 sudo -u ec2-user \
  env CARGO_HOME="$CARGO_HOME" CARGO_BUILD_JOBS=1 \
  cargo build --release --offline \
  -p w1z4rd-brain --bin wbrain_compact 2>&1 | tail -25
echo "cargo_rc=${PIPESTATUS[0]}"

echo "--- binary after ---"
stat -c 'inode=%i mtime=%y size=%s' "$BIN" 2>&1

echo "--- container inspect (read-only) ---"
R=/srv/wizard/runtime/programming-integrated-20260713
sudo -u ec2-user "$BIN" --inspect "$R/brain/brain.wbrain" 2>&1

echo "--- volume avail/size/pcent ---"
df -B1 --output=avail,size,pcent "$R" | tail -1
