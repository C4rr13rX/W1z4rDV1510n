//! Reclaim superseded neuron bodies from a `.wbrain` container.
//!
//! The container is append-only and has never had a compactor, so every
//! eviction since the first cut has left its previous body on disk. Measured on
//! the private training host 2026-09-10: 576.7 GB of file for 5,086,420
//! neurons, growing 206–257 GB/h and about two hours from filling a 1.1 TB
//! volume.
//!
//! # Usage
//!
//! ```text
//! wbrain_compact --inspect  <brain.wbrain>
//! wbrain_compact --in-place <brain.wbrain> [--keep-retired <path>]
//! wbrain_compact <source.wbrain> <destination.wbrain>
//! ```
//!
//! The brain MUST NOT be running. This tool takes no lock: the container has no
//! locking protocol, and a live server appends to the source while it is being
//! copied, which would publish a manifest pointing at records the copy never
//! saw. `--inspect` is always safe.
//!
//! `--in-place` renames the original aside, installs the compacted file, and
//! only then unlinks the original — so an interruption leaves a complete brain
//! on disk under one name or the other. Space is returned at the unlink, which
//! is also the only point `df` will move.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use w1z4rd_brain::store::compaction;
use w1z4rd_brain::store::container::BrainContainer;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         wbrain_compact --inspect <brain.wbrain>\n  \
         wbrain_compact --in-place <brain.wbrain> [--keep-retired <path>]\n  \
         wbrain_compact <source.wbrain> <destination.wbrain>"
    );
    ExitCode::from(2)
}

/// Report what the container holds without modifying it.
///
/// Prints the reference classes a compaction must remap, so a pass is never the
/// first thing to discover that a brain uses an addressing form the tool has
/// not been taught.
fn inspect(path: &Path) -> std::io::Result<()> {
    let container = BrainContainer::open(path)?;
    let bytes = container.byte_len()?;
    let manifest = container.manifest().cloned().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "container has no committed manifest",
        )
    })?;
    let mut live = 0_u64;
    let mut slot_tables = 0_u64;
    let mut offset_vectors = 0_u64;
    let mut label_indexes = 0_u64;
    let mut metadata_bytes = 0_u64;
    for pool in &manifest.pools {
        if pool.neuron_slot_table.is_some() {
            slot_tables += 1;
        }
        if !pool.neuron_offsets.is_empty() {
            offset_vectors += 1;
            live += pool.neuron_offsets.iter().filter(|s| s.is_some()).count() as u64;
        }
        label_indexes += pool.label_indexes.len() as u64;
        metadata_bytes += pool.pool_metadata.len() as u64;
    }
    println!("path                 {}", path.display());
    println!("bytes                {bytes} ({:.2} GB)", bytes as f64 / 1e9);
    println!("generation           {}", manifest.generation);
    println!("tick                 {}", manifest.tick);
    println!("pools                {}", manifest.pools.len());
    println!("pools_with_slot_table {slot_tables}");
    println!("pools_with_offset_vec {offset_vectors}");
    println!("live_in_offset_vecs  {live}");
    println!("label_index_records  {label_indexes}");
    println!("pool_metadata_bytes  {metadata_bytes}");
    println!("brain_metadata_bytes {}", manifest.brain_metadata.len());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return usage();
    }

    if args[0] == "--inspect" {
        if args.len() != 2 {
            return usage();
        }
        return match inspect(Path::new(&args[1])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("inspect failed: {error}");
                ExitCode::FAILURE
            }
        };
    }

    let started = std::time::Instant::now();
    let (report, target) = if args[0] == "--in-place" {
        if args.len() != 2 && args.len() != 4 {
            return usage();
        }
        let path = PathBuf::from(&args[1]);
        let keep = if args.len() == 4 {
            if args[2] != "--keep-retired" {
                return usage();
            }
            Some(PathBuf::from(&args[3]))
        } else {
            None
        };
        match compaction::compact_in_place(&path, keep.as_deref()) {
            Ok(report) => (report, path),
            Err(error) => {
                eprintln!("compaction failed: {error}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        if args.len() != 2 {
            return usage();
        }
        let source = PathBuf::from(&args[0]);
        let destination = PathBuf::from(&args[1]);
        let report = match compaction::compact(&source, &destination) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("compaction failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        // A two-path run leaves both files in place, so verification is the
        // caller's only evidence before they swap.
        if let Err(error) = compaction::verify(&destination, &report) {
            eprintln!("VERIFICATION FAILED, do not install this file: {error}");
            return ExitCode::FAILURE;
        }
        (report, destination)
    };

    println!(
        "{}",
        serde_json::json!({
            "target": target.display().to_string(),
            "pools": report.pools,
            "neurons_copied": report.neurons_copied,
            "auxiliary_copied": report.auxiliary_copied,
            "generation_directories_rewritten": report.generation_directories_rewritten,
            "source_bytes": report.source_bytes,
            "destination_bytes": report.destination_bytes,
            "reclaimed_bytes": report.reclaimed_bytes(),
            "reclaimed_gb": (report.reclaimed_bytes() as f64 / 1e9 * 100.0).round() / 100.0,
            "elapsed_seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        })
    );
    ExitCode::SUCCESS
}
