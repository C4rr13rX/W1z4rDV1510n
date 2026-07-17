#[path = "../brain_api.rs"]
mod brain_api;

use anyhow::{Context, Result};
use std::path::PathBuf;
use sysinfo::System;

fn main() -> Result<()> {
    let data_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(brain_api::default_node_brain_dir);
    let legacy = data_dir.join("brain.bin");
    let checkpoint_bytes = std::fs::metadata(&legacy)
        .with_context(|| format!("inspect {}", legacy.display()))?
        .len();
    let mut system = System::new();
    system.refresh_memory();
    let available_bytes = system.available_memory();
    let reserve_bytes = 8_u64 * 1024 * 1024 * 1024;
    let required_bytes = reserve_bytes;
    anyhow::ensure!(
        available_bytes >= required_bytes,
        "refusing streaming migration: {:.2} GiB available while the host reserve requires {:.2} GiB (checkpoint size {:.2} GiB)",
        available_bytes as f64 / 1024_f64.powi(3),
        required_bytes as f64 / 1024_f64.powi(3),
        checkpoint_bytes as f64 / 1024_f64.powi(3),
    );
    let serialized = brain_api::migrate_legacy_brain_container(&data_dir)
        .with_context(|| format!("migrate brain in {}", data_dir.display()))?;
    println!(
        "migrated {serialized} individually-addressable neurons to {}",
        data_dir.join("brain.wbrain").display()
    );
    Ok(())
}
