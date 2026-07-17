#[path = "../brain_api.rs"]
mod brain_api;

use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let data_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(brain_api::default_node_brain_dir);
    let serialized = brain_api::migrate_legacy_brain_container(&data_dir)
        .with_context(|| format!("migrate brain in {}", data_dir.display()))?;
    println!(
        "migrated {serialized} individually-addressable neurons to {}",
        data_dir.join("brain.wbrain").display()
    );
    Ok(())
}
