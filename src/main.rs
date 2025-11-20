use simfutures::config::RunConfig;
use simfutures::run_with_config;
use std::path::PathBuf;
use tracing::info;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Parallel future simulation CLI")]
struct Cli {
    /// Path to run configuration file (JSON/TOML supported via serde_json for now)
    #[arg(long, default_value = "run_config.json")]
    config: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config_text = std::fs::read_to_string(&cli.config)?;
    let config: RunConfig = serde_json::from_str(&config_text)?;
    let results = run_with_config(config)?;
    println!(
        "Best energy: {:.3} | symbols: {}",
        results.best_energy,
        results.best_state.symbol_states.len()
    );
    info!(
        target: "simfutures::cli",
        best_energy = results.best_energy,
        best_symbols = results.best_state.symbol_states.len(),
        "run completed"
    );
    Ok(())
}
