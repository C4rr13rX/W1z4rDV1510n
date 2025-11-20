use clap::Parser;
use simfutures::calibration::{calibrate_from_trajectories, EnergyCalibration};
use simfutures::config::{EnergyConfig, LoggingConfig};
use simfutures::logging::init_logging;
use simfutures::schema::Trajectory;
use std::fs;
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    /// Path to JSON file containing an array of Trajectory objects
    #[arg(long)]
    trajectories: PathBuf,
    /// Optional base energy config to merge with calibrated weights
    #[arg(long)]
    base_config: Option<PathBuf>,
    /// Optional output path for the calibrated EnergyConfig (JSON)
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_logging(&LoggingConfig::default())?;
    let data = fs::read_to_string(&args.trajectories)?;
    let trajectories: Vec<Trajectory> = serde_json::from_str(&data)?;
    let calibration: EnergyCalibration = calibrate_from_trajectories(&trajectories);
    let base = if let Some(path) = &args.base_config {
        let raw = fs::read_to_string(path)?;
        serde_json::from_str(&raw)?
    } else {
        EnergyConfig::default()
    };
    let calibrated = calibration.to_energy_config(&base);
    let json = serde_json::to_string_pretty(&calibrated)?;
    if let Some(output_path) = &args.output {
        fs::write(output_path, &json)?;
        info!(
            target: "simfutures::calibration",
            path = ?output_path,
            trajectories = trajectories.len(),
            "wrote calibrated energy config"
        );
    } else {
        println!("{json}");
        info!(
            target: "simfutures::calibration",
            trajectories = trajectories.len(),
            "printed calibrated energy config to stdout"
        );
    }
    Ok(())
}
