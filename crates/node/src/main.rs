use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;
use tracing_subscriber::EnvFilter;

mod chain;
mod config;
mod ledger;
mod openstack;
mod paths;
mod p2p;
mod runtime;
mod wallet;

use config::NodeConfig;
use runtime::NodeRuntime;
use wallet::{WalletStore, node_id_from_wallet};
use w1z4rdv1510n::hardware::HardwareProfile;

#[derive(Parser, Debug)]
#[command(name = "w1z4rdv1510n-node")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(long, default_value = "node_config.json")]
    config: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    Init {
        #[arg(long)]
        force: bool,
        #[arg(long)]
        report: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_level(true)
        .init();

    let config_path = PathBuf::from(&args.config);
    match args.command {
        Some(Command::Init { force, report }) => {
            let report_path = report.map(PathBuf::from);
            init_node(&config_path, force, report_path.as_deref())
        }
        None => run_node(&config_path),
    }
}

fn run_node(config_path: &Path) -> anyhow::Result<()> {
    let config = load_or_create_config(config_path)?;
    info!(
        target: "w1z4rdv1510n::node",
        node_id = config.node_id,
        "loaded node config"
    );
    let runtime = NodeRuntime::new(config)?;
    runtime.start()
}

fn init_node(config_path: &Path, force: bool, report_path: Option<&Path>) -> anyhow::Result<()> {
    let config = if config_path.exists() && !force {
        load_config(config_path)?
    } else {
        create_default_config(config_path)?
    };
    let wallet = WalletStore::load_or_create(&config.wallet)?;
    let profile = HardwareProfile::detect();
    let chain_ok = chain_status(&config);
    let report = InitReport {
        node_id: config.node_id.clone(),
        wallet_address: wallet.address.clone(),
        config_path: config_path.display().to_string(),
        data_dir: paths::node_data_dir().display().to_string(),
        listen_addr: config.network.listen_addr.clone(),
        bootstrap_peers: effective_bootstrap_peers(&config),
        cpu_cores: profile.cpu_cores,
        memory_gb: profile.total_memory_gb,
        has_gpu: profile.has_gpu,
        chain_ok,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = report_path {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn load_or_create_config(config_path: &Path) -> anyhow::Result<NodeConfig> {
    if config_path.exists() {
        return load_config(config_path);
    }
    create_default_config(config_path)
}

fn create_default_config(config_path: &Path) -> anyhow::Result<NodeConfig> {
    let mut config = NodeConfig::default();
    config.blockchain.enabled = true;
    config.ledger.enabled = true;
    config.ledger.backend = "local".to_string();
    let wallet = WalletStore::load_or_create(&config.wallet)?;
    config.node_id = node_id_from_wallet(&wallet.address);
    if let Some(parent) = config_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(config_path, serde_json::to_string_pretty(&config)?)?;
    Ok(config)
}

fn load_config(config_path: &Path) -> anyhow::Result<NodeConfig> {
    let raw = fs::read_to_string(config_path)?;
    let config: NodeConfig = serde_json::from_str(&raw)?;
    Ok(config)
}

fn effective_bootstrap_peers(config: &NodeConfig) -> Vec<String> {
    if !config.network.bootstrap_peers.is_empty() {
        return config.network.bootstrap_peers.clone();
    }
    if let Ok(value) = std::env::var("W1Z4RDV1510N_BOOTSTRAP_PEERS") {
        return value
            .split(',')
            .map(|entry| entry.trim().to_string())
            .filter(|entry| !entry.is_empty())
            .collect();
    }
    Vec::new()
}

fn chain_status(config: &NodeConfig) -> ChainStatus {
    match chain::ChainSpec::load(&config.chain_spec) {
        Ok(spec) => ChainStatus::Ready {
            chain_id: spec.chain_id,
            consensus: spec.consensus,
        },
        Err(err) => ChainStatus::Missing {
            error: err.to_string(),
        },
    }
}

#[derive(Debug, serde::Serialize)]
struct InitReport {
    node_id: String,
    wallet_address: String,
    config_path: String,
    data_dir: String,
    listen_addr: String,
    bootstrap_peers: Vec<String>,
    cpu_cores: usize,
    memory_gb: f64,
    has_gpu: bool,
    chain_ok: ChainStatus,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
enum ChainStatus {
    Ready { chain_id: String, consensus: String },
    Missing { error: String },
}
