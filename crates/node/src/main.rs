use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::info;
use tracing_subscriber::EnvFilter;
use serde::{Deserialize, Serialize};

mod chain;
mod bridge;
mod api;
mod config;
mod data_mesh;
mod identity;
mod label_queue;
mod ledger;
mod openstack;
mod paths;
mod p2p;
mod performance;
mod runtime;
mod sim;
mod wallet;

use config::NodeConfig;
use runtime::NodeRuntime;
use api::run_api;
use label_queue::{
    load_latest_fabric_share, load_latest_subnet_report, parse_label_queue,
    parse_visual_label_queue,
};
use wallet::{WalletStore, node_id_from_wallet};
use reqwest::blocking::Client;
use w1z4rdv1510n::hardware::HardwareProfile;
use w1z4rdv1510n::config::RunConfig;
use w1z4rdv1510n::blockchain::{BridgeIntent, bridge_intent_id, bridge_intent_payload};
use w1z4rdv1510n::bridge::ChainKind;
use w1z4rdv1510n::network::compute_payload_hash;
use w1z4rdv1510n::schema::Timestamp;
use w1z4rdv1510n::streaming::{
    AssociationVote, FigureAssociationTask, KnowledgeAssociation, KnowledgeDocument,
    KnowledgeIngestConfig, KnowledgeIngestReport, KnowledgeQueue, LabelQueueReport,
    NetworkPatternSummary, NlmJatsIngestor, VisualLabelReport,
};
use crate::data_mesh::PatternResponse;

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
    Sim {
        #[arg(long, default_value_t = 10000)]
        nodes: usize,
        #[arg(long, default_value_t = 100)]
        ticks: usize,
        #[arg(long, default_value_t = 100000)]
        max_messages_per_tick: usize,
        #[arg(long, default_value_t = 0.05)]
        work_rate: f64,
        #[arg(long, default_value_t = 0.02)]
        validation_fail_rate: f64,
        #[arg(long, default_value_t = 2)]
        fanout: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        throttle_ms: u64,
        #[arg(long)]
        max_nodes: Option<usize>,
        #[arg(long, default_value_t = 200000)]
        max_queue_depth: usize,
        #[arg(long)]
        out: Option<String>,
    },
    Api {
        #[arg(long, default_value = "127.0.0.1:8090")]
        addr: String,
    },
    BridgeIntentCreate {
        #[arg(long)]
        chain_id: String,
        #[arg(long, default_value = "OTHER")]
        chain_kind: String,
        #[arg(long)]
        asset: String,
        #[arg(long)]
        amount: f64,
        #[arg(long)]
        recipient_node_id: String,
        #[arg(long)]
        deposit_address: String,
        #[arg(long)]
        recipient_tag: Option<String>,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long)]
        created_at_unix: Option<i64>,
        #[arg(long)]
        out: Option<String>,
    },
    BridgeIntentVerify {
        #[arg(long)]
        file: Option<String>,
        #[arg(long)]
        json: Option<String>,
        #[arg(long)]
        expected_hash: Option<String>,
    },
    KnowledgeIngest {
        #[arg(long)]
        xml_file: Option<String>,
        #[arg(long)]
        xml: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        asset_root: Option<String>,
        #[arg(long)]
        require_image_bytes: bool,
        #[arg(long)]
        normalize_whitespace: bool,
        #[arg(long)]
        include_ocr_blocks: bool,
        #[arg(long, num_args = 1..)]
        ocr_command: Vec<String>,
        #[arg(long, default_value_t = 30)]
        ocr_timeout_secs: u64,
        #[arg(long)]
        out: Option<String>,
    },
    KnowledgeVote {
        #[arg(long)]
        ingest_file: Option<String>,
        #[arg(long)]
        ingest_json: Option<String>,
        #[arg(long)]
        votes_file: Option<String>,
        #[arg(long)]
        votes_json: Option<String>,
        #[arg(long)]
        out: Option<String>,
    },
    LabelQueue {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        out: Option<String>,
    },
    VisualLabelQueue {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        out: Option<String>,
    },
    SubnetReport {
        #[arg(long)]
        out: Option<String>,
    },
    HashApiKey {
        #[arg(long)]
        key: String,
    },
    MetacognitionTune {
        #[arg(long)]
        min_depth: Option<usize>,
        #[arg(long)]
        max_depth: Option<usize>,
        #[arg(long)]
        confident_depth: Option<usize>,
        #[arg(long)]
        accuracy_target: Option<f64>,
        #[arg(long)]
        confident_uncertainty_threshold: Option<f64>,
        #[arg(long)]
        novelty_depth_boost: Option<usize>,
        #[arg(long)]
        min_depth_samples: Option<u64>,
        #[arg(long)]
        depth_improvement_margin: Option<f64>,
        #[arg(long)]
        out: Option<String>,
    },
    PatternQuery {
        #[arg(long, default_value = "http://127.0.0.1:8090")]
        api_url: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        api_key_env: Option<String>,
        #[arg(long)]
        phenotype_hash: Option<String>,
        #[arg(long, value_delimiter = ',')]
        phenotype_tokens: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        behavior_signature: Vec<f64>,
        #[arg(long)]
        behavior_signature_file: Option<String>,
        #[arg(long)]
        species: Option<String>,
        #[arg(long)]
        max_results: Option<usize>,
        #[arg(long)]
        min_similarity: Option<f64>,
        #[arg(long)]
        broadcast: Option<bool>,
        #[arg(long)]
        wait_for_responses_ms: Option<u64>,
        #[arg(long)]
        out: Option<String>,
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
        Some(Command::Sim {
            nodes,
            ticks,
            max_messages_per_tick,
            work_rate,
            validation_fail_rate,
            fanout,
            seed,
            throttle_ms,
            max_nodes,
            max_queue_depth,
            out,
        }) => {
            let report = sim::run_simulation(sim::SimConfig {
                nodes,
                ticks,
                max_messages_per_tick,
                work_rate,
                validation_fail_rate,
                fanout,
                seed,
                throttle_ms,
                max_nodes,
                max_queue_depth,
            });
            let payload = serde_json::to_string_pretty(&report)?;
            println!("{payload}");
            if let Some(path) = out {
                fs::write(path, payload)?;
            }
            Ok(())
        }
        Some(Command::Api { addr }) => run_api_mode(&config_path, &addr),
        Some(Command::BridgeIntentCreate {
            chain_id,
            chain_kind,
            asset,
            amount,
            recipient_node_id,
            deposit_address,
            recipient_tag,
            idempotency_key,
            created_at_unix,
            out,
        }) => bridge_intent_create(
            chain_id,
            chain_kind,
            asset,
            amount,
            recipient_node_id,
            deposit_address,
            recipient_tag,
            idempotency_key,
            created_at_unix,
            out,
        ),
        Some(Command::BridgeIntentVerify { file, json, expected_hash }) => {
            bridge_intent_verify(file, json, expected_hash)
        }
        Some(Command::KnowledgeIngest {
            xml_file,
            xml,
            source,
            asset_root,
            require_image_bytes,
            normalize_whitespace,
            include_ocr_blocks,
            ocr_command,
            ocr_timeout_secs,
            out,
        }) => knowledge_ingest(
            xml_file,
            xml,
            source,
            asset_root,
            require_image_bytes,
            normalize_whitespace,
            include_ocr_blocks,
            ocr_command,
            ocr_timeout_secs,
            out,
        ),
        Some(Command::KnowledgeVote {
            ingest_file,
            ingest_json,
            votes_file,
            votes_json,
            out,
        }) => knowledge_vote(ingest_file, ingest_json, votes_file, votes_json, out),
        Some(Command::LabelQueue { limit, out }) => {
            label_queue_report(&config_path, limit, out)
        }
        Some(Command::VisualLabelQueue { limit, out }) => {
            visual_label_queue_report(&config_path, limit, out)
        }
        Some(Command::SubnetReport { out }) => subnet_report(&config_path, out),
        Some(Command::HashApiKey { key }) => {
            println!("{}", api::hash_api_key_hex(&key));
            Ok(())
        }
        Some(Command::MetacognitionTune {
            min_depth,
            max_depth,
            confident_depth,
            accuracy_target,
            confident_uncertainty_threshold,
            novelty_depth_boost,
            min_depth_samples,
            depth_improvement_margin,
            out,
        }) => metacognition_tune(
            &config_path,
            min_depth,
            max_depth,
            confident_depth,
            accuracy_target,
            confident_uncertainty_threshold,
            novelty_depth_boost,
            min_depth_samples,
            depth_improvement_margin,
            out,
        ),
        Some(Command::PatternQuery {
            api_url,
            api_key,
            api_key_env,
            phenotype_hash,
            phenotype_tokens,
            behavior_signature,
            behavior_signature_file,
            species,
            max_results,
            min_similarity,
            broadcast,
            wait_for_responses_ms,
            out,
        }) => pattern_query(
            &config_path,
            api_url,
            api_key,
            api_key_env,
            phenotype_hash,
            phenotype_tokens,
            behavior_signature,
            behavior_signature_file,
            species,
            max_results,
            min_similarity,
            broadcast,
            wait_for_responses_ms,
            out,
        ),
        None => run_node(&config_path),
    }
}

fn run_node(config_path: &Path) -> anyhow::Result<()> {
    let config = load_or_create_config(config_path)?;
    config.validate()?;
    info!(
        target: "w1z4rdv1510n::node",
        node_id = config.node_id,
        "loaded node config"
    );
    let runtime = NodeRuntime::new(config)?;
    runtime.run_until_shutdown()
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

fn run_api_mode(config_path: &Path, addr: &str) -> anyhow::Result<()> {
    let config = load_or_create_config(config_path)?;
    config.validate()?;
    let addr: std::net::SocketAddr = addr.parse()?;
    info!(
        target: "w1z4rdv1510n::node",
        api_addr = %addr,
        "starting node api"
    );
    run_api(config, addr)
}

fn metacognition_tune(
    config_path: &Path,
    min_depth: Option<usize>,
    max_depth: Option<usize>,
    confident_depth: Option<usize>,
    accuracy_target: Option<f64>,
    confident_uncertainty_threshold: Option<f64>,
    novelty_depth_boost: Option<usize>,
    min_depth_samples: Option<u64>,
    depth_improvement_margin: Option<f64>,
    out: Option<String>,
) -> anyhow::Result<()> {
    let config = load_or_create_config(config_path)?;
    let run_config_path = PathBuf::from(&config.streaming.run_config_path);
    let raw = fs::read_to_string(&run_config_path)?;
    let mut run_config: RunConfig = serde_json::from_str(&raw)?;
    let (min_depth_value, max_depth_value, confident_depth_value, accuracy_target_value, uncertainty_threshold_value);
    {
        let meta = &mut run_config.streaming.metacognition;
        if let Some(value) = min_depth {
            meta.min_reflection_depth = value;
        }
        if let Some(value) = max_depth {
            meta.max_reflection_depth = value;
        }
        if let Some(value) = confident_depth {
            meta.confident_depth = value;
        }
        if let Some(value) = accuracy_target {
            meta.accuracy_target = value;
        }
        if let Some(value) = confident_uncertainty_threshold {
            meta.confident_uncertainty_threshold = value;
        }
        if let Some(value) = novelty_depth_boost {
            meta.novelty_depth_boost = value;
        }
        if let Some(value) = min_depth_samples {
            meta.min_depth_samples = value;
        }
        if let Some(value) = depth_improvement_margin {
            meta.depth_improvement_margin = value;
        }
        min_depth_value = meta.min_reflection_depth;
        max_depth_value = meta.max_reflection_depth;
        confident_depth_value = meta.confident_depth;
        accuracy_target_value = meta.accuracy_target;
        uncertainty_threshold_value = meta.confident_uncertainty_threshold;
    }

    let payload = serde_json::to_string_pretty(&run_config)?;
    fs::write(&run_config_path, payload.as_bytes())?;
    if let Some(path) = out {
        fs::write(path, payload.as_bytes())?;
    }
    println!(
        "Updated metacognition settings in {} (min_depth={}, max_depth={}, confident_depth={}, accuracy_target={:.2}, uncertainty_threshold={:.2}).",
        run_config_path.display(),
        min_depth_value,
        max_depth_value,
        confident_depth_value,
        accuracy_target_value,
        uncertainty_threshold_value
    );
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

fn pattern_query(
    config_path: &Path,
    api_url: String,
    api_key: Option<String>,
    api_key_env: Option<String>,
    phenotype_hash: Option<String>,
    phenotype_tokens: Vec<String>,
    behavior_signature: Vec<f64>,
    behavior_signature_file: Option<String>,
    species: Option<String>,
    max_results: Option<usize>,
    min_similarity: Option<f64>,
    broadcast: Option<bool>,
    wait_for_responses_ms: Option<u64>,
    out: Option<String>,
) -> anyhow::Result<()> {
    let config = if config_path.exists() {
        load_config(config_path)?
    } else {
        NodeConfig::default()
    };
    let mut signature = behavior_signature;
    if let Some(path) = behavior_signature_file {
        let raw = fs::read_to_string(path)?;
        let mut from_file: Vec<f64> = serde_json::from_str(&raw)?;
        signature.append(&mut from_file);
    }
    if signature.is_empty() {
        anyhow::bail!("behavior_signature is required (use --behavior-signature or --behavior-signature-file)");
    }
    let request = PatternQueryRequest {
        phenotype_hash,
        phenotype_tokens: if phenotype_tokens.is_empty() {
            None
        } else {
            Some(phenotype_tokens)
        },
        behavior_signature: signature,
        species,
        max_results,
        min_similarity,
        broadcast,
        wait_for_responses_ms,
    };
    let api_key = resolve_api_key(&config, api_key, api_key_env);
    if config.api.require_api_key && api_key.is_none() {
        anyhow::bail!("api key required (set --api-key or --api-key-env)");
    }
    let url = normalize_pattern_query_url(&api_url);
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let mut builder = client.post(url).json(&request);
    if let Some(key) = api_key {
        builder = builder.header(&config.api.api_key_header, key);
    }
    let response = builder.send()?;
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        anyhow::bail!("pattern query failed ({}): {}", status, body);
    }
    let parsed: PatternQueryResponse = serde_json::from_str(&body)?;
    let payload = serde_json::to_string_pretty(&parsed)?;
    println!("{payload}");
    if let Some(path) = out {
        fs::write(path, payload.as_bytes())?;
    }
    Ok(())
}

fn resolve_api_key(
    config: &NodeConfig,
    api_key: Option<String>,
    api_key_env: Option<String>,
) -> Option<String> {
    if let Some(key) = api_key {
        return Some(key);
    }
    let env_key = api_key_env.unwrap_or_else(|| config.api.api_key_env.clone());
    if env_key.trim().is_empty() {
        return None;
    }
    std::env::var(env_key).ok()
}

fn normalize_pattern_query_url(api_url: &str) -> String {
    let trimmed = api_url.trim();
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    if with_scheme.ends_with("/network/patterns/query") {
        with_scheme
    } else {
        format!("{}/network/patterns/query", with_scheme.trim_end_matches('/'))
    }
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
struct BridgeIntentOutput {
    intent: BridgeIntent,
    payload: String,
    payload_hash: String,
}

#[derive(Debug, Serialize)]
struct PatternQueryRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    phenotype_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phenotype_tokens: Option<Vec<String>>,
    behavior_signature: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    species: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_results: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_similarity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    broadcast: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wait_for_responses_ms: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PatternQueryResponse {
    status: String,
    #[serde(default)]
    query_id: Option<String>,
    #[serde(default)]
    local_matches: Vec<NetworkPatternSummary>,
    #[serde(default)]
    responses: Vec<PatternResponse>,
}

#[derive(Debug, serde::Serialize)]
struct BridgeIntentVerifyOutput {
    status: &'static str,
    intent_id: String,
    payload_hash: String,
    payload: String,
}

fn bridge_intent_create(
    chain_id: String,
    chain_kind: String,
    asset: String,
    amount: f64,
    recipient_node_id: String,
    deposit_address: String,
    recipient_tag: Option<String>,
    idempotency_key: Option<String>,
    created_at_unix: Option<i64>,
    out: Option<String>,
) -> anyhow::Result<()> {
    let chain_kind = parse_chain_kind(&chain_kind)?;
    let created_at = Timestamp {
        unix: created_at_unix.unwrap_or_else(now_unix),
    };
    let idempotency_key = idempotency_key.unwrap_or_else(|| {
        default_intent_key(&chain_id, &asset, amount, &recipient_node_id)
    });
    let mut intent = BridgeIntent {
        intent_id: String::new(),
        chain_id,
        chain_kind,
        asset,
        amount,
        recipient_node_id,
        deposit_address,
        recipient_tag,
        idempotency_key,
        created_at,
    };
    let intent_id = bridge_intent_id(&intent);
    intent.intent_id = intent_id.clone();
    let payload = bridge_intent_payload(&intent);
    let payload_hash = compute_payload_hash(payload.as_bytes());
    if payload_hash != intent_id {
        anyhow::bail!("bridge intent payload hash mismatch");
    }
    let output = BridgeIntentOutput {
        intent,
        payload,
        payload_hash,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    Ok(())
}

fn bridge_intent_verify(
    file: Option<String>,
    json: Option<String>,
    expected_hash: Option<String>,
) -> anyhow::Result<()> {
    let raw = match (file, json) {
        (Some(path), None) => fs::read_to_string(path)?,
        (None, Some(json)) => json,
        (None, None) => anyhow::bail!("provide --file or --json"),
        _ => anyhow::bail!("provide only one of --file or --json"),
    };
    let intent: BridgeIntent = serde_json::from_str(&raw)?;
    if intent.intent_id.trim().is_empty() {
        anyhow::bail!("intent_id must be provided");
    }
    let payload = bridge_intent_payload(&intent);
    let payload_hash = compute_payload_hash(payload.as_bytes());
    if payload_hash != intent.intent_id {
        anyhow::bail!("intent_id does not match payload hash");
    }
    if let Some(expected) = expected_hash {
        let expected = expected.trim().to_ascii_lowercase();
        if payload_hash != expected {
            anyhow::bail!("payload hash does not match expected value");
        }
    }
    let output = BridgeIntentVerifyOutput {
        status: "OK",
        intent_id: intent.intent_id,
        payload_hash,
        payload,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct KnowledgeIngestOutput {
    timestamp: Timestamp,
    document: KnowledgeDocument,
    report: KnowledgeIngestReport,
    pending_tasks: Vec<FigureAssociationTask>,
}

#[derive(Debug, Serialize, Deserialize)]
struct KnowledgeVoteOutput {
    status: &'static str,
    associations: Vec<KnowledgeAssociation>,
    pending_tasks: Vec<FigureAssociationTask>,
    total_pending: usize,
}

#[derive(Debug, Serialize)]
struct LabelQueueOutput {
    status: &'static str,
    updated_at: Option<Timestamp>,
    queue: Option<LabelQueueReport>,
}

#[derive(Debug, Serialize)]
struct VisualLabelQueueOutput {
    status: &'static str,
    updated_at: Option<Timestamp>,
    queue: Option<VisualLabelReport>,
}

#[derive(Debug, Serialize)]
struct SubnetReportOutput {
    status: &'static str,
    report: Option<w1z4rdv1510n::streaming::SubnetworkReport>,
}

fn knowledge_ingest(
    xml_file: Option<String>,
    xml: Option<String>,
    source: Option<String>,
    asset_root: Option<String>,
    require_image_bytes: bool,
    normalize_whitespace: bool,
    include_ocr_blocks: bool,
    ocr_command: Vec<String>,
    ocr_timeout_secs: u64,
    out: Option<String>,
) -> anyhow::Result<()> {
    let xml = load_text_input(xml_file, xml)?;
    let mut config = KnowledgeIngestConfig::default();
    if let Some(source) = source {
        if !source.trim().is_empty() {
            config.source = source;
        }
    }
    config.asset_root = asset_root.map(PathBuf::from);
    config.require_image_bytes = require_image_bytes;
    config.normalize_whitespace = normalize_whitespace;
    config.include_ocr_blocks = include_ocr_blocks;
    if !ocr_command.is_empty() {
        config.ocr_command = Some(ocr_command);
    }
    config.ocr_timeout_secs = ocr_timeout_secs.max(1);
    let timestamp = Timestamp { unix: now_unix() };
    let ingestor = NlmJatsIngestor::new(config);
    let document = ingestor.parse_str(&xml, timestamp)?;
    let mut queue = KnowledgeQueue::default();
    let report = queue.enqueue_document(document.clone(), timestamp);
    let pending_tasks = queue
        .pending_report(timestamp)
        .map(|report| report.pending)
        .unwrap_or_default();
    let output = KnowledgeIngestOutput {
        timestamp,
        document,
        report,
        pending_tasks,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    Ok(())
}

fn knowledge_vote(
    ingest_file: Option<String>,
    ingest_json: Option<String>,
    votes_file: Option<String>,
    votes_json: Option<String>,
    out: Option<String>,
) -> anyhow::Result<()> {
    let ingest = load_ingest_output(ingest_file, ingest_json)?;
    let votes = load_votes(votes_file, votes_json)?;
    let mut queue = KnowledgeQueue::default();
    queue.enqueue_document(ingest.document.clone(), ingest.timestamp);
    let mut associations = Vec::new();
    for vote in votes {
        if let Some(association) = queue.record_vote(vote) {
            associations.push(association);
        }
    }
    let report = queue.pending_report(Timestamp { unix: now_unix() });
    let (pending_tasks, total_pending) = match report {
        Some(report) => (report.pending, report.total_pending),
        None => (Vec::new(), 0),
    };
    let output = KnowledgeVoteOutput {
        status: "OK",
        associations,
        pending_tasks,
        total_pending,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    Ok(())
}

fn label_queue_report(
    config_path: &Path,
    limit: Option<usize>,
    out: Option<String>,
) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let share = load_latest_fabric_share(&config)?;
    let mut report = share.as_ref().and_then(parse_label_queue);
    if let Some(report) = report.as_mut() {
        apply_limit(&mut report.pending, limit);
    }
    let updated_at = report
        .as_ref()
        .and_then(|_| share.as_ref().map(|share| share.timestamp));
    let output = LabelQueueOutput {
        status: if report.is_some() { "OK" } else { "EMPTY" },
        updated_at,
        queue: report,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    Ok(())
}

fn visual_label_queue_report(
    config_path: &Path,
    limit: Option<usize>,
    out: Option<String>,
) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let share = load_latest_fabric_share(&config)?;
    let mut report = share.as_ref().and_then(parse_visual_label_queue);
    if let Some(report) = report.as_mut() {
        apply_limit(&mut report.pending, limit);
    }
    let updated_at = report
        .as_ref()
        .and_then(|_| share.as_ref().map(|share| share.timestamp));
    let output = VisualLabelQueueOutput {
        status: if report.is_some() { "OK" } else { "EMPTY" },
        updated_at,
        queue: report,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    Ok(())
}

fn subnet_report(
    config_path: &Path,
    out: Option<String>,
) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let report = load_latest_subnet_report(&config)?;
    let output = SubnetReportOutput {
        status: if report.is_some() { "OK" } else { "EMPTY" },
        report,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    Ok(())
}

fn apply_limit<T>(pending: &mut Vec<T>, limit: Option<usize>) {
    if let Some(limit) = limit {
        if pending.len() > limit {
            pending.truncate(limit);
        }
    }
}

fn parse_chain_kind(raw: &str) -> anyhow::Result<ChainKind> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        anyhow::bail!("chain_kind must be non-empty");
    }
    let kind = match normalized.as_str() {
        "evm" | "ethereum" => ChainKind::Evm,
        "solana" => ChainKind::Solana,
        "bitcoin" | "btc" => ChainKind::Bitcoin,
        "cosmos" => ChainKind::Cosmos,
        "other" => ChainKind::Other,
        _ => anyhow::bail!("unsupported chain_kind: {raw}"),
    };
    Ok(kind)
}

fn default_intent_key(
    chain_id: &str,
    asset: &str,
    amount: f64,
    recipient_node_id: &str,
) -> String {
    format!(
        "intent|{}|{}|{:.6}|{}",
        chain_id, asset, amount, recipient_node_id
    )
}

fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn load_text_input(file: Option<String>, inline: Option<String>) -> anyhow::Result<String> {
    match (file, inline) {
        (Some(path), None) => Ok(fs::read_to_string(path)?),
        (None, Some(text)) => Ok(text),
        (None, None) => anyhow::bail!("provide --xml or --xml-file"),
        _ => anyhow::bail!("provide only one of --xml or --xml-file"),
    }
}

fn load_ingest_output(file: Option<String>, json: Option<String>) -> anyhow::Result<KnowledgeIngestOutput> {
    let raw = match (file, json) {
        (Some(path), None) => fs::read_to_string(path)?,
        (None, Some(json)) => json,
        (None, None) => anyhow::bail!("provide --ingest-file or --ingest-json"),
        _ => anyhow::bail!("provide only one of --ingest-file or --ingest-json"),
    };
    let output: KnowledgeIngestOutput = serde_json::from_str(&raw)?;
    Ok(output)
}

fn load_votes(file: Option<String>, json: Option<String>) -> anyhow::Result<Vec<AssociationVote>> {
    let raw = match (file, json) {
        (Some(path), None) => fs::read_to_string(path)?,
        (None, Some(json)) => json,
        (None, None) => anyhow::bail!("provide --votes-file or --votes-json"),
        _ => anyhow::bail!("provide only one of --votes-file or --votes-json"),
    };
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    if value.is_array() {
        Ok(serde_json::from_value(value)?)
    } else {
        let vote: AssociationVote = serde_json::from_value(value)?;
        Ok(vec![vote])
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
