use crate::config::{KnowledgeConfig, NodeConfig};
use crate::data_mesh::{
    DataIngestRequest, DataMeshEvent, DataMeshHandle, PatternQuery, start_data_mesh,
};
use crate::identity::{
    IdentityChallengeRequest, IdentityChallengeResponse, IdentityRuntime, IdentityStatusResponse,
    IdentityVerifyRequest, IdentityVerifyResponse,
};
use crate::label_queue::{parse_label_queue, parse_subnet_report, parse_visual_label_queue};
use crate::ledger::LocalLedger;
use crate::p2p::NodeNetwork;
use crate::paths::node_data_dir;
use crate::wallet::{WalletStore, node_id_from_wallet};
use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use blake2::{Blake2s256, Digest};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::fs;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::time::{timeout, Duration};
use tracing::warn;
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, BridgeIntent, NoopLedger, RewardBalance, bridge_intent_id,
};
use w1z4rdv1510n::neuro::{NeuroRuntime, NeuroRuntimeConfig, NeuroRuntimeHandle, NeuroSnapshot};
use w1z4rdv1510n::schema::EnvironmentSnapshot;
use w1z4rdv1510n::threat::{ThreatOverlay, ThreatScene};
use w1z4rdv1510n::bridge::{BridgeProof, BridgeVerificationMode, ChainKind, bridge_deposit_id};
use w1z4rdv1510n::config::{BridgeConfig, BridgeChainPolicy, RunConfig};
use w1z4rdv1510n::schema::Timestamp;
use w1z4rdv1510n::streaming::{
    AssociationVote, FigureAssociationTask, KnowledgeAssociation, KnowledgeDocument,
    HealthKnowledgeStore, KnowledgeIngestConfig, KnowledgeIngestReport, KnowledgeQueue,
    KnowledgeQueueReport, KnowledgeRuntime, LabelQueueReport, NeuralFabricShare,
    NetworkPatternSummary, NlmJatsIngestor, SubnetworkReport, VisualLabelReport,
    QaCandidateRecord, QaQueryReport, QaRuntime, QaRuntimeConfig,
    EquationMatrixRuntime, EquationMatrixConfig, Discipline, EemPeerPayload,
    ImageBitsConfig, ImageBitsEncoder,
    AudioBitsConfig, AudioBitsEncoder,
    TextBitsConfig, TextBitsEncoder, TextSpan, TextRole, TextEmphasis,
    MotionBitsConfig, MotionBitsEncoder, MotionSample,
    KeyboardBitsConfig, KeyboardBitsEncoder, KeyEvent,
};
use w1z4rdv1510n::causal::{CausalEdge, CausalRuntime};
use w1z4rdv1510n_cluster::{ClusterConfig, ClusterNode};

#[derive(Clone)]
struct ApiState {
    node_id: String,
    ledger_backend: String,
    data_enabled: bool,
    sensor_ingest_enabled: bool,
    bridge_enabled: bool,
    knowledge_enabled: bool,
    identity_enabled: bool,
    started_at: Instant,
    ledger: Arc<dyn BlockchainLedger>,
    bridge_config: BridgeConfig,
    require_api_key: bool,
    api_key_hashes: Vec<[u8; 32]>,
    api_key_header: HeaderName,
    limiter: Arc<Mutex<ApiLimiter>>,
    rate_limit_default_max: u32,
    rate_limit_bridge_max: u32,
    rate_limit_balance_max: u32,
    data_mesh: Option<DataMeshHandle>,
    metrics: Arc<ApiMetrics>,
    knowledge: Arc<Mutex<KnowledgeRuntime>>,
    knowledge_persist: KnowledgePersist,
    qa_runtime: Arc<Mutex<QaRuntime>>,
    identity: Arc<Mutex<IdentityRuntime>>,
    label_state: Arc<Mutex<LabelQueueState>>,
    fabric_share_kind: String,
    run_config_path: PathBuf,
    neuro: NeuroRuntimeHandle,
    threat: Arc<Mutex<ThreatScene>>,
    equation_matrix: Arc<EquationMatrixRuntime>,
    /// Named-process causal graph: sensor label clusters → identified physics
    /// processes. Edges are written by equations_apply; read via /causal/graph.
    causal: Arc<Mutex<CausalRuntime>>,
    /// First-reporter index for cross-node pattern source tracing.
    origin_index: Arc<Mutex<HashMap<String, PatternOrigin>>>,
    /// Live cluster node — None until cluster-init or cluster-join is called
    /// via POST /cluster/init or POST /cluster/join.
    cluster: Arc<tokio::sync::Mutex<Option<ClusterNode>>>,
    /// LAN address to advertise to cluster peers (from network.advertise_addr in config).
    /// When None the bind address is used, which means peers see 0.0.0.0.
    cluster_advertise_addr: Option<std::net::SocketAddr>,
}

#[derive(Clone)]
struct KnowledgePersist {
    enabled: bool,
    path: PathBuf,
}

#[derive(Default)]
struct LabelQueueState {
    label_queue: Option<LabelQueueReport>,
    visual_label_queue: Option<VisualLabelReport>,
    subnet_report: Option<SubnetworkReport>,
    updated_at: Option<Timestamp>,
}

/// Records which node first reported a pattern and how many nodes have since
/// corroborated it — the primitive for cross-node source tracing.
#[derive(Debug, Clone, Serialize)]
struct PatternOrigin {
    thread_id: String,
    first_seen_unix: u64,
    first_reporter_node_id: String,
    report_count: u32,
    reporting_nodes: Vec<String>,
}

#[derive(Serialize)]
struct BridgeSubmitResponse {
    status: &'static str,
    deposit_id: String,
    balance: Option<RewardBalance>,
}

#[derive(Deserialize)]
struct BridgeIntentRequest {
    chain_id: String,
    asset: String,
    amount: f64,
    recipient_node_id: String,
    #[serde(default)]
    idempotency_key: Option<String>,
}

#[derive(Deserialize)]
struct MetacognitionTuneRequest {
    #[serde(default)]
    min_depth: Option<usize>,
    #[serde(default)]
    max_depth: Option<usize>,
    #[serde(default)]
    confident_depth: Option<usize>,
    #[serde(default)]
    accuracy_target: Option<f64>,
    #[serde(default)]
    confident_uncertainty_threshold: Option<f64>,
    #[serde(default)]
    novelty_depth_boost: Option<usize>,
    #[serde(default)]
    min_depth_samples: Option<u64>,
    #[serde(default)]
    depth_improvement_margin: Option<f64>,
}

#[derive(Deserialize)]
struct PatternQueryRequest {
    #[serde(default)]
    phenotype_hash: Option<String>,
    #[serde(default)]
    phenotype_tokens: Option<Vec<String>>,
    #[serde(default)]
    behavior_signature: Vec<f64>,
    #[serde(default)]
    species: Option<String>,
    #[serde(default)]
    max_results: Option<usize>,
    #[serde(default)]
    min_similarity: Option<f64>,
    #[serde(default)]
    broadcast: Option<bool>,
    #[serde(default)]
    wait_for_responses_ms: Option<u64>,
}

#[derive(Serialize)]
struct PatternQueryResponse {
    status: String,
    #[serde(default)]
    query_id: Option<String>,
    #[serde(default)]
    local_matches: Vec<NetworkPatternSummary>,
    #[serde(default)]
    responses: Vec<crate::data_mesh::PatternResponse>,
}

#[derive(Serialize)]
struct MetacognitionTuneResponse {
    status: &'static str,
    metacognition: serde_json::Value,
}

#[derive(Serialize)]
struct BridgeIntentResponse {
    status: &'static str,
    intent_id: String,
    created_at: Timestamp,
    chain_id: String,
    chain_kind: ChainKind,
    asset: String,
    amount: f64,
    recipient_node_id: String,
    deposit_address: String,
    recipient_tag: Option<String>,
    min_confirmations: u32,
    verification: BridgeVerificationMode,
    relayer_quorum: u32,
    max_deposit_amount: f64,
}

#[derive(Serialize)]
struct BridgeChainInfo {
    chain_id: String,
    chain_kind: ChainKind,
    verification: BridgeVerificationMode,
    min_confirmations: u32,
    relayer_quorum: u32,
    allowed_assets: Vec<String>,
    max_deposit_amount: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    deposit_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recipient_tag_template: Option<String>,
}

#[derive(Default, Deserialize)]
struct KnowledgeIngestOptions {
    source: Option<String>,
    require_image_bytes: Option<bool>,
    normalize_whitespace: Option<bool>,
    include_ocr_blocks: Option<bool>,
    ocr_command: Option<Vec<String>>,
    ocr_timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
struct KnowledgeIngestRequest {
    #[serde(default)]
    xml: Option<String>,
    #[serde(default)]
    document: Option<KnowledgeDocument>,
    #[serde(default)]
    options: KnowledgeIngestOptions,
    #[serde(default)]
    include_tasks: bool,
}

#[derive(Deserialize)]
struct KnowledgeQueueQuery {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Default, Deserialize)]
struct QueueLimitQuery {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct KnowledgeIngestResponse {
    status: &'static str,
    report: KnowledgeIngestReport,
    pending: Option<Vec<FigureAssociationTask>>,
}

#[derive(Serialize)]
struct KnowledgeQueueResponse {
    status: &'static str,
    report: Option<KnowledgeQueueReport>,
}

#[derive(Serialize)]
struct KnowledgeVoteResponse {
    status: &'static str,
    association: Option<KnowledgeAssociation>,
}

#[derive(Serialize)]
struct LabelQueueSnapshot {
    status: &'static str,
    updated_at: Option<Timestamp>,
    queue: Option<LabelQueueReport>,
}

#[derive(Serialize)]
struct VisualLabelQueueSnapshot {
    status: &'static str,
    updated_at: Option<Timestamp>,
    queue: Option<VisualLabelReport>,
}

#[derive(Serialize)]
struct SubnetworkSnapshot {
    status: &'static str,
    updated_at: Option<Timestamp>,
    report: Option<SubnetworkReport>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct ApiHealthSnapshot {
    status: &'static str,
    node_id: String,
    uptime_secs: u64,
    data_mesh_enabled: bool,
    data_mesh_active: bool,
    bridge_enabled: bool,
    ledger_backend: String,
}

#[derive(Serialize)]
struct ApiMetricsSnapshot {
    requests_total: u64,
    health_requests: u64,
    bridge_requests: u64,
    balance_requests: u64,
    metrics_requests: u64,
    health_success: u64,
    bridge_chain_requests: u64,
    bridge_intent_requests: u64,
    data_ingest_requests: u64,
    data_status_requests: u64,
    knowledge_ingest_requests: u64,
    knowledge_queue_requests: u64,
    knowledge_vote_requests: u64,
    label_queue_requests: u64,
    visual_label_queue_requests: u64,
    subnet_requests: u64,
    identity_challenge_requests: u64,
    identity_verify_requests: u64,
    bridge_success: u64,
    balance_success: u64,
    bridge_chain_success: u64,
    bridge_intent_success: u64,
    data_ingest_success: u64,
    data_status_success: u64,
    knowledge_ingest_success: u64,
    knowledge_queue_success: u64,
    knowledge_vote_success: u64,
    label_queue_success: u64,
    visual_label_queue_success: u64,
    subnet_success: u64,
    identity_challenge_success: u64,
    identity_verify_success: u64,
    rate_limit_hits: u64,
    auth_failures: u64,
}

#[derive(Default)]
struct ApiMetrics {
    requests_total: AtomicU64,
    health_requests: AtomicU64,
    bridge_requests: AtomicU64,
    balance_requests: AtomicU64,
    metrics_requests: AtomicU64,
    health_success: AtomicU64,
    bridge_chain_requests: AtomicU64,
    bridge_intent_requests: AtomicU64,
    data_ingest_requests: AtomicU64,
    data_status_requests: AtomicU64,
    knowledge_ingest_requests: AtomicU64,
    knowledge_queue_requests: AtomicU64,
    knowledge_vote_requests: AtomicU64,
    label_queue_requests: AtomicU64,
    visual_label_queue_requests: AtomicU64,
    subnet_requests: AtomicU64,
    identity_challenge_requests: AtomicU64,
    identity_verify_requests: AtomicU64,
    bridge_success: AtomicU64,
    balance_success: AtomicU64,
    bridge_chain_success: AtomicU64,
    bridge_intent_success: AtomicU64,
    data_ingest_success: AtomicU64,
    data_status_success: AtomicU64,
    knowledge_ingest_success: AtomicU64,
    knowledge_queue_success: AtomicU64,
    knowledge_vote_success: AtomicU64,
    label_queue_success: AtomicU64,
    visual_label_queue_success: AtomicU64,
    subnet_success: AtomicU64,
    identity_challenge_success: AtomicU64,
    identity_verify_success: AtomicU64,
    rate_limit_hits: AtomicU64,
    auth_failures: AtomicU64,
}

impl ApiMetrics {
    fn inc_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_bridge_request(&self) {
        self.inc_request();
        self.bridge_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_health_request(&self) {
        self.inc_request();
        self.health_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_balance_request(&self) {
        self.inc_request();
        self.balance_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_metrics_request(&self) {
        self.inc_request();
        self.metrics_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_bridge_chain_request(&self) {
        self.inc_request();
        self.bridge_chain_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_bridge_intent_request(&self) {
        self.inc_request();
        self.bridge_intent_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_data_ingest_request(&self) {
        self.inc_request();
        self.data_ingest_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_data_status_request(&self) {
        self.inc_request();
        self.data_status_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_knowledge_ingest_request(&self) {
        self.inc_request();
        self.knowledge_ingest_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_knowledge_queue_request(&self) {
        self.inc_request();
        self.knowledge_queue_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_knowledge_vote_request(&self) {
        self.inc_request();
        self.knowledge_vote_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_label_queue_request(&self) {
        self.inc_request();
        self.label_queue_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_visual_label_queue_request(&self) {
        self.inc_request();
        self.visual_label_queue_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_subnet_request(&self) {
        self.inc_request();
        self.subnet_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_identity_challenge_request(&self) {
        self.inc_request();
        self.identity_challenge_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_identity_verify_request(&self) {
        self.inc_request();
        self.identity_verify_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_bridge_success(&self) {
        self.bridge_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_health_success(&self) {
        self.health_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_balance_success(&self) {
        self.balance_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_bridge_chain_success(&self) {
        self.bridge_chain_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_bridge_intent_success(&self) {
        self.bridge_intent_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_data_ingest_success(&self) {
        self.data_ingest_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_data_status_success(&self) {
        self.data_status_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_knowledge_ingest_success(&self) {
        self.knowledge_ingest_success
            .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_knowledge_queue_success(&self) {
        self.knowledge_queue_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_knowledge_vote_success(&self) {
        self.knowledge_vote_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_label_queue_success(&self) {
        self.label_queue_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_visual_label_queue_success(&self) {
        self.visual_label_queue_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_subnet_success(&self) {
        self.subnet_success.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_identity_challenge_success(&self) {
        self.identity_challenge_success
            .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_identity_verify_success(&self) {
        self.identity_verify_success
            .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_rate_limit_hit(&self) {
        self.rate_limit_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_auth_failure(&self) {
        self.auth_failures.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> ApiMetricsSnapshot {
        ApiMetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            health_requests: self.health_requests.load(Ordering::Relaxed),
            bridge_requests: self.bridge_requests.load(Ordering::Relaxed),
            balance_requests: self.balance_requests.load(Ordering::Relaxed),
            metrics_requests: self.metrics_requests.load(Ordering::Relaxed),
            health_success: self.health_success.load(Ordering::Relaxed),
            bridge_chain_requests: self.bridge_chain_requests.load(Ordering::Relaxed),
            bridge_intent_requests: self.bridge_intent_requests.load(Ordering::Relaxed),
            data_ingest_requests: self.data_ingest_requests.load(Ordering::Relaxed),
            data_status_requests: self.data_status_requests.load(Ordering::Relaxed),
            knowledge_ingest_requests: self.knowledge_ingest_requests.load(Ordering::Relaxed),
            knowledge_queue_requests: self.knowledge_queue_requests.load(Ordering::Relaxed),
            knowledge_vote_requests: self.knowledge_vote_requests.load(Ordering::Relaxed),
            label_queue_requests: self.label_queue_requests.load(Ordering::Relaxed),
            visual_label_queue_requests: self.visual_label_queue_requests.load(Ordering::Relaxed),
            subnet_requests: self.subnet_requests.load(Ordering::Relaxed),
            identity_challenge_requests: self
                .identity_challenge_requests
                .load(Ordering::Relaxed),
            identity_verify_requests: self.identity_verify_requests.load(Ordering::Relaxed),
            bridge_success: self.bridge_success.load(Ordering::Relaxed),
            balance_success: self.balance_success.load(Ordering::Relaxed),
            bridge_chain_success: self.bridge_chain_success.load(Ordering::Relaxed),
            bridge_intent_success: self.bridge_intent_success.load(Ordering::Relaxed),
            data_ingest_success: self.data_ingest_success.load(Ordering::Relaxed),
            data_status_success: self.data_status_success.load(Ordering::Relaxed),
            knowledge_ingest_success: self.knowledge_ingest_success.load(Ordering::Relaxed),
            knowledge_queue_success: self.knowledge_queue_success.load(Ordering::Relaxed),
            knowledge_vote_success: self.knowledge_vote_success.load(Ordering::Relaxed),
            label_queue_success: self.label_queue_success.load(Ordering::Relaxed),
            visual_label_queue_success: self.visual_label_queue_success.load(Ordering::Relaxed),
            subnet_success: self.subnet_success.load(Ordering::Relaxed),
            identity_challenge_success: self
                .identity_challenge_success
                .load(Ordering::Relaxed),
            identity_verify_success: self.identity_verify_success.load(Ordering::Relaxed),
            rate_limit_hits: self.rate_limit_hits.load(Ordering::Relaxed),
            auth_failures: self.auth_failures.load(Ordering::Relaxed),
        }
    }
}

pub fn run_api(mut config: NodeConfig, addr: SocketAddr) -> Result<()> {
    let ledger = build_ledger(&config)?;
    let mut data_mesh = None;
    if config.data.enabled {
        let wallet = Arc::new(WalletStore::load_or_create_signer(&config.wallet)?);
        if config.node_id.trim().is_empty() || config.node_id == "node-001" {
            config.node_id = node_id_from_wallet(&wallet.wallet().address);
        }
        let mut network = NodeNetwork::new(config.network.clone());
        network.start(&config.node_id)?;
        network.connect_bootstrap()?;
        let publisher = network
            .publisher()
            .ok_or_else(|| anyhow::anyhow!("p2p network not started"))?;
        let network_rx = network
            .take_message_receiver()
            .ok_or_else(|| anyhow::anyhow!("p2p message receiver unavailable"))?;
        let handle = start_data_mesh(
            config.data.clone(),
            config.node_id.clone(),
            wallet,
            ledger.clone(),
            publisher,
            network_rx,
        )?;
        data_mesh = Some(handle);
    }
    let mut api_key_hashes = Vec::new();
    if config.api.require_api_key {
        for hash in &config.api.api_key_hashes {
            api_key_hashes.push(decode_hash_hex(hash)?);
        }
        if !config.api.api_key_env.trim().is_empty() {
            if let Ok(raw) = std::env::var(&config.api.api_key_env) {
                let trimmed = raw.trim();
                if !trimmed.is_empty() {
                    let candidate = hash_api_key(trimmed);
                    if !api_key_hashes.contains(&candidate) {
                        api_key_hashes.push(candidate);
                    }
                }
            }
        }
        if api_key_hashes.is_empty() {
            anyhow::bail!("api key allowlist is empty");
        }
    }
    let api_key_header: HeaderName = config
        .api
        .api_key_header
        .parse()
        .context("invalid api_key_header")?;
    let limiter = Arc::new(Mutex::new(ApiLimiter::new(
        config.api.rate_limit_window_secs,
    )));
    let (knowledge_runtime, knowledge_persist) = build_knowledge_runtime(&config.knowledge);
    let qa_runtime = QaRuntime::new(QaRuntimeConfig::default());
    let identity_runtime = IdentityRuntime::new(config.identity.clone());
    let share_kind = config.streaming.share_payload_kind.clone();
    let neuro_pool_path = node_data_dir().join("neuro_pool.json");
    // Ensure data directory exists so save_pool can write to it.
    let _ = std::fs::create_dir_all(&node_data_dir());
    let neuro_handle: NeuroRuntimeHandle = std::sync::Arc::new(NeuroRuntime::new(
        &EnvironmentSnapshot {
            timestamp: w1z4rdv1510n::schema::Timestamp { unix: 0 },
            bounds: std::collections::HashMap::new(),
            symbols: Vec::new(),
            metadata: std::collections::HashMap::new(),
            stack_history: Vec::new(),
        },
        NeuroRuntimeConfig {
            enabled: true,
            pool_state_path: Some(neuro_pool_path.to_string_lossy().into_owned()),
            ..Default::default()
        },
    ));
    let ledger_backend = if config.ledger.enabled {
        config.ledger.backend.clone()
    } else {
        "disabled".to_string()
    };
    let state = ApiState {
        node_id: config.node_id.clone(),
        ledger_backend,
        data_enabled: config.data.enabled,
        sensor_ingest_enabled: config.workload.enable_sensor_ingest,
        bridge_enabled: config.blockchain.bridge.enabled,
        knowledge_enabled: config.knowledge.enabled,
        identity_enabled: config.identity.enabled,
        started_at: Instant::now(),
        ledger,
        bridge_config: config.blockchain.bridge.clone(),
        require_api_key: config.api.require_api_key,
        api_key_hashes,
        api_key_header,
        limiter,
        rate_limit_default_max: config.api.rate_limit_max_requests,
        rate_limit_bridge_max: config.api.rate_limit_bridge_max_requests,
        rate_limit_balance_max: config.api.rate_limit_balance_max_requests,
        data_mesh,
        metrics: Arc::new(ApiMetrics::default()),
        knowledge: Arc::new(Mutex::new(knowledge_runtime)),
        knowledge_persist,
        qa_runtime: Arc::new(Mutex::new(qa_runtime)),
        identity: Arc::new(Mutex::new(identity_runtime)),
        label_state: Arc::new(Mutex::new(LabelQueueState::default())),
        fabric_share_kind: share_kind.clone(),
        run_config_path: PathBuf::from(&config.streaming.run_config_path),
        neuro: neuro_handle,
        threat: Arc::new(Mutex::new(ThreatScene::new())),
        equation_matrix: {
            let eem_path = node_data_dir().join("equation_matrix.json");
            Arc::new(EquationMatrixRuntime::new(EquationMatrixConfig {
                persist_path: Some(eem_path.to_string_lossy().into_owned()),
                seed_on_init: true,
                ..Default::default()
            }))
        },
        causal: Arc::new(Mutex::new(CausalRuntime::new(
            w1z4rdv1510n::config::CausalDiscoveryConfig {
                enabled: true,
                ..Default::default()
            },
        ))),
        origin_index: Arc::new(Mutex::new(HashMap::new())),
        cluster: Arc::new(tokio::sync::Mutex::new(None)),
        cluster_advertise_addr: config.network.advertise_addr.as_deref()
            .and_then(|s| s.parse().ok()),
    };
    let data_limit = if config.data.enabled {
        config.data.max_payload_bytes.saturating_mul(2)
    } else {
        0
    };
    // Always allow at least 20 MiB so /media/train can accept high-DPI page images.
    // Bridge proof bytes and data payload limits can bump this higher if needed.
    let max_body = config
        .blockchain
        .bridge
        .max_proof_bytes
        .max(data_limit)
        .max(20 * 1024 * 1024);
    let identity_enabled = state.identity_enabled;
    let identity_runtime = Arc::clone(&state.identity);
    let identity_mesh = state.data_mesh.clone();
    let identity_share_kind = state.fabric_share_kind.clone();
    let label_state = Arc::clone(&state.label_state);
    let label_mesh = state.data_mesh.clone();
    let label_share_kind = state.fabric_share_kind.clone();
    let label_neuro = Arc::clone(&state.neuro);
    let app = Router::new()
        .route("/health", get(get_health))
        .route("/ready", get(get_ready))
        // ── Neuro sensor stream endpoints ──────────────────────────────────
        // POST /neuro/train  — feed an EnvironmentSnapshot; the neural fabric
        //                      observes symbol positions and updates Hebbian
        //                      weights, connecting sensor patterns across time.
        // GET  /neuro/snapshot — current activation, predictions, and motifs.
        .route("/neuro/train", post(neuro_train))
        .route("/neuro/snapshot", get(neuro_snapshot))
        // POST /neuro/record_episode — record a directly-resolved prediction
        //   episode (actual outcome already known; no future-frame resolution needed).
        //   Virtual sensors like the chess training loop call this after each game.
        .route("/neuro/record_episode", post(neuro_record_episode))
        // POST /neuro/register_prediction — register an explicit pending prediction
        //   for auto-resolution when a future observation frame matches.
        .route("/neuro/register_prediction", post(neuro_register_prediction))
        // ── Environmental Equation Matrix endpoints ────────────────────────
        // The EEM maps sensor observations → physics equations, grows as data
        // arrives, and shares discoveries with peer nodes.
        .route("/equations/search", get(equations_search))
        .route("/equations/ingest", post(equations_ingest))
        .route("/equations/apply", post(equations_apply))
        .route("/equations/report", get(equations_report))
        // Open hypothesis gaps — patterns the node can't yet explain.
        // GET returns all gaps sorted by cross-node corroboration count.
        // POST /peer_sync accepts an EemPeerPayload from a peer node.
        .route("/equations/gaps", get(equations_gaps))
        .route("/equations/peer_sync", post(equations_peer_sync))
        // Named-process causal graph built from EEM identifications.
        // Source tracing: walk edges backward to find origin of a pattern.
        .route("/causal/graph", get(causal_graph))
        // First-reporter index: which node first saw each pattern thread.
        .route("/network/patterns/sources", get(network_pattern_sources))
        .route("/bridge/proof", post(submit_bridge_proof))
        .route("/bridge/chains", get(get_bridge_chains))
        .route("/bridge/intent", post(submit_bridge_intent))
        .route("/data/ingest", post(submit_data_ingest))
        .route("/data/:data_id", get(get_data_status))
        .route("/knowledge/ingest", post(submit_knowledge_ingest))
        .route("/knowledge/queue", get(get_knowledge_queue))
        .route("/knowledge/vote", post(submit_knowledge_vote))
        .route("/qa/ingest", post(submit_qa_ingest))
        .route("/qa/query", post(submit_qa_query))
        .route("/streaming/metacognition", post(tune_metacognition))
        .route("/streaming/labels", get(get_label_queue))
        .route("/streaming/visual-labels", get(get_visual_label_queue))
        .route("/streaming/subnets", get(get_subnet_report))
        .route("/network/patterns/query", post(query_pattern_index))
        .route("/identity/challenge", post(submit_identity_challenge))
        .route("/identity/verify", post(submit_identity_verify))
        .route("/identity/:thread_id", get(get_identity_status))
        .route("/balance/:node_id", get(get_balance))
        .route("/metrics", get(get_metrics))
        .route("/threat/ingest", post(threat_ingest))
        .route("/threat/overlay", get(threat_overlay))
        .route("/threat/predict", post(threat_predict))
        // ── Cluster management (GUI + CLI-free join) ───────────────────────
        // GET  /cluster/status  — current cluster membership + ring size
        // POST /cluster/init    — start a new cluster on this machine
        // POST /cluster/join    — join an existing cluster with coordinator + OTP
        // POST /cluster/otp     — generate a new OTP (coordinator only)
        // POST /cluster/leave   — leave cluster gracefully (worker: returns ring to coordinator)
        // POST /cluster/resign  — coordinator steps down, triggers election, then leaves
        // POST /node/shutdown   — shut down the node process cleanly
        .route("/cluster/status", get(cluster_status))
        .route("/cluster/init",   post(cluster_init))
        .route("/cluster/join",   post(cluster_join))
        .route("/cluster/otp",    post(cluster_otp))
        .route("/cluster/leave",  post(cluster_leave))
        .route("/cluster/resign", post(cluster_resign))
        .route("/node/shutdown",  post(node_shutdown))
        // ── Multimodal media ingestion ─────────────────────────────────────
        // POST /media/train  — encode and train one modality or a combined page
        //   Body: { "modality": "image"|"audio"|"text"|"page",
        //           "data_b64": "<base64>",          // for image/audio
        //           "text": "...",                    // for text/page
        //           "spans": [...],                   // for page (with layout)
        //           "lr_scale": 1.0 }
        // POST /neuro/propagate — read what fires given seed labels, no training
        //   Body: { "seed_labels": [...], "hops": 3 }
        //   Response: { "activated": { "label": strength, ... } }
        .route("/media/train",          post(media_train))
        .route("/media/train_sequence", post(media_train_sequence))
        .route("/media/playback",       post(media_playback))
        .route("/neuro/propagate",      post(neuro_propagate))
        .route("/neuro/checkpoint",     post(neuro_checkpoint))
        .route("/neuro/motifs",         get(neuro_motifs))
        .route("/neuro/motifs/predict", post(neuro_motifs_predict))
        .route("/neuro/ask",            post(neuro_ask))
        .with_state(state)
        .layer(DefaultBodyLimit::max(max_body));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build api runtime")?;
    runtime.block_on(async move {
        if identity_enabled {
            if let Some(mesh) = identity_mesh {
                let identity = identity_runtime;
                let share_kind = identity_share_kind;
                tokio::spawn(async move {
                    identity_pattern_sync(mesh, identity, share_kind).await;
                });
            }
        }
        if let Some(mesh) = label_mesh {
            let label_state = label_state;
            let share_kind = label_share_kind;
            let neuro = label_neuro;
            tokio::spawn(async move {
                label_queue_sync(mesh, label_state, share_kind, neuro).await;
            });
        }
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .context("bind api listener")?;
        axum::serve(listener, app.into_make_service())
            .await
            .context("serve api")?;
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

async fn submit_bridge_proof(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(proof): Json<BridgeProof>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_bridge_request();
    if let Err(err) = rate_limit(&state, &headers, "bridge") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let deposit_id = bridge_deposit_id(&proof.deposit);
    let node_id = proof.deposit.recipient_node_id.clone();
    match state.ledger.submit_bridge_proof(proof) {
        Ok(()) => {
            state.metrics.inc_bridge_success();
            let balance = state.ledger.reward_balance(&node_id).ok();
            let response = BridgeSubmitResponse {
                status: "OK",
                deposit_id,
                balance,
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_default()))
        }
        Err(err) => {
            let response = ErrorResponse {
                error: err.to_string(),
            };
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
    }
}

async fn get_balance(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(node_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_balance_request();
    if let Err(err) = rate_limit(&state, &headers, "balance") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    match state.ledger.reward_balance(&node_id) {
        Ok(balance) => (
            {
                state.metrics.inc_balance_success();
                StatusCode::OK
            },
            Json(serde_json::to_value(balance).unwrap_or_default()),
        ),
        Err(err) => {
            let response = ErrorResponse {
                error: err.to_string(),
            };
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
    }
}

async fn get_bridge_chains(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_bridge_chain_request();
    if let Err(err) = rate_limit(&state, &headers, "bridge") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.bridge_config.enabled {
        let response = ErrorResponse {
            error: "bridge disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let chains: Vec<BridgeChainInfo> = state
        .bridge_config
        .chains
        .iter()
        .map(|chain| BridgeChainInfo {
            chain_id: chain.chain_id.clone(),
            chain_kind: chain.chain_kind.clone(),
            verification: chain.verification.clone(),
            min_confirmations: chain.min_confirmations,
            relayer_quorum: chain.relayer_quorum,
            allowed_assets: chain.allowed_assets.clone(),
            max_deposit_amount: chain.max_deposit_amount,
            deposit_address: sanitize_option(&chain.deposit_address),
            recipient_tag_template: sanitize_option(&chain.recipient_tag_template),
        })
        .collect();
    state.metrics.inc_bridge_chain_success();
    (
        StatusCode::OK,
        Json(serde_json::to_value(chains).unwrap_or_default()),
    )
}

async fn submit_bridge_intent(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(intent): Json<BridgeIntentRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_bridge_intent_request();
    if let Err(err) = rate_limit(&state, &headers, "bridge") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.bridge_config.enabled {
        let response = ErrorResponse {
            error: "bridge disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let chain_id = intent.chain_id.trim();
    if chain_id.is_empty() {
        let response = ErrorResponse {
            error: "chain_id must be provided".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let asset = intent.asset.trim();
    if asset.is_empty() {
        let response = ErrorResponse {
            error: "asset must be provided".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !intent.amount.is_finite() || intent.amount <= 0.0 {
        let response = ErrorResponse {
            error: "amount must be > 0 and finite".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let recipient_node_id = intent.recipient_node_id.trim();
    if recipient_node_id.is_empty() {
        let response = ErrorResponse {
            error: "recipient_node_id must be provided".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let Some(policy) = find_chain_policy(&state.bridge_config, chain_id) else {
        let response = ErrorResponse {
            error: "bridge chain not supported".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    };
    let allowed_asset = policy
        .allowed_assets
        .iter()
        .find(|allowed| allowed.eq_ignore_ascii_case(asset))
        .cloned();
    let Some(asset) = allowed_asset else {
        let response = ErrorResponse {
            error: "asset not allowed for chain".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    };
    if intent.amount > policy.max_deposit_amount {
        let response = ErrorResponse {
            error: "amount exceeds max_deposit_amount".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let deposit_address = match sanitize_option(&policy.deposit_address) {
        Some(address) => address,
        None => {
            let response = ErrorResponse {
                error: "bridge deposit_address not configured".to_string(),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
    };
    let recipient_tag = sanitize_option(&policy.recipient_tag_template)
        .map(|template| render_recipient_tag(&template, recipient_node_id));
    if let Some(tag) = &recipient_tag {
        if tag.trim().is_empty() {
            let response = ErrorResponse {
                error: "bridge recipient_tag_template produced empty tag".to_string(),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
    }
    let idempotency_key = sanitize_option(&intent.idempotency_key)
        .unwrap_or_else(|| default_idempotency_key(chain_id, &asset, intent.amount, recipient_node_id));
    let created_at = now_timestamp();
    let mut bridge_intent = BridgeIntent {
        intent_id: String::new(),
        chain_id: policy.chain_id.clone(),
        chain_kind: policy.chain_kind.clone(),
        asset: asset.clone(),
        amount: intent.amount,
        recipient_node_id: recipient_node_id.to_string(),
        deposit_address,
        recipient_tag,
        idempotency_key,
        created_at,
    };
    let intent_id = bridge_intent_id(&bridge_intent);
    bridge_intent.intent_id = intent_id;
    match state.ledger.submit_bridge_intent(bridge_intent) {
        Ok(stored) => {
            let response = BridgeIntentResponse {
                status: "OK",
                intent_id: stored.intent_id,
                created_at: stored.created_at,
                chain_id: stored.chain_id,
                chain_kind: stored.chain_kind,
                asset: stored.asset,
                amount: stored.amount,
                recipient_node_id: stored.recipient_node_id,
                deposit_address: stored.deposit_address,
                recipient_tag: stored.recipient_tag,
                min_confirmations: policy.min_confirmations,
                verification: policy.verification.clone(),
                relayer_quorum: policy.relayer_quorum,
                max_deposit_amount: policy.max_deposit_amount,
            };
            state.metrics.inc_bridge_intent_success();
            (
                StatusCode::OK,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
        Err(err) => {
            let response = ErrorResponse {
                error: err.to_string(),
            };
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
    }
}

async fn submit_data_ingest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<DataIngestRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_data_ingest_request();
    if let Err(err) = rate_limit(&state, &headers, "data") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.sensor_ingest_enabled {
        let response = ErrorResponse {
            error: "sensor ingest disabled by workload profile".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let Some(mesh) = state.data_mesh.as_ref() else {
        let response = ErrorResponse {
            error: "data mesh disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    };
    match mesh.ingest(request).await {
        Ok(response) => {
            state.metrics.inc_data_ingest_success();
            (
                StatusCode::OK,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
        Err(err) => {
            let response = ErrorResponse {
                error: err.to_string(),
            };
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
    }
}

async fn submit_knowledge_ingest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<KnowledgeIngestRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_knowledge_ingest_request();
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.knowledge_enabled {
        let response = ErrorResponse {
            error: "knowledge ingest disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let now = now_timestamp();
    let document = match (request.document, request.xml) {
        (Some(_), Some(_)) => {
            let response = ErrorResponse {
                error: "provide either document or xml, not both".to_string(),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
        (Some(document), None) => document,
        (None, Some(xml)) => {
            if xml.trim().is_empty() {
                let response = ErrorResponse {
                    error: "xml payload must be non-empty".to_string(),
                };
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::to_value(response).unwrap_or_default()),
                );
            }
            let config = build_ingest_config(request.options);
            let ingestor = NlmJatsIngestor::new(config);
            match ingestor.parse_str(&xml, now) {
                Ok(doc) => doc,
                Err(err) => {
                    let response = ErrorResponse {
                        error: format!("failed to parse JATS xml: {err}"),
                    };
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::to_value(response).unwrap_or_default()),
                    );
                }
            }
        }
        (None, None) => {
            let response = ErrorResponse {
                error: "document or xml must be provided".to_string(),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
    };
    let (report, pending) = {
        let mut knowledge = state.knowledge.lock().expect("knowledge mutex");
        let report = knowledge.ingest_document(document, now);
        let pending = if request.include_tasks {
            knowledge
                .queue_report(now)
                .map(|report| report.pending)
        } else {
            None
        };
        (report, pending)
    };
    let response = KnowledgeIngestResponse {
        status: "OK",
        report,
        pending,
    };
    if let Err(err) = persist_knowledge_runtime(&state.knowledge_persist, &state.knowledge) {
        let response = ErrorResponse {
            error: format!("failed to persist knowledge: {err}"),
        };
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    state.metrics.inc_knowledge_ingest_success();
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

async fn get_knowledge_queue(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<KnowledgeQueueQuery>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_knowledge_queue_request();
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.knowledge_enabled {
        let response = ErrorResponse {
            error: "knowledge queue disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let now = now_timestamp();
    let mut report = {
        let knowledge = state.knowledge.lock().expect("knowledge mutex");
        knowledge.queue_report(now)
    };
    if let (Some(report), Some(limit)) = (report.as_mut(), query.limit) {
        report.pending.truncate(limit.max(1));
    }
    let response = KnowledgeQueueResponse {
        status: "OK",
        report,
    };
    state.metrics.inc_knowledge_queue_success();
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

async fn submit_knowledge_vote(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(vote): Json<AssociationVote>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_knowledge_vote_request();
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.knowledge_enabled {
        let response = ErrorResponse {
            error: "knowledge vote disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if vote.worker_id.trim().is_empty() {
        let response = ErrorResponse {
            error: "worker_id must be provided".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let association = {
        let mut knowledge = state.knowledge.lock().expect("knowledge mutex");
        knowledge.submit_vote(vote)
    };
    if let Some(ref assoc) = association {
        state.metrics.inc_knowledge_vote_success();
        // Back-propagate verified association into the Hebbian fabric so the
        // neuro pool learns which knowledge concepts co-occur.
        let labels = vec![
            format!("knowledge::doc::{}", assoc.doc_id),
            format!("knowledge::figure::{}", assoc.figure_id),
            format!("knowledge::text::{}", assoc.text_block_id),
        ];
        state.neuro.train_weighted(&labels, 4.0, false);
    }
    let response = KnowledgeVoteResponse {
        status: if association.is_some() { "VERIFIED" } else { "PENDING" },
        association,
    };
    if let Err(err) = persist_knowledge_runtime(&state.knowledge_persist, &state.knowledge) {
        let response = ErrorResponse {
            error: format!("failed to persist knowledge: {err}"),
        };
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

// ─── Neuro sensor stream endpoints ────────────────────────────────────────────

/// POST /neuro/train
/// Body: `{ "snapshot": <EnvironmentSnapshot> }`
/// Observes the snapshot through the neural fabric — updates Hebbian weights
/// for every symbol co-occurrence in this sensor frame, connects spatial and
/// temporal context, and accumulates motif patterns over time.
#[derive(Deserialize)]
struct NeuroTrainRequest {
    snapshot: EnvironmentSnapshot,
}

#[derive(Serialize)]
struct NeuroTrainResponse {
    status: &'static str,
    symbols_observed: usize,
}

async fn neuro_train(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<NeuroTrainRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "neuro") {
        state.metrics.inc_rate_limit_hit();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(ErrorResponse { error: err.to_string() }).unwrap_or_default()),
        );
    }
    let n_symbols = request.snapshot.symbols.len();
    // observe_snapshot handles all label extraction, Hebbian updates, and
    // temporal pattern accumulation internally.
    state.neuro.observe_snapshot(&request.snapshot);

    // Also feed the sensor labels directly into the EEM so equations gain
    // evidence from every training frame — not just explicit /equations/apply calls.
    // Extract labels from symbol properties (type + property values).
    let sensor_labels: Vec<String> = {
        let mut labels: std::collections::HashSet<String> = std::collections::HashSet::new();
        for sym in &request.snapshot.symbols {
            labels.insert(format!("{:?}", sym.symbol_type).to_lowercase());
            for (k, v) in &sym.properties {
                if let Some(s) = v.as_str() {
                    labels.insert(format!("{}::{}", k.to_lowercase(), s.to_lowercase()));
                    labels.insert(s.to_lowercase());
                }
            }
        }
        for (k, v) in &request.snapshot.metadata {
            if let Some(s) = v.as_str() {
                labels.insert(format!("meta::{}::{}", k.to_lowercase(), s.to_lowercase()));
            }
        }
        labels.into_iter().collect()
    };
    if !sensor_labels.is_empty() {
        let dims = if request.snapshot.bounds.get("z").copied().unwrap_or(0.0) > 0.0 { 3u8 } else { 2u8 };
        let ctx = state.equation_matrix.apply_to_context(&sensor_labels, dims);
        // Reinforce every matched equation with evidence from this sensor frame.
        for candidate in &ctx.candidates {
            state.equation_matrix.reinforce(&candidate.equation.id);
        }
        // Open hypothesis gaps for label clusters with no equation match.
        if !ctx.unexplained_labels.is_empty() {
            state.equation_matrix.open_gap(
                ctx.unexplained_labels,
                dims,
                "auto: sensor labels with no equation match",
                Some(&state.node_id),
            );
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::to_value(NeuroTrainResponse {
            status: "ok",
            symbols_observed: n_symbols,
        }).unwrap_or_default()),
    )
}

/// GET /neuro/snapshot
/// Returns the current activation state of the neural fabric: active labels,
/// composite neurons, temporal predictions, top influence records, and working
/// memory — exactly what the streaming side produces after each sensor frame.
async fn neuro_snapshot(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "neuro") {
        state.metrics.inc_rate_limit_hit();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(ErrorResponse { error: err.to_string() }).unwrap_or_default()),
        );
    }
    let snap = state.neuro.snapshot();
    (
        StatusCode::OK,
        Json(serde_json::to_value(snap).unwrap_or_default()),
    )
}

/// POST /neuro/record_episode
/// Body: `{ "context_labels": [...], "predicted": "...", "actual": "...",
///          "streams": [...], "surprise": 0.0..1.0 }`
///
/// Directly records a resolved prediction episode into the episodic store.
/// Use this when the caller already knows the outcome (e.g. end of a chess game,
/// QA question with known answer).  The fabric's conditional sufficiency tracker
/// and inhibitory Hebbian updates are applied automatically.
///
/// `surprise` is a focal-loss-inspired score:  0 = expected result (low gradient),
/// 1 = completely wrong with high confidence (large gradient).  Pass
/// `(1 - p_correct)^2` or `p_confidence^2` depending on outcome.
#[derive(Deserialize)]
struct NeuroRecordEpisodeRequest {
    context_labels: Vec<String>,
    predicted: String,
    actual: String,
    #[serde(default)]
    streams: Vec<String>,
    #[serde(default)]
    surprise: f32,
}

async fn neuro_record_episode(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<NeuroRecordEpisodeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "neuro") {
        state.metrics.inc_rate_limit_hit();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": err.to_string() })),
        );
    }
    state.neuro.record_episode(
        req.context_labels,
        req.predicted,
        req.actual,
        req.streams,
        req.surprise.clamp(0.0, 1.0),
    );
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

/// POST /neuro/register_prediction
/// Body: `{ "context_labels": [...], "predicted": "...", "streams": [...],
///          "p_confidence": 0.0..1.0, "resolve_on": "<label or null>" }`
///
/// Registers an explicit pending prediction for auto-resolution.  The fabric
/// will watch future observation frames and resolve the prediction automatically
/// when `resolve_on` (or `predicted` if omitted) appears — without any further
/// action from the caller.
///
/// This is the entry point for virtual sensors.  Hardware sensors get auto-
/// predictions for free; this endpoint gives virtual sensors the same capability.
#[derive(Deserialize)]
struct NeuroRegisterPredictionRequest {
    context_labels: Vec<String>,
    predicted: String,
    #[serde(default)]
    streams: Vec<String>,
    #[serde(default = "default_confidence")]
    p_confidence: f32,
    #[serde(default)]
    resolve_on: Option<String>,
}

fn default_confidence() -> f32 { 0.5 }

async fn neuro_register_prediction(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<NeuroRegisterPredictionRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "neuro") {
        state.metrics.inc_rate_limit_hit();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": err.to_string() })),
        );
    }
    state.neuro.register_prediction(
        req.context_labels,
        req.predicted,
        req.streams,
        req.p_confidence.clamp(0.0, 1.0),
        req.resolve_on,
    );
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

// ─── Environmental Equation Matrix endpoints ───────────────────────────────────

/// GET /equations/search?q=<text>&limit=<n>
async fn equations_search(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let query = params.get("q").map(|s| s.trim().to_string()).unwrap_or_default();
    if query.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "q is required" })));
    }
    let limit: Option<usize> = params.get("limit").and_then(|s| s.parse().ok());
    let results = state.equation_matrix.search(&query, limit);
    (StatusCode::OK, Json(serde_json::json!({
        "query": query,
        "count": results.len(),
        "results": results.iter().map(|r| serde_json::json!({
            "id": r.equation.id,
            "text": r.equation.text,
            "latex": r.equation.latex,
            "discipline": r.equation.discipline.as_str(),
            "applicable_dims": r.equation.applicable_dims,
            "confidence": r.equation.confidence,
            "evidence_count": r.equation.evidence_count,
            "relevance": r.relevance,
            "related_ids": r.related_ids,
        })).collect::<Vec<_>>(),
    })))
}

/// POST /equations/ingest
/// Body: `{ "text": "...", "discipline": "classical_mechanics", "source": "..." }`
/// Parses equations from free text and adds them to the matrix.
#[derive(Deserialize)]
struct EquationsIngestRequest {
    text: String,
    #[serde(default)]
    discipline: Option<String>,
    #[serde(default)]
    source_node_id: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

async fn equations_ingest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<EquationsIngestRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let discipline = match request.discipline.as_deref() {
        Some("lagrangian_mechanics") => Discipline::LagrangianMechanics,
        Some("hamiltonian_mechanics") => Discipline::HamiltonianMechanics,
        Some("thermodynamics") => Discipline::Thermodynamics,
        Some("statistical_mechanics") => Discipline::StatisticalMechanics,
        Some("electromagnetism") => Discipline::Electromagnetism,
        Some("quantum_mechanics") => Discipline::QuantumMechanics,
        Some("quantum_field_theory") => Discipline::QuantumFieldTheory,
        Some("special_relativity") => Discipline::SpecialRelativity,
        Some("general_relativity") => Discipline::GeneralRelativity,
        Some("fluid_dynamics") => Discipline::FluidDynamics,
        Some("chaos_dynamics") => Discipline::ChaosDynamics,
        Some("topological_physics") => Discipline::TopologicalPhysics,
        Some("condensed_matter") => Discipline::CondensedMatter,
        Some("cosmology") => Discipline::Cosmology,
        Some("information_theory") => Discipline::InformationTheory,
        Some(other) => Discipline::Custom(other.to_string()),
        None => Discipline::Custom("unknown".into()),
    };
    let confidence = request.confidence.unwrap_or(0.6).clamp(0.0, 1.0);
    let ids = state.equation_matrix.ingest_text(
        &request.text,
        discipline,
        request.source_node_id,
        confidence,
    );
    (StatusCode::OK, Json(serde_json::json!({
        "status": "ok",
        "ingested": ids.len(),
        "ids": ids,
    })))
}

/// POST /equations/apply
/// Body: `{ "labels": ["force", "mass", ...], "dims": 3 }`
/// Returns equations that explain the current sensor context.
#[derive(Deserialize)]
struct EquationsApplyRequest {
    labels: Vec<String>,
    #[serde(default = "default_dims")]
    dims: u8,
}
fn default_dims() -> u8 { 3 }

async fn equations_apply(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<EquationsApplyRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    // Also activate the neuro fabric with these labels so both systems
    // learn from the same sensor context simultaneously.
    if !request.labels.is_empty() {
        state.neuro.train_weighted(&request.labels, 0.5, false);
    }
    let result = state.equation_matrix.apply_to_context(&request.labels, request.dims);
    // Record unexplained labels as hypothesis gaps
    if !result.unexplained_labels.is_empty() {
        state.equation_matrix.open_gap(
            result.unexplained_labels.clone(),
            request.dims,
            "sensor labels with no matching equation",
            Some(&state.node_id),
        );
    }

    // Feed identified equations into the named-process causal graph.
    // Each identification creates edges: "sensor::{label_hash}" → "process::{eq_id}"
    // so the causal graph accumulates a record of what physical processes were
    // active and which sensor contexts produced them. Walking these edges
    // backward from a process node traces the signal to its origin.
    if !result.candidates.is_empty() {
        let label_key: String = {
            let mut sorted = request.labels.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let joined = sorted.join(",");
            // Use blake2 to produce a short stable key for this label cluster.
            let hash_bytes: [u8; 32] = {
                use blake2::Digest;
                let mut h = Blake2s256::new();
                h.update(joined.as_bytes());
                h.finalize().into()
            };
            // Encode first 8 bytes as hex manually (no hex crate needed).
            let hex: String = hash_bytes[..8].iter().map(|b| format!("{:02x}", b)).collect();
            format!("sensor::{hex}")
        };
        if let Ok(mut causal) = state.causal.lock() {
            for candidate in &result.candidates {
                causal.observe_edge(CausalEdge {
                    source: label_key.clone(),
                    target: format!("process::{}", candidate.equation.id),
                    lag_secs: 0.0,
                    weight: candidate.relevance as f64,
                });
            }
        }
    }

    (StatusCode::OK, Json(serde_json::json!({
        "dims": request.dims,
        "candidates": result.candidates.iter().map(|r| serde_json::json!({
            "id": r.equation.id,
            "text": r.equation.text,
            "discipline": r.equation.discipline.as_str(),
            "confidence": r.equation.confidence,
            "relevance": r.relevance,
            "applicable_dims": r.equation.applicable_dims,
        })).collect::<Vec<_>>(),
        "open_gaps": result.open_gaps.len(),
        "unexplained_labels": result.unexplained_labels,
    })))
}

/// GET /equations/report
async fn equations_report(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let report = state.equation_matrix.report();
    (StatusCode::OK, Json(serde_json::to_value(report).unwrap_or_default()))
}

/// GET /equations/gaps
/// Returns all open hypothesis slots — sensor label clusters that fired
/// repeatedly but matched no equation. Sorted by cross-node corroboration
/// count (most-observed first). Slots with multiple reporting nodes are the
/// strongest signal that something real and currently unexplained is moving
/// through the network.
async fn equations_gaps(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let gaps = state.equation_matrix.open_gaps();
    (StatusCode::OK, Json(serde_json::json!({
        "count": gaps.len(),
        "gaps": gaps.iter().map(|g| serde_json::json!({
            "id": g.id,
            "description": g.description,
            "trigger_labels": g.trigger_labels,
            "sensor_dims": g.sensor_dims,
            "created_at": g.created_at,
            "observation_count": g.observation_count,
            "first_node_id": g.first_node_id,
            "corroborating_nodes": g.reporting_nodes.len(),
            "reporting_nodes": g.reporting_nodes,
        })).collect::<Vec<_>>(),
    })))
}

/// POST /equations/peer_sync
/// Body: `EemPeerPayload` JSON — receives a peer node's equation discoveries
/// and open gaps. Equations get lower confidence until corroborated locally.
/// Gaps from peers that match local gaps increment their corroboration count.
#[derive(Deserialize)]
struct PeerSyncRequest {
    payload: EemPeerPayload,
}

async fn equations_peer_sync(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<PeerSyncRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let source = request.payload.source_node_id.clone();
    let eq_count = request.payload.equations.len();
    let gap_count = request.payload.open_gaps.len();
    state.equation_matrix.merge_peer_payload(request.payload);
    (StatusCode::OK, Json(serde_json::json!({
        "status": "ok",
        "source_node_id": source,
        "equations_merged": eq_count,
        "gaps_merged": gap_count,
    })))
}

/// GET /causal/graph
/// Returns the named-process causal graph: directed edges from sensor label
/// cluster hashes to identified physics process nodes. Walking backward from
/// any "process::{id}" node to its "sensor::{hash}" sources — and then across
/// nodes via /network/patterns/sources — traces a propagating signal back to
/// its origin. Edge weight reflects how consistently a sensor context produces
/// that physical process identification.
async fn causal_graph(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let edges = if let Ok(causal) = state.causal.lock() {
        causal.graph().edges()
    } else {
        vec![]
    };
    (StatusCode::OK, Json(serde_json::json!({
        "edge_count": edges.len(),
        "edges": edges.iter().map(|e| serde_json::json!({
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
            "lag_secs": e.lag_secs,
        })).collect::<Vec<_>>(),
        "note": "source=sensor::{label_hash}, target=process::{eq_id}. Walk backward to trace signal origin.",
    })))
}

/// GET /network/patterns/sources
/// Returns the first-reporter origin index: for each pattern thread that has
/// been seen across nodes, records which node first reported it and how many
/// nodes have since corroborated it. The node that appears first with the
/// highest subsequent corroboration is the statistical origin of that pattern.
async fn network_pattern_sources(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({ "error": err.to_string() })));
    }
    let mut entries: Vec<serde_json::Value> = if let Ok(idx) = state.origin_index.lock() {
        let mut v: Vec<_> = idx.values().collect::<Vec<_>>().into_iter().cloned().collect();
        // Sort by report_count descending — most-corroborated patterns first.
        v.sort_by(|a, b| b.report_count.cmp(&a.report_count));
        v.into_iter().map(|o| serde_json::json!({
            "thread_id": o.thread_id,
            "first_seen_unix": o.first_seen_unix,
            "first_reporter_node_id": o.first_reporter_node_id,
            "report_count": o.report_count,
            "corroborating_node_count": o.reporting_nodes.len(),
            "reporting_nodes": o.reporting_nodes,
        })).collect()
    } else {
        vec![]
    };
    (StatusCode::OK, Json(serde_json::json!({
        "count": entries.len(),
        "patterns": entries,
    })))
}

// ─── Q&A fabric endpoints ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct QaIngestRequest {
    /// One or more Q&A candidate records (same schema as qa_candidates.jsonl).
    candidates: Vec<QaCandidateRecord>,
}

#[derive(Serialize)]
struct QaIngestResponse {
    status: &'static str,
    ingested: usize,
    total_pairs: u64,
    question_neurons: usize,
    answer_entries: usize,
}

#[derive(Deserialize)]
struct QaQueryRequest {
    question: String,
}

#[derive(Serialize)]
struct QaQueryResponse {
    status: &'static str,
    report: QaQueryReport,
}

async fn submit_qa_ingest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<QaIngestRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(ErrorResponse { error: err.to_string() }).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(ErrorResponse { error: err.to_string() }).unwrap_or_default()),
        );
    }
    if request.candidates.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(ErrorResponse { error: "candidates must not be empty".into() }).unwrap_or_default()),
        );
    }
    let now = now_timestamp();
    let (ingested, total_pairs, question_neurons, answer_entries) = {
        let mut qa = state.qa_runtime.lock().expect("qa mutex");
        let before = qa.pairs_ingested();
        qa.ingest_candidates(&request.candidates, now);
        let after = qa.pairs_ingested();
        (
            (after - before) as usize,
            after,
            qa.question_neuron_count(),
            qa.answer_count(),
        )
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(QaIngestResponse {
            status: "OK",
            ingested,
            total_pairs,
            question_neurons,
            answer_entries,
        }).unwrap_or_default()),
    )
}

async fn submit_qa_query(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<QaQueryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "knowledge") {
        state.metrics.inc_rate_limit_hit();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(ErrorResponse { error: err.to_string() }).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(ErrorResponse { error: err.to_string() }).unwrap_or_default()),
        );
    }
    if request.question.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(ErrorResponse { error: "question must not be empty".into() }).unwrap_or_default()),
        );
    }
    let now = now_timestamp();
    let report = {
        let mut qa = state.qa_runtime.lock().expect("qa mutex");
        qa.query(&request.question, now)
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(QaQueryResponse { status: "OK", report }).unwrap_or_default()),
    )
}

async fn tune_metacognition(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<MetacognitionTuneRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_request();
    if let Err(err) = rate_limit(&state, &headers, "metacognition") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let raw = match fs::read_to_string(&state.run_config_path) {
        Ok(raw) => raw,
        Err(err) => {
            let response = ErrorResponse {
                error: format!("failed to read run config: {err}"),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
    };
    let mut run_config: RunConfig = match serde_json::from_str(&raw) {
        Ok(config) => config,
        Err(err) => {
            let response = ErrorResponse {
                error: format!("invalid run config: {err}"),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
    };
    let meta_snapshot = {
        let meta = &mut run_config.streaming.metacognition;
        if let Some(value) = request.min_depth {
            meta.min_reflection_depth = value;
        }
        if let Some(value) = request.max_depth {
            meta.max_reflection_depth = value;
        }
        if let Some(value) = request.confident_depth {
            meta.confident_depth = value;
        }
        if let Some(value) = request.accuracy_target {
            meta.accuracy_target = value;
        }
        if let Some(value) = request.confident_uncertainty_threshold {
            meta.confident_uncertainty_threshold = value;
        }
        if let Some(value) = request.novelty_depth_boost {
            meta.novelty_depth_boost = value;
        }
        if let Some(value) = request.min_depth_samples {
            meta.min_depth_samples = value;
        }
        if let Some(value) = request.depth_improvement_margin {
            meta.depth_improvement_margin = value;
        }
        serde_json::to_value(meta).unwrap_or_default()
    };
    if let Err(err) = run_config.validate() {
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = fs::write(
        &state.run_config_path,
        serde_json::to_string_pretty(&run_config).unwrap_or_default(),
    ) {
        let response = ErrorResponse {
            error: format!("failed to write run config: {err}"),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let response = MetacognitionTuneResponse {
        status: "OK",
        metacognition: meta_snapshot,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

async fn submit_identity_challenge(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<IdentityChallengeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_identity_challenge_request();
    if let Err(err) = rate_limit(&state, &headers, "identity") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.identity_enabled {
        let response = ErrorResponse {
            error: "identity verification disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let now = now_timestamp();
    let challenge = {
        let mut runtime = state.identity.lock().expect("identity runtime lock");
        match runtime.issue_challenge(request, now) {
            Ok(challenge) => challenge,
            Err(err) => {
                let response = ErrorResponse {
                    error: err.to_string(),
                };
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::to_value(response).unwrap_or_default()),
                );
            }
        }
    };
    state.metrics.inc_identity_challenge_success();
    let response = IdentityChallengeResponse {
        status: "OK".to_string(),
        challenge,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

async fn submit_identity_verify(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<IdentityVerifyRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_identity_verify_request();
    if let Err(err) = rate_limit(&state, &headers, "identity") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.identity_enabled {
        let response = ErrorResponse {
            error: "identity verification disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let now = now_timestamp();
    let outcome = {
        let mut runtime = state.identity.lock().expect("identity runtime lock");
        match runtime.verify(request, now) {
            Ok(outcome) => outcome,
            Err(err) => {
                let response = IdentityVerifyResponse {
                    status: "REJECTED".to_string(),
                    binding: None,
                    evidence_hash: None,
                    reason: Some(err.to_string()),
                };
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::to_value(response).unwrap_or_default()),
                );
            }
        }
    };
    if let Err(err) = state.ledger.submit_identity_binding(outcome.binding.clone()) {
        let response = IdentityVerifyResponse {
            status: "REJECTED".to_string(),
            binding: None,
            evidence_hash: Some(outcome.evidence_hash),
            reason: Some(err.to_string()),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let (Some(mesh), Some(pattern)) = (state.data_mesh.as_ref(), outcome.updated_pattern) {
        let share = NeuralFabricShare {
            node_id: state.node_id.clone(),
            timestamp: now,
            tokens: Vec::new(),
            layers: Vec::new(),
            motifs: Vec::new(),
            motif_transitions: Vec::new(),
            network_patterns: vec![pattern],
            metacognition: None,
            metadata: HashMap::from([
                ("identity_thread_id".to_string(), serde_json::Value::String(outcome.binding.thread_id.clone())),
                ("identity_wallet".to_string(), serde_json::Value::String(outcome.binding.wallet_address.clone())),
                ("identity_challenge_id".to_string(), serde_json::Value::String(outcome.binding.challenge_id.clone())),
            ]),
        };
        let _ = mesh
            .ingest_fabric_share_with_kind(&share, &state.fabric_share_kind)
            .await;
    }
    state.metrics.inc_identity_verify_success();
    let response = IdentityVerifyResponse {
        status: "VERIFIED".to_string(),
        binding: Some(outcome.binding),
        evidence_hash: Some(outcome.evidence_hash),
        reason: None,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

async fn get_identity_status(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(thread_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_identity_verify_request();
    if let Err(err) = rate_limit(&state, &headers, "identity") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if !state.identity_enabled {
        let response = ErrorResponse {
            error: "identity verification disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    match state.ledger.identity_binding(&thread_id) {
        Ok(binding) => {
            state.metrics.inc_identity_verify_success();
            let response = IdentityStatusResponse {
                status: "OK".to_string(),
                binding: Some(binding),
            };
            (
                StatusCode::OK,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
        Err(err) => {
            let response = IdentityStatusResponse {
                status: err.to_string(),
                binding: None,
            };
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
    }
}

async fn get_data_status(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(data_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_data_status_request();
    if let Err(err) = rate_limit(&state, &headers, "data") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let Some(mesh) = state.data_mesh.as_ref() else {
        let response = ErrorResponse {
            error: "data mesh disabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    };
    match mesh.status(data_id).await {
        Ok(response) => {
            state.metrics.inc_data_status_success();
            (
                StatusCode::OK,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
        Err(err) => {
            let response = ErrorResponse {
                error: err.to_string(),
            };
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            )
        }
    }
}

async fn get_health(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_health_request();
    if let Err(err) = rate_limit(&state, &headers, "health") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let snapshot = build_health_snapshot(&state, "OK");
    state.metrics.inc_health_success();
    (
        StatusCode::OK,
        Json(serde_json::to_value(snapshot).unwrap_or_default()),
    )
}

async fn get_ready(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_health_request();
    if let Err(err) = rate_limit(&state, &headers, "health") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let ready = state.data_mesh.is_some() || !state.data_enabled;
    let status = if ready { "READY" } else { "NOT_READY" };
    let snapshot = build_health_snapshot(&state, status);
    if ready {
        state.metrics.inc_health_success();
    }
    (
        if ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE },
        Json(serde_json::to_value(snapshot).unwrap_or_default()),
    )
}

async fn get_metrics(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_metrics_request();
    if let Err(err) = rate_limit(&state, &headers, "metrics") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let snapshot = state.metrics.snapshot();
    (
        StatusCode::OK,
        Json(serde_json::to_value(snapshot).unwrap_or_default()),
    )
}

async fn get_label_queue(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<QueueLimitQuery>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_label_queue_request();
    if let Err(err) = rate_limit(&state, &headers, "labels") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let (queue, updated_at) = match state.label_state.lock() {
        Ok(state) => (state.label_queue.clone(), state.updated_at),
        Err(_) => (None, None),
    };
    let mut queue = queue;
    if let Some(queue) = queue.as_mut() {
        apply_limit(&mut queue.pending, query.limit);
    }
    let status = if queue.is_some() { "OK" } else { "EMPTY" };
    if queue.is_some() {
        state.metrics.inc_label_queue_success();
    }
    let snapshot = LabelQueueSnapshot {
        status,
        updated_at,
        queue,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(snapshot).unwrap_or_default()),
    )
}

async fn get_visual_label_queue(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<QueueLimitQuery>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_visual_label_queue_request();
    if let Err(err) = rate_limit(&state, &headers, "labels") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let (queue, updated_at) = match state.label_state.lock() {
        Ok(state) => (state.visual_label_queue.clone(), state.updated_at),
        Err(_) => (None, None),
    };
    let mut queue = queue;
    if let Some(queue) = queue.as_mut() {
        apply_limit(&mut queue.pending, query.limit);
    }
    let status = if queue.is_some() { "OK" } else { "EMPTY" };
    if queue.is_some() {
        state.metrics.inc_visual_label_queue_success();
    }
    let snapshot = VisualLabelQueueSnapshot {
        status,
        updated_at,
        queue,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(snapshot).unwrap_or_default()),
    )
}

async fn get_subnet_report(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    state.metrics.inc_subnet_request();
    if let Err(err) = rate_limit(&state, &headers, "subnets") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let (report, updated_at) = match state.label_state.lock() {
        Ok(state) => (state.subnet_report.clone(), state.updated_at),
        Err(_) => (None, None),
    };
    let status = if report.is_some() { "OK" } else { "EMPTY" };
    if report.is_some() {
        state.metrics.inc_subnet_success();
    }
    let snapshot = SubnetworkSnapshot {
        status,
        updated_at,
        report,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(snapshot).unwrap_or_default()),
    )
}

async fn query_pattern_index(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<PatternQueryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(err) = rate_limit(&state, &headers, "pattern_query") {
        state.metrics.inc_rate_limit_hit();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
        state.metrics.inc_auth_failure();
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    let Some(mesh) = state.data_mesh.as_ref() else {
        let response = ErrorResponse {
            error: "data mesh not enabled".to_string(),
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    };
    let query = PatternQuery {
        query_id: String::new(),
        requester_node_id: state.node_id.clone(),
        timestamp: now_timestamp(),
        phenotype_hash: request.phenotype_hash.unwrap_or_default(),
        phenotype_tokens: request.phenotype_tokens.unwrap_or_default(),
        behavior_signature: request.behavior_signature,
        species: request.species,
        max_results: request.max_results.unwrap_or(8),
        min_similarity: request.min_similarity,
        public_key: String::new(),
        signature: String::new(),
    };
    let local = match mesh.search_pattern_index(query.clone()).await {
        Ok(response) => response,
        Err(err) => {
            let response = ErrorResponse {
                error: err.to_string(),
            };
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(response).unwrap_or_default()),
            );
        }
    };
    let broadcast = request.broadcast.unwrap_or(true);
    let mut query_id = Some(local.query_id.clone());
    if broadcast {
        match mesh.publish_pattern_query(query) {
            Ok(id) => query_id = Some(id),
            Err(err) => {
                let response = ErrorResponse {
                    error: err.to_string(),
                };
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::to_value(response).unwrap_or_default()),
                );
            }
        }
    }
    let mut responses = Vec::new();
    if broadcast {
        let wait_ms = request.wait_for_responses_ms.unwrap_or(0).min(5_000);
        if wait_ms > 0 {
            let mut events = mesh.subscribe();
            let deadline = Instant::now() + Duration::from_millis(wait_ms);
            let mut responders = HashSet::new();
            while Instant::now() < deadline && responses.len() < 16 {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let Ok(event) = timeout(remaining, events.recv()).await else {
                    break;
                };
                let Ok(event) = event else {
                    continue;
                };
                let DataMeshEvent::PatternResponse { response } = event else {
                    continue;
                };
                if Some(&response.query_id) != query_id.as_ref() {
                    continue;
                }
                if responders.insert(response.responder_node_id.clone()) {
                    responses.push(response);
                }
            }
        }
    }
    // Update the first-reporter origin index from peer responses.
    // For each pattern returned by a peer, record the earliest-seen timestamp
    // and which node reported it — the primitive for source tracing.
    {
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Ok(mut idx) = state.origin_index.lock() {
            for peer_resp in &responses {
                for pattern in &peer_resp.matches {
                    let entry = idx.entry(pattern.thread_id.clone()).or_insert_with(|| PatternOrigin {
                        thread_id: pattern.thread_id.clone(),
                        first_seen_unix: pattern.last_seen.unix as u64,
                        first_reporter_node_id: peer_resp.responder_node_id.clone(),
                        report_count: 0,
                        reporting_nodes: Vec::new(),
                    });
                    entry.report_count += 1;
                    let ts = pattern.last_seen.unix as u64;
                    if ts < entry.first_seen_unix {
                        entry.first_seen_unix = ts;
                        entry.first_reporter_node_id = peer_resp.responder_node_id.clone();
                    }
                    if !entry.reporting_nodes.iter().any(|n| n == &peer_resp.responder_node_id) {
                        entry.reporting_nodes.push(peer_resp.responder_node_id.clone());
                    }
                }
            }
            // Record local matches under this node's own ID
            for m in &local.matches {
                let entry = idx.entry(m.thread_id.clone()).or_insert_with(|| PatternOrigin {
                    thread_id: m.thread_id.clone(),
                    first_seen_unix: now_unix,
                    first_reporter_node_id: state.node_id.clone(),
                    report_count: 0,
                    reporting_nodes: vec![state.node_id.clone()],
                });
                entry.report_count += 1;
            }
        }
    }

    let response = PatternQueryResponse {
        status: "OK".to_string(),
        query_id,
        local_matches: local.matches,
        responses,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap_or_default()),
    )
}

async fn identity_pattern_sync(
    mesh: DataMeshHandle,
    identity: Arc<Mutex<IdentityRuntime>>,
    share_kind: String,
) {
    let mut events = mesh.subscribe();
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(_) => continue,
        };
        let data_id = match event {
            DataMeshEvent::PayloadReady {
                data_id,
                payload_kind,
                ..
            } => {
                if payload_kind != share_kind {
                    continue;
                }
                data_id
            }
            DataMeshEvent::PatternResponse { response } => {
                if !response.matches.is_empty() {
                    if let Ok(mut runtime) = identity.lock() {
                        runtime.update_patterns(&response.matches);
                    }
                }
                continue;
            }
            DataMeshEvent::NodeMetrics { .. } => continue,
        };
        let payload = match mesh.load_payload(data_id).await {
            Ok(payload) => payload,
            Err(_) => continue,
        };
        let Ok(share) = serde_json::from_slice::<NeuralFabricShare>(&payload) else {
            continue;
        };
        if share.network_patterns.is_empty() {
            continue;
        }
        if let Ok(mut runtime) = identity.lock() {
            runtime.update_patterns(&share.network_patterns);
        }
    }
}

async fn label_queue_sync(
    mesh: DataMeshHandle,
    label_state: Arc<Mutex<LabelQueueState>>,
    share_kind: String,
    neuro: NeuroRuntimeHandle,
) {
    let mut events = mesh.subscribe();
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(_) => continue,
        };
        let data_id = match event {
            DataMeshEvent::PayloadReady {
                data_id,
                payload_kind,
                ..
            } => {
                if payload_kind != share_kind {
                    continue;
                }
                data_id
            }
            DataMeshEvent::NodeMetrics { .. } => continue,
            DataMeshEvent::PatternResponse { .. } => continue,
        };
        let payload = match mesh.load_payload(data_id).await {
            Ok(payload) => payload,
            Err(_) => continue,
        };
        let Ok(share) = serde_json::from_slice::<NeuralFabricShare>(&payload) else {
            continue;
        };
        let label_queue = parse_label_queue(&share);
        let visual_label_queue = parse_visual_label_queue(&share);
        let subnet_report = parse_subnet_report(&share);
        // Apply peer neuro snapshot: activate what fired on the peer so local
        // Hebbian weights learn cross-node co-occurrences at low learning rate.
        if let Some(raw) = share.metadata.get("neuro_snapshot") {
            if let Ok(peer_snap) = serde_json::from_value::<NeuroSnapshot>(raw.clone()) {
                let active: Vec<String> = peer_snap.active_labels.into_iter().collect();
                if !active.is_empty() {
                    neuro.train_weighted(&active, 0.3, false);
                }
            }
        }
        if label_queue.is_none() && visual_label_queue.is_none() && subnet_report.is_none() {
            continue;
        }
        if let Ok(mut state) = label_state.lock() {
            let mut updated = false;
            if let Some(queue) = label_queue {
                state.label_queue = Some(queue);
                updated = true;
            }
            if let Some(queue) = visual_label_queue {
                state.visual_label_queue = Some(queue);
                updated = true;
            }
            if let Some(report) = subnet_report {
                state.subnet_report = Some(report);
                updated = true;
            }
            if updated {
                state.updated_at = Some(share.timestamp);
            }
        }
    }
}

fn build_health_snapshot(state: &ApiState, status: &'static str) -> ApiHealthSnapshot {
    ApiHealthSnapshot {
        status,
        node_id: state.node_id.clone(),
        uptime_secs: state.started_at.elapsed().as_secs(),
        data_mesh_enabled: state.data_enabled,
        data_mesh_active: state.data_mesh.is_some(),
        bridge_enabled: state.bridge_enabled,
        ledger_backend: state.ledger_backend.clone(),
    }
}

fn authorize(state: &ApiState, headers: &HeaderMap) -> Result<()> {
    if !state.require_api_key {
        return Ok(());
    };
    let provided = headers
        .get(&state.api_key_header)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if provided.is_empty() {
        anyhow::bail!("missing api key");
    }
    let candidate = hash_api_key(&provided);
    if !hash_allowed(&candidate, &state.api_key_hashes) {
        anyhow::bail!("invalid api key");
    }
    Ok(())
}

fn rate_limit(state: &ApiState, headers: &HeaderMap, route: &str) -> Result<()> {
    let key = headers
        .get(&state.api_key_header)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("anonymous");
    let mut limiter = state.limiter.lock().expect("api limiter lock");
    let max_requests = match route {
        "bridge" => state.rate_limit_bridge_max,
        "balance" => state.rate_limit_balance_max,
        _ => state.rate_limit_default_max,
    };
    limiter.check_and_update(key, route, max_requests)
}

fn apply_limit<T>(pending: &mut Vec<T>, limit: Option<usize>) {
    if let Some(limit) = limit {
        if pending.len() > limit {
            pending.truncate(limit);
        }
    }
}

fn find_chain_policy<'a>(
    config: &'a BridgeConfig,
    chain_id: &str,
) -> Option<&'a BridgeChainPolicy> {
    config
        .chains
        .iter()
        .find(|chain| chain.chain_id.eq_ignore_ascii_case(chain_id))
}

fn sanitize_option(value: &Option<String>) -> Option<String> {
    value.as_ref().and_then(|item| {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn render_recipient_tag(template: &str, recipient_node_id: &str) -> String {
    template
        .replace("{node_id}", recipient_node_id)
        .replace("{recipient_node_id}", recipient_node_id)
}

fn default_idempotency_key(
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

fn now_timestamp() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    Timestamp { unix }
}

fn build_knowledge_runtime(config: &KnowledgeConfig) -> (KnowledgeRuntime, KnowledgePersist) {
    let persist = KnowledgePersist {
        enabled: config.enabled && config.persist_state,
        path: PathBuf::from(&config.state_path),
    };
    if persist.enabled && persist.path.exists() {
        if let Ok(raw) = fs::read_to_string(&persist.path) {
            if let Ok(runtime) = serde_json::from_str::<KnowledgeRuntime>(&raw) {
                return (runtime, persist);
            }
            warn!(
                target: "w1z4rdv1510n::node",
                path = %persist.path.display(),
                "failed to deserialize knowledge state; starting fresh"
            );
        } else {
            warn!(
                target: "w1z4rdv1510n::node",
                path = %persist.path.display(),
                "failed to read knowledge state; starting fresh"
            );
        }
    }
    let queue = KnowledgeQueue::new(config.queue.clone());
    let runtime = KnowledgeRuntime::new(queue, HealthKnowledgeStore::default());
    (runtime, persist)
}

fn persist_knowledge_runtime(
    persist: &KnowledgePersist,
    runtime: &Arc<Mutex<KnowledgeRuntime>>,
) -> Result<()> {
    if !persist.enabled {
        return Ok(());
    }
    let runtime = runtime.lock().expect("knowledge mutex");
    let payload = serde_json::to_string_pretty(&*runtime)?;
    if let Some(parent) = persist.path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let tmp_path = persist.path.with_extension("tmp");
    fs::write(&tmp_path, payload)?;
    fs::rename(tmp_path, &persist.path)?;
    Ok(())
}

fn build_ingest_config(options: KnowledgeIngestOptions) -> KnowledgeIngestConfig {
    let mut config = KnowledgeIngestConfig::default();
    if let Some(source) = options.source {
        if !source.trim().is_empty() {
            config.source = source;
        }
    }
    config.require_image_bytes = options.require_image_bytes.unwrap_or(false);
    if let Some(normalize) = options.normalize_whitespace {
        config.normalize_whitespace = normalize;
    }
    if let Some(include) = options.include_ocr_blocks {
        config.include_ocr_blocks = include;
    }
    if let Some(command) = options.ocr_command {
        if !command.is_empty() {
            config.ocr_command = Some(command);
        }
    }
    if let Some(timeout) = options.ocr_timeout_secs {
        config.ocr_timeout_secs = timeout.max(1);
    }
    config.asset_root = None;
    config
}

fn hash_allowed(candidate: &[u8; 32], allowlist: &[[u8; 32]]) -> bool {
    let mut matched = false;
    for allowed in allowlist {
        if constant_time_eq_bytes(candidate, allowed) {
            matched = true;
        }
    }
    matched
}

fn constant_time_eq_bytes(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

fn hash_api_key(key: &str) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..32]);
    out
}

pub fn hash_api_key_hex(key: &str) -> String {
    let digest = hash_api_key(key);
    hex_encode(&digest)
}

fn decode_hash_hex(hex: &str) -> Result<[u8; 32]> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("api key hash must be 32 bytes");
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.as_bytes().iter().copied();
    while let Some(high) = iter.next() {
        let low = iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("hex string has odd length"))?;
        let high_val = hex_value(high)?;
        let low_val = hex_value(low)?;
        out.push((high_val << 4) | low_val);
    }
    Ok(out)
}

fn hex_value(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => anyhow::bail!("invalid hex character"),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

#[derive(Debug)]
struct ApiLimiter {
    window_secs: u64,
    entries: std::collections::HashMap<String, ApiLimiterEntry>,
}

#[derive(Debug)]
struct ApiLimiterEntry {
    window_start: Instant,
    count: u32,
    last_seen: Instant,
}

impl ApiLimiter {
    fn new(window_secs: u64) -> Self {
        Self {
            window_secs,
            entries: std::collections::HashMap::new(),
        }
    }

    fn check_and_update(&mut self, key: &str, route: &str, max_requests: u32) -> Result<()> {
        let now = Instant::now();
        self.prune(now);
        let full_key = format!("{key}:{route}");
        let entry = self.entries.entry(full_key).or_insert(ApiLimiterEntry {
            window_start: now,
            count: 0,
            last_seen: now,
        });
        if now.duration_since(entry.window_start).as_secs() >= self.window_secs {
            entry.window_start = now;
            entry.count = 0;
        }
        entry.count = entry.count.saturating_add(1);
        entry.last_seen = now;
        if entry.count > max_requests {
            anyhow::bail!("rate limit exceeded");
        }
        Ok(())
    }

    fn prune(&mut self, now: Instant) {
        let ttl_secs = self.window_secs.saturating_mul(2);
        self.entries
            .retain(|_, entry| now.duration_since(entry.last_seen).as_secs() <= ttl_secs);
    }
}

fn build_ledger(config: &NodeConfig) -> Result<Arc<dyn BlockchainLedger>> {
    if !config.ledger.enabled {
        return Ok(Arc::new(NoopLedger::default()));
    }
    let backend = config.ledger.backend.trim().to_ascii_lowercase();
    match backend.as_str() {
        "local" | "file" => {
            let path = if config.ledger.endpoint.trim().is_empty() {
                node_data_dir().join("ledger.json")
            } else {
                config.ledger.endpoint.clone().into()
            };
            Ok(Arc::new(LocalLedger::load_or_create(
                path,
                config.blockchain.validator_policy.clone(),
                config.blockchain.fee_market.clone(),
                config.blockchain.bridge.clone(),
            )?))
        }
        "none" | "" => Ok(Arc::new(NoopLedger::default())),
        other => {
            tracing::warn!(
                target: "w1z4rdv1510n::node",
                backend = %other,
                "unknown ledger backend; falling back to noop ledger"
            );
            Ok(Arc::new(NoopLedger::default()))
        }
    }
}

// ─── Threat endpoints ─────────────────────────────────────────────────────────

/// Request body for `POST /threat/ingest` and `POST /threat/predict`.
///
/// `frame` maps entity_id → attribute key/value pairs from sensor streams.
/// Attributes mirror what the streaming processor extracts — see
/// `ThreatFieldEngine::ingest_from_attributes` for the full key list.
#[derive(Debug, Deserialize)]
struct ThreatIngestRequest {
    /// Unix timestamp for this observation frame.
    #[serde(default)]
    timestamp_unix: i64,
    /// Per-entity sensor attributes.
    frame: std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
}

/// Ingest sensor attributes into the threat scene (fire-and-forget).
///
/// Returns 204 No Content on success.
async fn threat_ingest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<ThreatIngestRequest>,
) -> StatusCode {
    if authorize(&state, &headers).is_err() {
        return StatusCode::UNAUTHORIZED;
    }
    let ts = Timestamp { unix: req.timestamp_unix };
    if let Ok(mut scene) = state.threat.lock() {
        scene.ingest_frame(&req.frame, ts);
    }
    StatusCode::NO_CONTENT
}

/// Return the current threat overlay for the scene.
///
/// Computes a fresh overlay snapshot from the most recently ingested frame.
async fn threat_overlay(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(e) = authorize(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": e.to_string() })),
        );
    }
    let ts = unix_now();
    let overlay = {
        match state.threat.lock() {
            Ok(mut scene) => scene.overlay(ts),
            Err(_) => return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "threat scene lock poisoned" })),
            ),
        }
    };
    match serde_json::to_value(&overlay) {
        Ok(v)  => (StatusCode::OK, Json(v)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

/// Ingest a frame of sensor attributes and immediately return the resulting overlay.
async fn threat_predict(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<ThreatIngestRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(e) = authorize(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": e.to_string() })),
        );
    }
    let ts = if req.timestamp_unix != 0 {
        Timestamp { unix: req.timestamp_unix }
    } else {
        unix_now()
    };
    let overlay: ThreatOverlay = {
        match state.threat.lock() {
            Ok(mut scene) => {
                scene.ingest_frame(&req.frame, ts);
                scene.overlay(ts)
            }
            Err(_) => return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "threat scene lock poisoned" })),
            ),
        }
    };
    match serde_json::to_value(&overlay) {
        Ok(v)  => (StatusCode::OK, Json(v)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

fn unix_now() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Timestamp { unix: secs }
}

// ── Cluster HTTP handlers ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ClusterInitReq {
    #[serde(default)]
    bind: Option<String>,
    #[serde(default)]
    otp_ttl_secs: Option<u64>,
}

#[derive(Deserialize)]
struct ClusterJoinReq {
    coordinator: String,
    otp: String,
    #[serde(default)]
    bind: Option<String>,
}

/// GET /cluster/status
async fn cluster_status(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let guard = state.cluster.lock().await;
    match guard.as_ref() {
        None => {
            let saved = ClusterNode::saved_state();
            (StatusCode::OK, Json(serde_json::json!({
                "status": "standalone",
                "saved_cluster_id": saved.map(|(id, _nid, _coord)| id.to_string()),
            })))
        }
        Some(node) => {
            let s = node.status().await;
            let coord_str = s.coordinator.as_ref().map(|id| id.to_string()).unwrap_or_default();
            let local_is_coord = s.coordinator.as_ref().map(|c| c == &s.local_id).unwrap_or(false);
            (StatusCode::OK, Json(serde_json::json!({
                "status": "joined",
                "cluster_id": s.cluster_id.to_string(),
                "local_id": s.local_id.to_string(),
                "coordinator": coord_str,
                "role": if local_is_coord { "coordinator" } else { "worker" },
                "ring_size": s.ring_slots,
                "nodes": s.nodes.iter().map(|n| serde_json::json!({
                    "id": n.id.to_string(),
                    "addr": n.addr,
                    "is_coordinator": n.is_coordinator,
                    "capabilities": {
                        "cpu_cores": n.capabilities.cpu_cores,
                        "os": n.capabilities.os,
                    }
                })).collect::<Vec<_>>(),
            })))
        }
    }
}

/// POST /cluster/init
/// Body: `{ "bind": "0.0.0.0:51611", "otp_ttl_secs": 600 }`  (all optional)
/// Returns `{ "cluster_id": "...", "otp": "WORD-NNNN" }`
async fn cluster_init(
    State(state): State<ApiState>,
    Json(req): Json<ClusterInitReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Reject if already in a cluster.
    {
        let guard = state.cluster.lock().await;
        if guard.is_some() {
            return (StatusCode::CONFLICT, Json(serde_json::json!({
                "error": "already in a cluster — call /cluster/status to inspect or restart node to reset"
            })));
        }
    }
    let bind_str = req.bind.unwrap_or_else(|| "0.0.0.0:51611".to_string());
    let bind_addr: SocketAddr = match bind_str.parse() {
        Ok(a) => a,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": format!("invalid bind address: {e}") }))),
    };
    // Derive advertise_addr: replace 0.0.0.0 with the configured LAN IP if known.
    let advertise_addr = state.cluster_advertise_addr.map(|mut a| {
        a.set_port(bind_addr.port());
        a
    });
    let config = ClusterConfig {
        bind_addr,
        advertise_addr,
        otp_ttl_secs: req.otp_ttl_secs.unwrap_or(600),
        ..Default::default()
    };
    match ClusterNode::init(config).await {
        Ok((node, otp)) => {
            let cluster_id = node.cluster_id.to_string();
            *state.cluster.lock().await = Some(node);
            tracing::info!("cluster initialised via HTTP API: {cluster_id}");
            (StatusCode::OK, Json(serde_json::json!({
                "status": "ok",
                "cluster_id": cluster_id,
                "otp": otp,
                "bind": bind_str,
            })))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

/// POST /cluster/join
/// Body: `{ "coordinator": "192.168.1.84:51611", "otp": "WORD-NNNN", "bind": "0.0.0.0:51611" }`
async fn cluster_join(
    State(state): State<ApiState>,
    Json(req): Json<ClusterJoinReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    {
        let guard = state.cluster.lock().await;
        if guard.is_some() {
            return (StatusCode::CONFLICT, Json(serde_json::json!({
                "error": "already in a cluster"
            })));
        }
    }
    let coord_addr: SocketAddr = match req.coordinator.parse() {
        Ok(a) => a,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": format!("invalid coordinator address: {e}") }))),
    };
    let bind_str = req.bind.unwrap_or_else(|| "0.0.0.0:51611".to_string());
    let bind_addr: SocketAddr = match bind_str.parse() {
        Ok(a) => a,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": format!("invalid bind address: {e}") }))),
    };
    let advertise_addr = state.cluster_advertise_addr.map(|mut a| {
        a.set_port(bind_addr.port());
        a
    });
    let config = ClusterConfig { bind_addr, advertise_addr, ..Default::default() };
    match ClusterNode::join(config, coord_addr, &req.otp).await {
        Ok(node) => {
            let s = node.status().await;
            let cluster_id = s.cluster_id.to_string();
            let node_count = s.nodes.len();
            *state.cluster.lock().await = Some(node);
            tracing::info!("joined cluster {cluster_id} via HTTP API ({node_count} nodes)");
            (StatusCode::OK, Json(serde_json::json!({
                "status": "ok",
                "cluster_id": cluster_id,
                "node_count": node_count,
            })))
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

/// POST /cluster/otp
/// Generates a fresh join OTP.  Only works if this node is coordinator.
async fn cluster_otp(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let guard = state.cluster.lock().await;
    match guard.as_ref() {
        None => (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "not in a cluster" }))),
        Some(node) => match node.new_otp().await {
            Ok(otp) => (StatusCode::OK, Json(serde_json::json!({ "otp": otp }))),
            Err(e)  => (StatusCode::FORBIDDEN, Json(serde_json::json!({ "error": e.to_string() }))),
        }
    }
}

/// POST /cluster/leave
/// Leave the cluster gracefully.  This node's ring slots are redistributed
/// to the remaining nodes by the coordinator via consistent hashing.
/// After this call the node is standalone again.
async fn cluster_leave(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let maybe_node = state.cluster.lock().await.take();
    match maybe_node {
        None => (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "not in a cluster" }))),
        Some(node) => match node.leave().await {
            Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "status": "ok", "message": "left cluster — now standalone" }))),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
        }
    }
}

/// POST /cluster/resign
/// Coordinator only.  Triggers an election so another node takes over,
/// then leaves the cluster.  Ring slots are redistributed automatically.
async fn cluster_resign(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let maybe_node = state.cluster.lock().await.take();
    match maybe_node {
        None => (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "not in a cluster" }))),
        Some(node) => match node.resign().await {
            Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "status": "ok", "message": "resigned — election triggered, now standalone" }))),
            Err(e) => (StatusCode::FORBIDDEN, Json(serde_json::json!({ "error": e.to_string() }))),
        }
    }
}

/// POST /node/shutdown
/// Shut down this node process cleanly.  If in a cluster, leaves/resigns first.
async fn node_shutdown(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Best-effort cluster leave before exiting.
    let maybe_node = state.cluster.lock().await.take();
    if let Some(node) = maybe_node {
        if node.is_coordinator().await {
            node.resign().await.ok();
        } else {
            node.leave().await.ok();
        }
    }
    tracing::info!("node shutdown requested via API");
    // Spawn a delayed exit so the HTTP response can be sent first.
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        std::process::exit(0);
    });
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok", "message": "shutting down" })))
}

// ── Multimodal media ingestion ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct MotionPointReq { x: f32, y: f32, t_secs: f32, #[serde(default)] click: bool }

#[derive(Deserialize)]
struct KeyEventReq {
    key: String,
    #[serde(default)] ctrl: bool,
    #[serde(default)] shift: bool,
    #[serde(default)] alt: bool,
    #[serde(default)] t_secs: f32,
}

#[derive(Deserialize)]
struct MediaTrainReq {
    /// "image" | "audio" | "text" | "page" | "motion" | "action" | "full"
    modality: String,
    /// Base64-encoded image bytes (JPEG/PNG). Used for image/page/full.
    #[serde(default)]
    data_b64: Option<String>,
    /// Plain text goal/description. Used for text/page/full.
    #[serde(default)]
    text: Option<String>,
    /// Structured text spans with layout metadata.
    #[serde(default)]
    spans: Option<Vec<TextSpanReq>>,
    /// Mouse trajectory for motion/full modalities.
    #[serde(default)]
    motion: Option<Vec<MotionPointReq>>,
    /// Keyboard events for action/full modalities.
    #[serde(default)]
    keys: Option<Vec<KeyEventReq>>,
    /// Learning rate scale (default 1.0).
    #[serde(default = "default_lr")]
    lr_scale: f32,
}

fn default_lr() -> f32 { 1.0 }

#[derive(Deserialize)]
struct TextSpanReq {
    text: String,
    #[serde(default = "default_role")]  role: String,
    #[serde(default = "default_one")]   size_ratio: f32,
    #[serde(default)]                   bold: bool,
    #[serde(default)]                   italic: bool,
    #[serde(default)]                   indent: usize,
    #[serde(default)]                   x_frac: f32,
    #[serde(default)]                   y_frac: f32,
    #[serde(default)]                   seq_index: usize,
    #[serde(default = "default_one_usize")] seq_total: usize,
}

fn default_role() -> String { "body".to_string() }
fn default_one() -> f32 { 1.0 }
fn default_one_usize() -> usize { 1 }

fn parse_text_role(s: &str) -> TextRole {
    match s {
        "heading"    => TextRole::Heading,
        "subheading" => TextRole::Subheading,
        "caption"    => TextRole::Caption,
        "list"       => TextRole::ListItem,
        "label"      => TextRole::Label,
        "code"       => TextRole::Code,
        "footnote"   => TextRole::Footnote,
        _            => TextRole::Body,
    }
}

/// Encode one frame's modalities into a flat, deduped label list.
/// Returns `Err` with a user-facing message if decoding fails.
fn encode_media_labels(
    modality: &str,
    data_b64: Option<&str>,
    text: Option<&str>,
    spans: Option<&[TextSpanReq]>,
    motion: Option<&[MotionPointReq]>,
    keys: Option<&[KeyEventReq]>,
) -> Result<Vec<String>, String> {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut labels: Vec<String> = Vec::new();

    // Image
    if modality == "image" || modality == "page" || modality == "full" {
        if let Some(data) = data_b64 {
            let bytes = b64.decode(data).map_err(|_| "invalid base64 for data_b64".to_string())?;
            let enc = ImageBitsEncoder::new(ImageBitsConfig::default());
            let out = enc.encode_bytes(&bytes)
                .ok_or_else(|| "could not decode image bytes".to_string())?;
            labels.extend_from_slice(&out.labels);
        }
    }

    // Audio — only for "audio"/"full" modalities.
    // "page" is image+text (PDF documents); it does NOT carry WAV data.
    if modality == "audio" || modality == "full" {
        if let Some(data) = data_b64 {
            let bytes = b64.decode(data).map_err(|_| "invalid base64 for data_b64".to_string())?;
            let enc = AudioBitsEncoder::new(AudioBitsConfig::default());
            let out = enc.encode_wav_bytes(&bytes)
                .ok_or_else(|| "could not decode WAV bytes".to_string())?;
            labels.extend_from_slice(&out.labels);
        }
    }

    // Text
    if modality == "text" || modality == "page" || modality == "full" {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        if let Some(span_reqs) = spans {
            let tspans: Vec<TextSpan> = span_reqs.iter().map(|sr| {
                let emphasis = match (sr.bold, sr.italic) {
                    (true,  true)  => TextEmphasis::BoldItalic,
                    (true,  false) => TextEmphasis::Bold,
                    (false, true)  => TextEmphasis::Italic,
                    _              => TextEmphasis::None,
                };
                TextSpan::positioned(
                    &sr.text, parse_text_role(&sr.role), sr.size_ratio, emphasis,
                    sr.indent, sr.x_frac, sr.y_frac, sr.seq_index, sr.seq_total,
                )
            }).collect();
            labels.extend_from_slice(&enc.encode_spans(&tspans).labels);
        } else if let Some(plain) = text {
            labels.extend_from_slice(&enc.encode_plain(plain).labels);
        }
    }

    // Motion
    if modality == "motion" || modality == "full" {
        if let Some(pts) = motion {
            let samples: Vec<MotionSample> = pts.iter()
                .map(|p| MotionSample { x: p.x, y: p.y, t_secs: p.t_secs, click: p.click })
                .collect();
            let out = MotionBitsEncoder::new(MotionBitsConfig::default()).encode_trajectory(&samples);
            labels.extend_from_slice(&out.labels);
        }
    }

    // Keyboard
    if modality == "action" || modality == "full" {
        if let Some(evts) = keys {
            let events: Vec<KeyEvent> = evts.iter()
                .map(|e| KeyEvent { key: e.key.clone(), ctrl: e.ctrl, shift: e.shift, alt: e.alt, t_secs: e.t_secs })
                .collect();
            let out = KeyboardBitsEncoder::new(KeyboardBitsConfig::default()).encode_sequence(&events);
            labels.extend_from_slice(&out.labels);
        }
    }

    labels.sort_unstable();
    labels.dedup();
    Ok(labels)
}

/// POST /media/train
/// Encode one or more modalities and train the NeuronPool in a single co-activation.
async fn media_train(
    State(state): State<ApiState>,
    Json(req): Json<MediaTrainReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    let labels = match encode_media_labels(
        &req.modality,
        req.data_b64.as_deref(),
        req.text.as_deref(),
        req.spans.as_deref(),
        req.motion.as_deref(),
        req.keys.as_deref(),
    ) {
        Ok(l) => l,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": e }))),
    };

    if labels.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "no labels produced — check modality and input" })));
    }

    let label_count = labels.len();
    // Run the Hebbian update on a blocking thread so we don't stall the async
    // executor (and therefore heartbeat / health / cluster tasks) during heavy
    // training sessions with large pools.  The pool mutex is held only inside
    // spawn_blocking, so concurrent requests still queue correctly.
    let neuro = state.neuro.clone();
    let lr = req.lr_scale;
    tokio::task::spawn_blocking(move || {
        neuro.train_weighted(&labels, lr, false);
    }).await.ok();

    // NOTE: auto-save removed — saving a multi-GB pool on every training call
    // saturates the disk and blocks Tokio threads.  Use POST /neuro/checkpoint
    // to persist the pool explicitly (the training script does this periodically).

    (StatusCode::OK, Json(serde_json::json!({
        "trained": true,
        "modality": req.modality,
        "label_count": label_count,
    })))
}

// ── Sequence training ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TrainSequenceFrame {
    /// Seconds from sequence start. Used to compute temporal gaps between frames.
    #[serde(default)]
    t_secs: f32,
    modality: String,
    #[serde(default)] data_b64: Option<String>,
    #[serde(default)] text: Option<String>,
    #[serde(default)] spans: Option<Vec<TextSpanReq>>,
    #[serde(default)] motion: Option<Vec<MotionPointReq>>,
    #[serde(default)] keys: Option<Vec<KeyEventReq>>,
    #[serde(default = "default_lr")] lr_scale: f32,
}

#[derive(Deserialize)]
struct TrainSequenceReq {
    /// Ordered frames. Each is encoded and trained independently, then adjacent
    /// frames are bridged with a temporally-decayed cross-activation.
    frames: Vec<TrainSequenceFrame>,
    /// Time-constant (seconds) for the bridge LR decay: bridge_lr = lr * exp(-dt / tau).
    /// Smaller tau = only very adjacent frames get linked. Default 2.0s.
    #[serde(default = "default_temporal_tau")]
    temporal_tau: f32,
}

fn default_temporal_tau() -> f32 { 2.0 }

/// POST /media/train_sequence
/// Train an ordered sequence of frames with temporal Hebbian bridging.
///
/// Each frame is trained at its own lr_scale.  Adjacent frame pairs are then
/// co-activated at a decayed rate: `bridge_lr = frame_lr * exp(-dt / tau)`,
/// creating soft "A tends to precede B" associations without overwriting
/// within-frame weights.
async fn media_train_sequence(
    State(state): State<ApiState>,
    Json(req): Json<TrainSequenceReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.frames.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "frames must not be empty" })));
    }

    let tau = req.temporal_tau.max(0.01); // guard against division by zero

    // Encode all frames (CPU-bound but not pool-locked) on the async executor.
    let mut frame_labels: Vec<Vec<String>> = Vec::with_capacity(req.frames.len());
    let mut frame_lrs: Vec<f32> = Vec::with_capacity(req.frames.len());
    let mut frame_ts: Vec<f32> = Vec::with_capacity(req.frames.len());
    let mut total_labels = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for (i, frame) in req.frames.iter().enumerate() {
        match encode_media_labels(
            &frame.modality,
            frame.data_b64.as_deref(),
            frame.text.as_deref(),
            frame.spans.as_deref(),
            frame.motion.as_deref(),
            frame.keys.as_deref(),
        ) {
            Ok(labels) if !labels.is_empty() => {
                total_labels += labels.len();
                frame_labels.push(labels);
            }
            Ok(_) => {
                errors.push(format!("frame {i}: no labels produced"));
                frame_labels.push(Vec::new());
            }
            Err(e) => {
                errors.push(format!("frame {i}: {e}"));
                frame_labels.push(Vec::new());
            }
        }
        frame_lrs.push(frame.lr_scale);
        frame_ts.push(frame.t_secs);
    }

    // All Hebbian updates on a blocking thread — keeps async executor free.
    let neuro = state.neuro.clone();
    let trained_frames = frame_labels.iter().filter(|l| !l.is_empty()).count();
    tokio::task::spawn_blocking(move || {
        // Pass 1: each frame independently.
        for (labels, lr) in frame_labels.iter().zip(frame_lrs.iter()) {
            if !labels.is_empty() {
                neuro.train_weighted(labels, *lr, false);
            }
        }
        // Pass 2: bridge adjacent frames.
        for i in 1..frame_labels.len() {
            let prev = &frame_labels[i - 1];
            let curr = &frame_labels[i];
            if prev.is_empty() || curr.is_empty() { continue; }
            let dt = (frame_ts[i] - frame_ts[i - 1]).abs();
            let base_lr = (frame_lrs[i - 1] + frame_lrs[i]) * 0.5;
            let bridge_lr = base_lr * (-dt / tau).exp();
            if bridge_lr < 1e-4 { continue; }
            let mut bridge: Vec<String> = prev.iter().chain(curr.iter()).cloned().collect();
            bridge.sort_unstable();
            bridge.dedup();
            neuro.train_weighted(&bridge, bridge_lr, false);
        }
    }).await.ok();

    let mut resp = serde_json::json!({
        "trained_frames": trained_frames,
        "total_frames": req.frames.len(),
        "total_labels": total_labels,
        "temporal_tau": tau,
    });
    if !errors.is_empty() {
        resp["warnings"] = serde_json::json!(errors);
    }
    (StatusCode::OK, Json(resp))
}

#[derive(Deserialize)]
struct PlaybackReq {
    /// English goal text — what should happen ("click the red button")
    goal: String,
    /// Current screenshot as base64 JPEG/PNG (optional but improves accuracy)
    #[serde(default)]
    screen_b64: Option<String>,
    #[serde(default = "default_hops")] hops: usize,
}

/// POST /media/playback
/// Given a goal and optional screenshot, predict what action to perform.
/// Returns the most activated motion zone (where to move the cursor)
/// and whether a click is expected there.
async fn media_playback(
    State(state): State<ApiState>,
    Json(req): Json<PlaybackReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD;
    let motion_cfg = MotionBitsConfig::default();

    // Build seed labels from goal text — discriminative words only.
    //
    // Strategy: keep only `txt:word_*` labels whose word is NOT a common
    // English stopword.  Common words like "click", "the", "a", "button"
    // co-train with every motion zone (they appear in ALL goals), so including
    // them floods all zones equally and kills discrimination.  Rare, goal-specific
    // words ("red", "blue", "green", "chess", "left", …) are the actual signal.
    //
    // If ALL words in the goal are stopwords we fall back to the full label set.
    const STOPWORDS: &[&str] = &[
        "click", "press", "tap", "touch", "select", "choose", "go", "move",
        "the", "a", "an", "this", "that", "it",
        "button", "link", "item", "target", "object", "thing", "on",
        "to", "at", "in", "of", "and", "or", "with",
    ];
    let text_enc = TextBitsEncoder::new(TextBitsConfig::default());
    let text_out = text_enc.encode_plain(&req.goal);

    // Collect only simple word labels for non-stopwords.
    // Labels like txt:word_red are kept; compound labels like
    // txt:word_click_zone_x0_y0 or txt:role_body_word_red are excluded because
    // the word extracted from them ("click_zone_x0_y0") wouldn't match a stopword
    // exactly but still contains noise.  We only accept labels whose word part
    // has no underscore (i.e. it's a single bare word).
    let discriminative: Vec<String> = text_out.labels.iter()
        .filter(|l| {
            if let Some(word) = l.strip_prefix("txt:word_") {
                // Only accept bare words (no underscore = not a compound label)
                !word.contains('_') && !STOPWORDS.contains(&word)
            } else {
                false
            }
        })
        .cloned()
        .collect();

    // Fall back to all text labels if nothing survived stopword filtering
    let mut seed_labels: Vec<String> = if discriminative.is_empty() {
        text_out.labels.clone()
    } else {
        discriminative
    };

    // Add screen image zone labels if provided — only zone-aggregate labels
    // (img:z{x}_{y}) so we get spatial grounding without per-hue noise.
    if let Some(ref data) = req.screen_b64 {
        if let Ok(bytes) = b64.decode(data) {
            let img_enc = ImageBitsEncoder::new(ImageBitsConfig::default());
            if let Some(img_out) = img_enc.encode_bytes(&bytes) {
                // Zone aggregate labels only — exclude per-hue/sat/val and edge labels
                // which are noisy when the scene is static across all goals.
                for label in &img_out.labels {
                    if label.starts_with("img:z") {
                        seed_labels.push(label.clone());
                    }
                }
            }
        }
    }

    seed_labels.sort_unstable();
    seed_labels.dedup();

    // Low threshold: playback queries need weak associations that training at
    // 50 examples won't have pushed above the default streaming threshold (0.55).
    let activated = state.neuro.propagate_all_threshold(&seed_labels, req.hops, 0.02);
    let activated_sorted: Vec<(String, f32)> = {
        let mut v: Vec<(String, f32)> = activated.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    };

    // Decode the highest-confidence motion target
    let target = MotionBitsEncoder::decode_target(&activated_sorted);

    // Check if click was predicted
    let click_strength = activated_sorted.iter()
        .find(|(l, _)| l == "act:click")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    let has_click = click_strength > 0.1;

    // Build response
    let action = if let Some((zx, zy, strength)) = target {
        let (x_frac, y_frac) = MotionBitsEncoder::new(motion_cfg).zone_to_frac(zx, zy);
        serde_json::json!({
            "type": if has_click { "move_and_click" } else { "move" },
            "zone_x": zx,
            "zone_y": zy,
            "x_frac": x_frac,
            "y_frac": y_frac,
            "confidence": strength,
            "click": has_click,
            "click_strength": click_strength,
        })
    } else {
        serde_json::json!({ "type": "none", "reason": "no motion labels activated" })
    };

    // Top activations for debugging
    let top: Vec<serde_json::Value> = activated_sorted.iter().take(20)
        .map(|(l, s)| serde_json::json!({ "label": l, "strength": s }))
        .collect();

    (StatusCode::OK, Json(serde_json::json!({
        "action": action,
        "top_activations": top,
    })))
}

#[derive(Deserialize)]
struct PropagateReq {
    seed_labels: Vec<String>,
    #[serde(default = "default_hops")] hops: usize,
    #[serde(default = "default_min_strength")] min_strength: f32,
    #[serde(default)] top_k: Option<usize>,
}
fn default_hops() -> usize { 2 }
fn default_min_strength() -> f32 { 0.02 }

/// POST /neuro/propagate
/// Feed seed labels from any modality; returns every label that fires above
/// threshold — including cross-modal activations. No weights are changed.
async fn neuro_propagate(
    State(state): State<ApiState>,
    Json(req): Json<PropagateReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.seed_labels.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "seed_labels must not be empty" })));
    }
    let activated = state.neuro.propagate_all_threshold(&req.seed_labels, req.hops, req.min_strength);
    let mut sorted: Vec<(String, f32)> = activated.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = req.top_k {
        sorted.truncate(k);
    }
    let result: Vec<serde_json::Value> = sorted.iter()
        .map(|(label, strength)| serde_json::json!({ "label": label, "strength": strength }))
        .collect();
    (StatusCode::OK, Json(serde_json::json!({ "activated": result })))
}

#[derive(Deserialize)]
struct AskReq {
    text: String,
    #[serde(default = "default_hops")] hops: usize,
    #[serde(default = "default_ask_top_k")] top_k: usize,
    #[serde(default = "default_ask_min_strength")] min_strength: f32,
}
fn default_ask_top_k() -> usize { 20 }
fn default_ask_min_strength() -> f32 { 0.05 }

/// POST /neuro/ask
/// Plain English in → the node returns what it associates with that text,
/// drawn directly from its trained Hebbian fabric and QA store.
/// This is the primary inference interface — the node speaking from what it learned.
async fn neuro_ask(
    State(state): State<ApiState>,
    Json(req): Json<AskReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "text must not be empty" })));
    }

    // 1. Hebbian associations from the neural fabric
    let associations = state.neuro.ask_text(&req.text, req.hops, req.top_k, req.min_strength);
    let assoc_json: Vec<serde_json::Value> = associations.iter().map(|(label, strength)| {
        // Strip the "txt:word_" prefix for clean output
        let word = label.strip_prefix("txt:word_").unwrap_or(label.as_str());
        serde_json::json!({ "word": word, "strength": strength })
    }).collect();

    // 2. QA recall from the question-answer store
    let now = now_timestamp();
    let qa_report = {
        let mut qa = state.qa_runtime.lock().expect("qa mutex");
        qa.query(&req.text, now)
    };
    let qa_answers: Vec<serde_json::Value> = qa_report.results.iter().map(|r| {
        serde_json::json!({
            "answer": r.answer,
            "confidence": r.confidence,
            "book_id": r.book_id,
            "page": r.page_index,
        })
    }).collect();

    (StatusCode::OK, Json(serde_json::json!({
        "question": req.text,
        "associations": assoc_json,
        "qa_answers": qa_answers,
    })))
}

/// POST /neuro/checkpoint
/// Force-save the NeuronPool to disk immediately.
/// Returns the path written and the neuron count.
async fn neuro_checkpoint(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    match state.neuro.save_pool() {
        Ok(()) => {
            let path = state.neuro.pool_state_path().unwrap_or("").to_string();
            (StatusCode::OK, Json(serde_json::json!({
                "saved": true,
                "path": path,
            })))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e }))),
    }
}

/// GET /neuro/motifs
/// Return all meta-motifs discovered so far across every hierarchy level.
/// Empty until enough recurring label-sequence patterns cross min_support (default 3).
///
/// Each meta-motif represents a label sequence that recurred enough times to
/// be promoted.  Higher-level motifs are sequences-of-sequences.
async fn neuro_motifs(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let motifs = state.neuro.meta_motifs();
    let count = motifs.len();
    (StatusCode::OK, Json(serde_json::json!({
        "count": count,
        "motifs": motifs,
    })))
}

#[derive(Deserialize)]
struct MotifPredictReq {
    /// The label seen most recently — predict what comes next.
    last_label: String,
    /// Max number of predictions to return (default 10).
    #[serde(default = "default_predict_limit")]
    limit: usize,
}
fn default_predict_limit() -> usize { 10 }

/// POST /neuro/motifs/predict
/// Given a label that appeared recently, return the most likely successor
/// labels based on learned transition probabilities across all motif levels.
///
/// Body: { "last_label": "txt:word_photosynthesis", "limit": 10 }
/// Response: { "predictions": [{ "label": "...", "probability": 0.43 }, ...] }
async fn neuro_motifs_predict(
    State(state): State<ApiState>,
    Json(req): Json<MotifPredictReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.last_label.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "last_label must not be empty" })));
    }
    let mut preds = state.neuro.motif_predictions(&req.last_label);
    preds.truncate(req.limit);
    let result: Vec<serde_json::Value> = preds.iter()
        .map(|(label, prob)| serde_json::json!({ "label": label, "probability": prob }))
        .collect();
    (StatusCode::OK, Json(serde_json::json!({ "predictions": result })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;
    use w1z4rdv1510n::network::compute_payload_hash;
    use w1z4rdv1510n::streaming::{FigureAsset, KnowledgeDocument, TextBlock};

    #[test]
    fn knowledge_persistence_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let state_path = dir.path().join("knowledge_state.json");
        let mut config = KnowledgeConfig::default();
        config.enabled = true;
        config.persist_state = true;
        config.state_path = state_path.to_string_lossy().into_owned();
        let (mut runtime, persist) = build_knowledge_runtime(&config);
        let doc = KnowledgeDocument {
            doc_id: "doc1".to_string(),
            source: "NLM".to_string(),
            title: Some("Sample".to_string()),
            text_blocks: vec![TextBlock {
                block_id: "t1".to_string(),
                text: "Figure 1 shows data.".to_string(),
                section: None,
                order: 0,
                figure_refs: vec!["F1".to_string()],
                source: "xml".to_string(),
                confidence: 1.0,
            }],
            figures: vec![FigureAsset {
                figure_id: "F1".to_string(),
                label: Some("Figure 1".to_string()),
                caption: Some("Cap".to_string()),
                image_ref: "fig1.png".to_string(),
                image_hash: compute_payload_hash(b"img"),
                order: 0,
                ocr_text: None,
            }],
            metadata: HashMap::new(),
        };
        let ts = Timestamp { unix: 1 };
        runtime.ingest_document(doc, ts);
        let runtime = Arc::new(Mutex::new(runtime));
        persist_knowledge_runtime(&persist, &runtime).expect("persist");
        let (loaded, _) = build_knowledge_runtime(&config);
        let report = loaded.queue_report(Timestamp { unix: 2 });
        assert!(report.is_some());
    }
}
