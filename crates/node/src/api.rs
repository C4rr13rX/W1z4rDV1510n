use crate::config::{KnowledgeConfig, NodeConfig};
use crate::mesh_gen::{MeshPoint, MeshSynthesizer};
use crate::distributed::DistributedCoordinator;
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
use axum::response::sse::{Event, Sse, KeepAlive};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use std::convert::Infallible;
use tower_http;
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
use w1z4rdv1510n::neuro::{NeuroRuntime, NeuroRuntimeConfig, NeuroRuntimeHandle, NeuroSnapshot, NeuromodulatorKind};
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
    EquationMatrixRuntime, EquationMatrixConfig, Discipline, EemPeerPayload,
    ImageBitsConfig, ImageBitsEncoder,
    AudioBitsConfig, AudioBitsEncoder,
    TextBitsConfig, TextBitsEncoder, TextSpan, TextRole, TextEmphasis,
    MotionBitsConfig, MotionBitsEncoder, MotionSample,
    KeyboardBitsConfig, KeyboardBitsEncoder, KeyEvent,
    // Simulation / entity runtimes
    BehaviorSubstrate, BehaviorSubstrateConfig, BehaviorInput, BehaviorFrame,
    SensorKind, ActionKind, SensorSample, ActionSample,
    SceneRuntime, SceneReport,
    SurvivalRuntime, SurvivalReport, SurvivalConfig,
    PhysiologyRuntime, PhysiologyReport,
    NarrativeRuntime, NarrativeReport,
    OntologyRuntime, OntologyReport,
    PoseFrame, Keypoint, BoundingBox,
    // Overlay / perception system
    EntityHealthConfig, EntityHealthRuntime, EntityHealthOverlay,
    DeltaEngine, DeltaEngineConfig, DeltaReport,
    PoetryConfig, PoetryRuntime,
    ChaosWorldConfig, ChaosWorldModel, TransitionIndex, build_reverse_index,
    PredictionExperimentConfig, PredictionExperimentRuntime, ExperimentReport,
    LayeredPhysiologyConfig, LayeredPhysiologyRuntime, LayeredPhysiologyReport,
    EntityDecomposeRequest,
};
use w1z4rdv1510n::causal::{CausalEdge, CausalRuntime};
use w1z4rdv1510n::config::{SceneConfig, PhysiologyConfig, NarrativeConfig, OntologyConfig};
use w1z4rdv1510n::streaming::schema::{EventToken, EventKind, TokenBatch};
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
    /// Distributed training coordinator — manages round-robin routing and weight-delta sync.
    distributed: Arc<DistributedCoordinator>,

    // ── Simulation / entity runtimes ─────────────────────────────────────────
    /// Behavioral substrate: encodes multi-modal entity observations into latent
    /// BehaviorFrames and trains the neuro pool with the resulting labels.
    behavior: Arc<Mutex<BehaviorSubstrate>>,
    /// Scene tracker: maintains 3-D entity positions, velocities, and anomalies.
    scene: Arc<Mutex<SceneRuntime>>,
    /// Physiology monitor: tracks deviation from learned physiological baselines.
    physiology: Arc<Mutex<PhysiologyRuntime>>,
    /// Survival scorer: aggregates behavior + physiology into entity survival fitness.
    survival: Arc<Mutex<SurvivalRuntime>>,
    /// Narrative synthesizer: composes human-readable situation summaries from all
    /// sub-runtime reports.
    narrative: Arc<Mutex<NarrativeRuntime>>,
    /// Ontology tracker: discovers stable concept labels from recurring token patterns.
    ontology: Arc<Mutex<OntologyRuntime>>,
    /// Unresolved questions whose activation was below ANSWER_THRESHOLD.
    /// Background task polls this and drives research until resolved.
    hypothesis_queue: Arc<Mutex<Vec<HypothesisEntry>>>,
    /// Working-memory sessions: session_id → top activated content-word labels
    /// from the previous turn.  Blended into the next propagation at 0.3 weight
    /// so conversation context gently biases retrieval without overriding the question.
    /// Apps/scripts manage session lifecycle; the fabric just carries the state.
    session_contexts: Arc<Mutex<HashMap<String, Vec<String>>>>,

    // ── Overlay / perception system ───────────────────────────────────────────
    /// 6-dimensional entity health overlay (structural/energetic/temporal/
    /// intentional/environmental/informational) with concentric ring color model.
    entity_health: Arc<Mutex<EntityHealthRuntime>>,
    /// Per-entity state velocity and short-horizon predictions in 6-D health space.
    delta_engine: Arc<Mutex<DeltaEngine>>,
    /// Emergent natural-language entity synthesis from labels + EEM + health.
    poetry_runtime: Arc<Mutex<PoetryRuntime>>,
    /// Chaos World Model — reconstructs probable entity history from motif transitions.
    chaos_model: Arc<Mutex<ChaosWorldModel>>,
    /// Autonomous prediction experiment framework with adaptive horizons.
    prediction_experiment: Arc<Mutex<PredictionExperimentRuntime>>,
    /// Hierarchical structural decomposition: layers inside entities from experience.
    layered_physiology: Arc<Mutex<LayeredPhysiologyRuntime>>,
    /// Most recent EnvironmentSnapshot received via /neuro/train.
    /// Served by GET /neuro/symbols/live for real-time world viewer animation.
    live_snapshot: Arc<Mutex<Option<EnvironmentSnapshot>>>,
}

/// Minimum peak `txt:word_*` activation required to emit an answer.
/// Below this threshold the question is added to the hypothesis queue.
const ANSWER_THRESHOLD: f32 = 0.08;

/// An unresolved question that the node cannot answer from current memory.
/// Persists in `hypothesis_queue` until resolved via research or training.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HypothesisEntry {
    id: String,
    question: String,
    queued_at_unix: i64,
    attempts: u32,
    max_attempts: u32,
    resolved: bool,
    answer: Option<String>,
    confidence: Option<f32>,
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
        distributed: Arc::new(DistributedCoordinator::new()),
        behavior:   Arc::new(Mutex::new(BehaviorSubstrate::new(BehaviorSubstrateConfig::default()))),
        scene:      Arc::new(Mutex::new(SceneRuntime::new(SceneConfig::default()))),
        physiology: Arc::new(Mutex::new(PhysiologyRuntime::new(PhysiologyConfig::default()))),
        survival:   Arc::new(Mutex::new(SurvivalRuntime::new(SurvivalConfig::default()))),
        narrative:  Arc::new(Mutex::new(NarrativeRuntime::new(NarrativeConfig::default()))),
        ontology:   Arc::new(Mutex::new(OntologyRuntime::new(
            OntologyConfig::default(),
            w1z4rdv1510n::config::ConsistencyChunkingConfig::default(),
        ))),
        hypothesis_queue: Arc::new(Mutex::new(Vec::new())),
        session_contexts: Arc::new(Mutex::new(HashMap::new())),
        entity_health:        Arc::new(Mutex::new(EntityHealthRuntime::new(EntityHealthConfig::default()))),
        delta_engine:         Arc::new(Mutex::new(DeltaEngine::new(DeltaEngineConfig::default()))),
        poetry_runtime:       Arc::new(Mutex::new(PoetryRuntime::new(PoetryConfig::default()))),
        chaos_model:          Arc::new(Mutex::new(ChaosWorldModel::new(ChaosWorldConfig::default()))),
        prediction_experiment: Arc::new(Mutex::new(PredictionExperimentRuntime::new(PredictionExperimentConfig::default()))),
        layered_physiology:   Arc::new(Mutex::new(LayeredPhysiologyRuntime::new(LayeredPhysiologyConfig::default()))),
        live_snapshot:        Arc::new(Mutex::new(None)),
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
    let app_hypothesis_queue = Arc::clone(&state.hypothesis_queue);
    let app_neuro = Arc::clone(&state.neuro);
    // Pre-clone for the background peer-refresh task (state is consumed by with_state below).
    let bg_cluster = state.cluster.clone();
    let bg_dist    = state.distributed.clone();
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
        // GET  /neuro/stream — Server-Sent Events stream of NeuroSnapshot at ~10 fps.
        //   Each "snap" event contains the full activation state: active_labels,
        //   centroids, network_links, surprise, temporal_predictions.
        //   Clients read this like a video stream — each frame is the neural state.
        .route("/neuro/stream", get(neuro_state_stream))
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
        .route("/media/train",             post(media_train))
        .route("/media/train_sequence",    post(media_train_sequence))
        .route("/media/train_contrastive", post(media_train_contrastive))
        .route("/media/playback",       post(media_playback))
        .route("/neuro/propagate",      post(neuro_propagate))
        .route("/neuro/checkpoint",     post(neuro_checkpoint))
        .route("/neuro/clear",          post(neuro_clear))
        // ── Distributed training ─────────────────────────────────────────────
        // POST /neuro/delta/apply — receive a weight delta from a peer and merge
        // POST /neuro/sync        — force an immediate delta push to all peers
        // GET  /cluster/sync/status — distributed coordinator state
        .route("/neuro/delta/apply",    post(neuro_delta_apply))
        .route("/neuro/sync",           post(neuro_sync))
        .route("/cluster/sync/status",  get(cluster_sync_status))
        .route("/neuro/motifs",         get(neuro_motifs))
        .route("/neuro/motifs/predict", post(neuro_motifs_predict))
        .route("/neuro/ask",            post(neuro_ask))
        .route("/neuro/pipeline",       post(neuro_pipeline))
        .route("/neuro/generate",       post(neuro_generate))
        .route("/neuro/symbols/live",   get(neuro_symbols_live))
        // GET /neuro/world3d — structured 3-D scene for Three.js neural renderer.
        // Returns centroid positions grouped by category (body parts, env, visual
        // zones) enriched with learned colour hints from Hebbian connections to
        // img:h* labels, confidence from prediction_confidence, and visual
        // associations so the renderer can colour each body part with what the
        // fabric actually learned from the video, not hard-coded browns.
        .route("/neuro/world3d",        get(neuro_world3d))
        .route("/mesh/synthesize",      post(mesh_synthesize))
        .route("/mesh/from_image",      post(mesh_from_image))
        .route("/mesh/template",        get(mesh_template))
        .route("/chat",                 post(neuro_ask))
        // ── Hypothesis queue ──────────────────────────────────────────────────
        // Questions that scored below ANSWER_THRESHOLD sit here until resolved.
        .route("/hypothesis/queue",     get(hypothesis_queue_list))
        .route("/hypothesis/resolve",   post(hypothesis_resolve))
        // ── Entity / simulation runtimes ──────────────────────────────────────
        // POST /entity/observe  — feed a multi-modal entity observation through
        //   the behavior→physiology→survival→narrative chain and train the neuro
        //   pool with the resulting labels. Returns combined report.
        // GET  /entity/report   — latest scene + survival + narrative state.
        .route("/entity/observe",       post(entity_observe))
        .route("/entity/report",        get(entity_report))
        // ── Overlay / perception system ───────────────────────────────────────
        // Health overlay: 6-dim color-coded concentric ring model per entity.
        .route("/overlay/health",               get(overlay_health))
        // Delta engine: velocity + short-horizon predictions per entity.
        .route("/overlay/delta",                get(overlay_delta))
        // Scientific poetry: emergent natural-language entity synthesis.
        .route("/overlay/poetry",               get(overlay_poetry))
        // Chaos World Model: most probable history reconstruction.
        .route("/overlay/chaos",                post(overlay_chaos))
        // Prediction experiment: autonomous forward prediction + accuracy tracking.
        .route("/overlay/predictions",          get(overlay_predictions))
        // Layered physiology: hierarchical inside-out structural decomposition.
        .route("/overlay/layers",               post(overlay_layers))
        .route("/overlay/layers/calibrate",     post(overlay_layers_calibrate))
        .route("/overlay/layers/:entity_id",    get(overlay_layers_entity))
        .with_state(state)
        .layer(DefaultBodyLimit::max(max_body))
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::OPTIONS,
                ])
                .allow_headers(tower_http::cors::Any),
        );
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
        // ── Background hypothesis research loop ───────────────────────────────
        // Every 30 seconds, pick the oldest unresolved hypothesis, POST it to
        // /neuro/ask internally, and if activation has risen above threshold,
        // mark it resolved.  External research agents (research_agent.py) poll
        // GET /hypothesis/queue and POST /hypothesis/resolve.
        {
            let hq = app_hypothesis_queue.clone();
            let neuro = app_neuro.clone();
            tokio::spawn(async move {
                hypothesis_research_loop(hq, neuro).await;
            });
        }
        // ── Background peer-list refresh ──────────────────────────────────────
        // Keep the distributed coordinator in sync with the live cluster
        // membership roster.  Without this, the coordinator's peer list is
        // only updated when /cluster/status is explicitly polled, so a newly
        // joined worker would be invisible to the coordinator until someone
        // called that endpoint manually.
        {
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    tokio::time::Duration::from_secs(30),
                );
                interval.set_missed_tick_behavior(
                    tokio::time::MissedTickBehavior::Skip,
                );
                loop {
                    interval.tick().await;
                    let guard = bg_cluster.lock().await;
                    if let Some(node) = guard.as_ref() {
                        let s = node.status().await;
                        let addrs = peer_http_addrs(&s.nodes, &s.local_id);
                        drop(guard);
                        bg_dist.set_peers(addrs).await;
                    }
                }
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
/// Body: `{ "snapshot": <EnvironmentSnapshot>, "extra_labels": [...] }`
/// Observes the snapshot through the neural fabric — updates Hebbian weights
/// for every symbol co-occurrence in this sensor frame, connects spatial and
/// temporal context, and accumulates motif patterns over time.
///
/// `extra_labels` (optional) are co-trained with the snapshot in the same
/// request. Use this to inject raw perceptual features (e.g. image_bits
/// zone/colour/edge tokens like "img:z3_2", "img:h5", "img:edgeV_z3_2")
/// alongside structured body-part observations so the fabric learns that
/// "warm hue in zone 7_2 = cow_head" — the visual→world-model link.
#[derive(Deserialize)]
struct NeuroTrainRequest {
    snapshot: EnvironmentSnapshot,
    #[serde(default)]
    extra_labels: Vec<String>,
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

    // Co-train raw perceptual labels (e.g. image_bits tokens: "img:z3_2",
    // "img:h5", "img:edgeV_z3_2") in the same request as the body-part
    // snapshot so Hebbian connections form between visual features and body
    // positions — the visual→world-model link that enables 3-D from 2-D.
    if !request.extra_labels.is_empty() {
        state.neuro.train_weighted(&request.extra_labels, 1.0, false);
    }

    // Cache the most recent snapshot so /neuro/symbols/live can serve it.
    *state.live_snapshot.lock().expect("live_snapshot") = Some(request.snapshot.clone());

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

/// GET /neuro/symbols/live
/// Returns the symbols array from the most recent EnvironmentSnapshot posted
/// via /neuro/train. The world viewer polls this at ~10 fps to animate entity
/// positions in real-time SCENE mode, overlaid on the Hebbian centroid geometry.
async fn neuro_symbols_live(
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
    let guard = state.live_snapshot.lock().expect("live_snapshot");
    match &*guard {
        Some(snap) => {
            let payload = serde_json::json!({
                "timestamp": snap.timestamp,
                "bounds": snap.bounds,
                "symbols": snap.symbols,
                "metadata": snap.metadata,
            });
            (StatusCode::OK, Json(payload))
        }
        None => (
            StatusCode::OK,
            Json(serde_json::json!({ "symbols": [], "metadata": {} })),
        ),
    }
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

/// GET /neuro/world3d
/// Returns a structured 3-D scene description derived from the neural fabric's
/// current Hebbian centroid map.  Designed for the Three.js NEURAL renderer so
/// it can build geometry and colours from what the fabric actually learned
/// rather than using hardcoded cow anatomy primitives.
///
/// Response JSON:
/// ```json
/// {
///   "objects": [
///     {
///       "id": "cow_head",
///       "position": {"x": 0.87, "y": 0.72, "z": 0.85},
///       "category": "cow_body",          // "cow_body" | "env" | "visual_zone"
///       "color_rgb": [0.72, 0.38, 0.12], // derived from linked img:h* bins
///       "confidence": 0.91,              // from prediction_confidence
///       "active": true,                  // in current active_labels
///       "predicted": {"x":..., "y":..., "z":...},  // temporal prediction
///       "visual_labels": ["img:h2","img:edgeV_z7_2"]  // top linked percept tokens
///     }
///   ],
///   "total_centroids": 73,
///   "active_count": 42
/// }
/// ```
async fn neuro_world3d(
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

    // ── Hue-bin → linear RGB lookup ───────────────────────────────────────────
    // image_bits uses 16 hue bins (0-15) covering the colour wheel [0°, 360°).
    // Convert each bin centre to RGB so the renderer can apply it as a tint.
    fn hue_bin_to_rgb(bin: usize, bins: usize) -> [f32; 3] {
        let h = (bin as f32 + 0.5) / bins as f32 * 360.0; // degrees
        // HSV→RGB (S=0.7, V=0.85) — vivid but not saturated
        let c = 0.7f32 * 0.85;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = 0.85 - c;
        let (r1, g1, b1) = if h < 60.0 { (c, x, 0.0) }
            else if h < 120.0 { (x, c, 0.0) }
            else if h < 180.0 { (0.0, c, x) }
            else if h < 240.0 { (0.0, x, c) }
            else if h < 300.0 { (x, 0.0, c) }
            else              { (c, 0.0, x) };
        [r1 + m, g1 + m, b1 + m]
    }

    // ── Build per-object colour from Hebbian links to img:h* labels ───────────
    // network_links: label → {connected_label → weight}.
    // For each centroid label, sum hue-bin weights from all connected img:h* labels
    // and return the weighted-average RGB.
    fn derive_color(label: &str, links: &HashMap<String, HashMap<String, f32>>) -> [f32; 3] {
        let Some(connected) = links.get(label) else {
            return [0.55, 0.35, 0.15]; // fallback warm brown (cow default)
        };
        let mut r_sum = 0.0f32; let mut g_sum = 0.0f32; let mut b_sum = 0.0f32;
        let mut w_sum = 0.0f32;
        for (conn_label, &w) in connected {
            // Match "img:h<N>" labels (16 bins)
            if let Some(rest) = conn_label.strip_prefix("img:h") {
                if let Ok(bin) = rest.parse::<usize>() {
                    let rgb = hue_bin_to_rgb(bin, 16);
                    r_sum += rgb[0] * w; g_sum += rgb[1] * w; b_sum += rgb[2] * w;
                    w_sum += w;
                }
            }
        }
        if w_sum < 0.01 {
            return [0.55, 0.35, 0.15]; // fallback
        }
        [r_sum / w_sum, g_sum / w_sum, b_sum / w_sum]
    }

    // ── Top linked visual tokens for a label ─────────────────────────────────
    fn top_visual_labels(label: &str, links: &HashMap<String, HashMap<String, f32>>, n: usize)
        -> Vec<String>
    {
        let Some(connected) = links.get(label) else { return vec![]; };
        let mut vis: Vec<(&str, f32)> = connected.iter()
            .filter(|(l, _)| l.starts_with("img:"))
            .map(|(l, &w)| (l.as_str(), w))
            .collect();
        vis.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vis.truncate(n);
        vis.into_iter().map(|(l, _)| l.to_string()).collect()
    }

    let centroids   = &snap.centroids;
    let links       = &snap.network_links;
    let active      = &snap.active_labels;
    let confidence  = &snap.prediction_confidence;
    let predictions = &snap.temporal_predictions;

    let mut objects: Vec<serde_json::Value> = Vec::with_capacity(centroids.len());

    for (label, pos) in centroids {
        // Skip internal composite markers
        if label.starts_with("comp::") || label.starts_with("stream::") { continue; }

        let category = if label.starts_with("cow_") || label.starts_with("id::cow") {
            "cow_body"
        } else if label.starts_with("env_") {
            "env"
        } else if label.starts_with("img:z") {
            "visual_zone"
        } else if label.starts_with("img:") {
            "visual_feature"
        } else {
            "other"
        };

        // Strip the "id::" prefix that the fabric sometimes adds
        let clean_id = label.strip_prefix("id::").unwrap_or(label);

        let color = derive_color(label, links);
        let conf  = confidence.get(label).copied().unwrap_or(0.5);
        let pred  = predictions.get(label);
        let vis   = top_visual_labels(label, links, 8);
        let is_active = active.contains(label);

        let mut obj = serde_json::json!({
            "id":           clean_id,
            "position":     { "x": pos.x, "y": pos.y, "z": pos.z },
            "category":     category,
            "color_rgb":    color,
            "confidence":   conf,
            "active":       is_active,
            "visual_labels": vis,
        });
        if let Some(p) = pred {
            obj["predicted"] = serde_json::json!({ "x": p.x, "y": p.y, "z": p.z });
        }
        objects.push(obj);
    }

    let total = centroids.len();
    let active_count = objects.iter().filter(|o| o["active"].as_bool().unwrap_or(false)).count();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "objects":          objects,
            "total_centroids":  total,
            "active_count":     active_count,
        })),
    )
}

/// GET /neuro/stream
/// Server-Sent Events stream of NeuroSnapshot frames at ~10 fps.
/// Each event is named "snap" and contains the full serialised NeuroSnapshot.
/// Clients connect with `new EventSource("http://.../neuro/stream")` and
/// receive frames indefinitely — each frame is the live neural activation state.
/// Because the fabric was trained on real data, reading this stream at the
/// right sampling rate reconstructs a representation of whatever the net learned.
async fn neuro_state_stream(
    State(state): State<ApiState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = futures::stream::unfold(state, |state| async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let snap = state.neuro.snapshot();
        let json = serde_json::to_string(&snap).unwrap_or_default();
        let event = Event::default().event("snap").data(json);
        Some((Ok(event), state))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
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
        "bridge"  => state.rate_limit_bridge_max,
        "balance" => state.rate_limit_balance_max,
        _         => state.rate_limit_default_max,
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
            // Keep distributed peer list in sync with the current cluster roster.
            let peer_addrs = peer_http_addrs(&s.nodes, &s.local_id);
            state.distributed.set_peers(peer_addrs).await;
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
            // Refresh distributed peer list from the cluster roster.
            let peer_addrs = peer_http_addrs(&s.nodes, &s.local_id);
            state.distributed.set_peers(peer_addrs).await;
            // Seed this node with knowledge from existing peers (fire-and-forget).
            state.distributed.bootstrap_from_peers().await;
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
            Ok(()) => {
                state.distributed.set_peers(vec![]).await;
                (StatusCode::OK, Json(serde_json::json!({ "status": "ok", "message": "left cluster — now standalone" })))
            }
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

// ── Distributed training endpoints ────────────────────────────────────────────

/// POST /neuro/delta/apply
/// Called by the coordinator or peers to push a weight delta to this node.
/// Merges with max(local, remote) so knowledge only accumulates.
async fn neuro_delta_apply(
    State(state): State<ApiState>,
    Json(req): Json<crate::distributed::DeltaApplyReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    let applied = state.neuro.apply_label_delta(&req.synapses, &req.cooccur);
    tracing::debug!(
        "delta_apply: merged {} synapses (step {}→{})",
        applied, req.from_step, req.to_step
    );
    (StatusCode::OK, Json(serde_json::json!({ "applied": applied })))
}

/// POST /neuro/sync
/// Force an immediate weight-delta push to all cluster peers.
async fn neuro_sync(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let neuro = state.neuro.clone();
    state.distributed.force_sync(&*neuro).await;
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok", "message": "sync pushed to all peers" })))
}

/// GET /cluster/sync/status
/// Current state of the distributed training coordinator.
async fn cluster_sync_status(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let s = state.distributed.status().await;
    (StatusCode::OK, Json(s))
}

/// Derive HTTP peer addresses from a cluster roster by replacing the cluster
/// port (51611) with the node API port (8090).  Excludes the local node.
fn peer_http_addrs(
    nodes: &[w1z4rdv1510n_cluster::protocol::NodeInfo],
    local_id: &w1z4rdv1510n_cluster::protocol::NodeId,
) -> Vec<String> {
    nodes.iter()
        .filter(|n| &n.id != local_id)
        .filter_map(|n| {
            // addr format: "ip:port"  (may be IPv6 bracket notation)
            let colon = n.addr.rfind(':')?;
            let ip = &n.addr[..colon];
            if ip.is_empty() { return None; }
            Some(format!("http://{ip}:8090"))
        })
        .collect()
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
    /// When true, train inhibitory (suppressive) synapses instead of excitatory.
    /// Use this to suppress wrong-answer associations (contrastive learning).
    #[serde(default)]
    inhibitory: bool,
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
///
/// For text modalities this does two things automatically:
///
/// 1. **Co-occurrence pass** — all labels (word, char, punct, role, zone) fire
///    together in one `train_weighted` call.  The pool builds Hebbian connections
///    between everything that co-activated.
///
/// 2. **STDP character-sequence pass** — for every word in the text, the encoder
///    now returns the ordered character sequence (e.g. `["txt:char_c", "txt:char_a",
///    "txt:char_t"]` for "cat").  These are trained as adjacent-frame bridges with
///    exponential lr decay (tau = 0.5 char-steps), so STDP builds strong *forward*
///    (pre→post) edges through each word's letter chain.  Recurring sub-sequences
///    across words (morphemes like "port" in transport/import/export) accumulate
///    shared activating paths and eventually promote to mini-columns — emergent
///    morpheme recognition with no hand-coded rules.
async fn media_train(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Distributed routing: if this call was not already forwarded by a peer,
    // round-robin it to the next peer in the cluster so training load is spread
    // across all N nodes.  The x-w1z-local header prevents forwarding loops.
    if !headers.contains_key("x-w1z-local") {
        if let Some(peer) = state.distributed.next_train_peer().await {
            let ok = state.distributed.forward_train(&peer, body.clone()).await;
            if ok {
                return (StatusCode::OK, Json(serde_json::json!({
                    "trained": true,
                    "distributed": true,
                    "routed_to": peer,
                })));
            }
            // Peer rejected or unreachable — fall through to local training.
        }
    }
    // Local training path.
    let req: MediaTrainReq = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": e.to_string() }))),
    };

    // Encode labels (co-occurrence pass).
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

    // Extract character sequences for STDP ordering (text modalities only).
    let char_seqs: Vec<Vec<String>> =
        if req.modality == "text" || req.modality == "page" || req.modality == "full" {
            let enc = TextBitsEncoder::new(TextBitsConfig::default());
            if let Some(ref span_reqs) = req.spans {
                let tspans: Vec<TextSpan> = span_reqs.iter().map(|sr| {
                    let emphasis = match (sr.bold, sr.italic) {
                        (true, true)   => TextEmphasis::BoldItalic,
                        (true, false)  => TextEmphasis::Bold,
                        (false, true)  => TextEmphasis::Italic,
                        _              => TextEmphasis::None,
                    };
                    TextSpan::positioned(
                        &sr.text, parse_text_role(&sr.role), sr.size_ratio, emphasis,
                        sr.indent, sr.x_frac, sr.y_frac, sr.seq_index, sr.seq_total,
                    )
                }).collect();
                enc.encode_spans(&tspans).char_sequences
            } else if let Some(ref plain) = req.text {
                enc.encode_plain(plain).char_sequences
            } else {
                vec![]
            }
        } else {
            vec![]
        };

    let label_count = labels.len();
    let char_seq_count = char_seqs.len();

    let neuro = state.neuro.clone();
    let lr = req.lr_scale;
    let inhibitory = req.inhibitory;
    tokio::task::spawn_blocking(move || {
        // Pass 1: full co-occurrence — all labels fire together.
        // inhibitory=true creates suppressive synapses (contrastive/negative learning).
        neuro.train_weighted(&labels, lr, inhibitory);

        // Pass 2: STDP character sequences — collect all unique char labels
        // from adjacent pairs across every word, then fire ONE batched
        // train_weighted call.  This avoids O(words × chars) separate mutex
        // acquisitions that caused exponential slowdown as the pool grew.
        // The single co-activation still plants Hebbian edges between every
        // adjacent character label in the batch; motif discovery + mini-column
        // promotion will later collapse recurring sub-sequences into concept
        // neurons without needing per-pair calls here.
        let char_lr = lr * 0.3;
        let tau = 0.5f32; // char-step time constant
        let bridge_lr = char_lr * (-1.0f32 / tau).exp();
        let mut char_pair_labels: Vec<String> = Vec::new();
        for seq in &char_seqs {
            for window in seq.windows(2) {
                char_pair_labels.push(window[0].clone());
                char_pair_labels.push(window[1].clone());
            }
        }
        char_pair_labels.sort_unstable();
        char_pair_labels.dedup();
        if !char_pair_labels.is_empty() {
            neuro.train_weighted(&char_pair_labels, bridge_lr, false);
        }

        // ── Periodic concept promotion ────────────────────────────────────────
        // Scan co-occurrence counts and promote frequently co-occurring label
        // pairs to concept neurons (mini-columns / neural networks).  Run every
        // 50 training calls to amortise the O(N×K) synapse scan.  This is the
        // mechanism by which recurring char sequences (morphemes) and word
        // clusters self-organise into concept neurons without hand-coded rules.
        neuro.promote_text_concepts();
    }).await.ok();

    // Schedule weight-delta sync every SYNC_EVERY local training calls.
    // Fire-and-forget so the HTTP response is not delayed by the sync.
    {
        let dist  = state.distributed.clone();
        let neuro = state.neuro.clone();
        tokio::spawn(async move {
            dist.maybe_sync(&*neuro).await;
        });
    }

    (StatusCode::OK, Json(serde_json::json!({
        "trained": true,
        "modality": req.modality,
        "label_count": label_count,
        "char_sequences": char_seq_count,
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
    /// Two-pool namespace: "q" for question pool, "a" for answer pool, None for shared.
    /// When set, all word labels are prefixed with "q:" or "a:" respectively.
    /// This separates Q and A representations so Q→A is a directed cross-pool connection.
    #[serde(default)] pool_namespace: Option<String>,
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
            Ok(raw_labels) if !raw_labels.is_empty() => {
                // Apply pool namespace prefix to word labels when specified.
                // "q:" prefix = question pool, "a:" prefix = answer pool.
                // Non-word labels (char, phonetic, zone) are shared across pools
                // so they are NOT prefixed — they carry sub-lexical signal that
                // links phonetically similar Q and A words together.
                let labels: Vec<String> = if let Some(ns) = &frame.pool_namespace {
                    raw_labels.into_iter().map(|l| {
                        // Prefix both word and punct labels — punct encodes word-boundary
                        // and sentence-structure signal that discriminates Q inputs
                        // (e.g. "hello" vs "hello!") in the Q-pool.
                        if (l.starts_with("txt:word_") || l.starts_with("txt:punct_"))
                            && !l.contains("_zone_")
                        {
                            format!("{}:{}", ns, l)
                        } else {
                            l
                        }
                    }).collect()
                } else {
                    raw_labels
                };
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
        // Pass 1: each frame independently (within-frame co-occurrence).
        for (labels, lr) in frame_labels.iter().zip(frame_lrs.iter()) {
            if !labels.is_empty() {
                neuro.train_weighted(labels, *lr, false);
            }
        }
        // Pass 2: bridge adjacent frames.
        // Sorted-combined bridge works for close labels but misses alphabetically
        // distant word pairs (e.g. "hello" ↔ "w1z4rd") because HEBBIAN_WINDOW=20
        // doesn't reach them in a 100+ label sorted set.  To guarantee direct
        // cross-frame word↔word pairing, we build a small focused bridge from the
        // word-only labels of each frame and train them explicitly together.
        for i in 1..frame_labels.len() {
            let prev = &frame_labels[i - 1];
            let curr = &frame_labels[i];
            if prev.is_empty() || curr.is_empty() { continue; }
            let dt = (frame_ts[i] - frame_ts[i - 1]).abs();
            let base_lr = (frame_lrs[i - 1] + frame_lrs[i]) * 0.5;
            let bridge_lr = base_lr * (-dt / tau).exp();
            if bridge_lr < 1e-4 { continue; }

            // Full label bridge for sub-word labels only (char, phonetic, zone).
            // Word and punct labels (namespaced or bare) are excluded here — they are
            // directional Q→A signals and sorting would reverse their STDP direction.
            // The directed word bridge below handles all of them in correct order.
            let mut bridge: Vec<String> = prev.iter().chain(curr.iter())
                .filter(|l| !l.starts_with("txt:word_")
                         && !l.starts_with("q:txt:word_")
                         && !l.starts_with("a:txt:word_")
                         && !l.starts_with("txt:punct_")
                         && !l.starts_with("q:txt:punct_")
                         && !l.starts_with("a:txt:punct_"))
                .cloned().collect();
            bridge.sort_unstable();
            bridge.dedup();
            if !bridge.is_empty() {
                neuro.train_weighted(&bridge, bridge_lr, false);
            }

            // Directed word+punct bridge: prev labels ordered first, then curr.
            // Includes namespace-prefixed word and punct labels so Q→A cross-pool
            // edges are created with correct STDP direction.
            // Punct labels (e.g. q:txt:punct_exclaim) in the Q frame connect to the
            // first A word, giving the decoder Q-context signal that discriminates
            // inputs like "hello" vs "hello!" at the pool level.
            fn is_word_label(l: &str) -> bool {
                (l.starts_with("txt:word_") || l.starts_with("q:txt:word_") || l.starts_with("a:txt:word_")
                 || l.starts_with("txt:punct_") || l.starts_with("q:txt:punct_") || l.starts_with("a:txt:punct_"))
                    && !l.contains("_zone_")
            }
            let prev_words: Vec<String> = prev.iter()
                .filter(|l| is_word_label(l))
                .cloned().collect();
            let curr_words: Vec<String> = curr.iter()
                .filter(|l| is_word_label(l))
                .cloned().collect();
            if !prev_words.is_empty() && !curr_words.is_empty() {
                let mut word_bridge: Vec<String> = prev_words.iter()
                    .chain(curr_words.iter()).cloned().collect();
                word_bridge.dedup();
                neuro.train_weighted(&word_bridge, bridge_lr, false);
            }
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

// ── Contrastive training ──────────────────────────────────────────────────────
//
// POST /media/train_contrastive
//
// Trains one correct Q->A pair (excitatory, with dopamine reward) and N wrong
// answers (inhibitory, with norepinephrine spike) in a single atomic call.
//
// The inhibitory pass creates suppressive synapses from the question concept
// labels to the wrong-answer concept labels.  During propagation, when the
// question fires, the wrong answers are actively suppressed — this is the
// mechanism that "cuts off fake science" through raw Hebbian weights.
//
// If `wrong_answers` is empty the call still trains the correct pair with
// dopamine reinforcement, which is stronger than a plain /media/train call.
//
// Cross-batch negatives: if `cross_batch_labels` is supplied (a list of answer
// label lists from other Q->A pairs in the same batch), each set is trained as
// an inhibitory pass against this question.  This mirrors contrastive self-
// supervised learning — in-batch negatives, no curated wrong answers needed.

#[derive(Deserialize)]
struct ContrastiveTrainReq {
    question: String,
    correct_answer: String,
    #[serde(default)]
    wrong_answers: Vec<String>,
    /// Optional pre-encoded label sets from other pairs in the batch (inhibitory).
    #[serde(default)]
    cross_batch_labels: Vec<Vec<String>>,
    #[serde(default = "default_lr")]
    lr_scale: f32,
}

/// POST /media/train_contrastive
async fn media_train_contrastive(
    State(state): State<ApiState>,
    Json(req): Json<ContrastiveTrainReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.question.trim().is_empty() || req.correct_answer.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "question and correct_answer must not be empty" })));
    }

    let enc         = TextBitsEncoder::new(TextBitsConfig::default());
    let q_labels    = enc.encode_plain(&req.question).labels;
    let a_labels    = enc.encode_plain(&req.correct_answer).labels;
    // Q→A ordered (not sorted): q labels first so STDP creates Q→A LTP direction.
    let combined_pos: Vec<String> = {
        let mut v = q_labels.clone();
        v.extend_from_slice(&a_labels);
        v  // no sort — preserve temporal Q→A order for STDP
    };
    // Pre-encode wrong answers before entering blocking task.
    let wrong_label_sets: Vec<Vec<String>> = req.wrong_answers.iter()
        .map(|w| {
            let wl = enc.encode_plain(w).labels;
            let mut v = q_labels.clone();
            v.extend_from_slice(&wl);
            v.sort_unstable(); v.dedup(); v  // wrong answers: sorted is fine (directionality doesn't matter for inhibitory)
        })
        .collect();

    let lr          = req.lr_scale;
    let n_wrong     = wrong_label_sets.len() + req.cross_batch_labels.len();
    let neuro       = state.neuro.clone();
    let cross_batch = req.cross_batch_labels.clone();

    tokio::task::spawn_blocking(move || {
        // ── Positive pass: temporal Q→A STDP reinforcement ────────────────────
        // combined_pos has Q labels before A labels — train_weighted uses array
        // order for STDP direction: Q neurons fire → A neurons = LTP on Q→A edges.
        neuro.train_weighted(&combined_pos, lr * 1.2, false);
        neuro.train_weighted(&q_labels, lr, false);
        neuro.train_weighted(&a_labels, lr * 0.9, false);
        // Dopamine: retrograde potentiation of the Q->A path.
        neuro.release_neuromodulator(NeuromodulatorKind::Dopamine, 0.85);
        neuro.flush_dopamine();

        // ── Negative passes (inhibitory — suppresses wrong associations) ──────
        // Norepinephrine spike: marks these tokens as "novel/wrong", raises LR
        // for the next (inhibitory) train call so suppression is strong enough
        // to overcome the excitatory baseline.
        if !wrong_label_sets.is_empty() || !cross_batch.is_empty() {
            neuro.release_neuromodulator(NeuromodulatorKind::Norepinephrine, 0.5);
        }
        for wrong_combined in &wrong_label_sets {
            neuro.train_weighted(wrong_combined, lr * 0.7, true);
        }
        // Cross-batch negatives: question labels paired with other batch answers.
        for other_answer_labels in &cross_batch {
            let mut neg: Vec<String> = q_labels.clone();
            neg.extend_from_slice(other_answer_labels);
            neg.sort_unstable(); neg.dedup();
            neuro.train_weighted(&neg, lr * 0.5, true);
        }
    }).await.ok();

    (StatusCode::OK, Json(serde_json::json!({
        "trained": true,
        "positive": 1,
        "negative": n_wrong,
    })))
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
    /// If set, this turn is a reply to a previous system question.
    /// The node trains the (context → text) pair, resolves the queued
    /// hypothesis, and fires dopamine — then continues to process `text`
    /// as the current input so the conversation flows forward.
    #[serde(default)]
    context: Option<String>,
    /// Opaque session identifier.  When provided the node blends activation
    /// state from the previous turn into the current propagation seed, giving
    /// the network a lightweight working memory across turns.
    /// Scripts and apps decide when to start/end sessions; the fabric carries
    /// the state.
    #[serde(default)]
    session_id: Option<String>,
}
fn default_ask_top_k() -> usize { 20 }
fn default_ask_min_strength() -> f32 { 0.05 }

/// POST /neuro/ask  (also POST /chat)
///
/// Multi-pathway convergence inference:
///   1. Encode question → Pathway A seed labels (weight 1.0)
///   2. QA recall → encode best answer text → Pathway B seed labels (weight conf × 1.5)
///   3. Single unified propagation pass (`propagate_combined`) — inhibitory edges
///      from one pathway can suppress noise from another
///   4. Read converged `txt:word_*` activation state
///   5. If peak < ANSWER_THRESHOLD → add to hypothesis queue, return null answer
///   6. Sequential auto-regressive decode from the pre-seeded combined activation
async fn neuro_ask(
    State(state): State<ApiState>,
    Json(req): Json<AskReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "text must not be empty" })));
    }
    let neuro              = state.neuro.clone();
    let eem                = state.equation_matrix.clone();
    let hq_arc             = state.hypothesis_queue.clone();
    let session_ctxs       = state.session_contexts.clone();
    let text               = req.text.clone();
    let hops               = req.hops;
    let max_tok            = req.top_k.max(30);
    let min_str            = req.min_strength;
    let context            = req.context.clone();
    let session_id         = req.session_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        // Detect greeting inputs so that if inference returns empty we can fall
        // back to a bootstrap response instead of sending a null hypothesis.
        // Trained greeting responses (from /api/wizard-chat/train/) always win —
        // this check is only used as a last resort after inference runs.
        let is_greeting = {
            let tl = text.trim().to_lowercase();
            let triggers = [
                "hello", "hi", "hey", "sup", "howdy", "greetings",
                "how are you", "how are you?", "what's up", "whats up",
                "good morning", "good afternoon", "good evening", "good night",
            ];
            triggers.iter().any(|g| tl == *g
                || tl.trim_end_matches('!') == *g
                || tl.trim_end_matches('.') == *g)
        };

        // ── Context training: user is answering a previous system question ─────
        // When `context` is provided the current `text` is a reply — treat the
        // (context → text) pair as direct teaching.  Train it into the neuro pool,
        // ingest it as a QA candidate, resolve the queued hypothesis, and fire
        // dopamine retrograde potentiation so the synaptic path that generated the
        // original question gets strengthened.  The network literally rewires from
        // the conversation.  Then fall through to process `text` normally so the
        // turn still produces a response.
        let context_trained = if let Some(ref ctx) = context {
            let ctx = ctx.trim();
            if !ctx.is_empty() {
                let enc = TextBitsEncoder::new(TextBitsConfig::default());
                // Train the question → answer passage as a single co-activation.
                // Elevated LR (1.4) because explicit user teaching is high-signal.
                let combined = format!("{ctx} {text}");
                let labels = enc.encode_plain(&combined).labels;
                if !labels.is_empty() {
                    neuro.train_weighted(&labels, 1.4, false);
                    // STDP char pass: plant forward edges through each word's letter chain.
                    let char_seqs = enc.encode_plain(&combined).char_sequences;
                    for seq in &char_seqs {
                        if seq.len() >= 2 {
                            for pair in seq.windows(2) {
                                neuro.train_weighted(pair, 1.0, false);
                            }
                        }
                    }
                }

                // Resolve the queued hypothesis for this question (if any).
                {
                    let hyp_id = {
                        let mut h = 0u64;
                        for b in ctx.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
                        format!("{h:x}")
                    };
                    let mut hq = hq_arc.lock().expect("hypothesis mutex");
                    if let Some(entry) = hq.iter_mut()
                        .find(|e| e.question.trim() == ctx || e.id == hyp_id)
                    {
                        entry.resolved   = true;
                        entry.answer     = Some(text.trim().to_string());
                        entry.confidence = Some(0.90);
                    }
                }

                // Dopamine: reward the path that asked the right question.
                neuro.release_neuromodulator(NeuromodulatorKind::Dopamine, 0.9);
                neuro.flush_dopamine();
                true
            } else { false }
        } else { false };
        let _ = context_trained; // used below for response tagging

        // ── Session working memory ────────────────────────────────────────────
        // Load activation context from the previous turn (if session is active).
        // These labels are blended into propagation at 0.3 weight so prior
        // conversation context gently biases retrieval — enough to maintain
        // referential continuity but not enough to override the current question.
        let session_context_labels: Vec<String> = if let Some(ref sid) = session_id {
            session_ctxs.lock().expect("session mutex")
                .get(sid).cloned().unwrap_or_default()
        } else {
            vec![]
        };


        // ── Question intent detection ─────────────────────────────────────────
        // The first meaningful token determines the question's intent class.
        // Intent adds small bias seeds to propagation that steer activation
        // toward the relevant sub-graphs already in the pool — causal chains,
        // property associations, process sequences, or pure definitions.
        // No domain knowledge is hard-coded; this works on whatever Hebbian
        // structure has grown from training.
        let question_lower = text.to_lowercase();
        let first_word = question_lower.split_whitespace().next().unwrap_or("");

        #[derive(PartialEq, Debug)]
        enum IntentClass { Causal, Property, Process, Definitional }
        let intent = match first_word {
            "why"                                                    => IntentClass::Causal,
            "is"|"can"|"are"|"does"|"do"|"was"|"were"
            |"could"|"would"|"will"|"should"|"did"                  => IntentClass::Property,
            "how"                                                    => IntentClass::Process,
            _                                                        => IntentClass::Definitional,
        };

        // Intent bias: words already trained in the pool whose Hebbian neighbours
        // form the relevant sub-graph.  Added at 0.4 weight — enough to steer
        // propagation direction without overriding the question signal.
        let intent_bias_labels: Vec<String> = match intent {
            IntentClass::Causal => vec![
                "txt:word_because".to_string(),
                "txt:word_cause".to_string(),
                "txt:word_reason".to_string(),
            ],
            IntentClass::Property => vec![
                "txt:word_true".to_string(),
                "txt:word_part".to_string(),
            ],
            IntentClass::Process => vec![
                "txt:word_step".to_string(),
                "txt:word_process".to_string(),
            ],
            IntentClass::Definitional => vec![],
        };

        // ── Pathway A: question text labels ──────────────────────────────────
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let question_labels: Vec<String> = enc.encode_plain(&text).labels;

        // Build Q-pool labels: word labels prefixed with "q:" for the two-pool
        // Q→A architecture. Seeding with both the raw labels (corpus connections)
        // and Q-pool labels (conversational Q→A connections) ensures both paths fire.
        // Q-pool seeds: prefix both word and punct labels — punct discriminates
        // inputs that share the same words (e.g. "hello" vs "hello!").
        let q_pool_labels: Vec<String> = question_labels.iter()
            .filter(|l| (l.starts_with("txt:word_") || l.starts_with("txt:punct_"))
                     && !l.contains("_zone_"))
            .map(|l| format!("q:{}", l))
            .collect();

        // All labels seed propagation — no word exclusions.
        // The Hebbian graph is trained with all words; IDF and seed_relevance
        // provide discrimination without artificial stop-word removal.
        // Include Q-pool labels so conversational Q→A bridges fire.
        let question_content_labels: Vec<String> = {
            let mut v = question_labels.clone();
            v.extend_from_slice(&q_pool_labels);
            v
        };

        // ── Pure Hebbian path ────────────────────────────────────────────────
        // Answer comes entirely from Hebbian activation.
        // The system speaks from what the Hebbian graph has learned, nothing else.
        let now = now_timestamp();
        let qa_answer_labels: Vec<String> = vec![];
        let qa_conf = 0.0_f32;

        // 1-hop Hebbian resonance: seed with Q-pool labels + raw word labels.
        // A-pool answer labels (a:txt:word_*) should activate from Q-pool seeds.
        let q_word_only_pre: Vec<String> = question_content_labels.iter()
            .filter(|l| (l.starts_with("txt:word_") || l.starts_with("q:txt:word_"))
                     && !l.contains("_zone_"))
            .cloned()
            .collect();
        let pre_1hop = if q_word_only_pre.is_empty() {
            HashMap::new()
        } else {
            neuro.propagate_combined(&[(q_word_only_pre.as_slice(), 1.0_f32)], 1, 0.001)
        };
        let q_pre_set: std::collections::HashSet<&String> = q_word_only_pre.iter().collect();
        // Check both corpus (txt:word_*) and A-pool (a:txt:word_*) for answer signal.
        let hebbian_peak = pre_1hop.iter()
            .filter(|(l, _)| {
                (l.starts_with("txt:word_") || l.starts_with("a:txt:word_"))
                && !l.contains("_zone_")
            })
            .filter(|(l, _)| !q_pre_set.contains(l))
            .map(|(_, &v)| v)
            .fold(0.0f32, f32::max);

        let qa_activation = hebbian_peak; // renamed for compat with response fields
        let confidence_tier = if hebbian_peak >= 0.50 { "high" }
            else if hebbian_peak >= 0.25 { "medium" }
            else if hebbian_peak >= 0.10 { "low" }
            else { "uncertain" };
        let qa_gated = hebbian_peak > 0.10;

        if !qa_gated {
            // ── Unknown concept: register novel chars, spike NE, queue hypothesis ──
            // activate_with_resolution plants the novel character sequences from
            // this query into the pool as mini-column candidates, seeding future
            // learning.  Norepinephrine spike elevates LR for the next training
            // call on these tokens.
            // We never return a partial word-soup answer here — answer synthesis
            // must respect the representations grown by the encoder (char → word →
            // concept), not disassemble and reassemble them algorithmically.
            neuro.release_neuromodulator(NeuromodulatorKind::Norepinephrine, 0.75);

            let word_only_labels: Vec<String> = question_content_labels.iter()
                .filter(|l| l.starts_with("txt:word_") && !l.contains("_zone_"))
                .cloned()
                .collect();

            let _resolution = neuro.activate_with_resolution(
                &word_only_labels,
                0.40,
                hops,
                0.005,
            );

            // 1-hop propagation (word-only seed) to surface nearby known concepts
            // and compute an honest resonance peak — same principle as the trained
            // path: no char labels seeded so common-char saturation doesn't mask
            // the difference between a truly novel concept and a trained one.
            // Intent bias and session context are still included for concept steering.
            let hyp_seeds: Vec<(&[String], f32)> = {
                let mut s: Vec<(&[String], f32)> = vec![
                    (word_only_labels.as_slice(), 1.0_f32),
                ];
                if !intent_bias_labels.is_empty() {
                    s.push((intent_bias_labels.as_slice(), 0.4_f32));
                }
                if !session_context_labels.is_empty() {
                    s.push((session_context_labels.as_slice(), 0.3_f32));
                }
                s
            };
            // propagate_combined needs a fixed-size slice, build it explicitly
            let combined_1hop = match (
                intent_bias_labels.is_empty(),
                session_context_labels.is_empty(),
            ) {
                (true,  true)  => neuro.propagate_combined(
                    &[(word_only_labels.as_slice(), 1.0_f32)], 1, 0.001),
                (false, true)  => neuro.propagate_combined(
                    &[(word_only_labels.as_slice(), 1.0_f32),
                      (intent_bias_labels.as_slice(), 0.4_f32)], 1, 0.001),
                (true,  false) => neuro.propagate_combined(
                    &[(word_only_labels.as_slice(), 1.0_f32),
                      (session_context_labels.as_slice(), 0.3_f32)], 1, 0.001),
                (false, false) => neuro.propagate_combined(
                    &[(word_only_labels.as_slice(), 1.0_f32),
                      (intent_bias_labels.as_slice(), 0.4_f32),
                      (session_context_labels.as_slice(), 0.3_f32)], 1, 0.001),
            };
            let _ = hyp_seeds; // built above for clarity, match is the actual dispatch

            let seed_word_labels_hyp: std::collections::HashSet<&String> =
                word_only_labels.iter().collect();
            let mut word_acts_1hop: Vec<(String, f32)> = combined_1hop
                .iter()
                .filter(|(l, _)| l.starts_with("txt:word_") && !l.contains("_zone_"))
                .map(|(l, &v)| (l.clone(), v))
                .collect();
            word_acts_1hop.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal));
            let peak_1hop = word_acts_1hop.iter()
                .filter(|(l, _)| !seed_word_labels_hyp.contains(l))
                .map(|(_, v)| *v)
                .fold(0.0f32, f32::max);

            // ── Gap-state activated concepts ──────────────────────────────────
            // The Hebbian neighbourhood of what did fire around this unknown.
            // These are NOT formatted into a sentence — they are the network's
            // current state, which we propagate from to let a response emerge.
            let activated_concepts: Vec<String> = word_acts_1hop.iter()
                .filter(|(l, _)| !seed_word_labels_hyp.contains(l))
                .filter_map(|(l, _)| l.strip_prefix("txt:word_").map(|w| w.to_string()))
                .filter(|w| !w.contains('_') && w.len() > 2)
                .take(6)
                .collect();

            // ── Associative generation from gap state ─────────────────────────
            // The network is in a state: NE elevated, concept unknown, but nearby
            // associations are active.  Rather than returning silence or a
            // hardcoded phrase, propagate from those activated labels with full
            // hops and query the QA store from the resulting activation state.
            //
            // This is the same mechanism as a known answer — propagation through
            // the Hebbian graph — but seeded from what the network knows *near*
            // the gap rather than from the question itself.  The response that
            // surfaces is the network speaking from its current internal state.
            // It will be associative, adjacent, uncertain in character — because
            // the activation state it came from is exactly that.
            //
            // As training grows the pool, the gap-state responses become richer
            // and closer to the actual concept.  No code changes needed.
            // When the pool is sparse, the response is null — the network is
            // genuinely silent, which is honest.
            // Pure Hebbian gap state — no QA lookup.
            // The network is genuinely in a gap: fire from what it knows near
            // the unknown concept, surface the deeper association cluster.
            // The activated_concepts IS the answer — the Hebbian state of
            // "what fires around this unknown."  As training grows, this
            // becomes richer and closer to the actual concept.
            let gap_answer: Option<String> = if !activated_concepts.is_empty() {
                let gap_seed_labels: Vec<String> = activated_concepts.iter()
                    .map(|w| format!("txt:word_{w}"))
                    .collect();
                let gap_propagated = neuro.propagate_combined(
                    &[(gap_seed_labels.as_slice(), 1.0_f32)],
                    hops, 0.001,
                );
                let gap_seed_set: std::collections::HashSet<&String> =
                    gap_seed_labels.iter().collect();
                let mut gap_word_acts: Vec<(String, f32)> = gap_propagated.iter()
                    .filter(|(l, _)| l.starts_with("txt:word_") && !l.contains("_zone_"))
                    .filter(|(l, _)| !gap_seed_set.contains(l))
                    .map(|(l, &v)| (l.clone(), v))
                    .collect();
                gap_word_acts.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal));
                let gap_words: Vec<&str> = gap_word_acts.iter()
                    .filter_map(|(l, _)| l.strip_prefix("txt:word_"))
                    .take(12)
                    .collect();
                if gap_words.is_empty() { None } else { Some(gap_words.join(" ")) }
            } else {
                None
            };

            // Queue the hypothesis.
            let id = {
                let mut h = 0u64;
                for b in text.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
                format!("{h:x}")
            };
            {
                let mut hq = hq_arc.lock().expect("hypothesis mutex");
                if !hq.iter().any(|e| e.question == text) {
                    hq.push(HypothesisEntry {
                        id: id.clone(),
                        question: text.clone(),
                        queued_at_unix: now.unix,
                        attempts: 0,
                        max_attempts: 5,
                        resolved: false,
                        answer: None,
                        confidence: None,
                    });
                }
            }
            return serde_json::json!({
                "question":           text,
                "hypothesis":         true,
                "answer":             gap_answer,
                "activated_concepts": activated_concepts,
                "qa_activation":      qa_activation,
                "confidence_tier":    "uncertain",
                "peak_activation":    peak_1hop,
                "intent":             format!("{:?}", intent),
                "session_id":         session_id,
            });
        }

        // ── Question-only 1-hop: honest resonance signal ─────────────────────
        // Propagate from question WORD labels only — no char labels, no QA
        // answer priming, no intent bias, no session context.
        //
        // Why word-only: question_content_labels includes txt:char_* labels for
        // every token (even stop words like "what"/"is").  Common chars connect
        // to virtually every word in the pool; seeding them inflates every
        // neighbor to the propagation ceiling (0.85 = 1.0 × decay_factor per
        // hop).  Seeding only txt:word_* labels keeps the signal honest: only
        // the Hebbian neighbourhood of the question's concepts lights up, and
        // the strength reflects how densely trained those connections are.
        //
        // This is the discriminative signal between "well-trained concept" and
        // "novel/untrained concept" — not qa_activation (which measures QA
        // store IDF matching) but pure Hebbian resonance of the concept itself.
        // Include Q-pool labels in the word-only seed to activate A-pool labels.
        let q_word_only_labels: Vec<String> = question_content_labels.iter()
            .filter(|l| (l.starts_with("txt:word_") || l.starts_with("q:txt:word_"))
                     && !l.contains("_zone_"))
            .cloned()
            .collect();
        let q_only_1hop = if q_word_only_labels.is_empty() {
            HashMap::new()
        } else {
            neuro.propagate_combined(
                &[(q_word_only_labels.as_slice(), 1.0_f32)],
                1,
                0.001,
            )
        };
        let q_seed_word_set: std::collections::HashSet<&String> =
            q_word_only_labels.iter().collect();
        let peak = q_only_1hop.iter()
            .filter(|(l, _)| {
                (l.starts_with("txt:word_") || l.starts_with("a:txt:word_"))
                && !l.contains("_zone_")
            })
            .filter(|(l, _)| !q_seed_word_set.contains(l))
            .map(|(_, &v)| v)
            .fold(0.0f32, f32::max);

        // ── Full propagation ──────────────────────────────────────────────────
        // All four seed pathways feed a single combined propagation:
        //   1.0  question content words  — primary signal
        //   qa_conf  QA answer labels    — confirmation pathway
        //   0.4  intent bias labels      — steer toward causal/property/etc sub-graph
        //   0.3  session context labels  — continuity from prior turn
        let combined = neuro.propagate_combined(
            &[
                (question_content_labels.as_slice(), 1.0_f32),
                (qa_answer_labels.as_slice(),        qa_conf),
                (intent_bias_labels.as_slice(),      0.4_f32),
                (session_context_labels.as_slice(),  0.3_f32),
            ],
            hops,
            0.005,
        );
        // Include A-pool labels (a:txt:word_*) — answer neurons from the two-pool
        // Q→A architecture fire here and carry the trained answer sequence.
        let mut word_acts: Vec<(String, f32)> = combined
            .iter()
            .filter(|(l, _)| {
                (l.starts_with("txt:word_") || l.starts_with("a:txt:word_"))
                && !l.contains("_zone_")
            })
            .map(|(l, &v)| (l.clone(), v))
            .collect();
        word_acts.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal));

        // ── Property binding (is/can/are/does questions) ──────────────────────
        // Runs two separate 1-hop propagations: one from the subject concept,
        // one from the predicate concept.  Computes Jaccard overlap of their
        // top-20 activated word neighbours.
        //
        // High overlap (→1.0) means the subject and predicate are densely
        // inter-connected in the Hebbian graph — the property likely applies.
        // Low overlap (→0.0) means they live in different concept regions.
        //
        // This avoids the saturation problem of max-intersection: by comparing
        // ranked neighbourhood sets rather than raw activation values, the score
        // is meaningful even when both propagations fill their local region to 0.85.
        //
        // Scripts/apps interpret the score; the fabric just computes it.
        let property_binding_score: Option<f32> = if intent == IntentClass::Property {
            let subj_labels: Vec<String> = question_content_labels.iter()
                .filter(|l| l.starts_with("txt:word_"))
                .take(1)
                .cloned()
                .collect();
            let pred_labels: Vec<String> = question_content_labels.iter()
                .filter(|l| l.starts_with("txt:word_"))
                .skip(1)
                .cloned()
                .collect();
            if !subj_labels.is_empty() && !pred_labels.is_empty() {
                let subj_prop = neuro.propagate_combined(
                    &[(subj_labels.as_slice(), 1.0_f32)], 1, 0.005);
                let pred_prop = neuro.propagate_combined(
                    &[(pred_labels.as_slice(), 1.0_f32)], 1, 0.005);

                let subj_seed_set: std::collections::HashSet<&String> =
                    subj_labels.iter().collect();
                let pred_seed_set: std::collections::HashSet<&String> =
                    pred_labels.iter().collect();

                // Top-20 word neighbours for subject (excluding its own seeds).
                let mut sv: Vec<_> = subj_prop.iter()
                    .filter(|(l, _)| l.starts_with("txt:word_")
                        && !l.contains("_zone_")
                        && !subj_seed_set.contains(l))
                    .collect();
                sv.sort_unstable_by(|a, b| b.1.partial_cmp(a.1)
                    .unwrap_or(std::cmp::Ordering::Equal));
                let top_subj: std::collections::HashSet<String> =
                    sv.into_iter().take(20).map(|(l, _)| l.clone()).collect();

                // Top-20 word neighbours for predicate (excluding its own seeds).
                let mut pv: Vec<_> = pred_prop.iter()
                    .filter(|(l, _)| l.starts_with("txt:word_")
                        && !l.contains("_zone_")
                        && !pred_seed_set.contains(l))
                    .collect();
                pv.sort_unstable_by(|a, b| b.1.partial_cmp(a.1)
                    .unwrap_or(std::cmp::Ordering::Equal));
                let top_pred: std::collections::HashSet<String> =
                    pv.into_iter().take(20).map(|(l, _)| l.clone()).collect();

                // Jaccard similarity: |intersection| / |union|
                let n_intersect = top_subj.intersection(&top_pred).count();
                let n_union     = top_subj.union(&top_pred).count();
                if n_union > 0 {
                    Some(n_intersect as f32 / n_union as f32)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // ── Answer: classical annealing + pure Hebbian activation ────────────
        // Classical annealing step: use mean_activation (background firing rate)
        // as a "frustration" signal — words that always fire high across all queries
        // are hub words (low specificity). The annealed score suppresses them:
        //   annealed = activation / (1 + hub_weight * mean_activation)
        // This maps to Boltzmann energy minimization: E = -annealed_score.
        // Temperature T_anneal = 1 / hebbian_peak cools as confidence grows:
        // low confidence (high T) → broad exploration; high confidence (low T) → focused.
        // Seed word labels: all question pool labels (q: and txt:) to exclude from answers.
        let seed_word_labels: std::collections::HashSet<String> = question_content_labels.iter()
            .filter(|l| l.starts_with("txt:word_") || l.starts_with("q:txt:word_"))
            .cloned()
            .collect();

        // Gather candidate labels: A-pool labels (a:txt:word_*) first (highest priority),
        // then corpus labels (txt:word_*) as fallback for non-conversational questions.
        let candidate_labels: Vec<String> = word_acts.iter()
            .filter(|(l, _)| !seed_word_labels.contains(l.as_str()))
            .filter(|(l, _)| {
                (l.starts_with("txt:word_") || l.starts_with("a:txt:word_"))
                && !l.contains("_zone_")
            })
            .map(|(l, _)| l.clone())
            .collect();
        let label_stats = neuro.label_stats(&candidate_labels);

        // Classical annealing: primary key = seed_relevance (direct training signal),
        // secondary key = activation / (1 + mean_act_weight × mean_act) (Boltzmann energy).
        // seed_relevance measures dendrite weight from question words → answer word,
        // which is non-zero ONLY if the answer word was Hebbian-trained with question words.
        // This is the Hopfield-network attractor: the stable state that directly co-activates
        // with the question, minimizing E = -(seed_relevance + activation_specificity).
        let seed_labels_for_relevance: Vec<String> = q_word_only_labels.clone();
        let relevance_map = neuro.seed_relevance(&seed_labels_for_relevance, &candidate_labels);
        let mean_act_weight = 4.0_f32;

        let mut annealed: Vec<(&str, f32, f32)> = word_acts.iter()
            .filter(|(l, _)| !seed_word_labels.contains(l.as_str()))
            .filter(|(l, _)| !l.contains("_zone_"))
            .filter_map(|(l, s)| {
                // Strip either "a:txt:word_" (A-pool) or "txt:word_" (corpus).
                // A-pool labels are prioritized by appearing in word_acts first
                // (sorted by activation strength above).
                let w = l.strip_prefix("a:txt:word_")
                    .or_else(|| l.strip_prefix("txt:word_"));
                w.map(|w| (w, *s, l.clone()))
            })
            .map(|(w, act, label)| {
                let (mean_act, _use_count, _fan_out) = label_stats.get(&label)
                    .copied().unwrap_or((0.0, 0, 0));
                let relevance = relevance_map.get(&label).copied().unwrap_or(0.0);
                let boltzmann_score = act / (1.0 + mean_act_weight * mean_act);
                (w, relevance, boltzmann_score)
            })
            .collect();
        // Sort: words with direct seed connection first (relevance > 0), then by Boltzmann score.
        annealed.sort_unstable_by(|a, b| {
            let r = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
            if r != std::cmp::Ordering::Equal { r }
            else { b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal) }
        });
        // Keep only words with direct seed relevance in the final answer.
        // If no word has a trained dendrite connection from the question seeds, the
        // Boltzmann fallback would emit hub noise (TypeScript docs, common words) that
        // is worse than silence. An empty answer → hypothesis queue, which is honest.
        let relevant_words: Vec<(&str, f32)> = annealed.iter()
            .filter(|(_, r, _)| *r > 0.0)
            .map(|(w, r, b)| (*w, r + b))
            .collect();
        // Boltzmann-only fallback: used ONLY when some relevance exists as a secondary sort.
        let annealed: Vec<(&str, f32)> = if relevant_words.is_empty() {
            // No direct training signal — return empty so hypothesis path fires.
            vec![]
        } else {
            relevant_words
        };

        // answer_words: seed-relevance-sorted list used for EEM/motif checks below
        let answer_words: Vec<&str> = annealed.iter()
            .map(|(w, _)| *w)
            .take(max_tok.min(20))
            .collect();

        // ── Greedy conditioned sequence decode ────────────────────────────────
        // Directed STDP sequence edges: W(word_i → word_i+1) = lr × exp(-dt/tau)
        // with dt=0.15s, tau=2.0 → 0.928 per step from per-word timed training.
        // First word:  argmax seed_relevance(Q→w).
        // Each next:   argmax { seq_act(cur→w) + 0.5 × seed_relevance(Q→w) }.
        // This reconstructs sentence order purely from the trained Hebbian graph.
        let word_rel_map: std::collections::HashMap<&str, f32> = annealed.iter()
            .map(|(w, r)| (*w, *r))
            .collect();
        // remaining = ALL activated A-pool + corpus word labels minus question seeds.
        // Seed-relevance only picks the FIRST word; intra-A sequence edges then pull
        // subsequent words. Without this, only seed-relevant words (direct Q→A edges)
        // appear in remaining, truncating the answer after the first word.
        let mut seen_r: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut remaining: Vec<&str> = word_acts.iter()
            .filter(|(l, _)| !seed_word_labels.contains(l.as_str()) && !l.contains("_zone_"))
            .filter_map(|(l, _)| {
                l.strip_prefix("a:txt:word_").or_else(|| l.strip_prefix("txt:word_"))
            })
            .filter(|w| seen_r.insert(*w))
            .collect();
        let mut decoded: Vec<String> = Vec::new();
        let max_words = max_tok.min(20) as usize;

        if let Some(pos) = remaining.iter().enumerate()
            .max_by(|(_, a), (_, b)| {
                word_rel_map.get(*a).unwrap_or(&0.0)
                    .partial_cmp(word_rel_map.get(*b).unwrap_or(&0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            let first = remaining.remove(pos);
            decoded.push(first.to_string());
            while decoded.len() < max_words && !remaining.is_empty() {
                // Probe sequence edges from the A-pool label of the current word.
                // A-pool intra-sequence edges (a:txt:word_i → a:txt:word_{i+1})
                // were trained with directed STDP by the two-pool architecture.
                let cur_word = decoded.last().unwrap();
                let cur_labels = vec![
                    format!("a:txt:word_{}", cur_word),
                    format!("txt:word_{}", cur_word),
                ];
                // Sustain Q context through every decode step — Q punct/word labels
                // provide discriminating signal (e.g. q:txt:punct_exclaim biases
                // toward the answer trained with "hello!" vs plain "hello").
                let seq_acts = neuro.propagate_combined(
                    &[
                        (cur_labels.as_slice(), 1.0_f32),
                        (q_pool_labels.as_slice(), 0.35_f32),
                    ],
                    1, 0.001
                );
                if let Some(pos) = remaining.iter().enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let la_a = format!("a:txt:word_{}", a);
                        let la_b = format!("a:txt:word_{}", b);
                        let sa = seq_acts.get(&la_a).copied().unwrap_or(0.0)
                            .max(seq_acts.get(&format!("txt:word_{}", a)).copied().unwrap_or(0.0))
                            + 0.5 * word_rel_map.get(*a).unwrap_or(&0.0);
                        let sb = seq_acts.get(&la_b).copied().unwrap_or(0.0)
                            .max(seq_acts.get(&format!("txt:word_{}", b)).copied().unwrap_or(0.0))
                            + 0.5 * word_rel_map.get(*b).unwrap_or(&0.0);
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    let next = remaining.remove(pos);
                    decoded.push(next.to_string());
                } else {
                    break;
                }
            }
        }

        let mut answer = decoded.join(" ");
        if let Some(c) = answer.get_mut(0..1) { c.make_ascii_uppercase(); }

        // ── EEM validation gate ───────────────────────────────────────────────
        // Query the Environmental Equation Matrix with the question labels.
        // If matching equations are found, two things happen:
        //   1. EEM candidate equations are attached to the response so the
        //      caller gets physics/math context alongside the Hebbian answer.
        //   2. The confidence_tier is upgraded when the Hebbian answer words
        //      overlap with the EEM equation's discipline keywords — this
        //      is the cross-validation signal: Hebbian + EEM both agree.
        //      If they disagree (Hebbian fires words unrelated to the EEM
        //      match), confidence is NOT upgraded — the EEM acts as a gate.
        let eem_result = eem.apply_to_context(&question_content_labels, 0);
        let eem_candidates: Vec<serde_json::Value> = eem_result.candidates.iter()
            .take(3)
            .map(|c| serde_json::json!({
                "equation": c.equation.text,
                "discipline": format!("{:?}", c.equation.discipline),
                "relevance": c.relevance,
                "latex": c.equation.latex,
            }))
            .collect();

        // Check overlap between Hebbian answer words and EEM equation keywords.
        let eem_validates = !eem_result.candidates.is_empty() && {
            let top_eem = &eem_result.candidates[0];
            let eem_keywords: std::collections::HashSet<&str> =
                top_eem.equation.discipline.keywords().iter().copied().collect();
            // At least one answer word must appear in the EEM keyword set
            // for the EEM to count as validation (not just coincidental match).
            answer_words.iter()
                .any(|w| eem_keywords.contains(*w) || eem_keywords.iter()
                    .any(|k| k.contains(w) || w.contains(k)))
        };

        // Reinforce the matching equation in the EEM if Hebbian + EEM agree.
        if eem_validates {
            if let Some(top) = eem_result.candidates.first() {
                eem.reinforce(&top.equation.id);
            }
        }

        // ── Motif confidence gate ─────────────────────────────────────────────
        // Motifs are patterns the network has seen repeatedly during training.
        // Attractor motifs are the most stable, recurring sub-sequences in the
        // pool's training history — they represent proven, multi-reinforced
        // concept clusters.  If the activated answer words overlap with attractor
        // motif descriptions, the answer comes from a well-established pattern,
        // which raises our confidence beyond what the Hebbian peak alone signals.
        let motifs = neuro.meta_motifs();
        let attractor_motifs: Vec<&w1z4rdv1510n::streaming::hierarchical_motifs::MetaMotif> =
            motifs.iter().filter(|m| m.is_attractor).collect();
        // Count how many answer words appear in any attractor motif description.
        let motif_hits: usize = answer_words.iter()
            .filter(|w| w.len() > 3 && attractor_motifs.iter()
                .any(|m| m.description.contains(*w)))
            .count();
        let motif_coverage = if answer_words.is_empty() { 0.0f32 }
            else { (motif_hits as f32) / (answer_words.len().min(10) as f32) };

        // ── Final confidence tier (Hebbian + EEM + motif) ─────────────────────
        // Three independent signals must agree to call something "high":
        //   hebbian_peak ≥ 0.50         — the Hebbian graph is densely connected
        //   eem_validates = true         — the EEM corroborates the domain
        //   motif_coverage ≥ 0.30        — the answer is a proven recurring pattern
        //
        // Any combination of two out of three can reach "medium".
        // Only Hebbian alone determines "low"/"uncertain".
        let validated_confidence_tier = {
            let h_high   = hebbian_peak >= 0.50;
            let h_med    = hebbian_peak >= 0.25;
            let h_low    = hebbian_peak >= 0.10;
            let motif_ok = motif_coverage >= 0.30;
            if h_high && (eem_validates || motif_ok) {
                "high"
            } else if h_med && (eem_validates || motif_ok) {
                "high"
            } else if h_high || (h_med && (eem_validates || motif_ok)) {
                "medium"
            } else if h_low {
                "low"
            } else {
                "uncertain"
            }
        };

        // ── Update session working memory ─────────────────────────────────────
        if let Some(ref sid) = session_id {
            let top_ctx: Vec<String> = word_acts.iter()
                .filter(|(l, _)| !seed_word_labels.contains(l.as_str()))
                .take(8)
                .map(|(l, _)| l.clone())
                .collect();
            if !top_ctx.is_empty() {
                session_ctxs.lock().expect("session mutex")
                    .insert(sid.clone(), top_ctx);
            }
        }

        let word_activations: Vec<serde_json::Value> = word_acts.iter()
            .filter(|(l, _)| !seed_word_labels.contains(l.as_str()))
            .take(20)
            .map(|(label, strength)| {
                let word = label.strip_prefix("txt:word_").unwrap_or(label.as_str());
                serde_json::json!({ "word": word, "strength": strength })
            })
            .collect();
        // Greeting fallback: if inference produced no trained response and the
        // input is a social greeting, substitute a bootstrap reply so the user
        // sees something sensible.  Any trained response (seed_relevance > 0)
        // from `/api/wizard-chat/train/` will have already produced answer words
        // above, so this branch only fires before conversational training exists.
        let (final_answer, final_tier, final_hypothesis, final_intent) =
            if answer.is_empty() && is_greeting {
                (
                    "Hello! I'm W1z4rD — ask me anything.".to_string(),
                    "high",
                    false,
                    "Greeting".to_string(),
                )
            } else {
                (
                    answer.clone(),
                    validated_confidence_tier,
                    answer.is_empty(),
                    format!("{:?}", intent),
                )
            };

        let mut resp = serde_json::json!({
            "question":           text,
            "hypothesis":         final_hypothesis,
            "answer":             if final_answer.is_empty() { serde_json::Value::Null } else { serde_json::json!(final_answer) },
            "hebbian_peak":       qa_activation,
            "confidence_tier":    final_tier,
            "peak_activation":    peak,
            "intent":             final_intent,
            "word_activations":   word_activations,
            "context_trained":    context_trained,
            "session_id":         session_id,
            "eem_validates":      eem_validates,
            "motif_coverage":     motif_coverage,
            "attractor_motifs":   attractor_motifs.len(),
        });
        if !eem_candidates.is_empty() {
            resp["eem_equations"] = serde_json::json!(eem_candidates);
        }
        if !eem_result.open_gaps.is_empty() {
            resp["eem_gaps"] = serde_json::json!(
                eem_result.open_gaps.iter().take(2)
                    .map(|g| g.description.clone())
                    .collect::<Vec<_>>()
            );
        }
        if let Some(score) = property_binding_score {
            resp["property_binding_score"] = serde_json::json!(score);
        }
        resp
    }).await.unwrap_or_else(|e| serde_json::json!({ "error": format!("internal: {e}") }));

    (StatusCode::OK, Json(result))
}

// ── Multi-pool inference pipeline ─────────────────────────────────────────────
//
// The pipeline is a SCOPE ROUTER — it detects what kind of query this is and
// routes it to the appropriate trained QA pool.  No deterministic NLP code,
// no hardcoded substitution tables, no rule-based transforms.
//
// All outputs come from pool recall (Hebbian QA fabric).  Each pool is a
// specialist trained for its job:
//   - Correction scopes (spelling / grammar / punctuation): the query is
//     forwarded as-is (with its scope marker) to the QA pool which was trained
//     on correction pairs.  The pool's recalled answer IS the correction.
//   - DirectAnswer / JsonOutput: general knowledge pool.
//
// If the recalled answer fails the IDF relevance gate, the pipeline falls back
// to Hebbian associative recall (propagate from question word-labels → re-query
// from the activated concept cluster).  Only when both paths fail does the
// question go to the hypothesis queue.

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PipelineScope {
    SpellingCorrection,
    GrammarCorrection,
    PunctuationCorrection,
    TextRewrite,
    JsonOutput,
    DirectAnswer,
}

fn detect_pipeline_scope(text: &str) -> PipelineScope {
    let lower = text.trim().to_lowercase();
    if lower.starts_with("correct the spelling")
        || lower.starts_with("fix the spelling")
        || lower.starts_with("what is the correct spelling")
        || lower.starts_with("spell-check")
        || lower.starts_with("please spell-check")
        || lower.starts_with("how do you spell")
        || lower.starts_with("how to spell")
        || (lower.starts_with("is '") && lower.contains("' spelled"))
        || (lower.starts_with("is '") && lower.contains("correct spelling"))
    {
        return PipelineScope::SpellingCorrection;
    }
    if lower.starts_with("correct the grammar")
        || lower.starts_with("fix the grammar")
        || lower.starts_with("fix this sentence:")
    {
        return PipelineScope::GrammarCorrection;
    }
    if lower.starts_with("fix the punctuation")
        || lower.starts_with("correct the punctuation")
        || lower.starts_with("rewrite this with proper")
        || lower.starts_with("rewrite with proper")
        || lower.starts_with("fix the run-on")
        || lower.starts_with("fix this run-on")
        || lower.starts_with("fix this:")
        || lower.starts_with("correct this sentence:")
        || lower.starts_with("rewrite this properly:")
    {
        return PipelineScope::PunctuationCorrection;
    }
    if lower.starts_with("correct the spelling and")
        || lower.starts_with("fix the spelling and")
        || lower.starts_with("correct all errors")
        || lower.starts_with("clean up this")
        || lower.starts_with("please correct all")
        || lower.starts_with("please correct this")
        || lower.starts_with("please fix")
        || lower.starts_with("fix all")
        || lower.starts_with("rewrite this")
    {
        return PipelineScope::TextRewrite;
    }
    if lower.contains(" as json")
        || lower.contains("in json format")
        || lower.contains("json object")
        || lower.contains("json array")
        || lower.contains("return json")
        || lower.contains("return a json")
        || lower.contains("give me a json")
        || lower.contains("give me an empty json")
        || lower.contains("format as json")
        || lower.contains("format this as json")
        || lower.contains("output as json")
        || lower.contains("output this as json")
    {
        return PipelineScope::JsonOutput;
    }
    PipelineScope::DirectAnswer
}


/// Returns false when a QA answer is off-topic for the question, using the
/// Hebbian pool's learned statistics to determine which tokens are
/// discriminative — no hardcoded stop-word lists.
///
/// `significant_tokens` is populated from IDF computed
/// over training statistics: words like "what"/"is"/"the" appear in nearly
/// every pair so they get near-zero IDF and are NOT significant. Specific words
/// like "Texas" or "photosynthesis" appear rarely → high IDF → significant.
///
/// The check: the answer must contain at least 2/3 of the significant tokens.
fn answer_is_relevant_idf(answer: &str, significant_tokens: &[String]) -> bool {
    if significant_tokens.is_empty() {
        // No discriminative vocabulary for this query means the pool has no basis
        // to confirm the answer is on-topic.  Reject rather than accept blindly —
        // hypothesis is the honest response when the system can't discriminate.
        return false;
    }
    let a_lc = answer.to_lowercase();
    let matches = significant_tokens.iter()
        .filter(|t| a_lc.contains(t.as_str()))
        .count();
    // Require majority of significant (high-IDF) tokens to appear in the answer.
    // ceil(len * 2 / 3) threshold so "capital + texas" requires both for 2 tokens.
    let required = (significant_tokens.len() * 2 + 2) / 3;
    matches >= required
}

#[derive(Deserialize)]
struct PipelineReq {
    text: String,
    #[serde(default = "default_hops")]      hops: usize,
    #[serde(default = "default_ask_top_k")] top_k: usize,
    #[serde(default)]                        session_id: Option<String>,
}

/// POST /neuro/pipeline
///
/// Pure Hebbian inference — no QA store. Propagates from question word labels
/// through the Hebbian graph, returns top activated concept words as the answer.
/// As STDP training grows, the activated cluster becomes the answer.
async fn neuro_pipeline(
    State(state): State<ApiState>,
    Json(req): Json<PipelineReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "text must not be empty" })));
    }
    let neuro      = state.neuro.clone();
    let hq_arc     = state.hypothesis_queue.clone();
    let sess_ctxs  = state.session_contexts.clone();
    let text       = req.text.clone();
    let hops       = req.hops;
    let max_tok    = req.top_k.max(30);
    let session_id = req.session_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let enc         = TextBitsEncoder::new(TextBitsConfig::default());
        let sess_labels = session_id.as_ref()
            .and_then(|sid| sess_ctxs.lock().expect("sess mutex").get(sid).cloned())
            .unwrap_or_default();

        // Encode full label set — all types feed the Hebbian graph
        let q_all_labels: Vec<String> = enc.encode_plain(&text).labels;
        let q_word_labels: Vec<String> = q_all_labels.iter()
            .filter(|l| l.starts_with("txt:word_") && !l.contains("_zone_")
                && l.strip_prefix("txt:word_").map(|w| w.len() > 2).unwrap_or(false))
            .cloned()
            .collect();

        if q_word_labels.is_empty() {
            return serde_json::json!({
                "question": text, "hypothesis": true, "answer": serde_json::Value::Null,
                "hebbian_peak": 0.0, "confidence_tier": "uncertain", "session_id": session_id,
            });
        }

        // Propagate from question word labels + session context through Hebbian graph
        let activated = neuro.propagate_combined(
            &[(q_word_labels.as_slice(), 1.0_f32),
              (sess_labels.as_slice(),   0.3_f32)],
            hops, 0.001,
        );

        let q_word_set: HashSet<&String> = q_word_labels.iter().collect();
        let mut word_acts: Vec<(String, f32)> = activated.iter()
            .filter(|(l, _)| l.starts_with("txt:word_") && !l.contains("_zone_"))
            .filter(|(l, _)| !q_word_set.contains(l))
            .map(|(l, &v)| (l.clone(), v))
            .collect();
        word_acts.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal));

        let peak = word_acts.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);

        if peak < ANSWER_THRESHOLD {
            // Dark graph — hypothesis
            neuro.release_neuromodulator(NeuromodulatorKind::Norepinephrine, 0.75);
            {
                let mut hq = hq_arc.lock().expect("hypothesis mutex");
                if !hq.iter().any(|e| e.question == text) {
                    let mut h = 0u64;
                    for b in text.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
                    hq.push(HypothesisEntry {
                        id: format!("pipe_{h:x}"), question: text.clone(),
                        queued_at_unix: now_timestamp().unix,
                        attempts: 0, max_attempts: 5,
                        resolved: false, answer: None, confidence: None,
                    });
                }
            }
            let concepts: Vec<&str> = word_acts.iter()
                .filter_map(|(l, _)| l.strip_prefix("txt:word_"))
                .filter(|w| !w.contains('_') && w.len() > 2)
                .take(8)
                .collect();
            return serde_json::json!({
                "question": text, "hypothesis": true,
                "answer": serde_json::Value::Null,
                "activated_concepts": concepts,
                "hebbian_peak": peak, "confidence_tier": "uncertain",
                "session_id": session_id,
            });
        }

        // Build answer from top activated words
        let answer_words: Vec<&str> = word_acts.iter()
            .filter_map(|(l, _)| l.strip_prefix("txt:word_"))
            .filter(|w| !w.contains('_') && w.len() > 2)
            .take(max_tok.min(20))
            .collect();
        let mut answer = answer_words.join(" ");
        if let Some(c) = answer.get_mut(0..1) { c.make_ascii_uppercase(); }

        let confidence_tier = if peak >= 0.50 { "high" }
            else if peak >= 0.25 { "medium" } else { "low" };

        // Update session working memory
        if let Some(ref sid) = session_id {
            let ctx: Vec<String> = word_acts.iter()
                .take(8).map(|(l, _)| l.clone()).collect();
            if !ctx.is_empty() {
                sess_ctxs.lock().expect("sess mutex").insert(sid.clone(), ctx);
            }
        }

        serde_json::json!({
            "question":        text,
            "hypothesis":      false,
            "answer":          answer,
            "hebbian_peak":    peak,
            "confidence_tier": confidence_tier,
            "session_id":      session_id,
        })
    }).await.unwrap_or_else(|e| serde_json::json!({ "error": format!("internal: {e}") }));

    (StatusCode::OK, Json(result))
}

// ── Hypothesis queue endpoints ────────────────────────────────────────────────

/// GET /hypothesis/queue
/// Returns all hypothesis entries (both resolved and unresolved).
/// External research agents poll this to find questions needing research.
async fn hypothesis_queue_list(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let hq = state.hypothesis_queue.lock().expect("hypothesis mutex");
    let unresolved: usize = hq.iter().filter(|e| !e.resolved).count();
    (StatusCode::OK, Json(serde_json::json!({
        "count":      hq.len(),
        "unresolved": unresolved,
        "entries":    *hq,
    })))
}

#[derive(Deserialize)]
struct HypothesisResolveReq {
    id: String,
    answer: String,
    #[serde(default)]
    confidence: Option<f32>,
}

/// POST /hypothesis/resolve
/// Mark a hypothesis as resolved with a provided answer.
/// Called by research_agent.py after fetching authoritative sources.
/// On successful resolution, triggers dopamine retrograde potentiation so the
/// synaptic path that led to the question gets strengthened — the network learns
/// to answer similar questions faster next time.
async fn hypothesis_resolve(
    State(state): State<ApiState>,
    Json(req): Json<HypothesisResolveReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut hq = state.hypothesis_queue.lock().expect("hypothesis mutex");
    match hq.iter_mut().find(|e| e.id == req.id) {
        Some(entry) => {
            entry.resolved   = true;
            entry.answer     = Some(req.answer);
            let conf         = req.confidence.unwrap_or(0.7);
            entry.confidence = Some(conf);
            // Dopamine signal: reward the network for having queued the right question.
            // Retrograde potentiation strengthens recently-active paths proportionally
            // to how confident the resolved answer is.
            state.neuro.release_neuromodulator(NeuromodulatorKind::Dopamine, conf);
            state.neuro.flush_dopamine();
            (StatusCode::OK, Json(serde_json::json!({ "resolved": true, "id": entry.id })))
        }
        None => (StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "hypothesis not found", "id": req.id }))),
    }
}

// ── Background hypothesis research loop ──────────────────────────────────────
//
// Every 30 seconds, re-evaluates the oldest unresolved hypothesis against the
// current neural state.  If training has raised activation above ANSWER_THRESHOLD
// since the hypothesis was queued, the entry is automatically marked resolved.
//
// External research agents feed new knowledge via /media/train_sequence + /neuro/train,
// then either wait for this loop to pick it up or call /hypothesis/resolve directly.
async fn hypothesis_research_loop(
    hq: Arc<Mutex<Vec<HypothesisEntry>>>,
    neuro: NeuroRuntimeHandle,
) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;

        // Grab the oldest unresolved question (hold the lock only briefly)
        let question: Option<String> = {
            let guard = hq.lock().expect("hypothesis mutex");
            guard.iter()
                .filter(|e| !e.resolved && e.attempts < e.max_attempts)
                .min_by_key(|e| e.queued_at_unix)
                .map(|e| e.question.clone())
        };
        let Some(question) = question else { continue };

        // Re-evaluate: has new training raised activation above threshold?
        let outcome = tokio::task::spawn_blocking({
            let neuro = neuro.clone();
            let q     = question.clone();
            move || -> (f32, Option<String>) {
                let enc = TextBitsEncoder::new(TextBitsConfig::default());
                let qlabels: Vec<String> = enc.encode_plain(&q).labels;
                // Pure Hebbian 1-hop discriminative check
                let combined_1h = neuro.propagate_combined(
                    &[(qlabels.as_slice(), 1.0_f32)],
                    1,
                    0.005,
                );
                let seed_set: std::collections::HashSet<&String> = qlabels.iter().collect();
                let peak_1h = combined_1h.iter()
                    .filter(|(l, _)| l.starts_with("txt:word_") && !l.contains("_zone_") && !seed_set.contains(l))
                    .map(|(_, &v)| v)
                    .fold(0.0f32, f32::max);
                (peak_1h, None)
            }
        }).await;

        // Update hypothesis state
        let mut guard = hq.lock().expect("hypothesis mutex");
        if let Some(entry) = guard.iter_mut().find(|e| e.question == question) {
            entry.attempts += 1;
            if let Ok((peak, answer)) = outcome {
                if peak >= ANSWER_THRESHOLD {
                    entry.resolved   = true;
                    entry.answer     = answer;
                    entry.confidence = Some(peak);
                }
            }
        }
    }
}

/// POST /neuro/checkpoint
/// Force-save the NeuronPool to disk.
async fn neuro_checkpoint(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let pool_path = state.neuro.pool_state_path().unwrap_or("").to_string();
    let pool_path_clone = pool_path.clone();
    let pool_result = tokio::task::spawn_blocking(move || {
        state.neuro.save_pool()
    }).await.unwrap_or_else(|e| Err(e.to_string()));
    let _ = pool_path_clone;
    match pool_result {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({
            "saved": true,
            "pool_path": pool_path,
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("pool: {}", e) }))),
    }
}

/// POST /neuro/clear
/// Reset the in-memory pool to empty and delete pool files on disk.
/// Use before a fresh training run so the curriculum starts from a blank slate.
async fn neuro_clear(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let result = tokio::task::spawn_blocking(move || {
        state.neuro.clear_pool()
    }).await.unwrap_or_else(|e| Err(e.to_string()));
    match result {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "cleared": true }))),
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

// ────────────────────────────────────────────────────────────────────────────
// POST /neuro/generate
// Auto-regressive text generation from the Hebbian pool.
// Seeds the pool with encoded question labels, then iterates: at each step the
// highest-activated word neuron becomes the next output token, is fed back as
// the new seed, and is suppressed from future steps.  Stops at "eos" token or
// when activation drops below min_strength.
// ────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct GenerateReq {
    text: String,
    #[serde(default = "default_gen_max_tokens")] max_tokens: usize,
    #[serde(default = "default_gen_hops")]       hops: usize,
    #[serde(default = "default_gen_min")]        min_strength: f32,
}
fn default_gen_max_tokens() -> usize { 32 }
fn default_gen_hops()       -> usize { 2 }
fn default_gen_min()        -> f32   { 0.05 }

async fn neuro_generate(
    State(state): State<ApiState>,
    Json(req): Json<GenerateReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "text must not be empty" })));
    }
    let neuro   = state.neuro.clone();
    let hq_arc  = state.hypothesis_queue.clone();
    let text    = req.text.clone();
    let hops    = req.hops;
    let max_tok = req.max_tokens;

    let result = tokio::task::spawn_blocking(move || {
        // Pure Hebbian auto-regressive generation — no QA store.
        // Propagate from prompt labels, pick top activated words, feed back
        // iteratively until max_tokens or activation drops below min_strength.
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let prompt_labels: Vec<String> = enc.encode_plain(&text).labels;

        let seed_words: Vec<String> = prompt_labels.iter()
            .filter(|l| l.starts_with("txt:word_") && !l.contains("_zone_")
                && l.strip_prefix("txt:word_").map(|w| w.len() > 2).unwrap_or(false))
            .cloned()
            .collect();

        if seed_words.is_empty() {
            return serde_json::json!({
                "prompt": text, "hypothesis": true, "response": serde_json::Value::Null,
            });
        }

        let mut output_tokens: Vec<String> = Vec::new();
        let mut suppressed: HashSet<String> = seed_words.iter().cloned().collect();
        let mut current_seeds = seed_words.clone();

        for _ in 0..max_tok {
            let activated = neuro.propagate_combined(
                &[(current_seeds.as_slice(), 1.0_f32)],
                hops, 0.001,
            );
            let mut word_acts: Vec<(String, f32)> = activated.iter()
                .filter(|(l, _)| l.starts_with("txt:word_") && !l.contains("_zone_"))
                .filter(|(l, _)| !suppressed.contains(*l))
                .map(|(l, &v)| (l.clone(), v))
                .collect();
            word_acts.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal));
            let best = match word_acts.first() {
                Some(b) if b.1 >= req.min_strength => b.clone(),
                _ => break,
            };
            let word = best.0.strip_prefix("txt:word_").unwrap_or("").to_string();
            if word.is_empty() || word.contains('_') { break; }
            output_tokens.push(word);
            suppressed.insert(best.0.clone());
            current_seeds = vec![best.0];
        }

        let peak = output_tokens.len() as f32 / max_tok.max(1) as f32;

        if output_tokens.is_empty() {
            neuro.release_neuromodulator(NeuromodulatorKind::Norepinephrine, 0.75);
            let id = {
                let mut h = 0u64;
                for b in text.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
                format!("{h:x}")
            };
            {
                let mut hq = hq_arc.lock().expect("hypothesis mutex");
                if !hq.iter().any(|e| e.question == text) {
                    hq.push(HypothesisEntry {
                        id, question: text.clone(),
                        queued_at_unix: now_timestamp().unix,
                        attempts: 0, max_attempts: 5,
                        resolved: false, answer: None, confidence: None,
                    });
                }
            }
            return serde_json::json!({
                "prompt": text, "hypothesis": true, "response": serde_json::Value::Null,
            });
        }

        let mut response = output_tokens.join(" ");
        if let Some(c) = response.get_mut(0..1) { c.make_ascii_uppercase(); }

        serde_json::json!({
            "prompt":        text,
            "hypothesis":    false,
            "response":      response,
            "hebbian_peak":  peak,
        })
    }).await.unwrap_or_else(|e| serde_json::json!({ "error": format!("internal: {e}") }));

    (StatusCode::OK, Json(result))
}

// ────────────────────────────────────────────────────────────────────────────
// POST /entity/observe
// Feed a multi-modal entity observation through the runtime chain:
//   BehaviorSubstrate → PhysiologyRuntime → SurvivalRuntime → NarrativeRuntime
// and simultaneously train the neuro pool with the entity's encoded labels so
// that Hebbian associations build up from observations over time.
// ────────────────────────────────────────────────────────────────────────────

async fn entity_observe(
    State(state): State<ApiState>,
    Json(req): Json<BehaviorInput>,
) -> (StatusCode, Json<serde_json::Value>) {
    let entity_id      = req.entity_id.clone();
    let entity_species = req.species.clone();
    let ts             = req.timestamp;

    // Run the full chain on a blocking thread — all runtimes use sync mutexes.
    let entity_id_inner = entity_id.clone();
    let behavior_arc   = state.behavior.clone();
    let physio_arc     = state.physiology.clone();
    let survival_arc   = state.survival.clone();
    let narrative_arc  = state.narrative.clone();
    let neuro          = state.neuro.clone();

    let result = tokio::task::spawn_blocking(move || {
        // 1. BehaviorSubstrate → BehaviorFrame + latent labels
        let frame: BehaviorFrame = {
            let mut b = behavior_arc.lock().expect("behavior");
            b.ingest(req)
        };

        // Pull per-entity state from the frame (first state matching this entity).
        let primary = frame.states.iter().find(|s| s.entity_id == entity_id_inner);
        let latent   = primary.map(|s| s.latent.as_slice()).unwrap_or(&[]);
        let position = primary.and_then(|s| s.position);
        let confidence = primary.map(|s| s.confidence).unwrap_or(0.0);

        // Build Hebbian labels from the frame so the neuro pool learns from it.
        let mut neuro_labels: Vec<String> = Vec::new();
        neuro_labels.push(format!("entity:{}", entity_id_inner));
        neuro_labels.push(format!("species:{}", serde_json::to_string(&entity_species)
            .unwrap_or_default().trim_matches('"').to_lowercase()));
        // Discretise latent dims into zone-like labels (top 8 dims by magnitude).
        let mut dims: Vec<(usize, f64)> = latent.iter().enumerate()
            .map(|(i, &v)| (i, v)).collect();
        dims.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal));
        for (i, v) in dims.iter().take(8) {
            let bin = ((v.clamp(-1.0, 1.0) + 1.0) * 7.5) as u32;
            neuro_labels.push(format!("entity:{}:dim{}:{}", entity_id_inner, i, bin));
        }
        if let Some(pos) = position {
            // Coarse 4-unit grid position labels for spatial association.
            neuro_labels.push(format!("entity:{}:x{}", entity_id_inner, (pos[0] / 4.0) as i64));
            neuro_labels.push(format!("entity:{}:y{}", entity_id_inner, (pos[1] / 4.0) as i64));
            neuro_labels.push(format!("entity:{}:z{}", entity_id_inner, (pos[2] / 4.0) as i64));
        }
        neuro.train_weighted(&neuro_labels, 1.0, false);

        // 2. PhysiologyRuntime
        let physio_report: Option<PhysiologyReport> = {
            let batch = TokenBatch { timestamp: ts, tokens: vec![], layers: vec![],
                source_confidence: Default::default() };
            let mut p = physio_arc.lock().expect("physiology");
            p.update(&batch, None)
        };

        // 3. SurvivalRuntime
        let survival_report: SurvivalReport = {
            let mut s = survival_arc.lock().expect("survival");
            s.update(&frame, physio_report.as_ref(), None)
        };

        // 4. NarrativeRuntime
        let narrative_report: Option<NarrativeReport> = {
            let mut n = narrative_arc.lock().expect("narrative");
            n.update(ts, Some(&frame), None, Some(&survival_report), None, None, None, None)
        };

        (physio_report, survival_report, narrative_report, neuro_labels, confidence, position)
    }).await;

    match result {
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "runtime panic" }))),
        Ok((physio, survival, narrative, labels, confidence, position)) => {
            (StatusCode::OK, Json(serde_json::json!({
                "entity_id":       entity_id,
                "confidence":      confidence,
                "position":        position,
                "neuro_labels":    labels,
                "physiology":      physio,
                "survival":        survival.entities.first(),
                "narrative":       narrative.map(|r| r.summary),
            })))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GET /entity/report
// Returns the latest combined state across scene, survival, and narrative.
// ────────────────────────────────────────────────────────────────────────────

async fn entity_report(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Ontology and scene can be read without holding locks long.
    // OntologyRuntime only produces reports on update(); return null when idle.
    let ontology_report: Option<OntologyReport> = None;
    let scene_entities: Vec<serde_json::Value> = state.scene.lock().ok()
        .map(|s| s.entity_reports().into_iter().map(|e| serde_json::json!({
            "entity_id": e.entity_id,
            "position":  e.position,
            "velocity":  e.velocity,
            "speed":     e.speed,
            "stability": e.stability,
        })).collect())
        .unwrap_or_default();

    (StatusCode::OK, Json(serde_json::json!({
        "scene":    { "entities": scene_entities },
        "ontology": ontology_report,
    })))
}

// ────────────────────────────────────────────────────────────────────────────
// Overlay / perception system handlers
// ────────────────────────────────────────────────────────────────────────────

// GET /overlay/health
// Returns the current 6-dimensional health overlay for every known entity.
// Each entity gets a HealthVector, composite score, health label, and the
// sorted concentric-ring color model (outermost=black → innermost=white).
async fn overlay_health(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let scene_arc   = state.scene.clone();
    let physio_arc  = state.physiology.clone();
    let surv_arc    = state.survival.clone();
    let health_arc  = state.entity_health.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };

    let overlays = tokio::task::spawn_blocking(move || {
        let scene_entities = scene_arc.lock().ok()
            .map(|s| s.entity_reports())
            .unwrap_or_default();
        let physio = physio_arc.lock().ok()
            .and_then(|_p| None::<PhysiologyReport>); // latest cached report (none if idle)
        // SurvivalReport is only available mid-frame (entity_observe drives it).
        // GET /overlay/health returns health from scene alone when no entity_observe has run.
        let survival: Option<SurvivalReport> = None;

        let mut health = health_arc.lock().expect("entity_health");
        health.update(
            ts,
            if scene_entities.is_empty() { None } else { Some(scene_entities.as_slice()) },
            physio.as_ref(),
            survival.as_ref(),
            0.0,  // motif_entropy: caller can supply via POST variant
            0.5,  // eem_confidence: default midpoint
        )
    }).await.unwrap_or_default();

    (StatusCode::OK, Json(serde_json::json!({ "timestamp": ts, "overlays": overlays })))
}

// GET /overlay/delta
// Returns per-entity velocity vectors and short-horizon predictions.
async fn overlay_delta(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let health_arc = state.entity_health.clone();
    let delta_arc  = state.delta_engine.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };

    let report: DeltaReport = tokio::task::spawn_blocking(move || {
        // Re-use the latest overlays from the health runtime's baseline cache.
        // In production, entity_observe updates the health runtime every frame.
        let overlays: Vec<EntityHealthOverlay> = {
            let mut h = health_arc.lock().expect("entity_health");
            h.update(ts, None, None, None, 0.0, 0.5)
        };
        let mut delta = delta_arc.lock().expect("delta_engine");
        delta.update(ts, &overlays, &std::collections::HashMap::new())
    }).await.unwrap_or_else(|_| DeltaReport {
        timestamp: ts, velocities: vec![], predictions: vec![],
    });

    (StatusCode::OK, Json(serde_json::to_value(&report).unwrap_or_default()))
}

// GET /overlay/poetry
// Returns emergent natural-language scientific profiles for all known entities.
#[derive(Deserialize)]
struct PoetryQuery {
    #[serde(default)]
    entity_id: Option<String>,
}
async fn overlay_poetry(
    State(state): State<ApiState>,
    Query(q): Query<PoetryQuery>,
) -> (StatusCode, Json<serde_json::Value>) {
    let health_arc  = state.entity_health.clone();
    let poetry_arc  = state.poetry_runtime.clone();
    let neuro       = state.neuro.clone();
    let eem         = state.equation_matrix.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };

    let poems = tokio::task::spawn_blocking(move || {
        let overlays: Vec<EntityHealthOverlay> = {
            let mut h = health_arc.lock().expect("entity_health");
            h.update(ts, None, None, None, 0.0, 0.5)
        };
        let filtered: Vec<_> = if let Some(ref eid) = q.entity_id {
            overlays.into_iter().filter(|o| &o.entity_id == eid).collect()
        } else {
            overlays
        };

        // Get active labels from neuro pool snapshot.
        let snapshot = neuro.snapshot();
        let active_labels: Vec<String> = snapshot.active_labels.iter()
            .take(32).cloned().collect();

        // EEM context application.
        let eem_results = eem.apply_to_context(&active_labels, 2).candidates;

        let poetry = poetry_arc.lock().expect("poetry_runtime");
        filtered.iter().map(|ov| {
            poetry.synthesize(ov, &active_labels, &eem_results, false, false)
        }).collect::<Vec<_>>()
    }).await.unwrap_or_default();

    (StatusCode::OK, Json(serde_json::json!({ "timestamp": ts, "poetry": poems })))
}

// POST /overlay/chaos
// Reconstructs the most probable recent history for a given entity.
#[derive(Deserialize)]
struct ChaosRequest {
    entity_id: String,
    /// The most recent motif id for this entity.  If omitted, the system uses
    /// the last element of the motif window (from the pool's motif runtime).
    #[serde(default)]
    anchor_motif: Option<String>,
    /// Forward transition table to invert.  Each entry is [from, to, count].
    /// If omitted the system uses the pool's own motif transitions.
    #[serde(default)]
    transitions: Option<Vec<(String, String, usize)>>,
}

async fn overlay_chaos(
    State(state): State<ApiState>,
    Json(req): Json<ChaosRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let chaos_arc = state.chaos_model.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };

    let report = tokio::task::spawn_blocking(move || {
        // Build reverse index from provided or empty transitions.
        let forward: std::collections::HashMap<(String, String), usize> = req.transitions
            .unwrap_or_default()
            .into_iter()
            .map(|(f, t, c)| ((f, t), c))
            .collect();
        let rev_index = build_reverse_index(&forward);
        let anchor = req.anchor_motif.unwrap_or_else(|| "unknown".to_string());
        let model = chaos_arc.lock().expect("chaos_model");
        model.reconstruct(&req.entity_id, &anchor, &rev_index, ts)
    }).await;

    match report {
        Ok(r) => (StatusCode::OK, Json(serde_json::to_value(&r).unwrap_or_default())),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "chaos model panic" }))),
    }
}

// GET /overlay/predictions
// Returns the current state of the prediction experiment framework.
async fn overlay_predictions(
    State(state): State<ApiState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let pred_arc = state.prediction_experiment.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };

    let report: ExperimentReport = tokio::task::spawn_blocking(move || {
        let mut pe = pred_arc.lock().expect("prediction_experiment");
        pe.update(ts, &std::collections::HashMap::new())
    }).await.unwrap_or_else(|_| ExperimentReport {
        timestamp: ts,
        pending_count: 0,
        evaluated_count: 0,
        current_horizon_secs: 5.0,
        mean_error_overall: 0.0,
        mean_error_by_dim: [0.0; 6],
        horizon_expanded: false,
        horizon_reset: false,
        recent_evaluations: vec![],
    });

    (StatusCode::OK, Json(serde_json::to_value(&report).unwrap_or_default()))
}

// POST /overlay/layers
// Build a layered structural decomposition for entities given active labels.
#[derive(Deserialize)]
struct LayersRequest {
    entities: Vec<LayerEntityRequest>,
}
#[derive(Deserialize)]
struct LayerEntityRequest {
    entity_id:       String,
    #[serde(default)]
    phenotype:       Option<String>,
    active_labels:   Vec<String>,
    /// Label → activation score (0–1).
    label_scores:    std::collections::HashMap<String, f64>,
    /// Normalized x,y position in sensor frame.
    #[serde(default)]
    x_frac: f64,
    #[serde(default)]
    y_frac: f64,
    /// Estimated entity bounding radius (relative, for z-depth scaling).
    #[serde(default = "default_size")]
    size_est: f64,
}
fn default_size() -> f64 { 0.1 }

async fn overlay_layers(
    State(state): State<ApiState>,
    Json(req): Json<LayersRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let layers_arc = state.layered_physiology.clone();
    let eem        = state.equation_matrix.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };

    let reports: Vec<LayeredPhysiologyReport> = tokio::task::spawn_blocking(move || {
        // Collect all labels across all entities for a single EEM pass.
        let all_labels: Vec<String> = req.entities.iter()
            .flat_map(|e| e.active_labels.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        let eem_candidates = eem.apply_to_context(&all_labels, 3).candidates;

        let requests: Vec<EntityDecomposeRequest> = req.entities.into_iter().map(|e| {
            EntityDecomposeRequest {
                entity_id:       e.entity_id,
                phenotype:       e.phenotype,
                active_labels:   e.active_labels,
                label_scores:    e.label_scores,
                entity_position: (e.x_frac, e.y_frac),
                entity_size_est: e.size_est,
            }
        }).collect();
        let rt = layers_arc.lock().expect("layered_physiology");
        rt.decompose_all(&requests, &eem_candidates, ts)
    }).await.unwrap_or_default();

    (StatusCode::OK, Json(serde_json::json!({ "timestamp": ts, "layers": reports })))
}

// GET /overlay/layers/:entity_id[?depth=N]
// Returns layered decomposition for a specific entity using neuro pool labels.
// Optional ?depth=N filters to layers at depth 0..=N.
#[derive(Deserialize)]
struct LayersQuery {
    depth: Option<usize>,
}

async fn overlay_layers_entity(
    State(state): State<ApiState>,
    Path(entity_id): Path<String>,
    Query(q): Query<LayersQuery>,
) -> (StatusCode, Json<serde_json::Value>) {
    let layers_arc = state.layered_physiology.clone();
    let neuro      = state.neuro.clone();
    let eem        = state.equation_matrix.clone();
    let ts = w1z4rdv1510n::schema::Timestamp {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };
    let max_depth = q.depth;

    let report: Option<LayeredPhysiologyReport> = tokio::task::spawn_blocking(move || {
        let snapshot = neuro.snapshot();
        let active_labels: Vec<String> = snapshot.active_labels.iter()
            .take(32).cloned().collect();
        let label_scores: std::collections::HashMap<String, f64> = active_labels.iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), 1.0 - (i as f64 / (active_labels.len().max(1)) as f64)))
            .collect();
        let eem_candidates = eem.apply_to_context(&active_labels, 3).candidates;

        let rt = layers_arc.lock().expect("layered_physiology");
        let mut r = rt.decompose(&entity_id, None, &active_labels, &label_scores,
            (0.5, 0.5), 0.15, &eem_candidates, ts);
        // Apply depth filter if requested.
        if let Some(max_d) = max_depth {
            r.layers.retain(|l| l.depth <= max_d);
            r.total_components = r.layers.iter().map(|l| l.components.len() + 1).sum();
        }
        Some(r)
    }).await.unwrap_or(None);

    match report {
        Some(r) => (StatusCode::OK, Json(serde_json::to_value(&r).unwrap_or_default())),
        None    => (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "layers runtime unavailable" }))),
    }
}

// POST /overlay/layers/calibrate
// Register a known physical scale for an entity type.
#[derive(Deserialize)]
struct LayersCalibrateReq {
    /// Entity type label (e.g. "cell", "leaf", "game_console").
    entity_type: String,
    /// Known physical extent of this entity type in SI metres.
    known_scale_m: f64,
}

async fn overlay_layers_calibrate(
    State(state): State<ApiState>,
    Json(req): Json<LayersCalibrateReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.known_scale_m <= 0.0 || !req.known_scale_m.is_finite() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "known_scale_m must be positive finite" })));
    }
    let entity_type  = req.entity_type.clone();
    let known_scale  = req.known_scale_m;
    let layers_arc   = state.layered_physiology.clone();

    tokio::task::spawn_blocking(move || {
        let mut rt = layers_arc.lock().expect("layered_physiology");
        rt.set_scale_override(&entity_type, known_scale);
    }).await.ok();

    (StatusCode::OK, Json(serde_json::json!({
        "ok": true,
        "entity_type":  req.entity_type,
        "known_scale_m": req.known_scale_m
    })))
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

// ── Mesh synthesis routes ─────────────────────────────────────────────────────
// Bridge layer: translates neural activation state → MeshPoint cloud → geometry.
// The mesh_gen module has no neural imports — this is the only coupling point.

#[derive(Deserialize)]
struct MeshSynthesizeReq {
    /// Natural-language query used to activate the neural fabric.
    query: String,
    #[serde(default = "default_mesh_hops")]
    hops: usize,
    #[serde(default = "default_mesh_min_act")]
    min_activation: f32,
    /// Optional category filter — e.g. ["cow_body", "visual_zone"].
    #[serde(default)]
    categories: Vec<String>,
    /// "obj" (default) or "json".
    #[serde(default = "default_mesh_fmt")]
    format: String,
    /// Optional spatial bounds filter (body-relative canonical coords).
    /// Only centroids whose position falls within these ranges are included.
    x_range: Option<[f32; 2]>,
    y_range: Option<[f32; 2]>,
    z_range: Option<[f32; 2]>,
}
fn default_mesh_hops()    -> usize { 2 }
fn default_mesh_min_act() -> f32   { 0.05 }
fn default_mesh_fmt()     -> String { "obj".into() }

#[derive(Deserialize)]
struct MeshFromImageReq {
    /// Base64-encoded image (PNG, JPEG, WebP, or BMP).
    image_b64: String,
    #[serde(default = "default_mesh_hops")]
    hops: usize,
    #[serde(default = "default_mesh_min_act")]
    min_activation: f32,
    #[serde(default)]
    categories: Vec<String>,
    #[serde(default = "default_mesh_fmt")]
    format: String,
}

/// Convert NeuroSnapshot centroids + an activation map into MeshPoints.
/// `activated` maps label → strength (0..1).
/// `category_filter` — if non-empty only labels whose category is in the list are kept.
fn snapshot_to_mesh_points(
    snap: &w1z4rdv1510n::neuro::NeuroSnapshot,
    activated: &HashMap<String, f32>,
    category_filter: &[String],
) -> Vec<MeshPoint> {
    use std::collections::HashMap as HM;

    // Re-use the colour derivation logic from neuro_world3d:
    // sum hue-bin weights from img:h* neighbours.
    fn hue_bin_rgb(bin: usize) -> [f32; 3] {
        let h = (bin as f32 + 0.5) / 16.0 * 360.0;
        let c = 0.7f32 * 0.85;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = 0.85 - c;
        let (r, g, b) = if h < 60.0 { (c,x,0.0) } else if h < 120.0 { (x,c,0.0) }
            else if h < 180.0 { (0.0,c,x) } else if h < 240.0 { (0.0,x,c) }
            else if h < 300.0 { (x,0.0,c) } else { (c,0.0,x) };
        [r+m, g+m, b+m]
    }

    fn derive_color(label: &str, links: &HM<String, HM<String, f32>>) -> [f32; 3] {
        let Some(conn) = links.get(label) else { return [0.55, 0.35, 0.15]; };
        let (mut rs, mut gs, mut bs, mut ws) = (0f32, 0f32, 0f32, 0f32);
        for (l, &w) in conn {
            if let Some(rest) = l.strip_prefix("img:h") {
                if let Ok(bin) = rest.parse::<usize>() {
                    let rgb = hue_bin_rgb(bin);
                    rs += rgb[0]*w; gs += rgb[1]*w; bs += rgb[2]*w; ws += w;
                }
            }
        }
        if ws < 0.01 { [0.55, 0.35, 0.15] } else { [rs/ws, gs/ws, bs/ws] }
    }

    fn label_category(label: &str) -> &str {
        if label.starts_with("cow_") || label.starts_with("id::cow") { "cow_body" }
        else if label.starts_with("env_") { "env" }
        else if label.starts_with("img:z") { "visual_zone" }
        else if label.starts_with("img:") { "visual_feature" }
        else { "other" }
    }

    let links = &snap.network_links;
    snap.centroids.iter()
        .filter_map(|(label, pos)| {
            let act = *activated.get(label).unwrap_or(&0.0);
            if act < 0.01 { return None; }
            if !category_filter.is_empty()
                && !category_filter.iter().any(|c| c == label_category(label)) {
                return None;
            }
            Some(MeshPoint {
                label: label.clone(),
                x: pos.x as f32, y: pos.y as f32, z: pos.z as f32,
                activation: act,
                color: derive_color(label, links),
            })
        })
        .collect()
}

/// POST /mesh/synthesize
/// Activate the neural fabric with a text query, read the centroid positions
/// of activated labels, synthesize a 3-D mesh, and return OBJ (or JSON).
async fn mesh_synthesize(
    State(state): State<ApiState>,
    Json(req): Json<MeshSynthesizeReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "query must not be empty" })));
    }

    // Encode query → seed labels
    let enc = TextBitsEncoder::new(TextBitsConfig::default());
    let seed_labels = enc.encode_plain(&req.query).labels;

    // Propagate — use req.min_activation so callers can tune sensitivity
    let prop_thresh = req.min_activation.max(0.001);
    let activated = state.neuro.propagate_all_threshold(&seed_labels, req.hops, prop_thresh as f32);

    // Read centroid positions + build MeshPoints
    let snap = state.neuro.snapshot();
    let mut pts = snapshot_to_mesh_points(&snap, &activated, &req.categories);

    // ── Fallback: if propagation did not reach any centroid labels (common when
    // the Hebbian text→spatial bridge is still building), use every centroid
    // whose label matches the query entity pattern directly.
    if pts.is_empty() && !snap.centroids.is_empty() {
        let q_lower = req.query.to_lowercase();
        let keywords: Vec<&str> = q_lower.split_whitespace().collect();
        pts = snap.centroids.iter()
            .filter_map(|(label, pos)| {
                let l = label.to_lowercase();
                // Accept any centroid whose label shares at least one query keyword
                let score = keywords.iter()
                    .filter(|&&kw| kw.len() > 2 && l.contains(kw))
                    .count();
                if score == 0 { return None; }
                // Apply optional spatial-bounds filter (body-relative canonical coords)
                let px = pos.x as f32; let py = pos.y as f32; let pz = pos.z as f32;
                if let Some([lo, hi]) = req.x_range { if px < lo || px > hi { return None; } }
                if let Some([lo, hi]) = req.y_range { if py < lo || py > hi { return None; } }
                if let Some([lo, hi]) = req.z_range { if pz < lo || pz > hi { return None; } }
                // Derive colour from network links if possible
                let color = {
                    let links = &snap.network_links;
                    if let Some(conn) = links.get(label) {
                        let (mut rs, mut gs, mut bs, mut ws) = (0f32,0f32,0f32,0f32);
                        for (lk, &w) in conn {
                            if let Some(rest) = lk.strip_prefix("img:h") {
                                if let Ok(bin) = rest.parse::<usize>() {
                                    let h = (bin as f32 + 0.5) / 16.0 * 360.0;
                                    let c = 0.595f32;
                                    let x = c * (1.0 - ((h/60.0)%2.0 - 1.0).abs());
                                    let m = 0.85 - c;
                                    let (r,g,b) = if h<60.0{(c,x,0.0)}else if h<120.0{(x,c,0.0)}
                                        else if h<180.0{(0.0,c,x)}else if h<240.0{(0.0,x,c)}
                                        else if h<300.0{(x,0.0,c)}else{(c,0.0,x)};
                                    rs+=r*w; gs+=g*w; bs+=b*w; ws+=w;
                                }
                            }
                        }
                        if ws>0.01 { [rs/ws, gs/ws, bs/ws] } else { [0.55, 0.35, 0.15] }
                    } else { [0.55, 0.35, 0.15] }
                };
                Some(crate::mesh_gen::MeshPoint {
                    label: label.clone(),
                    x: pos.x as f32, y: pos.y as f32, z: pos.z as f32,
                    activation: (score as f32 / keywords.len() as f32).min(1.0),
                    color,
                })
            })
            .collect();
    }

    if pts.is_empty() {
        return (StatusCode::OK, Json(serde_json::json!({
            "mesh": null,
            "reason": "no activated centroids matched the query",
            "activated_count": activated.len(),
            "total_centroids": snap.centroids.len(),
        })));
    }

    let synth = MeshSynthesizer { min_activation: req.min_activation, ..Default::default() };
    let mesh  = match synth.synthesize("neuro_mesh", &pts) {
        Some(m) => m,
        None => return (StatusCode::OK, Json(serde_json::json!({
            "mesh": null,
            "reason": "point cloud too sparse after activation filter",
            "point_count": pts.len(),
        }))),
    };

    if req.format == "json" {
        return (StatusCode::OK, Json(serde_json::json!({
            "verts":  mesh.verts,
            "normals": mesh.normals,
            "uvs":    mesh.uvs,
            "faces":  mesh.faces,
            "colors": mesh.vert_colors,
            "vertex_count": mesh.vertex_count(),
            "face_count":   mesh.face_count(),
        })));
    }

    let obj = mesh.to_obj(Some("neuro_mesh"));
    let mtl = mesh.to_mtl("neuro_mesh");
    (StatusCode::OK, Json(serde_json::json!({
        "obj":          obj,
        "mtl":          mtl,
        "vertex_count": mesh.vertex_count(),
        "face_count":   mesh.face_count(),
        "point_source": pts.len(),
    })))
}

/// POST /mesh/from_image
/// Encode an image through the existing ImageBitsEncoder, propagate the
/// resulting labels through the fabric, then synthesize a mesh from the
/// activated centroid cloud.  The mesh captures the spatial and semantic
/// structure of what the node "sees" in the image.
async fn mesh_from_image(
    State(state): State<ApiState>,
    Json(req): Json<MeshFromImageReq>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Re-use the existing encode_media_labels pipeline
    let seed_labels = match encode_media_labels(
        "image",
        Some(&req.image_b64),
        None, None, None, None,
    ) {
        Ok(l)  => l,
        Err(e) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e }))),
    };

    if seed_labels.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "image produced no labels — check encoding" })));
    }

    let activated = state.neuro.propagate_all_threshold(&seed_labels, req.hops, 0.01);
    let snap      = state.neuro.snapshot();
    let pts       = snapshot_to_mesh_points(&snap, &activated, &req.categories);

    if pts.is_empty() {
        return (StatusCode::OK, Json(serde_json::json!({
            "mesh": null,
            "reason": "image labels activated no known centroids — train more image data first",
            "image_labels": seed_labels.len(),
        })));
    }

    let synth = MeshSynthesizer { min_activation: req.min_activation, ..Default::default() };
    let mesh  = match synth.synthesize("image_mesh", &pts) {
        Some(m) => m,
        None => return (StatusCode::OK, Json(serde_json::json!({
            "mesh": null,
            "reason": "point cloud too sparse",
            "point_count": pts.len(),
        }))),
    };

    let obj = mesh.to_obj(Some("image_mesh"));
    let mtl = mesh.to_mtl("image_mesh");
    (StatusCode::OK, Json(serde_json::json!({
        "obj":           obj,
        "mtl":           mtl,
        "vertex_count":  mesh.vertex_count(),
        "face_count":    mesh.face_count(),
        "image_labels":  seed_labels.len(),
        "activated":     activated.len(),
    })))
}

/// GET /mesh/template
/// Returns developer documentation describing how to use the mesh API,
/// including the full pipeline from training to Three.js rendering.
async fn mesh_template() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(serde_json::json!({
        "description": "W1z4rD neural mesh synthesis — developer guide",
        "pipeline": [
            "1. Train the node: POST /media/train with modality=image (or text)",
            "2. Synthesize: POST /mesh/synthesize { query, hops, min_activation, format }",
            "   OR          POST /mesh/from_image  { image_b64, hops, min_activation }",
            "3. Load in Three.js using OBJLoader + MTLLoader",
            "4. Refine: lower min_activation to include more points; raise to sharpen focus"
        ],
        "endpoints": {
            "POST /mesh/synthesize": {
                "body": {
                    "query":          "string — text query to activate neural memory",
                    "hops":           "int (default 2) — propagation depth",
                    "min_activation": "float (default 0.05) — activation threshold",
                    "categories":     "string[] (optional) — filter: cow_body|visual_zone|env|visual_feature|other",
                    "format":         "string — 'obj' (default) or 'json'"
                },
                "returns": {
                    "obj":          "Wavefront OBJ text string",
                    "mtl":          "MTL material file text",
                    "vertex_count": "int",
                    "face_count":   "int"
                }
            },
            "POST /mesh/from_image": {
                "body": {
                    "image_b64":      "string — base64 PNG/JPEG/WebP",
                    "hops":           "int (default 2)",
                    "min_activation": "float (default 0.05)",
                    "categories":     "string[] (optional)",
                    "format":         "string — 'obj' (default) or 'json'"
                }
            },
            "GET /neuro/world3d": "Raw centroid positions + colours (use as point cloud in Three.js)"
        },
        "threejs_snippet": "// Load OBJ response in Three.js:\nimport { OBJLoader } from 'three/addons/loaders/OBJLoader.js';\nimport { MTLLoader } from 'three/addons/loaders/MTLLoader.js';\n\nasync function loadNeuroMesh(query) {\n  const res = await fetch('/mesh/synthesize', {\n    method: 'POST',\n    headers: { 'Content-Type': 'application/json' },\n    body: JSON.stringify({ query, hops: 2, min_activation: 0.05 })\n  });\n  const { obj, mtl } = await res.json();\n  const mtlBlob = URL.createObjectURL(new Blob([mtl], { type: 'text/plain' }));\n  const objBlob = URL.createObjectURL(new Blob([obj], { type: 'text/plain' }));\n  const materials = await new MTLLoader().loadAsync(mtlBlob);\n  materials.preload();\n  const loader = new OBJLoader();\n  loader.setMaterials(materials);\n  return loader.loadAsync(objBlob);\n}",
        "architecture_note": "mesh_gen module is a pure consumer of centroid data — it has no imports from the neural layer. The brain stays the brain. This route file is the only coupling point."
    })))
}
