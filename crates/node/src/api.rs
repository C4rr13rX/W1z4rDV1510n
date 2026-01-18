use crate::config::NodeConfig;
use crate::data_mesh::{DataIngestRequest, DataMeshHandle, start_data_mesh};
use crate::ledger::LocalLedger;
use crate::p2p::NodeNetwork;
use crate::paths::node_data_dir;
use crate::wallet::{WalletStore, node_id_from_wallet};
use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use blake2::{Blake2s256, Digest};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, BridgeIntent, NoopLedger, RewardBalance, bridge_intent_id,
};
use w1z4rdv1510n::bridge::{BridgeProof, BridgeVerificationMode, ChainKind, bridge_deposit_id};
use w1z4rdv1510n::config::{BridgeConfig, BridgeChainPolicy};
use w1z4rdv1510n::schema::Timestamp;

#[derive(Clone)]
struct ApiState {
    node_id: String,
    ledger_backend: String,
    data_enabled: bool,
    bridge_enabled: bool,
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
    bridge_success: u64,
    balance_success: u64,
    bridge_chain_success: u64,
    bridge_intent_success: u64,
    data_ingest_success: u64,
    data_status_success: u64,
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
    bridge_success: AtomicU64,
    balance_success: AtomicU64,
    bridge_chain_success: AtomicU64,
    bridge_intent_success: AtomicU64,
    data_ingest_success: AtomicU64,
    data_status_success: AtomicU64,
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
            bridge_success: self.bridge_success.load(Ordering::Relaxed),
            balance_success: self.balance_success.load(Ordering::Relaxed),
            bridge_chain_success: self.bridge_chain_success.load(Ordering::Relaxed),
            bridge_intent_success: self.bridge_intent_success.load(Ordering::Relaxed),
            data_ingest_success: self.data_ingest_success.load(Ordering::Relaxed),
            data_status_success: self.data_status_success.load(Ordering::Relaxed),
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
    let ledger_backend = if config.ledger.enabled {
        config.ledger.backend.clone()
    } else {
        "disabled".to_string()
    };
    let state = ApiState {
        node_id: config.node_id.clone(),
        ledger_backend,
        data_enabled: config.data.enabled,
        bridge_enabled: config.blockchain.bridge.enabled,
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
    };
    let data_limit = if config.data.enabled {
        config.data.max_payload_bytes.saturating_mul(2)
    } else {
        0
    };
    let max_body = config
        .blockchain
        .bridge
        .max_proof_bytes
        .max(data_limit)
        .max(1024);
    let app = Router::new()
        .route("/health", get(get_health))
        .route("/ready", get(get_ready))
        .route("/bridge/proof", post(submit_bridge_proof))
        .route("/bridge/chains", get(get_bridge_chains))
        .route("/bridge/intent", post(submit_bridge_intent))
        .route("/data/ingest", post(submit_data_ingest))
        .route("/data/:data_id", get(get_data_status))
        .route("/balance/:node_id", get(get_balance))
        .route("/metrics", get(get_metrics))
        .with_state(state)
        .layer(DefaultBodyLimit::max(max_body));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build api runtime")?;
    runtime.block_on(async move {
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
