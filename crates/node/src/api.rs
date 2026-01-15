use crate::config::NodeConfig;
use crate::ledger::LocalLedger;
use crate::paths::node_data_dir;
use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use blake2::{Blake2s256, Digest};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use w1z4rdv1510n::blockchain::{BlockchainLedger, NoopLedger, RewardBalance};
use w1z4rdv1510n::bridge::{BridgeProof, bridge_deposit_id};

#[derive(Clone)]
struct ApiState {
    ledger: Arc<dyn BlockchainLedger>,
    require_api_key: bool,
    api_key_hashes: Vec<[u8; 32]>,
    api_key_header: HeaderName,
    limiter: Arc<Mutex<ApiLimiter>>,
}

#[derive(Serialize)]
struct BridgeSubmitResponse {
    status: &'static str,
    deposit_id: String,
    balance: Option<RewardBalance>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn run_api(config: NodeConfig, addr: SocketAddr) -> Result<()> {
    let ledger = build_ledger(&config)?;
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
    let limiter = Arc::new(Mutex::new(ApiLimiter::new(60, 10)));
    let state = ApiState {
        ledger,
        require_api_key: config.api.require_api_key,
        api_key_hashes,
        api_key_header,
        limiter,
    };
    let max_body = config
        .blockchain
        .bridge
        .max_proof_bytes
        .max(1024);
    let app = Router::new()
        .route("/bridge/proof", post(submit_bridge_proof))
        .route("/balance/:node_id", get(get_balance))
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
    if let Err(err) = rate_limit(&state, &headers, "bridge") {
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
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
    if let Err(err) = rate_limit(&state, &headers, "balance") {
        let response = ErrorResponse {
            error: err.to_string(),
        };
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::to_value(response).unwrap_or_default()),
        );
    }
    if let Err(err) = authorize(&state, &headers) {
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
            StatusCode::OK,
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
    limiter.check_and_update(key, route)
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
    max_requests: u32,
    entries: std::collections::HashMap<String, ApiLimiterEntry>,
}

#[derive(Debug)]
struct ApiLimiterEntry {
    window_start: Instant,
    count: u32,
}

impl ApiLimiter {
    fn new(window_secs: u64, max_requests: u32) -> Self {
        Self {
            window_secs,
            max_requests,
            entries: std::collections::HashMap::new(),
        }
    }

    fn check_and_update(&mut self, key: &str, route: &str) -> Result<()> {
        let now = Instant::now();
        let full_key = format!("{key}:{route}");
        let entry = self.entries.entry(full_key).or_insert(ApiLimiterEntry {
            window_start: now,
            count: 0,
        });
        if now.duration_since(entry.window_start).as_secs() >= self.window_secs {
            entry.window_start = now;
            entry.count = 0;
        }
        entry.count = entry.count.saturating_add(1);
        if entry.count > self.max_requests {
            anyhow::bail!("rate limit exceeded");
        }
        Ok(())
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
