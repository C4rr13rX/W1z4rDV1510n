//! Standalone HTTP server for the W1z4rD brain.
//!
//! Embeds `w1z4rd_brain::Brain` with persistence at the configured
//! data directory and exposes a small REST surface so external
//! tools (the dashboard, the cluster, the experiment harness) can
//! drive it without entangling the main node's existing API.
//!
//! Endpoints (all JSON unless noted):
//!   GET  /health                       — liveness probe
//!   GET  /stats                        — `BrainStats`
//!   POST /observe  { pool_id, frame }  — frame is base64-url-safe of bytes
//!   POST /tick                         — advance_tick
//!   POST /integrate { query_pool, target_pool }
//!                                       — returns answer (base64) + grounding
//!   POST /checkpoint                   — flush to data dir
//!
//! Data dir is `W1Z4RDV1510N_DATA_DIR` (matching the main node's
//! conventions) or `./brain-data` if unset.  The brain is loaded
//! from `<data_dir>/brain.bin` on startup if it exists; otherwise a
//! `default_general_observer` identity is instantiated.  An
//! automatic checkpoint runs on SIGINT/SIGTERM.

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};
use w1z4rd_brain::{
    AtomEncoding, Brain, BrainIdentitySpec, BrainStats, BytePassthroughEncoding,
    PoolId, PoolPrototypeRegistry,
};

#[derive(Clone)]
struct AppState {
    brain: Arc<Mutex<Brain>>,
    checkpoint_path: PathBuf,
}

fn data_dir() -> PathBuf {
    std::env::var("W1Z4RDV1510N_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("brain-data"))
}

fn build_encodings(prefixes: &HashMap<PoolId, String>) -> HashMap<PoolId, Box<dyn AtomEncoding>> {
    prefixes.iter().map(|(pid, prefix)| {
        let leaked: &'static str = Box::leak(prefix.clone().into_boxed_str());
        let enc: Box<dyn AtomEncoding> = Box::new(BytePassthroughEncoding { prefix: leaked });
        (*pid, enc)
    }).collect()
}

fn load_or_build_brain(checkpoint_path: &PathBuf)
    -> Result<(Brain, HashMap<PoolId, String>)>
{
    // Always derive the prefix map from the identity (default observer).
    // If a snapshot is present, restore from it; otherwise build fresh
    // from the identity.
    let identity = BrainIdentitySpec::default_general_observer();
    let mut prefixes: HashMap<PoolId, String> = HashMap::new();
    prefixes.insert(0, "bind".into()); // binding pool, auto-created
    for ps in &identity.pools {
        prefixes.insert(ps.id, ps.atom_encoding_prefix.clone());
    }

    if checkpoint_path.exists() {
        info!("restoring brain from {}", checkpoint_path.display());
        let encodings = build_encodings(&prefixes);
        let (brain, missing) = Brain::restore(checkpoint_path, encodings)
            .with_context(|| format!("restore {}", checkpoint_path.display()))?;
        if !missing.is_empty() {
            warn!("restore missing encodings for pool ids {:?}", missing);
        }
        Ok((brain, prefixes))
    } else {
        info!("no checkpoint at {}; building fresh from default identity", checkpoint_path.display());
        let registry = PoolPrototypeRegistry::with_defaults();
        let brain = Brain::from_identity(&identity, &registry)
            .map_err(|e| anyhow::anyhow!("from_identity: {}", e))?;
        Ok((brain, prefixes))
    }
}

// -----------------------------------------------------------------
// Request / response types
// -----------------------------------------------------------------

#[derive(Deserialize)]
struct ObserveRequest {
    pool_id: PoolId,
    /// Frame bytes, base64-url-safe encoded (no padding).
    frame:   String,
}

#[derive(Serialize)]
struct ObserveResponse {
    fired_neurons: usize,
}

#[derive(Deserialize)]
struct IntegrateRequest {
    query_pool:  PoolId,
    target_pool: PoolId,
}

#[derive(Serialize)]
struct IntegrateResponse {
    /// base64-url-safe (no padding) of the answer bytes, or null
    /// when the brain is outside-grounding.
    answer:               Option<String>,
    confidence_tier:      String,
    fabric_confidence:    f32,
    eem_confidence:       Option<f32>,
    annealer_confidence:  Option<f32>,
    integrated_confidence: f32,
    outside_grounding:    bool,
    speculation_flag:     bool,
}

#[derive(Serialize)]
struct StatsResponse {
    tick:               u64,
    pool_count:         usize,
    total_neurons:      usize,
    total_concepts:     usize,
    total_binding:      usize,
    total_terminals:    usize,
    binding_pool_id:    PoolId,
    fingerprints_window: usize,
    checkpoint_path:    String,
}

#[derive(Serialize)]
struct CheckpointResponse {
    written_bytes: u64,
    path:          String,
}

// -----------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------

async fn health() -> &'static str { "ok\n" }

async fn stats(State(s): State<AppState>) -> Json<StatsResponse> {
    let brain = s.brain.lock().await;
    let bs: BrainStats = brain.stats();
    Json(StatsResponse {
        tick:               bs.tick,
        pool_count:         bs.pool_count,
        total_neurons:      bs.total_neurons,
        total_concepts:     bs.total_concepts,
        total_binding:      bs.total_binding,
        total_terminals:    bs.total_terminals,
        binding_pool_id:    bs.binding_pool_id,
        fingerprints_window: bs.fingerprints_window,
        checkpoint_path:    s.checkpoint_path.display().to_string(),
    })
}

async fn observe(
    State(s): State<AppState>,
    Json(req): Json<ObserveRequest>,
) -> Result<Json<ObserveResponse>, (axum::http::StatusCode, String)> {
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let frame = engine.decode(&req.frame).map_err(|e| {
        (axum::http::StatusCode::BAD_REQUEST, format!("invalid base64: {}", e))
    })?;
    let mut brain = s.brain.lock().await;
    let fired = brain.observe(req.pool_id, &frame);
    Ok(Json(ObserveResponse { fired_neurons: fired.len() }))
}

async fn tick(State(s): State<AppState>) -> Json<u64> {
    let mut brain = s.brain.lock().await;
    brain.advance_tick();
    Json(brain.stats().tick)
}

async fn integrate(
    State(s): State<AppState>,
    Json(req): Json<IntegrateRequest>,
) -> Json<IntegrateResponse> {
    let brain = s.brain.lock().await;
    let ans = brain.integrate(req.query_pool, req.target_pool);
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    Json(IntegrateResponse {
        answer:                ans.answer.as_ref().map(|b| engine.encode(b)),
        confidence_tier:       format!("{:?}", ans.confidence_tier),
        fabric_confidence:     ans.grounding.fabric_confidence,
        eem_confidence:        ans.grounding.eem_confidence,
        annealer_confidence:   ans.grounding.annealer_confidence,
        integrated_confidence: ans.grounding.integrated_confidence,
        outside_grounding:     ans.grounding.outside_grounding,
        speculation_flag:      ans.grounding.speculation_flag,
    })
}

async fn checkpoint(
    State(s): State<AppState>,
) -> Result<Json<CheckpointResponse>, (axum::http::StatusCode, String)> {
    let brain = s.brain.lock().await;
    brain.checkpoint(&s.checkpoint_path).map_err(|e| {
        (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
         format!("checkpoint failed: {}", e))
    })?;
    let bytes = std::fs::metadata(&s.checkpoint_path).map(|m| m.len()).unwrap_or(0);
    Ok(Json(CheckpointResponse {
        written_bytes: bytes,
        path:          s.checkpoint_path.display().to_string(),
    }))
}

// -----------------------------------------------------------------
// main
// -----------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let data = data_dir();
    std::fs::create_dir_all(&data).with_context(|| format!("mkdir {}", data.display()))?;
    let checkpoint_path = data.join("brain.bin");

    let (brain, _prefixes) = load_or_build_brain(&checkpoint_path)?;
    let bs = brain.stats();
    info!("brain ready  tick={}  pools={}  neurons={}  terminals={}",
        bs.tick, bs.pool_count, bs.total_neurons, bs.total_terminals);

    let state = AppState {
        brain:           Arc::new(Mutex::new(brain)),
        checkpoint_path: checkpoint_path.clone(),
    };

    let app = Router::new()
        .route("/health",     get(health))
        .route("/stats",      get(stats))
        .route("/observe",    post(observe))
        .route("/tick",       post(tick))
        .route("/integrate",  post(integrate))
        .route("/checkpoint", post(checkpoint))
        .with_state(state.clone());

    let port: u16 = std::env::var("W1Z4RD_BRAIN_PORT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8095);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("brain server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await
        .with_context(|| format!("bind {}", addr))?;

    // Graceful checkpoint on Ctrl-C.
    let shutdown_state = state.clone();
    let shutdown = async move {
        match tokio::signal::ctrl_c().await {
            Ok(_) => {
                info!("shutdown signal received; flushing checkpoint...");
                let brain = shutdown_state.brain.lock().await;
                if let Err(e) = brain.checkpoint(&shutdown_state.checkpoint_path) {
                    warn!("checkpoint on shutdown failed: {}", e);
                } else {
                    info!("checkpoint written to {}",
                        shutdown_state.checkpoint_path.display());
                }
            }
            Err(e) => warn!("ctrl_c handler error: {}", e),
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("axum serve")?;
    Ok(())
}
