//! Brain API — shared module used by BOTH the standalone brain_server
//! binary and the main node binary's `/brain/*` namespace.
//!
//! This is the Phase A–E substrate surface (Q→A database + consolidation
//! lock + soft domain gate + binding-concept shortcut + cross-domain
//! integrate_chain + self-tuning hill-climber + autonomous thinking loop)
//! exposed as an `axum::Router` that any caller can mount.
//!
//! The main node binary mounts this under `/brain/*` so the existing
//! node API stays intact and the new substrate sits alongside it
//! cleanly.  Anywhere both layers expose similar functionality, the
//! `/brain/*` route is the authoritative implementation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde_json::json;
use tokio::sync::Mutex;
use w1z4rd_brain::neuron::{NeuronId, NeuronRef, PoolId};
use w1z4rd_brain::pool::{AtomEncoding, BytePassthroughEncoding};
use w1z4rd_brain::{
    Brain, BrainConfig, BrainDeploymentSpec, BrainIdentitySpec, PoolConfig, PoolPrototypeRegistry,
};

// ---------------------------------------------------------------------
// Standard pool ids — must match brain_server.rs and any client script.
// ---------------------------------------------------------------------

pub const POOL_BINDING: PoolId = 0;
pub const POOL_TEXT: PoolId = 1;
pub const POOL_IMAGE: PoolId = 2;
pub const POOL_AUDIO: PoolId = 3;
pub const POOL_ACTION: PoolId = 4;
pub const POOL_TURN: PoolId = 5;

// ---------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------

/// Phase E continuous-thought controller state.  Atomics + std mutexes
/// for shared access from the background loop without holding the
/// brain lock.
#[derive(Debug)]
pub struct ThinkingState {
    pub enabled: AtomicBool,
    pub query_pool: AtomicU32,
    pub target_pool: AtomicU32,
    pub hops_taken: AtomicU64,
    pub last_seed: std::sync::Mutex<Option<Vec<u8>>>,
    pub last_answer: std::sync::Mutex<Option<Vec<u8>>>,
}

impl Default for ThinkingState {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            query_pool: AtomicU32::new(POOL_TEXT),
            target_pool: AtomicU32::new(POOL_ACTION),
            hops_taken: AtomicU64::new(0),
            last_seed: std::sync::Mutex::new(None),
            last_answer: std::sync::Mutex::new(None),
        }
    }
}

/// HTTP-layer cumulative timing for the two hot endpoints.  Lock-wait is
/// the time the handler spends `await`ing `brain.lock()` — the smoking
/// gun for the "background loops hold the mutex while the foreground
/// request piles up" hypothesis.  Handler-total is wall-clock from
/// entering the handler to returning.  Subtracting per-observe fabric
/// work from handler-total reveals serde/HTTP framing cost.
#[derive(Debug, Default)]
pub struct HttpProfile {
    pub observe_calls: AtomicU64,
    pub observe_lock_wait_ns: AtomicU64,
    pub observe_handler_ns: AtomicU64,
    pub tick_calls: AtomicU64,
    pub tick_lock_wait_ns: AtomicU64,
    pub tick_handler_ns: AtomicU64,
}

/// Router state passed to every brain handler.  Clone-friendly because
/// every field is Arc-backed.
#[derive(Clone)]
pub struct BrainApiState {
    pub brain: Arc<Mutex<Brain>>,
    pub thinking: Arc<ThinkingState>,
    pub http_profile: Arc<HttpProfile>,
}

// ---------------------------------------------------------------------
// Brain construction
// ---------------------------------------------------------------------

/// Build a fresh brain with the canonical five-pool topology used by
/// every Phase A–E test (binding/text/image/audio/action/turn).  These
/// pool configs are the empirically validated values from the main
/// session — same as `brain_server.rs::build_fresh_brain`.
pub fn build_default_brain() -> Result<Brain> {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);

    let mut text = PoolConfig::defaults("text", POOL_TEXT);
    text.recent_atoms_window = 65536;
    text.concept_emergence_threshold = 3;
    text.max_concept_member_count = 32;
    text.decay_rate = 0.00002;
    text.prune_floor = 0.001;
    brain.create_pool(text, leaked_encoding("t"));

    let mut image = PoolConfig::defaults("image", POOL_IMAGE);
    image.recent_atoms_window = 4096;
    image.concept_emergence_threshold = 3;
    image.max_concept_member_count = 32;
    image.decay_rate = 0.00002;
    image.prune_floor = 0.001;
    brain.create_pool(image, leaked_encoding("i"));

    let mut audio = PoolConfig::defaults("audio", POOL_AUDIO);
    audio.recent_atoms_window = 4096;
    audio.concept_emergence_threshold = 3;
    audio.max_concept_member_count = 32;
    audio.decay_rate = 0.00002;
    audio.prune_floor = 0.001;
    brain.create_pool(audio, leaked_encoding("a"));

    let mut action = PoolConfig::defaults("action", POOL_ACTION);
    action.recent_atoms_window = 65536;
    action.concept_emergence_threshold = 3;
    action.max_concept_member_count = 32;
    action.decay_rate = 0.00002;
    action.prune_floor = 0.001;
    brain.create_pool(action, leaked_encoding("act"));
    brain.designate_action_pool(POOL_ACTION);

    let mut turn = PoolConfig::defaults("turn", POOL_TURN);
    turn.recent_atoms_window = 32;
    turn.concept_emergence_threshold = u32::MAX;
    turn.max_concept_member_count = 4;
    turn.decay_rate = 0.001;
    turn.prune_floor = 0.01;
    brain.create_pool(turn, leaked_encoding("turn"));

    Ok(brain)
}

fn load_identity(path: &Path) -> Result<BrainIdentitySpec> {
    let identity = if path.extension().and_then(|v| v.to_str()) == Some("json") {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read brain identity {}: {}", path.display(), e))?;
        serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse brain identity {}: {}", path.display(), e))?
    } else {
        BrainIdentitySpec::load_toml(path)
            .map_err(|e| anyhow::anyhow!("load brain identity {}: {}", path.display(), e))?
    };
    Ok(identity)
}

/// Treat the encoding identity as durable brain metadata. A checkpoint owns
/// neurons but cannot reconstruct encoding trait objects, so a restart must
/// not depend solely on a process-local environment variable. The first
/// configured launch writes a canonical identity beside `brain.bin`; later
/// launches recover it automatically when the variable is absent.
fn resolve_identity(
    data_dir: &Path,
    configured_path: Option<&Path>,
) -> Result<Option<BrainIdentitySpec>> {
    let persisted = data_dir.join("brain.identity.toml");
    if let Some(path) = configured_path {
        let identity = load_identity(path)?;
        std::fs::create_dir_all(data_dir)
            .map_err(|e| anyhow::anyhow!("create brain data dir {}: {}", data_dir.display(), e))?;
        identity.save_toml(&persisted).map_err(|e| {
            anyhow::anyhow!("persist brain identity {}: {}", persisted.display(), e)
        })?;
        return Ok(Some(identity));
    }
    if persisted.exists() {
        return load_identity(&persisted).map(Some);
    }
    Ok(None)
}

fn configured_identity(data_dir: &Path) -> Result<Option<BrainIdentitySpec>> {
    let configured = std::env::var_os("W1Z4RD_BRAIN_IDENTITY").map(PathBuf::from);
    resolve_identity(data_dir, configured.as_deref())
}

fn configured_deployment() -> Result<Option<BrainDeploymentSpec>> {
    let Some(path) = std::env::var_os("W1Z4RD_BRAIN_DEPLOYMENT") else {
        return Ok(None);
    };
    let path = Path::new(&path);
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read brain deployment {}: {}", path.display(), e))?;
    let spec = if path.extension().and_then(|v| v.to_str()) == Some("json") {
        serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse brain deployment {}: {}", path.display(), e))?
    } else {
        BrainDeploymentSpec::load_toml(path)
            .map_err(|e| anyhow::anyhow!("parse brain deployment {}: {}", path.display(), e))?
    };
    Ok(Some(spec))
}

fn build_from_identity(identity: &BrainIdentitySpec) -> Result<Brain> {
    Brain::from_identity(identity, &PoolPrototypeRegistry::with_defaults())
        .map_err(|e| anyhow::anyhow!("build configured brain '{}': {}", identity.name, e))
}

/// Re-apply the deployed identity's operational pool configuration after a
/// checkpoint restore.  Checkpoints own learned neurons and terminals; the
/// identity file owns how those pools continue learning.  Without this step,
/// tuning an existing brain's decay, pruning, sparsity, or concept-emergence
/// policy silently had no effect until the brain was rebuilt from scratch.
fn apply_identity_pool_configs(brain: &mut Brain, identity: &BrainIdentitySpec) -> Result<()> {
    brain.set_min_atom_score(identity.min_atom_score);
    let registry = PoolPrototypeRegistry::with_defaults();
    for spec in &identity.pools {
        if let Some(pool) = brain.fabric().pool(spec.id) {
            pool.write().config = spec.to_pool_config();
            continue;
        }
        let encoding = registry
            .build(&spec.prototype, &spec.atom_encoding_prefix)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown pool prototype '{}' while adding configured pool {} ({})",
                    spec.prototype,
                    spec.id,
                    spec.name
                )
            })?;
        tracing::warn!(
            pool_id = spec.id,
            pool_name = %spec.name,
            "checkpoint lacks an identity pool; creating it empty"
        );
        brain.create_pool(spec.to_pool_config(), encoding);
    }
    Ok(())
}

fn leaked_encoding(prefix: &str) -> Box<dyn AtomEncoding> {
    let leaked: &'static str = Box::leak(prefix.to_string().into_boxed_str());
    Box::new(BytePassthroughEncoding { prefix: leaked })
}

/// Load a brain from `<data_dir>/brain.bin` if it exists, else build a
/// fresh one with the default topology. Replays the WAL tail after the most
/// recent snapshot marker, then attaches the same WAL for all future
/// mutations. Keeping this in the shared loader makes brain-only, embedded,
/// and merged-node modes obey the same durability contract.
pub fn load_or_build_brain(data_dir: &Path) -> Result<Brain> {
    let checkpoint = data_dir.join("brain.bin");
    let identity = configured_identity(data_dir)?;
    let mut brain = if checkpoint.exists() {
        let mut encs: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
        encs.insert(POOL_BINDING, leaked_encoding("bind"));
        if let Some(spec) = &identity {
            let registry = PoolPrototypeRegistry::with_defaults();
            for pool in &spec.pools {
                let encoding = registry
                    .build(&pool.prototype, &pool.atom_encoding_prefix)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown pool prototype '{}' while restoring pool {} ({})",
                            pool.prototype,
                            pool.id,
                            pool.name
                        )
                    })?;
                encs.insert(pool.id, encoding);
            }
        } else {
            encs.insert(POOL_TEXT, leaked_encoding("t"));
            encs.insert(POOL_IMAGE, leaked_encoding("i"));
            encs.insert(POOL_AUDIO, leaked_encoding("a"));
            encs.insert(POOL_ACTION, leaked_encoding("act"));
            encs.insert(POOL_TURN, leaked_encoding("turn"));
        }
        match Brain::restore(&checkpoint, encs) {
            Ok((_brain, missing)) if !missing.is_empty() => {
                anyhow::bail!(
                    "checkpoint {} requires encodings for missing pools {:?}; set W1Z4RD_BRAIN_IDENTITY to the brain's identity file",
                    checkpoint.display(),
                    missing
                );
            }
            Ok((mut brain, _missing)) => {
                if let Some(spec) = &identity {
                    apply_identity_pool_configs(&mut brain, spec)?;
                }
                brain
            }
            Err(e) => {
                tracing::warn!(
                    "brain restore failed at {}: {} — starting fresh",
                    checkpoint.display(),
                    e
                );
                match &identity {
                    Some(spec) => build_from_identity(spec)?,
                    None => build_default_brain()?,
                }
            }
        }
    } else {
        match &identity {
            Some(spec) => build_from_identity(spec)?,
            None => build_default_brain()?,
        }
    };
    // Recover mutations accepted after brain.bin's last SnapshotMarker
    // before attaching the live writer. Applying through the initial
    // NoopStore avoids echoing recovered events back into the WAL.
    match w1z4rd_brain::store::load_events_after_marker(data_dir) {
        Ok(events) if !events.is_empty() => {
            let stats = brain.apply_wal_events(&events);
            tracing::info!(
                events = stats.events_total,
                last_tick = stats.last_tick,
                events_since_snapshot = stats.events_since_snapshot,
                "replayed embedded-brain WAL tail"
            );
        }
        Ok(_) => tracing::info!("embedded-brain WAL has no post-snapshot events"),
        Err(error) => tracing::warn!(
            %error,
            data_dir = %data_dir.display(),
            "embedded-brain WAL replay failed; continuing from brain.bin"
        ),
    }
    match w1z4rd_brain::MmapWalStore::open(data_dir) {
        Ok(wal) => {
            let store: std::sync::Arc<dyn w1z4rd_brain::Store> =
                std::sync::Arc::new(wal);
            brain.set_store(store);
            tracing::info!(
                wal = %data_dir.join("brain.wal").display(),
                "attached embedded-brain WAL"
            );
        }
        Err(error) => tracing::warn!(
            %error,
            data_dir = %data_dir.display(),
            "embedded-brain WAL attach failed; checkpoint-only durability remains"
        ),
    }

    // Attach cold-tier files to every pool so the continuous tier
    // orchestrator can actually evict — without this, the orchestrator
    // sees `has_storage_tier()==false` and skips every pass, which
    // means RAM grows unbounded as neurons accumulate (the brain blew
    // up to 19 GB on the last run because of this).
    let n_attached = brain.attach_cold_tiers(data_dir);
    tracing::info!(
        "attached cold tiers to {} pools at {}",
        n_attached,
        data_dir.display()
    );
    if let (Some(identity), Some(deployment)) = (&identity, configured_deployment()?) {
        brain
            .configure_feedback_loops(identity, &deployment)
            .map_err(|e| anyhow::anyhow!("configure feedback loops: {}", e))?;
        tracing::info!(
            "configured {} online feedback loops",
            brain.feedback_loop_count()
        );
        // Enforce the deployment resource budget: translate max_resident_bytes
        // into the tier orchestrator's per-pool terminal target.  ~1000 bytes
        // per terminal is MEASURED, not theoretical: a 6.77M-terminal market
        // fabric held 6.4 GB resident (terminals + neurons + terminal_idx +
        // allocator slack + moment history).  Previously this field was
        // validated but never enforced — the spec promised a RAM budget the
        // brain ignored.  An explicit W1Z4RD_TIER_TARGET_TERMS env still wins.
        if std::env::var_os("W1Z4RD_TIER_TARGET_TERMS").is_none() {
            let budget = deployment.resource_budget.max_resident_bytes;
            if budget > 0 {
                let pools = identity.pools.len().max(1) as u64;
                let mut params = brain.fabric().orchestrator_params_snapshot();
                params.target_terminals_per_pool = ((budget / 1_000) / pools).max(10_000) as usize;
                brain.fabric().set_tier_orchestrator_params(params);
                tracing::info!(
                    "deployment resource budget {} bytes → {} terminals/pool eviction target",
                    budget,
                    params.target_terminals_per_pool
                );
            }
        }
    }
    Ok(brain)
}

/// Create the shared state used by the brain router.  Caller wraps
/// `Brain` so the state can be sent across tasks (router clone, etc.).
pub fn build_brain_api_state(brain: Brain) -> BrainApiState {
    BrainApiState {
        brain: Arc::new(Mutex::new(brain)),
        thinking: Arc::new(ThinkingState::default()),
        http_profile: Arc::new(HttpProfile::default()),
    }
}

// ---------------------------------------------------------------------
// Helpers — base64-url, response shapes
// ---------------------------------------------------------------------

fn b64_url_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    let s = s.trim();
    let pad = (4 - s.len() % 4) % 4;
    let padded = format!("{}{}", s, "=".repeat(pad));
    base64::engine::general_purpose::URL_SAFE
        .decode(padded.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(s.as_bytes()))
        .map_err(|e| e.to_string())
}

fn b64_url_no_pad(b: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

// ---------------------------------------------------------------------
// Handlers — Phase A–E surface, mirrored from brain_server.rs but
// implemented directly against BrainApiState so the main node binary
// can mount them.
// ---------------------------------------------------------------------

async fn h_health() -> &'static str {
    "ok\n"
}

async fn h_stats(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let b = s.brain.lock().await;
    let st = b.stats();
    Json(json!({
        "tick":            st.tick,
        "pool_count":      st.pool_count,
        "total_neurons":   st.total_neurons,
        "total_concepts":  st.total_concepts,
        "total_binding":   st.total_binding,
        // Backward-compatible name. This is the resident RAM working set,
        // not a structural-growth counter: prediction may page pre-existing
        // terminals back from SSD without learning anything.
        "total_terminals":    st.total_terminals,
        "resident_terminals": st.resident_terminals,
        "evicted_neurons":    st.evicted_neurons,
        "binding_pool_id": b.binding_pool_id(),
    }))
}

async fn h_observe(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let handler_t0 = std::time::Instant::now();
    let pool_id = req.get("pool_id").and_then(|v| v.as_u64()).unwrap_or(0) as PoolId;
    let frame_b64 = req.get("frame").and_then(|v| v.as_str()).unwrap_or("");
    let frame = match b64_url_decode(frame_b64) {
        Ok(b) => b,
        Err(e) => return Json(json!({"error": format!("bad frame base64: {}", e)})),
    };
    let lock_t0 = std::time::Instant::now();
    let mut brain = s.brain.lock().await;
    let lock_ns = lock_t0.elapsed().as_nanos() as u64;
    let fired = brain.observe(pool_id, &frame);
    drop(brain);
    let handler_ns = handler_t0.elapsed().as_nanos() as u64;
    s.http_profile.observe_calls.fetch_add(1, Ordering::Relaxed);
    s.http_profile
        .observe_lock_wait_ns
        .fetch_add(lock_ns, Ordering::Relaxed);
    s.http_profile
        .observe_handler_ns
        .fetch_add(handler_ns, Ordering::Relaxed);
    Json(json!({ "fired_count": fired.len() }))
}

/// Deterministic batch fast-forward for within-pool concept neurogenesis.
/// Input remains raw sensor frames; promoted concepts retain ordered links to
/// the original atom neurons and use the ordinary WAL/wiring path.
async fn h_pretrain(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let pool_id = req
        .get("pool_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_TEXT as u64) as PoolId;
    let min_recurrence = req
        .get("min_recurrence")
        .and_then(|v| v.as_u64())
        .unwrap_or(3)
        .clamp(2, u32::MAX as u64) as u32;
    let max_promotions = req
        .get("max_promotions")
        .and_then(|v| v.as_u64())
        .unwrap_or(1_024)
        .clamp(1, 10_000) as usize;
    let encoded: Vec<&str> = req
        .get("frames")
        .and_then(|v| v.as_array())
        .map(|frames| {
            frames
                .iter()
                .filter_map(|v| v.as_str())
                .take(4_096)
                .collect()
        })
        .unwrap_or_default();
    if encoded.is_empty() {
        return Json(json!({"error": "frames must contain at least one base64url frame"}));
    }
    let mut frames = Vec::with_capacity(encoded.len());
    for frame in encoded {
        match b64_url_decode(frame) {
            Ok(bytes) => frames.push(bytes),
            Err(error) => return Json(json!({"error": format!("bad frame base64: {}", error)})),
        }
    }

    let brain = s.brain.lock().await;
    let Some(pool) = brain.fabric().pool(pool_id) else {
        return Json(json!({"error": format!("unknown pool id {}", pool_id)}));
    };
    let report = pool.write().pretrain_recurring_patterns(
        &frames,
        brain.fabric().current_tick(),
        min_recurrence,
        max_promotions,
    );
    Json(json!({"pool_id": pool_id, "atom_grounded": true, "report": report}))
}

/// Activate several learned sensor streams in the same read-only inference
/// moment and decode only a binding supported by every supplied pool.
async fn h_predict_multi(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let target_pool = req
        .get("target_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_ACTION as u64) as PoolId;
    let Some(streams) = req.get("streams").and_then(|v| v.as_array()) else {
        return Json(json!({"error": "streams must be an array of {pool_id, frame}"}));
    };
    let mut decoded = Vec::new();
    for stream in streams.iter().take(64) {
        let Some(pool_id) = stream.get("pool_id").and_then(|v| v.as_u64()) else {
            return Json(json!({"error": "each stream requires pool_id"}));
        };
        let Some(frame) = stream.get("frame").and_then(|v| v.as_str()) else {
            return Json(json!({"error": "each stream requires a base64url frame"}));
        };
        match b64_url_decode(frame) {
            Ok(bytes) => decoded.push((pool_id as PoolId, bytes)),
            Err(error) => return Json(json!({"error": format!("bad frame base64: {}", error)})),
        }
    }
    if decoded.is_empty() {
        return Json(json!({"error": "at least one stream is required"}));
    }

    let mut brain = s.brain.lock().await;
    brain.clear_prediction_activation();
    let mut query_pools = Vec::with_capacity(decoded.len());
    for (pool_id, frame) in &decoded {
        if brain.fabric().pool(*pool_id).is_none() {
            brain.clear_prediction_activation();
            return Json(json!({"error": format!("unknown pool id {}", pool_id)}));
        }
        brain.activate_for_prediction(*pool_id, frame);
        query_pools.push(*pool_id);
    }
    let answer = brain.decode_best_trained_binding_multi(&query_pools, target_pool);
    brain.clear_prediction_activation();
    Json(json!({
        "integrated": true,
        "query_pools": query_pools,
        "target_pool": target_pool,
        "answer": answer.map(|bytes| b64_url_no_pad(&bytes)),
    }))
}

/// Select an abstract repair relation from the integrated neural moment, then
/// execute it against the current raw source. The neural fabric chooses the
/// transformation; deterministic composition preserves unseen identifiers.
async fn h_repair_predict(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let relation_pool = req
        .get("relation_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(11) as PoolId;
    let Some(source_b64) = req.get("source").and_then(|v| v.as_str()) else {
        return Json(json!({"error": "source must be a base64url frame"}));
    };
    let source_bytes = match b64_url_decode(source_b64) {
        Ok(value) => value,
        Err(error) => return Json(json!({"error": format!("bad source base64: {}", error)})),
    };
    let source = match std::str::from_utf8(&source_bytes) {
        Ok(value) => value,
        Err(error) => return Json(json!({"error": format!("source is not UTF-8: {}", error)})),
    };
    let Some(streams) = req.get("streams").and_then(|v| v.as_array()) else {
        return Json(json!({"error": "streams must be an array of {pool_id, frame}"}));
    };
    let mut decoded = Vec::new();
    for stream in streams.iter().take(64) {
        let Some(pool_id) = stream.get("pool_id").and_then(|v| v.as_u64()) else {
            return Json(json!({"error": "each stream requires pool_id"}));
        };
        let Some(frame) = stream.get("frame").and_then(|v| v.as_str()) else {
            return Json(json!({"error": "each stream requires a base64url frame"}));
        };
        match b64_url_decode(frame) {
            Ok(bytes) => decoded.push((pool_id as PoolId, bytes)),
            Err(error) => return Json(json!({"error": format!("bad frame base64: {}", error)})),
        }
    }
    if decoded.is_empty() {
        return Json(json!({"error": "at least one stream is required"}));
    }

    let mut brain = s.brain.lock().await;
    brain.clear_prediction_activation();
    let mut query_pools = Vec::new();
    for (pool_id, frame) in &decoded {
        if brain.fabric().pool(*pool_id).is_none() {
            brain.clear_prediction_activation();
            return Json(json!({"error": format!("unknown pool id {}", pool_id)}));
        }
        brain.activate_for_prediction(*pool_id, frame);
        query_pools.push(*pool_id);
    }
    let relation = brain.decode_best_trained_binding_multi(&query_pools, relation_pool);
    brain.clear_prediction_activation();
    let Some(relation) = relation else {
        return Json(json!({"integrated": true, "answer": null, "relation": null}));
    };
    match w1z4rd_brain::apply_code_repair_relation(source, &relation) {
        Ok(answer) => Json(json!({
            "integrated": true,
            "relation": b64_url_no_pad(&relation),
            "answer": b64_url_no_pad(answer.as_bytes()),
        })),
        Err(error) => Json(json!({
            "integrated": true,
            "relation": b64_url_no_pad(&relation),
            "answer": null,
            "composition_error": error.to_string(),
        })),
    }
}

async fn h_tick(State(s): State<BrainApiState>) -> Json<u64> {
    let handler_t0 = std::time::Instant::now();
    let lock_t0 = std::time::Instant::now();
    let mut brain = s.brain.lock().await;
    let lock_ns = lock_t0.elapsed().as_nanos() as u64;
    brain.advance_tick();
    let tick = brain.fabric().current_tick();
    drop(brain);
    let handler_ns = handler_t0.elapsed().as_nanos() as u64;
    s.http_profile.tick_calls.fetch_add(1, Ordering::Relaxed);
    s.http_profile
        .tick_lock_wait_ns
        .fetch_add(lock_ns, Ordering::Relaxed);
    s.http_profile
        .tick_handler_ns
        .fetch_add(handler_ns, Ordering::Relaxed);
    Json(tick)
}

async fn h_http_profile(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let obs_calls = s.http_profile.observe_calls.load(Ordering::Relaxed);
    let obs_lock = s.http_profile.observe_lock_wait_ns.load(Ordering::Relaxed);
    let obs_hand = s.http_profile.observe_handler_ns.load(Ordering::Relaxed);
    let tick_calls = s.http_profile.tick_calls.load(Ordering::Relaxed);
    let tick_lock = s.http_profile.tick_lock_wait_ns.load(Ordering::Relaxed);
    let tick_hand = s.http_profile.tick_handler_ns.load(Ordering::Relaxed);
    let mean = |ns: u64, n: u64| if n == 0 { 0 } else { (ns / n / 1_000) };
    Json(json!({
        "observe": {
            "calls":              obs_calls,
            "lock_wait_us_total": obs_lock / 1_000,
            "handler_us_total":   obs_hand / 1_000,
            "mean_lock_wait_us":  mean(obs_lock, obs_calls),
            "mean_handler_us":    mean(obs_hand, obs_calls),
            "lock_pct_of_handler": if obs_hand == 0 { 0.0 }
                else { (obs_lock as f64) * 100.0 / (obs_hand as f64) },
        },
        "tick": {
            "calls":              tick_calls,
            "lock_wait_us_total": tick_lock / 1_000,
            "handler_us_total":   tick_hand / 1_000,
            "mean_lock_wait_us":  mean(tick_lock, tick_calls),
            "mean_handler_us":    mean(tick_hand, tick_calls),
            "lock_pct_of_handler": if tick_hand == 0 { 0.0 }
                else { (tick_lock as f64) * 100.0 / (tick_hand as f64) },
        },
    }))
}

async fn h_set_domain(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let domain_id = req
        .get("domain_id")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
        .unwrap_or(0);
    let brain = s.brain.lock().await;
    brain.set_domain_for_new(domain_id);
    Json(json!({ "domain_for_new": domain_id }))
}

async fn h_domain_stats(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let hist = brain.domain_histogram();
    let entries: Vec<_> = hist
        .into_iter()
        .map(|((pool, domain), count)| json!({"pool": pool, "domain": domain, "count": count}))
        .collect();
    Json(json!({ "histogram": entries }))
}

async fn h_qa_db_stats(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    Json(json!({ "count": brain.qa_db().len(), "capacity": 4096 }))
}

async fn h_consolidation_stats(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    Json(json!({
        "locked_terminals": brain.locked_terminal_count(),
        "lock_threshold":   3u8,
        "tick_now":         brain.fabric().current_tick(),
    }))
}

async fn h_self_test(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let n = req
        .get("sample_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let mut brain = s.brain.lock().await;
    Json(json!(brain.self_test(n)))
}

async fn h_integrate(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let qp = req
        .get("query_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_TEXT as u64) as PoolId;
    let brain = s.brain.lock().await;
    let tp = req
        .get("target_pool")
        .and_then(|v| v.as_u64())
        .map(|v| v as PoolId)
        .or_else(|| brain.action_pool_id())
        .unwrap_or(POOL_ACTION);
    // Mirror brain_server's authoritative path: keep confidence/grounding
    // signals from integrate(), but prefer the decode_best_trained_binding
    // answer (which is what lifts paraphrase recall to 100% — the older
    // atom-coverage selection truncates / mis-routes when a single extra
    // atom is in the query frame).  Encode bytes as base64url so the
    // JSON wire format is consistent across all callers.
    let legacy = brain.integrate(qp, tp);
    let authoritative = brain.decode_best_trained_binding(qp, tp);
    let answer_bytes = authoritative.or(legacy.answer);
    let answer_b64 = answer_bytes.as_ref().map(|b| b64_url_no_pad(b));
    Json(json!({
        "answer":                answer_b64,
        "confidence_tier":       format!("{:?}", legacy.confidence_tier),
        "fabric_confidence":     legacy.grounding.fabric_confidence,
        "eem_confidence":        legacy.grounding.eem_confidence,
        "annealer_confidence":   legacy.grounding.annealer_confidence,
        "integrated_confidence": legacy.grounding.integrated_confidence,
        "outside_grounding":     legacy.grounding.outside_grounding,
        "speculation_flag":      legacy.grounding.speculation_flag,
    }))
}

/// Read-only prediction. Query activation is never admitted to the learning
/// moment and is cleared before releasing the brain lock.
async fn h_predict(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let qp = req
        .get("query_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_TEXT as u64) as PoolId;
    let frame = match b64_url_decode(req.get("frame").and_then(|v| v.as_str()).unwrap_or("")) {
        Ok(b) => b,
        Err(e) => return Json(json!({"error": format!("bad frame base64: {}", e)})),
    };
    let mut brain = s.brain.lock().await;
    let tp = req
        .get("target_pool")
        .and_then(|v| v.as_u64())
        .map(|v| v as PoolId)
        .or_else(|| brain.action_pool_id())
        .unwrap_or(POOL_ACTION);
    let fired = brain.activate_for_prediction(qp, &frame);
    let legacy = brain.integrate(qp, tp);
    let authoritative = brain.decode_best_trained_binding(qp, tp);
    let answer = authoritative.or(legacy.answer).map(|b| b64_url_no_pad(&b));
    brain.clear_prediction_activation();
    Json(json!({
        "answer": answer, "known_atom_count": fired.len(),
        "integrated_confidence": legacy.grounding.integrated_confidence,
        "outside_grounding": legacy.grounding.outside_grounding || fired.is_empty(),
        "speculation_flag": legacy.grounding.speculation_flag,
        "learning": false,
    }))
}

/// Politeness floor in MB (same env the tier orchestrator reads).
fn politeness_floor_mb() -> u64 {
    std::env::var("W1Z4RD_TIER_MIN_SYS_AVAIL_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_096)
}

/// Ingest backpressure: when the machine's available RAM is below the
/// politeness floor, training pushes must slow down — eviction cannot
/// outrun an unthrottled firehose (measured: 12.5 GB resident against an
/// 8 GB budget with the machine at 1.4 GB available).  Returns Some(reply)
/// when the caller should back off and retry.
fn ingest_backpressure() -> Option<Json<serde_json::Value>> {
    let floor = politeness_floor_mb();
    if floor == 0 {
        return None;
    }
    let avail = w1z4rd_brain::tier_orchestrator::TierOrchestrator::system_available_mb();
    if avail < floor {
        return Some(Json(json!({
            "consolidated": false,
            "backpressure": true,
            "available_mb": avail,
            "floor_mb": floor,
            "retry_after_ms": 2_000,
        })));
    }
    None
}

/// The only supervised hot-path operation that closes a Hebbian moment:
/// an input and its subsequently observed outcome are consolidated together.
async fn h_consolidate(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    if let Some(reply) = ingest_backpressure() {
        return reply;
    }
    let input_pool = req
        .get("input_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_TEXT as u64) as PoolId;
    let input = match b64_url_decode(
        req.get("input_frame")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    ) {
        Ok(b) => b,
        Err(e) => return Json(json!({"error": format!("bad input base64: {}", e)})),
    };
    let outcome = match b64_url_decode(
        req.get("outcome_frame")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    ) {
        Ok(b) => b,
        Err(e) => return Json(json!({"error": format!("bad outcome base64: {}", e)})),
    };
    let mut brain = s.brain.lock().await;
    let outcome_pool = req
        .get("outcome_pool")
        .and_then(|v| v.as_u64())
        .map(|v| v as PoolId)
        .or_else(|| brain.action_pool_id())
        .unwrap_or(POOL_ACTION);
    let input_fired = brain.observe(input_pool, &input).len();
    let outcome_fired = brain.observe(outcome_pool, &outcome).len();
    brain.advance_tick();
    Json(json!({"consolidated": true, "input_fired": input_fired,
                "outcome_fired": outcome_fired, "learning": true}))
}

/// Admit a semantic pathway only when the caller supplies an externally
/// confirmed outcome. Predictions cannot call this successfully by default.
async fn h_logic_consolidate(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    if req.get("outcome_confirmed").and_then(|v| v.as_bool()) != Some(true) {
        return Json(json!({"consolidated": false,
            "error": "outcome_confirmed=true is required"}));
    }
    let mut brain = s.brain.lock().await;
    if let Some(value) = req.get("relation") {
        match serde_json::from_value::<w1z4rd_brain::GroundedRelation>(value.clone()) {
            Ok(relation) => {
                brain.eem_mut().register_semantic_relation(relation);
                return Json(json!({"consolidated": true, "kind": "relation"}));
            }
            Err(e) => return Json(json!({"consolidated": false, "error": e.to_string()})),
        }
    }
    if let Some(value) = req.get("rule") {
        match serde_json::from_value::<w1z4rd_brain::CompositionRule>(value.clone()) {
            Ok(rule) => {
                brain.eem_mut().register_composition_rule(rule);
                return Json(json!({"consolidated": true, "kind": "rule"}));
            }
            Err(e) => return Json(json!({"consolidated": false, "error": e.to_string()})),
        }
    }
    Json(json!({"consolidated": false, "error": "relation or rule is required"}))
}

/// Resolve confirmed logical pathways in a disposable, read-only workspace.
async fn h_logic_compose(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let rounds = req
        .get("max_rounds")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .min(64) as usize;
    let predicate = req.get("predicate").and_then(|v| v.as_str());
    let brain = s.brain.lock().await;
    let workspace = brain.eem().compose_transient(rounds);
    let facts: Vec<_> = workspace
        .facts()
        .iter()
        .filter(|fact| predicate.map_or(true, |p| fact.predicate == p))
        .cloned()
        .collect();
    Json(json!({"learning": false, "facts": facts,
                "semantic_relation_count": brain.eem().semantic_relation_count(),
                "composition_rule_count": brain.eem().composition_rule_count()}))
}

/// Learn invariant structure and variable roles only from confirmed frames.
async fn h_logic_crystallize(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    if req.get("outcome_confirmed").and_then(|v| v.as_bool()) != Some(true) {
        return Json(json!({"consolidated": false,
            "error": "outcome_confirmed=true is required"}));
    }
    let frame = match req
        .get("frame")
        .cloned()
        .map(serde_json::from_value::<w1z4rd_brain::SemanticFrame>)
    {
        Some(Ok(frame)) => frame,
        Some(Err(e)) => return Json(json!({"consolidated": false, "error": e.to_string()})),
        None => return Json(json!({"consolidated": false, "error": "frame is required"})),
    };
    let mut brain = s.brain.lock().await;
    let relations = brain.eem_mut().consolidate_semantic_frame(frame);
    Json(json!({"consolidated": true, "relations": relations,
                "template_count": brain.eem().semantic_template_count()}))
}

/// Recognize roles in novel frames and compose them against durable EEM
/// pathways without changing either the crystallizer or the brain.
async fn h_logic_recognize(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let frames = match req
        .get("frames")
        .cloned()
        .map(serde_json::from_value::<Vec<w1z4rd_brain::SemanticFrame>>)
    {
        Some(Ok(frames)) => frames,
        Some(Err(e)) => return Json(json!({"learning": false, "error": e.to_string()})),
        None => return Json(json!({"learning": false, "error": "frames are required"})),
    };
    let rounds = req
        .get("max_rounds")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .min(64) as usize;
    let predicate = req.get("predicate").and_then(|v| v.as_str());
    let brain = s.brain.lock().await;
    let recognized: Vec<_> = frames
        .iter()
        .flat_map(|frame| brain.eem().recognize_semantic_frame(frame))
        .collect();
    let workspace = brain.eem().compose_with_transient(recognized, rounds);
    let facts: Vec<_> = workspace
        .facts()
        .iter()
        .filter(|fact| predicate.map_or(true, |p| fact.predicate == p))
        .cloned()
        .collect();
    Json(json!({"learning": false, "facts": facts,
                "template_count": brain.eem().semantic_template_count()}))
}

/// Union files from independently grounded action manifests. Conflicting
/// filenames abort composition rather than guessing which implementation wins.
fn merge_grounded_file_manifests(candidates: &[Vec<u8>]) -> Option<Vec<u8>> {
    let mut files = serde_json::Map::new();
    let mut manifests = 0usize;
    for bytes in candidates {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            continue;
        };
        let Some(candidate_files) = value.get("files").and_then(|files| files.as_object()) else {
            continue;
        };
        if candidate_files.is_empty() {
            continue;
        }
        let mut accepted = false;
        for (name, content) in candidate_files {
            if !content.is_string() {
                return None;
            }
            if let Some(existing) = files.get(name) {
                if existing != content {
                    return None;
                }
            } else {
                files.insert(name.clone(), content.clone());
                accepted = true;
            }
        }
        if accepted {
            manifests += 1;
        }
    }
    if manifests < 2 || files.len() < 2 {
        return None;
    }
    serde_json::to_vec(&serde_json::json!({"files": files})).ok()
}

/// Direct cross-pool corpus episode formation. Frames are atomized by each
/// pool's native encoder and remain lossless binding members, but bypass
/// ordinary per-frame concept emergence and all-to-all moment wiring.
async fn h_pretrain_binding(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let Some(items) = req.get("frames").and_then(|value| value.as_array()) else {
        return Json(json!({"error": "frames must be an array"}));
    };
    let mut frames = Vec::with_capacity(items.len());
    for item in items {
        let Some(pool_id) = item.get("pool_id").and_then(|value| value.as_u64()) else {
            return Json(json!({"error": "each frame requires pool_id"}));
        };
        let encoded = item.get("frame").and_then(|value| value.as_str()).unwrap_or("");
        let frame = match b64_url_decode(encoded) {
            Ok(frame) => frame,
            Err(error) => return Json(json!({"error": format!("bad frame base64: {}", error)})),
        };
        frames.push((pool_id as PoolId, frame));
    }
    let mut brain = s.brain.lock().await;
    let binding_id = brain.pretrain_binding_episode(&frames);
    if binding_id.is_some() {
        if let Err(error) = brain.store_clone().flush() {
            return Json(json!({"error": format!("WAL flush failed: {}", error)}));
        }
    }
    Json(json!({
        "ok": binding_id.is_some(),
        "binding_id": binding_id,
        "tick_now": brain.fabric().current_tick(),
        "frame_count": frames.len(),
    }))
}

/// Bounded bulk form of `h_pretrain_binding`. Holding the brain lock across a
/// small group removes HTTP and mutex overhead while preserving one tick and
/// one independently addressable binding per episode.
async fn h_pretrain_bindings(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let Some(episodes) = req.get("episodes").and_then(|value| value.as_array()) else {
        return Json(json!({"error": "episodes must be an array"}));
    };
    if episodes.is_empty() || episodes.len() > 256 {
        return Json(json!({"error": "episodes must contain 1..=256 items"}));
    }
    let mut decoded = Vec::with_capacity(episodes.len());
    for episode in episodes {
        let Some(items) = episode.get("frames").and_then(|value| value.as_array()) else {
            return Json(json!({"error": "each episode requires a frames array"}));
        };
        let mut frames = Vec::with_capacity(items.len());
        for item in items {
            let Some(pool_id) = item.get("pool_id").and_then(|value| value.as_u64()) else {
                return Json(json!({"error": "each frame requires pool_id"}));
            };
            let encoded = item.get("frame").and_then(|value| value.as_str()).unwrap_or("");
            let frame = match b64_url_decode(encoded) {
                Ok(frame) => frame,
                Err(error) => {
                    return Json(json!({"error": format!("bad frame base64: {}", error)}));
                }
            };
            frames.push((pool_id as PoolId, frame));
        }
        decoded.push(frames);
    }
    let mut brain = s.brain.lock().await;
    let binding_ids: Vec<_> = decoded
        .iter()
        .map(|frames| brain.pretrain_binding_episode(frames))
        .collect();
    let accepted = binding_ids.iter().filter(|id| id.is_some()).count();
    if accepted > 0 {
        if let Err(error) = brain.store_clone().flush() {
            return Json(json!({"error": format!("WAL flush failed: {}", error)}));
        }
    }
    Json(json!({
        "ok": accepted == binding_ids.len(),
        "accepted": accepted,
        "binding_ids": binding_ids,
        "tick_now": brain.fabric().current_tick(),
    }))
}

/// A directly grounded, complete project answer is stronger evidence than a
/// set of lower-level fragments that happen to share some broad features.
/// This prevents a learned whole artifact from being shadowed by a fragment
/// during paraphrase recall.
fn is_complete_file_manifest(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| value.get("files").and_then(|files| files.as_object()).cloned())
        .is_some_and(|files| {
            !files.is_empty()
                && files.iter().all(|(name, content)| {
                    !name.is_empty()
                        && !name.starts_with('/')
                        && !name.starts_with('\\')
                        && !name.split(['/', '\\']).any(|part| part == "..")
                        && content.as_str().is_some_and(|source| !source.is_empty())
                })
        })
}

/// Recover independently learned whole-project components from small exact
/// feature subsets inside a richer query. This is deliberately restricted to
/// complete safe manifests; repair fragments continue through the
/// outcome-weighted ranked decoder so failed actions cannot re-enter through
/// this path.
fn exact_manifest_subset_candidates(
    brain: &Brain,
    feature_pool: PoolId,
    labels: &[String],
    target_pool: PoolId,
) -> Vec<Vec<u8>> {
    if labels.len() < 4 {
        return Vec::new();
    }
    fn visit(
        brain: &Brain,
        feature_pool: PoolId,
        labels: &[String],
        target_pool: PoolId,
        size: usize,
        start: usize,
        selected: &mut Vec<String>,
        output: &mut Vec<Vec<u8>>,
    ) {
        if selected.len() == size {
            let decoded = brain.decode_exact_feature_binding(feature_pool, selected, target_pool);
            if let Some(bytes) = decoded {
                if is_complete_file_manifest(&bytes) && !output.contains(&bytes) {
                    output.push(bytes);
                }
            }
            return;
        }
        let remaining = size - selected.len();
        for index in start..=labels.len().saturating_sub(remaining) {
            selected.push(labels[index].clone());
            visit(
                brain,
                feature_pool,
                labels,
                target_pool,
                size,
                index + 1,
                selected,
                output,
            );
            selected.pop();
        }
    }
    let mut output = Vec::new();
    for size in 2..=3.min(labels.len()) {
        visit(
            brain,
            feature_pool,
            labels,
            target_pool,
            size,
            0,
            &mut Vec::new(),
            &mut output,
        );
    }
    output
}

/// Assemble independently grounded raw-source fragments into files. The
/// protocol carries only deterministic structural constraints; source remains
/// byte-atom learned evidence and is never invented by this function.
/// New curricula express order as role dependencies; legacy numeric slots
/// remain readable for checkpoint compatibility. Conflicts, missing
/// dependencies, and cycles abort composition rather than guessing.
fn merge_grounded_code_fragments(candidates: &[Vec<u8>]) -> Option<Vec<u8>> {
    #[derive(Clone, PartialEq, Eq)]
    struct RelativeFragment {
        source: String,
        after: std::collections::BTreeSet<String>,
    }
    let mut numeric: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<i64, String>,
    > = std::collections::BTreeMap::new();
    let mut relative: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, RelativeFragment>,
    > = std::collections::BTreeMap::new();
    let mut numeric_count = 0usize;
    let mut relative_count = 0usize;
    let mut outcomes: std::collections::BTreeMap<String, bool> =
        std::collections::BTreeMap::new();
    for bytes in candidates {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            continue;
        };
        let Some(outcome) = value.get("fragment_outcome").and_then(|v| v.as_object()) else {
            continue;
        };
        let (Some(evidence_id), Some(confirmed)) = (
            outcome.get("evidence_id").and_then(|v| v.as_str()),
            outcome.get("confirmed").and_then(|v| v.as_bool()),
        ) else {
            return None;
        };
        if evidence_id.is_empty() {
            return None;
        }
        if outcomes.insert(evidence_id.to_string(), confirmed).is_some() {
            return None; // contradictory/repeated control evidence is ambiguous
        }
    }
    for bytes in candidates {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            continue;
        };
        let Some(fragment) = value.get("code_fragment").and_then(|v| v.as_object()) else {
            continue;
        };
        let (Some(file), Some(source)) = (
            fragment.get("file").and_then(|v| v.as_str()),
            fragment.get("source").and_then(|v| v.as_str()),
        ) else {
            return None;
        };
        if file.is_empty()
            || file.starts_with('/')
            || file.starts_with('\\')
            || file.split(['/', '\\']).any(|part| part == "..")
            || source.is_empty()
        {
            return None;
        }
        if let Some(evidence_id) = fragment.get("evidence_id").and_then(|v| v.as_str()) {
            if outcomes.get(evidence_id) == Some(&false) {
                continue;
            }
        }
        if let Some(order) = fragment.get("order").and_then(|v| v.as_i64()) {
            let slots = numeric.entry(file.to_string()).or_default();
            if let Some(existing) = slots.get(&order) {
                if existing != source {
                    return None;
                }
                continue;
            }
            slots.insert(order, source.to_string());
            numeric_count += 1;
            continue;
        }
        let Some(role) = fragment.get("role").and_then(|v| v.as_str()) else {
            return None;
        };
        if role.is_empty()
            || !role
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | ':'))
        {
            return None;
        }
        let after_values = fragment
            .get("after")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut after = std::collections::BTreeSet::new();
        for value in after_values {
            let Some(dependency) = value.as_str() else {
                return None;
            };
            if dependency == role || dependency.is_empty() {
                return None;
            }
            after.insert(dependency.to_string());
        }
        let entry = RelativeFragment {
            source: source.to_string(),
            after,
        };
        let roles = relative.entry(file.to_string()).or_default();
        if let Some(existing) = roles.get(role) {
            if existing != &entry {
                return None;
            }
            continue;
        }
        roles.insert(role.to_string(), entry);
        relative_count += 1;
    }
    if relative_count >= 2 && numeric_count > 0 {
        return None; // one artifact must use one ordering contract
    }
    if relative_count < 2 && numeric_count < 2 {
        return None;
    }
    let mut rendered = serde_json::Map::new();
    if relative_count >= 2 {
        let mut graph: std::collections::BTreeMap<
            String,
            (String, String, std::collections::BTreeSet<String>),
        > = std::collections::BTreeMap::new();
        for (file, roles) in relative {
            for (role, fragment) in roles {
                let key = format!("{file}::{role}");
                let dependencies = fragment
                    .after
                    .into_iter()
                    .map(|dependency| {
                        if dependency.contains("::") {
                            dependency
                        } else {
                            format!("{file}::{dependency}")
                        }
                    })
                    .collect();
                graph.insert(key, (file.clone(), fragment.source, dependencies));
            }
        }
        let all_roles: std::collections::BTreeSet<String> = graph.keys().cloned().collect();
        if graph
            .values()
            .any(|(_, _, dependencies)| !dependencies.is_subset(&all_roles))
        {
            return None;
        }
        let mut emitted = std::collections::BTreeSet::new();
        let mut file_sources: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        while emitted.len() < graph.len() {
            let next = graph
                .iter()
                .find(|(key, (_, _, dependencies))| {
                    !emitted.contains(*key) && dependencies.is_subset(&emitted)
                })
                .map(|(key, (file, source, _))| (key.clone(), file.clone(), source.clone()));
            let Some((key, file, fragment_source)) = next else {
                return None;
            };
            file_sources.entry(file).or_default().push_str(&fragment_source);
            emitted.insert(key);
        }
        for (file, source) in file_sources {
            rendered.insert(file, serde_json::Value::String(source));
        }
    } else {
        for (file, slots) in numeric {
            rendered.insert(
                file,
                serde_json::Value::String(slots.into_values().collect()),
            );
        }
    }
    serde_json::to_vec(&serde_json::json!({"files": rendered})).ok()
}

/// POST /brain/chat — canonical chat endpoint on the merged node.
/// Mirrors brain_server's /chat behaviour so existing Django /
/// wizard_session callers can switch from `:8095/chat` to
/// `:8090/brain/chat` without code changes to the response shape.
///
/// Pipeline:
///   1. Observe the prompt into POOL_TEXT (same prompt-unwrap as
///      brain_server applies for Wizard-chat context boilerplate).
///   2. PRIMARY: decode_best_trained_binding(POOL_TEXT, POOL_ACTION)
///      — authoritative trained-pair recall via the Phase B v2
///      binding shortcut.
///   3. SECONDARY: integrate_autonomous(POOL_TEXT, POOL_ACTION) —
///      engages EEM chain_explore + annealer ranking + multi-fact
///      assembly for cross-domain composition.
///   4. Response shape: { reply, answer, decoder, predictions,
///      grounding, activated_concepts, word_activations } —
///      identical to brain_server.
async fn h_brain_chat(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let text = req.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let prompt = unwrap_wizard_prompt(text);

    let mut brain = s.brain.lock().await;
    let action_pool = brain.action_pool_id().unwrap_or(POOL_ACTION);
    brain.activate_for_prediction(POOL_TEXT, prompt.as_bytes());

    // Exact ordered sensory evidence can be established from the raw pool
    // immediately. Do this before semantic-routing diagnostics: those routes
    // scan broad feature-binding populations and made known-answer latency
    // grow with total curriculum size even though the exact binding index had
    // already reduced retrieval itself to O(1)-style lookup.
    let raw_is_exact = brain.has_exact_trained_binding(POOL_TEXT, action_pool);
    let exact_raw_trained = raw_is_exact
        .then(|| brain.decode_best_trained_binding(POOL_TEXT, action_pool))
        .flatten();

    // Parallel instruction feature pools participate in ordinary chat when
    // the deployed identity provides them. Raw POOL_TEXT remains present in
    // every query; derived pools add sparse intent evidence but never replace
    // the character substrate. Unknown intent atoms produce no activation and
    // therefore leave the legacy single-pool path untouched.
    let feature_pools: Vec<PoolId> = if raw_is_exact {
        Vec::new()
    } else {
        brain
            .fabric()
            .pool_ids()
            .into_iter()
            .filter(|pid| *pid != POOL_TEXT && *pid != action_pool)
            .filter(|pid| {
                brain
                    .fabric()
                    .pool(*pid)
                    .is_some_and(|pool| pool.read().encoding_name() == "instruction-intent")
            })
            .collect()
    };
    let mut chat_query_pools = vec![POOL_TEXT];
    let mut composition_features: Option<(PoolId, Vec<String>)> = None;
    let mut semantic_refinement_score: Option<f32> = None;
    let mut semantic_refinement_margin: Option<f32> = None;
    let mut inhibited_feature_pools = std::collections::HashSet::new();
    let mut directly_underspecified = false;
    for pool_id in feature_pools.iter().copied() {
        let mut labels = brain
            .fabric()
            .pool(pool_id)
            .map(|pool| pool.read().encoded_labels(prompt.as_bytes()))
            .unwrap_or_default();
        let inhibits_derived_readout = labels
            .iter()
            .any(|label| label.ends_with(":GROUNDING:UNDERSPECIFIED"));
        if inhibits_derived_readout {
            directly_underspecified = true;
            inhibited_feature_pools.insert(pool_id);
            continue;
        }
        let learned_route = brain.decode_best_binding_by_char_motifs_with_margin(
            POOL_TEXT,
            prompt.as_bytes(),
            pool_id,
            0.0,
            0.0,
        );
        if let Some((_, score, margin)) = learned_route.as_ref() {
            semantic_refinement_score = Some(*score);
            semantic_refinement_margin = Some(*margin);
        }
        let learned_frame = learned_route.as_ref().map(|(bytes, _, _)| bytes.as_slice());
        let mut effective_owned: Option<Vec<u8>> = None;
        if let Some(frame) = learned_frame {
            let learned_labels = brain
                .fabric()
                .pool(pool_id)
                .map(|pool| pool.read().encoded_labels(frame))
                .unwrap_or_default();
            let removes_one_spurious_diagnostic = learned_labels.len() == 2
                && labels.len() == 3
                && learned_labels.iter().all(|label| labels.contains(label));
            let route_score = learned_route
                .as_ref()
                .map(|(_, score, _)| *score)
                .unwrap_or(0.0);
            let route_margin = learned_route
                .as_ref()
                .map(|(_, _, margin)| *margin)
                .unwrap_or(0.0);
            let reliable_removal = route_score >= 0.39
                && route_margin >= 0.025
                && removes_one_spurious_diagnostic;
            if reliable_removal
                && !learned_labels
                    .iter()
                    .any(|label| label.ends_with(":GROUNDING:UNDERSPECIFIED"))
            {
                labels = learned_labels;
                effective_owned = Some(frame.to_vec());
            }
        }
        let effective_frame = effective_owned.as_deref().unwrap_or(prompt.as_bytes());
        // A lone diagnostic (most commonly only LANGUAGE:PYTHON) is too
        // broad to establish task grounding. Require a co-firing composition
        // such as LANGUAGE + BEHAVIOR before derived evidence may influence
        // readout. Raw characters still activate regardless of this gate.
        if brain
            .activate_for_prediction(pool_id, effective_frame)
            .len()
            >= 2
        {
            chat_query_pools.push(pool_id);
            // Language plus at least two independently grounded behaviors is
            // the minimal evidence for artifact composition.
            if labels.len() >= 2 {
                composition_features = Some((pool_id, labels));
            }
        }
    }

    // Learned semantic route: raw character atoms may have been co-trained
    // with sparse intent neurons even when the surface phrase contains none
    // of the hand-authored diagnostics. Decode that internal intent frame,
    // then re-stimulate the same feature pool used by grounded code actions.
    for pool_id in feature_pools.iter().copied() {
        if chat_query_pools.contains(&pool_id) || inhibited_feature_pools.contains(&pool_id) {
            continue;
        }
        let intent_frame = brain
            .decode_best_binding_by_char_motifs_with_margin(
                POOL_TEXT,
                prompt.as_bytes(),
                pool_id,
                0.275,
                0.025,
            )
            .map(|(bytes, _, _)| bytes);
        let Some(intent_frame) = intent_frame else {
            continue;
        };
        let labels = brain
            .fabric()
            .pool(pool_id)
            .map(|pool| pool.read().encoded_labels(&intent_frame))
            .unwrap_or_default();
        if labels
            .iter()
            .any(|label| label.ends_with(":GROUNDING:UNDERSPECIFIED"))
        {
            continue;
        }
        if brain.activate_for_prediction(pool_id, &intent_frame).len() >= 2 {
            chat_query_pools.push(pool_id);
            if labels.len() >= 2 {
                composition_features = Some((pool_id, labels));
            }
        }
    }

    // Authoritative trained-binding decode — Phase B v2. Pool ids are
    // identity-specific: discover conversational turn pools by role rather
    // than treating the default topology's pool 5 as universal.
    let turn_pools: Vec<PoolId> = brain
        .fabric()
        .pool_ids()
        .into_iter()
        .filter(|pool_id| {
            brain
                .fabric()
                .pool(*pool_id)
                .is_some_and(|pool| pool.read().name() == "turn")
        })
        .collect();
    let raw_trained = exact_raw_trained
        .or_else(|| {
            brain.decode_best_trained_binding_with_context(
                POOL_TEXT,
                action_pool,
                &chat_query_pools,
                &turn_pools,
            )
        });
    let mut feature_candidates = composition_features
        .as_ref()
        .map(|(pool_id, labels)| {
            brain.decode_ranked_feature_bindings_with_context(
                *pool_id,
                labels,
                action_pool,
                64,
                brain.fabric().pool(8).map(|_| 8),
                brain.fabric().pool(6).map(|_| 6),
                &chat_query_pools,
                &turn_pools,
            )
        })
        .unwrap_or_default();
    if let Some((pool_id, labels)) = composition_features.as_ref() {
        for candidate in
            exact_manifest_subset_candidates(&brain, *pool_id, labels, action_pool)
        {
            if !feature_candidates.contains(&candidate) {
                feature_candidates.push(candidate);
            }
        }
        // Raw characters and sparse diagnostics are independent evidence
        // pools. A complete safe manifest recalled by the raw pathway may be
        // one component of a richer feature-composed project, so let it join
        // the candidate set instead of using it only as a final fallback.
        if labels.len() >= 4 {
            if let Some(candidate) = raw_trained
                .as_ref()
                .filter(|bytes| is_complete_file_manifest(bytes))
            {
                if !feature_candidates.contains(candidate) {
                    feature_candidates.push(candidate.clone());
                }
            }
        }
    }
    let diagnostic_unweighted_candidates = composition_features
        .as_ref()
        .map(|(pool_id, labels)| {
            brain
                .decode_ranked_feature_bindings_with_context(
                    *pool_id,
                    labels,
                    action_pool,
                    64,
                    None,
                    None,
                    &chat_query_pools,
                    &turn_pools,
                )
                .len()
        })
        .unwrap_or(0);
    let exact_feature = composition_features.as_ref().and_then(|(pool_id, labels)| {
        brain.decode_exact_feature_binding(*pool_id, labels, action_pool)
    });
    let exact_complete_manifest = exact_feature
        .as_ref()
        .filter(|bytes| is_complete_file_manifest(bytes))
        .cloned();
    let diagnostic_intent_labels = composition_features
        .as_ref()
        .map(|(_, labels)| labels.clone())
        .unwrap_or_default();
    let diagnostic_exact_feature = exact_feature.is_some();
    let diagnostic_exact_manifest = exact_complete_manifest.is_some();
    let composed = merge_grounded_code_fragments(&feature_candidates)
        .or_else(|| merge_grounded_file_manifests(&feature_candidates));
    let exact_is_composition_prerequisite = exact_feature.as_ref().is_some_and(|exact| {
        exact_fragment_has_grounded_dependents(exact, &feature_candidates)
    });
    let trained_bytes = if raw_is_exact && raw_trained.is_some() {
        // Direct sensory evidence is the strongest tier. Derived diagnostic
        // pools may compose novel requests, but can never overwrite an
        // ordered prompt episode the brain actually observed.
        raw_trained.clone()
    } else if exact_complete_manifest.is_some() {
        exact_complete_manifest
    } else if exact_is_composition_prerequisite && composed.is_some() {
        composed
    } else if exact_feature.is_some() {
        // An exact sparse-intent episode is stronger evidence than a fuzzy
        // assembly of several partially matching artifacts.  In particular,
        // LANGUAGE + BEHAVIOR can identify a learned single-function answer
        // exactly while broad project fragments share enough diagnostics to
        // form a syntactically valid but unrelated composition.
        exact_feature
    } else if composed.is_some() {
        composed
    } else if raw_trained.is_some() {
        raw_trained
    } else if chat_query_pools.len() > 1 {
        brain.decode_best_trained_binding_multi(&chat_query_pools, action_pool)
    } else {
        None
    };
    let trained_decode: Option<String> =
        trained_bytes.map(|b| String::from_utf8_lossy(&b).into_owned());

    // Autonomous propagation is a fallback, not a second mandatory decode.
    // An exact trained binding is already atom-grounded evidence. Running a
    // full-fabric propagation after finding it made every known prompt pay an
    // O(total terminals) cost and rendered broad curricula unusable.
    let has_compositional_evidence = chat_query_pools.len() > 1
        || composition_features.is_some()
        || !feature_candidates.is_empty();
    let xpool = if trained_decode.as_ref().is_some_and(|s| !s.is_empty())
        || !has_compositional_evidence
    {
        None
    } else {
        Some(brain.integrate_autonomous(
            POOL_TEXT,
            action_pool,
            /*fabric_threshold*/ 0.0,
            /*chain_max_depth*/ 4,
            /*chain_max_visit*/ 200,
        ))
    };
    let xpool_reply: Option<String> = xpool.as_ref().and_then(|result| {
        if result.grounding.outside_grounding {
            None
        } else {
            result
                .answer
                .as_ref()
                .map(|b| String::from_utf8_lossy(b).into_owned())
        }
    });

    let reply = if directly_underspecified {
        // Explicit missing-information evidence inhibits every generative
        // route, including raw-character fuzzy recall.
        String::new()
    } else if let Some(td) = trained_decode.as_ref().filter(|s| !s.is_empty()) {
        td.clone()
    } else if xpool.as_ref().is_some_and(|result| {
        !result.grounding.outside_grounding
            && !result.grounding.speculation_flag
            && result.grounding.integrated_confidence >= 0.30
            && result.grounding.composition_used.len() >= 2
    })
    {
        // Novel prompts may compose an answer through multiple independently
        // learned pathways. This activation is transient; it is not a new
        // binding until an external outcome later confirms it.
        xpool_reply.clone().unwrap_or_default()
    } else {
        // A single weak path is not sufficient evidence: remain OOV-honest.
        String::new()
    };

    let outside_grounding = reply.is_empty()
        || (trained_decode.is_none()
            && xpool
                .as_ref()
                .is_none_or(|result| result.grounding.outside_grounding));

    let decoder = if trained_decode.as_ref().is_some_and(|s| !s.is_empty()) {
        "trained_binding"
    } else if xpool_reply.as_deref().is_some_and(|a| !a.is_empty()) {
        let result = xpool.as_ref().expect("xpool reply requires integration result");
        if result.grounding.eem_confidence.is_some()
            && result.grounding.fabric_confidence < 0.3
        {
            "eem"
        } else {
            "multi_pool"
        }
    } else {
        "char_chain"
    }
    .to_string();

    let activated: Vec<String> = xpool
        .as_ref()
        .map(|result| {
            result
                .grounding
                .composition_used
                .iter()
                .filter_map(|nref| {
                    brain
                        .fabric()
                        .pool(nref.pool)
                        .and_then(|p| p.read().get(nref.neuron).map(|n| n.label.clone()))
                })
                .collect()
        })
        .unwrap_or_default();
    let fabric_confidence = xpool
        .as_ref()
        .map_or(if trained_decode.is_some() { 1.0 } else { 0.0 }, |result| {
            result.grounding.fabric_confidence
        });
    let integrated_confidence = xpool
        .as_ref()
        .map_or(if trained_decode.is_some() { 1.0 } else { 0.0 }, |result| {
            result.grounding.integrated_confidence
        });
    let speculation_flag = xpool
        .as_ref()
        .is_some_and(|result| result.grounding.speculation_flag);
    brain.clear_prediction_activation();

    Json(json!({
        "reply":              reply,
        "answer":             reply,
        "decoder":            decoder,
        "predictions":        serde_json::Map::new(),
        "grounding": {
            "fabric_confidence":     fabric_confidence,
            "integrated_confidence": integrated_confidence,
            "outside_grounding":     outside_grounding,
            "speculation_flag":      speculation_flag,
        },
        "activated_concepts": activated,
        "word_activations":   Vec::<serde_json::Value>::new(),
        "semantic_refinement_score": semantic_refinement_score,
        "semantic_refinement_margin": semantic_refinement_margin,
        "intent_diagnostics": {
            "labels": diagnostic_intent_labels,
            "ranked_candidates": feature_candidates.len(),
            "unweighted_candidates": diagnostic_unweighted_candidates,
            "exact_feature": diagnostic_exact_feature,
            "exact_complete_manifest": diagnostic_exact_manifest,
        },
    }))
}

/// Strip Wizard-chat context-wrapper boilerplate so the brain only
/// observes the actual question.  The Django frontend prepends a
/// rolling context blob + "[Now answer concisely]\n<question>" cue;
/// without unwrapping we'd train against the boilerplate atoms more
/// than the real question.  Mirrors `brain_server::unwrap_wizard_prompt`.
fn unwrap_wizard_prompt(text: &str) -> &str {
    if let Some(idx) = text.rfind("[Now answer concisely]") {
        let after = &text[idx..];
        if let Some(nl) = after.find('\n') {
            return after[nl + 1..].trim();
        }
    }
    text.trim()
}

async fn h_integrate_chain(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let qp = req
        .get("query_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_TEXT as u64) as PoolId;
    let tp = req
        .get("target_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_ACTION as u64) as PoolId;
    let hops = req.get("max_hops").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let seed_b64 = req.get("seed").and_then(|v| v.as_str()).unwrap_or("");
    let seed = match b64_url_decode(seed_b64) {
        Ok(b) => b,
        Err(e) => return Json(json!({"error": format!("bad seed: {}", e)})),
    };
    let mut brain = s.brain.lock().await;
    let trail = brain.integrate_chain(qp, tp, &seed, hops);
    let steps: Vec<_> = trail
        .into_iter()
        .map(|(q, a)| {
            json!({ "query": b64_url_no_pad(&q),
                "answer": a.map(|b| b64_url_no_pad(&b)) })
        })
        .collect();
    Json(json!({ "steps": steps }))
}

async fn h_integrate_islands(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let sample = req
        .get("sample_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(500) as usize;
    let thr = req
        .get("similarity_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let brain = s.brain.lock().await;
    let bridges = brain.integrate_islands(sample, thr);
    Json(json!({
        "bridges_added": bridges,
        "tick_now":      brain.fabric().current_tick(),
        "sample_size":   sample,
        "similarity_threshold": thr,
    }))
}

async fn h_pool_concepts(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let pool_id = req.get("pool_id").and_then(|v| v.as_u64()).unwrap_or(0) as PoolId;
    let brain = s.brain.lock().await;
    let Some(pool) = brain.fabric().pool(pool_id) else {
        return Json(json!({"error": format!("unknown pool id {}", pool_id), "concepts": []}));
    };
    let p = pool.read();
    let concepts: Vec<_> = p
        .iter_neurons()
        .filter(|n| !n.is_atom())
        .map(|n| {
            let decoded = p.decode_concept_members(&n.members);
            json!({
                "neuron_id":    n.id,
                "label":        n.label.clone(),
                "member_count": n.members.len(),
                "decoded":      String::from_utf8_lossy(&decoded).to_string(),
                "use_count":    n.use_count,
            })
        })
        .collect();
    Json(json!({ "pool_id": pool_id, "concepts": concepts }))
}

/// Read-only inspection of exact ordered bindings for one query.
async fn h_binding_diagnose(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let text = req.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let qp = req
        .get("query_pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(POOL_TEXT as u64) as PoolId;
    let mut brain = s.brain.lock().await;
    let tp = req
        .get("target_pool")
        .and_then(|v| v.as_u64())
        .map(|v| v as PoolId)
        .or_else(|| brain.action_pool_id())
        .unwrap_or(POOL_ACTION);
    let known = brain.activate_for_prediction(qp, text.as_bytes());
    let query_seq = brain
        .fabric()
        .pool(qp)
        .map(|p| p.read().last_observed_sequence().to_vec())
        .unwrap_or_default();
    let mut exact = Vec::new();
    let mut fuzzy: Vec<(f32, serde_json::Value)> = Vec::new();
    if let (Some(qh), Some(th), Some(bh)) = (
        brain.fabric().pool(qp),
        brain.fabric().pool(tp),
        brain.fabric().pool(brain.binding_pool_id()),
    ) {
        let q = qh.read();
        let t = th.read();
        let bindings = bh.read();
        let firing_atoms: std::collections::HashSet<NeuronId> = q
            .currently_firing()
            .filter(|nid| q.get(*nid).is_some_and(|n| n.is_atom()))
            .collect();
        let firing_concepts: std::collections::HashSet<NeuronId> = q
            .currently_firing()
            .filter(|nid| q.get(*nid).is_some_and(|n| !n.is_atom()))
            .collect();
        for binding in bindings.iter_neurons().filter(|n| !n.is_atom()) {
            let member_pools: std::collections::BTreeSet<PoolId> =
                binding.members.iter().map(|member| member.pool).collect();
            let q_atoms: Vec<NeuronId> = binding
                .members
                .iter()
                .filter(|m| m.pool == qp && q.get(m.neuron).is_some_and(|n| n.is_atom()))
                .map(|m| m.neuron)
                .collect();
            let q_concepts: std::collections::HashSet<NeuronId> = binding
                .members
                .iter()
                .filter(|m| m.pool == qp && q.get(m.neuron).is_some_and(|n| !n.is_atom()))
                .map(|m| m.neuron)
                .collect();
            let q_atom_set: std::collections::HashSet<NeuronId> = q_atoms.iter().copied().collect();
            let atom_intersect = q_atom_set.intersection(&firing_atoms).count();
            let atom_precision = atom_intersect as f32 / q_atom_set.len().max(1) as f32;
            let atom_recall = atom_intersect as f32 / firing_atoms.len().max(1) as f32;
            let atom_score = atom_precision * atom_recall;
            let concept_intersect = q_concepts.intersection(&firing_concepts).count();
            let concept_precision = concept_intersect as f32 / q_concepts.len().max(1) as f32;
            let concept_recall = concept_intersect as f32 / firing_concepts.len().max(1) as f32;
            let concept_score = concept_precision * concept_recall;
            let target_atoms: Vec<NeuronRef> = binding
                .members
                .iter()
                .filter(|m| m.pool == tp && t.get(m.neuron).is_some_and(|n| n.is_atom()))
                .copied()
                .collect();
            let target =
                String::from_utf8_lossy(&t.decode_concept_members(&target_atoms)).to_string();
            fuzzy.push((
                atom_score.max(concept_score),
                json!({
                    "binding_id": binding.id,
                    "member_pools": member_pools.clone(),
                    "use_count": binding.use_count,
                    "sequence_match": q_atoms == query_seq,
                    "atom_score": atom_score,
                    "atom_precision": atom_precision,
                    "atom_recall": atom_recall,
                    "concept_score": concept_score,
                    "concept_precision": concept_precision,
                    "concept_recall": concept_recall,
                    "target": target,
                }),
            ));
            if q_atoms != query_seq {
                continue;
            }
            exact.push(json!({
                "binding_id": binding.id,
                "member_pools": member_pools,
                "use_count": binding.use_count,
                "query_atom_count": q_atoms.len(),
                "target_atom_count": target_atoms.len(),
                "target": target,
            }));
            if exact.len() >= 32 {
                break;
            }
        }
    }
    fuzzy.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top_matches: Vec<_> = fuzzy.into_iter().take(10).map(|(_, row)| row).collect();
    brain.clear_prediction_activation();
    Json(json!({
        "learning": false,
        "known_atom_count": known.len(),
        "query_sequence_length": query_seq.len(),
        "exact_binding_count": exact.len(),
        "exact_bindings": exact,
        "top_matches": top_matches,
    }))
}

async fn h_retune(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let n = req
        .get("sample_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let mut brain = s.brain.lock().await;
    Json(json!(brain.retune(n)))
}

async fn h_tuning_state(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    Json(json!(brain.tuning_state()))
}

async fn h_force_decay(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let v = req
        .get("decay_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(2e-5) as f32;
    let v = v.clamp(1e-7, 0.5);
    let brain = s.brain.lock().await;
    let pids = brain.fabric().pool_ids();
    for pid in &pids {
        if let Some(p) = brain.fabric().pool(*pid) {
            p.write().config.decay_rate = v;
        }
    }
    Json(json!({ "decay_rate": v, "pools_updated": pids.len() }))
}

async fn h_idle_ticks(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let n = req.get("n").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
    let mut brain = s.brain.lock().await;
    for _ in 0..n {
        brain.advance_tick();
    }
    Json(json!({ "ticks_advanced": n, "current_tick": brain.fabric().current_tick() }))
}

async fn h_sleep(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let min_use_count = req
        .get("min_use_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(2);
    let stale_ticks = req
        .get("stale_ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(1000);
    let brain = s.brain.lock().await;
    // Phase 0 — drain deferred promotions (when W1Z4RD_DEFER_PROMOTION
    // mode is active, this is where structure work crystallises).
    let promotions = brain.sleep_drain_promotions();
    // Phase 1 — prune weak concepts across every pool, collect the
    // pruned NeuronRefs so phase 2 can clean up inbound cross-pool
    // terminals that pointed at them.
    let mut pruned_set: ahash::AHashSet<w1z4rd_brain::NeuronRef> = ahash::AHashSet::new();
    for pid in brain.fabric().pool_ids() {
        let p = brain.sleep_pool_phase1(pid, min_use_count, stale_ticks);
        pruned_set.extend(p);
    }
    let pruned = pruned_set.len();
    // Phase 2 — for every pool, drop any inbound cross-pool terminals
    // targeting the pruned refs we just removed.
    for pid in brain.fabric().pool_ids() {
        brain.sleep_pool_phase2(pid, &pruned_set);
    }
    // Phase 3 — per-pool housekeeping.
    for pid in brain.fabric().pool_ids() {
        brain.sleep_pool_housekeeping(pid);
    }
    Json(json!({
        "promotions_drained": promotions,
        "concepts_pruned":    pruned,
        "tick_now":           brain.fabric().current_tick(),
    }))
}

async fn h_checkpoint(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let dir = default_node_brain_dir();
    let path = dir.join("brain.bin");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return Json(json!({ "ok": false, "error": format!("mkdir {}: {}", dir.display(), e) }));
    }
    let brain = s.brain.lock().await;
    match brain.checkpoint(&path) {
        Ok(()) => Json(json!({
            "ok": true,
            "path": path.display().to_string(),
            "tick": brain.fabric().current_tick(),
        })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

async fn h_observe_profile(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let snap = brain.fabric().observe_profile();
    let observes = snap.observes.max(1) as f64;
    let to_us = |ns: u64| (ns as f64 / 1_000.0) as u64;
    let mean_us = |ns: u64| ((ns as f64) / observes / 1_000.0) as u64;
    let pct = |ns: u64| {
        if snap.total_ns == 0 {
            0.0
        } else {
            (ns as f64) * 100.0 / (snap.total_ns as f64)
        }
    };
    Json(json!({
        "observes":             snap.observes,
        "atomize_us":           to_us(snap.atomize_ns),
        "atom_fire_us":         to_us(snap.atom_fire_ns),
        "lazy_decay_us":        to_us(snap.lazy_decay_ns),
        "collapse_us":          to_us(snap.collapse_ns),
        "concept_emergence_us": to_us(snap.concept_emergence_ns),
        "end_of_frame_us":      to_us(snap.end_of_frame_ns),
        "qa_capture_us":        to_us(snap.qa_capture_ns),
        "wal_events":           snap.wal_events,
        "wal_append_us":        to_us(snap.wal_append_ns),
        "total_us":             to_us(snap.total_ns),
        "total_ms":             (snap.total_ns as f64 / 1_000_000.0) as u64,
        "mean_per_observe_us": {
            "atomize":           mean_us(snap.atomize_ns),
            "atom_fire":         mean_us(snap.atom_fire_ns),
            "lazy_decay":        mean_us(snap.lazy_decay_ns),
            "collapse":          mean_us(snap.collapse_ns),
            "concept_emergence": mean_us(snap.concept_emergence_ns),
            "end_of_frame":      mean_us(snap.end_of_frame_ns),
            "qa_capture":        mean_us(snap.qa_capture_ns),
            "wal_append":        mean_us(snap.wal_append_ns),
            "total":             mean_us(snap.total_ns),
        },
        "phase_pct_of_total": {
            "atomize":           pct(snap.atomize_ns),
            "atom_fire":         pct(snap.atom_fire_ns),
            "lazy_decay":        pct(snap.lazy_decay_ns),
            "collapse":          pct(snap.collapse_ns),
            "concept_emergence": pct(snap.concept_emergence_ns),
            "end_of_frame":      pct(snap.end_of_frame_ns),
            "qa_capture":        pct(snap.qa_capture_ns),
            "wal_append":        pct(snap.wal_append_ns),
        },
    }))
}

async fn h_tier_orchestrator(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let snap = brain.fabric().tier_orchestrator_stats();
    let mean_per_pass_ns = if snap.passes == 0 {
        0
    } else {
        snap.total_ns / snap.passes
    };
    Json(json!({
        "passes":             snap.passes,
        "neurons_scanned":    snap.neurons_scanned,
        "neurons_evicted":    snap.neurons_evicted,
        "neurons_paged_in":   snap.neurons_paged_in,
        "evict_errors":       snap.evict_errors,
        "page_in_errors":     snap.page_in_errors,
        "last_pressure":      snap.last_pressure,
        "total_us":           snap.total_ns / 1_000,
        "total_ms":           snap.total_ns / 1_000_000,
        "mean_per_pass_us":   mean_per_pass_ns / 1_000,
    }))
}

/// POST /brain/tier_orchestrator/params — adjust the live orchestrator
/// params without restarting the node binary.  All fields optional;
/// any omitted field keeps its current value.  Returns the resolved
/// params after applying.
///
/// Example body: `{"target_terminals_per_pool": 5000000, "evict_threshold": 4.5}`
///
/// Special action keys:
///   `"action": "disable"` → orchestrator stops running on next tick.
///   `"action": "enable"`  → resets to env-driven defaults.
async fn h_tier_orchestrator_params(
    State(s): State<BrainApiState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    use w1z4rd_brain::tier_orchestrator::OrchestratorParams;
    let brain = s.brain.lock().await;
    // Quick action shortcut.
    if let Some(action) = body.get("action").and_then(|v| v.as_str()) {
        match action {
            "disable" => {
                brain
                    .fabric()
                    .set_tier_orchestrator_params(OrchestratorParams::disabled());
                return Json(json!({"status": "disabled"}));
            }
            "enable" => {
                brain
                    .fabric()
                    .set_tier_orchestrator_params(OrchestratorParams::from_env_or_disabled());
                return Json(json!({"status": "enabled", "source": "env_or_default"}));
            }
            _ => {}
        }
    }
    // Field-by-field override: start from current params and patch what's in body.
    let mut p = brain.fabric().orchestrator_params_snapshot();
    if let Some(v) = body.get("run_every_n_ticks").and_then(|x| x.as_u64()) {
        p.run_every_n_ticks = v;
    }
    if let Some(v) = body.get("scan_budget").and_then(|x| x.as_u64()) {
        p.scan_budget = v as usize;
    }
    if let Some(v) = body.get("max_evict_per_pass").and_then(|x| x.as_u64()) {
        p.max_evict_per_pass = v as usize;
    }
    if let Some(v) = body
        .get("target_terminals_per_pool")
        .and_then(|x| x.as_u64())
    {
        p.target_terminals_per_pool = v as usize;
    }
    if let Some(v) = body.get("evict_threshold").and_then(|x| x.as_f64()) {
        p.evict_threshold = v as f32;
    }
    if let Some(v) = body.get("w_terminals").and_then(|x| x.as_f64()) {
        p.w_terminals = v as f32;
    }
    if let Some(v) = body.get("w_staleness").and_then(|x| x.as_f64()) {
        p.w_staleness = v as f32;
    }
    if let Some(v) = body.get("w_inverse_salience").and_then(|x| x.as_f64()) {
        p.w_inverse_salience = v as f32;
    }
    if let Some(v) = body.get("w_pinned").and_then(|x| x.as_f64()) {
        p.w_pinned = v as f32;
    }
    if let Some(v) = body.get("decay_horizon_ticks").and_then(|x| x.as_u64()) {
        p.decay_horizon_ticks = v;
    }
    if let Some(v) = body.get("salience_eps").and_then(|x| x.as_f64()) {
        p.salience_eps = v as f32;
    }
    if let Some(v) = body.get("page_in_salience_floor").and_then(|x| x.as_f64()) {
        p.page_in_salience_floor = v as f32;
    }
    if let Some(v) = body.get("max_page_in_per_pass").and_then(|x| x.as_u64()) {
        p.max_page_in_per_pass = v as usize;
    }
    if let Some(v) = body.get("min_age_ticks").and_then(|x| x.as_u64()) {
        p.min_age_ticks = v;
    }
    brain.fabric().set_tier_orchestrator_params(p);
    Json(json!({
        "status": "params_set",
        "params": {
            "run_every_n_ticks":         p.run_every_n_ticks,
            "scan_budget":               p.scan_budget,
            "max_evict_per_pass":        p.max_evict_per_pass,
            "target_terminals_per_pool": p.target_terminals_per_pool,
            "evict_threshold":           p.evict_threshold,
            "w_terminals":               p.w_terminals,
            "w_staleness":               p.w_staleness,
            "w_inverse_salience":        p.w_inverse_salience,
            "w_pinned":                  p.w_pinned,
            "decay_horizon_ticks":       p.decay_horizon_ticks,
            "salience_eps":              p.salience_eps,
            "page_in_salience_floor":    p.page_in_salience_floor,
            "max_page_in_per_pass":      p.max_page_in_per_pass,
            "min_age_ticks":             p.min_age_ticks,
        },
        "enabled": p.run_every_n_ticks != u64::MAX && p.run_every_n_ticks != 0,
    }))
}

async fn h_tick_profile(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let snap = brain.fabric().profile.snapshot();
    let ticks = snap.ticks.max(1) as f64;
    let to_us = |ns: u64| (ns as f64 / 1_000.0) as u64;
    let to_ms = |ns: u64| (ns as f64 / 1_000_000.0) as u64;
    let mean_us = |ns: u64| ((ns as f64) / ticks / 1_000.0) as u64;
    let pct = |ns: u64| {
        if snap.total_ns == 0 {
            0.0
        } else {
            (ns as f64) * 100.0 / (snap.total_ns as f64)
        }
    };
    Json(json!({
        "ticks":                       snap.ticks,
        "cross_pool_atom_wiring_us":   to_us(snap.cross_pool_atom_wiring_ns),
        "cross_pool_concept_wiring_us":to_us(snap.cross_pool_concept_wiring_ns),
        "within_pool_temporal_us":     to_us(snap.within_pool_temporal_ns),
        "housekeeping_us":             to_us(snap.housekeeping_ns),
        "total_us":                    to_us(snap.total_ns),
        "total_ms":                    to_ms(snap.total_ns),
        "mean_per_tick_us": {
            "cross_pool_atom_wiring":    mean_us(snap.cross_pool_atom_wiring_ns),
            "cross_pool_concept_wiring": mean_us(snap.cross_pool_concept_wiring_ns),
            "within_pool_temporal":      mean_us(snap.within_pool_temporal_ns),
            "housekeeping":              mean_us(snap.housekeeping_ns),
            "total":                     mean_us(snap.total_ns),
        },
        "phase_pct_of_total": {
            "cross_pool_atom_wiring":    pct(snap.cross_pool_atom_wiring_ns),
            "cross_pool_concept_wiring": pct(snap.cross_pool_concept_wiring_ns),
            "within_pool_temporal":      pct(snap.within_pool_temporal_ns),
            "housekeeping":              pct(snap.housekeeping_ns),
        },
    }))
}

async fn h_sleep_pressure(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let deferred = std::env::var("W1Z4RD_DEFER_PROMOTION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    Json(json!({
        "deferred_promotion_enabled": deferred,
        "pending_promotions":         brain.pending_promotion_count(),
    }))
}

async fn h_thinking_start(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    if let Some(q) = req.get("query_pool").and_then(|v| v.as_u64()) {
        s.thinking.query_pool.store(q as u32, Ordering::Release);
    }
    if let Some(t) = req.get("target_pool").and_then(|v| v.as_u64()) {
        s.thinking.target_pool.store(t as u32, Ordering::Release);
    }
    if let Some(seed_b64) = req.get("seed").and_then(|v| v.as_str()) {
        if let Ok(b) = b64_url_decode(seed_b64) {
            *s.thinking.last_answer.lock().unwrap() = Some(b);
            *s.thinking.last_seed.lock().unwrap() = None;
        }
    }
    s.thinking.enabled.store(true, Ordering::Release);
    Json(json!({ "enabled": true }))
}

async fn h_thinking_stop(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    s.thinking.enabled.store(false, Ordering::Release);
    Json(json!({ "enabled": false }))
}

async fn h_thinking_status(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let seed = s.thinking.last_seed.lock().unwrap().clone();
    let answer = s.thinking.last_answer.lock().unwrap().clone();
    Json(json!({
        "enabled":      s.thinking.enabled.load(Ordering::Acquire),
        "query_pool":   s.thinking.query_pool.load(Ordering::Acquire),
        "target_pool":  s.thinking.target_pool.load(Ordering::Acquire),
        "hops_taken":   s.thinking.hops_taken.load(Ordering::Acquire),
        "last_seed":    seed.as_deref().map(b64_url_no_pad),
        "last_answer":  answer.as_deref().map(b64_url_no_pad),
    }))
}

/// Background thinking task — same logic as in `brain_server.rs`.
/// Acquires the brain lock briefly per hop, yields between hops so
/// /observe and /integrate preempt cleanly.
pub async fn run_thinking_loop(state: BrainApiState) {
    use std::time::Duration;
    let mut qa_cursor: usize = 0;
    loop {
        if !state.thinking.enabled.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            continue;
        }
        let qp = state.thinking.query_pool.load(Ordering::Acquire);
        let tp = state.thinking.target_pool.load(Ordering::Acquire);

        let last_answer_snap = state.thinking.last_answer.lock().unwrap().clone();
        let last_seed_snap = state.thinking.last_seed.lock().unwrap().clone();

        let seed: Option<Vec<u8>> = match last_answer_snap {
            Some(ans) if !ans.is_empty() && Some(&ans) != last_seed_snap.as_ref() => Some(ans),
            _ => {
                let brain = state.brain.lock().await;
                let len = brain.qa_db().len();
                if len == 0 {
                    None
                } else {
                    let idx = qa_cursor % len;
                    qa_cursor = qa_cursor.wrapping_add(1);
                    brain.qa_db().iter().nth(idx).map(|qp| qp.prompt.clone())
                }
            }
        };

        let Some(seed) = seed else {
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        };

        let answer = {
            let mut brain = state.brain.lock().await;
            brain.fabric_mut().observe(qp, &seed);
            brain.integrate(qp, tp).answer
        };

        *state.thinking.last_seed.lock().unwrap() = Some(seed);
        *state.thinking.last_answer.lock().unwrap() = answer;
        state.thinking.hops_taken.fetch_add(1, Ordering::AcqRel);

        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------
// Router builder
// ---------------------------------------------------------------------

/// Build the FULL brain endpoint router with state baked in.  Mounts
/// every Phase A–E handler INCLUDING the baseline `/observe`,
/// `/tick`, `/stats`, `/health`, `/integrate`, `/pool/concepts` ones.
/// Used by the main node binary under `/brain/*` where nothing else
/// is on that prefix.
pub fn brain_routes(state: BrainApiState) -> Router {
    Router::new()
        .route("/health", get(h_health))
        .route("/stats", get(h_stats))
        .route("/observe", post(h_observe))
        .route("/pretrain", post(h_pretrain))
        .route("/pretrain_binding", post(h_pretrain_binding))
        .route("/pretrain_bindings", post(h_pretrain_bindings))
        .route("/predict/multi", post(h_predict_multi))
        .route("/repair/predict", post(h_repair_predict))
        .route("/tick", post(h_tick))
        .route("/integrate", post(h_integrate))
        .route("/predict", post(h_predict))
        .route("/consolidate", post(h_consolidate))
        .route("/logic/consolidate", post(h_logic_consolidate))
        .route("/logic/compose", post(h_logic_compose))
        .route("/logic/crystallize", post(h_logic_crystallize))
        .route("/logic/recognize", post(h_logic_recognize))
        .route("/chat", post(h_brain_chat))
        .route("/pool/concepts", post(h_pool_concepts))
        .route("/binding/diagnose", post(h_binding_diagnose))
        .with_state(state.clone())
        .merge(brain_phase_routes(state))
}

/// Build a router with ONLY the new Phase A–E routes that didn't
/// exist before — `/qa_db_stats`, `/consolidation_stats`,
/// `/self_test`, `/integrate_chain`, `/integrate_islands`, `/retune`,
/// `/tuning_state`, `/force_decay`, `/idle_ticks`, `/thinking/*`,
/// `/set_domain`, `/domain_stats`, `/sleep_pressure`.
///
/// Used by the standalone `brain_server` binary which already has its
/// own elaborated `/observe`, `/tick`, `/stats`, `/health`,
/// `/integrate`, `/pool/concepts` handlers (with timing logs, cluster
/// shipping, extra Stage-10 fields, etc.).  Merging this router into
/// brain_server's main router avoids the duplicate definitions
/// without losing brain_server's diagnostic surface.
pub fn brain_phase_routes(state: BrainApiState) -> Router {
    brain_phase_routes_impl(state, true)
}

/// Force the attached WAL through the OS durability boundary.  This mirrors
/// the standalone server's top-level `/flush` handler so merged-node and
/// `/brain/*` clients do not need topology-specific route knowledge.
async fn h_flush(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    match brain.store_clone().flush() {
        Ok(()) => {
            let wal_path = default_node_brain_dir().join("brain.wal");
            let wal_bytes = std::fs::metadata(&wal_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            Json(json!({ "ok": true, "wal_bytes": wal_bytes }))
        }
        Err(error) => Json(json!({ "ok": false, "error": error.to_string() })),
    }
}

/// An exact fragment is not a complete answer when another grounded fragment
/// explicitly depends on its role.
fn exact_fragment_has_grounded_dependents(exact: &[u8], candidates: &[Vec<u8>]) -> bool {
    let Some(fragment) = serde_json::from_slice::<serde_json::Value>(exact)
        .ok()
        .and_then(|value| value.get("code_fragment").cloned())
    else {
        return false;
    };
    let Some(role) = fragment.get("role").and_then(|value| value.as_str()) else {
        return false;
    };
    let file = fragment
        .get("file")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let qualified = format!("{file}::{role}");
    candidates.iter().any(|candidate| {
        if candidate.as_slice() == exact {
            return false;
        }
        serde_json::from_slice::<serde_json::Value>(candidate)
            .ok()
            .and_then(|value| value.get("code_fragment").cloned())
            .and_then(|value| value.get("after").cloned())
            .and_then(|value| value.as_array().cloned())
            .is_some_and(|after| {
                after.iter().any(|dependency| {
                    dependency
                        .as_str()
                        .is_some_and(|dependency| dependency == role || dependency == qualified)
                })
            })
    })
}

/// Phase routes for the standalone brain server, which supplies its own
/// elaborated tick-profile, sleep, and checkpoint handlers.
pub fn brain_phase_routes_without_core(state: BrainApiState) -> Router {
    brain_phase_routes_impl(state, false)
}

fn brain_phase_routes_impl(state: BrainApiState, include_core_routes: bool) -> Router {
    let routes = Router::new()
        .route("/set_domain", post(h_set_domain))
        .route("/domain_stats", get(h_domain_stats))
        .route("/qa_db_stats", get(h_qa_db_stats))
        .route("/consolidation_stats", get(h_consolidation_stats))
        .route("/self_test", post(h_self_test))
        .route("/integrate_chain", post(h_integrate_chain))
        .route("/integrate_islands", post(h_integrate_islands))
        .route("/retune", post(h_retune))
        .route("/tuning_state", get(h_tuning_state))
        .route("/force_decay", post(h_force_decay))
        .route("/idle_ticks", post(h_idle_ticks))
        .route("/sleep_pressure", get(h_sleep_pressure))
        .route("/observe_profile", get(h_observe_profile))
        .route("/http_profile", get(h_http_profile))
        .route("/tier_orchestrator", get(h_tier_orchestrator))
        .route(
            "/tier_orchestrator/params",
            post(h_tier_orchestrator_params),
        )
        .route("/thinking/start", post(h_thinking_start))
        .route("/thinking/stop", post(h_thinking_stop))
        .route("/thinking/status", get(h_thinking_status));
    let routes = if include_core_routes {
        routes
            .route("/tick_profile", get(h_tick_profile))
            .route("/sleep", post(h_sleep))
            .route("/checkpoint", post(h_checkpoint))
            .route("/flush", post(h_flush))
    } else {
        routes
    };
    routes.with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_identity_persists_and_reloads_without_process_environment() {
        let identity_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../brains/coding_debug.identity.toml");
        let unique = format!(
            "w1z4rd_identity_contract_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let data_dir = std::env::temp_dir().join(unique);

        let configured = resolve_identity(&data_dir, Some(&identity_path))
            .unwrap()
            .unwrap();
        assert!(data_dir.join("brain.identity.toml").exists());
        let recovered = resolve_identity(&data_dir, None).unwrap().unwrap();
        assert_eq!(recovered.name, configured.name);
        assert_eq!(recovered.pools.len(), 12);
        assert_eq!(recovered.pools[11].prototype, "instruction-intent");

        std::fs::remove_dir_all(data_dir).ok();
    }

    #[test]
    fn deployed_identity_reconfigures_restored_pool_learning_policy() {
        let identity_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../brains/coding_small.identity.toml");
        let identity = BrainIdentitySpec::load_toml(identity_path).unwrap();
        let mut brain = build_from_identity(&identity).unwrap();
        let response = brain.fabric().pool(4).unwrap();
        response.write().config.max_concept_member_count = 32;

        apply_identity_pool_configs(&mut brain, &identity).unwrap();

        assert_eq!(response.read().config.max_concept_member_count, 1);
    }

    #[test]
    fn complete_manifest_is_distinguished_from_partial_fragment_evidence() {
        assert!(is_complete_file_manifest(
            br#"{"files":{"domain.py":"VALUE = 1\n","service.py":"from domain import VALUE\n"}}"#
        ));
        assert!(!is_complete_file_manifest(
            br#"{"code_fragment":{"file":"service.py","role":"import","source":"from domain import VALUE\n"}}"#
        ));
        assert!(!is_complete_file_manifest(
            br#"{"files":{"../escape.py":"VALUE = 1\n"}}"#
        ));
    }

    #[test]
    fn grounded_fragments_form_a_never_observed_file_in_slot_order() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"main.py","order":20,"source":"    return value\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"main.py","order":10,"source":"def identity(value):\n"}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments(&candidates).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(
            value["files"]["main.py"],
            "def identity(value):\n    return value\n"
        );
    }

    #[test]
    fn grounded_fragment_conflicts_are_rejected() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"main.py","order":10,"source":"a"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"main.py","order":10,"source":"b"}}"#.to_vec(),
        ];
        assert!(merge_grounded_code_fragments(&candidates).is_none());
    }

    #[test]
    fn grounded_relative_fragments_settle_dependencies_not_input_order() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"main.js","role":"return","after":["signature"],"source":"  return value;\n}\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"main.js","role":"signature","after":[],"source":"function identity(value) {\n"}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments(&candidates).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(
            value["files"]["main.js"],
            "function identity(value) {\n  return value;\n}\n"
        );
    }

    #[test]
    fn exact_prerequisite_fragment_yields_to_its_grounded_dependent() {
        let signature = br#"{"code_fragment":{"file":"main.js","role":"signature","after":[],"source":"function identity(value) {\n"}}"#.to_vec();
        let body = br#"{"code_fragment":{"file":"main.js","role":"return","after":["signature"],"source":"  return value;\n}\n"}}"#.to_vec();
        assert!(exact_fragment_has_grounded_dependents(
            &signature,
            &[signature.clone(), body]
        ));
        assert!(!exact_fragment_has_grounded_dependents(
            &signature,
            &[signature.clone()]
        ));
    }

    #[test]
    fn grounded_relative_fragment_cycles_are_rejected() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"main.js","role":"a","after":["b"],"source":"a"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"main.js","role":"b","after":["a"],"source":"b"}}"#.to_vec(),
        ];
        assert!(merge_grounded_code_fragments(&candidates).is_none());
    }

    #[test]
    fn cross_file_dependencies_and_rejected_evidence_settle_safely() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"domain.py","role":"model","after":[],"source":"VALUE=1\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"service.py","role":"import","after":["domain.py::model"],"evidence_id":"bad","source":"from missing import VALUE\n"}}"#.to_vec(),
            br#"{"fragment_outcome":{"evidence_id":"bad","confirmed":false}}"#.to_vec(),
            br#"{"code_fragment":{"file":"service.py","role":"import","after":["domain.py::model"],"evidence_id":"good","source":"from domain import VALUE\n"}}"#.to_vec(),
            br#"{"fragment_outcome":{"evidence_id":"good","confirmed":true}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments(&candidates).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(value["files"]["domain.py"], "VALUE=1\n");
        assert_eq!(value["files"]["service.py"], "from domain import VALUE\n");
    }
}

/// Data directory for the main node's embedded brain.  Looks at
/// `W1Z4RD_NODE_BRAIN_DIR` first (so the main node can keep its brain
/// state separate from the standalone brain_server), falls back to
/// `W1Z4RDV1510N_DATA_DIR/brain` and finally `brain-data`.
pub fn default_node_brain_dir() -> PathBuf {
    // Explicit override always wins.
    if let Ok(p) = std::env::var("W1Z4RD_NODE_BRAIN_DIR") {
        return PathBuf::from(p);
    }
    // If W1Z4RDV1510N_DATA_DIR is set: prefer the data dir itself when
    // a brain.bin already lives there directly (the supervisor's
    // long-standing production layout: D:\w1z4rdv1510n-data\brain.bin
    // alongside neuro_pool.json + equation_matrix.json + cold tiers).
    // Fall back to <data_dir>/brain as a subdir for fresh installs that
    // don't have brain.bin at the data-dir root yet.
    if let Ok(p) = std::env::var("W1Z4RDV1510N_DATA_DIR") {
        let base = PathBuf::from(p);
        if base.join("brain.bin").exists() {
            return base;
        }
        return base.join("brain");
    }
    PathBuf::from("brain-data")
}
