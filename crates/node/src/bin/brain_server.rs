//! Standalone HTTP server for the W1z4rD brain (crates/brain) with the
//! same multimodal endpoint shape the existing training scripts use,
//! plus a /chat endpoint that routes through Brain::generate().
//!
//! Pools (built from a custom identity, NOT default_general_observer):
//!     id=1  text    (keyboard/byte-passthrough, prefix "t")
//!     id=2  image   (byte-passthrough of raw image bytes, prefix "i")
//!     id=3  audio   (byte-passthrough of raw audio bytes, prefix "a")
//!     id=4  action  (byte-passthrough of action emissions, prefix "act")
//!  (id=0 is the auto-created binding pool, prefix "bind")
//!
//! Endpoints (JSON):
//!   GET  /health
//!   GET  /stats
//!   POST /observe                { pool_id, frame }           — frame base64url
//!   POST /tick
//!   POST /integrate              { query_pool, target_pool }
//!   POST /checkpoint
//!
//!   # Multimodal (compatible with scripts/sensors/*.py):
//!   POST /sensor/observe         { kind: "text"|"image"|"audio", bytes_b64 OR text }
//!                                returns { predictions: { pool_name: [labels] } }
//!   POST /sensor/observe_triple  { text, image_b64, audio_b64, lr? }
//!                                returns { img_labels, aud_labels }
//!   POST /chat                   { text }
//!                                returns { reply, predictions: { pool: [labels] } }
//!
//! Data dir is `W1Z4RDV1510N_DATA_DIR` (or ./brain-data).  Brain
//! persists to `<data_dir>/brain.bin` on shutdown.

// Stage 18.12 step 3: HTTP-backed RemoteTransport for §18 distributed
// substrate.  The cluster module is included here (rather than as a
// dependency module) because brain_server is a single-binary build.
#[path = "brain_server_cluster.rs"]
mod cluster;
// Re-export so it can be used from elsewhere if needed; unused here yet.
#[allow(unused_imports)]
use cluster::{HttpRemoteTransport, arc_transport};

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
    AtomEncoding, Brain, BrainConfig, BrainStats, BytePassthroughEncoding,
    PoolConfig, PoolId,
};

const POOL_TEXT:   PoolId = 1;
const POOL_IMAGE:  PoolId = 2;
const POOL_AUDIO:  PoolId = 3;
const POOL_ACTION: PoolId = 4;
/// Stage 11B — turn pool.  Client-supplied opaque turn IDs land here
/// as raw bytes; the brain treats them like any other atom, and the
/// cross-pool fingerprint between a turn neuron and a text neuron
/// becomes the binding the dialogue continuation walker can chain
/// through.  Brain owns no session semantics — the client (typically
/// the Wizard frontend) decides when to advance the turn id and when
/// to expire old turns via `POST /observe/expire`.
const POOL_TURN:   PoolId = 5;

fn pool_name(id: PoolId) -> &'static str {
    match id {
        0 => "binding",
        1 => "text",
        2 => "image",
        3 => "audio",
        4 => "action",
        _ => "other",
    }
}

#[derive(Clone)]
struct AppState {
    brain: Arc<Mutex<Brain>>,
    checkpoint_path: PathBuf,
    /// Per [`ARCHITECTURE.md`] §17.4: the current (or last completed) sleep
    /// job's progress.  Updated by the background tokio task spawned from
    /// the `/sleep` handler; readable by `/sleep/status`.  `None` means
    /// no sleep has ever been requested on this brain instance.
    sleep_status: Arc<std::sync::Mutex<Option<SleepJobStatus>>>,
    /// Stage 18.12 step 5: cluster membership state.  Tracks the
    /// ring of NodeIds + their HTTP base URLs.  Mutated by /cluster/join
    /// (peer adds itself) and /cluster/leave; read by /cluster/members
    /// and consumed by the head node's routing logic.
    cluster: Arc<std::sync::Mutex<ClusterMembership>>,
}

/// Stage 18.12 step 5: per-node cluster membership table.  Per
/// [`ARCHITECTURE.md`] §18.4.  Every brain_server instance owns one;
/// nodes update each other's tables through the /cluster/join +
/// /cluster/heartbeat protocol.
///
/// The legacy `crates/cluster` already implements OTP-key membership
/// for the node-level ring (port 51611); §18.4 extends that pattern
/// to the *brain-shard* layer (port 8095 HTTP) with these structures.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClusterMembership {
    /// This node's id within the ring.  0 = solo / unjoined.
    pub local_node:   w1z4rd_brain::store::NodeId,
    /// This node's externally-reachable base URL, e.g.
    /// `http://192.168.1.43:8095`.  Empty if not configured.
    pub local_addr:   String,
    /// Ring members (including self), ordered by NodeId.
    pub members:      Vec<MemberInfo>,
    /// Monotonically increasing counter used to allocate fresh NodeIds.
    pub next_node_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberInfo {
    pub node_id:   w1z4rd_brain::store::NodeId,
    pub addr:      String,
    /// Wall-time of last successful heartbeat (ms since UNIX epoch).
    /// 0 == never; set by heartbeat handler.
    pub last_heartbeat_ms: i64,
    /// Optional capacity advertisement — how many neurons this node
    /// can host.  Used by Stage 18.6 placement policy.  None = unknown.
    pub capacity_neurons:  Option<u64>,
}

impl ClusterMembership {
    pub fn solo(local_addr: impl Into<String>) -> Self {
        let local_node = w1z4rd_brain::store::NodeId(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64).unwrap_or(0);
        let addr = local_addr.into();
        Self {
            local_node,
            local_addr: addr.clone(),
            members: vec![MemberInfo {
                node_id:   local_node,
                addr,
                last_heartbeat_ms: now_ms,
                capacity_neurons:  None,
            }],
            next_node_id: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct SleepJobStatus {
    /// Monotonically increasing job id within a single brain process.
    job_id:           u64,
    /// "running" | "complete" | "failed"
    phase:            String,
    /// Pools in this brain.  Each entry transitions through phase1 →
    /// phase2 → housekeeping in order.
    pools_total:      usize,
    /// How many pools have completed PHASE 1 prune.
    pools_phase1_done: usize,
    /// How many pools have completed PHASE 2 cross-pool cleanup.
    pools_phase2_done: usize,
    /// How many pools have completed PHASE 3 housekeeping.
    pools_phase3_done: usize,
    /// Cumulative count of concepts pruned across all phase-1 pools so far.
    pruned_so_far:    usize,
    /// Cumulative moments replayed (set after replay phase completes).
    replayed:         usize,
    /// Tick at the start of this sleep run.
    tick_start:       u64,
    /// Tick at the end of this sleep run (set when phase = "complete").
    tick_end:         u64,
    /// Wall-time start (ms since UNIX epoch).
    started_at_ms:    i64,
    /// Wall-time end (ms since UNIX epoch), zero if still running.
    finished_at_ms:   i64,
    /// Set on phase = "failed".
    error:            Option<String>,
}

impl SleepJobStatus {
    fn new(job_id: u64, pools_total: usize, tick_start: u64) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64).unwrap_or(0);
        Self {
            job_id,
            phase:             "running".into(),
            pools_total,
            pools_phase1_done: 0,
            pools_phase2_done: 0,
            pools_phase3_done: 0,
            pruned_so_far:     0,
            replayed:          0,
            tick_start,
            tick_end:          tick_start,
            started_at_ms:     now_ms,
            finished_at_ms:    0,
            error:             None,
        }
    }
}

fn data_dir() -> PathBuf {
    std::env::var("W1Z4RDV1510N_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("brain-data"))
}

fn pool_prefixes() -> HashMap<PoolId, String> {
    let mut p = HashMap::new();
    p.insert(0,           "bind".to_string());
    p.insert(POOL_TEXT,   "t".to_string());
    p.insert(POOL_IMAGE,  "i".to_string());
    p.insert(POOL_AUDIO,  "a".to_string());
    p.insert(POOL_ACTION, "act".to_string());
    p.insert(POOL_TURN,   "turn".to_string());
    p
}

fn build_encodings(prefixes: &HashMap<PoolId, String>) -> HashMap<PoolId, Box<dyn AtomEncoding>> {
    prefixes.iter().map(|(pid, prefix)| {
        let leaked: &'static str = Box::leak(prefix.clone().into_boxed_str());
        let enc: Box<dyn AtomEncoding> = Box::new(BytePassthroughEncoding { prefix: leaked });
        (*pid, enc)
    }).collect()
}

fn build_fresh_brain() -> Result<Brain> {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 3;
    cfg.moment_history_window = 256;
    let mut brain = Brain::new(cfg);

    // Tuning notes: image/audio bytes have very different distributions
    // than text bytes (JPEG headers, FFT bin labels, etc.), so each pool
    // gets a wide recent_atoms window for solid concept emergence and
    // a relatively low decay so cross-modal binding has time to set.

    // Stage 13 — text pool recent_atoms_window must be large enough
    // that, under round-robin training across a wide corpus, an
    // individual prompt sequence (e.g. "apple") repeats within the
    // window for `concept_emergence_threshold` (=3) hits and gets
    // promoted to a concept neuron.
    //
    // Math (per ARCHITECTURE.md §4.D.1 — sequence has precedence over
    // bag-of-atoms):
    //   - Unified corpus is ~7K prompts × avg 7 bytes ≈ 49K bytes / epoch
    //   - Window must hold >= 2 full epochs so the per-prompt sequence
    //     count reaches 3 within window across reps 1-3 of training
    //   - 65,536 atoms ≈ 9.4K positions ≈ 1.3 epochs → safe margin
    //
    // Without this, prompt-concepts never emerge under round-robin
    // training (Stage 12 found the same defect on the action pool
    // and raised that window from 32 → 4096; the text pool needs a
    // larger value because its prompts cycle through more entries).
    let mut text = PoolConfig::defaults("text", POOL_TEXT);
    text.recent_atoms_window         = 65536;
    text.concept_emergence_threshold = 3;
    text.max_concept_member_count    = 32;
    text.decay_rate                  = 0.00002;
    text.prune_floor                 = 0.001;
    apply_env_overrides(&mut text);
    brain.create_pool(text,
        Box::new(BytePassthroughEncoding { prefix: "t" }) as Box<dyn AtomEncoding>);

    let mut image = PoolConfig::defaults("image", POOL_IMAGE);
    image.recent_atoms_window         = 4096;
    image.concept_emergence_threshold = 3;
    image.max_concept_member_count    = 32;
    image.decay_rate                  = 0.00002;
    image.prune_floor                 = 0.001;
    apply_env_overrides(&mut image);
    brain.create_pool(image,
        Box::new(BytePassthroughEncoding { prefix: "i" }) as Box<dyn AtomEncoding>);

    let mut audio = PoolConfig::defaults("audio", POOL_AUDIO);
    audio.recent_atoms_window         = 4096;
    audio.concept_emergence_threshold = 3;
    audio.max_concept_member_count    = 32;
    audio.decay_rate                  = 0.00002;
    audio.prune_floor                 = 0.001;
    apply_env_overrides(&mut audio);
    brain.create_pool(audio,
        Box::new(BytePassthroughEncoding { prefix: "a" }) as Box<dyn AtomEncoding>);

    // Stage 12 (config-only): match the action-pool window to the
    // text-pool window so K-12 / greeting responses can crystallize as
    // concept neurons.  Empirical math audit (scripts/math_audit_*.py)
    // showed that with the default window=32, no K-12 response had a
    // chance to repeat within recent_atoms — round-robin training
    // pushes 7K+ unrelated bytes between successive instances of any
    // given response.  Only toddler categories (animal/vehicle/color/
    // ...) emerged as concepts because they were trained in 8 dense
    // epochs that fit their 6-byte responses within the 32-byte
    // window.  Raising to 4096 (matching text pool) lets every K-12
    // response take part in concept emergence on the same footing.
    // max_concept_member_count raised so multi-byte responses like
    // "musical_instrument" (18 bytes) fit as a single concept.
    // Action pool window — same rationale as text pool above.  At 4096
    // (Stage 12) the high-population category responses (food, animal,
    // vehicle ...) emerge fine, but rarer responses (musical_instrument)
    // emerge late or not at all.  Bumping to 65536 matches the text
    // pool and gives consistent emergence across both sides of the
    // cross-pool fingerprint.
    let mut action = PoolConfig::defaults("action", POOL_ACTION);
    action.recent_atoms_window         = 65536;
    action.concept_emergence_threshold = 3;
    action.max_concept_member_count    = 32;
    action.decay_rate                  = 0.00002;
    action.prune_floor                 = 0.001;
    apply_env_overrides(&mut action);
    brain.create_pool(action,
        Box::new(BytePassthroughEncoding { prefix: "act" }) as Box<dyn AtomEncoding>);
    brain.designate_action_pool(POOL_ACTION);

    // Stage 11B — turn pool.  Aggressive decay + low prune floor so old
    // turn-id neurons fall out of the pool naturally (audit-2 LRU
    // behavior without needing a hard cap in the substrate).  The
    // recent_atoms window is small because we only need adjacency
    // information within a single conversation — older turns should
    // not influence newer ones.
    let mut turn = PoolConfig::defaults("turn", POOL_TURN);
    turn.recent_atoms_window         = 32;
    turn.concept_emergence_threshold = u32::MAX;  // turn ids never collapse into concepts
    turn.max_concept_member_count    = 4;
    turn.decay_rate                  = 0.001;     // ~50× faster than text/image/audio
    turn.prune_floor                 = 0.01;      // aggressive — old turn neurons recede fast
    apply_env_overrides(&mut turn);
    brain.create_pool(turn,
        Box::new(BytePassthroughEncoding { prefix: "turn" }) as Box<dyn AtomEncoding>);

    Ok(brain)
}

/// Read per-pool env-var overrides for tunable knobs.  Convention:
/// `BRAIN_<KNOB>_<POOLNAME>` (uppercased), with `BRAIN_<KNOB>_DEFAULT`
/// as a global fallback.  Pool-specific override wins over the default.
///
/// Dynamical-system knobs (parsed as ControlMode JSON, OR as a bare
/// number which becomes Constant(value)):
///   SPARSITY      → sparsity_mode             (ControlMode)
///   HET_LTD       → heterosynaptic_ltd_mode   (ControlMode)
///   PREDICT_GATE  → predict_gate_mode         (ControlMode)
///
/// JSON examples (for ControlMode):
///   BRAIN_SPARSITY_TEXT='{"Constant":0.7}'
///   BRAIN_SPARSITY_TEXT='{"DrivenBy":{"signal":"InvSurprise","scale":0.7,"offset":0.3,"min":0.05,"max":1.0}}'
///
/// Static knobs (still scalar):
///   SPARSITY_MIN  → sparsity_min_neurons (usize >= 1)
///   WINDOW        → recent_atoms_window  (usize)
///   EMERGENCE     → concept_emergence_threshold (u32)
///   MAX_MEMBERS   → max_concept_member_count (usize)
///   DECAY         → decay_rate (f32)
///   PRUNE_FLOOR   → prune_floor (f32)
///
/// All overrides are optional.
fn apply_env_overrides(cfg: &mut PoolConfig) {
    use w1z4rd_brain::ControlMode;
    fn read_f32(key: &str) -> Option<f32> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }
    fn read_usize(key: &str) -> Option<usize> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }
    fn read_u32(key: &str) -> Option<u32> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }
    /// ControlMode parser: accepts either a bare number (→ Constant)
    /// or JSON spec like '{"DrivenBy":{"signal":"InvSurprise","scale":...}}'.
    fn read_control_mode(key: &str) -> Option<ControlMode> {
        let raw = std::env::var(key).ok()?;
        let raw = raw.trim();
        if let Ok(v) = raw.parse::<f32>() {
            return Some(ControlMode::Constant(v));
        }
        match serde_json::from_str::<ControlMode>(raw) {
            Ok(m) => Some(m),
            Err(e) => {
                warn!("env var {} parse failed: {} (raw: {:?})", key, e, raw);
                None
            }
        }
    }
    let upper = cfg.name.to_uppercase();
    let pick_f32 = |knob: &str| -> Option<f32> {
        read_f32(&format!("BRAIN_{knob}_{upper}"))
            .or_else(|| read_f32(&format!("BRAIN_{knob}_DEFAULT")))
    };
    let pick_usize = |knob: &str| -> Option<usize> {
        read_usize(&format!("BRAIN_{knob}_{upper}"))
            .or_else(|| read_usize(&format!("BRAIN_{knob}_DEFAULT")))
    };
    let pick_u32 = |knob: &str| -> Option<u32> {
        read_u32(&format!("BRAIN_{knob}_{upper}"))
            .or_else(|| read_u32(&format!("BRAIN_{knob}_DEFAULT")))
    };
    let pick_mode = |knob: &str| -> Option<ControlMode> {
        read_control_mode(&format!("BRAIN_{knob}_{upper}"))
            .or_else(|| read_control_mode(&format!("BRAIN_{knob}_DEFAULT")))
    };
    if let Some(m) = pick_mode("SPARSITY")     { cfg.sparsity_mode = m; }
    if let Some(v) = pick_usize("SPARSITY_MIN"){ cfg.sparsity_min_neurons = v.max(1); }
    if let Some(m) = pick_mode("HET_LTD")      { cfg.heterosynaptic_ltd_mode = m; }
    if let Some(m) = pick_mode("PREDICT_GATE") { cfg.predict_gate_mode = m; }
    if let Some(v) = pick_usize("WINDOW")      { cfg.recent_atoms_window = v; }
    if let Some(v) = pick_u32("EMERGENCE")     { cfg.concept_emergence_threshold = v; }
    if let Some(v) = pick_usize("MAX_MEMBERS") { cfg.max_concept_member_count = v; }
    if let Some(v) = pick_f32("DECAY")         { cfg.decay_rate = v; }
    if let Some(v) = pick_f32("PRUNE_FLOOR")   { cfg.prune_floor = v; }
}

fn load_or_build_brain(checkpoint_path: &PathBuf) -> Result<Brain> {
    if checkpoint_path.exists() {
        info!("restoring brain from {}", checkpoint_path.display());
        let encodings = build_encodings(&pool_prefixes());
        let (brain, missing) = Brain::restore(checkpoint_path, encodings)
            .with_context(|| format!("restore {}", checkpoint_path.display()))?;
        if !missing.is_empty() {
            warn!("restore missing encodings for pool ids {:?}", missing);
        }
        Ok(brain)
    } else {
        info!("no checkpoint at {}; building fresh multimodal brain",
            checkpoint_path.display());
        build_fresh_brain()
    }
}

// -----------------------------------------------------------------
// Request / response types
// -----------------------------------------------------------------

#[derive(Deserialize)]
struct ObserveRequest {
    pool_id: PoolId,
    frame:   String, // base64-url-safe
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
    answer:                Option<String>,
    confidence_tier:       String,
    fabric_confidence:     f32,
    eem_confidence:        Option<f32>,
    annealer_confidence:   Option<f32>,
    integrated_confidence: f32,
    outside_grounding:     bool,
    speculation_flag:      bool,
}

#[derive(Serialize)]
struct StatsResponse {
    tick:                u64,
    pool_count:          usize,
    total_neurons:       usize,
    total_concepts:      usize,
    total_binding:       usize,
    total_terminals:     usize,
    binding_pool_id:     PoolId,
    fingerprints_window: usize,
    checkpoint_path:     String,
}

#[derive(Serialize)]
struct CheckpointResponse {
    written_bytes: u64,
    path:          String,
}

// ── /sensor/observe (single modality) ──

#[derive(Deserialize)]
struct SensorObserveRequest {
    kind:      String,            // "text" | "image" | "audio"
    bytes_b64: Option<String>,    // base64 (any flavor — we try both)
    text:      Option<String>,    // raw text (when kind="text")
}

#[derive(Serialize)]
struct SensorObserveResponse {
    fired_neurons: usize,
    predictions:   HashMap<String, Vec<String>>,
}

// ── /sensor/observe_triple ──

#[derive(Deserialize)]
struct ObserveTripleRequest {
    text:      String,
    image_b64: String,
    audio_b64: String,
    #[serde(default)]
    lr:        Option<f32>,
}

#[derive(Serialize)]
struct ObserveTripleResponse {
    img_labels: usize,
    aud_labels: usize,
    txt_labels: usize,
}

// ── /chat ──

#[derive(Deserialize)]
struct ChatRequest {
    text: String,
    #[serde(default)]
    max_steps: Option<usize>,
}

#[derive(Serialize)]
struct ChatResponse {
    // `reply` is the primary chat-engine response string.
    reply:           String,
    // `answer` mirrors `reply` so the Wizard-chat Django frontend's
    // legacy parser (which reads data["answer"]) sees a value here.
    answer:          String,
    // `decoder` tells the Wizard frontend which selection path
    // produced the reply.  "multi_pool" maps to high confidence in
    // the frontend's tier logic; "char_chain" maps to low.
    decoder:         String,
    predictions:     HashMap<String, Vec<String>>,
    grounding:       ChatGrounding,
    // Empty list mirrors what the legacy /chat endpoint returned so
    // the frontend's `activated_concepts` parse step has a no-op
    // fallback to walk.
    activated_concepts: Vec<String>,
    word_activations:   Vec<serde_json::Value>,
}

#[derive(Serialize, Default)]
struct ChatGrounding {
    fabric_confidence:     f32,
    integrated_confidence: f32,
    outside_grounding:     bool,
    speculation_flag:      bool,
}

// -----------------------------------------------------------------
// Handlers — basic
// -----------------------------------------------------------------

async fn health() -> &'static str { "ok\n" }

async fn stats(State(s): State<AppState>) -> Json<StatsResponse> {
    let brain = s.brain.lock().await;
    let bs: BrainStats = brain.stats();
    Json(StatsResponse {
        tick:                bs.tick,
        pool_count:          bs.pool_count,
        total_neurons:       bs.total_neurons,
        total_concepts:      bs.total_concepts,
        total_binding:       bs.total_binding,
        total_terminals:     bs.total_terminals,
        binding_pool_id:     bs.binding_pool_id,
        fingerprints_window: bs.fingerprints_window,
        checkpoint_path:     s.checkpoint_path.display().to_string(),
    })
}

async fn observe(
    State(s): State<AppState>,
    Json(req): Json<ObserveRequest>,
) -> Result<Json<ObserveResponse>, (axum::http::StatusCode, String)> {
    let frame = decode_base64_flexible(&req.frame)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("invalid base64: {}", e)))?;
    let mut brain = s.brain.lock().await;
    let fired = brain.observe(req.pool_id, &frame);
    Ok(Json(ObserveResponse { fired_neurons: fired.len() }))
}

async fn tick(State(s): State<AppState>) -> Json<u64> {
    let mut brain = s.brain.lock().await;
    brain.advance_tick();
    Json(brain.stats().tick)
}

/// Concept-first integration — the user's "deepest layer wins"
/// retrieval contract from ARCHITECTURE.md §4.D.1.  Does NOT touch
/// the legacy /integrate path.
///
/// Now ALSO returns the trained-binding decode as a sibling field
/// — this is the substrate's literal trained answer (the binding's
/// target-pool members verbatim), without the decoder residual that
/// concept-first scoring sometimes adds.
async fn integrate_concept_first(
    State(s): State<AppState>,
    Json(req): Json<IntegrateRequest>,
) -> Json<IntegrateConceptFirstResponse> {
    let brain = s.brain.lock().await;
    let scored = brain.integrate_concept_first(req.query_pool, req.target_pool);
    let trained = brain.decode_best_trained_binding(req.query_pool, req.target_pool);
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let (b64, utf8) = match &scored {
        Some(b) => (Some(engine.encode(b)),
                    std::str::from_utf8(b).ok().map(|s| s.to_string())),
        None    => (None, None),
    };
    let (tb64, tutf8) = match &trained {
        Some(b) => (Some(engine.encode(b)),
                    std::str::from_utf8(b).ok().map(|s| s.to_string())),
        None    => (None, None),
    };
    Json(IntegrateConceptFirstResponse {
        answer_b64:        b64,
        answer_utf8:       utf8,
        trained_answer_b64:  tb64,
        trained_answer_utf8: tutf8,
    })
}

#[derive(Serialize, Debug)]
struct IntegrateConceptFirstResponse {
    answer_b64:          Option<String>,
    answer_utf8:         Option<String>,
    /// Decoded bytes of the best-matching binding's target-pool
    /// members.  This is the substrate's literal trained answer
    /// for the query — the binding stores exactly what co-fired at
    /// training time, so its decode IS the trained response.
    trained_answer_b64:  Option<String>,
    trained_answer_utf8: Option<String>,
}

async fn integrate(
    State(s): State<AppState>,
    Json(req): Json<IntegrateRequest>,
) -> Json<IntegrateResponse> {
    let brain = s.brain.lock().await;
    // Keep confidence / grounding from the legacy integrate() — those
    // signals still reflect the substrate's atom-level binding-pool
    // routing and OOV gate.  But OVERRIDE the answer bytes with the
    // authoritative decoder used by /chat.  Per user direction the
    // integrate path's recall must hit 100% on trained input — that
    // requires the same Hebbian-frequency-weighted + min_atom_score
    // floor that lifts /chat to 32/32.  Without this, /integrate's
    // older atom-coverage routing picks wrong-category bindings or
    // truncates decodes (e.g. fish->'anim', hand->'aturena').
    let legacy = brain.integrate(req.query_pool, req.target_pool);
    let authoritative = brain.decode_best_trained_binding(req.query_pool, req.target_pool);
    let answer_bytes = authoritative.or(legacy.answer);
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    Json(IntegrateResponse {
        answer:                answer_bytes.as_ref().map(|b| engine.encode(b)),
        confidence_tier:       format!("{:?}", legacy.confidence_tier),
        fabric_confidence:     legacy.grounding.fabric_confidence,
        eem_confidence:        legacy.grounding.eem_confidence,
        annealer_confidence:   legacy.grounding.annealer_confidence,
        integrated_confidence: legacy.grounding.integrated_confidence,
        outside_grounding:     legacy.grounding.outside_grounding,
        speculation_flag:      legacy.grounding.speculation_flag,
    })
}

#[derive(Deserialize, Debug)]
struct ResonantRequest {
    query_pool:    PoolId,
    /// One or more target pool ids whose settled state should be
    /// decoded and returned.  Empty/omitted defaults to "every pool
    /// the brain knows about" (including the source pool, so callers
    /// see the persistent mould).
    #[serde(default)]
    target_pools:  Vec<PoolId>,
    /// Top concepts to decode per pool.  Default 8.
    #[serde(default)]
    top_per_pool:  Option<usize>,
    /// Max iterations the settling fixed-point will run.  Default 12.
    #[serde(default)]
    max_iter:      Option<usize>,
    /// Convergence threshold (L1 distance over top-K).  Default 0.01.
    #[serde(default)]
    eps:           Option<f32>,
}

#[derive(Serialize, Debug)]
struct ResonantConceptOut {
    pool:        PoolId,
    pool_name:   String,
    neuron_id:   w1z4rd_brain::NeuronId,
    label:       String,
    activation:  f32,
    bytes_b64:   String,
    bytes_utf8:  Option<String>,
}

#[derive(Serialize, Debug)]
struct ResonantPoolOut {
    pool:       PoolId,
    pool_name:  String,
    decoded:    Vec<ResonantConceptOut>,
}

#[derive(Serialize, Debug)]
struct ResonantResponse {
    iterations_run: usize,
    converged:      bool,
    pools:          Vec<ResonantPoolOut>,
}

/// Stage 13A — `/integrate_resonant`.  Runs the settling fixed-point
/// over the substrate and returns the decoded top-K concepts per
/// requested target pool.  Parallel to `/integrate`; the existing
/// retrieval path is unchanged.
async fn integrate_resonant(
    State(s): State<AppState>,
    Json(req): Json<ResonantRequest>,
) -> Json<ResonantResponse> {
    let brain = s.brain.lock().await;

    let target_pools: Vec<PoolId> = if req.target_pools.is_empty() {
        brain.fabric().pool_ids().into_iter().collect()
    } else {
        req.target_pools.clone()
    };
    let top_per_pool = req.top_per_pool.unwrap_or(8);
    let max_iter     = req.max_iter.unwrap_or(12);
    let eps          = req.eps.unwrap_or(0.01);

    let extrusion = brain.integrate_resonant(
        req.query_pool, &target_pools, top_per_pool, max_iter, eps);

    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let mut out_pools: Vec<ResonantPoolOut> = Vec::with_capacity(extrusion.pools.len());
    for pe in extrusion.pools {
        let pool_name = pool_name(pe.pool).to_string();
        let decoded: Vec<ResonantConceptOut> = pe.decoded.into_iter().map(|d| {
            let bytes_b64 = engine.encode(&d.bytes);
            let bytes_utf8 = std::str::from_utf8(&d.bytes).ok().map(|s| s.to_string());
            ResonantConceptOut {
                pool:       pe.pool,
                pool_name:  pool_name.clone(),
                neuron_id:  d.neuron.neuron,
                label:      d.label,
                activation: d.activation,
                bytes_b64,
                bytes_utf8,
            }
        }).collect();
        out_pools.push(ResonantPoolOut {
            pool:      pe.pool,
            pool_name,
            decoded,
        });
    }

    Json(ResonantResponse {
        iterations_run: extrusion.iterations_run,
        converged:      extrusion.converged,
        pools:          out_pools,
    })
}

#[derive(Deserialize, Debug)]
struct PoolConceptsRequest {
    pool_id:     PoolId,
    /// Substring filter on decoded bytes (case-insensitive). Empty
    /// = no filter, returns all concepts up to `limit`.
    #[serde(default)]
    contains:    Option<String>,
    #[serde(default)]
    limit:       Option<usize>,
}

#[derive(Serialize, Debug)]
struct PoolConceptOut {
    neuron_id:   w1z4rd_brain::NeuronId,
    label:       String,
    member_count: usize,
    decoded:     String,
    use_count:   u64,
}

#[derive(Serialize, Debug)]
struct PoolConceptsResponse {
    pool:        PoolId,
    pool_name:   String,
    total_concepts: usize,
    returned:    usize,
    concepts:    Vec<PoolConceptOut>,
}

/// Diagnostic — list emerged CONCEPT neurons in a pool with their
/// decoded byte sequence.  Read-only.  Used to verify which
/// hierarchical concepts the substrate has actually built.
async fn pool_concepts(
    State(s): State<AppState>,
    Json(req): Json<PoolConceptsRequest>,
) -> Result<Json<PoolConceptsResponse>, (axum::http::StatusCode, String)> {
    let brain = s.brain.lock().await;
    let pool_handle = brain.fabric().pool(req.pool_id).ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        format!("unknown pool id {}", req.pool_id),
    ))?;
    let pool = pool_handle.read();
    let limit = req.limit.unwrap_or(64);
    let filter = req.contains.as_deref().map(|s| s.to_lowercase());

    let mut total = 0usize;
    let mut out: Vec<PoolConceptOut> = Vec::new();
    for n in pool.iter_neurons() {
        if n.is_atom() { continue; }
        total += 1;
        let bytes = pool.decode_concept_members(&n.members);
        let decoded = String::from_utf8_lossy(&bytes).into_owned();
        if let Some(f) = &filter {
            if !decoded.to_lowercase().contains(f) {
                continue;
            }
        }
        if out.len() < limit {
            out.push(PoolConceptOut {
                neuron_id:    n.id,
                label:        n.label.clone(),
                member_count: n.members.len(),
                decoded,
                use_count:    n.use_count,
            });
        }
    }

    Ok(Json(PoolConceptsResponse {
        pool:           req.pool_id,
        pool_name:      pool_name(req.pool_id).to_string(),
        total_concepts: total,
        returned:       out.len(),
        concepts:       out,
    }))
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

/// `POST /flush` — per [`ARCHITECTURE.md`] §17.9, force-flush the WAL to
/// disk.  Cheap O(buffer) operation.  Use between training phases for a
/// quick durability barrier without paying the bincode-snapshot cost.
/// On a NoopStore brain (no WAL attached) this is a no-op acknowledgment.
#[derive(Serialize)]
struct FlushResponse {
    wal_bytes: u64,
}

async fn flush(
    State(s): State<AppState>,
) -> Result<Json<FlushResponse>, (axum::http::StatusCode, String)> {
    let brain = s.brain.lock().await;
    let store = brain.store_clone();
    store.flush().map_err(|e| {
        (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
         format!("WAL flush failed: {}", e))
    })?;
    Ok(Json(FlushResponse { wal_bytes: store.log_size_bytes() }))
}

#[derive(Deserialize, Default, Clone)]
struct SleepRequest {
    #[serde(default = "default_sleep_min_use_count")]
    min_use_count:    u64,
    #[serde(default = "default_sleep_stale_ticks")]
    stale_ticks:      u64,
    #[serde(default)]
    replay_count:     usize,
    #[serde(default = "default_sleep_replay_strength")]
    replay_strength:  f32,
    /// Stage 17.4: when true, spawn the sleep as a background tokio task
    /// and return immediately.  Poll `GET /sleep/status` for progress.
    /// Default false for backward compatibility — the legacy synchronous
    /// shape still works.
    #[serde(default)]
    background:       bool,
    /// Stage 17.7 full: when > 0.0, replay uses Boltzmann sampling over
    /// salience scores at this temperature (free-energy weighting).
    /// Default 0.0 = legacy uniform-temporal-order replay.  Suggested
    /// production value: 2.0 (soft preference for high-salience moments
    /// with some exploration).
    #[serde(default)]
    replay_beta:      f32,
    /// Stage 17.7 full: seed for the Boltzmann sampler.  Only used when
    /// replay_beta > 0.  Default 0 → implementation falls back to a
    /// process-specific constant; pass a non-zero value for reproducible
    /// runs.
    #[serde(default)]
    replay_seed:      u64,
}
fn default_sleep_min_use_count()  -> u64 { 2 }
fn default_sleep_stale_ticks()    -> u64 { 1000 }
fn default_sleep_replay_strength()-> f32 { 0.5 }

#[derive(Serialize)]
struct SleepResponse {
    pruned:    usize,
    replayed:  usize,
    tick_now:  u64,
    /// Stage 17.4 additions — backward compatible because old clients
    /// just ignore these.
    #[serde(default)]
    job_id:     u64,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    phase:      String,
}

/// P4 sleep cycle per [`ARCHITECTURE.md`] §17.4 — decomposed per-pool with
/// brain-mutex released between phases so /stats, /chat, /flush stay
/// responsive throughout.  Two response modes:
///
/// - `{ "background": false }` (default — backward compatible): handler
///   runs the full sleep synchronously and returns the totals.  Same shape
///   as the legacy /sleep response.  Long sleep operations may exceed
///   HTTP client timeouts at scale; use background mode then.
/// - `{ "background": true }`: handler spawns a tokio task to run the
///   sleep cycle, returns immediately with the `job_id` and current
///   status.  Poll `GET /sleep/status` for progress.
///
/// SHY hypothesis (Tononi & Cirelli, 2014) for the prune half;
/// hippocampal replay (Wilson & McNaughton, 1994; McClelland et al., 1995
/// CLS) for the replay half.  Both decomposed across pools so a sleep
/// pass on a 100M-terminal brain doesn't hold the brain mutex for the
/// entire scan.
async fn sleep_cycle(
    State(s): State<AppState>,
    Json(req): Json<SleepRequest>,
) -> Json<SleepResponse> {
    // Pick up the pool list and starting tick under a brief lock.
    let (pool_ids, tick_start) = {
        let brain = s.brain.lock().await;
        (brain.fabric().pool_ids(), brain.fabric().current_tick())
    };

    // Allocate a job id from the status slot (cheap monotonic counter).
    let job_id = {
        let mut slot = s.sleep_status.lock().unwrap();
        let next = slot.as_ref().map(|j| j.job_id + 1).unwrap_or(1);
        *slot = Some(SleepJobStatus::new(next, pool_ids.len(), tick_start));
        next
    };

    if req.background {
        // Spawn background; return immediately with current status.
        let state_for_task = s.clone();
        let req_for_task = req.clone();
        let pool_ids_for_task = pool_ids.clone();
        tokio::spawn(async move {
            run_decomposed_sleep(state_for_task, req_for_task, pool_ids_for_task, tick_start).await;
        });
        let snap = s.sleep_status.lock().unwrap().clone().unwrap();
        return Json(SleepResponse {
            pruned:    snap.pruned_so_far,
            replayed:  snap.replayed,
            tick_now:  tick_start,
            job_id,
            background: true,
            phase:     snap.phase,
        });
    }

    // Foreground mode: run the same decomposition inline, but yield
    // tokio between phases so /stats can interleave even within one
    // /sleep request.
    run_decomposed_sleep(s.clone(), req.clone(), pool_ids, tick_start).await;
    let snap = s.sleep_status.lock().unwrap().clone().unwrap();
    Json(SleepResponse {
        pruned:    snap.pruned_so_far,
        replayed:  snap.replayed,
        tick_now:  snap.tick_end,
        job_id,
        background: false,
        phase:     snap.phase,
    })
}

/// Drives the per-pool decomposed sleep cycle.  Called inline by the
/// foreground branch and via tokio::spawn by the background branch.
async fn run_decomposed_sleep(
    s:          AppState,
    req:        SleepRequest,
    pool_ids:   Vec<PoolId>,
    _tick_start: u64,
) {
    // AHashSet matches the Brain API's sleep_pool_phase2 signature.
    let mut all_pruned: ahash::AHashSet<w1z4rd_brain::NeuronRef> =
        ahash::AHashSet::new();

    // PHASE 1 — per-pool prune.  Brain mutex released between iterations.
    for pid in &pool_ids {
        let pruned = {
            let brain = s.brain.lock().await;
            brain.sleep_pool_phase1(*pid, req.min_use_count, req.stale_ticks)
        };
        // brain mutex released — /stats can interleave here.
        {
            let mut slot = s.sleep_status.lock().unwrap();
            if let Some(j) = slot.as_mut() {
                j.pruned_so_far += pruned.len();
                j.pools_phase1_done += 1;
            }
        }
        all_pruned.extend(pruned.into_iter());
        tokio::task::yield_now().await;
    }

    // PHASE 2 — per-pool cross-pool inbound cleanup.
    if !all_pruned.is_empty() {
        for pid in &pool_ids {
            {
                let brain = s.brain.lock().await;
                brain.sleep_pool_phase2(*pid, &all_pruned);
            }
            {
                let mut slot = s.sleep_status.lock().unwrap();
                if let Some(j) = slot.as_mut() {
                    j.pools_phase2_done += 1;
                }
            }
            tokio::task::yield_now().await;
        }
    } else {
        // Skip phase 2 cleanly; mark all pools done so status is consistent.
        let mut slot = s.sleep_status.lock().unwrap();
        if let Some(j) = slot.as_mut() {
            j.pools_phase2_done = pool_ids.len();
        }
    }

    // PHASE 3 — per-pool housekeeping (decay + sub-floor prune).
    for pid in &pool_ids {
        {
            let brain = s.brain.lock().await;
            brain.sleep_pool_housekeeping(*pid);
        }
        {
            let mut slot = s.sleep_status.lock().unwrap();
            if let Some(j) = slot.as_mut() {
                j.pools_phase3_done += 1;
            }
        }
        tokio::task::yield_now().await;
    }

    // REPLAY — CLS consolidation.  Brief brain-mutex hold; replay is
    // bounded by `count` so it's not the unbounded scan that phases 1-3
    // could be.  Stage 17.7 full: route through the free-energy
    // weighted sampler when `replay_beta > 0`.
    let replayed = if req.replay_count > 0 {
        let mut brain = s.brain.lock().await;
        if req.replay_beta > 0.0 {
            brain.replay_free_energy_weighted(
                req.replay_count, req.replay_strength,
                req.replay_beta, req.replay_seed,
            )
        } else {
            brain.replay_recent_moments(req.replay_count, req.replay_strength)
        }
    } else { 0 };

    let tick_end = {
        let brain = s.brain.lock().await;
        brain.fabric().current_tick()
    };
    let finished_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64).unwrap_or(0);
    let mut slot = s.sleep_status.lock().unwrap();
    if let Some(j) = slot.as_mut() {
        j.phase = "complete".into();
        j.replayed = replayed;
        j.tick_end = tick_end;
        j.finished_at_ms = finished_at_ms;
    }
}

/// `GET /sleep/status` — per [`ARCHITECTURE.md`] §17.4 — returns the
/// current (or most recent) sleep job's progress.  Useful when /sleep
/// was invoked with `background: true` and the caller wants to know
/// when consolidation has finished.  Returns 404 if no sleep has ever
/// been requested on this brain instance.
async fn sleep_status(
    State(s): State<AppState>,
) -> Result<Json<SleepJobStatus>, (axum::http::StatusCode, String)> {
    let slot = s.sleep_status.lock().unwrap();
    match slot.as_ref() {
        Some(j) => Ok(Json(j.clone())),
        None    => Err((axum::http::StatusCode::NOT_FOUND,
                        "no sleep job has been requested".into())),
    }
}

/// `GET /storage_state` — per [`ARCHITECTURE.md`] §17.8 — returns the
/// brain's current storage-tier observables.  Used by tooling, GA
/// search, and Wizard frontend to inspect the dynamical-system state.
async fn storage_state(
    State(s): State<AppState>,
) -> Json<w1z4rd_brain::store::StorageControlState> {
    let brain = s.brain.lock().await;
    Json(brain.storage_control_state())
}

/// `POST /eviction` — per [`ARCHITECTURE.md`] §17.4 — runs one pass of
/// the eviction actor.  Body is the `EvictionParams` (`max_salience_ema`,
/// `min_stale_ticks`, `target_per_pool`); all default if omitted.
/// Returns `EvictionStats`.  Called between training phases when memory
/// pressure rises; future Stage 17.8 dynamical wiring will drive this
/// from `working_set_pressure` automatically.
#[derive(Deserialize, Default)]
struct EvictionRequest {
    #[serde(default = "default_evict_max_salience")]
    max_salience_ema: f32,
    #[serde(default = "default_evict_stale_ticks")]
    min_stale_ticks:  u64,
    #[serde(default = "default_evict_target")]
    target_per_pool:  usize,
}
fn default_evict_max_salience() -> f32   { 0.1 }
fn default_evict_stale_ticks()  -> u64   { 1000 }
fn default_evict_target()       -> usize { 1024 }

async fn eviction(
    State(s): State<AppState>,
    Json(req): Json<EvictionRequest>,
) -> Json<w1z4rd_brain::EvictionStats> {
    let params = w1z4rd_brain::EvictionParams {
        max_salience_ema: req.max_salience_ema,
        min_stale_ticks:  req.min_stale_ticks,
        target_per_pool:  req.target_per_pool,
    };
    let brain = s.brain.lock().await;
    let stats = brain.run_eviction_pass(params);
    Json(stats)
}

// -----------------------------------------------------------------
// Stage 17.6 — cluster anti-entropy HTTP endpoints
// -----------------------------------------------------------------

/// `GET /cluster/pool_roots` — per [`ARCHITECTURE.md`] §17.6 — returns
/// each pool's deterministic Merkle root hex.  A peer compares its
/// map against ours to identify divergent pools.
#[derive(Serialize)]
struct PoolRootsResponse {
    /// pool_id → hex Merkle root (64 chars, BLAKE3)
    roots: HashMap<PoolId, String>,
    /// Brain's current tick at the moment roots were computed.  Cluster
    /// sync protocols use this to age-order conflicting states.
    tick:  u64,
}

async fn cluster_pool_roots(
    State(s): State<AppState>,
) -> Json<PoolRootsResponse> {
    let brain = s.brain.lock().await;
    let roots_raw = brain.cluster_pool_roots();
    let mut roots = HashMap::new();
    for (pid, r) in roots_raw {
        roots.insert(pid, r.to_hex());
    }
    Json(PoolRootsResponse {
        roots,
        tick: brain.fabric().current_tick(),
    })
}

/// `GET /cluster/pool_neurons/{pool_id}` — per [`ARCHITECTURE.md`] §17.6 —
/// returns the authoritative neuron list for `pool_id`.  Used by a peer
/// to fetch state it doesn't have (after detecting divergence via
/// `/cluster/pool_roots`).  Response is bincode-encoded for size.
async fn cluster_pool_neurons(
    State(s): State<AppState>,
    axum::extract::Path(pool_id): axum::extract::Path<PoolId>,
) -> Result<axum::response::Response<axum::body::Body>, (axum::http::StatusCode, String)> {
    let brain = s.brain.lock().await;
    let neurons = brain.cluster_pool_neurons(pool_id).ok_or((
        axum::http::StatusCode::NOT_FOUND,
        format!("no pool with id {}", pool_id),
    ))?;
    let bytes = bincode::serialize(&neurons).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        format!("serialize failed: {}", e),
    ))?;
    Ok(axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .body(axum::body::Body::from(bytes))
        .unwrap())
}

/// `POST /cluster/merge_pool` — per [`ARCHITECTURE.md`] §17.6 — accepts
/// neurons from a peer and merges them into the named pool.  Only
/// neurons whose id is the next sequential slot in the local pool are
/// inserted (preserves the sequential-id contract); incoming neurons
/// for already-filled slots are silently skipped.  Returns the count
/// of newly-inserted neurons.
#[derive(Deserialize)]
struct MergePoolRequest {
    pool_id: PoolId,
    /// base64-encoded bincode of `Vec<Neuron>`.  Same wire format as
    /// `GET /cluster/pool_neurons/{pool_id}` ⇒ symmetrical pull/push.
    neurons_b64: String,
}

#[derive(Serialize)]
struct MergePoolResponse {
    inserted: usize,
}

async fn cluster_merge_pool(
    State(s): State<AppState>,
    Json(req): Json<MergePoolRequest>,
) -> Result<Json<MergePoolResponse>, (axum::http::StatusCode, String)> {
    let bytes = decode_base64_flexible(&req.neurons_b64).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        format!("neurons_b64 decode: {}", e),
    ))?;
    let neurons: Vec<w1z4rd_brain::Neuron> = bincode::deserialize(&bytes).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        format!("neurons bincode: {}", e),
    ))?;
    let brain = s.brain.lock().await;
    let inserted = brain.cluster_merge_pool(req.pool_id, neurons);
    Ok(Json(MergePoolResponse { inserted }))
}

/// `POST /cluster/pull_from` — per [`ARCHITECTURE.md`] §17.6 — the
/// active anti-entropy sync client.  Hits the peer's /cluster/pool_roots
/// endpoint, identifies pools whose roots differ from ours, fetches the
/// peer's neuron list for each diverged pool, and merges into local
/// state via cluster_merge_pool.
///
/// This is "pull" (we ask the peer for state) rather than "push" (we
/// send state to the peer) — pull is safer because the local merge
/// policy stays in our control.
///
/// Request: `{ "peer_url": "http://192.168.1.43:8095" }`.  Optional
/// `pool_ids` whitelist restricts sync to specific pools.
#[derive(Deserialize)]
struct PullFromRequest {
    peer_url: String,
    #[serde(default)]
    pool_ids: Option<Vec<PoolId>>,
}

#[derive(Serialize, Default)]
struct PullFromResponse {
    peer_tick:           u64,
    local_tick:          u64,
    pools_compared:      usize,
    pools_diverged:      usize,
    pools_synced:        usize,
    neurons_inserted:    usize,
    errors:              Vec<String>,
}

// -----------------------------------------------------------------
// Stage 18.12 step 5 — cluster membership endpoints per
// [`ARCHITECTURE.md`] §18.4.  The seed-driven join protocol that
// extends legacy crates/cluster OTP membership to the brain-shard
// layer.
// -----------------------------------------------------------------

/// `GET /cluster/members` — return current membership snapshot.
async fn cluster_members(
    State(s): State<AppState>,
) -> Json<ClusterMembership> {
    Json(s.cluster.lock().unwrap().clone())
}

/// `POST /cluster/join` — a peer requests to join this node's ring.
/// Caller body: `{ addr, capacity_neurons? }`.  Response: the
/// post-join membership snapshot (joiner copies the ring into its
/// own state).
///
/// The seed-side (this endpoint) allocates a NodeId, adds the peer,
/// and returns the updated ring.  When the joiner has further peers
/// to inform, it propagates by calling join on each — or relies on
/// the heartbeat layer (Stage 18.12 step 8) to spread updates.
#[derive(Deserialize)]
struct JoinRequest {
    addr: String,
    #[serde(default)]
    capacity_neurons: Option<u64>,
}

#[derive(Serialize)]
struct JoinResponse {
    /// NodeId the seed allocated for the joiner.
    assigned_node_id: w1z4rd_brain::store::NodeId,
    /// Full membership snapshot post-join.  Joiner overwrites its
    /// local state with this.
    membership: ClusterMembership,
}

async fn cluster_join(
    State(s): State<AppState>,
    Json(req): Json<JoinRequest>,
) -> Result<Json<JoinResponse>, (axum::http::StatusCode, String)> {
    let (assigned, snapshot, local_node) = {
        let mut m = s.cluster.lock().unwrap();
        // Reject duplicates by addr — idempotent on retries.
        if let Some(existing) = m.members.iter().find(|mi| mi.addr == req.addr) {
            let id = existing.node_id;
            return Ok(Json(JoinResponse {
                assigned_node_id: id,
                membership: m.clone(),
            }));
        }
        let assigned = w1z4rd_brain::store::NodeId(m.next_node_id);
        m.next_node_id = m.next_node_id.saturating_add(1);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64).unwrap_or(0);
        m.members.push(MemberInfo {
            node_id: assigned,
            addr:    req.addr.clone(),
            last_heartbeat_ms: now_ms,
            capacity_neurons:  req.capacity_neurons,
        });
        m.members.sort_by_key(|mi| mi.node_id.0);
        let snapshot = m.clone();
        let local_node = m.local_node;
        (assigned, snapshot, local_node)
    };
    // Stage 18.12 step 6: peer joined — re-wire local topology so the
    // placement policy now considers the new ring member.
    wire_cluster_topology(&s, &snapshot, local_node).await;
    Ok(Json(JoinResponse { assigned_node_id: assigned, membership: snapshot }))
}

/// `POST /cluster/leave` — peer is leaving gracefully.  Body:
/// `{ node_id }`.  Returns the post-leave membership.
#[derive(Deserialize)]
struct LeaveRequest {
    node_id: w1z4rd_brain::store::NodeId,
}

async fn cluster_leave(
    State(s): State<AppState>,
    Json(req): Json<LeaveRequest>,
) -> Json<ClusterMembership> {
    let snapshot;
    let local_node;
    {
        let mut m = s.cluster.lock().unwrap();
        m.members.retain(|mi| mi.node_id != req.node_id);
        local_node = m.local_node;
        snapshot = m.clone();
    }
    // Stage 18.12 step 6: re-wire topology on departure too.
    wire_cluster_topology(&s, &snapshot, local_node).await;
    Json(snapshot)
}

/// `GET /cluster/aggregate_pool_neurons/{pool_id}` — per
/// [`ARCHITECTURE.md`] §18.8 gossip bridge.  Returns the unified view
/// of `pool_id` across the entire cluster: for each ring member, fetch
/// its /cluster/pool_neurons/{pool_id}, dedupe by id preferring the
/// copy with non-empty terminals (so post-eviction shipments win over
/// stale local placeholders).
///
/// This is the endpoint a standalone §17.6 anti-entropy peer should
/// call against a cluster head when it wants to see the cluster's full
/// state for a pool, treating the cluster as one logical brain.
async fn cluster_aggregate_pool_neurons(
    State(s): State<AppState>,
    axum::extract::Path(pool_id): axum::extract::Path<PoolId>,
) -> Result<axum::response::Response<axum::body::Body>, (axum::http::StatusCode, String)> {
    // Snapshot ring.
    let (local_node, members) = {
        let m = s.cluster.lock().unwrap();
        (m.local_node, m.members.clone())
    };

    // Collect contributions: (member_node_id, Vec<Neuron>).
    let mut contributions: Vec<(w1z4rd_brain::store::NodeId, Vec<w1z4rd_brain::Neuron>)> = Vec::new();

    // 1. Local contribution.
    {
        let brain = s.brain.lock().await;
        let local = brain.cluster_pool_neurons(pool_id).unwrap_or_default();
        contributions.push((local_node, local));
    }

    // 2. Remote contributions.  Best-effort: failures are warnings,
    // not fatal.
    let client = reqwest::Client::new();
    for peer in &members {
        if peer.node_id == local_node { continue; }
        let url = format!(
            "{}/cluster/pool_neurons/{}",
            peer.addr.trim_end_matches('/'),
            pool_id,
        );
        match client.get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send().await
        {
            Ok(resp) if resp.status().is_success() => {
                match resp.bytes().await {
                    Ok(body) => match bincode::deserialize::<Vec<w1z4rd_brain::Neuron>>(&body) {
                        Ok(ns) => contributions.push((peer.node_id, ns)),
                        Err(e) => warn!(
                            "aggregate: peer {} bincode parse failed: {}",
                            peer.node_id.0, e,
                        ),
                    },
                    Err(e) => warn!(
                        "aggregate: peer {} body read failed: {}",
                        peer.node_id.0, e,
                    ),
                }
            }
            Ok(resp) => warn!(
                "aggregate: peer {} status {}",
                peer.node_id.0, resp.status(),
            ),
            Err(e) => warn!(
                "aggregate: peer {} network: {}",
                peer.node_id.0, e,
            ),
        }
    }

    // 3. Dedupe by id, preferring the contribution with non-empty
    // terminals.  Sequential-id contract: ids are stable across the
    // ring (consistent-hash placement uses the same id space).
    use std::collections::HashMap as StdHashMap;
    let mut by_id: StdHashMap<w1z4rd_brain::NeuronId, w1z4rd_brain::Neuron> = StdHashMap::new();
    for (_node, ns) in contributions {
        for n in ns {
            let existing_has_terms = by_id.get(&n.id)
                .map(|e| !e.terminals.is_empty())
                .unwrap_or(false);
            let new_has_terms = !n.terminals.is_empty();
            if !existing_has_terms || new_has_terms {
                by_id.insert(n.id, n);
            }
        }
    }
    // Reassemble ordered by id for determinism.
    let mut merged: Vec<_> = by_id.into_iter().collect();
    merged.sort_by_key(|(id, _)| *id);
    let merged_neurons: Vec<w1z4rd_brain::Neuron> = merged.into_iter()
        .map(|(_, n)| n).collect();

    let body = bincode::serialize(&merged_neurons).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        format!("aggregate serialize: {}", e),
    ))?;
    Ok(axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .header("x-cluster-members", format!("{}", members.len()))
        .body(axum::body::Body::from(body))
        .unwrap())
}

/// `POST /cluster/heartbeat { from_node_id }` — peer announces it's alive.
/// Updates that member's `last_heartbeat_ms`.  Per [`ARCHITECTURE.md`]
/// §18.11.
#[derive(Deserialize)]
struct HeartbeatRequest {
    from_node_id: w1z4rd_brain::store::NodeId,
}
#[derive(Serialize)]
struct HeartbeatResponse {
    acknowledged: bool,
    member_count: usize,
}

async fn cluster_heartbeat(
    State(s): State<AppState>,
    Json(req): Json<HeartbeatRequest>,
) -> Json<HeartbeatResponse> {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64).unwrap_or(0);
    let mut m = s.cluster.lock().unwrap();
    let acked = m.members.iter_mut()
        .find(|x| x.node_id == req.from_node_id)
        .map(|x| { x.last_heartbeat_ms = now_ms; true })
        .unwrap_or(false);
    Json(HeartbeatResponse {
        acknowledged: acked,
        member_count: m.members.len(),
    })
}

/// Stage 18.12 step 8 — background heartbeat + stale-detection loop.
/// Per [`ARCHITECTURE.md`] §18.11.
///
/// Runs continuously after startup.  Every 5 seconds:
/// 1. POST /cluster/heartbeat to every non-self ring member.
/// 2. Sweep our own ring; any member whose last_heartbeat_ms is more
///    than 30s old is considered dead and removed.  Self never expires.
/// 3. If membership changed, re-wire the topology so the TieredStore
///    no longer routes to the dead peer.
async fn heartbeat_loop(state: AppState) {
    use std::time::Duration as StdDuration;
    let interval = StdDuration::from_secs(5);
    let dead_threshold_ms: i64 = 30_000;
    let client = reqwest::Client::new();
    loop {
        tokio::time::sleep(interval).await;
        // Snapshot membership.
        let (local_node, members) = {
            let m = state.cluster.lock().unwrap();
            (m.local_node, m.members.clone())
        };
        // 1. Send heartbeats.
        for peer in &members {
            if peer.node_id == local_node { continue; }
            let url = format!("{}/cluster/heartbeat",
                peer.addr.trim_end_matches('/'));
            let body = serde_json::json!({ "from_node_id": local_node });
            // Fire-and-forget — failures get caught by the sweep.
            let _ = client.post(&url)
                .json(&body)
                .timeout(StdDuration::from_secs(3))
                .send()
                .await;
        }
        // 2. Sweep stale.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64).unwrap_or(0);
        let (removed, snapshot) = {
            let mut m = state.cluster.lock().unwrap();
            let before = m.members.len();
            m.members.retain(|p|
                p.node_id == local_node
                || (now_ms - p.last_heartbeat_ms) < dead_threshold_ms);
            let removed = before - m.members.len();
            (removed, m.clone())
        };
        if removed > 0 {
            warn!(
                "Stage 18.12 step 8: removed {} stale member(s); re-wiring topology",
                removed,
            );
            // 3. Re-wire so the dead peer drops out of TieredStore routing.
            wire_cluster_topology(&state, &snapshot, local_node).await;
        }
    }
}

/// Stage 18.12 step 6 — propagate the current `ClusterMembership` to
/// every Pool's `TieredStore` so the placement policy operates on the
/// up-to-date ring.  Called after seed-join, after every successful
/// /cluster/join (seed side), and after /cluster/leave.
///
/// For each pool:
/// 1. Build a `RamStore` mirroring the local node's storage of that pool.
/// 2. Build one `RemoteNodeStore` per non-local member, each backed by
///    an `HttpRemoteTransport` at the member's advertised addr.
/// 3. Build a `TieredStore::solo(local_node, ram).set_remote(...)` and
///    `set_ring(...)` with the full ring.
/// 4. Attach it via `pool.set_tiered_store(arc)`.
///
/// Subsequent evict_neuron / page_in_neuron calls then route through
/// the TieredStore: local-home goes to RamStore (no functional change
/// from §17.4); remote-home goes over the wire to the peer's
/// /shard/put_neuron and /shard/neuron endpoints.
async fn wire_cluster_topology(
    state: &AppState,
    membership: &ClusterMembership,
    local_node: w1z4rd_brain::store::NodeId,
) {
    use std::sync::Arc as StdArc;
    use w1z4rd_brain::store::{
        NodeId, RamStore, RemoteNodeStore, TieredStore,
    };
    let brain = state.brain.lock().await;
    let pool_ids = brain.fabric().pool_ids();
    let mut pools_wired = 0usize;
    for pid in pool_ids {
        let Some(pool_arc) = brain.fabric().pool(pid) else { continue; };
        // Build a fresh RamStore — empty here.  The §18.4b thin hook
        // intercepts evict/page_in; this RamStore is the "remote-tier
        // miss / local-home destination" target.  It does NOT mirror
        // the pool's existing Vec<Neuron> (the step 4b-full refactor
        // would migrate Pool::neurons into RamStore wholesale; for
        // now they coexist).
        let ram = StdArc::new(RamStore::with_node_id(local_node));
        let tiered = StdArc::new(TieredStore::solo(local_node, ram));

        // Add a RemoteNodeStore for each non-self member.
        for m in &membership.members {
            if m.node_id == local_node { continue; }
            let transport = cluster::arc_transport(&m.addr);
            let remote = StdArc::new(RemoteNodeStore::new(
                transport, pid, m.node_id,
            ));
            tiered.set_remote(m.node_id, remote);
        }
        // Apply the full ring so placement spans all members.
        let ring: Vec<NodeId> = membership.members.iter().map(|m| m.node_id).collect();
        tiered.set_ring(ring);

        pool_arc.write().set_tiered_store(tiered);
        pools_wired += 1;
    }
    info!(
        "Stage 18.12 step 6: wired cluster topology into {} pool(s); local_node={} ring_size={}",
        pools_wired, local_node.0, membership.members.len(),
    );
}

// -----------------------------------------------------------------
// Stage 18.12 step 3 — per-neuron shard RPC endpoints
//
// These are the fine-grained transports used by RemoteNodeStore in the
// node crate.  GET fetches one neuron's full state by (pool, id); POST
// inserts/overwrites one neuron.  Bincode-bodied for size and speed.
// -----------------------------------------------------------------

/// `GET /shard/neuron/{pool_id}/{neuron_id}` — return the bincode-encoded
/// `Neuron` at the given coordinates.  404 if absent (id out of range
/// or no such pool).  Per [`ARCHITECTURE.md`] §18.5 (operation routing).
async fn shard_get_neuron(
    State(s): State<AppState>,
    axum::extract::Path((pool_id, neuron_id)):
        axum::extract::Path<(PoolId, w1z4rd_brain::NeuronId)>,
) -> Result<axum::response::Response<axum::body::Body>,
            (axum::http::StatusCode, String)>
{
    let brain = s.brain.lock().await;
    let pool = brain.fabric().pool(pool_id).ok_or((
        axum::http::StatusCode::NOT_FOUND,
        format!("no pool with id {}", pool_id),
    ))?;
    let neuron = {
        let p = pool.read();
        p.get(neuron_id).cloned()
    };
    let neuron = neuron.ok_or((
        axum::http::StatusCode::NOT_FOUND,
        format!("no neuron with id {} in pool {}", neuron_id, pool_id),
    ))?;
    let body = bincode::serialize(&neuron).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        format!("bincode serialize: {}", e),
    ))?;
    Ok(axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .body(axum::body::Body::from(body))
        .unwrap())
}

/// `POST /shard/put_neuron` — insert or overwrite the neuron at
/// `(pool_id, neuron.id)`.  Body is base64-encoded bincode of a struct
/// `{ pool_id: PoolId, neuron_b64: String }`; we use base64 so the JSON
/// envelope works through standard HTTP middleware without needing
/// raw-binary handling.  Refuses if the pool doesn't exist.
#[derive(Deserialize)]
struct ShardPutRequest {
    pool_id:    PoolId,
    neuron_b64: String,
}

#[derive(Serialize)]
struct ShardPutResponse {
    inserted: bool,
}

async fn shard_put_neuron(
    State(s): State<AppState>,
    Json(req): Json<ShardPutRequest>,
) -> Result<Json<ShardPutResponse>, (axum::http::StatusCode, String)> {
    let bytes = decode_base64_flexible(&req.neuron_b64).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        format!("neuron_b64 decode: {}", e),
    ))?;
    let neuron: w1z4rd_brain::Neuron = bincode::deserialize(&bytes).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        format!("neuron bincode: {}", e),
    ))?;
    // Stage 18.12 step 6+ — use the §18-aware accept_shard_insert which
    // handles arbitrary ids by padding with placeholders.  The legacy
    // §17.6 cluster_merge_pool requires sequential ids and rejects
    // shard puts whose ids don't match the receiver's next slot.
    let brain = s.brain.lock().await;
    let inserted = match brain.fabric().pool(req.pool_id) {
        Some(pool) => pool.write().accept_shard_insert(neuron),
        None => false,
    };
    Ok(Json(ShardPutResponse { inserted }))
}

async fn cluster_pull_from(
    State(s): State<AppState>,
    Json(req): Json<PullFromRequest>,
) -> Result<Json<PullFromResponse>, (axum::http::StatusCode, String)> {
    use std::collections::HashMap as StdHashMap;
    let peer = req.peer_url.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();

    // 1. Fetch peer's pool roots.
    let peer_roots_url = format!("{}/cluster/pool_roots", peer);
    let peer_resp = client.get(&peer_roots_url).send().await.map_err(|e| (
        axum::http::StatusCode::BAD_GATEWAY,
        format!("peer /cluster/pool_roots: {}", e),
    ))?;
    if !peer_resp.status().is_success() {
        return Err((axum::http::StatusCode::BAD_GATEWAY,
            format!("peer /cluster/pool_roots status: {}", peer_resp.status())));
    }
    #[derive(Deserialize)]
    struct PeerRootsResponse {
        roots: StdHashMap<PoolId, String>,
        tick:  u64,
    }
    let peer_roots: PeerRootsResponse = peer_resp.json().await.map_err(|e| (
        axum::http::StatusCode::BAD_GATEWAY,
        format!("peer pool_roots parse: {}", e),
    ))?;

    // 2. Local roots.
    let (local_roots_hex, local_tick) = {
        let brain = s.brain.lock().await;
        let raw = brain.cluster_pool_roots();
        let tick = brain.fabric().current_tick();
        let mut h = StdHashMap::new();
        for (pid, r) in raw { h.insert(pid, r.to_hex()); }
        (h, tick)
    };

    let mut out = PullFromResponse {
        peer_tick:  peer_roots.tick,
        local_tick,
        ..Default::default()
    };

    // 3. Identify diverged pools.
    let candidates: Vec<PoolId> = match &req.pool_ids {
        Some(ids) => ids.clone(),
        None      => peer_roots.roots.keys().copied().collect(),
    };
    for pid in candidates {
        out.pools_compared += 1;
        let peer_root  = peer_roots.roots.get(&pid);
        let local_root = local_roots_hex.get(&pid);
        if peer_root == local_root { continue; }
        if peer_root.is_none() { continue; }  // peer doesn't have this pool
        out.pools_diverged += 1;

        // 4. Fetch peer's neuron list for this pool.
        let pn_url = format!("{}/cluster/pool_neurons/{}", peer, pid);
        let pn_resp = match client.get(&pn_url).send().await {
            Ok(r)  => r,
            Err(e) => {
                out.errors.push(format!("pool {}: GET failed: {}", pid, e));
                continue;
            }
        };
        if !pn_resp.status().is_success() {
            out.errors.push(format!("pool {}: GET status {}", pid, pn_resp.status()));
            continue;
        }
        let body = match pn_resp.bytes().await {
            Ok(b)  => b,
            Err(e) => {
                out.errors.push(format!("pool {}: body read: {}", pid, e));
                continue;
            }
        };
        let neurons: Vec<w1z4rd_brain::Neuron> = match bincode::deserialize(&body) {
            Ok(n)  => n,
            Err(e) => {
                out.errors.push(format!("pool {}: bincode deserialize: {}", pid, e));
                continue;
            }
        };

        // 5. Merge into local.
        let inserted = {
            let brain = s.brain.lock().await;
            brain.cluster_merge_pool(pid, neurons)
        };
        out.neurons_inserted += inserted;
        out.pools_synced += 1;
    }

    Ok(Json(out))
}

// -----------------------------------------------------------------
// Handlers — multimodal
// -----------------------------------------------------------------

/// Decode base64 trying URL-safe (with/without padding) and standard
/// (with/without padding) — training scripts mix flavors.
fn decode_base64_flexible(s: &str) -> Result<Vec<u8>, String> {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
    if let Ok(b) = URL_SAFE_NO_PAD.decode(s) { return Ok(b); }
    if let Ok(b) = URL_SAFE.decode(s)        { return Ok(b); }
    if let Ok(b) = STANDARD_NO_PAD.decode(s) { return Ok(b); }
    if let Ok(b) = STANDARD.decode(s)        { return Ok(b); }
    Err("none of the base64 flavors decoded this payload".into())
}

/// Pull non-atom (concept) labels currently firing in a pool, top-N
/// by activation, for inclusion in `predictions` responses.
fn top_concept_labels(brain: &Brain, pool: PoolId, top_n: usize) -> Vec<String> {
    let arc = match brain.fabric().pool(pool) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let p = arc.read();
    let mut pairs: Vec<(String, f32)> = Vec::new();
    for nid in p.currently_firing() {
        if let Some(n) = p.get(nid) {
            let act = p.activation(nid);
            pairs.push((n.label.clone(), act));
        }
    }
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().take(top_n).map(|(l, _)| l).collect()
}

async fn sensor_observe(
    State(s): State<AppState>,
    Json(req): Json<SensorObserveRequest>,
) -> Result<Json<SensorObserveResponse>, (axum::http::StatusCode, String)> {
    let (pool_id, frame) = match req.kind.as_str() {
        "text" => {
            let txt = req.text.unwrap_or_default();
            (POOL_TEXT, txt.into_bytes())
        }
        "image" => {
            let b64 = req.bytes_b64.unwrap_or_default();
            let frame = decode_base64_flexible(&b64)
                .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
            (POOL_IMAGE, frame)
        }
        "audio" => {
            let b64 = req.bytes_b64.unwrap_or_default();
            let frame = decode_base64_flexible(&b64)
                .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
            (POOL_AUDIO, frame)
        }
        "turn" => {
            // Stage 11B: client-supplied opaque turn id (bytes).  May
            // arrive as plain `text` (recommended — turn ids are short
            // string tokens like "session-abc:turn-42") or as
            // `bytes_b64` for binary ids.  Brain owns no session
            // semantics: any byte sequence is fine.
            let frame = if let Some(t) = req.text.as_ref() {
                t.clone().into_bytes()
            } else {
                let b64 = req.bytes_b64.unwrap_or_default();
                decode_base64_flexible(&b64)
                    .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?
            };
            (POOL_TURN, frame)
        }
        other => {
            return Err((axum::http::StatusCode::BAD_REQUEST,
                format!("unknown kind {:?}", other)));
        }
    };
    let mut brain = s.brain.lock().await;
    let fired = brain.observe(pool_id, &frame);
    brain.advance_tick();

    // Predictions: walk all OTHER pools, take their top firing concepts.
    let mut predictions: HashMap<String, Vec<String>> = HashMap::new();
    for other in [POOL_TEXT, POOL_IMAGE, POOL_AUDIO] {
        if other == pool_id { continue; }
        let labels = top_concept_labels(&brain, other, 16);
        if !labels.is_empty() {
            predictions.insert(pool_name(other).to_string(), labels);
        }
    }

    Ok(Json(SensorObserveResponse {
        fired_neurons: fired.len(),
        predictions,
    }))
}

async fn sensor_observe_triple(
    State(s): State<AppState>,
    Json(req): Json<ObserveTripleRequest>,
) -> Result<Json<ObserveTripleResponse>, (axum::http::StatusCode, String)> {
    let img = decode_base64_flexible(&req.image_b64)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("image: {}", e)))?;
    let aud = decode_base64_flexible(&req.audio_b64)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("audio: {}", e)))?;
    let mut brain = s.brain.lock().await;
    let txt_fired = brain.observe(POOL_TEXT, req.text.as_bytes());
    let img_fired = brain.observe(POOL_IMAGE, &img);
    let aud_fired = brain.observe(POOL_AUDIO, &aud);
    brain.advance_tick();
    Ok(Json(ObserveTripleResponse {
        txt_labels: txt_fired.len(),
        img_labels: img_fired.len(),
        aud_labels: aud_fired.len(),
    }))
}

/// Strip Wizard-chat context-wrapper boilerplate so the brain only
/// observes the actual question.  The Django frontend prepends a
/// rolling context blob + "[Now answer concisely]\n<question>" cue;
/// without unwrapping we'd train against the boilerplate atoms more
/// than the real question.
fn unwrap_wizard_prompt(text: &str) -> &str {
    if let Some(idx) = text.rfind("[Now answer concisely]") {
        let after = &text[idx..];
        if let Some(nl) = after.find('\n') {
            return after[nl + 1..].trim();
        }
    }
    text.trim()
}

async fn chat(
    State(s): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let max_steps = req.max_steps.unwrap_or(8);
    let min_confidence = 0.01_f32;
    let mut brain = s.brain.lock().await;

    let prompt = unwrap_wizard_prompt(&req.text);

    // Observe the prompt into the text pool.
    brain.observe(POOL_TEXT, prompt.as_bytes());

    // PRIMARY: trained-binding decode.  Per ARCHITECTURE.md §4.D.1,
    // when the substrate has a binding that matches the firing
    // query state, its target-pool members ARE the trained answer.
    // Decoding those members in firing order gives the literal
    // trained response — 30/32 EXACT on toddler categorical
    // (animal/food/vehicle/color/toy/nature/body) without decoder
    // residual.
    let trained_decode: Option<String> = brain
        .decode_best_trained_binding(POOL_TEXT, POOL_ACTION)
        .map(|b| String::from_utf8_lossy(&b).into_owned());

    // SECONDARY: autonomous critical-thinking integrate.  Used as
    // fallback when no trained binding matches — and STILL the
    // source of the OOV honesty gate (outside_grounding=true).
    let xpool = brain.integrate_autonomous(
        POOL_TEXT, POOL_ACTION,
        /*fabric_threshold*/ 0.0,
        /*chain_max_depth*/   4,
        /*chain_max_visit*/   200);
    let xpool_reply: Option<String> = if xpool.grounding.outside_grounding {
        None
    } else {
        xpool.answer.as_ref().map(|b| String::from_utf8_lossy(b).into_owned())
    };

    // Selection: trained-binding decode is the AUTHORITATIVE answer.
    // When it returns Some, those bytes are the substrate's literal
    // trained response (binding target-pool members in firing order).
    // When it returns None, the substrate has no trained match that
    // passes the MIN_ATOM_SCORE floor — be OOV-honest and return
    // empty instead of falling through to xpool/generate, which can
    // hallucinate via partial atom bleed.
    //
    // Rationale (ARCHITECTURE.md §4.D.1): "trained input -> 100%
    // recall; untrained input -> integrate via knowledge with
    // confidence."  Until we have a confidence-gated integration
    // path, silence beats a wrong answer for the untrained case.
    let reply = if let Some(td) = trained_decode.as_ref().filter(|s| !s.is_empty()) {
        td.clone()
    } else {
        String::new()
    };

    // outside_grounding reflects the AUTHORITATIVE answer state: if
    // trained_decode found no binding above MIN_ATOM_SCORE, the
    // substrate has no trained recall for this query — that IS the
    // out-of-grounding signal, regardless of what xpool's softer
    // OOV heuristic concluded.  Keep xpool's flag as a secondary
    // signal only when reply is non-empty.
    let outside_grounding =
        reply.is_empty() || xpool.grounding.outside_grounding;
    let g = ChatGrounding {
        fabric_confidence:     xpool.grounding.fabric_confidence,
        integrated_confidence: xpool.grounding.integrated_confidence,
        outside_grounding,
        speculation_flag:      xpool.grounding.speculation_flag,
    };

    // Cross-modal predictions: re-observe prompt and read the firing
    // state in image and audio pools (cross-pool propagation does the
    // work automatically; we just snapshot what's active).
    brain.observe(POOL_TEXT, prompt.as_bytes());
    let mut predictions: HashMap<String, Vec<String>> = HashMap::new();
    // Stage 12 diagnostic: include POOL_ACTION so we can see which
    // action-pool concepts the integrate selection was choosing
    // between.  Pure observability — does NOT change /chat behavior.
    for other in [POOL_IMAGE, POOL_AUDIO, POOL_ACTION] {
        // Run propagation manually to populate cross-pool activation.
        let propagated = brain.fabric().propagate(POOL_TEXT);
        let empty: std::collections::HashMap<w1z4rd_brain::NeuronId, f32> = std::collections::HashMap::new();
        let act_map = propagated.get(&other);
        let it: Box<dyn Iterator<Item = (&w1z4rd_brain::NeuronId, &f32)>> = match act_map {
            Some(m) => Box::new(m.iter()),
            None    => Box::new(empty.iter()),
        };
        let act_pairs: Vec<(w1z4rd_brain::NeuronId, f32)> =
            it.map(|(k, v)| (*k, *v)).collect();
        if let Some(p) = brain.fabric().pool(other) {
            let p = p.read();
            let mut pairs: Vec<(String, f32)> = Vec::new();
            for (nid, a) in act_pairs.iter().copied() {
                if a < 0.05 { continue; }
                if let Some(n) = p.get(nid) {
                    pairs.push((n.label.clone(), a));
                }
            }
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let labels: Vec<String> = pairs.into_iter().take(16).map(|(l, _)| l).collect();
            if !labels.is_empty() {
                predictions.insert(pool_name(other).to_string(), labels);
            }
        }
    }

    // Decoder telemetry — which path produced the reply:
    //  "multi_pool" : fabric retrieval was confident (Stage 7 path)
    //  "eem"        : EEM chain exploration produced the answer
    //                 (Stage 8 ivy-growth path)
    //  "char_chain" : both upstream paths empty; same-pool generate
    //                 fragments are all we have (uncertain)
    let decoder = if xpool_reply.as_deref().map_or(false, |a| !a.is_empty()) {
        if xpool.grounding.eem_confidence.is_some()
            && xpool.grounding.fabric_confidence < 0.3
        {
            "eem"
        } else {
            "multi_pool"
        }
    } else {
        "char_chain"
    }.to_string();

    // Stage 12 diagnostic: surface `composition_used` neuron labels as
    // `activated_concepts` so /chat callers can see WHICH binding-pool
    // neurons fed the answer.  The labels follow the binding pool's
    // composite-label format: "p<pool>n<neuron>|..." for binding-pool
    // members.  Empty unless integrate_autonomous returned an answer.
    let activated: Vec<String> = xpool.grounding.composition_used.iter()
        .filter_map(|nref| {
            brain.fabric().pool(nref.pool).and_then(|p| {
                p.read().get(nref.neuron).map(|n| n.label.clone())
            })
        })
        .collect();

    Json(ChatResponse {
        reply: reply.clone(),
        answer: reply,
        decoder,
        predictions,
        grounding: g,
        activated_concepts: activated,
        word_activations: Vec::new(),
    })
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

    let mut brain = load_or_build_brain(&checkpoint_path)?;

    // Per ARCHITECTURE §17.9: when W1Z4RDV1510N_DATA_DIR is set (i.e. we have
    // a data dir, which we always do at this point), instantiate the
    // MmapWalStore at <data_dir>/brain.wal and attach it to the brain.
    // Every neurogenesis / concept-emergence / tick-advance event from this
    // point on appends to that WAL.  Crash recovery in a future commit will
    // replay it.  Opening the WAL is best-effort — if it fails (disk full,
    // permissions, format-version mismatch), the brain falls back to
    // NoopStore + bincode-checkpoint only, with a logged warning.
    match w1z4rd_brain::MmapWalStore::open(&data) {
        Ok(wal) => {
            let store: Arc<dyn w1z4rd_brain::Store> = Arc::new(wal);
            brain.set_store(store);
            info!("WAL attached at {}/brain.wal", data.display());
        }
        Err(e) => {
            warn!("WAL open failed at {}: {} — running in NoopStore mode",
                data.display(), e);
        }
    }

    // Stage 17.4 full: attach cold tiers to every pool so /eviction
    // has somewhere to write evicted neurons.  Best-effort; pools that
    // fail to attach run in non-evictable mode.
    let attached = brain.attach_cold_tiers(&data);
    info!("cold tiers attached on {} pool(s)", attached);

    // Stage 17.9 recovery: replay any WAL events past the last
    // SnapshotMarker so the brain reflects everything observed between
    // the last checkpoint and the previous shutdown / crash.  Best-
    // effort: if the WAL read fails, we log and continue with brain.bin
    // state only.
    match w1z4rd_brain::store::load_events_after_marker(&data) {
        Ok(events) if !events.is_empty() => {
            info!("WAL recovery: applying {} event(s) past last snapshot...",
                events.len());
            let stats = brain.apply_wal_events(&events);
            info!(
                "WAL recovery: total={} ticks_advanced_to={} since_last_snapshot={}",
                stats.events_total,
                stats.last_tick,
                stats.events_since_snapshot,
            );
        }
        Ok(_) => {
            info!("WAL recovery: no events past the last snapshot");
        }
        Err(e) => {
            warn!("WAL recovery failed: {} — continuing with brain.bin state only", e);
        }
    }

    let bs = brain.stats();
    info!("brain ready  tick={}  pools={}  neurons={}  terminals={}",
        bs.tick, bs.pool_count, bs.total_neurons, bs.total_terminals);

    // Stage 18.12 step 5: bootstrap solo cluster membership.  The
    // local_addr advertisement is finalised below once port + bind_ip
    // are known; here we use an empty placeholder that the post-bind
    // code patches.
    let cluster_state = ClusterMembership::solo(String::new());

    let state = AppState {
        brain:           Arc::new(Mutex::new(brain)),
        checkpoint_path: checkpoint_path.clone(),
        sleep_status:    Arc::new(std::sync::Mutex::new(None)),
        cluster:         Arc::new(std::sync::Mutex::new(cluster_state)),
    };

    let app = Router::new()
        .route("/health",                get(health))
        .route("/stats",                 get(stats))
        .route("/observe",               post(observe))
        .route("/tick",                  post(tick))
        .route("/integrate",             post(integrate))
        .route("/integrate_concept_first", post(integrate_concept_first))
        .route("/integrate_resonant",    post(integrate_resonant))
        .route("/pool/concepts",         post(pool_concepts))
        .route("/checkpoint",            post(checkpoint))
        .route("/flush",                 post(flush))
        .route("/sleep",                 post(sleep_cycle))
        .route("/sleep/status",          get(sleep_status))
        .route("/storage_state",         get(storage_state))
        .route("/eviction",              post(eviction))
        .route("/cluster/pool_roots",    get(cluster_pool_roots))
        .route("/cluster/pool_neurons/:pool_id", get(cluster_pool_neurons))
        .route("/cluster/merge_pool",    post(cluster_merge_pool))
        .route("/cluster/pull_from",     post(cluster_pull_from))
        .route("/cluster/members",       get(cluster_members))
        .route("/cluster/join",          post(cluster_join))
        .route("/cluster/leave",         post(cluster_leave))
        .route("/cluster/heartbeat",     post(cluster_heartbeat))
        .route("/cluster/aggregate_pool_neurons/:pool_id", get(cluster_aggregate_pool_neurons))
        .route("/shard/neuron/:pool_id/:neuron_id", get(shard_get_neuron))
        .route("/shard/put_neuron",      post(shard_put_neuron))
        .route("/sensor/observe",        post(sensor_observe))
        .route("/sensor/observe_triple", post(sensor_observe_triple))
        .route("/chat",                  post(chat))
        .with_state(state.clone());

    let port: u16 = std::env::var("W1Z4RD_BRAIN_PORT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8095);
    // Stage 17.6 cluster mode: bind to the IP given by W1Z4RD_BRAIN_BIND
    // (defaults to 127.0.0.1 for backward compat).  Set to "0.0.0.0" to
    // accept LAN connections from peer nodes; production cluster
    // deployments should set this to a specific interface or use a
    // firewall to restrict access.
    let bind_ip_str = std::env::var("W1Z4RD_BRAIN_BIND")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let bind_ip: std::net::IpAddr = bind_ip_str.parse()
        .with_context(|| format!("invalid W1Z4RD_BRAIN_BIND: {}", bind_ip_str))?;
    let addr = SocketAddr::new(bind_ip, port);
    info!("brain server listening on http://{}", addr);
    if bind_ip.is_unspecified() {
        info!("(0.0.0.0 bind — accepts connections from peer nodes; firewall accordingly)");
    }

    // Stage 18.12 step 5: finalise the cluster membership's
    // advertised address now that port + bind_ip are known.  Use
    // W1Z4RD_BRAIN_ADVERTISE_URL to override (e.g. when the node is
    // behind NAT or wants to be reached via a hostname).
    {
        let advertise = std::env::var("W1Z4RD_BRAIN_ADVERTISE_URL")
            .unwrap_or_else(|_| format!("http://{}:{}",
                if bind_ip.is_unspecified() { "127.0.0.1".to_string() }
                else { bind_ip.to_string() },
                port,
            ));
        let mut m = state.cluster.lock().unwrap();
        m.local_addr = advertise.clone();
        let local = m.local_node;
        if let Some(self_member) = m.members.iter_mut().find(|mi| mi.node_id == local) {
            self_member.addr = advertise;
        }
    }

    // Stage 18.12 step 5: optional join-on-startup.  When the env var
    // W1Z4RD_CLUSTER_SEED is set (e.g. "http://192.168.1.84:8095"), the
    // node POSTs /cluster/join to that seed and adopts the returned
    // membership.  Mistakes here are non-fatal: log + continue as solo.
    if let Ok(seed_url) = std::env::var("W1Z4RD_CLUSTER_SEED") {
        let seed = seed_url.trim_end_matches('/').to_string();
        let my_addr = state.cluster.lock().unwrap().local_addr.clone();
        info!("Stage 18.12: joining cluster via seed {}", seed);
        let client = reqwest::Client::new();
        let req_body = serde_json::json!({
            "addr": my_addr,
            "capacity_neurons": null,
        });
        match client.post(format!("{}/cluster/join", seed))
            .json(&req_body).send().await
        {
            Ok(resp) if resp.status().is_success() => {
                #[derive(Deserialize)]
                struct R {
                    assigned_node_id: w1z4rd_brain::store::NodeId,
                    membership: ClusterMembership,
                }
                match resp.json::<R>().await {
                    Ok(r) => {
                        let local_node;
                        let member_count;
                        let membership_snapshot;
                        {
                            let mut m = state.cluster.lock().unwrap();
                            // Preserve our own local_addr — the seed
                            // returns ITS OWN membership snapshot which
                            // has the seed's local_addr; the joiner keeps
                            // its own.
                            let our_addr = m.local_addr.clone();
                            *m = r.membership;
                            m.local_node = r.assigned_node_id;
                            m.local_addr = our_addr;
                            local_node = m.local_node;
                            member_count = m.members.len();
                            membership_snapshot = m.clone();
                        }
                        info!("Stage 18.12: joined cluster as node {} \
                              with {} members",
                            r.assigned_node_id.0, member_count);
                        // Stage 18.12 step 6: wire the cluster topology
                        // into every Pool's TieredStore so the placement
                        // policy actually routes operations to ring peers.
                        wire_cluster_topology(&state, &membership_snapshot, local_node).await;
                    }
                    Err(e) => warn!("seed-join parse: {}", e),
                }
            }
            Ok(resp) => warn!("seed-join status {}", resp.status()),
            Err(e)   => warn!("seed-join network: {}", e),
        }
    }

    // Stage 18.12 step 8: spawn the heartbeat + stale-detection
    // background task.  Inert in solo mode (single-member ring) — the
    // loop iterates immediately and sweeps nothing.  Comes alive once a
    // peer joins.
    {
        let hb_state = state.clone();
        tokio::spawn(async move { heartbeat_loop(hb_state).await; });
    }

    let listener = tokio::net::TcpListener::bind(addr).await
        .with_context(|| format!("bind {}", addr))?;

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
