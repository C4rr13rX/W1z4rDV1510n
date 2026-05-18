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
    brain.create_pool(text,
        Box::new(BytePassthroughEncoding { prefix: "t" }) as Box<dyn AtomEncoding>);

    let mut image = PoolConfig::defaults("image", POOL_IMAGE);
    image.recent_atoms_window         = 4096;
    image.concept_emergence_threshold = 3;
    image.max_concept_member_count    = 32;
    image.decay_rate                  = 0.00002;
    image.prune_floor                 = 0.001;
    brain.create_pool(image,
        Box::new(BytePassthroughEncoding { prefix: "i" }) as Box<dyn AtomEncoding>);

    let mut audio = PoolConfig::defaults("audio", POOL_AUDIO);
    audio.recent_atoms_window         = 4096;
    audio.concept_emergence_threshold = 3;
    audio.max_concept_member_count    = 32;
    audio.decay_rate                  = 0.00002;
    audio.prune_floor                 = 0.001;
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
    brain.create_pool(turn,
        Box::new(BytePassthroughEncoding { prefix: "turn" }) as Box<dyn AtomEncoding>);

    Ok(brain)
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
async fn integrate_concept_first(
    State(s): State<AppState>,
    Json(req): Json<IntegrateRequest>,
) -> Json<IntegrateConceptFirstResponse> {
    let brain = s.brain.lock().await;
    let answer_bytes = brain.integrate_concept_first(req.query_pool, req.target_pool);
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let (b64, utf8) = match &answer_bytes {
        Some(b) => (Some(engine.encode(b)),
                    std::str::from_utf8(b).ok().map(|s| s.to_string())),
        None    => (None, None),
    };
    Json(IntegrateConceptFirstResponse {
        answer_b64:  b64,
        answer_utf8: utf8,
    })
}

#[derive(Serialize, Debug)]
struct IntegrateConceptFirstResponse {
    answer_b64:  Option<String>,
    answer_utf8: Option<String>,
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

    // PRIMARY: autonomous critical-thinking integrate.  Tries
    // fabric retrieval first; if low confidence, falls through to
    // EEM grounded-fact chain exploration (ivy-growth across the
    // substrate's accumulated world knowledge).  Also enforces the
    // out-of-vocabulary gate: if no trained binding has its query
    // members substantially contained in the firing prompt atoms,
    // returns outside_grounding=true honestly rather than picking
    // a noisy strongest-match (the "Hello → color" hallucination
    // failure mode).
    let xpool = brain.integrate_autonomous(
        POOL_TEXT, POOL_ACTION,
        /*fabric_threshold*/ 0.0,    // accept any non-empty fabric answer
        /*chain_max_depth*/   4,
        /*chain_max_visit*/   200);
    // Honor outside_grounding: return EMPTY answer so the downstream
    // Wizard frontend renders the "no confident answer" UX instead
    // of pretending we know.
    let xpool_reply: Option<String> = if xpool.grounding.outside_grounding {
        None
    } else {
        xpool.answer.as_ref().map(|b| String::from_utf8_lossy(b).into_owned())
    };

    // SECONDARY: same-pool generate text→text ONLY when the prompt is
    // in-vocabulary but the cross-pool path returned nothing.  When
    // the prompt is out-of-vocabulary (outside_grounding), the
    // associative-fragment generate path is just hallucinated junk —
    // returning it would un-do the OOV honesty.  Keep silent instead.
    let reply = match xpool_reply {
        Some(ref a) if !a.is_empty() => a.clone(),
        _ if xpool.grounding.outside_grounding => String::new(),
        _ => {
            brain.observe(POOL_TEXT, prompt.as_bytes());
            let gen_bytes = brain.generate(POOL_TEXT, POOL_TEXT, max_steps, min_confidence);
            String::from_utf8_lossy(&gen_bytes).into_owned()
        }
    };

    let g = ChatGrounding {
        fabric_confidence:     xpool.grounding.fabric_confidence,
        integrated_confidence: xpool.grounding.integrated_confidence,
        outside_grounding:     xpool.grounding.outside_grounding,
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

    let brain = load_or_build_brain(&checkpoint_path)?;
    let bs = brain.stats();
    info!("brain ready  tick={}  pools={}  neurons={}  terminals={}",
        bs.tick, bs.pool_count, bs.total_neurons, bs.total_terminals);

    let state = AppState {
        brain:           Arc::new(Mutex::new(brain)),
        checkpoint_path: checkpoint_path.clone(),
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
        .route("/sensor/observe",        post(sensor_observe))
        .route("/sensor/observe_triple", post(sensor_observe_triple))
        .route("/chat",                  post(chat))
        .with_state(state.clone());

    let port: u16 = std::env::var("W1Z4RD_BRAIN_PORT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8095);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("brain server listening on http://{}", addr);

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
