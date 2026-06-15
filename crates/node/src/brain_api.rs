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
use axum::{Json, Router, extract::State, routing::{get, post}};
use serde_json::json;
use tokio::sync::Mutex;
use w1z4rd_brain::{Brain, BrainConfig, PoolConfig};
use w1z4rd_brain::neuron::PoolId;
use w1z4rd_brain::pool::{AtomEncoding, BytePassthroughEncoding};

// ---------------------------------------------------------------------
// Standard pool ids — must match brain_server.rs and any client script.
// ---------------------------------------------------------------------

pub const POOL_BINDING: PoolId = 0;
pub const POOL_TEXT:    PoolId = 1;
pub const POOL_IMAGE:   PoolId = 2;
pub const POOL_AUDIO:   PoolId = 3;
pub const POOL_ACTION:  PoolId = 4;
pub const POOL_TURN:    PoolId = 5;

// ---------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------

/// Phase E continuous-thought controller state.  Atomics + std mutexes
/// for shared access from the background loop without holding the
/// brain lock.
#[derive(Debug)]
pub struct ThinkingState {
    pub enabled:     AtomicBool,
    pub query_pool:  AtomicU32,
    pub target_pool: AtomicU32,
    pub hops_taken:  AtomicU64,
    pub last_seed:   std::sync::Mutex<Option<Vec<u8>>>,
    pub last_answer: std::sync::Mutex<Option<Vec<u8>>>,
}

impl Default for ThinkingState {
    fn default() -> Self {
        Self {
            enabled:     AtomicBool::new(false),
            query_pool:  AtomicU32::new(POOL_TEXT),
            target_pool: AtomicU32::new(POOL_ACTION),
            hops_taken:  AtomicU64::new(0),
            last_seed:   std::sync::Mutex::new(None),
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
    pub observe_calls:         AtomicU64,
    pub observe_lock_wait_ns:  AtomicU64,
    pub observe_handler_ns:    AtomicU64,
    pub tick_calls:            AtomicU64,
    pub tick_lock_wait_ns:     AtomicU64,
    pub tick_handler_ns:       AtomicU64,
}

/// Router state passed to every brain handler.  Clone-friendly because
/// every field is Arc-backed.
#[derive(Clone)]
pub struct BrainApiState {
    pub brain:        Arc<Mutex<Brain>>,
    pub thinking:     Arc<ThinkingState>,
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
    cfg.moment_history_window       = 256;
    let mut brain = Brain::new(cfg);

    let mut text = PoolConfig::defaults("text", POOL_TEXT);
    text.recent_atoms_window         = 65536;
    text.concept_emergence_threshold = 3;
    text.max_concept_member_count    = 32;
    text.decay_rate                  = 0.00002;
    text.prune_floor                 = 0.001;
    brain.create_pool(text, leaked_encoding("t"));

    let mut image = PoolConfig::defaults("image", POOL_IMAGE);
    image.recent_atoms_window         = 4096;
    image.concept_emergence_threshold = 3;
    image.max_concept_member_count    = 32;
    image.decay_rate                  = 0.00002;
    image.prune_floor                 = 0.001;
    brain.create_pool(image, leaked_encoding("i"));

    let mut audio = PoolConfig::defaults("audio", POOL_AUDIO);
    audio.recent_atoms_window         = 4096;
    audio.concept_emergence_threshold = 3;
    audio.max_concept_member_count    = 32;
    audio.decay_rate                  = 0.00002;
    audio.prune_floor                 = 0.001;
    brain.create_pool(audio, leaked_encoding("a"));

    let mut action = PoolConfig::defaults("action", POOL_ACTION);
    action.recent_atoms_window         = 65536;
    action.concept_emergence_threshold = 3;
    action.max_concept_member_count    = 32;
    action.decay_rate                  = 0.00002;
    action.prune_floor                 = 0.001;
    brain.create_pool(action, leaked_encoding("act"));
    brain.designate_action_pool(POOL_ACTION);

    let mut turn = PoolConfig::defaults("turn", POOL_TURN);
    turn.recent_atoms_window         = 32;
    turn.concept_emergence_threshold = u32::MAX;
    turn.max_concept_member_count    = 4;
    turn.decay_rate                  = 0.001;
    turn.prune_floor                 = 0.01;
    brain.create_pool(turn, leaked_encoding("turn"));

    Ok(brain)
}

fn leaked_encoding(prefix: &str) -> Box<dyn AtomEncoding> {
    let leaked: &'static str = Box::leak(prefix.to_string().into_boxed_str());
    Box::new(BytePassthroughEncoding { prefix: leaked })
}

/// Load a brain from `<data_dir>/brain.bin` if it exists, else build a
/// fresh one with the default topology.  WAL replay is the caller's
/// responsibility.
pub fn load_or_build_brain(data_dir: &Path) -> Result<Brain> {
    let checkpoint = data_dir.join("brain.bin");
    if checkpoint.exists() {
        let mut prefixes = HashMap::new();
        prefixes.insert(POOL_BINDING, "bind".to_string());
        prefixes.insert(POOL_TEXT,    "t".to_string());
        prefixes.insert(POOL_IMAGE,   "i".to_string());
        prefixes.insert(POOL_AUDIO,   "a".to_string());
        prefixes.insert(POOL_ACTION,  "act".to_string());
        prefixes.insert(POOL_TURN,    "turn".to_string());
        let encs: HashMap<PoolId, Box<dyn AtomEncoding>> = prefixes.iter()
            .map(|(pid, p)| (*pid, leaked_encoding(p)))
            .collect();
        match Brain::restore(&checkpoint, encs) {
            Ok((brain, _missing)) => return Ok(brain),
            Err(e) => {
                tracing::warn!("brain restore failed at {}: {} — starting fresh",
                    checkpoint.display(), e);
            }
        }
    }
    build_default_brain()
}

/// Create the shared state used by the brain router.  Caller wraps
/// `Brain` so the state can be sent across tasks (router clone, etc.).
pub fn build_brain_api_state(brain: Brain) -> BrainApiState {
    BrainApiState {
        brain:        Arc::new(Mutex::new(brain)),
        thinking:     Arc::new(ThinkingState::default()),
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

async fn h_health() -> &'static str { "ok\n" }

async fn h_stats(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let b = s.brain.lock().await;
    let st = b.stats();
    Json(json!({
        "tick":            st.tick,
        "pool_count":      st.pool_count,
        "total_neurons":   st.total_neurons,
        "total_concepts":  st.total_concepts,
        "total_binding":   st.total_binding,
        "total_terminals": st.total_terminals,
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
    s.http_profile.observe_lock_wait_ns.fetch_add(lock_ns, Ordering::Relaxed);
    s.http_profile.observe_handler_ns.fetch_add(handler_ns, Ordering::Relaxed);
    Json(json!({ "fired_count": fired.len() }))
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
    s.http_profile.tick_lock_wait_ns.fetch_add(lock_ns, Ordering::Relaxed);
    s.http_profile.tick_handler_ns.fetch_add(handler_ns, Ordering::Relaxed);
    Json(tick)
}

async fn h_http_profile(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let obs_calls = s.http_profile.observe_calls.load(Ordering::Relaxed);
    let obs_lock  = s.http_profile.observe_lock_wait_ns.load(Ordering::Relaxed);
    let obs_hand  = s.http_profile.observe_handler_ns.load(Ordering::Relaxed);
    let tick_calls = s.http_profile.tick_calls.load(Ordering::Relaxed);
    let tick_lock  = s.http_profile.tick_lock_wait_ns.load(Ordering::Relaxed);
    let tick_hand  = s.http_profile.tick_handler_ns.load(Ordering::Relaxed);
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
    let domain_id = req.get("domain_id")
        .and_then(|v| v.as_u64()).map(|n| n as u32).unwrap_or(0);
    let brain = s.brain.lock().await;
    brain.set_domain_for_new(domain_id);
    Json(json!({ "domain_for_new": domain_id }))
}

async fn h_domain_stats(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let hist = brain.domain_histogram();
    let entries: Vec<_> = hist.into_iter()
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
    let n = req.get("sample_count").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
    let mut brain = s.brain.lock().await;
    Json(json!(brain.self_test(n)))
}

async fn h_integrate(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let qp = req.get("query_pool").and_then(|v| v.as_u64()).unwrap_or(POOL_TEXT as u64) as PoolId;
    let tp = req.get("target_pool").and_then(|v| v.as_u64()).unwrap_or(POOL_ACTION as u64) as PoolId;
    let brain = s.brain.lock().await;
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
    brain.fabric_mut().observe(POOL_TEXT, prompt.as_bytes());

    // Authoritative trained-binding decode — Phase B v2.
    let trained_decode: Option<String> = brain
        .decode_best_trained_binding(POOL_TEXT, POOL_ACTION)
        .map(|b| String::from_utf8_lossy(&b).into_owned());

    // Autonomous fallback — EEM chain + annealer + multi-fact.
    let xpool = brain.integrate_autonomous(
        POOL_TEXT, POOL_ACTION,
        /*fabric_threshold*/ 0.0,
        /*chain_max_depth*/  4,
        /*chain_max_visit*/  200);
    let xpool_reply: Option<String> = if xpool.grounding.outside_grounding {
        None
    } else {
        xpool.answer.as_ref().map(|b| String::from_utf8_lossy(b).into_owned())
    };

    let reply = if let Some(td) = trained_decode.as_ref().filter(|s| !s.is_empty()) {
        td.clone()
    } else {
        // No trained binding above MIN_ATOM_SCORE — be OOV-honest
        // and return empty rather than fall through to the noisy
        // xpool path.  Same policy brain_server applies.
        String::new()
    };

    let outside_grounding = reply.is_empty() || xpool.grounding.outside_grounding;

    let decoder = if xpool_reply.as_deref().map_or(false, |a| !a.is_empty()) {
        if xpool.grounding.eem_confidence.is_some()
            && xpool.grounding.fabric_confidence < 0.3 { "eem" }
        else { "multi_pool" }
    } else { "char_chain" }.to_string();

    let activated: Vec<String> = xpool.grounding.composition_used.iter()
        .filter_map(|nref| {
            brain.fabric().pool(nref.pool).and_then(|p| {
                p.read().get(nref.neuron).map(|n| n.label.clone())
            })
        })
        .collect();

    Json(json!({
        "reply":              reply,
        "answer":             reply,
        "decoder":            decoder,
        "predictions":        serde_json::Map::new(),
        "grounding": {
            "fabric_confidence":     xpool.grounding.fabric_confidence,
            "integrated_confidence": xpool.grounding.integrated_confidence,
            "outside_grounding":     outside_grounding,
            "speculation_flag":      xpool.grounding.speculation_flag,
        },
        "activated_concepts": activated,
        "word_activations":   Vec::<serde_json::Value>::new(),
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
    let qp = req.get("query_pool").and_then(|v| v.as_u64()).unwrap_or(POOL_TEXT as u64) as PoolId;
    let tp = req.get("target_pool").and_then(|v| v.as_u64()).unwrap_or(POOL_ACTION as u64) as PoolId;
    let hops = req.get("max_hops").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let seed_b64 = req.get("seed").and_then(|v| v.as_str()).unwrap_or("");
    let seed = match b64_url_decode(seed_b64) {
        Ok(b) => b,
        Err(e) => return Json(json!({"error": format!("bad seed: {}", e)})),
    };
    let mut brain = s.brain.lock().await;
    let trail = brain.integrate_chain(qp, tp, &seed, hops);
    let steps: Vec<_> = trail.into_iter().map(|(q, a)| {
        json!({ "query": b64_url_no_pad(&q),
                "answer": a.map(|b| b64_url_no_pad(&b)) })
    }).collect();
    Json(json!({ "steps": steps }))
}

async fn h_integrate_islands(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let sample = req.get("sample_size").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
    let thr    = req.get("similarity_threshold").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
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
    let concepts: Vec<_> = p.iter_neurons()
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

async fn h_retune(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let n = req.get("sample_count").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
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
    let v = req.get("decay_rate").and_then(|v| v.as_f64()).unwrap_or(2e-5) as f32;
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
    for _ in 0..n { brain.advance_tick(); }
    Json(json!({ "ticks_advanced": n, "current_tick": brain.fabric().current_tick() }))
}

async fn h_sleep(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let min_use_count = req.get("min_use_count").and_then(|v| v.as_u64()).unwrap_or(2);
    let stale_ticks   = req.get("stale_ticks").and_then(|v| v.as_u64()).unwrap_or(1000);
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
    let pct = |ns: u64| if snap.total_ns == 0 { 0.0 }
        else { (ns as f64) * 100.0 / (snap.total_ns as f64) };
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
    let mean_per_pass_ns = if snap.passes == 0 { 0 } else { snap.total_ns / snap.passes };
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

async fn h_tick_profile(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let brain = s.brain.lock().await;
    let snap = brain.fabric().profile.snapshot();
    let ticks = snap.ticks.max(1) as f64;
    let to_us = |ns: u64| (ns as f64 / 1_000.0) as u64;
    let to_ms = |ns: u64| (ns as f64 / 1_000_000.0) as u64;
    let mean_us = |ns: u64| ((ns as f64) / ticks / 1_000.0) as u64;
    let pct = |ns: u64| if snap.total_ns == 0 { 0.0 }
        else { (ns as f64) * 100.0 / (snap.total_ns as f64) };
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
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
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
    let seed   = s.thinking.last_seed.lock().unwrap().clone();
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
        let last_seed_snap   = state.thinking.last_seed.lock().unwrap().clone();

        let seed: Option<Vec<u8>> = match last_answer_snap {
            Some(ans) if !ans.is_empty() && Some(&ans) != last_seed_snap.as_ref() => Some(ans),
            _ => {
                let brain = state.brain.lock().await;
                let len = brain.qa_db().len();
                if len == 0 { None } else {
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
        .route("/health",                 get(h_health))
        .route("/stats",                  get(h_stats))
        .route("/observe",                post(h_observe))
        .route("/tick",                   post(h_tick))
        .route("/integrate",              post(h_integrate))
        .route("/chat",                   post(h_brain_chat))
        .route("/pool/concepts",          post(h_pool_concepts))
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
    Router::new()
        .route("/set_domain",             post(h_set_domain))
        .route("/domain_stats",           get(h_domain_stats))
        .route("/qa_db_stats",            get(h_qa_db_stats))
        .route("/consolidation_stats",    get(h_consolidation_stats))
        .route("/self_test",              post(h_self_test))
        .route("/integrate_chain",        post(h_integrate_chain))
        .route("/integrate_islands",      post(h_integrate_islands))
        .route("/retune",                 post(h_retune))
        .route("/tuning_state",           get(h_tuning_state))
        .route("/force_decay",            post(h_force_decay))
        .route("/idle_ticks",             post(h_idle_ticks))
        .route("/sleep_pressure",         get(h_sleep_pressure))
        .route("/tick_profile",           get(h_tick_profile))
        .route("/observe_profile",        get(h_observe_profile))
        .route("/http_profile",           get(h_http_profile))
        .route("/tier_orchestrator",      get(h_tier_orchestrator))
        .route("/sleep",                  post(h_sleep))
        .route("/checkpoint",             post(h_checkpoint))
        .route("/thinking/start",         post(h_thinking_start))
        .route("/thinking/stop",          post(h_thinking_stop))
        .route("/thinking/status",        get(h_thinking_status))
        .with_state(state)
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
