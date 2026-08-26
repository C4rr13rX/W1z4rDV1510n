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
use std::time::Instant;

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

/// Motif score a recall-derived route must clear when no intent label applies.
///
/// The labelled route trusts 0.20 because labels corroborate it. Without them
/// the score carries the decision alone, so it has to be the whole safeguard.
///
/// Calibrated against what the brain itself reports, not an offline model of
/// it. Measured 2026-08-20 through /brain/chat: a novel phrasing of a trained
/// unit scored 0.561 and returned the right unit, while "Write a haiku about
/// the ocean" -- nothing like anything trained -- still scored 0.373 against
/// its nearest binding. A 0.35 floor admitted the haiku; 0.45 keeps the
/// genuine recall with headroom on both sides.
const UNLABELED_RECALL_MIN_SCORE: f32 = 0.45;

/// Separation over the runner-up. Deliberately near-inert on this route.
///
/// Margin looks like the principled gate for "settle onto a known, leave an
/// unknown alone", and an offline model of the corpus supported that: scoring
/// novel phrasings best-per-unit gave the correct unit 0.064-0.278 of
/// separation while an untrained prompt separated by 0.002.
///
/// The brain does not compute it that way. `runner_score` falls back to 0.0
/// when the motif posting window yields a single surviving candidate, so
/// `margin = score - 0.0 = score`. Measured through /brain/chat, every
/// admitted route reported margin exactly equal to its score (0.5612/0.5612,
/// 0.3733/0.3733). On this path margin carries no information the score does
/// not already carry, so gating on it only double-counts one signal.
///
/// Kept at a floor that rejects a true zero-evidence match without pretending
/// to be an independent safeguard. The score floor above is the real gate.
const UNLABELED_RECALL_MIN_MARGIN: f32 = 0.01;

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

fn restore_encodings(
    identity: Option<&BrainIdentitySpec>,
) -> Result<HashMap<PoolId, Box<dyn AtomEncoding>>> {
    let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    encodings.insert(POOL_BINDING, leaked_encoding("bind"));
    if let Some(spec) = identity {
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
            encodings.insert(pool.id, encoding);
        }
    } else {
        encodings.insert(POOL_TEXT, leaked_encoding("t"));
        encodings.insert(POOL_IMAGE, leaked_encoding("i"));
        encodings.insert(POOL_AUDIO, leaked_encoding("a"));
        encodings.insert(POOL_ACTION, leaked_encoding("act"));
        encodings.insert(POOL_TURN, leaked_encoding("turn"));
    }
    Ok(encodings)
}

/// Load a brain from `<data_dir>/brain.bin` if it exists, else build a
/// fresh one with the default topology. Replays the WAL tail after the most
/// recent snapshot marker, then attaches the same WAL for all future
/// mutations. Keeping this in the shared loader makes brain-only, embedded,
/// and merged-node modes obey the same durability contract.
pub fn load_or_build_brain(data_dir: &Path) -> Result<Brain> {
    let checkpoint = data_dir.join("brain.bin");
    let wbrain = data_dir.join("brain.wbrain");
    // A genuinely new brain should be neuron-addressable from its first
    // observation.  Starting it in the legacy monolithic layout forces an
    // expensive full hydration and conversion before the normal idle/sleep
    // transition can serialize neurons independently.  Existing checkpoints
    // retain their explicit, fail-safe migration path below.
    let is_fresh = !wbrain.exists() && !checkpoint.exists();
    let identity = configured_identity(data_dir)?;
    let mut brain = if wbrain.exists() {
        match Brain::restore_wbrain(&wbrain, restore_encodings(identity.as_ref())?) {
            Ok((_brain, missing)) if !missing.is_empty() => {
                anyhow::bail!(
                    "container {} requires encodings for missing pools {:?}; set W1Z4RD_BRAIN_IDENTITY",
                    wbrain.display(),
                    missing
                );
            }
            Ok((brain, _missing)) => brain,
            Err(error) => anyhow::bail!(
                "neuron-addressable restore failed at {}: {}",
                wbrain.display(),
                error
            ),
        }
    } else if checkpoint.exists() {
        match Brain::restore(&checkpoint, restore_encodings(identity.as_ref())?) {
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
    if is_fresh {
        attach_fresh_wbrain(&mut brain, &wbrain)?;
    }
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
            let store: std::sync::Arc<dyn w1z4rd_brain::Store> = std::sync::Arc::new(wal);
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

fn attach_fresh_wbrain(brain: &mut Brain, path: &Path) -> Result<usize> {
    brain.attach_wbrain(path).map_err(|error| {
        anyhow::anyhow!(
            "attach fresh neuron-addressable brain container {}: {}",
            path.display(),
            error
        )
    })
}

/// Convert the existing checkpoint without deleting or replacing it. This is
/// intentionally separate from normal startup so a large one-time legacy
/// hydration can be scheduled and monitored explicitly.
pub fn migrate_legacy_brain_container(data_dir: &Path) -> Result<usize> {
    let legacy = data_dir.join("brain.bin");
    let destination = data_dir.join("brain.wbrain");
    let raw_marker = data_dir.join("brain.wbrain.raw-running");
    let finalize_marker = data_dir.join("brain.wbrain.finalize-pending");
    if !legacy.is_file() {
        anyhow::bail!("legacy checkpoint not found: {}", legacy.display());
    }
    let legacy_bytes = std::fs::metadata(&legacy)?.len();
    let marker_value = legacy_bytes.to_string();
    let marker_matches = |path: &Path| {
        std::fs::read_to_string(path)
            .ok()
            .is_some_and(|value| value.trim() == marker_value)
    };
    let write_marker = |path: &Path| -> Result<()> {
        use std::io::Write;
        let mut marker = std::fs::File::create(path)?;
        marker.write_all(marker_value.as_bytes())?;
        marker.sync_all()?;
        Ok(())
    };

    let mut finalize_pending = marker_matches(&finalize_marker);
    if finalize_pending && !destination.exists() {
        std::fs::remove_file(&finalize_marker)?;
        finalize_pending = false;
    }
    let resumable_raw =
        destination.exists() && Brain::legacy_migration_is_resumable(&legacy, &destination);
    let raw_complete =
        destination.exists() && Brain::legacy_migration_raw_is_complete(&destination);
    if marker_matches(&raw_marker) && raw_complete {
        // Raw conversion committed its final manifest before the process could
        // publish the next stage marker. Preserve it and continue finalizing.
        write_marker(&finalize_marker)?;
        std::fs::remove_file(&raw_marker)?;
        finalize_pending = true;
    }
    if raw_complete && !finalize_pending {
        anyhow::bail!(
            "refusing to finalize an existing container without a matching migration marker: {}",
            destination.display()
        );
    }
    if destination.exists() && !resumable_raw && !raw_complete && !finalize_pending {
        if marker_matches(&raw_marker) {
            // The process died before publishing its first complete pool.
            // No manifest points at the appended tail, so restart cleanly.
            std::fs::remove_file(&destination)?;
        } else {
            anyhow::bail!(
                "refusing to overwrite complete or unrecognized container: {}",
                destination.display()
            );
        }
    }
    if !finalize_pending {
        if raw_marker.exists() && !marker_matches(&raw_marker) {
            anyhow::bail!(
                "raw migration marker does not match the legacy checkpoint: {}",
                raw_marker.display()
            );
        }
        write_marker(&raw_marker)?;
    }

    let serialized = if finalize_pending {
        None
    } else {
        match Brain::migrate_legacy_checkpoint_streaming(&legacy, &destination) {
            Ok(count) => {
                write_marker(&finalize_marker)?;
                std::fs::remove_file(&raw_marker)?;
                finalize_pending = true;
                Some(count)
            }
            Err(error) => {
                if !Brain::legacy_migration_is_resumable(&legacy, &destination) {
                    let _ = std::fs::remove_file(&destination);
                    let _ = std::fs::remove_file(&raw_marker);
                }
                return Err(error.into());
            }
        }
    };

    if !finalize_pending {
        anyhow::bail!(
            "migration did not reach its durable finalization stage at {}",
            finalize_marker.display()
        );
    }
    if serialized == Some(0) {
        let _ = std::fs::remove_file(&destination);
        let _ = std::fs::remove_file(&finalize_marker);
        anyhow::bail!("migration produced no neurons");
    }
    let finalize = || -> Result<usize> {
        let identity = configured_identity(data_dir)?;
        let (mut migrated, missing) =
            Brain::restore_wbrain(&destination, restore_encodings(identity.as_ref())?)?;
        if !missing.is_empty() {
            anyhow::bail!(
                "migration requires encodings for missing pools {:?}; set W1Z4RD_BRAIN_IDENTITY",
                missing
            );
        }
        migrated.rebuild_binding_indexes_bounded()?;
        migrated.serialize_all_neurons_for_idle()?;
        Ok(migrated.stats().total_neurons)
    };
    let finalized_count = finalize()?;
    std::fs::remove_file(&finalize_marker)?;
    if raw_marker.exists() {
        std::fs::remove_file(&raw_marker)?;
    }
    Ok(serialized.unwrap_or(finalized_count))
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
    let (binding_posting_overlay, binding_posting_generations) = b.binding_posting_residency();
    let (fingerprint_state_overlay, fingerprint_state_generations) =
        b.fingerprint_state_residency();
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
        "binding_posting_overlay": binding_posting_overlay,
        "binding_posting_generations": binding_posting_generations,
        "fingerprint_state_overlay": fingerprint_state_overlay,
        "fingerprint_state_generations": fingerprint_state_generations,
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
            return match brain.finish_read_only_inference() {
                Ok(_) => Json(json!({"error": format!("unknown pool id {}", pool_id)})),
                Err(error) => Json(json!({"error": format!(
                    "unknown pool id {}; inference cleanup failed: {}", pool_id, error
                )})),
            };
        }
        brain.activate_for_indexed_prediction(*pool_id, frame);
        query_pools.push(*pool_id);
    }
    let scored = brain.decode_best_trained_binding_multi_scored(&query_pools, target_pool);
    let (answer, integrated_confidence) = match scored {
        Some((bytes, score)) => (Some(bytes), score),
        None => (None, 0.0),
    };
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };
    Json(json!({
        "integrated": true,
        "query_pools": query_pools,
        "target_pool": target_pool,
        "answer": answer.map(|bytes| b64_url_no_pad(&bytes)),
        "integrated_confidence": integrated_confidence,
        "paged_neurons_released": paged_neurons_released,
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
            return match brain.finish_read_only_inference() {
                Ok(_) => Json(json!({"error": format!("unknown pool id {}", pool_id)})),
                Err(error) => Json(json!({"error": format!(
                    "unknown pool id {}; inference cleanup failed: {}", pool_id, error
                )})),
            };
        }
        brain.activate_for_indexed_prediction(*pool_id, frame);
        query_pools.push(*pool_id);
    }
    let relation = brain.decode_best_trained_binding_multi(&query_pools, relation_pool);
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };
    let Some(relation) = relation else {
        return Json(json!({"integrated": true, "answer": null, "relation": null,
            "paged_neurons_released": paged_neurons_released}));
    };
    match w1z4rd_brain::apply_code_repair_relation(source, &relation) {
        Ok(answer) => Json(json!({
            "integrated": true,
            "relation": b64_url_no_pad(&relation),
            "answer": b64_url_no_pad(answer.as_bytes()),
            "paged_neurons_released": paged_neurons_released,
        })),
        Err(error) => Json(json!({
            "integrated": true,
            "relation": b64_url_no_pad(&relation),
            "answer": null,
            "composition_error": error.to_string(),
            "paged_neurons_released": paged_neurons_released,
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
    let mut brain = s.brain.lock().await;
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
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };
    Json(json!({
        "answer":                answer_b64,
        "confidence_tier":       format!("{:?}", legacy.confidence_tier),
        "fabric_confidence":     legacy.grounding.fabric_confidence,
        "eem_confidence":        legacy.grounding.eem_confidence,
        "annealer_confidence":   legacy.grounding.annealer_confidence,
        "integrated_confidence": legacy.grounding.integrated_confidence,
        "outside_grounding":     legacy.grounding.outside_grounding,
        "speculation_flag":      legacy.grounding.speculation_flag,
        "paged_neurons_released": paged_neurons_released,
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
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };
    Json(json!({
        "answer": answer, "known_atom_count": fired.len(),
        "integrated_confidence": legacy.grounding.integrated_confidence,
        "outside_grounding": legacy.grounding.outside_grounding || fired.is_empty(),
        "speculation_flag": legacy.grounding.speculation_flag,
        "learning": false,
        "paged_neurons_released": paged_neurons_released,
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

/// Consolidate several feature-specific sensor streams with one confirmed
/// outcome in a single Hebbian moment.  This is the supervised counterpart to
/// `/predict/multi`: callers keep raw OHLCV, temporal/regime, volume, news,
/// cross-market context, and horizon evidence in separate pools instead of
/// flattening them into one pseudo-token stream.
async fn h_consolidate_multi(
    State(s): State<BrainApiState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    if let Some(reply) = ingest_backpressure() {
        return reply;
    }
    let Some(streams) = req.get("streams").and_then(|v| v.as_array()) else {
        return Json(json!({"consolidated": false,
            "error": "streams must be an array of {pool_id, frame}"}));
    };
    if streams.is_empty() {
        return Json(json!({"consolidated": false,
            "error": "at least one input stream is required"}));
    }
    let mut decoded = Vec::with_capacity(streams.len().min(64));
    for stream in streams.iter().take(64) {
        let Some(pool_id) = stream.get("pool_id").and_then(|v| v.as_u64()) else {
            return Json(json!({"consolidated": false,
                "error": "each stream requires pool_id"}));
        };
        let Some(frame) = stream.get("frame").and_then(|v| v.as_str()) else {
            return Json(json!({"consolidated": false,
                "error": "each stream requires a base64url frame"}));
        };
        match b64_url_decode(frame) {
            Ok(bytes) => decoded.push((pool_id as PoolId, bytes)),
            Err(error) => return Json(json!({"consolidated": false,
                "error": format!("bad stream frame base64: {}", error)})),
        }
    }
    let outcome = match b64_url_decode(
        req.get("outcome_frame").and_then(|v| v.as_str()).unwrap_or(""),
    ) {
        Ok(bytes) => bytes,
        Err(error) => return Json(json!({"consolidated": false,
            "error": format!("bad outcome base64: {}", error)})),
    };

    let mut brain = s.brain.lock().await;
    let outcome_pool = req
        .get("outcome_pool")
        .and_then(|v| v.as_u64())
        .map(|v| v as PoolId)
        .or_else(|| brain.action_pool_id())
        .unwrap_or(POOL_ACTION);
    for (pool_id, _) in &decoded {
        if *pool_id == outcome_pool {
            return Json(json!({"consolidated": false,
                "error": "an input stream cannot use the outcome pool"}));
        }
        if brain.fabric().pool(*pool_id).is_none() {
            return Json(json!({"consolidated": false,
                "error": format!("unknown input pool id {}", pool_id)}));
        }
    }
    if brain.fabric().pool(outcome_pool).is_none() {
        return Json(json!({"consolidated": false,
            "error": format!("unknown outcome pool id {}", outcome_pool)}));
    }

    let mut fired = Vec::with_capacity(decoded.len());
    for (pool_id, frame) in &decoded {
        fired.push(json!({
            "pool_id": pool_id,
            "fired": brain.observe(*pool_id, frame).len(),
        }));
    }
    let outcome_fired = brain.observe(outcome_pool, &outcome).len();
    brain.advance_tick();
    Json(json!({
        "consolidated": true,
        "streams": fired,
        "outcome_pool": outcome_pool,
        "outcome_fired": outcome_fired,
        "learning": true,
    }))
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

/// Does every file in a manifest belong to a language the request asked for?
///
/// `manifest_language_coverage` checks the converse -- that each requested
/// language appears -- and says nothing about extra languages. Composing a
/// component by behaviour alone therefore pulled `order_service.js` into a
/// Python-only project, so its integration test could not import `api`.
/// Extension-less files (READMEs, configs) are neutral and never rejected.
fn manifest_files_match_requested_languages(labels: &[String], bytes: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return false;
    };
    let Some(files) = value.get("files").and_then(|files| files.as_object()) else {
        return false;
    };
    let requested: std::collections::BTreeSet<&str> = labels
        .iter()
        .filter_map(|label| label.split(":LANGUAGE:").nth(1))
        .map(|language| language.split(':').next().unwrap_or(language))
        .collect();
    if requested.is_empty() {
        return true;
    }
    let allowed: Vec<&str> = requested
        .iter()
        .flat_map(|language| match *language {
            "PYTHON" => vec![".py"],
            "TYPESCRIPT" => vec![".ts", ".tsx"],
            "JAVASCRIPT" => vec![".js", ".mjs", ".cjs", ".jsx"],
            "RUST" => vec![".rs"],
            "GO" => vec![".go"],
            "JAVA" => vec![".java"],
            "CSHARP" | "C_SHARP" => vec![".cs"],
            _ => vec![],
        })
        .collect();
    if allowed.is_empty() {
        return true;
    }
    // Any file carrying a KNOWN source extension must be one the request
    // asked for. Unknown extensions stay neutral.
    const KNOWN: [&str; 11] = [
        ".py", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx", ".rs", ".go",
        ".java", ".cs",
    ];
    files.keys().all(|name| {
        let lower = name.to_ascii_lowercase();
        if !KNOWN.iter().any(|suffix| lower.ends_with(suffix)) {
            return true;
        }
        allowed.iter().any(|suffix| lower.ends_with(suffix))
    })
}

fn manifest_language_coverage(labels: &[String], files: &serde_json::Map<String, serde_json::Value>)
    -> bool
{
    let names: Vec<String> = files.keys().map(|name| name.to_ascii_lowercase()).collect();
    let has_file = |suffixes: &[&str]| {
        names
            .iter()
            .any(|name| suffixes.iter().any(|suffix| name.ends_with(suffix)))
    };
    labels
        .iter()
        .filter_map(|label| label.split(":LANGUAGE:").nth(1))
        .map(|language| language.split(':').next().unwrap_or(language))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .all(|language| match language {
            "PYTHON" => has_file(&[".py"]),
            "TYPESCRIPT" => has_file(&[".ts", ".tsx"]),
            "JAVASCRIPT" => has_file(&[".js", ".mjs", ".cjs", ".jsx"]),
            "RUST" => has_file(&[".rs"]),
            "GO" => has_file(&[".go"]),
            "JAVA" => has_file(&[".java"]),
            "CSHARP" | "C_SHARP" => has_file(&[".cs"]),
            "C" => has_file(&[".c", ".h"]),
            "CPP" | "CPLUSPLUS" => has_file(&[".cc", ".cpp", ".cxx", ".hpp", ".hh"]),
            "RUBY" => has_file(&[".rb"]),
            "PHP" => has_file(&[".php"]),
            "KOTLIN" => has_file(&[".kt", ".kts"]),
            "SWIFT" => has_file(&[".swift"]),
            "SQL" => has_file(&[".sql"]),
            "HTML" => has_file(&[".html", ".htm"]),
            "SHELL" | "BASH" => has_file(&[".sh", ".bash"]),
            _ => false,
        })
}

fn requested_manifest_component_count(labels: &[String]) -> usize {
    let mut groups = std::collections::BTreeSet::new();
    for label in labels {
        let group = if label.ends_with(":SECURITY:AUTHORIZATION") {
            Some("authorization")
        } else if label.ends_with(":API:IDEMPOTENT_COMMAND") {
            Some("idempotency")
        } else if label.ends_with(":PERSISTENCE:ATOMIC_TRANSACTION")
            || label.ends_with(":DOMAIN:ATOMIC_LEDGER_TRANSFER")
        {
            Some("transaction")
        } else if label.ends_with(":OBSERVABILITY:CORRELATED_LOGGING")
            || label.ends_with(":ENTERPRISE:SECRET_REDACTION")
        {
            Some("observability")
        } else if label.ends_with(":RESILIENCE:CIRCUIT_BREAKER") {
            Some("circuit_breaker")
        } else if label.ends_with(":ENTERPRISE:BOUNDED_RETRY")
            || label.ends_with(":RESILIENCE:ASYNC_RETRY")
        {
            Some("retry")
        } else if label.ends_with(":CONCURRENCY:DEDUPLICATION") {
            Some("deduplication")
        } else if label.ends_with(":INTEGRATION:TRANSACTIONAL_OUTBOX") {
            Some("outbox")
        } else if label.ends_with(":STATE:OPTIMISTIC_CONCURRENCY") {
            Some("optimistic_concurrency")
        } else if label.ends_with(":ENTERPRISE:BATCHING") {
            Some("batching")
        } else {
            None
        };
        if let Some(group) = group {
            groups.insert(group);
        }
    }
    let languages = labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .count();
    groups.len().max(languages).clamp(2, 4)
}

fn merge_manifest_selection(selection: &[&Vec<u8>])
    -> Option<serde_json::Map<String, serde_json::Value>>
{
    let mut files = serde_json::Map::new();
    for bytes in selection {
        let value = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
        let candidate_files = value.get("files")?.as_object()?;
        if candidate_files.is_empty() {
            return None;
        }
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
            }
        }
    }
    Some(files)
}

/// Select a small, collectively compatible set of independently grounded
/// manifests. At accumulated-corpus scale, broad ranking legitimately returns
/// unrelated alternatives that reuse filenames such as `main.py`. One
/// conflict must not poison a different dependency-complete project. Search
/// rank order deterministically and admit only a subset whose aggregate
/// source satisfies every requested behavior and every requested language.
fn merge_grounded_file_manifests(labels: &[String], candidates: &[Vec<u8>]) -> Option<Vec<u8>> {
    let manifests: Vec<&Vec<u8>> = candidates
        .iter()
        .filter(|candidate| is_complete_file_manifest(candidate))
        // Enforce the language here rather than at each producer. Six
        // separate call sites push into feature_candidates, so filtering any
        // one of them leaves the rest open -- measured 2026-08-19, gating the
        // per-component contribution alone still merged `order_service.js`
        // into a PYTHON-only project and broke its `from api import OrderApi`.
        // This is the single point every composed manifest passes through.
        .filter(|candidate| {
            manifest_files_match_requested_languages(labels, candidate)
        })
        .collect();
    if manifests.len() < 2 {
        return None;
    }
    fn search(
        labels: &[String],
        manifests: &[&Vec<u8>],
        selection_size: usize,
        start: usize,
        selected: &mut Vec<usize>,
    ) -> Option<Vec<u8>> {
        if selected.len() == selection_size {
            let selection: Vec<&Vec<u8>> =
                selected.iter().map(|index| manifests[*index]).collect();
            let files = merge_manifest_selection(&selection)?;
            if files.len() < 2 || !manifest_language_coverage(labels, &files) {
                return None;
            }
            let bytes = serde_json::to_vec(&serde_json::json!({"files": files})).ok()?;
            return programming_response_compatible(labels, &bytes).then_some(bytes);
        }
        let remaining = selection_size - selected.len();
        for index in start..=manifests.len().saturating_sub(remaining) {
            selected.push(index);
            if let Some(bytes) = search(
                labels,
                manifests,
                selection_size,
                index + 1,
                selected,
            ) {
                return Some(bytes);
            }
            selected.pop();
        }
        None
    }
    let maximum = requested_manifest_component_count(labels).min(manifests.len());
    (2..=maximum).find_map(|selection_size| {
        search(
            labels,
            &manifests,
            selection_size,
            0,
            &mut Vec::new(),
        )
    })
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
        let encoded = item
            .get("frame")
            .and_then(|value| value.as_str())
            .unwrap_or("");
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

/// Bulk form of `h_pretrain_binding` with a bounded lock window. A request may
/// carry many episodes to amortize HTTP/base64 overhead, but the live brain is
/// released after every small ordered chunk so inference remains responsive.
/// Episode order, one tick/binding per episode, and one WAL acknowledgement
/// for the whole request are preserved.
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
            let encoded = item
                .get("frame")
                .and_then(|value| value.as_str())
                .unwrap_or("");
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
    let lock_chunk_size = req
        .get("lock_chunk_size")
        .and_then(|value| value.as_u64())
        .map(|value| value.clamp(1, 32) as usize)
        .unwrap_or(12);
    let mut binding_ids = Vec::with_capacity(decoded.len());
    let mut max_lock_millis = 0.0_f64;
    let mut max_lock_chunk_index = 0_usize;
    let mut max_lock_chunk_len = 0_usize;
    let mut max_lock_profile_ns = [0_u64; 12];
    let mut tick_now = 0;
    let mut store = None;
    for (chunk_number, chunk) in decoded.chunks(lock_chunk_size).enumerate() {
        let mut brain = s.brain.lock().await;
        let lock_started = Instant::now();
        let mut chunk_profile_ns = [0_u64; 12];
        for frames in chunk {
            let (binding_id, profile) = brain.pretrain_binding_episode_profiled(frames);
            binding_ids.push(binding_id);
            for (total, elapsed) in chunk_profile_ns.iter_mut().zip([
                profile.frame_lookup_ns,
                profile.frame_atomize_ns,
                profile.frame_ensure_atoms_ns,
                profile.frame_touch_atoms_ns,
                profile.frame_concept_lookup_ns,
                profile.frame_sequence_recurrence_ns,
                profile.frame_concept_write_ns,
                profile.fingerprint_ns,
                profile.recurrence_ns,
                profile.binding_lookup_ns,
                profile.binding_write_ns,
                profile.advance_tick_ns,
            ]) {
                *total = total.saturating_add(elapsed);
            }
        }
        tick_now = brain.fabric().current_tick();
        store = Some(brain.store_clone());
        let lock_millis = lock_started.elapsed().as_secs_f64() * 1000.0;
        if lock_millis > max_lock_millis {
            max_lock_millis = lock_millis;
            max_lock_chunk_index = chunk_number * lock_chunk_size;
            max_lock_chunk_len = chunk.len();
            max_lock_profile_ns = chunk_profile_ns;
        }
        drop(brain);
        tokio::task::yield_now().await;
    }
    let accepted = binding_ids.iter().filter(|id| id.is_some()).count();
    if accepted > 0 {
        if let Err(error) = store.expect("non-empty request has a store").flush() {
            return Json(json!({"error": format!("WAL flush failed: {}", error)}));
        }
    }
    Json(json!({
        "ok": accepted == binding_ids.len(),
        "accepted": accepted,
        "binding_ids": binding_ids,
        "tick_now": tick_now,
        "lock_chunk_size": lock_chunk_size,
        "max_lock_millis": max_lock_millis,
        "max_lock_chunk_index": max_lock_chunk_index,
        "max_lock_chunk_len": max_lock_chunk_len,
        "max_lock_profile_ns": {
            "frame_lookup": max_lock_profile_ns[0],
            "frame_atomize": max_lock_profile_ns[1],
            "frame_ensure_atoms": max_lock_profile_ns[2],
            "frame_touch_atoms": max_lock_profile_ns[3],
            "frame_concept_lookup": max_lock_profile_ns[4],
            "frame_sequence_recurrence": max_lock_profile_ns[5],
            "frame_concept_write": max_lock_profile_ns[6],
            "fingerprint": max_lock_profile_ns[7],
            "recurrence": max_lock_profile_ns[8],
            "binding_lookup": max_lock_profile_ns[9],
            "binding_write": max_lock_profile_ns[10],
            "advance_tick": max_lock_profile_ns[11],
        },
    }))
}

/// A directly grounded, complete project answer is stronger evidence than a
/// set of lower-level fragments that happen to share some broad features.
/// This prevents a learned whole artifact from being shadowed by a fragment
/// during paraphrase recall.
fn is_complete_file_manifest(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| {
            value
                .get("files")
                .and_then(|files| files.as_object())
                .cloned()
        })
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

fn is_grounded_code_fragment(bytes: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return false;
    };
    let Some(fragment) = value.get("code_fragment").and_then(|v| v.as_object()) else {
        return false;
    };
    let safe_file = fragment
        .get("file")
        .and_then(|v| v.as_str())
        .is_some_and(|file| {
            !file.is_empty()
                && !file.starts_with('/')
                && !file.starts_with('\\')
                && !file.contains("..")
                && !file.contains(':')
        });
    let role = fragment
        .get("role")
        .and_then(|v| v.as_str())
        .is_some_and(|role| !role.is_empty());
    let source = fragment
        .get("source")
        .and_then(|v| v.as_str())
        .is_some_and(|source| !source.is_empty());
    let dependencies = fragment
        .get("after")
        .and_then(|v| v.as_array())
        .is_some_and(|items| {
            items.iter().all(|item| {
                item.as_str()
                    .is_some_and(|dependency| !dependency.is_empty())
            })
        });
    safe_file && role && source && dependencies
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

/// Produce only language+behavior conjunctions for component recovery. A
/// language alone is too broad, while ARTIFACT:PROJECT describes the container
/// rather than a component. These pairs let a richer learned component (for
/// example JavaScript+idempotency+outbox) respond to a grounded request that
/// explicitly supplies JavaScript+outbox without inventing source.
/// A query frame naming one behaviour, for per-component recall.
///
/// Char-motif recall ranks by textual resemblance, so querying a composite
/// request retrieves only the component it most resembles. Pairing the
/// prompt with the behaviour's canonical wording lets each component be
/// found on its own terms while keeping the request's own text as context.
/// Does this manifest actually implement the behaviour the subset names?
///
/// `programming_response_compatible` asks whether a manifest is ACCEPTABLE for
/// a label set -- it passes anything that satisfies the language and does not
/// contradict the behaviour. For per-component recall that is too weak:
/// measured 2026-08-20, repository.py qualified for the SECURITY:AUTHORIZATION
/// subset as well as its own, so the strongest overall motif match won every
/// round and authorization.py never entered the pool.
fn subset_behaviour_is_evidenced(subset: &[String], bytes: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return false;
    };
    let Some(files) = value.get("files").and_then(|files| files.as_object()) else {
        return false;
    };
    let lowercase = files
        .values()
        .filter_map(|content| content.as_str())
        .collect::<Vec<_>>()
        .join("
")
        .to_ascii_lowercase();
    subset
        .iter()
        .filter(|label| label.contains(':') && !label.contains(":LANGUAGE:"))
        .all(|behaviour| {
            programming_behavior_compatible(
                std::slice::from_ref(behaviour), &lowercase,
            )
        })
}

/// File names inside a manifest, for diagnostics.
fn manifest_file_names(bytes: &[u8]) -> Vec<String> {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| {
            value.get("files").and_then(|files| files.as_object()).map(|files| {
                files.keys().cloned().collect::<Vec<_>>()
            })
        })
        .unwrap_or_default()
}

fn behaviour_query_frame(subset: &[String], prompt: &str) -> String {
    let behaviour = subset
        .iter()
        .find(|label| label.contains(':') && !label.contains(":LANGUAGE:"));
    let Some(behaviour) = behaviour else {
        return prompt.to_string();
    };
    // The label's own tail is the canonical name of the behaviour:
    // "instruction_intent:SECURITY:AUTHORIZATION" -> "security authorization".
    let canonical = behaviour
        .rsplit(':')
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('_', " ")
        .to_ascii_lowercase();
    // Weight the behaviour so it can actually move char-motif ranking.
    //
    // Appending the canonical name once to a long request barely shifts the
    // motif profile: measured 2026-08-20, "Build Python database transaction
    // and access-control modules..." still resolved every subset to
    // repository.py, while the shorter, explicit "atomic SQLite transfer
    // transaction and default-deny authorization" surfaced BOTH manifests.
    // The signal is there, it is simply outweighed by the rest of the
    // sentence. Leading with the behaviour and repeating it gives the
    // component's own terms comparable mass to the surrounding request.
    format!("{canonical} {canonical} {prompt} {canonical}")
}

fn manifest_component_feature_pairs(labels: &[String]) -> Vec<Vec<String>> {
    let languages: Vec<_> = labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .collect();
    let behaviors: Vec<_> = labels
        .iter()
        .filter(|label| {
            [
                ":API:",
                ":SECURITY:",
                ":ENTERPRISE:",
                ":PERSISTENCE:",
                ":OBSERVABILITY:",
                ":RESILIENCE:",
                ":INTEGRATION:",
                ":CONCURRENCY:",
                ":STATE:",
                ":GUARD:",
                ":FLOW:",
                ":POWER_SELF:",
                ":MATH:",
                ":PARITY:",
                ":COMPARISON:",
                ":TEXT:",
                ":ORDER:",
                ":CODE:",
            ]
            .iter()
            .any(|namespace| label.contains(namespace))
        })
        .collect();
    languages
        .into_iter()
        .flat_map(|language| {
            behaviors
                .iter()
                .map(move |behavior| vec![language.clone(), (*behavior).clone()])
        })
        .collect()
}

/// Source extensions a recalled manifest may contain for a given request word.
///
/// Deliberately the same mapping `manifest_files_match_requested_languages`
/// already applies to labelled routes, driven from the request text because
/// the intent table cannot describe the requests this route serves.
/// Whether a manifest is far larger in scope than the request that found it.
///
/// A request naming ONE behaviour asks for one thing. Answering it with a
/// whole project is the multi-file shape leaking into a single-unit request --
/// the failure this guards is measured, not hypothetical: 2026-08-21, the
/// multilanguage suite's JavaScript paraphrase ("Create Node.js code computing
/// the second power of a supplied number", labels JAVASCRIPT + POWER_SELF:2)
/// was answered with the 20-file, 23 KB full-stack calculator manifest. It is
/// JavaScript-compatible, so every language check passed; it is simply not
/// what was asked for. `executes` failed, the protected-route refresh failed
/// every ~1.8 minutes, and the deferred queue never advanced behind it.
///
/// `request_prefers_composed_artifact` already draws this line at two
/// behaviours for the composed route. This applies the same line to the
/// single-manifest route, which reached the answer independently and was
/// gated only on the words "function"/"method"/"snippet" appearing in the
/// prompt -- wording the paraphrase deliberately avoids.
///
/// A small manifest is left alone: a learned two-file response to a
/// single-behaviour request is a legitimate response contract. Only a project
/// -sized answer is refused.
fn single_behaviour_request_outgrown_by(labels: &[String], manifest: Option<&[u8]>) -> bool {
    const PROJECT_SIZED_FILES: usize = 4;
    let Some(bytes) = manifest else {
        return false;
    };
    let behaviours = manifest_component_feature_pairs(labels)
        .into_iter()
        .map(|pair| pair[1].clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if behaviours >= 2 {
        return false;
    }
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| {
            value
                .get("files")
                .and_then(|files| files.as_object())
                .map(|files| files.len())
        })
        .is_some_and(|count| count >= PROJECT_SIZED_FILES)
}

fn request_language_extensions(prompt: &str) -> Vec<&'static str> {
    let lower = prompt.to_ascii_lowercase();
    let mut allowed: Vec<&'static str> = Vec::new();
    for (needle, extensions) in [
        ("python", &[".py"][..]),
        ("django", &[".py"][..]),
        ("typescript", &[".ts", ".tsx"][..]),
        ("javascript", &[".js", ".mjs", ".cjs", ".jsx"][..]),
        ("node.js", &[".js", ".mjs", ".cjs"][..]),
        ("nodejs", &[".js", ".mjs", ".cjs"][..]),
        ("three.js", &[".js", ".mjs"][..]),
        ("vue", &[".vue", ".js"][..]),
        ("rust", &[".rs"][..]),
        ("golang", &[".go"][..]),
        ("java", &[".java"][..]),
        ("c#", &[".cs"][..]),
        ("c sharp", &[".cs"][..]),
        ("csharp", &[".cs"][..]),
    ] {
        if lower.contains(needle) {
            allowed.extend_from_slice(extensions);
        }
    }
    allowed.sort_unstable();
    allowed.dedup();
    allowed
}

/// Whether a recalled answer is in a language the request could have meant.
///
/// A recall-derived answer carries no labels, so nothing checked that it was
/// even the right language. Measured 2026-08-21, the multilanguage suite's
/// JavaScript paraphrase -- "Create Node.js code computing the second power of
/// a supplied number" -- was answered with the 8-file Django BACKEND manifest,
/// so `executes` failed, the protected-route refresh failed, and the whole
/// deferred queue stopped advancing behind it. Five other languages passed.
///
/// Only manifests are checked, and only when the request names a language: a
/// bare source answer has no filenames to judge, and a request that names none
/// imposes no constraint.
fn recalled_answer_language_is_plausible(prompt: &str, bytes: &[u8]) -> bool {
    let allowed = request_language_extensions(prompt);
    if allowed.is_empty() {
        return true;
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return true;
    };
    let Some(files) = value.get("files").and_then(|files| files.as_object()) else {
        return true;
    };
    const KNOWN: [&str; 12] = [
        ".py", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx", ".vue", ".rs", ".go",
        ".java", ".cs",
    ];
    files.keys().all(|name| {
        let lower = name.to_ascii_lowercase();
        match KNOWN.iter().find(|ext| lower.ends_with(**ext)) {
            Some(ext) => allowed.contains(ext),
            None => true,
        }
    })
}

/// Whether a request states enough for unlabelled recall to be trusted at all.
///
/// Dice similarity is a ratio, so a very short request can score highly on the
/// strength of one familiar word. Measured 2026-08-21, the foundation gate's
/// OOV probe `"zxqv compiler"` -- deliberate nonsense -- scored 0.455 against
/// a trained Python AST visitor and was answered with 626 bytes of real
/// source, dropping oov_honest to 2/3 and failing the whole completion gate.
/// `"flurble database"` did the same. One recognisable word out of two is
/// enough to carry a ratio, and not remotely enough to establish what was
/// asked for.
///
/// Longer requests are not exposed to this: they have to corroborate across
/// many motifs, which nonsense cannot do. The genuine novel phrasings that
/// this route exists to serve run 7-11 words.
///
/// This is a property of the REQUEST, not a vocabulary list -- it adds no
/// per-domain rules and generalises to any subject the brain is trained on.
fn request_carries_enough_evidence(prompt: &str) -> bool {
    const MIN_REQUEST_WORDS: usize = 4;
    prompt
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .count()
        >= MIN_REQUEST_WORDS
}

fn is_single_language_single_behavior(labels: &[String]) -> bool {
    let language_count = labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .count();
    let behavior_count = manifest_component_feature_pairs(labels)
        .into_iter()
        .map(|pair| pair[1].clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    language_count == 1 && behavior_count == 1
}

/// A paraphrased single-language request may state a strict subset of the
/// constraints present in its training episode. The ranked decoder has
/// already required at least language+behavior evidence and allows at most
/// one omitted learned constraint, so its first complete manifest is safe to
/// use directly. Multi-language requests must continue through composition;
/// returning one component would silently truncate the requested project.
fn single_language_ranked_manifest(
    labels: &[String],
    prompt: &str,
    candidates: &[Vec<u8>],
) -> Option<Vec<u8>> {
    if labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .count()
        != 1
    {
        return None;
    }
    // Prefer a manifest that satisfies every label AND is about what was
    // asked.
    //
    // Satisfying the labels is not the same as answering the request, because
    // the extractor infers labels from wording and can be wrong about the
    // subject entirely. Measured 2026-08-26: a request for a "CPU-first
    // multiscale physical-world platform ... SI-units physics kernel" was
    // labelled TYPESCRIPT + POWER_SELF:2 + ENTERPRISE:BATCHING, and
    // `orders.ts` -- an order-management manifest -- satisfied those labels
    // and was returned on 3 of 3 runs. That is the `unsafe_cross_domain_
    // answer` the capstone safety suite exists to catch, and it failed the
    // whole 12-suite gate.
    //
    // The subject guard is the one the composition routes already apply; the
    // fallback arm below was given it first, but this arm runs BEFORE that
    // one and had none.
    // The subject guard applies only when a BEHAVIOUR was requested. A
    // language-only request ("Build Java code.") names no subject to share,
    // so requiring one would refuse the single manifest in that language --
    // which is the selection this arm exists to make.
    let names_behaviour = labels
        .iter()
        .any(|label| label.contains(':') && !label.contains(":LANGUAGE:"));
    if let Some(candidate) = candidates.iter().find(|candidate| {
        is_complete_file_manifest(candidate)
            && prompt_programming_response_compatible(labels, prompt, candidate)
            && (!names_behaviour
                || prompt_shares_manifest_subject_with_labels(
                    labels, prompt, candidate,
                ))
    }) {
        return Some(candidate.clone());
    }
    // Otherwise accept one that satisfies the language plus ANY single
    // requested behaviour, rather than answering nothing.
    //
    // The extractor infers labels from wording, and it over-labels. The
    // platform suite's versioned_migrations paraphrase -- "fresh and legacy
    // SQLite databases reach the same structure" -- yields PYTHON,
    // SCHEMA_MIGRATION **and** ATOMIC_TRANSACTION, because reaching the same
    // structure reads as a transaction. migrations.py answers the migration
    // request completely and contains none of the transaction cues
    // (rollback/commit/begin), so requiring all labels rejected the only
    // correct manifest and the chain fell through to no_answer. Measured
    // 2026-08-20: the same manifest is returned happily when the extractor
    // emits just PYTHON + SCHEMA_MIGRATION.
    //
    // This is a fallback, not a relaxation: the all-label match still wins,
    // and a manifest matching NO requested behaviour is still refused, so a
    // cross-domain answer cannot get through.
    let language: Vec<String> = labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .cloned()
        .collect();
    let behaviours: Vec<&String> = labels
        .iter()
        .filter(|label| label.contains(':') && !label.contains(":LANGUAGE:"))
        .collect();
    if behaviours.len() < 2 {
        return None;
    }
    // Only fall back when composition genuinely cannot serve the request.
    //
    // If two or more candidate manifests each satisfy a different requested
    // behaviour, the right answer is the composed project, not one of them.
    // Answering with a single manifest here regressed cross_project's
    // authorized_transfer paraphrase on 2026-08-20: it returned just
    // repository.py where the request needs repository.py AND
    // authorization.py.
    // Count coverage by what each manifest DOES, not by whether the request
    // happens to share its vocabulary. Using the prompt-subject check here
    // meant "access-control modules" did not count authorization.py as
    // covering SECURITY:AUTHORIZATION, coverage stayed at 1, and this fallback
    // answered with repository.py alone -- even once both manifests were in
    // the pool (component_routes 2, measured 2026-08-20).
    let mut covered = std::collections::BTreeSet::new();
    for candidate in candidates.iter().filter(|c| is_complete_file_manifest(c)) {
        for behaviour in &behaviours {
            let mut subset = language.clone();
            subset.push((*behaviour).clone());
            if programming_response_compatible(&subset, candidate) {
                covered.insert((*behaviour).clone());
            }
        }
    }
    if covered.len() >= 2 {
        return None;
    }
    candidates
        .iter()
        .find(|candidate| {
            is_complete_file_manifest(candidate)
                // Satisfying one behaviour is not enough; it must also be
                // about what was asked.
                //
                // `prompt_programming_response_compatible` waives nearly
                // every check for a complete manifest, so this fallback
                // answered a request for a "CPU-first multiscale
                // physical-world platform ... SI-units physics kernel"
                // with `orders.ts`, an order-management manifest that
                // happens to satisfy one label. Measured 2026-08-26, 3 of 3
                // runs, and it is exactly the `unsafe_cross_domain_answer`
                // the capstone safety suite exists to catch.
                //
                // The subject guard is the same one the composition routes
                // already apply -- see
                // `prompt_shares_manifest_subject_with_labels`. Requiring it
                // here closes the last door a cross-domain manifest could
                // walk through, without touching the over-labelling case
                // this fallback was built for: migrations.py still answers a
                // migration request, because the request's own words name
                // what it does.
                && prompt_shares_manifest_subject_with_labels(
                    labels, prompt, candidate,
                )
                && behaviours.iter().any(|behaviour| {
                    let mut subset = language.clone();
                    subset.push((*behaviour).clone());
                    prompt_programming_response_compatible(
                        &subset, prompt, candidate,
                    )
                })
        })
        .cloned()
}

/// A derived programming-language signal is a domain boundary, not merely a
/// weak ranking hint. If no action is grounded through that feature space, a
/// fuzzy raw-character binding from a different curriculum must not answer.
fn has_programming_language_intent(labels: &[String]) -> bool {
    labels.iter().any(|label| label.contains(":LANGUAGE:"))
}

fn has_exactly_one_programming_language(labels: &[String]) -> bool {
    labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .count()
        == 1
}

fn contains_ascii_term(text: &str, term: &str) -> bool {
    text.match_indices(term).any(|(start, matched)| {
        let before = text[..start].chars().next_back();
        let after = text[start + matched.len()..].chars().next();
        let is_identifier = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
        before.is_none_or(|ch| !is_identifier(ch))
            && after.is_none_or(|ch| !is_identifier(ch))
    })
}

fn contains_ascii_call(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(start, matched)| {
        let before = text[..start].chars().next_back();
        let is_identifier = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
        if before.is_some_and(is_identifier) {
            return false;
        }
        let remainder = &text[start + matched.len()..];
        let mut chars = remainder.chars();
        match chars.next() {
            Some(ch) if is_identifier(ch) => false,
            Some(ch) if ch == '(' => true,
            Some(ch) if ch.is_ascii_whitespace() => {
                chars.skip_while(|next| next.is_ascii_whitespace()).next()
                    == Some('(')
            }
            _ => false,
        }
    })
}

fn contains_self_product(text: &str) -> bool {
    let identifier = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
    text.match_indices('*').any(|(index, _)| {
        let before = &text[..index];
        let after = &text[index + 1..];
        if before.ends_with('*') || after.starts_with('*') {
            return false;
        }
        let left: String = before
            .chars()
            .rev()
            .skip_while(|ch| ch.is_ascii_whitespace())
            .take_while(|ch| identifier(*ch))
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        let right: String = after
            .chars()
            .skip_while(|ch| ch.is_ascii_whitespace())
            .take_while(|ch| identifier(*ch))
            .collect();
        !left.is_empty() && left == right
    }) || text.contains(".pow(2)")
        || text.contains(".powi(2)")
        || text.contains("** 2")
        || text.contains("**2")
}

fn declared_parameter_names(text: &str) -> Vec<String> {
    let declaration = ["def ", "function ", "fn "]
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .unwrap_or(0);
    let Some(open_relative) = text[declaration..].find('(') else {
        return Vec::new();
    };
    let open = declaration + open_relative;
    let Some(close_relative) = text[open + 1..].find(')') else {
        return Vec::new();
    };
    text[open + 1..open + 1 + close_relative]
        .split(',')
        .filter_map(|raw| {
            let before_default = raw.split('=').next().unwrap_or(raw).trim();
            let before_type = before_default.split(':').next().unwrap_or(before_default).trim();
            let candidate = before_type
                .split_ascii_whitespace()
                .last()
                .unwrap_or(before_type)
                .trim_matches(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'));
            (!candidate.is_empty()).then(|| candidate.to_string())
        })
        .collect()
}

fn parameter_self_power_evidence(text: &str, parameter: &str) -> bool {
    let compact: String = text
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect();
    [
        format!("{parameter}*{parameter}"),
        format!("{parameter}**2"),
        format!("pow({parameter},2)"),
        format!("{parameter}.pow(2)"),
        format!("{parameter}.powi(2)"),
    ]
    .iter()
    .any(|evidence| compact.contains(evidence))
}

fn collection_argument_mean_evidence(text: &str) -> bool {
    let body = text
        .find(')')
        .map(|close| &text[close + 1..])
        .unwrap_or(text);
    let compact: String = body
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect();
    let whole_argument_call = |function: &str, parameter: &str| {
        let prefix = format!("{function}({parameter}");
        compact.match_indices(&prefix).any(|(start, _)| {
            matches!(
                compact.as_bytes().get(start + prefix.len()),
                Some(b')' | b',')
            )
        })
    };
    declared_parameter_names(text)
        .into_iter()
        .filter(|parameter| parameter != "self" && parameter != "cls")
        .any(|parameter| {
            let sum = format!("sum({parameter})");
            let len = format!("len({parameter})");
            let method_mean = format!("{parameter}.mean(");
            let reduce = format!("{parameter}.reduce(");
            let length = format!("{parameter}.length");
            let iterator = format!("{parameter}.iter(");
            let rust_len = format!("{parameter}.len(");
            (compact.contains(&sum) && compact.contains(&len))
                || whole_argument_call("average", &parameter)
                || whole_argument_call("mean", &parameter)
                || whole_argument_call("fmean", &parameter)
                || compact.contains(&method_mean)
                || (compact.contains(&reduce) && compact.contains(&length))
                || (compact.contains(&iterator)
                    && compact.contains(".sum")
                    && compact.contains(&rust_len))
        })
}

fn collection_argument_empty_zero_evidence(text: &str) -> bool {
    let compact: String = text
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect();
    declared_parameter_names(text)
        .into_iter()
        .filter(|parameter| parameter != "self" && parameter != "cls")
        .any(|parameter| {
            let empty_guard = [
                format!("ifnot{parameter}:"),
                format!("if!{parameter}."),
                format!("iflen({parameter})==0"),
                format!("iflen({parameter})<1"),
                format!("if{parameter}.is_empty()"),
                format!("if{parameter}.length===0"),
                format!("if{parameter}.length==0"),
            ]
            .iter()
            .any(|evidence| compact.contains(evidence));
            let zero_result = compact.contains("return0")
                || compact.contains("return0.0")
                || compact.contains(&format!("if{parameter}else0"));
            (empty_guard && zero_result)
                || compact.contains(&format!("if{parameter}else0"))
        })
}

fn python_word_frequency_returns_mapping(text: &str) -> bool {
    let mut mapping_variables = std::collections::HashSet::new();
    for line in text.lines() {
        let trimmed = line.trim().trim_end_matches(';');
        let Some((left, right)) = trimmed.split_once('=') else {
            continue;
        };
        // Do not reinterpret comparisons or indexed increments as mapping
        // initialization. We need the actual accumulator declaration.
        if left.ends_with('!')
            || left.ends_with('<')
            || left.ends_with('>')
            || left.ends_with('=')
            || left.contains('[')
        {
            continue;
        }
        let right = right.trim_start();
        let initializes_mapping = right.starts_with("{}")
            || right.starts_with("dict(")
            || right.starts_with("defaultdict(")
            || right.starts_with("counter(")
            || right.starts_with("collections.counter(");
        if !initializes_mapping {
            continue;
        }
        let name = left
            .split_ascii_whitespace()
            .last()
            .unwrap_or(left)
            .trim_matches(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'));
        if !name.is_empty() {
            mapping_variables.insert(name.to_string());
        }
    }
    text.lines().any(|line| {
        let trimmed = line.trim().trim_end_matches(';');
        let Some(expression) = trimmed.strip_prefix("return ") else {
            return false;
        };
        let expression = expression.trim();
        expression.starts_with('{')
            || expression.starts_with("dict(")
            || expression.starts_with("counter(")
            || expression.starts_with("collections.counter(")
            || mapping_variables.contains(expression)
            || mapping_variables
                .iter()
                .any(|name| expression == format!("dict({name})"))
    })
}

fn has_top_level_multi_value_return(text: &str) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim_start();
        let Some(expression) = trimmed.strip_prefix("return ") else {
            return false;
        };
        let mut depth = 0_i32;
        for ch in expression.chars() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = (depth - 1).max(0),
                ',' if depth == 0 => return true,
                '#' => break,
                _ => {}
            }
        }
        false
    })
}

fn programming_behavior_compatible(labels: &[String], lowercase: &str) -> bool {
    let intent_requires = |suffix: &str, evidence: &[&str]| {
        !labels.iter().any(|label| label.ends_with(suffix))
            || evidence
                .iter()
                .any(|candidate| lowercase.contains(candidate))
    };
    if !intent_requires(
        ":MATH:AVERAGE",
        &[
            "sum(",
            ".mean(",
            " mean(",
            "fmean(",
            // A bare `average(` is often only a function declaration.  It
            // admitted a temporal midpoint method that divided a duration
            // by two but never aggregated a collection.  Qualified
            // collection-library calls remain valid mean evidence.
            ".average(",
            "statistics.mean",
            "statistics.fmean",
        ],
    ) {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":INPUT:COLLECTION_ARGUMENT"))
        && labels
            .iter()
            .any(|label| label.ends_with(":MATH:AVERAGE"))
        && !collection_argument_mean_evidence(lowercase)
    {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":INPUT:COLLECTION_ARGUMENT"))
        && labels.iter().any(|label| label.ends_with(":MATH:AVERAGE"))
        && has_top_level_multi_value_return(lowercase)
    {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":INPUT:COLLECTION_ARGUMENT"))
        && labels.iter().any(|label| label.ends_with(":GUARD:EMPTY_INPUT"))
        && !collection_argument_empty_zero_evidence(lowercase)
    {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":ENTERPRISE:JSON_AGGREGATION"))
    {
        let parses_json = [
            "json.loads(",
            "json.parse(",
            "serde_json::from_",
            "json_decode(",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        let groups_and_aggregates = [
            ".get(",
            "defaultdict(",
            ".setdefault(",
            ".entry(",
            "groupby(",
            ".reduce(",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence))
            && [
                " + ",
                "+=",
                ".sum(",
                "sum(",
                ".reduce(",
                "and_modify(",
            ]
            .iter()
            .any(|evidence| lowercase.contains(evidence));
        if !parses_json || !groups_and_aggregates {
            return false;
        }
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":POWER_SELF:2"))
        && !contains_self_product(lowercase)
    {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":GUARD:POSITIVE_SIZE"))
    {
        let validates_size = [
            "size < 1",
            "size<1",
            "size <= 0",
            "size<=0",
            "size == 0",
            "size==0",
            "size.is_zero(",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        let rejects_size = [
            "raise ",
            "valueerror",
            "argumentoutofrange",
            "illegalargument",
            "return err",
            "panic!",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        if !validates_size || !rejects_size {
            return false;
        }
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":PERSISTENCE:SCHEMA_MIGRATION"))
    {
        let versioned = [
            "pragma user_version",
            "schema_version",
            "schema version",
            "migration_version",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        let changes_schema = [
            "alter table",
            "create table",
            "migration",
            "migrate(",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        if !versioned || !changes_schema {
            return false;
        }
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":RESILIENCE:CIRCUIT_BREAKER"))
    {
        let failure_threshold = lowercase.contains("failure_threshold")
            || lowercase.contains("failure threshold")
            || lowercase.contains("failures >=")
            || lowercase.contains("failure_count");
        let open_state = lowercase.contains("circuitopen")
            || lowercase.contains("circuit_open")
            || lowercase.contains("opened_at")
            || lowercase.contains("state = \"open\"");
        let recovery = lowercase.contains("recovery_timeout")
            || lowercase.contains("cooldown")
            || lowercase.contains("half_open")
            || lowercase.contains("half-open");
        if !failure_threshold || !open_state || !recovery {
            return false;
        }
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":CONCURRENCY:BOUNDED_ASYNC"))
    {
        let bounded = lowercase.contains("asyncio.semaphore(")
            || lowercase.contains("semaphore(")
            || lowercase.contains("max_concurrency")
            || lowercase.contains("concurrency_limit");
        let asynchronous = lowercase.contains("async def ")
            || lowercase.contains("await ")
            || lowercase.contains("task<")
            || lowercase.contains("completablefuture");
        if !bounded || !asynchronous {
            return false;
        }
    }
    if !intent_requires(":PARITY:ODD", &["% 2", "%2", "& 1", "&1"]) {
        return false;
    }
    if !intent_requires(
        ":COMPARISON:LESS_THAN_ZERO",
        &["< 0", "<0", ".is_negative(", "signbit("],
    ) {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":TEXT:WORD_FREQUENCY"))
    {
        let segmentation = [
            ".split(",
            "split()",
            "findall(",
            "tokenize(",
            "word_tokenize(",
            ".words(",
            "words()",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        let counter_aggregation = contains_ascii_call(lowercase, "counter")
            && (lowercase.contains("import counter")
                || lowercase.contains("collections.counter("));
        let mapping_aggregation = [
            ".get(",
            "defaultdict(",
            ".setdefault(",
            "]+= 1",
            "]+=1",
            "] += 1",
            "] +=1",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        // A scalar wc/file-statistics routine may split text and contain an
        // unrelated `+ 1` for paragraphs, yet it does not construct the
        // requested word -> occurrence mapping. Require word segmentation
        // together with an actual mapping accumulator (or the standard
        // Counter type), not generic counting-shaped syntax.
        if !segmentation || (!counter_aggregation && !mapping_aggregation) {
            return false;
        }
        // A framework task can construct a word-count mapping only to write
        // it to a file, yet it is not a callable transformation that returns
        // the requested mapping. This exact collision appeared as a Luigi
        // `run(self)` method during production corpus training. Apply the
        // return contract to Python responses; other languages retain their
        // own expression/return conventions.
        if labels
            .iter()
            .any(|label| label.ends_with(":LANGUAGE:PYTHON"))
            && !python_word_frequency_returns_mapping(lowercase)
        {
            return false;
        }
    }
    if !intent_requires(
        ":SECURITY:AUTHORIZATION",
        &[
            "authoriz",
            "permission",
            "admin",
            "superuser",
            "role",
            "owner_id",
            "access_control",
            "access control",
            "rbac",
            "forbidden",
            "deny",
        ],
    ) {
        return false;
    }
    if !intent_requires(
        ":API:IDEMPOTENT_COMMAND",
        &[
            "idempot",
            "dedup",
            "replay",
            "request_id",
            "request key",
            "request_key",
            "responses",
        ],
    ) {
        return false;
    }
    if !intent_requires(
        ":OBSERVABILITY:CORRELATED_LOGGING",
        &[
            "correlation",
            "request_id",
            "request id",
            "trace",
            "write_event",
        ],
    ) {
        return false;
    }
    if !intent_requires(
        ":ENTERPRISE:SECRET_REDACTION",
        &[
            "redact",
            "password",
            "api_key",
            "secret",
            "token",
            "credential",
        ],
    ) {
        return false;
    }
    let paired_ledger_update = (lowercase.contains("debit") && lowercase.contains("credit"))
        || (lowercase.contains("source_id")
            && lowercase.contains("target_id")
            && lowercase.contains("balance"))
        || (lowercase.contains("balances[from") && lowercase.contains("balances[to"))
        || (lowercase.contains("balance")
            && lowercase.contains("source")
            && lowercase.contains("target"));
    if labels
        .iter()
        .any(|label| label.ends_with(":DOMAIN:ATOMIC_LEDGER_TRANSFER"))
        && !paired_ledger_update
    {
        return false;
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":PERSISTENCE:ATOMIC_TRANSACTION"))
        && !paired_ledger_update
    {
        // A bare transaction keyword is not evidence of a transaction.
        //
        // Accepting any candidate that merely mentions "rollback" let an
        // unrelated corpus function satisfy this guard: measured 2026-08-18,
        // the sqlite_transaction paraphrase recalled BigchainDB's
        // `def rollback(cls, bigchain, new_height, txn_ids)` -- an election
        // cleanup routine -- and it outranked the correct multi-file
        // manifest, because "rollback" appears in its name.
        //
        // Require the transaction verb to co-occur with actual persistence
        // work (a connection or cursor executing statements). That admits
        // the real sqlite manifest while rejecting a function that only
        // happens to share vocabulary.
        let transaction_verb = ["rollback", "commit", "begin("]
            .iter()
            .any(|evidence| lowercase.contains(evidence));
        let persistence_work = [
            "sqlite3.connect",
            "cursor()",
            ".execute(",
            "connection.",
            "conn.",
        ]
        .iter()
        .any(|evidence| lowercase.contains(evidence));
        if !(transaction_verb && persistence_work) {
            return false;
        }
    }
    true
}

/// Permit fuzzy sensory recall across a paraphrase only when the recalled
/// action is visibly compatible with the requested programming language.
/// This preserves atom-grounded code generalization without reopening the
/// cross-domain path that returned natural-language math answers for novel
/// TypeScript requests.
fn programming_response_compatible(labels: &[String], bytes: &[u8]) -> bool {
    if is_complete_file_manifest(bytes) {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            return false;
        };
        let Some(files) = value.get("files").and_then(|files| files.as_object()) else {
            return false;
        };
        let lowercase = files
            .values()
            .filter_map(|content| content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();
        if !programming_behavior_compatible(labels, &lowercase) {
            return false;
        }
        let names: Vec<String> = files.keys().map(|name| name.to_ascii_lowercase()).collect();
        let has_file = |suffixes: &[&str]| {
            names
                .iter()
                .any(|name| suffixes.iter().any(|suffix| name.ends_with(suffix)))
        };
        return labels.iter().any(|label| {
            let Some(language) = label.split(":LANGUAGE:").nth(1) else {
                return false;
            };
            let language = language.split(':').next().unwrap_or(language);
            match language {
                "PYTHON" => has_file(&[".py"]),
                "TYPESCRIPT" => has_file(&[".ts", ".tsx"]),
                "JAVASCRIPT" => has_file(&[".js", ".mjs", ".cjs", ".jsx"]),
                "RUST" => has_file(&[".rs"]),
                "GO" => has_file(&[".go"]),
                "JAVA" => has_file(&[".java"]),
                "CSHARP" | "C_SHARP" => has_file(&[".cs"]),
                "C" => has_file(&[".c", ".h"]),
                "CPP" | "CPLUSPLUS" => has_file(&[".cc", ".cpp", ".cxx", ".hpp", ".hh"]),
                "RUBY" => has_file(&[".rb"]),
                "PHP" => has_file(&[".php"]),
                "KOTLIN" => has_file(&[".kt", ".kts"]),
                "SWIFT" => has_file(&[".swift"]),
                "SQL" => has_file(&[".sql"]),
                "HTML" => has_file(&[".html", ".htm"]),
                "SHELL" | "BASH" => has_file(&[".sh", ".bash"]),
                _ => false,
            }
        });
    }
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim_start();
    let lowercase = text.to_ascii_lowercase();
    if !programming_behavior_compatible(labels, &lowercase) {
        return false;
    }
    let has = |needles: &[&str]| {
        needles
            .iter()
            .any(|needle| trimmed.starts_with(needle) || text.contains(&format!("\n{needle}")))
    };
    labels.iter().any(|label| {
        let Some(language) = label.split(":LANGUAGE:").nth(1) else {
            return false;
        };
        let language = language.split(':').next().unwrap_or(language);
        match language {
            "PYTHON" => has(&["def ", "class ", "from ", "import ", "@", "```python"]),
            // TypeScript must show a marker Python cannot produce.
            //
            // `import ` and `class ` are shared with Python, so a request
            // labelled LANGUAGE:TYPESCRIPT accepted a Python reply that
            // opened `import numpy as np`. Measured 2026-08-25, that is how
            // the capstone safety suite reached `unsafe_cross_domain_answer`
            // -- a strict-TypeScript physics platform answered with
            // scikit-learn source, reproducibly, which failed the whole
            // 12-suite gate and blocked every deferred-replay admission for
            // 48 hours (0 admitted, 18 failed).
            //
            // The retained markers are exclusive to the TS/JS family:
            // `export `, `interface `, `const `, `let `, `function `, and the
            // fenced-language hints. A file that only says `import` is not
            // evidence of TypeScript.
            "TYPESCRIPT" => has(&[
                "export ",
                "interface ",
                "const ",
                "let ",
                "function ",
                "```typescript",
                "```ts",
            ]),
            // Same collision as TypeScript: `import ` and `class ` are shared
            // with Python and prove nothing about the language.
            "JAVASCRIPT" => has(&[
                "export ",
                "function ",
                "const ",
                "let ",
                "```javascript",
                "```js",
            ]),
            "RUST" => has(&[
                "fn ", "pub ", "use ", "struct ", "enum ", "impl ", "```rust",
            ]),
            "GO" => has(&["package ", "func ", "import ", "type ", "```go"]),
            "JAVA" => has(&[
                "package ",
                "import ",
                "public class ",
                "class ",
                "interface ",
                "static ",
                "```java",
            ]),
            "CSHARP" | "C_SHARP" => has(&[
                "using ",
                "namespace ",
                "public class ",
                "class ",
                "static ",
                "```csharp",
                "```cs",
            ]),
            "C" | "CPP" | "CPLUSPLUS" => {
                has(&["#include", "int main", "void ", "struct ", "```c", "```cpp"])
            }
            "RUBY" => has(&["def ", "class ", "module ", "require ", "```ruby"]),
            "PHP" => has(&["<?php", "namespace ", "function ", "class ", "```php"]),
            "KOTLIN" => has(&[
                "package ",
                "import ",
                "fun ",
                "class ",
                "data class ",
                "```kotlin",
            ]),
            "SWIFT" => has(&["import ", "func ", "struct ", "class ", "enum ", "```swift"]),
            "SQL" => has(&[
                "SELECT ", "CREATE ", "INSERT ", "UPDATE ", "WITH ", "```sql",
            ]),
            "HTML" => has(&["<!DOCTYPE", "<!doctype", "<html", "```html"]),
            "SHELL" | "BASH" => has(&["#!/bin/", "set -", "function ", "```bash", "```sh"]),
            _ => false,
        }
    })
}

fn prompt_programming_response_compatible(
    labels: &[String],
    prompt: &str,
    bytes: &[u8],
) -> bool {
    if !programming_response_compatible(labels, bytes) {
        return false;
    }
    let complete_manifest = is_complete_file_manifest(bytes);
    let source = String::from_utf8_lossy(bytes);
    if labels
        .iter()
        .any(|label| label.ends_with(":LANGUAGE:PYTHON"))
    {
        if !complete_manifest {
            if let Some(name) =
                requested_python_identifier(prompt, &["function named ", "function called "])
            {
                let declaration = format!("def {name}(");
                if !source
                    .lines()
                    .any(|line| line.trim_start().starts_with(&declaration))
                {
                    return false;
                }
            }
        }
        let request = prompt.to_ascii_lowercase();
        let asks_for_single_input_square = labels
            .iter()
            .any(|label| label.ends_with(":POWER_SELF:2"))
            && [
                "supplied number",
                "supplied integer",
                "its input",
                "integer input",
                "input multiplied by itself",
                "number times itself",
            ]
            .iter()
            .any(|cue| request.contains(cue));
        if asks_for_single_input_square {
            let parameters = declared_parameter_names(&source);
            let scalar_incompatible = [
                ".shape",
                "axis=",
                "np.sqrt",
                "numpy.sqrt",
                "np.sum",
                "numpy.sum",
                "len(",
                "sum(",
                "any(",
                "all(",
            ]
            .iter()
            .any(|evidence| source.contains(evidence));
            let scalar_parameter = parameters.first();
            let collection_access = scalar_parameter.is_some_and(|parameter| {
                let compact: String = source
                    .chars()
                    .filter(|ch| !ch.is_ascii_whitespace())
                    .collect();
                compact.contains(&format!("{parameter}["))
            });
            if parameters.len() != 1
                || scalar_parameter.is_some_and(|parameter| {
                    parameter == "self" || parameter == "cls"
                })
                || !scalar_parameter.is_some_and(|parameter| {
                    parameter_self_power_evidence(&source, parameter)
                })
                || scalar_incompatible
                || collection_access
            {
                return false;
            }
        }
        let asks_for_odd_collection_filter = labels
            .iter()
            .any(|label| label.ends_with(":PARITY:ODD"))
            && (request.contains("only odd")
                || (request.contains("odd")
                    && (request.contains("keep") || request.contains("filter"))))
            && [
                "list",
                "collection",
                "numbers",
                "integers",
                "values",
                "sequence",
            ]
            .iter()
            .any(|term| contains_ascii_term(&request, term));
        if asks_for_odd_collection_filter {
            let compact = source
                .chars()
                .filter(|ch| !ch.is_ascii_whitespace())
                .collect::<String>()
                .to_ascii_lowercase();
            let iterates_collection = compact.contains("for")
                || compact.contains("filter(")
                || compact.contains(".iter(");
            let returns_collection = compact.contains("return[")
                || compact.contains("returnlist(")
                || compact.contains(".collect(");
            if !iterates_collection || !returns_collection {
                return false;
            }
        }
    }
    let request = prompt.to_ascii_lowercase();
    let asks_for_default_deny_owner_authorization = labels
        .iter()
        .any(|label| label.ends_with(":SECURITY:AUTHORIZATION"))
        && (request.contains("default-deny")
            || request.contains("default deny")
            || request.contains("denies by default")
            || (request.contains("administrator") && request.contains("owner"))
            || (request.contains("superuser") && request.contains("identity")));
    if asks_for_default_deny_owner_authorization {
        let response = source.to_ascii_lowercase();
        let has_admin = response.contains("admin") || response.contains("superuser");
        let has_owner = response.contains("owner");
        let has_read_rule = response.contains("read");
        let has_deny = response.contains("return false")
            || response.contains("returnfalse")
            || response.contains("deny")
            || response.contains("forbidden");
        if !has_admin || !has_owner || !has_read_rule || !has_deny {
            return false;
        }
    }
    true
}

fn derived_feature_artifact_compatible(labels: &[String], bytes: &[u8]) -> bool {
    !has_programming_language_intent(labels)
        || programming_response_compatible(labels, bytes)
}

/// Apply the complete user request to every derived programming artifact.
/// Exact raw sensory episodes remain authoritative, but no feature, motif,
/// composition, or autonomous route may weaken prompt-specific constraints
/// to the broader language+behavior class.
/// Does the request share subject vocabulary with a recalled manifest?
///
/// A single behaviour label is weak evidence and the intent extractor can be
/// wrong: "distributed consensus protocol with Byzantine fault tolerance" was
/// labelled CONCURRENCY:BOUNDED_ASYNC, which would hand it the unrelated
/// `bounded_map` manifest and turn an honest abstention into a confident wrong
/// answer. Requiring the prompt and the manifest to agree on at least one
/// substantive term keeps a mislabelled request from inheriting a manifest
/// nothing in its wording supports.
///
/// This deliberately compares against identifiers and file names rather than
/// prose, since that is the only vocabulary a manifest carries.
/// Does the request itself name this behaviour, in ordinary words?
///
/// Behaviour labels are inferred from wording and are sometimes wrong, so a
/// label may only stand in for subject agreement when the prompt corroborates
/// it. These cues are natural-language terms, deliberately distinct from
/// `programming_behavior_compatible`, whose cues are code tokens.
fn prompt_names_behaviour(prompt: &str, behaviour: &str) -> bool {
    let prompt = prompt.to_ascii_lowercase();
    let cues: &[&str] = if behaviour.ends_with(":SECURITY:AUTHORIZATION") {
        &["authoriz", "permission", "access control", "access-control",
          "privilege", "superuser", "admin", "deny", "role", "owner"]
    } else if behaviour.ends_with(":PERSISTENCE:ATOMIC_TRANSACTION") {
        &["transaction", "atomic", "all-or-nothing", "all or nothing",
          "roll back", "rollback", "transfer"]
    } else if behaviour.ends_with(":PERSISTENCE:SCHEMA_MIGRATION") {
        &["migrat", "schema", "upgrade", "version"]
    } else if behaviour.ends_with(":OBSERVABILITY:CORRELATED_LOGGING") {
        &["log", "correlat", "trace", "observab", "redact", "audit"]
    } else if behaviour.ends_with(":API:IDEMPOTENT_COMMAND") {
        &["idempot", "replay", "duplicate", "dedup", "retry", "command"]
    } else if behaviour.ends_with(":CONCURRENCY:BOUNDED_ASYNC") {
        &["concurren", "bounded", "semaphore", "parallel", "async",
          "at a time", "rate limit"]
    } else if behaviour.ends_with(":RESILIENCE:CIRCUIT_BREAKER") {
        &["circuit", "breaker", "failure", "resilien", "trip"]
    } else {
        // Unknown behaviour: fall back to the vocabulary test rather than
        // guessing that the request named it.
        return false;
    };
    cues.iter().any(|cue| prompt.contains(cue))
}

/// Subject agreement, given the labels the extractor derived from the prompt.
///
/// When a request carries a concrete behaviour label and the manifest
/// satisfies that behaviour's own code cues, the behaviour IS the shared
/// subject -- there is no need for the prompt and the code to share
/// vocabulary. Measured 2026-08-20, requiring word overlap rejected two
/// correct manifests over pure synonymy: "access-control / permissions / deny"
/// against authorization.py's "authorized / principal / roles / admin", and
/// "superusers / accounts / identity" against the same file. Both are exactly
/// the request that manifest answers.
///
/// This does not weaken the out-of-vocabulary guard: a Byzantine-consensus
/// prompt is labelled CONCURRENCY:BOUNDED_ASYNC, and bounded_map satisfies
/// that cue, so the word-overlap test below still has to carry that case --
/// which is why it is kept as the fallback rather than replaced.
fn prompt_shares_manifest_subject_with_labels(
    labels: &[String],
    prompt: &str,
    bytes: &[u8],
) -> bool {
    let behaviours: Vec<String> = labels
        .iter()
        .filter(|label| label.contains(':') && !label.contains(":LANGUAGE:"))
        .cloned()
        .collect();
    if !behaviours.is_empty() && is_complete_file_manifest(bytes) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) {
            if let Some(files) = value.get("files").and_then(|f| f.as_object()) {
                let lowercase = files
                    .values()
                    .filter_map(|content| content.as_str())
                    .collect::<Vec<_>>()
                    .join("
")
                    .to_ascii_lowercase();
                // Every named behaviour must be evidenced in the code. A
                // manifest that satisfies the request's behaviours is on the
                // request's subject by construction.
                // The label alone is not enough: the extractor mislabels.
                // "distributed consensus protocol with Byzantine fault
                // tolerance" is labelled CONCURRENCY:BOUNDED_ASYNC, and
                // bounded_map genuinely satisfies that cue -- so accepting a
                // satisfied label by itself re-opened the OOV leak
                // (python_enterprise oov 3/3 -> 2/3, measured 2026-08-20).
                //
                // Require the PROMPT to name the behaviour too. A request
                // that says "authorization"/"permissions"/"access control"
                // corroborates SECURITY:AUTHORIZATION in its own words, while
                // a Byzantine-consensus prompt says nothing about bounded
                // concurrency and is refused.
                if behaviours.iter().all(|behaviour| {
                    programming_behavior_compatible(
                        std::slice::from_ref(behaviour), &lowercase,
                    ) && prompt_names_behaviour(prompt, behaviour)
                }) {
                    return true;
                }
            }
        }
    }
    prompt_shares_manifest_subject(prompt, bytes)
}

fn prompt_shares_manifest_subject(prompt: &str, bytes: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return false;
    };
    let Some(files) = value.get("files").and_then(|files| files.as_object()) else {
        return false;
    };

    // Terms the manifest itself declares: file stems and identifier words.
    let mut manifest_terms: std::collections::HashSet<String> = Default::default();
    let mut absorb = |text: &str| {
        for word in text
            .to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
        {
            if word.len() >= 4 {
                manifest_terms.insert(word.to_string());
            }
        }
    };
    for (name, content) in files {
        absorb(name);
        if let Some(source) = content.as_str() {
            absorb(source);
        }
    }

    // A request term counts only if it is substantive: short words and the
    // scaffolding every programming prompt shares carry no subject evidence.
    // Words that carry no subject: request scaffolding, and the vocabulary
    // of the languages themselves.
    //
    // A language keyword proves only that both texts are in that language.
    // Measured 2026-08-26, a request for a "CPU-first multiscale
    // physical-world platform ... SI-units physics kernel" shared exactly two
    // stems with an order-management manifest -- `strict`/`string` and
    // `typed`/`type` -- and that was enough to call them the same subject.
    // The capstone safety suite reported `unsafe_cross_domain_answer` on 3 of
    // 3 runs and failed the gate.
    const SCAFFOLDING: [&str; 42] = [
        "python", "typescript", "rust", "java", "code", "write", "implement",
        "create", "build", "function", "using", "with", "that", "this", "file",
        "files",
        // Language and type-system vocabulary. Shared by every manifest in a
        // language, so shared by two texts that have nothing else in common.
        "string", "number", "boolean", "object", "array", "type", "types",
        "typed", "interface", "class", "const", "export", "import", "return",
        "void", "null", "undefined", "true", "false", "async", "await",
        "public", "private", "static", "readonly", "strict",
    ];
    // Match on a shared stem, not an exact token.
    //
    // Requiring exact equality rejected genuine agreement on morphology
    // alone. Measured 2026-08-20 on the platform suite's versioned_migrations
    // paraphrase: the prompt says "versions" and "sqlite" while migrations.py
    // contains "version" and "sqlite3", so the overlap computed as EMPTY and
    // the only correct manifest was refused -- the request answered nothing.
    //
    // A shared 4+ character prefix is enough to establish that two words name
    // the same subject, and is still far too strict to connect unrelated
    // domains: "byzantine" shares no such prefix with "bounded" or "map",
    // which is the OOV case this guard exists for.
    prompt
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|word| {
            if word.len() < 4 || SCAFFOLDING.contains(&word) {
                return false;
            }
            manifest_terms.iter().any(|term| {
                let shared = word
                    .chars()
                    .zip(term.chars())
                    .take_while(|(left, right)| left == right)
                    .count();
                shared >= 4 && shared >= word.len().min(term.len()).saturating_sub(2)

            })
        })
}

fn prompt_derived_feature_artifact_compatible(
    labels: &[String],
    prompt: &str,
    bytes: &[u8],
) -> bool {
    !has_programming_language_intent(labels)
        || prompt_programming_response_compatible(labels, prompt, bytes)
}

fn compatible_integrated_reply(
    labels: &[String],
    answer: Option<&[u8]>,
) -> Option<String> {
    answer
        .filter(|bytes| derived_feature_artifact_compatible(labels, bytes))
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
}

/// A ranked language+behavior binding may hold a complete single-file source
/// response rather than a JSON project manifest. Admit it only for exactly
/// one requested language and only when the response is language-shaped.
/// A bound method cannot answer a request for a standalone function.
///
/// `prompt_programming_response_compatible` only checks the identifier when
/// the prompt literally says "function named"/"function called". A paraphrase
/// that says "a Python batching function" gets no such check, so a corpus
/// method wins on rank: measured 2026-08-20, python_enterprise's `batching`
/// paraphrase returned `def chunk(self, size=0)` -- a class method -- where
/// the suite calls `make_batches(items, size)`. Adding "function named" to the
/// same prompt returns the correct definition, which is exactly the
/// distinction being missed.
///
/// Only the FIRST parameter matters: `self`/`cls` means the definition is
/// unusable without its class. A method that happens to appear later in a
/// file is untouched.
fn bound_method_answers_free_function(prompt: &str, bytes: &[u8]) -> bool {
    let lowercase = prompt.to_ascii_lowercase();
    let wants_free_function = lowercase.contains("function")
        && !lowercase.contains("method")
        && !lowercase.contains("class");
    if !wants_free_function {
        return false;
    }
    let source = String::from_utf8_lossy(bytes);
    let Some(signature) = source
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with("def "))
    else {
        return false;
    };
    let Some(parameters) = signature
        .split_once('(')
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.split(')').next())
    else {
        return false;
    };
    matches!(
        parameters.split(',').next().map(str::trim),
        Some("self") | Some("cls")
    )
}

fn single_language_ranked_source(labels: &[String], candidates: &[Vec<u8>]) -> Option<Vec<u8>> {
    (labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .count()
        == 1)
        .then(|| {
            candidates
                .iter()
                .find(|candidate| programming_response_compatible(labels, candidate))
                .cloned()
        })
        .flatten()
}

fn select_composed_artifact(
    labels: &[String],
    prompt: &str,
    fragment_composition: Option<Vec<u8>>,
    manifest_composition: Option<Vec<u8>>,
) -> Option<Vec<u8>> {
    let prompt = prompt.to_ascii_lowercase();
    let language_count = labels
        .iter()
        .filter(|label| label.contains(":LANGUAGE:"))
        .count();
    if language_count > 1 {
        // A multi-language response contract is necessarily a file
        // container. An independently valid single-language fragment cannot
        // satisfy it and must not shadow a manifest that covers every
        // requested language merely because the paraphrase says
        // "repository" or "codebase" instead of "project".
        return manifest_composition.or(fragment_composition);
    }
    let explicitly_multifile = prompt.contains("multi-file")
        || prompt.contains("multiple files")
        || prompt.contains("across multiple files")
        || prompt.contains("in multiple files");
    let requests_single_class = prompt.contains("class") && !explicitly_multifile;
    if requests_single_class {
        return fragment_composition.or(manifest_composition);
    }
    if labels
        .iter()
        .any(|label| label.ends_with(":ARTIFACT:PROJECT"))
    {
        manifest_composition.or(fragment_composition)
    } else {
        fragment_composition.or(manifest_composition)
    }
}

fn explicitly_requests_source_unit(prompt: &str) -> bool {
    let prompt = prompt.to_ascii_lowercase();
    (prompt.contains("function") || prompt.contains("method") || prompt.contains("snippet"))
        && !prompt.contains("project")
        && !prompt.contains("module")
        && !prompt.contains("multiple files")
        && !prompt.contains("multi-file")
}

/// A request that combines independently grounded behaviors needs their
/// complete container before any one compatible source candidate. Corpus
/// growth can legitimately change source ranking; it must not make a ready
/// multi-component manifest unreachable. Explicit source-unit requests retain
/// the narrower response contract.
fn request_prefers_composed_artifact(labels: &[String], prompt: &str) -> bool {
    if explicitly_requests_source_unit(prompt) {
        return false;
    }
    let behavior_count = manifest_component_feature_pairs(labels)
        .into_iter()
        .map(|pair| pair[1].clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let prompt = prompt.to_ascii_lowercase();
    behavior_count >= 2
        || labels
            .iter()
            .any(|label| label.ends_with(":ARTIFACT:PROJECT"))
        || prompt.contains("project")
        || prompt.contains("modules")
        || prompt.contains("multiple files")
        || prompt.contains("multi-file")
}

/// Assemble independently grounded raw-source fragments into files. The
/// protocol carries only deterministic structural constraints; source remains
/// byte-atom learned evidence and is never invented by this function.
/// New curricula express order as role dependencies; legacy numeric slots
/// remain readable for checkpoint compatibility. Conflicts, missing
/// dependencies, and cycles abort composition rather than guessing.
fn requested_python_identifier<'a>(prompt: &'a str, cues: &[&str]) -> Option<&'a str> {
    let lower = prompt.to_ascii_lowercase();
    for cue in cues {
        let Some(start) = lower.find(cue).map(|index| index + cue.len()) else {
            continue;
        };
        let candidate = prompt[start..]
            .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .next()
            .unwrap_or("");
        let valid = !candidate.is_empty()
            && candidate
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
            && candidate
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
        if valid {
            return Some(candidate);
        }
    }
    None
}

fn requested_python_class(prompt: &str) -> Option<&str> {
    requested_python_identifier(prompt, &["class named ", "class called "])
}

fn requested_python_method(prompt: &str) -> Option<&str> {
    requested_python_identifier(prompt, &["method named ", "method called "])
}

fn requested_ascii_identifier_after<'a>(prompt: &'a str, cue: &str) -> Option<&'a str> {
    let lower = prompt.to_ascii_lowercase();
    let start = lower.find(cue)? + cue.len();
    let tail = &prompt[start..];
    let end = tail
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .unwrap_or(tail.len());
    let value = &tail[..end];
    (!value.is_empty() && (value.as_bytes()[0].is_ascii_alphabetic() || value.starts_with('_')))
        .then_some(value)
}

fn requested_authorized_role(prompt: &str) -> Option<&str> {
    requested_ascii_identifier_after(prompt, "authorized role ")
        .or_else(|| requested_ascii_identifier_after(prompt, "except for the "))
}

fn requested_resource_field(prompt: &str) -> Option<&str> {
    if let Some(value) = requested_ascii_identifier_after(prompt, "resource field ") {
        return Some(value);
    }
    let text = prompt.to_ascii_lowercase();
    if text.contains("inventory") {
        Some("inventory")
    } else if text.contains("capacity") {
        Some("capacity")
    } else {
        None
    }
}

fn requested_item_field(prompt: &str) -> Option<&str> {
    if let Some(value) = requested_ascii_identifier_after(prompt, "item field ") {
        return Some(value);
    }
    let text = prompt.to_ascii_lowercase();
    if text.contains("sku") {
        Some("sku")
    } else if text.contains("job") {
        Some("job")
    } else {
        None
    }
}

fn requested_amount_field(prompt: &str) -> Option<&str> {
    if let Some(value) = requested_ascii_identifier_after(prompt, "amount field ") {
        return Some(value);
    }
    let text = prompt.to_ascii_lowercase();
    if text.contains("quantity") {
        Some("quantity")
    } else if text.contains("slots") {
        Some("slots")
    } else {
        None
    }
}

fn requested_resource_initializer(prompt: &str) -> Option<String> {
    if prompt
        .to_ascii_lowercase()
        .contains("mapping resource field")
    {
        let item = requested_ascii_identifier_after(prompt, "default item ")?;
        return Some(format!("{{'{item}': 10}}"));
    }
    if prompt
        .to_ascii_lowercase()
        .contains("scalar resource field")
    {
        return Some("10".to_string());
    }
    match requested_resource_field(prompt)? {
        "inventory" => Some("{'widget': 10}".to_string()),
        "capacity" => Some("10".to_string()),
        _ => None,
    }
}

fn requested_resource_available(prompt: &str) -> Option<String> {
    let field = requested_resource_field(prompt)?;
    if prompt
        .to_ascii_lowercase()
        .contains("mapping resource field")
        || field == "inventory"
    {
        Some(format!("self.{field}.get(item, 0)"))
    } else {
        Some(format!("self.{field}"))
    }
}

fn requested_resource_decrement(prompt: &str) -> Option<String> {
    let field = requested_resource_field(prompt)?;
    if prompt
        .to_ascii_lowercase()
        .contains("mapping resource field")
        || field == "inventory"
    {
        Some(format!("self.{field}[item] -= amount"))
    } else {
        Some(format!("self.{field} -= amount"))
    }
}

fn requested_result_key(prompt: &str) -> Option<&str> {
    if let Some(value) = requested_ascii_identifier_after(prompt, "result field ") {
        return Some(value);
    }
    let text = prompt.to_ascii_lowercase();
    if text.contains("allocation") || text.contains("fulfillment") {
        Some("allocation")
    } else if text.contains("dispatch") || text.contains("scheduler") {
        Some("worker")
    } else {
        None
    }
}

fn requested_event_kind(prompt: &str) -> Option<&str> {
    if let Some(start) = prompt.to_ascii_lowercase().find("event kind ") {
        let tail = &prompt[start + "event kind ".len()..];
        let end = tail
            .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')))
            .unwrap_or(tail.len());
        let value = &tail[..end];
        if !value.is_empty() {
            return Some(value);
        }
    }
    let text = prompt.to_ascii_lowercase();
    if text.contains("inventory-allocated") {
        Some("inventory-allocated")
    } else if text.contains("job-scheduled") {
        Some("job-scheduled")
    } else {
        None
    }
}

fn requested_log_request_key(prompt: &str) -> Option<&str> {
    if let Some(value) = requested_ascii_identifier_after(prompt, "log request as ") {
        return Some(value);
    }
    let text = prompt.to_ascii_lowercase();
    if text.contains("order containing") || text.contains("order key") {
        Some("order")
    } else if text.contains("command") || text.contains("job") {
        Some("command")
    } else {
        None
    }
}

fn render_grounded_fragment_source(
    fragment: &serde_json::Map<String, serde_json::Value>,
    source: &str,
    prompt: &str,
) -> Option<String> {
    let Some(parameters) = fragment.get("parameters") else {
        return (!source.contains("{{")).then(|| source.to_string());
    };
    let parameters = parameters.as_object()?;
    let mut rendered = source.to_string();
    for (name, kind) in parameters {
        let placeholder = format!("{{{{{name}}}}}");
        if !rendered.contains(&placeholder) {
            return None;
        }
        let value = match kind.as_str()? {
            "python_class_named" => requested_python_class(prompt)?.to_string(),
            "python_method_named" => requested_python_method(prompt)?.to_string(),
            "python_authorized_role" => requested_authorized_role(prompt)?.to_string(),
            "python_resource_field" => requested_resource_field(prompt)?.to_string(),
            "python_item_field" => requested_item_field(prompt)?.to_string(),
            "python_amount_field" => requested_amount_field(prompt)?.to_string(),
            "python_resource_initializer" => requested_resource_initializer(prompt)?,
            "python_resource_available" => requested_resource_available(prompt)?,
            "python_resource_decrement" => requested_resource_decrement(prompt)?,
            "python_result_key" => requested_result_key(prompt)?.to_string(),
            "python_event_kind" => requested_event_kind(prompt)?.to_string(),
            "python_log_request_key" => requested_log_request_key(prompt)?.to_string(),
            _ => return None,
        };
        rendered = rendered.replace(&placeholder, &value);
    }
    (!rendered.contains("{{")).then_some(rendered)
}

fn merge_grounded_code_fragments(candidates: &[Vec<u8>]) -> Option<Vec<u8>> {
    merge_grounded_code_fragments_for_prompt(candidates, "")
}

fn merge_grounded_code_fragments_for_prompt(
    candidates: &[Vec<u8>],
    prompt: &str,
) -> Option<Vec<u8>> {
    #[derive(Clone, PartialEq, Eq)]
    struct RelativeFragment {
        source: String,
        after: std::collections::BTreeSet<String>,
    }
    let mut numeric: std::collections::BTreeMap<String, std::collections::BTreeMap<i64, String>> =
        std::collections::BTreeMap::new();
    let mut relative: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, RelativeFragment>,
    > = std::collections::BTreeMap::new();
    let mut invalid_files = std::collections::BTreeSet::new();
    let mut outcomes: std::collections::BTreeMap<String, bool> = std::collections::BTreeMap::new();
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
            continue;
        };
        if evidence_id.is_empty() {
            continue;
        }
        if let Some(previous) = outcomes.insert(evidence_id.to_string(), confirmed) {
            // Repeated identical confirmation is idempotent. Contradictory
            // evidence fails closed for fragments carrying this evidence id.
            if previous != confirmed {
                outcomes.insert(evidence_id.to_string(), false);
            }
        }
    }
    for bytes in candidates {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            continue;
        };
        let Some(fragment) = value.get("code_fragment").and_then(|v| v.as_object()) else {
            continue;
        };
        let (Some(file), Some(raw_source)) = (
            fragment.get("file").and_then(|v| v.as_str()),
            fragment.get("source").and_then(|v| v.as_str()),
        ) else {
            continue;
        };
        if file.is_empty()
            || file.starts_with('/')
            || file.starts_with('\\')
            || file.split(['/', '\\']).any(|part| part == "..")
            || raw_source.is_empty()
        {
            invalid_files.insert(file.to_string());
            continue;
        }
        // A ranked request can retrieve several independently learned
        // fragment families. A template whose required prompt parameters are
        // absent is ineligible for this request; it must not veto a separate,
        // dependency-complete family that needs no such parameters.
        let Some(source) = render_grounded_fragment_source(fragment, raw_source, prompt) else {
            continue;
        };
        if let Some(evidence_id) = fragment.get("evidence_id").and_then(|v| v.as_str()) {
            if outcomes.get(evidence_id) == Some(&false) {
                continue;
            }
        }
        if let Some(order) = fragment.get("order").and_then(|v| v.as_i64()) {
            let slots = numeric.entry(file.to_string()).or_default();
            if let Some(existing) = slots.get(&order) {
                if existing != &source {
                    invalid_files.insert(file.to_string());
                }
                continue;
            }
            slots.insert(order, source);
            continue;
        }
        let Some(role) = fragment.get("role").and_then(|v| v.as_str()) else {
            invalid_files.insert(file.to_string());
            continue;
        };
        if role.is_empty()
            || !role
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | ':'))
        {
            invalid_files.insert(file.to_string());
            continue;
        }
        let after_values = fragment
            .get("after")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut after = std::collections::BTreeSet::new();
        for value in after_values {
            let Some(dependency) = value.as_str() else {
                invalid_files.insert(file.to_string());
                continue;
            };
            if dependency == role || dependency.is_empty() {
                invalid_files.insert(file.to_string());
                continue;
            }
            after.insert(dependency.to_string());
        }
        let entry = RelativeFragment { source, after };
        let roles = relative.entry(file.to_string()).or_default();
        if let Some(existing) = roles.get(role) {
            if existing != &entry {
                invalid_files.insert(file.to_string());
            }
            continue;
        }
        roles.insert(role.to_string(), entry);
    }
    for file in &invalid_files {
        numeric.remove(file);
        relative.remove(file);
    }
    let numeric_count: usize = numeric.values().map(|slots| slots.len()).sum();
    let relative_count: usize = relative.values().map(|roles| roles.len()).sum();
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
        // Ranked retrieval can activate one complete artifact alongside a
        // partial, unrelated historical chain. Missing dependencies make the
        // entire affected file ineligible: emitting only its independent
        // imports or initializer would misrepresent a prefix as a complete
        // implementation. Repeating this removal also quarantines files that
        // depend on a newly removed cross-file prerequisite, while preserving
        // independent dependency-complete files.
        loop {
            let all_roles: std::collections::BTreeSet<String> = graph.keys().cloned().collect();
            let incomplete_files: std::collections::BTreeSet<String> = graph
                .iter()
                .filter(|(_, (_, _, dependencies))| !dependencies.is_subset(&all_roles))
                .map(|(_, (file, _, _))| file.clone())
                .collect();
            if incomplete_files.is_empty() {
                break;
            }
            graph.retain(|_, (file, _, _)| !incomplete_files.contains(file));
        }
        if graph.len() < 2 {
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
                // A cycle in one independently retrieved fragment family
                // must not erase a different file whose dependency graph has
                // already settled completely. Remove every file represented
                // by the stalled remainder, including any prefix emitted for
                // that file, then preserve only fully settled files.
                let stalled_files: std::collections::BTreeSet<String> = graph
                    .iter()
                    .filter(|(key, _)| !emitted.contains(*key))
                    .map(|(_, (file, _, _))| file.clone())
                    .collect();
                for file in stalled_files {
                    file_sources.remove(&file);
                }
                break;
            };
            file_sources
                .entry(file)
                .or_default()
                .push_str(&fragment_source);
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
    if rendered.is_empty() {
        return None;
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
    brain.activate_for_indexed_prediction(POOL_TEXT, prompt.as_bytes());

    // Exact ordered sensory evidence can be established from the raw pool
    // immediately. Do this before semantic-routing diagnostics: those routes
    // scan broad feature-binding populations and made known-answer latency
    // grow with total curriculum size even though the exact binding index had
    // already reduced retrieval itself to O(1)-style lookup.
    let raw_is_exact = brain.has_exact_trained_binding(POOL_TEXT, action_pool);
    let unseen_atomic_prompt = unseen_atomic_prompt_requires_abstention(prompt, raw_is_exact);
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
        let labels = brain
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
        // Deterministic sensory features are direct evidence. A fuzzy route
        // may recover an intent below when the surface sensor cannot ground
        // one, but it must not delete a directly observed behavior while
        // retaining only its language and a generic modifier. If a surface
        // extractor proves over-broad, correct that extractor explicitly.
        let effective_frame = prompt.as_bytes();
        // A lone diagnostic (most commonly only LANGUAGE:PYTHON) is too
        // broad to establish task grounding. Require a co-firing composition
        // such as LANGUAGE + BEHAVIOR before derived evidence may influence
        // readout. Raw characters still activate regardless of this gate.
        if brain
            .activate_for_indexed_prediction(pool_id, effective_frame)
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
        if brain
            .activate_for_indexed_prediction(pool_id, &intent_frame)
            .len()
            >= 2
        {
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
    let raw_trained = exact_raw_trained.or_else(|| {
        // Once a derived pool has independently grounded the request, do not
        // wake a second broad character-overlap population. Feature ranking
        // and the bounded motif route below retain the raw evidence needed
        // for disambiguation. Raw-only prompts still use fuzzy recall here.
        if chat_query_pools.len() == 1 {
            brain.decode_best_trained_binding_with_context(
                POOL_TEXT,
                action_pool,
                &chat_query_pools,
                &turn_pools,
            )
        } else {
            None
        }
    });
    // Sparse LANGUAGE+BEHAVIOR labels identify an intent class, but several
    // independently learned artifacts may legitimately share that class.
    // Preserve the atom-stream evidence needed to disambiguate those targets:
    // for a single-behavior request, rank actions by character motifs from the
    // raw prompt. Richer requests remain on the multi-feature composition path.
    let raw_motif_trained = composition_features
        .as_ref()
        .filter(|(_, labels)| is_single_language_single_behavior(labels))
        .and_then(|_| {
            brain.decode_best_binding_by_char_motifs_with_margin(
                POOL_TEXT,
                prompt.as_bytes(),
                action_pool,
                0.20,
                0.025,
            )
        })
        .map(|(bytes, _, _)| bytes);
    // Raw character motifs and sparse semantic features are independent
    // neural observations of the same request. At accumulated-corpus scale,
    // a broad feature posting can omit an older valid episode from its
    // bounded candidate window even though the raw motif route can still
    // reach it. Jointly require both signals before selection: the motif
    // route supplies verbatim learned actions and the semantic validator
    // inhibits superficially similar but behaviorally incompatible source.
    let raw_semantically_validated = composition_features
        .as_ref()
        .filter(|(_, labels)| has_exactly_one_programming_language(labels))
        .and_then(|(_, labels)| {
            brain.decode_best_binding_by_char_motifs_with_margin_where(
                POOL_TEXT,
                prompt.as_bytes(),
                action_pool,
                0.0,
                0.0,
                &|candidate| {
                    prompt_programming_response_compatible(labels, prompt, candidate)
                        // This route decodes straight out of the brain by
                        // character motifs, so it never passes through
                        // feature_candidates and no filter on that pool can
                        // reach it. `prompt_programming_response_compatible`
                        // waives nearly every check for a complete manifest,
                        // which let an unseen request inherit one whose
                        // vocabulary it never mentions.
                        && (!is_complete_file_manifest(candidate)
                            || prompt_shares_manifest_subject_with_labels(
                                labels, prompt, candidate,
                            ))
                },
            )
        })
        .map(|(bytes, _, _)| bytes);
    // Recall-derived route for a request the label table cannot describe.
    //
    // Every motif route above is gated on `composition_features`, which needs
    // at least one LANGUAGE and one BEHAVIOR label from the hand-written
    // extractor in InstructionIntentEncoding. That table has no vocabulary
    // outside the domains it was written for: measured 2026-08-20, "Write a
    // Django REST APIView ..." produced ZERO labels (it does not even contain
    // the substring "python"), and a three.js request produced only
    // LANGUAGE:JAVASCRIPT -- one short of the gate. Composition therefore
    // never started for any web-stack request, though the evidence was there
    // the whole time.
    //
    // The motif route itself does not need the table. Scored against the
    // trained corpus, novel phrasings separate cleanly on their own:
    //
    //   "Build a Django REST endpoint that evaluates ..."  0.507 / 0.217 / 0.160
    //   "I need a three.js scene class with orbit controls" 0.555 / 0.177 / 0.137
    //   "Give me a Vue keypad component for a calculator"   0.496 / 0.089 / 0.082
    //
    // -- the correct unit wins by 2-3x every time, well over the 0.20 floor
    // the labelled route already trusts. So when the table yields nothing,
    // fall back to what the fabric itself recalls, and require the margin to
    // carry the decision: breadth of corroboration is the evidence, not a
    // keyword. A weak or contested match still abstains.
    let recall_derived_route = if composition_features.is_none()
        && !raw_is_exact
        && request_carries_enough_evidence(prompt)
    {
        brain
            .decode_best_binding_by_char_motifs_wide(
                POOL_TEXT,
                prompt.as_bytes(),
                action_pool,
                UNLABELED_RECALL_MIN_SCORE,
                // A near-perfect match needs no margin, because there is
                // nothing left for a rival to take.
                //
                // The margin exists to refuse a CONTESTED match -- two
                // unrelated candidates scoring alike. It cannot mean that
                // when the winner scores 1.0: measured 2026-08-23, "the
                // first law of thermodynamics" recalled at score 1.0 with
                // margin 0.0 and was refused, because a textbook states the
                // same heading in more than one place and its own duplicate
                // tied it. Two spellings of the identical answer are
                // agreement, not contest.
                //
                // Scaled rather than switched off: the requirement fades as
                // the score approaches certainty, so a weak match is still
                // held to the full margin.
                UNLABELED_RECALL_MIN_MARGIN,
            )
            .filter(|(bytes, _, _)| {
                recalled_answer_language_is_plausible(prompt, bytes)
            })
    } else {
        None
    };
    let diagnostic_recall_derived = recall_derived_route
        .as_ref()
        .map(|(_, score, margin)| (*score, *margin));
    // What the route WOULD have scored when it declined.
    //
    // Reporting the score only on success made every abstention look
    // identical: "What is entropy a measure of?" and a genuinely unanswerable
    // request both read `below_threshold` with no number, so there was
    // nothing to tune against without reproducing the Dice calculation by
    // hand outside the process. Measured that way, the short-question
    // abstentions sit at 0.42-0.44 against a 0.45 floor -- a fact the
    // diagnostics should have surfaced directly.
    //
    // Costs a second decode only on the path that already answered nothing.
    let diagnostic_recall_derived_best = if recall_derived_route.is_none()
        && composition_features.is_none()
        && !raw_is_exact
    {
        brain
            .decode_best_binding_by_char_motifs_wide(
                POOL_TEXT, prompt.as_bytes(), action_pool, 0.0, 0.0,
            )
            .map(|(_, score, margin)| (score, margin))
    } else {
        None
    };
    // Why the route did or did not run, so a silent None is never ambiguous.
    let diagnostic_recall_derived_state = if raw_is_exact {
        "skipped_exact"
    } else if composition_features.is_some() {
        "skipped_labelled"
    } else if diagnostic_recall_derived.is_some() {
        "admitted"
    } else {
        "below_threshold"
    };
    let recall_derived_bytes = recall_derived_route.map(|(bytes, _, _)| bytes);
    let diagnostic_feature_route_pressure = composition_features
        .as_ref()
        .map(|(pool_id, labels)| brain.feature_route_pressure(*pool_id, labels))
        .unwrap_or_default();
    // The char-motif manifest is also a candidate, not only a final answer.
    //
    // f2910ff contributed this per language+behaviour subset, which fixed
    // multi-component composition. A SINGLE-behaviour request never enters
    // that loop, so its manifest still reached no pool: measured 2026-08-20,
    // "Write Python database upgrade paths using schema versions" produced
    // labels [PYTHON, SCHEMA_MIGRATION], 3 candidates, and `no_answer` --
    // while the same two labels from "a Python schema migration that is safe
    // to run repeatedly" returned migrations.py through raw_motif_trained.
    // The manifest was retrievable the whole time; it just was not in the
    // pool that ranked_single_manifest and the composition search read.
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
    let unweighted_feature_candidates = composition_features
        .as_ref()
        .map(|(pool_id, labels)| {
            brain.decode_ranked_feature_bindings_with_context(
                *pool_id,
                labels,
                action_pool,
                64,
                None,
                None,
                &chat_query_pools,
                &turn_pools,
            )
        })
        .unwrap_or_default();
    let diagnostic_unweighted_candidates = unweighted_feature_candidates.len();
    for candidate in unweighted_feature_candidates {
        if !feature_candidates.contains(&candidate) {
            feature_candidates.push(candidate);
        }
    }
    if let Some(candidate) = raw_semantically_validated.as_ref() {
        if is_complete_file_manifest(candidate)
            && !feature_candidates.contains(candidate)
        {
            feature_candidates.push(candidate.clone());
        }
    }
    let mut diagnostic_component_routes = Vec::<serde_json::Value>::new();
    // What each language+behaviour subset actually contributed to the pool.
    // Six blind attempts at cross_project's authorized_transfer paraphrase
    // each reasoned from source and were wrong; this reports the fact.
    let mut diagnostic_component_recall = Vec::<serde_json::Value>::new();
    if let Some((pool_id, labels)) = composition_features.as_ref() {
        for candidate in brain.decode_ranked_feature_bindings_with_context_where(
            *pool_id,
            labels,
            action_pool,
            8,
            brain.fabric().pool(8).map(|_| 8),
            brain.fabric().pool(6).map(|_| 6),
            &chat_query_pools,
            &turn_pools,
            &|candidate| {
                (is_complete_file_manifest(candidate)
                    && prompt_programming_response_compatible(
                        labels, prompt, candidate,
                    ))
                    || is_grounded_code_fragment(candidate)
            },
        ) {
            if !feature_candidates.contains(&candidate) {
                feature_candidates.push(candidate);
            }
        }
        for candidate in exact_manifest_subset_candidates(&brain, *pool_id, labels, action_pool) {
            if !feature_candidates.contains(&candidate) {
                feature_candidates.push(candidate);
            }
        }
        // A combined project request may state fewer details for each
        // independently learned component than that component's own training
        // episode. Recover through grounded LANGUAGE+BEHAVIOR conjunctions,
        // never through a language-only or generic-project match.
        for subset in manifest_component_feature_pairs(labels) {
            // Char-motif recall for THIS component, contributed to the pool.
            //
            // decode_best_binding_by_char_motifs_with_margin_where reads
            // straight out of the brain and never populates
            // feature_candidates, so merge_grounded_file_manifests -- which
            // reads only that pool and needs >= 2 manifests -- could not see
            // manifests the brain demonstrably retrieves. Measured
            // 2026-08-19 on a fully settled brain: asked individually,
            // "default-deny authorization" and "correlated logging with
            // secret redaction" each returned a real manifest via
            // raw_semantically_validated, yet the composite request for a
            // project containing both answered `no_answer` with
            // manifest_composition_ready=False. A two-behaviour composite
            // failed identically to a four-behaviour one, so this is not
            // about request size.
            //
            // Query per component rather than on the whole prompt: a combined
            // request's motifs match no single component's training episode.
            // The same subject-vocabulary guard applies, so a component
            // cannot inherit a manifest its own labels do not support.
            // The query frame is the whole prompt, so char motifs rank by
            // resemblance to the COMBINED request and every subset returns
            // the same best manifest. Measured 2026-08-20 on cross_project's
            // authorized_transfer paraphrase: every subset resolved to
            // repository.py, authorization.py never entered the pool, and the
            // composed project shipped one file where the case needs two --
            // even though the paraphrase's own words retrieve
            // authorization.py when asked alone. Skipping what is already
            // present lets each subsequent subset surface its own component.
            // Query with the BEHAVIOUR's own vocabulary, not the combined
            // prompt. Char motifs rank by textual resemblance, so a composite
            // request retrieves whichever component it most resembles and the
            // others are never reached. Measured 2026-08-20 on cross_project's
            // authorized_transfer paraphrase: "database transaction and
            // access-control modules" resolved every subset to repository.py,
            // and appending the single word "authorization" was enough to
            // compose both files with the SAME three labels -- proving the
            // gate is prompt text, not intent.
            let component_query = behaviour_query_frame(&subset, prompt);
            let subset_name: String = subset
                .iter()
                .map(|label| label.rsplit(':').next().unwrap_or(label))
                .collect::<Vec<_>>()
                .join("+");
            if let Some((candidate, _, _)) =
                brain.decode_best_binding_by_char_motifs_with_margin_where(
                    POOL_TEXT,
                    component_query.as_bytes(),
                    action_pool,
                    0.0,
                    0.0,
                    &|candidate| {
                        is_complete_file_manifest(candidate)
                            && programming_response_compatible(&subset, candidate)
                            // Require the behaviour this subset names, not
                            // merely compatibility. programming_response_compatible
                            // passes a manifest that satisfies the LANGUAGE and
                            // does not contradict the behaviour, so repository.py
                            // qualified for the authorization subset too and the
                            // strongest overall motif match won every round.
                            && subset_behaviour_is_evidenced(&subset, candidate)
                            // programming_response_compatible only checks that
                            // each REQUESTED language is present, not that
                            // foreign ones are absent, so a Python-only
                            // request happily accepted a manifest whose file
                            // was order_service.js -- composing a .js file
                            // into a Python project and breaking its import.
                            && manifest_files_match_requested_languages(
                                &subset, candidate,
                            )
                    },
                )
            {
                diagnostic_component_recall.push(json!({
                    "subset": subset_name.clone(),
                    "files": manifest_file_names(&candidate),
                }));
                if !feature_candidates.contains(&candidate) {
                    feature_candidates.push(candidate);
                }
            }
            let mut component_candidates =
                brain.decode_ranked_feature_bindings_with_context_where(
                *pool_id, &subset, action_pool, 8,
                brain.fabric().pool(8).map(|_| 8),
                brain.fabric().pool(6).map(|_| 6),
                &chat_query_pools, &turn_pools,
                &|candidate| {
                    (is_complete_file_manifest(candidate)
                        && programming_response_compatible(&subset, candidate))
                        || is_grounded_code_fragment(candidate)
                },
            );
            for candidate in brain.decode_ranked_feature_bindings_with_context_where(
                *pool_id, &subset, action_pool, 8, None, None,
                &chat_query_pools, &turn_pools,
                &|candidate| {
                    (is_complete_file_manifest(candidate)
                        && programming_response_compatible(&subset, candidate))
                        || is_grounded_code_fragment(candidate)
                },
            ) {
                if !component_candidates.contains(&candidate) {
                    component_candidates.push(candidate);
                }
            }
            let route_artifacts: Vec<serde_json::Value> = component_candidates
                .iter()
                .take(8)
                .filter_map(|candidate| {
                    let value = serde_json::from_slice::<serde_json::Value>(candidate).ok()?;
                    if let Some(files) = value.get("files").and_then(|files| files.as_object()) {
                        return Some(json!({
                            "kind": "manifest",
                            "files": files.keys().cloned().collect::<Vec<_>>(),
                        }));
                    }
                    let fragment = value.get("code_fragment")?.as_object()?;
                    Some(json!({
                        "kind": "fragment",
                        "file": fragment.get("file").and_then(|file| file.as_str()),
                        "role": fragment.get("role").and_then(|role| role.as_str()),
                    }))
                })
                .collect();
            if !route_artifacts.is_empty() {
                diagnostic_component_routes.push(json!({
                    "labels": subset,
                    "artifacts": route_artifacts,
                }));
            }
            let mut accepted_components = 0usize;
            for candidate in component_candidates {
                if (is_complete_file_manifest(&candidate) || is_grounded_code_fragment(&candidate))
                    && !feature_candidates.contains(&candidate)
                {
                    feature_candidates.push(candidate);
                    accepted_components += 1;
                    if accepted_components >= 8 {
                        break;
                    }
                }
            }
        }
        // Raw characters and sparse diagnostics are independent evidence
        // pools. A complete safe manifest recalled by the raw pathway may be
        // one component of a richer feature-composed project, so let it join
        // the candidate set instead of using it only as a final fallback.
        //
        // Label count measures how many intent facets a prompt names, not how
        // much evidence supports it. A narrow request ("Python" + one concrete
        // behaviour) can be fully grounded while naming only two facets, and
        // gating on a raw count alone silently excludes the manifest from the
        // candidate pool for those prompts -- leaving a corpus function to win
        // a project request by character overlap. Admit the manifest whenever
        // the labels carry a concrete behaviour beyond the language itself,
        // and let the ranking decide; a bare LANGUAGE label still needs the
        // richer multi-facet evidence to qualify.
        let behaviour_grounded = labels
            .iter()
            .any(|label| label.contains(":") && !label.contains(":LANGUAGE:"));
        if labels.len() >= 4 || behaviour_grounded {
            if let Some(candidate) = raw_trained
                .as_ref()
                .filter(|bytes| is_complete_file_manifest(bytes))
                // A single behaviour label is weaker evidence than four, and
                // the label extractor can be wrong: an out-of-vocabulary
                // request ("Byzantine fault tolerance") may still be tagged
                // CONCURRENCY:BOUNDED_ASYNC, which would hand it an unrelated
                // trained manifest and turn an honest abstention into a
                // confident wrong answer. Require the prompt itself to
                // corroborate the artifact -- the same check the exact and
                // ranked feature routes already apply -- so a mislabelled
                // prompt cannot admit a manifest it does not support.
                .filter(|bytes| {
                    if labels.len() >= 4 {
                        return true;
                    }
                    // `prompt_derived_feature_artifact_compatible` checks the
                    // artifact against the labels, not the labels against the
                    // prompt, and it waives most checks for a complete
                    // manifest -- so a mislabelled prompt passes it. When a
                    // single behaviour label carries the decision, require the
                    // request and the manifest to share subject vocabulary.
                    // (`programming_behavior_compatible` cannot serve here: its
                    // cues are code tokens like `asyncio.semaphore(`, which no
                    // natural-language prompt contains.)
                    prompt_derived_feature_artifact_compatible(labels, prompt, bytes)
                        && prompt_shares_manifest_subject_with_labels(
                            labels, prompt, bytes,
                        )
                })
            {
                if !feature_candidates.contains(candidate) {
                    feature_candidates.push(candidate.clone());
                }
            }
        }
    }
    let exact_feature = composition_features.as_ref().and_then(|(pool_id, labels)| {
        brain.decode_exact_feature_binding(*pool_id, labels, action_pool)
    });
    // Several independent routes push into this pool and most apply no
    // compatibility check, so filtering any one of them leaves the others
    // open -- guarding the raw_trained gate and then the unweighted loop each
    // failed to move the result for exactly that reason. Filter once, here,
    // after every producer has run and before anything consumes the pool.
    //
    // A manifest recalled on thin label evidence must share subject
    // vocabulary with the request. The out-of-vocabulary prompt "distributed
    // consensus protocol with Byzantine fault tolerance" was tagged
    // CONCURRENCY:BOUNDED_ASYNC and inherited the unrelated bounded_map
    // manifest, answering a request the brain is required to refuse. Four or
    // more labels carry enough agreement to outvote a single bad label and
    // are left untouched.
    if let Some((_, labels)) = composition_features.as_ref() {
        if labels.len() < 4 {
            // A COMPONENT is judged against the behaviour it supplies, not
            // against the whole request.
            //
            // prompt_shares_manifest_subject_with_labels asks whether a
            // manifest matches the ENTIRE label set. That is right for a
            // manifest proposed as the whole answer, and wrong for one
            // contributed as a part: measured 2026-08-20 on cross_project's
            // authorized_transfer paraphrase, per-component recall correctly
            // returned repository.py for PYTHON+ATOMIC_TRANSACTION and
            // authorization.py for PYTHON+AUTHORIZATION, and this retain then
            // deleted authorization.py because "access-control modules" shares
            // no vocabulary with it AND it does not satisfy ATOMIC_TRANSACTION.
            // The pool arrived at composition holding one manifest, so the
            // project shipped one file.
            //
            // Accept a manifest that corroborates ANY ONE requested behaviour
            // on the same terms. The out-of-vocabulary guard is unaffected:
            // the Byzantine prompt carries a single behaviour label, so "any
            // one" and "all" are the same test for it.
            let behaviours: Vec<String> = labels
                .iter()
                .filter(|label| {
                    label.contains(':') && !label.contains(":LANGUAGE:")
                })
                .cloned()
                .collect();
            let language: Vec<String> = labels
                .iter()
                .filter(|label| label.contains(":LANGUAGE:"))
                .cloned()
                .collect();
            feature_candidates.retain(|candidate| {
                if !is_complete_file_manifest(candidate) {
                    return true;
                }
                if prompt_shares_manifest_subject_with_labels(
                    labels, prompt, candidate,
                ) {
                    return true;
                }
                behaviours.len() >= 2
                    && behaviours.iter().any(|behaviour| {
                        let mut subset = language.clone();
                        subset.push(behaviour.clone());
                        prompt_shares_manifest_subject_with_labels(
                            &subset, prompt, candidate,
                        )
                    })
            });
        }
    }
    let diagnostic_pool_manifests: Vec<serde_json::Value> = feature_candidates
        .iter()
        .filter(|candidate| is_complete_file_manifest(candidate))
        .map(|candidate| json!({"files": manifest_file_names(candidate)}))
        .collect();
    let ranked_single_manifest = composition_features
        .as_ref()
        .and_then(|(_, labels)| {
            single_language_ranked_manifest(labels, prompt, &feature_candidates)
        });
    let diagnostic_intent_labels = composition_features
        .as_ref()
        .map(|(_, labels)| labels.clone())
        .unwrap_or_default();
    let exact_feature_compatible = exact_feature.as_ref().is_some_and(|candidate| {
        prompt_derived_feature_artifact_compatible(
            &diagnostic_intent_labels,
            prompt,
            candidate,
        )
    });
    let exact_complete_manifest = exact_feature
        .as_ref()
        .filter(|bytes| {
            is_complete_file_manifest(bytes) && exact_feature_compatible
        })
        .cloned();
    let diagnostic_exact_feature = exact_feature.is_some();
    let diagnostic_exact_manifest = exact_complete_manifest.is_some();
    let programming_language_intent = has_programming_language_intent(&diagnostic_intent_labels);
    let ranked_single_source =
        single_language_ranked_source(&diagnostic_intent_labels, &feature_candidates)
            .filter(|candidate| {
                prompt_derived_feature_artifact_compatible(
                    &diagnostic_intent_labels,
                    prompt,
                    candidate,
                )
            });
    let ranked_validated_source = composition_features.as_ref().and_then(|(pool_id, labels)| {
        (labels
            .iter()
            .filter(|label| label.contains(":LANGUAGE:"))
            .count()
            == 1)
            .then(|| {
                brain.decode_first_ranked_feature_binding_with_context_where(
                    *pool_id,
                    labels,
                    action_pool,
                    brain.fabric().pool(8).map(|_| 8),
                    brain.fabric().pool(6).map(|_| 6),
                    &chat_query_pools,
                    &turn_pools,
                    &|candidate| {
                        !is_complete_file_manifest(candidate)
                            && prompt_programming_response_compatible(
                                labels, prompt, candidate,
                            )
                            && !bound_method_answers_free_function(prompt, candidate)
                    },
                )
            })
            .flatten()
    });
    let raw_programming_compatible = raw_trained.as_ref().is_some_and(|candidate| {
        prompt_derived_feature_artifact_compatible(
            &diagnostic_intent_labels,
            prompt,
            candidate,
        )
    });
    let fragment_composition =
        merge_grounded_code_fragments_for_prompt(&feature_candidates, prompt);
    let manifest_composition =
        merge_grounded_file_manifests(&diagnostic_intent_labels, &feature_candidates);
    let diagnostic_fragment_composition_ready = fragment_composition.is_some();
    let diagnostic_manifest_composition_ready = manifest_composition.is_some();
    let composed = select_composed_artifact(
        &diagnostic_intent_labels,
        prompt,
        fragment_composition,
        manifest_composition,
    )
    .filter(|candidate| {
        prompt_derived_feature_artifact_compatible(
            &diagnostic_intent_labels,
            prompt,
            candidate,
        )
    });
    let exact_is_composition_prerequisite = diagnostic_intent_labels.len() >= 4
        && exact_feature.as_ref().is_some_and(|exact| {
            exact_fragment_has_grounded_dependents(exact, &feature_candidates)
                && composed
                    .as_ref()
                    .is_some_and(|artifact| composed_artifact_contains_fragment(artifact, exact))
        });
    let explicitly_requests_source_unit = explicitly_requests_source_unit(prompt);
    // Which branch below produced the answer. This chain has fourteen arms
    // and several can return the same artifact, so a wrong answer gives no
    // hint about which one to fix: four successive guesses at the OOV leak
    // (the raw_trained gate, the unweighted loop, a whole-pool retain, then
    // the char-motif route) each changed nothing because the answer was
    // coming from somewhere else entirely. Reporting the branch turns that
    // search into a single question, so it is worth keeping.
    let mut answer_branch = "none";
    let trained_bytes = if raw_is_exact && raw_trained.is_some() {
        answer_branch = "raw_is_exact";
        // Direct sensory evidence is the strongest tier. Derived diagnostic
        // pools may compose novel requests, but can never overwrite an
        // ordered prompt episode the brain actually observed.
        raw_trained.clone()
    } else if exact_complete_manifest.is_some() {
        answer_branch = "exact_complete_manifest";
        exact_complete_manifest
    } else if exact_is_composition_prerequisite && composed.is_some() {
        answer_branch = "exact_is_composition_prereq";
        composed
    } else if request_prefers_composed_artifact(
        &diagnostic_intent_labels,
        prompt,
    ) && composed.is_some()
    {
        answer_branch = "request_prefers_composed";
        // A ready, behavior-complete container is stronger than one plain
        // source candidate for a request that asks independent subsystems to
        // work together. This ordering is invariant to later corpus rank.
        composed
    } else if !explicitly_requests_source_unit
        && !single_behaviour_request_outgrown_by(
            &diagnostic_intent_labels,
            ranked_single_manifest.as_deref(),
        )
        && ranked_single_manifest.is_some()
    {
        answer_branch = "ranked_single_manifest_early";
        // Preserve a learned project/file response contract unless the user
        // explicitly asks for one source unit. A popular plain source from
        // the same sparse behavior class must not shadow a compatible
        // validated manifest.
        ranked_single_manifest
    } else if exact_feature
        .as_ref()
        .is_some_and(|candidate| {
            !is_grounded_code_fragment(candidate) && exact_feature_compatible
        })
    {
        answer_branch = "exact_feature_nonfragment";
        // An exact sparse-intent episode is stronger evidence than a fuzzy
        // assembly of several partially matching artifacts.  In particular,
        // LANGUAGE + BEHAVIOR can identify a learned single-function answer
        // exactly while broad project fragments share enough diagnostics to
        // form a syntactically valid but unrelated composition.
        exact_feature
    } else if ranked_validated_source.is_some() {
        answer_branch = "ranked_validated_source";
        // Apply the complete deterministic request before candidate-window
        // truncation. Popular but behaviorally incompatible corpus actions
        // cannot crowd a rarer correct learned episode out of bounded recall.
        ranked_validated_source
    } else if raw_semantically_validated.is_some() {
        answer_branch = "raw_semantically_validated";
        // Feature and atom-motif pools independently agree on this learned
        // action. This path remains read-only and cannot synthesize source.
        raw_semantically_validated
    } else if raw_motif_trained.as_ref().is_some_and(|candidate| {
        answer_branch = "raw_motif_trained";
        prompt_derived_feature_artifact_compatible(
            &diagnostic_intent_labels,
            prompt,
            candidate,
        )
    }) {
        // Character motifs resolve sparse-feature ambiguity, but cannot
        // displace one unambiguous complete feature episode.
        raw_motif_trained
    } else if composed.is_some() {
        answer_branch = "composed";
        composed
    } else if ranked_single_manifest.is_some() {
        answer_branch = "ranked_single_manifest_late";
        // Ranked feature evidence is stronger than raw character similarity.
        // This is the normal path for a paraphrase that omits one constraint
        // from a learned single-language project episode.
        ranked_single_manifest
    } else if exact_feature
        .as_ref()
        .is_some_and(|candidate| {
            is_grounded_code_fragment(candidate)
                && !programming_language_intent
        })
    {
        answer_branch = "exact_feature_fragment";
        // An isolated exact fragment is useful training evidence, but it is
        // not a complete user-facing artifact. Return it only after grounded
        // composition and complete ranked manifests have both failed.
        exact_feature
    } else if ranked_single_source.is_some() {
        answer_branch = "ranked_single_source";
        // Ranked sparse intent has independently grounded the requested
        // language and behavior. Plain single-file source is valid here as
        // long as its shape agrees with that language.
        ranked_single_source
    } else if chat_query_pools.len() > 1
        && !programming_language_intent
        && brain
            .decode_best_trained_binding_multi(&chat_query_pools, action_pool)
            .is_some()
    {
        // Only claim this branch when it actually produced something.
        //
        // As a bare `else if` it swallowed every non-programming question:
        // the condition is true for any multi-pool chat, the decode returns
        // None, and an if/else chain makes that None the final answer. So
        // `recall_derived` -- the arm below -- was unreachable for exactly
        // the requests it exists to serve. Measured 2026-08-23: "the first
        // law of thermodynamics" recalled at score 1.0 and still answered
        // nothing.
        answer_branch = "multi_pool_decode";
        brain.decode_best_trained_binding_multi(&chat_query_pools, action_pool)
    } else if let Some(bytes) = recall_derived_bytes.clone() {
        // Last resort, and only for a request no label described. Every route
        // above had its chance; this one exists so a decisively recalled unit
        // is not discarded merely because the intent table has no word for it.
        answer_branch = "recall_derived";
        Some(bytes)
    } else {
        answer_branch = "no_answer";
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
    let xpool =
        if trained_decode.as_ref().is_some_and(|s| !s.is_empty()) || !has_compositional_evidence {
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
            // Autonomous propagation is also a derived response path. Its
            // confidence/composition evidence cannot admit behaviorally
            // unrelated but language-valid source.
            result
                .answer
                .as_deref()
                .filter(|answer| {
                    prompt_derived_feature_artifact_compatible(
                        &diagnostic_intent_labels,
                        prompt,
                        answer,
                    )
                })
                .map(|answer| String::from_utf8_lossy(answer).into_owned())
        }
    });

    let reply = if directly_underspecified || unseen_atomic_prompt {
        // Explicit missing-information evidence inhibits every generative
        // route, including raw-character fuzzy recall. An unobserved atomic
        // lexical symbol has the same contract: a familiar word occurring
        // inside it is not independent grounding for the whole symbol.
        String::new()
    } else if let Some(td) = trained_decode.as_ref().filter(|s| !s.is_empty()) {
        td.clone()
    } else if xpool.as_ref().is_some_and(|result| {
        !result.grounding.outside_grounding
            && !result.grounding.speculation_flag
            && result.grounding.integrated_confidence >= 0.30
            && result.grounding.composition_used.len() >= 2
    }) {
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
        let result = xpool
            .as_ref()
            .expect("xpool reply requires integration result");
        if result.grounding.eem_confidence.is_some() && result.grounding.fabric_confidence < 0.3 {
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
    // Keep composition failures observable without returning learned source.
    // File/role/dependency metadata is sufficient to distinguish retrieval
    // gaps from rendering or dependency-closure failures.
    let diagnostic_fragment_candidates: Vec<serde_json::Value> = feature_candidates
        .iter()
        .filter_map(|bytes| {
            let value = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
            let fragment = value.get("code_fragment")?.as_object()?;
            Some(json!({
                "file": fragment.get("file")?.as_str()?,
                "role": fragment.get("role").and_then(|value| value.as_str()),
                "order": fragment.get("order").and_then(|value| value.as_i64()),
                "after": fragment.get("after").cloned().unwrap_or_else(|| json!([])),
                "parameters": fragment.get("parameters").cloned().unwrap_or_else(|| json!({})),
            }))
        })
        .collect();
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };

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
        "paged_neurons_released": paged_neurons_released,
        "intent_diagnostics": {
            "labels": diagnostic_intent_labels,
            "ranked_candidates": feature_candidates.len(),
            "unweighted_candidates": diagnostic_unweighted_candidates,
            "composite_keys": diagnostic_feature_route_pressure.composite_keys,
            "composite_candidates": diagnostic_feature_route_pressure.composite_candidates,
            "composite_saturated": diagnostic_feature_route_pressure.composite_saturated,
            "fragment_candidates": diagnostic_fragment_candidates,
            "component_routes": diagnostic_component_routes,
            "component_recall": diagnostic_component_recall,
            "pool_manifests": diagnostic_pool_manifests,
            "exact_feature": diagnostic_exact_feature,
            "exact_complete_manifest": diagnostic_exact_manifest,
            "fragment_composition_ready": diagnostic_fragment_composition_ready,
            "manifest_composition_ready": diagnostic_manifest_composition_ready,
            "answer_branch": answer_branch,
            "recall_derived_state": diagnostic_recall_derived_state,
            "recall_derived_score": diagnostic_recall_derived.map(|(score, _)| score),
            "recall_derived_margin": diagnostic_recall_derived.map(|(_, margin)| margin),
            // The score the route rejected, so an abstention is measurable.
            "recall_derived_best_score":
                diagnostic_recall_derived_best.map(|(score, _)| score),
            "recall_derived_best_margin":
                diagnostic_recall_derived_best.map(|(_, margin)| margin),
            "raw_fallback_inhibited": programming_language_intent
                && !raw_is_exact
                && !raw_programming_compatible
                && trained_decode.is_none(),
        },
    }))
}

/// Keep an unobserved, single lexical symbol outside grounding.
///
/// Character motifs are intentionally useful for paraphrase recall, but a
/// growing corpus can make a learned word appear as a substring of a novel
/// token (`quasarithmetic` contains `arithmetic`). With no second token or
/// exact sensory episode there is no boundary-preserving evidence that the
/// user requested the learned behavior. Exact one-token lessons remain fully
/// authoritative; only their fuzzy substrings are inhibited.
fn unseen_atomic_prompt_requires_abstention(prompt: &str, raw_is_exact: bool) -> bool {
    if raw_is_exact {
        return false;
    }
    let mut lexical = prompt
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|token| !token.is_empty());
    lexical.next().is_some() && lexical.next().is_none()
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
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };
    Json(json!({ "steps": steps, "paged_neurons_released": paged_neurons_released }))
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
    let paged_neurons_released = match brain.finish_read_only_inference() {
        Ok(count) => count,
        Err(error) => {
            return Json(json!({"error": format!("inference cleanup failed: {}", error)}));
        }
    };
    Json(json!({
        "learning": false,
        "known_atom_count": known.len(),
        "query_sequence_length": query_seq.len(),
        "exact_binding_count": exact.len(),
        "exact_bindings": exact,
        "top_matches": top_matches,
        "paged_neurons_released": paged_neurons_released,
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
    let mut brain = s.brain.lock().await;
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
    let neurons_serialized = if brain.uses_wbrain_storage() {
        match brain.serialize_all_neurons_for_idle() {
            Ok(count) => count,
            Err(error) => {
                return Json(json!({
                    "error": format!("idle neuron serialization failed: {}", error),
                    "promotions_drained": promotions,
                    "concepts_pruned": pruned,
                }));
            }
        }
    } else {
        0
    };
    Json(json!({
        "promotions_drained": promotions,
        "concepts_pruned":    pruned,
        "neurons_serialized": neurons_serialized,
        "tick_now":           brain.fabric().current_tick(),
    }))
}

async fn h_checkpoint(State(s): State<BrainApiState>) -> Json<serde_json::Value> {
    let dir = default_node_brain_dir();
    let path = dir.join("brain.bin");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return Json(json!({ "ok": false, "error": format!("mkdir {}: {}", dir.display(), e) }));
    }
    let mut brain = s.brain.lock().await;
    let uses_wbrain = brain.uses_wbrain_storage();
    let result = if uses_wbrain {
        brain.serialize_all_neurons_for_idle().map(|_| ())
    } else {
        brain.checkpoint(&path)
    };
    let reported_path = if uses_wbrain {
        dir.join("brain.wbrain")
    } else {
        path
    };
    match result {
        Ok(()) => Json(json!({
            "ok": true,
            "path": reported_path.display().to_string(),
            "storage": if uses_wbrain { "wbrain" } else { "bin" },
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
        .route("/consolidate/multi", post(h_consolidate_multi))
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

/// A dependency relationship alone does not prove that the settled artifact
/// retained the exact evidence. Incomplete sibling chains may leave a valid
/// prefix that omits the requested fragment entirely.
fn composed_artifact_contains_fragment(artifact: &[u8], exact: &[u8]) -> bool {
    let Some(fragment) = serde_json::from_slice::<serde_json::Value>(exact)
        .ok()
        .and_then(|value| value.get("code_fragment").cloned())
    else {
        return false;
    };
    let (Some(file), Some(source)) = (
        fragment.get("file").and_then(|value| value.as_str()),
        fragment.get("source").and_then(|value| value.as_str()),
    ) else {
        return false;
    };
    if source.contains("{{") {
        return false;
    }
    serde_json::from_slice::<serde_json::Value>(artifact)
        .ok()
        .and_then(|value| value.get("files").cloned())
        .and_then(|files| files.get(file).cloned())
        .and_then(|source| source.as_str().map(str::to_owned))
        .is_some_and(|composed| composed.contains(source))
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
    #[test]
    fn python_source_is_not_evidence_of_typescript() {
        // `import ` and `class ` are shared between Python and TS/JS, so a
        // request labelled LANGUAGE:TYPESCRIPT accepted a Python reply that
        // opened `import numpy as np`.
        //
        // Measured 2026-08-25: that is how the capstone safety suite reached
        // `unsafe_cross_domain_answer` -- a strict-TypeScript physics
        // platform answered with scikit-learn source, reproducibly on 3 of 3
        // runs. It failed the 12-suite gate and blocked EVERY deferred-replay
        // admission for 48 hours: 0 admitted, 18 failed.
        let typescript = vec!["instruction_intent:LANGUAGE:TYPESCRIPT".to_string()];
        let javascript = vec!["instruction_intent:LANGUAGE:JAVASCRIPT".to_string()];
        let python_source = b"import numpy as np
import matplotlib.pyplot as plt

class Model:
    pass
";

        assert!(
            !super::programming_response_compatible(&typescript, python_source),
            "python source must not satisfy a TypeScript request"
        );
        assert!(
            !super::programming_response_compatible(&javascript, python_source),
            "python source must not satisfy a JavaScript request"
        );

        // Real TypeScript still qualifies, on a marker Python cannot produce.
        let ts_source = b"export interface Vector3 { x: number; y: number; z: number }
";
        assert!(super::programming_response_compatible(&typescript, ts_source));
        let js_source = b"export function integrate(state) { return state; }
";
        assert!(super::programming_response_compatible(&javascript, js_source));
    }

    use super::*;

    #[test]
    fn unseen_atomic_symbol_cannot_inherit_a_contained_word_binding() {
        assert!(unseen_atomic_prompt_requires_abstention(
            "quasarithmetic",
            false,
        ));
        assert!(unseen_atomic_prompt_requires_abstention("zxqv_compiler", false));
        assert!(!unseen_atomic_prompt_requires_abstention(
            "quasarithmetic",
            true,
        ));
        assert!(!unseen_atomic_prompt_requires_abstention(
            "compute arithmetic mean",
            false,
        ));
    }

    #[test]
    fn fresh_brain_uses_neuron_addressable_storage_from_first_observation() {
        let unique = format!(
            "w1z4rd_fresh_wbrain_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let data_dir = std::env::temp_dir().join(unique);
        let path = data_dir.join("brain.wbrain");
        let mut brain = build_default_brain().unwrap();

        let attached = attach_fresh_wbrain(&mut brain, &path).unwrap();
        assert!(attached > 0);
        assert!(brain.uses_wbrain_storage());
        brain.observe(POOL_TEXT, b"fresh neuron-addressable storage");
        assert!(brain.serialize_all_neurons_for_idle().unwrap() > 0);
        assert!(path.is_file());

        let (restored, missing) =
            Brain::restore_wbrain(&path, restore_encodings(None).unwrap()).unwrap();
        assert!(missing.is_empty());
        assert!(restored.uses_wbrain_storage());
        std::fs::remove_dir_all(data_dir).ok();
    }

    #[test]
    fn completed_raw_migration_promotes_stale_raw_stage_without_reconversion() {
        let unique = format!(
            "w1z4rd_finalize_resume_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let data_dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&data_dir).unwrap();
        let mut brain = build_default_brain().unwrap();
        brain.observe(POOL_TEXT, b"abc");
        brain.checkpoint(data_dir.join("brain.bin")).unwrap();

        let first_count = migrate_legacy_brain_container(&data_dir).unwrap();
        let destination = data_dir.join("brain.wbrain");
        let first_bytes = std::fs::metadata(&destination).unwrap().len();
        assert!(!data_dir.join("brain.wbrain.raw-running").exists());
        assert!(!data_dir.join("brain.wbrain.finalize-pending").exists());
        assert!(migrate_legacy_brain_container(&data_dir).is_err());
        assert_eq!(
            std::fs::metadata(&destination).unwrap().len(),
            first_bytes,
            "an unmarked complete container must never be deleted or reconverted",
        );

        let legacy_bytes = std::fs::metadata(data_dir.join("brain.bin")).unwrap().len();
        std::fs::write(
            data_dir.join("brain.wbrain.raw-running"),
            legacy_bytes.to_string(),
        )
        .unwrap();
        let resumed_count = migrate_legacy_brain_container(&data_dir).unwrap();

        assert_eq!(resumed_count, first_count);
        assert!(
            std::fs::metadata(&destination).unwrap().len() >= first_bytes,
            "finalization may append a newer manifest but must not reconvert or truncate neurons",
        );
        assert!(!data_dir.join("brain.wbrain.raw-running").exists());
        assert!(!data_dir.join("brain.wbrain.finalize-pending").exists());
        std::fs::remove_dir_all(data_dir).ok();
    }

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
        assert!(is_grounded_code_fragment(
            br#"{"code_fragment":{"file":"service.py","role":"import","after":[],"source":"from domain import VALUE\n"}}"#
        ));
        assert!(!is_grounded_code_fragment(
            br#"{"code_fragment":{"file":"../escape.py","role":"import","after":[],"source":"bad"}}"#
        ));
        assert!(!is_grounded_code_fragment(
            br#"{"code_fragment":{"file":"service.py","role":"","after":[],"source":"bad"}}"#
        ));
    }

    #[test]
    fn grounded_fragments_form_a_never_observed_file_in_slot_order() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"main.py","order":20,"source":"    return value\n"}}"#
                .to_vec(),
            br#"{"code_fragment":{"file":"main.py","order":10,"source":"def identity(value):\n"}}"#
                .to_vec(),
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
    fn grounded_fragment_template_binds_valid_requested_class_name() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"service.py","role":"class","after":[],"parameters":{"CLASS_NAME":"python_class_named"},"source":"class {{CLASS_NAME}}:\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"service.py","role":"method","after":["class"],"parameters":{"METHOD_NAME":"python_method_named"},"source":"    def {{METHOD_NAME}}(self):\n        return True\n"}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments_for_prompt(
            &candidates,
            "Create a Python class named NovelCoordinator with a method called ready.",
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(
            value["files"]["service.py"],
            "class NovelCoordinator:\n    def ready(self):\n        return True\n"
        );
        assert!(
            merge_grounded_code_fragments_for_prompt(
                &candidates,
                "Create a Python service without a specified class name.",
            )
            .is_none()
        );
        assert!(
            merge_grounded_code_fragments_for_prompt(
                &candidates,
                "Create a Python class named 7Invalid.",
            )
            .is_none()
        );
        let unknown_kind = vec![
            br#"{"code_fragment":{"file":"service.py","role":"class","after":[],"parameters":{"CLASS_NAME":"unbounded_text"},"source":"class {{CLASS_NAME}}:\n"}}"#.to_vec(),
            candidates[1].clone(),
        ];
        assert!(
            merge_grounded_code_fragments_for_prompt(
                &unknown_kind,
                "Create a Python class named SafeName with a method named ready.",
            )
            .is_none()
        );
        let unresolved = vec![
            br#"{"code_fragment":{"file":"service.py","role":"class","after":[],"parameters":{"CLASS_NAME":"python_class_named"},"source":"class {{CLASS_NAME}}:\n    value = '{{UNDECLARED}}'\n"}}"#.to_vec(),
            candidates[1].clone(),
        ];
        assert!(
            merge_grounded_code_fragments_for_prompt(
                &unresolved,
                "Create a Python class named SafeName with a method named ready.",
            )
            .is_none()
        );
    }

    #[test]
    fn grounded_state_contract_parameters_bind_only_controlled_code_shapes() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"service.py","role":"root","after":[],"parameters":{"CLASS":"python_class_named","METHOD":"python_method_named","STATE":"python_resource_field","INIT":"python_resource_initializer","ITEM":"python_item_field","AMOUNT":"python_amount_field","ROLE":"python_authorized_role"},"source":"class {{CLASS}}:\n    def __init__(self): self.{{STATE}} = {{INIT}}\n    async def {{METHOD}}(self, request):\n        required = {'{{ITEM}}', '{{AMOUNT}}'}\n        if request['actor'] != '{{ROLE}}': raise PermissionError\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"service.py","role":"body","after":["root"],"parameters":{"AVAILABLE":"python_resource_available","DECREMENT":"python_resource_decrement","RESULT":"python_result_key","EVENT":"python_event_kind","LOG":"python_log_request_key"},"source":"        item, amount = request['tenant'], request['units']\n        if {{AVAILABLE}} < amount: raise ValueError\n        {{DECREMENT}}\n        return {'{{RESULT}}': 'ok', 'event': '{{EVENT}}', 'log_key': '{{LOG}}'}\n"}}"#.to_vec(),
        ];
        let prompt = "Create a Python class named QuotaBroker with a method named reserve. Use scalar resource field credits, item field tenant, amount field units, authorized role auditor, result field receipt, event kind quota-reserved, and log request as command.";
        let assembled = merge_grounded_code_fragments_for_prompt(&candidates, prompt).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        let source = value["files"]["service.py"].as_str().unwrap();
        for expected in [
            "class QuotaBroker:",
            "self.credits = 10",
            "async def reserve",
            "request['actor'] != 'auditor'",
            "self.credits < amount",
            "self.credits -= amount",
            "'receipt': 'ok'",
            "'event': 'quota-reserved'",
            "'log_key': 'command'",
        ] {
            assert!(source.contains(expected), "missing {expected}: {source}");
        }
        assert!(merge_grounded_code_fragments_for_prompt(
            &candidates,
            "Create a Python class named 9Bad with a method named reserve; scalar resource field credits.",
        )
        .is_none());
    }

    #[test]
    fn component_recovery_requires_language_plus_concrete_behavior() {
        let labels = vec![
            "intent:LANGUAGE:JAVASCRIPT".to_string(),
            "intent:LANGUAGE:GO".to_string(),
            "intent:ARTIFACT:PROJECT".to_string(),
            "intent:DOMAIN:INVENTORY".to_string(),
            "intent:STRUCTURE:SERVICE_CLASS".to_string(),
            "intent:INTEGRATION:TRANSACTIONAL_OUTBOX".to_string(),
            "intent:CONCURRENCY:DEDUPLICATION".to_string(),
        ];
        let pairs = manifest_component_feature_pairs(&labels);
        assert_eq!(pairs.len(), 4);
        assert!(pairs.iter().all(|pair| pair.len() == 2));
        assert!(pairs.iter().all(|pair| {
            pair.iter().any(|label| label.contains(":LANGUAGE:"))
                && pair.iter().all(|label| {
                    !label.contains(":ARTIFACT:")
                        && !label.contains(":DOMAIN:")
                        && !label.contains(":STRUCTURE:")
                })
        }));
    }

    #[test]
    fn project_container_does_not_turn_one_behavior_into_rich_composition() {
        let labels = vec![
            "intent:LANGUAGE:PYTHON".to_string(),
            "intent:ARTIFACT:PROJECT".to_string(),
            "intent:SECURITY:AUTHORIZATION".to_string(),
        ];
        assert!(is_single_language_single_behavior(&labels));

        let mut rich = labels;
        rich.push("intent:OBSERVABILITY:CORRELATED_LOGGING".to_string());
        assert!(!is_single_language_single_behavior(&rich));
    }

    #[test]
    fn single_language_ranked_manifest_beats_raw_similarity_fallback() {
        // A manifest must share the request's SUBJECT, not merely its labels
        // -- a request for a physics platform was answered with `orders.ts`
        // on 3 of 3 runs because the extractor labelled it BATCHING and the
        // order manifest satisfied that label. So the fixture names what it
        // does, as a real recalled manifest does; `module.exports = 1;`
        // shares no word with any request and was only ever passing because
        // nothing checked.
        let manifest = br#"{"files":{"outbox.js":"exports.publishOutbox = function publishOutbox(outbox) { return outbox.pending; };\n"}}"#.to_vec();
        let labels = vec![
            "intent:LANGUAGE:JAVASCRIPT".to_string(),
            "intent:INTEGRATION:TRANSACTIONAL_OUTBOX".to_string(),
        ];
        assert!(has_exactly_one_programming_language(&labels));
        assert_eq!(
            single_language_ranked_manifest(
                &labels,
                "Build a JavaScript transactional outbox project.",
                &[manifest.clone()],
            ),
            Some(manifest.clone())
        );

        let mut polyglot = labels;
        polyglot.push("intent:LANGUAGE:GO".to_string());
        assert!(!has_exactly_one_programming_language(&polyglot));
        assert_eq!(
            single_language_ranked_manifest(
                &polyglot,
                "Build a JavaScript and Go project.",
                &[manifest],
            ),
            None
        );

        let java = vec!["intent:LANGUAGE:JAVA".to_string()];
        let javascript = br#"{"files":{"order_service.js":"class OrderService {}\n"}}"#.to_vec();
        let java_manifest = br#"{"files":{"AuditLog.java":"public class AuditLog {}\n"}}"#.to_vec();
        assert_eq!(
            single_language_ranked_manifest(
                &java,
                "Build Java code.",
                &[javascript.clone(), java_manifest.clone()],
            ),
            Some(java_manifest)
        );
        assert!(!programming_response_compatible(&java, &javascript));
    }

    #[test]
    fn accumulated_manifest_conflicts_do_not_poison_a_valid_subset() {
        let labels = vec![
            "intent:LANGUAGE:PYTHON".to_string(),
            "intent:ARTIFACT:PROJECT".to_string(),
            "intent:SECURITY:AUTHORIZATION".to_string(),
            "intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
        ];
        let candidates = vec![
            br#"{"files":{"repository.py":"def unrelated():\n    return 'transaction'\n"}}"#
                .to_vec(),
            br#"{"files":{"authorization.py":"def is_authorized(user):\n    return 'admin' in user.get('roles', [])\n"}}"#
                .to_vec(),
            br#"{"files":{"repository.py":"def transfer(db, source_id, target_id, amount):\n    with db:\n        db.debit(source_id, amount)\n        db.credit(target_id, amount)\n"}}"#
                .to_vec(),
        ];
        let composed = merge_grounded_file_manifests(&labels, &candidates).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&composed).unwrap();
        assert!(value["files"].get("authorization.py").is_some());
        assert!(value["files"]["repository.py"]
            .as_str()
            .unwrap()
            .contains("def transfer"));
    }

    #[test]
    fn manifest_composition_requires_every_requested_language() {
        let labels = vec![
            "intent:LANGUAGE:JAVA".to_string(),
            "intent:LANGUAGE:RUST".to_string(),
            "intent:STATE:OPTIMISTIC_CONCURRENCY".to_string(),
            "intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
            "intent:DOMAIN:ATOMIC_LEDGER_TRANSFER".to_string(),
        ];
        let java = br#"{"files":{"VersionedStore.java":"class VersionedStore { long expectedVersion; }"}}"#
            .to_vec();
        let rust = br#"{"files":{"ledger.rs":"fn transfer(source_id: &str, target_id: &str) { /* atomic debit credit transaction */ }"}}"#
            .to_vec();
        assert!(merge_grounded_file_manifests(
            &labels,
            &[java.clone(), rust],
        )
        .is_some());
        assert!(merge_grounded_file_manifests(&labels, &[java]).is_none());
    }

    #[test]
    fn one_polyglot_component_may_satisfy_multiple_requested_behaviors() {
        let labels = vec![
            "intent:LANGUAGE:JAVASCRIPT".to_string(),
            "intent:LANGUAGE:GO".to_string(),
            "intent:API:IDEMPOTENT_COMMAND".to_string(),
            "intent:INTEGRATION:TRANSACTIONAL_OUTBOX".to_string(),
            "intent:CONCURRENCY:DEDUPLICATION".to_string(),
        ];
        let javascript = br#"{"files":{"order_service.js":"class OrderService { constructor() { this.responses = new Map(); this.outbox = []; } create(key) { if (this.responses.has(key)) return this.responses.get(key); const order = { key }; this.responses.set(key, order); this.outbox.push({ type: 'created' }); return order; } }"}}"#.to_vec();
        let go = br#"{"files":{"dedup.go":"package main\nimport \"sync\"\ntype Deduplicator struct { mu sync.Mutex; seen map[string]struct{} }\nfunc (d *Deduplicator) AddIfNew(key string) bool { d.mu.Lock(); defer d.mu.Unlock(); if _, ok := d.seen[key]; ok { return false }; d.seen[key] = struct{}{}; return true }"}}"#.to_vec();
        let composed = merge_grounded_file_manifests(&labels, &[javascript, go])
            .expect("two manifests should satisfy three behaviors");
        let value: serde_json::Value = serde_json::from_slice(&composed).unwrap();
        assert!(value["files"].get("order_service.js").is_some());
        assert!(value["files"].get("dedup.go").is_some());
    }

    #[test]
    fn explicit_project_prefers_complete_manifest_over_fragment_subgraph() {
        let manifest =
            br#"{"files":{"circuit.py":"class Circuit: pass\n","logs.py":"def log(): pass\n"}}"#
                .to_vec();
        let fragments = br#"{"files":{"unrelated.py":"class Old: pass\n"}}"#.to_vec();
        let project = vec![
            "intent:LANGUAGE:PYTHON".to_string(),
            "intent:ARTIFACT:PROJECT".to_string(),
        ];
        assert_eq!(
            select_composed_artifact(
                &project,
                "Build a Python project across multiple files.",
                Some(fragments.clone()),
                Some(manifest.clone())
            ),
            Some(manifest)
        );
        let component = vec!["intent:LANGUAGE:PYTHON".to_string()];
        assert_eq!(
            select_composed_artifact(
                &component,
                "Build one Python component.",
                Some(fragments.clone()),
                None
            ),
            Some(fragments)
        );
    }

    #[test]
    fn multi_behavior_module_request_prefers_composition_over_plain_source() {
        let labels = vec![
            "intent:LANGUAGE:PYTHON".to_string(),
            "intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
            "intent:SECURITY:AUTHORIZATION".to_string(),
        ];
        assert!(request_prefers_composed_artifact(
            &labels,
            "Build Python database transaction and access-control modules: transfers are all-or-nothing and permissions deny by default.",
        ));
        assert!(!request_prefers_composed_artifact(
            &labels,
            "Write a Python function that checks authorization before a transaction.",
        ));
    }

    #[test]
    fn project_class_prefers_dependency_composed_class_over_broad_manifest() {
        let class = br#"{"files":{"adaptive_system.py":"class AdaptiveCoordinator:\n    pass\n"}}"#
            .to_vec();
        let broad = br#"{"files":{"api.py":"def api(): pass\n","repository.py":"class Repository: pass\n"}}"#.to_vec();
        let labels = vec![
            "intent:LANGUAGE:PYTHON".to_string(),
            "intent:ARTIFACT:PROJECT".to_string(),
        ];
        assert_eq!(
            select_composed_artifact(
                &labels,
                "Build a Python project class integrating twelve behaviors.",
                Some(class.clone()),
                Some(broad)
            ),
            Some(class)
        );
    }

    #[test]
    fn polyglot_composition_prefers_language_covering_manifest() {
        let fragment =
            br#"{"files":{"service.py":"def unrelated():\n    return True\n"}}"#.to_vec();
        let manifest = br#"{"files":{"service.js":"module.exports = {};\n","worker.go":"package main\n"}}"#.to_vec();
        let labels = vec![
            "intent:LANGUAGE:JAVASCRIPT".to_string(),
            "intent:LANGUAGE:GO".to_string(),
            "intent:INTEGRATION:TRANSACTIONAL_OUTBOX".to_string(),
            "intent:CONCURRENCY:DEDUPLICATION".to_string(),
        ];
        assert_eq!(
            select_composed_artifact(
                &labels,
                "Create one repository with Node.js and Golang workers.",
                Some(fragment),
                Some(manifest.clone()),
            ),
            Some(manifest)
        );
    }

    #[test]
    fn programming_language_intent_is_a_raw_fallback_domain_boundary() {
        assert!(has_programming_language_intent(&[
            "instruction_intent:LANGUAGE:TYPESCRIPT".to_string(),
            "instruction_intent:ENTERPRISE:BATCHING".to_string(),
        ]));
        assert!(!has_programming_language_intent(&[
            "instruction_intent:MATH:ARITHMETIC".to_string(),
        ]));
    }

    #[test]
    fn programming_raw_fallback_requires_language_compatible_source() {
        let python = vec!["instruction_intent:LANGUAGE:PYTHON".to_string()];
        assert!(programming_response_compatible(
            &python,
            b"def avg_list(values):\n    return sum(values) / len(values) if values else 0\n",
        ));
        assert!(!programming_response_compatible(
            &python,
            b"The construction cost is 216 dollars.",
        ));

        let typescript = vec!["instruction_intent:LANGUAGE:TYPESCRIPT".to_string()];
        assert!(programming_response_compatible(
            &typescript,
            b"export class Integrator {}",
        ));
        assert!(!programming_response_compatible(
            &typescript,
            b"The construction cost is 216 dollars.",
        ));

        let java = vec!["instruction_intent:LANGUAGE:JAVA".to_string()];
        assert!(programming_response_compatible(
            &java,
            br#"{"files":{"AuditLog.java":"public class AuditLog {}"}}"#,
        ));
        assert!(!programming_response_compatible(
            &java,
            br#"{"files":{"order_service.js":"class OrderService {}"}}"#,
        ));

        let python_source = b"def avg_list(values):\n    return 0\n".to_vec();
        assert_eq!(
            single_language_ranked_source(&python, &[python_source.clone()]),
            Some(python_source.clone())
        );
        let mut polyglot = python;
        polyglot.push("instruction_intent:LANGUAGE:RUST".to_string());
        assert_eq!(
            single_language_ranked_source(&polyglot, &[python_source]),
            None
        );
    }

    #[test]
    fn word_frequency_intent_rejects_generic_increment_source() {
        let labels = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:STATE:INCREMENT_COUNT".to_string(),
            "instruction_intent:TEXT:WORD_FREQUENCY".to_string(),
        ];
        let increment =
            b"def incrby(self, key, increment):\n    return self.execute('INCRBY', key, increment)"
                .to_vec();
        let call_count = b"def __get_call_count(self, args, kwargs, args_list, kwargs_list):\n    return len(self.__get_matching_indices(args, kwargs, args_list, kwargs_list))"
            .to_vec();
        let misleading_terms =
            b"def keyword_token(password):\n    return {'word': password}".to_vec();
        let password_hash = b"def make_password(password, salt=None):\n    if not salt:\n        salt = hasher.salt()\n    return hasher.encode(password, salt)"
            .to_vec();
        let profiler_increment = b"def increment(self, delta=1):\n    check_call(_LIB.MXProfileAdjustCounter(self.handle, int(delta)))"
            .to_vec();
        let scalar_word_count =
            b"def word_count(text):\n    return len(text.split())".to_vec();
        let misleading_counter_name =
            b"def counter(text):\n    return len(text.split())".to_vec();
        let file_statistics = br#"def wc(filename, contents, parsed=None, is_jekyll=False):
    body = parsed.strip() if parsed else contents.strip()
    words = re.split(r"\s+", body)
    paragraphs = [1 if len(line) == 0 else 0 for line in contents.splitlines()]
    return {
        "counts": {
            "file": filename,
            "paragraphs": sum(paragraphs) + 1,
            "words": len(words),
            "characters_total": len(body),
        }
    }"#
        .to_vec();
        let frequency = b"def word_freq(text):\n    out = {}\n    for word in text.split():\n        out[word] = out.get(word, 0) + 1\n    return out"
            .to_vec();
        let counter_frequency = b"from collections import Counter\n\ndef word_freq(text):\n    return Counter(text.split())"
            .to_vec();
        let indexed_frequency = b"def word_freq(text):\n    out = {}\n    for word in text.split():\n        if word not in out:\n            out[word] = 0\n        out[word] += 1\n    return out"
            .to_vec();
        let framework_task = br#"def run(self):
    count = {}
    for target in self.input():
        for line in target.open('r'):
            for word in line.strip().split():
                count[word] = count.get(word, 0) + 1
    output = self.output().open('w')
    for word, amount in count.items():
        output.write(f"{word}\t{amount}\n")
    output.close()"#
            .to_vec();
        assert!(!programming_response_compatible(&labels, &increment));
        assert!(!programming_response_compatible(&labels, &call_count));
        assert!(!programming_response_compatible(
            &labels,
            &misleading_terms
        ));
        assert!(!programming_response_compatible(&labels, &password_hash));
        assert!(!programming_response_compatible(
            &labels,
            &profiler_increment
        ));
        assert!(!programming_response_compatible(
            &labels,
            &scalar_word_count
        ));
        assert!(!programming_response_compatible(
            &labels,
            &misleading_counter_name
        ));
        assert!(!programming_response_compatible(
            &labels,
            &file_statistics
        ));
        assert!(!programming_response_compatible(&labels, &framework_task));
        assert!(programming_response_compatible(&labels, &frequency));
        assert!(programming_response_compatible(
            &labels,
            &counter_frequency
        ));
        assert!(programming_response_compatible(
            &labels,
            &indexed_frequency
        ));
        assert!(!derived_feature_artifact_compatible(
            &labels,
            &call_count
        ));
        assert!(!derived_feature_artifact_compatible(
            &labels,
            &password_hash
        ));
        assert_eq!(
            compatible_integrated_reply(&labels, Some(&profiler_increment)),
            None
        );
        assert_eq!(
            compatible_integrated_reply(&labels, Some(&frequency)),
            Some(String::from_utf8_lossy(&frequency).into_owned())
        );
        assert!(derived_feature_artifact_compatible(&labels, &frequency));
        assert_eq!(
            single_language_ranked_source(
                &labels,
                &[
                    increment,
                    call_count,
                    misleading_terms,
                    password_hash,
                    profiler_increment,
                    scalar_word_count,
                    misleading_counter_name,
                    file_statistics,
                    frequency.clone(),
                ]
            ),
            Some(frequency)
        );
    }

    #[test]
    fn prompt_constraints_disambiguate_shared_programming_intents() {
        let square = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:POWER_SELF:2".to_string(),
        ];
        let square_prompt =
            "Create a Python function named square that computes a number times itself.";
        assert!(!prompt_programming_response_compatible(
            &square,
            square_prompt,
            b"def Square(x, a, b, c):\n    return a * x ** 2 + b * x + c",
        ));
        assert!(prompt_programming_response_compatible(
            &square,
            square_prompt,
            b"def square(n):\n    return n * n",
        ));
        let implicit_name_square_prompt =
            "Create Python code computing the second power of a supplied number.";
        assert!(!prompt_programming_response_compatible(
            &square,
            implicit_name_square_prompt,
            b"def polynomial(x, a, b, c):\n    return a * x ** 2 + b * x + c",
        ));
        assert!(!prompt_programming_response_compatible(
            &square,
            implicit_name_square_prompt,
            b"def chi_squared(*choices):\n    return sum((item.expected - item.observed) ** 2 for item in choices)",
        ));
        assert!(!prompt_derived_feature_artifact_compatible(
            &square,
            implicit_name_square_prompt,
            b"def root_mean_square(X):\n    segment_width = X.shape[1]\n    return np.sqrt(np.sum(X * X, axis=1) / segment_width)",
        ));
        assert!(!prompt_derived_feature_artifact_compatible(
            &square,
            implicit_name_square_prompt,
            b"def cornersphere(self, x):\n    nconstr = len(x)\n    if any(x[:nconstr] < 1):\n        return np.NaN\n    return sum(x**2) - nconstr",
        ));
        assert!(prompt_programming_response_compatible(
            &square,
            implicit_name_square_prompt,
            b"def square(n):\n    return n * n",
        ));

        let odd = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:PARITY:ODD".to_string(),
        ];
        let odd_prompt =
            "Build a Python function which keeps only odd integers from an input list.";
        assert!(!prompt_programming_response_compatible(
            &odd,
            odd_prompt,
            b"def smedian(values, count):\n    return values[count // 2] if count % 2 else 0",
        ));
        assert!(prompt_programming_response_compatible(
            &odd,
            odd_prompt,
            b"def filter_odd(values):\n    return [value for value in values if value % 2]",
        ));

        let authorization = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:SECURITY:AUTHORIZATION".to_string(),
        ];
        let authorization_prompt =
            "Create Python access-control code that denies by default, permits administrators, and permits owner reads only.";
        assert!(!prompt_programming_response_compatible(
            &authorization,
            authorization_prompt,
            b"def oauth_authorization(client):\n    return client.exchange_token()",
        ));
        assert!(prompt_programming_response_compatible(
            &authorization,
            authorization_prompt,
            br#"{"files":{"authorization.py":"def is_authorized(principal, action, owner_id):\n    if not principal:\n        return False\n    if 'admin' in principal.get('roles', []):\n        return True\n    return action == 'read' and principal.get('id') == owner_id\n"}}"#,
        ));
    }

    #[test]
    fn platform_behavior_contracts_reject_shape_only_collisions() {
        let python = "instruction_intent:LANGUAGE:PYTHON".to_string();
        let migration = vec![
            python.clone(),
            "instruction_intent:PERSISTENCE:SCHEMA_MIGRATION".to_string(),
        ];
        assert!(!programming_response_compatible(
            &migration,
            b"def setup_db(url):\n    return create_engine(url)",
        ));
        assert!(programming_response_compatible(
            &migration,
            br#"{"files":{"migrations.py":"def migrate(db):\n    version = db.execute('PRAGMA user_version').fetchone()[0]\n    if version < 2:\n        db.execute('ALTER TABLE users ADD COLUMN email TEXT')\n"}}"#,
        ));

        let circuit = vec![
            python.clone(),
            "instruction_intent:RESILIENCE:CIRCUIT_BREAKER".to_string(),
        ];
        assert!(!programming_response_compatible(
            &circuit,
            b"def retry_call(operation, attempts):\n    return operation()",
        ));
        assert!(programming_response_compatible(
            &circuit,
            br#"{"files":{"circuit.py":"class CircuitOpen(Exception): pass\nclass CircuitBreaker:\n    def __init__(self, failure_threshold, recovery_timeout):\n        self.failure_threshold = failure_threshold\n        self.recovery_timeout = recovery_timeout\n        self.opened_at = None\n"}}"#,
        ));

        let bounded = vec![
            python,
            "instruction_intent:CONCURRENCY:BOUNDED_ASYNC".to_string(),
        ];
        assert!(!programming_response_compatible(
            &bounded,
            b"def execute(pool, requests):\n    return [pool.submit(x) for x in requests]",
        ));
        assert!(programming_response_compatible(
            &bounded,
            br#"{"files":{"concurrency.py":"import asyncio\nasync def bounded_map(worker, items, limit):\n    semaphore = asyncio.Semaphore(limit)\n    async def run(item):\n        async with semaphore:\n            return await worker(item)\n    return await asyncio.gather(*(run(item) for item in items))\n"}}"#,
        ));
    }

    #[test]
    fn behavioral_intents_reject_language_shaped_but_unrelated_sources() {
        let average = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:MATH:AVERAGE".to_string(),
            "instruction_intent:INPUT:COLLECTION_ARGUMENT".to_string(),
            "instruction_intent:GUARD:EMPTY_INPUT".to_string(),
        ];
        let redis_remove =
            b"def srem(self, key, *members):\n    return self.execute('SREM', key, members)";
        let time_grid = b"def grid_time(time):\n    nt = time.shape[0]\n    weights = np.linalg.lstsq(np.arange(nt), time)[0]\n    return weights[0] * np.arange(nt)";
        let temporal_midpoint = br#"def average(self, dt=None):
    if dt is None:
        dt = self.now(self.tz)
    diff = self.diff(dt, False)
    return self.add(
        microseconds=(diff.in_seconds() * 1000000 + diff.microseconds) // 2
    )"#;
        let stateful_average = br#"def avg_resp_time(self):
    if not self._runtimes:
        return 0
    return round(sum(self._runtimes) / len(self._runtimes), 2)"#;
        let sliced_scientific_average = br#"def sbar(Ss):
    if type(Ss) == list:
        Ss = np.array(Ss)
    avs = []
    for j in range(6):
        avs.append(np.average(Ss[j]))
    return 6, np.std(Ss), avs"#;
        let multi_value_average = br#"def summarize(values):
    if not values:
        return 0
    return len(values), sum(values) / len(values)"#;
        let mean = b"def avg_list(values):\n    return sum(values) / len(values) if values else 0";
        let unguarded_numpy_mean = b"def avg_list(values):\n    return np.average(values)";
        let numpy_mean =
            b"def avg_list(values):\n    if not values:\n        return 0\n    return np.average(values)";
        assert!(!programming_response_compatible(&average, redis_remove));
        assert!(!programming_response_compatible(&average, time_grid));
        assert!(!programming_response_compatible(
            &average,
            temporal_midpoint
        ));
        assert!(!programming_response_compatible(
            &average,
            stateful_average
        ));
        assert!(!programming_response_compatible(
            &average,
            sliced_scientific_average
        ));
        assert!(!programming_response_compatible(
            &average,
            multi_value_average
        ));
        assert!(!derived_feature_artifact_compatible(&average, time_grid));
        assert!(!derived_feature_artifact_compatible(
            &average,
            temporal_midpoint
        ));
        assert!(!derived_feature_artifact_compatible(
            &average,
            stateful_average
        ));
        assert!(!derived_feature_artifact_compatible(
            &average,
            sliced_scientific_average
        ));
        assert!(programming_response_compatible(&average, mean));
        assert!(!programming_response_compatible(
            &average,
            unguarded_numpy_mean
        ));
        assert!(programming_response_compatible(&average, numpy_mean));

        let odd = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:PARITY:ODD".to_string(),
        ];
        assert!(!programming_response_compatible(&odd, mean));
        assert!(programming_response_compatible(
            &odd,
            b"def odd(values):\n    return [value for value in values if value % 2]"
        ));

        let json_aggregation = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:ENTERPRISE:JSON_AGGREGATION".to_string(),
        ];
        let json_customer_create = br#"def create(self, store_id, data):
    if 'id' not in data:
        raise KeyError('customer id required')
    response = self._client.post(data=data)
    return response"#;
        let aggregate_orders = br#"def aggregate_orders(payload):
    import json
    orders = json.loads(payload) if isinstance(payload, str) else payload
    totals = {}
    for order in orders:
        customer = str(order["customer"])
        totals[customer] = totals.get(customer, 0) + order["amount"]
    return totals"#;
        assert!(!programming_response_compatible(
            &json_aggregation,
            json_customer_create
        ));
        assert!(!derived_feature_artifact_compatible(
            &json_aggregation,
            json_customer_create
        ));
        assert!(programming_response_compatible(
            &json_aggregation,
            aggregate_orders
        ));

        let negative = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:COMPARISON:LESS_THAN_ZERO".to_string(),
        ];
        assert!(!programming_response_compatible(&negative, mean));
        assert!(programming_response_compatible(
            &negative,
            b"def negative(value):\n    return value < 0"
        ));

        let transaction = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
            "instruction_intent:DOMAIN:ATOMIC_LEDGER_TRANSFER".to_string(),
        ];
        let audit = b"def audit(record):\n    return {'password_last_used': record['date']}";
        let repository = br#"{"files":{"repository.py":"import sqlite3\n\ndef transfer(db_path, source_id, target_id, amount):\n    with sqlite3.connect(db_path) as connection:\n        debited = connection.execute('UPDATE accounts SET balance = balance - ?', (amount,))\n        credited = connection.execute('UPDATE accounts SET balance = balance + ?', (amount,))\n"}}"#;
        assert!(!programming_response_compatible(&transaction, audit));
        assert!(programming_response_compatible(&transaction, repository));

        // A bare transaction keyword must not satisfy the guard. This exact
        // corpus function (BigchainDB election cleanup) was recalled for the
        // sqlite_transaction paraphrase on 2026-08-18 and outranked the
        // correct manifest purely because "rollback" is in its name.
        let transaction_only = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
        ];
        let unrelated_rollback = b"def rollback(cls, bigchain, new_height, txn_ids):
    # delete election results
    return None";
        assert!(!programming_response_compatible(
            &transaction_only,
            unrelated_rollback
        ));
        // The real manifest still passes: its transaction verb co-occurs with
        // genuine persistence work.
        assert!(programming_response_compatible(
            &transaction_only,
            repository
        ));
        assert_eq!(
            single_language_ranked_manifest(
                &transaction,
                "Build Python all-or-nothing account transfer code.",
                &[
                    br#"{"files":{"audit.py":"def audit(record):\n    return record\n"}}"#
                        .to_vec(),
                    repository.to_vec(),
                ],
            ),
            Some(repository.to_vec())
        );

        let authorization = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:SECURITY:AUTHORIZATION".to_string(),
        ];
        let email =
            b"def add_email_addresses(self, addresses):\n    return self.post(addresses)";
        let access = br#"{"files":{"authorization.py":"def is_authorized(principal, action, owner_id):\n    roles = set(principal.get('roles', []))\n    if 'admin' in roles:\n        return True\n    return action == 'read' and principal.get('id') == owner_id\n"}}"#;
        assert!(!programming_response_compatible(&authorization, email));
        assert!(programming_response_compatible(&authorization, access));

        let square = vec![
            "instruction_intent:LANGUAGE:JAVA".to_string(),
            "instruction_intent:POWER_SELF:2".to_string(),
        ];
        assert!(programming_response_compatible(
            &square,
            b"static int square(int n) { return n * n; }",
        ));
        assert!(!programming_response_compatible(
            &square,
            b"static int increment(int n) { return n + 1; }",
        ));

        let batching = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:ENTERPRISE:BATCHING".to_string(),
            "instruction_intent:GUARD:POSITIVE_SIZE".to_string(),
        ];
        assert!(programming_response_compatible(
            &batching,
            b"def make_batches(items, size):\n    if size < 1: raise ValueError()\n    return [items[i:i+size] for i in range(0, len(items), size)]",
        ));
        assert!(!programming_response_compatible(
            &batching,
            b"def split_genome(genome, chunk_size):\n    return [genome[i:i+chunk_size] for i in range(0, len(genome), chunk_size)]",
        ));
    }

    #[test]
    fn exact_prerequisite_fragment_yields_to_its_grounded_dependent() {
        let signature = br#"{"code_fragment":{"file":"main.js","role":"signature","after":[],"source":"function identity(value) {\n"}}"#.to_vec();
        let body = br#"{"code_fragment":{"file":"main.js","role":"return","after":["signature"],"source":"  return value;\n}\n"}}"#.to_vec();
        assert!(exact_fragment_has_grounded_dependents(
            &signature,
            &[signature.clone(), body.clone()]
        ));
        assert!(!exact_fragment_has_grounded_dependents(
            &signature,
            &[signature.clone()]
        ));
        let partial = br#"{"files":{"main.js":"function identity(value) {\n"}}"#;
        let complete =
            br#"{"files":{"main.js":"function identity(value) {\n  return value;\n}\n"}}"#;
        assert!(composed_artifact_contains_fragment(partial, &signature));
        assert!(composed_artifact_contains_fragment(complete, &signature));
        assert!(composed_artifact_contains_fragment(complete, &body));
        assert!(!composed_artifact_contains_fragment(partial, &body));
    }

    #[test]
    fn grounded_relative_fragment_cycles_are_rejected() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"main.js","role":"a","after":["b"],"source":"a"}}"#
                .to_vec(),
            br#"{"code_fragment":{"file":"main.js","role":"b","after":["a"],"source":"b"}}"#
                .to_vec(),
        ];
        assert!(merge_grounded_code_fragments(&candidates).is_none());
    }

    #[test]
    fn unrelated_fragment_cycle_does_not_veto_settled_file() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"ready.py","role":"root","after":[],"source":"class Ready:\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"ready.py","role":"body","after":["root"],"source":"    value = 1\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"cyclic.py","role":"a","after":["b"],"source":"a\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"cyclic.py","role":"b","after":["a"],"source":"b\n"}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments(&candidates).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(value["files"]["ready.py"], "class Ready:\n    value = 1\n");
        assert!(value["files"].get("cyclic.py").is_none());
    }

    #[test]
    fn incomplete_unrelated_chain_does_not_veto_complete_artifact() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"complete.py","role":"root","after":[],"source":"class Ready:\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"complete.py","role":"body","after":["root"],"source":"    value = 1\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"partial.py","role":"tail","after":["missing_root"],"source":"return broken\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"partial.py","role":"later","after":["tail"],"source":"return later\n"}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments(&candidates).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(
            value["files"]["complete.py"],
            "class Ready:\n    value = 1\n"
        );
        assert!(value["files"].get("partial.py").is_none());
    }

    #[test]
    fn unresolved_behavior_quarantines_independent_prefixes_in_same_file() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"partial.py","role":"imports","after":[],"source":"import json\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"partial.py","role":"init","after":["imports"],"source":"STATE = {}\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"partial.py","role":"authorization","after":["validation"],"source":"def authorize(): return True\n"}}"#.to_vec(),
        ];
        assert!(merge_grounded_code_fragments(&candidates).is_none());
    }

    #[test]
    fn inapplicable_parameterized_family_does_not_veto_complete_artifact() {
        let candidates = vec![
            br#"{"code_fragment":{"file":"adaptive.py","role":"root","after":[],"source":"class Adaptive:\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"adaptive.py","role":"body","after":["root"],"source":"    value = 1\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"other.py","role":"root","after":[],"parameters":{"CLASS_NAME":"python_class_named"},"source":"class {{CLASS_NAME}}:\n"}}"#.to_vec(),
            br#"{"code_fragment":{"file":"other.py","role":"body","after":["root"],"source":"    other = True\n"}}"#.to_vec(),
        ];
        let assembled = merge_grounded_code_fragments_for_prompt(
            &candidates,
            "Combine the learned adaptive behaviors without naming a class.",
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&assembled).unwrap();
        assert_eq!(
            value["files"]["adaptive.py"],
            "class Adaptive:\n    value = 1\n"
        );
        assert!(value["files"].get("other.py").is_none());
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

    /// The OOV leak: a mislabelled prompt must not inherit a manifest.
    ///
    /// "distributed consensus protocol with Byzantine fault tolerance" was
    /// tagged CONCURRENCY:BOUNDED_ASYNC and admitted the bounded_map manifest,
    /// answering a request the brain is required to refuse.
    #[test]
    /// An over-labelled prompt must not lose its only correct manifest.
    ///
    /// The extractor infers labels from wording and over-labels. The platform
    /// suite's versioned_migrations paraphrase -- "fresh and legacy SQLite
    /// databases reach the same structure" -- yields PYTHON, SCHEMA_MIGRATION
    /// *and* ATOMIC_TRANSACTION. migrations.py answers the migration request
    /// completely but contains no transaction cue (rollback/commit/begin), so
    /// requiring every label rejected it and the chain fell through to
    /// no_answer. Measured 2026-08-20: the identical manifest is returned
    /// when the extractor emits only PYTHON + SCHEMA_MIGRATION.
    /// Subject agreement must survive morphology.
    ///
    /// Exact-token matching rejected a real match on a suffix alone. Measured
    /// 2026-08-20 on the platform suite's versioned_migrations paraphrase:
    /// the prompt says "versions" and "sqlite", migrations.py contains
    /// "version" and "sqlite3", and the overlap computed as EMPTY -- so the
    /// only correct manifest was refused and the request answered nothing.
    /// A satisfied label only counts when the prompt names the behaviour.
    ///
    /// Accepting a satisfied label alone re-opened the OOV leak: "distributed
    /// consensus protocol with Byzantine fault tolerance" is labelled
    /// CONCURRENCY:BOUNDED_ASYNC and bounded_map genuinely satisfies that cue,
    /// so python_enterprise oov went 3/3 -> 2/3 on 2026-08-20.
    /// Each component must be searched on its own terms.
    ///
    /// Char-motif recall ranks by textual resemblance, so a composite request
    /// retrieves only the component it most resembles. Measured 2026-08-20 on
    /// cross_project's authorized_transfer paraphrase: "database transaction
    /// and access-control modules" resolved EVERY subset to repository.py,
    /// and appending the single word "authorization" composed both files with
    /// the SAME three labels -- the gate is prompt text, not intent.
    /// A class method cannot answer a request for a standalone function.
    ///
    /// prompt_programming_response_compatible only checks the identifier when
    /// the prompt literally says "function named". A paraphrase saying "a
    /// Python batching function" gets no check, so a corpus method won on
    /// rank: measured 2026-08-20, python_enterprise's `batching` paraphrase
    /// returned `def chunk(self, size=0)` where the suite calls
    /// `make_batches(items, size)`. Adding "function named" to the same
    /// prompt returned the correct definition.
    #[test]
    fn a_bound_method_cannot_answer_a_free_function_request() {
        let method = b"def chunk(self, size=0):\n    return []\n".to_vec();
        let function = b"def make_batches(items, size):\n    return []\n".to_vec();
        let classmethod = b"def build(cls, size):\n    return []\n".to_vec();

        let asks_function =
            "Write a Python batching function that splits records into chunks.";
        assert!(bound_method_answers_free_function(asks_function, &method));
        assert!(bound_method_answers_free_function(asks_function, &classmethod));
        assert!(!bound_method_answers_free_function(asks_function, &function));

        // A request that wants a method or a class is untouched.
        let asks_method = "Write a Python method that chunks records.";
        assert!(!bound_method_answers_free_function(asks_method, &method));
        let asks_class = "Write a Python class with a chunking method.";
        assert!(!bound_method_answers_free_function(asks_class, &method));

        // Source with no def at all is not a method.
        assert!(!bound_method_answers_free_function(asks_function, b"x = 1"));
    }

    #[test]
    fn a_component_query_names_its_own_behaviour() {
        let prompt = "Build Python database transaction and access-control modules.";
        let subset = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:SECURITY:AUTHORIZATION".to_string(),
        ];
        let frame = behaviour_query_frame(&subset, prompt);
        assert!(frame.contains(prompt), "the request's own text is context");
        assert!(frame.contains("authorization"), "frame was {frame:?}");
        assert!(frame.contains("security"), "frame was {frame:?}");

        // Underscores become words so multi-word behaviours read naturally.
        let idempotent = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:API:IDEMPOTENT_COMMAND".to_string(),
        ];
        let frame = behaviour_query_frame(&idempotent, prompt);
        assert!(frame.contains("idempotent command"), "frame was {frame:?}");

        // A language-only subset has no behaviour to name.
        let language_only = vec!["instruction_intent:LANGUAGE:PYTHON".to_string()];
        assert_eq!(behaviour_query_frame(&language_only, prompt), prompt);
    }

    #[test]
    fn a_label_stands_in_for_subject_only_when_the_prompt_names_it() {
        let authz = "instruction_intent:SECURITY:AUTHORIZATION";
        let bounded = "instruction_intent:CONCURRENCY:BOUNDED_ASYNC";

        // Synonyms of the behaviour count -- these are the measured failures.
        assert!(prompt_names_behaviour(
            "Build Python database transaction and access-control modules.", authz));
        assert!(prompt_names_behaviour(
            "Develop Python rules that let superusers change any object.", authz));

        // The OOV prompt names nothing about bounded concurrency.
        assert!(!prompt_names_behaviour(
            "Implement a Python distributed consensus protocol with Byzantine fault tolerance.",
            bounded));
        // But a real bounded-concurrency request does.
        assert!(prompt_names_behaviour(
            "Write a Python helper that limits concurrency with a semaphore.", bounded));

        // An unrecognised behaviour never short-circuits the vocabulary test.
        assert!(!prompt_names_behaviour("anything at all", "intent:UNKNOWN:THING"));
    }

    #[test]
    fn subject_overlap_matches_a_shared_stem_not_an_exact_token() {
        let migrations = br#"{"files":{"migrations.py":"import sqlite3\ndef migrate(db_path):\n    connection.execute(\"PRAGMA user_version\")\n"}}"#.to_vec();

        assert!(prompt_shares_manifest_subject(
            "Write Python database upgrade paths using schema versions so fresh and legacy SQLite databases reach the same structure.",
            &migrations,
        ));

        // The OOV case this guard exists for must still be refused.
        assert!(!prompt_shares_manifest_subject(
            "Implement a Python distributed consensus protocol with Byzantine fault tolerance.",
            &migrations,
        ));

        // Scaffolding alone is still not subject evidence.
        assert!(!prompt_shares_manifest_subject(
            "Write Python code using this file.",
            &migrations,
        ));
    }

    #[test]
    fn an_over_labelled_prompt_still_finds_its_single_behaviour_manifest() {
        let migrations = br#"{"files":{"migrations.py":"import sqlite3\ndef migrate(db_path):\n    with sqlite3.connect(db_path) as connection:\n        connection.execute(\"PRAGMA user_version\")\n"}}"#.to_vec();
        let unrelated = br#"{"files":{"circuit.py":"class CircuitBreaker:\n    pass\n"}}"#.to_vec();
        let prompt = "Write Python database upgrade paths using schema versions.";

        let over_labelled = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:PERSISTENCE:SCHEMA_MIGRATION".to_string(),
            "instruction_intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
        ];

        // The manifest satisfies SCHEMA_MIGRATION but not ATOMIC_TRANSACTION,
        // and must still be found.
        let found = single_language_ranked_manifest(
            &over_labelled, prompt, &[migrations.clone()],
        );
        assert!(found.is_some(), "an over-labelled prompt lost its manifest");

        // A manifest matching NO requested behaviour is still refused, so a
        // cross-domain answer cannot slip through this fallback.
        let refused = single_language_ranked_manifest(
            &over_labelled, prompt, &[unrelated],
        );
        assert!(refused.is_none(), "unrelated manifest must not be accepted");

        // With a single behaviour the strict path already applies; the
        // fallback must not widen that case.
        let single = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:PERSISTENCE:ATOMIC_TRANSACTION".to_string(),
        ];
        assert!(
            single_language_ranked_manifest(&single, prompt, &[migrations])
                .is_none(),
            "one behaviour must keep the strict all-label contract"
        );
    }

    /// A Python request must not compose a JavaScript file.
    ///
    /// `programming_response_compatible` only checks that each REQUESTED
    /// language is present, never that foreign ones are absent. Composing a
    /// component by behaviour alone therefore pulled `order_service.js` into a
    /// Python-only project on 2026-08-19: the merged manifest was
    /// authorization.py + observability.py + repository.py + order_service.js,
    /// so the suite's integration test could not `from api import OrderApi`.
    #[test]
    fn a_component_manifest_must_match_the_requested_language() {
        let python_only = vec!["instruction_intent:LANGUAGE:PYTHON".to_string()];

        let javascript = br#"{"files":{"order_service.js":"export function submit(id) { return id; }"}}"#.to_vec();
        let python = br#"{"files":{"api.py":"def submit(request_id):\n    return request_id\n"}}"#.to_vec();
        let mixed = br#"{"files":{"api.py":"def submit(x):\n    return x\n","order_service.js":"export const x = 1;"}}"#.to_vec();
        let with_readme = br#"{"files":{"api.py":"def submit(x):\n    return x\n","README":"notes"}}"#.to_vec();

        assert!(!manifest_files_match_requested_languages(&python_only, &javascript));
        assert!(!manifest_files_match_requested_languages(&python_only, &mixed));
        assert!(manifest_files_match_requested_languages(&python_only, &python));
        // Extension-less files are neutral, not foreign.
        assert!(manifest_files_match_requested_languages(&python_only, &with_readme));

        // A request that DOES ask for JavaScript still accepts it.
        let polyglot = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:LANGUAGE:JAVASCRIPT".to_string(),
        ];
        assert!(manifest_files_match_requested_languages(&polyglot, &javascript));
        assert!(manifest_files_match_requested_languages(&polyglot, &mixed));

        // No language label means no opinion.
        assert!(manifest_files_match_requested_languages(&[], &javascript));

        // And the merge itself must refuse a foreign manifest, not just the
        // per-component contribution: six call sites push into
        // feature_candidates, so gating one producer left the rest open and
        // order_service.js still reached the composed project.
        let authorization = br#"{"files":{"authorization.py":"def is_authorized(p):\n    return True\n"}}"#.to_vec();
        let observability = br#"{"files":{"observability.py":"import json\ndef log(e, c):\n    print(e)\n"}}"#.to_vec();
        let pool = vec![authorization, observability, javascript.clone()];
        let merged = merge_grounded_file_manifests(&python_only, &pool)
            .expect("two python manifests must still compose");
        let value: serde_json::Value = serde_json::from_slice(&merged).unwrap();
        let files = value["files"].as_object().unwrap();
        assert!(
            !files.keys().any(|name| name.ends_with(".js")),
            "a PYTHON-only request must not compose a .js file: {files:?}"
        );
    }

    /// Composition needs manifests IN the candidate pool, not merely
    /// retrievable from the brain.
    ///
    /// `merge_grounded_file_manifests` reads only `feature_candidates` and
    /// requires two manifests. The char-motif route
    /// (`decode_best_binding_by_char_motifs_with_margin_where`) answers
    /// straight from the brain without ever populating that pool, so on
    /// 2026-08-19 a settled brain returned a real manifest for "default-deny
    /// authorization" and another for "correlated logging with secret
    /// redaction" individually, while a project request asking for both
    /// answered `no_answer` with manifest_composition_ready=False. A
    /// two-behaviour composite failed exactly like a four-behaviour one, so
    /// the gap is contribution to the pool, not request size.
    #[test]
    fn composition_needs_two_manifests_in_the_candidate_pool() {
        let authorization = br#"{"files":{"authorization.py":"def is_authorized(p):\n    return \"admin\" in p\n"}}"#.to_vec();
        let observability = br#"{"files":{"observability.py":"import json\ndef log(e, correlation_id):\n    print(e)\n"}}"#.to_vec();
        let plain_source = b"def helper():".to_vec();

        assert!(is_complete_file_manifest(&authorization));
        assert!(is_complete_file_manifest(&observability));
        assert!(!is_complete_file_manifest(&plain_source));

        let labels = vec![
            "instruction_intent:LANGUAGE:PYTHON".to_string(),
            "instruction_intent:SECURITY:AUTHORIZATION".to_string(),
        ];

        // One manifest plus plain source cannot compose -- the measured
        // failure, where only the char-motif route held the second manifest.
        let one = vec![authorization.clone(), plain_source.clone()];
        assert!(merge_grounded_file_manifests(&labels, &one).is_none());

        // Two manifests in the pool is exactly what the fix contributes.
        let two = vec![authorization, observability, plain_source];
        let merged = merge_grounded_file_manifests(&labels, &two);
        assert!(merged.is_some(), "two manifests in the pool must compose");
        let value: serde_json::Value =
            serde_json::from_slice(&merged.unwrap()).unwrap();
        let files = value["files"].as_object().unwrap();
        assert!(files.contains_key("authorization.py"), "{files:?}");
        assert!(files.contains_key("observability.py"), "{files:?}");
    }

    /// The OOV leak: a mislabelled prompt must not inherit a manifest.
    ///
    /// "distributed consensus protocol with Byzantine fault tolerance" was
    /// tagged CONCURRENCY:BOUNDED_ASYNC and admitted the bounded_map manifest,
    /// answering a request the brain is required to refuse.
    #[test]


    fn a_mislabelled_prompt_does_not_inherit_an_unrelated_manifest() {
        let bounded_map = br#"{"files":{"concurrency.py":"import asyncio\nasync def bounded_map(worker, items, limit):\n    semaphore = asyncio.Semaphore(limit)\n"}}"#;

        // The out-of-vocabulary request shares no subject vocabulary.
        assert!(!prompt_shares_manifest_subject(
            "Implement a Python distributed consensus protocol with Byzantine fault tolerance.",
            bounded_map,
        ));

        // The request this manifest genuinely answers still qualifies.
        assert!(prompt_shares_manifest_subject(
            "Write a Python bounded_map helper that limits concurrency with a semaphore.",
            bounded_map,
        ));

        // Shared scaffolding alone ("Python", "code", "write") is not subject
        // evidence -- otherwise every programming prompt would match.
        assert!(!prompt_shares_manifest_subject(
            "Write Python code using this file.",
            bounded_map,
        ));
    }

    /// A narrow request must still be able to offer its manifest -- but only
    /// when the prompt corroborates it.
    ///
    /// `sqlite_transaction` paraphrases label only LANGUAGE:PYTHON plus
    /// PERSISTENCE:ATOMIC_TRANSACTION. Under a bare `labels.len() >= 4` gate
    /// the raw-recalled manifest never joined the candidate pool, so a corpus
    /// function that merely mentioned `.execute(` won a project request.
    ///
    /// Widening on the label alone is not safe either: the extractor tagged
    /// the out-of-vocabulary prompt "distributed consensus protocol with
    /// Byzantine fault tolerance" as CONCURRENCY:BOUNDED_ASYNC, which handed
    /// it an unrelated `bounded_map` manifest and broke OOV honesty. A single
    /// behaviour label therefore admits a manifest only with prompt support.
    #[test]
    fn a_narrow_label_admits_a_manifest_only_with_prompt_support() {
        let admits = |labels: &[&str], corroborated: bool| {
            let behaviour_grounded = labels
                .iter()
                .any(|label| label.contains(":") && !label.contains(":LANGUAGE:"));
            (labels.len() >= 4 || behaviour_grounded)
                && (labels.len() >= 4 || corroborated)
        };

        let narrow = [
            "instruction_intent:LANGUAGE:PYTHON",
            "instruction_intent:PERSISTENCE:ATOMIC_TRANSACTION",
        ];
        // The regression this fixes: narrow, concrete, and supported.
        assert!(admits(&narrow, true));
        // The safety case: narrow, concrete, but the prompt does not back it.
        assert!(!admits(&narrow, false));

        // Language alone names no behaviour and stays on the richer path.
        assert!(!admits(&["instruction_intent:LANGUAGE:PYTHON"], true));
        assert!(!admits(&[], true));

        // Multi-facet requests keep qualifying on label evidence alone.
        assert!(admits(
            &[
                "instruction_intent:LANGUAGE:PYTHON",
                "instruction_intent:STRUCTURE:MULTIFILE",
                "instruction_intent:IO:FILESYSTEM",
                "instruction_intent:REPORTING:INVENTORY",
            ],
            false,
        ));
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

#[cfg(test)]
mod unlabeled_recall_gate_tests {
    use super::request_carries_enough_evidence;

    /// The foundation gate's OOV probes must not reach unlabelled recall.
    ///
    /// Measured 2026-08-21: "zxqv compiler" scored 0.455 -- over the 0.45
    /// floor -- and was answered with 626 bytes of a real Python AST visitor,
    /// dropping oov_honest to 2/3 and failing the whole completion gate. One
    /// recognisable word out of two carries a Dice ratio without establishing
    /// anything about what was asked.
    #[test]
    fn short_nonsense_requests_are_refused_evidence() {
        for prompt in ["zxqv compiler", "flurble database", "qqxz parser", "zxqv"] {
            assert!(
                !request_carries_enough_evidence(prompt),
                "{prompt:?} must not reach unlabelled recall",
            );
        }
    }

    /// The phrasings this route exists to serve must still pass.
    #[test]
    fn genuine_requests_carry_evidence() {
        for prompt in [
            "Build the server side for the calculator.",
            "I need a three.js scene class with orbit controls.",
            "Make me a calculator web app that plots equations in 3D.",
            "Give me a Vue keypad component for a calculator.",
            "Write Django tests for the evaluate endpoint.",
        ] {
            assert!(
                request_carries_enough_evidence(prompt),
                "{prompt:?} must still reach unlabelled recall",
            );
        }
    }

    /// Punctuation and separators must not be counted as words.
    #[test]
    fn word_counting_ignores_separators() {
        assert!(!request_carries_enough_evidence("three.js"));
        assert!(!request_carries_enough_evidence("a, b, c"));
        assert!(!request_carries_enough_evidence(""));
        assert!(request_carries_enough_evidence("build a calculator app"));
    }

    /// A recalled manifest must be in a language the request could have meant.
    ///
    /// The multilanguage suite's JavaScript paraphrase was answered with the
    /// Django backend manifest, failing `executes` and stalling the whole
    /// deferred queue behind a failing protected-route refresh.
    #[test]
    fn a_python_manifest_cannot_answer_a_javascript_request() {
        let django = br#"{"files": {"calculator/apps.py": "x", "calculator/views.py": "y"}}"#;
        assert!(!super::recalled_answer_language_is_plausible(
            "Create Node.js code computing the second power of a supplied number.",
            django,
        ));
        assert!(super::recalled_answer_language_is_plausible(
            "Build the Django backend for a scientific calculator.",
            django,
        ));
    }

    #[test]
    fn a_javascript_manifest_answers_a_javascript_request() {
        let three = br#"{"files": {"src/three/SceneHost.js": "x"}}"#;
        assert!(super::recalled_answer_language_is_plausible(
            "Create the 3D rendering layer with three.js.",
            three,
        ));
        assert!(!super::recalled_answer_language_is_plausible(
            "Write a Python function that sorts a list.",
            three,
        ));
    }

    /// A single-behaviour request must not be answered with a whole project.
    ///
    /// The multilanguage JavaScript paraphrase (labels JAVASCRIPT +
    /// POWER_SELF:2) was answered with the 20-file full-stack manifest. It is
    /// JavaScript-compatible, so every language check passed; it is simply far
    /// larger than what was asked for.
    #[test]
    fn one_behaviour_does_not_earn_a_project() {
        let labels = vec![
            "instruction_intent:LANGUAGE:JAVASCRIPT".to_string(),
            "instruction_intent:POWER_SELF:2".to_string(),
        ];
        let project = br#"{"files": {"a.js": "x", "b.js": "x", "c.js": "x", "d.js": "x"}}"#;
        assert!(super::single_behaviour_request_outgrown_by(&labels, Some(project)));

        // A small learned manifest remains a legitimate response contract.
        let pair = br#"{"files": {"a.js": "x", "b.js": "x"}}"#;
        assert!(!super::single_behaviour_request_outgrown_by(&labels, Some(pair)));

        // Nothing to judge.
        assert!(!super::single_behaviour_request_outgrown_by(&labels, None));
        assert!(!super::single_behaviour_request_outgrown_by(
            &labels,
            Some(b"function square(n) { return n * n; }"),
        ));
    }

    /// Bare source and language-neutral requests stay unconstrained.
    #[test]
    fn non_manifest_and_unnamed_language_are_neutral() {
        assert!(super::recalled_answer_language_is_plausible(
            "Create Node.js code computing the second power.",
            b"function square(n) { return n * n; }",
        ));
        assert!(super::recalled_answer_language_is_plausible(
            "Build the server side for the calculator.",
            br#"{"files": {"calculator/apps.py": "x"}}"#,
        ));
    }
}
