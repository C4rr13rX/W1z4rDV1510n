//! Brain identity specification per [`ARCHITECTURE.md`] §3 + §11 Phase 10.
//!
//! `BrainIdentitySpec` declaratively describes what a brain IS:
//! its pools, their prototypes, the emergence thresholds, the
//! fabric/EEM/annealer configs.  It is serializable to TOML
//! (per spec §3.3 `"brains/observer_v1.toml"`) and the brain factory
//! ([`crate::Brain::from_identity`]) instantiates a brain from it.
//!
//! # Scope (this slice)
//!
//! Phase 10 lands the **identity** half (`BrainIdentitySpec`).  The
//! deployment half (`BrainDeploymentSpec` — cluster peers, action
//! endpoints, sensor sources) is intentionally deferred to Phase 8
//! because most of its fields wire up subsystems that don't exist
//! yet (motif gossip, real action routers).  Splitting the two
//! avoids speculative field churn.
//!
//! # Encodings via prototype registry
//!
//! Atom encodings are trait objects (`Box<dyn AtomEncoding>`); they
//! can't be serialized.  Instead, each `PoolSpec` references an
//! encoding by **prototype name** (a string).  The
//! [`PoolPrototypeRegistry`] maps prototype names to factory closures
//! that build the actual encoding.  Callers can extend the registry
//! with their own prototypes before calling `from_identity`.
//!
//! The default registry ships `"byte-passthrough"` — the language /
//! action / general-byte prototype shared by every test in this
//! crate.  Future prototypes (visual, FFT-audio, screen-pixel-bin)
//! plug in the same way.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::annealer::AnnealerConfig;
use crate::eem::EemConfig;
use crate::fabric::FabricConfig;
use crate::neuron::PoolId;
use crate::pool::{AtomEncoding, BytePassthroughEncoding, ControlMode, PoolConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolKind {
    /// Receives sensor frames via `Brain::observe`.
    SensoryInput,
    /// Designated as the action pool — its neurons emit `ActionEvent`s
    /// per spec §4.E.  At most one Action pool per brain; the first
    /// one in spec order wins.
    Action,
    /// Internal pool (binding, integration, future composite layers).
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSpec {
    pub name:                 String,
    pub id:                   PoolId,
    pub prototype:            String,
    /// Prefix used by byte-passthrough-style encodings to namespace
    /// atom labels.  Unused by encodings that don't need a prefix.
    pub atom_encoding_prefix: String,
    pub kind:                 PoolKind,
    /// Each `Option` overrides the corresponding `PoolConfig::defaults`
    /// field.  `None` means "keep the default."  This keeps TOML
    /// files concise — callers only spell out the fields they want
    /// to override.
    #[serde(default)]
    pub frame_rate:                  Option<u32>,
    #[serde(default)]
    pub recent_atoms_window:         Option<usize>,
    #[serde(default)]
    pub max_concept_member_count:    Option<usize>,
    #[serde(default)]
    pub concept_emergence_threshold: Option<u32>,
    #[serde(default)]
    pub max_weight:                  Option<f32>,
    #[serde(default)]
    pub decay_rate:                  Option<f32>,
    #[serde(default)]
    pub prune_floor:                 Option<f32>,
    #[serde(default)]
    pub plasticity_baseline:         Option<f32>,
    /// Dynamical feedback wiring discovered manually or by the GA.
    #[serde(default)]
    pub sparsity_mode:               Option<ControlMode>,
    #[serde(default)]
    pub heterosynaptic_ltd_mode:     Option<ControlMode>,
    #[serde(default)]
    pub predict_gate_mode:           Option<ControlMode>,
}

impl PoolSpec {
    pub fn sensory_byte_passthrough(name: impl Into<String>, id: PoolId, prefix: impl Into<String>) -> Self {
        Self::byte_passthrough(name, id, prefix, PoolKind::SensoryInput)
    }
    pub fn action_byte_passthrough(name: impl Into<String>, id: PoolId, prefix: impl Into<String>) -> Self {
        Self::byte_passthrough(name, id, prefix, PoolKind::Action)
    }
    fn byte_passthrough(
        name:   impl Into<String>,
        id:     PoolId,
        prefix: impl Into<String>,
        kind:   PoolKind,
    ) -> Self {
        Self {
            name: name.into(),
            id,
            prototype: "byte-passthrough".into(),
            atom_encoding_prefix: prefix.into(),
            kind,
            frame_rate:                  None,
            recent_atoms_window:         None,
            max_concept_member_count:    None,
            concept_emergence_threshold: None,
            max_weight:                  None,
            decay_rate:                  None,
            prune_floor:                 None,
            plasticity_baseline:         None,
            sparsity_mode:               None,
            heterosynaptic_ltd_mode:     None,
            predict_gate_mode:           None,
        }
    }

    /// Materialize a `PoolConfig` from this spec, applying overrides
    /// over `PoolConfig::defaults`.
    pub fn to_pool_config(&self) -> PoolConfig {
        let mut c = PoolConfig::defaults(self.name.clone(), self.id);
        if let Some(v) = self.frame_rate                  { c.frame_rate                  = v; }
        if let Some(v) = self.recent_atoms_window         { c.recent_atoms_window         = v; }
        if let Some(v) = self.max_concept_member_count    { c.max_concept_member_count    = v; }
        if let Some(v) = self.concept_emergence_threshold { c.concept_emergence_threshold = v; }
        if let Some(v) = self.max_weight                  { c.max_weight                  = v; }
        if let Some(v) = self.decay_rate                  { c.decay_rate                  = v; }
        if let Some(v) = self.prune_floor                 { c.prune_floor                 = v; }
        if let Some(v) = self.plasticity_baseline         { c.plasticity_baseline         = v; }
        if let Some(v) = &self.sparsity_mode              { c.sparsity_mode               = v.clone(); }
        if let Some(v) = &self.heterosynaptic_ltd_mode    { c.heterosynaptic_ltd_mode     = v.clone(); }
        if let Some(v) = &self.predict_gate_mode          { c.predict_gate_mode           = v.clone(); }
        c
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainIdentitySpec {
    pub name:    String,
    pub version: String,
    pub pools:   Vec<PoolSpec>,
    #[serde(default = "default_binding_threshold")]
    pub binding_emergence_threshold: u32,
    #[serde(default = "default_moment_window")]
    pub moment_history_window:       usize,
    #[serde(default)]
    pub fabric:   FabricConfig,
    #[serde(default)]
    pub eem:      EemConfig,
    #[serde(default)]
    pub annealer: AnnealerConfig,
}

/// Runtime placement and feedback wiring for one isolated brain instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainDeploymentSpec {
    pub instance_id: String,
    pub identity_path: PathBuf,
    pub data_dir: PathBuf,
    #[serde(default)]
    pub resource_budget: ResourceBudget,
    #[serde(default)]
    pub feedback_loops: Vec<FeedbackLoopSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ResourceBudget {
    pub max_resident_bytes: u64,
    pub max_neurons: u64,
    pub max_propagation_steps: u32,
    /// Online learning remains active; this only bounds scheduling rate.
    pub max_learning_steps_per_second: u32,
}

/// Domain-neutral feedback edge resolved against pool names in the identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeedbackLoopSpec {
    pub source_pool: String,
    pub target_pool: String,
    pub signal: String,
    #[serde(default = "default_feedback_gain")]
    pub gain: f32,
    /// Optional dynamical controller. When present, the runtime evaluates it
    /// against the source pool's live ControlState instead of using `gain`.
    #[serde(default)]
    pub gain_mode: Option<ControlMode>,
    #[serde(default)]
    pub delay_ticks: u32,
}

fn default_feedback_gain() -> f32 { 1.0 }

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DeploymentValidationError {
    #[error("instance_id must contain only ASCII letters, digits, '.', '_' or '-'")]
    InvalidInstanceId,
    #[error("data_dir must not be empty")]
    EmptyDataDir,
    #[error("feedback loop references unknown pool '{0}'")]
    UnknownFeedbackPool(String),
    #[error("feedback loop signal must not be empty")]
    EmptyFeedbackSignal,
    #[error("feedback loop gain must be finite")]
    NonFiniteFeedbackGain,
}

impl BrainDeploymentSpec {
    pub fn load_toml<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        toml::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn validate(&self, identity: &BrainIdentitySpec) -> Result<(), DeploymentValidationError> {
        if self.instance_id.is_empty() || !self.instance_id.bytes().all(|b|
            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
            return Err(DeploymentValidationError::InvalidInstanceId);
        }
        if self.data_dir.as_os_str().is_empty() {
            return Err(DeploymentValidationError::EmptyDataDir);
        }
        for edge in &self.feedback_loops {
            for pool in [&edge.source_pool, &edge.target_pool] {
                if !identity.pools.iter().any(|p| &p.name == pool) {
                    return Err(DeploymentValidationError::UnknownFeedbackPool(pool.clone()));
                }
            }
            if edge.signal.trim().is_empty() {
                return Err(DeploymentValidationError::EmptyFeedbackSignal);
            }
            if !edge.gain.is_finite() {
                return Err(DeploymentValidationError::NonFiniteFeedbackGain);
            }
        }
        Ok(())
    }

    pub fn snapshot_path(&self) -> PathBuf { self.data_dir.join("brain.bin") }
    pub fn wal_dir(&self) -> PathBuf { self.data_dir.join("wal") }
    pub fn cold_dir(&self) -> PathBuf { self.data_dir.join("cold") }
}

fn default_binding_threshold() -> u32  { 3 }
fn default_moment_window()     -> usize { 64 }

impl BrainIdentitySpec {
    /// Sensible default identity: one text sensor + one action pool.
    /// Matches spec §3.4's "general personal observer" starter.
    /// Caller extends by appending to `pools` or by composing their
    /// own from scratch.
    pub fn default_general_observer() -> Self {
        Self {
            name:    "general_observer".into(),
            version: "0.1.0".into(),
            pools:   vec![
                PoolSpec::sensory_byte_passthrough("text",   1, "t"),
                PoolSpec::action_byte_passthrough ("action", 2, "act"),
            ],
            binding_emergence_threshold: default_binding_threshold(),
            moment_history_window:       default_moment_window(),
            fabric:   FabricConfig::default(),
            eem:      EemConfig::default(),
            annealer: AnnealerConfig::default(),
        }
    }

    pub fn load_toml<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        toml::from_str(&s).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let s = toml::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        std::fs::write(path, s)
    }
}

/// Factory: given the encoding prefix from a `PoolSpec`, build the
/// trait object.  Closures must be `Send + Sync + 'static` so the
/// registry can be shared across threads.
pub type EncodingFactory = Box<dyn Fn(&str) -> Box<dyn AtomEncoding> + Send + Sync>;

/// Maps prototype names (`"byte-passthrough"`, etc.) to encoding
/// factories.  Use [`PoolPrototypeRegistry::with_defaults`] for the
/// shipped prototype set; extend via `register` for project-specific
/// encodings (visual, FFT audio, etc.).
pub struct PoolPrototypeRegistry {
    factories: HashMap<String, EncodingFactory>,
}

impl PoolPrototypeRegistry {
    pub fn new() -> Self { Self { factories: HashMap::new() } }

    /// Registry pre-populated with every prototype this crate ships.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register("byte-passthrough", |prefix| {
            // BytePassthroughEncoding holds `prefix: &'static str` so a
            // literal compiles; here we leak a small owned copy to
            // produce a 'static reference.  The leak is bounded by
            // total pool-creation calls in the process — typically a
            // handful per brain — so it's effectively a one-time per-
            // pool cost.  Swapping `BytePassthroughEncoding.prefix`
            // to `String` removes the leak; deferred to avoid churning
            // every existing test's literal construction.
            let leaked: &'static str = Box::leak(prefix.to_owned().into_boxed_str());
            Box::new(BytePassthroughEncoding { prefix: leaked })
        });
        r
    }

    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where F: Fn(&str) -> Box<dyn AtomEncoding> + Send + Sync + 'static
    {
        self.factories.insert(name.into(), Box::new(factory));
    }

    pub fn build(&self, prototype: &str, prefix: &str) -> Option<Box<dyn AtomEncoding>> {
        self.factories.get(prototype).map(|f| f(prefix))
    }

    pub fn known_prototypes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.factories.keys().cloned().collect();
        v.sort();
        v
    }
}

impl Default for PoolPrototypeRegistry {
    fn default() -> Self { Self::with_defaults() }
}

/// Errors `Brain::from_identity` can report.
#[derive(Debug, thiserror::Error)]
pub enum IdentityBuildError {
    #[error("unknown pool prototype '{0}' — register it before building")]
    UnknownPrototype(String),
    #[error("duplicate pool id {0} — every pool must have a unique id")]
    DuplicatePoolId(PoolId),
    #[error("pool id {0} collides with the auto-created binding pool id")]
    BindingPoolIdCollision(PoolId),
}
