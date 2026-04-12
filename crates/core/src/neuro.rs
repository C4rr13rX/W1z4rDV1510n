use crate::hardware::HardwareProfile;
use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolState, SymbolType, Timestamp};
use crate::streaming::hierarchical_motifs::{
    call_fingerprint, HierarchicalMotifConfig, HierarchicalMotifRuntime, MetaMotif,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

const RELATION_BINS: i32 = 4;

/// Configuration for the lightweight neurogenesis engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuroConfig {
    pub hebbian_lr: f32,
    pub decay: f32,
    pub composite_threshold: u64,
    pub max_neurons: usize,
    pub inhibitory_scale: f32,
    pub excitatory_scale: f32,
    /// Increment applied to fatigue when a neuron fires.
    pub fatigue_increment: f32,
    /// Decay applied to fatigue each step.
    pub fatigue_decay: f32,
    /// Top-k winner-take-all per zone to keep activity sparse.
    pub wta_k_per_zone: usize,
    /// Scale for STDP-lite weight nudges.
    pub stdp_scale: f32,
    /// Co-occurrence threshold for spawning minicolumn neurons from stable signatures.
    pub minicolumn_threshold: u64,
    /// Maximum number of minicolumn neurons retained.
    pub minicolumn_max: usize,
    /// When a minicolumn is active, scale down member activations.
    pub minicolumn_inhibit_scale: f32,
    /// EMA decay for minicolumn inhibition (higher keeps inhibition longer).
    pub minicolumn_inhibition_decay: f32,
    /// EMA decay for minicolumn stability (higher keeps stability longer).
    pub minicolumn_stability_decay: f32,
    /// Minimum stability before a minicolumn suppresses its members.
    pub minicolumn_activation_threshold: f32,
    /// Inhibition level at which a minicolumn releases suppression.
    pub minicolumn_collapse_threshold: f32,
    /// Maximum number of phenotype attributes to encode in a signature.
    pub minicolumn_attr_limit: usize,
    /// Minimum number of labels required to form a minicolumn signature.
    pub minicolumn_min_signature: usize,

    // ── Hot/cold paging ────────────────────────────────────────────────────────
    /// Evict neurons idle for this many steps.  None = use default (2000 steps).
    pub eviction_idle_steps: Option<u64>,
    /// Hard cap on hot-tier neuron count.  None = use default (80_000).
    pub hot_tier_max: Option<usize>,
}

impl Default for NeuroConfig {
    fn default() -> Self {
        Self {
            hebbian_lr: 0.05,
            decay: 0.99,
            composite_threshold: 25,
            max_neurons: 100_000,
            inhibitory_scale: 0.5,
            excitatory_scale: 1.0,
            fatigue_increment: 0.02,
            fatigue_decay: 0.98,
            wta_k_per_zone: 4,
            stdp_scale: 0.02,
            minicolumn_threshold: 18,
            minicolumn_max: 512,
            minicolumn_inhibit_scale: 0.35,
            minicolumn_inhibition_decay: 0.85,
            minicolumn_stability_decay: 0.9,
            minicolumn_activation_threshold: 0.65,
            minicolumn_collapse_threshold: 0.55,
            minicolumn_attr_limit: 6,
            minicolumn_min_signature: 3,
            eviction_idle_steps: None,
            hot_tier_max: None,
        }
    }
}

/// Runtime options to drive neuron/neural-network genesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuroRuntimeConfig {
    pub enabled: bool,
    pub min_activation: f32,
    #[serde(default)]
    pub neuro: NeuroConfig,
    /// Co-occurrence threshold to promote a pair into a reusable "network" module.
    pub module_threshold: u64,
    /// Soft cap on the number of emergent networks to avoid unbounded growth.
    pub max_networks: usize,
    /// Exponential smoothing factor for temporal velocity estimates.
    pub prediction_smoothing: f32,
    /// How many steps ahead to extrapolate temporal predictions.
    pub prediction_horizon: usize,
    /// Pull strength toward temporal predictions in the proposal kernel.
    pub prediction_pull: f64,
    /// Scale for curiosity-driven plasticity (uses surprise to adapt connections).
    pub curiosity_strength: f32,
    /// Working-memory capacity (recent motifs/contexts).
    pub working_memory: usize,
    /// Pull strength from working-memory motifs.
    pub working_memory_pull: f64,
    /// Optional path to persist the NeuronPool across restarts.
    /// If set and the file exists at startup, the pool is loaded from it.
    /// Call `save_pool()` or POST `/neuro/checkpoint` to flush.
    #[serde(default)]
    pub pool_state_path: Option<String>,
    /// Config for the hierarchical motif runtime embedded in NeuroRuntime.
    /// Motifs are observed on every `train_weighted` call so patterns in
    /// training data emerge continuously.
    #[serde(default)]
    pub motifs: HierarchicalMotifConfig,
}

impl Default for NeuroRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_activation: 0.55,
            neuro: NeuroConfig::default(),
            module_threshold: 40,
            max_networks: 256,
            prediction_smoothing: 0.3,
            prediction_horizon: 2,
            prediction_pull: 0.18,
            curiosity_strength: 0.05,
            working_memory: 6,
            working_memory_pull: 0.08,
            pool_state_path: None,
            motifs: HierarchicalMotifConfig::default(),
        }
    }
}

/// Maximum number of influence records retained per neuron.
/// Older/weaker records are evicted when the buffer is full.
const MAX_INFLUENCE_HISTORY: usize = 16;

/// A snapshot of the metadata context that was active when a neuron was
/// meaningfully activated during training.  Over time a neuron accumulates
/// these records, giving a compact provenance chain:
///   "this neuron fired most strongly during Sicilian opening games
///    played by Tal, and during jazz tracks in the key of F minor."
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InfluenceRecord {
    /// Data stream identifier (e.g. "chess", "audio", "text").
    pub stream: String,
    /// Significant metadata key-value pairs active during this training step.
    /// e.g. [("artist", "Radiohead"), ("album", "OK Computer"), ("year", "1997")]
    pub labels: Vec<(String, String)>,
    /// Cumulative Hebbian weight contribution from this context.
    pub strength: f32,
    /// Training step at which this record was last updated.
    pub step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub target: u32,
    pub weight: f32,
    pub inhibitory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dendrite {
    pub source: u32,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: u32,
    pub label: Option<String>,
    /// Optional symbol this neuron stands for (id::<symbol>).
    pub symbol_id: Option<String>,
    pub activation: f32,
    pub bias: f32,
    /// Incoming connections.
    pub dendrites: Vec<Dendrite>,
    /// Outgoing excitatory synapses.
    pub excitatory: Vec<Synapse>,
    /// Outgoing inhibitory synapses.
    pub inhibitory: Vec<Synapse>,
    pub born_at: u64,
    pub use_count: u64,
    /// Short-term fatigue (higher = more suppressed).
    pub fatigue: f32,
    /// Eligibility trace for STDP-lite updates.
    pub trace: f32,
    /// Rolling provenance: which metadata contexts shaped this neuron.
    /// Bounded to MAX_INFLUENCE_HISTORY; weakest evicted when full.
    pub influence_history: Vec<InfluenceRecord>,
    /// The pool step at which this neuron was last activated (for eviction tracking).
    #[serde(default)]
    pub last_active_step: u64,
}

impl Neuron {
    fn new(id: u32, label: Option<String>, born_at: u64) -> Self {
        Self {
            id,
            label,
            symbol_id: None,
            activation: 0.0,
            bias: 0.0,
            dendrites: Vec::new(),
            excitatory: Vec::new(),
            inhibitory: Vec::new(),
            born_at,
            use_count: 0,
            fatigue: 0.0,
            trace: 0.0,
            influence_history: Vec::new(),
            last_active_step: 0,
        }
    }

    /// Record a metadata context that contributed to this neuron's activation.
    /// Merges with an existing record for the same stream if present;
    /// evicts the weakest record when the buffer is full.
    pub fn record_influence(&mut self, record: InfluenceRecord) {
        if record.stream.is_empty() {
            return;
        }
        // Merge with an existing record for the same stream context
        let key_match = |existing: &InfluenceRecord| {
            existing.stream == record.stream && existing.labels == record.labels
        };
        if let Some(existing) = self.influence_history.iter_mut().find(|r| key_match(r)) {
            existing.strength += record.strength;
            existing.step = record.step;
            return;
        }
        if self.influence_history.len() >= MAX_INFLUENCE_HISTORY {
            // Evict the weakest record
            if let Some(min_pos) = self
                .influence_history
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
                .map(|(i, _)| i)
            {
                self.influence_history.remove(min_pos);
            }
        }
        self.influence_history.push(record);
    }
}

#[derive(Debug, Clone)]
struct MinicolumnSignature {
    key: String,
    labels: Vec<String>,
    attr_map: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct MiniColumn {
    id: u32,
    label: String,
    labels: Vec<String>,
    attr_map: HashMap<String, String>,
    members: Vec<u32>,
    stability: f32,
    inhibition: f32,
    born_at: u64,
    last_seen: u64,
    collapsed: bool,
    /// True when member activations changed since last signature match.
    /// Dirty columns skip the expensive best_signature_match computation.
    activation_dirty: bool,
    /// Cached evidence from last best_signature_match call.
    cached_evidence: f32,
    /// Cached conflict from last best_signature_match call.
    cached_conflict: f32,
}

/// Serde helper: serialize/deserialize `HashMap<(String,String), f32>` as a vec of triples.
mod cooccur_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    #[derive(Serialize, Deserialize)]
    struct Entry(String, String, f32);

    pub fn serialize<S>(map: &HashMap<(String, String), f32>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let v: Vec<Entry> = map
            .iter()
            .map(|((a, b), &v)| Entry(a.clone(), b.clone(), v))
            .collect();
        v.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<HashMap<(String, String), f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v: Vec<Entry> = Vec::deserialize(d)?;
        Ok(v.into_iter().map(|Entry(a, b, v)| ((a, b), v)).collect())
    }
}

/// One slot in the neuron array.  Hot = in RAM, Cold = evicted to disk, Free = recycled ID.
#[derive(Debug, Clone)]
enum NeuronSlot {
    Hot(Neuron),
    /// Neuron evicted to disk.  `last_active_step` is the step it was frozen at,
    /// used to apply lazy geometric decay when the neuron is warmed back up.
    Cold { last_active_step: u64 },
    /// Recycled/unused slot — ID available via the free list.
    Free,
}

impl NeuronSlot {
    fn as_hot(&self) -> Option<&Neuron> {
        if let NeuronSlot::Hot(n) = self { Some(n) } else { None }
    }
    fn as_hot_mut(&mut self) -> Option<&mut Neuron> {
        if let NeuronSlot::Hot(n) = self { Some(n) } else { None }
    }
}

// ── Custom serde for NeuronPool: serialize only hot neurons so checkpoints stay small.
// Wire format is identical to the old monolithic format ("neurons": [...Neuron...]),
// which means old checkpoint files load transparently.
mod pool_serde {
    use super::{Neuron, NeuronSlot};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(slots: &Vec<NeuronSlot>, s: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let hot: Vec<&Neuron> = slots.iter().filter_map(|sl| sl.as_hot()).collect();
        hot.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Vec<NeuronSlot>, D::Error>
    where D: Deserializer<'de> {
        let neurons: Vec<Neuron> = Vec::deserialize(d)?;
        Ok(neurons.into_iter().map(NeuronSlot::Hot).collect())
    }
}

/// Simple object-pool style neuron store with hot/cold paging.
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuronPool {
    #[serde(with = "pool_serde")]
    neurons: Vec<NeuronSlot>,
    free: Vec<u32>,
    label_to_id: HashMap<String, u32>,
    /// EMA co-occurrence rate per label pair.  Incremented with decay:
    /// `rate = rate * EMA_ALPHA + 1.0` on co-occurrence; equilibrium ≈ 33 at alpha=0.97.
    #[serde(with = "cooccur_serde")]
    cooccur: HashMap<(String, String), f32>,
    config: NeuroConfig,
    step: u64,
    /// Maximum cooccur entries before pruning low-rate entries.  `usize::MAX` = unlimited.
    cooccur_cap: usize,

    // ── Hot/cold paging ────────────────────────────────────────────────────────
    /// Base directory for cold neuron files.  None = eviction disabled (in-memory only).
    #[serde(skip)]
    cold_dir: Option<PathBuf>,
    /// Evict neurons idle for this many steps.  Default 2000 (~3 min at 10 Hz training).
    #[serde(skip)]
    eviction_idle_steps: u64,
    /// Hard cap on hot-tier size.  When exceeded, LRU eviction runs immediately.
    #[serde(skip)]
    hot_tier_max: usize,
    /// Cached count of Hot slots (avoids iterating to count).
    #[serde(skip)]
    hot_count: usize,
}

const COOCCUR_EMA_ALPHA: f32 = 0.97;
/// Default: evict neurons idle for 2000 steps (~3 minutes at normal training cadence).
const DEFAULT_EVICTION_IDLE_STEPS: u64 = 2000;
/// Default hot-tier cap: 80K neurons in RAM regardless of idle threshold.
const DEFAULT_HOT_TIER_MAX: usize = 80_000;

impl NeuronPool {
    pub fn new(config: NeuroConfig) -> Self {
        Self {
            neurons: Vec::new(),
            free: Vec::new(),
            label_to_id: HashMap::new(),
            cooccur: HashMap::new(),
            config,
            step: 0,
            cooccur_cap: usize::MAX,
            cold_dir: None,
            eviction_idle_steps: DEFAULT_EVICTION_IDLE_STEPS,
            hot_tier_max: DEFAULT_HOT_TIER_MAX,
            hot_count: 0,
        }
    }

    /// Configure hot/cold paging.  Called by NeuroRuntime::new after construction.
    pub fn set_cold_dir(&mut self, dir: PathBuf, idle_steps: u64, hot_max: usize) {
        self.cold_dir = Some(dir);
        self.eviction_idle_steps = idle_steps;
        self.hot_tier_max = hot_max;
        // Recount hot slots (important after loading a checkpoint that was all-hot).
        self.hot_count = self.neurons.iter().filter(|s| matches!(s, NeuronSlot::Hot(_))).count();
    }

    // ── Internal hot/cold helpers ──────────────────────────────────────────────

    /// Path to a cold neuron file for the given ID.
    fn cold_path(&self, id: u32) -> Option<PathBuf> {
        let base = self.cold_dir.as_ref()?;
        let shard = format!("shard_{:04x}", id / 1000);
        Some(base.join(shard).join(format!("n_{:06}.json", id)))
    }

    /// Ensure neuron `id` is in the hot tier.  Reads from disk if cold,
    /// applies lazy geometric decay, then replaces the Cold slot with Hot.
    /// No-op if already hot. Does nothing if cold_dir is not configured.
    fn ensure_hot(&mut self, id: u32) {
        let slot = match self.neurons.get(id as usize) {
            Some(NeuronSlot::Cold { .. }) => true,
            _ => return,
        };
        if !slot { return; }

        let frozen_step = if let Some(NeuronSlot::Cold { last_active_step }) = self.neurons.get(id as usize) {
            *last_active_step
        } else { return; };

        let path = match self.cold_path(id) {
            Some(p) => p,
            None => return,
        };

        let mut neuron = if path.exists() {
            match std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Neuron>(&s).ok()) {
                Some(n) => n,
                None => {
                    // File corrupt or missing — recreate a blank neuron at this ID
                    let mut n = Neuron::new(id, None, self.step);
                    n.last_active_step = frozen_step;
                    n
                }
            }
        } else {
            // File was never written (should not happen, but be defensive)
            let mut n = Neuron::new(id, None, self.step);
            n.last_active_step = frozen_step;
            n
        };

        // Apply lazy geometric decay for all the steps the neuron spent cold.
        let elapsed = self.step.saturating_sub(frozen_step) as f32;
        if elapsed > 0.0 {
            let act_factor = self.config.decay.powf(elapsed);
            let fat_factor = self.config.fatigue_decay.powf(elapsed);
            neuron.activation *= act_factor;
            neuron.fatigue    *= fat_factor;
            neuron.trace      *= act_factor;
        }

        self.neurons[id as usize] = NeuronSlot::Hot(neuron);
        self.hot_count += 1;
    }

    /// Write a neuron to its cold file and replace the slot with Cold.
    fn evict_to_cold(&mut self, id: u32) {
        let neuron = match self.neurons.get(id as usize) {
            Some(NeuronSlot::Hot(n)) => n.clone(),
            _ => return,
        };
        if let Some(path) = self.cold_path(id) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string(&neuron) {
                let tmp = path.with_extension("tmp");
                if std::fs::write(&tmp, &json).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
        }
        let last_active_step = neuron.last_active_step;
        self.neurons[id as usize] = NeuronSlot::Cold { last_active_step };
        self.hot_count -= 1;
    }

    /// Eviction pass: called periodically from `step()`.
    /// Evicts neurons that have been idle longer than `eviction_idle_steps`.
    fn run_eviction(&mut self) {
        if self.cold_dir.is_none() { return; }
        let idle_threshold = self.eviction_idle_steps;
        let current_step = self.step;
        let hot_max = self.hot_tier_max;

        // Collect IDs to evict: idle + activation below threshold.
        // We collect first to avoid borrow issues during mutation.
        let to_evict: Vec<u32> = self.neurons.iter().enumerate().filter_map(|(idx, slot)| {
            if let NeuronSlot::Hot(n) = slot {
                let idle = current_step.saturating_sub(n.last_active_step);
                // Evict if idle long enough AND activation has decayed to near zero.
                if idle >= idle_threshold && n.activation < 0.005 && n.trace < 0.005 {
                    return Some(idx as u32);
                }
            }
            None
        }).collect();

        for id in to_evict {
            self.evict_to_cold(id);
        }

        // Hard cap: if still over hot_tier_max, evict by LRU (oldest last_active_step first).
        if self.hot_count > hot_max {
            let mut hot_ids: Vec<(u64, u32)> = self.neurons.iter().enumerate().filter_map(|(idx, slot)| {
                if let NeuronSlot::Hot(n) = slot {
                    Some((n.last_active_step, idx as u32))
                } else { None }
            }).collect();
            hot_ids.sort_unstable_by_key(|(ts, _)| *ts); // oldest first
            let evict_count = self.hot_count.saturating_sub(hot_max);
            for (_, id) in hot_ids.into_iter().take(evict_count) {
                self.evict_to_cold(id);
            }
        }
    }

    /// Scan the cold directory and mark all evicted neurons as Cold slots in the slot array.
    /// Called on startup when a checkpoint is loaded alongside an existing cold directory.
    pub fn restore_cold_index(&mut self, cold_dir: &PathBuf) {
        let meta_path = cold_dir.join("meta.json");
        if let Ok(json) = std::fs::read_to_string(&meta_path) {
            if let Ok(index) = serde_json::from_str::<HashMap<u32, u64>>(&json) {
                for (id, last_active_step) in index {
                    let idx = id as usize;
                    // Grow slot array if needed (cold neurons beyond current hot count).
                    while self.neurons.len() <= idx {
                        self.neurons.push(NeuronSlot::Free);
                    }
                    // Only mark as Cold if not already Hot (hot checkpoint wins).
                    if !matches!(self.neurons.get(idx), Some(NeuronSlot::Hot(_))) {
                        self.neurons[idx] = NeuronSlot::Cold { last_active_step };
                        // Restore label_to_id mapping from cold file if not present.
                        if let Some(label) = self.read_cold_label(id) {
                            self.label_to_id.entry(label).or_insert(id);
                        }
                    }
                }
            }
        }
        // Recount hot slots after restoration.
        self.hot_count = self.neurons.iter().filter(|s| matches!(s, NeuronSlot::Hot(_))).count();
    }

    /// Persist cold-tier index (id → last_active_step) so it can be restored on next boot.
    fn save_cold_index(&self) {
        let cd = match &self.cold_dir { Some(d) => d, None => return };
        let index: HashMap<u32, u64> = self.neurons.iter().enumerate().filter_map(|(idx, slot)| {
            if let NeuronSlot::Cold { last_active_step } = slot {
                Some((idx as u32, *last_active_step))
            } else { None }
        }).collect();
        let meta_path = cd.join("meta.json");
        if let Ok(json) = serde_json::to_string(&index) {
            let tmp = meta_path.with_extension("tmp");
            let _ = std::fs::create_dir_all(cd);
            if std::fs::write(&tmp, &json).is_ok() {
                let _ = std::fs::rename(&tmp, &meta_path);
            }
        }
    }

    /// Read the label of a cold neuron without warming it up (label_for read-only path).
    fn read_cold_label(&self, id: u32) -> Option<String> {
        let path = self.cold_path(id)?;
        let json = std::fs::read_to_string(&path).ok()?;
        // Fast path: extract "label" field without full parse
        serde_json::from_str::<serde_json::Value>(&json).ok()
            .and_then(|v| v.get("label")?.as_str().map(|s| s.to_string()))
    }

    /// Access a hot neuron by slot index (read-only).
    fn get_hot(&self, idx: usize) -> Option<&Neuron> {
        self.neurons.get(idx)?.as_hot()
    }

    /// Access a hot neuron by slot index (mutable). Does NOT warm up cold neurons.
    fn get_hot_mut(&mut self, idx: usize) -> Option<&mut Neuron> {
        self.neurons.get_mut(idx)?.as_hot_mut()
    }

    pub fn step(&mut self) {
        self.step += 1;
        for slot in self.neurons.iter_mut() {
            if let NeuronSlot::Hot(neuron) = slot {
                neuron.activation *= self.config.decay;
                neuron.fatigue    *= self.config.fatigue_decay;
                neuron.trace      *= self.config.decay;
            }
        }
        // Eviction: every 200 steps (amortises the O(n) scan cost).
        if self.step % 200 == 0 {
            self.run_eviction();
        }
    }

    pub fn get_or_create(&mut self, label: &str) -> u32 {
        if let Some(&id) = self.label_to_id.get(label) {
            // Warm up cold neuron on demand.
            self.ensure_hot(id);
            return id;
        }
        let id = if let Some(id) = self.free.pop() {
            // Reuse a freed slot — it may have been cold; overwrite it directly.
            let step = self.step;
            let mut neuron = Neuron::new(id, Some(label.to_string()), step);
            neuron.symbol_id = label.strip_prefix("id::").map(|s| s.to_string());
            if matches!(self.neurons.get(id as usize), Some(NeuronSlot::Hot(_))) {
                self.hot_count -= 1; // was hot, being replaced
            }
            self.neurons[id as usize] = NeuronSlot::Hot(neuron);
            self.hot_count += 1;
            id
        } else {
            let id = self.neurons.len() as u32;
            if self.neurons.len() >= self.config.max_neurons {
                return id.saturating_sub(1);
            }
            let mut neuron = Neuron::new(id, Some(label.to_string()), self.step);
            neuron.symbol_id = label.strip_prefix("id::").map(|s| s.to_string());
            self.neurons.push(NeuronSlot::Hot(neuron));
            self.hot_count += 1;
            id
        };
        self.label_to_id.insert(label.to_string(), id);
        id
    }

    pub fn record_symbols(&mut self, symbols: &[String]) {
        self.record_symbols_with_meta(symbols, None);
    }

    /// Record symbols and optionally attach a metadata context to each activated neuron.
    ///
    /// The `meta_context` is a list of (key, value) pairs representing the
    /// training context — e.g. `[("stream", "chess"), ("opening", "Sicilian"),
    /// ("artist", "Radiohead"), ("album", "OK Computer")]`.  Each activated
    /// neuron accumulates these as `InfluenceRecord`s, building a provenance chain
    /// that can later be queried: "what influenced this neuron?".
    pub fn record_symbols_with_meta(
        &mut self,
        symbols: &[String],
        meta_context: Option<&Vec<(String, String)>>,
    ) {
        // activate neurons for symbols
        let mut ids = Vec::with_capacity(symbols.len());
        let current_step = self.step;
        let fatigue_increment = self.config.fatigue_increment;
        for label in symbols {
            let id = self.get_or_create(label);
            if let Some(neuron) = self.get_hot_mut(id as usize) {
                neuron.activation = (1.0 - neuron.fatigue).max(0.0);
                neuron.use_count += 1;
                neuron.trace += 0.1;
                neuron.fatigue = (neuron.fatigue + fatigue_increment).min(0.6);
                neuron.last_active_step = current_step;
            }
            ids.push(id);
        }
        // Record metadata influence on each activated neuron
        if let Some(meta) = meta_context {
            if !meta.is_empty() {
                let stream = meta
                    .iter()
                    .find(|(k, _)| k == "stream")
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default();
                let step = self.step;
                let influence = InfluenceRecord {
                    stream,
                    labels: meta.clone(),
                    strength: 0.1,
                    step,
                };
                for &id in &ids {
                    if let Some(neuron) = self.get_hot_mut(id as usize) {
                        if neuron.activation > 0.05 {
                            neuron.record_influence(influence.clone());
                        }
                    }
                }
            }
        }
        // co-occurrence tracking
        let mut uniq: Vec<String> = symbols.iter().cloned().collect();
        uniq.sort();
        uniq.dedup();
        for i in 0..uniq.len() {
            for j in (i + 1)..uniq.len() {
                let key = (uniq[i].clone(), uniq[j].clone());
                // EMA-based co-occurrence rate: converges to ~33 if seen every frame.
                let entry = self.cooccur.entry(key).or_insert(0.0);
                *entry = *entry * COOCCUR_EMA_ALPHA + 1.0;
            }
        }
        // On constrained hardware, prune low-rate cooccur entries to bound memory.
        if self.cooccur.len() > self.cooccur_cap {
            self.cooccur.retain(|_, v| *v > 1.5);
        }
        self.maybe_spawn_composites();
        // Hebbian strengthening between active pairs — scale delta by within-batch pair count.
        // If the same two IDs appear together multiple times in this call, the delta is proportionally larger.
        let mut pair_counts: HashMap<(u32, u32), u32> = HashMap::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = if ids[i] <= ids[j] { (ids[i], ids[j]) } else { (ids[j], ids[i]) };
                *pair_counts.entry((a, b)).or_insert(0) += 1;
            }
        }
        for ((a, b), cnt) in pair_counts {
            let scale = self.config.excitatory_scale * cnt as f32;
            self.hebbian_pair(a, b, scale, false);
        }
    }

    /// Reward-weighted Hebbian training.
    ///
    /// `lr_scale > 1.0` strengthens the update (correct prediction, positive reward).
    /// `inhibitory = true` applies an inhibitory delta (wrong prediction, negative reward).
    /// This is the primary entry point for the `FabricTrainer` feedback loop.
    pub fn train_weighted(&mut self, symbols: &[String], lr_scale: f32, inhibitory: bool) {
        self.train_weighted_with_meta(symbols, lr_scale, inhibitory, None);
    }

    /// Reward-weighted Hebbian training with metadata provenance.
    ///
    /// Same as `train_weighted` but attaches a metadata context to each
    /// activated neuron.  Use this when training from labeled data streams
    /// (audio tracks, text descriptions) so neurons accumulate a record of
    /// which training examples shaped them.
    pub fn train_weighted_with_meta(
        &mut self,
        symbols: &[String],
        lr_scale: f32,
        inhibitory: bool,
        meta_context: Option<&Vec<(String, String)>>,
    ) {
        if symbols.is_empty() || lr_scale <= 0.0 {
            return;
        }
        let mut ids = Vec::with_capacity(symbols.len());
        let current_step = self.step;
        for label in symbols {
            let id = self.get_or_create(label);
            if let Some(neuron) = self.get_hot_mut(id as usize) {
                neuron.activation = (1.0 - neuron.fatigue).max(0.0);
                neuron.use_count += 1;
                neuron.trace += 0.1 * lr_scale;
                neuron.last_active_step = current_step;
            }
            ids.push(id);
        }
        // Record metadata influence weighted by lr_scale
        if let Some(meta) = meta_context {
            if !meta.is_empty() {
                let stream = meta
                    .iter()
                    .find(|(k, _)| k == "stream")
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default();
                let step = self.step;
                let influence = InfluenceRecord {
                    stream,
                    labels: meta.clone(),
                    strength: 0.1 * lr_scale.clamp(0.01, 8.0),
                    step,
                };
                for &id in &ids {
                    if let Some(neuron) = self.get_hot_mut(id as usize) {
                        if neuron.activation > 0.05 {
                            neuron.record_influence(influence.clone());
                        }
                    }
                }
            }
        }
        let scale = self.config.excitatory_scale * lr_scale.clamp(0.01, 8.0);
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                self.hebbian_pair(ids[i], ids[j], scale, inhibitory);
            }
        }
    }

    pub fn active_ids(&self, min_activation: f32) -> HashSet<u32> {
        let mut set = HashSet::new();
        for slot in &self.neurons {
            if let NeuronSlot::Hot(n) = slot {
                if n.activation >= min_activation {
                    set.insert(n.id);
                }
            }
        }
        set
    }

    pub fn label_for(&self, id: u32) -> Option<String> {
        match self.neurons.get(id as usize) {
            Some(NeuronSlot::Hot(n)) => n.label.clone(),
            Some(NeuronSlot::Cold { .. }) => self.read_cold_label(id),
            _ => None,
        }
    }

    /// Propagate activation forward through excitatory synapses.
    ///
    /// Seeds the pool with `seed_labels` at full activation, then walks
    /// excitatory synapses for `hops` rounds, accumulating weighted activation
    /// at each reachable neuron.  Inhibitory synapses suppress their targets.
    ///
    /// Returns a map of `label → activation_strength` for every neuron that
    /// exceeded `min_activation` at any point during propagation.  This is the
    /// read-out of what the network "thinks" given the seed input — without
    /// modifying any weights or the pool's live state.
    pub fn propagate(
        &self,
        seed_labels: &[String],
        hops: usize,
        min_activation: f32,
    ) -> HashMap<String, f32> {
        // Current activation state — keyed by neuron ID to avoid label lookups in the hot path.
        // We work on a scratch copy so we never mutate the live pool state.
        // Cold neurons have activation=0 by definition so they safely contribute nothing.
        let n = self.neurons.len();
        let mut activation: Vec<f32> = vec![0.0; n];

        // Seed
        for label in seed_labels {
            if let Some(&id) = self.label_to_id.get(label) {
                if (id as usize) < n {
                    activation[id as usize] = 1.0;
                }
            }
        }

        // Hop-by-hop propagation — scratch buffer pre-allocated to avoid per-hop alloc.
        let mut next: Vec<f32> = vec![0.0; n];
        for _ in 0..hops {
            next.copy_from_slice(&activation);
            for (src_idx, &src_act) in activation.iter().enumerate() {
                if src_act < 0.001 {
                    continue;
                }
                // Only hot neurons have synapses accessible in RAM.
                let neuron = match self.get_hot(src_idx) {
                    Some(n) => n,
                    None => continue,
                };
                // Excitatory: add weighted activation
                for syn in &neuron.excitatory {
                    let tgt = syn.target as usize;
                    if tgt < n {
                        next[tgt] = (next[tgt] + src_act * syn.weight * 0.5).min(1.0);
                    }
                }
                // Inhibitory: suppress target
                for syn in &neuron.inhibitory {
                    let tgt = syn.target as usize;
                    if tgt < n {
                        next[tgt] = (next[tgt] - src_act * syn.weight.abs() * 0.3).max(0.0);
                    }
                }
            }
            // Decay so distant hops carry less weight
            for v in next.iter_mut() {
                *v *= 0.85;
            }
            std::mem::swap(&mut activation, &mut next);
        }

        // Collect results above threshold (hot neurons only — cold neurons cannot have activation)
        let mut result = HashMap::new();
        for (idx, act) in activation.iter().enumerate() {
            if *act >= min_activation {
                if let Some(label) = self.get_hot(idx).and_then(|n| n.label.as_ref()) {
                    result.insert(label.clone(), *act);
                }
            }
        }
        result
    }

    pub fn cooccurrences_above(&self, threshold: u64) -> Vec<((String, String), f32)> {
        let t = threshold as f32;
        self.cooccur
            .iter()
            .filter_map(|((a, b), rate)| {
                if *rate >= t {
                    Some(((a.clone(), b.clone()), *rate))
                } else {
                    None
                }
            })
            .collect()
    }

    fn hebbian_pair(&mut self, a: u32, b: u32, scale: f32, inhibitory: bool) {
        // Warm up both neurons before accessing them (no-op if already hot).
        self.ensure_hot(a);
        self.ensure_hot(b);
        let (Some(n_a), Some(n_b)) = (self.get_hot(a as usize), self.get_hot(b as usize))
        else {
            return;
        };
        let delta = self.config.hebbian_lr
            * (n_a.activation + n_a.trace)
            * (n_b.activation + n_b.trace)
            * scale;
        self.add_synapse(a, b, delta, inhibitory);
        self.add_synapse(b, a, delta, inhibitory);
    }

    fn add_synapse(&mut self, from: u32, to: u32, delta: f32, inhibitory: bool) {
        if let Some(neuron) = self.get_hot_mut(from as usize) {
            let list = if inhibitory {
                &mut neuron.inhibitory
            } else {
                &mut neuron.excitatory
            };
            // Lists are kept sorted by target for O(log n) lookup.
            match list.binary_search_by_key(&to, |s| s.target) {
                Ok(idx) => list[idx].weight += delta,
                Err(idx) => list.insert(idx, Synapse { target: to, weight: delta, inhibitory }),
            }
        }
        // add dendrite on target side (sorted by source)
        if let Some(target) = self.get_hot_mut(to as usize) {
            match target.dendrites.binary_search_by_key(&from, |d| d.source) {
                Ok(idx) => target.dendrites[idx].weight += delta,
                Err(idx) => target.dendrites.insert(idx, Dendrite { source: from, weight: delta }),
            }
        }
    }

    fn maybe_spawn_composites(&mut self) {
        let threshold = self.config.composite_threshold as f32;
        let mut new_labels = Vec::new();
        for ((a, b), count) in self.cooccur.iter() {
            if *count >= threshold {
                let comp_label = format!("comp::{a}+{b}");
                if !self.label_to_id.contains_key(&comp_label) {
                    new_labels.push(comp_label);
                }
            }
        }
        for label in new_labels {
            let id = self.get_or_create(&label);
            // connect to constituents
            if let Some((a, b)) = label.strip_prefix("comp::").and_then(|s| s.split_once('+')) {
                if self.label_to_id.contains_key(a) && self.label_to_id.contains_key(b) {
                    let a_id = self.get_or_create(a);
                    let b_id = self.get_or_create(b);
                    self.hebbian_pair(id, a_id, self.config.excitatory_scale, false);
                    self.hebbian_pair(id, b_id, self.config.excitatory_scale, false);
                }
            }
        }
    }

    pub fn active_labels(&self, min_activation: f32) -> HashSet<String> {
        let mut set = HashSet::new();
        for slot in self.neurons.iter() {
            if let NeuronSlot::Hot(n) = slot {
                if n.activation >= min_activation {
                    if let Some(label) = &n.label {
                        set.insert(label.clone());
                    }
                }
            }
        }
        set
    }

    pub fn step_counter(&self) -> u64 {
        self.step
    }
}

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub id: u32,
    pub label: String,
    pub members: Vec<u32>,
    pub born_at: u64,
    pub strength: f32,
    pub use_count: u64,
    pub level: u8,
    pub excites: HashMap<u32, f32>,
    pub inhibits: HashMap<u32, f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuralNetworkSnapshot {
    pub label: String,
    pub members: Vec<String>,
    pub strength: f32,
    pub level: u8,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinicolumnSnapshot {
    pub label: String,
    pub stability: f32,
    pub inhibition: f32,
    pub collapsed: bool,
    pub members: usize,
    pub born_at: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuroSnapshot {
    pub active_labels: HashSet<String>,
    pub active_composites: HashSet<String>,
    pub active_networks: Vec<NeuralNetworkSnapshot>,
    pub minicolumns: Vec<MinicolumnSnapshot>,
    pub centroids: HashMap<String, Position>,
    pub network_links: HashMap<String, HashMap<String, f32>>,
    pub temporal_predictions: HashMap<String, Position>,
    pub prediction_error: HashMap<String, f64>,
    pub prediction_confidence: HashMap<String, f64>,
    pub surprise: HashMap<String, f64>,
    pub working_memory: Vec<String>,
    pub temporal_motif_priors: HashMap<String, f64>,
    /// Which data streams are currently active (derived from `stream::*` labels).
    pub active_streams: HashSet<String>,
    /// Flattened key→value pairs from active `meta::key::value` labels.
    /// Represents the metadata context currently influencing the fabric.
    pub active_meta_labels: HashMap<String, String>,
    /// Aggregated influence provenance from the most active neurons.
    /// Each record describes a training context that shaped the current activation.
    /// Use this to answer: "what data influenced this output?"
    pub top_influences: Vec<InfluenceRecord>,
}

impl NeuroSnapshot {
    pub fn is_empty(&self) -> bool {
        self.active_labels.is_empty()
            && self.active_composites.is_empty()
            && self.active_networks.is_empty()
            && self.minicolumns.is_empty()
            && self.centroids.is_empty()
            && self.temporal_predictions.is_empty()
            && self.prediction_error.is_empty()
            && self.prediction_confidence.is_empty()
            && self.surprise.is_empty()
            && self.working_memory.is_empty()
            && self.temporal_motif_priors.is_empty()
            && self.top_influences.is_empty()
    }
}

// ── Episodic Memory ───────────────────────────────────────────────────────────
//
// The episodic store is a content-addressable, temporally-tagged circular buffer
// of *resolved* prediction episodes.  Each episode records:
//   - the feature vector that was active when the prediction was made
//   - what the fabric predicted vs. what actually happened
//   - which sensor/virtual-sensor streams contributed to the context
//   - a surprise score (0 = expected, 1 = totally wrong)
//   - an importance weight that rises with surprise and decays with age
//
// On a wrong prediction the runtime queries the store for the K most similar
// past episodes.  These "similar failures" surface which context variable
// differed in the cases where the fabric got it right, feeding the
// ConditionalSufficiencyTracker with expansion candidates.
//
// Cross-modal tagging: episodes carry ALL contributing stream names.  A query
// from the chess stream can therefore retrieve episodes that were shaped by the
// chemistry or audio streams, enabling genuine cross-modal recall.

/// Maximum episodes retained before the importance-weighted oldest are evicted.
const EPISODIC_CAPACITY: usize = 4096;

/// Decay applied to importance each step so old episodes fade.
const EPISODIC_IMPORTANCE_DECAY: f32 = 0.9995;

/// One resolved prediction episode.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Sparse context feature vector — labels active at prediction time.
    /// Stored as a sorted, deduped Vec so cosine similarity is O(min(a,b)).
    pub context_labels: Vec<String>,
    /// The label the fabric predicted.
    pub predicted: String,
    /// The label that actually occurred.
    pub actual: String,
    /// All stream names that contributed labels to this episode.
    /// Enables cross-modal retrieval: querying from "chess" can surface
    /// an episode that was co-labeled with "chemistry" or "audio".
    pub streams: Vec<String>,
    /// Fabric step at which this episode was recorded.
    pub timestamp: u64,
    /// Whether the prediction matched reality.
    pub correct: bool,
    /// Surprise magnitude (focal-loss-inspired): `(1 - p_correct)^2`.
    /// High when the model was confidently wrong.
    pub surprise: f32,
    /// Running importance weight.  Initialised = surprise; decays over time;
    /// bumped whenever a similar episode triggers a retrieval hit.
    pub importance: f32,
}

impl Episode {
    /// Compute a sparse cosine similarity between two sorted label vectors.
    /// Returns a value in [0, 1].
    pub fn label_similarity(a: &[String], b: &[String]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        // Count intersection via merge of two sorted slices.
        let mut i = 0;
        let mut j = 0;
        let mut common = 0usize;
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => { common += 1; i += 1; j += 1; }
                std::cmp::Ordering::Less   => { i += 1; }
                std::cmp::Ordering::Greater=> { j += 1; }
            }
        }
        // Cosine for binary (0/1) bag-of-labels: dot / (|a| * |b|)^0.5
        common as f32 / ((a.len() as f32) * (b.len() as f32)).sqrt()
    }
}

/// Vector-indexed, importance-weighted episodic memory store.
///
/// Capacity is capped at `EPISODIC_CAPACITY`.  When full, the entry with the
/// lowest `importance` score is evicted — preserving the most surprising /
/// most recently-recalled episodes regardless of their age.
#[derive(Debug)]
pub struct EpisodicStore {
    episodes: Vec<Episode>,
    /// Total episodes ever recorded (for logging / metrics).
    pub total_recorded: u64,
}

impl EpisodicStore {
    pub fn new() -> Self {
        Self {
            episodes: Vec::with_capacity(EPISODIC_CAPACITY.min(256)),
            total_recorded: 0,
        }
    }

    /// Record a newly-resolved prediction episode.
    pub fn record(&mut self, episode: Episode) {
        self.total_recorded += 1;
        if self.episodes.len() >= EPISODIC_CAPACITY {
            // Evict lowest-importance entry
            if let Some(min_pos) = self
                .episodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.episodes.remove(min_pos);
            }
        }
        self.episodes.push(episode);
    }

    /// Decay importance on every fabric step so stale episodes lose weight.
    pub fn step(&mut self) {
        for ep in self.episodes.iter_mut() {
            ep.importance *= EPISODIC_IMPORTANCE_DECAY;
        }
    }

    /// Retrieve the `k` most similar *wrong* past episodes to the given context.
    ///
    /// Similarity is sparse cosine over label sets.  A retrieval hit bumps the
    /// importance of matched episodes (rehearsal effect).  Only incorrect
    /// episodes are returned — the caller uses them to find which context
    /// variable differed in correct cases.
    pub fn query_similar_failures<'a>(
        &'a mut self,
        context_labels: &[String],
        k: usize,
    ) -> Vec<&'a Episode> {
        if self.episodes.is_empty() || k == 0 {
            return Vec::new();
        }
        // Score every incorrect episode
        let mut scored: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .filter(|(_, ep)| !ep.correct)
            .map(|(idx, ep)| {
                let sim = Episode::label_similarity(context_labels, &ep.context_labels);
                (idx, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // Bump importance of matched episodes (rehearsal)
        for &(idx, sim) in &scored {
            self.episodes[idx].importance =
                (self.episodes[idx].importance + sim * 0.1).min(2.0);
        }

        // Collect refs — borrow-safe because indices are unique
        scored
            .iter()
            .map(|&(idx, _)| &self.episodes[idx] as &Episode)
            .collect()
    }

    /// Retrieve top-k episodes (correct OR incorrect) most similar to context.
    /// Used by the annealer to find lowest-energy motifs.
    pub fn query_similar_any(
        &self,
        context_labels: &[String],
        k: usize,
    ) -> Vec<(f32, &Episode)> {
        if self.episodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(f32, &Episode)> = self
            .episodes
            .iter()
            .map(|ep| {
                let sim = Episode::label_similarity(context_labels, &ep.context_labels);
                (sim, ep)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Return number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

// ── Prediction Registry ───────────────────────────────────────────────────────
//
// The PredictionRegistry auto-resolves predictions without any per-script
// wiring.  Every time the fabric observes a new frame of labels, it:
//   1. Checks matured pending predictions against the frame's labels.
//   2. Emits resolved episodes into the EpisodicStore.
//   3. Returns the resolved episodes so the caller can apply Hebbian updates.
//
// Virtual sensors (chess training loop, QA runners, etc.) register explicit
// predictions via `NeuroRuntime::register_prediction()`.  Hardware sensors get
// auto-predictions from Hebbian propagation — the fabric predicts what label
// is most likely to appear next and registers that automatically.
//
// Resolution condition: a prediction resolves when the fabric sees a new frame
// that contains the expected outcome label (or enough steps have elapsed that
// a timeout-as-failure fires).

/// How many steps a prediction may wait before being timed-out as wrong.
const PREDICTION_TIMEOUT_STEPS: u64 = 64;

/// Confidence below which auto-propagated predictions are not registered
/// (avoids polluting the store with near-random guesses).
const PREDICTION_MIN_CONFIDENCE: f32 = 0.25;

/// One pending prediction waiting for resolution.
#[derive(Debug, Clone)]
struct PendingPrediction {
    /// Sorted, deduped labels that were active when the prediction was made.
    context_labels: Vec<String>,
    /// The label the fabric is predicting will appear.
    predicted: String,
    /// Stream names that contributed context labels.
    streams: Vec<String>,
    /// Pool step at which this prediction was registered.
    registered_at: u64,
    /// Model confidence in this prediction (0–1).
    p_confidence: f32,
    /// Optional resolution label override — for explicit registrations where
    /// the caller already knows the label to watch for (e.g. game outcome).
    /// When None, the fabric watches for `predicted` itself.
    resolve_on: Option<String>,
}

/// Circular-buffer prediction registry with automatic resolution.
#[derive(Debug)]
pub struct PredictionRegistry {
    pending: std::collections::VecDeque<PendingPrediction>,
    /// Maximum simultaneous pending predictions.
    max_pending: usize,
    /// Pool steps a prediction waits before being force-resolved as wrong.
    timeout_steps: u64,
}

impl PredictionRegistry {
    pub fn new() -> Self {
        Self {
            pending: std::collections::VecDeque::with_capacity(256),
            max_pending: 512,
            timeout_steps: PREDICTION_TIMEOUT_STEPS,
        }
    }

    /// Register a pending prediction.  Returns false if the registry is full
    /// (oldest is dropped to make room, so practically always succeeds).
    pub fn register(
        &mut self,
        context_labels: Vec<String>,
        predicted: String,
        streams: Vec<String>,
        p_confidence: f32,
        current_step: u64,
        resolve_on: Option<String>,
    ) -> bool {
        if p_confidence < PREDICTION_MIN_CONFIDENCE && resolve_on.is_none() {
            return false;
        }
        if self.pending.len() >= self.max_pending {
            self.pending.pop_front(); // drop oldest
        }
        self.pending.push_back(PendingPrediction {
            context_labels,
            predicted,
            streams,
            registered_at: current_step,
            p_confidence,
            resolve_on,
        });
        true
    }

    /// Check current active labels against pending predictions.
    ///
    /// Returns a list of resolved episodes — the caller applies Hebbian
    /// updates and records them into the EpisodicStore.
    ///
    /// Predictions that have timed out are also resolved (as wrong) so the
    /// episodic store learns "this context did not lead to the predicted outcome."
    pub fn tick(
        &mut self,
        active_labels: &[String],
        current_step: u64,
    ) -> Vec<ResolvedPrediction> {
        let active_set: std::collections::HashSet<&str> =
            active_labels.iter().map(|s| s.as_str()).collect();

        let mut resolved = Vec::new();
        let mut remaining = std::collections::VecDeque::with_capacity(self.pending.len());

        while let Some(pred) = self.pending.pop_front() {
            let watch_label = pred.resolve_on.as_deref().unwrap_or(&pred.predicted);
            let age = current_step.saturating_sub(pred.registered_at);

            let matched = active_set.contains(watch_label);
            let timed_out = age >= self.timeout_steps;

            if matched || timed_out {
                // Determine actual outcome
                let actual = if matched {
                    watch_label.to_string()
                } else {
                    // Timeout: what IS present that overlaps with prediction context?
                    // Use the most overlapping active label as the "actual".
                    active_labels
                        .iter()
                        .find(|l| pred.context_labels.contains(l))
                        .cloned()
                        .unwrap_or_else(|| "timeout".to_string())
                };

                // Focal-loss surprise: confident+correct→low, confident+wrong→high.
                let surprise = if matched {
                    (1.0 - pred.p_confidence).powi(2)
                } else {
                    pred.p_confidence.powi(2)
                };

                resolved.push(ResolvedPrediction {
                    context_labels: pred.context_labels,
                    predicted: pred.predicted,
                    actual,
                    streams: pred.streams,
                    correct: matched,
                    surprise,
                    p_confidence: pred.p_confidence,
                });
            } else {
                remaining.push_back(pred);
            }
        }

        self.pending = remaining;
        resolved
    }
}

/// The output of a successful auto-resolution from PredictionRegistry.
#[derive(Debug, Clone)]
pub struct ResolvedPrediction {
    pub context_labels: Vec<String>,
    pub predicted: String,
    pub actual: String,
    pub streams: Vec<String>,
    pub correct: bool,
    /// Focal-loss-inspired surprise score.
    pub surprise: f32,
    pub p_confidence: f32,
}

// ── Conditional Sufficiency Tracker ──────────────────────────────────────────
//
// Tracks per-context-signature entropy to detect when a context set makes
// outcomes conditionally independent — i.e. when the fabric knows enough.
//
// "When [king-side-castle, open-file, rook-active] all hold → outcome is
//  white-wins with p=0.91" — this context set is *sufficient* for that outcome.
//
// Once sufficiency is detected, the annealer biases proposals toward this
// motif (lowest-energy state) and the system stops exploring — saving CPU.

/// How many outcome observations to collect before computing entropy.
const SUFFICIENCY_MIN_SAMPLES: u64 = 8;
/// Entropy threshold below which we declare a context sufficient.
/// H < 0.35 bits ≈ p_max > 0.88 for a 3-class outcome.
const SUFFICIENCY_ENTROPY_THRESHOLD: f64 = 0.35;

/// Per-context-signature accumulator.
#[derive(Debug, Clone)]
struct SufficiencyBucket {
    /// Compressed key: sorted context labels joined by '|'
    context_key: String,
    /// Outcome label → count observed under this context.
    outcome_counts: HashMap<String, u64>,
    total: u64,
    /// Shannon entropy (nats) at last computation.
    entropy: f64,
    /// True once entropy has dropped below threshold.
    is_sufficient: bool,
}

impl SufficiencyBucket {
    fn new(context_key: String) -> Self {
        Self {
            context_key,
            outcome_counts: HashMap::new(),
            total: 0,
            entropy: f64::INFINITY,
            is_sufficient: false,
        }
    }

    fn observe(&mut self, outcome: &str) {
        *self.outcome_counts.entry(outcome.to_string()).or_insert(0) += 1;
        self.total += 1;
        if self.total >= SUFFICIENCY_MIN_SAMPLES {
            self.recompute_entropy();
        }
    }

    fn recompute_entropy(&mut self) {
        let n = self.total as f64;
        self.entropy = self
            .outcome_counts
            .values()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.ln()
            })
            .sum::<f64>();
        self.is_sufficient = self.entropy < SUFFICIENCY_ENTROPY_THRESHOLD;
    }
}

/// Conditional sufficiency tracker — detects minimal context sets where the
/// fabric's predictions become essentially deterministic.
#[derive(Debug)]
pub struct ConditionalSufficiencyTracker {
    /// context_key → bucket
    buckets: HashMap<String, SufficiencyBucket>,
    /// Sufficient context signatures found so far (for annealer biasing).
    pub sufficient_contexts: Vec<String>,
    /// Maximum number of buckets before pruning low-count ones.
    max_buckets: usize,
}

impl ConditionalSufficiencyTracker {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            sufficient_contexts: Vec::new(),
            max_buckets: 8192,
        }
    }

    /// Build the compressed context key from a set of active labels.
    /// We use only structurally informative labels (role, group, zone, attr)
    /// and ignore highly specific ones (id::, col::, region::) to keep the
    /// key space manageable.
    pub fn make_key(labels: &[String]) -> String {
        let mut filtered: Vec<&str> = labels
            .iter()
            .filter(|l| {
                l.starts_with("role::") || l.starts_with("group::")
                    || l.starts_with("zone::") || l.starts_with("attr::")
                    || l.starts_with("stream::") || l.starts_with("meta::")
            })
            .map(|s| s.as_str())
            .collect();
        filtered.sort();
        filtered.dedup();
        filtered.join("|")
    }

    /// Record an observed outcome under the given context.
    /// Returns true if this observation tips the context into sufficiency.
    pub fn observe(&mut self, context_labels: &[String], outcome: &str) -> bool {
        let key = Self::make_key(context_labels);
        if key.is_empty() {
            return false;
        }
        // Prune if over cap
        if self.buckets.len() >= self.max_buckets {
            // Remove the 10% of entries with lowest total observations
            let threshold = {
                let mut totals: Vec<u64> =
                    self.buckets.values().map(|b| b.total).collect();
                totals.sort();
                totals[totals.len() / 10]
            };
            self.buckets.retain(|_, b| b.total > threshold);
        }

        let bucket = self
            .buckets
            .entry(key.clone())
            .or_insert_with(|| SufficiencyBucket::new(key.clone()));

        let was_sufficient = bucket.is_sufficient;
        bucket.observe(outcome);

        if bucket.is_sufficient && !was_sufficient {
            if !self.sufficient_contexts.contains(&key) {
                self.sufficient_contexts.push(key);
            }
            return true; // newly sufficient
        }
        false
    }

    /// Return the entropy for a context key (None if not enough samples yet).
    pub fn entropy_for(&self, context_labels: &[String]) -> Option<f64> {
        let key = Self::make_key(context_labels);
        self.buckets.get(&key).map(|b| b.entropy)
    }

    /// True if we have enough observations to declare this context sufficient.
    pub fn is_sufficient(&self, context_labels: &[String]) -> bool {
        let key = Self::make_key(context_labels);
        self.buckets
            .get(&key)
            .map(|b| b.is_sufficient)
            .unwrap_or(false)
    }
}

#[derive(Debug, Default, Clone)]
struct PositionAccumulator {
    sum_x: f64,
    sum_y: f64,
    sum_z: f64,
    count: u64,
}

impl PositionAccumulator {
    fn add(&mut self, pos: &Position) {
        self.sum_x += pos.x;
        self.sum_y += pos.y;
        self.sum_z += pos.z;
        self.count += 1;
    }

    fn mean(&self) -> Option<Position> {
        if self.count == 0 {
            None
        } else {
            let inv = 1.0 / self.count as f64;
            Some(Position {
                x: self.sum_x * inv,
                y: self.sum_y * inv,
                z: self.sum_z * inv,
            })
        }
    }
}

#[derive(Debug)]
struct NeuroState {
    pool: NeuronPool,
    networks: Vec<NeuralNetwork>,
    label_to_network: HashMap<String, u32>,
    minicolumns: Vec<MiniColumn>,
    minicolumn_index: HashMap<String, usize>,
    minicolumn_counts: HashMap<String, u64>,
    positions: HashMap<String, PositionAccumulator>,
    network_cooccur: HashMap<(u32, u32), u64>,
    last_positions: HashMap<String, Position>,
    velocities: HashMap<String, Position>,
    prediction_error: HashMap<String, f64>,
    working_memory: Vec<String>,
    temporal_motif_counts: HashMap<String, u64>,
    /// Content-addressable episodic memory — resolved prediction episodes
    /// cross-tagged by stream source, retrieved on wrong predictions.
    episodic: EpisodicStore,
    /// Tracks per-context entropy to detect conditional sufficiency.
    sufficiency: ConditionalSufficiencyTracker,
    /// Auto-resolving prediction registry — predictions registered here are
    /// checked against every new observation frame automatically.
    registry: PredictionRegistry,
    /// Hierarchical motif discovery — observes every train_weighted call so
    /// recurring label-sequence patterns get promoted to meta-motifs.
    motifs: HierarchicalMotifRuntime,
}

impl NeuroState {
    fn new(config: NeuroConfig, motif_config: HierarchicalMotifConfig) -> Self {
        Self {
            pool: NeuronPool::new(config),
            networks: Vec::new(),
            label_to_network: HashMap::new(),
            minicolumns: Vec::new(),
            minicolumn_index: HashMap::new(),
            minicolumn_counts: HashMap::new(),
            positions: HashMap::new(),
            network_cooccur: HashMap::new(),
            last_positions: HashMap::new(),
            velocities: HashMap::new(),
            prediction_error: HashMap::new(),
            working_memory: Vec::new(),
            temporal_motif_counts: HashMap::new(),
            episodic: EpisodicStore::new(),
            sufficiency: ConditionalSufficiencyTracker::new(),
            registry: PredictionRegistry::new(),
            motifs: HierarchicalMotifRuntime::new(motif_config),
        }
    }

    fn record_state(
        &mut self,
        state: &DynamicState,
        symbols: &HashMap<String, Symbol>,
        min_activation: f32,
        module_threshold: u64,
        max_networks: usize,
        decay: f32,
        prediction_smoothing: f32,
        prediction_horizon: usize,
        curiosity_strength: f32,
        working_capacity: usize,
        snapshot_meta: Option<&HashMap<String, Value>>,
    ) {
        let mut observed: HashSet<String> = HashSet::with_capacity(state.symbol_states.len());
        let mut labels: Vec<String> = Vec::with_capacity(state.symbol_states.len() * 5 + 2);
        let mut zone_map: HashMap<String, Vec<u32>> = HashMap::new();
        let mut minicolumn_signatures: Vec<MinicolumnSignature> = Vec::new();
        for (symbol_id, symbol_state) in &state.symbol_states {
            observed.insert(symbol_id.clone());
            labels.push(format!("id::{symbol_id}"));
            labels.push(zone_label(&symbol_state.position));
            self.bump_position(&format!("id::{symbol_id}"), &symbol_state.position);
            self.bump_position(&zone_label(&symbol_state.position), &symbol_state.position);
            if let Some(symbol) = symbols.get(symbol_id) {
                let role = role_label(symbol);
                let group = group_label(symbol);
                let domain = domain_label(symbol);
                labels.push(format!("role::{role}"));
                labels.push(format!("group::{group}"));
                labels.push(format!("domain::{domain}"));
                labels.push(format!(
                    "col::{role}::{}",
                    zone_label(&symbol_state.position)
                ));
                labels.push(format!(
                    "region::{domain}::{}",
                    zone_label(&symbol_state.position)
                ));
                self.bump_position(&format!("role::{role}"), &symbol_state.position);
                self.bump_position(&format!("group::{group}"), &symbol_state.position);
                self.bump_position(&format!("domain::{domain}"), &symbol_state.position);
                self.bump_position(
                    &format!("col::{role}::{}", zone_label(&symbol_state.position)),
                    &symbol_state.position,
                );
                self.bump_position(
                    &format!("region::{domain}::{}", zone_label(&symbol_state.position)),
                    &symbol_state.position,
                );
                let phenotype =
                    phenotype_labels(symbol, self.pool.config.minicolumn_attr_limit);
                for (key, value) in &phenotype {
                    labels.push(format!("attr::{key}::{value}"));
                }
                if let Some(signature) = build_minicolumn_signature(
                    &role,
                    &phenotype,
                    self.pool.config.minicolumn_min_signature,
                ) {
                    minicolumn_signatures.push(signature);
                }
            }
            let id_label = format!("id::{symbol_id}");
            let id_idx = self.pool.get_or_create(&id_label);
            zone_map
                .entry(zone_label(&symbol_state.position))
                .or_default()
                .push(id_idx);
            self.update_temporal(symbol_id, &symbol_state.position, prediction_smoothing);
        }
        if !labels.is_empty() {
            let mut motif = labels
                .iter()
                .filter(|l| l.starts_with("role::") || l.starts_with("group::"))
                .cloned()
                .collect::<Vec<_>>();
            motif.sort();
            motif.dedup();
            if !motif.is_empty() {
                labels.push(format!("motif::{}", motif.join("|")));
            }
        }
        // ── Snapshot-level metadata → labels + influence context ──────────────
        // Every string-valued key in the snapshot metadata becomes:
        //   - a `meta::key::value` label in the fabric
        //   - a `stream::value` label if key == "source" or "stream"
        // These labels cross-wire with symbol labels through Hebbian learning,
        // so the fabric naturally discovers associations between positions,
        // motifs, and metadata contexts (openings, artists, players, etc.).
        let mut meta_context: Vec<(String, String)> = Vec::new();
        if let Some(meta) = snapshot_meta {
            for (key, value) in meta {
                let str_val = match value {
                    Value::String(s) => Some(s.as_str().to_string()),
                    Value::Number(n) => Some(n.to_string()),
                    Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                };
                if let Some(v) = str_val {
                    if v.is_empty() {
                        continue;
                    }
                    let clean_val = normalize_label_token(&v);
                    let clean_key = normalize_label_token(key);
                    if clean_key.is_empty() || clean_val.is_empty() {
                        continue;
                    }
                    // Stream label (source identifier)
                    if clean_key == "source" || clean_key == "stream" {
                        labels.push(format!("stream::{clean_val}"));
                        meta_context.push(("stream".to_string(), clean_val.clone()));
                    }
                    // Universal meta label — searchable without knowing the key
                    labels.push(format!("meta::{clean_key}::{clean_val}"));
                    meta_context.push((clean_key, clean_val));
                }
            }
        }

        labels.sort();
        labels.dedup();
        let meta_opt = if meta_context.is_empty() { None } else { Some(&meta_context) };
        self.pool.record_symbols_with_meta(&labels, meta_opt);
        self.update_minicolumns(&minicolumn_signatures);
        self.apply_zone_wta(&zone_map);
        self.apply_stdp();
        self.promote_networks(module_threshold, max_networks);
        self.refresh_network_strengths(min_activation, decay);
        self.promote_super_networks(module_threshold, max_networks, min_activation);
        self.refresh_cross_links(min_activation);
        self.apply_curiosity(curiosity_strength, &observed);
        self.update_working_memory(&labels, working_capacity);
        self.prune_temporal(prediction_horizon, &observed);

        // ── Prediction Registry: tick + auto-prediction ──────────────────────
        // 1. Resolve any matured pending predictions against this frame's labels.
        let current_step = self.pool.step;
        let resolved = self.registry.tick(&labels, current_step);

        // 2. For each resolved prediction: apply Hebbian update + episodic record.
        for rp in resolved {
            // Compute reward: positive for correct, negative for wrong.
            // Scale by surprise (focal-loss inspired) so confident failures
            // receive a proportionally stronger corrective signal.
            let reward_scale = if rp.correct { 1.0 + (1.0 - rp.surprise) } else { 1.0 + rp.surprise };
            if rp.correct {
                let scale = reward_scale.min(4.0);
                let mut ctx = rp.context_labels.clone();
                ctx.push(rp.actual.clone());
                self.pool.train_weighted(&ctx, scale, false);
                self.pool.train_weighted(&[rp.predicted.clone(), rp.actual.clone()], scale, false);
            } else {
                let scale = reward_scale.min(4.0);
                self.pool.train_weighted(&[rp.predicted.clone(), rp.actual.clone()], scale, true);
                if !rp.context_labels.is_empty() {
                    let mut ctx = rp.context_labels.clone();
                    ctx.push(rp.actual.clone());
                    self.pool.train_weighted(&ctx, 0.5, false);
                }
                // On wrong prediction: query episodic store for similar past
                // failures to surface the differentiating variable.
                // (The result is logged but acted on by the annealer externally.)
                let _similar = self.episodic.query_similar_failures(&rp.context_labels, 4);
            }
            // Feed sufficiency tracker.
            self.sufficiency.observe(&rp.context_labels, &rp.actual);
            // Record the resolved episode.
            let mut sorted_ctx = rp.context_labels.clone();
            sorted_ctx.sort();
            sorted_ctx.dedup();
            let importance = rp.surprise.max(0.01);
            self.episodic.record(Episode {
                context_labels: sorted_ctx,
                predicted: rp.predicted,
                actual: rp.actual,
                streams: rp.streams,
                timestamp: current_step,
                correct: rp.correct,
                surprise: rp.surprise,
                importance,
            });
        }

        // 3. Auto-prediction: propagate from current labels to guess next label.
        // Only register if we have enough active labels to make a meaningful guess.
        if labels.len() >= 3 {
            let propagated = self.pool.propagate(&labels, 2, min_activation);
            // Find the top propagated label that is NOT already in this frame.
            let label_set: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
            if let Some((predicted_label, strength)) = propagated
                .iter()
                .find(|(l, _)| !label_set.contains(l.as_str()))
            {
                // Extract stream names from meta labels in the current frame.
                let streams: Vec<String> = labels
                    .iter()
                    .filter_map(|l| l.strip_prefix("stream::").map(|s| s.to_string()))
                    .collect();
                let confidence = (*strength).clamp(0.0, 1.0);
                self.registry.register(
                    labels.clone(),
                    predicted_label.clone(),
                    streams,
                    confidence,
                    current_step,
                    None,
                );
            }
        }
    }

    fn bump_position(&mut self, label: &str, pos: &Position) {
        self.positions
            .entry(label.to_string())
            .or_default()
            .add(pos);
    }

    fn promote_networks(&mut self, threshold: u64, max_networks: usize) {
        if self.networks.len() >= max_networks {
            return;
        }
        let pairs = self.pool.cooccurrences_above(threshold);
        for ((a, b), count) in pairs {
            let label = format!("net::{a}|{b}");
            if self.label_to_network.contains_key(&label) {
                continue;
            }
            if self.networks.len() >= max_networks {
                break;
            }
            let a_id = self.pool.get_or_create(&a);
            let b_id = self.pool.get_or_create(&b);
            let net_id = self.networks.len() as u32;
            self.networks.push(NeuralNetwork {
                id: net_id,
                label: label.clone(),
                members: vec![a_id, b_id],
                born_at: self.pool.step_counter(),
                strength: count.sqrt().max(1.0),
                use_count: 0,
                level: 0,
                excites: HashMap::new(),
                inhibits: HashMap::new(),
            });
            self.label_to_network.insert(label, net_id);
        }
    }

    fn apply_zone_wta(&mut self, zones: &HashMap<String, Vec<u32>>) {
        let k = self.pool.config.wta_k_per_zone.max(1);
        for ids in zones.values() {
            if ids.len() <= k {
                continue;
            }
            let mut sorted = ids
                .iter()
                .filter_map(|id| {
                    self.pool.get_hot(*id as usize).map(|n| (n.activation, *id))
                })
                .collect::<Vec<_>>();
            sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let keep: HashSet<u32> = sorted.iter().take(k).map(|(_, id)| *id).collect();
            let inhibitory_scale = self.pool.config.inhibitory_scale;
            for (_, id) in sorted.iter().skip(k) {
                if let Some(neuron) = self.pool.get_hot_mut(*id as usize) {
                    neuron.activation *= inhibitory_scale;
                }
            }
            for id in &keep {
                let _ = self.pool.get_hot_mut(*id as usize); // no-op, lateral inhibition placeholder
            }
        }
    }

    fn update_minicolumns(&mut self, signatures: &[MinicolumnSignature]) {
        let config = self.pool.config.clone();
        if config.minicolumn_threshold == 0 || config.minicolumn_max == 0 {
            return;
        }
        let step = self.pool.step_counter();
        if signatures.is_empty() {
            let stability_decay = config.minicolumn_stability_decay.clamp(0.0, 1.0);
            let inhibition_decay = config.minicolumn_inhibition_decay.clamp(0.0, 1.0);
            let collapse_threshold = config.minicolumn_collapse_threshold.clamp(0.0, 1.0);
            for column in &mut self.minicolumns {
                column.stability *= stability_decay;
                column.inhibition *= inhibition_decay;
                column.collapsed = column.inhibition >= collapse_threshold;
                column.last_seen = step;
            }
            return;
        }

        for signature in signatures {
            let label = format!("mini::{}", signature.key);
            let count = self
                .minicolumn_counts
                .entry(label.clone())
                .or_insert(0);
            *count = count.saturating_add(1);
            if *count < config.minicolumn_threshold {
                continue;
            }
            if self.minicolumn_index.contains_key(&label) {
                continue;
            }
            if self.minicolumns.len() >= config.minicolumn_max {
                continue;
            }
            let id = self.pool.get_or_create(&label);
            let members = signature
                .labels
                .iter()
                .map(|label| self.pool.get_or_create(label))
                .collect::<Vec<_>>();
            let column = MiniColumn {
                id,
                label,
                labels: signature.labels.clone(),
                attr_map: signature.attr_map.clone(),
                members,
                stability: 0.0,
                inhibition: 0.0,
                born_at: step,
                last_seen: step,
                collapsed: false,
                activation_dirty: true,
                cached_evidence: 0.0,
                cached_conflict: 0.0,
            };
            self.minicolumn_index
                .insert(column.label.clone(), self.minicolumns.len());
            self.minicolumns.push(column);
        }

        let stability_decay = config.minicolumn_stability_decay.clamp(0.0, 1.0);
        let inhibition_decay = config.minicolumn_inhibition_decay.clamp(0.0, 1.0);
        let activation_threshold = config.minicolumn_activation_threshold.clamp(0.0, 1.0);
        let collapse_threshold = config.minicolumn_collapse_threshold.clamp(0.0, 1.0);
        let inhibit_scale = config.minicolumn_inhibit_scale.clamp(0.0, 1.0);

        // Build a flat set of all labels in the current signature batch for fast membership test.
        let sig_label_set: HashSet<&str> = signatures
            .iter()
            .flat_map(|s| s.labels.iter().map(|l| l.as_str()))
            .collect();

        for column in &mut self.minicolumns {
            // Fast-path: if no label in this column appears in the current signatures,
            // the column is unaffected this frame — skip the expensive match and let
            // stability/inhibition decay naturally.  Mark not-dirty so we use the cached zeros.
            let any_label_present = column.labels.iter().any(|l| sig_label_set.contains(l.as_str()));
            let (evidence, conflict) = if any_label_present || column.activation_dirty {
                let result = best_signature_match(column, signatures);
                column.cached_evidence = result.0;
                column.cached_conflict = result.1;
                column.activation_dirty = false;
                result
            } else {
                (column.cached_evidence, column.cached_conflict)
            };
            column.stability =
                column.stability * stability_decay + evidence * (1.0 - stability_decay);
            column.inhibition =
                column.inhibition * inhibition_decay + conflict * (1.0 - inhibition_decay);
            column.last_seen = step;
            column.collapsed = column.inhibition >= collapse_threshold;
            let active = column.stability >= activation_threshold && !column.collapsed;
            let column_activation = (column.stability * (1.0 - column.inhibition)).clamp(0.0, 1.0);
            if let Some(neuron) = self.pool.get_hot_mut(column.id as usize) {
                neuron.activation = neuron.activation.max(column_activation);
                neuron.use_count = neuron.use_count.saturating_add(active as u64);
            }
            if active {
                for member_id in &column.members {
                    if let Some(neuron) = self.pool.get_hot_mut(*member_id as usize) {
                        neuron.activation *= inhibit_scale;
                    }
                }
                // Members changed; recompute on next frame.
                column.activation_dirty = true;
            }
        }
    }

    fn refresh_network_strengths(&mut self, min_activation: f32, decay: f32) {
        let active = self.pool.active_ids(min_activation);
        let mut active_nets = Vec::new();
        for net in &mut self.networks {
            net.strength *= decay;
            if net.members.iter().all(|m| active.contains(m)) {
                net.use_count += 1;
                net.strength += 0.5;
                active_nets.push(net.id);
            }
            net.strength = net.strength.clamp(0.0, 16.0);
        }
        active_nets.sort();
        for i in 0..active_nets.len() {
            for j in (i + 1)..active_nets.len() {
                let key = (active_nets[i], active_nets[j]);
                *self.network_cooccur.entry(key).or_insert(0) += 1;
            }
        }
    }

    fn snapshot(&self, min_activation: f32, prediction_horizon: usize) -> NeuroSnapshot {
        let active_labels = self.pool.active_labels(min_activation);
        let active_composites = active_labels
            .iter()
            .filter(|l| l.starts_with("comp::"))
            .cloned()
            .collect::<HashSet<_>>();
        let centroids = self
            .positions
            .iter()
            .filter_map(|(label, acc)| acc.mean().map(|m| (label.clone(), m)))
            .collect::<HashMap<_, _>>();
        let mut active_networks = Vec::new();
        let mut network_links: HashMap<String, HashMap<String, f32>> = HashMap::new();
        for net in &self.networks {
            if net.strength <= 0.01 {
                continue;
            }
            let members = net
                .members
                .iter()
                .filter_map(|id| self.pool.label_for(*id))
                .collect::<Vec<_>>();
            active_networks.push(NeuralNetworkSnapshot {
                label: net.label.clone(),
                members,
                strength: net.strength,
                level: net.level,
            });
            let mut links = HashMap::new();
            for (tgt, w) in net.excites.iter() {
                if let Some(label) = self
                    .networks
                    .iter()
                    .find(|n| n.id == *tgt)
                    .map(|n| n.label.clone())
                {
                    links.insert(label, *w);
                }
            }
            if !links.is_empty() {
                network_links.insert(net.label.clone(), links);
            }
        }
        let minicolumns = self
            .minicolumns
            .iter()
            .map(|column| MinicolumnSnapshot {
                label: column.label.clone(),
                stability: column.stability,
                inhibition: column.inhibition,
                collapsed: column.collapsed,
                members: column.members.len(),
                born_at: column.born_at,
            })
            .collect::<Vec<_>>();
        let mut temporal_predictions = HashMap::new();
        for (symbol_id, last_pos) in &self.last_positions {
            if let Some(vel) = self.velocities.get(symbol_id) {
                let horizon = prediction_horizon.max(1) as f64;
                let predicted = Position {
                    x: last_pos.x + vel.x * horizon,
                    y: last_pos.y + vel.y * horizon,
                    z: last_pos.z + vel.z * horizon,
                };
                temporal_predictions.insert(symbol_id.clone(), predicted);
            }
        }
        let prediction_error = self.prediction_error.clone();
        let prediction_confidence = self
            .prediction_error
            .iter()
            .map(|(k, err)| (k.clone(), 1.0 / (1.0 + *err)))
            .collect::<HashMap<_, _>>();
        let surprise = self
            .prediction_error
            .iter()
            .map(|(k, err)| (k.clone(), (*err).min(5.0)))
            .collect::<HashMap<_, _>>();
        let total_motifs: u64 = self.temporal_motif_counts.values().sum();
        let temporal_motif_priors = if total_motifs > 0 {
            self.temporal_motif_counts
                .iter()
                .map(|(k, v)| (k.clone(), *v as f64 / total_motifs as f64))
                .collect()
        } else {
            HashMap::new()
        };
        // Extract active streams and meta labels from active label set
        let mut active_streams: HashSet<String> = HashSet::new();
        let mut active_meta_labels: HashMap<String, String> = HashMap::new();
        for label in &active_labels {
            if let Some(stream) = label.strip_prefix("stream::") {
                active_streams.insert(stream.to_string());
            } else if let Some(rest) = label.strip_prefix("meta::") {
                // meta::key::value
                if let Some((key, value)) = rest.split_once("::") {
                    active_meta_labels.insert(key.to_string(), value.to_string());
                }
            }
        }

        // Collect top influences from most active neurons (top 20 by activation)
        // Only hot neurons can be active — cold neurons have decayed activation.
        let mut neuron_activations: Vec<(f32, &Neuron)> = self
            .pool
            .neurons
            .iter()
            .filter_map(|slot| slot.as_hot())
            .filter(|n| n.activation >= min_activation && !n.influence_history.is_empty())
            .map(|n| (n.activation, n))
            .collect();
        neuron_activations
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // Aggregate influence records across top neurons, merging by stream+labels
        let mut influence_map: HashMap<String, InfluenceRecord> = HashMap::new();
        for (activation, neuron) in neuron_activations.iter().take(20) {
            for rec in &neuron.influence_history {
                let key = format!("{}::{}", rec.stream, rec.labels.iter().map(|(k,v)| format!("{k}={v}")).collect::<Vec<_>>().join(","));
                let entry = influence_map.entry(key).or_insert_with(|| InfluenceRecord {
                    stream: rec.stream.clone(),
                    labels: rec.labels.clone(),
                    strength: 0.0,
                    step: rec.step,
                });
                entry.strength += rec.strength * activation;
                entry.step = entry.step.max(rec.step);
            }
        }
        let mut top_influences: Vec<InfluenceRecord> = influence_map.into_values().collect();
        top_influences
            .sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        top_influences.truncate(16);

        NeuroSnapshot {
            active_labels,
            active_composites,
            active_networks,
            minicolumns,
            centroids,
            network_links,
            temporal_predictions,
            prediction_error,
            prediction_confidence,
            surprise,
            working_memory: self.working_memory.clone(),
            temporal_motif_priors,
            active_streams,
            active_meta_labels,
            top_influences,
        }
    }

    fn update_temporal(&mut self, symbol_id: &str, pos: &Position, smoothing: f32) {
        let last = self.last_positions.get(symbol_id).cloned();
        let vel = self
            .velocities
            .entry(symbol_id.to_string())
            .or_insert_with(Position::default);
        let alpha = smoothing.clamp(0.0, 1.0) as f64;
        if let Some(last_pos) = last {
            let dx = pos.x - last_pos.x;
            let dy = pos.y - last_pos.y;
            let dz = pos.z - last_pos.z;
            vel.x = vel.x * (1.0 - alpha) + dx * alpha;
            vel.y = vel.y * (1.0 - alpha) + dy * alpha;
            vel.z = vel.z * (1.0 - alpha) + dz * alpha;
            let err = (dx * dx + dy * dy + dz * dz).sqrt();
            let entry = self
                .prediction_error
                .entry(symbol_id.to_string())
                .or_insert(0.0);
            *entry = *entry * (1.0 - alpha) + err * alpha;
        } else {
            vel.x = 0.0;
            vel.y = 0.0;
            vel.z = 0.0;
            self.prediction_error
                .entry(symbol_id.to_string())
                .or_insert(0.0);
        }
        self.last_positions.insert(symbol_id.to_string(), *pos);
    }

    fn prune_temporal(&mut self, _horizon: usize, observed: &HashSet<String>) {
        // Drop stale entries not seen in this step.
        self.last_positions.retain(|k, _| observed.contains(k));
        self.velocities.retain(|k, _| observed.contains(k));
        self.prediction_error.retain(|k, _| observed.contains(k));
    }

    fn update_working_memory(&mut self, labels: &[String], capacity: usize) {
        if capacity == 0 {
            self.working_memory.clear();
            return;
        }
        let mut motif = labels
            .iter()
            .filter(|l| {
                l.starts_with("role::") || l.starts_with("group::") || l.starts_with("zone::")
            })
            .cloned()
            .collect::<Vec<_>>();
        motif.sort();
        motif.dedup();
        if motif.is_empty() {
            return;
        }
        let motif_label = format!("wm::{}", motif.join("|"));
        self.working_memory.push(motif_label.clone());
        if self.working_memory.len() > capacity {
            let overflow = self.working_memory.len() - capacity;
            self.working_memory.drain(0..overflow);
        }
        *self.temporal_motif_counts.entry(motif_label).or_insert(0) += 1;
    }

    fn apply_curiosity(&mut self, strength: f32, observed: &HashSet<String>) {
        if strength <= 0.0 {
            return;
        }
        for symbol_id in observed {
            let surprise = *self.prediction_error.get(symbol_id).unwrap_or(&0.0);
            if surprise <= 0.1 {
                continue;
            }
            let label = format!("id::{symbol_id}");
            let zone = self
                .last_positions
                .get(symbol_id)
                .map(zone_label)
                .unwrap_or_else(|| "zone::0,0".to_string());
            let a = self.pool.get_or_create(&label);
            let b = self.pool.get_or_create(&zone);
            let scaled = strength * surprise.min(3.0) as f32;
            self.pool
                .hebbian_pair(a, b, self.pool.config.excitatory_scale * scaled, false);
        }
    }

    fn promote_super_networks(&mut self, threshold: u64, max_networks: usize, min_activation: f32) {
        if self.networks.len() >= max_networks {
            return;
        }
        let active = self
            .networks
            .iter()
            .filter(|n| n.strength >= min_activation)
            .map(|n| n.id)
            .collect::<Vec<_>>();
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let key = (active[i], active[j]);
                let count = *self.network_cooccur.get(&key).unwrap_or(&0);
                if count >= threshold && self.networks.len() < max_networks {
                    let a = active[i];
                    let b = active[j];
                    let label = format!("super::{}&{}", a, b);
                    if self.label_to_network.contains_key(&label) {
                        continue;
                    }
                    let members = self
                        .networks
                        .iter()
                        .filter(|n| n.id == a || n.id == b)
                        .flat_map(|n| n.members.iter().cloned())
                        .collect::<Vec<_>>();
                    let net_id = self.networks.len() as u32;
                    self.networks.push(NeuralNetwork {
                        id: net_id,
                        label: label.clone(),
                        members,
                        born_at: self.pool.step_counter(),
                        strength: (count as f32).sqrt().max(1.0),
                        use_count: 0,
                        level: 1,
                        excites: HashMap::new(),
                        inhibits: HashMap::new(),
                    });
                    self.label_to_network.insert(label, net_id);
                }
            }
        }
    }

    fn refresh_cross_links(&mut self, min_activation: f32) {
        let active = self
            .networks
            .iter()
            .filter(|n| n.strength >= min_activation)
            .map(|n| n.id)
            .collect::<Vec<_>>();
        let mut updates: Vec<(u32, u32, f32)> = Vec::new();
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let (a, b) = (active[i], active[j]);
                updates.push((a, b, 0.05));
                updates.push((b, a, 0.05));
            }
        }
        for (src, dst, delta) in updates {
            if let Some(net) = self.networks.iter_mut().find(|n| n.id == src) {
                *net.excites.entry(dst).or_insert(0.0) += delta;
            }
        }
    }

    fn apply_stdp(&mut self) {
        let scale = self.pool.config.stdp_scale;
        let len = self.pool.neurons.len();
        for idx in 0..len {
            // Skip cold/free neurons — they have no activation and no synapses in RAM.
            let (pre_trace, pre_fatigue) = match self.pool.get_hot(idx) {
                Some(pre) => (pre.trace as f32, pre.fatigue),
                None => continue,
            };
            let excit_updates = {
                let neuron = match self.pool.get_hot(idx) { Some(n) => n, None => continue };
                neuron.excitatory.iter().map(|syn| {
                    let post_act = self.pool.get_hot(syn.target as usize)
                        .map(|p| p.activation as f32).unwrap_or(0.0);
                    let delta = scale * (pre_trace * post_act - pre_fatigue * syn.weight * 0.1);
                    (syn.target, delta, syn.inhibitory)
                }).collect::<Vec<_>>()
            };
            let inhib_updates = {
                let neuron = match self.pool.get_hot(idx) { Some(n) => n, None => continue };
                neuron.inhibitory.iter().map(|syn| {
                    let post_act = self.pool.get_hot(syn.target as usize)
                        .map(|p| p.activation as f32).unwrap_or(0.0);
                    let delta = scale * (pre_trace * post_act - pre_fatigue * syn.weight * 0.1);
                    (syn.target, delta, syn.inhibitory)
                }).collect::<Vec<_>>()
            };
            if let Some(neuron) = self.pool.get_hot_mut(idx) {
                for (syn, (_, delta, _)) in neuron.excitatory.iter_mut().zip(excit_updates.iter()) {
                    syn.weight = (syn.weight + delta).clamp(-2.0, 2.0);
                }
                for (syn, (_, delta, _)) in neuron.inhibitory.iter_mut().zip(inhib_updates.iter()) {
                    syn.weight = (syn.weight + delta).clamp(-2.0, 2.0);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct NeuroRuntime {
    config: NeuroRuntimeConfig,
    symbol_lookup: HashMap<String, Symbol>,
    inner: Arc<Mutex<NeuroState>>,
}

pub type NeuroRuntimeHandle = Arc<NeuroRuntime>;

impl NeuroRuntime {
    pub fn new(snapshot: &EnvironmentSnapshot, config: NeuroRuntimeConfig) -> Self {
        let symbol_lookup = snapshot
            .symbols
            .iter()
            .cloned()
            .map(|s| (s.id.clone(), s))
            .collect::<HashMap<_, _>>();
        let hw = HardwareProfile::detect();
        let hw_cap = hw.cooccur_cap();
        let mut state = NeuroState::new(config.neuro.clone(), config.motifs.clone());
        state.pool.cooccur_cap = hw_cap;

        // Derive cold-tier directory alongside the pool checkpoint file.
        let cold_dir = config.pool_state_path.as_deref().map(|p| {
            let pb = std::path::Path::new(p);
            let stem = pb.file_stem().unwrap_or_default().to_string_lossy();
            pb.parent().unwrap_or(std::path::Path::new("."))
                .join(format!("{stem}_cold"))
        });

        // Auto-load pool if a state path is configured and the file exists.
        if let Some(ref path) = config.pool_state_path {
            let pool_path = std::path::Path::new(path);
            if pool_path.exists() {
                // Use a streaming reader for large files to avoid double-buffering.
                match std::fs::File::open(pool_path) {
                    Ok(file) => match serde_json::from_reader::<_, NeuronPool>(std::io::BufReader::new(file)) {
                        Ok(mut pool) => {
                            pool.cooccur_cap = hw_cap;
                            // Wire hot/cold paging now that we have the path.
                            if let Some(ref cd) = cold_dir {
                                let idle = config.neuro.eviction_idle_steps
                                    .unwrap_or(DEFAULT_EVICTION_IDLE_STEPS);
                                let hot_max = config.neuro.hot_tier_max
                                    .unwrap_or(DEFAULT_HOT_TIER_MAX);
                                pool.set_cold_dir(cd.clone(), idle, hot_max);
                                // If cold_dir exists from a previous run, restore Cold slots.
                                pool.restore_cold_index(cd);
                            }
                            let hot = pool.hot_count;
                            let total = pool.neurons.len();
                            tracing::info!("Loaded NeuronPool from {path}: {hot} hot / {total} total slots");
                            state.pool = pool;
                        }
                        Err(e) => tracing::warn!("Failed to parse pool state at {path}: {e}"),
                    },
                    Err(e) => tracing::warn!("Failed to read pool state at {path}: {e}"),
                }
            } else if let Some(ref cd) = cold_dir {
                // No hot-tier checkpoint but cold dir may exist (restart after OOM).
                if cd.exists() {
                    let idle = config.neuro.eviction_idle_steps.unwrap_or(DEFAULT_EVICTION_IDLE_STEPS);
                    let hot_max = config.neuro.hot_tier_max.unwrap_or(DEFAULT_HOT_TIER_MAX);
                    state.pool.set_cold_dir(cd.clone(), idle, hot_max);
                    state.pool.restore_cold_index(cd);
                    tracing::info!("Restored cold-only pool from {}", cd.display());
                }
            }
        }
        // Wire paging even if no checkpoint loaded (fresh start).
        if state.pool.cold_dir.is_none() {
            if let Some(ref cd) = cold_dir {
                let idle = config.neuro.eviction_idle_steps.unwrap_or(DEFAULT_EVICTION_IDLE_STEPS);
                let hot_max = config.neuro.hot_tier_max.unwrap_or(DEFAULT_HOT_TIER_MAX);
                state.pool.set_cold_dir(cd.clone(), idle, hot_max);
            }
        }

        Self {
            config,
            symbol_lookup,
            inner: Arc::new(Mutex::new(state)),
        }
    }

    /// Serialize the NeuronPool to disk at the configured path (or the given override path).
    /// Returns an error string on failure.
    pub fn save_pool(&self) -> Result<(), String> {
        let path = self
            .config
            .pool_state_path
            .as_deref()
            .ok_or_else(|| "pool_state_path not configured".to_string())?;
        self.save_pool_to(path)
    }

    pub fn save_pool_to(&self, path: &str) -> Result<(), String> {
        let guard = self.inner.lock();
        // The custom pool_serde serializer emits only Hot neurons, so checkpoints
        // stay small regardless of how many Cold neurons are on disk.
        let json = serde_json::to_string(&guard.pool)
            .map_err(|e| format!("serialize error: {e}"))?;
        // Persist cold-tier index alongside the checkpoint.
        guard.pool.save_cold_index();
        drop(guard);
        // Atomic write: write to .tmp then rename.
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("write error: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("rename error: {e}"))?;
        Ok(())
    }

    pub fn pool_state_path(&self) -> Option<&str> {
        self.config.pool_state_path.as_deref()
    }

    pub fn observe_states<'a, I>(&self, states: I)
    where
        I: IntoIterator<Item = &'a DynamicState>,
    {
        if !self.config.enabled {
            return;
        }
        let mut guard = self.inner.lock();
        guard.pool.step();
        guard.episodic.step();
        for state in states {
            guard.record_state(
                state,
                &self.symbol_lookup,
                self.config.min_activation,
                self.config.module_threshold,
                self.config.max_networks,
                self.config.neuro.decay,
                self.config.prediction_smoothing,
                self.config.prediction_horizon,
                self.config.curiosity_strength,
                self.config.working_memory,
                None,
            );
        }
    }

    pub fn observe_snapshot(&self, snapshot: &EnvironmentSnapshot) {
        if !self.config.enabled {
            return;
        }
        let symbol_lookup = snapshot
            .symbols
            .iter()
            .cloned()
            .map(|s| (s.id.clone(), s))
            .collect::<HashMap<_, _>>();
        let state = dynamic_state_from_snapshot(snapshot);
        let mut guard = self.inner.lock();
        guard.pool.step();
        let meta = if snapshot.metadata.is_empty() {
            None
        } else {
            Some(&snapshot.metadata)
        };
        guard.record_state(
            &state,
            &symbol_lookup,
            self.config.min_activation,
            self.config.module_threshold,
            self.config.max_networks,
            self.config.neuro.decay,
            self.config.prediction_smoothing,
            self.config.prediction_horizon,
            self.config.curiosity_strength,
            self.config.working_memory,
            meta,
        );
    }

    /// Apply a reward-weighted Hebbian training signal directly to the pool.
    ///
    /// Called by the `FabricTrainer` at the end of each batch to close the
    /// feedback loop: every outcome, quantum result, and human label becomes
    /// a weight update in the NeuronPool.
    pub fn train_weighted(&self, symbols: &[String], lr_scale: f32, inhibitory: bool) {
        if !self.config.enabled || symbols.is_empty() {
            return;
        }
        let mut guard = self.inner.lock();
        guard.pool.train_weighted(symbols, lr_scale, inhibitory);

        // Feed every training call into the hierarchical motif runtime so
        // recurring label-sequence patterns are discovered continuously.
        let ts = Timestamp {
            unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
        };
        // Intra-call: observe all labels as a sorted sequence — discovers
        // repeated sub-sequences within each training call.
        let duration_per = if symbols.is_empty() { 1.0 } else { 1.0 / symbols.len() as f64 };
        guard.motifs.observe_label_sequence(symbols, ts, duration_per);

        // Inter-call: observe the call's fingerprint as a single level-0 motif
        // so the motif runtime also learns which sequences of CALLS repeat.
        // e.g. "cell-membrane call" → "osmosis call" → "ATP call" across K-12 pages.
        let fp = call_fingerprint(symbols);
        guard.motifs.observe_label_sequence(&[fp], ts, 1.0);
    }

    /// Return all meta-motifs discovered so far across every hierarchy level.
    /// Empty until enough recurring label patterns cross `min_support`.
    pub fn meta_motifs(&self) -> Vec<MetaMotif> {
        if !self.config.enabled {
            return Vec::new();
        }
        let guard = self.inner.lock();
        guard.motifs.meta_motifs()
    }

    /// Given a label that appeared recently, predict which labels are likely
    /// to appear next based on learned transition probabilities.
    /// Returns `(label, probability)` pairs sorted by probability descending.
    pub fn motif_predictions(&self, last_label: &str) -> Vec<(String, f64)> {
        if !self.config.enabled {
            return Vec::new();
        }
        let guard = self.inner.lock();
        guard.motifs.next_predictions(last_label)
    }

    pub fn snapshot(&self) -> NeuroSnapshot {
        if !self.config.enabled {
            return NeuroSnapshot::default();
        }
        let guard = self.inner.lock();
        guard.snapshot(self.config.min_activation, self.config.prediction_horizon)
    }

    /// Propagate activation from `input_labels` through the learned Hebbian
    /// weight graph and return only the labels belonging to `target_stream`.
    ///
    /// This is the single-frame cross-modal inference call.
    ///
    /// Example: feed a set of visual labels for a frame showing a mouth opening
    /// → receive audio labels that were Hebbianly linked during training.
    ///
    /// `hops` controls how many synapse-hops to walk (2–4 is typical; higher
    /// reaches more abstract associations at the cost of specificity).
    pub fn cross_stream_activate(
        &self,
        input_labels: &[String],
        target_stream: &str,
        hops: usize,
    ) -> Vec<CrossStreamActivation> {
        if !self.config.enabled || input_labels.is_empty() {
            return Vec::new();
        }
        let guard = self.inner.lock();
        let propagated = guard.pool.propagate(input_labels, hops, self.config.min_activation);

        // Filter to labels that belong to the target stream.
        // Stream membership is encoded in two ways:
        //   1. Label literally starts with "stream::<target_stream>"
        //   2. Label starts with "meta::stream::<target_stream>"
        //   3. The neuron's influence_history has a record with stream == target_stream
        let stream_prefix_a = format!("stream::{target_stream}");
        let stream_prefix_b = format!("meta::stream::{target_stream}");

        let mut results: Vec<CrossStreamActivation> = propagated
            .into_iter()
            .filter(|(label, _)| {
                label.starts_with(&stream_prefix_a)
                    || label.starts_with(&stream_prefix_b)
                    || guard.pool.label_to_id.get(label).and_then(|&id| {
                        guard.pool.get_hot(id as usize)
                    }).map(|n| {
                        n.influence_history.iter().any(|r| r.stream == target_stream)
                    }).unwrap_or(false)
            })
            .map(|(label, strength)| {
                // Collect top influences for this label from its neuron's history
                let influences = guard.pool.label_to_id.get(&label)
                    .and_then(|&id| guard.pool.get_hot(id as usize))
                    .map(|n| {
                        let mut inf = n.influence_history.clone();
                        inf.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
                        inf.truncate(4);
                        inf
                    })
                    .unwrap_or_default();
                CrossStreamActivation {
                    label,
                    strength,
                    influences,
                }
            })
            .collect();

        results.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Run cross-stream reconstruction over a sequence of input frames.
    ///
    /// This is the temporal version of `cross_stream_activate`.  Each element
    /// of `input_frames` is a set of labels from one time-step of the input
    /// stream (e.g. visual labels for frame 0, frame 1, …).  The method walks
    /// them in order and for each frame propagates through the weight graph to
    /// produce the corresponding target-stream activations.
    ///
    /// Temporal coherence: a fraction (`carry`) of the previous frame's output
    /// activations is added as context for the next frame.  This causes the
    /// output sequence to flow smoothly — if mouth-open activates a growl sound
    /// pattern at frame 5, that pattern persists into frame 6 rather than
    /// snapping off instantly, mirroring how real sound and motion are linked
    /// across time.
    ///
    /// # Example
    /// ```
    /// let frames: Vec<Vec<String>> = video_frames   // each = list of visual labels
    ///     .iter().map(|f| f.visual_labels.clone()).collect();
    /// let audio_sequence = runtime.reconstruct_sequence(&frames, "audio", 3, 0.25);
    /// // audio_sequence[i] = the audio labels the network predicts for video frame i
    /// ```
    pub fn reconstruct_sequence(
        &self,
        input_frames: &[Vec<String>],
        target_stream: &str,
        hops: usize,
        carry: f32,           // 0.0 = no temporal bleed; ignored when dynamic carry is computed
    ) -> SequenceReconstruction {
        if !self.config.enabled || input_frames.is_empty() {
            return SequenceReconstruction::default();
        }

        let _carry_hint = carry.clamp(0.0, 0.9); // kept for API compat; dynamic carry used below
        let guard = self.inner.lock();
        let min_act = self.config.min_activation;

        // Previous frame's propagated activations, re-injected as context
        let mut prev_output: HashMap<String, f32> = HashMap::new();

        let mut frames: Vec<CrossStreamFrame> = Vec::with_capacity(input_frames.len());

        for (frame_idx, input_labels) in input_frames.iter().enumerate() {
            // Dynamic carry: cosine similarity between this frame's labels and the previous
            // frame's labels.  High similarity → persistent context; low similarity → fresh start.
            let dynamic_carry = if frame_idx == 0 {
                0.0f32
            } else {
                let prev_labels = &input_frames[frame_idx - 1];
                let a_set: std::collections::HashSet<&str> =
                    prev_labels.iter().map(|s| s.as_str()).collect();
                let b_set: std::collections::HashSet<&str> =
                    input_labels.iter().map(|s| s.as_str()).collect();
                let intersection = a_set.intersection(&b_set).count() as f32;
                let denom = (a_set.len() as f32 * b_set.len() as f32).sqrt();
                if denom > 0.0 {
                    (intersection / denom).clamp(0.0, 0.95)
                } else {
                    0.0
                }
            };
            let effective_carry = dynamic_carry;

            // Build seed: current frame labels + carried-over context from prev frame output
            let mut seed_labels: Vec<String> = input_labels.clone();
            // Add carry context — inject prev output labels back as additional seeds
            // scaled by carry factor (they'll start at `carry` activation not 1.0)
            let carry_seed: Vec<String> = prev_output
                .iter()
                .filter(|&(_, &v)| v * effective_carry >= min_act)
                .map(|(k, _)| k.clone())
                .collect();
            seed_labels.extend(carry_seed);
            seed_labels.sort();
            seed_labels.dedup();

            // Propagate
            let propagated = guard.pool.propagate(&seed_labels, hops, min_act);

            // Filter and build output for this frame (same logic as cross_stream_activate)
            let stream_prefix_a = format!("stream::{target_stream}");
            let stream_prefix_b = format!("meta::stream::{target_stream}");

            let mut output: Vec<CrossStreamActivation> = propagated
                .iter()
                .filter(|(label, _)| {
                    label.starts_with(&stream_prefix_a)
                        || label.starts_with(&stream_prefix_b)
                        || guard.pool.label_to_id.get(*label).and_then(|&id| {
                            guard.pool.get_hot(id as usize)
                        }).map(|n| {
                            n.influence_history.iter().any(|r| r.stream == target_stream)
                        }).unwrap_or(false)
                })
                .map(|(label, &strength)| CrossStreamActivation {
                    label: label.clone(),
                    strength,
                    influences: guard.pool.label_to_id.get(label)
                        .and_then(|&id| guard.pool.get_hot(id as usize))
                        .map(|n| {
                            let mut inf = n.influence_history.clone();
                            inf.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
                            inf.truncate(4);
                            inf
                        })
                        .unwrap_or_default(),
                })
                .collect();

            output.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));

            // Build next frame's carry context
            prev_output = propagated
                .into_iter()
                .filter(|(label, _)| {
                    label.starts_with(&stream_prefix_a) || label.starts_with(&stream_prefix_b)
                        || guard.pool.label_to_id.get(label).and_then(|&id| {
                            guard.pool.get_hot(id as usize)
                        }).map(|n| {
                            n.influence_history.iter().any(|r| r.stream == target_stream)
                        }).unwrap_or(false)
                })
                .collect();

            frames.push(CrossStreamFrame {
                frame_index: frame_idx,
                input_labels: input_labels.clone(),
                output: output,
            });
        }

        SequenceReconstruction {
            target_stream: target_stream.to_string(),
            hops,
            carry,
            frames,
        }
    }

    /// Propagate activation from seed labels through all learned connections and
    /// return every label that fires above `min_activation`, regardless of which
    /// stream it belongs to.
    ///
    /// This is the "read state" call: give it one modality's labels, get back
    /// every associated label — including those from other modalities that were
    /// co-trained at the same timestamps.  No weights are changed.
    ///
    /// `min_activation`: use a low value (0.02–0.1) for exploratory playback
    /// queries. The pool's internal `config.min_activation` (default 0.55) is
    /// calibrated for live streaming, not for sparse training demos.
    pub fn propagate_all(
        &self,
        seed_labels: &[String],
        hops: usize,
    ) -> HashMap<String, f32> {
        self.propagate_all_threshold(seed_labels, hops, self.config.min_activation)
    }

    /// Same as `propagate_all` but with an explicit minimum activation threshold.
    pub fn propagate_all_threshold(
        &self,
        seed_labels: &[String],
        hops: usize,
        min_activation: f32,
    ) -> HashMap<String, f32> {
        if !self.config.enabled || seed_labels.is_empty() {
            return HashMap::new();
        }
        let guard = self.inner.lock();
        guard.pool.propagate(seed_labels, hops, min_activation)
    }

    // ── Prediction Registry + Episodic Memory API ────────────────────────────

    /// Explicitly register a pending prediction for a virtual sensor.
    ///
    /// This is the entry point for virtual sensors (chess training loop, QA
    /// runners, chemistry simulations, etc.) to participate in the same
    /// auto-resolve feedback loop as hardware sensors — without needing to know
    /// the fabric's internal architecture.
    ///
    /// `predicted` is the label the caller expects to appear in a future frame.
    /// `resolve_on` optionally overrides which label to watch for (useful when
    /// the prediction outcome has a different label than the prediction itself).
    /// `streams` names every sensor contributing to this prediction's context.
    ///
    /// The registry will automatically resolve the prediction within
    /// `PREDICTION_TIMEOUT_STEPS` pool steps, recording the outcome in the
    /// episodic store and applying corrective Hebbian updates.
    pub fn register_prediction(
        &self,
        context_labels: Vec<String>,
        predicted: String,
        streams: Vec<String>,
        p_confidence: f32,
        resolve_on: Option<String>,
    ) {
        if !self.config.enabled {
            return;
        }
        let mut guard = self.inner.lock();
        let current_step = guard.pool.step;
        guard.registry.register(
            context_labels,
            predicted,
            streams,
            p_confidence,
            current_step,
            resolve_on,
        );
    }

    /// Record a resolved prediction episode into the episodic store.
    ///
    /// Called automatically by `FabricTrainer` for every `PredictionResolved`
    /// signal.  Virtual sensors (chess training loop, QA runners, etc.) do NOT
    /// need to call this directly — the fabric wires it for them.
    ///
    /// `streams` must list every sensor/virtual-sensor stream whose labels
    /// contributed to this episode's context (enables cross-modal retrieval).
    pub fn record_episode(
        &self,
        context_labels: Vec<String>,
        predicted: String,
        actual: String,
        streams: Vec<String>,
        surprise: f32,
    ) {
        if !self.config.enabled {
            return;
        }
        let mut guard = self.inner.lock();
        let step = guard.pool.step;
        let correct = predicted == actual;
        let importance = surprise.max(0.01);

        // Feed into the sufficiency tracker with every contributing stream's
        // perspective — the outcome is the actual label.
        guard.sufficiency.observe(&context_labels, &actual);

        let mut sorted_ctx = context_labels.clone();
        sorted_ctx.sort();
        sorted_ctx.dedup();

        let mut sorted_streams = streams.clone();
        sorted_streams.sort();
        sorted_streams.dedup();

        let episode = Episode {
            context_labels: sorted_ctx,
            predicted,
            actual,
            streams: sorted_streams,
            timestamp: step,
            correct,
            surprise,
            importance,
        };
        guard.episodic.record(episode);
    }

    /// Query the episodic store for past wrong predictions similar to the
    /// current context.  Returns up to `k` similar failure episodes.
    ///
    /// The caller (annealer / ConditionalSufficiencyTracker expansion logic)
    /// uses these to identify which context variable was missing when the
    /// fabric got things wrong but gets things right in similar situations.
    pub fn query_episodic_failures(
        &self,
        context_labels: &[String],
        k: usize,
    ) -> Vec<EpisodicQueryResult> {
        if !self.config.enabled {
            return Vec::new();
        }
        let mut sorted_ctx: Vec<String> = context_labels.to_vec();
        sorted_ctx.sort();
        sorted_ctx.dedup();

        let mut guard = self.inner.lock();
        let hits = guard.episodic.query_similar_failures(&sorted_ctx, k);
        hits.iter()
            .map(|ep| EpisodicQueryResult {
                context_labels: ep.context_labels.clone(),
                predicted: ep.predicted.clone(),
                actual: ep.actual.clone(),
                streams: ep.streams.clone(),
                surprise: ep.surprise,
                importance: ep.importance,
                timestamp: ep.timestamp,
            })
            .collect()
    }

    /// Query the episodic store for the `k` most similar episodes regardless
    /// of correctness — used by the annealer to bias toward lowest-energy motifs.
    pub fn query_episodic_motifs(
        &self,
        context_labels: &[String],
        k: usize,
    ) -> Vec<EpisodicQueryResult> {
        if !self.config.enabled {
            return Vec::new();
        }
        let mut sorted_ctx: Vec<String> = context_labels.to_vec();
        sorted_ctx.sort();
        sorted_ctx.dedup();

        let guard = self.inner.lock();
        guard.episodic
            .query_similar_any(&sorted_ctx, k)
            .into_iter()
            .map(|(_, ep)| EpisodicQueryResult {
                context_labels: ep.context_labels.clone(),
                predicted: ep.predicted.clone(),
                actual: ep.actual.clone(),
                streams: ep.streams.clone(),
                surprise: ep.surprise,
                importance: ep.importance,
                timestamp: ep.timestamp,
            })
            .collect()
    }

    /// Returns true if the fabric considers the current context conditionally
    /// sufficient — i.e. entropy is low enough that predictions are essentially
    /// deterministic for this context signature.
    pub fn is_context_sufficient(&self, context_labels: &[String]) -> bool {
        if !self.config.enabled {
            return false;
        }
        let guard = self.inner.lock();
        guard.sufficiency.is_sufficient(context_labels)
    }

    /// Return the Shannon entropy (nats) for the given context, or None if
    /// not enough observations have accumulated yet.
    pub fn context_entropy(&self, context_labels: &[String]) -> Option<f64> {
        if !self.config.enabled {
            return None;
        }
        let guard = self.inner.lock();
        guard.sufficiency.entropy_for(context_labels)
    }

    /// Return a snapshot of all context signatures that have reached
    /// sufficiency (deterministic prediction found).  The annealer uses these
    /// as lowest-energy attractors.
    pub fn sufficient_contexts(&self) -> Vec<String> {
        if !self.config.enabled {
            return Vec::new();
        }
        let guard = self.inner.lock();
        guard.sufficiency.sufficient_contexts.clone()
    }

    /// Return the number of episodes currently stored.
    pub fn episodic_len(&self) -> usize {
        if !self.config.enabled {
            return 0;
        }
        let guard = self.inner.lock();
        guard.episodic.len()
    }
}

/// A resolved episodic memory query result — safe to send across the API
/// boundary without holding the lock.
#[derive(Debug, Clone, Serialize)]
pub struct EpisodicQueryResult {
    pub context_labels: Vec<String>,
    pub predicted: String,
    pub actual: String,
    /// All stream names that contributed to this episode's context.
    pub streams: Vec<String>,
    pub surprise: f32,
    pub importance: f32,
    pub timestamp: u64,
}

/// A single activated label in the target stream, with its propagated
/// strength and the training influences that shaped the neuron.
#[derive(Debug, Clone, Serialize)]
pub struct CrossStreamActivation {
    /// The label that was activated (e.g. "attr::frequency::low",
    /// "meta::artist::radiohead", "stream::audio").
    pub label: String,
    /// Propagated activation strength — higher = more strongly connected to input.
    pub strength: f32,
    /// The training contexts that most shaped this neuron.
    /// Use these to answer "why did this sound come up?" or "what influenced this?"
    pub influences: Vec<InfluenceRecord>,
}

/// One frame's worth of cross-stream reconstruction output.
#[derive(Debug, Clone, Serialize)]
pub struct CrossStreamFrame {
    pub frame_index: usize,
    /// The input labels that seeded this frame.
    pub input_labels: Vec<String>,
    /// The activated labels in the target stream for this frame,
    /// sorted strongest-first.
    pub output: Vec<CrossStreamActivation>,
}

/// The full output of `reconstruct_sequence` — one `CrossStreamFrame` per
/// input frame, in temporal order.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SequenceReconstruction {
    pub target_stream: String,
    pub hops: usize,
    pub carry: f32,
    pub frames: Vec<CrossStreamFrame>,
}

pub fn zone_label(position: &Position) -> String {
    let step = (RELATION_BINS.max(1)) as f64;
    let bx = ((position.x / step).floor() as i32).clamp(0, RELATION_BINS - 1);
    let by = ((position.y / step).floor() as i32).clamp(0, RELATION_BINS - 1);
    format!("zone::{bx},{by}")
}

fn dynamic_state_from_snapshot(snapshot: &EnvironmentSnapshot) -> DynamicState {
    let mut symbol_states = HashMap::new();
    for symbol in &snapshot.symbols {
        // Pass through the velocity vector if the data-stream preparer supplied it.
        // Also fall back to extracting (velocity_dx, velocity_dy) from properties
        // for streams that embed velocity inside the properties map.
        let velocity = symbol.velocity.or_else(|| {
            let dx = symbol.properties.get("velocity_dx").and_then(|v| v.as_f64());
            let dy = symbol.properties.get("velocity_dy").and_then(|v| v.as_f64());
            match (dx, dy) {
                (Some(x), Some(y)) if x != 0.0 || y != 0.0 => {
                    Some(Position { x, y, z: 0.0 })
                }
                _ => None,
            }
        });
        symbol_states.insert(
            symbol.id.clone(),
            SymbolState {
                position: symbol.position,
                velocity,
                internal_state: symbol.properties.clone(),
            },
        );
    }
    DynamicState {
        timestamp: snapshot.timestamp,
        symbol_states,
    }
}

fn role_label(symbol: &Symbol) -> String {
    if let Some(role) = symbol
        .properties
        .get("role")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    {
        role.to_uppercase()
    } else {
        match symbol.symbol_type {
            SymbolType::Person => "PERSON".to_string(),
            SymbolType::Wall => "WALL".to_string(),
            SymbolType::Exit => "EXIT".to_string(),
            SymbolType::Custom => "CUSTOM".to_string(),
            SymbolType::Object => "OBJECT".to_string(),
        }
    }
}

fn group_label(symbol: &Symbol) -> String {
    symbol
        .properties
        .get("group_id")
        .and_then(|v| v.as_str())
        .map(|g| g.to_lowercase())
        .or_else(|| {
            symbol
                .properties
                .get("side")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase())
        })
        .unwrap_or_else(|| "neutral".to_string())
}

fn domain_label(symbol: &Symbol) -> String {
    symbol
        .properties
        .get("domain")
        .and_then(|v| v.as_str())
        .map(|d| d.to_lowercase())
        .unwrap_or_else(|| "global".to_string())
}

fn phenotype_labels(symbol: &Symbol, max_labels: usize) -> Vec<(String, String)> {
    if max_labels == 0 {
        return Vec::new();
    }
    let mut raw_pairs = Vec::new();
    if let Some(Value::Object(attrs)) = symbol.properties.get("attributes") {
        collect_string_attributes(attrs.iter(), &mut raw_pairs);
    }
    collect_string_attributes(symbol.properties.iter(), &mut raw_pairs);
    let mut seen_keys = HashSet::new();
    let mut output = Vec::new();
    for (raw_key, raw_value) in raw_pairs {
        if !is_phenotype_key(&raw_key) {
            continue;
        }
        let key = normalize_label_token(&raw_key);
        let value = normalize_label_token(&raw_value);
        if key.is_empty() || value.is_empty() {
            continue;
        }
        if seen_keys.insert(key.clone()) {
            output.push((key, value));
            if output.len() >= max_labels {
                break;
            }
        }
    }
    output
}

fn collect_string_attributes<'a, I>(source: I, out: &mut Vec<(String, String)>)
where
    I: IntoIterator<Item = (&'a String, &'a Value)>,
{
    for (key, value) in source {
        if let Some(text) = value.as_str() {
            out.push((key.clone(), text.to_string()));
        }
    }
}

fn build_minicolumn_signature(
    role: &str,
    phenotype: &[(String, String)],
    min_labels: usize,
) -> Option<MinicolumnSignature> {
    if phenotype.is_empty() {
        return None;
    }
    let mut labels = Vec::with_capacity(phenotype.len() + 1);
    labels.push(format!("role::{role}"));
    for (key, value) in phenotype {
        labels.push(format!("attr::{key}::{value}"));
    }
    labels.sort();
    labels.dedup();
    if labels.len() < min_labels.max(1) {
        return None;
    }
    let mut attr_map = HashMap::new();
    for (key, value) in phenotype {
        attr_map.insert(key.clone(), value.clone());
    }
    let key = labels.join("|");
    Some(MinicolumnSignature {
        key,
        labels,
        attr_map,
    })
}

fn best_signature_match(
    column: &MiniColumn,
    signatures: &[MinicolumnSignature],
) -> (f32, f32) {
    if signatures.is_empty() {
        return (0.0, 0.0);
    }
    let mut best_evidence = 0.0;
    let mut best_conflict = 0.0;
    for signature in signatures {
        let evidence = signature_evidence(&column.labels, &signature.labels);
        if evidence <= 0.0 {
            continue;
        }
        let conflict = signature_conflict(&column.attr_map, &signature.attr_map);
        if evidence > best_evidence
            || ((evidence - best_evidence).abs() < 1e-6 && conflict < best_conflict)
        {
            best_evidence = evidence;
            best_conflict = conflict;
        }
    }
    (best_evidence, best_conflict)
}

fn signature_evidence(column: &[String], candidate: &[String]) -> f32 {
    if column.is_empty() {
        return 0.0;
    }
    let mut set = HashSet::new();
    for label in candidate {
        set.insert(label.as_str());
    }
    let matches = column.iter().filter(|label| set.contains(label.as_str())).count();
    matches as f32 / column.len() as f32
}

fn signature_conflict(
    column: &HashMap<String, String>,
    candidate: &HashMap<String, String>,
) -> f32 {
    if column.is_empty() {
        return 0.0;
    }
    let mut conflicts = 0.0_f32;
    let mut total = 0.0_f32;
    for (key, value) in column {
        total += 1.0;
        if let Some(other) = candidate.get(key) {
            if other != value {
                conflicts += 1.0;
            }
        }
    }
    if total <= 0.0 {
        0.0
    } else {
        (conflicts / total).clamp(0.0, 1.0)
    }
}

fn normalize_label_token(raw: &str) -> String {
    let mut out = String::new();
    let mut last_underscore = false;
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.len() > 40 {
        out.truncate(40);
    }
    out
}

fn is_phenotype_key(raw: &str) -> bool {
    let key = raw.trim().to_ascii_lowercase();
    if key.is_empty() {
        return false;
    }
    if key.starts_with("pos_")
        || key.starts_with("space_")
        || key.starts_with("camera_")
        || key.starts_with("track_")
    {
        return false;
    }
    if key.ends_with("_id") || key.contains("timestamp") || key.contains("time") {
        return false;
    }
    if matches!(
        key.as_str(),
        "entity_id"
            | "token_kind"
            | "duration_secs"
            | "confidence"
            | "source"
            | "signal_quality"
            | "snr_proxy"
            | "baseline_mean"
            | "baseline_std"
            | "baseline_samples"
            | "baseline_ready"
            | "space_frame"
            | "space_source"
            | "space_dimensionality"
            | "pos_x"
            | "pos_y"
            | "pos_z"
            | "bbox_area"
    ) {
        return false;
    }
    if key.starts_with("phenotype_")
        || key.starts_with("bio_")
        || key.starts_with("vehicle_")
        || key.starts_with("anatomy_")
        || key.starts_with("meta_")
    {
        return true;
    }
    matches!(
        key.as_str(),
        // Generic object phenotypes
        "color"
            | "make"
            | "model"
            | "brand"
            | "type"
            | "category"
            | "class"
            | "subtype"
            | "variant"
            | "species"
            | "breed"
            | "sex"
            | "body_type"
            | "vehicle_make"
            | "vehicle_model"
            | "vehicle_type"
            // Chess / board-game motion motifs
            | "piece"
            | "role"
            | "move_geometry"   // "diagonal", "orthogonal", "L_shape", "none"
            | "side"
            | "side_to_move"
            | "opening"
            | "eco"
            // Music / audio stream metadata
            | "artist"
            | "title"
            | "album"
            | "genre"
            | "year"
            | "instrument"
            | "key"
            | "mode"
            | "tempo"
            | "time_signature"
            // Generic multi-modal stream identifier
            | "stream"
            | "data_source"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{DynamicState, SymbolState, Timestamp};
    use std::collections::HashMap;

    fn car_symbol(id: &str, make: &str, model: &str) -> Symbol {
        let mut properties = HashMap::new();
        properties.insert("color".to_string(), Value::String("red".to_string()));
        properties.insert("make".to_string(), Value::String(make.to_string()));
        properties.insert("model".to_string(), Value::String(model.to_string()));
        Symbol {
            id: id.to_string(),
            symbol_type: SymbolType::Object,
            position: Position::default(),
            velocity: None,
            properties,
        }
    }

    fn state_for(id: &str, x: f64) -> DynamicState {
        let mut symbol_states = HashMap::new();
        symbol_states.insert(
            id.to_string(),
            SymbolState {
                position: Position { x, y: 1.0, z: 0.0 },
                velocity: None,
                internal_state: HashMap::new(),
            },
        );
        DynamicState {
            timestamp: Timestamp { unix: 1 },
            symbol_states,
        }
    }

    #[test]
    fn minicolumn_spawns_from_stable_signature() {
        let mut config = NeuroConfig::default();
        config.minicolumn_threshold = 2;
        config.minicolumn_max = 4;
        let mut state = NeuroState::new(config, HierarchicalMotifConfig::default());
        let mut symbols = HashMap::new();
        symbols.insert("car1".to_string(), car_symbol("car1", "hyundai", "sonata"));

        let step = state_for("car1", 1.0);
        state.pool.step();
        state.record_state(&step, &symbols, 0.1, 10, 8, 0.99, 0.3, 2, 0.0, 4, None);
        state.pool.step();
        state.record_state(&step, &symbols, 0.1, 10, 8, 0.99, 0.3, 2, 0.0, 4, None);

        assert_eq!(state.minicolumns.len(), 1);
        let column = &state.minicolumns[0];
        assert!(column.labels.iter().any(|l| l.contains("attr::make::hyundai")));
    }

    #[test]
    fn minicolumn_collapses_on_conflict() {
        let mut config = NeuroConfig::default();
        config.minicolumn_threshold = 1;
        config.minicolumn_max = 4;
        config.minicolumn_inhibition_decay = 0.0;
        config.minicolumn_collapse_threshold = 0.3;
        let mut state = NeuroState::new(config, HierarchicalMotifConfig::default());
        let mut symbols = HashMap::new();
        symbols.insert("car1".to_string(), car_symbol("car1", "hyundai", "sonata"));
        symbols.insert("car2".to_string(), car_symbol("car2", "chrysler", "500"));

        let hyundai = state_for("car1", 1.0);
        state.pool.step();
        state.record_state(&hyundai, &symbols, 0.1, 10, 8, 0.99, 0.3, 2, 0.0, 4, None);

        let chrysler = state_for("car2", 2.0);
        state.pool.step();
        state.record_state(&chrysler, &symbols, 0.1, 10, 8, 0.99, 0.3, 2, 0.0, 4, None);

        let column = state
            .minicolumns
            .iter()
            .find(|col| col.labels.iter().any(|l| l.contains("attr::make::hyundai")))
            .expect("hyundai column");
        assert!(column.inhibition > 0.0);
        assert!(column.collapsed);
    }
}
