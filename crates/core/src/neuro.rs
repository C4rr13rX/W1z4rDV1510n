use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolState, SymbolType};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
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

#[derive(Debug, Clone)]
pub struct Synapse {
    pub target: u32,
    pub weight: f32,
    pub inhibitory: bool,
}

#[derive(Debug, Clone)]
pub struct Dendrite {
    pub source: u32,
    pub weight: f32,
}

#[derive(Debug, Clone)]
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
}

/// Simple object-pool style neuron store with on-the-fly composite creation from co-occurring symbols.
#[derive(Debug)]
pub struct NeuronPool {
    neurons: Vec<Neuron>,
    free: Vec<u32>,
    label_to_id: HashMap<String, u32>,
    cooccur: HashMap<(String, String), u64>,
    config: NeuroConfig,
    step: u64,
}

impl NeuronPool {
    pub fn new(config: NeuroConfig) -> Self {
        Self {
            neurons: Vec::new(),
            free: Vec::new(),
            label_to_id: HashMap::new(),
            cooccur: HashMap::new(),
            config,
            step: 0,
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
        for neuron in self.neurons.iter_mut() {
            neuron.activation *= self.config.decay;
            neuron.fatigue *= self.config.fatigue_decay;
            neuron.trace *= self.config.decay;
        }
    }

    pub fn get_or_create(&mut self, label: &str) -> u32 {
        if let Some(id) = self.label_to_id.get(label) {
            return *id;
        }
        let id = if let Some(id) = self.free.pop() {
            let neuron = &mut self.neurons[id as usize];
            neuron.label = Some(label.to_string());
            neuron.symbol_id = label.strip_prefix("id::").map(|s| s.to_string());
            neuron.activation = 0.0;
            neuron.use_count = 0;
            neuron.born_at = self.step;
            id
        } else {
            let id = self.neurons.len() as u32;
            if self.neurons.len() >= self.config.max_neurons {
                return id.saturating_sub(1);
            }
            let mut neuron = Neuron::new(id, Some(label.to_string()), self.step);
            neuron.symbol_id = label.strip_prefix("id::").map(|s| s.to_string());
            self.neurons.push(neuron);
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
        for label in symbols {
            let id = self.get_or_create(label);
            if let Some(neuron) = self.neurons.get_mut(id as usize) {
                neuron.activation = (1.0 - neuron.fatigue).max(0.0);
                neuron.use_count += 1;
                neuron.trace += 0.1;
                neuron.fatigue = (neuron.fatigue + self.config.fatigue_increment).min(0.6);
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
                    if let Some(neuron) = self.neurons.get_mut(id as usize) {
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
                *self.cooccur.entry(key).or_insert(0) += 1;
            }
        }
        self.maybe_spawn_composites();
        // simple Hebbian strengthening between active pairs
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                self.hebbian_pair(ids[i], ids[j], self.config.excitatory_scale, false);
            }
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
        for label in symbols {
            let id = self.get_or_create(label);
            if let Some(neuron) = self.neurons.get_mut(id as usize) {
                neuron.activation = (1.0 - neuron.fatigue).max(0.0);
                neuron.use_count += 1;
                neuron.trace += 0.1 * lr_scale;
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
                    if let Some(neuron) = self.neurons.get_mut(id as usize) {
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
        for neuron in &self.neurons {
            if neuron.activation >= min_activation {
                set.insert(neuron.id);
            }
        }
        set
    }

    pub fn label_for(&self, id: u32) -> Option<String> {
        self.neurons.get(id as usize).and_then(|n| n.label.clone())
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

        // Hop-by-hop propagation
        for _ in 0..hops {
            let mut next = activation.clone();
            for (src_idx, &src_act) in activation.iter().enumerate() {
                if src_act < 0.001 {
                    continue;
                }
                let neuron = match self.neurons.get(src_idx) {
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
            activation = next;
        }

        // Collect results above threshold
        let mut result = HashMap::new();
        for (idx, act) in activation.iter().enumerate() {
            if *act >= min_activation {
                if let Some(label) = self.neurons.get(idx).and_then(|n| n.label.as_ref()) {
                    result.insert(label.clone(), *act);
                }
            }
        }
        result
    }

    pub fn cooccurrences_above(&self, threshold: u64) -> Vec<((String, String), u64)> {
        self.cooccur
            .iter()
            .filter_map(|((a, b), count)| {
                if *count >= threshold {
                    Some(((a.clone(), b.clone()), *count))
                } else {
                    None
                }
            })
            .collect()
    }

    fn hebbian_pair(&mut self, a: u32, b: u32, scale: f32, inhibitory: bool) {
        let (Some(n_a), Some(n_b)) = (self.neurons.get(a as usize), self.neurons.get(b as usize))
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
        if let Some(neuron) = self.neurons.get_mut(from as usize) {
            let list = if inhibitory {
                &mut neuron.inhibitory
            } else {
                &mut neuron.excitatory
            };
            if let Some(existing) = list.iter_mut().find(|s| s.target == to) {
                existing.weight += delta;
            } else {
                list.push(Synapse {
                    target: to,
                    weight: delta,
                    inhibitory,
                });
            }
        }
        // add dendrite on target side
        if let Some(target) = self.neurons.get_mut(to as usize) {
            if let Some(existing) = target.dendrites.iter_mut().find(|d| d.source == from) {
                existing.weight += delta;
            } else {
                target.dendrites.push(Dendrite {
                    source: from,
                    weight: delta,
                });
            }
        }
    }

    fn maybe_spawn_composites(&mut self) {
        let threshold = self.config.composite_threshold;
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
        for neuron in self.neurons.iter() {
            if neuron.activation >= min_activation {
                if let Some(label) = &neuron.label {
                    set.insert(label.clone());
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

#[derive(Debug, Clone, Default, Serialize)]
pub struct NeuralNetworkSnapshot {
    pub label: String,
    pub members: Vec<String>,
    pub strength: f32,
    pub level: u8,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MinicolumnSnapshot {
    pub label: String,
    pub stability: f32,
    pub inhibition: f32,
    pub collapsed: bool,
    pub members: usize,
    pub born_at: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
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
}

impl NeuroState {
    fn new(config: NeuroConfig) -> Self {
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
                strength: (count as f32).sqrt().max(1.0),
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
                    self.pool
                        .neurons
                        .get(*id as usize)
                        .map(|n| (n.activation, *id))
                })
                .collect::<Vec<_>>();
            sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let keep: HashSet<u32> = sorted.iter().take(k).map(|(_, id)| *id).collect();
            for (_, id) in sorted.iter().skip(k) {
                if let Some(neuron) = self.pool.neurons.get_mut(*id as usize) {
                    neuron.activation *= self.pool.config.inhibitory_scale;
                }
            }
            // lateral inhibition between kept neurons
            for id in &keep {
                if let Some(neuron) = self.pool.neurons.get_mut(*id as usize) {
                    neuron.activation *= 1.0;
                }
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

        for column in &mut self.minicolumns {
            let (evidence, conflict) = best_signature_match(column, signatures);
            column.stability =
                column.stability * stability_decay + evidence * (1.0 - stability_decay);
            column.inhibition =
                column.inhibition * inhibition_decay + conflict * (1.0 - inhibition_decay);
            column.last_seen = step;
            column.collapsed = column.inhibition >= collapse_threshold;
            let active = column.stability >= activation_threshold && !column.collapsed;
            let column_activation = (column.stability * (1.0 - column.inhibition)).clamp(0.0, 1.0);
            if let Some(neuron) = self.pool.neurons.get_mut(column.id as usize) {
                neuron.activation = neuron.activation.max(column_activation);
                neuron.use_count = neuron.use_count.saturating_add(active as u64);
            }
            if active {
                for member_id in &column.members {
                    if let Some(neuron) = self.pool.neurons.get_mut(*member_id as usize) {
                        neuron.activation *= inhibit_scale;
                    }
                }
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
        let mut neuron_activations: Vec<(f32, &Neuron)> = self
            .pool
            .neurons
            .iter()
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
            let (pre_trace, pre_fatigue) = {
                let pre = &self.pool.neurons[idx];
                (pre.trace as f32, pre.fatigue)
            };
            // borrow target activations first to avoid aliasing
            let excit_updates = {
                let neuron = &self.pool.neurons[idx];
                neuron
                    .excitatory
                    .iter()
                    .map(|syn| {
                        let post_act = self
                            .pool
                            .neurons
                            .get(syn.target as usize)
                            .map(|p| p.activation as f32)
                            .unwrap_or(0.0);
                        let delta = scale * (pre_trace * post_act - pre_fatigue * syn.weight * 0.1);
                        (syn.target, delta, syn.inhibitory)
                    })
                    .collect::<Vec<_>>()
            };
            let inhib_updates = {
                let neuron = &self.pool.neurons[idx];
                neuron
                    .inhibitory
                    .iter()
                    .map(|syn| {
                        let post_act = self
                            .pool
                            .neurons
                            .get(syn.target as usize)
                            .map(|p| p.activation as f32)
                            .unwrap_or(0.0);
                        let delta = scale * (pre_trace * post_act - pre_fatigue * syn.weight * 0.1);
                        (syn.target, delta, syn.inhibitory)
                    })
                    .collect::<Vec<_>>()
            };
            if let Some(neuron) = self.pool.neurons.get_mut(idx) {
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
        Self {
            config: config.clone(),
            symbol_lookup,
            inner: Arc::new(Mutex::new(NeuroState::new(config.neuro))),
        }
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
                        guard.pool.neurons.get(id as usize)
                    }).map(|n| {
                        n.influence_history.iter().any(|r| r.stream == target_stream)
                    }).unwrap_or(false)
            })
            .map(|(label, strength)| {
                // Collect top influences for this label from its neuron's history
                let influences = guard.pool.label_to_id.get(&label)
                    .and_then(|&id| guard.pool.neurons.get(id as usize))
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
        carry: f32,           // 0.0 = no temporal bleed, 1.0 = pure carry-forward
    ) -> SequenceReconstruction {
        if !self.config.enabled || input_frames.is_empty() {
            return SequenceReconstruction::default();
        }

        let carry = carry.clamp(0.0, 0.9);
        let guard = self.inner.lock();
        let min_act = self.config.min_activation;

        // Previous frame's propagated activations, re-injected as context
        let mut prev_output: HashMap<String, f32> = HashMap::new();

        let mut frames: Vec<CrossStreamFrame> = Vec::with_capacity(input_frames.len());

        for (frame_idx, input_labels) in input_frames.iter().enumerate() {
            // Build seed: current frame labels + carried-over context from prev frame output
            let mut seed_labels: Vec<String> = input_labels.clone();
            // Add carry context — inject prev output labels back as additional seeds
            // scaled by carry factor (they'll start at `carry` activation not 1.0)
            let carry_seed: Vec<String> = prev_output
                .iter()
                .filter(|&(_, &v)| v * carry >= min_act)
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
                            guard.pool.neurons.get(id as usize)
                        }).map(|n| {
                            n.influence_history.iter().any(|r| r.stream == target_stream)
                        }).unwrap_or(false)
                })
                .map(|(label, &strength)| CrossStreamActivation {
                    label: label.clone(),
                    strength,
                    influences: guard.pool.label_to_id.get(label)
                        .and_then(|&id| guard.pool.neurons.get(id as usize))
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
                            guard.pool.neurons.get(id as usize)
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
        let mut state = NeuroState::new(config);
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
        let mut state = NeuroState::new(config);
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
