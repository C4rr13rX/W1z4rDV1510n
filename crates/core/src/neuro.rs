use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolType};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
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
}

impl Default for NeuroRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_activation: 0.55,
            neuro: NeuroConfig::default(),
            module_threshold: 40,
            max_networks: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Synapse {
    pub target: u32,
    pub weight: f32,
    pub inhibitory: bool,
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub id: u32,
    pub label: Option<String>,
    pub activation: f32,
    pub bias: f32,
    pub excitatory: Vec<Synapse>,
    pub inhibitory: Vec<Synapse>,
    pub born_at: u64,
    pub use_count: u64,
}

impl Neuron {
    fn new(id: u32, label: Option<String>, born_at: u64) -> Self {
        Self {
            id,
            label,
            activation: 0.0,
            bias: 0.0,
            excitatory: Vec::new(),
            inhibitory: Vec::new(),
            born_at,
            use_count: 0,
        }
    }
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
        }
    }

    pub fn get_or_create(&mut self, label: &str) -> u32 {
        if let Some(id) = self.label_to_id.get(label) {
            return *id;
        }
        let id = if let Some(id) = self.free.pop() {
            let neuron = &mut self.neurons[id as usize];
            neuron.label = Some(label.to_string());
            neuron.activation = 0.0;
            neuron.use_count = 0;
            neuron.born_at = self.step;
            id
        } else {
            let id = self.neurons.len() as u32;
            if self.neurons.len() >= self.config.max_neurons {
                return id.saturating_sub(1);
            }
            self.neurons.push(Neuron::new(id, Some(label.to_string()), self.step));
            id
        };
        self.label_to_id.insert(label.to_string(), id);
        id
    }

    pub fn record_symbols(&mut self, symbols: &[String]) {
        // activate neurons for symbols
        let mut ids = Vec::with_capacity(symbols.len());
        for label in symbols {
            let id = self.get_or_create(label);
            if let Some(neuron) = self.neurons.get_mut(id as usize) {
                neuron.activation = 1.0;
                neuron.use_count += 1;
            }
            ids.push(id);
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
        self.neurons
            .get(id as usize)
            .and_then(|n| n.label.clone())
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
        let (Some(n_a), Some(n_b)) = (
            self.neurons.get(a as usize),
            self.neurons.get(b as usize),
        ) else {
            return;
        };
        let delta = self.config.hebbian_lr * n_a.activation * n_b.activation * scale;
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
}

#[derive(Debug, Clone, Default)]
pub struct NeuralNetworkSnapshot {
    pub label: String,
    pub members: Vec<String>,
    pub strength: f32,
}

#[derive(Debug, Clone, Default)]
pub struct NeuroSnapshot {
    pub active_labels: HashSet<String>,
    pub active_composites: HashSet<String>,
    pub active_networks: Vec<NeuralNetworkSnapshot>,
    pub centroids: HashMap<String, Position>,
}

impl NeuroSnapshot {
    pub fn is_empty(&self) -> bool {
        self.active_labels.is_empty()
            && self.active_composites.is_empty()
            && self.active_networks.is_empty()
            && self.centroids.is_empty()
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
    positions: HashMap<String, PositionAccumulator>,
}

impl NeuroState {
    fn new(config: NeuroConfig) -> Self {
        Self {
            pool: NeuronPool::new(config),
            networks: Vec::new(),
            label_to_network: HashMap::new(),
            positions: HashMap::new(),
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
    ) {
        let mut labels: Vec<String> = Vec::with_capacity(state.symbol_states.len() * 5 + 2);
        for (symbol_id, symbol_state) in &state.symbol_states {
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
                self.bump_position(&format!("role::{role}"), &symbol_state.position);
                self.bump_position(&format!("group::{group}"), &symbol_state.position);
                self.bump_position(&format!("domain::{domain}"), &symbol_state.position);
            }
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
        labels.sort();
        labels.dedup();
        self.pool.record_symbols(&labels);
        self.promote_networks(module_threshold, max_networks);
        self.refresh_network_strengths(min_activation, decay);
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
            });
            self.label_to_network.insert(label, net_id);
        }
    }

    fn refresh_network_strengths(&mut self, min_activation: f32, decay: f32) {
        let active = self.pool.active_ids(min_activation);
        for net in &mut self.networks {
            net.strength *= decay;
            if net.members.iter().all(|m| active.contains(m)) {
                net.use_count += 1;
                net.strength += 0.5;
            }
            net.strength = net.strength.clamp(0.0, 16.0);
        }
    }

    fn snapshot(&self, min_activation: f32) -> NeuroSnapshot {
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
            });
        }
        NeuroSnapshot {
            active_labels,
            active_composites,
            active_networks,
            centroids,
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
            );
        }
    }

    pub fn snapshot(&self) -> NeuroSnapshot {
        if !self.config.enabled {
            return NeuroSnapshot::default();
        }
        let guard = self.inner.lock();
        guard.snapshot(self.config.min_activation)
    }
}

pub fn zone_label(position: &Position) -> String {
    let step = (RELATION_BINS.max(1)) as f64;
    let bx = ((position.x / step).floor() as i32).clamp(0, RELATION_BINS - 1);
    let by = ((position.y / step).floor() as i32).clamp(0, RELATION_BINS - 1);
    format!("zone::{bx},{by}")
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
