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
    /// Increment applied to fatigue when a neuron fires.
    pub fatigue_increment: f32,
    /// Decay applied to fatigue each step.
    pub fatigue_decay: f32,
    /// Top-k winner-take-all per zone to keep activity sparse.
    pub wta_k_per_zone: usize,
    /// Scale for STDP-lite weight nudges.
    pub stdp_scale: f32,
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
        self.neurons.get(id as usize).and_then(|n| n.label.clone())
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

#[derive(Debug, Clone, Default)]
pub struct NeuralNetworkSnapshot {
    pub label: String,
    pub members: Vec<String>,
    pub strength: f32,
    pub level: u8,
}

#[derive(Debug, Clone, Default)]
pub struct NeuroSnapshot {
    pub active_labels: HashSet<String>,
    pub active_composites: HashSet<String>,
    pub active_networks: Vec<NeuralNetworkSnapshot>,
    pub centroids: HashMap<String, Position>,
    pub network_links: HashMap<String, HashMap<String, f32>>,
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
    network_cooccur: HashMap<(u32, u32), u64>,
}

impl NeuroState {
    fn new(config: NeuroConfig) -> Self {
        Self {
            pool: NeuronPool::new(config),
            networks: Vec::new(),
            label_to_network: HashMap::new(),
            positions: HashMap::new(),
            network_cooccur: HashMap::new(),
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
        let mut zone_map: HashMap<String, Vec<u32>> = HashMap::new();
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
            }
            let id_label = format!("id::{symbol_id}");
            let id_idx = self.pool.get_or_create(&id_label);
            zone_map
                .entry(zone_label(&symbol_state.position))
                .or_default()
                .push(id_idx);
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
        self.apply_zone_wta(&zone_map);
        self.apply_stdp();
        self.promote_networks(module_threshold, max_networks);
        self.refresh_network_strengths(min_activation, decay);
        self.promote_super_networks(module_threshold, max_networks, min_activation);
        self.refresh_cross_links(min_activation);
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
        NeuroSnapshot {
            active_labels,
            active_composites,
            active_networks,
            centroids,
            network_links,
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
