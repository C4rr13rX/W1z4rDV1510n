use crate::ml::MlModelHandle;
use crate::neuro::{NeuroRuntimeHandle, NeuroSnapshot, zone_label};
use crate::schema::{
    DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolState, SymbolType,
};
use crate::search::{PathResult, SearchModule};
use blake2::Digest;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs;
use std::sync::Arc;

const RELATION_BINS: i32 = 4;
const FACTOR_EPS: f64 = 1e-9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalConfig {
    pub local_move_prob: f64,
    #[serde(default)]
    pub group_move_prob: f64,
    #[serde(default)]
    pub swap_move_prob: f64,
    #[serde(default)]
    pub path_based_move_prob: f64,
    #[serde(default)]
    pub global_move_prob: f64,
    #[serde(default)]
    pub ml_guided_move_prob: f64,
    pub max_step_size: f64,
    #[serde(default)]
    pub use_parallel_updates: bool,
    #[serde(default)]
    pub adaptive_move_mixing: bool,
    /// Optional path to relational priors JSON to bias proposals (same format as build_relational_priors.py).
    #[serde(default)]
    pub relational_priors_path: Option<String>,
    /// Weight for factor priors when biasing proposals (role/zone/group).
    #[serde(default = "ProposalConfig::default_factor_weight")]
    pub factor_prior_weight: f64,
}

#[derive(Debug, Clone, Copy)]
enum MoveType {
    Local,
    Group,
    Swap,
    Path,
    Global,
    MlGuided,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            local_move_prob: 0.9,
            group_move_prob: 0.05,
            swap_move_prob: 0.05,
            path_based_move_prob: 0.0,
            global_move_prob: 0.0,
            ml_guided_move_prob: 0.0,
            max_step_size: 0.75,
            use_parallel_updates: false,
            adaptive_move_mixing: true,
            relational_priors_path: None,
            factor_prior_weight: ProposalConfig::default_factor_weight(),
        }
    }
}

impl ProposalConfig {
    fn default_factor_weight() -> f64 {
        1.0
    }
}

fn zone_bin(position: &Position) -> (i32, i32) {
    let step = (RELATION_BINS.max(1)) as f64;
    let bx = ((position.x / step).floor() as i32).clamp(0, RELATION_BINS - 1);
    let by = ((position.y / step).floor() as i32).clamp(0, RELATION_BINS - 1);
    (bx, by)
}

fn motif_signature(cache: &SnapshotCache, state: &DynamicState) -> Option<String> {
    if state.symbol_states.is_empty() {
        return None;
    }
    let mut tokens = Vec::with_capacity(state.symbol_states.len());
    for (symbol_id, symbol_state) in state.symbol_states.iter() {
        let (bx, by) = zone_bin(&symbol_state.position);
        let role = cache
            .symbols
            .get(symbol_id)
            .map(|sym| sym.role.as_str())
            .unwrap_or("UNK");
        tokens.push(format!("u:{role}:{bx}{by}"));
    }
    tokens.sort();
    tokens.dedup();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join("|"))
    }
}

#[derive(Debug, Clone, Default)]
struct RelationalPriors {
    factor_probs: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct RelationalPriorFile {
    #[serde(default)]
    factor_probs: HashMap<String, HashMap<String, f64>>,
}

fn load_relational_priors(path: &str) -> Option<RelationalPriors> {
    let content = fs::read_to_string(path).ok()?;
    let parsed: RelationalPriorFile = serde_json::from_str(&content).ok()?;
    Some(RelationalPriors {
        factor_probs: parsed.factor_probs,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotSignature {
    timestamp: i64,
    symbols_len: usize,
}

impl SnapshotSignature {
    fn from_snapshot(snapshot_0: &EnvironmentSnapshot) -> Self {
        Self {
            timestamp: snapshot_0.timestamp.unix,
            symbols_len: snapshot_0.symbols.len(),
        }
    }
}

#[derive(Debug, Clone)]
struct CachedSymbol {
    role: String,
    group: String,
    domain: String,
    target_position: Option<Position>,
}

#[derive(Debug, Clone)]
struct SnapshotCache {
    signature: SnapshotSignature,
    symbols: HashMap<String, CachedSymbol>,
}

impl SnapshotCache {
    fn new(snapshot_0: &EnvironmentSnapshot, signature: SnapshotSignature) -> Self {
        let mut positions = HashMap::new();
        let mut exits = Vec::new();
        for symbol in &snapshot_0.symbols {
            positions.insert(symbol.id.clone(), symbol.position);
            if matches!(symbol.symbol_type, SymbolType::Exit) {
                exits.push(symbol.position);
            }
        }
        let mut symbols = HashMap::new();
        for symbol in &snapshot_0.symbols {
            let role = role_from_symbol(symbol);
            let group = group_from_symbol(symbol);
            let domain = domain_from_symbol(symbol);
            let target_position = symbol
                .properties
                .get("goal_position")
                .and_then(value_to_position)
                .or_else(|| symbol.properties.get("goal").and_then(value_to_position))
                .or_else(|| {
                    symbol
                        .properties
                        .get("goal_id")
                        .or_else(|| symbol.properties.get("target_id"))
                        .and_then(|value| value.as_str())
                        .and_then(|goal_id| positions.get(goal_id))
                        .copied()
                })
                .or_else(|| nearest_exit_position_from_list(&symbol.position, &exits));
            symbols.insert(
                symbol.id.clone(),
                CachedSymbol {
                    role,
                    group,
                    domain,
                    target_position,
                },
            );
        }
        Self { signature, symbols }
    }
}

fn role_from_symbol(symbol: &Symbol) -> String {
    if let Some(Value::String(role)) = symbol.properties.get("role") {
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

fn group_from_symbol(symbol: &Symbol) -> String {
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

fn domain_from_symbol(symbol: &Symbol) -> String {
    symbol
        .properties
        .get("domain")
        .and_then(|v| v.as_str())
        .map(|d| d.to_lowercase())
        .unwrap_or_else(|| "global".to_string())
}

pub trait ProposalKernel: Send + Sync {
    fn propose(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
    ) -> DynamicState;
}

pub struct DefaultProposalKernel {
    config: ProposalConfig,
    rng: Mutex<StdRng>,
    search: Option<SearchModule>,
    ml_model: Option<MlModelHandle>,
    relational_priors: Option<RelationalPriors>,
    neuro: Option<NeuroRuntimeHandle>,
    context_cache: Mutex<HashMap<String, VecDeque<String>>>,
    snapshot_cache: Mutex<Option<Arc<SnapshotCache>>>,
}

impl DefaultProposalKernel {
    fn apply_relational_hint(&self, proposal: &mut DynamicState, motif: &str, rng: &mut StdRng) {
        if motif.is_empty() || self.relational_priors.is_none() {
            return;
        }
        let priors = self.relational_priors.as_ref().unwrap();
        // Small random walk guided by motif hash to diversify within the same motif class.
        for state in proposal.symbol_states.values_mut() {
            let jitter = 0.01 + (motif.len() as f64 % 7.0) * 0.001;
            state.position.x += rng.gen_range(-jitter..jitter);
            state.position.y += rng.gen_range(-jitter..jitter);
        }
        if let Some(zone_probs) = priors.factor_probs.get("zone") {
            if let Some((best_key, _)) = zone_probs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if best_key.len() >= 2 {
                    let tx = best_key[0..1].parse::<i32>().unwrap_or(0);
                    let ty = best_key[1..2].parse::<i32>().unwrap_or(0);
                    let step = RELATION_BINS.max(1) as f64;
                    let target_x = (tx as f64 + 0.5) * step;
                    let target_y = (ty as f64 + 0.5) * step;
                    for state in proposal.symbol_states.values_mut() {
                        let dx = target_x - state.position.x;
                        let dy = target_y - state.position.y;
                        state.position.x += dx * 0.05 * rng.gen_range(0.5..1.0);
                        state.position.y += dy * 0.05 * rng.gen_range(0.5..1.0);
                    }
                }
            }
        }
    }
    pub fn new(
        config: ProposalConfig,
        seed: u64,
        search: Option<SearchModule>,
        ml_model: Option<MlModelHandle>,
        neuro: Option<NeuroRuntimeHandle>,
    ) -> Self {
        let relational_priors = if let Some(path) = config.relational_priors_path.as_ref() {
            load_relational_priors(path)
        } else {
            None
        };
        Self {
            config,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            search,
            ml_model,
            relational_priors,
            neuro,
            context_cache: Mutex::new(HashMap::new()),
            snapshot_cache: Mutex::new(None),
        }
    }

    fn snapshot_cache(&self, snapshot_0: &EnvironmentSnapshot) -> Arc<SnapshotCache> {
        let signature = SnapshotSignature::from_snapshot(snapshot_0);
        let mut guard = self.snapshot_cache.lock();
        let rebuild = guard
            .as_ref()
            .map(|cache| cache.signature != signature)
            .unwrap_or(true);
        if rebuild {
            *guard = Some(Arc::new(SnapshotCache::new(snapshot_0, signature)));
        }
        guard
            .as_ref()
            .expect("snapshot cache missing")
            .clone()
    }
}

impl ProposalKernel for DefaultProposalKernel {
    fn propose(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
    ) -> DynamicState {
        let cache = self.snapshot_cache(snapshot_0);
        let mut rng = self.rng.lock();
        let move_type = self.choose_move_type(&mut rng, temperature);
        let neuro_snapshot = self.neuro.as_ref().map(|n| n.snapshot());
        let relational_motif = motif_signature(cache.as_ref(), current_state);
        let zone_key = current_state
            .symbol_states
            .values()
            .next()
            .map(|s| zone_label(&s.position));
        let cached_motif = zone_key.as_ref().and_then(|k| {
            self.context_cache
                .lock()
                .get(k)
                .and_then(|q| q.front().cloned())
        });
        let mut motif_hints = Vec::new();
        if let Some(m) = relational_motif.clone() {
            motif_hints.push(m);
        } else if let Some(k) = zone_key.as_ref() {
            if let Some(queue) = self.context_cache.lock().get(k) {
                motif_hints.extend(queue.iter().cloned());
            }
        }
        if motif_hints.is_empty() {
            if let Some(m) = cached_motif {
                motif_hints.push(m);
            }
        }
        if self.config.use_parallel_updates {
            return self.apply_parallel_move(
                move_type,
                snapshot_0,
                current_state,
                temperature,
                &mut rng,
                neuro_snapshot.as_ref(),
                motif_hints
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>()
                    .as_slice(),
                zone_key.clone(),
                cache.as_ref(),
            );
        }
        let proposal = current_state.clone();
        let mut proposal = match move_type {
            MoveType::Local => {
                self.local_move(snapshot_0, proposal, temperature, &mut rng, cache.as_ref())
            }
            MoveType::Group => {
                self.group_move(snapshot_0, proposal, temperature, &mut rng, cache.as_ref())
            }
            MoveType::Swap => self.swap_move(proposal, &mut rng),
            MoveType::Path => {
                self.path_move(snapshot_0, proposal, temperature, &mut rng, cache.as_ref())
            }
            MoveType::Global => self.global_move(proposal, temperature, &mut rng),
            MoveType::MlGuided => self.ml_guided_move(snapshot_0, &proposal, temperature, &mut rng),
        };
        if let Some(neuro) = neuro_snapshot.as_ref() {
            self.apply_neuro_to_state(&mut proposal, cache.as_ref(), neuro, &mut rng);
        }
        for motif in &motif_hints {
            self.apply_relational_hint(&mut proposal, motif, &mut rng);
        }
        self.reverse_anchor_pull(snapshot_0, &mut proposal, &mut rng);
        if let Some(zone) = zone_key {
            if let Some(m) = motif_signature(cache.as_ref(), &proposal) {
                let mut cache = self.context_cache.lock();
                let entry = cache.entry(zone).or_insert_with(VecDeque::new);
                entry.push_front(m);
                while entry.len() > 3 {
                    entry.pop_back();
                }
            }
        }
        proposal
    }
}

impl DefaultProposalKernel {
    fn choose_move_type(&self, rng: &mut StdRng, temperature: f64) -> MoveType {
        let (local, group, swap, path, global, ml_guided) = self.adapted_weights(temperature);
        let total_prob = local + group + swap + path + global + ml_guided;
        if total_prob <= f64::EPSILON {
            return MoveType::Local;
        }
        let roll = rng.gen_range(0.0..1.0) * total_prob;
        let local_threshold = local;
        let group_threshold = local_threshold + group;
        let swap_threshold = group_threshold + swap;
        let path_threshold = swap_threshold + path;
        let ml_threshold = path_threshold + ml_guided;
        if roll < local_threshold {
            MoveType::Local
        } else if roll < group_threshold {
            MoveType::Group
        } else if roll < swap_threshold {
            MoveType::Swap
        } else if roll < path_threshold {
            MoveType::Path
        } else if roll < ml_threshold {
            MoveType::MlGuided
        } else {
            MoveType::Global
        }
    }

    fn apply_parallel_move(
        &self,
        move_type: MoveType,
        snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        neuro_snapshot: Option<&NeuroSnapshot>,
        motifs: &[&str],
        zone_key: Option<String>,
        cache: &SnapshotCache,
    ) -> DynamicState {
        let mut proposals = match move_type {
            MoveType::Local => {
                self.local_move_parallel(snapshot_0, current_state, temperature, rng, cache)
            }
            MoveType::Group => {
                self.group_move_parallel(snapshot_0, current_state, temperature, rng, cache)
            }
            MoveType::Swap => self.swap_move_parallel(current_state, rng),
            MoveType::Path => {
                self.path_move_parallel(snapshot_0, current_state, temperature, rng, cache)
            }
            MoveType::Global => self.global_move_parallel(current_state, temperature, rng),
            MoveType::MlGuided => self.ml_guided_move(snapshot_0, current_state, temperature, rng),
        };
        if let Some(neuro) = neuro_snapshot {
            self.apply_neuro_to_state(&mut proposals, cache, neuro, rng);
        }
        for motif in motifs {
            self.apply_relational_hint(&mut proposals, motif, rng);
        }
        self.reverse_anchor_pull(snapshot_0, &mut proposals, rng);
        if let Some(zone) = zone_key {
            if let Some(m) = motif_signature(cache, &proposals) {
                let mut cache = self.context_cache.lock();
                let entry = cache.entry(zone).or_insert_with(VecDeque::new);
                entry.push_front(m);
                while entry.len() > 3 {
                    entry.pop_back();
                }
            }
        }
        proposals
    }

    fn step_scale(&self, temperature: f64) -> f64 {
        self.config.max_step_size * temperature.max(0.05)
    }

    fn local_move(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        mut proposal: DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) -> DynamicState {
        let Some(symbol_state) = proposal.symbol_states.values_mut().choose(rng) else {
            return proposal;
        };
        let step_scale = self.step_scale(temperature);
        let mut delta = || rng.gen_range(-step_scale..step_scale);
        symbol_state.position.x += delta();
        symbol_state.position.y += delta();
        symbol_state.position.z += delta();
        if let Some(priors) = &self.relational_priors {
            self.apply_factor_bias(symbol_state, None, snapshot_0, priors, rng, cache);
        }
        proposal
    }

    fn reverse_anchor_pull(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        proposal: &mut DynamicState,
        rng: &mut StdRng,
    ) {
        if snapshot_0.stack_history.is_empty() {
            return;
        }
        let future_frame = snapshot_0.stack_history.last().unwrap();
        for (symbol_id, symbol_state) in proposal.symbol_states.iter_mut() {
            if let Some(target) = future_frame.symbol_states.get(symbol_id) {
                let dx = target.position.x - symbol_state.position.x;
                let dy = target.position.y - symbol_state.position.y;
                let dz = target.position.z - symbol_state.position.z;
                let scale = 0.08 * rng.gen_range(0.5..1.0);
                symbol_state.position.x += dx * scale;
                symbol_state.position.y += dy * scale;
                symbol_state.position.z += dz * scale * 0.5;
            }
        }
    }

    fn group_move(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        mut proposal: DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) -> DynamicState {
        if proposal.symbol_states.is_empty() {
            return proposal;
        }
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for symbol in &snapshot_0.symbols {
            if let Some(group_id) = symbol.properties.get("group_id").and_then(|v| v.as_str()) {
                groups
                    .entry(group_id.to_string())
                    .or_default()
                    .push(symbol.id.clone());
            }
        }
        if groups.is_empty() {
            return self.local_move(snapshot_0, proposal, temperature, rng, cache);
        }
        let group_ids: Vec<String> = groups.keys().cloned().collect();
        let Some(group_id) = group_ids.choose(rng).cloned() else {
            return proposal;
        };
        let Some(members) = groups.get(&group_id) else {
            return proposal;
        };
        if members.is_empty() {
            return proposal;
        }
        let step_scale = self.step_scale(temperature);
        let dx = rng.gen_range(-step_scale..step_scale);
        let dy = rng.gen_range(-step_scale..step_scale);
        let dz = rng.gen_range(-step_scale..step_scale);
        for member_id in members {
            if let Some(state) = proposal.symbol_states.get_mut(member_id) {
                state.position.x += dx;
                state.position.y += dy;
                state.position.z += dz;
                if let Some(priors) = &self.relational_priors {
                    self.apply_factor_bias(state, Some(member_id), snapshot_0, priors, rng, cache);
                }
            }
        }
        proposal
    }

    fn global_move(
        &self,
        mut proposal: DynamicState,
        temperature: f64,
        rng: &mut StdRng,
    ) -> DynamicState {
        if proposal.symbol_states.is_empty() {
            return proposal;
        }
        let step_scale = self.step_scale(temperature) * 1.5;
        let jitter = step_scale * 0.25;
        let dx = rng.gen_range(-step_scale..step_scale);
        let dy = rng.gen_range(-step_scale..step_scale);
        let dz = rng.gen_range(-step_scale..step_scale);
        for symbol_state in proposal.symbol_states.values_mut() {
            symbol_state.position.x += dx + rng.gen_range(-jitter..jitter);
            symbol_state.position.y += dy + rng.gen_range(-jitter..jitter);
            symbol_state.position.z += dz + rng.gen_range(-jitter..jitter);
        }
        proposal
    }

    fn symbol_role<'a>(&self, cache: &'a SnapshotCache, symbol_id: &str) -> &'a str {
        cache
            .symbols
            .get(symbol_id)
            .map(|sym| sym.role.as_str())
            .unwrap_or("UNK")
    }

    fn symbol_group<'a>(&self, cache: &'a SnapshotCache, symbol_id: &str) -> &'a str {
        cache
            .symbols
            .get(symbol_id)
            .map(|sym| sym.group.as_str())
            .unwrap_or("neutral")
    }

    fn apply_factor_bias(
        &self,
        symbol_state: &mut SymbolState,
        symbol_id: Option<&str>,
        _snapshot_0: &EnvironmentSnapshot,
        priors: &RelationalPriors,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) {
        let weight = self.config.factor_prior_weight.max(0.0);
        if weight == 0.0 || priors.factor_probs.is_empty() {
            return;
        }
        let (bx, by) = zone_bin(&symbol_state.position);
        if let Some(zone_probs) = priors.factor_probs.get("zone") {
            if let Some((best_key, _)) = zone_probs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if best_key.len() >= 2 {
                    let tx = best_key[0..1].parse::<i32>().unwrap_or(bx);
                    let ty = best_key[1..2].parse::<i32>().unwrap_or(by);
                    if tx != bx || ty != by {
                        let step = RELATION_BINS.max(1) as f64;
                        let target_x = (tx as f64 + 0.5) * step;
                        let target_y = (ty as f64 + 0.5) * step;
                        let dx = target_x - symbol_state.position.x;
                        let dy = target_y - symbol_state.position.y;
                        symbol_state.position.x += dx * 0.1 * weight * rng.gen_range(0.5..1.0);
                        symbol_state.position.y += dy * 0.1 * weight * rng.gen_range(0.5..1.0);
                    }
                }
            }
        }
        if let Some(id) = symbol_id {
            if let Some(role_probs) = priors.factor_probs.get("role") {
                let role = self.symbol_role(cache, id);
                if let Some(prob) = role_probs.get(role) {
                    if *prob < FACTOR_EPS {
                        let jitter = 0.05 * weight;
                        symbol_state.position.x += rng.gen_range(-jitter..jitter);
                        symbol_state.position.y += rng.gen_range(-jitter..jitter);
                    }
                }
            }
            if let Some(group_probs) = priors.factor_probs.get("group") {
                let group = self.symbol_group(cache, id);
                if let Some(prob) = group_probs.get(group) {
                    if *prob < FACTOR_EPS {
                        let jitter = 0.05 * weight;
                        symbol_state.position.x += rng.gen_range(-jitter..jitter);
                        symbol_state.position.y += rng.gen_range(-jitter..jitter);
                    }
                }
            }
        }
    }

    fn apply_neuro_to_state(
        &self,
        proposal: &mut DynamicState,
        cache: &SnapshotCache,
        neuro: &NeuroSnapshot,
        rng: &mut StdRng,
    ) {
        if neuro.is_empty() {
            return;
        }
        for (symbol_id, symbol_state) in proposal.symbol_states.iter_mut() {
            let mut labels = vec![
                format!("id::{symbol_id}"),
                zone_label(&symbol_state.position),
            ];
            let role = self.symbol_role(cache, symbol_id);
            let group = self.symbol_group(cache, symbol_id);
            labels.push(format!("role::{role}"));
            labels.push(format!("group::{group}"));
            labels.push(format!(
                "col::{role}::{}",
                zone_label(&symbol_state.position)
            ));
            let domain = cache
                .symbols
                .get(symbol_id)
                .map(|sym| sym.domain.as_str())
                .unwrap_or("global");
            labels.push(format!(
                "region::{domain}::{}",
                zone_label(&symbol_state.position)
            ));
            for label in &labels {
                if let Some(center) = neuro.centroids.get(label) {
                    let dx = center.x - symbol_state.position.x;
                    let dy = center.y - symbol_state.position.y;
                    let dz = center.z - symbol_state.position.z;
                    symbol_state.position.x += dx * 0.12 * rng.gen_range(0.5..1.0);
                    symbol_state.position.y += dy * 0.12 * rng.gen_range(0.5..1.0);
                    symbol_state.position.z += dz * 0.05 * rng.gen_range(0.5..1.0);
                } else if !neuro.active_labels.contains(label) {
                    let jitter = 0.02;
                    symbol_state.position.x += rng.gen_range(-jitter..jitter);
                    symbol_state.position.y += rng.gen_range(-jitter..jitter);
                }
            }
            for net in &neuro.active_networks {
                if net.members.iter().any(|m| m == &format!("id::{symbol_id}")) {
                    let boost = (net.strength as f64).min(8.0) / 8.0 * (net.level as f64 + 1.0);
                    symbol_state.position.x += rng.gen_range(-0.01..0.01) * boost;
                    symbol_state.position.y += rng.gen_range(-0.01..0.01) * boost;
                }
            }
            for net in &neuro.active_networks {
                if !net.members.iter().any(|m| m == &format!("id::{symbol_id}")) {
                    continue;
                }
                if let Some(links) = neuro.network_links.get(&net.label) {
                    for (_target, weight) in links {
                        let w = (*weight as f64).min(2.0);
                        symbol_state.position.x += rng.gen_range(-0.005..0.005) * w;
                        symbol_state.position.y += rng.gen_range(-0.005..0.005) * w;
                    }
                }
            }
            if let Some(pred) = neuro.temporal_predictions.get(symbol_id) {
                let err = neuro
                    .prediction_error
                    .get(symbol_id)
                    .copied()
                    .unwrap_or(0.5)
                    .clamp(0.1, 5.0);
                let confidence = neuro
                    .prediction_confidence
                    .get(symbol_id)
                    .copied()
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let surprise = neuro.surprise.get(symbol_id).copied().unwrap_or(0.0);
                let weight = (1.0 / (1.0 + err)) * confidence as f64 * 0.4;
                symbol_state.position.x += (pred.x - symbol_state.position.x) * weight;
                symbol_state.position.y += (pred.y - symbol_state.position.y) * weight;
                symbol_state.position.z += (pred.z - symbol_state.position.z) * weight * 0.5;
                if surprise > 0.5 {
                    let jitter = 0.01 * surprise.min(2.0);
                    symbol_state.position.x += rng.gen_range(-jitter..jitter);
                    symbol_state.position.y += rng.gen_range(-jitter..jitter);
                }
            }
            if let Some(motif) = neuro.working_memory.last() {
                // Treat working-memory motif as a soft relational hint.
                let jitter = self.config.max_step_size.min(0.1);
                let hash = blake2::Blake2s256::digest(motif.as_bytes());
                let scale = (hash[0] as f64 / 255.0) * self.config.max_step_size * 0.2;
                symbol_state.position.x += rng.gen_range(-jitter..jitter) * scale;
                symbol_state.position.y += rng.gen_range(-jitter..jitter) * scale;
            }
        }
    }

    fn swap_move(&self, mut proposal: DynamicState, rng: &mut StdRng) -> DynamicState {
        if proposal.symbol_states.len() < 2 {
            return proposal;
        }
        let mut ids: Vec<String> = proposal.symbol_states.keys().cloned().collect();
        ids.shuffle(rng);
        if ids.len() < 2 {
            return proposal;
        }
        let first_id = ids[0].clone();
        let second_id = ids[1].clone();
        if first_id == second_id {
            return proposal;
        }
        let first_pos = proposal
            .symbol_states
            .get(&first_id)
            .map(|state| state.position);
        let second_pos = proposal
            .symbol_states
            .get(&second_id)
            .map(|state| state.position);
        if let (Some(pos_a), Some(pos_b)) = (first_pos, second_pos) {
            if let Some(state) = proposal.symbol_states.get_mut(&first_id) {
                state.position = pos_b;
            }
            if let Some(state) = proposal.symbol_states.get_mut(&second_id) {
                state.position = pos_a;
            }
        }
        proposal
    }

    fn path_move(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        mut proposal: DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) -> DynamicState {
        let Some(search_module) = &self.search else {
            return self.local_move(snapshot_0, proposal, temperature, rng, cache);
        };
        let Some((symbol_id, symbol_state)) = proposal
            .symbol_states
            .iter_mut()
            .choose(rng)
            .map(|(id, state)| (id.clone(), state))
        else {
            return proposal;
        };
        let Some(target) = self.target_position(cache, &symbol_id) else {
            return self.local_move(snapshot_0, proposal, temperature, rng, cache);
        };
        let mut target_state = DynamicState {
            timestamp: proposal.timestamp,
            symbol_states: HashMap::new(),
        };
        target_state.symbol_states.insert(
            symbol_id.clone(),
            SymbolState {
                position: target,
                velocity: None,
                internal_state: Default::default(),
            },
        );
        let paths = search_module.compute_paths(snapshot_0, &target_state);
        let Some(path_result) = paths.get(&symbol_id) else {
            return self.local_move(snapshot_0, proposal, temperature, rng, cache);
        };
        let Some(target_point) = next_path_point(path_result, &symbol_state.position) else {
            return proposal;
        };
        self.advance_symbol(symbol_state, &target_point, temperature);
        proposal
    }

    fn local_move_parallel(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) -> DynamicState {
        let step_scale = self.step_scale(temperature);
        let mut proposal = state.clone();
        for (symbol_id, symbol_state) in state.symbol_states.iter() {
            let dx = rng.gen_range(-step_scale..step_scale);
            let dy = rng.gen_range(-step_scale..step_scale);
            let dz = rng.gen_range(-step_scale..step_scale);
            if let Some(dest) = proposal.symbol_states.get_mut(symbol_id) {
                dest.position.x = symbol_state.position.x + dx;
                dest.position.y = symbol_state.position.y + dy;
                dest.position.z = symbol_state.position.z + dz;
                if let Some(priors) = &self.relational_priors {
                    self.apply_factor_bias(dest, Some(symbol_id), snapshot_0, priors, rng, cache);
                }
            }
        }
        proposal
    }

    fn group_move_parallel(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) -> DynamicState {
        if state.symbol_states.is_empty() {
            return state.clone();
        }
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for symbol in &snapshot_0.symbols {
            if let Some(group_id) = symbol.properties.get("group_id").and_then(|v| v.as_str()) {
                groups
                    .entry(group_id.to_string())
                    .or_default()
                    .push(symbol.id.clone());
            }
        }
        if groups.is_empty() {
            return self.local_move_parallel(snapshot_0, state, temperature, rng, cache);
        }
        let mut proposal = state.clone();
        let step_scale = self.step_scale(temperature);
        for members in groups.values() {
            let dx = rng.gen_range(-step_scale..step_scale);
            let dy = rng.gen_range(-step_scale..step_scale);
            let dz = rng.gen_range(-step_scale..step_scale);
            for member_id in members {
                if let (Some(original), Some(dest)) = (
                    state.symbol_states.get(member_id),
                    proposal.symbol_states.get_mut(member_id),
                ) {
                    dest.position.x = original.position.x + dx;
                    dest.position.y = original.position.y + dy;
                    dest.position.z = original.position.z + dz;
                    if let Some(priors) = &self.relational_priors {
                        self.apply_factor_bias(dest, Some(member_id), snapshot_0, priors, rng, cache);
                    }
                }
            }
        }
        proposal
    }

    fn swap_move_parallel(&self, state: &DynamicState, rng: &mut StdRng) -> DynamicState {
        if state.symbol_states.len() < 2 {
            return state.clone();
        }
        let mut ids: Vec<String> = state.symbol_states.keys().cloned().collect();
        ids.shuffle(rng);
        let mut proposal = state.clone();
        for pair in ids.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            let first_id = &pair[0];
            let second_id = &pair[1];
            if let (Some(first), Some(second)) = (
                state.symbol_states.get(first_id),
                state.symbol_states.get(second_id),
            ) {
                if let Some(dest) = proposal.symbol_states.get_mut(first_id) {
                    dest.position = second.position;
                }
                if let Some(dest) = proposal.symbol_states.get_mut(second_id) {
                    dest.position = first.position;
                }
            }
        }
        proposal
    }

    fn path_move_parallel(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
        cache: &SnapshotCache,
    ) -> DynamicState {
        let Some(search_module) = &self.search else {
            return self.local_move_parallel(snapshot_0, state, temperature, rng, cache);
        };
        let mut target_state = DynamicState {
            timestamp: state.timestamp,
            symbol_states: HashMap::new(),
        };
        for symbol_id in state.symbol_states.keys() {
            if let Some(target) = self.target_position(cache, symbol_id) {
                target_state.symbol_states.insert(
                    symbol_id.clone(),
                    SymbolState {
                        position: target,
                        velocity: None,
                        internal_state: Default::default(),
                    },
                );
            }
        }
        if target_state.symbol_states.is_empty() {
            return state.clone();
        }
        let paths = search_module.compute_paths(snapshot_0, &target_state);
        let mut proposal = state.clone();
        for (symbol_id, path_result) in paths {
            let Some(current_state) = state.symbol_states.get(&symbol_id) else {
                continue;
            };
            let Some(target_point) = next_path_point(&path_result, &current_state.position) else {
                continue;
            };
            if let Some(dest) = proposal.symbol_states.get_mut(&symbol_id) {
                self.advance_symbol(dest, &target_point, temperature);
            }
        }
        proposal
    }

    fn global_move_parallel(
        &self,
        state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
    ) -> DynamicState {
        if state.symbol_states.is_empty() {
            return state.clone();
        }
        let step_scale = self.step_scale(temperature) * 1.5;
        let jitter = step_scale * 0.25;
        let dx = rng.gen_range(-step_scale..step_scale);
        let dy = rng.gen_range(-step_scale..step_scale);
        let dz = rng.gen_range(-step_scale..step_scale);
        let mut proposal = state.clone();
        for (symbol_id, original) in state.symbol_states.iter() {
            if let Some(dest) = proposal.symbol_states.get_mut(symbol_id) {
                dest.position.x = original.position.x + dx + rng.gen_range(-jitter..jitter);
                dest.position.y = original.position.y + dy + rng.gen_range(-jitter..jitter);
                dest.position.z = original.position.z + dz + rng.gen_range(-jitter..jitter);
            }
        }
        proposal
    }

    fn advance_symbol(&self, symbol_state: &mut SymbolState, target: &Position, temperature: f64) {
        let dx = target.x - symbol_state.position.x;
        let dy = target.y - symbol_state.position.y;
        let dz = target.z - symbol_state.position.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < f64::EPSILON {
            return;
        }
        let step = self.step_scale(temperature).min(dist);
        symbol_state.position.x += dx / dist * step;
        symbol_state.position.y += dy / dist * step;
        symbol_state.position.z += dz / dist * step;
    }

    fn ml_guided_move(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
    ) -> DynamicState {
        if let Some(model) = &self.ml_model {
            if let Some(targets) = model.propose_moves(snapshot_0, current_state, temperature) {
                let mut proposal = current_state.clone();
                for (symbol_id, target) in targets {
                    if let Some(state) = proposal.symbol_states.get_mut(&symbol_id) {
                        self.advance_symbol(state, &target, temperature);
                    }
                }
                return proposal;
            }
        }
        self.global_move(current_state.clone(), temperature, rng)
    }

    fn adapted_weights(&self, temperature: f64) -> (f64, f64, f64, f64, f64, f64) {
        if !self.config.adaptive_move_mixing {
            return (
                self.config.local_move_prob,
                self.config.group_move_prob,
                self.config.swap_move_prob,
                self.config.path_based_move_prob,
                self.config.global_move_prob,
                if self.ml_model.is_some() {
                    self.config.ml_guided_move_prob
                } else {
                    0.0
                },
            );
        }
        let mut local = self.config.local_move_prob;
        let group = self.config.group_move_prob;
        let mut swap = self.config.swap_move_prob;
        let mut path = self.config.path_based_move_prob;
        let mut global = self.config.global_move_prob;
        let mut ml_guided = if self.ml_model.is_some() {
            self.config.ml_guided_move_prob
        } else {
            0.0
        };
        if temperature > 0.8 {
            global += 0.2 * temperature;
            local += 0.05 * temperature;
            ml_guided += 0.1 * temperature;
        }
        if temperature < 0.6 {
            path += 0.4 * (0.6 - temperature);
        }
        if temperature < 0.5 {
            swap *= 0.5;
        }
        if self.ml_model.is_none() {
            ml_guided = 0.0;
        }
        (local, group, swap, path, global, ml_guided)
    }

    fn target_position(&self, cache: &SnapshotCache, symbol_id: &str) -> Option<Position> {
        cache
            .symbols
            .get(symbol_id)
            .and_then(|symbol| symbol.target_position)
    }
}

fn value_to_position(value: &Value) -> Option<Position> {
    match value {
        Value::Object(map) => {
            let x = map.get("x").and_then(|v| v.as_f64())?;
            let y = map.get("y").and_then(|v| v.as_f64())?;
            let z = map.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some(Position { x, y, z })
        }
        Value::Array(values) if values.len() >= 2 => {
            let x = values.get(0)?.as_f64()?;
            let y = values.get(1)?.as_f64()?;
            let z = values.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some(Position { x, y, z })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel_with_config(mut config: ProposalConfig) -> DefaultProposalKernel {
        config.adaptive_move_mixing = true;
        DefaultProposalKernel::new(config, 7, None, None, None)
    }

    #[test]
    fn adaptive_weights_boost_global_at_high_temperature() {
        let kernel = kernel_with_config(ProposalConfig::default());
        let (local, _, _, _, global, _) = kernel.adapted_weights(0.95);
        assert!(global > kernel.config.global_move_prob);
        assert!(local > kernel.config.local_move_prob);
    }

    #[test]
    fn adaptive_weights_shift_to_path_and_reduce_swap_when_cool() {
        let kernel = kernel_with_config(ProposalConfig::default());
        let (_, _, swap, path, _, ml) = kernel.adapted_weights(0.3);
        assert!(ml >= 0.0);
        assert!(path > kernel.config.path_based_move_prob);
        assert!(swap < kernel.config.swap_move_prob);
    }

    #[test]
    fn choose_move_type_defaults_to_local_when_weights_zero() {
        let mut config = ProposalConfig::default();
        config.local_move_prob = 0.0;
        config.group_move_prob = 0.0;
        config.swap_move_prob = 0.0;
        config.path_based_move_prob = 0.0;
        config.global_move_prob = 0.0;
        config.adaptive_move_mixing = false;
        let kernel = DefaultProposalKernel::new(config, 11, None, None, None);
        let mut rng = StdRng::seed_from_u64(123);
        assert!(matches!(
            kernel.choose_move_type(&mut rng, 0.5),
            MoveType::Local
        ));
    }
}

fn distance(a: &Position, b: &Position) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn nearest_exit_position_from_list(origin: &Position, exits: &[Position]) -> Option<Position> {
    exits
        .iter()
        .min_by(|a, b| {
            distance(a, origin)
                .partial_cmp(&distance(b, origin))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
}

fn next_path_point(path_result: &PathResult, current: &Position) -> Option<Position> {
    match path_result {
        PathResult::Feasible { waypoints, .. } => waypoints
            .iter()
            .skip(1)
            .find(|point| distance(point, current) > 1e-3)
            .or_else(|| waypoints.get(1))
            .cloned(),
        PathResult::Infeasible { .. } => None,
    }
}
