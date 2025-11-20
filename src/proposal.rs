use crate::schema::{DynamicState, EnvironmentSnapshot, Position, SymbolState, SymbolType};
use crate::search::{PathResult, SearchModule};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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
    pub max_step_size: f64,
    #[serde(default)]
    pub use_parallel_updates: bool,
    #[serde(default)]
    pub adaptive_move_mixing: bool,
}

#[derive(Debug, Clone, Copy)]
enum MoveType {
    Local,
    Group,
    Swap,
    Path,
    Global,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            local_move_prob: 0.9,
            group_move_prob: 0.05,
            swap_move_prob: 0.05,
            path_based_move_prob: 0.0,
            global_move_prob: 0.0,
            max_step_size: 0.75,
            use_parallel_updates: false,
            adaptive_move_mixing: true,
        }
    }
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
}

impl DefaultProposalKernel {
    pub fn new(config: ProposalConfig, seed: u64, search: Option<SearchModule>) -> Self {
        Self {
            config,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            search,
        }
    }
}

impl ProposalKernel for DefaultProposalKernel {
    fn propose(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
    ) -> DynamicState {
        let mut rng = self.rng.lock();
        let move_type = self.choose_move_type(&mut rng, temperature);
        if self.config.use_parallel_updates {
            return self.apply_parallel_move(
                move_type,
                snapshot_0,
                current_state,
                temperature,
                &mut rng,
            );
        }
        let proposal = current_state.clone();
        match move_type {
            MoveType::Local => self.local_move(proposal, temperature, &mut rng),
            MoveType::Group => self.group_move(snapshot_0, proposal, temperature, &mut rng),
            MoveType::Swap => self.swap_move(proposal, &mut rng),
            MoveType::Path => self.path_move(snapshot_0, proposal, temperature, &mut rng),
            MoveType::Global => self.global_move(proposal, temperature, &mut rng),
        }
    }
}

impl DefaultProposalKernel {
    fn choose_move_type(&self, rng: &mut StdRng, temperature: f64) -> MoveType {
        let (local, group, swap, path, global) = self.adapted_weights(temperature);
        let total_prob = local + group + swap + path + global;
        if total_prob <= f64::EPSILON {
            return MoveType::Local;
        }
        let roll = rng.gen_range(0.0..1.0) * total_prob;
        let local_threshold = local;
        let group_threshold = local_threshold + group;
        let swap_threshold = group_threshold + swap;
        let path_threshold = swap_threshold + path;
        if roll < local_threshold {
            MoveType::Local
        } else if roll < group_threshold {
            MoveType::Group
        } else if roll < swap_threshold {
            MoveType::Swap
        } else if roll < path_threshold {
            MoveType::Path
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
    ) -> DynamicState {
        match move_type {
            MoveType::Local => self.local_move_parallel(current_state, temperature, rng),
            MoveType::Group => {
                self.group_move_parallel(snapshot_0, current_state, temperature, rng)
            }
            MoveType::Swap => self.swap_move_parallel(current_state, rng),
            MoveType::Path => self.path_move_parallel(snapshot_0, current_state, temperature, rng),
            MoveType::Global => self.global_move_parallel(current_state, temperature, rng),
        }
    }

    fn step_scale(&self, temperature: f64) -> f64 {
        self.config.max_step_size * temperature.max(0.05)
    }

    fn local_move(
        &self,
        mut proposal: DynamicState,
        temperature: f64,
        rng: &mut StdRng,
    ) -> DynamicState {
        let Some(symbol_state) = proposal.symbol_states.values_mut().choose(rng) else {
            return proposal;
        };
        let step_scale = self.step_scale(temperature);
        let mut delta = || rng.gen_range(-step_scale..step_scale);
        symbol_state.position.x += delta();
        symbol_state.position.y += delta();
        symbol_state.position.z += delta();
        proposal
    }

    fn group_move(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        mut proposal: DynamicState,
        temperature: f64,
        rng: &mut StdRng,
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
            return self.local_move(proposal, temperature, rng);
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
    ) -> DynamicState {
        let Some(search_module) = &self.search else {
            return self.local_move(proposal, temperature, rng);
        };
        let Some((symbol_id, symbol_state)) = proposal
            .symbol_states
            .iter_mut()
            .choose(rng)
            .map(|(id, state)| (id.clone(), state))
        else {
            return proposal;
        };
        let Some(target) = self.target_position(snapshot_0, &symbol_id) else {
            return self.local_move(proposal, temperature, rng);
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
            return self.local_move(proposal, temperature, rng);
        };
        let Some(target_point) = next_path_point(path_result, &symbol_state.position) else {
            return proposal;
        };
        self.advance_symbol(symbol_state, &target_point, temperature);
        proposal
    }

    fn local_move_parallel(
        &self,
        state: &DynamicState,
        temperature: f64,
        rng: &mut StdRng,
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
            return self.local_move_parallel(state, temperature, rng);
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
    ) -> DynamicState {
        let Some(search_module) = &self.search else {
            return self.local_move_parallel(state, temperature, rng);
        };
        let mut target_state = DynamicState {
            timestamp: state.timestamp,
            symbol_states: HashMap::new(),
        };
        for symbol_id in state.symbol_states.keys() {
            if let Some(target) = self.target_position(snapshot_0, symbol_id) {
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

    fn adapted_weights(&self, temperature: f64) -> (f64, f64, f64, f64, f64) {
        if !self.config.adaptive_move_mixing {
            return (
                self.config.local_move_prob,
                self.config.group_move_prob,
                self.config.swap_move_prob,
                self.config.path_based_move_prob,
                self.config.global_move_prob,
            );
        }
        let mut local = self.config.local_move_prob;
        let group = self.config.group_move_prob;
        let mut swap = self.config.swap_move_prob;
        let mut path = self.config.path_based_move_prob;
        let mut global = self.config.global_move_prob;
        if temperature > 0.8 {
            global += 0.2 * temperature;
            local += 0.05 * temperature;
        }
        if temperature < 0.6 {
            path += 0.4 * (0.6 - temperature);
        }
        if temperature < 0.5 {
            swap *= 0.5;
        }
        (local, group, swap, path, global)
    }

    fn target_position(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        symbol_id: &str,
    ) -> Option<Position> {
        let symbol = snapshot_0.symbols.iter().find(|s| s.id == symbol_id)?;
        symbol
            .properties
            .get("goal_position")
            .and_then(value_to_position)
            .or_else(|| symbol.properties.get("goal").and_then(value_to_position))
            .or_else(|| {
                symbol
                    .properties
                    .get("goal_id")
                    .and_then(|value| value.as_str())
                    .and_then(|goal_id| {
                        snapshot_0
                            .symbols
                            .iter()
                            .find(|candidate| candidate.id == goal_id)
                    })
                    .map(|goal_symbol| goal_symbol.position)
            })
            .or_else(|| self.nearest_exit_position(snapshot_0, &symbol.position))
    }

    fn nearest_exit_position(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        origin: &Position,
    ) -> Option<Position> {
        snapshot_0
            .symbols
            .iter()
            .filter(|symbol| matches!(symbol.symbol_type, SymbolType::Exit))
            .min_by(|a, b| {
                distance(&a.position, origin)
                    .partial_cmp(&distance(&b.position, origin))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|exit| exit.position)
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
        DefaultProposalKernel::new(config, 7, None)
    }

    #[test]
    fn adaptive_weights_boost_global_at_high_temperature() {
        let kernel = kernel_with_config(ProposalConfig::default());
        let (local, _, _, _, global) = kernel.adapted_weights(0.95);
        assert!(global > kernel.config.global_move_prob);
        assert!(local > kernel.config.local_move_prob);
    }

    #[test]
    fn adaptive_weights_shift_to_path_and_reduce_swap_when_cool() {
        let kernel = kernel_with_config(ProposalConfig::default());
        let (_, _, swap, path, _) = kernel.adapted_weights(0.3);
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
        let kernel = DefaultProposalKernel::new(config, 11, None);
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
