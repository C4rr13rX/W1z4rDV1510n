use crate::config::EnergyConfig;
use crate::ml::MlModelHandle;
use crate::schema::{
    DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolState, SymbolType, Timestamp,
};
use crate::search::{PathResult, SearchModule};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub trait EnergyEvaluator {
    fn energy(&self, state: &DynamicState) -> f64;
    fn batch_energy(&self, states: &[DynamicState]) -> Vec<f64>;
    fn breakdown(&self, state: &DynamicState) -> EnergyBreakdown;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBreakdown {
    pub total: f64,
    pub per_symbol: HashMap<String, SymbolEnergyBreakdown>,
    pub per_term: TermEnergyTotals,
}

impl EnergyBreakdown {
    fn new(
        per_symbol: HashMap<String, SymbolEnergyBreakdown>,
        per_term: TermEnergyTotals,
        total: f64,
    ) -> Self {
        Self {
            total,
            per_symbol,
            per_term,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolEnergyBreakdown {
    pub motion: f64,
    pub collision: f64,
    pub goal: f64,
    pub group: f64,
    pub environment: f64,
    pub path: f64,
    pub ml_prior: f64,
}

impl SymbolEnergyBreakdown {
    pub fn total(&self) -> f64 {
        self.motion
            + self.collision
            + self.goal
            + self.group
            + self.environment
            + self.path
            + self.ml_prior
    }

    fn add(&mut self, term: EnergyTerm, value: f64) {
        match term {
            EnergyTerm::Motion => self.motion += value,
            EnergyTerm::Collision => self.collision += value,
            EnergyTerm::Goal => self.goal += value,
            EnergyTerm::Group => self.group += value,
            EnergyTerm::Environment => self.environment += value,
            EnergyTerm::Path => self.path += value,
            EnergyTerm::MlPrior => self.ml_prior += value,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TermEnergyTotals {
    pub motion: f64,
    pub collision: f64,
    pub goal: f64,
    pub group: f64,
    pub environment: f64,
    pub path: f64,
    pub ml_prior: f64,
}

impl TermEnergyTotals {
    fn add(&mut self, term: EnergyTerm, value: f64) {
        match term {
            EnergyTerm::Motion => self.motion += value,
            EnergyTerm::Collision => self.collision += value,
            EnergyTerm::Goal => self.goal += value,
            EnergyTerm::Group => self.group += value,
            EnergyTerm::Environment => self.environment += value,
            EnergyTerm::Path => self.path += value,
            EnergyTerm::MlPrior => self.ml_prior += value,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EnergyTerm {
    Motion,
    Collision,
    Goal,
    Group,
    Environment,
    Path,
    MlPrior,
}

#[derive(Default)]
struct TermContribution {
    total: f64,
    per_symbol: HashMap<String, f64>,
}

impl TermContribution {
    fn add_symbol(&mut self, symbol_id: &str, value: f64) {
        if value == 0.0 {
            return;
        }
        self.total += value;
        *self.per_symbol.entry(symbol_id.to_string()).or_insert(0.0) += value;
    }

    fn add_pair(&mut self, first: &str, second: &str, value: f64) {
        if value == 0.0 {
            return;
        }
        let half = value / 2.0;
        self.add_symbol(first, half);
        self.add_symbol(second, half);
    }

    fn add_direct(&mut self, value: f64) {
        self.total += value;
    }
}

#[derive(Debug, Clone)]
struct WallPrimitive {
    center: Position,
    half_extents: Position,
}

impl WallPrimitive {
    fn from_symbol(symbol: &Symbol) -> Self {
        let half_x = property_as_f64(symbol, &["width", "length", "size_x"])
            .unwrap_or(1.0)
            .max(0.1)
            / 2.0;
        let half_y = property_as_f64(symbol, &["height", "size_y"])
            .unwrap_or(1.0)
            .max(0.1)
            / 2.0;
        let half_z = property_as_f64(symbol, &["thickness", "size_z"])
            .unwrap_or(1.0)
            .max(0.1)
            / 2.0;
        Self {
            center: symbol.position,
            half_extents: Position {
                x: half_x,
                y: half_y,
                z: half_z,
            },
        }
    }

    fn signed_distance(&self, point: &Position) -> f64 {
        let dx = (point.x - self.center.x).abs() - self.half_extents.x;
        let dy = (point.y - self.center.y).abs() - self.half_extents.y;
        let dz = (point.z - self.center.z).abs() - self.half_extents.z;
        let outside = Position {
            x: dx.max(0.0),
            y: dy.max(0.0),
            z: dz.max(0.0),
        };
        if dx <= 0.0 && dy <= 0.0 && dz <= 0.0 {
            dx.max(dy).max(dz)
        } else {
            (outside.x * outside.x + outside.y * outside.y + outside.z * outside.z).sqrt()
        }
    }
}

pub struct EnergyModel {
    snapshot_0: Arc<EnvironmentSnapshot>,
    config: EnergyConfig,
    symbol_lookup: HashMap<String, Symbol>,
    exit_symbols: Vec<Symbol>,
    ml_model: Option<MlModelHandle>,
    ml_predictions: Option<HashMap<String, Position>>,
    search_module: Option<SearchModule>,
    walls: Vec<WallPrimitive>,
}

impl EnergyModel {
    pub fn new(
        snapshot_0: Arc<EnvironmentSnapshot>,
        config: EnergyConfig,
        ml_model: Option<MlModelHandle>,
        search_module: Option<SearchModule>,
        target_time: Timestamp,
    ) -> Self {
        let symbol_lookup = snapshot_0
            .symbols
            .iter()
            .cloned()
            .map(|symbol| (symbol.id.clone(), symbol))
            .collect::<HashMap<_, _>>();
        let exit_symbols = snapshot_0
            .symbols
            .iter()
            .filter(|symbol| matches!(symbol.symbol_type, SymbolType::Exit))
            .cloned()
            .collect();
        let walls = snapshot_0
            .symbols
            .iter()
            .filter(|symbol| matches!(symbol.symbol_type, SymbolType::Wall))
            .map(WallPrimitive::from_symbol)
            .collect();
        let ml_predictions = ml_model
            .as_ref()
            .map(|hooks| hooks.predict_next_positions(snapshot_0.as_ref(), &target_time));
        Self {
            snapshot_0,
            config,
            symbol_lookup,
            exit_symbols,
            ml_model,
            ml_predictions,
            search_module,
            walls,
        }
    }

    pub fn energy(&self, dynamic_state: &DynamicState) -> f64 {
        self.energy_breakdown(dynamic_state).total
    }

    pub fn batch_energy(&self, states: &[DynamicState]) -> Vec<f64> {
        states.iter().map(|state| self.energy(state)).collect()
    }

    pub fn energy_breakdown(&self, dynamic_state: &DynamicState) -> EnergyBreakdown {
        let cfg = &self.config;
        let mut per_symbol = HashMap::new();
        let mut per_term = TermEnergyTotals::default();
        let mut total = 0.0;

        self.accumulate_term(
            &mut per_symbol,
            &mut per_term,
            &mut total,
            EnergyTerm::Motion,
            cfg.w_motion,
            self.motion_term_breakdown(dynamic_state),
        );
        self.accumulate_term(
            &mut per_symbol,
            &mut per_term,
            &mut total,
            EnergyTerm::Collision,
            cfg.w_collision,
            self.collision_term_breakdown(dynamic_state),
        );
        self.accumulate_term(
            &mut per_symbol,
            &mut per_term,
            &mut total,
            EnergyTerm::Goal,
            cfg.w_goal,
            self.goal_term_breakdown(dynamic_state),
        );
        self.accumulate_term(
            &mut per_symbol,
            &mut per_term,
            &mut total,
            EnergyTerm::Group,
            cfg.w_group_cohesion,
            self.group_term_breakdown(dynamic_state),
        );
        self.accumulate_term(
            &mut per_symbol,
            &mut per_term,
            &mut total,
            EnergyTerm::Environment,
            cfg.w_env_constraints,
            self.environment_term_breakdown(dynamic_state),
        );
        self.accumulate_term(
            &mut per_symbol,
            &mut per_term,
            &mut total,
            EnergyTerm::Path,
            cfg.w_path_feasibility,
            self.path_term_breakdown(dynamic_state),
        );
        if cfg.w_ml_prior > 0.0 {
            self.accumulate_term(
                &mut per_symbol,
                &mut per_term,
                &mut total,
                EnergyTerm::MlPrior,
                cfg.w_ml_prior,
                self.ml_term_breakdown(dynamic_state),
            );
        }

        EnergyBreakdown::new(per_symbol, per_term, total)
    }

    pub fn local_energy(&self, dynamic_state: &DynamicState, symbol_id: &str) -> f64 {
        let breakdown = self.energy_breakdown(dynamic_state);
        breakdown
            .per_symbol
            .get(symbol_id)
            .map(SymbolEnergyBreakdown::total)
            .unwrap_or(0.0)
    }

    fn accumulate_term(
        &self,
        per_symbol: &mut HashMap<String, SymbolEnergyBreakdown>,
        per_term: &mut TermEnergyTotals,
        total: &mut f64,
        term: EnergyTerm,
        weight: f64,
        contribution: TermContribution,
    ) {
        let TermContribution {
            total: raw_total,
            per_symbol: symbol_contribs,
        } = contribution;
        let weighted_total = raw_total * weight;
        if weighted_total.abs() > f64::EPSILON {
            per_term.add(term, weighted_total);
            *total += weighted_total;
        }
        for (symbol_id, value) in symbol_contribs {
            if value.abs() <= f64::EPSILON {
                continue;
            }
            per_symbol
                .entry(symbol_id)
                .or_default()
                .add(term, value * weight);
        }
    }

    fn motion_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let mut contribution = TermContribution::default();
        for (symbol_id, symbol_state) in state.symbol_states.iter() {
            if let Some(origin) = self.symbol_lookup.get(symbol_id) {
                let dist_sq = distance_squared(&origin.position, &symbol_state.position);
                contribution.add_symbol(symbol_id, dist_sq);
            }
        }
        contribution
    }

    fn collision_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let mut contribution = TermContribution::default();
        let entries: Vec<_> = state.symbol_states.iter().collect();
        for (idx, (first_id, first_state)) in entries.iter().enumerate() {
            for (second_id, second_state) in entries.iter().skip(idx + 1) {
                let r_first = self.symbol_radius(first_id);
                let r_second = self.symbol_radius(second_id);
                if r_first == 0.0 && r_second == 0.0 {
                    continue;
                }
                let dist = distance(&first_state.position, &second_state.position);
                let clearance = dist - (r_first + r_second);
                if clearance < 0.0 {
                    let overlap = -clearance;
                    let penalty = overlap * overlap + 1.0 / dist.max(1e-3);
                    contribution.add_pair(first_id, second_id, penalty);
                } else if clearance < 0.5 {
                    let penalty = (0.5 - clearance).powi(2);
                    contribution.add_pair(first_id, second_id, penalty);
                }
            }
        }
        contribution
    }

    fn goal_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let mut contribution = TermContribution::default();
        if self.exit_symbols.is_empty() && self.symbol_lookup.is_empty() {
            return contribution;
        }
        for (symbol_id, symbol_state) in state.symbol_states.iter() {
            let fallback = symbol_state.position;
            if let Some(goal_pos) = self.goal_position(symbol_id, &fallback) {
                let dist = distance(&symbol_state.position, &goal_pos);
                contribution.add_symbol(symbol_id, dist);
            }
        }
        contribution
    }

    fn group_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let mut groups: HashMap<String, Vec<GroupMember<'_>>> = HashMap::new();
        for (symbol_id, sym_state) in state.symbol_states.iter() {
            if let Some(origin) = self.symbol_lookup.get(symbol_id) {
                if let Some(group_id) = origin.properties.get("group_id").and_then(|v| v.as_str()) {
                    let spacing = origin
                        .properties
                        .get("group_spacing")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);
                    groups
                        .entry(group_id.to_string())
                        .or_default()
                        .push(GroupMember {
                            id: symbol_id.as_str(),
                            state: sym_state,
                            spacing,
                        });
                }
            }
        }

        let mut contribution = TermContribution::default();
        for members in groups.values() {
            if members.len() < 2 {
                continue;
            }
            let state_refs: Vec<_> = members.iter().map(|member| member.state).collect();
            let centroid_pos = centroid(&state_refs);
            for member in members {
                let dist = distance(&member.state.position, &centroid_pos);
                contribution.add_symbol(member.id, dist);
            }
            for (idx, first) in members.iter().enumerate() {
                for second in members.iter().skip(idx + 1) {
                    let preferred = (first.spacing + second.spacing) / 2.0;
                    let actual = distance(&first.state.position, &second.state.position);
                    let delta = (actual - preferred).abs();
                    contribution.add_pair(first.id, second.id, delta);
                }
            }
        }
        contribution
    }

    fn environment_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let bounds = &self.snapshot_0.bounds;
        let width = bounds.get("width").copied().unwrap_or(0.0);
        let height = bounds.get("height").copied().unwrap_or(0.0);
        let depth = bounds.get("depth").copied().unwrap_or(f64::INFINITY);
        let mut contribution = TermContribution::default();
        for (symbol_id, symbol_state) in state.symbol_states.iter() {
            let pos = symbol_state.position;
            let mut penalty = boundary_penalty(pos.x, 0.0, width)
                + boundary_penalty(pos.y, 0.0, height)
                + boundary_penalty(pos.z, 0.0, depth);
            if let Some(clearance) = self.nearest_wall_distance(&pos) {
                if clearance < 0.0 {
                    penalty += (-clearance + 0.1).powi(2) * 10.0;
                } else if clearance < 0.5 {
                    penalty += (0.5 - clearance) * 2.0;
                }
            }
            contribution.add_symbol(symbol_id, penalty);
        }
        contribution
    }

    fn path_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let mut contribution = TermContribution::default();
        let Some(search_module) = &self.search_module else {
            return contribution;
        };
        let paths = search_module.compute_paths(&self.snapshot_0, state);
        for (symbol_id, path) in paths {
            match path {
                PathResult::Feasible { diagnostics, .. } => {
                    let mut effort = diagnostics.cost.max(1.0);
                    if diagnostics.constraint_violations > 0 {
                        effort += diagnostics.constraint_violations as f64 * 10.0;
                    }
                    contribution.add_symbol(&symbol_id, effort);
                }
                PathResult::Infeasible { diagnostics, .. } => {
                    let mut penalty = 50.0 + diagnostics.cost.max(1.0);
                    if diagnostics.constraint_violations > 0 {
                        penalty += diagnostics.constraint_violations as f64 * 15.0;
                    }
                    contribution.add_symbol(&symbol_id, penalty);
                }
            }
        }
        contribution
    }

    fn ml_term_breakdown(&self, state: &DynamicState) -> TermContribution {
        let mut contribution = TermContribution::default();
        if let Some(predictions) = &self.ml_predictions {
            for (symbol_id, symbol_state) in state.symbol_states.iter() {
                if let Some(pred_position) = predictions.get(symbol_id) {
                    let penalty = distance_squared(&symbol_state.position, pred_position);
                    contribution.add_symbol(symbol_id, penalty);
                }
            }
            return contribution;
        }
        if let Some(model) = &self.ml_model {
            contribution.add_direct(model.score_state(state));
        }
        contribution
    }

    fn symbol_radius(&self, symbol_id: &str) -> f64 {
        self.symbol_lookup
            .get(symbol_id)
            .and_then(|symbol| property_as_f64(symbol, &["radius", "collision_radius", "size"]))
            .unwrap_or(0.5)
    }

    fn goal_position(&self, symbol_id: &str, fallback: &Position) -> Option<Position> {
        let symbol = self.symbol_lookup.get(symbol_id)?;
        symbol
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
                    .and_then(|goal_id| self.symbol_lookup.get(goal_id))
                    .map(|target| target.position)
            })
            .or_else(|| self.nearest_exit_position(fallback))
    }

    fn nearest_exit_position(&self, position: &Position) -> Option<Position> {
        self.exit_symbols
            .iter()
            .min_by(|first, second| {
                distance(&first.position, position)
                    .partial_cmp(&distance(&second.position, position))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|symbol| symbol.position)
    }

    fn nearest_wall_distance(&self, position: &Position) -> Option<f64> {
        self.walls
            .iter()
            .map(|wall| wall.signed_distance(position))
            .reduce(|a, b| if a.abs() < b.abs() { a } else { b })
    }
}

struct GroupMember<'a> {
    id: &'a str,
    state: &'a SymbolState,
    spacing: f64,
}

fn distance(a: &Position, b: &Position) -> f64 {
    distance_squared(a, b).sqrt()
}

fn distance_squared(a: &Position, b: &Position) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn centroid(states: &[&SymbolState]) -> Position {
    let count = states.len() as f64;
    let (sum_x, sum_y, sum_z) = states.iter().fold((0.0, 0.0, 0.0), |acc, state| {
        (
            acc.0 + state.position.x,
            acc.1 + state.position.y,
            acc.2 + state.position.z,
        )
    });
    Position {
        x: sum_x / count,
        y: sum_y / count,
        z: sum_z / count,
    }
}

fn boundary_penalty(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        (min - value + 1e-3).powi(2)
    } else if value > max {
        (value - max + 1e-3).powi(2)
    } else {
        0.0
    }
}

fn property_as_f64(symbol: &Symbol, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| symbol.properties.get(*key))
        .and_then(|value| value.as_f64())
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

impl EnergyEvaluator for EnergyModel {
    fn energy(&self, state: &DynamicState) -> f64 {
        EnergyModel::energy(self, state)
    }

    fn batch_energy(&self, states: &[DynamicState]) -> Vec<f64> {
        self.batch_energy(states)
    }

    fn breakdown(&self, state: &DynamicState) -> EnergyBreakdown {
        self.energy_breakdown(state)
    }
}
