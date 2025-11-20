use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Timestamp, Trajectory};
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub trait MLHooks: Send + Sync {
    fn predict_next_positions(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        t_end: &Timestamp,
    ) -> HashMap<String, Position>;

    fn score_configuration(&self, state: &DynamicState) -> f64;

    fn update_from_data(&self, _trajectories: &[Trajectory]) {}
}

pub type MlHooksHandle = Arc<dyn MLHooks>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLBackendType {
    None,
    SimpleRules,
    GoalAnchor,
    Custom,
}

impl Default for MLBackendType {
    fn default() -> Self {
        MLBackendType::None
    }
}

pub struct NullMLHooks;

impl MLHooks for NullMLHooks {
    fn predict_next_positions(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        _t_end: &Timestamp,
    ) -> HashMap<String, Position> {
        snapshot_0
            .symbols
            .iter()
            .map(|symbol| (symbol.id.clone(), symbol.position))
            .collect()
    }

    fn score_configuration(&self, _state: &DynamicState) -> f64 {
        0.0
    }
}

pub struct SimpleRulesMLHooks {
    rng: Mutex<StdRng>,
}

impl SimpleRulesMLHooks {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl MLHooks for SimpleRulesMLHooks {
    fn predict_next_positions(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        _t_end: &Timestamp,
    ) -> HashMap<String, Position> {
        let exits: Vec<_> = snapshot_0
            .symbols
            .iter()
            .filter(|symbol| matches!(symbol.symbol_type, crate::schema::SymbolType::Exit))
            .collect();
        let mut rng = self.rng.lock();
        snapshot_0
            .symbols
            .iter()
            .map(|symbol| {
                let next_pos = exits
                    .choose(&mut *rng)
                    .map(|exit_symbol| Position {
                        x: (symbol.position.x + exit_symbol.position.x) / 2.0,
                        y: (symbol.position.y + exit_symbol.position.y) / 2.0,
                        z: (symbol.position.z + exit_symbol.position.z) / 2.0,
                    })
                    .unwrap_or(symbol.position);
                (symbol.id.clone(), next_pos)
            })
            .collect()
    }

    fn score_configuration(&self, state: &DynamicState) -> f64 {
        let mut score = 0.0;
        let mut prev_pos: Option<Position> = None;
        for symbol_state in state.symbol_states.values() {
            if let Some(prev) = prev_pos {
                let dx = symbol_state.position.x - prev.x;
                let dy = symbol_state.position.y - prev.y;
                score += (dx * dx + dy * dy).sqrt();
            }
            prev_pos = Some(symbol_state.position);
        }
        score
    }
}

pub fn create_ml_hooks(backend: MLBackendType, seed: u64) -> MlHooksHandle {
    match backend {
        MLBackendType::None => Arc::new(NullMLHooks),
        MLBackendType::SimpleRules => Arc::new(SimpleRulesMLHooks::new(seed)),
        MLBackendType::GoalAnchor => Arc::new(GoalAnchorMLHooks::new(seed)),
        MLBackendType::Custom => Arc::new(NullMLHooks),
    }
}

pub struct GoalAnchorMLHooks {
    anchors: RwLock<HashMap<String, GoalStats>>,
    fallback: SimpleRulesMLHooks,
    horizon_seconds: f64,
}

impl GoalAnchorMLHooks {
    pub fn new(seed: u64) -> Self {
        Self {
            anchors: RwLock::new(HashMap::new()),
            fallback: SimpleRulesMLHooks::new(seed + 17),
            horizon_seconds: 5.0,
        }
    }

    fn anchor_for(&self, symbol_id: &str) -> Option<GoalStats> {
        self.anchors.read().get(symbol_id).copied()
    }
}

impl MLHooks for GoalAnchorMLHooks {
    fn predict_next_positions(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        t_end: &Timestamp,
    ) -> HashMap<String, Position> {
        let dt = ((t_end.unix - snapshot_0.timestamp.unix) as f64).max(0.0);
        let lerp = (dt / self.horizon_seconds).clamp(0.0, 1.0);
        let fallback = self.fallback.predict_next_positions(snapshot_0, t_end);
        snapshot_0
            .symbols
            .iter()
            .map(|symbol| {
                let predicted = if let Some(anchor) = self.anchor_for(&symbol.id) {
                    Position {
                        x: symbol.position.x + (anchor.x - symbol.position.x) * lerp,
                        y: symbol.position.y + (anchor.y - symbol.position.y) * lerp,
                        z: symbol.position.z + (anchor.z - symbol.position.z) * lerp,
                    }
                } else {
                    fallback
                        .get(&symbol.id)
                        .copied()
                        .unwrap_or(symbol.position)
                };
                (symbol.id.clone(), predicted)
            })
            .collect()
    }

    fn score_configuration(&self, state: &DynamicState) -> f64 {
        let anchors = self.anchors.read();
        let mut score = 0.0;
        for (symbol_id, symbol_state) in state.symbol_states.iter() {
            if let Some(anchor) = anchors.get(symbol_id) {
                let dx = symbol_state.position.x - anchor.x;
                let dy = symbol_state.position.y - anchor.y;
                let dz = symbol_state.position.z - anchor.z;
                score += dx * dx + dy * dy + dz * dz;
            }
        }
        score
    }

    fn update_from_data(&self, trajectories: &[Trajectory]) {
        let mut anchors = self.anchors.write();
        for trajectory in trajectories {
            let Some(last) = trajectory.sequence.last() else {
                continue;
            };
            for (symbol_id, symbol_state) in last.symbol_states.iter() {
                let entry = anchors.entry(symbol_id.clone()).or_insert(GoalStats {
                    x: symbol_state.position.x,
                    y: symbol_state.position.y,
                    z: symbol_state.position.z,
                    count: 1.0,
                });
                let count = entry.count + 1.0;
                entry.x = (entry.x * entry.count + symbol_state.position.x) / count;
                entry.y = (entry.y * entry.count + symbol_state.position.y) / count;
                entry.z = (entry.z * entry.count + symbol_state.position.z) / count;
                entry.count = count;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct GoalStats {
    x: f64,
    y: f64,
    z: f64,
    count: f64,
}



