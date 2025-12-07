use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Timestamp, Trajectory};
use parking_lot::{Mutex, RwLock};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub trait MLModel: Send + Sync {
    fn predict_next_positions(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        t_end: &Timestamp,
    ) -> HashMap<String, Position>;

    fn score_state(&self, state: &DynamicState) -> f64;

    fn propose_moves(
        &self,
        _snapshot_0: &EnvironmentSnapshot,
        _state: &DynamicState,
        _temperature: f64,
    ) -> Option<HashMap<String, Position>> {
        None
    }

    fn update_from_data(&self, _trajectories: &[Trajectory]) {}
}

pub type MlModelHandle = Arc<dyn MLModel>;

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

pub struct NullMLModel;

impl MLModel for NullMLModel {
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

    fn score_state(&self, _state: &DynamicState) -> f64 {
        0.0
    }
}

pub struct SimpleRulesMLModel {
    rng: Mutex<StdRng>,
}

impl SimpleRulesMLModel {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl MLModel for SimpleRulesMLModel {
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

    fn score_state(&self, state: &DynamicState) -> f64 {
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

    fn propose_moves(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        state: &DynamicState,
        _temperature: f64,
    ) -> Option<HashMap<String, Position>> {
        Some(self.predict_next_positions(snapshot_0, &state.timestamp))
    }
}

pub fn create_ml_model(backend: MLBackendType, seed: u64) -> MlModelHandle {
    match backend {
        MLBackendType::None => Arc::new(NullMLModel),
        MLBackendType::SimpleRules => Arc::new(SimpleRulesMLModel::new(seed)),
        MLBackendType::GoalAnchor => Arc::new(GoalAnchorMLModel::new(seed)),
        MLBackendType::Custom => Arc::new(NullMLModel),
    }
}

pub struct GoalAnchorMLModel {
    anchors: RwLock<HashMap<String, GoalStats>>,
    fallback: SimpleRulesMLModel,
    horizon_seconds: f64,
}

impl GoalAnchorMLModel {
    pub fn new(seed: u64) -> Self {
        Self {
            anchors: RwLock::new(HashMap::new()),
            fallback: SimpleRulesMLModel::new(seed + 17),
            horizon_seconds: 5.0,
        }
    }

    fn anchor_for(&self, symbol_id: &str) -> Option<GoalStats> {
        self.anchors.read().get(symbol_id).copied()
    }
}

impl MLModel for GoalAnchorMLModel {
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
                    fallback.get(&symbol.id).copied().unwrap_or(symbol.position)
                };
                (symbol.id.clone(), predicted)
            })
            .collect()
    }

    fn score_state(&self, state: &DynamicState) -> f64 {
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

    fn propose_moves(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        state: &DynamicState,
        _temperature: f64,
    ) -> Option<HashMap<String, Position>> {
        Some(self.predict_next_positions(snapshot_0, &state.timestamp))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{EnvironmentSnapshot, Properties, Symbol, SymbolState, SymbolType};

    fn snapshot_at(position: Position) -> EnvironmentSnapshot {
        EnvironmentSnapshot {
            timestamp: Timestamp { unix: 0 },
            bounds: HashMap::from([("width".into(), 10.0), ("height".into(), 10.0)]),
            symbols: vec![Symbol {
                id: "p1".into(),
                symbol_type: SymbolType::Person,
                position,
                properties: Properties::new(),
            }],
            metadata: Properties::new(),
            stack_history: Vec::new(),
        }
    }

    #[test]
    fn goal_anchor_learns_and_scores_zero_at_anchor() {
        let hooks = GoalAnchorMLModel::new(9);
        let mut last_state = DynamicState::default();
        last_state.symbol_states.insert(
            "p1".into(),
            SymbolState {
                position: Position {
                    x: 4.0,
                    y: 6.0,
                    z: 0.0,
                },
                ..Default::default()
            },
        );
        let trajectory = Trajectory {
            sequence: vec![last_state.clone()],
            metadata: Properties::new(),
        };
        hooks.update_from_data(&[trajectory]);
        let anchor = hooks.anchor_for("p1").expect("anchor exists");
        assert!((anchor.x - 4.0).abs() < 1e-6 && (anchor.y - 6.0).abs() < 1e-6);

        assert!(
            hooks.score_state(&last_state) < 1e-6,
            "state at anchor should score ~0"
        );
    }

    #[test]
    fn goal_anchor_predictions_follow_learned_target() {
        let hooks = GoalAnchorMLModel::new(1);
        let mut last_state = DynamicState::default();
        last_state.symbol_states.insert(
            "p1".into(),
            SymbolState {
                position: Position {
                    x: 8.0,
                    y: 2.0,
                    z: 0.0,
                },
                ..Default::default()
            },
        );
        hooks.update_from_data(&[Trajectory {
            sequence: vec![last_state],
            metadata: Properties::new(),
        }]);
        let snapshot = snapshot_at(Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        let t_end = Timestamp {
            unix: snapshot.timestamp.unix + 5,
        };
        let predictions = hooks.predict_next_positions(&snapshot, &t_end);
        let predicted = predictions.get("p1").unwrap();
        assert!(
            (predicted.x - 8.0).abs() < 1e-6 && (predicted.y - 2.0).abs() < 1e-6,
            "prediction should hit anchor when horizon elapsed"
        );
    }
}
