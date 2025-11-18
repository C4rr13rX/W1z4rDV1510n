use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Timestamp, Trajectory};
use rand::SeedableRng;
use rand::seq::SliceRandom;
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
    Rnn,
    Transformer,
    Gnn,
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
    rng: parking_lot::Mutex<rand::rngs::StdRng>,
}

impl SimpleRulesMLHooks {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: parking_lot::Mutex::new(rand::rngs::StdRng::seed_from_u64(seed)),
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
        MLBackendType::Rnn
        | MLBackendType::Transformer
        | MLBackendType::Gnn
        | MLBackendType::Custom => Arc::new(NullMLHooks),
    }
}
