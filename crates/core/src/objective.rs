use crate::schema::{DynamicState, Position};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRule {
    /// Optional symbol id to which this region applies; if None, applies to all symbols.
    pub symbol_id: Option<String>,
    pub min: Position,
    pub max: Position,
    /// Allowed slack when checking bounds.
    #[serde(default = "RegionRule::default_tolerance")]
    pub tolerance: f64,
}

impl RegionRule {
    fn default_tolerance() -> f64 {
        0.0
    }

    pub fn contains(&self, pos: &Position) -> bool {
        let tol = self.tolerance.max(0.0);
        pos.x + tol >= self.min.x
            && pos.x - tol <= self.max.x
            && pos.y + tol >= self.min.y
            && pos.y - tol <= self.max.y
            && pos.z + tol >= self.min.z
            && pos.z - tol <= self.max.z
    }

    pub fn miss_distance(&self, pos: &Position) -> f64 {
        let dx = if pos.x < self.min.x {
            self.min.x - pos.x
        } else if pos.x > self.max.x {
            pos.x - self.max.x
        } else {
            0.0
        };
        let dy = if pos.y < self.min.y {
            self.min.y - pos.y
        } else if pos.y > self.max.y {
            pos.y - self.max.y
        } else {
            0.0
        };
        let dz = if pos.z < self.min.z {
            self.min.z - pos.z
        } else if pos.z > self.max.z {
            pos.z - self.max.z
        } else {
            0.0
        };
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameObjectiveConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub win_regions: Vec<RegionRule>,
    #[serde(default)]
    pub win_symbols: Vec<String>,
    #[serde(default = "GameObjectiveConfig::default_win_reward")]
    pub win_reward: f64,
    #[serde(default = "GameObjectiveConfig::default_step_penalty")]
    pub step_penalty: f64,
}

impl GameObjectiveConfig {
    fn default_win_reward() -> f64 {
        5.0
    }
    fn default_step_penalty() -> f64 {
        0.0
    }
}

impl Default for GameObjectiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            win_regions: Vec::new(),
            win_symbols: Vec::new(),
            win_reward: Self::default_win_reward(),
            step_penalty: Self::default_step_penalty(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ObjectiveOutcome {
    pub win: bool,
    pub miss_cost: f64,
    pub step_penalty: f64,
}

#[derive(Debug, Clone)]
pub struct GameObjective {
    cfg: GameObjectiveConfig,
}

impl GameObjective {
    pub fn new(cfg: GameObjectiveConfig) -> Option<Self> {
        if cfg.enabled {
            Some(Self { cfg })
        } else {
            None
        }
    }

    pub fn evaluate(&self, state: &DynamicState) -> ObjectiveOutcome {
        let mut miss_cost = 0.0;
        let mut satisfied = true;
        for region in &self.cfg.win_regions {
            if let Some(target_symbol) = &region.symbol_id {
                if let Some(sym) = state.symbol_states.get(target_symbol) {
                    if !region.contains(&sym.position) {
                        satisfied = false;
                        miss_cost += region.miss_distance(&sym.position);
                    }
                } else {
                    satisfied = false;
                    miss_cost += 1.0;
                }
            } else {
                for sym in state.symbol_states.values() {
                    if !region.contains(&sym.position) {
                        satisfied = false;
                        miss_cost += region.miss_distance(&sym.position);
                    }
                }
            }
        }
        for sym_id in &self.cfg.win_symbols {
            if !state.symbol_states.contains_key(sym_id) {
                satisfied = false;
                miss_cost += 1.0;
            }
        }
        ObjectiveOutcome {
            win: satisfied,
            miss_cost,
            step_penalty: self.cfg.step_penalty,
        }
    }

    pub fn reward(&self) -> f64 {
        self.cfg.win_reward
    }
}
