use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Properties = HashMap<String, serde_json::Value>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Timestamp {
    pub unix: i64,
}

impl Default for Timestamp {
    fn default() -> Self {
        Self { unix: 0 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SymbolType {
    Person,
    Object,
    Wall,
    Exit,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub id: String,
    #[serde(rename = "type")]
    pub symbol_type: SymbolType,
    pub position: Position,
    #[serde(default)]
    pub properties: Properties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub timestamp: Timestamp,
    pub bounds: HashMap<String, f64>,
    pub symbols: Vec<Symbol>,
    #[serde(default)]
    pub metadata: Properties,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SymbolState {
    pub position: Position,
    #[serde(default)]
    pub velocity: Option<Position>,
    #[serde(default)]
    pub internal_state: Properties,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicState {
    pub timestamp: Timestamp,
    #[serde(default)]
    pub symbol_states: HashMap<String, SymbolState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleState {
    pub id: usize,
    pub current_state: DynamicState,
    pub energy: f64,
    pub weight: f64,
    #[serde(default)]
    pub history: Option<Vec<DynamicState>>,
    #[serde(default)]
    pub metadata: Properties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub particles: Vec<ParticleState>,
    pub temperature: f64,
    pub iteration: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub sequence: Vec<DynamicState>,
    #[serde(default)]
    pub metadata: Properties,
}
