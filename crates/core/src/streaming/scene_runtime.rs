use crate::config::SceneConfig;
use crate::schema::{Position, Timestamp};
use crate::streaming::schema::{EventToken, LayerState, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntityReport {
    pub entity_id: String,
    pub position: Position,
    pub velocity: Position,
    pub speed: f64,
    #[serde(default)]
    pub phenotype_signature: Option<String>,
    pub last_seen: Timestamp,
    pub stability: f64,
    #[serde(default)]
    pub static_age_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenePrediction {
    pub entity_id: String,
    pub predicted_position: Position,
    pub horizon_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAnomaly {
    pub entity_id: String,
    pub observed: Position,
    pub expected: Position,
    pub error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneReport {
    pub timestamp: Timestamp,
    pub chaos_index: f64,
    #[serde(default)]
    pub entities: Vec<SceneEntityReport>,
    #[serde(default)]
    pub predictions: Vec<ScenePrediction>,
    #[serde(default)]
    pub anomalies: Vec<SceneAnomaly>,
    #[serde(default)]
    pub static_entities: Vec<String>,
}

struct SceneEntityState {
    entity_id: String,
    position: Position,
    velocity: Position,
    speed: f64,
    phenotype_signature: Option<String>,
    last_seen: Timestamp,
    history: VecDeque<(Timestamp, Position)>,
    static_age_secs: f64,
}

pub struct SceneRuntime {
    config: SceneConfig,
    entities: HashMap<String, SceneEntityState>,
    last_predictions: HashMap<String, Position>,
}

impl SceneRuntime {
    pub fn new(config: SceneConfig) -> Self {
        Self {
            config,
            entities: HashMap::new(),
            last_predictions: HashMap::new(),
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<SceneReport> {
        if !self.config.enabled {
            return None;
        }
        let observations = collect_observations(batch);
        if observations.is_empty() && self.entities.is_empty() {
            return None;
        }
        let mut anomalies = Vec::new();
        let mut predictions = Vec::new();
        for observation in observations.values() {
            let state = self.entities.entry(observation.entity_id.clone()).or_insert_with(|| {
                SceneEntityState {
                    entity_id: observation.entity_id.clone(),
                    position: observation.position,
                    velocity: Position::default(),
                    speed: 0.0,
                    phenotype_signature: observation.phenotype_signature.clone(),
                    last_seen: observation.timestamp,
                    history: VecDeque::new(),
                    static_age_secs: 0.0,
                }
            });
            let dt = (observation.timestamp.unix - state.last_seen.unix).abs().max(1) as f64;
            let prev_pos = state.position;
            let dx = observation.position.x - prev_pos.x;
            let dy = observation.position.y - prev_pos.y;
            let dz = observation.position.z - prev_pos.z;
            let velocity = Position {
                x: dx / dt,
                y: dy / dt,
                z: dz / dt,
            };
            let speed = (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z).sqrt();
            if speed < self.config.static_speed_threshold {
                state.static_age_secs += dt;
            } else {
                state.static_age_secs = 0.0;
            }
            state.position = observation.position;
            state.velocity = velocity;
            state.speed = speed;
            state.phenotype_signature = observation.phenotype_signature.clone().or_else(|| state.phenotype_signature.clone());
            state.last_seen = observation.timestamp;
            state.history.push_back((observation.timestamp, observation.position));
            while state.history.len() > self.config.history_len.max(1) {
                state.history.pop_front();
            }
            if let Some(expected) = self.last_predictions.get(&observation.entity_id) {
                let err = position_error(expected, &observation.position);
                if err > self.config.anomaly_threshold {
                    anomalies.push(SceneAnomaly {
                        entity_id: observation.entity_id.clone(),
                        observed: observation.position,
                        expected: *expected,
                        error: err,
                    });
                }
            }
            let horizon = self.config.prediction_horizon_secs.max(1.0);
            let predicted = Position {
                x: observation.position.x + velocity.x * horizon,
                y: observation.position.y + velocity.y * horizon,
                z: observation.position.z + velocity.z * horizon,
            };
            self.last_predictions.insert(observation.entity_id.clone(), predicted);
            predictions.push(ScenePrediction {
                entity_id: observation.entity_id.clone(),
                predicted_position: predicted,
                horizon_secs: horizon,
            });
        }
        self.prune(batch.timestamp);
        let mut entities = Vec::new();
        let mut static_entities = Vec::new();
        for state in self.entities.values() {
            let stability = (1.0 / (1.0 + state.speed)).clamp(0.0, 1.0);
            if state.static_age_secs >= self.config.static_min_age_secs {
                static_entities.push(state.entity_id.clone());
            }
            entities.push(SceneEntityReport {
                entity_id: state.entity_id.clone(),
                position: state.position,
                velocity: state.velocity,
                speed: state.speed,
                phenotype_signature: state.phenotype_signature.clone(),
                last_seen: state.last_seen,
                stability,
                static_age_secs: state.static_age_secs,
            });
        }
        let chaos_index = chaos_index(&entities);
        Some(SceneReport {
            timestamp: batch.timestamp,
            chaos_index,
            entities,
            predictions,
            anomalies,
            static_entities,
        })
    }

    fn prune(&mut self, now: Timestamp) {
        let ttl = (self.config.prediction_horizon_secs * 4.0).max(5.0) as i64;
        self.entities.retain(|_, state| {
            now.unix.saturating_sub(state.last_seen.unix) <= ttl
        });
        if self.entities.len() > self.config.max_entities.max(1) {
            let mut entries = self
                .entities
                .iter()
                .map(|(key, state)| (key.clone(), state.last_seen.unix))
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.1);
            let excess = entries.len().saturating_sub(self.config.max_entities.max(1));
            for (entity_id, _) in entries.into_iter().take(excess) {
                self.entities.remove(&entity_id);
                self.last_predictions.remove(&entity_id);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SceneObservation {
    entity_id: String,
    position: Position,
    timestamp: Timestamp,
    phenotype_signature: Option<String>,
}

fn collect_observations(batch: &TokenBatch) -> HashMap<String, SceneObservation> {
    let mut observations = HashMap::new();
    for token in &batch.tokens {
        if let Some(obs) = observation_from_token(token) {
            observations.insert(obs.entity_id.clone(), obs);
        }
    }
    for layer in &batch.layers {
        if let Some(obs) = observation_from_layer(layer, batch.timestamp) {
            observations.entry(obs.entity_id.clone()).or_insert(obs);
        }
    }
    observations
}

fn observation_from_token(token: &EventToken) -> Option<SceneObservation> {
    let entity_id = token
        .attributes
        .get("entity_id")
        .and_then(|val| val.as_str())
        .map(|val| val.to_string())?;
    let (x, y, z) = position_from_attrs(&token.attributes)?;
    let phenotype_signature = token
        .attributes
        .get("phenotype_signature")
        .and_then(|val| val.as_str())
        .map(|val| val.to_string())
        .or_else(|| {
            token
                .attributes
                .get("metadata")
                .and_then(|val| val.as_object())
                .and_then(|meta| meta.get("phenotype_signature"))
                .and_then(|val| val.as_str())
                .map(|val| val.to_string())
        });
    Some(SceneObservation {
        entity_id,
        position: Position { x, y, z },
        timestamp: token.onset,
        phenotype_signature,
    })
}

fn observation_from_layer(layer: &LayerState, timestamp: Timestamp) -> Option<SceneObservation> {
    let entity_id = layer
        .attributes
        .get("entity_id")
        .and_then(|val| val.as_str())
        .map(|val| val.to_string())?;
    let (x, y, z) = position_from_attrs(&layer.attributes)?;
    Some(SceneObservation {
        entity_id,
        position: Position { x, y, z },
        timestamp,
        phenotype_signature: layer
            .attributes
            .get("phenotype_signature")
            .and_then(|val| val.as_str())
            .map(|val| val.to_string()),
    })
}

fn position_from_attrs(attrs: &HashMap<String, Value>) -> Option<(f64, f64, f64)> {
    let x = attrs.get("pos_x").and_then(|val| val.as_f64())?;
    let y = attrs.get("pos_y").and_then(|val| val.as_f64()).unwrap_or(0.0);
    let z = attrs.get("pos_z").and_then(|val| val.as_f64()).unwrap_or(0.0);
    Some((x, y, z))
}

fn position_error(a: &Position, b: &Position) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn chaos_index(entities: &[SceneEntityReport]) -> f64 {
    if entities.is_empty() {
        return 0.0;
    }
    let mut speeds = Vec::new();
    for entity in entities {
        speeds.push(entity.speed);
    }
    let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
    let mut var = 0.0;
    for value in speeds {
        let diff = value - mean;
        var += diff * diff;
    }
    let variance = var / entities.len() as f64;
    (variance / (1.0 + mean)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken};

    #[test]
    fn scene_runtime_tracks_positions() {
        let config = SceneConfig::default();
        let mut runtime = SceneRuntime::new(config);
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 1 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 1 },
                duration_secs: 1.0,
                confidence: 1.0,
                attributes: HashMap::from([
                    ("entity_id".to_string(), Value::String("e1".to_string())),
                    ("pos_x".to_string(), Value::from(1.0)),
                    ("pos_y".to_string(), Value::from(2.0)),
                ]),
                source: None,
            }],
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let report = runtime.update(&batch).expect("report");
        assert_eq!(report.entities.len(), 1);
        assert_eq!(report.entities[0].entity_id, "e1");
    }
}
