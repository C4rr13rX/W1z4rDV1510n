
use crate::schema::Timestamp;
use crate::streaming::motor::{MotorConfig, MotorFeatureExtractor, PoseFrame};
use crate::streaming::spatial::{insert_spatial_attrs, SpatialEstimate, SpatialEstimator};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpeciesKind {
    Human,
    Canine,
    Feline,
    Avian,
    Aquatic,
    Insect,
    Other(String),
}

impl Serialize for SpeciesKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value = match self {
            SpeciesKind::Human => "HUMAN".to_string(),
            SpeciesKind::Canine => "CANINE".to_string(),
            SpeciesKind::Feline => "FELINE".to_string(),
            SpeciesKind::Avian => "AVIAN".to_string(),
            SpeciesKind::Aquatic => "AQUATIC".to_string(),
            SpeciesKind::Insect => "INSECT".to_string(),
            SpeciesKind::Other(raw) => raw.to_string(),
        };
        serializer.serialize_str(&value)
    }
}

impl<'de> Deserialize<'de> for SpeciesKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let normalized = raw.trim().to_ascii_uppercase();
        let kind = match normalized.as_str() {
            "HUMAN" => SpeciesKind::Human,
            "CANINE" => SpeciesKind::Canine,
            "FELINE" => SpeciesKind::Feline,
            "AVIAN" => SpeciesKind::Avian,
            "AQUATIC" => SpeciesKind::Aquatic,
            "INSECT" => SpeciesKind::Insect,
            _ => SpeciesKind::Other(raw),
        };
        Ok(kind)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodySchema {
    pub species: SpeciesKind,
    pub sensor_map: Vec<SensorChannel>,
    pub actuator_map: Vec<ActionChannel>,
    pub latent_dim: usize,
    pub action_dim: usize,
    pub version: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorChannel {
    pub name: String,
    pub kind: SensorKind,
    pub fields: Vec<String>,
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionChannel {
    pub name: String,
    pub kind: ActionKind,
    pub fields: Vec<String>,
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SensorKind {
    Physiology,
    Motion,
    Voice,
    Gaze,
    Proximity,
    Environment,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActionKind {
    Locomotion,
    Vocalization,
    GazeShift,
    Posture,
    Interaction,
    Autonomic,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSample {
    pub kind: SensorKind,
    pub values: HashMap<String, f64>,
    #[serde(default)]
    pub quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSample {
    pub kind: ActionKind,
    pub values: HashMap<String, f64>,
    #[serde(default)]
    pub quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorInput {
    pub entity_id: String,
    pub timestamp: Timestamp,
    pub species: SpeciesKind,
    #[serde(default)]
    pub sensors: Vec<SensorSample>,
    #[serde(default)]
    pub actions: Vec<ActionSample>,
    #[serde(default)]
    pub pose: Option<PoseFrame>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorState {
    pub entity_id: String,
    pub timestamp: Timestamp,
    pub species: SpeciesKind,
    pub latent: Vec<f64>,
    pub action: Vec<f64>,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    pub confidence: f64,
    pub missing_ratio: f64,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

pub trait BodySchemaAdapter: Send + Sync {
    fn schema(&self) -> &BodySchema;
    fn map_input(
        &mut self,
        input: &BehaviorInput,
        previous: Option<&BehaviorState>,
    ) -> BehaviorState;
}

#[derive(Debug, Clone)]
struct FieldMap {
    latent: HashMap<String, usize>,
    action: HashMap<String, usize>,
}

impl FieldMap {
    fn new(latent_fields: &[&str], action_fields: &[&str]) -> Self {
        let latent = latent_fields
            .iter()
            .enumerate()
            .map(|(idx, field)| (field.to_string(), idx))
            .collect();
        let action = action_fields
            .iter()
            .enumerate()
            .map(|(idx, field)| (field.to_string(), idx))
            .collect();
        Self { latent, action }
    }

    fn latent_len(&self) -> usize {
        self.latent.len()
    }

    fn action_len(&self) -> usize {
        self.action.len()
    }
}

pub struct GenericBodySchemaAdapter {
    schema: BodySchema,
    field_map: FieldMap,
    motor: MotorFeatureExtractor,
}

impl GenericBodySchemaAdapter {
    pub fn new(species: SpeciesKind) -> Self {
        let latent_fields = [
            "motion:signal",
            "motion:energy",
            "motion:jitter",
            "motion:posture",
            "motion:stability",
            "physiology:heart_rate",
            "physiology:resp_rate",
            "physiology:temp",
            "voice:intensity",
            "voice:prosody",
            "gaze:fixation",
            "proximity:nearest",
            "environment:light",
            "environment:noise",
        ];
        let action_fields = [
            "action:locomotion",
            "action:gesture",
            "action:vocalization",
            "action:gaze_shift",
            "action:interaction",
        ];
        let field_map = FieldMap::new(&latent_fields, &action_fields);
        let sensor_map = vec![
            SensorChannel {
                name: "motion".to_string(),
                kind: SensorKind::Motion,
                fields: vec![
                    "signal".to_string(),
                    "energy".to_string(),
                    "jitter".to_string(),
                    "posture".to_string(),
                    "stability".to_string(),
                ],
                indices: vec![0, 1, 2, 3, 4],
            },
            SensorChannel {
                name: "physiology".to_string(),
                kind: SensorKind::Physiology,
                fields: vec![
                    "heart_rate".to_string(),
                    "resp_rate".to_string(),
                    "temp".to_string(),
                ],
                indices: vec![5, 6, 7],
            },
            SensorChannel {
                name: "voice".to_string(),
                kind: SensorKind::Voice,
                fields: vec!["intensity".to_string(), "prosody".to_string()],
                indices: vec![8, 9],
            },
            SensorChannel {
                name: "gaze".to_string(),
                kind: SensorKind::Gaze,
                fields: vec!["fixation".to_string()],
                indices: vec![10],
            },
            SensorChannel {
                name: "proximity".to_string(),
                kind: SensorKind::Proximity,
                fields: vec!["nearest".to_string()],
                indices: vec![11],
            },
            SensorChannel {
                name: "environment".to_string(),
                kind: SensorKind::Environment,
                fields: vec!["light".to_string(), "noise".to_string()],
                indices: vec![12, 13],
            },
        ];
        let actuator_map = vec![
            ActionChannel {
                name: "locomotion".to_string(),
                kind: ActionKind::Locomotion,
                fields: vec!["locomotion".to_string()],
                indices: vec![0],
            },
            ActionChannel {
                name: "gesture".to_string(),
                kind: ActionKind::Posture,
                fields: vec!["gesture".to_string()],
                indices: vec![1],
            },
            ActionChannel {
                name: "vocalization".to_string(),
                kind: ActionKind::Vocalization,
                fields: vec!["vocalization".to_string()],
                indices: vec![2],
            },
            ActionChannel {
                name: "gaze_shift".to_string(),
                kind: ActionKind::GazeShift,
                fields: vec!["gaze_shift".to_string()],
                indices: vec![3],
            },
            ActionChannel {
                name: "interaction".to_string(),
                kind: ActionKind::Interaction,
                fields: vec!["interaction".to_string()],
                indices: vec![4],
            },
        ];
        let schema = BodySchema {
            species,
            sensor_map,
            actuator_map,
            latent_dim: field_map.latent_len(),
            action_dim: field_map.action_len(),
            version: "v1".to_string(),
            description: "Generic motion/physiology/voice/gaze/proximity schema".to_string(),
        };
        Self {
            schema,
            field_map,
            motor: MotorFeatureExtractor::new(MotorConfig::default()),
        }
    }
}

impl BodySchemaAdapter for GenericBodySchemaAdapter {
    fn schema(&self) -> &BodySchema {
        &self.schema
    }

    fn map_input(
        &mut self,
        input: &BehaviorInput,
        previous: Option<&BehaviorState>,
    ) -> BehaviorState {
        let mut latent = vec![0.0; self.schema.latent_dim];
        let mut action = vec![0.0; self.schema.action_dim];
        let mut missing = 0usize;
        let mut total = latent.len() + action.len();

        let mut attributes = HashMap::new();
        let mut confidence_sum = 0.0;
        let mut confidence_count = 0.0;
        let mut position = None;
        apply_metadata_attributes(&mut attributes, &input.metadata);

        if let Some(pose) = input.pose.clone() {
            if let Some(motor) = self.motor.extract(pose) {
                write_latent(&self.field_map, "motion:signal", motor.signal, &mut latent);
                write_latent(&self.field_map, "motion:energy", motor.features.motion_energy, &mut latent);
                write_latent(&self.field_map, "motion:jitter", motor.features.micro_jitter, &mut latent);
                write_latent(&self.field_map, "motion:posture", motor.features.posture_shift, &mut latent);
                write_latent(&self.field_map, "motion:stability", motor.features.stability, &mut latent);
                attributes.insert(
                    "motion_confidence".to_string(),
                    Value::from(motor.features.confidence),
                );
                attributes.insert(
                    "motion_stability".to_string(),
                    Value::from(motor.features.stability),
                );
                confidence_sum += motor.features.confidence * motor.features.stability;
                confidence_count += 1.0;
            } else {
                missing += 5;
            }
        } else {
            missing += 5;
        }

        for sample in &input.sensors {
            let quality = sample.quality.clamp(0.0, 1.0);
            confidence_sum += quality;
            confidence_count += 1.0;
            for (key, value) in &sample.values {
                let composite = format!("{}:{}", sensor_prefix(sample.kind), key);
                let (value, was_missing) = sanitize_value(*value);
                if was_missing {
                    missing += 1;
                }
                write_latent(&self.field_map, &composite, value, &mut latent);
            }
        }

        for sample in &input.actions {
            let quality = sample.quality.clamp(0.0, 1.0);
            confidence_sum += quality;
            confidence_count += 1.0;
            for (key, value) in &sample.values {
                let composite = format!("action:{}", key);
                let (value, was_missing) = sanitize_value(*value);
                if was_missing {
                    missing += 1;
                }
                write_action(&self.field_map, &composite, value, &mut action);
            }
        }

        if input.actions.is_empty() {
            if let Some(prev) = previous {
                if action.is_empty() {
                    total = total.saturating_add(latent.len());
                    missing += latent.len();
                } else {
                    let action_len = action.len();
                    for (idx, val) in latent.iter().enumerate() {
                        let prev_val = prev.latent.get(idx).copied().unwrap_or(0.0);
                        let target = idx.min(action_len - 1);
                        action[target] += val - prev_val;
                    }
                }
            } else {
                total = total.saturating_add(action.len());
                missing += action.len();
            }
        }

        if let Some(pose) = &input.pose {
            if !pose.keypoints.is_empty() {
                let mut x_sum = 0.0;
                let mut y_sum = 0.0;
                let mut count = 0.0;
                for kp in &pose.keypoints {
                    let (x, x_missing) = sanitize_value(kp.x);
                    let (y, y_missing) = sanitize_value(kp.y);
                    if x_missing || y_missing {
                        continue;
                    }
                    x_sum += x;
                    y_sum += y;
                    count += 1.0;
                }
                if count > 0.0 {
                    position = Some([x_sum / count, y_sum / count, 0.0]);
                }
            }
        }

        let missing_ratio = if total > 0 {
            missing as f64 / total as f64
        } else {
            0.0
        };
        let confidence = if confidence_count > 0.0 {
            (confidence_sum / confidence_count).clamp(0.0, 1.0)
        } else {
            0.0
        };
        BehaviorState {
            entity_id: input.entity_id.clone(),
            timestamp: input.timestamp,
            species: input.species.clone(),
            latent,
            action,
            position,
            confidence,
            missing_ratio,
            attributes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorGraph {
    pub timestamp: Timestamp,
    pub nodes: HashMap<String, BehaviorState>,
    pub edges: Vec<BehaviorEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorEdge {
    pub source: String,
    pub target: String,
    pub proximity: Option<f64>,
    pub coupling: CouplingMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    pub coherence: f64,
    pub phase_locking: Option<f64>,
    pub transfer_entropy: Option<f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSignature {
    pub mean_proximity: f64,
    pub mean_coherence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorMotif {
    pub id: String,
    pub entity_id: String,
    pub support: usize,
    pub duration_secs: f64,
    pub description_length: f64,
    pub prototype: Vec<Vec<f64>>,
    pub time_frequency: TimeFrequencySummary,
    pub graph_signature: GraphSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifTransition {
    pub from: String,
    pub to: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFrequencySummary {
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPrediction {
    pub current_motif: Option<String>,
    pub next_motif_distribution: Vec<(String, f64)>,
    pub next_state_entropy: f64,
    pub is_attractor: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorFrame {
    pub timestamp: Timestamp,
    pub states: Vec<BehaviorState>,
    pub graph: BehaviorGraph,
    pub motifs: Vec<BehaviorMotif>,
    pub prediction: Option<BehaviorPrediction>,
    pub backpressure: BackpressureStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BackpressureStatus {
    Ok,
    Hold { reason: String },
    Drop { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConstraint {
    pub index: usize,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftObjective {
    pub name: String,
    pub target_state: Vec<f64>,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BehaviorConstraints {
    pub forbidden_actions: Vec<ActionConstraint>,
    pub soft_objectives: Vec<SoftObjective>,
}

impl BehaviorConstraints {
    fn apply(&self, action: &mut [f64]) {
        for constraint in &self.forbidden_actions {
            if constraint.index >= action.len() {
                continue;
            }
            if constraint.min.is_none() && constraint.max.is_none() {
                action[constraint.index] = 0.0;
                continue;
            }
            if let Some(min) = constraint.min {
                action[constraint.index] = action[constraint.index].max(min);
            }
            if let Some(max) = constraint.max {
                action[constraint.index] = action[constraint.index].min(max);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorSubstrateConfig {
    pub max_history: usize,
    pub missing_ratio_threshold: f64,
    pub min_confidence: f64,
    pub fixed_window: usize,
    pub change_point_window: usize,
    pub change_point_threshold: f64,
    pub dtw_threshold: f64,
    pub motif_min_len: usize,
    pub motif_max_len: usize,
    pub motif_max_count: usize,
    pub transition_max_count: usize,
    pub stft_window: usize,
    pub stft_hop: usize,
    pub stft_bins: usize,
    pub entropy_attractor_threshold: f64,
}

impl Default for BehaviorSubstrateConfig {
    fn default() -> Self {
        Self {
            max_history: 512,
            missing_ratio_threshold: 0.4,
            min_confidence: 0.2,
            fixed_window: 32,
            change_point_window: 8,
            change_point_threshold: 0.6,
            dtw_threshold: 2.0,
            motif_min_len: 8,
            motif_max_len: 128,
            motif_max_count: 256,
            transition_max_count: 1024,
            stft_window: 16,
            stft_hop: 8,
            stft_bins: 6,
            entropy_attractor_threshold: 0.5,
        }
    }
}

pub struct BehaviorSubstrate {
    config: BehaviorSubstrateConfig,
    adapters: HashMap<String, Box<dyn BodySchemaAdapter>>,
    history: HashMap<String, VecDeque<BehaviorState>>,
    motifs: MotifExtractor,
    constraints: BehaviorConstraints,
    spatial: SpatialEstimator,
}

impl BehaviorSubstrate {
    pub fn new(config: BehaviorSubstrateConfig) -> Self {
        Self {
            motifs: MotifExtractor::new(config.clone()),
            config,
            adapters: HashMap::new(),
            history: HashMap::new(),
            constraints: BehaviorConstraints::default(),
            spatial: SpatialEstimator::default(),
        }
    }

    pub fn with_constraints(mut self, constraints: BehaviorConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn register_adapter(
        &mut self,
        entity_id: &str,
        adapter: Box<dyn BodySchemaAdapter>,
    ) {
        self.adapters.insert(entity_id.to_string(), adapter);
    }

    pub fn ingest(&mut self, input: BehaviorInput) -> BehaviorFrame {
        let adapter = self
            .adapters
            .entry(input.entity_id.clone())
            .or_insert_with(|| Box::new(GenericBodySchemaAdapter::new(input.species.clone())));
        let history = self.history.entry(input.entity_id.clone()).or_insert_with(VecDeque::new);
        let prev_state = history.back();
        let mut state = adapter.map_input(&input, prev_state);
        let mut spatial = self.spatial.estimate(input.pose.as_ref(), &input.metadata);
        if spatial.position.is_none() {
            if let Some(pos) = state.position {
                spatial.position = Some(pos);
                spatial.dimensionality = spatial.dimensionality.max(2);
                spatial.confidence = spatial.confidence.max(state.confidence);
                spatial.source = "adapter".to_string();
            }
        }
        if spatial.position.is_some() {
            state.position = spatial.position;
        }
        insert_spatial_attrs(&mut state.attributes, &spatial);
        self.constraints.apply(&mut state.action);

        history.push_back(state.clone());
        while history.len() > self.config.max_history.max(1) {
            history.pop_front();
        }

        let mut backpressure = BackpressureStatus::Ok;
        if state.missing_ratio > self.config.missing_ratio_threshold {
            backpressure = BackpressureStatus::Hold {
                reason: "missing_ratio_exceeded".to_string(),
            };
        } else if state.confidence < self.config.min_confidence {
            backpressure = BackpressureStatus::Hold {
                reason: "low_confidence".to_string(),
            };
        }

        let states = self.latest_states();
        let graph = self.build_graph(&states);

        let mut motifs = Vec::new();
        let mut prediction = None;
        if matches!(backpressure, BackpressureStatus::Ok) {
            motifs = self.motifs.update(&self.history, &graph);
            prediction = self.motifs.predict_next();
        }

        BehaviorFrame {
            timestamp: input.timestamp,
            states,
            graph,
            motifs,
            prediction,
            backpressure,
        }
    }

    pub fn ingest_shared_motifs(&mut self, motifs: &[BehaviorMotif]) {
        self.motifs.merge_shared(motifs);
    }

    pub fn ingest_shared_transitions(&mut self, transitions: &[MotifTransition]) {
        self.motifs.merge_transitions(transitions);
    }

    pub fn transition_snapshot(&self) -> Vec<MotifTransition> {
        self.motifs
            .transition_snapshot(self.config.transition_max_count)
    }

    pub fn history_for(&self, entity_id: &str, max_len: usize) -> Vec<BehaviorState> {
        let Some(history) = self.history.get(entity_id) else {
            return Vec::new();
        };
        if max_len == 0 {
            return Vec::new();
        }
        let skip = history.len().saturating_sub(max_len);
        history.iter().skip(skip).cloned().collect()
    }

    pub fn motif_count(&self) -> usize {
        self.motifs.motif_count()
    }

    pub fn transition_count(&self) -> usize {
        self.motifs.transition_count()
    }

    pub fn estimate_spatial(&self, input: &BehaviorInput) -> SpatialEstimate {
        self.spatial.estimate(input.pose.as_ref(), &input.metadata)
    }

    fn latest_states(&self) -> Vec<BehaviorState> {
        let mut states = Vec::new();
        for history in self.history.values() {
            if let Some(state) = history.back() {
                states.push(state.clone());
            }
        }
        states
    }

    fn build_graph(&self, states: &[BehaviorState]) -> BehaviorGraph {
        let mut nodes = HashMap::new();
        for state in states {
            nodes.insert(state.entity_id.clone(), state.clone());
        }
        let mut edges = Vec::new();
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                let a = &states[i];
                let b = &states[j];
                let proximity = match (a.position, b.position) {
                    (Some(pa), Some(pb)) => {
                        let dx = pa[0] - pb[0];
                        let dy = pa[1] - pb[1];
                        let dz = pa[2] - pb[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        Some(1.0 / (1.0 + dist))
                    }
                    _ => None,
                };
                let coherence = cosine_similarity(&a.latent, &b.latent);
                let phase_locking = match (
                    a.attributes.get("phase").and_then(|v| v.as_f64()),
                    b.attributes.get("phase").and_then(|v| v.as_f64()),
                ) {
                    (Some(pa), Some(pb)) => Some(phase_lock(pa, pb)),
                    _ => None,
                };
                let transfer_entropy = lagged_transfer_entropy(
                    self.history.get(&a.entity_id),
                    self.history.get(&b.entity_id),
                );
                let mut confidence = coherence;
                if let Some(prox) = proximity {
                    confidence = (confidence + prox) * 0.5;
                }
                if confidence <= 0.0 {
                    confidence = proximity.unwrap_or(0.0);
                }
                let coupling = CouplingMetrics {
                    coherence,
                    phase_locking,
                    transfer_entropy,
                    confidence: confidence.clamp(0.0, 1.0),
                };
                edges.push(BehaviorEdge {
                    source: a.entity_id.clone(),
                    target: b.entity_id.clone(),
                    proximity,
                    coupling: coupling.clone(),
                });
                edges.push(BehaviorEdge {
                    source: b.entity_id.clone(),
                    target: a.entity_id.clone(),
                    proximity,
                    coupling,
                });
            }
        }
        BehaviorGraph {
            timestamp: states
                .first()
                .map(|state| state.timestamp)
                .unwrap_or(Timestamp { unix: 0 }),
            nodes,
            edges,
        }
    }
}

struct MotifExtractor {
    config: BehaviorSubstrateConfig,
    motifs: Vec<BehaviorMotif>,
    transitions: HashMap<(String, String), usize>,
    last_motif: HashMap<String, String>,
}

impl MotifExtractor {
    fn new(config: BehaviorSubstrateConfig) -> Self {
        Self {
            config,
            motifs: Vec::new(),
            transitions: HashMap::new(),
            last_motif: HashMap::new(),
        }
    }

    fn update(
        &mut self,
        history: &HashMap<String, VecDeque<BehaviorState>>,
        graph: &BehaviorGraph,
    ) -> Vec<BehaviorMotif> {
        let mut latest_motifs = Vec::new();
        for (entity_id, sequence) in history {
            let segments = segment_sequence(
                sequence,
                self.config.change_point_window,
                self.config.change_point_threshold,
                self.config.fixed_window,
                self.config.motif_min_len,
                self.config.motif_max_len,
            );
            for segment in segments {
                let motif = self.assign_motif(entity_id, &segment, graph);
                if let Some(prev) = self.last_motif.get(entity_id) {
                    let key = (prev.clone(), motif.id.clone());
                    *self.transitions.entry(key).or_insert(0) += 1;
                }
                self.last_motif.insert(entity_id.clone(), motif.id.clone());
                latest_motifs.push(motif);
            }
        }
        self.prune_transitions();
        latest_motifs
    }

    fn predict_next(&self) -> Option<BehaviorPrediction> {
        let current = self.last_motif.values().next()?.clone();
        let mut distribution = Vec::new();
        let mut total = 0.0;
        for ((from, to), count) in &self.transitions {
            if from == &current {
                total += *count as f64;
                distribution.push((to.clone(), *count as f64));
            }
        }
        if total <= 0.0 {
            return Some(BehaviorPrediction {
                current_motif: Some(current),
                next_motif_distribution: Vec::new(),
                next_state_entropy: 0.0,
                is_attractor: true,
            });
        }
        for entry in &mut distribution {
            entry.1 /= total;
        }
        let entropy = shannon_entropy(&distribution);
        Some(BehaviorPrediction {
            current_motif: Some(current),
            next_motif_distribution: distribution,
            next_state_entropy: entropy,
            is_attractor: entropy <= self.config.entropy_attractor_threshold,
        })
    }

    fn assign_motif(
        &mut self,
        entity_id: &str,
        segment: &Segment,
        graph: &BehaviorGraph,
    ) -> BehaviorMotif {
        let signature = graph_signature(entity_id, graph);
        let features = time_frequency_summary(
            &segment.signal,
            self.config.stft_window,
            self.config.stft_hop,
            self.config.stft_bins,
        );
        let mut best_idx = None;
        let mut best_score = f64::MAX;
        for (idx, motif) in self.motifs.iter().enumerate() {
            let dtw = soft_dtw_distance(&segment.sequence, &motif.prototype, 0.5);
            let graph_score = graph_distance(&signature, &motif.graph_signature);
            let score = dtw + graph_score;
            if score < best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        let duration_secs = segment.duration_secs;
        let description_length = mdl_cost(duration_secs, best_score);

        let mut motif = if let Some(idx) = best_idx {
            if best_score <= self.config.dtw_threshold {
                let existing = &mut self.motifs[idx];
                existing.support += 1;
                existing.description_length =
                    0.9 * existing.description_length + 0.1 * description_length;
                existing.time_frequency = blend_time_frequency(&existing.time_frequency, &features);
                existing.graph_signature = blend_graph_signature(&existing.graph_signature, &signature);
                existing.clone()
            } else {
                self.create_motif(entity_id, segment, features, signature, description_length)
            }
        } else {
            self.create_motif(entity_id, segment, features, signature, description_length)
        };

        self.prune_motifs();
        motif.entity_id = entity_id.to_string();
        motif
    }

    fn create_motif(
        &mut self,
        entity_id: &str,
        segment: &Segment,
        features: TimeFrequencySummary,
        signature: GraphSignature,
        description_length: f64,
    ) -> BehaviorMotif {
        let motif = BehaviorMotif {
            id: format!("motif-{}", self.motifs.len() + 1),
            entity_id: entity_id.to_string(),
            support: 1,
            duration_secs: segment.duration_secs,
            description_length,
            prototype: segment.sequence.clone(),
            time_frequency: features,
            graph_signature: signature,
        };
        self.motifs.push(motif.clone());
        motif
    }

    fn prune_motifs(&mut self) {
        let max_count = self.config.motif_max_count.max(1);
        if self.motifs.len() <= max_count {
            return;
        }
        self.motifs.sort_by(|a, b| {
            a.support
                .cmp(&b.support)
                .then_with(|| b.description_length.total_cmp(&a.description_length))
        });
        self.motifs.truncate(max_count);
    }

    fn transition_snapshot(&self, max_count: usize) -> Vec<MotifTransition> {
        let max_count = max_count.max(1);
        let mut entries: Vec<_> = self
            .transitions
            .iter()
            .map(|((from, to), count)| MotifTransition {
                from: from.clone(),
                to: to.clone(),
                count: *count as u64,
            })
            .collect();
        entries.sort_by(|a, b| b.count.cmp(&a.count));
        entries.truncate(max_count);
        entries
    }

    fn merge_transitions(&mut self, incoming: &[MotifTransition]) {
        if incoming.is_empty() {
            return;
        }
        for transition in incoming {
            if transition.from.trim().is_empty() || transition.to.trim().is_empty() {
                continue;
            }
            if transition.count == 0 {
                continue;
            }
            let count = transition.count.min(usize::MAX as u64) as usize;
            let key = (transition.from.clone(), transition.to.clone());
            let entry = self.transitions.entry(key).or_insert(0);
            *entry = entry.saturating_add(count.max(1));
        }
        self.prune_transitions();
    }

    fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    fn prune_transitions(&mut self) {
        let max_count = self.config.transition_max_count.max(1);
        if self.transitions.len() <= max_count {
            return;
        }
        let mut entries: Vec<_> = self
            .transitions
            .iter()
            .map(|((from, to), count)| (from.clone(), to.clone(), *count))
            .collect();
        entries.sort_by(|a, b| b.2.cmp(&a.2));
        entries.truncate(max_count);
        self.transitions.clear();
        for (from, to, count) in entries {
            self.transitions.insert((from, to), count);
        }
    }

    fn merge_shared(&mut self, incoming: &[BehaviorMotif]) {
        for motif in incoming {
            if motif.prototype.is_empty() {
                continue;
            }
            let mut best_idx = None;
            let mut best_score = f64::MAX;
            for (idx, existing) in self.motifs.iter().enumerate() {
                if existing.prototype.is_empty() {
                    continue;
                }
                let dtw = soft_dtw_distance(&motif.prototype, &existing.prototype, 0.5);
                let graph_score = graph_distance(&motif.graph_signature, &existing.graph_signature);
                let score = dtw + graph_score;
                if score < best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }
            if let Some(idx) = best_idx {
                if best_score <= self.config.dtw_threshold {
                    let existing = &mut self.motifs[idx];
                    let support = motif.support.max(1);
                    existing.support = existing.support.saturating_add(support);
                    existing.description_length =
                        0.7 * existing.description_length + 0.3 * motif.description_length;
                    existing.time_frequency =
                        blend_time_frequency(&existing.time_frequency, &motif.time_frequency);
                    existing.graph_signature =
                        blend_graph_signature(&existing.graph_signature, &motif.graph_signature);
                    if let Some(blended) = blend_prototype(&existing.prototype, &motif.prototype, 0.8)
                    {
                        existing.prototype = blended;
                    } else if motif.support > existing.support {
                        existing.prototype = motif.prototype.clone();
                    }
                    continue;
                }
            }
            self.motifs.push(motif.clone());
        }
        self.prune_motifs();
    }

    fn motif_count(&self) -> usize {
        self.motifs.len()
    }
}

#[derive(Debug, Clone)]
struct Segment {
    sequence: Vec<Vec<f64>>,
    signal: Vec<f64>,
    duration_secs: f64,
}

fn segment_sequence(
    sequence: &VecDeque<BehaviorState>,
    window: usize,
    threshold: f64,
    fixed_window: usize,
    min_len: usize,
    max_len: usize,
) -> Vec<Segment> {
    if sequence.len() < min_len {
        return Vec::new();
    }
    let signal: Vec<f64> = sequence
        .iter()
        .map(|state| (vector_norm(&state.latent) + 0.5 * vector_norm(&state.action)).clamp(0.0, 10.0))
        .collect();
    let change_points = detect_change_points(&signal, window, threshold);
    let mut segments = Vec::new();
    if change_points.is_empty() {
        segments.extend(fixed_window_segments(sequence, fixed_window, min_len, max_len));
        return segments;
    }
    let mut start = 0usize;
    for idx in change_points {
        if idx <= start + 1 {
            continue;
        }
        let end = idx.min(sequence.len());
        if end - start >= min_len {
            segments.push(build_segment(sequence, &signal, start, end));
        }
        start = end;
    }
    if sequence.len() - start >= min_len {
        segments.push(build_segment(sequence, &signal, start, sequence.len()));
    }
    segments
}

fn fixed_window_segments(
    sequence: &VecDeque<BehaviorState>,
    window: usize,
    min_len: usize,
    max_len: usize,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    if window == 0 {
        return segments;
    }
    let mut start = 0usize;
    let max_len = max_len.max(min_len);
    let total = sequence.len();
    while start < total {
        let end = (start + window).min(total);
        let len = end - start;
        if len >= min_len && len <= max_len {
            let signal: Vec<f64> = sequence
                .iter()
                .skip(start)
                .take(len)
                .map(|state| (vector_norm(&state.latent) + 0.5 * vector_norm(&state.action)).clamp(0.0, 10.0))
                .collect();
            segments.push(build_segment(sequence, &signal, start, end));
        }
        start = end;
    }
    segments
}

fn build_segment(
    sequence: &VecDeque<BehaviorState>,
    signal: &[f64],
    start: usize,
    end: usize,
) -> Segment {
    let mut seq = Vec::new();
    let mut sig = Vec::new();
    let mut duration_secs = 0.0;
    let mut last_ts = None;
    for state in sequence.iter().skip(start).take(end - start) {
        seq.push(state.latent.clone());
        sig.push(*signal.get(seq.len() - 1).unwrap_or(&0.0));
        if let Some(prev) = last_ts {
            duration_secs += (state.timestamp.unix - prev) as f64;
        }
        last_ts = Some(state.timestamp.unix);
    }
    Segment {
        sequence: seq,
        signal: sig,
        duration_secs,
    }
}

fn detect_change_points(signal: &[f64], window: usize, threshold: f64) -> Vec<usize> {
    let mut points = Vec::new();
    if window == 0 || signal.len() < window * 2 {
        return points;
    }
    for idx in window..(signal.len() - window) {
        let before = &signal[idx - window..idx];
        let after = &signal[idx..idx + window];
        let mean_before = mean(before);
        let mean_after = mean(after);
        if (mean_after - mean_before).abs() >= threshold {
            points.push(idx);
        }
    }
    points
}

fn time_frequency_summary(signal: &[f64], window: usize, hop: usize, bins: usize) -> TimeFrequencySummary {
    let window = window.max(2);
    let hop = hop.max(1);
    let bins = bins.max(1);
    let mut amp_accum = vec![0.0; bins];
    let mut phase_accum = vec![0.0; bins];
    let mut count = 0.0;
    let mut start = 0usize;
    while start + window <= signal.len() {
        let slice = &signal[start..start + window];
        for bin in 0..bins {
            let freq = 2.0 * PI * (bin as f64 + 1.0) / window as f64;
            let mut sum_sin = 0.0;
            let mut sum_cos = 0.0;
            for (idx, val) in slice.iter().enumerate() {
                let phase = freq * idx as f64;
                sum_sin += val * phase.sin();
                sum_cos += val * phase.cos();
            }
            let amplitude = (sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / window as f64;
            let phase = sum_sin.atan2(sum_cos);
            amp_accum[bin] += amplitude;
            phase_accum[bin] += phase;
        }
        count += 1.0;
        start += hop;
    }
    if count > 0.0 {
        for bin in 0..bins {
            amp_accum[bin] /= count;
            phase_accum[bin] /= count;
        }
    }
    TimeFrequencySummary {
        amplitudes: amp_accum,
        phases: phase_accum,
    }
}

fn blend_time_frequency(a: &TimeFrequencySummary, b: &TimeFrequencySummary) -> TimeFrequencySummary {
    let len = a.amplitudes.len().min(b.amplitudes.len()).max(1);
    let mut amps = Vec::with_capacity(len);
    let mut phases = Vec::with_capacity(len);
    for idx in 0..len {
        let amp = 0.8 * a.amplitudes[idx] + 0.2 * b.amplitudes[idx];
        let phase = 0.8 * a.phases[idx] + 0.2 * b.phases[idx];
        amps.push(amp);
        phases.push(phase);
    }
    TimeFrequencySummary {
        amplitudes: amps,
        phases,
    }
}

fn graph_signature(entity_id: &str, graph: &BehaviorGraph) -> GraphSignature {
    let mut prox_sum = 0.0;
    let mut coh_sum = 0.0;
    let mut count = 0.0;
    for edge in &graph.edges {
        if edge.source != entity_id {
            continue;
        }
        if let Some(prox) = edge.proximity {
            prox_sum += prox;
        }
        coh_sum += edge.coupling.coherence;
        count += 1.0;
    }
    if count <= 0.0 {
        return GraphSignature {
            mean_proximity: 0.0,
            mean_coherence: 0.0,
        };
    }
    GraphSignature {
        mean_proximity: prox_sum / count,
        mean_coherence: coh_sum / count,
    }
}

fn blend_graph_signature(a: &GraphSignature, b: &GraphSignature) -> GraphSignature {
    GraphSignature {
        mean_proximity: 0.8 * a.mean_proximity + 0.2 * b.mean_proximity,
        mean_coherence: 0.8 * a.mean_coherence + 0.2 * b.mean_coherence,
    }
}

fn graph_distance(a: &GraphSignature, b: &GraphSignature) -> f64 {
    (a.mean_proximity - b.mean_proximity).abs() + (a.mean_coherence - b.mean_coherence).abs()
}

fn blend_prototype(a: &[Vec<f64>], b: &[Vec<f64>], alpha: f64) -> Option<Vec<Vec<f64>>> {
    if a.len() != b.len() {
        return None;
    }
    let mut blended = Vec::with_capacity(a.len());
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        if row_a.len() != row_b.len() {
            return None;
        }
        let mut row = Vec::with_capacity(row_a.len());
        for (val_a, val_b) in row_a.iter().zip(row_b.iter()) {
            row.push(alpha * val_a + (1.0 - alpha) * val_b);
        }
        blended.push(row);
    }
    Some(blended)
}

fn mdl_cost(duration_secs: f64, score: f64) -> f64 {
    let base = 1.0;
    base + duration_secs * score.max(0.0)
}

fn shannon_entropy(distribution: &[(String, f64)]) -> f64 {
    let mut entropy = 0.0;
    for (_, prob) in distribution {
        let p = prob.clamp(1e-6, 1.0);
        entropy -= p * p.ln();
    }
    entropy
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a <= 1e-6 || norm_b <= 1e-6 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0)
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|val| val * val).sum::<f64>().sqrt()
}

fn phase_lock(a: f64, b: f64) -> f64 {
    let mut diff = a - b;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }
    (diff.cos() + 1.0) * 0.5
}

fn lagged_transfer_entropy(
    a: Option<&VecDeque<BehaviorState>>,
    b: Option<&VecDeque<BehaviorState>>,
) -> Option<f64> {
    let a = a?;
    let b = b?;
    if a.len() < 2 || b.is_empty() {
        return None;
    }
    let a_prev = a.get(a.len() - 2)?;
    let b_curr = b.back()?;
    let a_signal = vector_norm(&a_prev.latent);
    let b_signal = vector_norm(&b_curr.latent);
    Some(((a_signal - b_signal).abs() / (1.0 + b_signal)).clamp(0.0, 1.0))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn sensor_prefix(kind: SensorKind) -> &'static str {
    match kind {
        SensorKind::Physiology => "physiology",
        SensorKind::Motion => "motion",
        SensorKind::Voice => "voice",
        SensorKind::Gaze => "gaze",
        SensorKind::Proximity => "proximity",
        SensorKind::Environment => "environment",
        SensorKind::Other => "other",
    }
}

fn sanitize_value(value: f64) -> (f64, bool) {
    if value.is_finite() {
        (value, false)
    } else {
        (0.0, true)
    }
}

fn write_latent(field_map: &FieldMap, key: &str, value: f64, latent: &mut [f64]) {
    if let Some(idx) = field_map.latent.get(key) {
        if *idx < latent.len() {
            latent[*idx] = value;
        }
    }
}

fn write_action(field_map: &FieldMap, key: &str, value: f64, action: &mut [f64]) {
    if let Some(idx) = field_map.action.get(key) {
        if *idx < action.len() {
            action[*idx] = value;
        }
    }
}

fn apply_metadata_attributes(attrs: &mut HashMap<String, Value>, metadata: &HashMap<String, Value>) {
    for key in [
        "phenotype",
        "size_class",
        "age_bucket",
        "cohort_id",
        "genotype",
        "lineage",
        "anatomy_site",
        "context_scope",
        "species_tag",
    ] {
        if let Some(value) = metadata.get(key) {
            attrs.insert(key.to_string(), value.clone());
        }
    }
}

fn dtw_distance(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = l2_distance(&a[i - 1], &b[j - 1]);
            let best = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + best;
        }
    }
    dp[n][m] / (n + m) as f64
}

fn soft_dtw_distance(a: &[Vec<f64>], b: &[Vec<f64>], gamma: f64) -> f64 {
    if gamma <= 0.0 {
        return dtw_distance(a, b);
    }
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = l2_distance(&a[i - 1], &b[j - 1]);
            let softmin = soft_min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1], gamma);
            dp[i][j] = cost + softmin;
        }
    }
    dp[n][m] / (n + m) as f64
}

fn soft_min(a: f64, b: f64, c: f64, gamma: f64) -> f64 {
    let ea = (-a / gamma).exp();
    let eb = (-b / gamma).exp();
    let ec = (-c / gamma).exp();
    let sum = ea + eb + ec;
    if sum <= 0.0 {
        return a.min(b).min(c);
    }
    -gamma * sum.ln()
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input(entity_id: &str, timestamp: i64, value: f64) -> BehaviorInput {
        BehaviorInput {
            entity_id: entity_id.to_string(),
            timestamp: Timestamp { unix: timestamp },
            species: SpeciesKind::Human,
            sensors: vec![SensorSample {
                kind: SensorKind::Motion,
                values: HashMap::from([
                    ("signal".to_string(), value),
                    ("energy".to_string(), value * 0.5),
                ]),
                quality: 1.0,
            }],
            actions: Vec::new(),
            pose: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn dtw_respects_time_warp_similarity() {
        let a = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let b = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let dist = dtw_distance(&a, &b);
        assert!(dist < 0.6);
    }

    #[test]
    fn change_point_falls_back_to_fixed_windows() {
        let mut sequence = VecDeque::new();
        for idx in 0..64 {
            sequence.push_back(BehaviorState {
                entity_id: "e1".to_string(),
                timestamp: Timestamp { unix: idx },
                species: SpeciesKind::Human,
                latent: vec![0.1, 0.1],
                action: vec![0.0],
                position: None,
                confidence: 1.0,
                missing_ratio: 0.0,
                attributes: HashMap::new(),
            });
        }
        let segments = segment_sequence(&sequence, 8, 10.0, 16, 8, 32);
        assert!(!segments.is_empty());
    }

    #[test]
    fn substrate_holds_on_missing_data() {
        let mut substrate = BehaviorSubstrate::new(BehaviorSubstrateConfig::default());
        let input = BehaviorInput {
            entity_id: "e1".to_string(),
            timestamp: Timestamp { unix: 1 },
            species: SpeciesKind::Human,
            sensors: vec![SensorSample {
                kind: SensorKind::Physiology,
                values: HashMap::from([("heart_rate".to_string(), f64::NAN)]),
                quality: 0.1,
            }],
            actions: Vec::new(),
            pose: None,
            metadata: HashMap::new(),
        };
        let frame = substrate.ingest(input);
        assert!(matches!(frame.backpressure, BackpressureStatus::Hold { .. }));
    }

    #[test]
    fn substrate_merges_shared_motifs() {
        let mut substrate = BehaviorSubstrate::new(BehaviorSubstrateConfig::default());
        let motif = BehaviorMotif {
            id: "motif-remote".to_string(),
            entity_id: "e1".to_string(),
            support: 2,
            duration_secs: 12.0,
            description_length: 3.0,
            prototype: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            time_frequency: TimeFrequencySummary {
                amplitudes: vec![0.2, 0.3],
                phases: vec![0.1, 0.2],
            },
            graph_signature: GraphSignature {
                mean_proximity: 0.2,
                mean_coherence: 0.4,
            },
        };
        substrate.ingest_shared_motifs(&[motif]);
        assert!(substrate.motif_count() >= 1);
    }

    #[test]
    fn substrate_merges_shared_transitions() {
        let mut substrate = BehaviorSubstrate::new(BehaviorSubstrateConfig::default());
        let transitions = vec![MotifTransition {
            from: "m1".to_string(),
            to: "m2".to_string(),
            count: 3,
        }];
        substrate.ingest_shared_transitions(&transitions);
        assert!(substrate.transition_count() >= 1);
    }

    #[test]
    fn substrate_relabels_motifs_for_new_entity() {
        let mut substrate = BehaviorSubstrate::new(BehaviorSubstrateConfig::default());
        for idx in 0..16 {
            let _ = substrate.ingest(sample_input("e1", idx, 0.2));
        }
        let mut last_frame = None;
        for idx in 0..16 {
            last_frame = Some(substrate.ingest(sample_input("e2", 100 + idx, 0.2)));
        }
        let frame = last_frame.expect("frame");
        assert!(frame.motifs.iter().any(|motif| motif.entity_id == "e2"));
    }
}
