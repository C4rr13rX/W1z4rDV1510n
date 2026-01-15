use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StreamSource {
    PeopleVideo,
    CrowdTraffic,
    PublicTopics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEnvelope {
    pub source: StreamSource,
    pub timestamp: Timestamp,
    pub payload: StreamPayload,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum StreamPayload {
    Bytes { data: Vec<u8> },
    Json { value: serde_json::Value },
    Text { value: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventKind {
    BehavioralAtom,
    BehavioralToken,
    CrowdToken,
    TrafficToken,
    TopicEventToken,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventToken {
    pub id: String,
    pub kind: EventKind,
    pub onset: Timestamp,
    pub duration_secs: f64,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub source: Option<StreamSource>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LayerKind {
    UltradianMicroArousal,
    UltradianBrac,
    UltradianMeso,
    FlowDensity,
    FlowVelocity,
    FlowDirectionality,
    FlowStopGoWave,
    FlowMotif,
    FlowSeasonalDaily,
    FlowSeasonalWeekly,
    TopicBurst,
    TopicDecay,
    TopicExcitation,
    TopicLeadLag,
    TopicPeriodicity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerState {
    pub kind: LayerKind,
    pub timestamp: Timestamp,
    /// Phase in radians for rhythmic layers.
    pub phase: f64,
    /// Normalized amplitude in [0,1] when applicable.
    pub amplitude: f64,
    /// Cross-layer coherence score in [0,1].
    pub coherence: f64,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBatch {
    pub timestamp: Timestamp,
    #[serde(default)]
    pub tokens: Vec<EventToken>,
    #[serde(default)]
    pub layers: Vec<LayerState>,
    #[serde(default)]
    pub source_confidence: HashMap<StreamSource, f64>,
}
