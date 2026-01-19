use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::dimensions::DimensionReport;
use crate::streaming::schema::TokenBatch;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelCandidate {
    pub id: String,
    pub timestamp: Timestamp,
    pub summary: String,
    #[serde(default)]
    pub entity_id: Option<String>,
    pub feature: String,
    pub priority: f64,
    pub source: String,
    #[serde(default)]
    pub evidence: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelQueueReport {
    pub timestamp: Timestamp,
    pub pending: Vec<LabelCandidate>,
    pub total_pending: usize,
}

#[derive(Debug, Clone)]
pub struct LabelQueueConfig {
    pub max_pending: usize,
    pub min_priority: f64,
}

impl Default for LabelQueueConfig {
    fn default() -> Self {
        Self {
            max_pending: 512,
            min_priority: 0.1,
        }
    }
}

pub struct LabelQueue {
    config: LabelQueueConfig,
    pending: VecDeque<LabelCandidate>,
    known_features: HashSet<String>,
    known_ids: HashSet<String>,
}

impl LabelQueue {
    pub fn new(config: LabelQueueConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            known_features: HashSet::new(),
            known_ids: HashSet::new(),
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        dimension_report: Option<&DimensionReport>,
    ) -> Option<LabelQueueReport> {
        let mut added = false;
        if let Some(report) = dimension_report {
            for info in &report.emergent {
                let summary = format!(
                    "Emergent dimension {name} (mean {mean:.3}, std {std:.3})",
                    name = info.name,
                    mean = info.mean,
                    std = info.std
                );
                let mut evidence = HashMap::new();
                evidence.insert("count".to_string(), Value::from(info.count as u64));
                let candidate = LabelCandidate {
                    id: candidate_id(&info.name, report.timestamp),
                    timestamp: report.timestamp,
                    summary,
                    entity_id: None,
                    feature: info.name.clone(),
                    priority: 0.8,
                    source: "dimension_tracker".to_string(),
                    evidence,
                };
                added |= self.push(candidate);
            }
        }

        for token in &batch.tokens {
            let entity_id = token
                .attributes
                .get("entity_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            for (key, value) in &token.attributes {
                if should_ignore_key(key) {
                    continue;
                }
                let feature_key = format!("token:{:?}:{key}", token.kind);
                if !self.known_features.insert(feature_key.clone()) {
                    continue;
                }
                let summary = label_summary(&feature_key);
                let priority = numeric_priority(value);
                let mut evidence = HashMap::new();
                evidence.insert("token_kind".to_string(), Value::String(format!("{:?}", token.kind)));
                evidence.insert("raw_value".to_string(), value.clone());
                let candidate = LabelCandidate {
                    id: candidate_id(&feature_key, batch.timestamp),
                    timestamp: batch.timestamp,
                    summary,
                    entity_id: entity_id.clone(),
                    feature: feature_key,
                    priority,
                    source: "token_attribute".to_string(),
                    evidence,
                };
                added |= self.push(candidate);
            }
        }

        for layer in &batch.layers {
            let entity_id = layer
                .attributes
                .get("entity_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let base = format!("layer:{:?}", layer.kind);
            for (key, value) in &layer.attributes {
                if should_ignore_key(key) {
                    continue;
                }
                let feature_key = format!("{base}:{key}");
                if !self.known_features.insert(feature_key.clone()) {
                    continue;
                }
                let summary = label_summary(&feature_key);
                let priority = numeric_priority(value);
                let mut evidence = HashMap::new();
                evidence.insert("layer_kind".to_string(), Value::String(format!("{:?}", layer.kind)));
                evidence.insert("raw_value".to_string(), value.clone());
                let candidate = LabelCandidate {
                    id: candidate_id(&feature_key, batch.timestamp),
                    timestamp: batch.timestamp,
                    summary,
                    entity_id: entity_id.clone(),
                    feature: feature_key,
                    priority,
                    source: "layer_attribute".to_string(),
                    evidence,
                };
                added |= self.push(candidate);
            }
        }

        if !added {
            return None;
        }

        Some(LabelQueueReport {
            timestamp: batch.timestamp,
            pending: self.pending.iter().cloned().collect(),
            total_pending: self.pending.len(),
        })
    }

    fn push(&mut self, candidate: LabelCandidate) -> bool {
        if candidate.priority < self.config.min_priority {
            return false;
        }
        if !self.known_ids.insert(candidate.id.clone()) {
            return false;
        }
        self.pending.push_back(candidate);
        while self.pending.len() > self.config.max_pending.max(1) {
            if let Some(old) = self.pending.pop_front() {
                self.known_ids.remove(&old.id);
            }
        }
        true
    }
}

impl Default for LabelQueue {
    fn default() -> Self {
        Self::new(LabelQueueConfig::default())
    }
}

fn candidate_id(feature: &str, ts: Timestamp) -> String {
    let payload = format!("label|{}|{}", feature, ts.unix);
    compute_payload_hash(payload.as_bytes())
}

fn should_ignore_key(key: &str) -> bool {
    matches!(
        key,
        "entity_id"
            | "motif_id"
            | "support"
            | "description_length"
            | "graph_coherence"
            | "graph_proximity"
            | "tf_amplitudes"
            | "tf_phases"
            | "space_frame"
            | "space_dimensionality"
            | "space_confidence"
            | "space_source"
            | "pos_x"
            | "pos_y"
            | "pos_z"
    )
}

fn label_summary(feature: &str) -> String {
    let label_map = label_dictionary();
    if let Some(summary) = label_map.get(feature) {
        return summary.to_string();
    }
    format!("Feature: {feature}")
}

fn label_dictionary() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("token:BehavioralAtom:motion_energy", "Motion energy"),
        ("token:BehavioralAtom:micro_jitter", "Micro jitter"),
        ("token:BehavioralAtom:posture_shift", "Posture shift"),
        ("token:BehavioralAtom:stability", "Motion stability"),
        ("token:BehavioralAtom:motor_signal", "Motor activation"),
        ("token:BehavioralAtom:motor_signal_norm", "Motor activation (normalized)"),
        ("layer:UltradianMicroArousal:amplitude", "Ultradian micro-arousal amplitude"),
        ("layer:UltradianBrac:amplitude", "Ultradian BRAC amplitude"),
        ("layer:UltradianMeso:amplitude", "Ultradian meso amplitude"),
    ])
}

fn numeric_priority(value: &Value) -> f64 {
    if let Some(val) = value.as_f64() {
        let score = (val.abs() / 5.0).clamp(0.1, 1.0);
        return score;
    }
    0.2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource};

    #[test]
    fn label_queue_collects_new_features() {
        let mut queue = LabelQueue::default();
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 1 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 1 },
                duration_secs: 1.0,
                confidence: 1.0,
                attributes: HashMap::from([("novel_feature".to_string(), Value::from(0.9))]),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 1 },
                phase: 0.2,
                amplitude: 0.4,
                coherence: 0.6,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::new(),
        };
        let report = queue.update(&batch, None).expect("report");
        assert!(!report.pending.is_empty());
    }
}
