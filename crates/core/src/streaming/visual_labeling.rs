use crate::blockchain::WorkKind;
use crate::config::VisualLabelConfig;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::schema::{EventKind, EventToken, StreamSource, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualLabelTask {
    pub id: String,
    pub work_id: String,
    pub work_kind: WorkKind,
    pub timestamp: Timestamp,
    pub summary: String,
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(default)]
    pub frame_id: Option<String>,
    #[serde(default)]
    pub image_ref: Option<String>,
    pub bbox: Value,
    #[serde(default)]
    pub label_hint: Option<String>,
    pub priority: f64,
    pub reward_score: f64,
    #[serde(default)]
    pub evidence: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualLabelReport {
    pub timestamp: Timestamp,
    pub pending: Vec<VisualLabelTask>,
    pub total_pending: usize,
}

pub struct VisualLabelQueue {
    config: VisualLabelConfig,
    pending: VecDeque<VisualLabelTask>,
    known_ids: HashSet<String>,
}

impl VisualLabelQueue {
    pub fn new(config: VisualLabelConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            known_ids: HashSet::new(),
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<VisualLabelReport> {
        if !self.config.enabled {
            return None;
        }
        let mut added = false;
        let mut count = 0usize;
        for token in &batch.tokens {
            if count >= self.config.max_per_batch.max(1) {
                break;
            }
            if !matches!(
                token.source,
                Some(StreamSource::PeopleVideo) | Some(StreamSource::TextAnnotations)
            ) {
                continue;
            }
            let bbox = find_bbox(&token.attributes);
            let Some(bbox) = bbox else {
                continue;
            };
            let frame_id = attr_string(&token.attributes, "frame_id")
                .or_else(|| attr_string(&token.attributes, "frame_ref"));
            let image_ref = attr_string(&token.attributes, "image_ref")
                .or_else(|| attr_string(&token.attributes, "frame_path"));
            if self.config.require_frame_ref && frame_id.is_none() && image_ref.is_none() {
                continue;
            }
            let label_hint = label_hint(&token.attributes);
            let priority = priority_from_token(token);
            if priority < self.config.min_priority {
                continue;
            }
            let id = task_id(token, &frame_id, &image_ref, batch.timestamp);
            if !self.known_ids.insert(id.clone()) {
                continue;
            }
            let summary = summary_for(token, &label_hint);
            let mut evidence = HashMap::new();
            evidence.insert("token_kind".to_string(), Value::String(format!("{:?}", token.kind)));
            evidence.insert("confidence".to_string(), Value::from(token.confidence));
            if let Some(source) = token.source {
                evidence.insert("source".to_string(), Value::String(format!("{:?}", source)));
            }
            if let Some(text) = label_hint.as_ref() {
                evidence.insert("label_hint".to_string(), Value::String(text.clone()));
            }
            let task = VisualLabelTask {
                id: id.clone(),
                work_id: work_id_for(&id),
                work_kind: WorkKind::HumanAnnotation,
                timestamp: batch.timestamp,
                summary,
                entity_id: attr_string(&token.attributes, "entity_id"),
                frame_id,
                image_ref,
                bbox,
                label_hint,
                priority,
                reward_score: priority,
                evidence,
            };
            self.pending.push_back(task);
            count += 1;
            added = true;
        }
        while self.pending.len() > self.config.max_pending.max(1) {
            if let Some(task) = self.pending.pop_front() {
                self.known_ids.remove(&task.id);
            }
        }
        if !added {
            return None;
        }
        Some(VisualLabelReport {
            timestamp: batch.timestamp,
            pending: self.pending.iter().cloned().collect(),
            total_pending: self.pending.len(),
        })
    }
}

impl Default for VisualLabelQueue {
    fn default() -> Self {
        Self::new(VisualLabelConfig::default())
    }
}

fn task_id(
    token: &EventToken,
    frame_id: &Option<String>,
    image_ref: &Option<String>,
    timestamp: Timestamp,
) -> String {
    let payload = format!(
        "visual|{:?}|{}|{}|{}",
        token.kind,
        frame_id.clone().unwrap_or_default(),
        image_ref.clone().unwrap_or_default(),
        timestamp.unix
    );
    compute_payload_hash(payload.as_bytes())
}

fn work_id_for(task_id: &str) -> String {
    compute_payload_hash(format!("work|visual|{task_id}").as_bytes())
}

fn find_bbox(attrs: &HashMap<String, Value>) -> Option<Value> {
    for key in ["text_bbox", "bbox", "bounding_box", "box"] {
        if let Some(val) = attrs.get(key) {
            return Some(val.clone());
        }
    }
    None
}

fn label_hint(attrs: &HashMap<String, Value>) -> Option<String> {
    for key in ["text", "label", "caption", "description"] {
        if let Some(text) = attrs.get(key).and_then(|val| val.as_str()) {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn attr_string(attrs: &HashMap<String, Value>, key: &str) -> Option<String> {
    attrs.get(key).and_then(|val| val.as_str()).map(|s| s.to_string())
}

fn priority_from_token(token: &EventToken) -> f64 {
    let conf = token.confidence.clamp(0.0, 1.0);
    match token.kind {
        EventKind::TextAnnotation => (0.5 + 0.5 * conf).clamp(0.1, 1.0),
        _ => conf.max(0.1),
    }
}

fn summary_for(token: &EventToken, hint: &Option<String>) -> String {
    if let Some(text) = hint.as_ref() {
        return format!("Label region for '{text}'");
    }
    format!("Label region for {:?}", token.kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;

    #[test]
    fn visual_queue_emits_task() {
        let mut queue = VisualLabelQueue::default();
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 10 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::TextAnnotation,
                onset: Timestamp { unix: 10 },
                duration_secs: 1.0,
                confidence: 0.8,
                attributes: HashMap::from([
                    ("text".to_string(), Value::String("STOP".to_string())),
                    ("frame_id".to_string(), Value::String("frame-1".to_string())),
                    ("text_bbox".to_string(), Value::Array(vec![Value::from(0.0), Value::from(0.0), Value::from(10.0), Value::from(10.0)])),
                ]),
                source: Some(StreamSource::TextAnnotations),
            }],
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let report = queue.update(&batch).expect("report");
        assert_eq!(report.pending.len(), 1);
        assert_eq!(report.pending[0].frame_id.as_deref(), Some("frame-1"));
    }
}
