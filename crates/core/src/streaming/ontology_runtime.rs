use crate::config::{ConsistencyChunkingConfig, OntologyConfig};
use crate::consistency::{ChunkingSignal, ConsistencyChunker};
use crate::schema::Timestamp;
use crate::streaming::schema::{EventToken, LayerState, TokenBatch};
use crate::network::compute_payload_hash;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyLabel {
    pub id: String,
    pub summary: String,
    pub support: usize,
    pub window_secs: f64,
    pub level: String,
    #[serde(default)]
    pub provenance: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyReport {
    pub timestamp: Timestamp,
    pub version: String,
    pub labels: Vec<OntologyLabel>,
    pub total_templates: usize,
}

struct OntologySample {
    timestamp: Timestamp,
    signatures: Vec<String>,
}

struct OntologyWindow {
    duration_secs: f64,
    level: String,
    samples: VecDeque<OntologySample>,
    last_counts: HashMap<String, usize>,
}

pub struct OntologyRuntime {
    config: OntologyConfig,
    chunker: ConsistencyChunker,
    windows: Vec<OntologyWindow>,
    version_counter: u64,
}

impl OntologyRuntime {
    pub fn new(config: OntologyConfig, consistency: ConsistencyChunkingConfig) -> Self {
        let windows = build_windows(config.window_minutes);
        Self {
            config,
            chunker: ConsistencyChunker::new(consistency),
            windows,
            version_counter: 0,
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<OntologyReport> {
        if !self.config.enabled {
            return None;
        }
        let signatures = signatures_from_batch(batch);
        if signatures.is_empty() {
            return None;
        }
        let sample = OntologySample {
            timestamp: batch.timestamp,
            signatures,
        };
        let mut labels = Vec::new();
        for window in &mut self.windows {
            window.samples.push_back(OntologySample {
                timestamp: sample.timestamp,
                signatures: sample.signatures.clone(),
            });
            window_trim(window, batch.timestamp);
            let counts = window_counts(&window.samples);
            let total = counts.values().sum::<usize>().max(1);
            for (signature, count) in &counts {
                let support_ratio = *count as f64 / total as f64;
                let novelty = (1.0 - support_ratio).clamp(0.0, 1.0);
                let uncertainty = (1.0 / (*count as f64).sqrt()).clamp(0.0, 1.0);
                let prev = window.last_counts.get(signature).copied().unwrap_or(*count);
                let denom = prev.max(*count).max(1) as f64;
                let drift = (*count as f64 - prev as f64).abs() / denom;
                let signal = ChunkingSignal {
                    motif_id: signature.clone(),
                    novelty,
                    uncertainty,
                    drift: drift.clamp(0.0, 1.0),
                };
                if let crate::consistency::ChunkingDecision::UseTemplate(template) =
                    self.chunker.observe(&signal)
                {
                    let summary = signature_summary(signature);
                    let mut provenance = HashMap::new();
                    provenance.insert("support_ratio".to_string(), Value::from(support_ratio));
                    provenance.insert("window_secs".to_string(), Value::from(window.duration_secs));
                    provenance.insert("template_support".to_string(), Value::from(template.support as u64));
                    labels.push(OntologyLabel {
                        id: template.id.clone(),
                        summary,
                        support: *count,
                        window_secs: window.duration_secs,
                        level: window.level.clone(),
                        provenance,
                    });
                }
            }
            window.last_counts = counts;
        }

        if labels.is_empty() {
            return None;
        }
        self.version_counter += 1;
        let version = format!("{}{}", self.config.version_prefix, self.version_counter);
        Some(OntologyReport {
            timestamp: batch.timestamp,
            version,
            labels,
            total_templates: self.chunker.codebook().templates.len(),
        })
    }
}

fn build_windows(base_minutes: usize) -> Vec<OntologyWindow> {
    let base = base_minutes.max(1) as f64 * 60.0;
    let windows = [
        (base, "minute".to_string()),
        (base * 6.0, "hour".to_string()),
        (base * 24.0, "day".to_string()),
        (base * 24.0 * 7.0, "week".to_string()),
    ];
    windows
        .into_iter()
        .map(|(duration_secs, level)| OntologyWindow {
            duration_secs,
            level,
            samples: VecDeque::new(),
            last_counts: HashMap::new(),
        })
        .collect()
}

fn signatures_from_batch(batch: &TokenBatch) -> Vec<String> {
    let mut signatures = Vec::new();
    for token in &batch.tokens {
        signatures.push(token_signature(token));
    }
    for layer in &batch.layers {
        signatures.push(layer_signature(layer));
    }
    signatures
}

fn token_signature(token: &EventToken) -> String {
    let mut parts = vec![format!("token::{:?}", token.kind)];
    for key in ["entity_id", "atom_type", "topic", "zone_id", "sensor_id"] {
        if let Some(val) = token.attributes.get(key).and_then(|v| v.as_str()) {
            if !val.trim().is_empty() {
                parts.push(format!("{key}={val}"));
            }
        }
    }
    let payload = parts.join("|");
    format!("tok:{}", compute_payload_hash(payload.as_bytes()))
}

fn layer_signature(layer: &LayerState) -> String {
    let payload = format!("layer::{:?}", layer.kind);
    format!("layer:{}", compute_payload_hash(payload.as_bytes()))
}

fn signature_summary(signature: &str) -> String {
    if let Some(stripped) = signature.strip_prefix("tok:") {
        return format!("event_pattern:{stripped}");
    }
    if let Some(stripped) = signature.strip_prefix("layer:") {
        return format!("layer_pattern:{stripped}");
    }
    format!("pattern:{signature}")
}

fn window_trim(window: &mut OntologyWindow, now: Timestamp) {
    let cutoff = now.unix as f64 - window.duration_secs;
    while let Some(front) = window.samples.front() {
        if front.timestamp.unix as f64 >= cutoff {
            break;
        }
        window.samples.pop_front();
    }
}

fn window_counts(samples: &VecDeque<OntologySample>) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for sample in samples {
        for signature in &sample.signatures {
            *counts.entry(signature.clone()).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource};

    #[test]
    fn ontology_runtime_emits_label_after_support() {
        let config = OntologyConfig {
            enabled: true,
            window_minutes: 1,
            version_prefix: "v".to_string(),
        };
        let consistency = ConsistencyChunkingConfig {
            enabled: true,
            novelty_threshold: 0.9,
            uncertainty_threshold: 0.9,
            min_support: 2,
        };
        let mut runtime = OntologyRuntime::new(config, consistency);
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 100 },
                duration_secs: 1.0,
                confidence: 0.9,
                attributes: HashMap::new(),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 100 },
                phase: 0.1,
                amplitude: 0.7,
                coherence: 0.4,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::new(),
        };
        assert!(runtime.update(&batch).is_none());
        let second = runtime
            .update(&TokenBatch {
                timestamp: Timestamp { unix: 110 },
                ..batch
            })
            .expect("report");
        assert!(!second.labels.is_empty());
    }
}
