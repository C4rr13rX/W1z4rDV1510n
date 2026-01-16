use crate::branching::{BranchNode, BranchingFutures, Retrodiction};
use crate::config::BranchingFuturesConfig;
use crate::schema::Timestamp;
use crate::streaming::causal_stream::CausalReport;
use crate::streaming::physiology_runtime::PhysiologyReport;
use crate::streaming::schema::{EventKind, TokenBatch};
use crate::streaming::temporal::{EventIntensity, TemporalInferenceReport};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingReport {
    pub timestamp: Timestamp,
    pub root_id: u64,
    pub branches: Vec<BranchNode>,
    pub retrodictions: Vec<Retrodiction>,
    pub total_nodes: usize,
}

pub struct StreamingBranchingRuntime {
    config: BranchingFuturesConfig,
    futures: BranchingFutures,
    retrodictions: Vec<Retrodiction>,
}

impl StreamingBranchingRuntime {
    pub fn new(config: BranchingFuturesConfig) -> Self {
        let root_payload = serde_json::json!({"root": true});
        Self {
            config: config.clone(),
            futures: BranchingFutures::new(config, root_payload),
            retrodictions: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        report: &TemporalInferenceReport,
        causal: Option<&CausalReport>,
        physiology: Option<&PhysiologyReport>,
    ) -> Option<BranchingReport> {
        if !self.config.enabled {
            return None;
        }
        let now = report.timestamp;
        let root = self.futures.root_id();
        let mut branches = Vec::new();
        let mut intensities = report.event_intensities.clone();
        intensities.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total_intensity: f64 = intensities.iter().map(|e| e.intensity).sum();
        for intensity in intensities.iter().take(self.config.max_branches.min(16)) {
            let probability = if total_intensity > 0.0 {
                (intensity.intensity / total_intensity).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let uncertainty = (1.0 - probability).clamp(0.0, 1.0);
            let payload = branch_payload(intensity, causal, physiology);
            if let Some(id) = self
                .futures
                .add_branch(root, now, probability, uncertainty, payload)
            {
                if let Some(node) = self.futures.nodes().get(&id) {
                    branches.push(node.clone());
                }
            }
        }
        let mut new_retrodictions = Vec::new();
        if self.config.retrodiction_enabled {
            let observed = observed_event_kinds(batch);
            for intensity in intensities.iter() {
                if new_retrodictions.len() >= self.config.retrodiction_max {
                    break;
                }
                if observed.contains(&intensity.kind) {
                    continue;
                }
                if intensity.intensity < self.config.retrodiction_min_intensity {
                    continue;
                }
                let confidence = if total_intensity > 0.0 {
                    (intensity.intensity / total_intensity).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let timestamp = Timestamp {
                    unix: (now.unix - 1).max(0),
                };
                let payload = serde_json::json!({
                    "event_kind": format!("{:?}", intensity.kind),
                    "expected_time_secs": intensity.expected_time_secs,
                    "intensity": intensity.intensity,
                });
                let retrodiction = Retrodiction {
                    timestamp,
                    payload,
                    confidence,
                };
                new_retrodictions.push(retrodiction.clone());
                self.retrodictions.push(retrodiction);
            }
            if self.retrodictions.len() > self.config.retrodiction_max {
                let overflow = self.retrodictions.len() - self.config.retrodiction_max;
                self.retrodictions.drain(0..overflow);
            }
        }

        Some(BranchingReport {
            timestamp: now,
            root_id: root,
            branches,
            retrodictions: new_retrodictions,
            total_nodes: self.futures.nodes().len(),
        })
    }
}

fn branch_payload(
    intensity: &EventIntensity,
    causal: Option<&CausalReport>,
    physiology: Option<&PhysiologyReport>,
) -> Value {
    let mut payload = HashMap::new();
    payload.insert(
        "event_kind".to_string(),
        Value::String(format!("{:?}", intensity.kind)),
    );
    payload.insert(
        "expected_time_secs".to_string(),
        Value::from(intensity.expected_time_secs),
    );
    payload.insert("intensity".to_string(), Value::from(intensity.intensity));
    if let Some(causal_report) = causal {
        if !causal_report.interventions.is_empty() {
            let deltas = causal_report
                .interventions
                .iter()
                .take(4)
                .map(|delta| {
                    serde_json::json!({
                        "target": delta.target,
                        "expected_delta": delta.expected_delta,
                        "lag_secs": delta.lag_secs,
                    })
                })
                .collect::<Vec<_>>();
            payload.insert("counterfactuals".to_string(), Value::from(deltas));
        }
    }
    if let Some(phys_report) = physiology {
        payload.insert(
            "physiology_overall_index".to_string(),
            Value::from(phys_report.overall_index),
        );
        payload.insert(
            "physiology_contexts".to_string(),
            Value::from(phys_report.deviations.len() as u64),
        );
    }
    Value::Object(payload.into_iter().collect())
}

fn observed_event_kinds(batch: &TokenBatch) -> HashSet<EventKind> {
    let mut set = HashSet::new();
    for token in &batch.tokens {
        set.insert(token.kind);
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, LayerKind};
    use crate::streaming::temporal::{
        CoherencePrediction, DirichletPosterior, HypergraphStats, LayerPrediction,
    };

    #[test]
    fn branching_runtime_creates_retrodictions() {
        let mut config = BranchingFuturesConfig::default();
        config.enabled = true;
        config.retrodiction_enabled = true;
        config.retrodiction_min_intensity = 0.2;
        let mut runtime = StreamingBranchingRuntime::new(config);
        let report = TemporalInferenceReport {
            timestamp: Timestamp { unix: 100 },
            layer_predictions: vec![LayerPrediction {
                kind: LayerKind::UltradianMicroArousal,
                domain: crate::streaming::DomainKind::People,
                predicted_phase: 0.1,
                predicted_amplitude: 0.7,
                predicted_coherence: 0.4,
                drift_phase: 0.0,
                drift_amplitude: 0.0,
                drift_coherence: 0.0,
            }],
            coherence: vec![CoherencePrediction {
                layer_a: LayerKind::UltradianMicroArousal,
                layer_b: LayerKind::UltradianBrac,
                coherence: 0.2,
                drift: 0.0,
                cross_domain: false,
            }],
            event_intensities: vec![EventIntensity {
                kind: EventKind::BehavioralAtom,
                intensity: 1.0,
                expected_time_secs: 1.0,
                base_rate: 0.2,
            }],
            evidential: vec![DirichletPosterior {
                label: "event_kind".to_string(),
                categories: vec!["Behavioral".to_string()],
                alpha: vec![1.0],
            }],
            next_event: Some(EventKind::BehavioralAtom),
            hypergraph: Some(HypergraphStats { nodes: 2, edges: 1 }),
        };
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let update = runtime.update(&batch, &report, None, None).expect("report");
        assert!(!update.branches.is_empty());
        assert!(!update.retrodictions.is_empty());
    }
}
