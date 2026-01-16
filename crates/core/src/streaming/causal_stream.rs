use crate::causal::{CausalDelta, CausalEdge, CausalGraph, Intervention};
use crate::config::CausalDiscoveryConfig;
use crate::schema::Timestamp;
use crate::streaming::temporal::{EventIntensity, TemporalInferenceReport};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalReport {
    pub timestamp: Timestamp,
    pub edges: Vec<CausalEdge>,
    pub interventions: Vec<CausalDelta>,
    pub node_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone)]
struct NodeState {
    value: f64,
    last_timestamp: Timestamp,
    samples: usize,
}

#[derive(Debug, Clone)]
struct NodeObservation {
    id: String,
    value: f64,
}

pub struct StreamingCausalRuntime {
    config: CausalDiscoveryConfig,
    graph: CausalGraph,
    nodes: HashMap<String, NodeState>,
    last_timestamp: Option<Timestamp>,
}

impl StreamingCausalRuntime {
    pub fn new(config: CausalDiscoveryConfig) -> Self {
        Self {
            config,
            graph: CausalGraph::default(),
            nodes: HashMap::new(),
            last_timestamp: None,
        }
    }

    pub fn update(&mut self, report: &TemporalInferenceReport) -> Option<CausalReport> {
        if !self.config.enabled {
            return None;
        }
        let now = report.timestamp;
        let dt = self
            .last_timestamp
            .map(|prev| timestamp_diff_secs(prev, now))
            .unwrap_or(0.0);
        if dt > self.config.max_lag_secs {
            self.last_timestamp = Some(now);
            return None;
        }

        let observations = collect_observations(report);
        if observations.is_empty() {
            return None;
        }
        let mut deltas = HashMap::new();
        for obs in &observations {
            let state = self.nodes.entry(obs.id.clone()).or_insert(NodeState {
                value: obs.value,
                last_timestamp: now,
                samples: 0,
            });
            let delta = obs.value - state.value;
            deltas.insert(obs.id.clone(), (delta, state.samples));
            state.value = obs.value;
            state.last_timestamp = now;
            state.samples = state.samples.saturating_add(1);
        }

        let mut ids: Vec<String> = observations.iter().map(|obs| obs.id.clone()).collect();
        ids.sort();
        ids.dedup();
        if ids.len() > self.config.max_nodes {
            ids.truncate(self.config.max_nodes);
        }
        let mut edges = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = &ids[i];
                let b = &ids[j];
                let (delta_a, samples_a) = match deltas.get(a) {
                    Some(value) => *value,
                    None => continue,
                };
                let (delta_b, samples_b) = match deltas.get(b) {
                    Some(value) => *value,
                    None => continue,
                };
                if samples_a < 1 || samples_b < 1 {
                    continue;
                }
                let weight = scaled_weight(delta_a, delta_b);
                let confidence = weight.abs();
                if confidence < self.config.min_weight {
                    continue;
                }
                let edge = CausalEdge {
                    source: a.clone(),
                    target: b.clone(),
                    lag_secs: dt,
                    weight,
                };
                self.graph.update_edge(edge.clone());
                edges.push(edge);
            }
        }
        self.prune_edges();

        let mut interventions = Vec::new();
        if self.config.intervention_enabled {
            if let Some(target) = top_event_intensity(&report.event_intensities) {
                let node = format!("event::{:?}", target.kind);
                let intervention = Intervention {
                    node: node.clone(),
                    delta: target.intensity.max(0.0),
                    timestamp: now,
                };
                interventions = self.graph.intervene(intervention);
            }
        }

        self.last_timestamp = Some(now);
        Some(CausalReport {
            timestamp: now,
            edges: self.graph.edges(),
            interventions,
            node_count: self.nodes.len(),
            edge_count: self.graph.len(),
        })
    }

    fn prune_edges(&mut self) {
        let max_edges = self.config.max_edges.max(1);
        if self.graph.len() <= max_edges {
            return;
        }
        let mut ordered = self.graph.edges();
        ordered.sort_by(|a, b| {
            b.weight
                .abs()
                .partial_cmp(&a.weight.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ordered.truncate(max_edges);
        let keep: HashSet<(String, String)> = ordered
            .iter()
            .map(|edge| (edge.source.clone(), edge.target.clone()))
            .collect();
        self.graph.retain_edges(|edge| {
            keep.contains(&(edge.source.clone(), edge.target.clone()))
        });
    }
}

fn collect_observations(report: &TemporalInferenceReport) -> Vec<NodeObservation> {
    let mut obs = Vec::new();
    for intensity in &report.event_intensities {
        obs.push(NodeObservation {
            id: format!("event::{:?}", intensity.kind),
            value: intensity.intensity.max(0.0),
        });
    }
    for layer in &report.layer_predictions {
        obs.push(NodeObservation {
            id: format!("layer::{:?}", layer.kind),
            value: layer.predicted_amplitude.clamp(0.0, 1.0),
        });
    }
    obs
}

fn scaled_weight(delta_a: f64, delta_b: f64) -> f64 {
    let score = delta_a * delta_b;
    if score == 0.0 {
        return 0.0;
    }
    let denom = 1.0 + score.abs();
    (score / denom).clamp(-1.0, 1.0)
}

fn top_event_intensity(intensities: &[EventIntensity]) -> Option<EventIntensity> {
    intensities
        .iter()
        .cloned()
        .max_by(|a, b| {
            a.intensity
                .partial_cmp(&b.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn timestamp_diff_secs(a: Timestamp, b: Timestamp) -> f64 {
    (a.unix - b.unix).abs() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, LayerKind};
    use crate::streaming::temporal::{
        CoherencePrediction, DirichletPosterior, EventIntensity, HypergraphStats, LayerPrediction,
    };

    #[test]
    fn causal_runtime_updates_graph() {
        let mut config = CausalDiscoveryConfig::default();
        config.enabled = true;
        config.intervention_enabled = true;
        let mut runtime = StreamingCausalRuntime::new(config);
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
                intensity: 1.2,
                expected_time_secs: 1.0,
                base_rate: 0.5,
            }],
            evidential: vec![DirichletPosterior {
                label: "event_kind".to_string(),
                categories: vec!["Behavioral".to_string()],
                alpha: vec![1.0],
            }],
            next_event: Some(EventKind::BehavioralAtom),
            hypergraph: Some(HypergraphStats { nodes: 2, edges: 1 }),
        };
        let first = runtime.update(&report);
        assert!(first.is_some());
        let report2 = TemporalInferenceReport {
            timestamp: Timestamp { unix: 101 },
            layer_predictions: vec![LayerPrediction {
                predicted_amplitude: 0.9,
                ..report.layer_predictions[0].clone()
            }],
            event_intensities: vec![EventIntensity {
                intensity: 1.6,
                ..report.event_intensities[0].clone()
            }],
            ..report.clone()
        };
        let second = runtime.update(&report2).expect("report");
        assert!(second.node_count >= 2);
    }
}
