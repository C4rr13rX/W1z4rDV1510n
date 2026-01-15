use crate::config::CausalDiscoveryConfig;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub source: String,
    pub target: String,
    pub lag_secs: f64,
    pub weight: f64,
}

#[derive(Debug, Default)]
pub struct CausalGraph {
    edges: HashMap<(String, String), CausalEdge>,
}

impl CausalGraph {
    pub fn update_edge(&mut self, edge: CausalEdge) {
        let key = (edge.source.clone(), edge.target.clone());
        self.edges.insert(key, edge);
    }

    pub fn edges(&self) -> Vec<CausalEdge> {
        self.edges.values().cloned().collect()
    }

    pub fn intervene(&self, intervention: Intervention) -> Vec<CausalDelta> {
        let mut deltas = Vec::new();
        for edge in self.edges.values() {
            if edge.source == intervention.node {
                deltas.push(CausalDelta {
                    target: edge.target.clone(),
                    expected_delta: edge.weight * intervention.delta,
                    lag_secs: edge.lag_secs,
                });
            }
        }
        deltas
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intervention {
    pub node: String,
    pub delta: f64,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDelta {
    pub target: String,
    pub expected_delta: f64,
    pub lag_secs: f64,
}

pub struct CausalRuntime {
    config: CausalDiscoveryConfig,
    graph: CausalGraph,
}

impl CausalRuntime {
    pub fn new(config: CausalDiscoveryConfig) -> Self {
        Self {
            config,
            graph: CausalGraph::default(),
        }
    }

    pub fn observe_edge(&mut self, edge: CausalEdge) {
        if !self.config.enabled {
            return;
        }
        if edge.lag_secs <= self.config.max_lag_secs {
            self.graph.update_edge(edge);
        }
    }

    pub fn graph(&self) -> &CausalGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_intervention_returns_delta() {
        let mut graph = CausalGraph::default();
        graph.update_edge(CausalEdge {
            source: "a".to_string(),
            target: "b".to_string(),
            lag_secs: 10.0,
            weight: 2.0,
        });
        let deltas = graph.intervene(Intervention {
            node: "a".to_string(),
            delta: 1.5,
            timestamp: Timestamp { unix: 0 },
        });
        assert_eq!(deltas.len(), 1);
        assert!((deltas[0].expected_delta - 3.0).abs() < 1e-6);
    }
}
