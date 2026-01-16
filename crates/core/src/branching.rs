use crate::config::BranchingFuturesConfig;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchNode {
    pub id: u64,
    pub parent: Option<u64>,
    pub timestamp: Timestamp,
    pub probability: f64,
    pub uncertainty: f64,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub children: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Retrodiction {
    pub timestamp: Timestamp,
    pub payload: serde_json::Value,
    pub confidence: f64,
}

pub struct BranchingFutures {
    config: BranchingFuturesConfig,
    nodes: HashMap<u64, BranchNode>,
    root_id: u64,
    next_id: u64,
}

impl BranchingFutures {
    pub fn new(config: BranchingFuturesConfig, root_payload: serde_json::Value) -> Self {
        let root = BranchNode {
            id: 0,
            parent: None,
            timestamp: Timestamp { unix: 0 },
            probability: 1.0,
            uncertainty: 0.0,
            payload: root_payload,
            children: Vec::new(),
        };
        let mut nodes = HashMap::new();
        nodes.insert(0, root);
        Self {
            config,
            nodes,
            root_id: 0,
            next_id: 1,
        }
    }

    pub fn add_branch(
        &mut self,
        parent: u64,
        timestamp: Timestamp,
        probability: f64,
        uncertainty: f64,
        payload: serde_json::Value,
    ) -> Option<u64> {
        if !self.config.enabled {
            return None;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let node = BranchNode {
            id,
            parent: Some(parent),
            timestamp,
            probability,
            uncertainty,
            payload,
            children: Vec::new(),
        };
        self.nodes.insert(id, node);
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.push(id);
        }
        self.prune_if_needed();
        Some(id)
    }

    pub fn root_id(&self) -> u64 {
        self.root_id
    }

    pub fn nodes(&self) -> &HashMap<u64, BranchNode> {
        &self.nodes
    }

    fn prune_if_needed(&mut self) {
        if self.nodes.len() <= self.config.max_branches {
            return;
        }
        let mut ordered: Vec<_> = self.nodes.values().cloned().collect();
        ordered.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ordered.truncate(self.config.max_branches);
        let keep: std::collections::HashSet<u64> = ordered.iter().map(|n| n.id).collect();
        self.nodes.retain(|id, _| keep.contains(id));
        for node in self.nodes.values_mut() {
            node.children.retain(|child| keep.contains(child));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branching_adds_nodes_when_enabled() {
        let config = BranchingFuturesConfig {
            enabled: true,
            max_branches: 4,
            max_depth: 3,
            retrodiction_enabled: true,
            retrodiction_min_intensity: 0.8,
            retrodiction_max: 4,
        };
        let mut futures = BranchingFutures::new(config, serde_json::json!({"root": true}));
        let root = futures.root_id();
        let id = futures.add_branch(
            root,
            Timestamp { unix: 10 },
            0.5,
            0.2,
            serde_json::json!({"event": "a"}),
        );
        assert!(id.is_some());
    }
}
