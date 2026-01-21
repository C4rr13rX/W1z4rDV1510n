use crate::config::StreamingHypergraphConfig;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource, TokenBatch};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DomainKind {
    People,
    Crowd,
    Topics,
    Text,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum HypergraphNodeKind {
    Token(EventKind),
    Layer(LayerKind),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphNode {
    pub id: String,
    pub kind: HypergraphNodeKind,
    pub domain: DomainKind,
    pub last_seen: Timestamp,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub last_seen: Timestamp,
    pub confidence: f64,
    pub relation: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HypergraphUpdate {
    pub nodes: Vec<HypergraphNode>,
    pub edges: Vec<HypergraphEdge>,
    pub node_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone)]
struct NodeCandidate {
    id: String,
    kind: HypergraphNodeKind,
    domain: DomainKind,
    confidence: f64,
}

pub struct MultiDomainHypergraph {
    config: StreamingHypergraphConfig,
    nodes: HashMap<String, HypergraphNode>,
    edges: HashMap<(String, String), HypergraphEdge>,
}

impl MultiDomainHypergraph {
    pub fn new(config: StreamingHypergraphConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<HypergraphUpdate> {
        if !self.config.enabled {
            return None;
        }
        let now = batch.timestamp;
        let mut candidates = self.collect_candidates(batch);
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if candidates.len() > self.config.max_nodes_per_batch {
            candidates.truncate(self.config.max_nodes_per_batch);
        }

        let mut update = HypergraphUpdate::default();
        for candidate in &candidates {
            let entry = self.nodes.entry(candidate.id.clone()).or_insert_with(|| {
                HypergraphNode {
                    id: candidate.id.clone(),
                    kind: candidate.kind.clone(),
                    domain: candidate.domain,
                    last_seen: now,
                    weight: candidate.confidence,
                }
            });
            entry.weight = entry.weight * self.config.edge_decay + candidate.confidence;
            entry.last_seen = now;
            update.nodes.push(entry.clone());
        }

        let domain_conf = domain_confidence(batch);
        let mut seen_edges = HashSet::new();
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let a = &candidates[i];
                let b = &candidates[j];
                let gate = domain_conf.get(&a.domain).copied().unwrap_or(1.0)
                    * domain_conf.get(&b.domain).copied().unwrap_or(1.0);
                let confidence = (a.confidence * b.confidence * gate).clamp(0.0, 1.0);
                if confidence <= self.config.min_weight {
                    continue;
                }
                let (left, right) = if a.id <= b.id {
                    (a.id.clone(), b.id.clone())
                } else {
                    (b.id.clone(), a.id.clone())
                };
                let key = (left.clone(), right.clone());
                if seen_edges.contains(&key) {
                    continue;
                }
                seen_edges.insert(key.clone());
                let entry = self.edges.entry(key.clone()).or_insert_with(|| HypergraphEdge {
                    source: left.clone(),
                    target: right.clone(),
                    weight: 0.0,
                    last_seen: now,
                    confidence,
                    relation: "co_occurrence".to_string(),
                });
                entry.weight = entry.weight * self.config.edge_decay + confidence;
                entry.last_seen = now;
                entry.confidence = confidence;
                update.edges.push(entry.clone());
            }
        }

        self.prune(now);
        update.node_count = self.nodes.len();
        update.edge_count = self.edges.len();
        Some(update)
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.nodes.len(), self.edges.len())
    }

    pub fn nodes(&self) -> &HashMap<String, HypergraphNode> {
        &self.nodes
    }

    pub fn edges(&self) -> &HashMap<(String, String), HypergraphEdge> {
        &self.edges
    }

    fn collect_candidates(&self, batch: &TokenBatch) -> Vec<NodeCandidate> {
        let mut out = Vec::new();
        for token in &batch.tokens {
            let confidence = token_confidence(token);
            if confidence <= 0.0 {
                continue;
            }
            out.push(NodeCandidate {
                id: token_node_id(token),
                kind: HypergraphNodeKind::Token(token.kind),
                domain: domain_for_event(token.kind),
                confidence,
            });
        }
        for layer in &batch.layers {
            let confidence = layer_confidence(layer);
            if confidence <= 0.0 {
                continue;
            }
            out.push(NodeCandidate {
                id: layer_node_id(layer),
                kind: HypergraphNodeKind::Layer(layer.kind),
                domain: domain_for_layer(layer.kind),
                confidence,
            });
        }
        out
    }

    fn prune(&mut self, now: Timestamp) {
        let ttl = self.config.edge_ttl_secs.max(0.0);
        let min_weight = self.config.min_weight.max(0.0);
        if ttl > 0.0 {
            self.edges.retain(|_, edge| {
                timestamp_diff_secs(edge.last_seen, now) <= ttl && edge.weight >= min_weight
            });
            self.nodes.retain(|_, node| timestamp_diff_secs(node.last_seen, now) <= ttl * 2.0);
        } else {
            self.edges
                .retain(|_, edge| edge.weight >= min_weight);
        }
        if self.edges.len() > self.config.max_edges {
            let mut ordered: Vec<_> = self
                .edges
                .iter()
                .map(|(k, v)| (k.clone(), v.weight))
                .collect();
            ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ordered.truncate(self.config.max_edges);
            let keep: HashSet<(String, String)> = ordered.into_iter().map(|(k, _)| k).collect();
            self.edges.retain(|k, _| keep.contains(k));
        }
    }
}

fn token_node_id(token: &EventToken) -> String {
    if !token.id.trim().is_empty() {
        return format!("token:{}", token.id.trim());
    }
    let mut marker = format!("token|{:?}|{}|{:.4}", token.kind, token.onset.unix, token.duration_secs);
    if let Some(entity) = token.attributes.get("entity_id").and_then(|v| v.as_str()) {
        marker.push('|');
        marker.push_str(entity);
    }
    format!("token:{}", compute_payload_hash(marker.as_bytes()))
}

fn layer_node_id(layer: &LayerState) -> String {
    format!("layer:{:?}", layer.kind)
}

fn token_confidence(token: &EventToken) -> f64 {
    let conf = if token.confidence <= 0.0 { 1.0 } else { token.confidence };
    conf.clamp(0.0, 1.0)
}

fn layer_confidence(layer: &LayerState) -> f64 {
    let conf = layer.amplitude * 0.7 + layer.coherence * 0.3;
    conf.clamp(0.0, 1.0)
}

fn domain_for_event(kind: EventKind) -> DomainKind {
    match kind {
        EventKind::BehavioralAtom | EventKind::BehavioralToken => DomainKind::People,
        EventKind::CrowdToken | EventKind::TrafficToken => DomainKind::Crowd,
        EventKind::TopicEventToken => DomainKind::Topics,
        EventKind::TextAnnotation => DomainKind::Text,
    }
}

fn domain_for_layer(kind: LayerKind) -> DomainKind {
    match kind {
        LayerKind::UltradianMicroArousal
        | LayerKind::UltradianBrac
        | LayerKind::UltradianMeso => DomainKind::People,
        LayerKind::FlowDensity
        | LayerKind::FlowVelocity
        | LayerKind::FlowDirectionality
        | LayerKind::FlowStopGoWave
        | LayerKind::FlowMotif
        | LayerKind::FlowSeasonalDaily
        | LayerKind::FlowSeasonalWeekly => DomainKind::Crowd,
        LayerKind::TopicBurst
        | LayerKind::TopicDecay
        | LayerKind::TopicExcitation
        | LayerKind::TopicLeadLag
        | LayerKind::TopicPeriodicity => DomainKind::Topics,
    }
}

fn domain_confidence(batch: &TokenBatch) -> HashMap<DomainKind, f64> {
    let mut map = HashMap::new();
    for (source, confidence) in &batch.source_confidence {
        let domain = match source {
            StreamSource::PeopleVideo => DomainKind::People,
            StreamSource::CrowdTraffic => DomainKind::Crowd,
            StreamSource::PublicTopics => DomainKind::Topics,
            StreamSource::TextAnnotations => DomainKind::Text,
        };
        map.insert(domain, confidence.clamp(0.0, 1.0));
    }
    map
}

fn timestamp_diff_secs(a: Timestamp, b: Timestamp) -> f64 {
    (a.unix - b.unix).abs() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;

    #[test]
    fn hypergraph_updates_edges_for_tokens_and_layers() {
        let config = StreamingHypergraphConfig {
            enabled: true,
            max_edges: 100,
            edge_decay: 0.9,
            edge_ttl_secs: 600.0,
            min_weight: 0.0,
            max_nodes_per_batch: 16,
        };
        let mut graph = MultiDomainHypergraph::new(config);
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
                amplitude: 0.8,
                coherence: 0.6,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::from([(StreamSource::PeopleVideo, 0.8)]),
        };
        let update = graph.update(&batch).expect("update");
        assert!(update.node_count >= 2);
        assert!(update.edge_count >= 1);
    }
}
