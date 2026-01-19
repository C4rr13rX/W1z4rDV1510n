use crate::schema::Timestamp;
use crate::streaming::behavior::BehaviorMotif;
use crate::streaming::schema::{EventToken, LayerState, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralFabricShare {
    pub node_id: String,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub tokens: Vec<EventToken>,
    #[serde(default)]
    pub layers: Vec<LayerState>,
    #[serde(default)]
    pub motifs: Vec<BehaviorMotif>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl NeuralFabricShare {
    pub fn from_batch(node_id: String, batch: TokenBatch, motifs: Vec<BehaviorMotif>) -> Self {
        Self {
            node_id,
            timestamp: batch.timestamp,
            tokens: batch.tokens,
            layers: batch.layers,
            motifs,
            metadata: HashMap::new(),
        }
    }

    pub fn to_token_batch(&self) -> TokenBatch {
        let mut tokens = self.tokens.clone();
        for token in &mut tokens {
            token.source = None;
            token
                .attributes
                .insert("origin_node_id".to_string(), Value::String(self.node_id.clone()));
            token.attributes.insert(
                "origin_timestamp".to_string(),
                Value::from(self.timestamp.unix),
            );
        }
        let mut layers = self.layers.clone();
        for layer in &mut layers {
            layer
                .attributes
                .insert("origin_node_id".to_string(), Value::String(self.node_id.clone()));
            layer.attributes.insert(
                "origin_timestamp".to_string(),
                Value::from(self.timestamp.unix),
            );
        }
        TokenBatch {
            timestamp: self.timestamp,
            tokens,
            layers,
            source_confidence: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, StreamSource};

    #[test]
    fn share_roundtrips_to_token_batch() {
        let share = NeuralFabricShare {
            node_id: "node-a".to_string(),
            timestamp: Timestamp { unix: 10 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 10 },
                duration_secs: 1.0,
                confidence: 0.9,
                attributes: HashMap::new(),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: Vec::new(),
            motifs: Vec::new(),
            metadata: HashMap::new(),
        };
        let batch = share.to_token_batch();
        assert_eq!(batch.tokens.len(), 1);
        let token = &batch.tokens[0];
        assert!(token.source.is_none());
        assert_eq!(
            token.attributes.get("origin_node_id"),
            Some(&Value::String("node-a".to_string()))
        );
    }
}
