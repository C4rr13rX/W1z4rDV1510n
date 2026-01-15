use crate::config::ConsistencyChunkingConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub id: String,
    pub motif_id: String,
    pub support: usize,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub templates: HashMap<String, Template>,
}

impl Codebook {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn insert(&mut self, template: Template) {
        self.templates.insert(template.id.clone(), template);
    }
}

#[derive(Debug, Clone)]
pub struct ChunkingSignal {
    pub motif_id: String,
    pub novelty: f64,
    pub uncertainty: f64,
    pub drift: f64,
}

#[derive(Debug, Clone)]
pub enum ChunkingDecision {
    UseTemplate(Template),
    Escalate { reason: String },
}

pub struct ConsistencyChunker {
    config: ConsistencyChunkingConfig,
    support_counts: HashMap<String, usize>,
    codebook: Codebook,
}

impl ConsistencyChunker {
    pub fn new(config: ConsistencyChunkingConfig) -> Self {
        Self {
            config,
            support_counts: HashMap::new(),
            codebook: Codebook::new(),
        }
    }

    pub fn observe(&mut self, signal: &ChunkingSignal) -> ChunkingDecision {
        if !self.config.enabled {
            return ChunkingDecision::Escalate {
                reason: "chunking_disabled".to_string(),
            };
        }
        if signal.novelty > self.config.novelty_threshold {
            return ChunkingDecision::Escalate {
                reason: "novelty_exceeded".to_string(),
            };
        }
        if signal.uncertainty > self.config.uncertainty_threshold {
            return ChunkingDecision::Escalate {
                reason: "uncertainty_exceeded".to_string(),
            };
        }
        if signal.drift > self.config.uncertainty_threshold {
            return ChunkingDecision::Escalate {
                reason: "drift_exceeded".to_string(),
            };
        }

        let count = self
            .support_counts
            .entry(signal.motif_id.clone())
            .and_modify(|c| *c += 1)
            .or_insert(1);

        if *count < self.config.min_support {
            return ChunkingDecision::Escalate {
                reason: "insufficient_support".to_string(),
            };
        }

        let template = self
            .codebook
            .templates
            .get(&signal.motif_id)
            .cloned()
            .unwrap_or_else(|| {
                let template = Template {
                    id: signal.motif_id.clone(),
                    motif_id: signal.motif_id.clone(),
                    support: *count,
                    attributes: HashMap::new(),
                };
                template
            });
        self.codebook.insert(template.clone());
        ChunkingDecision::UseTemplate(template)
    }

    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_escalates_until_support_threshold() {
        let config = ConsistencyChunkingConfig {
            enabled: true,
            novelty_threshold: 0.9,
            uncertainty_threshold: 0.9,
            min_support: 3,
        };
        let mut chunker = ConsistencyChunker::new(config);
        let signal = ChunkingSignal {
            motif_id: "m1".to_string(),
            novelty: 0.1,
            uncertainty: 0.1,
            drift: 0.1,
        };
        assert!(matches!(
            chunker.observe(&signal),
            ChunkingDecision::Escalate { .. }
        ));
        assert!(matches!(
            chunker.observe(&signal),
            ChunkingDecision::Escalate { .. }
        ));
        assert!(matches!(
            chunker.observe(&signal),
            ChunkingDecision::UseTemplate(_)
        ));
    }

    #[test]
    fn chunker_escalates_on_high_novelty() {
        let config = ConsistencyChunkingConfig {
            enabled: true,
            novelty_threshold: 0.2,
            uncertainty_threshold: 0.9,
            min_support: 1,
        };
        let mut chunker = ConsistencyChunker::new(config);
        let signal = ChunkingSignal {
            motif_id: "m2".to_string(),
            novelty: 0.5,
            uncertainty: 0.1,
            drift: 0.1,
        };
        assert!(matches!(
            chunker.observe(&signal),
            ChunkingDecision::Escalate { .. }
        ));
    }
}
