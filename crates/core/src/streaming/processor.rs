use crate::config::StreamingConfig;
use crate::orchestrator::{RunOutcome, run_with_snapshot};
use crate::streaming::flow::{FlowLayerExtractor, FlowSample, FlowConfig};
use crate::streaming::schema::{
    EventKind, EventToken, StreamEnvelope, StreamPayload, StreamSource, TokenBatch,
};
use crate::streaming::symbolize::{SymbolizeConfig, token_batch_to_snapshot};
use crate::streaming::topic::{TopicEventExtractor, TopicSample, TopicConfig};
use crate::streaming::ultradian::{SignalSample, UltradianLayerExtractor};
use crate::config::RunConfig;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct PeopleSignal {
    entity_id: String,
    signal: f64,
}

#[derive(Debug, Deserialize)]
struct FlowSignal {
    density: f64,
    velocity: f64,
    direction_deg: f64,
}

#[derive(Debug, Deserialize)]
struct TopicSignal {
    topic: String,
    intensity: f64,
    #[serde(default)]
    metadata: HashMap<String, Value>,
}

pub struct StreamingProcessor {
    config: StreamingConfig,
    people_extractors: HashMap<String, UltradianLayerExtractor>,
    flow_extractor: FlowLayerExtractor,
    topic_extractor: TopicEventExtractor,
}

impl StreamingProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            people_extractors: HashMap::new(),
            flow_extractor: FlowLayerExtractor::new(FlowConfig::default()),
            topic_extractor: TopicEventExtractor::new(TopicConfig::default()),
        }
    }

    pub fn ingest(&mut self, envelope: StreamEnvelope) -> Option<TokenBatch> {
        if !self.config.enabled {
            return None;
        }
        match envelope.source {
            StreamSource::PeopleVideo => {
                if !self.config.ingest.people_video {
                    return None;
                }
                self.handle_people_video(envelope)
            }
            StreamSource::CrowdTraffic => {
                if !self.config.ingest.crowd_traffic {
                    return None;
                }
                self.handle_crowd_traffic(envelope)
            }
            StreamSource::PublicTopics => {
                if !self.config.ingest.public_topics {
                    return None;
                }
                self.handle_public_topics(envelope)
            }
        }
    }

    fn handle_people_video(&mut self, envelope: StreamEnvelope) -> Option<TokenBatch> {
        let payload = match envelope.payload {
            StreamPayload::Json { value } => value,
            _ => return None,
        };
        let signal: PeopleSignal = serde_json::from_value(payload).ok()?;
        let extractor = self
            .people_extractors
            .entry(signal.entity_id.clone())
            .or_insert_with(UltradianLayerExtractor::new);
        extractor.push_sample(SignalSample {
            timestamp: envelope.timestamp,
            value: signal.signal,
        });
        let mut layers = extractor.extract_layers();
        for layer in &mut layers {
            layer.attributes.insert(
                "entity_id".to_string(),
                Value::String(signal.entity_id.clone()),
            );
        }
        let tokens = vec![EventToken {
            id: String::new(),
            kind: EventKind::BehavioralAtom,
            onset: envelope.timestamp,
            duration_secs: 1.0,
            confidence: 1.0,
            attributes: HashMap::from([
                ("entity_id".to_string(), Value::String(signal.entity_id)),
                ("signal".to_string(), Value::from(signal.signal)),
            ]),
            source: Some(StreamSource::PeopleVideo),
        }];
        Some(TokenBatch {
            timestamp: envelope.timestamp,
            tokens,
            layers,
            source_confidence: HashMap::from([(StreamSource::PeopleVideo, confidence_from_meta(&envelope.metadata))]),
        })
    }

    fn handle_crowd_traffic(&mut self, envelope: StreamEnvelope) -> Option<TokenBatch> {
        let payload = match envelope.payload {
            StreamPayload::Json { value } => value,
            _ => return None,
        };
        let signal: FlowSignal = serde_json::from_value(payload).ok()?;
        self.flow_extractor.push_sample(FlowSample {
            timestamp: envelope.timestamp,
            density: signal.density,
            velocity: signal.velocity,
            direction_deg: signal.direction_deg,
        });
        let extraction = self.flow_extractor.extract();
        Some(TokenBatch {
            timestamp: envelope.timestamp,
            tokens: extraction.tokens,
            layers: extraction.layers,
            source_confidence: HashMap::from([(StreamSource::CrowdTraffic, confidence_from_meta(&envelope.metadata))]),
        })
    }

    fn handle_public_topics(&mut self, envelope: StreamEnvelope) -> Option<TokenBatch> {
        let payload = match envelope.payload {
            StreamPayload::Json { value } => value,
            _ => return None,
        };
        let signal: TopicSignal = serde_json::from_value(payload).ok()?;
        self.topic_extractor.push_sample(TopicSample {
            timestamp: envelope.timestamp,
            topic: signal.topic,
            intensity: signal.intensity,
            metadata: signal.metadata,
        });
        let extraction = self.topic_extractor.extract();
        Some(TokenBatch {
            timestamp: envelope.timestamp,
            tokens: extraction.tokens,
            layers: extraction.layers,
            source_confidence: HashMap::from([(StreamSource::PublicTopics, confidence_from_meta(&envelope.metadata))]),
        })
    }
}

pub struct StreamingInference {
    processor: StreamingProcessor,
    run_config: RunConfig,
    symbolizer: SymbolizeConfig,
}

impl StreamingInference {
    pub fn new(run_config: RunConfig) -> Self {
        let processor = StreamingProcessor::new(run_config.streaming.clone());
        Self {
            processor,
            run_config,
            symbolizer: SymbolizeConfig::default(),
        }
    }

    pub fn handle_envelope(
        &mut self,
        envelope: StreamEnvelope,
    ) -> anyhow::Result<Option<RunOutcome>> {
        let batch = match self.processor.ingest(envelope) {
            Some(batch) => batch,
            None => return Ok(None),
        };
        let snapshot = token_batch_to_snapshot(&batch, &self.symbolizer);
        let outcome = run_with_snapshot(snapshot, self.run_config.clone())?;
        Ok(Some(outcome))
    }
}

fn confidence_from_meta(metadata: &HashMap<String, Value>) -> f64 {
    metadata
        .get("confidence")
        .and_then(|val| val.as_f64())
        .unwrap_or(1.0)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{StreamEnvelope, StreamPayload};

    #[test]
    fn processor_handles_people_signal() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.people_video = true;
        let mut processor = StreamingProcessor::new(config);
        let mut last = None;
        for idx in 0..40 {
            let envelope = StreamEnvelope {
                source: StreamSource::PeopleVideo,
                timestamp: Timestamp { unix: 10 + idx },
                payload: StreamPayload::Json {
                    value: serde_json::json!({
                        "entity_id": "e1",
                        "signal": 0.9
                    }),
                },
                metadata: HashMap::new(),
            };
            last = processor.ingest(envelope);
        }
        let batch = last.expect("batch");
        assert!(!batch.layers.is_empty());
    }

    #[test]
    fn processor_handles_flow_signal() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.crowd_traffic = true;
        let mut processor = StreamingProcessor::new(config);
        let envelope = StreamEnvelope {
            source: StreamSource::CrowdTraffic,
            timestamp: Timestamp { unix: 10 },
            payload: StreamPayload::Json {
                value: serde_json::json!({
                    "density": 12.0,
                    "velocity": 3.0,
                    "direction_deg": 45.0
                }),
            },
            metadata: HashMap::new(),
        };
        let batch = processor.ingest(envelope);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert!(!batch.layers.is_empty());
    }

    #[test]
    fn processor_handles_topic_signal() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.public_topics = true;
        let mut processor = StreamingProcessor::new(config);
        let envelope = StreamEnvelope {
            source: StreamSource::PublicTopics,
            timestamp: Timestamp { unix: 10 },
            payload: StreamPayload::Json {
                value: serde_json::json!({
                    "topic": "event",
                    "intensity": 2.5,
                    "metadata": { "source": "rss" }
                }),
            },
            metadata: HashMap::new(),
        };
        let batch = processor.ingest(envelope);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert!(!batch.layers.is_empty());
    }
}
