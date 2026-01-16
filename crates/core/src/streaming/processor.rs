use crate::config::StreamingConfig;
use crate::orchestrator::{RunOutcome, run_with_snapshot};
use crate::streaming::flow::{FlowLayerExtractor, FlowSample, FlowConfig};
use crate::streaming::align::StreamingAligner;
use crate::streaming::motor::{MotorFeatureExtractor, PoseFrame};
use crate::streaming::schema::{
    EventKind, EventToken, StreamEnvelope, StreamPayload, StreamSource, TokenBatch,
};
use crate::streaming::symbolize::{SymbolizeConfig, token_batch_to_snapshot};
use crate::streaming::topic::{TopicEventExtractor, TopicSample, TopicConfig};
use crate::streaming::spike_runtime::StreamingSpikeRuntime;
use crate::streaming::temporal::TemporalInferenceCore;
use crate::streaming::ultradian::{SignalSample, UltradianLayerExtractor};
use crate::config::RunConfig;
use serde::Deserialize;
use serde_json::{Map, Value};
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
    motor_extractor: MotorFeatureExtractor,
    flow_extractor: FlowLayerExtractor,
    topic_extractor: TopicEventExtractor,
}

impl StreamingProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            people_extractors: HashMap::new(),
            motor_extractor: MotorFeatureExtractor::new(crate::streaming::motor::MotorConfig::default()),
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
        let StreamEnvelope {
            timestamp,
            payload,
            metadata,
            ..
        } = envelope;
        let payload = match payload {
            StreamPayload::Json { value } => value,
            _ => return None,
        };
        if let Ok(signal) = serde_json::from_value::<PeopleSignal>(payload.clone()) {
            return self.handle_people_signal(signal, timestamp, metadata);
        }
        let mut frame: PoseFrame = serde_json::from_value(payload).ok()?;
        if frame.timestamp.is_none() {
            frame.timestamp = Some(timestamp);
        }
        self.handle_people_pose(frame, metadata)
    }

    fn handle_people_signal(
        &mut self,
        signal: PeopleSignal,
        timestamp: crate::schema::Timestamp,
        metadata: HashMap<String, Value>,
    ) -> Option<TokenBatch> {
        let extractor = self
            .people_extractors
            .entry(signal.entity_id.clone())
            .or_insert_with(UltradianLayerExtractor::new);
        extractor.push_sample(SignalSample {
            timestamp,
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
            onset: timestamp,
            duration_secs: 1.0,
            confidence: 1.0,
            attributes: HashMap::from([
                ("entity_id".to_string(), Value::String(signal.entity_id)),
                ("signal".to_string(), Value::from(signal.signal)),
            ]),
            source: Some(StreamSource::PeopleVideo),
        }];
        Some(TokenBatch {
            timestamp,
            tokens,
            layers,
            source_confidence: HashMap::from([(
                StreamSource::PeopleVideo,
                confidence_from_meta(&metadata),
            )]),
        })
    }

    fn handle_people_pose(
        &mut self,
        frame: PoseFrame,
        metadata: HashMap<String, Value>,
    ) -> Option<TokenBatch> {
        let output = self.motor_extractor.extract(frame)?;
        let extractor = self
            .people_extractors
            .entry(output.entity_id.clone())
            .or_insert_with(UltradianLayerExtractor::new);
        extractor.push_sample(SignalSample {
            timestamp: output.timestamp,
            value: output.signal,
        });
        let mut layers = extractor.extract_layers();
        for layer in &mut layers {
            layer.attributes.insert(
                "entity_id".to_string(),
                Value::String(output.entity_id.clone()),
            );
        }
        let mut attributes = HashMap::new();
        attributes.insert(
            "entity_id".to_string(),
            Value::String(output.entity_id.clone()),
        );
        attributes.insert("atom_type".to_string(), Value::String("motor".to_string()));
        attributes.insert("motor_signal".to_string(), Value::from(output.signal));
        attributes.insert("motor_signal_raw".to_string(), Value::from(output.raw_signal));
        attributes.insert(
            "motor_signal_norm".to_string(),
            Value::from(output.normalized_signal),
        );
        attributes.insert("baseline_mean".to_string(), Value::from(output.baseline_mean));
        attributes.insert("baseline_std".to_string(), Value::from(output.baseline_std));
        attributes.insert(
            "baseline_samples".to_string(),
            Value::from(output.baseline_samples as u64),
        );
        attributes.insert(
            "baseline_ready".to_string(),
            Value::from(output.baseline_ready),
        );
        attributes.insert(
            "motion_energy".to_string(),
            Value::from(output.features.motion_energy),
        );
        attributes.insert(
            "motion_variance".to_string(),
            Value::from(output.features.motion_variance),
        );
        attributes.insert(
            "posture_shift".to_string(),
            Value::from(output.features.posture_shift),
        );
        attributes.insert(
            "micro_jitter".to_string(),
            Value::from(output.features.micro_jitter),
        );
        attributes.insert(
            "keypoint_confidence".to_string(),
            Value::from(output.features.confidence),
        );
        attributes.insert(
            "keypoint_count".to_string(),
            Value::from(output.keypoint_count as u64),
        );
        if let Some(area) = output.bbox_area {
            attributes.insert("bbox_area".to_string(), Value::from(area));
        }
        if !output.metadata.is_empty() {
            attributes.insert(
                "metadata".to_string(),
                Value::Object(map_from_metadata(&output.metadata)),
            );
        }
        let tokens = vec![EventToken {
            id: String::new(),
            kind: EventKind::BehavioralAtom,
            onset: output.timestamp,
            duration_secs: 1.0,
            confidence: output.features.confidence,
            attributes,
            source: Some(StreamSource::PeopleVideo),
        }];
        Some(TokenBatch {
            timestamp: output.timestamp,
            tokens,
            layers,
            source_confidence: HashMap::from([(
                StreamSource::PeopleVideo,
                confidence_from_meta(&metadata),
            )]),
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
    aligner: StreamingAligner,
    temporal: TemporalInferenceCore,
    spike_runtime: Option<StreamingSpikeRuntime>,
    run_config: RunConfig,
    symbolizer: SymbolizeConfig,
}

impl StreamingInference {
    pub fn new(run_config: RunConfig) -> Self {
        let processor = StreamingProcessor::new(run_config.streaming.clone());
        let aligner = StreamingAligner::new(&run_config.streaming);
        let temporal = TemporalInferenceCore::new(
            run_config.streaming.temporal.clone(),
            run_config.streaming.hypergraph.clone(),
        );
        let spike_runtime = if run_config.streaming.spike.enabled {
            Some(StreamingSpikeRuntime::new(run_config.streaming.spike.clone()))
        } else {
            None
        };
        Self {
            processor,
            aligner,
            temporal,
            spike_runtime,
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
        let batch = match self.aligner.push(batch) {
            Some(batch) => batch,
            None => return Ok(None),
        };
        let temporal_report = self.temporal.update(&batch);
        let spike_messages = self
            .spike_runtime
            .as_mut()
            .and_then(|runtime| runtime.route_batch(&batch, temporal_report.as_ref()));
        let mut snapshot = token_batch_to_snapshot(&batch, &self.symbolizer);
        if let Some(report) = temporal_report {
            snapshot.metadata.insert(
                "temporal_inference".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(mut messages) = spike_messages {
            if let Some(runtime) = &self.spike_runtime {
                if runtime.embed_in_snapshot() {
                    messages.truncate(runtime.max_frames_per_snapshot());
                    snapshot.metadata.insert(
                        "spike_messages".to_string(),
                        serde_json::to_value(messages)?,
                    );
                }
            }
        }
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

fn map_from_metadata(input: &HashMap<String, Value>) -> Map<String, Value> {
    let mut map = Map::new();
    for (key, value) in input {
        map.insert(key.clone(), value.clone());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{StreamEnvelope, StreamPayload};
    use std::f64::consts::PI;

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
    fn processor_handles_people_pose_frame() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.people_video = true;
        let mut processor = StreamingProcessor::new(config);
        let period_secs = 30.0 * 60.0;
        let omega = 2.0 * PI / period_secs;
        let base = 1_000_000_i64;
        let mut last = None;
        for idx in 0..40 {
            let t = base + idx * 60;
            let x = (omega * t as f64).sin();
            let envelope = StreamEnvelope {
                source: StreamSource::PeopleVideo,
                timestamp: Timestamp { unix: t },
                payload: StreamPayload::Json {
                    value: serde_json::json!({
                        "entity_id": "e1",
                        "keypoints": [
                            { "name": "head", "x": x, "y": 0.0, "confidence": 1.0 },
                            { "name": "hand", "x": x + 0.4, "y": 0.2, "confidence": 0.9 },
                            { "name": "foot", "x": x - 0.3, "y": -0.1, "confidence": 0.8 }
                        ]
                    }),
                },
                metadata: HashMap::new(),
            };
            last = processor.ingest(envelope);
        }
        let batch = last.expect("batch");
        assert!(!batch.tokens.is_empty());
        let token = &batch.tokens[0];
        assert!(token.attributes.contains_key("motion_energy"));
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
