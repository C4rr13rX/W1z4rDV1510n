use crate::config::StreamingConfig;
use crate::orchestrator::{RunOutcome, run_with_snapshot};
use crate::streaming::behavior::{
    BehaviorInput, BehaviorMotif, BehaviorState, BehaviorSubstrate, BehaviorSubstrateConfig,
    MotifTransition, SensorKind, SensorSample, SpeciesKind,
};
use crate::streaming::fabric::NeuralFabricShare;
use crate::streaming::flow::{FlowLayerExtractor, FlowSample, FlowConfig};
use crate::streaming::align::StreamingAligner;
use crate::streaming::motor::{MotorFeatureExtractor, PoseFrame};
use crate::streaming::analysis_runtime::StreamingAnalysisRuntime;
use crate::streaming::branching_runtime::StreamingBranchingRuntime;
use crate::compute::QuantumExecutor;
use crate::streaming::causal_stream::StreamingCausalRuntime;
use crate::streaming::cross_modal::{CrossModalReport, CrossModalRuntime};
use crate::streaming::appearance::AppearanceExtractor;
use crate::streaming::ocr_runtime::FrameOcrRuntime;
use crate::streaming::dimensions::DimensionTracker;
use crate::streaming::health_overlay::HealthOverlayRuntime;
use crate::streaming::quality::StreamingQualityRuntime;
use crate::streaming::knowledge::{AssociationVote, KnowledgeDocument, KnowledgeIngestReport, KnowledgeRuntime};
use crate::streaming::survival::SurvivalRuntime;
use crate::streaming::scene_runtime::SceneRuntime;
use crate::streaming::network_fabric::{NetworkPatternRuntime, NetworkPatternSummary};
use crate::streaming::ontology_runtime::OntologyRuntime;
use crate::streaming::physiology_runtime::PhysiologyRuntime;
use crate::streaming::plasticity_runtime::StreamingPlasticityRuntime;
use crate::streaming::neuro_bridge::{NeuroStreamBridge, SubstreamRuntime};
use crate::streaming::motif_playback::{MotifPlaybackQueue, MotifReplay, build_motif_replays};
use crate::streaming::narrative_runtime::NarrativeRuntime;
use crate::streaming::metacognition_runtime::{MetacognitionRuntime, MetacognitionShare};
use crate::streaming::schema::{
    EventKind, EventToken, StreamEnvelope, StreamPayload, StreamSource, TokenBatch,
};
use crate::network::compute_payload_hash;
use crate::streaming::labeling::LabelQueue;
use crate::streaming::visual_labeling::VisualLabelQueue;
use crate::streaming::symbolize::{SymbolizeConfig, token_batch_to_snapshot};
use crate::streaming::topic::{TopicEventExtractor, TopicSample, TopicConfig};
use crate::streaming::spike_runtime::StreamingSpikeRuntime;
use crate::streaming::subnet_registry::{SubnetworkRegistry, SubnetworkReport};
use crate::streaming::temporal::TemporalInferenceCore;
use crate::streaming::ultradian::{SignalSample, UltradianLayerExtractor};
use crate::streaming::spatial::insert_spatial_attrs;
use crate::streaming::tracking::PoseTracker;
use crate::neuro::{NeuroRuntime, NeuroSnapshot};
use crate::schema::EnvironmentSnapshot;
use crate::config::{MetacognitionConfig, RunConfig};
use crate::quantum_calibration::QuantumCalibrationState;
use crate::quantum_executor::QuantumHttpExecutor;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

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

#[derive(Debug, Deserialize)]
struct TextAnnotationPayload {
    text: String,
    #[serde(default)]
    entity_id: Option<String>,
    #[serde(default)]
    frame_id: Option<String>,
    #[serde(default)]
    annotation_id: Option<String>,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    attributes: HashMap<String, Value>,
    #[serde(default)]
    confidence: Option<f64>,
}

pub struct StreamingProcessor {
    config: StreamingConfig,
    people_extractors: HashMap<String, UltradianLayerExtractor>,
    motor_extractor: MotorFeatureExtractor,
    pose_tracker: PoseTracker,
    ocr_runtime: Option<FrameOcrRuntime>,
    appearance: AppearanceExtractor,
    behavior_substrate: BehaviorSubstrate,
    flow_extractor: FlowLayerExtractor,
    topic_extractor: TopicEventExtractor,
    last_behavior_motifs: Vec<BehaviorMotif>,
    last_behavior_frame: Option<crate::streaming::behavior::BehaviorFrame>,
    last_motif_replays: Vec<MotifReplay>,
}

impl StreamingProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        let ocr_runtime = if config.ocr.enabled {
            Some(FrameOcrRuntime::new(config.ocr.clone()))
        } else {
            None
        };
        let appearance = AppearanceExtractor::new(config.appearance.clone());
        Self {
            config,
            people_extractors: HashMap::new(),
            motor_extractor: MotorFeatureExtractor::new(crate::streaming::motor::MotorConfig::default()),
            pose_tracker: PoseTracker::default(),
            ocr_runtime,
            appearance,
            behavior_substrate: BehaviorSubstrate::new(BehaviorSubstrateConfig::default()),
            flow_extractor: FlowLayerExtractor::new(FlowConfig::default()),
            topic_extractor: TopicEventExtractor::new(TopicConfig::default()),
            last_behavior_motifs: Vec::new(),
            last_behavior_frame: None,
            last_motif_replays: Vec::new(),
        }
    }

    pub fn ingest(&mut self, envelope: StreamEnvelope) -> Option<TokenBatch> {
        if !self.config.enabled {
            return None;
        }
        self.last_behavior_motifs.clear();
        self.last_behavior_frame = None;
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
            StreamSource::TextAnnotations => {
                if !self.config.ingest.text_annotations {
                    return None;
                }
                self.handle_text_annotation(envelope)
            }
        }
    }

    pub fn take_last_motifs(&mut self) -> Vec<BehaviorMotif> {
        std::mem::take(&mut self.last_behavior_motifs)
    }

    pub fn take_last_behavior_frame(&mut self) -> Option<crate::streaming::behavior::BehaviorFrame> {
        self.last_behavior_frame.take()
    }

    pub fn take_last_motif_replays(&mut self) -> Vec<MotifReplay> {
        std::mem::take(&mut self.last_motif_replays)
    }

    pub fn ingest_fabric_share(&mut self, share: NeuralFabricShare) -> Option<TokenBatch> {
        if !self.config.enabled {
            return None;
        }
        self.last_behavior_motifs.clear();
        self.last_behavior_frame = None;
        if !share.motifs.is_empty() {
            self.behavior_substrate.ingest_shared_motifs(&share.motifs);
        }
        if !share.motif_transitions.is_empty() {
            self.behavior_substrate
                .ingest_shared_transitions(&share.motif_transitions);
        }
        Some(share.to_token_batch())
    }

    pub fn behavior_transition_snapshot(&self) -> Vec<MotifTransition> {
        if !self.config.layer_flags.behavior_enabled {
            return Vec::new();
        }
        self.behavior_substrate.transition_snapshot()
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
        let quality = sample_quality_from_meta(&metadata);
        let entity_id = signal.entity_id.clone();
        let mut layers = Vec::new();
        if self.config.layer_flags.ultradian_enabled {
            let extractor = self
                .people_extractors
                .entry(signal.entity_id.clone())
                .or_insert_with(UltradianLayerExtractor::new);
            extractor.push_sample(SignalSample {
                timestamp,
                value: signal.signal,
                quality,
            });
            layers = extractor.extract_layers();
            for layer in &mut layers {
                layer.attributes.insert(
                    "entity_id".to_string(),
                    Value::String(signal.entity_id.clone()),
                );
            }
        }
        let behavior_input = BehaviorInput {
            entity_id,
            timestamp,
            species: species_from_meta(&metadata),
            sensors: vec![SensorSample {
                kind: SensorKind::Motion,
                values: HashMap::from([("signal".to_string(), signal.signal)]),
                quality,
            }],
            actions: Vec::new(),
            pose: None,
            metadata: metadata.clone(),
        };
        let spatial = self.behavior_substrate.estimate_spatial(&behavior_input);
        for layer in &mut layers {
            insert_spatial_attrs(&mut layer.attributes, &spatial);
        }
        let mut attributes = HashMap::from([
            ("entity_id".to_string(), Value::String(behavior_input.entity_id.clone())),
            ("signal".to_string(), Value::from(signal.signal)),
            ("signal_quality".to_string(), Value::from(quality)),
        ]);
        let snr_proxy = (signal.signal.abs() * 2.0 * quality).clamp(0.0, 20.0);
        attributes.insert("snr_proxy".to_string(), Value::from(snr_proxy));
        insert_spatial_attrs(&mut attributes, &spatial);
        let mut tokens = vec![EventToken {
            id: String::new(),
            kind: EventKind::BehavioralAtom,
            onset: timestamp,
            duration_secs: 1.0,
            confidence: 1.0,
            attributes,
            source: Some(StreamSource::PeopleVideo),
        }];
        self.append_behavior_tokens(behavior_input, &mut tokens, StreamSource::PeopleVideo);
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
        mut frame: PoseFrame,
        metadata: HashMap<String, Value>,
    ) -> Option<TokenBatch> {
        if let Some(runtime) = &mut self.ocr_runtime {
            runtime.maybe_enrich(&mut frame, &metadata);
        }
        if self.appearance.enabled() {
            if let Some(features) = self.appearance.extract(&frame) {
                frame.metadata.insert(
                    "phenotype_signature".to_string(),
                    Value::String(features.signature.clone()),
                );
                frame.metadata.insert(
                    "phenotype_confidence".to_string(),
                    Value::from(features.confidence),
                );
                let vector = features
                    .vector
                    .iter()
                    .map(|value| Value::from(*value))
                    .collect::<Vec<_>>();
                frame
                    .metadata
                    .insert("phenotype_vector".to_string(), Value::Array(vector));
                if !features.components.is_empty() {
                    let mut components = Map::new();
                    for (key, value) in features.components {
                        components.insert(key, Value::from(value));
                    }
                    frame
                        .metadata
                        .insert("phenotype_components".to_string(), Value::Object(components));
                }
            }
        }
        let behavior_frame = frame.clone();
        let tracking = self.pose_tracker.assign(&frame);
        if frame.entity_id.trim().is_empty() || frame.entity_id.trim().eq_ignore_ascii_case("unknown")
        {
            frame.entity_id = tracking.track_id.clone();
        }
        let frame_bbox = frame.bbox.clone();
        let output = self.motor_extractor.extract(frame)?;
        let meta_quality = sample_quality_from_meta(&metadata);
        let signal_quality = (meta_quality
            * output.features.confidence
            * output.features.stability)
            .clamp(0.0, 1.0);
        let mut layers = Vec::new();
        if self.config.layer_flags.ultradian_enabled {
            let extractor = self
                .people_extractors
                .entry(output.entity_id.clone())
                .or_insert_with(UltradianLayerExtractor::new);
            extractor.push_sample(SignalSample {
                timestamp: output.timestamp,
                value: output.signal,
                quality: signal_quality,
            });
            let mut extracted = extractor.extract_layers();
            for layer in &mut extracted {
                layer.attributes.insert(
                    "entity_id".to_string(),
                    Value::String(output.entity_id.clone()),
                );
            }
            layers = extracted;
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
        attributes.insert(
            "signal_quality".to_string(),
            Value::from(signal_quality),
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
            "camera_motion".to_string(),
            Value::from(output.features.camera_motion),
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
            "stability".to_string(),
            Value::from(output.features.stability),
        );
        attributes.insert(
            "keypoint_confidence".to_string(),
            Value::from(output.features.confidence),
        );
        attributes.insert(
            "keypoint_count".to_string(),
            Value::from(output.keypoint_count as u64),
        );
        attributes.insert(
            "track_id".to_string(),
            Value::String(tracking.track_id.clone()),
        );
        attributes.insert(
            "track_confidence".to_string(),
            Value::from(tracking.confidence),
        );
        attributes.insert(
            "track_age_secs".to_string(),
            Value::from(tracking.age_secs),
        );
        attributes.insert(
            "track_distance".to_string(),
            Value::from(tracking.distance),
        );
        attributes.insert(
            "track_reused".to_string(),
            Value::from(tracking.reused),
        );
        attributes.insert(
            "camera_shift_dx".to_string(),
            Value::from(output.camera_shift_dx),
        );
        attributes.insert(
            "camera_shift_dy".to_string(),
            Value::from(output.camera_shift_dy),
        );
        attributes.insert(
            "camera_shift_confidence".to_string(),
            Value::from(output.camera_shift_confidence),
        );
        attributes.insert(
            "camera_shift_source".to_string(),
            Value::String(output.camera_shift_source.clone()),
        );
        let signal_strength = output.features.motion_energy + 0.5 * output.features.posture_shift;
        let noise_strength = output.features.micro_jitter + output.features.camera_motion;
        let snr_proxy = if noise_strength > 1e-6 {
            signal_strength / noise_strength
        } else {
            signal_strength
        };
        let snr_proxy = (snr_proxy * output.features.confidence).clamp(0.0, 50.0);
        attributes.insert("snr_proxy".to_string(), Value::from(snr_proxy));
        if let Some(area) = output.bbox_area {
            attributes.insert("bbox_area".to_string(), Value::from(area));
        }
        if let Some(bbox) = frame_bbox {
            attributes.insert("bbox".to_string(), bbox_to_value(&bbox));
        }
        if !output.metadata.is_empty() {
            attributes.insert(
                "metadata".to_string(),
                Value::Object(map_from_metadata(&output.metadata)),
            );
        }
        apply_phenotype_attrs(&mut attributes, &output.metadata);
        if let Some(frame_id) = frame_ref_from_metadata(&output.metadata) {
            attributes.insert("frame_id".to_string(), Value::String(frame_id));
        }
        let mut behavior_metadata = metadata.clone();
        for (key, value) in &output.metadata {
            behavior_metadata.insert(key.clone(), value.clone());
        }
        let behavior_input = BehaviorInput {
            entity_id: output.entity_id.clone(),
            timestamp: output.timestamp,
            species: species_from_meta(&metadata),
            sensors: Vec::new(),
            actions: Vec::new(),
            pose: Some(behavior_frame),
            metadata: behavior_metadata,
        };
        let spatial = self.behavior_substrate.estimate_spatial(&behavior_input);
        for layer in &mut layers {
            insert_spatial_attrs(&mut layer.attributes, &spatial);
        }
        insert_spatial_attrs(&mut attributes, &spatial);
        let mut tokens = vec![EventToken {
            id: String::new(),
            kind: EventKind::BehavioralAtom,
            onset: output.timestamp,
            duration_secs: 1.0,
            confidence: output.features.confidence,
            attributes,
            source: Some(StreamSource::PeopleVideo),
        }];
        self.append_behavior_tokens(behavior_input, &mut tokens, StreamSource::PeopleVideo);
        let text_tokens = extract_text_overlay_tokens(
            &output.metadata,
            &output.entity_id,
            output.timestamp,
            confidence_from_meta(&metadata),
        );
        if !text_tokens.is_empty() {
            tokens.extend(text_tokens.clone());
        }
        let mut source_confidence = HashMap::from([(
            StreamSource::PeopleVideo,
            confidence_from_meta(&metadata),
        )]);
        if !text_tokens.is_empty() {
            let avg = text_tokens
                .iter()
                .map(|token| token.confidence)
                .sum::<f64>()
                / text_tokens.len() as f64;
            source_confidence.insert(StreamSource::TextAnnotations, avg.clamp(0.0, 1.0));
        }
        Some(TokenBatch {
            timestamp: output.timestamp,
            tokens,
            layers,
            source_confidence,
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

    fn handle_text_annotation(&mut self, envelope: StreamEnvelope) -> Option<TokenBatch> {
        let payload = envelope.payload;
        let mut annotation = match payload {
            StreamPayload::Json { value } => serde_json::from_value::<TextAnnotationPayload>(value).ok()?,
            StreamPayload::Text { value } => TextAnnotationPayload {
                text: value,
                entity_id: None,
                frame_id: None,
                annotation_id: None,
                labels: Vec::new(),
                attributes: HashMap::new(),
                confidence: None,
            },
            _ => return None,
        };
        if annotation.text.trim().is_empty() {
            return None;
        }
        let mut attributes = HashMap::new();
        let text_hash = compute_payload_hash(annotation.text.as_bytes());
        attributes.insert("text".to_string(), Value::String(annotation.text.clone()));
        attributes.insert("text_hash".to_string(), Value::String(text_hash));
        attributes.insert("text_len".to_string(), Value::from(annotation.text.len() as u64));
        if let Some(entity_id) = annotation.entity_id.take() {
            attributes.insert("entity_id".to_string(), Value::String(entity_id));
        }
        if let Some(frame_id) = annotation.frame_id.take() {
            attributes.insert("frame_id".to_string(), Value::String(frame_id));
        }
        if let Some(annotation_id) = annotation.annotation_id.take() {
            attributes.insert("annotation_id".to_string(), Value::String(annotation_id));
        }
        if !annotation.labels.is_empty() {
            attributes.insert(
                "labels".to_string(),
                Value::Array(annotation.labels.into_iter().map(Value::String).collect()),
            );
        }
        if !annotation.attributes.is_empty() {
            attributes.insert(
                "metadata".to_string(),
                Value::Object(map_from_metadata(&annotation.attributes)),
            );
        }
        let confidence = annotation
            .confidence
            .unwrap_or_else(|| confidence_from_meta(&envelope.metadata))
            .clamp(0.0, 1.0);
        let token = EventToken {
            id: String::new(),
            kind: EventKind::TextAnnotation,
            onset: envelope.timestamp,
            duration_secs: 1.0,
            confidence,
            attributes,
            source: Some(StreamSource::TextAnnotations),
        };
        Some(TokenBatch {
            timestamp: envelope.timestamp,
            tokens: vec![token],
            layers: Vec::new(),
            source_confidence: HashMap::from([(StreamSource::TextAnnotations, confidence)]),
        })
    }

    fn append_behavior_tokens(
        &mut self,
        input: BehaviorInput,
        tokens: &mut Vec<EventToken>,
        source: StreamSource,
    ) {
        if !self.config.layer_flags.behavior_enabled {
            return;
        }
        let entity_id = input.entity_id.clone();
        let frame = self.behavior_substrate.ingest(input);
        self.last_behavior_frame = Some(frame.clone());
        if !frame.motifs.is_empty() {
            let history = self.behavior_substrate.history_for(&entity_id, 256);
            let replays = build_motif_replays(&frame, &history, 96);
            if !replays.is_empty() {
                self.last_motif_replays.extend(replays);
            }
        }
        let motifs = frame.motifs;
        self.last_behavior_motifs
            .extend(motifs.iter().cloned());
        let state = frame.states.iter().find(|state| state.entity_id == entity_id);
        for motif in motifs.into_iter().filter(|motif| motif.entity_id == entity_id) {
            let mut attributes = HashMap::new();
            attributes.insert("entity_id".to_string(), Value::String(motif.entity_id.clone()));
            attributes.insert("motif_id".to_string(), Value::String(motif.id.clone()));
            attributes.insert("support".to_string(), Value::from(motif.support as u64));
            attributes.insert(
                "description_length".to_string(),
                Value::from(motif.description_length),
            );
            attributes.insert(
                "graph_coherence".to_string(),
                Value::from(motif.graph_signature.mean_coherence),
            );
            attributes.insert(
                "graph_proximity".to_string(),
                Value::from(motif.graph_signature.mean_proximity),
            );
            attributes.insert(
                "tf_amplitudes".to_string(),
                Value::Array(motif.time_frequency.amplitudes.iter().copied().map(Value::from).collect()),
            );
            attributes.insert(
                "tf_phases".to_string(),
                Value::Array(motif.time_frequency.phases.iter().copied().map(Value::from).collect()),
            );
            if let Some(state) = state {
                copy_spatial_from_state(&mut attributes, state);
            }
            tokens.push(EventToken {
                id: String::new(),
                kind: EventKind::BehavioralToken,
                onset: frame.timestamp,
                duration_secs: motif.duration_secs,
                confidence: motif_confidence(&motif),
                attributes,
                source: Some(source),
            });
        }
    }
}

pub struct StreamingInference {
    processor: StreamingProcessor,
    aligner: StreamingAligner,
    quality: StreamingQualityRuntime,
    temporal: TemporalInferenceCore,
    causal: StreamingCausalRuntime,
    branching: StreamingBranchingRuntime,
    plasticity: StreamingPlasticityRuntime,
    ontology: OntologyRuntime,
    physiology: PhysiologyRuntime,
    analysis: StreamingAnalysisRuntime,
    dimensions: DimensionTracker,
    labels: LabelQueue,
    visual_labels: VisualLabelQueue,
    motif_playback: MotifPlaybackQueue,
    metacognition: MetacognitionRuntime,
    narrative: NarrativeRuntime,
    health_overlay: HealthOverlayRuntime,
    survival: SurvivalRuntime,
    scene: SceneRuntime,
    knowledge: KnowledgeRuntime,
    cross_modal: CrossModalRuntime,
    network_fabric: NetworkPatternRuntime,
    subnet_registry: SubnetworkRegistry,
    spike_runtime: Option<StreamingSpikeRuntime>,
    neuro_bridge: Option<NeuroStreamBridge>,
    substreams: SubstreamRuntime,
    run_config: RunConfig,
    symbolizer: SymbolizeConfig,
    last_batch: Option<TokenBatch>,
    last_motifs: Vec<BehaviorMotif>,
    last_network_patterns: Vec<NetworkPatternSummary>,
    last_metacognition_share: Option<MetacognitionShare>,
    last_report_metadata: HashMap<String, Value>,
}

impl StreamingInference {
    pub fn new(run_config: RunConfig) -> Self {
        let processor = StreamingProcessor::new(run_config.streaming.clone());
        let aligner = StreamingAligner::new(&run_config.streaming);
        let quality = StreamingQualityRuntime::new(run_config.streaming.quality.clone());
        let temporal = TemporalInferenceCore::new(
            run_config.streaming.temporal.clone(),
            run_config.streaming.hypergraph.clone(),
        );
        let causal = StreamingCausalRuntime::new(run_config.streaming.causal.clone());
        let mut quantum_config = run_config.quantum.clone();
        if run_config.streaming.branching.quantum_enabled {
            if let Some(path) = quantum_config.calibration_path.as_ref() {
                if let Ok(state) = QuantumCalibrationState::load(path) {
                    quantum_config = state.apply(&quantum_config);
                }
            }
        }
        let quantum_executor: Option<Box<dyn QuantumExecutor>> =
            if run_config.streaming.branching.quantum_enabled
                && run_config.quantum.remote_enabled
                && run_config.compute.allow_quantum
                && !run_config.compute.quantum_endpoints.is_empty()
            {
                match QuantumHttpExecutor::new(run_config.compute.quantum_endpoints.clone()) {
                    Ok(executor) => Some(Box::new(executor)),
                    Err(_) => None,
                }
            } else {
                None
            };
        let branching = StreamingBranchingRuntime::new(
            run_config.streaming.branching.clone(),
            quantum_config,
            quantum_executor,
        );
        let plasticity = StreamingPlasticityRuntime::new(run_config.streaming.plasticity.clone());
        let ontology =
            OntologyRuntime::new(run_config.streaming.ontology.clone(), run_config.streaming.consistency.clone());
        let mut physiology_config = run_config.streaming.physiology.clone();
        physiology_config.enabled =
            physiology_config.enabled && run_config.streaming.layer_flags.physiology_enabled;
        let physiology = PhysiologyRuntime::new(physiology_config);
        let analysis = StreamingAnalysisRuntime::new(run_config.streaming.analysis.clone());
        let dimensions = DimensionTracker::default();
        let labels = LabelQueue::default();
        let visual_labels = VisualLabelQueue::new(run_config.streaming.visual_label.clone());
        let motif_playback = MotifPlaybackQueue::default();
        let metacognition = MetacognitionRuntime::new(run_config.streaming.metacognition.clone());
        let narrative = NarrativeRuntime::new(run_config.streaming.narrative.clone());
        let health_overlay = HealthOverlayRuntime::default();
        let survival = SurvivalRuntime::default();
        let scene = SceneRuntime::new(run_config.streaming.scene.clone());
        let knowledge = KnowledgeRuntime::default();
        let cross_modal = CrossModalRuntime::new(run_config.streaming.cross_modal.clone());
        let network_fabric = NetworkPatternRuntime::new(run_config.streaming.network_fabric.clone());
        let mut subnet_registry = SubnetworkRegistry::new(run_config.streaming.subnet_registry.clone());
        subnet_registry.load_state();
        let spike_runtime = if run_config.streaming.spike.enabled {
            Some(StreamingSpikeRuntime::new(run_config.streaming.spike.clone()))
        } else {
            None
        };
        let neuro_bridge = if run_config.neuro.enabled {
            let snapshot = empty_snapshot();
            let runtime = Arc::new(NeuroRuntime::new(&snapshot, run_config.neuro.clone()));
            Some(NeuroStreamBridge::new(runtime, SymbolizeConfig::default()))
        } else {
            None
        };
        let substreams = SubstreamRuntime::default();
        Self {
            processor,
            aligner,
            quality,
            temporal,
            causal,
            branching,
            plasticity,
            ontology,
            physiology,
            analysis,
            dimensions,
            labels,
            visual_labels,
            motif_playback,
            metacognition,
            narrative,
            health_overlay,
            survival,
            scene,
            knowledge,
            cross_modal,
            network_fabric,
            subnet_registry,
            spike_runtime,
            neuro_bridge,
            substreams,
            run_config,
            symbolizer: SymbolizeConfig::default(),
            last_batch: None,
            last_motifs: Vec::new(),
            last_network_patterns: Vec::new(),
            last_metacognition_share: None,
            last_report_metadata: HashMap::new(),
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
        self.process_batch(batch)
    }

    pub fn handle_fabric_share(
        &mut self,
        share: NeuralFabricShare,
    ) -> anyhow::Result<Option<RunOutcome>> {
        let mut share = share;
        if !share.motifs.is_empty() {
            let weight = peer_weight_from_metadata(&share.metadata);
            for motif in &mut share.motifs {
                motif.apply_share_provenance(&share.node_id, share.timestamp, weight);
            }
        }
        if !share.network_patterns.is_empty() {
            let weight = peer_weight_from_metadata(&share.metadata);
            for pattern in &mut share.network_patterns {
                let incoming_weight = if pattern.peer_weight.is_finite() && pattern.peer_weight > 0.0 {
                    pattern.peer_weight
                } else {
                    weight
                };
                pattern.peer_weight = incoming_weight.min(weight);
                if !share.node_id.trim().is_empty()
                    && !pattern.origin_nodes.iter().any(|id| id == &share.node_id)
                {
                    pattern.origin_nodes.push(share.node_id.clone());
                }
            }
            self.network_fabric
                .ingest_shared_patterns(&share.network_patterns);
        }
        if let Some(report) = share.metacognition.as_ref() {
            let weight = peer_weight_from_metadata(&share.metadata);
            self.metacognition.ingest_peer_share(report, weight);
        }
        if let Some(value) = share.metadata.get("subnet_registry") {
            if let Ok(report) = serde_json::from_value::<SubnetworkReport>(value.clone()) {
                self.subnet_registry.ingest_shared(&report);
            }
        }
        if let Some(value) = share.metadata.get("cross_modal_report") {
            if let Ok(report) = serde_json::from_value::<CrossModalReport>(value.clone()) {
                self.cross_modal.ingest_report(&report);
            }
        }
        let batch = match self.processor.ingest_fabric_share(share) {
            Some(batch) => batch,
            None => return Ok(None),
        };
        self.process_batch(batch)
    }

    pub fn ingest_knowledge_document(
        &mut self,
        doc: KnowledgeDocument,
        timestamp: crate::schema::Timestamp,
    ) -> KnowledgeIngestReport {
        self.knowledge.ingest_document(doc, timestamp)
    }

    pub fn submit_knowledge_vote(&mut self, vote: AssociationVote) -> Option<crate::streaming::knowledge::KnowledgeAssociation> {
        self.knowledge.submit_vote(vote)
    }

    pub fn take_last_fabric_share(&mut self, node_id: String) -> Option<NeuralFabricShare> {
        let batch = self.last_batch.take()?;
        let motifs = std::mem::take(&mut self.last_motifs);
        let mut share = NeuralFabricShare::from_batch(node_id, batch, motifs);
        share.motif_transitions = self.processor.behavior_transition_snapshot();
        share.network_patterns = std::mem::take(&mut self.last_network_patterns);
        share.metacognition = self.last_metacognition_share.take();
        if !self.last_report_metadata.is_empty() {
            share.metadata = std::mem::take(&mut self.last_report_metadata);
        }
        if !share.motifs.is_empty() {
            let weight = peer_weight_from_metadata(&share.metadata);
            let node_id = share.node_id.clone();
            for motif in &mut share.motifs {
                motif.apply_share_provenance(&node_id, share.timestamp, weight);
            }
        }
        if !share.network_patterns.is_empty() {
            let weight = peer_weight_from_metadata(&share.metadata);
            for pattern in &mut share.network_patterns {
                let incoming_weight = if pattern.peer_weight.is_finite() && pattern.peer_weight > 0.0 {
                    pattern.peer_weight
                } else {
                    weight
                };
                pattern.peer_weight = incoming_weight.min(weight);
            }
        }
        Some(share)
    }

    pub fn update_metacognition_config(&mut self, config: MetacognitionConfig) {
        self.metacognition.update_config(config.clone());
        self.run_config.streaming.metacognition = config;
    }

    fn process_batch(&mut self, batch: TokenBatch) -> anyhow::Result<Option<RunOutcome>> {
        let mut batch = batch;
        let _ = self.quality.update(&mut batch);
        let mut batch = match self.aligner.push(batch) {
            Some(batch) => batch,
            None => return Ok(None),
        };
        let mut neuro_snapshot: Option<NeuroSnapshot> = None;
        if let Some(bridge) = &mut self.neuro_bridge {
            neuro_snapshot = bridge.observe_batch(&batch);
        }
        if let Some(snapshot) = &neuro_snapshot {
            let feedback = neuro_feedback_confidence(snapshot);
            self.aligner.apply_neuro_feedback(feedback);
            self.quality.apply_neuro_feedback(feedback);
            self.substreams.update_from_neuro(snapshot, batch.timestamp);
        }
        self.substreams.update_from_batch(&batch);
        let substream_output = self.substreams.ingest(&batch);
        if !substream_output.tokens.is_empty() {
            batch.tokens.extend(substream_output.tokens);
        }
        if !substream_output.layers.is_empty() {
            batch.layers.extend(substream_output.layers);
        }
        let cross_modal_report = self.cross_modal.update(&batch);
        self.last_batch = Some(batch.clone());
        self.last_motifs = self.processor.take_last_motifs();
        let behavior_frame = self.processor.take_last_behavior_frame();
        let motif_replays = self.processor.take_last_motif_replays();
        self.last_report_metadata.clear();
        let quality_report = self
            .quality
            .report_for(batch.timestamp, batch.source_confidence.keys().copied());
        let dimension_report = self.dimensions.update(&batch);
        let scene_report = self.scene.update(&batch);
        let physiology_report = self
            .physiology
            .update(&batch, Some(self.knowledge.store()));
        let label_report = self.labels.update(&batch, dimension_report.as_ref());
        let visual_label_report = self.visual_labels.update(&batch);
        let motif_playback_report = self
            .motif_playback
            .update(&motif_replays, batch.timestamp);
        let network_report = self
            .network_fabric
            .update(&batch, behavior_frame.as_ref());
        let survival_report = behavior_frame
            .as_ref()
            .map(|frame| self.survival.update(frame, physiology_report.as_ref(), Some(self.knowledge.store())));
        let health_report =
            self.health_overlay
                .update(
                    &batch,
                    physiology_report.as_ref(),
                    dimension_report.as_ref(),
                    survival_report.as_ref(),
                );
        let analysis_report = self.analysis.update(&batch);
        let temporal_report = self.temporal.update(&batch);
        let plasticity_report = temporal_report
            .as_ref()
            .and_then(|report| self.plasticity.update(&batch, report, physiology_report.as_ref()));
        let ontology_report = self.ontology.update(&batch);
        let causal_report = temporal_report
            .as_ref()
            .and_then(|report| self.causal.update(report));
        let branching_report = temporal_report.as_ref().and_then(|report| {
            self.branching
                .update(&batch, report, causal_report.as_ref(), physiology_report.as_ref())
        });
        let metacognition_report = self.metacognition.update(
            &batch,
            behavior_frame.as_ref(),
            health_report.as_ref(),
            survival_report.as_ref(),
            temporal_report.as_ref(),
        );
        self.last_metacognition_share = metacognition_report
            .as_ref()
            .map(|report| self.metacognition.share_snapshot(report));
        let narrative_report = self.narrative.update(
            batch.timestamp,
            behavior_frame.as_ref(),
            health_report.as_ref(),
            survival_report.as_ref(),
            temporal_report.as_ref(),
            causal_report.as_ref(),
            branching_report.as_ref(),
            metacognition_report.as_ref(),
        );
        let spike_messages = self.spike_runtime.as_mut().and_then(|runtime| {
            runtime.route_batch(
                &batch,
                temporal_report.as_ref(),
                causal_report.as_ref(),
                branching_report.as_ref(),
                plasticity_report.as_ref(),
                ontology_report.as_ref(),
                physiology_report.as_ref(),
            )
        });
        let mut snapshot = token_batch_to_snapshot(&batch, &self.symbolizer);
        let mut report_metadata = HashMap::new();
        if let Some(report) = &neuro_snapshot {
            report_metadata.insert(
                "neuro_snapshot".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &cross_modal_report {
            report_metadata.insert(
                "cross_modal_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "cross_modal_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        let substream_report = self.substreams.report(batch.timestamp);
        let subnet_report = self.subnet_registry.update(&batch, &substream_report);
        report_metadata.insert(
            "substream_report".to_string(),
            serde_json::to_value(&substream_report)?,
        );
        snapshot.metadata.insert(
            "substream_report".to_string(),
            serde_json::to_value(substream_report)?,
        );
        if let Some(report) = &subnet_report {
            report_metadata.insert(
                "subnet_registry".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "subnet_registry".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = temporal_report {
            report_metadata.insert(
                "temporal_inference".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "temporal_inference".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = plasticity_report {
            report_metadata.insert(
                "plasticity_report".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "plasticity_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = ontology_report {
            report_metadata.insert(
                "ontology_report".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "ontology_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = physiology_report {
            report_metadata.insert(
                "physiology_report".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "physiology_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &dimension_report {
            report_metadata.insert(
                "dimension_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "dimension_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &scene_report {
            report_metadata.insert(
                "scene_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "scene_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &quality_report {
            report_metadata.insert(
                "quality_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "quality_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &label_report {
            report_metadata.insert(
                "label_queue".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "label_queue".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &visual_label_report {
            report_metadata.insert(
                "visual_label_queue".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "visual_label_queue".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &motif_playback_report {
            report_metadata.insert(
                "motif_playback_queue".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "motif_playback_queue".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = self.knowledge.queue_report(batch.timestamp) {
            report_metadata.insert(
                "knowledge_queue".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "knowledge_queue".to_string(),
                serde_json::to_value(report)?,
            );
        }
        let knowledge_store_report = self.knowledge.store_report(batch.timestamp);
        report_metadata.insert(
            "knowledge_store".to_string(),
            serde_json::to_value(&knowledge_store_report)?,
        );
        snapshot.metadata.insert(
            "knowledge_store".to_string(),
            serde_json::to_value(knowledge_store_report)?,
        );
        if let Some(report) = &survival_report {
            report_metadata.insert(
                "survival_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "survival_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &health_report {
            report_metadata.insert(
                "health_overlay".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "health_overlay".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &metacognition_report {
            report_metadata.insert(
                "metacognition_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "metacognition_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &narrative_report {
            report_metadata.insert(
                "narrative_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "narrative_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &analysis_report {
            report_metadata.insert(
                "analysis_report".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "analysis_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = &network_report {
            report_metadata.insert(
                "network_patterns".to_string(),
                serde_json::to_value(report)?,
            );
            snapshot.metadata.insert(
                "network_patterns".to_string(),
                serde_json::to_value(report)?,
            );
            self.last_network_patterns = report.shared_patterns.clone();
        } else {
            self.last_network_patterns.clear();
        }
        if let Some(report) = causal_report {
            report_metadata.insert(
                "causal_report".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "causal_report".to_string(),
                serde_json::to_value(report)?,
            );
        }
        if let Some(report) = branching_report {
            report_metadata.insert(
                "branching_futures".to_string(),
                serde_json::to_value(&report)?,
            );
            snapshot.metadata.insert(
                "branching_futures".to_string(),
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
        self.last_report_metadata = report_metadata;
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

fn sample_quality_from_meta(metadata: &HashMap<String, Value>) -> f64 {
    metadata
        .get("quality")
        .or_else(|| metadata.get("confidence"))
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

fn apply_phenotype_attrs(attrs: &mut HashMap<String, Value>, metadata: &HashMap<String, Value>) {
    for key in ["phenotype_signature", "phenotype_vector", "phenotype_confidence"] {
        if let Some(value) = metadata.get(key) {
            attrs.insert(key.to_string(), value.clone());
        }
    }
}

fn bbox_to_value(bbox: &crate::streaming::motor::BoundingBox) -> Value {
    let mut obj = Map::new();
    obj.insert("x".to_string(), Value::from(bbox.x));
    obj.insert("y".to_string(), Value::from(bbox.y));
    obj.insert("width".to_string(), Value::from(bbox.width));
    obj.insert("height".to_string(), Value::from(bbox.height));
    Value::Object(obj)
}

fn extract_text_overlay_tokens(
    metadata: &HashMap<String, Value>,
    entity_id: &str,
    timestamp: crate::schema::Timestamp,
    default_confidence: f64,
) -> Vec<EventToken> {
    let blocks = collect_text_blocks(metadata);
    if blocks.is_empty() {
        return Vec::new();
    }
    let frame_ref = frame_ref_from_metadata(metadata);
    let origin = metadata
        .get("ocr_engine")
        .and_then(|value| value.as_str())
        .filter(|text| !text.trim().is_empty())
        .unwrap_or("video_ocr");
    let mut tokens = Vec::new();
    for (idx, block) in blocks.into_iter().enumerate() {
        let text = block.text.trim();
        if text.is_empty() {
            continue;
        }
        let text_hash = compute_payload_hash(text.as_bytes());
        let annotation_id = compute_payload_hash(
            format!("ocr|{}|{}|{}", text_hash, timestamp.unix, idx).as_bytes(),
        );
        let mut attributes = HashMap::new();
        attributes.insert("text".to_string(), Value::String(text.to_string()));
        attributes.insert("text_hash".to_string(), Value::String(text_hash));
        attributes.insert("text_len".to_string(), Value::from(text.len() as u64));
        attributes.insert("entity_id".to_string(), Value::String(entity_id.to_string()));
        attributes.insert("annotation_id".to_string(), Value::String(annotation_id));
        attributes.insert("text_origin".to_string(), Value::String(origin.to_string()));
        if let Some(frame_id) = frame_ref.as_ref() {
            attributes.insert("frame_id".to_string(), Value::String(frame_id.clone()));
        }
        if let Some(conf) = block.confidence {
            attributes.insert("text_confidence".to_string(), Value::from(conf));
        }
        if let Some(bbox) = block.bbox {
            attributes.insert("text_bbox".to_string(), bbox);
        }
        let confidence = block
            .confidence
            .unwrap_or(default_confidence)
            .clamp(0.0, 1.0);
        tokens.push(EventToken {
            id: String::new(),
            kind: EventKind::TextAnnotation,
            onset: timestamp,
            duration_secs: 1.0,
            confidence,
            attributes,
            source: Some(StreamSource::TextAnnotations),
        });
    }
    tokens
}

struct TextOverlayBlock {
    text: String,
    confidence: Option<f64>,
    bbox: Option<Value>,
}

fn collect_text_blocks(metadata: &HashMap<String, Value>) -> Vec<TextOverlayBlock> {
    let mut blocks = Vec::new();
    let default_conf = metadata
        .get("ocr_confidence")
        .or_else(|| metadata.get("text_confidence"))
        .and_then(|value| value.as_f64())
        .filter(|val| val.is_finite())
        .map(|val| val.clamp(0.0, 1.0));
    if let Some(text) = metadata.get("ocr_text").and_then(|value| value.as_str()) {
        if !text.trim().is_empty() {
            blocks.push(TextOverlayBlock {
                text: text.to_string(),
                confidence: default_conf,
                bbox: None,
            });
        }
    }
    if let Some(value) = metadata.get("detected_text") {
        match value {
            Value::String(text) => {
                if !text.trim().is_empty() {
                    blocks.push(TextOverlayBlock {
                        text: text.to_string(),
                        confidence: default_conf,
                        bbox: None,
                    });
                }
            }
            Value::Array(items) => {
                for item in items {
                    if let Some(text) = item.as_str() {
                        if text.trim().is_empty() {
                            continue;
                        }
                        blocks.push(TextOverlayBlock {
                            text: text.to_string(),
                            confidence: default_conf,
                            bbox: None,
                        });
                        continue;
                    }
                    if let Some(obj) = item.as_object() {
                        let text = obj.get("text").and_then(|val| val.as_str());
                        if let Some(text) = text {
                            if text.trim().is_empty() {
                                continue;
                            }
                            let conf = obj.get("confidence").and_then(|val| val.as_f64());
                            let bbox = obj
                                .get("bbox")
                                .or_else(|| obj.get("box"))
                                .cloned();
                            blocks.push(TextOverlayBlock {
                                text: text.to_string(),
                                confidence: conf.or(default_conf),
                                bbox,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(Value::Array(items)) = metadata.get("text_blocks") {
        for item in items {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let Some(text) = obj.get("text").and_then(|val| val.as_str()) else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let conf = obj.get("confidence").and_then(|val| val.as_f64());
            let bbox = obj.get("bbox").or_else(|| obj.get("box")).cloned();
            blocks.push(TextOverlayBlock {
                text: text.to_string(),
                confidence: conf.or(default_conf),
                bbox,
            });
        }
    }
    blocks
}

fn frame_ref_from_metadata(metadata: &HashMap<String, Value>) -> Option<String> {
    for key in ["frame_id", "frame_ref", "frame_path", "image_ref"] {
        if let Some(text) = metadata.get(key).and_then(|value| value.as_str()) {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn empty_snapshot() -> EnvironmentSnapshot {
    EnvironmentSnapshot {
        timestamp: crate::schema::Timestamp { unix: 0 },
        bounds: HashMap::from([
            ("width".to_string(), 1.0),
            ("height".to_string(), 1.0),
            ("depth".to_string(), 1.0),
        ]),
        symbols: Vec::new(),
        metadata: HashMap::new(),
        stack_history: Vec::new(),
    }
}

fn copy_spatial_from_state(attrs: &mut HashMap<String, Value>, state: &BehaviorState) {
    for key in [
        "space_frame",
        "space_dimensionality",
        "space_confidence",
        "space_source",
        "pos_x",
        "pos_y",
        "pos_z",
    ] {
        if let Some(val) = state.attributes.get(key) {
            attrs.insert(key.to_string(), val.clone());
        }
    }
    if let Some(pos) = state.position {
        attrs.entry("pos_x".to_string()).or_insert_with(|| Value::from(pos[0]));
        attrs.entry("pos_y".to_string()).or_insert_with(|| Value::from(pos[1]));
        attrs.entry("pos_z".to_string()).or_insert_with(|| Value::from(pos[2]));
    }
}

fn species_from_meta(metadata: &HashMap<String, Value>) -> SpeciesKind {
    let raw = metadata
        .get("species")
        .and_then(|value| value.as_str())
        .unwrap_or("HUMAN");
    match raw.trim().to_ascii_uppercase().as_str() {
        "HUMAN" => SpeciesKind::Human,
        "CANINE" => SpeciesKind::Canine,
        "FELINE" => SpeciesKind::Feline,
        "AVIAN" => SpeciesKind::Avian,
        "AQUATIC" => SpeciesKind::Aquatic,
        "INSECT" => SpeciesKind::Insect,
        _ => SpeciesKind::Other(raw.to_string()),
    }
}

fn motif_confidence(motif: &BehaviorMotif) -> f64 {
    let support = motif.support as f64;
    let support_factor = support / (support + 3.0);
    let coherence = motif.graph_signature.mean_coherence.clamp(0.0, 1.0);
    (0.6 * support_factor + 0.4 * coherence).clamp(0.05, 1.0)
}

fn peer_weight_from_metadata(metadata: &HashMap<String, Value>) -> f64 {
    let weight = metadata
        .get("peer_weight")
        .and_then(|val| val.as_f64())
        .or_else(|| metadata.get("peer_score").and_then(|val| val.as_f64()))
        .or_else(|| {
            metadata
                .get("analysis_report")
                .and_then(|val| val.get("token_confidence"))
                .and_then(|val| val.get("mean"))
                .and_then(|val| val.as_f64())
        })
        .or_else(|| {
            metadata
                .get("quality_report")
                .and_then(|val| val.get("overall_quality"))
                .and_then(|val| val.as_f64())
        })
        .or_else(|| {
            metadata
                .get("plasticity_report")
                .and_then(|val| val.get("calibration_score"))
                .and_then(|val| val.as_f64())
        })
        .unwrap_or(1.0);
    weight.clamp(0.1, 2.0)
}

fn neuro_feedback_confidence(snapshot: &NeuroSnapshot) -> f64 {
    let confidence_mean = mean_map_values(&snapshot.prediction_confidence).unwrap_or(0.5);
    let surprise_mean = mean_map_values(&snapshot.surprise).unwrap_or(0.0);
    let surprise_confidence = 1.0 / (1.0 + surprise_mean);
    ((confidence_mean + surprise_confidence) * 0.5).clamp(0.0, 1.0)
}

fn mean_map_values(map: &HashMap<String, f64>) -> Option<f64> {
    if map.is_empty() {
        return None;
    }
    let sum: f64 = map.values().sum();
    Some(sum / map.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, StreamEnvelope, StreamPayload};
    use std::f64::consts::PI;

    #[test]
    fn processor_handles_people_signal() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.people_video = true;
        config.layer_flags.ultradian_enabled = true;
        config.layer_flags.behavior_enabled = true;
        let mut processor = StreamingProcessor::new(config);
        let mut last = None;
        let period_secs = 30.0 * 60.0;
        let omega = 2.0 * PI / period_secs;
        let base = 1_000_000_i64;
        for idx in 0..80 {
            let t = base + idx * 60;
            let signal = (omega * t as f64).sin();
            let envelope = StreamEnvelope {
                source: StreamSource::PeopleVideo,
                timestamp: Timestamp { unix: t },
                payload: StreamPayload::Json {
                    value: serde_json::json!({
                        "entity_id": "e1",
                        "signal": signal
                    }),
                },
                metadata: HashMap::new(),
            };
            last = processor.ingest(envelope);
        }
        let batch = last.expect("batch");
        assert!(!batch.layers.is_empty());
        assert!(batch.tokens.iter().any(|token| token.kind == EventKind::BehavioralToken));
    }

    #[test]
    fn processor_handles_people_pose_frame() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.people_video = true;
        config.layer_flags.ultradian_enabled = true;
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
    fn processor_extracts_text_overlay_tokens() {
        let mut config = StreamingConfig::default();
        config.enabled = true;
        config.ingest.people_video = true;
        let mut processor = StreamingProcessor::new(config);
        let envelope = StreamEnvelope {
            source: StreamSource::PeopleVideo,
            timestamp: Timestamp { unix: 100 },
            payload: StreamPayload::Json {
                value: serde_json::json!({
                    "entity_id": "e1",
                    "timestamp": { "unix": 100 },
                    "keypoints": [
                        { "name": "head", "x": 0.0, "y": 0.0, "confidence": 1.0 },
                        { "name": "hand", "x": 0.2, "y": 0.1, "confidence": 0.9 },
                        { "name": "foot", "x": -0.1, "y": -0.2, "confidence": 0.8 }
                    ],
                    "metadata": {
                        "ocr_text": "STOP SIGN",
                        "frame_id": "frame-1",
                        "ocr_confidence": 0.92
                    }
                }),
            },
            metadata: HashMap::new(),
        };
        let batch = processor.ingest(envelope).expect("batch");
        let text_tokens: Vec<&EventToken> = batch
            .tokens
            .iter()
            .filter(|token| token.kind == EventKind::TextAnnotation)
            .collect();
        assert!(!text_tokens.is_empty());
        let attrs = &text_tokens[0].attributes;
        assert_eq!(
            attrs.get("text").and_then(|val| val.as_str()),
            Some("STOP SIGN")
        );
        assert_eq!(
            attrs.get("frame_id").and_then(|val| val.as_str()),
            Some("frame-1")
        );
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
