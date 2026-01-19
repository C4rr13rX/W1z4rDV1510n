use crate::config::StreamingConfig;
use crate::orchestrator::{RunOutcome, run_with_snapshot};
use crate::streaming::behavior::{
    BehaviorInput, BehaviorMotif, BehaviorState, BehaviorSubstrate, BehaviorSubstrateConfig,
    SensorKind, SensorSample, SpeciesKind,
};
use crate::streaming::fabric::NeuralFabricShare;
use crate::streaming::flow::{FlowLayerExtractor, FlowSample, FlowConfig};
use crate::streaming::align::StreamingAligner;
use crate::streaming::motor::{MotorFeatureExtractor, PoseFrame};
use crate::streaming::branching_runtime::StreamingBranchingRuntime;
use crate::streaming::causal_stream::StreamingCausalRuntime;
use crate::streaming::dimensions::DimensionTracker;
use crate::streaming::health_overlay::HealthOverlayRuntime;
use crate::streaming::survival::SurvivalRuntime;
use crate::streaming::ontology_runtime::OntologyRuntime;
use crate::streaming::physiology_runtime::PhysiologyRuntime;
use crate::streaming::plasticity_runtime::StreamingPlasticityRuntime;
use crate::streaming::schema::{
    EventKind, EventToken, StreamEnvelope, StreamPayload, StreamSource, TokenBatch,
};
use crate::streaming::labeling::LabelQueue;
use crate::streaming::symbolize::{SymbolizeConfig, token_batch_to_snapshot};
use crate::streaming::topic::{TopicEventExtractor, TopicSample, TopicConfig};
use crate::streaming::spike_runtime::StreamingSpikeRuntime;
use crate::streaming::temporal::TemporalInferenceCore;
use crate::streaming::ultradian::{SignalSample, UltradianLayerExtractor};
use crate::streaming::spatial::insert_spatial_attrs;
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
    behavior_substrate: BehaviorSubstrate,
    flow_extractor: FlowLayerExtractor,
    topic_extractor: TopicEventExtractor,
    last_behavior_motifs: Vec<BehaviorMotif>,
    last_behavior_frame: Option<crate::streaming::behavior::BehaviorFrame>,
}

impl StreamingProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            people_extractors: HashMap::new(),
            motor_extractor: MotorFeatureExtractor::new(crate::streaming::motor::MotorConfig::default()),
            behavior_substrate: BehaviorSubstrate::new(BehaviorSubstrateConfig::default()),
            flow_extractor: FlowLayerExtractor::new(FlowConfig::default()),
            topic_extractor: TopicEventExtractor::new(TopicConfig::default()),
            last_behavior_motifs: Vec::new(),
            last_behavior_frame: None,
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
        }
    }

    pub fn take_last_motifs(&mut self) -> Vec<BehaviorMotif> {
        std::mem::take(&mut self.last_behavior_motifs)
    }

    pub fn take_last_behavior_frame(&mut self) -> Option<crate::streaming::behavior::BehaviorFrame> {
        self.last_behavior_frame.take()
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
        Some(share.to_token_batch())
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
        frame: PoseFrame,
        metadata: HashMap<String, Value>,
    ) -> Option<TokenBatch> {
        let behavior_frame = frame.clone();
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
        if let Some(area) = output.bbox_area {
            attributes.insert("bbox_area".to_string(), Value::from(area));
        }
        if !output.metadata.is_empty() {
            attributes.insert(
                "metadata".to_string(),
                Value::Object(map_from_metadata(&output.metadata)),
            );
        }
        let behavior_input = BehaviorInput {
            entity_id: output.entity_id.clone(),
            timestamp: output.timestamp,
            species: species_from_meta(&metadata),
            sensors: Vec::new(),
            actions: Vec::new(),
            pose: Some(behavior_frame),
            metadata: metadata.clone(),
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
    temporal: TemporalInferenceCore,
    causal: StreamingCausalRuntime,
    branching: StreamingBranchingRuntime,
    plasticity: StreamingPlasticityRuntime,
    ontology: OntologyRuntime,
    physiology: PhysiologyRuntime,
    dimensions: DimensionTracker,
    labels: LabelQueue,
    health_overlay: HealthOverlayRuntime,
    survival: SurvivalRuntime,
    spike_runtime: Option<StreamingSpikeRuntime>,
    run_config: RunConfig,
    symbolizer: SymbolizeConfig,
    last_batch: Option<TokenBatch>,
    last_motifs: Vec<BehaviorMotif>,
    last_report_metadata: HashMap<String, Value>,
}

impl StreamingInference {
    pub fn new(run_config: RunConfig) -> Self {
        let processor = StreamingProcessor::new(run_config.streaming.clone());
        let aligner = StreamingAligner::new(&run_config.streaming);
        let temporal = TemporalInferenceCore::new(
            run_config.streaming.temporal.clone(),
            run_config.streaming.hypergraph.clone(),
        );
        let causal = StreamingCausalRuntime::new(run_config.streaming.causal.clone());
        let branching = StreamingBranchingRuntime::new(run_config.streaming.branching.clone());
        let plasticity = StreamingPlasticityRuntime::new(run_config.streaming.plasticity.clone());
        let ontology =
            OntologyRuntime::new(run_config.streaming.ontology.clone(), run_config.streaming.consistency.clone());
        let mut physiology_config = run_config.streaming.physiology.clone();
        physiology_config.enabled =
            physiology_config.enabled && run_config.streaming.layer_flags.physiology_enabled;
        let physiology = PhysiologyRuntime::new(physiology_config);
        let dimensions = DimensionTracker::default();
        let labels = LabelQueue::default();
        let health_overlay = HealthOverlayRuntime::default();
        let survival = SurvivalRuntime::default();
        let spike_runtime = if run_config.streaming.spike.enabled {
            Some(StreamingSpikeRuntime::new(run_config.streaming.spike.clone()))
        } else {
            None
        };
        Self {
            processor,
            aligner,
            temporal,
            causal,
            branching,
            plasticity,
            ontology,
            physiology,
            dimensions,
            labels,
            health_overlay,
            survival,
            spike_runtime,
            run_config,
            symbolizer: SymbolizeConfig::default(),
            last_batch: None,
            last_motifs: Vec::new(),
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
        let batch = match self.processor.ingest_fabric_share(share) {
            Some(batch) => batch,
            None => return Ok(None),
        };
        self.process_batch(batch)
    }

    pub fn take_last_fabric_share(&mut self, node_id: String) -> Option<NeuralFabricShare> {
        let batch = self.last_batch.take()?;
        let motifs = std::mem::take(&mut self.last_motifs);
        let mut share = NeuralFabricShare::from_batch(node_id, batch, motifs);
        if !self.last_report_metadata.is_empty() {
            share.metadata = std::mem::take(&mut self.last_report_metadata);
        }
        Some(share)
    }

    fn process_batch(&mut self, batch: TokenBatch) -> anyhow::Result<Option<RunOutcome>> {
        let batch = match self.aligner.push(batch) {
            Some(batch) => batch,
            None => return Ok(None),
        };
        self.last_batch = Some(batch.clone());
        self.last_motifs = self.processor.take_last_motifs();
        let behavior_frame = self.processor.take_last_behavior_frame();
        self.last_report_metadata.clear();
        let dimension_report = self.dimensions.update(&batch);
        let physiology_report = self.physiology.update(&batch);
        let label_report = self.labels.update(&batch, dimension_report.as_ref());
        let survival_report = behavior_frame
            .as_ref()
            .map(|frame| self.survival.update(frame, physiology_report.as_ref()));
        let health_report =
            self.health_overlay
                .update(
                    &batch,
                    physiology_report.as_ref(),
                    dimension_report.as_ref(),
                    survival_report.as_ref(),
                );
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
        config.layer_flags.ultradian_enabled = true;
        config.layer_flags.behavior_enabled = true;
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
