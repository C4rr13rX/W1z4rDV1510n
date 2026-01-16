use crate::config::StreamingSpikeConfig;
use crate::spike::{NeuronKind, SpikeConfig, SpikeInput, SpikeMessage, SpikeMessageBus, SpikePool};
use crate::streaming::branching_runtime::BranchingReport;
use crate::streaming::causal_stream::CausalReport;
use crate::streaming::schema::TokenBatch;
use crate::streaming::temporal::{DirichletPosterior, TemporalInferenceReport};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamingSpikePoolKind {
    Ingest,
    TokenGrammar,
    LayerExtraction,
    Fusion,
    Hypergraph,
    Temporal,
    EventIntensity,
    Evidential,
    Causal,
    Branching,
    Retrodiction,
    Ontology,
}

impl StreamingSpikePoolKind {
    pub fn all() -> &'static [StreamingSpikePoolKind] {
        &[
            StreamingSpikePoolKind::Ingest,
            StreamingSpikePoolKind::TokenGrammar,
            StreamingSpikePoolKind::LayerExtraction,
            StreamingSpikePoolKind::Fusion,
            StreamingSpikePoolKind::Hypergraph,
            StreamingSpikePoolKind::Temporal,
            StreamingSpikePoolKind::EventIntensity,
            StreamingSpikePoolKind::Evidential,
            StreamingSpikePoolKind::Causal,
            StreamingSpikePoolKind::Branching,
            StreamingSpikePoolKind::Retrodiction,
            StreamingSpikePoolKind::Ontology,
        ]
    }
}

pub struct StreamingSpikeRuntime {
    config: StreamingSpikeConfig,
    pools: HashMap<StreamingSpikePoolKind, SpikePool>,
    neuron_map: HashMap<(StreamingSpikePoolKind, String), u32>,
}

impl StreamingSpikeRuntime {
    pub fn new(config: StreamingSpikeConfig) -> Self {
        let spike_config = SpikeConfig {
            threshold: config.threshold,
            membrane_decay: config.membrane_decay,
            refractory_steps: config.refractory_steps,
        };
        let mut pools = HashMap::new();
        for kind in StreamingSpikePoolKind::all() {
            let id = format!("streaming::{}", pool_label(*kind));
            pools.insert(*kind, SpikePool::new(id, spike_config.clone()));
        }
        Self {
            config,
            pools,
            neuron_map: HashMap::new(),
        }
    }

    pub fn route_batch(
        &mut self,
        batch: &TokenBatch,
        temporal: Option<&TemporalInferenceReport>,
        causal: Option<&CausalReport>,
        branching: Option<&BranchingReport>,
    ) -> Option<Vec<SpikeMessage>> {
        if !self.config.enabled {
            return None;
        }
        let mut inputs: HashMap<StreamingSpikePoolKind, Vec<SpikeInput>> = HashMap::new();
        self.route_ingest(batch, &mut inputs);
        self.route_tokens(batch, &mut inputs);
        self.route_layers(batch, &mut inputs);
        self.route_fusion(batch, &mut inputs);
        if let Some(report) = temporal {
            self.route_temporal(report, &mut inputs);
            self.route_event_intensity(report, &mut inputs);
            self.route_evidential(report, &mut inputs);
            self.route_hypergraph(report, &mut inputs);
        }
        if let Some(report) = causal {
            self.route_causal(report, &mut inputs);
        }
        if let Some(report) = branching {
            self.route_branching(report, &mut inputs);
        }

        let mut bus = SpikeMessageBus::default();
        for (kind, mut pool_inputs) in inputs {
            if pool_inputs.is_empty() {
                continue;
            }
            trim_inputs(&mut pool_inputs, self.config.max_inputs_per_pool);
            if let Some(pool) = self.pools.get_mut(&kind) {
                pool.enqueue_inputs(pool_inputs);
                let frame = pool.step(batch.timestamp);
                bus.publish(SpikeMessage {
                    pool_id: pool.id.clone(),
                    frame,
                });
            }
        }
        let messages = bus.drain();
        if messages.is_empty() {
            None
        } else {
            Some(messages)
        }
    }

    pub fn embed_in_snapshot(&self) -> bool {
        self.config.embed_in_snapshot
    }

    pub fn max_frames_per_snapshot(&self) -> usize {
        self.config.max_frames_per_snapshot.max(1)
    }

    fn route_ingest(&mut self, batch: &TokenBatch, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let total_items = batch.tokens.len() + batch.layers.len();
        if total_items == 0 {
            return;
        }
        let activity = (total_items as f64 / self.config.max_inputs_per_pool.max(1) as f64)
            .clamp(0.0, 1.0);
        let confidence = average_confidence(batch).clamp(0.0, 1.0);
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Ingest, "ingest") {
            push_input(inputs, StreamingSpikePoolKind::Ingest, SpikeInput {
                target: neuron,
                excitatory: (activity * confidence) as f32,
                inhibitory: (1.0 - confidence) as f32,
            });
        }
    }

    fn route_tokens(&mut self, batch: &TokenBatch, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        for token in &batch.tokens {
            let label = format!("event::{:?}", token.kind);
            let conf = token.confidence.clamp(0.0, 1.0);
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::TokenGrammar, &label) {
                push_input(inputs, StreamingSpikePoolKind::TokenGrammar, SpikeInput {
                    target: neuron,
                    excitatory: conf as f32,
                    inhibitory: 0.0,
                });
            }
        }
    }

    fn route_layers(&mut self, batch: &TokenBatch, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        for layer in &batch.layers {
            let label = format!("layer::{:?}", layer.kind);
            let excitatory = (layer.amplitude * (1.0 + layer.coherence)).clamp(0.0, 2.0);
            let inhibitory = (1.0 - layer.coherence).clamp(0.0, 1.0);
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::LayerExtraction, &label) {
                push_input(inputs, StreamingSpikePoolKind::LayerExtraction, SpikeInput {
                    target: neuron,
                    excitatory: excitatory as f32,
                    inhibitory: inhibitory as f32,
                });
            }
        }
    }

    fn route_fusion(&mut self, batch: &TokenBatch, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let mut sources = 0.0;
        let mut conf_sum = 0.0;
        for (source, confidence) in &batch.source_confidence {
            sources += 1.0;
            conf_sum += confidence.clamp(0.0, 1.0);
            let label = format!("source::{source:?}");
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Fusion, &label) {
                push_input(inputs, StreamingSpikePoolKind::Fusion, SpikeInput {
                    target: neuron,
                    excitatory: confidence.clamp(0.0, 1.0) as f32,
                    inhibitory: 0.0,
                });
            }
        }
        if sources > 0.0 {
            let avg = (conf_sum / sources).clamp(0.0, 1.0);
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Fusion, "fusion_ready") {
                push_input(inputs, StreamingSpikePoolKind::Fusion, SpikeInput {
                    target: neuron,
                    excitatory: avg as f32,
                    inhibitory: (1.0 - avg) as f32,
                });
            }
        }
    }

    fn route_temporal(&mut self, report: &TemporalInferenceReport, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        for pred in &report.layer_predictions {
            let label = format!("pred::{:?}", pred.kind);
            let excitatory = pred.predicted_amplitude.clamp(0.0, 1.0);
            let drift = pred.drift_amplitude.abs() + pred.drift_coherence.abs();
            let inhibitory = drift.clamp(0.0, 1.0);
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Temporal, &label) {
                push_input(inputs, StreamingSpikePoolKind::Temporal, SpikeInput {
                    target: neuron,
                    excitatory: excitatory as f32,
                    inhibitory: inhibitory as f32,
                });
            }
        }
    }

    fn route_event_intensity(&mut self, report: &TemporalInferenceReport, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let norm = self.config.intensity_norm.max(1e-6);
        for intensity in &report.event_intensities {
            let label = format!("intensity::{:?}", intensity.kind);
            let excitatory = (intensity.intensity / norm).clamp(0.0, 1.0);
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::EventIntensity, &label) {
                push_input(inputs, StreamingSpikePoolKind::EventIntensity, SpikeInput {
                    target: neuron,
                    excitatory: excitatory as f32,
                    inhibitory: 0.0,
                });
            }
        }
    }

    fn route_evidential(&mut self, report: &TemporalInferenceReport, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let norm = self.config.evidence_norm.max(1e-6);
        for posterior in &report.evidential {
            let (confidence, uncertainty) = posterior_scores(posterior, norm);
            let label = format!("evidence::{}", posterior.label);
            if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Evidential, &label) {
                push_input(inputs, StreamingSpikePoolKind::Evidential, SpikeInput {
                    target: neuron,
                    excitatory: confidence as f32,
                    inhibitory: uncertainty as f32,
                });
            }
        }
    }

    fn route_hypergraph(&mut self, report: &TemporalInferenceReport, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let Some(stats) = &report.hypergraph else {
            return;
        };
        let node_norm = self.config.hypergraph_node_norm.max(1.0);
        let edge_norm = self.config.hypergraph_edge_norm.max(1.0);
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Hypergraph, "nodes") {
            let excitatory = (stats.nodes as f64 / node_norm).clamp(0.0, 1.0);
            push_input(inputs, StreamingSpikePoolKind::Hypergraph, SpikeInput {
                target: neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
        }
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Hypergraph, "edges") {
            let excitatory = (stats.edges as f64 / edge_norm).clamp(0.0, 1.0);
            push_input(inputs, StreamingSpikePoolKind::Hypergraph, SpikeInput {
                target: neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
        }
    }

    fn route_causal(&mut self, report: &CausalReport, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let edge_norm = self.config.hypergraph_edge_norm.max(1.0);
        let intervention_norm = self.config.intensity_norm.max(1.0);
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Causal, "edges") {
            let excitatory = (report.edge_count as f64 / edge_norm).clamp(0.0, 1.0);
            push_input(inputs, StreamingSpikePoolKind::Causal, SpikeInput {
                target: neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
        }
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Causal, "interventions") {
            let excitatory = (report.interventions.len() as f64 / intervention_norm).clamp(0.0, 1.0);
            push_input(inputs, StreamingSpikePoolKind::Causal, SpikeInput {
                target: neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
        }
    }

    fn route_branching(&mut self, report: &BranchingReport, inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>) {
        let branch_norm = self.config.hypergraph_node_norm.max(1.0);
        let retro_norm = self.config.hypergraph_edge_norm.max(1.0);
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Branching, "branches") {
            let excitatory = (report.branches.len() as f64 / branch_norm).clamp(0.0, 1.0);
            push_input(inputs, StreamingSpikePoolKind::Branching, SpikeInput {
                target: neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
        }
        if let Some(neuron) = self.neuron_for(StreamingSpikePoolKind::Retrodiction, "retrodictions") {
            let excitatory = (report.retrodictions.len() as f64 / retro_norm).clamp(0.0, 1.0);
            push_input(inputs, StreamingSpikePoolKind::Retrodiction, SpikeInput {
                target: neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
        }
    }

    fn neuron_for(&mut self, pool: StreamingSpikePoolKind, label: &str) -> Option<u32> {
        let key = (pool, label.to_string());
        if let Some(id) = self.neuron_map.get(&key) {
            return Some(*id);
        }
        let pool_ref = self.pools.get_mut(&pool)?;
        if pool_ref.neuron_count() >= self.config.max_neurons_per_pool {
            return None;
        }
        let neuron = pool_ref.add_neuron(NeuronKind::Excitatory);
        self.neuron_map.insert(key, neuron);
        Some(neuron)
    }
}

fn pool_label(kind: StreamingSpikePoolKind) -> &'static str {
    match kind {
        StreamingSpikePoolKind::Ingest => "ingest",
        StreamingSpikePoolKind::TokenGrammar => "token_grammar",
        StreamingSpikePoolKind::LayerExtraction => "layer_extract",
        StreamingSpikePoolKind::Fusion => "fusion",
        StreamingSpikePoolKind::Hypergraph => "hypergraph",
        StreamingSpikePoolKind::Temporal => "temporal",
        StreamingSpikePoolKind::EventIntensity => "event_intensity",
        StreamingSpikePoolKind::Evidential => "evidential",
        StreamingSpikePoolKind::Causal => "causal",
        StreamingSpikePoolKind::Branching => "branching",
        StreamingSpikePoolKind::Retrodiction => "retrodiction",
        StreamingSpikePoolKind::Ontology => "ontology",
    }
}

fn push_input(
    inputs: &mut HashMap<StreamingSpikePoolKind, Vec<SpikeInput>>,
    pool: StreamingSpikePoolKind,
    input: SpikeInput,
) {
    inputs.entry(pool).or_default().push(input);
}

fn trim_inputs(inputs: &mut Vec<SpikeInput>, max_inputs: usize) {
    if inputs.len() <= max_inputs {
        return;
    }
    inputs.sort_by(|a, b| {
        let strength_a = a.excitatory - a.inhibitory;
        let strength_b = b.excitatory - b.inhibitory;
        strength_b
            .partial_cmp(&strength_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    inputs.truncate(max_inputs);
}

fn average_confidence(batch: &TokenBatch) -> f64 {
    if batch.source_confidence.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    for value in batch.source_confidence.values() {
        sum += value.clamp(0.0, 1.0);
    }
    sum / batch.source_confidence.len() as f64
}

fn posterior_scores(posterior: &DirichletPosterior, norm: f64) -> (f64, f64) {
    if posterior.alpha.is_empty() {
        return (0.0, 0.0);
    }
    let sum = posterior.alpha.iter().sum::<f64>();
    let k = posterior.alpha.len() as f64;
    let confidence = (sum / norm).clamp(0.0, 1.0);
    let entropy = dirichlet_entropy(posterior, sum, k);
    (confidence, entropy)
}

fn dirichlet_entropy(posterior: &DirichletPosterior, sum: f64, k: f64) -> f64 {
    if sum <= 0.0 || k <= 1.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for alpha in &posterior.alpha {
        let prob = (*alpha / sum).clamp(1e-9, 1.0);
        entropy -= prob * prob.ln();
    }
    let max_entropy = k.ln();
    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource};

    #[test]
    fn spike_runtime_routes_token_and_layer_inputs() {
        let mut config = StreamingSpikeConfig::default();
        config.enabled = true;
        config.max_inputs_per_pool = 8;
        let mut runtime = StreamingSpikeRuntime::new(config);
        let batch = TokenBatch {
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
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 10 },
                phase: 0.1,
                amplitude: 0.8,
                coherence: 0.6,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::from([(StreamSource::PeopleVideo, 0.9)]),
        };
        let messages = runtime
            .route_batch(&batch, None, None, None)
            .expect("messages");
        assert!(!messages.is_empty());
    }
}
