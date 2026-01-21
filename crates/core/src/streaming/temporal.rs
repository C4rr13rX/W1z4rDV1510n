use crate::config::{StreamingHypergraphConfig, TemporalInferenceConfig};
use crate::schema::Timestamp;
use crate::streaming::hypergraph::{DomainKind, MultiDomainHypergraph};
use crate::streaming::schema::{EventKind, LayerKind, TokenBatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPrediction {
    pub kind: LayerKind,
    pub domain: DomainKind,
    pub predicted_phase: f64,
    pub predicted_amplitude: f64,
    pub predicted_coherence: f64,
    pub drift_phase: f64,
    pub drift_amplitude: f64,
    pub drift_coherence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencePrediction {
    pub layer_a: LayerKind,
    pub layer_b: LayerKind,
    pub coherence: f64,
    pub drift: f64,
    pub cross_domain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventIntensity {
    pub kind: EventKind,
    pub intensity: f64,
    pub expected_time_secs: f64,
    pub base_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirichletPosterior {
    pub label: String,
    pub categories: Vec<String>,
    pub alpha: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphStats {
    pub nodes: usize,
    pub edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInferenceReport {
    pub timestamp: Timestamp,
    pub layer_predictions: Vec<LayerPrediction>,
    pub coherence: Vec<CoherencePrediction>,
    pub event_intensities: Vec<EventIntensity>,
    pub evidential: Vec<DirichletPosterior>,
    pub next_event: Option<EventKind>,
    pub hypergraph: Option<HypergraphStats>,
}

#[derive(Debug, Clone)]
struct LayerMemory {
    ema_phase: f64,
    ema_amplitude: f64,
    ema_coherence: f64,
    last_phase: f64,
    last_amplitude: f64,
    last_coherence: f64,
    last_timestamp: Timestamp,
}

#[derive(Debug, Clone)]
struct EventMemory {
    decayed_count: f64,
    base_rate: f64,
    last_timestamp: Timestamp,
}

pub struct TemporalInferenceCore {
    config: TemporalInferenceConfig,
    hypergraph: MultiDomainHypergraph,
    layers: HashMap<LayerKind, LayerMemory>,
    events: HashMap<EventKind, EventMemory>,
    excitations: HashMap<(EventKind, EventKind), f64>,
    coherence_pairs: HashMap<(LayerKind, LayerKind), f64>,
    event_dirichlet: HashMap<EventKind, f64>,
    regime_dirichlet: HashMap<DomainKind, [f64; 3]>,
    last_timestamp: Option<Timestamp>,
}

impl TemporalInferenceCore {
    pub fn new(config: TemporalInferenceConfig, hypergraph: StreamingHypergraphConfig) -> Self {
        Self {
            config,
            hypergraph: MultiDomainHypergraph::new(hypergraph),
            layers: HashMap::new(),
            events: HashMap::new(),
            excitations: HashMap::new(),
            coherence_pairs: HashMap::new(),
            event_dirichlet: HashMap::new(),
            regime_dirichlet: HashMap::new(),
            last_timestamp: None,
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<TemporalInferenceReport> {
        if !self.config.enabled {
            return None;
        }
        let now = batch.timestamp;
        let dt = self
            .last_timestamp
            .map(|prev| timestamp_diff_secs(prev, now))
            .unwrap_or(0.0);
        let decay = (-dt / self.config.event_decay_tau_secs.max(1.0)).exp();
        for memory in self.events.values_mut() {
            memory.decayed_count *= decay;
        }
        for value in self.excitations.values_mut() {
            *value *= decay;
        }

        let hyper_update = self.hypergraph.update(batch);

        let event_counts = self.update_events(batch, now);
        let layer_predictions = self.update_layers(batch, now);
        let coherence = self.update_coherence(&layer_predictions);
        let event_intensities = self.predict_event_intensities(&event_counts);
        let next_event = event_intensities
            .iter()
            .max_by(|a, b| {
                a.intensity
                    .partial_cmp(&b.intensity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| entry.kind);
        let evidential = self.build_evidential(&event_counts, &event_intensities);
        self.last_timestamp = Some(now);
        Some(TemporalInferenceReport {
            timestamp: now,
            layer_predictions,
            coherence,
            event_intensities,
            evidential,
            next_event,
            hypergraph: hyper_update.map(|u| HypergraphStats {
                nodes: u.node_count,
                edges: u.edge_count,
            }),
        })
    }

    fn update_layers(&mut self, batch: &TokenBatch, now: Timestamp) -> Vec<LayerPrediction> {
        let alpha = self.config.layer_ema_alpha.clamp(0.0, 1.0);
        let coh_alpha = self.config.coherence_alpha.clamp(0.0, 1.0);
        for layer in &batch.layers {
            let entry = self.layers.entry(layer.kind).or_insert_with(|| LayerMemory {
                ema_phase: layer.phase,
                ema_amplitude: layer.amplitude,
                ema_coherence: layer.coherence,
                last_phase: layer.phase,
                last_amplitude: layer.amplitude,
                last_coherence: layer.coherence,
                last_timestamp: now,
            });
            entry.ema_phase = blend_phase(entry.ema_phase, layer.phase, alpha);
            entry.ema_amplitude = lerp(entry.ema_amplitude, layer.amplitude, alpha);
            entry.ema_coherence = lerp(entry.ema_coherence, layer.coherence, coh_alpha);
            entry.last_phase = layer.phase;
            entry.last_amplitude = layer.amplitude;
            entry.last_coherence = layer.coherence;
            entry.last_timestamp = now;
        }
        let mut predictions = Vec::new();
        for (kind, mem) in &self.layers {
            let drift_phase = wrap_phase_diff(mem.last_phase, mem.ema_phase);
            let drift_amp = mem.last_amplitude - mem.ema_amplitude;
            let drift_coh = mem.last_coherence - mem.ema_coherence;
            let predicted_phase = wrap_phase(mem.ema_phase + drift_phase);
            let predicted_amplitude = mem.ema_amplitude;
            let predicted_coherence = mem.ema_coherence;
            predictions.push(LayerPrediction {
                kind: *kind,
                domain: domain_for_layer(*kind),
                predicted_phase,
                predicted_amplitude,
                predicted_coherence,
                drift_phase,
                drift_amplitude: drift_amp,
                drift_coherence: drift_coh,
            });
        }
        predictions
    }

    fn update_events(&mut self, batch: &TokenBatch, now: Timestamp) -> HashMap<EventKind, f64> {
        let base_alpha = self.config.base_rate_alpha.clamp(0.0, 1.0);
        let mut counts: HashMap<EventKind, f64> = HashMap::new();
        let mut tokens_by_kind: Vec<(EventKind, f64)> = Vec::new();
        for token in &batch.tokens {
            let conf = token.confidence.max(0.0);
            *counts.entry(token.kind).or_insert(0.0) += conf;
            tokens_by_kind.push((token.kind, conf));
        }
        for (kind, count) in &counts {
            let entry = self.events.entry(*kind).or_insert_with(|| EventMemory {
                decayed_count: 0.0,
                base_rate: *count,
                last_timestamp: now,
            });
            entry.decayed_count += *count;
            entry.base_rate = lerp(entry.base_rate, *count, base_alpha);
            entry.last_timestamp = now;
            *self.event_dirichlet.entry(*kind).or_insert(self.config.dirichlet_prior) += *count;
        }
        self.update_excitation(&tokens_by_kind);
        counts
    }

    fn update_excitation(&mut self, tokens: &[(EventKind, f64)]) {
        for i in 0..tokens.len() {
            for j in (i + 1)..tokens.len() {
                let (a, a_conf) = tokens[i];
                let (b, b_conf) = tokens[j];
                if domain_for_event(a) == domain_for_event(b) {
                    continue;
                }
                let boost = (a_conf * b_conf * self.config.excitation_boost).clamp(0.0, 1.0);
                if boost <= 0.0 {
                    continue;
                }
                *self.excitations.entry((a, b)).or_insert(0.0) += boost;
                *self.excitations.entry((b, a)).or_insert(0.0) += boost;
            }
        }
    }

    fn predict_event_intensities(
        &self,
        counts: &HashMap<EventKind, f64>,
    ) -> Vec<EventIntensity> {
        let mut intensities = Vec::new();
        for kind in all_event_kinds() {
            let memory = self.events.get(&kind);
            let decayed = memory.map(|m| m.decayed_count).unwrap_or(0.0);
            let base_rate = memory.map(|m| m.base_rate).unwrap_or(0.0);
            let mut intensity = base_rate + decayed;
            for other in all_event_kinds() {
                if other == kind {
                    continue;
                }
                let other_memory = self.events.get(&other);
                if let Some(other_mem) = other_memory {
                    if let Some(weight) = self.excitations.get(&(other, kind)) {
                        intensity += weight * other_mem.decayed_count;
                    }
                }
            }
            if let Some(increment) = counts.get(&kind) {
                intensity += *increment;
            }
            intensity = intensity.max(self.config.intensity_floor);
            let expected_time_secs = 1.0 / intensity.max(1e-6);
            intensities.push(EventIntensity {
                kind,
                intensity,
                expected_time_secs,
                base_rate,
            });
        }
        intensities
    }

    fn update_coherence(&mut self, layers: &[LayerPrediction]) -> Vec<CoherencePrediction> {
        let mut output = Vec::new();
        for i in 0..layers.len() {
            for j in (i + 1)..layers.len() {
                let a = &layers[i];
                let b = &layers[j];
                let phase_diff = wrap_phase_diff(a.predicted_phase, b.predicted_phase);
                let phase_align = (phase_diff.cos() + 1.0) * 0.5;
                let weight = (a.predicted_amplitude * b.predicted_amplitude).max(1e-6);
                let coherence = (phase_align * weight).clamp(0.0, 1.0);
                let key = if layer_kind_rank(a.kind) <= layer_kind_rank(b.kind) {
                    (a.kind, b.kind)
                } else {
                    (b.kind, a.kind)
                };
                let prev = self.coherence_pairs.get(&key).copied().unwrap_or(coherence);
                let drift = coherence - prev;
                self.coherence_pairs.insert(key, coherence);
                output.push(CoherencePrediction {
                    layer_a: a.kind,
                    layer_b: b.kind,
                    coherence,
                    drift,
                    cross_domain: a.domain != b.domain,
                });
            }
        }
        output
    }

    fn build_evidential(
        &mut self,
        event_counts: &HashMap<EventKind, f64>,
        intensities: &[EventIntensity],
    ) -> Vec<DirichletPosterior> {
        let mut posteriors = Vec::new();
        let mut event_categories = Vec::new();
        let mut event_alpha = Vec::new();
        for kind in all_event_kinds() {
            event_categories.push(format!("{:?}", kind));
            let alpha = self
                .event_dirichlet
                .get(&kind)
                .copied()
                .unwrap_or(self.config.dirichlet_prior);
            event_alpha.push(alpha);
        }
        posteriors.push(DirichletPosterior {
            label: "event_kind".to_string(),
            categories: event_categories,
            alpha: event_alpha,
        });

        let mut domain_intensity: HashMap<DomainKind, f64> = HashMap::new();
        for intensity in intensities {
            *domain_intensity
                .entry(domain_for_event(intensity.kind))
                .or_insert(0.0) += intensity.intensity;
        }

        for domain in [
            DomainKind::People,
            DomainKind::Crowd,
            DomainKind::Topics,
            DomainKind::Text,
        ] {
            let total = *domain_intensity.get(&domain).unwrap_or(&0.0);
            let regime = classify_regime(total, self.config.calm_threshold, self.config.surge_threshold);
            let entry = self.regime_dirichlet.entry(domain).or_insert([
                self.config.dirichlet_prior,
                self.config.dirichlet_prior,
                self.config.dirichlet_prior,
            ]);
            match regime.as_str() {
                "CALM" => entry[0] += 1.0,
                "ACTIVE" => entry[1] += 1.0,
                "SURGE" => entry[2] += 1.0,
                _ => {}
            }
            posteriors.push(DirichletPosterior {
                label: format!("domain_regime::{:?}", domain),
                categories: vec!["CALM".to_string(), "ACTIVE".to_string(), "SURGE".to_string()],
                alpha: entry.to_vec(),
            });
        }

        if !event_counts.is_empty() {
            let _ = event_counts;
        }
        posteriors
    }
}

fn timestamp_diff_secs(a: Timestamp, b: Timestamp) -> f64 {
    (a.unix - b.unix).abs() as f64
}

fn lerp(a: f64, b: f64, alpha: f64) -> f64 {
    a + (b - a) * alpha
}

fn blend_phase(current: f64, target: f64, alpha: f64) -> f64 {
    let diff = wrap_phase_diff(target, current);
    wrap_phase(current + diff * alpha)
}

fn wrap_phase(value: f64) -> f64 {
    let mut v = value;
    while v > PI {
        v -= 2.0 * PI;
    }
    while v < -PI {
        v += 2.0 * PI;
    }
    v
}

fn wrap_phase_diff(a: f64, b: f64) -> f64 {
    let mut diff = a - b;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }
    diff
}

fn classify_regime(total: f64, calm: f64, surge: f64) -> String {
    if total <= calm {
        "CALM".to_string()
    } else if total >= surge {
        "SURGE".to_string()
    } else {
        "ACTIVE".to_string()
    }
}

fn all_event_kinds() -> [EventKind; 6] {
    [
        EventKind::BehavioralAtom,
        EventKind::BehavioralToken,
        EventKind::CrowdToken,
        EventKind::TrafficToken,
        EventKind::TopicEventToken,
        EventKind::TextAnnotation,
    ]
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

fn layer_kind_rank(kind: LayerKind) -> u8 {
    match kind {
        LayerKind::UltradianMicroArousal => 0,
        LayerKind::UltradianBrac => 1,
        LayerKind::UltradianMeso => 2,
        LayerKind::FlowDensity => 3,
        LayerKind::FlowVelocity => 4,
        LayerKind::FlowDirectionality => 5,
        LayerKind::FlowStopGoWave => 6,
        LayerKind::FlowMotif => 7,
        LayerKind::FlowSeasonalDaily => 8,
        LayerKind::FlowSeasonalWeekly => 9,
        LayerKind::TopicBurst => 10,
        LayerKind::TopicDecay => 11,
        LayerKind::TopicExcitation => 12,
        LayerKind::TopicLeadLag => 13,
        LayerKind::TopicPeriodicity => 14,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventToken, LayerState, StreamSource};

    #[test]
    fn temporal_core_emits_intensity_and_posteriors() {
        let config = TemporalInferenceConfig::default();
        let hyper_cfg = StreamingHypergraphConfig::default();
        let mut core = TemporalInferenceCore::new(config, hyper_cfg);
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 100 },
                duration_secs: 1.0,
                confidence: 1.0,
                attributes: HashMap::new(),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 100 },
                phase: 0.1,
                amplitude: 0.7,
                coherence: 0.4,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::new(),
        };
        let report = core.update(&batch).expect("report");
        assert!(!report.event_intensities.is_empty());
        assert!(!report.layer_predictions.is_empty());
        assert!(!report.evidential.is_empty());
    }
}
