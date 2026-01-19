use crate::config::StreamingAnalysisConfig;
use crate::math_toolbox as math;
use crate::schema::Timestamp;
use crate::streaming::schema::{EventKind, LayerKind, StreamSource, TokenBatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummary {
    pub count: usize,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p90: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSummary {
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisDrift {
    pub token_kind_js: f64,
    pub layer_amplitude_delta: f64,
    pub layer_coherence_delta: f64,
    pub source_confidence_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub timestamp: Timestamp,
    pub token_count: usize,
    pub layer_count: usize,
    pub token_kind_counts: HashMap<String, usize>,
    pub layer_kind_counts: HashMap<String, usize>,
    pub token_kind_entropy: f64,
    pub source_confidence_entropy: f64,
    pub token_duration: StatsSummary,
    pub token_confidence: StatsSummary,
    pub layer_amplitude: StatsSummary,
    pub layer_coherence: StatsSummary,
    pub layer_phase: PhaseSummary,
    pub drift: Option<AnalysisDrift>,
}

#[derive(Debug, Clone)]
struct AnalysisSnapshot {
    token_kind_probs: Vec<f64>,
    layer_amplitude_mean: f64,
    layer_coherence_mean: f64,
    source_confidence_entropy: f64,
}

pub struct StreamingAnalysisRuntime {
    config: StreamingAnalysisConfig,
    batch_counter: usize,
    last_snapshot: Option<AnalysisSnapshot>,
}

impl StreamingAnalysisRuntime {
    pub fn new(config: StreamingAnalysisConfig) -> Self {
        Self {
            config,
            batch_counter: 0,
            last_snapshot: None,
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<AnalysisReport> {
        if !self.config.enabled {
            return None;
        }
        self.batch_counter = self.batch_counter.saturating_add(1);
        if self.batch_counter % self.config.interval_batches.max(1) != 0 {
            return None;
        }
        let tokens = batch
            .tokens
            .iter()
            .take(self.config.max_tokens.max(1));
        let layers = batch
            .layers
            .iter()
            .take(self.config.max_layers.max(1));

        let mut token_count = 0usize;
        let mut layer_count = 0usize;
        let mut token_kind_counts = HashMap::new();
        let mut layer_kind_counts = HashMap::new();
        let mut token_durations = Vec::new();
        let mut token_confidences = Vec::new();
        let mut layer_amplitudes = Vec::new();
        let mut layer_coherences = Vec::new();
        let mut layer_phases = Vec::new();

        for token in tokens {
            token_count += 1;
            let key = event_kind_label(token.kind).to_string();
            *token_kind_counts.entry(key).or_insert(0) += 1;
            token_durations.push(token.duration_secs);
            token_confidences.push(token.confidence);
        }
        for layer in layers {
            layer_count += 1;
            let key = layer_kind_label(layer.kind).to_string();
            *layer_kind_counts.entry(key).or_insert(0) += 1;
            layer_amplitudes.push(layer.amplitude);
            layer_coherences.push(layer.coherence);
            layer_phases.push(layer.phase);
        }

        let token_kind_probs = kind_probs(&token_kind_counts);
        let token_kind_entropy = math::entropy(&token_kind_probs);
        let source_confidence = source_confidence_vector(&batch.source_confidence);
        let source_confidence_entropy = math::entropy(&source_confidence);

        let token_duration = summary_stats(&token_durations);
        let token_confidence = summary_stats(&token_confidences);
        let layer_amplitude = summary_stats(&layer_amplitudes);
        let layer_coherence = summary_stats(&layer_coherences);
        let layer_phase = phase_summary(&layer_phases);

        let snapshot = AnalysisSnapshot {
            token_kind_probs: token_kind_probs.clone(),
            layer_amplitude_mean: layer_amplitude.mean,
            layer_coherence_mean: layer_coherence.mean,
            source_confidence_entropy,
        };
        let drift = self
            .last_snapshot
            .as_ref()
            .and_then(|prev| build_drift(prev, &snapshot));
        self.last_snapshot = Some(snapshot);

        Some(AnalysisReport {
            timestamp: batch.timestamp,
            token_count,
            layer_count,
            token_kind_counts,
            layer_kind_counts,
            token_kind_entropy,
            source_confidence_entropy,
            token_duration,
            token_confidence,
            layer_amplitude,
            layer_coherence,
            layer_phase,
            drift,
        })
    }
}

fn summary_stats(values: &[f64]) -> StatsSummary {
    let count = values.iter().filter(|v| v.is_finite()).count();
    let mean = math::mean(values).unwrap_or(0.0);
    let stddev = math::stddev(values, 0.0).unwrap_or(0.0);
    let (min, max) = math::min_max(values).unwrap_or((0.0, 0.0));
    let p50 = math::percentile(values, 0.5).unwrap_or(0.0);
    let p90 = math::percentile(values, 0.9).unwrap_or(0.0);
    StatsSummary {
        count,
        mean,
        stddev,
        min,
        max,
        p50,
        p90,
    }
}

fn phase_summary(values: &[f64]) -> PhaseSummary {
    let count = values.iter().filter(|v| v.is_finite()).count();
    let mean = math::circular_mean(values).unwrap_or(0.0);
    let variance = math::circular_variance(values).unwrap_or(0.0);
    PhaseSummary {
        count,
        mean,
        variance,
    }
}

fn kind_probs(counts: &HashMap<String, usize>) -> Vec<f64> {
    let order = [
        event_kind_label(EventKind::BehavioralAtom),
        event_kind_label(EventKind::BehavioralToken),
        event_kind_label(EventKind::CrowdToken),
        event_kind_label(EventKind::TrafficToken),
        event_kind_label(EventKind::TopicEventToken),
    ];
    let mut values = Vec::with_capacity(order.len());
    for key in order {
        values.push(*counts.get(key).unwrap_or(&0) as f64);
    }
    math::normalize_probs(&values)
}

fn source_confidence_vector(map: &HashMap<StreamSource, f64>) -> Vec<f64> {
    let order = [
        StreamSource::PeopleVideo,
        StreamSource::CrowdTraffic,
        StreamSource::PublicTopics,
    ];
    let mut values = Vec::with_capacity(order.len());
    for source in order {
        values.push(map.get(&source).copied().unwrap_or(0.0));
    }
    math::normalize_probs(&values)
}

fn build_drift(prev: &AnalysisSnapshot, curr: &AnalysisSnapshot) -> Option<AnalysisDrift> {
    let token_kind_js = math::js_divergence(&prev.token_kind_probs, &curr.token_kind_probs)?;
    let layer_amplitude_delta = (curr.layer_amplitude_mean - prev.layer_amplitude_mean).abs();
    let layer_coherence_delta = (curr.layer_coherence_mean - prev.layer_coherence_mean).abs();
    let source_confidence_delta = (curr.source_confidence_entropy - prev.source_confidence_entropy).abs();
    Some(AnalysisDrift {
        token_kind_js,
        layer_amplitude_delta,
        layer_coherence_delta,
        source_confidence_delta,
    })
}

fn event_kind_label(kind: EventKind) -> &'static str {
    match kind {
        EventKind::BehavioralAtom => "BEHAVIORAL_ATOM",
        EventKind::BehavioralToken => "BEHAVIORAL_TOKEN",
        EventKind::CrowdToken => "CROWD_TOKEN",
        EventKind::TrafficToken => "TRAFFIC_TOKEN",
        EventKind::TopicEventToken => "TOPIC_EVENT_TOKEN",
    }
}

fn layer_kind_label(kind: LayerKind) -> &'static str {
    match kind {
        LayerKind::UltradianMicroArousal => "ULTRADIAN_MICRO_AROUSAL",
        LayerKind::UltradianBrac => "ULTRADIAN_BRAC",
        LayerKind::UltradianMeso => "ULTRADIAN_MESO",
        LayerKind::FlowDensity => "FLOW_DENSITY",
        LayerKind::FlowVelocity => "FLOW_VELOCITY",
        LayerKind::FlowDirectionality => "FLOW_DIRECTIONALITY",
        LayerKind::FlowStopGoWave => "FLOW_STOP_GO_WAVE",
        LayerKind::FlowMotif => "FLOW_MOTIF",
        LayerKind::FlowSeasonalDaily => "FLOW_SEASONAL_DAILY",
        LayerKind::FlowSeasonalWeekly => "FLOW_SEASONAL_WEEKLY",
        LayerKind::TopicBurst => "TOPIC_BURST",
        LayerKind::TopicDecay => "TOPIC_DECAY",
        LayerKind::TopicExcitation => "TOPIC_EXCITATION",
        LayerKind::TopicLeadLag => "TOPIC_LEAD_LAG",
        LayerKind::TopicPeriodicity => "TOPIC_PERIODICITY",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventToken, LayerState, StreamSource};

    #[test]
    fn analysis_report_emits_on_interval() {
        let mut runtime = StreamingAnalysisRuntime::new(StreamingAnalysisConfig {
            enabled: true,
            interval_batches: 2,
            max_tokens: 16,
            max_layers: 16,
        });
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 1 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralToken,
                onset: Timestamp { unix: 1 },
                duration_secs: 2.0,
                confidence: 0.8,
                attributes: HashMap::new(),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 1 },
                phase: 0.5,
                amplitude: 0.7,
                coherence: 0.6,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::from([(StreamSource::PeopleVideo, 0.9)]),
        };
        assert!(runtime.update(&batch).is_none());
        assert!(runtime.update(&batch).is_some());
    }
}
