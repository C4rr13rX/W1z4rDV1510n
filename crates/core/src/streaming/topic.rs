use crate::schema::Timestamp;
use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct TopicSample {
    pub timestamp: Timestamp,
    pub topic: String,
    pub intensity: f64,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct TopicConfig {
    pub window_secs: f64,
    pub ema_alpha: f64,
    pub burst_threshold: f64,
    pub min_intensity: f64,
    pub burst_duration_secs: f64,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            window_secs: 30.0 * 60.0,
            ema_alpha: 0.2,
            burst_threshold: 2.0,
            min_intensity: 0.1,
            burst_duration_secs: 120.0,
        }
    }
}

#[derive(Debug, Default)]
pub struct TopicExtraction {
    pub layers: Vec<LayerState>,
    pub tokens: Vec<EventToken>,
}

pub struct TopicEventExtractor {
    config: TopicConfig,
    samples: VecDeque<TopicSample>,
    ema: HashMap<String, f64>,
    burst_history: HashMap<String, VecDeque<i64>>,
}

impl TopicEventExtractor {
    pub fn new(config: TopicConfig) -> Self {
        Self {
            config,
            samples: VecDeque::new(),
            ema: HashMap::new(),
            burst_history: HashMap::new(),
        }
    }

    pub fn push_sample(&mut self, sample: TopicSample) {
        let topic = sample.topic.clone();
        let ema_next = {
            let entry = self.ema.entry(topic.clone()).or_insert(sample.intensity);
            let next =
                self.config.ema_alpha * sample.intensity + (1.0 - self.config.ema_alpha) * *entry;
            *entry = next;
            next
        };
        if self.is_burst(sample.intensity, ema_next) {
            let history = self.burst_history.entry(topic).or_default();
            history.push_back(sample.timestamp.unix);
        }
        self.samples.push_back(sample);
        self.trim();
    }

    pub fn extract(&self) -> TopicExtraction {
        let mut extraction = TopicExtraction::default();
        let latest = match self.samples.back() {
            Some(sample) => sample.timestamp,
            None => return extraction,
        };
        let cutoff = latest.unix as f64 - self.config.window_secs;
        let mut window: Vec<&TopicSample> = Vec::new();
        for sample in &self.samples {
            if sample.timestamp.unix as f64 >= cutoff {
                window.push(sample);
            }
        }
        if window.is_empty() {
            return extraction;
        }

        let mut max_burst: f64 = 0.0;
        let mut decay_sum = 0.0;
        let mut decay_count = 0.0;
        let mut burst_count = 0u64;
        let mut earliest_burst: Option<i64> = None;
        let mut latest_burst: Option<i64> = None;

        for sample in &window {
            let ema = self.ema.get(&sample.topic).copied().unwrap_or(sample.intensity);
            let burst_score = sample.intensity / ema.max(1e-6);
            if self.is_burst(sample.intensity, ema) {
                burst_count += 1;
                max_burst = max_burst.max(burst_score);
                let ts = sample.timestamp.unix;
                earliest_burst = Some(earliest_burst.map(|v| v.min(ts)).unwrap_or(ts));
                latest_burst = Some(latest_burst.map(|v| v.max(ts)).unwrap_or(ts));
                extraction.tokens.push(topic_token(
                    sample,
                    burst_score,
                    self.config.burst_duration_secs,
                ));
            } else {
                let delta = ema - sample.intensity;
                if delta > 0.0 {
                    decay_sum += delta / ema.max(1e-6);
                    decay_count += 1.0;
                }
            }
        }

        let burst_amp = (max_burst / self.config.burst_threshold).clamp(0.0, 1.0);
        let decay_amp = if decay_count > 0.0 {
            (decay_sum / decay_count).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let excitation_amp = (burst_count as f64 / window.len() as f64).clamp(0.0, 1.0);
        let lead_lag_amp = if let (Some(start), Some(end)) = (earliest_burst, latest_burst) {
            let span = (end - start) as f64;
            (span / self.config.window_secs).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (period_amp, period_phase, period_secs) = self.periodicity(latest);

        extraction.layers.push(layer(
            LayerKind::TopicBurst,
            latest,
            0.0,
            burst_amp,
            0.0,
            layer_attrs("burst_count", burst_count, period_secs),
        ));
        extraction.layers.push(layer(
            LayerKind::TopicDecay,
            latest,
            0.0,
            decay_amp,
            0.0,
            layer_attrs("decay_samples", decay_count as u64, period_secs),
        ));
        extraction.layers.push(layer(
            LayerKind::TopicExcitation,
            latest,
            0.0,
            excitation_amp,
            0.0,
            layer_attrs("excitation_samples", window.len() as u64, period_secs),
        ));
        extraction.layers.push(layer(
            LayerKind::TopicLeadLag,
            latest,
            0.0,
            lead_lag_amp,
            0.0,
            layer_attrs("lead_lag_span_secs", lead_lag_amp, period_secs),
        ));
        extraction.layers.push(layer(
            LayerKind::TopicPeriodicity,
            latest,
            period_phase,
            period_amp,
            0.0,
            layer_attrs("period_secs", period_secs, period_secs),
        ));
        apply_topic_coherence(&mut extraction.layers);

        extraction
    }

    fn is_burst(&self, intensity: f64, ema: f64) -> bool {
        intensity >= self.config.min_intensity
            && intensity / ema.max(1e-6) >= self.config.burst_threshold
    }

    fn periodicity(&self, now: Timestamp) -> (f64, f64, f64) {
        let mut intervals = Vec::new();
        for history in self.burst_history.values() {
            if history.len() < 3 {
                continue;
            }
            let mut local = Vec::new();
            let mut iter = history.iter();
            let mut prev = match iter.next() {
                Some(val) => *val,
                None => continue,
            };
            for ts in iter {
                let delta = (*ts - prev) as f64;
                if delta > 0.0 {
                    local.push(delta);
                }
                prev = *ts;
            }
            intervals.extend(local);
        }
        if intervals.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let mut variance = 0.0;
        for val in &intervals {
            variance += (val - mean).powi(2);
        }
        let std = (variance / intervals.len() as f64).sqrt();
        let cv = std / mean.max(1e-6);
        let amplitude = (1.0 - cv).clamp(0.0, 1.0);
        let phase = if mean > 0.0 {
            let elapsed = (now.unix as f64) % mean;
            2.0 * PI * (elapsed / mean)
        } else {
            0.0
        };
        (amplitude, phase, mean)
    }

    fn trim(&mut self) {
        let latest = match self.samples.back() {
            Some(sample) => sample.timestamp,
            None => return,
        };
        let cutoff = latest.unix as f64 - self.config.window_secs;
        while let Some(front) = self.samples.front() {
            if front.timestamp.unix as f64 >= cutoff {
                break;
            }
            self.samples.pop_front();
        }
        for history in self.burst_history.values_mut() {
            while let Some(front) = history.front() {
                if *front as f64 >= cutoff {
                    break;
                }
                history.pop_front();
            }
        }
    }
}

fn topic_token(sample: &TopicSample, burst_score: f64, duration_secs: f64) -> EventToken {
    let mut attrs = sample.metadata.clone();
    attrs.insert("topic".to_string(), Value::String(sample.topic.clone()));
    attrs.insert("intensity".to_string(), Value::from(sample.intensity));
    attrs.insert("burst_score".to_string(), Value::from(burst_score));
    EventToken {
        id: String::new(),
        kind: EventKind::TopicEventToken,
        onset: sample.timestamp,
        duration_secs,
        confidence: burst_score.clamp(0.0, 1.0),
        attributes: attrs,
        source: Some(StreamSource::PublicTopics),
    }
}

fn layer(
    kind: LayerKind,
    timestamp: Timestamp,
    phase: f64,
    amplitude: f64,
    coherence: f64,
    mut attributes: HashMap<String, Value>,
) -> LayerState {
    attributes.insert("timestamp_unix".to_string(), Value::from(timestamp.unix));
    LayerState {
        kind,
        timestamp,
        phase,
        amplitude,
        coherence,
        attributes,
    }
}

fn layer_attrs(key: &str, value: impl Into<Value>, period_secs: f64) -> HashMap<String, Value> {
    let mut attrs = HashMap::new();
    attrs.insert(key.to_string(), value.into());
    if period_secs > 0.0 {
        attrs.insert("period_secs".to_string(), Value::from(period_secs));
    }
    attrs
}

fn apply_topic_coherence(layers: &mut [LayerState]) {
    let mut sum = 0.0;
    for layer in layers.iter() {
        sum += layer.amplitude;
    }
    let avg = if layers.is_empty() { 0.0 } else { sum / layers.len() as f64 };
    for layer in layers.iter_mut() {
        layer.coherence = avg.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_extractor_emits_burst_and_layer() {
        let mut extractor = TopicEventExtractor::new(TopicConfig {
            burst_threshold: 1.2,
            ..TopicConfig::default()
        });
        let base = 1_000_000_i64;
        for idx in 0..10 {
            extractor.push_sample(TopicSample {
                timestamp: Timestamp { unix: base + idx },
                topic: "news".to_string(),
                intensity: if idx == 9 { 5.0 } else { 1.0 },
                metadata: HashMap::new(),
            });
        }
        let extraction = extractor.extract();
        assert!(extraction
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::TopicBurst)));
        assert!(extraction
            .tokens
            .iter()
            .any(|token| matches!(token.kind, EventKind::TopicEventToken)));
    }
}
