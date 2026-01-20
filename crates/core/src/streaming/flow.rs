use crate::schema::Timestamp;
use crate::math_toolbox::RunningStats;
use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource};
use serde_json::Value;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct FlowSample {
    pub timestamp: Timestamp,
    pub density: f64,
    pub velocity: f64,
    pub direction_deg: f64,
}

#[derive(Debug, Clone)]
pub struct FlowConfig {
    pub window_secs: f64,
    pub max_age_secs: f64,
    pub density_norm: f64,
    pub velocity_norm: f64,
    pub stop_go_velocity_threshold: f64,
    pub stop_go_ratio_threshold: f64,
    pub congestion_density_threshold: f64,
    pub motif_density_threshold: f64,
    pub motif_velocity_threshold: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            window_secs: 15.0 * 60.0,
            max_age_secs: 60.0 * 60.0,
            density_norm: 50.0,
            velocity_norm: 20.0,
            stop_go_velocity_threshold: 1.0,
            stop_go_ratio_threshold: 0.25,
            congestion_density_threshold: 30.0,
            motif_density_threshold: 25.0,
            motif_velocity_threshold: 2.0,
        }
    }
}

#[derive(Debug, Default)]
pub struct FlowExtraction {
    pub layers: Vec<LayerState>,
    pub tokens: Vec<EventToken>,
}

pub struct FlowLayerExtractor {
    config: FlowConfig,
    samples: VecDeque<FlowSample>,
}

impl FlowLayerExtractor {
    pub fn new(config: FlowConfig) -> Self {
        Self {
            config,
            samples: VecDeque::new(),
        }
    }

    pub fn push_sample(&mut self, sample: FlowSample) {
        self.samples.push_back(sample);
        self.trim(sample.timestamp);
    }

    pub fn extract(&self) -> FlowExtraction {
        let mut extraction = FlowExtraction::default();
        let latest = match self.samples.back() {
            Some(sample) => sample.timestamp,
            None => return extraction,
        };
        let cutoff = latest.unix as f64 - self.config.window_secs;
        let mut window = Vec::new();
        for sample in &self.samples {
            if sample.timestamp.unix as f64 >= cutoff {
                window.push(*sample);
            }
        }
        if window.is_empty() {
            return extraction;
        }

        let mut density_sum = 0.0;
        let mut velocity_sum = 0.0;
        let mut dir_sin = 0.0;
        let mut dir_cos = 0.0;
        let mut stop_go_count = 0.0;
        for sample in &window {
            density_sum += sample.density;
            velocity_sum += sample.velocity;
            let rad = sample.direction_deg.to_radians();
            dir_sin += rad.sin();
            dir_cos += rad.cos();
            if sample.velocity <= self.config.stop_go_velocity_threshold {
                stop_go_count += 1.0;
            }
        }
        let count = window.len() as f64;
        let mean_density = density_sum / count;
        let mean_velocity = velocity_sum / count;
        let dir_mag = (dir_sin.powi(2) + dir_cos.powi(2)).sqrt() / count.max(1.0);
        let dir_phase = dir_sin.atan2(dir_cos);
        let stop_go_ratio = stop_go_count / count.max(1.0);
        let (density_std, velocity_std, snr_proxy) =
            window_stats(&window, mean_density, mean_velocity);

        extraction.layers.push(layer_state(
            LayerKind::FlowDensity,
            latest,
            0.0,
            (mean_density / self.config.density_norm).clamp(0.0, 1.0),
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));
        extraction.layers.push(layer_state(
            LayerKind::FlowVelocity,
            latest,
            0.0,
            (mean_velocity / self.config.velocity_norm).clamp(0.0, 1.0),
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));
        extraction.layers.push(layer_state(
            LayerKind::FlowDirectionality,
            latest,
            dir_phase,
            dir_mag.clamp(0.0, 1.0),
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));

        extraction.layers.push(layer_state(
            LayerKind::FlowStopGoWave,
            latest,
            0.0,
            stop_go_ratio.clamp(0.0, 1.0),
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));

        let motif_amp = if mean_density >= self.config.motif_density_threshold
            && mean_velocity <= self.config.motif_velocity_threshold
        {
            1.0
        } else {
            (mean_density / self.config.motif_density_threshold).clamp(0.0, 1.0) * 0.5
        };
        extraction.layers.push(layer_state(
            LayerKind::FlowMotif,
            latest,
            0.0,
            motif_amp,
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));

        let (daily_phase, weekly_phase) = seasonal_phases(latest);
        extraction.layers.push(layer_state(
            LayerKind::FlowSeasonalDaily,
            latest,
            daily_phase,
            1.0,
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));
        extraction.layers.push(layer_state(
            LayerKind::FlowSeasonalWeekly,
            latest,
            weekly_phase,
            1.0,
            0.0,
            base_attrs(&window, mean_density, mean_velocity, density_std, velocity_std, snr_proxy),
        ));

        apply_flow_coherence(&mut extraction.layers);

        if mean_density >= self.config.congestion_density_threshold {
            extraction.tokens.push(token(
                EventKind::CrowdToken,
                latest,
                60.0,
                0.8,
                [
                    ("density", Value::from(mean_density)),
                    ("velocity", Value::from(mean_velocity)),
                    ("snr_proxy", Value::from(snr_proxy)),
                ],
            ));
        }
        if stop_go_ratio >= self.config.stop_go_ratio_threshold {
            extraction.tokens.push(token(
                EventKind::TrafficToken,
                latest,
                60.0,
                stop_go_ratio,
                [
                    ("stop_go_ratio", Value::from(stop_go_ratio)),
                    ("velocity", Value::from(mean_velocity)),
                    ("snr_proxy", Value::from(snr_proxy)),
                ],
            ));
        }

        extraction
    }

    fn trim(&mut self, now: Timestamp) {
        let cutoff = now.unix as f64 - self.config.max_age_secs;
        while let Some(front) = self.samples.front() {
            if front.timestamp.unix as f64 >= cutoff {
                break;
            }
            self.samples.pop_front();
        }
    }
}

fn base_attrs(
    window: &[FlowSample],
    density: f64,
    velocity: f64,
    density_std: f64,
    velocity_std: f64,
    snr_proxy: f64,
) -> HashMap<String, Value> {
    let mut attrs = HashMap::new();
    attrs.insert("sample_count".to_string(), Value::from(window.len() as u64));
    attrs.insert("mean_density".to_string(), Value::from(density));
    attrs.insert("mean_velocity".to_string(), Value::from(velocity));
    attrs.insert("density_std".to_string(), Value::from(density_std));
    attrs.insert("velocity_std".to_string(), Value::from(velocity_std));
    attrs.insert("snr_proxy".to_string(), Value::from(snr_proxy));
    attrs
}

fn window_stats(window: &[FlowSample], mean_density: f64, mean_velocity: f64) -> (f64, f64, f64) {
    let mut density_stats = RunningStats::default();
    let mut velocity_stats = RunningStats::default();
    for sample in window {
        density_stats.update(sample.density);
        velocity_stats.update(sample.velocity);
    }
    let density_std = density_stats.stddev(0).unwrap_or(0.0);
    let velocity_std = velocity_stats.stddev(0).unwrap_or(0.0);
    let density_snr = mean_density / density_std.max(1e-6);
    let velocity_snr = mean_velocity / velocity_std.max(1e-6);
    let snr_proxy = ((density_snr + velocity_snr) * 0.5).clamp(0.0, 50.0);
    (density_std, velocity_std, snr_proxy)
}

fn layer_state(
    kind: LayerKind,
    timestamp: Timestamp,
    phase: f64,
    amplitude: f64,
    coherence: f64,
    attributes: HashMap<String, Value>,
) -> LayerState {
    LayerState {
        kind,
        timestamp,
        phase,
        amplitude,
        coherence,
        attributes,
    }
}

fn token(
    kind: EventKind,
    timestamp: Timestamp,
    duration_secs: f64,
    confidence: f64,
    attributes: impl IntoIterator<Item = (&'static str, Value)>,
) -> EventToken {
    let mut attrs = HashMap::new();
    for (key, value) in attributes {
        attrs.insert(key.to_string(), value);
    }
    EventToken {
        id: String::new(),
        kind,
        onset: timestamp,
        duration_secs,
        confidence,
        attributes: attrs,
        source: Some(StreamSource::CrowdTraffic),
    }
}

fn apply_flow_coherence(layers: &mut [LayerState]) {
    let mut sum = 0.0;
    let mut count = 0.0;
    for layer in layers.iter() {
        sum += layer.amplitude;
        count += 1.0;
    }
    let avg = if count > 0.0 { sum / count } else { 0.0 };
    for layer in layers.iter_mut() {
        layer.coherence = avg.clamp(0.0, 1.0);
    }
}

fn seasonal_phases(timestamp: Timestamp) -> (f64, f64) {
    let day_secs = 86_400.0;
    let week_secs = 7.0 * day_secs;
    let t = timestamp.unix as f64;
    let day_phase = 2.0 * PI * ((t % day_secs) / day_secs);
    let week_phase = 2.0 * PI * ((t % week_secs) / week_secs);
    (day_phase, week_phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_extractor_emits_density_and_tokens() {
        let mut extractor = FlowLayerExtractor::new(FlowConfig {
            congestion_density_threshold: 5.0,
            ..FlowConfig::default()
        });
        let base = 1_000_000_i64;
        for idx in 0..30 {
            extractor.push_sample(FlowSample {
                timestamp: Timestamp { unix: base + idx },
                density: 10.0,
                velocity: 0.5,
                direction_deg: 90.0,
            });
        }
        let extraction = extractor.extract();
        let density_layer = extraction
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::FlowDensity));
        assert!(density_layer.is_some());
        assert!(extraction.tokens.iter().any(|token| {
            matches!(token.kind, EventKind::CrowdToken)
        }));
    }
}
