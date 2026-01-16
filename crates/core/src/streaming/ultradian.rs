use crate::schema::Timestamp;
use crate::streaming::schema::{LayerKind, LayerState};
use std::collections::VecDeque;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct SignalSample {
    pub timestamp: Timestamp,
    pub value: f64,
}

#[derive(Debug)]
pub struct SignalSeries {
    samples: VecDeque<SignalSample>,
    max_age_secs: f64,
}

impl SignalSeries {
    pub fn new(max_age_secs: f64) -> Self {
        Self {
            samples: VecDeque::new(),
            max_age_secs: max_age_secs.max(1.0),
        }
    }

    pub fn push(&mut self, sample: SignalSample) {
        self.samples.push_back(sample);
        self.trim(sample.timestamp);
    }

    pub fn samples(&self) -> &VecDeque<SignalSample> {
        &self.samples
    }

    fn trim(&mut self, now: Timestamp) {
        let cutoff = now.unix as f64 - self.max_age_secs;
        while let Some(front) = self.samples.front() {
            if front.timestamp.unix as f64 >= cutoff {
                break;
            }
            self.samples.pop_front();
        }
    }
}

#[derive(Debug, Clone)]
pub struct UltradianBand {
    pub kind: LayerKind,
    pub min_period_secs: f64,
    pub max_period_secs: f64,
}

impl UltradianBand {
    pub fn center_period(&self) -> f64 {
        0.5 * (self.min_period_secs + self.max_period_secs)
    }
}

pub struct UltradianLayerExtractor {
    bands: Vec<UltradianBand>,
    series: SignalSeries,
    min_samples: usize,
}

impl UltradianLayerExtractor {
    pub fn new() -> Self {
        let bands = vec![
            UltradianBand {
                kind: LayerKind::UltradianMicroArousal,
                min_period_secs: 20.0 * 60.0,
                max_period_secs: 40.0 * 60.0,
            },
            UltradianBand {
                kind: LayerKind::UltradianBrac,
                min_period_secs: 80.0 * 60.0,
                max_period_secs: 120.0 * 60.0,
            },
            UltradianBand {
                kind: LayerKind::UltradianMeso,
                min_period_secs: 2.0 * 3600.0,
                max_period_secs: 6.0 * 3600.0,
            },
        ];
        let max_age_secs = bands
            .iter()
            .map(|band| band.max_period_secs * 3.0)
            .fold(0.0, f64::max);
        Self::with_bands(bands, max_age_secs, 32)
    }

    pub fn with_bands(
        bands: Vec<UltradianBand>,
        max_age_secs: f64,
        min_samples: usize,
    ) -> Self {
        Self {
            bands,
            series: SignalSeries::new(max_age_secs),
            min_samples: min_samples.max(4),
        }
    }

    pub fn push_sample(&mut self, sample: SignalSample) {
        self.series.push(sample);
    }

    pub fn extract_layers(&self) -> Vec<LayerState> {
        let latest = self
            .series
            .samples()
            .back()
            .map(|sample| sample.timestamp)
            .unwrap_or(Timestamp { unix: 0 });
        let mut layers: Vec<LayerState> = self
            .bands
            .iter()
            .filter_map(|band| self.estimate_layer(band, latest))
            .collect();
        apply_coherence(&mut layers);
        layers
    }

    fn estimate_layer(&self, band: &UltradianBand, timestamp: Timestamp) -> Option<LayerState> {
        let window_secs = band.max_period_secs * 2.0;
        let cutoff = timestamp.unix as f64 - window_secs;
        let mut samples = Vec::new();
        for sample in self.series.samples() {
            if sample.timestamp.unix as f64 >= cutoff {
                samples.push(*sample);
            }
        }
        if samples.len() < self.min_samples {
            return None;
        }
        let period = band.center_period();
        let omega = 2.0 * PI / period.max(1.0);
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;
        let mut sum = 0.0;
        let mut sum_abs = 0.0;
        for sample in &samples {
            let t = sample.timestamp.unix as f64;
            let value = sample.value;
            sum += value;
            sum_abs += value.abs();
            sum_sin += value * (omega * t).sin();
            sum_cos += value * (omega * t).cos();
        }
        let mean = sum / samples.len() as f64;
        let mean_abs = sum_abs / samples.len() as f64;
        let magnitude = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
        let denom = (mean_abs.max(1e-6)) * samples.len() as f64;
        let amplitude = (magnitude / denom).clamp(0.0, 1.0);
        let phase = sum_sin.atan2(sum_cos);
        let mut attributes = std::collections::HashMap::new();
        attributes.insert(
            "min_period_secs".to_string(),
            serde_json::Value::from(band.min_period_secs),
        );
        attributes.insert(
            "max_period_secs".to_string(),
            serde_json::Value::from(band.max_period_secs),
        );
        attributes.insert(
            "center_period_secs".to_string(),
            serde_json::Value::from(period),
        );
        attributes.insert(
            "sample_count".to_string(),
            serde_json::Value::from(samples.len() as u64),
        );
        attributes.insert("mean".to_string(), serde_json::Value::from(mean));
        attributes.insert(
            "mean_abs".to_string(),
            serde_json::Value::from(mean_abs),
        );
        Some(LayerState {
            kind: band.kind,
            timestamp,
            phase,
            amplitude,
            coherence: 0.0,
            attributes,
        })
    }
}

fn apply_coherence(layers: &mut [LayerState]) {
    if layers.len() < 2 {
        if let Some(layer) = layers.first_mut() {
            layer.coherence = 1.0;
        }
        return;
    }
    for idx in 0..layers.len() {
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for j in 0..layers.len() {
            if idx == j {
                continue;
            }
            let phase_diff = wrap_phase_diff(layers[idx].phase, layers[j].phase);
            let phase_align = (phase_diff.cos() + 1.0) * 0.5;
            let weight = (layers[idx].amplitude * layers[j].amplitude).max(1e-6);
            weighted_sum += phase_align * weight;
            weight_total += weight;
        }
        layers[idx].coherence = if weight_total > 0.0 {
            (weighted_sum / weight_total).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_detects_micro_arousal_band() {
        let mut extractor = UltradianLayerExtractor::new();
        let period_secs = 30.0 * 60.0;
        let omega = 2.0 * PI / period_secs;
        let start = 1_000_000_i64;
        for idx in 0..360 {
            let t = start + (idx * 60) as i64;
            let value = (omega * t as f64).sin();
            extractor.push_sample(SignalSample {
                timestamp: Timestamp { unix: t },
                value,
            });
        }
        let layers = extractor.extract_layers();
        let micro = layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::UltradianMicroArousal));
        assert!(micro.is_some());
        let micro = micro.unwrap();
        assert!(micro.amplitude > 0.2);
        assert!((micro.phase).is_finite());
    }
}
