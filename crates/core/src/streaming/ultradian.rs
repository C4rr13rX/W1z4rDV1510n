use crate::schema::Timestamp;
use crate::streaming::schema::{LayerKind, LayerState};
use std::collections::VecDeque;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct SignalSample {
    pub timestamp: Timestamp,
    pub value: f64,
    pub quality: f64,
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
    candidate_count: usize,
    min_quality: f64,
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
        Self::with_bands(bands, max_age_secs, 32, 5, 0.1)
    }

    pub fn with_bands(
        bands: Vec<UltradianBand>,
        max_age_secs: f64,
        min_samples: usize,
        candidate_count: usize,
        min_quality: f64,
    ) -> Self {
        Self {
            bands,
            series: SignalSeries::new(max_age_secs),
            min_samples: min_samples.max(4),
            candidate_count: candidate_count.max(1),
            min_quality: min_quality.clamp(0.0, 1.0),
        }
    }

    pub fn push_sample(&mut self, sample: SignalSample) {
        let mut sample = sample;
        sample.quality = sample.quality.clamp(0.0, 1.0);
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
                if sample.quality >= self.min_quality {
                    samples.push(*sample);
                }
            }
        }
        if samples.len() < self.min_samples {
            return None;
        }
        let mut weight_total = 0.0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut min_quality: f64 = 1.0;
        let mut max_quality: f64 = 0.0;
        for sample in &samples {
            let weight = sample.quality.clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            let value = sample.value;
            weight_total += weight;
            sum += weight * value;
            sum_sq += weight * value * value;
            min_quality = min_quality.min(weight);
            max_quality = max_quality.max(weight);
        }
        if weight_total <= 0.0 {
            return None;
        }
        let mean = sum / weight_total;
        let variance = (sum_sq / weight_total) - mean * mean;
        let rms = variance.max(0.0).sqrt().max(1e-6);
        let mut best_period = band.center_period();
        let mut best_amplitude = 0.0;
        let mut best_phase = 0.0;
        for period in self.candidate_periods(band) {
            let omega = 2.0 * PI / period.max(1.0);
            let mut sum_sin = 0.0;
            let mut sum_cos = 0.0;
            for sample in &samples {
                let weight = sample.quality.clamp(0.0, 1.0);
                if weight <= 0.0 {
                    continue;
                }
                let t = sample.timestamp.unix as f64;
                let centered = sample.value - mean;
                sum_sin += weight * centered * (omega * t).sin();
                sum_cos += weight * centered * (omega * t).cos();
            }
            let magnitude = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
            let amplitude = (magnitude / (weight_total * rms) * 2.0_f64.sqrt()).clamp(0.0, 1.0);
            if amplitude > best_amplitude {
                best_amplitude = amplitude;
                best_phase = sum_sin.atan2(sum_cos);
                best_period = period;
            }
        }
        let phase = best_phase;
        let amplitude = best_amplitude;
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
            serde_json::Value::from(band.center_period()),
        );
        attributes.insert(
            "best_period_secs".to_string(),
            serde_json::Value::from(best_period),
        );
        attributes.insert(
            "sample_count".to_string(),
            serde_json::Value::from(samples.len() as u64),
        );
        attributes.insert(
            "weighted_samples".to_string(),
            serde_json::Value::from(weight_total),
        );
        attributes.insert("mean".to_string(), serde_json::Value::from(mean));
        attributes.insert("rms".to_string(), serde_json::Value::from(rms));
        attributes.insert(
            "quality_avg".to_string(),
            serde_json::Value::from((weight_total / samples.len() as f64).clamp(0.0, 1.0)),
        );
        attributes.insert(
            "quality_min".to_string(),
            serde_json::Value::from(min_quality),
        );
        attributes.insert(
            "quality_max".to_string(),
            serde_json::Value::from(max_quality),
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

    fn candidate_periods(&self, band: &UltradianBand) -> Vec<f64> {
        if self.candidate_count <= 1 || band.min_period_secs <= 0.0 || band.max_period_secs <= 0.0 {
            return vec![band.center_period()];
        }
        let min_log = band.min_period_secs.ln();
        let max_log = band.max_period_secs.ln();
        let mut periods = Vec::with_capacity(self.candidate_count);
        let steps = (self.candidate_count - 1) as f64;
        for idx in 0..self.candidate_count {
            let frac = if steps > 0.0 {
                idx as f64 / steps
            } else {
                0.0
            };
            let period = (min_log + frac * (max_log - min_log)).exp();
            periods.push(period);
        }
        periods
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
                quality: 1.0,
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

    #[test]
    fn extractor_ignores_low_quality_samples() {
        let mut extractor = UltradianLayerExtractor::new();
        let period_secs = 30.0 * 60.0;
        let omega = 2.0 * PI / period_secs;
        let start = 2_000_000_i64;
        for idx in 0..40 {
            let t = start + (idx * 60) as i64;
            let value = (omega * t as f64).sin();
            extractor.push_sample(SignalSample {
                timestamp: Timestamp { unix: t },
                value,
                quality: 0.05,
            });
        }
        let layers = extractor.extract_layers();
        assert!(layers.is_empty());
    }
}
