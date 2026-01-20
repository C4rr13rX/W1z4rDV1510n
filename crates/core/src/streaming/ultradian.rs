use crate::schema::Timestamp;
use crate::streaming::schema::{LayerKind, LayerState};
use crate::math_toolbox::RunningStats;
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
    min_cycles: usize,
    min_confidence: f64,
    phase_smoothing: f64,
    resample_jitter_threshold: f64,
    resample_max_points: usize,
    last_phases: std::collections::HashMap<LayerKind, f64>,
    last_phase_times: std::collections::HashMap<LayerKind, Timestamp>,
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
        Self::with_bands(bands, max_age_secs, 32, 5, 0.1, 2, 0.1, 0.6, 0.35, 512)
    }

    pub fn with_bands(
        bands: Vec<UltradianBand>,
        max_age_secs: f64,
        min_samples: usize,
        candidate_count: usize,
        min_quality: f64,
        min_cycles: usize,
        min_confidence: f64,
        phase_smoothing: f64,
        resample_jitter_threshold: f64,
        resample_max_points: usize,
    ) -> Self {
        Self {
            bands,
            series: SignalSeries::new(max_age_secs),
            min_samples: min_samples.max(4),
            candidate_count: candidate_count.max(1),
            min_quality: min_quality.clamp(0.0, 1.0),
            min_cycles: min_cycles.max(1),
            min_confidence: min_confidence.clamp(0.0, 1.0),
            phase_smoothing: phase_smoothing.clamp(0.0, 1.0),
            resample_jitter_threshold: resample_jitter_threshold.clamp(0.0, 5.0),
            resample_max_points: resample_max_points.max(64),
            last_phases: std::collections::HashMap::new(),
            last_phase_times: std::collections::HashMap::new(),
        }
    }

    pub fn push_sample(&mut self, sample: SignalSample) {
        let mut sample = sample;
        sample.quality = sample.quality.clamp(0.0, 1.0);
        self.series.push(sample);
    }

    pub fn extract_layers(&mut self) -> Vec<LayerState> {
        let latest = self
            .series
            .samples()
            .back()
            .map(|sample| sample.timestamp)
            .unwrap_or(Timestamp { unix: 0 });
        let mut layers = Vec::new();
        let bands = self.bands.clone();
        for band in &bands {
            if let Some(layer) = self.estimate_layer(band, latest) {
                layers.push(layer);
            }
        }
        apply_coherence(&mut layers);
        layers
    }

    fn estimate_layer(&mut self, band: &UltradianBand, timestamp: Timestamp) -> Option<LayerState> {
        let window_secs = band.max_period_secs * 2.0;
        let cutoff = timestamp.unix as f64 - window_secs;
        let mut samples = Vec::new();
        let mut min_ts = f64::INFINITY;
        let mut max_ts = f64::NEG_INFINITY;
        for sample in self.series.samples() {
            if sample.timestamp.unix as f64 >= cutoff {
                if sample.quality >= self.min_quality {
                    samples.push(*sample);
                    let ts = sample.timestamp.unix as f64;
                    min_ts = min_ts.min(ts);
                    max_ts = max_ts.max(ts);
                }
            }
        }
        if samples.len() < self.min_samples {
            return None;
        }
        let (sample_dt_mean, sample_dt_std, jitter_ratio) = sample_dt_metrics(&samples);
        let mut resampled = false;
        if jitter_ratio >= self.resample_jitter_threshold && samples.len() >= self.min_samples {
            if let Some(resampled_samples) =
                resample_uniform(&samples, band, self.resample_max_points)
            {
                if resampled_samples.len() >= self.min_samples {
                    samples = resampled_samples;
                    resampled = true;
                }
            }
        }
        let span_secs = (max_ts - min_ts).max(0.0);
        let min_span_secs = band.min_period_secs * self.min_cycles as f64;
        if span_secs < min_span_secs {
            return None;
        }
        let mut weight_total = 0.0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut sum_t = 0.0;
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
            sum_t += weight * sample.timestamp.unix as f64;
            min_quality = min_quality.min(weight);
            max_quality = max_quality.max(weight);
        }
        if weight_total <= 0.0 {
            return None;
        }
        let mean = sum / weight_total;
        let variance = (sum_sq / weight_total) - mean * mean;
        let t_mean = sum_t / weight_total;
        let mut slope_num = 0.0;
        let mut slope_den = 0.0;
        for sample in &samples {
            let weight = sample.quality.clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            let dt = sample.timestamp.unix as f64 - t_mean;
            slope_num += weight * dt * (sample.value - mean);
            slope_den += weight * dt * dt;
        }
        let slope = if slope_den > 1e-6 { slope_num / slope_den } else { 0.0 };
        let mut residual_sum_sq = 0.0;
        for sample in &samples {
            let weight = sample.quality.clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            let dt = sample.timestamp.unix as f64 - t_mean;
            let detrended = (sample.value - mean) - slope * dt;
            residual_sum_sq += weight * detrended * detrended;
        }
        let rms = (residual_sum_sq / weight_total).max(0.0).sqrt().max(1e-6);
        let (sample_dt_mean_resampled, sample_dt_std_resampled, _) = sample_dt_metrics(&samples);
        let mut best_period = band.center_period();
        let mut best_amplitude = 0.0;
        let mut best_phase = 0.0;
        let mut best_magnitude = 0.0;
        let t_ref = timestamp.unix as f64;
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
                let centered = (sample.value - mean) - slope * (t - t_mean);
                let t_offset = t - t_ref;
                sum_sin += weight * centered * (omega * t_offset).sin();
                sum_cos += weight * centered * (omega * t_offset).cos();
            }
            let magnitude = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
            let amplitude = (magnitude / (weight_total * rms) * 2.0_f64.sqrt()).clamp(0.0, 1.0);
            if amplitude > best_amplitude {
                best_amplitude = amplitude;
                best_phase = sum_sin.atan2(sum_cos);
                best_period = period;
                best_magnitude = magnitude;
            }
        }
        let omega_best = 2.0 * PI / best_period.max(1.0);
        let amplitude_fit = if weight_total > 0.0 {
            (2.0 * best_magnitude / weight_total).abs()
        } else {
            0.0
        };
        let mut noise_sum = 0.0;
        for sample in &samples {
            let weight = sample.quality.clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            let t = sample.timestamp.unix as f64;
            let centered = (sample.value - mean) - slope * (t - t_mean);
            let t_offset = t - t_ref;
            let fitted = amplitude_fit * (omega_best * t_offset + best_phase).sin();
            let residual = centered - fitted;
            noise_sum += weight * residual * residual;
        }
        let noise_rms = if weight_total > 0.0 {
            (noise_sum / weight_total).sqrt()
        } else {
            0.0
        };
        let noise_rms = if noise_rms.is_finite() {
            noise_rms.max(1e-6)
        } else {
            0.0
        };
        let snr_ratio = if noise_rms > 0.0 {
            (amplitude_fit / noise_rms).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let phase_raw = best_phase;
        let phase = if let Some(prev) = self.last_phases.get(&band.kind) {
            let delta = wrap_phase_diff(phase_raw, *prev);
            wrap_phase(prev + delta * self.phase_smoothing)
        } else {
            wrap_phase(phase_raw)
        };
        let amplitude = best_amplitude;
        let cycles = if band.center_period() > 0.0 {
            span_secs / band.center_period()
        } else {
            0.0
        };
        let coverage = (span_secs / window_secs).clamp(0.0, 1.0);
        let quality_avg = (weight_total / samples.len() as f64).clamp(0.0, 1.0);
        let drift_ratio = (slope.abs() * band.center_period() / rms).clamp(0.0, 10.0);
        let stability_factor = 1.0 / (1.0 + drift_ratio);
        let confidence = (amplitude * coverage * quality_avg * stability_factor).clamp(0.0, 1.0);
        if confidence < self.min_confidence {
            return None;
        }
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
            "window_span_secs".to_string(),
            serde_json::Value::from(span_secs),
        );
        attributes.insert(
            "coverage_ratio".to_string(),
            serde_json::Value::from(coverage),
        );
        attributes.insert(
            "cycles_observed".to_string(),
            serde_json::Value::from(cycles),
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
        attributes.insert(
            "variance".to_string(),
            serde_json::Value::from(variance),
        );
        attributes.insert("rms".to_string(), serde_json::Value::from(rms));
        attributes.insert(
            "oscillation_amplitude".to_string(),
            serde_json::Value::from(amplitude_fit),
        );
        attributes.insert(
            "noise_rms".to_string(),
            serde_json::Value::from(noise_rms),
        );
        attributes.insert(
            "snr_ratio".to_string(),
            serde_json::Value::from(snr_ratio),
        );
        attributes.insert(
            "sample_dt_mean".to_string(),
            serde_json::Value::from(sample_dt_mean_resampled),
        );
        attributes.insert(
            "sample_dt_std".to_string(),
            serde_json::Value::from(sample_dt_std_resampled),
        );
        attributes.insert(
            "sample_dt_mean_raw".to_string(),
            serde_json::Value::from(sample_dt_mean),
        );
        attributes.insert(
            "sample_dt_std_raw".to_string(),
            serde_json::Value::from(sample_dt_std),
        );
        attributes.insert(
            "jitter_ratio_raw".to_string(),
            serde_json::Value::from(jitter_ratio),
        );
        attributes.insert(
            "resampled".to_string(),
            serde_json::Value::from(resampled),
        );
        attributes.insert(
            "quality_avg".to_string(),
            serde_json::Value::from(quality_avg),
        );
        attributes.insert(
            "quality_min".to_string(),
            serde_json::Value::from(min_quality),
        );
        attributes.insert(
            "quality_max".to_string(),
            serde_json::Value::from(max_quality),
        );
        attributes.insert("slope".to_string(), serde_json::Value::from(slope));
        attributes.insert(
            "stability_factor".to_string(),
            serde_json::Value::from(stability_factor),
        );
        attributes.insert(
            "confidence".to_string(),
            serde_json::Value::from(confidence),
        );
        attributes.insert(
            "phase_raw".to_string(),
            serde_json::Value::from(phase_raw),
        );
        if let Some(prev_phase) = self.last_phases.get(&band.kind) {
            if let Some(prev_time) = self.last_phase_times.get(&band.kind) {
                let dt = (timestamp.unix - prev_time.unix).abs() as f64;
                if dt > 0.0 {
                    let drift = wrap_phase_diff(phase, *prev_phase) / dt;
                    attributes.insert(
                        "phase_drift_rad_s".to_string(),
                        serde_json::Value::from(drift),
                    );
                }
            }
        }
        self.last_phases.insert(band.kind, phase);
        self.last_phase_times.insert(band.kind, timestamp);
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

fn sample_dt_metrics(samples: &[SignalSample]) -> (f64, f64, f64) {
    let mut stats = RunningStats::default();
    let mut last_ts: Option<f64> = None;
    for sample in samples {
        let ts = sample.timestamp.unix as f64;
        if let Some(prev) = last_ts {
            let dt = (ts - prev).abs();
            if dt.is_finite() && dt > 0.0 {
                stats.update(dt);
            }
        }
        last_ts = Some(ts);
    }
    let mean = stats.mean().unwrap_or(0.0);
    let std = stats.stddev(0).unwrap_or(0.0);
    let jitter_ratio = if mean > 0.0 {
        (std / mean).clamp(0.0, 10.0)
    } else {
        0.0
    };
    (mean, std, jitter_ratio)
}

fn resample_uniform(
    samples: &[SignalSample],
    band: &UltradianBand,
    max_points: usize,
) -> Option<Vec<SignalSample>> {
    if samples.len() < 2 {
        return None;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by_key(|sample| sample.timestamp.unix);
    let (dt_mean, _, _) = sample_dt_metrics(&ordered);
    if dt_mean <= 0.0 {
        return None;
    }
    let min_dt = (band.min_period_secs / 24.0).max(1.0);
    let max_dt = (band.max_period_secs / 8.0).max(min_dt);
    let mut target_dt = dt_mean.clamp(min_dt, max_dt);
    let start = ordered.first()?.timestamp.unix as f64;
    let end = ordered.last()?.timestamp.unix as f64;
    let span = (end - start).max(0.0);
    if span <= 0.0 {
        return None;
    }
    let mut points = (span / target_dt).ceil() as usize + 1;
    let max_points = max_points.max(64);
    if points > max_points {
        target_dt = span / max_points as f64;
        points = max_points;
    }
    let mut resampled = Vec::with_capacity(points);
    let mut idx = 0usize;
    for step in 0..points {
        let t = start + (step as f64 * target_dt);
        while idx + 1 < ordered.len() && ordered[idx + 1].timestamp.unix as f64 <= t {
            idx += 1;
        }
        let a = &ordered[idx];
        if (a.timestamp.unix as f64 - t).abs() < 1e-6 {
            resampled.push(*a);
            continue;
        }
        if idx + 1 >= ordered.len() {
            break;
        }
        let b = &ordered[idx + 1];
        let ta = a.timestamp.unix as f64;
        let tb = b.timestamp.unix as f64;
        let denom = (tb - ta).max(1e-6);
        let w = ((t - ta) / denom).clamp(0.0, 1.0);
        let value = a.value + w * (b.value - a.value);
        let quality = a.quality + w * (b.quality - a.quality);
        resampled.push(SignalSample {
            timestamp: Timestamp { unix: t.round() as i64 },
            value,
            quality: quality.clamp(0.0, 1.0),
        });
    }
    Some(resampled)
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

fn wrap_phase(mut phase: f64) -> f64 {
    while phase > PI {
        phase -= 2.0 * PI;
    }
    while phase < -PI {
        phase += 2.0 * PI;
    }
    phase
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

    #[test]
    fn extractor_requires_min_cycles() {
        let mut extractor = UltradianLayerExtractor::new();
        let start = 3_000_000_i64;
        for idx in 0..40 {
            let t = start + (idx * 10) as i64;
            let value = (idx as f64 * 0.1).sin();
            extractor.push_sample(SignalSample {
                timestamp: Timestamp { unix: t },
                value,
                quality: 1.0,
            });
        }
        let layers = extractor.extract_layers();
        assert!(layers.is_empty());
    }

    #[test]
    fn extractor_tracks_phase_drift() {
        let mut extractor = UltradianLayerExtractor::new();
        let period_secs = 30.0 * 60.0;
        let omega = 2.0 * PI / period_secs;
        let start = 4_000_000_i64;
        for idx in 0..80 {
            let t = start + (idx * 60) as i64;
            let value = (omega * t as f64).sin();
            extractor.push_sample(SignalSample {
                timestamp: Timestamp { unix: t },
                value,
                quality: 1.0,
            });
        }
        let first_layers = extractor.extract_layers();
        assert!(first_layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::UltradianMicroArousal)));
        for idx in 80..90 {
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
            .find(|layer| matches!(layer.kind, LayerKind::UltradianMicroArousal))
            .expect("micro layer");
        assert!(micro.attributes.contains_key("phase_drift_rad_s"));
    }
}
