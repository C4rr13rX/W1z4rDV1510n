use crate::schema::Timestamp;
use crate::streaming::schema::{LayerKind, LayerState};
use crate::math_toolbox::{self, RunningStats};
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

#[derive(Debug, Clone)]
pub struct SubUltradianBand {
    pub name: &'static str,
    pub min_period_secs: f64,
    pub max_period_secs: f64,
}

impl SubUltradianBand {
    pub fn center_period(&self) -> f64 {
        0.5 * (self.min_period_secs + self.max_period_secs)
    }
}

pub struct UltradianLayerExtractor {
    bands: Vec<UltradianBand>,
    sub_bands: Vec<SubUltradianBand>,
    series: SignalSeries,
    min_samples: usize,
    candidate_count: usize,
    min_quality: f64,
    min_cycles: usize,
    min_confidence: f64,
    phase_smoothing: f64,
    resample_jitter_threshold: f64,
    resample_max_points: usize,
    sub_band_candidate_count: usize,
    sub_band_segments: usize,
    sub_band_min_cycles: usize,
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
        let sub_bands = vec![
            SubUltradianBand {
                name: "fast",
                min_period_secs: 30.0,
                max_period_secs: 120.0,
            },
            SubUltradianBand {
                name: "mid",
                min_period_secs: 120.0,
                max_period_secs: 600.0,
            },
            SubUltradianBand {
                name: "slow",
                min_period_secs: 600.0,
                max_period_secs: 1200.0,
            },
        ];
        Self::with_bands(
            bands,
            sub_bands,
            max_age_secs,
            32,
            5,
            0.1,
            2,
            0.1,
            0.6,
            0.35,
            512,
            3,
            4,
            2,
        )
    }

    pub fn with_bands(
        bands: Vec<UltradianBand>,
        sub_bands: Vec<SubUltradianBand>,
        max_age_secs: f64,
        min_samples: usize,
        candidate_count: usize,
        min_quality: f64,
        min_cycles: usize,
        min_confidence: f64,
        phase_smoothing: f64,
        resample_jitter_threshold: f64,
        resample_max_points: usize,
        sub_band_candidate_count: usize,
        sub_band_segments: usize,
        sub_band_min_cycles: usize,
    ) -> Self {
        Self {
            bands,
            sub_bands,
            series: SignalSeries::new(max_age_secs),
            min_samples: min_samples.max(4),
            candidate_count: candidate_count.max(1),
            min_quality: min_quality.clamp(0.0, 1.0),
            min_cycles: min_cycles.max(1),
            min_confidence: min_confidence.clamp(0.0, 1.0),
            phase_smoothing: phase_smoothing.clamp(0.0, 1.0),
            resample_jitter_threshold: resample_jitter_threshold.clamp(0.0, 5.0),
            resample_max_points: resample_max_points.max(64),
            sub_band_candidate_count: sub_band_candidate_count.max(1),
            sub_band_segments: sub_band_segments.max(2),
            sub_band_min_cycles: sub_band_min_cycles.max(1),
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
        let sub_band_metrics = estimate_sub_bands(
            &self.sub_bands,
            &samples,
            min_ts,
            max_ts,
            mean,
            slope,
            t_mean,
            t_ref,
            weight_total,
            rms,
            best_period,
            phase_raw,
            self.sub_band_candidate_count,
            self.sub_band_segments,
            self.sub_band_min_cycles,
        );
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
        for (key, value) in sub_band_metrics {
            attributes.insert(key, value);
        }
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

#[derive(Debug, Clone, Copy)]
struct BandEstimate {
    period_secs: f64,
    amplitude_norm: f64,
    amplitude_raw: f64,
    snr: f64,
}

#[derive(Debug, Clone, Copy)]
struct PeriodogramMetrics {
    peak_period: f64,
    peak_power: f64,
    band_power: f64,
    entropy: f64,
}

#[derive(Debug, Clone, Copy)]
struct WaveletMetrics {
    energy_mean: f64,
    energy_std: f64,
    energy_peak: f64,
}

#[derive(Debug, Clone, Copy)]
struct EmdMetrics {
    imf_energy: f64,
    residual_energy: f64,
    envelope_mean: f64,
    envelope_peak: f64,
    zero_cross_hz: f64,
}

fn estimate_sub_bands(
    sub_bands: &[SubUltradianBand],
    samples: &[SignalSample],
    min_ts: f64,
    max_ts: f64,
    mean: f64,
    slope: f64,
    t_mean: f64,
    t_ref: f64,
    weight_total: f64,
    rms: f64,
    ultra_period: f64,
    ultra_phase: f64,
    candidate_count: usize,
    segments: usize,
    min_cycles: usize,
) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    let span = (max_ts - min_ts).max(0.0);
    if samples.is_empty() || span <= 0.0 {
        return out;
    }
    for band in sub_bands {
        if span < band.min_period_secs * min_cycles as f64 {
            continue;
        }
        let Some(estimate) = estimate_band(
            samples,
            band,
            mean,
            slope,
            t_mean,
            t_ref,
            weight_total,
            rms,
            candidate_count,
        ) else {
            continue;
        };
        let periodogram = periodogram_band_metrics(
            samples,
            band,
            mean,
            slope,
            t_mean,
            t_ref,
            weight_total,
            candidate_count.saturating_mul(2),
        );
        let multitaper = multitaper_band_metrics(
            samples,
            band,
            mean,
            slope,
            t_mean,
            t_ref,
            weight_total,
            min_ts,
            max_ts,
            candidate_count.saturating_mul(2),
        );
        let wavelet = wavelet_band_metrics(
            samples,
            estimate.period_secs,
            mean,
            slope,
            t_mean,
            min_ts,
            max_ts,
            segments,
        );
        let emd = emd_band_metrics(samples, estimate.period_secs, min_ts, max_ts);
        let pac = phase_amplitude_coupling(
            samples,
            min_ts,
            max_ts,
            estimate.period_secs,
            mean,
            slope,
            t_mean,
            t_ref,
            ultra_period,
            ultra_phase,
            segments,
        );
        let prefix = format!("sub_band_{}", band.name);
        out.push((
            format!("{prefix}_period_secs"),
            serde_json::Value::from(estimate.period_secs),
        ));
        out.push((
            format!("{prefix}_amplitude"),
            serde_json::Value::from(estimate.amplitude_norm),
        ));
        out.push((
            format!("{prefix}_amplitude_raw"),
            serde_json::Value::from(estimate.amplitude_raw),
        ));
        out.push((
            format!("{prefix}_snr"),
            serde_json::Value::from(estimate.snr),
        ));
        if let Some(metrics) = periodogram {
            out.push((
                format!("{prefix}_psd_peak_period_secs"),
                serde_json::Value::from(metrics.peak_period),
            ));
            out.push((
                format!("{prefix}_psd_peak_power"),
                serde_json::Value::from(metrics.peak_power),
            ));
            out.push((
                format!("{prefix}_psd_band_power"),
                serde_json::Value::from(metrics.band_power),
            ));
            out.push((
                format!("{prefix}_psd_entropy"),
                serde_json::Value::from(metrics.entropy),
            ));
        }
        if let Some(metrics) = multitaper {
            out.push((
                format!("{prefix}_psd_mt_peak_period_secs"),
                serde_json::Value::from(metrics.peak_period),
            ));
            out.push((
                format!("{prefix}_psd_mt_peak_power"),
                serde_json::Value::from(metrics.peak_power),
            ));
            out.push((
                format!("{prefix}_psd_mt_band_power"),
                serde_json::Value::from(metrics.band_power),
            ));
            out.push((
                format!("{prefix}_psd_mt_entropy"),
                serde_json::Value::from(metrics.entropy),
            ));
        }
        if let Some(metrics) = wavelet {
            out.push((
                format!("{prefix}_wavelet_energy_mean"),
                serde_json::Value::from(metrics.energy_mean),
            ));
            out.push((
                format!("{prefix}_wavelet_energy_std"),
                serde_json::Value::from(metrics.energy_std),
            ));
            out.push((
                format!("{prefix}_wavelet_energy_peak"),
                serde_json::Value::from(metrics.energy_peak),
            ));
        }
        if let Some(metrics) = emd {
            out.push((
                format!("{prefix}_emd_imf_energy"),
                serde_json::Value::from(metrics.imf_energy),
            ));
            out.push((
                format!("{prefix}_emd_residual_energy"),
                serde_json::Value::from(metrics.residual_energy),
            ));
            out.push((
                format!("{prefix}_emd_envelope_mean"),
                serde_json::Value::from(metrics.envelope_mean),
            ));
            out.push((
                format!("{prefix}_emd_envelope_peak"),
                serde_json::Value::from(metrics.envelope_peak),
            ));
            out.push((
                format!("{prefix}_emd_zero_cross_hz"),
                serde_json::Value::from(metrics.zero_cross_hz),
            ));
        }
        out.push((
            format!("{prefix}_pac"),
            serde_json::Value::from(pac),
        ));
    }
    out
}

fn estimate_band(
    samples: &[SignalSample],
    band: &SubUltradianBand,
    mean: f64,
    slope: f64,
    t_mean: f64,
    t_ref: f64,
    weight_total: f64,
    rms: f64,
    candidate_count: usize,
) -> Option<BandEstimate> {
    if weight_total <= 0.0 || rms <= 0.0 {
        return None;
    }
    let mut best_period = band.center_period();
    let mut best_phase = 0.0;
    let mut best_magnitude = 0.0;
    for period in band_candidate_periods(band, candidate_count) {
        let omega = 2.0 * PI / period.max(1.0);
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;
        for sample in samples {
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
        if magnitude > best_magnitude {
            best_magnitude = magnitude;
            best_period = period;
            best_phase = sum_sin.atan2(sum_cos);
        }
    }
    if best_magnitude <= 0.0 {
        return None;
    }
    let amplitude_raw = (2.0 * best_magnitude / weight_total).abs();
    let amplitude_norm = (best_magnitude / (weight_total * rms) * 2.0_f64.sqrt()).clamp(0.0, 1.0);
    let omega = 2.0 * PI / best_period.max(1.0);
    let mut noise_sum = 0.0;
    for sample in samples {
        let weight = sample.quality.clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }
        let t = sample.timestamp.unix as f64;
        let centered = (sample.value - mean) - slope * (t - t_mean);
        let t_offset = t - t_ref;
        let fitted = amplitude_raw * (omega * t_offset + best_phase).sin();
        let residual = centered - fitted;
        noise_sum += weight * residual * residual;
    }
    let noise_rms = if noise_sum.is_finite() && weight_total > 0.0 {
        (noise_sum / weight_total).sqrt().max(1e-6)
    } else {
        1e-6
    };
    let snr = (amplitude_raw / noise_rms).clamp(0.0, 100.0);
    Some(BandEstimate {
        period_secs: best_period,
        amplitude_norm,
        amplitude_raw,
        snr,
    })
}

fn periodogram_band_metrics(
    samples: &[SignalSample],
    band: &SubUltradianBand,
    mean: f64,
    slope: f64,
    t_mean: f64,
    t_ref: f64,
    weight_total: f64,
    candidate_count: usize,
) -> Option<PeriodogramMetrics> {
    if samples.len() < 2 || weight_total <= 0.0 {
        return None;
    }
    let periods = band_candidate_periods(band, candidate_count.max(2));
    if periods.is_empty() {
        return None;
    }
    let mut powers = Vec::with_capacity(periods.len());
    let mut peak_period = periods[0];
    let mut peak_power = 0.0;
    for period in periods {
        let omega = 2.0 * PI / period.max(1.0);
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;
        for sample in samples {
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
        let amplitude_raw = (2.0 * magnitude / weight_total).abs();
        let power = (amplitude_raw * amplitude_raw).max(0.0);
        if power > peak_power {
            peak_power = power;
            peak_period = period;
        }
        powers.push(power);
    }
    let band_power = math_toolbox::mean(&powers).unwrap_or(0.0);
    let entropy = math_toolbox::entropy(&powers);
    Some(PeriodogramMetrics {
        peak_period,
        peak_power,
        band_power,
        entropy,
    })
}

fn multitaper_band_metrics(
    samples: &[SignalSample],
    band: &SubUltradianBand,
    mean: f64,
    slope: f64,
    t_mean: f64,
    t_ref: f64,
    weight_total: f64,
    min_ts: f64,
    max_ts: f64,
    candidate_count: usize,
) -> Option<PeriodogramMetrics> {
    if samples.len() < 2 || weight_total <= 0.0 {
        return None;
    }
    let span = (max_ts - min_ts).max(0.0);
    if span <= 0.0 {
        return None;
    }
    let periods = band_candidate_periods(band, candidate_count.max(2));
    if periods.is_empty() {
        return None;
    }
    let mut powers = Vec::with_capacity(periods.len());
    let mut peak_period = periods[0];
    let mut peak_power = 0.0;
    for period in periods {
        let omega = 2.0 * PI / period.max(1.0);
        let mut taper_powers = Vec::new();
        for taper in 0..3 {
            let mut sum_sin = 0.0;
            let mut sum_cos = 0.0;
            let mut taper_weight_total = 0.0;
            for sample in samples {
                let weight = sample.quality.clamp(0.0, 1.0);
                if weight <= 0.0 {
                    continue;
                }
                let t = sample.timestamp.unix as f64;
                let centered = (sample.value - mean) - slope * (t - t_mean);
                let t_offset = t - t_ref;
                let x = ((t - min_ts) / span).clamp(0.0, 1.0);
                let taper_weight = taper_weight(taper, x);
                let w = weight * taper_weight;
                if w <= 0.0 {
                    continue;
                }
                sum_sin += w * centered * (omega * t_offset).sin();
                sum_cos += w * centered * (omega * t_offset).cos();
                taper_weight_total += w;
            }
            if taper_weight_total <= 0.0 {
                continue;
            }
            let magnitude = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
            let amplitude_raw = (2.0 * magnitude / taper_weight_total).abs();
            let power = (amplitude_raw * amplitude_raw).max(0.0);
            taper_powers.push(power);
        }
        if taper_powers.is_empty() {
            powers.push(0.0);
            continue;
        }
        let power = math_toolbox::mean(&taper_powers).unwrap_or(0.0);
        if power > peak_power {
            peak_power = power;
            peak_period = period;
        }
        powers.push(power);
    }
    let band_power = math_toolbox::mean(&powers).unwrap_or(0.0);
    let entropy = math_toolbox::entropy(&powers);
    Some(PeriodogramMetrics {
        peak_period,
        peak_power,
        band_power,
        entropy,
    })
}

fn taper_weight(taper: usize, x: f64) -> f64 {
    let theta = 2.0 * PI * x;
    match taper {
        0 => (0.5 - 0.5 * theta.cos()).clamp(0.0, 1.0),
        1 => (0.54 - 0.46 * theta.cos()).clamp(0.0, 1.0),
        _ => (0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos()).clamp(0.0, 1.0),
    }
}

fn wavelet_band_metrics(
    samples: &[SignalSample],
    period: f64,
    mean: f64,
    slope: f64,
    t_mean: f64,
    min_ts: f64,
    max_ts: f64,
    segments: usize,
) -> Option<WaveletMetrics> {
    if samples.len() < 2 || period <= 0.0 || segments < 2 {
        return None;
    }
    let span = (max_ts - min_ts).max(0.0);
    if span <= 0.0 {
        return None;
    }
    let omega = 2.0 * PI / period.max(1.0);
    let sigma = (period * 0.5).max(1.0);
    let segment_len = span / segments as f64;
    let mut stats = RunningStats::default();
    let mut peak = 0.0;
    for idx in 0..segments {
        let center = min_ts + (idx as f64 + 0.5) * segment_len;
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;
        let mut weight_total = 0.0;
        for sample in samples {
            let weight = sample.quality.clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            let t = sample.timestamp.unix as f64;
            let centered = (sample.value - mean) - slope * (t - t_mean);
            let dt = t - center;
            let gauss = (-dt * dt / (2.0 * sigma * sigma)).exp();
            let w = weight * gauss;
            if w <= 0.0 {
                continue;
            }
            sum_sin += w * centered * (omega * dt).sin();
            sum_cos += w * centered * (omega * dt).cos();
            weight_total += w;
        }
        if weight_total <= 0.0 {
            continue;
        }
        let magnitude = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
        let amplitude_raw = (2.0 * magnitude / weight_total).abs();
        let energy = (amplitude_raw * amplitude_raw).max(0.0);
        stats.update(energy);
        if energy > peak {
            peak = energy;
        }
    }
    if stats.count() == 0 {
        return None;
    }
    let energy_mean = stats.mean().unwrap_or(0.0);
    let energy_std = stats.stddev(0).unwrap_or(0.0);
    Some(WaveletMetrics {
        energy_mean,
        energy_std,
        energy_peak: peak,
    })
}

fn emd_band_metrics(
    samples: &[SignalSample],
    period: f64,
    min_ts: f64,
    max_ts: f64,
) -> Option<EmdMetrics> {
    if samples.len() < 3 || period <= 0.0 {
        return None;
    }
    let span = (max_ts - min_ts).max(0.0);
    if span <= 0.0 {
        return None;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by_key(|sample| sample.timestamp.unix);
    let (dt_mean, _, _) = sample_dt_metrics(&ordered);
    if dt_mean <= 0.0 {
        return None;
    }
    let window_len = (period / dt_mean).round() as usize;
    let window_len = window_len.clamp(3, ordered.len());
    let mut values = Vec::with_capacity(ordered.len());
    for sample in &ordered {
        values.push(sample.value);
    }
    let smooth = math_toolbox::moving_average(&values, window_len);
    if smooth.len() != values.len() {
        return None;
    }
    let mut imf = Vec::with_capacity(values.len());
    let mut residual_energy = 0.0;
    let mut imf_energy = 0.0;
    let mut envelope_mean = 0.0;
    let mut envelope_peak = 0.0;
    let mut zero_crossings = 0usize;
    let mut prev_imf: f64 = 0.0;
    for (idx, (value, base)) in values.iter().zip(smooth.iter()).enumerate() {
        let component = value - base;
        imf.push(component);
        imf_energy += component * component;
        residual_energy += base * base;
        let envelope = component.abs();
        envelope_mean += envelope;
        if envelope > envelope_peak {
            envelope_peak = envelope;
        }
        if idx > 0 && prev_imf.signum() != 0.0 && component.signum() != 0.0 {
            if prev_imf.signum() != component.signum() {
                zero_crossings += 1;
            }
        }
        prev_imf = component;
    }
    let count = imf.len() as f64;
    if count <= 0.0 {
        return None;
    }
    imf_energy /= count;
    residual_energy /= count;
    envelope_mean /= count;
    let zero_cross_hz = if span > 0.0 {
        zero_crossings as f64 / (2.0 * span)
    } else {
        0.0
    };
    Some(EmdMetrics {
        imf_energy,
        residual_energy,
        envelope_mean,
        envelope_peak,
        zero_cross_hz,
    })
}

fn band_candidate_periods(band: &SubUltradianBand, count: usize) -> Vec<f64> {
    if count <= 1 || band.min_period_secs <= 0.0 || band.max_period_secs <= 0.0 {
        return vec![band.center_period()];
    }
    let min_log = band.min_period_secs.ln();
    let max_log = band.max_period_secs.ln();
    let steps = (count - 1) as f64;
    let mut periods = Vec::with_capacity(count);
    for idx in 0..count {
        let frac = if steps > 0.0 { idx as f64 / steps } else { 0.0 };
        let period = (min_log + frac * (max_log - min_log)).exp();
        periods.push(period);
    }
    periods
}

fn phase_amplitude_coupling(
    samples: &[SignalSample],
    min_ts: f64,
    max_ts: f64,
    band_period: f64,
    mean: f64,
    slope: f64,
    t_mean: f64,
    t_ref: f64,
    ultra_period: f64,
    ultra_phase: f64,
    segments: usize,
) -> f64 {
    if samples.is_empty() || segments < 2 || band_period <= 0.0 || ultra_period <= 0.0 {
        return 0.0;
    }
    let span = (max_ts - min_ts).max(0.0);
    if span <= 0.0 {
        return 0.0;
    }
    let segment_len = span / segments as f64;
    let mut sum_amp = 0.0;
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    for idx in 0..segments {
        let start = min_ts + idx as f64 * segment_len;
        let end = if idx + 1 == segments {
            max_ts
        } else {
            start + segment_len
        };
        let mut seg_samples = Vec::new();
        for sample in samples {
            let ts = sample.timestamp.unix as f64;
            if ts >= start && ts <= end {
                seg_samples.push(*sample);
            }
        }
        let Some(amplitude) =
            segment_amplitude_raw(&seg_samples, band_period, mean, slope, t_mean, t_ref)
        else {
            continue;
        };
        if amplitude <= 0.0 {
            continue;
        }
        let mid = 0.5 * (start + end);
        let phase = phase_at_time(ultra_phase, ultra_period, t_ref, mid);
        sum_amp += amplitude;
        sum_cos += amplitude * phase.cos();
        sum_sin += amplitude * phase.sin();
    }
    if sum_amp <= 0.0 {
        return 0.0;
    }
    ((sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / sum_amp).clamp(0.0, 1.0)
}

fn segment_amplitude_raw(
    samples: &[SignalSample],
    period: f64,
    mean: f64,
    slope: f64,
    t_mean: f64,
    t_ref: f64,
) -> Option<f64> {
    if samples.len() < 2 || period <= 0.0 {
        return None;
    }
    let omega = 2.0 * PI / period.max(1.0);
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut weight_total = 0.0;
    for sample in samples {
        let weight = sample.quality.clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }
        let t = sample.timestamp.unix as f64;
        let centered = (sample.value - mean) - slope * (t - t_mean);
        let t_offset = t - t_ref;
        sum_sin += weight * centered * (omega * t_offset).sin();
        sum_cos += weight * centered * (omega * t_offset).cos();
        weight_total += weight;
    }
    if weight_total <= 0.0 {
        return None;
    }
    let magnitude = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
    Some((2.0 * magnitude / weight_total).abs())
}

fn phase_at_time(phase_ref: f64, period: f64, t_ref: f64, t: f64) -> f64 {
    if period <= 0.0 {
        return phase_ref;
    }
    let omega = 2.0 * PI / period.max(1.0);
    wrap_phase(phase_ref + omega * (t - t_ref))
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
