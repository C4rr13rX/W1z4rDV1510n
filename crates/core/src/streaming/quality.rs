use crate::config::StreamingQualityConfig;
use crate::math_toolbox::RunningStats;
use crate::schema::Timestamp;
use crate::streaming::schema::{StreamSource, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceQuality {
    pub source: StreamSource,
    pub sample_dt_mean: f64,
    pub sample_dt_std: f64,
    pub jitter_ratio: f64,
    pub snr_ratio: f64,
    pub snr_score: f64,
    pub token_confidence_mean: f64,
    pub layer_amplitude_mean: f64,
    pub layer_coherence_mean: f64,
    pub missing_ratio: f64,
    pub drift_ratio: f64,
    pub base_confidence: f64,
    pub quality_score: f64,
    pub quality_ema: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub timestamp: Timestamp,
    pub sources: Vec<SourceQuality>,
    pub overall_quality: f64,
}

#[derive(Debug, Clone)]
struct QualityState {
    last_timestamp: Option<Timestamp>,
    dt_stats: RunningStats,
    quality_ema: f64,
}

pub struct StreamingQualityRuntime {
    config: StreamingQualityConfig,
    states: HashMap<StreamSource, QualityState>,
    latest: HashMap<StreamSource, SourceQuality>,
}

impl StreamingQualityRuntime {
    pub fn new(config: StreamingQualityConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            latest: HashMap::new(),
        }
    }

    pub fn update(&mut self, batch: &mut TokenBatch) -> Option<SourceQuality> {
        if !self.config.enabled {
            return None;
        }
        let source = infer_batch_source(batch)?;
        let state = self.states.entry(source).or_insert_with(|| QualityState {
            last_timestamp: None,
            dt_stats: RunningStats::default(),
            quality_ema: 1.0,
        });
        if let Some(prev) = state.last_timestamp {
            let dt = (batch.timestamp.unix - prev.unix).abs() as f64;
            if dt.is_finite() && dt > 0.0 {
                state.dt_stats.update(dt);
            }
        }
        state.last_timestamp = Some(batch.timestamp);

        let token_confidence_mean = mean_token_confidence(batch, source);
        let (layer_amplitude_mean, layer_coherence_mean, drift_ratio) =
            layer_metrics(batch, &self.config);
        let missing_ratio = missing_ratio(batch).clamp(0.0, 1.0);
        let snr_ratio = mean_snr_ratio(batch).unwrap_or(self.config.snr_norm);
        let snr_ratio = snr_ratio.clamp(0.0, self.config.snr_cap.max(1.0));
        let snr_score = if self.config.snr_norm > 0.0 {
            (snr_ratio / self.config.snr_norm).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let base_confidence = base_confidence(
            token_confidence_mean,
            layer_coherence_mean,
            batch.source_confidence.get(&source).copied(),
        );

        let (sample_dt_mean, sample_dt_std) = dt_stats(state);
        let jitter_ratio = if sample_dt_mean > 0.0 {
            (sample_dt_std / sample_dt_mean).clamp(0.0, 10.0)
        } else {
            0.0
        };
        let jitter_score = 1.0 / (1.0 + self.config.jitter_weight * jitter_ratio);
        let missing_score =
            (1.0 - self.config.missing_weight * missing_ratio).clamp(0.0, 1.0);
        let drift_score = (1.0 - self.config.drift_weight * drift_ratio).clamp(0.0, 1.0);

        let snr_weight = self.config.snr_weight.clamp(0.0, 1.0);
        let snr_factor = (1.0 - snr_weight) + snr_weight * snr_score;
        let mut quality_score =
            base_confidence * jitter_score * missing_score * drift_score * snr_factor;
        if quality_score < self.config.min_quality {
            quality_score = 0.0;
        }
        let alpha = self.config.quality_ema_alpha.clamp(0.0, 1.0);
        state.quality_ema = alpha * quality_score + (1.0 - alpha) * state.quality_ema;

        let quality = SourceQuality {
            source,
            sample_dt_mean,
            sample_dt_std,
            jitter_ratio,
            snr_ratio,
            snr_score,
            token_confidence_mean,
            layer_amplitude_mean,
            layer_coherence_mean,
            missing_ratio,
            drift_ratio,
            base_confidence,
            quality_score,
            quality_ema: state.quality_ema,
        };
        self.latest.insert(source, quality.clone());
        batch
            .source_confidence
            .insert(source, quality_score.clamp(0.0, 1.0));
        Some(quality)
    }

    pub fn latest(&self, source: StreamSource) -> Option<&SourceQuality> {
        self.latest.get(&source)
    }

    pub fn report_for(
        &self,
        timestamp: Timestamp,
        sources: impl IntoIterator<Item = StreamSource>,
    ) -> Option<QualityReport> {
        let mut report_sources = Vec::new();
        for source in sources {
            if let Some(entry) = self.latest.get(&source) {
                report_sources.push(entry.clone());
            }
        }
        if report_sources.is_empty() {
            return None;
        }
        let overall = report_sources
            .iter()
            .map(|entry| entry.quality_score)
            .sum::<f64>()
            / report_sources.len() as f64;
        Some(QualityReport {
            timestamp,
            sources: report_sources,
            overall_quality: overall.clamp(0.0, 1.0),
        })
    }
}

fn infer_batch_source(batch: &TokenBatch) -> Option<StreamSource> {
    if batch.source_confidence.len() == 1 {
        return batch.source_confidence.keys().next().copied();
    }
    let mut counts: HashMap<StreamSource, usize> = HashMap::new();
    for token in &batch.tokens {
        if let Some(source) = token.source {
            *counts.entry(source).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(source, _)| source)
}

fn mean_token_confidence(batch: &TokenBatch, source: StreamSource) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for token in &batch.tokens {
        if token.source.unwrap_or(source) != source {
            continue;
        }
        sum += token.confidence.clamp(0.0, 1.0);
        count += 1.0;
    }
    if count > 0.0 {
        (sum / count).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn layer_metrics(batch: &TokenBatch, config: &StreamingQualityConfig) -> (f64, f64, f64) {
    let mut amp_sum = 0.0;
    let mut coh_sum = 0.0;
    let mut drift_sum = 0.0;
    let mut count = 0.0;
    for layer in &batch.layers {
        amp_sum += layer.amplitude.clamp(0.0, 1.0);
        coh_sum += layer.coherence.clamp(0.0, 1.0);
        drift_sum += drift_from_attrs(&layer.attributes, config);
        count += 1.0;
    }
    if count <= 0.0 {
        return (1.0, 1.0, 0.0);
    }
    (
        (amp_sum / count).clamp(0.0, 1.0),
        (coh_sum / count).clamp(0.0, 1.0),
        (drift_sum / count).clamp(0.0, 1.0),
    )
}

fn drift_from_attrs(attrs: &HashMap<String, Value>, config: &StreamingQualityConfig) -> f64 {
    if let Some(val) = attrs.get("stability_factor").and_then(|v| v.as_f64()) {
        return (1.0 - val.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    }
    if let Some(val) = attrs.get("phase_drift_rad_s").and_then(|v| v.as_f64()) {
        let norm = config.drift_norm_rad_s.max(1e-6);
        return (val.abs() / norm).clamp(0.0, 1.0);
    }
    if let Some(val) = attrs.get("slope").and_then(|v| v.as_f64()) {
        let norm = config.drift_norm_slope.max(1e-6);
        return (val.abs() / norm).clamp(0.0, 1.0);
    }
    0.0
}

fn missing_ratio(batch: &TokenBatch) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for token in &batch.tokens {
        if let Some(val) = missing_from_attrs(&token.attributes) {
            sum += val;
            count += 1.0;
        }
    }
    for layer in &batch.layers {
        if let Some(val) = missing_from_attrs(&layer.attributes) {
            sum += val;
            count += 1.0;
        }
    }
    if count > 0.0 {
        (sum / count).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn mean_snr_ratio(batch: &TokenBatch) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0.0;
    for token in &batch.tokens {
        if let Some(val) = snr_from_attrs(&token.attributes) {
            sum += val;
            count += 1.0;
        }
    }
    for layer in &batch.layers {
        if let Some(val) = snr_from_attrs(&layer.attributes) {
            sum += val;
            count += 1.0;
        }
    }
    if count > 0.0 {
        Some((sum / count).max(0.0))
    } else {
        None
    }
}

fn snr_from_attrs(attrs: &HashMap<String, Value>) -> Option<f64> {
    for key in ["snr_ratio", "snr_proxy", "signal_snr", "snr"] {
        if let Some(val) = attrs.get(key).and_then(|v| v.as_f64()) {
            return Some(val.max(0.0));
        }
    }
    None
}

fn missing_from_attrs(attrs: &HashMap<String, Value>) -> Option<f64> {
    if let Some(val) = attrs.get("missing_ratio").and_then(|v| v.as_f64()) {
        return Some(val.clamp(0.0, 1.0));
    }
    if let Some(val) = attrs.get("quality_avg").and_then(|v| v.as_f64()) {
        return Some((1.0 - val.clamp(0.0, 1.0)).clamp(0.0, 1.0));
    }
    if let Some(val) = attrs.get("signal_quality").and_then(|v| v.as_f64()) {
        return Some((1.0 - val.clamp(0.0, 1.0)).clamp(0.0, 1.0));
    }
    if let Some(val) = attrs.get("keypoint_confidence").and_then(|v| v.as_f64()) {
        return Some((1.0 - val.clamp(0.0, 1.0)).clamp(0.0, 1.0));
    }
    None
}

fn base_confidence(
    token_confidence: f64,
    layer_coherence: f64,
    source_confidence: Option<f64>,
) -> f64 {
    let mut values = vec![token_confidence, layer_coherence];
    if let Some(conf) = source_confidence {
        values.push(conf);
    }
    let sum: f64 = values.iter().sum();
    if values.is_empty() {
        1.0
    } else {
        (sum / values.len() as f64).clamp(0.0, 1.0)
    }
}

fn dt_stats(state: &QualityState) -> (f64, f64) {
    let mean = state.dt_stats.mean().unwrap_or(0.0);
    let std = state.dt_stats.stddev(0).unwrap_or(0.0);
    (mean.max(0.0), std.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState};

    fn sample_batch(ts: i64, missing_ratio: f64) -> TokenBatch {
        TokenBatch {
            timestamp: Timestamp { unix: ts },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: ts },
                duration_secs: 1.0,
                confidence: 0.9,
                attributes: HashMap::from([(
                    "missing_ratio".to_string(),
                    Value::from(missing_ratio),
                )]),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: ts },
                phase: 0.1,
                amplitude: 0.7,
                coherence: 0.8,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::from([(StreamSource::PeopleVideo, 1.0)]),
        }
    }

    #[test]
    fn quality_runtime_scales_confidence() {
        let config = StreamingQualityConfig::default();
        let mut runtime = StreamingQualityRuntime::new(config);
        let mut batch = sample_batch(10, 0.5);
        runtime.update(&mut batch);
        let quality = runtime
            .latest(StreamSource::PeopleVideo)
            .expect("quality");
        assert!(quality.quality_score < 1.0);
        assert!(batch
            .source_confidence
            .get(&StreamSource::PeopleVideo)
            .copied()
            .unwrap_or(1.0)
            <= 1.0);
    }

    #[test]
    fn quality_report_returns_sources() {
        let mut runtime = StreamingQualityRuntime::new(StreamingQualityConfig::default());
        let mut batch = sample_batch(10, 0.1);
        runtime.update(&mut batch);
        let report = runtime
            .report_for(Timestamp { unix: 12 }, [StreamSource::PeopleVideo])
            .expect("report");
        assert_eq!(report.sources.len(), 1);
    }
}
