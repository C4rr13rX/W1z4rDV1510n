use crate::config::OnlinePlasticityConfig;
use crate::plasticity::{OnlinePlasticity, OutcomeSignal, PlasticityDecision};
use crate::schema::Timestamp;
use crate::streaming::hypergraph::DomainKind;
use crate::streaming::physiology_runtime::PhysiologyReport;
use crate::streaming::schema::{EventKind, TokenBatch};
use crate::streaming::temporal::{DirichletPosterior, TemporalInferenceReport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityUpdate {
    pub bucket: String,
    pub surprise: f64,
    pub update_weight: f64,
    pub teacher_ema: f64,
    pub rollback: bool,
    pub queued_replay: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityReport {
    pub timestamp: Timestamp,
    pub horizon_secs: f64,
    pub calibration_score: f64,
    pub drift_score: f64,
    pub surprise_score: f64,
    pub updates: Vec<PlasticityUpdate>,
    pub reservoir_buckets: usize,
}

struct TeacherState {
    ema_score: f64,
    last_score: f64,
}

struct HorizonManager {
    current_secs: f64,
    ema_score: f64,
    improvement_streak: usize,
}

impl HorizonManager {
    fn new(config: &OnlinePlasticityConfig) -> Self {
        Self {
            current_secs: config.horizon_initial_secs.max(1.0),
            ema_score: 0.0,
            improvement_streak: 0,
        }
    }

    fn update(&mut self, score: f64, config: &OnlinePlasticityConfig) -> f64 {
        let alpha = config.ema_teacher_alpha.clamp(0.0, 1.0);
        let prev = self.ema_score;
        self.ema_score = alpha * self.ema_score + (1.0 - alpha) * score;
        if self.ema_score > prev {
            self.improvement_streak = self.improvement_streak.saturating_add(1);
        } else {
            self.improvement_streak = 0;
        }
        if self.improvement_streak >= config.horizon_improvement_steps
            && self.current_secs < config.horizon_max_secs
        {
            let next = self.current_secs * config.horizon_growth_factor.max(1.0);
            self.current_secs = next.min(config.horizon_max_secs);
            self.improvement_streak = 0;
        }
        self.current_secs
    }
}

pub struct StreamingPlasticityRuntime {
    config: OnlinePlasticityConfig,
    plasticity: OnlinePlasticity,
    teachers: HashMap<String, TeacherState>,
    horizon: HorizonManager,
}

impl StreamingPlasticityRuntime {
    pub fn new(config: OnlinePlasticityConfig) -> Self {
        Self {
            horizon: HorizonManager::new(&config),
            plasticity: OnlinePlasticity::new(config.clone()),
            config,
            teachers: HashMap::new(),
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        report: &TemporalInferenceReport,
        physiology: Option<&PhysiologyReport>,
    ) -> Option<PlasticityReport> {
        if !self.config.enabled {
            return None;
        }
        let physiology_boost = physiology.map(|report| (report.overall_index / 3.0).clamp(0.0, 1.0));
        let metrics = collect_domain_metrics(report, physiology_boost);
        if metrics.is_empty() {
            return None;
        }
        let mut updates = Vec::new();
        let mut cal_sum = 0.0;
        let mut drift_sum = 0.0;
        let mut surprise_sum = 0.0;
        let mut domain_count = 0.0;
        for (domain, metric) in &metrics {
            domain_count += 1.0;
            cal_sum += metric.calibration;
            drift_sum += metric.drift;
            surprise_sum += metric.surprise;
            let context = context_for_domain(*domain, batch);
            let bucket = format!("{}|{}", domain_label(*domain), context);
            let teacher_ema = self.update_teacher(&bucket, metric.calibration);

            let (mut reason, queued_replay, mut update_weight) = if metric.drift
                > self.config.drift_threshold
            {
                ("drift_protection".to_string(), false, 0.0)
            } else {
                let signal = OutcomeSignal {
                    domain: domain_label(*domain).to_string(),
                    context,
                    surprise: metric.surprise,
                    timestamp: report.timestamp,
                };
                let decision = self.plasticity.observe_outcome(signal);
                match decision {
                    PlasticityDecision::NoUpdate { reason: why } => (why, false, 0.0),
                    PlasticityDecision::QueueReplay { bucket: _ } => {
                        let update = metric.surprise.min(self.config.trust_region.max(0.0));
                        ("replay_queued".to_string(), true, update)
                    }
                }
            };
            let mut rollback = false;

            if metric.calibration + self.config.rollback_threshold < teacher_ema {
                rollback = true;
                update_weight = 0.0;
                reason = "calibration_regressed".to_string();
            }

            updates.push(PlasticityUpdate {
                bucket,
                surprise: metric.surprise,
                update_weight,
                teacher_ema,
                rollback,
                queued_replay,
                reason,
            });
        }

        let calibration_score = if domain_count > 0.0 {
            (cal_sum / domain_count).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let drift_score = if domain_count > 0.0 {
            (drift_sum / domain_count).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let surprise_score = if domain_count > 0.0 {
            (surprise_sum / domain_count).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let horizon_secs = self.horizon.update(calibration_score, &self.config);

        Some(PlasticityReport {
            timestamp: report.timestamp,
            horizon_secs,
            calibration_score,
            drift_score,
            surprise_score,
            updates,
            reservoir_buckets: self.plasticity.bucket_count(),
        })
    }

    fn update_teacher(&mut self, bucket: &str, calibration: f64) -> f64 {
        let entry = self.teachers.entry(bucket.to_string()).or_insert(TeacherState {
            ema_score: calibration,
            last_score: calibration,
        });
        let alpha = self.config.ema_teacher_alpha.clamp(0.0, 1.0);
        let next = alpha * entry.ema_score + (1.0 - alpha) * calibration;
        entry.last_score = calibration;
        entry.ema_score = next;
        next
    }
}

#[derive(Clone, Copy)]
struct DomainMetrics {
    drift: f64,
    surprise: f64,
    calibration: f64,
}

fn collect_domain_metrics(
    report: &TemporalInferenceReport,
    physiology_boost: Option<f64>,
) -> HashMap<DomainKind, DomainMetrics> {
    let mut metrics = HashMap::new();
    for domain in [DomainKind::People, DomainKind::Crowd, DomainKind::Topics] {
        let drift = layer_drift(report, domain);
        let uncertainty = domain_uncertainty(report, domain);
        let surge = domain_surge(report, domain);
        let mut surprise = ((drift + uncertainty + surge) / 3.0).clamp(0.0, 1.0);
        let mut calibration = (1.0 - (drift + uncertainty) * 0.5).clamp(0.0, 1.0);
        if let (DomainKind::People, Some(boost)) = (domain, physiology_boost) {
            surprise = ((surprise + boost) * 0.5).clamp(0.0, 1.0);
            calibration = (calibration * (1.0 - boost)).clamp(0.0, 1.0);
        }
        metrics.insert(
            domain,
            DomainMetrics {
                drift,
                surprise,
                calibration,
            },
        );
    }
    metrics
}

fn layer_drift(report: &TemporalInferenceReport, domain: DomainKind) -> f64 {
    let mut drift_sum = 0.0;
    let mut count = 0.0;
    for pred in &report.layer_predictions {
        if pred.domain != domain {
            continue;
        }
        let phase = (pred.drift_phase.abs() / PI).clamp(0.0, 1.0);
        let amp = pred.drift_amplitude.abs().clamp(0.0, 1.0);
        let coh = pred.drift_coherence.abs().clamp(0.0, 1.0);
        drift_sum += (phase + amp + coh) / 3.0;
        count += 1.0;
    }
    if count > 0.0 {
        (drift_sum / count).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn domain_uncertainty(report: &TemporalInferenceReport, domain: DomainKind) -> f64 {
    let label = format!("domain_regime::{:?}", domain);
    let mut sum = 0.0;
    let mut count = 0.0;
    for posterior in &report.evidential {
        if posterior.label != label {
            continue;
        }
        sum += dirichlet_entropy(posterior);
        count += 1.0;
    }
    if count > 0.0 {
        (sum / count).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn domain_surge(report: &TemporalInferenceReport, domain: DomainKind) -> f64 {
    let mut intensity_sum = 0.0;
    let mut base_sum = 0.0;
    for intensity in &report.event_intensities {
        if domain_for_event(intensity.kind) != domain {
            continue;
        }
        intensity_sum += intensity.intensity;
        base_sum += intensity.base_rate;
    }
    if intensity_sum <= base_sum {
        return 0.0;
    }
    ((intensity_sum - base_sum) / base_sum.max(1e-6)).clamp(0.0, 1.0)
}

fn dirichlet_entropy(posterior: &DirichletPosterior) -> f64 {
    if posterior.alpha.is_empty() {
        return 0.0;
    }
    let sum = posterior.alpha.iter().sum::<f64>();
    let k = posterior.alpha.len() as f64;
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

fn context_for_domain(domain: DomainKind, batch: &TokenBatch) -> String {
    match domain {
        DomainKind::People => attribute_context(batch, &["entity_id", "person_id"]).unwrap_or_else(|| "people".to_string()),
        DomainKind::Crowd => attribute_context(batch, &["zone_id", "sensor_id", "segment_id"]).unwrap_or_else(|| "crowd".to_string()),
        DomainKind::Topics => attribute_context(batch, &["topic", "topic_id"]).unwrap_or_else(|| "topics".to_string()),
        DomainKind::Unknown => "unknown".to_string(),
    }
}

fn attribute_context(batch: &TokenBatch, keys: &[&str]) -> Option<String> {
    for token in &batch.tokens {
        for key in keys {
            if let Some(val) = token.attributes.get(*key).and_then(|v| v.as_str()) {
                if !val.trim().is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    for layer in &batch.layers {
        for key in keys {
            if let Some(val) = layer.attributes.get(*key).and_then(|v| v.as_str()) {
                if !val.trim().is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}

fn domain_label(domain: DomainKind) -> &'static str {
    match domain {
        DomainKind::People => "people",
        DomainKind::Crowd => "crowd",
        DomainKind::Topics => "topics",
        DomainKind::Unknown => "unknown",
    }
}

fn domain_for_event(kind: EventKind) -> DomainKind {
    match kind {
        EventKind::BehavioralAtom | EventKind::BehavioralToken => DomainKind::People,
        EventKind::CrowdToken | EventKind::TrafficToken => DomainKind::Crowd,
        EventKind::TopicEventToken => DomainKind::Topics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventToken, LayerKind, LayerState, StreamSource};

    #[test]
    fn plasticity_runtime_emits_report() {
        let mut config = OnlinePlasticityConfig::default();
        config.enabled = true;
        config.surprise_threshold = 0.1;
        let mut runtime = StreamingPlasticityRuntime::new(config);
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 100 },
                duration_secs: 1.0,
                confidence: 1.0,
                attributes: HashMap::from([("entity_id".to_string(), "e1".into())]),
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
        let report = TemporalInferenceReport {
            timestamp: Timestamp { unix: 100 },
            layer_predictions: vec![],
            coherence: vec![],
            event_intensities: vec![],
            evidential: vec![],
            next_event: None,
            hypergraph: None,
        };
        let output = runtime.update(&batch, &report, None).expect("report");
        assert_eq!(output.timestamp.unix, 100);
    }
}
