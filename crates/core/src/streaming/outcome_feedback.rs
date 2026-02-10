use crate::config::OutcomeFeedbackConfig;
use crate::schema::Timestamp;
use crate::streaming::behavior::BehaviorFrame;
use crate::streaming::schema::{EventKind, TokenBatch};
use crate::streaming::temporal::TemporalInferenceReport;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PredictionTarget {
    NextEventKind,
    NextMotif,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingPrediction {
    id: String,
    target: PredictionTarget,
    created_at: Timestamp,
    deadline: Timestamp,
    distribution: Vec<(String, f64)>,
    baseline: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeFeedbackSlice {
    pub target: PredictionTarget,
    pub samples: usize,
    pub avg_log_loss: f64,
    pub avg_baseline_log_loss: f64,
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeFeedbackReport {
    pub timestamp: Timestamp,
    pub resolved: usize,
    pub pending: usize,
    pub avg_log_loss: f64,
    pub avg_baseline_log_loss: f64,
    pub reward: f64,
    #[serde(default)]
    pub slices: Vec<OutcomeFeedbackSlice>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

pub struct StreamingOutcomeFeedbackRuntime {
    config: OutcomeFeedbackConfig,
    pending: VecDeque<PendingPrediction>,
    counter: u64,
}

impl StreamingOutcomeFeedbackRuntime {
    pub fn new(config: OutcomeFeedbackConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            counter: 0,
        }
    }

    pub fn update_config(&mut self, config: OutcomeFeedbackConfig) {
        self.config = config;
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        behavior: Option<&BehaviorFrame>,
        temporal: Option<&TemporalInferenceReport>,
    ) -> Option<OutcomeFeedbackReport> {
        if !self.config.enabled {
            return None;
        }
        let now = batch.timestamp;

        // Resolve matured predictions.
        let mut resolved = Vec::new();
        while let Some(front) = self.pending.front() {
            if front.deadline.unix > now.unix {
                break;
            }
            resolved.push(self.pending.pop_front().expect("front"));
        }

        let mut total_loss = 0.0;
        let mut total_base_loss = 0.0;
        let mut total_samples = 0usize;
        let mut by_target: HashMap<PredictionTarget, (usize, f64, f64)> = HashMap::new();

        for pred in &resolved {
            let observed = match pred.target {
                PredictionTarget::NextEventKind => observed_event_kind(batch).map(|k| format!("{:?}", k)),
                PredictionTarget::NextMotif => behavior
                    .and_then(|frame| frame.prediction.as_ref())
                    .and_then(|p| p.current_motif.clone()),
            };
            let Some(label) = observed else {
                continue;
            };
            let loss = log_loss(&pred.distribution, &label, self.config.min_prob);
            let base_loss = log_loss(&pred.baseline, &label, self.config.min_prob);
            total_loss += loss;
            total_base_loss += base_loss;
            total_samples += 1;
            let entry = by_target.entry(pred.target).or_insert((0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += loss;
            entry.2 += base_loss;
        }

        // Register new predictions.
        if let Some(report) = temporal {
            if !report.event_intensities.is_empty() {
                let dist = event_distribution(report);
                let base = event_baseline(report);
                self.push_prediction(
                    PredictionTarget::NextEventKind,
                    now,
                    Timestamp {
                        unix: now.unix.saturating_add(self.config.horizon_secs.max(1)),
                    },
                    dist,
                    base,
                );
            }
        }
        if let Some(frame) = behavior {
            if let Some(pred) = frame.prediction.as_ref() {
                if !pred.next_motif_distribution.is_empty() {
                    let dist = normalize_str_pairs(&pred.next_motif_distribution);
                    let base = uniform_baseline(&dist);
                    self.push_prediction(
                        PredictionTarget::NextMotif,
                        now,
                        Timestamp {
                            unix: now.unix.saturating_add(self.config.horizon_secs.max(1)),
                        },
                        dist,
                        base,
                    );
                }
            }
        }

        while self.pending.len() > self.config.max_pending.max(1) {
            self.pending.pop_front();
        }

        if total_samples == 0 {
            return None;
        }
        let avg_loss = total_loss / total_samples as f64;
        let avg_base = total_base_loss / total_samples as f64;
        let reward = (avg_base - avg_loss).clamp(-10.0, 10.0);
        let mut slices = Vec::new();
        for (target, (n, sum_loss, sum_base)) in by_target {
            if n == 0 {
                continue;
            }
            let avg = sum_loss / n as f64;
            let base = sum_base / n as f64;
            slices.push(OutcomeFeedbackSlice {
                target,
                samples: n,
                avg_log_loss: avg,
                avg_baseline_log_loss: base,
                reward: (base - avg).clamp(-10.0, 10.0),
            });
        }
        slices.sort_by_key(|s| s.target as u8);

        Some(OutcomeFeedbackReport {
            timestamp: now,
            resolved: resolved.len(),
            pending: self.pending.len(),
            avg_log_loss: avg_loss,
            avg_baseline_log_loss: avg_base,
            reward,
            slices,
            metadata: HashMap::new(),
        })
    }

    fn push_prediction(
        &mut self,
        target: PredictionTarget,
        created_at: Timestamp,
        deadline: Timestamp,
        distribution: Vec<(String, f64)>,
        baseline: Vec<(String, f64)>,
    ) {
        self.counter = self.counter.wrapping_add(1);
        let id = format!("pred-{}-{}", created_at.unix, self.counter);
        self.pending.push_back(PendingPrediction {
            id,
            target,
            created_at,
            deadline,
            distribution,
            baseline,
        });
    }
}

fn observed_event_kind(batch: &TokenBatch) -> Option<EventKind> {
    batch
        .tokens
        .iter()
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|t| t.kind)
}

fn log_loss(dist: &[(String, f64)], observed: &str, min_prob: f64) -> f64 {
    let min_prob = min_prob.clamp(1e-12, 1.0);
    let mut p = min_prob;
    for (label, prob) in dist {
        if label == observed {
            p = prob.max(min_prob);
            break;
        }
    }
    -p.ln()
}

fn normalize_str_pairs(dist: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = dist
        .iter()
        .map(|(k, v)| (k.clone(), v.max(0.0)))
        .collect();
    let sum: f64 = out.iter().map(|(_, v)| *v).sum();
    if sum > 0.0 {
        for entry in &mut out {
            entry.1 /= sum;
        }
    } else if !out.is_empty() {
        let uniform = 1.0 / out.len() as f64;
        for entry in &mut out {
            entry.1 = uniform;
        }
    }
    out
}

fn uniform_baseline(dist: &[(String, f64)]) -> Vec<(String, f64)> {
    if dist.is_empty() {
        return Vec::new();
    }
    let uniform = 1.0 / dist.len() as f64;
    dist.iter().map(|(k, _)| (k.clone(), uniform)).collect()
}

fn event_distribution(report: &TemporalInferenceReport) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = report
        .event_intensities
        .iter()
        .map(|e| (format!("{:?}", e.kind), e.intensity.max(0.0)))
        .collect();
    let sum: f64 = out.iter().map(|(_, v)| *v).sum();
    if sum > 0.0 {
        for entry in &mut out {
            entry.1 /= sum;
        }
    }
    out
}

fn event_baseline(report: &TemporalInferenceReport) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = report
        .event_intensities
        .iter()
        .map(|e| (format!("{:?}", e.kind), e.base_rate.max(0.0)))
        .collect();
    let sum: f64 = out.iter().map(|(_, v)| *v).sum();
    if sum > 0.0 {
        for entry in &mut out {
            entry.1 /= sum;
        }
        return out;
    }
    uniform_baseline(&event_distribution(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::temporal::{DirichletPosterior, EventIntensity, HypergraphStats, LayerPrediction};
    use crate::streaming::schema::{EventKind, LayerKind};
    use crate::streaming::temporal::TemporalInferenceReport;

    #[test]
    fn outcome_feedback_scores_event_prediction() {
        let cfg = OutcomeFeedbackConfig::default();
        let mut rt = StreamingOutcomeFeedbackRuntime::new(cfg);
        let now = Timestamp { unix: 10 };
        let report = TemporalInferenceReport {
            timestamp: now,
            layer_predictions: vec![LayerPrediction {
                kind: LayerKind::UltradianMicroArousal,
                domain: crate::streaming::DomainKind::People,
                predicted_phase: 0.0,
                predicted_amplitude: 0.0,
                predicted_coherence: 0.0,
                drift_phase: 0.0,
                drift_amplitude: 0.0,
                drift_coherence: 0.0,
            }],
            coherence: vec![],
            event_intensities: vec![EventIntensity {
                kind: EventKind::BehavioralAtom,
                intensity: 1.0,
                expected_time_secs: 1.0,
                base_rate: 0.2,
            }],
            evidential: vec![DirichletPosterior {
                label: "x".to_string(),
                categories: vec!["y".to_string()],
                alpha: vec![1.0],
            }],
            next_event: Some(EventKind::BehavioralAtom),
            hypergraph: Some(HypergraphStats { nodes: 1, edges: 1 }),
        };
        let batch0 = TokenBatch {
            timestamp: now,
            tokens: vec![],
            layers: vec![],
            source_confidence: HashMap::new(),
        };
        assert!(rt.update(&batch0, None, Some(&report)).is_none());

        let batch1 = TokenBatch {
            timestamp: Timestamp { unix: 10 + OutcomeFeedbackConfig::default().horizon_secs },
            tokens: vec![crate::streaming::schema::EventToken {
                id: "t".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: now,
                duration_secs: 1.0,
                confidence: 0.9,
                attributes: HashMap::new(),
                source: None,
            }],
            layers: vec![],
            source_confidence: HashMap::new(),
        };
        let out = rt.update(&batch1, None, None);
        assert!(out.is_some());
        let out = out.unwrap();
        assert!(out.avg_log_loss.is_finite());
        assert!(out.reward.is_finite());
    }
}




