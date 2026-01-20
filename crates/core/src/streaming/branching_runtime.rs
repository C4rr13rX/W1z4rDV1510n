use crate::branching::{BranchNode, BranchingFutures, Retrodiction};
use crate::compute::{ComputeJobKind, QuantumExecutor, QuantumJob};
use crate::config::{BranchingFuturesConfig, QuantumConfig};
use crate::math_toolbox as math;
use crate::schema::Timestamp;
use crate::streaming::causal_stream::CausalReport;
use crate::streaming::physiology_runtime::PhysiologyReport;
use crate::streaming::schema::{EventKind, TokenBatch};
use crate::streaming::temporal::{EventIntensity, TemporalInferenceReport};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingReport {
    pub timestamp: Timestamp,
    pub root_id: u64,
    pub branches: Vec<BranchNode>,
    pub retrodictions: Vec<Retrodiction>,
    pub total_nodes: usize,
    #[serde(default)]
    pub quantum: Option<BranchingQuantumReport>,
}

pub struct StreamingBranchingRuntime {
    config: BranchingFuturesConfig,
    futures: BranchingFutures,
    retrodictions: Vec<Retrodiction>,
    quantum: Option<QuantumBranchingRuntime>,
}

impl StreamingBranchingRuntime {
    pub fn new(
        config: BranchingFuturesConfig,
        quantum: QuantumConfig,
        executor: Option<Box<dyn QuantumExecutor>>,
    ) -> Self {
        let root_payload = serde_json::json!({"root": true});
        let quantum = if config.quantum_enabled {
            Some(QuantumBranchingRuntime::new(quantum, executor))
        } else {
            None
        };
        Self {
            config: config.clone(),
            futures: BranchingFutures::new(config, root_payload),
            retrodictions: Vec::new(),
            quantum,
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        report: &TemporalInferenceReport,
        causal: Option<&CausalReport>,
        physiology: Option<&PhysiologyReport>,
    ) -> Option<BranchingReport> {
        if !self.config.enabled {
            return None;
        }
        let now = report.timestamp;
        let root = self.futures.root_id();
        let mut branches = Vec::new();
        let mut intensities = report.event_intensities.clone();
        intensities.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total_intensity: f64 = intensities.iter().map(|e| e.intensity).sum();
        let candidate_count = self.config.max_branches.min(16).min(intensities.len());
        let candidates: Vec<EventIntensity> = intensities.iter().take(candidate_count).cloned().collect();
        let base_probs = base_probabilities(&candidates, total_intensity);
        let mut probabilities = base_probs.clone();
        let mut quantum_report = None;
        if let Some(quantum) = &mut self.quantum {
            let quantum_count = self
                .config
                .quantum_max_candidates
                .min(candidates.len().max(1));
            if let Some(outcome) = quantum.score(
                now,
                &candidates[..quantum_count],
                causal,
                physiology,
            ) {
                let blend = self.config.quantum_blend_alpha.clamp(0.0, 1.0);
                let base_mass: f64 = base_probs.iter().take(quantum_count).sum();
                for (idx, prob) in probabilities.iter_mut().enumerate() {
                    if idx < quantum_count {
                        let quantum_prob = outcome.probabilities[idx];
                        *prob = (1.0 - blend) * *prob + blend * quantum_prob * base_mass;
                    }
                }
                quantum_report = Some(outcome.report(blend));
            }
        }
        for (idx, intensity) in candidates.iter().enumerate() {
            let probability = probabilities.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let uncertainty = (1.0 - probability).clamp(0.0, 1.0);
            let payload = branch_payload(intensity, causal, physiology);
            if let Some(id) = self
                .futures
                .add_branch(root, now, probability, uncertainty, payload)
            {
                if let Some(node) = self.futures.nodes().get(&id) {
                    branches.push(node.clone());
                }
            }
        }
        let mut new_retrodictions = Vec::new();
        if self.config.retrodiction_enabled {
            let observed = observed_event_kinds(batch);
            for intensity in intensities.iter() {
                if new_retrodictions.len() >= self.config.retrodiction_max {
                    break;
                }
                if observed.contains(&intensity.kind) {
                    continue;
                }
                if intensity.intensity < self.config.retrodiction_min_intensity {
                    continue;
                }
                let confidence = if total_intensity > 0.0 {
                    (intensity.intensity / total_intensity).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let timestamp = Timestamp {
                    unix: (now.unix - 1).max(0),
                };
                let payload = serde_json::json!({
                    "event_kind": format!("{:?}", intensity.kind),
                    "expected_time_secs": intensity.expected_time_secs,
                    "intensity": intensity.intensity,
                });
                let retrodiction = Retrodiction {
                    timestamp,
                    payload,
                    confidence,
                };
                new_retrodictions.push(retrodiction.clone());
                self.retrodictions.push(retrodiction);
            }
            if self.retrodictions.len() > self.config.retrodiction_max {
                let overflow = self.retrodictions.len() - self.config.retrodiction_max;
                self.retrodictions.drain(0..overflow);
            }
        }

        Some(BranchingReport {
            timestamp: now,
            root_id: root,
            branches,
            retrodictions: new_retrodictions,
            total_nodes: self.futures.nodes().len(),
            quantum: quantum_report,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantumBranchCandidate {
    event_kind: String,
    intensity: f64,
    expected_time_secs: f64,
    base_rate: f64,
    #[serde(default)]
    payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantumBranchScoringRequest {
    timestamp: Timestamp,
    candidates: Vec<QuantumBranchCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantumBranchScoringResponse {
    scores: Vec<f64>,
    #[serde(default)]
    probabilities: Option<Vec<f64>>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingQuantumReport {
    pub used_remote: bool,
    pub blend_alpha: f64,
    pub candidates: usize,
    pub score_scale: f64,
    pub temperature: f64,
    pub samples: usize,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantumBranchCalibration {
    score_scale: f64,
    temperature: f64,
    samples: usize,
}

impl Default for QuantumBranchCalibration {
    fn default() -> Self {
        Self {
            score_scale: 1.0,
            temperature: 0.7,
            samples: 0,
        }
    }
}

struct QuantumBranchingRuntime {
    config: QuantumConfig,
    executor: Option<Box<dyn QuantumExecutor>>,
    calibration: QuantumBranchCalibration,
}

impl QuantumBranchingRuntime {
    fn new(config: QuantumConfig, executor: Option<Box<dyn QuantumExecutor>>) -> Self {
        let mut calibration = QuantumBranchCalibration::default();
        if config.slice_temperature_scale > 0.0 {
            calibration.temperature = config.slice_temperature_scale;
        }
        Self {
            config,
            executor,
            calibration,
        }
    }

    fn score(
        &mut self,
        now: Timestamp,
        candidates: &[EventIntensity],
        causal: Option<&CausalReport>,
        physiology: Option<&PhysiologyReport>,
    ) -> Option<QuantumScoreOutcome> {
        if candidates.is_empty() {
            return None;
        }
        let local_probs = self.local_probabilities(candidates);
        let mut used_remote = false;
        let mut metadata = HashMap::new();
        let mut probabilities = local_probs.clone();
        if self.config.remote_enabled {
            if let Some(executor) = self.executor.as_ref() {
                if let Ok(result) = submit_quantum_branch_job(
                    executor.as_ref(),
                    now,
                    candidates,
                    causal,
                    physiology,
                    self.config.remote_timeout_secs,
                ) {
                    if let Some(remote_probs) = normalize_quantum_response(&result) {
                        used_remote = true;
                        metadata = result.metadata;
                        self.calibration.update_from_remote(
                            &local_probs,
                            &remote_probs,
                            self.config.calibration_alpha,
                        );
                        probabilities = remote_probs;
                    }
                }
            }
        }
        Some(QuantumScoreOutcome {
            probabilities,
            used_remote,
            metadata,
            calibration: self.calibration.clone(),
            candidate_count: candidates.len(),
        })
    }

    fn local_probabilities(&self, candidates: &[EventIntensity]) -> Vec<f64> {
        let temp = self.calibration.temperature.max(1e-3);
        let scale = self.calibration.score_scale.max(0.1);
        let time_scale = 120.0;
        let mut scores = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let intensity = candidate.intensity.clamp(0.0, 1.0);
            let base_rate = candidate.base_rate.clamp(0.0, 1.0);
            let time_penalty = (candidate.expected_time_secs / time_scale).clamp(0.0, 1.0);
            let energy = ((1.0 - intensity) + time_penalty * 0.25 + (1.0 - base_rate) * 0.1) * scale;
            scores.push(-energy / temp);
        }
        math::softmax(&scores)
    }
}

struct QuantumScoreOutcome {
    probabilities: Vec<f64>,
    used_remote: bool,
    metadata: HashMap<String, String>,
    calibration: QuantumBranchCalibration,
    candidate_count: usize,
}

impl QuantumScoreOutcome {
    fn report(&self, blend_alpha: f64) -> BranchingQuantumReport {
        BranchingQuantumReport {
            used_remote: self.used_remote,
            blend_alpha,
            candidates: self.candidate_count,
            score_scale: self.calibration.score_scale,
            temperature: self.calibration.temperature,
            samples: self.calibration.samples,
            metadata: self.metadata.clone(),
        }
    }
}

impl QuantumBranchCalibration {
    fn update_from_remote(&mut self, local: &[f64], remote: &[f64], alpha: f64) {
        let alpha = alpha.clamp(0.0, 1.0);
        let local_top = max_value(local).max(1e-6);
        let remote_top = max_value(remote).max(1e-6);
        let scale_target = (remote_top / local_top).clamp(0.5, 2.0);
        self.score_scale = blend(self.score_scale, self.score_scale * scale_target, alpha);
        let local_entropy = math::entropy(local).max(1e-6);
        let remote_entropy = math::entropy(remote).max(1e-6);
        let entropy_ratio = (remote_entropy / local_entropy).clamp(0.25, 4.0);
        let temp_target = (self.temperature * entropy_ratio).clamp(0.05, 5.0);
        self.temperature = blend(self.temperature, temp_target, alpha);
        self.samples = self.samples.saturating_add(1);
    }
}

fn submit_quantum_branch_job(
    executor: &dyn QuantumExecutor,
    now: Timestamp,
    candidates: &[EventIntensity],
    causal: Option<&CausalReport>,
    physiology: Option<&PhysiologyReport>,
    timeout_secs: u64,
) -> anyhow::Result<QuantumBranchScoringResponse> {
    let request = QuantumBranchScoringRequest {
        timestamp: now,
        candidates: candidates
            .iter()
            .map(|candidate| QuantumBranchCandidate {
                event_kind: format!("{:?}", candidate.kind),
                intensity: candidate.intensity,
                expected_time_secs: candidate.expected_time_secs,
                base_rate: candidate.base_rate,
                payload: branch_payload(candidate, causal, physiology),
            })
            .collect(),
    };
    let payload = serde_json::to_vec(&request)?;
    let job = QuantumJob {
        kind: ComputeJobKind::BranchScoring,
        payload,
        timeout_secs,
    };
    let result = executor.submit(job)?;
    let mut response: QuantumBranchScoringResponse = serde_json::from_slice(&result.payload)?;
    for (key, value) in result.metadata {
        response.metadata.entry(key).or_insert(value);
    }
    Ok(response)
}

fn normalize_quantum_response(response: &QuantumBranchScoringResponse) -> Option<Vec<f64>> {
    if let Some(probs) = response.probabilities.as_ref() {
        if probs.is_empty() {
            return None;
        }
        return Some(math::normalize_probs(probs));
    }
    if response.scores.is_empty() {
        return None;
    }
    Some(math::softmax(&response.scores))
}

fn base_probabilities(intensities: &[EventIntensity], total: f64) -> Vec<f64> {
    intensities
        .iter()
        .map(|entry| {
            if total > 0.0 {
                (entry.intensity / total).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn blend(current: f64, target: f64, alpha: f64) -> f64 {
    current * (1.0 - alpha) + target * alpha
}

fn max_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(0.0, |acc, val| if val > acc { val } else { acc })
}

fn branch_payload(
    intensity: &EventIntensity,
    causal: Option<&CausalReport>,
    physiology: Option<&PhysiologyReport>,
) -> Value {
    let mut payload = HashMap::new();
    payload.insert(
        "event_kind".to_string(),
        Value::String(format!("{:?}", intensity.kind)),
    );
    payload.insert(
        "expected_time_secs".to_string(),
        Value::from(intensity.expected_time_secs),
    );
    payload.insert("intensity".to_string(), Value::from(intensity.intensity));
    if let Some(causal_report) = causal {
        if !causal_report.interventions.is_empty() {
            let deltas = causal_report
                .interventions
                .iter()
                .take(4)
                .map(|delta| {
                    serde_json::json!({
                        "target": delta.target,
                        "expected_delta": delta.expected_delta,
                        "lag_secs": delta.lag_secs,
                    })
                })
                .collect::<Vec<_>>();
            payload.insert("counterfactuals".to_string(), Value::from(deltas));
        }
    }
    if let Some(phys_report) = physiology {
        payload.insert(
            "physiology_overall_index".to_string(),
            Value::from(phys_report.overall_index),
        );
        payload.insert(
            "physiology_contexts".to_string(),
            Value::from(phys_report.deviations.len() as u64),
        );
    }
    Value::Object(payload.into_iter().collect())
}

fn observed_event_kinds(batch: &TokenBatch) -> HashSet<EventKind> {
    let mut set = HashSet::new();
    for token in &batch.tokens {
        set.insert(token.kind);
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, LayerKind};
    use crate::streaming::temporal::{
        CoherencePrediction, DirichletPosterior, HypergraphStats, LayerPrediction,
    };

    #[test]
    fn branching_runtime_creates_retrodictions() {
        let mut config = BranchingFuturesConfig::default();
        config.enabled = true;
        config.retrodiction_enabled = true;
        config.retrodiction_min_intensity = 0.2;
        let mut runtime = StreamingBranchingRuntime::new(config, QuantumConfig::default(), None);
        let report = TemporalInferenceReport {
            timestamp: Timestamp { unix: 100 },
            layer_predictions: vec![LayerPrediction {
                kind: LayerKind::UltradianMicroArousal,
                domain: crate::streaming::DomainKind::People,
                predicted_phase: 0.1,
                predicted_amplitude: 0.7,
                predicted_coherence: 0.4,
                drift_phase: 0.0,
                drift_amplitude: 0.0,
                drift_coherence: 0.0,
            }],
            coherence: vec![CoherencePrediction {
                layer_a: LayerKind::UltradianMicroArousal,
                layer_b: LayerKind::UltradianBrac,
                coherence: 0.2,
                drift: 0.0,
                cross_domain: false,
            }],
            event_intensities: vec![EventIntensity {
                kind: EventKind::BehavioralAtom,
                intensity: 1.0,
                expected_time_secs: 1.0,
                base_rate: 0.2,
            }],
            evidential: vec![DirichletPosterior {
                label: "event_kind".to_string(),
                categories: vec!["Behavioral".to_string()],
                alpha: vec![1.0],
            }],
            next_event: Some(EventKind::BehavioralAtom),
            hypergraph: Some(HypergraphStats { nodes: 2, edges: 1 }),
        };
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let update = runtime.update(&batch, &report, None, None).expect("report");
        assert!(!update.branches.is_empty());
        assert!(!update.retrodictions.is_empty());
    }
}
