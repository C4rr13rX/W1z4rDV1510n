use crate::config::MetacognitionConfig;
use crate::schema::Timestamp;
use crate::streaming::behavior::BehaviorFrame;
use crate::streaming::health_overlay::HealthOverlayReport;
use crate::streaming::schema::{EventKind, TokenBatch};
use crate::streaming::survival::{SurvivalEntityMetrics, SurvivalReport};
use crate::streaming::temporal::TemporalInferenceReport;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionExperiment {
    pub name: String,
    pub model_accuracy: f64,
    pub baseline_accuracy: f64,
    pub winner: String,
    #[serde(default)]
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyNote {
    pub entity_id: String,
    pub label: String,
    pub confidence: f64,
    #[serde(default)]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionEntity {
    pub entity_id: String,
    pub reflection_depth: usize,
    #[serde(default)]
    pub health_score: Option<f64>,
    #[serde(default)]
    pub survival_score: Option<f64>,
    #[serde(default)]
    pub empathy_label: Option<String>,
    #[serde(default)]
    pub empathy_confidence: Option<f64>,
    #[serde(default)]
    pub risk_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionReport {
    pub timestamp: Timestamp,
    pub reflection_depth: usize,
    pub model_accuracy: f64,
    pub baseline_accuracy: f64,
    pub pending_hypotheses: usize,
    pub resolved_hypotheses: usize,
    #[serde(default)]
    pub experiments: Vec<MetacognitionExperiment>,
    #[serde(default)]
    pub entities: Vec<MetacognitionEntity>,
    #[serde(default)]
    pub empathy_notes: Vec<EmpathyNote>,
    pub summary: String,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthAccuracyShare {
    pub depth: usize,
    pub correct: u64,
    pub total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionShare {
    pub timestamp: Timestamp,
    pub reflection_depth: usize,
    pub model_accuracy: f64,
    pub baseline_accuracy: f64,
    pub uncertainty: f64,
    #[serde(default)]
    pub depth_accuracy: Vec<DepthAccuracyShare>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HypothesisSource {
    Model,
    Baseline,
}

#[derive(Debug, Clone)]
struct PredictionHypothesis {
    id: String,
    source: HypothesisSource,
    event_kind: EventKind,
    created_at: Timestamp,
    deadline: Timestamp,
    confidence: f64,
    reflection_depth: usize,
}

#[derive(Debug, Clone)]
struct HypothesisOutcome {
    source: HypothesisSource,
    correct: bool,
    reflection_depth: usize,
}

#[derive(Default)]
struct AccuracyTracker {
    model_total: u64,
    model_correct: u64,
    baseline_total: u64,
    baseline_correct: u64,
    resolved_total: u64,
}

#[derive(Default, Clone)]
struct DepthStats {
    total: u64,
    correct: u64,
}

impl DepthStats {
    fn record(&mut self, correct: bool) {
        self.total = self.total.saturating_add(1);
        if correct {
            self.correct = self.correct.saturating_add(1);
        }
    }

    fn accuracy(&self) -> f64 {
        ratio(self.correct, self.total)
    }
}

pub struct MetacognitionRuntime {
    config: MetacognitionConfig,
    pending: VecDeque<PredictionHypothesis>,
    outcomes: VecDeque<HypothesisOutcome>,
    accuracy: AccuracyTracker,
    depth_stats: HashMap<usize, DepthStats>,
    peer_depth_stats: HashMap<usize, DepthStats>,
}

impl MetacognitionRuntime {
    pub fn new(config: MetacognitionConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            outcomes: VecDeque::new(),
            accuracy: AccuracyTracker::default(),
            depth_stats: HashMap::new(),
            peer_depth_stats: HashMap::new(),
        }
    }

    pub fn update_config(&mut self, config: MetacognitionConfig) {
        self.config = config;
    }

    pub fn share_snapshot(&self, report: &MetacognitionReport) -> MetacognitionShare {
        let uncertainty = report
            .metadata
            .get("uncertainty")
            .and_then(|val| val.as_f64())
            .unwrap_or(0.5);
        let mut depth_accuracy: Vec<DepthAccuracyShare> = self
            .depth_stats
            .iter()
            .map(|(depth, stats)| DepthAccuracyShare {
                depth: *depth,
                correct: stats.correct,
                total: stats.total,
            })
            .collect();
        depth_accuracy.sort_by_key(|entry| entry.depth);
        MetacognitionShare {
            timestamp: report.timestamp,
            reflection_depth: report.reflection_depth,
            model_accuracy: report.model_accuracy,
            baseline_accuracy: report.baseline_accuracy,
            uncertainty,
            depth_accuracy,
        }
    }

    pub fn ingest_peer_share(&mut self, share: &MetacognitionShare) {
        for entry in &share.depth_accuracy {
            let stats = self
                .peer_depth_stats
                .entry(entry.depth.max(1))
                .or_default();
            stats.total = stats.total.saturating_add(entry.total);
            stats.correct = stats.correct.saturating_add(entry.correct);
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        behavior_frame: Option<&BehaviorFrame>,
        health: Option<&HealthOverlayReport>,
        survival: Option<&SurvivalReport>,
        temporal: Option<&TemporalInferenceReport>,
    ) -> Option<MetacognitionReport> {
        if !self.config.enabled {
            return None;
        }
        let now = batch.timestamp;
        let observed = observed_event_kinds(batch);
        let resolved = self.resolve_hypotheses(now, &observed);
        while self.outcomes.len() > self.config.max_history.max(1) {
            self.outcomes.pop_front();
        }

        let (model_accuracy, baseline_accuracy) = self.accuracy();
        let uncertainty = evidential_uncertainty(temporal);
        let substream_stats = substream_stats(batch, self.config.substream_stability_threshold);
        let entity_count = behavior_frame
            .map(|frame| frame.states.len())
            .unwrap_or(0);
        let merged_stats = merge_depth_stats(&self.depth_stats, &self.peer_depth_stats);
        let decision = decide_reflection_depth(
            self.config.max_reflection_depth,
            self.config.min_reflection_depth,
            self.config.confident_depth,
            self.config.accuracy_target,
            self.config.confident_uncertainty_threshold,
            model_accuracy,
            uncertainty,
            substream_stats.novelty,
            &merged_stats,
            self.config.novelty_depth_boost,
            self.config.min_depth_samples,
            self.config.depth_improvement_margin,
        );
        let reflection_depth = decision.depth;
        let hypothesis_budget = hypothesis_budget(
            self.config.max_new_hypotheses,
            reflection_depth,
            self.config.max_reflection_depth,
        );
        let new_hypotheses =
            self.generate_hypotheses(now, batch, temporal, reflection_depth, hypothesis_budget);
        for hypothesis in new_hypotheses {
            self.pending.push_back(hypothesis);
            while self.pending.len() > self.config.max_pending.max(1) {
                self.pending.pop_front();
            }
        }

        let entities = build_entity_reflections(
            health,
            survival,
            reflection_depth,
            self.config.max_reflection_depth,
        );
        let empathy_notes = build_empathy_notes(survival);
        let (depth_accuracy_score, depth_samples) =
            depth_accuracy(&merged_stats, reflection_depth);
        let (local_depth_accuracy, local_depth_samples) =
            depth_accuracy(&self.depth_stats, reflection_depth);
        let mut experiments = Vec::new();
        experiments.push(MetacognitionExperiment {
            name: "event_prediction".to_string(),
            model_accuracy,
            baseline_accuracy,
            winner: if model_accuracy >= baseline_accuracy {
                "model".to_string()
            } else {
                "baseline".to_string()
            },
            samples: self.accuracy.resolved_total as usize,
        });
        experiments.push(MetacognitionExperiment {
            name: "reflection_depth".to_string(),
            model_accuracy: depth_accuracy_score,
            baseline_accuracy: model_accuracy,
            winner: if depth_accuracy_score >= model_accuracy {
                "depth".to_string()
            } else {
                "global".to_string()
            },
            samples: depth_samples as usize,
        });
        let summary = format!(
            "Reflection depth {} (model {:.2}, baseline {:.2}, depth {:.2}, uncertainty {:.2}, {}).",
            reflection_depth,
            model_accuracy,
            baseline_accuracy,
            depth_accuracy_score,
            uncertainty,
            decision.reason
        );
        let mut metadata = HashMap::from([
            ("substream_novelty".to_string(), Value::from(substream_stats.novelty)),
            ("substream_count".to_string(), Value::from(substream_stats.count as u64)),
            ("uncertainty".to_string(), Value::from(uncertainty)),
            ("depth_reason".to_string(), Value::from(decision.reason.clone())),
            ("depth_target".to_string(), Value::from(decision.base_depth as u64)),
            ("hypothesis_budget".to_string(), Value::from(hypothesis_budget as u64)),
            ("entity_count".to_string(), Value::from(entity_count as u64)),
            (
                "depth_accuracy_local".to_string(),
                Value::from(local_depth_accuracy),
            ),
            (
                "depth_samples_local".to_string(),
                Value::from(local_depth_samples),
            ),
        ]);
        if let Some(min_stability) = substream_stats.min_stability {
            metadata.insert(
                "substream_min_stability".to_string(),
                Value::from(min_stability),
            );
        }
        if let Some(best_depth) = decision.best_depth {
            metadata.insert("best_depth".to_string(), Value::from(best_depth as u64));
        }
        if let Some(best_accuracy) = decision.best_accuracy {
            metadata.insert("best_depth_accuracy".to_string(), Value::from(best_accuracy));
        }

        Some(MetacognitionReport {
            timestamp: now,
            reflection_depth,
            model_accuracy,
            baseline_accuracy,
            pending_hypotheses: self.pending.len(),
            resolved_hypotheses: resolved,
            experiments,
            entities,
            empathy_notes,
            summary,
            metadata,
        })
    }

    fn resolve_hypotheses(
        &mut self,
        now: Timestamp,
        observed: &HashSet<EventKind>,
    ) -> usize {
        if self.pending.is_empty() {
            return 0;
        }
        let mut resolved = Vec::new();
        let mut kept = VecDeque::new();
        while let Some(hypothesis) = self.pending.pop_front() {
            if now.unix > hypothesis.deadline.unix {
                resolved.push(HypothesisOutcome {
                    source: hypothesis.source,
                    correct: false,
                    reflection_depth: hypothesis.reflection_depth,
                });
                continue;
            }
            if observed.contains(&hypothesis.event_kind) {
                resolved.push(HypothesisOutcome {
                    source: hypothesis.source,
                    correct: true,
                    reflection_depth: hypothesis.reflection_depth,
                });
                continue;
            }
            kept.push_back(hypothesis);
        }
        self.pending = kept;
        for outcome in &resolved {
            self.record_outcome(outcome);
        }
        resolved.len()
    }

    fn generate_hypotheses(
        &self,
        now: Timestamp,
        batch: &TokenBatch,
        temporal: Option<&TemporalInferenceReport>,
        reflection_depth: usize,
        hypothesis_budget: usize,
    ) -> Vec<PredictionHypothesis> {
        let mut out = Vec::new();
        let mut created = HashSet::new();
        if let Some(report) = temporal {
            let mut intensities = report.event_intensities.clone();
            intensities.sort_by(|a, b| {
                b.intensity
                    .partial_cmp(&a.intensity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for intensity in intensities
                .into_iter()
                .take(hypothesis_budget.max(1))
            {
                if intensity.intensity < self.config.min_event_intensity {
                    continue;
                }
                let window = intensity
                    .expected_time_secs
                    .min(self.config.prediction_window_secs as f64)
                    .max(1.0);
                let deadline = Timestamp {
                    unix: now.unix + window.round() as i64,
                };
                let id = format!("model|{:?}|{}|{}", intensity.kind, now.unix, deadline.unix);
                if created.insert(id.clone()) {
                    out.push(PredictionHypothesis {
                        id,
                        source: HypothesisSource::Model,
                        event_kind: intensity.kind,
                        created_at: now,
                        deadline,
                        confidence: intensity.intensity.clamp(0.0, 1.0),
                        reflection_depth,
                    });
                }
            }
        }
        if let Some(event_kind) = baseline_event_kind(batch) {
            let deadline = Timestamp {
                unix: now.unix + self.config.prediction_window_secs.max(1) as i64,
            };
            let id = format!("baseline|{:?}|{}|{}", event_kind, now.unix, deadline.unix);
            if created.insert(id.clone()) {
                out.push(PredictionHypothesis {
                    id,
                    source: HypothesisSource::Baseline,
                    event_kind,
                    created_at: now,
                    deadline,
                    confidence: 0.5,
                    reflection_depth,
                });
            }
        }
        out
    }

    fn record_outcome(&mut self, outcome: &HypothesisOutcome) {
        self.accuracy.resolved_total = self.accuracy.resolved_total.saturating_add(1);
        match outcome.source {
            HypothesisSource::Model => {
                self.accuracy.model_total = self.accuracy.model_total.saturating_add(1);
                if outcome.correct {
                    self.accuracy.model_correct = self.accuracy.model_correct.saturating_add(1);
                }
            }
            HypothesisSource::Baseline => {
                self.accuracy.baseline_total = self.accuracy.baseline_total.saturating_add(1);
                if outcome.correct {
                    self.accuracy.baseline_correct = self.accuracy.baseline_correct.saturating_add(1);
                }
            }
        }
        self.depth_stats
            .entry(outcome.reflection_depth.max(1))
            .or_default()
            .record(outcome.correct);
        self.outcomes.push_back(outcome.clone());
    }

    fn accuracy(&self) -> (f64, f64) {
        let model = ratio(self.accuracy.model_correct, self.accuracy.model_total);
        let baseline = ratio(self.accuracy.baseline_correct, self.accuracy.baseline_total);
        (model, baseline)
    }
}

impl Default for MetacognitionRuntime {
    fn default() -> Self {
        Self::new(MetacognitionConfig::default())
    }
}

fn ratio(correct: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.5;
    }
    (correct as f64 / total as f64).clamp(0.0, 1.0)
}

struct ReflectionDecision {
    depth: usize,
    base_depth: usize,
    reason: String,
    best_depth: Option<usize>,
    best_accuracy: Option<f64>,
}

fn observed_event_kinds(batch: &TokenBatch) -> HashSet<EventKind> {
    let mut set = HashSet::new();
    for token in &batch.tokens {
        set.insert(token.kind);
    }
    set
}

fn baseline_event_kind(batch: &TokenBatch) -> Option<EventKind> {
    let mut counts: HashMap<EventKind, usize> = HashMap::new();
    for token in &batch.tokens {
        *counts.entry(token.kind).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|entry| entry.0)
}

fn evidential_uncertainty(report: Option<&TemporalInferenceReport>) -> f64 {
    let Some(report) = report else { return 0.5; };
    if report.evidential.is_empty() {
        return 0.5;
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for posterior in &report.evidential {
        let total: f64 = posterior.alpha.iter().sum();
        let k = posterior.alpha.len().max(1) as f64;
        let confidence = (total / (total + k)).clamp(0.0, 1.0);
        sum += 1.0 - confidence;
        count += 1.0;
    }
    if count > 0.0 {
        (sum / count).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn reflection_depth(
    max_depth: usize,
    min_depth: usize,
    accuracy_target: f64,
    accuracy: f64,
    uncertainty: f64,
) -> usize {
    let max_depth = max_depth.max(1);
    let min_depth = min_depth.max(1).min(max_depth);
    let gap = (accuracy_target - accuracy).max(0.0);
    let bump = (gap * max_depth as f64).ceil() as usize;
    let uncertainty_boost = if uncertainty > 0.6 { 1 } else { 0 };
    let depth = min_depth + bump + uncertainty_boost;
    depth.clamp(min_depth, max_depth)
}

fn decide_reflection_depth(
    max_depth: usize,
    min_depth: usize,
    confident_depth: usize,
    accuracy_target: f64,
    confident_uncertainty_threshold: f64,
    model_accuracy: f64,
    uncertainty: f64,
    substream_novelty: bool,
    depth_stats: &HashMap<usize, DepthStats>,
    novelty_boost: usize,
    min_depth_samples: u64,
    depth_improvement_margin: f64,
) -> ReflectionDecision {
    let max_depth = max_depth.max(1);
    let min_depth = min_depth.max(1).min(max_depth);
    let confident_depth = confident_depth.max(min_depth).min(max_depth);
    let base_depth = reflection_depth(
        max_depth,
        min_depth,
        accuracy_target,
        model_accuracy,
        uncertainty,
    );
    let mut depth = base_depth;
    let mut reason = "accuracy_gap".to_string();
    if model_accuracy >= accuracy_target
        && uncertainty <= confident_uncertainty_threshold
        && !substream_novelty
    {
        depth = confident_depth;
        reason = "accuracy_target_met".to_string();
    }
    if substream_novelty {
        depth = (depth + novelty_boost).min(max_depth);
        reason = "substream_novelty".to_string();
    }
    let mut best_depth = None;
    let mut best_accuracy = None;
    if let Some((candidate_depth, candidate_accuracy, _)) =
        best_depth_stats(depth_stats, min_depth_samples)
    {
        best_depth = Some(candidate_depth);
        best_accuracy = Some(candidate_accuracy);
        if !substream_novelty
            && model_accuracy < accuracy_target
            && candidate_accuracy + depth_improvement_margin >= model_accuracy
        {
            depth = candidate_depth.min(max_depth);
            reason = "depth_accuracy_best".to_string();
        }
    }
    ReflectionDecision {
        depth,
        base_depth,
        reason,
        best_depth,
        best_accuracy,
    }
}

fn hypothesis_budget(max_new: usize, depth: usize, max_depth: usize) -> usize {
    let max_depth = max_depth.max(1);
    let scaled = (max_new as f64 * depth as f64 / max_depth as f64).ceil() as usize;
    scaled.max(1)
}

fn depth_accuracy(depth_stats: &HashMap<usize, DepthStats>, depth: usize) -> (f64, u64) {
    let depth = depth.max(1);
    if let Some(stats) = depth_stats.get(&depth) {
        (stats.accuracy(), stats.total)
    } else {
        (0.5, 0)
    }
}

fn merge_depth_stats(
    local: &HashMap<usize, DepthStats>,
    peers: &HashMap<usize, DepthStats>,
) -> HashMap<usize, DepthStats> {
    let mut merged = local.clone();
    for (depth, stats) in peers {
        let entry = merged.entry(*depth).or_default();
        entry.total = entry.total.saturating_add(stats.total);
        entry.correct = entry.correct.saturating_add(stats.correct);
    }
    merged
}

fn best_depth_stats(
    depth_stats: &HashMap<usize, DepthStats>,
    min_samples: u64,
) -> Option<(usize, f64, u64)> {
    let mut best: Option<(usize, f64, u64)> = None;
    for (depth, stats) in depth_stats {
        if stats.total < min_samples {
            continue;
        }
        let accuracy = stats.accuracy();
        match best {
            Some((_, best_accuracy, _)) if accuracy <= best_accuracy => {}
            _ => best = Some((*depth, accuracy, stats.total)),
        }
    }
    best
}

struct SubstreamStats {
    novelty: bool,
    count: usize,
    min_stability: Option<f64>,
}

fn substream_stats(batch: &TokenBatch, stability_threshold: f64) -> SubstreamStats {
    let mut count = 0usize;
    let mut novelty = false;
    let mut min_stability: Option<f64> = None;
    for token in &batch.tokens {
        if token.attributes.get("substream_key").is_none() {
            continue;
        }
        count += 1;
        if let Some(stability) = token
            .attributes
            .get("substream_stability")
            .and_then(|val| val.as_f64())
        {
            if stability < stability_threshold {
                novelty = true;
            }
            min_stability = Some(min_stability.map_or(stability, |min| min.min(stability)));
        }
    }
    if count >= 3 {
        novelty = true;
    }
    SubstreamStats {
        novelty,
        count,
        min_stability,
    }
}

fn build_entity_reflections(
    health: Option<&HealthOverlayReport>,
    survival: Option<&SurvivalReport>,
    global_depth: usize,
    max_depth: usize,
) -> Vec<MetacognitionEntity> {
    let mut entities = Vec::new();
    let mut health_map: HashMap<String, f64> = HashMap::new();
    if let Some(report) = health {
        for entry in &report.entities {
            health_map.insert(entry.entity_id.clone(), entry.score);
        }
    }
    let mut survival_map: HashMap<String, &SurvivalEntityMetrics> = HashMap::new();
    if let Some(report) = survival {
        for entry in &report.entities {
            survival_map.insert(entry.entity_id.clone(), entry);
        }
    }
    let mut ids: HashSet<String> = HashSet::new();
    ids.extend(health_map.keys().cloned());
    ids.extend(survival_map.keys().cloned());
    for entity_id in ids {
        let health_score = health_map.get(&entity_id).copied();
        let survival_entry = survival_map.get(&entity_id).copied();
        let risk = health_score.map(|score| (1.0 - score).clamp(0.0, 1.0));
        let depth = risk
            .map(|risk| 1 + (risk * (max_depth.saturating_sub(1)) as f64).round() as usize)
            .unwrap_or(global_depth)
            .max(global_depth)
            .min(max_depth.max(1));
        let (empathy_label, empathy_confidence) = survival_entry
            .map(empathy_from_survival)
            .unwrap_or((None, None));
        entities.push(MetacognitionEntity {
            entity_id,
            reflection_depth: depth,
            health_score,
            survival_score: survival_entry.map(|entry| entry.survival_score),
            empathy_label,
            empathy_confidence,
            risk_score: risk,
        });
    }
    entities
}

fn build_empathy_notes(survival: Option<&SurvivalReport>) -> Vec<EmpathyNote> {
    let mut notes = Vec::new();
    let Some(report) = survival else { return notes; };
    for entry in &report.entities {
        let (label, confidence, reasons) = empathy_label(entry);
        notes.push(EmpathyNote {
            entity_id: entry.entity_id.clone(),
            label,
            confidence,
            reasons,
        });
    }
    notes
}

fn empathy_from_survival(entry: &SurvivalEntityMetrics) -> (Option<String>, Option<f64>) {
    let (label, confidence, _) = empathy_label(entry);
    (Some(label), Some(confidence))
}

fn empathy_label(entry: &SurvivalEntityMetrics) -> (String, f64, Vec<String>) {
    let mut reasons = Vec::new();
    let conflict = entry.conflict_in.clamp(0.0, 1.0);
    let cooperation = entry.cooperation_in.clamp(0.0, 1.0);
    let play = entry.play_index.clamp(0.0, 1.0);
    let survival = entry.survival_score.clamp(0.0, 1.0);
    let intent = entry.intent_magnitude.clamp(0.0, 1.0);
    let mut label = "neutral".to_string();
    let mut confidence = 0.4;
    if conflict >= 0.6 && survival < 0.5 {
        label = "distressed".to_string();
        confidence = (conflict + (1.0 - survival)) * 0.5;
        reasons.push("conflict_high".to_string());
    } else if cooperation >= 0.6 {
        label = "cooperative".to_string();
        confidence = cooperation;
        reasons.push("cooperation_high".to_string());
    } else if play >= 0.6 {
        label = "playful".to_string();
        confidence = play;
        reasons.push("playful_pattern".to_string());
    } else if survival >= 0.8 {
        label = "stable".to_string();
        confidence = survival;
        reasons.push("survival_high".to_string());
    }
    if intent >= 0.6 {
        reasons.push("intent_signal_strong".to_string());
    }
    (label, confidence.clamp(0.0, 1.0), reasons)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken};
    use serde_json::Value;

    #[test]
    fn metacognition_tracks_hypotheses() {
        let mut runtime = MetacognitionRuntime::default();
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 10 },
            tokens: vec![],
            layers: vec![],
            source_confidence: HashMap::new(),
        };
        let report = runtime.update(&batch, None, None, None, None);
        assert!(report.is_some());
        let report = report.unwrap();
        assert!(report.summary.contains("Reflection depth"));
    }

    #[test]
    fn metacognition_prefers_best_depth_when_better() {
        let mut stats = HashMap::new();
        stats.insert(2, DepthStats { total: 10, correct: 6 });
        stats.insert(4, DepthStats { total: 10, correct: 9 });
        let decision = decide_reflection_depth(
            6,
            1,
            2,
            0.8,
            0.4,
            0.5,
            0.5,
            false,
            &stats,
            1,
            6,
            0.05,
        );
        assert_eq!(decision.depth, 4);
        assert_eq!(decision.reason, "depth_accuracy_best");
    }

    #[test]
    fn substream_novelty_escalates_depth() {
        let mut stats = HashMap::new();
        stats.insert(1, DepthStats { total: 10, correct: 6 });
        let decision = decide_reflection_depth(
            5,
            1,
            2,
            0.8,
            0.4,
            0.6,
            0.7,
            true,
            &stats,
            2,
            6,
            0.05,
        );
        assert!(decision.depth >= 3);
        assert_eq!(decision.reason, "substream_novelty");
    }

    #[test]
    fn substream_stats_flags_low_stability() {
        let mut attrs = HashMap::new();
        attrs.insert("substream_key".to_string(), Value::String("mini::attr::speed::fast".to_string()));
        attrs.insert("substream_stability".to_string(), Value::from(0.4));
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 42 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralToken,
                onset: Timestamp { unix: 42 },
                duration_secs: 1.0,
                confidence: 0.8,
                attributes: attrs,
                source: None,
            }],
            layers: vec![],
            source_confidence: HashMap::new(),
        };
        let stats = substream_stats(&batch, 0.6);
        assert!(stats.novelty);
        assert_eq!(stats.count, 1);
    }
}
