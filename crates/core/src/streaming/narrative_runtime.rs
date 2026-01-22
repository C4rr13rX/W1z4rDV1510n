use crate::config::NarrativeConfig;
use crate::schema::Timestamp;
use crate::streaming::branching_runtime::BranchingReport;
use crate::streaming::causal_stream::CausalReport;
use crate::streaming::behavior::BehaviorFrame;
use crate::streaming::health_overlay::{HealthEntityOverlay, HealthOverlayReport};
use crate::streaming::metacognition_runtime::MetacognitionReport;
use crate::streaming::survival::{SurvivalEntityMetrics, SurvivalReport};
use crate::streaming::temporal::TemporalInferenceReport;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeStep {
    pub timestamp: Timestamp,
    pub text: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeZoomSummary {
    pub label: String,
    pub window_secs: i64,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEntitySummary {
    pub entity_id: String,
    pub severity: String,
    pub risk_score: f64,
    #[serde(default)]
    pub health_score: Option<f64>,
    #[serde(default)]
    pub survival_score: Option<f64>,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    #[serde(default)]
    pub health_label: Option<String>,
    #[serde(default)]
    pub health_color: Option<String>,
    pub summary: String,
    #[serde(default)]
    pub severity_color: Option<String>,
    #[serde(default)]
    pub empathy_label: Option<String>,
    #[serde(default)]
    pub empathy_confidence: Option<f64>,
    #[serde(default)]
    pub reflection_depth: Option<usize>,
    #[serde(default)]
    pub zoom: Vec<NarrativeZoomSummary>,
    #[serde(default)]
    pub steps: Vec<NarrativeStep>,
    #[serde(default)]
    pub evidence: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeReport {
    pub timestamp: Timestamp,
    pub overall_risk: f64,
    pub overall_severity: String,
    #[serde(default)]
    pub overall_color: Option<String>,
    pub summary: String,
    #[serde(default)]
    pub entities: Vec<NarrativeEntitySummary>,
    #[serde(default)]
    pub steps: Vec<NarrativeStep>,
}

#[derive(Debug, Clone)]
struct NarrativePoint {
    timestamp: Timestamp,
    health_score: Option<f64>,
    survival_score: Option<f64>,
    risk_score: f64,
    conflict: Option<f64>,
    cooperation: Option<f64>,
    play_index: Option<f64>,
    intent: Option<f64>,
    position: Option<[f64; 3]>,
    motif_id: Option<String>,
}

pub struct NarrativeRuntime {
    config: NarrativeConfig,
    history: HashMap<String, VecDeque<NarrativePoint>>,
}

impl NarrativeRuntime {
    pub fn new(config: NarrativeConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
        }
    }

    pub fn update(
        &mut self,
        timestamp: Timestamp,
        behavior_frame: Option<&BehaviorFrame>,
        health: Option<&HealthOverlayReport>,
        survival: Option<&SurvivalReport>,
        temporal: Option<&TemporalInferenceReport>,
        causal: Option<&CausalReport>,
        branching: Option<&BranchingReport>,
        metacognition: Option<&MetacognitionReport>,
    ) -> Option<NarrativeReport> {
        if !self.config.enabled {
            return None;
        }
        let mut entity_ids = collect_entity_ids(behavior_frame, health, survival);
        if entity_ids.is_empty() {
            return None;
        }
        let health_map = map_health(health);
        let survival_map = map_survival(survival);
        let meta_map = map_metacognition(metacognition);
        let motif_map = map_motifs(behavior_frame);
        for entity_id in &entity_ids {
            let overlay = health_map.get(entity_id).copied();
            let survival_entry = survival_map.get(entity_id).copied();
            let risk = risk_score(overlay.map(|entry| entry.score), survival_entry);
            let point = NarrativePoint {
                timestamp,
                health_score: overlay.map(|entry| entry.score),
                survival_score: survival_entry.map(|entry| entry.survival_score),
                risk_score: risk,
                conflict: survival_entry.map(|entry| entry.conflict_in),
                cooperation: survival_entry.map(|entry| entry.cooperation_in),
                play_index: survival_entry.map(|entry| entry.play_index),
                intent: survival_entry.map(|entry| entry.intent_magnitude),
                position: overlay.and_then(|entry| entry.position),
                motif_id: motif_map.get(entity_id).cloned(),
            };
            let history = self
                .history
                .entry(entity_id.clone())
                .or_insert_with(VecDeque::new);
            history.push_back(point);
            while history.len() > self.config.max_history.max(1) {
                history.pop_front();
            }
        }
        entity_ids.sort();

        let mut summaries = Vec::new();
        let mut overall_risk = 0.0;
        let mut overall_count = 0.0;
        for entity_id in entity_ids {
            let overlay = health_map.get(&entity_id).copied();
            let survival_entry = survival_map.get(&entity_id).copied();
            let history = self.history.get(&entity_id);
            let summary = build_entity_summary(
                &self.config,
                &entity_id,
                timestamp,
                overlay,
                survival_entry,
                history,
                temporal,
                causal,
                branching,
                meta_map.get(&entity_id).copied(),
            );
            if let Some(summary) = summary {
                overall_risk += summary.risk_score;
                overall_count += 1.0;
                summaries.push(summary);
            }
        }
        summaries.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap_or(std::cmp::Ordering::Equal));
        summaries.truncate(self.config.max_entities.max(1));
        let overall_risk = if overall_count > 0.0 {
            (overall_risk / overall_count).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let overall_severity = severity_label(overall_risk).to_string();
        let overall_color = Some(grayscale_color((1.0 - overall_risk).clamp(0.0, 1.0)));
        let summary = if let Some(top) = summaries.first() {
            format!(
                "Overall risk is {} (score {:.2}). Highest risk entity {} at {:.2}.",
                overall_severity, overall_risk, top.entity_id, top.risk_score
            )
        } else {
            format!(
                "Overall risk is {} (score {:.2}).",
                overall_severity, overall_risk
            )
        };
        let steps = build_global_steps(timestamp, temporal, causal, branching, self.config.max_steps);
        Some(NarrativeReport {
            timestamp,
            overall_risk,
            overall_severity,
            overall_color,
            summary,
            entities: summaries,
            steps,
        })
    }
}

impl Default for NarrativeRuntime {
    fn default() -> Self {
        Self::new(NarrativeConfig::default())
    }
}

fn collect_entity_ids(
    behavior_frame: Option<&BehaviorFrame>,
    health: Option<&HealthOverlayReport>,
    survival: Option<&SurvivalReport>,
) -> Vec<String> {
    let mut ids = HashSet::new();
    if let Some(report) = health {
        for entry in &report.entities {
            if !entry.entity_id.trim().is_empty() {
                ids.insert(entry.entity_id.clone());
            }
        }
    }
    if let Some(report) = survival {
        for entry in &report.entities {
            if !entry.entity_id.trim().is_empty() {
                ids.insert(entry.entity_id.clone());
            }
        }
    }
    if let Some(frame) = behavior_frame {
        for state in &frame.states {
            if !state.entity_id.trim().is_empty() {
                ids.insert(state.entity_id.clone());
            }
        }
    }
    ids.into_iter().collect()
}

fn map_health(
    report: Option<&HealthOverlayReport>,
) -> HashMap<String, &HealthEntityOverlay> {
    let mut map = HashMap::new();
    if let Some(report) = report {
        for entry in &report.entities {
            map.insert(entry.entity_id.clone(), entry);
        }
    }
    map
}

fn map_survival(
    report: Option<&SurvivalReport>,
) -> HashMap<String, &SurvivalEntityMetrics> {
    let mut map = HashMap::new();
    if let Some(report) = report {
        for entry in &report.entities {
            map.insert(entry.entity_id.clone(), entry);
        }
    }
    map
}

fn map_metacognition(
    report: Option<&MetacognitionReport>,
) -> HashMap<String, &crate::streaming::metacognition_runtime::MetacognitionEntity> {
    let mut map = HashMap::new();
    if let Some(report) = report {
        for entry in &report.entities {
            map.insert(entry.entity_id.clone(), entry);
        }
    }
    map
}

fn map_motifs(frame: Option<&BehaviorFrame>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(frame) = frame {
        for motif in &frame.motifs {
            map.insert(motif.entity_id.clone(), motif.id.clone());
        }
    }
    map
}

fn build_entity_summary(
    config: &NarrativeConfig,
    entity_id: &str,
    now: Timestamp,
    overlay: Option<&HealthEntityOverlay>,
    survival: Option<&SurvivalEntityMetrics>,
    history: Option<&VecDeque<NarrativePoint>>,
    temporal: Option<&TemporalInferenceReport>,
    causal: Option<&CausalReport>,
    branching: Option<&BranchingReport>,
    metacognition: Option<&crate::streaming::metacognition_runtime::MetacognitionEntity>,
) -> Option<NarrativeEntitySummary> {
    let health_score = overlay.map(|entry| entry.score);
    let survival_score = survival.map(|entry| entry.survival_score);
    let risk = risk_score(health_score, survival);
    let severity = severity_label(risk).to_string();
    let health_label = overlay.map(|entry| entry.label.clone());
    let health_color = overlay.map(|entry| entry.color.clone());
    let severity_color = severity_color(risk, health_color.as_deref());
    let position = overlay.and_then(|entry| entry.position);
    let empathy_label = metacognition.and_then(|entry| entry.empathy_label.clone());
    let empathy_confidence = metacognition.and_then(|entry| entry.empathy_confidence);
    let reflection_depth = metacognition.map(|entry| entry.reflection_depth);
    let mut summary_parts = Vec::new();
    if let Some(score) = health_score {
        let label = health_label.clone().unwrap_or_else(|| health_label_from_score(score).to_string());
        summary_parts.push(format!("Health {} (score {:.2})", label, score));
    }
    summary_parts.push(format!("risk {} ({:.2})", severity, risk));
    if let Some(survival_score) = survival_score {
        summary_parts.push(format!("survival {:.2}", survival_score));
    }
    if let Some(delta) = history
        .and_then(|points| risk_window_delta(points, now, config.recent_window_secs))
    {
        summary_parts.push(format!("risk {} ({:+.2})", trend_label(Some(delta)), delta));
    }
    if let Some(entry) = survival {
        if entry.conflict_in >= 0.5 {
            summary_parts.push("conflict elevated".to_string());
        } else if entry.cooperation_in >= 0.6 {
            summary_parts.push("cooperation strong".to_string());
        }
    }
    if let Some(label) = &empathy_label {
        summary_parts.push(format!("empathy {}", label));
    }
    let summary = if summary_parts.is_empty() {
        format!("Entity {} observed with limited telemetry.", entity_id)
    } else {
        format!("Entity {}: {}.", entity_id, summary_parts.join(", "))
    };

    let mut zoom = Vec::new();
    if let Some(history) = history {
        let windows = [
            ("recent".to_string(), config.recent_window_secs),
            ("short_term".to_string(), config.short_window_secs),
            ("session".to_string(), config.long_window_secs),
        ];
        for (label, window) in windows {
            let summary = summarize_window(history, now, window);
            zoom.push(NarrativeZoomSummary {
                label,
                window_secs: window,
                summary,
            });
        }
    }

    let mut steps = Vec::new();
    if let Some(history) = history {
        if let Some(point) = history.back() {
            if let Some(motif_id) = &point.motif_id {
                steps.push(NarrativeStep {
                    timestamp: point.timestamp,
                    text: format!("Observed motif {}", motif_id),
                    confidence: 0.7,
                });
            }
        }
    }
    if let Some(next_event) = temporal.and_then(|report| report.next_event) {
        steps.push(NarrativeStep {
            timestamp: now,
            text: format!("Next likely event {:?}", next_event),
            confidence: 0.6,
        });
    }
    if let Some(causal_report) = causal {
        let mut edges = causal_report.edges.clone();
        edges.sort_by(|a, b| b.weight.abs().partial_cmp(&a.weight.abs()).unwrap_or(std::cmp::Ordering::Equal));
        for edge in edges.into_iter().take(2) {
            steps.push(NarrativeStep {
                timestamp: now,
                text: format!("Causal link {} -> {} (weight {:.2})", edge.source, edge.target, edge.weight),
                confidence: edge.weight.abs().clamp(0.2, 1.0),
            });
        }
    }
    if let Some(report) = branching {
        for ret in report.retrodictions.iter().take(2) {
            let text = ret
                .payload
                .get("event_kind")
                .and_then(|val| val.as_str())
                .map(|kind| format!("Probable missed event {}", kind))
                .unwrap_or_else(|| "Probable missed event".to_string());
            steps.push(NarrativeStep {
                timestamp: ret.timestamp,
                text,
                confidence: ret.confidence.clamp(0.0, 1.0),
            });
        }
    }
    steps.truncate(config.max_steps.max(1));

    let mut evidence = HashMap::new();
    evidence.insert("risk_score".to_string(), Value::from(risk));
    if let Some(score) = health_score {
        evidence.insert("health_score".to_string(), Value::from(score));
    }
    if let Some(score) = survival_score {
        evidence.insert("survival_score".to_string(), Value::from(score));
    }
    if let Some(entry) = survival {
        evidence.insert("conflict".to_string(), Value::from(entry.conflict_in));
        evidence.insert("cooperation".to_string(), Value::from(entry.cooperation_in));
    }
    if let Some(delta) = history
        .and_then(|points| risk_window_delta(points, now, config.recent_window_secs))
    {
        evidence.insert("risk_delta_recent".to_string(), Value::from(delta));
    }

    Some(NarrativeEntitySummary {
        entity_id: entity_id.to_string(),
        severity,
        risk_score: risk,
        health_score,
        survival_score,
        position,
        health_label,
        health_color,
        summary,
        severity_color,
        empathy_label,
        empathy_confidence,
        reflection_depth,
        zoom,
        steps,
        evidence,
    })
}

fn summarize_window(history: &VecDeque<NarrativePoint>, now: Timestamp, window_secs: i64) -> String {
    let start = Timestamp {
        unix: now.unix.saturating_sub(window_secs.max(1)),
    };
    let window: Vec<&NarrativePoint> = history
        .iter()
        .filter(|point| point.timestamp.unix >= start.unix)
        .collect();
    if window.is_empty() {
        return "No recent telemetry for this window.".to_string();
    }
    let first = window.first().unwrap();
    let last = window.last().unwrap();
    let health_delta = delta_optional(first.health_score, last.health_score);
    let survival_delta = delta_optional(first.survival_score, last.survival_score);
    let risk_delta = last.risk_score - first.risk_score;
    let conflict_avg = avg_optional(window.iter().filter_map(|p| p.conflict));
    let coop_avg = avg_optional(window.iter().filter_map(|p| p.cooperation));
    let play_avg = avg_optional(window.iter().filter_map(|p| p.play_index));
    let intent_avg = avg_optional(window.iter().filter_map(|p| p.intent));
    let displacement = position_distance(first.position, last.position);
    let trend_health = trend_label(health_delta);
    let trend_survival = trend_label(survival_delta);
    let samples = window.len();
    let mut parts = Vec::new();
    if let Some(delta) = health_delta {
        parts.push(format!("health {} ({:+.2})", trend_health, delta));
    } else {
        parts.push("health unknown".to_string());
    }
    if let Some(delta) = survival_delta {
        parts.push(format!("survival {} ({:+.2})", trend_survival, delta));
    }
    parts.push(format!("risk {} ({:+.2})", trend_label(Some(risk_delta)), risk_delta));
    if let Some(conflict) = conflict_avg {
        parts.push(format!("conflict {:.2}", conflict));
    }
    if let Some(coop) = coop_avg {
        parts.push(format!("cooperation {:.2}", coop));
    }
    if let Some(play) = play_avg {
        parts.push(format!("play {:.2}", play));
    }
    if let Some(intent) = intent_avg {
        parts.push(format!("intent {:.2}", intent));
    }
    if let Some(distance) = displacement {
        parts.push(format!("movement {:.2}u", distance));
    }
    format!(
        "Window {}s: {} ({} samples).",
        window_secs,
        parts.join(", "),
        samples
    )
}

fn delta_optional(first: Option<f64>, last: Option<f64>) -> Option<f64> {
    match (first, last) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    }
}

fn avg_optional<I>(iter: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0.0;
    for value in iter {
        sum += value;
        count += 1.0;
    }
    if count > 0.0 {
        Some(sum / count)
    } else {
        None
    }
}

fn trend_label(delta: Option<f64>) -> &'static str {
    let Some(delta) = delta else { return "stable"; };
    if delta > 0.05 {
        "up"
    } else if delta < -0.05 {
        "down"
    } else {
        "stable"
    }
}

fn position_distance(a: Option<[f64; 3]>, b: Option<[f64; 3]>) -> Option<f64> {
    let (Some(a), Some(b)) = (a, b) else { return None; };
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    Some((dx * dx + dy * dy + dz * dz).sqrt())
}

fn risk_window_delta(
    history: &VecDeque<NarrativePoint>,
    now: Timestamp,
    window_secs: i64,
) -> Option<f64> {
    let start = Timestamp {
        unix: now.unix.saturating_sub(window_secs.max(1)),
    };
    let mut iter = history
        .iter()
        .filter(|point| point.timestamp.unix >= start.unix);
    let first = iter.next()?;
    let last = iter.last().unwrap_or(first);
    Some(last.risk_score - first.risk_score)
}

fn risk_score(health: Option<f64>, survival: Option<&SurvivalEntityMetrics>) -> f64 {
    let health_risk = health.map(|score| (1.0 - score).clamp(0.0, 1.0)).unwrap_or(0.5);
    let mut risk = health_risk;
    if let Some(entry) = survival {
        let survival_risk = (1.0 - entry.survival_score).clamp(0.0, 1.0);
        let conflict = entry.conflict_in.clamp(0.0, 1.0);
        let cooperation = entry.cooperation_in.clamp(0.0, 1.0);
        risk = 0.55 * health_risk + 0.25 * survival_risk + 0.25 * conflict - 0.1 * cooperation;
    }
    risk.clamp(0.0, 1.0)
}

fn severity_label(score: f64) -> &'static str {
    if score >= 0.75 {
        "critical"
    } else if score >= 0.55 {
        "elevated"
    } else if score >= 0.35 {
        "guarded"
    } else {
        "low"
    }
}

fn health_label_from_score(score: f64) -> &'static str {
    if score >= 0.85 {
        "optimal"
    } else if score >= 0.6 {
        "stable"
    } else if score >= 0.35 {
        "strained"
    } else {
        "critical"
    }
}

fn severity_color(risk: f64, health_color: Option<&str>) -> Option<String> {
    if let Some(color) = health_color {
        return Some(color.to_string());
    }
    Some(grayscale_color((1.0 - risk).clamp(0.0, 1.0)))
}

fn grayscale_color(score: f64) -> String {
    let value = (score.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{value:02X}{value:02X}{value:02X}")
}

fn build_global_steps(
    now: Timestamp,
    temporal: Option<&TemporalInferenceReport>,
    causal: Option<&CausalReport>,
    branching: Option<&BranchingReport>,
    max_steps: usize,
) -> Vec<NarrativeStep> {
    let mut steps = Vec::new();
    if let Some(report) = temporal {
        if let Some(event) = report.next_event {
            steps.push(NarrativeStep {
                timestamp: report.timestamp,
                text: format!("System next event forecast {:?}", event),
                confidence: 0.6,
            });
        }
        if let Some(hyper) = &report.hypergraph {
            steps.push(NarrativeStep {
                timestamp: report.timestamp,
                text: format!(
                    "Hypergraph updated ({} nodes, {} edges)",
                    hyper.nodes, hyper.edges
                ),
                confidence: 0.4,
            });
        }
    }
    if let Some(report) = causal {
        let mut edges = report.edges.clone();
        edges.sort_by(|a, b| b.weight.abs().partial_cmp(&a.weight.abs()).unwrap_or(std::cmp::Ordering::Equal));
        for edge in edges.into_iter().take(2) {
            steps.push(NarrativeStep {
                timestamp: report.timestamp,
                text: format!("Causal driver {} -> {} (weight {:.2})", edge.source, edge.target, edge.weight),
                confidence: edge.weight.abs().clamp(0.2, 1.0),
            });
        }
    }
    if let Some(report) = branching {
        for branch in report.branches.iter().take(2) {
            let event = branch
                .payload
                .get("event_kind")
                .and_then(|val| val.as_str())
                .unwrap_or("event");
            let intensity = branch
                .payload
                .get("intensity")
                .and_then(|val| val.as_f64())
                .unwrap_or(0.0);
            steps.push(NarrativeStep {
                timestamp: branch.timestamp,
                text: format!(
                    "Future branch {} prob {:.2} (intensity {:.2})",
                    event, branch.probability, intensity
                ),
                confidence: (1.0 - branch.uncertainty).clamp(0.0, 1.0),
            });
        }
    }
    if steps.is_empty() {
        steps.push(NarrativeStep {
            timestamp: now,
            text: "No additional narrative context available.".to_string(),
            confidence: 0.2,
        });
    }
    steps.truncate(max_steps.max(1));
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::health_overlay::HealthEntityOverlay;
    use crate::streaming::survival::SurvivalEntityMetrics;

    #[test]
    fn narrative_report_emits_summary() {
        let mut runtime = NarrativeRuntime::default();
        let health = HealthOverlayReport {
            timestamp: Timestamp { unix: 10 },
            entities: vec![HealthEntityOverlay {
                entity_id: "e1".to_string(),
                position: Some([1.0, 2.0, 0.0]),
                score: 0.4,
                color: "#444444".to_string(),
                label: "strained".to_string(),
                dimensions: Vec::new(),
                survival_score: None,
                cooperation: None,
                conflict: None,
                play_index: None,
                baseline_score: None,
            }],
            palette: Vec::new(),
        };
        let survival = SurvivalReport {
            timestamp: Timestamp { unix: 10 },
            entities: vec![SurvivalEntityMetrics {
                entity_id: "e1".to_string(),
                phenotype_key: None,
                survival_score: 0.5,
                baseline_score: 0.6,
                cooperation_in: 0.2,
                conflict_in: 0.7,
                play_index: 0.1,
                intent_magnitude: 0.4,
                physiology_score: None,
            }],
            interactions: Vec::new(),
        };
        let report = runtime.update(
            Timestamp { unix: 10 },
            None,
            Some(&health),
            Some(&survival),
            None,
            None,
            None,
            None,
        );
        assert!(report.is_some());
        let report = report.unwrap();
        assert_eq!(report.entities.len(), 1);
        assert!(!report.summary.is_empty());
        assert!(report.entities[0].severity_color.is_some());
    }
}
