use crate::schema::Timestamp;
use crate::streaming::behavior::{BehaviorFrame, BehaviorState, SpeciesKind};
use crate::streaming::knowledge::HealthKnowledgeStore;
use crate::streaming::physiology_runtime::PhysiologyReport;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalInteraction {
    pub source: String,
    pub target: String,
    pub proximity: f64,
    pub intent_alignment: f64,
    pub force: f64,
    pub cooperation: f64,
    pub conflict: f64,
    pub play_index: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalEntityMetrics {
    pub entity_id: String,
    #[serde(default)]
    pub phenotype_key: Option<String>,
    pub survival_score: f64,
    pub baseline_score: f64,
    pub cooperation_in: f64,
    pub conflict_in: f64,
    pub play_index: f64,
    pub intent_magnitude: f64,
    #[serde(default)]
    pub physiology_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalReport {
    pub timestamp: Timestamp,
    pub entities: Vec<SurvivalEntityMetrics>,
    pub interactions: Vec<SurvivalInteraction>,
}

#[derive(Debug, Clone)]
pub struct SurvivalConfig {
    pub physiology_weight: f64,
    pub cooperation_weight: f64,
    pub conflict_weight: f64,
    pub play_weight: f64,
    pub baseline_alpha: f64,
    pub conflict_update_threshold: f64,
    pub min_baseline_samples: usize,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            physiology_weight: 0.45,
            cooperation_weight: 0.25,
            conflict_weight: 0.2,
            play_weight: 0.1,
            baseline_alpha: 0.2,
            conflict_update_threshold: 0.6,
            min_baseline_samples: 6,
        }
    }
}

#[derive(Debug, Clone)]
struct SurvivalBaseline {
    mean: f64,
    var: f64,
    samples: usize,
    last_update: Timestamp,
}

impl SurvivalBaseline {
    fn new(score: f64, ts: Timestamp) -> Self {
        Self {
            mean: score,
            var: 0.0,
            samples: 1,
            last_update: ts,
        }
    }

    fn update(&mut self, score: f64, alpha: f64, ts: Timestamp) {
        let alpha = alpha.clamp(0.0, 1.0);
        let diff = score - self.mean;
        self.mean += alpha * diff;
        self.var = (1.0 - alpha) * (self.var + alpha * diff * diff);
        self.samples += 1;
        self.last_update = ts;
    }
}

pub struct SurvivalRuntime {
    config: SurvivalConfig,
    baselines: HashMap<String, SurvivalBaseline>,
    phenotype_baselines: HashMap<String, SurvivalBaseline>,
}

impl SurvivalRuntime {
    pub fn new(config: SurvivalConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
            phenotype_baselines: HashMap::new(),
        }
    }

    pub fn update(
        &mut self,
        frame: &BehaviorFrame,
        physiology: Option<&PhysiologyReport>,
        knowledge: Option<&HealthKnowledgeStore>,
    ) -> SurvivalReport {
        let physiology_scores = physiology_scores(physiology);
        let intents = compute_intents(&frame.states);
        let mut interactions = Vec::new();
        let mut coop_in: HashMap<String, f64> = HashMap::new();
        let mut conflict_in: HashMap<String, f64> = HashMap::new();
        let mut play_in: HashMap<String, f64> = HashMap::new();

        for edge in &frame.graph.edges {
            let Some(source) = intents.get(&edge.source) else { continue };
            let Some(target) = intents.get(&edge.target) else { continue };
            let proximity = edge.proximity.unwrap_or(0.0).clamp(0.0, 1.0);
            let alignment = intent_alignment(&source.vector, &target.vector);
            let force = proximity * (source.magnitude + target.magnitude) * 0.5;
            let coherence = edge.coupling.coherence.clamp(0.0, 1.0);
            let cooperation = (alignment * coherence).clamp(0.0, 1.0);
            let conflict = ((1.0 - alignment) * force).clamp(0.0, 1.0);
            let play_index = (force * alignment * coherence).clamp(0.0, 1.0);

            *coop_in.entry(edge.target.clone()).or_insert(0.0) += cooperation;
            *conflict_in.entry(edge.target.clone()).or_insert(0.0) += conflict;
            *play_in.entry(edge.target.clone()).or_insert(0.0) += play_index;

            interactions.push(SurvivalInteraction {
                source: edge.source.clone(),
                target: edge.target.clone(),
                proximity,
                intent_alignment: alignment,
                force,
                cooperation,
                conflict,
                play_index,
            });
        }

        let mut entities = Vec::new();
        for state in &frame.states {
            let intent = intents.get(&state.entity_id);
            let coop = normalize_sum(coop_in.get(&state.entity_id).copied().unwrap_or(0.0));
            let conflict = normalize_sum(conflict_in.get(&state.entity_id).copied().unwrap_or(0.0));
            let play = normalize_sum(play_in.get(&state.entity_id).copied().unwrap_or(0.0));
            let phys = physiology_scores.get(&state.entity_id).copied();
            let physiology_score = phys.unwrap_or(0.5);
            let phenotype_key = phenotype_key(state);
            let mut survival_score = combine_scores(
                physiology_score,
                coop,
                conflict,
                play,
                &self.config,
            );
            survival_score = knowledge
                .map(|store| store.apply_survival_bias(phenotype_key.as_deref(), survival_score))
                .unwrap_or(survival_score);
            let baseline = self
                .baselines
                .entry(state.entity_id.clone())
                .or_insert_with(|| SurvivalBaseline::new(survival_score, frame.timestamp));
            let allow_update = conflict <= self.config.conflict_update_threshold;
            if allow_update {
                baseline.update(survival_score, self.config.baseline_alpha, frame.timestamp);
            }
            let mut baseline_score = baseline.mean;
            if let Some(key) = phenotype_key.clone() {
                let entry = self
                    .phenotype_baselines
                    .entry(key.clone())
                    .or_insert_with(|| SurvivalBaseline::new(survival_score, frame.timestamp));
                if allow_update {
                    entry.update(survival_score, self.config.baseline_alpha, frame.timestamp);
                }
                baseline_score = blend_baseline(baseline_score, entry.mean, entry.samples);
            }

            entities.push(SurvivalEntityMetrics {
                entity_id: state.entity_id.clone(),
                phenotype_key,
                survival_score,
                baseline_score,
                cooperation_in: coop,
                conflict_in: conflict,
                play_index: play,
                intent_magnitude: intent.map(|val| val.magnitude).unwrap_or(0.0),
                physiology_score: phys,
            });
        }

        SurvivalReport {
            timestamp: frame.timestamp,
            entities,
            interactions,
        }
    }
}

impl Default for SurvivalRuntime {
    fn default() -> Self {
        Self::new(SurvivalConfig::default())
    }
}

#[derive(Debug, Clone)]
struct IntentMetrics {
    vector: Vec<f64>,
    magnitude: f64,
}

fn compute_intents(states: &[BehaviorState]) -> HashMap<String, IntentMetrics> {
    let mut map = HashMap::new();
    for state in states {
        let vector = if !state.action.is_empty() {
            state.action.clone()
        } else {
            state.latent.clone()
        };
        let magnitude = vector_norm(&vector).clamp(0.0, 5.0);
        map.insert(
            state.entity_id.clone(),
            IntentMetrics { vector, magnitude },
        );
    }
    map
}

fn intent_alignment(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a <= 1e-6 || norm_b <= 1e-6 {
        return 0.5;
    }
    let cos = (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0);
    ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|val| val * val).sum::<f64>().sqrt()
}

fn normalize_sum(value: f64) -> f64 {
    (value / (1.0 + value)).clamp(0.0, 1.0)
}

fn combine_scores(
    physiology: f64,
    cooperation: f64,
    conflict: f64,
    play: f64,
    config: &SurvivalConfig,
) -> f64 {
    let physiology = physiology.clamp(0.0, 1.0);
    let cooperation = cooperation.clamp(0.0, 1.0);
    let conflict_score = (1.0 - conflict).clamp(0.0, 1.0);
    let play = play.clamp(0.0, 1.0);
    let weight_sum = config.physiology_weight
        + config.cooperation_weight
        + config.conflict_weight
        + config.play_weight;
    if weight_sum <= 0.0 {
        return physiology;
    }
    let score = physiology * config.physiology_weight
        + cooperation * config.cooperation_weight
        + conflict_score * config.conflict_weight
        + play * config.play_weight;
    (score / weight_sum).clamp(0.0, 1.0)
}

fn physiology_scores(report: Option<&PhysiologyReport>) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let Some(report) = report else {
        return map;
    };
    for deviation in &report.deviations {
        let score = (1.0 / (1.0 + deviation.deviation_index.abs())).clamp(0.0, 1.0);
        map.insert(deviation.context.clone(), score);
    }
    map
}

fn phenotype_key(state: &BehaviorState) -> Option<String> {
    let mut parts = vec![species_label(state.species.clone())];
    for key in ["phenotype", "size_class", "age_bucket", "cohort_id", "genotype"] {
        if let Some(val) = state.attributes.get(key).and_then(|v| v.as_str()) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
    }
    if parts.len() <= 1 {
        return None;
    }
    Some(parts.join("|"))
}

fn species_label(species: SpeciesKind) -> String {
    match species {
        SpeciesKind::Human => "HUMAN".to_string(),
        SpeciesKind::Canine => "CANINE".to_string(),
        SpeciesKind::Feline => "FELINE".to_string(),
        SpeciesKind::Avian => "AVIAN".to_string(),
        SpeciesKind::Aquatic => "AQUATIC".to_string(),
        SpeciesKind::Insect => "INSECT".to_string(),
        SpeciesKind::Other(raw) => raw,
    }
}

fn blend_baseline(entity: f64, phenotype: f64, phenotype_samples: usize) -> f64 {
    if phenotype_samples == 0 {
        return entity;
    }
    let weight = (phenotype_samples as f64 / (phenotype_samples as f64 + 10.0)).clamp(0.0, 0.7);
    (entity * (1.0 - weight) + phenotype * weight).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::behavior::{BehaviorEdge, BehaviorGraph, CouplingMetrics};
    use std::collections::HashMap;

    #[test]
    fn survival_reports_cooperation_from_alignment() {
        let state_a = BehaviorState {
            entity_id: "a".to_string(),
            timestamp: Timestamp { unix: 1 },
            species: SpeciesKind::Human,
            latent: vec![0.2, 0.3],
            action: vec![0.5, 0.5],
            position: Some([0.0, 0.0, 0.0]),
            confidence: 1.0,
            missing_ratio: 0.0,
            attributes: HashMap::new(),
        };
        let state_b = BehaviorState {
            entity_id: "b".to_string(),
            timestamp: Timestamp { unix: 1 },
            species: SpeciesKind::Human,
            latent: vec![0.2, 0.3],
            action: vec![0.5, 0.5],
            position: Some([1.0, 0.0, 0.0]),
            confidence: 1.0,
            missing_ratio: 0.0,
            attributes: HashMap::new(),
        };
        let coupling = CouplingMetrics {
            coherence: 0.8,
            phase_locking: None,
            transfer_entropy: None,
            confidence: 0.8,
        };
        let graph = BehaviorGraph {
            timestamp: Timestamp { unix: 1 },
            nodes: HashMap::from([
                ("a".to_string(), state_a.clone()),
                ("b".to_string(), state_b.clone()),
            ]),
            edges: vec![
                BehaviorEdge {
                    source: "a".to_string(),
                    target: "b".to_string(),
                    proximity: Some(0.7),
                    coupling: coupling.clone(),
                },
                BehaviorEdge {
                    source: "b".to_string(),
                    target: "a".to_string(),
                    proximity: Some(0.7),
                    coupling,
                },
            ],
        };
        let frame = BehaviorFrame {
            timestamp: Timestamp { unix: 1 },
            states: vec![state_a, state_b],
            graph,
            motifs: Vec::new(),
            prediction: None,
            backpressure: crate::streaming::behavior::BackpressureStatus::Ok,
        };
        let mut runtime = SurvivalRuntime::default();
        let report = runtime.update(&frame, None, None);
        assert_eq!(report.entities.len(), 2);
        let coop = report
            .entities
            .iter()
            .find(|e| e.entity_id == "a")
            .unwrap()
            .cooperation_in;
        assert!(coop > 0.0);
    }
}
