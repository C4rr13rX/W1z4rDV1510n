use crate::config::PhysiologyConfig;
use crate::schema::Timestamp;
use crate::streaming::knowledge::HealthKnowledgeStore;
use crate::streaming::schema::{EventToken, LayerKind, LayerState, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

const FEATURE_COUNT: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologyDeviation {
    pub context: String,
    pub deviation_index: f64,
    pub deviation_vector: Vec<f64>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologyReport {
    pub timestamp: Timestamp,
    pub deviations: Vec<PhysiologyDeviation>,
    pub overall_index: f64,
    pub template_count: usize,
}

struct PhysiologyTemplate {
    mean: Vec<f64>,
    variance: Vec<f64>,
    samples: usize,
    cohort_key: Option<String>,
    last_update: Timestamp,
}

#[derive(Default)]
struct FeatureAccumulator {
    sums: Vec<f64>,
    counts: Vec<usize>,
}

impl FeatureAccumulator {
    fn new() -> Self {
        Self {
            sums: vec![0.0; FEATURE_COUNT],
            counts: vec![0; FEATURE_COUNT],
        }
    }

    fn add(&mut self, index: usize, value: f64) {
        if index >= FEATURE_COUNT {
            return;
        }
        self.sums[index] += value;
        self.counts[index] += 1;
    }

    fn finalize(&self) -> Vec<f64> {
        let mut output = vec![0.0; FEATURE_COUNT];
        for idx in 0..FEATURE_COUNT {
            let count = self.counts[idx];
            if count > 0 {
                output[idx] = self.sums[idx] / count as f64;
            }
        }
        output
    }
}

struct ContextMeta {
    cohort_key: Option<String>,
    phenotype_key: Option<String>,
}

pub struct PhysiologyRuntime {
    config: PhysiologyConfig,
    templates: HashMap<String, PhysiologyTemplate>,
    template_order: VecDeque<String>,
}

impl PhysiologyRuntime {
    pub fn new(config: PhysiologyConfig) -> Self {
        Self {
            config,
            templates: HashMap::new(),
            template_order: VecDeque::new(),
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        knowledge: Option<&HealthKnowledgeStore>,
    ) -> Option<PhysiologyReport> {
        if !self.config.enabled {
            return None;
        }
        let (features, meta) = collect_features(batch);
        if features.is_empty() {
            return None;
        }
        let mut deviations = Vec::new();
        let mut overall = 0.0;
        let mut count = 0.0;
        for (context, accumulator) in features {
            let vector = accumulator.finalize();
            let cohort_key = meta.get(&context).and_then(|m| m.cohort_key.clone());
            let phenotype_key = meta.get(&context).and_then(|m| m.phenotype_key.clone());
            let template = self.templates.get_mut(&context);
            if template.is_none() {
                let mean = apply_prior(
                    &vector,
                    cohort_key.as_deref(),
                    phenotype_key.as_deref(),
                    &self.templates,
                    self.config.prior_strength,
                    knowledge,
                );
                let variance = vec![self.config.covariance_floor.max(1e-6); FEATURE_COUNT];
                let template = PhysiologyTemplate {
                    mean,
                    variance,
                    samples: 1,
                    cohort_key: cohort_key.clone(),
                    last_update: batch.timestamp,
                };
                self.templates.insert(context.clone(), template);
                self.template_order.push_back(context.clone());
            }
            let template = self
                .templates
                .get_mut(&context)
                .expect("template exists");
            let (index, deviation_vector) =
                deviation_index(&vector, &template.mean, &template.variance, self.config.covariance_floor);
            let allow_update = template.samples < self.config.min_samples
                || index <= self.config.max_deviation_update;
            if allow_update {
                update_template(
                    template,
                    &vector,
                    self.config.update_alpha,
                    self.config.covariance_floor,
                    batch.timestamp,
                );
            }
            deviations.push(PhysiologyDeviation {
                context: context.clone(),
                deviation_index: index,
                deviation_vector,
                sample_count: template.samples,
            });
            overall += index;
            count += 1.0;
        }
        self.prune_templates();
        let overall_index = if count > 0.0 {
            overall / count
        } else {
            0.0
        };
        Some(PhysiologyReport {
            timestamp: batch.timestamp,
            deviations,
            overall_index,
            template_count: self.templates.len(),
        })
    }

    fn prune_templates(&mut self) {
        let max_templates = self.config.max_templates.max(1);
        while self.templates.len() > max_templates {
            if let Some(oldest) = self.template_order.pop_front() {
                self.templates.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

fn collect_features(batch: &TokenBatch) -> (HashMap<String, FeatureAccumulator>, HashMap<String, ContextMeta>) {
    let mut features: HashMap<String, FeatureAccumulator> = HashMap::new();
    let mut meta: HashMap<String, ContextMeta> = HashMap::new();
    for token in &batch.tokens {
        let context = context_from_attrs(&token.attributes);
        let entry = features.entry(context.clone()).or_insert_with(FeatureAccumulator::new);
        let cohort_key = cohort_from_attrs(&token.attributes);
        let phenotype_key = phenotype_from_attrs(&token.attributes);
        meta.entry(context.clone())
            .and_modify(|m| {
                if m.cohort_key.is_none() {
                    m.cohort_key = cohort_key.clone();
                }
                if m.phenotype_key.is_none() {
                    m.phenotype_key = phenotype_key.clone();
                }
            })
            .or_insert(ContextMeta { cohort_key, phenotype_key });
        apply_token_features(entry, token);
    }
    for layer in &batch.layers {
        let context = context_from_attrs(&layer.attributes);
        let entry = features.entry(context.clone()).or_insert_with(FeatureAccumulator::new);
        let phenotype_key = phenotype_from_attrs(&layer.attributes);
        meta.entry(context.clone())
            .and_modify(|m| {
                if m.phenotype_key.is_none() {
                    m.phenotype_key = phenotype_key.clone();
                }
            })
            .or_insert(ContextMeta {
                cohort_key: None,
                phenotype_key,
            });
        apply_layer_features(entry, layer);
    }
    (features, meta)
}

fn context_from_attrs(attrs: &HashMap<String, Value>) -> String {
    attrs
        .get("entity_id")
        .and_then(|v| v.as_str())
        .filter(|val| !val.trim().is_empty())
        .map(|val| val.to_string())
        .unwrap_or_else(|| "global".to_string())
}

fn cohort_from_attrs(attrs: &HashMap<String, Value>) -> Option<String> {
    for key in ["cohort_id", "cohort", "age_bucket"] {
        if let Some(val) = attrs.get(key).and_then(|v| v.as_str()) {
            if !val.trim().is_empty() {
                return Some(val.to_string());
            }
        }
    }
    None
}

fn phenotype_from_attrs(attrs: &HashMap<String, Value>) -> Option<String> {
    if let Some(val) = attrs.get("phenotype_key").and_then(|v| v.as_str()) {
        if !val.trim().is_empty() {
            return Some(val.to_string());
        }
    }
    let mut parts = Vec::new();
    for key in ["species", "phenotype", "size_class", "age_bucket", "cohort_id", "genotype"] {
        if let Some(val) = attrs.get(key).and_then(|v| v.as_str()) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
    }
    if parts.len() > 1 {
        Some(parts.join("|"))
    } else {
        None
    }
}

fn apply_layer_features(acc: &mut FeatureAccumulator, layer: &LayerState) {
    let (amp, coh) = (layer.amplitude.clamp(0.0, 1.0), layer.coherence.clamp(0.0, 1.0));
    match layer.kind {
        LayerKind::UltradianMicroArousal => {
            acc.add(0, amp);
            acc.add(1, coh);
        }
        LayerKind::UltradianBrac => {
            acc.add(2, amp);
            acc.add(3, coh);
        }
        LayerKind::UltradianMeso => {
            acc.add(4, amp);
            acc.add(5, coh);
        }
        _ => {}
    }
}

fn apply_token_features(acc: &mut FeatureAccumulator, token: &EventToken) {
    let is_motor = token
        .attributes
        .get("atom_type")
        .and_then(|v| v.as_str())
        .map(|val| val == "motor")
        .unwrap_or(false);
    if !is_motor {
        return;
    }
    if let Some(val) = token.attributes.get("motor_signal_norm").and_then(|v| v.as_f64()) {
        acc.add(6, val.clamp(0.0, 1.0));
    } else if let Some(val) = token.attributes.get("motor_signal").and_then(|v| v.as_f64()) {
        acc.add(6, val.clamp(0.0, 1.0));
    }
    if let Some(val) = token.attributes.get("motion_energy").and_then(|v| v.as_f64()) {
        acc.add(7, val.clamp(0.0, 1.0));
    }
    if let Some(val) = token.attributes.get("posture_shift").and_then(|v| v.as_f64()) {
        acc.add(8, val.clamp(0.0, 1.0));
    }
    if let Some(val) = token.attributes.get("micro_jitter").and_then(|v| v.as_f64()) {
        acc.add(9, val.clamp(0.0, 1.0));
    }
}

fn apply_prior(
    vector: &[f64],
    cohort_key: Option<&str>,
    phenotype_key: Option<&str>,
    templates: &HashMap<String, PhysiologyTemplate>,
    prior_strength: f64,
    knowledge: Option<&HealthKnowledgeStore>,
) -> Vec<f64> {
    let Some(cohort) = cohort_key else {
        return knowledge
            .map(|store| store.apply_physiology_prior(phenotype_key, vector, prior_strength))
            .unwrap_or_else(|| vector.to_vec());
    };
    let mut prior = vec![0.0; FEATURE_COUNT];
    let mut count = 0.0;
    for template in templates.values() {
        if template.cohort_key.as_deref() != Some(cohort) {
            continue;
        }
        for (idx, val) in template.mean.iter().enumerate() {
            prior[idx] += *val;
        }
        count += 1.0;
    }
    if count <= 0.0 {
        return vector.to_vec();
    }
    for val in prior.iter_mut() {
        *val /= count;
    }
    let strength = prior_strength.clamp(0.0, 1.0);
    let mut blended = vec![0.0; FEATURE_COUNT];
    for idx in 0..FEATURE_COUNT {
        blended[idx] = vector[idx] * (1.0 - strength) + prior[idx] * strength;
    }
    knowledge
        .map(|store| store.apply_physiology_prior(phenotype_key, &blended, prior_strength))
        .unwrap_or(blended)
}

fn update_template(
    template: &mut PhysiologyTemplate,
    vector: &[f64],
    alpha: f64,
    cov_floor: f64,
    timestamp: Timestamp,
) {
    let alpha = alpha.clamp(0.0, 1.0);
    for idx in 0..FEATURE_COUNT {
        let diff = vector[idx] - template.mean[idx];
        template.mean[idx] += alpha * diff;
        let var = (1.0 - alpha) * template.variance[idx] + alpha * diff * diff;
        template.variance[idx] = var.max(cov_floor);
    }
    template.samples += 1;
    template.last_update = timestamp;
}

fn deviation_index(
    vector: &[f64],
    mean: &[f64],
    variance: &[f64],
    cov_floor: f64,
) -> (f64, Vec<f64>) {
    let mut sum = 0.0;
    let mut deviations = vec![0.0; FEATURE_COUNT];
    for idx in 0..FEATURE_COUNT {
        let var = variance[idx].max(cov_floor);
        let diff = vector[idx] - mean[idx];
        let z = diff / var.sqrt();
        deviations[idx] = z;
        sum += z * z;
    }
    (sum.sqrt(), deviations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, LayerState, StreamSource};

    #[test]
    fn physiology_runtime_reports_deviation() {
        let mut config = PhysiologyConfig::default();
        config.enabled = true;
        let mut runtime = PhysiologyRuntime::new(config);
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 100 },
                duration_secs: 1.0,
                confidence: 0.9,
                attributes: HashMap::from([
                    ("entity_id".to_string(), Value::String("e1".to_string())),
                    ("atom_type".to_string(), Value::String("motor".to_string())),
                    ("motor_signal_norm".to_string(), Value::from(0.6)),
                    ("motion_energy".to_string(), Value::from(0.5)),
                ]),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 100 },
                phase: 0.1,
                amplitude: 0.7,
                coherence: 0.4,
                attributes: HashMap::from([("entity_id".to_string(), Value::String("e1".to_string()))]),
            }],
            source_confidence: HashMap::new(),
        };
        let report = runtime.update(&batch, None).expect("report");
        assert_eq!(report.template_count, 1);
        assert_eq!(report.deviations.len(), 1);
    }
}
