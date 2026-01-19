use crate::blockchain::WorkKind;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub block_id: String,
    pub text: String,
    #[serde(default)]
    pub section: Option<String>,
    #[serde(default)]
    pub order: usize,
    #[serde(default)]
    pub figure_refs: Vec<String>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigureAsset {
    pub figure_id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub caption: Option<String>,
    pub image_ref: String,
    pub image_hash: String,
    #[serde(default)]
    pub order: usize,
    #[serde(default)]
    pub ocr_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDocument {
    pub doc_id: String,
    pub source: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub text_blocks: Vec<TextBlock>,
    #[serde(default)]
    pub figures: Vec<FigureAsset>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCandidate {
    pub block_id: String,
    pub text_excerpt: String,
    pub text_hash: String,
    #[serde(default)]
    pub match_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AssociationStatus {
    Pending,
    Verified,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationVote {
    pub task_id: String,
    pub worker_id: String,
    pub selected_block_id: String,
    pub text_hash: String,
    pub image_hash: String,
    pub confidence: f64,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigureAssociationTask {
    pub task_id: String,
    pub work_id: String,
    pub work_kind: WorkKind,
    pub doc_id: String,
    pub figure_id: String,
    pub image_ref: String,
    pub image_hash: String,
    #[serde(default)]
    pub candidates: Vec<TextCandidate>,
    pub created_at: Timestamp,
    pub status: AssociationStatus,
    pub reward_score: f64,
    #[serde(default)]
    pub votes: Vec<AssociationVote>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeAssociation {
    pub association_id: String,
    pub doc_id: String,
    pub figure_id: String,
    pub text_block_id: String,
    pub text_hash: String,
    pub image_hash: String,
    pub confidence: f64,
    pub verified_by: usize,
    pub verified_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeIngestReport {
    pub doc_id: String,
    pub tasks_created: usize,
    pub pending_tasks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeQueueReport {
    pub timestamp: Timestamp,
    pub pending: Vec<FigureAssociationTask>,
    pub total_pending: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeQueueConfig {
    pub min_votes: usize,
    pub min_confidence: f64,
    pub max_pending: usize,
    pub candidate_limit: usize,
    pub reward_base: f64,
    pub reward_per_candidate: f64,
}

impl Default for KnowledgeQueueConfig {
    fn default() -> Self {
        Self {
            min_votes: 2,
            min_confidence: 0.85,
            max_pending: 256,
            candidate_limit: 6,
            reward_base: 0.8,
            reward_per_candidate: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeQueue {
    config: KnowledgeQueueConfig,
    documents: HashMap<String, KnowledgeDocument>,
    tasks: HashMap<String, FigureAssociationTask>,
    pending: VecDeque<String>,
}

impl KnowledgeQueue {
    pub fn new(config: KnowledgeQueueConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            tasks: HashMap::new(),
            pending: VecDeque::new(),
        }
    }

    pub fn enqueue_document(&mut self, doc: KnowledgeDocument, timestamp: Timestamp) -> KnowledgeIngestReport {
        let doc_id = doc.doc_id.clone();
        let tasks = build_tasks(&doc, timestamp, &self.config);
        let tasks_created = tasks.len();
        self.documents.insert(doc_id.clone(), doc);
        for task in tasks {
            if self.tasks.len() >= self.config.max_pending {
                break;
            }
            let task_id = task.task_id.clone();
            self.tasks.insert(task_id.clone(), task);
            self.pending.push_back(task_id);
        }
        KnowledgeIngestReport {
            doc_id,
            tasks_created,
            pending_tasks: self.pending.len(),
        }
    }

    pub fn record_vote(&mut self, vote: AssociationVote) -> Option<KnowledgeAssociation> {
        let task_id = vote.task_id.clone();
        let association = {
            let task = self.tasks.get_mut(&task_id)?;
            if !matches!(task.status, AssociationStatus::Pending) {
                return None;
            }
            if task.image_hash != vote.image_hash {
                return None;
            }
            if !task.candidates.iter().any(|cand| {
                cand.block_id == vote.selected_block_id && cand.text_hash == vote.text_hash
            }) {
                return None;
            }
            task.votes.push(vote);
            maybe_verify_task(task, &self.config)
        };
        if association.is_some() {
            self.pending.retain(|id| id != &task_id);
        }
        association
    }

    pub fn pending_report(&self, timestamp: Timestamp) -> Option<KnowledgeQueueReport> {
        if self.pending.is_empty() {
            return None;
        }
        let mut pending_tasks = Vec::new();
        for task_id in &self.pending {
            if let Some(task) = self.tasks.get(task_id) {
                if matches!(task.status, AssociationStatus::Pending) {
                    pending_tasks.push(task.clone());
                }
            }
        }
        if pending_tasks.is_empty() {
            return None;
        }
        Some(KnowledgeQueueReport {
            timestamp,
            pending: pending_tasks,
            total_pending: self.pending.len(),
        })
    }

    pub fn document(&self, doc_id: &str) -> Option<&KnowledgeDocument> {
        self.documents.get(doc_id)
    }

}

impl Default for KnowledgeQueue {
    fn default() -> Self {
        Self::new(KnowledgeQueueConfig::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhenotypePrior {
    pub phenotype_key: String,
    pub survival_bias: f64,
    #[serde(default)]
    pub physiology_mean: Option<Vec<f64>>,
    pub confidence: f64,
    #[serde(default)]
    pub sources: Vec<String>,
    pub updated_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeStoreReport {
    pub timestamp: Timestamp,
    pub phenotype_priors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HealthKnowledgeStore {
    phenotype_priors: HashMap<String, PhenotypePrior>,
    prior_alpha: f64,
}

impl HealthKnowledgeStore {
    pub fn new(prior_alpha: f64) -> Self {
        Self {
            phenotype_priors: HashMap::new(),
            prior_alpha: prior_alpha.clamp(0.0, 1.0),
        }
    }

    pub fn ingest_association(
        &mut self,
        association: &KnowledgeAssociation,
        document: &KnowledgeDocument,
    ) {
        let Some(phenotype_key) = phenotype_key_from_meta(&document.metadata) else {
            return;
        };
        let survival_bias = document
            .metadata
            .get("survival_bias")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .clamp(-0.5, 0.5);
        let physiology_mean = document
            .metadata
            .get("physiology_mean")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|val| val.as_f64()).collect::<Vec<_>>())
            .filter(|vec| !vec.is_empty());
        let confidence = association.confidence.clamp(0.0, 1.0);
        let entry = self
            .phenotype_priors
            .entry(phenotype_key.clone())
            .or_insert_with(|| PhenotypePrior {
                phenotype_key: phenotype_key.clone(),
                survival_bias,
                physiology_mean: physiology_mean.clone(),
                confidence,
                sources: Vec::new(),
                updated_at: association.verified_at,
            });
        entry.survival_bias = blend(entry.survival_bias, survival_bias, self.prior_alpha * confidence);
        if let Some(mean) = physiology_mean {
            if let Some(existing) = &entry.physiology_mean {
                entry.physiology_mean = Some(blend_vec(existing, &mean, self.prior_alpha * confidence));
            } else {
                entry.physiology_mean = Some(mean);
            }
        }
        entry.confidence = blend(entry.confidence, confidence, self.prior_alpha);
        entry.updated_at = association.verified_at;
        let source_id = format!("{}:{}", document.doc_id, association.figure_id);
        if !entry.sources.contains(&source_id) {
            entry.sources.push(source_id);
        }
    }

    pub fn apply_survival_bias(&self, phenotype_key: Option<&str>, score: f64) -> f64 {
        let Some(key) = phenotype_key else {
            return score;
        };
        let Some(prior) = self.phenotype_priors.get(key) else {
            return score;
        };
        let adjustment = prior.survival_bias * prior.confidence;
        (score + adjustment).clamp(0.0, 1.0)
    }

    pub fn apply_physiology_prior(
        &self,
        phenotype_key: Option<&str>,
        vector: &[f64],
        strength: f64,
    ) -> Vec<f64> {
        let Some(key) = phenotype_key else {
            return vector.to_vec();
        };
        let Some(prior) = self.phenotype_priors.get(key) else {
            return vector.to_vec();
        };
        let Some(mean) = &prior.physiology_mean else {
            return vector.to_vec();
        };
        let alpha = (strength.clamp(0.0, 1.0) * prior.confidence).clamp(0.0, 1.0);
        blend_vec(vector, mean, alpha)
    }

    pub fn report(&self, timestamp: Timestamp) -> KnowledgeStoreReport {
        KnowledgeStoreReport {
            timestamp,
            phenotype_priors: self.phenotype_priors.len(),
        }
    }
}

impl Default for HealthKnowledgeStore {
    fn default() -> Self {
        Self::new(0.2)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeRuntime {
    queue: KnowledgeQueue,
    store: HealthKnowledgeStore,
}

impl KnowledgeRuntime {
    pub fn new(queue: KnowledgeQueue, store: HealthKnowledgeStore) -> Self {
        Self { queue, store }
    }

    pub fn ingest_document(&mut self, doc: KnowledgeDocument, timestamp: Timestamp) -> KnowledgeIngestReport {
        self.queue.enqueue_document(doc, timestamp)
    }

    pub fn submit_vote(&mut self, vote: AssociationVote) -> Option<KnowledgeAssociation> {
        let association = self.queue.record_vote(vote)?;
        if let Some(doc) = self.queue.document(&association.doc_id) {
            self.store.ingest_association(&association, doc);
        }
        Some(association)
    }

    pub fn queue_report(&self, timestamp: Timestamp) -> Option<KnowledgeQueueReport> {
        self.queue.pending_report(timestamp)
    }

    pub fn store(&self) -> &HealthKnowledgeStore {
        &self.store
    }

    pub fn store_report(&self, timestamp: Timestamp) -> KnowledgeStoreReport {
        self.store.report(timestamp)
    }
}

impl Default for KnowledgeRuntime {
    fn default() -> Self {
        Self::new(KnowledgeQueue::default(), HealthKnowledgeStore::default())
    }
}

fn build_tasks(
    doc: &KnowledgeDocument,
    timestamp: Timestamp,
    config: &KnowledgeQueueConfig,
) -> Vec<FigureAssociationTask> {
    let mut tasks = Vec::new();
    for figure in &doc.figures {
        let candidates = select_candidates(doc, figure, config.candidate_limit);
        if candidates.is_empty() {
            continue;
        }
        let task_id = compute_payload_hash(format!("task|{}|{}", doc.doc_id, figure.figure_id).as_bytes());
        let reward_score = config.reward_base
            + (candidates.len() as f64 * config.reward_per_candidate);
        let task = FigureAssociationTask {
            task_id: task_id.clone(),
            work_id: compute_payload_hash(format!("work|{}", task_id).as_bytes()),
            work_kind: WorkKind::HumanAnnotation,
            doc_id: doc.doc_id.clone(),
            figure_id: figure.figure_id.clone(),
            image_ref: figure.image_ref.clone(),
            image_hash: figure.image_hash.clone(),
            candidates,
            created_at: timestamp,
            status: AssociationStatus::Pending,
            reward_score,
            votes: Vec::new(),
        };
        tasks.push(task);
    }
    tasks
}

fn select_candidates(
    doc: &KnowledgeDocument,
    figure: &FigureAsset,
    limit: usize,
) -> Vec<TextCandidate> {
    let mut candidates = Vec::new();
    let limit = limit.max(1);
    let figure_id = normalize_figure_id(&figure.figure_id);
    let label_id = figure.label.as_ref().map(|label| normalize_figure_id(label));

    for block in &doc.text_blocks {
        let refs: Vec<String> = block.figure_refs.iter().map(|val| normalize_figure_id(val)).collect();
        if refs.iter().any(|val| val == &figure_id)
            || label_id.as_ref().map(|label| refs.iter().any(|val| val == label)).unwrap_or(false)
        {
            candidates.push(candidate_from_block(block, "xref"));
        }
    }

    if candidates.is_empty() {
        for block in &doc.text_blocks {
            if mentions_figure(&block.text, &figure_id)
                || label_id
                    .as_ref()
                    .map(|label| mentions_figure(&block.text, label))
                    .unwrap_or(false)
            {
                candidates.push(candidate_from_block(block, "text_hint"));
            }
        }
    }

    if candidates.is_empty() {
        for block in doc.text_blocks.iter().take(limit) {
            candidates.push(candidate_from_block(block, "fallback"));
        }
    }

    candidates.truncate(limit);
    candidates
}

fn mentions_figure(text: &str, figure_id: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    if lower.contains(figure_id) {
        return true;
    }
    lower.contains("fig.") || lower.contains("figure")
}

fn candidate_from_block(block: &TextBlock, reason: &str) -> TextCandidate {
    let text_hash = compute_payload_hash(block.text.as_bytes());
    TextCandidate {
        block_id: block.block_id.clone(),
        text_excerpt: excerpt(&block.text, 240),
        text_hash,
        match_reason: reason.to_string(),
    }
}

fn excerpt(text: &str, max_len: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_len {
        return trimmed.to_string();
    }
    let mut output = trimmed[..max_len].to_string();
    output.push_str("...");
    output
}

fn normalize_figure_id(raw: &str) -> String {
    raw.trim()
        .to_ascii_lowercase()
        .replace([' ', '\t', '\n', '\r'], "")
        .replace("figure", "fig")
}

fn phenotype_key_from_meta(metadata: &HashMap<String, Value>) -> Option<String> {
    if let Some(val) = metadata.get("phenotype_key").and_then(|v| v.as_str()) {
        if !val.trim().is_empty() {
            return Some(val.to_string());
        }
    }
    let mut parts = Vec::new();
    for key in ["species", "phenotype", "size_class", "age_bucket", "cohort_id", "genotype"] {
        if let Some(val) = metadata.get(key).and_then(|v| v.as_str()) {
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

fn blend(current: f64, incoming: f64, alpha: f64) -> f64 {
    let alpha = alpha.clamp(0.0, 1.0);
    current * (1.0 - alpha) + incoming * alpha
}

fn blend_vec(a: &[f64], b: &[f64], alpha: f64) -> Vec<f64> {
    let len = a.len().min(b.len()).max(1);
    let alpha = alpha.clamp(0.0, 1.0);
    let mut blended = Vec::with_capacity(len);
    for idx in 0..len {
        blended.push(a[idx] * (1.0 - alpha) + b[idx] * alpha);
    }
    blended
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;

    #[test]
    fn queue_verifies_association() {
        let mut queue = KnowledgeQueue::default();
        let doc = KnowledgeDocument {
            doc_id: "doc1".to_string(),
            source: "NLM".to_string(),
            title: Some("Sample".to_string()),
            text_blocks: vec![TextBlock {
                block_id: "t1".to_string(),
                text: "Figure 1 shows a gait change.".to_string(),
                section: None,
                order: 0,
                figure_refs: vec!["F1".to_string()],
                source: "xml".to_string(),
                confidence: 1.0,
            }],
            figures: vec![FigureAsset {
                figure_id: "Figure 1".to_string(),
                label: Some("Figure 1".to_string()),
                caption: Some("Gait".to_string()),
                image_ref: "fig1.png".to_string(),
                image_hash: compute_payload_hash(b"img"),
                order: 0,
                ocr_text: None,
            }],
            metadata: HashMap::new(),
        };
        let ts = Timestamp { unix: 10 };
        let report = queue.enqueue_document(doc, ts);
        assert!(report.tasks_created > 0);
        let task = queue.pending_report(ts).unwrap().pending.pop().unwrap();
        let vote = AssociationVote {
            task_id: task.task_id.clone(),
            worker_id: "w1".to_string(),
            selected_block_id: task.candidates[0].block_id.clone(),
            text_hash: task.candidates[0].text_hash.clone(),
            image_hash: task.image_hash.clone(),
            confidence: 0.9,
            timestamp: ts,
        };
        assert!(queue.record_vote(vote).is_none());
        let vote2 = AssociationVote {
            task_id: task.task_id.clone(),
            worker_id: "w2".to_string(),
            selected_block_id: task.candidates[0].block_id.clone(),
            text_hash: task.candidates[0].text_hash.clone(),
            image_hash: task.image_hash.clone(),
            confidence: 0.95,
            timestamp: ts,
        };
        let assoc = queue.record_vote(vote2).expect("assoc");
        assert_eq!(assoc.text_block_id, "t1");
    }
}
fn maybe_verify_task(
    task: &mut FigureAssociationTask,
    config: &KnowledgeQueueConfig,
) -> Option<KnowledgeAssociation> {
    if task.votes.len() < config.min_votes {
        return None;
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut score_sum = 0.0;
    for vote in &task.votes {
        *counts.entry(vote.selected_block_id.clone()).or_insert(0) += 1;
        score_sum += vote.confidence.clamp(0.0, 1.0);
    }
    let (selected, count) = counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .unwrap_or((String::new(), 0));
    if count < config.min_votes {
        return None;
    }
    let avg_conf = score_sum / task.votes.len().max(1) as f64;
    if avg_conf < config.min_confidence {
        return None;
    }
    let text_hash = task
        .candidates
        .iter()
        .find(|cand| cand.block_id == selected)
        .map(|cand| cand.text_hash.clone())
        .unwrap_or_default();
    task.status = AssociationStatus::Verified;
    Some(KnowledgeAssociation {
        association_id: compute_payload_hash(task.task_id.as_bytes()),
        doc_id: task.doc_id.clone(),
        figure_id: task.figure_id.clone(),
        text_block_id: selected,
        text_hash,
        image_hash: task.image_hash.clone(),
        confidence: avg_conf,
        verified_by: task.votes.len(),
        verified_at: task.created_at,
    })
}
