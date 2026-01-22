use crate::config::NetworkFabricConfig;
use crate::math_toolbox as math;
use crate::schema::Timestamp;
use crate::streaming::behavior::{BehaviorFrame, BehaviorGraph, BehaviorState, SpeciesKind};
use crate::streaming::schema::TokenBatch;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

const SIGNATURE_LEN: usize = 8;
const BEHAVIOR_EMA_ALPHA: f64 = 0.2;
const CONFIDENCE_EMA_ALPHA: f64 = 0.3;
const PHENOTYPE_WEIGHT: f64 = 0.45;
const BEHAVIOR_WEIGHT: f64 = 0.35;
const SPATIAL_WEIGHT: f64 = 0.2;
const MAX_PATTERN_REVISIONS: usize = 96;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPatternSummary {
    pub thread_id: String,
    pub last_seen: Timestamp,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    pub phenotype_hash: String,
    #[serde(default)]
    pub phenotype_tokens: Vec<String>,
    #[serde(default)]
    pub behavior_signature: Vec<f64>,
    pub support: u64,
    pub confidence: f64,
    pub novelty_score: f64,
    #[serde(default)]
    pub revision_id: String,
    #[serde(default)]
    pub origin_nodes: Vec<String>,
    #[serde(default)]
    pub opt_in_claim: Option<String>,
    #[serde(default)]
    pub species: Option<String>,
    #[serde(default)]
    pub peer_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMatch {
    pub entity_id: String,
    pub thread_id: String,
    pub match_score: f64,
    pub phenotype_score: f64,
    pub behavior_score: f64,
    pub spatial_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPatternReport {
    pub timestamp: Timestamp,
    pub observation_count: usize,
    pub thread_count: usize,
    pub new_threads: usize,
    pub matches: Vec<NetworkMatch>,
    pub shared_patterns: Vec<NetworkPatternSummary>,
}

pub struct NetworkPatternRuntime {
    config: NetworkFabricConfig,
    threads: HashMap<String, EntityThread>,
    seen_revisions: HashMap<String, VecDeque<String>>,
}

impl NetworkPatternRuntime {
    pub fn new(config: NetworkFabricConfig) -> Self {
        Self {
            config,
            threads: HashMap::new(),
            seen_revisions: HashMap::new(),
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        behavior_frame: Option<&BehaviorFrame>,
    ) -> Option<NetworkPatternReport> {
        if !self.config.enabled {
            return None;
        }
        let observations = if let Some(frame) = behavior_frame {
            observations_from_behavior(frame)
        } else {
            observations_from_tokens(batch)
        };
        if observations.is_empty() {
            return None;
        }
        let mut matches = Vec::new();
        let mut new_threads = 0usize;
        for observation in observations {
            if let Some((thread_id, score, components)) = self.best_match(&observation) {
                if score >= self.config.match_threshold {
                    if let Some(thread) = self.threads.get_mut(&thread_id) {
                        thread.update(&observation, &self.config);
                    }
                    matches.push(NetworkMatch {
                        entity_id: observation.entity_id.clone(),
                        thread_id,
                        match_score: score,
                        phenotype_score: components.0,
                        behavior_score: components.1,
                        spatial_score: components.2,
                    });
                    continue;
                }
            }
            let thread_id = thread_id_for(&observation);
            let thread = EntityThread::from_observation(&thread_id, &observation);
            self.threads.insert(thread_id, thread);
            new_threads = new_threads.saturating_add(1);
        }
        self.prune_threads();
        let shared_patterns = self.shared_patterns();
        Some(NetworkPatternReport {
            timestamp: batch.timestamp,
            observation_count: matches.len() + new_threads,
            thread_count: self.threads.len(),
            new_threads,
            matches,
            shared_patterns,
        })
    }

    pub fn ingest_shared_patterns(&mut self, patterns: &[NetworkPatternSummary]) {
        if !self.config.enabled || patterns.is_empty() {
            return;
        }
        for pattern in patterns {
            let mut incoming = pattern.clone();
            ensure_revision_id(&mut incoming);
            normalize_peer_weight(&mut incoming);
            if !self.record_revision(&incoming) {
                continue;
            }
            if let Some(thread) = self.threads.get_mut(&incoming.thread_id) {
                thread.merge_summary(&incoming, &self.config);
                continue;
            }
            let observation = EntityObservation::from_summary(&incoming);
            if let Some((thread_id, score, _)) = self.best_match(&observation) {
                if score >= self.config.match_threshold {
                    if let Some(thread) = self.threads.get_mut(&thread_id) {
                        thread.merge_summary(&incoming, &self.config);
                    }
                    continue;
                }
            }
            let mut thread = EntityThread::from_summary(&incoming);
            thread.thread_id = incoming.thread_id.clone();
            self.threads.insert(incoming.thread_id.clone(), thread);
        }
        self.prune_threads();
    }

    fn best_match(&self, observation: &EntityObservation) -> Option<(String, f64, (f64, f64, f64))> {
        let mut best = None;
        for (thread_id, thread) in &self.threads {
            let phenotype_score = phenotype_similarity(&thread.phenotype_tokens, &observation.phenotype_tokens);
            let behavior_score = behavior_similarity(&thread.behavior_signature, &observation.behavior_signature);
            let spatial_score = spatial_similarity(
                thread.position,
                observation.position,
                thread.last_seen,
                observation.timestamp,
                self.config.max_speed_units_per_sec,
            );
            let match_score = weighted_match_score(phenotype_score, behavior_score, spatial_score)
                * observation.confidence.max(0.05);
            if let Some((_, best_score, _)) = &best {
                if match_score <= *best_score {
                    continue;
                }
            }
            best = Some((
                thread_id.clone(),
                match_score,
                (phenotype_score, behavior_score, spatial_score),
            ));
        }
        best
    }

    fn shared_patterns(&self) -> Vec<NetworkPatternSummary> {
        let mut entries: Vec<_> = self
            .threads
            .values()
            .map(|thread| thread.summary())
            .collect();
        entries.sort_by(|a, b| b.last_seen.unix.cmp(&a.last_seen.unix));
        entries
            .into_iter()
            .filter(|summary| {
                summary.support as usize >= self.config.min_support
                    || summary.novelty_score >= 0.6
            })
            .take(self.config.max_shared_threads.max(1))
            .collect()
    }

    fn prune_threads(&mut self) {
        let max_threads = self.config.max_threads.max(1);
        if self.threads.len() <= max_threads {
            return;
        }
        let mut entries: Vec<_> = self
            .threads
            .iter()
            .map(|(id, thread)| (id.clone(), thread.last_seen.unix))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(max_threads);
        let keep: HashSet<String> = entries.into_iter().map(|(id, _)| id).collect();
        self.threads.retain(|id, _| keep.contains(id));
    }

    fn record_revision(&mut self, summary: &NetworkPatternSummary) -> bool {
        let thread_id = summary.thread_id.trim();
        let revision = summary.revision_id.trim();
        if thread_id.is_empty() || revision.is_empty() {
            return true;
        }
        let entry = self
            .seen_revisions
            .entry(thread_id.to_string())
            .or_insert_with(VecDeque::new);
        if entry.iter().any(|value| value == revision) {
            return false;
        }
        entry.push_back(revision.to_string());
        while entry.len() > MAX_PATTERN_REVISIONS {
            entry.pop_front();
        }
        true
    }
}

#[derive(Debug, Clone)]
struct EntityThread {
    thread_id: String,
    last_seen: Timestamp,
    position: Option<[f64; 3]>,
    phenotype_tokens: Vec<String>,
    phenotype_hash: String,
    behavior_signature: Vec<f64>,
    support: u64,
    confidence: f64,
    origin_nodes: HashSet<String>,
    opt_in_claim: Option<String>,
    species: Option<String>,
}

impl EntityThread {
    fn from_observation(thread_id: &str, observation: &EntityObservation) -> Self {
        let mut origin_nodes = HashSet::new();
        if let Some(node_id) = &observation.origin_node_id {
            origin_nodes.insert(node_id.clone());
        }
        Self {
            thread_id: thread_id.to_string(),
            last_seen: observation.timestamp,
            position: observation.position,
            phenotype_tokens: observation.phenotype_tokens.clone(),
            phenotype_hash: observation.phenotype_hash.clone(),
            behavior_signature: observation.behavior_signature.clone(),
            support: observation.support.max(1),
            confidence: observation.confidence,
            origin_nodes,
            opt_in_claim: observation.opt_in_claim.clone(),
            species: observation.species.clone(),
        }
    }

    fn from_summary(summary: &NetworkPatternSummary) -> Self {
        Self {
            thread_id: summary.thread_id.clone(),
            last_seen: summary.last_seen,
            position: summary.position,
            phenotype_tokens: summary.phenotype_tokens.clone(),
            phenotype_hash: summary.phenotype_hash.clone(),
            behavior_signature: summary.behavior_signature.clone(),
            support: summary.support.max(1),
            confidence: summary.confidence,
            origin_nodes: summary.origin_nodes.iter().cloned().collect(),
            opt_in_claim: summary.opt_in_claim.clone(),
            species: summary.species.clone(),
        }
    }

    fn update(&mut self, observation: &EntityObservation, config: &NetworkFabricConfig) {
        self.last_seen = observation.timestamp;
        if observation.position.is_some() {
            self.position = observation.position;
        }
        self.support = self.support.saturating_add(observation.support.max(1));
        self.confidence = (1.0 - CONFIDENCE_EMA_ALPHA) * self.confidence
            + CONFIDENCE_EMA_ALPHA * observation.confidence;
        blend_signature(&mut self.behavior_signature, &observation.behavior_signature);
        merge_tokens(&mut self.phenotype_tokens, &observation.phenotype_tokens, config);
        self.phenotype_hash = phenotype_hash(&self.phenotype_tokens);
        if self.opt_in_claim.is_none() {
            self.opt_in_claim = observation.opt_in_claim.clone();
        }
        if self.species.is_none() {
            self.species = observation.species.clone();
        }
        if let Some(node_id) = &observation.origin_node_id {
            self.origin_nodes.insert(node_id.clone());
        }
    }

    fn merge_summary(&mut self, summary: &NetworkPatternSummary, config: &NetworkFabricConfig) {
        self.last_seen = summary.last_seen;
        if summary.position.is_some() {
            self.position = summary.position;
        }
        let peer_weight = peer_weight_value(summary.peer_weight);
        let weighted_support =
            ((summary.support.max(1) as f64) * peer_weight).max(1.0) as u64;
        self.support = self.support.saturating_add(weighted_support);
        let alpha = (CONFIDENCE_EMA_ALPHA * peer_weight).clamp(0.05, 0.9);
        self.confidence = (1.0 - alpha) * self.confidence + alpha * summary.confidence;
        blend_signature_weighted(&mut self.behavior_signature, &summary.behavior_signature, alpha);
        merge_tokens(&mut self.phenotype_tokens, &summary.phenotype_tokens, config);
        self.phenotype_hash = phenotype_hash(&self.phenotype_tokens);
        if self.opt_in_claim.is_none() {
            self.opt_in_claim = summary.opt_in_claim.clone();
        }
        if self.species.is_none() {
            self.species = summary.species.clone();
        }
        for node in &summary.origin_nodes {
            self.origin_nodes.insert(node.clone());
        }
    }

    fn summary(&self) -> NetworkPatternSummary {
        NetworkPatternSummary {
            thread_id: self.thread_id.clone(),
            last_seen: self.last_seen,
            position: self.position,
            phenotype_hash: self.phenotype_hash.clone(),
            phenotype_tokens: self.phenotype_tokens.clone(),
            behavior_signature: self.behavior_signature.clone(),
            support: self.support,
            confidence: self.confidence.clamp(0.0, 1.0),
            novelty_score: novelty_score(self.support),
            revision_id: pattern_revision_id(
                &self.thread_id,
                self.last_seen,
                &self.phenotype_hash,
                &self.behavior_signature,
                self.support,
            ),
            origin_nodes: self.origin_nodes.iter().cloned().collect(),
            opt_in_claim: self.opt_in_claim.clone(),
            species: self.species.clone(),
            peer_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
struct EntityObservation {
    entity_id: String,
    timestamp: Timestamp,
    position: Option<[f64; 3]>,
    phenotype_tokens: Vec<String>,
    phenotype_hash: String,
    behavior_signature: Vec<f64>,
    confidence: f64,
    support: u64,
    origin_node_id: Option<String>,
    opt_in_claim: Option<String>,
    species: Option<String>,
}

impl EntityObservation {
    fn from_summary(summary: &NetworkPatternSummary) -> Self {
        Self {
            entity_id: summary.thread_id.clone(),
            timestamp: summary.last_seen,
            position: summary.position,
            phenotype_tokens: summary.phenotype_tokens.clone(),
            phenotype_hash: summary.phenotype_hash.clone(),
            behavior_signature: summary.behavior_signature.clone(),
            confidence: summary.confidence,
            support: summary.support.max(1),
            origin_node_id: summary.origin_nodes.first().cloned(),
            opt_in_claim: summary.opt_in_claim.clone(),
            species: summary.species.clone(),
        }
    }
}

fn observations_from_behavior(frame: &BehaviorFrame) -> Vec<EntityObservation> {
    let mut observations = Vec::new();
    for state in &frame.states {
        observations.push(observation_from_state(state, &frame.graph));
    }
    observations
}

fn observation_from_state(state: &BehaviorState, graph: &BehaviorGraph) -> EntityObservation {
    let species = Some(species_tag(&state.species));
    let phenotype_tokens = phenotype_tokens_from_attrs(&state.attributes, species.as_deref());
    let phenotype_hash = phenotype_hash(&phenotype_tokens);
    let behavior_signature = behavior_signature_from_state(state, graph);
    let confidence = state.confidence.clamp(0.0, 1.0);
    let opt_in_claim = identity_claim_from_attrs(&state.attributes);
    let position = state.position.or_else(|| position_from_attrs(&state.attributes));
    EntityObservation {
        entity_id: state.entity_id.clone(),
        timestamp: state.timestamp,
        position,
        phenotype_tokens,
        phenotype_hash,
        behavior_signature,
        confidence,
        support: 1,
        origin_node_id: None,
        opt_in_claim,
        species,
    }
}

fn observations_from_tokens(batch: &TokenBatch) -> Vec<EntityObservation> {
    let mut accumulators: HashMap<String, ObservationAccumulator> = HashMap::new();
    for token in &batch.tokens {
        let Some(entity_id) = entity_id_from_token(token) else {
            continue;
        };
        let phenotype_tokens = phenotype_tokens_from_token(&token.attributes);
        let phenotype_hash = phenotype_hash(&phenotype_tokens);
        let behavior_signature = behavior_signature_from_token(&token.attributes);
        let position = position_from_attrs(&token.attributes);
        let origin_node_id = token
            .attributes
            .get("origin_node_id")
            .and_then(|val| val.as_str())
            .map(|val| val.to_string());
        let opt_in_claim = identity_claim_from_attrs(&token.attributes);
        let species = token
            .attributes
            .get("species")
            .and_then(|val| val.as_str())
            .map(|val| val.to_string());
        let entry = accumulators
            .entry(entity_id.clone())
            .or_insert_with(|| ObservationAccumulator::new(entity_id.clone(), token.onset));
        entry.push(
            token.onset,
            token.confidence,
            position,
            phenotype_tokens,
            phenotype_hash,
            behavior_signature,
            origin_node_id,
            opt_in_claim,
            species,
        );
    }
    accumulators
        .into_values()
        .map(|acc| acc.finalize())
        .collect()
}

struct ObservationAccumulator {
    entity_id: String,
    timestamp: Timestamp,
    position: Option<[f64; 3]>,
    phenotype_tokens: HashSet<String>,
    phenotype_hash: String,
    behavior_signature: Vec<f64>,
    confidence_sum: f64,
    count: u64,
    origin_node_id: Option<String>,
    opt_in_claim: Option<String>,
    species: Option<String>,
}

impl ObservationAccumulator {
    fn new(entity_id: String, timestamp: Timestamp) -> Self {
        Self {
            entity_id,
            timestamp,
            position: None,
            phenotype_tokens: HashSet::new(),
            phenotype_hash: String::new(),
            behavior_signature: vec![0.0; SIGNATURE_LEN],
            confidence_sum: 0.0,
            count: 0,
            origin_node_id: None,
            opt_in_claim: None,
            species: None,
        }
    }

    fn push(
        &mut self,
        timestamp: Timestamp,
        confidence: f64,
        position: Option<[f64; 3]>,
        phenotype_tokens: Vec<String>,
        phenotype_hash: String,
        behavior_signature: Vec<f64>,
        origin_node_id: Option<String>,
        opt_in_claim: Option<String>,
        species: Option<String>,
    ) {
        if timestamp.unix > self.timestamp.unix {
            self.timestamp = timestamp;
        }
        if position.is_some() {
            self.position = position;
        }
        for token in phenotype_tokens {
            self.phenotype_tokens.insert(token);
        }
        if self.phenotype_hash.is_empty() && !phenotype_hash.is_empty() {
            self.phenotype_hash = phenotype_hash;
        }
        for (idx, value) in behavior_signature.iter().enumerate() {
            if idx < self.behavior_signature.len() {
                self.behavior_signature[idx] += value;
            }
        }
        self.confidence_sum += confidence;
        self.count = self.count.saturating_add(1);
        if self.origin_node_id.is_none() {
            self.origin_node_id = origin_node_id;
        }
        if self.opt_in_claim.is_none() {
            self.opt_in_claim = opt_in_claim;
        }
        if self.species.is_none() {
            self.species = species;
        }
    }

    fn finalize(mut self) -> EntityObservation {
        let count = self.count.max(1) as f64;
        for value in &mut self.behavior_signature {
            *value /= count;
        }
        let phenotype_tokens: Vec<String> = self.phenotype_tokens.into_iter().collect();
        let phenotype_hash = if self.phenotype_hash.is_empty() {
            phenotype_hash(&phenotype_tokens)
        } else {
            self.phenotype_hash.clone()
        };
        let confidence = (self.confidence_sum / count).clamp(0.0, 1.0);
        EntityObservation {
            entity_id: self.entity_id,
            timestamp: self.timestamp,
            position: self.position,
            phenotype_tokens,
            phenotype_hash,
            behavior_signature: self.behavior_signature,
            confidence,
            support: self.count.max(1),
            origin_node_id: self.origin_node_id,
            opt_in_claim: self.opt_in_claim,
            species: self.species,
        }
    }
}

fn entity_id_from_token(token: &crate::streaming::schema::EventToken) -> Option<String> {
    token
        .attributes
        .get("entity_id")
        .and_then(|val| val.as_str())
        .map(|val| val.to_string())
}

fn phenotype_tokens_from_attrs(attrs: &HashMap<String, Value>, species: Option<&str>) -> Vec<String> {
    let keys = [
        "phenotype",
        "phenotype_signature",
        "size_class",
        "age_bucket",
        "cohort_id",
        "genotype",
        "lineage",
        "anatomy_site",
        "context_scope",
        "species_tag",
    ];
    let mut raw_tokens = Vec::new();
    for key in keys {
        if let Some(value) = attrs.get(key) {
            if let Some(token) = token_from_value(key, value) {
                raw_tokens.push(token);
            }
        }
    }
    if let Some(species) = species {
        raw_tokens.push(format!("species={species}"));
    }
    hash_tokens(&raw_tokens)
}

fn phenotype_tokens_from_token(attrs: &HashMap<String, Value>) -> Vec<String> {
    let keys = [
        "phenotype",
        "phenotype_signature",
        "size_class",
        "age_bucket",
        "cohort_id",
        "genotype",
        "lineage",
        "anatomy_site",
        "context_scope",
        "species_tag",
        "species",
    ];
    let mut raw_tokens = Vec::new();
    for key in keys {
        if let Some(value) = attrs.get(key) {
            if let Some(token) = token_from_value(key, value) {
                raw_tokens.push(token);
            }
        }
    }
    if let Some(meta) = attrs.get("metadata").and_then(|val| val.as_object()) {
        for key in keys {
            if let Some(value) = meta.get(key) {
                if let Some(token) = token_from_value(key, value) {
                    raw_tokens.push(token);
                }
            }
        }
    }
    hash_tokens(&raw_tokens)
}

fn token_from_value(key: &str, value: &Value) -> Option<String> {
    if let Some(val) = value.as_str() {
        return Some(format!("{key}={}", val.trim()));
    }
    if let Some(val) = value.as_f64() {
        return Some(format!("{key}={val:.4}"));
    }
    if let Some(val) = value.as_bool() {
        return Some(format!("{key}={val}"));
    }
    None
}

fn behavior_signature_from_state(state: &BehaviorState, graph: &BehaviorGraph) -> Vec<f64> {
    let latent_mean = math::mean(&state.latent).unwrap_or(0.0);
    let action_mean = math::mean(&state.action).unwrap_or(0.0);
    let latent_var = math::variance(&state.latent, 0.0).unwrap_or(0.0);
    let action_var = math::variance(&state.action, 0.0).unwrap_or(0.0);
    let latent_norm = math::l2_norm(&state.latent);
    let action_norm = math::l2_norm(&state.action);
    let (graph_coherence, graph_proximity) = graph_signature_for_entity(graph, &state.entity_id);
    vec![
        latent_norm,
        action_norm,
        latent_mean,
        action_mean,
        latent_var,
        action_var,
        graph_coherence,
        graph_proximity,
    ]
}

fn behavior_signature_from_token(attrs: &HashMap<String, Value>) -> Vec<f64> {
    let keys = [
        "motion_energy",
        "motion_variance",
        "micro_jitter",
        "posture_shift",
        "stability",
        "graph_coherence",
        "graph_proximity",
        "motor_signal_norm",
    ];
    let mut out = vec![0.0; SIGNATURE_LEN];
    for (idx, key) in keys.iter().enumerate() {
        if idx >= out.len() {
            break;
        }
        if let Some(value) = attrs.get(*key).and_then(|val| val.as_f64()) {
            out[idx] = value;
        }
    }
    out
}

fn graph_signature_for_entity(graph: &BehaviorGraph, entity_id: &str) -> (f64, f64) {
    let mut coherence_sum = 0.0;
    let mut proximity_sum = 0.0;
    let mut count = 0.0;
    for edge in &graph.edges {
        if edge.source != entity_id && edge.target != entity_id {
            continue;
        }
        coherence_sum += edge.coupling.coherence;
        if let Some(prox) = edge.proximity {
            proximity_sum += prox;
        }
        count += 1.0;
    }
    if count <= 0.0 {
        return (0.0, 0.0);
    }
    (coherence_sum / count, proximity_sum / count)
}

fn identity_claim_from_attrs(attrs: &HashMap<String, Value>) -> Option<String> {
    if let Some(value) = attrs.get("identity_claim").and_then(|val| val.as_str()) {
        let claim = value.trim();
        if !claim.is_empty() {
            return Some(claim.to_string());
        }
    }
    if attrs
        .get("identity_opt_in")
        .and_then(|val| val.as_bool())
        .unwrap_or(false)
    {
        if let Some(value) = attrs.get("identity_tag").and_then(|val| val.as_str()) {
            let claim = value.trim();
            if !claim.is_empty() {
                return Some(claim.to_string());
            }
        }
    }
    if let Some(meta) = attrs.get("metadata").and_then(|val| val.as_object()) {
        if let Some(value) = meta.get("identity_claim").and_then(|val| val.as_str()) {
            let claim = value.trim();
            if !claim.is_empty() {
                return Some(claim.to_string());
            }
        }
    }
    None
}

fn position_from_attrs(attrs: &HashMap<String, Value>) -> Option<[f64; 3]> {
    let x = attrs.get("pos_x").and_then(|val| val.as_f64())?;
    let y = attrs.get("pos_y").and_then(|val| val.as_f64())?;
    let z = attrs
        .get("pos_z")
        .and_then(|val| val.as_f64())
        .unwrap_or(0.0);
    Some([x, y, z])
}

fn species_tag(species: &SpeciesKind) -> String {
    match species {
        SpeciesKind::Human => "HUMAN".to_string(),
        SpeciesKind::Canine => "CANINE".to_string(),
        SpeciesKind::Feline => "FELINE".to_string(),
        SpeciesKind::Avian => "AVIAN".to_string(),
        SpeciesKind::Aquatic => "AQUATIC".to_string(),
        SpeciesKind::Insect => "INSECT".to_string(),
        SpeciesKind::Other(raw) => raw.to_string(),
    }
}

fn merge_tokens(
    existing: &mut Vec<String>,
    incoming: &[String],
    config: &NetworkFabricConfig,
) {
    let mut set: HashSet<String> = existing.iter().cloned().collect();
    for token in incoming {
        if set.len() >= config.max_phenotype_tokens.max(1) {
            break;
        }
        set.insert(token.clone());
    }
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out.truncate(config.max_phenotype_tokens.max(1));
    *existing = out;
}

fn blend_signature(existing: &mut Vec<f64>, incoming: &[f64]) {
    if existing.len() != incoming.len() {
        *existing = incoming.to_vec();
        return;
    }
    for (val, inc) in existing.iter_mut().zip(incoming.iter()) {
        *val = (1.0 - BEHAVIOR_EMA_ALPHA) * *val + BEHAVIOR_EMA_ALPHA * inc;
    }
}

fn blend_signature_weighted(existing: &mut Vec<f64>, incoming: &[f64], alpha: f64) {
    if existing.len() != incoming.len() {
        *existing = incoming.to_vec();
        return;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;
    for (val, inc) in existing.iter_mut().zip(incoming.iter()) {
        *val = beta * *val + alpha * inc;
    }
}

fn phenotype_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<&String> = a.iter().collect();
    let set_b: HashSet<&String> = b.iter().collect();
    let mut intersection = 0usize;
    for token in &set_a {
        if set_b.contains(token) {
            intersection += 1;
        }
    }
    let union = set_a.len() + set_b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        (intersection as f64 / union as f64).clamp(0.0, 1.0)
    }
}

fn behavior_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    match math::cosine_similarity(a, b) {
        Some(score) => ((score + 1.0) * 0.5).clamp(0.0, 1.0),
        None => 0.0,
    }
}

fn ensure_revision_id(summary: &mut NetworkPatternSummary) {
    if !summary.revision_id.trim().is_empty() {
        return;
    }
    summary.revision_id = pattern_revision_id(
        &summary.thread_id,
        summary.last_seen,
        &summary.phenotype_hash,
        &summary.behavior_signature,
        summary.support,
    );
}

fn normalize_peer_weight(summary: &mut NetworkPatternSummary) {
    summary.peer_weight = peer_weight_value(summary.peer_weight);
}

fn peer_weight_value(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.clamp(0.1, 2.0)
    } else {
        1.0
    }
}

fn pattern_revision_id(
    thread_id: &str,
    last_seen: Timestamp,
    phenotype_hash: &str,
    signature: &[f64],
    support: u64,
) -> String {
    let signature_hash = hash_to_hex(signature_fingerprint(signature));
    let payload = format!(
        "pattern|{}|{}|{}|{}|{}",
        thread_id.trim(),
        last_seen.unix,
        phenotype_hash.trim(),
        support,
        signature_hash
    );
    hash_to_hex(fnv1a_hash(payload.as_bytes()))
}

fn signature_fingerprint(signature: &[f64]) -> u64 {
    if signature.is_empty() {
        return 0;
    }
    let mut bytes = Vec::with_capacity(signature.len() * 8);
    for value in signature {
        let normalized = if value.is_finite() { *value } else { 0.0 };
        bytes.extend_from_slice(&normalized.to_le_bytes());
    }
    fnv1a_hash(&bytes)
}

fn spatial_similarity(
    a: Option<[f64; 3]>,
    b: Option<[f64; 3]>,
    a_time: Timestamp,
    b_time: Timestamp,
    max_speed: f64,
) -> f64 {
    let Some(a_pos) = a else {
        return 0.5;
    };
    let Some(b_pos) = b else {
        return 0.5;
    };
    let dx = a_pos[0] - b_pos[0];
    let dy = a_pos[1] - b_pos[1];
    let dz = a_pos[2] - b_pos[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    let dt = (a_time.unix - b_time.unix).abs().max(1) as f64;
    let allowable = max_speed.max(0.0) * dt + 0.5;
    if allowable <= 0.0 {
        return 0.0;
    }
    if distance <= allowable {
        1.0
    } else {
        (allowable / distance).clamp(0.0, 1.0)
    }
}

fn weighted_match_score(phenotype: f64, behavior: f64, spatial: f64) -> f64 {
    let sum = PHENOTYPE_WEIGHT + BEHAVIOR_WEIGHT + SPATIAL_WEIGHT;
    if sum <= 0.0 {
        return 0.0;
    }
    let score =
        PHENOTYPE_WEIGHT * phenotype + BEHAVIOR_WEIGHT * behavior + SPATIAL_WEIGHT * spatial;
    (score / sum).clamp(0.0, 1.0)
}

fn novelty_score(support: u64) -> f64 {
    let val = support as f64;
    (1.0 / (1.0 + val.sqrt())).clamp(0.0, 1.0)
}

fn hash_tokens(tokens: &[String]) -> Vec<String> {
    let mut hashes: Vec<String> = tokens
        .iter()
        .map(|token| hash_to_hex(fnv1a_hash(token.as_bytes())))
        .collect();
    hashes.sort();
    hashes.dedup();
    hashes
}

fn phenotype_hash(tokens: &[String]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let mut joined = String::new();
    for token in tokens {
        joined.push_str(token);
        joined.push('|');
    }
    hash_to_hex(fnv1a_hash(joined.as_bytes()))
}

fn thread_id_for(observation: &EntityObservation) -> String {
    let key = format!(
        "{}:{}:{}",
        observation.entity_id,
        observation.timestamp.unix,
        observation.phenotype_hash
    );
    format!("thread-{}", hash_to_hex(fnv1a_hash(key.as_bytes())))
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn hash_to_hex(hash: u64) -> String {
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::behavior::{BackpressureStatus, BehaviorFrame, BehaviorGraph, CouplingMetrics};

    fn sample_state(id: &str, t: i64, x: f64) -> BehaviorState {
        BehaviorState {
            entity_id: id.to_string(),
            timestamp: Timestamp { unix: t },
            species: SpeciesKind::Human,
            latent: vec![0.2, 0.4],
            action: vec![0.1],
            position: Some([x, 0.0, 0.0]),
            confidence: 0.9,
            missing_ratio: 0.0,
            attributes: HashMap::from([(
                "phenotype".to_string(),
                Value::String("sample".to_string()),
            )]),
        }
    }

    #[test]
    fn network_fabric_matches_same_entity() {
        let mut runtime = NetworkPatternRuntime::new(NetworkFabricConfig {
            enabled: true,
            max_threads: 64,
            max_shared_threads: 16,
            min_support: 1,
            match_threshold: 0.5,
            max_speed_units_per_sec: 5.0,
            max_phenotype_tokens: 16,
        });
        let state_a = sample_state("e1", 10, 0.0);
        let graph = BehaviorGraph {
            timestamp: Timestamp { unix: 10 },
            nodes: HashMap::from([(state_a.entity_id.clone(), state_a.clone())]),
            edges: vec![crate::streaming::behavior::BehaviorEdge {
                source: "e1".to_string(),
                target: "e1".to_string(),
                proximity: Some(0.2),
                coupling: CouplingMetrics {
                    coherence: 0.7,
                    phase_locking: None,
                    transfer_entropy: None,
                    confidence: 1.0,
                },
            }],
        };
        let frame = BehaviorFrame {
            timestamp: Timestamp { unix: 10 },
            states: vec![state_a],
            graph,
            motifs: Vec::new(),
            prediction: None,
            backpressure: BackpressureStatus::Ok,
        };
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 10 },
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let report = runtime.update(&batch, Some(&frame)).expect("report");
        assert_eq!(report.new_threads, 1);

        let state_b = sample_state("e1", 12, 0.2);
        let graph_b = BehaviorGraph {
            timestamp: Timestamp { unix: 12 },
            nodes: HashMap::from([(state_b.entity_id.clone(), state_b.clone())]),
            edges: Vec::new(),
        };
        let frame_b = BehaviorFrame {
            timestamp: Timestamp { unix: 12 },
            states: vec![state_b],
            graph: graph_b,
            motifs: Vec::new(),
            prediction: None,
            backpressure: BackpressureStatus::Ok,
        };
        let report_b = runtime.update(&batch, Some(&frame_b)).expect("report");
        assert_eq!(report_b.new_threads, 0);
        assert!(!report_b.matches.is_empty());
    }

    #[test]
    fn network_fabric_dedupes_shared_revisions() {
        let mut runtime = NetworkPatternRuntime::new(NetworkFabricConfig {
            enabled: true,
            max_threads: 64,
            max_shared_threads: 16,
            min_support: 1,
            match_threshold: 0.5,
            max_speed_units_per_sec: 5.0,
            max_phenotype_tokens: 16,
        });
        let mut pattern = NetworkPatternSummary {
            thread_id: "thread-a".to_string(),
            last_seen: Timestamp { unix: 10 },
            position: Some([0.0, 0.0, 0.0]),
            phenotype_hash: "ph1".to_string(),
            phenotype_tokens: vec!["p1".to_string()],
            behavior_signature: vec![0.1; SIGNATURE_LEN],
            support: 2,
            confidence: 0.6,
            novelty_score: 0.3,
            revision_id: String::new(),
            origin_nodes: vec!["node-a".to_string()],
            opt_in_claim: None,
            species: Some("HUMAN".to_string()),
            peer_weight: 1.0,
        };
        ensure_revision_id(&mut pattern);
        runtime.ingest_shared_patterns(&[pattern.clone()]);
        runtime.ingest_shared_patterns(&[pattern]);
        assert_eq!(runtime.threads.len(), 1);
        let thread = runtime.threads.values().next().expect("thread");
        assert!(thread.support <= 4);
    }

    #[test]
    fn peer_weight_influences_merge_confidence() {
        let config = NetworkFabricConfig::default();
        let base = NetworkPatternSummary {
            thread_id: "thread-w".to_string(),
            last_seen: Timestamp { unix: 10 },
            position: None,
            phenotype_hash: "ph".to_string(),
            phenotype_tokens: vec!["p1".to_string()],
            behavior_signature: vec![0.2; SIGNATURE_LEN],
            support: 2,
            confidence: 0.4,
            novelty_score: 0.3,
            revision_id: "rev-1".to_string(),
            origin_nodes: Vec::new(),
            opt_in_claim: None,
            species: Some("HUMAN".to_string()),
            peer_weight: 1.0,
        };
        let mut thread_low = EntityThread::from_summary(&base);
        let mut low = base.clone();
        low.confidence = 1.0;
        low.peer_weight = 0.2;
        thread_low.merge_summary(&low, &config);
        let low_conf = thread_low.confidence;

        let mut thread_high = EntityThread::from_summary(&base);
        let mut high = base;
        high.confidence = 1.0;
        high.peer_weight = 1.8;
        thread_high.merge_summary(&high, &config);
        assert!(thread_high.confidence > low_conf);
    }
}
