use crate::config::CrossModalConfig;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::spike::{NeuronKind, SpikeConfig, SpikeInput, SpikePool};
use crate::streaming::schema::{EventKind, EventToken, StreamSource, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalLink {
    pub text_key: String,
    pub video_key: String,
    pub support: u64,
    pub weight: f64,
    pub last_seen: Timestamp,
    #[serde(default)]
    pub text_preview: String,
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(default)]
    pub frame_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalQuery {
    pub source_key: String,
    #[serde(default)]
    pub matches: Vec<CrossModalLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalReport {
    pub timestamp: Timestamp,
    pub total_links: usize,
    pub new_links: usize,
    #[serde(default)]
    pub top_links: Vec<CrossModalLink>,
    #[serde(default)]
    pub video_queries: Vec<CrossModalQuery>,
    #[serde(default)]
    pub text_queries: Vec<CrossModalQuery>,
}

#[derive(Debug, Clone)]
struct ObservedVideo {
    key: String,
    timestamp: Timestamp,
    confidence: f64,
    entity_id: Option<String>,
    frame_ref: Option<String>,
}

#[derive(Debug, Clone)]
struct ObservedText {
    key: String,
    timestamp: Timestamp,
    confidence: f64,
    entity_id: Option<String>,
    frame_ref: Option<String>,
    text_preview: String,
}

#[derive(Debug, Clone)]
struct AssociationState {
    text_key: String,
    video_key: String,
    support: u64,
    weight: f64,
    last_seen: Timestamp,
    text_preview: String,
    entity_id: Option<String>,
    frame_ref: Option<String>,
}

pub struct CrossModalRuntime {
    config: CrossModalConfig,
    associations: HashMap<(String, String), AssociationState>,
    video_index: HashMap<String, HashSet<String>>,
    text_index: HashMap<String, HashSet<String>>,
    recent_video: VecDeque<ObservedVideo>,
    recent_text: VecDeque<ObservedText>,
    spike_pool: SpikePool,
    label_to_neuron: HashMap<String, u32>,
    neuron_labels: HashMap<u32, String>,
    connections: HashSet<(u32, u32)>,
    composites: HashMap<(String, String), u32>,
}

impl CrossModalRuntime {
    pub fn new(config: CrossModalConfig) -> Self {
        let spike_config = SpikeConfig {
            threshold: config.spike.threshold,
            membrane_decay: config.spike.membrane_decay,
            refractory_steps: config.spike.refractory_steps,
        };
        Self {
            config,
            associations: HashMap::new(),
            video_index: HashMap::new(),
            text_index: HashMap::new(),
            recent_video: VecDeque::new(),
            recent_text: VecDeque::new(),
            spike_pool: SpikePool::new("cross_modal", spike_config),
            label_to_neuron: HashMap::new(),
            neuron_labels: HashMap::new(),
            connections: HashSet::new(),
            composites: HashMap::new(),
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<CrossModalReport> {
        if !self.config.enabled {
            return None;
        }
        let now = batch.timestamp;
        let mut new_videos = Vec::new();
        let mut new_texts = Vec::new();
        for token in &batch.tokens {
            if is_text_token(token) {
                if let Some(text) = observed_text(token, self.config.max_text_len) {
                    new_texts.push(text);
                }
                continue;
            }
            if is_video_token(token) {
                if let Some(video) = observed_video(token) {
                    new_videos.push(video);
                }
            }
        }
        if new_videos.is_empty() && new_texts.is_empty() {
            self.prune_recent(now);
            return None;
        }
        if !self.config.spike.enabled {
            return self.update_associative(now, new_videos, new_texts);
        }
        self.update_spike(now, new_videos, new_texts)
    }

    pub fn ingest_report(&mut self, report: &CrossModalReport) {
        if !self.config.enabled {
            return;
        }
        for link in &report.top_links {
            let key = (link.text_key.clone(), link.video_key.clone());
            let entry = self.associations.entry(key.clone()).or_insert_with(|| AssociationState {
                text_key: link.text_key.clone(),
                video_key: link.video_key.clone(),
                support: 0,
                weight: 0.0,
                last_seen: link.last_seen,
                text_preview: link.text_preview.clone(),
                entity_id: link.entity_id.clone(),
                frame_ref: link.frame_ref.clone(),
            });
            entry.support = entry.support.saturating_add(link.support);
            entry.weight = (entry.weight + link.weight).clamp(0.0, 1.0);
            entry.last_seen = link.last_seen;
            self.video_index
                .entry(link.video_key.clone())
                .or_default()
                .insert(link.text_key.clone());
            self.text_index
                .entry(link.text_key.clone())
                .or_default()
                .insert(link.video_key.clone());
            if self.config.spike.enabled && link.support as usize >= self.config.min_support {
                self.ensure_composite(&link.text_key, &link.video_key);
            }
        }
        self.trim_links();
    }

    pub fn query_text_for_video(&self, video_key: &str, limit: usize) -> Vec<CrossModalLink> {
        let Some(texts) = self.video_index.get(video_key) else {
            return Vec::new();
        };
        let mut links = Vec::new();
        for text_key in texts {
            if let Some(link) = self.link_for(text_key, video_key) {
                links.push(link);
            }
        }
        sort_links(&mut links);
        links.truncate(limit.max(1));
        links
    }

    pub fn query_video_for_text(&self, text_key: &str, limit: usize) -> Vec<CrossModalLink> {
        let Some(videos) = self.text_index.get(text_key) else {
            return Vec::new();
        };
        let mut links = Vec::new();
        for video_key in videos {
            if let Some(link) = self.link_for(text_key, video_key) {
                links.push(link);
            }
        }
        sort_links(&mut links);
        links.truncate(limit.max(1));
        links
    }

    fn update_associative(
        &mut self,
        now: Timestamp,
        new_videos: Vec<ObservedVideo>,
        new_texts: Vec<ObservedText>,
    ) -> Option<CrossModalReport> {
        for video in &new_videos {
            self.recent_video.push_back(video.clone());
        }
        for text in &new_texts {
            self.recent_text.push_back(text.clone());
        }
        self.prune_recent(now);
        let mut updated_pairs: HashSet<(String, String)> = HashSet::new();
        let mut new_links = 0usize;

        let recent_videos = self.recent_video.iter().cloned().collect::<Vec<_>>();
        let recent_texts = self.recent_text.iter().cloned().collect::<Vec<_>>();
        for text in &new_texts {
            for video in &recent_videos {
                if let Some(weight) = match_weight(text, video, self.config.temporal_tolerance_secs) {
                    if weight < self.config.min_confidence {
                        continue;
                    }
                    if self.update_link(text, video, weight, &mut updated_pairs) {
                        new_links += 1;
                    }
                }
            }
        }
        for video in &new_videos {
            for text in &recent_texts {
                if let Some(weight) = match_weight(text, video, self.config.temporal_tolerance_secs) {
                    if weight < self.config.min_confidence {
                        continue;
                    }
                    if self.update_link(text, video, weight, &mut updated_pairs) {
                        new_links += 1;
                    }
                }
            }
        }

        self.trim_links();
        Some(self.build_report(now, &new_videos, &new_texts, new_links))
    }

    fn update_spike(
        &mut self,
        now: Timestamp,
        new_videos: Vec<ObservedVideo>,
        new_texts: Vec<ObservedText>,
    ) -> Option<CrossModalReport> {
        let (spiked_videos, spiked_texts) =
            self.spike_observed(now, &new_videos, &new_texts);
        if spiked_videos.is_empty() && spiked_texts.is_empty() {
            self.prune_recent(now);
            return None;
        }
        for video in &spiked_videos {
            self.recent_video.push_back(video.clone());
        }
        for text in &spiked_texts {
            self.recent_text.push_back(text.clone());
        }
        self.prune_recent(now);

        let mut updated_pairs: HashSet<(String, String)> = HashSet::new();
        let mut new_links = 0usize;
        let recent_videos = self.recent_video.iter().cloned().collect::<Vec<_>>();
        let recent_texts = self.recent_text.iter().cloned().collect::<Vec<_>>();
        for text in &spiked_texts {
            for video in &recent_videos {
                if let Some(weight) = match_weight(text, video, self.config.temporal_tolerance_secs) {
                    if weight < self.config.min_confidence {
                        continue;
                    }
                    if self.update_link(text, video, weight, &mut updated_pairs) {
                        new_links += 1;
                    }
                }
            }
        }
        for video in &spiked_videos {
            for text in &recent_texts {
                if let Some(weight) = match_weight(text, video, self.config.temporal_tolerance_secs) {
                    if weight < self.config.min_confidence {
                        continue;
                    }
                    if self.update_link(text, video, weight, &mut updated_pairs) {
                        new_links += 1;
                    }
                }
            }
        }

        self.trim_links();
        Some(self.build_report(now, &spiked_videos, &spiked_texts, new_links))
    }

    fn spike_observed(
        &mut self,
        now: Timestamp,
        new_videos: &[ObservedVideo],
        new_texts: &[ObservedText],
    ) -> (Vec<ObservedVideo>, Vec<ObservedText>) {
        let max_inputs = self.config.spike.max_inputs_per_modality.max(1);
        let selected_texts = select_top_by_confidence(new_texts, max_inputs, |t| t.confidence);
        let selected_videos = select_top_by_confidence(new_videos, max_inputs, |v| v.confidence);
        let mut inputs = Vec::new();
        let mut text_neurons = Vec::new();
        let mut video_neurons = Vec::new();

        for text in selected_texts {
            let label = text_neuron_label(&text.key);
            let Some(id) = self.neuron_for_label(&label, NeuronKind::Excitatory) else {
                continue;
            };
            let strength = spike_input_strength(text.confidence, self.config.spike.input_gain);
            text_neurons.push((text, id));
            inputs.push(SpikeInput {
                target: id,
                excitatory: strength,
                inhibitory: 0.0,
            });
        }

        for video in selected_videos {
            let label = video_neuron_label(&video.key);
            let Some(id) = self.neuron_for_label(&label, NeuronKind::Excitatory) else {
                continue;
            };
            let strength = spike_input_strength(video.confidence, self.config.spike.input_gain);
            video_neurons.push((video, id));
            inputs.push(SpikeInput {
                target: id,
                excitatory: strength,
                inhibitory: 0.0,
            });
        }

        if inputs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        self.spike_pool.enqueue_inputs(inputs);
        let frame = self.spike_pool.step(now);
        let mut spiked_ids = HashSet::new();
        for spike in frame.spikes {
            spiked_ids.insert(spike.neuron_id);
        }

        let mut spiked_texts = Vec::new();
        for (text, id) in text_neurons {
            if spiked_ids.contains(&id) {
                spiked_texts.push(text);
            }
        }
        let mut spiked_videos = Vec::new();
        for (video, id) in video_neurons {
            if spiked_ids.contains(&id) {
                spiked_videos.push(video);
            }
        }
        (spiked_videos, spiked_texts)
    }

    fn build_report(
        &self,
        now: Timestamp,
        new_videos: &[ObservedVideo],
        new_texts: &[ObservedText],
        new_links: usize,
    ) -> CrossModalReport {
        let mut top_links = self
            .associations
            .values()
            .filter(|link| link.support >= self.config.min_support as u64)
            .map(AssociationState::to_link)
            .collect::<Vec<_>>();
        sort_links(&mut top_links);
        top_links.truncate(self.config.max_report_links.max(1));

        let mut video_queries = Vec::new();
        for video in new_videos {
            let matches = self.query_text_for_video(&video.key, self.config.max_matches_per_item);
            if matches.is_empty() {
                continue;
            }
            video_queries.push(CrossModalQuery {
                source_key: video.key.clone(),
                matches,
            });
        }

        let mut text_queries = Vec::new();
        for text in new_texts {
            let matches = self.query_video_for_text(&text.key, self.config.max_matches_per_item);
            if matches.is_empty() {
                continue;
            }
            text_queries.push(CrossModalQuery {
                source_key: text.key.clone(),
                matches,
            });
        }

        CrossModalReport {
            timestamp: now,
            total_links: self.associations.len(),
            new_links,
            top_links,
            video_queries,
            text_queries,
        }
    }

    fn neuron_for_label(&mut self, label: &str, kind: NeuronKind) -> Option<u32> {
        if let Some(id) = self.label_to_neuron.get(label) {
            return Some(*id);
        }
        if self.spike_pool.neuron_count() >= self.config.spike.max_neurons {
            return None;
        }
        let id = self.spike_pool.add_neuron(kind);
        self.label_to_neuron.insert(label.to_string(), id);
        self.neuron_labels.insert(id, label.to_string());
        Some(id)
    }

    fn connect_once(&mut self, from: u32, to: u32, weight: f32) {
        if self.connections.insert((from, to)) {
            self.spike_pool.connect(from, to, weight);
        }
    }

    fn ensure_composite(&mut self, text_key: &str, video_key: &str) {
        let key = (text_key.to_string(), video_key.to_string());
        if self.composites.contains_key(&key) {
            return;
        }
        let text_label = text_neuron_label(text_key);
        let video_label = video_neuron_label(video_key);
        let Some(text_id) = self.neuron_for_label(&text_label, NeuronKind::Excitatory) else {
            return;
        };
        let Some(video_id) = self.neuron_for_label(&video_label, NeuronKind::Excitatory) else {
            return;
        };
        let composite_label = composite_neuron_label(text_key, video_key);
        let Some(comp_id) = self.neuron_for_label(&composite_label, NeuronKind::Inhibitory) else {
            return;
        };
        let exc = self.config.spike.composite_excitatory_weight;
        let inh = self.config.spike.composite_inhibitory_weight;
        self.connect_once(text_id, comp_id, exc);
        self.connect_once(video_id, comp_id, exc);
        self.connect_once(comp_id, text_id, inh);
        self.connect_once(comp_id, video_id, inh);
        self.composites.insert(key, comp_id);
    }

    fn update_link(
        &mut self,
        text: &ObservedText,
        video: &ObservedVideo,
        weight: f64,
        updated: &mut HashSet<(String, String)>,
    ) -> bool {
        let key = (text.key.clone(), video.key.clone());
        if !updated.insert(key.clone()) {
            return false;
        }
        let mut composite_keys: Option<(String, String)> = None;
        {
            let entry = self.associations.entry(key.clone()).or_insert_with(|| AssociationState {
                text_key: text.key.clone(),
                video_key: video.key.clone(),
                support: 0,
                weight: 0.0,
                last_seen: text.timestamp,
                text_preview: text.text_preview.clone(),
                entity_id: video.entity_id.clone().or(text.entity_id.clone()),
                frame_ref: video.frame_ref.clone().or(text.frame_ref.clone()),
            });
            entry.support = entry.support.saturating_add(1);
            entry.weight = blend(entry.weight, weight, 0.3);
            entry.last_seen = text.timestamp;
            if entry.text_preview.is_empty() {
                entry.text_preview = text.text_preview.clone();
            }
            if entry.entity_id.is_none() {
                entry.entity_id = video.entity_id.clone().or(text.entity_id.clone());
            }
            if entry.frame_ref.is_none() {
                entry.frame_ref = video.frame_ref.clone().or(text.frame_ref.clone());
            }
            if self.config.spike.enabled && entry.support >= self.config.min_support as u64 {
                composite_keys = Some((entry.text_key.clone(), entry.video_key.clone()));
            }
        }
        if let Some((text_key, video_key)) = composite_keys {
            self.ensure_composite(&text_key, &video_key);
        }
        self.video_index
            .entry(video.key.clone())
            .or_default()
            .insert(text.key.clone());
        self.text_index
            .entry(text.key.clone())
            .or_default()
            .insert(video.key.clone());
        true
    }

    fn trim_links(&mut self) {
        if self.associations.len() <= self.config.max_links {
            return;
        }
        let mut values = self.associations.values().cloned().collect::<Vec<_>>();
        values.sort_by(|a, b| {
            b.support
                .cmp(&a.support)
                .then_with(|| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| b.last_seen.unix.cmp(&a.last_seen.unix))
        });
        values.truncate(self.config.max_links.max(1));
        let keep: HashSet<(String, String)> = values
            .iter()
            .map(|link| (link.text_key.clone(), link.video_key.clone()))
            .collect();
        self.associations.retain(|key, _| keep.contains(key));
        self.rebuild_indexes();
    }

    fn prune_recent(&mut self, now: Timestamp) {
        let max_age = self.config.max_age_secs.max(0);
        if max_age == 0 {
            return;
        }
        while let Some(front) = self.recent_video.front() {
            if now.unix.saturating_sub(front.timestamp.unix) <= max_age {
                break;
            }
            self.recent_video.pop_front();
        }
        while let Some(front) = self.recent_text.front() {
            if now.unix.saturating_sub(front.timestamp.unix) <= max_age {
                break;
            }
            self.recent_text.pop_front();
        }
        while self.recent_video.len() > self.config.max_recent.max(1) {
            self.recent_video.pop_front();
        }
        while self.recent_text.len() > self.config.max_recent.max(1) {
            self.recent_text.pop_front();
        }
    }

    fn rebuild_indexes(&mut self) {
        self.video_index.clear();
        self.text_index.clear();
        for link in self.associations.values() {
            self.video_index
                .entry(link.video_key.clone())
                .or_default()
                .insert(link.text_key.clone());
            self.text_index
                .entry(link.text_key.clone())
                .or_default()
                .insert(link.video_key.clone());
        }
    }

    fn link_for(&self, text_key: &str, video_key: &str) -> Option<CrossModalLink> {
        let key = (text_key.to_string(), video_key.to_string());
        self.associations.get(&key).map(AssociationState::to_link)
    }
}

impl AssociationState {
    fn to_link(&self) -> CrossModalLink {
        CrossModalLink {
            text_key: self.text_key.clone(),
            video_key: self.video_key.clone(),
            support: self.support,
            weight: self.weight,
            last_seen: self.last_seen,
            text_preview: self.text_preview.clone(),
            entity_id: self.entity_id.clone(),
            frame_ref: self.frame_ref.clone(),
        }
    }
}

fn is_video_token(token: &EventToken) -> bool {
    matches!(token.source, Some(StreamSource::PeopleVideo))
}

fn is_text_token(token: &EventToken) -> bool {
    matches!(token.source, Some(StreamSource::TextAnnotations)) || token.kind == EventKind::TextAnnotation
}

fn observed_video(token: &EventToken) -> Option<ObservedVideo> {
    let key = video_key(token)?;
    Some(ObservedVideo {
        key,
        timestamp: token.onset,
        confidence: token.confidence.clamp(0.0, 1.0),
        entity_id: attr_string(&token.attributes, "entity_id"),
        frame_ref: attr_frame_ref(&token.attributes),
    })
}

fn observed_text(token: &EventToken, max_len: usize) -> Option<ObservedText> {
    let text = attr_text(&token.attributes)?;
    let key = text_key(token, &text);
    let preview = truncate_text(&text, max_len);
    Some(ObservedText {
        key,
        timestamp: token.onset,
        confidence: token.confidence.clamp(0.0, 1.0),
        entity_id: attr_string(&token.attributes, "entity_id"),
        frame_ref: attr_frame_ref(&token.attributes),
        text_preview: preview,
    })
}

fn match_weight(text: &ObservedText, video: &ObservedVideo, tolerance_secs: f64) -> Option<f64> {
    if let (Some(frame_text), Some(frame_video)) = (&text.frame_ref, &video.frame_ref) {
        if frame_text != frame_video {
            return None;
        }
    }
    if let (Some(entity_text), Some(entity_video)) = (&text.entity_id, &video.entity_id) {
        if entity_text != entity_video {
            return None;
        }
    }
    let diff = (text.timestamp.unix - video.timestamp.unix).abs() as f64;
    let time_weight = if tolerance_secs <= 0.0 {
        if diff == 0.0 { 1.0 } else { 0.0 }
    } else {
        (1.0 - diff / tolerance_secs).clamp(0.0, 1.0)
    };
    if time_weight <= 0.0 {
        return None;
    }
    let mut weight = (text.confidence * video.confidence).sqrt().clamp(0.0, 1.0);
    if text.frame_ref.is_some() && video.frame_ref == text.frame_ref {
        weight = (weight * 1.15).clamp(0.0, 1.0);
    }
    if text.entity_id.is_some() && video.entity_id == text.entity_id {
        weight = (weight * 1.05).clamp(0.0, 1.0);
    }
    Some((weight * time_weight).clamp(0.0, 1.0))
}

fn video_key(token: &EventToken) -> Option<String> {
    if let Some(frame) = attr_frame_ref(&token.attributes) {
        return Some(format!("frame::{frame}"));
    }
    if let Some(entity_id) = attr_string(&token.attributes, "entity_id") {
        return Some(format!("entity::{entity_id}"));
    }
    if let Some(track_id) = attr_string(&token.attributes, "track_id") {
        return Some(format!("track::{track_id}"));
    }
    if !token.id.trim().is_empty() {
        return Some(format!("token::{}", token.id.trim()));
    }
    let hash = compute_payload_hash(format!("{:?}|{}", token.kind, token.onset.unix).as_bytes());
    Some(format!("token::{hash}"))
}

fn text_key(token: &EventToken, text: &str) -> String {
    if let Some(annotation_id) = attr_string(&token.attributes, "annotation_id") {
        return format!("annotation::{annotation_id}");
    }
    let hash = compute_payload_hash(text.as_bytes());
    format!("text::{hash}")
}

fn attr_string(attrs: &HashMap<String, Value>, key: &str) -> Option<String> {
    attrs.get(key).and_then(|val| val.as_str()).map(|s| s.to_string())
}

fn attr_text(attrs: &HashMap<String, Value>) -> Option<String> {
    for key in ["text", "label", "caption", "description"] {
        if let Some(text) = attrs.get(key).and_then(|val| val.as_str()) {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn attr_frame_ref(attrs: &HashMap<String, Value>) -> Option<String> {
    for key in ["frame_id", "frame_ref", "frame_path", "image_ref"] {
        if let Some(text) = attrs.get(key).and_then(|val| val.as_str()) {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let trimmed = text.trim();
    if trimmed.len() <= max_len {
        return trimmed.to_string();
    }
    let mut out = trimmed[..max_len].to_string();
    out.push_str("...");
    out
}

fn sort_links(links: &mut Vec<CrossModalLink>) {
    links.sort_by(|a, b| {
        b.support
            .cmp(&a.support)
            .then_with(|| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| b.last_seen.unix.cmp(&a.last_seen.unix))
    });
}

fn blend(current: f64, incoming: f64, alpha: f64) -> f64 {
    let alpha = alpha.clamp(0.0, 1.0);
    current * (1.0 - alpha) + incoming * alpha
}

fn spike_input_strength(confidence: f64, gain: f32) -> f32 {
    let input = (confidence.clamp(0.0, 1.0) as f32) * gain;
    input.clamp(0.0, 2.0)
}

fn select_top_by_confidence<T, F>(items: &[T], limit: usize, score: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T) -> f64,
{
    if items.is_empty() {
        return Vec::new();
    }
    let mut ranked = items.to_vec();
    ranked.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(limit.max(1));
    ranked
}

fn text_neuron_label(key: &str) -> String {
    format!("text::{key}")
}

fn video_neuron_label(key: &str) -> String {
    format!("video::{key}")
}

fn composite_neuron_label(text_key: &str, video_key: &str) -> String {
    format!("comp::{text_key}+{video_key}")
}

// ── Tri-modal linker: audio ↔ video ↔ text ───────────────────────────────────

/// An audio observation extracted from a token batch.
#[derive(Debug, Clone)]
struct ObservedAudio {
    key: String,
    timestamp: Timestamp,
    confidence: f64,
    entity_id: Option<String>,
    frame_ref: Option<String>,
}

/// A three-way association between a text token, a video token, and an audio token
/// that co-occurred within the temporal tolerance window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriModalLink {
    pub text_key: String,
    pub video_key: String,
    pub audio_key: String,
    pub support: u64,
    pub weight: f64,
    pub last_seen: Timestamp,
    #[serde(default)]
    pub text_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriModalReport {
    pub timestamp: Timestamp,
    pub total_links: usize,
    pub new_links: usize,
    #[serde(default)]
    pub top_links: Vec<TriModalLink>,
}

#[derive(Debug, Clone)]
struct TriAssociationState {
    text_key: String,
    video_key: String,
    audio_key: String,
    support: u64,
    weight: f64,
    last_seen: Timestamp,
    text_preview: String,
}

impl TriAssociationState {
    fn to_link(&self) -> TriModalLink {
        TriModalLink {
            text_key: self.text_key.clone(),
            video_key: self.video_key.clone(),
            audio_key: self.audio_key.clone(),
            support: self.support,
            weight: self.weight,
            last_seen: self.last_seen,
            text_preview: self.text_preview.clone(),
        }
    }
}

/// Three-way co-occurrence linker: audio ↔ video ↔ text.
///
/// When all three modalities appear within `temporal_tolerance_secs` of each
/// other in the same token batch, their association is strengthened.  Pairwise
/// audio↔video and audio↔text links are also accumulated as a by-product and
/// are queryable independently.
///
/// Reuses `CrossModalConfig` — no extra configuration fields required.
pub struct TriModalRuntime {
    config: CrossModalConfig,
    /// Three-way association table keyed by (text_key, video_key, audio_key).
    tri: HashMap<(String, String, String), TriAssociationState>,
    /// audio_key → set of text_keys seen together.
    audio_text: HashMap<String, HashSet<String>>,
    /// audio_key → set of video_keys seen together.
    audio_video: HashMap<String, HashSet<String>>,
    recent_audio: VecDeque<ObservedAudio>,
    recent_video: VecDeque<ObservedVideo>,
    recent_text: VecDeque<ObservedText>,
}

impl TriModalRuntime {
    pub fn new(config: CrossModalConfig) -> Self {
        Self {
            config,
            tri: HashMap::new(),
            audio_text: HashMap::new(),
            audio_video: HashMap::new(),
            recent_audio: VecDeque::new(),
            recent_video: VecDeque::new(),
            recent_text: VecDeque::new(),
        }
    }

    /// Ingest a token batch. Returns a report when new links are formed.
    pub fn update(&mut self, batch: &TokenBatch) -> Option<TriModalReport> {
        if !self.config.enabled {
            return None;
        }
        let now = batch.timestamp;
        let mut new_audios: Vec<ObservedAudio> = Vec::new();
        let mut new_videos: Vec<ObservedVideo> = Vec::new();
        let mut new_texts: Vec<ObservedText> = Vec::new();

        for token in &batch.tokens {
            if is_audio_token(token) {
                if let Some(a) = observed_audio(token) { new_audios.push(a); }
            } else if is_video_token(token) {
                if let Some(v) = observed_video(token) { new_videos.push(v); }
            } else if is_text_token(token) {
                if let Some(t) = observed_text(token, self.config.max_text_len) { new_texts.push(t); }
            }
        }

        if new_audios.is_empty() && new_videos.is_empty() && new_texts.is_empty() {
            self.prune(now);
            return None;
        }

        for a in &new_audios { self.recent_audio.push_back(a.clone()); }
        for v in &new_videos { self.recent_video.push_back(v.clone()); }
        for t in &new_texts  { self.recent_text.push_back(t.clone()); }
        self.prune(now);

        // Snapshot as owned Vecs to avoid holding borrows on self while mutating self.tri.
        let all_audios: Vec<ObservedAudio> = self.recent_audio.iter().cloned().collect();
        let all_videos: Vec<ObservedVideo> = self.recent_video.iter().cloned().collect();
        let all_texts:  Vec<ObservedText>  = self.recent_text.iter().cloned().collect();

        let tol = self.config.temporal_tolerance_secs;
        let min_conf = self.config.min_confidence;
        let mut new_links = 0usize;

        // Match triples — at least one modality must be new in this batch.
        let new_audio_keys: HashSet<&str> = new_audios.iter().map(|a| a.key.as_str()).collect();
        let new_video_keys: HashSet<&str> = new_videos.iter().map(|v| v.key.as_str()).collect();
        let new_text_keys:  HashSet<&str> = new_texts.iter().map(|t| t.key.as_str()).collect();

        for audio in &all_audios {
            for video in &all_videos {
                // Skip if neither audio nor video is new (text may be).
                if !new_audio_keys.contains(audio.key.as_str())
                    && !new_video_keys.contains(video.key.as_str())
                {
                    continue;
                }
                let av_w = match match_weight_audio(audio, video, tol) {
                    Some(w) if w >= min_conf => w,
                    _ => continue,
                };
                for text in &all_texts {
                    if !new_audio_keys.contains(audio.key.as_str())
                        && !new_video_keys.contains(video.key.as_str())
                        && !new_text_keys.contains(text.key.as_str())
                    {
                        continue;
                    }
                    let at_w = match match_weight_text_audio(text, audio, tol) {
                        Some(w) if w >= min_conf => w,
                        _ => continue,
                    };
                    let triple_w = (av_w * at_w).sqrt().clamp(0.0, 1.0);
                    let tkey = (text.key.clone(), video.key.clone(), audio.key.clone());
                    let is_new = !self.tri.contains_key(&tkey);
                    {
                        let entry = self.tri.entry(tkey).or_insert_with(|| TriAssociationState {
                            text_key:     text.key.clone(),
                            video_key:    video.key.clone(),
                            audio_key:    audio.key.clone(),
                            support:      0,
                            weight:       0.0,
                            last_seen:    now,
                            text_preview: text.text_preview.clone(),
                        });
                        entry.support = entry.support.saturating_add(1);
                        entry.weight  = blend(entry.weight, triple_w, 0.3);
                        entry.last_seen = now;
                    } // entry borrow ends — safe to mutate indexes below
                    if is_new {
                        new_links += 1;
                        self.audio_text.entry(audio.key.clone()).or_default().insert(text.key.clone());
                        self.audio_video.entry(audio.key.clone()).or_default().insert(video.key.clone());
                    }
                }
            }
        }

        if new_links == 0 && self.tri.is_empty() {
            return None;
        }

        self.trim();
        let top = self.top_links(self.config.max_report_links);
        Some(TriModalReport {
            timestamp: now,
            total_links: self.tri.len(),
            new_links,
            top_links: top,
        })
    }

    /// Given an audio key, return (text, video) pairs that co-occurred with it.
    pub fn query_for_audio(&self, audio_key: &str, limit: usize) -> Vec<TriModalLink> {
        let mut links: Vec<TriModalLink> = self.tri.values()
            .filter(|s| s.audio_key == audio_key)
            .map(|s| s.to_link())
            .collect();
        sort_tri_links(&mut links);
        links.truncate(limit.max(1));
        links
    }

    /// Given a text key, return (video, audio) pairs that co-occurred with it.
    pub fn query_for_text(&self, text_key: &str, limit: usize) -> Vec<TriModalLink> {
        let mut links: Vec<TriModalLink> = self.tri.values()
            .filter(|s| s.text_key == text_key)
            .map(|s| s.to_link())
            .collect();
        sort_tri_links(&mut links);
        links.truncate(limit.max(1));
        links
    }

    /// Given a video key, return (text, audio) pairs that co-occurred with it.
    pub fn query_for_video(&self, video_key: &str, limit: usize) -> Vec<TriModalLink> {
        let mut links: Vec<TriModalLink> = self.tri.values()
            .filter(|s| s.video_key == video_key)
            .map(|s| s.to_link())
            .collect();
        sort_tri_links(&mut links);
        links.truncate(limit.max(1));
        links
    }

    /// Ingest a TriModalReport produced by another node (gossip / federation).
    pub fn ingest_report(&mut self, report: &TriModalReport) {
        if !self.config.enabled { return; }
        for link in &report.top_links {
            let key = (link.text_key.clone(), link.video_key.clone(), link.audio_key.clone());
            let is_new = !self.tri.contains_key(&key);
            {
                let entry = self.tri.entry(key).or_insert_with(|| TriAssociationState {
                    text_key:     link.text_key.clone(),
                    video_key:    link.video_key.clone(),
                    audio_key:    link.audio_key.clone(),
                    support:      0,
                    weight:       0.0,
                    last_seen:    link.last_seen,
                    text_preview: link.text_preview.clone(),
                });
                entry.support = entry.support.saturating_add(link.support);
                entry.weight  = blend(entry.weight, link.weight, 0.3).clamp(0.0, 1.0);
                entry.last_seen = link.last_seen;
            }
            if is_new {
                self.audio_text.entry(link.audio_key.clone()).or_default().insert(link.text_key.clone());
                self.audio_video.entry(link.audio_key.clone()).or_default().insert(link.video_key.clone());
            }
        }
        self.trim();
    }

    fn prune(&mut self, now: Timestamp) {
        let max_age = self.config.max_age_secs;
        let max_recent = self.config.max_recent;
        prune_deque(&mut self.recent_audio, now, max_age, max_recent);
        prune_deque(&mut self.recent_video, now, max_age, max_recent);
        prune_deque(&mut self.recent_text, now, max_age, max_recent);
    }

    fn trim(&mut self) {
        let max = self.config.max_links;
        if self.tri.len() <= max { return; }
        // Evict lowest-support entries.
        let mut entries: Vec<_> = self.tri.iter()
            .map(|(k, s)| (k.clone(), s.support))
            .collect();
        entries.sort_by_key(|(_, sup)| *sup);
        let remove = self.tri.len() - max;
        for (key, _) in entries.into_iter().take(remove) {
            self.tri.remove(&key);
        }
    }

    fn top_links(&self, n: usize) -> Vec<TriModalLink> {
        let mut links: Vec<TriModalLink> = self.tri.values().map(|s| s.to_link()).collect();
        sort_tri_links(&mut links);
        links.truncate(n);
        links
    }
}

fn is_audio_token(token: &EventToken) -> bool {
    matches!(token.source, Some(StreamSource::AudioFrame) | Some(StreamSource::VideoFrame))
}

fn observed_audio(token: &EventToken) -> Option<ObservedAudio> {
    let key = audio_key(token)?;
    Some(ObservedAudio {
        key,
        timestamp: token.onset,
        confidence: token.confidence.clamp(0.0, 1.0),
        entity_id: attr_string(&token.attributes, "entity_id"),
        frame_ref: attr_frame_ref(&token.attributes),
    })
}

fn audio_key(token: &EventToken) -> Option<String> {
    // Prefer an explicit segment/clip ID.
    for attr in ["segment_id", "clip_id", "audio_id"] {
        if let Some(v) = attr_string(&token.attributes, attr) {
            return Some(format!("audio::{v}"));
        }
    }
    if let Some(frame) = attr_frame_ref(&token.attributes) {
        return Some(format!("audio_frame::{frame}"));
    }
    if let Some(entity) = attr_string(&token.attributes, "entity_id") {
        return Some(format!("audio_entity::{entity}"));
    }
    if !token.id.trim().is_empty() {
        return Some(format!("audio_token::{}", token.id.trim()));
    }
    let hash = compute_payload_hash(
        format!("audio|{}|{}", token.onset.unix, token.confidence.to_bits()).as_bytes(),
    );
    Some(format!("audio_hash::{hash}"))
}

/// Temporal + entity/frame match weight between an audio and video observation.
fn match_weight_audio(audio: &ObservedAudio, video: &ObservedVideo, tolerance_secs: f64) -> Option<f64> {
    if let (Some(fa), Some(fv)) = (&audio.frame_ref, &video.frame_ref) {
        if fa != fv { return None; }
    }
    if let (Some(ea), Some(ev)) = (&audio.entity_id, &video.entity_id) {
        if ea != ev { return None; }
    }
    let diff = (audio.timestamp.unix - video.timestamp.unix).abs() as f64;
    let time_weight = if tolerance_secs <= 0.0 {
        if diff == 0.0 { 1.0 } else { 0.0 }
    } else {
        (1.0 - diff / tolerance_secs).clamp(0.0, 1.0)
    };
    if time_weight <= 0.0 { return None; }
    let w = (audio.confidence * video.confidence).sqrt().clamp(0.0, 1.0);
    Some((w * time_weight).clamp(0.0, 1.0))
}

/// Temporal + entity/frame match weight between a text and audio observation.
fn match_weight_text_audio(text: &ObservedText, audio: &ObservedAudio, tolerance_secs: f64) -> Option<f64> {
    if let (Some(ft), Some(fa)) = (&text.frame_ref, &audio.frame_ref) {
        if ft != fa { return None; }
    }
    if let (Some(et), Some(ea)) = (&text.entity_id, &audio.entity_id) {
        if et != ea { return None; }
    }
    let diff = (text.timestamp.unix - audio.timestamp.unix).abs() as f64;
    let time_weight = if tolerance_secs <= 0.0 {
        if diff == 0.0 { 1.0 } else { 0.0 }
    } else {
        (1.0 - diff / tolerance_secs).clamp(0.0, 1.0)
    };
    if time_weight <= 0.0 { return None; }
    let w = (text.confidence * audio.confidence).sqrt().clamp(0.0, 1.0);
    Some((w * time_weight).clamp(0.0, 1.0))
}

fn prune_deque<T>(deque: &mut VecDeque<T>, now: Timestamp, max_age: i64, max_len: usize)
where
    T: HasTimestamp,
{
    while let Some(front) = deque.front() {
        if (now.unix - front.ts().unix).abs() > max_age {
            deque.pop_front();
        } else {
            break;
        }
    }
    while deque.len() > max_len {
        deque.pop_front();
    }
}

trait HasTimestamp {
    fn ts(&self) -> Timestamp;
}
impl HasTimestamp for ObservedAudio { fn ts(&self) -> Timestamp { self.timestamp } }
impl HasTimestamp for ObservedVideo  { fn ts(&self) -> Timestamp { self.timestamp } }
impl HasTimestamp for ObservedText   { fn ts(&self) -> Timestamp { self.timestamp } }

fn sort_tri_links(links: &mut Vec<TriModalLink>) {
    links.sort_by(|a, b| {
        b.support.cmp(&a.support)
            .then_with(|| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| b.last_seen.unix.cmp(&a.last_seen.unix))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CrossModalSpikeConfig;
    use crate::schema::Timestamp;

    #[test]
    fn cross_modal_links_text_to_video() {
        let config = CrossModalConfig {
            enabled: true,
            temporal_tolerance_secs: 2.0,
            min_support: 1,
            max_links: 16,
            max_recent: 16,
            min_confidence: 0.1,
            max_text_len: 120,
            max_age_secs: 30,
            max_matches_per_item: 4,
            max_report_links: 8,
            spike: CrossModalSpikeConfig {
                enabled: true,
                max_neurons: 128,
                max_inputs_per_modality: 8,
                input_gain: 1.4,
                threshold: 0.8,
                membrane_decay: 0.95,
                refractory_steps: 1,
                composite_excitatory_weight: 0.6,
                composite_inhibitory_weight: 0.8,
            },
        };
        let mut runtime = CrossModalRuntime::new(config);
        let ts = Timestamp { unix: 100 };
        let batch = TokenBatch {
            timestamp: ts,
            tokens: vec![
                EventToken {
                    id: "v1".to_string(),
                    kind: EventKind::BehavioralAtom,
                    onset: ts,
                    duration_secs: 1.0,
                    confidence: 0.9,
                    attributes: HashMap::from([
                        ("entity_id".to_string(), Value::String("e1".to_string())),
                        ("frame_id".to_string(), Value::String("f1".to_string())),
                    ]),
                    source: Some(StreamSource::PeopleVideo),
                },
                EventToken {
                    id: "t1".to_string(),
                    kind: EventKind::TextAnnotation,
                    onset: ts,
                    duration_secs: 1.0,
                    confidence: 0.8,
                    attributes: HashMap::from([
                        ("text".to_string(), Value::String("Person raises hand".to_string())),
                        ("entity_id".to_string(), Value::String("e1".to_string())),
                        ("frame_id".to_string(), Value::String("f1".to_string())),
                    ]),
                    source: Some(StreamSource::TextAnnotations),
                },
            ],
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let report = runtime.update(&batch).expect("report");
        assert!(!report.top_links.is_empty());
        assert!(report
            .video_queries
            .iter()
            .any(|query| !query.matches.is_empty()));
    }
}
