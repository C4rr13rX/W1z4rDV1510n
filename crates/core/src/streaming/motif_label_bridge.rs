use crate::blockchain::WorkKind;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::behavior::{BehaviorFrame, BehaviorMotif};
use crate::streaming::hierarchical_motifs::MetaMotif;
use crate::streaming::schema::{EventToken, LayerState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifLabelBridgeConfig {
    pub enabled: bool,
    pub max_pending: usize,
    pub include_meta_motifs: bool,
    pub max_context_tokens: usize,
    pub max_context_layers: usize,
    pub min_confidence: f64,
}

impl Default for MotifLabelBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_pending: 512,
            include_meta_motifs: true,
            max_context_tokens: 32,
            max_context_layers: 16,
            min_confidence: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifLabelTask {
    pub id: String,
    pub work_id: String,
    pub work_kind: WorkKind,
    pub timestamp: Timestamp,
    pub motif_id: String,
    pub entity_id: String,
    /// 0 = atomic motif, 1+ = meta-motif level.
    pub level: usize,
    pub summary: String,
    pub priority: f64,
    pub reward_score: f64,
    pub snapshot: MotifSnapshot,
    #[serde(default)]
    pub evidence: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifSnapshot {
    pub motif_id: String,
    pub entity_id: String,
    pub level: usize,
    pub support: usize,
    pub duration_secs: f64,
    pub children: Vec<String>,
    /// Mean of prototype vectors (for atomic motifs) or signature (for meta).
    pub prototype_summary: Vec<f64>,
    pub context_tokens: Vec<EventToken>,
    pub context_layers: Vec<LayerState>,
    pub frame_refs: Vec<String>,
    pub bounding_boxes: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifLabelReport {
    pub timestamp: Timestamp,
    pub pending: Vec<MotifLabelTask>,
    pub total_pending: usize,
    pub newly_queued: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic task ID derived from motif + entity + level.
fn task_id(motif_id: &str, entity_id: &str, level: usize) -> String {
    let raw = format!("{}:{}:{}", motif_id, entity_id, level);
    compute_payload_hash(raw.as_bytes())
}

/// Deterministic work ID for blockchain proof-of-useful-work linkage.
fn work_id(motif_id: &str, timestamp: &Timestamp) -> String {
    let raw = format!("annotation:{}:{}", motif_id, timestamp.unix);
    compute_payload_hash(raw.as_bytes())
}

/// Compute priority from motif support and duration — longer, more frequent
/// motifs are more valuable to label first.
fn compute_priority(support: usize, duration_secs: f64) -> f64 {
    let support_f = (support as f64).ln_1p();
    let duration_f = duration_secs.ln_1p();
    support_f * 0.6 + duration_f * 0.4
}

/// Compute reward score (higher for rarer, shorter motifs that are harder to
/// label and therefore merit higher incentive).
fn compute_reward(support: usize, duration_secs: f64) -> f64 {
    let rarity = 1.0 / (1.0 + support as f64);
    let brevity = 1.0 / (1.0 + duration_secs);
    (rarity * 0.5 + brevity * 0.5).clamp(0.0, 1.0)
}

/// Build a one-line summary for an atomic motif.
fn atomic_summary(motif: &BehaviorMotif) -> String {
    format!(
        "Atomic motif {} (entity {}, support={}, {:.1}s)",
        motif.id, motif.entity_id, motif.support, motif.duration_secs
    )
}

/// Mean-pool a set of prototype vectors into a single summary vector.
fn prototype_mean(prototype: &[Vec<f64>]) -> Vec<f64> {
    if prototype.is_empty() {
        return Vec::new();
    }
    let dim = prototype[0].len();
    if dim == 0 {
        return Vec::new();
    }
    let n = prototype.len() as f64;
    let mut mean = vec![0.0f64; dim];
    for row in prototype {
        for (i, v) in row.iter().enumerate() {
            if i < dim {
                mean[i] += v;
            }
        }
    }
    for v in &mut mean {
        *v /= n;
    }
    mean
}

/// Truncate slices to configured limits and clone into owned vecs.
fn clip_tokens(tokens: &[EventToken], max: usize) -> Vec<EventToken> {
    tokens.iter().take(max).cloned().collect()
}

fn clip_layers(layers: &[LayerState], max: usize) -> Vec<LayerState> {
    layers.iter().take(max).cloned().collect()
}

// ---------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------

pub struct MotifLabelBridge {
    config: MotifLabelBridgeConfig,
    pending: VecDeque<MotifLabelTask>,
    seen: HashSet<String>,
    newly_queued: usize,
}

impl MotifLabelBridge {
    pub fn new(config: MotifLabelBridgeConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            seen: HashSet::new(),
            newly_queued: 0,
        }
    }

    /// Queue atomic motifs from a behaviour frame together with their
    /// surrounding context tokens and layer states.
    pub fn queue_motifs(
        &mut self,
        frame: &BehaviorFrame,
        context_tokens: &[EventToken],
        context_layers: &[LayerState],
        timestamp: Timestamp,
    ) -> Option<MotifLabelReport> {
        if !self.config.enabled {
            return None;
        }

        self.newly_queued = 0;

        for motif in &frame.motifs {
            if self.pending.len() >= self.config.max_pending {
                break;
            }

            let tid = task_id(&motif.id, &motif.entity_id, 0);
            if self.seen.contains(&tid) {
                continue;
            }

            let snapshot = MotifSnapshot {
                motif_id: motif.id.clone(),
                entity_id: motif.entity_id.clone(),
                level: 0,
                support: motif.support,
                duration_secs: motif.duration_secs,
                children: Vec::new(),
                prototype_summary: prototype_mean(&motif.prototype),
                context_tokens: clip_tokens(context_tokens, self.config.max_context_tokens),
                context_layers: clip_layers(context_layers, self.config.max_context_layers),
                frame_refs: Vec::new(),
                bounding_boxes: Vec::new(),
            };

            let task = MotifLabelTask {
                id: tid.clone(),
                work_id: work_id(&motif.id, &timestamp),
                work_kind: WorkKind::HumanAnnotation,
                timestamp: timestamp.clone(),
                motif_id: motif.id.clone(),
                entity_id: motif.entity_id.clone(),
                level: 0,
                summary: atomic_summary(motif),
                priority: compute_priority(motif.support, motif.duration_secs),
                reward_score: compute_reward(motif.support, motif.duration_secs),
                snapshot,
                evidence: HashMap::new(),
            };

            self.seen.insert(tid);
            self.pending.push_back(task);
            self.newly_queued += 1;
        }

        if self.newly_queued == 0 {
            return None;
        }

        Some(self.build_report(timestamp))
    }

    /// Queue hierarchical meta-motifs discovered at any level above atomic.
    pub fn queue_meta_motifs(
        &mut self,
        meta_motifs: &[MetaMotif],
        timestamp: Timestamp,
    ) -> Option<MotifLabelReport> {
        if !self.config.enabled || !self.config.include_meta_motifs {
            return None;
        }

        self.newly_queued = 0;

        for mm in meta_motifs {
            if self.pending.len() >= self.config.max_pending {
                break;
            }

            // Use a placeholder entity for meta-motifs (they span entities).
            let entity_id = "meta".to_string();
            let tid = task_id(&mm.id, &entity_id, mm.level);
            if self.seen.contains(&tid) {
                continue;
            }

            let snapshot = MotifSnapshot {
                motif_id: mm.id.clone(),
                entity_id: entity_id.clone(),
                level: mm.level,
                support: mm.support,
                duration_secs: mm.duration_secs,
                children: mm.children.clone(),
                prototype_summary: mm.signature.clone(),
                context_tokens: Vec::new(),
                context_layers: Vec::new(),
                frame_refs: Vec::new(),
                bounding_boxes: Vec::new(),
            };

            let summary = format!(
                "Meta-motif {} (level={}, support={}, {:.1}s, entropy={:.3}, attractor={}): {}",
                mm.id, mm.level, mm.support, mm.duration_secs, mm.entropy, mm.is_attractor,
                mm.description,
            );

            let task = MotifLabelTask {
                id: tid.clone(),
                work_id: work_id(&mm.id, &timestamp),
                work_kind: WorkKind::HumanAnnotation,
                timestamp: timestamp.clone(),
                motif_id: mm.id.clone(),
                entity_id,
                level: mm.level,
                summary,
                priority: compute_priority(mm.support, mm.duration_secs),
                reward_score: compute_reward(mm.support, mm.duration_secs),
                snapshot,
                evidence: HashMap::new(),
            };

            self.seen.insert(tid);
            self.pending.push_back(task);
            self.newly_queued += 1;
        }

        if self.newly_queued == 0 {
            return None;
        }

        Some(self.build_report(timestamp))
    }

    /// Number of tasks waiting for human labels.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn build_report(&self, timestamp: Timestamp) -> MotifLabelReport {
        MotifLabelReport {
            timestamp,
            pending: self.pending.iter().cloned().collect(),
            total_pending: self.pending.len(),
            newly_queued: self.newly_queued,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::behavior::{
        BackpressureStatus, BehaviorGraph, BehaviorMotif, GraphSignature,
        MotifLineage, MotifProvenance, TimeFrequencySummary,
    };
    use crate::streaming::schema::EventKind;

    fn ts(unix: i64) -> Timestamp {
        Timestamp { unix }
    }

    fn make_motif(id: &str, entity: &str) -> BehaviorMotif {
        BehaviorMotif {
            id: id.to_string(),
            entity_id: entity.to_string(),
            support: 5,
            duration_secs: 2.0,
            description_length: 0.0,
            prototype: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            time_frequency: TimeFrequencySummary {
                amplitudes: vec![0.5],
                phases: vec![0.0],
            },
            graph_signature: GraphSignature {
                mean_proximity: 0.5,
                mean_coherence: 0.4,
            },
            lineage: MotifLineage::default(),
            provenance: MotifProvenance::default(),
        }
    }

    fn make_frame(motifs: Vec<BehaviorMotif>) -> BehaviorFrame {
        BehaviorFrame {
            timestamp: ts(100),
            states: Vec::new(),
            graph: BehaviorGraph {
                timestamp: ts(100),
                nodes: HashMap::new(),
                edges: Vec::new(),
            },
            motifs,
            motif_assignment_ambiguities: Vec::new(),
            prediction: None,
            backpressure: BackpressureStatus::Ok,
        }
    }

    fn make_meta(id: &str, level: usize) -> MetaMotif {
        MetaMotif {
            id: id.to_string(),
            level,
            children: vec!["child-a".into(), "child-b".into()],
            support: 10,
            duration_secs: 5.0,
            signature: vec![0.1, 0.2, 0.3],
            entropy: 1.5,
            is_attractor: false,
            description: "test meta motif".into(),
        }
    }

    #[test]
    fn queue_motifs_creates_tasks() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig::default());
        let frame = make_frame(vec![
            make_motif("m1", "e1"),
            make_motif("m2", "e1"),
        ]);

        let report = bridge.queue_motifs(&frame, &[], &[], ts(200));
        assert!(report.is_some());
        let rpt = report.unwrap();
        assert_eq!(rpt.newly_queued, 2);
        assert_eq!(rpt.total_pending, 2);
        assert_eq!(bridge.pending_count(), 2);

        // All tasks should be HumanAnnotation at level 0.
        for task in &rpt.pending {
            assert_eq!(task.level, 0);
            assert!(matches!(task.work_kind, WorkKind::HumanAnnotation));
        }
    }

    #[test]
    fn duplicate_motifs_are_skipped() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig::default());
        let frame = make_frame(vec![make_motif("m1", "e1")]);

        bridge.queue_motifs(&frame, &[], &[], ts(100));
        let report = bridge.queue_motifs(&frame, &[], &[], ts(200));
        // Second call should produce nothing new.
        assert!(report.is_none());
        assert_eq!(bridge.pending_count(), 1);
    }

    #[test]
    fn max_pending_enforced() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig {
            max_pending: 1,
            ..MotifLabelBridgeConfig::default()
        });

        let frame = make_frame(vec![
            make_motif("m1", "e1"),
            make_motif("m2", "e1"),
        ]);

        let rpt = bridge.queue_motifs(&frame, &[], &[], ts(100)).unwrap();
        assert_eq!(rpt.newly_queued, 1);
        assert_eq!(bridge.pending_count(), 1);
    }

    #[test]
    fn queue_meta_motifs_creates_tasks() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig::default());
        let metas = vec![make_meta("mm1", 1), make_meta("mm2", 2)];

        let rpt = bridge.queue_meta_motifs(&metas, ts(300)).unwrap();
        assert_eq!(rpt.newly_queued, 2);
        assert_eq!(rpt.total_pending, 2);

        assert_eq!(rpt.pending[0].level, 1);
        assert_eq!(rpt.pending[1].level, 2);
        assert_eq!(rpt.pending[0].snapshot.children.len(), 2);
    }

    #[test]
    fn meta_motifs_skipped_when_disabled() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig {
            include_meta_motifs: false,
            ..MotifLabelBridgeConfig::default()
        });

        let metas = vec![make_meta("mm1", 1)];
        assert!(bridge.queue_meta_motifs(&metas, ts(300)).is_none());
        assert_eq!(bridge.pending_count(), 0);
    }

    #[test]
    fn disabled_bridge_queues_nothing() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig {
            enabled: false,
            ..MotifLabelBridgeConfig::default()
        });

        let frame = make_frame(vec![make_motif("m1", "e1")]);
        assert!(bridge.queue_motifs(&frame, &[], &[], ts(100)).is_none());
        assert_eq!(bridge.pending_count(), 0);
    }

    #[test]
    fn context_tokens_clipped_to_max() {
        let mut bridge = MotifLabelBridge::new(MotifLabelBridgeConfig {
            max_context_tokens: 2,
            ..MotifLabelBridgeConfig::default()
        });

        let tokens: Vec<EventToken> = (0..5)
            .map(|i| EventToken {
                id: format!("t{}", i),
                kind: EventKind::BehavioralAtom,
                onset: ts(i),
                duration_secs: 1.0,
                confidence: 0.9,
                attributes: HashMap::new(),
                source: None,
            })
            .collect();

        let frame = make_frame(vec![make_motif("m1", "e1")]);
        let rpt = bridge.queue_motifs(&frame, &tokens, &[], ts(100)).unwrap();
        assert_eq!(rpt.pending[0].snapshot.context_tokens.len(), 2);
    }

    #[test]
    fn prototype_summary_is_mean() {
        let proto = vec![vec![2.0, 4.0], vec![4.0, 6.0]];
        let mean = prototype_mean(&proto);
        assert_eq!(mean, vec![3.0, 5.0]);
    }
}
