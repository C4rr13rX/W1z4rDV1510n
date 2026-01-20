use crate::blockchain::WorkKind;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::behavior::{BehaviorFrame, BehaviorMotif, BehaviorState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifReplayFrame {
    pub timestamp: Timestamp,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    pub latent_norm: f64,
    pub action_norm: f64,
    pub confidence: f64,
    pub missing_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifReplay {
    pub id: String,
    pub motif_id: String,
    pub entity_id: String,
    pub start: Timestamp,
    pub end: Timestamp,
    pub duration_secs: f64,
    pub support: usize,
    pub description_length: f64,
    pub frame_count: usize,
    #[serde(default)]
    pub frames: Vec<MotifReplayFrame>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifPlaybackTask {
    pub id: String,
    pub work_id: String,
    pub work_kind: WorkKind,
    pub timestamp: Timestamp,
    pub summary: String,
    pub entity_id: String,
    pub motif_id: String,
    pub priority: f64,
    pub reward_score: f64,
    pub replay: MotifReplay,
    #[serde(default)]
    pub evidence: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifPlaybackReport {
    pub timestamp: Timestamp,
    pub pending: Vec<MotifPlaybackTask>,
    pub total_pending: usize,
}

#[derive(Debug, Clone)]
pub struct MotifPlaybackConfig {
    pub max_pending: usize,
    pub min_support: usize,
    pub max_frames: usize,
    pub min_priority: f64,
}

impl Default for MotifPlaybackConfig {
    fn default() -> Self {
        Self {
            max_pending: 256,
            min_support: 1,
            max_frames: 96,
            min_priority: 0.2,
        }
    }
}

pub struct MotifPlaybackQueue {
    config: MotifPlaybackConfig,
    pending: VecDeque<MotifPlaybackTask>,
    known_ids: HashSet<String>,
}

impl MotifPlaybackQueue {
    pub fn new(config: MotifPlaybackConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            known_ids: HashSet::new(),
        }
    }

    pub fn update(&mut self, replays: &[MotifReplay], now: Timestamp) -> Option<MotifPlaybackReport> {
        if replays.is_empty() {
            return None;
        }
        let mut added = false;
        for replay in replays {
            if replay.support < self.config.min_support {
                continue;
            }
            let priority = replay_priority(replay);
            if priority < self.config.min_priority {
                continue;
            }
            let id = playback_id(&replay.id, now);
            if !self.known_ids.insert(id.clone()) {
                continue;
            }
            let work_id = work_id_for(&replay.id, now);
            let summary = if replay.summary.is_empty() {
                format!(
                    "Motif replay {motif} for {entity} ({frames} frames)",
                    motif = replay.motif_id,
                    entity = replay.entity_id,
                    frames = replay.frame_count
                )
            } else {
                replay.summary.clone()
            };
            let mut evidence = HashMap::new();
            evidence.insert("support".to_string(), Value::from(replay.support as u64));
            evidence.insert(
                "duration_secs".to_string(),
                Value::from(replay.duration_secs),
            );
            evidence.insert(
                "frame_count".to_string(),
                Value::from(replay.frame_count as u64),
            );
            if let Some(frame) = replay.frames.first() {
                if let Some(pos) = frame.position {
                    evidence.insert("start_pos_x".to_string(), Value::from(pos[0]));
                    evidence.insert("start_pos_y".to_string(), Value::from(pos[1]));
                    evidence.insert("start_pos_z".to_string(), Value::from(pos[2]));
                }
            }
            let task = MotifPlaybackTask {
                id,
                work_id,
                work_kind: WorkKind::HumanAnnotation,
                timestamp: now,
                summary,
                entity_id: replay.entity_id.clone(),
                motif_id: replay.motif_id.clone(),
                priority,
                reward_score: priority,
                replay: replay.clone(),
                evidence,
            };
            self.pending.push_back(task);
            added = true;
        }
        while self.pending.len() > self.config.max_pending.max(1) {
            if let Some(task) = self.pending.pop_front() {
                self.known_ids.remove(&task.id);
            }
        }
        if !added {
            return None;
        }
        Some(MotifPlaybackReport {
            timestamp: now,
            pending: self.pending.iter().cloned().collect(),
            total_pending: self.pending.len(),
        })
    }
}

impl Default for MotifPlaybackQueue {
    fn default() -> Self {
        Self::new(MotifPlaybackConfig::default())
    }
}

pub fn build_motif_replays(
    frame: &BehaviorFrame,
    history: &[BehaviorState],
    max_frames: usize,
) -> Vec<MotifReplay> {
    if frame.motifs.is_empty() || history.is_empty() {
        return Vec::new();
    }
    let mut replays = Vec::new();
    for motif in &frame.motifs {
        if motif.entity_id.is_empty() {
            continue;
        }
        let end = frame.timestamp;
        let duration = motif.duration_secs.max(1.0);
        let start_unix = end
            .unix
            .saturating_sub(duration.round().max(1.0) as i64);
        let mut frames: Vec<MotifReplayFrame> = history
            .iter()
            .filter(|state| state.timestamp.unix >= start_unix)
            .map(|state| replay_frame(state))
            .collect();
        if frames.is_empty() {
            continue;
        }
        if frames.len() > max_frames.max(1) {
            let stride = (frames.len() / max_frames.max(1)).max(1);
            frames = frames.into_iter().step_by(stride).collect();
        }
        let frame_count = frames.len();
        let id = replay_id(motif, end);
        let mut attrs = HashMap::new();
        attrs.insert("entity_id".to_string(), Value::String(motif.entity_id.clone()));
        attrs.insert("motif_id".to_string(), Value::String(motif.id.clone()));
        replays.push(MotifReplay {
            id,
            motif_id: motif.id.clone(),
            entity_id: motif.entity_id.clone(),
            start: Timestamp { unix: start_unix },
            end,
            duration_secs: duration,
            support: motif.support,
            description_length: motif.description_length,
            frame_count,
            frames,
            summary: motif_summary(motif),
            attributes: attrs,
        });
    }
    replays
}

fn replay_frame(state: &BehaviorState) -> MotifReplayFrame {
    MotifReplayFrame {
        timestamp: state.timestamp,
        position: state
            .position
            .or_else(|| position_from_attrs(&state.attributes)),
        latent_norm: vector_norm(&state.latent),
        action_norm: vector_norm(&state.action),
        confidence: state.confidence,
        missing_ratio: state.missing_ratio,
    }
}

fn position_from_attrs(attrs: &HashMap<String, Value>) -> Option<[f64; 3]> {
    let x = attrs.get("pos_x").and_then(|v| v.as_f64());
    let y = attrs.get("pos_y").and_then(|v| v.as_f64());
    let z = attrs.get("pos_z").and_then(|v| v.as_f64()).unwrap_or(0.0);
    match (x, y) {
        (Some(x), Some(y)) => Some([x, y, z]),
        _ => None,
    }
}

fn motif_summary(motif: &BehaviorMotif) -> String {
    format!(
        "Motif {motif} (support {support}, duration {duration:.1}s)",
        motif = motif.id,
        support = motif.support,
        duration = motif.duration_secs
    )
}

fn replay_id(motif: &BehaviorMotif, ts: Timestamp) -> String {
    let payload = format!("motif-replay|{}|{}|{}", motif.id, motif.entity_id, ts.unix);
    compute_payload_hash(payload.as_bytes())
}

fn playback_id(replay_id: &str, ts: Timestamp) -> String {
    let payload = format!("playback|{}|{}", replay_id, ts.unix);
    compute_payload_hash(payload.as_bytes())
}

fn work_id_for(replay_id: &str, ts: Timestamp) -> String {
    let payload = format!("work|playback|{}|{}", replay_id, ts.unix);
    compute_payload_hash(payload.as_bytes())
}

fn replay_priority(replay: &MotifReplay) -> f64 {
    let support = replay.support as f64 / (replay.support as f64 + 3.0);
    let complexity = 1.0 / (1.0 + replay.description_length.abs());
    (0.6 * support + 0.4 * complexity).clamp(0.0, 1.0)
}

fn vector_norm(vec: &[f64]) -> f64 {
    let sum = vec.iter().map(|v| v * v).sum::<f64>();
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;

    #[test]
    fn playback_queue_emits_task() {
        let motif = MotifReplay {
            id: "r1".to_string(),
            motif_id: "m1".to_string(),
            entity_id: "e1".to_string(),
            start: Timestamp { unix: 10 },
            end: Timestamp { unix: 20 },
            duration_secs: 10.0,
            support: 3,
            description_length: 1.2,
            frame_count: 2,
            frames: vec![
                MotifReplayFrame {
                    timestamp: Timestamp { unix: 10 },
                    position: Some([1.0, 2.0, 0.0]),
                    latent_norm: 1.0,
                    action_norm: 0.4,
                    confidence: 0.9,
                    missing_ratio: 0.0,
                },
                MotifReplayFrame {
                    timestamp: Timestamp { unix: 20 },
                    position: Some([2.0, 2.5, 0.0]),
                    latent_norm: 1.2,
                    action_norm: 0.6,
                    confidence: 0.8,
                    missing_ratio: 0.1,
                },
            ],
            summary: String::new(),
            attributes: HashMap::new(),
        };
        let mut queue = MotifPlaybackQueue::default();
        let report = queue
            .update(&[motif], Timestamp { unix: 20 })
            .expect("report");
        assert_eq!(report.pending.len(), 1);
    }

    #[test]
    fn build_motif_replays_uses_history() {
        let motif = BehaviorMotif {
            id: "m1".to_string(),
            entity_id: "e1".to_string(),
            support: 2,
            duration_secs: 3.0,
            description_length: 1.0,
            prototype: Vec::new(),
            time_frequency: crate::streaming::behavior::TimeFrequencySummary {
                amplitudes: Vec::new(),
                phases: Vec::new(),
            },
            graph_signature: crate::streaming::behavior::GraphSignature {
                mean_coherence: 0.5,
                mean_proximity: 0.5,
            },
            lineage: crate::streaming::behavior::MotifLineage::default(),
            provenance: crate::streaming::behavior::MotifProvenance::default(),
        };
        let frame = BehaviorFrame {
            timestamp: Timestamp { unix: 10 },
            states: Vec::new(),
            graph: crate::streaming::behavior::BehaviorGraph {
                timestamp: Timestamp { unix: 10 },
                nodes: HashMap::new(),
                edges: Vec::new(),
            },
            motifs: vec![motif],
            prediction: None,
            backpressure: crate::streaming::behavior::BackpressureStatus::Ok,
        };
        let history = vec![
            BehaviorState {
                entity_id: "e1".to_string(),
                timestamp: Timestamp { unix: 8 },
                species: crate::streaming::behavior::SpeciesKind::Human,
                latent: vec![0.1, 0.2],
                action: vec![0.2],
                position: Some([1.0, 1.0, 0.0]),
                confidence: 0.9,
                missing_ratio: 0.0,
                attributes: HashMap::new(),
            },
            BehaviorState {
                entity_id: "e1".to_string(),
                timestamp: Timestamp { unix: 10 },
                species: crate::streaming::behavior::SpeciesKind::Human,
                latent: vec![0.2, 0.3],
                action: vec![0.2],
                position: Some([2.0, 1.5, 0.0]),
                confidence: 0.9,
                missing_ratio: 0.0,
                attributes: HashMap::new(),
            },
        ];
        let replays = build_motif_replays(&frame, &history, 10);
        assert_eq!(replays.len(), 1);
        assert_eq!(replays[0].frame_count, 2);
        assert!(replays[0]
            .frames
            .iter()
            .any(|frame| frame.position == Some([2.0, 1.5, 0.0])));
    }
}
