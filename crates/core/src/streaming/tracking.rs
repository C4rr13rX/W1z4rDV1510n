use crate::schema::Timestamp;
use crate::streaming::motor::{BoundingBox, PoseFrame};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PoseTrackerConfig {
    pub max_tracks: usize,
    pub max_idle_secs: f64,
    pub match_distance: f64,
    pub iou_weight: f64,
}

impl Default for PoseTrackerConfig {
    fn default() -> Self {
        Self {
            max_tracks: 1024,
            max_idle_secs: 5.0,
            match_distance: 1.5,
            iou_weight: 0.4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackingResult {
    pub track_id: String,
    pub confidence: f64,
    pub age_secs: f64,
    pub distance: f64,
    pub reused: bool,
}

#[derive(Debug, Clone)]
struct TrackState {
    track_id: String,
    first_seen: Timestamp,
    last_seen: Timestamp,
    centroid: (f64, f64),
    bbox: Option<BoundingBox>,
    hits: usize,
}

pub struct PoseTracker {
    config: PoseTrackerConfig,
    tracks: HashMap<String, TrackState>,
    next_id: u64,
}

impl PoseTracker {
    pub fn new(config: PoseTrackerConfig) -> Self {
        Self {
            config,
            tracks: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn assign(&mut self, frame: &PoseFrame) -> TrackingResult {
        let timestamp = frame.timestamp.unwrap_or(Timestamp { unix: 0 });
        self.prune(timestamp);
        let fixed_id = normalize_entity_id(&frame.entity_id);
        if let Some(id) = fixed_id {
            let centroid = pose_centroid(frame).unwrap_or((0.0, 0.0));
            let bbox = frame.bbox.clone();
            let state = self
                .tracks
                .entry(id.clone())
                .or_insert_with(|| TrackState {
                    track_id: id.clone(),
                    first_seen: timestamp,
                    last_seen: timestamp,
                    centroid,
                    bbox: bbox.clone(),
                    hits: 0,
                });
            state.last_seen = timestamp;
            state.centroid = centroid;
            state.bbox = bbox;
            state.hits = state.hits.saturating_add(1);
            let age_secs = (timestamp.unix - state.first_seen.unix).abs() as f64;
            return TrackingResult {
                track_id: id,
                confidence: 1.0,
                age_secs,
                distance: 0.0,
                reused: true,
            };
        }

        let Some(centroid) = pose_centroid(frame) else {
            return self.create_track(timestamp, (0.0, 0.0), None, 0.0, false);
        };
        let bbox = frame.bbox.clone();
        let mut best_id = None;
        let mut best_score = f64::INFINITY;
        let mut best_distance = 0.0;
        let mut best_state = None;
        for (id, state) in &self.tracks {
            let dt = (timestamp.unix - state.last_seen.unix).abs() as f64;
            if dt > self.config.max_idle_secs {
                continue;
            }
            let distance = distance_norm(centroid, state.centroid, &bbox, &state.bbox);
            let iou = bbox_iou(&bbox, &state.bbox);
            let score = distance / (1.0 + self.config.iou_weight * iou);
            if score < best_score {
                best_score = score;
                best_distance = distance;
                best_id = Some(id.clone());
                best_state = Some(state.clone());
            }
        }
        if let Some(id) = best_id {
            if best_score <= self.config.match_distance {
                let state = self.tracks.get_mut(&id).expect("track state");
                state.last_seen = timestamp;
                state.centroid = centroid;
                state.bbox = bbox.clone();
                state.hits = state.hits.saturating_add(1);
                let age_secs = (timestamp.unix - state.first_seen.unix).abs() as f64;
                let confidence =
                    track_confidence(best_score, best_state.as_ref().map(|s| s.hits).unwrap_or(0));
                return TrackingResult {
                    track_id: id,
                    confidence,
                    age_secs,
                    distance: best_distance,
                    reused: true,
                };
            }
        }
        self.create_track(timestamp, centroid, bbox, best_distance, false)
    }

    fn create_track(
        &mut self,
        timestamp: Timestamp,
        centroid: (f64, f64),
        bbox: Option<BoundingBox>,
        distance: f64,
        reused: bool,
    ) -> TrackingResult {
        let track_id = format!("track-{}", self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        let state = TrackState {
            track_id: track_id.clone(),
            first_seen: timestamp,
            last_seen: timestamp,
            centroid,
            bbox: bbox.clone(),
            hits: 1,
        };
        self.tracks.insert(track_id.clone(), state);
        self.enforce_limit();
        TrackingResult {
            track_id,
            confidence: 0.5,
            age_secs: 0.0,
            distance,
            reused,
        }
    }

    fn prune(&mut self, now: Timestamp) {
        let max_idle = self.config.max_idle_secs.max(0.0);
        self.tracks.retain(|_, state| {
            let dt = (now.unix - state.last_seen.unix).abs() as f64;
            dt <= max_idle
        });
        self.enforce_limit();
    }

    fn enforce_limit(&mut self) {
        let max_tracks = self.config.max_tracks.max(1);
        if self.tracks.len() <= max_tracks {
            return;
        }
        let mut entries: Vec<_> = self.tracks.values().cloned().collect();
        entries.sort_by_key(|state| state.last_seen.unix);
        let excess = entries.len().saturating_sub(max_tracks);
        for state in entries.iter().take(excess) {
            self.tracks.remove(&state.track_id);
        }
    }
}

impl Default for PoseTracker {
    fn default() -> Self {
        Self::new(PoseTrackerConfig::default())
    }
}

fn normalize_entity_id(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lowered = trimmed.to_ascii_lowercase();
    if matches!(lowered.as_str(), "unknown" | "person" | "entity") {
        return None;
    }
    Some(trimmed.to_string())
}

fn pose_centroid(frame: &PoseFrame) -> Option<(f64, f64)> {
    if !frame.keypoints.is_empty() {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0.0;
        for kp in &frame.keypoints {
            if kp.x.is_finite() && kp.y.is_finite() {
                sum_x += kp.x;
                sum_y += kp.y;
                count += 1.0;
            }
        }
        if count > 0.0 {
            return Some((sum_x / count, sum_y / count));
        }
    }
    frame.bbox.as_ref().map(|bbox| {
        (
            bbox.x + bbox.width * 0.5,
            bbox.y + bbox.height * 0.5,
        )
    })
}

fn distance_norm(
    a: (f64, f64),
    b: (f64, f64),
    bbox_a: &Option<BoundingBox>,
    bbox_b: &Option<BoundingBox>,
) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let diag = bbox_a
        .as_ref()
        .or_else(|| bbox_b.as_ref())
        .map(|bbox| {
            let w = bbox.width.abs();
            let h = bbox.height.abs();
            (w * w + h * h).sqrt().max(1e-6)
        })
        .unwrap_or(1.0);
    dist / diag
}

fn bbox_iou(a: &Option<BoundingBox>, b: &Option<BoundingBox>) -> f64 {
    let (Some(a), Some(b)) = (a.as_ref(), b.as_ref()) else {
        return 0.0;
    };
    let ax1 = a.x;
    let ay1 = a.y;
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx1 = b.x;
    let by1 = b.y;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;
    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = a.width.abs() * a.height.abs();
    let area_b = b.width.abs() * b.height.abs();
    let union = (area_a + area_b - inter).max(1e-6);
    (inter / union).clamp(0.0, 1.0)
}

fn track_confidence(score: f64, hits: usize) -> f64 {
    let hit_factor = hits as f64 / (hits as f64 + 2.0);
    (1.0 / (1.0 + score)).clamp(0.0, 1.0) * hit_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::motor::{Keypoint, PoseFrame};
    use std::collections::HashMap;

    #[test]
    fn tracker_reuses_track_for_nearby_pose() {
        let mut tracker = PoseTracker::default();
        let frame_a = PoseFrame {
            entity_id: "".to_string(),
            timestamp: Some(Timestamp { unix: 10 }),
            keypoints: vec![Keypoint {
                name: Some("head".to_string()),
                x: 1.0,
                y: 1.0,
                z: None,
                confidence: Some(1.0),
            }],
            bbox: None,
            metadata: HashMap::new(),
        };
        let frame_b = PoseFrame {
            entity_id: "".to_string(),
            timestamp: Some(Timestamp { unix: 11 }),
            keypoints: vec![Keypoint {
                name: Some("head".to_string()),
                x: 1.1,
                y: 1.0,
                z: None,
                confidence: Some(1.0),
            }],
            bbox: None,
            metadata: HashMap::new(),
        };
        let a = tracker.assign(&frame_a);
        let b = tracker.assign(&frame_b);
        assert_eq!(a.track_id, b.track_id);
        assert!(b.reused);
    }
}
