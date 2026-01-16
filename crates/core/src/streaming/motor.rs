use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Keypoint {
    pub name: Option<String>,
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PoseFrame {
    pub entity_id: String,
    #[serde(default)]
    pub timestamp: Option<Timestamp>,
    #[serde(default)]
    pub keypoints: Vec<Keypoint>,
    #[serde(default)]
    pub bbox: Option<BoundingBox>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct MotorFeatures {
    pub motion_energy: f64,
    pub motion_variance: f64,
    pub posture_shift: f64,
    pub micro_jitter: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct MotorFeatureOutput {
    pub entity_id: String,
    pub timestamp: Timestamp,
    pub keypoint_count: usize,
    pub bbox_area: Option<f64>,
    pub features: MotorFeatures,
    pub raw_signal: f64,
    pub normalized_signal: f64,
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub baseline_samples: usize,
    pub baseline_ready: bool,
    pub signal: f64,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct MotorConfig {
    pub velocity_norm: f64,
    pub jitter_norm: f64,
    pub posture_norm: f64,
    pub min_keypoints: usize,
    pub signal_alpha: f64,
    pub baseline_alpha: f64,
    pub baseline_min_samples: usize,
    pub baseline_std_floor: f64,
    pub baseline_norm_clamp: f64,
}

impl Default for MotorConfig {
    fn default() -> Self {
        Self {
            velocity_norm: 1.5,
            jitter_norm: 2.0,
            posture_norm: 0.75,
            min_keypoints: 3,
            signal_alpha: 0.3,
            baseline_alpha: 0.05,
            baseline_min_samples: 12,
            baseline_std_floor: 0.05,
            baseline_norm_clamp: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
struct KeypointState {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    confidence: f64,
}

#[derive(Debug, Clone)]
struct EntityMotionState {
    last_timestamp: Timestamp,
    keypoints: HashMap<String, KeypointState>,
    last_centroid: Option<(f64, f64)>,
    last_dispersion: f64,
    signal_ema: f64,
    baseline_mean: f64,
    baseline_var: f64,
    baseline_samples: usize,
}

impl EntityMotionState {
    fn new(timestamp: Timestamp) -> Self {
        Self {
            last_timestamp: timestamp,
            keypoints: HashMap::new(),
            last_centroid: None,
            last_dispersion: 0.0,
            signal_ema: 0.0,
            baseline_mean: 0.0,
            baseline_var: 0.0,
            baseline_samples: 0,
        }
    }

    fn update_baseline(
        &mut self,
        raw_signal: f64,
        config: &MotorConfig,
    ) -> (f64, f64, usize, bool, f64) {
        if self.baseline_samples == 0 {
            self.baseline_mean = raw_signal;
            self.baseline_var = 0.0;
            self.baseline_samples = 1;
        } else {
            let alpha = config.baseline_alpha.clamp(0.0, 1.0);
            let diff = raw_signal - self.baseline_mean;
            self.baseline_mean += alpha * diff;
            self.baseline_var = (1.0 - alpha) * (self.baseline_var + alpha * diff * diff);
            self.baseline_samples = self.baseline_samples.saturating_add(1);
        }
        let baseline_std = self.baseline_var.sqrt().max(config.baseline_std_floor);
        let baseline_ready = self.baseline_samples >= config.baseline_min_samples;
        let normalized = if baseline_ready {
            let clamp = config.baseline_norm_clamp.max(0.1);
            ((raw_signal - self.baseline_mean) / baseline_std).clamp(-clamp, clamp)
        } else {
            raw_signal
        };
        (
            self.baseline_mean,
            baseline_std,
            self.baseline_samples,
            baseline_ready,
            normalized,
        )
    }
}

pub struct MotorFeatureExtractor {
    config: MotorConfig,
    states: HashMap<String, EntityMotionState>,
}

impl MotorFeatureExtractor {
    pub fn new(config: MotorConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
        }
    }

    pub fn extract(&mut self, mut frame: PoseFrame) -> Option<MotorFeatureOutput> {
        let timestamp = frame.timestamp?;
        let (keypoints, bbox_area, dispersion_hint) = normalize_keypoints(&frame);
        if keypoints.is_empty() {
            return None;
        }
        let keypoint_count = keypoints.len();
        let state = self
            .states
            .entry(frame.entity_id.clone())
            .or_insert_with(|| EntityMotionState::new(timestamp));

        let dt = (timestamp.unix - state.last_timestamp.unix).abs() as f64;
        let dt = if dt > 0.0 { dt } else { 1.0 };

        let mut speed_sum = 0.0;
        let mut speed_sq_sum = 0.0;
        let mut jitter_sum = 0.0;
        let mut conf_sum = 0.0;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        let mut centroid_weight = 0.0;

        let mut next_keypoints = HashMap::new();
        for (label, kp) in keypoints {
            let conf = kp.confidence.clamp(0.0, 1.0);
            let prev = state.keypoints.get(&label);
            let (vx, vy, accel) = if let Some(prev_state) = prev {
                let dx = kp.x - prev_state.x;
                let dy = kp.y - prev_state.y;
                let vx = dx / dt;
                let vy = dy / dt;
                let dvx = vx - prev_state.vx;
                let dvy = vy - prev_state.vy;
                let accel = (dvx * dvx + dvy * dvy).sqrt() / dt;
                (vx, vy, accel)
            } else {
                (0.0, 0.0, 0.0)
            };
            let speed = (vx * vx + vy * vy).sqrt();
            speed_sum += speed * conf;
            speed_sq_sum += speed * speed;
            jitter_sum += accel * conf;
            conf_sum += conf;
            centroid_x += kp.x * conf;
            centroid_y += kp.y * conf;
            centroid_weight += conf;
            next_keypoints.insert(
                label,
                KeypointState {
                    x: kp.x,
                    y: kp.y,
                    vx,
                    vy,
                    confidence: conf,
                },
            );
        }

        let mean_speed = if conf_sum > 0.0 {
            speed_sum / conf_sum
        } else {
            0.0
        };
        let variance = if keypoint_count > 0 {
            let mean_unweighted = speed_sq_sum / keypoint_count as f64;
            (mean_unweighted - mean_speed * mean_speed).max(0.0)
        } else {
            0.0
        };
        let micro_jitter = if conf_sum > 0.0 {
            jitter_sum / conf_sum
        } else {
            0.0
        };
        let confidence = if keypoint_count > 0 {
            conf_sum / keypoint_count as f64
        } else {
            0.0
        };

        let (centroid_x, centroid_y) = if centroid_weight > 0.0 {
            (centroid_x / centroid_weight, centroid_y / centroid_weight)
        } else {
            (0.0, 0.0)
        };
        let mut dispersion = dispersion_hint.unwrap_or_else(|| {
            let mut sum = 0.0;
            for kp in next_keypoints.values() {
                let dx = kp.x - centroid_x;
                let dy = kp.y - centroid_y;
                sum += (dx * dx + dy * dy).sqrt();
            }
            if keypoint_count > 0 {
                sum / keypoint_count as f64
            } else {
                0.0
            }
        });
        if !dispersion.is_finite() {
            dispersion = 0.0;
        }

        let centroid_shift = state
            .last_centroid
            .map(|(px, py)| ((centroid_x - px).powi(2) + (centroid_y - py).powi(2)).sqrt())
            .unwrap_or(0.0);
        let dispersion_shift = (dispersion - state.last_dispersion).abs();
        let posture_shift = centroid_shift + dispersion_shift;

        let motion_norm = (mean_speed / self.config.velocity_norm).clamp(0.0, 1.0);
        let jitter_norm = (micro_jitter / self.config.jitter_norm).clamp(0.0, 1.0);
        let posture_norm = (posture_shift / self.config.posture_norm).clamp(0.0, 1.0);
        let raw_signal =
            (motion_norm * 0.6 + jitter_norm * 0.3 + posture_norm * 0.1).clamp(0.0, 1.0);
        let (baseline_mean, baseline_std, baseline_samples, baseline_ready, normalized_signal) =
            state.update_baseline(raw_signal, &self.config);
        let alpha = self.config.signal_alpha.clamp(0.0, 1.0);
        let signal = alpha * normalized_signal + (1.0 - alpha) * state.signal_ema;

        state.last_timestamp = timestamp;
        state.keypoints = next_keypoints;
        state.last_centroid = Some((centroid_x, centroid_y));
        state.last_dispersion = dispersion;
        state.signal_ema = signal;

        Some(MotorFeatureOutput {
            entity_id: frame.entity_id,
            timestamp,
            keypoint_count,
            bbox_area,
            features: MotorFeatures {
                motion_energy: mean_speed,
                motion_variance: variance,
                posture_shift,
                micro_jitter,
                confidence,
            },
            raw_signal,
            normalized_signal,
            baseline_mean,
            baseline_std,
            baseline_samples,
            baseline_ready,
            signal,
            metadata: std::mem::take(&mut frame.metadata),
        })
    }
}

fn normalize_keypoints(
    frame: &PoseFrame,
) -> (Vec<(String, KeypointState)>, Option<f64>, Option<f64>) {
    if !frame.keypoints.is_empty() {
        let mut normalized = Vec::with_capacity(frame.keypoints.len());
        for (idx, kp) in frame.keypoints.iter().enumerate() {
            let label = kp
                .name
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("k{idx}"));
            normalized.push((
                label,
                KeypointState {
                    x: kp.x,
                    y: kp.y,
                    vx: 0.0,
                    vy: 0.0,
                    confidence: kp.confidence.unwrap_or(1.0),
                },
            ));
        }
        return (normalized, None, None);
    }
    let Some(bbox) = &frame.bbox else {
        return (Vec::new(), None, None);
    };
    let area = bbox.width.abs() * bbox.height.abs();
    let center_x = bbox.x + bbox.width * 0.5;
    let center_y = bbox.y + bbox.height * 0.5;
    let dispersion_hint = (bbox.width.abs() + bbox.height.abs()) * 0.5;
    let normalized = vec![(
        "bbox_center".to_string(),
        KeypointState {
            x: center_x,
            y: center_y,
            vx: 0.0,
            vy: 0.0,
            confidence: 1.0,
        },
    )];
    (normalized, Some(area), Some(dispersion_hint))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_extractor_emits_signal_for_motion() {
        let mut extractor = MotorFeatureExtractor::new(MotorConfig::default());
        let frame_a = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 1000 }),
            keypoints: vec![
                Keypoint {
                    name: Some("head".to_string()),
                    x: 0.0,
                    y: 0.0,
                    confidence: Some(1.0),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 1.0,
                    y: 0.5,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("foot".to_string()),
                    x: -0.5,
                    y: -0.2,
                    confidence: Some(0.8),
                },
            ],
            bbox: None,
            metadata: HashMap::new(),
        };
        let frame_b = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 1001 }),
            keypoints: vec![
                Keypoint {
                    name: Some("head".to_string()),
                    x: 0.2,
                    y: 0.1,
                    confidence: Some(1.0),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 1.3,
                    y: 0.7,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("foot".to_string()),
                    x: -0.3,
                    y: -0.1,
                    confidence: Some(0.8),
                },
            ],
            bbox: None,
            metadata: HashMap::new(),
        };
        let _ = extractor.extract(frame_a);
        let output = extractor.extract(frame_b).expect("output");
        assert!(output.raw_signal > 0.0);
        assert!(output.signal.is_finite());
        assert!(output.features.motion_energy > 0.0);
        assert!(output.baseline_samples >= 2);
    }
}
