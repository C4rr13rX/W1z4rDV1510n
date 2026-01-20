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
    pub z: Option<f64>,
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
    pub camera_motion: f64,
    pub stability: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct MotorFeatureOutput {
    pub entity_id: String,
    pub timestamp: Timestamp,
    pub keypoint_count: usize,
    pub bbox_area: Option<f64>,
    pub features: MotorFeatures,
    pub camera_shift_dx: f64,
    pub camera_shift_dy: f64,
    pub camera_shift_confidence: f64,
    pub camera_shift_source: String,
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
    pub camera_flow_min_keypoints: usize,
    pub camera_flow_max_std: f64,
    pub camera_motion_alpha: f64,
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
            camera_flow_min_keypoints: 4,
            camera_flow_max_std: 0.35,
            camera_motion_alpha: 0.4,
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
struct RawKeypointState {
    x: f64,
    y: f64,
    confidence: f64,
}

#[derive(Debug, Clone)]
struct MotionSample {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    prev_vx: f64,
    prev_vy: f64,
    confidence: f64,
}

#[derive(Debug, Clone)]
struct EntityMotionState {
    last_timestamp: Timestamp,
    keypoints: HashMap<String, KeypointState>,
    raw_keypoints: HashMap<String, RawKeypointState>,
    last_centroid: Option<(f64, f64)>,
    last_dispersion: f64,
    last_centroid_velocity: Option<(f64, f64)>,
    signal_ema: f64,
    baseline_mean: f64,
    baseline_var: f64,
    baseline_samples: usize,
    camera_shift_ema: (f64, f64),
    camera_confidence_ema: f64,
}

impl EntityMotionState {
    fn new(timestamp: Timestamp) -> Self {
        Self {
            last_timestamp: timestamp,
            keypoints: HashMap::new(),
            raw_keypoints: HashMap::new(),
            last_centroid: None,
            last_dispersion: 0.0,
            last_centroid_velocity: None,
            signal_ema: 0.0,
            baseline_mean: 0.0,
            baseline_var: 0.0,
            baseline_samples: 0,
            camera_shift_ema: (0.0, 0.0),
            camera_confidence_ema: 0.0,
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

#[derive(Debug, Clone)]
struct CameraShift {
    dx: f64,
    dy: f64,
    confidence: f64,
    source: &'static str,
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
        let state = self
            .states
            .entry(frame.entity_id.clone())
            .or_insert_with(|| EntityMotionState::new(timestamp));

        let dt = (timestamp.unix - state.last_timestamp.unix).abs() as f64;
        let dt = if dt > 0.0 { dt } else { 1.0 };
        let raw_keypoints = raw_keypoints_from_frame(&frame);
        let shift = estimate_camera_shift(&frame, &raw_keypoints, state, &self.config);
        let (shift_dx, shift_dy, shift_confidence, shift_source) =
            apply_camera_shift(shift, state, &self.config);
        let (keypoints, bbox_area, dispersion_hint) =
            normalize_keypoints(&frame, Some((shift_dx, shift_dy)));
        if keypoints.is_empty() {
            return None;
        }
        let keypoint_count = keypoints.len();

        let mut conf_sum = 0.0;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        let mut centroid_weight = 0.0;
        let mut centroid_vx = 0.0;
        let mut centroid_vy = 0.0;

        let mut next_keypoints = HashMap::new();
        let mut samples: Vec<MotionSample> = Vec::with_capacity(keypoint_count);
        for (label, kp) in keypoints {
            let conf = kp.confidence.clamp(0.0, 1.0);
            let prev = state.keypoints.get(&label);
            let (vx, vy, prev_vx, prev_vy) = if let Some(prev_state) = prev {
                let dx = kp.x - prev_state.x;
                let dy = kp.y - prev_state.y;
                let vx = dx / dt;
                let vy = dy / dt;
                (vx, vy, prev_state.vx, prev_state.vy)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            conf_sum += conf;
            centroid_x += kp.x * conf;
            centroid_y += kp.y * conf;
            centroid_weight += conf;
            centroid_vx += vx * conf;
            centroid_vy += vy * conf;
            next_keypoints.insert(
                label.clone(),
                KeypointState {
                    x: kp.x,
                    y: kp.y,
                    vx,
                    vy,
                    confidence: conf,
                },
            );
            samples.push(MotionSample {
                x: kp.x,
                y: kp.y,
                vx,
                vy,
                prev_vx,
                prev_vy,
                confidence: conf,
            });
        }

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
        let (centroid_vx, centroid_vy) = if conf_sum > 0.0 {
            (centroid_vx / conf_sum, centroid_vy / conf_sum)
        } else {
            (0.0, 0.0)
        };
        let camera_motion = if shift_confidence > 0.0 {
            (shift_dx * shift_dx + shift_dy * shift_dy).sqrt() / dt
        } else {
            (centroid_vx * centroid_vx + centroid_vy * centroid_vy).sqrt()
        };
        let (prev_centroid_vx, prev_centroid_vy) =
            state.last_centroid_velocity.unwrap_or((0.0, 0.0));

        let mut rel_speed_sum = 0.0;
        let mut rel_speed_sq_sum = 0.0;
        let mut rel_jitter_sum = 0.0;
        for sample in &samples {
            let rel_vx = sample.vx - centroid_vx;
            let rel_vy = sample.vy - centroid_vy;
            let prev_rel_vx = sample.prev_vx - prev_centroid_vx;
            let prev_rel_vy = sample.prev_vy - prev_centroid_vy;
            let rel_speed = (rel_vx * rel_vx + rel_vy * rel_vy).sqrt();
            rel_speed_sum += rel_speed * sample.confidence;
            rel_speed_sq_sum += rel_speed * rel_speed;
            let rel_dvx = rel_vx - prev_rel_vx;
            let rel_dvy = rel_vy - prev_rel_vy;
            let rel_accel = (rel_dvx * rel_dvx + rel_dvy * rel_dvy).sqrt() / dt;
            rel_jitter_sum += rel_accel * sample.confidence;
        }

        let mean_speed = if conf_sum > 0.0 {
            rel_speed_sum / conf_sum
        } else {
            0.0
        };
        let variance = if keypoint_count > 0 {
            let mean_unweighted = rel_speed_sq_sum / keypoint_count as f64;
            (mean_unweighted - mean_speed * mean_speed).max(0.0)
        } else {
            0.0
        };
        let micro_jitter = if conf_sum > 0.0 {
            rel_jitter_sum / conf_sum
        } else {
            0.0
        };

        let mut dispersion = dispersion_hint.unwrap_or_else(|| {
            let mut sum = 0.0;
            for sample in &samples {
                let dx = sample.x - centroid_x;
                let dy = sample.y - centroid_y;
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
        let camera_dominance = if camera_motion + mean_speed > 1e-6 {
            (camera_motion / (camera_motion + mean_speed)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let posture_shift = dispersion_shift + centroid_shift * (1.0 - camera_dominance);

        let motion_norm = (mean_speed / self.config.velocity_norm).clamp(0.0, 1.0);
        let jitter_norm = (micro_jitter / self.config.jitter_norm).clamp(0.0, 1.0);
        let posture_norm = (posture_shift / self.config.posture_norm).clamp(0.0, 1.0);
        let stability = (1.0 - jitter_norm).clamp(0.0, 1.0)
            * (1.0 - 0.8 * camera_dominance).clamp(0.0, 1.0);
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
        state.last_centroid_velocity = Some((centroid_vx, centroid_vy));
        state.signal_ema = signal;
        state.raw_keypoints = raw_keypoints;

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
                camera_motion,
                stability,
                confidence,
            },
            camera_shift_dx: shift_dx,
            camera_shift_dy: shift_dy,
            camera_shift_confidence: shift_confidence,
            camera_shift_source: shift_source,
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
    camera_shift: Option<(f64, f64)>,
) -> (Vec<(String, KeypointState)>, Option<f64>, Option<f64>) {
    let (shift_x, shift_y) = camera_shift.unwrap_or((0.0, 0.0));
    if !frame.keypoints.is_empty() {
        let mut normalized = Vec::with_capacity(frame.keypoints.len());
        for (idx, kp) in frame.keypoints.iter().enumerate() {
            let label = keypoint_label(kp.name.as_deref(), idx);
            normalized.push((
                label,
                KeypointState {
                    x: kp.x - shift_x,
                    y: kp.y - shift_y,
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
    let center_x = bbox.x + bbox.width * 0.5 - shift_x;
    let center_y = bbox.y + bbox.height * 0.5 - shift_y;
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

fn keypoint_label(name: Option<&str>, idx: usize) -> String {
    name.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("k{idx}"))
}

fn raw_keypoints_from_frame(frame: &PoseFrame) -> HashMap<String, RawKeypointState> {
    let (raw, _, _) = normalize_keypoints(frame, None);
    let mut map = HashMap::new();
    for (label, kp) in raw {
        map.insert(
            label,
            RawKeypointState {
                x: kp.x,
                y: kp.y,
                confidence: kp.confidence,
            },
        );
    }
    map
}

fn estimate_camera_shift(
    frame: &PoseFrame,
    raw_keypoints: &HashMap<String, RawKeypointState>,
    state: &EntityMotionState,
    config: &MotorConfig,
) -> CameraShift {
    if let Some(shift) = camera_shift_from_metadata(&frame.metadata) {
        return shift;
    }
    if let Some(shift) = camera_shift_from_keypoints(raw_keypoints, &state.raw_keypoints, config)
    {
        return shift;
    }
    CameraShift {
        dx: 0.0,
        dy: 0.0,
        confidence: 0.0,
        source: "none",
    }
}

fn apply_camera_shift(
    shift: CameraShift,
    state: &mut EntityMotionState,
    config: &MotorConfig,
) -> (f64, f64, f64, String) {
    let alpha = config.camera_motion_alpha.clamp(0.0, 1.0);
    if shift.confidence > 0.0 {
        state.camera_shift_ema.0 =
            alpha * shift.dx + (1.0 - alpha) * state.camera_shift_ema.0;
        state.camera_shift_ema.1 =
            alpha * shift.dy + (1.0 - alpha) * state.camera_shift_ema.1;
        state.camera_confidence_ema =
            alpha * shift.confidence + (1.0 - alpha) * state.camera_confidence_ema;
        return (
            state.camera_shift_ema.0,
            state.camera_shift_ema.1,
            state.camera_confidence_ema,
            shift.source.to_string(),
        );
    }
    let decay = 1.0 - alpha;
    state.camera_shift_ema.0 *= decay;
    state.camera_shift_ema.1 *= decay;
    state.camera_confidence_ema *= decay;
    let source = if state.camera_confidence_ema > 0.0 {
        "ema"
    } else {
        "none"
    };
    (
        state.camera_shift_ema.0,
        state.camera_shift_ema.1,
        state.camera_confidence_ema,
        source.to_string(),
    )
}

fn camera_shift_from_metadata(metadata: &HashMap<String, Value>) -> Option<CameraShift> {
    let confidence = shift_confidence_from_metadata(metadata);
    if let Some((dx, dy)) =
        vector_from_metadata(metadata, &["camera_shift", "imu_shift", "optical_flow", "flow"])
    {
        return Some(CameraShift {
            dx,
            dy,
            confidence,
            source: "metadata",
        });
    }
    let pairs = [
        ("camera_dx", "camera_dy", "camera"),
        ("imu_dx", "imu_dy", "imu"),
        ("shift_dx", "shift_dy", "shift"),
        ("flow_dx", "flow_dy", "flow"),
    ];
    for (dx_key, dy_key, source) in pairs {
        let dx = metadata.get(dx_key).and_then(|val| val.as_f64());
        let dy = metadata.get(dy_key).and_then(|val| val.as_f64());
        if let (Some(dx), Some(dy)) = (dx, dy) {
            if dx.is_finite() && dy.is_finite() {
                return Some(CameraShift {
                    dx,
                    dy,
                    confidence,
                    source,
                });
            }
        }
    }
    None
}

fn camera_shift_from_keypoints(
    current: &HashMap<String, RawKeypointState>,
    previous: &HashMap<String, RawKeypointState>,
    config: &MotorConfig,
) -> Option<CameraShift> {
    if current.is_empty() || previous.is_empty() {
        return None;
    }
    let mut samples = Vec::new();
    let mut sum_w = 0.0;
    let mut sum_dx = 0.0;
    let mut sum_dy = 0.0;
    for (label, curr) in current {
        let Some(prev) = previous.get(label) else {
            continue;
        };
        let dx = curr.x - prev.x;
        let dy = curr.y - prev.y;
        if !dx.is_finite() || !dy.is_finite() {
            continue;
        }
        let weight = ((curr.confidence + prev.confidence) * 0.5).clamp(0.05, 1.0);
        samples.push((dx, dy, weight));
        sum_w += weight;
        sum_dx += dx * weight;
        sum_dy += dy * weight;
    }
    let sample_len = samples.len();
    if sample_len < config.camera_flow_min_keypoints {
        return None;
    }
    if sum_w <= 0.0 {
        return None;
    }
    let mean_dx = sum_dx / sum_w;
    let mean_dy = sum_dy / sum_w;
    let mean_mag = (mean_dx * mean_dx + mean_dy * mean_dy).sqrt();
    if mean_mag < 1e-6 {
        return None;
    }
    let mut var = 0.0;
    for (dx, dy, weight) in &samples {
        let ddx = *dx - mean_dx;
        let ddy = *dy - mean_dy;
        var += *weight * (ddx * ddx + ddy * ddy);
    }
    let std = (var / sum_w).sqrt();
    let max_std = config.camera_flow_max_std.max(1e-6);
    let std_norm = (std / mean_mag.max(1e-6)).min(10.0);
    if std_norm > max_std {
        return None;
    }
    let consistency = (1.0 - (std_norm / max_std).clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let count_factor = sample_len as f64 / (sample_len as f64 + 2.0);
    let confidence = (consistency * count_factor).clamp(0.0, 1.0);
    Some(CameraShift {
        dx: mean_dx,
        dy: mean_dy,
        confidence,
        source: "keypoints",
    })
}

fn shift_confidence_from_metadata(metadata: &HashMap<String, Value>) -> f64 {
    for key in [
        "camera_shift_confidence",
        "shift_confidence",
        "imu_confidence",
        "flow_confidence",
    ] {
        if let Some(val) = metadata.get(key).and_then(|val| val.as_f64()) {
            return val.clamp(0.0, 1.0);
        }
    }
    1.0
}

fn vector_from_metadata(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<(f64, f64)> {
    for key in keys {
        if let Some(value) = metadata.get(*key) {
            if let Some(vec) = vector_from_value(value) {
                return Some(vec);
            }
        }
    }
    None
}

fn vector_from_value(value: &Value) -> Option<(f64, f64)> {
    if let Some(arr) = value.as_array() {
        if arr.len() >= 2 {
            if let (Some(x), Some(y)) = (arr[0].as_f64(), arr[1].as_f64()) {
                if x.is_finite() && y.is_finite() {
                    return Some((x, y));
                }
            }
        }
    }
    if let Some(obj) = value.as_object() {
        let x = obj
            .get("x")
            .or_else(|| obj.get("dx"))
            .and_then(|v| v.as_f64());
        let y = obj
            .get("y")
            .or_else(|| obj.get("dy"))
            .and_then(|v| v.as_f64());
        if let (Some(x), Some(y)) = (x, y) {
            if x.is_finite() && y.is_finite() {
                return Some((x, y));
            }
        }
    }
    None
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
                    z: None,
                    confidence: Some(1.0),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 1.0,
                    y: 0.5,
                    z: None,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("foot".to_string()),
                    x: -0.5,
                    y: -0.2,
                    z: None,
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
                    z: None,
                    confidence: Some(1.0),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 1.3,
                    y: 0.7,
                    z: None,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("foot".to_string()),
                    x: -0.3,
                    y: -0.1,
                    z: None,
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

    #[test]
    fn motor_extractor_reduces_camera_shake_signal() {
        let mut extractor = MotorFeatureExtractor::new(MotorConfig::default());
        let frame_a = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 2000 }),
            keypoints: vec![
                Keypoint {
                    name: Some("head".to_string()),
                    x: 0.0,
                    y: 0.0,
                    z: None,
                    confidence: Some(1.0),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 1.0,
                    y: 0.5,
                    z: None,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("foot".to_string()),
                    x: -0.5,
                    y: -0.2,
                    z: None,
                    confidence: Some(0.8),
                },
            ],
            bbox: None,
            metadata: HashMap::new(),
        };
        let frame_b = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 2001 }),
            keypoints: vec![
                Keypoint {
                    name: Some("head".to_string()),
                    x: 2.0,
                    y: 0.0,
                    z: None,
                    confidence: Some(1.0),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 3.0,
                    y: 0.5,
                    z: None,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("foot".to_string()),
                    x: 1.5,
                    y: -0.2,
                    z: None,
                    confidence: Some(0.8),
                },
            ],
            bbox: None,
            metadata: HashMap::new(),
        };
        let _ = extractor.extract(frame_a);
        let output = extractor.extract(frame_b).expect("output");
        assert!(output.features.camera_motion > 1.0);
        assert!(output.features.motion_energy < 0.1);
        assert!(output.raw_signal < 0.1);
        assert!(output.features.stability < 0.4);
    }
}
