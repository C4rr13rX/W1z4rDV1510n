use crate::network::compute_payload_hash;
use crate::streaming::motor::PoseFrame;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SpatialEstimate {
    pub position: Option<[f64; 3]>,
    pub confidence: f64,
    pub dimensionality: usize,
    pub frame: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct SpatialConfig {
    pub depth_from_bbox_scale: f64,
    pub depth_from_bbox_bias: f64,
    pub latent_scale: f64,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            depth_from_bbox_scale: 1.25,
            depth_from_bbox_bias: 0.0,
            latent_scale: 10.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialEstimator {
    config: SpatialConfig,
}

impl SpatialEstimator {
    pub fn new(config: SpatialConfig) -> Self {
        Self { config }
    }

    pub fn estimate(
        &self,
        pose: Option<&PoseFrame>,
        metadata: &HashMap<String, Value>,
    ) -> SpatialEstimate {
        let frame = frame_from_metadata(metadata);
        let mut dimensionality = dimensionality_from_metadata(metadata);
        if let Some(mut position) = position_from_metadata(metadata) {
            if position.len() == 2 {
                position.push(0.0);
                dimensionality = dimensionality.max(2);
            } else {
                dimensionality = dimensionality.max(3);
            }
            return SpatialEstimate {
                position: Some([position[0], position[1], position[2]]),
                confidence: 0.9,
                dimensionality,
                frame,
                source: "metadata".to_string(),
            };
        }

        if let Some(pose) = pose {
            if let Some(pos) = position_from_keypoints(pose) {
                dimensionality = dimensionality.max(if pos[2].abs() > 1e-6 { 3 } else { 2 });
                return SpatialEstimate {
                    position: Some(pos),
                    confidence: keypoint_confidence(pose).max(0.2),
                    dimensionality,
                    frame,
                    source: "keypoints".to_string(),
                };
            }
            if let Some(pos) = position_from_bbox(pose, metadata, &self.config) {
                dimensionality = dimensionality.max(3);
                return SpatialEstimate {
                    position: Some(pos),
                    confidence: 0.4,
                    dimensionality,
                    frame,
                    source: "bbox_depth".to_string(),
                };
            }
        }

        if let Some(pos) = latent_position_from_metadata(metadata, &self.config) {
            dimensionality = dimensionality.max(3);
            return SpatialEstimate {
                position: Some(pos),
                confidence: 0.15,
                dimensionality,
                frame: "latent".to_string(),
                source: "context_hash".to_string(),
            };
        }

        SpatialEstimate {
            position: None,
            confidence: 0.0,
            dimensionality,
            frame,
            source: "none".to_string(),
        }
    }
}

impl Default for SpatialEstimator {
    fn default() -> Self {
        Self::new(SpatialConfig::default())
    }
}

pub fn insert_spatial_attrs(attrs: &mut HashMap<String, Value>, estimate: &SpatialEstimate) {
    attrs.insert(
        "space_frame".to_string(),
        Value::String(estimate.frame.clone()),
    );
    attrs.insert(
        "space_dimensionality".to_string(),
        Value::from(estimate.dimensionality as u64),
    );
    attrs.insert(
        "space_confidence".to_string(),
        Value::from(estimate.confidence),
    );
    attrs.insert(
        "space_source".to_string(),
        Value::String(estimate.source.clone()),
    );
    if let Some(pos) = estimate.position {
        attrs.insert("pos_x".to_string(), Value::from(pos[0]));
        attrs.insert("pos_y".to_string(), Value::from(pos[1]));
        attrs.insert("pos_z".to_string(), Value::from(pos[2]));
    }
}

fn frame_from_metadata(metadata: &HashMap<String, Value>) -> String {
    for key in ["space_frame", "frame", "space_id"] {
        if let Some(val) = metadata.get(key).and_then(|v| v.as_str()) {
            if !val.trim().is_empty() {
                return val.to_string();
            }
        }
    }
    "image".to_string()
}

fn dimensionality_from_metadata(metadata: &HashMap<String, Value>) -> usize {
    for key in ["dimensionality", "latent_dim", "dimensions"] {
        if let Some(val) = metadata.get(key) {
            if let Some(dim) = val.as_u64() {
                return dim.max(1) as usize;
            }
            if let Some(arr) = val.as_array() {
                return arr.len().max(1);
            }
        }
    }
    3
}

fn position_from_metadata(metadata: &HashMap<String, Value>) -> Option<Vec<f64>> {
    for key in ["position", "world_position", "position_xyz", "pos"] {
        if let Some(val) = metadata.get(key) {
            if let Some(arr) = val.as_array() {
                let mut coords = Vec::new();
                for entry in arr.iter().take(3) {
                    if let Some(num) = entry.as_f64() {
                        coords.push(num);
                    }
                }
                if coords.len() >= 2 {
                    return Some(coords);
                }
            }
            if let Some(obj) = val.as_object() {
                let x = obj.get("x").and_then(|v| v.as_f64());
                let y = obj.get("y").and_then(|v| v.as_f64());
                let z = obj.get("z").and_then(|v| v.as_f64());
                if let (Some(x), Some(y)) = (x, y) {
                    return Some(vec![x, y, z.unwrap_or(0.0)]);
                }
            }
        }
    }
    None
}

fn position_from_keypoints(pose: &PoseFrame) -> Option<[f64; 3]> {
    if pose.keypoints.is_empty() {
        return None;
    }
    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut z_sum = 0.0;
    let mut count = 0.0;
    let mut z_count = 0.0;
    for kp in &pose.keypoints {
        if !kp.x.is_finite() || !kp.y.is_finite() {
            continue;
        }
        x_sum += kp.x;
        y_sum += kp.y;
        count += 1.0;
        if let Some(z) = kp.z {
            if z.is_finite() {
                z_sum += z;
                z_count += 1.0;
            }
        }
    }
    if count <= 0.0 {
        return None;
    }
    let z = if z_count > 0.0 { z_sum / z_count } else { 0.0 };
    Some([x_sum / count, y_sum / count, z])
}

fn position_from_bbox(
    pose: &PoseFrame,
    metadata: &HashMap<String, Value>,
    config: &SpatialConfig,
) -> Option<[f64; 3]> {
    let bbox = pose.bbox.as_ref()?;
    let area = bbox.width.abs() * bbox.height.abs();
    if area <= 1e-6 {
        return None;
    }
    let center_x = bbox.x + bbox.width * 0.5;
    let center_y = bbox.y + bbox.height * 0.5;
    let scale = metadata
        .get("depth_scale")
        .and_then(|v| v.as_f64())
        .unwrap_or(config.depth_from_bbox_scale);
    let bias = metadata
        .get("depth_bias")
        .and_then(|v| v.as_f64())
        .unwrap_or(config.depth_from_bbox_bias);
    let depth = scale / area.sqrt().max(1e-6) + bias;
    Some([center_x, center_y, depth])
}

fn keypoint_confidence(pose: &PoseFrame) -> f64 {
    if pose.keypoints.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for kp in &pose.keypoints {
        if let Some(conf) = kp.confidence {
            sum += conf.clamp(0.0, 1.0);
            count += 1.0;
        }
    }
    if count > 0.0 {
        (sum / count).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn latent_position_from_metadata(
    metadata: &HashMap<String, Value>,
    config: &SpatialConfig,
) -> Option<[f64; 3]> {
    let mut parts = Vec::new();
    for key in ["context_scope", "anatomy_site", "context_id", "sample_id"] {
        if let Some(val) = metadata.get(key).and_then(|v| v.as_str()) {
            if !val.trim().is_empty() {
                parts.push(val.trim().to_string());
            }
        }
    }
    if parts.is_empty() {
        return None;
    }
    let seed = parts.join("|");
    let coords = hash_to_unit_floats(&seed, 3);
    Some([
        coords[0] * config.latent_scale,
        coords[1] * config.latent_scale,
        coords[2] * config.latent_scale,
    ])
}

fn hash_to_unit_floats(seed: &str, count: usize) -> Vec<f64> {
    let hash = compute_payload_hash(seed.as_bytes());
    let bytes = hex_to_bytes(&hash);
    let mut values = Vec::with_capacity(count);
    for idx in 0..count {
        let b = bytes.get(idx % bytes.len()).copied().unwrap_or(0);
        values.push(b as f64 / 255.0);
    }
    values
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.as_bytes().iter().copied();
    while let Some(high) = iter.next() {
        let low = match iter.next() {
            Some(val) => val,
            None => break,
        };
        let high_val = hex_value(high);
        let low_val = hex_value(low);
        out.push((high_val << 4) | low_val);
    }
    out
}

fn hex_value(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::motor::{Keypoint, PoseFrame};

    #[test]
    fn estimates_position_from_keypoints_with_depth() {
        let pose = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 1 }),
            keypoints: vec![
                Keypoint {
                    name: Some("head".to_string()),
                    x: 1.0,
                    y: 2.0,
                    z: Some(3.0),
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 2.0,
                    y: 3.0,
                    z: Some(5.0),
                    confidence: Some(0.8),
                },
            ],
            bbox: None,
            metadata: HashMap::new(),
        };
        let estimator = SpatialEstimator::default();
        let estimate = estimator.estimate(Some(&pose), &HashMap::new());
        let pos = estimate.position.expect("position");
        assert!((pos[0] - 1.5).abs() < 1e-6);
        assert!((pos[1] - 2.5).abs() < 1e-6);
        assert!((pos[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn estimates_position_from_bbox_depth() {
        let pose = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 1 }),
            keypoints: Vec::new(),
            bbox: Some(crate::streaming::motor::BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 2.0,
                height: 2.0,
            }),
            metadata: HashMap::new(),
        };
        let estimator = SpatialEstimator::default();
        let estimate = estimator.estimate(Some(&pose), &HashMap::new());
        let pos = estimate.position.expect("position");
        assert!((pos[0] - 1.0).abs() < 1e-6);
        assert!((pos[1] - 1.0).abs() < 1e-6);
        assert!(pos[2] > 0.0);
    }
}
