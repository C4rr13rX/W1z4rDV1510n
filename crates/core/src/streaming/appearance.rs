use crate::config::AppearanceConfig;
use crate::network::compute_payload_hash;
use crate::streaming::motor::{BoundingBox, Keypoint, PoseFrame};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppearanceFeatures {
    pub vector: Vec<f64>,
    pub signature: String,
    pub confidence: f64,
    #[serde(default)]
    pub components: HashMap<String, f64>,
}

pub struct AppearanceExtractor {
    config: AppearanceConfig,
}

impl AppearanceExtractor {
    pub fn new(config: AppearanceConfig) -> Self {
        Self { config }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn extract(&self, frame: &PoseFrame) -> Option<AppearanceFeatures> {
        if !self.config.enabled {
            return None;
        }
        let bbox = frame.bbox.as_ref();
        let keypoints = &frame.keypoints;
        if keypoints.len() < self.config.min_keypoints && bbox.is_none() {
            return None;
        }

        let mut components = HashMap::new();
        if let Some(bbox) = bbox {
            let width = bbox.width.abs().max(1e-6);
            let height = bbox.height.abs().max(1e-6);
            let aspect = width / height;
            let area = width * height;
            let area_norm = (area.sqrt() / 1000.0).clamp(0.0, 10.0);
            components.insert("bbox_aspect".to_string(), aspect.clamp(0.1, 10.0));
            components.insert("bbox_area_norm".to_string(), area_norm);
        }

        if !keypoints.is_empty() {
            let (spread_x, spread_y) = keypoint_spread(keypoints, bbox);
            components.insert("kp_spread_x".to_string(), spread_x);
            components.insert("kp_spread_y".to_string(), spread_y);
            let named = named_keypoints(keypoints);
            if let Some(dist) = distance_between(&named, "left_shoulder", "right_shoulder") {
                components.insert("shoulder_width".to_string(), normalize_by_bbox(dist, bbox));
            }
            if let Some(dist) = distance_between(&named, "left_hip", "right_hip") {
                components.insert("hip_width".to_string(), normalize_by_bbox(dist, bbox));
            }
            if let Some(dist) = torso_length(&named) {
                components.insert("torso_length".to_string(), normalize_by_bbox(dist, bbox));
            }
            if let Some(dist) = limb_length(&named, "shoulder", "wrist") {
                components.insert("arm_length".to_string(), normalize_by_bbox(dist, bbox));
            }
            if let Some(dist) = limb_length(&named, "hip", "ankle") {
                components.insert("leg_length".to_string(), normalize_by_bbox(dist, bbox));
            }
        }

        if components.is_empty() {
            return None;
        }

        let mut keys = components.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        let mut vector = Vec::new();
        for key in keys {
            if let Some(value) = components.get(&key) {
                vector.push(*value);
            }
        }
        vector.truncate(self.config.max_features.max(1));
        let confidence = appearance_confidence(&components, keypoints.len(), bbox.is_some())
            .max(self.config.confidence_floor)
            .clamp(0.0, 1.0);
        let signature = signature_from_vector(&vector, self.config.signature_precision);
        Some(AppearanceFeatures {
            vector,
            signature,
            confidence,
            components,
        })
    }
}

fn signature_from_vector(vector: &[f64], precision: f64) -> String {
    let precision = precision.max(1e-6);
    let mut parts = Vec::new();
    for value in vector {
        let bucket = (value / precision).round() * precision;
        parts.push(format!("{bucket:.4}"));
    }
    compute_payload_hash(format!("phenotype|{}", parts.join("|")).as_bytes())
}

fn appearance_confidence(
    components: &HashMap<String, f64>,
    keypoints: usize,
    has_bbox: bool,
) -> f64 {
    let mut score = components.len() as f64 / 8.0;
    if keypoints > 0 {
        score += (keypoints as f64 / 10.0).min(1.0) * 0.3;
    }
    if has_bbox {
        score += 0.2;
    }
    score.clamp(0.0, 1.0)
}

fn keypoint_spread(keypoints: &[Keypoint], bbox: Option<&BoundingBox>) -> (f64, f64) {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;
    for kp in keypoints {
        if kp.x.is_finite() && kp.y.is_finite() {
            sum_x += kp.x;
            sum_y += kp.y;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return (0.0, 0.0);
    }
    let mean_x = sum_x / count;
    let mean_y = sum_y / count;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for kp in keypoints {
        if kp.x.is_finite() && kp.y.is_finite() {
            var_x += (kp.x - mean_x).powi(2);
            var_y += (kp.y - mean_y).powi(2);
        }
    }
    let std_x = (var_x / count).sqrt();
    let std_y = (var_y / count).sqrt();
    let (norm_x, norm_y) = if let Some(bbox) = bbox {
        let w = bbox.width.abs().max(1e-6);
        let h = bbox.height.abs().max(1e-6);
        (std_x / w, std_y / h)
    } else {
        (std_x, std_y)
    };
    (norm_x.clamp(0.0, 2.0), norm_y.clamp(0.0, 2.0))
}

fn named_keypoints(keypoints: &[Keypoint]) -> HashMap<String, (f64, f64)> {
    let mut out = HashMap::new();
    for kp in keypoints {
        let Some(name) = kp.name.as_deref() else { continue };
        let canonical = canonical_name(name);
        if canonical.is_empty() {
            continue;
        }
        out.insert(canonical, (kp.x, kp.y));
    }
    out
}

fn canonical_name(raw: &str) -> String {
    let mut name = raw.to_ascii_lowercase();
    name = name.replace('-', "_");
    let side = if name.contains("left") || name.starts_with("l_") {
        "left"
    } else if name.contains("right") || name.starts_with("r_") {
        "right"
    } else {
        ""
    };
    let part = if name.contains("shoulder") {
        "shoulder"
    } else if name.contains("hip") {
        "hip"
    } else if name.contains("wrist") {
        "wrist"
    } else if name.contains("ankle") {
        "ankle"
    } else if name.contains("elbow") {
        "elbow"
    } else if name.contains("knee") {
        "knee"
    } else {
        ""
    };
    if side.is_empty() || part.is_empty() {
        return String::new();
    }
    format!("{side}_{part}")
}

fn distance_between(points: &HashMap<String, (f64, f64)>, a: &str, b: &str) -> Option<f64> {
    let pa = points.get(a)?;
    let pb = points.get(b)?;
    Some(((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt())
}

fn torso_length(points: &HashMap<String, (f64, f64)>) -> Option<f64> {
    let (ls, rs) = (points.get("left_shoulder")?, points.get("right_shoulder")?);
    let (lh, rh) = (points.get("left_hip")?, points.get("right_hip")?);
    let mid_shoulder = ((ls.0 + rs.0) * 0.5, (ls.1 + rs.1) * 0.5);
    let mid_hip = ((lh.0 + rh.0) * 0.5, (lh.1 + rh.1) * 0.5);
    Some(((mid_shoulder.0 - mid_hip.0).powi(2) + (mid_shoulder.1 - mid_hip.1).powi(2)).sqrt())
}

fn limb_length(points: &HashMap<String, (f64, f64)>, proximal: &str, distal: &str) -> Option<f64> {
    let left = distance_between(points, &format!("left_{proximal}"), &format!("left_{distal}"));
    let right = distance_between(points, &format!("right_{proximal}"), &format!("right_{distal}"));
    match (left, right) {
        (Some(a), Some(b)) => Some((a + b) * 0.5),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        _ => None,
    }
}

fn normalize_by_bbox(value: f64, bbox: Option<&BoundingBox>) -> f64 {
    if let Some(bbox) = bbox {
        let scale = bbox.height.abs().max(1e-6);
        (value / scale).clamp(0.0, 3.0)
    } else {
        value.clamp(0.0, 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;

    #[test]
    fn appearance_extracts_signature() {
        let config = AppearanceConfig::default();
        let extractor = AppearanceExtractor::new(config);
        let frame = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 1 }),
            keypoints: vec![
                Keypoint { name: Some("left_shoulder".to_string()), x: 0.0, y: 0.0, z: None, confidence: None },
                Keypoint { name: Some("right_shoulder".to_string()), x: 1.0, y: 0.0, z: None, confidence: None },
                Keypoint { name: Some("left_hip".to_string()), x: 0.0, y: 1.0, z: None, confidence: None },
                Keypoint { name: Some("right_hip".to_string()), x: 1.0, y: 1.0, z: None, confidence: None },
            ],
            bbox: Some(BoundingBox { x: 0.0, y: 0.0, width: 1.0, height: 2.0 }),
            metadata: HashMap::new(),
        };
        let features = extractor.extract(&frame).expect("features");
        assert!(!features.signature.is_empty());
        assert!(!features.vector.is_empty());
        assert!(features.confidence > 0.0);
    }
}
