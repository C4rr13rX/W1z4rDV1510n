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

#[derive(Debug, Clone)]
struct Ray {
    origin: [f64; 3],
    direction: [f64; 3],
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

        if let Some((pos, confidence)) = triangulate_position(metadata) {
            dimensionality = dimensionality.max(3);
            return SpatialEstimate {
                position: Some(pos),
                confidence,
                dimensionality,
                frame,
                source: "triangulation".to_string(),
            };
        }

        let stereo_depth = stereo_depth_from_metadata(metadata);
        if let Some(pose) = pose {
            if let Some(mut pos) = position_from_keypoints(pose) {
                if pos[2].abs() <= 1e-6 {
                    if let Some(depth) = stereo_depth {
                        pos[2] = depth;
                    }
                }
                dimensionality = dimensionality.max(if pos[2].abs() > 1e-6 { 3 } else { 2 });
                return SpatialEstimate {
                    position: Some(pos),
                    confidence: keypoint_confidence(pose).max(0.2),
                    dimensionality,
                    frame,
                    source: "keypoints".to_string(),
                };
            }
            if let Some(depth) = stereo_depth {
                if let Some(pos) = position_from_bbox_with_depth(pose, depth) {
                    dimensionality = dimensionality.max(3);
                    return SpatialEstimate {
                        position: Some(pos),
                        confidence: 0.55,
                        dimensionality,
                        frame,
                        source: "stereo_depth".to_string(),
                    };
                }
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

fn position_from_bbox_with_depth(pose: &PoseFrame, depth: f64) -> Option<[f64; 3]> {
    let bbox = pose.bbox.as_ref()?;
    let center_x = bbox.x + bbox.width * 0.5;
    let center_y = bbox.y + bbox.height * 0.5;
    if !center_x.is_finite() || !center_y.is_finite() || !depth.is_finite() {
        return None;
    }
    Some([center_x, center_y, depth])
}

fn stereo_depth_from_metadata(metadata: &HashMap<String, Value>) -> Option<f64> {
    for key in ["stereo_depth", "depth_m", "depth"] {
        if let Some(depth) = metadata.get(key).and_then(|val| val.as_f64()) {
            if depth.is_finite() {
                return Some(depth);
            }
        }
    }
    let disparity = metadata
        .get("stereo_disparity")
        .or_else(|| metadata.get("disparity"))
        .or_else(|| metadata.get("disparity_px"))
        .and_then(|val| val.as_f64());
    let baseline = metadata
        .get("camera_baseline_m")
        .or_else(|| metadata.get("baseline_m"))
        .or_else(|| metadata.get("baseline"))
        .and_then(|val| val.as_f64());
    let focal = metadata
        .get("focal_length_px")
        .or_else(|| metadata.get("focal_px"))
        .or_else(|| metadata.get("focal_length"))
        .and_then(|val| val.as_f64());
    match (disparity, baseline, focal) {
        (Some(d), Some(b), Some(f)) if d.is_finite() && b.is_finite() && f.is_finite() => {
            if d.abs() <= 1e-6 {
                None
            } else {
                Some((b * f / d).abs())
            }
        }
        _ => None,
    }
}

fn triangulate_position(metadata: &HashMap<String, Value>) -> Option<([f64; 3], f64)> {
    let rays = rays_from_metadata(metadata);
    if rays.len() < 2 {
        return None;
    }
    let mut a = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for ray in &rays {
        let d = normalize_vec3(ray.direction)?;
        let m = [
            [1.0 - d[0] * d[0], -d[0] * d[1], -d[0] * d[2]],
            [-d[1] * d[0], 1.0 - d[1] * d[1], -d[1] * d[2]],
            [-d[2] * d[0], -d[2] * d[1], 1.0 - d[2] * d[2]],
        ];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += m[i][j];
            }
            b[i] += m[i][0] * ray.origin[0]
                + m[i][1] * ray.origin[1]
                + m[i][2] * ray.origin[2];
        }
    }
    let pos = solve_3x3(a, b)?;
    if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
        return None;
    }
    let mut residual = 0.0;
    for ray in &rays {
        if let Some(d) = normalize_vec3(ray.direction) {
            residual += distance_point_to_ray(pos, ray.origin, d);
        }
    }
    let avg_residual = residual / rays.len() as f64;
    let confidence = (1.0 / (1.0 + avg_residual)).clamp(0.05, 0.95);
    Some((pos, confidence))
}

fn rays_from_metadata(metadata: &HashMap<String, Value>) -> Vec<Ray> {
    for key in ["camera_rays", "rays", "views", "camera_views"] {
        if let Some(values) = metadata.get(key).and_then(|val| val.as_array()) {
            let mut rays = Vec::new();
            for value in values {
                if let Some(ray) = ray_from_value(value) {
                    rays.push(ray);
                }
            }
            if !rays.is_empty() {
                return rays;
            }
        }
    }
    Vec::new()
}

fn ray_from_value(value: &Value) -> Option<Ray> {
    let obj = value.as_object()?;
    let origin = obj
        .get("origin")
        .or_else(|| obj.get("camera_origin"))
        .or_else(|| obj.get("position"))
        .and_then(vec3_from_value)?;
    let direction = obj
        .get("direction")
        .or_else(|| obj.get("ray"))
        .or_else(|| obj.get("ray_dir"))
        .and_then(vec3_from_value)?;
    Some(Ray { origin, direction })
}

fn vec3_from_value(value: &Value) -> Option<[f64; 3]> {
    if let Some(arr) = value.as_array() {
        if arr.len() >= 3 {
            let x = arr[0].as_f64()?;
            let y = arr[1].as_f64()?;
            let z = arr[2].as_f64()?;
            if x.is_finite() && y.is_finite() && z.is_finite() {
                return Some([x, y, z]);
            }
        }
    }
    if let Some(obj) = value.as_object() {
        let x = obj.get("x").and_then(|val| val.as_f64())?;
        let y = obj.get("y").and_then(|val| val.as_f64())?;
        let z = obj.get("z").and_then(|val| val.as_f64())?;
        if x.is_finite() && y.is_finite() && z.is_finite() {
            return Some([x, y, z]);
        }
    }
    None
}

fn normalize_vec3(vec: [f64; 3]) -> Option<[f64; 3]> {
    let norm = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    if norm <= 1e-9 {
        return None;
    }
    Some([vec[0] / norm, vec[1] / norm, vec[2] / norm])
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let mut pivot = i;
        for row in (i + 1)..3 {
            if a[row][i].abs() > a[pivot][i].abs() {
                pivot = row;
            }
        }
        if a[pivot][i].abs() < 1e-9 {
            return None;
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let diag = a[i][i];
        for col in i..3 {
            a[i][col] /= diag;
        }
        b[i] /= diag;
        for row in 0..3 {
            if row == i {
                continue;
            }
            let factor = a[row][i];
            for col in i..3 {
                a[row][col] -= factor * a[i][col];
            }
            b[row] -= factor * b[i];
        }
    }
    Some([b[0], b[1], b[2]])
}

fn distance_point_to_ray(point: [f64; 3], origin: [f64; 3], direction: [f64; 3]) -> f64 {
    let vx = point[0] - origin[0];
    let vy = point[1] - origin[1];
    let vz = point[2] - origin[2];
    let cx = vy * direction[2] - vz * direction[1];
    let cy = vz * direction[0] - vx * direction[2];
    let cz = vx * direction[1] - vy * direction[0];
    (cx * cx + cy * cy + cz * cz).sqrt()
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
    use serde_json::json;

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

    #[test]
    fn estimates_position_from_stereo_metadata() {
        let pose = PoseFrame {
            entity_id: "e1".to_string(),
            timestamp: Some(Timestamp { unix: 1 }),
            keypoints: vec![
                Keypoint {
                    name: Some("head".to_string()),
                    x: 1.0,
                    y: 2.0,
                    z: None,
                    confidence: Some(0.9),
                },
                Keypoint {
                    name: Some("hand".to_string()),
                    x: 2.0,
                    y: 3.0,
                    z: None,
                    confidence: Some(0.8),
                },
            ],
            bbox: None,
            metadata: HashMap::new(),
        };
        let mut metadata = HashMap::new();
        metadata.insert("stereo_disparity".to_string(), Value::from(10.0));
        metadata.insert("camera_baseline_m".to_string(), Value::from(0.2));
        metadata.insert("focal_length_px".to_string(), Value::from(100.0));
        let estimator = SpatialEstimator::default();
        let estimate = estimator.estimate(Some(&pose), &metadata);
        let pos = estimate.position.expect("position");
        assert!((pos[2] - 2.0).abs() < 1e-6);
        assert!(estimate.dimensionality >= 3);
    }

    #[test]
    fn estimates_position_from_camera_rays() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "camera_rays".to_string(),
            json!([
                { "origin": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0] },
                { "origin": [0.0, 1.0, 0.0], "direction": [1.0, -1.0, 0.0] }
            ]),
        );
        let estimator = SpatialEstimator::default();
        let estimate = estimator.estimate(None, &metadata);
        let pos = estimate.position.expect("position");
        assert!((pos[0] - 1.0).abs() < 1e-6);
        assert!(pos[1].abs() < 1e-6);
    }
}
