use crate::schema::Timestamp;
use crate::streaming::dimensions::DimensionReport;
use crate::streaming::physiology_runtime::PhysiologyReport;
use crate::streaming::schema::TokenBatch;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

const PHYSIOLOGY_DIMENSIONS: [&str; 10] = [
    "ultradian_micro_amp",
    "ultradian_micro_coh",
    "ultradian_brac_amp",
    "ultradian_brac_coh",
    "ultradian_meso_amp",
    "ultradian_meso_coh",
    "motor_signal",
    "motion_energy",
    "posture_shift",
    "micro_jitter",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDimensionScore {
    pub name: String,
    pub score: f64,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthEntityOverlay {
    pub entity_id: String,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    pub score: f64,
    pub color: String,
    pub label: String,
    #[serde(default)]
    pub dimensions: Vec<HealthDimensionScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDimensionPalette {
    pub name: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthOverlayReport {
    pub timestamp: Timestamp,
    pub entities: Vec<HealthEntityOverlay>,
    pub palette: Vec<HealthDimensionPalette>,
}

#[derive(Debug, Clone)]
pub struct HealthOverlayConfig {
    pub enabled: bool,
    pub score_ema_alpha: f64,
    pub deviation_scale: f64,
    pub max_palette: usize,
}

impl Default for HealthOverlayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            score_ema_alpha: 0.3,
            deviation_scale: 3.0,
            max_palette: 32,
        }
    }
}

pub struct HealthOverlayRuntime {
    config: HealthOverlayConfig,
    baseline: HashMap<String, f64>,
}

impl HealthOverlayRuntime {
    pub fn new(config: HealthOverlayConfig) -> Self {
        Self {
            config,
            baseline: HashMap::new(),
        }
    }

    pub fn update(
        &mut self,
        batch: &TokenBatch,
        physiology: Option<&PhysiologyReport>,
        dimensions: Option<&DimensionReport>,
    ) -> Option<HealthOverlayReport> {
        if !self.config.enabled {
            return None;
        }
        let physiology = physiology?;
        let positions = extract_positions(batch);
        let mut entities = Vec::new();
        for deviation in &physiology.deviations {
            let raw_score = health_score_from_deviation(deviation.deviation_index);
            let score = self.update_baseline(&deviation.context, raw_score);
            let color = grayscale_color(score);
            let label = health_label(score).to_string();
            let position = positions.get(&deviation.context).copied();
            let mut dims = Vec::new();
            for (idx, val) in deviation.deviation_vector.iter().enumerate() {
                if idx >= PHYSIOLOGY_DIMENSIONS.len() {
                    break;
                }
                let dim_score = dimension_score(*val, self.config.deviation_scale);
                let dim_color = dimension_color(idx, dim_score);
                dims.push(HealthDimensionScore {
                    name: PHYSIOLOGY_DIMENSIONS[idx].to_string(),
                    score: dim_score,
                    color: dim_color,
                });
            }
            entities.push(HealthEntityOverlay {
                entity_id: deviation.context.clone(),
                position,
                score,
                color,
                label,
                dimensions: dims,
            });
        }
        let palette = dimension_palette(dimensions, self.config.max_palette);
        Some(HealthOverlayReport {
            timestamp: batch.timestamp,
            entities,
            palette,
        })
    }

    fn update_baseline(&mut self, key: &str, score: f64) -> f64 {
        let alpha = self.config.score_ema_alpha.clamp(0.0, 1.0);
        let entry = self.baseline.entry(key.to_string()).or_insert(score);
        *entry = alpha * score + (1.0 - alpha) * *entry;
        entry.clamp(0.0, 1.0)
    }
}

impl Default for HealthOverlayRuntime {
    fn default() -> Self {
        Self::new(HealthOverlayConfig::default())
    }
}

fn extract_positions(batch: &TokenBatch) -> HashMap<String, [f64; 3]> {
    let mut positions = HashMap::new();
    for token in &batch.tokens {
        let Some(entity) = token.attributes.get("entity_id").and_then(|v| v.as_str()) else {
            continue;
        };
        if let Some(pos) = position_from_attrs(&token.attributes) {
            positions.insert(entity.to_string(), pos);
        }
    }
    for layer in &batch.layers {
        let Some(entity) = layer.attributes.get("entity_id").and_then(|v| v.as_str()) else {
            continue;
        };
        if positions.contains_key(entity) {
            continue;
        }
        if let Some(pos) = position_from_attrs(&layer.attributes) {
            positions.insert(entity.to_string(), pos);
        }
    }
    positions
}

fn position_from_attrs(attrs: &HashMap<String, Value>) -> Option<[f64; 3]> {
    let x = attrs.get("pos_x").and_then(|v| v.as_f64());
    let y = attrs.get("pos_y").and_then(|v| v.as_f64());
    let z = attrs.get("pos_z").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if let (Some(x), Some(y)) = (x, y) {
        return Some([x, y, z]);
    }
    if let Some(Value::Array(arr)) = attrs.get("position") {
        let mut coords = [0.0; 3];
        for (idx, val) in arr.iter().take(3).enumerate() {
            if let Some(num) = val.as_f64() {
                coords[idx] = num;
            }
        }
        return Some(coords);
    }
    None
}

fn health_score_from_deviation(index: f64) -> f64 {
    (1.0 / (1.0 + index.abs())).clamp(0.0, 1.0)
}

fn dimension_score(z: f64, scale: f64) -> f64 {
    let scale = scale.max(1e-3);
    let normalized = (z.abs() / scale).clamp(0.0, 1.0);
    (1.0 - normalized).clamp(0.0, 1.0)
}

fn health_label(score: f64) -> &'static str {
    if score >= 0.85 {
        "optimal"
    } else if score >= 0.6 {
        "stable"
    } else if score >= 0.35 {
        "strained"
    } else {
        "critical"
    }
}

fn grayscale_color(score: f64) -> String {
    let value = (score.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{value:02X}{value:02X}{value:02X}")
}

fn dimension_color(index: usize, score: f64) -> String {
    let hue = ((index as f64 * 0.61803398875) % 1.0) * 360.0;
    let (r, g, b) = hsv_to_rgb(hue, 0.6, 0.4 + 0.6 * score.clamp(0.0, 1.0));
    format!("#{r:02X}{g:02X}{b:02X}")
}

fn dimension_palette(dimensions: Option<&DimensionReport>, max_len: usize) -> Vec<HealthDimensionPalette> {
    let Some(report) = dimensions else {
        return Vec::new();
    };
    let mut palette = Vec::new();
    for (idx, dim) in report.dimensions.iter().enumerate() {
        if palette.len() >= max_len {
            break;
        }
        let color = dimension_color(idx, 0.75);
        palette.push(HealthDimensionPalette {
            name: dim.name.clone(),
            color,
        });
    }
    palette
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let h_prime = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if (0.0..1.0).contains(&h_prime) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&h_prime) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&h_prime) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&h_prime) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&h_prime) {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    let r = ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    let g = ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    let b = ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::physiology_runtime::PhysiologyDeviation;
    use crate::streaming::schema::TokenBatch;

    #[test]
    fn health_overlay_reports_color() {
        let mut runtime = HealthOverlayRuntime::default();
        let report = PhysiologyReport {
            timestamp: Timestamp { unix: 5 },
            deviations: vec![PhysiologyDeviation {
                context: "e1".to_string(),
                deviation_index: 0.5,
                deviation_vector: vec![0.2; 10],
                sample_count: 3,
            }],
            overall_index: 0.5,
            template_count: 1,
        };
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 5 },
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let overlay = runtime
            .update(&batch, Some(&report), None)
            .expect("overlay");
        assert_eq!(overlay.entities.len(), 1);
        assert!(overlay.entities[0].color.starts_with('#'));
    }
}
