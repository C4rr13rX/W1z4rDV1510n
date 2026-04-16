//! 6-dimensional entity health model with concentric ring color overlay.
//!
//! The six semantic health dimensions map to existing runtime outputs:
//!
//! | Dimension     | Sources                                              |
//! |---------------|------------------------------------------------------|
//! | structural    | SceneEntityReport.stability + inverted physiology deviation |
//! | energetic     | physiology deviation_vector mean + survival physiology |
//! | temporal      | inverted motif entropy + attractor bonus             |
//! | intentional   | intent_magnitude + cooperation/conflict balance     |
//! | environmental | survival score + EEM match confidence               |
//! | informational | EEM top-candidate confidence + neuro label certainty|
//!
//! Visual model: concentric rings ordered outermost=black (near-destruction)
//! to innermost=white (optimal). Ring radius = dimension score. At score=1.0
//! the single white ring fills the overlay; at score=0.0 all rings are black.

use crate::streaming::physiology_runtime::PhysiologyReport;
use crate::streaming::scene_runtime::SceneEntityReport;
use crate::streaming::survival::{SurvivalEntityMetrics, SurvivalReport};
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Core types ────────────────────────────────────────────────────────────────

/// The six semantic health dimensions for one entity at one moment.
/// All values are 0.0 (critical/unknown) → 1.0 (optimal/certain).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthVector {
    /// Physical integrity and spatial coherence.
    pub structural:    f64,
    /// Activity level and metabolic-like energy state.
    pub energetic:     f64,
    /// Consistency over time; how predictable the entity's trajectory is.
    pub temporal:      f64,
    /// Clarity and coherence of observed intent / purpose.
    pub intentional:   f64,
    /// Coupling to and interaction quality with the surrounding environment.
    pub environmental: f64,
    /// Epistemic certainty: how well the system understands this entity.
    pub informational: f64,
}

impl HealthVector {
    /// Weighted mean across all six dimensions (equal weights).
    pub fn composite(&self) -> f64 {
        (self.structural + self.energetic + self.temporal
            + self.intentional + self.environmental + self.informational)
            / 6.0
    }

    pub fn as_array(&self) -> [f64; 6] {
        [self.structural, self.energetic, self.temporal,
         self.intentional, self.environmental, self.informational]
    }

    /// L2 distance in 6-D space between two vectors.
    pub fn distance(&self, other: &HealthVector) -> f64 {
        self.as_array().iter()
            .zip(other.as_array().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Element-wise difference (self - other), clamped to [-1, 1].
    pub fn delta(&self, other: &HealthVector) -> HealthVector {
        let a = self.as_array();
        let b = other.as_array();
        HealthVector {
            structural:    (a[0] - b[0]).clamp(-1.0, 1.0),
            energetic:     (a[1] - b[1]).clamp(-1.0, 1.0),
            temporal:      (a[2] - b[2]).clamp(-1.0, 1.0),
            intentional:   (a[3] - b[3]).clamp(-1.0, 1.0),
            environmental: (a[4] - b[4]).clamp(-1.0, 1.0),
            informational: (a[5] - b[5]).clamp(-1.0, 1.0),
        }
    }

    pub fn dim_name(idx: usize) -> &'static str {
        match idx {
            0 => "structural",
            1 => "energetic",
            2 => "temporal",
            3 => "intentional",
            4 => "environmental",
            _ => "informational",
        }
    }
}

/// One ring in the concentric-ring visual model.
/// Rings are sorted outermost=lowest-score to innermost=highest-score.
/// Black (#000000) = near-destruction; White (#FFFFFF) = optimal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRing {
    /// Normalized radius 0.0–1.0 (1.0 = outer edge of the overlay circle).
    pub radius: f64,
    /// Hex color string (e.g. "#A0A0A0").
    pub color:  String,
    /// Which health dimension this ring represents.
    pub dimension: String,
    /// The raw 0.0–1.0 score for this dimension.
    pub score: f64,
}

/// Full overlay record for one entity at one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHealthOverlay {
    pub entity_id:   String,
    pub timestamp:   Timestamp,
    pub vector:      HealthVector,
    /// Composite 0.0–1.0 summary.
    pub composite:   f64,
    /// Health label: "optimal" | "stable" | "strained" | "critical".
    pub label:       String,
    /// Concentric rings ordered outermost → innermost (black → white).
    pub rings:       Vec<HealthRing>,
    /// Position in 3-D space if available.
    pub position:    Option<[f64; 3]>,
    /// Fractional entropy of the motif trajectory (0=attractor, 1=chaotic).
    pub motif_entropy: f64,
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHealthConfig {
    /// EMA alpha for per-entity health baselines (0=frozen, 1=no memory).
    pub baseline_alpha: f64,
    /// Physiology deviation scale: how many sigma counts as fully degraded.
    pub deviation_scale: f64,
    /// Weight of survival score in the environmental dimension (0–1).
    pub survival_weight: f64,
    /// Weight of intent_magnitude in the intentional dimension (0–1).
    pub intent_weight: f64,
    /// EEM confidence threshold below which informational score is halved.
    pub eem_min_confidence: f64,
    /// Maximum motif entropy assumed when normalising (beyond this = fully chaotic).
    pub max_motif_entropy: f64,
}

impl Default for EntityHealthConfig {
    fn default() -> Self {
        Self {
            baseline_alpha:    0.25,
            deviation_scale:   3.0,
            survival_weight:   0.55,
            intent_weight:     0.60,
            eem_min_confidence: 0.30,
            max_motif_entropy:  4.0,
        }
    }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

/// Maintains per-entity EMA baselines and computes `EntityHealthOverlay`
/// from the existing runtime report set (scene + physiology + survival).
pub struct EntityHealthRuntime {
    config:    EntityHealthConfig,
    baselines: HashMap<String, HealthVector>,
}

impl EntityHealthRuntime {
    pub fn new(config: EntityHealthConfig) -> Self {
        Self { config, baselines: HashMap::new() }
    }

    /// Compute the overlay for every entity present in at least one of the
    /// provided reports.  Pass `None` for any runtime that did not fire this
    /// frame.
    /// Compute the overlay for every entity present in at least one of the
    /// provided reports.  Pass `None` for any runtime that did not fire this
    /// frame.
    ///
    /// `motif_entropy` — combined entropy from HierarchicalMotifRuntime.
    /// `eem_confidence` — top EEM candidate confidence (0–1).
    pub fn update(
        &mut self,
        timestamp: Timestamp,
        scene:      Option<&[SceneEntityReport]>,
        physiology: Option<&PhysiologyReport>,
        survival:   Option<&SurvivalReport>,
        motif_entropy: f64,
        eem_confidence: f64,
    ) -> Vec<EntityHealthOverlay> {
        // Build fast-lookup maps.
        let scene_map:   HashMap<&str, &SceneEntityReport> = scene
            .map(|s| s.iter().map(|e| (e.entity_id.as_str(), e)).collect())
            .unwrap_or_default();
        let physio_map:  HashMap<&str, f64> = physiology
            .map(|p| p.deviations.iter()
                .map(|d| (d.context.as_str(), d.deviation_index))
                .collect())
            .unwrap_or_default();
        let survival_map: HashMap<&str, &SurvivalEntityMetrics> = survival
            .map(|s| s.entities.iter().map(|e| (e.entity_id.as_str(), e)).collect())
            .unwrap_or_default();

        // Union of all entity IDs seen this frame.
        let mut entity_ids: Vec<String> = Vec::new();
        for id in scene_map.keys() { entity_ids.push(id.to_string()); }
        for id in physio_map.keys() {
            if !scene_map.contains_key(id) { entity_ids.push(id.to_string()); }
        }
        for id in survival_map.keys() {
            if !scene_map.contains_key(id) && !physio_map.contains_key(id) {
                entity_ids.push(id.to_string());
            }
        }

        let mut overlays = Vec::with_capacity(entity_ids.len());
        for eid in &entity_ids {
            let scene_ent  = scene_map.get(eid.as_str()).copied();
            let phys_dev   = physio_map.get(eid.as_str()).copied().unwrap_or(0.0);
            let surv       = survival_map.get(eid.as_str()).copied();
            let position   = scene_ent.map(|s| [s.position.x, s.position.y, s.position.z]);

            let raw = self.compute_raw(
                scene_ent, phys_dev, surv,
                motif_entropy, eem_confidence,
            );
            let smoothed = self.smooth(eid, raw);
            let composite = smoothed.composite();
            let rings = make_rings(&smoothed);
            let label = health_label(composite).to_string();

            overlays.push(EntityHealthOverlay {
                entity_id:    eid.clone(),
                timestamp,
                vector:       smoothed,
                composite,
                label,
                rings,
                position,
                motif_entropy: (motif_entropy / self.config.max_motif_entropy).clamp(0.0, 1.0),
            });
        }
        overlays
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn compute_raw(
        &self,
        scene:        Option<&SceneEntityReport>,
        phys_dev:     f64,
        surv:         Option<&SurvivalEntityMetrics>,
        motif_entropy:f64,
        eem_confidence: f64,
    ) -> HealthVector {
        // 1. Structural — spatial stability + inverse physiology deviation.
        let stability  = scene.map(|s| s.stability).unwrap_or(0.5);
        let phys_score = (1.0 / (1.0 + phys_dev.abs() / self.config.deviation_scale)).clamp(0.0, 1.0);
        let structural = (stability * 0.6 + phys_score * 0.4).clamp(0.0, 1.0);

        // 2. Energetic — physiology activity + survival physiology score.
        let activity    = scene.map(|s| (s.speed / 5.0_f64).clamp(0.0, 1.0)).unwrap_or(0.3);
        let surv_phys   = surv.map(|s| s.survival_score).unwrap_or(0.5);
        let energetic   = (activity * 0.4 + surv_phys * 0.4 + phys_score * 0.2).clamp(0.0, 1.0);

        // 3. Temporal — inverted motif entropy + attractor bonus.
        let max_ent = self.config.max_motif_entropy.max(0.01);
        let entropy_score = (1.0 - (motif_entropy / max_ent).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let temporal = entropy_score;

        // 4. Intentional — intent magnitude + cooperation/conflict balance.
        let intent   = surv.map(|s| s.intent_magnitude.clamp(0.0, 1.0)).unwrap_or(0.0);
        let coop_cf  = surv.map(|s| {
            let net = (s.cooperation_in - s.conflict_in + 1.0) / 2.0;
            net.clamp(0.0, 1.0)
        }).unwrap_or(0.5);
        let w = self.config.intent_weight.clamp(0.0, 1.0);
        let intentional = (intent * w + coop_cf * (1.0 - w)).clamp(0.0, 1.0);

        // 5. Environmental — survival score × environment coupling weight.
        let surv_score = surv.map(|s| s.survival_score).unwrap_or(0.5);
        let w2 = self.config.survival_weight.clamp(0.0, 1.0);
        let environmental = (surv_score * w2 + structural * (1.0 - w2)).clamp(0.0, 1.0);

        // 6. Informational — EEM confidence + epistemic baseline.
        let eem_c = if eem_confidence >= self.config.eem_min_confidence {
            eem_confidence
        } else {
            eem_confidence * 0.5
        };
        let informational = (eem_c * 0.7 + phys_score * 0.3).clamp(0.0, 1.0);

        HealthVector { structural, energetic, temporal, intentional, environmental, informational }
    }

    fn smooth(&mut self, entity_id: &str, raw: HealthVector) -> HealthVector {
        let alpha = self.config.baseline_alpha.clamp(0.0, 1.0);
        let entry = self.baselines.entry(entity_id.to_string()).or_insert_with(|| raw.clone());
        let smoothed = HealthVector {
            structural:    ema(alpha, raw.structural,    entry.structural),
            energetic:     ema(alpha, raw.energetic,     entry.energetic),
            temporal:      ema(alpha, raw.temporal,      entry.temporal),
            intentional:   ema(alpha, raw.intentional,   entry.intentional),
            environmental: ema(alpha, raw.environmental, entry.environmental),
            informational: ema(alpha, raw.informational, entry.informational),
        };
        *entry = smoothed.clone();
        smoothed
    }
}

impl Default for EntityHealthRuntime {
    fn default() -> Self { Self::new(EntityHealthConfig::default()) }
}

// ── Ring model ────────────────────────────────────────────────────────────────

/// Build the concentric ring set from a health vector.
/// Rings are sorted outermost (lowest score = black) to innermost (highest = white).
pub fn make_rings(v: &HealthVector) -> Vec<HealthRing> {
    let names = ["structural", "energetic", "temporal",
                 "intentional", "environmental", "informational"];
    let scores = v.as_array();
    // Sort indices by score ascending (worst → best)
    let mut order: Vec<usize> = (0..6).collect();
    order.sort_unstable_by(|&a, &b| scores[a].partial_cmp(&scores[b])
        .unwrap_or(std::cmp::Ordering::Equal));

    order.iter().enumerate().map(|(rank, &dim)| {
        let score  = scores[dim];
        // Radius: outermost ring gets radius 1.0, innermost gets 1/6
        let radius = 1.0 - (rank as f64 / 7.0);
        let color  = grayscale(score);
        HealthRing {
            radius,
            color,
            dimension: names[dim].to_string(),
            score,
        }
    }).collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ema(alpha: f64, new: f64, old: f64) -> f64 {
    (alpha * new + (1.0 - alpha) * old).clamp(0.0, 1.0)
}

pub fn grayscale(score: f64) -> String {
    let v = (score.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{v:02X}{v:02X}{v:02X}")
}

pub fn health_label(composite: f64) -> &'static str {
    if composite >= 0.85 { "optimal" }
    else if composite >= 0.60 { "stable" }
    else if composite >= 0.35 { "strained" }
    else { "critical" }
}

/// Convert a 6-D health vector into a single hex color that blends across all
/// six dimension hues, weighted by score.  Low composite → dark; high → bright.
/// Used as the "dominant color" for the overlay at a glance.
pub fn blend_color(v: &HealthVector) -> String {
    // Per-dimension hues (evenly spread around the wheel).
    const HUES: [f64; 6] = [0.0, 60.0, 120.0, 210.0, 270.0, 330.0];
    let scores = v.as_array();
    let total_weight: f64 = scores.iter().sum::<f64>() + 1e-9;
    let hue = HUES.iter().zip(scores.iter())
        .map(|(&h, &s)| h * s)
        .sum::<f64>() / total_weight;
    let composite = v.composite();
    let saturation = if composite > 0.8 { 0.15 } else { 0.70 };
    let value = 0.15 + composite * 0.85;
    let (r, g, b) = hsv_to_rgb(hue, saturation, value);
    format!("#{r:02X}{g:02X}{b:02X}")
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let hp = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if hp < 1.0 { (c, x, 0.0) }
        else if hp < 2.0 { (x, c, 0.0) }
        else if hp < 3.0 { (0.0, c, x) }
        else if hp < 4.0 { (0.0, x, c) }
        else if hp < 5.0 { (x, 0.0, c) }
        else { (c, 0.0, x) };
    let m = v - c;
    let to_u8 = |f: f64| (f + m).clamp(0.0, 1.0).mul_add(255.0, 0.5) as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grayscale_extremes() {
        assert_eq!(grayscale(0.0), "#000000");
        assert_eq!(grayscale(1.0), "#FFFFFF");
    }
    #[test]
    fn rings_sorted_by_score() {
        let v = HealthVector {
            structural: 0.9, energetic: 0.1, temporal: 0.5,
            intentional: 0.7, environmental: 0.3, informational: 0.6,
        };
        let rings = make_rings(&v);
        // Outermost ring (first) should have the lowest score.
        assert!(rings[0].score <= rings[rings.len()-1].score);
    }
    #[test]
    fn composite_bounds() {
        let v = HealthVector { structural: 1.0, energetic: 1.0, temporal: 1.0,
            intentional: 1.0, environmental: 1.0, informational: 1.0 };
        assert!((v.composite() - 1.0).abs() < 1e-9);
    }
}
