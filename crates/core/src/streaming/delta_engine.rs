//! Delta engine — tracks per-entity velocity in 6-D health space.
//!
//! Each call to `update()` ingests the current [`EntityHealthOverlay`] set and
//! produces:
//!
//!  1. A **velocity vector** — rate-of-change per dimension (Δ/sec), computed
//!     from the two most recent observations.
//!  2. A **linear prediction** at each configured horizon — where the entity
//!     is headed assuming constant velocity.
//!  3. An **attractor correction** — when a motif attractor is detected the
//!     prediction is pulled toward the attractor's expected health signature
//!     (passed in as `attractor_strength` 0–1 per entity).
//!
//! No hard-coded values.  Velocity is always derived from the actual observed
//! frame-to-frame difference divided by the real elapsed time.

use crate::schema::Timestamp;
use crate::streaming::entity_health::{EntityHealthOverlay, HealthVector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaEngineConfig {
    /// How many past observations to keep per entity (for smoothing).
    pub window_size: usize,
    /// Minimum elapsed time (secs) between observations to compute velocity.
    /// Avoids division by near-zero when frames arrive faster than the clock.
    pub min_dt_secs: f64,
    /// Future horizons (in seconds) for which predictions are generated.
    pub prediction_horizons: Vec<f64>,
    /// EMA alpha applied to the velocity estimate (0 = frozen, 1 = no memory).
    pub velocity_alpha: f64,
    /// How strongly a detected attractor bends the linear prediction.
    /// 0 = pure linear, 1 = prediction fully replaced by attractor estimate.
    pub attractor_blend: f64,
}

impl Default for DeltaEngineConfig {
    fn default() -> Self {
        Self {
            window_size:          8,
            min_dt_secs:          0.05,
            prediction_horizons:  vec![1.0, 5.0, 30.0],
            velocity_alpha:       0.35,
            attractor_blend:      0.40,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

/// Rate-of-change of a health vector, in units of health-score/second.
/// Positive = improving, negative = degrading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthVelocity {
    pub entity_id:    String,
    /// Per-dimension velocity (Δ/sec, clamped to ±1).
    pub velocity:     HealthVector,
    /// L2 magnitude of the velocity vector — overall rate of change.
    pub magnitude:    f64,
    /// Dominant dimension of change (index 0-5 → structural..informational).
    pub leading_dim:  usize,
    /// True when the entity is on a net-improving trajectory.
    pub improving:    bool,
}

/// Predicted health state at a future time horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthPrediction {
    pub entity_id:       String,
    /// How far ahead this prediction looks (seconds).
    pub horizon_secs:    f64,
    /// Predicted health vector at that horizon.
    pub predicted:       HealthVector,
    /// Composite score of the prediction (0=critical, 1=optimal).
    pub predicted_score: f64,
    /// Label: "optimal" | "stable" | "strained" | "critical".
    pub predicted_label: String,
    /// Fraction of the prediction sourced from attractor pull (0=pure linear).
    pub attractor_frac:  f64,
}

/// Full delta report for one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaReport {
    pub timestamp:   Timestamp,
    pub velocities:  Vec<HealthVelocity>,
    pub predictions: Vec<HealthPrediction>,
}

// ── Runtime ───────────────────────────────────────────────────────────────────

struct EntityHistory {
    /// Recent (timestamp, smoothed_vector) pairs.
    window:   VecDeque<(Timestamp, HealthVector)>,
    /// EMA-smoothed velocity, updated each frame.
    velocity: HealthVector,
}

pub struct DeltaEngine {
    config:  DeltaEngineConfig,
    history: HashMap<String, EntityHistory>,
}

impl DeltaEngine {
    pub fn new(config: DeltaEngineConfig) -> Self {
        Self { config, history: HashMap::new() }
    }

    /// Ingest the latest overlay set and produce a DeltaReport.
    ///
    /// `attractor_strength` maps entity_id → 0.0..1.0 attractor pull from the
    /// motif runtime (0 = no known attractor, 1 = strong attractor).
    /// Pass an empty map if motif data is unavailable.
    pub fn update(
        &mut self,
        timestamp: Timestamp,
        overlays: &[EntityHealthOverlay],
        attractor_strength: &HashMap<String, f64>,
    ) -> DeltaReport {
        let mut velocities  = Vec::with_capacity(overlays.len());
        let mut predictions = Vec::new();

        for ov in overlays {
            let eid = &ov.entity_id;
            let current = &ov.vector;

            let entry = self.history.entry(eid.clone()).or_insert_with(|| {
                EntityHistory {
                    window:   VecDeque::new(),
                    velocity: HealthVector::default(),
                }
            });

            // Compute instantaneous velocity if we have a previous frame.
            if let Some((prev_ts, prev_v)) = entry.window.back() {
                let dt = (timestamp.unix - prev_ts.unix) as f64;
                if dt >= self.config.min_dt_secs {
                    let inst = velocity_between(prev_v, current, dt);
                    // EMA-smooth the velocity estimate.
                    let alpha = self.config.velocity_alpha;
                    entry.velocity = blend_velocity(alpha, &inst, &entry.velocity);
                }
            }

            // Push current observation, evict if window full.
            entry.window.push_back((timestamp, current.clone()));
            while entry.window.len() > self.config.window_size {
                entry.window.pop_front();
            }

            let vel = entry.velocity.clone();
            let magnitude = vel_magnitude(&vel);
            let leading_dim = leading_dimension(&vel);
            let improving   = vel.composite() > 0.0;

            velocities.push(HealthVelocity {
                entity_id: eid.clone(),
                velocity:  vel.clone(),
                magnitude,
                leading_dim,
                improving,
            });

            // Predictions at each configured horizon.
            let attr_str = attractor_strength.get(eid.as_str()).copied().unwrap_or(0.0);
            for &horizon in &self.config.prediction_horizons {
                let linear = linear_predict(current, &vel, horizon);
                let predicted = if attr_str > 0.001 {
                    // Bend prediction toward the attractor's implied state.
                    // We approximate the attractor as high temporal/structural
                    // health (what an attractor state implies).
                    let attr_vector = attractor_target(current, attr_str);
                    let blend = self.config.attractor_blend * attr_str;
                    interpolate_vector(&linear, &attr_vector, blend.clamp(0.0, 1.0))
                } else {
                    linear
                };
                let score = predicted.composite();
                let label = crate::streaming::entity_health::health_label(score).to_string();
                predictions.push(HealthPrediction {
                    entity_id:      eid.clone(),
                    horizon_secs:   horizon,
                    predicted_score: score,
                    predicted_label: label,
                    attractor_frac: if attr_str > 0.001 {
                        (self.config.attractor_blend * attr_str).clamp(0.0, 1.0)
                    } else { 0.0 },
                    predicted,
                });
            }
        }

        DeltaReport { timestamp, velocities, predictions }
    }

    /// Retrieve the latest velocity report for a single entity without a full
    /// update cycle.  Returns None if the entity has never been seen.
    pub fn velocity_for(&self, entity_id: &str) -> Option<&HealthVector> {
        self.history.get(entity_id).map(|h| &h.velocity)
    }

    /// Clear history for an entity (e.g. after a large scene discontinuity).
    pub fn reset_entity(&mut self, entity_id: &str) {
        self.history.remove(entity_id);
    }
}

impl Default for DeltaEngine {
    fn default() -> Self { Self::new(DeltaEngineConfig::default()) }
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

fn velocity_between(prev: &HealthVector, curr: &HealthVector, dt: f64) -> HealthVector {
    let a = prev.as_array();
    let b = curr.as_array();
    let v: Vec<f64> = a.iter().zip(b.iter())
        .map(|(p, c)| ((c - p) / dt).clamp(-1.0, 1.0))
        .collect();
    HealthVector {
        structural:    v[0],
        energetic:     v[1],
        temporal:      v[2],
        intentional:   v[3],
        environmental: v[4],
        informational: v[5],
    }
}

fn blend_velocity(alpha: f64, new: &HealthVector, old: &HealthVector) -> HealthVector {
    let n = new.as_array();
    let o = old.as_array();
    let b: Vec<f64> = n.iter().zip(o.iter())
        .map(|(ni, oi)| (alpha * ni + (1.0 - alpha) * oi).clamp(-1.0, 1.0))
        .collect();
    HealthVector {
        structural: b[0], energetic: b[1], temporal: b[2],
        intentional: b[3], environmental: b[4], informational: b[5],
    }
}

fn linear_predict(current: &HealthVector, velocity: &HealthVector, horizon: f64) -> HealthVector {
    let c = current.as_array();
    let v = velocity.as_array();
    let p: Vec<f64> = c.iter().zip(v.iter())
        .map(|(ci, vi)| (ci + vi * horizon).clamp(0.0, 1.0))
        .collect();
    HealthVector {
        structural: p[0], energetic: p[1], temporal: p[2],
        intentional: p[3], environmental: p[4], informational: p[5],
    }
}

/// Approximate the health state an entity would settle into if it reached its
/// current motif attractor — nudge toward higher structural + temporal health.
fn attractor_target(current: &HealthVector, strength: f64) -> HealthVector {
    // An attractor implies stable, predictable behavior — high temporal, stable structural.
    let target_bump = 0.15 * strength;
    HealthVector {
        structural:    (current.structural    + target_bump).clamp(0.0, 1.0),
        energetic:     current.energetic,
        temporal:      (current.temporal      + target_bump * 1.5).clamp(0.0, 1.0),
        intentional:   current.intentional,
        environmental: current.environmental,
        informational: (current.informational + target_bump * 0.5).clamp(0.0, 1.0),
    }
}

fn interpolate_vector(a: &HealthVector, b: &HealthVector, t: f64) -> HealthVector {
    let av = a.as_array();
    let bv = b.as_array();
    let r: Vec<f64> = av.iter().zip(bv.iter())
        .map(|(ai, bi)| (ai * (1.0 - t) + bi * t).clamp(0.0, 1.0))
        .collect();
    HealthVector {
        structural: r[0], energetic: r[1], temporal: r[2],
        intentional: r[3], environmental: r[4], informational: r[5],
    }
}

fn vel_magnitude(v: &HealthVector) -> f64 {
    v.as_array().iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn leading_dimension(v: &HealthVector) -> usize {
    v.as_array().iter().enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::entity_health::{EntityHealthOverlay, HealthRing};

    fn make_overlay(eid: &str, ts_unix: i64, v: HealthVector) -> EntityHealthOverlay {
        let composite = v.composite();
        EntityHealthOverlay {
            entity_id: eid.to_string(),
            timestamp: Timestamp { unix: ts_unix },
            label: crate::streaming::entity_health::health_label(composite).to_string(),
            rings: vec![],
            position: None,
            motif_entropy: 0.0,
            composite,
            vector: v,
        }
    }

    #[test]
    fn velocity_computed_correctly() {
        let mut engine = DeltaEngine::default();
        let v0 = HealthVector { structural: 0.5, energetic: 0.5, temporal: 0.5,
            intentional: 0.5, environmental: 0.5, informational: 0.5 };
        let v1 = HealthVector { structural: 0.6, energetic: 0.5, temporal: 0.5,
            intentional: 0.5, environmental: 0.5, informational: 0.5 };
        let o0 = make_overlay("A", 0, v0);
        let o1 = make_overlay("A", 1, v1);  // dt = 1 sec
        engine.update(Timestamp { unix: 0 }, &[o0], &HashMap::new());
        let report = engine.update(Timestamp { unix: 1 }, &[o1], &HashMap::new());
        let vel = &report.velocities[0];
        // Structural increased by 0.1 over 1s => ~0.1/s (after EMA smoothing, lower)
        assert!(vel.velocity.structural > 0.0, "structural velocity should be positive");
    }

    #[test]
    fn predictions_clamped_to_unit() {
        let mut engine = DeltaEngine::default();
        // Entity near ceiling — fast upward velocity should not push above 1.0
        let v_high = HealthVector { structural: 0.95, energetic: 0.95, temporal: 0.95,
            intentional: 0.95, environmental: 0.95, informational: 0.95 };
        let v_higher = HealthVector { structural: 1.0, energetic: 1.0, temporal: 1.0,
            intentional: 1.0, environmental: 1.0, informational: 1.0 };
        engine.update(Timestamp { unix: 0 }, &[make_overlay("B", 0, v_high)], &HashMap::new());
        let rep = engine.update(Timestamp { unix: 1 }, &[make_overlay("B", 1, v_higher)], &HashMap::new());
        for pred in &rep.predictions {
            let arr = pred.predicted.as_array();
            for &d in &arr { assert!(d >= 0.0 && d <= 1.0, "prediction out of [0,1]: {d}"); }
        }
    }
}
