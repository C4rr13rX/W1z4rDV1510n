/// 6-dimensional universal health model.
///
/// Applies to any entity — human, animal, plant, machine — whose operational
/// state can be observed through sensor streams.  The six dimensions are chosen
/// to be orthogonal yet covering: you can score any known failure mode of any
/// physical system along at least one of them.
///
/// Color encoding
/// ──────────────
/// Health → HSV where:
///   Value       = overall scalar  (0 = dead/black, 1 = optimal/white)
///   Hue         = weighted mean of dimension hue anchors, dominated by the
///                 weakest dimensions (what is wrong drives the color)
///   Saturation  = deviation from the established baseline for this entity type
///                 (0 = exactly at baseline, 1 = maximally off-baseline)
///
/// Scope stacking
/// ──────────────
/// Each entity can carry health vectors at multiple scopes.  Scopes compose
/// multiplicatively: organism health = f(body_part_1, body_part_2, …).
/// A body-part collapse propagates upward into organism health via the
/// propagation engine (see `propagation.rs`).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ─── Dimensions ───────────────────────────────────────────────────────────────

/// The six universal health dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthDimension {
    /// Structural Integrity — can the physical substrate hold its topology?
    /// Hue anchor: 0° (red)
    StructuralIntegrity,
    /// Energetic Flux — can the entity acquire, convert, and distribute energy?
    /// Hue anchor: 30° (orange)
    EnergeticFlux,
    /// Regulatory Control — can subsystems maintain homeostasis and coordinate?
    /// Hue anchor: 210° (blue)
    RegulatoryControl,
    /// Functional Output — can the entity execute its characteristic operations?
    /// Hue anchor: 120° (green)
    FunctionalOutput,
    /// Adaptive Reserve — how much capacity remains to absorb further stress?
    /// Hue anchor: 270° (violet)
    AdaptiveReserve,
    /// Temporal Coherence — are the entity's rhythms and cycles intact?
    /// Hue anchor: 60° (yellow)
    TemporalCoherence,
}

impl HealthDimension {
    pub const ALL: [HealthDimension; 6] = [
        HealthDimension::StructuralIntegrity,
        HealthDimension::EnergeticFlux,
        HealthDimension::RegulatoryControl,
        HealthDimension::FunctionalOutput,
        HealthDimension::AdaptiveReserve,
        HealthDimension::TemporalCoherence,
    ];

    /// HSV hue anchor in degrees [0, 360).
    pub fn hue_anchor(self) -> f32 {
        match self {
            HealthDimension::StructuralIntegrity => 0.0,
            HealthDimension::EnergeticFlux => 30.0,
            HealthDimension::RegulatoryControl => 210.0,
            HealthDimension::FunctionalOutput => 120.0,
            HealthDimension::AdaptiveReserve => 270.0,
            HealthDimension::TemporalCoherence => 60.0,
        }
    }

    /// Human-readable short name.
    pub fn short_name(self) -> &'static str {
        match self {
            HealthDimension::StructuralIntegrity => "SI",
            HealthDimension::EnergeticFlux => "EF",
            HealthDimension::RegulatoryControl => "RC",
            HealthDimension::FunctionalOutput => "FO",
            HealthDimension::AdaptiveReserve => "AR",
            HealthDimension::TemporalCoherence => "TC",
        }
    }

    pub fn index(self) -> usize {
        match self {
            HealthDimension::StructuralIntegrity => 0,
            HealthDimension::EnergeticFlux => 1,
            HealthDimension::RegulatoryControl => 2,
            HealthDimension::FunctionalOutput => 3,
            HealthDimension::AdaptiveReserve => 4,
            HealthDimension::TemporalCoherence => 5,
        }
    }
}

// ─── Scope ────────────────────────────────────────────────────────────────────

/// Spatial / organisational scope at which a health vector is computed.
/// Scopes nest: body_part → organ → organism → group → environment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// A specific body region or component part (e.g. "left_leg", "cpu_core_0").
    BodyPart(String),
    /// An organ or subsystem (e.g. "cardiovascular", "power_supply").
    Organ(String),
    /// The whole organism or machine.
    Organism,
    /// A social / functional group.
    Group(String),
    /// The ambient environment.
    Environment,
}

impl Scope {
    pub fn label(&self) -> String {
        match self {
            Scope::BodyPart(s) => format!("body_part::{s}"),
            Scope::Organ(s) => format!("organ::{s}"),
            Scope::Organism => "organism".to_string(),
            Scope::Group(s) => format!("group::{s}"),
            Scope::Environment => "environment".to_string(),
        }
    }

    /// Propagation weight — how strongly does damage at this scope affect the
    /// parent scope?  Body parts and organs propagate strongly; groups weakly.
    pub fn propagation_weight(&self) -> f32 {
        match self {
            Scope::BodyPart(_) => 0.7,
            Scope::Organ(_) => 0.85,
            Scope::Organism => 1.0,
            Scope::Group(_) => 0.3,
            Scope::Environment => 0.15,
        }
    }
}

// ─── HealthVector ─────────────────────────────────────────────────────────────

/// Six-dimensional health state for a single entity at a single scope.
///
/// All fields are in [0.0, 1.0] where 0 = worst, 1 = optimal.
/// `uncertainty` is in [0.0, 1.0] where 1 = completely unknown.
/// When uncertainty is high the system is still naïve about this entity type
/// and the scores should be interpreted cautiously.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthVector {
    /// Structural Integrity
    pub si: f32,
    /// Energetic Flux
    pub ef: f32,
    /// Regulatory Control
    pub rc: f32,
    /// Functional Output
    pub fo: f32,
    /// Adaptive Reserve
    pub ar: f32,
    /// Temporal Coherence
    pub tc: f32,
    /// Per-dimension uncertainty [0,1]
    pub uncertainty: [f32; 6],
}

impl Default for HealthVector {
    fn default() -> Self {
        // Start maximally uncertain — mid-point health, full uncertainty.
        Self {
            si: 0.5,
            ef: 0.5,
            rc: 0.5,
            fo: 0.5,
            ar: 0.5,
            tc: 0.5,
            uncertainty: [1.0; 6],
        }
    }
}

impl HealthVector {
    /// Create a fully healthy, fully certain vector (known-optimal baseline).
    pub fn optimal() -> Self {
        Self {
            si: 1.0, ef: 1.0, rc: 1.0, fo: 1.0, ar: 1.0, tc: 1.0,
            uncertainty: [0.0; 6],
        }
    }

    /// Create a dead vector (all dimensions collapsed).
    pub fn dead() -> Self {
        Self {
            si: 0.0, ef: 0.0, rc: 0.0, fo: 0.0, ar: 0.0, tc: 0.0,
            uncertainty: [0.0; 6],
        }
    }

    pub fn get(&self, dim: HealthDimension) -> f32 {
        match dim {
            HealthDimension::StructuralIntegrity => self.si,
            HealthDimension::EnergeticFlux       => self.ef,
            HealthDimension::RegulatoryControl   => self.rc,
            HealthDimension::FunctionalOutput    => self.fo,
            HealthDimension::AdaptiveReserve     => self.ar,
            HealthDimension::TemporalCoherence   => self.tc,
        }
    }

    pub fn set(&mut self, dim: HealthDimension, val: f32) {
        let v = val.clamp(0.0, 1.0);
        match dim {
            HealthDimension::StructuralIntegrity => self.si = v,
            HealthDimension::EnergeticFlux       => self.ef = v,
            HealthDimension::RegulatoryControl   => self.rc = v,
            HealthDimension::FunctionalOutput    => self.fo = v,
            HealthDimension::AdaptiveReserve     => self.ar = v,
            HealthDimension::TemporalCoherence   => self.tc = v,
        }
    }

    pub fn as_array(&self) -> [f32; 6] {
        [self.si, self.ef, self.rc, self.fo, self.ar, self.tc]
    }

    /// Weighted mean — dimensions with high uncertainty contribute less.
    pub fn overall_score(&self) -> f32 {
        let vals = self.as_array();
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for (i, &v) in vals.iter().enumerate() {
            let w = 1.0 - self.uncertainty[i];  // high uncertainty → low weight
            sum += v * w;
            weight_sum += w;
        }
        if weight_sum < 1e-6 { return 0.5; }
        (sum / weight_sum).clamp(0.0, 1.0)
    }

    /// Apply a delta to one dimension (negative = damage, positive = recovery).
    /// Uncertainty decreases with each observation (we learn more about the entity).
    pub fn apply_delta(&mut self, dim: HealthDimension, delta: f32, confidence: f32) {
        let i = dim.index();
        let current = self.get(dim);
        let new_val = (current + delta).clamp(0.0, 1.0);
        self.set(dim, new_val);
        // Uncertainty decays toward zero as confident observations accumulate.
        self.uncertainty[i] = (self.uncertainty[i] * (1.0 - confidence * 0.1)).clamp(0.0, 1.0);
    }

    /// Lerp toward a target vector (used for baseline convergence).
    pub fn lerp_toward(&self, target: &HealthVector, alpha: f32) -> HealthVector {
        let mut out = self.clone();
        let vals = self.as_array();
        let tvals = target.as_array();
        for i in 0..6 {
            let new_val = vals[i] + (tvals[i] - vals[i]) * alpha;
            let dim = HealthDimension::ALL[i];
            out.set(dim, new_val);
            out.uncertainty[i] = self.uncertainty[i] + (target.uncertainty[i] - self.uncertainty[i]) * alpha;
        }
        out
    }

    /// Encode as HSV (hue 0–360, saturation 0–1, value 0–1).
    pub fn to_hsv(&self, baseline: Option<&HealthVector>) -> HsvColor {
        health_to_hsv(self, baseline)
    }

    /// Encode as #RRGGBB hex string.
    pub fn to_hex(&self, baseline: Option<&HealthVector>) -> String {
        let hsv = self.to_hsv(baseline);
        let (r, g, b) = hsv_to_rgb(hsv.h, hsv.s, hsv.v);
        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }
}

/// HSV color triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsvColor {
    /// Hue [0, 360)
    pub h: f32,
    /// Saturation [0, 1]
    pub s: f32,
    /// Value [0, 1]  (0 = black, 1 = full brightness)
    pub v: f32,
}

/// Encode a health vector as HSV.
///
/// Value  = overall_score (0 = dead/black → 1 = optimal/white)
/// Hue    = weighted mean of dimension hue anchors; weight is
///          INVERSELY proportional to score so worst dimensions drive color.
/// Sat    = deviation from baseline (or 1.0 if no baseline known).
pub fn health_to_hsv(hv: &HealthVector, baseline: Option<&HealthVector>) -> HsvColor {
    let v = hv.overall_score();

    // Hue: compute weighted circular mean of hue anchors.
    // Worst dimensions (lowest score) get highest weight — they tell you
    // what is wrong, which is the informative part of the color.
    let vals = hv.as_array();
    let mut sin_sum = 0.0f32;
    let mut cos_sum = 0.0f32;
    let mut total_w = 0.0f32;
    for (i, &score) in vals.iter().enumerate() {
        let dim = HealthDimension::ALL[i];
        let angle = dim.hue_anchor().to_radians();
        // Weight: inverted score, modulated by certainty.
        let certainty = 1.0 - hv.uncertainty[i];
        let w = (1.0 - score) * certainty + 0.01; // small floor to avoid zero weight
        sin_sum += w * angle.sin();
        cos_sum += w * angle.cos();
        total_w += w;
    }
    let h = if total_w > 1e-6 {
        let mean_angle = (sin_sum / total_w).atan2(cos_sum / total_w);
        let deg = mean_angle.to_degrees();
        if deg < 0.0 { deg + 360.0 } else { deg }
    } else {
        120.0 // default green if no information
    };

    // Saturation: how far is this from the established baseline?
    let s = if let Some(bl) = baseline {
        let bl_vals = bl.as_array();
        let mut dev_sum = 0.0f32;
        for i in 0..6 {
            let certainty = 1.0 - hv.uncertainty[i];
            dev_sum += (vals[i] - bl_vals[i]).abs() * certainty;
        }
        (dev_sum / 6.0 * 3.0).clamp(0.0, 1.0) // scale so ±33% avg deviation = full saturation
    } else {
        // No baseline yet — saturation represents uncertainty directly.
        let mean_uncertainty: f32 = hv.uncertainty.iter().sum::<f32>() / 6.0;
        mean_uncertainty * 0.5 // moderate saturation while learning
    };

    HsvColor { h, s, v }
}

/// Convert HSV (h in degrees) to RGB u8.
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    if s < 1e-6 {
        let c = (v * 255.0).round() as u8;
        return (c, c, c);
    }
    let h = h % 360.0;
    let i = (h / 60.0).floor() as u32;
    let f = h / 60.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    ((r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8)
}

// ─── Scoped health record ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedHealth {
    pub entity_id: String,
    pub scope: Scope,
    pub health: HealthVector,
    /// Hex color derived from health vector and entity baseline.
    pub color_hex: String,
    /// Plain-text interpretation ("optimal", "stressed", "critical", "dead").
    pub label: String,
}

impl ScopedHealth {
    pub fn new(entity_id: String, scope: Scope, health: HealthVector, baseline: Option<&HealthVector>) -> Self {
        let color_hex = health.to_hex(baseline);
        let label = health_label(health.overall_score());
        Self { entity_id, scope, health, color_hex, label }
    }
}

fn health_label(score: f32) -> String {
    match score {
        s if s >= 0.85 => "optimal".to_string(),
        s if s >= 0.65 => "healthy".to_string(),
        s if s >= 0.45 => "stressed".to_string(),
        s if s >= 0.25 => "degraded".to_string(),
        s if s >= 0.05 => "critical".to_string(),
        _              => "dead".to_string(),
    }
}

// ─── Baseline tracker ─────────────────────────────────────────────────────────

const BASELINE_WINDOW: usize = 256;

/// Rolling baseline per entity type.  The system starts naïve (all dimensions
/// at 0.5 with full uncertainty) and converges toward the true healthy baseline
/// as it accumulates observations of functioning entities.
#[derive(Debug, Clone)]
pub struct HealthBaseline {
    entity_type: String,
    /// Ring buffer of recent health observations while entity was "functioning".
    window: VecDeque<HealthVector>,
    /// Current running mean — the working baseline.
    pub mean: HealthVector,
    /// Number of observations ingested.
    pub observations: u64,
}

impl HealthBaseline {
    pub fn new(entity_type: impl Into<String>) -> Self {
        Self {
            entity_type: entity_type.into(),
            window: VecDeque::with_capacity(BASELINE_WINDOW),
            mean: HealthVector::default(),
            observations: 0,
        }
    }

    /// Ingest an observation.  Only non-critical observations count as
    /// "baseline" (we don't want the mean to drift toward sickness).
    pub fn observe(&mut self, hv: &HealthVector) {
        if hv.overall_score() < 0.4 {
            return; // degraded state — don't pollute baseline
        }
        if self.window.len() >= BASELINE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(hv.clone());
        self.observations += 1;
        self.recalculate_mean();
    }

    fn recalculate_mean(&mut self) {
        if self.window.is_empty() { return; }
        let n = self.window.len() as f32;
        let mut sums = [0.0f32; 6];
        let mut unc_sums = [0.0f32; 6];
        for hv in &self.window {
            let a = hv.as_array();
            for i in 0..6 { sums[i] += a[i]; unc_sums[i] += hv.uncertainty[i]; }
        }
        self.mean.si = sums[0] / n;
        self.mean.ef = sums[1] / n;
        self.mean.rc = sums[2] / n;
        self.mean.fo = sums[3] / n;
        self.mean.ar = sums[4] / n;
        self.mean.tc = sums[5] / n;
        for i in 0..6 { self.mean.uncertainty[i] = unc_sums[i] / n; }
    }

    pub fn entity_type(&self) -> &str { &self.entity_type }
    pub fn is_established(&self) -> bool { self.observations >= 10 }
}

// ─── Baseline registry ────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct BaselineRegistry {
    baselines: HashMap<String, HealthBaseline>,
}

impl BaselineRegistry {
    pub fn observe(&mut self, entity_type: &str, hv: &HealthVector) {
        self.baselines
            .entry(entity_type.to_string())
            .or_insert_with(|| HealthBaseline::new(entity_type))
            .observe(hv);
    }

    pub fn get(&self, entity_type: &str) -> Option<&HealthBaseline> {
        self.baselines.get(entity_type)
    }

    pub fn get_mean(&self, entity_type: &str) -> Option<&HealthVector> {
        self.baselines.get(entity_type).map(|b| &b.mean)
    }
}

// ─── Entity health state ──────────────────────────────────────────────────────

/// Full health state for one entity: one vector per scope, plus type tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHealthState {
    pub entity_id: String,
    pub entity_type: String,
    /// Health vectors keyed by scope label.
    pub scopes: HashMap<String, ScopedHealth>,
    /// Organism-level overall score (0 = dead, 1 = optimal).
    pub organism_score: f32,
    /// Overall color hex at organism scope.
    pub organism_color: String,
}

impl EntityHealthState {
    pub fn new(entity_id: impl Into<String>, entity_type: impl Into<String>) -> Self {
        let eid = entity_id.into();
        let etype = entity_type.into();
        let hv = HealthVector::default();
        let color = hv.to_hex(None);
        Self {
            entity_id: eid.clone(),
            entity_type: etype,
            scopes: {
                let mut m = HashMap::new();
                let sh = ScopedHealth::new(eid, Scope::Organism, hv, None);
                m.insert(Scope::Organism.label(), sh);
                m
            },
            organism_score: 0.5,
            organism_color: color,
        }
    }

    pub fn update_scope(&mut self, scope: Scope, hv: HealthVector, baseline: Option<&HealthVector>) {
        let label = scope.label();
        let sh = ScopedHealth::new(self.entity_id.clone(), scope, hv, baseline);
        if sh.scope == Scope::Organism {
            self.organism_score = sh.health.overall_score();
            self.organism_color = sh.color_hex.clone();
        }
        self.scopes.insert(label, sh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_health_is_white() {
        let hv = HealthVector::optimal();
        let hsv = hv.to_hsv(None);
        assert!(hsv.v > 0.95, "value should be near 1.0 for optimal health");
        let hex = hv.to_hex(None);
        // White or near-white
        assert!(hex.starts_with('#'));
    }

    #[test]
    fn dead_health_is_black() {
        let hv = HealthVector::dead();
        let hsv = hv.to_hsv(None);
        assert!(hsv.v < 0.05, "value should be near 0 for dead entity");
        let hex = hv.to_hex(None);
        assert_eq!(hex, "#000000");
    }

    #[test]
    fn si_damage_shifts_hue_toward_red() {
        let mut hv = HealthVector::optimal();
        hv.uncertainty = [0.0; 6];
        hv.si = 0.1; // severe structural damage
        let hsv = hv.to_hsv(None);
        // Hue should be in red territory (< 60° or > 300°)
        assert!(hsv.h < 80.0 || hsv.h > 300.0, "SI damage should shift toward red, got {}", hsv.h);
    }

    #[test]
    fn baseline_tracker_converges() {
        let mut bl = HealthBaseline::new("human");
        let hv = HealthVector::optimal();
        for _ in 0..20 { bl.observe(&hv); }
        assert!(bl.is_established());
        assert!(bl.mean.si > 0.9);
    }
}
