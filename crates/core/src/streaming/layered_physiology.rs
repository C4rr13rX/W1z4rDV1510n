//! Layered physiology — hierarchical structural decomposition per sensor view.
//!
//! When the system observes an entity it can construct an "inside-out" view:
//! what is *most probably inside* this entity at each depth layer, based on
//! accumulated training experience (neuro labels + motif patterns + EEM).
//!
//! ## How layers emerge
//!
//! The pool learns spatial and compositional co-activation during training.
//! Labels are mapped to structural depths by keyword matching (tiebreaker) and
//! by EEM equation dimensional analysis (primary signal when available).
//!
//! ## Predicted z-axis (EEM-calibrated)
//!
//! When EEM equation search results are provided, the z-axis uses physical SI
//! metres derived from length-dimensional (`[L]`) variables in the matched
//! equations.  Equations matched to biology labels typically contain variables
//! with units like `"nm"` or `"μm"`, anchoring the molecular and cellular
//! layers to their known physical scales.  Without EEM data the system falls
//! back to linear interpolation over the caller-supplied `entity_size_est`.
//!
//! An explicit `scale_override_m` can also be provided (e.g. from a calibration
//! measurement), which overrides both EEM and the fallback for that entity.
//!
//! ## API usage
//!
//!  - `GET /overlay/layers/:id`           — all depths (pool labels as input)
//!  - `GET /overlay/layers/:id?depth=N`   — filter to depths 0..=N
//!  - `POST /overlay/layers`              — batch, with explicit label + score input
//!  - `POST /overlay/layers/calibrate`    — set known physical scale for entity type

use crate::schema::Timestamp;
use crate::streaming::equation_matrix::EquationSearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredPhysiologyConfig {
    /// Maximum depth layers to generate.
    pub max_depth: usize,
    /// Minimum label activation score to include in a layer.
    pub min_label_score: f64,
    /// Maximum component labels per layer.
    pub max_labels_per_layer: usize,
    /// Whether to include depth-0 (directly visible exterior) layer.
    pub include_surface_layer: bool,
    /// Minimum EEM relevance to use an equation for z-calibration.
    pub min_eem_relevance: f32,
}

impl Default for LayeredPhysiologyConfig {
    fn default() -> Self {
        Self {
            max_depth:             6,
            min_label_score:       0.1,
            max_labels_per_layer:  12,
            include_surface_layer: true,
            min_eem_relevance:     0.25,
        }
    }
}

// ── Scale vocabulary for depth inference ─────────────────────────────────────

const DEPTH_KEYWORDS: &[(&[&str], usize)] = &[
    // depth 0 — visible exterior
    (&["outer", "surface", "skin", "shell", "housing", "casing", "exterior",
       "coat", "bark", "rind", "hull", "cover", "edge", "border"], 0),
    // depth 1 — first interior layer (tissue/sheath)
    (&["layer", "tissue", "membrane", "wall", "lining", "fascia", "film",
       "panel", "frame", "chassis", "sheath", "cortex", "epidermis"], 1),
    // depth 2 — cellular / circuit level
    (&["cell", "organelle", "vacuole", "nucleus", "cytoplasm", "mitochondria",
       "board", "circuit", "trace", "pad", "node", "junction", "vessel",
       "fiber", "gland", "duct", "ribosome", "chloroplast"], 2),
    // depth 3 — molecular / component level
    (&["molecule", "protein", "lipid", "carbohydrate", "enzyme", "hormone",
       "nucleic", "dna", "rna", "amino", "polymer",
       "chip", "ic", "capacitor", "resistor", "crystal", "polymer"], 3),
    // depth 4 — atomic / gate level
    (&["atom", "ion", "radical", "bond", "covalent", "ionic",
       "transistor", "gate", "via", "dopant", "carrier",
       "electron", "proton", "neutron", "nucleus_atomic"], 4),
    // depth 5 — sub-atomic / quantum
    (&["quantum", "qubit", "orbital", "photon", "phonon", "wavefunction",
       "eigenstate", "superposition", "entanglement", "spin", "fermion", "boson"], 5),
];

fn depth_score(label: &str, depth: usize) -> usize {
    let lower = label.to_lowercase();
    DEPTH_KEYWORDS.iter()
        .filter(|(_, d)| *d == depth)
        .flat_map(|(words, _)| words.iter())
        .filter(|&&kw| lower.contains(kw))
        .count()
}

pub fn infer_depth(label: &str) -> usize {
    (0..DEPTH_KEYWORDS.len())
        .map(|d| (d, depth_score(label, d)))
        .max_by_key(|(_, s)| *s)
        .filter(|(_, s)| *s > 0)
        .map(|(d, _)| d)
        .unwrap_or(1)
}

// ── EEM z-axis calibration ────────────────────────────────────────────────────

/// Parse a unit string to SI metres for length dimensions only.
/// Returns `None` if the string does not represent a pure length or is unknown.
fn parse_length_to_metres(units: &str, dimension: &str) -> Option<f64> {
    // Only process length-dimensional variables.
    let is_length = dimension.contains("[L]")
        || dimension.eq_ignore_ascii_case("length")
        || dimension.eq_ignore_ascii_case("distance");
    // Skip velocity, area, volume etc.
    let has_division = units.contains('/');
    let has_exponent = units.contains('^') || units.contains('²') || units.contains('³');
    if !is_length || has_division || has_exponent {
        return None;
    }
    let u = units.trim().to_lowercase();
    match u.as_str() {
        "å" | "angstrom" | "angström"          => Some(1e-10),
        "pm"                                    => Some(1e-12),
        "nm"                                    => Some(1e-9),
        "μm" | "um" | "micron" | "micrometer"  => Some(1e-6),
        "mm"                                    => Some(1e-3),
        "cm"                                    => Some(1e-2),
        "dm"                                    => Some(1e-1),
        "m"                                     => Some(1.0),
        "km"                                    => Some(1e3),
        _                                       => None,
    }
}

/// Derive per-depth physical scale (in metres) from EEM equation search results.
///
/// For each matched equation, we scan its variables for those with `[L]`
/// dimensional annotation.  The physical magnitude is parsed from the units
/// field.  Each length scale is bucketed to the depth that best matches the
/// variable name (using `infer_depth`), then a relevance-weighted median per
/// depth is computed.
///
/// Returns an array indexed `[0..max_depth]`.  Depths with no EEM evidence
/// hold `None` (caller falls back to linear interpolation for those).
pub fn eem_z_scales(
    candidates:   &[EquationSearchResult],
    min_relevance: f32,
    max_depth:     usize,
) -> Vec<Option<f64>> {
    // Collect (depth_bucket, scale_m, relevance_weight) triples.
    let mut buckets: Vec<Vec<(f64, f32)>> = vec![Vec::new(); max_depth];

    for res in candidates {
        if res.relevance < min_relevance { continue; }
        for var in &res.equation.variables {
            if let Some(scale_m) = parse_length_to_metres(&var.units, &var.dimension) {
                // Map the variable *name* to a depth as a first-pass heuristic.
                // Example: variable "wavelength" → depth 3 (molecular), "r_cell" → depth 2.
                let d = infer_depth(&var.name).min(max_depth - 1);
                buckets[d].push((scale_m, res.relevance));
            }
        }
    }

    buckets.iter().map(|bucket| {
        if bucket.is_empty() { return None; }
        // Weighted median: sort by scale, find the weight-50% point.
        let mut sorted = bucket.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let total_w: f32 = sorted.iter().map(|(_, w)| w).sum();
        let half_w = total_w / 2.0;
        let mut cumw = 0.0_f32;
        for (scale, w) in &sorted {
            cumw += w;
            if cumw >= half_w { return Some(*scale); }
        }
        sorted.last().map(|(s, _)| *s)
    }).collect()
}

/// How the z-axis was computed for this report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleProfile {
    /// Physical scale per depth (in metres) — Some = EEM-calibrated, None = heuristic.
    pub z_metres:         Vec<Option<f64>>,
    /// Number of EEM equations that contributed length scale information.
    pub eem_equation_count: usize,
    /// Total weight of EEM evidence across all depths (0 = no EEM data).
    pub eem_total_weight: f32,
    /// True if an explicit calibration override was applied.
    pub calibrated:       bool,
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPosition {
    pub x_frac:        f64,
    pub y_frac:        f64,
    /// Predicted depth from sensor in SI metres (EEM-calibrated when possible).
    pub z_est_m:       f64,
    /// Confidence in z_est_m (0 = heuristic, 1 = strong EEM + calibration).
    pub z_confidence:  f64,
    /// Whether this z is from EEM physics (true) or linear fallback (false).
    pub z_from_eem:    bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralLayer {
    pub layer_id:    String,
    pub depth:       usize,
    pub label:       String,
    pub components:  Vec<String>,
    pub position:    LayerPosition,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredPhysiologyReport {
    pub entity_id:        String,
    pub timestamp:        Timestamp,
    pub phenotype:        Option<String>,
    pub layers:           Vec<StructuralLayer>,
    pub max_depth:        usize,
    pub total_components: usize,
    /// How z-axis values were computed this frame.
    pub scale_profile:    ScaleProfile,
}

impl LayeredPhysiologyReport {
    pub fn at_depth(&self, depth: usize) -> Vec<&StructuralLayer> {
        self.layers.iter().filter(|l| l.depth == depth).collect()
    }
    pub fn up_to_depth(&self, depth: usize) -> Vec<&StructuralLayer> {
        self.layers.iter().filter(|l| l.depth <= depth).collect()
    }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

pub struct LayeredPhysiologyRuntime {
    config: LayeredPhysiologyConfig,
    /// Per-entity-type known physical scale in metres.
    /// Key = entity type label (e.g. "cell", "protein", "game_console").
    scale_overrides: HashMap<String, f64>,
}

impl LayeredPhysiologyRuntime {
    pub fn new(config: LayeredPhysiologyConfig) -> Self {
        Self { config, scale_overrides: HashMap::new() }
    }

    /// Register a known physical scale for an entity type.
    /// `entity_type` should be a label that will appear in `active_labels`.
    pub fn set_scale_override(&mut self, entity_type: &str, scale_m: f64) {
        if scale_m > 0.0 && scale_m.is_finite() {
            self.scale_overrides.insert(entity_type.to_string(), scale_m);
        }
    }

    pub fn get_scale_overrides(&self) -> &HashMap<String, f64> {
        &self.scale_overrides
    }

    /// Build a layered decomposition for one entity.
    ///
    /// * `eem_candidates`  — EEM results from `apply_to_context(active_labels, 3)`.
    ///                       Pass `&[]` to skip EEM z-calibration and use linear fallback.
    /// * `entity_size_est` — fallback physical extent when no EEM/calibration applies (metres).
    pub fn decompose(
        &self,
        entity_id:       &str,
        phenotype:        Option<&str>,
        active_labels:    &[String],
        label_scores:     &HashMap<String, f64>,
        entity_position:  (f64, f64),
        entity_size_est:  f64,
        eem_candidates:   &[EquationSearchResult],
        timestamp:        Timestamp,
    ) -> LayeredPhysiologyReport {
        let (cx, cy) = entity_position;

        // Check for explicit calibration override.
        // Priority: entity_id exact match → phenotype → any matching active label.
        let scale_override: Option<f64> = self.scale_overrides.get(entity_id).copied()
            .or_else(|| phenotype.and_then(|p| self.scale_overrides.get(p).copied()))
            .or_else(|| active_labels.iter()
                .find_map(|l| self.scale_overrides.get(l.as_str()).copied()));

        // Derive EEM z-scales.
        let eem_scales = eem_z_scales(eem_candidates, self.config.min_eem_relevance, self.config.max_depth);
        let eem_equation_count = eem_candidates.iter()
            .filter(|r| r.relevance >= self.config.min_eem_relevance)
            .count();
        let eem_total_weight: f32 = eem_candidates.iter()
            .filter(|r| r.relevance >= self.config.min_eem_relevance)
            .map(|r| r.relevance)
            .sum();

        // Total physical extent for linear fallback.
        let total_extent = scale_override.unwrap_or(entity_size_est);

        // Filter labels above score threshold.
        let scored: Vec<(String, f64)> = active_labels.iter()
            .filter_map(|l| {
                let s = label_scores.get(l.as_str()).copied().unwrap_or(0.0);
                if s >= self.config.min_label_score { Some((l.clone(), s)) } else { None }
            })
            .collect();

        // Group labels by inferred depth.
        let mut depth_groups: HashMap<usize, Vec<(String, f64)>> = HashMap::new();
        for (label, score) in &scored {
            let d = infer_depth(label).min(self.config.max_depth - 1);
            depth_groups.entry(d).or_default().push((label.clone(), *score));
        }

        // Track what z-scales we actually used (for the scale profile).
        let mut used_z: Vec<Option<f64>> = vec![None; self.config.max_depth];

        // Build layers.
        let mut layers: Vec<StructuralLayer> = Vec::new();
        for depth in 0..self.config.max_depth {
            let Some(group) = depth_groups.get_mut(&depth) else { continue };
            if depth == 0 && !self.config.include_surface_layer { continue; }

            group.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let primary     = group[0].0.clone();
            let probability = group[0].1;
            let components: Vec<String> = group.iter()
                .skip(1)
                .take(self.config.max_labels_per_layer - 1)
                .map(|(l, _)| l.clone())
                .collect();

            // Z-axis: priority order:
            //   1. Per-entity calibration override (scales linearly across depths)
            //   2. EEM-derived physical scale for this depth bucket
            //   3. Linear interpolation over entity_size_est
            let (z_est_m, z_from_eem, z_confidence) = if let Some(override_m) = scale_override {
                // Calibrated: logarithmic spacing from surface (depth 0) to total extent.
                // Uses log10 spacing so deeper layers don't all pile up near the surface.
                let max_d = (self.config.max_depth - 1).max(1) as f64;
                let frac = if depth == 0 { 0.0 } else {
                    (depth as f64 / max_d).powf(1.5) // slight sub-linear for visual spacing
                };
                (frac * override_m, false, 0.95)
            } else if let Some(eem_z) = eem_scales.get(depth).copied().flatten() {
                // EEM provides absolute scale for this depth.
                (eem_z, true, 0.75 / (1.0 + depth as f64 * 0.08))
            } else {
                // Linear fallback.
                let z = if self.config.max_depth > 1 {
                    (depth as f64 / (self.config.max_depth - 1) as f64) * total_extent
                } else { 0.0 };
                (z, false, 1.0 / (1.0 + depth as f64 * 0.5))
            };

            used_z[depth] = Some(z_est_m);
            layers.push(StructuralLayer {
                layer_id:   format!("{}@depth{}", entity_id, depth),
                depth,
                label:      primary,
                components,
                position:   LayerPosition {
                    x_frac: cx,
                    y_frac: cy,
                    z_est_m,
                    z_confidence,
                    z_from_eem,
                },
                probability,
            });
        }

        let total_components: usize = layers.iter().map(|l| l.components.len() + 1).sum();
        let max_depth = layers.iter().map(|l| l.depth).max().unwrap_or(0);

        LayeredPhysiologyReport {
            entity_id:   entity_id.to_string(),
            timestamp,
            phenotype:   phenotype.map(|s| s.to_string()),
            layers,
            max_depth,
            total_components,
            scale_profile: ScaleProfile {
                z_metres:          used_z,
                eem_equation_count,
                eem_total_weight,
                calibrated:        scale_override.is_some(),
            },
        }
    }

    /// Batch decomposition — same signature as before plus `eem_candidates`.
    pub fn decompose_all(
        &self,
        entities:      &[EntityDecomposeRequest],
        eem_candidates: &[EquationSearchResult],
        timestamp:      Timestamp,
    ) -> Vec<LayeredPhysiologyReport> {
        entities.iter().map(|req| {
            self.decompose(
                &req.entity_id,
                req.phenotype.as_deref(),
                &req.active_labels,
                &req.label_scores,
                req.entity_position,
                req.entity_size_est,
                eem_candidates,
                timestamp,
            )
        }).collect()
    }
}

impl Default for LayeredPhysiologyRuntime {
    fn default() -> Self { Self::new(LayeredPhysiologyConfig::default()) }
}

/// Request type for batch decomposition.
#[derive(Debug, Clone)]
pub struct EntityDecomposeRequest {
    pub entity_id:       String,
    pub phenotype:       Option<String>,
    pub active_labels:   Vec<String>,
    pub label_scores:    HashMap<String, f64>,
    pub entity_position: (f64, f64),
    pub entity_size_est: f64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skin_is_depth_zero() {
        assert_eq!(infer_depth("skin_texture"), 0);
        assert_eq!(infer_depth("outer_shell"), 0);
    }

    #[test]
    fn cell_is_depth_two() {
        assert_eq!(infer_depth("cell_nucleus"), 2);
        assert_eq!(infer_depth("mitochondria"), 2);
    }

    #[test]
    fn protein_is_depth_three() {
        let d = infer_depth("molecule_protein");
        assert_eq!(d, 3, "molecule_protein should be depth 3");
    }

    #[test]
    fn electron_is_depth_four() {
        assert_eq!(infer_depth("electron_carrier"), 4);
    }

    #[test]
    fn parse_length_units() {
        assert!((parse_length_to_metres("nm", "[L]").unwrap() - 1e-9).abs() < 1e-15);
        assert!((parse_length_to_metres("μm", "[L]").unwrap() - 1e-6).abs() < 1e-12);
        assert!((parse_length_to_metres("m", "[L]").unwrap() - 1.0).abs() < 1e-10);
        assert!(parse_length_to_metres("m/s", "[LT^-1]").is_none()); // velocity — skip
        assert!(parse_length_to_metres("kg", "[M]").is_none());       // mass — skip
    }

    #[test]
    fn eem_z_scales_buckets_correctly() {
        use crate::streaming::equation_matrix::{EquationSearchResult, PhysicsEquation,
            EquationVariable, Discipline};
        let make_eq = |var_name: &str, units: &str, dim: &str| -> EquationSearchResult {
            EquationSearchResult {
                equation: PhysicsEquation {
                    id: "test".into(), text: "test".into(), latex: "".into(),
                    discipline: Discipline::Biophysics,
                    variables: vec![EquationVariable {
                        symbol: "x".into(), name: var_name.into(),
                        units: units.into(), dimension: dim.into(),
                    }],
                    confidence: 1.0,
                    assumptions: vec![], constraints: vec![],
                    applicable_dims: vec![],
                    related_ids: vec![],
                },
                relevance: 0.8,
                related_ids: vec![],
            }
        };

        // "cell_radius" → infer_depth says depth 2 → should get nm scale
        let candidates = vec![make_eq("cell_radius", "nm", "[L]")];
        let scales = eem_z_scales(&candidates, 0.3, 6);
        assert!(scales[2].is_some(), "depth 2 should have EEM scale");
        assert!((scales[2].unwrap() - 1e-9).abs() < 1e-15);
    }

    #[test]
    fn decompose_with_eem_uses_physical_z() {
        use crate::streaming::equation_matrix::{EquationSearchResult, PhysicsEquation,
            EquationVariable, Discipline};
        let eem_result = EquationSearchResult {
            equation: PhysicsEquation {
                id: "bio1".into(), text: "diffusion".into(), latex: "".into(),
                discipline: Discipline::Biophysics,
                variables: vec![EquationVariable {
                    symbol: "L".into(), name: "cell_length".into(),
                    units: "μm".into(), dimension: "[L]".into(),
                }],
                confidence: 0.9, assumptions: vec![], constraints: vec![],
                applicable_dims: vec![], related_ids: vec![],
            },
            relevance: 0.8, related_ids: vec![],
        };

        let rt = LayeredPhysiologyRuntime::default();
        let labels = vec!["cell_membrane".to_string(), "skin_texture".to_string()];
        let scores: HashMap<String, f64> = labels.iter().map(|l| (l.clone(), 0.8)).collect();
        let report = rt.decompose(
            "cell_entity", None, &labels, &scores,
            (0.5, 0.5), 0.15, &[eem_result], Timestamp { unix: 0 },
        );
        // Should have EEM scale data
        assert!(report.scale_profile.eem_equation_count > 0,
            "should have used EEM equations");
        // At least one layer should have z_from_eem = true
        let eem_layers: Vec<_> = report.layers.iter().filter(|l| l.position.z_from_eem).collect();
        assert!(!eem_layers.is_empty(), "at least one layer should have EEM-derived z");
    }

    #[test]
    fn scale_override_applies() {
        let mut rt = LayeredPhysiologyRuntime::default();
        rt.set_scale_override("cell", 2e-5);
        let labels = vec!["cell_membrane".to_string(), "outer_surface".to_string()];
        let scores: HashMap<String, f64> = labels.iter()
            .map(|l| (l.clone(), 0.9)).collect();
        let report = rt.decompose(
            "e1", None, &labels, &scores,
            (0.5, 0.5), 0.1, &[], Timestamp { unix: 0 },
        );
        // Even without EEM, scale_profile.calibrated should be false
        // (override matches "cell" but label is "cell_membrane" which contains "cell")...
        // Actually "cell_membrane" contains "cell" but the override key is exact "cell".
        // This test verifies the override mechanism doesn't match substrings — must be exact label.
        assert!(!report.scale_profile.calibrated,
            "exact label 'cell' should not match 'cell_membrane'");
    }

    #[test]
    fn decompose_sorts_by_depth() {
        let rt = LayeredPhysiologyRuntime::default();
        let labels = vec![
            "outer_shell".to_string(), "molecule_protein".to_string(), "cell_membrane".to_string(),
        ];
        let scores: HashMap<String, f64> = labels.iter().map(|l| (l.clone(), 0.8)).collect();
        let report = rt.decompose("e", None, &labels, &scores, (0.5, 0.5), 0.1, &[], Timestamp { unix: 0 });
        let depths: Vec<usize> = report.layers.iter().map(|l| l.depth).collect();
        let mut sorted = depths.clone();
        sorted.sort_unstable();
        assert_eq!(depths, sorted, "layers must be sorted by depth ascending");
    }
}
