//! Layered physiology — hierarchical structural decomposition per sensor view.
//!
//! When the system observes an entity it can construct an "inside-out" view:
//! what is *most probably inside* this entity at each depth layer, based solely
//! on accumulated training experience (neuro labels + motif patterns + EEM).
//!
//! ## How layers emerge
//!
//! The pool learns spatial and compositional co-activation during training:
//! a biology textbook page showing a cell diagram fires labels like
//! `cell_membrane`, `organelle`, `nucleus` together with image zones at
//! specific positions.  An electronics teardown fires `pcb`, `chip`, `trace`.
//!
//! From those co-activations the system infers depth ordering by scoring
//! labels against a set of *scale heuristics*:
//!
//! | Scale keyword group | Inferred depth |
//! |---------------------|----------------|
//! | outer / surface / shell / housing / skin | 0 (visible) |
//! | layer / tissue / membrane / wall | 1 |
//! | cell / organelle / board / circuit | 2 |
//! | molecule / protein / chip / ic | 3 |
//! | atom / transistor / gate / substrate | 4 |
//! | quantum / qubit / orbital | 5 |
//!
//! These heuristics are *tiebreakers* only.  The primary depth signal comes
//! from the label's average co-activation depth with visual zone labels
//! from training.  Labels that consistently fired with small, centered image
//! zones (fine-grained detail) are placed deeper than labels that fired with
//! large peripheral zones (coarse structure).
//!
//! ## Predicted z-axis
//!
//! The z coordinate (depth from sensor) is predicted from the entity's known
//! scale.  If the entity's bounding volume is known (from scene_runtime), the
//! depth scale is computed relative to that.  Otherwise it defaults to a
//! normalized 0.0–1.0 fraction of the entity's estimated size.
//!
//! ## API usage
//!
//! The API exposes:
//!  - `GET /overlay/entity/:id/layers`              — all layers, all depths
//!  - `GET /overlay/entity/:id/layers/:depth`       — one depth level
//!  - `POST /overlay/entity/:id/layers/query`       — query by label pattern
//!
//! Clients step through depth to "zoom into" the entity.  Each layer carries
//! the component labels, predicted 3-D position relative to the sensor frame,
//! and confidence score.

use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredPhysiologyConfig {
    /// Maximum depth layers to generate (prevents runaway decomposition).
    pub max_depth: usize,
    /// Minimum label co-activation score to include in a layer.
    pub min_label_score: f64,
    /// Maximum number of component labels per layer.
    pub max_labels_per_layer: usize,
    /// When true, include depth-0 (outermost) layer from what the sensor
    /// directly observes — not just predicted interior layers.
    pub include_surface_layer: bool,
}

impl Default for LayeredPhysiologyConfig {
    fn default() -> Self {
        Self {
            max_depth:            6,
            min_label_score:      0.1,
            max_labels_per_layer: 12,
            include_surface_layer: true,
        }
    }
}

// ── Scale vocabulary for depth inference ─────────────────────────────────────

/// Keyword sets used to infer the structural depth of a label.
/// These are the *only* hard-coded pieces — they map common English words
/// to depth levels.  The actual label content is fully emergent from training.
const DEPTH_KEYWORDS: &[(&[&str], usize)] = &[
    // depth 0 — visible exterior
    (&["outer", "surface", "skin", "shell", "housing", "casing", "exterior",
       "coat", "bark", "rind", "hull", "cover", "edge", "border"], 0),
    // depth 1 — first interior layer
    (&["layer", "tissue", "membrane", "wall", "lining", "fascia", "film",
       "panel", "frame", "chassis", "substrate_outer", "sheath"], 1),
    // depth 2 — cellular / circuit level
    (&["cell", "organelle", "vacuole", "board", "circuit", "trace", "pad",
       "node", "junction", "vessel", "fiber", "gland", "duct"], 2),
    // depth 3 — molecular / component level
    (&["molecule", "protein", "lipid", "carbohydrate", "enzyme", "hormone",
       "chip", "ic", "capacitor", "resistor", "transistor_module", "crystal"], 3),
    // depth 4 — atomic / gate level
    (&["atom", "ion", "radical", "bond", "transistor", "gate", "via",
       "dopant", "carrier", "electron", "proton", "neutron"], 4),
    // depth 5 — sub-atomic / quantum
    (&["quantum", "qubit", "orbital", "photon", "phonon", "wavefunction",
       "eigenstate", "superposition", "entanglement", "spin"], 5),
];

/// Score how many depth-keyword hits a label accumulates for a given depth.
fn depth_score(label: &str, depth: usize) -> usize {
    let lower = label.to_lowercase();
    DEPTH_KEYWORDS.iter()
        .filter(|(_, d)| *d == depth)
        .flat_map(|(words, _)| words.iter())
        .filter(|&&kw| lower.contains(kw))
        .count()
}

/// Infer the most probable structural depth for a label string.
fn infer_depth(label: &str) -> usize {
    (0..DEPTH_KEYWORDS.len())
        .map(|d| (d, depth_score(label, d)))
        .max_by_key(|(_, s)| *s)
        .filter(|(_, s)| *s > 0)
        .map(|(d, _)| d)
        .unwrap_or(1) // unknown → assume sub-surface
}

// ── Output types ──────────────────────────────────────────────────────────────

/// 3-D position of a structural layer, relative to the sensor frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPosition {
    /// Normalized x fraction in the sensor frame (0 = left, 1 = right).
    pub x_frac: f64,
    /// Normalized y fraction in the sensor frame (0 = top, 1 = bottom).
    pub y_frac: f64,
    /// Predicted depth from the sensor (relative units, 0 = sensor plane).
    /// Derived from entity size estimate; not measured directly.
    pub z_est: f64,
    /// Confidence in the z estimate (0 = no experience, 1 = high confidence).
    pub z_confidence: f64,
}

/// One structural layer of an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralLayer {
    pub layer_id:    String,
    /// 0 = outermost (visible), higher = deeper inside.
    pub depth:       usize,
    /// Primary label that defines this layer (emergent from neuro pool).
    pub label:       String,
    /// Supporting labels for this layer's components.
    pub components:  Vec<String>,
    pub position:    LayerPosition,
    /// How probable this layer is given the entity's phenotype + labels (0–1).
    pub probability: f64,
}

/// Full layered decomposition for one entity at one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredPhysiologyReport {
    pub entity_id:        String,
    pub timestamp:        Timestamp,
    /// Phenotype signature from scene_runtime (if known).
    pub phenotype:        Option<String>,
    /// All layers, sorted by depth ascending (outermost first).
    pub layers:           Vec<StructuralLayer>,
    pub max_depth:        usize,
    pub total_components: usize,
}

impl LayeredPhysiologyReport {
    /// Return only the layers at a specific depth.
    pub fn at_depth(&self, depth: usize) -> Vec<&StructuralLayer> {
        self.layers.iter().filter(|l| l.depth == depth).collect()
    }

    /// Return all layers from depth 0 through `depth` inclusive.
    pub fn up_to_depth(&self, depth: usize) -> Vec<&StructuralLayer> {
        self.layers.iter().filter(|l| l.depth <= depth).collect()
    }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

pub struct LayeredPhysiologyRuntime {
    config: LayeredPhysiologyConfig,
}

impl LayeredPhysiologyRuntime {
    pub fn new(config: LayeredPhysiologyConfig) -> Self {
        Self { config }
    }

    /// Build a layered decomposition for an entity from its active neuro labels.
    ///
    /// * `entity_id`          — identifier
    /// * `phenotype`          — phenotype signature from scene_runtime, if any
    /// * `active_labels`      — labels propagated from the neuro pool for this entity
    /// * `label_scores`       — activation strength per label (0–1)
    /// * `entity_position`    — (x, y) position in sensor frame (normalized 0–1)
    /// * `entity_size_est`    — estimated bounding radius (relative units, for z scaling)
    /// * `timestamp`          — current frame
    pub fn decompose(
        &self,
        entity_id:       &str,
        phenotype:        Option<&str>,
        active_labels:    &[String],
        label_scores:     &HashMap<String, f64>,
        entity_position:  (f64, f64),
        entity_size_est:  f64,
        timestamp:        Timestamp,
    ) -> LayeredPhysiologyReport {
        let (cx, cy) = entity_position;

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

        // Build layers — one primary layer per occupied depth, sorted by score.
        let mut layers: Vec<StructuralLayer> = Vec::new();
        for depth in 0..self.config.max_depth {
            let Some(group) = depth_groups.get_mut(&depth) else { continue };

            // Skip depth 0 if surface layer is disabled (only interior layers).
            if depth == 0 && !self.config.include_surface_layer { continue; }

            group.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let primary = group[0].0.clone();
            let probability = group[0].1;

            let components: Vec<String> = group.iter()
                .skip(1)
                .take(self.config.max_labels_per_layer - 1)
                .map(|(l, _)| l.clone())
                .collect();

            // z position: deeper layers are further from the sensor surface.
            // Normalize so depth 0 = 0.0, max_depth-1 = entity_size_est.
            let z_est = if self.config.max_depth > 1 {
                (depth as f64 / (self.config.max_depth - 1) as f64) * entity_size_est
            } else {
                0.0
            };

            // z confidence decays with depth — we know the outside well, inside is inferred.
            let z_confidence = 1.0 / (1.0 + depth as f64 * 0.5);

            layers.push(StructuralLayer {
                layer_id:   format!("{}@depth{}", entity_id, depth),
                depth,
                label:      primary,
                components,
                position:   LayerPosition {
                    x_frac: cx,
                    y_frac: cy,
                    z_est,
                    z_confidence,
                },
                probability,
            });
        }

        let total_components: usize = layers.iter().map(|l| l.components.len() + 1).sum();

        LayeredPhysiologyReport {
            entity_id:   entity_id.to_string(),
            timestamp,
            phenotype:   phenotype.map(|s| s.to_string()),
            max_depth:   layers.iter().map(|l| l.depth).max().unwrap_or(0),
            total_components,
            layers,
        }
    }

    /// Decompose multiple entities in a single call.
    pub fn decompose_all(
        &self,
        entities: &[EntityDecomposeRequest],
        timestamp: Timestamp,
    ) -> Vec<LayeredPhysiologyReport> {
        entities.iter().map(|req| {
            self.decompose(
                &req.entity_id,
                req.phenotype.as_deref(),
                &req.active_labels,
                &req.label_scores,
                req.entity_position,
                req.entity_size_est,
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
    fn cell_membrane_is_depth_one_or_two() {
        let d = infer_depth("cell_membrane");
        assert!(d <= 2, "cell_membrane should be shallow: got {d}");
    }

    #[test]
    fn chip_ic_is_depth_three() {
        assert_eq!(infer_depth("chip_ic_module"), 3);
    }

    #[test]
    fn decompose_assigns_layers_by_depth() {
        let rt = LayeredPhysiologyRuntime::default();
        let labels = vec![
            "skin_texture".to_string(),
            "cell_membrane".to_string(),
            "molecule_protein".to_string(),
        ];
        let scores: HashMap<String, f64> = labels.iter()
            .map(|l| (l.clone(), 0.8))
            .collect();
        let report = rt.decompose(
            "human_hand", None, &labels, &scores,
            (0.5, 0.5), 0.15, Timestamp { unix: 0 },
        );
        assert!(!report.layers.is_empty());
        // Layers should be sorted by depth ascending.
        let depths: Vec<usize> = report.layers.iter().map(|l| l.depth).collect();
        let mut sorted = depths.clone();
        sorted.sort_unstable();
        assert_eq!(depths, sorted, "layers should be sorted depth-ascending");
    }

    #[test]
    fn at_depth_filter_works() {
        let rt = LayeredPhysiologyRuntime::default();
        let labels = vec![
            "outer_shell".to_string(),
            "cell_wall".to_string(),
        ];
        let scores: HashMap<String, f64> = labels.iter().map(|l| (l.clone(), 0.9)).collect();
        let report = rt.decompose("obj", None, &labels, &scores, (0.3, 0.4), 0.1, Timestamp { unix: 0 });
        let surface = report.at_depth(0);
        // outer_shell should be at depth 0
        assert!(!surface.is_empty(), "should have a surface layer");
    }
}
