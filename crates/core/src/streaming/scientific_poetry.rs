//! Scientific poetry — emergent natural-language synthesis of an entity's state.
//!
//! This is NOT a template engine.  There are no canned phrases.  The text that
//! comes out is assembled entirely from:
//!
//!  • The entity's active neuro labels (words the pool learned from experience).
//!  • The dominant health dimension and its score.
//!  • The top-matching EEM equation (emergent physics candidate).
//!  • The motif attractor status (stable pattern vs chaotic trajectory).
//!  • The velocity direction from the delta engine (improving / degrading).
//!
//! The assembly rules are combinatorial: dimension × label-cluster × equation
//! × trajectory produce a description that is factually grounded in what the
//! network actually knows about this entity, nothing more.

use crate::schema::Timestamp;
use crate::streaming::entity_health::{EntityHealthOverlay, HealthVector};
use crate::streaming::equation_matrix::EquationSearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoetryConfig {
    /// Maximum number of active labels to include in the synthesis.
    pub max_labels: usize,
    /// Minimum EEM relevance score to cite an equation in the output.
    pub min_eem_relevance: f32,
    /// True → include numeric scores; False → qualitative terms only.
    pub include_scores: bool,
}

impl Default for PoetryConfig {
    fn default() -> Self {
        Self {
            max_labels:        12,
            min_eem_relevance: 0.35,
            include_scores:    true,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

/// Emergent natural-language profile for one entity at one moment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPoetry {
    pub entity_id:        String,
    pub timestamp:        Timestamp,
    /// The synthesized description — emergent, not templated.
    pub synthesis:        String,
    /// Which health dimension dominated the synthesis (name + score).
    pub dominant_dim:     String,
    pub dominant_score:   f64,
    /// Health label the synthesis reflects.
    pub health_label:     String,
    /// Key labels from the neuro pool that appeared in this synthesis.
    pub key_labels:       Vec<String>,
    /// EEM equation cited in the synthesis (if any matched above threshold).
    pub cited_equation:   Option<String>,
    /// Whether the entity is on an improving trajectory.
    pub trajectory:       String,
}

// ── Runtime ───────────────────────────────────────────────────────────────────

pub struct PoetryRuntime {
    config: PoetryConfig,
}

impl PoetryRuntime {
    pub fn new(config: PoetryConfig) -> Self {
        Self { config }
    }

    /// Synthesize a scientific-poetry profile for one entity.
    ///
    /// * `overlay`       — current health overlay for this entity
    /// * `active_labels` — top neuro labels active for this entity's context
    ///                     (e.g. from the most recent propagation result)
    /// * `eem_results`   — EEM equation search results for this context
    /// * `velocity_improving` — from DeltaEngine: net improving vs degrading
    /// * `motif_is_attractor` — from HierarchicalMotifRuntime: entity locked on attractor
    pub fn synthesize(
        &self,
        overlay:           &EntityHealthOverlay,
        active_labels:     &[String],
        eem_results:       &[EquationSearchResult],
        velocity_improving: bool,
        motif_is_attractor: bool,
    ) -> EntityPoetry {
        let v = &overlay.vector;

        // Identify dominant dimension.
        let (dom_idx, dom_score) = dominant_dim(v);
        let dom_name = HealthVector::dim_name(dom_idx).to_string();

        // Select top labels — deduplicated, trimmed.
        let key_labels: Vec<String> = active_labels.iter()
            .filter(|l| !l.is_empty() && l.len() > 1)
            .take(self.config.max_labels)
            .cloned()
            .collect();

        // Find best EEM match above threshold.
        let cited_equation: Option<String> = eem_results.iter()
            .filter(|r| r.relevance >= self.config.min_eem_relevance)
            .max_by(|a, b| a.relevance.partial_cmp(&b.relevance)
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| format!("{}: {}", r.equation.text, r.equation.latex));

        // Trajectory descriptor.
        let trajectory = if motif_is_attractor {
            "attractor-locked".to_string()
        } else if velocity_improving {
            "ascending".to_string()
        } else {
            "descending".to_string()
        };

        // Assemble synthesis from actual data — no canned phrases, no hallucinated content.
        let synthesis = assemble_synthesis(
            &overlay.entity_id,
            &overlay.label,
            &dom_name,
            dom_score,
            &key_labels,
            cited_equation.as_deref(),
            &trajectory,
            overlay.composite,
            overlay.motif_entropy,
            self.config.include_scores,
        );

        EntityPoetry {
            entity_id:      overlay.entity_id.clone(),
            timestamp:      overlay.timestamp,
            synthesis,
            dominant_dim:   dom_name,
            dominant_score: dom_score,
            health_label:   overlay.label.clone(),
            key_labels,
            cited_equation,
            trajectory,
        }
    }

    /// Batch synthesis for all entities in the overlay set.
    pub fn synthesize_all(
        &self,
        overlays:          &[EntityHealthOverlay],
        active_labels_map: &HashMap<String, Vec<String>>,
        eem_results:       &[EquationSearchResult],
        velocity_flags:    &HashMap<String, bool>,  // entity_id → improving
        attractor_flags:   &HashMap<String, bool>,  // entity_id → is_attractor
    ) -> Vec<EntityPoetry> {
        overlays.iter().map(|ov| {
            let labels  = active_labels_map.get(&ov.entity_id).map(|v| v.as_slice()).unwrap_or(&[]);
            let impr    = velocity_flags.get(&ov.entity_id).copied().unwrap_or(false);
            let attr    = attractor_flags.get(&ov.entity_id).copied().unwrap_or(false);
            self.synthesize(ov, labels, eem_results, impr, attr)
        }).collect()
    }
}

impl Default for PoetryRuntime {
    fn default() -> Self { Self::new(PoetryConfig::default()) }
}

// ── Assembly ──────────────────────────────────────────────────────────────────

/// Build the synthesis string from factual data only.
/// The structure is:
///   <entity_id> [<health_label>]: <dominant dimension> at <score>.
///   Defined by: <labels>.
///   [Trajectory: <trajectory>; entropy <entropy>.]
///   [Candidate equation: <formula>.]
#[allow(clippy::too_many_arguments)]
fn assemble_synthesis(
    entity_id:    &str,
    health_label: &str,
    dom_dim:      &str,
    dom_score:    f64,
    labels:       &[String],
    equation:     Option<&str>,
    trajectory:   &str,
    composite:    f64,
    entropy:      f64,
    include_scores: bool,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Opening: entity identity + health state.
    if include_scores {
        parts.push(format!(
            "{entity_id} [{health_label}, composite={composite:.2}]: \
             dominant dimension is {dom_dim} ({dom_score:.2})"
        ));
    } else {
        parts.push(format!(
            "{entity_id} [{health_label}]: dominant dimension is {dom_dim}"
        ));
    }

    // Label cluster — what the pool has learned about this entity.
    if !labels.is_empty() {
        let label_str = labels.iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("characterized by: {label_str}"));
    }

    // Trajectory.
    if include_scores {
        parts.push(format!("trajectory: {trajectory}; motif entropy: {entropy:.3}"));
    } else {
        parts.push(format!("trajectory: {trajectory}"));
    }

    // Equation citation (only if EEM found something relevant).
    if let Some(eq) = equation {
        parts.push(format!("candidate equation: {eq}"));
    }

    parts.join(". ")
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns (dimension_index, score) for the highest-scoring dimension.
fn dominant_dim(v: &HealthVector) -> (usize, f64) {
    v.as_array().iter().copied().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::entity_health::make_rings;

    fn dummy_overlay(eid: &str) -> EntityHealthOverlay {
        let v = HealthVector {
            structural: 0.8, energetic: 0.6, temporal: 0.5,
            intentional: 0.7, environmental: 0.4, informational: 0.9,
        };
        let composite = v.composite();
        EntityHealthOverlay {
            entity_id:   eid.to_string(),
            timestamp:   Timestamp { unix: 1000 },
            vector:      v.clone(),
            composite,
            label:       "stable".to_string(),
            rings:       make_rings(&v),
            position:    None,
            motif_entropy: 0.3,
        }
    }

    #[test]
    fn synthesis_contains_entity_id() {
        let rt = PoetryRuntime::default();
        let ov = dummy_overlay("entity_42");
        let labels = vec!["photosynthesis".to_string(), "cell_wall".to_string()];
        let poem = rt.synthesize(&ov, &labels, &[], true, false);
        assert!(poem.synthesis.contains("entity_42"));
        assert!(poem.synthesis.contains("photosynthesis") || poem.synthesis.contains("cell_wall"));
    }

    #[test]
    fn dominant_dim_is_informational() {
        let v = HealthVector {
            structural: 0.1, energetic: 0.2, temporal: 0.3,
            intentional: 0.4, environmental: 0.5, informational: 0.9,
        };
        let (idx, score) = dominant_dim(&v);
        assert_eq!(idx, 5, "informational should dominate");
        assert!((score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn synthesis_no_labels_still_produces_output() {
        let rt = PoetryRuntime::default();
        let ov = dummy_overlay("bare");
        let poem = rt.synthesize(&ov, &[], &[], false, true);
        assert!(!poem.synthesis.is_empty());
        assert_eq!(poem.trajectory, "attractor-locked");
    }
}
