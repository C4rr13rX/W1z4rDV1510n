//! Chaos World Model — reconstructs the most probable history of an entity
//! by reverse-traversing the learned motif transition graph.
//!
//! The HierarchicalMotifRuntime builds transition tables that record how often
//! motif A is followed by motif B.  For *forward* prediction those tables are
//! read left-to-right: given A, what comes next?  The Chaos World Model reads
//! the same tables *right-to-left*: given B (the observed present), what
//! probably came before?
//!
//! Because the motif id sequence is the compressed behavioural fingerprint of
//! the entity, this backward walk reconstructs a credible recent history without
//! needing direct sensor access to the past.
//!
//! The reconstruction terminates when:
//!   • `max_depth` steps are reached, or
//!   • all candidate predecessors have probability below `min_transition_prob`.

use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosWorldConfig {
    /// Maximum number of steps to reconstruct backward.
    pub max_depth: usize,
    /// Minimum transition probability for a predecessor to be considered.
    pub min_transition_prob: f64,
    /// How many candidates to explore at each backward step (beam width).
    pub beam_width: usize,
}

impl Default for ChaosWorldConfig {
    fn default() -> Self {
        Self {
            max_depth:            8,
            min_transition_prob:  0.05,
            beam_width:           3,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

/// One step in the reconstructed history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStep {
    /// The motif id at this step.
    pub motif_id:    String,
    /// Probability that this motif preceded the one after it in the chain.
    pub probability: f64,
    /// Cumulative probability of this entire sequence up to this step.
    pub cumulative:  f64,
    /// Estimated offset from the present (in motif-steps, not seconds).
    pub steps_ago:   usize,
}

/// Full reconstruction for one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosHistoryReport {
    pub entity_id:          String,
    pub timestamp:          Timestamp,
    /// Reconstructed steps, ordered most-recent-first (steps_ago ascending).
    pub steps:              Vec<HistoryStep>,
    /// Cumulative probability of the most probable complete path.
    pub overall_confidence: f64,
    /// Depth actually reached (may be < max_depth if transitions ran out).
    pub reconstruction_depth: usize,
    /// The motif id the reconstruction started from.
    pub anchor_motif:       String,
}

// ── Transition index ──────────────────────────────────────────────────────────

/// Condensed form of the transition tables extracted from HierarchicalMotifRuntime.
/// `from_motif_id → Vec<(to_motif_id, normalized_probability)>` sorted desc.
pub type TransitionIndex = HashMap<String, Vec<(String, f64)>>;

/// Build a reverse-lookup index from a forward transition map.
/// Input: `(from, to) → count`; Output: `to → Vec<(from, probability)>`.
pub fn build_reverse_index(
    forward_transitions: &HashMap<(String, String), usize>,
) -> TransitionIndex {
    // Accumulate counts keyed by `to`.
    let mut rev: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    for ((from, to), &count) in forward_transitions {
        rev.entry(to.clone()).or_default().push((from.clone(), count));
    }

    // Normalise each `to` bucket.
    let mut index: TransitionIndex = HashMap::new();
    for (to, preds) in rev {
        let total: usize = preds.iter().map(|(_, c)| c).sum();
        if total == 0 { continue; }
        let total_f = total as f64;
        let mut probs: Vec<(String, f64)> = preds.into_iter()
            .map(|(from, c)| (from, c as f64 / total_f))
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        index.insert(to, probs);
    }
    index
}

// ── Runtime ───────────────────────────────────────────────────────────────────

/// The Chaos World Model reconstructs probable entity histories from motif data.
/// It does not store any per-entity state itself — it operates on snapshots of
/// the motif transition tables that the caller extracts from HierarchicalMotifRuntime.
pub struct ChaosWorldModel {
    config: ChaosWorldConfig,
}

impl ChaosWorldModel {
    pub fn new(config: ChaosWorldConfig) -> Self {
        Self { config }
    }

    /// Reconstruct the most probable history for an entity.
    ///
    /// * `entity_id`      — identifier for output attribution only
    /// * `anchor_motif`   — the most recently observed motif id for this entity
    ///                      (e.g. from `HierarchicalMotifRuntime::window_tail(1)[0]`)
    /// * `reverse_index`  — built from the runtime's transition tables via
    ///                      `build_reverse_index()`
    /// * `timestamp`      — current frame timestamp
    pub fn reconstruct(
        &self,
        entity_id:     &str,
        anchor_motif:  &str,
        reverse_index: &TransitionIndex,
        timestamp:     Timestamp,
    ) -> ChaosHistoryReport {
        let mut steps: Vec<HistoryStep> = Vec::new();
        let mut cumulative = 1.0_f64;

        // Beam search — keep at most `beam_width` candidate sequences.
        // Each item: (current_motif_id, cumulative_prob_so_far, path_so_far)
        let mut beam: Vec<(String, f64, Vec<(String, f64)>)> = vec![
            (anchor_motif.to_string(), 1.0, Vec::new())
        ];
        let mut best_path: Vec<HistoryStep> = Vec::new();
        let mut best_cum  = 0.0_f64;

        for depth in 0..self.config.max_depth {
            let mut next_beam: Vec<(String, f64, Vec<(String, f64)>)> = Vec::new();

            for (current_id, current_cum, path) in &beam {
                let predecessors = match reverse_index.get(current_id.as_str()) {
                    Some(p) => p,
                    None    => continue,
                };

                for (pred_id, pred_prob) in predecessors.iter().take(self.config.beam_width) {
                    if *pred_prob < self.config.min_transition_prob {
                        break; // sorted desc, so all remaining are also below threshold
                    }
                    let new_cum = current_cum * pred_prob;
                    let mut new_path = path.clone();
                    new_path.push((pred_id.clone(), *pred_prob));
                    next_beam.push((pred_id.clone(), new_cum, new_path));
                }
            }

            if next_beam.is_empty() {
                break;
            }

            // Sort beam by cumulative prob desc, keep top `beam_width`.
            next_beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(self.config.beam_width);

            // Track best-so-far path.
            if let Some((_, cum, path)) = next_beam.first() {
                if *cum > best_cum || best_cum == 0.0 {
                    best_cum = *cum;
                    // Rebuild steps from path (most recent first).
                    best_path = path.iter().enumerate().map(|(i, (mid, prob))| {
                        HistoryStep {
                            motif_id:    mid.clone(),
                            probability: *prob,
                            cumulative:  {
                                // Recompute cumulative for this step.
                                path.iter().take(i + 1).map(|(_, p)| p).product()
                            },
                            steps_ago: i + 1,
                        }
                    }).collect();
                }
            }

            beam = next_beam;
        }

        let reconstruction_depth = best_path.len();
        let overall_confidence   = if reconstruction_depth > 0 { best_cum } else { 0.0 };

        ChaosHistoryReport {
            entity_id:          entity_id.to_string(),
            timestamp,
            steps:              best_path,
            overall_confidence,
            reconstruction_depth,
            anchor_motif:       anchor_motif.to_string(),
        }
    }

    /// Reconstruct histories for multiple entities in one call.
    pub fn reconstruct_all(
        &self,
        entities:       &[(&str, &str)],   // (entity_id, anchor_motif)
        reverse_index:  &TransitionIndex,
        timestamp:      Timestamp,
    ) -> Vec<ChaosHistoryReport> {
        entities.iter().map(|(eid, anchor)| {
            self.reconstruct(eid, anchor, reverse_index, timestamp)
        }).collect()
    }
}

impl Default for ChaosWorldModel {
    fn default() -> Self { Self::new(ChaosWorldConfig::default()) }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> TransitionIndex {
        // Encode: A→B→C (high prob) and A→D (low prob)
        let mut fwd: HashMap<(String, String), usize> = HashMap::new();
        fwd.insert(("A".into(), "B".into()), 8);
        fwd.insert(("A".into(), "D".into()), 2);
        fwd.insert(("B".into(), "C".into()), 10);
        build_reverse_index(&fwd)
    }

    #[test]
    fn reverse_index_built_correctly() {
        let idx = make_index();
        // C was preceded by B (prob 1.0).
        let c_preds = idx.get("C").expect("C should have predecessors");
        assert_eq!(c_preds[0].0, "B");
        assert!((c_preds[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn reconstruct_from_c_finds_b_then_a() {
        let idx   = make_index();
        let model = ChaosWorldModel::default();
        let report = model.reconstruct("ent1", "C", &idx, Timestamp { unix: 100 });
        assert!(report.reconstruction_depth >= 1);
        // First step back from C should be B.
        assert_eq!(report.steps[0].motif_id, "B");
    }

    #[test]
    fn empty_index_returns_zero_depth() {
        let idx   = TransitionIndex::new();
        let model = ChaosWorldModel::default();
        let report = model.reconstruct("e", "X", &idx, Timestamp { unix: 0 });
        assert_eq!(report.reconstruction_depth, 0);
        assert_eq!(report.overall_confidence, 0.0);
    }
}
