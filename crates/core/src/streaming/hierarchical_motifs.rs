use crate::hardware::HardwareProfile;
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::behavior::BehaviorMotif;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalMotifConfig {
    pub enabled: bool,
    /// Minimum number of child motifs in a meta-motif.
    pub min_sequence_len: usize,
    /// Maximum number of child motifs in a meta-motif.
    pub max_sequence_len: usize,
    /// Normalized edit-distance threshold below which two sequences match.
    pub similarity_threshold: f64,
    /// Minimum repetitions of a sequence before it is promoted.
    pub min_support: usize,
    /// Transition entropy below which we flag as attractor.
    pub promote_entropy_threshold: f64,
}

impl Default for HierarchicalMotifConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_sequence_len: 2,
            max_sequence_len: 16,
            similarity_threshold: 0.6,
            min_support: 3,
            promote_entropy_threshold: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalMotifReport {
    pub timestamp: Timestamp,
    pub levels: Vec<LevelReport>,
    pub newly_promoted: Vec<MetaMotif>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelReport {
    pub level: usize,
    pub motif_count: usize,
    pub transition_count: usize,
    pub entropy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaMotif {
    pub id: String,
    pub level: usize,
    pub children: Vec<String>,
    pub support: usize,
    pub duration_secs: f64,
    pub signature: Vec<f64>,
    pub entropy: f64,
    pub is_attractor: bool,
    pub description: String,
}

// ---------------------------------------------------------------------------
// Internal level bookkeeping
// ---------------------------------------------------------------------------

/// A single observed occurrence of a motif-id sequence at some level.
#[derive(Debug, Clone)]
struct SequenceOccurrence {
    ids: Vec<String>,
    first_seen: i64,
    last_seen: i64,
    count: usize,
    total_duration: f64,
}

/// Per-level state: tracks the sliding window of motif ids, transition
/// counts, and candidate sequences waiting for promotion.
#[derive(Debug, Clone)]
struct MotifLevel {
    /// Rolling window of observed motif ids at this level.
    window: VecDeque<String>,
    /// Transition counts: (from, to) -> count.
    transitions: HashMap<(String, String), usize>,
    /// Candidate sequences keyed by a canonical hash of the id list.
    candidates: HashMap<String, SequenceOccurrence>,
    /// Meta-motifs that have already been promoted out of this level.
    promoted: HashMap<String, MetaMotif>,
}

impl MotifLevel {
    fn new() -> Self {
        Self {
            window: VecDeque::new(),
            transitions: HashMap::new(),
            candidates: HashMap::new(),
            promoted: HashMap::new(),
        }
    }

    /// Total distinct motif ids observed in the current window.
    fn motif_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for id in &self.window {
            seen.insert(id.as_str());
        }
        seen.len()
    }

    /// Shannon entropy of the transition distribution.
    fn transition_entropy(&self) -> f64 {
        let total: usize = self.transitions.values().sum();
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f64;
        let mut entropy = 0.0_f64;
        for &count in self.transitions.values() {
            if count > 0 {
                let p = count as f64 / total_f;
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

// ---------------------------------------------------------------------------
// Sequence similarity (normalized edit distance)
// ---------------------------------------------------------------------------

/// Levenshtein edit distance between two slices of motif ids, normalized to
/// [0, 1] by dividing by the maximum of the two lengths.  A value of 0 means
/// identical; 1 means completely different.
fn normalized_edit_distance(a: &[String], b: &[String]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 && m == 0 {
        return 0.0;
    }

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m] as f64 / n.max(m) as f64
}

/// Produce a deterministic hash for an ordered list of motif ids.
fn sequence_hash(ids: &[String]) -> String {
    let payload = ids.join("|");
    compute_payload_hash(payload.as_bytes())
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

pub struct HierarchicalMotifRuntime {
    config: HierarchicalMotifConfig,
    levels: Vec<MotifLevel>,
    tick: u64,
    /// Window size cap per level.  `None` = unlimited (high-spec machines).
    /// On constrained hardware this is set to `config.max_sequence_len * 4`.
    window_cap: Option<usize>,
}

impl HierarchicalMotifRuntime {
    pub fn new(config: HierarchicalMotifConfig) -> Self {
        let hw = HardwareProfile::detect();
        let window_cap = if hw.motif_window_cap().is_some() {
            Some(config.max_sequence_len * 4)
        } else {
            None
        };
        Self {
            config,
            levels: Vec::new(),
            tick: 0,
            window_cap,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Feed level-0 motifs from `BehaviorSubstrate` and propagate upward.
    /// Returns a report whenever at least one level has content.
    pub fn observe(
        &mut self,
        motifs: &[BehaviorMotif],
        timestamp: Timestamp,
    ) -> Option<HierarchicalMotifReport> {
        if !self.config.enabled || motifs.is_empty() {
            return None;
        }
        self.tick += 1;

        // Collect atomic motif ids into level 0.
        let mut level_ids: Vec<String> = motifs.iter().map(|m| m.id.clone()).collect();
        let mut level_durations: Vec<f64> = motifs.iter().map(|m| m.duration_secs).collect();

        let mut newly_promoted: Vec<MetaMotif> = Vec::new();

        let mut lvl = 0;
        loop {
            if level_ids.is_empty() {
                break;
            }
            // Grow the levels vec on demand — no ceiling.
            if lvl >= self.levels.len() {
                self.levels.push(MotifLevel::new());
            }
            let promoted = self.ingest_level(lvl, &level_ids, &level_durations, timestamp.unix);

            level_ids = promoted.iter().map(|m| m.id.clone()).collect();
            level_durations = promoted.iter().map(|m| m.duration_secs).collect();
            newly_promoted.extend(promoted);
            lvl += 1;
        }

        let levels: Vec<LevelReport> = self
            .levels
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.window.is_empty())
            .map(|(i, l)| LevelReport {
                level: i,
                motif_count: l.motif_count(),
                transition_count: l.transitions.len(),
                entropy: l.transition_entropy(),
            })
            .collect();

        if levels.is_empty() && newly_promoted.is_empty() {
            return None;
        }

        Some(HierarchicalMotifReport {
            timestamp,
            levels,
            newly_promoted,
        })
    }

    /// Return all meta-motifs discovered so far across every level.
    pub fn meta_motifs(&self) -> Vec<MetaMotif> {
        let mut out = Vec::new();
        for level in &self.levels {
            out.extend(level.promoted.values().cloned());
        }
        out
    }

    /// Given the most recent motif id seen (`last_id`), return the learned
    /// transition probabilities to successor motifs at every level, sorted by
    /// probability descending. This is the core sequence-completion primitive
    /// that the annealer uses to bias proposals toward fabric-predicted states.
    ///
    /// Returns `(next_motif_id, probability)` pairs. Probabilities are
    /// normalized within each level; results across levels are merged by
    /// taking the max probability for each successor.
    pub fn next_predictions(&self, last_id: &str) -> Vec<(String, f64)> {
        let mut acc: HashMap<String, f64> = HashMap::new();
        for level in &self.levels {
            let total_from: usize = level
                .transitions
                .iter()
                .filter(|((from, _), _)| from == last_id)
                .map(|(_, &count)| count)
                .sum();
            if total_from == 0 {
                continue;
            }
            let total_f = total_from as f64;
            for ((from, to), &count) in &level.transitions {
                if from != last_id {
                    continue;
                }
                let prob = count as f64 / total_f;
                let entry = acc.entry(to.clone()).or_insert(0.0);
                if prob > *entry {
                    *entry = prob;
                }
            }
        }
        let mut predictions: Vec<(String, f64)> = acc.into_iter().collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions
    }

    /// The current tail of the level-0 window — the most recently observed
    /// motif id sequence. Used by the annealer to seed next_predictions().
    pub fn window_tail(&self, n: usize) -> Vec<String> {
        self.levels
            .first()
            .map(|lvl| lvl.window.iter().rev().take(n).cloned().collect::<Vec<_>>())
            .unwrap_or_default()
    }

    /// Combined transition entropy across all active levels — a scalar
    /// measure of how certain the fabric is about what comes next.
    /// 0.0 = completely certain (attractor), high = unpredictable.
    pub fn mean_transition_entropy(&self) -> f64 {
        let entropies: Vec<f64> = self
            .levels
            .iter()
            .filter(|l| !l.transitions.is_empty())
            .map(|l| l.transition_entropy())
            .collect();
        if entropies.is_empty() {
            return 1.0; // unknown → treat as uncertain
        }
        entropies.iter().sum::<f64>() / entropies.len() as f64
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    /// Ingest a batch of motif ids at the given level.  Returns any newly
    /// promoted meta-motifs (which become input for the next level).
    fn ingest_level(
        &mut self,
        lvl: usize,
        ids: &[String],
        durations: &[f64],
        unix: i64,
    ) -> Vec<MetaMotif> {
        let level = &mut self.levels[lvl];

        // Record transitions and extend the window.
        // On constrained hardware the window is capped to avoid unbounded growth.
        let cap = self.window_cap;
        for (i, id) in ids.iter().enumerate() {
            if let Some(prev) = level.window.back() {
                let key = (prev.clone(), id.clone());
                *level.transitions.entry(key).or_insert(0) += 1;
            }
            level.window.push_back(id.clone());
            if let Some(max_win) = cap {
                if level.window.len() > max_win {
                    level.window.pop_front();
                }
            }
            let _ = i; // durations used below
        }

        // Slide sub-sequences of every valid length over the window and
        // match against existing candidates (or create new ones).
        let window_slice: Vec<String> = level.window.iter().cloned().collect();
        let min_len = self.config.min_sequence_len;
        let max_len = self.config.max_sequence_len.min(window_slice.len());

        for seq_len in min_len..=max_len {
            if window_slice.len() < seq_len {
                break;
            }
            let start = if window_slice.len() > seq_len {
                window_slice.len() - seq_len
            } else {
                0
            };
            let subseq = &window_slice[start..start + seq_len];
            let hash = sequence_hash(subseq);

            // Try to find an existing candidate that is similar enough.
            let matched_key = self.find_similar_candidate(lvl, subseq);

            let level = &mut self.levels[lvl];
            match matched_key {
                Some(key) => {
                    if let Some(occ) = level.candidates.get_mut(&key) {
                        occ.count += 1;
                        occ.last_seen = unix;
                        // Accumulate duration from input batch (approximate).
                        let dur_sum: f64 = durations.iter().copied().sum();
                        occ.total_duration += dur_sum / durations.len().max(1) as f64
                            * seq_len as f64;
                    }
                }
                None => {
                    level.candidates.insert(
                        hash,
                        SequenceOccurrence {
                            ids: subseq.to_vec(),
                            first_seen: unix,
                            last_seen: unix,
                            count: 1,
                            total_duration: {
                                let dur_sum: f64 = durations.iter().copied().sum();
                                dur_sum / durations.len().max(1) as f64 * seq_len as f64
                            },
                        },
                    );
                }
            }
        }

        // Promote candidates that have reached min_support.
        let mut promoted = Vec::new();
        let level = &mut self.levels[lvl];
        let keys_to_promote: Vec<String> = level
            .candidates
            .iter()
            .filter(|(_, occ)| occ.count >= self.config.min_support)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_promote {
            if let Some(occ) = level.candidates.remove(&key) {
                let entropy = level.transition_entropy();
                let is_attractor = entropy < self.config.promote_entropy_threshold
                    && entropy > 0.0;

                let sig = build_signature(&occ.ids);
                let meta_id = format!(
                    "meta-L{}-{}",
                    lvl + 1,
                    &sequence_hash(&occ.ids)[..12]
                );

                let meta = MetaMotif {
                    id: meta_id.clone(),
                    level: lvl + 1,
                    children: occ.ids.clone(),
                    support: occ.count,
                    duration_secs: occ.total_duration / occ.count.max(1) as f64,
                    signature: sig,
                    entropy,
                    is_attractor,
                    description: format!(
                        "L{} meta-motif ({} children, support {})",
                        lvl + 1,
                        occ.ids.len(),
                        occ.count
                    ),
                };
                level.promoted.insert(meta_id, meta.clone());
                promoted.push(meta);
            }
        }

        // Prune stale candidates that have not been seen in a long time
        // relative to the tick counter (heuristic: older than 64 ticks).
        let stale_horizon = unix.saturating_sub(3600);
        level
            .candidates
            .retain(|_, occ| occ.last_seen >= stale_horizon);

        promoted
    }

    /// Find an existing candidate at `lvl` whose id sequence is similar
    /// enough (normalized edit distance below threshold) to `subseq`.
    fn find_similar_candidate(&self, lvl: usize, subseq: &[String]) -> Option<String> {
        let level = &self.levels[lvl];
        let query_len = subseq.len();
        for (key, occ) in &level.candidates {
            let cand_len = occ.ids.len();
            // Length filter: if the length ratio already exceeds the threshold, edit distance
            // can't possibly be within the threshold — skip the O(n*m) computation.
            let max_len = query_len.max(cand_len);
            if max_len > 0 {
                let len_diff = (query_len as f64 - cand_len as f64).abs() / max_len as f64;
                if len_diff > self.config.similarity_threshold {
                    continue;
                }
            }
            let dist = normalized_edit_distance(&occ.ids, subseq);
            if dist <= self.config.similarity_threshold {
                return Some(key.clone());
            }
        }
        None
    }
}

/// Build a compact numeric signature from an ordered list of motif ids.
/// We hash each id to a float in [0, 1) and collect them.
fn build_signature(ids: &[String]) -> Vec<f64> {
    ids.iter()
        .map(|id| {
            let h = compute_payload_hash(id.as_bytes());
            // Take first 8 hex chars -> u32 -> normalize to [0,1).
            let nibble = &h[..8.min(h.len())];
            let val = u32::from_str_radix(nibble, 16).unwrap_or(0);
            val as f64 / u32::MAX as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::behavior::{
        GraphSignature, MotifLineage, MotifProvenance, TimeFrequencySummary,
    };

    fn make_motif(id: &str, dur: f64) -> BehaviorMotif {
        BehaviorMotif {
            id: id.to_string(),
            entity_id: "test-entity".to_string(),
            support: 1,
            duration_secs: dur,
            description_length: 0.0,
            prototype: vec![],
            time_frequency: TimeFrequencySummary {
                amplitudes: vec![],
                phases: vec![],
            },
            graph_signature: GraphSignature {
                mean_proximity: 0.0,
                mean_coherence: 0.0,
            },
            lineage: MotifLineage::default(),
            provenance: MotifProvenance::default(),
        }
    }

    #[test]
    fn promotes_recurring_sequence_to_meta_motif() {
        let config = HierarchicalMotifConfig {
            enabled: true,
            min_sequence_len: 3,
            max_sequence_len: 3,
            similarity_threshold: 0.0, // exact match only
            min_support: 3,
            promote_entropy_threshold: 1.0,
        };

        let mut runtime = HierarchicalMotifRuntime::new(config);

        let sequence = vec![
            make_motif("A", 1.0),
            make_motif("B", 1.0),
            make_motif("C", 1.0),
        ];

        // Feed the same 3-motif sequence 4 times.
        let mut promoted_any = false;
        for i in 0..4 {
            let ts = Timestamp { unix: 1000 + i };
            if let Some(report) = runtime.observe(&sequence, ts) {
                if !report.newly_promoted.is_empty() {
                    promoted_any = true;
                    let meta = &report.newly_promoted[0];
                    assert_eq!(meta.level, 1);
                    assert_eq!(meta.children, vec!["A", "B", "C"]);
                    assert!(meta.support >= 3);
                }
            }
        }

        assert!(promoted_any, "expected a meta-motif to be promoted");

        // Verify it appears in the full catalogue.
        let all = runtime.meta_motifs();
        assert!(!all.is_empty());
        assert_eq!(all[0].level, 1);
    }

    #[test]
    fn does_not_promote_below_min_support() {
        let config = HierarchicalMotifConfig {
            min_support: 5,
            ..HierarchicalMotifConfig::default()
        };
        let mut runtime = HierarchicalMotifRuntime::new(config);

        let sequence = vec![make_motif("X", 1.0), make_motif("Y", 2.0)];

        // Only feed it twice — not enough.
        for i in 0..2 {
            let ts = Timestamp { unix: 2000 + i };
            if let Some(report) = runtime.observe(&sequence, ts) {
                assert!(
                    report.newly_promoted.is_empty(),
                    "should not promote with only {} observations",
                    i + 1
                );
            }
        }

        assert!(runtime.meta_motifs().is_empty());
    }

    #[test]
    fn edit_distance_exact_match() {
        let a: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let b: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        assert!((normalized_edit_distance(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn edit_distance_complete_mismatch() {
        let a: Vec<String> = vec!["A".into(), "B".into()];
        let b: Vec<String> = vec!["C".into(), "D".into()];
        assert!((normalized_edit_distance(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn entropy_zero_for_single_transition() {
        let mut level = MotifLevel::new();
        level
            .transitions
            .insert(("A".into(), "B".into()), 10);
        // Single transition type => zero entropy.
        assert!((level.transition_entropy()).abs() < 1e-9);
    }
}
