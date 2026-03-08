use crate::math_toolbox as math;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganicEncoderConfig {
    pub enabled: bool,
    pub max_basis_vectors: usize,
    pub patch_size: usize,
    pub novelty_threshold: f64,
    pub cooccurrence_window: usize,
    pub ema_alpha: f64,
    pub max_history: usize,
    pub min_activation: f64,
}

impl Default for OrganicEncoderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_basis_vectors: 256,
            patch_size: 8,
            novelty_threshold: 0.3,
            cooccurrence_window: 16,
            ema_alpha: 0.05,
            max_history: 512,
            min_activation: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedFeature {
    pub basis_id: String,
    pub activation: f64,
    pub novelty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganicEncoderReport {
    pub timestamp: Timestamp,
    pub basis_count: usize,
    pub features: Vec<EncodedFeature>,
    pub newly_spawned: Vec<String>,
    pub cooccurrence_groups: Vec<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Internal basis vector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BasisVector {
    id: String,
    weights: Vec<f64>,
    activation_count: u64,
}

// ---------------------------------------------------------------------------
// OrganicEncoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganicEncoder {
    config: OrganicEncoderConfig,
    bases: Vec<BasisVector>,
    next_id: u64,
    /// Rolling window of (timestamp, active-basis-ids) for co-occurrence tracking.
    activation_history: VecDeque<(Timestamp, Vec<String>)>,
}

impl OrganicEncoder {
    pub fn new(config: OrganicEncoderConfig) -> Self {
        Self {
            config,
            bases: Vec::new(),
            next_id: 0,
            activation_history: VecDeque::new(),
        }
    }

    pub fn basis_count(&self) -> usize {
        self.bases.len()
    }

    // -----------------------------------------------------------------------
    // Core encoder
    // -----------------------------------------------------------------------

    pub fn encode(&mut self, raw: &[f64], timestamp: Timestamp) -> OrganicEncoderReport {
        if raw.is_empty() || !self.config.enabled {
            return OrganicEncoderReport {
                timestamp,
                basis_count: self.bases.len(),
                features: Vec::new(),
                newly_spawned: Vec::new(),
                cooccurrence_groups: Vec::new(),
            };
        }

        // Normalize input so cosine similarity is meaningful.
        let input = normalize_vec(raw);

        // --- Project onto every existing basis vector ---
        let mut features: Vec<EncodedFeature> = Vec::with_capacity(self.bases.len());
        let mut max_sim: f64 = -1.0;

        for basis in &self.bases {
            let sim = similarity(&input, &basis.weights);
            if sim > max_sim {
                max_sim = sim;
            }
            let novelty = 1.0 - sim.max(0.0);
            features.push(EncodedFeature {
                basis_id: basis.id.clone(),
                activation: sim.max(0.0),
                novelty,
            });
        }

        // --- Novelty check: spawn new basis if nothing matches well ---
        let mut newly_spawned: Vec<String> = Vec::new();

        let is_novel = self.bases.is_empty() || max_sim < (1.0 - self.config.novelty_threshold);
        if is_novel && self.bases.len() < self.config.max_basis_vectors {
            let id = self.spawn_basis(&input);
            features.push(EncodedFeature {
                basis_id: id.clone(),
                activation: 1.0,
                novelty: 1.0,
            });
            newly_spawned.push(id);
        }

        // --- Hebbian adaptation: blend matching bases toward the input ---
        let alpha = self.config.ema_alpha;
        let min_act = self.config.min_activation;
        for basis in &mut self.bases {
            let sim = similarity(&input, &basis.weights);
            if sim >= min_act {
                basis.activation_count += 1;
                // EMA update: w ← (1 - α)·w + α·input, then re-normalize.
                for (w, &x) in basis.weights.iter_mut().zip(input.iter()) {
                    *w = (1.0 - alpha) * *w + alpha * x;
                }
                let norm = math::l2_norm(&basis.weights);
                if norm > 0.0 {
                    for w in &mut basis.weights {
                        *w /= norm;
                    }
                }
            }
        }

        // --- Record activation for co-occurrence tracking ---
        let active_ids: Vec<String> = features
            .iter()
            .filter(|f| f.activation >= self.config.min_activation)
            .map(|f| f.basis_id.clone())
            .collect();

        self.activation_history
            .push_back((timestamp, active_ids));
        while self.activation_history.len() > self.config.max_history {
            self.activation_history.pop_front();
        }

        // --- Build co-occurrence groups from the recent window ---
        let cooccurrence_groups = self.compute_cooccurrence_groups();

        OrganicEncoderReport {
            timestamp,
            basis_count: self.bases.len(),
            features,
            newly_spawned,
            cooccurrence_groups,
        }
    }

    // -----------------------------------------------------------------------
    // Image-patch encoder (V1-like Gabor feature extraction)
    // -----------------------------------------------------------------------

    pub fn encode_image_patch(
        &mut self,
        pixels: &[f64],
        width: usize,
        height: usize,
        timestamp: Timestamp,
    ) -> OrganicEncoderReport {
        let edge_features = extract_gabor_features(pixels, width, height, self.config.patch_size);
        self.encode(&edge_features, timestamp)
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn spawn_basis(&mut self, template: &[f64]) -> String {
        let id = format!("basis_{}", self.next_id);
        self.next_id += 1;
        self.bases.push(BasisVector {
            id: id.clone(),
            weights: template.to_vec(),
            activation_count: 1,
        });
        id
    }

    /// Find groups of basis IDs that frequently co-activate within a sliding
    /// window of `cooccurrence_window` steps.
    fn compute_cooccurrence_groups(&self) -> Vec<Vec<String>> {
        let window = self.config.cooccurrence_window;
        let history_len = self.activation_history.len();
        let start = if history_len > window {
            history_len - window
        } else {
            0
        };

        // Count pairwise co-occurrences.
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut single_counts: HashMap<String, usize> = HashMap::new();

        for (_ts, ids) in self.activation_history.iter().skip(start) {
            for id in ids {
                *single_counts.entry(id.clone()).or_insert(0) += 1;
            }
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let a = ids[i].clone();
                    let b = ids[j].clone();
                    let key = if a < b { (a, b) } else { (b, a) };
                    *pair_counts.entry(key).or_insert(0) += 1;
                }
            }
        }

        // A pair forms a group if their co-occurrence ratio (Jaccard-like)
        // exceeds 0.5 relative to the less-frequent member.
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for ((a, b), count) in &pair_counts {
            let min_single = single_counts
                .get(a)
                .copied()
                .unwrap_or(0)
                .min(single_counts.get(b).copied().unwrap_or(0));
            if min_single > 0 && (*count as f64 / min_single as f64) >= 0.5 {
                adjacency
                    .entry(a.clone())
                    .or_default()
                    .push(b.clone());
                adjacency
                    .entry(b.clone())
                    .or_default()
                    .push(a.clone());
            }
        }

        // Connected-component grouping via BFS.
        let mut visited: HashMap<String, bool> = HashMap::new();
        let mut groups: Vec<Vec<String>> = Vec::new();

        for node in adjacency.keys() {
            if *visited.get(node).unwrap_or(&false) {
                continue;
            }
            let mut component: Vec<String> = Vec::new();
            let mut queue: VecDeque<String> = VecDeque::new();
            queue.push_back(node.clone());
            visited.insert(node.clone(), true);

            while let Some(current) = queue.pop_front() {
                component.push(current.clone());
                if let Some(neighbors) = adjacency.get(&current) {
                    for nb in neighbors {
                        if !*visited.get(nb).unwrap_or(&false) {
                            visited.insert(nb.clone(), true);
                            queue.push_back(nb.clone());
                        }
                    }
                }
            }
            component.sort();
            if component.len() >= 2 {
                groups.push(component);
            }
        }
        groups.sort();
        groups
    }
}

// ---------------------------------------------------------------------------
// Gabor-like oriented gradient extraction
// ---------------------------------------------------------------------------

/// Compute oriented gradient energy at 0°, 45°, 90°, 135° across spatial
/// patches, mimicking V1 simple-cell receptive fields.
fn extract_gabor_features(
    pixels: &[f64],
    width: usize,
    height: usize,
    patch_size: usize,
) -> Vec<f64> {
    let ps = patch_size.max(2);
    let cols = if width >= ps { width - ps + 1 } else { 1 };
    let rows = if height >= ps { height - ps + 1 } else { 1 };
    let step = ps; // non-overlapping stride

    // 4 orientations per patch → feature vector
    let mut features: Vec<f64> = Vec::new();

    let get_px = |r: usize, c: usize| -> f64 {
        if r < height && c < width {
            pixels[r * width + c]
        } else {
            0.0
        }
    };

    let mut pr = 0;
    while pr < rows {
        let mut pc = 0;
        while pc < cols {
            let (mut e0, mut e45, mut e90, mut e135) = (0.0, 0.0, 0.0, 0.0);
            let mut count = 0u32;

            for dr in 0..(ps.saturating_sub(1)) {
                for dc in 0..(ps.saturating_sub(1)) {
                    let r = pr + dr;
                    let c = pc + dc;
                    let center = get_px(r, c);

                    // 0° (horizontal): difference along rows
                    let d_horiz = get_px(r + 1, c) - center;
                    // 90° (vertical): difference along columns
                    let d_vert = get_px(r, c + 1) - center;
                    // 45° (diagonal ↘): difference along main diagonal
                    let d_diag_main = get_px(r + 1, c + 1) - center;
                    // 135° (diagonal ↙): difference along anti-diagonal
                    let d_diag_anti = get_px(r + 1, c.wrapping_sub(1).min(width)) - center;

                    e0 += d_horiz * d_horiz;
                    e90 += d_vert * d_vert;
                    e45 += d_diag_main * d_diag_main;
                    e135 += d_diag_anti * d_diag_anti;
                    count += 1;
                }
            }

            // Mean energy per orientation for this patch.
            if count > 0 {
                let n = count as f64;
                features.push((e0 / n).sqrt());
                features.push((e45 / n).sqrt());
                features.push((e90 / n).sqrt());
                features.push((e135 / n).sqrt());
            }

            pc += step;
        }
        pr += step;
    }

    // If image was too small to produce patches, fall back to raw values.
    if features.is_empty() {
        return pixels.to_vec();
    }

    features
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize_vec(v: &[f64]) -> Vec<f64> {
    let norm = math::l2_norm(v);
    if norm <= 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    math::cosine_similarity(a, b).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(v: i64) -> Timestamp {
        Timestamp { unix: v }
    }

    #[test]
    fn encoder_spawns_basis_for_novel_input() {
        let config = OrganicEncoderConfig {
            novelty_threshold: 0.3,
            max_basis_vectors: 16,
            ..OrganicEncoderConfig::default()
        };
        let mut enc = OrganicEncoder::new(config);

        // First input should always spawn a basis.
        let input_a = vec![1.0, 0.0, 0.0, 0.0];
        let report = enc.encode(&input_a, ts(1));
        assert_eq!(enc.basis_count(), 1);
        assert_eq!(report.newly_spawned.len(), 1);

        // A very different input should spawn another basis.
        let input_b = vec![0.0, 0.0, 0.0, 1.0];
        let report = enc.encode(&input_b, ts(2));
        assert_eq!(enc.basis_count(), 2);
        assert_eq!(report.newly_spawned.len(), 1);

        // An input close to input_a should NOT spawn a new basis.
        let input_a_like = vec![0.95, 0.05, 0.0, 0.0];
        let report = enc.encode(&input_a_like, ts(3));
        assert_eq!(enc.basis_count(), 2, "similar input must not spawn new basis");
        assert!(
            report.newly_spawned.is_empty(),
            "no new basis for similar input"
        );
    }

    #[test]
    fn encoder_adapts_basis_via_hebbian_learning() {
        let config = OrganicEncoderConfig {
            ema_alpha: 0.5, // aggressive learning rate for test visibility
            novelty_threshold: 0.3,
            max_basis_vectors: 4,
            ..OrganicEncoderConfig::default()
        };
        let mut enc = OrganicEncoder::new(config);

        // Seed a basis with [1, 0, 0].
        let seed = vec![1.0, 0.0, 0.0];
        enc.encode(&seed, ts(1));
        assert_eq!(enc.basis_count(), 1);

        let original_weights = enc.bases[0].weights.clone();

        // Feed a slightly rotated vector many times → basis should move toward it.
        let nudge = vec![0.9, 0.4, 0.0];
        for t in 2..20 {
            enc.encode(&nudge, ts(t));
        }

        // After Hebbian adaptation the basis should have gained positive
        // weight in the second component (originally zero).
        assert!(
            enc.bases[0].weights[1] > original_weights[1] + 0.01,
            "Hebbian learning should shift basis toward repeated stimulus"
        );
    }

    #[test]
    fn encoder_tracks_cooccurrence_groups() {
        let config = OrganicEncoderConfig {
            novelty_threshold: 0.3,
            cooccurrence_window: 32,
            min_activation: 0.1,
            max_basis_vectors: 16,
            ema_alpha: 0.001, // low alpha so bases stay put
            ..OrganicEncoderConfig::default()
        };
        let mut enc = OrganicEncoder::new(config);

        // Create two well-separated bases.
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        enc.encode(&a, ts(1));
        enc.encode(&b, ts(2));
        assert_eq!(enc.basis_count(), 2);

        // Now feed inputs that activate BOTH bases simultaneously.
        // A combination vector will have nonzero similarity with both bases.
        let combo = vec![0.7, 0.0, 0.0, 0.7];
        for t in 3..20 {
            enc.encode(&combo, ts(t));
        }

        let report = enc.encode(&combo, ts(20));
        // Both basis_0 and basis_1 should appear in a co-occurrence group.
        let has_group = report
            .cooccurrence_groups
            .iter()
            .any(|g| g.len() >= 2);
        assert!(
            has_group,
            "co-occurring bases should form a group, got: {:?}",
            report.cooccurrence_groups
        );
    }
}
