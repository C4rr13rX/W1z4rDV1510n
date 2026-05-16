//! Temporal-prediction annealer per [`ARCHITECTURE.md`] §4.C.
//!
//! Stores a per-pool rolling history of activation frames; on demand,
//! predicts the next frame for a target pool by running simulated
//! annealing over the candidate-next-frame space.  The final cost
//! becomes `annealer_confidence` on the [`crate::GroundingReport`].
//!
//! # Algorithm
//!
//! - **State:** a candidate next-frame activation map (neuron_id → f32).
//! - **Cost:** weighted sum of L2 distance between the candidate and
//!   every historically-observed "next frame", weighted by how similar
//!   the preceding frame was to the most recently observed frame.
//!   Lower cost = better-predicted continuation.
//! - **Anneal:** start from the weighted-average warm-start; perturb a
//!   random neuron's activation each step; accept worse states with
//!   probability `exp(-Δcost / T)`; cool linearly from
//!   `temperature_initial` to `temperature_final` over `steps`.
//! - **Confidence:** `1 / (1 + best_cost / corpus_baseline)`.  Sharp
//!   predictions (best_cost ≪ historical noise floor) approach 1;
//!   diffuse predictions approach 0.
//!
//! # Backend note
//!
//! The spec specifies `argmin` (MIT/Apache) as the annealing framework.
//! This module ships an inline SA loop so tests stay fast and the
//! brain crate stays self-contained.  Swapping to argmin is a
//! mechanical change: implement `Anneal`, `CostFunction`, and run
//! `Executor::new(SimulatedAnnealing::new(...), ...)`.  The surface
//! ([`Annealer::record_frame`], [`Annealer::predict_next`],
//! [`Annealer::report_actual`]) does not change.

use ahash::AHashMap;
use std::collections::VecDeque;

use crate::neuron::{NeuronId, PoolId};

/// One activation snapshot: neuron_id → activation.  Used as both
/// history element and the SA candidate state.
pub type Frame = AHashMap<NeuronId, f32>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnnealerConfig {
    /// Per-pool history depth.  Older frames roll off the back.
    pub history_window:        usize,
    /// Number of SA steps per `predict_next` call.
    pub steps:                 u32,
    /// Initial temperature.  Higher = more random walk early.
    pub temperature_initial:   f32,
    /// Final temperature.  Lower = greedier convergence at the end.
    pub temperature_final:     f32,
    /// Per-step perturbation magnitude (added/subtracted to a single
    /// random neuron's activation).
    pub perturbation_scale:    f32,
    /// Bandwidth for the trailing-frame similarity kernel: weight =
    /// exp(-distance / sigma).  Larger sigma = more inclusive corpus.
    pub similarity_sigma:      f32,
    /// Deterministic PRNG seed.  Tests rely on determinism here so
    /// confidence numbers are reproducible.
    pub rng_seed:              u64,
}

impl Default for AnnealerConfig {
    fn default() -> Self {
        Self {
            history_window:      64,
            steps:               64,
            temperature_initial: 1.0,
            temperature_final:   0.01,
            perturbation_scale:  0.15,
            similarity_sigma:    1.0,
            rng_seed:            0xC0FFEE_u64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub predicted_frame: Frame,
    pub final_cost:      f32,
    /// Confidence ∈ [0,1] derived from convergence energy.
    pub confidence:      f32,
}

pub struct Annealer {
    pub config: AnnealerConfig,
    history:    AHashMap<PoolId, VecDeque<Frame>>,
}

/// xorshift64* — a tiny deterministic PRNG.  Local to this module so
/// the annealer has no external rand-crate dependency.  Sufficient
/// quality for SA acceptance decisions.
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the zero-state degenerate cycle.
        Self(if seed == 0 { 0xDEADBEEF_CAFEBABE } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32_unit(&mut self) -> f32 {
        // Top 24 bits → [0,1).  Stable across runs given the same seed.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn next_in_range(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() as usize) % n }
    }
}

impl Annealer {
    pub fn new(config: AnnealerConfig) -> Self {
        Self { config, history: AHashMap::new() }
    }

    pub fn history_len(&self, pool: PoolId) -> usize {
        self.history.get(&pool).map(|v| v.len()).unwrap_or(0)
    }

    pub fn last_frame(&self, pool: PoolId) -> Option<&Frame> {
        self.history.get(&pool).and_then(|v| v.back())
    }

    /// Append a frame to the pool's rolling history.  Frames older
    /// than `config.history_window` roll off the front.  Empty frames
    /// are NOT recorded — an empty frame is "nothing fired this tick,"
    /// which is not a predictive signal and would poison the corpus.
    pub fn record_frame(&mut self, pool: PoolId, frame: Frame) {
        if frame.is_empty() { return; }
        let window = self.config.history_window;
        let dq = self.history.entry(pool).or_default();
        if dq.len() >= window { dq.pop_front(); }
        dq.push_back(frame);
    }

    /// Predict the next frame for `pool` via SA.  Returns `None` when
    /// the pool has fewer than 2 history entries (no pair to learn
    /// from).  The spec's "convergence energy = confidence" surfaces
    /// in `PredictionResult.confidence`.
    pub fn predict_next(&self, pool: PoolId) -> Option<PredictionResult> {
        let hist = self.history.get(&pool)?;
        if hist.len() < 2 { return None; }

        let last = hist.back()?;
        // Pairs (prev_i, next_i) drawn from consecutive frames.
        let pairs: Vec<(&Frame, &Frame)> = hist.iter()
            .zip(hist.iter().skip(1))
            .collect();

        // Trailing-frame similarity weights.  `last` is identical to
        // pairs.last().1 (the most recent next-frame); that pair has
        // similarity from prev_i = pairs.last().0.  We use ALL pairs,
        // including the most recent — the SA is searching the
        // candidate space, not just retrieving.
        let sigma = self.config.similarity_sigma.max(1e-6);
        let weights: Vec<f32> = pairs.iter()
            .map(|(prev, _)| (-frame_distance(prev, last) / sigma).exp())
            .collect();
        let weight_sum: f32 = weights.iter().sum();

        // Warm start: weighted average of historical next-frames.
        let mut state: Frame = AHashMap::new();
        if weight_sum > 1e-9 {
            for ((_, next), w) in pairs.iter().zip(weights.iter()) {
                let w_norm = w / weight_sum;
                for (&nid, &v) in next.iter() {
                    *state.entry(nid).or_insert(0.0) += w_norm * v;
                }
            }
        } else {
            // Degenerate: no similarity at all; warm-start with the
            // unweighted mean.
            for (_, next) in pairs.iter() {
                for (&nid, &v) in next.iter() {
                    *state.entry(nid).or_insert(0.0) += v / pairs.len() as f32;
                }
            }
        }

        let baseline = corpus_baseline(&pairs);
        let mut cost = weighted_cost(&state, &pairs, &weights);
        let mut best_state = state.clone();
        let mut best_cost  = cost;

        let mut rng = XorShift64::new(self.config.rng_seed);
        let neurons: Vec<NeuronId> = state.keys().copied().collect();
        if !neurons.is_empty() {
            let t_init = self.config.temperature_initial.max(1e-6);
            let t_final = self.config.temperature_final.max(1e-6);
            let steps = self.config.steps.max(1);
            for step in 0..steps {
                let progress = step as f32 / steps as f32;
                let t = t_init * (t_final / t_init).powf(progress);

                // Perturb: jitter a random neuron's activation.
                let pick = neurons[rng.next_in_range(neurons.len())];
                let delta = (rng.next_f32_unit() * 2.0 - 1.0) * self.config.perturbation_scale;
                let prev_val = *state.get(&pick).unwrap_or(&0.0);
                let new_val  = (prev_val + delta).max(0.0);
                state.insert(pick, new_val);

                let new_cost = weighted_cost(&state, &pairs, &weights);
                let d = new_cost - cost;
                let accept = if d <= 0.0 {
                    true
                } else {
                    rng.next_f32_unit() < (-d / t).exp()
                };
                if accept {
                    cost = new_cost;
                    if new_cost < best_cost {
                        best_cost  = new_cost;
                        best_state = state.clone();
                    }
                } else {
                    // Revert.
                    state.insert(pick, prev_val);
                }
            }
        }

        let confidence = 1.0 / (1.0 + (best_cost / baseline.max(1e-6)));
        Some(PredictionResult {
            predicted_frame: best_state,
            final_cost:      best_cost,
            confidence:      confidence.clamp(0.0, 1.0),
        })
    }

    /// Compare a previously-predicted frame against the observed
    /// actual.  Returns the L2 prediction error — caller can feed
    /// this back as a per-region plasticity multiplier per spec §4.C.
    pub fn report_actual(&self, predicted: &Frame, actual: &Frame) -> f32 {
        frame_distance(predicted, actual)
    }

    pub fn snapshot(&self) -> crate::persistence::AnnealerSnapshot {
        let mut history: std::collections::HashMap<PoolId, std::collections::VecDeque<std::collections::HashMap<NeuronId, f32>>> = std::collections::HashMap::new();
        for (pid, frames) in self.history.iter() {
            let frames_std: std::collections::VecDeque<std::collections::HashMap<NeuronId, f32>> = frames
                .iter()
                .map(|f| f.iter().map(|(k, v)| (*k, *v)).collect())
                .collect();
            history.insert(*pid, frames_std);
        }
        crate::persistence::AnnealerSnapshot {
            config:  self.config.clone(),
            history,
        }
    }

    pub fn from_snapshot(snap: crate::persistence::AnnealerSnapshot) -> Self {
        let mut history = AHashMap::new();
        for (pid, frames) in snap.history {
            let dq: std::collections::VecDeque<Frame> = frames
                .into_iter()
                .map(|f| f.into_iter().collect::<Frame>())
                .collect();
            history.insert(pid, dq);
        }
        Self { config: snap.config, history }
    }
}

/// L2 distance over the union of neuron ids (missing keys count as 0).
fn frame_distance(a: &Frame, b: &Frame) -> f32 {
    let mut total = 0.0_f32;
    for (nid, &va) in a.iter() {
        let vb = b.get(nid).copied().unwrap_or(0.0);
        let d = va - vb;
        total += d * d;
    }
    for (nid, &vb) in b.iter() {
        if !a.contains_key(nid) {
            total += vb * vb;
        }
    }
    total.sqrt()
}

fn weighted_cost(state: &Frame, pairs: &[(&Frame, &Frame)], weights: &[f32]) -> f32 {
    // Weighted AVERAGE distance, not weighted sum.  Summing lets a
    // long tail of low-weight pairs drown out the high-weight signal
    // — the corpus would penalize accurate predictions for matching
    // the dominant pattern.  Averaging keeps the signal sharp.
    let mut weighted = 0.0_f32;
    let mut wsum = 0.0_f32;
    for ((_, next), &w) in pairs.iter().zip(weights.iter()) {
        weighted += w * frame_distance(state, next);
        wsum += w;
    }
    if wsum < 1e-9 { 0.0 } else { weighted / wsum }
}

/// Reference distance for confidence normalization: the mean pairwise
/// distance among historical next-frames.  If the corpus is tight,
/// baseline is small and predictions need to be very tight to score
/// high.  If the corpus is noisy, baseline is large and even
/// moderately-tight predictions earn confidence.
fn corpus_baseline(pairs: &[(&Frame, &Frame)]) -> f32 {
    if pairs.len() < 2 { return 1.0; }
    let mut total = 0.0_f32;
    let mut n = 0;
    for i in 0..pairs.len() {
        for j in (i + 1)..pairs.len() {
            total += frame_distance(pairs[i].1, pairs[j].1);
            n += 1;
        }
    }
    if n == 0 { 1.0 } else { total / n as f32 }
}
