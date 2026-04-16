//! Prediction experiment framework — autonomous self-testing against the future.
//!
//! The framework registers forward predictions ("entity E will have health
//! vector V at time T") and, when the horizon elapses, compares them against
//! the actual observed state.
//!
//! Accuracy is tracked per dimension and in aggregate.  When mean error stays
//! below `accuracy_target` for `scale_window` consecutive evaluations, the
//! prediction horizon is expanded by `horizon_scale_factor`.  When error rises
//! above `error_floor_for_reset` the horizon contracts back to the initial
//! value so the system re-calibrates at a range it can actually predict.
//!
//! The error signal is exposed as `ExperimentReport.mean_error` per dimension
//! so the API layer can drive Hebbian plasticity signals proportional to
//! prediction miss.
//!
//! The EEM minimum-confidence constraint is enforced by the caller: only
//! register predictions for entities where EEM confidence > threshold.

use crate::schema::Timestamp;
use crate::streaming::entity_health::HealthVector;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionExperimentConfig {
    /// Starting prediction horizon in seconds.
    pub initial_horizon_secs: f64,
    /// Multiply horizon by this factor when accuracy threshold is met.
    pub horizon_scale_factor: f64,
    /// Maximum prediction horizon the system will attempt.
    pub max_horizon_secs: f64,
    /// Mean per-dimension error (0–1) below which the horizon expands.
    pub accuracy_target: f64,
    /// Mean error above which the horizon contracts back to initial.
    pub error_floor_for_reset: f64,
    /// Number of consecutive evaluations below `accuracy_target` required
    /// before expanding the horizon.
    pub scale_window: usize,
    /// Maximum number of pending (unresolved) predictions to keep.
    pub max_pending: usize,
}

impl Default for PredictionExperimentConfig {
    fn default() -> Self {
        Self {
            initial_horizon_secs:   5.0,
            horizon_scale_factor:   1.5,
            max_horizon_secs:       300.0,
            accuracy_target:        0.08,
            error_floor_for_reset:  0.25,
            scale_window:           5,
            max_pending:            256,
        }
    }
}

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingPrediction {
    pub id:          String,
    pub entity_id:   String,
    /// Unix timestamp when this prediction was registered.
    pub registered:  Timestamp,
    /// Unix timestamp after which the prediction is evaluated.
    pub due_at:      Timestamp,
    pub horizon_secs: f64,
    pub predicted:   HealthVector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatedPrediction {
    pub id:           String,
    pub entity_id:    String,
    pub horizon_secs: f64,
    pub predicted:    HealthVector,
    pub actual:       HealthVector,
    /// Per-dimension absolute error (|predicted - actual|).
    pub error:        HealthVector,
    /// Mean absolute error across all six dimensions.
    pub mean_error:   f64,
    pub registered:   Timestamp,
    pub evaluated_at: Timestamp,
}

/// Current state of the experiment framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub timestamp:           Timestamp,
    /// Number of predictions currently waiting to be evaluated.
    pub pending_count:       usize,
    /// Number of evaluations completed so far.
    pub evaluated_count:     usize,
    /// Current prediction horizon being tested.
    pub current_horizon_secs: f64,
    /// Mean absolute error across all recently evaluated predictions (0–1).
    pub mean_error_overall:  f64,
    /// Per-dimension mean absolute error for the current evaluation window.
    pub mean_error_by_dim:   [f64; 6],
    /// True if the horizon was expanded this frame.
    pub horizon_expanded:    bool,
    /// True if the horizon was reset this frame.
    pub horizon_reset:       bool,
    /// Last N evaluated predictions.
    pub recent_evaluations:  Vec<EvaluatedPrediction>,
}

// ── Runtime ───────────────────────────────────────────────────────────────────

pub struct PredictionExperimentRuntime {
    config:           PredictionExperimentConfig,
    pending:          Vec<PendingPrediction>,
    eval_window:      VecDeque<EvaluatedPrediction>,
    current_horizon:  f64,
    /// Recent error values for horizon adaptation.
    error_history:    VecDeque<f64>,
    /// Count of consecutive evaluations below accuracy_target.
    below_target_run: usize,
    evaluated_total:  usize,
}

impl PredictionExperimentRuntime {
    pub fn new(config: PredictionExperimentConfig) -> Self {
        let initial = config.initial_horizon_secs;
        Self {
            config,
            pending:          Vec::new(),
            eval_window:      VecDeque::new(),
            current_horizon:  initial,
            error_history:    VecDeque::new(),
            below_target_run: 0,
            evaluated_total:  0,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Register a prediction.  The caller decides whether to predict; this
    /// method records it.  Returns the prediction id.
    pub fn register(
        &mut self,
        entity_id: String,
        predicted: HealthVector,
        now: Timestamp,
    ) -> String {
        // Evict oldest if at capacity.
        while self.pending.len() >= self.config.max_pending {
            self.pending.remove(0);
        }
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = format!("pred-{}-{}", now.unix, COUNTER.fetch_add(1, Ordering::Relaxed));
        let due = Timestamp { unix: now.unix + self.current_horizon as i64 };
        self.pending.push(PendingPrediction {
            id: id.clone(),
            entity_id,
            registered: now,
            due_at: due,
            horizon_secs: self.current_horizon,
            predicted,
        });
        id
    }

    /// Advance the frame: resolve all due predictions against the provided
    /// actual health vectors.  Returns the ExperimentReport.
    ///
    /// `actual_states` maps entity_id → current health vector.
    pub fn update(
        &mut self,
        now: Timestamp,
        actual_states: &std::collections::HashMap<String, HealthVector>,
    ) -> ExperimentReport {
        let mut horizon_expanded = false;
        let mut horizon_reset    = false;
        let mut new_evals: Vec<EvaluatedPrediction> = Vec::new();

        // Evaluate all due predictions.
        let (due, pending): (Vec<_>, Vec<_>) = self.pending.drain(..)
            .partition(|p| now.unix >= p.due_at.unix);
        self.pending = pending;

        for pred in due {
            let Some(actual) = actual_states.get(&pred.entity_id) else { continue };
            let error  = abs_error(&pred.predicted, actual);
            let mean_e = error.composite();

            new_evals.push(EvaluatedPrediction {
                id:           pred.id.clone(),
                entity_id:    pred.entity_id.clone(),
                horizon_secs: pred.horizon_secs,
                predicted:    pred.predicted.clone(),
                actual:       actual.clone(),
                error:        error.clone(),
                mean_error:   mean_e,
                registered:   pred.registered,
                evaluated_at: now,
            });

            self.error_history.push_back(mean_e);
            while self.error_history.len() > self.config.scale_window * 2 {
                self.error_history.pop_front();
            }
            self.evaluated_total += 1;
        }

        // Update eval window.
        for ev in &new_evals {
            self.eval_window.push_back(ev.clone());
        }
        while self.eval_window.len() > 32 {
            self.eval_window.pop_front();
        }

        // Horizon adaptation.
        let window_errors: Vec<f64> = self.error_history.iter()
            .rev().take(self.config.scale_window).copied().collect();

        if !window_errors.is_empty() {
            let recent_mean: f64 = window_errors.iter().sum::<f64>() / window_errors.len() as f64;

            if recent_mean > self.config.error_floor_for_reset {
                // Too many errors — reset to initial horizon.
                if self.current_horizon > self.config.initial_horizon_secs * 1.1 {
                    self.current_horizon  = self.config.initial_horizon_secs;
                    self.below_target_run = 0;
                    self.error_history.clear();
                    horizon_reset = true;
                }
            } else if recent_mean < self.config.accuracy_target {
                self.below_target_run += 1;
                if self.below_target_run >= self.config.scale_window {
                    let new_h = (self.current_horizon * self.config.horizon_scale_factor)
                        .min(self.config.max_horizon_secs);
                    if new_h > self.current_horizon + 0.01 {
                        self.current_horizon  = new_h;
                        self.below_target_run = 0;
                        self.error_history.clear();
                        horizon_expanded = true;
                    }
                }
            } else {
                // Neither good nor bad — reset the consecutive-good counter.
                self.below_target_run = 0;
            }
        }

        // Build mean error summary.
        let mean_error_overall = if !self.eval_window.is_empty() {
            self.eval_window.iter().map(|e| e.mean_error).sum::<f64>()
                / self.eval_window.len() as f64
        } else { 0.0 };

        let mean_error_by_dim = dim_mean_errors(&self.eval_window);

        ExperimentReport {
            timestamp: now,
            pending_count: self.pending.len(),
            evaluated_count: self.evaluated_total,
            current_horizon_secs: self.current_horizon,
            mean_error_overall,
            mean_error_by_dim,
            horizon_expanded,
            horizon_reset,
            recent_evaluations: self.eval_window.iter().rev().take(8).cloned().collect(),
        }
    }

    pub fn current_horizon_secs(&self) -> f64 { self.current_horizon }
    pub fn pending_count(&self) -> usize       { self.pending.len() }
    pub fn evaluated_count(&self) -> usize     { self.evaluated_total }
}

impl Default for PredictionExperimentRuntime {
    fn default() -> Self { Self::new(PredictionExperimentConfig::default()) }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn abs_error(a: &HealthVector, b: &HealthVector) -> HealthVector {
    let av = a.as_array();
    let bv = b.as_array();
    let e: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| (x - y).abs()).collect();
    HealthVector {
        structural: e[0], energetic: e[1], temporal: e[2],
        intentional: e[3], environmental: e[4], informational: e[5],
    }
}

fn dim_mean_errors(window: &VecDeque<EvaluatedPrediction>) -> [f64; 6] {
    if window.is_empty() { return [0.0; 6]; }
    let n = window.len() as f64;
    let sums = window.iter().fold([0.0_f64; 6], |mut acc, ev| {
        let arr = ev.error.as_array();
        for (i, &v) in arr.iter().enumerate() { acc[i] += v; }
        acc
    });
    [sums[0]/n, sums[1]/n, sums[2]/n, sums[3]/n, sums[4]/n, sums[5]/n]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn health(v: f64) -> HealthVector {
        HealthVector { structural: v, energetic: v, temporal: v,
            intentional: v, environmental: v, informational: v }
    }

    #[test]
    fn register_then_evaluate() {
        let mut rt = PredictionExperimentRuntime::default();
        let now = Timestamp { unix: 0 };
        let pred = health(0.7);
        let id = rt.register("e1".into(), pred.clone(), now);
        assert_eq!(rt.pending_count(), 1);

        // Advance past horizon.
        let future = Timestamp { unix: rt.current_horizon_secs() as i64 + 1 };
        let mut actual = HashMap::new();
        actual.insert("e1".to_string(), health(0.75)); // small error
        let rep = rt.update(future, &actual);
        assert_eq!(rep.evaluated_count, 1);
        assert_eq!(rep.pending_count, 0);
        assert!(rep.mean_error_overall < 0.1);
    }

    #[test]
    fn horizon_expands_after_consistent_accuracy() {
        let config = PredictionExperimentConfig {
            initial_horizon_secs: 1.0,
            accuracy_target: 0.15,
            scale_window: 3,
            horizon_scale_factor: 2.0,
            ..Default::default()
        };
        let mut rt = PredictionExperimentRuntime::new(config);
        let mut actual = HashMap::new();
        actual.insert("e".to_string(), health(0.5));

        let initial_horizon = rt.current_horizon_secs();
        let mut expanded = false;
        for i in 0..10i64 {
            rt.register("e".into(), health(0.51), Timestamp { unix: i });
            let rep = rt.update(Timestamp { unix: i + 2 }, &actual);
            if rep.horizon_expanded { expanded = true; break; }
        }
        assert!(expanded || rt.current_horizon_secs() > initial_horizon,
            "horizon should have expanded with consistent low error");
    }
}
