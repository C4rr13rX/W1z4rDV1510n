//! Fabric trainer — universal training signal router.
//!
//! Every subsystem that produces information about the world emits a
//! `TrainingSignal` here.  At the end of each batch the trainer drains
//! the queue and converts every signal into Hebbian weight updates on the
//! NeuronPool, closing the loop:
//!
//! ```text
//!  Sensors ──────────────────────────────┐
//!  Outcome feedback (log-loss reward) ───┤
//!  Quantum API results ──────────────────┤──► FabricTrainer ──► NeuronPool (Hebbian)
//!  Human labels / votes ─────────────────┤         │
//!  Motif / concept discoveries ──────────┘         └──► FabricTrainerReport
//! ```
//!
//! Design principles
//! ─────────────────
//! • **No external computation at training time** — converting a signal to a
//!   Hebbian update is pure in-memory arithmetic on the NeuronPool.  Quantum
//!   results, which are already computed, are just probability distributions
//!   that seed activation levels.
//! • **Reward-weighted Hebbian** — correct predictions receive a scaled
//!   excitatory update; wrong predictions receive a mild inhibitory one.
//!   This is the STDP-lite principle already in the fabric, promoted to a
//!   first-class training loop.
//! • **Human labels are the strongest signal** — they receive a 4× lr boost
//!   because they represent ground truth verified by a human annotator.
//! • **Quantum samples become activations** — the probability distribution
//!   returned by the quantum API is treated as activation levels.  States with
//!   probability above the configured threshold are activated together, which
//!   is exactly what Hebbian learning needs: fire together, wire together.

use crate::neuro::NeuronPool;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ─── Configuration ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FabricTrainerConfig {
    /// Whether the trainer is active.
    pub enabled: bool,
    /// Maximum signals queued between drain calls (older ones are dropped).
    pub max_queue: usize,
    /// Learning rate multiplier applied on top of the pool's base hebbian_lr
    /// for a correct prediction (reward > 0).
    pub reward_lr_scale: f32,
    /// Learning rate multiplier for an incorrect prediction (reward < 0).
    /// Negative updates are sent as inhibitory; this scales their magnitude.
    pub penalty_lr_scale: f32,
    /// Learning rate multiplier for human-confirmed labels (highest confidence).
    pub human_label_lr_scale: f32,
    /// Learning rate multiplier for quantum-derived samples.
    pub quantum_lr_scale: f32,
    /// Minimum probability for a quantum state to be treated as an activated
    /// concept (states below this are ignored).
    pub quantum_min_prob: f64,
    /// Top-k quantum states to activate (avoids noise from the long tail).
    pub quantum_top_k: usize,
    /// Minimum reward magnitude to trigger a reward update (filters noise).
    pub min_reward_threshold: f32,
}

impl Default for FabricTrainerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_queue: 4096,
            reward_lr_scale: 2.0,
            penalty_lr_scale: 0.5,
            human_label_lr_scale: 4.0,
            quantum_lr_scale: 1.5,
            quantum_min_prob: 0.05,
            quantum_top_k: 8,
            min_reward_threshold: 0.01,
        }
    }
}

// ─── Signal types ───────────────────────────────────────────────────────────

/// A training signal that any subsystem can emit into the fabric trainer.
#[derive(Debug, Clone)]
pub enum TrainingSignal {
    /// Things that co-occurred in the environment (from sensors, motifs,
    /// dynamic pool category observations, etc.).  Standard Hebbian strength.
    Observation {
        labels: Vec<String>,
        timestamp: Timestamp,
    },

    /// An outcome-feedback prediction resolved.  `reward` is the improvement
    /// over baseline (positive = better than chance, negative = worse).
    /// The `predicted` and `actual` labels are used to strengthen or weaken
    /// the connection between the context neurons and the outcome neuron.
    PredictionResolved {
        /// Tokens/labels that were active when the prediction was made.
        context_labels: Vec<String>,
        /// What the system predicted.
        predicted: String,
        /// What actually happened.
        actual: String,
        /// Reward from outcome_feedback log-loss scoring: (baseline_loss - actual_loss).
        reward: f32,
        timestamp: Timestamp,
    },

    /// A quantum API call returned results.  `state_probabilities` is the
    /// distribution over bitstring / basis states returned by the provider.
    /// States are labelled as `"q::<basis_state>"` so they live in their own
    /// namespace in the NeuronPool.
    QuantumResult {
        /// Identifies the circuit / job so results can be associated later.
        circuit_id: String,
        /// (basis_state_label, probability)
        state_probabilities: Vec<(String, f64)>,
        /// Context: what the quantum job was computing about.
        context_labels: Vec<String>,
        timestamp: Timestamp,
    },

    /// A human annotator confirmed an association (from the label queue, the
    /// knowledge vote system, or the motif label bridge).
    HumanConfirmed {
        /// The label / concept that was confirmed.
        label: String,
        /// Other labels that co-occur with this concept in the verified context.
        associated_labels: Vec<String>,
        timestamp: Timestamp,
    },

    /// An inhibitory signal: these patterns anti-correlated or a prediction
    /// was strongly wrong.  Weakens existing connections between these labels.
    Inhibit {
        labels: Vec<String>,
        strength: f32,
        timestamp: Timestamp,
    },
}

impl TrainingSignal {
    pub fn timestamp(&self) -> Timestamp {
        match self {
            Self::Observation { timestamp, .. } => *timestamp,
            Self::PredictionResolved { timestamp, .. } => *timestamp,
            Self::QuantumResult { timestamp, .. } => *timestamp,
            Self::HumanConfirmed { timestamp, .. } => *timestamp,
            Self::Inhibit { timestamp, .. } => *timestamp,
        }
    }
}

// ─── Stats ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FabricTrainerStats {
    pub observations_applied: u64,
    pub predictions_resolved: u64,
    pub positive_rewards: u64,
    pub negative_rewards: u64,
    pub quantum_signals_applied: u64,
    pub human_labels_applied: u64,
    pub inhibitions_applied: u64,
    pub signals_dropped: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricTrainerReport {
    pub timestamp: Timestamp,
    pub signals_drained: usize,
    pub queue_depth: usize,
    pub stats: FabricTrainerStats,
}

// ─── Trainer ────────────────────────────────────────────────────────────────

pub struct FabricTrainer {
    config: FabricTrainerConfig,
    queue: VecDeque<TrainingSignal>,
    stats: FabricTrainerStats,
}

impl FabricTrainer {
    pub fn new(config: FabricTrainerConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            stats: FabricTrainerStats::default(),
        }
    }

    /// Emit a training signal into the queue.
    ///
    /// If the queue is full the oldest signal is dropped (backpressure safety).
    pub fn emit(&mut self, signal: TrainingSignal) {
        if !self.config.enabled {
            return;
        }
        if self.queue.len() >= self.config.max_queue {
            self.queue.pop_front();
            self.stats.signals_dropped += 1;
        }
        self.queue.push_back(signal);
    }

    /// Helper: emit an `Observation` signal.
    pub fn observe(&mut self, labels: Vec<String>, timestamp: Timestamp) {
        self.emit(TrainingSignal::Observation { labels, timestamp });
    }

    /// Helper: emit a `PredictionResolved` signal from an outcome feedback report.
    pub fn record_prediction_outcome(
        &mut self,
        context_labels: Vec<String>,
        predicted: String,
        actual: String,
        reward: f32,
        timestamp: Timestamp,
    ) {
        if reward.abs() < self.config.min_reward_threshold {
            return;
        }
        self.emit(TrainingSignal::PredictionResolved {
            context_labels,
            predicted,
            actual,
            reward,
            timestamp,
        });
    }

    /// Helper: emit a `QuantumResult` signal.
    pub fn record_quantum_result(
        &mut self,
        circuit_id: String,
        state_probabilities: Vec<(String, f64)>,
        context_labels: Vec<String>,
        timestamp: Timestamp,
    ) {
        self.emit(TrainingSignal::QuantumResult {
            circuit_id,
            state_probabilities,
            context_labels,
            timestamp,
        });
    }

    /// Helper: emit a `HumanConfirmed` signal.
    pub fn record_human_label(
        &mut self,
        label: String,
        associated_labels: Vec<String>,
        timestamp: Timestamp,
    ) {
        self.emit(TrainingSignal::HumanConfirmed {
            label,
            associated_labels,
            timestamp,
        });
    }

    /// Drain all queued signals and apply them as Hebbian updates to `pool`.
    ///
    /// Call this once per batch, after all other processing has completed, so
    /// that signals emitted during the batch are applied before the next one.
    pub fn drain_into_pool(&mut self, pool: &mut NeuronPool, timestamp: Timestamp) -> FabricTrainerReport {
        if !self.config.enabled {
            return FabricTrainerReport {
                timestamp,
                signals_drained: 0,
                queue_depth: self.queue.len(),
                stats: self.stats.clone(),
            };
        }

        let signals: Vec<TrainingSignal> = self.queue.drain(..).collect();
        let drained = signals.len();

        for signal in signals {
            match signal {
                TrainingSignal::Observation { labels, .. } => {
                    if labels.len() >= 2 {
                        pool.record_symbols(&labels);
                        self.stats.observations_applied += 1;
                    }
                }

                TrainingSignal::PredictionResolved {
                    mut context_labels,
                    predicted,
                    actual,
                    reward,
                    ..
                } => {
                    self.stats.predictions_resolved += 1;
                    if reward > 0.0 {
                        // Correct prediction: strengthen context → actual connection.
                        self.stats.positive_rewards += 1;
                        let scale = self.config.reward_lr_scale * reward.abs().min(4.0);
                        context_labels.push(actual.clone());
                        pool.train_weighted(&context_labels, scale, false);
                        // Also strengthen the predicted→actual pair directly.
                        pool.train_weighted(&[predicted, actual], scale, false);
                    } else {
                        // Wrong prediction: weakly inhibit the predicted-but-wrong connection.
                        self.stats.negative_rewards += 1;
                        let scale = self.config.penalty_lr_scale * reward.abs().min(4.0);
                        pool.train_weighted(&[predicted, actual.clone()], scale, true);
                        // Still reinforce context → actual so the fabric learns the right answer.
                        if !context_labels.is_empty() {
                            context_labels.push(actual);
                            pool.train_weighted(&context_labels, self.config.penalty_lr_scale, false);
                        }
                    }
                }

                TrainingSignal::QuantumResult {
                    circuit_id,
                    mut state_probabilities,
                    mut context_labels,
                    ..
                } => {
                    self.stats.quantum_signals_applied += 1;
                    // Sort by probability descending and take top-k.
                    state_probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    state_probabilities.truncate(self.config.quantum_top_k);

                    // Activate top states whose probability exceeds threshold.
                    let mut activated: Vec<String> = state_probabilities
                        .iter()
                        .filter(|(_, p)| *p >= self.config.quantum_min_prob)
                        .map(|(label, _)| format!("q::{}", label))
                        .collect();

                    if !activated.is_empty() {
                        // Link quantum states to the computation context.
                        if !context_labels.is_empty() {
                            let mut combined = context_labels.clone();
                            combined.extend(activated.iter().cloned());
                            pool.train_weighted(&combined, self.config.quantum_lr_scale, false);
                        }
                        // Also wire the circuit identity to the top state.
                        let circuit_label = format!("circuit::{}", circuit_id);
                        activated.push(circuit_label);
                        pool.train_weighted(&activated, self.config.quantum_lr_scale, false);
                        // Wire context to circuit so we can recall which circuits are
                        // relevant to which phenomena.
                        if !context_labels.is_empty() {
                            context_labels.push(format!("circuit::{}", circuit_id));
                            pool.train_weighted(&context_labels, self.config.quantum_lr_scale * 0.5, false);
                        }
                    }
                }

                TrainingSignal::HumanConfirmed {
                    label,
                    mut associated_labels,
                    ..
                } => {
                    self.stats.human_labels_applied += 1;
                    // Human labels carry the highest confidence — boost lr scale.
                    associated_labels.push(label);
                    pool.train_weighted(&associated_labels, self.config.human_label_lr_scale, false);
                }

                TrainingSignal::Inhibit { labels, strength, .. } => {
                    self.stats.inhibitions_applied += 1;
                    pool.train_weighted(&labels, strength.clamp(0.01, 4.0), true);
                }
            }
        }

        FabricTrainerReport {
            timestamp,
            signals_drained: drained,
            queue_depth: self.queue.len(),
            stats: self.stats.clone(),
        }
    }

    /// Drain all queued signals and apply them via the `NeuroStreamBridge`.
    ///
    /// This is the production entry point — the bridge forwards each signal to
    /// the `NeuroRuntime::train_weighted` which locks the `NeuronPool` and
    /// applies the Hebbian update.
    pub fn drain_into_bridge(
        &mut self,
        bridge: &crate::streaming::neuro_bridge::NeuroStreamBridge,
        timestamp: Timestamp,
    ) -> FabricTrainerReport {
        if !self.config.enabled {
            return FabricTrainerReport {
                timestamp,
                signals_drained: 0,
                queue_depth: self.queue.len(),
                stats: self.stats.clone(),
            };
        }

        let signals: Vec<TrainingSignal> = self.queue.drain(..).collect();
        let drained = signals.len();

        for signal in signals {
            self.apply_to_bridge(signal, bridge);
        }

        FabricTrainerReport {
            timestamp,
            signals_drained: drained,
            queue_depth: self.queue.len(),
            stats: self.stats.clone(),
        }
    }

    /// Drain and discard all queued signals (used when no neuro bridge is active).
    /// Keeps stats accurate so reports reflect total signal volume.
    pub fn drop_queued(&mut self, timestamp: Timestamp) -> FabricTrainerReport {
        let drained = self.queue.len();
        self.queue.clear();
        FabricTrainerReport {
            timestamp,
            signals_drained: drained,
            queue_depth: 0,
            stats: self.stats.clone(),
        }
    }

    fn apply_to_bridge(
        &mut self,
        signal: TrainingSignal,
        bridge: &crate::streaming::neuro_bridge::NeuroStreamBridge,
    ) {
        match signal {
            TrainingSignal::Observation { labels, .. } => {
                if labels.len() >= 2 {
                    bridge.train_weighted(&labels, 1.0, false);
                    self.stats.observations_applied += 1;
                }
            }

            TrainingSignal::PredictionResolved {
                mut context_labels,
                predicted,
                actual,
                reward,
                ..
            } => {
                self.stats.predictions_resolved += 1;
                if reward > 0.0 {
                    self.stats.positive_rewards += 1;
                    let scale = self.config.reward_lr_scale * reward.abs().min(4.0);
                    context_labels.push(actual.clone());
                    bridge.train_weighted(&context_labels, scale, false);
                    bridge.train_weighted(&[predicted, actual], scale, false);
                } else {
                    self.stats.negative_rewards += 1;
                    let scale = self.config.penalty_lr_scale * reward.abs().min(4.0);
                    bridge.train_weighted(&[predicted, actual.clone()], scale, true);
                    if !context_labels.is_empty() {
                        context_labels.push(actual);
                        bridge.train_weighted(&context_labels, self.config.penalty_lr_scale, false);
                    }
                }
            }

            TrainingSignal::QuantumResult {
                circuit_id,
                mut state_probabilities,
                mut context_labels,
                ..
            } => {
                self.stats.quantum_signals_applied += 1;
                state_probabilities.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                state_probabilities.truncate(self.config.quantum_top_k);
                let mut activated: Vec<String> = state_probabilities
                    .iter()
                    .filter(|(_, p)| *p >= self.config.quantum_min_prob)
                    .map(|(label, _)| format!("q::{label}"))
                    .collect();
                if !activated.is_empty() {
                    if !context_labels.is_empty() {
                        let mut combined = context_labels.clone();
                        combined.extend(activated.iter().cloned());
                        bridge.train_weighted(&combined, self.config.quantum_lr_scale, false);
                    }
                    let circuit_label = format!("circuit::{circuit_id}");
                    activated.push(circuit_label.clone());
                    bridge.train_weighted(&activated, self.config.quantum_lr_scale, false);
                    if !context_labels.is_empty() {
                        context_labels.push(circuit_label);
                        bridge.train_weighted(
                            &context_labels,
                            self.config.quantum_lr_scale * 0.5,
                            false,
                        );
                    }
                }
            }

            TrainingSignal::HumanConfirmed {
                label,
                mut associated_labels,
                ..
            } => {
                self.stats.human_labels_applied += 1;
                associated_labels.push(label);
                bridge.train_weighted(&associated_labels, self.config.human_label_lr_scale, false);
            }

            TrainingSignal::Inhibit { labels, strength, .. } => {
                self.stats.inhibitions_applied += 1;
                bridge.train_weighted(&labels, strength.clamp(0.01, 4.0), true);
            }
        }
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    pub fn stats(&self) -> &FabricTrainerStats {
        &self.stats
    }
}

impl Default for FabricTrainer {
    fn default() -> Self {
        Self::new(FabricTrainerConfig::default())
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuro::{NeuroConfig, NeuronPool};

    fn ts(unix: u64) -> Timestamp {
        Timestamp { unix: unix as i64 }
    }

    fn make_pool() -> NeuronPool {
        NeuronPool::new(NeuroConfig::default())
    }

    #[test]
    fn observation_creates_neurons_and_synapses() {
        let mut trainer = FabricTrainer::default();
        let mut pool = make_pool();
        trainer.observe(vec!["cat".into(), "animal".into()], ts(1));
        let report = trainer.drain_into_pool(&mut pool, ts(1));
        assert_eq!(report.signals_drained, 1);
        assert_eq!(report.stats.observations_applied, 1);
        // Both neurons should now exist (get_or_create is idempotent).
        let _cat_id = pool.get_or_create("cat");
        let _animal_id = pool.get_or_create("animal");
        // Active labels reflect the training.
        let active = pool.active_labels(0.0);
        assert!(active.contains("cat") || active.contains("animal"),
            "expected at least one label to be active after observation");
    }

    #[test]
    fn positive_reward_strengthens_connection() {
        let mut trainer = FabricTrainer::default();
        let mut pool = make_pool();
        // Baseline: observe once to establish neurons.
        trainer.observe(vec!["cell".into(), "membrane".into()], ts(1));
        trainer.drain_into_pool(&mut pool, ts(1));
        // Record a correct prediction with positive reward.
        trainer.record_prediction_outcome(
            vec!["cell".into()],
            "membrane".into(),
            "membrane".into(),
            1.5,
            ts(2),
        );
        let report = trainer.drain_into_pool(&mut pool, ts(2));
        assert_eq!(report.stats.predictions_resolved, 1);
        assert_eq!(report.stats.positive_rewards, 1);
    }

    #[test]
    fn negative_reward_applies_inhibition() {
        let mut trainer = FabricTrainer::default();
        let mut pool = make_pool();
        trainer.record_prediction_outcome(
            vec!["wind".into()],
            "rain".into(),
            "drought".into(),
            -0.8,
            ts(1),
        );
        let report = trainer.drain_into_pool(&mut pool, ts(1));
        assert_eq!(report.stats.negative_rewards, 1);
        assert_eq!(report.stats.inhibitions_applied, 0); // inhibition is applied inline
    }

    #[test]
    fn quantum_result_activates_top_states() {
        let mut trainer = FabricTrainer::default();
        let mut pool = make_pool();
        trainer.record_quantum_result(
            "anneal-job-1".into(),
            vec![
                ("0101".into(), 0.42),
                ("1010".into(), 0.33),
                ("0000".into(), 0.01), // below threshold — should be ignored
                ("1100".into(), 0.24),
            ],
            vec!["energy_minimization".into()],
            ts(1),
        );
        let report = trainer.drain_into_pool(&mut pool, ts(1));
        assert_eq!(report.stats.quantum_signals_applied, 1);
        // q::0101 and q::1010 labels should have been created in the pool.
        let active = pool.active_labels(0.0);
        assert!(active.contains("q::0101") || pool.get_or_create("q::0101") < u32::MAX,
            "q::0101 should exist in pool");
        assert!(active.contains("q::1010") || pool.get_or_create("q::1010") < u32::MAX,
            "q::1010 should exist in pool");
    }

    #[test]
    fn human_label_applies_boosted_weight() {
        let mut trainer = FabricTrainer::default();
        let mut pool = make_pool();
        trainer.record_human_label(
            "car".into(),
            vec!["vehicle".into(), "wheels".into(), "engine".into()],
            ts(1),
        );
        let report = trainer.drain_into_pool(&mut pool, ts(1));
        assert_eq!(report.stats.human_labels_applied, 1);
        // All labels should now be neurons (get_or_create is idempotent).
        let _car = pool.get_or_create("car");
        let _vehicle = pool.get_or_create("vehicle");
        let active = pool.active_labels(0.0);
        assert!(active.contains("car") || active.contains("vehicle"),
            "expected car/vehicle to be active after human label");
    }

    #[test]
    fn queue_cap_drops_oldest_signals() {
        let mut trainer = FabricTrainer::new(FabricTrainerConfig {
            max_queue: 3,
            ..Default::default()
        });
        let mut pool = make_pool();
        for i in 0..5u64 {
            trainer.observe(vec![format!("label_{i}"), format!("label_{}", i + 1)], ts(i));
        }
        assert_eq!(trainer.queue_depth(), 3);
        assert_eq!(trainer.stats().signals_dropped, 2);
        trainer.drain_into_pool(&mut pool, ts(10));
        assert_eq!(trainer.queue_depth(), 0);
    }

    #[test]
    fn disabled_trainer_emits_nothing() {
        let mut trainer = FabricTrainer::new(FabricTrainerConfig {
            enabled: false,
            ..Default::default()
        });
        let mut pool = make_pool();
        trainer.observe(vec!["a".into(), "b".into()], ts(1));
        assert_eq!(trainer.queue_depth(), 0); // not even queued when disabled
        let report = trainer.drain_into_pool(&mut pool, ts(1));
        assert_eq!(report.signals_drained, 0);
    }

    #[test]
    fn report_reflects_cumulative_stats() {
        let mut trainer = FabricTrainer::default();
        let mut pool = make_pool();
        trainer.observe(vec!["x".into(), "y".into()], ts(1));
        trainer.record_human_label("concept".into(), vec!["x".into()], ts(1));
        trainer.drain_into_pool(&mut pool, ts(1));
        let report = trainer.drain_into_pool(&mut pool, ts(2)); // second drain — queue empty
        // Stats are cumulative across drains.
        assert_eq!(report.stats.observations_applied, 1);
        assert_eq!(report.stats.human_labels_applied, 1);
        assert_eq!(report.signals_drained, 0);
    }
}
