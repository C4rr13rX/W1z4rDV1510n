/// Multi-entity consensus aggregator and wave function collapse tracker.
///
/// The "wave function" metaphor
/// ─────────────────────────────
/// At any moment the joint future of all entities in a scene is a
/// probability distribution over possible outcomes.  Initially entropy
/// is high (many possible futures).  As behavioural evidence accumulates
/// the distribution narrows.  When entropy drops below a threshold fast
/// enough, the system declares the future has "collapsed" onto a small set
/// of outcomes — the threat is about to materialise.
///
/// Multi-entity consensus
/// ──────────────────────
/// Each entity model produces an independent threat estimate.  These are
/// combined using the complementary probability rule:
///
///   consensus = 1 - ∏(1 - estimate_i)
///
/// This is the same formula used to combine independent witnesses:
/// if three independent entities each estimate 50% probability of a threat,
/// the consensus is 1 - 0.5³ = 87.5%.  This mirrors exactly how a room
/// full of people simultaneously sensing danger amplifies the signal.

use crate::schema::Timestamp;
use crate::threat::intent::IntentSignal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ─── Wave function state ──────────────────────────────────────────────────────

/// A snapshot of the probability distribution over futures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveFunctionState {
    /// Shannon entropy of the distribution [0, 1].
    /// 1.0 = maximum uncertainty (many possible futures)
    /// 0.0 = fully collapsed (outcome determined)
    pub entropy: f32,
    /// Confidence in the single most probable future [0,1].
    pub dominant_confidence: f32,
    /// Rate of entropy change per frame (negative = collapsing fast).
    pub collapse_rate: f32,
    /// Whether the wave function has collapsed below the alarm threshold.
    pub collapsed: bool,
    /// Which observations caused the most recent entropy reduction.
    pub collapse_triggers: Vec<String>,
    /// The inferred dominant outcome label.
    pub dominant_outcome: String,
    pub timestamp: Timestamp,
}

impl WaveFunctionState {
    pub fn default_uncertain(timestamp: Timestamp) -> Self {
        Self {
            entropy: 1.0,
            dominant_confidence: 0.0,
            collapse_rate: 0.0,
            collapsed: false,
            collapse_triggers: Vec::new(),
            dominant_outcome: "unknown".to_string(),
            timestamp,
        }
    }
}

// ─── Consensus snapshot ───────────────────────────────────────────────────────

/// Full consensus result for the current frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusSnapshot {
    pub timestamp: Timestamp,
    /// Combined threat confidence across all entities [0,1].
    pub consensus_confidence: f32,
    /// Number of entities contributing to the consensus.
    pub entity_count: usize,
    /// Number of entities flagging elevated threat.
    pub threat_entity_count: usize,
    /// Wave function state.
    pub wave_function: WaveFunctionState,
    /// Per-entity threat scores.
    pub entity_scores: Vec<(String, f32)>,
    /// Whether this frame crosses the alarm threshold.
    pub alarm: bool,
    /// Alarm level: "low", "medium", "high", "critical".
    pub alarm_level: String,
}

impl ConsensusSnapshot {
    pub fn empty(timestamp: Timestamp) -> Self {
        Self {
            timestamp,
            consensus_confidence: 0.0,
            entity_count: 0,
            threat_entity_count: 0,
            wave_function: WaveFunctionState::default_uncertain(timestamp),
            entity_scores: Vec::new(),
            alarm: false,
            alarm_level: "none".to_string(),
        }
    }
}

// ─── Consensus engine ─────────────────────────────────────────────────────────

const ENTROPY_HISTORY: usize = 16;
const COLLAPSE_THRESHOLD: f32 = 0.35;
const COLLAPSE_RATE_THRESHOLD: f32 = -0.05; // entropy dropping > 5% per frame
const ALARM_THRESHOLD: f32 = 0.55;
const CRITICAL_THRESHOLD: f32 = 0.80;

#[derive(Debug)]
pub struct ConsensusEngine {
    /// Ring buffer of recent entropy values for computing collapse rate.
    entropy_history: VecDeque<f32>,
    /// Last computed consensus.
    last_consensus: Option<ConsensusSnapshot>,
    /// Minimum number of entities before consensus is meaningful.
    min_entities: usize,
}

impl Default for ConsensusEngine {
    fn default() -> Self {
        Self {
            entropy_history: VecDeque::with_capacity(ENTROPY_HISTORY),
            last_consensus: None,
            min_entities: 2,
        }
    }
}

impl ConsensusEngine {
    pub fn new(min_entities: usize) -> Self {
        Self { min_entities, ..Self::default() }
    }

    /// Compute consensus from the current frame's intent signals.
    pub fn compute(&mut self, signals: &[IntentSignal], timestamp: Timestamp) -> ConsensusSnapshot {
        if signals.is_empty() {
            return ConsensusSnapshot::empty(timestamp);
        }

        // Extract per-entity threat scores.
        let entity_scores: Vec<(String, f32)> = signals.iter()
            .map(|s| (s.entity_id.clone(), s.threat_score))
            .collect();
        let threat_entity_count = entity_scores.iter().filter(|(_, s)| *s > 0.15).count();

        // Complementary probability consensus.
        let consensus_confidence = if entity_scores.len() >= self.min_entities {
            let complement_product: f32 = entity_scores.iter()
                .map(|(_, s)| 1.0 - s.clamp(0.0, 1.0))
                .product();
            (1.0 - complement_product).clamp(0.0, 1.0)
        } else {
            // Single entity: use its score directly but with confidence penalty.
            entity_scores.iter().map(|(_, s)| *s).fold(0.0f32, f32::max) * 0.7
        };

        // Wave function entropy: treat as information entropy of the
        // "threat vs no-threat" Bernoulli distribution.
        let p = consensus_confidence.clamp(1e-6, 1.0 - 1e-6);
        let entropy = -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / 2.0f32.ln();

        // Update entropy history.
        if self.entropy_history.len() >= ENTROPY_HISTORY {
            self.entropy_history.pop_front();
        }
        self.entropy_history.push_back(entropy);

        // Compute collapse rate (linear regression slope on entropy history).
        let collapse_rate = compute_entropy_slope(&self.entropy_history);

        // Identify collapse triggers (signals that drove the most reduction).
        let collapse_triggers: Vec<String> = signals.iter()
            .filter(|s| s.threat_score > 0.3)
            .map(|s| format!(
                "{}:{} ({:.0}%)",
                s.entity_id,
                s.dominant_intent.label(),
                s.threat_score * 100.0,
            ))
            .collect();

        let dominant_outcome = infer_outcome(consensus_confidence, signals);
        let collapsed = entropy < COLLAPSE_THRESHOLD && collapse_rate < COLLAPSE_RATE_THRESHOLD;

        let wave_function = WaveFunctionState {
            entropy,
            dominant_confidence: consensus_confidence,
            collapse_rate,
            collapsed,
            collapse_triggers,
            dominant_outcome,
            timestamp,
        };

        let alarm = consensus_confidence >= ALARM_THRESHOLD || collapsed;
        let alarm_level = alarm_level(consensus_confidence, collapsed);

        let snapshot = ConsensusSnapshot {
            timestamp,
            consensus_confidence,
            entity_count: signals.len(),
            threat_entity_count,
            wave_function,
            entity_scores,
            alarm,
            alarm_level,
        };
        self.last_consensus = Some(snapshot.clone());
        snapshot
    }

    pub fn last(&self) -> Option<&ConsensusSnapshot> { self.last_consensus.as_ref() }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Compute slope of entropy over the history window using simple linear fit.
/// Negative slope = entropy is decreasing = wave function collapsing.
fn compute_entropy_slope(history: &VecDeque<f32>) -> f32 {
    let n = history.len();
    if n < 2 { return 0.0; }
    let n_f = n as f32;
    let x_mean = (n_f - 1.0) / 2.0;
    let y_mean: f32 = history.iter().sum::<f32>() / n_f;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &y) in history.iter().enumerate() {
        let x = i as f32;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean).powi(2);
    }
    if den.abs() < 1e-9 { 0.0 } else { num / den }
}

fn infer_outcome(confidence: f32, signals: &[IntentSignal]) -> String {
    if confidence < 0.15 { return "normal".to_string(); }
    if confidence < 0.35 { return "elevated_attention".to_string(); }
    if confidence < 0.55 { return "probable_threat".to_string(); }
    // Find the dominant intent across entities weighted by score.
    use crate::threat::intent::IntentClass;
    let armed = signals.iter().any(|s| s.dominant_intent == IntentClass::ArmedThreat);
    let direct = signals.iter().any(|s| s.dominant_intent == IntentClass::DirectThreat);
    if armed   { return "armed_threat_imminent".to_string(); }
    if direct  { return "physical_threat_imminent".to_string(); }
    "threat_imminent".to_string()
}

fn alarm_level(confidence: f32, collapsed: bool) -> String {
    if collapsed || confidence >= CRITICAL_THRESHOLD { return "critical".to_string(); }
    if confidence >= ALARM_THRESHOLD               { return "high".to_string(); }
    if confidence >= 0.35                          { return "medium".to_string(); }
    if confidence >= 0.15                          { return "low".to_string(); }
    "none".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threat::intent::{IntentClass, IntentSignal};
    use std::collections::HashMap;

    fn make_signal(entity_id: &str, threat_score: f32, intent: IntentClass) -> IntentSignal {
        IntentSignal {
            entity_id: entity_id.to_string(),
            distribution: HashMap::new(),
            dominant_intent: intent,
            dominant_confidence: threat_score,
            threat_score,
            target_entities: Vec::new(),
            predicted_impacts: Vec::new(),
            timestamp: Timestamp { unix: 0 },
        }
    }

    #[test]
    fn multiple_threat_entities_amplify_consensus() {
        let mut engine = ConsensusEngine::default();
        let ts = Timestamp { unix: 0 };
        let signals = vec![
            make_signal("e1", 0.5, IntentClass::ArmedThreat),
            make_signal("e2", 0.4, IntentClass::Survey),
            make_signal("e3", 0.5, IntentClass::Normal),
        ];
        let snap = engine.compute(&signals, ts);
        // Three 50% estimates → consensus > 75%
        assert!(snap.consensus_confidence > 0.6, "got {}", snap.consensus_confidence);
    }

    #[test]
    fn all_normal_no_alarm() {
        let mut engine = ConsensusEngine::default();
        let ts = Timestamp { unix: 0 };
        let signals = vec![
            make_signal("e1", 0.0, IntentClass::Normal),
            make_signal("e2", 0.05, IntentClass::Normal),
        ];
        let snap = engine.compute(&signals, ts);
        assert!(!snap.alarm);
        assert_eq!(snap.alarm_level, "none");
    }

    #[test]
    fn entropy_drops_on_repeated_threat() {
        let mut engine = ConsensusEngine::default();
        let ts = Timestamp { unix: 0 };
        let signals = vec![
            make_signal("e1", 0.9, IntentClass::ArmedThreat),
            make_signal("e2", 0.8, IntentClass::ArmedThreat),
        ];
        // Feed the same high-threat signal many times.
        for _ in 0..20 {
            engine.compute(&signals, ts);
        }
        let snap = engine.last().unwrap();
        assert!(snap.wave_function.entropy < 0.5, "entropy should be low: {}", snap.wave_function.entropy);
        assert!(snap.alarm);
    }
}
