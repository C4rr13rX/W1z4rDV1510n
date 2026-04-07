/// Intent inference engine.
///
/// Maps sequences of behavioural observations to intent distributions.
/// The engine is deliberately naïve on first run — it starts with a
/// prior over common intents and sharpens as motif-matching accumulates
/// evidence across frames.
///
/// Intent classes are ordered by threat level.  The engine works by scoring
/// each observable behavioural signal against the signature of each intent
/// class, then combining scores through a soft-max into a probability
/// distribution.  The Hebbian neural fabric (via the neuro bridge) eventually
/// learns these signatures organically — the explicit signatures here act as
/// priors that get refined over time.
///
/// Gas station robbery example
/// ────────────────────────────
/// T+0:  enter + pause → low SURVEY signal
/// T+10: exit-monitoring gaze → SURVEY confidence rises
/// T+15: hand moves toward waistband → CONCEAL signal fires
/// T+20: proxemics approach without product interaction → APPROACH_DEMAND
/// T+25: convergence of SURVEY + CONCEAL + APPROACH_DEMAND → ARMED_THREAT
/// T+30: wave function collapses — consensus confidence > threshold

use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Intent classes ───────────────────────────────────────────────────────────

/// Ordered by threat severity (0 = benign → high = imminent harm).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum IntentClass {
    /// Normal expected behaviour for entity type in this context.
    Normal = 0,
    /// Entity appears to be observing / assessing the environment.
    Survey = 1,
    /// Entity is approaching another without expected social cues.
    Approach = 2,
    /// Entity is concealing an object or their own behaviour.
    Conceal = 3,
    /// Entity is positioning for environmental control (exits, sightlines).
    ControlEnvironment = 4,
    /// Entity appears to be preparing a demand under duress.
    ApproachDemand = 5,
    /// Entity is in flight from a situation.
    Flee = 6,
    /// Entity presents a direct physical threat to another entity.
    DirectThreat = 7,
    /// Entity is armed and presenting an active threat.
    ArmedThreat = 8,
}

impl IntentClass {
    /// Threat severity scalar [0,1] corresponding to intent class.
    pub fn threat_level(self) -> f32 {
        match self {
            IntentClass::Normal           => 0.0,
            IntentClass::Survey           => 0.1,
            IntentClass::Approach         => 0.2,
            IntentClass::Conceal          => 0.4,
            IntentClass::ControlEnvironment => 0.45,
            IntentClass::ApproachDemand   => 0.6,
            IntentClass::Flee             => 0.5,
            IntentClass::DirectThreat     => 0.8,
            IntentClass::ArmedThreat      => 1.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            IntentClass::Normal             => "NORMAL",
            IntentClass::Survey             => "SURVEY",
            IntentClass::Approach           => "APPROACH",
            IntentClass::Conceal            => "CONCEAL",
            IntentClass::ControlEnvironment => "CONTROL_ENVIRONMENT",
            IntentClass::ApproachDemand     => "APPROACH_DEMAND",
            IntentClass::Flee               => "FLEE",
            IntentClass::DirectThreat       => "DIRECT_THREAT",
            IntentClass::ArmedThreat        => "ARMED_THREAT",
        }
    }
}

// ─── Predicted health impact ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedHealthImpact {
    /// Entity whose health is predicted to be affected.
    pub entity_id: String,
    /// Which health dimension is primarily impacted.
    pub dimension_index: usize,
    /// Predicted delta to the dimension [negative = damage].
    pub delta: f32,
    /// Seconds until impact is predicted.
    pub time_horizon_secs: f32,
    /// Confidence in this prediction [0,1].
    pub confidence: f32,
}

// ─── Behavioural signal ───────────────────────────────────────────────────────

/// A single observable behavioural signal with its strength.
#[derive(Debug, Clone)]
pub struct BehaviouralSignal {
    pub signal_type: BehaviouralSignalType,
    /// Signal strength [0,1].
    pub strength: f32,
    /// Confidence in the observation [0,1].
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BehaviouralSignalType {
    /// Pausing/dwelling longer than context norm.
    ExcessiveDwell,
    /// Gaze directed at exits rather than context-appropriate targets.
    ExitMonitoring,
    /// Scanning for observers / witnesses.
    WitnessScanning,
    /// Hand moved to concealment position (waistband, jacket, bag).
    HandConcealmentMove,
    /// Carried object detected as partially concealed.
    ConcealdObject,
    /// Approaching without socially-contextual purpose (no product interaction).
    PurposelessApproach,
    /// Body orientation staying toward exit while advancing.
    ExitOrientedAdvance,
    /// Normal product/context interaction.
    ContextInteraction,
    /// Running / rapid movement away.
    RapidFlight,
    /// Raised arm / aggressive gesture.
    AggressiveGesture,
    /// Stationary position block (blocking exit or counter).
    PositionalBlock,
    /// Voice / audio stress signal.
    VocalStress,
}

impl BehaviouralSignalType {
    /// Prior weight of this signal toward each intent class.
    /// Index: [Normal, Survey, Approach, Conceal, ControlEnv, ApproachDemand, Flee, DirectThreat, Armed]
    pub fn intent_weights(self) -> [f32; 9] {
        match self {
            BehaviouralSignalType::ExcessiveDwell       => [0.0, 0.7, 0.1, 0.1, 0.3, 0.2, 0.0, 0.0, 0.2],
            BehaviouralSignalType::ExitMonitoring       => [0.0, 0.5, 0.1, 0.2, 0.8, 0.4, 0.2, 0.1, 0.5],
            BehaviouralSignalType::WitnessScanning      => [0.0, 0.4, 0.0, 0.3, 0.7, 0.3, 0.0, 0.1, 0.4],
            BehaviouralSignalType::HandConcealmentMove  => [0.0, 0.1, 0.0, 1.0, 0.2, 0.3, 0.0, 0.3, 0.9],
            BehaviouralSignalType::ConcealdObject       => [0.0, 0.1, 0.0, 0.9, 0.1, 0.3, 0.0, 0.4, 1.0],
            BehaviouralSignalType::PurposelessApproach  => [0.0, 0.2, 0.8, 0.1, 0.3, 0.7, 0.0, 0.2, 0.4],
            BehaviouralSignalType::ExitOrientedAdvance  => [0.0, 0.2, 0.5, 0.1, 0.6, 0.5, 0.3, 0.2, 0.5],
            BehaviouralSignalType::ContextInteraction   => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            BehaviouralSignalType::RapidFlight          => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            BehaviouralSignalType::AggressiveGesture    => [0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.9, 0.7],
            BehaviouralSignalType::PositionalBlock      => [0.0, 0.1, 0.2, 0.0, 0.7, 0.4, 0.0, 0.2, 0.3],
            BehaviouralSignalType::VocalStress          => [0.0, 0.1, 0.1, 0.1, 0.1, 0.4, 0.2, 0.5, 0.4],
        }
    }
}

// ─── Intent signal ────────────────────────────────────────────────────────────

/// The output of the intent inference engine for one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSignal {
    pub entity_id: String,
    /// Probability distribution over all intent classes [0,1], sums to ≤1.
    pub distribution: HashMap<String, f32>,
    /// The most probable intent class.
    pub dominant_intent: IntentClass,
    /// Confidence in the dominant intent [0,1].
    pub dominant_confidence: f32,
    /// Aggregated threat level [0,1] — weighted sum of intent × threat_level.
    pub threat_score: f32,
    /// Target entities most likely affected by this intent.
    pub target_entities: Vec<String>,
    /// Predicted health impacts if intent is acted upon.
    pub predicted_impacts: Vec<PredictedHealthImpact>,
    pub timestamp: Timestamp,
}

impl IntentSignal {
    pub fn is_threat(&self) -> bool { self.threat_score > 0.3 }
    pub fn is_armed_threat(&self) -> bool { self.dominant_intent == IntentClass::ArmedThreat }
}

// ─── Per-entity intent tracker ────────────────────────────────────────────────

const SIGNAL_DECAY: f32 = 0.85; // how much signal persists across frames
const INTENT_CLASSES: [IntentClass; 9] = [
    IntentClass::Normal,
    IntentClass::Survey,
    IntentClass::Approach,
    IntentClass::Conceal,
    IntentClass::ControlEnvironment,
    IntentClass::ApproachDemand,
    IntentClass::Flee,
    IntentClass::DirectThreat,
    IntentClass::ArmedThreat,
];

#[derive(Debug)]
struct EntityIntentTracker {
    entity_id: String,
    /// Accumulated evidence weights for each intent class.
    scores: [f32; 9],
    /// Last output.
    last_signal: Option<IntentSignal>,
}

impl EntityIntentTracker {
    fn new(entity_id: impl Into<String>) -> Self {
        let mut scores = [0.0f32; 9];
        scores[0] = 0.5; // prior: entity is probably normal
        Self { entity_id: entity_id.into(), scores, last_signal: None }
    }

    fn observe(&mut self, signal: BehaviouralSignal) {
        let weights = signal.signal_type.intent_weights();
        let scale = signal.strength * signal.confidence;
        for (i, &w) in weights.iter().enumerate() {
            self.scores[i] = (self.scores[i] + w * scale * 0.3).clamp(0.0, 4.0);
        }
    }

    fn decay(&mut self) {
        for s in &mut self.scores { *s *= SIGNAL_DECAY; }
        self.scores[0] = self.scores[0].max(0.05); // Normal always has floor
    }

    fn build_signal(
        &mut self,
        timestamp: Timestamp,
        target_entities: Vec<String>,
    ) -> IntentSignal {
        self.decay();

        // Softmax over scores.
        let max = self.scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = self.scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        let mut distribution = HashMap::new();
        let mut dominant_idx = 0;
        let mut dominant_prob = 0.0f32;
        let mut threat_score = 0.0f32;

        for (i, (&prob, &cls)) in probs.iter().zip(INTENT_CLASSES.iter()).enumerate() {
            distribution.insert(cls.label().to_string(), prob);
            if prob > dominant_prob {
                dominant_prob = prob;
                dominant_idx = i;
            }
            threat_score += prob * cls.threat_level();
        }

        let dominant_intent = INTENT_CLASSES[dominant_idx];

        // Build predicted health impacts for high-threat intents.
        let predicted_impacts = if threat_score > 0.3 {
            build_health_impacts(&target_entities, threat_score, dominant_intent)
        } else {
            Vec::new()
        };

        let signal = IntentSignal {
            entity_id: self.entity_id.clone(),
            distribution,
            dominant_intent,
            dominant_confidence: dominant_prob,
            threat_score,
            target_entities,
            predicted_impacts,
            timestamp,
        };
        self.last_signal = Some(signal.clone());
        signal
    }
}

fn build_health_impacts(
    targets: &[String],
    threat_score: f32,
    intent: IntentClass,
) -> Vec<PredictedHealthImpact> {
    let mut impacts = Vec::new();
    let horizon = match intent {
        IntentClass::ArmedThreat    => 10.0,
        IntentClass::DirectThreat   => 15.0,
        IntentClass::ApproachDemand => 30.0,
        _                           => 60.0,
    };
    // SI damage is primary for physical threats.
    // AR drain is universal for any threat.
    for eid in targets {
        if threat_score > 0.5 {
            impacts.push(PredictedHealthImpact {
                entity_id: eid.clone(),
                dimension_index: 0, // SI
                delta: -threat_score * 0.7,
                time_horizon_secs: horizon,
                confidence: threat_score * 0.8,
            });
        }
        impacts.push(PredictedHealthImpact {
            entity_id: eid.clone(),
            dimension_index: 4, // AR
            delta: -threat_score * 0.4,
            time_horizon_secs: horizon * 0.5,
            confidence: threat_score * 0.9,
        });
        // TC disruption (rhythm breaks immediately under threat).
        impacts.push(PredictedHealthImpact {
            entity_id: eid.clone(),
            dimension_index: 5, // TC
            delta: -threat_score * 0.5,
            time_horizon_secs: 5.0,
            confidence: threat_score,
        });
    }
    impacts
}

// ─── Intent inference engine ──────────────────────────────────────────────────

/// Manages per-entity intent tracking and provides a scene-level intent map.
#[derive(Debug, Default)]
pub struct IntentInferenceEngine {
    trackers: HashMap<String, EntityIntentTracker>,
}

impl IntentInferenceEngine {
    /// Observe a set of behavioural signals for an entity.
    pub fn observe(
        &mut self,
        entity_id: &str,
        signals: Vec<BehaviouralSignal>,
    ) {
        let tracker = self.trackers
            .entry(entity_id.to_string())
            .or_insert_with(|| EntityIntentTracker::new(entity_id));
        for s in signals { tracker.observe(s); }
    }

    /// Ingest from token attributes — maps standard attribute keys to
    /// behavioural signal types automatically.
    pub fn ingest_from_attributes(
        &mut self,
        entity_id: &str,
        attrs: &HashMap<String, serde_json::Value>,
    ) {
        let mut signals = Vec::new();
        let mappings: &[(&str, BehaviouralSignalType, f32)] = &[
            ("exit_monitoring",     BehaviouralSignalType::ExitMonitoring,      1.0),
            ("witness_scan",        BehaviouralSignalType::WitnessScanning,     1.0),
            ("hand_conceal",        BehaviouralSignalType::HandConcealmentMove, 1.0),
            ("concealed_object",    BehaviouralSignalType::ConcealdObject,      1.0),
            ("purposeless_approach",BehaviouralSignalType::PurposelessApproach, 1.0),
            ("exit_advance",        BehaviouralSignalType::ExitOrientedAdvance, 1.0),
            ("context_interaction", BehaviouralSignalType::ContextInteraction,  1.0),
            ("rapid_flight",        BehaviouralSignalType::RapidFlight,         1.0),
            ("aggressive_gesture",  BehaviouralSignalType::AggressiveGesture,   1.0),
            ("positional_block",    BehaviouralSignalType::PositionalBlock,      1.0),
            ("vocal_stress",        BehaviouralSignalType::VocalStress,         0.8),
            ("excessive_dwell",     BehaviouralSignalType::ExcessiveDwell,      0.9),
        ];
        for (key, sig_type, conf_scale) in mappings {
            if let Some(v) = attrs.get(*key).and_then(|v| v.as_f64()) {
                if v > 0.05 {
                    signals.push(BehaviouralSignal {
                        signal_type: *sig_type,
                        strength: (v as f32).clamp(0.0, 1.0),
                        confidence: *conf_scale,
                    });
                }
            }
        }
        if !signals.is_empty() { self.observe(entity_id, signals); }
    }

    /// Produce intent signals for all tracked entities.
    pub fn compute(
        &mut self,
        timestamp: Timestamp,
        entity_pairs: &HashMap<String, Vec<String>>, // entity_id → nearby entity ids
    ) -> Vec<IntentSignal> {
        let ids: Vec<String> = self.trackers.keys().cloned().collect();
        ids.iter().map(|id| {
            let targets = entity_pairs.get(id).cloned().unwrap_or_default();
            self.trackers.get_mut(id).unwrap().build_signal(timestamp, targets)
        }).collect()
    }

    /// Returns a flat map entity_id → threat_score, consumed by the threat field.
    pub fn threat_map(&self) -> HashMap<String, f32> {
        self.trackers.iter().filter_map(|(id, t)| {
            t.last_signal.as_ref().map(|s| (id.clone(), s.threat_score))
        }).collect()
    }

    pub fn entity_count(&self) -> usize { self.trackers.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn armed_threat_signals_dominate() {
        let mut engine = IntentInferenceEngine::default();
        let ts = Timestamp { unix: 0 };
        // Simulate robbery build-up.
        engine.observe("robber", vec![
            BehaviouralSignal { signal_type: BehaviouralSignalType::ExcessiveDwell, strength: 0.8, confidence: 0.9 },
            BehaviouralSignal { signal_type: BehaviouralSignalType::ExitMonitoring, strength: 0.9, confidence: 0.9 },
            BehaviouralSignal { signal_type: BehaviouralSignalType::HandConcealmentMove, strength: 1.0, confidence: 0.95 },
            BehaviouralSignal { signal_type: BehaviouralSignalType::ConcealdObject, strength: 1.0, confidence: 1.0 },
            BehaviouralSignal { signal_type: BehaviouralSignalType::PurposelessApproach, strength: 0.9, confidence: 0.9 },
        ]);
        let mut pairs = HashMap::new();
        pairs.insert("robber".to_string(), vec!["clerk".to_string()]);
        let signals = engine.compute(ts, &pairs);
        let robber_sig = signals.iter().find(|s| s.entity_id == "robber").unwrap();
        assert!(robber_sig.threat_score > 0.4, "threat score should be elevated: {}", robber_sig.threat_score);
        assert!(!robber_sig.predicted_impacts.is_empty());
    }

    #[test]
    fn normal_shopping_is_low_threat() {
        let mut engine = IntentInferenceEngine::default();
        let ts = Timestamp { unix: 0 };
        engine.observe("shopper", vec![
            BehaviouralSignal { signal_type: BehaviouralSignalType::ContextInteraction, strength: 0.9, confidence: 1.0 },
        ]);
        let signals = engine.compute(ts, &HashMap::new());
        let s = signals.iter().find(|s| s.entity_id == "shopper").unwrap();
        assert!(s.threat_score < 0.2, "normal shopper should have low threat: {}", s.threat_score);
    }
}
