/// Threat analysis sub-system.
///
/// Provides a full pipeline from raw sensor attributes to a scene-level threat
/// overlay with health predictions:
///
///   ┌─ subradian  ─── arousal bands (Startle / AcuteArousal / ThreatVigilance / SustainedArousal)
///   ├─ intent     ─── behavioural intent inference (Normal → ArmedThreat)
///   ├─ field      ─── spatial threat heat map (proxemics + orientation convergence)
///   ├─ consensus  ─── multi-entity wave-function collapse
///   ├─ propagation─── health delta propagation through symbol graphs
///   ├─ health     ─── 6D universal health vector + HSV color encoding
///   └─ overlay    ─── unified API output (ThreatOverlay + HealthPrediction)

pub mod health;
pub mod subradian;
pub mod field;
pub mod intent;
pub mod consensus;
pub mod propagation;
pub mod overlay;

// ─── Convenience re-exports ───────────────────────────────────────────────────

// Health
pub use health::{
    EntityHealthState, HealthBaseline, BaselineRegistry,
    HealthDimension, HealthVector, HsvColor, Scope, ScopedHealth,
    health_to_hsv, hsv_to_rgb,
};

// Sub-ultradian
pub use subradian::{
    BehavioralSample, SubUltradianBand, SubUltradianBandState,
    SubUltradianDetector, SubUltradianRegistry, SubUltradianState,
};

// Spatial threat field
pub use field::{
    EntitySpatialRecord, ProxemicsZone, ThreatCell, ThreatField,
    ThreatFieldEngine,
};

// Intent inference
pub use intent::{
    BehaviouralSignal, BehaviouralSignalType,
    IntentClass, IntentInferenceEngine, IntentSignal, PredictedHealthImpact,
};

// Consensus
pub use consensus::{
    ConsensusEngine, ConsensusSnapshot, WaveFunctionState,
};

// Propagation
pub use propagation::{
    PropagationEdge, PropagationGraph, PropagationRegistry, PropagationType,
    human_body_graph,
};

// Overlay
pub use overlay::{
    EntityOverlaySummary, HealthPrediction, ThreatOverlay, ThreatOverlayEngine,
};

// ─── Combined scene engine ────────────────────────────────────────────────────

use crate::schema::Timestamp;
use std::collections::HashMap;

/// Convenience wrapper that owns all threat sub-system engines.
///
/// Feed a frame of sensor attributes via `ingest_frame()` and call
/// `overlay()` to get the complete `ThreatOverlay` for that frame.
#[derive(Debug, Default)]
pub struct ThreatScene {
    pub field_engine:    ThreatFieldEngine,
    pub intent_engine:   IntentInferenceEngine,
    pub sub_registry:    SubUltradianRegistry,
    pub consensus:       ConsensusEngine,
    pub propagation:     PropagationRegistry,
    pub overlay_engine:  ThreatOverlayEngine,
    /// Health state per entity — kept in sync with propagation output.
    pub entity_health:   HashMap<String, EntityHealthState>,
    /// Tracked entity IDs (propagation graphs are registered separately).
    entity_ids:          std::collections::HashSet<String>,
}

impl ThreatScene {
    pub fn new() -> Self { Self::default() }

    /// Register an entity with a propagation graph (call once per entity).
    /// Uses the standard human body topology by default.
    pub fn register_human(&mut self, entity_id: &str) {
        let graph = human_body_graph(entity_id);
        *self.propagation.get_or_create(entity_id) = graph;
        self.entity_ids.insert(entity_id.to_string());
    }

    /// Ingest one frame of per-entity sensor attributes.
    ///
    /// `frame` maps entity_id → attribute key/value pairs.
    pub fn ingest_frame(
        &mut self,
        frame: &HashMap<String, HashMap<String, serde_json::Value>>,
        timestamp: Timestamp,
    ) {
        for (eid, attrs) in frame {
            self.entity_ids.insert(eid.clone());
            // Sub-ultradian arousal.
            self.sub_registry.ingest_from_attributes(eid, timestamp, attrs);
            // Intent inference.
            self.intent_engine.ingest_from_attributes(eid, attrs);
            // Spatial threat field.
            self.field_engine.ingest_from_attributes(eid, "person", timestamp, attrs);
        }
    }

    /// Compute and return the full `ThreatOverlay` for the current frame.
    pub fn overlay(&mut self, timestamp: Timestamp) -> ThreatOverlay {
        // Build entity proximity pairs for intent engine (all-pairs for now).
        let ids: Vec<String> = self.entity_ids.iter().cloned().collect();
        let mut entity_pairs: HashMap<String, Vec<String>> = HashMap::new();
        for id in &ids {
            let others: Vec<String> = ids.iter().filter(|o| *o != id).cloned().collect();
            entity_pairs.insert(id.clone(), others);
        }

        // Collect intent signals.
        let intent_signals = self.intent_engine.compute(timestamp, &entity_pairs);

        // Consensus across all intent signals.
        let consensus_snap = self.consensus.compute(&intent_signals, timestamp);

        // Build intent map for field computation.
        let intent_map = self.intent_engine.threat_map();

        // Compute the spatial threat field.
        let field = self.field_engine.compute(timestamp, &intent_map);

        // Apply predicted impacts to the propagation graph.
        for sig in &intent_signals {
            for impact in &sig.predicted_impacts {
                if let Some(dim) = HealthDimension::ALL.get(impact.dimension_index) {
                    // Apply damage to "torso" as the generic impact site.
                    self.propagation.apply_threat_delta(
                        &impact.entity_id,
                        "torso",
                        *dim,
                        impact.delta,
                        impact.confidence,
                    );
                }
            }
        }
        self.propagation.step_all();

        // Collect updated entity health states.
        let mut health_states: Vec<EntityHealthState> = Vec::new();
        for eid in &self.entity_ids {
            let hs = self.propagation.entity_health(eid, None)
                .unwrap_or_else(|| EntityHealthState::new(eid.clone(), "person"));
            self.entity_health.insert(eid.clone(), hs.clone());
            health_states.push(hs);
        }

        let arousals = self.sub_registry.all_snapshots();

        self.overlay_engine.assemble(
            timestamp,
            field,
            consensus_snap,
            intent_signals,
            arousals,
            health_states,
        )
    }
}
