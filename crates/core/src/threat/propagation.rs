/// Health delta propagation engine.
///
/// When one symbol's health changes, that change propagates through its
/// structural and functional relationships to connected symbols.  The
/// propagation weight on each edge is learned from observation — initially
/// seeded by physics priors (proximity, structural connection type, energy
/// flow direction) and then refined through the Hebbian fabric.
///
/// Design
/// ──────
/// Each entity is a graph of symbols (body parts, organs, subsystems).
/// Edges carry a propagation weight and a propagation type:
///
///   Structural  — physical connection (bone → bone, gear → gear)
///   Vascular    — fluid/energy flow (artery → tissue, wire → component)
///   Neural      — control signal (nerve → muscle, controller → actuator)
///   Proximity   — spatial proximity without structural link
///
/// Damage flows from damaged symbols to connected symbols at each step.
/// The propagation is lazy — only triggered when a source health delta
/// exceeds a threshold, preventing constant re-computation.
///
/// Scope stacking
/// ──────────────
/// After intra-entity propagation, the entity-level health is computed as
/// the weighted harmonic mean of its symbol-level scores.  The harmonic
/// mean is used (rather than arithmetic) because it gives more weight to
/// weak links — one completely failed subsystem should pull the organism
/// score down significantly even if everything else is healthy.

use crate::threat::health::{EntityHealthState, HealthDimension, HealthVector, Scope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Edge types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropagationType {
    /// Physical structural link (bone, chassis, frame).
    Structural,
    /// Fluid / energy flow (blood, electricity, hydraulics).
    Vascular,
    /// Neural / control signal (nerves, CAN bus, firmware).
    Neural,
    /// Spatial proximity without structural link.
    Proximity,
}

impl PropagationType {
    /// Base propagation speed multiplier — faster = damage arrives sooner.
    pub fn speed(self) -> f32 {
        match self {
            PropagationType::Neural     => 1.0,  // near-instantaneous
            PropagationType::Vascular   => 0.7,  // seconds to minutes
            PropagationType::Structural => 0.5,  // slower mechanical transmission
            PropagationType::Proximity  => 0.1,  // ambient / indirect
        }
    }

    /// Which dimensions propagate strongly along this edge type.
    /// Returns per-dimension propagation scale [0,1].
    pub fn dimension_scales(self) -> [f32; 6] {
        // [SI, EF, RC, FO, AR, TC]
        match self {
            PropagationType::Structural => [1.0, 0.3, 0.1, 0.5, 0.2, 0.1],
            PropagationType::Vascular   => [0.2, 1.0, 0.3, 0.6, 0.4, 0.2],
            PropagationType::Neural     => [0.1, 0.2, 1.0, 0.8, 0.3, 0.9],
            PropagationType::Proximity  => [0.3, 0.1, 0.2, 0.2, 0.5, 0.3],
        }
    }
}

// ─── Graph edge ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationEdge {
    pub from_symbol: String,
    pub to_symbol: String,
    pub propagation_type: PropagationType,
    /// Learned propagation weight [0,1].  Starts at a physics prior and
    /// is updated by observation (Hebbian: co-occurring health changes wire together).
    pub weight: f32,
    /// Directional — does damage propagate only from→to, or bidirectionally?
    pub bidirectional: bool,
}

impl PropagationEdge {
    pub fn new(from: &str, to: &str, ptype: PropagationType, weight: f32) -> Self {
        Self {
            from_symbol: from.to_string(),
            to_symbol: to.to_string(),
            propagation_type: ptype,
            weight,
            bidirectional: true,
        }
    }

    pub fn unidirectional(from: &str, to: &str, ptype: PropagationType, weight: f32) -> Self {
        Self { bidirectional: false, ..Self::new(from, to, ptype, weight) }
    }
}

// ─── Symbol health record ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SymbolHealth {
    symbol_id: String,
    health: HealthVector,
    /// Health from previous frame — used to detect deltas.
    prev_health: HealthVector,
    /// Scope this symbol occupies.
    scope: Scope,
}

// ─── Propagation graph ────────────────────────────────────────────────────────

/// Per-entity symbol relationship graph with health propagation.
#[derive(Debug)]
pub struct PropagationGraph {
    entity_id: String,
    symbols: HashMap<String, SymbolHealth>,
    edges: Vec<PropagationEdge>,
    /// Minimum health delta to trigger propagation (noise floor).
    delta_threshold: f32,
}

impl PropagationGraph {
    pub fn new(entity_id: impl Into<String>) -> Self {
        Self {
            entity_id: entity_id.into(),
            symbols: HashMap::new(),
            edges: Vec::new(),
            delta_threshold: 0.02,
        }
    }

    /// Register a symbol with an initial health vector and scope.
    pub fn add_symbol(&mut self, symbol_id: &str, scope: Scope, health: HealthVector) {
        self.symbols.insert(symbol_id.to_string(), SymbolHealth {
            symbol_id: symbol_id.to_string(),
            health: health.clone(),
            prev_health: health,
            scope,
        });
    }

    /// Add a propagation edge between two symbols.
    pub fn add_edge(&mut self, edge: PropagationEdge) {
        self.edges.push(edge);
    }

    /// Apply a health delta to one symbol (e.g. from the threat field).
    pub fn apply_delta(&mut self, symbol_id: &str, dim: HealthDimension, delta: f32, confidence: f32) {
        if let Some(sym) = self.symbols.get_mut(symbol_id) {
            sym.prev_health = sym.health.clone();
            sym.health.apply_delta(dim, delta, confidence);
        }
    }

    /// Propagate health changes through the graph for one step.
    /// Returns the number of symbols that received a meaningful propagation.
    pub fn propagate_step(&mut self) -> usize {
        // Collect deltas from symbols that changed.
        let mut pending: Vec<(String, [f32; 6])> = Vec::new();
        for sym in self.symbols.values() {
            let prev = sym.prev_health.as_array();
            let curr = sym.health.as_array();
            let mut changed = false;
            let mut deltas = [0.0f32; 6];
            for i in 0..6 {
                let d = curr[i] - prev[i];
                if d.abs() > self.delta_threshold {
                    deltas[i] = d;
                    changed = true;
                }
            }
            if changed { pending.push((sym.symbol_id.clone(), deltas)); }
        }

        if pending.is_empty() { return 0; }

        // Build adjacency: from_symbol → [(to_symbol, edge)]
        let mut adjacency: HashMap<String, Vec<(String, &PropagationEdge)>> = HashMap::new();
        for edge in &self.edges {
            adjacency.entry(edge.from_symbol.clone())
                .or_default()
                .push((edge.to_symbol.clone(), edge));
            if edge.bidirectional {
                adjacency.entry(edge.to_symbol.clone())
                    .or_default()
                    .push((edge.from_symbol.clone(), edge));
            }
        }

        // Apply propagated deltas (collect first, apply after to avoid borrow conflict).
        let mut propagated: HashMap<String, [f32; 6]> = HashMap::new();
        for (source_id, source_deltas) in &pending {
            if let Some(neighbours) = adjacency.get(source_id) {
                for (target_id, edge) in neighbours {
                    if target_id == source_id { continue; }
                    let dim_scales = edge.propagation_type.dimension_scales();
                    let entry = propagated.entry(target_id.clone()).or_insert([0.0f32; 6]);
                    for i in 0..6 {
                        // Attenuate by edge weight + dimension scale + propagation speed.
                        let transmitted = source_deltas[i]
                            * edge.weight
                            * dim_scales[i]
                            * edge.propagation_type.speed()
                            * 0.6; // global attenuation so propagation decays
                        // Only propagate damage (negative deltas), not recovery.
                        // Recovery is earned through organic improvement, not transmitted.
                        if transmitted < 0.0 {
                            entry[i] += transmitted;
                        }
                    }
                }
            }
        }

        let propagated_count = propagated.len();
        for (target_id, deltas) in propagated {
            if let Some(sym) = self.symbols.get_mut(&target_id) {
                sym.prev_health = sym.health.clone();
                for (i, &d) in deltas.iter().enumerate() {
                    if d.abs() > self.delta_threshold {
                        sym.health.apply_delta(HealthDimension::ALL[i], d, 0.5);
                    }
                }
            }
        }
        propagated_count
    }

    /// Update edge weights from co-occurring health changes (Hebbian learning).
    pub fn hebbian_update(&mut self) {
        // If two connected symbols both degraded in the same frame,
        // increase the edge weight (they are co-varying → stronger link).
        let deltas: HashMap<String, f32> = self.symbols.iter().map(|(id, sym)| {
            let prev_score = sym.prev_health.overall_score();
            let curr_score = sym.health.overall_score();
            (id.clone(), curr_score - prev_score)
        }).collect();

        for edge in &mut self.edges {
            let delta_from = deltas.get(&edge.from_symbol).copied().unwrap_or(0.0);
            let delta_to   = deltas.get(&edge.to_symbol).copied().unwrap_or(0.0);
            // Co-occurring degradation → strengthen edge (fire together, wire together).
            if delta_from < -0.01 && delta_to < -0.01 {
                edge.weight = (edge.weight + 0.02).min(1.0);
            }
        }
    }

    /// Compute organism-level health as weighted harmonic mean of symbol scores.
    pub fn organism_health(&self) -> HealthVector {
        if self.symbols.is_empty() { return HealthVector::default(); }

        let mut dim_sums = [0.0f32; 6];
        let mut inv_sums = [0.0f32; 6];
        let mut unc_sums = [0.0f32; 6];
        let n = self.symbols.len() as f32;

        for sym in self.symbols.values() {
            let arr = sym.health.as_array();
            // Harmonic mean contribution: 1/x (handle near-zero carefully).
            for i in 0..6 {
                let v = arr[i].max(0.001);
                inv_sums[i] += 1.0 / v;
                dim_sums[i] += arr[i];
                unc_sums[i] += sym.health.uncertainty[i];
            }
        }

        let mut out = HealthVector::default();
        for (i, dim) in HealthDimension::ALL.iter().enumerate() {
            // Harmonic mean = n / Σ(1/xi)
            let harmonic = if inv_sums[i] > 1e-9 { n / inv_sums[i] } else { 0.0 };
            out.set(*dim, harmonic.clamp(0.0, 1.0));
            out.uncertainty[i] = unc_sums[i] / n;
        }
        out
    }

    /// Build an EntityHealthState from the current graph.
    pub fn to_entity_health(&self, baseline: Option<&HealthVector>) -> EntityHealthState {
        let mut state = EntityHealthState::new(self.entity_id.clone(), "entity");
        // Add organism scope.
        let org_hv = self.organism_health();
        state.update_scope(Scope::Organism, org_hv, baseline);
        // Add per-symbol scopes.
        for sym in self.symbols.values() {
            state.update_scope(sym.scope.clone(), sym.health.clone(), baseline);
        }
        state
    }
}

// ─── Propagation registry ─────────────────────────────────────────────────────

/// Manages propagation graphs for all tracked entities.
#[derive(Debug, Default)]
pub struct PropagationRegistry {
    graphs: HashMap<String, PropagationGraph>,
}

impl PropagationRegistry {
    /// Get or create a propagation graph for an entity.
    pub fn get_or_create(&mut self, entity_id: &str) -> &mut PropagationGraph {
        self.graphs
            .entry(entity_id.to_string())
            .or_insert_with(|| PropagationGraph::new(entity_id))
    }

    pub fn get(&self, entity_id: &str) -> Option<&PropagationGraph> {
        self.graphs.get(entity_id)
    }

    /// Step all graphs one propagation cycle.
    pub fn step_all(&mut self) {
        for graph in self.graphs.values_mut() {
            graph.propagate_step();
            graph.hebbian_update();
        }
    }

    /// Apply a health delta from the threat field to an entity's body symbol.
    pub fn apply_threat_delta(
        &mut self,
        entity_id: &str,
        symbol_id: &str,
        dim: HealthDimension,
        delta: f32,
        confidence: f32,
    ) {
        if let Some(graph) = self.graphs.get_mut(entity_id) {
            graph.apply_delta(symbol_id, dim, delta, confidence);
        }
    }

    /// Compute organism-level health for one entity.
    pub fn organism_health(&self, entity_id: &str) -> Option<HealthVector> {
        self.graphs.get(entity_id).map(|g| g.organism_health())
    }

    /// Build EntityHealthState for one entity.
    pub fn entity_health(&self, entity_id: &str, baseline: Option<&HealthVector>) -> Option<EntityHealthState> {
        self.graphs.get(entity_id).map(|g| g.to_entity_health(baseline))
    }
}

// ─── Human body default graph ─────────────────────────────────────────────────

/// Create a default human body propagation graph with standard anatomy edges.
pub fn human_body_graph(entity_id: &str) -> PropagationGraph {
    let mut graph = PropagationGraph::new(entity_id);
    let hv = HealthVector::default();

    // Core body parts.
    let parts = [
        ("head",          Scope::BodyPart("head".to_string())),
        ("torso",         Scope::BodyPart("torso".to_string())),
        ("left_arm",      Scope::BodyPart("left_arm".to_string())),
        ("right_arm",     Scope::BodyPart("right_arm".to_string())),
        ("left_leg",      Scope::BodyPart("left_leg".to_string())),
        ("right_leg",     Scope::BodyPart("right_leg".to_string())),
        ("cardiovascular",Scope::Organ("cardiovascular".to_string())),
        ("nervous_system",Scope::Organ("nervous_system".to_string())),
    ];
    for (id, scope) in parts {
        graph.add_symbol(id, scope, hv.clone());
    }

    // Structural edges (bones/structure).
    let structural = [
        ("head",   "torso",     0.6),
        ("torso",  "left_arm",  0.5),
        ("torso",  "right_arm", 0.5),
        ("torso",  "left_leg",  0.5),
        ("torso",  "right_leg", 0.5),
    ];
    for (a, b, w) in structural {
        graph.add_edge(PropagationEdge::new(a, b, PropagationType::Structural, w));
    }

    // Vascular edges (blood/energy flow — unidirectional from heart outward).
    let vascular = [
        ("cardiovascular", "head",      0.8),
        ("cardiovascular", "torso",     0.8),
        ("cardiovascular", "left_arm",  0.7),
        ("cardiovascular", "right_arm", 0.7),
        ("cardiovascular", "left_leg",  0.7),
        ("cardiovascular", "right_leg", 0.7),
    ];
    for (a, b, w) in vascular {
        graph.add_edge(PropagationEdge::unidirectional(a, b, PropagationType::Vascular, w));
    }

    // Neural edges (control signals from head/CNS).
    let neural = [
        ("nervous_system", "left_arm",  0.9),
        ("nervous_system", "right_arm", 0.9),
        ("nervous_system", "left_leg",  0.9),
        ("nervous_system", "right_leg", 0.9),
        ("nervous_system", "cardiovascular", 0.7),
        ("head",           "nervous_system", 0.9),
    ];
    for (a, b, w) in neural {
        graph.add_edge(PropagationEdge::unidirectional(a, b, PropagationType::Neural, w));
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_damage_to_cardiovascular_propagates_to_limbs() {
        let mut graph = human_body_graph("patient");
        // Simulate bullet wound to cardiovascular system.
        graph.apply_delta("cardiovascular", HealthDimension::StructuralIntegrity, -0.8, 1.0);
        graph.apply_delta("cardiovascular", HealthDimension::EnergeticFlux, -0.9, 1.0);
        // Run several propagation steps.
        for _ in 0..5 { graph.propagate_step(); }
        let org = graph.organism_health();
        assert!(org.ef < 0.5, "energetic flux should propagate damage: {}", org.ef);
        // Limbs should also show degradation from vascular damage.
        let left_arm = &graph.symbols["left_arm"];
        assert!(left_arm.health.ef < 0.48, "left arm EF should degrade: {}", left_arm.health.ef);
    }

    #[test]
    fn isolated_arm_damage_does_not_kill_organism() {
        let mut graph = human_body_graph("patient2");
        graph.apply_delta("left_arm", HealthDimension::StructuralIntegrity, -1.0, 1.0);
        for _ in 0..5 { graph.propagate_step(); }
        let org = graph.organism_health();
        // Organism should be stressed but not dead.
        assert!(org.overall_score() > 0.3, "organism should survive arm damage: {}", org.overall_score());
    }
}
