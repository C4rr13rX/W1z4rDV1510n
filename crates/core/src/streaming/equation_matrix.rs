//! Environmental Equation Matrix (EEM)
//!
//! A decentralized, self-growing graph of physics equations that bridges
//! the gap between raw sensor streams and physical interpretation.
//!
//! ## Design
//!
//! Each node maintains its own EEM — a weighted directed graph where:
//!   * **Nodes** are physics equations (text + LaTeX + variables + metadata)
//!   * **Edges** are semantic relationships (derives_from, unifies, special_case…)
//!
//! The matrix is:
//!   * **Dimensional-aware**: equations are tagged with the spatial dimensions
//!     they apply to.  Anyons are explicitly 2-D topological phenomena;
//!     Maxwell's equations in full form require 3-D; thermodynamic identities
//!     are dimension-agnostic.
//!   * **Sensor-driven**: `apply_to_context(labels, dims)` takes the active
//!     neuro-fabric labels plus the sensor dimensionality and returns candidate
//!     equations that might explain the current observation.
//!   * **Confidence-evolving**: `reinforce(id)` increments evidence counts and
//!     updates confidence as more sensor data corroborates an equation.
//!   * **P2P-shareable**: `to_peer_payload()` / `merge_peer_payload()` let nodes
//!     broadcast discoveries and absorb each other's findings over the gossip
//!     network without any central authority.
//!   * **Self-researching**: when the neuro fabric discovers a novel pattern, the
//!     EEM records it as a `HypothesisSlot` — an open gap awaiting an equation
//!     that explains it.
//!
//! ## Physics coverage (seed)
//!
//! Classical → Lagrangian/Hamiltonian → Thermodynamics → Statistical Mechanics →
//! Electromagnetism (Maxwell) → Quantum Mechanics (Schrödinger / Heisenberg /
//! Dirac) → Quantum Field Theory → Special + General Relativity →
//! Fluid Dynamics (Navier-Stokes) → Chaos / Nonlinear Dynamics →
//! Topological Physics (anyons, Chern-Simons) → Information Theory

use crate::network::compute_payload_hash;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── Taxonomy ─────────────────────────────────────────────────────────────────

/// Physics discipline — from Newtonian mechanics to topological quantum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Discipline {
    ClassicalMechanics,
    LagrangianMechanics,
    HamiltonianMechanics,
    Thermodynamics,
    StatisticalMechanics,
    Electromagnetism,
    QuantumMechanics,
    QuantumFieldTheory,
    SpecialRelativity,
    GeneralRelativity,
    FluidDynamics,
    ChaosDynamics,
    TopologicalPhysics,
    CondensedMatter,
    Cosmology,
    InformationTheory,
    Custom(String),
}

impl Discipline {
    /// Canonical keyword tags used to match against neuro-fabric labels.
    pub fn keywords(&self) -> &[&str] {
        match self {
            Self::ClassicalMechanics => &["force","mass","acceleration","momentum","work","energy","gravity","collision","torque","inertia"],
            Self::LagrangianMechanics => &["lagrangian","action","euler","lagrange","generalized","coordinate","variational"],
            Self::HamiltonianMechanics => &["hamiltonian","phase","space","canonical","poisson","bracket","symplectic"],
            Self::Thermodynamics => &["temperature","entropy","heat","pressure","volume","work","gibbs","enthalpy","carnot","cycle","thermodynamic"],
            Self::StatisticalMechanics => &["partition","boltzmann","microstate","macrostate","ensemble","fermi","bose","statistical"],
            Self::Electromagnetism => &["electric","magnetic","field","flux","charge","current","maxwell","photon","electromagnetic","wave","capacitor","inductor"],
            Self::QuantumMechanics => &["wave","function","schrodinger","eigenvalue","operator","spin","superposition","uncertainty","quantum","particle","probability","planck","hbar"],
            Self::QuantumFieldTheory => &["feynman","diagram","propagator","lagrangian","field","interaction","renormalization","gauge","boson","fermion","vacuum"],
            Self::SpecialRelativity => &["lorentz","spacetime","invariant","four","vector","light","speed","relativistic","doppler","simultaneity"],
            Self::GeneralRelativity => &["curvature","metric","tensor","einstein","geodesic","gravity","spacetime","black","hole","cosmological"],
            Self::FluidDynamics => &["fluid","flow","velocity","pressure","viscosity","turbulence","navier","stokes","reynolds","bernoulli","vortex","compressible"],
            Self::ChaosDynamics => &["chaos","lyapunov","attractor","bifurcation","strange","sensitivity","nonlinear","lorenz","fractal","logistic"],
            Self::TopologicalPhysics => &["anyon","topological","chern","simons","berry","phase","braid","winding","knot","hall","edge","state","majorana"],
            Self::CondensedMatter => &["phonon","magnon","crystal","lattice","band","fermi","surface","superconductor","cooper","pair","bcs"],
            Self::Cosmology => &["hubble","redshift","inflation","dark","matter","dark","energy","cmb","cosmic","expansion","friedmann"],
            Self::InformationTheory => &["entropy","shannon","information","mutual","channel","capacity","kolmogorov","bit","nats","compression"],
            Self::Custom(_) => &[],
        }
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Custom(s) => s.clone(),
            other => format!("{:?}", other),
        }
    }
}

/// Semantic relationship type between two equations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
    /// B can be mathematically derived from A.
    DerivesFrom,
    /// A is a special case of B (e.g. Newtonian limit of relativistic eq).
    SpecialCase,
    /// A and B describe the same phenomenon from different perspectives.
    Bridges,
    /// A is a unification of B and C.
    Unifies,
    /// A contradicts B in some overlapping regime — tension to resolve.
    Contradicts,
    /// A is an approximation of B valid in some limit.
    Approximates,
    /// A generalises B to more dimensions or domains.
    Generalizes,
}

// ─── Core data structures ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationVariable {
    /// Symbol as it appears in the equation (e.g. "F", "ħ", "∇")
    pub symbol: String,
    /// Descriptive name
    pub name: String,
    /// SI units string (e.g. "kg·m/s²")
    pub units: String,
    /// Dimensional analysis string (e.g. "[M L T^-2]")
    pub dimension: String,
}

/// A physics equation node in the Environmental Equation Matrix graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsEquation {
    /// Stable identifier (hash of text + discipline).
    pub id: String,
    /// Human-readable form.
    pub text: String,
    /// LaTeX representation.
    pub latex: String,
    /// Which physics domain this belongs to.
    pub discipline: Discipline,
    /// Variables that appear in this equation.
    pub variables: Vec<EquationVariable>,
    /// Confidence in this equation [0, 1].  Starts at 0.9 for seeded equations,
    /// grows as sensor evidence accumulates, decays if no corroboration arrives.
    pub confidence: f32,
    /// Physical assumptions under which this equation holds.
    pub assumptions: Vec<String>,
    /// Mathematical constraints on variables.
    pub constraints: Vec<String>,
    /// Which spatial dimensions this equation applies to.
    ///   `[2]` → 2-D only (anyons),  `[3]` → 3-D only,  `[2, 3]` → both.
    ///   `[]` → dimension-agnostic.
    pub applicable_dims: Vec<u8>,
    /// Citation strings (paper, textbook, DOI).
    pub citations: Vec<String>,
    /// Node ID that first introduced this equation (None = seeded locally).
    pub source_node_id: Option<String>,
    /// Number of sensor observations that corroborate this equation.
    pub evidence_count: u32,
    /// Unix timestamp of ingestion.
    pub ingested_at: u64,
}

impl PhysicsEquation {
    fn new(
        text: impl Into<String>,
        latex: impl Into<String>,
        discipline: Discipline,
        variables: Vec<EquationVariable>,
        applicable_dims: Vec<u8>,
        assumptions: &[&str],
        confidence: f32,
    ) -> Self {
        let text = text.into();
        let latex = latex.into();
        let id_src = format!("{}::{}", discipline.as_str(), text);
        let id = compute_payload_hash(id_src.as_bytes());
        Self {
            id,
            text,
            latex,
            discipline,
            variables,
            applicable_dims,
            assumptions: assumptions.iter().map(|s| s.to_string()).collect(),
            constraints: Vec::new(),
            confidence,
            citations: Vec::new(),
            source_node_id: None,
            evidence_count: 0,
            ingested_at: now_unix(),
        }
    }

    /// Returns true if this equation is applicable given a sensor's dimensionality.
    pub fn matches_dims(&self, sensor_dims: u8) -> bool {
        self.applicable_dims.is_empty() || self.applicable_dims.contains(&sensor_dims)
    }
}

/// Directed edge between two equations in the EEM graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationLink {
    pub from_id: String,
    pub to_id: String,
    pub relation_type: LinkType,
    /// Semantic weight [0, 1].
    pub weight: f32,
    pub notes: String,
    pub created_at: u64,
}

/// An open gap: a sensor pattern that has no equation explanation yet.
/// The system records these so future data / peer nodes can fill them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisSlot {
    pub id: String,
    pub description: String,
    /// Neuro labels that fired but had no matching equation.
    pub trigger_labels: Vec<String>,
    /// Sensor dimensionality at time of observation.
    pub sensor_dims: u8,
    pub created_at: u64,
    /// How many times this gap has been observed.
    pub observation_count: u32,
}

// ─── Search ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationSearchResult {
    pub equation: PhysicsEquation,
    /// How strongly this result matches the query (0..1).
    pub relevance: f32,
    /// Equations reachable from this one via graph traversal.
    pub related_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextApplicationResult {
    /// Equations whose discipline keywords overlap with the active labels.
    pub candidates: Vec<EquationSearchResult>,
    /// Open hypothesis slots matching the given context.
    pub open_gaps: Vec<HypothesisSlot>,
    /// Labels that matched nothing — candidates for new hypothesis slots.
    pub unexplained_labels: Vec<String>,
}

// ─── Peer exchange ────────────────────────────────────────────────────────────

/// Lightweight payload shared between nodes over the gossip network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EemPeerPayload {
    pub source_node_id: String,
    /// New/updated equations this node has discovered.
    pub equations: Vec<PhysicsEquation>,
    /// New links this node has established.
    pub links: Vec<EquationLink>,
    /// Hypothesis slots this node has opened (unsolved gaps).
    pub open_gaps: Vec<HypothesisSlot>,
    pub created_at: u64,
}

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EquationMatrixConfig {
    /// Persist the EEM to this JSON file on every write.
    pub persist_path: Option<String>,
    /// Minimum confidence for an equation to be included in search results.
    pub min_confidence: f32,
    /// Confidence increment per corroborating sensor observation.
    pub evidence_boost: f32,
    /// Confidence decay per time step with no observation.
    pub decay_rate: f32,
    /// Maximum graph traversal hops for related-equation discovery.
    pub max_traversal_hops: usize,
    /// Maximum results returned from search.
    pub max_search_results: usize,
    /// Maximum hypothesis slots kept open simultaneously.
    pub max_open_gaps: usize,
    /// Whether to seed foundational physics equations on first init.
    pub seed_on_init: bool,
}

impl Default for EquationMatrixConfig {
    fn default() -> Self {
        Self {
            persist_path: None,
            min_confidence: 0.3,
            evidence_boost: 0.02,
            decay_rate: 0.0005,
            max_traversal_hops: 4,
            max_search_results: 12,
            max_open_gaps: 256,
            seed_on_init: true,
        }
    }
}

// ─── EEM state (persisted) ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct EemState {
    equations: HashMap<String, PhysicsEquation>,
    links: Vec<EquationLink>,
    open_gaps: Vec<HypothesisSlot>,
    seeded: bool,
}

// ─── Runtime ─────────────────────────────────────────────────────────────────

/// The Environmental Equation Matrix runtime.
///
/// Thread-safe via internal `parking_lot::Mutex`.  Designed to be wrapped in
/// `Arc<EquationMatrixRuntime>` and shared across the node API threads.
pub struct EquationMatrixRuntime {
    config: EquationMatrixConfig,
    state: parking_lot::Mutex<EemState>,
}

impl EquationMatrixRuntime {
    pub fn new(config: EquationMatrixConfig) -> Self {
        let mut state = EemState::default();
        // Try to load persisted state
        if let Some(ref path) = config.persist_path {
            if let Ok(raw) = std::fs::read_to_string(path) {
                if let Ok(loaded) = serde_json::from_str::<EemState>(&raw) {
                    state = loaded;
                }
            }
        }
        let runtime = Self {
            config,
            state: parking_lot::Mutex::new(state),
        };
        if runtime.config.seed_on_init {
            runtime.seed_base_equations();
        }
        runtime
    }

    // ── Ingestion ────────────────────────────────────────────────────────────

    /// Add a fully-formed equation to the matrix.
    /// Idempotent: duplicate IDs are silently ignored.
    pub fn ingest_equation(&self, eq: PhysicsEquation) {
        let mut st = self.state.lock();
        st.equations.entry(eq.id.clone()).or_insert(eq);
        drop(st);
        self.persist();
    }

    /// Parse equations from free text and ingest them.
    /// Uses simple `<lhs> = <rhs>` pattern matching; confidence is lower for
    /// auto-extracted equations than for hand-curated ones.
    pub fn ingest_text(
        &self,
        text: &str,
        discipline: Discipline,
        source_node_id: Option<String>,
        confidence: f32,
    ) -> Vec<String> {
        let extracted = extract_equations_from_text(text);
        let mut ids = Vec::new();
        for (eq_text, vars) in extracted {
            let id_src = format!("{}::{}", discipline.as_str(), eq_text);
            let id = compute_payload_hash(id_src.as_bytes());
            let eq = PhysicsEquation {
                id: id.clone(),
                text: eq_text.clone(),
                latex: eq_text,
                discipline: discipline.clone(),
                variables: vars.into_iter().map(|sym| EquationVariable {
                    symbol: sym.clone(),
                    name: sym,
                    units: String::new(),
                    dimension: String::new(),
                }).collect(),
                confidence,
                assumptions: Vec::new(),
                constraints: Vec::new(),
                applicable_dims: Vec::new(),
                citations: Vec::new(),
                source_node_id: source_node_id.clone(),
                evidence_count: 0,
                ingested_at: now_unix(),
            };
            {
                let mut st = self.state.lock();
                st.equations.entry(id.clone()).or_insert(eq);
            }
            ids.push(id);
        }
        if !ids.is_empty() {
            self.persist();
        }
        ids
    }

    /// Add a semantic link between two equations.
    pub fn add_link(&self, from_id: &str, to_id: &str, relation: LinkType, weight: f32, notes: &str) {
        let mut st = self.state.lock();
        let already = st.links.iter().any(|l| l.from_id == from_id && l.to_id == to_id && l.relation_type == relation);
        if !already {
            st.links.push(EquationLink {
                from_id: from_id.to_string(),
                to_id: to_id.to_string(),
                relation_type: relation,
                weight,
                notes: notes.to_string(),
                created_at: now_unix(),
            });
        }
        drop(st);
        self.persist();
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Text-based search across equation text and LaTeX.
    pub fn search(&self, query: &str, limit: Option<usize>) -> Vec<EquationSearchResult> {
        let limit = limit.unwrap_or(self.config.max_search_results);
        let tokens: Vec<String> = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }
        let st = self.state.lock();
        let mut scored: Vec<(f32, &PhysicsEquation)> = st.equations.values()
            .filter(|e| e.confidence >= self.config.min_confidence)
            .filter_map(|e| {
                let relevance = score_equation(e, &tokens);
                if relevance > 0.0 { Some((relevance, e)) } else { None }
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let link_index = build_link_index(&st.links);
        scored.into_iter().map(|(relevance, eq)| {
            let related_ids = traverse_graph(&link_index, &eq.id, 2);
            EquationSearchResult {
                equation: eq.clone(),
                relevance,
                related_ids,
            }
        }).collect()
    }

    /// Given a set of currently active neuro labels and a sensor dimensionality,
    /// return equations that might explain the current observation.
    pub fn apply_to_context(
        &self,
        active_labels: &[String],
        sensor_dims: u8,
    ) -> ContextApplicationResult {
        let label_tokens: Vec<String> = active_labels.iter()
            .flat_map(|l| tokenize(l))
            .collect();

        let st = self.state.lock();
        let link_index = build_link_index(&st.links);

        let mut candidates: Vec<(f32, &PhysicsEquation)> = st.equations.values()
            .filter(|e| e.confidence >= self.config.min_confidence)
            .filter(|e| e.matches_dims(sensor_dims))
            .filter_map(|e| {
                // Score by discipline keyword overlap + text token overlap
                let kw_score = keyword_overlap(e, &label_tokens);
                let text_score = score_equation(e, &label_tokens);
                let combined = kw_score * 0.6 + text_score * 0.4;
                if combined > 0.0 { Some((combined, e)) } else { None }
            })
            .collect();
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_search_results);

        let candidate_results: Vec<EquationSearchResult> = candidates.into_iter().map(|(rel, eq)| {
            let related_ids = traverse_graph(&link_index, &eq.id, self.config.max_traversal_hops);
            EquationSearchResult { equation: eq.clone(), relevance: rel, related_ids }
        }).collect();

        let explained_labels: HashSet<&str> = candidate_results.iter()
            .flat_map(|r| r.equation.discipline.keywords().iter().copied())
            .collect();
        let unexplained: Vec<String> = label_tokens.iter()
            .filter(|t| t.len() >= 4 && !explained_labels.contains(t.as_str()))
            .cloned()
            .collect();

        let open_gaps: Vec<HypothesisSlot> = st.open_gaps.iter()
            .filter(|g| g.sensor_dims == sensor_dims || g.sensor_dims == 0)
            .filter(|g| g.trigger_labels.iter().any(|l| label_tokens.contains(l)))
            .take(6)
            .cloned()
            .collect();

        ContextApplicationResult {
            candidates: candidate_results,
            open_gaps,
            unexplained_labels: unexplained,
        }
    }

    /// Called when sensor data corroborates an equation — increases confidence.
    pub fn reinforce(&self, eq_id: &str) {
        let mut st = self.state.lock();
        if let Some(eq) = st.equations.get_mut(eq_id) {
            eq.evidence_count += 1;
            eq.confidence = (eq.confidence + self.config.evidence_boost).min(1.0);
        }
    }

    /// Decay all confidences slightly — equations not corroborated by sensor data
    /// gradually lose priority.
    pub fn decay_step(&self) {
        let mut st = self.state.lock();
        for eq in st.equations.values_mut() {
            if eq.source_node_id.is_some() {
                // Peer-discovered equations decay unless locally reinforced
                eq.confidence = (eq.confidence - self.config.decay_rate).max(0.01);
            }
        }
    }

    /// Record a novel pattern that has no current equation explanation.
    pub fn open_gap(&self, trigger_labels: Vec<String>, sensor_dims: u8, description: &str) {
        let id_src = format!("gap::{}::{}", sensor_dims, trigger_labels.join(","));
        let id = compute_payload_hash(id_src.as_bytes());
        let mut st = self.state.lock();
        if let Some(existing) = st.open_gaps.iter_mut().find(|g| g.id == id) {
            existing.observation_count += 1;
            return;
        }
        if st.open_gaps.len() < self.config.max_open_gaps {
            st.open_gaps.push(HypothesisSlot {
                id,
                description: description.to_string(),
                trigger_labels,
                sensor_dims,
                created_at: now_unix(),
                observation_count: 1,
            });
        }
    }

    // ── P2P exchange ─────────────────────────────────────────────────────────

    /// Serialize discoveries for sharing with peers.
    pub fn to_peer_payload(&self, node_id: &str) -> EemPeerPayload {
        let st = self.state.lock();
        // Only share equations with meaningful evidence
        let equations: Vec<PhysicsEquation> = st.equations.values()
            .filter(|e| e.evidence_count > 0 || e.source_node_id.is_none())
            .cloned()
            .collect();
        EemPeerPayload {
            source_node_id: node_id.to_string(),
            equations,
            links: st.links.clone(),
            open_gaps: st.open_gaps.iter().filter(|g| g.observation_count >= 2).cloned().collect(),
            created_at: now_unix(),
        }
    }

    /// Integrate equations and links received from a peer node.
    pub fn merge_peer_payload(&self, payload: EemPeerPayload) {
        let mut st = self.state.lock();
        for mut eq in payload.equations {
            eq.source_node_id = Some(payload.source_node_id.clone());
            // Peers get lower initial confidence; local evidence will grow it
            eq.confidence = eq.confidence.min(0.7);
            st.equations.entry(eq.id.clone()).or_insert(eq);
        }
        for link in payload.links {
            let already = st.links.iter().any(|l| l.from_id == link.from_id && l.to_id == link.to_id && l.relation_type == link.relation_type);
            if !already {
                st.links.push(link);
            }
        }
        for gap in payload.open_gaps {
            let already = st.open_gaps.iter().any(|g| g.id == gap.id);
            if !already && st.open_gaps.len() < self.config.max_open_gaps {
                st.open_gaps.push(gap);
            }
        }
        drop(st);
        self.persist();
    }

    // ── Reporting ─────────────────────────────────────────────────────────────

    pub fn report(&self) -> EquationMatrixReport {
        let st = self.state.lock();
        let by_discipline: HashMap<String, usize> = {
            let mut map: HashMap<String, usize> = HashMap::new();
            for eq in st.equations.values() {
                *map.entry(eq.discipline.as_str()).or_insert(0) += 1;
            }
            map
        };
        let top_equations: Vec<EquationSummary> = {
            let mut sorted: Vec<&PhysicsEquation> = st.equations.values().collect();
            sorted.sort_by(|a, b| {
                let score_a = a.confidence * (1.0 + a.evidence_count as f32 * 0.1);
                let score_b = b.confidence * (1.0 + b.evidence_count as f32 * 0.1);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted.truncate(20);
            sorted.into_iter().map(|e| EquationSummary {
                id: e.id.clone(),
                text: e.text.clone(),
                discipline: e.discipline.as_str(),
                confidence: e.confidence,
                evidence_count: e.evidence_count,
                applicable_dims: e.applicable_dims.clone(),
            }).collect()
        };
        EquationMatrixReport {
            total_equations: st.equations.len(),
            total_links: st.links.len(),
            open_gaps: st.open_gaps.len(),
            by_discipline,
            top_equations,
        }
    }

    pub fn all_equations(&self) -> Vec<PhysicsEquation> {
        let st = self.state.lock();
        st.equations.values().cloned().collect()
    }

    pub fn get_equation(&self, id: &str) -> Option<PhysicsEquation> {
        let st = self.state.lock();
        st.equations.get(id).cloned()
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    fn persist(&self) {
        if let Some(ref path) = self.config.persist_path {
            let st = self.state.lock();
            if let Ok(json) = serde_json::to_string_pretty(&*st) {
                let _ = std::fs::write(path, json);
            }
        }
    }

    // ── Seeding ───────────────────────────────────────────────────────────────

    /// Populate the matrix with foundational physics equations spanning all
    /// domains from Newtonian mechanics to topological quantum phenomena.
    pub fn seed_base_equations(&self) {
        {
            let st = self.state.lock();
            if st.seeded {
                return;
            }
        }

        let seeds = build_seed_equations();
        let links = build_seed_links(&seeds);

        let mut st = self.state.lock();
        for eq in seeds {
            st.equations.entry(eq.id.clone()).or_insert(eq);
        }
        for link in links {
            let already = st.links.iter().any(|l| l.from_id == link.from_id && l.to_id == link.to_id);
            if !already {
                st.links.push(link);
            }
        }
        st.seeded = true;
        drop(st);
        self.persist();
    }
}

// ─── Report types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationSummary {
    pub id: String,
    pub text: String,
    pub discipline: String,
    pub confidence: f32,
    pub evidence_count: u32,
    pub applicable_dims: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationMatrixReport {
    pub total_equations: usize,
    pub total_links: usize,
    pub open_gaps: usize,
    pub by_discipline: HashMap<String, usize>,
    pub top_equations: Vec<EquationSummary>,
}

// ─── Graph helpers ────────────────────────────────────────────────────────────

type LinkIndex = HashMap<String, Vec<(String, f32)>>;

fn build_link_index(links: &[EquationLink]) -> LinkIndex {
    let mut idx: LinkIndex = HashMap::new();
    for link in links {
        idx.entry(link.from_id.clone()).or_default().push((link.to_id.clone(), link.weight));
        idx.entry(link.to_id.clone()).or_default().push((link.from_id.clone(), link.weight));
    }
    idx
}

fn traverse_graph(index: &LinkIndex, start: &str, hops: usize) -> Vec<String> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start.to_string(), 0usize));
    visited.insert(start.to_string());
    let mut result = Vec::new();
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= hops { continue; }
        if let Some(neighbors) = index.get(&node) {
            for (neighbor, _weight) in neighbors {
                if visited.insert(neighbor.clone()) {
                    result.push(neighbor.clone());
                    queue.push_back((neighbor.clone(), depth + 1));
                }
            }
        }
    }
    result
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_lowercase())
        .collect()
}

fn score_equation(eq: &PhysicsEquation, tokens: &[String]) -> f32 {
    if tokens.is_empty() { return 0.0; }
    let text_lower = eq.text.to_lowercase();
    let latex_lower = eq.latex.to_lowercase();
    let hits = tokens.iter().filter(|t| text_lower.contains(t.as_str()) || latex_lower.contains(t.as_str())).count();
    (hits as f32 / tokens.len() as f32) * eq.confidence
}

fn keyword_overlap(eq: &PhysicsEquation, tokens: &[String]) -> f32 {
    let kws = eq.discipline.keywords();
    if kws.is_empty() { return 0.0; }
    let hits = tokens.iter().filter(|t| kws.contains(&t.as_str())).count();
    (hits as f32 / kws.len() as f32).min(1.0) * eq.confidence
}

// ─── Text equation extractor ─────────────────────────────────────────────────

fn extract_equations_from_text(text: &str) -> Vec<(String, Vec<String>)> {
    let mut results = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains('=') && trimmed.len() > 3 && trimmed.len() < 256 {
            let vars: Vec<String> = trimmed
                .split(|c: char| !c.is_alphabetic() && c != '_' && c != 'ħ' && c != 'Ω' && c != 'λ' && c != 'μ')
                .filter(|t| !t.is_empty() && t.len() <= 6)
                .map(String::from)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            results.push((trimmed.to_string(), vars));
        }
    }
    results
}

// ─── Seed equations ───────────────────────────────────────────────────────────

fn v(sym: &str, name: &str, units: &str, dim: &str) -> EquationVariable {
    EquationVariable { symbol: sym.into(), name: name.into(), units: units.into(), dimension: dim.into() }
}

fn build_seed_equations() -> Vec<PhysicsEquation> {
    vec![
        // ── Classical Mechanics ────────────────────────────────────────────
        PhysicsEquation::new("F = m * a", r"F = m \cdot a",
            Discipline::ClassicalMechanics,
            vec![v("F","Force","N","[M L T^-2]"), v("m","mass","kg","[M]"), v("a","acceleration","m/s²","[L T^-2]")],
            vec![2,3], &["inertial reference frame","non-relativistic"], 0.95),

        PhysicsEquation::new("p = m * v", r"p = m \cdot v",
            Discipline::ClassicalMechanics,
            vec![v("p","momentum","kg·m/s","[M L T^-1]"), v("m","mass","kg","[M]"), v("v","velocity","m/s","[L T^-1]")],
            vec![2,3], &["non-relativistic"], 0.95),

        PhysicsEquation::new("W = F * d * cos(θ)", r"W = F d \cos\theta",
            Discipline::ClassicalMechanics,
            vec![v("W","work","J","[M L^2 T^-2]"), v("F","force","N","[M L T^-2]"), v("d","displacement","m","[L]")],
            vec![2,3], &["constant force"], 0.93),

        PhysicsEquation::new("E_k = (1/2) * m * v^2", r"E_k = \frac{1}{2}mv^2",
            Discipline::ClassicalMechanics,
            vec![v("E_k","kinetic energy","J","[M L^2 T^-2]"), v("m","mass","kg","[M]"), v("v","speed","m/s","[L T^-1]")],
            vec![2,3], &["non-relativistic"], 0.95),

        PhysicsEquation::new("F_g = G * m1 * m2 / r^2", r"F_g = G\frac{m_1 m_2}{r^2}",
            Discipline::ClassicalMechanics,
            vec![v("G","gravitational constant","m³/(kg·s²)","[L^3 M^-1 T^-2]"), v("r","separation","m","[L]")],
            vec![3], &["point masses","weak field"], 0.95),

        // ── Lagrangian / Hamiltonian ───────────────────────────────────────
        PhysicsEquation::new("L = T - V", r"\mathcal{L} = T - V",
            Discipline::LagrangianMechanics,
            vec![v("L","Lagrangian","J","[M L^2 T^-2]"), v("T","kinetic energy","J","[M L^2 T^-2]"), v("V","potential energy","J","[M L^2 T^-2]")],
            vec![2,3], &[], 0.95),

        PhysicsEquation::new("d/dt(∂L/∂q̇) - ∂L/∂q = 0", r"\frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot{q}}-\frac{\partial\mathcal{L}}{\partial q}=0",
            Discipline::LagrangianMechanics,
            vec![v("q","generalized coordinate","","")],
            vec![2,3], &["holonomic constraints"], 0.95),

        PhysicsEquation::new("H = T + V", r"\mathcal{H} = T + V",
            Discipline::HamiltonianMechanics,
            vec![v("H","Hamiltonian","J","[M L^2 T^-2]")],
            vec![2,3], &[], 0.95),

        // ── Thermodynamics ────────────────────────────────────────────────
        PhysicsEquation::new("dU = T * dS - P * dV", r"dU = TdS - PdV",
            Discipline::Thermodynamics,
            vec![v("U","internal energy","J","[M L^2 T^-2]"), v("T","temperature","K","[Θ]"), v("S","entropy","J/K","[M L^2 T^-2 Θ^-1]"), v("P","pressure","Pa","[M L^-1 T^-2]"), v("V","volume","m³","[L^3]")],
            vec![], &["reversible process"], 0.95),

        PhysicsEquation::new("P * V = n * R * T", r"PV = nRT",
            Discipline::Thermodynamics,
            vec![v("P","pressure","Pa","[M L^-1 T^-2]"), v("V","volume","m³","[L^3]"), v("n","moles","mol","[N]"), v("R","gas constant","J/(mol·K)",""), v("T","temperature","K","[Θ]")],
            vec![], &["ideal gas","no intermolecular forces"], 0.93),

        PhysicsEquation::new("ΔS ≥ 0 (second law)", r"\Delta S \geq 0",
            Discipline::Thermodynamics,
            vec![v("S","entropy","J/K","")],
            vec![], &["isolated system"], 0.95),

        // ── Statistical Mechanics ─────────────────────────────────────────
        PhysicsEquation::new("S = k_B * ln(Ω)", r"S = k_B \ln\Omega",
            Discipline::StatisticalMechanics,
            vec![v("S","entropy","J/K",""), v("k_B","Boltzmann constant","J/K",""), v("Ω","number of microstates","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("Z = Σ exp(-E_i / k_B * T)", r"Z = \sum_i e^{-E_i/k_BT}",
            Discipline::StatisticalMechanics,
            vec![v("Z","partition function","",""), v("E_i","energy of microstate i","J","")],
            vec![], &["canonical ensemble"], 0.93),

        // ── Electromagnetism (Maxwell's 4) ────────────────────────────────
        PhysicsEquation::new("∇ · E = ρ / ε_0", r"\nabla \cdot \mathbf{E} = \rho/\varepsilon_0",
            Discipline::Electromagnetism,
            vec![v("E","electric field","V/m",""), v("ρ","charge density","C/m³",""), v("ε_0","permittivity","F/m","")],
            vec![3], &["SI units"], 0.95),

        PhysicsEquation::new("∇ · B = 0", r"\nabla \cdot \mathbf{B} = 0",
            Discipline::Electromagnetism,
            vec![v("B","magnetic field","T","")],
            vec![3], &["no magnetic monopoles"], 0.95),

        PhysicsEquation::new("∇ × E = -∂B/∂t", r"\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t",
            Discipline::Electromagnetism,
            vec![v("E","electric field","V/m",""), v("B","magnetic field","T","")],
            vec![3], &[], 0.95),

        PhysicsEquation::new("∇ × B = μ_0 * J + μ_0 * ε_0 * ∂E/∂t", r"\nabla\times\mathbf{B}=\mu_0\mathbf{J}+\mu_0\varepsilon_0\frac{\partial\mathbf{E}}{\partial t}",
            Discipline::Electromagnetism,
            vec![v("B","magnetic field","T",""), v("J","current density","A/m²",""), v("μ_0","permeability","H/m","")],
            vec![3], &[], 0.95),

        // ── Quantum Mechanics ─────────────────────────────────────────────
        PhysicsEquation::new("iħ * ∂ψ/∂t = Ĥ * ψ", r"i\hbar\frac{\partial\psi}{\partial t}=\hat{H}\psi",
            Discipline::QuantumMechanics,
            vec![v("ψ","wave function","",""), v("ħ","reduced Planck","J·s",""), v("Ĥ","Hamiltonian operator","J","")],
            vec![2,3], &["non-relativistic","single particle"], 0.95),

        PhysicsEquation::new("E = ħ * ω", r"E = \hbar\omega",
            Discipline::QuantumMechanics,
            vec![v("E","energy","J",""), v("ħ","reduced Planck","J·s",""), v("ω","angular frequency","rad/s","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("Δx * Δp ≥ ħ/2", r"\Delta x\,\Delta p \geq \hbar/2",
            Discipline::QuantumMechanics,
            vec![v("Δx","position uncertainty","m",""), v("Δp","momentum uncertainty","kg·m/s","")],
            vec![2,3], &["canonical conjugates"], 0.95),

        PhysicsEquation::new("[x̂, p̂] = iħ", r"[\hat{x},\hat{p}]=i\hbar",
            Discipline::QuantumMechanics,
            vec![v("x̂","position operator","",""), v("p̂","momentum operator","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("ψ(x,t) = A * exp(i*(k*x - ω*t))", r"\psi = Ae^{i(kx-\omega t)}",
            Discipline::QuantumMechanics,
            vec![v("k","wave number","1/m",""), v("ω","angular frequency","rad/s","")],
            vec![], &["free particle","plane wave"], 0.93),

        // ── Quantum Field Theory ──────────────────────────────────────────
        PhysicsEquation::new("L = ψ̄(iγ^μ∂_μ - m)ψ (Dirac)", r"\mathcal{L}=\bar\psi(i\gamma^\mu\partial_\mu-m)\psi",
            Discipline::QuantumFieldTheory,
            vec![v("ψ","Dirac spinor","",""), v("γ^μ","gamma matrices","",""), v("m","mass","","")],
            vec![3], &["special relativity","flat spacetime"], 0.90),

        PhysicsEquation::new("L = -1/4 * F_μν * F^μν (gauge field)", r"\mathcal{L}=-\frac{1}{4}F_{\mu\nu}F^{\mu\nu}",
            Discipline::QuantumFieldTheory,
            vec![v("F_μν","field strength tensor","","")],
            vec![3], &["U(1) gauge invariance"], 0.90),

        // ── Special Relativity ────────────────────────────────────────────
        PhysicsEquation::new("E^2 = (p*c)^2 + (m*c^2)^2", r"E^2=(pc)^2+(mc^2)^2",
            Discipline::SpecialRelativity,
            vec![v("E","energy","J",""), v("p","momentum","kg·m/s",""), v("m","rest mass","kg",""), v("c","speed of light","m/s","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("t' = γ * (t - v*x/c^2)", r"t'=\gamma(t-vx/c^2)",
            Discipline::SpecialRelativity,
            vec![v("γ","Lorentz factor","",""), v("v","relative velocity","m/s",""), v("c","speed of light","m/s","")],
            vec![], &["inertial frames"], 0.95),

        PhysicsEquation::new("γ = 1 / sqrt(1 - v^2/c^2)", r"\gamma=\frac{1}{\sqrt{1-v^2/c^2}}",
            Discipline::SpecialRelativity,
            vec![v("γ","Lorentz factor","",""), v("v","velocity","m/s",""), v("c","speed of light","m/s","")],
            vec![], &["v < c"], 0.95),

        // ── General Relativity ────────────────────────────────────────────
        PhysicsEquation::new("G_μν + Λ*g_μν = 8πG/c^4 * T_μν", r"G_{\mu\nu}+\Lambda g_{\mu\nu}=\frac{8\pi G}{c^4}T_{\mu\nu}",
            Discipline::GeneralRelativity,
            vec![v("G_μν","Einstein tensor","",""), v("T_μν","stress-energy tensor","",""), v("Λ","cosmological constant","","")],
            vec![3], &["classical gravity","smooth spacetime"], 0.95),

        // ── Fluid Dynamics ─────────────────────────────────────────────────
        PhysicsEquation::new("ρ(∂v/∂t + v·∇v) = -∇P + μ∇²v + f", r"\rho(\partial_t\mathbf{v}+\mathbf{v}\cdot\nabla\mathbf{v})=-\nabla P+\mu\nabla^2\mathbf{v}+\mathbf{f}",
            Discipline::FluidDynamics,
            vec![v("ρ","density","kg/m³",""), v("v","velocity","m/s",""), v("P","pressure","Pa",""), v("μ","viscosity","Pa·s","")],
            vec![2,3], &["Newtonian fluid","incompressible"], 0.92),

        PhysicsEquation::new("∇ · v = 0 (incompressible)", r"\nabla\cdot\mathbf{v}=0",
            Discipline::FluidDynamics,
            vec![v("v","velocity field","m/s","")],
            vec![2,3], &["incompressible flow"], 0.92),

        PhysicsEquation::new("Re = ρ * v * L / μ", r"Re = \frac{\rho v L}{\mu}",
            Discipline::FluidDynamics,
            vec![v("Re","Reynolds number","",""), v("L","characteristic length","m","")],
            vec![2,3], &[], 0.93),

        // ── Chaos / Nonlinear Dynamics ────────────────────────────────────
        PhysicsEquation::new("λ = lim(t→∞) (1/t) * ln(|δZ(t)| / |δZ(0)|)", r"\lambda=\lim_{t\to\infty}\frac{1}{t}\ln\frac{|\delta Z(t)|}{|\delta Z(0)|}",
            Discipline::ChaosDynamics,
            vec![v("λ","Lyapunov exponent","1/s",""), v("δZ","perturbation","","")],
            vec![], &["ergodic system"], 0.90),

        PhysicsEquation::new("dx/dt = σ*(y-x); dy/dt = x*(ρ-z)-y; dz/dt = x*y - β*z (Lorenz)",
            r"\dot{x}=\sigma(y-x),\;\dot{y}=x(\rho-z)-y,\;\dot{z}=xy-\beta z",
            Discipline::ChaosDynamics,
            vec![v("σ","Prandtl number","",""), v("ρ","Rayleigh number","",""), v("β","geometric factor","","")],
            vec![3], &[], 0.90),

        PhysicsEquation::new("x_{n+1} = r * x_n * (1 - x_n) (logistic map)", r"x_{n+1}=rx_n(1-x_n)",
            Discipline::ChaosDynamics,
            vec![v("x_n","population fraction","",""), v("r","growth rate","","")],
            vec![], &["discrete time","bounded population"], 0.90),

        // ── Topological Physics (Anyons live here) ────────────────────────
        // Anyons are quasiparticles that only exist in 2-D systems.
        // Under exchange, their wave function acquires a phase e^{iθ} where
        // θ can be anything — not just 0 (bosons) or π (fermions).
        PhysicsEquation::new("ψ → e^(iθ) * ψ (anyon exchange in 2D)", r"\psi \to e^{i\theta}\psi",
            Discipline::TopologicalPhysics,
            vec![v("θ","anyonic statistical angle","rad",""), v("ψ","many-body wave function","","")],
            vec![2], &["strictly 2-D system","topological order","quasiparticle excitations"], 0.88),

        PhysicsEquation::new("L_CS = k/4π * ε^μνρ * A_μ * ∂_ν * A_ρ (Chern-Simons)", r"\mathcal{L}_{CS}=\frac{k}{4\pi}\varepsilon^{\mu\nu\rho}A_\mu\partial_\nu A_\rho",
            Discipline::TopologicalPhysics,
            vec![v("k","level (integer)","",""), v("A_μ","gauge field","","")],
            vec![2], &["2+1 dimensional spacetime","topological field theory"], 0.85),

        PhysicsEquation::new("γ_Berry = i * ∮ <ψ|∇_R|ψ> · dR (Berry phase)", r"\gamma=i\oint\langle\psi|\nabla_R|\psi\rangle\cdot dR",
            Discipline::TopologicalPhysics,
            vec![v("γ","Berry phase","rad",""), v("R","parameter space path","","")],
            vec![2,3], &["adiabatic evolution","closed path in parameter space"], 0.88),

        PhysicsEquation::new("σ_H = ν * e^2 / h (quantum Hall)", r"\sigma_H = \nu\frac{e^2}{h}",
            Discipline::TopologicalPhysics,
            vec![v("σ_H","Hall conductance","S",""), v("ν","filling factor (integer or fractional)","",""), v("h","Planck constant","J·s","")],
            vec![2], &["strong magnetic field","low temperature","2-D electron gas"], 0.90),

        // ── Condensed Matter ──────────────────────────────────────────────
        PhysicsEquation::new("E_gap = 2Δ (BCS superconductor gap)", r"E_{gap}=2\Delta",
            Discipline::CondensedMatter,
            vec![v("Δ","superconducting gap","J",""), v("E_gap","energy gap","J","")],
            vec![], &["BCS theory","weak coupling"], 0.88),

        // ── Cosmology ─────────────────────────────────────────────────────
        PhysicsEquation::new("H = ȧ / a (Hubble parameter)", r"H = \dot{a}/a",
            Discipline::Cosmology,
            vec![v("H","Hubble parameter","km/s/Mpc",""), v("a","scale factor","","")],
            vec![], &["isotropic and homogeneous universe","FLRW metric"], 0.90),

        PhysicsEquation::new("(ȧ/a)^2 = 8πG/3 * ρ - k*c^2/a^2 + Λ*c^2/3 (Friedmann)", r"\left(\frac{\dot a}{a}\right)^2=\frac{8\pi G}{3}\rho-\frac{kc^2}{a^2}+\frac{\Lambda c^2}{3}",
            Discipline::Cosmology,
            vec![v("k","curvature parameter","",""), v("ρ","energy density","J/m³","")],
            vec![], &["FLRW cosmology"], 0.90),

        // ── Information Theory ────────────────────────────────────────────
        PhysicsEquation::new("H = -Σ p_i * log2(p_i) (Shannon entropy)", r"H=-\sum_i p_i\log_2 p_i",
            Discipline::InformationTheory,
            vec![v("H","Shannon entropy","bits",""), v("p_i","probability of outcome i","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("I(X;Y) = H(X) - H(X|Y) (mutual information)", r"I(X;Y)=H(X)-H(X|Y)",
            Discipline::InformationTheory,
            vec![v("I","mutual information","bits",""), v("H","entropy","bits","")],
            vec![], &[], 0.93),
    ]
}

fn build_seed_links(eqs: &[PhysicsEquation]) -> Vec<EquationLink> {
    // Build a quick lookup by text fragment for legible link construction.
    let by_text: HashMap<&str, &str> = eqs.iter()
        .map(|e| (e.text.as_str(), e.id.as_str()))
        .collect();

    let mut links = Vec::new();

    let mut link = |from_frag: &str, to_frag: &str, rel: LinkType, w: f32, note: &str| {
        // Match by substring to survive minor text changes
        let from_id = by_text.iter().find(|(k, _)| k.contains(from_frag)).map(|(_, v)| *v);
        let to_id   = by_text.iter().find(|(k, _)| k.contains(to_frag)).map(|(_, v)| *v);
        if let (Some(f), Some(t)) = (from_id, to_id) {
            if f != t {
                links.push(EquationLink {
                    from_id: f.to_string(),
                    to_id: t.to_string(),
                    relation_type: rel,
                    weight: w,
                    notes: note.to_string(),
                    created_at: now_unix(),
                });
            }
        }
    };

    // Classical → Relativity limit
    link("F = m * a", "E^2 = (p*c)^2", LinkType::SpecialCase, 0.9, "F=ma is the v≪c limit of relativistic dynamics");
    link("p = m * v", "E^2 = (p*c)^2", LinkType::SpecialCase, 0.9, "Newtonian momentum is the low-velocity limit");
    link("p = m * v", "t' = γ", LinkType::Approximates, 0.8, "Newtonian momentum is non-relativistic approximation");

    // Lagrangian ↔ Hamiltonian
    link("L = T - V", "H = T + V", LinkType::Bridges, 0.95, "Legendre transform connects Lagrangian and Hamiltonian");
    link("d/dt(∂L/∂q̇)", "H = T + V", LinkType::DerivesFrom, 0.9, "Hamilton's equations derive from Euler-Lagrange via Legendre");

    // Lagrangian → Newton
    link("d/dt(∂L/∂q̇)", "F = m * a", LinkType::Generalizes, 0.9, "Euler-Lagrange generalizes Newton to arbitrary coordinates");

    // Thermodynamics ↔ Statistical Mechanics
    link("dU = T * dS", "S = k_B * ln(Ω)", LinkType::Bridges, 0.95, "Boltzmann connects micro (Ω) to macro (S,T)");
    link("S = k_B * ln(Ω)", "Z = Σ exp", LinkType::DerivesFrom, 0.9, "Partition function is foundation of statistical mechanics");

    // Schrödinger → Classical limit
    link("iħ * ∂ψ/∂t", "d/dt(∂L/∂q̇)", LinkType::Generalizes, 0.85, "Schrödinger equation generalizes classical Lagrangian mechanics");
    link("iħ * ∂ψ/∂t", "H = T + V", LinkType::DerivesFrom, 0.9, "Hamiltonian operator in Schrödinger eq = classical Hamiltonian quantized");

    // QM uncertainty → classical limit
    link("Δx * Δp ≥ ħ/2", "F = m * a", LinkType::Approximates, 0.7, "Classical mechanics is limit ħ→0 of quantum mechanics");

    // Maxwell → EM wave / photon
    link("∇ × E = -∂B/∂t", "E = ħ * ω", LinkType::Bridges, 0.9, "EM wave quantization gives photon energy");
    link("∇ × B = μ_0", "∇ × E = -∂B/∂t", LinkType::Bridges, 0.95, "Coupled Maxwell equations produce EM waves");

    // Relativity ↔ GR
    link("E^2 = (p*c)^2", "G_μν + Λ*g_μν", LinkType::SpecialCase, 0.85, "Special relativity is GR in flat spacetime");
    link("G_μν + Λ*g_μν", "(ȧ/a)^2", LinkType::DerivesFrom, 0.9, "Friedmann equations derive from Einstein field equations");

    // Chaos ↔ statistical mechanics (ergodic bridge)
    link("λ = lim(t→∞)", "Z = Σ exp", LinkType::Bridges, 0.7, "Ergodic chaos underpins statistical mechanics ensemble equivalence");

    // Information ↔ thermodynamics (Boltzmann-Shannon equivalence)
    link("H = -Σ p_i", "S = k_B * ln(Ω)", LinkType::Bridges, 0.95, "Shannon entropy and Boltzmann entropy have identical form; related by k_B");

    // Anyons ↔ quantum Hall
    link("ψ → e^(iθ)", "σ_H = ν * e^2 / h", LinkType::Bridges, 0.90, "Anyons are the quasiparticle excitations of the fractional quantum Hall state");
    link("L_CS = k/4π", "ψ → e^(iθ)", LinkType::DerivesFrom, 0.88, "Chern-Simons action governs anyon statistics");
    link("γ_Berry = i * ∮", "ψ → e^(iθ)", LinkType::Bridges, 0.85, "Berry phase of anyon braiding is the statistical angle θ");

    // Fluid → chaos
    link("ρ(∂v/∂t", "dx/dt = σ*(y-x)", LinkType::Bridges, 0.80, "Lorenz equations derived from Navier-Stokes truncation");
    link("Re = ρ * v * L", "λ = lim(t→∞)", LinkType::Bridges, 0.75, "High Reynolds number turbulence exhibits positive Lyapunov exponents");

    links
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_and_search() {
        let rt = EquationMatrixRuntime::new(EquationMatrixConfig::default());
        let results = rt.search("force mass acceleration", None);
        assert!(!results.is_empty(), "should find F=ma");
        assert!(results[0].equation.text.contains("F = m"));
    }

    #[test]
    fn anyon_is_2d_only() {
        let rt = EquationMatrixRuntime::new(EquationMatrixConfig::default());
        let ctx2 = rt.apply_to_context(&["anyon".into(), "topological".into(), "braid".into()], 2);
        let ctx3 = rt.apply_to_context(&["anyon".into(), "topological".into(), "braid".into()], 3);
        let has_anyon_2d = ctx2.candidates.iter().any(|r| r.equation.text.contains("anyon"));
        let has_anyon_3d = ctx3.candidates.iter().any(|r| r.equation.text.contains("anyon"));
        assert!(has_anyon_2d, "anyon equation should appear in 2D context");
        assert!(!has_anyon_3d, "anyon equation must NOT appear in 3D context");
    }

    #[test]
    fn peer_merge() {
        let a = EquationMatrixRuntime::new(EquationMatrixConfig::default());
        let b = EquationMatrixRuntime::new(EquationMatrixConfig { seed_on_init: false, ..Default::default() });
        let payload = a.to_peer_payload("node-a");
        b.merge_peer_payload(payload);
        let rep = b.report();
        assert!(rep.total_equations > 0, "b should receive equations from a");
    }
}
