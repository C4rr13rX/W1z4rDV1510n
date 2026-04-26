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
            Self::Custom(s) => match s.as_str() {
                "GameTheory" => &["nash","equilibrium","strategy","payoff","game","minimax","player","dominant","cooperation","defection","prisoner","dilemma","auction","mechanism","coalition","shapley","replicator","evolutionary","stable","bayesian","hotelling","focal","schelling","zero-sum"],
                "MarketingScience" => &["diffusion","adoption","viral","spread","bass","imitation","word","mouth","contagion","cascade","influence","persuasion","advertising","brand","market","share","elasticity","network","effect","preference","metcalfe","zipf","pareto","clv","retention","churn","adstock","lanchester","gompertz","threshold","logistic","sigmoid","growth"],
                "CrossDisciplinary" => &["emergence","self-organization","complexity","power","law","scale","free","percolation","ising","mean","field","opinion","tipping","point","critical","phase","transition","fitness","landscape","coevolution","arms","race","cascade","failure","segregation","free","energy","maximum","entropy","jaynes","universal"],
                // ── Engineering & applied science disciplines (added 2026-04-26) ──
                "ControlTheory" => &["controller","feedback","feedforward","pid","proportional","integral","derivative","stability","lyapunov","routh","hurwitz","pole","zero","gain","phase","margin","transfer","function","plant","setpoint","steady","state","error","damping","overshoot","laplace","z-transform","state","space","kalman","filter","lqr","mpc","predictive"],
                "SignalProcessing" => &["fourier","transform","frequency","spectrum","aliasing","nyquist","sampling","convolution","filter","fft","dft","stft","wavelet","window","hann","hamming","bandpass","lowpass","highpass","fir","iir","z-transform","modulation","demodulation","snr","decibel","quantization","interpolation","correlation","spectral"],
                "Chemistry" => &["rate","arrhenius","equilibrium","reaction","reactant","product","catalyst","stoichiometry","activation","energy","enthalpy","entropy","gibbs","spontaneous","ph","poh","acid","base","buffer","henderson","hasselbalch","beer","lambert","absorbance","concentration","molarity","molality","oxidation","reduction","redox","nernst","electrochemistry","le","chatelier","kinetics"],
                "ComputerScience" => &["algorithm","complexity","big","o","amortized","queue","stack","tree","graph","heap","hash","sort","search","traversal","recurrence","master","theorem","amdahl","gustafson","brewer","cap","fault","tolerance","caching","page","fault","memory","hierarchy","cache","line","throughput","latency","little","queueing","poisson","arrival","service","utilization"],
                "Optimization" => &["gradient","descent","newton","method","conjugate","quasi","bfgs","lagrange","multiplier","karush","kuhn","tucker","kkt","duality","simplex","interior","point","convex","constraint","feasible","optimal","minimize","maximize","objective","loss","penalty","barrier","stochastic","sgd","momentum","adam","rmsprop","adagrad","line","search","trust","region"],
                "MachineLearning" => &["perceptron","sigmoid","relu","tanh","softmax","cross","entropy","mean","squared","error","backpropagation","weight","update","layer","activation","embedding","attention","transformer","convolution","pooling","dropout","batch","norm","layer","norm","gradient","clipping","learning","rate","schedule","cosine","warmup","f1","precision","recall","auc","roc","bleu","perplexity","kl","divergence"],
                "ElectricalEngineering" => &["voltage","current","resistance","ohm","kirchhoff","kvl","kcl","capacitor","inductor","resistor","impedance","admittance","reactance","time","constant","rc","rl","rlc","resonance","power","factor","real","reactive","apparent","watt","volt","ampere","amp","ohms","henry","farad","tesla","weber","decibel","db","gain","bode","nyquist"],
                "StructuralEngineering" => &["stress","strain","modulus","young","elastic","plastic","yield","ultimate","tensile","compressive","shear","bending","moment","beam","deflection","euler","buckling","critical","load","mohr","circle","von","mises","fatigue","creep","fracture","toughness","poisson","ratio","hooke","truss","frame","column","slenderness"],
                "Epidemiology" => &["susceptible","infected","recovered","exposed","sir","seir","reproduction","number","r0","attack","rate","incubation","latent","period","contagion","transmission","contact","tracing","vaccination","threshold","herd","immunity","epidemic","pandemic","outbreak","prevalence","incidence","mortality","case","fatality","compartmental"],
                "Economics" => &["utility","supply","demand","equilibrium","price","quantity","elasticity","cobb","douglas","production","function","is-lm","aggregate","phillips","curve","unemployment","inflation","gdp","time","value","money","present","future","npv","irr","discount","rate","compound","interest","capm","beta","sharpe","arbitrage","black","scholes","option","strike","volatility","garch","heteroskedasticity"],
                _ => &[],
            },
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
    /// Node ID that first reported this gap (None = locally originated).
    #[serde(default)]
    pub first_node_id: Option<String>,
    /// All node IDs that have reported this gap (for corroboration scoring).
    #[serde(default)]
    pub reporting_nodes: Vec<String>,
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

    /// Combined apply + reinforce step.  Calls `apply_to_context`, reinforces
    /// every candidate whose relevance ≥ `relevance_threshold`, and returns
    /// the matched equations together with their discipline keywords.  The
    /// keyword list is what `motif_eem_feedback_step` feeds back into the
    /// motif runtime so equation matches drive new motif activations — the
    /// network develops "physics motifs" as the input stream re-fires the
    /// same equation patterns over time.
    pub fn apply_and_reinforce(
        &self,
        active_labels: &[String],
        sensor_dims: u8,
        relevance_threshold: f32,
    ) -> Vec<(EquationSearchResult, Vec<String>)> {
        let result = self.apply_to_context(active_labels, sensor_dims);
        let mut out = Vec::new();
        for cand in result.candidates {
            if cand.relevance < relevance_threshold { continue; }
            self.reinforce(&cand.equation.id);
            let keywords: Vec<String> = cand.equation.discipline.keywords()
                .iter().map(|k| (*k).to_string()).collect();
            out.push((cand, keywords));
        }
        out
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
    /// `reporter_node_id` is the node that observed this gap (None = this node).
    pub fn open_gap(
        &self,
        trigger_labels: Vec<String>,
        sensor_dims: u8,
        description: &str,
        reporter_node_id: Option<&str>,
    ) {
        let mut sorted = trigger_labels.clone();
        sorted.sort_unstable();
        let id_src = format!("gap::{}::{}", sensor_dims, sorted.join(","));
        let id = compute_payload_hash(id_src.as_bytes());
        let mut st = self.state.lock();
        if let Some(existing) = st.open_gaps.iter_mut().find(|g| g.id == id) {
            existing.observation_count += 1;
            if let Some(node) = reporter_node_id {
                if !existing.reporting_nodes.iter().any(|n| n == node) {
                    existing.reporting_nodes.push(node.to_string());
                }
            }
            return;
        }
        if st.open_gaps.len() < self.config.max_open_gaps {
            let reporting_nodes = reporter_node_id.map(|n| vec![n.to_string()]).unwrap_or_default();
            st.open_gaps.push(HypothesisSlot {
                id,
                description: description.to_string(),
                trigger_labels,
                sensor_dims,
                created_at: now_unix(),
                observation_count: 1,
                first_node_id: reporter_node_id.map(|n| n.to_string()),
                reporting_nodes,
            });
        }
    }

    /// Returns all open hypothesis slots, sorted by corroboration (most-reported first).
    pub fn open_gaps(&self) -> Vec<HypothesisSlot> {
        let mut st = self.state.lock();
        st.open_gaps.sort_by(|a, b| {
            let a_score = a.observation_count + a.reporting_nodes.len() as u32;
            let b_score = b.observation_count + b.reporting_nodes.len() as u32;
            b_score.cmp(&a_score)
        });
        st.open_gaps.clone()
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
            if let Some(existing) = st.open_gaps.iter_mut().find(|g| g.id == gap.id) {
                existing.observation_count = existing.observation_count.saturating_add(gap.observation_count);
                for node in &gap.reporting_nodes {
                    if !existing.reporting_nodes.iter().any(|n| n == node) {
                        existing.reporting_nodes.push(node.clone());
                    }
                }
                // Propagate source node attribution
                if existing.first_node_id.is_none() {
                    existing.first_node_id = gap.first_node_id.clone()
                        .or_else(|| Some(payload.source_node_id.clone()));
                }
            } else if st.open_gaps.len() < self.config.max_open_gaps {
                let mut incoming = gap;
                // Mark the peer as an additional reporter
                if !incoming.reporting_nodes.iter().any(|n| n == &payload.source_node_id) {
                    incoming.reporting_nodes.push(payload.source_node_id.clone());
                }
                if incoming.first_node_id.is_none() {
                    incoming.first_node_id = Some(payload.source_node_id.clone());
                }
                st.open_gaps.push(incoming);
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
    let mut eqs = vec![
        // ══════════════════════════════════════════════════════════════════
        // CLASSICAL MECHANICS
        // ══════════════════════════════════════════════════════════════════
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
    ];

    // ══════════════════════════════════════════════════════════════════════
    // CLASSICAL MECHANICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("τ = r × F (torque)", r"\boldsymbol{\tau}=\mathbf{r}\times\mathbf{F}",
            Discipline::ClassicalMechanics,
            vec![v("τ","torque","N·m","[M L^2 T^-2]"), v("r","position vector","m","[L]"), v("F","force","N","[M L T^-2]")],
            vec![3], &["rigid body"], 0.95),

        PhysicsEquation::new("L_ang = r × p (angular momentum)", r"\mathbf{L}=\mathbf{r}\times\mathbf{p}",
            Discipline::ClassicalMechanics,
            vec![v("L_ang","angular momentum","kg·m²/s","[M L^2 T^-1]"), v("r","position","m",""), v("p","momentum","kg·m/s","")],
            vec![3], &[], 0.95),

        PhysicsEquation::new("I = Σ m_i * r_i^2 (moment of inertia)", r"I=\sum_i m_i r_i^2",
            Discipline::ClassicalMechanics,
            vec![v("I","moment of inertia","kg·m²","[M L^2]"), v("r_i","distance from axis","m","")],
            vec![2,3], &["rigid body"], 0.93),

        PhysicsEquation::new("τ = I * α (rotational Newton 2nd law)", r"\tau=I\alpha",
            Discipline::ClassicalMechanics,
            vec![v("α","angular acceleration","rad/s²","[T^-2]")],
            vec![2,3], &["rigid body"], 0.93),

        PhysicsEquation::new("E_total = E_k + E_p (conservation of energy)", r"E_{total}=E_k+E_p=\text{const}",
            Discipline::ClassicalMechanics,
            vec![v("E_k","kinetic energy","J",""), v("E_p","potential energy","J","")],
            vec![2,3], &["conservative forces","isolated system"], 0.95),

        PhysicsEquation::new("v^2 = v_0^2 + 2*a*d (kinematic)", r"v^2=v_0^2+2ad",
            Discipline::ClassicalMechanics,
            vec![v("v","final velocity","m/s",""), v("v_0","initial velocity","m/s",""), v("d","displacement","m","")],
            vec![2,3], &["constant acceleration"], 0.93),

        PhysicsEquation::new("F_s = -k * x (Hooke's law)", r"F_s=-kx",
            Discipline::ClassicalMechanics,
            vec![v("k","spring constant","N/m","[M T^-2]"), v("x","displacement","m","[L]")],
            vec![2,3], &["elastic limit not exceeded"], 0.95),

        PhysicsEquation::new("ω = sqrt(k/m) (simple harmonic oscillator)", r"\omega=\sqrt{k/m}",
            Discipline::ClassicalMechanics,
            vec![v("ω","angular frequency","rad/s",""), v("k","spring constant","N/m",""), v("m","mass","kg","")],
            vec![2,3], &["no damping"], 0.93),

        PhysicsEquation::new("x(t) = A*cos(ω*t + φ) (SHO solution)", r"x(t)=A\cos(\omega t+\varphi)",
            Discipline::ClassicalMechanics,
            vec![v("A","amplitude","m",""), v("φ","phase","rad","")],
            vec![2,3], &["undamped harmonic oscillator"], 0.93),

        PhysicsEquation::new("J = F * Δt (impulse-momentum theorem)", r"\mathbf{J}=\mathbf{F}\Delta t=\Delta\mathbf{p}",
            Discipline::ClassicalMechanics,
            vec![v("J","impulse","N·s","[M L T^-1]")],
            vec![2,3], &[], 0.93),

        PhysicsEquation::new("v_cm = Σ(m_i * v_i) / M (center of mass velocity)", r"\mathbf{v}_{cm}=\frac{\sum m_i\mathbf{v}_i}{M}",
            Discipline::ClassicalMechanics,
            vec![v("M","total mass","kg",""), v("v_cm","center of mass velocity","m/s","")],
            vec![2,3], &[], 0.92),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // WAVES AND OSCILLATIONS
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("∂²u/∂t² = c² * ∇²u (wave equation)", r"\frac{\partial^2 u}{\partial t^2}=c^2\nabla^2 u",
            Discipline::ClassicalMechanics,
            vec![v("u","wave displacement","",""), v("c","wave speed","m/s","")],
            vec![2,3], &["linear medium","non-dispersive"], 0.95),

        PhysicsEquation::new("f = 1/T (frequency-period)", r"f=1/T",
            Discipline::ClassicalMechanics,
            vec![v("f","frequency","Hz","[T^-1]"), v("T","period","s","[T]")],
            vec![], &[], 0.95),

        PhysicsEquation::new("v = f * λ (wave speed)", r"v=f\lambda",
            Discipline::ClassicalMechanics,
            vec![v("λ","wavelength","m","[L]"), v("f","frequency","Hz","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("I ∝ A^2 (wave intensity)", r"I\propto A^2",
            Discipline::ClassicalMechanics,
            vec![v("I","intensity","W/m²",""), v("A","amplitude","","")],
            vec![], &["linear medium"], 0.92),

        PhysicsEquation::new("β = 10*log10(I/I_0) (decibel scale)", r"\beta=10\log_{10}(I/I_0)",
            Discipline::ClassicalMechanics,
            vec![v("β","sound level","dB",""), v("I_0","reference intensity","W/m²","")],
            vec![], &[], 0.90),

        PhysicsEquation::new("f_obs = f_src*(v+v_obs)/(v+v_src) (Doppler)", r"f_{obs}=f_{src}\frac{v+v_{obs}}{v+v_{src}}",
            Discipline::ClassicalMechanics,
            vec![v("f_obs","observed frequency","Hz",""), v("v_obs","observer velocity","m/s",""), v("v_src","source velocity","m/s","")],
            vec![], &["non-relativistic","wave medium exists"], 0.92),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // LAGRANGIAN / HAMILTONIAN — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("S = ∫ L dt (action functional)", r"S=\int_{t_1}^{t_2}\mathcal{L}\,dt",
            Discipline::LagrangianMechanics,
            vec![v("S","action","J·s","[M L^2 T^-1]"), v("L","Lagrangian","J","")],
            vec![2,3], &["principle of stationary action"], 0.95),

        PhysicsEquation::new("δS = 0 (principle of stationary action)", r"\delta S=0",
            Discipline::LagrangianMechanics,
            vec![v("S","action","J·s","")],
            vec![], &["variational principle"], 0.95),

        PhysicsEquation::new("q_dot = ∂H/∂p; p_dot = -∂H/∂q (Hamilton's equations)", r"\dot{q}=\frac{\partial H}{\partial p},\quad\dot{p}=-\frac{\partial H}{\partial q}",
            Discipline::HamiltonianMechanics,
            vec![v("q","generalized coordinate","",""), v("p","conjugate momentum","",""), v("H","Hamiltonian","J","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("{A,B} = Σ(∂A/∂q*∂B/∂p - ∂A/∂p*∂B/∂q) (Poisson bracket)", r"\{A,B\}=\sum_i\!\left(\frac{\partial A}{\partial q_i}\frac{\partial B}{\partial p_i}-\frac{\partial A}{\partial p_i}\frac{\partial B}{\partial q_i}\right)",
            Discipline::HamiltonianMechanics,
            vec![v("A","observable","",""), v("B","observable","","")],
            vec![], &["classical phase space"], 0.92),

        PhysicsEquation::new("dA/dt = {A,H} + ∂A/∂t (equation of motion via Poisson bracket)", r"\frac{dA}{dt}=\{A,H\}+\frac{\partial A}{\partial t}",
            Discipline::HamiltonianMechanics,
            vec![v("A","observable","",""), v("H","Hamiltonian","J","")],
            vec![], &[], 0.92),

        PhysicsEquation::new("I_action = ∮ p dq (adiabatic invariant)", r"I=\oint p\,dq",
            Discipline::HamiltonianMechanics,
            vec![v("I_action","action variable","J·s",""), v("p","momentum","",""), v("q","coordinate","","")],
            vec![], &["adiabatic process","integrable system"], 0.88),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // THERMODYNAMICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("G = H - T*S (Gibbs free energy)", r"G=H-TS",
            Discipline::Thermodynamics,
            vec![v("G","Gibbs free energy","J",""), v("H","enthalpy","J",""), v("T","temperature","K",""), v("S","entropy","J/K","")],
            vec![], &["constant pressure"], 0.95),

        PhysicsEquation::new("A = U - T*S (Helmholtz free energy)", r"A=U-TS",
            Discipline::Thermodynamics,
            vec![v("A","Helmholtz free energy","J",""), v("U","internal energy","J",""), v("T","temperature","K","")],
            vec![], &["constant volume"], 0.93),

        PhysicsEquation::new("η_Carnot = 1 - T_cold/T_hot (Carnot efficiency)", r"\eta_{Carnot}=1-\frac{T_c}{T_h}",
            Discipline::Thermodynamics,
            vec![v("η","efficiency","",""), v("T_c","cold reservoir temperature","K",""), v("T_h","hot reservoir temperature","K","")],
            vec![], &["reversible heat engine"], 0.95),

        PhysicsEquation::new("dS = δQ_rev / T (Clausius)", r"dS=\frac{\delta Q_{rev}}{T}",
            Discipline::Thermodynamics,
            vec![v("S","entropy","J/K",""), v("Q","heat transferred reversibly","J",""), v("T","temperature","K","")],
            vec![], &["reversible process"], 0.95),

        PhysicsEquation::new("C_p - C_v = R (Mayer's relation)", r"C_p-C_v=R",
            Discipline::Thermodynamics,
            vec![v("C_p","heat capacity at constant pressure","J/(mol·K)",""), v("C_v","heat capacity at constant volume","J/(mol·K)","")],
            vec![], &["ideal gas"], 0.92),

        PhysicsEquation::new("μ_chem = (∂G/∂N)_{T,P} (chemical potential)", r"\mu=\left(\frac{\partial G}{\partial N}\right)_{T,P}",
            Discipline::Thermodynamics,
            vec![v("μ_chem","chemical potential","J/mol",""), v("N","particle number","","")],
            vec![], &[], 0.92),

        PhysicsEquation::new("(∂P/∂V)_T * (∂V/∂T)_P * (∂T/∂P)_V = -1 (cyclic relation)", r"\left(\frac{\partial P}{\partial V}\right)_T\!\left(\frac{\partial V}{\partial T}\right)_P\!\left(\frac{\partial T}{\partial P}\right)_V=-1",
            Discipline::Thermodynamics,
            vec![v("P","pressure","Pa",""), v("V","volume","m³",""), v("T","temperature","K","")],
            vec![], &["equation of state"], 0.88),

        PhysicsEquation::new("(P + a/V^2)(V - b) = RT (van der Waals)", r"\left(P+\frac{a}{V^2}\right)(V-b)=RT",
            Discipline::Thermodynamics,
            vec![v("a","intermolecular attraction","Pa·m^6/mol²",""), v("b","excluded volume","m³/mol","")],
            vec![], &["real gas correction"], 0.90),

        PhysicsEquation::new("Q = m * c_s * ΔT (heat capacity)", r"Q=mc_s\Delta T",
            Discipline::Thermodynamics,
            vec![v("c_s","specific heat capacity","J/(kg·K)",""), v("m","mass","kg","")],
            vec![], &["no phase change"], 0.95),

        PhysicsEquation::new("P = σ * T^4 (Stefan-Boltzmann blackbody)", r"P=\sigma T^4",
            Discipline::Thermodynamics,
            vec![v("σ","Stefan-Boltzmann constant","W/(m²·K^4)",""), v("T","temperature","K",""), v("P","radiated power per area","W/m²","")],
            vec![], &["blackbody radiation"], 0.95),

        PhysicsEquation::new("u(ν,T) = 8πhν³/c³ * 1/(exp(hν/k_B T)-1) (Planck blackbody)", r"u(\nu,T)=\frac{8\pi h\nu^3}{c^3}\frac{1}{e^{h\nu/k_BT}-1}",
            Discipline::Thermodynamics,
            vec![v("ν","frequency","Hz",""), v("u","spectral energy density","J/m³/Hz","")],
            vec![], &["thermal equilibrium radiation"], 0.95),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // STATISTICAL MECHANICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("F = -k_B * T * ln(Z) (Helmholtz from partition fn)", r"F=-k_BT\ln Z",
            Discipline::StatisticalMechanics,
            vec![v("F","Helmholtz free energy","J",""), v("Z","partition function","","")],
            vec![], &["canonical ensemble"], 0.93),

        PhysicsEquation::new("<E> = -∂ln(Z)/∂β (mean energy from partition fn)", r"\langle E\rangle=-\frac{\partial\ln Z}{\partial\beta}",
            Discipline::StatisticalMechanics,
            vec![v("β","inverse temperature 1/(k_B T)","J^-1",""), v("Z","partition function","","")],
            vec![], &["canonical ensemble"], 0.93),

        PhysicsEquation::new("f_FD(E) = 1/(exp((E-μ)/k_B T)+1) (Fermi-Dirac distribution)", r"f_{FD}=\frac{1}{e^{(E-\mu)/k_BT}+1}",
            Discipline::StatisticalMechanics,
            vec![v("f_FD","Fermi-Dirac occupation","",""), v("E","energy","J",""), v("μ_chem","chemical potential","J","")],
            vec![], &["fermions","Pauli exclusion"], 0.95),

        PhysicsEquation::new("f_BE(E) = 1/(exp((E-μ)/k_B T)-1) (Bose-Einstein distribution)", r"f_{BE}=\frac{1}{e^{(E-\mu)/k_BT}-1}",
            Discipline::StatisticalMechanics,
            vec![v("f_BE","Bose-Einstein occupation","",""), v("E","energy","J","")],
            vec![], &["bosons","indistinguishable particles"], 0.95),

        PhysicsEquation::new("n(E) = g(E) * f(E) (density of states × occupation)", r"n(E)=g(E)f(E)",
            Discipline::StatisticalMechanics,
            vec![v("n","electron density","1/J/m³",""), v("g","density of states","1/J/m³",""), v("f","occupation function","","")],
            vec![], &[], 0.92),

        PhysicsEquation::new("C_v = k_B * β² * <(ΔE)²> (heat capacity fluctuations)", r"C_v=k_B\beta^2\langle(\Delta E)^2\rangle",
            Discipline::StatisticalMechanics,
            vec![v("C_v","heat capacity","J/K",""), v("ΔE","energy fluctuation","J","")],
            vec![], &["canonical ensemble"], 0.90),

        PhysicsEquation::new("Ξ = Σ exp(-β(E_n - μ*N_n)) (grand partition function)", r"\Xi=\sum_{n}e^{-\beta(E_n-\mu N_n)}",
            Discipline::StatisticalMechanics,
            vec![v("Ξ","grand partition function","",""), v("N_n","particle number","","")],
            vec![], &["grand canonical ensemble"], 0.90),

        PhysicsEquation::new("ρ_eq = exp(-β*H) / Z (Boltzmann density matrix)", r"\hat\rho=\frac{e^{-\beta\hat H}}{Z}",
            Discipline::StatisticalMechanics,
            vec![v("ρ_eq","density matrix","",""), v("H","Hamiltonian","J","")],
            vec![], &["thermal equilibrium","quantum statistical mechanics"], 0.90),

        PhysicsEquation::new("P_i = exp(-E_i/k_B T) / Z (Boltzmann probability)", r"P_i=\frac{e^{-E_i/k_BT}}{Z}",
            Discipline::StatisticalMechanics,
            vec![v("P_i","probability of microstate i","","")],
            vec![], &["canonical ensemble"], 0.95),

        PhysicsEquation::new("ΔF = -k_B T * ln(<exp(-ΔU/k_B T)>) (Jarzynski equality)", r"\Delta F=-k_BT\ln\langle e^{-\Delta U/k_BT}\rangle",
            Discipline::StatisticalMechanics,
            vec![v("ΔF","free energy difference","J",""), v("ΔU","work performed","J","")],
            vec![], &["nonequilibrium process","connects to equilibrium"], 0.85),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // ELECTROMAGNETISM — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("F = q*(E + v×B) (Lorentz force)", r"\mathbf{F}=q(\mathbf{E}+\mathbf{v}\times\mathbf{B})",
            Discipline::Electromagnetism,
            vec![v("q","charge","C","[A T]"), v("E","electric field","V/m",""), v("B","magnetic field","T","")],
            vec![3], &[], 0.95),

        PhysicsEquation::new("V = I * R (Ohm's law)", r"V=IR",
            Discipline::Electromagnetism,
            vec![v("V","voltage","V","[M L^2 T^-3 A^-1]"), v("I","current","A","[A]"), v("R","resistance","Ω","[M L^2 T^-3 A^-2]")],
            vec![], &["linear resistor","ohmic material"], 0.95),

        PhysicsEquation::new("P = I * V = I²*R (electrical power)", r"P=IV=I^2R=V^2/R",
            Discipline::Electromagnetism,
            vec![v("P","power","W","[M L^2 T^-3]")],
            vec![], &[], 0.95),

        PhysicsEquation::new("U_E = (1/2)*ε_0*E² (electric field energy density)", r"u_E=\frac{1}{2}\varepsilon_0 E^2",
            Discipline::Electromagnetism,
            vec![v("u_E","electric energy density","J/m³","")],
            vec![3], &[], 0.93),

        PhysicsEquation::new("U_B = B²/(2*μ_0) (magnetic field energy density)", r"u_B=\frac{B^2}{2\mu_0}",
            Discipline::Electromagnetism,
            vec![v("u_B","magnetic energy density","J/m³","")],
            vec![3], &[], 0.93),

        PhysicsEquation::new("S_Poynting = E×H (Poynting vector, EM power flow)", r"\mathbf{S}=\mathbf{E}\times\mathbf{H}",
            Discipline::Electromagnetism,
            vec![v("S_Poynting","Poynting vector","W/m²",""), v("H","magnetic intensity","A/m","")],
            vec![3], &[], 0.93),

        PhysicsEquation::new("c = 1/sqrt(μ_0*ε_0) (speed of light from Maxwell)", r"c=\frac{1}{\sqrt{\mu_0\varepsilon_0}}",
            Discipline::Electromagnetism,
            vec![v("c","speed of light","m/s",""), v("μ_0","permeability","H/m",""), v("ε_0","permittivity","F/m","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("E = -∇V - ∂A/∂t (electric field from potentials)", r"\mathbf{E}=-\nabla V-\frac{\partial\mathbf{A}}{\partial t}",
            Discipline::Electromagnetism,
            vec![v("V","electric potential","V",""), v("A","magnetic vector potential","T·m","")],
            vec![3], &[], 0.93),

        PhysicsEquation::new("B = ∇×A (magnetic field from vector potential)", r"\mathbf{B}=\nabla\times\mathbf{A}",
            Discipline::Electromagnetism,
            vec![v("B","magnetic field","T",""), v("A","magnetic vector potential","T·m","")],
            vec![3], &[], 0.93),

        PhysicsEquation::new("∇²V = -ρ/ε_0 (Poisson equation electrostatics)", r"\nabla^2 V=-\frac{\rho}{\varepsilon_0}",
            Discipline::Electromagnetism,
            vec![v("V","electric potential","V",""), v("ρ","charge density","C/m³","")],
            vec![3], &["electrostatics"], 0.93),

        PhysicsEquation::new("Z_0 = sqrt(μ_0/ε_0) (impedance of free space)", r"Z_0=\sqrt{\mu_0/\varepsilon_0}\approx377\,\Omega",
            Discipline::Electromagnetism,
            vec![v("Z_0","impedance of free space","Ω","")],
            vec![], &[], 0.90),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // QUANTUM MECHANICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("-ħ²/(2m) * ∇²ψ + V*ψ = E*ψ (time-independent Schrödinger)", r"-\frac{\hbar^2}{2m}\nabla^2\psi+V\psi=E\psi",
            Discipline::QuantumMechanics,
            vec![v("ψ","wave function","",""), v("V","potential energy","J",""), v("E","energy eigenvalue","J",""), v("m","mass","kg","")],
            vec![2,3], &["stationary state","time-independent potential"], 0.95),

        PhysicsEquation::new("E_n = -m*e^4/(2ħ²*n²) (hydrogen energy levels)", r"E_n=-\frac{me^4}{2\hbar^2 n^2}=-\frac{13.6\,\text{eV}}{n^2}",
            Discipline::QuantumMechanics,
            vec![v("E_n","energy of nth level","J",""), v("n","principal quantum number","",""), v("e","electron charge","C","")],
            vec![], &["hydrogen atom","Bohr model","Coulomb potential"], 0.95),

        PhysicsEquation::new("<A> = ∫ ψ* Â ψ dτ (quantum expectation value)", r"\langle A\rangle=\int\psi^*\hat{A}\psi\,d\tau",
            Discipline::QuantumMechanics,
            vec![v("A","observable","",""), v("ψ","wave function","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("P = |ψ|² (Born rule, probability density)", r"P=|\psi|^2",
            Discipline::QuantumMechanics,
            vec![v("P","probability density","1/m³",""), v("ψ","wave function","","")],
            vec![2,3], &[], 0.95),

        PhysicsEquation::new("ψ_total = Σ c_n * φ_n (superposition principle)", r"\psi=\sum_n c_n\phi_n",
            Discipline::QuantumMechanics,
            vec![v("c_n","expansion coefficient","",""), v("φ_n","eigenstate","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("d<A>/dt = i/ħ * <[H,A]> + <∂A/∂t> (Ehrenfest theorem)", r"\frac{d\langle A\rangle}{dt}=\frac{i}{\hbar}\langle[H,A]\rangle+\left\langle\frac{\partial A}{\partial t}\right\rangle",
            Discipline::QuantumMechanics,
            vec![v("A","observable","",""), v("H","Hamiltonian","J","")],
            vec![], &[], 0.92),

        PhysicsEquation::new("|ψ(t)> = exp(-i*H*t/ħ)|ψ(0)> (time evolution operator)", r"|\psi(t)\rangle=e^{-i\hat H t/\hbar}|\psi(0)\rangle",
            Discipline::QuantumMechanics,
            vec![v("H","Hamiltonian operator","J","")],
            vec![], &["time-independent H"], 0.93),

        PhysicsEquation::new("T = exp(-2*κ*L) (quantum tunneling probability)", r"T\approx e^{-2\kappa L}",
            Discipline::QuantumMechanics,
            vec![v("T","transmission probability","",""), v("κ","decay constant","1/m",""), v("L","barrier width","m","")],
            vec![], &["rectangular barrier","thick barrier approximation"], 0.90),

        PhysicsEquation::new("κ = sqrt(2m(V-E))/ħ (tunneling decay constant)", r"\kappa=\frac{\sqrt{2m(V-E)}}{\hbar}",
            Discipline::QuantumMechanics,
            vec![v("κ","decay constant","1/m",""), v("V","barrier height","J",""), v("E","particle energy","J","")],
            vec![], &["classically forbidden region"], 0.90),

        PhysicsEquation::new("E_n = ħ*ω*(n + 1/2) (quantum harmonic oscillator)", r"E_n=\hbar\omega\!\left(n+\tfrac{1}{2}\right)",
            Discipline::QuantumMechanics,
            vec![v("E_n","energy eigenvalue","J",""), v("n","quantum number","",""), v("ω","angular frequency","rad/s","")],
            vec![], &["harmonic potential"], 0.95),

        PhysicsEquation::new("[S_i, S_j] = iħ*ε_ijk*S_k (spin commutation)", r"[S_i,S_j]=i\hbar\varepsilon_{ijk}S_k",
            Discipline::QuantumMechanics,
            vec![v("S","spin operator","ħ",""), v("ε_ijk","Levi-Civita symbol","","")],
            vec![], &[], 0.93),

        PhysicsEquation::new("S² |s,m> = ħ²*s(s+1)|s,m> (spin eigenvalue)", r"\hat{S}^2|s,m\rangle=\hbar^2 s(s+1)|s,m\rangle",
            Discipline::QuantumMechanics,
            vec![v("s","spin quantum number","",""), v("m","magnetic quantum number","","")],
            vec![], &[], 0.93),

        PhysicsEquation::new("ρ_density = |ψ><ψ| (density matrix pure state)", r"\hat\rho=|\psi\rangle\langle\psi|",
            Discipline::QuantumMechanics,
            vec![v("ρ_density","density matrix","","")],
            vec![], &["pure quantum state"], 0.92),

        PhysicsEquation::new("S_vN = -Tr(ρ ln ρ) (von Neumann entropy)", r"S_{vN}=-\text{Tr}(\hat\rho\ln\hat\rho)",
            Discipline::QuantumMechanics,
            vec![v("S_vN","von Neumann entropy","",""), v("ρ","density matrix","","")],
            vec![], &["quantum information","entanglement measure"], 0.92),

        PhysicsEquation::new("|Bell⟩ = (|00⟩ + |11⟩)/√2 (Bell state entanglement)", r"|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}}",
            Discipline::QuantumMechanics,
            vec![],
            vec![], &["two-qubit system","maximally entangled"], 0.90),

        PhysicsEquation::new("de Broglie: λ = h/p", r"\lambda=h/p",
            Discipline::QuantumMechanics,
            vec![v("λ","de Broglie wavelength","m",""), v("p","momentum","kg·m/s",""), v("h","Planck constant","J·s","")],
            vec![], &[], 0.95),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // QUANTUM FIELD THEORY — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("Z = ∫ Dφ exp(iS[φ]/ħ) (Feynman path integral)", r"Z=\int\mathcal{D}\phi\,e^{iS[\phi]/\hbar}",
            Discipline::QuantumFieldTheory,
            vec![v("Z","partition function / generating functional","",""), v("S","action","J·s",""), v("φ","field configuration","","")],
            vec![3], &["quantum field theory","perturbative or non-perturbative"], 0.90),

        PhysicsEquation::new("□φ + m²φ = 0 (Klein-Gordon equation)", r"(\Box+m^2)\phi=0",
            Discipline::QuantumFieldTheory,
            vec![v("φ","scalar field","",""), v("m","mass","kg",""), v("□","d'Alembertian","","")],
            vec![3], &["free scalar field","special relativity"], 0.92),

        PhysicsEquation::new("(iγ^μ∂_μ - m)ψ = 0 (Dirac equation)", r"(i\gamma^\mu\partial_\mu-m)\psi=0",
            Discipline::QuantumFieldTheory,
            vec![v("ψ","Dirac spinor","",""), v("γ^μ","Dirac gamma matrices","",""), v("m","rest mass","kg","")],
            vec![3], &["relativistic fermion"], 0.92),

        PhysicsEquation::new("L_QED = ψ̄(iγ^μD_μ - m)ψ - 1/4*F_μν*F^μν (QED Lagrangian)", r"\mathcal{L}_{QED}=\bar\psi(i\gamma^\mu D_\mu-m)\psi-\tfrac{1}{4}F_{\mu\nu}F^{\mu\nu}",
            Discipline::QuantumFieldTheory,
            vec![v("D_μ","covariant derivative","",""), v("ψ","electron field","","")],
            vec![3], &["quantum electrodynamics","U(1) gauge theory"], 0.90),

        PhysicsEquation::new("α = e²/(4π*ε_0*ħ*c) ≈ 1/137 (fine structure constant)", r"\alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}\approx\frac{1}{137}",
            Discipline::QuantumFieldTheory,
            vec![v("α","fine structure constant","",""), v("e","elementary charge","C","")],
            vec![], &["electromagnetism coupling strength"], 0.95),

        PhysicsEquation::new("L_SM = kinetic + Yukawa + Higgs + gauge (Standard Model Lagrangian)", r"\mathcal{L}_{SM}=\mathcal{L}_{kinetic}+\mathcal{L}_{Yukawa}+\mathcal{L}_{Higgs}+\mathcal{L}_{gauge}",
            Discipline::QuantumFieldTheory,
            vec![],
            vec![3], &["SU(3)×SU(2)×U(1) gauge theory","all known fundamental particles"], 0.88),

        PhysicsEquation::new("V(φ) = μ²φ²/2 + λφ⁴/4 (Mexican hat / Higgs potential)", r"V(\phi)=\frac{\mu^2}{2}|\phi|^2+\frac{\lambda}{4}|\phi|^4",
            Discipline::QuantumFieldTheory,
            vec![v("φ","Higgs field","",""), v("μ","mass parameter","",""), v("λ","self-coupling","","")],
            vec![], &["spontaneous symmetry breaking","μ²<0 for SSB"], 0.90),

        PhysicsEquation::new("m_W = g*v/2 (W boson mass from Higgs mechanism)", r"m_W=\frac{gv}{2}",
            Discipline::QuantumFieldTheory,
            vec![v("m_W","W boson mass","GeV",""), v("g","SU(2) coupling","",""), v("v","Higgs vacuum expectation value","GeV","")],
            vec![], &["electroweak symmetry breaking"], 0.88),

        PhysicsEquation::new("β(g) = μ * dg/dμ (renormalization group beta function)", r"\beta(g)=\mu\frac{dg}{d\mu}",
            Discipline::QuantumFieldTheory,
            vec![v("g","coupling constant","",""), v("μ","renormalization scale","GeV","")],
            vec![], &["renormalization group","running coupling"], 0.88),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // SPECIAL RELATIVITY — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("ds² = -c²dt² + dx² + dy² + dz² (Minkowski metric)", r"ds^2=-c^2dt^2+dx^2+dy^2+dz^2",
            Discipline::SpecialRelativity,
            vec![v("ds","spacetime interval","m",""), v("c","speed of light","m/s","")],
            vec![], &["flat spacetime","special relativity"], 0.95),

        PhysicsEquation::new("x' = γ(x - vt); t' = γ(t - vx/c²) (full Lorentz transform)", r"x'=\gamma(x-vt),\quad t'=\gamma(t-vx/c^2)",
            Discipline::SpecialRelativity,
            vec![v("x","position","m",""), v("t","time","s",""), v("v","relative velocity","m/s","")],
            vec![], &["inertial frames","v along x axis"], 0.95),

        PhysicsEquation::new("Δt' = γ * Δt (time dilation)", r"\Delta t'=\gamma\Delta t",
            Discipline::SpecialRelativity,
            vec![v("Δt","proper time","s",""), v("Δt'","dilated time","s","")],
            vec![], &["time dilation"], 0.95),

        PhysicsEquation::new("L' = L/γ (length contraction)", r"L'=L/\gamma",
            Discipline::SpecialRelativity,
            vec![v("L","proper length","m",""), v("L'","contracted length","m","")],
            vec![], &["length contraction","along direction of motion"], 0.95),

        PhysicsEquation::new("p^μ = (E/c, p_x, p_y, p_z) (4-momentum)", r"p^\mu=(E/c,\,p_x,\,p_y,\,p_z)",
            Discipline::SpecialRelativity,
            vec![v("p^μ","4-momentum","",""), v("E","energy","J","")],
            vec![], &["special relativity"], 0.93),

        PhysicsEquation::new("E = γ*m*c² (relativistic energy)", r"E=\gamma mc^2",
            Discipline::SpecialRelativity,
            vec![v("E","total energy","J",""), v("m","rest mass","kg",""), v("γ","Lorentz factor","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("v_rel = (v1 + v2)/(1 + v1*v2/c²) (relativistic velocity addition)", r"v_{rel}=\frac{v_1+v_2}{1+v_1v_2/c^2}",
            Discipline::SpecialRelativity,
            vec![v("v_rel","relative velocity","m/s",""), v("v1","velocity 1","m/s",""), v("v2","velocity 2","m/s","")],
            vec![], &["collinear velocities"], 0.93),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // GENERAL RELATIVITY — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("ds² = g_μν dx^μ dx^ν (curved spacetime metric)", r"ds^2=g_{\mu\nu}dx^\mu dx^\nu",
            Discipline::GeneralRelativity,
            vec![v("g_μν","metric tensor","",""), v("ds","spacetime interval","m","")],
            vec![3], &["curved spacetime","GR"], 0.95),

        PhysicsEquation::new("Γ^λ_μν = (1/2)g^λρ(∂_μg_νρ + ∂_νg_μρ - ∂_ρg_μν) (Christoffel symbols)", r"\Gamma^\lambda_{\mu\nu}=\frac{1}{2}g^{\lambda\rho}(\partial_\mu g_{\nu\rho}+\partial_\nu g_{\mu\rho}-\partial_\rho g_{\mu\nu})",
            Discipline::GeneralRelativity,
            vec![v("Γ","Christoffel symbols","",""), v("g_μν","metric tensor","","")],
            vec![3], &[], 0.90),

        PhysicsEquation::new("d²x^μ/dτ² + Γ^μ_αβ dx^α/dτ dx^β/dτ = 0 (geodesic equation)", r"\frac{d^2x^\mu}{d\tau^2}+\Gamma^\mu_{\alpha\beta}\frac{dx^\alpha}{d\tau}\frac{dx^\beta}{d\tau}=0",
            Discipline::GeneralRelativity,
            vec![v("x^μ","spacetime coordinate","",""), v("τ","proper time","s","")],
            vec![3], &["free fall","no external forces"], 0.92),

        PhysicsEquation::new("R^ρ_σμν = ∂_μΓ^ρ_νσ - ∂_νΓ^ρ_μσ + ... (Riemann curvature tensor)", r"R^\rho{}_{\sigma\mu\nu}=\partial_\mu\Gamma^\rho_{\nu\sigma}-\partial_\nu\Gamma^\rho_{\mu\sigma}+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}-\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}",
            Discipline::GeneralRelativity,
            vec![v("R^ρ_σμν","Riemann tensor","","")],
            vec![3], &[], 0.88),

        PhysicsEquation::new("ds² = -(1-2GM/rc²)c²dt² + dr²/(1-2GM/rc²) + r²dΩ² (Schwarzschild)", r"ds^2=-\!\left(1-\frac{2GM}{rc^2}\right)c^2dt^2+\frac{dr^2}{1-2GM/rc^2}+r^2d\Omega^2",
            Discipline::GeneralRelativity,
            vec![v("M","mass","kg",""), v("r","radial coordinate","m",""), v("G","gravitational constant","m³/(kg·s²)","")],
            vec![3], &["spherically symmetric","non-rotating mass"], 0.93),

        PhysicsEquation::new("r_s = 2GM/c² (Schwarzschild radius)", r"r_s=\frac{2GM}{c^2}",
            Discipline::GeneralRelativity,
            vec![v("r_s","Schwarzschild radius","m",""), v("M","mass","kg","")],
            vec![], &["black hole event horizon"], 0.95),

        PhysicsEquation::new("T_Hawking = ħ*c³/(8πGMk_B) (Hawking temperature)", r"T_H=\frac{\hbar c^3}{8\pi GMk_B}",
            Discipline::GeneralRelativity,
            vec![v("T_Hawking","Hawking temperature","K",""), v("M","black hole mass","kg","")],
            vec![], &["black hole thermodynamics","quantum gravity"], 0.88),

        PhysicsEquation::new("S_BH = A*c³/(4Għ) (Bekenstein-Hawking entropy)", r"S_{BH}=\frac{Ac^3}{4G\hbar}",
            Discipline::GeneralRelativity,
            vec![v("S_BH","black hole entropy","J/K",""), v("A","event horizon area","m²","")],
            vec![], &["black hole thermodynamics","holographic principle"], 0.88),

        PhysicsEquation::new("h_+ or h_× ~ (G/c²) * (d²Q/dt²) / r (gravitational waves)", r"h_{+,\times}\sim\frac{G}{c^4}\frac{\ddot{Q}}{r}",
            Discipline::GeneralRelativity,
            vec![v("h","gravitational wave strain","",""), v("Q","mass quadrupole moment","kg·m²",""), v("r","distance","m","")],
            vec![3], &["far field","quadrupole approximation"], 0.88),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // NUCLEAR AND PARTICLE PHYSICS
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("E_binding = (Z*m_p + N*m_n - M_nucleus)*c² (nuclear binding energy)", r"E_B=(Zm_p+Nm_n-M_{nuc})c^2",
            Discipline::CondensedMatter,
            vec![v("E_binding","binding energy","MeV",""), v("Z","proton number","",""), v("N","neutron number","","")],
            vec![], &["nuclear physics"], 0.92),

        PhysicsEquation::new("N(t) = N_0 * exp(-λ_decay * t) (radioactive decay)", r"N(t)=N_0 e^{-\lambda t}",
            Discipline::CondensedMatter,
            vec![v("N","number of nuclei","",""), v("λ_decay","decay constant","1/s",""), v("t","time","s","")],
            vec![], &["radioactive decay"], 0.95),

        PhysicsEquation::new("t_half = ln(2)/λ_decay (half-life)", r"t_{1/2}=\frac{\ln 2}{\lambda}",
            Discipline::CondensedMatter,
            vec![v("t_half","half-life","s",""), v("λ_decay","decay constant","1/s","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("Q = (m_initial - m_final)*c² (Q-value nuclear reaction)", r"Q=(m_i-m_f)c^2",
            Discipline::CondensedMatter,
            vec![v("Q","Q-value","MeV",""), v("m_initial","initial mass","kg",""), v("m_final","final mass","kg","")],
            vec![], &["nuclear reaction energetics"], 0.92),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // CONDENSED MATTER — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("ε(k) = ħ²k²/(2m) (free electron dispersion)", r"\varepsilon(\mathbf{k})=\frac{\hbar^2k^2}{2m}",
            Discipline::CondensedMatter,
            vec![v("ε","energy","J",""), v("k","wave vector","1/m",""), v("m","effective mass","kg","")],
            vec![], &["free electron model","no periodic potential"], 0.92),

        PhysicsEquation::new("ε(k) = ε_0 - 2t*(cos(k_x*a) + cos(k_y*a)) (tight-binding)", r"\varepsilon(\mathbf k)=\varepsilon_0-2t\sum_i\cos(k_i a)",
            Discipline::CondensedMatter,
            vec![v("t","hopping integral","J",""), v("a","lattice constant","m",""), v("ε_0","on-site energy","J","")],
            vec![2,3], &["tight-binding model","periodic lattice"], 0.90),

        PhysicsEquation::new("j = σ * E (Ohm's law microscopic form)", r"\mathbf{j}=\sigma\mathbf{E}",
            Discipline::CondensedMatter,
            vec![v("j","current density","A/m²",""), v("σ","conductivity","S/m",""), v("E","electric field","V/m","")],
            vec![], &[], 0.93),

        PhysicsEquation::new("σ = n*e²*τ/m (Drude conductivity)", r"\sigma=\frac{ne^2\tau}{m}",
            Discipline::CondensedMatter,
            vec![v("n","carrier density","1/m³",""), v("τ","scattering time","s",""), v("m","effective mass","kg","")],
            vec![], &["Drude model","free electron gas"], 0.90),

        PhysicsEquation::new("ψ_BCS = Π_k (u_k + v_k * c†_k↑ c†_{-k↓})|0> (BCS ground state)", r"|\psi_{BCS}\rangle=\prod_{\mathbf k}(u_k+v_k c^\dagger_{\mathbf k\uparrow}c^\dagger_{-\mathbf k\downarrow})|0\rangle",
            Discipline::CondensedMatter,
            vec![v("u_k","coherence factor (empty)","",""), v("v_k","coherence factor (filled)","","")],
            vec![], &["BCS superconductivity","Cooper pairs"], 0.88),

        PhysicsEquation::new("Ω_Hall = e*B/(m*c) (cyclotron frequency)", r"\omega_c=\frac{eB}{mc}",
            Discipline::CondensedMatter,
            vec![v("Ω_Hall","cyclotron frequency","rad/s",""), v("B","magnetic field","T",""), v("m","effective mass","kg","")],
            vec![2,3], &["free electron in magnetic field"], 0.92),

        PhysicsEquation::new("χ = -∂²F/∂B² (magnetic susceptibility from free energy)", r"\chi=-\frac{\partial^2 F}{\partial B^2}",
            Discipline::CondensedMatter,
            vec![v("χ","magnetic susceptibility","",""), v("F","free energy","J",""), v("B","magnetic field","T","")],
            vec![], &[], 0.88),

        PhysicsEquation::new("k_F = (3π²n)^(1/3) (Fermi wave vector 3D)", r"k_F=(3\pi^2 n)^{1/3}",
            Discipline::CondensedMatter,
            vec![v("k_F","Fermi wave vector","1/m",""), v("n","electron density","1/m³","")],
            vec![3], &["free electron model","spin-1/2"], 0.92),

        PhysicsEquation::new("E_F = ħ²k_F²/(2m) (Fermi energy)", r"E_F=\frac{\hbar^2k_F^2}{2m}",
            Discipline::CondensedMatter,
            vec![v("E_F","Fermi energy","J",""), v("k_F","Fermi wave vector","1/m","")],
            vec![], &["free electron gas","T=0"], 0.93),

        PhysicsEquation::new("Φ_0 = h/(2e) (superconducting flux quantum)", r"\Phi_0=\frac{h}{2e}=2.07\times10^{-15}\,\text{Wb}",
            Discipline::CondensedMatter,
            vec![v("Φ_0","flux quantum","Wb",""), v("e","elementary charge","C","")],
            vec![], &["superconductor","flux quantization"], 0.92),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // FLUID DYNAMICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("P + (1/2)*ρ*v² + ρ*g*h = const (Bernoulli)", r"P+\tfrac{1}{2}\rho v^2+\rho gh=\text{const}",
            Discipline::FluidDynamics,
            vec![v("P","pressure","Pa",""), v("ρ","density","kg/m³",""), v("v","flow speed","m/s",""), v("h","height","m","")],
            vec![2,3], &["steady flow","incompressible","inviscid","streamline"], 0.93),

        PhysicsEquation::new("∂ρ/∂t + ∇·(ρv) = 0 (continuity equation)", r"\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{v})=0",
            Discipline::FluidDynamics,
            vec![v("ρ","density","kg/m³",""), v("v","velocity field","m/s","")],
            vec![2,3], &["conservation of mass"], 0.95),

        PhysicsEquation::new("Ma = v/c_sound (Mach number)", r"Ma=v/c_s",
            Discipline::FluidDynamics,
            vec![v("Ma","Mach number","",""), v("c_sound","speed of sound","m/s","")],
            vec![], &[], 0.92),

        PhysicsEquation::new("ω_vorticity = ∇ × v (vorticity)", r"\boldsymbol\omega=\nabla\times\mathbf{v}",
            Discipline::FluidDynamics,
            vec![v("ω_vorticity","vorticity","1/s","")],
            vec![3], &[], 0.90),

        PhysicsEquation::new("dω/dt = (ω·∇)v + ν∇²ω (vorticity transport)", r"\frac{D\boldsymbol\omega}{Dt}=(\boldsymbol\omega\cdot\nabla)\mathbf{v}+\nu\nabla^2\boldsymbol\omega",
            Discipline::FluidDynamics,
            vec![v("ν","kinematic viscosity","m²/s","")],
            vec![3], &["incompressible Newtonian fluid"], 0.88),

        PhysicsEquation::new("Fr = v/sqrt(g*L) (Froude number)", r"Fr=v/\sqrt{gL}",
            Discipline::FluidDynamics,
            vec![v("Fr","Froude number","",""), v("L","length scale","m","")],
            vec![], &["free surface flows"], 0.88),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // CHAOS AND NONLINEAR DYNAMICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("Λ = lim sup (1/n) Σ ln|f'(x_i)| (Lyapunov exponent discrete)", r"\Lambda=\limsup_{n\to\infty}\frac{1}{n}\sum_{i=0}^{n-1}\ln|f'(x_i)|",
            Discipline::ChaosDynamics,
            vec![v("f","iteration map","",""), v("Λ","Lyapunov exponent","1","")],
            vec![], &["discrete dynamical system"], 0.90),

        PhysicsEquation::new("dx/dt = F(x) + σξ(t) (stochastic differential equation)", r"\dot{x}=F(x)+\sigma\xi(t)",
            Discipline::ChaosDynamics,
            vec![v("F","deterministic drift","",""), v("σ","noise amplitude","",""), v("ξ","Gaussian white noise","","")],
            vec![], &["Langevin equation","Markov process"], 0.90),

        PhysicsEquation::new("∂P/∂t = -∂(F*P)/∂x + D*∂²P/∂x² (Fokker-Planck equation)", r"\frac{\partial P}{\partial t}=-\frac{\partial(FP)}{\partial x}+D\frac{\partial^2 P}{\partial x^2}",
            Discipline::ChaosDynamics,
            vec![v("P","probability density","",""), v("D","diffusion coefficient","m²/s",""), v("F","drift","","")],
            vec![], &["stochastic process","probability flow"], 0.90),

        PhysicsEquation::new("D_f = log(N)/log(1/r) (fractal dimension box-counting)", r"D_f=\frac{\log N}{\log(1/r)}",
            Discipline::ChaosDynamics,
            vec![v("D_f","fractal dimension","",""), v("N","number of boxes","",""), v("r","box size","","")],
            vec![], &["self-similar fractal","box-counting method"], 0.88),

        PhysicsEquation::new("ṙ = r(1-r²) - I_syn*sin(θ) (Stuart-Landau / coupled oscillator)", r"\dot r=r(1-r^2),\quad\dot\theta=\omega-\frac{K}{N}\sum_j\sin(\theta_j-\theta)",
            Discipline::ChaosDynamics,
            vec![v("r","amplitude","",""), v("K","coupling strength","",""), v("θ","phase","rad","")],
            vec![], &["limit cycle","Kuramoto model of synchronization"], 0.88),

        PhysicsEquation::new("r_order = |(1/N)*Σ exp(iθ_j)| (Kuramoto order parameter)", r"r=\left|\frac{1}{N}\sum_{j=1}^N e^{i\theta_j}\right|",
            Discipline::ChaosDynamics,
            vec![v("r_order","order parameter","",""), v("θ_j","phase of oscillator j","rad","")],
            vec![], &["Kuramoto synchronization transition"], 0.88),

        PhysicsEquation::new("H(q,p,t) = H_0 + εH_1 (Hamiltonian perturbation / KAM)", r"H=H_0(\mathbf p)+\varepsilon H_1(\mathbf q,\mathbf p,t)",
            Discipline::ChaosDynamics,
            vec![v("H_0","integrable Hamiltonian","J",""), v("H_1","perturbation","J",""), v("ε","perturbation parameter","","")],
            vec![], &["KAM theorem","near-integrable systems"], 0.85),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // OPTICS AND PHOTONICS
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("n1*sin(θ1) = n2*sin(θ2) (Snell's law)", r"n_1\sin\theta_1=n_2\sin\theta_2",
            Discipline::Electromagnetism,
            vec![v("n1","refractive index 1","",""), v("n2","refractive index 2","",""), v("θ1","angle of incidence","rad",""), v("θ2","angle of refraction","rad","")],
            vec![2,3], &["geometric optics","interface between media"], 0.95),

        PhysicsEquation::new("1/f = 1/d_o + 1/d_i (thin lens equation)", r"\frac{1}{f}=\frac{1}{d_o}+\frac{1}{d_i}",
            Discipline::Electromagnetism,
            vec![v("f","focal length","m",""), v("d_o","object distance","m",""), v("d_i","image distance","m","")],
            vec![], &["thin lens","paraxial approximation"], 0.93),

        PhysicsEquation::new("I = I_0 * cos²(θ) (Malus's law, polarization)", r"I=I_0\cos^2\theta",
            Discipline::Electromagnetism,
            vec![v("I","transmitted intensity","W/m²",""), v("I_0","incident intensity","W/m²",""), v("θ","polarizer angle","rad","")],
            vec![], &["linear polarization"], 0.92),

        PhysicsEquation::new("d*sin(θ) = m*λ (diffraction grating condition)", r"d\sin\theta=m\lambda",
            Discipline::Electromagnetism,
            vec![v("d","grating spacing","m",""), v("m","diffraction order","",""), v("λ","wavelength","m","")],
            vec![], &["far-field diffraction","grating"], 0.93),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // TOPOLOGICAL PHYSICS — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("C_1 = (1/2π) ∫ F dk_x dk_y (Chern number)", r"C_1=\frac{1}{2\pi}\int_{BZ}F_{xy}\,dk_x\,dk_y",
            Discipline::TopologicalPhysics,
            vec![v("C_1","first Chern number (integer)","",""), v("F_xy","Berry curvature","","")],
            vec![2], &["2D Brillouin zone","topological band invariant"], 0.88),

        PhysicsEquation::new("F_xy = ∂_x A_y - ∂_y A_x (Berry curvature)", r"F_{xy}=\partial_{k_x}A_{k_y}-\partial_{k_y}A_{k_x}",
            Discipline::TopologicalPhysics,
            vec![v("F_xy","Berry curvature","",""), v("A_k","Berry connection","","")],
            vec![2], &["momentum space","adiabatic evolution"], 0.88),

        PhysicsEquation::new("Z_2 = (1/π) ∮ A dk - (1/2π) ∫ F dk² (Z2 topological invariant)", r"Z_2=\frac{1}{\pi}\left(\oint_{\partial BZ}\mathbf A\cdot d\mathbf k-\int_{BZ}F\,d^2k\right)\mod 2",
            Discipline::TopologicalPhysics,
            vec![],
            vec![2], &["time-reversal symmetric insulator","topological insulator"], 0.85),

        PhysicsEquation::new("H_Kitaev = -μΣc†c - t*Σ(c†c+h.c.) - Δ*Σ(cc+h.c.) (Kitaev chain)", r"H=-\mu\sum_i c_i^\dagger c_i-\sum_i(tc_{i+1}^\dagger c_i+\Delta c_{i+1}c_i+\text{h.c.})",
            Discipline::TopologicalPhysics,
            vec![v("μ","chemical potential","J",""), v("t","hopping","J",""), v("Δ","p-wave pairing","J","")],
            vec![2], &["1D topological superconductor","Majorana end modes"], 0.85),

        PhysicsEquation::new("e* = e/3 (fractional charge quasiparticle in FQHE)", r"e^*=e/m\quad(m=1,3,5,\ldots)",
            Discipline::TopologicalPhysics,
            vec![v("e*","fractional quasiparticle charge","C",""), v("m","odd integer","","")],
            vec![2], &["fractional quantum Hall effect","Laughlin state"], 0.88),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // COSMOLOGY — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("z = (λ_obs - λ_emit)/λ_emit (cosmological redshift)", r"z=\frac{\lambda_{obs}-\lambda_{emit}}{\lambda_{emit}}",
            Discipline::Cosmology,
            vec![v("z","redshift","",""), v("λ_obs","observed wavelength","m",""), v("λ_emit","emitted wavelength","m","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("d_L = (1+z) * ∫ c dz / H(z) (luminosity distance)", r"d_L=(1+z)\int_0^z\frac{c\,dz'}{H(z')}",
            Discipline::Cosmology,
            vec![v("d_L","luminosity distance","Mpc",""), v("H","Hubble parameter","km/s/Mpc","")],
            vec![], &["FLRW cosmology"], 0.88),

        PhysicsEquation::new("ρ_crit = 3H²/(8πG) (critical density)", r"\rho_c=\frac{3H^2}{8\pi G}",
            Discipline::Cosmology,
            vec![v("ρ_crit","critical density","kg/m³",""), v("H","Hubble parameter","1/s","")],
            vec![], &["flat universe condition"], 0.92),

        PhysicsEquation::new("Ω = ρ/ρ_crit (density parameter)", r"\Omega=\rho/\rho_c",
            Discipline::Cosmology,
            vec![v("Ω","density parameter","",""), v("ρ","actual density","kg/m³","")],
            vec![], &["Ω=1 flat, Ω>1 closed, Ω<1 open"], 0.92),

        PhysicsEquation::new("H(z) = H_0*sqrt(Ω_m*(1+z)³ + Ω_Λ) (expansion history)", r"H(z)=H_0\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}",
            Discipline::Cosmology,
            vec![v("H_0","Hubble constant today","km/s/Mpc",""), v("Ω_m","matter density parameter","",""), v("Ω_Λ","dark energy density parameter","","")],
            vec![], &["ΛCDM model","flat universe"], 0.90),

        PhysicsEquation::new("T_CMB ∝ (1+z) (CMB temperature scaling)", r"T_{CMB}=T_0(1+z)",
            Discipline::Cosmology,
            vec![v("T_CMB","CMB temperature","K",""), v("T_0","present CMB temperature 2.725K","K","")],
            vec![], &["photon number conservation","adiabatic expansion"], 0.92),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // INFORMATION THEORY — extended
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("C = B * log2(1 + S/N) (Shannon channel capacity)", r"C=B\log_2(1+S/N)",
            Discipline::InformationTheory,
            vec![v("C","channel capacity","bits/s",""), v("B","bandwidth","Hz",""), v("S","signal power","W",""), v("N","noise power","W","")],
            vec![], &["AWGN channel","Shannon-Hartley theorem"], 0.95),

        PhysicsEquation::new("D_KL(P||Q) = Σ P*log(P/Q) (KL divergence)", r"D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}",
            Discipline::InformationTheory,
            vec![v("D_KL","KL divergence","nats",""), v("P","true distribution","",""), v("Q","approximate distribution","","")],
            vec![], &["non-symmetric"], 0.95),

        PhysicsEquation::new("K(x) ≈ min|p|: U(p)=x (Kolmogorov complexity)", r"K(x)\approx\min_{p:U(p)=x}|p|",
            Discipline::InformationTheory,
            vec![v("K","Kolmogorov complexity","bits",""), v("p","program","",""), v("U","universal Turing machine","","")],
            vec![], &["incomputable in general","fundamental measure of algorithmic information"], 0.85),

        PhysicsEquation::new("H(X|Y) = H(X,Y) - H(Y) (conditional entropy)", r"H(X|Y)=H(X,Y)-H(Y)",
            Discipline::InformationTheory,
            vec![v("H","entropy","bits","")],
            vec![], &[], 0.93),

        PhysicsEquation::new("F_Fisher = E[(∂logP/∂θ)²] (Fisher information)", r"\mathcal{F}=\mathbb{E}\!\left[\left(\frac{\partial\ln P}{\partial\theta}\right)^2\right]",
            Discipline::InformationTheory,
            vec![v("F_Fisher","Fisher information","",""), v("θ","parameter","",""), v("P","probability distribution","","")],
            vec![], &["Cramér-Rao bound","statistical inference"], 0.90),

        PhysicsEquation::new("Δθ ≥ 1/sqrt(F_Fisher) (Cramér-Rao bound)", r"\text{Var}(\hat\theta)\geq\frac{1}{\mathcal F}",
            Discipline::InformationTheory,
            vec![v("θ","estimated parameter","",""), v("F_Fisher","Fisher information","","")],
            vec![], &["unbiased estimator"], 0.90),

        PhysicsEquation::new("H_max = log2(N) (maximum entropy uniform distribution)", r"H_{max}=\log_2 N",
            Discipline::InformationTheory,
            vec![v("N","number of equally likely outcomes","","")],
            vec![], &["uniform distribution","maximum entropy principle"], 0.92),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // MATHEMATICAL PHYSICS — PDEs AND CORE EQUATIONS
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("∂u/∂t = α * ∇²u (heat / diffusion equation)", r"\frac{\partial u}{\partial t}=\alpha\nabla^2 u",
            Discipline::ClassicalMechanics,
            vec![v("u","scalar field (temperature, concentration)","",""), v("α","diffusivity","m²/s","")],
            vec![2,3], &["linear diffusion","isotropic medium"], 0.95),

        PhysicsEquation::new("∇²φ = 0 (Laplace equation)", r"\nabla^2\varphi=0",
            Discipline::ClassicalMechanics,
            vec![v("φ","scalar potential","","")],
            vec![2,3], &["no sources","equilibrium field"], 0.95),

        PhysicsEquation::new("∇²φ = f (Poisson equation — general)", r"\nabla^2\varphi=f",
            Discipline::ClassicalMechanics,
            vec![v("f","source term","","")],
            vec![2,3], &[], 0.93),

        PhysicsEquation::new("u_t + u*u_x = ν*u_xx (Burgers equation)", r"u_t+uu_x=\nu u_{xx}",
            Discipline::ChaosDynamics,
            vec![v("u","velocity field","m/s",""), v("ν","viscosity","m²/s","")],
            vec![], &["nonlinear advection-diffusion","proto-turbulence"], 0.88),

        PhysicsEquation::new("u_t = u_xx - u³ + u (Ginzburg-Landau equation)", r"\partial_t u=\partial_{xx}u+u-u^3",
            Discipline::CondensedMatter,
            vec![v("u","order parameter","","")],
            vec![], &["phase transitions","pattern formation"], 0.88),

        PhysicsEquation::new("u_t + 6*u*u_x + u_xxx = 0 (KdV equation)", r"u_t+6uu_x+u_{xxx}=0",
            Discipline::ChaosDynamics,
            vec![v("u","wave amplitude","","")],
            vec![], &["soliton solutions","shallow water waves"], 0.88),

        PhysicsEquation::new("iψ_t = -ψ_xx/2 + |ψ|²*ψ (nonlinear Schrödinger / NLS)", r"i\psi_t=-\tfrac{1}{2}\psi_{xx}+|\psi|^2\psi",
            Discipline::QuantumMechanics,
            vec![v("ψ","complex wave amplitude","","")],
            vec![], &["self-focusing nonlinear medium","optical solitons","BEC"], 0.88),

        PhysicsEquation::new("∂_t ρ + ∇·j = 0 (continuity — conservation law general)", r"\partial_t\rho+\nabla\cdot\mathbf{j}=0",
            Discipline::ClassicalMechanics,
            vec![v("ρ","conserved density","",""), v("j","flux","","")],
            vec![2,3], &["any conserved quantity"], 0.95),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // BIOPHYSICS AND COMPLEX SYSTEMS
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("dN/dt = r*N*(1 - N/K) (logistic population growth)", r"\frac{dN}{dt}=rN\!\left(1-\frac{N}{K}\right)",
            Discipline::ChaosDynamics,
            vec![v("N","population","",""), v("r","growth rate","1/s",""), v("K","carrying capacity","","")],
            vec![], &["bounded population growth"], 0.90),

        PhysicsEquation::new("dx/dt = a*x - b*x*y; dy/dt = c*x*y - d*y (Lotka-Volterra)", r"\dot x=ax-bxy,\quad\dot y=cxy-dy",
            Discipline::ChaosDynamics,
            vec![v("x","prey population","",""), v("y","predator population","",""), v("a","prey growth","1/s",""), v("d","predator death","1/s","")],
            vec![], &["predator-prey dynamics"], 0.90),

        PhysicsEquation::new("C*dV/dt = -I_ion + I_ext (Hodgkin-Huxley neuron)", r"C_m\frac{dV}{dt}=-\sum I_{ion}+I_{ext}",
            Discipline::ChaosDynamics,
            vec![v("V","membrane voltage","V",""), v("C","membrane capacitance","F",""), v("I_ion","ionic currents","A",""), v("I_ext","external current","A","")],
            vec![], &["neuron membrane dynamics","action potential"], 0.88),

        PhysicsEquation::new("ξ_x(τ) = <x(t+τ)*x(t)> (autocorrelation function)", r"\xi_x(\tau)=\langle x(t+\tau)x(t)\rangle",
            Discipline::ChaosDynamics,
            vec![v("τ","time lag","s",""), v("x","stochastic process","","")],
            vec![], &["stationary process"], 0.90),

        PhysicsEquation::new("P(x,t) = (4πDt)^(-1/2) exp(-x²/4Dt) (diffusion Green's function)", r"P(x,t)=\frac{1}{\sqrt{4\pi Dt}}e^{-x^2/4Dt}",
            Discipline::ChaosDynamics,
            vec![v("P","probability density","1/m",""), v("D","diffusion coefficient","m²/s",""), v("x","displacement","m","")],
            vec![], &["free diffusion","1D","point source initial condition"], 0.92),

        PhysicsEquation::new("<x²> = 2*D*t (Einstein relation for diffusion)", r"\langle x^2\rangle=2Dt",
            Discipline::ChaosDynamics,
            vec![v("x","displacement","m",""), v("D","diffusion coefficient","m²/s",""), v("t","time","s","")],
            vec![], &["Brownian motion","random walk"], 0.93),

        PhysicsEquation::new("D = k_B*T/(6πηr) (Stokes-Einstein diffusion coefficient)", r"D=\frac{k_BT}{6\pi\eta r}",
            Discipline::ChaosDynamics,
            vec![v("η","dynamic viscosity","Pa·s",""), v("r","particle radius","m",""), v("T","temperature","K","")],
            vec![], &["Brownian particle in fluid"], 0.90),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // PLASMA PHYSICS
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("ω_p = sqrt(n*e²/(ε_0*m)) (plasma frequency)", r"\omega_p=\sqrt{\frac{ne^2}{\varepsilon_0 m}}",
            Discipline::Electromagnetism,
            vec![v("ω_p","plasma frequency","rad/s",""), v("n","electron density","1/m³",""), v("m","electron mass","kg","")],
            vec![], &["unmagnetized plasma"], 0.90),

        PhysicsEquation::new("λ_D = sqrt(ε_0*k_B*T/(n*e²)) (Debye length)", r"\lambda_D=\sqrt{\frac{\varepsilon_0 k_BT}{ne^2}}",
            Discipline::Electromagnetism,
            vec![v("λ_D","Debye screening length","m",""), v("n","carrier density","1/m³",""), v("T","temperature","K","")],
            vec![], &["plasma shielding length"], 0.90),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // MATHEMATICAL TOOLS USED IN THE NODE
    // (These appear in math_toolbox.rs and throughout the streaming runtime)
    // ══════════════════════════════════════════════════════════════════════
    eqs.extend(vec![
        PhysicsEquation::new("σ(x) = 1/(1 + exp(-x)) (logistic sigmoid)", r"\sigma(x)=\frac{1}{1+e^{-x}}",
            Discipline::InformationTheory,
            vec![v("x","input","","")],
            vec![], &["activation function","probability mapping"], 0.95),

        PhysicsEquation::new("softmax_i = exp(x_i)/Σ exp(x_j) (softmax normalization)", r"\text{softmax}_i=\frac{e^{x_i}}{\sum_j e^{x_j}}",
            Discipline::InformationTheory,
            vec![],
            vec![], &["categorical probability distribution","numerical: subtract max first"], 0.95),

        PhysicsEquation::new("P(H|E) = P(E|H)*P(H)/P(E) (Bayes' theorem)", r"P(H|E)=\frac{P(E|H)P(H)}{P(E)}",
            Discipline::InformationTheory,
            vec![v("H","hypothesis","",""), v("E","evidence","","")],
            vec![], &[], 0.95),

        PhysicsEquation::new("Δw = η * δ_pre * δ_post (Hebbian learning rule)", r"\Delta w=\eta\,\delta_{pre}\,\delta_{post}",
            Discipline::InformationTheory,
            vec![v("w","synaptic weight","",""), v("η","learning rate","",""), v("δ_pre","presynaptic activity","",""), v("δ_post","postsynaptic activity","","")],
            vec![], &["Hebb's rule","neurons that fire together wire together"], 0.93),

        PhysicsEquation::new("x̄_n = x̄_{n-1} + (x_n - x̄_{n-1})/n (Welford online mean)", r"\bar x_n=\bar x_{n-1}+\frac{x_n-\bar x_{n-1}}{n}",
            Discipline::InformationTheory,
            vec![v("x̄","running mean","",""), v("n","count","","")],
            vec![], &["numerically stable online algorithm"], 0.92),

        PhysicsEquation::new("EMA_t = α*x_t + (1-α)*EMA_{t-1} (exponential moving average)", r"EMA_t=\alpha x_t+(1-\alpha)EMA_{t-1}",
            Discipline::InformationTheory,
            vec![v("α","smoothing factor","",""), v("x_t","observation","","")],
            vec![], &["online low-pass filter"], 0.95),

        PhysicsEquation::new("cos_sim(a,b) = a·b/(||a||*||b||) (cosine similarity)", r"\text{sim}(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\|\mathbf b\|}",
            Discipline::InformationTheory,
            vec![],
            vec![], &["direction similarity","independent of magnitude"], 0.95),

        PhysicsEquation::new("r_Pearson = Σ(x-x̄)(y-ȳ)/(n*σ_x*σ_y) (Pearson correlation)", r"r=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{n\sigma_x\sigma_y}",
            Discipline::InformationTheory,
            vec![v("r_Pearson","Pearson correlation","","")],
            vec![], &["linear correlation","stationary distributions"], 0.93),

        PhysicsEquation::new("DTW(s,t) = min alignment cost between sequences (dynamic time warping)", r"DTW(s,t)=\min_{\pi}\sum_{(i,j)\in\pi}d(s_i,t_j)",
            Discipline::InformationTheory,
            vec![v("s","sequence 1","",""), v("t","sequence 2","","")],
            vec![], &["time-series alignment","elastic matching"], 0.90),
    ]);

    // ── Game Theory ──────────────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("Nash equilibrium: no player benefits by unilateral deviation (u_i(s*) ≥ u_i(s_i, s*_{-i}))", r"u_i(s^*)\geq u_i(s_i,s^*_{-i})\;\forall i,s_i",
            Discipline::Custom("GameTheory".into()),
            vec![v("u_i","utility of player i","",""), v("s*","Nash equilibrium strategy profile","","")],
            vec![], &["equilibrium","strategy","rationality","self-reinforcing"], 0.98),

        PhysicsEquation::new("Mixed strategy Nash equilibrium: E[u_i|σ_i] = E[u_i|σ_i'] for all strategies in support", r"\sum_{s}u_i(s)\sigma_i(s)=\sum_{s}u_i(s)\sigma_i'(s)",
            Discipline::Custom("GameTheory".into()),
            vec![v("σ_i","mixed strategy distribution","","")],
            vec![], &["mixed strategy","indifference","probability","randomization"], 0.95),

        PhysicsEquation::new("Minimax theorem: max_x min_y f(x,y) = min_y max_x f(x,y) (von Neumann)", r"\max_x\min_y f(x,y)=\min_y\max_x f(x,y)",
            Discipline::Custom("GameTheory".into()),
            vec![v("f","payoff function","","")],
            vec![], &["zero-sum","adversarial","minimax","optimal strategy"], 0.97),

        PhysicsEquation::new("Replicator dynamics: dx_i/dt = x_i * (f_i - φ) (evolutionary game theory)", r"\dot x_i=x_i(f_i-\bar\phi)",
            Discipline::Custom("GameTheory".into()),
            vec![v("x_i","population share of strategy i","",""), v("f_i","fitness of strategy i","",""), v("φ","mean population fitness","","")],
            vec![], &["evolution","population dynamics","strategy selection","natural selection","replicator"], 0.95),

        PhysicsEquation::new("Evolutionary stable strategy (ESS): u(ESS, ESS) > u(mutant, ESS)", r"u(s^*,s^*)>u(s',s^*)\;\forall s'\neq s^*",
            Discipline::Custom("GameTheory".into()),
            vec![v("u","payoff","",""), v("s*","ESS strategy","","")],
            vec![], &["evolutionary stable","invasion barrier","population","stable equilibrium"], 0.94),

        PhysicsEquation::new("Prisoner's dilemma: T > R > P > S and 2R > T+S", r"T>R>P>S,\quad 2R>T+S",
            Discipline::Custom("GameTheory".into()),
            vec![v("T","temptation payoff","",""), v("R","reward payoff","",""), v("P","punishment payoff","",""), v("S","sucker payoff","","")],
            vec![], &["cooperation","defection","coordination failure","social dilemma"], 0.97),

        PhysicsEquation::new("Folk theorem: any payoff vector above minmax is achievable by repeated game equilibrium", r"v_i\geq\underline{v}_i\Rightarrow v\text{ sustained by subgame perfect eq.}",
            Discipline::Custom("GameTheory".into()),
            vec![v("v_i","achievable payoff","",""), v("v̲_i","minmax value","","")],
            vec![], &["repeated game","cooperation","reputation","long-run equilibrium","sustained"], 0.92),

        PhysicsEquation::new("Shapley value: φ_i = Σ_{S} [|S|!(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]", r"\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(n-|S|-1)!}{n!}[v(S\cup\{i\})-v(S)]",
            Discipline::Custom("GameTheory".into()),
            vec![v("φ_i","fair value share of player i","",""), v("v","coalition value function","","")],
            vec![], &["coalition","fair division","cooperative game","contribution","attribution"], 0.93),

        PhysicsEquation::new("Price of anarchy ρ = worst_eq_cost / optimal_cost", r"\rho=\frac{\max_{\text{eq}}C(\text{eq})}{C(\text{opt})}",
            Discipline::Custom("GameTheory".into()),
            vec![v("ρ","price of anarchy","",""), v("C","social cost","","")],
            vec![], &["efficiency loss","decentralization","social optimum","coordination cost"], 0.91),

        PhysicsEquation::new("Hotelling's spatial competition: firms locate at median voter/consumer", r"x^*=\text{median}(F)",
            Discipline::Custom("GameTheory".into()),
            vec![v("x*","optimal position","",""), v("F","consumer distribution","","")],
            vec![], &["spatial competition","positioning","median voter","product differentiation"], 0.89),

        PhysicsEquation::new("Bayesian Nash equilibrium: s_i* maximizes E[u_i | type θ_i]", r"s_i^*(\theta_i)=\arg\max_{s_i}\mathbb{E}_{\theta_{-i}}[u_i(s_i,s_{-i}^*,\theta)]",
            Discipline::Custom("GameTheory".into()),
            vec![v("θ_i","private type/information of player i","","")],
            vec![], &["incomplete information","beliefs","private information","signaling"], 0.93),

        PhysicsEquation::new("Auction revenue equivalence: E[revenue] equal across auction formats", r"\mathbb{E}[R_{\text{first}}]=\mathbb{E}[R_{\text{second}}]=\mathbb{E}[R_{\text{Dutch}}]",
            Discipline::Custom("GameTheory".into()),
            vec![v("R","auction revenue","","")],
            vec![], &["auction","mechanism design","revenue","bidding strategy"], 0.90),

        PhysicsEquation::new("Schelling coordination: equilibrium at salient focal points (Schelling points)", r"s^*=\arg\max_{s}\Pr(\text{others choose }s)",
            Discipline::Custom("GameTheory".into()),
            vec![v("s*","focal point / Schelling point","","")],
            vec![], &["coordination","focal point","salience","convention","social norm"], 0.88),
    ]);

    // ── Marketing Science ─────────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("Bass diffusion model: dN/dt = (p + q*N/M) * (M - N)", r"\frac{dN}{dt}=\left(p+\frac{q}{M}N\right)(M-N)",
            Discipline::Custom("MarketingScience".into()),
            vec![v("N","cumulative adopters","",""), v("M","total market potential","",""), v("p","coefficient of innovation","",""), v("q","coefficient of imitation","","")],
            vec![], &["diffusion","adoption","innovation","word of mouth","market penetration","spread"], 0.97),

        PhysicsEquation::new("Viral coefficient k = invites_sent * conversion_rate (k>1 exponential growth)", r"k=i\cdot c,\quad N_t=N_0\cdot k^t",
            Discipline::Custom("MarketingScience".into()),
            vec![v("k","viral coefficient","",""), v("i","invites per user","",""), v("c","conversion rate","","")],
            vec![], &["viral","exponential growth","referral","word of mouth","network effect","spreading"], 0.96),

        PhysicsEquation::new("Customer lifetime value: CLV = M * (r/(1+d-r))", r"CLV=M\cdot\frac{r}{1+d-r}",
            Discipline::Custom("MarketingScience".into()),
            vec![v("M","margin per period","",""), v("r","retention rate","",""), v("d","discount rate","","")],
            vec![], &["customer value","retention","loyalty","lifetime value","churn"], 0.94),

        PhysicsEquation::new("Lanchester's square law: dB/dt = -α*R² (resource attrition)", r"\frac{dB}{dt}=-\alpha R^2",
            Discipline::Custom("MarketingScience".into()),
            vec![v("B","remaining force B","",""), v("R","force R strength","",""), v("α","attrition rate","","")],
            vec![], &["market share","competitive attrition","advertising war","brand competition","dominance"], 0.90),

        PhysicsEquation::new("Advertising adstock: A_t = spend_t + λ * A_{t-1} (carryover effect)", r"A_t=x_t+\lambda A_{t-1}",
            Discipline::Custom("MarketingScience".into()),
            vec![v("A_t","adstock at time t","",""), v("x_t","ad spend","",""), v("λ","decay / carryover rate","","")],
            vec![], &["advertising","carryover","memory","brand recall","media effect"], 0.93),

        PhysicsEquation::new("Price elasticity of demand: ε = (ΔQ/Q) / (ΔP/P)", r"\varepsilon=\frac{\partial Q/Q}{\partial P/P}",
            Discipline::Custom("MarketingScience".into()),
            vec![v("ε","price elasticity","",""), v("Q","quantity demanded","",""), v("P","price","","")],
            vec![], &["price sensitivity","demand","elasticity","pricing strategy"], 0.97),

        PhysicsEquation::new("Metcalfe's law: network value V ∝ n²", r"V\propto n^2",
            Discipline::Custom("MarketingScience".into()),
            vec![v("V","network value","",""), v("n","number of users","","")],
            vec![], &["network effect","platform","viral","social network","connections"], 0.95),

        PhysicsEquation::new("Reed's law: group-forming network value V ∝ 2^n", r"V\propto 2^n",
            Discipline::Custom("MarketingScience".into()),
            vec![v("V","group-network value","",""), v("n","number of users","","")],
            vec![], &["group formation","community","network effect","exponential value"], 0.88),

        PhysicsEquation::new("Independent cascade model: node v activates u with probability p_uv", r"P(\text{activate }u)=1-\prod_{v\in\text{active neighbors}}(1-p_{uv})",
            Discipline::Custom("MarketingScience".into()),
            vec![v("p_uv","edge influence probability","","")],
            vec![], &["influence propagation","cascade","social contagion","viral spread","seeding"], 0.94),

        PhysicsEquation::new("Linear threshold model: node activates when Σ w_uv ≥ threshold θ_v", r"\sum_{u\in N(v)}w_{uv}\geq\theta_v\Rightarrow v\text{ activates}",
            Discipline::Custom("MarketingScience".into()),
            vec![v("w_uv","influence weight","",""), v("θ_v","activation threshold","","")],
            vec![], &["threshold","social influence","tipping point","collective behavior","norm adoption"], 0.93),

        PhysicsEquation::new("Gompertz adoption curve: N(t) = M * exp(-exp(-k(t-t0)))", r"N(t)=M\exp(-\exp(-k(t-t_0)))",
            Discipline::Custom("MarketingScience".into()),
            vec![v("N","cumulative adopters","",""), v("M","market ceiling","",""), v("k","growth rate","",""), v("t0","inflection time","","")],
            vec![], &["adoption","S-curve","market saturation","diffusion","growth"], 0.91),

        PhysicsEquation::new("Marketing mix effectiveness: Sales = f(price, place, product, promotion)", r"S=\alpha\cdot P^{\beta_1}\cdot D^{\beta_2}\cdot A^{\beta_3}",
            Discipline::Custom("MarketingScience".into()),
            vec![v("S","sales","",""), v("α","baseline","",""), v("P","price","",""), v("D","distribution","",""), v("A","advertising","","")],
            vec![], &["marketing mix","4Ps","attribution","spend optimization"], 0.90),

        PhysicsEquation::new("Persuasion / attitude change: ΔA = E * R * I (elaboration likelihood)", r"\Delta A=E\cdot R\cdot I",
            Discipline::Custom("MarketingScience".into()),
            vec![v("E","elaboration likelihood","",""), v("R","route strength","",""), v("I","involvement","","")],
            vec![], &["persuasion","attitude","messaging","framing","central route","peripheral route"], 0.87),

        PhysicsEquation::new("Preferential attachment: P(link to node k) = k_degree / Σ degrees (Barabási-Albert)", r"P_k=\frac{k}{\sum_j k_j}",
            Discipline::Custom("MarketingScience".into()),
            vec![v("P_k","probability of new link","",""), v("k","node degree","","")],
            vec![], &["rich get richer","power law","network growth","influence","hub formation"], 0.95),

        PhysicsEquation::new("Zipf's law: frequency ∝ 1/rank (power law in language and markets)", r"f(r)\propto r^{-\alpha},\;\alpha\approx 1",
            Discipline::Custom("MarketingScience".into()),
            vec![v("f","frequency","",""), v("r","rank","","")],
            vec![], &["power law","word frequency","market share","brand ranking","Pareto"], 0.93),

        PhysicsEquation::new("Pareto principle: 80% of effects from 20% of causes", r"F(x)=1-\left(\frac{x_m}{x}\right)^\alpha",
            Discipline::Custom("MarketingScience".into()),
            vec![v("x_m","minimum value","",""), v("α","Pareto shape","","")],
            vec![], &["80/20","Pareto","power law","customer concentration","revenue concentration"], 0.94),
    ]);

    // ── Chaos Theory (extended) ───────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("Lorenz attractor: dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz", r"\dot x=\sigma(y-x),\;\dot y=x(\rho-z)-y,\;\dot z=xy-\beta z",
            Discipline::ChaosDynamics,
            vec![v("σ","Prandtl number","",""), v("ρ","Rayleigh number","",""), v("β","geometric factor","","")],
            vec![], &["Lorenz","strange attractor","chaos","butterfly effect","sensitive dependence"], 0.97),

        PhysicsEquation::new("Logistic map: x_{n+1} = r * x_n * (1 - x_n) (period doubling to chaos)", r"x_{n+1}=rx_n(1-x_n)",
            Discipline::ChaosDynamics,
            vec![v("r","growth parameter","",""), v("x_n","population fraction","","")],
            vec![], &["bifurcation","period doubling","chaos","logistic","discrete map"], 0.98),

        PhysicsEquation::new("Feigenbaum constant δ ≈ 4.669 (universal period-doubling ratio)", r"\delta=\lim_{n\to\infty}\frac{r_{n}-r_{n-1}}{r_{n+1}-r_n}\approx 4.669",
            Discipline::ChaosDynamics,
            vec![v("δ","Feigenbaum constant","",""), v("r_n","n-th bifurcation parameter","","")],
            vec![], &["universality","bifurcation","period doubling","Feigenbaum","renormalization"], 0.95),

        PhysicsEquation::new("Rössler attractor: dx/dt=-(y+z), dy/dt=x+ay, dz/dt=b+z(x-c)", r"\dot x=-(y+z),\;\dot y=x+ay,\;\dot z=b+z(x-c)",
            Discipline::ChaosDynamics,
            vec![v("a","parameter a","",""), v("b","parameter b","",""), v("c","parameter c","","")],
            vec![], &["Rossler","attractor","chaos","spiral","strange attractor"], 0.90),

        PhysicsEquation::new("Liouville theorem: dρ/dt = 0 (phase space volume conserved in Hamiltonian systems)", r"\frac{d\rho}{dt}=\frac{\partial\rho}{\partial t}+\{H,\rho\}=0",
            Discipline::ChaosDynamics,
            vec![v("ρ","phase space density","","")],
            vec![], &["phase space","incompressible","Hamiltonian","conservation","ergodic"], 0.95),

        PhysicsEquation::new("Correlation dimension D_2 = lim_{ε→0} log C(ε) / log ε", r"D_2=\lim_{\varepsilon\to 0}\frac{\log C(\varepsilon)}{\log\varepsilon}",
            Discipline::ChaosDynamics,
            vec![v("D_2","correlation dimension","",""), v("C(ε)","correlation integral","","")],
            vec![], &["attractor dimension","fractal","chaos","embedding","Grassberger-Procaccia"], 0.91),

        PhysicsEquation::new("Poincaré recurrence: almost all phase space trajectories return arbitrarily close to initial state", r"\mu(\{x:\exists T>0,\phi^T(x)\in U\})=\mu(U)",
            Discipline::ChaosDynamics,
            vec![v("μ","invariant measure","",""), v("φ^T","flow map","","")],
            vec![], &["recurrence","ergodic","return","quasi-periodic","long-term behavior"], 0.88),

        PhysicsEquation::new("Tent map: x_{n+1} = μ*min(x_n, 1-x_n) (exact chaos for μ=2)", r"x_{n+1}=\mu\min(x_n,1-x_n)",
            Discipline::ChaosDynamics,
            vec![v("μ","tent parameter","","")],
            vec![], &["tent map","exact solution","chaos","topological conjugacy","binary shift"], 0.88),

        PhysicsEquation::new("KAM theorem: most invariant tori survive small perturbations if ω is Diophantine", r"|k\cdot\omega|\geq\frac{\gamma}{|k|^\tau}\;\forall k\neq 0",
            Discipline::ChaosDynamics,
            vec![v("ω","frequency vector","",""), v("γ","KAM constant","",""), v("τ","Diophantine exponent","","")],
            vec![], &["KAM","invariant tori","integrability","perturbation","nearly integrable"], 0.90),

        PhysicsEquation::new("Entropy production rate σ = dS/dt > 0 (irreversible processes)", r"\sigma=\frac{dS}{dt}=\sum_k J_k X_k\geq 0",
            Discipline::ChaosDynamics,
            vec![v("σ","entropy production rate","",""), v("J_k","thermodynamic flux","",""), v("X_k","thermodynamic force","","")],
            vec![], &["irreversibility","entropy production","dissipation","nonequilibrium","arrow of time"], 0.93),

        PhysicsEquation::new("Sensitivity to initial conditions: |δx(t)| = |δx(0)| * exp(λ_max * t)", r"|\delta x(t)|=|\delta x(0)|\,e^{\lambda_{\max}t}",
            Discipline::ChaosDynamics,
            vec![v("λ_max","maximum Lyapunov exponent","","")],
            vec![], &["butterfly effect","sensitive dependence","chaos","prediction horizon","unpredictability"], 0.98),
    ]);

    // ── Quantum Mechanics and Quantum Information (extended) ──────────────────
    eqs.extend(vec![
        PhysicsEquation::new("Lindblad master equation: dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})", r"\frac{d\rho}{dt}=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger-\tfrac{1}{2}\{L_k^\dagger L_k,\rho\}\right)",
            Discipline::QuantumMechanics,
            vec![v("ρ","density matrix","",""), v("H","Hamiltonian","",""), v("L_k","Lindblad jump operators","","")],
            vec![], &["open quantum system","decoherence","dissipation","quantum noise","Markovian"], 0.95),

        PhysicsEquation::new("Quantum Zeno effect: P(no decay in t) = |<ψ|e^{-iHt/ħ}|ψ>|^{2N} → 1 as N→∞", r"P_{\text{survival}}=\left|\langle\psi|e^{-iHt/(N\hbar)}|\psi\rangle\right|^{2N}\to 1",
            Discipline::QuantumMechanics,
            vec![v("N","number of measurements","","")],
            vec![], &["quantum Zeno","measurement","freezing","frequent observation","watched pot"], 0.90),

        PhysicsEquation::new("Wigner function: W(x,p) = (1/πħ) ∫ ψ*(x+y)ψ(x-y) e^{2ipy/ħ} dy", r"W(x,p)=\frac{1}{\pi\hbar}\int\psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy",
            Discipline::QuantumMechanics,
            vec![v("W","Wigner quasi-probability distribution","","")],
            vec![], &["phase space","quasi-probability","quantum optics","negativity","non-classical"], 0.90),

        PhysicsEquation::new("Rabi oscillation: P_↑(t) = sin²(Ωt/2) (two-level system driven)", r"P_\uparrow(t)=\sin^2\!\left(\frac{\Omega t}{2}\right)",
            Discipline::QuantumMechanics,
            vec![v("Ω","Rabi frequency","","")],
            vec![], &["two-level system","driven qubit","oscillation","resonance","quantum control"], 0.94),

        PhysicsEquation::new("Bloch sphere: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩", r"|\psi\rangle=\cos\tfrac{\theta}{2}|0\rangle+e^{i\phi}\sin\tfrac{\theta}{2}|1\rangle",
            Discipline::QuantumMechanics,
            vec![v("θ","polar angle on Bloch sphere","",""), v("φ","azimuthal angle","","")],
            vec![], &["qubit","Bloch sphere","superposition","state representation","quantum bit"], 0.97),

        PhysicsEquation::new("Quantum mutual information: I(A:B) = S(A) + S(B) - S(AB)", r"I(A:B)=S(A)+S(B)-S(AB)",
            Discipline::QuantumMechanics,
            vec![v("S(A)","von Neumann entropy of A","","")],
            vec![], &["quantum correlation","entanglement","mutual information","subsystem"], 0.94),

        PhysicsEquation::new("Quantum discord: D(A|B) = I(A:B) - C(A:B) (beyond-entanglement correlations)", r"D(A|B)=I(A:B)-\mathcal{C}(A:B)",
            Discipline::QuantumMechanics,
            vec![v("D","quantum discord","",""), v("C","classical correlations","","")],
            vec![], &["quantum correlation","discord","beyond entanglement","non-classical","measurement"], 0.88),

        PhysicsEquation::new("Grover search speedup: O(√N) queries vs O(N) classical", r"T_{\text{Grover}}=O(\sqrt{N})",
            Discipline::QuantumMechanics,
            vec![v("N","search space size","","")],
            vec![], &["quantum speedup","search","Grover","quadratic speedup","quantum algorithm"], 0.95),

        PhysicsEquation::new("Shor's algorithm: period-finding in O((log N)^3) — exponential speedup for factoring", r"T_{\text{Shor}}=O((\log N)^3)",
            Discipline::QuantumMechanics,
            vec![v("N","integer to factor","","")],
            vec![], &["quantum algorithm","factoring","cryptography","exponential speedup","period finding"], 0.95),

        PhysicsEquation::new("Quantum error correction: 3-qubit bit flip code encodes 1 logical qubit", r"|0_L\rangle=|000\rangle,\;|1_L\rangle=|111\rangle",
            Discipline::QuantumMechanics,
            vec![v("|0_L⟩","logical zero","",""), v("|1_L⟩","logical one","","")],
            vec![], &["error correction","fault tolerance","logical qubit","redundancy","quantum code"], 0.92),

        PhysicsEquation::new("Quantum Fisher information: F_Q[ρ,H] = 2 Σ_{m,n} |<m|H|n>|² (p_m-p_n)² / (p_m+p_n)", r"F_Q[\rho,H]=2\sum_{m\neq n}\frac{(p_m-p_n)^2}{p_m+p_n}|\langle m|H|n\rangle|^2",
            Discipline::QuantumMechanics,
            vec![v("F_Q","quantum Fisher information","","")],
            vec![], &["quantum metrology","parameter estimation","Heisenberg limit","precision","sensing"], 0.91),

        PhysicsEquation::new("Decoherence time: τ_D ~ τ_relax / (Δx/λ_dB)² (larger/hotter = faster decoherence)", r"\tau_D\sim\tau_{\text{relax}}\left(\frac{\lambda_{dB}}{\Delta x}\right)^2",
            Discipline::QuantumMechanics,
            vec![v("τ_D","decoherence time","",""), v("Δx","spatial superposition size","",""), v("λ_dB","thermal de Broglie wavelength","","")],
            vec![], &["decoherence","quantum to classical","measurement","environment","macroscopic"], 0.92),

        PhysicsEquation::new("Quantum teleportation: fidelity F = 1 for perfect Bell state (F > 2/3 beats classical)", r"F=\langle\psi_{\text{out}}|\psi_{\text{in}}\rangle=1\text{ (ideal)}",
            Discipline::QuantumMechanics,
            vec![v("F","teleportation fidelity","","")],
            vec![], &["quantum teleportation","Bell state","classical limit","fidelity","entanglement"], 0.93),
    ]);

    // ── Cross-Disciplinary Bridges ─────────────────────────────────────────────
    // These equations link disciplines where canonical connections don't exist —
    // the core purpose of the EEM: surfacing the physics of unbounded problems.
    eqs.extend(vec![
        PhysicsEquation::new("Free energy principle: F = E_q[log q - log p] = KL[q||p] - log p(y) (Friston)", r"F=\mathbb{E}_q[\log q(\phi)-\log p(\phi,y)]\geq -\log p(y)",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("F","variational free energy","",""), v("q","recognition density","",""), v("p","generative model","","")],
            vec![], &["free energy","active inference","prediction error","perception","action","self-organization","belief updating"], 0.92),

        PhysicsEquation::new("Maximum entropy principle: p* = argmax H(p) subject to constraints E_p[f_k] = c_k (Jaynes)", r"p^*=\arg\max_p\left[-\sum_x p(x)\log p(x)\right]\;\text{s.t. }\mathbb{E}_p[f_k]=c_k",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("p*","maximum entropy distribution","",""), v("f_k","constraint functions","","")],
            vec![], &["maximum entropy","Jaynes","unbiased inference","least assumption","statistical mechanics","information theory"], 0.95),

        PhysicsEquation::new("Power law: P(x) ∝ x^{-α} (scale-free, self-similar)", r"P(x)\propto x^{-\alpha}",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("α","power law exponent","","")],
            vec![], &["power law","scale-free","heavy tail","self-similarity","critical phenomena","scale invariance"], 0.97),

        PhysicsEquation::new("Percolation threshold: giant component emerges at p_c (network phase transition)", r"p_c=\frac{1}{\langle k\rangle}\text{ (random graph)}",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("p_c","percolation threshold","",""), v("⟨k⟩","mean degree","","")],
            vec![], &["percolation","phase transition","network","connectivity","cascade","contagion"], 0.93),

        PhysicsEquation::new("Small-world network: L ~ log N / log k, C >> C_random (Watts-Strogatz)", r"L\sim\frac{\log N}{\log k},\quad C\gg C_{\text{rand}}",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("L","average path length","",""), v("C","clustering coefficient","",""), v("N","nodes","","")],
            vec![], &["small world","six degrees","network","clustering","social network","information spread"], 0.94),

        PhysicsEquation::new("Ising model: H = -J Σ s_i s_j - h Σ s_i (collective opinion dynamics)", r"H=-J\sum_{\langle i,j\rangle}s_is_j-h\sum_i s_i",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("J","coupling strength","",""), v("h","external field","",""), v("s_i","spin / opinion ±1","","")],
            vec![], &["opinion dynamics","social physics","collective behavior","phase transition","ferromagnetism","majority rule"], 0.94),

        PhysicsEquation::new("Mean field theory: m = tanh(βJzm + βh) (self-consistent opinion)", r"m=\tanh(\beta Jzm+\beta h)",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("m","mean field / average opinion","",""), v("z","coordination number","",""), v("β","inverse temperature","","")],
            vec![], &["mean field","self-consistent","opinion","social influence","critical point"], 0.92),

        PhysicsEquation::new("Renormalization group fixed point: β(g*) = 0, universality class determined by RG flow", r"\beta(g^*)=\mu\frac{dg}{d\mu}\bigg|_{g=g^*}=0",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("g*","fixed point coupling","",""), v("β","beta function","","")],
            vec![], &["universality","fixed point","scale invariance","critical phenomena","RG","cross-scale"], 0.93),

        PhysicsEquation::new("Complexity: K(x) = shortest program length (Kolmogorov) — limits compression", r"K(x)=\min_{p:\mathcal{U}(p)=x}|p|",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("K(x)","Kolmogorov complexity","","")],
            vec![], &["complexity","incompressibility","randomness","minimum description","algorithmic information"], 0.90),

        PhysicsEquation::new("Integrated information Φ: consciousness = irreducible causal structure (Tononi)", r"\Phi=\min_{\text{partition}}I(M_0;M_t)",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("Φ","integrated information","","")],
            vec![], &["consciousness","integrated information","irreducibility","causation","whole vs parts"], 0.82),

        PhysicsEquation::new("Fitness landscape: W(genotype) evolves by gradient ascent on W — applies to ideas/products", r"\frac{dp_i}{dt}=p_i(W_i-\bar W)",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("W_i","fitness of type i","",""), v("p_i","frequency","","")],
            vec![], &["fitness landscape","selection","adaptation","memetics","innovation","strategy evolution"], 0.91),

        PhysicsEquation::new("Cascade failure: σ²_load > σ²_capacity → systemic collapse (hidden cross-dependency)", r"P(\text{cascade})\to 1\text{ when }\frac{\sigma_\ell^2}{\sigma_c^2}>1",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("σ_ℓ","load variance","",""), v("σ_c","capacity variance","","")],
            vec![], &["cascade","systemic risk","contagion","fragility","interdependency","network failure"], 0.90),

        PhysicsEquation::new("Red Queen hypothesis: dx/dt = x*(f(x,y)-φ), dy/dt = y*(g(x,y)-φ) — coevolution arms race", r"\dot x=x(f(x,y)-\bar\phi),\quad\dot y=y(g(x,y)-\bar\phi)",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("f","fitness of x given y","",""), v("g","fitness of y given x","","")],
            vec![], &["coevolution","arms race","Red Queen","competitive escalation","technology race","market dynamics"], 0.89),

        PhysicsEquation::new("Schelling segregation: local preference → global segregation (emergent macro from micro)", r"u_i=1\text{ if }f_{\text{same}}\geq\theta,\;0\text{ otherwise}",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("θ","tolerance threshold","",""), v("f_same","fraction of similar neighbors","","")],
            vec![], &["emergence","segregation","tipping point","local preference","global pattern","self-organization"], 0.91),

        PhysicsEquation::new("NK complexity model: W(σ) = (1/N) Σ f_i(σ_i, neighbors) — rugged fitness landscape", r"W(\sigma)=\frac{1}{N}\sum_{i=1}^N f_i(\sigma_i,\sigma_{j_1},\ldots,\sigma_{j_K})",
            Discipline::Custom("CrossDisciplinary".into()),
            vec![v("N","elements","",""), v("K","interdependencies per element","",""), v("f_i","local fitness contribution","","")],
            vec![], &["rugged landscape","complexity","interdependency","adaptation","local optima","search"], 0.88),
    ]);

    // ══════════════════════════════════════════════════════════════════════
    // ENGINEERING & APPLIED SCIENCE  (added 2026-04-26)
    // 65 equations across Control / DSP / Chemistry / CS / Optimization /
    // ML / Electrical / Structural / Epidemiology / Economics
    // ══════════════════════════════════════════════════════════════════════

    // ── Control Theory ────────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("u(t) = K_p e(t) + K_i ∫e dt + K_d de/dt", r"u(t)=K_pe(t)+K_i\int e\,dt+K_d\frac{de}{dt}",
            Discipline::Custom("ControlTheory".into()),
            vec![v("u","control signal","",""), v("e","setpoint error","",""), v("K_p","proportional gain","",""), v("K_i","integral gain","",""), v("K_d","derivative gain","","")],
            vec![], &["pid controller","setpoint tracking","linear plant"], 0.92),
        PhysicsEquation::new("G(s) = Y(s)/U(s)", r"G(s)=\frac{Y(s)}{U(s)}",
            Discipline::Custom("ControlTheory".into()),
            vec![v("G","transfer function","",""), v("s","Laplace variable","","")],
            vec![], &["lti system","zero initial conditions"], 0.94),
        PhysicsEquation::new("ẋ = Ax + Bu, y = Cx + Du", r"\dot{x}=Ax+Bu,\;y=Cx+Du",
            Discipline::Custom("ControlTheory".into()),
            vec![v("x","state vector","",""), v("u","input","",""), v("y","output","","")],
            vec![], &["state-space form","linear time-invariant"], 0.95),
        PhysicsEquation::new("V̇(x) < 0 (Lyapunov stability)", r"\dot V(x)<0",
            Discipline::Custom("ControlTheory".into()),
            vec![v("V","Lyapunov function","","")],
            vec![], &["positive definite V","equilibrium at origin"], 0.93),
        PhysicsEquation::new("PM = ∠G(jω_gc) + 180°", r"\text{PM}=\angle G(j\omega_{gc})+180^\circ",
            Discipline::Custom("ControlTheory".into()),
            vec![v("PM","phase margin","°","")],
            vec![], &["bode analysis","gain-crossover frequency"], 0.91),
        PhysicsEquation::new("x̂_{k|k} = x̂_{k|k-1} + K_k(z_k - H x̂_{k|k-1})", r"\hat x_{k|k}=\hat x_{k|k-1}+K_k(z_k-H\hat x_{k|k-1})",
            Discipline::Custom("ControlTheory".into()),
            vec![v("K_k","Kalman gain","",""), v("z_k","measurement","","")],
            vec![], &["linear gaussian","optimal estimator"], 0.94),
    ]);

    // ── Signal Processing ─────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("X(f) = ∫ x(t) e^{-j 2π f t} dt", r"X(f)=\int x(t)e^{-j2\pi ft}dt",
            Discipline::Custom("SignalProcessing".into()),
            vec![v("X","spectrum","",""), v("x","time-domain signal","",""), v("f","frequency","Hz","")],
            vec![], &["fourier transform","absolutely integrable"], 0.95),
        PhysicsEquation::new("f_s ≥ 2 f_max  (Nyquist)", r"f_s\geq 2f_{\max}",
            Discipline::Custom("SignalProcessing".into()),
            vec![v("f_s","sampling rate","Hz",""), v("f_max","signal bandwidth","Hz","")],
            vec![], &["bandlimited signal"], 0.96),
        PhysicsEquation::new("(x * h)(t) = ∫ x(τ) h(t-τ) dτ", r"(x*h)(t)=\int x(\tau)h(t-\tau)d\tau",
            Discipline::Custom("SignalProcessing".into()),
            vec![v("h","filter impulse response","","")],
            vec![], &["lti filter"], 0.95),
        PhysicsEquation::new("X(z) = Σ x[n] z^{-n}", r"X(z)=\sum_{n=-\infty}^{\infty}x[n]z^{-n}",
            Discipline::Custom("SignalProcessing".into()),
            vec![v("z","z-transform variable","","")],
            vec![], &["discrete-time","sampled signal"], 0.93),
        PhysicsEquation::new("SNR_{dB} = 10 log_{10}(P_signal / P_noise)", r"\text{SNR}_{dB}=10\log_{10}\frac{P_{\text{signal}}}{P_{\text{noise}}}",
            Discipline::Custom("SignalProcessing".into()),
            vec![v("P_signal","signal power","W","")],
            vec![], &[], 0.94),
        PhysicsEquation::new("C = B log_2(1 + S/N)  (Shannon-Hartley)", r"C=B\log_2(1+S/N)",
            Discipline::Custom("SignalProcessing".into()),
            vec![v("C","channel capacity","bits/s",""), v("B","bandwidth","Hz","")],
            vec![], &["awgn channel"], 0.95),
    ]);

    // ── Chemistry ─────────────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("k = A e^{-E_a / R T}  (Arrhenius)", r"k=Ae^{-E_a/RT}",
            Discipline::Custom("Chemistry".into()),
            vec![v("k","rate constant","",""), v("E_a","activation energy","J/mol",""), v("R","gas constant","J/(mol·K)","")],
            vec![], &["elementary reaction","temperature dependence"], 0.95),
        PhysicsEquation::new("rate = k [A]^m [B]^n", r"\text{rate}=k[A]^m[B]^n",
            Discipline::Custom("Chemistry".into()),
            vec![v("k","rate constant","",""), v("[A]","molar concentration","mol/L","")],
            vec![], &["empirical rate law"], 0.93),
        PhysicsEquation::new("ΔG = ΔH - TΔS", r"\Delta G=\Delta H-T\Delta S",
            Discipline::Custom("Chemistry".into()),
            vec![v("ΔG","Gibbs free energy","J/mol","")],
            vec![], &["constant T, P"], 0.95),
        PhysicsEquation::new("pH = -log_{10}[H+]", r"\text{pH}=-\log_{10}[\text{H}^+]",
            Discipline::Custom("Chemistry".into()),
            vec![v("[H+]","hydrogen ion molarity","mol/L","")],
            vec![], &["aqueous solution"], 0.96),
        PhysicsEquation::new("pH = pK_a + log_{10}([A-]/[HA])  (Henderson-Hasselbalch)", r"\text{pH}=\text{p}K_a+\log_{10}\frac{[A^-]}{[\text{HA}]}",
            Discipline::Custom("Chemistry".into()),
            vec![v("pK_a","acid dissociation","",""), v("[A-]","conjugate base","",""), v("[HA]","weak acid","","")],
            vec![], &["buffer solution"], 0.94),
        PhysicsEquation::new("A = ε l c  (Beer-Lambert)", r"A=\varepsilon l c",
            Discipline::Custom("Chemistry".into()),
            vec![v("A","absorbance","",""), v("ε","molar absorptivity","",""), v("l","path length","cm",""), v("c","concentration","mol/L","")],
            vec![], &["dilute solution","monochromatic light"], 0.94),
        PhysicsEquation::new("E = E° - (RT/nF) ln Q  (Nernst)", r"E=E^\circ-\frac{RT}{nF}\ln Q",
            Discipline::Custom("Chemistry".into()),
            vec![v("E","cell potential","V",""), v("F","Faraday constant","C/mol",""), v("Q","reaction quotient","","")],
            vec![], &["redox half-cell","equilibrium displacement"], 0.94),
    ]);

    // ── Computer Science ──────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("T(n) = a T(n/b) + f(n)  (Master theorem)", r"T(n)=aT(n/b)+f(n)",
            Discipline::Custom("ComputerScience".into()),
            vec![v("a","subproblems","",""), v("b","subproblem size factor","","")],
            vec![], &["divide-and-conquer recurrence"], 0.95),
        PhysicsEquation::new("S(n) = 1 / ((1-p) + p/n)  (Amdahl)", r"S(n)=\frac{1}{(1-p)+p/n}",
            Discipline::Custom("ComputerScience".into()),
            vec![v("p","parallelizable fraction","",""), v("n","processors","","")],
            vec![], &["fixed problem size"], 0.95),
        PhysicsEquation::new("S(n) = (1-p) + n p  (Gustafson)", r"S(n)=(1-p)+np",
            Discipline::Custom("ComputerScience".into()),
            vec![v("p","parallel fraction at scale","",""), v("n","processors","","")],
            vec![], &["fixed time","scaled problem"], 0.93),
        PhysicsEquation::new("L = λ W  (Little's Law)", r"L=\lambda W",
            Discipline::Custom("ComputerScience".into()),
            vec![v("L","items in system","",""), v("λ","arrival rate","1/s",""), v("W","time in system","s","")],
            vec![], &["steady-state queue"], 0.96),
        PhysicsEquation::new("ρ = λ/μ  (M/M/1 utilization)", r"\rho=\lambda/\mu",
            Discipline::Custom("ComputerScience".into()),
            vec![v("μ","service rate","1/s","")],
            vec![], &["poisson arrivals","exponential service"], 0.94),
    ]);

    // ── Optimization ──────────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("x_{k+1} = x_k - α ∇f(x_k)", r"x_{k+1}=x_k-\alpha\nabla f(x_k)",
            Discipline::Custom("Optimization".into()),
            vec![v("α","learning rate","",""), v("∇f","gradient","","")],
            vec![], &["differentiable f","step size α"], 0.95),
        PhysicsEquation::new("x_{k+1} = x_k - [H f(x_k)]^{-1} ∇f(x_k)", r"x_{k+1}=x_k-[Hf(x_k)]^{-1}\nabla f(x_k)",
            Discipline::Custom("Optimization".into()),
            vec![v("Hf","Hessian","","")],
            vec![], &["twice differentiable","H invertible"], 0.93),
        PhysicsEquation::new("∇f(x*) = Σ λ_i ∇g_i(x*)  (Lagrange)", r"\nabla f(x^*)=\sum_i\lambda_i\nabla g_i(x^*)",
            Discipline::Custom("Optimization".into()),
            vec![v("λ","multipliers","","")],
            vec![], &["equality constraints","regular point"], 0.94),
        PhysicsEquation::new("KKT: ∇f + Σλ∇g + Σμ∇h = 0, μ≥0, μ_i h_i = 0", r"\nabla\mathcal{L}=0,\;\mu\geq 0,\;\mu_i h_i=0",
            Discipline::Custom("Optimization".into()),
            vec![],
            vec![], &["constraint qualification","convex problem"], 0.93),
        PhysicsEquation::new("V(s) = max_a [r(s,a) + γ Σ p(s'|s,a) V(s')]  (Bellman)", r"V(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V(s')\right]",
            Discipline::Custom("Optimization".into()),
            vec![v("V","value function","",""), v("γ","discount factor","","")],
            vec![], &["markov decision process"], 0.94),
    ]);

    // ── Machine Learning ──────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("y = σ(Wx + b)  (perceptron)", r"y=\sigma(Wx+b)",
            Discipline::Custom("MachineLearning".into()),
            vec![v("W","weights","",""), v("b","bias","",""), v("σ","activation","","")],
            vec![], &["differentiable activation"], 0.94),
        PhysicsEquation::new("σ(z) = 1/(1+e^{-z})", r"\sigma(z)=\frac{1}{1+e^{-z}}",
            Discipline::Custom("MachineLearning".into()),
            vec![],
            vec![], &["logistic activation"], 0.95),
        PhysicsEquation::new("softmax(z)_i = e^{z_i} / Σ e^{z_j}", r"\text{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}",
            Discipline::Custom("MachineLearning".into()),
            vec![],
            vec![], &["multi-class normalization"], 0.95),
        PhysicsEquation::new("L = -Σ y_i log(ŷ_i)  (cross-entropy)", r"L=-\sum_i y_i\log\hat y_i",
            Discipline::Custom("MachineLearning".into()),
            vec![v("y","one-hot label","",""), v("ŷ","predicted probability","","")],
            vec![], &["classification","calibrated softmax"], 0.95),
        PhysicsEquation::new("∂L/∂W_l = δ_l a_{l-1}^T  (backprop)", r"\frac{\partial L}{\partial W_l}=\delta_l a_{l-1}^T",
            Discipline::Custom("MachineLearning".into()),
            vec![v("δ","layer gradient","","")],
            vec![], &["chain rule","feedforward network"], 0.94),
        PhysicsEquation::new("Attention(Q,K,V) = softmax(QK^T/√d) V", r"\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V",
            Discipline::Custom("MachineLearning".into()),
            vec![v("Q","queries","",""), v("K","keys","",""), v("V","values","","")],
            vec![], &["scaled dot-product","transformer block"], 0.94),
        PhysicsEquation::new("w ← w + η (a_pre + trace_pre)(a_post + trace_post)  (Hebb-STDP)", r"w\leftarrow w+\eta(a_{\text{pre}}+\tau_{\text{pre}})(a_{\text{post}}+\tau_{\text{post}})",
            Discipline::Custom("MachineLearning".into()),
            vec![v("η","learning rate","",""), v("a","activation","",""), v("τ","trace","","")],
            vec![], &["local learning rule","neuromorphic"], 0.92),
    ]);

    // ── Electrical Engineering ────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("V = I R  (Ohm's law)", r"V=IR",
            Discipline::Custom("ElectricalEngineering".into()),
            vec![v("V","voltage","V",""), v("I","current","A",""), v("R","resistance","Ω","")],
            vec![], &["linear resistor","steady current"], 0.97),
        PhysicsEquation::new("Σ V = 0 around loop  (KVL)", r"\sum_k V_k=0",
            Discipline::Custom("ElectricalEngineering".into()),
            vec![],
            vec![], &["lumped element circuit"], 0.96),
        PhysicsEquation::new("Σ I = 0 at node  (KCL)", r"\sum_k I_k=0",
            Discipline::Custom("ElectricalEngineering".into()),
            vec![],
            vec![], &["charge conservation"], 0.96),
        PhysicsEquation::new("τ = R C  (RC time constant)", r"\tau=RC",
            Discipline::Custom("ElectricalEngineering".into()),
            vec![v("τ","time constant","s",""), v("C","capacitance","F","")],
            vec![], &["first-order rc circuit"], 0.95),
        PhysicsEquation::new("ω_0 = 1/√(LC)  (LC resonance)", r"\omega_0=1/\sqrt{LC}",
            Discipline::Custom("ElectricalEngineering".into()),
            vec![v("L","inductance","H",""), v("C","capacitance","F","")],
            vec![], &["lossless lc tank"], 0.94),
        PhysicsEquation::new("P = V I cos(φ)  (real power)", r"P=VI\cos\phi",
            Discipline::Custom("ElectricalEngineering".into()),
            vec![v("φ","phase angle","rad","")],
            vec![], &["sinusoidal steady state"], 0.95),
    ]);

    // ── Structural / Mechanical Engineering ───────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("σ = E ε  (Hooke's law for stress)", r"\sigma=E\varepsilon",
            Discipline::Custom("StructuralEngineering".into()),
            vec![v("σ","stress","Pa",""), v("E","Young's modulus","Pa",""), v("ε","strain","","")],
            vec![], &["linear elastic regime"], 0.96),
        PhysicsEquation::new("P_cr = π² E I / (K L)²  (Euler buckling)", r"P_{cr}=\frac{\pi^2 EI}{(KL)^2}",
            Discipline::Custom("StructuralEngineering".into()),
            vec![v("I","moment of inertia","",""), v("L","length","m",""), v("K","effective length factor","","")],
            vec![], &["slender column","ideal pinning"], 0.94),
        PhysicsEquation::new("σ_v = √(½[(σ_1-σ_2)² + (σ_2-σ_3)² + (σ_1-σ_3)²])  (von Mises)", r"\sigma_v=\sqrt{\tfrac{1}{2}[(\sigma_1-\sigma_2)^2+(\sigma_2-\sigma_3)^2+(\sigma_1-\sigma_3)^2]}",
            Discipline::Custom("StructuralEngineering".into()),
            vec![],
            vec![], &["ductile yielding criterion"], 0.93),
        PhysicsEquation::new("y(x) = (P L³)/(48 E I)  (simply supported beam, center deflection)", r"y_{\max}=\frac{PL^3}{48EI}",
            Discipline::Custom("StructuralEngineering".into()),
            vec![v("P","point load at center","N","")],
            vec![], &["simply supported beam","linear elastic"], 0.93),
    ]);

    // ── Epidemiology ──────────────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("dS/dt = -β S I / N", r"\frac{dS}{dt}=-\beta SI/N",
            Discipline::Custom("Epidemiology".into()),
            vec![v("S","susceptible","",""), v("I","infected","",""), v("β","transmission rate","",""), v("N","total population","","")],
            vec![], &["sir model","well-mixed population"], 0.94),
        PhysicsEquation::new("dI/dt = β S I / N - γ I", r"\frac{dI}{dt}=\beta SI/N-\gamma I",
            Discipline::Custom("Epidemiology".into()),
            vec![v("γ","recovery rate","","")],
            vec![], &["sir model"], 0.94),
        PhysicsEquation::new("dR/dt = γ I", r"\frac{dR}{dt}=\gamma I",
            Discipline::Custom("Epidemiology".into()),
            vec![v("R","recovered","","")],
            vec![], &["sir model"], 0.95),
        PhysicsEquation::new("R_0 = β / γ", r"R_0=\beta/\gamma",
            Discipline::Custom("Epidemiology".into()),
            vec![v("R_0","basic reproduction number","","")],
            vec![], &["fully susceptible population"], 0.95),
        PhysicsEquation::new("v_c = 1 - 1/R_0  (herd immunity threshold)", r"v_c=1-1/R_0",
            Discipline::Custom("Epidemiology".into()),
            vec![v("v_c","vaccination threshold","","")],
            vec![], &["uniform mixing","perfect vaccine"], 0.93),
    ]);

    // ── Economics & Finance ───────────────────────────────────────────────
    eqs.extend(vec![
        PhysicsEquation::new("Y = A K^α L^{1-α}  (Cobb-Douglas)", r"Y=AK^\alpha L^{1-\alpha}",
            Discipline::Custom("Economics".into()),
            vec![v("Y","output","",""), v("K","capital","",""), v("L","labor","",""), v("A","total factor productivity","","")],
            vec![], &["constant returns to scale"], 0.93),
        PhysicsEquation::new("E_d = (%ΔQ_d) / (%ΔP)  (price elasticity of demand)", r"E_d=\frac{\%\Delta Q_d}{\%\Delta P}",
            Discipline::Custom("Economics".into()),
            vec![v("Q_d","quantity demanded","",""), v("P","price","","")],
            vec![], &["small price change","ceteris paribus"], 0.92),
        PhysicsEquation::new("PV = FV / (1 + r)^t", r"\text{PV}=\frac{\text{FV}}{(1+r)^t}",
            Discipline::Custom("Economics".into()),
            vec![v("PV","present value","",""), v("FV","future value","",""), v("r","discount rate","",""), v("t","time periods","","")],
            vec![], &["constant rate","compounded periodically"], 0.96),
        PhysicsEquation::new("E[R_i] = R_f + β_i (E[R_m] - R_f)  (CAPM)", r"E[R_i]=R_f+\beta_i(E[R_m]-R_f)",
            Discipline::Custom("Economics".into()),
            vec![v("R_f","risk-free rate","",""), v("β_i","asset beta","",""), v("R_m","market return","","")],
            vec![], &["efficient market","mean-variance"], 0.92),
        PhysicsEquation::new("C = S_0 N(d_1) - K e^{-rT} N(d_2)  (Black-Scholes call)", r"C=S_0 N(d_1)-Ke^{-rT}N(d_2)",
            Discipline::Custom("Economics".into()),
            vec![v("C","call price","",""), v("S_0","spot","",""), v("K","strike","",""), v("T","time to expiry","","")],
            vec![], &["log-normal price","constant volatility","european option"], 0.91),
        PhysicsEquation::new("S = (E[R_p] - R_f) / σ_p  (Sharpe ratio)", r"S=\frac{E[R_p]-R_f}{\sigma_p}",
            Discipline::Custom("Economics".into()),
            vec![v("σ_p","portfolio std dev","","")],
            vec![], &["mean-variance frame"], 0.93),
    ]);

    eqs
}

// ─── EEM ↔ Motif feedback loop ────────────────────────────────────────────────

/// One feedback step between the equation matrix and the hierarchical motif
/// runtime.  Two arrows:
///
///   1. Active labels → motif observation → EEM apply_and_reinforce.
///      Recurring patterns in the input stream automatically drive equation
///      reinforcement; equations not corroborated by repeated activations
///      gradually decay (`decay_step`).
///
///   2. Matched equations → discipline keywords → motif observation.
///      Any equation that matches strongly seeds its discipline keyword list
///      back into the motif runtime, so the network develops "physics motifs"
///      — recurring sequences of equation keywords that the motif graph
///      tracks alongside raw sensor labels.  Over time, mass behavior in the
///      input stream is detected as a motif over equation-keyword sequences,
///      not just over raw labels.
///
/// `relevance_threshold` controls which equations get reinforced (default
/// 0.3).  Returns the matched-equation count.
pub fn motif_eem_feedback_step(
    motifs: &mut crate::streaming::HierarchicalMotifRuntime,
    eem: &EquationMatrixRuntime,
    active_labels: &[String],
    sensor_dims: u8,
    timestamp: crate::schema::Timestamp,
    relevance_threshold: f32,
) -> usize {
    if active_labels.is_empty() { return 0; }

    // (1) Forward: observe raw labels in the motif runtime.
    let _ = motifs.observe_label_sequence(active_labels, timestamp, 1.0);

    // (2) EEM matches + reinforces from the same labels.
    let matches = eem.apply_and_reinforce(
        active_labels, sensor_dims, relevance_threshold,
    );

    // (3) Backward: every matched equation's keywords go back into the motif
    //     runtime as a synthetic level-0 sequence prefixed with `eem:` so the
    //     keywords are tracked separately from raw sensor labels and don't
    //     collide with their original neuron ids.
    let mut matched_count = 0;
    for (_result, keywords) in &matches {
        if keywords.is_empty() { continue; }
        matched_count += 1;
        let kw_labels: Vec<String> = keywords.iter()
            .map(|k| format!("eem:{k}"))
            .collect();
        let _ = motifs.observe_label_sequence(&kw_labels, timestamp, 0.5);
    }
    matched_count
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
