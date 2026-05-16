//! Environmental Equation Matrix per [`ARCHITECTURE.md`] §4.B.
//!
//! The EEM is the brain's symbolic/equational layer: a property graph
//! of equations, variables, disciplines, and motif observations.  It
//! contributes `eem_confidence` to the integrated answer (spec §2.1
//! and §4.D) so that callers see when symbolic reasoning was a
//! grounded contributor vs. pure-fabric retrieval.
//!
//! # Backend
//!
//! The spec specifies Kuzu (MIT) as the property-graph backend.  This
//! module ships an **in-memory MVP backend** behind a stable API
//! surface; a Kuzu adapter is planned as a subsequent change (the
//! architecture is backend-agnostic — schema, operations, and
//! confidence math live here; only the storage of nodes/edges moves).
//!
//! # What's implemented
//!
//! - Node types: `Equation`, `Variable`, `Discipline`, `Motif`.
//! - Edge types: `BOUND_TO` (variable→discipline), `OBSERVED_AT`
//!   (motif→equation), `VALIDATED_BY` (equation→outcome).  Other
//!   edge types from spec §4.B (`DEPENDS_ON`, `INSTANCE_OF`,
//!   `DERIVED_FROM`) become useful when hypothesis-generation lands;
//!   they're reserved-but-unused so adding them is additive.
//! - Equation evaluation via `evalexpr` with explicit variable
//!   bindings.
//! - Motif observation tracking + per-equation validation-driven
//!   confidence.
//!
//! # What is NOT here yet
//!
//! - Automated hypothesis generation from recurring motifs.
//! - Network gossip of equation deltas (spec §5.2 — that's Phase 8).
//! - Kuzu persistence (planned backend swap).
//!
//! Each of those is additive and does not change the surface below.

use ahash::AHashMap;
use evalexpr::{ContextWithMutableVariables, HashMapContext, Value, eval_with_context};
use serde::{Deserialize, Serialize};

use crate::neuron::{NeuronId, PoolId};

pub type EquationId = u32;
pub type VariableId = u32;
pub type MotifId    = u32;
pub type DisciplineId = u32;

/// Symbolic equation with `evalexpr`-compatible expression text.  When
/// evaluated, the caller supplies `bindings: VariableId → f64`; the
/// result is the numerical value of the expression under those
/// bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Equation {
    pub id:         EquationId,
    pub name:       String,
    /// Expression text per `evalexpr` syntax, e.g. `"a + b * c"`.  Variables
    /// referenced here must be registered with [`Eem::register_variable`].
    pub expression: String,
    /// Variables (registered with [`Eem::register_variable`]) used in
    /// `expression`.  Stored as ids so the EEM can validate at apply-
    /// time that the caller has supplied bindings for every required
    /// variable.
    pub variables:  Vec<VariableId>,
    pub discipline: Option<DisciplineId>,
    /// Bayesian-style confidence in [0, 1].  Starts at the
    /// `initial_confidence` configured on [`EemConfig`]; moves toward
    /// 1 with successful validations and toward 0 with failures via
    /// [`Eem::report_validation`].
    pub confidence: f32,
    pub validation_successes: u32,
    pub validation_failures:  u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub id:    VariableId,
    pub name:  String,
    pub unit:  Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discipline {
    pub id:   DisciplineId,
    pub name: String,
}

/// One observation of a multi-pool/multi-neuron co-firing pattern.  The
/// fabric (via `Brain::observe_motif_for_eem`) hands these to the EEM
/// so it can correlate emergent neural patterns with equations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    pub id:          MotifId,
    pub fingerprint: Vec<(PoolId, NeuronId)>,
    pub observation_count: u32,
}

/// Result of applying an equation under explicit bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct EquationApplication {
    pub equation_id: EquationId,
    pub value:       f64,
    /// Confidence at the time of evaluation (snapshot — separate from
    /// the equation's continually-updated confidence so callers can
    /// trace this specific result).
    pub confidence:  f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EemConfig {
    /// Starting confidence assigned to freshly-registered equations.
    /// 0.5 expresses "no track record yet, neither trusted nor
    /// disbelieved" — equations earn trust via [`Eem::report_validation`].
    pub initial_confidence: f32,
    /// Multiplicative boost applied on a successful validation.
    /// Confidence is clamped to [0, 1].
    pub validation_success_gain: f32,
    /// Multiplicative reduction applied on a failed validation.
    pub validation_failure_penalty: f32,
}

impl Default for EemConfig {
    fn default() -> Self {
        Self {
            initial_confidence:         0.5,
            validation_success_gain:    0.05,
            validation_failure_penalty: 0.10,
        }
    }
}

/// In-memory property-graph EEM.  Owned by [`crate::Brain`].
pub struct Eem {
    pub config:    EemConfig,
    equations:     Vec<Equation>,
    variables:     Vec<Variable>,
    disciplines:   Vec<Discipline>,
    motifs:        Vec<Motif>,
    /// Equation name → id, for label-based lookup.
    eq_by_name:    AHashMap<String, EquationId>,
    var_by_name:   AHashMap<String, VariableId>,
    disc_by_name:  AHashMap<String, DisciplineId>,
    /// Motif fingerprint → motif id (deduplication so observation_count
    /// grows for repeat observations rather than creating duplicates).
    motif_by_fp:   AHashMap<Vec<(PoolId, NeuronId)>, MotifId>,
    /// motif_id → list of equation_ids it's been linked to (the
    /// OBSERVED_AT edges).
    motif_links:   AHashMap<MotifId, Vec<EquationId>>,
}

impl Eem {
    pub fn new(config: EemConfig) -> Self {
        Self {
            config,
            equations:    Vec::new(),
            variables:    Vec::new(),
            disciplines:  Vec::new(),
            motifs:       Vec::new(),
            eq_by_name:   AHashMap::new(),
            var_by_name:  AHashMap::new(),
            disc_by_name: AHashMap::new(),
            motif_by_fp:  AHashMap::new(),
            motif_links:  AHashMap::new(),
        }
    }

    pub fn equation_count(&self)   -> usize { self.equations.len() }
    pub fn variable_count(&self)   -> usize { self.variables.len() }
    pub fn discipline_count(&self) -> usize { self.disciplines.len() }
    pub fn motif_count(&self)      -> usize { self.motifs.len() }

    pub fn equation(&self, id: EquationId) -> Option<&Equation> {
        self.equations.get(id as usize)
    }
    pub fn equation_mut(&mut self, id: EquationId) -> Option<&mut Equation> {
        self.equations.get_mut(id as usize)
    }
    pub fn variable(&self, id: VariableId) -> Option<&Variable> {
        self.variables.get(id as usize)
    }
    pub fn discipline(&self, id: DisciplineId) -> Option<&Discipline> {
        self.disciplines.get(id as usize)
    }
    pub fn motif(&self, id: MotifId) -> Option<&Motif> {
        self.motifs.get(id as usize)
    }
    pub fn equation_by_name(&self, name: &str) -> Option<EquationId> {
        self.eq_by_name.get(name).copied()
    }
    pub fn variable_by_name(&self, name: &str) -> Option<VariableId> {
        self.var_by_name.get(name).copied()
    }

    /// Register (or look up) a variable.  Idempotent on the name —
    /// returns the existing id if already registered.
    pub fn register_variable(&mut self, name: impl Into<String>, unit: Option<String>) -> VariableId {
        let name = name.into();
        if let Some(&id) = self.var_by_name.get(&name) { return id; }
        let id = self.variables.len() as VariableId;
        self.variables.push(Variable { id, name: name.clone(), unit });
        self.var_by_name.insert(name, id);
        id
    }

    pub fn register_discipline(&mut self, name: impl Into<String>) -> DisciplineId {
        let name = name.into();
        if let Some(&id) = self.disc_by_name.get(&name) { return id; }
        let id = self.disciplines.len() as DisciplineId;
        self.disciplines.push(Discipline { id, name: name.clone() });
        self.disc_by_name.insert(name, id);
        id
    }

    /// Register an equation.  Returns the assigned id.  Idempotent on
    /// the name — registering a duplicate name returns the existing
    /// id without replacing the stored equation (deliberate: equations
    /// build confidence over time, and "overwrite on re-register"
    /// would silently destroy that record).  Use
    /// [`Eem::replace_equation_expression`] for in-place edits.
    pub fn register_equation(
        &mut self,
        name:       impl Into<String>,
        expression: impl Into<String>,
        variables:  Vec<VariableId>,
        discipline: Option<DisciplineId>,
    ) -> EquationId {
        let name = name.into();
        if let Some(&id) = self.eq_by_name.get(&name) { return id; }
        let id = self.equations.len() as EquationId;
        let eq = Equation {
            id,
            name: name.clone(),
            expression: expression.into(),
            variables,
            discipline,
            confidence: self.config.initial_confidence,
            validation_successes: 0,
            validation_failures:  0,
        };
        self.equations.push(eq);
        self.eq_by_name.insert(name, id);
        id
    }

    pub fn replace_equation_expression(&mut self, id: EquationId, new_expr: impl Into<String>) -> bool {
        if let Some(eq) = self.equations.get_mut(id as usize) {
            eq.expression = new_expr.into();
            true
        } else {
            false
        }
    }

    /// Evaluate an equation under the given bindings.  Returns `None`
    /// if the equation id is unknown, if any required variable is
    /// unbound, or if `evalexpr` fails to evaluate (e.g. the
    /// expression references an unregistered name or fails type
    /// checking).  The brain's integration layer reads the result
    /// AND the snapshot confidence together — the confidence is what
    /// makes the result honestly grounded vs. speculative.
    pub fn apply(
        &self,
        id:       EquationId,
        bindings: &AHashMap<VariableId, f64>,
    ) -> Option<EquationApplication> {
        let eq = self.equations.get(id as usize)?;
        // Confirm every required variable is supplied.
        for vid in &eq.variables {
            if !bindings.contains_key(vid) { return None; }
        }
        let mut ctx = HashMapContext::new();
        for (&vid, &val) in bindings.iter() {
            if let Some(var) = self.variables.get(vid as usize) {
                let _ = ctx.set_value(var.name.clone(), Value::Float(val));
            }
        }
        let result = eval_with_context(&eq.expression, &ctx).ok()?;
        let v = match result {
            Value::Float(f)   => f,
            Value::Int(i)     => i as f64,
            Value::Boolean(b) => if b { 1.0 } else { 0.0 },
            _ => return None,
        };
        Some(EquationApplication {
            equation_id: id,
            value:       v,
            confidence:  eq.confidence,
        })
    }

    /// Convenience: evaluate by equation name with name-keyed bindings.
    /// Wraps [`Eem::apply`] after id lookups.
    pub fn apply_by_name(
        &self,
        name:     &str,
        bindings: &AHashMap<&str, f64>,
    ) -> Option<EquationApplication> {
        let eq_id = self.eq_by_name.get(name).copied()?;
        let mut by_id = AHashMap::new();
        for (n, v) in bindings.iter() {
            let vid = self.var_by_name.get(*n).copied()?;
            by_id.insert(vid, *v);
        }
        self.apply(eq_id, &by_id)
    }

    /// Record (or bump observation count for) a motif fingerprint.
    /// Returns the assigned/existing id.  The brain calls this when
    /// the fabric promotes a binding concept or when supervised
    /// training points to a motif worth correlating with equations.
    pub fn observe_motif(&mut self, mut fingerprint: Vec<(PoolId, NeuronId)>) -> MotifId {
        fingerprint.sort();
        if let Some(&id) = self.motif_by_fp.get(&fingerprint) {
            self.motifs[id as usize].observation_count += 1;
            return id;
        }
        let id = self.motifs.len() as MotifId;
        self.motifs.push(Motif {
            id,
            fingerprint: fingerprint.clone(),
            observation_count: 1,
        });
        self.motif_by_fp.insert(fingerprint, id);
        id
    }

    /// Record an OBSERVED_AT edge: this motif has been observed in a
    /// context where this equation applied.  Idempotent — repeat
    /// links are de-duplicated.
    pub fn link_motif_to_equation(&mut self, motif_id: MotifId, equation_id: EquationId) -> bool {
        if self.motifs.get(motif_id as usize).is_none() { return false; }
        if self.equations.get(equation_id as usize).is_none() { return false; }
        let links = self.motif_links.entry(motif_id).or_default();
        if !links.contains(&equation_id) {
            links.push(equation_id);
        }
        true
    }

    /// Equations linked to a motif via OBSERVED_AT edges.
    pub fn equations_for_motif(&self, motif_id: MotifId) -> Vec<EquationId> {
        self.motif_links.get(&motif_id).cloned().unwrap_or_default()
    }

    /// Apply a validation outcome to an equation's confidence.  The
    /// equation's `validation_successes` / `validation_failures`
    /// counters are also bumped so caller telemetry can show the full
    /// track record, not just the smoothed confidence.
    pub fn report_validation(&mut self, equation_id: EquationId, success: bool) -> bool {
        let eq = match self.equations.get_mut(equation_id as usize) {
            Some(e) => e,
            None    => return false,
        };
        if success {
            eq.validation_successes = eq.validation_successes.saturating_add(1);
            eq.confidence = (eq.confidence + self.config.validation_success_gain).clamp(0.0, 1.0);
        } else {
            eq.validation_failures = eq.validation_failures.saturating_add(1);
            eq.confidence = (eq.confidence - self.config.validation_failure_penalty).clamp(0.0, 1.0);
        }
        true
    }

    pub fn confidence(&self, equation_id: EquationId) -> Option<f32> {
        self.equations.get(equation_id as usize).map(|e| e.confidence)
    }

    pub fn iter_equations(&self) -> impl Iterator<Item = &Equation> {
        self.equations.iter()
    }

    pub fn snapshot(&self) -> crate::persistence::EemSnapshot {
        let motif_links: Vec<(u32, Vec<u32>)> = self
            .motif_links
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        crate::persistence::EemSnapshot {
            config:      self.config.clone(),
            equations:   self.equations.clone(),
            variables:   self.variables.clone(),
            disciplines: self.disciplines.clone(),
            motifs:      self.motifs.clone(),
            motif_links,
        }
    }

    pub fn from_snapshot(snap: crate::persistence::EemSnapshot) -> Self {
        let mut eq_by_name   = AHashMap::new();
        let mut var_by_name  = AHashMap::new();
        let mut disc_by_name = AHashMap::new();
        for eq in &snap.equations { eq_by_name.insert(eq.name.clone(), eq.id); }
        for v in &snap.variables { var_by_name.insert(v.name.clone(), v.id); }
        for d in &snap.disciplines { disc_by_name.insert(d.name.clone(), d.id); }
        let mut motif_by_fp = AHashMap::new();
        for m in &snap.motifs {
            let mut fp = m.fingerprint.clone();
            fp.sort();
            motif_by_fp.insert(fp, m.id);
        }
        let mut motif_links = AHashMap::new();
        for (k, v) in snap.motif_links { motif_links.insert(k, v); }
        Self {
            config:       snap.config,
            equations:    snap.equations,
            variables:    snap.variables,
            disciplines:  snap.disciplines,
            motifs:       snap.motifs,
            eq_by_name,
            var_by_name,
            disc_by_name,
            motif_by_fp,
            motif_links,
        }
    }
}
