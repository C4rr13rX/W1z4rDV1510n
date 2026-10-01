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
use crate::workspace::{CompositionRule, GroundedRelation, PatternValue, RelationPattern,
    TransientWorkspace, TypedValue};
use crate::crystallizer::{SemanticCrystallizer, SemanticFrame};

pub type EquationId = u32;
pub type VariableId = u32;
pub type MotifId    = u32;
pub type DisciplineId = u32;
pub type FactId       = u32;

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

/// Result of [`Eem::chain_explore`].  `reached_members` maps every
/// (pool, neuron) ref reached during the walk to its best-chain
/// confidence (product of fact confidences along the shortest path).
/// `visited_facts` is the set of fact ids traversed — caller can
/// inspect what reasoning chain produced an answer.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub reached_members: ahash::AHashMap<(PoolId, NeuronId), f32>,
    pub visited_facts:   Vec<FactId>,
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

/// One grounded fact the EEM has crystallized from validated sensor
/// experience.  A fact = a co-firing pattern across pools that the
/// substrate observed enough times (via binding-concept promotion) to
/// promote into a stable cross-pool relationship.
///
/// Facts are the EEM's bridge between the fabric's concept graph and
/// the equation/variable layer: each fact's `members` are NeuronRefs
/// pointing into specific pools, so the chain explorer can walk from
/// firing concepts to facts that involve them, then to OTHER concepts
/// in those facts (i.e., the trained partners), then to ANOTHER fact
/// involving those partners, and so on — ivy-growth equation chaining
/// over grounded experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedFact {
    pub id:                FactId,
    /// Brain's binding-pool neuron id that this fact crystallized from.
    pub source_binding:    NeuronId,
    /// (pool, neuron) refs of every atom/concept that co-fired in the
    /// binding event.  Stored sorted for deterministic chain matching.
    pub members:           Vec<(PoolId, NeuronId)>,
    /// Confidence ∈ [0, 1].  Starts at 1.0 (it crystallized from
    /// repeated co-firing — high prior).  Decays/grows as the fact is
    /// re-validated or contradicted.
    pub confidence:        f32,
    /// How many times this binding fingerprint has been observed.
    pub observation_count: u32,
}

/// In-memory property-graph EEM.  Owned by [`crate::Brain`].
pub struct Eem {
    pub config:    EemConfig,
    equations:     Vec<Equation>,
    variables:     Vec<Variable>,
    disciplines:   Vec<Discipline>,
    motifs:        Vec<Motif>,
    facts:         Vec<GroundedFact>,
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
    /// Reverse index: (pool, neuron) → fact ids that include it.
    /// Enables O(1) "what facts involve this concept" lookups during
    /// chain exploration.
    fact_by_member: AHashMap<(PoolId, NeuronId), Vec<FactId>>,
    /// Source binding id → fact id (so re-emergence of the same
    /// binding doesn't create duplicate facts; instead bumps
    /// observation_count).
    fact_by_source: AHashMap<NeuronId, FactId>,
    /// Outcome-confirmed semantic pathways. The inference workspace clones
    /// these, derives transient joins, and never writes results back here.
    semantic_relations: Vec<GroundedRelation>,
    composition_rules: Vec<CompositionRule>,
    crystallizer: SemanticCrystallizer,
    /// The brain's induced vocabulary: every distinct string it has been
    /// taught as an ANSWER. Nothing here is a word list — a symbol enters
    /// only by having been the answer to a trained question, so the set is
    /// whatever the corpus taught and is empty on an untrained brain.
    induced_symbols: ahash::AHashSet<Vec<u8>>,
    /// Distinct lengths present in `induced_symbols`, descending. Searching a
    /// query for symbols by LENGTH is what keeps the producer O(query bytes)
    /// instead of O(vocabulary): iterating the set per episode measured as the
    /// only superlinear term, 526 symbols x 23,808 episodes at scale 64.
    induced_lengths: Vec<usize>,
}

impl Eem {
    pub fn new(config: EemConfig) -> Self {
        Self {
            config,
            equations:    Vec::new(),
            variables:    Vec::new(),
            disciplines:  Vec::new(),
            motifs:       Vec::new(),
            facts:        Vec::new(),
            eq_by_name:   AHashMap::new(),
            var_by_name:  AHashMap::new(),
            disc_by_name: AHashMap::new(),
            motif_by_fp:  AHashMap::new(),
            motif_links:  AHashMap::new(),
            fact_by_member: AHashMap::new(),
            fact_by_source: AHashMap::new(),
            semantic_relations: Vec::new(),
            composition_rules: Vec::new(),
            crystallizer: SemanticCrystallizer::default(),
            induced_symbols: ahash::AHashSet::new(),
            induced_lengths: Vec::new(),
        }
    }

    pub fn equation_count(&self)   -> usize { self.equations.len() }
    pub fn variable_count(&self)   -> usize { self.variables.len() }
    pub fn discipline_count(&self) -> usize { self.disciplines.len() }
    pub fn motif_count(&self)      -> usize { self.motifs.len() }
    pub fn fact_count(&self)       -> usize { self.facts.len() }
    pub fn semantic_relation_count(&self) -> usize { self.semantic_relations.len() }
    pub fn composition_rule_count(&self) -> usize { self.composition_rules.len() }
    pub fn semantic_template_count(&self) -> usize { self.crystallizer.template_count() }
    pub fn fact(&self, id: FactId) -> Option<&GroundedFact> {
        self.facts.get(id as usize)
    }
    pub fn iter_facts(&self) -> impl Iterator<Item = &GroundedFact> {
        self.facts.iter()
    }

    pub fn register_semantic_relation(&mut self, relation: GroundedRelation) {
        if let Some(existing) = self.semantic_relations.iter_mut().find(|item|
            item.predicate == relation.predicate && item.arguments == relation.arguments) {
            existing.confidence = existing.confidence.max(relation.confidence);
            existing.provenance.extend(relation.provenance);
        } else {
            self.semantic_relations.push(relation);
        }
    }

    pub fn induced_symbol_count(&self) -> usize { self.induced_symbols.len() }

    /// Predicate of every relation the training path induces, and of the
    /// premises of [`Self::INDUCED_CHAIN_RULE`].
    pub const INDUCED_PREDICATE: &'static str = "episode";
    /// Name of the one composition rule training installs.
    pub const INDUCED_CHAIN_RULE: &'static str = "episode_chain";
    /// Predicate of what that rule concludes. Distinct from the premise
    /// predicate on purpose: a conclusion that could re-match a premise makes
    /// `resolve` iterate its own output, which is `facts^2` per round for no
    /// new derivation at depth 2.
    pub const INDUCED_CHAIN_PREDICATE: &'static str = "episode_chain";
    /// Predicate of a question decomposed at TWO symbol occurrences at once.
    ///
    /// One symbol slot is not enough, and the first version of this shipped
    /// unsound because of it. With premises `episode(_, room, tail, link)` and
    /// `episode(free, link, tail2, answer)` the second premise's context is a
    /// FREE variable, so `"r001 lamp on?" -> "desk"` joined
    /// `"r003 desk material?"` and concluded r003's material for r001's lamp --
    /// a well-formed derivation of a false fact, measured as
    /// `["", "r001", " lamp on?", " material?", "cloth"]` sourced from
    /// `r001 lamp on?` and `r003 desk material?`. A two-symbol decomposition
    /// gives the rule a slot for the shared anchor, so the join pins BOTH the
    /// link and the thing it belongs to.
    pub const INDUCED_PAIR_PREDICATE: &'static str = "episode_pair";
    /// `ctx` is unmatchable context — the part of a question that is not a
    /// symbol. `sym` is a member of the induced vocabulary. Both an answer and
    /// an in-query occurrence carry `sym`, because they are the same type: a
    /// symbol is a string the brain was taught to produce. `unify` rejects a
    /// binding across differing kinds, so without that the join cannot be
    /// stated at all.
    const KIND_CTX: &'static str = "ctx";
    const KIND_SYM: &'static str = "sym";
    /// Ceiling on DURABLE induced relations. The instances are per-fact and the
    /// goal is RAM flat in corpus size, so they cannot all be kept: one 4-arity
    /// `GroundedRelation` measured ~460 B, which at 11,904 facts is 5.5 MB —
    /// 14 % of a 39.5 MB scale-64 peak, and a previous pass already spent
    /// 5.3 MB here for 0 integration. Past the cap the vocabulary and the rule
    /// still grow (both O(1) in facts) and an instance is rebuilt per query via
    /// [`Self::compose_with_transient`], which is what `workspace.rs`'s own doc
    /// comment says the design is.
    pub const MAX_INDUCED_RELATIONS: usize = 2048;

    /// Feed the composition engine from one training episode.
    ///
    /// # Why a question has to be taken apart first
    ///
    /// `TransientWorkspace` joins by binding a typed variable to a whole
    /// argument and comparing with `==` (`workspace.rs::unify`). The join
    /// integration needs over a string world is not that: `"r003 lamp on?"
    /// -> "desk"` meets `"r003 desk material?" -> "oak"` at the string
    /// `desk`, which is the ANSWER of the first and sits INSIDE the QUERY of
    /// the second. Registered as `(query, answer)` pairs those two relations
    /// share no argument, so `compose_transient` derives nothing from them
    /// under any rule — which is exactly what
    /// `tests/composition_inputs_census.rs` asserts.
    ///
    /// So the query is split at every occurrence of an induced symbol into
    /// `(before, symbol, after, answer)`, putting the join key in its own slot.
    /// The vocabulary is learned: a symbol is a string this brain was taught to
    /// ANSWER. There is no delimiter, no token list and no segmentation rule,
    /// and on a brain taught nothing the producer emits nothing.
    ///
    /// Returns how many relations this episode added.
    pub fn induce_from_episode(&mut self, query: &[u8], answer: &[u8]) -> usize {
        if answer.is_empty() {
            return 0;
        }
        self.install_induced_chain_rule();
        // The answer joins the vocabulary FIRST, so a corpus that teaches the
        // same string as an answer and then uses it inside a later question is
        // decomposed on the second episode rather than never.
        if self.induced_symbols.insert(answer.to_vec()) {
            let len = answer.len();
            if let Err(at) = self.induced_lengths.binary_search_by(|probe| len.cmp(probe)) {
                self.induced_lengths.insert(at, len);
            }
        }
        if self.semantic_relations.len() >= Self::MAX_INDUCED_RELATIONS {
            return 0;
        }
        let provenance = String::from_utf8_lossy(query).into_owned();
        let answer_value = TypedValue::new(Self::KIND_SYM, String::from_utf8_lossy(answer));
        let text = |range: std::ops::Range<usize>| {
            TypedValue::new(Self::KIND_CTX, String::from_utf8_lossy(&query[range]))
        };
        let symbol = |range: std::ops::Range<usize>| {
            TypedValue::new(Self::KIND_SYM, String::from_utf8_lossy(&query[range]))
        };
        let occurrences = self.induced_occurrences(query);
        let mut added = 0;
        let mut relations = Vec::with_capacity(occurrences.len() * 2);
        for &(start, len) in &occurrences {
            relations.push((
                Self::INDUCED_PREDICATE,
                vec![
                    text(0..start),
                    symbol(start..start + len),
                    text(start + len..query.len()),
                    answer_value.clone(),
                ],
            ));
        }
        // Every ordered pair of NON-OVERLAPPING occurrences. The anchor is the
        // earlier one and the link the later one; which is which is decided by
        // position, never by what either string means.
        for &(first, first_len) in &occurrences {
            for &(second, second_len) in &occurrences {
                if second < first + first_len {
                    continue;
                }
                relations.push((
                    Self::INDUCED_PAIR_PREDICATE,
                    vec![
                        text(0..first),
                        symbol(first..first + first_len),
                        text(first + first_len..second),
                        symbol(second..second + second_len),
                        text(second + second_len..query.len()),
                        answer_value.clone(),
                    ],
                ));
            }
        }
        for (predicate, arguments) in relations {
            if self.semantic_relations.len() >= Self::MAX_INDUCED_RELATIONS {
                break;
            }
            let before_count = self.semantic_relations.len();
            self.register_semantic_relation(GroundedRelation::new(
                predicate,
                arguments,
                1.0,
                provenance.clone(),
            ));
            if self.semantic_relations.len() > before_count {
                added += 1;
            }
        }
        added
    }

    /// Every `(start, len)` at which an induced symbol occurs in `query`,
    /// longest first, skipping an occurrence that IS the whole query (a
    /// relation whose context is empty on both sides joins nothing and only
    /// spends the cap).
    fn induced_occurrences(&self, query: &[u8]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &len in &self.induced_lengths {
            if len == 0 || len > query.len() || len == query.len() {
                continue;
            }
            for start in 0..=(query.len() - len) {
                if self.induced_symbols.contains(&query[start..start + len]) {
                    out.push((start, len));
                }
            }
        }
        out
    }

    /// The one rule training installs: chain two episodes through a symbol
    /// that one of them ANSWERS and the other CONTAINS. Idempotent, and O(1) in
    /// corpus size — there is exactly one of these however much is taught.
    fn install_induced_chain_rule(&mut self) {
        if self.composition_rules.iter().any(|rule| rule.name == Self::INDUCED_CHAIN_RULE) {
            return;
        }
        let ctx = |name: &str| PatternValue::var(name, Self::KIND_CTX);
        let sym = |name: &str| PatternValue::var(name, Self::KIND_SYM);
        self.composition_rules.push(CompositionRule {
            name: Self::INDUCED_CHAIN_RULE.to_string(),
            premises: vec![
                // A question about `anchor` whose answer is `link`.
                RelationPattern::new(
                    Self::INDUCED_PREDICATE,
                    vec![ctx("before"), sym("anchor"), ctx("after"), sym("link")],
                ),
                // A question about the SAME anchor that also mentions `link`.
                // Both variables are bound by the first premise, so the join
                // pins the link AND the thing it belongs to -- which is what
                // makes the conclusion true rather than merely well formed.
                RelationPattern::new(
                    Self::INDUCED_PAIR_PREDICATE,
                    vec![
                        ctx("link_before"),
                        sym("anchor"),
                        ctx("link_middle"),
                        sym("link"),
                        ctx("link_after"),
                        sym("conclusion"),
                    ],
                ),
            ],
            conclusion: RelationPattern::new(
                Self::INDUCED_CHAIN_PREDICATE,
                vec![
                    ctx("before"),
                    sym("anchor"),
                    ctx("after"),
                    ctx("link_after"),
                    sym("conclusion"),
                ],
            ),
        });
    }

    pub fn register_composition_rule(&mut self, rule: CompositionRule) {
        if let Some(existing) = self.composition_rules.iter_mut().find(|item| item.name == rule.name) {
            *existing = rule;
        } else {
            self.composition_rules.push(rule);
        }
    }

    /// Construct and resolve a disposable workspace from consolidated EEM
    /// pathways. Derived relations are never inserted into the EEM.
    pub fn compose_transient(&self, max_rounds: usize) -> TransientWorkspace {
        let mut workspace = TransientWorkspace::new(self.semantic_relations.clone());
        workspace.resolve(&self.composition_rules, max_rounds);
        workspace
    }

    /// Outcome-confirmed structural experience. Newly crystallized relation
    /// instances become durable EEM pathways; templates learn only here.
    pub fn consolidate_semantic_frame(&mut self, frame: SemanticFrame) -> Vec<GroundedRelation> {
        let relations = self.crystallizer.observe(frame);
        for relation in relations.iter().cloned() {
            self.register_semantic_relation(relation);
        }
        relations
    }

    /// Query-time role recognition. Neither templates nor EEM relations change.
    pub fn recognize_semantic_frame(&self, frame: &SemanticFrame) -> Vec<GroundedRelation> {
        self.crystallizer.recognize(frame)
    }

    pub fn compose_with_transient(&self, relations: impl IntoIterator<Item = GroundedRelation>,
                                  max_rounds: usize) -> TransientWorkspace {
        let mut facts = self.semantic_relations.clone();
        facts.extend(relations);
        let mut workspace = TransientWorkspace::new(facts);
        workspace.resolve(&self.composition_rules, max_rounds);
        workspace
    }

    /// Register (or update) a grounded fact for a binding emergence
    /// event.  Idempotent on `source_binding` — repeat calls bump
    /// `observation_count` rather than creating duplicates.  Members
    /// are stored sorted for deterministic chain-walking.
    pub fn register_fact(
        &mut self,
        source_binding: NeuronId,
        mut members: Vec<(PoolId, NeuronId)>,
    ) -> FactId {
        members.sort();
        if let Some(&fid) = self.fact_by_source.get(&source_binding) {
            if let Some(f) = self.facts.get_mut(fid as usize) {
                f.observation_count = f.observation_count.saturating_add(1);
                f.confidence = (f.confidence + 0.02).min(1.0);
            }
            return fid;
        }
        let id = self.facts.len() as FactId;
        let fact = GroundedFact {
            id,
            source_binding,
            members: members.clone(),
            confidence: 1.0,
            observation_count: 1,
        };
        for &m in &members {
            self.fact_by_member.entry(m).or_default().push(id);
        }
        self.fact_by_source.insert(source_binding, id);
        self.facts.push(fact);
        id
    }

    /// Find every grounded fact that includes the given (pool, neuron)
    /// ref as one of its members.  This is the "Stage 3 equation
    /// matcher" — given a firing concept, find facts of the substrate's
    /// world-knowledge that are relevant.  O(1) via reverse index.
    pub fn facts_involving(&self, pool: PoolId, neuron: NeuronId) -> Vec<&GroundedFact> {
        match self.fact_by_member.get(&(pool, neuron)) {
            Some(ids) => ids.iter()
                .filter_map(|id| self.facts.get(*id as usize))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Bulk variant: union of all facts that involve ANY of the given
    /// member refs, deduplicated.  Used by `Brain::integrate_autonomous`
    /// when launching chain exploration from the firing concepts of a
    /// query.
    pub fn facts_for_concepts<I>(&self, members: I) -> Vec<&GroundedFact>
    where I: IntoIterator<Item = (PoolId, NeuronId)>
    {
        let mut seen: ahash::AHashSet<FactId> = ahash::AHashSet::new();
        let mut out = Vec::new();
        for m in members {
            if let Some(ids) = self.fact_by_member.get(&m) {
                for &id in ids {
                    if seen.insert(id) {
                        if let Some(f) = self.facts.get(id as usize) {
                            out.push(f);
                        }
                    }
                }
            }
        }
        out
    }

    /// Chain-explore the fact graph starting from `seed_members`.  At
    /// each step, find facts that involve any current member; expand
    /// to include OTHER members of those facts.  Bounds by `max_depth`
    /// (graph hops) and `max_visit` (total facts traversed) so the
    /// walk doesn't explode.
    ///
    /// Returns the set of (pool, neuron) refs reached AND the set of
    /// facts traversed along the way.  Caller can score / decode the
    /// result.  Per spec: this is the ivy-growth equation chain.
    ///
    /// Confidence of a reached member = product of `fact.confidence`
    /// along the shortest chain (recorded in `member_confidence`).
    pub fn chain_explore(
        &self,
        seed_members: &[(PoolId, NeuronId)],
        max_depth:    usize,
        max_visit:    usize,
    ) -> ChainResult {
        let mut frontier: ahash::AHashMap<(PoolId, NeuronId), f32> = ahash::AHashMap::new();
        let mut reached:  ahash::AHashMap<(PoolId, NeuronId), f32> = ahash::AHashMap::new();
        let mut visited_facts: ahash::AHashSet<FactId> = ahash::AHashSet::new();
        for m in seed_members {
            frontier.insert(*m, 1.0);
            reached.insert(*m, 1.0);
        }
        for _depth in 0..max_depth {
            if visited_facts.len() >= max_visit || frontier.is_empty() { break; }
            let mut next_frontier: ahash::AHashMap<(PoolId, NeuronId), f32> = ahash::AHashMap::new();
            for (member, src_conf) in frontier.iter() {
                if visited_facts.len() >= max_visit { break; }
                let fids = match self.fact_by_member.get(member) {
                    Some(ids) => ids.clone(),
                    None => continue,
                };
                for fid in fids {
                    if !visited_facts.insert(fid) { continue; }
                    let fact = match self.facts.get(fid as usize) {
                        Some(f) => f,
                        None => continue,
                    };
                    let new_conf = src_conf * fact.confidence;
                    for &other in &fact.members {
                        if other == *member { continue; }
                        // Keep the BEST confidence seen for any reached
                        // member.
                        let entry = next_frontier.entry(other).or_insert(0.0);
                        if new_conf > *entry { *entry = new_conf; }
                        let r_entry = reached.entry(other).or_insert(0.0);
                        if new_conf > *r_entry { *r_entry = new_conf; }
                    }
                }
            }
            frontier = next_frontier;
        }
        ChainResult {
            reached_members: reached,
            visited_facts:   visited_facts.into_iter().collect(),
        }
    }

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
            facts:       self.facts.clone(),
            semantic_relations: self.semantic_relations.clone(),
            composition_rules: self.composition_rules.clone(),
            crystallizer: self.crystallizer.clone(),
            induced_symbols: self.induced_symbols.iter().cloned().collect(),
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
        // Rebuild fact reverse indices.
        let mut fact_by_member: AHashMap<(PoolId, NeuronId), Vec<FactId>> = AHashMap::new();
        let mut fact_by_source: AHashMap<NeuronId, FactId> = AHashMap::new();
        for f in &snap.facts {
            for &m in &f.members {
                fact_by_member.entry(m).or_default().push(f.id);
            }
            fact_by_source.insert(f.source_binding, f.id);
        }
        // Rebuilt rather than serialized twice: the lengths are derived from
        // the symbols, and a snapshot that carried both could disagree.
        let mut induced_symbols: ahash::AHashSet<Vec<u8>> = ahash::AHashSet::new();
        let mut induced_lengths: Vec<usize> = Vec::new();
        for symbol in snap.induced_symbols {
            let len = symbol.len();
            if induced_symbols.insert(symbol) {
                if let Err(at) = induced_lengths.binary_search_by(|probe| len.cmp(probe)) {
                    induced_lengths.insert(at, len);
                }
            }
        }
        Self {
            config:         snap.config,
            equations:      snap.equations,
            variables:      snap.variables,
            disciplines:    snap.disciplines,
            motifs:         snap.motifs,
            facts:          snap.facts,
            semantic_relations: snap.semantic_relations,
            composition_rules: snap.composition_rules,
            crystallizer: snap.crystallizer,
            eq_by_name,
            var_by_name,
            disc_by_name,
            motif_by_fp,
            motif_links,
            fact_by_member,
            fact_by_source,
            induced_symbols,
            induced_lengths,
        }
    }
}
