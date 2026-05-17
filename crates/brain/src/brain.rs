//! The top-level Brain struct per [`ARCHITECTURE.md`] §3 and §4.
//!
//! Owns a [`Fabric`] of pools and tracks the multi-pool firing history
//! needed for binding-concept emergence (spec §4.A) and for grounding-
//! report production at integration time (spec §2 / §4.D).
//!
//! Phase 1 supplied the substrate primitives.  This file ties them
//! together into something you can `create`, `observe` into, and
//! `integrate` from, with every output carrying a [`GroundingReport`].
//! That's the answer contract.

use ahash::AHashMap;
use std::collections::VecDeque;

use crate::action::{ActionEvent, ActionId};
use crate::annealer::{Annealer, AnnealerConfig};
use crate::eem::{Eem, EemConfig};
use crate::fabric::{Fabric, FabricConfig};
use crate::grounding::{AnswerWithGrounding, ConfidenceTier, GroundingReport};
use crate::identity::{
    BrainIdentitySpec, IdentityBuildError, PoolKind, PoolPrototypeRegistry,
};
use crate::network::{
    BrainId, GossipEquation, GossipMotif, NetworkState, PeerAccuracy,
    PeerContribution,
};
use crate::neuron::{NeuronId, NeuronKind, NeuronRef, PoolId, Neuron};
use crate::pool::{AtomEncoding, BytePassthroughEncoding, Pool, PoolConfig};

/// Sorted (pool, neuron) signature of a single tick's multi-pool
/// firing.  Used to detect recurring binding patterns: when the same
/// multi-pool firing-set has been observed `binding_emergence_threshold`
/// times within the history window, the brain births a binding concept
/// in the binding pool whose members reference every neuron in the
/// signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MomentFingerprint {
    pairs: Vec<(PoolId, NeuronId)>,
}

impl MomentFingerprint {
    fn from_fabric_moment(fired: &AHashMap<PoolId, Vec<NeuronId>>) -> Option<Self> {
        let mut pairs: Vec<(PoolId, NeuronId)> = fired.iter()
            .flat_map(|(&pid, ns)| ns.iter().map(move |&nid| (pid, nid)))
            .collect();
        if pairs.is_empty() { return None; }
        let pools_represented: std::collections::HashSet<PoolId> =
            pairs.iter().map(|(p, _)| *p).collect();
        // Binding candidates require ≥2 pools — single-pool firing is
        // a within-pool concept-emergence concern, not a binding one.
        if pools_represented.len() < 2 { return None; }
        pairs.sort();
        Some(Self { pairs })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrainConfig {
    pub fabric: FabricConfig,
    /// How many distinct ticks the same multi-pool firing fingerprint
    /// must recur before a binding concept is promoted.
    pub binding_emergence_threshold: u32,
    /// Sliding window over which fingerprint recurrence is counted.
    /// Older moments age out so only "recently sustained" co-firings
    /// produce bindings.
    pub moment_history_window: usize,
    /// Per-pool config for the auto-created "binding pool" that hosts
    /// binding concepts.  Binding pool atoms are never used for sensor
    /// input — it only holds composite concepts whose members live in
    /// other pools.
    pub binding_pool_config: PoolConfig,
    /// EEM (Environmental Equation Matrix) config.  Spec §4.B.  An
    /// empty EEM is created on `Brain::new`; the caller seeds it via
    /// `brain.eem_mut().register_equation(...)`.
    pub eem: EemConfig,
    /// Temporal-prediction annealer config.  Spec §4.C.  An empty
    /// annealer is created on `Brain::new`; the brain automatically
    /// captures one frame per pool per `advance_tick`.
    pub annealer: AnnealerConfig,
}

impl Default for BrainConfig {
    fn default() -> Self {
        let mut binding_pool_config = PoolConfig::defaults("binding", 0);
        // Binding pool doesn't need atom-sequence emergence — it only
        // hosts cross-pool members.  Disable by setting the threshold
        // unreachably high.
        binding_pool_config.concept_emergence_threshold = u32::MAX;
        Self {
            fabric: FabricConfig::default(),
            binding_emergence_threshold: 3,
            moment_history_window: 64,
            binding_pool_config,
            eem: EemConfig::default(),
            annealer: AnnealerConfig::default(),
        }
    }
}

/// Stats surfaced via [`Brain::stats`].
#[derive(Debug, Clone, Default)]
pub struct BrainStats {
    pub tick:                u64,
    pub pool_count:          usize,
    pub total_neurons:       usize,
    pub total_concepts:      usize,
    pub total_binding:       usize,
    pub total_terminals:     usize,
    pub binding_pool_id:     PoolId,
    pub fingerprints_window: usize,
}

pub struct Brain {
    fabric:                       Fabric,
    config:                       BrainConfig,
    binding_pool_id:              PoolId,
    /// Fingerprint history.  Bounded by `moment_history_window`.
    moment_history:               VecDeque<MomentFingerprint>,
    /// Active-count of each fingerprint within the window.
    binding_recurrences:          AHashMap<MomentFingerprint, u32>,
    /// Fingerprints that have already been promoted, so we don't
    /// re-promote on every re-occurrence past the threshold.
    promoted_fingerprints:        AHashMap<MomentFingerprint, NeuronId>,
    /// Phase 7: action layer state.  `None` until the caller
    /// designates an action pool via `designate_action_pool`.
    action_pool_id:               Option<PoolId>,
    /// Already-emitted action events waiting on outcome feedback.
    /// `feed_outcome` looks up the action_id and reinforces (or
    /// weakens) the source→action_neuron terminals per the outcome
    /// score.  Bounded by `action_history_max` to keep memory finite.
    pending_actions:              AHashMap<ActionId, ActionEvent>,
    next_action_id:               ActionId,
    /// Tracks which action ids have already been emitted this tick to
    /// prevent firing the same action neuron more than once per tick.
    emitted_this_tick:            ahash::AHashSet<NeuronRef>,
    /// Phase 5: Environmental Equation Matrix.  Owned by the brain so
    /// integration can consult equations and report their confidence
    /// alongside the fabric's.  Caller seeds equations directly via
    /// `eem_mut().register_equation(...)`.
    eem:                          Eem,
    /// Phase 6: Temporal-prediction annealer.  Captures one frame per
    /// pool per tick; consulted by `integrate_with_prediction`.
    annealer:                     Annealer,
    /// Phase 8: distributed-network state.  Pending outbound motif/
    /// equation gossip, received peer motifs, peer accuracy track
    /// record.  Transport (cluster) lives outside this crate; this
    /// state is the brain-side data model.
    network:                      NetworkState,
}

impl Brain {
    /// Construct a fresh brain with no sensor pools yet.  The binding
    /// pool is auto-created at pool_id = `binding_pool_config.id`.
    /// Sensor pools get created by the caller via `create_pool`.
    pub fn new(config: BrainConfig) -> Self {
        let mut fabric = Fabric::new(config.fabric.clone());
        let binding_pool_id = config.binding_pool_config.id;
        let binding_pool = Pool::new(
            config.binding_pool_config.clone(),
            Box::new(BytePassthroughEncoding { prefix: "bind" }),
        );
        fabric.register_pool(binding_pool);
        let window = config.moment_history_window;
        let eem = Eem::new(config.eem.clone());
        let annealer = Annealer::new(config.annealer.clone());
        Self {
            fabric,
            config,
            binding_pool_id,
            moment_history:        VecDeque::with_capacity(window),
            binding_recurrences:   AHashMap::new(),
            promoted_fingerprints: AHashMap::new(),
            action_pool_id:        None,
            pending_actions:       AHashMap::new(),
            next_action_id:        1,
            emitted_this_tick:     ahash::AHashSet::new(),
            eem,
            annealer,
            network:               NetworkState::new(""),
        }
    }

    pub fn eem(&self) -> &Eem { &self.eem }
    pub fn eem_mut(&mut self) -> &mut Eem { &mut self.eem }
    pub fn annealer(&self) -> &Annealer { &self.annealer }
    pub fn annealer_mut(&mut self) -> &mut Annealer { &mut self.annealer }

    pub fn fabric(&self) -> &Fabric { &self.fabric }
    pub fn fabric_mut(&mut self) -> &mut Fabric { &mut self.fabric }
    pub fn binding_pool_id(&self) -> PoolId { self.binding_pool_id }

    /// Register a sensor pool.  The caller supplies the atomization
    /// contract via [`AtomEncoding`].  Returns the assigned pool id.
    pub fn create_pool(
        &mut self,
        config:   PoolConfig,
        encoding: Box<dyn AtomEncoding>,
    ) -> PoolId {
        let pool = Pool::new(config, encoding);
        self.fabric.register_pool(pool)
    }

    /// Observe a sensor frame into `pool_id`.  Fires atoms; returns
    /// the IDs that fired.  Multi-pool observation in the same tick
    /// is the normal mode — call `advance_tick` only when ready to
    /// close the moment.
    pub fn observe(&mut self, pool_id: PoolId, frame: &[u8]) -> Vec<NeuronId> {
        self.fabric.observe(pool_id, frame)
    }

    /// Close the current tick.  Performs:
    /// 1. Cross-pool axon wiring for any co-fired pairs (Fabric does this).
    /// 2. Binding-concept emergence: if the current moment's multi-pool
    ///    firing fingerprint has recurred ≥ threshold times within the
    ///    history window, promote it.
    /// 3. Per-pool housekeeping (decay + prune).
    pub fn advance_tick(&mut self) {
        // Snapshot the current moment's firing BEFORE the fabric
        // advances (which clears it).  Build a fingerprint from
        // multi-pool firing for binding-concept tracking.
        let fingerprint = {
            let moment = self.fabric.current_moment();
            MomentFingerprint::from_fabric_moment(&moment.fired)
        };

        // Capture per-pool activation frames into the annealer's
        // history.  One frame per pool per tick; empty frames are
        // skipped by Annealer::record_frame.  Done BEFORE
        // advance_tick because the fabric's housekeeping doesn't
        // clear pool activation, but the next tick's observes will.
        for pid in self.fabric.pool_ids() {
            if let Some(pool) = self.fabric.pool(pid) {
                let p = pool.read();
                let frame: AHashMap<NeuronId, f32> = p.currently_firing()
                    .map(|nid| (nid, p.activation(nid)))
                    .collect();
                drop(p);
                if !frame.is_empty() {
                    self.annealer.record_frame(pid, frame);
                }
            }
        }

        self.fabric.advance_tick();
        self.emitted_this_tick.clear();

        if let Some(fp) = fingerprint {
            self.register_fingerprint(fp);
        }
    }

    fn register_fingerprint(&mut self, fp: MomentFingerprint) {
        // Decay the oldest entry out of the window.
        if self.moment_history.len() >= self.config.moment_history_window {
            if let Some(old) = self.moment_history.pop_front() {
                if let Some(c) = self.binding_recurrences.get_mut(&old) {
                    *c = c.saturating_sub(1);
                    if *c == 0 { self.binding_recurrences.remove(&old); }
                }
            }
        }
        self.moment_history.push_back(fp.clone());
        let count = self.binding_recurrences.entry(fp.clone()).or_insert(0);
        *count = count.saturating_add(1);

        let recurrence = *count;
        if recurrence >= self.config.binding_emergence_threshold
            && !self.promoted_fingerprints.contains_key(&fp)
        {
            let new_id = self.promote_binding_concept(&fp);
            if let Some(id) = new_id {
                // Auto-emit a gossip record for the just-promoted
                // binding (spec §5.1: motif fingerprints are gossiped
                // on discovery).  Transport drains via
                // `drain_motif_gossip`.
                self.network.pending_motif_out.push(GossipMotif {
                    source_brain:      self.network.brain_id.clone(),
                    fingerprint:       fp.pairs.clone(),
                    observation_count: recurrence,
                    local_confidence:  (recurrence as f32
                        / (self.config.binding_emergence_threshold.max(1) as f32))
                        .min(1.0),
                    observed_at_tick:  self.fabric.current_tick(),
                });
                self.promoted_fingerprints.insert(fp, id);
            }
        }
    }

    fn promote_binding_concept(&mut self, fp: &MomentFingerprint) -> Option<NeuronId> {
        let binding_pool = self.fabric.pool(self.binding_pool_id)?;
        let mut binding = binding_pool.write();
        // Composite label = sorted member references, joined.  Stable
        // and unique per fingerprint.
        let label: String = fp.pairs.iter()
            .map(|(p, n)| format!("p{}n{}", p, n))
            .collect::<Vec<_>>()
            .join("|");
        if binding.label_to_id(&label).is_some() {
            return None;  // already exists, idempotent.
        }
        let members: Vec<NeuronRef> = fp.pairs.iter()
            .map(|(p, n)| NeuronRef::new(*p, *n))
            .collect();
        let max_w = binding.config.max_weight;
        let now = self.fabric.current_tick();
        // Create the binding concept directly with cross-pool members.
        let id = binding.neuron_count() as NeuronId;
        let mut neuron = Neuron::new_concept(
            id, label.clone(), NeuronKind::Excitatory, members.clone(), now,
        );
        // Wire concept → member terminals top-down so activating the
        // binding fires its constituent neurons in all pools.
        for m in &members {
            neuron.reinforce_terminal(*m, 0.5, now, max_w);
        }
        binding.append_neuron(neuron, label.clone());
        drop(binding);

        // Wire member → binding terminals bottom-up so co-firing all
        // members activates the binding.
        let binding_ref = NeuronRef::new(self.binding_pool_id, id);
        for m in &members {
            if let Some(p) = self.fabric.pool(m.pool) {
                let mut pp = p.write();
                let mxw = pp.config.max_weight;
                if let Some(n) = pp.get_mut(m.neuron) {
                    n.reinforce_terminal(binding_ref, 0.5, now, mxw);
                }
            }
        }
        Some(id)
    }

    /// Raw read of a pool's current activation map.  No interpretation,
    /// no grounding — just the substrate state.  Spec §1.6.
    pub fn read_activation(&self, pool_id: PoolId) -> AHashMap<NeuronId, f32> {
        let pool = match self.fabric.pool(pool_id) {
            Some(p) => p,
            None    => return AHashMap::new(),
        };
        let p = pool.read();
        let mut out = AHashMap::new();
        for nid in p.currently_firing() {
            out.insert(nid, p.activation(nid));
        }
        out
    }

    /// Produce an [`AnswerWithGrounding`] by propagating from
    /// `query_pool`'s currently-firing state and reading the resulting
    /// activation in `target_pool`.  Every output carries a
    /// [`GroundingReport`] per spec §2.7.
    ///
    /// The brain doesn't decide whether to surface uncertainty —
    /// uncertainty is always surfaced.  Caller reads the report.
    pub fn integrate(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
    ) -> AnswerWithGrounding {
        // 1. Propagate from query pool across the fabric.
        let propagated = self.fabric.propagate(query_pool);

        // 2. Input atom coverage in the query pool: how many of the
        // currently-firing neurons there map to known atoms.  At this
        // phase every fired neuron is by construction in `label_to_id`,
        // so coverage is 1.0 if anything fired and 0 otherwise.  Once
        // sensor adapters route unknown bytes through atom birth this
        // remains 1.0 for known input, with adaptive plasticity per
        // spec §2.5 handling the novelty signal at the concept layer.
        let query_pool_handle = match self.fabric.pool(query_pool) {
            Some(p) => p,
            None    => return AnswerWithGrounding::unknown(
                format!("unknown pool id {}", query_pool), target_pool),
        };
        let fired_in_query = query_pool_handle.read()
            .currently_firing().collect::<Vec<_>>();
        if fired_in_query.is_empty() {
            return AnswerWithGrounding::unknown(
                "no input observed in query pool — cannot integrate", target_pool);
        }
        let input_atom_coverage = 1.0_f32;

        // 3. Read target pool activation from propagation results.
        let empty: AHashMap<NeuronId, f32> = AHashMap::new();
        let target_activation = propagated.get(&target_pool).unwrap_or(&empty);

        // 4. Compute Jaccard between the strongest target concept's
        // member-atom set (within the same target pool only) and the
        // input atoms.  Internal use only — spec §1.6 says no Jaccard
        // ranker exposed at the API surface; here it's a measurement
        // for the grounding report, not a selection mechanism.
        let target_pool_handle = match self.fabric.pool(target_pool) {
            Some(p) => p,
            None    => return AnswerWithGrounding::unknown(
                format!("unknown target pool id {}", target_pool), target_pool),
        };
        let target = target_pool_handle.read();

        // Stage 3b (selection-side): build a set of target-pool
        // concepts that are DIRECTLY targeted by a query-pool concept
        // via Stage 3's concept→concept cross-pool terminal.  These
        // are the specific binding partners the substrate identified
        // during training; they get a score boost in selection so the
        // density-based picker doesn't get drowned out by short,
        // shared atom-level concepts ("00" beating "C000" etc.).
        let directly_targeted: ahash::AHashSet<NeuronId> = {
            let q = query_pool_handle.read();
            let mut set = ahash::AHashSet::new();
            for nid in q.currently_firing() {
                if let Some(qn) = q.get(nid) {
                    if qn.is_atom() { continue; }
                    for t in &qn.terminals {
                        if t.target.pool != target_pool { continue; }
                        if let Some(tn) = target.get(t.target.neuron) {
                            if !tn.is_atom() {
                                set.insert(t.target.neuron);
                            }
                        }
                    }
                }
            }
            set
        };

        // Selection (Stage 6 — coverage-aware density):
        //
        //   score = avg_member_activation
        //         × pathway_boost
        //         × sqrt(direct_member_count)
        //         × unique_ratio
        //
        // where:
        //   avg_member_activation = mean of target_activation across the
        //                           concept's in-pool member neurons.
        //                           High when ALL members of the concept
        //                           are firing strongly — i.e., the
        //                           concept's full pattern is present.
        //   sqrt(direct_member_count) — rewards longer concepts when
        //                           coverage is good (so "color" wins
        //                           over "co" when all of c,o,l,r are
        //                           active), without letting concepts
        //                           with 20+ repetitions dominate (sqrt
        //                           grows slowly).
        //   unique_ratio = unique_member_ids / direct_member_count.
        //                  Penalizes concepts with REPEATED members —
        //                  the long-pattern concepts from Experiment A
        //                  (4× "x+y}" pattern) have unique_ratio 0.25
        //                  while clean concepts have 1.0.
        //   pathway_boost — same 4× factor as before for concepts that
        //                  a query-pool concept directly targets via
        //                  Stage 3 concept→concept cross-pool wiring.
        //
        // This replaces the earlier `(act × boost) / expanded_size`
        // density rule, which favored short concepts so aggressively
        // that trained full-word responses lost to 2-byte prefix
        // concepts during chat retrieval.
        let pathway_boost = 4.0_f32;
        let mut strongest: Option<(NeuronRef, f32)> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;
        let mut all_active_concepts: Vec<NeuronRef> = Vec::new();
        for (&nid, &act) in target_activation.iter() {
            if act < 0.001 { continue; }
            if let Some(n) = target.get(nid) {
                if !n.is_atom() {
                    let r = NeuronRef::new(target_pool, nid);
                    all_active_concepts.push(r);

                    let in_pool_members: Vec<NeuronId> = n.members.iter()
                        .filter(|m| m.pool == target_pool)
                        .map(|m| m.neuron)
                        .collect();
                    let member_count = in_pool_members.len().max(1);
                    let unique_count: usize = in_pool_members.iter()
                        .collect::<ahash::AHashSet<_>>().len().max(1);
                    let unique_ratio = unique_count as f32 / member_count as f32;
                    let member_activation_sum: f32 = in_pool_members.iter()
                        .map(|nid| target_activation.get(nid).copied().unwrap_or(0.0))
                        .sum();
                    let avg_member_act = member_activation_sum / member_count as f32;

                    let boost = if directly_targeted.contains(&nid) { pathway_boost } else { 1.0 };
                    let length_factor = (member_count as f32).sqrt();
                    let score = avg_member_act * boost * length_factor * unique_ratio;
                    // Hold on to raw activation for grounding metric.
                    let _ = act;
                    let _ = target.expanded_size(&n.members);
                    if score > best_score {
                        best_score = score;
                        strongest = Some((r, act));
                    }
                }
            }
        }

        // Outside-grounding detection: nothing in the target pool was
        // sufficiently activated.  Spec §2.2.
        let (strongest_match, strongest_activation) = match strongest {
            Some(x) => (Some(x.0), x.1),
            None    => {
                return AnswerWithGrounding {
                    answer: None,
                    grounding: GroundingReport {
                        input_atom_coverage,
                        strongest_match: None,
                        strongest_match_jaccard: 0.0,
                        composition_used: all_active_concepts,
                        fabric_confidence: 0.0,
                        eem_confidence: None,
                        annealer_confidence: None,
                        integrated_confidence: 0.0,
                        outside_grounding: true,
                        speculation_flag: false,
                        peer_contributions: Vec::new(),
                    },
                    confidence_tier: ConfidenceTier::Ungrounded,
                    next_steps_if_ungrounded: vec![
                        crate::grounding::RequestObservation {
                            domain: "target pool has no activated concept for this input".into(),
                            examples_needed: 1,
                            why: "extend grounding by co-observing query and target streams together".into(),
                            pool: target_pool,
                        },
                    ],
                };
            }
        };

        // Jaccard against the strongest match's member set (for the
        // grounding report only).  Pure measurement, not a gate.
        let strongest_jaccard = if let Some(r) = strongest_match {
            let input_set: std::collections::HashSet<NeuronId> = fired_in_query
                .iter().copied().collect();
            // Member NeuronRefs may live in any pool.  We compare only
            // those whose pool == query_pool — atoms in the target
            // concept that originate from the query stream.
            if let Some(n) = target.get(r.neuron) {
                let concept_in_query: std::collections::HashSet<NeuronId> = n.members.iter()
                    .filter(|m| m.pool == query_pool)
                    .map(|m| m.neuron)
                    .collect();
                if input_set.is_empty() && concept_in_query.is_empty() {
                    0.0
                } else {
                    let inter = input_set.intersection(&concept_in_query).count() as f32;
                    let union = input_set.union(&concept_in_query).count().max(1) as f32;
                    inter / union
                }
            } else { 0.0 }
        } else { 0.0 };

        // Speculation flag: when the strongest match's members span
        // multiple pools (binding concept) OR when the answer arrived
        // via cross-pool propagation rather than direct input retrieval.
        // The latter is detected here as "query_pool != target_pool":
        // the activation reaching target_pool came through axon
        // terminals, which is compositional by definition.
        let speculation_flag = query_pool != target_pool
            || strongest_match.map(|r| {
                target.get(r.neuron)
                    .map(|n| n.is_binding(target_pool))
                    .unwrap_or(false)
            }).unwrap_or(false);

        // Fabric confidence = fraction of propagated activation
        // captured by the strongest match (its share of total
        // concept-level activation in the target pool).
        let total_concept_act: f32 = target_activation.iter()
            .filter(|&(&nid, _)| target.get(nid).map_or(false, |n| !n.is_atom()))
            .map(|(_, &a)| a).sum();
        let fabric_confidence = if total_concept_act > 1e-9 {
            (strongest_activation / total_concept_act).clamp(0.0, 1.0)
        } else { 0.0 };

        // Integrated confidence = fabric_confidence for now (EEM and
        // annealer are not online until phases 5–6; their contribution
        // weights stay implicitly zero).  Spec §4.D will combine all
        // three once they're built.
        let integrated_confidence = fabric_confidence;

        // Decode the strongest match's members back through the target
        // pool's encode/decode contract.  Uses the recursive walker so
        // that a concept-of-concepts gets decoded all the way down to
        // its atom leaves.  Spec §3.4.
        let answer = strongest_match.and_then(|r| {
            target.get(r.neuron).map(|n| {
                target.decode_concept_members(&n.members)
            })
        });

        let outside_grounding = fabric_confidence < 0.01;
        let confidence_tier = ConfidenceTier::from_confidence(
            integrated_confidence, outside_grounding, speculation_flag);

        AnswerWithGrounding {
            answer,
            grounding: GroundingReport {
                input_atom_coverage,
                strongest_match,
                strongest_match_jaccard: strongest_jaccard,
                composition_used: all_active_concepts,
                fabric_confidence,
                eem_confidence: None,
                annealer_confidence: None,
                integrated_confidence,
                outside_grounding,
                speculation_flag,
                peer_contributions: Vec::new(),
            },
            confidence_tier,
            next_steps_if_ungrounded: Vec::new(),
        }
    }

    // ---------------------------------------------------------------
    // Phase 5 — EEM-augmented integration (spec §4.D + §4.B).
    //
    // `integrate` returns the fabric's view with `eem_confidence:
    // None` because no equation was consulted.  When the caller has
    // an equation that should apply to the current context, they
    // call `integrate_with_equation(query, target, eq, bindings)`
    // and the EEM's confidence enters the grounding report.  The
    // integrated_confidence becomes the geometric mean of fabric and
    // EEM confidences — both must be grounded for the integrated
    // answer to be strong, per spec §2.7.
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // Stage 2 — Sequential generation (continuation of spec §4.E).
    //
    // `integrate` returns the strongest concept's decoded bytes for
    // ONE step.  `generate` chains those steps into a sequence: emit
    // a chunk, feed it back as input via observe + advance_tick, find
    // the next non-emitted strongest concept in target_pool, append.
    // Termination is on confidence floor, max step count, or when no
    // unfired concept clears the floor.
    //
    // This is what turns the brain from a recognizer into a speaker.
    // Generation is recurrent retrieval — each emitted chunk becomes
    // the next step's context.  Open-ended novel composition is NOT
    // here; that requires phrase-level concepts (level-3+) trained on
    // enough natural language to have emerged the relevant
    // compositions.  What IS here: the substrate emits whatever
    // sequence its learned concept structure supports.
    // ---------------------------------------------------------------

    /// Generate a sequence of bytes by iterating
    /// integrate-then-observe-back.  Each step:
    ///   1. Propagate from the most-recent firing state (query pool at
    ///      step 0, target pool thereafter).
    ///   2. Consult the temporal annealer (`Annealer::predict_next`)
    ///      for `target_pool` — its predicted-next-frame activations
    ///      are blended additively into each candidate's score so the
    ///      pick is influenced by "what historically follows the
    ///      current state," not just propagation strength alone.  When
    ///      the annealer has <2 history frames the prediction is
    ///      absent and selection falls back to pure density (Stage 2).
    ///   3. Score active target-pool concepts by
    ///         `(act + annealer_weight × pred_act) / expanded_size`
    ///      skipping concepts already emitted in this call.
    ///   4. Decode the winner via the recursive walker and append to
    ///      the output.
    ///   5. Observe the decoded chunk into target_pool so its atoms
    ///      fire and any matching concepts collapse (mini-column
    ///      collapse runs the same way it does for sensor input);
    ///      then `advance_tick` to capture the state into history.
    ///
    /// `max_steps` is a hard cap; `min_confidence` is the activation
    /// floor (concepts below it don't qualify, ending generation).
    /// The set of already-emitted concepts is reset per call.
    ///
    /// Caller is responsible for observing the prompt into `query_pool`
    /// before calling — `generate` itself does NOT take a prompt
    /// argument because the substrate's "prompt" IS the current pool
    /// state, not a fresh input.  This matches how observation works
    /// elsewhere in the API.
    pub fn generate(
        &mut self,
        query_pool:     PoolId,
        target_pool:    PoolId,
        max_steps:      usize,
        min_confidence: f32,
    ) -> Vec<u8> {
        self.generate_weighted(query_pool, target_pool, max_steps, min_confidence, 0.5)
    }

    /// Like [`Brain::generate`] but with an explicit `annealer_weight`
    /// controlling how much the temporal-prediction signal influences
    /// next-concept selection.  0.0 = pure density (Stage 2);
    /// 0.5 = balanced (default); 1.0 = predicted-activation as
    /// important as propagation-activation.
    pub fn generate_weighted(
        &mut self,
        query_pool:      PoolId,
        target_pool:     PoolId,
        max_steps:       usize,
        min_confidence:  f32,
        annealer_weight: f32,
    ) -> Vec<u8> {
        let mut output: Vec<u8> = Vec::new();
        let mut already_emitted: ahash::AHashSet<NeuronRef> = ahash::AHashSet::new();

        for step in 0..max_steps {
            let source = if step == 0 { query_pool } else { target_pool };
            let propagated = self.fabric.propagate(source);

            // Stage 3: ask the annealer what the next frame should
            // look like given the target pool's recent history.  None
            // when there's <2 history frames; selection then falls
            // back to pure-density (Stage 2).
            let prediction = self.annealer.predict_next(target_pool);
            let pred_frame = prediction.as_ref().map(|p| &p.predicted_frame);

            // Find the best non-emitted concept in target_pool.
            let chunk: Option<(NeuronRef, Vec<u8>)> = {
                let empty: AHashMap<NeuronId, f32> = AHashMap::new();
                let target_act = propagated.get(&target_pool).unwrap_or(&empty);
                let pool_arc = match self.fabric.pool(target_pool) {
                    Some(p) => p,
                    None => break,
                };
                let p = pool_arc.read();

                let mut best: Option<(NeuronRef, f32)> = None;
                let mut best_score = f32::NEG_INFINITY;
                for (&nid, &act) in target_act.iter() {
                    if act < min_confidence { continue; }
                    let n = match p.get(nid) {
                        Some(n) => n,
                        None => continue,
                    };
                    if n.is_atom() { continue; }
                    let r = NeuronRef::new(target_pool, nid);
                    if already_emitted.contains(&r) { continue; }
                    let size = p.expanded_size(&n.members).max(1);
                    let pred_act = pred_frame
                        .and_then(|pf| pf.get(&nid).copied())
                        .unwrap_or(0.0);
                    let blended = act + annealer_weight * pred_act;
                    let score = blended / size as f32;
                    if score > best_score {
                        best_score = score;
                        best = Some((r, act));
                    }
                }
                best.and_then(|(r, _)| p.get(r.neuron).map(|n| (r, p.decode_concept_members(&n.members))))
            };

            let (r, bytes) = match chunk {
                Some(x) => x,
                None => break,
            };
            if bytes.is_empty() { break; }

            output.extend_from_slice(&bytes);
            already_emitted.insert(r);

            // Feed the chunk back into target_pool — its atoms fire,
            // its concepts collapse, the annealer captures the frame.
            self.observe(target_pool, &bytes);
            self.advance_tick();
        }

        output
    }

    /// Like [`Brain::integrate`] but additionally consults the
    /// temporal-prediction annealer for the target pool: the
    /// annealer's convergence energy enters `annealer_confidence` in
    /// the grounding report, and `integrated_confidence` becomes the
    /// geometric mean of fabric and annealer confidences.  When the
    /// annealer has fewer than 2 history frames for the target pool,
    /// `annealer_confidence` stays `None`.
    pub fn integrate_with_prediction(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
    ) -> AnswerWithGrounding {
        let mut base = self.integrate(query_pool, target_pool);
        if let Some(pred) = self.annealer.predict_next(target_pool) {
            base.grounding.annealer_confidence = Some(pred.confidence);
            let f = base.grounding.fabric_confidence;
            let a = pred.confidence;
            base.grounding.integrated_confidence = (f * a).sqrt();
            base.confidence_tier = ConfidenceTier::from_confidence(
                base.grounding.integrated_confidence,
                base.grounding.outside_grounding,
                base.grounding.speculation_flag,
            );
        }
        base
    }

    /// Like [`Brain::integrate`] but additionally consults the EEM:
    /// the named equation is evaluated under `bindings` and its
    /// confidence enters the grounding report.  If the equation is
    /// not registered or any binding is missing, `eem_confidence`
    /// stays `None` (honestly: the EEM could not contribute).
    pub fn integrate_with_equation(
        &self,
        query_pool:    PoolId,
        target_pool:   PoolId,
        equation_name: &str,
        bindings:      &AHashMap<&str, f64>,
    ) -> AnswerWithGrounding {
        let mut base = self.integrate(query_pool, target_pool);
        let application = self.eem.apply_by_name(equation_name, bindings);
        if let Some(app) = application {
            base.grounding.eem_confidence = Some(app.confidence);
            // Combine fabric and EEM confidences as a geometric mean
            // when both are present.  Lifting one without the other
            // would mask uncertainty.
            let f = base.grounding.fabric_confidence;
            let e = app.confidence;
            base.grounding.integrated_confidence = (f * e).sqrt();
            base.confidence_tier = ConfidenceTier::from_confidence(
                base.grounding.integrated_confidence,
                base.grounding.outside_grounding,
                base.grounding.speculation_flag,
            );
        }
        base
    }

    // ---------------------------------------------------------------
    // Phase 7 — Action layer (spec §4.E).
    //
    // Designating an action pool makes that pool's atoms act as action
    // neurons: when they fire (typically through cross-pool propagation
    // from sensor input), `take_action` collects them into
    // `ActionEvent`s with source attribution.  The caller routes the
    // events externally and later calls `feed_outcome` with a score
    // that reinforces or weakens the source→action terminals.
    // ---------------------------------------------------------------

    /// Designate which pool hosts action atoms.  Subsequent atom births
    /// in that pool (via `register_action`) create action neurons.  At
    /// most one action pool per brain; calling again replaces the
    /// designation.
    pub fn designate_action_pool(&mut self, pool_id: PoolId) {
        self.action_pool_id = Some(pool_id);
    }

    pub fn action_pool_id(&self) -> Option<PoolId> {
        self.action_pool_id
    }

    /// Create (or look up) an action atom in the designated action
    /// pool.  Returns the neuron id of the action atom — callers store
    /// this and pair it with their external effect (webhook URL, MQTT
    /// topic, function call name, etc.).
    ///
    /// Panics if no action pool has been designated, or if the
    /// designated pool id is unknown.  Action registration is a
    /// configuration step; getting it wrong is a programmer error,
    /// not a runtime condition.
    pub fn register_action(&mut self, label: String) -> NeuronRef {
        let pool_id = self.action_pool_id
            .expect("designate_action_pool must be called before register_action");
        let pool = self.fabric.pool(pool_id)
            .expect("action pool id is not a registered pool");
        let mut p = pool.write();
        if let Some(existing) = p.label_to_id(&label) {
            return NeuronRef::new(pool_id, existing);
        }
        let now = self.fabric.current_tick();
        let id = p.neuron_count() as NeuronId;
        let neuron = Neuron::new_atom(id, label.clone(), NeuronKind::Excitatory, now);
        p.append_neuron(neuron, label);
        NeuronRef::new(pool_id, id)
    }

    /// Manually fire an action neuron (e.g. during supervised training
    /// where the trainer wants to drive the action while sources are
    /// also firing).  This deposits activation into the action pool;
    /// `take_action` is what surfaces the corresponding event.
    pub fn fire_action(&mut self, action: NeuronRef, strength: f32) {
        if let Some(p) = self.fabric.pool(action.pool) {
            let tick = self.fabric.current_tick();
            p.write().inject_activation(action.neuron, strength, tick);
        }
        // Register in the moment buffer so cross-pool wiring on tick
        // close grows source→action terminals.
        self.fabric.mark_fired(action.pool, action.neuron);
    }

    /// Collect every action neuron that fired this tick, attribute its
    /// sources (currently-firing neurons in other pools whose terminals
    /// target the action neuron), and emit one `ActionEvent` per
    /// firing.  Caller routes externally and later supplies
    /// `feed_outcome`.
    ///
    /// "Fired" means: present in the action pool's `currently_firing`
    /// set (from `fire_action`), OR reached via cross-pool propagation
    /// from any non-action pool with activation ≥ `min_activation`.
    /// The propagation path is what gives the substrate its agency:
    /// sensor input → cross-pool axons → action firing → external
    /// effect → outcome → reinforcement.
    ///
    /// Re-fires of the same action neuron within a single tick are
    /// deduplicated.  Across ticks, each firing produces its own event.
    pub fn take_action(&mut self, min_activation: f32) -> Vec<ActionEvent> {
        let pool_id = match self.action_pool_id {
            Some(p) => p,
            None    => return Vec::new(),
        };
        let pool = match self.fabric.pool(pool_id) {
            Some(p) => p,
            None    => return Vec::new(),
        };

        // Step 1: collect direct firings (from `fire_action` or
        // observed atoms in the action pool itself).
        let mut firings: AHashMap<NeuronId, (String, f32)> = AHashMap::new();
        {
            let p = pool.read();
            for nid in p.currently_firing() {
                if let Some(n) = p.get(nid) {
                    firings.insert(nid, (n.label.clone(), p.activation(nid)));
                }
            }
        }

        // Step 2: propagate from each non-action pool that has firing
        // neurons; merge activation reaching the action pool.
        for src_pool_id in self.fabric.pool_ids() {
            if src_pool_id == pool_id { continue; }
            let any_firing = match self.fabric.pool(src_pool_id) {
                Some(p) => p.read().currently_firing().next().is_some(),
                None    => false,
            };
            if !any_firing { continue; }
            let propagated = self.fabric.propagate(src_pool_id);
            if let Some(act_map) = propagated.get(&pool_id) {
                let p = pool.read();
                for (&nid, &act) in act_map.iter() {
                    if act < min_activation { continue; }
                    if let Some(n) = p.get(nid) {
                        let entry = firings.entry(nid)
                            .or_insert_with(|| (n.label.clone(), 0.0));
                        entry.1 = entry.1.max(act);
                    }
                }
            }
        }

        if firings.is_empty() { return Vec::new(); }

        let tick = self.fabric.current_tick();
        let mut events = Vec::with_capacity(firings.len());

        // Step 3: for each fired action neuron, find currently-firing
        // sources elsewhere in the fabric whose terminals target it.
        // Credit attribution by inverse-terminal lookup — there's no
        // fabric-level routing table, so we scan owning neurons.
        for (action_nid, (action_label, activation)) in firings {
            let action_ref = NeuronRef::new(pool_id, action_nid);
            if self.emitted_this_tick.contains(&action_ref) { continue; }

            let mut sources: Vec<NeuronRef> = Vec::new();
            for src_pool_id in self.fabric.pool_ids() {
                if src_pool_id == pool_id { continue; }
                let src_pool = match self.fabric.pool(src_pool_id) {
                    Some(p) => p,
                    None    => continue,
                };
                let sp = src_pool.read();
                let firing_here: ahash::AHashSet<NeuronId> = sp.currently_firing().collect();
                for nid in firing_here.iter().copied() {
                    if let Some(n) = sp.get(nid) {
                        if n.terminals.iter().any(|t| t.target == action_ref) {
                            sources.push(NeuronRef::new(src_pool_id, nid));
                        }
                    }
                }
            }

            let id = self.next_action_id;
            self.next_action_id += 1;
            let event = ActionEvent {
                id,
                action_neuron: action_ref,
                action_label,
                sources,
                fired_at_tick: tick,
                activation,
            };
            self.pending_actions.insert(id, event.clone());
            self.emitted_this_tick.insert(action_ref);
            events.push(event);
        }

        events
    }

    /// Apply an outcome score to a previously-emitted action.  Positive
    /// scores reinforce source→action terminals; negative scores
    /// weaken them.  The action_id is removed from `pending_actions`
    /// after application — outcomes are one-shot per firing.
    ///
    /// Returns true if the action_id was found and feedback applied.
    /// Returns false if the id is unknown (already-fed-back or never-
    /// emitted).
    pub fn feed_outcome(&mut self, action_id: ActionId, score: f32) -> bool {
        let event = match self.pending_actions.remove(&action_id) {
            Some(e) => e,
            None    => return false,
        };
        let now = self.fabric.current_tick();
        for src in &event.sources {
            let src_pool = match self.fabric.pool(src.pool) {
                Some(p) => p,
                None    => continue,
            };
            let mut sp = src_pool.write();
            let max_w = sp.config.max_weight;
            if let Some(n) = sp.get_mut(src.neuron) {
                // Positive: strengthen via the same idempotent
                // reinforce path.  Negative: directly subtract from
                // the existing terminal's weight (clamped at 0; prune
                // happens on the next housekeeping pass).
                if score >= 0.0 {
                    n.reinforce_terminal(event.action_neuron, score, now, max_w);
                } else {
                    if let Some(t) = n.terminals.iter_mut()
                        .find(|t| t.target == event.action_neuron)
                    {
                        t.weight = (t.weight + score).max(0.0);
                        t.last_fired_tick = now;
                    }
                }
            }
        }
        true
    }

    /// Number of action events awaiting outcome feedback.  Surfaced
    /// for caller telemetry — a runaway value indicates outcomes are
    /// not being supplied for fired actions.
    pub fn pending_action_count(&self) -> usize {
        self.pending_actions.len()
    }

    // ---------------------------------------------------------------
    // Phase 8 — Distributed network (spec §5).
    //
    // The brain crate defines protocol + data model; transport lives
    // outside (cluster gossip plugs in here via the drain/ingest
    // methods).  Spec §1.7 puts the network as a property of the
    // brain factory, not a baked-in stack.
    // ---------------------------------------------------------------

    pub fn brain_id(&self) -> &str { &self.network.brain_id }
    pub fn set_brain_id(&mut self, id: impl Into<String>) {
        self.network.brain_id = id.into();
    }

    /// Drain all pending outbound motif records.  Transport calls
    /// this periodically (or after each `advance_tick`) and ships
    /// the result over the cluster.  Returns the drained motifs;
    /// the brain's outbound queue is left empty.
    pub fn drain_motif_gossip(&mut self) -> Vec<GossipMotif> {
        std::mem::take(&mut self.network.pending_motif_out)
    }

    /// Number of motif records currently queued for outbound gossip.
    pub fn pending_motif_count(&self) -> usize {
        self.network.pending_motif_out.len()
    }

    /// Ingest motif records gossiped from peer brains.  Records from
    /// this brain's own id are dropped (self-loop guard) — motif
    /// echoes from broadcast topologies should not show up in the
    /// network index.  Repeat records from the same peer about the
    /// same fingerprint update the existing entry rather than
    /// duplicating.
    pub fn ingest_motif_gossip(&mut self, motifs: Vec<GossipMotif>) {
        for mut m in motifs {
            if m.source_brain == self.network.brain_id { continue; }
            m.fingerprint.sort();
            let key = (m.source_brain.clone(), m.fingerprint.clone());
            if let Some(slot) = self.network.received_motifs.iter_mut()
                .find(|(k, _)| *k == key)
            {
                slot.1 = m;
            } else {
                self.network.received_motifs.push((key, m));
            }
        }
    }

    pub fn received_motif_count(&self) -> usize {
        self.network.received_motifs.len()
    }

    pub fn received_motifs_from(&self, peer: &str) -> Vec<&GossipMotif> {
        self.network.received_motifs.iter()
            .filter(|((b, _), _)| b == peer)
            .map(|(_, m)| m)
            .collect()
    }

    /// Snapshot every registered equation as a `GossipEquation`
    /// suitable for cluster broadcast.  Simple whole-state-sync is
    /// honest and idempotent — receivers merge by confidence.
    pub fn export_equations_for_gossip(&self) -> Vec<GossipEquation> {
        let id = self.network.brain_id.clone();
        self.eem.iter_equations().map(|eq| {
            let variable_names = eq.variables.iter()
                .filter_map(|vid| self.eem.variable(*vid).map(|v| v.name.clone()))
                .collect();
            let discipline_name = eq.discipline
                .and_then(|did| self.eem.discipline(did).map(|d| d.name.clone()));
            GossipEquation {
                source_brain:         id.clone(),
                name:                 eq.name.clone(),
                expression:           eq.expression.clone(),
                variable_names,
                discipline_name,
                confidence:           eq.confidence,
                validation_successes: eq.validation_successes,
                validation_failures:  eq.validation_failures,
            }
        }).collect()
    }

    /// Merge peer equation deltas into the local EEM.  Conflict
    /// resolution per spec §5.2: per-equation confidence wins.  A
    /// peer equation strictly higher in confidence than the local
    /// copy replaces the expression and adopts the peer's validation
    /// counters; otherwise the local copy is kept.  Unknown
    /// equations are added at the peer's confidence.
    pub fn ingest_equation_gossip(&mut self, equations: Vec<GossipEquation>) {
        for ge in equations {
            if ge.source_brain == self.network.brain_id { continue; }
            // Variables: register by name (idempotent).
            let var_ids: Vec<_> = ge.variable_names.iter()
                .map(|n| self.eem.register_variable(n.clone(), None))
                .collect();
            let disc_id = ge.discipline_name.as_ref()
                .map(|n| self.eem.register_discipline(n.clone()));

            match self.eem.equation_by_name(&ge.name) {
                Some(eq_id) => {
                    let local_conf = self.eem.confidence(eq_id).unwrap_or(0.0);
                    if ge.confidence > local_conf {
                        // Replace expression and re-seat confidence /
                        // validation counters from the peer.
                        self.eem.replace_equation_expression(eq_id, ge.expression.clone());
                        if let Some(eq) = self.eem.equation_mut(eq_id) {
                            eq.confidence           = ge.confidence;
                            eq.validation_successes = ge.validation_successes;
                            eq.validation_failures  = ge.validation_failures;
                        }
                    }
                }
                None => {
                    let eq_id = self.eem.register_equation(
                        ge.name.clone(),
                        ge.expression.clone(),
                        var_ids,
                        disc_id,
                    );
                    if let Some(eq) = self.eem.equation_mut(eq_id) {
                        eq.confidence           = ge.confidence;
                        eq.validation_successes = ge.validation_successes;
                        eq.validation_failures  = ge.validation_failures;
                    }
                }
            }
        }
    }

    pub fn peer_accuracy(&self, peer: &str) -> Option<&PeerAccuracy> {
        self.network.peer_accuracy.get(peer)
    }

    /// Record an outcome for a peer's contribution.  Drives the
    /// per-peer accuracy rate used by `integrate_with_peers`.
    pub fn report_peer_outcome(&mut self, peer: impl Into<String>, success: bool) {
        let entry = self.network.peer_accuracy.entry(peer.into()).or_default();
        if success {
            entry.successful_contributions = entry.successful_contributions.saturating_add(1);
        } else {
            entry.failed_contributions = entry.failed_contributions.saturating_add(1);
        }
    }

    pub fn known_peers(&self) -> Vec<String> {
        let mut v: Vec<String> = self.network.peer_accuracy.keys().cloned().collect();
        v.sort();
        v
    }

    /// Like [`Brain::integrate`] but folds caller-supplied peer
    /// contributions into the grounding report.  Each peer's
    /// `fabric_confidence` is weighted by its [`PeerAccuracy::rate`]
    /// (unknown peers get 0.5 — neutral) and surfaces in
    /// `GroundingReport.peer_contributions`.  The
    /// `integrated_confidence` becomes a weighted blend of the local
    /// fabric confidence and the average weighted peer confidence.
    pub fn integrate_with_peers(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
        peers:       &[PeerContribution],
    ) -> AnswerWithGrounding {
        let mut base = self.integrate(query_pool, target_pool);
        if peers.is_empty() { return base; }

        let mut weighted: Vec<(BrainId, f32)> = Vec::with_capacity(peers.len());
        let mut sum_weights = 0.0_f32;
        let mut sum_weighted_conf = 0.0_f32;
        for p in peers {
            let rate = self.network.peer_accuracy
                .get(&p.brain_id).map(|a| a.rate()).unwrap_or(0.5);
            let weighted_conf = (p.fabric_confidence * rate).clamp(0.0, 1.0);
            weighted.push((p.brain_id.clone(), weighted_conf));
            sum_weights += rate;
            sum_weighted_conf += weighted_conf * rate;
        }

        let local_fabric = base.grounding.fabric_confidence;
        let peer_blend = if sum_weights > 1e-9 {
            sum_weighted_conf / sum_weights
        } else { 0.0 };

        // Blend local + peer 50/50 by weighted contribution mass.
        // A stronger local signal still dominates; a confident-and-
        // accurate peer chorus lifts the integrated score.
        let blended = (local_fabric + peer_blend) / 2.0;
        base.grounding.peer_contributions = weighted;
        base.grounding.integrated_confidence = blended;
        base.confidence_tier = ConfidenceTier::from_confidence(
            base.grounding.integrated_confidence,
            base.grounding.outside_grounding,
            base.grounding.speculation_flag,
        );
        base
    }

    // ---------------------------------------------------------------
    // Phase 10 — Brain factory (spec §3 + §11).
    //
    // Builds a fully-wired brain from a declarative
    // `BrainIdentitySpec` plus a `PoolPrototypeRegistry` that knows
    // how to instantiate encodings.  The Action pool (if any) is
    // auto-designated; the EEM and annealer come up empty for the
    // caller to seed.
    // ---------------------------------------------------------------

    pub fn from_identity(
        identity: &BrainIdentitySpec,
        registry: &PoolPrototypeRegistry,
    ) -> Result<Self, IdentityBuildError> {
        // Detect collisions and binding-pool-id reservations up front
        // so the brain isn't half-built when an error surfaces.
        let binding_pool_id: PoolId = 0;
        let mut seen_ids = ahash::AHashSet::new();
        for ps in &identity.pools {
            if ps.id == binding_pool_id {
                return Err(IdentityBuildError::BindingPoolIdCollision(ps.id));
            }
            if !seen_ids.insert(ps.id) {
                return Err(IdentityBuildError::DuplicatePoolId(ps.id));
            }
        }

        let mut binding_pool_config = PoolConfig::defaults("binding", binding_pool_id);
        binding_pool_config.concept_emergence_threshold = u32::MAX;
        let config = BrainConfig {
            fabric:                      identity.fabric.clone(),
            binding_emergence_threshold: identity.binding_emergence_threshold,
            moment_history_window:       identity.moment_history_window,
            binding_pool_config,
            eem:                         identity.eem.clone(),
            annealer:                    identity.annealer.clone(),
        };
        let mut brain = Self::new(config);

        for ps in &identity.pools {
            let encoding = registry
                .build(&ps.prototype, &ps.atom_encoding_prefix)
                .ok_or_else(|| IdentityBuildError::UnknownPrototype(ps.prototype.clone()))?;
            brain.create_pool(ps.to_pool_config(), encoding);
        }

        // Auto-designate the first Action pool, if any.  At most one
        // Action pool is meaningful per spec §4.E; later Action pools
        // in the spec are stored but not designated.
        for ps in &identity.pools {
            if matches!(ps.kind, PoolKind::Action) {
                brain.designate_action_pool(ps.id);
                break;
            }
        }

        Ok(brain)
    }

    // ---------------------------------------------------------------
    // Stage 4 — Sleep cycle (spec §6.3).
    //
    // Identifies weak/stale concept neurons across all pools and
    // soft-prunes them: their outgoing terminals are zeroed and all
    // inbound terminals (within-pool AND cross-pool) are removed.
    // Then runs an extra housekeeping pass on every pool so the
    // zeroed terminals drop below the prune floor.
    //
    // Sleep is what makes "survival of fittest" actually surface
    // strong emergence — without it, every cross-pair boundary
    // pattern that crosses the emergence threshold sticks around and
    // dilutes the substrate's concept space.
    // ---------------------------------------------------------------

    /// Soft-prune all concept neurons across all pools whose
    /// `use_count` is below `min_use_count` and whose `last_fired_tick`
    /// is more than `stale_ticks` in the past.
    ///
    /// Returns the total number of concepts pruned.
    pub fn sleep(&mut self, min_use_count: u64, stale_ticks: u64) -> usize {
        let current_tick = self.fabric.current_tick();
        let mut total_pruned = 0usize;
        let mut pruned_refs: ahash::AHashSet<NeuronRef> = ahash::AHashSet::new();

        // Phase 1: per-pool prune; collect (pool, neuron) refs.
        for pid in self.fabric.pool_ids() {
            if let Some(pool) = self.fabric.pool(pid) {
                let pruned_ids = pool.write()
                    .prune_weak_concepts(min_use_count, stale_ticks, current_tick);
                total_pruned += pruned_ids.len();
                for nid in pruned_ids {
                    pruned_refs.insert(NeuronRef::new(pid, nid));
                }
            }
        }

        // Phase 2: clean cross-pool inbound terminals targeting
        // anything pruned.
        if !pruned_refs.is_empty() {
            for pid in self.fabric.pool_ids() {
                if let Some(pool) = self.fabric.pool(pid) {
                    pool.write().prune_inbound_to(&pruned_refs);
                }
            }
        }

        // Phase 3: extra housekeeping so any zero-weight residuals
        // drop below the prune floor.
        for pid in self.fabric.pool_ids() {
            if let Some(pool) = self.fabric.pool(pid) {
                pool.write().tick_housekeeping();
            }
        }

        total_pruned
    }

    // ---------------------------------------------------------------
    // Phase 9 — Checkpoint / restore (spec §6 + §11).
    //
    // Persists the entire learned state of the brain to a single
    // bincode-encoded file.  Restore rebuilds it verbatim; the only
    // thing the caller re-supplies is the pool encodings (which are
    // stateless trait objects).
    // ---------------------------------------------------------------

    /// Capture this brain's full persistent state into a snapshot
    /// struct.  Use [`crate::persistence::save_snapshot`] to write it
    /// to disk, or hold it in memory for tests.
    pub fn snapshot(&self) -> crate::persistence::BrainSnapshot {
        let moment_history: std::collections::VecDeque<crate::persistence::SerializableFingerprint> =
            self.moment_history.iter()
                .map(|f| crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() })
                .collect();
        let binding_recurrences: Vec<(crate::persistence::SerializableFingerprint, u32)> =
            self.binding_recurrences.iter()
                .map(|(f, &c)| (crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() }, c))
                .collect();
        let promoted_fingerprints: Vec<(crate::persistence::SerializableFingerprint, NeuronId)> =
            self.promoted_fingerprints.iter()
                .map(|(f, &n)| (crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() }, n))
                .collect();
        let pending_actions: Vec<(ActionId, ActionEvent)> = self.pending_actions
            .iter().map(|(k, v)| (*k, v.clone())).collect();
        crate::persistence::BrainSnapshot {
            binding_pool_id:             self.binding_pool_id,
            binding_emergence_threshold: self.config.binding_emergence_threshold,
            moment_history_window:       self.config.moment_history_window,
            fabric:                      self.fabric.snapshot(),
            eem:                         self.eem.snapshot(),
            annealer:                    self.annealer.snapshot(),
            moment_history,
            binding_recurrences,
            promoted_fingerprints,
            action_pool_id:              self.action_pool_id,
            pending_actions,
            next_action_id:              self.next_action_id,
        }
    }

    /// Write this brain's state to `path`.  Convenience wrapper over
    /// `snapshot()` + `persistence::save_snapshot()`.
    pub fn checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        crate::persistence::save_snapshot(&self.snapshot(), path)
    }

    /// Rebuild a brain from a snapshot, supplying fresh encodings
    /// for every pool id in the snapshot.  The binding pool is
    /// included in `encodings`; pass a `BytePassthroughEncoding`
    /// matching the original prefix (`"bind"` for default brains).
    ///
    /// Returns `(brain, missing_pool_ids)` where `missing_pool_ids`
    /// lists pool ids in the snapshot for which no encoding was
    /// supplied — those pools are silently skipped.  Callers should
    /// treat a non-empty missing list as a misconfiguration.
    pub fn from_snapshot(
        snap:      crate::persistence::BrainSnapshot,
        encodings: std::collections::HashMap<PoolId, Box<dyn AtomEncoding>>,
    ) -> (Self, Vec<PoolId>) {
        let (fabric, missing) = Fabric::from_snapshot(snap.fabric, encodings);
        let config = BrainConfig {
            fabric: fabric.config.clone(),
            binding_emergence_threshold: snap.binding_emergence_threshold,
            moment_history_window:       snap.moment_history_window,
            binding_pool_config:         PoolConfig::defaults("binding", snap.binding_pool_id),
            eem:                         snap.eem.config.clone(),
            annealer:                    snap.annealer.config.clone(),
        };

        let mut moment_history = VecDeque::with_capacity(snap.moment_history_window);
        for f in snap.moment_history {
            moment_history.push_back(MomentFingerprint { pairs: f.pairs });
        }
        let mut binding_recurrences = AHashMap::new();
        for (f, c) in snap.binding_recurrences {
            binding_recurrences.insert(MomentFingerprint { pairs: f.pairs }, c);
        }
        let mut promoted_fingerprints = AHashMap::new();
        for (f, n) in snap.promoted_fingerprints {
            promoted_fingerprints.insert(MomentFingerprint { pairs: f.pairs }, n);
        }
        let mut pending_actions = AHashMap::new();
        for (k, v) in snap.pending_actions { pending_actions.insert(k, v); }

        let brain = Self {
            fabric,
            config,
            binding_pool_id:       snap.binding_pool_id,
            moment_history,
            binding_recurrences,
            promoted_fingerprints,
            action_pool_id:        snap.action_pool_id,
            pending_actions,
            next_action_id:        snap.next_action_id,
            emitted_this_tick:     ahash::AHashSet::new(),
            eem:                   crate::eem::Eem::from_snapshot(snap.eem),
            annealer:              crate::annealer::Annealer::from_snapshot(snap.annealer),
            network:               NetworkState::new(""),
        };
        (brain, missing)
    }

    /// Load a brain from `path`.  Convenience wrapper over
    /// `persistence::load_snapshot()` + `from_snapshot()`.
    pub fn restore<P: AsRef<std::path::Path>>(
        path:      P,
        encodings: std::collections::HashMap<PoolId, Box<dyn AtomEncoding>>,
    ) -> std::io::Result<(Self, Vec<PoolId>)> {
        let snap = crate::persistence::load_snapshot(path)?;
        Ok(Self::from_snapshot(snap, encodings))
    }

    pub fn stats(&self) -> BrainStats {
        let mut stats = BrainStats {
            tick:                self.fabric.current_tick(),
            pool_count:          0,
            total_neurons:       0,
            total_concepts:      0,
            total_binding:       0,
            total_terminals:     0,
            binding_pool_id:     self.binding_pool_id,
            fingerprints_window: self.moment_history.len(),
        };
        for pid in self.fabric.pool_ids() {
            if let Some(p) = self.fabric.pool(pid) {
                let pool = p.read();
                stats.pool_count += 1;
                let nc = pool.neuron_count();
                stats.total_neurons += nc;
                let cc = pool.concept_count();
                stats.total_concepts += cc;
                stats.total_terminals += pool.iter_neurons()
                    .map(|n| n.terminals.len()).sum::<usize>();
                if pid == self.binding_pool_id {
                    stats.total_binding += cc;
                }
            }
        }
        stats
    }
}
