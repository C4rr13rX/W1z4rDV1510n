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

use crate::fabric::{Fabric, FabricConfig};
use crate::grounding::{AnswerWithGrounding, ConfidenceTier, GroundingReport};
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

#[derive(Debug, Clone)]
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
        Self {
            fabric,
            config,
            binding_pool_id,
            moment_history:        VecDeque::with_capacity(window),
            binding_recurrences:   AHashMap::new(),
            promoted_fingerprints: AHashMap::new(),
        }
    }

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

        self.fabric.advance_tick();

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

        if *count >= self.config.binding_emergence_threshold
            && !self.promoted_fingerprints.contains_key(&fp)
        {
            let new_id = self.promote_binding_concept(&fp);
            if let Some(id) = new_id {
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

        let mut strongest: Option<(NeuronRef, f32)> = None;
        let mut all_active_concepts: Vec<NeuronRef> = Vec::new();
        for (&nid, &act) in target_activation.iter() {
            if act < 0.001 { continue; }
            if let Some(n) = target.get(nid) {
                if !n.is_atom() {
                    let r = NeuronRef::new(target_pool, nid);
                    all_active_concepts.push(r);
                    if strongest.map_or(true, |(_, s)| act > s) {
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

        // Decode the strongest match's atom members back through the
        // target pool's encode/decode contract.
        let answer = strongest_match.and_then(|r| {
            target.get(r.neuron).map(|n| {
                let member_atoms: Vec<(&str, f32)> = n.members.iter()
                    .filter(|m| m.pool == target_pool)
                    .filter_map(|m| {
                        target.get(m.neuron).map(|m_neuron| {
                            (m_neuron.label.as_str(),
                             target_activation.get(&m.neuron).copied().unwrap_or(1.0))
                        })
                    })
                    .collect();
                target.encoding_reassemble(&member_atoms)
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
            },
            confidence_tier,
            next_steps_if_ungrounded: Vec::new(),
        }
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
