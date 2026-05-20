//! Pool primitive per [`ARCHITECTURE.md`] §3.1 and §4.A.
//!
//! A pool is a collection of neurons sharing a sensor resolution and an
//! atomization contract.  Atoms enter via [`Pool::observe_frame`]; concept
//! emergence happens automatically as repeating sequences accumulate in
//! `recent_atoms`.  The pool itself doesn't reach across pool boundaries —
//! cross-pool wiring is a [`crate::Fabric`] responsibility because it
//! requires the moment buffer (what fired in every pool at tick T).

use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::neuron::{Neuron, NeuronId, NeuronKind, NeuronRef, PoolId};

/// Encode/decode contract per spec §3.1.  Each pool ships with one of these
/// declaring how raw sensor frames decompose into atom labels and how an
/// activation map reassembles into a frame.
///
/// Implementations live with sensor adapters, not in the pool itself — the
/// pool just holds the trait object.
pub trait AtomEncoding: Send + Sync {
    /// Decompose a raw sensor frame into a list of atom labels at the
    /// pool's bit resolution.  Order matters: it's the firing order that
    /// concept emergence uses as the position signal.
    fn atomize(&self, frame: &[u8]) -> Vec<String>;

    /// Reassemble an activation map (atom label → activation strength) into
    /// a raw sensor frame.  Inverse of `atomize` for the same input.
    fn reassemble(&self, active_atoms: &[(&str, f32)]) -> Vec<u8>;

    /// Stable name for this encoding ("bytes/utf8", "rgb-8x8/15fps", etc).
    /// Surfaced in the pool's developmental profile.
    fn name(&self) -> &'static str;
}

/// The single simplest encoding: each byte of the input becomes an atom
/// labeled `prefix:<base64(byte)>`.  Used by language pools and any
/// stream that doesn't need further bit-resolution work.  Other encodings
/// (image pixel bins, FFT audio bins) live with sensor adapters.
pub struct BytePassthroughEncoding {
    pub prefix: &'static str,
}

impl AtomEncoding for BytePassthroughEncoding {
    fn atomize(&self, frame: &[u8]) -> Vec<String> {
        use base64::Engine;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        frame.iter()
            .map(|&b| format!("{}:{}", self.prefix, engine.encode([b])))
            .collect()
    }

    fn reassemble(&self, active_atoms: &[(&str, f32)]) -> Vec<u8> {
        use base64::Engine;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        // Strongest-first; the encoding is lossy when multiple atoms fire,
        // but for a pure byte-stream pool a single byte fires per atom slot
        // by construction.  This implementation picks the strongest atom
        // and decodes its base64 payload.
        let mut out = Vec::with_capacity(active_atoms.len());
        for (label, _) in active_atoms {
            if let Some(payload) = label.strip_prefix(self.prefix)
                .and_then(|s| s.strip_prefix(':'))
            {
                if let Ok(bytes) = engine.decode(payload) {
                    out.extend_from_slice(&bytes);
                }
            }
        }
        out
    }

    fn name(&self) -> &'static str { "byte-passthrough" }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub name:                        String,
    pub id:                          PoolId,
    pub frame_rate:                  u32,
    pub recent_atoms_window:         usize,
    /// A sequence of length 2..=`max_concept_member_count` that recurs at
    /// least `concept_emergence_threshold` times in `recent_atoms` is
    /// promoted to a concept.  Per spec §4.A.
    pub max_concept_member_count:    usize,
    pub concept_emergence_threshold: u32,
    pub max_weight:                  f32,
    pub decay_rate:                  f32,
    pub prune_floor:                 f32,
    pub plasticity_baseline:         f32,

    /// k-WTA sparsity target.  After every `tick_housekeeping`, only the
    /// top fraction of currently-firing neurons (ranked by activation)
    /// remain firing — the rest are inhibited.  Biological motivation:
    /// V1 firing rates ~2-5% (Vinje & Gallant 2000); cortical sparse
    /// coding (Olshausen & Field 1996); local interneuron k-WTA
    /// (Maass 2000).  Default 1.0 = no sparsity (substrate behaviour
    /// unchanged).  Range: (0.0, 1.0].  Empirical falsification of
    /// Stage 14 showed unconstrained Hebbian co-firing produces
    /// runaway concept-of-concept emergence (mega-bindings of 797 +
    /// members); k-WTA is the smallest biological mechanism that
    /// directly bounds this.
    #[serde(default = "default_sparsity_top_k_frac")]
    pub sparsity_top_k_frac:         f32,
    /// Minimum number of neurons that stay firing after the k-WTA
    /// gate even when `sparsity_top_k_frac` would round to fewer.
    /// Prevents complete silence in low-traffic pools.
    #[serde(default = "default_sparsity_min_neurons")]
    pub sparsity_min_neurons:        usize,
}

fn default_sparsity_top_k_frac() -> f32 { 1.0 }
fn default_sparsity_min_neurons() -> usize { 1 }

impl PoolConfig {
    pub fn defaults(name: impl Into<String>, id: PoolId) -> Self {
        Self {
            name: name.into(),
            id,
            frame_rate: 30,
            recent_atoms_window: 32,
            max_concept_member_count: 8,
            concept_emergence_threshold: 3,
            max_weight: 4.0,
            decay_rate: 0.0005,
            prune_floor: 0.01,
            plasticity_baseline: 0.1,
            sparsity_top_k_frac: 1.0,
            sparsity_min_neurons: 1,
        }
    }
}

/// Track of which atom-sequences have recurred and how many times.  When a
/// sequence's count crosses `concept_emergence_threshold`, the pool promotes
/// it to a concept neuron whose members are those atom IDs in observed
/// firing order.  Once promoted, the sequence is cleared from the tracker
/// (the concept neuron now carries the identity).
type SequenceFingerprint = Vec<NeuronId>;

pub struct Pool {
    pub config:       PoolConfig,
    encoding:         Box<dyn AtomEncoding>,
    neurons:          Vec<Neuron>,
    label_to_id:      AHashMap<String, NeuronId>,
    /// Stage 13 — atom-multiset dedup index.  Key = sorted Vec of atom
    /// leaf NeuronIds (the multiset signature of a concept's full
    /// expansion).  Value = the FIRST concept promoted with that
    /// multiset (the canonical one).  Prevents permutation-variant
    /// concepts like ("f,o,o,d" + "o,o,d,f" + "ood,f") from cluttering
    /// the pool when round-robin training under destructive collapse
    /// produces multiple member orderings of the same byte multiset.
    /// See scripts/brain_dense_burst_toddler_categorical.py for the
    /// empirical observation that motivated this.
    concept_multiset_to_id: AHashMap<Vec<NeuronId>, NeuronId>,
    /// Streaming buffer of recently-fired atom/concept IDs.  Drives concept
    /// emergence.  Bounded by `config.recent_atoms_window`.
    recent_atoms:     VecDeque<NeuronId>,
    /// Per-sequence recurrence count.  Concept emergence fires when a
    /// sequence's count crosses the threshold.
    sequences:        AHashMap<SequenceFingerprint, u32>,
    /// IDs of neurons currently firing (activation > min_activation).
    /// Rebuilt every observe call.  Read by the Fabric for the moment
    /// buffer and cross-pool wiring.
    currently_firing: AHashSet<NeuronId>,
    /// Per-neuron transient activation for the current tick.  Cleared at
    /// the start of each observe call.
    activation:       AHashMap<NeuronId, f32>,
}

impl Pool {
    pub fn new(config: PoolConfig, encoding: Box<dyn AtomEncoding>) -> Self {
        let window = config.recent_atoms_window;
        Self {
            config,
            encoding,
            neurons:          Vec::new(),
            label_to_id:      AHashMap::new(),
            concept_multiset_to_id: AHashMap::new(),
            recent_atoms:     VecDeque::with_capacity(window),
            sequences:        AHashMap::new(),
            currently_firing: AHashSet::new(),
            activation:       AHashMap::new(),
        }
    }

    pub fn id(&self) -> PoolId        { self.config.id }
    pub fn name(&self) -> &str        { &self.config.name }
    pub fn neuron_count(&self) -> usize { self.neurons.len() }
    pub fn concept_count(&self) -> usize {
        self.neurons.iter().filter(|n| !n.is_atom()).count()
    }
    pub fn currently_firing(&self) -> impl Iterator<Item = NeuronId> + '_ {
        self.currently_firing.iter().copied()
    }
    pub fn get(&self, id: NeuronId) -> Option<&Neuron>           { self.neurons.get(id as usize) }
    pub fn get_mut(&mut self, id: NeuronId) -> Option<&mut Neuron> { self.neurons.get_mut(id as usize) }
    pub fn iter_neurons(&self) -> impl Iterator<Item = &Neuron>  { self.neurons.iter() }
    pub fn iter_neurons_mut(&mut self) -> impl Iterator<Item = &mut Neuron> { self.neurons.iter_mut() }

    pub fn label_to_id(&self, label: &str) -> Option<NeuronId> {
        self.label_to_id.get(label).copied()
    }

    /// Insert a fully-constructed concept neuron.  Used by the Brain to
    /// place binding concepts (members span multiple pools) into the
    /// binding pool without going through atom-stream emergence.  The
    /// neuron's `id` field is overwritten to match the assigned slot.
    pub fn append_neuron(&mut self, mut neuron: Neuron, label: String) -> NeuronId {
        let id = self.neurons.len() as NeuronId;
        neuron.id = id;
        self.neurons.push(neuron);
        self.label_to_id.insert(label, id);
        id
    }

    /// Reassemble an activation map back into a sensor frame through
    /// this pool's encoding contract.  Used by [`crate::Brain::integrate`]
    /// to produce the `answer: Option<Vec<u8>>` payload.  Spec §3.4.
    pub fn encoding_reassemble(&self, active: &[(&str, f32)]) -> Vec<u8> {
        self.encoding.reassemble(active)
    }

    /// Recursive expanded size: number of atom leaves a concept (or
    /// list of members) decodes to.  Used by [`crate::Brain::integrate`]
    /// to score candidate strongest-matches — concepts that expand to
    /// fewer bytes are more specific and preferred when their per-byte
    /// activation density is comparable.
    pub fn expanded_size(&self, members: &[NeuronRef]) -> usize {
        let mut total = 0;
        for m in members {
            if m.pool != self.config.id { continue; }
            if let Some(n) = self.neurons.get(m.neuron as usize) {
                if n.is_atom() {
                    total += 1;
                } else {
                    total += self.expanded_size(&n.members);
                }
            }
        }
        total.max(1)
    }

    /// Recursively decode a concept's member list into raw bytes.
    /// Atom members decode through the pool's encoding contract; concept
    /// members recurse into their own `members`.  Members in OTHER pools
    /// are skipped (caller decodes those separately if needed).
    ///
    /// This is the hierarchy-aware decode path used by
    /// [`crate::Brain::integrate`] when the strongest match is a
    /// concept-of-concepts.  Without recursion, a concept whose
    /// members are themselves concepts would decode to nothing — its
    /// member labels (e.g. `"c:eA~c:Kw"`) aren't valid base64 to the
    /// atom-level reassemble.
    pub fn decode_concept_members(&self, members: &[NeuronRef]) -> Vec<u8> {
        let mut out = Vec::new();
        for m in members {
            if m.pool != self.config.id { continue; }
            let neuron = match self.neurons.get(m.neuron as usize) {
                Some(n) => n,
                None => continue,
            };
            if neuron.is_atom() {
                let pairs = [(neuron.label.as_str(), 1.0_f32)];
                out.extend(self.encoding.reassemble(&pairs));
            } else {
                let nested = self.decode_concept_members(&neuron.members);
                out.extend(nested);
            }
        }
        out
    }

    /// Snapshot the persistent learning state of this pool.  Transient
    /// runtime state (currently_firing, activation) is intentionally
    /// dropped — those represent a single tick's in-flight signal and
    /// rebuild on the next `observe`.  Spec §6.1.
    pub fn snapshot(&self) -> crate::persistence::PoolSnapshot {
        let label_to_id: std::collections::HashMap<String, NeuronId> = self
            .label_to_id
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let sequences: Vec<(Vec<NeuronId>, u32)> = self
            .sequences
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        crate::persistence::PoolSnapshot {
            config:       self.config.clone(),
            neurons:      self.neurons.clone(),
            label_to_id,
            recent_atoms: self.recent_atoms.clone(),
            sequences,
        }
    }

    /// Construct a Pool from a snapshot and a fresh encoding.  The
    /// caller is responsible for supplying an encoding compatible with
    /// the atom labels stored in the snapshot — e.g. the same
    /// `BytePassthroughEncoding { prefix }` used originally.  If the
    /// encoding contract differs, atoms still hold their labels but
    /// decode-side reassembly may produce different bytes.
    pub fn from_snapshot(
        snap:     crate::persistence::PoolSnapshot,
        encoding: Box<dyn AtomEncoding>,
    ) -> Self {
        let mut label_to_id = AHashMap::new();
        for (k, v) in snap.label_to_id { label_to_id.insert(k, v); }
        let mut sequences = AHashMap::new();
        for (k, v) in snap.sequences { sequences.insert(k, v); }
        let mut pool = Self {
            config:           snap.config,
            encoding,
            neurons:          snap.neurons,
            label_to_id,
            concept_multiset_to_id: AHashMap::new(),
            recent_atoms:     snap.recent_atoms,
            sequences,
            currently_firing: AHashSet::new(),
            activation:       AHashMap::new(),
        };
        // Rebuild the multiset dedup index from restored concept
        // neurons.  The index isn't part of the snapshot format (it
        // can always be rebuilt from members) so restore from older
        // snapshots remains lossless.
        let concept_ids: Vec<NeuronId> = pool.neurons.iter()
            .filter(|n| !n.is_atom())
            .map(|n| n.id)
            .collect();
        for cid in concept_ids {
            let member_ids: Vec<NeuronId> = pool.neurons[cid as usize].members.iter()
                .filter(|m| m.pool == pool.config.id)
                .map(|m| m.neuron)
                .collect();
            let mut leaves = pool.expand_to_atom_leaves(&member_ids);
            leaves.sort();
            pool.concept_multiset_to_id.entry(leaves).or_insert(cid);
        }
        pool
    }

    /// Recursively expand a member list into the atom-leaf NeuronIds.
    /// Concept members are walked into their own members.  Members in
    /// OTHER pools are skipped.
    fn expand_to_atom_leaves(&self, member_ids: &[NeuronId]) -> Vec<NeuronId> {
        let mut leaves = Vec::new();
        for &id in member_ids {
            if let Some(n) = self.neurons.get(id as usize) {
                if n.is_atom() {
                    leaves.push(id);
                } else {
                    let sub: Vec<NeuronId> = n.members.iter()
                        .filter(|m| m.pool == self.config.id)
                        .map(|m| m.neuron)
                        .collect();
                    leaves.extend(self.expand_to_atom_leaves(&sub));
                }
            }
        }
        leaves
    }

    pub fn activation(&self, id: NeuronId) -> f32 {
        self.activation.get(&id).copied().unwrap_or(0.0)
    }

    /// Atomize a sensor frame and fire the corresponding atom neurons.
    /// Creates atoms for previously-unseen labels (neurogenesis).
    ///
    /// After **each** atom fires, the substrate runs:
    ///  1. **Mini-column collapse** ([`collapse_tail_to_concept`]) —
    ///     if the tail of `recent_atoms` matches an already-emerged
    ///     concept's member sequence, those tail entries are *replaced*
    ///     by the concept's id and the concept fires.  The collapse
    ///     loops so that level-2 columns (concepts-of-concepts) can
    ///     form in the same observation when their level-1 components
    ///     have just collapsed.  This is the spec §4.A "Hierarchy
    ///     birth" mechanism — bytes become morpheme concepts, morpheme
    ///     concepts become word concepts, word concepts become phrase
    ///     concepts, all driven by the same Hebbian sequence rule.
    ///  2. **Per-atom emergence accounting** ([`check_concept_emergence`])
    ///     — every tail pattern ending at the just-fired entry gets its
    ///     recurrence count bumped.  Patterns ending mid-word (like the
    ///     morpheme `r-u-n` inside `r-u-n-n-i-n-g`) are now caught;
    ///     previously the per-frame accounting only saw the final
    ///     position.
    ///
    /// The returned `fired` list is **atom-only** — concept firings do
    /// NOT enter the moment buffer.  Cross-pool wiring stays
    /// atom-level so fan-out remains bounded.
    pub fn observe_frame(&mut self, frame: &[u8], tick: u64) -> Vec<NeuronId> {
        let labels = self.encoding.atomize(frame);
        let mut fired = Vec::with_capacity(labels.len());
        self.activation.clear();
        self.currently_firing.clear();

        for label in labels {
            let id = self.ensure_atom(label, tick);
            self.activation.insert(id, 1.0);
            self.currently_firing.insert(id);
            fired.push(id);
            self.push_recent(id);

            if let Some(n) = self.neurons.get_mut(id as usize) {
                n.use_count = n.use_count.saturating_add(1);
                n.last_fired_tick = tick;
            }

            // Mini-column collapse: pop matched atoms from the tail of
            // recent_atoms and push the concept id in their place.
            // Loop so level-2 columns also collapse in one pass.
            while self.collapse_tail_to_concept(tick) {}

            // Per-atom emergence counting.  Patterns ending at the
            // current tail entry (which may be a concept after
            // collapse) get their counts bumped.
            self.check_concept_emergence(tick);
        }

        fired
    }

    /// Find the LONGEST tail of `recent_atoms` whose joined-label
    /// matches an existing concept's label.  If found, pop those tail
    /// entries and push the concept id — this is the mini-column
    /// collapse.  Returns `true` if a collapse happened (so caller can
    /// loop to detect level-2+ collapses on the new tail).
    fn collapse_tail_to_concept(&mut self, tick: u64) -> bool {
        let buf_len = self.recent_atoms.len();
        if buf_len < 2 { return false; }
        let max_len = self.config.max_concept_member_count.min(buf_len);

        let mut found: Option<(usize, NeuronId)> = None;
        // Longest tail wins — wider column receptive field.
        for len in (2..=max_len).rev() {
            let start = buf_len - len;
            let mut composite = String::new();
            let mut ok = true;
            for (i, id) in self.recent_atoms.iter().skip(start).enumerate() {
                if let Some(n) = self.neurons.get(*id as usize) {
                    if i > 0 { composite.push('~'); }
                    composite.push_str(&n.label);
                } else { ok = false; break; }
            }
            if !ok { continue; }
            if let Some(&cid) = self.label_to_id.get(&composite) {
                if let Some(n) = self.neurons.get(cid as usize) {
                    if !n.is_atom() {
                        found = Some((len, cid));
                        break;
                    }
                }
            }
        }

        if let Some((len, cid)) = found {
            for _ in 0..len {
                self.recent_atoms.pop_back();
            }
            self.recent_atoms.push_back(cid);
            self.activation.insert(cid, 1.0);
            self.currently_firing.insert(cid);
            if let Some(n) = self.neurons.get_mut(cid as usize) {
                n.use_count = n.use_count.saturating_add(1);
                n.last_fired_tick = tick;
            }
            true
        } else {
            false
        }
    }

    /// Inject activation into a specific neuron from a propagation walk
    /// (used by the Fabric when a terminal targets one of this pool's
    /// neurons).  Idempotent across multiple injections in the same tick:
    /// activations sum, the highest-firing concept wins downstream.
    pub fn inject_activation(&mut self, id: NeuronId, delta: f32, tick: u64) {
        if let Some(slot) = self.neurons.get_mut(id as usize) {
            slot.last_fired_tick = tick;
            slot.use_count = slot.use_count.saturating_add(1);
        }
        *self.activation.entry(id).or_insert(0.0) += delta;
        if delta > 0.0 {
            self.currently_firing.insert(id);
        }
    }

    /// Tick housekeeping: every neuron's terminals decay; sub-floor terminals
    /// are pruned.  Spec §1.5: this is the only forgetting mechanism.
    pub fn tick_housekeeping(&mut self) {
        let decay = self.config.decay_rate;
        let floor = self.config.prune_floor;
        for n in self.neurons.iter_mut() {
            n.decay_and_prune(decay, floor);
        }
        self.apply_kwta_sparsity();
    }

    /// Biologically-motivated k-WTA sparsity gate.  After housekeeping
    /// decay, sort `currently_firing` by activation strength descending
    /// and keep only the top `sparsity_top_k_frac` fraction (with a
    /// floor of `sparsity_min_neurons`).  The rest are evicted from
    /// the firing set AND from the activation map so downstream
    /// propagation does not see them.
    ///
    /// Rationale: unconstrained Hebbian co-firing accumulates ever-
    /// larger concept-of-concept bindings (empirically 797+ members
    /// in the Stage 14 falsification).  Biology constrains this via
    /// local inhibitory interneurons enforcing 2-5% firing rates.
    /// k-WTA captures the functional effect without modelling
    /// individual interneurons.
    fn apply_kwta_sparsity(&mut self) {
        let frac = self.config.sparsity_top_k_frac;
        if frac >= 1.0 { return; }
        let n_firing = self.currently_firing.len();
        if n_firing == 0 { return; }
        let target_k = ((frac * n_firing as f32).ceil() as usize)
            .max(self.config.sparsity_min_neurons)
            .min(n_firing);
        if target_k >= n_firing { return; }

        // Collect (id, activation) for all currently-firing neurons.
        let mut ranked: Vec<(NeuronId, f32)> = self.currently_firing.iter()
            .map(|&id| (id, *self.activation.get(&id).unwrap_or(&0.0)))
            .collect();
        // Sort descending by activation; ties broken by id (stable).
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // Evict everything below rank target_k.
        for &(id, _) in ranked.iter().skip(target_k) {
            self.currently_firing.remove(&id);
            self.activation.remove(&id);
        }
    }

    /// Sleep-cycle pruning per spec §6.3.  Walks every concept neuron;
    /// if its `use_count` is below `min_use_count` AND it was last
    /// fired more than `stale_ticks` ago, it's treated as noise (an
    /// emergence artifact that never consolidated through repeated
    /// use) and made inert:
    ///   - Its outgoing terminals are zeroed (housekeeping prunes
    ///     them on subsequent ticks).
    ///   - All other neurons' terminals TARGETING this concept are
    ///     removed.
    ///
    /// The concept itself remains in the `neurons` Vec because
    /// removing it would re-index every higher-id neuron — an
    /// expensive cascading rewrite.  Soft prune is the spec-aligned
    /// "survival of fittest" outcome: weak emergence withers, strong
    /// emergence persists.
    ///
    /// Returns the set of pruned concept ids.
    pub fn prune_weak_concepts(
        &mut self,
        min_use_count: u64,
        stale_ticks:   u64,
        current_tick:  u64,
    ) -> ahash::AHashSet<NeuronId> {
        // Identify pruneable concept ids first to avoid borrow conflicts.
        let mut to_prune: ahash::AHashSet<NeuronId> = ahash::AHashSet::new();
        for n in self.neurons.iter() {
            if n.is_atom() { continue; }
            let age = current_tick.saturating_sub(n.last_fired_tick);
            if n.use_count < min_use_count && age > stale_ticks {
                to_prune.insert(n.id);
            }
        }
        if to_prune.is_empty() { return to_prune; }

        // Zero outgoing terminals of pruned concepts and remove their
        // entries from label_to_id (so they can't collapse again).
        for &cid in &to_prune {
            if let Some(n) = self.neurons.get_mut(cid as usize) {
                n.terminals.clear();
                let label = n.label.clone();
                self.label_to_id.remove(&label);
            }
        }

        // Remove inbound terminals targeting pruned concepts from EVERY
        // other neuron in this pool.  Cross-pool terminals targeting
        // these concepts from other pools must be cleaned up by the
        // caller (Fabric::sleep does this).
        let pool_id = self.config.id;
        for n in self.neurons.iter_mut() {
            n.terminals.retain(|t| {
                !(t.target.pool == pool_id && to_prune.contains(&t.target.neuron))
            });
        }

        // Clear currently_firing / activation entries for pruned ids.
        for cid in &to_prune {
            self.currently_firing.remove(cid);
            self.activation.remove(cid);
        }

        // Remove sequences map entries that reference pruned ids.
        self.sequences.retain(|k, _| !k.iter().any(|id| to_prune.contains(id)));

        // Remove recent_atoms entries that point to pruned ids.
        self.recent_atoms.retain(|id| !to_prune.contains(id));

        to_prune
    }

    /// Remove terminals in this pool that target any of the
    /// specified (pool, neuron) ids.  Used by `Fabric::sleep` to
    /// clean up cross-pool terminals after another pool pruned
    /// some of its concepts.
    pub fn prune_inbound_to(
        &mut self,
        targets: &ahash::AHashSet<NeuronRef>,
    ) -> usize {
        let mut removed = 0;
        for n in self.neurons.iter_mut() {
            let before = n.terminals.len();
            n.terminals.retain(|t| !targets.contains(&t.target));
            removed += before - n.terminals.len();
        }
        removed
    }

    fn ensure_atom(&mut self, label: String, tick: u64) -> NeuronId {
        if let Some(&id) = self.label_to_id.get(&label) {
            return id;
        }
        let id = self.neurons.len() as NeuronId;
        let neuron = Neuron::new_atom(id, label.clone(), NeuronKind::Excitatory, tick);
        self.neurons.push(neuron);
        self.label_to_id.insert(label, id);
        id
    }

    fn push_recent(&mut self, id: NeuronId) {
        if self.recent_atoms.len() >= self.config.recent_atoms_window {
            self.recent_atoms.pop_front();
        }
        self.recent_atoms.push_back(id);
    }

    /// For each run of length 2..=max ending at the most recent atom,
    /// bump that run's count.  When a run crosses the emergence threshold,
    /// promote it to a concept neuron.  Runs longer than `max` would
    /// produce overly-specific concepts (memorize one phrase verbatim);
    /// the cap keeps emergent concepts useful.
    fn check_concept_emergence(&mut self, tick: u64) {
        let buf_len = self.recent_atoms.len();
        if buf_len < 2 { return; }

        let max_len = self.config.max_concept_member_count.min(buf_len);
        let threshold = self.config.concept_emergence_threshold;
        let mut to_promote: Vec<SequenceFingerprint> = Vec::new();

        for len in 2..=max_len {
            let start = buf_len - len;
            let run: SequenceFingerprint = self.recent_atoms
                .iter()
                .skip(start)
                .copied()
                .collect();
            let count = self.sequences.entry(run.clone()).or_insert(0);
            *count = count.saturating_add(1);
            if *count == threshold {
                to_promote.push(run);
            }
        }

        for run in to_promote {
            self.promote_to_concept(run, tick);
        }
    }

    fn promote_to_concept(&mut self, members: SequenceFingerprint, tick: u64) {
        // Composite label = concatenation of member labels, separated by a
        // glyph that can't appear in base64-url-safe payloads.  Stable and
        // human-readable for debugging.
        let composite_label: String = members.iter()
            .map(|id| self.neurons[*id as usize].label.as_str())
            .collect::<Vec<_>>()
            .join("~");
        if self.label_to_id.contains_key(&composite_label) {
            // Already promoted (e.g. via a different sequence path).  Skip.
            return;
        }

        // Stage 13 — atom-multiset dedup.  Two sequences with the same
        // atom-leaf multiset (e.g. [f,o,o,d] and [o,o,d,f], or
        // [a, pp-concept, l, e] and [a, p, p, l, e]) represent the
        // SAME word; round-robin training under destructive collapse
        // can spawn many such variant orderings.  Keep only the FIRST
        // (canonical) one — subsequent variants are dropped.
        //
        // The decoded byte sequence of the first-promoted concept
        // tends to be the linguistically correct one because it
        // emerged from a clean atom run before fragment collapse
        // disrupted the sequence.
        // Atom-leaf sequence in original order (used for the
        // periodicity check below).
        let leaves_seq: Vec<NeuronId> = self.expand_to_atom_leaves(&members);
        // Sorted copy used as the multiset key.
        let mut leaves: Vec<NeuronId> = leaves_seq.clone();
        leaves.sort();
        if self.concept_multiset_to_id.contains_key(&leaves) {
            return; // canonical concept already exists for this multiset
        }

        // Stage 13.1 — runaway-emergence guard part 1: refuse to
        // promote a concept whose member list contains the SAME
        // concept neuron more than once.  Catches the layer-2+
        // recursive tower [sport-concept, sport-concept, ...].
        let mut seen: ahash::AHashSet<NeuronId> = ahash::AHashSet::new();
        for &mid in &members {
            if let Some(n) = self.neurons.get(mid as usize) {
                if !n.is_atom() {
                    if !seen.insert(mid) {
                        return; // duplicate concept member — runaway pattern
                    }
                }
            }
        }

        // Note: emergence-time periodicity/ratio guards were
        // empirically tried and removed.  They blocked legitimate
        // K-12 concept emergence as collateral damage.  The
        // runaway-concept issue is now addressed at READ time in
        // `Brain::integrate_concept_first` via a sanity filter on
        // the candidate concept's decoded byte length.  This keeps
        // emergence flexible enough for real vocabulary while
        // preventing runaways from winning selection.
        let id = self.neurons.len() as NeuronId;
        let member_refs: Vec<NeuronRef> = members.iter()
            .map(|m| NeuronRef::new(self.config.id, *m))
            .collect();
        let mut concept = Neuron::new_concept(
            id, composite_label.clone(), NeuronKind::Excitatory, member_refs, tick,
        );
        // Wire member→concept terminals (atom→concept bottom-up) and
        // concept→member (concept→atom top-down).  Both Hebbian-strengthen
        // on subsequent activations.
        for &mid in &members {
            let target = NeuronRef::new(self.config.id, id);
            if let Some(member_neuron) = self.neurons.get_mut(mid as usize) {
                member_neuron.reinforce_terminal(target, 0.5, tick, self.config.max_weight);
            }
            concept.reinforce_terminal(
                NeuronRef::new(self.config.id, mid),
                0.5, tick, self.config.max_weight,
            );
        }
        self.neurons.push(concept);
        self.label_to_id.insert(composite_label, id);
        self.concept_multiset_to_id.insert(leaves, id);
    }
}
