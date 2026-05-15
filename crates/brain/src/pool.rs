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
}

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

    pub fn activation(&self, id: NeuronId) -> f32 {
        self.activation.get(&id).copied().unwrap_or(0.0)
    }

    /// Atomize a sensor frame and fire the corresponding atom neurons.
    /// Creates atoms for previously-unseen labels (neurogenesis).  Returns
    /// the IDs that fired, in order — the caller (Fabric) uses these to
    /// update the moment buffer.
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
        }

        // Sequence-fingerprint accounting + concept-emergence detection.
        // We look back through the recent-atoms window for runs that end
        // at the latest atom; each run gets its count incremented.
        self.check_concept_emergence(tick);
        fired
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
    }
}
