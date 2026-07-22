//! Pool primitive per [`ARCHITECTURE.md`] §3.1 and §4.A.
//!
//! A pool is a collection of neurons sharing a sensor resolution and an
//! atomization contract.  Atoms enter via [`Pool::observe_frame`]; concept
//! emergence happens automatically as repeating sequences accumulate in
//! `recent_atoms`.  The pool itself doesn't reach across pool boundaries —
//! cross-pool wiring is a [`crate::Fabric`] responsibility because it
//! requires the moment buffer (what fired in every pool at tick T).

use ahash::{AHashMap, AHashSet};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use crate::neuron::{Neuron, NeuronId, NeuronKind, NeuronRef, PoolId};
use crate::store::posting_index::{self, PostingIndexBuilder};
use crate::store::{
    AuxiliaryRecordRef, ColdTier, CountingBloom, NeuronStore, NoopStore, Store, TieredStore,
    WalEvent, WbrainFile, WbrainNeuronStore,
};

const CONCEPT_GENERATION_DIRECTORY_MAGIC: &[u8; 8] = b"W1ZCGEN1";
const CONCEPT_GENERATION_DIRECTORY_KIND: u32 = 0x4347_454E; // "CGEN"
const SEQUENCE_GENERATION_DIRECTORY_MAGIC: &[u8; 8] = b"W1ZSGEN1";
const SEQUENCE_GENERATION_DIRECTORY_KIND: u32 = 0x5347_454E; // "SGEN"
const CONCEPT_SEQUENCE_KEY: u8 = 0;
const CONCEPT_ATOM_LEAVES_KEY: u8 = 1;
const SEQUENCE_RECURRENCE_KEY: u8 = 2;

/// Stable neuron address space with independently resident bodies.
///
/// Each sleeping slot is one nullable pointer rather than a zeroed
/// [`Neuron`]. A body is present only while that exact neuron participates in
/// inference, training, or bounded maintenance. Dense, legacy brains still
/// occupy every slot and retain their historical behavior.
#[derive(Clone)]
enum NeuronSlots {
    Dense {
        slots: Vec<Option<Box<Neuron>>>,
        kinds: Vec<NeuronKind>,
        concepts: Vec<bool>,
        born_ticks: Vec<u64>,
    },
    Paged {
        slot_count: usize,
        concept_count: usize,
        residents: AHashMap<NeuronId, Box<Neuron>>,
    },
}

impl Default for NeuronSlots {
    fn default() -> Self {
        Self::Dense {
            slots: Vec::new(),
            kinds: Vec::new(),
            concepts: Vec::new(),
            born_ticks: Vec::new(),
        }
    }
}

impl NeuronSlots {
    fn new() -> Self {
        Self::default()
    }

    fn from_dense(neurons: Vec<Neuron>) -> Self {
        let kinds = neurons.iter().map(|neuron| neuron.kind).collect();
        let concepts = neurons.iter().map(|neuron| !neuron.is_atom()).collect();
        let born_ticks = neurons.iter().map(|neuron| neuron.born_tick).collect();
        Self::Dense {
            slots: neurons
                .into_iter()
                .map(|neuron| Some(Box::new(neuron)))
                .collect(),
            kinds,
            concepts,
            born_ticks,
        }
    }

    fn sleeping(slot_count: usize, concept_count: usize) -> Self {
        Self::Paged {
            slot_count,
            concept_count,
            residents: AHashMap::new(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense { slots, .. } => slots.len(),
            Self::Paged { slot_count, .. } => *slot_count,
        }
    }

    fn resident_count(&self) -> usize {
        match self {
            Self::Dense { slots, .. } => slots.iter().filter(|slot| slot.is_some()).count(),
            Self::Paged { residents, .. } => residents.len(),
        }
    }

    fn resident_ids(&self) -> Vec<NeuronId> {
        let mut ids: Vec<NeuronId> = match self {
            Self::Dense { slots, .. } => slots
                .iter()
                .enumerate()
                .filter_map(|(id, slot)| slot.as_ref().map(|_| id as NeuronId))
                .collect(),
            Self::Paged { residents, .. } => residents.keys().copied().collect(),
        };
        ids.sort_unstable();
        ids
    }

    fn get(&self, index: usize) -> Option<&Neuron> {
        match self {
            Self::Dense { slots, .. } => slots.get(index)?.as_deref(),
            Self::Paged { residents, .. } => residents.get(&(index as NeuronId)).map(Box::as_ref),
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Neuron> {
        match self {
            Self::Dense { slots, .. } => slots.get_mut(index)?.as_deref_mut(),
            Self::Paged { residents, .. } => {
                residents.get_mut(&(index as NeuronId)).map(Box::as_mut)
            }
        }
    }

    fn push(&mut self, neuron: Neuron) {
        match self {
            Self::Dense {
                slots,
                kinds,
                concepts,
                born_ticks,
            } => {
                kinds.push(neuron.kind);
                concepts.push(!neuron.is_atom());
                born_ticks.push(neuron.born_tick);
                slots.push(Some(Box::new(neuron)));
            }
            Self::Paged {
                slot_count,
                concept_count,
                residents,
            } => {
                debug_assert_eq!(neuron.id as usize, *slot_count);
                if !neuron.is_atom() {
                    *concept_count += 1;
                }
                *slot_count += 1;
                residents.insert(neuron.id, Box::new(neuron));
            }
        }
    }

    fn set(&mut self, index: usize, neuron: Neuron) {
        match self {
            Self::Dense {
                slots,
                kinds,
                concepts,
                born_ticks,
            } => {
                kinds[index] = neuron.kind;
                concepts[index] = !neuron.is_atom();
                born_ticks[index] = neuron.born_tick;
                slots[index] = Some(Box::new(neuron));
            }
            Self::Paged {
                slot_count,
                concept_count,
                residents,
            } => {
                let prior_concept = residents
                    .get(&(index as NeuronId))
                    .is_some_and(|prior| !prior.is_atom());
                let next_concept = !neuron.is_atom();
                if prior_concept != next_concept {
                    if next_concept {
                        *concept_count += 1;
                    } else {
                        *concept_count = concept_count.saturating_sub(1);
                    }
                }
                *slot_count = (*slot_count).max(index + 1);
                residents.insert(index as NeuronId, Box::new(neuron));
            }
        }
    }

    fn sleep(&mut self, index: usize) -> Option<Neuron> {
        match self {
            Self::Dense { slots, .. } => slots.get_mut(index)?.take().map(|neuron| *neuron),
            Self::Paged { residents, .. } => {
                residents.remove(&(index as NeuronId)).map(|neuron| *neuron)
            }
        }
    }

    fn discard_paged_residents(&mut self) -> Option<usize> {
        match self {
            Self::Dense { .. } => None,
            Self::Paged { residents, .. } => {
                let discarded = residents.len();
                *residents = AHashMap::new();
                Some(discarded)
            }
        }
    }

    fn wake(&mut self, index: usize, neuron: Neuron) -> bool {
        match self {
            Self::Dense { slots, .. } => {
                let Some(slot) = slots.get_mut(index) else {
                    return false;
                };
                *slot = Some(Box::new(neuron));
                true
            }
            Self::Paged {
                slot_count,
                residents,
                ..
            } => {
                if index >= *slot_count {
                    return false;
                }
                residents.insert(index as NeuronId, Box::new(neuron));
                true
            }
        }
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &Neuron> + '_> {
        match self {
            Self::Dense { slots, .. } => Box::new(slots.iter().filter_map(|slot| slot.as_deref())),
            Self::Paged { residents, .. } => Box::new(residents.values().map(Box::as_ref)),
        }
    }

    fn iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut Neuron> + '_> {
        match self {
            Self::Dense { slots, .. } => {
                Box::new(slots.iter_mut().filter_map(|slot| slot.as_deref_mut()))
            }
            Self::Paged { residents, .. } => Box::new(residents.values_mut().map(Box::as_mut)),
        }
    }

    fn dense_clone(&self) -> Vec<Neuron> {
        assert_eq!(
            self.resident_count(),
            self.len(),
            "legacy snapshot requires every neuron body to be resident"
        );
        let mut neurons: Vec<_> = self.iter().cloned().collect();
        neurons.sort_unstable_by_key(|neuron| neuron.id);
        neurons
    }

    fn kinds(&self) -> &[NeuronKind] {
        match self {
            Self::Dense { kinds, .. } => kinds,
            Self::Paged { .. } => &[],
        }
    }

    fn concept_flags(&self) -> &[bool] {
        match self {
            Self::Dense { concepts, .. } => concepts,
            Self::Paged { .. } => &[],
        }
    }

    fn born_ticks(&self) -> &[u64] {
        match self {
            Self::Dense { born_ticks, .. } => born_ticks,
            Self::Paged { .. } => &[],
        }
    }

    fn concept_count(&self) -> usize {
        match self {
            Self::Dense { concepts, .. } => {
                concepts.iter().filter(|is_concept| **is_concept).count()
            }
            Self::Paged { concept_count, .. } => *concept_count,
        }
    }

    fn is_concept(&self, id: NeuronId) -> bool {
        match self {
            Self::Dense { concepts, .. } => concepts.get(id as usize).copied().unwrap_or(false),
            Self::Paged { residents, .. } => {
                residents.get(&id).is_some_and(|neuron| !neuron.is_atom())
            }
        }
    }
}

impl Serialize for NeuronSlots {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.resident_count() != self.len() {
            return Err(serde::ser::Error::custom(
                "legacy snapshot requires every neuron body to be resident",
            ));
        }
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for neuron in self.iter() {
            sequence.serialize_element(neuron)?;
        }
        sequence.end()
    }
}

impl Index<usize> for NeuronSlots {
    type Output = Neuron;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .expect("attempted direct access to a sleeping neuron")
    }
}

impl IndexMut<usize> for NeuronSlots {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("attempted direct mutable access to a sleeping neuron")
    }
}

impl<'a> IntoIterator for &'a NeuronSlots {
    type Item = &'a Neuron;
    type IntoIter = Box<dyn Iterator<Item = &'a Neuron> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut NeuronSlots {
    type Item = &'a mut Neuron;
    type IntoIter = Box<dyn Iterator<Item = &'a mut Neuron> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

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
        frame
            .iter()
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
            if let Some(payload) = label
                .strip_prefix(self.prefix)
                .and_then(|s| s.strip_prefix(':'))
            {
                if let Ok(bytes) = engine.decode(payload) {
                    out.extend_from_slice(&bytes);
                }
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        "byte-passthrough"
    }
}

/// Substrate-internal signal that can drive a knob.  These are the
/// observables the pool tracks every tick.  Genes pick WHICH signal
/// drives WHICH knob, not the knob's static value — the dynamical-
/// system interpretation of evolutionary search.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ControlSignal {
    /// EMA of unpredicted-firing fraction.  Range ~[0, 1].  High =
    /// novel input.
    Surprise,
    /// 1 - Surprise.  Range ~[0, 1].  High = predicted / stable.
    InvSurprise,
    /// EMA of firing-set size, normalised against neuron_count.
    /// Range ~[0, 0.5] typically.  High = dense firing.
    FiringRate,
    /// 1 - FiringRate (clamped).  Range ~[0, 1].  High = sparse firing.
    InvFiringRate,
    /// EMA of decode_best_trained_binding's winning atom_score for
    /// queries against this pool.  Range ~[0, 1].  High = retrieval
    /// landing on confident bindings.
    DecodePrecisionEma,
    /// 1 - DecodePrecisionEma.  High = retrieval struggling.
    InvDecodePrecisionEma,
    /// EMA of concept_count.  Range potentially huge — normalised
    /// by `concept_count_norm` parameter in DrivenBy.
    ConceptCountEma,
    /// EMA of terminal count.  High = dense connectivity.
    TerminalCountEma,
}

/// How a knob's effective value is computed each tick.  Genome
/// encodes one ControlMode per knob; the GA explores wirings
/// (which signal × scale × offset).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ControlMode {
    /// Static value — backward-compatible default.
    Constant(f32),
    /// effective = clamp(min, max, offset + scale * signal_value()).
    DrivenBy {
        signal: ControlSignal,
        scale: f32,
        offset: f32,
        min: f32,
        max: f32,
    },
}

impl ControlMode {
    /// Compute the effective value given current pool ControlState.
    pub fn evaluate(&self, st: &ControlState) -> f32 {
        match self {
            ControlMode::Constant(v) => *v,
            ControlMode::DrivenBy {
                signal,
                scale,
                offset,
                min,
                max,
            } => {
                let raw = match signal {
                    ControlSignal::Surprise => st.surprise,
                    ControlSignal::InvSurprise => 1.0 - st.surprise,
                    ControlSignal::FiringRate => st.firing_rate,
                    ControlSignal::InvFiringRate => 1.0 - st.firing_rate,
                    ControlSignal::DecodePrecisionEma => st.decode_precision,
                    ControlSignal::InvDecodePrecisionEma => 1.0 - st.decode_precision,
                    ControlSignal::ConceptCountEma => st.concept_count_norm,
                    ControlSignal::TerminalCountEma => st.terminal_count_norm,
                };
                (offset + scale * raw).clamp(*min, *max)
            }
        }
    }
}

/// Snapshot of one pool's observable signals for ControlMode evaluation.
/// Materialised once per tick / per-knob-read, not stored — small.
#[derive(Debug, Clone, Copy)]
pub struct ControlState {
    pub surprise: f32,            // [0, 1] from recent_surprise EMA
    pub firing_rate: f32,         // [0, 1] normalised
    pub decode_precision: f32,    // [0, 1] from decode_precision_ema
    pub concept_count_norm: f32,  // log10(concept_count+1)/4 ≈ [0, ~1]
    pub terminal_count_norm: f32, // log10(terminal_count+1)/7 ≈ [0, ~1]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub name: String,
    pub id: PoolId,
    pub frame_rate: u32,
    pub recent_atoms_window: usize,
    /// A sequence of length 2..=`max_concept_member_count` that recurs at
    /// least `concept_emergence_threshold` times in `recent_atoms` is
    /// promoted to a concept.  Per spec §4.A.
    pub max_concept_member_count: usize,
    pub concept_emergence_threshold: u32,
    pub max_weight: f32,
    pub decay_rate: f32,
    pub prune_floor: f32,
    pub plasticity_baseline: f32,

    /// k-WTA sparsity target — DYNAMICAL.  ControlMode driven by a
    /// substrate-internal signal each tick.  Default is
    /// Constant(1.0) (no-op).  When DrivenBy is selected, e.g.
    /// `DrivenBy { signal: InvSurprise, scale: 0.7, offset: 0.3 }`,
    /// sparsity adapts: stable / predicted firings get sparsified;
    /// novel / surprising firings keep more atoms active.
    /// Biological motivation: cortical inhibitory interneurons
    /// modulate sparsity dynamically with input regularity, not
    /// statically.
    #[serde(default = "default_sparsity_mode")]
    pub sparsity_mode: ControlMode,
    /// Minimum number of neurons that stay firing after the k-WTA
    /// gate even when `sparsity_top_k_frac` would round to fewer.
    /// Prevents complete silence in low-traffic pools.
    #[serde(default = "default_sparsity_min_neurons")]
    pub sparsity_min_neurons: usize,

    /// Heterosynaptic long-term depression — DYNAMICAL.  ControlMode
    /// driven by substrate state.  Constant(0.0) default = disabled.
    /// When DrivenBy is selected, ratio adapts: high terminal density
    /// could drive higher LTD; high decode precision could lower it
    /// (don't weaken what's working).
    #[serde(default = "default_heterosynaptic_ltd_mode")]
    pub heterosynaptic_ltd_mode: ControlMode,

    /// Predictive-coding gate — DYNAMICAL.  Constant(0.0) default
    /// = always emerge.  When DrivenBy(InvSurprise, ...) is selected,
    /// the gate self-tightens when surprise drops (substrate becomes
    /// stable → emergence pauses), and self-loosens when surprise
    /// rises (novel input → emergence resumes).
    #[serde(default = "default_predict_gate_mode")]
    pub predict_gate_mode: ControlMode,
}

/// A parallel, lossy code sensor that emits structural events while the raw
/// byte pool remains the authoritative source representation. Identifiers are
/// deliberately represented by grammatical role rather than spelling, so a
/// renamed function can reinstate the same learned structural concept. This
/// is a feature extractor, not a replacement tokenizer or a decoder.
pub struct CodeStructureEncoding {
    pub prefix: String,
}

/// Parallel intent sensor for natural-language coding instructions. It emits
/// sparse semantic diagnostics while the raw instruction pool still carries
/// every character. These are feature labels grounded by co-firing with raw
/// atoms, source structure, execution failure, and the eventual repair.
pub struct InstructionIntentEncoding {
    pub prefix: String,
}

impl AtomEncoding for InstructionIntentEncoding {
    fn atomize(&self, frame: &[u8]) -> Vec<String> {
        let text = String::from_utf8_lossy(frame).to_ascii_lowercase();
        // Internal learned-route frames re-stimulate semantic atoms that were
        // previously grounded by raw character streams. This is not exposed
        // as user tokenization: labels are target-pool observations learned
        // through the same co-firing binding path as any other modality.
        if text.contains("@intent:") {
            let mut labels = Vec::new();
            for line in text.lines() {
                let Some((_, semantic)) = line.split_once("@intent:") else {
                    continue;
                };
                let semantic = semantic.trim().to_ascii_uppercase();
                if !semantic.is_empty()
                    && semantic
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ":_-".contains(ch))
                {
                    let label = format!("{}:{}", self.prefix, semantic);
                    if !labels.contains(&label) {
                        labels.push(label);
                    }
                }
            }
            return labels;
        }
        let mut features = Vec::new();
        let coding_context = [
            "fix ",
            "correct ",
            "change ",
            "repair ",
            "prevent ",
            "create ",
            "write ",
            "implement ",
            "build ",
            "produce ",
            "make ",
            "develop ",
            "code",
            "function",
            "compute",
            "calculate",
            "return",
        ]
        .iter()
        .any(|cue| text.contains(cue));
        if !coding_context {
            return features;
        }
        let mut emit = |feature: &str| {
            let label = format!("{}:{}", self.prefix, feature);
            if !features.contains(&label) {
                features.push(label);
            }
        };
        // Language is an independent feature rather than part of the task
        // label. Co-firing lets bindings represent, for example,
        // LANGUAGE:RUST + POWER_SELF:2 without discarding the raw words.
        if text.contains("python") {
            emit("LANGUAGE:PYTHON");
        }
        if text.contains("javascript") || text.contains("node.js") || text.contains("nodejs") {
            emit("LANGUAGE:JAVASCRIPT");
        }
        if text.contains("typescript") {
            emit("LANGUAGE:TYPESCRIPT");
        }
        if text.contains("c#")
            || text.contains("c sharp")
            || text.contains("dotnet")
            || text.contains(".net")
        {
            emit("LANGUAGE:CSHARP");
        }
        if text.contains("golang")
            || text.starts_with("go ")
            || text.contains(" go ")
            || text.contains("go function")
            || text.contains("go code")
        {
            emit("LANGUAGE:GO");
        }
        if text.contains("rust") {
            emit("LANGUAGE:RUST");
        }
        if text.contains("java") && !text.contains("javascript") {
            emit("LANGUAGE:JAVA");
        }
        if text.contains("cube") || text.contains("third power") || text.contains("three times") {
            emit("POWER_SELF:3");
        } else if text.contains("square")
            || text.contains("second power")
            || text.contains("product of its argument with itself")
            || text.contains("multiplies the value by itself")
        {
            emit("POWER_SELF:2");
        }
        if text.contains("negative")
            || text.contains("below zero")
            || text.contains("less than zero")
        {
            emit("COMPARISON:LESS_THAN_ZERO");
        }
        if text.contains("average") || text.contains("arithmetic mean") {
            emit("MATH:AVERAGE");
        }
        if text.contains("empty") || text.contains("no elements") {
            emit("GUARD:EMPTY_INPUT");
        }
        if text.contains("odd") || text.contains("odd parity") {
            emit("PARITY:ODD");
        }
        if text.contains("increment")
            || text.contains("increments")
            || text.contains("repeated words")
            || text.contains("increase their stored total")
            || (text.contains("word") && text.contains("count"))
        {
            emit("STATE:INCREMENT_COUNT");
        }
        if text.contains("uppercase") {
            emit("TEXT:UPPERCASE");
        }
        if text.contains("largest") || text.contains("maximum") {
            emit("ORDER:MAXIMUM");
        }
        if text.contains("factorial") {
            emit("MATH:FACTORIAL");
        }
        // Code-fragment composition features. These describe structural and
        // behavioral constraints in a parallel pool; the source bytes remain
        // wholly represented in the raw action pool.
        if text.contains("clamp") || text.contains("clamping") {
            emit("CODE:CLAMP");
        }
        if text.contains("function signature")
            || text.contains("declaration line")
            || ((text.contains("clamp") || text.contains("clamping")) && text.contains("function"))
        {
            emit("CODE:FUNCTION_SIGNATURE");
        }
        if text.contains("lower-bound")
            || text.contains("lower bound")
            || text.contains("below the minimum")
            || (text.contains("minimum") && text.contains("floor"))
        {
            emit("GUARD:LOWER_BOUND");
        }
        if text.contains("upper-bound")
            || text.contains("upper bound")
            || text.contains("above the maximum")
            || (text.contains("maximum") && text.contains("cap"))
        {
            emit("GUARD:UPPER_BOUND");
        }
        if text.contains("otherwise return the input")
            || text.contains("otherwise returns the input")
            || text.contains("otherwise return the original")
            || text.contains("otherwise returns the original")
            || text.contains("otherwise leaves the value unchanged")
            || text.contains("within the bounds")
        {
            emit("FLOW:RETURN_INPUT");
        }
        if text.contains("input validation")
            || text.contains("validate input")
            || text.contains("validates input")
            || text.contains("required field")
            || text.contains("required user")
            || (text.contains("validate")
                && (text.contains("order")
                    || text.contains("command")
                    || text.contains("request")
                    || text.contains("payload")))
        {
            emit("ENTERPRISE:INPUT_VALIDATION");
        }
        let no_retry = text.contains("no retries")
            || text.contains("without retries")
            || text.contains("without retry")
            || text.contains("fail immediately")
            || text.contains("first transient failure");
        if no_retry {
            emit("RESILIENCE:NO_RETRY");
        }
        if !no_retry
            && (text.contains("attempts")
                || text.contains("transient failure")
                || (text.contains("retry")
                    && (text.contains("async")
                        || text.contains("asynchronous")
                        || text.contains("bounded")
                        || text.contains("last exception")
                        || text.contains("after success"))))
        {
            emit("ENTERPRISE:BOUNDED_RETRY");
        }
        if text.contains("json")
            && (text.contains("aggregat") || text.contains("summar") || text.contains("totals"))
        {
            emit("ENTERPRISE:JSON_AGGREGATION");
        }
        if text.contains("redact")
            || text.contains("mask secrets")
            || text.contains("masks secrets")
            || text.contains("remove credentials")
            || text.contains("removing credentials")
            || text.contains("cleanses nested credentials")
            || ((text.contains("mask") || text.contains("masks")) && text.contains("credential"))
            || (text.contains("password") && text.contains("token"))
        {
            emit("ENTERPRISE:SECRET_REDACTION");
        }
        if text.contains("batch") || text.contains("chunk") {
            emit("ENTERPRISE:BATCHING");
        }
        if text.contains("multi-file")
            || text.contains("multiple files")
            || (text.contains("domain") && text.contains("service file"))
        {
            emit("ARCHITECTURE:MULTIFILE_SERVICE");
        }
        if text.contains("project")
            || text.contains("code in multiple files")
            || text.contains("code across multiple files")
        {
            emit("ARTIFACT:PROJECT");
        }
        if text.contains("inventory")
            || text.contains("stock reservation")
            || text.contains("reserve stock")
        {
            emit("DOMAIN:INVENTORY");
        }
        if text.contains("domain module") || text.contains("domain file") {
            emit("STRUCTURE:DOMAIN_MODULE");
        }
        if text.contains("service module")
            || text.contains("service file")
            || ((text.contains("multi-file")
                || text.contains("multiple files")
                || text.contains("code in multiple files"))
                && text.contains("service"))
        {
            emit("STRUCTURE:SERVICE_MODULE");
        }
        if text.contains("exception declaration") || text.contains("exception class") {
            emit("STRUCTURE:EXCEPTION");
        }
        if text.contains("import statement") || text.contains("import line") {
            emit("STRUCTURE:IMPORT");
        }
        // A requested named executable class is the same structural role as
        // a training episode phrased as a "service class".  Keeping those
        // phrases on separate atoms can remove the root of an otherwise
        // complete dependency chain while every behavioral child fires.
        if text.contains("service class")
            || text.contains("class named ")
            || text.contains("class called ")
        {
            emit("STRUCTURE:SERVICE_CLASS");
        }
        if text.contains("reserve method") || text.contains("reservation method") {
            emit("STRUCTURE:RESERVE_METHOD");
        }
        if text.contains("over-reservation")
            || text.contains("over reservation")
            || text.contains("insufficient stock")
            || ((text.contains("inventory") || text.contains("stock") || text.contains("reserv"))
                && (text.contains("safely") || text.contains("safe reservation")))
        {
            emit("GUARD:INSUFFICIENT_STOCK");
        }
        if text.contains("sqlite")
            || (text.contains("database") && text.contains("transaction"))
            || text.contains("atomic transfer")
            || text.contains("all-or-nothing database transfer")
            || (text.contains("all-or-nothing") && text.contains("transaction"))
            || (text.contains("restore")
                && (text.contains("inventory")
                    || text.contains("state")
                    || text.contains("capacity"))
                && (text.contains("commit") || text.contains("fulfillment")))
        {
            emit("PERSISTENCE:ATOMIC_TRANSACTION");
        }
        if text.contains("async")
            && (text.contains("concurr") || text.contains("semaphore") || text.contains("parallel"))
        {
            emit("CONCURRENCY:BOUNDED_ASYNC");
        }
        if text.contains("authoriz")
            || text.contains("access-control")
            || text.contains("access control")
            || text.contains("rbac")
            || text.contains("default-deny")
            || text.contains("default deny")
            || text.contains("denies by default")
            || (text.contains("superuser")
                && (text.contains("operation")
                    || text.contains("own")
                    || (text.contains("restrict") && text.contains("identity"))))
            || ((text.contains("administrator") || text.contains("admin"))
                && (text.contains("refused") || text.contains("own")))
        {
            emit("SECURITY:AUTHORIZATION");
        }
        if text.contains("idempoten")
            || (text.contains("api") && text.contains("replay"))
            || (text.contains("replay")
                && (text.contains("order key") || text.contains("request key"))
                && (text.contains("response") || text.contains("result")))
            || ((text.contains("repeating") || text.contains("resending"))
                && (text.contains("first result") || text.contains("same response"))
                && text.contains("order"))
            || ((text.contains("request key") || text.contains("command key"))
                && (text.contains("retr") || text.contains("first result")))
        {
            emit("API:IDEMPOTENT_COMMAND");
        }
        if text.contains("migration")
            || text.contains("schema version")
            || text.contains("upgrade path")
        {
            emit("PERSISTENCE:SCHEMA_MIGRATION");
        }
        if text.contains("observability")
            || text.contains("structured log")
            || text.contains("correlation id")
            || text.contains("correlation-id")
            || (text.contains("correlat") && text.contains("log"))
            || (text.contains("request trac")
                && (text.contains("event record") || text.contains("audit")))
            || (text.contains("request identifier")
                && (text.contains("audit") || text.contains("machine-readable")))
        {
            emit("OBSERVABILITY:CORRELATED_LOGGING");
        }
        if text.contains("circuit breaker")
            || text.contains("circuit-breaker")
            || (text.contains("opens after") && text.contains("cooldown"))
            || (text.contains("circuit") && text.contains("cooldown"))
            || (text.contains("circuit")
                && text.contains("open")
                && (text.contains("reset") || text.contains("failure")))
        {
            emit("RESILIENCE:CIRCUIT_BREAKER");
        }
        if text.contains("transactional outbox")
            || text.contains("transactional-outbox")
            || text.contains("outbox event")
            || text.contains("outbox-event")
        {
            emit("INTEGRATION:TRANSACTIONAL_OUTBOX");
        }
        if text.contains("deduplicat")
            || text.contains("duplicate work")
            || (text.contains("duplicate")
                && text.contains("work")
                && (text.contains("order") || text.contains("sku")))
        {
            emit("CONCURRENCY:DEDUPLICATION");
        }
        if !no_retry
            && (text.contains("async") || text.contains("asynchronous"))
            && text.contains("retry")
        {
            emit("RESILIENCE:ASYNC_RETRY");
        }
        if text.contains("optimistic concurr")
            || text.contains("optimistic-concurr")
            || text.contains("expected version")
            || text.contains("expected-version")
            || (text.contains("expected") && text.contains("version"))
            || text.contains("stale write")
            || text.contains("version-checked")
            || text.contains("version checked")
            || (text.contains("stale") && text.contains("writer"))
        {
            emit("STATE:OPTIMISTIC_CONCURRENCY");
        }
        if (text.contains("ledger") && text.contains("transfer"))
            || text.contains("all-or-nothing account transfer")
            || ((text.contains("account movement") || text.contains("move funds"))
                && text.contains("missing account")
                && text.contains("insufficient"))
        {
            // A ledger transfer is a specialized atomic transaction. Emit
            // both levels so broad rollback semantics and the specialized
            // account operation co-fire rather than becoming isolated atoms.
            emit("PERSISTENCE:ATOMIC_TRANSACTION");
            emit("DOMAIN:ATOMIC_LEDGER_TRANSFER");
            emit("ENTERPRISE:INPUT_VALIDATION");
        }
        if ((text.contains("debit") && text.contains("credit"))
            || text.contains("restores the debit"))
            && (text.contains("neither happens")
                || text.contains("cannot be completed")
                || text.contains("account is invalid"))
        {
            emit("PERSISTENCE:ATOMIC_TRANSACTION");
            emit("DOMAIN:ATOMIC_LEDGER_TRANSFER");
        }
        if text.contains("unspecified")
            || text.contains("unknown protocol")
            || (text.contains("protocol") && text.contains("unknown"))
            || (text.contains("schema") && text.contains("unknown"))
            || text.contains("undocumented")
            || text.contains("has not been provided")
            || text.contains("have not been provided")
            || text.contains("not provided")
            || text.contains("without any service objectives")
            || (text.contains("without any")
                && (text.contains("requirements")
                    || text.contains("objectives")
                    || text.contains("metrics")))
        {
            emit("GROUNDING:UNDERSPECIFIED");
        }
        features
    }

    fn reassemble(&self, active_atoms: &[(&str, f32)]) -> Vec<u8> {
        let prefix = format!("{}:", self.prefix);
        let mut output = Vec::new();
        for (label, activation) in active_atoms {
            if *activation <= 0.0 {
                continue;
            }
            if let Some(semantic) = label.strip_prefix(&prefix) {
                output.extend_from_slice(b"@intent:");
                output.extend_from_slice(semantic.as_bytes());
                output.push(b'\n');
            }
        }
        output
    }
    fn name(&self) -> &'static str {
        "instruction-intent"
    }
}

impl AtomEncoding for CodeStructureEncoding {
    fn atomize(&self, frame: &[u8]) -> Vec<String> {
        let text = String::from_utf8_lossy(frame);
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        let mut after_def = false;
        let mut in_params = false;
        let mut line_start = true;
        let mut parameter_names = AHashSet::new();
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'\n' {
                out.push(format!("{}:NEWLINE", self.prefix));
                line_start = true;
                i += 1;
                continue;
            }
            if b.is_ascii_whitespace() {
                let start = i;
                while i < bytes.len() && bytes[i] != b'\n' && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if line_start {
                    out.push(format!("{}:INDENT:{}", self.prefix, i - start));
                }
                continue;
            }
            line_start = false;
            if b.is_ascii_alphabetic() || b == b'_' {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &text[start..i];
                let feature = match word {
                    "def" | "return" | "if" | "else" | "for" | "in" | "while" | "class" | "try"
                    | "except" | "raise" | "import" | "from" => {
                        after_def = word == "def";
                        format!("KW:{}", word)
                    }
                    "sum" | "len" | "range" | "enumerate" | "zip" | "str" | "int" | "float"
                    | "list" | "dict" | "set" => format!("BUILTIN:{}", word),
                    _ if after_def => {
                        after_def = false;
                        "FUNCTION_NAME".to_string()
                    }
                    _ if in_params => {
                        parameter_names.insert(word.to_string());
                        "PARAMETER_DECL".to_string()
                    }
                    _ if parameter_names.contains(word) => "PARAMETER_REF".to_string(),
                    _ => "IDENTIFIER".to_string(),
                };
                out.push(format!("{}:{}", self.prefix, feature));
                continue;
            }
            if b.is_ascii_digit() {
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                out.push(format!("{}:NUMBER", self.prefix));
                continue;
            }
            let feature: String = match b {
                b'(' => {
                    in_params =
                        after_def || out.last().is_some_and(|v| v.ends_with(":FUNCTION_NAME"));
                    "LPAREN".into()
                }
                b')' => {
                    in_params = false;
                    "RPAREN".into()
                }
                b'[' => "LBRACKET".into(),
                b']' => "RBRACKET".into(),
                b'{' => "LBRACE".into(),
                b'}' => "RBRACE".into(),
                b':' => "COLON".into(),
                b',' => "COMMA".into(),
                b'.' => "DOT".into(),
                b'\'' | b'\"' => "QUOTE".into(),
                b'+' | b'-' | b'*' | b'/' | b'%' | b'<' | b'>' | b'=' | b'!' => {
                    let start = i;
                    i += 1;
                    while i < bytes.len() && matches!(bytes[i], b'=' | b'<' | b'>' | b'/' | b'*') {
                        i += 1;
                    }
                    out.push(format!("{}:OP:{}", self.prefix, &text[start..i]));
                    continue;
                }
                _ => "PUNCT".into(),
            };
            out.push(format!("{}:{}", self.prefix, feature));
            i += 1;
        }
        out
    }

    fn reassemble(&self, _active_atoms: &[(&str, f32)]) -> Vec<u8> {
        Vec::new()
    }
    fn name(&self) -> &'static str {
        "code-structure"
    }
}

/// Result of deterministic batch pretraining. Every promoted pattern is an
/// ordinary concept neuron whose members are the original atom neurons; no
/// alternate vocabulary or opaque token identity is introduced.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PretrainReport {
    pub frames: usize,
    pub atoms_observed: usize,
    pub recurring_candidates: usize,
    pub concepts_promoted: usize,
}

fn default_sparsity_mode() -> ControlMode {
    ControlMode::Constant(1.0)
}
fn default_sparsity_min_neurons() -> usize {
    1
}
fn default_heterosynaptic_ltd_mode() -> ControlMode {
    ControlMode::Constant(0.0)
}
fn default_predict_gate_mode() -> ControlMode {
    ControlMode::Constant(0.0)
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
            sparsity_mode: ControlMode::Constant(1.0),
            sparsity_min_neurons: 1,
            heterosynaptic_ltd_mode: ControlMode::Constant(0.0),
            predict_gate_mode: ControlMode::Constant(0.0),
        }
    }
}

/// Track of which atom-sequences have recurred and how many times.  When a
/// sequence's count crosses `concept_emergence_threshold`, the pool promotes
/// it to a concept neuron whose members are those atom IDs in observed
/// firing order.  Once promoted, the sequence is cleared from the tracker
/// (the concept neuron now carries the identity).
type SequenceFingerprint = Vec<NeuronId>;

#[derive(Serialize, serde::Deserialize)]
struct WbrainPoolMetadata {
    config: PoolConfig,
    recent_atoms: VecDeque<NeuronId>,
    sequences: Vec<(SequenceFingerprint, u32)>,
    legacy_sequence_ledger: Option<AuxiliaryRecordRef>,
    concept_multiset_to_id: Vec<(Vec<NeuronId>, NeuronId)>,
    concept_sequence_to_id: Vec<(Vec<NeuronId>, NeuronId)>,
    legacy_concept_sequence_index: Option<AuxiliaryRecordRef>,
    neuron_kinds: Vec<NeuronKind>,
    concept_slots: Vec<bool>,
    born_ticks: Vec<u64>,
    concept_count: usize,
    total_terminals: usize,
}

pub(crate) struct StreamedPoolMetadata {
    pub config: PoolConfig,
    pub recent_atoms: VecDeque<NeuronId>,
    pub sequences: Vec<(SequenceFingerprint, u32)>,
    pub legacy_sequence_ledger: Option<AuxiliaryRecordRef>,
    pub concept_sequence_to_id: Vec<(Vec<NeuronId>, NeuronId)>,
    pub legacy_concept_sequence_index: Option<AuxiliaryRecordRef>,
    pub neuron_kinds: Vec<NeuronKind>,
    pub concept_slots: Vec<bool>,
    pub born_ticks: Vec<u64>,
    pub concept_count: usize,
    pub total_terminals: usize,
}

pub struct Pool {
    pub config: PoolConfig,
    encoding: Box<dyn AtomEncoding>,
    neurons: NeuronSlots,
    label_to_id: AHashMap<String, NeuronId>,
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

    /// Ordered-sequence index for concepts.  Key is the `members` vec
    /// in firing order — distinguishes "ABC" from "CBA" (positional
    /// concepts), whereas `concept_multiset_to_id` deduplicates
    /// permutations.  Maintained alongside `label_to_id` so the
    /// observe hot path can call collapse_tail_to_concept WITHOUT
    /// building a composite label string (which used to dominate
    /// per-atom cost — string allocation + hash on tens of chars vs.
    /// hashing a Vec<u32> directly).
    ///
    /// Populated from `members` of every concept on snapshot restore
    /// (Pool::from_snapshot), then maintained in lockstep with
    /// `label_to_id` on every promote_to_concept.
    concept_sequence_to_id: AHashMap<Vec<NeuronId>, NeuronId>,
    /// Exact ordered concept membership retained as a disk hash index for
    /// migrated brains. Only concepts created or touched after reopening
    /// enter the live map above.
    legacy_concept_sequence_index: Option<AuxiliaryRecordRef>,
    /// Streaming buffer of recently-fired atom/concept IDs.  Drives concept
    /// emergence.  Bounded by `config.recent_atoms_window`.
    recent_atoms: VecDeque<NeuronId>,
    /// Per-sequence recurrence count.  Concept emergence fires when a
    /// sequence's count crosses the threshold.
    sequences: AHashMap<SequenceFingerprint, u32>,
    /// Cold, exact legacy recurrence state retained inside `.wbrain`.
    /// Lookups are routed through the on-disk index rather than hydrating
    /// every historical candidate sequence at startup.
    legacy_sequence_ledger: Option<AuxiliaryRecordRef>,
    /// IDs of neurons currently firing (activation > min_activation).
    /// Rebuilt every observe call.  Read by the Fabric for the moment
    /// buffer and cross-pool wiring.
    currently_firing: AHashSet<NeuronId>,
    /// Per-neuron transient activation for the current tick.  Cleared at
    /// the start of each observe call.
    activation: AHashMap<NeuronId, f32>,

    /// EMA of "surprise" — fraction of firing atoms that were NOT
    /// in the previous tick's intra-pool terminal-target prediction.
    /// Drives the predictive-coding gate on concept emergence.
    /// Cleared on new(); transient runtime state (not serialised).
    recent_surprise: f32,

    /// Last-observed atom sequence (ordered, no concepts).  Rebuilt
    /// on every observe_frame call.  Lets the decoder distinguish
    /// anagram queries: 'sad' query has last_observed_sequence
    /// [s,a,d] while 'das' query has [d,a,s].  Transient runtime
    /// state (not serialised).
    last_observed_sequence: Vec<NeuronId>,

    /// EMA of `currently_firing.len()` post-k-WTA — driven by every
    /// observe.  Used as a ControlSignal: sparsity controllers can
    /// read this to drive their own thresholds.
    firing_rate_ema: f32,

    /// EMA of post-emergence concept_count.  Tracks how aggressively
    /// the pool is crystallising new concepts.  Rising fast =
    /// concept-of-concept runaway signal.
    concept_count_ema: f32,

    /// EMA of total terminal count across all neurons.  Tracks
    /// synaptic density.  High terminal count + small neuron count
    /// = dense connectivity (high-LTD candidate).
    terminal_count_ema: f32,

    /// EXACT total terminal count across all neurons in this pool.
    /// Maintained incrementally on every terminal mutation:
    ///   + Neuron::reinforce_terminal returning true (new terminal added)
    ///   - Neuron::decay_and_prune returning N pruned
    ///   - prune_weak_concepts zeroing terminals
    ///   - terminals.clear() during concept emergence
    /// Reading this is O(1) — replaces the O(N) `iter().map(|n|
    /// n.terminals.len()).sum()` that used to live in brain.stats() and
    /// in pool.update_emas() and that turned every /stats call into a
    /// full-fabric scan.
    pub(crate) total_terminals: usize,

    /// Pending concept-emergence promotions deferred from observe_frame
    /// to the next sleep cycle.  When W1Z4RD_DEFER_PROMOTION=1 (default
    /// in current builds), check_concept_emergence enqueues sequence
    /// fingerprints here instead of calling promote_to_concept inline.
    /// This makes /observe a pure write path — atom firing + terminal
    /// reinforcement only — and moves structure growth (neuron
    /// allocation, member↔concept terminal wiring, label/bloom/multiset
    /// registration) to /sleep, matching the CLS biological model
    /// (Kumaran 2016, McClelland-McNaughton-O'Reilly 1995).
    ///
    /// Pool isn't serde-derived (PoolSnapshot is the persisted form),
    /// so this field is naturally transient — every fresh load starts
    /// with an empty queue.  A pending promotion in flight at snapshot
    /// time will simply re-accumulate its threshold count on next pass.
    pub(crate) pending_promotions: Vec<SequenceFingerprint>,

    /// Domain id stamped onto every atom + concept created from now
    /// on.  Default 0 means "global / unassigned domain" (current
    /// behaviour).  Set by the operator via /set_domain before
    /// training a specific corpus so the resulting concepts cluster
    /// into the island that corpus represents.
    ///
    /// Transient — not part of the snapshot, defaults back to 0 on
    /// restart so the brain doesn't accidentally keep tagging new
    /// material with a stale domain.
    pub(crate) domain_for_new: u32,

    /// Persistence backend per [`ARCHITECTURE.md`] §17.9.  Default is a
    /// [`NoopStore`] — pools constructed without an explicit store stay
    /// purely in-memory.  When the brain plugs in an [`crate::store::MmapWalStore`]
    /// via [`Pool::set_store`], every mutation that affects durable
    /// state (atom creation, concept emergence, terminal pruning) is
    /// appended to the WAL before the in-memory mutation is exposed.
    /// `Arc<dyn Store>` so it can be cheaply shared across all pools.
    store: Arc<dyn Store>,

    /// Counting Bloom filter over `label_to_id` keys per
    /// [`ARCHITECTURE.md`] §17.3.  Maintained alongside the HashMap.
    /// Stage 17.3: used as a fast probabilistic existence pre-check;
    /// the HashMap is still authoritative.  When Stage 17.4 full ships
    /// the demand-paged loader, this filter answers "is this label
    /// maybe on disk?" without seeking, short-circuiting cold misses.
    bloom: CountingBloom,

    /// Cold-tier file for evicted neurons per [`ARCHITECTURE.md`] §17.4.
    /// `None` means eviction is disabled — the pool keeps every neuron
    /// in RAM (the legacy mode that all small-scale tests use).  Attach
    /// via [`Pool::set_cold_tier`].
    cold_tier: Option<Arc<ColdTier>>,

    /// Stage 18.12 step 4b: §18 distributed storage hook.  When set,
    /// the eviction/page-in path routes through this `NeuronStore`
    /// instead of the legacy `cold_tier`.  Typically a `TieredStore`
    /// composing local RAM + local cold disk + zero-or-more remote
    /// peer stores.  When `None` (the default), the pool falls back
    /// to legacy `cold_tier` semantics — zero behavioural change for
    /// existing callers.
    ///
    /// Attached via [`Pool::set_tiered_store`].  Per [`ARCHITECTURE.md`]
    /// §18.2 this is the abstraction below `Pool::neurons` that lets
    /// the brain transparently use storage from multiple hosts.
    tiered_store: Option<Arc<TieredStore>>,

    /// Single-file neuron-addressable backing store. Unlike the legacy cold
    /// tier, this is authoritative for every neuron kind, including atoms.
    /// The in-memory `neurons` vector then contains either the one live body
    /// or a compact routing slot for that stable id.
    wbrain_store: Option<Arc<WbrainNeuronStore>>,
    /// Read-only inference stubs loaded from the shape prefix of a `.wbrain`
    /// record. These carry identity/label/members but deliberately omit the
    /// potentially enormous terminal payload. A full-access request upgrades
    /// the stub through `ensure_loaded`; request cleanup drops it outright.
    shape_only_residents: AHashSet<NeuronId>,

    /// Disk offsets for currently-evicted neurons.  An ID in this map
    /// is one whose in-RAM slot has been zeroed out (terminals + members
    /// cleared) and whose authoritative state lives at this byte offset
    /// in `cold_tier`'s file.  Empty if eviction is disabled.
    cold_offsets: AHashMap<NeuronId, u64>,

    /// Set of neuron IDs currently in the evicted state.  Maintained
    /// alongside `cold_offsets` (membership is identical); kept as a
    /// separate set for O(1) "is this evicted?" probes from iteration
    /// + decode paths.
    evicted: AHashSet<NeuronId>,

    /// External signal — set by Brain::decode_best_trained_binding
    /// when this pool is the query_pool of a decode.  Rolling avg
    /// of the winning binding's atom_score (the precision×recall
    /// produced).  Used as a ControlSignal so decode-time floor
    /// adapters can read it.
    pub decode_precision_ema: f32,
}

/// Borrowed, wire-compatible view of [`crate::persistence::PoolSnapshot`].
///
/// `Pool::snapshot()` must remain available for tests and callers that need
/// an owned value, but production checkpoints must not clone every neuron's
/// terminal vector first.  This serializer emits the exact same six fields
/// in the same order while borrowing the live pool.  Bincode therefore
/// remains compatible with `PoolSnapshot` on restore and peak checkpoint
/// memory is independent of synapse count.
pub(crate) struct PoolSnapshotRef<'a>(pub(crate) &'a Pool);

struct PairSeq<'a, K, V>(&'a AHashMap<K, V>);

impl<K, V> Serialize for PairSeq<'_, K, V>
where
    K: Serialize,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for pair in self.0.iter() {
            seq.serialize_element(&pair)?;
        }
        seq.end()
    }
}

impl Serialize for PoolSnapshotRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let pool = self.0;
        let mut state = serializer.serialize_struct("PoolSnapshot", 6)?;
        state.serialize_field("config", &pool.config)?;
        state.serialize_field("neurons", &pool.neurons)?;
        state.serialize_field("label_to_id", &pool.label_to_id)?;
        state.serialize_field("recent_atoms", &pool.recent_atoms)?;
        state.serialize_field("sequences", &PairSeq(&pool.sequences))?;
        state.serialize_field("cold_offsets", &PairSeq(&pool.cold_offsets))?;
        state.end()
    }
}

impl Pool {
    pub fn new(config: PoolConfig, encoding: Box<dyn AtomEncoding>) -> Self {
        let window = config.recent_atoms_window;
        Self {
            config,
            encoding,
            neurons: NeuronSlots::new(),
            label_to_id: AHashMap::new(),
            concept_multiset_to_id: AHashMap::new(),
            concept_sequence_to_id: AHashMap::new(),
            legacy_concept_sequence_index: None,
            recent_atoms: VecDeque::with_capacity(window),
            sequences: AHashMap::new(),
            legacy_sequence_ledger: None,
            currently_firing: AHashSet::new(),
            activation: AHashMap::new(),
            recent_surprise: 1.0,
            firing_rate_ema: 0.0,
            concept_count_ema: 0.0,
            terminal_count_ema: 0.0,
            pending_promotions: Vec::new(),
            domain_for_new: 0,
            total_terminals: 0,
            decode_precision_ema: 0.0,
            last_observed_sequence: Vec::new(),
            store: Arc::new(NoopStore),
            // Bloom sized for 100K initial capacity — resizes by doubling
            // when load crosses threshold (TODO Stage 17.4 full).  At this
            // scale the filter is ~175 KB per pool — negligible.
            bloom: CountingBloom::with_expected_capacity(100_000),
            cold_tier: None,
            tiered_store: None,
            wbrain_store: None,
            shape_only_residents: AHashSet::new(),
            cold_offsets: AHashMap::new(),
            evicted: AHashSet::new(),
        }
    }

    /// Stage 17.4 — attach a cold-tier backing file so neurons can be
    /// evicted to disk and paged back in.  When `None` (the default),
    /// eviction is disabled and every neuron stays in RAM.
    pub fn set_cold_tier(&mut self, tier: Arc<ColdTier>) {
        self.cold_tier = Some(tier);
    }

    /// Stage 18.12 step 4b — attach a `TieredStore` so eviction/page-in
    /// route through the distributed-substrate abstraction instead of
    /// (or in addition to) the legacy local-only `cold_tier`.  Per
    /// [`ARCHITECTURE.md`] §18.2.
    ///
    /// Composition: when both `cold_tier` and `tiered_store` are set,
    /// the tiered_store takes precedence.  Pool's evict path writes
    /// through `tiered_store.put(id, n)` which internally dispatches
    /// to RAM / cold disk / remote node based on the placement policy.
    /// Page-in reads via `tiered_store.get(id)`.
    pub fn set_tiered_store(&mut self, store: Arc<TieredStore>) {
        self.tiered_store = Some(store);
    }

    pub fn set_wbrain_store(&mut self, store: Arc<WbrainNeuronStore>) {
        self.wbrain_store = Some(store);
    }

    pub fn has_wbrain_store(&self) -> bool {
        self.wbrain_store.is_some()
    }

    /// Stage 18.12 step 4b — diagnostic: true when this pool has a
    /// `TieredStore` attached (i.e. is running in distributed mode).
    pub fn has_tiered_store(&self) -> bool {
        self.tiered_store.is_some()
    }

    /// Stage 18.12 step 7 — what NodeId is the home for neuron `id`,
    /// according to this pool's TieredStore?  Returns None if the pool
    /// has no tiered_store attached (solo mode — answer would always
    /// be the local node anyway).  Used by `Brain::scan_cross_shard_deposits`
    /// to decide which deposits belong to which peer.
    pub fn tiered_home_for(&self, id: NeuronId) -> Option<crate::store::NodeId> {
        self.tiered_store.as_ref().map(|s| {
            use crate::store::NeuronStore;
            s.home_for(id)
        })
    }

    /// Stage 17.4 — true if neuron `id` is currently evicted to disk.
    /// O(1) lookup.  Iteration paths use this to skip placeholder
    /// entries in `neurons` for IDs that have been paged out.
    pub fn is_evicted(&self, id: NeuronId) -> bool {
        if self.wbrain_store.is_some() {
            (id as usize) < self.neurons.len() && self.neurons.get(id as usize).is_none()
        } else {
            self.evicted.contains(&id)
        }
    }

    /// True when this pool has either a cold-tier file or a
    /// distributed tiered store attached — i.e., when eviction is
    /// actually possible.  The tier orchestrator checks this before
    /// scanning so it doesn't spam evict_errors when the brain spun
    /// up without storage configured.
    pub fn has_storage_tier(&self) -> bool {
        self.cold_tier.is_some() || self.tiered_store.is_some() || self.wbrain_store.is_some()
    }

    /// Stage 17.4 — number of currently-evicted neurons.  Diagnostic +
    /// `StorageControlState::working_set_pressure` signal source.
    pub fn evicted_count(&self) -> usize {
        if self.wbrain_store.is_some() {
            self.neurons
                .len()
                .saturating_sub(self.neurons.resident_count())
        } else {
            self.evicted.len()
        }
    }

    /// Stage 17.4 — count of neurons currently held in RAM (live).
    /// Equals `neuron_count() - evicted_count()`.
    pub fn live_count(&self) -> usize {
        self.neurons.resident_count()
    }

    /// Stage 17.4 — evict one neuron to the cold tier.  Serialises its
    /// full state to disk, then zeroes the in-memory slot's terminals
    /// and members (which are the dominant memory consumers per spec
    /// §17.1).  Label + id + kind stay in the slot so `label_to_id` +
    /// `iter_neurons` semantics around tombstones stay sane.
    ///
    /// Returns `Ok(true)` if the neuron was evicted; `Ok(false)` if it
    /// was already evicted (idempotent) or doesn't exist; `Err` if the
    /// cold tier is unattached or the I/O failed.
    pub fn evict_neuron(&mut self, id: NeuronId) -> std::io::Result<bool> {
        if self.shape_only_residents.remove(&id) {
            self.neurons.sleep(id as usize);
            return Ok(true);
        }
        if self.is_evicted(id) {
            return Ok(false);
        }

        // Stage 18.12 step 4b: prefer the tiered store when both are
        // present (the §18 distributed path).  Fall back to the
        // §17.4 cold-tier path when only that's attached.
        if self.wbrain_store.is_none() && self.tiered_store.is_none() && self.cold_tier.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no tiered_store or cold_tier attached to this pool",
            ));
        }
        let Some(n) = self.neurons.get(id as usize) else {
            return Ok(false);
        };
        // Legacy cold storage retained atoms permanently. A .wbrain store is
        // neuron-addressable and therefore sleeps atoms as required by the
        // brain lifecycle; legacy deployments keep their old behavior.
        if n.is_atom() && self.wbrain_store.is_none() {
            return Ok(false);
        }

        // Choose backend: the single-file local store is authoritative when
        // attached, followed by the distributed and legacy paths.
        let used_wbrain = if let Some(store) = self.wbrain_store.clone() {
            store.persist_sleeping(n)?;
            true
        } else {
            false
        };
        let used_tiered = if used_wbrain {
            false
        } else if let Some(store) = self.tiered_store.clone() {
            store.put(id, n.clone());
            true
        } else {
            // Legacy §17.4 cold-tier path.
            let tier = self.cold_tier.clone().unwrap();
            let offset = tier.append_neuron(n)?;
            self.cold_offsets.insert(id, offset);
            false
        };

        if used_wbrain {
            // The complete body is durable. Remove this one allocation from
            // RAM; the stable address remains as a null pointer in
            // `NeuronSlots`. No per-neuron sentinel survives sleep.
            let removed = self
                .neurons
                .sleep(id as usize)
                .expect("persisted neuron must still occupy its live slot");
            debug_assert_eq!(removed.id, id);
        } else {
            // Legacy cold tiers retain a compact in-place identity because
            // their older indexes are not sufficient to reconstruct atoms.
            let n = self.neurons.get_mut(id as usize).unwrap();
            let dropped = n.terminals.len();
            n.terminals.clear();
            n.terminals.shrink_to_fit();
            n.terminal_idx.clear();
            n.terminal_idx.shrink_to_fit();
            n.prediction_error_ema = 0.0;
            self.total_terminals = self.total_terminals.saturating_sub(dropped);
            // Salience/EMA stay in RAM — they're the signal eviction policy
            // uses to choose what to evict, so they remain readable post-evict.
        }

        if self.wbrain_store.is_none() {
            self.evicted.insert(id);
        }

        // Per ARCHITECTURE §17.9: log the eviction.
        let event = WalEvent::NeuronEvicted {
            pool_id: self.config.id,
            neuron_id: id,
        };
        if let Err(e) = self.store.append(&event) {
            tracing::warn!("WAL append failed for evict({}): {}", id, e);
        }
        if used_tiered {
            tracing::debug!(
                "evicted neuron pool={} id={} via tiered_store (§18)",
                self.config.id,
                id,
            );
        }
        Ok(true)
    }

    /// Stage 17.4 — page a previously-evicted neuron back into RAM.
    /// Reads its serialised form from the cold tier at the stored
    /// offset, restores `terminals` + `members` in the in-memory slot,
    /// and removes the eviction markers.
    ///
    /// Returns `Ok(true)` if the neuron was paged in; `Ok(false)` if
    /// it wasn't evicted to begin with (caller can ignore); `Err` if
    /// the cold tier is missing or the read failed.
    pub fn page_in_neuron(&mut self, id: NeuronId) -> std::io::Result<bool> {
        if self.shape_only_residents.remove(&id) {
            self.neurons.sleep(id as usize);
        }
        if !self.is_evicted(id) {
            return Ok(false);
        }
        let paging_wbrain = self.wbrain_store.is_some();

        // Stage 18.12 step 4b: prefer the tiered store when attached.
        // It routes through local cold disk OR a remote peer depending
        // on the placement policy.  When absent, fall back to the
        // §17.4 cold-tier + cold_offsets path.
        let restored: Neuron = if let Some(store) = self.wbrain_store.clone() {
            let neuron = store.get(id).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(".wbrain store has no neuron id {}", id),
                )
            })?;
            store.release_cached(id);
            neuron
        } else if let Some(store) = self.tiered_store.clone() {
            match store.get(id) {
                Some(n) => n,
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("tiered_store has no neuron id {} (home check?)", id),
                    ));
                }
            }
        } else {
            let Some(tier) = self.cold_tier.clone() else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "no tiered_store or cold_tier attached to this pool",
                ));
            };
            let Some(&offset) = self.cold_offsets.get(&id) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no cold offset recorded for neuron id {}", id),
                ));
            };
            tier.read_neuron(offset)?
        };
        // Sanity check: restored id must match.
        if restored.id != id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "cold-tier id mismatch: expected {}, got {}",
                    id, restored.id
                ),
            ));
        }
        if self.neurons.get(id as usize).is_none() {
            if !self.neurons.wake(id as usize, restored) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("neuron address {} no longer exists", id),
                ));
            }
            self.cold_offsets.remove(&id);
            self.evicted.remove(&id);
            // `.wbrain` metadata keeps the complete learned terminal count;
            // paging a body changes residency, not learned topology.
            return Ok(true);
        }
        let Some(slot) = self.neurons.get_mut(id as usize) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("neuron slot {} no longer exists", id),
            ));
        };
        // MERGE, don't overwrite: a concept can keep firing while evicted
        // (other neurons' terminals still point at it), so cross-pool
        // wiring may have grown fresh terminals on the cleared slot.
        // Overwriting with the cold copy would silently erase everything
        // learned since eviction.  Cold terminals are the base; any
        // in-RAM terminal for a target the cold copy doesn't know wins a
        // slot, and for shared targets the stronger weight survives.
        // Terminals grown while evicted are already in the pool gauge;
        // only the net growth from restoring the cold copy gets added.
        let live_before = slot.terminals.len();
        slot.label = restored.label;
        slot.kind = restored.kind;
        slot.born_tick = restored.born_tick;
        if slot.terminals.is_empty() {
            slot.terminals = restored.terminals;
        } else {
            let mut merged = restored.terminals;
            for live in slot.terminals.drain(..) {
                match merged.iter_mut().find(|t| t.target == live.target) {
                    Some(cold) => {
                        if live.weight > cold.weight {
                            cold.weight = live.weight;
                        }
                        cold.consolidation = cold.consolidation.max(live.consolidation);
                    }
                    None => merged.push(live),
                }
            }
            slot.terminals = merged;
        }
        // Eviction cleared terminal_idx; without a rebuild here,
        // add_or_strengthen would re-append duplicates for every target
        // the index no longer knows about.
        slot.rebuild_terminal_idx();
        // Evict subtracted the cold copy's terminals from the pool gauge;
        // without the symmetric add the pressure factor undercounts after
        // every evict/page-in cycle and eviction gets lazier over time.
        let restored_terms = slot.terminals.len().saturating_sub(live_before);
        slot.members = restored.members;
        slot.prediction_error_ema = restored.prediction_error_ema;
        // Use the more-current salience: in-RAM may have decayed below
        // the cold copy's value, or vice versa.  Eviction freezes
        // salience at evict-time; choose the cold copy as authoritative.
        slot.salience = restored.salience;
        slot.salience_ema = restored.salience_ema;
        slot.use_count = restored.use_count.max(slot.use_count);
        slot.last_fired_tick = restored.last_fired_tick.max(slot.last_fired_tick);

        self.cold_offsets.remove(&id);
        self.evicted.remove(&id);
        if !paging_wbrain {
            self.total_terminals += restored_terms;
        }
        Ok(true)
    }

    /// Stage 17.4 — convenience: page `id` in if it's evicted, no-op
    /// otherwise.  Used by callers that don't care which state the
    /// neuron is in, just that it's accessible.
    pub fn ensure_loaded(&mut self, id: NeuronId) -> std::io::Result<()> {
        if self.shape_only_residents.contains(&id) || self.is_evicted(id) {
            self.page_in_neuron(id)?;
        }
        Ok(())
    }

    /// Load only the stable routing shape of a sleeping neuron for a
    /// read-only inference request. Terminal vectors remain on SSD. A later
    /// `ensure_loaded` transparently replaces this stub with the full body.
    pub(crate) fn ensure_shape_loaded_read_only(&mut self, id: NeuronId) -> std::io::Result<()> {
        if self.neurons.get(id as usize).is_some() {
            return Ok(());
        }
        let Some(store) = self.wbrain_store.as_ref() else {
            return self.ensure_loaded(id);
        };
        let Some((is_atom, label, members)) = store.neuron_shape(id)? else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(".wbrain store has no neuron shape for id {}", id),
            ));
        };
        let neuron = if is_atom {
            Neuron::new_atom(id, label, NeuronKind::Excitatory, 0)
        } else {
            Neuron::new_concept(id, label, NeuronKind::Excitatory, members, 0)
        };
        if !self.neurons.wake(id as usize, neuron) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("neuron address {} no longer exists", id),
            ));
        }
        self.shape_only_residents.insert(id);
        Ok(())
    }

    /// Serialize every live neuron after maintenance has completed. This is
    /// the explicit idle/sleep boundary: no salience or age policy applies,
    /// because the knowledge is retained on SSD rather than pruned.
    pub fn serialize_all_neurons_for_idle(&mut self) -> std::io::Result<usize> {
        if self.wbrain_store.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "idle serialization requires a .wbrain store",
            ));
        }
        let mut serialized = 0;
        // A paged brain may have millions of stable addresses but only the
        // request-local working set is awake. Iterating the logical address
        // extent here made per-binding finalization quadratic in brain size.
        for id in self.neurons.resident_ids() {
            if self.evict_neuron(id)? {
                serialized += 1;
            }
        }
        let store = self.wbrain_store.as_ref().unwrap();
        store.flush_resident_labels_to_disk()?;
        self.flush_concept_sequence_overlay()?;
        self.flush_sequence_recurrence_overlay()?;
        self.label_to_id.clear();
        self.concept_sequence_to_id.clear();
        self.concept_multiset_to_id.clear();
        self.sequences.clear();
        self.bloom = CountingBloom::with_expected_capacity(100_000);
        Ok(serialized)
    }

    /// Release bodies paged in by a provably read-only maintenance pass.
    /// Unlike the ordinary idle transition this performs no neuron rewrite
    /// and no full logical-slot scan. It is intentionally crate-private: a
    /// caller must know that no resident body was mutated after page-in.
    pub(crate) fn discard_wbrain_residents_read_only(&mut self) -> std::io::Result<usize> {
        if self.wbrain_store.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "read-only resident discard requires a .wbrain store",
            ));
        }
        let discarded = self.neurons.discard_paged_residents().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "read-only resident discard requires paged neuron slots",
            )
        })?;
        self.shape_only_residents.clear();
        Ok(discarded)
    }

    /// Read only the shape needed by index maintenance, then immediately
    /// release the paged body. Concept children are returned as stable refs;
    /// atom bytes are decoded while the label is resident.
    pub(crate) fn read_neuron_shape_bounded(
        &mut self,
        id: NeuronId,
    ) -> std::io::Result<Option<(bool, Vec<NeuronRef>, Vec<u8>)>> {
        if let Some(store) = self.wbrain_store.as_ref() {
            let shape = store.neuron_shape(id)?;
            return Ok(shape.map(|(is_atom, label, members)| {
                let decoded = if is_atom {
                    let pairs = [(label.as_str(), 1.0_f32)];
                    self.encoding.reassemble(&pairs)
                } else {
                    Vec::new()
                };
                (is_atom, members, decoded)
            }));
        }
        self.ensure_loaded(id)?;
        let shape = self.neurons.get(id as usize).map(|neuron| {
            let is_atom = neuron.is_atom();
            let decoded = if is_atom {
                let pairs = [(neuron.label.as_str(), 1.0_f32)];
                self.encoding.reassemble(&pairs)
            } else {
                Vec::new()
            };
            (is_atom, neuron.members.clone(), decoded)
        });
        self.discard_wbrain_residents_read_only()?;
        Ok(shape)
    }

    fn flush_concept_sequence_overlay(&mut self) -> std::io::Result<usize> {
        let entry_count = self
            .concept_sequence_to_id
            .len()
            .saturating_add(self.concept_multiset_to_id.len());
        if entry_count == 0 {
            return Ok(0);
        }
        let store = self.wbrain_store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "concept index flush requires a .wbrain store",
            )
        })?;
        let file = store.file();
        let directory = file.path().parent().map(ToOwned::to_owned).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                ".wbrain path has no parent directory",
            )
        })?;
        let mut builder =
            PostingIndexBuilder::create(&directory, self.config.id, entry_count as u64)?;
        for (sequence, id) in &self.concept_sequence_to_id {
            builder.insert(
                &Self::concept_posting_key(CONCEPT_SEQUENCE_KEY, sequence),
                *id,
            )?;
        }
        for (sequence, id) in &self.concept_multiset_to_id {
            builder.insert(
                &Self::concept_posting_key(CONCEPT_ATOM_LEAVES_KEY, sequence),
                *id,
            )?;
        }
        let delta = builder.finish(&file, self.config.id)?;
        let root = self.publish_auxiliary_generation(
            &file,
            self.legacy_concept_sequence_index,
            delta,
            CONCEPT_GENERATION_DIRECTORY_MAGIC,
            CONCEPT_GENERATION_DIRECTORY_KIND,
            "concept index",
        )?;
        self.legacy_concept_sequence_index = Some(root);
        Ok(entry_count)
    }

    fn flush_sequence_recurrence_overlay(&mut self) -> std::io::Result<usize> {
        if self.sequences.is_empty() {
            return Ok(0);
        }
        let store = self.wbrain_store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "sequence ledger flush requires a .wbrain store",
            )
        })?;
        let file = store.file();
        let directory = file.path().parent().map(ToOwned::to_owned).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                ".wbrain path has no parent directory",
            )
        })?;
        let mut builder =
            PostingIndexBuilder::create(&directory, self.config.id, self.sequences.len() as u64)?;
        for (sequence, count) in &self.sequences {
            builder.insert(
                &Self::concept_posting_key(SEQUENCE_RECURRENCE_KEY, sequence),
                *count,
            )?;
        }
        let delta = builder.finish(&file, self.config.id)?;
        let root = self.publish_auxiliary_generation(
            &file,
            self.legacy_sequence_ledger,
            delta,
            SEQUENCE_GENERATION_DIRECTORY_MAGIC,
            SEQUENCE_GENERATION_DIRECTORY_KIND,
            "sequence ledger",
        )?;
        self.legacy_sequence_ledger = Some(root);
        Ok(self.sequences.len())
    }

    fn publish_auxiliary_generation(
        &self,
        file: &Arc<WbrainFile>,
        existing: Option<AuxiliaryRecordRef>,
        delta: AuxiliaryRecordRef,
        directory_magic: &[u8; 8],
        directory_kind: u32,
        description: &str,
    ) -> std::io::Result<AuxiliaryRecordRef> {
        let mut generations = match existing {
            Some(root) => self.auxiliary_generations(root, directory_magic, description)?,
            None => Vec::new(),
        };
        generations.push(delta);
        if generations.len() == 1 {
            return Ok(delta);
        }
        file.append_auxiliary(self.config.id, directory_kind, |writer| {
            writer.write_all(directory_magic)?;
            writer.write_all(&(generations.len() as u64).to_le_bytes())?;
            for reference in &generations {
                writer.write_all(&reference.offset.to_le_bytes())?;
                writer.write_all(&reference.len.to_le_bytes())?;
            }
            Ok(())
        })
    }

    pub fn concept_index_residency(&self) -> (usize, usize) {
        let overlay_entries = self
            .concept_sequence_to_id
            .len()
            .saturating_add(self.concept_multiset_to_id.len());
        let generation_count = self
            .legacy_concept_sequence_index
            .and_then(|root| {
                self.auxiliary_generations(
                    root,
                    CONCEPT_GENERATION_DIRECTORY_MAGIC,
                    "concept index",
                )
                .ok()
            })
            .map_or(0, |generations| generations.len());
        (overlay_entries, generation_count)
    }

    pub fn sequence_ledger_residency(&self) -> (usize, usize) {
        let generation_count = self
            .legacy_sequence_ledger
            .and_then(|root| {
                self.auxiliary_generations(
                    root,
                    SEQUENCE_GENERATION_DIRECTORY_MAGIC,
                    "sequence ledger",
                )
                .ok()
            })
            .map_or(0, |generations| generations.len());
        (self.sequences.len(), generation_count)
    }

    /// Store the compact address-space and routing state needed to reopen this
    /// pool without deserializing any neuron body.
    pub fn stage_wbrain_metadata(&self) -> std::io::Result<()> {
        let store = self.wbrain_store.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Unsupported, "pool has no .wbrain store")
        })?;
        let metadata = WbrainPoolMetadata {
            config: self.config.clone(),
            recent_atoms: self.recent_atoms.clone(),
            sequences: self
                .sequences
                .iter()
                .map(|(sequence, count)| (sequence.clone(), *count))
                .collect(),
            legacy_sequence_ledger: self.legacy_sequence_ledger,
            concept_multiset_to_id: self
                .concept_multiset_to_id
                .iter()
                .map(|(sequence, id)| (sequence.clone(), *id))
                .collect(),
            concept_sequence_to_id: self
                .concept_sequence_to_id
                .iter()
                .map(|(sequence, id)| (sequence.clone(), *id))
                .collect(),
            legacy_concept_sequence_index: self.legacy_concept_sequence_index,
            neuron_kinds: self.neurons.kinds().to_vec(),
            concept_slots: self.neurons.concept_flags().to_vec(),
            born_ticks: self.neurons.born_ticks().to_vec(),
            concept_count: self.neurons.concept_count(),
            total_terminals: self.total_terminals,
        };
        let bytes = bincode::serialize(&metadata)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        store.set_pool_metadata(bytes);
        Ok(())
    }

    pub(crate) fn stage_streamed_wbrain_metadata(
        store: &WbrainNeuronStore,
        metadata: StreamedPoolMetadata,
    ) -> std::io::Result<()> {
        let metadata = WbrainPoolMetadata {
            config: metadata.config,
            recent_atoms: metadata.recent_atoms,
            sequences: metadata.sequences,
            legacy_sequence_ledger: metadata.legacy_sequence_ledger,
            concept_multiset_to_id: Vec::new(),
            concept_sequence_to_id: metadata.concept_sequence_to_id,
            legacy_concept_sequence_index: metadata.legacy_concept_sequence_index,
            neuron_kinds: metadata.neuron_kinds,
            concept_slots: metadata.concept_slots,
            born_ticks: metadata.born_ticks,
            concept_count: metadata.concept_count,
            total_terminals: metadata.total_terminals,
        };
        let bytes = bincode::serialize(&metadata)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        store.set_pool_metadata(bytes);
        Ok(())
    }

    /// Rebuild the two derived concept-deduplication indexes without keeping
    /// every concept body resident. Legacy snapshots did not persist these
    /// indexes, so a streaming conversion reconstructs one local concept tree
    /// at a time and returns it to sleep before advancing.
    pub(crate) fn rebuild_concept_indexes_bounded(&mut self) -> std::io::Result<usize> {
        self.concept_multiset_to_id.clear();
        self.concept_sequence_to_id.clear();
        if self.legacy_concept_sequence_index.is_some() {
            // Streaming migration already constructed the authoritative
            // on-disk sequence index while each concept body was in hand.
            return Ok(self.neurons.concept_count());
        }
        let mut rebuilt = 0;
        for raw_id in 0..self.neurons.len() {
            let concept_id = raw_id as NeuronId;
            if !self.is_concept_slot(concept_id) {
                continue;
            }
            rebuilt += 1;
            let mut pending = vec![concept_id];
            let mut visited = AHashSet::new();
            while let Some(id) = pending.pop() {
                if !visited.insert(id) {
                    continue;
                }
                self.ensure_loaded(id)?;
                if let Some(neuron) = self.neurons.get(id as usize) {
                    pending.extend(
                        neuron
                            .members
                            .iter()
                            .filter(|member| member.pool == self.config.id)
                            .map(|member| member.neuron),
                    );
                }
            }
            let member_ids: Vec<NeuronId> = self.neurons[concept_id as usize]
                .members
                .iter()
                .filter(|member| member.pool == self.config.id)
                .map(|member| member.neuron)
                .collect();
            if let Some(leaves) = self.expand_to_atom_leaves_bounded(&member_ids, 512) {
                self.concept_multiset_to_id
                    .entry(leaves)
                    .or_insert(concept_id);
            }
            self.concept_sequence_to_id
                .entry(member_ids)
                .or_insert(concept_id);
            self.serialize_all_neurons_for_idle()?;
        }
        Ok(rebuilt)
    }

    /// Rebuild only the compact pool address space. Every neuron body remains
    /// asleep in `.wbrain` until its stable id is explicitly requested.
    pub fn from_wbrain_store(
        encoding: Box<dyn AtomEncoding>,
        store: Arc<WbrainNeuronStore>,
    ) -> std::io::Result<Self> {
        let metadata: WbrainPoolMetadata = bincode::deserialize(&store.pool_metadata())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let mut label_to_id = AHashMap::new();
        for (label, id) in store.labels_snapshot() {
            label_to_id.insert(label, id);
        }
        let slot_count = store.slot_count();
        let neurons = NeuronSlots::sleeping(slot_count, metadata.concept_count);
        let mut bloom = CountingBloom::with_expected_capacity(label_to_id.len().max(100_000));
        for label in label_to_id.keys() {
            bloom.insert(label);
        }
        Ok(Self {
            config: metadata.config,
            encoding,
            neurons,
            label_to_id,
            concept_multiset_to_id: metadata.concept_multiset_to_id.into_iter().collect(),
            concept_sequence_to_id: metadata.concept_sequence_to_id.into_iter().collect(),
            legacy_concept_sequence_index: metadata.legacy_concept_sequence_index,
            recent_atoms: metadata.recent_atoms,
            sequences: metadata.sequences.into_iter().collect(),
            legacy_sequence_ledger: metadata.legacy_sequence_ledger,
            currently_firing: AHashSet::new(),
            activation: AHashMap::new(),
            recent_surprise: 1.0,
            last_observed_sequence: Vec::new(),
            firing_rate_ema: 0.0,
            concept_count_ema: 0.0,
            terminal_count_ema: 0.0,
            total_terminals: metadata.total_terminals,
            pending_promotions: Vec::new(),
            domain_for_new: 0,
            store: Arc::new(NoopStore),
            bloom,
            cold_tier: None,
            tiered_store: None,
            wbrain_store: Some(store),
            shape_only_residents: AHashSet::new(),
            cold_offsets: AHashMap::new(),
            evicted: AHashSet::new(),
            decode_precision_ema: 0.0,
        })
    }

    /// Stage 17.9 — recovery-time atom creation.  Inserts a neuron at
    /// an explicit `NeuronId` (asserts the id == neurons.len() so the
    /// natural append ordering of the original training timeline is
    /// preserved).  Skips the WAL append (recovery doesn't re-emit
    /// events) and the Bloom + label_to_id updates ARE applied so the
    /// pool is consistent post-recovery.  Returns `true` if the
    /// insertion proceeded; `false` if the slot was already taken
    /// (idempotent on replays).
    pub fn replay_atom_create(
        &mut self,
        id: NeuronId,
        label: String,
        kind: NeuronKind,
        born_tick: u64,
    ) -> bool {
        if (id as usize) < self.neurons.len() {
            // Slot already filled — recovery already replayed this event,
            // or brain.bin already restored this neuron.  Idempotent skip.
            return false;
        }
        // Strict invariant: id must equal next slot.  Otherwise the
        // sequential-id contract is broken and downstream code that
        // indexes by id would corrupt.  Refuse rather than fudge.
        if (id as usize) != self.neurons.len() {
            tracing::warn!(
                "replay_atom_create: id={} but next slot is {}; refusing",
                id,
                self.neurons.len(),
            );
            return false;
        }
        let n = Neuron::new_atom(id, label.clone(), kind, born_tick);
        self.neurons.push(n);
        self.bloom.insert(&label);
        self.label_to_id.insert(label, id);
        true
    }

    /// Stage 17.9 — recovery-time concept creation.  Mirror of
    /// [`replay_atom_create`] for concept neurons (non-empty members).
    pub fn replay_concept_create(
        &mut self,
        id: NeuronId,
        label: String,
        kind: NeuronKind,
        members: Vec<NeuronRef>,
        born_tick: u64,
    ) -> bool {
        if (id as usize) < self.neurons.len() {
            return false;
        }
        if (id as usize) != self.neurons.len() {
            tracing::warn!(
                "replay_concept_create: id={} but next slot is {}; refusing",
                id,
                self.neurons.len(),
            );
            return false;
        }
        let local_members: Vec<NeuronId> = members
            .iter()
            .filter(|member| member.pool == self.config.id)
            .map(|member| member.neuron)
            .collect();
        let n = Neuron::new_concept(id, label.clone(), kind, members, born_tick);
        self.neurons.push(n);
        self.bloom.insert(&label);
        self.label_to_id.insert(label, id);
        self.concept_sequence_to_id
            .entry(local_members.clone())
            .or_insert(id);
        if let Some(leaves) = self.expand_to_atom_leaves_bounded(&local_members, 512) {
            self.concept_multiset_to_id.entry(leaves).or_insert(id);
        }
        true
    }

    /// Stage 17.6 deeper — insert a fully-constructed peer neuron at
    /// the next free id slot, preserving terminals + members + salience
    /// + use_count.  Mirror of `replay_atom_create` / `replay_concept_create`
    /// but accepts the WHOLE neuron struct so cluster sync transfers
    /// terminal weights too, not just topology.
    ///
    /// Refuses if `neuron.id != self.neurons.len()` (sequential-id
    /// contract) or if the id slot is already filled.  Returns `true`
    /// on successful insertion.
    pub fn replay_full_neuron(&mut self, neuron: Neuron) -> bool {
        let next = self.neurons.len() as NeuronId;
        if neuron.id != next {
            tracing::warn!(
                "replay_full_neuron: id={} but next slot is {}; refusing",
                neuron.id,
                next,
            );
            return false;
        }
        let label = neuron.label.clone();
        self.bloom.insert(&label);
        self.label_to_id.insert(label, neuron.id);
        self.neurons.push(neuron);
        true
    }

    /// Stage 18.12 step 6+ — accept a neuron from a cluster peer's
    /// /shard/put_neuron call.  Unlike `replay_full_neuron` (which
    /// enforces sequential ids per the §17.6 cluster_pull contract),
    /// this method accepts arbitrary ids and pads the Vec with empty
    /// placeholder neurons to preserve dense indexing.
    ///
    /// Placeholders have empty labels and don't enter the label_to_id
    /// index, so they're invisible to neurogenesis collision checks and
    /// decode-path lookups.  They DO occupy a slot in `neurons` so that
    /// `Pool::get(id)` still returns the correct neuron by index.
    ///
    /// Returns `true` if the incoming neuron was inserted (always, in
    /// practice — overrides existing entries).  Per [`ARCHITECTURE.md`]
    /// §18.5 shard-receive semantics.
    pub fn accept_shard_insert(&mut self, neuron: Neuron) -> bool {
        let id = neuron.id;
        let label = neuron.label.clone();
        let idx = id as usize;
        // Pad with empty-label placeholders up to (but not including) idx.
        while self.neurons.len() < idx {
            let pid = self.neurons.len() as NeuronId;
            // Placeholder: empty label, atom kind, no members/terminals.
            // Not added to label_to_id (only labelled neurons enter it).
            let placeholder = Neuron::new_atom(pid, String::new(), NeuronKind::Excitatory, 0);
            self.neurons.push(placeholder);
        }
        if idx == self.neurons.len() {
            self.neurons.push(neuron);
        } else {
            // idx < neurons.len() — overwrite the existing slot.  Could
            // be a stale placeholder or a previously-merged neuron;
            // peer-provided state wins.
            self.neurons.set(idx, neuron);
        }
        if !label.is_empty() {
            self.bloom.insert(&label);
            self.label_to_id.insert(label, id);
        }
        true
    }

    /// Stage 17.9 — recovery-time eviction mark.  Adds an entry to the
    /// `evicted` set + `cold_offsets` index without actually writing
    /// to the cold tier (the cold-tier file's record already exists on
    /// disk from the original training run; replaying the
    /// `NeuronEvicted` event just rebuilds the in-memory index for it).
    /// Caller must have already replayed the corresponding
    /// AtomCreated/ConceptEmerged event so the slot exists.
    ///
    /// Note: the current `NeuronEvicted` event variant doesn't carry the
    /// cold-tier offset. If an eviction happened after the checkpoint, the
    /// restored snapshot still contains the complete resident neuron, so the
    /// safe recovery is to keep it resident and let the orchestrator evict it
    /// again. Installing a guessed/zero offset would make paging corrupt.
    pub fn replay_neuron_evicted(&mut self, id: NeuronId) {
        if !self.cold_offsets.contains_key(&id) {
            return;
        }
        self.evicted.insert(id);
    }

    /// Stage 17.3 — probabilistic existence pre-check over the label
    /// index.  Returns `false` only when the label is definitively
    /// absent; `true` indicates "maybe present" (false-positive rate
    /// is ~1e-4 at default Bloom capacity).  Callers needing exact
    /// answers fall through to [`Pool::label_to_id`].
    pub fn label_might_exist(&self, label: &str) -> bool {
        self.bloom.might_contain(label)
            || self
                .wbrain_store
                .as_ref()
                .is_some_and(|store| store.label_to_id(label).is_some())
    }

    /// Stage 17.3 — diagnostic: number of byte slots the Bloom filter
    /// occupies.  Used by `/stats` to surface filter size.
    pub fn bloom_byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Stage 17.3 — diagnostic: number of distinct keys inserted into
    /// the Bloom filter (approximate; counts transitions from 0).
    pub fn bloom_inserted_keys(&self) -> usize {
        self.bloom.inserted_keys()
    }

    /// Stage 17.6 — compute the deterministic Merkle root for this
    /// pool's current learned state.  Two pools with identical training
    /// history produce identical roots; cluster anti-entropy diff
    /// (follow-up commit, needs the cluster transport) operates on
    /// these roots.  Per [`ARCHITECTURE.md`] §17.6.
    ///
    /// `fabric_tick` is supplied by the caller (usually `Fabric::current_tick`)
    /// because the pool doesn't own the global tick — and per-pool roots
    /// should reflect the tick at which they're computed for cluster
    /// timestamp-ordering of diff exchanges.
    pub fn merkle_root(&self, fabric_tick: u64) -> crate::store::PoolRoot {
        let seqs: Vec<(Vec<NeuronId>, u32)> = self
            .sequences
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let dense_neurons = self.neurons.dense_clone();
        crate::store::compute_pool_root(
            &self.config,
            &dense_neurons,
            &seqs,
            &self.bloom,
            fabric_tick,
        )
    }

    /// Plug a persistence backend in per [`ARCHITECTURE.md`] §17.9.  After
    /// this call every neurogenesis / concept-emergence event in this pool
    /// is appended to the store's WAL.  Returns `self` for chaining.
    pub fn set_store(&mut self, store: Arc<dyn Store>) {
        self.store = store;
    }

    pub fn id(&self) -> PoolId {
        self.config.id
    }
    pub fn name(&self) -> &str {
        &self.config.name
    }
    pub fn encoding_name(&self) -> &'static str {
        self.encoding.name()
    }
    /// Inspect the deterministic sensor labels for a frame without creating
    /// neurons or changing activation. This supports inhibitory diagnostics
    /// (for example, explicitly missing specifications) during prediction.
    pub fn encoded_labels(&self, frame: &[u8]) -> Vec<String> {
        self.encoding.atomize(frame)
    }
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Set the domain id stamped onto every atom + concept created
    /// from now on (island architecture).  Doesn't retroactively
    /// re-tag existing neurons.  0 = unassigned / global (default).
    pub fn set_domain_for_new(&mut self, domain_id: u32) {
        self.domain_for_new = domain_id;
    }

    pub fn domain_for_new(&self) -> u32 {
        self.domain_for_new
    }

    /// Count neurons per domain.  For operator visibility into how
    /// the island clustering is going.
    pub fn domain_histogram(&self) -> std::collections::HashMap<u32, usize> {
        let mut h = std::collections::HashMap::new();
        for n in &self.neurons {
            *h.entry(n.domain_id).or_insert(0) += 1;
        }
        h
    }

    /// O(1) read of the maintained total-terminal counter.  Replaces
    /// the formerly O(N) `iter_neurons().map(|n| n.terminals.len()).sum()`
    /// pattern that turned every brain.stats() call into a fabric-wide
    /// scan.
    pub fn total_terminals(&self) -> usize {
        self.total_terminals
    }

    /// Terminals whose owning neuron body is currently resident in RAM.
    /// This walks only the bounded resident set; sleeping slot metadata and
    /// serialized neuron bodies are never paged in for telemetry.
    pub fn resident_terminal_count(&self) -> usize {
        self.neurons
            .resident_ids()
            .into_iter()
            .filter_map(|id| self.neurons.get(id as usize))
            .map(|neuron| neuron.terminals.len())
            .sum()
    }

    pub fn concept_count(&self) -> usize {
        self.neurons.concept_count()
    }
    pub(crate) fn is_concept_slot(&self, id: NeuronId) -> bool {
        self.wbrain_store.as_ref().map_or_else(
            || self.neurons.is_concept(id),
            |store| store.slot_is_concept(id),
        )
    }
    /// Number of neuron slots (including evicted ones whose terminals
    /// have been shed).  Tier orchestrator uses this for round-robin
    /// scan bounds.
    pub fn neurons_len(&self) -> usize {
        self.neurons.len()
    }
    /// Indexed read of a neuron slot — `None` past the end.  Used by
    /// the tier orchestrator's bounded scan loop.
    pub fn neuron_at(&self, idx: usize) -> Option<&Neuron> {
        self.neurons.get(idx)
    }

    /// Snapshot the pool's observable signals for ControlMode evaluation.
    /// Normalised so signals are roughly in [0, 1] and ControlModes can
    /// compose without each signal having its own scale calibration.
    pub fn control_state(&self) -> ControlState {
        let neurons = self.neurons.len() as f32;
        let firing_rate = if neurons > 0.0 {
            (self.firing_rate_ema / neurons).clamp(0.0, 1.0)
        } else {
            0.0
        };
        ControlState {
            surprise: self.recent_surprise.clamp(0.0, 1.0),
            firing_rate,
            decode_precision: self.decode_precision_ema.clamp(0.0, 1.0),
            // log-norm so a pool with 10K concepts isn't 10× more
            // influential than one with 1K.  /4 keeps log10(10K)=4 at 1.0.
            concept_count_norm: (self.concept_count_ema.max(1.0).log10() / 4.0).clamp(0.0, 1.0),
            terminal_count_norm: (self.terminal_count_ema.max(1.0).log10() / 7.0).clamp(0.0, 1.0),
        }
    }

    /// Update observable EMAs.  Called every observe + every
    /// tick_housekeeping.  α = 0.1 — recent ~10 events weighted heavily.
    fn update_emas(&mut self) {
        const ALPHA: f32 = 0.1;
        let firing = self.currently_firing.len() as f32;
        self.firing_rate_ema = self.firing_rate_ema * (1.0 - ALPHA) + firing * ALPHA;
        let n_concepts = self.neurons.concept_count() as f32;
        self.concept_count_ema = self.concept_count_ema * (1.0 - ALPHA) + n_concepts * ALPHA;
        // O(1) — formerly an O(N) walk; total_terminals is maintained
        // incrementally on every terminal mutation.
        let n_terms = self.total_terminals as f32;
        self.terminal_count_ema = self.terminal_count_ema * (1.0 - ALPHA) + n_terms * ALPHA;
    }

    /// Record one externally confirmed precision sample. Read-only decoding
    /// must not call this: an unverified answer is not a learning outcome.
    pub fn record_decode_precision(&mut self, score: f32) {
        const ALPHA: f32 = 0.15;
        self.decode_precision_ema =
            self.decode_precision_ema * (1.0 - ALPHA) + score.clamp(0.0, 1.0) * ALPHA;
    }
    pub fn currently_firing(&self) -> impl Iterator<Item = NeuronId> + '_ {
        self.currently_firing.iter().copied()
    }
    /// Activate only atoms that already exist, without neurogenesis,
    /// recurrence accounting, decay, WAL writes, or Hebbian updates.
    /// The caller must invoke `clear_prediction_activation` after readout.
    pub fn activate_known_frame_for_prediction(&mut self, frame: &[u8]) -> Vec<NeuronId> {
        self.activate_known_frame_for_prediction_mode(frame, false)
    }

    /// Indexed inference needs identity, labels, and concept membership but
    /// routes through posting indexes rather than terminal propagation.
    pub(crate) fn activate_known_frame_shape_for_prediction(
        &mut self,
        frame: &[u8],
    ) -> Vec<NeuronId> {
        self.activate_known_frame_for_prediction_mode(frame, true)
    }

    fn activate_known_frame_for_prediction_mode(
        &mut self,
        frame: &[u8],
        shape_only: bool,
    ) -> Vec<NeuronId> {
        self.clear_prediction_activation();
        let fired: Vec<NeuronId> = self
            .encoding
            .atomize(frame)
            .into_iter()
            .filter_map(|label| self.label_to_id(&label))
            .collect();
        for &id in &fired {
            let loaded = if shape_only {
                self.ensure_shape_loaded_read_only(id)
            } else {
                self.ensure_loaded(id)
            };
            if let Err(error) = loaded {
                tracing::warn!("prediction page-in failed for neuron {}: {}", id, error);
                continue;
            }
            self.activation.insert(id, 1.0);
            self.currently_firing.insert(id);
        }
        // Replay the same hierarchical mini-column collapse used while
        // learning, but against a query-local stack.  Previously prediction
        // activated atoms only, so every learned prompt concept was inert at
        // inference and concept-tier retrieval could never generalise.  This
        // path performs no neurogenesis, recurrence counting, Hebbian update,
        // or mutation of the learned recent-atom buffer.
        let mut stack: Vec<NeuronId> = Vec::with_capacity(fired.len());
        for &atom in &fired {
            stack.push(atom);
            loop {
                let max_len = self.config.max_concept_member_count.min(stack.len());
                if max_len < 2 {
                    break;
                }
                let mut matched: Option<(usize, NeuronId)> = None;
                for len in (2..=max_len).rev() {
                    let start = stack.len() - len;
                    if let Some(cid) = self.concept_id_for_sequence(&stack[start..]) {
                        matched = Some((len, cid));
                        break;
                    }
                }
                let Some((len, cid)) = matched else { break };
                let loaded = if shape_only {
                    self.ensure_shape_loaded_read_only(cid)
                } else {
                    self.ensure_loaded(cid)
                };
                if let Err(error) = loaded {
                    tracing::warn!("prediction concept page-in failed for {}: {}", cid, error);
                    break;
                }
                stack.truncate(stack.len() - len);
                stack.push(cid);
                self.activation.insert(cid, 1.0);
                self.currently_firing.insert(cid);
            }
        }
        self.last_observed_sequence = fired.clone();
        fired
    }

    /// Fast-forward recurrence evidence from a batch of raw frames.
    ///
    /// This is deliberately not tokenization: frames are atomized by the
    /// pool's existing encoding, candidate sequences contain those exact atom
    /// neuron ids in firing order, and promotion uses the normal concept birth
    /// path. It merely counts recurrence in one deterministic pass.
    pub fn pretrain_recurring_patterns(
        &mut self,
        frames: &[Vec<u8>],
        tick: u64,
        min_recurrence: u32,
        max_promotions: usize,
    ) -> PretrainReport {
        let min_recurrence = min_recurrence.max(2);
        let max_len = self.config.max_concept_member_count;
        let mut counts: AHashMap<Vec<NeuronId>, u32> = AHashMap::new();
        let mut atoms_observed = 0usize;

        for frame in frames {
            let sequence: Vec<NeuronId> = self
                .encoding
                .atomize(frame)
                .into_iter()
                .map(|label| self.ensure_atom(label, tick))
                .collect();
            atoms_observed += sequence.len();
            for &id in &sequence {
                if let Some(neuron) = self.neurons.get_mut(id as usize) {
                    neuron.use_count = neuron.use_count.saturating_add(1);
                    neuron.last_fired_tick = tick;
                }
            }
            let frame_max = max_len.min(sequence.len());
            if frame_max < 2 {
                continue;
            }
            for len in 2..=frame_max {
                for window in sequence.windows(len) {
                    let count = counts.entry(window.to_vec()).or_insert(0);
                    *count = count.saturating_add(1);
                }
            }
        }

        let mut candidates: Vec<(Vec<NeuronId>, u32)> = counts
            .into_iter()
            .filter(|(_, count)| *count >= min_recurrence)
            .collect();
        candidates.sort_by(|(a_seq, a_count), (b_seq, b_count)| {
            let a_evidence = (*a_count as u64) * (a_seq.len() as u64);
            let b_evidence = (*b_count as u64) * (b_seq.len() as u64);
            b_evidence
                .cmp(&a_evidence)
                .then_with(|| b_count.cmp(a_count))
                .then_with(|| b_seq.len().cmp(&a_seq.len()))
                .then_with(|| a_seq.cmp(b_seq))
        });
        let recurring_candidates = candidates.len();
        let before = self.concept_count();
        for (sequence, _) in candidates.into_iter().take(max_promotions) {
            self.promote_to_concept(sequence, tick);
        }

        PretrainReport {
            frames: frames.len(),
            atoms_observed,
            recurring_candidates,
            concepts_promoted: self.concept_count().saturating_sub(before),
        }
    }

    /// Atomize one frame for a cross-pool pre-training episode without
    /// ordinary within-pool recurrence accounting or concept emergence.
    /// Existing encoders remain authoritative and every returned id is an
    /// ordinary atom neuron, so direct bindings stay grounded in the same
    /// substrate as live observations.
    pub fn ensure_frame_atoms_for_pretrain(&mut self, frame: &[u8], tick: u64) -> Vec<NeuronId> {
        let sequence: Vec<NeuronId> = self
            .encoding
            .atomize(frame)
            .into_iter()
            .map(|label| self.ensure_atom(label, tick))
            .collect();
        for &id in &sequence {
            if let Some(neuron) = self.neurons.get_mut(id as usize) {
                neuron.use_count = neuron.use_count.saturating_add(1);
                neuron.last_fired_tick = tick;
            }
        }
        self.last_observed_sequence = sequence.clone();
        sequence
    }

    /// Return an ordered pretraining frame, collapsing it to one reusable,
    /// atom-grounded concept only after recurrence. The first observation
    /// remains raw atoms, so unique targets have no concept overhead. From the
    /// second observation onward, paraphrase bindings store one shared concept
    /// reference instead of another full response membership vector.
    pub fn ensure_frame_concept_for_pretrain(&mut self, frame: &[u8], tick: u64) -> Vec<NeuronId> {
        let sequence = self.ensure_frame_atoms_for_pretrain(frame, tick);
        if sequence.is_empty() {
            return Vec::new();
        }
        if let Some(id) = self.concept_id_for_sequence(&sequence) {
            if let Some(neuron) = self.neurons.get_mut(id as usize) {
                neuron.use_count = neuron.use_count.saturating_add(1);
                neuron.last_fired_tick = tick;
            }
            return vec![id];
        }

        let recurrence = match self.increment_sequence_recurrence(&sequence) {
            Ok(recurrence) => recurrence,
            Err(error) => {
                eprintln!(
                    "refusing concept promotion because cold recurrence lookup failed in pool {}: {error}",
                    self.config.id
                );
                return sequence;
            }
        };
        if recurrence < 2 {
            return sequence;
        }

        let mut hasher = DefaultHasher::new();
        sequence.hash(&mut hasher);
        let label = format!("pretrain-frame:{}:{:016x}", sequence.len(), hasher.finish(),);
        let members: Vec<NeuronRef> = sequence
            .iter()
            .map(|&id| NeuronRef::new(self.config.id, id))
            .collect();
        let id = self.neurons.len() as NeuronId;
        let neuron = Neuron::new_concept(id, label.clone(), NeuronKind::Excitatory, members, tick);
        let id = self.append_neuron(neuron, label);
        self.concept_sequence_to_id.insert(sequence.clone(), id);
        if sequence.len() <= 512 {
            self.concept_multiset_to_id.entry(sequence).or_insert(id);
        }
        vec![id]
    }

    fn sequence_recurrence_in_generation(
        &self,
        reference: AuxiliaryRecordRef,
        sequence: &[NeuronId],
    ) -> std::io::Result<Option<u32>> {
        const LEDGER_MAGIC: &[u8; 8] = b"W1ZSEQ01";
        let store = self.wbrain_store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cold sequence ledger has no .wbrain store",
            )
        })?;
        let mut header = [0_u8; 24];
        store.read_auxiliary_exact(reference, 0, &mut header)?;
        if &header[..8] != LEDGER_MAGIC {
            let key = Self::concept_posting_key(SEQUENCE_RECURRENCE_KEY, sequence);
            return posting_index::lookup(store, reference, &key, 1)
                .map(|counts| counts.into_iter().next());
        }
        let bucket_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
        if bucket_count == 0 || !bucket_count.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid cold sequence bucket count",
            ));
        }
        let mut hasher = blake3::Hasher::new();
        for id in sequence {
            hasher.update(&id.to_le_bytes());
        }
        let digest = hasher.finalize();
        let hash = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let bucket = hash & (bucket_count - 1);
        let mut raw = [0_u8; 8];
        store.read_auxiliary_exact(reference, 24 + bucket * 8, &mut raw)?;
        let mut link = u64::from_le_bytes(raw);
        let mut traversed = 0_u64;
        while link != 0 {
            traversed += 1;
            if traversed > 1_000_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "cold sequence chain cycle",
                ));
            }
            let record = link - 1;
            let mut record_header = [0_u8; 20];
            store.read_auxiliary_exact(reference, record, &mut record_header)?;
            let next = u64::from_le_bytes(record_header[..8].try_into().unwrap());
            let record_hash = u64::from_le_bytes(record_header[8..16].try_into().unwrap());
            let member_count =
                u32::from_le_bytes(record_header[16..20].try_into().unwrap()) as usize;
            if record_hash == hash && member_count == sequence.len() {
                let mut encoded_members = vec![0_u8; member_count * 4];
                store.read_auxiliary_exact(reference, record + 20, &mut encoded_members)?;
                let matches =
                    encoded_members
                        .chunks_exact(4)
                        .zip(sequence)
                        .all(|(raw, expected)| {
                            u32::from_le_bytes(raw.try_into().unwrap()) == *expected
                        });
                if matches {
                    let mut member_raw = [0_u8; 4];
                    store.read_auxiliary_exact(
                        reference,
                        record + 20 + (member_count as u64) * 4,
                        &mut member_raw,
                    )?;
                    return Ok(Some(u32::from_le_bytes(member_raw)));
                }
            }
            link = next;
        }
        Ok(None)
    }

    fn cold_sequence_recurrence(&self, sequence: &[NeuronId]) -> std::io::Result<Option<u32>> {
        let Some(root) = self.legacy_sequence_ledger else {
            return Ok(None);
        };
        for reference in self
            .auxiliary_generations(root, SEQUENCE_GENERATION_DIRECTORY_MAGIC, "sequence ledger")?
            .into_iter()
            .rev()
        {
            if let Some(count) = self.sequence_recurrence_in_generation(reference, sequence)? {
                return Ok(Some(count));
            }
        }
        Ok(None)
    }

    fn concept_posting_key(kind: u8, sequence: &[NeuronId]) -> Vec<u8> {
        let mut key = Vec::with_capacity(1 + sequence.len() * 4);
        key.push(kind);
        for id in sequence {
            key.extend_from_slice(&id.to_le_bytes());
        }
        key
    }

    fn auxiliary_generations(
        &self,
        root: AuxiliaryRecordRef,
        directory_magic: &[u8; 8],
        description: &str,
    ) -> std::io::Result<Vec<AuxiliaryRecordRef>> {
        let store = self.wbrain_store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cold concept index has no .wbrain store",
            )
        })?;
        let mut magic = [0_u8; 8];
        store.read_auxiliary_exact(root, 0, &mut magic)?;
        if &magic != directory_magic {
            return Ok(vec![root]);
        }
        let mut raw_count = [0_u8; 8];
        store.read_auxiliary_exact(root, 8, &mut raw_count)?;
        let count = u64::from_le_bytes(raw_count);
        let required = 16_u64
            .checked_add(count.checked_mul(16).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{description} generation directory overflow"),
                )
            })?)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{description} generation directory overflow"),
                )
            })?;
        if required > root.len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{description} generation directory is truncated"),
            ));
        }
        let capacity = usize::try_from(count).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{description} generation count exceeds address space"),
            )
        })?;
        let mut generations = Vec::with_capacity(capacity);
        for index in 0..count {
            let mut raw = [0_u8; 16];
            store.read_auxiliary_exact(root, 16 + index * 16, &mut raw)?;
            let reference = AuxiliaryRecordRef {
                offset: u64::from_le_bytes(raw[..8].try_into().unwrap()),
                len: u64::from_le_bytes(raw[8..].try_into().unwrap()),
            };
            if reference.offset == 0 || reference.len < 8 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{description} generation contains an invalid reference"),
                ));
            }
            generations.push(reference);
        }
        Ok(generations)
    }

    fn concept_id_in_generation(
        &self,
        reference: AuxiliaryRecordRef,
        key_kind: u8,
        sequence: &[NeuronId],
    ) -> std::io::Result<Option<NeuronId>> {
        const INDEX_MAGIC: &[u8; 8] = b"W1ZCID01";
        let store = self.wbrain_store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cold concept index has no .wbrain store",
            )
        })?;
        let mut header = [0_u8; 24];
        store.read_auxiliary_exact(reference, 0, &mut header)?;
        if &header[..8] != INDEX_MAGIC {
            let key = Self::concept_posting_key(key_kind, sequence);
            return posting_index::lookup(store, reference, &key, 1)
                .map(|ids| ids.into_iter().next());
        }
        if key_kind != CONCEPT_SEQUENCE_KEY {
            return Ok(None);
        }
        let bucket_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
        if bucket_count == 0 || !bucket_count.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid cold concept bucket count",
            ));
        }
        let mut hasher = blake3::Hasher::new();
        for id in sequence {
            hasher.update(&id.to_le_bytes());
        }
        let digest = hasher.finalize();
        let hash = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let bucket = hash & (bucket_count - 1);
        let mut raw = [0_u8; 8];
        store.read_auxiliary_exact(reference, 24 + bucket * 8, &mut raw)?;
        let mut link = u64::from_le_bytes(raw);
        let mut traversed = 0_u64;
        while link != 0 {
            traversed += 1;
            if traversed > 1_000_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "cold concept chain cycle",
                ));
            }
            let record = link - 1;
            let mut record_header = [0_u8; 20];
            store.read_auxiliary_exact(reference, record, &mut record_header)?;
            let next = u64::from_le_bytes(record_header[..8].try_into().unwrap());
            let record_hash = u64::from_le_bytes(record_header[8..16].try_into().unwrap());
            let member_count =
                u32::from_le_bytes(record_header[16..20].try_into().unwrap()) as usize;
            if record_hash == hash && member_count == sequence.len() {
                let mut encoded_members = vec![0_u8; member_count * 4];
                store.read_auxiliary_exact(reference, record + 20, &mut encoded_members)?;
                let matches =
                    encoded_members
                        .chunks_exact(4)
                        .zip(sequence)
                        .all(|(raw, expected)| {
                            u32::from_le_bytes(raw.try_into().unwrap()) == *expected
                        });
                if matches {
                    let mut member_raw = [0_u8; 4];
                    store.read_auxiliary_exact(
                        reference,
                        record + 20 + (member_count as u64) * 4,
                        &mut member_raw,
                    )?;
                    return Ok(Some(u32::from_le_bytes(member_raw)));
                }
            }
            link = next;
        }
        Ok(None)
    }

    fn cold_concept_id(
        &self,
        key_kind: u8,
        sequence: &[NeuronId],
    ) -> std::io::Result<Option<NeuronId>> {
        let Some(root) = self.legacy_concept_sequence_index else {
            return Ok(None);
        };
        for reference in self
            .auxiliary_generations(root, CONCEPT_GENERATION_DIRECTORY_MAGIC, "concept index")?
            .into_iter()
            .rev()
        {
            if let Some(id) = self.concept_id_in_generation(reference, key_kind, sequence)? {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    fn concept_id_for_sequence(&self, sequence: &[NeuronId]) -> Option<NeuronId> {
        if let Some(id) = self.concept_sequence_to_id.get(sequence) {
            return Some(*id);
        }
        match self.cold_concept_id(CONCEPT_SEQUENCE_KEY, sequence) {
            Ok(id) => id,
            Err(error) => {
                eprintln!(
                    "cold concept lookup failed in pool {}: {error}",
                    self.config.id
                );
                None
            }
        }
    }

    fn concept_id_for_atom_leaves(&self, sequence: &[NeuronId]) -> Option<NeuronId> {
        if let Some(id) = self.concept_multiset_to_id.get(sequence) {
            return Some(*id);
        }
        match self.cold_concept_id(CONCEPT_ATOM_LEAVES_KEY, sequence) {
            Ok(id) => id,
            Err(error) => {
                eprintln!(
                    "cold atom-leaf concept lookup failed in pool {}: {error}",
                    self.config.id
                );
                None
            }
        }
    }

    fn increment_sequence_recurrence(
        &mut self,
        sequence: &SequenceFingerprint,
    ) -> std::io::Result<u32> {
        if let Some(count) = self.sequences.get_mut(sequence) {
            *count = count.saturating_add(1);
            return Ok(*count);
        }
        let old = self.cold_sequence_recurrence(sequence)?.unwrap_or(0);
        let next = old.saturating_add(1);
        self.sequences.insert(sequence.clone(), next);
        Ok(next)
    }

    /// Remove query-local activation while preserving all learned neurons,
    /// terminals, weights, counters, and consolidation state.
    pub fn clear_prediction_activation(&mut self) {
        self.activation.clear();
        self.currently_firing.clear();
        self.last_observed_sequence.clear();
    }
    /// Atom sequence from the most recent observe_frame call.
    /// Used by the decoder for anagram disambiguation.
    pub fn last_observed_sequence(&self) -> &[NeuronId] {
        &self.last_observed_sequence
    }
    pub fn get(&self, id: NeuronId) -> Option<&Neuron> {
        self.neurons.get(id as usize)
    }
    pub fn get_mut(&mut self, id: NeuronId) -> Option<&mut Neuron> {
        self.neurons.get_mut(id as usize)
    }
    pub fn iter_neurons(&self) -> impl Iterator<Item = &Neuron> {
        self.neurons.iter()
    }
    pub fn iter_neurons_mut(&mut self) -> impl Iterator<Item = &mut Neuron> {
        self.neurons.iter_mut()
    }

    pub fn label_to_id(&self, label: &str) -> Option<NeuronId> {
        self.label_to_id.get(label).copied().or_else(|| {
            self.wbrain_store
                .as_ref()
                .and_then(|store| store.label_to_id(label))
        })
    }

    /// Insert a fully-constructed concept neuron.  Used by the Brain to
    /// place binding concepts (members span multiple pools) into the
    /// binding pool without going through atom-stream emergence.  The
    /// neuron's `id` field is overwritten to match the assigned slot.
    pub fn append_neuron(&mut self, mut neuron: Neuron, label: String) -> NeuronId {
        let id = self.neurons.len() as NeuronId;
        neuron.id = id;
        // Per ARCHITECTURE §17.9: WAL append BEFORE in-memory mutation
        // is considered durable.  If neuron has members it's a concept
        // (binding concepts have cross-pool members); otherwise it's an
        // atom path (rare here — atoms usually come through ensure_atom).
        let event = if neuron.members.is_empty() {
            WalEvent::AtomCreated {
                pool_id: self.config.id,
                id,
                label: label.clone(),
                kind: neuron.kind,
                born_tick: neuron.born_tick,
            }
        } else {
            WalEvent::ConceptEmerged {
                pool_id: self.config.id,
                id,
                label: label.clone(),
                kind: neuron.kind,
                members: neuron.members.clone(),
                born_tick: neuron.born_tick,
            }
        };
        if let Err(e) = self.store.append(&event) {
            tracing::warn!("WAL append failed for append_neuron(id={}): {}", id, e);
        }
        self.neurons.push(neuron);
        self.bloom.insert(&label);
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
            if m.pool != self.config.id {
                continue;
            }
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
            if m.pool != self.config.id {
                continue;
            }
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
        // Stage 17.4 step 5: snapshot the cold-tier offset index so a
        // process restart preserves what's-on-disk vs in-RAM.  The
        // cold-tier file itself is data on disk — its name is derived
        // from pool id (see brain_server's attach_cold_tiers call), so
        // restoring the index is sufficient to re-link to it.
        let cold_offsets: Vec<(NeuronId, u64)> =
            self.cold_offsets.iter().map(|(k, v)| (*k, *v)).collect();
        crate::persistence::PoolSnapshot {
            config: self.config.clone(),
            neurons: self.neurons.dense_clone(),
            label_to_id,
            recent_atoms: self.recent_atoms.clone(),
            sequences,
            cold_offsets,
        }
    }

    /// Construct a Pool from a snapshot and a fresh encoding.  The
    /// caller is responsible for supplying an encoding compatible with
    /// the atom labels stored in the snapshot — e.g. the same
    /// `BytePassthroughEncoding { prefix }` used originally.  If the
    /// encoding contract differs, atoms still hold their labels but
    /// decode-side reassembly may produce different bytes.
    pub fn from_snapshot(
        snap: crate::persistence::PoolSnapshot,
        encoding: Box<dyn AtomEncoding>,
    ) -> Self {
        let mut label_to_id = AHashMap::new();
        for (k, v) in snap.label_to_id {
            label_to_id.insert(k, v);
        }
        let mut sequences = AHashMap::new();
        for (k, v) in snap.sequences {
            sequences.insert(k, v);
        }
        // Stage 17.3 — rebuild the Bloom side-car from the restored
        // label index.  Bloom is not part of the snapshot format (it's a
        // probabilistic structure that's cheap to rebuild and avoids a
        // breaking format change).  Rebuild before constructing Self so
        // we don't have to mutate after the move.
        let mut bloom = CountingBloom::with_expected_capacity(label_to_id.len().max(100_000));
        for k in label_to_id.keys() {
            bloom.insert(k);
        }

        // Rebuild every neuron's terminal_idx — it's #[serde(skip)] so
        // restored snapshots load with an empty index, which would make
        // reinforce_terminal think every connection is new (catastrophic:
        // it would push duplicate terminals and double neurogenesis).
        // The rebuild restores the address-by-name invariant before any
        // training-path code can touch a neuron.  One-time O(total_terminals)
        // cost on load — ~5-10s at 170M terminals on this host.
        let mut neurons = snap.neurons;
        for n in neurons.iter_mut() {
            n.rebuild_terminal_idx();
        }
        // Compute exact count once before moving neurons into the
        // struct.  Subsequent mutations maintain it incrementally.
        let total_terminals_init: usize = neurons.iter().map(|n| n.terminals.len()).sum();
        let mut pool = Self {
            config: snap.config,
            encoding,
            neurons: NeuronSlots::from_dense(neurons),
            label_to_id,
            concept_multiset_to_id: AHashMap::new(),
            concept_sequence_to_id: AHashMap::new(),
            legacy_concept_sequence_index: None,
            recent_atoms: snap.recent_atoms,
            sequences,
            legacy_sequence_ledger: None,
            currently_firing: AHashSet::new(),
            activation: AHashMap::new(),
            recent_surprise: 1.0,
            firing_rate_ema: 0.0,
            concept_count_ema: 0.0,
            terminal_count_ema: 0.0,
            pending_promotions: Vec::new(),
            domain_for_new: 0,
            total_terminals: total_terminals_init,
            decode_precision_ema: 0.0,
            last_observed_sequence: Vec::new(),
            // Restored-from-snapshot pools start with NoopStore; caller
            // re-attaches the live WAL backend via set_store after restore
            // so events from subsequent observations flow to the right log.
            store: Arc::new(NoopStore),
            bloom,
            // Stage 17.4 step 5: restore cold-tier offset index AND
            // the evicted set from the snapshot.  The actual cold-tier
            // file handle (`cold_tier`) is set separately by the brain
            // via attach_cold_tiers after fabric registration, because
            // the data_dir isn't known at Pool construction time.
            cold_tier: None,
            tiered_store: None, // §18.12 step 4b
            wbrain_store: None,
            shape_only_residents: AHashSet::new(),
            cold_offsets: snap.cold_offsets.iter().copied().collect(),
            evicted: snap.cold_offsets.iter().map(|(id, _)| *id).collect(),
        };
        // Rebuild the multiset dedup index from restored concept
        // neurons.  The index isn't part of the snapshot format (it
        // can always be rebuilt from members) so restore from older
        // snapshots remains lossless.
        let concept_ids: Vec<NeuronId> = pool
            .neurons
            .iter()
            .filter(|n| !n.is_atom())
            .map(|n| n.id)
            .collect();
        for cid in concept_ids {
            let member_ids: Vec<NeuronId> = pool.neurons[cid as usize]
                .members
                .iter()
                .filter(|m| m.pool == pool.config.id)
                .map(|m| m.neuron)
                .collect();
            let leaves = pool.expand_to_atom_leaves_bounded(&member_ids, 512);
            // ORDERED-sequence dedup (was: sorted multiset).  Each
            // unique atom-leaf order gets its own canonical concept,
            // so anagrams like sad/das remain distinct.
            if let Some(leaves) = leaves {
                pool.concept_multiset_to_id.entry(leaves).or_insert(cid);
            }

            // Sequence index keyed by the IMMEDIATE member ids (not
            // expanded to atom leaves) — collapse_tail_to_concept
            // looks up by what's currently in recent_atoms, which may
            // already contain concept ids from prior collapses.
            pool.concept_sequence_to_id.entry(member_ids).or_insert(cid);
        }
        pool
    }

    /// Expand a hierarchical concept only while its flattened atom sequence
    /// remains within `limit`. Mature concepts-of-concepts can represent very
    /// long repeated ancestry; eagerly materialising every flattened sequence
    /// during restore made a small checkpoint consume tens of gigabytes.
    fn expand_to_atom_leaves_bounded(
        &self,
        member_ids: &[NeuronId],
        limit: usize,
    ) -> Option<Vec<NeuronId>> {
        fn append_ids(
            pool: &Pool,
            ids: &[NeuronId],
            limit: usize,
            out: &mut Vec<NeuronId>,
        ) -> bool {
            for &id in ids {
                let Some(neuron) = pool.neurons.get(id as usize) else {
                    continue;
                };
                if neuron.is_atom() {
                    if out.len() == limit {
                        return false;
                    }
                    out.push(id);
                } else if !append_refs(pool, &neuron.members, limit, out) {
                    return false;
                }
            }
            true
        }
        fn append_refs(
            pool: &Pool,
            members: &[NeuronRef],
            limit: usize,
            out: &mut Vec<NeuronId>,
        ) -> bool {
            for member in members
                .iter()
                .filter(|member| member.pool == pool.config.id)
            {
                if !append_ids(pool, std::slice::from_ref(&member.neuron), limit, out) {
                    return false;
                }
            }
            true
        }
        let mut leaves = Vec::new();
        append_ids(self, member_ids, limit, &mut leaves).then_some(leaves)
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
    pub fn observe_frame(
        &mut self,
        frame: &[u8],
        tick: u64,
        profile: Option<&crate::fabric::ObserveProfile>,
    ) -> Vec<NeuronId> {
        use std::sync::atomic::Ordering::Relaxed;
        let atomize_t0 = std::time::Instant::now();
        let labels = self.encoding.atomize(frame);
        if let Some(p) = profile {
            p.atomize_ns
                .fetch_add(atomize_t0.elapsed().as_nanos() as u64, Relaxed);
        }
        let mut fired = Vec::with_capacity(labels.len());
        let decay_rate = self.config.decay_rate;
        let prune_floor = self.config.prune_floor;
        let mut pruned_terminals_this_call: usize = 0;
        // Per-phase accumulators for the per-atom loop below.  Folded
        // into the Arc<ObserveProfile> once at the bottom so we don't
        // hit the atomic on every iteration.
        let mut atom_fire_ns: u64 = 0;
        let mut lazy_decay_ns: u64 = 0;
        let mut collapse_ns: u64 = 0;
        let mut concept_emergence_ns: u64 = 0;

        // Predictive-coding prediction (P3): before clearing the
        // firing set, compute the intra-pool prediction of what will
        // fire next based on terminals of currently-firing neurons.
        // Substrate of L6→L4 cortical prediction.  Always computed
        // (modest overhead) so `recent_surprise` is always available
        // as a ControlSignal — knobs can read it whether or not
        // predict_gate_mode is non-Constant.
        let prediction: AHashSet<NeuronId> = if !self.currently_firing.is_empty() {
            let mut set = AHashSet::with_capacity(self.currently_firing.len() * 4);
            for &nid in &self.currently_firing {
                if let Some(n) = self.neurons.get(nid as usize) {
                    for t in &n.terminals {
                        if t.target.pool == self.config.id {
                            set.insert(t.target.neuron);
                        }
                    }
                }
            }
            set
        } else {
            AHashSet::new()
        };

        self.activation.clear();
        self.currently_firing.clear();

        for label in labels {
            let atom_t0 = std::time::Instant::now();
            let id = self.ensure_atom(label, tick);
            self.activation.insert(id, 1.0);
            self.currently_firing.insert(id);
            fired.push(id);
            self.push_recent(id);
            atom_fire_ns += atom_t0.elapsed().as_nanos() as u64;

            let decay_t0 = std::time::Instant::now();
            if let Some(n) = self.neurons.get_mut(id as usize) {
                n.use_count = n.use_count.saturating_add(1);
                n.last_fired_tick = tick;
                // Lazy decay: apply this neuron's pending decay backlog
                // before any code reads or writes its terminals this tick.
                // Math identity with eager mode: (1-ε)^k applied once on
                // access == k single-tick eager applications.  Fast-path
                // for elapsed==0.
                pruned_terminals_this_call += n.apply_pending_decay(tick, decay_rate, prune_floor);
            }
            lazy_decay_ns += decay_t0.elapsed().as_nanos() as u64;

            // Mini-column collapse: pop matched atoms from the tail of
            // recent_atoms and push the concept id in their place.
            // Loop so level-2 columns also collapse in one pass.
            let collapse_t0 = std::time::Instant::now();
            while self.collapse_tail_to_concept(tick) {}
            collapse_ns += collapse_t0.elapsed().as_nanos() as u64;

            // Per-atom emergence counting.  Patterns ending at the
            // current tail entry (which may be a concept after
            // collapse) get their counts bumped.
            let emergence_t0 = std::time::Instant::now();
            self.check_concept_emergence(tick);
            concept_emergence_ns += emergence_t0.elapsed().as_nanos() as u64;
        }

        let end_t0 = std::time::Instant::now();
        // Predictive-coding surprise update (P3).  Always update
        // (cheap) so the EMA can serve as a ControlSignal for any
        // ControlMode that reads it.  Gate decisions downstream
        // (`check_concept_emergence` and any DrivenBy on Surprise)
        // read this value.
        if !self.currently_firing.is_empty() {
            let unpredicted: usize = self
                .currently_firing
                .iter()
                .filter(|id| !prediction.contains(id))
                .count();
            let surprise = unpredicted as f32 / self.currently_firing.len() as f32;
            // EMA with α = 0.3 (recent surprise weighted ~3 ticks).
            self.recent_surprise = self.recent_surprise * 0.7 + surprise * 0.3;
        }

        // P1 k-WTA sparsity gate.  CRITICAL: must run BEFORE the
        // Fabric captures the moment fingerprint from currently_firing,
        // otherwise mega-bindings still form (the falsification cause).
        // Originally lived in tick_housekeeping which runs AFTER moment
        // capture in Fabric::tick — that placement made the gate
        // useless against the binding-pool runaway.
        self.apply_kwta_sparsity();

        // Update EMAs feeding the dynamical control state.  Done
        // AFTER k-WTA so firing_rate reflects post-gate sparsity.
        self.update_emas();

        // Snapshot the ordered firing sequence for this observation.
        // Read by the decoder's sequence-match preempt to distinguish
        // anagram queries (e.g. 'sad' [s,a,d] vs 'das' [d,a,s]).
        self.last_observed_sequence = fired.clone();

        // Maintain the O(1) terminals counter for the lazy-decay
        // accumulations from firing neurons.
        self.total_terminals = self
            .total_terminals
            .saturating_sub(pruned_terminals_this_call);
        let end_of_frame_ns = end_t0.elapsed().as_nanos() as u64;

        // Fold per-atom accumulators into the shared Arc<ObserveProfile>
        // with a single atomic add per phase (rather than per atom).
        if let Some(p) = profile {
            p.atom_fire_ns.fetch_add(atom_fire_ns, Relaxed);
            p.lazy_decay_ns.fetch_add(lazy_decay_ns, Relaxed);
            p.collapse_ns.fetch_add(collapse_ns, Relaxed);
            p.concept_emergence_ns
                .fetch_add(concept_emergence_ns, Relaxed);
            p.end_of_frame_ns.fetch_add(end_of_frame_ns, Relaxed);
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
        if buf_len < 2 {
            return false;
        }
        let max_len = self.config.max_concept_member_count.min(buf_len);

        let mut found: Option<(usize, NeuronId)> = None;
        // Longest tail wins — wider column receptive field.
        //
        // Vec<NeuronId>-keyed lookup replaces the old string-concat
        // composite_label path.  The old version allocated a fresh
        // String for every (atom, length) probe and hashed it — for a
        // 200-atom observe with max_len=7, that was up to 1400 heap
        // allocations + char-by-char hashes per call.  Vec<u32> hashes
        // ~4× faster and has zero allocation when keyed into AHashMap.
        // Empirically the same semantics: the index is populated from
        // members on snapshot restore and maintained in lockstep with
        // label_to_id on every promote_to_concept.
        let mut probe: Vec<NeuronId> = Vec::with_capacity(max_len);
        for len in (2..=max_len).rev() {
            let start = buf_len - len;
            probe.clear();
            probe.extend(self.recent_atoms.iter().skip(start).copied());
            if let Some(cid) = self.concept_id_for_sequence(&probe) {
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
    /// Lazy variant of tick_housekeeping — decays only neurons that
    /// fired this tick.  Other neurons' decay is amortised at next
    /// access via Neuron::apply_pending_decay (uses each neuron's
    /// last_decayed_tick).  Cost: O(|currently_firing|), not O(N).
    ///
    /// Mathematically identical to eager mode because
    /// `weight × (1-ε)^k ≡ k applications of (1-ε)`.  Empirically
    /// validated against the canonical toddler-32 baseline.
    pub fn tick_housekeeping_lazy(&mut self, current_tick: u64) {
        let decay = self.config.decay_rate;
        let floor = self.config.prune_floor;
        // Clone the firing set so we can mutate neurons while iterating.
        let firing_ids: Vec<NeuronId> = self.currently_firing.iter().copied().collect();
        let mut total_pruned: usize = 0;
        for nid in firing_ids {
            if let Some(n) = self.neurons.get_mut(nid as usize) {
                total_pruned += n.apply_pending_decay(current_tick, decay, floor);
            }
        }
        self.total_terminals = self.total_terminals.saturating_sub(total_pruned);
        self.apply_heterosynaptic_ltd(current_tick);
    }

    pub fn tick_housekeeping(&mut self, current_tick: u64) {
        let decay = self.config.decay_rate;
        let floor = self.config.prune_floor;
        let mut total_pruned: usize = 0;
        for n in self.neurons.iter_mut() {
            total_pruned += n.decay_and_prune(decay, floor);
            // Eager mode keeps last_decayed_tick in sync so a future
            // switch to lazy mode doesn't double-apply elapsed decay.
            n.last_decayed_tick = current_tick;
        }
        // Maintain the O(1) total_terminals counter — decay_and_prune
        // returns the number of terminals it dropped below the floor.
        self.total_terminals = self.total_terminals.saturating_sub(total_pruned);
        self.apply_heterosynaptic_ltd(current_tick);
        // Note: k-WTA sparsity gate moved to observe_frame() so it
        // runs BEFORE the Fabric captures the moment fingerprint.
        // Calling it again here would be redundant.
    }

    /// Heterosynaptic LTD: for each neuron whose terminal was
    /// reinforced this tick (terminal.last_fired_tick == current_tick),
    /// weaken all of its OTHER terminals by `weight *= 1 - ratio`.
    /// Biological mechanism: when one synapse undergoes LTP, the
    /// neuron's other synapses undergo a small homeostatic LTD that
    /// keeps total dendritic input normalised.  Without this,
    /// pure Hebbian potentiation accumulates indefinitely.
    fn apply_heterosynaptic_ltd(&mut self, current_tick: u64) {
        let state = self.control_state();
        let ratio = self
            .config
            .heterosynaptic_ltd_mode
            .evaluate(&state)
            .clamp(0.0, 0.9);
        if ratio <= 0.0 {
            return;
        }
        let keep = 1.0 - ratio;
        for n in self.neurons.iter_mut() {
            let has_recent = n
                .terminals
                .iter()
                .any(|t| t.last_fired_tick == current_tick);
            if !has_recent {
                continue;
            }
            for t in n.terminals.iter_mut() {
                if t.last_fired_tick != current_tick {
                    t.weight *= keep;
                }
            }
        }
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
        // Evaluate ControlMode against current pool ControlState.
        // Constant(1.0) returns 1.0 → early return (no behaviour change).
        // DrivenBy adapts each call based on the substrate's own signals.
        let state = self.control_state();
        let frac = self.config.sparsity_mode.evaluate(&state).clamp(0.01, 1.0);
        if frac >= 1.0 {
            return;
        }
        let n_firing = self.currently_firing.len();
        if n_firing == 0 {
            return;
        }
        let target_k = ((frac * n_firing as f32).ceil() as usize)
            .max(self.config.sparsity_min_neurons)
            .min(n_firing);
        if target_k >= n_firing {
            return;
        }

        // Collect (id, activation) for all currently-firing neurons.
        let mut ranked: Vec<(NeuronId, f32)> = self
            .currently_firing
            .iter()
            .map(|&id| (id, *self.activation.get(&id).unwrap_or(&0.0)))
            .collect();
        // Sort descending by activation; ties broken by id (stable).
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
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
        stale_ticks: u64,
        current_tick: u64,
    ) -> ahash::AHashSet<NeuronId> {
        // Identify pruneable concept ids first to avoid borrow conflicts.
        let mut to_prune: ahash::AHashSet<NeuronId> = ahash::AHashSet::new();
        for n in self.neurons.iter() {
            if n.is_atom() {
                continue;
            }
            let age = current_tick.saturating_sub(n.last_fired_tick);
            if n.use_count < min_use_count && age > stale_ticks {
                to_prune.insert(n.id);
            }
        }
        if to_prune.is_empty() {
            return to_prune;
        }

        // Zero outgoing terminals of pruned concepts and remove their
        // entries from label_to_id (so they can't collapse again).
        // Also decrement the Bloom side-car (Stage 17.3) so a later
        // `label_might_exist` query for the same label correctly says no.
        let mut dropped: usize = 0;
        for &cid in &to_prune {
            if let Some(n) = self.neurons.get_mut(cid as usize) {
                dropped += n.terminals.len();
                n.terminals.clear();
                n.terminal_idx.clear();
                let label = n.label.clone();
                self.label_to_id.remove(&label);
                self.bloom.remove(&label);
            }
        }

        // Remove inbound terminals targeting pruned concepts from EVERY
        // other neuron in this pool.  Cross-pool terminals targeting
        // these concepts from other pools must be cleaned up by the
        // caller (Fabric::sleep does this).  Every neuron whose
        // terminals get retained must have its terminal_idx rebuilt.
        let pool_id = self.config.id;
        for n in self.neurons.iter_mut() {
            let before = n.terminals.len();
            n.terminals
                .retain(|t| !(t.target.pool == pool_id && to_prune.contains(&t.target.neuron)));
            if n.terminals.len() != before {
                n.rebuild_terminal_idx();
                dropped += before - n.terminals.len();
            }
        }
        self.total_terminals = self.total_terminals.saturating_sub(dropped);

        // Clear currently_firing / activation entries for pruned ids.
        for cid in &to_prune {
            self.currently_firing.remove(cid);
            self.activation.remove(cid);
        }

        // Remove sequences map entries that reference pruned ids.
        self.sequences
            .retain(|k, _| !k.iter().any(|id| to_prune.contains(id)));

        // Remove recent_atoms entries that point to pruned ids.
        self.recent_atoms.retain(|id| !to_prune.contains(id));

        to_prune
    }

    /// Remove terminals in this pool that target any of the
    /// specified (pool, neuron) ids.  Used by `Fabric::sleep` to
    /// clean up cross-pool terminals after another pool pruned
    /// some of its concepts.
    pub fn prune_inbound_to(&mut self, targets: &ahash::AHashSet<NeuronRef>) -> usize {
        let mut removed = 0;
        for n in self.neurons.iter_mut() {
            let before = n.terminals.len();
            n.terminals.retain(|t| !targets.contains(&t.target));
            if n.terminals.len() != before {
                n.rebuild_terminal_idx();
                removed += before - n.terminals.len();
            }
        }
        self.total_terminals = self.total_terminals.saturating_sub(removed);
        removed
    }

    fn ensure_atom(&mut self, label: String, tick: u64) -> NeuronId {
        if let Some(id) = self.label_to_id(&label) {
            if let Err(error) = self.ensure_loaded(id) {
                tracing::warn!("training page-in failed for atom {}: {}", id, error);
            }
            return id;
        }
        let id = self.neurons.len() as NeuronId;
        let mut neuron = Neuron::new_atom(id, label.clone(), NeuronKind::Excitatory, tick);
        // Island architecture: stamp every newly-created atom with the
        // pool's current `domain_for_new`.  Operator sets this via
        // /set_domain before training a corpus so the resulting atoms
        // cluster into that island.  Default 0 = global / unassigned,
        // which makes the gate a no-op (current behaviour).
        neuron.domain_id = self.domain_for_new;
        // Per ARCHITECTURE §17.9: append BEFORE in-memory exposure.
        let event = WalEvent::AtomCreated {
            pool_id: self.config.id,
            id,
            label: label.clone(),
            kind: neuron.kind,
            born_tick: tick,
        };
        if let Err(e) = self.store.append(&event) {
            tracing::warn!("WAL append failed for ensure_atom({:?}): {}", label, e);
        }
        self.neurons.push(neuron);
        self.bloom.insert(&label);
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
        // P3 predictive-coding gate.  Only crystallise new concepts
        // when the substrate's recent surprise is above the gate
        // strength.  Already-predicted patterns add no information
        // and would just inflate the concept inventory.
        // Evaluate predict-gate ControlMode against current pool state.
        // Constant(0.0) → 0.0 → `recent_surprise < 0` is never true →
        // emergence always proceeds (backward-compatible default).
        // DrivenBy lets the gate adapt: e.g. `DrivenBy(InvSurprise, 0.5, 0.3)`
        // tightens the gate when the substrate is predicting well.
        let state = self.control_state();
        let gate = self
            .config
            .predict_gate_mode
            .evaluate(&state)
            .clamp(0.0, 0.95);
        if gate > 0.0 && self.recent_surprise < gate {
            return;
        }
        let buf_len = self.recent_atoms.len();
        if buf_len < 2 {
            return;
        }

        let max_len = self.config.max_concept_member_count.min(buf_len);
        let threshold = self.config.concept_emergence_threshold;
        let mut to_promote: Vec<SequenceFingerprint> = Vec::new();

        for len in 2..=max_len {
            let start = buf_len - len;
            let run: SequenceFingerprint = self.recent_atoms.iter().skip(start).copied().collect();
            match self.increment_sequence_recurrence(&run) {
                Ok(count) if count == threshold => to_promote.push(run),
                Ok(_) => {}
                Err(error) => eprintln!(
                    "refusing concept emergence because cold recurrence lookup failed in pool {}: {error}",
                    self.config.id
                ),
            }
        }

        // Deferred-promotion mode (option 3 in the optimisation discussion):
        // when W1Z4RD_DEFER_PROMOTION is set, enqueue promotions instead
        // of crystallising inline.  Observe path becomes pure
        // address-by-name HashMap writes — no neuron allocation, no
        // member↔concept terminal wiring, no WAL append for new
        // neurogenesis.  Structure growth runs during /sleep via
        // drain_pending_promotions.
        //
        // Matches the CLS biological model — hippocampal fast learning
        // (sequence count + threshold detection) happens online; cortical
        // structure consolidation (concept crystallisation, wiring)
        // happens during replay.
        let deferred = std::env::var("W1Z4RD_DEFER_PROMOTION")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if deferred {
            self.pending_promotions.extend(to_promote);
        } else {
            for run in to_promote {
                self.promote_to_concept(run, tick);
            }
        }
    }

    /// Crystallise every deferred promotion that has accumulated since
    /// the last drain.  Called by Brain::sleep on each pool.  Returns
    /// the number of promotions actually applied (some may dedupe
    /// against newly-existing concepts).  The pool's
    /// `pending_promotions` Vec is drained empty.
    ///
    /// Safe to call repeatedly — empty Vec is a no-op.
    pub fn drain_pending_promotions(&mut self, tick: u64) -> usize {
        let pending = std::mem::take(&mut self.pending_promotions);
        let n = pending.len();
        for run in pending {
            self.promote_to_concept(run, tick);
        }
        n
    }

    /// Snapshot of the pending-promotion queue depth.  Surfaced via the
    /// /sleep_pressure HTTP endpoint so the operator can see when the
    /// brain is overdue for a sleep cycle.  Sleep is the only mechanism
    /// that crystallises deferred concepts, so a growing queue means
    /// the corpus is novel relative to the trained fabric.
    pub fn pending_promotion_count(&self) -> usize {
        self.pending_promotions.len()
    }

    fn promote_to_concept(&mut self, members: SequenceFingerprint, tick: u64) {
        // Composite label = concatenation of member labels, separated by a
        // glyph that can't appear in base64-url-safe payloads.  Stable and
        // human-readable for debugging.
        let composite_label: String = members
            .iter()
            .map(|id| self.neurons[*id as usize].label.as_str())
            .collect::<Vec<_>>()
            .join("~");
        if self.label_to_id.contains_key(&composite_label) {
            // Already promoted (e.g. via a different sequence path).  Skip.
            return;
        }

        // Stage 13 — atom-leaf SEQUENCE dedup (was: multiset).
        // Originally we dedup'd by SORTED atom-leaf set so fragments
        // like [f,o,o,d] and [o,o,d,f] (artifacts of destructive
        // collapse) wouldn't multiply.  But anagram-pair prompts
        // ('sad'/'das', 'rose'/'eros', 'cat'/'act') ALSO have
        // identical multisets and the multiset rule collapsed them
        // into one concept — losing the substrate's ability to
        // distinguish.  K-12 'sad' lost to 'das->animal' for this
        // reason.
        //
        // Switch to ORDERED-sequence dedup: only dedup when the
        // atom-leaf sequence is IDENTICAL in order.  Anagrams now
        // emerge as distinct concepts.  Fragment variants from
        // collapse remain a theoretical concern but k-WTA sparsity
        // + heterosynaptic LTD + Hebbian freq weighting in decode
        // now dominate that risk: the canonical fragment gets the
        // highest use_count and outscores noise variants.
        // Exact immediate structure is a bounded deterministic duplicate
        // guard even when the flattened atom ancestry is intentionally too
        // large to materialise.
        if self.concept_id_for_sequence(&members).is_some() {
            return;
        }
        let leaves_seq = self.expand_to_atom_leaves_bounded(&members, 512);
        if leaves_seq
            .as_ref()
            .is_some_and(|leaves| self.concept_id_for_atom_leaves(leaves).is_some())
        {
            return; // canonical concept already exists for this ordered sequence
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
        let member_refs: Vec<NeuronRef> = members
            .iter()
            .map(|m| NeuronRef::new(self.config.id, *m))
            .collect();
        // Per ARCHITECTURE §17.9: append the ConceptEmerged event before
        // the in-memory neuron is exposed.
        let event = WalEvent::ConceptEmerged {
            pool_id: self.config.id,
            id,
            label: composite_label.clone(),
            kind: NeuronKind::Excitatory,
            members: member_refs.clone(),
            born_tick: tick,
        };
        if let Err(e) = self.store.append(&event) {
            tracing::warn!("WAL append failed for promote_to_concept(id={}): {}", id, e);
        }
        let mut concept = Neuron::new_concept(
            id,
            composite_label.clone(),
            NeuronKind::Excitatory,
            member_refs,
            tick,
        );
        // Island architecture: stamp the new concept with the pool's
        // current `domain_for_new` so it joins the active island.
        // When a concept emerges from members that are themselves in a
        // domain, we honour the member's domain if domain_for_new is 0
        // (so emergent higher-order concepts stay in their members'
        // island even if the operator forgot to set domain_for_new).
        concept.domain_id = if self.domain_for_new != 0 {
            self.domain_for_new
        } else {
            // Inherit the dominant domain of members.  Look up each
            // member's domain_id; if any non-zero majority, use it.
            let dom_votes: std::collections::HashMap<u32, u32> = members
                .iter()
                .filter_map(|&m| self.neurons.get(m as usize).map(|n| n.domain_id))
                .filter(|&d| d != 0)
                .fold(std::collections::HashMap::new(), |mut acc, d| {
                    *acc.entry(d).or_insert(0) += 1;
                    acc
                });
            dom_votes
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(d, _)| d)
                .unwrap_or(0)
        };
        // Wire member→concept terminals (atom→concept bottom-up) and
        // concept→member (concept→atom top-down).  Both Hebbian-strengthen
        // on subsequent activations.
        let mut added_terminals: usize = 0;
        for &mid in &members {
            let target = NeuronRef::new(self.config.id, id);
            if let Some(member_neuron) = self.neurons.get_mut(mid as usize) {
                if member_neuron.reinforce_terminal(target, 0.5, tick, self.config.max_weight) {
                    added_terminals += 1;
                }
            }
            if concept.reinforce_terminal(
                NeuronRef::new(self.config.id, mid),
                0.5,
                tick,
                self.config.max_weight,
            ) {
                added_terminals += 1;
            }
        }
        self.total_terminals += added_terminals;
        self.neurons.push(concept);
        self.bloom.insert(&composite_label);
        self.label_to_id.insert(composite_label, id);
        if let Some(leaves_seq) = leaves_seq {
            self.concept_multiset_to_id.insert(leaves_seq, id);
        }
        // Sequence index: keyed by the IMMEDIATE member ids in firing
        // order — matches what collapse_tail_to_concept builds from
        // recent_atoms.  Same lockstep with label_to_id.
        let member_ids: Vec<NeuronId> = members.iter().copied().collect();
        self.concept_sequence_to_id.insert(member_ids, id);
    }
}

#[cfg(test)]
mod resident_slot_tests {
    use super::*;

    #[test]
    fn paged_resident_ids_do_not_scan_or_materialize_the_sleeping_extent() {
        let mut slots = NeuronSlots::sleeping(10_000_000, 0);
        assert!(slots.wake(
            42,
            Neuron::new_atom(42, "test:42".into(), NeuronKind::Excitatory, 1),
        ));
        assert!(slots.wake(
            9_000_000,
            Neuron::new_atom(
                9_000_000,
                "test:9000000".into(),
                NeuronKind::Excitatory,
                1,
            ),
        ));
        assert_eq!(slots.resident_ids(), vec![42, 9_000_000]);
    }
}
