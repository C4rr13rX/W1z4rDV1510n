//! Neuron primitives per [`ARCHITECTURE.md`] §1.2.
//!
//! One unified `Neuron` struct represents atoms, concepts, concept-of-concepts,
//! binding concepts, and action neurons.  Distinguished by `kind`, `members`,
//! and `terminals`.  All cross-pool wiring lives on the neuron via
//! `terminals: Vec<Terminal>` whose target is a [`NeuronRef`] that can point
//! into any pool.  No fabric-level routing table exists.

use serde::{Deserialize, Serialize};

/// Stable identifier for a pool within a brain.  Strings are concise and
/// readable; the per-brain pool count is small (single digits to low
/// thousands at most) so this is cheap.
pub type PoolId = u32;

/// Stable identifier for a neuron within its pool.  Reused slots get the
/// same id; eviction to cold tier preserves the id.
pub type NeuronId = u32;

/// Cross-pool-capable neuron reference.  Used in [`Neuron::members`] and
/// [`Terminal::target`] so binding concepts and cross-pool axons are
/// expressed by the same primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuronRef {
    pub pool: PoolId,
    pub neuron: NeuronId,
}

impl NeuronRef {
    #[inline]
    pub const fn new(pool: PoolId, neuron: NeuronId) -> Self {
        Self { pool, neuron }
    }
}

/// Dale's principle (spec §1.2): outgoing terminals must match the neuron's
/// kind.  Mixing excitatory and inhibitory outputs on the same neuron is
/// biologically invalid and forbidden at the substrate level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronKind {
    Excitatory,
    Inhibitory,
    /// Reserved for neuromodulator-projecting neurons.  Modulates plasticity
    /// at target neurons; not used for direct activation propagation.
    Modulatory,
}

/// One axon terminal.  Target can live in any pool — the same struct
/// expresses within-pool synapses and cross-pool axons.
///
/// `consolidation` counts how many distinct training events (in different
/// ticks) reinforced this terminal.  Used by spec §4.A's "myelination via
/// consolidation gain" — well-used terminals propagate with higher effective
/// weight.  Plain `weight` saturates against `max_weight` so a single
/// hyper-active session can't dominate the substrate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Terminal {
    pub target:          NeuronRef,
    pub weight:          f32,
    pub consolidation:   u8,
    pub last_fired_tick: u64,
}

impl Terminal {
    #[inline]
    pub fn new(target: NeuronRef, weight: f32, tick: u64) -> Self {
        Self { target, weight, consolidation: 0, last_fired_tick: tick }
    }

    /// Effective contribution = weight scaled by consolidation level.
    /// Consolidation gain matches spec §4.A: well-used edges propagate
    /// proportionally stronger without exceeding biological-style bounds.
    #[inline]
    pub fn effective_weight(&self) -> f32 {
        // Consolidation contributes a sub-linear boost capped at +50%.
        let cons_gain = 1.0 + (self.consolidation as f32 * 0.05).min(0.5);
        self.weight * cons_gain
    }
}

/// The unified neuron.  Atoms have empty `members`; concepts have one or
/// more members (possibly cross-pool — that's a binding concept).
///
/// `prediction_error_ema` enables per-region adaptive plasticity per spec
/// §2.5: a neuron whose recent firings have been poorly predicted gets
/// higher local plasticity, while a stable neuron's plasticity is low.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id:                   NeuronId,
    pub label:                String,
    pub kind:                 NeuronKind,
    /// Empty for atoms.  Non-empty for concepts.  Order matters: it's the
    /// firing-order of the concept's constituent atoms/concepts as
    /// observed at promotion time.  Spec §1.1 — position is encoded by
    /// order in this Vec, NEVER by mangling atom identity.
    pub members:              Vec<NeuronRef>,
    pub terminals:            Vec<Terminal>,
    pub born_tick:            u64,
    pub last_fired_tick:      u64,
    pub use_count:            u64,
    pub prediction_error_ema: f32,
}

impl Neuron {
    pub fn new_atom(id: NeuronId, label: String, kind: NeuronKind, tick: u64) -> Self {
        Self {
            id,
            label,
            kind,
            members: Vec::new(),
            terminals: Vec::new(),
            born_tick: tick,
            last_fired_tick: tick,
            use_count: 0,
            prediction_error_ema: 0.0,
        }
    }

    pub fn new_concept(
        id:      NeuronId,
        label:   String,
        kind:    NeuronKind,
        members: Vec<NeuronRef>,
        tick:    u64,
    ) -> Self {
        Self {
            id,
            label,
            kind,
            members,
            terminals: Vec::new(),
            born_tick: tick,
            last_fired_tick: tick,
            use_count: 0,
            prediction_error_ema: 0.0,
        }
    }

    /// True for atoms (no members), false for concepts of any kind.
    #[inline]
    pub fn is_atom(&self) -> bool {
        self.members.is_empty()
    }

    /// True for binding concepts (members span more than one pool).  This
    /// distinguishes within-pool hierarchical concepts from cross-pool
    /// bindings using the same struct.
    pub fn is_binding(&self, self_pool: PoolId) -> bool {
        self.members.iter().any(|m| m.pool != self_pool)
    }

    /// Insert or strengthen a terminal toward `target`.  Idempotent on
    /// repeated calls: the weight saturates against `max_weight` and the
    /// consolidation counter increments once per distinct tick.
    pub fn reinforce_terminal(
        &mut self,
        target:     NeuronRef,
        delta:      f32,
        tick:       u64,
        max_weight: f32,
    ) {
        // Linear search is fine for typical fan-outs (terminal counts are
        // bounded by survival pruning; a healthy neuron carries dozens to
        // low hundreds).  When this becomes a bottleneck, swap for a
        // sorted-Vec binary search keyed by target.
        if let Some(t) = self.terminals.iter_mut().find(|t| t.target == target) {
            t.weight = (t.weight + delta).min(max_weight);
            if t.last_fired_tick != tick {
                t.consolidation = t.consolidation.saturating_add(1);
                t.last_fired_tick = tick;
            }
        } else {
            self.terminals.push(Terminal::new(target, delta.min(max_weight), tick));
        }
    }

    /// Decay every terminal toward zero by `(1 - epsilon)` and remove any
    /// that fall below `floor`.  Called once per tick on every neuron by
    /// the pool's housekeeping pass.  Spec §1.5: this IS the only
    /// forgetting mechanism.
    pub fn decay_and_prune(&mut self, epsilon: f32, floor: f32) -> usize {
        let mut pruned = 0;
        self.terminals.retain_mut(|t| {
            t.weight *= 1.0 - epsilon;
            if t.weight < floor {
                pruned += 1;
                false
            } else {
                true
            }
        });
        pruned
    }
}
