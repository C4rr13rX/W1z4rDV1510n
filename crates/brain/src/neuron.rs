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
///
/// `salience` is the brain-emitted retention signal per [`ARCHITECTURE.md`]
/// §17.5.  The substrate's *own* training rule writes to it; the storage
/// tier's eviction policy reads from it.  Range \[0.0, 1.0\] where 1.0
/// means "must retain" and 0.0 means "forgettable."  Salience is updated
/// by two coupled signals (cf. Schultz 2007 dopaminergic gating, Frémaux
/// & Gerstner 2016 three-factor plasticity, McClelland et al. 1995 CLS,
/// Tononi & Cirelli 2014 SHY):
/// 1. Reward / precision-modulated co-firing — neurons that participate
///    in a successful decode get a salience boost in lockstep with their
///    terminal-weight strengthening.
/// 2. Compression utility under replay (deferred to Stage 17.7) — a
///    neuron whose absence would substantially raise free-energy on
///    replayed moments has its salience boosted.
///
/// `salience_ema` is the smoothed long-horizon trace used by the
/// eviction policy in Stage 17.4 full.  EMA (vs. raw salience) prevents
/// transient high-confidence decodes from gaming the retention policy.
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
    /// Stage 17.5: brain-emitted retention signal.  Defaults to 0.0 on
    /// creation; updated by reward-modulated training paths (see
    /// `Neuron::bump_salience`).  `#[serde(default)]` so older
    /// brain.bin files without this field deserialize cleanly.
    #[serde(default)]
    pub salience:             f32,
    /// Long-horizon EMA of `salience`.  Used by eviction policy.
    #[serde(default)]
    pub salience_ema:         f32,
    /// Tick at which this neuron's terminals were last decayed.
    /// Enables lazy decay — on access, we compute (1 - ε)^(now -
    /// last_decayed_tick) and apply it in one shot rather than walking
    /// every terminal on every tick.  Mathematically identical to the
    /// eager per-tick sweep, modulo terminals on neurons that never
    /// get accessed (those leak until a sleep-cycle full sweep
    /// reclaims them).  `#[serde(default)]` so older brain.bin files
    /// without this field load cleanly — they'll get `last_decayed_tick
    /// = 0`, which means the next access will apply the entire elapsed
    /// decay backlog (correct, just expensive that one time).
    #[serde(default)]
    pub last_decayed_tick:    u64,
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
            salience: 0.0,
            salience_ema: 0.0,
            last_decayed_tick: tick,
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
            // Concepts start with a tiny baseline salience so they're not
            // evicted before they accumulate any reward signal at all.
            // 0.01 keeps them above noise but well below atoms in active use.
            salience: 0.01,
            salience_ema: 0.01,
            last_decayed_tick: tick,
        }
    }

    /// Stage 17.5: bump this neuron's salience by `delta`, then update its
    /// EMA.  Called by the brain when this neuron participates in a
    /// successful decode (reward signal).  `delta` is typically the
    /// decode's `atom_score` — a high-precision recall maps to a large
    /// bump; a marginal recall to a small one.  Saturates at 1.0.
    ///
    /// Frémaux & Gerstner (2016) — three-factor (pre × post × reward)
    /// plasticity governs which synapses + which post-synaptic neurons
    /// get long-term tags.  We track salience on the neuron (post),
    /// while terminal weight is updated separately on the synapse.
    pub fn bump_salience(&mut self, delta: f32) {
        const ALPHA: f32 = 0.15;  // EMA: ~recent-10 weighting
        self.salience = (self.salience + delta.max(0.0)).min(1.0);
        self.salience_ema =
            self.salience_ema * (1.0 - ALPHA) + self.salience * ALPHA;
    }

    /// Stage 17.5: decay this neuron's salience by `gamma` (multiplicative).
    /// Called during sleep on neurons that did NOT participate in the
    /// recent replay — captures the "use it or lose it" half of the
    /// systems-consolidation hypothesis (Frankland & Bontempi 2005).
    /// Floor at 0 to avoid negative salience.
    pub fn decay_salience(&mut self, gamma: f32) {
        let g = gamma.clamp(0.0, 1.0);
        self.salience = (self.salience * (1.0 - g)).max(0.0);
        // EMA naturally follows the per-bump update, no extra step here.
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
    /// Reinforce a terminal toward `target`.  Returns `true` iff this
    /// call added a brand-new terminal (so callers that hold the owning
    /// Pool can keep an O(1) `total_terminals` counter accurate without
    /// re-scanning every neuron).  Returns `false` when an existing
    /// terminal was strengthened in place.
    pub fn reinforce_terminal(
        &mut self,
        target:     NeuronRef,
        delta:      f32,
        tick:       u64,
        max_weight: f32,
    ) -> bool {
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
            false
        } else {
            self.terminals.push(Terminal::new(target, delta.min(max_weight), tick));
            true
        }
    }

    /// Lazy decay: apply (1 - epsilon)^(now - last_decayed_tick) to every
    /// terminal in one shot, prune any that fell below `floor`, then
    /// update `last_decayed_tick = now`.  Mathematically identical to the
    /// eager per-tick sweep — `weight × (1-ε) × (1-ε) × … (k times)` ≡
    /// `weight × (1-ε)^k` — so the brain's recall behaviour is unchanged.
    ///
    /// Returns the number of terminals pruned.  Caller must subtract
    /// that count from the owning Pool's `total_terminals` to keep the
    /// O(1) counter consistent.
    ///
    /// Cheap when `now == last_decayed_tick` (no elapsed → fast path:
    /// no walk).  At elapsed=k, cost is O(|terminals|) — same as the
    /// eager sweep but only when the neuron is actually accessed.
    pub fn apply_pending_decay(&mut self, now: u64, epsilon: f32, floor: f32) -> usize {
        let elapsed = now.saturating_sub(self.last_decayed_tick);
        if elapsed == 0 || epsilon <= 0.0 {
            self.last_decayed_tick = now;
            return 0;
        }
        let factor = (1.0 - epsilon).powi(elapsed.min(i32::MAX as u64) as i32);
        let mut pruned = 0;
        self.terminals.retain_mut(|t| {
            t.weight *= factor;
            if t.weight < floor {
                pruned += 1;
                false
            } else {
                true
            }
        });
        self.last_decayed_tick = now;
        pruned
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
