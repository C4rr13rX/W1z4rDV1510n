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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
    /// Runtime identity bit that lets a serialized concept release its member
    /// payload while asleep without masquerading as an atom. It is skipped in
    /// legacy bincode snapshots: loaded concepts still identify themselves by
    /// their nonempty members, and constructors set it for new concepts.
    #[serde(skip)]
    concept_identity:         bool,
    /// Empty for atoms.  Non-empty for concepts.  Order matters: it's the
    /// firing-order of the concept's constituent atoms/concepts as
    /// observed at promotion time.  Spec §1.1 — position is encoded by
    /// order in this Vec, NEVER by mangling atom identity.
    pub members:              Vec<NeuronRef>,
    pub terminals:            Vec<Terminal>,
    /// Address-by-name index over `terminals`: `target → position in
    /// the Vec`.  A cache, never canonical state (#[serde(skip)]) — the
    /// same question is answered by scanning `terminals`, so the map is
    /// kept ONLY on neurons whose fan-out is at least
    /// [`Neuron::TERMINAL_INDEX_THRESHOLD`].  Below it the Vec is the
    /// index and this map is empty and unallocated.
    ///
    /// Why: measured at scale 64 of the scorecard, 9,392 of 9,776
    /// neurons hold NO terminals and 48 hold ≥128 (24,576 of the 24,975
    /// terminals in the fabric), yet the map's capacity was 297,178
    /// entries — 5.05 MB — because `clear()` retains capacity, so every
    /// neuron pruned back to empty kept ~28 buckets forever.
    ///
    /// INVARIANT: either `terminal_idx.len() == terminals.len()` and for
    /// every i, `terminal_idx[terminals[i].target] == i`; or the map is
    /// absent and lookups scan.  [`Neuron::find_terminal`] reads the
    /// discriminator off those two lengths, so a prune that forgets to
    /// rebuild degrades to a scan rather than to a wrong answer.
    ///
    /// BOXED, and that is the point. An inline `AHashMap` is a hasher plus a
    /// `RawTable` — measured 64 bytes, 29.6 % of a 216-byte `Neuron` and its
    /// largest single field — and every neuron paid it whether or not a map
    /// was ever allocated. Keeping the BUCKETS off non-hubs (the threshold
    /// above) did nothing about the header. `None` is 8 bytes, so the same
    /// minority that allocates a map is now the only minority that carries
    /// one. Private so the absent case cannot be spelled two ways: go
    /// through [`Neuron::find_terminal`] and
    /// [`Neuron::release_terminal_index`].
    #[serde(skip)]
    terminal_idx:             Option<Box<ahash::AHashMap<NeuronRef, usize>>>,
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
    /// reclaims them).
    ///
    /// `#[serde(skip)]` because the snapshot format is bincode, which
    /// is positional — adding a serialized field would break existing
    /// brain.bin files.  On load this field defaults to 0; the next
    /// access applies the entire elapsed-since-tick-0 backlog at once.
    /// For freshly-loaded brains the fabric's current tick may be
    /// arbitrary, so we initialise lazy-decay state on the first
    /// access path rather than at restore time.
    #[serde(skip)]
    pub last_decayed_tick:    u64,

    /// Domain tag — the cluster of co-firing concepts this neuron
    /// belongs to.  Default 0 means "unassigned / global domain"
    /// (current behaviour: all neurons share one global domain).
    ///
    /// Assigned by the integration cycle (Brain::integrate, future)
    /// via co-firing-graph community detection.  Concepts in the same
    /// domain co-fire together strongly; concepts across domains
    /// have weak direct co-firing but may share `bridges`.
    ///
    /// Within-domain wiring runs on the hot observe path (fast common
    /// case).  Cross-domain wiring is deferred to the integration
    /// cycle and surfaces as `bridges`, not regular `terminals`.
    /// Together they preserve cross-domain reasoning ("X is like Y")
    /// without paying the O(N²) cost on every tick.
    #[serde(skip)]
    pub domain_id:            u32,

    /// Cross-domain bridges — analogical connections to concepts in
    /// other domains.  Maintained by the integration cycle, NOT by
    /// the observe hot path.  Bridges decay slower than terminals
    /// (configurable; default ~10x slower) so structural analogies
    /// learned during integration stay stable across normal training.
    ///
    /// `#[serde(skip)]` for the same reason as `last_decayed_tick`:
    /// bincode is positional and adding serialized fields breaks
    /// existing brain.bin files.  Bridges rebuild from co-firing
    /// patterns during the next integration cycle.
    #[serde(skip)]
    pub bridges:              Vec<Terminal>,
}

impl Neuron {
    pub fn new_atom(id: NeuronId, label: String, kind: NeuronKind, tick: u64) -> Self {
        Self {
            id,
            label,
            kind,
            concept_identity: false,
            members: Vec::new(),
            terminals: Vec::new(),
            born_tick: tick,
            last_fired_tick: tick,
            use_count: 0,
            prediction_error_ema: 0.0,
            salience: 0.0,
            salience_ema: 0.0,
            last_decayed_tick: tick,
            terminal_idx: None,
            domain_id: 0,
            bridges: Vec::new(),
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
            concept_identity: true,
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
            terminal_idx: None,
            domain_id: 0,
            bridges: Vec::new(),
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
        !self.concept_identity && self.members.is_empty()
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
        // Address-by-name semantics: the caller already knows which
        // target it wants.  `terminal_idx.get(&target)` answers "does
        // the dendrite at `target` exist?" in O(1).  Two paths:
        //
        //   Some(idx) → the connection exists; strengthen it in place.
        //   None      → no connection yet; neurogenesis: append a new
        //               Terminal and register its index.
        //
        // INVARIANT preserved: index entry stays consistent with the
        // Vec because we only ever `push` (which is O(1) and doesn't
        // shift existing positions).  Any caller that prunes terminals
        // MUST call rebuild_terminal_idx — handled by decay_and_prune
        // and apply_pending_decay below.
        if let Some(idx) = self.find_terminal(&target) {
            let t = &mut self.terminals[idx];
            t.weight = (t.weight + delta).min(max_weight);
            if t.last_fired_tick != tick {
                t.consolidation = t.consolidation.saturating_add(1);
                t.last_fired_tick = tick;
            }
            false
        } else {
            let idx = self.terminals.len();
            self.terminals.push(Terminal::new(target, delta.min(max_weight), tick));
            if self.terminals.len() >= Self::TERMINAL_INDEX_THRESHOLD {
                if self.terminal_index_len() == idx {
                    // The index already covers 0..idx — extend it.
                    if let Some(m) = self.terminal_idx.as_mut() {
                        m.insert(target, idx);
                    } else {
                        self.rebuild_terminal_idx();
                    }
                } else {
                    // This push crossed the threshold (or the index was
                    // released by a prune): build it from the Vec.
                    self.rebuild_terminal_idx();
                }
            }
            true
        }
    }

    /// Fan-out at or above which a neuron pays for a hash index over its
    /// terminals.  Below it a linear scan over `len` 24-byte terminals is
    /// both correct and cheaper than the map that would answer it: at
    /// scale 64 only 48 of 9,776 neurons reach this threshold, and they
    /// hold 24,576 of the 24,975 terminals.
    ///
    /// 64 is the smallest power of two above the measured non-hub maximum
    /// (7 terminals), so no ordinary neuron allocates a map at all.
    pub const TERMINAL_INDEX_THRESHOLD: usize = 64;

    /// Position in `terminals` of the terminal aimed at `target`.
    ///
    /// Uses `terminal_idx` when it covers every terminal, otherwise
    /// scans.  The discriminator is the two lengths, so the answer is
    /// correct whether or not a caller rebuilt the index — a missing
    /// rebuild costs a scan, never a wrong position.
    #[inline]
    pub fn find_terminal(&self, target: &NeuronRef) -> Option<usize> {
        if let Some(m) = self.terminal_idx.as_deref() {
            if !self.terminals.is_empty() && m.len() == self.terminals.len() {
                return m.get(target).copied();
            }
        }
        self.terminals.iter().position(|t| &t.target == target)
    }

    /// Entries in the terminal index, 0 when no index is allocated.
    #[inline]
    pub fn terminal_index_len(&self) -> usize {
        self.terminal_idx.as_deref().map_or(0, |m| m.len())
    }

    /// Usable slots the terminal index has allocated, 0 when there is none.
    /// The census charges this through `brain::hash_table_bytes`, which turns
    /// it into the bucket count actually allocated.
    #[inline]
    pub fn terminal_index_capacity(&self) -> usize {
        self.terminal_idx.as_deref().map_or(0, |m| m.capacity())
    }

    /// Position of `target` according to the index alone, without the scan
    /// fallback. For tests and the store round-trip, which must distinguish
    /// "the index answered" from "a scan did".
    #[inline]
    pub fn terminal_index_get(&self, target: &NeuronRef) -> Option<usize> {
        self.terminal_idx.as_deref().and_then(|m| m.get(target).copied())
    }

    /// Drop the index and its buckets outright. `clear()` would keep every
    /// bucket, which is the allocation this field exists to avoid paying for
    /// on a neuron that no longer needs one.
    #[inline]
    pub fn release_terminal_index(&mut self) {
        self.terminal_idx = None;
    }

    /// Number of terminals whose consolidation has reached the lock
    /// threshold.  These are decay-exempt and form the 100%-recall floor.
    pub fn locked_terminal_count(&self) -> usize {
        self.terminals.iter().filter(|t| t.consolidation >= Self::CONSOLIDATION_LOCK).count()
    }

    /// Rebuild the `terminal_idx` cache from `terminals`.  O(|terminals|).
    /// Call after any operation that removes entries from `terminals`
    /// (retain, drain, clear, pop) — append-only operations preserve
    /// the invariant and do NOT need a rebuild.
    ///
    /// This is also where both of a neuron's terminal allocations are
    /// RELEASED.  `Vec::retain` and `HashMap::clear` keep capacity, so a
    /// neuron grown to 24 terminals and pruned back to 0 held 24 terminal
    /// slots and ~28 hash buckets for the rest of its life.  Measured at
    /// scale 64: 254,736 Vec slots for 24,975 live terminals and 297,178
    /// map entries, 11.2 MB between them on a 15.3 MB body total.
    pub fn rebuild_terminal_idx(&mut self) {
        if self.terminals.len() < Self::TERMINAL_INDEX_THRESHOLD {
            // Below the threshold the Vec is the index. Drop the map
            // outright — `clear()` alone would keep every bucket, and now
            // the 64-byte header goes with it.
            self.terminal_idx = None;
        } else {
            let mut m = ahash::AHashMap::with_capacity(self.terminals.len());
            for (i, t) in self.terminals.iter().enumerate() {
                m.insert(t.target, i);
            }
            self.terminal_idx = Some(Box::new(m));
        }
        // Return the spare half of a pruned terminal Vec, but only when the
        // block is worth returning. Releasing the last 4 slots as well was
        // MEASURED WORSE: it cut accounted bytes by 0.36 MB and raised the
        // scale-64 peak 37.0 -> 38.0 MB, because 9,392 frees of 96 bytes
        // fragment the arena instead of returning pages to the OS. Accounted
        // bytes are not resident bytes; the peak is the number that is gated.
        if self.terminals.capacity() > 4 && self.terminals.len() * 2 <= self.terminals.capacity() {
            self.terminals.shrink_to_fit();
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
    /// Consolidation-lock threshold: terminals whose `consolidation` count
    /// has reached this value or higher are considered "trained" and are
    /// exempt from background decay and prune.  This is the 100%-recall
    /// floor — once a terminal has been reinforced this many times on
    /// distinct ticks, the underlying QA pair is permanent.
    ///
    /// Chosen as `3` so a single training pass (one tick) does NOT lock;
    /// a probe-correct/probe-correct/probe-correct triple does.  Phase B
    /// will make this self-tune from QA recall stats.
    pub const CONSOLIDATION_LOCK: u8 = 3;

    pub fn apply_pending_decay(&mut self, now: u64, epsilon: f32, floor: f32) -> usize {
        // Snapshot-loaded neurons have last_decayed_tick == 0 because
        // the field is #[serde(skip)] (the snapshot format is positional
        // bincode and a serialized field would break older brain.bin
        // files).  Bootstrap to `now` without applying any backlog —
        // the brain's pre-restore decay is already baked into the
        // terminal weights as they were when serialised.  Trying to
        // apply (1-ε)^(now) on top would multiply weights by ~1e-16
        // at typical post-restore tick counts and instantly nuke
        // every terminal in the fabric.
        if self.last_decayed_tick == 0 {
            self.last_decayed_tick = now;
            return 0;
        }
        let elapsed = now.saturating_sub(self.last_decayed_tick);
        if elapsed == 0 || epsilon <= 0.0 {
            self.last_decayed_tick = now;
            return 0;
        }
        let factor = (1.0 - epsilon).powi(elapsed.min(i32::MAX as u64) as i32);
        let mut pruned = 0;
        self.terminals.retain_mut(|t| {
            if t.consolidation >= Self::CONSOLIDATION_LOCK {
                // Locked: well-trained terminal, exempt from decay/prune.
                // This is the 100%-recall anchor.
                return true;
            }
            t.weight *= factor;
            if t.weight < floor {
                pruned += 1;
                false
            } else {
                true
            }
        });
        if pruned > 0 {
            self.rebuild_terminal_idx();
        }
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
            if t.consolidation >= Self::CONSOLIDATION_LOCK {
                return true;
            }
            t.weight *= 1.0 - epsilon;
            if t.weight < floor {
                pruned += 1;
                false
            } else {
                true
            }
        });
        if pruned > 0 {
            self.rebuild_terminal_idx();
        }
        pruned
    }
}

#[cfg(test)]
mod terminal_index_tests {
    use super::*;

    fn atom() -> Neuron {
        Neuron::new_atom(0, "atom".into(), NeuronKind::Excitatory, 1)
    }

    fn wire(n: &mut Neuron, count: u32) {
        for i in 0..count {
            n.reinforce_terminal(NeuronRef::new(1, i), 0.5, 1, 1.0);
        }
    }

    /// What a neuron costs before it holds anything, field by field.
    ///
    /// `neuron_body_bytes.struct` is the second-largest line in the scale-64
    /// census (2.01 MB over 9,776 neurons) and it is pure `size_of`, so it
    /// falls only if the struct shrinks. Printing the components is how the
    /// next change gets chosen on a number rather than on a guess: an inline
    /// `AHashMap` is a hasher plus a `RawTable`, paid by EVERY neuron whether
    /// or not the map is ever allocated, while only neurons at or above
    /// `TERMINAL_INDEX_THRESHOLD` populate one.
    #[test]
    fn the_struct_is_priced_field_by_field() {
        use std::mem::size_of;
        let total = size_of::<Neuron>();
        let header = size_of::<ahash::AHashMap<NeuronRef, usize>>();
        let field = size_of::<Option<Box<ahash::AHashMap<NeuronRef, usize>>>>();
        println!("size_of::<Neuron>() = {total}");
        for (name, n) in [
            ("label: String", size_of::<String>()),
            ("members: Vec", size_of::<Vec<NeuronRef>>()),
            ("terminals: Vec", size_of::<Vec<Terminal>>()),
            ("bridges: Vec", size_of::<Vec<Terminal>>()),
            ("terminal_idx: Option<Box<AHashMap>>", field),
            ("  (the AHashMap it points at)", header),
            ("id: NeuronId", size_of::<NeuronId>()),
            ("3 x u64 ticks/count", 3 * size_of::<u64>()),
        ] {
            println!("  {n:4} B  {:5.1}%  {name}", 100.0 * n as f64 / total as f64);
        }
        // Not an assertion about the exact layout, which the compiler owns.
        // Two facts the layout must keep: the index costs a pointer inline,
        // and a null-pointer-optimised `Option<Box<_>>` adds nothing for the
        // absent case, which is 9,728 of 9,776 neurons at scale 64.
        assert_eq!(
            field,
            size_of::<usize>(),
            "the terminal index field is {field} B, so it is no longer just a pointer"
        );
        assert!(
            header >= 4 * field,
            "the AHashMap header is {header} B against a {field} B field -- if the map \
             ever becomes pointer-sized, inlining it is cheaper than boxing it"
        );
    }

    /// The majority case: a low-fan-out neuron allocates NO index at all,
    /// and every terminal is still addressable by name. This is the 5.05 MB
    /// the map cost at scale 64, where 9,392 of 9,776 neurons hold nothing.
    #[test]
    fn a_neuron_below_the_threshold_carries_no_index() {
        let mut n = atom();
        wire(&mut n, Neuron::TERMINAL_INDEX_THRESHOLD as u32 - 1);
        assert_eq!(n.terminal_index_capacity(), 0, "no map below the threshold");
        for i in 0..n.terminals.len() as u32 {
            assert_eq!(n.find_terminal(&NeuronRef::new(1, i)), Some(i as usize));
        }
        assert_eq!(n.find_terminal(&NeuronRef::new(1, 9999)), None);
        // And a repeat reinforcement must strengthen in place, not append.
        let before = n.terminals.len();
        assert!(!n.reinforce_terminal(NeuronRef::new(1, 3), 0.5, 2, 10.0));
        assert_eq!(n.terminals.len(), before);
    }

    /// Crossing the threshold builds the index, and it must agree with a
    /// scan at EVERY position — an off-by-one at the crossing would return
    /// the wrong terminal, which a lookup-miss test cannot see.
    #[test]
    fn crossing_the_threshold_builds_an_index_that_agrees_with_a_scan() {
        let mut n = atom();
        wire(&mut n, Neuron::TERMINAL_INDEX_THRESHOLD as u32 + 40);
        assert_eq!(n.terminal_index_len(), n.terminals.len(), "index covers every terminal");
        for (i, t) in n.terminals.iter().enumerate() {
            assert_eq!(n.find_terminal(&t.target), Some(i));
            assert_eq!(n.terminal_index_get(&t.target), Some(i));
        }
    }

    /// A prune back to empty must RELEASE both allocations. `Vec::retain`
    /// and `HashMap::clear` keep capacity, which is how a neuron grown to
    /// 24 terminals and pruned to 0 kept its bytes for life.
    #[test]
    fn a_prune_to_empty_releases_both_allocations() {
        let mut n = atom();
        wire(&mut n, Neuron::TERMINAL_INDEX_THRESHOLD as u32 + 40);
        assert!(n.terminal_index_capacity() > 0 && n.terminals.capacity() > 0);
        // Nothing is consolidation-locked after one tick, so a decay past
        // the floor prunes every terminal.
        let wired = n.terminals.len();
        let pruned = n.apply_pending_decay(500, 0.5, 0.01);
        assert_eq!(pruned, wired, "every one of the {wired} terminals pruned");
        assert!(n.terminals.is_empty(), "terminals pruned to empty");
        assert_eq!(n.terminal_index_capacity(), 0, "index allocation released");
        assert!(
            n.terminals.capacity() <= 4,
            "terminal Vec released, held {} slots",
            n.terminals.capacity(),
        );
    }
}
