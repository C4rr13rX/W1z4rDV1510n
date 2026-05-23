//! Fabric per [`ARCHITECTURE.md`] §1.3 and §4.A.
//!
//! Holds the pools, the moment buffer, and the propagation algorithm.
//! Cross-pool wiring is NOT stored in the fabric — it lives on each
//! neuron's `terminals` vec.  The fabric's responsibility is the
//! tick-aligned co-temporal wiring that creates those terminals when
//! neurons in different pools fire within the same temporal window.
//!
//! Propagation is one uniform loop: for each firing neuron, walk its
//! terminals, deposit activation at the targets (which may be in any
//! pool).  Pool boundaries are organizational, not structural.

use ahash::AHashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::neuron::{NeuronId, NeuronRef, PoolId};
use crate::pool::Pool;
use crate::store::{NoopStore, Store, WalEvent};

/// Per-phase nanosecond counters for advance_tick.  Cumulative since
/// fabric construction; the brain server exposes them via /tick_profile
/// so we can find the dominant cost without spinning up a Rust
/// profiler.  Atomics so reads from the HTTP layer don't need the
/// brain mutex.
#[derive(Debug, Default)]
pub struct TickProfile {
    pub ticks:                          AtomicU64,
    pub cross_pool_atom_wiring_ns:      AtomicU64,
    pub cross_pool_concept_wiring_ns:   AtomicU64,
    pub within_pool_temporal_ns:        AtomicU64,
    pub housekeeping_ns:                AtomicU64,
    pub total_ns:                       AtomicU64,
}

impl TickProfile {
    pub fn snapshot(&self) -> TickProfileSnapshot {
        TickProfileSnapshot {
            ticks:                        self.ticks.load(Ordering::Relaxed),
            cross_pool_atom_wiring_ns:    self.cross_pool_atom_wiring_ns.load(Ordering::Relaxed),
            cross_pool_concept_wiring_ns: self.cross_pool_concept_wiring_ns.load(Ordering::Relaxed),
            within_pool_temporal_ns:      self.within_pool_temporal_ns.load(Ordering::Relaxed),
            housekeeping_ns:              self.housekeeping_ns.load(Ordering::Relaxed),
            total_ns:                     self.total_ns.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TickProfileSnapshot {
    pub ticks:                        u64,
    pub cross_pool_atom_wiring_ns:    u64,
    pub cross_pool_concept_wiring_ns: u64,
    pub within_pool_temporal_ns:      u64,
    pub housekeeping_ns:              u64,
    pub total_ns:                     u64,
}

/// What fired in every pool at a given tick.  Used by cross-pool wiring:
/// pairs of neurons firing within the same tick get an axon terminal
/// grown between them, Hebbian-strengthened on repeat co-firing.
#[derive(Debug, Clone)]
pub struct Moment {
    pub tick:      u64,
    /// pool_id → set of neuron_ids that fired this tick.
    pub fired:     AHashMap<PoolId, Vec<NeuronId>>,
}

impl Moment {
    fn new(tick: u64) -> Self {
        Self { tick, fired: AHashMap::new() }
    }
}

/// Fabric-level config.  Pool-specific config lives on each Pool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FabricConfig {
    /// Reinforcement applied to a cross-pool terminal each time both
    /// endpoints fire in the same tick.  Smaller than within-pool atom→
    /// concept reinforcement because cross-pool bonds should build up
    /// only with sustained co-occurrence.
    pub cross_pool_lr: f32,
    /// Maximum hops in propagation.  One hop = atoms fire concepts.
    /// Two hops = concepts fire concepts (or cross-pool targets).  Spec
    /// §4.A: propagation is one uniform loop with hop-decay.
    pub max_hops:      usize,
    /// Per-hop decay multiplier — activation × decay each hop.  Mirrors
    /// biological signal attenuation along long axons.
    pub hop_decay:     f32,
    /// Reinforcement applied to CROSS-POOL terminals between concept
    /// neurons (not atoms) that co-fire at the same tick.  This is
    /// the position-aware binding mechanism: a concept encodes its
    /// member sequence's position via the order of its `members` vec;
    /// concept→concept cross-pool terminals therefore bind specific
    /// positional sequences across pools.  Without this, only
    /// atom-level wiring runs, which loses positional discrimination
    /// because firing-sets are deduplicated within a tick.
    pub cross_pool_concept_lr: f32,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            cross_pool_lr:         0.15,
            max_hops:              2,
            hop_decay:             0.85,
            cross_pool_concept_lr: 0.20,
        }
    }
}

pub struct Fabric {
    pub config: FabricConfig,
    pools:      AHashMap<PoolId, Arc<RwLock<Pool>>>,
    current:    Moment,
    tick:       u64,
    /// Per-pool: which CONCEPT neurons fired during the most-recently
    /// closed tick.  Stage 2b uses this to grow concept→concept
    /// terminals when concept A fires then concept B fires next tick
    /// in the same pool — the within-pool temporal "what follows
    /// what" wiring.  Transient state; not persisted in snapshots.
    prev_tick_concepts: AHashMap<PoolId, Vec<NeuronId>>,
    /// Persistence backend per [`ARCHITECTURE.md`] §17.9.  Cloned into
    /// each pool on registration via [`Fabric::set_store`] so neurogenesis
    /// events flow to the same WAL.  Default is [`NoopStore`].
    store: Arc<dyn Store>,
    /// Per-phase advance_tick timing.  Atomic so /tick_profile reads
    /// don't need the brain mutex.  Initialised to zero.
    pub profile: Arc<TickProfile>,
}

impl Fabric {
    pub fn new(config: FabricConfig) -> Self {
        Self {
            config,
            pools:              AHashMap::new(),
            current:            Moment::new(0),
            tick:               0,
            prev_tick_concepts: AHashMap::new(),
            store:              Arc::new(NoopStore),
            profile:            Arc::new(TickProfile::default()),
        }
    }

    /// Snapshot of the per-phase tick timing.  Cumulative counters; for
    /// per-tick mean divide by snapshot.ticks.
    pub fn tick_profile(&self) -> TickProfileSnapshot {
        self.profile.snapshot()
    }

    /// Attach a persistence backend per [`ARCHITECTURE.md`] §17.9.  Fans
    /// out to every already-registered pool; pools registered after this
    /// call inherit the same backend.  Idempotent — calling with the
    /// same `Arc<dyn Store>` is a no-op apart from the fan-out re-set.
    pub fn set_store(&mut self, store: Arc<dyn Store>) {
        self.store = store.clone();
        for (_pid, pool) in self.pools.iter() {
            pool.write().set_store(store.clone());
        }
    }

    /// Clone the persistence handle.  Used by [`crate::Brain::checkpoint`]
    /// to flush the WAL and emit a snapshot marker.
    pub fn store_clone(&self) -> Arc<dyn Store> { self.store.clone() }

    pub fn current_tick(&self) -> u64 { self.tick }

    /// Stage 17.9 — explicit tick set, used only by WAL recovery.
    /// Normal training advances the tick via [`Fabric::advance_tick`],
    /// which also runs housekeeping; this method does NOT run
    /// housekeeping and is intended purely to fast-forward the fabric
    /// past replayed events.  Refuses to set the tick backwards.
    pub fn set_tick(&mut self, new_tick: u64) {
        if new_tick > self.tick {
            self.tick = new_tick;
            self.current = Moment::new(new_tick);
        }
    }

    /// Snapshot the fabric (configs + all pool states + tick).  The
    /// current in-flight moment is intentionally not captured — see
    /// [`crate::persistence`] for the rationale.
    pub fn snapshot(&self) -> crate::persistence::FabricSnapshot {
        let pool_order: Vec<PoolId> = self.pool_ids();
        let mut pools = std::collections::HashMap::new();
        for pid in &pool_order {
            if let Some(p) = self.pools.get(pid) {
                pools.insert(*pid, p.read().snapshot());
            }
        }
        crate::persistence::FabricSnapshot {
            config: self.config.clone(),
            tick:   self.tick,
            pool_order,
            pools,
        }
    }

    /// Rebuild a fabric from a snapshot.  `encodings` keys must
    /// match every pool id in the snapshot — missing encodings cause
    /// the pool to be skipped (logged via the returned Vec).
    pub fn from_snapshot(
        snap:      crate::persistence::FabricSnapshot,
        mut encodings: std::collections::HashMap<PoolId, Box<dyn crate::pool::AtomEncoding>>,
    ) -> (Self, Vec<PoolId>) {
        let mut fabric = Self {
            config:             snap.config,
            pools:              AHashMap::new(),
            current:            Moment::new(snap.tick),
            tick:               snap.tick,
            prev_tick_concepts: AHashMap::new(),
            // Snapshot-restored fabric defaults to NoopStore; caller plugs
            // in the live backend via set_store after restore so subsequent
            // observations get logged.
            store:              Arc::new(NoopStore),
            // Fresh profile on restore — pre-snapshot ticks aren't
            // attributed here, only ticks after this point.
            profile:            Arc::new(TickProfile::default()),
        };
        let mut missing = Vec::new();
        for pid in snap.pool_order {
            let pool_snap = match snap.pools.get(&pid) {
                Some(s) => s.clone(),
                None    => { missing.push(pid); continue; }
            };
            let encoding = match encodings.remove(&pid) {
                Some(e) => e,
                None    => { missing.push(pid); continue; }
            };
            let pool = crate::pool::Pool::from_snapshot(pool_snap, encoding);
            fabric.register_pool(pool);
        }
        (fabric, missing)
    }

    /// Read the current (un-closed) tick's moment.  Used by [`crate::Brain`]
    /// to capture the multi-pool firing fingerprint for binding-concept
    /// emergence before `advance_tick` clears it.
    pub fn current_moment(&self) -> &Moment { &self.current }

    pub fn register_pool(&mut self, mut pool: Pool) -> PoolId {
        let id = pool.id();
        // Per ARCHITECTURE §17.9: a newly-registered pool inherits the
        // fabric's current persistence backend so its neurogenesis events
        // flow to the same WAL.  Logs the PoolRegistered baseline first
        // so recovery can recreate pools with the right config.
        let config = pool.config.clone();
        let encoding_name = pool.encoding_name().to_string();
        let event = WalEvent::PoolRegistered {
            pool_id: id,
            config,
            encoding_name,
        };
        if let Err(e) = self.store.append(&event) {
            tracing::warn!("WAL append failed for register_pool({}): {}", id, e);
        }
        pool.set_store(self.store.clone());
        self.pools.insert(id, Arc::new(RwLock::new(pool)));
        id
    }

    pub fn pool(&self, id: PoolId) -> Option<Arc<RwLock<Pool>>> {
        self.pools.get(&id).cloned()
    }

    pub fn pool_ids(&self) -> Vec<PoolId> {
        let mut ids: Vec<PoolId> = self.pools.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Advance to the next tick.  Closes the previous tick's moment
    /// (wiring cross-pool bonds for any co-fired pairs) and starts a
    /// fresh moment for the new tick.  Also runs per-pool housekeeping
    /// (decay + prune).
    pub fn advance_tick(&mut self) {
        let tick_t0 = std::time::Instant::now();

        // Close the prior moment: any pair of neurons in different pools
        // that both fired this tick gets a cross-pool axon terminal
        // grown / strengthened.  Same-pool co-firing isn't wired here
        // because within-pool concept emergence (the Pool's job) is the
        // proper mechanism for that — wiring atom→atom here would just
        // duplicate the atom-pair Hebbian effect.
        let phase_t0 = std::time::Instant::now();
        let lr = self.config.cross_pool_lr;
        let pool_ids: Vec<PoolId> = self.current.fired.keys().copied().collect();

        for i in 0..pool_ids.len() {
            for j in (i + 1)..pool_ids.len() {
                let pid_a = pool_ids[i];
                let pid_b = pool_ids[j];
                let fired_a = self.current.fired.get(&pid_a).cloned().unwrap_or_default();
                let fired_b = self.current.fired.get(&pid_b).cloned().unwrap_or_default();
                if fired_a.is_empty() || fired_b.is_empty() { continue; }

                let pool_a = self.pools.get(&pid_a).unwrap().clone();
                let pool_b = self.pools.get(&pid_b).unwrap().clone();
                {
                    let mut pa = pool_a.write();
                    let max_w = pa.config.max_weight;
                    let mut added: usize = 0;
                    for &na in &fired_a {
                        if let Some(neuron) = pa.get_mut(na) {
                            for &nb in &fired_b {
                                if neuron.reinforce_terminal(
                                    NeuronRef::new(pid_b, nb),
                                    lr, self.tick, max_w,
                                ) { added += 1; }
                            }
                        }
                    }
                    pa.total_terminals += added;
                }
                {
                    let mut pb = pool_b.write();
                    let max_w = pb.config.max_weight;
                    let mut added: usize = 0;
                    for &nb in &fired_b {
                        if let Some(neuron) = pb.get_mut(nb) {
                            for &na in &fired_a {
                                if neuron.reinforce_terminal(
                                    NeuronRef::new(pid_a, na),
                                    lr, self.tick, max_w,
                                ) { added += 1; }
                            }
                        }
                    }
                    pb.total_terminals += added;
                }
            }
        }

        self.profile.cross_pool_atom_wiring_ns
            .fetch_add(phase_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Stage 3 (position-aware binding): cross-pool concept→concept
        // wiring.  Concepts encode positional sequences in their
        // member-vec; co-firing concepts across pools represent
        // position-aligned bindings.  Atom-level wiring above can't
        // distinguish "P000" from "P010" because their firing-sets
        // are identical.  Concept-level wiring CAN, because the
        // emerged P000-concept and P010-concept are distinct neurons.
        let phase_t0 = std::time::Instant::now();
        let concept_lr = self.config.cross_pool_concept_lr;
        if concept_lr > 0.0 {
            // Snapshot per-pool concept firings from currently_firing.
            let mut concepts_by_pool: AHashMap<PoolId, Vec<NeuronId>> = AHashMap::new();
            for (pid, pool) in self.pools.iter() {
                let p = pool.read();
                let firing: Vec<NeuronId> = p.currently_firing()
                    .filter_map(|nid| p.get(nid).and_then(|n| {
                        if n.is_atom() { None } else { Some(nid) }
                    }))
                    .collect();
                if !firing.is_empty() {
                    concepts_by_pool.insert(*pid, firing);
                }
            }

            let active_pool_ids: Vec<PoolId> = concepts_by_pool.keys().copied().collect();
            for i in 0..active_pool_ids.len() {
                for j in (i + 1)..active_pool_ids.len() {
                    let pid_a = active_pool_ids[i];
                    let pid_b = active_pool_ids[j];
                    let concepts_a = concepts_by_pool[&pid_a].clone();
                    let concepts_b = concepts_by_pool[&pid_b].clone();

                    let pool_a = self.pools.get(&pid_a).unwrap().clone();
                    let pool_b = self.pools.get(&pid_b).unwrap().clone();
                    {
                        let mut pa = pool_a.write();
                        let max_w = pa.config.max_weight;
                        let mut added: usize = 0;
                        for &na in &concepts_a {
                            if let Some(n) = pa.get_mut(na) {
                                for &nb in &concepts_b {
                                    if n.reinforce_terminal(
                                        NeuronRef::new(pid_b, nb),
                                        concept_lr, self.tick, max_w,
                                    ) { added += 1; }
                                }
                            }
                        }
                        pa.total_terminals += added;
                    }
                    {
                        let mut pb = pool_b.write();
                        let max_w = pb.config.max_weight;
                        let mut added: usize = 0;
                        for &nb in &concepts_b {
                            if let Some(n) = pb.get_mut(nb) {
                                for &na in &concepts_a {
                                    if n.reinforce_terminal(
                                        NeuronRef::new(pid_a, na),
                                        concept_lr, self.tick, max_w,
                                    ) { added += 1; }
                                }
                            }
                        }
                        pb.total_terminals += added;
                    }
                }
            }
        }

        self.profile.cross_pool_concept_wiring_ns
            .fetch_add(phase_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Stage 2b: within-pool concept→concept temporal wiring.
        // For each pool, identify the CONCEPT neurons currently firing
        // (atoms excluded — they're handled by within-pool emergence
        // and cross-pool axons separately).  Grow a terminal from
        // every concept that fired last tick to every concept firing
        // now.  This is what gives the substrate "what follows what"
        // structure above the atom layer — needed for coherent
        // sequential generation (Stage 2's `Brain::generate`).
        let phase_t0 = std::time::Instant::now();
        let all_pool_ids: Vec<PoolId> = self.pools.keys().copied().collect();
        for pid in all_pool_ids {
            let pool = match self.pools.get(&pid) {
                Some(p) => p.clone(),
                None => continue,
            };
            let current_concepts: Vec<NeuronId> = {
                let p = pool.read();
                p.currently_firing()
                    .filter_map(|nid| p.get(nid).and_then(|n| {
                        if n.is_atom() { None } else { Some(nid) }
                    }))
                    .collect()
            };

            if let Some(prev) = self.prev_tick_concepts.get(&pid).cloned() {
                if !prev.is_empty() && !current_concepts.is_empty() {
                    let mut pw = pool.write();
                    let max_w = pw.config.max_weight;
                    let mut added: usize = 0;
                    for &src in &prev {
                        // Skip if source no longer exists (pruned).
                        if pw.get(src).is_none() { continue; }
                        for &dst in &current_concepts {
                            if src == dst { continue; }
                            if let Some(n) = pw.get_mut(src) {
                                if n.reinforce_terminal(
                                    NeuronRef::new(pid, dst),
                                    0.3,
                                    self.tick,
                                    max_w,
                                ) { added += 1; }
                            }
                        }
                    }
                    pw.total_terminals += added;
                }
            }
            self.prev_tick_concepts.insert(pid, current_concepts);
        }

        self.profile.within_pool_temporal_ns
            .fetch_add(phase_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Per-pool housekeeping: decay every terminal, prune sub-floor,
        // apply heterosynaptic LTD, apply k-WTA sparsity.
        //
        // Mode is W1Z4RD_TICK_HOUSEKEEPING:
        //   eager  (default): walk every neuron in every pool, apply
        //          (1-ε) to every terminal.  O(total_terminals) per tick.
        //          Was the dominant cost — 99.994 % of empty-tick time
        //          per /tick_profile diagnostics in commit c5f5642.
        //   lazy:  decay only neurons that fired THIS tick.  All other
        //          neurons' decay is applied lazily on next access via
        //          Neuron::apply_pending_decay, using last_decayed_tick.
        //          Mathematically identical to eager because
        //          weight × (1-ε)^k applied once == k eager applications.
        //          Cost is O(|firing_set|), not O(total_terminals).
        //   skip:  no-op.  For benchmarking only — terminal weights never
        //          decay; over long runs the fabric bloats with dead
        //          terminals.  Use lazy for production.
        let phase_t0 = std::time::Instant::now();
        let mode = std::env::var("W1Z4RD_TICK_HOUSEKEEPING")
            .unwrap_or_else(|_| "eager".to_string());
        match mode.as_str() {
            "skip" => { /* nothing */ }
            "lazy" => {
                // Decay only what fired this tick.  Cross-pool wiring
                // sites also call apply_pending_decay on the neuron
                // they're about to reinforce, so any neuron involved
                // in cross-pool growth gets accurate decay applied
                // before its weights are touched.
                let tick = self.tick;
                for (_, pool) in self.pools.iter() {
                    pool.write().tick_housekeeping_lazy(tick);
                }
            }
            _ /* eager */ => {
                for (_, pool) in self.pools.iter() {
                    pool.write().tick_housekeeping(self.tick);
                }
            }
        }
        self.profile.housekeeping_ns
            .fetch_add(phase_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        self.profile.ticks.fetch_add(1, Ordering::Relaxed);
        self.profile.total_ns
            .fetch_add(tick_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        self.tick += 1;
        self.current = Moment::new(self.tick);

        // Per ARCHITECTURE §17.9: TickAdvanced is the per-tick boundary
        // marker.  Lets crash-recovery rebuild the fabric's tick counter
        // and order events temporally.
        let event = WalEvent::TickAdvanced { new_tick: self.tick };
        if let Err(e) = self.store.append(&event) {
            tracing::warn!("WAL append failed for TickAdvanced({}): {}", self.tick, e);
        }
    }

    /// Observe a sensor frame into the named pool.  Records what fired
    /// into the current moment.  The fabric does not advance the tick
    /// automatically — the caller chooses when to close the moment via
    /// `advance_tick`, allowing multiple pools to be observed in the same
    /// tick window (which is exactly when cross-pool wiring should happen).
    pub fn observe(&mut self, pool_id: PoolId, frame: &[u8]) -> Vec<NeuronId> {
        let pool = self.pools.get(&pool_id).expect("unknown pool").clone();
        let fired = pool.write().observe_frame(frame, self.tick);
        self.current.fired.entry(pool_id).or_default().extend(fired.iter().copied());
        fired
    }

    /// Register a neuron as having fired this tick without going through
    /// `observe_frame`.  Used by injected activations (e.g.
    /// [`crate::Brain::fire_action`] during supervised training) so the
    /// moment buffer captures them and cross-pool wiring runs on tick
    /// close.  Idempotent on repeated calls.
    pub fn mark_fired(&mut self, pool_id: PoolId, neuron: NeuronId) {
        let entry = self.current.fired.entry(pool_id).or_default();
        if !entry.contains(&neuron) {
            entry.push(neuron);
        }
    }

    /// Uniform propagation per spec §4.A.  Walks the terminals of every
    /// currently-firing neuron in `from_pool`, depositing activation at
    /// each target (which may be in any pool).  Repeats for `max_hops`.
    ///
    /// Returns the activation map per pool — caller reads this to see
    /// what the input atoms activated across the whole fabric.
    pub fn propagate(&self, from_pool: PoolId) -> AHashMap<PoolId, AHashMap<NeuronId, f32>> {
        let mut acts: AHashMap<PoolId, AHashMap<NeuronId, f32>> = AHashMap::new();
        let pool_arc = match self.pools.get(&from_pool) {
            Some(p) => p.clone(),
            None => return acts,
        };
        // Seed with currently-firing neurons from the source pool.
        {
            let pool = pool_arc.read();
            let mut seed: AHashMap<NeuronId, f32> = AHashMap::new();
            for nid in pool.currently_firing() {
                seed.insert(nid, pool.activation(nid));
            }
            acts.insert(from_pool, seed);
        }

        for hop in 0..self.config.max_hops {
            let snapshot = acts.clone();
            for (pid, neurons) in snapshot.iter() {
                let pool_a = match self.pools.get(pid) {
                    Some(p) => p.clone(),
                    None => continue,
                };
                let pool = pool_a.read();
                for (&nid, &activation) in neurons.iter() {
                    if activation < 0.001 { continue; }
                    let neuron = match pool.get(nid) {
                        Some(n) => n,
                        None => continue,
                    };
                    // Use sqrt(fan-out) normalization (spec §1.5 spirit —
                    // common neurons spread their signal thinly, no kwta).
                    let fan_out = (neuron.terminals.len() as f32).sqrt().max(1.0);
                    for t in &neuron.terminals {
                        let contribution = activation
                            * t.effective_weight()
                            * self.config.hop_decay
                            / fan_out;
                        if contribution.abs() < 0.0001 { continue; }
                        let tgt_pool = acts.entry(t.target.pool).or_default();
                        *tgt_pool.entry(t.target.neuron).or_insert(0.0) += contribution;
                    }
                }
            }
            // Hop-level decay applied via hop_decay above; nothing more
            // to do for now.  Future: add inhibitory effects via
            // neuron.kind == Inhibitory subtracting from target activation.
            let _ = hop;
        }

        acts
    }

    /// Stage 13A — iterative settling for the "epoxy mould" creation
    /// model (ARCHITECTURE.md §1.3 resonance dynamics).
    ///
    /// Where [`Self::propagate`] does `max_hops` of monotonic
    /// accumulation, `settle` runs *fixed-point iteration*: each step
    /// propagates one hop forward from the current state, then
    /// **sharpens** each pool's activation map (soft top-K
    /// normalization) so the next iteration propagates from the
    /// sharpened state rather than the accumulated total.  This is
    /// what lets cross-pool feedback resonate — a coherent state in
    /// one pool fires its concepts in another pool, which fires back,
    /// and the substrate settles into a joint configuration where all
    /// active pools "agree."
    ///
    /// Convergence: returns as soon as no pool's top-K activation set
    /// has changed more than `eps` since the previous iteration, or
    /// after `max_iter` iterations.
    ///
    /// This is PARALLEL to `propagate`; the existing one-pass path
    /// remains the contract for `integrate()` and `/chat`.  Stage 13
    /// only adds the resonant path — it does not modify the
    /// retrieval-side dynamics that the Stage 7-12 work depends on.
    pub fn settle(
        &self,
        from_pool: PoolId,
        max_iter:  usize,
        top_k:     usize,
        eps:       f32,
    ) -> SettleResult {
        let pool_arc = match self.pools.get(&from_pool) {
            Some(p) => p.clone(),
            None    => return SettleResult::empty(),
        };

        // Seed state from source pool's currently-firing neurons.
        // The seed atoms are the "mould" — they must remain pinned at
        // activation 1.0 for the entire settling run so cross-pool
        // feedback can flow around them rather than displace them.
        // Without this pin, the substrate collapses to whichever
        // attractor has the densest axon network, regardless of
        // what was queried.
        let mut state: AHashMap<PoolId, AHashMap<NeuronId, f32>> = AHashMap::new();
        let seed_atoms: AHashMap<NeuronId, f32> = {
            let pool = pool_arc.read();
            let mut seed: AHashMap<NeuronId, f32> = AHashMap::new();
            for nid in pool.currently_firing() {
                seed.insert(nid, 1.0);
            }
            seed
        };
        if seed_atoms.is_empty() { return SettleResult::empty(); }
        state.insert(from_pool, seed_atoms.clone());

        let mut iterations_run = 0usize;
        let mut converged = false;
        // Damping retains some of the prior state into the next
        // iteration — without this, the source pool's activation
        // could be replaced by feedback from other pools and the
        // mould "drifts" off the prompt.  0.5 retains half the prior
        // sharpened activation as a persistent constraint.
        let damping: f32 = 0.5;

        for it in 0..max_iter {
            iterations_run = it + 1;

            // One propagation hop forward from the *current* state.
            let mut next: AHashMap<PoolId, AHashMap<NeuronId, f32>> = AHashMap::new();
            // Carry forward damped current state — this is the
            // "persistence" that prevents the mould from drifting.
            for (pid, neurons) in state.iter() {
                let entry = next.entry(*pid).or_default();
                for (&nid, &a) in neurons.iter() {
                    *entry.entry(nid).or_insert(0.0) += a * damping;
                }
            }
            // Propagate from each currently-active neuron one hop.
            for (pid, neurons) in state.iter() {
                let pool_a = match self.pools.get(pid) {
                    Some(p) => p.clone(),
                    None    => continue,
                };
                let pool = pool_a.read();
                for (&nid, &activation) in neurons.iter() {
                    if activation < 0.001 { continue; }
                    let neuron = match pool.get(nid) {
                        Some(n) => n,
                        None    => continue,
                    };
                    let fan_out = (neuron.terminals.len() as f32).sqrt().max(1.0);
                    for t in &neuron.terminals {
                        let contribution = activation
                            * t.effective_weight()
                            * self.config.hop_decay
                            / fan_out;
                        if contribution.abs() < 0.0001 { continue; }
                        let tgt_pool = next.entry(t.target.pool).or_default();
                        *tgt_pool.entry(t.target.neuron).or_insert(0.0) += contribution;
                    }
                }
            }

            // Sharpen each pool: keep top_k by activation, normalize
            // top to 1.0.  This is the soft-WTA that makes successive
            // iterations propagate from a SHARPER state, not the
            // accumulated sum.
            let mut sharpened: AHashMap<PoolId, AHashMap<NeuronId, f32>> =
                AHashMap::with_capacity(next.len());
            for (pid, neurons) in next.iter() {
                let mut entries: Vec<(NeuronId, f32)> = neurons.iter()
                    .map(|(k, v)| (*k, *v)).collect();
                entries.sort_by(|a, b|
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                entries.truncate(top_k);
                let max = entries.first().map(|(_, a)| *a).unwrap_or(0.0);
                let mut pool_state: AHashMap<NeuronId, f32> =
                    AHashMap::with_capacity(entries.len());
                if max > 0.0 {
                    for (nid, a) in entries {
                        pool_state.insert(nid, a / max);
                    }
                }
                if !pool_state.is_empty() {
                    sharpened.insert(*pid, pool_state);
                }
            }

            // Pin the mould — re-clamp the source pool's seed atoms
            // to activation 1.0 in the sharpened state.  Without this,
            // feedback from other pools can sharpen the source pool
            // around DIFFERENT atoms than the query, and the substrate
            // settles onto the wrong constraint.
            let source_entry = sharpened.entry(from_pool).or_default();
            for (&nid, &v) in seed_atoms.iter() {
                source_entry.insert(nid, v);
            }

            // Convergence check: did any pool's top-K set change by
            // more than eps?  We compare sharpened to previous state.
            let delta = top_k_delta(&state, &sharpened, top_k);
            state = sharpened;
            if delta < eps {
                converged = true;
                break;
            }
        }

        SettleResult {
            pool_activations: state,
            iterations_run,
            converged,
        }
    }
}

/// Stage 13A — return shape for [`Fabric::settle`].
#[derive(Debug, Clone)]
pub struct SettleResult {
    pub pool_activations: AHashMap<PoolId, AHashMap<NeuronId, f32>>,
    pub iterations_run:   usize,
    pub converged:        bool,
}

impl SettleResult {
    fn empty() -> Self {
        Self {
            pool_activations: AHashMap::new(),
            iterations_run:   0,
            converged:        true,
        }
    }
    /// Top-N (NeuronId, activation) entries in a given pool from the
    /// settled state.  Returned in descending activation order.
    pub fn top_in_pool(&self, pool: PoolId, n: usize) -> Vec<(NeuronId, f32)> {
        let pool_map = match self.pool_activations.get(&pool) {
            Some(m) => m,
            None    => return Vec::new(),
        };
        let mut entries: Vec<(NeuronId, f32)> = pool_map.iter()
            .map(|(k, v)| (*k, *v)).collect();
        entries.sort_by(|a, b|
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(n);
        entries
    }
}

/// L1-distance between two settled states over their top-K active
/// neurons per pool.  Used as the convergence criterion inside
/// [`Fabric::settle`].
fn top_k_delta(
    prev: &AHashMap<PoolId, AHashMap<NeuronId, f32>>,
    next: &AHashMap<PoolId, AHashMap<NeuronId, f32>>,
    top_k: usize,
) -> f32 {
    let mut total = 0.0_f32;
    let pools: ahash::AHashSet<&PoolId> = prev.keys().chain(next.keys()).collect();
    for pid in pools {
        let p = prev.get(pid);
        let n = next.get(pid);
        // Collect top-k of each side.
        let topk = |m: Option<&AHashMap<NeuronId, f32>>| -> Vec<(NeuronId, f32)> {
            let mut v: Vec<(NeuronId, f32)> = m.into_iter()
                .flat_map(|h| h.iter().map(|(k, v)| (*k, *v)))
                .collect();
            v.sort_by(|a, b|
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            v.truncate(top_k);
            v
        };
        let pv = topk(p);
        let nv = topk(n);
        let pset: AHashMap<NeuronId, f32> = pv.into_iter().collect();
        let nset: AHashMap<NeuronId, f32> = nv.into_iter().collect();
        let all_keys: ahash::AHashSet<NeuronId> = pset.keys().chain(nset.keys()).copied().collect();
        for k in all_keys {
            let a = pset.get(&k).copied().unwrap_or(0.0);
            let b = nset.get(&k).copied().unwrap_or(0.0);
            total += (a - b).abs();
        }
    }
    total
}
