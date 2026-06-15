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

/// Per-phase nanosecond counters for `Pool::observe_frame` + the brain
/// orchestration around it.  Same Arc<…> + AtomicU64 pattern as
/// [`TickProfile`] so the HTTP layer reads counters without taking the
/// brain mutex.  Exposed as `/brain/observe_profile` for the same
/// "find the dominant cost without a sampling profiler" workflow.
#[derive(Debug, Default)]
pub struct ObserveProfile {
    pub observes:               AtomicU64,
    /// `encoding.atomize(frame)` — turning bytes into atom labels.
    pub atomize_ns:             AtomicU64,
    /// `ensure_atom` + `push_recent` + activation/firing inserts for
    /// every atom in the frame.
    pub atom_fire_ns:           AtomicU64,
    /// `apply_pending_decay` on each fired atom neuron — walks its
    /// terminals, top suspect for big-brain slowness.
    pub lazy_decay_ns:          AtomicU64,
    /// `collapse_tail_to_concept` loop (one pass per atom, may loop
    /// for multi-level collapses).
    pub collapse_ns:            AtomicU64,
    /// `check_concept_emergence` — sequence count bumps + threshold
    /// detection (or deferred enqueue under W1Z4RD_DEFER_PROMOTION).
    pub concept_emergence_ns:   AtomicU64,
    /// k-WTA sparsity, predictive-coding surprise update, EMA refresh
    /// at end of observe_frame.
    pub end_of_frame_ns:        AtomicU64,
    /// QA-capture path: scan `recent_frames` + push to `qa_db`.  Pure
    /// Brain::observe overhead on top of `fabric.observe`.
    pub qa_capture_ns:          AtomicU64,
    /// WAL events appended during this observe (atom births, fires,
    /// concept promotions).  Includes mmap append cost.
    pub wal_events:             AtomicU64,
    pub wal_append_ns:          AtomicU64,
    pub total_ns:               AtomicU64,
}

impl ObserveProfile {
    pub fn snapshot(&self) -> ObserveProfileSnapshot {
        use std::sync::atomic::Ordering::Relaxed;
        ObserveProfileSnapshot {
            observes:             self.observes.load(Relaxed),
            atomize_ns:           self.atomize_ns.load(Relaxed),
            atom_fire_ns:         self.atom_fire_ns.load(Relaxed),
            lazy_decay_ns:        self.lazy_decay_ns.load(Relaxed),
            collapse_ns:          self.collapse_ns.load(Relaxed),
            concept_emergence_ns: self.concept_emergence_ns.load(Relaxed),
            end_of_frame_ns:      self.end_of_frame_ns.load(Relaxed),
            qa_capture_ns:        self.qa_capture_ns.load(Relaxed),
            wal_events:           self.wal_events.load(Relaxed),
            wal_append_ns:        self.wal_append_ns.load(Relaxed),
            total_ns:             self.total_ns.load(Relaxed),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ObserveProfileSnapshot {
    pub observes:             u64,
    pub atomize_ns:           u64,
    pub atom_fire_ns:         u64,
    pub lazy_decay_ns:        u64,
    pub collapse_ns:          u64,
    pub concept_emergence_ns: u64,
    pub end_of_frame_ns:      u64,
    pub qa_capture_ns:        u64,
    pub wal_events:           u64,
    pub wal_append_ns:        u64,
    pub total_ns:             u64,
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
    /// Per-phase observe-path timing.  Sibling to `profile` but covers
    /// `Pool::observe_frame` + the Brain orchestration around it (QA
    /// capture, WAL append).  Exposed as `/brain/observe_profile`.
    pub observe_profile: Arc<ObserveProfile>,
    /// Continuous cost-aware tier orchestrator state.  Runs in
    /// `advance_tick` on a configurable cadence.  Default params come
    /// from env (`W1Z4RD_TIER_*`) so production can iterate without
    /// recompiling.  See [`crate::tier_orchestrator`].
    pub(crate) orchestrator:       parking_lot::Mutex<crate::tier_orchestrator::TierOrchestrator>,
    pub(crate) orchestrator_params: parking_lot::Mutex<crate::tier_orchestrator::OrchestratorParams>,
    /// Cumulative orchestrator counters.  Atomic so `/tier_orchestrator_stats`
    /// reads them without taking the brain mutex.
    pub orchestrator_stats: Arc<crate::tier_orchestrator::OrchestratorStats>,
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
            observe_profile:    Arc::new(ObserveProfile::default()),
            orchestrator:       parking_lot::Mutex::new(crate::tier_orchestrator::TierOrchestrator::new()),
            orchestrator_params: parking_lot::Mutex::new(crate::tier_orchestrator::OrchestratorParams::from_env_or_disabled()),
            orchestrator_stats: Arc::new(crate::tier_orchestrator::OrchestratorStats::default()),
        }
    }

    /// Snapshot of the per-phase tick timing.  Cumulative counters; for
    /// per-tick mean divide by snapshot.ticks.
    pub fn tick_profile(&self) -> TickProfileSnapshot {
        self.profile.snapshot()
    }

    /// Snapshot of the per-phase observe-path timing.  Cumulative
    /// counters; for per-observe mean divide by snapshot.observes.
    pub fn observe_profile(&self) -> ObserveProfileSnapshot {
        self.observe_profile.snapshot()
    }

    /// Cumulative tier-orchestrator counters.  Exposed as
    /// `/brain/tier_orchestrator_stats`.
    pub fn tier_orchestrator_stats(&self) -> crate::tier_orchestrator::OrchestratorStatsSnapshot {
        self.orchestrator_stats.snapshot()
    }

    /// Replace the active orchestrator params (used by tests + the
    /// /brain/tier_orchestrator/params endpoint).
    pub fn set_tier_orchestrator_params(&self, params: crate::tier_orchestrator::OrchestratorParams) {
        *self.orchestrator_params.lock() = params;
    }

    /// Snapshot of the current params (so callers can edit-and-set).
    pub fn orchestrator_params_snapshot(&self) -> crate::tier_orchestrator::OrchestratorParams {
        *self.orchestrator_params.lock()
    }

    /// One pass of the cost-aware tier orchestrator.  Invoked from
    /// [`Fabric::advance_tick`] at the configured cadence; can also
    /// be called manually (e.g. from a maintenance loop) for batch
    /// drains.
    ///
    /// Returns the number of neurons evicted this pass (`0` when the
    /// cadence gate skips the pass).
    pub fn run_tier_orchestrator_pass(&mut self) -> usize {
        use crate::tier_orchestrator::TierOrchestrator;
        let t0 = std::time::Instant::now();
        let params = *self.orchestrator_params.lock();
        // Cadence gate.
        if params.run_every_n_ticks == 0 || params.run_every_n_ticks == u64::MAX {
            return 0;
        }
        if self.tick % params.run_every_n_ticks != 0 {
            return 0;
        }
        let stats = &self.orchestrator_stats;
        stats.passes.fetch_add(1, Ordering::Relaxed);
        let current_tick = self.tick;
        let pool_ids: Vec<PoolId> = self.pools.keys().copied().collect();
        let mut total_evicted: usize = 0;
        let mut last_pressure_x1k: u64 = 1000;
        for pid in pool_ids {
            let Some(pool_arc) = self.pools.get(&pid) else { continue };
            // Phase A: read pool — gather candidates within budget,
            // measure pressure, score, pick evict set.
            let candidates: Vec<(NeuronId, f32)> = {
                let p = pool_arc.read();
                let n_total = p.neurons_len();
                if n_total == 0 { continue; }
                let pressure = TierOrchestrator::pressure_factor(
                    p.total_terminals, params.target_terminals_per_pool);
                last_pressure_x1k = (pressure * 1000.0) as u64;
                let start = self.orchestrator.lock().advance_cursor(
                    pid, params.scan_budget.min(n_total), n_total);
                let mut chosen: Vec<(NeuronId, f32)> = Vec::new();
                let budget = params.scan_budget.min(n_total);
                let mut scanned: u64 = 0;
                for k in 0..budget {
                    let idx = (start + k) % n_total;
                    let Some(n) = p.neuron_at(idx) else { continue };
                    scanned += 1;
                    // Filters (mirror Brain::run_eviction_pass policy):
                    if n.is_atom() { continue; }
                    if p.is_evicted(n.id) { continue; }
                    // Newborn protection — give freshly created concepts
                    // at least `min_age_ticks` to demonstrate their
                    // salience.  Without this the orchestrator can throw
                    // a concept away on the same tick it was born.
                    if current_tick.saturating_sub(n.born_tick) < params.min_age_ticks {
                        continue;
                    }
                    // Pin candidate: members.is_empty() && !is_atom() means
                    // a concept that has decoded children referenced from
                    // other pools — we conservatively don't pin here, but
                    // the score function takes a `pinned` flag callers can
                    // wire in later.
                    let s = TierOrchestrator::score(
                        &params,
                        n.terminals.len(),
                        n.last_fired_tick,
                        current_tick,
                        n.salience_ema,
                        false,
                    );
                    if TierOrchestrator::should_evict(&params, s, pressure) {
                        chosen.push((n.id, s));
                        if chosen.len() >= params.max_evict_per_pass { break; }
                    }
                }
                stats.neurons_scanned.fetch_add(scanned, Ordering::Relaxed);
                chosen
            };
            // Phase B: write pool — actually evict.  Brief write lock.
            if !candidates.is_empty() {
                let mut p = pool_arc.write();
                for (nid, _score) in candidates {
                    match p.evict_neuron(nid) {
                        Ok(true) => {
                            stats.neurons_evicted.fetch_add(1, Ordering::Relaxed);
                            total_evicted += 1;
                        }
                        Ok(false) => {} // skipped (atom / already evicted)
                        Err(_) => {
                            stats.evict_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        stats.last_pressure_x1k.store(last_pressure_x1k, Ordering::Relaxed);
        stats.total_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        total_evicted
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
            observe_profile:    Arc::new(ObserveProfile::default()),
            orchestrator:       parking_lot::Mutex::new(crate::tier_orchestrator::TierOrchestrator::new()),
            orchestrator_params: parking_lot::Mutex::new(crate::tier_orchestrator::OrchestratorParams::from_env_or_disabled()),
            orchestrator_stats: Arc::new(crate::tier_orchestrator::OrchestratorStats::default()),
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
        // Per-tick (not cumulative) phase nanoseconds so we can log the
        // breakdown for individual slow ticks.  The Arc<TickProfile>
        // atomics still accumulate as before; these are local deltas.
        let mut per_tick = [0u64; 4];   // atom, concept, temporal, housekeeping

        // Island-architecture gate: when enabled, the hot cross-pool
        // wiring path only fires for neurons sharing a domain_id (or
        // when either side is unassigned, domain_id=0).  Cross-domain
        // wiring becomes "bridge candidate" work, deferred to the
        // integration cycle (Brain::integrate).  Read once per tick to
        // keep the inner loop branch-cheap.
        let domain_mode = std::env::var("W1Z4RD_DOMAIN_MODE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // Soft-gate scale: cross-domain wiring still forms, but at a
        // fraction of the within-domain learning rate.  Bridges grow
        // continuously during normal training; sustained cross-domain
        // co-firing strengthens them to the lock threshold over time
        // and gives the substrate "X is like Y" reasoning without an
        // explicit integration cycle.  Defaults to 0.1; tunable via
        // W1Z4RD_CROSS_DOMAIN_SCALE for Phase B self-tuning.
        // Top-K cap on firing CONCEPTS per pool per tick.  Phase 2
        // (cross-pool concept wiring) and Phase 3 (within-pool temporal
        // concept wiring) are both O(|firing_a| × |firing_b|), so once
        // the brain has millions of concepts and propagation can light
        // up thousands of them per tick, tick time grows quadratically
        // and observe / tick latency explodes (we measured 2-3 s at
        // 7.2 M neurons before this cap).  Capping each pool's
        // firing-concept set to its top-K by activation bounds tick
        // time to O(K²) regardless of brain size, at the cost of only
        // wiring the most-active concepts each tick — sparse / weak
        // concepts still get wired the tick they fire above the cut.
        //
        // Atoms are NOT capped — phase 1's atom wiring is already
        // bounded by the byte-alphabet (≤ 256 distinct atoms per
        // pool).  Set W1Z4RD_TICK_CONCEPT_CAP=0 to disable the cap
        // (legacy unbounded behaviour, fine for small brains).
        let concept_cap: usize = std::env::var("W1Z4RD_TICK_CONCEPT_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256);

        let cross_domain_scale: f32 = std::env::var("W1Z4RD_CROSS_DOMAIN_SCALE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.1);

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
                // Resolve domain_id for each firing neuron once per
                // tick rather than per (a,b) pair.  When domain_mode is
                // off, we skip the lookup entirely and the wiring loop
                // behaves identically to the pre-island code path.
                let fired_a_dom: Vec<u32> = if domain_mode {
                    let pa = pool_a.read();
                    fired_a.iter().map(|&id| pa.get(id).map(|n| n.domain_id).unwrap_or(0)).collect()
                } else { Vec::new() };
                let fired_b_dom: Vec<u32> = if domain_mode {
                    let pb = pool_b.read();
                    fired_b.iter().map(|&id| pb.get(id).map(|n| n.domain_id).unwrap_or(0)).collect()
                } else { Vec::new() };
                let cross_domain = |i: usize, j: usize| -> bool {
                    if !domain_mode { return false; }
                    let da = fired_a_dom[i];
                    let db = fired_b_dom[j];
                    da != 0 && db != 0 && da != db
                };
                {
                    let mut pa = pool_a.write();
                    let max_w = pa.config.max_weight;
                    let mut added: usize = 0;
                    for (i, &na) in fired_a.iter().enumerate() {
                        if let Some(neuron) = pa.get_mut(na) {
                            for (j, &nb) in fired_b.iter().enumerate() {
                                let eff_lr = if cross_domain(i, j) { lr * cross_domain_scale } else { lr };
                                if neuron.reinforce_terminal(
                                    NeuronRef::new(pid_b, nb),
                                    eff_lr, self.tick, max_w,
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
                    for (j, &nb) in fired_b.iter().enumerate() {
                        if let Some(neuron) = pb.get_mut(nb) {
                            for (i, &na) in fired_a.iter().enumerate() {
                                let eff_lr = if cross_domain(i, j) { lr * cross_domain_scale } else { lr };
                                if neuron.reinforce_terminal(
                                    NeuronRef::new(pid_a, na),
                                    eff_lr, self.tick, max_w,
                                ) { added += 1; }
                            }
                        }
                    }
                    pb.total_terminals += added;
                }
            }
        }

        let dt = phase_t0.elapsed().as_nanos() as u64;
        per_tick[0] = dt;
        self.profile.cross_pool_atom_wiring_ns.fetch_add(dt, Ordering::Relaxed);

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
            // Snapshot per-pool concept firings from currently_firing,
            // capped at top-K by activation when W1Z4RD_TICK_CONCEPT_CAP
            // is set.  This is the load-bearing fix that keeps Phase 2
            // tick time O(K²) regardless of brain size.
            let mut concepts_by_pool: AHashMap<PoolId, Vec<NeuronId>> = AHashMap::new();
            for (pid, pool) in self.pools.iter() {
                let p = pool.read();
                let mut firing: Vec<(NeuronId, f32)> = p.currently_firing()
                    .filter_map(|nid| p.get(nid).and_then(|n| {
                        if n.is_atom() { None } else { Some((nid, p.activation(nid))) }
                    }))
                    .collect();
                if concept_cap > 0 && firing.len() > concept_cap {
                    // Partial sort by activation descending, then truncate.
                    firing.sort_by(|a, b| b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal));
                    firing.truncate(concept_cap);
                }
                let firing_ids: Vec<NeuronId> = firing.into_iter()
                    .map(|(id, _)| id).collect();
                if !firing_ids.is_empty() {
                    concepts_by_pool.insert(*pid, firing_ids);
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
                    // Same domain gating as cross-pool atom wiring,
                    // applied at concept granularity.
                    let dom_a: Vec<u32> = if domain_mode {
                        let pa = pool_a.read();
                        concepts_a.iter().map(|&id| pa.get(id).map(|n| n.domain_id).unwrap_or(0)).collect()
                    } else { Vec::new() };
                    let dom_b: Vec<u32> = if domain_mode {
                        let pb = pool_b.read();
                        concepts_b.iter().map(|&id| pb.get(id).map(|n| n.domain_id).unwrap_or(0)).collect()
                    } else { Vec::new() };
                    let cross_dom = |i: usize, j: usize| -> bool {
                        if !domain_mode { return false; }
                        let da = dom_a[i]; let db = dom_b[j];
                        da != 0 && db != 0 && da != db
                    };
                    {
                        let mut pa = pool_a.write();
                        let max_w = pa.config.max_weight;
                        let mut added: usize = 0;
                        for (i, &na) in concepts_a.iter().enumerate() {
                            if let Some(n) = pa.get_mut(na) {
                                for (j, &nb) in concepts_b.iter().enumerate() {
                                    let eff_lr = if cross_dom(i, j) { concept_lr * cross_domain_scale } else { concept_lr };
                                    if n.reinforce_terminal(
                                        NeuronRef::new(pid_b, nb),
                                        eff_lr, self.tick, max_w,
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
                        for (j, &nb) in concepts_b.iter().enumerate() {
                            if let Some(n) = pb.get_mut(nb) {
                                for (i, &na) in concepts_a.iter().enumerate() {
                                    let eff_lr = if cross_dom(i, j) { concept_lr * cross_domain_scale } else { concept_lr };
                                    if n.reinforce_terminal(
                                        NeuronRef::new(pid_a, na),
                                        eff_lr, self.tick, max_w,
                                    ) { added += 1; }
                                }
                            }
                        }
                        pb.total_terminals += added;
                    }
                }
            }
        }

        let dt = phase_t0.elapsed().as_nanos() as u64;
        per_tick[1] = dt;
        self.profile.cross_pool_concept_wiring_ns.fetch_add(dt, Ordering::Relaxed);

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
                let mut firing: Vec<(NeuronId, f32)> = p.currently_firing()
                    .filter_map(|nid| p.get(nid).and_then(|n| {
                        if n.is_atom() { None } else { Some((nid, p.activation(nid))) }
                    }))
                    .collect();
                // Phase 3 cap by activation — same rationale as Phase 2.
                if concept_cap > 0 && firing.len() > concept_cap {
                    firing.sort_by(|a, b| b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal));
                    firing.truncate(concept_cap);
                }
                firing.into_iter().map(|(id, _)| id).collect()
            };

            if let Some(prev_full) = self.prev_tick_concepts.get(&pid).cloned() {
                // The PREVIOUS-tick concept set may have been recorded
                // before the cap landed (or saved from a brain.bin
                // checkpoint), so it may be far larger than `concept_cap`.
                // Cap it on-read to bound the Phase 3 product
                // |prev| × |current| at K².  Activation isn't available
                // for the previous tick (we don't snapshot it), so we
                // just truncate — the substrate's training signal is
                // dominated by the most-recent firings anyway.
                let prev: Vec<NeuronId> = if concept_cap > 0
                    && prev_full.len() > concept_cap {
                    prev_full.into_iter().take(concept_cap).collect()
                } else { prev_full };
                if !prev.is_empty() && !current_concepts.is_empty() {
                    let mut pw = pool.write();
                    let max_w = pw.config.max_weight;
                    // Domain gate at within-pool granularity: prevent
                    // sequence-temporal wiring across islands (a code
                    // concept "follows" a chemistry concept only as a
                    // coincidence — let the integration cycle decide
                    // if that's structurally meaningful).
                    let dom_src: Vec<u32> = if domain_mode {
                        prev.iter().map(|&id| pw.get(id).map(|n| n.domain_id).unwrap_or(0)).collect()
                    } else { Vec::new() };
                    let dom_dst: Vec<u32> = if domain_mode {
                        current_concepts.iter().map(|&id| pw.get(id).map(|n| n.domain_id).unwrap_or(0)).collect()
                    } else { Vec::new() };
                    let mut added: usize = 0;
                    for (i, &src) in prev.iter().enumerate() {
                        // Skip if source no longer exists (pruned).
                        if pw.get(src).is_none() { continue; }
                        for (j, &dst) in current_concepts.iter().enumerate() {
                            if src == dst { continue; }
                            let eff_lr = if domain_mode {
                                let da = dom_src[i]; let db = dom_dst[j];
                                if da != 0 && db != 0 && da != db {
                                    0.3 * cross_domain_scale
                                } else { 0.3 }
                            } else { 0.3 };
                            if let Some(n) = pw.get_mut(src) {
                                if n.reinforce_terminal(
                                    NeuronRef::new(pid, dst),
                                    eff_lr,
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

        let dt = phase_t0.elapsed().as_nanos() as u64;
        per_tick[2] = dt;
        self.profile.within_pool_temporal_ns.fetch_add(dt, Ordering::Relaxed);

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
        let dt = phase_t0.elapsed().as_nanos() as u64;
        per_tick[3] = dt;
        self.profile.housekeeping_ns.fetch_add(dt, Ordering::Relaxed);

        // Phase 5: continuous cost-aware tier orchestration.  Runs at
        // the configured cadence (default every tick).  No-op when the
        // pool has no cold tier attached.  See tier_orchestrator module.
        let _ = self.run_tier_orchestrator_pass();

        let tick_total = tick_t0.elapsed().as_nanos() as u64;
        self.profile.ticks.fetch_add(1, Ordering::Relaxed);
        self.profile.total_ns.fetch_add(tick_total, Ordering::Relaxed);

        // Outlier-tick diagnostic: when a single tick crosses the
        // configurable threshold (default 5000 ms), log its breakdown
        // so we can see which phase blew up.  Most ticks are <<1 s; an
        // occasional outlier above 5 s is the symptom we're hunting.
        let threshold_ms = std::env::var("W1Z4RD_SLOW_TICK_MS")
            .ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(5_000);
        if tick_total > threshold_ms * 1_000_000 {
            let total_ms = tick_total / 1_000_000;
            let atom_ms = per_tick[0] / 1_000_000;
            let concept_ms = per_tick[1] / 1_000_000;
            let temporal_ms = per_tick[2] / 1_000_000;
            let hk_ms = per_tick[3] / 1_000_000;
            let accounted = atom_ms + concept_ms + temporal_ms + hk_ms;
            // tick_total - accounted = time spent in observe/store/etc
            // outside the four phases we instrument.
            let unaccounted_ms = total_ms.saturating_sub(accounted);
            eprintln!(
                "[slow-tick] tick={} total={}ms  atom={}ms concept={}ms temporal={}ms hk={}ms unaccounted={}ms",
                self.tick, total_ms, atom_ms, concept_ms, temporal_ms, hk_ms, unaccounted_ms,
            );
        }

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
        let total_t0 = std::time::Instant::now();
        let pool = self.pools.get(&pool_id).expect("unknown pool").clone();
        let prof = self.observe_profile.clone();
        let fired = pool.write().observe_frame(frame, self.tick, Some(&prof));
        self.current.fired.entry(pool_id).or_default().extend(fired.iter().copied());
        let total_ns = total_t0.elapsed().as_nanos() as u64;
        prof.observes.fetch_add(1, Ordering::Relaxed);
        prof.total_ns.fetch_add(total_ns, Ordering::Relaxed);
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
