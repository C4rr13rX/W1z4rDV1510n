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

use crate::neuron::{NeuronId, NeuronRef, PoolId};
use crate::pool::Pool;

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
#[derive(Debug, Clone)]
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
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self { cross_pool_lr: 0.15, max_hops: 2, hop_decay: 0.85 }
    }
}

pub struct Fabric {
    pub config: FabricConfig,
    pools:      AHashMap<PoolId, Arc<RwLock<Pool>>>,
    current:    Moment,
    tick:       u64,
}

impl Fabric {
    pub fn new(config: FabricConfig) -> Self {
        Self {
            config,
            pools:   AHashMap::new(),
            current: Moment::new(0),
            tick:    0,
        }
    }

    pub fn current_tick(&self) -> u64 { self.tick }

    /// Read the current (un-closed) tick's moment.  Used by [`crate::Brain`]
    /// to capture the multi-pool firing fingerprint for binding-concept
    /// emergence before `advance_tick` clears it.
    pub fn current_moment(&self) -> &Moment { &self.current }

    pub fn register_pool(&mut self, pool: Pool) -> PoolId {
        let id = pool.id();
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
        // Close the prior moment: any pair of neurons in different pools
        // that both fired this tick gets a cross-pool axon terminal
        // grown / strengthened.  Same-pool co-firing isn't wired here
        // because within-pool concept emergence (the Pool's job) is the
        // proper mechanism for that — wiring atom→atom here would just
        // duplicate the atom-pair Hebbian effect.
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
                    for &na in &fired_a {
                        if let Some(neuron) = pa.get_mut(na) {
                            for &nb in &fired_b {
                                neuron.reinforce_terminal(
                                    NeuronRef::new(pid_b, nb),
                                    lr, self.tick, max_w,
                                );
                            }
                        }
                    }
                }
                {
                    let mut pb = pool_b.write();
                    let max_w = pb.config.max_weight;
                    for &nb in &fired_b {
                        if let Some(neuron) = pb.get_mut(nb) {
                            for &na in &fired_a {
                                neuron.reinforce_terminal(
                                    NeuronRef::new(pid_a, na),
                                    lr, self.tick, max_w,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Per-pool housekeeping: decay every terminal, prune sub-floor.
        for (_, pool) in self.pools.iter() {
            pool.write().tick_housekeeping();
        }

        self.tick += 1;
        self.current = Moment::new(self.tick);
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
}
