//! Continuous cost-aware tier orchestrator.
//!
//! User vision: the brain must decide — online, during training AND
//! prediction — whether holding a neuron in RAM is cheaper than the
//! cost of serializing it now plus the probability-weighted cost of
//! paging it back in later.  No static time threshold, no manual mode
//! switch, no separate "training" vs "predicting" tier policy.  Just
//! a continuous evaluator that runs every tick and keeps the working
//! set tight against a target RAM budget.
//!
//! This module ships only the POLICY.  The mechanical evict / page-in
//! primitives already exist on [`Pool`]:
//!   - [`Pool::evict_neuron`]  — serializes terminals to cold tier,
//!     `shrink_to_fit`s the RAM-resident slot.
//!   - [`Pool::page_in_neuron`] — restores terminals from cold tier.
//! The orchestrator just picks which neurons to flip.
//!
//! ## Cost model
//!
//! For each candidate neuron we compute an `EvictScore`.  Larger score
//! → eviction is more profitable.  Score has three additive components
//! plus one multiplicative gate:
//!
//! ```text
//! score = w_terminals * ln(1 + terminal_count)         // bigger → more savings
//!       + w_staleness * (current_tick - last_fired_tick) / decay_horizon
//!       + w_inverse_salience * (1 / (eps + salience_ema))
//!       - w_pinned * pinned_indicator
//! ```
//!
//! Eviction fires when `score > threshold * pressure_factor`, where
//! `pressure_factor` is `(actual_terminals / target_terminals)` clamped
//! to `[0.25, 4.0]`.  When the brain is far under budget, pressure_factor
//! is small and the threshold is high → almost nothing is evicted.  When
//! the brain is over budget, pressure_factor shrinks the threshold so
//! more candidates qualify, and the orchestrator drains aggressively.
//!
//! Atoms and binding-pool neurons are excluded — same as the manual
//! [`Brain::run_eviction_pass`] policy.

use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;

use crate::neuron::PoolId;

/// Tunable knobs.  All overridable via env so production can iterate
/// without recompilation.  Defaults chosen so a fresh brain doesn't
/// thrash before its working set has settled.
#[derive(Debug, Clone, Copy)]
pub struct OrchestratorParams {
    /// Run the orchestrator every N ticks.  1 = every tick.  Higher
    /// values trade responsiveness for less per-tick CPU.
    pub run_every_n_ticks: u64,
    /// Per-pool budget for the round-robin scan window.  The
    /// orchestrator looks at this many neuron slots per pass, not the
    /// whole pool.  Lower keeps per-tick cost bounded on large brains.
    pub scan_budget:       usize,
    /// Per-pool cap on evictions per pass.  Keeps a single pass from
    /// holding the brain lock long.
    pub max_evict_per_pass: usize,
    /// Target total terminals per pool — the implicit RAM budget.
    /// When `total_terminals` is above target, pressure_factor < 1
    /// and the eviction threshold contracts.  Set to 0 to disable
    /// pressure-aware behavior (uses static threshold).
    pub target_terminals_per_pool: usize,
    /// Static cost-benefit threshold.  See module-level math.  Tune
    /// up to evict less, down to evict more.
    pub evict_threshold:   f32,
    /// Co-efficients in the scoring polynomial.
    pub w_terminals:        f32,
    pub w_staleness:        f32,
    pub w_inverse_salience: f32,
    pub w_pinned:           f32,
    /// Staleness numerator — divides (tick - last_fired_tick).  Big
    /// horizon → staleness influence ramps slowly.  Small → ramps fast.
    pub decay_horizon_ticks: u64,
    /// Cap on per-cycle salience_ema considered for the inverse term;
    /// guards against division blowups for never-fired neurons.
    pub salience_eps:       f32,
    /// Proactive page-in: when a fired neuron's outgoing terminals
    /// point at an evicted target, page the target back in if its
    /// salience_ema (saved across eviction) exceeds this floor.
    /// Default 0.0 — no proactive page-in.
    pub page_in_salience_floor: f32,
    /// Hard cap on per-pass page-ins so a hot tick can't blow the
    /// working set back out.
    pub max_page_in_per_pass: usize,
    /// Newborn protection — a concept whose `current_tick - born_tick`
    /// is below this is never evicted, regardless of score.  Without
    /// this the orchestrator throws away brand-new concepts before
    /// they've had a chance to fire a second time and prove their
    /// salience.
    pub min_age_ticks: u64,
    /// Politeness floor: minimum SYSTEM available RAM (MB) the node must
    /// leave for everything else on the machine.  When available system
    /// memory drops toward this floor, the system pressure factor
    /// overrides the per-pool terminal budget and forces aggressive
    /// eviction — the brain must never be the reason the computer locks
    /// up.  0 disables the system-memory check.
    pub min_system_available_mb: u64,
}

impl Default for OrchestratorParams {
    fn default() -> Self {
        // Tuned so a fresh brain doesn't evict anything until either
        // (a) the pool is meaningfully over `target_terminals_per_pool`
        // (pressure_factor < 1 contracts the effective threshold), or
        // (b) a neuron is genuinely stale (last_fired_tick >> tick AND
        // salience_ema near zero).  See the under-budget integration
        // test for the boundary behavior.
        Self {
            run_every_n_ticks: 1,
            scan_budget: 1024,
            max_evict_per_pass: 256,
            target_terminals_per_pool: 1_000_000,
            evict_threshold: 5.0,
            w_terminals: 0.5,
            w_staleness: 2.0,
            w_inverse_salience: 1.0,
            // Pin penalty must dominate even extreme staleness; with
            // w_staleness=2 and a 100k-tick horizon the staleness term
            // alone can reach ~200, so the pin penalty has to be
            // bigger than that ceiling to actually protect a pinned
            // neuron from eviction.
            w_pinned: 10_000.0,
            decay_horizon_ticks: 1_000,
            salience_eps: 0.01,
            page_in_salience_floor: 0.0,
            // Demand paging is ON by default — an evicted neuron that
            // activation is about to reach gets its terminals back from
            // the cold tier before propagation walks it.  0 previously
            // meant the page-in half of the cost model never ran at all.
            max_page_in_per_pass: 128,
            min_age_ticks: 1_000,
            min_system_available_mb: 4_096,
        }
    }
}

impl OrchestratorParams {
    /// Default for `Fabric::new` and snapshot restore.
    ///
    /// The orchestrator runs continuously (the user's "human brain
    /// always doing online tier management" vision) — so on the node
    /// binary it should default ON with conservative params.  But the
    /// manual `Brain::run_eviction_pass` test (and any caller that
    /// relies on neurons NOT silently evicting during bare ticks)
    /// needs a clean off-switch.
    ///
    /// Resolution:
    ///   - `W1Z4RD_TIER_ORCHESTRATOR=off` → explicitly disabled.
    ///   - Any other value, including unset → ENABLED with
    ///     conservative defaults from [`Self::from_env`].
    ///
    /// Tests that need the orchestrator OFF set
    /// `W1Z4RD_TIER_ORCHESTRATOR=off` in the env or call
    /// `set_tier_orchestrator_params(OrchestratorParams::disabled())`
    /// explicitly during setup.
    pub fn from_env_or_disabled() -> Self {
        let off = std::env::var("W1Z4RD_TIER_ORCHESTRATOR")
            .map(|v| matches!(v.to_lowercase().as_str(),
                              "0" | "off" | "false" | "no" | "disabled"))
            .unwrap_or(false);
        if off { Self::disabled() } else { Self::from_env() }
    }

    /// Build a params bundle from env vars, falling back to defaults.
    /// Recognized vars (all optional):
    ///   W1Z4RD_TIER_RUN_EVERY_N
    ///   W1Z4RD_TIER_SCAN_BUDGET
    ///   W1Z4RD_TIER_MAX_EVICT
    ///   W1Z4RD_TIER_TARGET_TERMS
    ///   W1Z4RD_TIER_THRESHOLD
    ///   W1Z4RD_TIER_W_TERM, W1Z4RD_TIER_W_STALE, W1Z4RD_TIER_W_INVSAL, W1Z4RD_TIER_W_PIN
    ///   W1Z4RD_TIER_DECAY_HORIZON
    ///   W1Z4RD_TIER_PAGEIN_FLOOR
    ///   W1Z4RD_TIER_MAX_PAGEIN
    pub fn from_env() -> Self {
        let mut p = Self::default();
        fn u(name: &str, dst: &mut u64) {
            if let Ok(v) = std::env::var(name) { if let Ok(x) = v.parse() { *dst = x; } }
        }
        fn us(name: &str, dst: &mut usize) {
            if let Ok(v) = std::env::var(name) { if let Ok(x) = v.parse() { *dst = x; } }
        }
        fn f(name: &str, dst: &mut f32) {
            if let Ok(v) = std::env::var(name) { if let Ok(x) = v.parse() { *dst = x; } }
        }
        u ("W1Z4RD_TIER_RUN_EVERY_N",  &mut p.run_every_n_ticks);
        us("W1Z4RD_TIER_SCAN_BUDGET",  &mut p.scan_budget);
        us("W1Z4RD_TIER_MAX_EVICT",    &mut p.max_evict_per_pass);
        us("W1Z4RD_TIER_TARGET_TERMS", &mut p.target_terminals_per_pool);
        f ("W1Z4RD_TIER_THRESHOLD",    &mut p.evict_threshold);
        f ("W1Z4RD_TIER_W_TERM",       &mut p.w_terminals);
        f ("W1Z4RD_TIER_W_STALE",      &mut p.w_staleness);
        f ("W1Z4RD_TIER_W_INVSAL",     &mut p.w_inverse_salience);
        f ("W1Z4RD_TIER_W_PIN",        &mut p.w_pinned);
        u ("W1Z4RD_TIER_DECAY_HORIZON", &mut p.decay_horizon_ticks);
        f ("W1Z4RD_TIER_PAGEIN_FLOOR", &mut p.page_in_salience_floor);
        us("W1Z4RD_TIER_MAX_PAGEIN",   &mut p.max_page_in_per_pass);
        u ("W1Z4RD_TIER_MIN_AGE_TICKS", &mut p.min_age_ticks);
        u ("W1Z4RD_TIER_MIN_SYS_AVAIL_MB", &mut p.min_system_available_mb);
        p
    }

    /// "Run only when explicitly invoked" mode.  Used by tests and by
    /// callers that want the manual eviction-pass API.
    pub fn disabled() -> Self {
        Self {
            run_every_n_ticks: u64::MAX,
            ..Self::default()
        }
    }
}

/// Cumulative counters.  All AtomicU64 so /tier_orchestrator_stats
/// can read them without holding the brain lock.
#[derive(Debug, Default)]
pub struct OrchestratorStats {
    pub passes:            AtomicU64,
    pub neurons_scanned:   AtomicU64,
    pub neurons_evicted:   AtomicU64,
    pub neurons_paged_in:  AtomicU64,
    pub evict_errors:      AtomicU64,
    pub page_in_errors:    AtomicU64,
    pub last_pressure_x1k: AtomicU64,  // pressure_factor * 1000, last pass
    pub total_ns:          AtomicU64,
}

impl OrchestratorStats {
    pub fn snapshot(&self) -> OrchestratorStatsSnapshot {
        OrchestratorStatsSnapshot {
            passes:            self.passes.load(Ordering::Relaxed),
            neurons_scanned:   self.neurons_scanned.load(Ordering::Relaxed),
            neurons_evicted:   self.neurons_evicted.load(Ordering::Relaxed),
            neurons_paged_in:  self.neurons_paged_in.load(Ordering::Relaxed),
            evict_errors:      self.evict_errors.load(Ordering::Relaxed),
            page_in_errors:    self.page_in_errors.load(Ordering::Relaxed),
            last_pressure:     (self.last_pressure_x1k.load(Ordering::Relaxed) as f32) / 1000.0,
            total_ns:          self.total_ns.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct OrchestratorStatsSnapshot {
    pub passes:           u64,
    pub neurons_scanned:  u64,
    pub neurons_evicted:  u64,
    pub neurons_paged_in: u64,
    pub evict_errors:     u64,
    pub page_in_errors:   u64,
    pub last_pressure:    f32,
    pub total_ns:         u64,
}

/// Per-pool round-robin scan cursor.  Persisting it across passes makes
/// the orchestrator walk the whole neuron table over time even though
/// each pass only touches `scan_budget` entries.
#[derive(Debug, Default)]
pub struct TierOrchestrator {
    cursors:   AHashMap<PoolId, usize>,
}

impl TierOrchestrator {
    pub fn new() -> Self {
        Self { cursors: AHashMap::new() }
    }

    /// Returns the next cursor position for `pool` after scanning
    /// `consumed` slots starting at the stored cursor.  Wraps mod len.
    pub fn advance_cursor(&mut self, pool: PoolId, consumed: usize, len: usize) -> usize {
        if len == 0 { return 0; }
        let cur = *self.cursors.get(&pool).unwrap_or(&0);
        let next = (cur + consumed) % len;
        self.cursors.insert(pool, next);
        cur
    }

    /// Compute the eviction score for a single candidate.  Pure; no
    /// I/O.  Used by both the orchestrator and tests.
    pub fn score(
        params: &OrchestratorParams,
        terminal_count: usize,
        last_fired_tick: u64,
        current_tick: u64,
        salience_ema: f32,
        pinned: bool,
    ) -> f32 {
        let term = params.w_terminals * (1.0 + terminal_count as f32).ln();
        let stale_dt = current_tick.saturating_sub(last_fired_tick) as f32;
        let stale = params.w_staleness * (stale_dt / params.decay_horizon_ticks.max(1) as f32);
        let inv_sal = params.w_inverse_salience
            * (1.0 / (params.salience_eps.max(1e-6) + salience_ema.max(0.0)));
        let pin = if pinned { params.w_pinned } else { 0.0 };
        term + stale + inv_sal - pin
    }

    /// Pressure factor.  > 1 means we're under target (evict less);
    /// < 1 means we're over target (evict more aggressively).
    pub fn pressure_factor(actual: usize, target: usize) -> f32 {
        if target == 0 { return 1.0; }
        let raw = (actual as f32) / (target as f32);
        // Higher raw means more terminals than budget → smaller
        // factor → smaller threshold → more eviction.
        let inv = 1.0 / raw.max(0.001);
        inv.clamp(0.25, 4.0)
    }

    /// System-memory pressure factor — the politeness gate.  Pure math
    /// so it's testable; callers pass the current available system RAM.
    ///
    /// Mapping (min = `min_system_available_mb`):
    ///   available >= 2*min → 4.0  (machine has headroom; no extra effort)
    ///   available == min   → 1.0  (at the floor; normal aggression)
    ///   available <= min/2 → 0.25 (starving the machine; max aggression)
    /// Linear in between.  The caller combines this with the terminal
    /// budget via `min()`, so whichever signal is more urgent wins.
    pub fn system_pressure_factor(available_mb: u64, min_mb: u64) -> f32 {
        if min_mb == 0 { return 4.0; }
        let a = available_mb as f32;
        let m = min_mb as f32;
        if a >= 2.0 * m {
            return 4.0;
        }
        if a >= m {
            // [m, 2m) → [1.0, 4.0)
            return 1.0 + 3.0 * ((a - m) / m);
        }
        // [m/2, m) → [0.25, 1.0); below m/2 clamps to 0.25
        (0.25 + 0.75 * ((a - m / 2.0) / (m / 2.0)).max(0.0)).min(1.0)
    }

    /// Current available system memory in MB.  Uses a cached
    /// [`sysinfo::System`] refreshed at most every ~2 s so calling this
    /// every tick costs a mutex lock + timestamp compare, not a syscall
    /// storm.  Returns `u64::MAX` if the probe fails (never force
    /// eviction on bad data).
    pub fn system_available_mb() -> u64 {
        use std::sync::Mutex;
        use std::time::{Duration, Instant};
        static PROBE: Mutex<Option<(sysinfo::System, Instant)>> = Mutex::new(None);
        let Ok(mut guard) = PROBE.lock() else { return u64::MAX };
        let now = Instant::now();
        let needs_refresh = match &*guard {
            Some((_, at)) => now.duration_since(*at) > Duration::from_secs(2),
            None => true,
        };
        if needs_refresh {
            let mut sys = match guard.take() {
                Some((sys, _)) => sys,
                None => sysinfo::System::new(),
            };
            sys.refresh_memory();
            *guard = Some((sys, now));
        }
        match &*guard {
            Some((sys, _)) => sys.available_memory() / (1024 * 1024),
            None => u64::MAX,
        }
    }

    /// Decide whether a candidate evicts.  Pure; used by tests.
    pub fn should_evict(
        params: &OrchestratorParams,
        score: f32,
        pressure_factor: f32,
    ) -> bool {
        score > params.evict_threshold * pressure_factor
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_factor_in_range() {
        // Way over budget → small factor (more eviction)
        assert!(TierOrchestrator::pressure_factor(10_000_000, 1_000_000) < 0.5);
        // Way under budget → large factor (less eviction)
        assert!(TierOrchestrator::pressure_factor(100, 1_000_000) > 1.5);
        // Exactly at budget → ~1
        let f = TierOrchestrator::pressure_factor(1_000_000, 1_000_000);
        assert!((f - 1.0).abs() < 0.01);
    }

    #[test]
    fn system_pressure_factor_politeness_curve() {
        // Plenty of headroom (>= 2x floor) → no extra effort
        assert_eq!(TierOrchestrator::system_pressure_factor(16_384, 4_096), 4.0);
        assert_eq!(TierOrchestrator::system_pressure_factor(8_192, 4_096), 4.0);
        // At the floor → normal aggression
        let at_floor = TierOrchestrator::system_pressure_factor(4_096, 4_096);
        assert!((at_floor - 1.0).abs() < 0.01);
        // Halfway between floor and 2x floor → between 1 and 4
        let mid = TierOrchestrator::system_pressure_factor(6_144, 4_096);
        assert!(mid > 1.0 && mid < 4.0);
        // Below half the floor → maximum aggression
        assert_eq!(TierOrchestrator::system_pressure_factor(1_000, 4_096), 0.25);
        assert_eq!(TierOrchestrator::system_pressure_factor(0, 4_096), 0.25);
        // Between half-floor and floor → between 0.25 and 1
        let squeezed = TierOrchestrator::system_pressure_factor(3_072, 4_096);
        assert!(squeezed > 0.25 && squeezed < 1.0);
        // Disabled check never forces eviction
        assert_eq!(TierOrchestrator::system_pressure_factor(0, 0), 4.0);
    }

    #[test]
    fn score_grows_with_terminal_count_and_staleness() {
        let p = OrchestratorParams::default();
        let small = TierOrchestrator::score(&p, 10,    100, 200,   0.5, false);
        let big   = TierOrchestrator::score(&p, 10_000, 100, 200,   0.5, false);
        assert!(big > small, "more terminals → higher evict score");
        let fresh = TierOrchestrator::score(&p, 1000, 200,  200,  0.5, false);
        let stale = TierOrchestrator::score(&p, 1000, 100,  10_000, 0.5, false);
        assert!(stale > fresh, "more stale → higher evict score");
    }

    #[test]
    fn pinned_overrides_high_score() {
        let p = OrchestratorParams::default();
        let pinned = TierOrchestrator::score(&p, 10_000, 0, 100_000, 0.0, true);
        // Pinned penalty of 100 should make even the most evictable
        // candidate net negative — i.e., never crosses threshold > 1.
        assert!(pinned < 0.0, "pinned neuron must net negative");
    }

    #[test]
    fn cursor_round_robin_wraps() {
        let mut o = TierOrchestrator::new();
        // Scan 3 slots out of 10 → next cursor = 3
        let c0 = o.advance_cursor(1, 3, 10);
        assert_eq!(c0, 0);
        let c1 = o.advance_cursor(1, 3, 10);
        assert_eq!(c1, 3);
        // Walk past end → wraps mod 10
        o.advance_cursor(1, 5, 10); // cursor now 11 % 10 = 1
        let c2 = o.advance_cursor(1, 0, 10);
        assert_eq!(c2, 1);
    }

    #[test]
    fn disabled_params_never_fire() {
        let p = OrchestratorParams::disabled();
        assert_eq!(p.run_every_n_ticks, u64::MAX);
    }
}
