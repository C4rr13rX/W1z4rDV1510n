//! Polite-cooperation resource gate.
//!
//! Iterative consumers (motif promotion cascade, future search loops,
//! background sweepers) ask a [`ResourceGate`] before each iteration.
//! The gate combines three signals:
//!
//!   1. **Consumer progress** — if the consumer reports it didn't
//!      find anything new last iteration, there's nothing more to
//!      find at this scale right now.  Returning `false` lets the
//!      consumer stop without a fixed cap; the next call gets a
//!      fresh chance with new data.  No "depth_limit = 5" hard
//!      stops anywhere.
//!
//!   2. **System-wide memory pressure** — when the *machine's* free
//!      memory drops below the configured floor (default 10 % of
//!      total), the gate returns `false` so we yield to whatever
//!      other process needs RAM.  This is the polite-cohabitation
//!      signal: we don't run flat-out when the user is doing other
//!      work on the same box.
//!
//!   3. **System-wide CPU pressure** — when overall CPU utilisation
//!      exceeds the configured ceiling (default 90 %), back off so
//!      foreground tasks get headroom.
//!
//! Defaults are derived from [`HardwareProfile`] so the same gate
//! behaves appropriately on a constrained laptop versus a 256 GB
//! workstation.  Operators can override via [`GateConfig`].
//!
//! The `ResourceGate` trait is the polite contract: any iterative
//! consumer can implement against it; multiple consumers can share
//! one gate instance via `Arc<dyn ResourceGate>` so they see
//! consistent system-pressure samples.

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

use crate::hardware::HardwareProfile;


/// Per-consumer iteration context.  The consumer mutates this each
/// pass to report whether it made progress; the gate reads it to
/// decide whether more iterations are warranted.
#[derive(Debug, Clone)]
pub struct WorkContext {
    pub label: &'static str,
    pub iterations: usize,
    pub progressed_last_iter: bool,
}

impl WorkContext {
    pub fn new(label: &'static str) -> Self {
        // First call should run — start "as if" the previous iteration
        // made progress so `should_continue` lets the consumer take
        // its first pass.
        Self { label, iterations: 0, progressed_last_iter: true }
    }

    pub fn note_iteration(&mut self, progressed: bool) {
        self.iterations += 1;
        self.progressed_last_iter = progressed;
    }
}


/// Polite back-off contract.  Iterative consumers consult an
/// implementation before each round and stop when it returns false.
pub trait ResourceGate: Send + Sync {
    fn should_continue(&self, ctx: &WorkContext) -> bool;
}


/// Tunable thresholds for the system-pressure-aware gate.  Defaults
/// are derived from [`HardwareProfile`] in [`GateConfig::for_profile`].
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Minimum fraction (0..1) of system memory that must remain free
    /// before the gate hands out another iteration.  Below this we
    /// yield so other processes on the host get RAM.  The single most
    /// important knob for "polite cohabitation."
    pub min_system_free_fraction: f32,
    /// Maximum system-wide CPU utilisation (0..1) we'll accept before
    /// backing off.  Default 0.90 — leaves headroom for foreground.
    pub max_system_cpu_fraction: f32,
    /// Optional hard cap on this process's RSS (MB).  `u64::MAX`
    /// disables — the system-free check is usually a better signal
    /// since it captures cohabitation rather than self-bloat.
    pub mem_hard_mb: u64,
    /// Sampling interval — how often the gate refreshes its view of
    /// system memory and CPU.  Lower is more responsive; higher is
    /// cheaper.  500 ms is the default.
    pub sample_interval_ms: u64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self::for_profile(&HardwareProfile::detect())
    }
}

impl GateConfig {
    /// Hardware-aware defaults.  A constrained machine (the laptop
    /// path in HardwareProfile) gets a higher free-memory floor and a
    /// lower CPU ceiling so we share the box more aggressively.  A
    /// powerful workstation can run hotter without disturbing
    /// foreground work.
    pub fn for_profile(profile: &HardwareProfile) -> Self {
        if profile.is_constrained() {
            Self {
                min_system_free_fraction: 0.20,  // keep 20 % free for others
                max_system_cpu_fraction:  0.75,
                mem_hard_mb:              u64::MAX,
                sample_interval_ms:       500,
            }
        } else {
            Self {
                min_system_free_fraction: 0.10,  // 10 % free is enough on big iron
                max_system_cpu_fraction:  0.90,
                mem_hard_mb:              u64::MAX,
                sample_interval_ms:       500,
            }
        }
    }
}


/// Default gate — checks consumer progress + sampled system memory + CPU.
///
/// Polite to other processes on the host: when their demand pushes
/// system free memory below `min_system_free_fraction` or system CPU
/// above `max_system_cpu_fraction`, we yield until the next sample
/// shows pressure has eased.
pub struct SystemGate {
    state:  RwLock<GateState>,
    config: GateConfig,
}

#[derive(Default)]
struct GateState {
    last_sample:    Option<Instant>,
    sys:            Option<System>,
    /// Fraction of total memory currently AVAILABLE on the host.
    free_fraction:  f32,
    /// Mean across all logical CPUs at last sample.  0..1.
    cpu_fraction:   f32,
    /// Our process RSS (MB) — only checked when mem_hard_mb is finite.
    rss_mb:         u64,
}

impl SystemGate {
    pub fn new(config: GateConfig) -> Self {
        Self { state: RwLock::new(GateState::default()), config }
    }

    /// Refresh system memory + CPU at most once per
    /// `sample_interval_ms`.  Costs ~µs per call when the cache is
    /// fresh; the actual `refresh_*` invocation is the expensive bit.
    fn maybe_sample(&self) {
        let mut s = self.state.write();
        let now = Instant::now();
        let due = match s.last_sample {
            None    => true,
            Some(t) => now.duration_since(t).as_millis() as u64
                          >= self.config.sample_interval_ms,
        };
        if !due { return; }
        if s.sys.is_none() {
            s.sys = Some(System::new_with_specifics(
                RefreshKind::new()
                    .with_memory(MemoryRefreshKind::everything())
                    .with_cpu(CpuRefreshKind::everything()),
            ));
        }
        // Collect samples into locals first; the `sys` mutable borrow
        // and the `s.<field> =` writes can't overlap (E0499), so we
        // commit the readings after the sys borrow ends.  Skip update
        // entirely if sys somehow isn't initialised — leaves prior
        // sample values in place rather than writing misleading zeros.
        let track_rss = self.config.mem_hard_mb != u64::MAX;
        let sample = s.sys.as_mut().map(|sys| {
            sys.refresh_memory();
            sys.refresh_cpu();
            let total = sys.total_memory().max(1) as f32;
            let avail = sys.available_memory() as f32;
            let free_frac = avail / total;
            let cpus = sys.cpus();
            let cpu_frac = if cpus.is_empty() { 0.0 } else {
                let sum: f32 = cpus.iter().map(|c| c.cpu_usage()).sum();
                (sum / cpus.len() as f32) / 100.0
            };
            let rss_mb_opt = if track_rss {
                let pid = sysinfo::Pid::from_u32(std::process::id());
                sys.refresh_processes();
                Some(sys.process(pid).map(|p| p.memory()).unwrap_or(0)
                       / (1024 * 1024))
            } else {
                None
            };
            (free_frac, cpu_frac, rss_mb_opt)
        });
        if let Some((free_frac, cpu_frac, rss_mb_opt)) = sample {
            s.free_fraction = free_frac;
            s.cpu_fraction  = cpu_frac;
            if let Some(rss) = rss_mb_opt {
                s.rss_mb = rss;
            }
        }
        s.last_sample = Some(now);
    }
}

impl ResourceGate for SystemGate {
    fn should_continue(&self, ctx: &WorkContext) -> bool {
        // Consumer-side termination: the prior iteration found nothing
        // new, so there's nothing more to extract at this scale right
        // now.  Returning false lets the consumer release its work
        // budget; the next call (with fresh data) gets a clean restart.
        if ctx.iterations > 0 && !ctx.progressed_last_iter {
            return false;
        }
        self.maybe_sample();
        let s = self.state.read();
        // System-wide memory pressure — others need RAM, yield.
        if s.free_fraction < self.config.min_system_free_fraction {
            return false;
        }
        // System-wide CPU pressure — others need cycles, yield.
        if s.cpu_fraction > self.config.max_system_cpu_fraction {
            return false;
        }
        // Optional self-RSS hard cap — only relevant when configured.
        if self.config.mem_hard_mb != u64::MAX && s.rss_mb > self.config.mem_hard_mb {
            return false;
        }
        true
    }
}


/// Always permits the next iteration unless the consumer reports no
/// progress.  Use in tests or when the consumer is intrinsically bounded.
pub struct AlwaysOnGate;

impl ResourceGate for AlwaysOnGate {
    fn should_continue(&self, ctx: &WorkContext) -> bool {
        !(ctx.iterations > 0 && !ctx.progressed_last_iter)
    }
}


/// Convenience: the production default gate — `SystemGate` with
/// `HardwareProfile`-derived thresholds.  Returns an `Arc` so multiple
/// consumers can share one instance and thereby see consistent
/// system-pressure samples.
pub fn default_gate() -> Arc<dyn ResourceGate> {
    Arc::new(SystemGate::new(GateConfig::default()))
}
