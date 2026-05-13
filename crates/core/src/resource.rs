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


// ── User-input idle detection ───────────────────────────────────────────────
//
// On Windows we use GetLastInputInfo (Win32 API) to measure how long ago
// the system saw any keyboard / mouse input.  This is cheap (a single
// syscall, ~µs) and is sampled on EVERY should_continue call so the
// gate reacts to the operator coming back to the machine within a
// single poll cycle of whatever iterative consumer is running.
//
// The promise to the operator: "I move my mouse, maybe a second freeze
// is allowed, then resources release."  With poll cadences in the
// 100ms-1s range and the gate flipping to "back off" on any input
// activity, that's exactly the behaviour we get.

#[cfg(target_os = "windows")]
fn system_idle_seconds() -> f32 {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetLastInputInfo, LASTINPUTINFO,
    };
    use windows_sys::Win32::System::SystemInformation::GetTickCount;
    unsafe {
        let mut info: LASTINPUTINFO = std::mem::zeroed();
        info.cbSize = std::mem::size_of::<LASTINPUTINFO>() as u32;
        if GetLastInputInfo(&mut info) == 0 {
            return f32::MAX;  // syscall failed — treat as idle
        }
        let now_ticks  = GetTickCount();
        // GetTickCount wraps every ~49 days; tolerate that by treating
        // wrap-around as fresh activity.
        let elapsed_ms = now_ticks.wrapping_sub(info.dwTime);
        (elapsed_ms as f32) / 1000.0
    }
}

#[cfg(not(target_os = "windows"))]
fn system_idle_seconds() -> f32 {
    // Non-Windows: we don't have a portable cheap API for system-wide
    // input idle.  Returning a large value disables the user-input
    // back-off entirely (consumers run as before).  Add an
    // X11/Wayland/AppKit implementation here if you need it on those
    // platforms.
    f32::MAX
}


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
    /// **User-activity back-off threshold.**  When the OS reports
    /// that keyboard/mouse activity happened more recently than this
    /// many seconds ago, the gate returns false from every
    /// should_continue call.  This is the lever that frees the
    /// machine the instant the operator comes back to it: with
    /// `user_activity_idle_threshold_secs = 1.5`, any input within
    /// the last 1.5 seconds halts every iterative consumer
    /// immediately, and they resume the moment the operator stops
    /// touching the keyboard / mouse for that long.
    ///
    /// Set to `f32::MAX` to disable user-activity back-off entirely
    /// (training-server-only deployments where there is no
    /// interactive user to share with).
    pub user_activity_idle_threshold_secs: f32,
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
                // Constrained machine = laptop, almost certainly
                // interactive.  React within 1.5s of user touching
                // anything.
                user_activity_idle_threshold_secs: 1.5,
            }
        } else {
            Self {
                min_system_free_fraction: 0.10,  // 10 % free is enough on big iron
                max_system_cpu_fraction:  0.90,
                mem_hard_mb:              u64::MAX,
                sample_interval_ms:       500,
                // Workstation: same default user-activity threshold —
                // the user has explicitly asked for the brain to back
                // off "instantly" when they come back to the machine.
                user_activity_idle_threshold_secs: 1.5,
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
        // User-activity back-off — checked first, before any
        // expensive system-pressure sampling, because this is the
        // path that has to fire fastest.  Cost: one syscall (~µs).
        // The operator's contract: any mouse / keyboard activity in
        // the last `user_activity_idle_threshold_secs` halts every
        // iterative consumer immediately so the machine snaps back
        // to interactive responsiveness.
        if self.config.user_activity_idle_threshold_secs != f32::MAX {
            let idle = system_idle_seconds();
            if idle < self.config.user_activity_idle_threshold_secs {
                return false;
            }
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
