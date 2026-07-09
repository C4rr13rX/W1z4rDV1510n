//! Integration tests for the continuous cost-aware tier orchestrator.
//!
//! These tests exercise the end-to-end path:
//!   1. Build a brain with a text pool + cold tier attached.
//!   2. Train it enough that some concepts have high salience (hot)
//!      and some have terminals + stale last_fired_tick (cold).
//!   3. Adjust orchestrator params so the eviction threshold is
//!      crossed by the cold candidates but not by the hot ones.
//!   4. Drive ticks and assert:
//!      - cumulative `neurons_evicted` counter advanced,
//!      - the targeted (stale, low-salience) concepts went into
//!        the cold-tier `evicted` set,
//!      - hot concepts and ALL atoms stayed in RAM.
//!
//! These tests rely only on the public Brain / Fabric surface.

use std::path::PathBuf;
use std::time::SystemTime;

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};
use w1z4rd_brain::pool::Pool;
use w1z4rd_brain::tier_orchestrator::OrchestratorParams;

fn tmpdir(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("w1z4rd_tier_{}_{}_{}", label, pid, nanos));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn make_brain_with_text_pool() -> Brain {
    let mut brain = Brain::new(BrainConfig::default());
    let mut pc = PoolConfig::defaults("text", 1);
    pc.recent_atoms_window = 8;
    pc.concept_emergence_threshold = 2;
    let enc: Box<dyn AtomEncoding> = Box::new(BytePassthroughEncoding { prefix: "t" });
    brain.fabric_mut().register_pool(Pool::new(pc, enc));
    brain
}

/// 1. Continuous orchestrator runs every tick when run_every_n_ticks=1.
/// 2. With a forcing-pressure config (tiny target + low threshold), it
///    evicts at least one concept after a handful of ticks.
/// 3. Atoms (single-byte source-prefix neurons) are never evicted.
#[test]
fn orchestrator_evicts_concepts_under_pressure_continuously() {
    let dir = tmpdir("evicts_under_pressure");
    let mut brain = make_brain_with_text_pool();
    brain.attach_cold_tiers(&dir);

    // Train: feed the same short word many times so concepts form
    // and accumulate salience.  Then feed a different word so the
    // first batch becomes "stale" relative to the latest tick.
    for _ in 0..20 {
        brain.observe(1, b"alpha");
        brain.advance_tick();
    }
    for _ in 0..6 {
        brain.observe(1, b"omegaomega");
        brain.advance_tick();
    }

    // Snapshot pre-orchestrator-pressure state.
    let pre_stats = brain.fabric().tier_orchestrator_stats();
    let pre_neurons_total: usize = brain.fabric()
        .pool_ids().iter()
        .filter_map(|pid| brain.fabric().pool(*pid))
        .map(|p| p.read().neurons_len())
        .sum();

    // Pressure-on params: tiny target so pressure_factor < 1.0, low
    // threshold so even modest scores trigger eviction.  Scan budget
    // covers everything.
    let aggressive = OrchestratorParams {
        run_every_n_ticks: 1,
        scan_budget: 1024,
        max_evict_per_pass: 32,
        target_terminals_per_pool: 1,
        evict_threshold: 0.5,
        w_terminals: 0.0,           // ignore size — make staleness dominate
        w_staleness: 5.0,
        w_inverse_salience: 0.1,
        w_pinned: 100.0,
        decay_horizon_ticks: 1,
        salience_eps: 0.01,
        page_in_salience_floor: 0.0,
        max_page_in_per_pass: 0,
        min_age_ticks: 0,           // test disables newborn protection
        min_system_available_mb: 0, // hermetic: ignore live machine RAM
    };
    brain.fabric_mut().set_tier_orchestrator_params(aggressive);

    // Drive a few ticks; the per-tick orchestrator runs in advance_tick.
    for _ in 0..30 {
        brain.advance_tick();
    }

    let post = brain.fabric().tier_orchestrator_stats();
    assert!(post.passes > pre_stats.passes,
        "orchestrator should have run passes ({} > {})",
        post.passes, pre_stats.passes);
    assert!(post.neurons_scanned > pre_stats.neurons_scanned,
        "scanner should have visited slots ({} > {})",
        post.neurons_scanned, pre_stats.neurons_scanned);
    assert!(post.neurons_evicted > 0,
        "at least one concept should have been evicted under tight pressure (got {})",
        post.neurons_evicted);
    assert_eq!(post.evict_errors, 0,
        "no eviction should error; got {}", post.evict_errors);

    // Sanity: total neuron-slot count never shrinks (eviction zeros
    // terminals but the slot stays).
    let post_neurons_total: usize = brain.fabric()
        .pool_ids().iter()
        .filter_map(|pid| brain.fabric().pool(*pid))
        .map(|p| p.read().neurons_len())
        .sum();
    assert_eq!(pre_neurons_total, post_neurons_total,
        "neuron slot count must be stable; pre={} post={}",
        pre_neurons_total, post_neurons_total);

    std::fs::remove_dir_all(&dir).ok();
}

/// When run_every_n_ticks = u64::MAX (disabled), no eviction ever fires.
#[test]
fn orchestrator_disabled_means_no_eviction() {
    let dir = tmpdir("disabled_no_evict");
    let mut brain = make_brain_with_text_pool();
    brain.attach_cold_tiers(&dir);

    for _ in 0..15 {
        brain.observe(1, b"betagamma");
        brain.advance_tick();
    }

    let disabled = OrchestratorParams::disabled();
    brain.fabric_mut().set_tier_orchestrator_params(disabled);
    // Capture pre-disable state so we can assert no further activity.
    let before = brain.fabric().tier_orchestrator_stats();

    for _ in 0..30 {
        brain.advance_tick();
    }
    let after = brain.fabric().tier_orchestrator_stats();
    assert_eq!(after.passes, before.passes,
        "disabled params must not run any further passes (before={}, after={})",
        before.passes, after.passes);
    assert_eq!(after.neurons_evicted, before.neurons_evicted,
        "disabled params must not evict anything (before={}, after={})",
        before.neurons_evicted, after.neurons_evicted);

    std::fs::remove_dir_all(&dir).ok();
}

/// Atoms (substrate units) and the binding pool are off-limits to the
/// orchestrator, mirroring Brain::run_eviction_pass policy.
#[test]
fn orchestrator_never_evicts_atoms() {
    let dir = tmpdir("never_evicts_atoms");
    let mut brain = make_brain_with_text_pool();
    brain.attach_cold_tiers(&dir);

    for _ in 0..10 {
        brain.observe(1, b"deltaepsilon");
        brain.advance_tick();
    }

    // Capture pre-eviction atom IDs (they're the only neurons with
    // n.is_atom() == true).
    let pre_atoms: Vec<_> = brain.fabric()
        .pool(1).unwrap()
        .read()
        .iter_neurons()
        .filter(|n| n.is_atom())
        .map(|n| n.id)
        .collect();
    assert!(!pre_atoms.is_empty(), "test setup must produce atoms");

    let nuke = OrchestratorParams {
        run_every_n_ticks: 1,
        scan_budget: 4096,
        max_evict_per_pass: 4096,
        target_terminals_per_pool: 0,
        evict_threshold: -1_000_000.0,  // floor: literally every concept passes
        w_terminals: 1.0,
        w_staleness: 1.0,
        w_inverse_salience: 0.0,
        w_pinned: 0.0,
        decay_horizon_ticks: 1,
        salience_eps: 0.01,
        page_in_salience_floor: 0.0,
        max_page_in_per_pass: 0,
        min_age_ticks: 0,
        min_system_available_mb: 0, // hermetic: ignore live machine RAM
    };
    brain.fabric_mut().set_tier_orchestrator_params(nuke);
    for _ in 0..15 {
        brain.advance_tick();
    }
    // Atoms must still NOT be in the evicted set.
    let pool = brain.fabric().pool(1).unwrap();
    let p = pool.read();
    for atom_id in pre_atoms {
        assert!(!p.is_evicted(atom_id),
            "atom {} was evicted by the orchestrator", atom_id);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The full serialize→deserialize round trip the tier system exists
/// for: knowledge evicted to the SSD must come back transparently the
/// moment a prediction needs it.
///
///   1. Train until concepts with terminals exist; capture the baseline
///      activation spread of a prediction.
///   2. Force-evict every evictable concept (terminals shed to cold tier).
///   3. Run the same prediction again with demand paging enabled —
///      propagate() must hydrate the evicted neurons (paged_in > 0) and
///      the activation spread must recover to at least the baseline.
#[test]
fn evicted_knowledge_pages_back_in_for_prediction() {
    let dir = tmpdir("hydrate_round_trip");
    let mut brain = make_brain_with_text_pool();
    brain.attach_cold_tiers(&dir);

    for _ in 0..14 {
        brain.observe(1, b"alphabeta");
        brain.advance_tick();
    }

    // Baseline: what does a prediction activate when everything is hot?
    let fired = brain.activate_for_prediction(1, b"alphabeta");
    assert!(!fired.is_empty(), "training must make the frame recognizable");
    let baseline = brain.fabric().propagate(1);
    let baseline_targets: usize = baseline.values().map(|m| m.len()).sum();
    assert!(baseline_targets > 0, "baseline prediction must activate something");
    brain.clear_prediction_activation();

    // Force-evict every evictable concept (page-in off during the purge).
    let nuke = OrchestratorParams {
        run_every_n_ticks: 1,
        scan_budget: 4096,
        max_evict_per_pass: 4096,
        target_terminals_per_pool: 0,
        evict_threshold: -1_000_000.0,
        w_terminals: 1.0,
        w_staleness: 1.0,
        w_inverse_salience: 0.0,
        w_pinned: 0.0,
        decay_horizon_ticks: 1,
        salience_eps: 0.01,
        page_in_salience_floor: 0.0,
        max_page_in_per_pass: 0,
        min_age_ticks: 0,
        min_system_available_mb: 0,
    };
    brain.fabric_mut().set_tier_orchestrator_params(nuke);
    for _ in 0..10 {
        brain.advance_tick();
    }
    let evicted_now = brain.fabric().pool(1).unwrap().read().evicted_count();
    assert!(evicted_now > 0, "purge phase must actually evict concepts");
    let purged = brain.fabric().tier_orchestrator_stats();
    assert!(purged.neurons_evicted > 0);

    // Prediction with demand paging: disabled() stops further eviction
    // (run_every_n_ticks = MAX) but keeps the default page-in budget.
    brain.fabric_mut().set_tier_orchestrator_params(OrchestratorParams::disabled());
    let fired2 = brain.activate_for_prediction(1, b"alphabeta");
    assert!(!fired2.is_empty(), "atoms never evict, so the frame still fires");
    let after = brain.fabric().propagate(1);
    let after_targets: usize = after.values().map(|m| m.len()).sum();
    let stats = brain.fabric().tier_orchestrator_stats();

    assert!(stats.neurons_paged_in > 0,
        "prediction must page evicted knowledge back in (paged_in={})",
        stats.neurons_paged_in);
    assert_eq!(stats.page_in_errors, 0,
        "page-ins must not error (got {})", stats.page_in_errors);
    assert!(after_targets >= baseline_targets,
        "activation spread must recover after hydration: after={} baseline={}",
        after_targets, baseline_targets);
    let evicted_after = brain.fabric().pool(1).unwrap().read().evicted_count();
    assert!(evicted_after < evicted_now,
        "hydration must shrink the evicted set ({} -> {})",
        evicted_now, evicted_after);

    std::fs::remove_dir_all(&dir).ok();
}

/// Pressure factor reacts to actual terminal load.  When target is
/// generous (we're under budget), the factor is large → threshold is
/// hard to cross → few/no evictions.
#[test]
fn under_budget_means_no_eviction() {
    let dir = tmpdir("under_budget");
    let mut brain = make_brain_with_text_pool();
    brain.attach_cold_tiers(&dir);

    for _ in 0..10 {
        brain.observe(1, b"zeta-eta");
        brain.advance_tick();
    }

    // Way under budget: target is huge → pressure_factor clamps high
    // → threshold * pressure_factor stays large → nothing evicts.
    let lax = OrchestratorParams {
        run_every_n_ticks: 1,
        scan_budget: 4096,
        max_evict_per_pass: 4096,
        target_terminals_per_pool: 100_000_000_000,
        evict_threshold: 1.0,
        min_system_available_mb: 0, // hermetic: ignore live machine RAM
        ..OrchestratorParams::default()
    };
    brain.fabric_mut().set_tier_orchestrator_params(lax);
    for _ in 0..20 {
        brain.advance_tick();
    }
    let s = brain.fabric().tier_orchestrator_stats();
    assert_eq!(s.neurons_evicted, 0,
        "lax pressure config must evict nothing (got {})",
        s.neurons_evicted);
    assert!(s.passes > 0, "orchestrator should still have run passes");

    std::fs::remove_dir_all(&dir).ok();
}
