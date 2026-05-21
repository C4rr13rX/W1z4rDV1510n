//! Storage-tier dynamical-system control per [`ARCHITECTURE.md`] §17.8.
//!
//! The same control architecture that governs the substrate's
//! plasticity knobs (`crate::pool::ControlMode` / `ControlSignal` /
//! `ControlState`) also governs the storage tier.  This module defines
//! the parallel structures.
//!
//! # Stage 17.8 (this file) — interface + observable signals
//!
//! - [`StorageControlState`] aggregates the live observables that
//!   storage policy reads each tick: salience entropy, Bloom load,
//!   working-set pressure (stub until §17.4 full), cache hit rate
//!   (stub until §17.4 full), replay value score (stub until §17.7
//!   full free-energy replay).
//! - [`StorageConfig`] mirrors `PoolConfig`'s pattern with ControlMode-
//!   typed knobs: eviction_mode, replay_rate, shard_rebalance_threshold,
//!   bloom_resize_pressure.
//! - The Brain exposes its current `StorageControlState` via
//!   `Brain::storage_control_state` so the brain_server can surface it
//!   via `/storage_state` and GA search can read it.
//!
//! Stages that will plug behaviors into these knobs:
//! - 17.4 full: `eviction_mode` drives how aggressively the eviction
//!   actor evicts working-set neurons to cold tier.
//! - 17.4 full: `bloom_resize_pressure` drives when the Bloom doubles.
//! - 17.6 full: `shard_rebalance_threshold` drives when the cluster
//!   re-partitions the co-activation graph.
//! - 17.7 full: `replay_rate` drives how many moments replay per
//!   sleep cycle (free-energy weighted).

use serde::{Deserialize, Serialize};

use crate::pool::ControlMode;

/// Live observables of the storage tier's behaviour.  Updated by the
/// brain each time storage policy is read (typically once per `/observe`
/// or once per `/sleep`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StorageControlState {
    /// Working-set bytes / configured budget.  Range \[0, ∞); typically
    /// \[0, 1].  Above 1.0 means pressure is past the budget — eviction
    /// runs more aggressively.  Stage 17.4 full will populate this from
    /// the LRU working-set cache; until then this is 0.0.
    pub working_set_pressure:       f32,

    /// EMA of working-set hit rate per /observe.  1.0 = every requested
    /// neuron was in the working set; 0.0 = every observe required a
    /// cold-tier page-in.  Stage 17.4 full populates this; until then
    /// it's 0.0 (no cold tier exists, so all hits trivially come from
    /// the working set).
    pub cache_hit_rate:             f32,

    /// EMA of free-energy reduction per replay tick.  Stage 17.7 full
    /// will compute this from the annealer's energy estimate before/after
    /// each replay.  Until then 0.0.
    pub replay_value_score:         f32,

    /// Shannon entropy of the salience distribution across all neurons
    /// in all pools.  High entropy → brain is exploring (many neurons
    /// at similar salience).  Low entropy → brain is concentrating
    /// (a few high-salience neurons dominate).  Stage 17.5 makes this
    /// computable today.
    pub salience_distribution_entropy: f32,

    /// Cumulative Bloom load factor = sum(inserted_keys) / sum(slots)
    /// across all pools.  Above 0.5 indicates rising false-positive
    /// rate; eventual `bloom_resize_pressure` knob would respond.
    pub bloom_load:                 f32,
}

impl StorageControlState {
    /// Compute Shannon entropy from a binned histogram.  Bins must sum
    /// to non-zero or the result is 0.0.  H(X) = -Σ p log2 p, in bits.
    pub fn entropy_from_bins(bins: &[u64]) -> f32 {
        let total: u64 = bins.iter().sum();
        if total == 0 { return 0.0; }
        let inv = 1.0 / total as f64;
        let mut h = 0.0f64;
        for &b in bins {
            if b > 0 {
                let p = b as f64 * inv;
                h -= p * p.log2();
            }
        }
        h as f32
    }
}

/// Mirrors `PoolConfig`'s knob pattern — ControlMode-typed fields that
/// the brain evaluates each cycle against `StorageControlState`.  Per
/// [`ARCHITECTURE.md`] §17.8 dynamical-system principle extended to
/// storage: every knob is either `Constant` (legacy / GA-pinned) or
/// `DrivenBy` an observable signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// How aggressively the eviction actor reclaims working-set neurons
    /// to cold tier.  Higher values → evict more.  Stage 17.4 full
    /// consumes this.  Default `Constant(0.5)`.
    pub eviction_mode:              ControlMode,

    /// How many moments replay per sleep cycle (as a fraction of
    /// `moment_history_window`).  Stage 17.7 full consumes this.
    /// Default `Constant(0.25)`.
    pub replay_rate:                ControlMode,

    /// When the cluster's co-activation graph diverges by this much
    /// from the current partition, trigger re-balance.  Stage 17.6 full
    /// consumes this.  Default `Constant(0.2)`.
    pub shard_rebalance_threshold:  ControlMode,

    /// When Bloom load crosses this, double the filter.  Stage 17.4
    /// full consumes this.  Default `Constant(0.5)`.
    pub bloom_resize_pressure:      ControlMode,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            eviction_mode:              ControlMode::Constant(0.5),
            replay_rate:                ControlMode::Constant(0.25),
            shard_rebalance_threshold:  ControlMode::Constant(0.2),
            bloom_resize_pressure:      ControlMode::Constant(0.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_is_log2_n() {
        // 4 equally-populated bins → entropy = log2(4) = 2 bits.
        let bins = [10u64, 10, 10, 10];
        let h = StorageControlState::entropy_from_bins(&bins);
        assert!((h - 2.0).abs() < 1e-5, "expected ~2.0 bits, got {}", h);
    }

    #[test]
    fn entropy_concentrated_is_low() {
        // All mass in one bin → entropy = 0.
        let bins = [40u64, 0, 0, 0];
        let h = StorageControlState::entropy_from_bins(&bins);
        assert!(h.abs() < 1e-5, "expected 0 bits, got {}", h);
    }

    #[test]
    fn entropy_empty_is_zero() {
        let bins = [0u64; 4];
        let h = StorageControlState::entropy_from_bins(&bins);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn storage_config_defaults_are_constants() {
        let c = StorageConfig::default();
        match c.eviction_mode {
            ControlMode::Constant(v) => assert!((v - 0.5).abs() < 1e-5),
            _ => panic!("default eviction_mode should be Constant"),
        }
    }
}
