//! Write-ahead-log event types per [`ARCHITECTURE.md`] §17.9.
//!
//! Every mutation the substrate performs that affects learned state — atom
//! neurogenesis, concept emergence, terminal reinforcement, tick advancement,
//! pool registration, sleep-prune, evict-to-disk — is expressed as a
//! [`WalEvent`].  The event stream is the *durable record* of the brain;
//! after a crash, replaying the WAL reconstructs the brain bit-identical to
//! the moment before the crash (modulo the last buffered append, ≤4 KB).
//!
//! # Event taxonomy
//!
//! These are deliberately *fine-grained* — one event per atomic mutation —
//! so crash recovery never reconstructs from inconsistent partial state.
//! Coarser snapshot events (`PoolRegistered`, `Snapshot`) record initial
//! conditions; the brain replays incremental events from those baselines.
//!
//! Layout discipline: every variant is `#[repr(C)]`-stable through bincode's
//! variant-tag-then-fields encoding.  Adding new variants is forward-safe;
//! removing or reordering existing variants is a breaking change to the WAL
//! format and requires a `wal_format_version` bump in [`crate::store::wal`].

use serde::{Deserialize, Serialize};

use crate::neuron::{NeuronId, NeuronKind, NeuronRef, PoolId};
use crate::pool::PoolConfig;

/// One incremental update to a terminal's reinforcement state.
///
/// Note that `weight_quantized` is the new total weight (NOT a delta) so
/// recovery doesn't need to re-derive saturation.  `consolidation` is the
/// current consolidation counter post-update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminalDelta {
    pub src:               NeuronRef,
    pub dst:               NeuronRef,
    /// Post-update weight, quantized to 8 bits over `[0.0, max_weight]`.
    /// Stage 17.1: stored as f32 for now; quantization comes in stage
    /// 17.4 once the eviction actor is doing the on-disk encoding.
    pub weight:            f32,
    pub consolidation:     u8,
    pub last_fired_tick:   u64,
}

/// Every mutation the brain performs that affects durable learned state.
///
/// See [`crate::store::recovery::replay_into_brain`] for the inverse — how
/// each event reconstructs the corresponding in-memory mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEvent {
    /// A pool was created or re-registered with a specific config.  Recovery
    /// recreates the pool with this exact config before replaying any
    /// neuron-level events.
    PoolRegistered {
        pool_id:     PoolId,
        config:      PoolConfig,
        encoding_name: String,
    },

    /// A new atom neuron was created.  Recovery re-creates the neuron at
    /// the assigned `id` slot.  The substrate guarantees `id` is the next
    /// available slot at creation time so replay reproduces ID assignment
    /// faithfully provided the same event order.
    AtomCreated {
        pool_id:     PoolId,
        id:          NeuronId,
        label:       String,
        kind:        NeuronKind,
        born_tick:   u64,
    },

    /// A concept neuron was promoted from a recurring sequence in
    /// `recent_atoms` (within-pool) or from cross-pool co-firing
    /// (binding concept).  Members capture the exact promotion-time
    /// sequence; recovery does NOT re-run emergence detection from the
    /// raw atom stream, it inserts the concept directly.
    ConceptEmerged {
        pool_id:     PoolId,
        id:          NeuronId,
        label:       String,
        kind:        NeuronKind,
        members:     Vec<NeuronRef>,
        born_tick:   u64,
    },

    /// One terminal was reinforced (created if new, weight-bumped if
    /// existing).  Recovery applies this idempotently — the post-update
    /// `weight` is authoritative, not a delta.
    TerminalReinforced(TerminalDelta),

    /// Decay-and-prune pass removed terminals below `prune_floor`.  We
    /// log the FINAL state of the source neuron's terminal list rather
    /// than the per-edge removals; this collapses what would otherwise
    /// be O(pruned_count) events per tick into one event per neuron
    /// touched by the prune pass.
    NeuronTerminalsPruned {
        pool_id:     PoolId,
        neuron_id:   NeuronId,
        /// Remaining terminals after prune (full list, sorted by target).
        survivors:   Vec<TerminalDelta>,
    },

    /// A neuron was evicted from RAM to cold tier — informational only at
    /// stage 17.1 (no actual eviction happens yet).  Stage 17.4 will use
    /// this to coordinate `WorkingSet` ↔ disk paging.
    NeuronEvicted {
        pool_id:     PoolId,
        neuron_id:   NeuronId,
    },

    /// `Fabric::advance_tick` ran.  Recovery uses the latest such event
    /// to set the fabric's current tick.  Decay-and-prune work that
    /// happens during a tick is logged as separate `NeuronTerminalsPruned`
    /// events before this one (within the same tick).
    TickAdvanced {
        new_tick:    u64,
    },

    /// Brain checkpoint barrier — written by `Brain::checkpoint()`.  Marks
    /// a point at which all preceding events have been fsynced to disk.
    /// Recovery may skip ahead to the *latest* `SnapshotMarker` whose
    /// snapshot file is intact, then replay events after it.
    SnapshotMarker {
        tick:        u64,
        wall_time_ms: i64,
    },
}

impl WalEvent {
    /// Stable name for diagnostics + Merkle hashing of event streams.
    pub fn variant_name(&self) -> &'static str {
        match self {
            WalEvent::PoolRegistered { .. }        => "PoolRegistered",
            WalEvent::AtomCreated { .. }           => "AtomCreated",
            WalEvent::ConceptEmerged { .. }        => "ConceptEmerged",
            WalEvent::TerminalReinforced(_)        => "TerminalReinforced",
            WalEvent::NeuronTerminalsPruned { .. } => "NeuronTerminalsPruned",
            WalEvent::NeuronEvicted { .. }         => "NeuronEvicted",
            WalEvent::TickAdvanced { .. }          => "TickAdvanced",
            WalEvent::SnapshotMarker { .. }        => "SnapshotMarker",
        }
    }
}
