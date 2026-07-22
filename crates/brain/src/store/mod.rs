//! Storage layer per [`ARCHITECTURE.md`] §17.
//!
//! This module is the *persistence substrate* for the brain.  Per spec §17.9,
//! the training loop *is* the write-ahead log: every mutation that affects
//! learned state appends a `WalEvent` to the pool's append-only log before
//! the in-memory state is considered durable.  There is no separate
//! `checkpoint()` phase that allocates a multi-gigabyte bincode blob in RAM.
//!
//! # Stage 1 scope (this file)
//!
//! - [`Store`] trait + the two default implementations: [`NoopStore`] for
//!   tests + brains constructed without `W1Z4RDV1510N_DATA_DIR`, and
//!   `MmapWalStore` for production use.
//! - [`WalEvent`] enum covering every state-changing operation the brain
//!   currently performs.
//! - Crash-replay recovery in [`recovery::replay_into_brain`].
//!
//! Stages 17.3 (Bloom-gated neurogenesis), 17.4 (eviction actor), 17.5
//! (salience), 17.6 (cluster anti-entropy), 17.7 (free-energy replay), and
//! 17.8 (self-tuning) all build on top of this module; none of them touch
//! the existing Pool API yet.

pub mod bloom;
pub mod cold;
pub mod container;
pub mod control;
pub mod event;
pub mod merkle;
pub mod neuron_store;
pub(crate) mod posting_index;
pub mod recovery;
pub mod wal;
pub mod wbrain_store;

pub use bloom::CountingBloom;
pub use cold::ColdTier;
pub use container::{
    AuxiliaryRecordRef, BrainContainer, BrainContainerManifest, PoolContainerManifest,
};
pub use control::{StorageConfig, StorageControlState};
pub use event::{TerminalDelta, WalEvent};
pub use merkle::{PoolRoot, compute_pool_root};
pub use neuron_store::{
    CapacityWeightedPlacement, ColdDiskStore, ConsistentHashPlacement, HebbianClusteredPlacement,
    NeuronStore, NodeId, PlacementPolicy, RamStore, RemoteNodeStore, RemoteTransport, TieredStore,
};
pub use recovery::{RecoveryStats, load_events_after_marker, replay_into_brain};
pub use wal::{MmapWalStore, NoopStore, Store};
pub use wbrain_store::{WbrainFile, WbrainNeuronStore};
