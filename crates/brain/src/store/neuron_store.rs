//! NeuronStore trait per [`ARCHITECTURE.md`] §18.2.
//!
//! The abstraction below `Pool::neurons`.  Hides whether a neuron lives
//! in local RAM, on local disk, or on a peer node's RAM/disk.  §18
//! generalises §17.4's local cold tier into a *distributed* tier across
//! arbitrary numbers of cluster hosts, by allowing the store impl to be
//! a remote-RPC wrapper.
//!
//! # Stage 18.12 step 1 (this file)
//!
//! Defines the trait and the [`RamStore`] default impl — wraps a
//! `Vec<Neuron>` keyed by `NeuronId` (the existing layout).  No
//! functional change to existing code; this just establishes the
//! abstraction so later stages can plug in other backends.
//!
//! Subsequent stages add:
//! - Stage 18.12 step 2: `ColdDiskStore` adapter over §17.4's `ColdTier`.
//! - Stage 18.12 step 3: `RemoteNodeStore` — HTTP/bincode client to
//!   a peer brain's `/shard/get_neuron` endpoint.
//! - Stage 18.12 step 4: `TieredStore` composing the three with a
//!   placement policy (consistent hash, salience-tiered, co-firing).

use crate::neuron::{Neuron, NeuronId};

/// Logical cluster-node identifier.  In Solo mode the brain has a
/// single NodeId, conventionally 0; in cluster modes the head assigns
/// one to each joining worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default,
         serde::Serialize, serde::Deserialize)]
pub struct NodeId(pub u32);

/// The single point of indirection between `Pool` and physical storage
/// of neurons.  Per [`ARCHITECTURE.md`] §18.2 every neuron access goes
/// through this trait so the storage layout can be changed without
/// touching substrate code.
///
/// Implementations are expected to be cheap to `Arc::clone` so a Pool
/// can share its store with eviction actors, cluster-sync threads, and
/// the placement policy without contention.
pub trait NeuronStore: Send + Sync {
    /// Fetch the neuron with `id`.  Returns `None` if the id is
    /// unknown to this store.  For tiered stores, the lookup walks
    /// RAM → cold disk → remote nodes in that order.
    fn get(&self, id: NeuronId) -> Option<Neuron>;

    /// Insert or overwrite the neuron at `id`.  The store implementation
    /// decides where to physically place it (e.g. tiered stores consult
    /// their placement policy).
    fn put(&self, id: NeuronId, n: Neuron);

    /// Remove the neuron at `id`.  Idempotent.
    fn delete(&self, id: NeuronId);

    /// Iterate over every id this store knows about.  Order is
    /// implementation-defined; the iterator may be empty.
    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a>;

    /// Number of neurons in the store.  Cheap O(1) for the default
    /// impl; tiered stores may aggregate across tiers.
    fn len(&self) -> usize;

    /// True iff `len() == 0`.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Determine which cluster node is the authoritative home for
    /// `id` — used by the placement policy in tiered stores.  For
    /// solo-mode stores this returns the local NodeId always.
    fn home_for(&self, _id: NeuronId) -> NodeId {
        NodeId(0)
    }
}

// ============================================================================
// RamStore — default local-RAM impl wrapping a Vec<Neuron>.
// ============================================================================

/// Default in-RAM store.  Matches the current `Pool::neurons: Vec<Neuron>`
/// layout: ids are dense and equal to the index.  Out-of-range ids
/// return `None` on `get`; `put` either updates the existing slot or,
/// if `id == len()`, appends a new slot.
pub struct RamStore {
    inner: parking_lot::RwLock<Vec<Option<Neuron>>>,
    node_id: NodeId,
}

impl RamStore {
    pub fn new() -> Self {
        Self {
            inner: parking_lot::RwLock::new(Vec::new()),
            node_id: NodeId(0),
        }
    }

    pub fn with_node_id(node_id: NodeId) -> Self {
        Self {
            inner: parking_lot::RwLock::new(Vec::new()),
            node_id,
        }
    }

    /// Construct from an existing Vec<Neuron>.  Used by Stage 18.12
    /// step 4 to migrate `Pool::neurons` into the store without
    /// regenerating every neuron.
    pub fn from_vec(neurons: Vec<Neuron>) -> Self {
        let inner: Vec<Option<Neuron>> = neurons.into_iter()
            .map(Some)
            .collect();
        Self {
            inner: parking_lot::RwLock::new(inner),
            node_id: NodeId(0),
        }
    }
}

impl Default for RamStore {
    fn default() -> Self { Self::new() }
}

impl NeuronStore for RamStore {
    fn get(&self, id: NeuronId) -> Option<Neuron> {
        let g = self.inner.read();
        g.get(id as usize).and_then(|slot| slot.clone())
    }

    fn put(&self, id: NeuronId, n: Neuron) {
        let mut g = self.inner.write();
        let idx = id as usize;
        if idx == g.len() {
            g.push(Some(n));
        } else if idx < g.len() {
            g[idx] = Some(n);
        } else {
            // Sparse insert: pad with None tombstones.  Caller is
            // responsible for keeping ids dense in normal training
            // paths; this branch covers crash-recovery / cluster-pull
            // patterns where ids arrive non-sequentially.
            while g.len() < idx {
                g.push(None);
            }
            g.push(Some(n));
        }
    }

    fn delete(&self, id: NeuronId) {
        let mut g = self.inner.write();
        if let Some(slot) = g.get_mut(id as usize) {
            *slot = None;
        }
    }

    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a> {
        // Snapshot the present ids under a read lock; the iterator
        // owns the snapshot so the caller doesn't have to hold the
        // lock for the whole iteration.
        let g = self.inner.read();
        let ids: Vec<NeuronId> = g.iter().enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|_| i as NeuronId))
            .collect();
        Box::new(ids.into_iter())
    }

    fn len(&self) -> usize {
        // Count present slots (skip tombstones).
        self.inner.read().iter().filter(|s| s.is_some()).count()
    }

    fn home_for(&self, _id: NeuronId) -> NodeId {
        self.node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::NeuronKind;

    fn n(id: u32, label: &str) -> Neuron {
        Neuron::new_atom(id, label.to_string(), NeuronKind::Excitatory, 0)
    }

    #[test]
    fn empty_store_is_empty() {
        let s = RamStore::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.get(0).is_none());
        assert_eq!(s.iter_ids().count(), 0);
    }

    #[test]
    fn put_then_get_round_trips() {
        let s = RamStore::new();
        s.put(0, n(0, "t:a"));
        s.put(1, n(1, "t:b"));
        assert_eq!(s.len(), 2);
        let r0 = s.get(0).unwrap();
        let r1 = s.get(1).unwrap();
        assert_eq!(r0.label, "t:a");
        assert_eq!(r1.label, "t:b");
    }

    #[test]
    fn from_vec_preserves_neurons() {
        let v = vec![n(0, "x"), n(1, "y"), n(2, "z")];
        let s = RamStore::from_vec(v);
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(1).unwrap().label, "y");
    }

    #[test]
    fn delete_removes_from_iteration() {
        let s = RamStore::new();
        s.put(0, n(0, "a"));
        s.put(1, n(1, "b"));
        s.put(2, n(2, "c"));
        assert_eq!(s.len(), 3);
        s.delete(1);
        assert_eq!(s.len(), 2);
        let ids: Vec<NeuronId> = s.iter_ids().collect();
        assert_eq!(ids, vec![0, 2]);
        assert!(s.get(1).is_none());
    }

    #[test]
    fn out_of_range_get_is_none() {
        let s = RamStore::new();
        s.put(0, n(0, "a"));
        assert!(s.get(99).is_none());
    }

    #[test]
    fn sparse_put_pads_with_tombstones() {
        let s = RamStore::new();
        s.put(0, n(0, "first"));
        // Skip ids 1 and 2; put id 3.  Length should still report 2
        // (only present slots count).
        s.put(3, n(3, "fourth"));
        assert_eq!(s.len(), 2);
        assert!(s.get(0).is_some());
        assert!(s.get(1).is_none());
        assert!(s.get(2).is_none());
        assert!(s.get(3).is_some());
        let ids: Vec<NeuronId> = s.iter_ids().collect();
        assert_eq!(ids, vec![0, 3]);
    }

    #[test]
    fn home_for_returns_configured_node_id() {
        let s = RamStore::with_node_id(NodeId(7));
        assert_eq!(s.home_for(0), NodeId(7));
        assert_eq!(s.home_for(999_999), NodeId(7));
    }
}
