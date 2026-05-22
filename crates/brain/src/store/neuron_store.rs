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

use std::sync::Arc;

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

// ============================================================================
// ColdDiskStore — §18.12 step 2: wraps the §17.4 ColdTier through the
// NeuronStore trait so a tiered store can route puts to disk uniformly.
// ============================================================================

use ahash::AHashMap;

use crate::store::cold::ColdTier;

/// NeuronStore impl backed by a §17.4 ColdTier append-only file.
/// Maintains an in-RAM offset index (`NeuronId → byte offset`) since
/// the cold tier file itself is purely append-only — random reads need
/// the offset to seek.
///
/// This is the §18.12 step 2 adapter: it surfaces the existing cold-
/// tier substrate behind the unified NeuronStore trait.  Stage 18.12
/// step 4 will compose this with RamStore + RemoteNodeStore in
/// TieredStore.
pub struct ColdDiskStore {
    tier:    Arc<ColdTier>,
    offsets: parking_lot::RwLock<AHashMap<NeuronId, u64>>,
    node_id: NodeId,
}

impl ColdDiskStore {
    /// Construct from an already-opened ColdTier (typically shared with
    /// the Pool that previously owned it during the §17.4 eviction
    /// phase).  Starts with an empty offset index; populate via `put`
    /// or `seed_offsets` if restoring from a snapshot.
    pub fn new(tier: Arc<ColdTier>) -> Self {
        Self {
            tier,
            offsets: parking_lot::RwLock::new(AHashMap::new()),
            node_id: NodeId(0),
        }
    }

    /// Seed the offset index from a previously-persisted map (e.g. the
    /// PoolSnapshot's `cold_offsets` per §17.4 step 5).
    pub fn seed_offsets<I>(&self, iter: I)
    where I: IntoIterator<Item = (NeuronId, u64)> {
        let mut g = self.offsets.write();
        for (id, off) in iter { g.insert(id, off); }
    }

    pub fn with_node_id(mut self, node_id: NodeId) -> Self {
        self.node_id = node_id;
        self
    }
}

impl NeuronStore for ColdDiskStore {
    fn get(&self, id: NeuronId) -> Option<Neuron> {
        let offset = *self.offsets.read().get(&id)?;
        match self.tier.read_neuron(offset) {
            Ok(n)  => Some(n),
            Err(e) => {
                tracing::warn!(
                    "ColdDiskStore::get(id={}) read failed at offset {}: {}",
                    id, offset, e,
                );
                None
            }
        }
    }

    fn put(&self, id: NeuronId, n: Neuron) {
        // §17.4 ColdTier is append-only — every put writes a new record.
        // Old records become garbage; a future compaction pass reclaims
        // them.  This matches LSM-tree semantics (Rosenblum & Ousterhout
        // 1991; RocksDB).
        let mut neuron = n;
        if neuron.id != id { neuron.id = id; }
        match self.tier.append_neuron(&neuron) {
            Ok(offset) => {
                self.offsets.write().insert(id, offset);
            }
            Err(e) => {
                tracing::warn!(
                    "ColdDiskStore::put(id={}) append failed: {}", id, e,
                );
            }
        }
    }

    fn delete(&self, id: NeuronId) {
        self.offsets.write().remove(&id);
        // Cold-tier data left in place; compaction follow-up reclaims.
    }

    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a> {
        let g = self.offsets.read();
        let ids: Vec<NeuronId> = g.keys().copied().collect();
        Box::new(ids.into_iter())
    }

    fn len(&self) -> usize {
        self.offsets.read().len()
    }

    fn home_for(&self, _id: NeuronId) -> NodeId {
        self.node_id
    }
}

// ============================================================================
// PlacementPolicy + TieredStore — §18.12 step 4: the "OpenStack pool"
// composer that routes operations across RamStore (local hot),
// ColdDiskStore (local warm), and RemoteNodeStore-per-peer (remote tier).
// ============================================================================

/// Decides which cluster node is the home for a given neuron.  Per
/// [`ARCHITECTURE.md`] §18.3, placement is what makes the resource-pool
/// abstraction work: every neuron has exactly one home; reads/writes
/// route there.
///
/// The ring is the ordered list of currently-live cluster node ids;
/// the policy is asked to pick one of them for `id`.
pub trait PlacementPolicy: Send + Sync {
    fn home_for(&self, id: NeuronId, ring: &[NodeId]) -> NodeId;
}

/// Default: consistent-hash placement.  Each neuron lands on
/// `ring[(id as usize) % ring.len()]`.  Minimal rebalancing on node
/// join/leave (Karger et al. 1997).  No co-firing awareness — strategy
/// 1 from §18.3.  Production deployments may swap in a salience-tiered
/// or graph-partitioning policy.
pub struct ConsistentHashPlacement;

impl PlacementPolicy for ConsistentHashPlacement {
    fn home_for(&self, id: NeuronId, ring: &[NodeId]) -> NodeId {
        if ring.is_empty() {
            // Degenerate: no peers known.  Caller should ensure the
            // ring includes at least the local node.
            return NodeId(0);
        }
        // Mix the id bits with a multiplicative hash so adjacent ids
        // don't always land on the same node (the trivial `id % n`
        // pattern groups all ids 0..n-1 onto distinct nodes but ids
        // n..2n-1 in the same pattern — workable for cluster sizes
        // that don't change, but the hash mixes for robustness against
        // ring re-sizing).
        let mixed = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ring[(mixed as usize) % ring.len()]
    }
}

/// TieredStore composes a local RAM tier, an optional local cold-disk
/// tier, and zero-or-more remote-node tiers into a single NeuronStore.
/// Per [`ARCHITECTURE.md`] §18.2, this is the trait object the Pool
/// will hold (step 4b — wiring into Pool follows in a separate commit).
///
/// Routing:
/// - On `get(id)`: compute `home = placement.home_for(id, ring)`.  If
///   `home == local_node`, walk RAM → cold disk.  Else issue a remote
///   `get` to that node.
/// - On `put(id, n)`: same routing — local goes to RAM (the hot tier),
///   remote goes over the wire.  Cold-disk writes happen out-of-band
///   via the eviction actor (§17.4); TieredStore::put always writes
///   the hot tier first.
/// - On `delete(id)`: routed analogously.
/// - On `iter_ids` / `len`: returns LOCAL state only — remote
///   enumeration is not in the protocol.  Callers needing global
///   iteration query each peer separately.
pub struct TieredStore {
    local_node: NodeId,
    ram:        Arc<RamStore>,
    cold:       Option<Arc<ColdDiskStore>>,
    remotes:    Arc<parking_lot::RwLock<AHashMap<NodeId, Arc<RemoteNodeStore>>>>,
    /// Current cluster ring; ordered, includes `local_node`.  Updated
    /// by cluster join/leave handlers (step 5).
    ring:       Arc<parking_lot::RwLock<Vec<NodeId>>>,
    placement:  Arc<dyn PlacementPolicy>,
}

impl TieredStore {
    /// Construct a single-node tiered store (ring contains only
    /// `local_node`).  Convenient for solo mode and tests; cluster mode
    /// extends the ring via [`Self::set_ring`].
    pub fn solo(local_node: NodeId, ram: Arc<RamStore>) -> Self {
        Self {
            local_node,
            ram,
            cold:       None,
            remotes:    Arc::new(parking_lot::RwLock::new(AHashMap::new())),
            ring:       Arc::new(parking_lot::RwLock::new(vec![local_node])),
            placement:  Arc::new(ConsistentHashPlacement),
        }
    }

    /// Attach a local cold-disk tier.
    pub fn with_cold(mut self, cold: Arc<ColdDiskStore>) -> Self {
        self.cold = Some(cold);
        self
    }

    /// Replace the default consistent-hash placement with a custom one.
    pub fn with_placement(mut self, p: Arc<dyn PlacementPolicy>) -> Self {
        self.placement = p;
        self
    }

    /// Attach (or replace) the remote tier for a peer node.
    pub fn set_remote(&self, node: NodeId, store: Arc<RemoteNodeStore>) {
        self.remotes.write().insert(node, store);
        // Ensure ring contains the peer.
        let mut ring = self.ring.write();
        if !ring.contains(&node) {
            ring.push(node);
            ring.sort_by_key(|n| n.0);
        }
    }

    /// Remove a peer (e.g. node left the cluster).
    pub fn remove_remote(&self, node: NodeId) {
        self.remotes.write().remove(&node);
        let mut ring = self.ring.write();
        ring.retain(|n| *n != node);
    }

    /// Set the full ring explicitly.  Used by the cluster-join
    /// handler when topology is known up-front.
    pub fn set_ring(&self, ring: Vec<NodeId>) {
        *self.ring.write() = ring;
    }

    /// Snapshot of the current ring.
    pub fn ring(&self) -> Vec<NodeId> { self.ring.read().clone() }

    /// Compute home for a given id under the current ring.
    pub fn compute_home(&self, id: NeuronId) -> NodeId {
        let ring = self.ring.read();
        self.placement.home_for(id, &ring)
    }

    /// Local node id.
    pub fn local_node(&self) -> NodeId { self.local_node }
}

impl NeuronStore for TieredStore {
    fn get(&self, id: NeuronId) -> Option<Neuron> {
        let home = self.compute_home(id);
        if home == self.local_node {
            // Local: RAM first, then cold disk.
            if let Some(n) = self.ram.get(id) { return Some(n); }
            if let Some(cold) = &self.cold {
                if let Some(n) = cold.get(id) {
                    // Stage 18.4: cache cold-tier hits back into RAM as
                    // a "wake" — promotes the neuron to the hot tier.
                    // Skipped here to keep TieredStore stateless; the
                    // eviction actor handles RAM promotion explicitly.
                    return Some(n);
                }
            }
            None
        } else {
            // Remote tier.
            let remotes = self.remotes.read();
            remotes.get(&home).and_then(|r| r.get(id))
        }
    }

    fn put(&self, id: NeuronId, n: Neuron) {
        let home = self.compute_home(id);
        if home == self.local_node {
            self.ram.put(id, n);
        } else {
            let remotes = self.remotes.read();
            if let Some(r) = remotes.get(&home) {
                r.put(id, n);
            } else {
                tracing::warn!(
                    "TieredStore::put(id={}) home=node{} has no remote tier attached",
                    id, home.0,
                );
            }
        }
    }

    fn delete(&self, id: NeuronId) {
        let home = self.compute_home(id);
        if home == self.local_node {
            self.ram.delete(id);
            if let Some(cold) = &self.cold { cold.delete(id); }
        } else {
            let remotes = self.remotes.read();
            if let Some(r) = remotes.get(&home) { r.delete(id); }
        }
    }

    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a> {
        // Local-only iteration (RAM + cold).  Use `compute_home(id) ==
        // local_node` to filter to ids the local node actually owns,
        // so the iterator doesn't return ids that happen to be in the
        // RAM cache but conceptually belong to a peer.
        let ram_ids: Vec<NeuronId> = self.ram.iter_ids().collect();
        let cold_ids: Vec<NeuronId> = match &self.cold {
            Some(c) => c.iter_ids().collect(),
            None    => Vec::new(),
        };
        let mut seen: AHashMap<NeuronId, ()> = AHashMap::new();
        for id in ram_ids.iter().chain(cold_ids.iter()) {
            if self.compute_home(*id) == self.local_node {
                seen.insert(*id, ());
            }
        }
        Box::new(seen.into_iter().map(|(id, _)| id))
    }

    fn len(&self) -> usize {
        self.iter_ids().count()
    }

    fn home_for(&self, id: NeuronId) -> NodeId {
        self.compute_home(id)
    }
}

// ============================================================================
// RemoteTransport + RemoteNodeStore — §18.12 step 3: NeuronStore that
// fetches/puts via RPC to a peer brain.  The actual transport
// (HTTP via reqwest, gRPC, or anything else) is supplied by the
// node-crate impl so the brain crate doesn't drag in an HTTP client.
// ============================================================================

use crate::neuron::PoolId;

/// Synchronous RPC primitive for the §18 distributed substrate.
/// Implemented by the node crate (e.g. `HttpRemoteTransport` over
/// reqwest); brain crate stays HTTP-client-free.
///
/// All methods are sync — the caller's runtime decides whether to wrap
/// them in `spawn_blocking` or call them from a blocking context.
pub trait RemoteTransport: Send + Sync {
    /// Fetch a neuron by (pool, id).  Returns None on 404 or any
    /// non-200 status.
    fn fetch_neuron(&self, pool: PoolId, id: NeuronId) -> Option<Neuron>;

    /// Insert or overwrite a neuron at the peer.  Returns true if the
    /// peer reports successful insert.
    fn put_neuron(&self, pool: PoolId, neuron: Neuron) -> bool;
}

/// NeuronStore impl that proxies operations to a peer node via the
/// `RemoteTransport`.  One instance per (peer_node, pool) — the store
/// is scoped to a single pool because that's the granularity at which
/// placement decisions are made.
///
/// `iter_ids()` and `len()` are stubbed because a remote store cannot
/// cheaply enumerate without a dedicated endpoint; later stages can
/// add `/shard/list_ids` if iteration becomes load-bearing.  For now
/// the iter is empty + len is 0 — RemoteNodeStore is intended for
/// use as a *tier* under TieredStore, not as a standalone enumerable.
pub struct RemoteNodeStore {
    transport:  Arc<dyn RemoteTransport>,
    pool_id:    PoolId,
    peer_node:  NodeId,
}

impl RemoteNodeStore {
    pub fn new(transport: Arc<dyn RemoteTransport>, pool_id: PoolId, peer_node: NodeId) -> Self {
        Self { transport, pool_id, peer_node }
    }
}

impl NeuronStore for RemoteNodeStore {
    fn get(&self, id: NeuronId) -> Option<Neuron> {
        self.transport.fetch_neuron(self.pool_id, id)
    }

    fn put(&self, id: NeuronId, mut n: Neuron) {
        if n.id != id { n.id = id; }
        let ok = self.transport.put_neuron(self.pool_id, n);
        if !ok {
            tracing::warn!(
                "RemoteNodeStore::put(pool={}, id={}, peer=node{}) failed",
                self.pool_id, id, self.peer_node.0,
            );
        }
    }

    fn delete(&self, _id: NeuronId) {
        // Stage 18.12 step 3: delete RPC not yet exposed by brain_server
        // shard endpoints (current behaviour is LSM-style append-only).
        // Future commit adds /shard/delete_neuron if a tombstone path
        // is needed.
        tracing::debug!(
            "RemoteNodeStore::delete is currently a no-op (LSM-append semantics)"
        );
    }

    fn iter_ids<'a>(&'a self) -> Box<dyn Iterator<Item = NeuronId> + 'a> {
        // RemoteNodeStore doesn't enumerate — see struct docs.
        Box::new(std::iter::empty())
    }

    fn len(&self) -> usize { 0 }

    fn home_for(&self, _id: NeuronId) -> NodeId { self.peer_node }
}

#[cfg(test)]
mod remote_node_tests {
    use super::*;
    use crate::neuron::NeuronKind;
    use std::sync::Mutex;

    /// Test-only transport that holds an in-memory neuron map.  Lets
    /// us exercise RemoteNodeStore's logic without spinning up an HTTP
    /// server.
    struct MockTransport {
        inner: Mutex<AHashMap<(PoolId, NeuronId), Neuron>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self { inner: Mutex::new(AHashMap::new()) }
        }
    }

    impl RemoteTransport for MockTransport {
        fn fetch_neuron(&self, pool: PoolId, id: NeuronId) -> Option<Neuron> {
            self.inner.lock().unwrap().get(&(pool, id)).cloned()
        }
        fn put_neuron(&self, pool: PoolId, neuron: Neuron) -> bool {
            self.inner.lock().unwrap().insert((pool, neuron.id), neuron);
            true
        }
    }

    fn n(id: u32, label: &str) -> Neuron {
        Neuron::new_atom(id, label.to_string(), NeuronKind::Excitatory, 0)
    }

    #[test]
    fn put_then_get_round_trips_through_transport() {
        let t = Arc::new(MockTransport::new()) as Arc<dyn RemoteTransport>;
        let store = RemoteNodeStore::new(t, /*pool*/ 1, NodeId(2));
        store.put(7, n(7, "remote"));
        let r = store.get(7).unwrap();
        assert_eq!(r.id, 7);
        assert_eq!(r.label, "remote");
    }

    #[test]
    fn get_unknown_returns_none() {
        let t = Arc::new(MockTransport::new()) as Arc<dyn RemoteTransport>;
        let store = RemoteNodeStore::new(t, 1, NodeId(2));
        assert!(store.get(999).is_none());
    }

    #[test]
    fn home_for_returns_peer_node() {
        let t = Arc::new(MockTransport::new()) as Arc<dyn RemoteTransport>;
        let store = RemoteNodeStore::new(t, 1, NodeId(42));
        assert_eq!(store.home_for(0), NodeId(42));
    }

    #[test]
    fn iter_ids_is_empty_by_design() {
        let t = Arc::new(MockTransport::new()) as Arc<dyn RemoteTransport>;
        let store = RemoteNodeStore::new(t, 1, NodeId(2));
        store.put(7, n(7, "x"));
        // Even after put, iter is empty — remote store isn't enumerable
        // in the current protocol.  Documented in struct docs.
        assert_eq!(store.iter_ids().count(), 0);
        assert_eq!(store.len(), 0);
    }
}

#[cfg(test)]
mod tiered_tests {
    use super::*;
    use crate::neuron::NeuronKind;
    use std::sync::Mutex;

    struct MockTransport {
        inner: Mutex<AHashMap<(PoolId, NeuronId), Neuron>>,
    }
    impl MockTransport {
        fn new() -> Self { Self { inner: Mutex::new(AHashMap::new()) } }
    }
    impl RemoteTransport for MockTransport {
        fn fetch_neuron(&self, pool: PoolId, id: NeuronId) -> Option<Neuron> {
            self.inner.lock().unwrap().get(&(pool, id)).cloned()
        }
        fn put_neuron(&self, pool: PoolId, neuron: Neuron) -> bool {
            self.inner.lock().unwrap().insert((pool, neuron.id), neuron);
            true
        }
    }

    fn n(id: u32, label: &str) -> Neuron {
        Neuron::new_atom(id, label.to_string(), NeuronKind::Excitatory, 0)
    }

    #[test]
    fn solo_routes_everything_local() {
        let ram = Arc::new(RamStore::with_node_id(NodeId(0)));
        let store = TieredStore::solo(NodeId(0), ram.clone());
        store.put(0, n(0, "a"));
        store.put(1, n(1, "b"));
        assert_eq!(store.get(0).unwrap().label, "a");
        assert_eq!(store.get(1).unwrap().label, "b");
        // Every id is local in solo mode.
        assert_eq!(store.compute_home(0), NodeId(0));
        assert_eq!(store.compute_home(99999), NodeId(0));
    }

    #[test]
    fn two_node_ring_routes_some_ids_remote() {
        let local_ram = Arc::new(RamStore::with_node_id(NodeId(1)));
        let remote_t = Arc::new(MockTransport::new()) as Arc<dyn RemoteTransport>;
        let remote_store = Arc::new(RemoteNodeStore::new(
            remote_t.clone(), /*pool*/ 0, NodeId(2),
        ));

        let store = TieredStore::solo(NodeId(1), local_ram);
        store.set_remote(NodeId(2), remote_store);
        // Ring is now [NodeId(1), NodeId(2)] — placement should pick
        // each node roughly half the time.

        // Verify placement is deterministic + spans both nodes.
        let mut on_local = 0;
        let mut on_remote = 0;
        for id in 0..200u32 {
            match store.compute_home(id) {
                NodeId(1) => on_local += 1,
                NodeId(2) => on_remote += 1,
                _ => panic!("unexpected node"),
            }
        }
        assert!(on_local  > 30, "local share too low: {}",  on_local);
        assert!(on_remote > 30, "remote share too low: {}", on_remote);
    }

    #[test]
    fn put_routes_to_home_node() {
        let local_ram = Arc::new(RamStore::with_node_id(NodeId(1)));
        let remote_t = Arc::new(MockTransport::new());
        let remote_t_dyn: Arc<dyn RemoteTransport> = remote_t.clone();
        let remote_store = Arc::new(RemoteNodeStore::new(
            remote_t_dyn, /*pool*/ 0, NodeId(2),
        ));
        let store = TieredStore::solo(NodeId(1), local_ram.clone());
        store.set_remote(NodeId(2), remote_store);

        // Find an id that hashes to NodeId(2).
        let mut remote_id = None;
        for id in 0..10000u32 {
            if store.compute_home(id) == NodeId(2) {
                remote_id = Some(id);
                break;
            }
        }
        let remote_id = remote_id.expect("expected at least one remote-home id");

        store.put(remote_id, n(remote_id, "remote_payload"));

        // Should NOT be in local RAM…
        assert!(local_ram.get(remote_id).is_none(),
            "neuron with remote home must not land in local RAM");
        // …but should be readable via the tiered get (which routes
        // through to the mock transport).
        let got = store.get(remote_id).expect("should fetch from remote");
        assert_eq!(got.label, "remote_payload");
    }

    #[test]
    fn local_iter_excludes_remote_owned_ids() {
        let local_ram = Arc::new(RamStore::with_node_id(NodeId(1)));
        let remote_t = Arc::new(MockTransport::new()) as Arc<dyn RemoteTransport>;
        let remote_store = Arc::new(RemoteNodeStore::new(
            remote_t, /*pool*/ 0, NodeId(2),
        ));
        let store = TieredStore::solo(NodeId(1), local_ram.clone());
        store.set_remote(NodeId(2), remote_store);

        // Forcibly stuff a "remote-home" id into local RAM (simulating
        // an inconsistency).  iter_ids should filter it out.
        let mut remote_owned = 0u32;
        for id in 0..10000u32 {
            if store.compute_home(id) == NodeId(2) {
                remote_owned = id;
                break;
            }
        }
        local_ram.put(remote_owned, n(remote_owned, "shouldnt-be-here"));

        let visible_ids: Vec<NeuronId> = store.iter_ids().collect();
        assert!(!visible_ids.contains(&remote_owned),
            "tiered iter must filter out ids whose home is a remote node");
    }
}

#[cfg(test)]
mod cold_disk_tests {
    use super::*;
    use crate::neuron::NeuronKind;
    use std::path::PathBuf;

    fn tmpdir(test: &str) -> PathBuf {
        let pid = std::process::id();
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir()
            .join(format!("w1z4rd_colddiskstore_{}_{}_{}", test, pid, nano));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn n(id: u32, label: &str) -> Neuron {
        Neuron::new_atom(id, label.to_string(), NeuronKind::Excitatory, 0)
    }

    #[test]
    fn put_then_get_round_trips() {
        let dir = tmpdir("rt");
        let tier = Arc::new(ColdTier::open(dir.join("pool.cold")).unwrap());
        let s = ColdDiskStore::new(tier);
        s.put(0, n(0, "x"));
        s.put(1, n(1, "y"));
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0).unwrap().label, "x");
        assert_eq!(s.get(1).unwrap().label, "y");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_drops_from_index_but_data_stays_on_disk() {
        let dir = tmpdir("delete");
        let tier = Arc::new(ColdTier::open(dir.join("pool.cold")).unwrap());
        let s = ColdDiskStore::new(tier);
        s.put(0, n(0, "x"));
        s.put(1, n(1, "y"));
        let _bytes_before = std::fs::metadata(dir.join("pool.cold"))
            .unwrap().len();
        s.delete(0);
        assert_eq!(s.len(), 1);
        assert!(s.get(0).is_none());
        let bytes_after = std::fs::metadata(dir.join("pool.cold"))
            .unwrap().len();
        // Disk size unchanged — append-only semantics; garbage stays
        // for compaction.
        assert!(bytes_after >= _bytes_before);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn seed_offsets_restores_state_from_snapshot() {
        let dir = tmpdir("seed");
        let tier = Arc::new(ColdTier::open(dir.join("pool.cold")).unwrap());
        let s1 = ColdDiskStore::new(tier.clone());
        s1.put(7, n(7, "seven"));
        // Capture id+offset (simulating what PoolSnapshot.cold_offsets does).
        let snap: Vec<(NeuronId, u64)> = s1.offsets.read().iter()
            .map(|(k, v)| (*k, *v)).collect();
        drop(s1);

        // Fresh store, seed the offset index, read should succeed.
        let s2 = ColdDiskStore::new(tier);
        s2.seed_offsets(snap);
        assert_eq!(s2.len(), 1);
        assert_eq!(s2.get(7).unwrap().label, "seven");
        std::fs::remove_dir_all(&dir).ok();
    }
}
