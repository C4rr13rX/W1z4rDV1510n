//! Distributed training coordinator.
//!
//! Makes the cluster behave as a single logical training VM:
//!
//!  • **Round-robin routing** — incoming `/media/train` calls are forwarded to
//!    peers in turn so the compute load is spread evenly across all N nodes.
//!    Each page trains on exactly one node.
//!
//!  • **Weight-delta sync** — every `SYNC_EVERY` training calls the node that
//!    did the work pushes a compact label-keyed delta to all other nodes via
//!    their `POST /neuro/delta/apply` endpoint.  Recipients merge with
//!    `max(local, remote)` so knowledge only accumulates.
//!
//!  • **QA broadcast** — every `/qa/ingest` call is fanned out to all peers so
//!    the QA store is fully replicated across the cluster instantly.
//!
//! No node is special once the cluster is formed.  Any node can receive any
//! API call; the coordinator role (for routing purposes) is determined by
//! checking whether this node is the cluster leader.

use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};
use w1z4rdv1510n::neuro::SynapseDelta;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Push a weight delta to all peers after every N local training calls.
/// Lower = better convergence, higher network traffic.
/// Higher = less traffic, longer before peers learn this node's training.
const SYNC_EVERY: u64 = 50;

/// HTTP client timeout for peer calls (training forwards + delta pushes).
const PEER_TIMEOUT_SECS: u64 = 60;

// ── Types ─────────────────────────────────────────────────────────────────────

/// HTTP address of a peer node (e.g. `"http://100.x.x.x:8090"`).
pub type PeerAddr = String;

/// Payload sent to `POST /neuro/delta/apply` on each peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaApplyReq {
    pub from_step: u64,
    pub to_step:   u64,
    pub synapses:  Vec<SynapseDelta>,
    pub cooccur:   Vec<(String, String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaApplyResp {
    pub applied: usize,
}

// ── Coordinator ───────────────────────────────────────────────────────────────

/// Shared state managed by the distributed coordinator.
pub struct DistributedCoordinator {
    /// HTTP base URLs of all known peers (excludes self).
    /// Updated whenever the cluster membership changes.
    peers: Arc<RwLock<Vec<PeerAddr>>>,

    /// Round-robin index for routing training calls.
    rr: Arc<AtomicUsize>,

    /// Number of training calls routed or executed locally since last sync.
    calls_since_sync: Arc<AtomicU64>,

    /// Pool step at the last delta sync.
    last_sync_step: Arc<AtomicU64>,

    /// Shared HTTP client — reused across all peer calls.
    client: reqwest::Client,
}

impl DistributedCoordinator {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(PEER_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client");
        Self {
            peers:            Arc::new(RwLock::new(Vec::new())),
            rr:               Arc::new(AtomicUsize::new(0)),
            calls_since_sync: Arc::new(AtomicU64::new(0)),
            last_sync_step:   Arc::new(AtomicU64::new(0)),
            client,
        }
    }

    // ── Peer list management ──────────────────────────────────────────────────

    /// Replace the peer list with fresh addresses from the cluster roster.
    /// Called after each cluster membership change.
    pub async fn set_peers(&self, addrs: Vec<PeerAddr>) {
        let mut guard = self.peers.write().await;
        *guard = addrs;
    }

    pub async fn peer_count(&self) -> usize {
        self.peers.read().await.len()
    }

    // ── Training routing ──────────────────────────────────────────────────────

    /// Return the peer that should handle the next training call, or `None` if
    /// this turn belongs to the local node.
    ///
    /// The rotation has N+1 slots — one per peer plus one for self — so every
    /// node in the cluster trains roughly 1/(N+1) of all calls.  Without self
    /// in the rotation the coordinator would train 0 % of the load while
    /// workers accumulate all knowledge, causing the coordinator's neuro pool
    /// to drift indefinitely.
    pub async fn next_train_peer(&self) -> Option<PeerAddr> {
        let peers = self.peers.read().await;
        if peers.is_empty() {
            return None;
        }
        // Slot 0 → train locally; slots 1..=N → forward to peers[slot-1].
        let total = peers.len() + 1;
        let idx = self.rr.fetch_add(1, Ordering::Relaxed) % total;
        if idx == 0 { None } else { Some(peers[idx - 1].clone()) }
    }

    /// Forward a raw JSON training body to a specific peer's `/media/train`.
    /// Returns `true` if the peer accepted and trained successfully.
    pub async fn forward_train(
        &self,
        peer: &str,
        body: serde_json::Value,
    ) -> bool {
        let url = format!("{peer}/media/train");
        match self.client.post(&url).header("x-w1z-local", "1").json(&body).send().await {
            Ok(resp) if resp.status().is_success() => true,
            Ok(resp) => {
                warn!("forward_train to {peer} returned {}", resp.status());
                false
            }
            Err(e) => {
                warn!("forward_train to {peer} failed: {e}");
                false
            }
        }
    }

    // ── Weight-delta sync ─────────────────────────────────────────────────────

    /// Increment the call counter and, if `SYNC_EVERY` calls have elapsed,
    /// push a weight delta to all peers.
    ///
    /// `neuro` must implement the `export_delta_since` / `pool_step` methods
    /// added to `NeuroRuntime`.
    pub async fn maybe_sync<N>(&self, neuro: &N)
    where
        N: NeuroDeltaSource + Send + Sync,
    {
        let n = self.calls_since_sync.fetch_add(1, Ordering::Relaxed) + 1;
        if n < SYNC_EVERY {
            return;
        }
        self.calls_since_sync.store(0, Ordering::Relaxed);
        self.push_delta_to_peers(neuro).await;
    }

    /// Unconditionally export and push the current delta to all peers.
    /// Useful for a forced sync triggered by `POST /neuro/sync`.
    pub async fn force_sync<N>(&self, neuro: &N)
    where
        N: NeuroDeltaSource + Send + Sync,
    {
        self.calls_since_sync.store(0, Ordering::Relaxed);
        self.push_delta_to_peers(neuro).await;
    }

    async fn push_delta_to_peers<N>(&self, neuro: &N)
    where
        N: NeuroDeltaSource + Send + Sync,
    {
        let peers = self.peers.read().await.clone();
        if peers.is_empty() {
            return;
        }

        let since_step = self.last_sync_step.load(Ordering::Relaxed);
        let (synapses, cooccur) = neuro.export_delta_since(since_step);
        let to_step = neuro.pool_step();

        if synapses.is_empty() && cooccur.is_empty() {
            debug!("sync: no delta since step {since_step}, skipping");
            self.last_sync_step.store(to_step, Ordering::Relaxed);
            return;
        }

        debug!(
            "sync: pushing {} synapses + {} cooccur entries to {} peers",
            synapses.len(),
            cooccur.len(),
            peers.len()
        );

        let payload = DeltaApplyReq { from_step: since_step, to_step, synapses, cooccur };

        // Fan out to all peers concurrently — fire-and-forget; don't block
        // the caller's response on slow peers.
        let client = self.client.clone();
        let peers_clone = peers.clone();
        let payload_clone = payload.clone();
        tokio::spawn(async move {
            let mut handles = Vec::with_capacity(peers_clone.len());
            for peer in &peers_clone {
                let url    = format!("{peer}/neuro/delta/apply");
                let c      = client.clone();
                let p      = payload_clone.clone();
                handles.push(tokio::spawn(async move {
                    match c.post(&url).json(&p).send().await {
                        Ok(r) if r.status().is_success() => {
                            if let Ok(resp) = r.json::<DeltaApplyResp>().await {
                                debug!("delta applied on {url}: {} synapses", resp.applied);
                            }
                        }
                        Ok(r)  => warn!("delta push to {url} failed: {}", r.status()),
                        Err(e) => warn!("delta push to {url} error: {e}"),
                    }
                }));
            }
            for h in handles { let _ = h.await; }
        });

        self.last_sync_step.store(to_step, Ordering::Relaxed);
    }

    // ── QA broadcast ─────────────────────────────────────────────────────────

    /// Fan-out a QA ingest body to all peers' `/qa/ingest` endpoints.
    /// Fire-and-forget — does not block the caller.
    pub async fn broadcast_qa_ingest(&self, body: serde_json::Value) {
        let peers = self.peers.read().await.clone();
        if peers.is_empty() { return; }
        let client = self.client.clone();
        tokio::spawn(async move {
            for peer in &peers {
                let url = format!("{peer}/qa/ingest");
                let c = client.clone();
                let b = body.clone();
                tokio::spawn(async move {
                    if let Err(e) = c.post(&url).header("x-w1z-local", "1").json(&b).send().await {
                        warn!("qa broadcast to {url} failed: {e}");
                    }
                });
            }
        });
    }

    // ── New-worker bootstrap ──────────────────────────────────────────────────

    /// Called after a worker joins the cluster.  Asks each known peer to:
    ///   1. Refresh their own cluster/status (so they add the new worker to
    ///      their distributed peer list), then
    ///   2. Push their current weight delta to all peers (including the new
    ///      worker), seeding it with the cluster's accumulated knowledge.
    ///
    /// Fire-and-forget — does not block the join response.
    pub async fn bootstrap_from_peers(&self) {
        let peers = self.peers.read().await.clone();
        if peers.is_empty() { return; }
        let client = self.client.clone();
        tokio::spawn(async move {
            // Short delay so the coordinator's TCP accept loop has registered
            // the new member before we ask it to push.
            tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
            for peer in &peers {
                // Trigger peer's cluster/status so it includes us in its peer list.
                let _ = client.get(format!("{peer}/cluster/status")).send().await;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
            for peer in &peers {
                // Ask peer to push its weight delta to all its peers (now including us).
                let _ = client.post(format!("{peer}/neuro/sync")).send().await;
            }
        });
    }

    // ── Status ────────────────────────────────────────────────────────────────

    pub async fn status(&self) -> serde_json::Value {
        let peers         = self.peers.read().await.clone();
        let last_sync     = self.last_sync_step.load(Ordering::Relaxed);
        let calls_pending = self.calls_since_sync.load(Ordering::Relaxed);
        let rr_pos        = self.rr.load(Ordering::Relaxed);
        serde_json::json!({
            "peers": peers,
            "last_sync_step": last_sync,
            "calls_since_last_sync": calls_pending,
            "rr_position": rr_pos,
            "sync_every": SYNC_EVERY,
        })
    }
}

impl Default for DistributedCoordinator {
    fn default() -> Self { Self::new() }
}

// ── Trait for neuro handle injection ─────────────────────────────────────────

/// Abstraction over `NeuroRuntimeHandle` so the coordinator doesn't need to
/// import the full core crate directly (avoids circular deps in tests).
pub trait NeuroDeltaSource {
    fn export_delta_since(
        &self,
        since_step: u64,
    ) -> (Vec<SynapseDelta>, Vec<(String, String, f32)>);
    fn pool_step(&self) -> u64;
}

impl NeuroDeltaSource for w1z4rdv1510n::neuro::NeuroRuntime {
    fn export_delta_since(
        &self,
        since_step: u64,
    ) -> (Vec<SynapseDelta>, Vec<(String, String, f32)>) {
        w1z4rdv1510n::neuro::NeuroRuntime::export_delta_since(self, since_step)
    }
    fn pool_step(&self) -> u64 {
        w1z4rdv1510n::neuro::NeuroRuntime::pool_step(self)
    }
}
