//! Wire protocol — length-prefixed JSON frames over raw TCP.
//!
//! Frame layout:
//!   [4 bytes LE u32 = body length][body bytes (UTF-8 JSON)]
//!
//! All messages are tagged with a `type` field so both sides can dispatch
//! without a separate framing header.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use uuid::Uuid;

/// Port every W1z4rD cluster node listens on by default.
/// 51611  →  SIGIL in leet (5=S 1=I 6=G 1=I 1=L).
pub const CLUSTER_PORT: u16 = 51_611;

/// Maximum frame body size (16 MiB).  Prevents memory exhaustion on bad input.
pub const MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;

// ── Node identity ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0.to_string()[..8])
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Static capabilities a node advertises on join.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub hostname:   String,
    pub os:         String,
    pub cpu_cores:  u32,
    pub ram_bytes:  u64,
    pub has_gpu:    bool,
}

/// A node as seen by the cluster roster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id:           NodeId,
    pub addr:         String,   // "ip:port"
    pub capabilities: NodeCapabilities,
    pub joined_at:    u64,      // unix seconds
    pub is_coordinator: bool,
}

// ── Ring snapshot ─────────────────────────────────────────────────────────────

/// One virtual-node entry on the consistent-hash ring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingEntry {
    pub token:   u64,    // position on the ring
    pub node_id: NodeId,
}

// ── Distributed training types ────────────────────────────────────────────────

/// One synapse weight exported for cross-node synchronisation.
/// Uses label strings rather than local neuron IDs so the delta is portable
/// across nodes that may have assigned different IDs to the same concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseDelta {
    /// Label of the pre-synaptic (source) neuron.
    pub from_label: String,
    /// Label of the post-synaptic (target) neuron.
    pub to_label: String,
    /// Absolute weight value (not a diff) — recipient takes max(local, remote).
    pub weight: f32,
    pub inhibitory: bool,
    /// Number of dopamine-flush captures this synapse has accumulated.
    /// Recipients take max(local, remote) so consolidation history
    /// accumulates across the cluster.  Serde default 0 keeps the wire
    /// format backward-compatible with peers running older builds.
    #[serde(default)]
    pub consolidation: u8,
}

/// Compact snapshot of weight changes produced by one node since its last sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolDelta {
    /// Pool step counter at the start of this window.
    pub from_step: u64,
    /// Pool step counter at the end of this window.
    pub to_step: u64,
    /// Changed synapses.
    pub synapses: Vec<SynapseDelta>,
    /// Changed co-occurrence rates: (label_a, label_b, rate).
    pub cooccur: Vec<(String, String, f32)>,
}

// ── Messages ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    // ── Join handshake ────────────────────────────────────────────────────────
    /// New node → coordinator: request to join.
    Hello {
        node_id:      NodeId,
        otp_hash:     String,  // argon2 hash of the OTP the coordinator issued
        capabilities: NodeCapabilities,
        listen_addr:  String,
    },
    /// Coordinator → new node: accepted, here is the full cluster state.
    Welcome {
        cluster_id:      Uuid,
        roster:          Vec<NodeInfo>,
        ring:            Vec<RingEntry>,
        replication_factor: u8,
    },
    /// Coordinator → new node: rejected.
    Rejected { reason: String },

    // ── Gossip / membership ───────────────────────────────────────────────────
    /// Any node → all peers: periodic keepalive.
    Heartbeat { node_id: NodeId, ts: u64 },
    HeartbeatAck { node_id: NodeId },

    /// Coordinator → all: a node has joined the cluster.
    MemberJoined { node: NodeInfo, ring: Vec<RingEntry> },
    /// Coordinator → all: a node has left or timed out.
    MemberLeft { node_id: NodeId, ring: Vec<RingEntry> },

    // ── Leader election (Bully algorithm) ─────────────────────────────────────
    /// A node proposes itself as coordinator.
    ElectionPropose { candidate_id: NodeId, priority: u64 },
    /// Peers acknowledge a proposal (higher-priority node can override).
    ElectionAck { voter_id: NodeId, for_id: NodeId },
    /// Winning candidate announces itself.
    CoordinatorAnnounce { coordinator_id: NodeId },

    // ── Neural fabric routing ─────────────────────────────────────────────────
    /// Route a subset of an observation's labels to the owning shard node.
    LabelRoute {
        request_id: Uuid,
        labels:     Vec<String>,
        /// Serialised EnvironmentSnapshot JSON — only the fields needed.
        payload:    serde_json::Value,
    },
    LabelRouteAck { request_id: Uuid },

    /// Write a shard state delta to a replica.
    ShardWrite {
        request_id: Uuid,
        shard_key:  String,
        delta:      serde_json::Value,
    },
    ShardWriteAck { request_id: Uuid },

    // ── Distributed training ─────────────────────────────────────────────────
    /// Coordinator → peer: forward a /media/train payload so the peer trains it.
    /// Used for round-robin load distribution — each page trains on exactly one
    /// node; weight sync later propagates the knowledge everywhere.
    TrainForward {
        request_id: Uuid,
        /// Serialised MediaTrainReq JSON body — peer deserialises and trains.
        payload: serde_json::Value,
    },
    /// Peer → coordinator: result of a forwarded training call.
    TrainForwardAck {
        request_id: Uuid,
        success: bool,
        label_count: usize,
    },

    /// Node → peers: push weight deltas after a training window.
    /// Recipients merge using max(local, remote) for each synapse weight so
    /// knowledge only accumulates — it never regresses.
    WeightDelta {
        request_id: Uuid,
        delta: PoolDelta,
    },
    /// Peer → sender: delta received and applied.
    WeightDeltaAck {
        request_id: Uuid,
        /// Number of synapses merged.
        applied: usize,
    },

    // ── Graceful departure ────────────────────────────────────────────────────
    /// A node tells the coordinator it is leaving cleanly.
    /// The coordinator removes it from the ring and broadcasts MemberLeft.
    /// Ring slots are redistributed to remaining nodes by consistent hashing.
    GracefulLeave { node_id: NodeId },
    /// Coordinator → departing node: departure acknowledged, ring updated.
    LeaveAck,
    /// Coordinator is stepping down — triggers election so another node takes
    /// over before the coordinator disconnects.
    ResignCoordinator { node_id: NodeId },

    // ── Cluster management (human-facing) ─────────────────────────────────────
    /// Ask the coordinator for the current status.
    StatusRequest,
    StatusResponse {
        cluster_id:  Uuid,
        coordinator: NodeId,
        nodes:       Vec<NodeInfo>,
        ring_size:   usize,
    },
}

// ── Framing helpers ───────────────────────────────────────────────────────────

/// Write one framed message to any async writer.
pub async fn send_msg<W: AsyncWrite + Unpin>(w: &mut W, msg: &Message) -> anyhow::Result<()> {
    let body = serde_json::to_vec(msg)?;
    let len = body.len() as u32;
    if len > MAX_FRAME_BYTES {
        anyhow::bail!("message too large: {} bytes", len);
    }
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await?;
    Ok(())
}

/// Read one framed message from any async reader.
pub async fn recv_msg<R: AsyncRead + Unpin>(r: &mut R) -> anyhow::Result<Message> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        anyhow::bail!("incoming frame too large: {} bytes", len);
    }
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body).await?;
    Ok(serde_json::from_slice(&body)?)
}

// ── Capability detection ──────────────────────────────────────────────────────

/// Detect the local machine's capabilities at runtime.
pub fn local_capabilities() -> NodeCapabilities {
    NodeCapabilities {
        hostname:  hostname(),
        os:        std::env::consts::OS.to_string(),
        cpu_cores: num_cpus(),
        ram_bytes: ram_bytes(),
        has_gpu:   false, // extend later
    }
}

fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn ram_bytes() -> u64 {
    // Best-effort; falls back to 0 on platforms where we can't detect it.
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        return kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
            }
        }
    }
    0
}
