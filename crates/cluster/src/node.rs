//! The core `ClusterNode` type — the public API surface for this crate.
//!
//! Responsibilities:
//!   - Start as coordinator OR join an existing cluster via OTP.
//!   - Run the heartbeat loop and election loop in background tasks.
//!   - Accept incoming peer connections and route messages.
//!   - Expose `local_labels` / `remote_labels` so callers can split
//!     an observation across shards without knowing about the cluster.

use crate::{
    membership::{Membership, COORD_TIMEOUT_SECS, ELECTION_WAIT_SECS, HEARTBEAT_INTERVAL_SECS},
    otp::OtpRegistry,
    protocol::{self, local_capabilities, Message, NodeCapabilities, NodeId, NodeInfo, CLUSTER_PORT},
    ring::HashRing,
    transport::{accept_loop, ConnectionPool},
};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::{
    net::{SocketAddr, ToSocketAddrs},
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
    sync::{Mutex, RwLock},
};
use uuid::Uuid;

// ── Persistence ───────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct PersistedState {
    cluster_id:     String,
    bind_addr:      String,
    is_coordinator: bool,
}

fn state_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    let dir = std::path::Path::new(&home).join(".w1z4rd");
    std::fs::create_dir_all(&dir).ok();
    dir.join("cluster_state.json")
}

fn save_state(cluster_id: Uuid, bind_addr: SocketAddr, is_coordinator: bool) {
    let ps = PersistedState {
        cluster_id:     cluster_id.to_string(),
        bind_addr:      bind_addr.to_string(),
        is_coordinator,
    };
    if let Ok(json) = serde_json::to_string_pretty(&ps) {
        std::fs::write(state_path(), json).ok();
    }
}

fn load_saved_cluster_id() -> Option<(Uuid, bool)> {
    let data = std::fs::read_to_string(state_path()).ok()?;
    let ps: PersistedState = serde_json::from_str(&data).ok()?;
    let id = ps.cluster_id.parse().ok()?;
    Some((id, ps.is_coordinator))
}

// ── Config ────────────────────────────────────────────────────────────────────

pub struct ClusterConfig {
    pub bind_addr:          SocketAddr,
    /// The address advertised to other cluster members.  When `None` the node
    /// falls back to `bind_addr`, which means peers see `0.0.0.0` — only
    /// correct for loopback / single-machine clusters.  Set this to the
    /// machine's LAN IP so remote peers can actually reach it.
    pub advertise_addr:     Option<SocketAddr>,
    pub replication_factor: u8,
    pub otp_ttl_secs:       u64,
}

impl ClusterConfig {
    /// The address we tell other nodes to connect back to us on.
    ///
    /// Priority:
    ///   1. Explicit `advertise_addr` from config (always wins).
    ///   2. Auto-detected LAN IP — when `bind_addr` is `0.0.0.0` we ask the OS
    ///      which local interface it would use to reach an external host.  No
    ///      packets are sent; this is purely a routing-table lookup.
    ///   3. `bind_addr` verbatim (fallback — will be `0.0.0.0` only if detection
    ///      fails, which should never happen on a machine with network access).
    pub fn effective_addr(&self) -> SocketAddr {
        if let Some(addr) = self.advertise_addr {
            return addr;
        }
        let port = self.bind_addr.port();
        if !self.bind_addr.ip().is_unspecified() {
            return self.bind_addr;
        }
        // bind_addr is 0.0.0.0 — detect the real outbound IP.
        if let Some(ip) = detect_lan_ip() {
            return SocketAddr::new(ip, port);
        }
        self.bind_addr
    }
}

/// Ask the OS which local IP it would use to reach an external host.
/// Uses a UDP socket connected to 8.8.8.8:80 — no packets are sent,
/// the OS just resolves the route and fills in the local address.
fn detect_lan_ip() -> Option<std::net::IpAddr> {
    use std::net::UdpSocket;
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let local = socket.local_addr().ok()?;
    let ip = local.ip();
    // Reject loopback or another unspecified address.
    if ip.is_loopback() || ip.is_unspecified() {
        return None;
    }
    Some(ip)
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            bind_addr:          format!("0.0.0.0:{CLUSTER_PORT}").parse().unwrap(),
            advertise_addr:     None,
            replication_factor: 2,
            otp_ttl_secs:       600, // 10 minutes
        }
    }
}

// ── ClusterNode ───────────────────────────────────────────────────────────────

/// A running cluster node.  Clone is cheap (everything behind Arc).
#[derive(Clone)]
pub struct ClusterNode {
    pub local_id:    NodeId,
    pub cluster_id:  Uuid,
    membership:      Arc<RwLock<Membership>>,
    ring:            Arc<RwLock<HashRing>>,
    pool:            ConnectionPool,
    otp_registry:    Arc<Mutex<OtpRegistry>>,
    config:          Arc<ClusterConfig>,
    rep_factor:      u8,
}

impl ClusterNode {
    // ── Initialise as coordinator ─────────────────────────────────────────────

    /// Start a cluster.  Reuses a saved cluster_id if one exists on disk so
    /// the coordinator can restart without orphaning its workers.
    /// Returns `(node, otp_string)`.
    pub async fn init(config: ClusterConfig) -> anyhow::Result<(Self, String)> {
        let local_id   = NodeId::new();
        // Reuse the persisted cluster_id if this machine was previously coordinator,
        // so workers that were already joined don't see a foreign cluster.
        let cluster_id = load_saved_cluster_id()
            .filter(|(_, was_coord)| *was_coord)
            .map(|(id, _)| id)
            .unwrap_or_else(Uuid::new_v4);
        let caps       = local_capabilities();
        let join_ts    = unix_now();

        let local_info = NodeInfo {
            id:             local_id.clone(),
            addr:           config.effective_addr().to_string(),
            capabilities:   caps,
            joined_at:      join_ts,
            is_coordinator: true,
        };

        let mut membership = Membership::new(local_id.clone(), local_info);
        membership.set_coordinator(local_id.clone());

        let mut ring = HashRing::new();
        ring.add_node(&local_id);

        let node = Self {
            local_id:    local_id.clone(),
            cluster_id,
            membership:  Arc::new(RwLock::new(membership)),
            ring:        Arc::new(RwLock::new(ring)),
            pool:        ConnectionPool::new(),
            otp_registry: Arc::new(Mutex::new(OtpRegistry::default())),
            config:      Arc::new(config),
            rep_factor:  2,
        };

        let otp = node.generate_otp().await?;
        save_state(cluster_id, node.config.bind_addr, true);
        node.clone().run_background().await;
        Ok((node, otp))
    }

    // ── Join an existing cluster ──────────────────────────────────────────────

    /// Join an existing cluster.
    pub async fn join(
        config: ClusterConfig,
        coordinator_addr: SocketAddr,
        otp: &str,
    ) -> anyhow::Result<Self> {
        let local_id  = NodeId::new();
        let caps      = local_capabilities();
        let join_ts   = unix_now();

        // Connect to coordinator.
        let stream = TcpStream::connect(coordinator_addr)
            .await
            .context("connect to coordinator")?;
        stream.set_nodelay(true)?;
        let (r, w) = stream.into_split();
        let mut reader = BufReader::new(r);
        let mut writer = BufWriter::new(w);

        // Send Hello.
        let hello = Message::Hello {
            node_id:      local_id.clone(),
            otp_hash:     otp.to_string(),
            capabilities: caps.clone(),
            listen_addr:  config.effective_addr().to_string(),
        };
        protocol::send_msg(&mut writer, &hello).await?;

        // Await Welcome or Rejected.
        let resp = protocol::recv_msg(&mut reader).await?;
        match resp {
            Message::Welcome { cluster_id, roster, ring, replication_factor } => {
                let local_info = NodeInfo {
                    id:             local_id.clone(),
                    addr:           config.effective_addr().to_string(),
                    capabilities:   caps,
                    joined_at:      join_ts,
                    is_coordinator: false,
                };

                let membership_state = {
                    let mut m = Membership::new(local_id.clone(), local_info);
                    for ni in &roster {
                        m.upsert(ni.clone());
                        // Find coordinator.
                        if ni.is_coordinator {
                            m.set_coordinator(ni.id.clone());
                        }
                    }
                    m
                };

                let ring_state = HashRing::from_entries(&ring);

                let pool = ConnectionPool::new();
                for ni in &roster {
                    if ni.id == local_id { continue; }
                    if let Ok(addr) = ni.addr.parse::<SocketAddr>() {
                        pool.register(ni.id.clone(), addr);
                    }
                }

                let node = Self {
                    local_id:    local_id.clone(),
                    cluster_id,
                    membership:  Arc::new(RwLock::new(membership_state)),
                    ring:        Arc::new(RwLock::new(ring_state)),
                    pool,
                    otp_registry: Arc::new(Mutex::new(OtpRegistry::default())),
                    config:      Arc::new(config),
                    rep_factor:  replication_factor,
                };
                save_state(cluster_id, node.config.bind_addr, false);
                node.clone().run_background().await;
                Ok(node)
            }
            Message::Rejected { reason } => {
                anyhow::bail!("cluster rejected join: {reason}");
            }
            other => {
                anyhow::bail!("unexpected response: {:?}", other);
            }
        }
    }

    // ── OTP management ────────────────────────────────────────────────────────

    /// Generate a new OTP (coordinator only).
    pub async fn generate_otp(&self) -> anyhow::Result<String> {
        let mut reg = self.otp_registry.lock().await;
        reg.generate(Duration::from_secs(self.config.otp_ttl_secs))
    }

    /// Generate a fresh OTP on an already-running coordinator node.
    /// Returns `Err` if this node is not currently the coordinator.
    pub async fn new_otp(&self) -> anyhow::Result<String> {
        if !self.is_coordinator().await {
            anyhow::bail!("only the coordinator can generate OTPs");
        }
        self.generate_otp().await
    }

    /// Returns the persisted cluster_id and coordinator flag for the local machine,
    /// or None if this machine has never been in a cluster.
    pub fn saved_state() -> Option<(Uuid, bool)> {
        load_saved_cluster_id()
    }

    // ── Label routing (the core distributed-fabric API) ───────────────────────

    /// True if this node is the current coordinator.
    pub async fn is_coordinator(&self) -> bool {
        self.membership.read().await.is_coordinator()
    }

    /// Returns the labels that belong to THIS node's shard.
    pub async fn local_labels<'a>(&self, labels: &'a [String]) -> Vec<&'a String> {
        let ring = self.ring.read().await;
        labels
            .iter()
            .filter(|l| {
                ring.owner_of(l)
                    .map(|id| id == &self.local_id)
                    .unwrap_or(true) // empty ring → everything is local
            })
            .collect()
    }

    /// Returns the labels grouped by the remote node that owns each shard.
    pub async fn remote_labels<'a>(
        &self,
        labels: &'a [String],
    ) -> Vec<(NodeId, Vec<&'a String>)> {
        let ring = self.ring.read().await;
        let (_, remote) = ring.partition(labels, &self.local_id);
        remote
    }

    /// Forward a label-route message to a remote node (fire-and-forget).
    pub async fn forward_labels(
        &self,
        target: &NodeId,
        labels: Vec<String>,
        payload: serde_json::Value,
    ) {
        let msg = Message::LabelRoute {
            request_id: Uuid::new_v4(),
            labels,
            payload,
        };
        if let Err(e) = self.pool.send(target, &msg).await {
            tracing::warn!("label route to {target} failed: {e}");
        }
    }

    // ── Status ────────────────────────────────────────────────────────────────

    pub async fn status(&self) -> ClusterStatus {
        let m    = self.membership.read().await;
        let ring = self.ring.read().await;
        ClusterStatus {
            cluster_id:  self.cluster_id,
            local_id:    self.local_id.clone(),
            coordinator: m.coordinator_id().cloned(),
            nodes:       m.all_nodes().into_iter().cloned().collect(),
            ring_slots:  ring.len(),
        }
    }

    // ── Background tasks ──────────────────────────────────────────────────────

    async fn run_background(self) {
        let bind_addr = self.config.bind_addr;

        // Accept loop.
        let accept_node = self.clone();
        tokio::spawn(async move {
            let listener = match TcpListener::bind(bind_addr).await {
                Ok(l) => l,
                Err(e) => { tracing::error!("cluster bind {bind_addr}: {e}"); return; }
            };
            tracing::info!("cluster listening on {bind_addr} (SIGIL port 51611)");
            accept_loop(listener, move |stream, addr| {
                let n = accept_node.clone();
                async move { n.handle_incoming(stream, addr).await }
            }).await.ok();
        });

        // Heartbeat loop.
        let hb_node = self.clone();
        tokio::spawn(async move { hb_node.heartbeat_loop().await });

        // Watchdog / election loop.
        let wd_node = self.clone();
        tokio::spawn(async move { wd_node.watchdog_loop().await });
    }

    // ── Incoming connection handler ───────────────────────────────────────────

    async fn handle_incoming(&self, stream: TcpStream, _addr: SocketAddr) {
        let (r, w) = stream.into_split();
        let mut reader = BufReader::new(r);
        let mut writer = BufWriter::new(w);

        loop {
            let msg = match protocol::recv_msg(&mut reader).await {
                Ok(m)  => m,
                Err(_) => return,
            };
            if let Some(reply) = self.dispatch(msg).await {
                if protocol::send_msg(&mut writer, &reply).await.is_err() {
                    return;
                }
            }
        }
    }

    async fn dispatch(&self, msg: Message) -> Option<Message> {
        match msg {
            // ── Join handshake ────────────────────────────────────────────────
            Message::Hello { node_id, otp_hash, capabilities, listen_addr } => {
                let valid = {
                    let mut reg = self.otp_registry.lock().await;
                    reg.validate(&otp_hash)
                };
                if !valid {
                    return Some(Message::Rejected { reason: "invalid or expired OTP".into() });
                }
                let join_ts = unix_now();
                let new_info = NodeInfo {
                    id:             node_id.clone(),
                    addr:           listen_addr.clone(),
                    capabilities,
                    joined_at:      join_ts,
                    is_coordinator: false,
                };
                // Register in ring + membership.
                {
                    let mut ring = self.ring.write().await;
                    ring.add_node(&node_id);
                }
                {
                    let mut m = self.membership.write().await;
                    m.upsert(new_info.clone());
                }
                if let Ok(addr) = listen_addr.parse::<SocketAddr>() {
                    self.pool.register(node_id.clone(), addr);
                }
                // Tell everyone else about the new member.
                let ring_entries = self.ring.read().await.to_entries();
                self.pool.broadcast(&Message::MemberJoined {
                    node:  new_info.clone(),
                    ring:  ring_entries.clone(),
                }).await;

                // Build Welcome.
                let roster = self.membership.read().await.all_nodes().into_iter().cloned().collect();
                Some(Message::Welcome {
                    cluster_id:         self.cluster_id,
                    roster,
                    ring:               ring_entries,
                    replication_factor: self.rep_factor,
                })
            }

            // ── Heartbeat ─────────────────────────────────────────────────────
            Message::Heartbeat { node_id, .. } => {
                self.membership.write().await.touch(&node_id);
                Some(Message::HeartbeatAck { node_id: self.local_id.clone() })
            }
            Message::HeartbeatAck { node_id } => {
                self.membership.write().await.touch(&node_id);
                None
            }

            // ── Membership updates from coordinator ───────────────────────────
            Message::MemberJoined { node, ring } => {
                self.membership.write().await.upsert(node.clone());
                *self.ring.write().await = HashRing::from_entries(&ring);
                if let Ok(addr) = node.addr.parse::<SocketAddr>() {
                    self.pool.register(node.id, addr);
                }
                None
            }
            Message::MemberLeft { node_id, ring } => {
                self.membership.write().await.remove(&node_id);
                self.pool.remove(&node_id);
                *self.ring.write().await = HashRing::from_entries(&ring);
                None
            }

            // ── Election ──────────────────────────────────────────────────────
            Message::ElectionPropose { candidate_id, priority } => {
                let my_priority = self.membership.read().await.local_priority();
                if my_priority > priority {
                    // Override — propose ourselves.
                    self.pool.broadcast(&Message::ElectionPropose {
                        candidate_id: self.local_id.clone(),
                        priority:     my_priority,
                    }).await;
                }
                Some(Message::ElectionAck {
                    voter_id:  self.local_id.clone(),
                    for_id:    candidate_id,
                })
            }
            Message::CoordinatorAnnounce { coordinator_id } => {
                self.membership.write().await.set_coordinator(coordinator_id);
                None
            }

            // ── Label routing ─────────────────────────────────────────────────
            Message::LabelRoute { request_id, labels, payload } => {
                // The owning node processes these labels against its local shard.
                // Actual neuro processing is handled by the layer above us
                // (the node API) via a callback.  For now just ack.
                tracing::debug!("label route: {} labels (req {})", labels.len(), request_id);
                Some(Message::LabelRouteAck { request_id })
            }

            // ── Graceful departure ────────────────────────────────────────────
            Message::GracefulLeave { node_id } => {
                // Only the coordinator processes leave requests.
                if self.is_coordinator().await {
                    tracing::info!("graceful leave from {node_id}");
                    self.ring.write().await.remove_node(&node_id);
                    self.membership.write().await.remove(&node_id);
                    self.pool.remove(&node_id);
                    let ring_entries = self.ring.read().await.to_entries();
                    self.pool.broadcast(&Message::MemberLeft {
                        node_id,
                        ring: ring_entries,
                    }).await;
                }
                Some(Message::LeaveAck)
            }
            Message::ResignCoordinator { node_id } => {
                // Another node is triggering an election because the coordinator is resigning.
                tracing::info!("coordinator {node_id} is resigning — running election");
                self.run_election().await;
                None
            }

            // ── Status ────────────────────────────────────────────────────────
            Message::StatusRequest => {
                let m    = self.membership.read().await;
                let ring = self.ring.read().await;
                Some(Message::StatusResponse {
                    cluster_id:  self.cluster_id,
                    coordinator: m.coordinator_id().cloned().unwrap_or_else(NodeId::new),
                    nodes:       m.all_nodes().into_iter().cloned().collect(),
                    ring_size:   ring.len(),
                })
            }

            _ => None,
        }
    }

    // ── Graceful departure ────────────────────────────────────────────────────

    /// Leave the cluster gracefully as a worker.
    /// Notifies the coordinator, which redistributes this node's ring slots
    /// to the remaining nodes via consistent hashing and broadcasts MemberLeft.
    pub async fn leave(&self) -> anyhow::Result<()> {
        let coord_addr = {
            let m = self.membership.read().await;
            let coord_id = m.coordinator_id().cloned();
            drop(m);
            match coord_id {
                Some(id) if id != self.local_id => self.pool.addr_of(&id),
                _ => None,
            }
        };

        if let Some(addr) = coord_addr {
            // Tell coordinator we're leaving.
            match TcpStream::connect(addr).await {
                Ok(stream) => {
                    stream.set_nodelay(true).ok();
                    let (r, w) = stream.into_split();
                    let mut reader = BufReader::new(r);
                    let mut writer = BufWriter::new(w);
                    protocol::send_msg(&mut writer, &Message::GracefulLeave {
                        node_id: self.local_id.clone(),
                    }).await?;
                    // Wait for ack (or timeout).
                    let _ = tokio::time::timeout(
                        Duration::from_secs(5),
                        protocol::recv_msg(&mut reader),
                    ).await;
                }
                Err(e) => tracing::warn!("could not reach coordinator to send leave: {e}"),
            }
        } else {
            // We ARE the coordinator (single node or already standalone) —
            // nothing to notify, just clean up locally.
            tracing::info!("sole coordinator leaving — cluster dissolved");
        }

        // Clear local state so the node goes back to standalone.
        *self.membership.write().await = {
            let caps = protocol::local_capabilities();
            let info = NodeInfo {
                id:             self.local_id.clone(),
                addr:           self.config.effective_addr().to_string(),
                capabilities:   caps,
                joined_at:      unix_now(),
                is_coordinator: false,
            };
            crate::membership::Membership::new(self.local_id.clone(), info)
        };
        *self.ring.write().await = crate::ring::HashRing::new();
        self.pool.clear();
        // Remove persisted state so the node doesn't try to rejoin on restart.
        let _ = std::fs::remove_file(state_path());
        tracing::info!("left cluster — now standalone");
        Ok(())
    }

    /// Resign as coordinator: trigger a new election first so another node
    /// takes over cleanly, then leave the cluster.
    /// Ring slots owned by this node are redistributed to remaining peers.
    pub async fn resign(&self) -> anyhow::Result<()> {
        if !self.is_coordinator().await {
            anyhow::bail!("only the coordinator can resign");
        }
        tracing::info!("coordinator resigning — triggering election");
        // Broadcast resign so peers start electing immediately.
        self.pool.broadcast(&Message::ResignCoordinator {
            node_id: self.local_id.clone(),
        }).await;
        // Give peers time to elect a new coordinator.
        tokio::time::sleep(Duration::from_secs(ELECTION_WAIT_SECS + 1)).await;
        // Now leave as a regular worker.
        self.leave().await
    }

    // ── Heartbeat loop ────────────────────────────────────────────────────────

    async fn heartbeat_loop(&self) {
        let interval = Duration::from_secs(HEARTBEAT_INTERVAL_SECS);
        loop {
            tokio::time::sleep(interval).await;
            let msg = Message::Heartbeat {
                node_id: self.local_id.clone(),
                ts:      unix_now(),
            };
            self.pool.broadcast(&msg).await;

            // If coordinator: prune dead nodes and announce departures.
            if self.is_coordinator().await {
                let dead = self.membership.write().await.prune_dead();
                for id in dead {
                    self.ring.write().await.remove_node(&id);
                    self.pool.remove(&id);
                    let ring_entries = self.ring.read().await.to_entries();
                    self.pool.broadcast(&Message::MemberLeft {
                        node_id: id,
                        ring:    ring_entries,
                    }).await;
                }
            }
        }
    }

    // ── Watchdog / election loop ──────────────────────────────────────────────

    async fn watchdog_loop(&self) {
        loop {
            tokio::time::sleep(Duration::from_secs(HEARTBEAT_INTERVAL_SECS)).await;
            let silence = self.membership.read().await.coordinator_silence_secs();
            if let Some(secs) = silence {
                if secs >= COORD_TIMEOUT_SECS {
                    tracing::warn!("coordinator silent for {secs}s — starting election");
                    self.run_election().await;
                }
            }
        }
    }

    async fn run_election(&self) {
        let priority = self.membership.read().await.local_priority();
        let propose  = Message::ElectionPropose {
            candidate_id: self.local_id.clone(),
            priority,
        };
        self.pool.broadcast(&propose).await;
        // Wait for overrides.
        tokio::time::sleep(Duration::from_secs(ELECTION_WAIT_SECS)).await;
        // If no higher-priority node overrode us (no new coordinator set), we win.
        let coord = self.membership.read().await.coordinator_id().cloned();
        if coord.is_none() || coord.as_ref() == Some(&self.local_id) {
            tracing::info!("election won — becoming coordinator");
            self.membership.write().await.set_coordinator(self.local_id.clone());
            self.pool.broadcast(&Message::CoordinatorAnnounce {
                coordinator_id: self.local_id.clone(),
            }).await;
        }
    }
}

// ── Status snapshot ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ClusterStatus {
    pub cluster_id:  Uuid,
    pub local_id:    NodeId,
    pub coordinator: Option<NodeId>,
    pub nodes:       Vec<NodeInfo>,
    pub ring_slots:  usize,
}

impl std::fmt::Display for ClusterStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coord = self.coordinator.as_ref()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".into());
        writeln!(f, "Cluster  : {}", self.cluster_id)?;
        writeln!(f, "Local    : {}", self.local_id)?;
        writeln!(f, "Coord    : {coord}")?;
        writeln!(f, "Ring     : {} virtual slots", self.ring_slots)?;
        writeln!(f, "Nodes    : {}", self.nodes.len())?;
        for n in &self.nodes {
            let role = if n.is_coordinator { " [coordinator]" } else { "" };
            writeln!(f, "  {} @ {} ({} cores, {}){}", n.id, n.addr, n.capabilities.cpu_cores, n.capabilities.os, role)?;
        }
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
