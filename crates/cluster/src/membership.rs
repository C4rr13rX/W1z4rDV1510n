//! Cluster membership registry, heartbeat tracking, and Bully-algorithm
//! leader election.
//!
//! The Bully algorithm:
//!   1. Any node that hasn't heard from the coordinator in COORD_TIMEOUT
//!      seconds calls an election by broadcasting ElectionPropose with its
//!      own priority (join_timestamp, so older = higher priority).
//!   2. Any node that receives an ElectionPropose with *lower* priority
//!      than itself overrides by sending its own ElectionPropose.
//!   3. The node with the highest priority that nobody overrides wins and
//!      broadcasts CoordinatorAnnounce.
//!   4. All nodes update their coordinator pointer on CoordinatorAnnounce.

use crate::protocol::{NodeId, NodeInfo};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

/// A node is considered dead after this many seconds without a heartbeat.
/// Raised to 90s to tolerate coordinators under heavy training/compute load.
pub const HEARTBEAT_TIMEOUT_SECS: u64 = 90;
/// Coordinator is considered dead after this many seconds without a heartbeat.
/// Raised to 120s for the same reason.
pub const COORD_TIMEOUT_SECS: u64 = 120;
/// Interval between outgoing heartbeat pulses.
pub const HEARTBEAT_INTERVAL_SECS: u64 = 5;
/// Time to wait for overriding proposals before declaring victory.
pub const ELECTION_WAIT_SECS: u64 = 3;

#[derive(Debug)]
struct NodeState {
    info:         NodeInfo,
    last_seen:    Instant,
}

/// In-memory roster of all known cluster members.
#[derive(Debug, Default)]
pub struct Membership {
    nodes:          HashMap<NodeId, NodeState>,
    coordinator_id: Option<NodeId>,
    local_id:       NodeId,
    local_join_ts:  u64,
}

impl Membership {
    pub fn new(local_id: NodeId, local_info: NodeInfo) -> Self {
        let join_ts = local_info.joined_at;
        let mut m = Membership {
            local_id:      local_id.clone(),
            local_join_ts: join_ts,
            ..Default::default()
        };
        m.upsert(local_info);
        m
    }

    pub fn upsert(&mut self, info: NodeInfo) {
        let id = info.id.clone();
        self.nodes.insert(id, NodeState { info, last_seen: Instant::now() });
    }

    pub fn touch(&mut self, id: &NodeId) {
        if let Some(s) = self.nodes.get_mut(id) {
            s.last_seen = Instant::now();
        }
    }

    pub fn remove(&mut self, id: &NodeId) {
        self.nodes.remove(id);
        if self.coordinator_id.as_ref() == Some(id) {
            self.coordinator_id = None;
        }
    }

    pub fn set_coordinator(&mut self, id: NodeId) {
        // Demote all first, then promote the winner.
        for ns in self.nodes.values_mut() {
            ns.info.is_coordinator = false;
        }
        if let Some(s) = self.nodes.get_mut(&id) {
            s.info.is_coordinator = true;
        }
        self.coordinator_id = Some(id);
    }

    pub fn coordinator_id(&self) -> Option<&NodeId> {
        self.coordinator_id.as_ref()
    }

    pub fn is_coordinator(&self) -> bool {
        self.coordinator_id.as_ref() == Some(&self.local_id)
    }

    pub fn local_id(&self) -> &NodeId {
        &self.local_id
    }

    /// Elapsed seconds since the coordinator last sent a heartbeat.
    /// Returns `None` if we have no coordinator or the coordinator is us.
    pub fn coordinator_silence_secs(&self) -> Option<u64> {
        let coord_id = self.coordinator_id.as_ref()?;
        if coord_id == &self.local_id {
            return None; // we are the coordinator
        }
        let state = self.nodes.get(coord_id)?;
        Some(state.last_seen.elapsed().as_secs())
    }

    /// Election priority score — higher wins.
    ///
    /// Composite of:
    ///   - CPU cores      (weight 40): more cores = better coordinator
    ///   - RAM            (weight 30): more RAM = better coordinator (score capped at 64 GiB)
    ///   - Uptime/seniority (weight 20): older node_ts = longer-lived = more stable
    ///   - Availability   (weight 10): tie-break by join timestamp (older = better)
    ///
    /// All components are normalised to 0–100 and combined, then scaled to u64
    /// so the wire format stays a single integer.
    pub fn local_priority(&self) -> u64 {
        let info = match self.nodes.get(&self.local_id) {
            Some(s) => &s.info,
            None => return 0,
        };
        let caps = &info.capabilities;

        // CPU: 1 core = 1 pt, capped at 128.
        let cpu_score = caps.cpu_cores.min(128) as f64;

        // RAM: score as GiB, capped at 64 GiB.
        let ram_gib = (caps.ram_bytes / (1024 * 1024 * 1024)) as f64;
        let ram_score = ram_gib.min(64.0);

        // Seniority: invert join_ts so older = larger score (capped at ~1 year of seconds).
        const YEAR_SECS: u64 = 365 * 24 * 3600;
        let age_score = YEAR_SECS.saturating_sub(self.local_join_ts % YEAR_SECS) as f64
            / YEAR_SECS as f64
            * 100.0;

        // Weighted sum.
        let score = cpu_score * 0.40 + ram_score * 0.30 + age_score * 0.20;

        // Scale to u64, preserving relative ordering.
        (score * 1_000_000.0) as u64
    }

    /// All live nodes (heartbeat seen within HEARTBEAT_TIMEOUT_SECS).
    pub fn live_nodes(&self) -> Vec<&NodeInfo> {
        let deadline = Duration::from_secs(HEARTBEAT_TIMEOUT_SECS);
        self.nodes
            .values()
            .filter(|s| s.last_seen.elapsed() < deadline)
            .map(|s| &s.info)
            .collect()
    }

    /// All nodes regardless of liveness.
    pub fn all_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes.values().map(|s| &s.info).collect()
    }

    /// Prune nodes that have timed out; returns the list of removed IDs.
    pub fn prune_dead(&mut self) -> Vec<NodeId> {
        let deadline = Duration::from_secs(HEARTBEAT_TIMEOUT_SECS);
        let dead: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(id, s)| *id != &self.local_id && s.last_seen.elapsed() > deadline)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &dead {
            self.nodes.remove(id);
            if self.coordinator_id.as_ref() == Some(id) {
                self.coordinator_id = None;
            }
        }
        dead
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_node(&self, id: &NodeId) -> Option<&NodeInfo> {
        self.nodes.get(id).map(|s| &s.info)
    }
}
