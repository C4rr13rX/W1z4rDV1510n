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
pub const HEARTBEAT_TIMEOUT_SECS: u64 = 15;
/// Coordinator is considered dead after this many seconds without a heartbeat.
pub const COORD_TIMEOUT_SECS: u64 = 20;
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

    /// Election priority: lower join timestamp = higher priority (older wins).
    /// Ties broken by node UUID (lexicographic, deterministic).
    pub fn local_priority(&self) -> u64 {
        // Invert join_ts so smaller = "older" = numerically larger priority.
        u64::MAX - self.local_join_ts
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
