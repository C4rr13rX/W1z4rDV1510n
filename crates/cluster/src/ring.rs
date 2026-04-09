//! Consistent-hash ring for fabric shard assignment.
//!
//! Each physical node gets `VIRTUAL_NODES` positions on the ring so the
//! distribution stays even as nodes join and leave.  Labels are mapped to
//! owners by `hash(label) → walk clockwise → first token ≥ hash`.
//!
//! The ring is purely deterministic; given the same set of NodeIds it always
//! produces the same assignment.  Rebalancing on join/leave is automatic.

use blake2::{Blake2b512, Digest};
use std::collections::BTreeMap;
use crate::protocol::NodeId;
use serde::{Deserialize, Serialize};

/// Virtual nodes per physical node.  Higher = more even distribution.
const VIRTUAL_NODES: u32 = 150;

/// A position → NodeId mapping that implements the ring.
#[derive(Debug, Clone, Default)]
pub struct HashRing {
    /// Sorted: token → NodeId.
    ring: BTreeMap<u64, NodeId>,
}

impl HashRing {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the ring (creates VIRTUAL_NODES virtual positions).
    pub fn add_node(&mut self, id: &NodeId) {
        for i in 0..VIRTUAL_NODES {
            let token = vnode_token(id, i);
            self.ring.insert(token, id.clone());
        }
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, id: &NodeId) {
        for i in 0..VIRTUAL_NODES {
            let token = vnode_token(id, i);
            self.ring.remove(&token);
        }
    }

    /// Return the NodeId that owns the given label.
    /// Returns `None` only if the ring is empty.
    pub fn owner_of(&self, label: &str) -> Option<&NodeId> {
        if self.ring.is_empty() {
            return None;
        }
        let h = label_token(label);
        // Walk clockwise: first entry with token >= h, wrapping around.
        self.ring
            .range(h..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id)
    }

    /// Partition a set of labels into (local_labels, remote_labels_by_node).
    pub fn partition<'a>(
        &self,
        labels: &'a [String],
        local_id: &NodeId,
    ) -> (Vec<&'a String>, Vec<(NodeId, Vec<&'a String>)>) {
        use std::collections::HashMap;
        let mut local  = Vec::new();
        let mut remote: HashMap<NodeId, Vec<&'a String>> = HashMap::new();

        for label in labels {
            match self.owner_of(label) {
                Some(owner) if owner == local_id => local.push(label),
                Some(owner) => remote.entry(owner.clone()).or_default().push(label),
                None        => local.push(label), // degenerate: no ring yet
            }
        }
        (local, remote.into_iter().collect())
    }

    /// Serialise the ring as a Vec of (token, node_id) for the wire protocol.
    pub fn to_entries(&self) -> Vec<crate::protocol::RingEntry> {
        self.ring
            .iter()
            .map(|(&token, id)| crate::protocol::RingEntry {
                token,
                node_id: id.clone(),
            })
            .collect()
    }

    /// Rebuild the ring from wire-protocol entries.
    pub fn from_entries(entries: &[crate::protocol::RingEntry]) -> Self {
        let mut ring = BTreeMap::new();
        for e in entries {
            ring.insert(e.token, e.node_id.clone());
        }
        Self { ring }
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }
}

// ── Hashing helpers ───────────────────────────────────────────────────────────

fn label_token(label: &str) -> u64 {
    hash_bytes(label.as_bytes())
}

fn vnode_token(id: &NodeId, vnode: u32) -> u64 {
    let mut data = id.0.as_bytes().to_vec();
    data.extend_from_slice(&vnode.to_le_bytes());
    hash_bytes(&data)
}

fn hash_bytes(data: &[u8]) -> u64 {
    let mut h = Blake2b512::new();
    h.update(data);
    let out = h.finalize();
    u64::from_le_bytes(out[..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_empty_returns_none() {
        let ring = HashRing::new();
        assert!(ring.owner_of("chess::e4").is_none());
    }

    #[test]
    fn ring_single_node_owns_everything() {
        let id = NodeId::new();
        let mut ring = HashRing::new();
        ring.add_node(&id);
        assert_eq!(ring.owner_of("chess::e4"), Some(&id));
        assert_eq!(ring.owner_of("qa::what_is_entropy"), Some(&id));
    }

    #[test]
    fn ring_two_nodes_splits_labels() {
        let id1 = NodeId::new();
        let id2 = NodeId::new();
        let mut ring = HashRing::new();
        ring.add_node(&id1);
        ring.add_node(&id2);

        let labels: Vec<String> = (0..200).map(|i| format!("label_{i}")).collect();
        let owned_by_1 = labels.iter().filter(|l| ring.owner_of(l) == Some(&id1)).count();
        let owned_by_2 = labels.iter().filter(|l| ring.owner_of(l) == Some(&id2)).count();
        // With 150 virtual nodes each, distribution should be roughly 50/50.
        assert!(owned_by_1 > 50 && owned_by_2 > 50, "bad split: {owned_by_1}/{owned_by_2}");
    }

    #[test]
    fn remove_node_reroutes_labels() {
        let id1 = NodeId::new();
        let id2 = NodeId::new();
        let mut ring = HashRing::new();
        ring.add_node(&id1);
        ring.add_node(&id2);
        ring.remove_node(&id2);
        // After removal, everything should go to id1.
        let label = "chess::e4";
        assert_eq!(ring.owner_of(label), Some(&id1));
    }
}
