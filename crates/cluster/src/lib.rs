//! W1z4rD Cluster — distributed neural fabric across n machines.
//!
//! Default port: 51611 (SIGIL in leet: 5=S 1=I 6=G 1=I 1=L)
//!
//! # Quick start
//!
//! ```text
//! # Machine A — start a new cluster
//! w1z4rd_node cluster init
//! # prints: Cluster ready.  OTP: RAVEN-7834  (expires 10m)
//!
//! # Machine B — join
//! w1z4rd_node cluster join --coordinator 192.168.1.10 --otp RAVEN-7834
//!
//! # Any machine — status
//! w1z4rd_node cluster status --coordinator 192.168.1.10
//! ```
//!
//! Once joined, each node owns a shard of the neural fabric determined by the
//! consistent-hash ring.  Sensor observations are split at the label level:
//! labels owned by the local shard are processed in-process; labels owned by
//! remote shards are forwarded over TCP on port 51611.
//!
//! # Fault tolerance
//!
//! The coordinator continues to do neural work — it is just a node with the
//! extra duty of managing membership and the hash ring.  If the coordinator
//! dies, surviving nodes run a Bully-algorithm election (oldest node wins)
//! and announce a new coordinator automatically.  The cluster keeps processing
//! throughout; only ring-rebalancing is paused until the election settles
//! (~ELECTION_WAIT_SECS = 3 seconds).
//!
//! # Replication
//!
//! Each shard write is forwarded to one replica node (replication factor 2).
//! On primary failure the replica takes over ownership automatically via the
//! ring rebalance that happens when the dead node is pruned.

pub mod membership;
pub mod node;
pub mod otp;
pub mod protocol;
pub mod ring;
pub mod transport;

pub use node::{ClusterConfig, ClusterNode, ClusterStatus};
pub use protocol::CLUSTER_PORT;
