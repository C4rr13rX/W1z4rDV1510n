//! Connection pool for node-to-node TCP connections.
//!
//! Each peer gets one long-lived TCP stream.  If a send fails the connection
//! is torn down and rebuilt on the next attempt.

use crate::protocol::{self, Message, NodeId};
use anyhow::Context;
use dashmap::DashMap;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncWrite, BufReader, BufWriter},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};

/// Timeout for establishing a new peer connection.
/// Prevents the coordinator's broadcast from hanging when a joining node's
/// port isn't bound yet, which would otherwise block prune_dead from running.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// A pooled connection to a peer.
struct Conn {
    addr:   SocketAddr,
    writer: Mutex<BufWriter<tokio::net::tcp::OwnedWriteHalf>>,
    reader: Mutex<BufReader<tokio::net::tcp::OwnedReadHalf>>,
}

impl Conn {
    async fn connect(addr: SocketAddr) -> anyhow::Result<Arc<Self>> {
        let stream = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr))
            .await
            .with_context(|| format!("connect to {addr}: timed out"))?
            .with_context(|| format!("connect to {addr}"))?;
        stream.set_nodelay(true)?;
        let (r, w) = stream.into_split();
        Ok(Arc::new(Conn {
            addr,
            writer: Mutex::new(BufWriter::new(w)),
            reader: Mutex::new(BufReader::new(r)),
        }))
    }

    async fn send(&self, msg: &Message) -> anyhow::Result<()> {
        let mut w = self.writer.lock().await;
        protocol::send_msg(&mut *w, msg).await
    }

    async fn recv(&self) -> anyhow::Result<Message> {
        let mut r = self.reader.lock().await;
        protocol::recv_msg(&mut *r).await
    }
}

/// Thread-safe pool of peer connections keyed by NodeId.
#[derive(Clone, Default)]
pub struct ConnectionPool {
    conns: Arc<DashMap<NodeId, Arc<Conn>>>,
    addrs: Arc<DashMap<NodeId, SocketAddr>>,
}

impl ConnectionPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a peer address (without connecting yet).
    pub fn register(&self, id: NodeId, addr: SocketAddr) {
        self.addrs.insert(id, addr);
    }

    /// Remove a peer.
    pub fn remove(&self, id: &NodeId) {
        self.conns.remove(id);
        self.addrs.remove(id);
    }

    /// Look up the registered address for a peer without connecting.
    pub fn addr_of(&self, id: &NodeId) -> Option<SocketAddr> {
        self.addrs.get(id).map(|e| *e)
    }

    /// Disconnect all peers and clear the pool (used on graceful leave).
    pub fn clear(&self) {
        self.conns.clear();
        self.addrs.clear();
    }

    /// Send a message to a peer, reconnecting if needed.
    pub async fn send(&self, id: &NodeId, msg: &Message) -> anyhow::Result<()> {
        let conn = self.get_or_connect(id).await?;
        if let Err(e) = conn.send(msg).await {
            self.conns.remove(id);
            return Err(e);
        }
        Ok(())
    }

    /// Send a message and wait for exactly one reply.
    pub async fn request(&self, id: &NodeId, msg: &Message) -> anyhow::Result<Message> {
        let conn = self.get_or_connect(id).await?;
        conn.send(msg).await.map_err(|e| { self.conns.remove(id); e })?;
        conn.recv().await.map_err(|e| { self.conns.remove(id); e })
    }

    /// Broadcast to all registered peers (best-effort; errors logged, not returned).
    pub async fn broadcast(&self, msg: &Message) {
        let ids: Vec<NodeId> = self.addrs.iter().map(|e| e.key().clone()).collect();
        for id in ids {
            if let Err(e) = self.send(&id, msg).await {
                tracing::warn!("broadcast to {} failed: {e}", id);
            }
        }
    }

    /// Broadcast to all registered peers except one (best-effort; errors logged).
    /// Used when the coordinator notifies existing members of a new join — the
    /// joining node already received full state via Welcome and must not receive
    /// a MemberJoined broadcast before its accept_loop is bound.
    pub async fn broadcast_except(&self, exclude: &NodeId, msg: &Message) {
        let ids: Vec<NodeId> = self.addrs.iter()
            .filter(|e| e.key() != exclude)
            .map(|e| e.key().clone())
            .collect();
        for id in ids {
            if let Err(e) = self.send(&id, msg).await {
                tracing::warn!("broadcast to {} failed: {e}", id);
            }
        }
    }

    async fn get_or_connect(&self, id: &NodeId) -> anyhow::Result<Arc<Conn>> {
        if let Some(conn) = self.conns.get(id) {
            return Ok(conn.clone());
        }
        let addr = self.addrs.get(id)
            .ok_or_else(|| anyhow::anyhow!("unknown peer {}", id))?
            .clone();
        let conn = Conn::connect(addr).await?;
        self.conns.insert(id.clone(), conn.clone());
        Ok(conn)
    }
}

/// Accept incoming connections in a loop, dispatching each to `handler`.
pub async fn accept_loop<F, Fut>(
    listener: TcpListener,
    handler: F,
) -> anyhow::Result<()>
where
    F: Fn(TcpStream, SocketAddr) -> Fut + Clone + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                stream.set_nodelay(true).ok();
                let h = handler.clone();
                tokio::spawn(async move { h(stream, addr).await });
            }
            Err(e) => {
                tracing::warn!("accept error: {e}");
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}
