use crate::config::{NetworkSecurityConfig, NodeNetworkConfig};
use crate::paths::node_data_dir;
use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic, MessageAcceptance, MessageAuthenticity};
use libp2p::identify;
use libp2p::kad::{self, store::MemoryStore};
use libp2p::mdns;
use libp2p::swarm::SwarmEvent;
use libp2p::{
    connection_limits, identity, multiaddr::Protocol, Multiaddr, PeerId, Swarm, SwarmBuilder,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn};
use w1z4rdv1510n::network::{NetworkEnvelope, NETWORK_ENVELOPE_VERSION};

const IDENTITY_KEY_FILE: &str = "p2p.key";

pub struct NodeNetwork {
    config: NodeNetworkConfig,
    peer_count: Arc<AtomicUsize>,
    command_tx: Option<mpsc::UnboundedSender<NetworkCommand>>,
}

impl NodeNetwork {
    pub fn new(config: NodeNetworkConfig) -> Self {
        Self {
            config,
            peer_count: Arc::new(AtomicUsize::new(0)),
            command_tx: None,
        }
    }

    pub fn start(&mut self, node_id: &str) -> Result<()> {
        if self.command_tx.is_some() {
            return Ok(());
        }
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        self.command_tx = Some(command_tx);
        let config = self.config.clone();
        let node_id = node_id.to_string();
        let peer_count = self.peer_count.clone();
        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    warn!(target: "w1z4rdv1510n::node", error = %err, "failed to start p2p runtime");
                    return;
                }
            };
            runtime.block_on(async move {
                if let Err(err) = run_swarm(config, node_id, peer_count, command_rx).await {
                    warn!(target: "w1z4rdv1510n::node", error = %err, "p2p swarm exited");
                }
            });
        });
        Ok(())
    }

    pub fn connect_bootstrap(&mut self) -> Result<()> {
        let peers = collect_bootstrap(&self.config)?;
        if peers.is_empty() {
            warn!(
                target: "w1z4rdv1510n::node",
                "no bootstrap peers configured; relying on mDNS discovery"
            );
            return Ok(());
        }
        let Some(tx) = &self.command_tx else {
            anyhow::bail!("p2p network not started");
        };
        for peer in peers {
            let _ = tx.send(NetworkCommand::Dial(peer));
        }
        Ok(())
    }

    pub fn peer_count(&self) -> usize {
        self.peer_count.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
enum NetworkCommand {
    Dial(Multiaddr),
}

#[derive(libp2p::swarm::NetworkBehaviour)]
#[behaviour(prelude = "libp2p::swarm::derive_prelude", out_event = "NodeBehaviourEvent")]
struct NodeBehaviour {
    gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
    kademlia: kad::Behaviour<MemoryStore>,
    mdns: mdns::tokio::Behaviour,
    limits: connection_limits::Behaviour,
}

#[derive(Debug)]
enum NodeBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Identify(identify::Event),
    Kademlia(kad::Event),
    Mdns(mdns::Event),
    Limits(void::Void),
}

impl From<gossipsub::Event> for NodeBehaviourEvent {
    fn from(event: gossipsub::Event) -> Self {
        NodeBehaviourEvent::Gossipsub(event)
    }
}

impl From<identify::Event> for NodeBehaviourEvent {
    fn from(event: identify::Event) -> Self {
        NodeBehaviourEvent::Identify(event)
    }
}

impl From<kad::Event> for NodeBehaviourEvent {
    fn from(event: kad::Event) -> Self {
        NodeBehaviourEvent::Kademlia(event)
    }
}

impl From<mdns::Event> for NodeBehaviourEvent {
    fn from(event: mdns::Event) -> Self {
        NodeBehaviourEvent::Mdns(event)
    }
}

impl From<void::Void> for NodeBehaviourEvent {
    fn from(event: void::Void) -> Self {
        match event {}
    }
}

async fn run_swarm(
    config: NodeNetworkConfig,
    node_id: String,
    peer_count: Arc<AtomicUsize>,
    mut command_rx: mpsc::UnboundedReceiver<NetworkCommand>,
) -> Result<()> {
    let identity = load_or_create_identity()?;
    let local_peer_id = PeerId::from(identity.public());
    let mut swarm = SwarmBuilder::with_existing_identity(identity)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_dns()?
        .with_behaviour(|key| {
            let gossipsub = build_gossipsub(key, &config)?;
            let identify = identify::Behaviour::new(identify::Config::new(
                "w1z4rdv1510n/1.0".to_string(),
                key.public(),
            ));
            let peer_id = PeerId::from(key.public());
            let mut kademlia = kad::Behaviour::new(peer_id, MemoryStore::new(peer_id));
            kademlia.set_mode(Some(kad::Mode::Client));
            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
            let limits = connection_limits::Behaviour::new(build_connection_limits(&config));
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(NodeBehaviour {
                gossipsub,
                identify,
                kademlia,
                mdns,
                limits,
            })
        })?
        .build();
    swarm
        .listen_on(parse_listen_addr(&config.listen_addr)?)
        .context("listen on network address")?;

    info!(
        target: "w1z4rdv1510n::node",
        peer_id = %local_peer_id,
        node_id = node_id.as_str(),
        "p2p swarm started"
    );

    let guard = Arc::new(Mutex::new(MessageGuard::new(&config.security)));

    loop {
        tokio::select! {
            Some(command) = command_rx.recv() => {
                match command {
                    NetworkCommand::Dial(addr) => {
                        if let Err(err) = swarm.dial(addr.clone()) {
                            warn!(target: "w1z4rdv1510n::node", error = %err, address = %addr, "dial failed");
                        }
                    }
                }
            }
            event = swarm.select_next_some() => {
                handle_swarm_event(event, &mut swarm, &peer_count, &config, &guard)?;
            }
        }
    }
}

fn handle_swarm_event(
    event: SwarmEvent<NodeBehaviourEvent>,
    swarm: &mut Swarm<NodeBehaviour>,
    peer_count: &Arc<AtomicUsize>,
    config: &NodeNetworkConfig,
    guard: &Arc<Mutex<MessageGuard>>,
) -> Result<()> {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            info!(
                target: "w1z4rdv1510n::node",
                address = %address,
                "p2p listening"
            );
        }
        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
            peer_count.fetch_add(1, Ordering::Relaxed);
            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
        }
        SwarmEvent::ConnectionClosed { .. } => {
            peer_count.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_sub(1))
            }).ok();
        }
        SwarmEvent::Behaviour(NodeBehaviourEvent::Gossipsub(event)) => {
            handle_gossipsub_event(event, swarm, config, guard);
        }
        SwarmEvent::Behaviour(NodeBehaviourEvent::Mdns(event)) => match event {
            mdns::Event::Discovered(list) => {
                for (peer_id, addr) in list {
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                    let _ = swarm.dial(addr);
                }
            }
            mdns::Event::Expired(list) => {
                for (peer_id, addr) in list {
                    swarm.behaviour_mut().kademlia.remove_address(&peer_id, &addr);
                }
            }
        },
        _ => {}
    }
    Ok(())
}

fn build_gossipsub(
    identity: &identity::Keypair,
    config: &NodeNetworkConfig,
) -> Result<gossipsub::Behaviour> {
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(10))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .validate_messages()
        .max_transmit_size(config.security.max_message_bytes)
        .max_messages_per_rpc(Some(config.security.max_messages_per_rpc))
        .message_id_fn(|message| message_id_for(message))
        .build()
        .map_err(|err| anyhow!("gossipsub config: {err}"))?;
    let mut gossipsub = gossipsub::Behaviour::new(
        MessageAuthenticity::Signed(identity.clone()),
        gossipsub_config,
    )
    .map_err(|err| anyhow!("gossipsub init: {err}"))?;
    let topic = IdentTopic::new(config.gossip_protocol.clone());
    gossipsub.subscribe(&topic)?;
    Ok(gossipsub)
}

fn handle_gossipsub_event(
    event: gossipsub::Event,
    swarm: &mut Swarm<NodeBehaviour>,
    config: &NodeNetworkConfig,
    guard: &Arc<Mutex<MessageGuard>>,
) {
    if let gossipsub::Event::Message {
        propagation_source,
        message_id,
        message,
    } = event
    {
        let acceptance = validate_message(&message, config, guard);
        let _ = swarm.behaviour_mut().gossipsub.report_message_validation_result(
            &message_id,
            &propagation_source,
            acceptance,
        );
    }
}

fn validate_message(
    message: &gossipsub::Message,
    config: &NodeNetworkConfig,
    guard: &Arc<Mutex<MessageGuard>>,
) -> MessageAcceptance {
    if message.data.len() > config.security.max_message_bytes {
        return MessageAcceptance::Reject;
    }
    if message.topic.as_str() != config.gossip_protocol {
        return MessageAcceptance::Reject;
    }
    let envelope: NetworkEnvelope = match serde_json::from_slice(&message.data) {
        Ok(envelope) => envelope,
        Err(_) => return MessageAcceptance::Reject,
    };
    if envelope.version != NETWORK_ENVELOPE_VERSION {
        return MessageAcceptance::Reject;
    }
    if envelope.validate_basic().is_err() {
        return MessageAcceptance::Reject;
    }
    let now = now_unix();
    if validate_envelope_clock(&envelope, config, now).is_err() {
        return MessageAcceptance::Reject;
    }
    if config.security.require_signed_payloads {
        if envelope.public_key.trim().is_empty() || envelope.signature.trim().is_empty() {
            return MessageAcceptance::Reject;
        }
        if verify_envelope_signature(&envelope).is_err() {
            return MessageAcceptance::Reject;
        }
    }
    match guard.lock() {
        Ok(mut guard) => {
            if guard.validate(&envelope, now).is_err() {
                return MessageAcceptance::Reject;
            }
        }
        Err(_) => return MessageAcceptance::Reject,
    }
    MessageAcceptance::Accept
}

fn build_connection_limits(config: &NodeNetworkConfig) -> connection_limits::ConnectionLimits {
    let total_limit = config
        .max_peers
        .min(config.security.max_established_total.max(1) as usize)
        .max(1) as u32;
    connection_limits::ConnectionLimits::default()
        .with_max_pending_incoming(Some(config.security.max_pending_incoming))
        .with_max_pending_outgoing(Some(config.security.max_pending_outgoing))
        .with_max_established_incoming(Some(config.security.max_established_incoming))
        .with_max_established_outgoing(Some(config.security.max_established_outgoing))
        .with_max_established(Some(total_limit))
        .with_max_established_per_peer(Some(config.security.max_established_per_peer))
}

fn message_id_for(message: &gossipsub::Message) -> gossipsub::MessageId {
    use blake2::{Blake2s256, Digest};
    let mut hasher = Blake2s256::new();
    hasher.update(message.topic.as_str().as_bytes());
    hasher.update(&message.data);
    let digest = hasher.finalize();
    gossipsub::MessageId::from(hex_encode(&digest))
}

struct MessageGuardConfig {
    max_seen_message_ids: usize,
    message_id_ttl_secs: i64,
    max_messages_per_key: u32,
    key_rate_window_secs: i64,
    max_tracked_public_keys: usize,
    public_key_ttl_secs: i64,
}

struct KeyWindow {
    window_start: i64,
    count: u32,
    last_seen: i64,
}

struct MessageGuard {
    config: MessageGuardConfig,
    recent_ids: VecDeque<(String, i64)>,
    recent_set: HashSet<String>,
    key_windows: HashMap<String, KeyWindow>,
    key_seen: VecDeque<(String, i64)>,
}

impl MessageGuard {
    fn new(config: &NetworkSecurityConfig) -> Self {
        Self {
            config: MessageGuardConfig {
                max_seen_message_ids: config.max_seen_message_ids.max(1),
                message_id_ttl_secs: config.message_id_ttl_secs.max(0),
                max_messages_per_key: config.max_messages_per_key_per_window.max(1),
                key_rate_window_secs: config.key_rate_window_secs.max(1),
                max_tracked_public_keys: config.max_tracked_public_keys.max(1),
                public_key_ttl_secs: config.public_key_ttl_secs.max(0),
            },
            recent_ids: VecDeque::new(),
            recent_set: HashSet::new(),
            key_windows: HashMap::new(),
            key_seen: VecDeque::new(),
        }
    }

    fn validate(&mut self, envelope: &NetworkEnvelope, now: i64) -> Result<()> {
        self.prune_ids(now);
        if self.recent_set.contains(&envelope.message_id) {
            anyhow::bail!("duplicate message id");
        }
        self.track_message_id(envelope.message_id.clone(), now);

        let key = if envelope.public_key.trim().is_empty() {
            "anonymous".to_string()
        } else {
            envelope.public_key.clone()
        };
        self.prune_keys(now);
        self.enforce_key_rate(&key, now)?;
        self.key_seen.push_back((key, now));
        self.prune_keys(now);
        Ok(())
    }

    fn enforce_key_rate(&mut self, key: &str, now: i64) -> Result<()> {
        let window = self
            .key_windows
            .entry(key.to_string())
            .or_insert_with(|| KeyWindow {
                window_start: now,
                count: 0,
                last_seen: now,
            });
        if now - window.window_start >= self.config.key_rate_window_secs {
            window.window_start = now;
            window.count = 0;
        }
        window.count = window.count.saturating_add(1);
        window.last_seen = now;
        if window.count > self.config.max_messages_per_key {
            anyhow::bail!("public key rate limit exceeded");
        }
        Ok(())
    }

    fn track_message_id(&mut self, message_id: String, now: i64) {
        self.recent_ids.push_back((message_id.clone(), now));
        self.recent_set.insert(message_id);
        let max_ids = self.config.max_seen_message_ids;
        while self.recent_ids.len() > max_ids {
            if let Some((id, _)) = self.recent_ids.pop_front() {
                self.recent_set.remove(&id);
            }
        }
    }

    fn prune_ids(&mut self, now: i64) {
        let ttl = self.config.message_id_ttl_secs;
        while let Some((id, ts)) = self.recent_ids.front() {
            if now - *ts <= ttl {
                break;
            }
            let id = id.clone();
            self.recent_ids.pop_front();
            self.recent_set.remove(&id);
        }
    }

    fn prune_keys(&mut self, now: i64) {
        let ttl = self.config.public_key_ttl_secs;
        while let Some((key, ts)) = self.key_seen.front() {
            if now - *ts <= ttl {
                break;
            }
            let key = key.clone();
            let ts = *ts;
            self.key_seen.pop_front();
            if let Some(entry) = self.key_windows.get(&key) {
                if entry.last_seen <= ts {
                    self.key_windows.remove(&key);
                }
            }
        }
        let max_keys = self.config.max_tracked_public_keys;
        while self.key_windows.len() > max_keys {
            if let Some((key, ts)) = self.key_seen.pop_front() {
                if let Some(entry) = self.key_windows.get(&key) {
                    if entry.last_seen <= ts {
                        self.key_windows.remove(&key);
                    }
                }
            } else {
                break;
            }
        }
    }
}

fn validate_envelope_clock(
    envelope: &NetworkEnvelope,
    config: &NodeNetworkConfig,
    now: i64,
) -> Result<()> {
    let max_age = config.security.max_message_age_secs;
    let max_skew = config.security.max_clock_skew_secs;
    if envelope.timestamp_unix > now + max_skew {
        anyhow::bail!("message timestamp too far in future");
    }
    let age = now - envelope.timestamp_unix;
    if age > max_age {
        anyhow::bail!("message timestamp too old");
    }
    Ok(())
}

fn verify_envelope_signature(envelope: &NetworkEnvelope) -> Result<()> {
    let public_key = decode_public_key(&envelope.public_key)?;
    let signature = decode_signature(&envelope.signature)?;
    let payload = envelope.signing_payload();
    public_key
        .verify(payload.as_bytes(), &signature)
        .map_err(|err| anyhow!("signature verify failed: {err}"))?;
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.as_bytes().iter().copied();
    while let Some(high) = iter.next() {
        let low = iter
            .next()
            .ok_or_else(|| anyhow!("hex string has odd length"))?;
        let high_val = hex_value(high)?;
        let low_val = hex_value(low)?;
        out.push((high_val << 4) | low_val);
    }
    Ok(out)
}

fn hex_value(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => anyhow::bail!("invalid hex character"),
    }
}

fn decode_public_key(hex: &str) -> Result<VerifyingKey> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("public key must be 32 bytes");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    VerifyingKey::from_bytes(&arr).map_err(|err| anyhow!("invalid public key: {err}"))
}

fn decode_signature(hex: &str) -> Result<Signature> {
    let bytes = hex_decode(hex)?;
    Signature::from_slice(&bytes).map_err(|err| anyhow!("invalid signature: {err}"))
}

fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_rejects_duplicate_message_ids() {
        let config = NetworkSecurityConfig::default();
        let mut guard = MessageGuard::new(&config);
        let envelope = NetworkEnvelope::new("test", b"payload", 100);
        guard.validate(&envelope, 100).expect("first ok");
        assert!(guard.validate(&envelope, 100).is_err());
    }

    #[test]
    fn guard_rate_limits_by_public_key() {
        let mut config = NetworkSecurityConfig::default();
        config.max_messages_per_key_per_window = 1;
        config.key_rate_window_secs = 60;
        let mut guard = MessageGuard::new(&config);

        let mut first = NetworkEnvelope::new("test", b"payload-1", 200);
        first.public_key = "key1".to_string();
        first.message_id = first.expected_message_id();
        guard.validate(&first, 200).expect("first ok");

        let mut second = NetworkEnvelope::new("test", b"payload-2", 201);
        second.public_key = "key1".to_string();
        second.message_id = second.expected_message_id();
        assert!(guard.validate(&second, 201).is_err());
    }
}

fn parse_listen_addr(value: &str) -> Result<Multiaddr> {
    if value.contains('/') {
        return value.parse().context("parse listen multiaddr");
    }
    let socket: SocketAddr = value.parse().context("parse listen socket")?;
    let mut addr = Multiaddr::empty();
    match socket {
        SocketAddr::V4(v4) => {
            addr.push(Protocol::Ip4(*v4.ip()));
            addr.push(Protocol::Tcp(v4.port()));
        }
        SocketAddr::V6(v6) => {
            addr.push(Protocol::Ip6(*v6.ip()));
            addr.push(Protocol::Tcp(v6.port()));
        }
    }
    Ok(addr)
}

fn collect_bootstrap(config: &NodeNetworkConfig) -> Result<Vec<Multiaddr>> {
    let mut peers = config.bootstrap_peers.clone();
    if peers.is_empty() {
        if let Ok(value) = std::env::var("W1Z4RDV1510N_BOOTSTRAP_PEERS") {
            for entry in value.split(',') {
                let trimmed = entry.trim();
                if !trimmed.is_empty() {
                    peers.push(trimmed.to_string());
                }
            }
        }
    }
    let mut addrs = Vec::new();
    for peer in peers {
        match peer.parse::<Multiaddr>() {
            Ok(addr) => addrs.push(addr),
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::node",
                    error = %err,
                    peer = peer.as_str(),
                    "invalid bootstrap multiaddr"
                );
            }
        }
    }
    Ok(addrs)
}

fn load_or_create_identity() -> Result<identity::Keypair> {
    let path = node_data_dir().join(IDENTITY_KEY_FILE);
    if path.exists() {
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let keypair = identity::Keypair::from_protobuf_encoding(&bytes)
            .map_err(|err| anyhow!("decode identity key: {err}"))?;
        return Ok(keypair);
    }
    let keypair = identity::Keypair::generate_ed25519();
    let encoded = keypair
        .to_protobuf_encoding()
        .map_err(|err| anyhow!("encode identity key: {err}"))?;
    write_binary(&path, &encoded)?;
    Ok(keypair)
}

fn write_binary(path: &PathBuf, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
