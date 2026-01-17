use crate::config::{NetworkSecurityConfig, NodeNetworkConfig, NodeNetworkRoutingConfig};
use crate::paths::node_data_dir;
use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use futures::StreamExt;
use libp2p::autonat;
use libp2p::gossipsub::{
    self, IdentTopic, MessageAcceptance, MessageAuthenticity, PeerScoreParams,
    PeerScoreThresholds, TopicScoreParams,
};
use libp2p::identify;
use libp2p::kad::{self, store::MemoryStore};
use libp2p::mdns;
use libp2p::relay;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{ConnectionId, SwarmEvent};
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
    message_rx: Option<mpsc::UnboundedReceiver<NetworkEnvelope>>,
}

impl NodeNetwork {
    pub fn new(config: NodeNetworkConfig) -> Self {
        Self {
            config,
            peer_count: Arc::new(AtomicUsize::new(0)),
            command_tx: None,
            message_rx: None,
        }
    }

    pub fn start(&mut self, node_id: &str) -> Result<()> {
        if self.command_tx.is_some() {
            return Ok(());
        }
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        self.command_tx = Some(command_tx);
        self.message_rx = Some(message_rx);
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
                if let Err(err) = run_swarm(config, node_id, peer_count, command_rx, message_tx).await {
                    warn!(target: "w1z4rdv1510n::node", error = %err, "p2p swarm exited");
                }
            });
        });
        Ok(())
    }

    pub fn connect_bootstrap(&mut self) -> Result<()> {
        let mut peers = collect_bootstrap(&self.config)?;
        if self.config.routing.enable_relay {
            let mut relays = collect_relays(&self.config)?;
            peers.append(&mut relays);
        }
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

    pub fn publisher(&self) -> Option<NetworkPublisher> {
        self.command_tx
            .as_ref()
            .map(|tx| NetworkPublisher { tx: tx.clone() })
    }

    pub fn take_message_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<NetworkEnvelope>> {
        self.message_rx.take()
    }
}

#[derive(Clone)]
pub struct NetworkPublisher {
    tx: mpsc::UnboundedSender<NetworkCommand>,
}

impl NetworkPublisher {
    pub fn publish(&self, envelope: NetworkEnvelope) -> Result<()> {
        self.tx
            .send(NetworkCommand::Publish(envelope))
            .map_err(|_| anyhow!("p2p command channel closed"))
    }
}

#[cfg(test)]
pub(crate) fn test_publisher() -> NetworkPublisher {
    let (tx, _rx) = mpsc::unbounded_channel::<NetworkCommand>();
    NetworkPublisher { tx }
}

#[derive(Debug)]
enum NetworkCommand {
    Dial(Multiaddr),
    Publish(NetworkEnvelope),
}

#[derive(libp2p::swarm::NetworkBehaviour)]
#[behaviour(prelude = "libp2p::swarm::derive_prelude", out_event = "NodeBehaviourEvent")]
struct NodeBehaviour {
    gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
    kademlia: kad::Behaviour<MemoryStore>,
    mdns: mdns::tokio::Behaviour,
    relay: relay::client::Behaviour,
    autonat: autonat::Behaviour,
    limits: connection_limits::Behaviour,
}

#[derive(Debug)]
enum NodeBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Identify(identify::Event),
    Kademlia(kad::Event),
    Mdns(mdns::Event),
    Relay(relay::client::Event),
    Autonat(autonat::Event),
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

impl From<relay::client::Event> for NodeBehaviourEvent {
    fn from(event: relay::client::Event) -> Self {
        NodeBehaviourEvent::Relay(event)
    }
}

impl From<autonat::Event> for NodeBehaviourEvent {
    fn from(event: autonat::Event) -> Self {
        NodeBehaviourEvent::Autonat(event)
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
    message_tx: mpsc::UnboundedSender<NetworkEnvelope>,
) -> Result<()> {
    let identity = load_or_create_identity()?;
    let local_peer_id = PeerId::from(identity.public());
    let mut swarm = if config.routing.enable_quic {
        build_swarm_with_quic(identity, &config)?
    } else {
        build_swarm_with_tcp(identity, &config)?
    };
    swarm
        .listen_on(parse_listen_addr(&config.listen_addr)?)
        .context("listen on network address")?;
    if config.routing.enable_quic {
        if let Some(addr) = quic_listen_addr(&config.listen_addr)? {
            if let Err(err) = swarm.listen_on(addr.clone()) {
                warn!(
                    target: "w1z4rdv1510n::node",
                    error = %err,
                    address = %addr,
                    "failed to listen on quic address"
                );
            }
        }
    }
    for addr in &config.routing.external_addresses {
        match addr.parse::<Multiaddr>() {
            Ok(parsed) => swarm.add_external_address(parsed),
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::node",
                    error = %err,
                    address = addr.as_str(),
                    "invalid external address"
                );
            }
        }
    }
    if config.routing.enable_relay && config.routing.reserve_relays {
        if let Ok(relays) = collect_relays(&config) {
            for relay in relays {
                let listen_addr = relay_reservation_addr(&relay);
                if let Err(err) = swarm.listen_on(listen_addr.clone()) {
                    warn!(
                        target: "w1z4rdv1510n::node",
                        error = %err,
                        address = %listen_addr,
                        "failed to reserve relay address"
                    );
                }
            }
        }
    }
    if config.routing.enable_autonat {
        if let Ok(relays) = collect_relays(&config) {
            for relay in relays {
                if let Some(peer_id) = peer_id_from_multiaddr(&relay) {
                    swarm
                        .behaviour_mut()
                        .autonat
                        .add_server(peer_id, Some(relay.clone()));
                }
            }
        }
    }

    info!(
        target: "w1z4rdv1510n::node",
        peer_id = %local_peer_id,
        node_id = node_id.as_str(),
        "p2p swarm started"
    );

    let guard = Arc::new(Mutex::new(MessageGuard::new(&config.security)));
    let topic = IdentTopic::new(config.gossip_protocol.clone());
    let mut dial_backoff = DialBackoff::new(&config.routing);
    let mut pending_dials = HashMap::new();

    loop {
        tokio::select! {
            Some(command) = command_rx.recv() => {
                match command {
                    NetworkCommand::Dial(addr) => {
                        dial_with_backoff(
                            &mut swarm,
                            &mut dial_backoff,
                            &mut pending_dials,
                            addr,
                        );
                    }
                    NetworkCommand::Publish(envelope) => {
                        if let Ok(payload) = serde_json::to_vec(&envelope) {
                            if payload.len() > config.security.max_message_bytes {
                                warn!(
                                    target: "w1z4rdv1510n::node",
                                    size = payload.len(),
                                    "dropping publish: payload exceeds max_message_bytes"
                                );
                            } else if let Err(err) = swarm.behaviour_mut().gossipsub.publish(topic.clone(), payload) {
                                warn!(
                                    target: "w1z4rdv1510n::node",
                                    error = %err,
                                    "gossipsub publish failed"
                                );
                            }
                        }
                    }
                }
            }
            event = swarm.select_next_some() => {
                handle_swarm_event(
                    event,
                    &mut swarm,
                    &peer_count,
                    &config,
                    &guard,
                    &message_tx,
                    &mut dial_backoff,
                    &mut pending_dials,
                )?;
            }
        }
    }
}

fn build_swarm_with_tcp(
    identity: identity::Keypair,
    config: &NodeNetworkConfig,
) -> Result<Swarm<NodeBehaviour>> {
    let config_clone = config.clone();
    let swarm = SwarmBuilder::with_existing_identity(identity)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_dns()?
        .with_relay_client(
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_behaviour(move |key, relay| build_behaviour(key, relay, &config_clone))?
        .build();
    Ok(swarm)
}

fn build_swarm_with_quic(
    identity: identity::Keypair,
    config: &NodeNetworkConfig,
) -> Result<Swarm<NodeBehaviour>> {
    let config_clone = config.clone();
    let swarm = SwarmBuilder::with_existing_identity(identity)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_quic()
        .with_dns()?
        .with_relay_client(
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_behaviour(move |key, relay| build_behaviour(key, relay, &config_clone))?
        .build();
    Ok(swarm)
}

fn build_behaviour(
    key: &identity::Keypair,
    relay: relay::client::Behaviour,
    config: &NodeNetworkConfig,
) -> Result<NodeBehaviour, Box<dyn std::error::Error + Send + Sync>> {
    let gossipsub = build_gossipsub(key, config).map_err(|err| -> Box<dyn std::error::Error + Send + Sync> {
        err.into()
    })?;
    let identify = identify::Behaviour::new(identify::Config::new(
        "w1z4rdv1510n/1.0".to_string(),
        key.public(),
    ));
    let peer_id = PeerId::from(key.public());
    let mut kademlia = kad::Behaviour::new(peer_id, MemoryStore::new(peer_id));
    kademlia.set_mode(Some(kad::Mode::Client));
    let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)
        .map_err(|err| -> Box<dyn std::error::Error + Send + Sync> { err.into() })?;
    let autonat_config = build_autonat_config(config);
    let autonat = autonat::Behaviour::new(peer_id, autonat_config);
    let limits = connection_limits::Behaviour::new(build_connection_limits(config));
    Ok(NodeBehaviour {
        gossipsub,
        identify,
        kademlia,
        mdns,
        relay,
        autonat,
        limits,
    })
}

fn build_autonat_config(config: &NodeNetworkConfig) -> autonat::Config {
    let mut autonat_config = autonat::Config::default();
    if !config.routing.enable_autonat {
        autonat_config.use_connected = false;
        let delay = Duration::from_secs(365 * 24 * 60 * 60);
        autonat_config.boot_delay = delay;
        autonat_config.retry_interval = delay;
        autonat_config.refresh_interval = delay;
    }
    autonat_config
}

fn handle_swarm_event(
    event: SwarmEvent<NodeBehaviourEvent>,
    swarm: &mut Swarm<NodeBehaviour>,
    peer_count: &Arc<AtomicUsize>,
    config: &NodeNetworkConfig,
    guard: &Arc<Mutex<MessageGuard>>,
    message_tx: &mpsc::UnboundedSender<NetworkEnvelope>,
    dial_backoff: &mut DialBackoff,
    pending_dials: &mut HashMap<ConnectionId, Multiaddr>,
) -> Result<()> {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            info!(
                target: "w1z4rdv1510n::node",
                address = %address,
                "p2p listening"
            );
        }
        SwarmEvent::ConnectionEstablished { peer_id, connection_id, .. } => {
            peer_count.fetch_add(1, Ordering::Relaxed);
            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
            if let Some(addr) = pending_dials.remove(&connection_id) {
                dial_backoff.record_success(&addr);
            }
        }
        SwarmEvent::ConnectionClosed { .. } => {
            peer_count.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_sub(1))
            }).ok();
        }
        SwarmEvent::OutgoingConnectionError { connection_id, error, .. } => {
            if let Some(addr) = pending_dials.remove(&connection_id) {
                let delay = dial_backoff.record_failure(&addr);
                warn!(
                    target: "w1z4rdv1510n::node",
                    error = %error,
                    address = %addr,
                    backoff_secs = delay,
                    "outgoing dial failed"
                );
            }
        }
        SwarmEvent::NewExternalAddrCandidate { address } => {
            if config.routing.use_observed_addresses {
                swarm.add_external_address(address);
            }
        }
        SwarmEvent::ExternalAddrConfirmed { address } => {
            if config.routing.use_observed_addresses {
                swarm.add_external_address(address);
            }
        }
        SwarmEvent::Behaviour(NodeBehaviourEvent::Gossipsub(event)) => {
            handle_gossipsub_event(event, swarm, config, guard, message_tx);
        }
        SwarmEvent::Behaviour(NodeBehaviourEvent::Identify(event)) => {
            if let identify::Event::Received { info, .. } = event {
                if config.routing.use_observed_addresses {
                    swarm.add_external_address(info.observed_addr);
                }
            }
        }
        SwarmEvent::Behaviour(NodeBehaviourEvent::Mdns(event)) => match event {
            mdns::Event::Discovered(list) => {
                for (peer_id, addr) in list {
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                    dial_with_backoff(
                        swarm,
                        dial_backoff,
                        pending_dials,
                        addr,
                    );
                }
            }
            mdns::Event::Expired(list) => {
                for (peer_id, addr) in list {
                    swarm.behaviour_mut().kademlia.remove_address(&peer_id, &addr);
                }
            }
        },
        SwarmEvent::Behaviour(NodeBehaviourEvent::Relay(_event)) => {}
        SwarmEvent::Behaviour(NodeBehaviourEvent::Autonat(event)) => {
            if let autonat::Event::StatusChanged { new, .. } = event {
                if let autonat::NatStatus::Public(address) = new {
                    if config.routing.use_observed_addresses {
                        swarm.add_external_address(address);
                    }
                }
            }
        }
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
    if config.routing.enable_peer_scoring {
        let mut params = PeerScoreParams::default();
        params.topics.insert(topic.hash(), TopicScoreParams::default());
        let thresholds = PeerScoreThresholds::default();
        gossipsub
            .with_peer_score(params, thresholds)
            .map_err(|err| anyhow!("gossipsub peer score init: {err}"))?;
    }
    Ok(gossipsub)
}

struct DialBackoffEntry {
    failures: u32,
    next_allowed_unix: i64,
}

struct DialBackoff {
    base_secs: i64,
    max_secs: i64,
    entries: HashMap<String, DialBackoffEntry>,
}

impl DialBackoff {
    fn new(config: &NodeNetworkRoutingConfig) -> Self {
        Self {
            base_secs: config.dial_backoff_base_secs as i64,
            max_secs: config.dial_backoff_max_secs as i64,
            entries: HashMap::new(),
        }
    }

    fn allow(&self, addr: &Multiaddr, now: i64) -> bool {
        self.entries
            .get(&addr.to_string())
            .map(|entry| now >= entry.next_allowed_unix)
            .unwrap_or(true)
    }

    fn remaining(&self, addr: &Multiaddr, now: i64) -> i64 {
        self.entries
            .get(&addr.to_string())
            .map(|entry| (entry.next_allowed_unix - now).max(0))
            .unwrap_or(0)
    }

    fn record_failure(&mut self, addr: &Multiaddr) -> i64 {
        let now = now_unix();
        let entry = self
            .entries
            .entry(addr.to_string())
            .or_insert(DialBackoffEntry {
                failures: 0,
                next_allowed_unix: now,
            });
        entry.failures = entry.failures.saturating_add(1);
        let mut delay = self.base_secs.max(1);
        for _ in 1..entry.failures {
            delay = delay.saturating_mul(2).min(self.max_secs);
        }
        delay = delay.min(self.max_secs);
        entry.next_allowed_unix = now.saturating_add(delay);
        delay
    }

    fn record_success(&mut self, addr: &Multiaddr) {
        self.entries.remove(&addr.to_string());
    }
}

fn dial_with_backoff(
    swarm: &mut Swarm<NodeBehaviour>,
    dial_backoff: &mut DialBackoff,
    pending_dials: &mut HashMap<ConnectionId, Multiaddr>,
    addr: Multiaddr,
) {
    let now = now_unix();
    if !dial_backoff.allow(&addr, now) {
        let remaining = dial_backoff.remaining(&addr, now);
        warn!(
            target: "w1z4rdv1510n::node",
            address = %addr,
            backoff_remaining = remaining,
            "dial suppressed due to backoff"
        );
        return;
    }
    let opts = dial_opts_for_address(addr.clone());
    let connection_id = opts.connection_id();
    pending_dials.insert(connection_id, addr.clone());
    if let Err(err) = swarm.dial(opts) {
        pending_dials.remove(&connection_id);
        let delay = dial_backoff.record_failure(&addr);
        warn!(
            target: "w1z4rdv1510n::node",
            error = %err,
            address = %addr,
            backoff_secs = delay,
            "dial failed"
        );
    }
}

fn dial_opts_for_address(addr: Multiaddr) -> DialOpts {
    if let Some(peer_id) = peer_id_from_multiaddr(&addr) {
        DialOpts::peer_id(peer_id)
            .addresses(vec![addr])
            .extend_addresses_through_behaviour()
            .build()
    } else {
        DialOpts::unknown_peer_id().address(addr).build()
    }
}

fn handle_gossipsub_event(
    event: gossipsub::Event,
    swarm: &mut Swarm<NodeBehaviour>,
    config: &NodeNetworkConfig,
    guard: &Arc<Mutex<MessageGuard>>,
    message_tx: &mpsc::UnboundedSender<NetworkEnvelope>,
) {
    if let gossipsub::Event::Message {
        propagation_source,
        message_id,
        message,
    } = event
    {
        let (acceptance, envelope) = validate_message(&message, config, guard);
        let accepted = matches!(acceptance, MessageAcceptance::Accept);
        let _ = swarm.behaviour_mut().gossipsub.report_message_validation_result(
            &message_id,
            &propagation_source,
            acceptance,
        );
        if accepted {
            if let Some(envelope) = envelope {
                let _ = message_tx.send(envelope);
            }
        }
    }
}

fn validate_message(
    message: &gossipsub::Message,
    config: &NodeNetworkConfig,
    guard: &Arc<Mutex<MessageGuard>>,
) -> (MessageAcceptance, Option<NetworkEnvelope>) {
    if message.data.len() > config.security.max_message_bytes {
        return (MessageAcceptance::Reject, None);
    }
    if message.topic.as_str() != config.gossip_protocol {
        return (MessageAcceptance::Reject, None);
    }
    let envelope: NetworkEnvelope = match serde_json::from_slice(&message.data) {
        Ok(envelope) => envelope,
        Err(_) => return (MessageAcceptance::Reject, None),
    };
    if envelope.version != NETWORK_ENVELOPE_VERSION {
        return (MessageAcceptance::Reject, None);
    }
    if envelope.validate_basic().is_err() {
        return (MessageAcceptance::Reject, None);
    }
    let now = now_unix();
    if validate_envelope_clock(&envelope, config, now).is_err() {
        return (MessageAcceptance::Reject, None);
    }
    if config.security.require_signed_payloads {
        if envelope.public_key.trim().is_empty() || envelope.signature.trim().is_empty() {
            return (MessageAcceptance::Reject, None);
        }
        if verify_envelope_signature(&envelope).is_err() {
            return (MessageAcceptance::Reject, None);
        }
    }
    match guard.lock() {
        Ok(mut guard) => {
            if guard.validate(&envelope, now).is_err() {
                return (MessageAcceptance::Reject, None);
            }
        }
        Err(_) => return (MessageAcceptance::Reject, None),
    }
    (MessageAcceptance::Accept, Some(envelope))
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

fn quic_listen_addr(value: &str) -> Result<Option<Multiaddr>> {
    if value.contains('/') {
        let addr: Multiaddr = value.parse().context("parse quic listen multiaddr")?;
        let has_udp = addr.iter().any(|p| matches!(p, Protocol::Udp(_)));
        let has_quic = addr.iter().any(|p| matches!(p, Protocol::QuicV1));
        if has_udp && has_quic {
            return Ok(Some(addr));
        }
        return Ok(None);
    }
    let socket: SocketAddr = value.parse().context("parse quic listen socket")?;
    let mut addr = Multiaddr::empty();
    match socket {
        SocketAddr::V4(v4) => {
            addr.push(Protocol::Ip4(*v4.ip()));
            addr.push(Protocol::Udp(v4.port()));
        }
        SocketAddr::V6(v6) => {
            addr.push(Protocol::Ip6(*v6.ip()));
            addr.push(Protocol::Udp(v6.port()));
        }
    }
    addr.push(Protocol::QuicV1);
    Ok(Some(addr))
}

fn relay_reservation_addr(relay: &Multiaddr) -> Multiaddr {
    let mut addr = relay.clone();
    let has_circuit = addr.iter().any(|p| matches!(p, Protocol::P2pCircuit));
    if !has_circuit {
        addr.push(Protocol::P2pCircuit);
    }
    addr
}

fn peer_id_from_multiaddr(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|p| match p {
        Protocol::P2p(peer_id) => Some(peer_id),
        _ => None,
    })
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

fn collect_relays(config: &NodeNetworkConfig) -> Result<Vec<Multiaddr>> {
    let mut addrs = Vec::new();
    for relay in &config.routing.relay_servers {
        if relay.trim().is_empty() {
            continue;
        }
        match relay.parse::<Multiaddr>() {
            Ok(addr) => addrs.push(addr),
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::node",
                    error = %err,
                    relay = relay.as_str(),
                    "invalid relay multiaddr"
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
