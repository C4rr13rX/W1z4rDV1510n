use crate::config::NodeNetworkConfig;
use crate::paths::node_data_dir;
use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic, MessageAuthenticity};
use libp2p::identify;
use libp2p::kad::{self, store::MemoryStore};
use libp2p::mdns;
use libp2p::swarm::SwarmEvent;
use libp2p::{identity, multiaddr::Protocol, Multiaddr, PeerId, Swarm, SwarmBuilder};
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn};

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
}

#[derive(Debug)]
enum NodeBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Identify(identify::Event),
    Kademlia(kad::Event),
    Mdns(mdns::Event),
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
            let kademlia = kad::Behaviour::new(peer_id, MemoryStore::new(peer_id));
            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(NodeBehaviour {
                gossipsub,
                identify,
                kademlia,
                mdns,
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
                handle_swarm_event(event, &mut swarm, &peer_count)?;
            }
        }
    }
}

fn handle_swarm_event(
    event: SwarmEvent<NodeBehaviourEvent>,
    swarm: &mut Swarm<NodeBehaviour>,
    peer_count: &Arc<AtomicUsize>,
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
        .validation_mode(gossipsub::ValidationMode::Permissive)
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
