use crate::chain::ChainSpec;
use crate::config::NodeConfig;
use crate::ledger::LocalLedger;
use crate::openstack::OpenStackControlPlane;
use crate::paths::node_data_dir;
use crate::wallet::{WalletSigner, WalletStore, node_id_from_wallet};
use crate::p2p::NodeNetwork;
use anyhow::Result;
use tracing::{info, warn};
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, NoopLedger, NodeCapabilities, NodeRegistration, ValidatorHeartbeat,
};
use w1z4rdv1510n::compute::{ComputeJobKind, ComputeRouter};
use w1z4rdv1510n::config::NodeRole;
use w1z4rdv1510n::hardware::HardwareProfile;
use w1z4rdv1510n::schema::Timestamp;

pub struct NodeRuntime {
    config: NodeConfig,
    chain_spec: ChainSpec,
    compute_router: ComputeRouter,
    network: NodeNetwork,
    ledger: Box<dyn BlockchainLedger>,
    profile: HardwareProfile,
    openstack: Option<OpenStackControlPlane>,
    wallet: WalletSigner,
}

impl NodeRuntime {
    pub fn new(mut config: NodeConfig) -> Result<Self> {
        let wallet = WalletStore::load_or_create_signer(&config.wallet)?;
        if config.network.bootstrap_peers.is_empty() && !config.blockchain.bootstrap_peers.is_empty()
        {
            config.network.bootstrap_peers = config.blockchain.bootstrap_peers.clone();
        }
        if config.node_id.trim().is_empty() || config.node_id == "node-001" {
            config.node_id = node_id_from_wallet(&wallet.wallet().address);
        }
        let chain_spec = ChainSpec::load(&config.chain_spec)?;
        let profile = HardwareProfile::detect();
        let compute_router = ComputeRouter::new(config.compute.clone(), config.cluster.clone());
        let network = NodeNetwork::new(config.network.clone());
        let ledger = build_ledger(&config)?;
        let openstack = if config.openstack.enabled {
            Some(OpenStackControlPlane::new(config.openstack.clone()))
        } else {
            None
        };
        Ok(Self {
            config,
            chain_spec,
            compute_router,
            network,
            ledger,
            profile,
            openstack,
            wallet,
        })
    }

    pub fn start(mut self) -> Result<()> {
        self.network.start(&self.config.node_id)?;
        self.network.connect_bootstrap()?;
        self.register_node()?;
        self.maybe_send_validator_heartbeat()?;
        let target = self.compute_router.route(ComputeJobKind::TensorHeavy);
        info!(
            target: "w1z4rdv1510n::node",
            node_id = self.config.node_id,
            peer_count = self.network.peer_count(),
            compute_target = ?target,
            chain_id = self.chain_spec.chain_id,
            wallet_address = self.wallet.wallet().address.as_str(),
            "node runtime started"
        );
        Ok(())
    }

    fn register_node(&self) -> Result<()> {
        if !self.config.blockchain.enabled {
            warn!(
                target: "w1z4rdv1510n::node",
                "blockchain disabled; skipping registration"
            );
            return Ok(());
        }
        let capabilities = NodeCapabilities {
            cpu_cores: self.profile.cpu_cores,
            memory_gb: self.profile.total_memory_gb,
            gpu_count: if self.profile.has_gpu { 1 } else { 0 },
        };
        let registration = NodeRegistration {
            node_id: self.config.node_id.clone(),
            role: self.config.node_role.clone(),
            capabilities,
            sensors: self.config.sensors.clone(),
            wallet_address: self.wallet.wallet().address.clone(),
            wallet_public_key: self.wallet.wallet().public_key.clone(),
            signature: String::new(),
        };
        let registration = self.wallet.sign_node_registration(registration);
        if let Err(err) = self.ledger.register_node(registration) {
            warn!(
                target: "w1z4rdv1510n::node",
                error = %err,
                "ledger registration failed; running in offline mode"
            );
        }
        Ok(())
    }

    fn maybe_send_validator_heartbeat(&self) -> Result<()> {
        if !self.config.blockchain.enabled {
            return Ok(());
        }
        if !matches!(self.config.node_role, NodeRole::Validator) {
            return Ok(());
        }
        let heartbeat = ValidatorHeartbeat {
            node_id: self.config.node_id.clone(),
            timestamp: now_timestamp(),
            fee_paid: 0.0,
            signature: String::new(),
        };
        let heartbeat = self.wallet.sign_validator_heartbeat(heartbeat);
        if let Err(err) = self.ledger.submit_validator_heartbeat(heartbeat) {
            warn!(
                target: "w1z4rdv1510n::node",
                error = %err,
                "validator heartbeat rejected"
            );
        }
        Ok(())
    }

    pub fn openstack_control_plane(&self) -> Option<&OpenStackControlPlane> {
        self.openstack.as_ref()
    }
}

fn build_ledger(config: &NodeConfig) -> Result<Box<dyn BlockchainLedger>> {
    if !config.ledger.enabled {
        return Ok(Box::new(NoopLedger::default()));
    }
    let backend = config.ledger.backend.trim().to_ascii_lowercase();
    match backend.as_str() {
        "local" | "file" => {
            let path = if config.ledger.endpoint.trim().is_empty() {
                node_data_dir().join("ledger.json")
            } else {
                config.ledger.endpoint.clone().into()
            };
            Ok(Box::new(LocalLedger::load_or_create(
                path,
                config.blockchain.validator_policy.clone(),
                config.blockchain.fee_market.clone(),
                config.blockchain.bridge.clone(),
            )?))
        }
        "none" | "" => Ok(Box::new(NoopLedger::default())),
        other => {
            warn!(
                target: "w1z4rdv1510n::node",
                backend = %other,
                "unknown ledger backend; falling back to noop ledger"
            );
            Ok(Box::new(NoopLedger::default()))
        }
    }
}

fn now_timestamp() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    Timestamp { unix }
}
