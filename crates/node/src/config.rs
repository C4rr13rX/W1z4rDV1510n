use serde::{Deserialize, Serialize};
use crate::paths::node_data_dir;
use w1z4rdv1510n::blockchain::SensorDescriptor;
use w1z4rdv1510n::config::{
    BlockchainConfig, ClusterConfig, ComputeRoutingConfig, LedgerConfig, NodeRole,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    pub node_id: String,
    pub node_role: NodeRole,
    pub network: NodeNetworkConfig,
    pub openstack: OpenStackConfig,
    pub wallet: WalletConfig,
    pub blockchain: BlockchainConfig,
    pub compute: ComputeRoutingConfig,
    pub cluster: ClusterConfig,
    pub ledger: LedgerConfig,
    pub sensors: Vec<SensorDescriptor>,
    pub chain_spec: ChainSpecConfig,
    pub energy_reporting: EnergyReportingConfig,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            node_id: "node-001".to_string(),
            node_role: NodeRole::default(),
            network: NodeNetworkConfig::default(),
            openstack: OpenStackConfig::default(),
            wallet: WalletConfig::default(),
            blockchain: BlockchainConfig::default(),
            compute: ComputeRoutingConfig::default(),
            cluster: ClusterConfig::default(),
            ledger: LedgerConfig::default(),
            sensors: Vec::new(),
            chain_spec: ChainSpecConfig::default(),
            energy_reporting: EnergyReportingConfig::default(),
        }
    }
}

impl NodeConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.node_id.trim().is_empty(), "node_id must be non-empty");
        anyhow::ensure!(!self.network.listen_addr.trim().is_empty(), "listen_addr must be set");
        anyhow::ensure!(self.network.max_peers > 0, "network.max_peers must be > 0");
        anyhow::ensure!(
            self.network.security.max_message_bytes >= 100,
            "network.security.max_message_bytes must be >= 100"
        );
        anyhow::ensure!(
            self.network.security.max_messages_per_rpc > 0,
            "network.security.max_messages_per_rpc must be > 0"
        );
        anyhow::ensure!(
            self.network.security.max_established_total > 0,
            "network.security.max_established_total must be > 0"
        );
        anyhow::ensure!(
            self.wallet.enabled,
            "wallet must be enabled for production nodes"
        );
        if self.ledger.enabled {
            anyhow::ensure!(
                !self.ledger.backend.trim().is_empty(),
                "ledger.backend must be set when ledger is enabled"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeNetworkConfig {
    pub listen_addr: String,
    pub bootstrap_peers: Vec<String>,
    pub max_peers: usize,
    pub gossip_protocol: String,
    #[serde(default)]
    pub security: NetworkSecurityConfig,
}

impl Default for NodeNetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8088".to_string(),
            bootstrap_peers: Vec::new(),
            max_peers: 128,
            gossip_protocol: "w1z4rdv1510n-gossip".to_string(),
            security: NetworkSecurityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkSecurityConfig {
    pub max_message_bytes: usize,
    pub max_messages_per_rpc: usize,
    pub max_pending_incoming: u32,
    pub max_pending_outgoing: u32,
    pub max_established_incoming: u32,
    pub max_established_outgoing: u32,
    pub max_established_total: u32,
    pub max_established_per_peer: u32,
}

impl Default for NetworkSecurityConfig {
    fn default() -> Self {
        Self {
            max_message_bytes: 262_144,
            max_messages_per_rpc: 128,
            max_pending_incoming: 64,
            max_pending_outgoing: 64,
            max_established_incoming: 256,
            max_established_outgoing: 256,
            max_established_total: 512,
            max_established_per_peer: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenStackConfig {
    pub enabled: bool,
    #[serde(default)]
    pub mode: OpenStackMode,
    pub region: String,
    pub interface: String,
}

impl Default for OpenStackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: OpenStackMode::default(),
            region: "RegionOne".to_string(),
            interface: "public".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WalletConfig {
    pub enabled: bool,
    pub path: String,
    pub auto_create: bool,
    pub encrypted: bool,
    pub passphrase_env: String,
    pub prompt_on_load: bool,
}

impl Default for WalletConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: default_wallet_path(),
            auto_create: true,
            encrypted: true,
            passphrase_env: "W1Z4RDV1510N_WALLET_PASSPHRASE".to_string(),
            prompt_on_load: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OpenStackMode {
    LocalControlPlane,
}

impl Default for OpenStackMode {
    fn default() -> Self {
        OpenStackMode::LocalControlPlane
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChainSpecConfig {
    pub genesis_path: String,
    pub reward_contract_path: String,
    pub bridge_contract_path: String,
    pub token_standard_path: String,
}

impl Default for ChainSpecConfig {
    fn default() -> Self {
        Self {
            genesis_path: "chain/genesis.json".to_string(),
            reward_contract_path: "chain/reward_contract.json".to_string(),
            bridge_contract_path: "chain/bridge_contract.json".to_string(),
            token_standard_path: "chain/token_standard.json".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergyReportingConfig {
    pub enabled: bool,
    pub sample_interval_secs: u64,
}

fn default_wallet_path() -> String {
    let path = node_data_dir().join("wallet.json");
    path.to_string_lossy().into_owned()
}

impl Default for EnergyReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_interval_secs: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_deserializes_from_json() {
        let raw = r#"{
            "node_id": "node-01",
            "node_role": "WORKER",
            "network": { "listen_addr": "0.0.0.0:1", "bootstrap_peers": [], "max_peers": 1, "gossip_protocol": "g", "security": { "max_message_bytes": 1024, "max_messages_per_rpc": 4, "max_pending_incoming": 1, "max_pending_outgoing": 1, "max_established_incoming": 2, "max_established_outgoing": 2, "max_established_total": 4, "max_established_per_peer": 1 } },
            "openstack": { "enabled": false, "mode": "LOCAL_CONTROL_PLANE", "region": "RegionOne", "interface": "public" },
            "wallet": { "enabled": true, "path": "wallet.json", "auto_create": true, "encrypted": true, "passphrase_env": "W1Z4RDV1510N_WALLET_PASSPHRASE", "prompt_on_load": true },
            "blockchain": { "enabled": false, "chain_id": "w1", "consensus": "poa", "bootstrap_peers": [], "node_role": "WORKER", "reward_policy": { "sensor_reward_weight": 1.0, "compute_reward_weight": 1.0, "energy_efficiency_weight": 1.0, "uptime_reward_weight": 1.0 }, "energy_efficiency": { "target_watts": 150.0, "efficiency_baseline": 1.0 }, "attestation": { "endpoint": "", "required": false }, "require_sensor_attestation": false },
            "compute": { "allow_gpu": false, "allow_quantum": false, "quantum_endpoints": [] },
            "cluster": { "enabled": false, "mode": "local", "min_nodes": 1, "openstack_minimal": false },
            "ledger": { "enabled": false, "backend": "none", "endpoint": "" },
            "sensors": [],
            "chain_spec": { "genesis_path": "chain/genesis.json", "reward_contract_path": "chain/reward_contract.json", "bridge_contract_path": "chain/bridge_contract.json", "token_standard_path": "chain/token_standard.json" },
            "energy_reporting": { "enabled": true, "sample_interval_secs": 30 }
        }"#;
        let parsed: NodeConfig = serde_json::from_str(raw).expect("parse config");
        assert_eq!(parsed.node_id, "node-01");
        assert!(matches!(parsed.node_role, NodeRole::Worker));
    }
}
