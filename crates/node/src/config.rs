use crate::paths::node_data_dir;
use crate::chain::ChainSpec;
use serde::{Deserialize, Serialize};
use w1z4rdv1510n::blockchain::SensorDescriptor;
use w1z4rdv1510n::config::{
    BlockchainConfig, ClusterConfig, ComputeRoutingConfig, LedgerConfig, NodeRole,
};
use w1z4rdv1510n::streaming::KnowledgeQueueConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    pub node_id: String,
    pub node_role: NodeRole,
    pub network: NodeNetworkConfig,
    pub api: NodeApiConfig,
    pub openstack: OpenStackConfig,
    pub wallet: WalletConfig,
    pub data: DataMeshConfig,
    pub streaming: StreamingRuntimeConfig,
    pub workload: WorkloadProfileConfig,
    pub peer_scoring: PeerScoringConfig,
    pub knowledge: KnowledgeConfig,
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
            api: NodeApiConfig::default(),
            openstack: OpenStackConfig::default(),
            wallet: WalletConfig::default(),
            data: DataMeshConfig::default(),
            streaming: StreamingRuntimeConfig::default(),
            workload: WorkloadProfileConfig::default(),
            peer_scoring: PeerScoringConfig::default(),
            knowledge: KnowledgeConfig::default(),
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
            self.network.security.max_message_age_secs >= 0,
            "network.security.max_message_age_secs must be >= 0"
        );
        anyhow::ensure!(
            self.network.security.max_clock_skew_secs >= 0,
            "network.security.max_clock_skew_secs must be >= 0"
        );
        anyhow::ensure!(
            self.network.security.max_seen_message_ids > 0,
            "network.security.max_seen_message_ids must be > 0"
        );
        anyhow::ensure!(
            self.network.security.message_id_ttl_secs >= 0,
            "network.security.message_id_ttl_secs must be >= 0"
        );
        anyhow::ensure!(
            self.network.security.max_messages_per_key_per_window > 0,
            "network.security.max_messages_per_key_per_window must be > 0"
        );
        anyhow::ensure!(
            self.network.security.key_rate_window_secs > 0,
            "network.security.key_rate_window_secs must be > 0"
        );
        anyhow::ensure!(
            self.network.security.max_tracked_public_keys > 0,
            "network.security.max_tracked_public_keys must be > 0"
        );
        anyhow::ensure!(
            self.network.security.public_key_ttl_secs >= 0,
            "network.security.public_key_ttl_secs must be >= 0"
        );
        if self.network.routing.enable_relay {
            anyhow::ensure!(
                !self.network.routing.relay_servers.is_empty(),
                "network.routing.relay_servers must be non-empty when relay is enabled"
            );
        }
        for addr in &self.network.routing.external_addresses {
            anyhow::ensure!(
                !addr.trim().is_empty(),
                "network.routing.external_addresses entries must be non-empty"
            );
        }
        if self.network.routing.dial_backoff_base_secs == 0
            || self.network.routing.dial_backoff_max_secs == 0
        {
            anyhow::bail!("network.routing.dial_backoff_*_secs must be > 0");
        }
        anyhow::ensure!(
            self.network.routing.dial_backoff_max_secs >= self.network.routing.dial_backoff_base_secs,
            "network.routing.dial_backoff_max_secs must be >= dial_backoff_base_secs"
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
        if self.blockchain.enabled {
            let spec = ChainSpec::load(&self.chain_spec)?;
            anyhow::ensure!(
                spec.chain_id == self.blockchain.chain_id,
                "blockchain.chain_id must match chain spec"
            );
            anyhow::ensure!(
                spec.consensus == self.blockchain.consensus,
                "blockchain.consensus must match chain spec"
            );
            anyhow::ensure!(
                self.blockchain.validator_policy.heartbeat_interval_secs > 0,
                "blockchain.validator_policy.heartbeat_interval_secs must be > 0"
            );
            anyhow::ensure!(
                self.blockchain.validator_policy.max_missed_heartbeats > 0,
                "blockchain.validator_policy.max_missed_heartbeats must be > 0"
            );
            anyhow::ensure!(
                self.blockchain.validator_policy.downtime_penalty_score >= 0.0,
                "blockchain.validator_policy.downtime_penalty_score must be >= 0"
            );
            if self.blockchain.bridge.enabled {
                let bridge = &self.blockchain.bridge;
                anyhow::ensure!(
                    bridge.max_proof_bytes > 0,
                    "blockchain.bridge.max_proof_bytes must be > 0"
                );
                anyhow::ensure!(
                    !bridge.chains.is_empty(),
                    "blockchain.bridge.chains must be non-empty when enabled"
                );
                for chain in &bridge.chains {
                    anyhow::ensure!(
                        !chain.chain_id.trim().is_empty(),
                        "blockchain.bridge.chain_id must be non-empty"
                    );
                    anyhow::ensure!(
                        chain.min_confirmations > 0,
                        "blockchain.bridge.min_confirmations must be > 0"
                    );
                    anyhow::ensure!(
                        chain.relayer_quorum > 0,
                        "blockchain.bridge.relayer_quorum must be > 0"
                    );
                    anyhow::ensure!(
                        chain.max_deposit_amount.is_finite() && chain.max_deposit_amount > 0.0,
                        "blockchain.bridge.max_deposit_amount must be > 0 and finite"
                    );
                    anyhow::ensure!(
                        !chain.allowed_assets.is_empty(),
                        "blockchain.bridge.allowed_assets must be non-empty"
                    );
                    if let Some(address) = &chain.deposit_address {
                        anyhow::ensure!(
                            !address.trim().is_empty(),
                            "blockchain.bridge.deposit_address must be non-empty when set"
                        );
                    }
                    if let Some(template) = &chain.recipient_tag_template {
                        anyhow::ensure!(
                            !template.trim().is_empty(),
                            "blockchain.bridge.recipient_tag_template must be non-empty when set"
                        );
                    }
                    if matches!(
                        chain.verification,
                        w1z4rdv1510n::bridge::BridgeVerificationMode::RelayerQuorum
                    ) {
                        anyhow::ensure!(
                            chain.relayer_public_keys.len() as u32 >= chain.relayer_quorum,
                            "blockchain.bridge.relayer_public_keys must satisfy relayer_quorum"
                        );
                    }
                }
            }
            if self.blockchain.fee_market.enabled {
                let fee = &self.blockchain.fee_market;
                anyhow::ensure!(
                    fee.base_fee.is_finite()
                        && fee.min_base_fee.is_finite()
                        && fee.max_base_fee.is_finite(),
                    "blockchain.fee_market base_fee/min_base_fee/max_base_fee must be finite"
                );
                anyhow::ensure!(
                    fee.min_base_fee >= 0.0,
                    "blockchain.fee_market.min_base_fee must be >= 0"
                );
                anyhow::ensure!(
                    fee.min_base_fee <= fee.max_base_fee,
                    "blockchain.fee_market.min_base_fee must be <= max_base_fee"
                );
                anyhow::ensure!(
                    fee.base_fee >= fee.min_base_fee && fee.base_fee <= fee.max_base_fee,
                    "blockchain.fee_market.base_fee must be within [min_base_fee, max_base_fee]"
                );
                anyhow::ensure!(
                    fee.target_txs_per_window > 0,
                    "blockchain.fee_market.target_txs_per_window must be > 0"
                );
                anyhow::ensure!(
                    fee.window_secs > 0,
                    "blockchain.fee_market.window_secs must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&fee.adjustment_rate),
                    "blockchain.fee_market.adjustment_rate must be in [0,1]"
                );
            }
        }
        if self.api.require_api_key {
            anyhow::ensure!(
                !self.api.api_key_header.trim().is_empty(),
                "api.api_key_header must be non-empty when api.require_api_key is true"
            );
            anyhow::ensure!(
                !self.api.api_key_hashes.is_empty() || !self.api.api_key_env.trim().is_empty(),
                "api.api_key_hashes or api.api_key_env must be set when api.require_api_key is true"
            );
            for hash in &self.api.api_key_hashes {
                anyhow::ensure!(
                    is_hex_string(hash, 64),
                    "api.api_key_hashes entries must be 64-char hex strings"
                );
            }
        }
        anyhow::ensure!(
            self.api.rate_limit_window_secs > 0,
            "api.rate_limit_window_secs must be > 0"
        );
        anyhow::ensure!(
            self.api.rate_limit_max_requests > 0,
            "api.rate_limit_max_requests must be > 0"
        );
        anyhow::ensure!(
            self.api.rate_limit_bridge_max_requests > 0,
            "api.rate_limit_bridge_max_requests must be > 0"
        );
        anyhow::ensure!(
            self.api.rate_limit_balance_max_requests > 0,
            "api.rate_limit_balance_max_requests must be > 0"
        );
        if self.data.enabled {
            anyhow::ensure!(
                self.data.chunk_size_bytes > 0,
                "data.chunk_size_bytes must be > 0"
            );
            anyhow::ensure!(
                self.data.max_payload_bytes >= self.data.chunk_size_bytes,
                "data.max_payload_bytes must be >= chunk_size_bytes"
            );
            anyhow::ensure!(
                self.data.chunk_size_bytes.saturating_mul(2)
                    <= self.network.security.max_message_bytes,
                "data.chunk_size_bytes must fit within network.security.max_message_bytes"
            );
            anyhow::ensure!(
                self.data.replication_factor > 0,
                "data.replication_factor must be > 0"
            );
            anyhow::ensure!(
                self.data.receipt_quorum > 0,
                "data.receipt_quorum must be > 0"
            );
            anyhow::ensure!(
                self.data.receipt_quorum <= self.data.replication_factor,
                "data.receipt_quorum must be <= replication_factor"
            );
            anyhow::ensure!(
                !self.data.storage_path.trim().is_empty(),
                "data.storage_path must be set when data is enabled"
            );
            if self.data.maintenance_enabled {
                anyhow::ensure!(
                    self.data.maintenance_interval_secs > 0,
                    "data.maintenance_interval_secs must be > 0 when maintenance is enabled"
                );
                anyhow::ensure!(
                    self.data.max_repair_requests_per_tick > 0,
                    "data.max_repair_requests_per_tick must be > 0 when maintenance is enabled"
                );
            }
            anyhow::ensure!(
                self.data.storage_reward_base.is_finite() && self.data.storage_reward_base >= 0.0,
                "data.storage_reward_base must be >= 0 and finite"
            );
            anyhow::ensure!(
                self.data.storage_reward_per_mb.is_finite() && self.data.storage_reward_per_mb >= 0.0,
                "data.storage_reward_per_mb must be >= 0 and finite"
            );
        }
        if self.streaming.enabled {
            anyhow::ensure!(
                !self.streaming.run_config_path.trim().is_empty(),
                "streaming.run_config_path must be set when streaming is enabled"
            );
            anyhow::ensure!(
                !self.streaming.stream_payload_kind.trim().is_empty(),
                "streaming.stream_payload_kind must be non-empty"
            );
            anyhow::ensure!(
                !self.streaming.share_payload_kind.trim().is_empty(),
                "streaming.share_payload_kind must be non-empty"
            );
            anyhow::ensure!(
                self.streaming.ultradian_node,
                "streaming.ultradian_node must be true when streaming is enabled"
            );
        }
        if self.peer_scoring.enabled {
            anyhow::ensure!(
                self.peer_scoring.publish_interval_secs > 0,
                "peer_scoring.publish_interval_secs must be > 0"
            );
            anyhow::ensure!(
                self.peer_scoring.report_ttl_secs > 0,
                "peer_scoring.report_ttl_secs must be > 0"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.peer_scoring.ema_alpha),
                "peer_scoring.ema_alpha must be in [0,1]"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.peer_scoring.efficiency_offload_threshold),
                "peer_scoring.efficiency_offload_threshold must be in [0,1]"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.peer_scoring.capacity_threshold),
                "peer_scoring.capacity_threshold must be in [0,1]"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.peer_scoring.accuracy_threshold),
                "peer_scoring.accuracy_threshold must be in [0,1]"
            );
            anyhow::ensure!(
                self.peer_scoring.min_peer_reports > 0,
                "peer_scoring.min_peer_reports must be > 0"
            );
            anyhow::ensure!(
                self.peer_scoring.target_latency_ms > 0.0,
                "peer_scoring.target_latency_ms must be > 0"
            );
        }
        if self.knowledge.enabled {
            anyhow::ensure!(
                !self.knowledge.state_path.trim().is_empty(),
                "knowledge.state_path must be set when knowledge is enabled"
            );
            anyhow::ensure!(
                self.knowledge.queue.min_votes > 0,
                "knowledge.queue.min_votes must be > 0"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.knowledge.queue.min_confidence),
                "knowledge.queue.min_confidence must be in [0,1]"
            );
            anyhow::ensure!(
                self.knowledge.queue.max_pending > 0,
                "knowledge.queue.max_pending must be > 0"
            );
            anyhow::ensure!(
                self.knowledge.queue.candidate_limit > 0,
                "knowledge.queue.candidate_limit must be > 0"
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
    pub routing: NodeNetworkRoutingConfig,
    #[serde(default)]
    pub security: NetworkSecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeApiConfig {
    pub require_api_key: bool,
    pub api_key_env: String,
    pub api_key_header: String,
    #[serde(default)]
    pub api_key_hashes: Vec<String>,
    pub rate_limit_window_secs: u64,
    pub rate_limit_max_requests: u32,
    pub rate_limit_bridge_max_requests: u32,
    pub rate_limit_balance_max_requests: u32,
}

impl Default for NodeApiConfig {
    fn default() -> Self {
        Self {
            require_api_key: false,
            api_key_env: "W1Z4RDV1510N_API_KEY".to_string(),
            api_key_header: "x-api-key".to_string(),
            api_key_hashes: Vec::new(),
            rate_limit_window_secs: 60,
            rate_limit_max_requests: 10,
            rate_limit_bridge_max_requests: 10,
            rate_limit_balance_max_requests: 10,
        }
    }
}

impl Default for NodeNetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8088".to_string(),
            bootstrap_peers: Vec::new(),
            max_peers: 128,
            gossip_protocol: "w1z4rdv1510n-gossip".to_string(),
            routing: NodeNetworkRoutingConfig::default(),
            security: NetworkSecurityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeNetworkRoutingConfig {
    pub enable_relay: bool,
    pub relay_servers: Vec<String>,
    pub external_addresses: Vec<String>,
    pub reserve_relays: bool,
    pub enable_quic: bool,
    pub enable_autonat: bool,
    pub use_observed_addresses: bool,
    pub enable_peer_scoring: bool,
    pub dial_backoff_base_secs: u64,
    pub dial_backoff_max_secs: u64,
}

impl Default for NodeNetworkRoutingConfig {
    fn default() -> Self {
        Self {
            enable_relay: false,
            relay_servers: Vec::new(),
            external_addresses: Vec::new(),
            reserve_relays: true,
            enable_quic: false,
            enable_autonat: true,
            use_observed_addresses: true,
            enable_peer_scoring: true,
            dial_backoff_base_secs: 5,
            dial_backoff_max_secs: 300,
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
    pub require_signed_payloads: bool,
    pub max_message_age_secs: i64,
    pub max_clock_skew_secs: i64,
    pub max_seen_message_ids: usize,
    pub message_id_ttl_secs: i64,
    pub max_messages_per_key_per_window: u32,
    pub key_rate_window_secs: i64,
    pub max_tracked_public_keys: usize,
    pub public_key_ttl_secs: i64,
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
            require_signed_payloads: true,
            max_message_age_secs: 300,
            max_clock_skew_secs: 10,
            max_seen_message_ids: 50_000,
            message_id_ttl_secs: 600,
            max_messages_per_key_per_window: 120,
            key_rate_window_secs: 60,
            max_tracked_public_keys: 10_000,
            public_key_ttl_secs: 600,
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
#[serde(default)]
pub struct DataMeshConfig {
    pub enabled: bool,
    pub storage_path: String,
    pub host_storage: bool,
    pub max_payload_bytes: usize,
    pub chunk_size_bytes: usize,
    pub replication_factor: usize,
    pub receipt_quorum: usize,
    pub require_manifest_signature: bool,
    pub require_receipt_signature: bool,
    pub max_pending_chunks: usize,
    pub maintenance_enabled: bool,
    pub maintenance_interval_secs: u64,
    pub retention_days: u32,
    pub max_storage_bytes: u64,
    pub max_repair_requests_per_tick: usize,
    pub storage_reward_enabled: bool,
    pub storage_reward_base: f64,
    pub storage_reward_per_mb: f64,
}

impl Default for DataMeshConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_path: default_data_path(),
            host_storage: true,
            max_payload_bytes: 512 * 1024,
            chunk_size_bytes: 32 * 1024,
            replication_factor: 3,
            receipt_quorum: 2,
            require_manifest_signature: true,
            require_receipt_signature: true,
            max_pending_chunks: 1024,
            maintenance_enabled: true,
            maintenance_interval_secs: 300,
            retention_days: 30,
            max_storage_bytes: 0,
            max_repair_requests_per_tick: 64,
            storage_reward_enabled: true,
            storage_reward_base: 0.15,
            storage_reward_per_mb: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingRuntimeConfig {
    pub enabled: bool,
    pub ultradian_node: bool,
    pub run_config_path: String,
    pub publish_streams: bool,
    pub publish_shares: bool,
    pub consume_streams: bool,
    pub consume_shares: bool,
    pub stream_payload_kind: String,
    pub share_payload_kind: String,
    pub min_cpu_cores: usize,
    pub min_memory_gb: f64,
}

impl Default for StreamingRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ultradian_node: false,
            run_config_path: "run_config.json".to_string(),
            publish_streams: true,
            publish_shares: true,
            consume_streams: true,
            consume_shares: true,
            stream_payload_kind: "stream.envelope.v1".to_string(),
            share_payload_kind: "neural.fabric.v1".to_string(),
            min_cpu_cores: 2,
            min_memory_gb: 4.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadProfileConfig {
    pub enable_sensor_ingest: bool,
    pub enable_stream_processing: bool,
    pub enable_share_publish: bool,
    pub enable_share_consume: bool,
    pub enable_storage: bool,
}

impl Default for WorkloadProfileConfig {
    fn default() -> Self {
        Self {
            enable_sensor_ingest: true,
            enable_stream_processing: true,
            enable_share_publish: true,
            enable_share_consume: true,
            enable_storage: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PeerScoringConfig {
    pub enabled: bool,
    pub publish_interval_secs: u64,
    pub report_ttl_secs: u64,
    pub ema_alpha: f64,
    pub efficiency_offload_threshold: f64,
    pub capacity_threshold: f64,
    pub accuracy_threshold: f64,
    pub min_peer_reports: usize,
    pub target_latency_ms: f64,
}

impl Default for PeerScoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            publish_interval_secs: 30,
            report_ttl_secs: 300,
            ema_alpha: 0.2,
            efficiency_offload_threshold: 0.25,
            capacity_threshold: 0.5,
            accuracy_threshold: 0.4,
            min_peer_reports: 3,
            target_latency_ms: 500.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeConfig {
    pub enabled: bool,
    pub persist_state: bool,
    pub state_path: String,
    pub queue: KnowledgeQueueConfig,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            persist_state: false,
            state_path: default_knowledge_path(),
            queue: KnowledgeQueueConfig::default(),
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

fn default_data_path() -> String {
    let path = node_data_dir().join("data");
    path.to_string_lossy().into_owned()
}

fn default_knowledge_path() -> String {
    let path = node_data_dir().join("knowledge_state.json");
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

fn is_hex_string(value: &str, length: usize) -> bool {
    if value.len() != length {
        return false;
    }
    value
        .as_bytes()
        .iter()
        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F'))
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
