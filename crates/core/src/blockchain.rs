use crate::bridge::{BridgeProof, ChainKind};
use crate::config::{BlockchainConfig, NodeRole, RewardPolicyConfig};
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorDescriptor {
    pub sensor_id: String,
    pub kind: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRegistration {
    pub node_id: String,
    pub role: NodeRole,
    pub capabilities: NodeCapabilities,
    #[serde(default)]
    pub sensors: Vec<SensorDescriptor>,
    #[serde(default)]
    pub wallet_address: String,
    #[serde(default)]
    pub wallet_public_key: String,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyEfficiencySample {
    pub node_id: String,
    pub timestamp: Timestamp,
    pub watts: f64,
    pub throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorCommitment {
    pub node_id: String,
    pub sensor_id: String,
    pub timestamp: Timestamp,
    pub payload_hash: String,
    #[serde(default)]
    pub fee_paid: f64,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardEventKind {
    SensorContribution,
    ComputeContribution,
    EnergyEfficiency,
    Uptime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEvent {
    pub node_id: String,
    pub kind: RewardEventKind,
    pub timestamp: Timestamp,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardBalance {
    pub node_id: String,
    pub balance: f64,
    pub updated_at: Timestamp,
    #[serde(default)]
    pub wallet_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeDeposit {
    pub deposit_id: String,
    pub node_id: String,
    pub amount: f64,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeIntent {
    pub intent_id: String,
    pub chain_id: String,
    #[serde(default)]
    pub chain_kind: ChainKind,
    pub asset: String,
    pub amount: f64,
    pub recipient_node_id: String,
    #[serde(default)]
    pub deposit_address: String,
    #[serde(default)]
    pub recipient_tag: Option<String>,
    pub idempotency_key: String,
    pub created_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkKind {
    SensorIngest,
    StorageContribution,
    ComputeTask,
    ModelUpdate,
    CausalDiscovery,
    Forecasting,
    HumanAnnotation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkAssignment {
    pub work_id: String,
    pub kind: WorkKind,
    pub issued_at: Timestamp,
    pub expires_at: Timestamp,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkProof {
    pub work_id: String,
    pub node_id: String,
    pub kind: WorkKind,
    pub completed_at: Timestamp,
    pub score: f64,
    #[serde(default)]
    pub fee_paid: f64,
    #[serde(default)]
    pub metrics: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainTransfer {
    #[serde(default)]
    pub node_id: String,
    pub source_chain: String,
    pub target_chain: String,
    pub token_symbol: String,
    pub amount: u64,
    #[serde(default)]
    pub fee_paid: f64,
    pub payload_hash: String,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidatorStatus {
    Active,
    Inactive,
    Jailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorRecord {
    pub node_id: String,
    pub status: ValidatorStatus,
    pub last_heartbeat: Timestamp,
    pub missed_heartbeats: u32,
    #[serde(default)]
    pub jailed_until: Option<Timestamp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorHeartbeat {
    pub node_id: String,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub fee_paid: f64,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidatorSlashReason {
    Downtime,
    DoubleSign,
    Equivocation,
    Governance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSlashEvent {
    pub node_id: String,
    pub timestamp: Timestamp,
    pub reason: ValidatorSlashReason,
    #[serde(default)]
    pub penalty_score: f64,
    #[serde(default)]
    pub signature: String,
}

pub fn work_proof_payload(proof: &WorkProof) -> String {
    format!(
        "work|{}|{}|{:?}|{}|{:.6}|{:.6}",
        proof.work_id,
        proof.node_id,
        proof.kind,
        proof.completed_at.unix,
        proof.score,
        proof.fee_paid
    )
}

pub fn sensor_commitment_payload(commitment: &SensorCommitment) -> String {
    format!(
        "sensor|{}|{}|{}|{}|{:.6}",
        commitment.node_id,
        commitment.sensor_id,
        commitment.timestamp.unix,
        commitment.payload_hash,
        commitment.fee_paid
    )
}

pub fn cross_chain_transfer_payload(transfer: &CrossChainTransfer) -> String {
    format!(
        "xfer|{}|{}|{}|{}|{}|{:.6}|{}|{}",
        transfer.node_id,
        transfer.source_chain,
        transfer.target_chain,
        transfer.token_symbol,
        transfer.amount,
        transfer.fee_paid,
        transfer.payload_hash,
        transfer.timestamp.unix
    )
}

pub fn validator_heartbeat_payload(heartbeat: &ValidatorHeartbeat) -> String {
    format!(
        "heartbeat|{}|{}|{:.6}",
        heartbeat.node_id, heartbeat.timestamp.unix, heartbeat.fee_paid
    )
}

pub fn stake_deposit_payload(deposit: &StakeDeposit) -> String {
    format!(
        "stake|{}|{}|{:.6}|{}|{}",
        deposit.node_id,
        deposit.deposit_id,
        deposit.amount,
        deposit.timestamp.unix,
        deposit.source
    )
}

pub fn bridge_intent_payload(intent: &BridgeIntent) -> String {
    format!(
        "bridge_intent|{}|{:?}|{}|{:.6}|{}|{}|{}|{}",
        intent.chain_id,
        intent.chain_kind,
        intent.asset,
        intent.amount,
        intent.recipient_node_id,
        intent.deposit_address,
        intent.recipient_tag.clone().unwrap_or_default(),
        intent.idempotency_key
    )
}

pub fn bridge_intent_id(intent: &BridgeIntent) -> String {
    let payload = bridge_intent_payload(intent);
    compute_payload_hash(payload.as_bytes())
}

pub fn validator_slash_payload(slash: &ValidatorSlashEvent) -> String {
    format!(
        "slash|{}|{:?}|{}|{:.6}",
        slash.node_id, slash.reason, slash.timestamp.unix, slash.penalty_score
    )
}

fn node_role_label(role: &NodeRole) -> &'static str {
    match role {
        NodeRole::Validator => "VALIDATOR",
        NodeRole::Worker => "WORKER",
        NodeRole::Sensor => "SENSOR",
    }
}

pub fn node_registration_payload(registration: &NodeRegistration) -> String {
    format!(
        "register|{}|{}|{}|{}",
        registration.node_id,
        node_role_label(&registration.role),
        registration.wallet_address,
        registration.wallet_public_key
    )
}

pub trait BlockchainLedger: Send + Sync {
    fn register_node(&self, registration: NodeRegistration) -> Result<()>;
    fn submit_sensor_commitment(&self, commitment: SensorCommitment) -> Result<()>;
    fn submit_energy_sample(&self, sample: EnergyEfficiencySample) -> Result<()>;
    fn submit_reward_event(&self, event: RewardEvent) -> Result<()>;
    fn reward_balance(&self, node_id: &str) -> Result<RewardBalance>;
    fn submit_stake_deposit(&self, _deposit: StakeDeposit) -> Result<()> {
        anyhow::bail!("stake deposit submission not implemented")
    }
    fn submit_bridge_proof(&self, _proof: BridgeProof) -> Result<()> {
        anyhow::bail!("bridge proof submission not implemented")
    }

    fn submit_bridge_intent(&self, _intent: BridgeIntent) -> Result<BridgeIntent> {
        anyhow::bail!("bridge intent submission not implemented")
    }

    fn submit_work_proof(&self, _proof: WorkProof) -> Result<()> {
        anyhow::bail!("work proof submission not implemented")
    }

    fn submit_cross_chain_transfer(&self, _transfer: CrossChainTransfer) -> Result<()> {
        anyhow::bail!("cross-chain transfer not implemented")
    }

    fn submit_validator_heartbeat(&self, _heartbeat: ValidatorHeartbeat) -> Result<()> {
        anyhow::bail!("validator heartbeat not implemented")
    }

    fn validator_record(&self, _node_id: &str) -> Result<ValidatorRecord> {
        anyhow::bail!("validator record not implemented")
    }
}

#[derive(Debug, Default)]
pub struct NoopLedger;

impl BlockchainLedger for NoopLedger {
    fn register_node(&self, _registration: NodeRegistration) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_sensor_commitment(&self, _commitment: SensorCommitment) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_energy_sample(&self, _sample: EnergyEfficiencySample) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_reward_event(&self, _event: RewardEvent) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn reward_balance(&self, _node_id: &str) -> Result<RewardBalance> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_stake_deposit(&self, _deposit: StakeDeposit) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_bridge_proof(&self, _proof: BridgeProof) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_bridge_intent(&self, _intent: BridgeIntent) -> Result<BridgeIntent> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_work_proof(&self, _proof: WorkProof) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }

    fn submit_cross_chain_transfer(&self, _transfer: CrossChainTransfer) -> Result<()> {
        anyhow::bail!("ledger not configured")
    }
}

pub struct RewardPolicy {
    cfg: RewardPolicyConfig,
}

impl RewardPolicy {
    pub fn new(cfg: RewardPolicyConfig) -> Self {
        Self { cfg }
    }

    pub fn score_event(&self, event: &RewardEvent) -> f64 {
        let weight = match event.kind {
            RewardEventKind::SensorContribution => self.cfg.sensor_reward_weight,
            RewardEventKind::ComputeContribution => self.cfg.compute_reward_weight,
            RewardEventKind::EnergyEfficiency => self.cfg.energy_efficiency_weight,
            RewardEventKind::Uptime => self.cfg.uptime_reward_weight,
        };
        event.score * weight
    }
}

pub struct BlockchainRuntime {
    cfg: BlockchainConfig,
}

impl BlockchainRuntime {
    pub fn new(cfg: BlockchainConfig) -> Self {
        Self { cfg }
    }

    pub fn enabled(&self) -> bool {
        self.cfg.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_policy_scales_score() {
        let cfg = RewardPolicyConfig {
            sensor_reward_weight: 2.0,
            compute_reward_weight: 1.0,
            energy_efficiency_weight: 1.0,
            uptime_reward_weight: 1.0,
        };
        let policy = RewardPolicy::new(cfg);
        let event = RewardEvent {
            node_id: "n1".to_string(),
            kind: RewardEventKind::SensorContribution,
            timestamp: Timestamp { unix: 0 },
            score: 1.5,
        };
        let scored = policy.score_event(&event);
        assert!((scored - 3.0).abs() < 1e-6);
    }
}
