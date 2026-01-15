use crate::config::{BlockchainConfig, NodeRole, RewardPolicyConfig};
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
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkKind {
    SensorIngest,
    ComputeTask,
    ModelUpdate,
    CausalDiscovery,
    Forecasting,
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
    pub metrics: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainTransfer {
    pub source_chain: String,
    pub target_chain: String,
    pub token_symbol: String,
    pub amount: u64,
    pub payload_hash: String,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub signature: String,
}

pub trait BlockchainLedger: Send + Sync {
    fn register_node(&self, registration: NodeRegistration) -> Result<()>;
    fn submit_sensor_commitment(&self, commitment: SensorCommitment) -> Result<()>;
    fn submit_energy_sample(&self, sample: EnergyEfficiencySample) -> Result<()>;
    fn submit_reward_event(&self, event: RewardEvent) -> Result<()>;
    fn reward_balance(&self, node_id: &str) -> Result<RewardBalance>;

    fn submit_work_proof(&self, _proof: WorkProof) -> Result<()> {
        anyhow::bail!("work proof submission not implemented")
    }

    fn submit_cross_chain_transfer(&self, _transfer: CrossChainTransfer) -> Result<()> {
        anyhow::bail!("cross-chain transfer not implemented")
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
