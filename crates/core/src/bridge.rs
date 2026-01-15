use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ChainKind {
    Evm,
    Solana,
    Bitcoin,
    Cosmos,
    Other,
}

impl Default for ChainKind {
    fn default() -> Self {
        ChainKind::Other
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BridgeVerificationMode {
    RelayerQuorum,
    Optimistic,
    LightClient,
    ZkProof,
}

impl Default for BridgeVerificationMode {
    fn default() -> Self {
        BridgeVerificationMode::RelayerQuorum
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeDeposit {
    pub chain_id: String,
    #[serde(default)]
    pub chain_kind: ChainKind,
    pub tx_hash: String,
    pub log_index: u64,
    pub asset: String,
    pub amount: f64,
    pub recipient_node_id: String,
    pub observed_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeSignature {
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BridgeVerification {
    RelayerQuorum { signatures: Vec<BridgeSignature> },
    Optimistic {
        relayer: BridgeSignature,
        challenge_expires_at: Timestamp,
    },
    LightClient { proof_hex: String },
    ZkProof { proof_hex: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeProof {
    pub deposit: BridgeDeposit,
    pub verification: BridgeVerification,
}

pub fn bridge_deposit_payload(deposit: &BridgeDeposit) -> String {
    format!(
        "bridge|{}|{:?}|{}|{}|{}|{:.6}|{}|{}",
        deposit.chain_id,
        deposit.chain_kind,
        deposit.tx_hash,
        deposit.log_index,
        deposit.asset,
        deposit.amount,
        deposit.recipient_node_id,
        deposit.observed_at.unix
    )
}

pub fn bridge_deposit_id(deposit: &BridgeDeposit) -> String {
    format!("{}:{}:{}", deposit.chain_id, deposit.tx_hash, deposit.log_index)
}
