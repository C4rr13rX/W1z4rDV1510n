use crate::config::ChainSpecConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainSpec {
    pub chain_id: String,
    pub consensus: String,
    pub genesis: GenesisConfig,
    pub reward_contract: RewardContractSpec,
    pub bridge_contract: BridgeContractSpec,
    pub token_standard: TokenStandardSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisConfig {
    pub chain_id: String,
    pub consensus: String,
    pub initial_validators: Vec<String>,
    pub token_symbol: String,
    pub initial_supply: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardContractSpec {
    pub contract_id: String,
    pub version: String,
    pub reward_policy: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeContractSpec {
    pub contract_id: String,
    pub version: String,
    pub supported_chains: Vec<String>,
    pub token_mapping: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStandardSpec {
    pub symbol: String,
    pub name: String,
    pub decimals: u8,
    pub transferable: bool,
}

impl ChainSpec {
    pub fn load(paths: &ChainSpecConfig) -> Result<Self> {
        let genesis: GenesisConfig = read_json(&paths.genesis_path)?;
        let reward_contract: RewardContractSpec = read_json(&paths.reward_contract_path)?;
        let bridge_contract: BridgeContractSpec = read_json(&paths.bridge_contract_path)?;
        let token_standard: TokenStandardSpec = read_json(&paths.token_standard_path)?;
        Ok(Self {
            chain_id: genesis.chain_id.clone(),
            consensus: genesis.consensus.clone(),
            genesis,
            reward_contract,
            bridge_contract,
            token_standard,
        })
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T> {
    let raw = fs::read_to_string(path)?;
    let parsed = serde_json::from_str(&raw)?;
    Ok(parsed)
}
