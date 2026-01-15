use crate::config::ChainSpecConfig;
use anyhow::{Result, ensure};
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
        let spec = Self {
            chain_id: genesis.chain_id.clone(),
            consensus: genesis.consensus.clone(),
            genesis,
            reward_contract,
            bridge_contract,
            token_standard,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.chain_id.trim().is_empty(), "chain_id must be non-empty");
        ensure!(!self.consensus.trim().is_empty(), "consensus must be non-empty");
        ensure!(
            self.chain_id == self.genesis.chain_id,
            "genesis.chain_id must match chain_id"
        );
        ensure!(
            self.consensus == self.genesis.consensus,
            "genesis.consensus must match consensus"
        );
        ensure!(
            !self.genesis.token_symbol.trim().is_empty(),
            "genesis.token_symbol must be non-empty"
        );
        ensure!(
            self.token_standard.symbol == self.genesis.token_symbol,
            "token_standard.symbol must match genesis token_symbol"
        );
        ensure!(
            !self.token_standard.name.trim().is_empty(),
            "token_standard.name must be non-empty"
        );
        ensure!(
            !self.reward_contract.contract_id.trim().is_empty(),
            "reward_contract.contract_id must be non-empty"
        );
        ensure!(
            !self.reward_contract.version.trim().is_empty(),
            "reward_contract.version must be non-empty"
        );
        ensure!(
            !self.bridge_contract.contract_id.trim().is_empty(),
            "bridge_contract.contract_id must be non-empty"
        );
        ensure!(
            !self.bridge_contract.version.trim().is_empty(),
            "bridge_contract.version must be non-empty"
        );
        if self.consensus.eq_ignore_ascii_case("poa") {
            ensure!(
                !self.genesis.initial_validators.is_empty(),
                "genesis.initial_validators must be non-empty for poa"
            );
        }
        if self.token_standard.transferable {
            ensure!(
                !self.bridge_contract.supported_chains.is_empty(),
                "bridge_contract.supported_chains must be non-empty when token is transferable"
            );
        }
        Ok(())
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T> {
    let raw = fs::read_to_string(path)?;
    let parsed = serde_json::from_str(&raw)?;
    Ok(parsed)
}
