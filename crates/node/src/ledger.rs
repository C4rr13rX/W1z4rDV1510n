use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, CrossChainTransfer, EnergyEfficiencySample, NodeRegistration, RewardBalance,
    RewardEvent, RewardEventKind, SensorCommitment, WorkKind, WorkProof,
};
use w1z4rdv1510n::schema::Timestamp;

const MAX_LOG_ITEMS: usize = 10_000;

#[derive(Debug, Default, Serialize, Deserialize)]
struct LocalLedgerState {
    nodes: HashMap<String, NodeRegistration>,
    balances: HashMap<String, RewardBalance>,
    reward_events: Vec<RewardEvent>,
    work_proofs: Vec<WorkProof>,
    sensor_commitments: Vec<SensorCommitment>,
    energy_samples: Vec<EnergyEfficiencySample>,
    cross_chain_transfers: Vec<CrossChainTransfer>,
}

pub struct LocalLedger {
    state: Mutex<LocalLedgerState>,
    path: PathBuf,
}

impl LocalLedger {
    pub fn load_or_create(path: PathBuf) -> Result<Self> {
        if path.exists() {
            let state = read_state(&path)?;
            return Ok(Self {
                state: Mutex::new(state),
                path,
            });
        }
        let state = LocalLedgerState::default();
        write_state(&path, &state)?;
        Ok(Self {
            state: Mutex::new(state),
            path,
        })
    }

    fn persist(&self, state: &LocalLedgerState) -> Result<()> {
        write_state(&self.path, state)
    }

    fn balance_entry<'a>(
        state: &'a mut LocalLedgerState,
        node_id: &str,
        wallet_address: &str,
    ) -> &'a mut RewardBalance {
        state
            .balances
            .entry(node_id.to_string())
            .or_insert_with(|| RewardBalance {
                node_id: node_id.to_string(),
                balance: 0.0,
                updated_at: now(),
                wallet_address: wallet_address.to_string(),
            })
    }
}

impl BlockchainLedger for LocalLedger {
    fn register_node(&self, registration: NodeRegistration) -> Result<()> {
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        let node_id = registration.node_id.clone();
        let wallet_address = registration.wallet_address.clone();
        state.nodes.insert(node_id.clone(), registration);
        let entry = Self::balance_entry(&mut state, &node_id, &wallet_address);
        if entry.wallet_address.is_empty() {
            entry.wallet_address = wallet_address;
        }
        entry.updated_at = now();
        self.persist(&state)?;
        Ok(())
    }

    fn submit_sensor_commitment(&self, commitment: SensorCommitment) -> Result<()> {
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        state.sensor_commitments.push(commitment);
        trim_events(&mut state.sensor_commitments);
        self.persist(&state)?;
        Ok(())
    }

    fn submit_energy_sample(&self, sample: EnergyEfficiencySample) -> Result<()> {
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        state.energy_samples.push(sample);
        trim_events(&mut state.energy_samples);
        self.persist(&state)?;
        Ok(())
    }

    fn submit_reward_event(&self, event: RewardEvent) -> Result<()> {
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        let wallet_address = state
            .nodes
            .get(&event.node_id)
            .map(|reg| reg.wallet_address.clone())
            .unwrap_or_default();
        let entry = Self::balance_entry(&mut state, &event.node_id, &wallet_address);
        entry.balance += event.score;
        entry.updated_at = event.timestamp.clone();
        state.reward_events.push(event);
        trim_events(&mut state.reward_events);
        self.persist(&state)?;
        Ok(())
    }

    fn reward_balance(&self, node_id: &str) -> Result<RewardBalance> {
        let state = self.state.lock().expect("local ledger mutex poisoned");
        if let Some(balance) = state.balances.get(node_id) {
            return Ok(balance.clone());
        }
        let wallet_address = state
            .nodes
            .get(node_id)
            .map(|reg| reg.wallet_address.clone())
            .unwrap_or_default();
        Ok(RewardBalance {
            node_id: node_id.to_string(),
            balance: 0.0,
            updated_at: now(),
            wallet_address,
        })
    }

    fn submit_work_proof(&self, proof: WorkProof) -> Result<()> {
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        let reward_kind = match proof.kind {
            WorkKind::SensorIngest => RewardEventKind::SensorContribution,
            WorkKind::ComputeTask
            | WorkKind::ModelUpdate
            | WorkKind::CausalDiscovery
            | WorkKind::Forecasting => RewardEventKind::ComputeContribution,
        };
        let reward_event = RewardEvent {
            node_id: proof.node_id.clone(),
            kind: reward_kind,
            timestamp: proof.completed_at,
            score: proof.score,
        };
        let wallet_address = state
            .nodes
            .get(&proof.node_id)
            .map(|reg| reg.wallet_address.clone())
            .unwrap_or_default();
        let entry = Self::balance_entry(&mut state, &proof.node_id, &wallet_address);
        entry.balance += proof.score;
        entry.updated_at = reward_event.timestamp;
        state.reward_events.push(reward_event);
        trim_events(&mut state.reward_events);
        state.work_proofs.push(proof);
        trim_events(&mut state.work_proofs);
        self.persist(&state)?;
        Ok(())
    }

    fn submit_cross_chain_transfer(&self, transfer: CrossChainTransfer) -> Result<()> {
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        state.cross_chain_transfers.push(transfer);
        trim_events(&mut state.cross_chain_transfers);
        self.persist(&state)?;
        Ok(())
    }
}

fn read_state(path: &Path) -> Result<LocalLedgerState> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let state = serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    Ok(state)
}

fn write_state(path: &Path, state: &LocalLedgerState) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create ledger directory {}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(state)?;
    fs::write(path, payload).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn now() -> Timestamp {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    Timestamp { unix }
}

fn trim_events<T>(items: &mut Vec<T>) {
    if items.len() > MAX_LOG_ITEMS {
        let excess = items.len() - MAX_LOG_ITEMS;
        items.drain(0..excess);
    }
}
