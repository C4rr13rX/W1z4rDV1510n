use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
    work_ids: HashSet<String>,
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
        if commitment.payload_hash.trim().is_empty() {
            anyhow::bail!("sensor commitment payload hash must be non-empty");
        }
        let public_key = node_public_key(&commitment.node_id, &self.state)?;
        let payload = sensor_commitment_payload(&commitment);
        verify_signature(&public_key, &payload, &commitment.signature)?;
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        state.sensor_commitments.push(commitment);
        trim_events(&mut state.sensor_commitments);
        self.persist(&state)?;
        Ok(())
    }

    fn submit_energy_sample(&self, sample: EnergyEfficiencySample) -> Result<()> {
        if sample.watts.is_sign_negative() || sample.throughput.is_sign_negative() {
            anyhow::bail!("energy sample values must be non-negative");
        }
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        state.energy_samples.push(sample);
        trim_events(&mut state.energy_samples);
        self.persist(&state)?;
        Ok(())
    }

    fn submit_reward_event(&self, event: RewardEvent) -> Result<()> {
        if event.score.is_sign_negative() {
            anyhow::bail!("reward score must be non-negative");
        }
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
        if proof.work_id.trim().is_empty() {
            anyhow::bail!("work proof id must be non-empty");
        }
        if proof.score.is_sign_negative() {
            anyhow::bail!("work proof score must be non-negative");
        }
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if state.work_ids.contains(&proof.work_id) {
            anyhow::bail!("duplicate work proof id");
        }
        let public_key = node_public_key_locked(&proof.node_id, &state)?;
        let payload = work_proof_payload(&proof);
        verify_signature(&public_key, &payload, &proof.signature)?;
        state.work_ids.insert(proof.work_id.clone());
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
        if transfer.amount == 0 {
            anyhow::bail!("cross-chain transfer amount must be > 0");
        }
        if transfer.source_chain.trim().is_empty() || transfer.target_chain.trim().is_empty() {
            anyhow::bail!("cross-chain transfer requires chain identifiers");
        }
        if transfer.token_symbol.trim().is_empty() {
            anyhow::bail!("cross-chain transfer requires token symbol");
        }
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

fn node_public_key(node_id: &str, state: &Mutex<LocalLedgerState>) -> Result<String> {
    let state = state.lock().expect("local ledger mutex poisoned");
    node_public_key_locked(node_id, &state)
}

fn node_public_key_locked(node_id: &str, state: &LocalLedgerState) -> Result<String> {
    let public_key = state
        .nodes
        .get(node_id)
        .map(|reg| reg.wallet_public_key.clone())
        .unwrap_or_default();
    if public_key.trim().is_empty() {
        anyhow::bail!("node public key missing");
    }
    Ok(public_key)
}

fn work_proof_payload(proof: &WorkProof) -> Vec<u8> {
    format!(
        "work|{}|{}|{:?}|{}|{:.6}",
        proof.work_id,
        proof.node_id,
        proof.kind,
        proof.completed_at.unix,
        proof.score
    )
    .into_bytes()
}

fn sensor_commitment_payload(commitment: &SensorCommitment) -> Vec<u8> {
    format!(
        "sensor|{}|{}|{}|{}",
        commitment.node_id, commitment.sensor_id, commitment.timestamp.unix, commitment.payload_hash
    )
    .into_bytes()
}

fn verify_signature(public_key_hex: &str, payload: &[u8], signature_hex: &str) -> Result<()> {
    if signature_hex.trim().is_empty() {
        anyhow::bail!("signature is required");
    }
    let public_key = decode_public_key(public_key_hex)?;
    let signature = decode_signature(signature_hex)?;
    public_key
        .verify(payload, &signature)
        .map_err(|err| anyhow!("signature verify failed: {err}"))?;
    Ok(())
}

fn decode_public_key(hex: &str) -> Result<VerifyingKey> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("public key must be 32 bytes");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    VerifyingKey::from_bytes(&arr).map_err(|err| anyhow!("invalid public key: {err}"))
}

fn decode_signature(hex: &str) -> Result<Signature> {
    let bytes = hex_decode(hex)?;
    Signature::from_slice(&bytes).map_err(|err| anyhow!("invalid signature: {err}"))
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.as_bytes().iter().copied();
    while let Some(high) = iter.next() {
        let low = iter
            .next()
            .ok_or_else(|| anyhow!("hex string has odd length"))?;
        let high_val = hex_value(high)?;
        let low_val = hex_value(low)?;
        out.push((high_val << 4) | low_val);
    }
    Ok(out)
}

fn hex_value(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => anyhow::bail!("invalid hex character"),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer;
    use tempfile::tempdir;

    #[test]
    fn rejects_duplicate_work_ids() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let ledger = LocalLedger::load_or_create(path).expect("ledger");
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let public_key = hex_encode(signing_key.verifying_key().to_bytes().as_slice());
        let registration = NodeRegistration {
            node_id: "n1".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Worker,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address: "w1ztest".to_string(),
            wallet_public_key: public_key.clone(),
        };
        ledger.register_node(registration).expect("register");
        let proof = WorkProof {
            work_id: "w1".to_string(),
            node_id: "n1".to_string(),
            kind: WorkKind::ComputeTask,
            completed_at: Timestamp { unix: 1 },
            score: 1.0,
            metrics: HashMap::new(),
            signature: String::new(),
        };
        let payload = work_proof_payload(&proof);
        let signature = signing_key.sign(&payload).to_bytes();
        let mut signed = proof.clone();
        signed.signature = hex_encode(&signature);
        ledger.submit_work_proof(signed.clone()).expect("first");
        let second = ledger.submit_work_proof(signed);
        assert!(second.is_err());
    }
}
