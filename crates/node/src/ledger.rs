use anyhow::{anyhow, Context, Result};
use blake2::{Blake2s256, Digest};
use crate::wallet::address_from_public_key;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, CrossChainTransfer, EnergyEfficiencySample, NodeRegistration, RewardBalance,
    RewardEvent, RewardEventKind, SensorCommitment, WorkKind, WorkProof,
    cross_chain_transfer_payload, node_registration_payload, sensor_commitment_payload,
    work_proof_payload,
};
use w1z4rdv1510n::schema::Timestamp;

const MAX_LOG_ITEMS: usize = 10_000;

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct LocalLedgerState {
    nodes: HashMap<String, NodeRegistration>,
    balances: HashMap<String, RewardBalance>,
    reward_events: Vec<RewardEvent>,
    work_proofs: Vec<WorkProof>,
    work_id_queue: VecDeque<String>,
    work_id_set: HashSet<String>,
    sensor_commitments: Vec<SensorCommitment>,
    sensor_commitment_ids: VecDeque<String>,
    sensor_commitment_id_set: HashSet<String>,
    energy_samples: Vec<EnergyEfficiencySample>,
    cross_chain_transfers: Vec<CrossChainTransfer>,
    cross_chain_transfer_ids: VecDeque<String>,
    cross_chain_transfer_id_set: HashSet<String>,
    audit_log: Vec<AuditEvent>,
    audit_head: String,
    audit_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditEvent {
    sequence: u64,
    timestamp: Timestamp,
    kind: String,
    event_id: String,
    prev_hash: String,
    hash: String,
}

pub struct LocalLedger {
    state: Mutex<LocalLedgerState>,
    path: PathBuf,
}

impl LocalLedger {
    pub fn load_or_create(path: PathBuf) -> Result<Self> {
        if path.exists() {
            let mut state = read_state(&path)?;
            rebuild_indices(&mut state);
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
        if registration.node_id.trim().is_empty() {
            anyhow::bail!("node id must be non-empty");
        }
        if registration.wallet_public_key.trim().is_empty() {
            anyhow::bail!("wallet public key must be provided");
        }
        if registration.wallet_address.trim().is_empty() {
            anyhow::bail!("wallet address must be provided");
        }
        if registration.capabilities.cpu_cores == 0 {
            anyhow::bail!("node must report at least one cpu core");
        }
        if !registration.capabilities.memory_gb.is_finite()
            || registration.capabilities.memory_gb <= 0.0
        {
            anyhow::bail!("node memory must be finite and > 0");
        }
        let derived_address = address_from_public_key_hex(&registration.wallet_public_key)?;
        if derived_address != registration.wallet_address {
            anyhow::bail!("wallet address does not match public key");
        }
        let payload = node_registration_payload(&registration);
        verify_signature(
            &registration.wallet_public_key,
            payload.as_bytes(),
            &registration.signature,
        )?;
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if let Some(existing) = state.nodes.get(&registration.node_id) {
            if !existing.wallet_public_key.trim().is_empty()
                && existing.wallet_public_key != registration.wallet_public_key
            {
                anyhow::bail!("node registration key mismatch");
            }
        }
        let node_id = registration.node_id.clone();
        let wallet_address = registration.wallet_address.clone();
        state.nodes.insert(node_id.clone(), registration);
        let entry = Self::balance_entry(&mut state, &node_id, &wallet_address);
        if entry.wallet_address.is_empty() {
            entry.wallet_address = wallet_address;
        }
        entry.updated_at = now();
        let audit_id = payload_id(payload.as_bytes());
        record_audit(&mut state, "REGISTER_NODE", audit_id, now());
        self.persist(&state)?;
        Ok(())
    }

    fn submit_sensor_commitment(&self, commitment: SensorCommitment) -> Result<()> {
        if commitment.node_id.trim().is_empty() {
            anyhow::bail!("sensor commitment node id must be non-empty");
        }
        if commitment.sensor_id.trim().is_empty() {
            anyhow::bail!("sensor commitment sensor id must be non-empty");
        }
        if commitment.payload_hash.trim().is_empty() {
            anyhow::bail!("sensor commitment payload hash must be non-empty");
        }
        let public_key = node_public_key(&commitment.node_id, &self.state)?;
        let payload = sensor_commitment_payload(&commitment);
        verify_signature(&public_key, payload.as_bytes(), &commitment.signature)?;
        let commitment_id = payload_id(payload.as_bytes());
        let audit_id = commitment_id.clone();
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if state.sensor_commitment_id_set.contains(&commitment_id) {
            anyhow::bail!("duplicate sensor commitment");
        }
        let state = &mut *state;
        push_event_with_id(
            &mut state.sensor_commitments,
            &mut state.sensor_commitment_ids,
            &mut state.sensor_commitment_id_set,
            commitment,
            commitment_id,
        );
        if let Some(last) = state.sensor_commitments.last() {
            record_audit(state, "SENSOR_COMMITMENT", audit_id, last.timestamp.clone());
        }
        self.persist(state)?;
        Ok(())
    }

    fn submit_energy_sample(&self, sample: EnergyEfficiencySample) -> Result<()> {
        if sample.node_id.trim().is_empty() {
            anyhow::bail!("energy sample node id must be non-empty");
        }
        if !sample.watts.is_finite() || !sample.throughput.is_finite() {
            anyhow::bail!("energy sample values must be finite");
        }
        if sample.watts.is_sign_negative() || sample.throughput.is_sign_negative() {
            anyhow::bail!("energy sample values must be non-negative");
        }
        let payload = energy_sample_payload(&sample);
        let timestamp = sample.timestamp;
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        state.energy_samples.push(sample);
        trim_events(&mut state.energy_samples);
        let audit_id = payload_id(payload.as_bytes());
        record_audit(&mut state, "ENERGY_SAMPLE", audit_id, timestamp);
        self.persist(&state)?;
        Ok(())
    }

    fn submit_reward_event(&self, event: RewardEvent) -> Result<()> {
        if event.node_id.trim().is_empty() {
            anyhow::bail!("reward event node id must be non-empty");
        }
        if !event.score.is_finite() {
            anyhow::bail!("reward score must be finite");
        }
        if event.score.is_sign_negative() {
            anyhow::bail!("reward score must be non-negative");
        }
        let payload = reward_event_payload(&event);
        let timestamp = event.timestamp;
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
        let audit_id = payload_id(payload.as_bytes());
        record_audit(&mut state, "REWARD_EVENT", audit_id, timestamp);
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
        if proof.node_id.trim().is_empty() {
            anyhow::bail!("work proof node id must be non-empty");
        }
        if !proof.score.is_finite() {
            anyhow::bail!("work proof score must be finite");
        }
        if proof.score.is_sign_negative() {
            anyhow::bail!("work proof score must be non-negative");
        }
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if state.work_id_set.contains(&proof.work_id) {
            anyhow::bail!("duplicate work proof id");
        }
        let public_key = node_public_key_locked(&proof.node_id, &state)?;
        let payload = work_proof_payload(&proof);
        verify_signature(&public_key, payload.as_bytes(), &proof.signature)?;
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
        let reward_payload = reward_event_payload(&reward_event);
        let reward_timestamp = reward_event.timestamp;
        let wallet_address = state
            .nodes
            .get(&proof.node_id)
            .map(|reg| reg.wallet_address.clone())
            .unwrap_or_default();
        let entry = Self::balance_entry(&mut state, &proof.node_id, &wallet_address);
        entry.balance += proof.score;
        entry.updated_at = reward_timestamp;
        state.reward_events.push(reward_event);
        trim_events(&mut state.reward_events);
        let reward_audit_id = payload_id(reward_payload.as_bytes());
        record_audit(&mut state, "REWARD_EVENT", reward_audit_id, reward_timestamp);
        let proof_id = proof.work_id.clone();
        let state = &mut *state;
        push_event_with_id(
            &mut state.work_proofs,
            &mut state.work_id_queue,
            &mut state.work_id_set,
            proof,
            proof_id,
        );
        let proof_audit_id = payload_id(payload.as_bytes());
        record_audit(state, "WORK_PROOF", proof_audit_id, reward_timestamp);
        self.persist(state)?;
        Ok(())
    }

    fn submit_cross_chain_transfer(&self, transfer: CrossChainTransfer) -> Result<()> {
        if transfer.node_id.trim().is_empty() {
            anyhow::bail!("cross-chain transfer node id must be non-empty");
        }
        if transfer.amount == 0 {
            anyhow::bail!("cross-chain transfer amount must be > 0");
        }
        if transfer.source_chain.trim().is_empty() || transfer.target_chain.trim().is_empty() {
            anyhow::bail!("cross-chain transfer requires chain identifiers");
        }
        if transfer.token_symbol.trim().is_empty() {
            anyhow::bail!("cross-chain transfer requires token symbol");
        }
        if transfer.payload_hash.trim().is_empty() {
            anyhow::bail!("cross-chain transfer payload hash must be non-empty");
        }
        let public_key = node_public_key(&transfer.node_id, &self.state)?;
        let payload = cross_chain_transfer_payload(&transfer);
        verify_signature(&public_key, payload.as_bytes(), &transfer.signature)?;
        let transfer_id = payload_id(payload.as_bytes());
        let audit_id = transfer_id.clone();
        let timestamp = transfer.timestamp;
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if state.cross_chain_transfer_id_set.contains(&transfer_id) {
            anyhow::bail!("duplicate cross-chain transfer");
        }
        let state = &mut *state;
        push_event_with_id(
            &mut state.cross_chain_transfers,
            &mut state.cross_chain_transfer_ids,
            &mut state.cross_chain_transfer_id_set,
            transfer,
            transfer_id,
        );
        record_audit(state, "CROSS_CHAIN_TRANSFER", audit_id, timestamp);
        self.persist(state)?;
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

fn rebuild_indices(state: &mut LocalLedgerState) {
    trim_events(&mut state.work_proofs);
    trim_events(&mut state.sensor_commitments);
    trim_events(&mut state.cross_chain_transfers);
    trim_events(&mut state.reward_events);
    trim_events(&mut state.energy_samples);
    trim_events(&mut state.audit_log);

    state.work_id_queue.clear();
    state.work_id_set.clear();
    for proof in &state.work_proofs {
        if state.work_id_set.insert(proof.work_id.clone()) {
            state.work_id_queue.push_back(proof.work_id.clone());
        }
    }

    state.sensor_commitment_ids.clear();
    state.sensor_commitment_id_set.clear();
    for commitment in &state.sensor_commitments {
        let payload = sensor_commitment_payload(commitment);
        let id = payload_id(payload.as_bytes());
        if state.sensor_commitment_id_set.insert(id.clone()) {
            state.sensor_commitment_ids.push_back(id);
        }
    }

    state.cross_chain_transfer_ids.clear();
    state.cross_chain_transfer_id_set.clear();
    for transfer in &state.cross_chain_transfers {
        let payload = cross_chain_transfer_payload(transfer);
        let id = payload_id(payload.as_bytes());
        if state.cross_chain_transfer_id_set.insert(id.clone()) {
            state.cross_chain_transfer_ids.push_back(id);
        }
    }

    rebuild_audit(state);
}

fn rebuild_audit(state: &mut LocalLedgerState) {
    if let Some(last) = state.audit_log.last() {
        state.audit_head = last.hash.clone();
        state.audit_sequence = last.sequence;
    }
}

fn push_event_with_id<T>(
    items: &mut Vec<T>,
    ids: &mut VecDeque<String>,
    set: &mut HashSet<String>,
    item: T,
    id: String,
) {
    items.push(item);
    ids.push_back(id.clone());
    set.insert(id);
    if items.len() > MAX_LOG_ITEMS {
        let excess = items.len() - MAX_LOG_ITEMS;
        items.drain(0..excess);
        for _ in 0..excess {
            if let Some(old) = ids.pop_front() {
                set.remove(&old);
            }
        }
    }
}

fn record_audit(
    state: &mut LocalLedgerState,
    kind: &str,
    event_id: String,
    timestamp: Timestamp,
) {
    let sequence = state.audit_sequence.saturating_add(1);
    let prev_hash = state.audit_head.clone();
    let hash = audit_hash(sequence, &timestamp, kind, &event_id, &prev_hash);
    let event = AuditEvent {
        sequence,
        timestamp,
        kind: kind.to_string(),
        event_id,
        prev_hash,
        hash: hash.clone(),
    };
    state.audit_sequence = sequence;
    state.audit_head = hash;
    state.audit_log.push(event);
    trim_events(&mut state.audit_log);
}

fn payload_id(payload: &[u8]) -> String {
    let mut hasher = Blake2s256::new();
    hasher.update(payload);
    let digest = hasher.finalize();
    hex_encode(&digest)
}

fn audit_hash(
    sequence: u64,
    timestamp: &Timestamp,
    kind: &str,
    event_id: &str,
    prev_hash: &str,
) -> String {
    let marker = format!(
        "audit|{}|{}|{}|{}|{}",
        sequence, timestamp.unix, kind, event_id, prev_hash
    );
    payload_id(marker.as_bytes())
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

fn address_from_public_key_hex(public_key_hex: &str) -> Result<String> {
    let bytes = decode_public_key_bytes(public_key_hex)?;
    Ok(address_from_public_key(&bytes))
}

fn reward_event_payload(event: &RewardEvent) -> String {
    format!(
        "reward|{}|{:?}|{}|{:.6}",
        event.node_id, event.kind, event.timestamp.unix, event.score
    )
}

fn energy_sample_payload(sample: &EnergyEfficiencySample) -> String {
    format!(
        "energy|{}|{}|{:.6}|{:.6}",
        sample.node_id, sample.timestamp.unix, sample.watts, sample.throughput
    )
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
    let bytes = decode_public_key_bytes(hex)?;
    VerifyingKey::from_bytes(&bytes).map_err(|err| anyhow!("invalid public key: {err}"))
}

fn decode_public_key_bytes(hex: &str) -> Result<[u8; 32]> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("public key must be 32 bytes");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
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
        let public_bytes = signing_key.verifying_key().to_bytes();
        let public_key = hex_encode(public_bytes.as_slice());
        let wallet_address = address_from_public_key(&public_bytes);
        let mut registration = NodeRegistration {
            node_id: "n1".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Worker,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address,
            wallet_public_key: public_key.clone(),
            signature: String::new(),
        };
        let registration_payload = node_registration_payload(&registration);
        let registration_signature = signing_key.sign(registration_payload.as_bytes()).to_bytes();
        registration.signature = hex_encode(&registration_signature);
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
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        let mut signed = proof.clone();
        signed.signature = hex_encode(&signature);
        ledger.submit_work_proof(signed.clone()).expect("first");
        let second = ledger.submit_work_proof(signed);
        assert!(second.is_err());
    }

    #[test]
    fn accepts_signed_cross_chain_transfer() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let ledger = LocalLedger::load_or_create(path).expect("ledger");
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let public_bytes = signing_key.verifying_key().to_bytes();
        let public_key = hex_encode(public_bytes.as_slice());
        let wallet_address = address_from_public_key(&public_bytes);
        let mut registration = NodeRegistration {
            node_id: "n1".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Worker,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address,
            wallet_public_key: public_key.clone(),
            signature: String::new(),
        };
        let registration_payload = node_registration_payload(&registration);
        let registration_signature = signing_key.sign(registration_payload.as_bytes()).to_bytes();
        registration.signature = hex_encode(&registration_signature);
        ledger.register_node(registration).expect("register");
        let transfer = CrossChainTransfer {
            node_id: "n1".to_string(),
            source_chain: "w1z".to_string(),
            target_chain: "eth".to_string(),
            token_symbol: "W1Z".to_string(),
            amount: 100,
            payload_hash: "payload".to_string(),
            timestamp: Timestamp { unix: 10 },
            signature: String::new(),
        };
        let payload = cross_chain_transfer_payload(&transfer);
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        let mut signed = transfer.clone();
        signed.signature = hex_encode(&signature);
        ledger
            .submit_cross_chain_transfer(signed)
            .expect("transfer");
    }
}
