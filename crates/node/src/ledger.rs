use anyhow::{anyhow, Context, Result};
use blake2::{Blake2s256, Digest};
use crate::bridge::{BridgeVerifier, VerifiedBridgeDeposit};
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
    RewardEvent, RewardEventKind, SensorCommitment, StakeDeposit, WorkKind, WorkProof,
    ValidatorHeartbeat, ValidatorRecord, ValidatorSlashEvent, ValidatorSlashReason,
    ValidatorStatus,
    cross_chain_transfer_payload, node_registration_payload, sensor_commitment_payload,
    stake_deposit_payload, validator_heartbeat_payload, validator_slash_payload, work_proof_payload,
};
use w1z4rdv1510n::bridge::BridgeProof;
use w1z4rdv1510n::config::{BridgeConfig, FeeMarketConfig, ValidatorPolicyConfig};
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
    stake_deposits: Vec<StakeDeposit>,
    stake_deposit_ids: VecDeque<String>,
    stake_deposit_id_set: HashSet<String>,
    validators: HashMap<String, ValidatorRecord>,
    validator_heartbeats: Vec<ValidatorHeartbeat>,
    validator_heartbeat_ids: VecDeque<String>,
    validator_heartbeat_id_set: HashSet<String>,
    validator_slashes: Vec<ValidatorSlashEvent>,
    #[serde(default)]
    fee_market: FeeMarketState,
    audit_log: Vec<AuditEvent>,
    audit_head: String,
    audit_sequence: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct FeeMarketState {
    base_fee: f64,
    window_start_unix: i64,
    window_tx_count: u32,
    total_fees_collected: f64,
    pool_balance: f64,
    fees_by_node: HashMap<String, f64>,
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
    validator_policy: ValidatorPolicyConfig,
    fee_market: FeeMarketConfig,
    bridge: BridgeConfig,
}

impl LocalLedger {
    pub fn load_or_create(
        path: PathBuf,
        validator_policy: ValidatorPolicyConfig,
        fee_market: FeeMarketConfig,
        bridge: BridgeConfig,
    ) -> Result<Self> {
        if path.exists() {
            let mut state = read_state(&path)?;
            rebuild_indices(&mut state);
            normalize_fee_market_state(&mut state, &fee_market);
            return Ok(Self {
                state: Mutex::new(state),
                path,
                validator_policy,
                fee_market,
                bridge,
            });
        }
        let mut state = LocalLedgerState::default();
        initialize_fee_market_state(&mut state, &fee_market, now());
        write_state(&path, &state)?;
        Ok(Self {
            state: Mutex::new(state),
            path,
            validator_policy,
            fee_market,
            bridge,
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
        validate_fee_paid(commitment.fee_paid, "sensor commitment")?;
        let public_key = node_public_key(&commitment.node_id, &self.state)?;
        let payload = sensor_commitment_payload(&commitment);
        verify_signature(&public_key, payload.as_bytes(), &commitment.signature)?;
        let commitment_id = payload_id(payload.as_bytes());
        let audit_id = commitment_id.clone();
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if state.sensor_commitment_id_set.contains(&commitment_id) {
            anyhow::bail!("duplicate sensor commitment");
        }
        charge_fee(
            &mut *state,
            &self.fee_market,
            &commitment.node_id,
            commitment.timestamp,
            commitment.fee_paid,
            "SENSOR_COMMITMENT",
            0.0,
        )?;
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

    fn submit_stake_deposit(&self, deposit: StakeDeposit) -> Result<()> {
        if deposit.deposit_id.trim().is_empty() {
            anyhow::bail!("stake deposit id must be non-empty");
        }
        if deposit.node_id.trim().is_empty() {
            anyhow::bail!("stake deposit node id must be non-empty");
        }
        if !deposit.amount.is_finite() || deposit.amount <= 0.0 {
            anyhow::bail!("stake deposit amount must be > 0 and finite");
        }
        let public_key = node_public_key(&deposit.node_id, &self.state)?;
        let payload = stake_deposit_payload(&deposit);
        verify_signature(&public_key, payload.as_bytes(), &deposit.signature)?;
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        record_stake_deposit(&mut state, deposit, payload)?;
        self.persist(&state)?;
        Ok(())
    }

    fn submit_bridge_proof(&self, proof: BridgeProof) -> Result<()> {
        let verifier = BridgeVerifier::new(self.bridge.clone());
        let verified: VerifiedBridgeDeposit = verifier.verify(&proof)?;
        let deposit = StakeDeposit {
            deposit_id: verified.deposit_id,
            node_id: verified.node_id,
            amount: verified.amount,
            timestamp: verified.timestamp,
            source: verified.source,
            signature: String::new(),
        };
        let payload = stake_deposit_payload(&deposit);
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        record_stake_deposit(&mut state, deposit, payload)?;
        self.persist(&state)?;
        Ok(())
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
        validate_fee_paid(proof.fee_paid, "work proof")?;
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        if state.work_id_set.contains(&proof.work_id) {
            anyhow::bail!("duplicate work proof id");
        }
        let public_key = node_public_key_locked(&proof.node_id, &state)?;
        let payload = work_proof_payload(&proof);
        verify_signature(&public_key, payload.as_bytes(), &proof.signature)?;
        // Allow fees to be covered by the reward earned for this proof.
        charge_fee(
            &mut *state,
            &self.fee_market,
            &proof.node_id,
            proof.completed_at,
            proof.fee_paid,
            "WORK_PROOF",
            proof.score,
        )?;
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
        validate_fee_paid(transfer.fee_paid, "cross-chain transfer")?;
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
        charge_fee(
            &mut *state,
            &self.fee_market,
            &transfer.node_id,
            transfer.timestamp,
            transfer.fee_paid,
            "CROSS_CHAIN_TRANSFER",
            0.0,
        )?;
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

    fn submit_validator_heartbeat(&self, heartbeat: ValidatorHeartbeat) -> Result<()> {
        if heartbeat.node_id.trim().is_empty() {
            anyhow::bail!("validator heartbeat node id must be non-empty");
        }
        validate_fee_paid(heartbeat.fee_paid, "validator heartbeat")?;
        let public_key = node_public_key(&heartbeat.node_id, &self.state)?;
        let payload = validator_heartbeat_payload(&heartbeat);
        verify_signature(&public_key, payload.as_bytes(), &heartbeat.signature)?;
        let heartbeat_id = payload_id(payload.as_bytes());
        let audit_id = heartbeat_id.clone();
        let mut state = self.state.lock().expect("local ledger mutex poisoned");
        ensure_validator_node(&state, &heartbeat.node_id)?;
        if state.validator_heartbeat_id_set.contains(&heartbeat_id) {
            anyhow::bail!("duplicate validator heartbeat");
        }
        let heartbeat_timestamp = heartbeat.timestamp;
        let mut slashed = None;
        {
            let record = state
                .validators
                .entry(heartbeat.node_id.clone())
                .or_insert_with(|| ValidatorRecord {
                    node_id: heartbeat.node_id.clone(),
                    status: ValidatorStatus::Active,
                    last_heartbeat: heartbeat_timestamp,
                    missed_heartbeats: 0,
                    jailed_until: None,
                });
            if record.status == ValidatorStatus::Jailed {
                if let Some(until) = record.jailed_until {
                    if heartbeat_timestamp.unix < until.unix {
                        anyhow::bail!("validator is jailed");
                    }
                }
                record.status = ValidatorStatus::Active;
                record.missed_heartbeats = 0;
                record.jailed_until = None;
            }
            let interval = self.validator_policy.heartbeat_interval_secs as i64;
            if interval <= 0 {
                anyhow::bail!("validator heartbeat interval must be > 0");
            }
            if heartbeat_timestamp.unix < record.last_heartbeat.unix {
                anyhow::bail!("validator heartbeat timestamp regression");
            }
            let delta = heartbeat_timestamp.unix - record.last_heartbeat.unix;
            let missed = (delta.saturating_sub(1) / interval) as u32;
            if missed == 0 {
                record.status = ValidatorStatus::Active;
                record.missed_heartbeats = 0;
            } else {
                record.status = ValidatorStatus::Inactive;
                record.missed_heartbeats = record.missed_heartbeats.saturating_add(missed);
            }
            record.last_heartbeat = heartbeat_timestamp;
            if record.missed_heartbeats >= self.validator_policy.max_missed_heartbeats {
                record.status = ValidatorStatus::Jailed;
                record.missed_heartbeats = 0;
                if self.validator_policy.jail_duration_secs > 0 {
                    record.jailed_until = Some(Timestamp {
                        unix: heartbeat_timestamp.unix
                            + self.validator_policy.jail_duration_secs as i64,
                    });
                } else {
                    record.jailed_until = None;
                }
                slashed = Some(ValidatorSlashEvent {
                    node_id: heartbeat.node_id.clone(),
                    timestamp: heartbeat_timestamp,
                    reason: ValidatorSlashReason::Downtime,
                    penalty_score: self.validator_policy.downtime_penalty_score,
                    signature: String::new(),
                });
            }
        }
        charge_fee(
            &mut *state,
            &self.fee_market,
            &heartbeat.node_id,
            heartbeat_timestamp,
            heartbeat.fee_paid,
            "VALIDATOR_HEARTBEAT",
            0.0,
        )?;
        let state = &mut *state;
        push_event_with_id(
            &mut state.validator_heartbeats,
            &mut state.validator_heartbeat_ids,
            &mut state.validator_heartbeat_id_set,
            heartbeat,
            heartbeat_id,
        );
        record_audit(state, "VALIDATOR_HEARTBEAT", audit_id, heartbeat_timestamp);
        if let Some(slash) = slashed {
            let slash_payload = validator_slash_payload(&slash);
            let slash_id = payload_id(slash_payload.as_bytes());
            state.validator_slashes.push(slash);
            trim_events(&mut state.validator_slashes);
            record_audit(state, "VALIDATOR_SLASH", slash_id, heartbeat_timestamp);
        }
        self.persist(state)?;
        Ok(())
    }

    fn validator_record(&self, node_id: &str) -> Result<ValidatorRecord> {
        let state = self.state.lock().expect("local ledger mutex poisoned");
        if let Some(record) = state.validators.get(node_id) {
            return Ok(record.clone());
        }
        if let Some(registration) = state.nodes.get(node_id) {
            if matches!(registration.role, w1z4rdv1510n::config::NodeRole::Validator) {
                return Ok(ValidatorRecord {
                    node_id: node_id.to_string(),
                    status: ValidatorStatus::Inactive,
                    last_heartbeat: Timestamp::default(),
                    missed_heartbeats: 0,
                    jailed_until: None,
                });
            }
        }
        anyhow::bail!("validator record not found");
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
    trim_events(&mut state.stake_deposits);
    trim_events(&mut state.validator_heartbeats);
    trim_events(&mut state.validator_slashes);
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

    state.stake_deposit_ids.clear();
    state.stake_deposit_id_set.clear();
    for deposit in &state.stake_deposits {
        let id = deposit.deposit_id.clone();
        if id.trim().is_empty() {
            continue;
        }
        if state.stake_deposit_id_set.insert(id.clone()) {
            state.stake_deposit_ids.push_back(id);
        }
    }

    state.validator_heartbeat_ids.clear();
    state.validator_heartbeat_id_set.clear();
    for heartbeat in &state.validator_heartbeats {
        let payload = validator_heartbeat_payload(heartbeat);
        let id = payload_id(payload.as_bytes());
        if state.validator_heartbeat_id_set.insert(id.clone()) {
            state.validator_heartbeat_ids.push_back(id);
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

fn record_stake_deposit(
    state: &mut LocalLedgerState,
    deposit: StakeDeposit,
    payload: String,
) -> Result<()> {
    if deposit.deposit_id.trim().is_empty() {
        anyhow::bail!("stake deposit id must be non-empty");
    }
    if deposit.node_id.trim().is_empty() {
        anyhow::bail!("stake deposit node id must be non-empty");
    }
    if !deposit.amount.is_finite() || deposit.amount <= 0.0 {
        anyhow::bail!("stake deposit amount must be > 0 and finite");
    }
    if state.stake_deposit_id_set.contains(&deposit.deposit_id) {
        anyhow::bail!("duplicate stake deposit id");
    }
    let wallet_address = state
        .nodes
        .get(&deposit.node_id)
        .map(|reg| reg.wallet_address.clone())
        .unwrap_or_default();
    let entry = LocalLedger::balance_entry(state, &deposit.node_id, &wallet_address);
    entry.balance += deposit.amount;
    entry.updated_at = deposit.timestamp;
    let deposit_id = deposit.deposit_id.clone();
    let timestamp = deposit.timestamp;
    push_event_with_id(
        &mut state.stake_deposits,
        &mut state.stake_deposit_ids,
        &mut state.stake_deposit_id_set,
        deposit,
        deposit_id,
    );
    let audit_id = payload_id(payload.as_bytes());
    record_audit(state, "STAKE_DEPOSIT", audit_id, timestamp);
    Ok(())
}

fn initialize_fee_market_state(
    state: &mut LocalLedgerState,
    config: &FeeMarketConfig,
    timestamp: Timestamp,
) {
    if !config.enabled {
        return;
    }
    state.fee_market.base_fee = clamp_base_fee(config.base_fee, config);
    state.fee_market.window_start_unix = timestamp.unix;
}

fn normalize_fee_market_state(state: &mut LocalLedgerState, config: &FeeMarketConfig) {
    if !config.enabled {
        return;
    }
    if state.fee_market.window_start_unix == 0 {
        state.fee_market.window_start_unix = now().unix;
    }
    if !state.fee_market.base_fee.is_finite() || state.fee_market.base_fee < 0.0 {
        state.fee_market.base_fee = config.base_fee;
    }
    if state.fee_market.base_fee == 0.0 && config.base_fee > 0.0 {
        state.fee_market.base_fee = config.base_fee;
    }
    state.fee_market.base_fee = clamp_base_fee(state.fee_market.base_fee, config);
    if !state.fee_market.pool_balance.is_finite() || state.fee_market.pool_balance < 0.0 {
        state.fee_market.pool_balance = 0.0;
    }
}

fn roll_fee_window_if_needed(
    state: &mut LocalLedgerState,
    config: &FeeMarketConfig,
    timestamp: Timestamp,
) {
    if !config.enabled {
        return;
    }
    let window_secs = config.window_secs as i64;
    if window_secs <= 0 {
        return;
    }
    if state.fee_market.window_start_unix == 0 {
        state.fee_market.window_start_unix = timestamp.unix;
        return;
    }
    let elapsed = timestamp.unix - state.fee_market.window_start_unix;
    if elapsed < 0 {
        state.fee_market.window_start_unix = timestamp.unix;
        state.fee_market.window_tx_count = 0;
        return;
    }
    if elapsed < window_secs {
        return;
    }
    let target = config.target_txs_per_window.max(1) as f64;
    let usage = state.fee_market.window_tx_count as f64 / target;
    let mut base_fee = state.fee_market.base_fee;
    if !base_fee.is_finite() {
        base_fee = config.base_fee;
    }
    let delta = (usage - 1.0) * config.adjustment_rate;
    base_fee *= 1.0 + delta;
    state.fee_market.base_fee = clamp_base_fee(base_fee, config);
    distribute_fee_pool(state, timestamp);
    state.fee_market.window_tx_count = 0;
    state.fee_market.window_start_unix = timestamp.unix;
}

fn charge_fee(
    state: &mut LocalLedgerState,
    config: &FeeMarketConfig,
    node_id: &str,
    timestamp: Timestamp,
    fee_paid: f64,
    kind: &str,
    pending_credit: f64,
) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    if !fee_paid.is_finite() || fee_paid.is_sign_negative() {
        anyhow::bail!("fee_paid must be non-negative and finite");
    }
    if !pending_credit.is_finite() || pending_credit.is_sign_negative() {
        anyhow::bail!("pending credit must be non-negative and finite");
    }
    if state.fee_market.window_start_unix == 0 {
        state.fee_market.window_start_unix = timestamp.unix;
    }
    if state.fee_market.base_fee == 0.0 && config.base_fee > 0.0 {
        state.fee_market.base_fee = config.base_fee;
    }
    state.fee_market.base_fee = clamp_base_fee(state.fee_market.base_fee, config);
    roll_fee_window_if_needed(state, config, timestamp);
    let base_fee = state.fee_market.base_fee;
    if fee_paid + 1e-9 < base_fee {
        anyhow::bail!("fee_paid below base fee");
    }
    let wallet_address = state
        .nodes
        .get(node_id)
        .map(|reg| reg.wallet_address.clone())
        .unwrap_or_default();
    let entry = LocalLedger::balance_entry(state, node_id, &wallet_address);
    let available = entry.balance + pending_credit;
    if fee_paid > available + 1e-9 {
        anyhow::bail!("insufficient balance to cover fee");
    }
    entry.balance -= fee_paid;
    entry.updated_at = timestamp;
    state.fee_market.window_tx_count = state.fee_market.window_tx_count.saturating_add(1);
    state.fee_market.total_fees_collected += fee_paid;
    state.fee_market.pool_balance += fee_paid;
    *state
        .fee_market
        .fees_by_node
        .entry(node_id.to_string())
        .or_insert(0.0) += fee_paid;
    let fee_payload = fee_event_payload(node_id, timestamp, fee_paid, base_fee, kind);
    let fee_id = payload_id(fee_payload.as_bytes());
    record_audit(state, "FEE_CHARGE", fee_id, timestamp);
    Ok(())
}

fn clamp_base_fee(value: f64, config: &FeeMarketConfig) -> f64 {
    let mut fee = if value.is_finite() { value } else { config.base_fee };
    if fee.is_sign_negative() {
        fee = 0.0;
    }
    if config.min_base_fee <= config.max_base_fee {
        fee = fee.clamp(config.min_base_fee, config.max_base_fee);
    }
    fee
}

fn fee_event_payload(
    node_id: &str,
    timestamp: Timestamp,
    fee_paid: f64,
    base_fee: f64,
    kind: &str,
) -> String {
    format!(
        "fee|{}|{}|{:.6}|{:.6}|{}",
        node_id, timestamp.unix, fee_paid, base_fee, kind
    )
}

fn distribute_fee_pool(state: &mut LocalLedgerState, timestamp: Timestamp) {
    if !state.fee_market.pool_balance.is_finite() || state.fee_market.pool_balance <= 0.0 {
        return;
    }
    let mut active_validators: Vec<String> = state
        .validators
        .iter()
        .filter(|(_, record)| {
            record.status == ValidatorStatus::Active
                && record
                    .jailed_until
                    .map(|until| timestamp.unix >= until.unix)
                    .unwrap_or(true)
        })
        .map(|(node_id, _)| node_id.clone())
        .collect();
    if active_validators.is_empty() {
        return;
    }
    active_validators.sort();
    let count = active_validators.len() as f64;
    let share = state.fee_market.pool_balance / count;
    if !share.is_finite() || share <= 0.0 {
        return;
    }
    for node_id in active_validators {
        let wallet_address = state
            .nodes
            .get(&node_id)
            .map(|reg| reg.wallet_address.clone())
            .unwrap_or_default();
        let entry = LocalLedger::balance_entry(state, &node_id, &wallet_address);
        entry.balance += share;
        entry.updated_at = timestamp;
        let reward_event = RewardEvent {
            node_id: node_id.clone(),
            kind: RewardEventKind::Uptime,
            timestamp,
            score: share,
        };
        let reward_payload = reward_event_payload(&reward_event);
        state.reward_events.push(reward_event);
        trim_events(&mut state.reward_events);
        let audit_id = payload_id(reward_payload.as_bytes());
        record_audit(state, "FEE_DISTRIBUTION", audit_id, timestamp);
    }
    let distributed_total = share * count;
    state.fee_market.pool_balance = (state.fee_market.pool_balance - distributed_total).max(0.0);
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

fn ensure_validator_node(state: &LocalLedgerState, node_id: &str) -> Result<()> {
    let role = state
        .nodes
        .get(node_id)
        .map(|reg| reg.role.clone())
        .ok_or_else(|| anyhow!("validator node not registered"))?;
    if !matches!(role, w1z4rdv1510n::config::NodeRole::Validator) {
        anyhow::bail!("node is not a validator");
    }
    Ok(())
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

fn validate_fee_paid(fee_paid: f64, label: &str) -> Result<()> {
    if !fee_paid.is_finite() {
        anyhow::bail!("{label} fee_paid must be finite");
    }
    if fee_paid.is_sign_negative() {
        anyhow::bail!("{label} fee_paid must be non-negative");
    }
    Ok(())
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
        let ledger = LocalLedger::load_or_create(
            path,
            ValidatorPolicyConfig::default(),
            FeeMarketConfig::default(),
            BridgeConfig::default(),
        )
        .expect("ledger");
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
            fee_paid: 0.0,
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
        let ledger = LocalLedger::load_or_create(
            path,
            ValidatorPolicyConfig::default(),
            FeeMarketConfig::default(),
            BridgeConfig::default(),
        )
        .expect("ledger");
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
            fee_paid: 0.0,
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

    #[test]
    fn validator_heartbeat_jails_after_missed_intervals() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let policy = ValidatorPolicyConfig {
            heartbeat_interval_secs: 10,
            max_missed_heartbeats: 2,
            jail_duration_secs: 60,
            downtime_penalty_score: 1.0,
        };
        let ledger = LocalLedger::load_or_create(
            path,
            policy,
            FeeMarketConfig::default(),
            BridgeConfig::default(),
        )
        .expect("ledger");
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let public_bytes = signing_key.verifying_key().to_bytes();
        let public_key = hex_encode(public_bytes.as_slice());
        let wallet_address = address_from_public_key(&public_bytes);
        let mut registration = NodeRegistration {
            node_id: "val1".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Validator,
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

        let mut heartbeat = ValidatorHeartbeat {
            node_id: "val1".to_string(),
            timestamp: Timestamp { unix: 0 },
            fee_paid: 0.0,
            signature: String::new(),
        };
        let payload = validator_heartbeat_payload(&heartbeat);
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        heartbeat.signature = hex_encode(&signature);
        ledger
            .submit_validator_heartbeat(heartbeat)
            .expect("heartbeat");

        let mut late = ValidatorHeartbeat {
            node_id: "val1".to_string(),
            timestamp: Timestamp { unix: 25 },
            fee_paid: 0.0,
            signature: String::new(),
        };
        let payload = validator_heartbeat_payload(&late);
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        late.signature = hex_encode(&signature);
        ledger
            .submit_validator_heartbeat(late)
            .expect("late heartbeat");

        let record = ledger.validator_record("val1").expect("record");
        assert!(matches!(record.status, ValidatorStatus::Jailed));
        assert!(record.jailed_until.is_some());
    }

    #[test]
    fn fee_market_rejects_underpaying_work_proof() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let policy = ValidatorPolicyConfig::default();
        let fee_market = FeeMarketConfig {
            enabled: true,
            base_fee: 1.0,
            min_base_fee: 1.0,
            max_base_fee: 5.0,
            target_txs_per_window: 10,
            window_secs: 60,
            adjustment_rate: 0.1,
        };
        let ledger =
            LocalLedger::load_or_create(path, policy, fee_market, BridgeConfig::default())
                .expect("ledger");
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
            fee_paid: 0.5,
            metrics: HashMap::new(),
            signature: String::new(),
        };
        let payload = work_proof_payload(&proof);
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        let mut signed = proof.clone();
        signed.signature = hex_encode(&signature);
        let result = ledger.submit_work_proof(signed);
        assert!(result.is_err());
    }

    #[test]
    fn fee_charge_debits_balance() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let policy = ValidatorPolicyConfig::default();
        let fee_market = FeeMarketConfig {
            enabled: true,
            base_fee: 1.0,
            min_base_fee: 1.0,
            max_base_fee: 5.0,
            target_txs_per_window: 10,
            window_secs: 60,
            adjustment_rate: 0.1,
        };
        let ledger =
            LocalLedger::load_or_create(path, policy, fee_market, BridgeConfig::default())
                .expect("ledger");
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

        let reward = RewardEvent {
            node_id: "n1".to_string(),
            kind: RewardEventKind::ComputeContribution,
            timestamp: Timestamp { unix: 1 },
            score: 2.0,
        };
        ledger.submit_reward_event(reward).expect("reward");

        let commitment = SensorCommitment {
            node_id: "n1".to_string(),
            sensor_id: "sensor-1".to_string(),
            timestamp: Timestamp { unix: 2 },
            payload_hash: "payload".to_string(),
            fee_paid: 1.0,
            signature: String::new(),
        };
        let payload = sensor_commitment_payload(&commitment);
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        let mut signed = commitment.clone();
        signed.signature = hex_encode(&signature);
        ledger
            .submit_sensor_commitment(signed)
            .expect("commitment");

        let balance = ledger.reward_balance("n1").expect("balance");
        assert!((balance.balance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fee_pool_distributes_to_active_validators() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let policy = ValidatorPolicyConfig::default();
        let fee_market = FeeMarketConfig {
            enabled: true,
            base_fee: 0.0,
            min_base_fee: 0.0,
            max_base_fee: 5.0,
            target_txs_per_window: 10,
            window_secs: 10,
            adjustment_rate: 0.0,
        };
        let ledger =
            LocalLedger::load_or_create(path, policy, fee_market, BridgeConfig::default())
                .expect("ledger");

        let signing_key_val1 = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let public_bytes_val1 = signing_key_val1.verifying_key().to_bytes();
        let public_key_val1 = hex_encode(public_bytes_val1.as_slice());
        let wallet_address_val1 = address_from_public_key(&public_bytes_val1);
        let mut registration_val1 = NodeRegistration {
            node_id: "val1".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Validator,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address: wallet_address_val1,
            wallet_public_key: public_key_val1.clone(),
            signature: String::new(),
        };
        let payload = node_registration_payload(&registration_val1);
        let signature = signing_key_val1.sign(payload.as_bytes()).to_bytes();
        registration_val1.signature = hex_encode(&signature);
        ledger.register_node(registration_val1).expect("register val1");

        let mut heartbeat_val1 = ValidatorHeartbeat {
            node_id: "val1".to_string(),
            timestamp: Timestamp { unix: 1 },
            fee_paid: 0.0,
            signature: String::new(),
        };
        let payload = validator_heartbeat_payload(&heartbeat_val1);
        let signature = signing_key_val1.sign(payload.as_bytes()).to_bytes();
        heartbeat_val1.signature = hex_encode(&signature);
        ledger
            .submit_validator_heartbeat(heartbeat_val1)
            .expect("heartbeat val1");

        let signing_key_val2 = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let public_bytes_val2 = signing_key_val2.verifying_key().to_bytes();
        let public_key_val2 = hex_encode(public_bytes_val2.as_slice());
        let wallet_address_val2 = address_from_public_key(&public_bytes_val2);
        let mut registration_val2 = NodeRegistration {
            node_id: "val2".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Validator,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address: wallet_address_val2,
            wallet_public_key: public_key_val2.clone(),
            signature: String::new(),
        };
        let payload = node_registration_payload(&registration_val2);
        let signature = signing_key_val2.sign(payload.as_bytes()).to_bytes();
        registration_val2.signature = hex_encode(&signature);
        ledger.register_node(registration_val2).expect("register val2");

        let mut heartbeat_val2 = ValidatorHeartbeat {
            node_id: "val2".to_string(),
            timestamp: Timestamp { unix: 1 },
            fee_paid: 0.0,
            signature: String::new(),
        };
        let payload = validator_heartbeat_payload(&heartbeat_val2);
        let signature = signing_key_val2.sign(payload.as_bytes()).to_bytes();
        heartbeat_val2.signature = hex_encode(&signature);
        ledger
            .submit_validator_heartbeat(heartbeat_val2)
            .expect("heartbeat val2");

        let signing_key_worker = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let public_bytes_worker = signing_key_worker.verifying_key().to_bytes();
        let public_key_worker = hex_encode(public_bytes_worker.as_slice());
        let wallet_address_worker = address_from_public_key(&public_bytes_worker);
        let mut registration_worker = NodeRegistration {
            node_id: "worker1".to_string(),
            role: w1z4rdv1510n::config::NodeRole::Worker,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address: wallet_address_worker,
            wallet_public_key: public_key_worker.clone(),
            signature: String::new(),
        };
        let payload = node_registration_payload(&registration_worker);
        let signature = signing_key_worker.sign(payload.as_bytes()).to_bytes();
        registration_worker.signature = hex_encode(&signature);
        ledger
            .register_node(registration_worker)
            .expect("register worker");

        let reward = RewardEvent {
            node_id: "worker1".to_string(),
            kind: RewardEventKind::ComputeContribution,
            timestamp: Timestamp { unix: 1 },
            score: 5.0,
        };
        ledger.submit_reward_event(reward).expect("reward");

        let commitment = SensorCommitment {
            node_id: "worker1".to_string(),
            sensor_id: "sensor-1".to_string(),
            timestamp: Timestamp { unix: 2 },
            payload_hash: "payload".to_string(),
            fee_paid: 4.0,
            signature: String::new(),
        };
        let payload = sensor_commitment_payload(&commitment);
        let signature = signing_key_worker.sign(payload.as_bytes()).to_bytes();
        let mut signed = commitment.clone();
        signed.signature = hex_encode(&signature);
        ledger
            .submit_sensor_commitment(signed)
            .expect("commitment");

        let rollover = SensorCommitment {
            node_id: "worker1".to_string(),
            sensor_id: "sensor-2".to_string(),
            timestamp: Timestamp { unix: 15 },
            payload_hash: "payload2".to_string(),
            fee_paid: 0.0,
            signature: String::new(),
        };
        let payload = sensor_commitment_payload(&rollover);
        let signature = signing_key_worker.sign(payload.as_bytes()).to_bytes();
        let mut signed = rollover.clone();
        signed.signature = hex_encode(&signature);
        ledger
            .submit_sensor_commitment(signed)
            .expect("rollover commitment");

        let balance_val1 = ledger.reward_balance("val1").expect("balance val1");
        let balance_val2 = ledger.reward_balance("val2").expect("balance val2");
        assert!((balance_val1.balance - 2.0).abs() < 1e-6);
        assert!((balance_val2.balance - 2.0).abs() < 1e-6);

        let balance_worker = ledger.reward_balance("worker1").expect("balance worker");
        assert!((balance_worker.balance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stake_deposit_increases_balance_and_rejects_duplicate() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let ledger = LocalLedger::load_or_create(
            path,
            ValidatorPolicyConfig::default(),
            FeeMarketConfig::default(),
            BridgeConfig::default(),
        )
        .expect("ledger");
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

        let deposit = StakeDeposit {
            deposit_id: "dep1".to_string(),
            node_id: "n1".to_string(),
            amount: 3.0,
            timestamp: Timestamp { unix: 5 },
            source: "faucet".to_string(),
            signature: String::new(),
        };
        let payload = stake_deposit_payload(&deposit);
        let signature = signing_key.sign(payload.as_bytes()).to_bytes();
        let mut signed = deposit.clone();
        signed.signature = hex_encode(&signature);
        ledger.submit_stake_deposit(signed.clone()).expect("deposit");

        let balance = ledger.reward_balance("n1").expect("balance");
        assert!((balance.balance - 3.0).abs() < 1e-6);

        let duplicate = ledger.submit_stake_deposit(signed);
        assert!(duplicate.is_err());
    }

    #[test]
    fn bridge_proof_credits_deposit() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("ledger.json");
        let relayer_key1 = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let relayer_key2 = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let relayer_pub1 = hex_encode(relayer_key1.verifying_key().to_bytes().as_slice());
        let relayer_pub2 = hex_encode(relayer_key2.verifying_key().to_bytes().as_slice());
        let bridge = BridgeConfig {
            enabled: true,
            challenge_window_secs: 3600,
            max_proof_bytes: 262_144,
            chains: vec![w1z4rdv1510n::config::BridgeChainPolicy {
                chain_id: "ethereum".to_string(),
                chain_kind: w1z4rdv1510n::bridge::ChainKind::Evm,
                verification: w1z4rdv1510n::bridge::BridgeVerificationMode::RelayerQuorum,
                min_confirmations: 12,
                relayer_quorum: 2,
                relayer_public_keys: vec![relayer_pub1.clone(), relayer_pub2.clone()],
                allowed_assets: vec!["USDC".to_string()],
                max_deposit_amount: 10000.0,
            }],
        };
        let ledger = LocalLedger::load_or_create(
            path,
            ValidatorPolicyConfig::default(),
            FeeMarketConfig::default(),
            bridge,
        )
        .expect("ledger");
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

        let deposit = w1z4rdv1510n::bridge::BridgeDeposit {
            chain_id: "ethereum".to_string(),
            chain_kind: w1z4rdv1510n::bridge::ChainKind::Evm,
            tx_hash: "0xabc123".to_string(),
            log_index: 1,
            asset: "USDC".to_string(),
            amount: 25.0,
            recipient_node_id: "n1".to_string(),
            observed_at: Timestamp { unix: 10 },
        };
        let payload = w1z4rdv1510n::bridge::bridge_deposit_payload(&deposit);
        let sig1 = relayer_key1.sign(payload.as_bytes()).to_bytes();
        let sig2 = relayer_key2.sign(payload.as_bytes()).to_bytes();
        let proof = w1z4rdv1510n::bridge::BridgeProof {
            deposit,
            verification: w1z4rdv1510n::bridge::BridgeVerification::RelayerQuorum {
                signatures: vec![
                    w1z4rdv1510n::bridge::BridgeSignature {
                        public_key: relayer_pub1,
                        signature: hex_encode(&sig1),
                    },
                    w1z4rdv1510n::bridge::BridgeSignature {
                        public_key: relayer_pub2,
                        signature: hex_encode(&sig2),
                    },
                ],
            },
        };
        ledger.submit_bridge_proof(proof).expect("bridge proof");

        let balance = ledger.reward_balance("n1").expect("balance");
        assert!((balance.balance - 25.0).abs() < 1e-6);
    }
}
