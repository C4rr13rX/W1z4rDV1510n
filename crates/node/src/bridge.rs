use anyhow::{Result, anyhow};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json;
use std::collections::HashSet;
use w1z4rdv1510n::bridge::{
    BridgeProof, BridgeSignature, BridgeVerification, BridgeVerificationMode, bridge_deposit_id,
    bridge_deposit_payload,
};
use w1z4rdv1510n::config::{BridgeChainPolicy, BridgeConfig};
use w1z4rdv1510n::schema::Timestamp;

#[derive(Debug, Clone)]
pub struct VerifiedBridgeDeposit {
    pub deposit_id: String,
    pub node_id: String,
    pub amount: f64,
    pub timestamp: Timestamp,
    pub source: String,
}

pub struct BridgeVerifier {
    config: BridgeConfig,
}

impl BridgeVerifier {
    pub fn new(config: BridgeConfig) -> Self {
        Self { config }
    }

    pub fn verify(&self, proof: &BridgeProof) -> Result<VerifiedBridgeDeposit> {
        if !self.config.enabled {
            anyhow::bail!("bridge disabled");
        }
        let proof_bytes = serde_json::to_vec(proof)?;
        if proof_bytes.len() > self.config.max_proof_bytes {
            anyhow::bail!("bridge proof exceeds max_proof_bytes");
        }
        let deposit = &proof.deposit;
        if deposit.chain_id.trim().is_empty() {
            anyhow::bail!("bridge chain_id must be non-empty");
        }
        if deposit.tx_hash.trim().is_empty() {
            anyhow::bail!("bridge tx_hash must be non-empty");
        }
        if deposit.asset.trim().is_empty() {
            anyhow::bail!("bridge asset must be non-empty");
        }
        if deposit.recipient_node_id.trim().is_empty() {
            anyhow::bail!("bridge recipient_node_id must be non-empty");
        }
        if !deposit.amount.is_finite() || deposit.amount <= 0.0 {
            anyhow::bail!("bridge amount must be > 0 and finite");
        }
        let policy = self
            .find_chain_policy(&deposit.chain_id)
            .ok_or_else(|| anyhow!("bridge chain not supported"))?;
        if deposit.chain_kind != policy.chain_kind {
            anyhow::bail!("bridge chain_kind mismatch");
        }
        if deposit.amount > policy.max_deposit_amount {
            anyhow::bail!("bridge amount exceeds max_deposit_amount");
        }
        if !asset_allowed(&deposit.asset, &policy.allowed_assets) {
            anyhow::bail!("bridge asset not allowed");
        }
        match (&policy.verification, &proof.verification) {
            (BridgeVerificationMode::RelayerQuorum, BridgeVerification::RelayerQuorum { signatures }) => {
                verify_relayer_quorum(
                    signatures,
                    &policy.relayer_public_keys,
                    policy.relayer_quorum,
                    deposit,
                )?;
            }
            (BridgeVerificationMode::Optimistic, _) => {
                anyhow::bail!("optimistic bridge verification not implemented");
            }
            (BridgeVerificationMode::LightClient, _) => {
                anyhow::bail!("light client verification not implemented");
            }
            (BridgeVerificationMode::ZkProof, _) => {
                anyhow::bail!("zk proof verification not implemented");
            }
            _ => {
                anyhow::bail!("bridge verification mode mismatch");
            }
        }
        Ok(VerifiedBridgeDeposit {
            deposit_id: bridge_deposit_id(deposit),
            node_id: deposit.recipient_node_id.clone(),
            amount: deposit.amount,
            timestamp: deposit.observed_at,
            source: format!("bridge:{}:{}", deposit.chain_id, deposit.asset),
        })
    }

    fn find_chain_policy(&self, chain_id: &str) -> Option<&BridgeChainPolicy> {
        self.config
            .chains
            .iter()
            .find(|policy| policy.chain_id.trim().eq_ignore_ascii_case(chain_id.trim()))
    }
}

fn asset_allowed(asset: &str, allowed_assets: &[String]) -> bool {
    let target = asset.trim().to_ascii_uppercase();
    allowed_assets
        .iter()
        .any(|entry| entry.trim().eq_ignore_ascii_case(&target))
}

fn verify_relayer_quorum(
    signatures: &[BridgeSignature],
    allowed_public_keys: &[String],
    quorum: u32,
    deposit: &w1z4rdv1510n::bridge::BridgeDeposit,
) -> Result<()> {
    if allowed_public_keys.is_empty() || quorum == 0 {
        anyhow::bail!("bridge relayer quorum misconfigured");
    }
    let mut approved = HashSet::new();
    let allowed: HashSet<String> = allowed_public_keys
        .iter()
        .map(|key| key.trim().to_ascii_lowercase())
        .collect();
    let payload = bridge_deposit_payload(deposit);
    for signature in signatures {
        let public_key = signature.public_key.trim().to_ascii_lowercase();
        if !allowed.contains(&public_key) {
            continue;
        }
        if approved.contains(&public_key) {
            continue;
        }
        verify_signature(&public_key, payload.as_bytes(), &signature.signature)?;
        approved.insert(public_key);
    }
    if approved.len() < quorum as usize {
        anyhow::bail!("bridge relayer quorum not met");
    }
    Ok(())
}

fn verify_signature(public_key_hex: &str, payload: &[u8], signature_hex: &str) -> Result<()> {
    if signature_hex.trim().is_empty() {
        anyhow::bail!("bridge signature is required");
    }
    let public_key = decode_public_key(public_key_hex)?;
    let signature = decode_signature(signature_hex)?;
    public_key
        .verify(payload, &signature)
        .map_err(|err| anyhow!("bridge signature verify failed: {err}"))?;
    Ok(())
}

fn decode_public_key(hex: &str) -> Result<VerifyingKey> {
    let bytes = decode_public_key_bytes(hex)?;
    VerifyingKey::from_bytes(&bytes).map_err(|err| anyhow!("bridge invalid public key: {err}"))
}

fn decode_public_key_bytes(hex: &str) -> Result<[u8; 32]> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("bridge public key must be 32 bytes");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

fn decode_signature(hex: &str) -> Result<Signature> {
    let bytes = hex_decode(hex)?;
    Signature::from_slice(&bytes).map_err(|err| anyhow!("bridge invalid signature: {err}"))
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.as_bytes().iter().copied();
    while let Some(high) = iter.next() {
        let low = iter
            .next()
            .ok_or_else(|| anyhow!("bridge hex string has odd length"))?;
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
        _ => anyhow::bail!("bridge invalid hex character"),
    }
}
