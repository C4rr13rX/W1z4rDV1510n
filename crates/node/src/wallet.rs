use crate::config::WalletConfig;
use anyhow::{anyhow, Context, Result};
use argon2::{Argon2, Params, Version};
use blake2::{Blake2s256, Digest};
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::ChaCha20Poly1305;
use ed25519_dalek::{Signature, SigningKey, Signer};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use w1z4rdv1510n::blockchain::{
    CrossChainTransfer, NodeRegistration, SensorCommitment, ValidatorHeartbeat, WorkProof,
    cross_chain_transfer_payload, node_registration_payload, sensor_commitment_payload,
    validator_heartbeat_payload, work_proof_payload,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletFile {
    pub address: String,
    pub public_key: String,
    pub secret_key: String,
    pub created_at_unix: u64,
    #[serde(default)]
    pub key_type: WalletKeyType,
    #[serde(default)]
    pub legacy_address: Option<String>,
    #[serde(default)]
    pub legacy_public_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WalletKeyType {
    Ed25519,
    LegacySeed,
}

impl Default for WalletKeyType {
    fn default() -> Self {
        WalletKeyType::Ed25519
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "SCREAMING_SNAKE_CASE")]
enum WalletDisk {
    Plain { wallet: WalletFile },
    Encrypted { encrypted: EncryptedWalletFile },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedWalletFile {
    pub kdf: KdfParams,
    pub nonce_hex: String,
    pub ciphertext_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdfParams {
    pub salt_hex: String,
    pub mem_kib: u32,
    pub iterations: u32,
    pub parallelism: u32,
}

#[derive(Debug, Clone)]
pub struct WalletInfo {
    pub address: String,
    pub public_key: String,
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct WalletSigner {
    info: WalletInfo,
    signing_key: SigningKey,
}

impl WalletSigner {
    pub fn wallet(&self) -> &WalletInfo {
        &self.info
    }

    pub fn sign_payload(&self, payload: &[u8]) -> String {
        let signature: Signature = self.signing_key.sign(payload);
        hex_encode(&signature.to_bytes())
    }

    pub fn sign_work_proof(&self, mut proof: WorkProof) -> WorkProof {
        let payload = work_proof_payload(&proof);
        proof.signature = self.sign_payload(payload.as_bytes());
        proof
    }

    pub fn sign_node_registration(&self, mut registration: NodeRegistration) -> NodeRegistration {
        let payload = node_registration_payload(&registration);
        registration.signature = self.sign_payload(payload.as_bytes());
        registration
    }

    pub fn sign_validator_heartbeat(&self, mut heartbeat: ValidatorHeartbeat) -> ValidatorHeartbeat {
        let payload = validator_heartbeat_payload(&heartbeat);
        heartbeat.signature = self.sign_payload(payload.as_bytes());
        heartbeat
    }

    pub fn sign_sensor_commitment(&self, mut commitment: SensorCommitment) -> SensorCommitment {
        let payload = sensor_commitment_payload(&commitment);
        commitment.signature = self.sign_payload(payload.as_bytes());
        commitment
    }

    pub fn sign_cross_chain_transfer(&self, mut transfer: CrossChainTransfer) -> CrossChainTransfer {
        let payload = cross_chain_transfer_payload(&transfer);
        transfer.signature = self.sign_payload(payload.as_bytes());
        transfer
    }
}

pub struct WalletStore;

impl WalletStore {
    pub fn load_or_create(config: &WalletConfig) -> Result<WalletInfo> {
        if !config.enabled {
            anyhow::bail!("wallet disabled in config");
        }
        let path = PathBuf::from(&config.path);
        if path.exists() {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("read wallet file {}", path.display()))?;
            let wallet = read_wallet(&raw, config, &path)?;
            let (wallet, info, changed) = normalize_wallet(wallet, &path)?;
            if changed {
                write_wallet(&path, &wallet, config, false)?;
            }
            return Ok(info);
        }
        if !config.auto_create {
            anyhow::bail!("wallet file missing and auto_create is false");
        }
        let wallet = generate_wallet();
        write_wallet(&path, &wallet, config, true)?;
        Ok(WalletInfo {
            address: wallet.address,
            public_key: wallet.public_key,
            path,
        })
    }

    pub fn load_or_create_signer(config: &WalletConfig) -> Result<WalletSigner> {
        let info = Self::load_or_create(config)?;
        let raw = fs::read_to_string(&info.path)
            .with_context(|| format!("read wallet file {}", info.path.display()))?;
        let wallet = read_wallet(&raw, config, &info.path)?;
        let (wallet, info, changed) = normalize_wallet(wallet, &info.path)?;
        if changed {
            write_wallet(&info.path, &wallet, config, false)?;
        }
        let signing_key = signing_key_from_secret(&wallet.secret_key)?;
        Ok(WalletSigner { info, signing_key })
    }
}

pub fn node_id_from_wallet(address: &str) -> String {
    let suffix = address.strip_prefix("w1z").unwrap_or(address);
    let short = &suffix[..suffix.len().min(12)];
    format!("node-{}", short)
}

fn generate_wallet() -> WalletFile {
    let signing_key = SigningKey::generate(&mut OsRng);
    let secret = signing_key.to_bytes();
    let public = signing_key.verifying_key().to_bytes();
    let address = address_from_public_key(&public);
    WalletFile {
        address,
        public_key: hex_encode(&public),
        secret_key: hex_encode(&secret),
        created_at_unix: unix_time(),
        key_type: WalletKeyType::Ed25519,
        legacy_address: None,
        legacy_public_key: None,
    }
}

fn read_wallet(raw: &str, config: &WalletConfig, path: &Path) -> Result<WalletFile> {
    if let Ok(disk) = serde_json::from_str::<WalletDisk>(raw) {
        match disk {
            WalletDisk::Plain { wallet } => {
                if config.encrypted {
                    let passphrase = resolve_passphrase(config, false)?;
                    let encrypted = encrypt_wallet(&wallet, &passphrase)?;
                    write_wallet_disk(path, WalletDisk::Encrypted { encrypted })?;
                }
                return Ok(wallet);
            }
            WalletDisk::Encrypted { encrypted } => {
                let passphrase = resolve_passphrase(config, false)?;
                let wallet = decrypt_wallet(&encrypted, &passphrase)?;
                return Ok(wallet);
            }
        }
    }
    let wallet: WalletFile = serde_json::from_str(raw)
        .with_context(|| format!("parse wallet file {}", path.display()))?;
    if config.encrypted {
        let passphrase = resolve_passphrase(config, false)?;
        let encrypted = encrypt_wallet(&wallet, &passphrase)?;
        write_wallet_disk(path, WalletDisk::Encrypted { encrypted })?;
    }
    Ok(wallet)
}

fn write_wallet(path: &Path, wallet: &WalletFile, config: &WalletConfig, creating: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("create wallet directory {}", parent.display())
        })?;
    }
    let disk = if config.encrypted {
        let passphrase = resolve_passphrase(config, creating)?;
        let encrypted = encrypt_wallet(wallet, &passphrase)?;
        WalletDisk::Encrypted { encrypted }
    } else {
        WalletDisk::Plain {
            wallet: wallet.clone(),
        }
    };
    write_wallet_disk(path, disk)?;
    Ok(())
}

fn normalize_wallet(mut wallet: WalletFile, path: &Path) -> Result<(WalletFile, WalletInfo, bool)> {
    let signing_key = signing_key_from_secret(&wallet.secret_key)?;
    let public = signing_key.verifying_key().to_bytes();
    let public_hex = hex_encode(&public);
    let address = address_from_public_key(&public);
    let mut changed = false;
    if wallet.public_key != public_hex {
        if wallet.legacy_public_key.is_none() {
            wallet.legacy_public_key = Some(wallet.public_key.clone());
        }
        wallet.public_key = public_hex;
        changed = true;
    }
    if wallet.address != address {
        if wallet.legacy_address.is_none() {
            wallet.legacy_address = Some(wallet.address.clone());
        }
        wallet.address = address;
        changed = true;
    }
    if !matches!(wallet.key_type, WalletKeyType::Ed25519) {
        wallet.key_type = WalletKeyType::Ed25519;
        changed = true;
    }
    let info = WalletInfo {
        address: wallet.address.clone(),
        public_key: wallet.public_key.clone(),
        path: path.to_path_buf(),
    };
    Ok((wallet, info, changed))
}

fn signing_key_from_secret(secret_hex: &str) -> Result<SigningKey> {
    let bytes = decode_secret_key(secret_hex)?;
    Ok(SigningKey::from_bytes(&bytes))
}

fn write_wallet_disk(path: &Path, disk: WalletDisk) -> Result<()> {
    let payload = serde_json::to_string_pretty(&disk)?;
    fs::write(path, payload).with_context(|| format!("write wallet file {}", path.display()))?;
    Ok(())
}

fn resolve_passphrase(config: &WalletConfig, creating: bool) -> Result<String> {
    if !config.passphrase_env.trim().is_empty() {
        if let Ok(value) = std::env::var(&config.passphrase_env) {
            if !value.trim().is_empty() {
                return Ok(value);
            }
        }
    }
    if !config.prompt_on_load {
        anyhow::bail!("wallet passphrase required but prompting is disabled");
    }
    let passphrase = prompt_passphrase(creating)?;
    if passphrase.trim().is_empty() {
        anyhow::bail!("wallet passphrase cannot be empty");
    }
    Ok(passphrase)
}

fn prompt_passphrase(creating: bool) -> Result<String> {
    let first = rpassword::prompt_password("Enter wallet passphrase: ")?;
    if !creating {
        return Ok(first);
    }
    let second = rpassword::prompt_password("Confirm wallet passphrase: ")?;
    if first != second {
        anyhow::bail!("passphrases do not match");
    }
    Ok(first)
}

fn encrypt_wallet(wallet: &WalletFile, passphrase: &str) -> Result<EncryptedWalletFile> {
    let mut salt = [0u8; 16];
    OsRng.fill_bytes(&mut salt);
    let params =
        Params::new(65536, 3, 1, Some(32)).map_err(|err| anyhow!("argon2 params: {err}"))?;
    let mem_kib = params.m_cost();
    let iterations = params.t_cost();
    let parallelism = params.p_cost();
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    argon2
        .hash_password_into(passphrase.as_bytes(), &salt, &mut key)
        .map_err(|err| anyhow!("argon2 derive key: {err}"))?;
    let cipher = ChaCha20Poly1305::new((&key).into());
    let mut nonce = [0u8; 12];
    OsRng.fill_bytes(&mut nonce);
    let plaintext = serde_json::to_vec(wallet)?;
    let ciphertext = cipher.encrypt((&nonce).into(), plaintext.as_ref())?;
    Ok(EncryptedWalletFile {
        kdf: KdfParams {
            salt_hex: hex_encode(&salt),
            mem_kib,
            iterations,
            parallelism,
        },
        nonce_hex: hex_encode(&nonce),
        ciphertext_hex: hex_encode(&ciphertext),
    })
}

fn decrypt_wallet(encrypted: &EncryptedWalletFile, passphrase: &str) -> Result<WalletFile> {
    let salt = hex_decode(&encrypted.kdf.salt_hex)?;
    let nonce = hex_decode(&encrypted.nonce_hex)?;
    let ciphertext = hex_decode(&encrypted.ciphertext_hex)?;
    if salt.len() != 16 {
        anyhow::bail!("wallet salt must be 16 bytes");
    }
    if nonce.len() != 12 {
        anyhow::bail!("wallet nonce must be 12 bytes");
    }
    let params = Params::new(
        encrypted.kdf.mem_kib,
        encrypted.kdf.iterations,
        encrypted.kdf.parallelism,
        Some(32),
    )
    .map_err(|err| anyhow!("argon2 params: {err}"))?;
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    argon2
        .hash_password_into(passphrase.as_bytes(), &salt, &mut key)
        .map_err(|err| anyhow!("argon2 derive key: {err}"))?;
    let cipher = ChaCha20Poly1305::new((&key).into());
    let plaintext = cipher.decrypt((&nonce[..]).into(), ciphertext.as_ref())?;
    let wallet: WalletFile = serde_json::from_slice(&plaintext)?;
    Ok(wallet)
}

fn hash_bytes(input: &[u8]) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(input);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..32]);
    out
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

fn decode_secret_key(hex: &str) -> Result<[u8; 32]> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("secret key must be 32 bytes");
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

pub fn address_from_public_key(public_key: &[u8]) -> String {
    let hash = hash_bytes(public_key);
    format!("w1z{}", hex_encode(&hash[0..20]))
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Verifier;
    use std::collections::HashMap;
    use w1z4rdv1510n::blockchain::{ValidatorHeartbeat, WorkKind};
    use w1z4rdv1510n::config::NodeRole;
    use w1z4rdv1510n::schema::Timestamp;

    #[test]
    fn signer_produces_verifiable_signatures() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let public_bytes = signing_key.verifying_key().to_bytes();
        let info = WalletInfo {
            address: address_from_public_key(&public_bytes),
            public_key: hex_encode(&public_bytes),
            path: PathBuf::new(),
        };
        let signer = WalletSigner { info, signing_key };

        let proof = WorkProof {
            work_id: "w1".to_string(),
            node_id: "n1".to_string(),
            kind: WorkKind::ComputeTask,
            completed_at: Timestamp { unix: 1 },
            score: 1.0,
            metrics: HashMap::new(),
            signature: String::new(),
        };
        let signed_proof = signer.sign_work_proof(proof);
        assert_signature(&signer, work_proof_payload(&signed_proof), &signed_proof.signature);

        let registration = NodeRegistration {
            node_id: "n1".to_string(),
            role: NodeRole::Worker,
            capabilities: w1z4rdv1510n::blockchain::NodeCapabilities {
                cpu_cores: 1,
                memory_gb: 1.0,
                gpu_count: 0,
            },
            sensors: Vec::new(),
            wallet_address: signer.wallet().address.clone(),
            wallet_public_key: signer.wallet().public_key.clone(),
            signature: String::new(),
        };
        let signed_registration = signer.sign_node_registration(registration);
        assert_signature(
            &signer,
            node_registration_payload(&signed_registration),
            &signed_registration.signature,
        );

        let heartbeat = ValidatorHeartbeat {
            node_id: "n1".to_string(),
            timestamp: Timestamp { unix: 4 },
            signature: String::new(),
        };
        let signed_heartbeat = signer.sign_validator_heartbeat(heartbeat);
        assert_signature(
            &signer,
            validator_heartbeat_payload(&signed_heartbeat),
            &signed_heartbeat.signature,
        );

        let commitment = SensorCommitment {
            node_id: "n1".to_string(),
            sensor_id: "sensor-1".to_string(),
            timestamp: Timestamp { unix: 2 },
            payload_hash: "payload".to_string(),
            signature: String::new(),
        };
        let signed_commitment = signer.sign_sensor_commitment(commitment);
        assert_signature(
            &signer,
            sensor_commitment_payload(&signed_commitment),
            &signed_commitment.signature,
        );

        let transfer = CrossChainTransfer {
            node_id: "n1".to_string(),
            source_chain: "w1z".to_string(),
            target_chain: "eth".to_string(),
            token_symbol: "W1Z".to_string(),
            amount: 100,
            payload_hash: "payload".to_string(),
            timestamp: Timestamp { unix: 3 },
            signature: String::new(),
        };
        let signed_transfer = signer.sign_cross_chain_transfer(transfer);
        assert_signature(
            &signer,
            cross_chain_transfer_payload(&signed_transfer),
            &signed_transfer.signature,
        );
    }

    fn assert_signature(signer: &WalletSigner, payload: String, signature_hex: &str) {
        let signature_bytes = hex_decode(signature_hex).expect("decode signature");
        let signature = Signature::from_slice(&signature_bytes).expect("signature");
        signer
            .signing_key
            .verifying_key()
            .verify(payload.as_bytes(), &signature)
            .expect("verify signature");
    }
}
