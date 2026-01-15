use crate::config::WalletConfig;
use anyhow::{anyhow, Context, Result};
use argon2::{Argon2, Params, Version};
use blake2::{Blake2s256, Digest};
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::ChaCha20Poly1305;
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletFile {
    pub address: String,
    pub public_key: String,
    pub secret_key: String,
    pub created_at_unix: u64,
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
            return Ok(WalletInfo {
                address: wallet.address,
                public_key: wallet.public_key,
                path,
            });
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
}

pub fn node_id_from_wallet(address: &str) -> String {
    let suffix = address.strip_prefix("w1z").unwrap_or(address);
    let short = &suffix[..suffix.len().min(12)];
    format!("node-{}", short)
}

fn generate_wallet() -> WalletFile {
    let mut secret = [0u8; 32];
    OsRng.fill_bytes(&mut secret);
    let public = hash_bytes(&secret);
    let address = format!("w1z{}", hex_encode(&hash_bytes(&public)[0..20]));
    WalletFile {
        address,
        public_key: hex_encode(&public),
        secret_key: hex_encode(&secret),
        created_at_unix: unix_time(),
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

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
