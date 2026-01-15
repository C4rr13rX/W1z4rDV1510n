use anyhow::{Result, anyhow};
use blake2::{Blake2s256, Digest};
use serde::{Deserialize, Serialize};

pub const NETWORK_ENVELOPE_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEnvelope {
    pub version: u16,
    pub message_id: String,
    pub timestamp_unix: i64,
    pub payload_kind: String,
    pub payload_hex: String,
    pub payload_hash: String,
    #[serde(default)]
    pub public_key: String,
    #[serde(default)]
    pub signature: String,
}

impl NetworkEnvelope {
    pub fn new(payload_kind: impl Into<String>, payload: &[u8], timestamp_unix: i64) -> Self {
        let payload_kind = payload_kind.into();
        let payload_hash = compute_payload_hash(payload);
        let payload_hex = hex_encode(payload);
        let message_id = compute_message_id(
            NETWORK_ENVELOPE_VERSION,
            timestamp_unix,
            &payload_kind,
            &payload_hash,
            "",
        );
        Self {
            version: NETWORK_ENVELOPE_VERSION,
            message_id,
            timestamp_unix,
            payload_kind,
            payload_hex,
            payload_hash,
            public_key: String::new(),
            signature: String::new(),
        }
    }

    pub fn signing_payload(&self) -> String {
        format!(
            "net|{}|{}|{}|{}|{}",
            self.version,
            self.message_id,
            self.timestamp_unix,
            self.payload_kind,
            self.payload_hash
        )
    }

    pub fn expected_message_id(&self) -> String {
        compute_message_id(
            self.version,
            self.timestamp_unix,
            &self.payload_kind,
            &self.payload_hash,
            &self.public_key,
        )
    }

    pub fn payload_bytes(&self) -> Result<Vec<u8>> {
        if self.payload_hex.is_empty() {
            anyhow::bail!("payload_hex must be non-empty");
        }
        let bytes = hex_decode(&self.payload_hex)?;
        let hash = compute_payload_hash(&bytes);
        if hash != self.payload_hash {
            anyhow::bail!("payload hash mismatch");
        }
        Ok(bytes)
    }

    pub fn validate_basic(&self) -> Result<()> {
        if self.version != NETWORK_ENVELOPE_VERSION {
            anyhow::bail!("unsupported network envelope version");
        }
        if self.message_id.trim().is_empty() {
            anyhow::bail!("message_id must be non-empty");
        }
        if self.payload_kind.trim().is_empty() {
            anyhow::bail!("payload_kind must be non-empty");
        }
        if !is_valid_kind(&self.payload_kind) {
            anyhow::bail!("payload_kind must be ascii alnum/._-");
        }
        let _ = self.payload_bytes()?;
        let expected = self.expected_message_id();
        if expected != self.message_id {
            anyhow::bail!("message_id mismatch");
        }
        Ok(())
    }
}

pub fn compute_payload_hash(payload: &[u8]) -> String {
    let mut hasher = Blake2s256::new();
    hasher.update(payload);
    let digest = hasher.finalize();
    hex_encode(&digest)
}

pub fn compute_message_id(
    version: u16,
    timestamp_unix: i64,
    payload_kind: &str,
    payload_hash: &str,
    public_key: &str,
) -> String {
    let marker = format!(
        "net|{}|{}|{}|{}|{}",
        version, timestamp_unix, payload_kind, payload_hash, public_key
    );
    let mut hasher = Blake2s256::new();
    hasher.update(marker.as_bytes());
    let digest = hasher.finalize();
    hex_encode(&digest)
}

fn is_valid_kind(value: &str) -> bool {
    value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
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
