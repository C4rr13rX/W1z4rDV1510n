use crate::config::IdentityConfig;
use crate::wallet::address_from_public_key;
use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use rand::{Rng, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use w1z4rdv1510n::blockchain::{IdentityBinding, identity_binding_payload};
use w1z4rdv1510n::math_toolbox as math;
use w1z4rdv1510n::network::compute_payload_hash;
use w1z4rdv1510n::schema::Timestamp;
use w1z4rdv1510n::streaming::NetworkPatternSummary;

const CODE_CHARS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityChallenge {
    pub challenge_id: String,
    pub thread_id: String,
    pub issued_at: Timestamp,
    pub expires_at: Timestamp,
    pub code: String,
    #[serde(default)]
    pub target_position: Option<[f64; 3]>,
    pub position_tolerance: f64,
    #[serde(default)]
    pub expected_signature: Vec<f64>,
    pub min_behavior_similarity: f64,
    pub min_match_score: f64,
    #[serde(default)]
    pub claim_label: Option<String>,
    pub instruction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityChallengeRequest {
    pub thread_id: String,
    #[serde(default)]
    pub opt_in: bool,
    #[serde(default)]
    pub force_new: bool,
    #[serde(default)]
    pub claim_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityChallengeResponse {
    pub status: String,
    pub challenge: IdentityChallenge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityEvidence {
    pub challenge_id: String,
    pub thread_id: String,
    pub observed_at: Timestamp,
    pub code: String,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    #[serde(default)]
    pub observed_signature: Vec<f64>,
    #[serde(default)]
    pub match_score: Option<f64>,
    #[serde(default)]
    pub sensor_id: Option<String>,
    #[serde(default)]
    pub origin_node_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityVerifyRequest {
    pub challenge_id: String,
    pub thread_id: String,
    pub wallet_address: String,
    pub wallet_public_key: String,
    pub wallet_signature: String,
    pub code: String,
    #[serde(default)]
    pub observed_at_unix: Option<i64>,
    #[serde(default)]
    pub position: Option<[f64; 3]>,
    #[serde(default)]
    pub observed_signature: Vec<f64>,
    #[serde(default)]
    pub match_score: Option<f64>,
    #[serde(default)]
    pub sensor_id: Option<String>,
    #[serde(default)]
    pub origin_node_id: Option<String>,
    #[serde(default)]
    pub claim_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityVerifyResponse {
    pub status: String,
    pub binding: Option<IdentityBinding>,
    #[serde(default)]
    pub evidence_hash: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityStatusResponse {
    pub status: String,
    pub binding: Option<IdentityBinding>,
}

#[derive(Debug)]
pub struct IdentityRuntime {
    config: IdentityConfig,
    challenges: HashMap<String, IdentityChallenge>,
    patterns: HashMap<String, NetworkPatternSummary>,
    thread_index: HashMap<String, String>,
}

impl IdentityRuntime {
    pub fn new(config: IdentityConfig) -> Self {
        Self {
            config,
            challenges: HashMap::new(),
            patterns: HashMap::new(),
            thread_index: HashMap::new(),
        }
    }

    pub fn update_patterns(&mut self, patterns: &[NetworkPatternSummary]) {
        if !self.config.enabled || patterns.is_empty() {
            return;
        }
        for pattern in patterns {
            if pattern.thread_id.trim().is_empty() {
                continue;
            }
            match self.patterns.get_mut(&pattern.thread_id) {
                Some(existing) => {
                    if should_replace_pattern(existing, pattern) {
                        *existing = pattern.clone();
                    }
                }
                None => {
                    self.patterns
                        .insert(pattern.thread_id.clone(), pattern.clone());
                }
            }
        }
        self.prune_patterns();
    }

    pub fn issue_challenge(
        &mut self,
        request: IdentityChallengeRequest,
        now: Timestamp,
    ) -> Result<IdentityChallenge> {
        if !self.config.enabled {
            anyhow::bail!("identity verification disabled");
        }
        if request.thread_id.trim().is_empty() {
            anyhow::bail!("thread_id must be provided");
        }
        if !request.opt_in {
            anyhow::bail!("opt_in must be true for identity challenges");
        }
        self.prune_expired(now);
        if !request.force_new {
            if let Some(challenge_id) = self.thread_index.get(&request.thread_id) {
                if let Some(existing) = self.challenges.get(challenge_id) {
                    if existing.expires_at.unix > now.unix {
                        return Ok(existing.clone());
                    }
                }
            }
        }
        let pattern = self
            .patterns
            .get(&request.thread_id)
            .ok_or_else(|| anyhow!("unknown thread_id"))?
            .clone();
        let code = generate_code(self.config.code_length.max(4));
        let issued_at = now;
        let expires_at = Timestamp {
            unix: issued_at.unix + self.config.challenge_ttl_secs as i64,
        };
        let challenge_id = compute_payload_hash(
            format!(
                "challenge|{}|{}|{}",
                request.thread_id, issued_at.unix, code
            )
            .as_bytes(),
        );
        let instruction = build_instruction(
            &code,
            pattern.position,
            self.config.position_tolerance,
            &pattern.behavior_signature,
        );
        let challenge = IdentityChallenge {
            challenge_id: challenge_id.clone(),
            thread_id: request.thread_id.clone(),
            issued_at,
            expires_at,
            code,
            target_position: pattern.position,
            position_tolerance: self.config.position_tolerance,
            expected_signature: pattern.behavior_signature.clone(),
            min_behavior_similarity: self.config.behavior_similarity_threshold,
            min_match_score: self.config.min_match_score,
            claim_label: request.claim_label.clone(),
            instruction,
        };
        self.thread_index
            .insert(request.thread_id.clone(), challenge_id.clone());
        self.challenges.insert(challenge_id, challenge.clone());
        self.prune_challenges();
        Ok(challenge)
    }

    pub fn verify(
        &mut self,
        request: IdentityVerifyRequest,
        now: Timestamp,
    ) -> Result<IdentityVerificationOutcome> {
        if !self.config.enabled {
            anyhow::bail!("identity verification disabled");
        }
        self.prune_expired(now);
        let Some(challenge) = self.challenges.get(&request.challenge_id) else {
            anyhow::bail!("challenge not found or expired");
        };
        if challenge.thread_id != request.thread_id {
            anyhow::bail!("thread_id does not match challenge");
        }
        if now.unix > challenge.expires_at.unix {
            anyhow::bail!("challenge expired");
        }
        if request.code.trim() != challenge.code {
            anyhow::bail!("verification code mismatch");
        }
        let observed_at = Timestamp {
            unix: request.observed_at_unix.unwrap_or(now.unix),
        };
        let evidence = IdentityEvidence {
            challenge_id: request.challenge_id.clone(),
            thread_id: request.thread_id.clone(),
            observed_at,
            code: request.code.clone(),
            position: request.position,
            observed_signature: request.observed_signature.clone(),
            match_score: request.match_score,
            sensor_id: request.sensor_id.clone(),
            origin_node_id: request.origin_node_id.clone(),
        };
        let behavior_similarity = behavior_similarity(
            &challenge.expected_signature,
            &evidence.observed_signature,
        );
        if behavior_similarity < self.config.behavior_similarity_threshold {
            anyhow::bail!("behavior similarity below threshold");
        }
        let spatial_score = spatial_score(
            challenge.target_position,
            evidence.position,
            challenge.position_tolerance,
        );
        if challenge.target_position.is_some() && spatial_score <= 0.0 {
            anyhow::bail!("position outside challenge tolerance");
        }
        let computed_match = match_score(behavior_similarity, spatial_score);
        if let Some(match_score) = evidence.match_score {
            if match_score < self.config.min_match_score {
                anyhow::bail!("match score below minimum threshold");
            }
        } else if computed_match < self.config.min_match_score {
            anyhow::bail!("match score below minimum threshold");
        }
        let evidence_hash = identity_evidence_hash(&evidence);
        let binding = IdentityBinding {
            thread_id: request.thread_id.clone(),
            wallet_address: request.wallet_address.clone(),
            wallet_public_key: request.wallet_public_key.clone(),
            challenge_id: request.challenge_id.clone(),
            evidence_hash: evidence_hash.clone(),
            verified_at: observed_at,
            signature: request.wallet_signature.clone(),
        };
        verify_wallet_binding(&binding)?;
        let updated_pattern = self.apply_identity_claim(
            &binding,
            request.claim_label.or_else(|| challenge.claim_label.clone()),
        );
        Ok(IdentityVerificationOutcome {
            binding,
            evidence_hash,
            updated_pattern,
        })
    }

    fn apply_identity_claim(
        &mut self,
        binding: &IdentityBinding,
        claim_label: Option<String>,
    ) -> Option<NetworkPatternSummary> {
        let pattern = self.patterns.get_mut(&binding.thread_id)?;
        if let Some(label) = claim_label {
            pattern.opt_in_claim = Some(label);
        } else if pattern.opt_in_claim.is_none() {
            pattern.opt_in_claim = Some(format!("wallet:{}", binding.wallet_address));
        }
        if !pattern.origin_nodes.iter().any(|node| node == "identity-api") {
            pattern.origin_nodes.push("identity-api".to_string());
        }
        Some(pattern.clone())
    }

    fn prune_expired(&mut self, now: Timestamp) {
        self.challenges
            .retain(|_, challenge| challenge.expires_at.unix > now.unix);
        self.thread_index.retain(|_, challenge_id| {
            self.challenges.contains_key(challenge_id)
        });
    }

    fn prune_challenges(&mut self) {
        let max_pending = self.config.max_pending_challenges.max(1);
        if self.challenges.len() <= max_pending {
            return;
        }
        let mut entries: Vec<_> = self
            .challenges
            .iter()
            .map(|(id, challenge)| (id.clone(), challenge.expires_at.unix))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(max_pending);
        let keep: HashSet<String> = entries.into_iter().map(|(id, _)| id).collect();
        self.challenges.retain(|id, _| keep.contains(id));
        self.thread_index.retain(|_, challenge_id| keep.contains(challenge_id));
    }

    fn prune_patterns(&mut self) {
        let max_cached = self.config.max_cached_patterns.max(1);
        if self.patterns.len() <= max_cached {
            return;
        }
        let mut entries: Vec<_> = self
            .patterns
            .iter()
            .map(|(id, pattern)| (id.clone(), pattern.last_seen.unix))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(max_cached);
        let keep: HashSet<String> = entries.into_iter().map(|(id, _)| id).collect();
        self.patterns.retain(|id, _| keep.contains(id));
    }
}

pub struct IdentityVerificationOutcome {
    pub binding: IdentityBinding,
    pub evidence_hash: String,
    pub updated_pattern: Option<NetworkPatternSummary>,
}

fn should_replace_pattern(
    existing: &NetworkPatternSummary,
    incoming: &NetworkPatternSummary,
) -> bool {
    if incoming.last_seen.unix > existing.last_seen.unix {
        return true;
    }
    if incoming.last_seen.unix < existing.last_seen.unix {
        return false;
    }
    let incoming_weight = if incoming.peer_weight.is_finite() && incoming.peer_weight > 0.0 {
        incoming.peer_weight
    } else {
        1.0
    };
    let existing_weight = if existing.peer_weight.is_finite() && existing.peer_weight > 0.0 {
        existing.peer_weight
    } else {
        1.0
    };
    let incoming_score =
        incoming.confidence * incoming_weight + (incoming.support as f64).ln_1p();
    let existing_score =
        existing.confidence * existing_weight + (existing.support as f64).ln_1p();
    incoming_score > existing_score
}

fn generate_code(length: usize) -> String {
    let mut out = String::with_capacity(length.max(4));
    let mut rng = OsRng;
    for _ in 0..length.max(4) {
        let idx = rng.gen_range(0..CODE_CHARS.len());
        out.push(CODE_CHARS[idx] as char);
    }
    out
}

fn build_instruction(
    code: &str,
    position: Option<[f64; 3]>,
    tolerance: f64,
    signature: &[f64],
) -> String {
    let mut parts = Vec::new();
    if let Some(pos) = position {
        parts.push(format!(
            "Stand near ({:.3}, {:.3}, {:.3}) within {:.2} units",
            pos[0], pos[1], pos[2], tolerance
        ));
    }
    parts.push(format!(
        "Display the code '{}' clearly to the camera",
        code
    ));
    if !signature.is_empty() {
        parts.push("Hold a motion pattern consistent with your recent behavior".to_string());
    }
    if parts.is_empty() {
        "Present the code to the camera".to_string()
    } else {
        parts.join("; ")
    }
}

fn behavior_similarity(expected: &[f64], observed: &[f64]) -> f64 {
    if expected.is_empty() || observed.is_empty() || expected.len() != observed.len() {
        return 0.0;
    }
    math::cosine_similarity(expected, observed)
        .map(|score| ((score + 1.0) * 0.5).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

fn spatial_score(
    expected: Option<[f64; 3]>,
    observed: Option<[f64; 3]>,
    tolerance: f64,
) -> f64 {
    let (Some(expected), Some(observed)) = (expected, observed) else {
        return 1.0;
    };
    let dx = expected[0] - observed[0];
    let dy = expected[1] - observed[1];
    let dz = expected[2] - observed[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if distance <= tolerance {
        1.0
    } else if tolerance > 0.0 {
        (tolerance / distance).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn match_score(behavior_similarity: f64, spatial_score: f64) -> f64 {
    (0.6 * behavior_similarity + 0.4 * spatial_score).clamp(0.0, 1.0)
}

fn identity_evidence_hash(evidence: &IdentityEvidence) -> String {
    let payload = format!(
        "evidence|{}|{}|{}|{}|{}|{}|{}",
        evidence.challenge_id,
        evidence.thread_id,
        evidence.observed_at.unix,
        evidence.code,
        position_key(evidence.position),
        signature_key(&evidence.observed_signature),
        match_score_key(evidence.match_score)
    );
    compute_payload_hash(payload.as_bytes())
}

fn position_key(position: Option<[f64; 3]>) -> String {
    match position {
        Some(pos) => format!("{:.4},{:.4},{:.4}", pos[0], pos[1], pos[2]),
        None => "none".to_string(),
    }
}

fn signature_key(signature: &[f64]) -> String {
    if signature.is_empty() {
        return "none".to_string();
    }
    let mut out = String::new();
    for (idx, value) in signature.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:.4}", value));
    }
    out
}

fn match_score_key(score: Option<f64>) -> String {
    score
        .map(|val| format!("{:.4}", val))
        .unwrap_or_else(|| "none".to_string())
}

fn verify_wallet_binding(binding: &IdentityBinding) -> Result<()> {
    let derived = address_from_public_key_hex(&binding.wallet_public_key)?;
    if derived != binding.wallet_address {
        anyhow::bail!("wallet address does not match public key");
    }
    let payload = identity_binding_payload(binding);
    let public_key = decode_public_key(&binding.wallet_public_key)?;
    let signature = decode_signature(&binding.signature)?;
    public_key
        .verify(payload.as_bytes(), &signature)
        .map_err(|err| anyhow!("wallet signature invalid: {err}"))?;
    Ok(())
}

fn address_from_public_key_hex(public_key_hex: &str) -> Result<String> {
    let bytes = decode_public_key_bytes(public_key_hex)?;
    Ok(address_from_public_key(&bytes))
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
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
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
