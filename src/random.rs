use blake2::{Blake2s256, Digest};
use parking_lot::Mutex;
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type RandomProviderHandle = Arc<dyn RandomProvider>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RandomProviderType {
    Deterministic,
    OsEntropy,
    JitterExperimental,
}

impl Default for RandomProviderType {
    fn default() -> Self {
        Self::Deterministic
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomConfig {
    #[serde(default)]
    pub provider: RandomProviderType,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for RandomConfig {
    fn default() -> Self {
        Self {
            provider: RandomProviderType::Deterministic,
            seed: Some(7),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomProviderDescriptor {
    pub provider: RandomProviderType,
    pub deterministic: bool,
    pub seed: Option<u64>,
}

pub trait RandomProvider: Send + Sync {
    fn next_seed(&self, label: &str) -> u64;
    fn descriptor(&self) -> RandomProviderDescriptor;
}

pub fn create_random_provider(
    config: &RandomConfig,
    legacy_seed: u64,
) -> anyhow::Result<RandomProviderHandle> {
    let provider = match config.provider {
        RandomProviderType::Deterministic => {
            let seed = config.seed.unwrap_or(legacy_seed);
            Arc::new(DeterministicRandomProvider::new(seed)) as RandomProviderHandle
        }
        RandomProviderType::OsEntropy => Arc::new(OsEntropyRandomProvider::new()?),
        #[cfg(feature = "experimental-hw")]
        RandomProviderType::JitterExperimental => Arc::new(JitterRandomProvider::new()),
        #[cfg(not(feature = "experimental-hw"))]
        RandomProviderType::JitterExperimental => {
            anyhow::bail!("jitter experimental RNG requires the `experimental-hw` feature")
        }
    };
    Ok(provider)
}

fn hash_seed(base: u64, label: &str) -> u64 {
    let mut hasher = Blake2s256::new();
    hasher.update(base.to_le_bytes());
    hasher.update(label.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[0..8]);
    u64::from_le_bytes(bytes)
}

struct DeterministicRandomProvider {
    base_seed: u64,
}

impl DeterministicRandomProvider {
    fn new(base_seed: u64) -> Self {
        Self { base_seed }
    }
}

impl RandomProvider for DeterministicRandomProvider {
    fn next_seed(&self, label: &str) -> u64 {
        hash_seed(self.base_seed, label)
    }

    fn descriptor(&self) -> RandomProviderDescriptor {
        RandomProviderDescriptor {
            provider: RandomProviderType::Deterministic,
            deterministic: true,
            seed: Some(self.base_seed),
        }
    }
}

struct OsEntropyRandomProvider {
    rng: Mutex<OsRng>,
}

impl OsEntropyRandomProvider {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            rng: Mutex::new(OsRng),
        })
    }
}

impl RandomProvider for OsEntropyRandomProvider {
    fn next_seed(&self, _label: &str) -> u64 {
        self.rng.lock().next_u64()
    }

    fn descriptor(&self) -> RandomProviderDescriptor {
        RandomProviderDescriptor {
            provider: RandomProviderType::OsEntropy,
            deterministic: false,
            seed: None,
        }
    }
}

#[cfg(feature = "experimental-hw")]
struct JitterRandomProvider {
    counter: Mutex<u64>,
}

#[cfg(feature = "experimental-hw")]
impl JitterRandomProvider {
    fn new() -> Self {
        Self {
            counter: Mutex::new(0),
        }
    }

    fn sample_jitter(&self, label: &str) -> u64 {
        use std::time::{Duration, Instant};
        let mut accum = 0u64;
        let mut last = Instant::now();
        for _ in 0..128 {
            let now = Instant::now();
            let delta = now.duration_since(last);
            accum ^= jitter_bits(delta);
            last = now;
        }
        let mut count_guard = self.counter.lock();
        *count_guard = count_guard.wrapping_add(1);
        accum ^ hash_seed(*count_guard, label)
    }
}

#[cfg(feature = "experimental-hw")]
fn jitter_bits(delta: Duration) -> u64 {
    let nanos = delta.as_nanos() as u64;
    let shifted = (nanos.rotate_left(13)) ^ nanos.rotate_right(7);
    shifted ^ (delta.as_micros() as u64).rotate_left(5)
}

#[cfg(feature = "experimental-hw")]
impl RandomProvider for JitterRandomProvider {
    fn next_seed(&self, label: &str) -> u64 {
        self.sample_jitter(label)
    }

    fn descriptor(&self) -> RandomProviderDescriptor {
        RandomProviderDescriptor {
            provider: RandomProviderType::JitterExperimental,
            deterministic: false,
            seed: None,
        }
    }
}
