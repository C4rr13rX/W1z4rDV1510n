use crate::schema::{DynamicState, EnvironmentSnapshot};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalConfig {
    pub local_move_prob: f64,
    pub max_step_size: f64,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            local_move_prob: 0.9,
            max_step_size: 0.75,
        }
    }
}

pub trait ProposalKernel: Send + Sync {
    fn propose(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
    ) -> DynamicState;
}

pub struct DefaultProposalKernel {
    config: ProposalConfig,
    rng: Mutex<StdRng>,
}

impl DefaultProposalKernel {
    pub fn new(config: ProposalConfig, seed: u64) -> Self {
        Self {
            config,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl ProposalKernel for DefaultProposalKernel {
    fn propose(
        &self,
        _snapshot_0: &EnvironmentSnapshot,
        current_state: &DynamicState,
        temperature: f64,
    ) -> DynamicState {
        let mut rng = self.rng.lock();
        let mut proposal = current_state.clone();
        let Some(symbol_state) = proposal.symbol_states.values_mut().choose(&mut *rng) else {
            return current_state.clone();
        };
        let step_scale = self.config.max_step_size * temperature.max(0.05);
        let mut delta = || rng.gen_range(-step_scale..step_scale);
        symbol_state.position.x += delta();
        symbol_state.position.y += delta();
        symbol_state.position.z += delta();
        proposal.timestamp = current_state.timestamp;
        proposal
    }
}
