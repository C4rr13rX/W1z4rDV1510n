use crate::energy::EnergyModel;
use crate::ml::MlHooksHandle;
use crate::schema::{
    DynamicState, EnvironmentSnapshot, ParticleState, Population, Position, SymbolState, Timestamp,
};
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitStrategyConfig {
    pub use_ml_priors: bool,
    pub noise_level: f64,
    pub random_fraction: f64,
    pub copy_snapshot_fraction: f64,
}

impl Default for InitStrategyConfig {
    fn default() -> Self {
        Self {
            use_ml_priors: true,
            noise_level: 0.5,
            random_fraction: 0.2,
            copy_snapshot_fraction: 0.1,
        }
    }
}

pub fn init_population(
    snapshot_0: &EnvironmentSnapshot,
    t_end: Timestamp,
    ml_hooks: Option<&MlHooksHandle>,
    energy_model: &EnergyModel,
    n_particles: usize,
    init_strategy: &InitStrategyConfig,
    rng: &mut StdRng,
) -> Population {
    let predictions = ml_hooks.map(|hooks| hooks.predict_next_positions(snapshot_0, &t_end));
    let mut particles = Vec::with_capacity(n_particles);
    for idx in 0..n_particles {
        let mut symbol_states = HashMap::new();
        for symbol in &snapshot_0.symbols {
            let mut position = if init_strategy.use_ml_priors {
                predictions
                    .as_ref()
                    .and_then(|preds| preds.get(&symbol.id))
                    .copied()
                    .unwrap_or(symbol.position)
            } else {
                symbol.position
            };
            if rng.gen_bool(init_strategy.random_fraction) {
                position = Position {
                    x: rng.gen_range(0.0..snapshot_0.bounds.get("width").copied().unwrap_or(1.0)),
                    y: rng.gen_range(0.0..snapshot_0.bounds.get("height").copied().unwrap_or(1.0)),
                    z: rng.gen_range(0.0..snapshot_0.bounds.get("depth").copied().unwrap_or(1.0)),
                };
            } else if !rng.gen_bool(init_strategy.copy_snapshot_fraction) {
                let noise = init_strategy.noise_level;
                position.x += rng.gen_range(-noise..noise);
                position.y += rng.gen_range(-noise..noise);
                position.z += rng.gen_range(-noise..noise);
            }
            symbol_states.insert(
                symbol.id.clone(),
                SymbolState {
                    position,
                    velocity: None,
                    internal_state: Default::default(),
                },
            );
        }
        let dynamic_state = DynamicState {
            timestamp: t_end,
            symbol_states,
        };
        let energy = energy_model.energy(&dynamic_state);
        particles.push(ParticleState {
            id: idx,
            current_state: dynamic_state,
            energy,
            weight: 1.0 / n_particles as f64,
            history: None,
            metadata: Default::default(),
        });
    }
    Population {
        particles,
        temperature: 0.0,
        iteration: 0,
    }
}

pub fn normalize_weights(population: &mut Population) {
    let sum: f64 = population.particles.iter().map(|p| p.weight).sum();
    if sum <= f64::EPSILON {
        let uniform = 1.0 / population.particles.len().max(1) as f64;
        for particle in &mut population.particles {
            particle.weight = uniform;
        }
    } else {
        for particle in &mut population.particles {
            particle.weight /= sum;
        }
    }
}

pub fn resample_population(population: &mut Population, rng: &mut StdRng) {
    let weights: Vec<f64> = population.particles.iter().map(|p| p.weight).collect();
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for w in weights {
        acc += w;
        cumulative.push(acc);
    }
    let mut new_particles = Vec::with_capacity(population.particles.len());
    for id in 0..population.particles.len() {
        let sample = rng.r#gen::<f64>();
        let idx = cumulative
            .iter()
            .position(|value| *value >= sample)
            .unwrap_or(cumulative.len() - 1);
        let mut cloned = population.particles[idx].clone();
        cloned.id = id;
        new_particles.push(cloned);
    }
    population.particles = new_particles;
    normalize_weights(population);
}

pub fn clone_and_mutate(population: &mut Population, rng: &mut StdRng, mutation_rate: f64) {
    for particle in &mut population.particles {
        if rng.gen_bool(mutation_rate) {
            for symbol_state in particle.current_state.symbol_states.values_mut() {
                symbol_state.position.x += rng.gen_range(-0.1..0.1);
                symbol_state.position.y += rng.gen_range(-0.1..0.1);
            }
        }
    }
}
