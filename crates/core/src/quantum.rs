use crate::config::{AnnealingScheduleConfig, HomeostasisConfig, QuantumConfig, ResampleConfig};
use crate::energy::EnergyModel;
use crate::hardware::HardwareBackendHandle;
use crate::neuro::NeuroRuntimeHandle;
use crate::proposal::ProposalKernel;
use crate::schema::{DynamicState, EnvironmentSnapshot, Population, Position};
use crate::search::SearchModule;
use crate::state_population::{clone_and_mutate, normalize_weights, resample_population};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

#[derive(Debug)]
pub struct QuantumAnnealOutcome {
    pub slices: Vec<Population>,
    pub energy_trace: Vec<f64>,
    pub acceptance_trace: Vec<f64>,
}

pub fn anneal_quantum(
    mut slices: Vec<Population>,
    snapshot_0: &EnvironmentSnapshot,
    energy_model: &EnergyModel,
    kernel: &dyn ProposalKernel,
    search_module: Option<&SearchModule>,
    schedule: &AnnealingScheduleConfig,
    resample_config: &ResampleConfig,
    hardware_backend: &HardwareBackendHandle,
    neuro: Option<NeuroRuntimeHandle>,
    homeostasis: &HomeostasisConfig,
    rng: &mut StdRng,
    quantum: &QuantumConfig,
) -> QuantumAnnealOutcome {
    let trotter = slices.len().max(1);
    let tensor_executor = hardware_backend.tensor_executor();
    let log_interval = (schedule.n_iterations / 10).max(1);
    let mut homeo_state = HomeostasisState::new(homeostasis.clone());

    let mut energy_trace = Vec::with_capacity(schedule.n_iterations);
    let mut acceptance_trace = Vec::with_capacity(schedule.n_iterations);

    for iteration in 0..schedule.n_iterations {
        let base_temp = temperature(iteration, schedule);
        let slice_temp_base =
            base_temp * quantum.slice_temperature_scale / (trotter as f64).max(1.0);
        let (slice_temp, boosted) = homeo_state.effective_temperature(slice_temp_base);
        let progress = iteration as f64 / schedule.n_iterations.max(1) as f64;
        let driver_scale = quantum.driver_strength
            + (quantum.driver_final_strength - quantum.driver_strength) * progress;

        let mut accepted_total = 0usize;
        let mut total_particles = 0usize;
        for slice_idx in 0..trotter {
            let prev_idx = if slice_idx == 0 {
                trotter - 1
            } else {
                slice_idx - 1
            };
            let next_idx = (slice_idx + 1) % trotter;
            let prev_lookup = neighbor_lookup(&slices[prev_idx]);
            let next_lookup = neighbor_lookup(&slices[next_idx]);
            let particle_seeds: Vec<u64> = (0..slices[slice_idx].particles.len())
                .map(|_| rng.r#gen())
                .collect();
            let accepted_counter = Arc::new(Mutex::new(0usize));
            let update_particle = {
                let accepted_counter = Arc::clone(&accepted_counter);
                let prev_lookup = prev_lookup.clone();
                let next_lookup = next_lookup.clone();
                move |particle: &mut crate::schema::ParticleState| {
                    let mut local_rng = StdRng::seed_from_u64(particle_seeds[particle.id]);
                    let prev_state = prev_lookup.get(&particle.id);
                    let next_state = next_lookup.get(&particle.id);
                    let current_driver =
                        driver_penalty(&particle.current_state, prev_state, next_state);
                    let mut proposal =
                        kernel.propose(snapshot_0, &particle.current_state, slice_temp);
                    if let Some(search) = search_module {
                        search.enforce_hard_constraints(
                            snapshot_0,
                            &mut proposal,
                            Some(hardware_backend.as_ref()),
                        );
                    }
                    let proposal_driver = driver_penalty(&proposal, prev_state, next_state);
                    let classical_energy = energy_model.energy(&proposal);
                    let total_new =
                        composite_energy(classical_energy, proposal_driver, driver_scale);
                    let total_old = composite_energy(particle.energy, current_driver, driver_scale);
                    if local_rng.r#gen::<f64>()
                        < acceptance_probability(total_old, total_new, slice_temp)
                    {
                        particle.current_state = proposal;
                        particle.energy = classical_energy;
                        let mut guard = accepted_counter.lock();
                        *guard += 1;
                    }
                    let current_driver =
                        driver_penalty(&particle.current_state, prev_state, next_state);
                    let weighted_energy =
                        composite_energy(particle.energy, current_driver, driver_scale);
                    particle.weight = (-weighted_energy / slice_temp.max(1e-3)).exp();
                }
            };
            hardware_backend.map_particles(&mut slices[slice_idx], &update_particle);
            normalize_weights(&mut slices[slice_idx], tensor_executor.as_deref());
            let ess_ratio = effective_sample_size(&slices[slice_idx]);
            if resample_config.enabled
                && ess_ratio < resample_config.effective_sample_size_threshold
            {
                let mut mutation_rate = resample_config.mutation_rate;
                if boosted {
                    mutation_rate *= 1.0 + homeostasis.mutation_boost;
                }
                resample_population(&mut slices[slice_idx], rng);
                clone_and_mutate(&mut slices[slice_idx], rng, mutation_rate);
                normalize_weights(&mut slices[slice_idx], tensor_executor.as_deref());
                debug!(
                    target: "w1z4rdv1510n::annealing::quantum",
                    slice = slice_idx,
                    iteration,
                    ess_ratio,
                    "resampled slice due to ESS drop"
                );
            }
            if let Some(neuro_runtime) = neuro.as_ref() {
                neuro_runtime
                    .observe_states(slices[slice_idx].particles.iter().map(|p| &p.current_state));
            }
            accepted_total += *accepted_counter.lock();
            total_particles += slices[slice_idx].particles.len();
        }

        if quantum.worldline_mix_prob > 0.0 && slices.len() > 1 {
            apply_worldline_mix(
                &mut slices,
                snapshot_0,
                search_module,
                quantum.worldline_mix_prob,
                slice_temp,
                driver_scale,
                energy_model,
                tensor_executor.as_deref(),
                rng,
            );
        }

        let neighbor_maps: Vec<_> = slices.iter().map(neighbor_lookup).collect();
        let mut min_energy = f64::INFINITY;
        for (slice_idx, slice) in slices.iter().enumerate() {
            let prev_idx = if slice_idx == 0 {
                neighbor_maps.len() - 1
            } else {
                slice_idx - 1
            };
            let next_idx = (slice_idx + 1) % neighbor_maps.len();
            for particle in &slice.particles {
                let driver = driver_penalty(
                    &particle.current_state,
                    neighbor_maps[prev_idx].get(&particle.id),
                    neighbor_maps[next_idx].get(&particle.id),
                );
                let total = composite_energy(particle.energy, driver, driver_scale);
                if total < min_energy {
                    min_energy = total;
                }
            }
        }

        let acceptance_ratio = if total_particles == 0 {
            0.0
        } else {
            accepted_total as f64 / total_particles as f64
        };
        energy_trace.push(min_energy);
        acceptance_trace.push(acceptance_ratio);
        homeo_state.observe(min_energy, acceptance_ratio, iteration);
        if iteration % log_interval == 0 || iteration + 1 == schedule.n_iterations {
            info!(
                target: "w1z4rdv1510n::annealing::quantum",
                iteration,
                base_temp,
                slice_temp,
                driver_scale,
                min_energy,
                acceptance_ratio,
                plateau_iters = homeo_state.stagnant_iters,
                homeostasis_boost = boosted,
                "quantum annealing iteration summary"
            );
        }
    }

    QuantumAnnealOutcome {
        slices,
        energy_trace,
        acceptance_trace,
    }
}

fn temperature(iteration: usize, schedule: &AnnealingScheduleConfig) -> f64 {
    match schedule.schedule_type {
        crate::config::ScheduleType::Linear => {
            let progress = iteration as f64 / schedule.n_iterations.max(1) as f64;
            schedule.t_start + (schedule.t_end - schedule.t_start) * progress
        }
        crate::config::ScheduleType::Exponential => {
            let ratio = schedule.t_end / schedule.t_start.max(1e-3);
            schedule.t_start * ratio.powf(iteration as f64 / schedule.n_iterations.max(1) as f64)
        }
        crate::config::ScheduleType::Custom => schedule.t_end,
    }
}

fn acceptance_probability(old: f64, new: f64, temperature: f64) -> f64 {
    if new <= old {
        1.0
    } else {
        (-((new - old) / temperature.max(1e-3))).exp()
    }
}

fn effective_sample_size(population: &Population) -> f64 {
    let sum_sq: f64 = population.particles.iter().map(|p| p.weight.powi(2)).sum();
    if sum_sq <= f64::EPSILON {
        1.0
    } else {
        1.0 / sum_sq / population.particles.len().max(1) as f64
    }
}

#[derive(Debug, Clone)]
struct HomeostasisState {
    best_energy: f64,
    stagnant_iters: usize,
    config: HomeostasisConfig,
}

impl HomeostasisState {
    fn new(config: HomeostasisConfig) -> Self {
        Self {
            best_energy: f64::INFINITY,
            stagnant_iters: 0,
            config,
        }
    }

    fn effective_temperature(&self, base: f64) -> (f64, bool) {
        if !self.config.enabled || self.stagnant_iters < self.config.patience {
            (base, false)
        } else {
            (
                base * (1.0 + self.config.reheat_scale),
                self.config.reheat_scale > 0.0,
            )
        }
    }

    fn observe(&mut self, min_energy: f64, _acceptance: f64, iteration: usize) {
        if !self.config.enabled {
            return;
        }
        if min_energy + self.config.energy_plateau_tolerance < self.best_energy {
            self.best_energy = min_energy;
            self.stagnant_iters = 0;
        } else {
            self.stagnant_iters += 1;
        }
        if self.stagnant_iters == self.config.patience {
            tracing::info!(
                target: "w1z4rdv1510n::homeostasis",
                iteration,
                best_energy = self.best_energy,
                "homeostasis plateau detected; applying mutation/temperature boost (quantum)"
            );
        }
    }
}

fn driver_penalty(
    state: &DynamicState,
    prev: Option<&DynamicState>,
    next: Option<&DynamicState>,
) -> f64 {
    let mut accum = 0.0;
    let mut count = 0.0;
    if let Some(prev_state) = prev {
        accum += state_distance(state, prev_state);
        count += 1.0;
    }
    if let Some(next_state) = next {
        accum += state_distance(state, next_state);
        count += 1.0;
    }
    if count > 0.0 { accum / count } else { 0.0 }
}

fn state_distance(a: &DynamicState, b: &DynamicState) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for (symbol_id, state_a) in a.symbol_states.iter() {
        if let Some(state_b) = b.symbol_states.get(symbol_id) {
            let dx = state_a.position.x - state_b.position.x;
            let dy = state_a.position.y - state_b.position.y;
            let dz = state_a.position.z - state_b.position.z;
            total += (dx * dx + dy * dy + dz * dz).sqrt();
            count += 1.0;
        }
    }
    if count > 0.0 { total / count } else { 0.0 }
}

fn neighbor_lookup(population: &Population) -> HashMap<usize, DynamicState> {
    population
        .particles
        .iter()
        .map(|p| (p.id, p.current_state.clone()))
        .collect()
}

fn composite_energy(classical: f64, driver: f64, driver_scale: f64) -> f64 {
    classical + driver * driver_scale
}

#[allow(clippy::too_many_arguments)]
fn apply_worldline_mix(
    slices: &mut [Population],
    snapshot_0: &EnvironmentSnapshot,
    search_module: Option<&SearchModule>,
    mix_prob: f64,
    slice_temp: f64,
    driver_scale: f64,
    energy_model: &EnergyModel,
    tensor: Option<&dyn crate::tensor::TensorExecutor>,
    rng: &mut StdRng,
) {
    if mix_prob <= 0.0 || slices.is_empty() {
        return;
    }
    let n_particles = slices[0].particles.len();
    for slice in slices.iter() {
        if slice.particles.len() != n_particles {
            return;
        }
    }

    for particle_idx in 0..n_particles {
        if !rng.gen_bool(mix_prob) {
            continue;
        }
        let mut accum: HashMap<String, Position> = HashMap::new();
        let mut counts: HashMap<String, f64> = HashMap::new();
        for slice in slices.iter() {
            let particle = &slice.particles[particle_idx];
            for (symbol_id, symbol_state) in particle.current_state.symbol_states.iter() {
                let entry = accum.entry(symbol_id.clone()).or_insert(Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                });
                entry.x += symbol_state.position.x;
                entry.y += symbol_state.position.y;
                entry.z += symbol_state.position.z;
                *counts.entry(symbol_id.clone()).or_insert(0.0) += 1.0;
            }
        }
        let mut mean: HashMap<String, Position> = HashMap::new();
        for (symbol_id, sum) in accum {
            let count = counts.get(&symbol_id).copied().unwrap_or(1.0).max(1e-3);
            mean.insert(
                symbol_id,
                Position {
                    x: sum.x / count,
                    y: sum.y / count,
                    z: sum.z / count,
                },
            );
        }
        for slice in slices.iter_mut() {
            if let Some(particle) = slice.particles.get_mut(particle_idx) {
                for (symbol_id, target) in mean.iter() {
                    if let Some(state) = particle.current_state.symbol_states.get_mut(symbol_id) {
                        state.position = *target;
                    }
                }
                if let Some(search) = search_module {
                    let mut state = particle.current_state.clone();
                    search.enforce_hard_constraints(snapshot_0, &mut state, None);
                    particle.current_state = state;
                }
            }
        }
    }

    let neighbor_maps: Vec<_> = slices.iter().map(neighbor_lookup).collect();
    for (slice_idx, slice) in slices.iter_mut().enumerate() {
        let prev_idx = if slice_idx == 0 {
            neighbor_maps.len() - 1
        } else {
            slice_idx - 1
        };
        let next_idx = (slice_idx + 1) % neighbor_maps.len();
        for particle in slice.particles.iter_mut() {
            let prev_state = neighbor_maps[prev_idx].get(&particle.id);
            let next_state = neighbor_maps[next_idx].get(&particle.id);
            let driver = driver_penalty(&particle.current_state, prev_state, next_state);
            let classical = energy_model.energy(&particle.current_state);
            particle.energy = classical;
            let weighted = composite_energy(classical, driver, driver_scale);
            particle.weight = (-weighted / slice_temp.max(1e-3)).exp();
        }
        normalize_weights(slice, tensor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EnergyConfig;
    use crate::ml::NullMLModel;
    use crate::schema::{DynamicState, Properties, Symbol, SymbolState, SymbolType, Timestamp};

    fn simple_snapshot() -> EnvironmentSnapshot {
        EnvironmentSnapshot {
            timestamp: Timestamp { unix: 0 },
            bounds: HashMap::from([("width".into(), 4.0), ("height".into(), 4.0)]),
            symbols: vec![Symbol {
                id: "p1".into(),
                symbol_type: SymbolType::Person,
                position: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                properties: Properties::new(),
            }],
            metadata: Properties::new(),
            stack_history: Vec::new(),
        }
    }

    fn single_state(x: f64) -> DynamicState {
        let mut state = DynamicState {
            timestamp: Timestamp { unix: 0 },
            symbol_states: HashMap::new(),
        };
        state.symbol_states.insert(
            "p1".into(),
            SymbolState {
                position: Position { x, y: 0.0, z: 0.0 },
                ..Default::default()
            },
        );
        state
    }

    #[test]
    fn driver_penalty_tracks_neighbor_distance() {
        let a = single_state(0.0);
        let b = single_state(2.0);
        let c = single_state(4.0);
        let penalty = super::driver_penalty(&b, Some(&a), Some(&c));
        assert!(
            penalty > 1.9 && penalty < 2.1,
            "expected average distance near 2.0, got {penalty}"
        );
    }

    #[test]
    fn composite_energy_scales_driver_term() {
        let base = 5.0;
        let driver = 2.0;
        let scaled = super::composite_energy(base, driver, 0.5);
        assert!(
            (scaled - 6.0).abs() < 1e-6,
            "composite energy should include scaled driver"
        );
    }

    #[test]
    fn worldline_mix_updates_positions() {
        let snapshot = simple_snapshot();
        let energy_model = EnergyModel::new(
            Arc::new(snapshot.clone()),
            EnergyConfig::default(),
            Some(Arc::new(NullMLModel)),
            None,
            Timestamp { unix: 0 },
            None,
            None,
        );
        let mut slices = vec![
            Population {
                particles: vec![crate::schema::ParticleState {
                    id: 0,
                    current_state: single_state(0.0),
                    energy: 0.0,
                    weight: 1.0,
                    history: None,
                    metadata: Properties::new(),
                }],
                temperature: 1.0,
                iteration: 0,
            },
            Population {
                particles: vec![crate::schema::ParticleState {
                    id: 0,
                    current_state: single_state(4.0),
                    energy: 0.0,
                    weight: 1.0,
                    history: None,
                    metadata: Properties::new(),
                }],
                temperature: 1.0,
                iteration: 0,
            },
        ];
        let mut rng = StdRng::seed_from_u64(7);
        apply_worldline_mix(
            &mut slices,
            &snapshot,
            None,
            1.0,
            1.0,
            0.5,
            &energy_model,
            None,
            &mut rng,
        );
        let pos0 = slices[0].particles[0].current_state.symbol_states["p1"]
            .position
            .x;
        let pos1 = slices[1].particles[0].current_state.symbol_states["p1"]
            .position
            .x;
        assert!(
            (pos0 - 2.0).abs() < 1e-6 && (pos1 - 2.0).abs() < 1e-6,
            "worldline mix should align slices to mean position"
        );
    }
}
