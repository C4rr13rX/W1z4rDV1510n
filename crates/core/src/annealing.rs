use crate::config::{AnnealingScheduleConfig, HomeostasisConfig, ResampleConfig, ScheduleType};
use crate::energy::EnergyModel;
use crate::hardware::HardwareBackendHandle;
use crate::neuro::NeuroRuntimeHandle;
use crate::proposal::ProposalKernel;
use crate::schema::{EnvironmentSnapshot, Population};
use crate::search::SearchModule;
use crate::state_population::{clone_and_mutate, normalize_weights, resample_population};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use tracing::{debug, info};

#[tracing::instrument(
    skip_all,
    fields(n_particles = population.particles.len(), iterations = schedule.n_iterations)
)]
pub fn anneal(
    mut population: Population,
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
) -> (Population, Vec<f64>, Vec<f64>) {
    let mut energy_trace = Vec::with_capacity(schedule.n_iterations);
    let mut acceptance_trace = Vec::with_capacity(schedule.n_iterations);
    let log_interval = (schedule.n_iterations / 10).max(1);
    let tensor_executor = hardware_backend.tensor_executor();
    let mut homeo_state = HomeostasisState::new(homeostasis.clone());
    if let Some(planner) = hardware_backend.planner() {
        debug!(
            target: "w1z4rdv1510n::annealing",
            threads = planner.threads_for(population.particles.len()),
            "backend planner engaged for annealing"
        );
    }
    if let Some(neuro_runtime) = neuro.as_ref() {
        neuro_runtime.observe_states(population.particles.iter().map(|p| &p.current_state));
    }
    for iteration in 0..schedule.n_iterations {
        let base_temperature = temperature(iteration, schedule);
        let (temperature, boosted) = homeo_state.effective_temperature(base_temperature);
        population.temperature = temperature;
        population.iteration = iteration;
        let particle_seeds: Vec<u64> = (0..population.particles.len())
            .map(|_| rng.r#gen())
            .collect();
        let accepted_counter = Arc::new(Mutex::new(0usize));
        let update_particle = {
            let accepted_counter = Arc::clone(&accepted_counter);
            move |particle: &mut crate::schema::ParticleState| {
                let mut local_rng = StdRng::seed_from_u64(particle_seeds[particle.id]);
                let mut proposal = kernel.propose(snapshot_0, &particle.current_state, temperature);
                if let Some(search) = search_module {
                    search.enforce_hard_constraints(
                        snapshot_0,
                        &mut proposal,
                        Some(hardware_backend.as_ref()),
                    );
                }
                let new_energy = energy_model.energy(&proposal);
                let accept_prob = acceptance_probability(particle.energy, new_energy, temperature);
                if local_rng.r#gen::<f64>() < accept_prob {
                    particle.current_state = proposal;
                    particle.energy = new_energy;
                    let mut guard = accepted_counter.lock();
                    *guard += 1;
                }
                particle.weight = (-particle.energy / temperature.max(1e-3)).exp();
            }
        };
        hardware_backend.map_particles(&mut population, &update_particle);
        normalize_weights(&mut population, tensor_executor.as_deref());

        let ess_ratio = effective_sample_size(&population);
        if resample_config.enabled && ess_ratio < resample_config.effective_sample_size_threshold {
            let mut mutation_rate = resample_config.mutation_rate;
            if boosted {
                mutation_rate *= 1.0 + homeostasis.mutation_boost;
            }
            resample_population(&mut population, rng);
            clone_and_mutate(&mut population, rng, mutation_rate);
            normalize_weights(&mut population, tensor_executor.as_deref());
            debug!(
                target: "w1z4rdv1510n::annealing",
                iteration,
                ess_ratio,
                "resampled population due to ESS drop"
            );
        }

        if let Some(neuro_runtime) = neuro.as_ref() {
            neuro_runtime.observe_states(population.particles.iter().map(|p| &p.current_state));
        }

        let min_energy = population
            .particles
            .iter()
            .map(|p| p.energy)
            .fold(f64::INFINITY, f64::min);
        energy_trace.push(min_energy);
        let accepted_count = *accepted_counter.lock();
        let acceptance_ratio = if population.particles.is_empty() {
            0.0
        } else {
            accepted_count as f64 / population.particles.len() as f64
        };
        acceptance_trace.push(acceptance_ratio);
        homeo_state.observe(min_energy, acceptance_ratio, iteration);
        if iteration % log_interval == 0 || iteration + 1 == schedule.n_iterations {
            info!(
                target: "w1z4rdv1510n::annealing",
                iteration,
                temperature,
                min_energy,
                ess_ratio,
                acceptance_ratio,
                plateau_iters = homeo_state.stagnant_iters,
                homeostasis_boost = boosted,
                "annealing iteration summary"
            );
        }
    }
    (population, energy_trace, acceptance_trace)
}

fn temperature(iteration: usize, schedule: &AnnealingScheduleConfig) -> f64 {
    match schedule.schedule_type {
        ScheduleType::Linear => {
            let progress = iteration as f64 / schedule.n_iterations.max(1) as f64;
            schedule.t_start + (schedule.t_end - schedule.t_start) * progress
        }
        ScheduleType::Exponential => {
            let ratio = schedule.t_end / schedule.t_start.max(1e-3);
            schedule.t_start * ratio.powf(iteration as f64 / schedule.n_iterations.max(1) as f64)
        }
        ScheduleType::Custom => schedule.t_end,
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
                "homeostasis plateau detected; applying mutation/temperature boost"
            );
        }
    }
}
