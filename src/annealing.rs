use crate::config::{AnnealingScheduleConfig, ResampleConfig, ScheduleType};
use crate::energy::EnergyModel;
use crate::hardware::HardwareBackendHandle;
use crate::proposal::ProposalKernel;
use crate::schema::{EnvironmentSnapshot, Population};
use crate::search::SearchModule;
use crate::state_population::{clone_and_mutate, normalize_weights, resample_population};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    rng: &mut StdRng,
) -> (Population, Vec<f64>) {
    let mut energy_trace = Vec::with_capacity(schedule.n_iterations);
    let log_interval = (schedule.n_iterations / 10).max(1);
    for iteration in 0..schedule.n_iterations {
        let temperature = temperature(iteration, schedule);
        population.temperature = temperature;
        population.iteration = iteration;
        let particle_seeds: Vec<u64> = (0..population.particles.len())
            .map(|_| rng.r#gen())
            .collect();
        let update_particle = |particle: &mut crate::schema::ParticleState| {
            let mut local_rng = StdRng::seed_from_u64(particle_seeds[particle.id]);
            let mut proposal = kernel.propose(snapshot_0, &particle.current_state, temperature);
            if let Some(search) = search_module {
                search.enforce_hard_constraints(snapshot_0, &mut proposal);
            }
            let new_energy = energy_model.energy(&proposal);
            let accept_prob = acceptance_probability(particle.energy, new_energy, temperature);
            if local_rng.r#gen::<f64>() < accept_prob {
                particle.current_state = proposal;
                particle.energy = new_energy;
            }
            particle.weight = (-particle.energy / temperature.max(1e-3)).exp();
        };
        hardware_backend.map_particles(&mut population, &update_particle);
        normalize_weights(&mut population);

        let ess_ratio = effective_sample_size(&population);
        if resample_config.enabled && ess_ratio < resample_config.effective_sample_size_threshold {
            resample_population(&mut population, rng);
            clone_and_mutate(&mut population, rng, resample_config.mutation_rate);
            debug!(
                target: "w1z4rdv1510n::annealing",
                iteration,
                ess_ratio,
                "resampled population due to ESS drop"
            );
        }

        let min_energy = population
            .particles
            .iter()
            .map(|p| p.energy)
            .fold(f64::INFINITY, f64::min);
        energy_trace.push(min_energy);
        if iteration % log_interval == 0 || iteration + 1 == schedule.n_iterations {
            info!(
                target: "w1z4rdv1510n::annealing",
                iteration,
                temperature,
                min_energy,
                ess_ratio,
                "annealing iteration summary"
            );
        }
    }
    (population, energy_trace)
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
