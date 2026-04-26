use crate::config::{AnnealingScheduleConfig, HomeostasisConfig, ResampleConfig, ScheduleType};
use crate::energy::EnergyModel;
use crate::hardware::HardwareBackendHandle;
use crate::logging::{LiveFrameSink, LiveNeuroSink};
use crate::neuro::NeuroRuntimeHandle;
use crate::proposal::ProposalKernel;
use crate::schema::{EnvironmentSnapshot, Population};
use crate::search::SearchModule;
use crate::state_population::{clone_and_mutate, normalize_weights, resample_population};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AO};
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
    live_frame: Option<&LiveFrameSink>,
    live_neuro: Option<&LiveNeuroSink>,
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
    // Track last neuro coherence for fabric-modulated temperature.
    let mut fabric_coherence: f64 = 0.5; // neutral until first snapshot

    for iteration in 0..schedule.n_iterations {
        let base_temperature = temperature(iteration, schedule);
        let (mut temperature, boosted) = homeo_state.effective_temperature(base_temperature);

        // Coherence-modulated temperature: when the neural fabric has high
        // confidence about what comes next (tight motif alignment, low
        // prediction error), let the annealer converge faster by reducing
        // temperature. When the fabric is uncertain, keep temperature high
        // so the annealer explores. This replaces a blind cooling schedule
        // with the fabric's own epistemic state as the thermostat.
        //
        //   T_eff = T_schedule * (1 / (coherence + 0.1))   [coherence in 0..1]
        //
        // coherence=1 → T_eff = T/1.1  (nearly unchanged, already confident)
        // coherence=0 → T_eff = T/0.1 = 10×T (push to explore)
        // coherence=0.5 → T_eff = T/0.6 ≈ 1.7×T (mild exploration boost)
        //
        // Combined with homeostasis: if both fire the result is multiplicative,
        // which is intentional — confused + stagnant = maximum exploration.
        if schedule.n_iterations > 1 {
            let coherence_scale = 1.0 / (fabric_coherence + 0.1).min(2.0);
            temperature *= coherence_scale;
        }
        population.temperature = temperature;
        population.iteration = iteration;
        let enforce_constraints = search_module
            .map(|module| iteration % module.config.constraint_check_every.max(1) == 0)
            .unwrap_or(false);
        let particle_seeds: Vec<u64> = (0..population.particles.len())
            .map(|_| rng.r#gen())
            .collect();
        let accepted_counter = Arc::new(AtomicU64::new(0));
        let update_particle = {
            let accepted_counter = Arc::clone(&accepted_counter);
            move |particle: &mut crate::schema::ParticleState| {
                let mut local_rng = StdRng::seed_from_u64(particle_seeds[particle.id]);
                let mut proposal = kernel.propose(snapshot_0, &particle.current_state, temperature);
                if enforce_constraints {
                    if let Some(search) = search_module {
                        search.enforce_hard_constraints(
                            snapshot_0,
                            &mut proposal,
                            Some(hardware_backend.as_ref()),
                        );
                    }
                }
                let new_energy = energy_model.energy(&proposal);
                let accept_prob = acceptance_probability(particle.energy, new_energy, temperature);
                if local_rng.r#gen::<f64>() < accept_prob {
                    particle.current_state = proposal;
                    particle.energy = new_energy;
                    accepted_counter.fetch_add(1, AO::Relaxed);
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
            // Update fabric coherence for the next iteration's temperature modulation.
            // Coherence = mean prediction confidence across all tracked symbols.
            // When no predictions exist yet, stay at 0.5 (neutral).
            let snap = neuro_runtime.snapshot();
            if !snap.prediction_confidence.is_empty() {
                let sum: f64 = snap.prediction_confidence.values().sum();
                fabric_coherence = (sum / snap.prediction_confidence.len() as f64).clamp(0.0, 1.0);
            }
        }

        let min_energy = population
            .particles
            .iter()
            .map(|p| p.energy)
            .fold(f64::INFINITY, f64::min);
        if let Some(sink) = live_frame {
            if let Some(best) = population
                .particles
                .iter()
                .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(Ordering::Equal))
            {
                sink.write_frame(iteration, best.energy, &best.current_state);
            }
        }
        if let (Some(neuro_runtime), Some(neuro_sink)) = (neuro.as_ref(), live_neuro) {
            let snapshot = neuro_runtime.snapshot();
            neuro_sink.write_snapshot(iteration, &snapshot);
        }
        energy_trace.push(min_energy);
        let accepted_count = accepted_counter.load(AO::Relaxed) as usize;
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

/// Anneal a *sequence* of environment snapshots with temporal continuity.
///
/// Plain `anneal()` takes one snapshot and runs N iterations to convergence;
/// it has no concept of time.  For real sensor streams (video frames, mouse
/// trajectories, telemetry windows) we need each frame to start from the
/// prior frame's converged state, not from random — otherwise every frame
/// pays the full convergence cost and the fabric never benefits from
/// frame-to-frame continuity.
///
/// Behaviour per frame:
///   1. Particles carry over from the prior frame (warm-start).
///   2. If `mutation_between_frames > 0`, lightly perturb each carried
///      particle so the kernel still has something to explore — necessary
///      for frames where the snapshot has actually changed.
///   3. Run the standard `anneal()` for `per_frame_iterations` (typically
///      far fewer than the cold-start `schedule.n_iterations` because the
///      population is already near-optimum after the first frame).
///   4. Push the converged population into the result series and use it as
///      the starting state for the next frame.
///
/// Returns one `(Population, energy_trace, acceptance_trace)` triple per
/// frame, in input order.
#[tracing::instrument(skip_all, fields(n_frames = snapshots.len()))]
pub fn anneal_sequence(
    initial_population: Population,
    snapshots: &[EnvironmentSnapshot],
    energy_model: &EnergyModel,
    kernel: &dyn ProposalKernel,
    search_module: Option<&SearchModule>,
    schedule: &AnnealingScheduleConfig,
    resample_config: &ResampleConfig,
    hardware_backend: &HardwareBackendHandle,
    neuro: Option<NeuroRuntimeHandle>,
    homeostasis: &HomeostasisConfig,
    live_frame: Option<&LiveFrameSink>,
    live_neuro: Option<&LiveNeuroSink>,
    mutation_between_frames: f64,
    per_frame_iterations: usize,
    rng: &mut StdRng,
) -> Vec<(Population, Vec<f64>, Vec<f64>)> {
    let mut series = Vec::with_capacity(snapshots.len());
    let mut population = initial_population;

    for (frame_idx, snap) in snapshots.iter().enumerate() {
        // Warm-start: light mutation of carried particles between frames so
        // the proposal kernel has somewhere to explore on the new frame.
        // Skip on the first frame — caller-supplied initial population is
        // assumed fresh.
        if frame_idx > 0 && mutation_between_frames > 0.0 {
            clone_and_mutate(&mut population, rng, mutation_between_frames);
            normalize_weights(&mut population, hardware_backend.tensor_executor().as_deref());
        }

        // Per-frame schedule: same shape, fewer iterations.
        let frame_schedule = AnnealingScheduleConfig {
            n_iterations: per_frame_iterations.max(1),
            ..schedule.clone()
        };

        let (converged, e_trace, a_trace) = anneal(
            population,
            snap,
            energy_model,
            kernel,
            search_module,
            &frame_schedule,
            resample_config,
            hardware_backend,
            neuro.clone(),
            homeostasis,
            live_frame,
            live_neuro,
            rng,
        );
        population = converged.clone();
        series.push((converged, e_trace, a_trace));

        if let Some(neuro_runtime) = neuro.as_ref() {
            // Hand the frame's converged particles to the neuro fabric so the
            // motif runtime sees the state-sequence — this is what lets the
            // proposal kernel use temporal motif priors on the *next* frame.
            neuro_runtime.observe_states(population.particles.iter().map(|p| &p.current_state));
        }
    }

    series
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
