use crate::annealing::anneal;
use crate::config::{OutputFormat, RunConfig};
use crate::energy::EnergyModel;
use crate::hardware::{HardwareBackendType, create_hardware_backend};
use crate::logging::init_logging;
use crate::ml::create_ml_model;
use crate::neuro::NeuroRuntime;
use crate::proposal::DefaultProposalKernel;
use crate::quantum::anneal_quantum;
use crate::random::{RandomProviderDescriptor, create_random_provider};
use crate::results::{Results, analyze_results};
use crate::schema::{EnvironmentSnapshot, Population};
use crate::search::SearchModule;
use crate::state_population::init_population;
use anyhow::Context;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;
use std::sync::Arc;
use tracing::{debug, info};

pub struct RunOutcome {
    pub results: Results,
    pub random_provider: RandomProviderDescriptor,
    pub hardware_backend: HardwareBackendType,
    pub acceptance_ratio: Option<f64>,
}

pub fn run_with_config(config: RunConfig) -> anyhow::Result<Results> {
    let snapshot = load_snapshot(&config)?;
    let outcome = run_with_snapshot(snapshot, config)?;
    Ok(outcome.results)
}

pub fn run_with_snapshot(
    snapshot: EnvironmentSnapshot,
    config: RunConfig,
) -> anyhow::Result<RunOutcome> {
    init_logging(&config.logging)?;
    config.validate()?;
    let random_provider = create_random_provider(&config.random, config.random_seed)?;
    let random_descriptor = random_provider.descriptor();
    info!(
        target: "w1z4rdv1510n::random",
        provider = ?random_descriptor,
        "random provider initialized"
    );
    info!(
        target: "w1z4rdv1510n::orchestrator",
        n_particles = config.n_particles,
        iterations = config.schedule.n_iterations,
        hardware = ?config.hardware_backend,
        ml_backend = ?config.ml_backend,
        "starting simulation run"
    );
    let snapshot = Arc::new(snapshot);
    info!(
        target: "w1z4rdv1510n::orchestrator",
        symbols = snapshot.symbols.len(),
        timestamp = snapshot.timestamp.unix,
        "snapshot loaded"
    );
    let neuro_runtime = if config.neuro.enabled {
        Some(Arc::new(NeuroRuntime::new(snapshot.as_ref(), config.neuro.clone())))
    } else {
        None
    };
    let orchestrator_seed = random_provider.next_seed("orchestrator_rng");
    let mut rng = StdRng::seed_from_u64(orchestrator_seed);
    debug!(
        target: "w1z4rdv1510n::random",
        module = "orchestrator_rng",
        seed = orchestrator_seed
    );
    let ml_seed = random_provider.next_seed("ml_model");
    debug!(
        target: "w1z4rdv1510n::random",
        module = "ml_model",
        seed = ml_seed
    );
    let ml_model = create_ml_model(config.ml_backend.clone(), ml_seed);
    let search_module = SearchModule::new(config.search.clone());
    let hardware_seed = random_provider.next_seed("hardware_backend");
    let (hardware_backend, resolved_backend) = create_hardware_backend(
        config.hardware_backend.clone(),
        hardware_seed,
        &config.experimental_hardware,
        &config.hardware_overrides,
    )?;
    debug!(
        target: "w1z4rdv1510n::random",
        module = "hardware_backend",
        seed = hardware_seed
    );
    let energy_model = EnergyModel::new(
        Arc::clone(&snapshot),
        config.energy.clone(),
        Some(ml_model.clone()),
        Some(search_module.clone()),
        config.t_end,
        hardware_backend.tensor_executor(),
        neuro_runtime.clone(),
    );
    let base_population = init_population(
        snapshot.as_ref(),
        config.t_end,
        Some(&ml_model),
        &energy_model,
        config.n_particles,
        &config.init_strategy,
        &mut rng,
    );
    let initial_min_energy = base_population
        .particles
        .iter()
        .map(|p| p.energy)
        .fold(f64::INFINITY, f64::min);
    debug!(
        target: "w1z4rdv1510n::orchestrator",
        n_particles = base_population.particles.len(),
        min_energy = initial_min_energy,
        "population initialized"
    );

    let kernel_seed = random_provider.next_seed("proposal_kernel");
    let kernel = DefaultProposalKernel::new(
        config.proposal.clone(),
        kernel_seed,
        Some(search_module.clone()),
        Some(ml_model.clone()),
        neuro_runtime.clone(),
    );
    debug!(
        target: "w1z4rdv1510n::random",
        module = "proposal_kernel",
        seed = kernel_seed
    );
    let (population, energy_trace, acceptance_trace) = if config.quantum.enabled {
        let mut slices = Vec::with_capacity(config.quantum.trotter_slices.max(1));
        slices.push(base_population);
        for _ in 1..config.quantum.trotter_slices.max(1) {
            slices.push(init_population(
                snapshot.as_ref(),
                config.t_end,
                Some(&ml_model),
                &energy_model,
                config.n_particles,
                &config.init_strategy,
                &mut rng,
            ));
        }
        let outcome = anneal_quantum(
            slices,
            snapshot.as_ref(),
            &energy_model,
            &kernel,
            Some(&search_module),
            &config.schedule,
            &config.resample,
            &hardware_backend,
            &mut rng,
            &config.quantum,
        );
        let best_slice = select_best_population(&outcome.slices);
        (best_slice, outcome.energy_trace, outcome.acceptance_trace)
    } else {
        anneal(
            base_population,
            snapshot.as_ref(),
            &energy_model,
            &kernel,
            Some(&search_module),
            &config.schedule,
            &config.resample,
            &hardware_backend,
            &mut rng,
        )
    };
    let results = analyze_results(
        &population,
        energy_trace,
        &energy_model,
        Some(&search_module),
        snapshot.as_ref(),
    );
    info!(
        target: "w1z4rdv1510n::orchestrator",
        best_energy = results.best_energy,
        best_symbols = results.best_state.symbol_states.len(),
        "annealing run complete"
    );
    maybe_persist_results(&results, &config)?;
    Ok(RunOutcome {
        results,
        random_provider: random_descriptor,
        hardware_backend: resolved_backend,
        acceptance_ratio: acceptance_trace.last().copied(),
    })
}

fn select_best_population(slices: &[Population]) -> Population {
    slices
        .iter()
        .cloned()
        .min_by(|a, b| {
            min_population_energy(a)
                .partial_cmp(&min_population_energy(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|| {
            slices.first().cloned().unwrap_or(Population {
                particles: Vec::new(),
                temperature: 0.0,
                iteration: 0,
            })
        })
}

fn min_population_energy(population: &Population) -> f64 {
    population
        .particles
        .iter()
        .map(|p| p.energy)
        .fold(f64::INFINITY, f64::min)
}

fn load_snapshot(config: &RunConfig) -> anyhow::Result<EnvironmentSnapshot> {
    let data = fs::read_to_string(&config.snapshot_file)
        .with_context(|| format!("Failed to read {:?}", config.snapshot_file))?;
    let snapshot: EnvironmentSnapshot = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse snapshot {:?}", config.snapshot_file))?;
    Ok(snapshot)
}

fn maybe_persist_results(results: &Results, config: &RunConfig) -> anyhow::Result<()> {
    if let Some(path) = &config.output.output_path {
        let serialized = match config.output.format {
            OutputFormat::Json => serde_json::to_string_pretty(results)?,
            OutputFormat::Msgpack | OutputFormat::Custom => serde_json::to_string(results)?,
        };
        info!(
            target: "w1z4rdv1510n::results",
            path = ?path,
            format = ?config.output.format,
            "writing results"
        );
        fs::write(path, serialized)
            .with_context(|| format!("Failed to write results to {:?}", path))?;
    }
    if let Some(path) = &config.output.summary_path {
        let summary = serde_json::json!({
            "best_energy": results.best_energy,
            "best_symbol_count": results.best_state.symbol_states.len(),
            "energy_trace": results.diagnostics.energy_trace,
            "diversity_metric": results.diagnostics.diversity_metric,
        });
        info!(
            target: "w1z4rdv1510n::results",
            path = ?path,
            "writing summary"
        );
        fs::write(path, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("Failed to write summary to {:?}", path))?;
    }
    Ok(())
}
