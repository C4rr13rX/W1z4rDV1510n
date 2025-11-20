use crate::annealing::anneal;
use crate::config::{OutputFormat, RunConfig};
use crate::energy::EnergyModel;
use crate::hardware::create_hardware_backend;
use crate::ml::create_ml_hooks;
use crate::proposal::DefaultProposalKernel;
use crate::results::{Results, analyze_results};
use crate::schema::EnvironmentSnapshot;
use crate::search::SearchModule;
use crate::state_population::init_population;
use anyhow::Context;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;
use std::sync::Arc;

pub fn run_with_config(config: RunConfig) -> anyhow::Result<Results> {
    config.validate()?;
    let snapshot = Arc::new(load_snapshot(&config)?);
    let mut rng = StdRng::seed_from_u64(config.random_seed);
    let ml_hooks = create_ml_hooks(config.ml_backend.clone(), config.random_seed);
    let search_module = SearchModule::new(config.search.clone());
    let energy_model = EnergyModel::new(
        Arc::clone(&snapshot),
        config.energy.clone(),
        Some(ml_hooks.clone()),
        Some(search_module.clone()),
        config.t_end,
    );
    let population = init_population(
        snapshot.as_ref(),
        config.t_end,
        Some(&ml_hooks),
        &energy_model,
        config.n_particles,
        &config.init_strategy,
        &mut rng,
    );

    let kernel = DefaultProposalKernel::new(
        config.proposal.clone(),
        config.random_seed + 1,
        Some(search_module.clone()),
    );
    let hardware_backend = create_hardware_backend(config.hardware_backend.clone(), config.random_seed);
    let (population, energy_trace) = anneal(
        population,
        snapshot.as_ref(),
        &energy_model,
        &kernel,
        Some(&search_module),
        &config.schedule,
        &config.resample,
        &hardware_backend,
        &mut rng,
    );
    let results = analyze_results(
        &population,
        energy_trace,
        &energy_model,
        Some(&search_module),
        snapshot.as_ref(),
    );
    maybe_persist_results(&results, &config)?;
    Ok(results)
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
        fs::write(path, serialized)
            .with_context(|| format!("Failed to write results to {:?}", path))?;
    }
    Ok(())
}
