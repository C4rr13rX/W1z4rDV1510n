use crate::annealing::anneal;
use crate::config::{OutputFormat, RunConfig};
use crate::energy::EnergyModel;
use crate::hardware::{HardwareBackendType, create_hardware_backend};
use crate::logging::{LiveFrameSink, LiveNeuroSink, init_logging};
use crate::ml::create_ml_model;
use crate::neuro::NeuroRuntime;
use crate::objective::GameObjective;
use crate::proposal::DefaultProposalKernel;
use crate::quantum::anneal_quantum;
use crate::quantum_calibration::{
    QuantumCalibrationState, build_calibration_job, build_calibration_request,
    parse_calibration_response,
};
use crate::quantum_executor::QuantumHttpExecutor;
use crate::random::{RandomProviderDescriptor, create_random_provider};
use crate::results::{Results, analyze_results};
use crate::schema::{EnvironmentSnapshot, Population, Position};
use crate::search::SearchModule;
use crate::state_population::init_population;
use anyhow::Context;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashSet;
use std::fs;
use std::sync::Arc;
use tracing::{debug, info, warn};
use crate::compute::QuantumExecutor;

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
    validate_snapshot(&snapshot)?;
    let live_frame = LiveFrameSink::from_config(&config.logging);
    let live_neuro = LiveNeuroSink::from_config(&config.logging);
    let random_provider = create_random_provider(&config.random, config.random_seed)?;
    let random_descriptor = random_provider.descriptor();
    let calibration_seed = random_provider.next_seed("quantum_calibration");
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
    let calibration_run_id = format!(
        "quantum-calibration-{}-{}",
        snapshot.timestamp.unix,
        calibration_seed
    );
    info!(
        target: "w1z4rdv1510n::orchestrator",
        symbols = snapshot.symbols.len(),
        timestamp = snapshot.timestamp.unix,
        "snapshot loaded"
    );
    let mut calibration_state = config
        .quantum
        .calibration_path
        .as_ref()
        .and_then(|path| match QuantumCalibrationState::load(path) {
            Ok(state) => Some(state),
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::annealing::quantum",
                    error = %err,
                    path = ?path,
                    "failed to load quantum calibration state"
                );
                None
            }
        });
    let mut quantum_config = config.quantum.clone();
    if let Some(state) = calibration_state.as_ref() {
        quantum_config = state.apply(&quantum_config);
    }
    let quantum_executor = if config.quantum.remote_enabled
        && config.compute.allow_quantum
        && !config.compute.quantum_endpoints.is_empty()
    {
        match QuantumHttpExecutor::new(config.compute.quantum_endpoints.clone()) {
            Ok(executor) => Some(executor),
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::annealing::quantum",
                    error = %err,
                    "failed to initialize quantum HTTP executor"
                );
                None
            }
        }
    } else {
        None
    };
    let neuro_runtime = if config.neuro.enabled {
        Some(Arc::new(NeuroRuntime::new(
            snapshot.as_ref(),
            config.neuro.clone(),
        )))
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
        GameObjective::new(config.objective.clone()),
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
            neuro_runtime.clone(),
            &config.homeostasis,
            &mut rng,
            &quantum_config,
            live_frame.as_ref(),
            live_neuro.as_ref(),
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
            neuro_runtime.clone(),
            &config.homeostasis,
            live_frame.as_ref(),
            live_neuro.as_ref(),
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
    if config.quantum.remote_enabled {
        if calibration_state.is_none() {
            warn!(
                target: "w1z4rdv1510n::annealing::quantum",
                "quantum remote enabled but calibration_path is not set; skipping calibration"
            );
        } else if let Some(executor) = quantum_executor.as_ref() {
            let request = build_calibration_request(
                calibration_run_id.clone(),
                results.best_energy,
                acceptance_trace.last().copied(),
                &results.diagnostics.energy_trace,
                &quantum_config,
                config.quantum.remote_trace_samples,
            );
            match build_calibration_job(&request, config.quantum.remote_timeout_secs) {
                Ok(job) => match executor.submit(job) {
                    Ok(result) => match parse_calibration_response(result) {
                        Ok(response) => {
                            if let Some(state) = calibration_state.as_mut() {
                                state.update_from_adjustment(
                                    &response.adjustments,
                                    config.quantum.calibration_alpha,
                                );
                                if let Some(path) = config.quantum.calibration_path.as_ref() {
                                    if let Err(err) = state.save(path) {
                                        warn!(
                                            target: "w1z4rdv1510n::annealing::quantum",
                                            error = %err,
                                            path = ?path,
                                            "failed to persist quantum calibration"
                                        );
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            warn!(
                                target: "w1z4rdv1510n::annealing::quantum",
                                error = %err,
                                "quantum calibration response rejected"
                            );
                        }
                    },
                    Err(err) => {
                        warn!(
                            target: "w1z4rdv1510n::annealing::quantum",
                            error = %err,
                            "quantum calibration request failed"
                        );
                    }
                },
                Err(err) => {
                    warn!(
                        target: "w1z4rdv1510n::annealing::quantum",
                        error = %err,
                        "failed to build quantum calibration job"
                    );
                }
            }
        } else {
            warn!(
                target: "w1z4rdv1510n::annealing::quantum",
                "quantum remote enabled but no executor available"
            );
        }
    }
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

fn validate_snapshot(snapshot: &EnvironmentSnapshot) -> anyhow::Result<()> {
    let width = *snapshot.bounds.get("width").unwrap_or(&f64::NAN);
    let height = *snapshot.bounds.get("height").unwrap_or(&f64::NAN);
    let depth = snapshot.bounds.get("depth").copied();
    anyhow::ensure!(
        width.is_finite() && width > 0.0 && height.is_finite() && height > 0.0,
        "snapshot bounds must include positive width and height"
    );
    if let Some(d) = depth {
        anyhow::ensure!(
            d.is_finite() && d > 0.0,
            "depth, if provided, must be positive"
        );
    }
    let mut ids = HashSet::new();
    for symbol in &snapshot.symbols {
        anyhow::ensure!(
            ids.insert(&symbol.id),
            "duplicate symbol id '{}' in snapshot",
            symbol.id
        );
        ensure_position("symbol", &symbol.id, &symbol.position, width, height, depth)?;
    }
    for (frame_idx, frame) in snapshot.stack_history.iter().enumerate() {
        for (id, state) in &frame.symbol_states {
            ensure_position("stack_history", id, &state.position, width, height, depth)
                .with_context(|| format!("stack_history frame {}", frame_idx))?;
            anyhow::ensure!(
                ids.contains(id),
                "stack_history references unknown symbol id '{}'",
                id
            );
        }
    }
    Ok(())
}

fn ensure_position(
    scope: &str,
    id: &str,
    pos: &Position,
    width: f64,
    height: f64,
    depth: Option<f64>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
        "{} '{}' position must be finite",
        scope,
        id
    );
    anyhow::ensure!(
        pos.x >= 0.0 && pos.x <= width && pos.y >= 0.0 && pos.y <= height,
        "{} '{}' position is out of bounds",
        scope,
        id
    );
    if let Some(d) = depth {
        anyhow::ensure!(
            pos.z >= 0.0 && pos.z <= d,
            "{} '{}' z position is out of bounds",
            scope,
            id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        EnvironmentSnapshot, Properties, Symbol, SymbolState, SymbolType, Timestamp,
    };

    fn base_snapshot() -> EnvironmentSnapshot {
        EnvironmentSnapshot {
            timestamp: Timestamp { unix: 0 },
            bounds: [("width".into(), 10.0), ("height".into(), 5.0)]
                .into_iter()
                .collect(),
            symbols: vec![Symbol {
                id: "a".into(),
                symbol_type: SymbolType::Person,
                position: Position {
                    x: 1.0,
                    y: 1.0,
                    z: 0.0,
                },
                properties: Properties::new(),
            }],
            metadata: Properties::new(),
            stack_history: Vec::new(),
        }
    }

    #[test]
    fn snapshot_validation_rejects_duplicate_ids() {
        let mut snap = base_snapshot();
        snap.symbols.push(Symbol {
            id: "a".into(),
            symbol_type: SymbolType::Person,
            position: Position {
                x: 2.0,
                y: 2.0,
                z: 0.0,
            },
            properties: Properties::new(),
        });
        assert!(validate_snapshot(&snap).is_err());
    }

    #[test]
    fn snapshot_validation_rejects_non_finite_positions() {
        let mut snap = base_snapshot();
        snap.symbols[0].position.x = f64::NAN;
        assert!(validate_snapshot(&snap).is_err());
    }

    #[test]
    fn snapshot_validation_rejects_missing_bounds() {
        let mut snap = base_snapshot();
        snap.bounds.clear();
        assert!(validate_snapshot(&snap).is_err());
    }

    #[test]
    fn snapshot_validation_rejects_out_of_bounds_symbol() {
        let mut snap = base_snapshot();
        snap.symbols[0].position.x = 50.0;
        assert!(validate_snapshot(&snap).is_err());
    }

    #[test]
    fn snapshot_validation_rejects_out_of_bounds_stack_symbol() {
        let mut snap = base_snapshot();
        snap.stack_history.push(crate::schema::DynamicState {
            timestamp: Timestamp { unix: 0 },
            symbol_states: [(
                "a".into(),
                SymbolState {
                    position: Position {
                        x: -1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
        });
        assert!(validate_snapshot(&snap).is_err());
    }

    #[test]
    fn snapshot_validation_rejects_unknown_stack_symbol() {
        let mut snap = base_snapshot();
        snap.stack_history.push(crate::schema::DynamicState {
            timestamp: Timestamp { unix: 0 },
            symbol_states: [(
                "missing".into(),
                SymbolState {
                    position: Position {
                        x: 1.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
        });
        assert!(validate_snapshot(&snap).is_err());
    }

    #[test]
    fn snapshot_validation_passes_for_valid_snapshot() {
        let snap = base_snapshot();
        assert!(validate_snapshot(&snap).is_ok());
    }
}
