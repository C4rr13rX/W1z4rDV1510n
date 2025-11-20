# SimFutures – Parallel Future Annealing (Rust)

This repository hosts a Rust-first scaffold for the multi-hypothesis simulation engine you described. The current implementation establishes configuration schemas, module boundaries, and a working binary (`predict_state`) that loads a snapshot/config pair, spins up a particle population, runs a lightweight annealing loop, and emits best-state/diagnostic data.

## Project layout

```text
src/
  annealing.rs        # Temperature schedule + MH loop + resampling hooks
  config.rs           # RunConfig + nested config structs (serde-friendly)
  energy.rs           # Weighted energy components (motion/collision/etc.) + diagnostics
  hardware.rs         # Backend abstraction (CPU baseline today)
  ml.rs               # ML hook trait + null & simple heuristic backends
  orchestrator.rs     # load config/snapshot, wire modules, persist results
  proposal.rs         # Configurable proposal kernel with local moves
  results.rs          # Ground-state selection + diagnostics helpers
  schema.rs           # Core data types (timestamps, symbols, population)
  search.rs           # Occupancy-aware grid planner + constraint repair
  state_population.rs # Population init/resample/mutation utilities
  main.rs             # CLI entrypoint (predict_state --config run.json)
```

Key crates: `serde` (I/O), `clap` (CLI), `rand` (sampling), `anyhow` (error flow), `parking_lot` (cheap locking for RNG-bound components).

## Usage

1. Prepare a JSON snapshot (matches `EnvironmentSnapshot`) and a `RunConfig` JSON. Minimal example:

```jsonc
{
  "snapshot_file": "snapshot.json",
  "t_end": { "unix": 1731900000 },
  "n_particles": 256,
  "schedule": { "t_start": 5.0, "t_end": 0.1, "n_iterations": 100 },
  "energy": { "w_motion": 1, "w_collision": 5, "w_goal": 1, "w_group_cohesion": 0.5, "w_ml_prior": 0, "w_path_feasibility": 1, "w_env_constraints": 3 },
  "proposal": { "local_move_prob": 1.0, "max_step_size": 1.0 },
  "init_strategy": { "use_ml_priors": true, "noise_level": 0.5, "random_fraction": 0.2, "copy_snapshot_fraction": 0.1 },
  "resample": { "enabled": true, "effective_sample_size_threshold": 0.3, "mutation_rate": 0.05 },
  "search": { "cell_size": 1.0 },
  "ml_backend": "SIMPLE_RULES",
  "hardware_backend": "Cpu",
  "random_seed": 42,
  "output": { "save_best_state": true, "output_path": "results.json", "format": "Json" }
}
```

2. Build & run:

```powershell
cargo run -- --config run_config.json
```

This prints the best energy and writes `results.json` when configured.

## Current capabilities

- ? Rust crate wiring with dedicated modules per architectural block.
- ? Data schemas and serde-powered config ingestion.
- ? Basic ML hooks (null + heuristic), proposal kernel, and annealing loop.
- ? CPU backend abstraction for particle updates and result persistence.
- ? Working CLI with JSON config I/O.
- ? Richer energy model with volumetric collisions, wall penalties, ML priors, and per-symbol diagnostics (surfaced via `Results.diagnostics.best_state_breakdown`).
- ? Occupancy-aware search module with configurable A* grid planner, wall/object masking, constraint projection, and path diagnostics that feed energy scoring.
- ? Search grid caching + overlap repair keeps proposals feasible even in dense environments, with optional teleportation when no path exists.
- ? Synthetic timeline dataset generator + automation scripts for spawning background runs and tailing their logs.
- ? Hardware backend abstraction with CPU, multi-threaded CPU, and external stubs, plus an exposed noise-source hook for experimental entropy providers.
- ? Hardware backends now auto-select (via `HardwareBackendType::Auto`) based on detected CPU cores, memory, GPU/env hints (`SIMFUTURES_HAS_GPU`, `SIMFUTURES_DISTRIBUTED`), scaling from Pi-class devices up to multi-GPU clusters.
- ? Results now capture per-symbol path diagnostics (feasibility, path length, constraint violations) for the best state, giving downstream consumers a richer view into path feasibility.
- ? Proposal kernel now mixes local/group/swap/path/global moves adaptively based on temperature, improving diversity at high temperatures and focusing on path-following as the search cools.
- ? ML backend can ingest historical trajectories (via the RNN backend option) to learn per-symbol goal anchors, yielding better position predictions and plausibility scoring than the simple heuristic fallback.

## Synthetic timeline dataset

Use the helper script to fabricate a dense, multi-entity scenario with chaotic thought labels and timelines:

```powershell
python scripts/generate_synthetic_dataset.py --output-dir data
```

This emits:

- `data/synthetic_snapshot.json` — >250 symbols (people, furniture, vehicles, walls, exits) with metadata hooks.
- `data/timeline_train.jsonl` / `data/timeline_test.jsonl` — ~43k timeline rows containing per-symbol positions, velocities, and multi-label "thought"/attention annotations.
- `data/synthetic_run_config.json` — ready-to-run config pointing at the snapshot.
- `data/synthetic_dataset_metadata.json` — summary counts, split sizes, and timeline statistics.

Tweaks (number of entities, steps, search cell size, etc.) can be passed via CLI flags; see `--help` for the full list.

## Background runner + log tail

Kick off `predict_state` in the background with stdout/stderr streamed into `logs/`:

```powershell
pwsh -File scripts/run_background_simulation.ps1        # add -DryRun to only preview the command
```

The script reports the PID, log file path, and writes `logs/latest_run.json` for downstream tooling. Tail progress (from another terminal) using:

```powershell
pwsh -File scripts/tail_simulation_logs.ps1             # accepts -LogPath and -StopAfterSeconds overrides
```

The tailer watches the log specified in `latest_run.json` by default, prints the last 40 lines, and keeps streaming until you Ctrl+C (or until `-StopAfterSeconds` elapses).

## Training data & calibration

If you have recorded trajectories (see `src/schema::Trajectory` for the JSON shape) you can automatically derive reasonable energy weights via the calibration utility:

```powershell
cargo run --bin calibrate_energy -- --trajectories data/training_trajectories.json --output calibrated_energy.json
```

The input should be a JSON array of trajectories (each trajectory is a sequence of `DynamicState` snapshots). The tool inspects displacement, collision frequency, goal adherence, and group spread to produce a set of recommended energy weights. The resulting JSON can be merged into your `RunConfig.energy` block or used as a starting point for manual tuning.

## Next steps / roadmap

1. **Search module**: extend the new occupancy grid/A* planner with cached builds, nav-mesh or multi-resolution search, and richer multi-agent coordination (reservation tables, crowd-flow penalties).
2. **Population ops**: add stratified/systematic resampling modes, diversified mutation kernels, and history tracking per particle.
3. **Annealing upgrades**: support parallel temperature ladders, configurable acceptance criteria, and pluggable schedules (import from config via `ScheduleType::Custom` hook).
4. **Hardware backends**: add multi-threaded CPU backend and stubs for GPU/external solver hooks; expose `NoiseSource` API for experimental entropy sources.
5. **ML integration**: implement real ML backends (bindings to ONNX, local Torch, etc.), plus streaming updates via `update_from_data`.
6. **I/O & API**: add REST/gRPC surface, config validation, richer snapshot serialization (Msgpack) once `OutputFormat::Msgpack` is wired.
7. **Testing**: create unit tests per module (energy terms, proposal invariants, config parsing) and deterministic synthetic scenarios for regression coverage.

Each module was documented and structured so another AI (or developer) can independently extend it—continue following that contract when fleshing out future stages. Feel free to tag TODOs inline as deeper behaviors are implemented.

---

## Next update hand-off

- Cache the occupancy grid per snapshot (or diff it incrementally) so annealing proposals no longer rebuild the environment mask each call; expose metrics so configs can tune cell size adaptively.
- Surface the new path diagnostics (length, constraint violations, visited nodes) in the CLI/log output and persist them in `Results` for downstream analysis/visualization.
- Layer in cooperative routing: allow the planner/resampler to treat fellow agents as soft obstacles (time-expanded or reservation-based) so multi-agent coordination benefits from the grid model.
