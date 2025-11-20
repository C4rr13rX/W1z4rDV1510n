# W1z4rDV1510n

W1z4rDV1510n is a Rust-first annealing engine for simulating many parallel futures from a symbolic snapshot. It runs adaptive proposal kernels, search-aware constraints, and configurable energy functions to estimate the most likely end-state at `t_end`, while logging rich diagnostics for downstream analysis.

---

## Key Features

- **Automatic hardware scaling** – `HardwareBackendType::Auto` inspects CPU cores, memory, GPU hints, and cluster schedulers to pick CPU, multi-threaded CPU, or distributed backends. Overrides are available via config or env vars (`W1z4rDV1510n_HAS_GPU`, `W1z4rDV1510n_DISTRIBUTED`).
- **Search-integrated proposals** – Occupancy grids + A* planning, teleport-on-failure, and overlap repair keep particles feasible. Cached grids eliminate redundant rebuilds.
- **Goal-aware ML priors** – ML hooks (`None`, `SimpleRules`, `GoalAnchor`) provide initialization hints and contribute to energy scoring. GoalAnchor learns anchor destinations from trajectories.
- **Configurable randomness** – Deterministic, OS entropy, and (feature-gated) jitter-based RNG providers with per-module seed logging and reproducible runs when deterministic mode is selected.
- **ML-guided proposals** – Enable `proposal.ml_guided_move_prob` (>0) to let ML backends suggest coordinated moves in the sampler.
- **Structured logging** – Deterministic `tracing` configuration (JSON/compact) with per-iteration metrics, ESS resampling notices, hardware detection logs, and path diagnostics.
- **Calibration tooling** – `calibrate_energy` inspects recorded trajectories to recommend energy weights, plus schema validation and summary stats.
- **Validation scripts** – Python helpers convert chess PGNs and run perpetual accuracy loops to benchmark ML + logging pipelines.
- **Unit-test coverage** – Hardware selection, proposal mixing, search constraint repairs, and ML calibration each have dedicated tests (`cargo test`).

---

## Repository Layout

```
├── Cargo.toml / Cargo.lock
├── README.md
├── src/
│   ├── annealing.rs          ← MH loop, resampling, schedules
│   ├── calibration.rs        ← trajectory statistics → energy weights
│   ├── config.rs             ← RunConfig + nested serde structs
│   ├── hardware.rs           ← backend abstraction + auto detection
│   ├── logging.rs            ← tracing setup (env filters, file writers)
│   ├── ml.rs                 ← ML hooks (Null, SimpleRules, GoalAnchor)
│   ├── orchestrator.rs       ← wiring layer (snapshot + config → results)
│   ├── proposal.rs           ← adaptive move selection + kernels
│   ├── results.rs            ← best-state selection, path diagnostics
│   ├── schema.rs             ← timestamps, symbols, particles, trajectories
│   ├── search.rs             ← occupancy grids, A*, constraint repair
│   └── state_population.rs   ← init/resample/mutation utilities
├── src/main.rs               ← `predict_state` CLI
├── src/bin/calibrate_energy.rs
└── scripts/
    ├── preprocess_chess_games.py
    └── chess_training_loop.py
```

---

## Requirements

- Rust 1.74+ (install via [rustup](https://rustup.rs)).
- Python 3.10+ (optional, for data scripts + chess validation).
- Optional: `python-chess`, `tqdm` (`pip install python-chess tqdm`).

---

## Build & Test

```powershell
cargo fmt
cargo test
```

Tests cover calibration heuristics, hardware auto-selection, search constraint repairs, proposal mixing, and GoalAnchor ML updates.

---

## Running the Simulator

1. **Author a snapshot + config.** A `RunConfig` example (`run_config.json`):

```json
{
  "snapshot_file": "data/snapshot.json",
  "t_end": { "unix": 1732003600 },
  "n_particles": 512,
  "schedule": { "t_start": 5.0, "t_end": 0.2, "n_iterations": 200, "schedule_type": "Linear" },
  "energy": {
    "w_motion": 1.0,
    "w_collision": 5.0,
    "w_goal": 1.0,
    "w_group_cohesion": 0.5,
    "w_ml_prior": 0.2,
    "w_path_feasibility": 1.0,
    "w_env_constraints": 3.0
  },
  "proposal": {
    "local_move_prob": 0.5,
    "group_move_prob": 0.2,
    "swap_move_prob": 0.1,
    "path_based_move_prob": 0.15,
    "global_move_prob": 0.05,
    "ml_guided_move_prob": 0.05,
    "max_step_size": 1.0,
    "use_parallel_updates": true,
    "adaptive_move_mixing": true
  },
  "init_strategy": { "use_ml_priors": true, "noise_level": 0.4, "random_fraction": 0.2, "copy_snapshot_fraction": 0.1 },
  "resample": { "enabled": true, "effective_sample_size_threshold": 0.3, "mutation_rate": 0.05 },
  "search": { "cell_size": 0.75, "teleport_on_no_path": true },
  "ml_backend": "GoalAnchor",
  "hardware_backend": "Auto",
  "logging": { "log_level": "INFO", "json": true, "log_path": "logs/run.jsonl" },
  "output": { "save_best_state": true, "save_population_summary": true, "output_path": "logs/results.json", "format": "Json" },
  "random": { "provider": "DETERMINISTIC", "seed": 1337 }
}
```

2. **Run the CLI:**

```powershell
cargo run --bin predict_state -- --config run_config.json
```

Output:

- Console summary (`Best energy: … | symbols: …`).
- `logs/results.json` (best state, energy breakdown, path diagnostics).
- `logs/run.jsonl` (if `log_path` set): structured logs with iteration metrics.

### Logging Options

- `RunConfig.logging.log_level` - default level (override with `W1z4rDV1510n_LOG=debug` or `SIMFUTURES_LOG=debug` for backward compatibility).
- `logging.json` - `true` for JSON logs to stdout/file; `false` for compact text.
- `logging.log_path` - optional JSONL file target; directories auto-created.
- `HardwareBackendType::Auto` writes detection + fallback info at INFO/WARN.
- `random.provider` - choose `DETERMINISTIC` (reproducible with `seed`), `OS_ENTROPY` (draws from OS RNG), or `JITTER_EXPERIMENTAL` (feature-gated physical jitter). Major module seeds are logged at DEBUG level.

## REST Service API

Launch the built-in REST service (defaults to `0.0.0.0:8080` or override with `W1Z4RDV1510N_SERVICE_ADDR`):

```powershell
W1Z4RDV1510N_SERVICE_ADDR=0.0.0.0:8080 cargo run --bin service
```

POST `/predict` with a JSON body containing a `config` (same `RunConfig` schema; `snapshot_file` can be omitted) and an inline `snapshot`. Example request:

```json
{
  "config": {
    "t_end": { "unix": 1732003600 },
    "n_particles": 256,
    "snapshot_file": "inline",
    "energy": { "w_motion": 1.0, "w_collision": 5.0, "w_goal": 1.0, "w_group_cohesion": 0.5, "w_ml_prior": 0.2, "w_path_feasibility": 1.0, "w_env_constraints": 3.0 },
    "proposal": { "local_move_prob": 0.5, "group_move_prob": 0.2, "swap_move_prob": 0.1, "path_based_move_prob": 0.15, "global_move_prob": 0.05, "ml_guided_move_prob": 0.05, "max_step_size": 1.0 },
    "search": { "cell_size": 0.75 },
    "random": { "provider": "OS_ENTROPY" }
  },
  "snapshot": {
    "timestamp": { "unix": 1732000000 },
    "bounds": { "width": 12.0, "height": 8.0 },
    "symbols": [
      { "id": "agent_a", "type": "PERSON", "position": { "x": 2, "y": 2 }, "properties": {} },
      { "id": "exit_1", "type": "EXIT", "position": { "x": 11, "y": 7 }, "properties": {} }
    ]
  }
}
```

The response returns the usual `Results` plus telemetry metadata:

```json
{
  "results": { "...": "..." },
  "telemetry": {
    "random_provider": { "provider": "OS_ENTROPY", "deterministic": false, "seed": null },
    "hardware_backend": "MultiThreadedCpu"
  }
}
```

Errors yield HTTP 400 with `{ "error": "<message>" }`.

### Path Diagnostics

`Results.path_report` includes, per symbol:

```json
{
  "feasible": true,
  "path_length": 12.4,
  "visited_nodes": 187,
  "constraint_violations": 0,
  "failure_reason": null
}
```

Use these to flag actors teleported out of walls or lacking a viable path.

### Hardware Overrides

- Set `hardware_backend` in config to `Cpu`, `MultiThreadedCpu`, `Gpu`, etc.
- `W1z4rDV1510n_HAS_GPU=1` → Auto backend may choose GPU even without CUDA env vars.
- `W1z4rDV1510n_DISTRIBUTED=1` or `SLURM_JOB_ID` present → Auto prefers distributed backend.

---

## Energy Calibration Workflow

Use `calibrate_energy` to extract motion/collision/goal/group statistics from recorded trajectories:

```powershell
cargo run --bin calibrate_energy -- `
  --trajectories data/training_trajectories.json `
  --base-config config/energy_base.json `
  --output config/energy_calibrated.json
```

- Input: JSON array of `Trajectory` objects (`sequence: [DynamicState, …]`).
- Output: `EnergyConfig` with recommended weights (merging in other base fields).
- The tool validates sequence length, counts the samples contributing to each statistic, and prints summary stats so misformatted data is caught early.
- Integrate with a deployment pipeline by generating the calibrated file, then pointing `RunConfig.energy` at it before invoking `predict_state`.

---

## Chess Validation Loop (Optional)

1. Download PGNs to `data/chess/` (already ignored by git). Example sources: [PGNMentor](https://www.pgnmentor.com/).
2. Convert PGNs to JSONL:

```powershell
python scripts/preprocess_chess_games.py --max-games 25000
```

3. Run the perpetual accuracy loop:

```powershell
python scripts/chess_training_loop.py `
  --max-games 20000 `
  --epochs-per-iteration 5 `
  --log-file logs/chess_training_metrics.log
```

- Hashed SAN features + player/opening metadata feed a softmax outcome model at multiple observation scopes (6→40 plies) and frequency-based move predictors (+1→+20 moves ahead).
- Each iteration prints integer accuracies and appends a JSON record to `logs/chess_training_metrics.log`, enabling long-term analysis while the process runs unattended.

---

## Suggested Workflows

1. **Local simulation** – craft snapshot/config, run `predict_state`, examine `logs/results.json` + path diagnostics.
2. **Trajectory calibration** – collect ground-truth trajectories, run `calibrate_energy`, inject weights into production configs.
3. **Hardware smoke test** – set `hardware_backend = "Auto"` and verify logs (CPU-only, GPU-enabled, cluster hints).
4. **Continuous regression** – keep the chess script running overnight to observe accuracy trends; inspect the JSON log in the morning.

---

## Contributing

- Format + lint: `cargo fmt && cargo clippy`.
- Test: `cargo test` (unit tests for hardware, proposals, search, ML, calibration).
- Python helpers expect `python-chess` + `tqdm`; install via `pip install python-chess tqdm`.
- Heavy artifacts (`data/chess`, `logs/`, generated configs) are ignored via `.gitignore`.
