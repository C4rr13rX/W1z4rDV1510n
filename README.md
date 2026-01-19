# W1z4rDV1510n

**Ultradian Rhythm Dynamic Stability Analysis System** - flag-driven mode for layered ultradian phase/amplitude/coherence tracking, stability baselines, and cross-entity motif discovery with human-first governance.

W1z4rDV1510n is a Rust-first, quantum-inspired annealer fused with a brain-like neural fabric for predicting and completing symbol sequences, plus a CPU-first node stack for distributed execution and multi-chain stablecoin deposits. It simulates many parallel futures from any symbolic snapshot (chess plies, reaction steps, traffic frames, code edits) and uses relational priors, neurogenesis, and simulated quantum annealing to fill gaps and forecast where symbols will be next on an i5/32GB CPU.

---

## Why this matters (engineers, scientists, data folks)

- **Domain-agnostic sequence reasoning** – chess games, USPTO reaction steps, crowd/traffic frames, clickstreams, market ticks, log lines, code edits: if you can encode symbols/roles/positions, the sampler can predict/complete the sequence.
- **Gap-filling + forecasting** – handles sparse or staggered histories, predicts missing frames, and projects plausible futures with stack hashing + relational priors.
- **Low-overhead ML** – no GPU required; thread-capped, CPU-friendly kernels with deterministic seeds keep experiments reproducible on modest hardware.
- **Inspectable & tunable** – every energy term, prior, neuro plasticity knob, and homeostasis control is JSON-configurable and logged.
- **Cross-discipline fusion** – quantum-inspired annealing, graph/relational hashing, and cortical-style neurogenesis cooperate in one engine.

## Core capabilities

- **Hardware-smart execution** - `HardwareBackendType::Auto` chooses CPU, multi-threaded CPU, GPU, or distributed; env/overrides cap threads and disable accelerators when needed.
- **Search-integrated proposals** - occupancy grids + A* planning, teleport-on-failure, and overlap repair keep particles feasible; caches avoid rebuild churn.
- **Goal/role/relational priors** - ML hooks (`None`, `SimpleRules`, `GoalAnchor`), relational graphs, and hashed motifs bias initialization and energy scoring without heavy models.
- **Quantum-inspired stacks** - optional `quantum` mode couples Trotter slices; `energy.w_stack_hash` aligns with past trajectories (`stack_history`) and pulls toward likely futures for gap-filling.
- **Brain-like neuro layer** - neurons carry dendrites/axon splits, excitatory & inhibitory synapses, light synaptic fatigue, winner-take-all sparsity, and STDP-lite nudges. The pool learns co-occurring roles/zones, spawns composite neurons + cross-linked mini-networks, and feeds proposal biasing plus `energy.w_neuro_alignment`.
- **Resource-aware scheduling** - capped rayon pools, memory-aware budgets (`W1Z4RDV1510N_THREAD_BUDGET`, `hardware_overrides.max_threads`), and chunked updates keep an i5/32GB responsive.
- **Homeostasis controller** - detects plateaus and reheats/mutates briefly to escape local minima while chasing 90-100% accuracy.
- **Structured telemetry** - deterministic tracing (JSON/compact), ESS/resampling notices, occupancy/path diagnostics, hardware detection logs, and persisted run artifacts.
- **Knowledge ingest + human annotation queue** - JATS (NLM/NCBI) parsing, figure/text association tasks, optional OCR blocks, and reward-ready verification.
- **Tooling** - `calibrate_energy` for weight tuning, REST API with persistence, relational prior builder, chess/chemistry scripts, node simulation, and comprehensive tests (`cargo test`).

## Node stack capabilities

- **P2P networking** - libp2p gossipsub + Kademlia + mDNS discovery with rate limits, message validation, and clock skew checks.
- **Local ledger** - reward tracking, validator heartbeats, fee market scaffolding, and immutable audit logs.
- **Wallet** - encrypted Ed25519 wallet with deterministic address derivation for node identity and rewards.
- **Bridge + deposits** - multi-chain stablecoin deposits with relayer quorum verification, intent idempotency, and offline tooling.
- **API services** - auth + rate limits + metrics + health/readiness for bridge, balance, and operations endpoints.
- **Data mesh ingestion** - manifest/chunk/receipt replication with quorum tracking, retention GC, and integrity audits; `/data/ingest` + `/data/:data_id` endpoints.
- **Simulation + OpenStack stubs** - 10k node simulation and minimal OpenStack control-plane abstractions for future cluster plans.

## Governance and streaming flags

- **Governance defaults** - no face ID, no PII, public-only enforcement, immutable audit logs, explainability on, and optional DP/federated learning toggles.
- **Streaming flags** - config-driven ingestion for people video, crowd/traffic, and public topics, plus layer flags for ultradian/flow/topic/physiology.

## How it works (stack)

- **Annealer**: multi-proposal MCMC + simulated/quantum annealing with adaptive move mixing.
- **Relational priors**: graph-hash motifs and transition tables steer moves toward historically consistent structures.
- **Neural fabric**: emergent motifs, centroids, mini-networks, cross-network links; bias proposals and energy toward coherent “cortical” patterns.
- **Homeostasis**: detects stagnation and temporarily raises temperature/mutation to keep searching.

## Example workflows

- **Chess (gap-fill + forecast)**: train lightweight outcome/move models (`scripts/chess_training_loop.py`), export sparse stacks, and run the annealer with `energy.w_stack_hash` to complete missing plies and predict futures.
- **Chemistry (USPTO reactions)**: fetch USPTO-50K or full Lowe dataset (`scripts/fetch_uspto_reactions.py`), hash reaction steps into symbols/roles, build relational priors, and anneal to propose likely next steps or full routes.
- **People/traffic/time-series**: encode agents/roles/zones per frame; use relational priors + neuro alignment to forecast trajectories and fill missing observations.

### Architecture Overview

- **config** - validated `RunConfig`/`RunMode`, hardware overrides, logging/output knobs, and production guardrails.
- **state_population** - particle initialization, weight normalization (tensor-accelerated), resampling, and mutation utilities.
- **proposal** - adaptive mixing of local/group/swap/path/global moves plus ML-guided proposals and temperature-aware weights.
- **search** - occupancy grid caching, teleport-on-no-path, overlap repair, and A* diagnostics feeding the path energy term.
- **energy** - decomposed cost terms (motion, collision, goals, env constraints, ML priors) with tensor/vector fast paths when available.
- **hardware** - CPU, multi-threaded CPU, RAM-optimized, GPU (`wgpu`), distributed, and experimental backends + auto-detection/fallback logic.
- **service_api / service_storage / telemetry** - shared request/response structs, run persistence, health/readiness metrics, and run-id telemetry for the Axum service.
- **bin/predict_state** - CLI driver for JSON configs; **bin/service** - REST API with health probes + persistent runs.
- **crates/node** - node runtime (config, wallet, p2p, local ledger, bridge, API, OpenStack stubs, simulation).
- **chain/** - chain spec, reward, bridge, and token-standard JSON configs.

## Multi-Chain Bridge (stablecoin deposits)

- **Goal** - fund nodes with stablecoin deposits (USDC/USDT) from existing chains, no bank account required.
- **Flow** - deposit on source chain -> relayer quorum signs canonical `BridgeDeposit` payload -> submit `BridgeProof` -> ledger mints `StakeDeposit`.
- **Security tiers** - `RELAYER_QUORUM` (implemented), `OPTIMISTIC` (challenge window), `LIGHT_CLIENT`/`ZK` (highest security, planned).
- **Config** - `blockchain.bridge` in `node_config_example.json` controls chain policies, relayer keys/quorum, allowed assets, max deposit sizes, and per-chain `deposit_address` + `recipient_tag_template`.
- **API** - `GET /bridge/chains` lists enabled chains; `POST /bridge/intent` returns deposit instructions with an idempotent `intent_id` (optional `idempotency_key` supported); `POST /bridge/proof` submits relayer-signed proofs; `GET /balance/:node_id` and `GET /metrics` report status.
- **CLI** - `w1z4rdv1510n-node bridge-intent-create` generates offline intents + payload hashes; `w1z4rdv1510n-node bridge-intent-verify` validates intent payload hashes.
- **Schema** - `schemas/bridge_intent_schema.json` defines the `BridgeIntent` JSON shape for offline tooling and validation.
- **Extensibility** - add new chains by appending policies; `chain/bridge_contract.json` lists supported chains/assets.

---

## Repository Layout

- Cargo.toml (workspace)
- README.md
- crates/core (engine + binaries)
- crates/node (node runtime, ledger, wallet, p2p, API, bridge, simulation)
- crates/experimental-hw (experimental hardware backends)
- chain/ (chain spec + token/bridge configs)
- schemas/ (JSON schemas for offline tooling)
- scripts/
- data/, logs/, etc.

## Requirements

- Rust 1.74+ (install via [rustup](https://rustup.rs)).
- Python 3.10+ (optional, for data scripts + chess validation).
- Optional: `python-chess`, `tqdm` (`pip install python-chess tqdm`).

### Optional Cargo features

- `gpu` – enables the true GPU backend (requires a Vulkan/Metal/DirectX12-capable adapter).
- `experimental-hw` – builds the experimental hardware backend; must be combined with `mode = "LAB_EXPERIMENTAL"`.
- `distributed` – reserved for future remote/offload integrations (presently a no-op feature flag).

---

## Build & Test

```powershell
cargo fmt
cargo test
```

Tests cover calibration heuristics, hardware auto-selection, search constraint repairs, proposal mixing, GoalAnchor ML updates, and node ledger/bridge logic. For node-only tests: `cargo test -p w1z4rdv1510n-node`.

---

## Node quickstart

1. **Initialize config + wallet**

```powershell
cargo run --bin w1z4rdv1510n-node -- init
```

2. **Run the node (blocks until Ctrl+C)**

```powershell
cargo run --bin w1z4rdv1510n-node
```

3. **Start the node API (optional)**

```powershell
cargo run --bin w1z4rdv1510n-node -- api --addr 127.0.0.1:8090
```

The node API exposes `GET /health` and `GET /ready` (auth required if `api.require_api_key` is enabled).

4. **Simulate a network (optional)**

```powershell
cargo run --bin w1z4rdv1510n-node -- sim --nodes 10000 --ticks 100
```

5. **Create and verify a bridge intent offline (optional)**

```powershell
cargo run --bin w1z4rdv1510n-node -- bridge-intent-create `
  --chain-id ethereum --chain-kind evm --asset USDC --amount 25 `
  --recipient-node-id node-abc123 --deposit-address 0xbridge `
  --recipient-tag node:node-abc123

cargo run --bin w1z4rdv1510n-node -- bridge-intent-verify --json '{...}'
```

6. **Knowledge ingest (offline CLI)**

```powershell
cargo run --bin w1z4rdv1510n-node -- knowledge-ingest --xml-file data/article.xml --out logs/knowledge_ingest.json
cargo run --bin w1z4rdv1510n-node -- knowledge-vote --ingest-file logs/knowledge_ingest.json --votes-file logs/knowledge_votes.json
```

Config lives in `node_config.json` (generated from defaults). `node_config_example.json` shows all available fields, including QUIC/AutoNAT routing toggles, API keys, rate limits, bridge policies, and data mesh retention/audit settings.

---

## Running the Simulator

1. **Author a snapshot + config.** A `RunConfig` example (`run_config.json`):

```json
{
  "snapshot_file": "data/snapshot.json",
  "mode": "PRODUCTION",
  "t_end": { "unix": 1732003600 },
  "n_particles": 512,
  "schedule": { "t_start": 5.0, "t_end": 0.2, "n_iterations": 200, "schedule_type": "Linear" },
  "energy": {
    "w_motion": 1.0,
    "w_collision": 5.0,
    "w_goal": 1.0,
    "w_group_cohesion": 0.5,
    "w_ml_prior": 0.2,
    "w_relational_prior": 0.4,
    "w_neuro_alignment": 0.2,
    "relational_priors_path": "data/relational_priors.json",
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
  "hardware_overrides": {
    "allow_gpu": false,
    "allow_distributed": true,
    "max_threads": 12
  },
  "neuro": { "enabled": true, "min_activation": 0.6, "module_threshold": 40, "max_networks": 256 },
  "homeostasis": { "enabled": true, "energy_plateau_tolerance": 0.0001, "patience": 8, "mutation_boost": 0.5, "reheat_scale": 0.2 },
  "logging": { "log_level": "INFO", "json": true, "log_path": "logs/run.jsonl" },
  "output": { "save_best_state": true, "save_population_summary": true, "output_path": "logs/results.json", "format": "Json" },
  "random": { "provider": "DETERMINISTIC", "seed": 1337 },
  "experimental_hardware": { "enabled": false }
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
- **Quantum + stack hashing:** add a `stack_history` array to the snapshot, set `energy.w_stack_hash` > 0, and toggle `"quantum": { "enabled": true, ... }` to run the coupled Trotter-slice annealer. `energy.stack_alignment_topk`, `energy.stack_alignment_weight`, `energy.stack_future_horizon`, and `energy.stack_future_weight` let you bias toward the closest history frames (by overlap distance) and softly pull particles toward where those frames are headed, which is handy when the observed sequence is sparse/staggered. `quantum.driver_final_strength` lets you taper the transverse-field driver toward the end of the schedule.

### Logging Options

- `RunConfig.logging.log_level` - default level (override with `W1z4rDV1510n_LOG=debug` or `SIMFUTURES_LOG=debug` for backward compatibility).
- `logging.json` - `true` for JSON logs to stdout/file; `false` for compact text.
- `logging.log_path` - optional JSONL file target; directories auto-created.
- `HardwareBackendType::Auto` writes detection + fallback info at INFO/WARN.
- `random.provider` - choose `DETERMINISTIC` (reproducible with `seed`), `OS_ENTROPY` (draws from OS RNG), or `JITTER_EXPERIMENTAL` (feature-gated physical jitter). Major module seeds are logged at DEBUG level.
- `mode` - defaults to `PRODUCTION`. Switch to `LAB_EXPERIMENTAL` when using jitter RNG or the experimental hardware backend. Production mode automatically rejects unsafe combinations.
- `hardware_backend` - add `"Experimental"` (requires `--features experimental-hw` and `experimental_hardware.enabled: true`) to blend physical telemetry into the annealer. Sensors are read passively; no destructive behavior occurs.
- `experimental_hardware` - tuning knobs for the experimental backend (`use_thermal`, `use_performance_counters`, `max_sample_interval_secs`). Leave `enabled: false` unless running in lab mode.

### Neurogenesis & Alignment

- Enable the lightweight neural fabric via `neuro.enabled: true`; tune decay/thresholds with `neuro.min_activation`, `neuro.module_threshold`, and `neuro.max_networks`. Inside the inner `neuro` block, adjust `fatigue_increment` / `fatigue_decay`, `wta_k_per_zone` (lateral inhibition), and `stdp_scale` (STDP-lite) for brain-like dynamics.
- Reward consistency with emergent centroids/networks by setting `energy.w_neuro_alignment > 0` (combine with relational/stack terms for domain-agnostic structure).
- Proposal kernels automatically consume neuro snapshots to nudge symbols toward active centroids and downweight unlikely roles/zones.
- Keep the sampler “alive” with the homeostasis loop: when best energy plateaus for `homeostasis.patience` iterations, mutation and temperature are briefly boosted (`mutation_boost`, `reheat_scale`) to escape local minima and continue chasing higher accuracy.

### Hardware & Safety Configuration

- `hardware_overrides.allow_gpu/allow_distributed` explicitly disable accelerators even when Auto detects them. Defaults honor env vars (`W1Z4RDV1510N_ALLOW_GPU`, `SIMFUTURES_ALLOW_GPU`, etc.).
- `hardware_overrides.max_threads` caps the logical CPU core count the scheduler uses (also configurable via `W1Z4RDV1510N_MAX_THREADS`). Handy when sharing multi-socket machines.
- `W1Z4RDV1510N_THREAD_BUDGET` provides an extra cap for CPU worker pools; the multi-threaded backend chunks particle updates to honor this budget and keep midrange machines responsive.
- Environment hints:
  - `W1Z4RDV1510N_HAS_GPU` / `SIMFUTURES_HAS_GPU` - force-detect a GPU even if CUDA tooling is not present.
  - `W1Z4RDV1510N_DISTRIBUTED` / `SIMFUTURES_DISTRIBUTED` - hint that we're running under a cluster scheduler (Auto may pick the distributed backend).
  - `CUDA_VISIBLE_DEVICES` – honored transparently by the GPU backend.
- Production guardrails (`mode = "PRODUCTION"`) reject jitter RNGs, experimental hardware, or explicit `hardware_backend = "Experimental"`.

### Service Endpoints & Telemetry

- Launch via `W1Z4RDV1510N_SERVICE_ADDR=0.0.0.0:8080 cargo run --bin service`.
- `POST /predict` – same payload as the CLI accepts; response now includes a monotonic `run_id`, backend, and acceptance ratio.
- `GET /healthz` – always returns HTTP 200 with a `HealthReport { status, ready, uptime, total_requests, ... }`.
- `GET /readyz` - returns HTTP 200 only after at least one successful run (otherwise HTTP 503 but with the same JSON payload). Perfect for load balancer readiness checks.
- Structured telemetry records start/end timestamps, backend choice, and best energy per run; logs also include the `run_id` for correlation.
- `GET /runs?limit=20` - lists the most recent persisted runs (best energy, backend, timestamp). `GET /runs/{run_id}` returns the original request + response payload for that run, exactly as they were processed.
- Every POSTed job is persisted under `logs/service_runs/run_<id>.json` by default (override via `W1Z4RDV1510N_SERVICE_STORAGE`). This makes post-mortem analysis and dataset building straightforward.
- Knowledge ingest endpoints (node API): `POST /knowledge/ingest` (JATS XML or document), `GET /knowledge/queue`, and `POST /knowledge/vote`. The node API does not read local asset paths; include image bytes via OCR tooling or keep `require_image_bytes=false`.

## Chess validation loop

The chess pipeline under `scripts/` provides a deterministic, hardware-aware benchmark for proposal diversity and ML-guided reasoning:

1. `preprocess_chess_games.py` ingests PGNs (`data/chess/raw`) into `processed_games.jsonl`.
2. `chess_training_loop.py` hashes move histories + player metadata, builds prefix features for multiple time horizons, and trains:
   - Softmax regressors for outcomes at different ply scopes.
   - Frequency-based move predictors for +1…+20 moves ahead given sliding context windows.
3. The loop runs indefinitely (or until `--max-iterations`), logging accuracy per scope/horizon to `logs/chess_training_metrics.log`.

Recent upgrades:

- Thread-aware initialization: sets `OMP_NUM_THREADS` / MKL / OpenBLAS defaults based on CPU count so NumPy kernels utilize all cores (and play nicely with lightweight devices).
- Incremental histogramming: outcome features reuse cumulative hashed counts instead of repeated `np.bincount`, yielding a large speed-up per iteration.
- Hashed move contexts: move models store compact integer contexts, drastically reducing dictionary pressure while keeping exact move labels for evaluation.
- Motif + relational priors: `build_relational_priors.py` distills motifs/transitions from any sequence dataset; `chess_training_loop.py` can blend them (`--relational-priors`, `--prior-blend`, `--prior-topk`) to bias toward recurring relational structure.
- Role-aware motifs + factorized move priors: motif hashes and move labels now include role/side/zone bins; factorized priors (`--factor-blend`) reduce sparsity without heavy models.
- Clustered/anchor priors + beam re-rank: hashed style clusters and anchor motifs add structure; a shallow beam (`--beam-width`, `--beam-steps`) re-ranks longer horizons cheaply.
- Fully parameterized CLI: `--outcome-scopes`, `--move-horizons`, `--context-window`, `--context-stride`, `--epochs-per-iteration`, `--max-runtime-minutes`, `--summary-file`, `--factor-blend`, `--beam-*`, etc., make it easy to sweep experiments or run a deterministic overnight job.
- Continuous logging: every iteration appends JSON metrics (iteration, duration, per-scope accuracy) so you can tail progress in real time.
- Multi-frame reinforcement: `--multi-frame-windows` samples multiple temporal windows per game, while `--anneal-*` + `--reinforcement-*` hook the annealing engine into the chess loop for positive/negative feedback on simultaneous futures.

Example launch (runs indefinitely, retraining each iteration):

```powershell
python scripts/chess_training_loop.py `
  --max-games 20000 `
  --epochs-per-iteration 5 `
  --outcome-scopes 6 10 14 18 22 26 30 40 `
  --move-horizons 1 5 10 15 20 `
  --context-window 8 `
  --context-stride 2 `
  --log-file logs/chess_training_metrics.log
```

The script automatically seeds NumPy/Python RNGs, prints detected CPU/thread settings, and reports integer accuracies for each horizon so you can gauge convergence quickly.

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
    "run_id": 42,
    "random_provider": { "provider": "OS_ENTROPY", "deterministic": false, "seed": null },
    "hardware_backend": "MultiThreadedCpu",
    "best_energy": -12.3,
    "acceptance_ratio": 0.44
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

- For a deterministic overnight run with relational + role-aware/factorized priors, beam re-rank, automatic summary, and a wall-clock cap:

```powershell
python scripts/chess_training_loop.py `
  --max-games 8000 `
  --max-iterations 0 `
  --max-runtime-minutes 600 `
  --epochs-per-iteration 4 `
  --relational-priors data/relational_priors.json `
  --prior-blend 0.6 `
  --prior-topk 100 `
  --factor-blend 0.6 `
  --beam-width 12 `
  --beam-steps 3 `
  --log-file logs/chess_eval_priors_long.log `
  --summary-file logs/chess_run_summary.txt
```

- Hashed SAN features + player/opening metadata feed a softmax outcome model at multiple observation scopes (6→40 plies) and frequency-based move predictors (+1→+20 moves ahead).
- Each iteration prints integer accuracies and appends a JSON record to `logs/chess_training_metrics.log`, enabling long-term analysis while the process runs unattended.

Fetching chemistry (USPTO) datasets for symbol-sequence tests:

- Use `scripts/fetch_uspto_reactions.py` to download both the full Lowe USPTO set and the curated USPTO-50K subset into `data/uspto/`:

```powershell
python scripts/fetch_uspto_reactions.py             # full + 50k
python scripts/fetch_uspto_reactions.py --only 50k  # just the 50k subset
```

The script first tries the Figshare mirrors (50k: `25325623`, full: `25242010`) and falls back through a list of public mirrors; downloads can be slow, so you can also drop `uspto_50k.zip` into `data/uspto/` manually. Convert the extracted reaction data into your symbol schema (roles/groups/steps) before running priors/training on them.
- To drive the Rust annealer on real games, run `scripts/export_chess_stack_snapshot.py --game-index 3 --plies 14 --output data/chess/stack_snapshot.json` to emit an `EnvironmentSnapshot` with a `stack_history` of early plies. Pair it with `energy.w_stack_hash` and `quantum.enabled=true` in your config.
- Use `--stride` / `--stride-offset` / `--skip-terminal` on `export_chess_stack_snapshot.py` to intentionally sparsify the history (e.g., keep every 3rd ply) when testing the gap-filling stack alignment / future-lookahead terms.

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
