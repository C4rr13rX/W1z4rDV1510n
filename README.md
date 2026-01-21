# W1z4rDV1510n

**Ultradian Rhythm Dynamic Stability Analysis System** - flag-driven mode for layered ultradian phase/amplitude/coherence tracking, stability baselines, and cross-entity motif discovery with human-first governance.

W1z4rDV1510n is a Rust-first, CPU-first intelligence stack that combines:
- Symbol matrix inference (annealer + neural priors).
- Streaming ultradian analysis (people, crowd/traffic, public topics).
- A distributed node runtime with P2P mesh, ledger incentives, and multi-chain deposits.

Designed to run on modest desktops and scale across many nodes.

---

## Core innovations

- CPU+RAM neural fabric: neuron pools, neurogenesis, and spiking signal routing without GPUs.
- Multimodal stream conversion into layered dynamics and discrete event tokens, aligned with temporal tolerance and confidence gating.
- Species-agnostic behavior substrate with body-schema adapters, time-frequency motifs, DTW/soft-DTW similarity, and MDL-based compression.
- Multi-domain hypergraph + temporal inference with phase/amplitude prediction, cross-layer coherence, event intensities, and Dirichlet evidential uncertainty.
- Online plasticity with surprise-weighted replay, trust-region updates, rollback, and horizon growth controls.
- Causal discovery with time-lag edges and counterfactual branching futures with retrodiction.
- Knowledge ingestion for JATS/NLM + figure association queues, plus human-work reward scaffolding.
- P2P data mesh with replication receipts, storage rewards, peer scoring, and offload decisions.
- Opt-in identity verification bound to wallets using behavior-derived challenges (no face ID, no PII).

---

## What it does (full stack)

### 1) Symbol matrix inference engine
- Builds population-based state proposals with energy models, MCMC-style acceptance, and annealing schedules.
- Supports classical and quantum-inspired annealing; calibration hooks align simulator parameters with remote quantum runs.
- Integrates neural priors and temporal motif signals into proposal scoring.
- Runs as CLI (`predict_state`) or REST service (`service`) for batch jobs.

### 2) Streaming ultradian analysis pipeline
- Ingests three stream types:
  - People video or pose frames (bodycam/CCTV) -> motor features + behavioral atoms.
  - Crowd/traffic signals -> flow layers (density, velocity, directionality, stop-go waves, motifs, daily/weekly cycles).
  - Public topic streams -> event layers (burst/decay/excitation/lead-lag/periodicity).
- Extracts ultradian micro-arousal, BRAC, and meso layers as phase/amplitude/coherence.
- Aligns across modalities with tolerance windows and per-source confidence gating.
- Emits tokens (Behavioral Atoms/Tokens, Crowd/Traffic Tokens, Topic Event Tokens) and layer states.
- Optional OCR adapter enriches video frames with text blocks that become TextAnnotation tokens.
- Visual label queue emits bbox/region tasks so humans can annotate regions and wire labels into the neural fabric.

### 3) Behavior substrate and motifs
- Body-schema adapters map multimodal sensors into a shared latent state + action vector.
- Change-point segmentation plus fixed windows for stable coverage.
- Motif discovery using DTW/soft-DTW similarity, graph signatures, and MDL costs.
- Behavior graph coupling metrics: proximity, coherence, phase-locking, transfer-entropy proxies.
- Attractor detection via next-state entropy with optional constraints and soft objectives.

### 4) Multi-domain fusion and temporal inference
- Learned multi-domain hypergraph links tokens and layers with decay, TTL, and gating.
- Temporal inference predicts phase/amplitude/coherence drifts, cross-layer coherence, and next-event intensities.
- Event intensities include mutual excitation across domains.
- Evidential outputs use Dirichlet posteriors for event and regime uncertainty.

### 5) Causal graph and branching futures
- Streaming causal graph updates with time-lag edges and intervention deltas.
- Counterfactual do()-style interventions inform branch payloads.
- Branching futures include confidence/uncertainty and retrodicted missed events.

### 6) Online learning, consistency chunking, and ontology
- Surprise-weighted replay reservoir with trust-region updates and rollback protection.
- Horizon manager only expands when calibration improves.
- Consistency chunking builds reusable templates (codebook) from stable motifs.
- Ontology runtime versions labels across minute/hour/day/week windows.

### 7) Knowledge ingestion and labeling queues
- JATS/NLM ingestion with text blocks, figure assets, and OCR hooks.
- Figure-to-text association tasks with voting and confidence thresholds.
- Label queue for emergent dimensions and novel token/layer attributes.

### 8) Health, survival, and overlays
- Physiology template bank with covariance-aware deviation scoring.
- Survival metrics from behavior graphs: cooperation, conflict, play, intent magnitude.
- Health overlay output with per-entity palette and dimension scores.

### 9) Network-wide neural fabric
- Shares motifs, transitions, and network pattern summaries across nodes.
- Entity threads track phenotype tokens, behavior signatures, and plausible travel-time continuity.
- Patterns propagate through data mesh to keep nodes aligned.

### 10) Node stack and incentives
- P2P networking (libp2p gossipsub + Kademlia + mDNS) with rate limits and peer scoring.
- Data mesh: manifests, chunking, replication receipts, integrity audits, and repair requests.
- Local ledger: validator heartbeats, fee-market scaffolding, audit chain, and reward events.
- Multi-chain bridge: intent tracking + relayer-quorum proof verification (other modes are stubbed).
- Encrypted wallet for node identity and rewards.
- Large-scale simulation with hardware-aware caps to avoid overload.

### 11) Opt-in identity verification
- Behavior-derived challenges (position + motion signature + code) bound to wallets on-chain.
- Supports re-issuing bindings to a new wallet via API.
- No face ID, no biometric identity resolution.

### 12) Compute and hardware awareness
- Auto-detects CPU/GPU/cluster hints to select optimized backends.
- CPU RAM-optimized backend with NUMA/large-page hints.
- Optional GPU backend (build with `--features gpu`) for bulk bit operations.
- Distributed backend is local-only (thread segmentation) today; remote workers are planned.
- Quantum endpoints via HTTP executor with auth headers and timeout control.

---

## Governance and safety defaults

- Governance config disables face ID and PII by default and enforces public-only ingestion at the config level.
- Immutable audit logging in the ledger for ingest, rewards, and bridge intents.
- Explainability is on by default via rich report metadata (temporal, causal, branching, physiology, ontology).
- Federated learning and DP flags exist in config and are intended for future adapters.

---

## Status and maturity

- OpenStack integration is a local control-plane stub used for node planning; no external OpenStack API is required.
- Bridge verification currently supports relayer quorum; optimistic, light-client, and ZK modes are placeholders.
- GPU and experimental hardware backends require build features (`gpu`, `experimental-hw`).

---

## Configuration highlights

- Node config: `node_config.json` (generated) or `node_config_example.json` (reference).
- Core run config: `run_config*.json` (annealer + streaming).

Key flags:
- `streaming.enabled` + `streaming.ultradian_node` enable ultradian analysis in the node.
- `streaming.layer_flags.*` gate ultradian, behavior, physiology, and related layers in the run config.
- `data.host_storage` and `workload.enable_storage` control storage participation.
- `peer_scoring.enabled` enables accuracy/efficiency sharing and offload decisions.
- `identity.enabled` enables behavior-based identity verification (ledger required).
- `compute.allow_gpu` and `compute.allow_quantum` gate optional backends.

---

## Quickstart (node)

1. Initialize config + wallet:

```powershell
cargo run --bin w1z4rdv1510n-node -- init
```

2. Run the node:

```powershell
cargo run --bin w1z4rdv1510n-node
```

3. Start the node API:

```powershell
cargo run --bin w1z4rdv1510n-node -- api --addr 127.0.0.1:8090
```

4. Inspect local label queues (requires data mesh storage on the node):

```powershell
cargo run --bin w1z4rdv1510n-node -- label-queue --limit 25
cargo run --bin w1z4rdv1510n-node -- visual-label-queue --limit 25
```

5. Simulate a network:

```powershell
cargo run --bin w1z4rdv1510n-node -- sim --nodes 10000 --ticks 100
```

---

## Quickstart (streaming service)

Run the streaming loop on JSONL envelopes:

```powershell
cargo run --bin streaming_service -- --config run_config.json --input stream.jsonl
```

Pipe RTSP/video pose output into the service:

```powershell
python scripts/rtsp_pose_bridge.py --source rtsp://camera --emit-image ref --image-dir data/frames | cargo run --bin streaming_service -- --config run_config_streaming_ocr.json --input -
```

Enable OCR with a ready config:

```powershell
cargo run --bin streaming_service -- --config run_config_streaming_ocr.json --input stream.jsonl
```

---

## Quickstart (annealer)

Run the CLI:

```powershell
cargo run --bin predict_state -- --config run_config.json
```

Run the REST service:

```powershell
W1Z4RDV1510N_SERVICE_ADDR=0.0.0.0:8080 cargo run --bin service
```

---

## API highlights

Node API (auth optional, see `node_config_example.json`):
- /health, /ready, /metrics
- /data/ingest, /data/:data_id
- /knowledge/ingest, /knowledge/queue, /knowledge/vote
- /streaming/labels, /streaming/visual-labels
- /bridge/chains, /bridge/intent, /bridge/proof, /balance/:node_id
- /identity/challenge, /identity/verify, /identity/:thread_id

Service API:
- POST /predict
- GET /healthz, /readyz
- GET /runs, /runs/{run_id}

---

## Tools and utilities

- `scripts/rtsp_pose_bridge.py` - RTSP/video to pose JSONL bridge.
- `run_config_streaming_ocr.json` - streaming profile with OCR enabled (Tesseract command).
- `scripts/live_viz_server.py` + `scripts/run_with_viz.py` - local visualization loop.
- Dataset fetchers/preprocessors (USPTO, Tox21, ETH/UCY, exoplanets, genomics, OWID).
- `calibrate_energy` - tune energy weights from trajectories.
- `scripts/prepare_textbook_qa_dataset.py` - render textbook PDFs into page images + OCR text, emit review queues, and build QA candidate datasets (requires `pdftoppm` or `mutool`, plus `pdftotext` or `tesseract`).

Example (auto-detects `../StateOfLoci/textbooks` if present):

```powershell
python scripts/prepare_textbook_qa_dataset.py --max-books 3 --max-pages 25 --skip-existing
```

---

## Repository layout

- `crates/core` - annealer, streaming inference, neuro/spike runtimes, math toolbox.
- `crates/node` - P2P, data mesh, ledger, wallet, API, identity, simulation.
- `crates/experimental-hw` - experimental hardware backends.
- `chain/` - genesis + reward/bridge/token specs.
- `schemas/` - JSON schemas for bridge intents.
- `scripts/` - data ingest + visualization utilities.

---

## Build and test

```powershell
cargo fmt
cargo test
```
