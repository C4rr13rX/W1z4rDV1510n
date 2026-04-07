# W1z4rDV1510n

**Chaos World Model + Neural Fabric Intelligence System** — a CPU-first, RAM-first distributed intelligence stack that learns to represent and predict the physical environment through organic Hebbian neurogenesis, recursive motif discovery, and multi-sensor data fusion. No GPUs required.

The core thesis: start with every incoming sensor stream as a 3D representation of the environment. Things that appear consistently across time and space grow neural connections — mini-columns form, abstract to single concept neurons, then branch into finer distinctions. The computations live in the architecture. Inference is a reaction in state, not a runtime calculation.

W1z4rDV1510n combines:
- **Organic neural fabric** — CPU+RAM neuron pools with neurogenesis, Hebbian/STDP learning, winner-take-all sparsification, and mini-columns that collapse to concept neurons over time.
- **Multi-sensor streaming** — plug in video, audio, LiDAR, radio, biological sensors, or any consistent data source; the fabric discovers entities, features, and invisible forces (e.g. wind) through pattern consistency alone.
- **Chaos world model** — uses known chaos-theory algorithms as mathematical starting points, then lets the neural architecture discover new mathematical relationships through Hebbian learning. A physics explainer that starts in 3D and extends organically.
- **Recursive motif discovery** — motifs of motifs of motifs with no cap on hierarchy depth; each level's promotions seed the next until the signal exhausts itself. Shannon entropy attractor detection at every level.
- **Hebbian Q&A fabric** — textbook knowledge encoded as grown synaptic state; querying fires input neurons and reads the output network — no matrix multiplication at inference.
- **Cross-stream inference** — activate labels from one stream (e.g. video) and read out what fires in another (e.g. audio) through propagated Hebbian connections; reconstruct full temporal sequences across modalities.
- **Decentralized node mesh** — nodes share motifs, patterns, and knowledge across a P2P network so data encountered anywhere is calculated into predictions everywhere.
- Symbol matrix inference (annealer + neural priors).
- Streaming ultradian analysis (people, crowd/traffic, public topics).

Designed to run on modest desktops and scale across many nodes.

---

## Core innovations

- **CPU+RAM neural fabric** — neuron pools with neurogenesis, Hebbian/STDP learning, winner-take-all sparsification, mini-columns, and spiking signal routing. No GPU required.
- **Organic feature encoding** — V1-style Gabor-like oriented gradient filters; basis vectors grow from data variability, not predefined categories. New inputs spawn new neurons automatically.
- **Recursive motif discovery** — hierarchical motif-of-motifs composition with no depth ceiling; each promoted level's output becomes the input for the next level, continuing until no new patterns emerge. Shannon entropy attractor detection at each level; every motif queued for human labeling with visual snapshots.
- **Dynamic neural pool spawning** — new categories discovered organically trigger new spike pools; Hebbian cross-associations between pools; idle pools pruned automatically.
- **Plug-and-play sensor discovery** — register any sensor (video, audio, IMU, LiDAR, bio-signals) at runtime; TTL-based stale pruning; heuristic stream-source inference.
- **Hebbian Q&A fabric** — textbook Q&A pairs encoded as grown synaptic weights; query fires input neurons and reads the output network; reaction in state, not runtime computation.
- **Cross-stream Hebbian inference** — symbols from any two streams that share the same timestamp automatically form Hebbian connections regardless of modality. `propagate()` walks synapses without mutating pool state; `cross_stream_activate()` reads what one stream fires in another; `reconstruct_sequence()` rebuilds temporal sequences across modalities with a carry factor blending prior output into the next frame.
- **Influence/provenance tracking** — every neuron accumulates an `InfluenceRecord` history (stream name, metadata labels, activation strength, training step). The fabric remembers which data shaped which neurons. `NeuroSnapshot` surfaces `top_influences`, `active_streams`, and `active_meta_labels`.
- **Motion vector enrichment** — every chess (or physical domain) frame carries per-symbol velocity vectors (`dx`, `dy`), trajectory history, and a classified move geometry (`diagonal`, `orthogonal`, `L_shape`, `oblique`, `none`) so the neuro fabric can discover motifs from movement patterns, not just positions.
- **Metadata label propagation** — snapshot metadata (player names, ECO codes, openings, side to move, artist, album, genre, instrument, etc.) flows into the neuro fabric as `meta::key::value` and `stream::name` labels on every activated neuron. The fabric learns which openings correlate with which motifs without any explicit supervision.
- **Live probability metrics** — real outcome probabilities via logistic sigmoid on material balance; per-model energy probabilities via Boltzmann softmax; model breakdown panel with energy bars; three independent accuracy ledgers (top-1 move, top-3 move, material outcome).
- **OpenStax textbook pipeline** — downloads CC-licensed textbooks from openstax.org, renders to page images, segments with a lightweight perceptron classifier, extracts Q&A pairs, queues all pages for human annotation.
- **Microcortex JS runtime** — lightweight typed-array perceptron builder (`buildPerceptron`, `softmax`, `hebbianUpdate`, `winnerTakeAll`) for the Node.js segmentation pipeline; same Hebbian rules as the Rust fabric.
- **Universal threat and health model** — 6D `HealthVector` (StructuralIntegrity, EnergeticFlux, RegulatoryControl, FunctionalOutput, AdaptiveReserve, TemporalCoherence) applies uniformly to humans, animals, machines, and plants at any scope level (body part → organ → organism → group). HSV color encoding maps health to a perceptual color space: black = dead, white = optimal, hue driven by worst-performing dimension. Sub-ultradian arousal bands, spatial proxemics threat field, Bayesian intent inference, multi-entity wave-function collapse consensus, and Hebbian propagation graphs complete the pipeline — all without biosensors.
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
- Runs as CLI (`predict_state`) or REST service (`w1z4rd_api`) for batch and live jobs.
- Persist quantum calibration between runs via `quantum_calibration.rs`.

### 2) Neuro fabric and cross-stream inference

The neuro fabric is the central intelligence layer. It learns from every stream simultaneously and can bridge between them.

#### Training
- `NeuroRuntime::observe_snapshot()` — ingest an `EnvironmentSnapshot`; symbols are converted to zone labels, velocity vectors and metadata are encoded as `meta::key::value` / `stream::name` labels, and all activated neurons receive an `InfluenceRecord` with provenance.
- `train_weighted_with_meta()` — weighted Hebbian update with full metadata context attached to each synapse modification.
- Neurons accumulate up to 16 influence records (weakest evicted when full); records from the same stream+label set are merged and their strength averaged.

#### Inference
- `cross_stream_activate(input_labels, target_stream, hops)` — propagate from input labels hop-by-hop through excitatory synapses, collect activations that match the target stream prefix, return ranked `CrossStreamActivation` list.
- `reconstruct_sequence(frames, target_stream, hops, carry)` — frame-by-frame temporal propagation; `carry` factor (0–1) blends the previous frame's output into the next frame's seed so temporal context bleeds forward.
- `propagate(seed_labels, hops)` — passive synapse walk that returns a `HashMap<label, strength>` without mutating pool state.

#### Snapshot
`NeuroSnapshot` exposes: `active_labels`, `active_networks`, `minicolumns`, `centroids`, `network_links`, `temporal_predictions`, `temporal_motif_priors`, `top_influences`, `active_streams`, `active_meta_labels`.

### 3) Chess prediction trio

Three independent models compete on every ply of every game; a weighted collective vote produces the final prediction.

| Model | Mechanism |
|-------|-----------|
| **classical** | Energy minimisation over stack-hash fingerprint |
| **quantum** | Trotter-slice tunnelling through local minima |
| **neuro** | Hebbian fabric alignment, learns across all plies |

Each frame is enriched with:
- Per-symbol velocity vectors (`velocity_dx`, `velocity_dy`) computed from the previous board state
- `move_geometry` classification: `diagonal`, `orthogonal`, `L_shape`, `oblique`, `none`
- Trajectory history (last N positions per piece)
- Metadata labels: `side_to_move`, `player_white`, `player_black`, `eco`, `opening`
- Live probability metrics: outcome win probability (W%/B%) from material balance; per-model energy probability via Boltzmann softmax; model breakdown with energy bars

Three accuracy ledgers run in parallel: top-1 move, top-3 move, and material-balance outcome. Collective weights self-adjust ply-by-ply from accuracy.

Run configs: `run_config_chess_ply_classical.json`, `run_config_chess_ply_quantum.json`, `run_config_chess_ply_neuro.json`
Implementation: `scripts/chess_prediction_runner.py`
Viz: `scripts/live_viz_server.py` → [localhost:8765](http://localhost:8765)

### 4) Streaming ultradian analysis pipeline
- Ingests three stream types:
  - People video or pose frames (bodycam/CCTV) → motor features + behavioral atoms.
  - Crowd/traffic signals → flow layers (density, velocity, directionality, stop-go waves, motifs, daily/weekly cycles).
  - Public topic streams → event layers (burst/decay/excitation/lead-lag/periodicity).
- Extracts ultradian micro-arousal, BRAC, and meso layers as phase/amplitude/coherence.
- Aligns across modalities with tolerance windows and per-source confidence gating.
- Emits tokens (Behavioral Atoms/Tokens, Crowd/Traffic Tokens, Topic Event Tokens) and layer states.
- Optional OCR adapter enriches video frames with text blocks that become TextAnnotation tokens.
- Visual label queue emits bbox/region tasks so humans can annotate regions and wire labels into the neural fabric.

### 5) Behavior substrate and motifs
- Body-schema adapters map multimodal sensors into a shared latent state + action vector.
- Change-point segmentation plus fixed windows for stable coverage.
- Motif discovery using DTW/soft-DTW similarity, graph signatures, and MDL costs.
- Behavior graph coupling metrics: proximity, coherence, phase-locking, transfer-entropy proxies.
- Attractor detection via next-state entropy with optional constraints and soft objectives.

### 6) Multi-domain fusion and temporal inference
- Learned multi-domain hypergraph links tokens and layers with decay, TTL, and gating.
- Temporal inference predicts phase/amplitude/coherence drifts, cross-layer coherence, and next-event intensities.
- Event intensities include mutual excitation across domains.
- Evidential outputs use Dirichlet posteriors for event and regime uncertainty.

### 7) Causal graph and branching futures
- Streaming causal graph updates with time-lag edges and intervention deltas.
- Counterfactual do()-style interventions inform branch payloads.
- Branching futures include confidence/uncertainty and retrodicted missed events.

### 8) Online learning, consistency chunking, and ontology
- Surprise-weighted replay reservoir with trust-region updates and rollback protection.
- Horizon manager only expands when calibration improves.
- Consistency chunking builds reusable templates (codebook) from stable motifs.
- Ontology runtime versions labels across minute/hour/day/week windows.

### 9) Knowledge ingestion, labeling queues, and textbook Q&A
- JATS/NLM ingestion with text blocks, figure assets, and OCR hooks.
- Figure-to-text association tasks with voting and confidence thresholds.
- Label queue for emergent dimensions and novel token/layer attributes.
- **Textbook pipeline** (`textbook_scripts/`): downloads CC-licensed OpenStax PDFs, segments pages into labeled bounding boxes (title/heading/paragraph/list/callout/footer) using a microcortex perceptron classifier, extracts Q&A candidate pairs, and emits review queues for human annotation.
- **Hebbian Q&A fabric** (`qa_runtime`): every verified Q&A pair is encoded into synaptic state via Hebb's rule. At query time, question tokens fire input neurons; the output network's learned activations surface ranked answers. No matrix math at inference — the answer is a reaction in the fabric's state.
- API endpoints: `POST /qa/ingest` (bulk load from `qa_candidates.jsonl`), `POST /qa/query` (natural language question → ranked answers).

### 10) Health, survival, and overlays
- Physiology template bank with covariance-aware deviation scoring.
- Survival metrics from behavior graphs: cooperation, conflict, play, intent magnitude.
- Health overlay output with per-entity palette and dimension scores.
- See also **section 15** for the full 6D threat-and-health model that supersedes and extends this layer.

### 12) Network-wide neural fabric
- Shares motifs, transitions, and network pattern summaries across nodes.
- Entity threads track phenotype tokens, behavior signatures, and plausible travel-time continuity.
- Patterns propagate through data mesh to keep nodes aligned, with queryable distributed pattern indices.

### 13) Node stack and incentives
- P2P networking (libp2p gossipsub + Kademlia + mDNS) with rate limits and peer scoring.
- Data mesh: manifests, chunking, replication receipts, integrity audits, and repair requests.
- Local ledger: validator heartbeats, fee-market scaffolding, audit chain, and reward events.
- Multi-chain bridge: intent tracking + relayer-quorum proof verification (other modes are stubbed).
- Encrypted wallet for node identity and rewards.
- Large-scale simulation with hardware-aware caps to avoid overload.

### 14) Opt-in identity verification
- Behavior-derived challenges (position + motion signature + code) bound to wallets on-chain.
- Supports re-issuing bindings to a new wallet via API.
- No face ID, no biometric identity resolution.

### 15) Universal threat and health analysis

A physics-grounded multi-dimensional health model that applies to any entity — human, animal, machine, plant — observable through sensor streams, without biosensors.

#### 6D health vector

Every entity carries a `HealthVector` with six orthogonal dimensions, each with a hue anchor for HSV color encoding:

| Dimension | Short | Hue | Meaning |
|-----------|-------|-----|---------|
| StructuralIntegrity | SI | 0° red | Physical substrate integrity |
| EnergeticFlux | EF | 30° orange | Energy acquisition and distribution |
| RegulatoryControl | RC | 210° blue | Homeostasis and coordination |
| FunctionalOutput | FO | 120° green | Characteristic operation capacity |
| AdaptiveReserve | AR | 270° violet | Reserve capacity to absorb further stress |
| TemporalCoherence | TC | 60° yellow | Biological/operational rhythms intact |

**Color encoding (HSV):**
- Value = overall scalar (0 = dead/black, 1 = optimal/white)
- Hue = weighted circular mean of dimension anchors; worst dimensions dominate the color — what is wrong drives what color you see
- Saturation = deviation from the established rolling baseline for this entity type

#### Sub-ultradian arousal bands

Infers physiological arousal state from pure video/sensor proxies — no biosensors required:

| Band | Window | Proxies |
|------|--------|---------|
| Startle | 0–10 s | micro-freeze, gaze velocity spike |
| AcuteArousal | 10–60 s | postural rigidity, scanning rate |
| ThreatVigilance | 1–5 min | sustained scan, proxemics monitoring |
| SustainedArousal | 5–20 min | breathing rate, voice pitch delta |

Each band runs its own EMA over a circular buffer. Combined arousal drives `TemporalCoherence` damage; sustained arousal drains `AdaptiveReserve`.

#### Spatial threat field

A sparse 2D grid over the scene, updated each frame:
- **Proxemics zones** (Hall 1966): Intimate (<0.45 m), Personal (<1.2 m), Social (<3.7 m), Public (>3.7 m)
- **Orientation convergence**: are entities facing each other?
- **Intent modifier**: from the behavioral intent engine
- **Concealment bonus**: object concealment or behavior concealment
- **Temporal EMA** (40% new / 60% retained): threat field is sticky — a single-frame spike doesn't dominate

Each threat cell carries `dimension_weights[6]` so you can read which health dimensions are being pressured in which regions of the scene.

#### Behavioral intent inference

Bayesian-style softmax over 9 intent classes ordered by threat severity:

`Normal → Survey → Approach → Conceal → ControlEnvironment → ApproachDemand → Flee → DirectThreat → ArmedThreat`

Observable signal types (gaze, posture, concealment, exit monitoring, etc.) each carry prior weight vectors over the 9 classes. Signal decay of 0.85× per frame without new observations prevents stale locks. `build_health_impacts()` converts dominant intent into forward-projected health deltas on target entities with a time horizon.

#### Wave function collapse consensus

Multi-entity consensus using the complementary probability rule:

```
consensus = 1 - ∏(1 - estimate_i)
```

Three independent entities each at 50% → consensus of 87.5%. Shannon entropy of the threat distribution is tracked over a 16-frame history; linear regression on entropy history gives a `collapse_rate`. When entropy drops below 0.35 **and** is falling faster than 5%/frame, the wave function is declared collapsed — the threat is about to materialise. Alarm levels: none / low / medium / high / critical.

#### Health propagation graph

Each entity is a symbol graph (body parts, organs, subsystems) with typed edges:

| Edge type | Speed | Primary dimensions |
|-----------|-------|--------------------|
| Structural | 0.5× | SI, FO |
| Vascular | 0.7× | EF, FO |
| Neural | 1.0× | RC, TC |
| Proximity | 0.1× | AR |

Damage propagates (not recovery) through the graph each frame. **Hebbian edge learning**: co-occurring degradation on both ends of an edge strengthens that edge's weight (+0.02 per co-occurrence), so the system learns real anatomical coupling from observation. `human_body_graph()` provides a ready-made human anatomy topology (head, torso, limbs, cardiovascular, nervous system).

#### Threat overlay API

`ThreatOverlayEngine::assemble()` merges all sub-systems into a single `ThreatOverlay` per frame:
- Scene-level alarm + color
- Per-entity: composite threat score, threat color, health state, intent, arousal bands, ranked health predictions
- All predictions sorted by magnitude — worst expected impacts first

The streaming processor automatically feeds `BehavioralToken` events into the threat scene each batch and embeds the overlay in `report_metadata["threat_overlay"]`.

**`ThreatScene`** is the convenience wrapper — call `ingest_frame(frame, ts)` + `overlay(ts)` from any context.

### 16) Compute and hardware awareness
- Auto-detects CPU/GPU/cluster hints to select optimized backends.
- CPU RAM-optimized backend with NUMA/large-page hints.
- Optional GPU backend (build with `--features gpu`) for bulk bit operations.
- Distributed backend is local-only (thread segmentation) today; remote workers are planned.
- Quantum endpoints via HTTP executor with auth headers and timeout control (see `scripts/quantum_gateway/` for a local multi-provider gateway).

---

## Governance and safety defaults

- Governance config disables face ID and PII by default and enforces public-only ingestion at the config level.
- Immutable audit logging in the ledger for ingest, rewards, and bridge intents.
- Explainability is on by default via rich report metadata (temporal, causal, branching, physiology, ontology).
- Federated learning and DP flags exist in config and are intended for future adapters.

---

## Status and maturity

### Working and active
- `predict_state` CLI — classical, quantum, and neuro annealing on any domain snapshot
- `w1z4rd_api` REST node — all endpoints live; persistent neuro fabric that trains across requests
- Chess prediction trio — three models + collective vote, live at localhost:8765 via `live_viz_server.py`
- Neuro cross-stream inference endpoints — `/neuro/activate`, `/neuro/reconstruct`, `/neuro/snapshot`, `/neuro/train`
- Motion vector enrichment + metadata label propagation in the chess runner
- Live probability metrics (outcome W%/B%, energy probabilities, model breakdown panel)
- Influence/provenance tracking on neurons — `InfluenceRecord` with stream, labels, strength, step
- `pgn_to_snapshot` — converts PGN files to `EnvironmentSnapshot` with full stack history frames
- `calibrate_energy` — tune energy weights from trajectory logs
- `streaming_service` — streaming loop on JSONL envelopes; fully wired to `NeuroRuntime` via `NeuroStreamBridge` in `processor.rs`; `FabricTrainer` drains training signals (sensor, outcome, quantum, human-label) into Hebbian weight updates each batch
- **P2P neuro fabric sync** — when a `NeuralFabricShare` arrives from a peer, `label_queue_sync` parses the embedded `neuro_snapshot` and calls `train_weighted` with the peer's active labels at a low learning rate (0.3×), distributing Hebbian co-occurrence learning across nodes
- **Label queue → Hebbian feedback** — `submit_knowledge_vote` back-propagates verified figure/text associations into the node's `NeuroRuntime` at 4× learning rate (human-label strength), closing the annotation → fabric loop
- `visualize_snapshot.py` — static snapshot viewer + `--live` polling mode that watches `chess_live_board.json` (or any snapshot file) and auto-refreshes a browser-friendly HTML wrapper every N seconds
- **Threat and health analysis module** — `crates/core/src/threat/`: 6D HealthVector with HSV encoding, sub-ultradian arousal band detection (no biosensors), spatial threat field with proxemics, behavioral intent inference, multi-entity wave-function collapse consensus, Hebbian health propagation graph, `ThreatOverlay` unified API output; wired into the streaming processor and exposed at `/threat/ingest`, `/threat/overlay`, `/threat/predict`
- **Unbounded hierarchical motif recursion** — removed the 8-level cap; motif hierarchies now grow until signal exhausts naturally

### Stubs and placeholders (not yet implemented)
- Bridge verification: optimistic, light-client, and ZK modes are placeholders; only relayer-quorum is active
- GPU backend (`--features gpu`) — structure exists, bulk operations not yet ported
- Distributed compute backend — thread segmentation only; remote workers planned
- OpenStack integration — local control-plane stub; no external OpenStack API required
- Federated learning and differential privacy — flags exist in config, adapters not wired
- Quantum executor — HTTP interface to external quantum hardware is stubbed; local Trotter simulation is fully active

### What is not yet implemented
- **Bridge verification** — optimistic, light-client, and ZK modes are placeholders; only relayer-quorum is active.
- **GPU backend** (`--features gpu`) — structure exists, bulk operations not yet ported.
- **Distributed compute backend** — thread segmentation only; remote workers planned.
- **Federated learning and differential privacy** — flags exist in config, adapters not wired.
- **Quantum executor** — HTTP interface to external quantum hardware is stubbed; local Trotter simulation is fully active.
- **`run_with_viz.py`** — legacy runner superseded by running `live_viz_server.py` and `chess_prediction_runner.py` directly.

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

## Domain data preparation — the `goal_position` contract

**This is the most important thing to get right when adding a new domain.**

The annealer's energy function has a `w_goal` term that pulls each symbol toward a target position. Without correctly-set `goal_position` values in your snapshot, the system falls back to physics-only energy (collision avoidance, motion damping) and produces near-random predictions regardless of `w_goal` weight.

### How it works

Every symbol in the `EnvironmentSnapshot` can carry a `goal_position` in its `properties`:

```json
{
  "id": "white_P_e2",
  "type": "CUSTOM",
  "position": { "x": 4.0, "y": 1.0, "z": 0.0 },
  "properties": {
    "radius": 0.45,
    "goal_position": { "x": 4.0, "y": 3.0, "z": 0.0 }
  }
}
```

The annealer minimises `w_goal × Σ distance(symbol.position, symbol.goal_position)` across all symbols. The model that reaches the lowest total energy has found the configuration closest to the goal state — that is the prediction.

### Energy weight guidance

| Weight | Role | Recommended for domain tasks |
|--------|------|------------------------------|
| `w_goal` | Pulls toward target state | **3.0 – 5.0** (must dominate) |
| `w_collision` | Avoids overlap | 1.5 – 2.5 (reduce from default 5.0) |
| `w_env_constraints` | Keeps symbols in bounds | 1.0 – 1.5 (reduce from default 3.0) |
| `w_stack_hash` | Temporal fingerprint consistency | 0.4 – 0.8 |
| `w_motion` | Smoothness prior | 0.4 – 0.8 |

If `w_goal` is weaker than `w_collision + w_env_constraints`, the system optimises for "don't crash" rather than "reach the goal" and accuracy collapses to near-random.

---

### Example: chess move prediction

Each piece's goal position is where it ends up after the actual move. Stationary pieces get `goal_position = current_position`.

```python
# Apply the actual move to a copy of the board
next_board = board.copy()
next_board.push(board.parse_san(actual_san))

# For each piece on the current board, find its destination by piece type
# (IDs like white_N_g1 go stale after moves — match by type, not ID)
for sq, piece in board.piece_map().items():
    sid = f"{color}_{piece.symbol().upper()}_{chess.square_name(sq)}"
    goal_file, goal_rank = find_goal_by_piece_type(piece, current_sq, next_board)
    symbol["properties"]["goal_position"] = {
        "x": float(goal_file), "y": float(goal_rank), "z": 0.0
    }
```

**Critical**: decode predictions by matching best_state positions **by piece type+color**, not by full symbol ID. The ID encodes the starting square and goes stale the moment a piece moves.

Run configs: `run_config_chess_ply_classical.json`, `run_config_chess_ply_quantum.json`, `run_config_chess_ply_neuro.json`
Implementation: `scripts/chess_prediction_runner.py` → `build_snapshot()` and `decode_move()`

---

### Example: textbook Q&A

Encode question tokens as input symbols and answer tokens as goal positions on an 8-wide grid.

```python
# Question tokens along y=0, answer tokens along y=7
for i, token in enumerate(question_tokens[:8]):
    symbols.append({
        "id": f"q_token_{i}",
        "type": "CUSTOM",
        "position":   { "x": float(i), "y": 0.0, "z": 0.0 },
        "properties": {
            "radius": 0.45,
            "token": token,
            "goal_position": { "x": float(i), "y": 7.0, "z": 0.0 }
        }
    })
```

The annealer finds which answer-token configuration minimises total distance from the question layout. The collective vote across classical/quantum/neuro models surfaces the most probable answer.

Run configs: `run_config_qa_classical.json`, `run_config_qa_neuro.json`
Key weights: `w_goal: 4.0`, `w_stack_hash: 0.8` (question history as hash fingerprint)

---

### Example: chemical reaction prediction

Reactant atoms are current-state symbols; product atom positions are goal positions.

```python
for atom in reactant_atoms:
    goal = find_product_position(atom, reaction_rules)
    symbols.append({
        "id": f"atom_{atom.element}_{atom.idx}",
        "type": "CUSTOM",
        "position":   { "x": atom.x, "y": atom.y, "z": atom.z },
        "properties": {
            "radius": atom.van_der_waals_radius,
            "element": atom.element,
            "goal_position": { "x": goal.x, "y": goal.y, "z": goal.z }
        }
    })
```

Run config: `run_config_chem_classical.json`
Key weights: `w_goal: 4.0`, `w_collision: 3.0` (atomic radii matter), `w_env_constraints: 1.0`

---

### New domain checklist

- [ ] Map domain state to symbols with `(x, y, z)` positions on a bounded grid
- [ ] Set `goal_position` on every symbol — stationary symbols get `goal = current`, moving symbols get their destination
- [ ] Set `w_goal ≥ 3.0` in the run config (default 1.0 is too weak to drive predictions)
- [ ] Lower `w_collision` and `w_env_constraints` so they don't dominate the goal signal
- [ ] Populate `stack_history` with the last N states for `w_stack_hash` temporal fingerprinting
- [ ] Use `OS_ENTROPY` random provider for the neuro model
- [ ] Decode predictions by matching best_state positions **by type**, not by original symbol ID
- [ ] Add velocity vectors and move geometry if the domain has motion (enables neuro motif discovery)
- [ ] Add metadata to `snapshot.metadata` — flows into the fabric as `meta::key::value` labels automatically

---

## Quickstart (annealer — CLI)

```powershell
cargo build --bin predict_state
cargo run --bin predict_state -- --config run_config_chess_ply_classical.json
```

## Quickstart (REST API node)

```powershell
cargo build --bin w1z4rd_api
W1Z4RDV1510N_SERVICE_ADDR=0.0.0.0:8080 cargo run --bin w1z4rd_api
```

Or use the pre-built binary (avoiding AV false positives — see note below):

```powershell
# Run from the project root so relative storage paths resolve correctly
bin\w1z4rd_node.exe
```

## Quickstart (chess trio + live viz)

```powershell
# Terminal 1: viz server (localhost:8765)
python scripts/live_viz_server.py

# Terminal 2: chess prediction runner
python scripts/chess_prediction_runner.py
```

The runner auto-detects the `w1z4rd_api` node at `http://localhost:8080` and routes through it when available, falling back to direct `predict_state` subprocess otherwise.

## Quickstart (PGN → snapshot)

Convert a PGN file to a full `EnvironmentSnapshot` with one frame per ply:

```powershell
cargo run --bin pgn_to_snapshot -- --pgn path/to/game.pgn --out logs/game_snapshot.json
cargo run --bin pgn_to_snapshot -- --pgn path/to/game.pgn --out logs/game_snapshot.json --max-plies 20
```

## Quickstart (P2P node)

```powershell
# Initialize config + wallet
cargo run --bin w1z4rdv1510n-node -- init

# Run the node
cargo run --bin w1z4rdv1510n-node

# Start the node API
cargo run --bin w1z4rdv1510n-node -- api --addr 127.0.0.1:8090

# Inspect local label queues
cargo run --bin w1z4rdv1510n-node -- label-queue --limit 25
cargo run --bin w1z4rdv1510n-node -- visual-label-queue --limit 25

# Simulate a network
cargo run --bin w1z4rdv1510n-node -- sim --nodes 10000 --ticks 100

# Query the distributed pattern index
cargo run --bin w1z4rdv1510n-node -- pattern-query --behavior-signature 0.12,0.3,0.98 --broadcast true --wait-for-responses-ms 1500
```

## Quickstart (streaming service)

```powershell
cargo run --bin streaming_service -- --config run_config.json --input stream.jsonl
```

Pipe RTSP/video pose output into the service:

```powershell
python scripts/rtsp_pose_bridge.py --source rtsp://camera --emit-image ref --image-dir data/frames | cargo run --bin streaming_service -- --config run_config_streaming_ocr.json --input -
```

---

## API reference

### `w1z4rd_api` (port 8080 by default)

Set listen address with env var: `W1Z4RDV1510N_SERVICE_ADDR=0.0.0.0:8080`
Set storage path with env var: `W1Z4RDV1510N_SERVICE_STORAGE=logs/service_runs`

#### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Service health + uptime stats |
| GET | `/readyz` | Ready check (returns 503 until warmed up) |

#### Prediction
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Submit snapshot for classical/quantum/neuro prediction |
| GET | `/runs` | List completed runs |
| GET | `/runs/{run_id}` | Get a specific run result |
| GET | `/jobs` | List async job queue |
| GET | `/jobs/{job_id}` | Poll async job status |

#### Neuro fabric
| Method | Path | Description |
|--------|------|-------------|
| POST | `/neuro/train` | Observe a multi-stream snapshot → update Hebbian weights |
| POST | `/neuro/activate` | Input labels from one stream → read out target stream labels |
| POST | `/neuro/reconstruct` | Sequence of input frames → reconstructed target stream frames |
| GET | `/neuro/snapshot` | Current neuro state: active labels, networks, centroids, influences |

**`POST /neuro/activate` body:**
```json
{
  "input_labels": ["stream::chess", "meta::side_to_move::white", "zone_e4"],
  "target_stream": "audio",
  "hops": 3
}
```

**`POST /neuro/reconstruct` body:**
```json
{
  "frames": [["stream::video", "zone_e4"], ["stream::video", "zone_e5"]],
  "target_stream": "audio",
  "hops": 3,
  "carry": 0.2
}
```

#### Threat analysis
| Method | Path | Description |
|--------|------|-------------|
| POST | `/threat/ingest` | Ingest per-entity sensor attributes for the current frame (fire-and-forget, returns 204) |
| GET | `/threat/overlay` | Return the current `ThreatOverlay` — scene alarm, per-entity health + intent + arousal + predictions |
| POST | `/threat/predict` | Ingest a frame of attributes and immediately return the resulting overlay in one round trip |

**`POST /threat/ingest` and `POST /threat/predict` body:**
```json
{
  "timestamp_unix": 1732003600,
  "frame": {
    "person_1": {
      "pos_x": 2.5, "pos_y": 1.0, "pos_z": 0.0,
      "orient_x": 0.0, "orient_y": 1.0,
      "gaze_velocity": 0.8,
      "postural_rigidity": 0.6,
      "concealed_object": false,
      "exit_monitoring": 0.7,
      "hand_conceal": 0.5
    }
  }
}
```

The overlay response includes `scene_alarm`, `scene_color`, `threat_entity_count`, and per-entity summaries with `composite_threat`, `threat_color`, `health`, `intent`, `arousal`, and ranked `predictions`.

### Node API (`w1z4rdv1510n-node`, port 8090 by default)
- `/health`, `/ready`, `/metrics`
- `/data/ingest`, `/data/:data_id`
- `/knowledge/ingest`, `/knowledge/queue`, `/knowledge/vote`
- `/qa/ingest` (POST), `/qa/query` (POST)
- `/streaming/labels`, `/streaming/visual-labels`, `/streaming/subnets`
- `/network/patterns/query`
- `/bridge/chains`, `/bridge/intent`, `/bridge/proof`, `/balance/:node_id`
- `/identity/challenge`, `/identity/verify`, `/identity/:thread_id`

---

## Tools and utilities

| Script | Description |
|--------|-------------|
| `scripts/chess_prediction_runner.py` | Chess trio runner with live board output, motion vectors, metadata labels, live probability metrics |
| `scripts/live_viz_server.py` | Live visualization server (localhost:8765); model breakdown panel, W%/B% outcome bars, energy probabilities |
| `scripts/rtsp_pose_bridge.py` | RTSP/video to pose JSONL bridge |
| `scripts/rss_topic_bridge.py` | RSS/Atom to PublicTopics StreamEnvelope JSONL bridge |
| `scripts/traffic_sensor_bridge.py` | Traffic sensor JSON/CSV to CrowdTraffic StreamEnvelope JSONL bridge |
| `scripts/run_with_viz.py` | Legacy viz runner (superseded by direct viz server + chess runner pair) |
| `scripts/visualize_snapshot.py` | Snapshot viewer + `--live` polling mode; watches `chess_live_board.json` or any snapshot file and auto-refreshes a browser HTML wrapper every N seconds; accepts `--interval`, `--open` flags |
| `scripts/preprocess_chess_games.py` | Preprocess PGN files into training sequences |
| `scripts/build_relational_priors.py` | Build relational priors from training data |
| `scripts/export_chess_stack_snapshot.py` | Export chess stack snapshots for analysis |
| `scripts/export_sequence_snapshot.py` | Export sequence snapshots |
| `scripts/export_uspto_snapshot.py` | Export USPTO reaction snapshots |
| `scripts/prepare_textbook_qa_dataset.py` | Render textbook PDFs into page images + OCR, emit review queues, build QA candidates |
| `scripts/fetch_*.py` | Dataset fetchers: ChEMBL, ETH/UCY, exoplanets, OWID COVID, Tox21, USPTO reactions |
| `scripts/preprocess_*.py` | Preprocessors: ETH/UCY sequences, genomic sequences, HYG star catalog, OWID, Tox21, USPTO50k |
| `scripts/generate_synthetic_dataset.py` | Synthetic dataset generator for testing |
| `scripts/eval_uspto_batch.py` | USPTO batch evaluation runner |
| `scripts/chess_training_loop.py` | Extended chess training loop |

Dataset fetchers/preprocessors support: USPTO reactions, Tox21, ETH/UCY pedestrian trajectories, exoplanets, OWID COVID, ChEMBL, genomic sequences, HYG star catalog.

### Textbook Q&A pipeline (end to end)

```powershell
# 1. Download OpenStax textbooks (test mode — no actual download)
node textbook_scripts/download_and_process.mjs

# 2. Download for real
$env:TEXTBOOK_TEST_MODE="false"; $env:TEXTBOOK_MAX_BOOKS="5"; $env:TEXTBOOK_SUBJECTS="science,math"
node textbook_scripts/download_and_process.mjs

# 3. Extract Q&A pairs from downloaded PDFs
python scripts/prepare_textbook_qa_dataset.py --max-books 3 --max-pages 25 --skip-existing

# 4. Ingest Q&A pairs into the Hebbian fabric via the node API
#    (node must be running: cargo run --bin w1z4rdv1510n-node -- api)
curl -X POST http://127.0.0.1:8090/qa/ingest \
  -H "Content-Type: application/json" \
  -d '{"candidates": [{"qa_id":"q1","question":"What is photosynthesis?","answer":"The process by which plants convert light into energy.","book_id":"biology-2e","page_index":42,"confidence":0.85,"evidence":"","review_status":"PENDING"}]}'

# 5. Query the fabric
curl -X POST http://127.0.0.1:8090/qa/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is photosynthesis?"}'
```

---

## Repository layout

```
crates/core/
  src/
    main.rs                          — predict_state CLI entry point
    bin/
      service.rs                     — w1z4rd_api REST node (prediction + neuro endpoints)
      calibrate_energy.rs            — energy weight calibration from trajectories
      pgn_to_snapshot.rs             — PGN → EnvironmentSnapshot converter
    streaming/
      qa_runtime.rs                  — Hebbian Q&A associative memory
      hierarchical_motifs.rs         — recursive motif-of-motifs (unbounded depth, Shannon entropy)
      organic_encoder.rs             — V1-style feature encoding with Gabor filters + EMA basis
      dynamic_pools.rs               — organic neural pool spawning with Hebbian cross-associations
      sensor_registry.rs             — plug-and-play sensor auto-discovery with TTL
      motif_label_bridge.rs          — motif → human labeling queue bridge
      neuro_bridge.rs                — bridge between streaming pipeline and NeuroRuntime
      fabric.rs, fabric_trainer.rs   — streaming neural fabric integration
      [+ 30 other streaming modules]
    neuro.rs                         — NeuronPool, NeuroRuntime, cross-stream inference, influence tracking
    quantum.rs                       — Trotter-slice quantum annealing
    quantum_calibration.rs           — persistent calibration state between runs
    energy.rs                        — energy functions (goal, collision, motion, stack-hash)
    annealing.rs                     — MCMC classical annealer
    schema.rs                        — EnvironmentSnapshot, Symbol (with velocity), DynamicState
    orchestrator.rs                  — run coordinator
    ml.rs                            — ML prior integration
    spike.rs                         — spike pool and routing
    threat/
      health.rs                      — 6D HealthVector, HSV color encoding, scope stacking, baseline tracker
      subradian.rs                   — sub-ultradian arousal bands (Startle/AcuteArousal/ThreatVigilance/SustainedArousal)
      field.rs                       — sparse 2D spatial threat field with proxemics + orientation convergence
      intent.rs                      — behavioral intent inference (Normal → ArmedThreat), health impact prediction
      consensus.rs                   — multi-entity wave-function collapse, Shannon entropy, alarm levels
      propagation.rs                 — Hebbian health propagation graph, human_body_graph() factory
      overlay.rs                     — ThreatOverlay + HealthPrediction unified API output
      mod.rs                         — ThreatScene convenience wrapper (ingest_frame + overlay)

crates/node/
  src/
    main.rs                          — w1z4rdv1510n-node CLI
    api.rs                           — node REST API
    p2p.rs                           — libp2p gossipsub + Kademlia + mDNS
    data_mesh.rs                     — chunking, replication, integrity
    ledger.rs                        — local audit chain + rewards
    identity.rs                      — behavior-derived identity verification
    label_queue.rs                   — human annotation queue

crates/experimental-hw/              — experimental hardware backends (GPU stub)

packages/microcortex/                — lightweight JS neural fabric (Hebbian, WTA, softmax)

scripts/                             — Python runners, bridges, dataset tools
textbook_scripts/                    — OpenStax download + segmentation pipeline
run_config*.json                     — annealer + streaming configuration profiles
chain/                               — genesis + reward/bridge/token specs
schemas/                             — JSON schemas for bridge intents
bin/                                 — pre-built executables (AV exclusion required on Windows)
```

---

## Windows / Antivirus note

On Windows with Avira (and likely other AV products), self-compiled Rust executables may trigger HEUR/AGEN heuristic detection. The `w1z4rd_api` binary is the most commonly flagged.

Workaround:
1. Add your project's `target\` folder to AV excluded paths.
2. Build to a temp directory: `$env:CARGO_TARGET_DIR="C:\Temp\w1z4rd_build"; cargo build --release --bin w1z4rd_api`
3. Copy the binary to `bin\` with a different name (e.g. `w1z4rd_node.exe`) to avoid name-based triggers.
4. Run from the project root so relative storage paths (`logs/service_runs`) resolve correctly.

The `predict_state` binary is generally not flagged and can be built and run normally.

---

## Build and test

```powershell
cargo fmt
cargo test
cargo build --bin predict_state
cargo build --bin w1z4rd_api
cargo build --bin calibrate_energy
```
