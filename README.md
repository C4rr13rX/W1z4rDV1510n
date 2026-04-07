# W1z4rDV1510n

A distributed intelligence node that learns to physically describe its environment — not by classification, but by growing structure from what it observes. CPU and RAM native. No GPUs required.

The node is an **instrument, not an agent**. It observes every incoming sensor stream, builds a living representation of the environment inside itself, and reports what it sees in the language of physics. It does not act on that data. Whatever acts on its outputs — a script, a decision system, a human — does so with full transparency into how the node arrived at its conclusions. The node is the measurement device. Everything else is up to you.

Every sensor stream — a chess board, a video camera, a news feed, a social graph, a chemical state — arrives in the same generic format. The node has no knowledge of any specific domain. It sees positions, labels, and co-occurrences. What it learns from a chess game transfers to what it knows about crowd dynamics, and vice versa. The neural fabric that grows from one domain is the same fabric another domain trains.

The Environmental Equation Matrix sits at the center of this. As the neural fabric fires labels from sensor data, the EEM continuously asks: *which physical laws govern what I'm currently observing?* It works across 282 equations and 24 disciplines simultaneously — because the environment doesn't respect disciplinary boundaries. A crowd tipping toward panic obeys the same Kuramoto coupling equations as synchronized chemical oscillators. A viral narrative spreading through a social network follows the same Bass diffusion curve as a product launch. A coordinated information campaign has a measurable Lyapunov exponent. The EEM names these processes from raw observation, without being told what to look for.

When the EEM can't explain a label cluster — when something is happening that doesn't match any known equation — it opens a hypothesis gap and records which nodes are seeing it. A gap observed independently across many nodes in a distributed deployment is the system's way of saying: *something real is happening here that we don't have a name for yet.* That is a more useful signal than a classification.

---

**What this does that isn't done elsewhere:**

- **Physics as the common language across domains.** Most multi-modal systems learn domain-specific representations and build bridges between them. This system skips that entirely — everything is expressed in the same dimensional sensor format from the start, and the physics equations are the shared vocabulary across all of it.

- **The node is an observer, not a decision-maker.** Intelligence architectures almost universally couple observation to action. Here they are explicitly separated. The node reports; agents built on top decide. This means the node's outputs are auditable, neutral, and composable — you can put any decision layer on top without changing the instrument.

- **Hypothesis gaps as first-class output.** The node tracks what it *cannot* explain with equal rigor to what it can. The gap leaderboard — ordered by how many independent nodes corroborate the same unexplained pattern — is often more actionable than the list of identified equations.

- **Cross-node source tracing without coordination.** Each node independently records which node first reported each pattern and how many subsequently corroborated it. No central coordinator, no shared state beyond the gossip layer. The origin of a propagating signal emerges from timestamp ordering across independent observers.

- **Equations gain and lose confidence from sensor evidence.** The EEM is not a static lookup table. Equations that consistently match what sensors are reporting grow stronger. Those that don't, decay. The system learns which physics are actually present in its deployment environment, not just which physics exist in textbooks.

- **Hardware-adaptive from first principles.** No batch sizes, thread counts, or memory limits are hard-coded anywhere. Every parameter is derived from live hardware measurement at startup. The same binary runs on a Raspberry Pi and a workstation and adapts its behavior to each.

- **Game theory and marketing science as physics.** Social, strategic, and market dynamics are expressed as equations with the same status as thermodynamics or quantum mechanics. A Nash equilibrium is detected the same way a Boltzmann distribution is — by matching sensor label clusters to equation keywords. The system doesn't know it's looking at a market; it just knows what the math says.

---

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Node (Rust)                              │
│                                                                 │
│  NeuroRuntime  ──  EquationMatrix  ──  QaRuntime               │
│       │                  │                  │                   │
│  KnowledgeRuntime  ──  HierarchicalMotifs  ──  FabricTrainer   │
│       │                                         │               │
│  P2P Gossip / Data Mesh / Blockchain Layer                      │
└─────────────────────────────────────────────────────────────────┘
          ▲                ▲                 ▲
          │                │                 │
   chess_training    rtsp_pose_bridge   rss_topic_bridge
   _loop.py          .py                .py
   (app / script)    (app / script)     (app / script)
```

**Apps send generic `EnvironmentSnapshot` objects. The node has no knowledge of chess, poses, or topics — it sees symbols with positions, labels, and metadata. Every app trains the same neural fabric.**

---

## Core components

### Neural Fabric (`NeuroRuntime`)
- CPU+RAM neuron pools with neurogenesis, Hebbian/STDP learning, winner-take-all sparsification, and mini-columns that collapse to concept neurons over time
- `observe_snapshot()` — ingest any `EnvironmentSnapshot`; symbols become zone labels, velocity vectors and metadata flow in as `meta::key::value` labels
- `cross_stream_activate()` — propagate from one stream's labels and read what fires in another stream; cross-modal inference through grown Hebbian connections
- `reconstruct_sequence()` — frame-by-frame temporal propagation with dynamic carry factor (cosine similarity between consecutive frames)
- `propagate()` — passive synapse walk returning label→strength map without mutating pool state
- Influence/provenance tracking: every neuron records which streams and data shaped it
- Double-buffered propagation, binary-search synapse lists, EMA co-occurrence tracking, dirty-flag mini-columns — all hardware-adaptive, no hard-coded limits

### Environmental Equation Matrix (`EquationMatrixRuntime`)
- A self-growing directed graph of physics equations spanning all domains from Newtonian mechanics to topological quantum phenomena
- **282 seed equations across 24 disciplines**:
  - Classical Mechanics (F=ma through SHO, impulse, center of mass, torque, rotational dynamics)
  - Waves & Oscillations (wave equation, Doppler, decibel scale, intensity)
  - Lagrangian / Hamiltonian mechanics (action principle, Poisson brackets, adiabatic invariants)
  - Thermodynamics (all four laws, Gibbs / Helmholtz / Carnot, van der Waals, Stefan-Boltzmann, Planck blackbody)
  - Statistical Mechanics (Boltzmann / Fermi-Dirac / Bose-Einstein, partition functions, Jarzynski equality, grand canonical)
  - Electromagnetism & Optics (full Maxwell set, Lorentz force, Poynting vector, Snell's law, diffraction grating, Malus's law)
  - Quantum Mechanics (Schrödinger TDSE/TISE, hydrogen levels, QHO, tunneling, spin commutation, density matrix, Bell state, de Broglie)
  - Quantum Field Theory (Dirac equation, Klein-Gordon, QED Lagrangian, Standard Model Lagrangian, Higgs potential, renormalization group)
  - Special Relativity (full Lorentz transform, 4-momentum, velocity addition, Minkowski metric)
  - General Relativity (Einstein field equations, geodesic, Christoffel symbols, Riemann tensor, Schwarzschild metric, Hawking temperature, gravitational waves)
  - Nuclear & Particle Physics (binding energy, radioactive decay, half-life, Q-value)
  - Condensed Matter (BCS ground state, tight-binding, Drude, cyclotron frequency, flux quantum, Fermi energy)
  - Fluid Dynamics (Navier-Stokes, Bernoulli, vorticity transport, Mach/Froude numbers)
  - Chaos / Nonlinear Dynamics (Lyapunov exponent, Fokker-Planck, Langevin, fractal dimension, Kuramoto coupled oscillators, KAM theorem)
  - Topological Physics (Chern number, Berry curvature, Z₂ invariant, Kitaev chain, fractional charge e*=e/3)
  - Cosmology (Friedmann equations, redshift, luminosity distance, critical density, CMB temperature scaling)
  - Information Theory (Shannon entropy, channel capacity, KL divergence, Fisher information, Cramér-Rao bound, Kolmogorov complexity)
  - Mathematical Physics PDEs (heat equation, Laplace, Poisson, Burgers, Ginzburg-Landau, KdV, nonlinear Schrödinger)
  - Biophysics & Complex Systems (logistic growth, Lotka-Volterra, Hodgkin-Huxley neuron, Einstein diffusion, Stokes-Einstein)
  - Plasma Physics (plasma frequency, Debye length)
  - Mathematical Tools (Bayes' theorem, Hebbian learning, sigmoid/softmax, cosine similarity, Pearson correlation, DTW, EMA)
  - **Game Theory** (Nash equilibrium, minimax, replicator dynamics, ESS, prisoner's dilemma, folk theorem, Shapley value, price of anarchy, Bayesian Nash, Hotelling spatial competition, Schelling focal points, auction revenue equivalence)
  - **Marketing Science** (Bass diffusion, viral coefficient k, CLV, Lanchester's square law, adstock carryover, price elasticity, Metcalfe's law, Reed's law, independent cascade, linear threshold, Gompertz adoption, marketing mix, persuasion/ELM, preferential attachment, Zipf's law, Pareto 80/20)
  - **Chaos Theory extended** (Lorenz attractor, logistic map, Feigenbaum constant, Rössler attractor, Liouville theorem, correlation dimension, Poincaré recurrence, tent map, KAM theorem, entropy production rate, sensitive dependence)
  - **Quantum extended** (Lindblad master equation, quantum Zeno effect, Wigner function, Rabi oscillations, Bloch sphere, quantum mutual information, quantum discord, Grover search, Shor's algorithm, quantum error correction, quantum Fisher information, decoherence time, teleportation fidelity)
  - **Cross-Disciplinary Bridges** (Friston free energy principle, Jaynes maximum entropy, power laws, percolation threshold, small-world networks, Ising opinion dynamics, mean field theory, RG fixed points, Kolmogorov complexity, integrated information Φ, fitness landscapes, cascade failure, Red Queen coevolution, Schelling segregation, NK complexity model)
- **23 semantic links** between equations: `derives_from`, `bridges`, `special_case`, `unifies`, `approximates`, `generalizes`, `contradicts`
- **Dimension-aware**: equations tagged with spatial applicability. Anyons (`ψ → e^{iθ}ψ`) are strictly 2D — they will not surface in a 3D sensor context. Maxwell's equations are 3D. Thermodynamic identities are dimension-agnostic
- **Sensor-driven**: `apply_to_context(labels, dims)` takes active neuro-fabric labels + sensor dimensionality and returns candidate equations explaining the current observation
- **Confidence evolution**: grows from sensor evidence, decays without corroboration — equations compete for relevance against what the node is actually experiencing
- **Hypothesis gap tracking**: unexplained sensor patterns recorded as open `HypothesisSlot` entries — the node acknowledges what it can't yet explain
- **P2P-shareable**: `EemPeerPayload` lets nodes broadcast equation discoveries and merge each other's findings over the gossip network
- Persisted to disk; reloaded on restart

### Recursive Motif Discovery (`HierarchicalMotifRuntime`)
- Motifs of motifs of motifs with no cap on hierarchy depth
- Each level's promotions seed the next until the signal exhausts itself
- Shannon entropy attractor detection at every level
- Hardware-adaptive window caps and length filters — no hard-coded limits
- Edit-distance similarity with fast length pre-filter

### Hebbian Q&A Fabric (`QaRuntime`)
- Textbook and domain knowledge encoded as grown synaptic state
- Querying fires input neurons and reads the output network — no matrix multiplication at inference
- The answer is a reaction in the fabric's state
- API: `POST /qa/ingest`, `POST /qa/query`

### Knowledge Graph (`KnowledgeRuntime`)
- JATS/NLM ingestion with text blocks, figure assets, and OCR hooks
- Figure-to-text association tasks with voting and confidence thresholds
- Label queue for emergent dimensions and novel attributes
- Hebbian links between knowledge entities and neuro-fabric labels

### Node Modes
Two operating modes — no code changes required, controlled by `node_config.json`:

| Mode | Wallet | Data Mesh | Blockchain | Use case |
|------|--------|-----------|------------|----------|
| `SENSOR` | Optional | Off | Off | Local AI, training loops, development |
| `PRODUCTION` | Required | On | On | Full Web3 hybrid, cluster computing |

The node API (`/health`, `/neuro/train`, `/equations/search`, `/qa/ingest`, etc.) runs in both modes.

---

## Node API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Node status and uptime |
| `/neuro/train` | POST | Feed an `EnvironmentSnapshot` to the neural fabric |
| `/neuro/snapshot` | GET | Current activation, predictions, motifs, top influences |
| `/equations/search` | GET | Text search across the equation matrix |
| `/equations/apply` | POST | Find equations that explain active sensor labels + dims |
| `/equations/ingest` | POST | Add equations from free text |
| `/equations/report` | GET | Full EEM report: counts by discipline, top equations |
| `/equations/gaps` | GET | Open hypothesis slots sorted by cross-node corroboration |
| `/equations/peer_sync` | POST | Receive `EemPeerPayload` from a peer node |
| `/causal/graph` | GET | Named-process causal graph: sensor clusters → physics processes |
| `/network/patterns/sources` | GET | First-reporter origin index for cross-node source tracing |
| `/qa/ingest` | POST | Ingest Q&A pairs into the Hebbian fabric |
| `/qa/query` | POST | Query the Q&A fabric |
| `/knowledge/ingest` | POST | Ingest documents into the knowledge graph |
| `/knowledge/vote` | POST | Vote on knowledge associations |
| `/streaming/labels` | GET | Label queue for human annotation |
| `/network/patterns/query` | POST | Query distributed pattern index |
| `/threat/ingest` | POST | Ingest behavioral threat data |
| `/threat/overlay` | GET | Current threat and health overlay |
| `/metrics` | GET | Node performance metrics |

---

## Sensor format

Every data source translates to `EnvironmentSnapshot`:

```json
{
  "timestamp": { "unix": 1712345678 },
  "bounds": { "x": 8.0, "y": 8.0, "z": 0.0 },
  "symbols": [
    {
      "id": "piece_white_K_e1",
      "type": "CUSTOM",
      "position": { "x": 4.0, "y": 0.0, "z": 0.0 },
      "properties": {
        "role": "K", "color": "white", "zone": "0,0",
        "stream": "chess", "result": "1-0"
      }
    }
  ],
  "metadata": { "stream": "chess", "player_white": "Magnus" },
  "stack_history": []
}
```

A chess piece, a LiDAR point, a stock tick, a chemical state, a crowd zone — all the same format. The node learns from all of them simultaneously through the same neural fabric.

**2D and 3D sensors are handled natively.** The `z` coordinate is 0.0 for 2D sensors; the fabric learns spatial patterns regardless of dimensionality. The equation matrix uses the `dims` field to filter which physics equations apply — anyons only surface for 2D sensor contexts, Maxwell in 3D, thermodynamics everywhere.

---

## Decentralized node mesh

- **P2P networking**: libp2p gossipsub + Kademlia + mDNS; rate limits and peer scoring
- **Data mesh**: manifests, chunking, replication receipts, integrity audits, repair requests, storage rewards
- **Neural fabric sharing**: `NeuralFabricShare` payloads broadcast motifs and network pattern summaries; peer neuro snapshots train local weights at low learning rate — what one node learns propagates everywhere
- **Equation sharing**: `EemPeerPayload` propagates equation discoveries and hypothesis gaps across nodes; peer equations get lower initial confidence and must be corroborated by local sensor data
- **Local ledger**: validator heartbeats, fee-market scaffolding, audit chain, reward events
- **Multi-chain bridge**: intent tracking + relayer-quorum proof verification
- **Encrypted wallet**: node identity and rewards; optional in SENSOR mode

---

## What it does (full stack)

### 1) Symbol matrix inference engine
- Population-based state proposals with energy models, MCMC-style acceptance, and annealing schedules
- Classical and quantum-inspired annealing; calibration hooks for remote quantum hardware
- Neural priors and temporal motif signals integrated into proposal scoring
- Runs as CLI (`predict_state`) or REST service for batch and live jobs
- AtomicU64 lock-free acceptance counters; hardware-adaptive annealing

### 2) Neural fabric and cross-stream inference

#### Training
- `observe_snapshot()` — ingest any `EnvironmentSnapshot`; converts symbols to zone/role/metadata labels, Hebbian updates across all activated neurons
- `train_weighted_with_meta()` — weighted Hebbian update with full metadata context attached to each synapse modification
- Neurons accumulate up to 16 influence records (weakest evicted when full); records from the same stream+label set merged and strength averaged

#### Inference
- `cross_stream_activate(labels, target_stream, hops)` — propagate hop-by-hop through Hebbian synapses, collect activations matching target stream
- `reconstruct_sequence(frames, target_stream, hops)` — temporal sequence reconstruction with dynamic carry factor from cosine similarity between consecutive frames
- `propagate(seed_labels, hops)` — passive synapse walk without mutating pool state

#### Snapshot
`NeuroSnapshot` exposes: `active_labels`, `active_networks`, `minicolumns`, `centroids`, `network_links`, `temporal_predictions`, `temporal_motif_priors`, `top_influences`, `active_streams`, `active_meta_labels`, `working_memory`

### 3) Environmental Equation Matrix

The EEM bridges the gap between raw sensor patterns and physical interpretation. As the neuro fabric fires labels, the EEM surfaces candidate equations governing the observed phenomenon. It is a complete map of modern physics — 214 equations across 21 disciplines — compiled into the node so that any sensor stream can be interpreted through the lens of physical law.

- Equations accumulate sensor-driven evidence; those that consistently explain observations gain confidence; those that don't, decay
- When the fabric fires labels that match no equation, a `HypothesisSlot` is opened — the node records it as an unexplained phenomenon awaiting discovery
- Peer nodes share their equation discoveries via `EemPeerPayload`; merged equations must be corroborated by local sensor data before gaining full confidence
- The system is self-researching: new sensor patterns drive new hypotheses; peer knowledge fills gaps

**Coverage highlights**:
- From F=ma through the Standard Model Lagrangian and Higgs potential
- Thermodynamic identities from all four laws to Jarzynski's fluctuation theorem
- Quantum mechanics through QFT: Dirac equation, renormalization group, Bell states
- GR: Einstein field equations, Schwarzschild metric, Hawking temperature, gravitational waves
- Topological physics: Chern numbers, Berry curvature, Z₂ invariants, Kitaev chain
- Chaos and complexity: Lyapunov exponents, Fokker-Planck, Kuramoto oscillators, KAM theorem
- Biophysics: Hodgkin-Huxley neuron model, Lotka-Volterra, Einstein/Stokes-Einstein diffusion
- Information theory: Shannon, KL divergence, Fisher information, Cramér-Rao bound

**Pattern source detection** works through three interlocked mechanisms — no domain-specific detector needed:

1. **EEM auto-apply on every training frame** — `POST /neuro/train` now automatically runs `apply_to_context` on the snapshot's sensor labels. Every equation that matches gains evidence; unmatched label clusters open or increment a `HypothesisSlot`. The EEM accumulates a continuous record of which physics processes are active in the sensor stream.

2. **Named-process causal graph** (`/causal/graph`) — each `equations_apply` call writes directed edges `sensor::{label_cluster_hash} → process::{equation_id}` into a causal graph. Over time, coordinated signals appear as high-weight edges from a single sensor cluster to a specific process node (e.g. `process::kuramoto_coupling`). Walking those edges backward across time-stamps traces the wave front to its origin.

3. **Cross-node first-reporter index** (`/network/patterns/sources`) — every pattern thread returned by peer nodes is recorded with the node ID and timestamp of first sighting. The node that appears earliest with the highest subsequent corroboration is the statistical source of that pattern — this is how a coordinated campaign propagating through the network can be traced back to its injection point without any application-specific logic.

Gap escalation: `HypothesisSlot` entries now carry `first_node_id` and `reporting_nodes`. Slots observed across multiple independent nodes are ranked highest in `/equations/gaps` and included in `EemPeerPayload` for network-wide escalation via `POST /equations/peer_sync`. A gap that multiple nodes see but nobody can explain is the strongest possible signal of a genuinely novel phenomenon in the environment.

**Anyon note**: anyons are quasiparticles that exist only in 2D topological systems. Their exchange statistics (`ψ → e^{iθ}ψ`) are neither bosonic (θ=0) nor fermionic (θ=π) but can be any angle — this is a fundamental consequence of 2D topology. The EEM correctly enforces this: the anyon equation, Chern-Simons action, and fractional charge equations only surface when the active sensor context is flagged as 2D. A 3D sensor stream will never see them.

### 4) Streaming ultradian analysis pipeline
- People video / pose frames → motor features + behavioral atoms
- Crowd/traffic signals → flow layers (density, velocity, directionality, stop-go waves, daily/weekly cycles)
- Public topic streams → event layers (burst/decay/excitation/lead-lag/periodicity)
- Extracts ultradian micro-arousal, BRAC, and meso layers as phase/amplitude/coherence
- Aligns across modalities with tolerance windows and per-source confidence gating
- Optional OCR adapter enriches video frames with text blocks

### 5) Behavior substrate and motifs
- Body-schema adapters map multimodal sensors into shared latent state + action vector
- Change-point segmentation plus fixed windows for stable coverage
- Motif discovery using DTW/soft-DTW similarity, graph signatures, and MDL costs
- Behavior graph coupling metrics: proximity, coherence, phase-locking, transfer-entropy proxies
- Attractor detection via next-state entropy with optional constraints and soft objectives

### 6) Multi-domain fusion and temporal inference
- Learned multi-domain hypergraph links tokens and layers with decay, TTL, and gating
- Temporal inference predicts phase/amplitude/coherence drifts, cross-layer coherence, and next-event intensities
- Evidential outputs use Dirichlet posteriors for event and regime uncertainty

### 7) Causal graph and branching futures
- Streaming causal graph updates with time-lag edges and intervention deltas
- Counterfactual do()-style interventions inform branch payloads
- Branching futures include confidence/uncertainty and retrodicted missed events

### 8) Online learning, consistency chunking, and ontology
- Surprise-weighted replay reservoir with trust-region updates and rollback protection
- Horizon manager only expands when calibration improves
- Consistency chunking builds reusable templates (codebook) from stable motifs
- Ontology runtime versions labels across minute/hour/day/week windows

### 9) Knowledge ingestion and textbook Q&A
- JATS/NLM ingestion with text blocks, figure assets, and OCR hooks
- Figure-to-text association tasks with voting and confidence thresholds
- **Textbook pipeline** (`textbook_scripts/`): downloads CC-licensed OpenStax PDFs, segments pages into labeled bounding boxes using a microcortex perceptron classifier, extracts Q&A candidate pairs, emits review queues for human annotation
- **Hebbian Q&A fabric**: verified Q&A pairs encoded into synaptic state; at query time, question tokens fire input neurons; output network surfaces ranked answers — no matrix math at inference

### 10) Health, survival, and threat overlays

A physics-grounded multi-dimensional health model applying to any entity — human, animal, machine, plant — observable through sensor streams without biosensors.

#### 6D health vector

| Dimension | Short | Hue | Meaning |
|-----------|-------|-----|---------|
| StructuralIntegrity | SI | 0° red | Physical substrate integrity |
| EnergeticFlux | EF | 30° orange | Energy acquisition and distribution |
| RegulatoryControl | RC | 210° blue | Homeostasis and coordination |
| FunctionalOutput | FO | 120° green | Characteristic operation capacity |
| AdaptiveReserve | AR | 270° violet | Reserve capacity to absorb further stress |
| TemporalCoherence | TC | 60° yellow | Biological/operational rhythms intact |

HSV color encoding: Value = overall scalar (0=dead/black, 1=optimal/white); Hue = weighted circular mean of dimension anchors; Saturation = deviation from rolling baseline.

#### Spatial threat field and intent inference
- Sparse 2D grid proxemics zones (Hall 1966): Intimate / Personal / Social / Public
- Bayesian-style softmax over 9 intent classes: Normal → Survey → Approach → Conceal → ControlEnvironment → ApproachDemand → Flee → DirectThreat → ArmedThreat
- Signal decay prevents stale locks; `build_health_impacts()` projects forward health deltas with time horizon

#### Wave function collapse consensus
Multi-entity consensus via complementary probability rule: `consensus = 1 - ∏(1 - estimate_i)`. Shannon entropy tracked over 16-frame history; when entropy drops below 0.35 and is falling >5%/frame, the wave function is declared collapsed. Alarm levels: none / low / medium / high / critical.

#### Health propagation graph
Typed edges (Structural, Vascular, Neural, Proximity) with speeds and dimension weights. Hebbian edge learning: co-occurring degradation on both ends of an edge strengthens that edge (+0.02 per co-occurrence) — the system learns anatomical coupling from observation.

### 11) Network-wide neural fabric
- Shares motifs, transitions, and network pattern summaries across nodes
- Entity threads track phenotype tokens, behavior signatures, and plausible travel-time continuity
- Queryable distributed pattern indices keep nodes aligned
- Peer neuro snapshots train local weights at low learning rate so cross-node learning is automatic

### 12) Node stack and incentives
- P2P networking (libp2p gossipsub + Kademlia + mDNS) with rate limits and peer scoring
- Data mesh: manifests, chunking, replication receipts, integrity audits, repair requests
- Local ledger: validator heartbeats, fee-market scaffolding, audit chain, reward events
- Multi-chain bridge: intent tracking + relayer-quorum proof verification
- Encrypted wallet for node identity and rewards
- SENSOR / PRODUCTION mode switch — no passphrase prompt in SENSOR mode

### 13) Opt-in identity verification
- Behavior-derived challenges (position + motion signature + code) bound to wallets on-chain
- Supports re-issuing bindings to a new wallet via API
- No face ID, no biometric identity resolution

### 14) Adaptive resource governance (`ResourceMonitor`)
- **No hard-coded CPU, RAM, or batch limits anywhere in the system**
- Monitors live system usage via `psutil`; reserves configurable headroom for OS and interactive use
- Ramps workload up in small steps when resources are plentiful; ramps down fast when under pressure
- Applied to batch sizes, game/sample counts, and iteration sleep — everything scales together
- Scripts that use the node implement `ResourceMonitor`; the node itself is always responsive

---

## Apps / scripts (examples)

These scripts use the node architecture — the node has no knowledge of their domains:

| Script | Domain | What it sends |
|--------|--------|---------------|
| `chess_training_loop.py` | Chess | Board positions as 2D `EnvironmentSnapshot`; moves as Q&A pairs |
| `rtsp_pose_bridge.py` | Video / pose | Pose keypoints as 3D symbol positions |
| `rss_topic_bridge.py` | Topics | Topic signals as streaming events |
| `traffic_sensor_bridge.py` | Traffic | Flow data as crowd sensor stream |
| `textbook_scripts/` | Knowledge | Q&A pairs from OpenStax textbooks |

**Adding a new domain**: translate your data to `EnvironmentSnapshot`, `POST` to `/neuro/train`, optionally send to `/qa/ingest` and `/equations/apply`. The node learns from it alongside everything else.

---

## Running

```bash
# Build everything
cargo build --release

# Start the node in SENSOR mode (no wallet prompt)
./target/release/w1z4rdv1510n-node api --addr 127.0.0.1:8090

# Check node health
curl http://localhost:8090/health

# Check what the equation matrix knows
curl http://localhost:8090/equations/report

# Find equations for a 2D topological sensor context
curl -X POST http://localhost:8090/equations/apply \
  -H "Content-Type: application/json" \
  -d '{"labels": ["anyon", "topological", "braid", "hall"], "dims": 2}'

# Start a domain script (chess example)
python scripts/chess_training_loop.py --max-games 2000

# Start the live visualizer
python scripts/live_viz_server.py --open
```

---

## Configuration

`node_config.json` controls the node. Key fields:

```json
{
  "node_mode": "SENSOR",
  "wallet": { "prompt_on_load": false },
  "streaming": { "enabled": true, "ultradian_node": true },
  "knowledge": { "enabled": true },
  "blockchain": { "enabled": true, "chain_id": "w1z4rdv1510n-l1" },
  "sensors": []
}
```

Apps register their own sensor descriptors at startup — the node config stays domain-agnostic.

---

## Hardware

Designed to run on modest desktops and scale across many nodes. The system measures its own hardware profile at startup (`HardwareProfile::detect()`) and adapts:

- Cooccurrence cap scales with available RAM
- Motif window caps enforce on constrained hardware
- Annealing uses lock-free `AtomicU64` counters
- All limits are derived from measurement, never hard-coded
