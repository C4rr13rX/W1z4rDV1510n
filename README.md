# W1z4rDV1510n

A distributed intelligence node that learns to physically describe its environment — not by classification, but by growing structure from what it observes. CPU and RAM native. No GPUs required.

---

## What this is — for neuroscientists and AI architects

Most neural networks are fixed-topology function approximators trained offline. This is a different thing: a **living, spiking-inspired neural fabric** that grows its own architecture in RAM, trained online from any sensor stream.

The learning stack implements the current neuroscience canon in software:

- **SDR / k-Winners-Take-All (kWTA)** — After each propagation hop, only the top 2% of activated neurons survive. This enforces cortical sparsity (1–5%) in real neural tissue, eliminates saturation, and gives each pattern a unique sparse code. Without this, Hebbian accumulation drives all neurons toward uniform activation and the fabric loses discriminative power.
- **STDP with asymmetric long-term potentiation/depression** — `hebbian_pair(a, b)` is direction-aware. `a` (pre-synaptic, fires first) → `b` (post-synaptic) gets LTP (×1.0). `b` → `a` gets LTD (×−0.3). The result: the fabric encodes causal order. "photosynthesis→glucose" is a stronger edge than "glucose→photosynthesis."
- **Homeostatic synaptic scaling** — Every 500 steps, each neuron's outgoing weights are multiplicatively scaled toward a target mean activation of 0.10, correcting at 4% per pass. Preserves relative weight ratios while preventing runaway Hebbian growth. This is the computational equivalent of Turrigiano homeostatic plasticity.
- **Per-neuron EMA activation tracking** — Each neuron maintains a slow exponential moving average of its own activation (`τ ≈ 2000 steps`), the input signal that drives homeostatic scaling — the same mechanism as sliding threshold models of intrinsic excitability.
- **Neuromodulator system (DA / NE / ACh / serotonin)** — Four neuromodulator concentrations per pool, each with distinct decay dynamics. Acetylcholine gates plasticity multiplicatively. Norepinephrine boosts effective learning rate up to 3×. Dopamine enables retrograde potentiation. All decay toward tonic baseline each step.
- **Three-factor Hebbian / dopamine retrograde potentiation** — Neurons with high activation trace are tagged at dopamine release. `flush_dopamine_potentiation()` applies `Δw = lr × dopamine_tag × weight` to their outgoing synapses. This is the computational correlate of reward-modulated STDP — the reward signal (dopamine) potentiates the connections that led to the outcome.
- **Predictive coding** — `propagate_predictive()` implements a first-order hierarchical predictive coding loop. Hop 0 propagates full activation. Subsequent hops propagate only the residual `(actual − prediction).max(0.0)`. Neurons that activate exactly as predicted pass zero signal upstream — only surprise propagates. Prediction EMAs update online each training pass (`α = 0.10`).
- **Neuromodulator-gated learning rate** — In `train_weighted_with_meta()`: `effective_lr = lr_scale × ACh × (1 + NE × 2.0)`. When the hypothesis queue fires a NE spike (failed QA gate), the next training run runs at elevated learning rate — attention sharpens on surprising inputs.
- **Dual memory systems (CLS theory)** — The QA store is the hippocampus: fast one-shot episodic retrieval. The NeuronPool is the neocortex: slow statistical learning. They interact — a QA hit seeds a specific pool activation pattern, combining the precision of episodic memory with the generalization of distributed representations.
- **Multi-pathway convergence inference** — `propagate_combined()` accepts question labels (weight 1.0) and QA answer labels (weight = `qa_conf × 1.5`) simultaneously, biasing propagation toward the answer's territory before the pool activates. Confidence gate: `qa_confidence ≥ 0.5 AND active_question_neurons > 0`.
- **Hypothesis → research feedback loop** — Questions that fall below the QA confidence gate are queued in the hypothesis queue. `research_agent.py` polls the queue, fetches Wikipedia and ArXiv answers, ingests them via `/qa/ingest` + `/media/train`, and resolves them via `/hypothesis/resolve`, which triggers a DA flush — reward signal for correct prediction resolution.

**For architects:** The system is designed to be observable at every level. Every neuron records its influence history. Every synapse carries provenance. The neuromodulator state is readable via API. The hypothesis queue is an explicit epistemic state — the system knows what it doesn't know and acts on it.

---

## Benchmarks

| Task | Score | Notes |
|------|-------|-------|
| QA retrieval accuracy | **0.951** | Hebbian Q&A fabric, `POST /qa/query` |
| Chat / generate quality | **0.630** | `POST /chat`, `POST /neuro/generate` |

The chat/generate score reflects the current training corpus. Both endpoints use the dual-memory (CLS) architecture: QA store for high-confidence fast retrieval, neuro pool for generalization.

---

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
┌──────────────────────────────────────────────────────────────────────┐
│                     w1z4rd_node  (one binary)                        │
│                                                                      │
│  :8080  Neuro API ── NeuroRuntime ── EquationMatrix ── QaRuntime    │
│         Media API ── /media/train  /media/playback  /neuro/propagate│
│         Chat API  ── /chat  /neuro/generate                         │
│         Hyp  API  ── /hypothesis/queue  /hypothesis/resolve         │
│  :8090  Node  API ── ClusterNode  ── HashRing ── OtpRegistry        │
│                                                                      │
│  KnowledgeRuntime ── HierarchicalMotifs ── FabricTrainer            │
│  P2P Gossip / Data Mesh / Blockchain Layer                           │
└──────────────────────────────────────────────────────────────────────┘
          ▲                 ▲                  ▲                ▲
          │                 │                  │                │
   ┌──────────────────────────────────────────────────────────┐
   │          Multimodal encoder library  (crates/core)        │
   │  ImageBitsEncoder  AudioBitsEncoder  TextBitsEncoder      │
   │  MotionBitsEncoder  KeyboardBitsEncoder                   │
   │  shared 8×8 grid vocabulary — img:, aud:, txt:, mov:, key:│
   └──────────────────────────────────────────────────────────┘
          ▲                 ▲                  ▲                ▲
          │                 │                  │                │
   chess_training     rtsp_pose_bridge   rss_topic_bridge  w1z4rd-dashboard
   _loop.py           .py                .py               (GUI client binary)
   research_agent.py  train_obstacle.py  playback_obstacle.py
   (app / script)     (app / script)     (app / script)
```

Both APIs start automatically when the node binary launches — no separate service process. The dashboard is a lightweight GUI binary that polls the node APIs; it does not need to run on the same machine.

**The node is domain-agnostic. Scripts and apps define the domain.** The node has no knowledge of chess, poses, pixels, or cursor movement — it sees labels and co-occurrences. The encoder library translates raw sensor data (images, audio, text, mouse trajectories, keystrokes) into the node's label vocabulary. Everything is composable because everything speaks the same language.

---

## Cluster

Multiple nodes join into a single virtual node that partitions the neural fabric across all available hardware. From the fabric's perspective there is one brain; from the hardware's perspective each machine holds a slice of it.

```
Machine A (coordinator)            Machine B                 Machine C
┌─────────────────────┐           ┌───────────────┐         ┌───────────────┐
│  w1z4rd_node        │  SIGIL    │  w1z4rd_node  │  SIGIL  │  w1z4rd_node  │
│  :8080 :8090        │ ◄────────►│  :8080 :8090  │◄───────►│  :8080 :8090  │
│  ring owner 0–42    │  :51611   │  ring 43–85   │ :51611  │  ring 86–127  │
└─────────────────────┘           └───────────────┘         └───────────────┘
```

**How it works:**

| Concern | Mechanism |
|---------|-----------|
| Node discovery & pairing | One-time OTP (`WORD-NNNN` format, argon2id, TTL-bounded, single-use) |
| Label partitioning | Consistent hash ring — 150 virtual nodes per machine (Blake2b512); labels routed to their owner without replication |
| Coordinator election | Bully algorithm — highest-priority (oldest) surviving node wins; coordinator is a full participant, not a dedicated master |
| Failure detection | Heartbeat every 5 s; node declared lost after 15 s; election triggered when coordinator silent for 20 s |
| Transport | Length-prefixed JSON frames over raw TCP; `DashMap` connection pool per node |
| Default port | **51611** — SIGIL in leet speak (5=S, 1=I, 6=G, 1=I, 1=L) |
| Scale | Unbounded — 2 to N nodes; the ring rebalances on every join/leave |

**Cluster commands:**

```bash
# Machine A — start the cluster on a specific LAN IP, prints a join OTP
w1z4rd_node cluster-init --bind 192.168.1.10:51611

# Machine B — join using the OTP printed by machine A
w1z4rd_node cluster-join --coordinator 192.168.1.10:51611 --otp EMBER-4821

# Any node — print cluster topology and ring status
w1z4rd_node cluster-status --node 192.168.1.10:51611

# Coordinator — generate a fresh OTP for a new joiner (coordinator must be running)
w1z4rd_node cluster-otp
```

**Startup scripts** (in `scripts/`):
- `start_cluster.bat` — start this machine as coordinator (Windows, prints OTP)
- `start_worker.ps1` — join an existing cluster (Windows, prompts for OTP)

OTPs are single-use and expire in 10 minutes by design. Generate a fresh one with `cluster-otp` or by restarting `cluster-init` before each new worker joins.

The neural fabric scales horizontally without changing the API. A script POSTing to `/neuro/train` on any node trains the distributed fabric; `/neuro/snapshot` on any node returns the global view.

---

## What shifts a prediction outcome

Every prediction is the result of multiple components voting simultaneously. Understanding the full stack matters when tuning behavior or writing scripts.

| Component | Mechanism |
|-----------|-----------|
| **STDP / asymmetric Hebbian weights** | Co-occurrence history with directional bias — pre→post synapses (causal order) are potentiated; post→pre synapses are depressed (LTD ×−0.3). Strongest causal paths win at inference |
| **SDR / k-Winners-Take-All** | After each propagation hop, only the top 2% of active neurons survive. Enforces cortical sparsity, eliminates pool saturation, gives each concept a unique sparse code |
| **Homeostatic synaptic scaling** | Every 500 steps, per-neuron outgoing weights are rescaled multiplicatively toward target activation (0.10). Preserves relative ratios while preventing runaway growth |
| **Neuromodulators (ACh / NE / DA / serotonin)** | ACh gates plasticity multiplicatively; NE raises effective learning rate up to 3× on surprising inputs; DA enables retrograde potentiation of recently active synapses; serotonin provides tonic stability baseline |
| **Dopamine retrograde potentiation** | On hypothesis resolution, DA is released at the reward signal level. Neurons with high activation trace are tagged; `flush_dopamine_potentiation()` strengthens their outgoing synapses — three-factor Hebbian (activity × activity × reward) |
| **Predictive coding residuals** | `propagate_predictive()` — only prediction error (surprise) propagates beyond the first hop. Neurons activating as expected pass zero signal; unexpected activations dominate propagation |
| **Dual memory (CLS) — QA fast path** | QA store (hippocampus analog) provides high-confidence episodic answers. Confidence ≥ 0.5 gates into multi-pathway inference; output is capped at QA answer word count to prevent pool noise |
| **Multi-pathway convergence inference** | `propagate_combined()` seeds from both question labels (weight 1.0) and QA answer labels (weight = qa_conf × 1.5) simultaneously — answer territory biases pool propagation before it activates |
| **NE spike on hypothesis queue** | When QA gate fails, NE is released (0.75 units). Next training run runs at elevated learning rate — the system applies stronger correction to its own uncertainty |
| **Mini-columns** | Neuron groups that collapse to single concept neurons over time; once promoted they fire as a unit with high confidence |
| **Working memory carry** | Cosine similarity between consecutive frames sets a dynamic carry factor; similar frames reinforce prior context, dissimilar frames reset it |
| **Temporal motif priors** | `HierarchicalMotifRuntime` mines recurring sequences at unbounded depth; the proposal kernel pulls candidates toward centroids of motif classes the fabric expects next |
| **Classical annealer** | Searches minimum-energy state configurations; temperature is coherence-modulated — confident fabric → fast convergence; uncertain fabric → high-temperature exploration |
| **Environmental Equation Matrix** | 282 equations across 24 disciplines vote on active sensor labels; matching equations reinforce associated labels; hypothesis gaps suppress confidence when nothing matches |
| **Surprise-weighted replay** | Persistent mispredictions are replayed at higher frequency — the system applies stronger correction to its worst errors |
| **Peer node learning** | In cluster mode, neuro snapshots from other nodes train local weights at low learning rate; distributed observations converge into shared knowledge |
| **ResourceMonitor** | Under CPU/RAM pressure, batch sizes and update frequency drop — slower adaptation, more conservative predictions |

---

## Core components

### Neural Fabric (`NeuroRuntime`)

A spiking-inspired Hebbian neural pool implementing the full neuroscience plasticity stack in RAM. No matrix operations — inference is a propagation event through grown synaptic connections.

**Learning mechanisms (all active simultaneously):**
- **STDP** — asymmetric `hebbian_pair(a, b)`: a→b gets LTP, b→a gets LTD. Directional knowledge encoding.
- **kWTA sparsification** — top 2% active neurons per hop survive. Sparse distributed representations.
- **Homeostatic scaling** — per-neuron multiplicative weight correction every 500 steps; targets mean activation 0.10.
- **Three-factor Hebbian / dopamine retrograde** — `apply_dopamine()` tags trace-active neurons; `flush_dopamine_potentiation()` strengthens their outgoing synapses on reward.
- **Neuromodulator-gated plasticity** — `effective_lr = lr_scale × ACh × (1 + NE × 2.0)` in every training call.
- **Prediction EMA** — per-neuron `prediction` field updated online (α=0.10); drives predictive coding propagation.
- **Max-weight cap (4.0)** — `add_synapse()` clamps at 4.0; eliminates unbounded accumulation.

**Inference methods:**
- `propagate(seed_labels, hops)` — passive synapse walk with kWTA at each hop; returns label→strength map
- `propagate_weighted(pathways, hops)` — multi-pathway propagation with per-seed weights
- `propagate_combined(question_labels, qa_answer_labels, qa_conf, hops)` — dual-memory convergence inference
- `propagate_predictive(pathways, hops, min_activation)` — predictive coding; only surprise propagates beyond hop 0
- `cross_stream_activate(labels, target_stream, hops)` — cross-modal inference through Hebbian connections
- `reconstruct_sequence(frames, target_stream, hops)` — temporal sequence reconstruction with cosine carry factor

**Runtime API:**
- `release_neuromodulator(kind, amount)` — spike DA / NE / ACh / serotonin
- `flush_dopamine()` — apply retrograde potentiation to tagged synapses
- `neuromodulator_state()` — read current concentrations
- `observe_snapshot()` — ingest any `EnvironmentSnapshot`
- Influence/provenance tracking: every neuron records which streams and data shaped it
- Double-buffered propagation, binary-search synapse lists, EMA co-occurrence tracking, dirty-flag mini-columns — all hardware-adaptive

### Multimodal encoder library (`crates/core/src/streaming/`)

Translates raw sensor data into the node's label vocabulary. The encoders live in the core library — **not** in the node binary. Scripts import them, encode data, and POST labels to the node API. The node never sees pixels, waveforms, or keystrokes directly.

All encoders share the same default **8×8 spatial grid**. A cursor at zone `(3,2)`, an image feature at zone `(3,2)`, and a text span at zone `(3,2)` all emit the same zone label family. Hebbian learning connects them automatically.

| Encoder | Input | Label prefixes | Notes |
|---------|-------|----------------|-------|
| `ImageBitsEncoder` | JPEG/PNG bytes or raw RGB | `img:z{x}_{y}`, `img:h{n}`, `img:edge{dir}_z{x}_{y}` | HSV histogram + Sobel edges per grid zone |
| `AudioBitsEncoder` | PCM f32 or WAV bytes | `aud:freq{n}`, `aud:amp{n}`, `aud:freq{n}_t{t}` | Hann-windowed DFT; no external FFT dependency |
| `TextBitsEncoder` | `TextSpan` structs or plain `&str` | `txt:word_{w}`, `txt:role_{r}`, `txt:phon_{ng}`, `txt:zone_x{n}_y{n}` | Layout-aware: captures font size, role, emphasis, indent, position |
| `MotionBitsEncoder` | `Vec<MotionSample {x,y,t,click}>` | `mov:zone_x{n}_y{n}`, `mov:endpoint_x{n}_y{n}`, `act:click` | Also `decode_target()` for inference |
| `KeyboardBitsEncoder` | `Vec<KeyEvent {key,ctrl,shift,alt,t}>` | `key:k_{name}`, `key:combo_{mods}{key}`, `txt:word_{w}` | Cross-modal: emits `txt:word_*` for typed words |

**Layout is data.** A heading at the top of a page emits `txt:role_heading + txt:zone_x0_y0 + txt:size_large + txt:emph_bold`. Position, font, role, and emphasis are all Hebbian-connected to whatever content appears there. PDF structure is spatial signal, not decoration.

**Cross-modal associations emerge from co-training.** Feeding image + text + motion together in one `POST /media/train` call Hebbian-connects all three. At inference time, seeding from text labels fires the associated motion zones — no decoder needed.

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
- Persisted across restarts; hot-tier caching for high-confidence pairs
- API: `POST /qa/ingest`, `POST /qa/query`

### Hypothesis Queue and Research Feedback Loop

When `neuro_ask` or `neuro_generate` fails the QA confidence gate (< 0.5), the question is added to the hypothesis queue and a norepinephrine spike is released (NE = 0.75). The `research_agent.py` script runs as a background service:

1. `GET /hypothesis/queue` — fetch open questions
2. Fetch Wikipedia REST API + ArXiv Atom API for each question
3. `POST /qa/ingest` — ingest the answer into the QA store
4. `POST /media/train` — train the neuro pool on the answer text
5. `POST /hypothesis/resolve` — mark the hypothesis resolved with a confidence score

On resolution, dopamine is released proportional to confidence and `flush_dopamine_potentiation()` runs — the connections that led to the correct prediction are retroactively strengthened. This is the computational equivalent of the hippocampal-cortical consolidation loop.

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

Both APIs (`:8080` neuro, `:8090` node) run in either mode. Cluster commands are always available regardless of mode.

---

## Node API endpoints

All endpoints are on `:8080`. Start the node with the `api` subcommand:
```bash
./bin/w1z4rd_node --config node_config.json api --addr 0.0.0.0:8080
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Node status and uptime |
| `/neuro/train` | POST | Feed an `EnvironmentSnapshot` to the neural fabric |
| `/neuro/snapshot` | GET | Current activation, predictions, motifs, top influences |
| `/neuro/propagate` | POST | Feed seed labels → returns all labels that fire above threshold; cross-modal inference |
| `/neuro/generate` | POST | Generate text from the neuro pool using QA gate + pre-computed activation map decode |
| `/media/train` | POST | Encode and co-train one or more modalities: `image`, `audio`, `text`, `motion`, `action`, `full` |
| `/media/playback` | POST | Given goal text + optional screenshot → predict action zone (where to move cursor / what to click) |
| `/chat` | POST | Chat endpoint — dual-memory CLS inference: QA fast path + pool generalization |
| `/hypothesis/queue` | GET | Open hypothesis entries (questions that failed the QA confidence gate) |
| `/hypothesis/resolve` | POST | Resolve a hypothesis with an answer and confidence; triggers dopamine flush |
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

The annealer predicts environment states by searching over candidate configurations, accepting or rejecting them by energy. It is not a standalone optimizer — it runs inside the neural motif architecture.

**Motif-driven proposals.** Instead of perturbing from the current state randomly, the proposal kernel reads `temporal_motif_priors` from the neural fabric's snapshot and pulls proposed symbol positions toward the centroids of motif classes the fabric expects to see next. `prediction_pull` and `working_memory_pull` are configurable per-deployment; no values are hard-coded.

**Coherence-modulated temperature.** The cooling schedule is not a fixed curve. After each iteration the annealer reads `mean_prediction_confidence` from the neuro snapshot and sets `T_eff = T_schedule / (confidence + 0.1)`. When the fabric is confident — tight motif alignment, low prediction error — temperature drops and the annealer converges fast. When the fabric is uncertain — novel patterns, conflicting motifs — temperature stays high and the annealer explores. The fabric's epistemic state is the thermostat.

**Motif-prior energy term.** The `motif_transition` energy term penalizes proposed states that contradict the fabric's learned sequence expectations. States where symbols sit in positions consistent with high-prior motif classes get lower energy. States matching no learned motif expectation receive a novelty penalty proportional to distance from all known centroids. The energy landscape is shaped by experience, not only physics constraints.

- Homeostasis: if min-energy stagnates across `patience` iterations, temperature is reheated and mutation rate boosted — prevents premature convergence
- Population resampling with ESS threshold: when particle diversity collapses, resample and mutate
- Classical and quantum-inspired annealing; calibration hooks for remote quantum hardware
- AtomicU64 lock-free acceptance counters; hardware-adaptive parallelism

### 2) Neural fabric and cross-stream inference

#### Training
- `observe_snapshot()` — ingest any `EnvironmentSnapshot`; converts symbols to zone/role/metadata labels, Hebbian updates across all activated neurons
- `train_weighted_with_meta()` — weighted Hebbian update with full metadata context; neuromodulator-gated `effective_lr = lr_scale × ACh × (1 + NE × 2.0)`; prediction EMA updated each pass
- Neurons accumulate up to 16 influence records (weakest evicted when full); records from the same stream+label set merged and strength averaged

#### Inference
- `cross_stream_activate(labels, target_stream, hops)` — propagate hop-by-hop through Hebbian synapses, collect activations matching target stream
- `reconstruct_sequence(frames, target_stream, hops)` — temporal sequence reconstruction with dynamic carry factor from cosine similarity between consecutive frames
- `propagate(seed_labels, hops)` — passive synapse walk with kWTA per hop; does not mutate pool state
- `propagate_predictive(pathways, hops, min_activation)` — only prediction error propagates beyond hop 0

#### Snapshot
`NeuroSnapshot` exposes: `active_labels`, `active_networks`, `minicolumns`, `centroids`, `network_links`, `temporal_predictions`, `temporal_motif_priors`, `top_influences`, `active_streams`, `active_meta_labels`, `working_memory`

### 3) Environmental Equation Matrix

The EEM bridges the gap between raw sensor patterns and physical interpretation. As the neuro fabric fires labels, the EEM surfaces candidate equations governing the observed phenomenon. It is a complete map of modern physics — 282 equations across 24 disciplines — compiled into the node so that any sensor stream can be interpreted through the lens of physical law.

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

Motifs are the currency the annealer and the EEM both trade in. The hierarchical motif runtime mines recurring temporal sequences from the behavior substrate, promotes them through levels, and exposes predictions that everything else builds on.

- Body-schema adapters map multimodal sensors into shared latent state + action vector
- Change-point segmentation plus fixed windows for stable coverage
- Motif discovery using normalized edit distance (Levenshtein), graph signatures, and MDL costs
- Attractor detection: when transition entropy drops below threshold, the sequence is flagged as an attractor — a stable regime the environment is locked into
- `next_predictions(last_id)` — given the most recent motif, returns learned transition probabilities to successors across all levels; this is what the proposal kernel samples from
- `mean_transition_entropy()` — fabric certainty scalar used by the annealer as coherence signal for temperature modulation
- `window_tail(n)` — current observation window tail for seeding predictions
- Meta-motifs promoted through unbounded levels: level-0 sequences become level-1 meta-motifs, which become level-2 meta-motifs, until the signal exhausts
- Behavior graph coupling metrics: proximity, coherence, phase-locking, transfer-entropy proxies

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
- **Autonomous research loop** (`scripts/research_agent.py`): polls `/hypothesis/queue`, fetches Wikipedia + ArXiv, ingests answers, resolves hypotheses with DA reward signal

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

## Script application patterns

The node exposes a generic API. Scripts define the domain. Below are four categories of scripts that can be written against the existing API with no changes to the node.

### Conversational / LLM-style

Feed conversation pairs through `/qa/ingest` (`question → answer`). At inference time `/chat` or `/qa/query` fires question tokens through the dual-memory CLS architecture — QA store for high-confidence fast retrieval, neuro pool for generalization and cross-domain bleed. `/neuro/generate` uses the pre-computed activation map decode (single propagation pass + rank sort) for free-form generation.

This is associative retrieval + predictive coding, not autoregressive generation. Responses are activated from trained state rather than generated token-by-token. Suitable for factual recall, domain Q&A, and follow-up questions that share vocabulary with prior exchanges.

### Code assistance

Code has strong motif structure — function signatures, boilerplate, API call sequences promote quickly through `HierarchicalMotifRuntime`. A bridge script tokenizes source files into labeled symbols (function names, keywords, identifiers at line/column positions) and feeds them as `EnvironmentSnapshot` sequences. Known problem→solution pairs go through `/qa/ingest`.

**This is compositional generation, not retrieval.** Given a plain English instruction the fabric has never seen — "create a vehicle tracker class" — the process is:

1. The instruction fires labels in the language stream: `vehicle`, `tracker`, `class`, `position`, `update`, `state`
2. Cross-stream activation propagates those labels into the code stream via Hebbian synapses: `class` → `encapsulation`, `__init__`, `self`, `attribute`; `tracker` → `list`, `append`, `query`, `history`; `position` → `x`, `y`, `float`
3. Mini-columns that have collapsed from repeated co-activation fire as units — `class + __init__ + self + attributes` becomes a single concept-level activation
4. The motif runtime supplies structural sequence priors: `class_definition → attribute_block → constructor → methods` is a high-confidence learned transition
5. The annealer searches for the minimum-energy configuration of code symbols that satisfies all activated constraints simultaneously — it constructs a class that has never existed in the training data

The result is genuinely novel — assembled from the intersection of everything the instruction activated across all trained streams. The mechanism differs from autoregressive LLM token prediction (learned conditional probability distributions over tokens), but the compositional capability is equivalent: novel code from learned structural knowledge, not memorized retrieval.

### Scientific and engineering Q&A

The strongest native fit. The EEM already contains the relevant equations; the textbook pipeline (`textbook_scripts/`) already extracts Q&A pairs from OpenStax PDFs; `/knowledge/ingest` handles JATS/NLM papers. A script ingests domain material, then at query time combines `/chat` with `/equations/apply` — the answer comes from both the Hebbian fabric and equation matching.

Cross-domain transfer is a genuine advantage: a thermodynamics question activates energy-balance connections built from every other domain trained simultaneously. When no equation matches well, the system reports a hypothesis gap explicitly rather than returning a confident wrong answer. The research agent then fetches an answer and the system learns from it autonomously.

### GUI / screenshot understanding

The node does not process raw pixels. A bridge script handles vision: take a screenshot → pass through an OCR/vision tool (Tesseract or an API call) → extract UI elements with bounding boxes → convert each to a symbol with position and properties → `POST` to `/neuro/train` or use as query context.

```json
{
  "id": "button_OK",
  "type": "UI_ELEMENT",
  "position": { "x": 412, "y": 308, "z": 0 },
  "properties": { "label": "OK", "element_type": "button", "app": "chrome" }
}
```

The fabric learns which UI element arrangements co-occur with which operations; the motif system learns GUI workflows across Windows, macOS, Linux, Android — any platform whose screens can be OCR'd. Requires training exposure to each application before generalizing to it.

### Multimodal screen navigation (cursor / interaction learning)

The `ImageBitsEncoder`, `TextBitsEncoder`, and `MotionBitsEncoder` share an 8×8 grid, which means a cursor trajectory toward pixel zone (3,2), an image feature at zone (3,2), and text at zone (3,2) all train the same neuron family simultaneously through `POST /media/train` with `modality: "full"`.

**Training**: a script captures screenshots, goal text, and mouse trajectories, then POSTs them together. The fabric Hebbian-connects text labels (what the goal says) with motion labels (where the cursor went).

**Inference** (`POST /media/playback`): given goal text and optionally a screenshot, the node propagates from the text's discriminative labels through the learned synapses and returns the predicted action zone and whether a click is expected. The script converts the zone back to pixel coordinates and acts.

**Scope boundary**: The node learns label associations. The *script* operates Playwright, takes screenshots, moves the mouse, and clicks. The node never drives UI directly.

```python
# Training loop (train_obstacle.py pattern)
result = httpx.post("/media/train", json={
    "modality": "full",
    "data_b64": screenshot_b64,      # -> img: labels
    "text": "click the red button",  # -> txt: labels
    "motion": trajectory_points,     # -> mov: labels, act:click
    "lr_scale": 1.0,
})

# Inference (playback_obstacle.py pattern)
result = httpx.post("/media/playback", json={
    "goal": "click the red button",
    "hops": 1,
})
# -> {"action": {"type": "move_and_click", "zone_x": 0, "zone_y": 1, ...}}
```

**Anchor training** is the key technique for discrimination: alongside full-trajectory training, post a focused call with only the unique discriminative word and the endpoint point at `lr_scale: 5.0`. This creates a strong direct association that survives the noise from shared words ("click", "the", "button") that appear in every goal.

```python
# Anchor: just the color word + endpoint, high weight
httpx.post("/media/train", json={
    "modality": "full",
    "text": "red",                          # unique word only
    "motion": [{"x": fx, "y": fy, "t_secs": 0.0, "click": True}],
    "lr_scale": 5.0,                        # outweighs shared-word noise
})
```

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
| `scripts/train_obstacle.py` | Screen navigation | Screenshot + goal text + mouse trajectory via `/media/train`; anchor pairs for discrimination |
| `scripts/playback_obstacle.py` | Screen navigation | Goal text → `/media/playback` → predicted zone → Playwright cursor action |
| `scripts/research_agent.py` | Hypothesis resolution | Polls `/hypothesis/queue`, fetches Wikipedia + ArXiv, ingests via `/qa/ingest` + `/media/train`, resolves via `/hypothesis/resolve` |
| `scripts/train_full_pipeline.py` | Full training benchmark | End-to-end QA + chat quality scoring |

**Adding a new domain**: translate your data to `EnvironmentSnapshot`, `POST` to `/neuro/train`, optionally send to `/qa/ingest` and `/equations/apply`. The node learns from it alongside everything else.

For multimodal domains (image + text + motion): use `ImageBitsEncoder`, `TextBitsEncoder`, `MotionBitsEncoder` from the encoder library to convert raw data to labels, then `POST /media/train`. The node's neural fabric is the same one everything else trains.

---

## Running

```bash
# Build everything
cargo build --release

# Copy to bin/
cp target/release/w1z4rdv1510n-node  bin/w1z4rd_node      # Linux/macOS
cp target/release/w1z4rdv1510n-node.exe  bin/w1z4rd_node.exe  # Windows

# Start the node (requires 'api' subcommand)
./bin/w1z4rd_node --config node_config.json api --addr 0.0.0.0:8080

# Check node health
curl http://localhost:8080/health

# Chat with the node
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "what is photosynthesis"}'

# Generate from the neuro pool
curl -X POST http://localhost:8080/neuro/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "photosynthesis", "max_tokens": 20}'

# Check the hypothesis queue (questions the node couldn't answer with confidence)
curl http://localhost:8080/hypothesis/queue

# Check what the equation matrix knows
curl http://localhost:8080/equations/report

# See open hypothesis gaps (sorted by cross-node corroboration)
curl http://localhost:8080/equations/gaps

# Propagate from seed labels (cross-modal inference)
curl -X POST http://localhost:8080/neuro/propagate \
  -H "Content-Type: application/json" \
  -d '{"seed_labels": ["txt:word_red"], "hops": 1}'

# Run the autonomous research agent (resolves hypothesis queue entries via Wikipedia + ArXiv)
python scripts/research_agent.py

# --- Dashboard GUI (separate binary, connects to the node API) ---
./bin/w1z4rd-dashboard                          # localhost defaults

# --- Obstacle course demo (screen navigation from natural language) ---

# Terminal 1: node
./bin/w1z4rd_node --config node_config.json api --addr 0.0.0.0:8080

# Terminal 2: train (opens browser, records mouse trajectories, POSTs to /media/train)
pip install playwright httpx && playwright install chromium
python scripts/train_obstacle.py --reps 10

# Terminal 2: playback (opens browser, asks /media/playback, moves cursor to predicted zone)
python scripts/playback_obstacle.py --auto

# --- Run chess training + live visualizer together ---

# Terminal 1: node
./bin/w1z4rd_node --config node_config.json api --addr 0.0.0.0:8080

# Terminal 2: chess sensor script (feeds board states to the node as EnvironmentSnapshots)
python scripts/chess_training_loop.py --max-games 8000

# Terminal 3: live visualizer (opens browser automatically)
python scripts/live_viz_server.py --board-file logs/chess_live_board.json --port 8765 --open
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
