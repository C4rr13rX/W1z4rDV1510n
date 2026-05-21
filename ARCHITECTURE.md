# Brain Construction Kit — Architecture Spec

## Preamble

This is the canonical architectural specification. It describes a brain construction kit: software for building configurable, distributed, biologically-faithful cognitive substrates that run on commodity hardware (RAM + CPU + SSD, no GPU). The kit produces brains that observe sensor streams, learn through Hebbian plasticity, predict outcomes, and act through routed action neurons. Multiple brains can gossip motifs across a P2P network.

This is not an LLM. It is not a Q&A system. It is not a "neural network" in the modern ML sense. It is the substrate for cognition, designed to compound toward AGI through biologically-faithful constraints applied to honest-resolution sensor data, without an alignment layer between substrate and output.

The document has: substrate locks (non-negotiable structural decisions), the answer contract, configuration surfaces (tunable per-brain), the five subsystems in detail, distributed-network design, resource model, the API surface, the MIT-set tool selections, what gets ripped from the existing code, the developmental trajectory, and a phased implementation roadmap.

Any code in this repository that contradicts this spec is drift and is to be corrected.

---

## 1. Substrate Locks (Non-Negotiable)

These structural decisions are load-bearing. Everything else depends on them. Changing one means re-deriving most of the architecture.

### 1.1 Atom Identity = Raw Sensor Bits

An atom is the smallest indivisible feature of a sensor stream at the pool's bit resolution. Atom labels use base64-url-safe encoding as transport, never as identity transformation.

- No position augmentation (`txt:Vw@0`, `txt:Vw@1` — gone).
- No precomputed bigrams or trigrams as atoms — those are concepts that emerge.
- No IDF weighting at the atom layer.

The atom 'W' is one neuron. Its identity is the raw byte (or pixel bin, FFT bin, etc.). Position is encoded by firing order in the temporal stream and by concept neurons whose `members` field stores ordered references. Position is never encoded in atom identity.

### 1.2 Unified Neuron Type

One `Neuron` struct represents atoms, concepts, concept-of-concepts, binding concepts, and action neurons. Distinguished by:

- `kind: NeuronKind` — `Excitatory | Inhibitory | Modulatory` (Dale's principle: outgoing terminals must match the neuron's kind).
- `members: Vec<NeuronRef>` — empty for atoms, non-empty for concepts. Each `NeuronRef = (PoolId, NeuronId)`, so members can span pools (binding concepts).
- `terminals: Vec<Terminal>` — outgoing axon terminals. Each terminal carries `target: NeuronRef`, `weight: f32`, `consolidation: u8`, `last_fired_tick: u64`. Terminals can point within or across pools.

A concept neuron with members in multiple pools IS a binding concept. The same struct represents both within-pool hierarchical concepts and cross-pool bindings.

### 1.3 Per-Neuron Cross-Pool Wiring

No fabric-level routing table. Cross-pool wiring lives in `Neuron.terminals`. Propagation walks terminals uniformly; pool boundaries are organizational, not structural. Axons grow toward dendrites across pools through co-temporal Hebbian wiring on the moment buffer.

### 1.4 Time = Global Tick + Per-Pool Frame Rate

A single monotonic `tick: u64` advances globally. Each pool has a configured `frame_rate` (ticks per sensor frame) declaring its temporal resolution. All temporal binding, decay, and prediction operations reference ticks. `session_id` and string-keyed temporal scopes do not exist.

### 1.5 Hebbian + Decay + Prune as Only Learning Rule

Plasticity is local Hebbian with neuromodulator gating. No backpropagation, no global loss function, no attention mechanism, no embedding layer. Every tick, all terminal weights decay by ε. Terminals below `prune_floor` are pruned. Neurons with no remaining terminals are pruned.

Per-domain plasticity tracked by per-concept prediction-error EMA, used as a local plasticity multiplier.

### 1.6 API Returns Activation State + Grounding, Not Interpretations

The Brain API exposes `read_activation`, `integrate`, `action_outputs`. None return "the answer." They return state + grounding. Decisions and interpretations are caller responsibilities. There is no `/chat` decoder, no confidence-threshold gate, no Jaccard ranker at the API surface, no decoder fallback. **Output without a `GroundingReport` does not exist.**

### 1.7 Five Subsystems

Every brain contains:
- **Fabric** — Hebbian neural substrate (atoms, concepts, binding concepts, axon terminals)
- **EEM** — Environmental Equation Matrix (Kuzu-backed property graph of equations, variables, disciplines, motifs)
- **Annealer** — temporal frame predictor (simulated annealing over candidate next-frame activations)
- **Integration** — weighted combination of the above three, producing `AnswerWithGrounding`
- **Action** — routed output pool whose neuron firings produce external effects via deployment-specified channels

### 1.8 Distributed Motif Gossip on the Cluster Layer

Multiple brains share motif discoveries through P2P gossip on the existing cluster substrate (`crates/cluster`). Network EEM stays approximately coherent without strict consensus. Peer-augmented grounding contributes to integration with provenance.

---

## 2. The Answer Contract

**At every moment, the brain produces the best answer its current grounding allows, with honest disclosure of how grounded that answer is.**

This is the property that makes "best answer at any developmental stage" real rather than aspirational.

### 2.1 GroundingReport on Every Output

```rust
pub struct GroundingReport {
    pub input_atom_coverage: f32,        // What fraction of input atoms map to known atoms
    pub strongest_match_concept: Option<NeuronRef>,
    pub strongest_match_jaccard: f32,    // Concept↔input member-set overlap
    pub composition_used: Vec<NeuronRef>,
    pub fabric_confidence: f32,
    pub eem_confidence: f32,
    pub annealer_confidence: f32,
    pub integrated_confidence: f32,
    pub outside_grounding: bool,
    pub speculation_flag: bool,
    pub peer_contributions: Vec<(BrainId, Contribution)>,
}
```

### 2.2 Outside-Grounding Detection

When input atoms have low coverage, strongest concept Jaccard is below threshold, AND all subsystems return low confidence → integrated output is `Unknown { reason, what_would_help }`. Not a refusal. Not a fallback. An honest acknowledgment with a concrete path to grounding.

### 2.3 Speculation Flag

When the integrated output uses cross-pool composition or binding rather than direct concept match → `speculation_flag: true`. The brain can still produce composed answers (that's one of its powers), but it does not pretend retrieval and interpolation are the same.

### 2.4 Question-Asking via Action Pool

When grounding is insufficient and the brain wants to improve, it emits `RequestObservation { domain, examples_needed, why }` through the action pool. Routes to user / peer brains / configured external sources via deployment spec. This is the curiosity loop externalized.

### 2.5 Per-Region Adaptive Plasticity

Each concept tracks its recent prediction-error EMA. High error → high local plasticity (rapid learning). Low error → low local plasticity (stable knowledge). A brain can be infantile in new domains while mature in established ones simultaneously.

### 2.6 Network-Augmented Grounding

When local grounding is insufficient, integration can query peer brains via cluster gossip. Peer contributions enter integration weighted by each peer's accuracy track record on the domain. Provenance preserved in `peer_contributions`.

### 2.7 The Single Rule

**Every output must carry its grounding. Outputs without grounding are not outputs.** Enforced by structure, not by alignment. There is no "be honest" layer added on top — the substrate produces what the substrate has, the GroundingReport makes that visible, the caller sees both.

---

## 3. Configuration

### 3.1 BrainIdentitySpec

Declarative description of what the brain IS. Stable across environments. Reproducible. Versionable. Shareable.

```rust
pub struct BrainIdentitySpec {
    pub name: String,
    pub version: SemVer,
    pub pools: Vec<PoolSpec>,
    pub decay_rate: f32,
    pub prune_floor: f32,
    pub plasticity_baseline: f32,
    pub neuromodulator_gating: NeuromodulatorConfig,
    pub integration_weights_init: IntegrationWeights,
    pub eem_seed: PathOrUrl,
    pub baseline_policy: BaselinePolicy,        // PerPool | Global | Hybrid
    pub word_emergence_threshold: usize,        // delta-pattern recurrence count
    pub concept_emergence_threshold: usize,
    pub binding_emergence_threshold: usize,
}

pub struct PoolSpec {
    pub name: String,
    pub atom_encoding: AtomEncoding,            // bit resolution + decompose fn
    pub frame_rate: u32,                        // ticks per frame
    pub hot_tier_max: usize,
    pub memory_budget_mb: u32,
    pub cold_tier_path: Option<PathBuf>,
    pub kind: PoolKind,                         // SensoryInput | Action | Internal
}
```

### 3.2 BrainDeploymentSpec

Environment-specific configuration. Different per deployment.

```rust
pub struct BrainDeploymentSpec {
    pub storage_root: PathBuf,
    pub cluster_peers: Vec<PeerAddress>,
    pub sensor_sources: HashMap<PoolName, SensorSource>,
    pub action_endpoints: HashMap<ActionType, ActionRouter>,
    pub log_paths: LogPaths,
    pub resource_limits: ResourceLimits,
}
```

### 3.3 Brain Lifecycle

```rust
let identity = BrainIdentitySpec::load("brains/observer_v1.toml")?;
let deployment = BrainDeploymentSpec::load("deployment/local.toml")?;
let brain = Brain::launch(identity, deployment)?;

brain.observe(pool_id, frame, tick);
let answer = brain.integrate(query_context);
let actions = brain.action_outputs(since_tick);
brain.feed_outcome(action_id, outcome);
brain.checkpoint();
```

### 3.4 Pool Prototypes

The kit ships composable templates:
- `visual_pool` — image atoms at configurable resolution
- `auditory_pool` — FFT-bin atoms at configurable window and rate
- `language_pool` — byte atoms, monotonic frame index
- `motor_pool` — action atoms with deployment-defined routing
- `screen_pool` — pixel-bin atoms from screen capture
- `keystroke_pool` — keyboard event atoms with timestamps

Users compose pools to construct a brain identity. The kit provides a sensible default identity for a "general personal observer" brain.

---

## 4. The Five Subsystems

### 4.A Hebbian Fabric

**Data model:**
- `Neuron` (kind, members, terminals, last_fired_tick, prediction_error_ema)
- `NeuronPool` (collection of neurons, encode/decode contracts, `recent_atoms` ring buffer, `sequences` map, baseline activation EMA)
- `Fabric` (collection of pools, moment buffer)

**Emergence rules (all automatic, no manual `promote` calls):**
- **Atom birth:** new sensor input atom not in `label_to_id` → create atom neuron.
- **Concept birth:** atom sequence of length L observed N times in `recent_atoms` → create concept neuron whose members are those atoms in firing order.
- **Hierarchy birth:** concept sequence observed N times → create concept-of-concepts.
- **Binding birth:** multi-pool firing pattern observed N times in moment buffer → create binding concept whose members reference neurons in multiple pools.

**Plasticity:**
- Co-firing neurons strengthen the terminal between them.
- Every tick: all terminal weights × (1 − ε).
- Below `prune_floor`: terminal removed.
- Neurons with no remaining terminals: pruned (survival of fittest).
- Per-concept prediction-error EMA modulates local plasticity.
- Neuromodulator state gates learning rate per pool.

**Propagation:**
- Atoms fire from sensor input.
- Each firing neuron walks its terminals, depositing activation at target neurons (within or cross-pool).
- Decay per hop.
- No kwta sparsity gate. No confidence thresholds. Sparsity emerges from decay + pruning.

### 4.B Environmental Equation Matrix (EEM)

**Storage:** Kuzu property graph (MIT licensed).

**Schema:**
- Node types: `Equation`, `Variable`, `Discipline`, `Motif`, `Hypothesis`
- Edge types: `DEPENDS_ON`, `INSTANCE_OF`, `BOUND_TO`, `OBSERVED_AT`, `DERIVED_FROM`, `VALIDATED_BY`

**Operations:**
- Receive motif fingerprints from the fabric → insert `Motif` nodes, link `OBSERVED_AT` to relevant `Equation` nodes.
- Apply equation to context: bind variables, evaluate via `evalexpr`, return result + confidence.
- Hypothesize new equations when recurring motifs suggest missing rules; validate against future observations.
- Per-equation confidence updated based on validation track record.

**Feedback loop:** EEM outputs enter back through a designated sensor pool, training the fabric on observed equation-result patterns.

### 4.C Classical Annealer

**Storage:** temporal pattern indices over recent-history activation traces.

**Algorithm (argmin-backed simulated annealing):**
- State: candidate next-frame activation pattern for a target pool.
- Cost: prediction error against learned temporal patterns; lower cost = better-predicted continuation.
- Anneal: cooling schedule, accept-with-probability, converge to stable state.
- Output: converged predicted-next-frame + convergence energy (= confidence).

**Feedback:** observed actual-next-frame compared against prediction. Prediction error feeds back as a plasticity signal on the temporal pools' relevant concepts.

### 4.D Integration Layer

**Inputs at every integration tick:**
- Fabric: activation state + chosen concept(s) + grounding metrics
- EEM: applied equation result + confidence
- Annealer: predicted next-frame + convergence energy

**Algorithm:**
1. Compute `outside_grounding` flag: all subsystems below their grounding thresholds?
2. If outside grounding → return `Unknown { reason, what_would_help }`.
3. Otherwise → combine outputs weighted by per-context accuracy tracking.
4. Track per-context-type accuracy of each subsystem over time; weights evolve.
5. Detect speculation: is the answer direct retrieval or composition?
6. Assemble GroundingReport with full provenance.

**Output:** `AnswerWithGrounding { answer, grounding, confidence_tier, next_steps_if_ungrounded }`.

### 4.D.1 Inference is Hierarchical Confidence Traversal — NOT Flat Retrieval

**This is the central inference contract.  Every integration path
must honor it; flattening the hierarchy is the architectural defect
the Stage 7-13 work kept reintroducing.**

The substrate stores knowledge as a **layered hierarchy of neurons**:

  Layer 0 — atoms (raw sensor bits, one byte per atom in byte pools)
  Layer 1 — atom concepts: emerged sequences of atoms ("a-p-p-l-e")
  Layer 2 — concept-of-concept: emerged sequences of concepts
            ("apple-juice", "the-cat-sat")
  Layer 3+ — recursive; same emergence rule applies
  Binding pool — cross-pool composites; members may live in any pool
                 at any layer

Atoms are the substrate for learning concepts.  Concepts are the
substrate for learning meta-concepts.  Both directions are wired
automatically:

  - **Bottom-up**: each member atom gets a terminal to its parent
    concept at promotion time.  Activation flows up the hierarchy
    by propagation through those terminals.
  - **Top-down**: each concept gets terminals back to its member
    atoms at promotion time.  When a concept fires (e.g., by
    cross-pool feedback), its members re-fire.
  - **Collapse**: at observation time, the longest matching tail of
    `recent_atoms` collapses to the deepest concept that recognises
    it.  Both the constituent atoms AND the concept end up in
    `currently_firing` — that is the multi-layer signal subsequent
    inference must walk.

**The inference rule:**

When the integration layer reads the firing state to produce an
answer, it walks the hierarchy **top-down by layer depth, gated by
confidence**:

```
for layer in (deepest firing layer .. atoms):
    candidates = bindings/concepts in target pool whose members
                 intersect the firing set at this layer
    if any candidate has confidence ≥ θ_layer:
        return decode(that candidate)
# no layer was confident:
    construct from atom-sequence using within-pool transitions
    (the generative path — bytes the substrate never saw as a unit
     but assembles from accumulated co-occurrence)
```

**Consequences:**

  - **Trained input → trained output.**  When the input collapses
    to an emerged concept and that concept has a confident
    cross-pool binding, the deepest-layer path wins on the first
    try.  This is the "glorified lookup table" behavior — fast and
    exact, but only for inputs the substrate has fully crystallised.
  - **Untrained input → untrained output.**  When no concept layer
    has a confident match, the inference does NOT fail.  It drops
    to the atom layer and assembles the answer from the
    confidence-weighted superposition of every component atom's
    learned cross-pool response.  The within-pool atom-to-atom
    transition weights (also learned automatically) order the
    emission as a sequence.  The output is **constructed**, not
    retrieved — sequence-coherent bytes the substrate never
    produced as a whole during training, but every transition
    individually has Hebbian support.
  - **No autoregressive token loop.**  The generative path is one
    settling pass over the fabric followed by a confidence-ordered
    walk of the action-pool atom field.  No iterative next-token
    sampling.
  - **The coverage gate is layer-aware.**  When a firing concept's
    own constituent atoms also fire (the normal case after
    `collapse_tail_to_concept`), those atoms count as concept-layer
    evidence, not atom-layer noise.  Any gate that flattens this is
    an architectural bug.

**This is what mini-columns + concept neurons + atom-to-concept
propagation were designed to do.**  The substrate already wires it;
the integration layer must traverse it correctly.

### 4.E Action Layer

**Action pool:** a designated pool whose neurons are not sensor-input atoms but `ActionAtom`s. When an action neuron fires:
- Lookup `action_endpoints[neuron.action_type]` in deployment spec.
- Route firing to external channel (webhook, MQTT, agent call, human notification, etc.).
- Record action_id for outcome tracking.

**Outcome neurons:** external feedback enters its own pool. Outcome-vs-prediction error reinforces or weakens the synapses that led to the action. Functional dopamine.

`RequestObservation` is a first-class action type — the brain's externally-visible request for more grounding.

---

## 5. Distributed Network

### 5.1 Motif Gossip Protocol

Each brain emits motif fingerprints over the cluster on discovery. Receiving brains merge into their local motif index. Each brain maintains:
- Local motif index (own discoveries)
- Network motif index (received from peers)

Motif fingerprints include: pool, atom signature, recurrence count, source brain ID, observation timestamp, local confidence.

### 5.2 Network EEM Coherence

When a brain's EEM gains a new equation/link from motif feedback, the delta propagates via cluster gossip. Peer brains merge the delta with conflict resolution by per-equation confidence score. No strict consensus required.

### 5.3 Peer-Augmented Grounding

When local integration's GroundingReport indicates outside-grounding for some context, integration can query the network: "any brain that has grounding in this domain?" Peer brains respond with their fabric activation summary + EEM result + annealer prediction for the same input. Local integration combines all responses weighted by peer accuracy track record.

### 5.4 Global Health Baseline (Hybrid)

Per-pool baselines (long-term EMA of activation patterns) are local. The global health concept is a binding concept whose members reference per-pool baselines, plus optional aggregation with peer brain baselines via gossip. Deltas detected at both local and global levels.

---

## 6. Resource Model

### 6.1 Three-Tier Memory

- **Hot:** in RAM, currently-firing working set. Sized by `hot_tier_max` per pool.
- **Warm:** in RAM but eviction-eligible. LRU + activation-frequency hybrid.
- **Cold:** on SSD via redb. Bincode-serialized neurons. One DB file per pool.

### 6.2 Predictive Paging

Based on the `recent_atoms` buffer, the brain precomputes which concepts are about to fire and pre-loads them from cold into hot before propagation needs them. Demotes concepts whose member atoms haven't fired in K ticks.

### 6.3 Sleep Cycle

When activity rate falls below threshold for N ticks:
1. Flush hot tier to disk.
2. Run CLS replay (`multi_pool → slow_pool` consolidation).
3. Prune unconsolidated stale concepts from multi_pool.
4. Wake on next observation.

### 6.4 Memory Budget Enforcement

Each pool has `memory_budget_mb`. When approached, eviction runs more aggressively. Survival pruning runs continuously.

---

## 7. API Surface

```rust
impl Brain {
    pub fn create(identity: BrainIdentitySpec, deployment: BrainDeploymentSpec) -> Result<Brain>;
    pub fn launch(self) -> RunningBrain;
    
    pub fn observe(&self, pool: PoolId, frame: Frame, tick: u64) -> Result<ObservationEvent>;
    pub fn read_activation(&self, pool: PoolId, since_tick: u64) -> ActivationMap;
    
    pub fn integrate(&self, query_context: Option<Context>) -> AnswerWithGrounding;
    pub fn reframe(&self, input: Frame) -> Vec<Alternative>;
    
    pub fn action_outputs(&self, since: u64) -> Vec<ActionEvent>;
    pub fn feed_outcome(&self, action_id: ActionId, outcome: Observation);
    
    pub fn checkpoint(&self) -> Result<()>;
    pub fn sleep(&self) -> Result<()>;
    pub fn wake(&self) -> Result<()>;
    
    pub fn stats(&self) -> BrainState;
    pub fn developmental_profile(&self) -> DevelopmentalProfile;
}
```

No `/chat`. No `/query/integrated` as a decoder endpoint. Caller observes, reads activation or integrates, decodes outputs through each pool's encode/decode contract.

---

## 8. Implementation Tooling (MIT-set)

| Subsystem | Tool | License |
|---|---|---|
| Cold-tier storage | `redb` | MIT |
| Pool concurrency | `dashmap` + per-pool `RwLock` | MIT |
| Serialization | `bincode` | MIT |
| Zero-copy paging (later) | `rkyv` | MIT |
| EEM graph store | `kuzu` | MIT |
| EEM expression eval | `evalexpr` | MIT |
| Annealer framework | `argmin` | MIT/Apache |
| Pool-parallel propagation | `rayon` | MIT/Apache |
| SIMD inner math | `wide` | MIT/Apache/Zlib |
| Active-set bitmaps | `roaring` | MIT/Apache |
| Channels (moment bus, gossip) | `flume` | MIT/Apache |
| Image sensor | `image` | MIT/Apache |
| Audio FFT | `rustfft` | MIT/Apache |
| Audio/Video decode | shell out to FFmpeg (external binary) | n/a |
| PDF sensor | `lopdf` + `pdf-extract` | MIT |
| HTTP outbound (action routing) | `reqwest` | MIT/Apache |

### Stays Custom

Neuron / Pool / Fabric data model. Concept and binding-concept emergence rules. Hebbian update logic. Motif discovery. Delta detection. Word emergence. Integration weighting. Action routing policy. These are the architectural fingerprint.

---

## 9. What Gets Ripped From Current Code

- Position-augmented atoms (`position_augmented_atoms`, `position_and_ngram_atoms`)
- Precomputed bigrams/trigrams as atoms
- `kwta` / `sdr_sparsity` gates in propagation
- `top_input_match_concept` Jaccard ranker at API surface (moved to internal use only)
- `default_mp_confidence_threshold` + three-fraction confidence math at API surface
- `use_idf` / IDF weighting
- `paired_text` semantics in `/sensor/observe`
- `session_id` as temporal scope
- `char_chain` decoder fallback in `/chat`
- `/chat` as a decoder-producing endpoint
- `MultiPoolFabric.cross` HashMap (replaced by per-neuron `terminals`)
- `multi_pool_train_pair` / `_temporal` family (replaced by automatic co-firing wiring)

---

## 10. Developmental Trajectory

The same architecture handles every stage. What differs is the GroundingReport.

| Stage | Description | Typical GroundingReport |
|---|---|---|
| 0: Fresh | Just created, no observations | `outside_grounding: true`, `next_steps: [observe X]` |
| 1: Infant | Days of observation, atoms forming | Partial answers with high uncertainty, frequent `outside_grounding: true` |
| 2: Adolescent | Concepts forming, cross-pool wiring developing | Composed answers with moderate confidence, occasional `speculation_flag: true` |
| 3: Adult | Mature concepts, deep hierarchies, robust integration | High-confidence integrated answers with rich provenance, rare `outside_grounding` |

Per-domain maturity is tracked separately and exposed via `developmental_profile`. A brain can be Adult-level in one pool while Infant-level in another simultaneously.

---

## 11. Implementation Roadmap

**Phase 1: Substrate Refactor**
- Strip ripped components from Section 9.
- Implement atom identity (raw bits only).
- Refactor Neuron struct (unified, kind enum, NeuronRef members, terminals).
- Per-neuron axon terminals + uniform propagation.
- Pool encode/decode contracts.

**Phase 2: Emergent Mechanisms**
- Automatic concept formation from `recent_atoms`.
- Moment buffer at fabric level.
- Automatic cross-pool wiring from co-firing.
- Automatic binding-concept formation.
- Synapse decay + pruning per tick.

**Phase 3: Answer Contract**
- GroundingReport on every output.
- Outside-grounding detection.
- Speculation flag.
- Per-concept prediction-error EMA + per-region adaptive plasticity.

**Phase 4: Integration Layer**
- Three-subsystem combination.
- Per-context accuracy tracking.
- Provenance assembly.
- Replace `/query/integrated` cascade with new contract.

**Phase 5: EEM Migration**
- Adopt Kuzu backend.
- Schema definition.
- Port `seed_base_equations`.
- Implement motif feedback loop.

**Phase 6: Annealer**
- argmin-backed simulated annealing.
- Cost function over temporal patterns.
- Convergence energy as confidence.
- Prediction-error feedback to fabric plasticity.

**Phase 7: Action Layer**
- Action pool + `ActionAtom` type.
- Routing via deployment spec.
- Outcome neurons + reinforcement feedback.
- `RequestObservation` as first-class action.

**Phase 8: Distributed Network**
- Motif gossip over cluster.
- Network EEM coherence.
- Peer-augmented integration.
- Global health baseline aggregation.

**Phase 9: Resource Optimization**
- redb cold tier.
- Predictive paging.
- Sleep cycle automation.
- Memory budget enforcement.
- DashMap + per-pool RwLock.

**Phase 10: Configuration Polish**
- BrainIdentitySpec / BrainDeploymentSpec parsers.
- Pool prototypes library.
- Brain::launch lifecycle.
- Default identity for "general personal observer."

Phases 1–3 are load-bearing for everything else. Phases 4–7 produce the three-subsystem integrated brain. Phases 8–10 are scaling and polish.

---

## 17. Storage & Wake-Sleep (Supersedes §6)

The substrate is not a monolithic in-memory data structure that occasionally serializes itself. That model caps brain size at "fits in RAM" and breaks any "infinite brain" claim before it starts. The substrate is a **content-addressed, append-only, demand-paged neuron store**. RAM holds only the working set. Disk is the substrate of record. The brain *is its own database*.

The §6 three-tier model (`hot`/`warm`/`cold` with bincode-snapshot checkpoints) is superseded by this section. A monolithic `save_snapshot()` of a brain of arbitrary size cannot fit in physical RAM during serialization; this has been demonstrated empirically.

This section is **non-negotiable** for any brain that must scale beyond physical-RAM capacity.

### 17.1 Terminals Are the Content-Addressed Primitive

Terminals outnumber neurons by 100–500×. They dominate the substrate. The storage primitive is the terminal, not the neuron.

```rust
pub struct TerminalRecord {
    pub src_id:   NeuronId,          // 16 bytes, content-hash
    pub dst_id:   NeuronId,          // 16 bytes, content-hash
    pub strength: u8,                // 8-bit quantized; Hebbian noise floor is above ±1/256
    pub use_count_log2: u8,          // log2(use_count) clipped to [0, 255]
    pub last_fired_tick_delta: u32,  // delta from pool's current_tick; reset on overflow
    pub flags:    u8,                // {top_k, in_sketch, dirty, evicted}
}
// Total: 38 bytes per terminal (was 24 bytes uncompressed for live + ~8 GB for the live blob serialize)
```

A terminal's identity:
```
terminal_id = blake3(src_id || dst_id || pool_pair) [first 16 bytes]
```

Identity-from-content means: identical sensor history on two machines produces identical terminal IDs at identical disk offsets. This is the precondition for the cluster shape in §17.6.

**Tiered representation per neuron.** Each source neuron retains the top-K terminals (default K=32) as explicit `TerminalRecord`s. The long tail goes into a **count-min sketch** (Cormode & Muthukrishnan, 2005) sized per neuron based on its `freq_weight`. Strong recall stays deterministic; weak co-occurrence statistics survive lossily — which is what they already are in the substrate's noise floor.

**Adjacency channel.** Per-pool-pair existence (does src→dst exist at all, ignoring strength) is stored as a **roaring bitmap** (Lemire et al., 2016) over `(src_id, dst_id)` pairs. ~1 bit per terminal for the presence channel. Decouples "does this connection exist" from "how strong is it."

### 17.2 Storage Layout Follows Hebbian Co-Occurrence

Neurons that fire together must be stored together. Disk pages are organized so that **co-fired neurons land on adjacent pages**, making the OS prefetcher pull in their neighbors for free.

This is not a heuristic. It is the literal organizing principle of cortical columns (Mountcastle, 1997): physical proximity in cortex *is* feature proximity. The substrate's storage layout must match its computational invariant.

**Mechanism.** During the eviction sleep (§17.4), an online graph partitioner (Karypis & Kumar, 1998 — online METIS variant) reshuffles cold-tier pages to pack co-fired neurons into the same disk page. The partitioner's input is the live co-activation matrix from the moment buffer; no separate index. Runs at most once per sleep epoch, bounded by available CPU.

**For the cluster.** The same algorithm at larger granularity. Co-firing neurons land on the **same shard**. Data placement is a continuously-learned graph partition driven by the brain's own Hebbian statistics. Cluster shape emerges from training, not from config. This is the live-substrate equivalent of learned index structures (Kraska et al., 2018).

### 17.3 Bloom-Gated Neurogenesis

Before neurogenesis creates a new atom or concept neuron, the candidate `neuron_id` (which is already a content hash per §1.1) is checked against a per-pool **counting Bloom filter** (Bloom, 1970; Fan et al., 2000) covering all known neuron_ids.

- Bloom returns **"maybe present"** → page-in attempt against the on-disk index. On miss, fall through to neurogenesis.
- Bloom returns **"definitely absent"** → neurogenesis fires immediately. No scan, no read.

Bloom parameters: target false-positive rate 1e-4 at 100M neuron capacity. ~14 bits/key. Per-pool bloom fits in ~175 MB at the 100M scale — kept fully in RAM. Resized via doubling when load factor exceeds threshold.

**Why this matters.** Existing neurogenesis code performs an O(N) scan over `label_to_id` to check for duplicates. At 2.4M neurons, that's already milliseconds per observe; at 100M it's hundreds of ms. The Bloom check is O(k) where k is the hash count (~7) — constant in brain size.

### 17.4 Eviction Sleep = Background Actor, Not a Mutex-Held Phase

The legacy `sleep_prune` holds a write lock over every terminal in every pool while scanning the full graph. At 463M terminals this exceeds 30 s reliably. Replaced by two **background actors**:

**Eviction sleep.** Drains a `cold_candidate: Channel<NeuronRef>` populated lazily by the hot path. For each candidate:
1. Read its current state (under brief per-neuron lock, not pool-wide).
2. Append a consolidated `NeuronRecord + TerminalRecord[]` to the pool's append log.
3. Update the offset index (atomic CAS on a memory-mapped table).
4. Drop from the working-set cache.
5. Bloom slot stays — only changes from "present in RAM" to "present on disk."

Wall-time bound: O(working_set_eviction_count), not O(total_terminals). The 30 s timeout on 463M terminals becomes <100 ms on a 100K-neuron working-set sweep.

**Replay sleep.** Independent actor. Samples moments from the moment buffer by **energy-minimizing weighting** (§17.7), pages-in the participating neurons (which may trigger evictions of less-salient ones), and re-fires them at low rate. Reuses the existing annealer crate (Friston, 2010-grounded annealing energy).

Both actors run cooperatively scheduled, yielding between per-record steps so HTTP handlers stay responsive. `/stats`, `/observe`, `/chat` never block on sleep.

### 17.5 Brain-Emitted Salience — The Brain Learns What to Forget

Each `Neuron` gains a learned `salience: f32` field. The substrate's own training rule writes to it. The eviction policy reads from it.

```rust
pub struct Neuron {
    // ... existing fields ...
    pub salience:       f32,    // 0.0 (forgettable) ... 1.0 (must retain)
    pub salience_ema:   f32,    // EMA of recent salience updates
}
```

**How salience is learned.** Two coupled signals:
1. **Reward/precision-modulated co-firing** (cf. Schultz, 2007 dopaminergic gating; Frémaux & Gerstner, 2016 three-factor plasticity). When the brain's output matches reality (high precision on `/integrate`), terminal weights AND each participating neuron's salience are incremented in lockstep. Salience tracks "this neuron contributed to a correct prediction."
2. **Compression utility under replay** (cf. McClelland et al., 1995 CLS; Tononi & Cirelli, 2014 synaptic homeostasis hypothesis). During replay sleep, a neuron whose absence would substantially raise free-energy across replayed moments has its salience boosted. Salience tracks "this neuron is structurally necessary for what I've learned."

The combined signal is the substrate's own answer to *"which of my own neurons matter for my future self?"* — the cognitive prerequisite for tiered retention.

**Why this is the load-bearing innovation.** Modern ML systems retain everything (expensive) or use externally-imposed heuristics (LRU, LFU, age). Biological brains use salience-tagged consolidation (Frankland & Bontempi, 2005). No published runtime, to our knowledge, lets the *substrate itself* dictate its own retention policy. This section is the closest computational analogue to systems consolidation in the mammalian hippocampal-cortical circuit.

**Eviction reads salience as one signal among several.** The actual policy is dynamical (§17.8); salience alone does not determine eviction. It biases the priority queue.

### 17.6 Cluster Shape — Anti-Entropy via Merkle Diff

Because every `NeuronId` and `TerminalId` is a content hash, every pool has a deterministic Merkle root over its append log:
```
pool_root = merkle(neurons || terminals || sequences || salience_index)
```

Cluster sync between nodes is **anti-entropy**, not RPC log replay (Lakshman & Malik, 2010 Cassandra; Demers et al., 1987 epidemic algorithms):
1. Compare root hashes → identify divergent subtrees.
2. Recurse into mismatched subtrees → identify divergent records.
3. Exchange only the divergent records.

A node offline for one hour catches up in **O(diff)** bandwidth, not O(events). A node that has never seen another's training catches up in **O(unique-records)**.

**Sharding.** Neurons are assigned to nodes by **co-activation graph partitioning** (§17.2), not by `neuron_id` range hashing. The partition is reshuffled during sleep epochs based on observed co-firing. The shard map is itself part of the Merkle-rooted state, so all nodes converge on the same map without consensus protocol.

**Read path across nodes:** `GET /neuron/{id}` from the shard-owner. The brain becomes a distributed content store with Hebbian learning bolted onto the read/write path.

**Provenance.** Signed `(pool_root, timestamp)` tuples are cryptographic snapshots — anyone can prove a brain was in state X at time T by showing the root, signature, and a path through the Merkle tree. Reproducible-research-grade snapshot provenance.

### 17.7 Replay = Energy Minimization on a Learned Landscape

Replay sleep does not uniformly re-fire hot moments. It samples moments weighted by their predicted **reduction in free energy** under the substrate's current state (Friston, 2010 active inference; Hinton et al., 1995 Helmholtz wake-sleep).

Concretely, for moment `m` in the buffer:
```
replay_weight(m) ∝ exp( -β · free_energy_delta(m) )
free_energy_delta(m) = predicted_free_energy_after_replay(m) - current_free_energy
```

where `free_energy` is computed by the existing annealer crate (no new mechanism; it already implements simulated-annealing energy estimation over moment configurations). Replay preferentially picks moments that **resolve current prediction error**.

This is the wake-sleep algorithm (Hinton et al., 1995) implemented as the persistence loop:
- **Wake** = observe + propagate + write append log.
- **Sleep** = generate from substrate + measure divergence from observed + replay the high-divergence moments + adjust salience.

The brain's storage tier is also its consolidation loop. They are not separate processes.

### 17.8 Self-Tuning Eviction (Dynamical-System Principle Extended)

The same `ControlMode / ControlSignal / ControlState` architecture from the substrate's existing control loops governs the storage tier:

```rust
pub struct StorageControlState {
    pub working_set_pressure:        f32,   // current_ws_bytes / ws_budget_bytes
    pub cache_hit_rate:              f32,   // EMA of working-set hits per /observe
    pub replay_value_score:          f32,   // EMA of free-energy reduction per replay tick
    pub salience_distribution_entropy: f32, // H(salience) over working set
}

pub struct StorageConfig {
    pub eviction_mode:  ControlMode,   // gates eviction aggressiveness
    pub replay_rate:    ControlMode,   // gates replay throughput
    pub shard_rebalance_threshold: ControlMode,  // when to re-partition
    pub bloom_resize_pressure: ControlMode,
}
```

**Policy emerges from signals, not from knobs.** Examples:
- High pressure + low entropy of salience (brain is focused on few items) → evict aggressively from the long tail.
- High pressure + high entropy (brain is exploring) → evict more cautiously; bias replay to consolidate first.
- Low hit rate + low replay_value → working set is thrashing; shrink it.
- High hit rate + high replay_value → working set is productive; let it grow up to budget.

The storage policy is a dynamical system driven by the same control architecture that governs everything else. No retention knobs.

### 17.9 Crash Safety — The Training Loop Is the WAL

Every `observe` that mutates state appends to its pool's log before the in-memory mutation is exposed. There is no separate "checkpoint" — persistence is continuous.

```
observe(frame):
    parse_atoms(frame) -> atom_ids
    for each atom_id:
        if bloom.maybe_present(atom_id):
            try wake_neuron(atom_id) -> neuron
            else: neuron = neurogenesis(atom_id); append_log(neuron); bloom.insert(atom_id)
        else:
            neuron = neurogenesis(atom_id); append_log(neuron); bloom.insert(atom_id)
        update_terminals(neuron); append_log(terminals_delta)
        // in-memory state is now updated; durable
    advance_tick()
```

Crash recovery is rolling forward the WAL since last checkpoint marker (Mohan et al., 1992 ARIES; Rosenblum & Ousterhout, 1991 log-structured filesystems). Worst-case data loss is the last buffered append (≤4 KB / ≤1 ms at typical training rates).

**The legacy `Brain::checkpoint()` becomes a flush + barrier**, not a serialize. O(buffer) cost, sub-millisecond. Hold no Mutex. `POST /checkpoint` is a no-op acknowledgment in steady state.

### 17.10 Implementation Sequence — Each Stage Ships Independently

Every stage compiles, passes existing tests, and preserves the canonical 100% baseline (toddler 32/32 + greetings 7/7 + K-12 16/16 + multi_fact 5/5 + integrate 32/32 + OOV 3/3). Regression on any of these blocks promotion to the next stage.

1. **§17.1 + §17.9 first:** `crates/brain/src/store/` module. Traits `TerminalStore`, `NeuronStore`, `PoolIndex`. Default impl: mmap'd append log + offset index + per-pool Bloom. Brain's existing API unchanged; storage is plumbed underneath. New tests: crash-replay round-trip, deterministic content addressing across runs.

2. **§17.3:** Bloom-gated neurogenesis. Replaces the O(N) label scan. Test: `neurogenesis_throughput_bench` shows constant-time insert past 10M neurons.

3. **§17.4 (eviction half):** Working-set LRU around `Pool`. Cold-candidate channel. Background actor evicts. `sleep_prune` decomposed into `evict_actor` + `replay_actor`. Test: `/observe`, `/stats`, `/chat` stay responsive (<10 ms 99p) under continuous eviction.

4. **§17.5:** Add `salience` field; wire reward/precision-modulated update path. Initial GA-tuned weighting. Test: ablation — turning salience-driven eviction off must regress on multi-corpus brains but not on the canonical baseline (which fits entirely in working set).

5. **§17.7:** Replay-actor uses annealer for energy-minimization weighting. Test: replay reduces measured free-energy on held-out moments vs. uniform replay.

6. **§17.6:** Merkle root per pool; `RemoteNeuronStore` shard impl; anti-entropy sync protocol. Test: two-node cluster reproduces canonical 100% baseline; one-node-offline catch-up via Merkle diff.

7. **§17.8:** `StorageControlState` + `StorageConfig` wired through the existing ControlMode plumbing. Removes static thresholds.

Each stage is independently committable. The brain never goes offline. **The full-15-corpus training that previously killed the brain mid-checkpoint runs end-to-end after stage 1 alone.**

### 17.11 Prior Art — What This Stands On, What It Extends

| Mechanism | Prior art | Extension here |
|---|---|---|
| Content-addressed object store | Merkle (1987); Git (Torvalds, 2005); IPFS (Benet, 2014) | Applied to neurons + terminals; identity = sensor content |
| Counting Bloom filter | Bloom (1970); Fan et al. (2000) | Gates a *learning rule* (neurogenesis), not just lookup |
| LSM / mmap kv store | Rosenblum & Ousterhout (1991); LMDB (Chu, 2011); RocksDB (Facebook, 2013) | Payload is a learning neuron; eviction policy is biological |
| Wake-sleep / Helmholtz | Hinton et al. (1995) | Wake-sleep *is* the persistence loop, not a separate algorithm |
| Free-energy active inference | Friston (2005, 2010) | Replay sampling weighted by free-energy delta |
| Complementary learning systems | McClelland, McNaughton & O'Reilly (1995); Kumaran et al. (2016) | CLS as a literal tiered storage architecture |
| Synaptic homeostasis | Tononi & Cirelli (2014) | SHY-driven downscaling as eviction signal |
| Hippocampal replay | Wilson & McNaughton (1994); Foster & Wilson (2007) | Replay queue = sleep actor; pages neurons by replay priority |
| Salience-tagged consolidation | Frankland & Bontempi (2005); O'Reilly & Frank (2006) | Salience as a runtime-readable eviction signal — *brain dictates retention* |
| Online graph partitioning | Karypis & Kumar (1998) METIS | Online + driven by Hebbian co-activation matrix |
| Learned data placement | Kraska et al. (2018) learned indexes | Placement learned continuously from substrate firing |
| WAL / ARIES | Mohan et al. (1992) | Training observe = WAL write; crash recovery = WAL replay |
| Anti-entropy / Merkle sync | Demers et al. (1987); Cassandra (Lakshman & Malik, 2010) | Cluster brain sync by Merkle diff over pool roots |
| Roaring bitmap | Lemire et al. (2016) | Per-pool-pair adjacency channel |
| Count-min sketch | Cormode & Muthukrishnan (2005) | Long-tail terminal compression below top-K |

**Novel as a combination:** a learning substrate where (a) the unit of storage is the synapse with quantize+sketch tiering, (b) physical layout follows Hebbian co-occurrence, (c) the brain emits its own retention signal, (d) wake-sleep is the persistence algorithm, (e) cluster sync is Merkle anti-entropy over the same content-addressed objects, and (f) all storage policy is the same dynamical system as the substrate.

No published system has these together. The systems contributions (terminal-as-CAS-object + Hebbian-driven page layout + salience-gated retention + Helmholtz-as-WAL) and the computational-neuroscience contributions (CLS-as-runtime + SHY-as-eviction + free-energy-weighted replay) are independently publishable.

---

## 12. Closing

This document supersedes ad-hoc design decisions in the codebase. Code that contradicts it is drift. The architecture is committed to:

- **Substrate purity** — no alignment layer between substrate and output.
- **Honest grounding** — every output carries its report.
- **Continuous adaptation** — no training cutoff, no frozen weights.
- **Distributed cognition** — network of brains, not a single model.
- **Biologically-faithful constraints** — Hebbian only, decay-and-prune, multimodal grounding, embodied via sensors.

The brain factory produces brains. The brains produce activation states with grounding. The caller decides what those mean. The substrate stays out of the way.
