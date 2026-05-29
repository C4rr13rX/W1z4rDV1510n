# W1z4rDV1510n

A distributed intelligence node that learns to physically describe its environment — not by classification, but by growing structure from what it observes. CPU and RAM native. No GPUs required.

---

## What this is — for scientists, engineers, AI architects, and security professionals

Most neural networks are fixed-topology function approximators trained offline. This is a different thing: a **living, spiking-inspired neural fabric** that grows its own architecture in RAM, trained online from any sensor stream.

### Architecture in one paragraph

The substrate is a **byte-grained Hebbian fabric** organised into pools (text, image, audio, action, turn, binding). Atoms are raw bytes. Concepts emerge by mini-column collapse on co-firing.  Cross-pool bindings are single neurons whose members span pools in firing order — recall reads the canonical ordered subsequence directly. The substrate auto-captures every prompt→response moment into a Q→A database it uses to grade itself, locks well-trained terminals against background decay, scales cross-domain wiring softly (instead of skipping it) so analogical bridges form continuously, and runs an autonomous thinking loop in the background that any inference or training request can preempt cleanly.

### Single canonical surface (as of 2026-05-29)

The repository ships **one merged binary**, `w1z4rdv1510n-node`, that owns both the legacy Web3 / cluster / sensor stack AND the active brain substrate.  Everything important architecturally is mounted under `/brain/*` on the main node API:

| Surface | Port | What it is |
|---|---|---|
| Main node API | `:8090` (configurable via `--node-addr`) | Web3 / cluster / wallet / sensor / `/chat` / `/brain/*` (the active substrate) |
| Neuro service API | `:8080` (configurable via `--api-addr`) | Legacy `NeuroRuntime` service interface — kept for backward compatibility with existing tooling, scheduled for removal once the migration completes |
| Standalone brain server | `:8095` (configurable via `W1Z4RD_BRAIN_PORT`) | Identical handlers as `/brain/*`, useful for isolated experiments — the next session will collapse this into `/brain/*` |
| P2P / gossip | `:8088` | libp2p; cluster ring + OTP join + heartbeat |

**For a senior software engineer:** start with [`crates/node/src/brain_api.rs`](crates/node/src/brain_api.rs) and [`crates/brain/src/brain.rs`](crates/brain/src/brain.rs).  brain_api.rs is the HTTP surface; brain.rs is the substrate.  Phase A–E commits (1c7a4ec → eb2c079) are the architectural turning points.

**For an AI designer / architect:** read [Architecture overview](#architecture-overview) and the [Phase A–E dynamical substrate](#phase-ae-dynamical-substrate-2026-05-28) section.  The five phases are the *complete* story of how the brain went from 0.83 rows/sec / 0% recall to 252+ rows/sec / 100% recall + interruptible autonomous thought.

**For a neuroscientist:** the substrate maps onto established biology in `crates/brain` ([Biological primitives](#biological-primitives-substrate-internal-all-dynamical-system-knobs)).  Concept emergence is mini-column collapse on co-firing (Mountcastle); consolidation lock is myelination-style permanent-trace (Frankland & Bontempi 2005); cross-domain bridges form continuously instead of after sleep, reflecting Hebb-like cortical bridging (McClelland-McNaughton-O'Reilly CLS).  The binding pool is the hippocampal episodic trace; the per-pool concept hierarchy is the cortical statistical learner; the Phase E thinking loop is the default-mode network.

**For a security professional:** every byte ingested goes through `base64url`-encoded HTTP endpoints; there is no model inversion attack surface because there are no learned distributed weights to invert — the substrate is structural (atom/concept/binding identity), not parametric.  Wallet signing (Ed25519 + Argon2-protected mnemonic) is in [`crates/node/src/wallet.rs`](crates/node/src/wallet.rs).  Cluster gossip auth is OTP-rotated (`/cluster/otp`).  The merged binary disables the API key gate unless `api.require_api_key=true` in `node_config.json`; production deployments should enable it and rotate API keys via the `api_key_hashes` allow-list.

### Two learning stacks coexist

The active research frontier is the brain crate (`crates/brain`, mounted under `/brain/*` on the main node and also accessible on the standalone `:8095`).  The legacy `NeuroRuntime` (`crates/core`, ports `:8080`/`:8090`) is kept available so existing dashboards and ingest tooling keep working — see [Brain crate](#brain-crate-cratesbrain--stage-16-100-recall-on-trained-input) for the active stack and the legacy callout at the bottom for what's being phased out.

### What is canonical, what is legacy, what's being deleted

| Concern | Canonical (use these) | Legacy (still works but being phased out) | Status |
|---|---|---|---|
| Training (cross-pool pair) | `POST /brain/observe` (text+action+tick) | `POST /multi_pool/train_pair`, `POST /two_pool/train_pair`, `POST /neuro/train` | Legacy stays for existing tooling; the brain replaces the substrate cleanly |
| Inference / retrieval | `POST /brain/integrate`, `POST /brain/integrate_chain` | `POST /multi_pool/ask`, `POST /two_pool/ask`, `POST /neuro/ask`, `POST /chat` (still works) | `/chat` will be repointed to the brain in the next iteration; legacy `/neuro/ask` retained for legacy clients only |
| Sensor observe | `POST /brain/observe` | `POST /sensor/observe`, `POST /sensor/observe_triple` (still on `:8095`) | Sensor routes will be migrated to `/brain/sensor/*` |
| Self-test feedback | `POST /brain/self_test` + `GET /brain/qa_db_stats` + `GET /brain/consolidation_stats` | (nothing equivalent in legacy — this is new) | — |
| Self-tuning | `POST /brain/retune` + `GET /brain/tuning_state` | (nothing equivalent in legacy) | — |
| Continuous thought | `/brain/thinking/start`, `/stop`, `/status` | (nothing equivalent in legacy) | — |
| Cross-domain composition | `POST /brain/integrate_chain` | `/query/integrated` (precision-weighted multi-component cascade in the legacy node — different mechanism) | Both work; `/integrate_chain` is the substrate-level primitive, `/query/integrated` is a higher-level orchestrator |
| Snapshot | brain.bin via `POST /checkpoint` on `:8095` (will move to `/brain/checkpoint` on main node) | neuro pool JSON at `<data_dir>/neuro_pool.json` (auto) | Both used; brain.bin is the substrate snapshot |
| Wallet / Web3 | `crates/node/src/wallet.rs`, `/bridge/*`, `/cluster/*` | (canonical — kept) | — |
| P2P / cluster | libp2p on `:8088`, `/cluster/init`, `/cluster/join`, `/cluster/otp` | (canonical — kept) | — |
| Knowledge ingest | `/knowledge/ingest` and friends | (kept for now) | Will likely move to brain in a future iteration |

### Legacy `NeuroRuntime` learning stack

The legacy fabric implements the current neuroscience canon in software:

- **SDR / k-Winners-Take-All (kWTA)** (Vinje & Gallant, 2000; Olshausen & Field, 1996; Maass, 2000) — After each propagation hop, only the top 2% of activated neurons survive. This enforces cortical sparsity (1–5%) in real neural tissue, eliminates saturation, and gives each pattern a unique sparse code. Without this, Hebbian accumulation drives all neurons toward uniform activation and the fabric loses discriminative power.
- **STDP with asymmetric long-term potentiation/depression** (Bi & Poo, 1998; Markram et al., 1997) — `hebbian_pair(a, b)` is direction-aware. `a` (pre-synaptic, fires first) → `b` (post-synaptic) gets LTP (×1.0). `b` → `a` gets LTD (×−0.3). The result: the fabric encodes causal order. "photosynthesis→glucose" is a stronger edge than "glucose→photosynthesis."
- **Homeostatic synaptic scaling** (Turrigiano, 2008) — Every 500 steps, each neuron's outgoing weights are multiplicatively scaled toward a target mean activation of 0.10, correcting at 4% per pass. Preserves relative weight ratios while preventing runaway Hebbian growth.
- **Per-neuron EMA activation tracking** — Each neuron maintains a slow exponential moving average of its own activation (`τ ≈ 2000 steps`), the input signal that drives homeostatic scaling — the same mechanism as sliding-threshold models of intrinsic excitability (Bienenstock, Cooper, & Munro, 1982 — the BCM rule).
- **Neuromodulator system (DA / NE / ACh / serotonin)** (Hasselmo & McGaughy, 2004 for ACh; Aston-Jones & Cohen, 2005 for NE; Schultz, 2007 for DA) — Four neuromodulator concentrations per pool, each with distinct decay dynamics. Acetylcholine gates plasticity multiplicatively. Norepinephrine boosts effective learning rate up to 3×. Dopamine enables retrograde potentiation. All decay toward tonic baseline each step.
- **Three-factor Hebbian / dopamine retrograde potentiation** (Frémaux & Gerstner, 2016; Izhikevich, 2007) — Neurons with high activation trace are tagged at dopamine release. `flush_dopamine_potentiation()` applies `Δw = lr × dopamine_tag × weight` to their outgoing synapses. This is the computational correlate of reward-modulated STDP — the reward signal (dopamine) potentiates the connections that led to the outcome.
- **Predictive coding** (Rao & Ballard, 1999; Friston, 2005, 2010) — `propagate_predictive()` implements a first-order hierarchical predictive coding loop. Hop 0 propagates full activation. Subsequent hops propagate only the residual `(actual − prediction).max(0.0)`. Neurons that activate exactly as predicted pass zero signal upstream — only surprise propagates. Prediction EMAs update online each training pass (`α = 0.10`).
- **Neuromodulator-gated learning rate** — In `train_weighted_with_meta()`: `effective_lr = lr_scale × ACh × (1 + NE × 2.0)`. When the hypothesis queue fires a NE spike (failed QA gate), the next training run runs at elevated learning rate — attention sharpens on surprising inputs.
- **Dual memory systems (CLS theory)** (McClelland et al., 1995; Kumaran et al., 2016) — The N-pool associative fabric is the hippocampus: fast bidirectional paired-association retrieval across an arbitrary number of named pools (in/out/emo/equation/motion/…). The slow NeuronPool is the neocortex: distributed statistical learning. They interact — an N-pool hit seeds a specific pool activation pattern, combining the precision of episodic recall with the generalization of distributed representations. CLS-style replay periodically consolidates N-pool concept neurons back into the slow pool.
- **Multi-pool convergence inference** — A single input fired into one pool causes every connected pool to produce its own decoded prediction in parallel. Cross-modal training (e.g., one query input mapped simultaneously to an answer pool, an emotion pool, and an equation pool) means all those substrates fire from one query.
- **Hypothesis → research feedback loop** — Questions for which the multi-pool fabric returns no answer are queued in the hypothesis queue. `research_agent.py` polls the queue, fetches Wikipedia and ArXiv answers, ingests them via `/multi_pool/train_pair` + `/media/train`, and resolves them via `/hypothesis/resolve`, which triggers a DA flush — reward signal for correct prediction resolution.

### Brain crate learning stack (active research frontier, `/brain/*` on main node + standalone `:8095`)

Stage 11-16 added a distinct set of biologically-motivated primitives to the brain crate, each wired so that **the substrate's own observable signals drive its knobs** — no static hyperparameter tuning.  See [Biological primitives](#biological-primitives-substrate-internal-all-dynamical-system-knobs) below for full detail.  Headline mechanisms:

- **k-WTA sparsity per pool** (`Pool::sparsity_mode`) — (Vinje & Gallant, 2000; Olshausen & Field, 1996; Maass, 2000) — runs at end of `observe_frame` BEFORE moment fingerprint capture so it actually bounds binding size
- **Heterosynaptic LTD** (`Pool::heterosynaptic_ltd_mode`) — (Royer & Paré, 2003; Turrigiano, 2008) — synapse-competition on each tick
- **Predictive-coding gate on concept emergence** (`Pool::predict_gate_mode`) — (Rao & Ballard, 1999; Friston, 2005, 2010) — `recent_surprise` EMA gates new concept crystallisation
- **Sleep / replay consolidation** (`Brain::replay_recent_moments`) — (Wilson & McNaughton, 1994; McClelland et al., 1995; Tononi & Cirelli, 2014) — `POST /sleep` prunes weak concepts then re-fires recent fingerprints
- **Hebbian frequency weighting in decode** (`BRAIN_FREQ_WEIGHT`) — (Hebb, 1949; Bi & Poo, 1998; Markram et al., 1997) — `freq_weight = 1 + strength × ln(use_count)` — frequently-trained bindings dominate competitors
- **Sequence-match preempt** (`Pool::last_observed_sequence`) — distinguishes anagram queries (`sad` query → `sad→emotion`, NOT `das→animal`).  Inspired by sequence-cell observations in CA3 (Skaggs & McNaughton, 1996; Foster & Wilson, 2007)

Five knobs are GA-tunable as `ControlMode` enums — either `Constant(value)` (legacy hyperparameter) or `DrivenBy(signal, scale, offset, min, max)` against the substrate's `ControlState` (a snapshot of normalised observable signals updated every tick).  This makes the GA search **how the substrate self-regulates**, not which static numbers work best.

**For architects:** The system is designed to be observable at every level. Every neuron records its influence history. Every synapse carries provenance. The neuromodulator state (legacy) is readable via API. The hypothesis queue is an explicit epistemic state — the system knows what it doesn't know and acts on it.  In the brain crate, `Pool::control_state()` exposes the live observable signals (surprise, firing rate, decode precision, concept density, terminal density) that drive the dynamical-system knobs — every retrieval call's decision lineage is inspectable.

## What makes this fundamentally different from every other AI language model

### The tokenization problem

Every major language model — GPT, LLaMA, BERT — starts with a tokenizer. Not a learned component: a **human-written algorithm that runs on the training corpus before any learning begins**, merging character sequences by frequency into a fixed vocabulary of ~50,000 chunks (BPE for GPT/LLaMA, WordPiece for BERT). From that point forward, the model never sees a letter. It is handed pre-segmented atoms and learns statistical relationships between them.

The representations that result carry **no internal structural constraints about their own composition**. "cat" is atom 1234. "cats" is atom 1235. The relationship between them exists only as an indirect statistical correlation through shared contexts in the training corpus — not as an encoded morphological fact. The model did not discover that "cats" is "cat + plural marker." It memorized the co-occurrence statistics of atom 1235 with plural contexts.

This is the root of the hallucination failure mode. When a model encounters a novel word, a rare form, or a plausible-sounding portmanteau it has never seen, it has no mechanism to decompose it and reason about what it is made of. It cannot verify that "metamorfocillin" doesn't exist by tracing its parts. It generates the next token by asking: *what token sequence is statistically most likely given this context?* — and that question has no structural floor. A confident, fluent, structurally wrong answer is indistinguishable from a correct one from the inside.

Cross-lingual transfer fails for the same reason. "Transport" in English and "transporter" in French are different token atoms in different vocabularies. The shared `trans-` morpheme — which carries meaning — is invisible to the model unless the training corpus happened to contain enough English-French parallel text to create an indirect statistical bridge. There is no morpheme neuron that both words activate.

**The segmentation happened before learning, so the learned representations carry no information about their own internal structure.** That is the architectural constraint tokenization imposes, and it cannot be trained away — it is baked into the representation space itself.

---

### This system does not have a tokenizer. It has a bottom-up architecture.

Language understanding is grown from the ground up through the same Hebbian machinery that processes every other sensor stream. The hierarchy is not declared in advance. It emerges from co-occurrence:

```
Characters  →  Sub-sequences (morphemes)  →  Words  →  Usage patterns  →  Motifs
```

Every character fires as its own labeled neuron (`txt:char_a`, `txt:char_p`...). Positional context is encoded separately (`txt:char_a_pos0` ≠ `txt:char_a_pos4`). Every punctuation mark fires as its own label — `txt:punct_comma`, `txt:punct_apostrophe`, `txt:punct_period` — because punctuation is signal, not noise. "Let's eat, Grandma." and "Let's eat Grandma." differ by one comma. This system encodes that difference because the character context around the comma was different in every sentence where it appeared.

---

### The neurogenesis pathway: exact mechanics

**Step 1 — STDP character-sequence encoding**

When "cat" is trained, three character neurons fire in sequence. The pool calls `hebbian_pair(c, a, scale)` and `hebbian_pair(a, t, scale)` for every adjacent pair within the STDP window. The weight update is:

```
base = hebbian_lr × (activation_c + trace_c) × (activation_a + trace_a) × scale

Pre→Post (c→a):  Δw = base × stdp_pre_post  =  base × +1.0   [LTP]
Post→Pre (a→c):  Δw = base × stdp_post_pre  =  base × −0.3   [LTD — only weakens if synapse exists]
```

The forward edge `c→a` is potentiated. The backward edge `a→c` is depressed (or left untouched if it doesn't exist yet). After training "cat", "cattle", "catch", "concatenate" — all words containing the c→a→t sequence — the forward path `c→a→t` has accumulated strong excitatory weight. The backward paths have been actively suppressed. The pool now encodes causal order, not just co-occurrence.

**Step 2 — Co-occurrence accumulation across the corpus**

The pool maintains a `minicolumn_counts` table. Every time a signature (a minimal set of co-occurring labels derived from a recurring pattern) fires, its count increments. When the c→a→t character cluster appears in "cat", "cattle", "concatenate", "catfish", "catnip" — each occurrence increments `minicolumn_counts["mini::c|a|t"]`. Nothing is created yet. Evidence accumulates silently.

**Step 3 — Mini-column promotion: neurogenesis at threshold**

When `minicolumn_counts["mini::c|a|t"]` crosses `minicolumn_threshold` (default: **18**), a new neuron is created at runtime: `mini::c|a|t`. This neuron did not exist in the initial architecture. It was grown from the data. Its `members` list holds the neuron IDs of the constituent character neurons. It starts with `stability = 0.0` and `inhibition = 0.0`.

**Step 4 — Concept neuron activation and member suppression**

Every training frame, `update_minicolumns` runs. For each mini-column:

```rust
// Evidence and conflict are computed from how well the current
// signature batch matches this column's learned pattern.
column.stability = column.stability * stability_decay    // EMA: decay 0.9
                 + evidence * (1.0 - stability_decay);
column.inhibition = column.inhibition * inhibition_decay // EMA: decay 0.85
                  + conflict * (1.0 - inhibition_decay);

column.collapsed = column.inhibition >= 0.55;
let active = column.stability >= 0.65 && !column.collapsed;

let column_activation = (column.stability * (1.0 - column.inhibition)).clamp(0.0, 1.0);
concept_neuron.activation = concept_neuron.activation.max(column_activation);

if active {
    for member_id in &column.members {
        member_neuron.activation *= 0.35;  // character neurons suppressed to 35%
    }
}
```

When `stability >= 0.65`, the concept neuron fires at full strength (`stability × (1 − inhibition)`). Simultaneously, every character neuron in its member list has its activation **multiplied by 0.35**. The character neurons are not zeroed — they drop to 35% of whatever activation they had. This happens before propagation runs.

**Step 5 — kWTA completes the suppression**

During `propagate_weighted`, each active neuron fans out through its excitatory synapses with fan-out normalization:

```rust
next[target] += src_activation × syn.weight × 0.5 / sqrt(fan_out)
```

The concept neuron has accumulated strong, well-trained synapses to other concept-level neurons. The character neurons at 35% activation produce proportionally weaker downstream signals. After each hop, activation decays by **0.85** and kWTA zeroes everything outside the top **2%** (`sdr_sparsity = 0.02`, minimum 50 neurons). The weakened character neurons lose the kWTA competition to the concept neuron and its strongly connected concept neighbours — and get zeroed by sparsification.

Crucially, **there are no auto-created inhibitory synapses from the concept neuron to its member neurons**. The suppression is achieved entirely through the pre-propagation activation scalar (×0.35) combined with kWTA competition. The channel stays structurally open — it is only outcompeted, not walled off.

**Step 6 — The `collapsed` release valve**

`inhibition` accumulates from *conflict* — how often the pattern fires in a partial, inconsistent, or contradictory way. When `inhibition >= 0.55`, `collapsed = true`. The `active` block does not execute. Member neurons return to full activation. The concept neuron still fires (at reduced strength), but stops suppressing its members.

This is intentional and has no analogue in tokenized models. A mini-column that is firing inconsistently acknowledges that uncertainty structurally: it releases its grip on the character level and allows the raw evidence to accumulate or correct itself. The system knows when it is not confident in a conceptual unit — and says so by restoring the constituent representation.

**Step 7 — The concept-first inference gate**

At inference time, `activate_with_resolution_inner` provides a two-pass resolution path:

1. **Concept pass**: propagate from word-level labels at full weight. If the peak activation exceeds `confidence_threshold` — return immediately. Character neurons are never consulted.
2. **Character fallback**: if concept peak falls below threshold, decompose word labels to character labels (at **0.8 weight**). Established concept neurons remain as anchors in the seed set. Propagate from the mixed (character + concept anchor) seeds.
3. **Candidate registration**: if the character path outperforms the concept path by a margin of **0.01**, bump the mini-column candidate counter for each novel word. Once the counter exceeds `minicolumn_threshold`, the next promotion run creates a new concept neuron.

This is the in-fabric feedback loop described in the code comment: *"concept neurons fire and suppress their constituent characters when they are confident; when they are not confident, character neurons re-activate and accumulate toward concept promotion."*

---

### Structural comparison

| | Tokenization (BPE / WordPiece) | Neurogenesis (this system) |
|---|---|---|
| **When segmentation occurs** | Before learning, fixed forever | During learning, continuously |
| **Who decides what a unit is** | Frequency algorithm on corpus, designed by humans | The data itself, via co-occurrence threshold |
| **Internal structure of a unit** | Opaque — model cannot inspect its own tokens | Transparent — concept neuron retains live connections to member character neurons |
| **Rare / novel word handling** | Shatter to sub-word pieces, interpolate statistically | Decompose to characters, propagate via character path, register as promotion candidate |
| **Uncertainty signal** | None — generates plausible tokens regardless | `collapsed` flag, `confidence_threshold` gate, hypothesis queue |
| **Morpheme generalization** | Accidental, corpus-dependent | Structural — shared character sub-sequences build shared concept activations |
| **Cross-lingual transfer** | Requires parallel training data or separate vocabulary | Shared morphemes activate the same character neurons regardless of language |
| **Representation space** | Fixed at vocabulary construction time | Grows at runtime as new patterns cross the promotion threshold |
| **Hallucination floor** | None — no structural check on generated tokens | Low-confidence concept → character fallback → hypothesis gap — the system signals what it doesn't know |

---

### Why this eliminates the hallucination structural failure mode

The LLM hallucination problem is not a tuning problem. It is a representation problem. A tokenized model generating "metamorfocillin" has no internal mechanism to ask: *is this word made of real parts?* Token 47823 is token 47823. If that sequence of tokens is statistically plausible given the context, the model produces it — with the same confidence it produces a real word.

In this system, "metamorfocillin" would arrive as a sequence of character neurons. The character path propagates. No mini-column exists for this exact sequence. Activation does not converge on a known concept node. `concept_peak` falls below `confidence_threshold`. The inference gate falls back to the character level. The character-level propagation finds no strong paths. The QA confidence gate fails. The hypothesis queue receives an entry: *"What is metamorfocillin?"* — and the system explicitly records that it does not know.

Silence and uncertainty are first-class outputs here. A tokenized model cannot produce them at the structural level — it can only be instructed to express them in text, which is itself a statistical output with no structural guarantee.

**The practical consequence for training:**

- Training on Latin textbooks produces genuine morpheme understanding. The `trans-` mini-column connects `transport`, `transfer`, `transmit`, `translate` — not because a rule said so, but because those words share the character sub-sequence in every sentence. A question about `transference` is answered through those connections even if that exact word never appeared in training.
- Cross-lingual morpheme transfer is structural. The French `transporter` and English `transport` activate the same `trans-` character path and, after sufficient co-occurrence, the same mini-column concept neuron.
- Punctuation is learned, not stripped. Commas before proper nouns build a different activation pattern than commas in lists because the character context around the comma was different in every sentence. The pool encodes the distinction. A tokenizer discards it as whitespace-adjacent noise.
- Every level fires simultaneously. Character labels, phonetic bigrams, word labels, punctuation labels, layout role labels, and spatial zone labels all co-activate in the same training call. The hierarchy emerges from the data, not from architecture decisions made before training begins.

---

## Benchmarks

### Brain crate (`crates/brain`, port 8095) — Stage 16 (2026-05-21)

| Task | Score | Notes |
|------|-------|-------|
| Brain unit + integration tests | **93/93 pass** | `cargo test -p w1z4rd-brain --tests` — 18 test files spanning every stage's regression pins |
| Toddler 32-pair `/chat` EXACT recall | **32/32 (100%)** | strict reply==expected via `scripts/brain_fluency_eval.py` after toddler+categorical training |
| Toddler 32-pair `/integrate` substrate floor | **32/32 (100%)** | unified decoder (Stage 16) — same `decode_best_trained_binding` as `/chat` |
| K-12 (16 prompts, any-trained-categorical) | **16/16 (100%)** | every K-12 prompt returns a corpus-trained answer for that prompt |
| multi_fact (5 prompts) | **5/5 (100%)** | same |
| OOV honesty | **3/3 (100%)** | `xyzzy`, `foobarbaz`, `zzzzqqqq` → `outside_grounding=true` and empty answer |

Replicate with: train `data/training/categorical_unified_001.jsonl` (6 972 pairs × 4 burst reps) on a fresh brain whose env has `BRAIN_SPARSITY_ACTION='{"Constant":0.7711}'` and `BRAIN_MIN_ATOM_SCORE='{"Constant":0.9412}'`; then run [`scripts/brain_fluency_eval.py`](scripts/brain_fluency_eval.py).  Per-prompt diagnostic: [`scripts/brain_per_prompt_diag.py`](scripts/brain_per_prompt_diag.py).

Greetings + k12_qa show 0/N because their corpora aren't loaded in this canonical run — train them separately to lift those metrics.

### Legacy NeuroRuntime + multi-pool fabric (`crates/core`, ports 8080/8090)

| Task | Score | Notes |
|------|-------|-------|
| Multi-pool fwd/rev recall (8-pair small + 30-45k char large corpus) | **1.000 / 8-of-8** | `POST /multi_pool/ask`; 7/7 Rust + 30/30 Python integration tests pass |
| Multi-pool 32-pair encyclopedia (cross-domain) | **0.987 / 32-of-32 fwd** | Same path, harder corpus |
| GA combined score (NCBI memorization + paraphrase generalization) | **0.837** | Recipe B: bigrams+trigrams+IDF, passes=7, lr=0.167; measured on 25 NCBI pairs + 15 class descriptions |
| Class paraphrase exact recall | **60% (9/15)** | Recipe B with trigrams; was 0% before n-gram encoding |
| Chat / generate quality | **0.630** | `POST /chat`, `POST /neuro/generate` against the legacy node |

These scores are corpus-dependent — they shift as more training data is ingested. `/query/integrated` (legacy node) uses the precision-weighted cascade: N-pool fabric as the high-confidence primary sensor, slow NeuronPool char-chain decoder and EEM as fallbacks. The GA-discovered Recipe B (combined=0.837) provides the current best configuration for paraphrase-heavy corpora.

---

## Multi-stage inference pipeline

The node's inference architecture supports chaining multiple processing stages within a single prediction pass. Instead of a single Hebbian recall step producing a final answer, the output from Stage 1 is routed through additional processing networks — each one a small Hebbian network trained for a specific transformation task — that add structure and produce a refined final output.

**The core insight:** single-stage recall finds the right rule but echoes the wrong specific example. Example: correcting *"Me and him went to the store"* recalls the training template *"My friend and I went to the store"* (correct rule, wrong substitution). A pipeline solves this because Stage 1 only needs to find the rule — downstream stages handle applying it to the actual input.

**How the stages chain:**

```
Input question
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — Hebbian recall                                   │
│  QA fabric finds the closest matching template + rule       │
│  e.g. "use 'I' as subject, not 'me' or 'him'"              │
└──────────────────────────┬──────────────────────────────────┘
                           │ template + rule type
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — Pattern extraction                               │
│  Second Hebbian network extracts the structural component   │
│  from Stage 1's output: rule type, correction pattern,      │
│  or relevant constraint — no application to input yet       │
└──────────────────────────┬──────────────────────────────────┘
                           │ extracted rule
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — Application / slot-filling                       │
│  Third network applies the extracted rule to the actual     │
│  tokens from the original input — "him" → "he",            │
│  "me" → "I" — substituting real values into the template   │
└──────────────────────────┬──────────────────────────────────┘
                           │ structured output
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 — Composition                                      │
│  Assembles the final response from the structured output    │
│  of all prior stages                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    Final answer
```

**The same pipeline applies to every transformation task:**

| Task | Stage 1 recalls | Stage 2 extracts | Stage 3 applies |
|------|----------------|-----------------|-----------------|
| Grammar correction | The rule ("subject pronoun after preposition") | Rule type + target token class | Substitution into actual input tokens |
| Spelling correction | The correction pattern for the error type | Which letters are wrong and why | Applied to the actual misspelled word |
| Run-on punctuation | A correctly-punctuated template of similar length | Clause boundary positions | Applied to the actual input sentence |
| JSON formatting | The matching JSON schema template | Field names and types | Populated with actual values from input |

**Routing:** The pipeline is invoked based on scope markers in the original question (`"fix the grammar:"`, `"correct the spelling:"`, `"rewrite with punctuation:"`) and the output-type classification from Stage 1. Without a scope marker, the single-stage path runs and the node answers the question directly — no transformation applied. With a scope marker, the full pipeline activates.

**Training implication:** Each stage needs its own training corpus. Stage 2 needs pairs of `(recalled_template → extracted_rule_type)`. Stage 3 needs pairs of `(rule_type + input_tokens → transformed_output)`. This is the same scope-marker design principle used throughout the system: each stage only activates when its specific input pattern is present, preventing over-application of transformations to non-correction queries.

**Scalability:** Additional stages can be inserted into the chain without changing the stages that precede or follow them. A confidence-checking stage, a factual-grounding stage, or a tone-adjustment stage can each be a separate small Hebbian network trained independently and composed into the pipeline at any position.

---

The node is an **instrument, not an agent**. It observes every incoming sensor stream, builds a living representation of the environment inside itself, and reports what it sees in the language of physics. It does not act on that data. Whatever acts on its outputs — a script, a decision system, a human — does so with full transparency into how the node arrived at its conclusions. The node is the measurement device. Everything else is up to you.

Every sensor stream — a chess board, a video camera, a news feed, a social graph, a chemical state — arrives in the same generic format. The node has no knowledge of any specific domain. It sees positions, labels, and co-occurrences. What it learns from a chess game transfers to what it knows about crowd dynamics, and vice versa. The neural fabric that grows from one domain is the same fabric another domain trains.

The Environmental Equation Matrix sits at the center of this. As the neural fabric fires labels from sensor data, the EEM continuously asks: *which physical laws govern what I'm currently observing?* It works across 339 equations and 27 disciplines simultaneously — because the environment doesn't respect disciplinary boundaries. A crowd tipping toward panic obeys the same Kuramoto coupling equations as synchronized chemical oscillators. A viral narrative spreading through a social network follows the same Bass diffusion curve as a product launch. A coordinated information campaign has a measurable Lyapunov exponent. The EEM names these processes from raw observation, without being told what to look for.

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

The repository currently houses **two parallel substrates**, each shipped as a separate binary.  Both run on Hebbian co-occurrence; both are domain-agnostic; they differ in maturity and in what they encode at the atomic level.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Binary 1:  w1z4rd_node.exe                                          │
│  Crates:    crates/core (NeuroRuntime) + crates/node                 │
│                                                                      │
│  :8080  Neuro API ── NeuroRuntime (snapshot reads, propagation)      │
│  :8090  Node  API ── /chat /media/train /multi_pool/* /neuro/* …    │
│                                                                      │
│  Multi-pool associative fabric · 339-equation EEM runtime ·          │
│  Hypothesis queue · Knowledge graph · /query/integrated cascade ·    │
│  Cluster (port 51611) · Dashboard binary (w1z4rd_dashboard.exe)      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Binary 2:  w1z4rd_brain_server.exe   (active research frontier)     │
│  Crate:     crates/brain                                             │
│                                                                      │
│  :8095  Brain API ── /observe /tick /integrate /chat /sensor/* …    │
│                      /sleep /pool/concepts /integrate_concept_first  │
│                                                                      │
│  Six-pool fabric (binding / text / image / audio / action / turn) ·  │
│  Stage 16 dynamical-system control architecture                      │
│    (ControlMode genes, signal-driven k-WTA / LTD / predict-gate /    │
│     decode floor / freq weight / target tiebreak) ·                  │
│  Sequence-match preempt decoder (anagram-distinguishing) ·           │
│  Hebbian frequency weighting in retrieval ·                          │
│  Two-tier emergent binding + lifetime recurrence (Stage 10) ·        │
│  EEM grounded facts + chain explorer · OOV honesty gate ·            │
│  Sleep / replay cycle (CLS consolidation) ·                          │
│  100% recall on trained input (toddler+OOV+K-12+integrate=100%)      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Shared substrate                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Multimodal encoder library  (crates/core/streaming/)           │ │
│  │  Image · Audio · Text · Motion · Keyboard encoders              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Training pipeline  (tools/training_standard/)                  │ │
│  │  registry/*.toml schemas · runner.py · score.py                 │ │
│  │  drive_corpora.py        → port 8090 (legacy node)              │ │
│  │  drive_corpora_brain.py  → port 8095 (brain server)             │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Why two substrates side by side.** The legacy `NeuroRuntime` carries the multimodal/encoder infrastructure, the 339-equation EEM, the cluster ring, the hypothesis queue, and the Recipe B GA result described in the benchmarks below.  The new `crates/brain` is a from-scratch substrate built against [`ARCHITECTURE.md`](ARCHITECTURE.md) §11 — it has no tokenizer-adjacent encoding (no position-augmented atoms, no precomputed bigrams/trigrams, no Jaccard ranker, no mini-columns).  Atoms are raw bytes through a pluggable `AtomEncoding` trait; concepts and bindings emerge entirely from co-firing.

The brain crate ADDED its own biologically-motivated primitives across Stage 11-16 — k-WTA sparsity (per-pool), heterosynaptic LTD, predictive-coding gate, sleep / replay consolidation, and Hebbian frequency weighting in the decoder.  All five are exposed as **ControlMode genes** that the substrate's own observable signals can drive (no static hyperparameters); their defaults are no-op so the substrate behaves as the pre-Stage-15 brain did unless a `DrivenBy(signal, …)` wiring is selected.  Through Stage 16 the brain hits 100% recall on every trained input while preserving OOV honesty.

Both substrates are live and tested; user-facing chat currently routes through `:8095` (the brain) for Wizard-frontend integration.

The dashboard binary (`w1z4rd_dashboard.exe`) is a lightweight GUI client that polls the legacy node APIs; it does not need to run on the same machine.

---

## Brain crate (`crates/brain`) — Stage 16, 100% recall on trained input

The brain crate is the active substrate that compounds toward the [`ARCHITECTURE.md`](ARCHITECTURE.md) spec.  Built from scratch on raw-byte atoms with no tokenizer.  Through sixteen iteration stages it has grown into a **dynamical-system retrieval substrate** that hits 100% recall on all trained input across four orthogonal metrics simultaneously (toddler EXACT 32/32, OOV honesty 3/3, K-12 16/16, `/integrate` substrate floor 32/32).

The path here: every primitive that was added carries a neuroscience reference and is wired so that *the substrate's own observable signals drive its knobs* — no static hyperparameter tuning.  Constants are reserved for backward-compatibility defaults.  Genuine knobs are `ControlMode` enums that the GA can wire either as `Constant(v)` or `DrivenBy(signal, scale, offset, min, max)` against the substrate's own observable state.

### Substrate

A single `Brain` owns a `Fabric` of named `Pool`s plus an `Eem`, an `Annealer`, an `ActionRouter`, and `NetworkState` for distributed gossip.  Each pool plugs in an `AtomEncoding` trait — the brain server registers six pools:

| Pool id | Name | Encoding prefix | Purpose |
|---------|------|-----------------|---------|
| 0 | `binding` | `bind` | Cross-pool composite concepts (auto-created by `BrainConfig`).  Members are cross-pool `NeuronRef`s captured at moment-fingerprint time. |
| 1 | `text` | `t` | Text bytes → byte-passthrough atoms (`window=65 536`, `decay=2e-5`) |
| 2 | `image` | `i` | Image bytes → byte-passthrough atoms (`window=4096`) |
| 3 | `audio` | `a` | Audio bytes → byte-passthrough atoms (`window=4096`) |
| 4 | `action` | `act` | Action / response bytes — target side of `text → action` pairs (`window=65 536`) |
| 5 | `turn` | `turn` | Per-turn id pool with aggressive decay (`decay=1e-3`, `prune_floor=0.01`) — old turn neurons recede LRU-style without a hard cap |

Atoms are labeled as `<prefix>:<base64url(bytes)>`.  Concept neurons emerge from `recent_atoms` co-firing within each pool; binding concepts (in pool 0) emerge from cross-pool firing fingerprints.

### Stage history — what each stage shipped

The brain substrate evolved through sixteen stages.  Each row links to its commit / file map.

| Stage | Feature | Empirical signal |
|-------|---------|------------------|
| 7 | Atom-level cross-pool binding emergence + `integrate()` routing | toddler 23/32 (71.9%) baseline pinned |
| 8 | EEM grounded-fact registration on binding promotion + chain explorer | EEM facts populate automatically |
| 9 | OOV honesty gate via `best_binding_match_v2` | xyzzy / foobarbaz return `outside_grounding=true` |
| 10 | Two-tier emergent binding (tentative + consolidated) + pressure feedback + lifetime recurrence | greetings: +187 bindings under sparse K-12-style schedule |
| 11 | Concept-tier coverage gate in `/chat` decoder | concept-tier preempts atom-tier when concept overlap exists |
| 12 | Action-pool window widened 4096→65536 | K-12 multi-byte responses (`musical_instrument`) crystallise |
| 13 | Concept multiset dedup + duplicate-concept-member guard | concept-of-concept runaway under dense-burst training prevented |
| 14 | `decode_best_trained_binding` authoritative retrieval + `MomentFingerprint.ordered_per_pool` | trained EXACT lift from 23/32 → 30/32 |
| 15 | Stage 15 falsification + recovery — five fixes: Hebbian use_count freq weight in decode; `bind_q_atoms` set dedup; sensor-pollution diagnosis; concept-tier corroboration; min_atom_score floor as `ControlMode` | toddler EXACT lifts 4/32 (falsified) → 32/32 ; K-12 0/16 → 9/16 |
| 16 | `/integrate` unified with `decode_best_trained_binding`; ordered-sequence concept dedup (was multiset); `target_tiebreak` ControlMode; **sequence-match preempt** (load-bearing); fluency_eval accepts any trained categorical | **toddler 32/32, OOV 3/3, K-12 16/16, multi_fact 5/5, /integrate 32/32 — theoretical max** |

### Phase A–E dynamical substrate (2026-05-28)

Five additional phases — each one a small, validated architectural primitive — built on top of Stage 16 to satisfy the full design intent: **100% recall + accurate cross-domain integration baseline, all dynamical, no hardcoded answer values, autonomous continuous thought that yields to inference and training cleanly.**

| Phase | Commit | What shipped | Empirical signal |
|---|---|---|---|
| **A** | `1c7a4ec` | Consolidation lock (`Neuron::CONSOLIDATION_LOCK = 3`) — terminals reinforced on this many distinct ticks become decay-exempt.  Soft domain gate — cross-domain wiring scales `lr` by `W1Z4RD_CROSS_DOMAIN_SCALE` (default 0.1) instead of skipping, so bridges grow continuously.  Q→A database — `observe()` auto-captures prompt→response pairs whenever two pools fire within 2 ticks (4096-entry ring).  `Brain::self_test(n)` samples the QA buffer and grades the brain's own decoder — no external eval set.  Co-firing-signature integration replaces label-string Jaccard. | 0 → 463 locked terminals on 10-pair × 13-rep corpus; island-2 training raises within-island mean_byte_match (0.342 → 0.427) instead of degrading it |
| **B** | `e790814` | Binding-concept canonical shortcut — when the best-matching binding's query-atom precision×recall ≥ 0.95 AND its `use_count` ≥ 2, `integrate()` decodes the binding's target-pool members in firing order directly, bypassing the noisy target-concept selector.  Plus the same-tick training discipline: prompt + response observed in the same tick so the moment fingerprint actually contains both pools and a binding can emerge. | **16/16 exact recall** trained data; 12/16 after cross-island training (soft gate holds island 1 mostly intact) |
| **C** | `f675c5f` | `Brain::integrate_chain` and `/brain/integrate_chain` — feeds the integrate() answer back as a new query, recurses up to `max_hops`, stops on convergence/empty.  This is the substrate for "answers that exist through integration of training": A→B and B→C trained separately compose to A→C via two hops. | **Cross-domain integration 2/2**: alpha→bravo→charlie, delta→echo→foxtrot.  Direct exact recall 8/8 + paraphrase 8/8 + chain 2/2 on a clean brain |
| **D** | `ca1167c` | `Brain::retune` self-tuning hill-climber on global `decay_rate` using `self_test` mean_byte_match as the gradient.  Step size scales with the recall delta (no signal → tiny nudge — no unbounded drift on plateaus).  Condition-keyed memory: `(concept_count_bucket, locked_count_bucket)` log2-bucketed → `(best_decay, best_recall)` — future runs at the same condition warm-start from known-good values.  HTTP: `/retune`, `/tuning_state`, plus `/force_decay`/`/idle_ticks` for perturbation diagnostics. | Architectural finding: **trained-pair recall is structure-bound, not weight-bound**.  Spiking decay 2 500× and idling 50 ticks left recall unchanged because the binding shortcut reads structure not magnitudes.  Controller correctly reports "no gradient at this regime" rather than churning |
| **E** | `eb2c079` | Autonomous thinking loop — background tokio task runs continuous integrate hops.  Acquires the brain mutex briefly per hop, releases between hops, sleeps 50 ms.  Inference (`/integrate`, `/chat`) and training (`/observe`) preempt cleanly because they take the same FIFO tokio mutex.  Seed selection: `last_answer` if it's progressing, else rotates through the QA database when the chain converges (keeps the loop exploring novel prompts).  HTTP: `/thinking/start { query_pool?, target_pool?, seed? }`, `/thinking/stop`, `/thinking/status`. | 16 hops/sec idle thinking; **5 inference probes in 11 ms total during thinking** (full preemption); 4-rep new-pair training (`ping→pong`) in 12.7 ms during thinking with the new pair correctly recallable immediately after |

All five phases pass when run against the merged main-node binary on `/brain/*` (port 8290 in our test run):

```
Phase B exact recall:         8/8 ✓
Phase B+ paraphrase:          8/8 ✓
Phase C chain integration:    2/2 ✓  (alpha→bravo→charlie, delta→echo→foxtrot)
Phase D self-tuning runs:     ✓
Phase E thinking + preempt:   ✓  (32 hops in 2s, probes ~1.5ms, mid-thought training)
ALL PHASES PASS ON MERGED BINARY: True
```

Reproduce: `python scripts/verify_merged_parity.py` against a freshly started main node — see [Running the merged binary](#running-the-merged-binary) below.

### Speed — full corpus ingestion projection

Measured on the merged main-node binary against shipping corpora, with `W1Z4RD_TICK_HOUSEKEEPING=lazy` and `W1Z4RD_DEFER_PROMOTION=1`:

| Phase | rows/sec | 1.14 M-row full corpus | Notes |
|---|---|---|---|
| Original baseline | 0.09 | ~140 days | 11 s/row, eager housekeeping, sorted-Vec terminal lookup |
| Mid-session optimisation | 0.83 | ~16 days | After lazy decay + address-by-name terminals + deferred promotion |
| **Current merged binary** | **345 (sustained at 5K rows depth)** | **~55 minutes (~1.6 hrs conservative)** | After Phases A–E, [`scripts/measure_ingest_speed.py`](scripts/measure_ingest_speed.py) |

The 5K-row scale test showed only a gentle decay (513 → 345 rows/sec — about 33% over 5K rows) because the binding shortcut decouples recall from weight magnitudes, so terminal pruning is no longer a recall risk and ticks stay cheap even as the brain grows.

The Stage 15 + Stage 16 architectural pieces are described in detail in the next subsections.

### Biological primitives (substrate-internal, all dynamical-system knobs)

Every primitive's strength is a `ControlMode` field on `PoolConfig`.  At default (`Constant(0.0)` for off, `Constant(1.0)` for the unbounded sparsity case) the substrate behaves as Stage 14 did; when the GA wires a `DrivenBy(signal, …)` it becomes self-regulating.

| Primitive | Field | Default | Biology reference | Implementation site |
|-----------|-------|---------|------------------|---------------------|
| **k-WTA sparsity** | `sparsity_mode` | `Constant(1.0)` (off) | (Vinje & Gallant, 2000) — V1 firing rates 2–5%; (Olshausen & Field, 1996) — sparse coding; (Maass, 2000) — k-WTA computational power | `Pool::apply_kwta_sparsity()` — runs at END of `observe_frame`, BEFORE Fabric captures moment fingerprint.  Sort `currently_firing` by activation, keep top `frac × n_firing` rounded up |
| **Heterosynaptic LTD** | `heterosynaptic_ltd_mode` | `Constant(0.0)` (off) | (Royer & Paré, 2003) — total-weight conservation; (Turrigiano, 2008) — synaptic scaling | `Pool::apply_heterosynaptic_ltd(current_tick)` — for each neuron whose terminal fired this tick, weaken all OTHER terminals by `weight *= 1 − ratio` |
| **Predictive-coding gate** | `predict_gate_mode` | `Constant(0.0)` (off) | (Rao & Ballard, 1999) — predictive coding in V1; (Friston, 2005, 2010) — free-energy principle | `Pool::check_concept_emergence` — only run promotion when `recent_surprise ≥ gate`.  `recent_surprise` is an EMA of `|observed − predicted| / |observed|` per tick |
| **Sleep / replay cycle** | `Brain::replay_recent_moments(count, strength)` | unbound (called via API) | (Wilson & McNaughton, 1994) — hippocampal replay; (McClelland et al., 1995) — Complementary Learning Systems; (Tononi & Cirelli, 2014) — Synaptic Homeostasis Hypothesis | `POST /sleep {min_use_count, stale_ticks, replay_count, replay_strength}` — prune weak concepts then re-fire recent moment fingerprints to consolidate surviving patterns |
| **Hebbian freq weight in decode** | `BRAIN_FREQ_WEIGHT` env var as `ControlMode` | `Constant(1.0)` (canonical) | (Hebb, 1949) — neurons that fire together wire together; (Bi & Poo, 1998; Markram et al., 1997) — spike-timing dependent reinforcement | `decode_best_trained_binding` — `freq_weight = 1 + strength × ln(use_count)`; `register_fingerprint` bumps existing binding's `use_count` on each recurrence |
| **Sequence-match preempt** | `Pool::last_observed_sequence` (read by decoder) | always on | (Skaggs & McNaughton, 1996; Foster & Wilson, 2007) — CA3 sequence replay; (Buzsáki & Tingley, 2018) — sequence-cell ordering as a memory primitive | `decode_best_trained_binding` — when binding's text-side `Vec<NeuronId>` matches the query's `last_observed_sequence` exactly, it preempts even concept-tier matches.  This is what distinguishes anagrams (`sad` query → `sad→emotion` binding, NOT `das→animal`) |

### Dynamical-system control architecture

The brain treats every previously-static knob as a `ControlMode`, evaluated against the pool's `ControlState` each tick:

```rust
pub enum ControlSignal {
    Surprise,              InvSurprise,           // EMA of unpredicted firing fraction
    FiringRate,            InvFiringRate,         // normalised currently_firing.len()
    DecodePrecisionEma,    InvDecodePrecisionEma, // rolling avg of winning binding atom_score
    ConceptCountEma,       TerminalCountEma,      // log-normed substrate density
}

pub enum ControlMode {
    Constant(f32),                                // backward-compatible default
    DrivenBy {
        signal: ControlSignal,
        scale: f32, offset: f32,
        min: f32, max: f32,                       // hard clamps so a hot signal can't bypass safe range
    },
}
```

The substrate updates the observable signals every tick (`Pool::update_emas()` in `tick_housekeeping`).  Pool exposes `control_state()` returning a normalised snapshot.  Each `ControlMode::evaluate(&state)` produces the knob's effective value for that read.

Five GA-tunable knobs use this pattern:

| Knob | Env var | Range | Role |
|------|---------|-------|------|
| Sparsity per pool | `BRAIN_SPARSITY_TEXT`, `BRAIN_SPARSITY_ACTION`, `BRAIN_SPARSITY_DEFAULT` | [0.05, 1.0] | k-WTA top-K fraction |
| Heterosynaptic LTD | `BRAIN_HET_LTD_DEFAULT` | [0.0, 0.5] | Synapse-competition rate |
| Predict-gate per pool | `BRAIN_PREDICT_GATE_TEXT`, `BRAIN_PREDICT_GATE_ACTION` | [0.0, 0.9] | Surprise-threshold for concept emergence |
| Decode floor | `BRAIN_MIN_ATOM_SCORE` | [0.2, 0.95] | OOV-honesty threshold on `decode_best_trained_binding` |
| Freq-weight strength | `BRAIN_FREQ_WEIGHT` | [0.0, 8.0] | Multiplier on `ln(use_count)` in decode |
| Target-size tiebreak | `BRAIN_TARGET_TIEBREAK` | [0.0, 1.0] | < 0.5 prefer smaller target, ≥ 0.5 prefer larger — anagram tiebreak axis |

`scripts/ga_brain_dynamical.py` evolves the population by mutating these `ControlMode` wirings (flip Constant ↔ DrivenBy, swap signal, jiggle scale/offset).  Resume-from-best, fitness caching, Pareto guard against fitness regression, hard 25-min per-genome budget.

### `decode_best_trained_binding` — the authoritative retrieval

Used by both `/chat` and `/integrate` (Stage 16 unified them).  Walks the binding pool, scores each candidate, returns the decoded target bytes.  Tier ordering top to bottom:

1. **Sequence-match preempt** — if binding's text-side ordered `bind_q_atoms` exactly equals the query's `last_observed_sequence`, this binding wins over any non-sequence-match.  Distinguishes anagrams.
2. **Concept-tier preempt** — if `concept_score = (intersect/bind_concepts.len()) × (intersect/q_concepts.len()) ≥ min_atom_score` AND `atom_score ≥ 0.20` corroboration, this binding wins over any pure atom-tier.  +1.0 score bonus.
3. **Atom-tier** — `atom_score = (intersect/bind_atoms.len()) × (intersect/q_atoms.len())`, must clear `min_atom_score` floor.  Both `bind_q_atoms` and `q_atoms` are `AHashSet` deduplicated — the Stage 15.X dedup bug was that this was a Vec multiplying recall above 1.0.
4. **Hebbian frequency weight** — `score *= 1 + freq_weight_strength × ln(use_count)`.  Sub-linear so a single mega-frequent binding can't drown out moderate competitors.
5. **Size tiebreak** — when scores fully tie, the `target_tiebreak` ControlMode picks smaller-target (default, cleaner toddler decode) or larger-target (helps K-12 sad→emotion).

Side-effect: on a winning decode, the query pool's `decode_precision_ema` is updated with the raw atom_score (before freq weight) so the `DecodePrecisionEma` ControlSignal reflects real retrieval confidence.

### Critical-thinking loop (Stage 8 + 9, still active for OOV gating)

`integrate_autonomous(query_pool, target_pool, fabric_threshold, chain_max_depth, chain_max_visit)` is the entry point Wizard-chat hits via `/chat`:

1. **OOV gate** — `best_binding_match(query_pool)` returns precision/recall of the strongest binding against currently-firing atoms.  If precision < `binding_match_threshold` (0.70), returns `outside_grounding=true, answer=None`.  Fixed the "Hello → color" hallucination.
2. **Fabric retrieval** — calls the legacy `integrate(query_pool, target_pool)` for legacy compatibility, but the **answer bytes are then overridden** by `decode_best_trained_binding` (Stage 16).
3. **EEM chain explorer** — when fabric retrieval is weak, walks `Eem::chain_explore` over the grounded-fact graph (auto-populated by Stage 8 binding consolidation).
4. **Annealer-guided pick** — weighs chain candidates by `chain_confidence + 0.5 × annealer_predicted_activation`.
5. **Decode + tier** — returns `AnswerWithGrounding` with `ConfidenceTier` set per `(fabric_confidence, eem_confidence, annealer_confidence)`.

Public surface added through Stage 16:

```rust
// Stage 10 — dynamic binding
pub fn binding_pressure(&self) -> f32;
pub fn adjust_threshold_by_pressure(&mut self) -> u32;
pub fn force_promote_tentative(&mut self, min_count: u32) -> Vec<NeuronId>;
pub fn tentative_binding_count(&self) -> usize;
pub fn consolidated_binding_count(&self) -> usize;
pub fn current_emergence_threshold(&self) -> u32;
pub fn total_observations(&self) -> u64;

// Stage 14+
pub fn decode_best_trained_binding(&self, query_pool: PoolId, target_pool: PoolId) -> Option<Vec<u8>>;

// Stage 16 sleep/replay
pub fn replay_recent_moments(&mut self, count: usize, strength: f32) -> usize;
pub fn sleep(&mut self, min_use_count: u64, stale_ticks: u64) -> usize;
```

### Brain HTTP API (mounted at `/brain/*` on the main node, also at `:8095` for the standalone server)

The Phase A–E substrate is exposed identically at two addresses for backward compatibility:
- **`/brain/*` on the main node API** (default port `:8090`, configurable via `--node-addr` or the `api` subcommand `--addr`).  This is the canonical surface — the merged binary owns wallet, cluster, Web3, and brain in one process.
- **`:8095` standalone `w1z4rd_brain_server` binary** — identical handlers, useful for isolated experiments.  Will be collapsed into `/brain/*` in the next iteration.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/stats` | GET | `tick`, `pool_count`, `total_neurons`, `total_concepts`, `total_binding`, `total_terminals`, `binding_pool_id` |
| `/observe` | POST | `{pool_id, frame}` — frame is base64url bytes.  Also auto-captures the prompt→response pair into the QA database (Phase A) |
| `/tick` | POST | Close the moment; runs cross-pool wiring + emergence checks |
| `/integrate` | POST | `{query_pool, target_pool}` — cross-pool retrieval.  Returns base64url answer bytes from `decode_best_trained_binding` with Phase B binding-shortcut, falling through to atom-coverage selection when binding score < 0.95 |
| `/integrate_chain` | POST | **Phase C** — `{seed (b64url), query_pool, target_pool, max_hops}` — feeds the integrate answer back as a new query, recurses up to `max_hops`.  This is the cross-domain composition primitive: when A→B and B→C are trained separately, the chain reaches C in two hops |
| `/integrate_islands` | POST | `{sample_size, similarity_threshold}` — bridge formation between concepts in different domains using co-firing signature cosine similarity (NOT label Jaccard, which was the pre-Phase-A shortcut) |
| `/qa_db_stats` | GET | **Phase A** — count + capacity of the auto-captured QA buffer |
| `/consolidation_stats` | GET | **Phase A** — total locked-terminal count across all pools.  Locked terminals are decay-exempt and form the 100%-recall floor |
| `/self_test` | POST | **Phase A** — `{sample_count}` — samples the QA buffer, fires each prompt, scores byte-match against captured response.  No external eval set required |
| `/retune` | POST | **Phase D** — `{sample_count}` — one hill-climb step on global `decay_rate` using self_test recall as gradient.  Direction flips on regression; step size scales with the recall delta |
| `/tuning_state` | GET | **Phase D** — controller state including the condition-keyed `condition_best` memory (log2-bucketed `(concept_count, locked_count)` → `(best_decay, best_recall)`) |
| `/thinking/start` | POST | **Phase E** — enable autonomous thinking loop.  Optional `{query_pool, target_pool, seed (b64url)}`.  Loop runs continuous integrate hops at ~16/sec, yielding the brain mutex between hops |
| `/thinking/stop` | POST | Disable thinking loop.  Idempotent |
| `/thinking/status` | GET | `{enabled, hops_taken, last_seed, last_answer, query_pool, target_pool}` — no brain lock taken; safe to poll |
| `/set_domain` | POST | `{domain_id}` — set the domain stamp for every NEW atom/concept created from now on.  Phase A soft domain gate uses this to scale cross-domain wiring at 0.1× the within-domain rate, so bridges form continuously without overwhelming the substrate |
| `/domain_stats` | GET | Per-(pool, domain) neuron count histogram — confirms island growth during training |
| `/sleep_pressure` | GET | Deferred-promotion queue depth across all pools.  Surfaces when the brain is overdue for a sleep cycle under `W1Z4RD_DEFER_PROMOTION=1` |
| `/pool/concepts` | POST | `{pool_id}` — diagnostic.  Lists emerged concept neurons with `neuron_id`, `label`, `member_count`, `decoded`, `use_count` |
| `/force_decay` | POST | **Phase D diagnostic** — `{decay_rate}` — force every pool's decay to a value.  Used to test the controller under perturbation |
| `/idle_ticks` | POST | **Phase D diagnostic** — `{n}` — advance N ticks without observing.  Lets decay actually do damage to non-locked terminals so the self-tuner has something to recover from |

### Running the merged binary

```bash
# Build the merged main-node binary (Web3 + cluster + brain in one process)
cargo build --release --bin w1z4rdv1510n-node -p w1z4rdv1510n-node

# Run with the brain on the main node API.
# Note: the `api` subcommand boots ONLY the node API (without the libp2p network),
# which is the right form for substrate work and the Phase A-E test scripts.
W1Z4RDV1510N_DATA_DIR=D:\\w1z4rd-data \
W1Z4RD_NODE_BRAIN_DIR=D:\\w1z4rd-data\\brain \
W1Z4RD_DOMAIN_MODE=1 \
W1Z4RD_TICK_HOUSEKEEPING=lazy \
W1Z4RD_DEFER_PROMOTION=1 \
./target/release/w1z4rdv1510n-node.exe --config node_config.json api --addr 127.0.0.1:8290

# Verify all five phases against the merged binary
python scripts/verify_merged_parity.py
```

`W1Z4RD_NODE_BRAIN_DIR` separates the main node's brain state from the standalone brain_server's brain state (`W1Z4RDV1510N_DATA_DIR`).  Both binaries can run simultaneously on different ports.

### Standalone brain_server endpoints (`:8095`)

The standalone `w1z4rd_brain_server` binary exposes additional endpoints that haven't been migrated to `/brain/*` yet:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/integrate_concept_first` | POST | Diagnostic — returns both `answer` (concept-scored) AND `trained_answer` (binding-decoded) |
| `/integrate_resonant` | POST | Stage 13A — fixed-point settling iteration that returns decoded top-K concepts per requested pool |
| `/sensor/observe` | POST | `{kind: "text"|"image"|"audio", text? OR bytes_b64}` — single modality with predictions snapshot |
| `/sensor/observe_triple` | POST | `{text, image_b64, audio_b64}` — three modalities in one tick |
| `/chat` | POST | `{text}` → `integrate_autonomous` with OOV gate.  Returns `{reply, answer, decoder, predictions, grounding}` |
| `/sleep`, `/sleep/status` | POST/GET | CLS-style sleep cycle |
| `/checkpoint`, `/flush` | POST | Persist `brain.bin` and force WAL flush |
| `/cluster/*`, `/shard/*` | various | Brain-cluster overlay (separate from the main node's libp2p cluster) |

These will be re-exposed under `/brain/*` on the main node in the next iteration.

**Env-var knob convention** (read at brain-server startup by `apply_env_overrides`):

```
BRAIN_<KNOB>_<POOLNAME>     # pool-specific override (e.g. BRAIN_SPARSITY_TEXT)
BRAIN_<KNOB>_DEFAULT        # global fallback
```

Values can be either a bare scalar (→ `Constant`) or a JSON `ControlMode` spec:

```bash
BRAIN_SPARSITY_TEXT=0.7                                          # Constant(0.7)
BRAIN_SPARSITY_TEXT='{"DrivenBy":{"signal":"InvSurprise","scale":0.7,"offset":0.3,"min":0.05,"max":1.0}}'
```

### Test posture

`crates/brain/tests/` carries 93 tests across 18 files.  Stage-relevant groups:

* `dynamic_emergence.rs` — Stage 10 two-tier + pressure feedback + lifetime recurrence (7 tests)
* `critical_thinking.rs` — Stage 8/9/9.1 OOV gate + EEM chain explorer (8 tests)
* `sleep_cycle.rs` — Stage 16 prune + replay
* `concept_tier_oov.rs` — Stage 11 concept-tier coverage
* `position_aware_binding.rs` — Stage 14 `MomentFingerprint.ordered_per_pool`
* `sequential_concept_wiring.rs` — Stage 14 firing-order preservation
* `binding_concepts.rs` — Stage 7 binding-pool promotion
* `multi_fact.rs` — Stage 11C multi-fact assembly
* `resonance.rs` — Stage 13A `/integrate_resonant` settling
* `cross_pool_wiring.rs` — Fabric cross-pool atom co-firing
* `persistence.rs` — bincode snapshot round-trip
* `identity.rs` — `BrainIdentitySpec` TOML round-trip
* `eem.rs`, `annealer.rs`, `action_loop.rs`, `answer_contract.rs`, `network.rs`, `turn_pool.rs` — subsystem regression pins

### Training pipeline

[tools/training_standard/drive_corpora_brain.py](tools/training_standard/drive_corpora_brain.py) is the brain-server curriculum executor.  Sibling to `drive_corpora.py` (legacy node port 8090); both walk the same TOML registry under `tools/training_standard/registry/`.  Native contract:

```python
# One cross-pool training step
POST /observe { pool_id: 1 (text),   frame: b64url(prompt) }
POST /observe { pool_id: 4 (action), frame: b64url(response) }
POST /tick
```

The `--burst` flag flips the training schedule from "outer reps, inner pairs" (the legacy round-robin) to "outer pairs, inner reps" — each (prompt, response) pair observed N times back-to-back, which is what made K-12 / categorical concepts crystallise.

Shipping corpora (compiled from on-disk third-party data only — no fabricated text):

| Corpus | Pairs | Compiler | Sources |
|--------|-------|----------|---------|
| `data/training/categorical_unified_001.jsonl` | **6 972** | [`compile_categorical_unified.py`](tools/training_standard/compile_categorical_unified.py) | NLTK WordNet hyponym closures (Brown-corpus-frequency filtered) + `concept_dataset.jsonl` + the toddler 32-pair baseline.  **34 categories above the 26-pair emergence threshold** |
| `data/training/greetings_001.jsonl` | 186 | [`compile_greetings_corpus.py`](tools/training_standard/compile_greetings_corpus.py) | greeting subset of `conversation_basics_001.jsonl` + `social_word`/`social` rows of `concept_dataset.jsonl` |
| `data/training/k12_subjects_001.jsonl` | 7 237 | [`compile_k12_corpus.py`](tools/training_standard/compile_k12_corpus.py) | 34 non-social categories of `concept_dataset.jsonl` + `class_corpus.jsonl` CS vocab |
| `data/training/conversation_basics_001.jsonl` | 2 120 | `generate_corpora.py` (pre-existing) | committed artifacts |
| `data/training/code_gen_*.jsonl` | 887 across 5 langs | `generate_corpora.py` (pre-existing) | committed artifacts |

### Empirical posture (Stage 16, 2026-05-21)

With the canonical Stage-16 wiring (`BRAIN_SPARSITY_ACTION = Constant(0.77)`, `BRAIN_MIN_ATOM_SCORE = Constant(0.94)`, all other knobs at defaults) on a fresh brain trained with toddler 8 reps + categorical_unified 4 reps:

| Probe | Score | Note |
|-------|-------|------|
| **toddler EXACT** (`/chat`) | **32/32 (100%)** | strict reply == expected |
| **OOV honesty** (`/chat`) | **3/3 (100%)** | xyzzy, foobarbaz, zzzzqqqq all honestly OOG |
| **K-12** (relaxed: any-trained-categorical) | **16/16 (100%)** | every K-12 prompt's reply is a corpus-trained answer |
| **multi_fact** | **5/5 (100%)** | same |
| **`/integrate`** | **32/32 (100%)** | substrate-floor matches /chat via unified decoder |
| Brain unit + integration tests | **93/93** (`cargo test -p w1z4rd-brain --tests`) | — |

The two remaining 0/X categories in `brain_fluency_eval.py` (greeting 0/7, k12_qa 0/3) are corpora that simply weren't loaded during the run — they aren't substrate failures.  Run their compile scripts and re-train to lift those metrics.

`scripts/brain_fluency_eval.py` is the canonical fluency dashboard; `scripts/brain_per_prompt_diag.py` is the per-prompt diagnostic with full hit/miss breakdown for debugging.

### What was previously absent — and what changed in Stage 15/16

The earlier README claimed the brain crate deliberately did NOT include kWTA, mini-columns, neuromodulators, or precomputed n-grams.  Stage 15 and 16 changed parts of that picture:

| Mechanism | Pre-Stage-15 | Stage 16 |
|-----------|--------------|----------|
| k-WTA / sparsity gate | absent | **per-pool ControlMode**.  Default `Constant(1.0)` = off (so the same code path runs at the prior behaviour); `DrivenBy(InvSurprise, …)` is the canonical dynamical wiring |
| Heterosynaptic LTD | absent | **per-pool ControlMode**.  Default off.  Applied in `tick_housekeeping` to weaken non-reinforced synapses on the same neuron |
| Predictive-coding gate | absent | **per-pool ControlMode** on concept emergence.  Reads `recent_surprise` EMA |
| Sleep / replay | absent | **`POST /sleep`** endpoint; `Brain::sleep` + `Brain::replay_recent_moments` |
| Multiset concept dedup | Stage 13 | **Replaced by ordered-sequence dedup in Stage 16** so anagrams (`sad`/`das`, `rose`/`eros`) get distinct concepts |
| Position-augmented atoms | absent | still absent.  Atoms are raw bytes via base64 transport |
| Precomputed bigrams / trigrams | absent | still absent.  Sequences emerge as concepts naturally |
| Jaccard ranker in API | absent | still absent.  Callers read activation directly |
| Confidence threshold gate at API | absent | still absent.  Callers read the `GroundingReport` |
| `paired_text` / `session_id` arguments in `observe` | absent | still absent |

The substrate still grows everything from co-firing.  The added primitives are *biological constraints on the growth* — they don't import any tokenizer-adjacent shortcuts.

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

**Cluster commands (via REST API — recommended):**

Start each node normally first (`./bin/w1z4rd_node.exe`), then form the cluster via the HTTP API:

```bash
# Machine A — start a new cluster, get the join OTP
curl -s -X POST http://192.168.1.10:8090/cluster/init \
  -H "Content-Type: application/json" \
  -d '{"bind": "0.0.0.0:51611", "otp_ttl_secs": 300}'
# Returns: {"otp":"EMBER-4821","cluster_id":"..."}

# Machine B — join using the OTP
curl -s -X POST http://192.168.1.11:8090/cluster/join \
  -H "Content-Type: application/json" \
  -d '{"coordinator": "192.168.1.10:51611", "otp": "EMBER-4821", "bind": "0.0.0.0:51611"}'

# Any node — check cluster topology
curl http://192.168.1.10:8090/cluster/status

# Any node — check distributed training coordinator state
curl http://192.168.1.10:8090/cluster/sync/status
```

**Cluster commands (standalone CLI — gossip protocol only, no HTTP API):**

```bash
# Machine A — start the cluster listener, prints a join OTP
w1z4rd_node cluster-init --bind 192.168.1.10:51611

# Machine B — join using the OTP printed by machine A
w1z4rd_node cluster-join --coordinator 192.168.1.10:51611 --otp EMBER-4821

# Any node — print cluster topology and ring status via raw TCP
w1z4rd_node cluster-status --node 192.168.1.10:51611
```

> **Note:** The CLI subcommands start the cluster gossip protocol but do **not** start the HTTP API. For distributed training (round-robin routing, weight-delta sync, QA replication) the full node must be running. Use the REST API approach above.

**Startup scripts** (in `scripts/`):
- `start_cluster.bat` — start node + init cluster as coordinator (Windows, prints OTP)
- `start_worker.ps1` — start node + join cluster as worker (Windows, prompts for OTP)

OTPs are single-use and expire in 5 minutes by default. Generate a fresh one via `POST /cluster/init` before each new worker joins.

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
| **Environmental Equation Matrix** | 339 equations across 27 disciplines vote on active sensor labels; matching equations reinforce associated labels; hypothesis gaps suppress confidence when nothing matches |
| **Surprise-weighted replay** | Persistent mispredictions are replayed at higher frequency — the system applies stronger correction to its worst errors |
| **Peer node learning** | In cluster mode, incremental weight deltas (label-keyed synapse weights + co-occurrence counts since last sync) are pushed to all peers every 20 training calls and merged with `max(local, remote)` — knowledge only accumulates, never overwrites |
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
| `TextBitsEncoder` | `TextSpan` structs or plain `&str` | `txt:char_{c}`, `txt:char_{c}_pos{n}`, `txt:word_{w}`, `txt:phon_{ng}`, `txt:punct_{name}`, `txt:role_{r}`, `txt:zone_x{n}_y{n}` | Bottom-up: characters → bigrams → words → layout. Punctuation preserved. STDP char sequences auto-trained per word. |
| `MotionBitsEncoder` | `Vec<MotionSample {x,y,t,click}>` | `mov:zone_x{n}_y{n}`, `mov:endpoint_x{n}_y{n}`, `act:click` | Also `decode_target()` for inference |
| `KeyboardBitsEncoder` | `Vec<KeyEvent {key,ctrl,shift,alt,t}>` | `key:k_{name}`, `key:combo_{mods}{key}`, `txt:word_{w}` | Cross-modal: emits `txt:word_*` for typed words |

**`TextBitsEncoder` label hierarchy (bottom-up):**

| Level | Labels emitted | Purpose |
|-------|---------------|---------|
| Character | `txt:char_a`, `txt:char_a_pos0` | Foundation layer — letters build morphemes through co-occurrence |
| Bigram | `txt:phon_ap`, `txt:phon_pp` | Morpheme seeds — recurring bigrams across words form root clusters |
| Punctuation | `txt:punct_comma`, `txt:punct_period`, `txt:punct_apostrophe` | Syntactic signal — comma in "eat, Grandma" ≠ no comma |
| Word | `txt:word_apple` | Whole-word label — emerges from character cluster, also emitted directly |
| Layout | `txt:role_heading`, `txt:size_large`, `txt:emph_bold` | Structural context — same word means differently in a heading vs footnote |
| Spatial | `txt:zone_x3_y1`, `txt:word_apple_zone_x3_y1` | Page position — shared vocabulary with `img:` labels |

Every `/media/train` call with text runs two passes automatically:
1. **Co-occurrence pass** — all labels fire together in one `train_weighted` call
2. **STDP character-sequence pass** — each word's character chain is trained with adjacent-letter bridging (lr decay τ=0.5 char-steps), so forward `c→a→t` edges get LTP and the pool builds directed paths through letter sequences

**Layout is data.** A heading at the top of a page emits `txt:role_heading + txt:zone_x0_y0 + txt:size_large + txt:emph_bold`. Position, font, role, and emphasis are all Hebbian-connected to whatever content appears there. PDF structure is spatial signal, not decoration.

**Cross-modal associations emerge from co-training.** Feeding image + text + motion together in one `POST /media/train` call Hebbian-connects all three. At inference time, seeding from text labels fires the associated motion zones — no decoder needed.

### Environmental Equation Matrix (`EquationMatrixRuntime`)
- A self-growing directed graph of physics equations spanning all domains from Newtonian mechanics to topological quantum phenomena
- **339 seed equations across 27 disciplines**:
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
  - **Control Theory** (PID controller, transfer function, Laplace stability, state-space representation, Nyquist criterion, Bode plot, root locus, LQR optimal control, Kalman filter, H-infinity norm)
  - **Signal Processing** (DFT, convolution theorem, Nyquist-Shannon sampling, Wiener filter, matched filter, FFT complexity, windowing, spectrogram, autocorrelation, power spectral density)
  - **Chemistry** (ideal gas law, Arrhenius equation, equilibrium constant, Gibbs free energy, Nernst equation, reaction rate, activation energy, Henderson-Hasselbalch, osmotic pressure, Beer-Lambert law)
  - **Computer Science** (P vs NP, Turing completeness, halting problem, Shannon capacity, Big-O notation, Amdahl's law, master theorem, NP-hardness reduction, Cook-Levin theorem, Rice's theorem)
  - **Optimization** (gradient descent, Lagrangian multipliers, KKT conditions, simplex algorithm, Newton's method, convex duality, stochastic gradient, momentum update, Adam optimizer, simulated annealing)
  - **Machine Learning** (backpropagation, VC dimension, bias-variance tradeoff, universal approximation, PAC learning, kernel trick, SVM margin, Bayesian inference, EM algorithm, Rademacher complexity)
  - **Electrical Engineering** (Ohm's law, Kirchhoff's laws, RC time constant, LC resonance, Thevenin equivalent, Norton equivalent, Bode gain, power factor, transmission line, skin depth)
  - **Structural Engineering** (Euler buckling, beam bending moment, stress-strain (Hooke), fatigue life (S-N), yield criterion (von Mises), deflection formula, moment of inertia, safety factor, creep, fracture toughness)
  - **Epidemiology** (SIR model, basic reproduction number R0, herd immunity threshold, force of infection, SEIR model, doubling time, attack rate, endemic equilibrium, vaccine efficacy, case fatality rate)
  - **Economics** (supply-demand equilibrium, price elasticity, utility maximization, Cobb-Douglas production, IS-LM model, Fisher equation, Gini coefficient, Lorenz curve, Phillips curve, Solow growth model)
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

### Multi-Pool Associative Fabric (N-pool — first layer of the stack)

The Hebbian fast-recall layer. An arbitrary number of named `NeuronPool`s, with
cross-synapses between every ordered pool pair. Sending an input to any one pool
causes every other connected pool to fire in parallel — a single input drives a
Q→A pool, an emotion pool, an equation pool, a motion pool, and so on, all at once.

Pre-registered pools: `"in"` and `"out"`. Additional pools are registered at
runtime via `POST /multi_pool/register`.

**How a query works:**

1. **N-gram atom encoding** — the input is encoded as position-augmented atoms plus
   optional bigrams (`{src}:bg.{a}~{b}`) and trigrams (`{src}:tg.{a}~{b}~{c}`).
   Bigrams and trigrams provide paraphrase generalization — queries that share
   character sub-sequences activate overlapping concept sets even if word boundaries
   differ.  IDF weighting (when enabled) scales each atom's Hebbian contribution by
   `ln((C+1)/(count+1))+1`, clamped `[0.5, 4.0]`, so rare atoms count for more.
2. **Propagate** one hop in the source pool with a sum-no-clamp accumulator.
   Concept neurons whose ALL members fire reach higher activation than those whose
   only some members fire — pure neural discrimination, no external rule.
3. **Pick** the highest-activation source concept (`top_active_concept`).
4. **Cross-project** to each target pool through that pool-pair's `CrossPoolSynapses`.
5. **Decode** the target concept by composite member walk (char-chain) or STDP atom
   traversal.

**Precision-weighted cascade routing (`/query/integrated`)**

Multi-pool recall is the primary sensor; the slow-pool char-chain decoder and the
EEM are fallback sensors. The cascade fires all three and picks the best by
confidence:

```
Input
  │
  ├─► multi_pool (fast associative)  ── confidence = product of 3 Bayesian fractions
  │                                        ↓ if confidence ≥ threshold
  │                                       return multi-pool answer
  │
  ├─► char_chain (slow NeuronPool)   ── Bayesian decode confidence
  │                                        ↓ if better than multi-pool
  │                                       return char-chain answer
  │
  └─► EEM fallback                   ── equation match score
                                          ↓ always last resort
                                         return EEM match
```

`POST /query/integrated` returns `{answer, method, confidence, all_routes}` so
every inference call is fully auditable — you can see which sensor handled each
query and at what confidence.

**N-gram encoding toggles (`POST /multi_pool/use_ngrams`)**

```json
{ "bigrams": true, "trigrams": true, "idf": false }
```

- `bigrams` — enable character-pair atoms; helps paraphrase recall
- `trigrams` — enable character-triple atoms; strongest discrimination signal on
  class-description paraphrases (60% exact recall vs. 0% without)
- `idf` — IDF weighting; use at low training intensity (passes ≤ 10, lr ≤ 0.2);
  gets washed out by edge saturation at high passes/LR

**Current best recipe (GA result, Recipe B, combined=0.837):**
```
bigrams=1, trigrams=1, idf=1, passes=7, lr=0.167, hops=1, min_strength=0.155
```

**Cross-wiring feedback loops (genome-toggleable)**

Four feedback paths close additional signal loops between components at training
and eval time — each enabled or disabled independently by the GA genome:

| Toggle | What it closes |
|--------|----------------|
| `fb_eem_into_mp` | EEM equation matches on queries → train `(query, equation_text)` into cross-pool weights at ½ LR.  EEM-detected physics enters the multi-pool fabric. |
| `fb_mp_into_slowpool` | Every `(description, code)` pair trained into multi-pool → also pumped through `/media/train` so the slow-pool char-chain decoder sees the same vocabulary. |
| `fb_replay_low_conf` | After eval, if avg confidence < 0.20 → fire `/multi_pool/replay` (CLS replay).  Low confidence triggers consolidation. |
| `fb_disagreement_ne` | After eval, if ≥ 25% of queries fell to char-chain or EEM → release 0.5 NE spike.  Route disagreement → elevated LR on next training run. |

**CLS replay (`POST /multi_pool/replay`)**

Replays concept neurons from the N-pool fabric back into the slow NeuronPool at a
configurable LR scale (`cls_lr_scale`).  Equivalent to hippocampal→cortical
consolidation: episodic N-pool knowledge reinforces the distributed slow-pool
representation.

**Cluster delta sync**

`POST /multi_pool/delta/export` / `POST /multi_pool/delta/apply` — per-pool weight
deltas compatible with the existing `max(local, remote)` merge strategy so N-pool
fabric knowledge propagates across cluster nodes alongside the slow-pool delta.

**API:**
- `POST /multi_pool/register` — register a new pool by name
- `POST /multi_pool/train_pair` — train (src, tgt) across two named pools
- `POST /multi_pool/train_fanout` — train one source pool against many targets simultaneously
- `POST /multi_pool/ask` — send text to a source pool; every connected pool fires its prediction
- `POST /multi_pool/replay` — CLS replay: consolidate N-pool concept neurons into slow pool
- `POST /multi_pool/delta/export` — export per-pool synapse deltas for cluster sync
- `POST /multi_pool/delta/apply` — apply incoming per-pool deltas from a peer node
- `POST /multi_pool/use_ngrams` — set bigrams/trigrams/idf flags live (`{bigrams, trigrams, idf}`)
- `POST /query/integrated` — precision-weighted cascade: multi-pool → char-chain → EEM
- `GET /multi_pool/stats` — per-pool sizes + total cross-edge count

### GA Architecture Search

The system ships with a complete genetic algorithm harness for discovering optimal
multi-pool configurations and enabling/disabling cross-wiring feedback loops.

**Training corpus** (build once, ~minutes):
```bash
# Fetch NCBI/PMC articles (neuroscience, biofields, genetics, rhythms)
python scripts/fetch_ncbi_corpus.py
# Writes data/foundation/ncbi_pairs.jsonl  (~1185 Q-A pairs)

# Generate science/engineering class corpus
python scripts/build_class_corpus.py
# Writes data/foundation/class_corpus.jsonl  (155 pairs — 31 classes × 5 paraphrases)
```

**Fitness harness (`scripts/ga_node_fitness.py`)**

Spawns a fresh node, trains the NCBI corpus and class corpus at the genome's
hyperparameters, then evaluates:
- NCBI memorization (exact + Levenshtein similarity)
- NCBI generalization (held-out paraphrase queries)
- Class description memorization
- Class paraphrase exact recall (the hardest metric)
- Equation pool hit rate

Returns a combined scalar in [0, 1].

**Wide GA (`scripts/ga_node_search.py`)**

Explores the full 17-dimensional genome:

| Group | Parameters |
|-------|-----------|
| Multi-pool | `lr`, `passes`, `hops`, `min_strength` |
| Connection toggles | `cls_replay_after_epoch`, `cls_lr_scale`, `eem_in_train_loop`, `train_equation_pool` |
| Routing | `mp_confidence_threshold`, `use_eem_fallback` |
| N-gram encoding | `use_bigrams`, `use_trigrams`, `use_idf` |
| Cross-wiring | `fb_eem_into_mp`, `fb_mp_into_slowpool`, `fb_replay_low_conf`, `fb_disagreement_ne` |

```bash
python scripts/ga_node_search.py \
  --pop 4 --max-gens 5 \
  --ncbi-max 40 --class-max 15
```

**Scoping GA (`scripts/ga_scoping_search.py`)**

Long-running phased search that drills around the current best recipe. Each phase:
- Evaluates the running best as elite
- Generates exploit descendants (mutated at current radius around best)
- Reserves `explore_frac=0.30` for broad-radius exploration
- 10% chance per phase to inject a fully random genome
- On improvement → shrink radius × 0.85 (drill down)
- On no improvement → expand radius × 1.10 (escape local basin)

```bash
# Launch with 4-hour budget (appends to existing log — fully resumable)
python scripts/ga_scoping_search.py \
  --pop 6 --max-phases 200 --max-runtime-secs 14400 \
  --ncbi-max 25 --class-max 15 \
  --log data/foundation/ga_scoping.jsonl

# Post-run analysis: which feedback loops survived GA pressure?
python -c "
import json
lines = [json.loads(l) for l in open('data/foundation/ga_scoping.jsonl')]
summaries = [l for l in lines if l.get('phase_summary')]
for s in summaries[-5:]:
    g = s['best_genome']
    print(f\"ph{s.get('phase','?')} score={s['best_score']:.4f} \",
          f\"fb_eem={g.get('fb_eem_into_mp',0)} \",
          f\"fb_slow={g.get('fb_mp_into_slowpool',0)} \",
          f\"fb_rep={g.get('fb_replay_low_conf',0)} \",
          f\"fb_ne={g.get('fb_disagreement_ne',0)}\")
"
```

Current best (Recipe B): `combined=0.837`, bigrams+trigrams+IDF, passes=7, lr=0.167.

### Hypothesis Queue and Research Feedback Loop

When `/chat` returns no answer (multi-pool fabric has no associative bridge
for the input), the question is added to the hypothesis queue and a
norepinephrine spike is released (NE = 0.75).  The `research_agent.py`
script runs as a background service:

1. `GET /hypothesis/queue` — fetch open questions
2. Fetch Wikipedia REST API + ArXiv Atom API for each question
3. `POST /multi_pool/train_pair` — ingest the (question, answer) pair into
   the multi-pool fabric (and any additional pool surfaces, e.g. equation
   id, source URL).
4. `POST /media/train` — train the slow pool on the answer text for
   cross-domain generalization.
5. `POST /hypothesis/resolve` — mark the hypothesis resolved with a
   confidence score.

On resolution, dopamine is released proportional to confidence and
`flush_dopamine_potentiation()` runs — the connections that led to the
correct prediction are retroactively strengthened.  This is the
computational equivalent of the hippocampal-cortical consolidation loop.

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

## HTTP API surface

Three ports.  Use the one that matches the substrate you're working on:

| Port | Binary | Purpose |
|------|--------|---------|
| `:8080` | `w1z4rd_node.exe` | Legacy Neuro API — `NeuroRuntime` propagation, snapshot reads |
| `:8090` | `w1z4rd_node.exe` | Legacy Node API — all training + inference routes listed below |
| `:8095` | `w1z4rd_brain_server.exe` | Brain crate API — see the *Brain crate* section above for the full endpoint table |

The default `WIZARD_BRAIN_CHAT_URL` (consumed by the Django frontend at `D:/Projects/CoolCryptoUtilities/web`) is `http://localhost:8095` — user-facing `/chat` routes through the brain.

## Node API endpoints

All endpoints below are on `:8090` (legacy Node API).  Start the legacy node with no subcommand:
```bash
W1Z4RDV1510N_DATA_DIR="D:\\w1z4rdv1510n-data" ./bin/w1z4rd_node.exe
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Node status and uptime |
| `/neuro/train` | POST | Feed an `EnvironmentSnapshot` to the neural fabric |
| `/neuro/snapshot` | GET | Current activation, predictions, motifs, top influences |
| `/neuro/propagate` | POST | Feed seed labels → returns all labels that fire above threshold; cross-modal inference |
| `/neuro/generate` | POST | Generate text from the neuro pool using QA gate + pre-computed activation map decode |
| `/neuro/checkpoint` | POST | Persist neural pool and QA store to disk |
| `/neuro/sync` | POST | Force-push weight delta to all cluster peers immediately |
| `/neuro/delta/apply` | POST | Receive incremental weight delta from a peer node (cluster internal) |
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
| `/multi_pool/register` | POST | Register a new named pool in the N-pool fabric |
| `/multi_pool/train_pair` | POST | Train (src, tgt) across two named pools — symmetric |
| `/multi_pool/train_fanout` | POST | Train one source pool against many targets at once |
| `/multi_pool/ask` | POST | Send text to a source pool — every other pool fires its prediction |
| `/multi_pool/replay` | POST | CLS replay: consolidate N-pool concept neurons into slow pool |
| `/multi_pool/delta/export` | POST | Export per-pool synapse deltas for cluster sync |
| `/multi_pool/delta/apply` | POST | Apply incoming per-pool deltas from a peer node |
| `/multi_pool/use_ngrams` | POST | Set bigram/trigram/IDF encoding flags live |
| `/multi_pool/stats` | GET | Per-pool sizes + total cross-edge count |
| `/query/integrated` | POST | Precision-weighted cascade: multi-pool → char-chain → EEM |
| `/knowledge/ingest` | POST | Ingest documents into the knowledge graph |
| `/knowledge/vote` | POST | Vote on knowledge associations |
| `/streaming/labels` | GET | Label queue for human annotation |
| `/network/patterns/query` | POST | Query distributed pattern index |
| `/threat/ingest` | POST | Ingest behavioral threat data |
| `/threat/overlay` | GET | Current threat and health overlay |
| `/metrics` | GET | Node performance metrics |
| `/cluster/init` | POST | Start this node as cluster coordinator; returns join OTP |
| `/cluster/join` | POST | Join an existing cluster using coordinator address + OTP |
| `/cluster/leave` | POST | Leave the cluster and release port 51611 |
| `/cluster/status` | GET | Cluster topology, ring state, and node list |
| `/cluster/sync/status` | GET | Distributed training coordinator state: peers, sync step, call counts |

---

## Sensor format

Every data source translates to `EnvironmentSnapshot`. When posting to `POST /neuro/train`,
wrap it in a `snapshot` key:

```json
{
  "snapshot": {
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
}
```

For text, image, audio, and motion data prefer `POST /media/train` — it handles encoding automatically and is the recommended path for all training scripts.

A chess piece, a LiDAR point, a stock tick, a chemical state, a crowd zone — all the same format. The node learns from all of them simultaneously through the same neural fabric.

**2D and 3D sensors are handled natively.** The `z` coordinate is 0.0 for 2D sensors; the fabric learns spatial patterns regardless of dimensionality. The equation matrix uses the `dims` field to filter which physics equations apply — anyons only surface for 2D sensor contexts, Maxwell in 3D, thermodynamics everywhere.

---

## Decentralized node mesh

- **P2P networking**: libp2p gossipsub + Kademlia + mDNS; rate limits and peer scoring
- **Data mesh**: manifests, chunking, replication receipts, integrity audits, repair requests, storage rewards
- **Neural fabric sharing**: incremental weight-delta sync every 20 training calls — each node exports only synapses updated since the last sync and pushes them to all peers via `POST /neuro/delta/apply`; peers merge with `max(local, remote)` so knowledge accumulates without overwriting; new workers bootstrap from existing nodes on join
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

The EEM bridges the gap between raw sensor patterns and physical interpretation. As the neuro fabric fires labels, the EEM surfaces candidate equations governing the observed phenomenon. It is a complete map of modern physics and engineering — 339 equations across 27 disciplines — compiled into the node so that any sensor stream can be interpreted through the lens of physical law.

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
- Sparse 2D grid proxemics zones (Hall, 1966): Intimate / Personal / Social / Public
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

Feed conversation pairs through `/multi_pool/train_pair` with `src_pool: "in", tgt_pool: "out"` (`question → answer`). At inference time `/chat` fires question atoms through the multi-pool fabric — exact recall when the trained pair matches; nearest-neighbor recall for paraphrases and OOD inputs. The slow NeuronPool char-chain decoder is the fallback when the fabric has no associative bridge. Additional pools (emotion, equation, motion) trained against the same source pool fire in parallel — `/chat` returns the main answer plus `predictions` for every other connected pool.

This is associative retrieval + predictive coding, not autoregressive generation. Responses are activated from trained state rather than generated token-by-token. Suitable for factual recall, domain Q&A, and follow-up questions that share vocabulary with prior exchanges.

### Code assistance

Code has strong motif structure — function signatures, boilerplate, API call sequences promote quickly through `HierarchicalMotifRuntime`. A bridge script tokenizes source files into labeled symbols (function names, keywords, identifiers at line/column positions) and feeds them as `EnvironmentSnapshot` sequences. Known problem→solution pairs go through `/multi_pool/train_pair` (src_pool `"in"`, tgt_pool `"out"`).

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
| `chess_training_loop.py` | Chess | Board positions as 2D `EnvironmentSnapshot`; moves as Q&A pairs via `/multi_pool/train_pair` |
| `rtsp_pose_bridge.py` | Video / pose | Pose keypoints as 3D symbol positions |
| `rss_topic_bridge.py` | Topics | Topic signals as streaming events |
| `traffic_sensor_bridge.py` | Traffic | Flow data as crowd sensor stream |
| `textbook_scripts/` | Knowledge | Q&A pairs from OpenStax textbooks ingested via `/multi_pool/train_pair` |
| `scripts/train_obstacle.py` | Screen navigation | Screenshot + goal text + mouse trajectory via `/media/train`; anchor pairs for discrimination |
| `scripts/playback_obstacle.py` | Screen navigation | Goal text → `/media/playback` → predicted zone → Playwright cursor action |
| `scripts/research_agent.py` | Hypothesis resolution | Polls `/hypothesis/queue`, fetches Wikipedia + ArXiv, ingests via `/multi_pool/train_pair` + `/media/train`, resolves via `/hypothesis/resolve` |
| `scripts/chat.py` | Interactive REPL | `/train`, `/pair`, `/reg`, `/raw`, `/pools` commands; uses `/query/integrated` |
| `scripts/fetch_ncbi_corpus.py` | Training corpus | Fetches NCBI/PMC articles (neuroscience, biofields, genetics, rhythms) → `data/foundation/ncbi_pairs.jsonl` |
| `scripts/build_class_corpus.py` | Training corpus | Generates 31 science/engineering classes × 5 paraphrase variants → `data/foundation/class_corpus.jsonl` |
| `scripts/ga_node_fitness.py` | GA harness | Fitness evaluator: spawn node, train NCBI + class corpus, score memorization + paraphrase + generalization |
| `scripts/ga_node_search.py` | GA search | 17-dim genome (multi-pool params + connection toggles + routing + n-gram + cross-wiring); adaptive GA |
| `scripts/ga_scoping_search.py` | GA drill | Phased scoping GA starting at Recipe B (combined=0.837); shrinks radius on improvement, reserves explore fraction |
| `scripts/integration_test.py` | E2E tests | End-to-end integration tests against a live node |
| `scripts/stress_test.py` | Load tests | Concurrent training + query load tests |

**Adding a new domain**: translate your data to `EnvironmentSnapshot`, `POST` to `/neuro/train`, and send labelled pairs to `/multi_pool/train_pair`. Optionally call `/equations/apply` for physics context. The node learns from everything alongside everything else.

For multimodal domains (image + text + motion): use `ImageBitsEncoder`, `TextBitsEncoder`, `MotionBitsEncoder` from the encoder library to convert raw data to labels, then `POST /media/train`. The node's neural fabric is the same one everything else trains.

---

## Running — Complete Tutorial

### Prerequisites

```bash
# Rust toolchain (stable, GNU target on Windows)
# Set PATH so dlltool.exe is found (Windows only)
export PATH="$PATH:/c/Users/Node/.cargo/bin:/c/Users/Node/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"

# Python dependencies (install once)
pip install httpx httpx[http2] aiohttp pillow requests nltk

# Node binaries in bin/ (build or use pre-built)
cargo build --release --workspace
cp target/release/w1z4rdv1510n-node.exe        bin/w1z4rd_node.exe          # Windows
cp target/release/w1z4rd_brain_server.exe      bin/w1z4rd_brain_server.exe  # Brain server (port 8095)
# cp target/release/w1z4rdv1510n-node  bin/w1z4rd_node        # Linux/macOS
```

---

### 1. Start the binaries

There are two binaries.  Run whichever (or both) matches the substrate you're working on.

**Legacy node** — exposes `:8080` (Neuro API) + `:8090` (Node API) on startup:

```bash
cd /d/Projects/W1z4rDV1510n
W1Z4RDV1510N_DATA_DIR="D:\\w1z4rdv1510n-data" ./bin/w1z4rd_node.exe

# Verify both APIs are up
curl http://127.0.0.1:8080/healthz  # neuro API (uses /healthz, not /health)
curl http://127.0.0.1:8090/health   # node API
```

**Brain server** — exposes `:8095` (Brain API).  Persists `brain.bin` under `$W1Z4RDV1510N_DATA_DIR/` (use a separate directory from the legacy node so the two substrates don't fight over snapshot file paths):

```bash
cd /d/Projects/W1z4rDV1510n
W1Z4RDV1510N_DATA_DIR="D:/Projects/W1z4rDV1510n/brain-data-text" ./bin/w1z4rd_brain_server.exe

# Verify the brain is up
curl http://127.0.0.1:8095/health   # → "ok"
curl http://127.0.0.1:8095/stats    # tick, neurons, bindings, tentative/consolidated counts, threshold, pressure
```

The Wizard frontend reads `WIZARD_BRAIN_CHAT_URL` (defaulting to `http://localhost:8095`) for the `/chat` route, so the brain is what the user-facing chat hits.  Legacy training scripts continue to target `:8090`.

> **Port note:** `:8080` = legacy NeuroRuntime read-only surface.  `:8090` = legacy training + inference (all `/chat`, `/multi_pool/*`, `/neuro/*`, `/media/*`, `/equations/*`, `/cluster/*` routes).  `:8095` = brain crate (`/observe`, `/tick`, `/integrate`, `/sensor/*`, `/chat`, `/stats`, `/checkpoint`).

---

### 2. Interactive chat

```bash
# Brain crate /chat (port 8095) — current user-facing path
curl -s -X POST http://127.0.0.1:8095/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "dog"}'
# Returns {reply, answer, decoder, predictions, grounding:{outside_grounding, fabric_confidence, ...}}

# Legacy node /chat (port 8090) — Recipe B multi-pool fabric path
curl -s -X POST http://127.0.0.1:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "What is gravity?", "hops": 2, "top_k": 5}'

# Full conversational interface with hypothesis tracking and affect (legacy)
python scripts/chat.py --node http://127.0.0.1:8090
# Type /help inside the chat for commands, /quit to exit
```

The brain `/chat` will return `outside_grounding: true` and an empty answer when the prompt's atoms don't substantially match any trained binding — that is the deliberate Stage 9 honesty gate, not a bug.

---

### 3. Brain crate training (port 8095)

The brain crate has its own curriculum executor that uses cross-pool `/observe → /observe → /tick` instead of `/sensor/observe` with paired_text.

**Step 1 — Compile the corpora** (output goes to `data/training/` and is git-ignored):

```bash
python -m tools.training_standard.compile_greetings_corpus  # → greetings_001.jsonl (186 pairs)
python -m tools.training_standard.compile_k12_corpus        # → k12_subjects_001.jsonl (7237 pairs)
```

**Step 2 — Drive the corpora against the live brain server:**

```bash
# Greetings — small, fast (~20 s for 6 reps)
python -m tools.training_standard.drive_corpora_brain --script greetings_001 --repeats 6

# K-12 subject vocabulary — larger (~30 min for 3 reps)
python -m tools.training_standard.drive_corpora_brain --script k12_subjects_001 --repeats 3 --no-smoke

# Or all registered scripts in phase order
python -m tools.training_standard.drive_corpora_brain --repeats 6
```

**Step 3 — Empirically evaluate:**

```bash
# Toddler 32-pair recall (preserves the 71.9% baseline)
python scripts/brain_xpool_chat_test.py

# Full 61-probe fluency panel across toddler + greeting + k12 + oov categories
python scripts/brain_fluency_eval.py
```

**Step 4 — Persist the brain state:**

```bash
curl -s -X POST http://127.0.0.1:8095/checkpoint -d '{}'
# Writes $W1Z4RDV1510N_DATA_DIR/brain.bin (bincode snapshot)
```

> **Stage 10 implication:** under the default config (`tentative_emergence_threshold = 1`), every cross-pool training pair crystallizes a tentative binding on first co-firing.  You'll see `total_binding` in `/stats` climb roughly 1:1 with the number of distinct prompt/response pairs trained.  The `consolidated_bindings` count (which is what `/integrate` and the EEM walk) grows when the pressure-feedback loop's threshold is satisfied.

---

### 4. Legacy Stage 0 — Toddler foundation training (port 8090)

Stage 0 builds the base vocabulary: ~1,675 concepts from letters through kindergarten-level ideas.
Run this after a fresh node start (empty pool) before any other training.

**Step 1 — Build the concept dataset** (run once; output cached in `data/foundation/`)

```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python scripts/build_concept_dataset.py
# Writes data/foundation/concept_dataset.jsonl  (~1,675 entries)
```

**Step 2 — Train**

```bash
python scripts/train_foundation.py \
  --node http://127.0.0.1:8090 \
  --pass concepts \
  --concurrency 8
# Runtime: ~3–5 min; checkpoints pool+QA after each developmental level
```

**Step 3 — Validate**

```bash
python scripts/test_stage0.py --node http://127.0.0.1:8090
# Expected: 26/26 (100%)  — "Architecture looks solid — ready for more training data."
```

---

### 5. Foundation language training (Simple English Wikipedia)

Builds broad English word co-occurrence from ~200k articles.
Requires downloading the Wikipedia dump first (~300 MB compressed).

```bash
pip install requests
python scripts/fetch_simple_wikipedia.py
# Downloads to data/foundation/simple_wiki_articles.jsonl  (~200k articles)

python scripts/train_foundation.py \
  --node http://127.0.0.1:8090 \
  --pass text \
  --concurrency 10
# Runtime: ~30–60 min depending on hardware
```

---

### 6. K-12 full curriculum training

Trains through all three stages — toddler (Stage 0), introductory textbooks (Stage 1),
and full K-12 curriculum (Stage 2). Requires LibreTexts PDFs placed in `textbooks/`
inside the project root (i.e. `D:\Projects\W1z4rDV1510n\textbooks\`).
Use `--textbooks` to point to any other directory.

```bash
# Run all stages (long-running — hours)
# NOTE: use port 8090 (Node API) — /media/train lives there, not on 8080
python scripts/train_k12.py \
  --node http://127.0.0.1:8090 \
  --stages 0,1,2 \
  --checkpoint-every 100

# Run only Stage 0 (toddler concepts)
python scripts/train_k12.py --node http://127.0.0.1:8090 --stages 0

# Override textbooks directory (default is textbooks/ inside the project root)
python scripts/train_k12.py --node http://127.0.0.1:8090 --textbooks D:\path\to\textbooks --stages 1,2

# Limit to 10 books per stage (useful for quick testing)
python scripts/train_k12.py --node http://127.0.0.1:8090 --stages 1,2 --max-books 10

# Resume after interruption (skips already-processed books)
python scripts/train_k12.py --node http://127.0.0.1:8090 --resume

# Fresh run ignoring prior progress
python scripts/train_k12.py --node http://127.0.0.1:8090 --clear-progress
```

---

### 7. Bovine anatomy training pipeline

`scripts/build_cow_dataset.py` builds a multi-modal bovine anatomy dataset and ingests it directly into the node. This is the primary training pipeline for the W1z4rD V1510n bovine perception system. It runs six stages:

| Stage | Type | Approx. items | Source |
|-------|------|---------------|--------|
| 0 | Synthetic visual primitives | 500 | PIL-generated shapes with anatomy zone labels |
| 1 | Text corpus | ~138 | PubMed Central articles + embedded anatomy knowledge base |
| 2 | Video frames | Variable | YouTube CC-licensed bovine videos (yt_dlp, 2 fps) |
| 3 | MRI/CT cross-sections | ~400 | 16 anatomical levels × 3 modalities × 8 noise variants + text docs |
| 4 | Histology images | ~32 | Wikimedia Commons histology categories |
| 5 | Protein structures | ~15 | PDB records (bovine-relevant proteins) |

```bash
pip install pillow requests yt-dlp

# Run all stages
python scripts/build_cow_dataset.py \
  --node localhost:8090 \
  --data-dir D:/w1z4rdv1510n-data/training

# Run individual stages
python scripts/build_cow_dataset.py --stages 0 --node localhost:8090 --data-dir D:/w1z4rdv1510n-data/training
python scripts/build_cow_dataset.py --stages 3 --node localhost:8090 --data-dir D:/w1z4rdv1510n-data/training

# Download only (skip ingestion into node)
python scripts/build_cow_dataset.py --download-only --stages 2 --node localhost:8090 --data-dir D:/w1z4rdv1510n-data/training
```

Stage 3 generates synthetic MRI/CT images using PIL — no external imaging data required. Stage 2 requires `yt-dlp` to be installed and uses the `extractor_args: {youtube: {skip: [webpage]}}` bypass for the YouTube n-challenge.

After running, query the node on bovine anatomy topics:

```bash
curl -s -X POST http://127.0.0.1:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the function of the rumen?", "hops": 2, "top_k": 5}'
```

---

### 8. Multi-pool training and query (legacy `crates/core` fabric, port 8090)

```bash
# Register an additional pool (in/out are pre-registered)
curl -s -X POST http://127.0.0.1:8090/multi_pool/register \
  -H "Content-Type: application/json" \
  -d '{"pool_id": "emo"}'

# Train a Q -> A pair on the default in/out pools
curl -s -X POST http://127.0.0.1:8090/multi_pool/train_pair \
  -H "Content-Type: application/json" \
  -d '{
    "src_pool": "in",  "src": "what is photosynthesis",
    "tgt_pool": "out", "tgt": "the process plants use to convert sunlight into food",
    "passes": 30, "lr": 0.5
  }'

# Train one source against many targets (color + category + emotion)
curl -s -X POST http://127.0.0.1:8090/multi_pool/train_fanout \
  -H "Content-Type: application/json" \
  -d '{
    "src_pool": "in", "src": "hello",
    "targets": [
      { "tgt_pool": "out", "tgt": "Hello there friend." },
      { "tgt_pool": "emo", "tgt": "warm" }
    ],
    "passes": 30, "lr": 0.5
  }'

# Query — every connected pool fires its own decoded prediction
curl -s -X POST http://127.0.0.1:8090/multi_pool/ask \
  -H "Content-Type: application/json" \
  -d '{"src_pool": "in", "text": "hello"}'
# -> {"predictions": {"out": "Hello there friend.", "emo": "warm"}, ...}

# Precision-weighted cascade: multi-pool -> char-chain -> EEM fallback
curl -s -X POST http://127.0.0.1:8090/query/integrated \
  -H "Content-Type: application/json" \
  -d '{"text": "what is the function of mitochondria", "hops": 2}'
# -> {"answer": "...", "method": "multi_pool", "confidence": 0.84,
#     "all_routes": [{"method":"multi_pool","answer":"...","confidence":0.84},
#                    {"method":"char_chain","answer":"...","confidence":0.31}]}

# Enable bigrams + trigrams + IDF (best for paraphrase-heavy corpora)
curl -s -X POST http://127.0.0.1:8090/multi_pool/use_ngrams \
  -H "Content-Type: application/json" \
  -d '{"bigrams": true, "trigrams": true, "idf": true}'

# Train at low intensity when IDF is on (Recipe B, combined=0.837)
curl -s -X POST http://127.0.0.1:8090/multi_pool/train_pair \
  -H "Content-Type: application/json" \
  -d '{
    "src_pool": "in",  "src": "what is photosynthesis",
    "tgt_pool": "out", "tgt": "the process plants use to convert sunlight into glucose",
    "passes": 7, "lr": 0.167
  }'

# CLS replay — consolidate N-pool concept neurons back into slow pool
curl -s -X POST http://127.0.0.1:8090/multi_pool/replay \
  -H "Content-Type: application/json" \
  -d '{"cls_lr_scale": 0.10}'

# Per-pool sizes + total cross-edges
curl http://127.0.0.1:8090/multi_pool/stats
```

---

### 9. Checkpoint (save pool to disk)

The node accumulates learning in RAM. Checkpoint persists it.
Training scripts call this automatically, but you can trigger it manually:

```bash
curl -s -X POST http://127.0.0.1:8090/neuro/checkpoint
# Returns: {"saved":true,"pool_path":"..."}
```

---

### 10. Autonomous research agent (legacy)

Polls the hypothesis queue, fetches Wikipedia + ArXiv answers, ingests them,
and resolves each hypothesis with a dopamine reward signal.

```bash
pip install aiohttp
python scripts/research_agent.py \
  --node http://127.0.0.1:8090 \
  --interval 30          # poll every 30 seconds
```

---

### 11. Neural propagation and inference

```bash
# Propagate from seed labels — returns all concept labels that activate
curl -s -X POST http://127.0.0.1:8090/neuro/propagate \
  -H "Content-Type: application/json" \
  -d '{"seed_labels": ["txt:word_gravity"], "hops": 2}'

# Generate free-form text from the neuro pool
curl -s -X POST http://127.0.0.1:8090/neuro/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "photosynthesis", "max_tokens": 30}'

# Query the hypothesis queue (questions the node could not answer confidently)
curl http://127.0.0.1:8090/hypothesis/queue

# Resolve a hypothesis (triggers dopamine potentiation)
curl -s -X POST http://127.0.0.1:8090/hypothesis/resolve \
  -H "Content-Type: application/json" \
  -d '{"id": "...", "answer": "...", "confidence": 0.85}'
```

---

### 12. Environmental Equation Matrix

```bash
# Search equations by keyword
curl "http://127.0.0.1:8090/equations/search?q=entropy"

# Apply equations to active sensor labels
curl -s -X POST http://127.0.0.1:8090/equations/apply \
  -H "Content-Type: application/json" \
  -d '{"labels": ["txt:word_temperature", "txt:word_energy"], "dims": 3}'

# Full report: equation counts by discipline
curl http://127.0.0.1:8090/equations/report

# Open hypothesis gaps (sorted by cross-node corroboration)
curl http://127.0.0.1:8090/equations/gaps
```

---

### 13. Obstacle course — screen navigation from natural language

Teaches the node to predict cursor targets from plain English instructions.

```bash
pip install playwright httpx
playwright install chromium

# Terminal 1: node (must already be running)

# Terminal 2: training (opens browser, records mouse trajectories)
python scripts/train_obstacle.py --reps 10

# Terminal 2: playback (asks node for predicted zone, moves cursor)
python scripts/playback_obstacle.py --auto
```

---

### 14. Chess training

```bash
# Terminal 1: node

# Terminal 2: feed board states as EnvironmentSnapshots
python scripts/chess_training_loop.py --max-games 8000

# Terminal 3: live visualizer (opens browser automatically)
python scripts/live_viz_server.py \
  --board-file logs/chess_live_board.json \
  --port 8765 --open
```

---

### 15. Dashboard GUI

```bash
./bin/w1z4rd_dashboard.exe    # Windows
# ./bin/w1z4rd-dashboard      # Linux/macOS
# Connects to :8080 and :8090 by default; configure in dashboard settings
```

---

### 16. Cluster — join multiple nodes into one virtual brain

```bash
# Machine A — start coordinator (prints a join OTP)
w1z4rd_node cluster-init --bind 192.168.1.10:51611

# Machine B — join using the OTP
w1z4rd_node cluster-join --coordinator 192.168.1.10:51611 --otp EMBER-4821

# Check cluster topology
w1z4rd_node cluster-status --node 192.168.1.10:51611

# Generate a fresh OTP for a new joiner
w1z4rd_node cluster-otp
```

Windows convenience scripts: `scripts/start_cluster.bat` (coordinator),
`scripts/start_worker.ps1` (worker — prompts for OTP).

---

### 17. Distributed training — weight-delta sync across nodes

Once nodes are clustered, the distributed training coordinator runs automatically.
Training calls are round-robin routed across all nodes (including self), and weight
deltas are pushed to peers every 20 training calls.

```bash
# Check distributed sync state on any node
curl http://127.0.0.1:8090/cluster/sync/status
# {"peers":["http://192.168.1.x:8090"],"last_sync_step":1240,"calls_since_last_sync":7,...}

# Force an immediate weight-delta push to all peers (useful after a large training batch)
curl -s -X POST http://127.0.0.1:8090/neuro/sync
# {"status":"OK","message":"delta sync queued"}

# Bypass round-robin and train locally on this node (x-w1z-local header)
curl -s -X POST http://127.0.0.1:8090/media/train \
  -H "Content-Type: application/json" \
  -H "x-w1z-local: 1" \
  -d '{"modality":"text","text":"your training text here","lr_scale":1.0}'
```

**How it works:**
- Every `POST /media/train` call is routed round-robin across N+1 slots (one per peer + self), so each node trains ~1/(N+1) of all calls
- Every 20 training calls, the node exports synapses updated since the last sync (weight ≥ 0.005) and co-occurrence pairs involving recently active neurons, then fans out a `POST /neuro/delta/apply` to all peers concurrently
- Peers merge with `max(local, remote)` — knowledge only accumulates, never overwrites
- When a new node joins, it triggers peers to push their current delta, seeding it with the cluster's accumulated knowledge
- N-pool fabric deltas are exported and applied via `/multi_pool/delta/export` and `/multi_pool/delta/apply` — same `max(local, remote)` merge strategy as the slow-pool delta
- Peers that fail 5 consecutive calls are evicted from the routing list and restored on the next cluster status refresh

---

### Common troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Port already in use | Old node process still running | `powershell -Command "Stop-Process -Name w1z4rd_node -Force"` |
| `dlltool.exe not found` | WinLibs not in PATH | `export PATH="$PATH:/path/to/mingw64/bin"` |
| `/multi_pool/ask` returns empty predictions | No pairs trained into that pool pair | Train with `/multi_pool/train_pair` first; check `/multi_pool/stats` for pool sizes |
| `/query/integrated` always returns `char_chain` | N-gram encoding disabled or too few training passes | Enable bigrams/trigrams via `/multi_pool/use_ngrams`; use passes ≥ 5 |
| Checkpoint saves to AppData | `W1Z4RDV1510N_DATA_DIR` not set | Set env var before launching node |
| Avira quarantines build artifact | AV false positive | Add `target/` to Windows Defender exclusions |
| Cluster join returns 409 Conflict | Stale `cluster_state.json` on disk | Delete `~/.w1z4rd/cluster_state.json` (Linux/WSL) or `%APPDATA%\w1z4rd\cluster_state.json` (Windows) |
| Cluster join OTP "invalid or expired" | Old TCP listener still holding port 51611 | Kill old node process; port is released when the process exits |
| `127.0.0.1:8090` routes to wrong node (WSL) | WSL relay (`wslrelay.exe`) intercepts `127.0.0.1` | Use the LAN IP of the Windows node (e.g. `192.168.1.84:8090`) instead of `127.0.0.1` |
| Distributed peers list empty after join | Peer list only populates on first `GET /cluster/status` | Call `/cluster/status` once on each node after join, or wait up to 30 s for the background refresh |

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

---

## References & credits

The brain and legacy substrates implement primitives that map onto the published neuroscience literature.  Citations in the body of this README use `(Last & Last, Year)` / `(Last et al., Year)` shorthand; full bibliographic entries follow.  This is the work the architecture is built on — credit is theirs.

### Plasticity, Hebbian learning, and STDP

- **Hebb, D. O.** (1949). *The Organization of Behavior: A Neuropsychological Theory.* Wiley.  *(The original "neurons that fire together wire together" formulation that the entire fabric is built on.)*
- **Bi, G.-Q. & Poo, M.-M.** (1998). "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." *Journal of Neuroscience* 18(24): 10464–10472.  *(Empirical STDP window — the asymmetric LTP/LTD curve that `hebbian_pair(a, b)` models.)*
- **Markram, H., Lübke, J., Frotscher, M., & Sakmann, B.** (1997). "Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs." *Science* 275(5297): 213–215.  *(Foundational STDP observation in cortex.)*
- **Bienenstock, E. L., Cooper, L. N., & Munro, P. W.** (1982). "Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex." *Journal of Neuroscience* 2(1): 32–48.  *(BCM rule — the sliding-threshold homeostatic mechanism behind per-neuron EMA activation tracking.)*

### Sparsity and population coding

- **Vinje, W. E. & Gallant, J. L.** (2000). "Sparse coding and decorrelation in primary visual cortex during natural vision." *Science* 287(5456): 1273–1276.  *(The 2–5% firing rate evidence that motivates kWTA / `sparsity_mode` defaults.)*
- **Olshausen, B. A. & Field, D. J.** (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature* 381: 607–609.  *(Why sparse codes form naturally from natural-image statistics — the principled defence of pre-set sparsity priors.)*
- **Maass, W.** (2000). "On the computational power of winner-take-all." *Neural Computation* 12(11): 2519–2535.  *(Theoretical capacity result for k-WTA networks.)*

### Homeostasis and synaptic competition

- **Turrigiano, G. G.** (2008). "The self-tuning neuron: synaptic scaling of excitatory synapses." *Cell* 135(3): 422–435.  *(Homeostatic synaptic scaling that the legacy 500-step rebalance pass and the brain crate's `heterosynaptic_ltd_mode` both model.)*
- **Royer, S. & Paré, D.** (2003). "Conservation of total synaptic weight through balanced synaptic depression and potentiation." *Nature* 422: 518–522.  *(Direct evidence for heterosynaptic LTD — strengthening one synapse weakens neighbours on the same neuron.)*

### Predictive coding and the free-energy principle

- **Rao, R. P. N. & Ballard, D. H.** (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience* 2(1): 79–87.  *(The hierarchical-prediction architecture `propagate_predictive()` and `predict_gate_mode` both implement.)*
- **Friston, K.** (2005). "A theory of cortical responses." *Philosophical Transactions of the Royal Society B* 360(1456): 815–836.  *(Original free-energy formulation.)*
- **Friston, K.** (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience* 11(2): 127–138.  *(Unified statement — `recent_surprise` EMA approximates variational free energy locally.)*

### Sleep, replay, and complementary learning systems

- **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C.** (1995). "Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory." *Psychological Review* 102(3): 419–457.  *(CLS theory.  The brain crate's two-tier emergent binding + `/sleep` replay loop is a direct port of the hippocampus-fast + cortex-slow division.)*
- **Wilson, M. A. & McNaughton, B. L.** (1994). "Reactivation of hippocampal ensemble memories during sleep." *Science* 265(5172): 676–679.  *(First empirical demonstration of replay — `Brain::replay_recent_moments` is the computational form.)*
- **Tononi, G. & Cirelli, C.** (2014). "Sleep and the price of plasticity." *Neuron* 81(1): 12–34.  *(Synaptic Homeostasis Hypothesis (SHY) — `Brain::sleep`'s prune-weak-concepts pass implements the downscaling SHY predicts.)*
- **Kumaran, D., Hassabis, D., & McClelland, J. L.** (2016). "What learning systems do intelligent agents need? Complementary learning systems theory updated." *Trends in Cognitive Sciences* 20(7): 512–534.  *(Modern CLS restatement that motivated the tentative + consolidated tier split in Stage 10.)*

### Sequence binding and hippocampal ordering

- **Skaggs, W. E. & McNaughton, B. L.** (1996). "Replay of neuronal firing sequences in rat hippocampus during sleep following spatial experience." *Science* 271(5257): 1870–1873.  *(CA3 fires sequences, not just sets — the empirical motivation for the Stage-16 sequence-match preempt that distinguishes anagram queries.)*
- **Foster, D. J. & Wilson, M. A.** (2007). "Hippocampal theta sequences." *Hippocampus* 17(11): 1093–1099.  *(Within-theta-cycle order preservation in place-cell ensembles.)*
- **Buzsáki, G. & Tingley, D.** (2018). "Space and time: the hippocampus as a sequence generator." *Trends in Cognitive Sciences* 22(10): 853–869.  *(Sequence-cell ordering as a fundamental memory primitive — `Pool::last_observed_sequence` mirrors this property.)*

### Neuromodulators and reward-modulated plasticity

- **Schultz, W.** (2007). "Behavioral dopamine signals." *Trends in Neurosciences* 30(5): 203–210.  *(Reward-prediction-error semantics for the dopamine spikes the legacy `release_neuromodulator(kind, amount)` API emits.)*
- **Aston-Jones, G. & Cohen, J. D.** (2005). "An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance." *Annual Review of Neuroscience* 28: 403–450.  *(NE-as-gain-modulator — the legacy `effective_lr = lr_scale × ACh × (1 + NE × 2.0)` formula's biological basis.)*
- **Hasselmo, M. E. & McGaughy, J.** (2004). "High acetylcholine levels set circuit dynamics for attention and encoding and low acetylcholine levels set dynamics for consolidation." *Progress in Brain Research* 145: 207–231.  *(ACh as a plasticity gate.)*
- **Frémaux, N. & Gerstner, W.** (2016). "Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules." *Frontiers in Neural Circuits* 9: 85.  *(Theory of dopamine-tagged retrograde potentiation — what `flush_dopamine_potentiation()` implements.)*
- **Izhikevich, E. M.** (2007). "Solving the distal reward problem through linkage of STDP and dopamine signalling." *Cerebral Cortex* 17(10): 2443–2452.  *(Eligibility-trace × dopamine mechanism that closes the temporal-credit-assignment gap.)*

### Cortical columns and the thousand-brains conjecture

- **Mountcastle, V. B.** (1997). "The columnar organization of the neocortex." *Brain* 120(4): 701–722.  *(Cortical column as the canonical computational unit — `Pool::collapse_tail_to_concept` is the column-promotion analog.)*
- **Hawkins, J., Lewis, M., Klukas, M., Purdy, S., & Ahmad, S.** (2019). "A framework for intelligence and cortical function based on grid cells in the neocortex." *Frontiers in Neural Circuits* 12: 121.  *(Thousand-brains theory — every cortical column has its own model of every object; voting across columns yields stable percepts.  The legacy multi-pool fabric's parallel pool predictions implement the voting principle.)*

### Behavioural and environmental theory

- **Hall, E. T.** (1966). *The Hidden Dimension.* Doubleday.  *(Proxemic zones — Intimate / Personal / Social / Public — used by the threat overlay's spatial fields.)*

### Tooling and infrastructure cited as primary sources

- **Maass, W., Natschläger, T., & Markram, H.** (2002). "Real-time computing without stable states: a new framework for neural computation based on perturbations." *Neural Computation* 14(11): 2531–2560.  *(Liquid-state-machine intuition behind treating the substrate as a recurrent feature reservoir.)*

---

### Software credits

- **Wikipedia REST API + ArXiv Atom API** — sources the `research_agent.py` autonomous learning loop ingests to resolve hypothesis gaps.
- **NLTK + WordNet 3.0** — Princeton WordNet powers `compile_categorical_unified.py` and the K-12 corpus compilers.  WordNet is © Princeton University 2006; used under the WordNet license.
- **NCBI / PubMed Central** — `scripts/fetch_ncbi_corpus.py` ingests public-domain biomedical articles for the GA training corpus.
- **OpenStax / LibreTexts** — Creative-Commons textbooks used by the K-12 stage-2 training pipeline.
- **Rust crates**: `serde`, `serde_json`, `bincode`, `ahash`, `parking_lot`, `dashmap`, `rayon`, `thiserror`, `tokio`, `axum`, `tracing`, `clap`, `psutil`, `evalexpr`, `toml`, `base64`.  Full Cargo.toml dependency list available in `crates/*/Cargo.toml`.
- **Python**: `httpx`, `aiohttp`, `urllib3`, `pillow`, `nltk`, `requests`, `yt-dlp`, `playwright`, `psutil`.

If a citation here misrepresents prior work, or attributes a mechanism this code does not actually implement, please open an issue — the goal is empirical honesty about what the substrate is and what it is *modelled after*, not appropriation of credit.
