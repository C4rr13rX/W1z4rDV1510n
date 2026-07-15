# Wizard Vision Brain Configuration Field Guide

## Purpose and scope

This is the living, evidence-based guide for configuring Wizard Vision brains. It records choices made through brain identities, curricula, benchmarks, and genetic searches that improved learning, retention, integration, inference, or efficiency.

This document deliberately excludes core neurofabric implementation details. A core repair belongs here only when it changes a practical configuration limit or invalidates earlier configuration advice.

Future work must revise or remove guidance when stronger evidence contradicts it. Do not preserve a recommendation merely because it was once useful.

## Evidence labels

- **Proven live**: reproduced on the persisted programming brain and protected by a read-only benchmark.
- **Experiment-supported**: measured in a controlled comparison or genetic search, but not yet broadly replicated.
- **Provisional**: a useful working hypothesis that still needs an ablation or broader search.
- **Rejected**: tested and found harmful, misleading, or unnecessarily expensive.

## Current known-good principles

### Preserve atoms and add derived views in parallel

**Proven live.** Raw character/byte sensory pools remain the grounding substrate. Feature extraction must add parallel pools; it must not replace raw input with conventional tokens. Derived pools may represent source structure, instruction intent, execution outcome, repair delta, environment, or other stable features. Their co-firing bindings retain a path back to the atom-level episode.

This permits hierarchical compression into motifs and concepts while preserving sensitivity to deviations such as typos, malformed syntax, and small contextual changes.

### Use several feature-specific pools and bind their simultaneous evidence

**Proven live.** A coding episode is more useful when its distinct signals enter separate pools and co-fire in the same moment. The current programming identity separates at least:

- instruction text;
- source text;
- derived source structure;
- derived instruction intent;
- console before the repair;
- failure outcome;
- corrected source or response;
- repair delta;
- console after the repair;
- resolution outcome;
- execution and platform context.

The binding/integration layer can then learn relations such as `language + requested behavior + failure signature + repair + successful outcome`, rather than treating an entire episode as one undifferentiated string.

### Give sparse semantic intent its own pool

**Experiment-supported, strongly.** The debug topology GA used pools `[1,2,3,5,6,10]` and achieved fitness `19.0547`, with exact `1.0`, held-out execution `1.0`, structural transfer `1.0`, and OOV honesty `0.6667`. Adding instruction-intent pool `12` produced fitness `20.6832`, exact `1.0`, held-out execution `1.0`, structural transfer `1.0`, and OOV honesty `1.0`.

The intent representation should encode combinations such as language, behavior, artifact type, and important constraints. Route on sufficiently specific combinations. Language-only or generic “project” evidence is too broad for safe composition in the current programming brain.

**Proven live for polyglot composition.** Restricting component retrieval to complete `language + behavior` evidence produced `6/6` correct multi-language projects, `12/12` executable components, and `2/2` honest OOV responses on the persisted brain. The read-only bracket remained at tick `198824`.

**Proven live for paraphrase routing.** A ranked complete manifest supported by one language plus one concrete behavior must outrank a raw character-similarity fallback. After reinforcing six canonical language/second-power episodes, raw similarity became biased toward the recent square examples. Four unrelated enterprise paraphrases still produced the correct ranked manifest candidate, but a response-selection rule ignored a lone non-exact manifest and returned the square fallback. Selecting that grounded manifest fixed native, platform, and cross-language paraphrases while retaining polyglot composition and OOV honesty. This is an inference-policy lesson; do not “fix” it by memorizing every paraphrase.

### Keep long supervised action outputs out of concept promotion

**Proven live.** For source-code/response action pools, use `max_concept_member_count = 1` unless a new experiment demonstrates a safe alternative. Allowing response suffixes to undergo ordinary concept neurogenesis caused about 6.46 million terminals before reaching the midpoint of a 582-example Python curriculum, without improving recall.

The action remains atom-grounded and binding-addressable; this setting prevents redundant suffix concepts from consuming the brain.

### Model failure, repair, and success as separate causal signals

**Proven live.** Do not train only `instruction -> corrected code`. When execution data is available, present an episode containing the failing source, console output, environment, failure classification, repair delta, corrected source, new console output, and resolution. This lets the brain learn the transition from failure to repair to verified success.

Relevant environment features include language and version, operating system, architecture, dependency/toolchain versions, resource constraints, and command. Include a feature only when it can affect the observed behavior; irrelevant metadata increases collisions.

### Represent underspecification and OOV explicitly

**Experiment-supported, strongly.** The instruction-intent pool improved OOV honesty from `2/3` to `3/3` in the GA curriculum while preserving other scores. An unknown or underconstrained request should activate explicit evidence for insufficient specification rather than weakly matching the nearest familiar language or task.

False confident generation is worse than an honest unsupported response. Every curriculum gate therefore needs held-out OOV and underspecification cases.

### Configure for integration, not lookup alone

**Proven live.** Exact recall is necessary but insufficient. Every candidate brain configuration must also pass:

- paraphrases of trained instructions;
- held-out inputs that preserve behavior but change values or structure;
- structural transfer;
- cross-language transfer where appropriate;
- multi-component composition;
- OOV honesty;
- execution of generated artifacts;
- retention of earlier curricula.

A candidate that raises exact recall while lowering any protected integration or retention gate is not an improvement.

Integration must include experiential learning, not only safe retrieval. A coding brain should receive a previously unseen rule or API contract in the task context, attempt a solution, execute it, and co-activate the instruction, source, environment, console output, failure outcome, repair delta/relation, corrected source, successful console output, and verified resolution. Re-query both the experienced task and a held-out task that shares the abstract repair but changes identifiers and constants. Exact post-correction recall alone is memorization; improvement on the held-out task is the transfer evidence. Always re-run protected retention after admitting the experience.

The stronger target is causal multi-domain integration. Train disciplines independently, request a never-observed artifact whose requirements intersect them, execute the result, and then remove each premise one at a time. Every removal must suppress the corresponding intent feature and make the deliberately complete fixture fail; otherwise the benchmark only demonstrates retrieval or accidental overlap. Include mutually exclusive premises to prove that the active context can override an otherwise familiar policy. Measure traversal quality with correctness, execution, causal ablations, latency, activated-feature count, and retention—not with fluent surface form. After a verified success or repair, admit the full environment episode transactionally so later structurally similar tasks can reuse the relationship without erasing protected knowledge.

Do not keep validating integration against the same surface domain used to teach the fragments. A serious holdout changes the class name, domain objects, state transitions, identifiers, constants, method contract, and execution harness while retaining only the abstract discipline intersection. Passing the original synthesis but failing this domain-shifted holdout means the brain learned a useful composition motif but has not yet generalized the motif into a reusable program-construction procedure.

Grounded source motifs may expose explicitly declared output parameters without becoming a tokenizer or unconstrained generator. The response bytes still come from learned atom-level evidence; a fragment declares a named placeholder and its restricted parameter kind, while the decoder fills it only from a validated value explicitly present in the current prompt. Missing, malformed, undeclared, or unresolved parameters invalidate the composition. This separates invariant learned structure from task-specific symbols and lets the same wired motif instantiate a genuinely new class identity deterministically.

Measure multi-hop synthesis separately from near-neighbor transfer. Construct tasks whose solution was never present in the curriculum and whose indispensable premises are distributed across several learned disciplines. Require the brain to bind those premises into an executable artifact, not merely name the contributing topics. A strong gate should include: a runnable or mechanically checkable result; held-out changes to identifiers, values, and surface wording; premise ablations where removing one required fact prevents or changes the answer; contradiction tests where an updated premise propagates to the result; and a traversal budget that records inference latency and the number of activated concepts/pools. Passing exact recall or one-hop repair does not establish this capability. The target is correct, efficient traversal of a long compatible chain followed by novel composition and environmental verification.

## Known-good programming baseline

The persisted programming brain currently derives from `brains/coding_debug.identity.toml`. Treat these as a starting point for similar supervised coding brains, not universal constants.

| Setting | Current value | Evidence and rationale |
|---|---:|---|
| Binding threshold | `3` | Proven live across the retained curriculum. |
| Binding moment window | `256` | Proven live for the richer coding episode; the smaller coding identity used `128`. |
| Minimum atom score | `0.94` | Current known-good recall baseline. |
| Instruction-intent recent window | `256` | Supports sparse intent evidence over the full episode. |
| Instruction-intent max concept members | `8` | Current known-good compact semantic concepts. |
| Typical sensory concept threshold | `6` | GA favored `6` over `3` and narrowly over `5` in one search; replicate before generalizing. |
| Typical sensory decay | `0.00002` | Current live retention baseline. |
| Typical sensory prune threshold | `0.001` | Current live retention baseline; must remain guarded by old-curriculum tests. |
| Corrected-source/action recent window | `128` | Current known-good supervised output window. |
| Corrected-source/action max concept members | `1` | Prevents destructive response-suffix neurogenesis growth. |

## Genetic-search findings

| Experiment | Result | Interpretation |
|---|---|---|
| Concept threshold `3 -> 5 -> 6` | Fitness `18.8259 -> 19.0239 -> 19.0547` | **Experiment-supported.** Threshold `6` was best in this narrow debug search, but the difference from `5` was small. |
| Add instruction-intent pool `12` | Fitness `19.0547 -> 20.6832`; OOV `0.6667 -> 1.0` | **Experiment-supported, strongly.** Retain the intent pool for coding and test analogous semantic pools in other domains. |
| Joint-score thresholds `0.35`, `0.55`, `0.75` | Equal fitness `19.0547` | **Provisional.** These thresholds were behaviorally equivalent on the small curriculum. They require harder collision cases before selection. |
| Joint-score threshold `0.90` | Fitness fell to `16.4547`; held-out execution `0.6` | **Rejected for this workload.** It discarded useful combined evidence. |
| Reduce max concept members `24 -> 16` | Fitness `18.8259 -> 18.8457` | **Inconclusive.** Small gain; insufficient evidence for a default change. |
| Remove evidence pool `3` | Fitness remained `18.8259` on small gate | **Inconclusive.** The gate was not broad enough to prove that the pool is redundant. Do not remove it from the live brain without a full retention ablation. |

GA fitness is meaningful only when all essential gates are components of the objective. Include accuracy, execution, transfer, OOV honesty, retention, and resource costs. A fast but forgetful or confidently wrong brain must not win.

## Findings that failed or were revised

- **Rejected:** treating a single raw-text pool as sufficient for paraphrase, OOV, and cross-language routing. Atom grounding remains essential, but derived feature pools provide the discriminating evidence.
- **Rejected:** ordinary concept promotion for long response/source action streams. It produced severe terminal growth without recall benefit.
- **Rejected:** routing generated components from language evidence alone. It can retrieve the wrong behavior in that language.
- **Rejected:** generic artifact labels such as “project” as sufficient composition evidence. Prefer complete `language + behavior` or more specific feature sets.
- **Rejected:** allowing a recent raw character-similarity match to outrank a complete artifact supported by ranked `language + behavior` evidence. Recency can otherwise pull unrelated requests toward the newest same-language lesson.
- **Rejected:** allowing fuzzy raw-character fallback when a programming-language feature fired but no compatible code action was grounded. During MathInstruct training, an unseen strict-TypeScript physics-project request returned an unrelated construction-cost solution with confidence `1.0`. A recognized language feature is therefore a domain boundary: exact raw episodes remain authoritative, but unresolved programming intent must compose from compatible feature bindings or return honest OOV.
- **Rejected for the current debug workload:** `min_joint_score = 0.90`; it reduced held-out execution and structural transfer.
- **Operational warning:** compiler cache, first-run setup, permissions, or toolchain failures can look like inference failures. Benchmarks must use controlled environments and separately report model selection, compilation, execution, and environmental failure.

## Curriculum and retention protocol

1. Establish atom-level grounding and basic concepts with the toddler curriculum.
2. Add K-12 knowledge to supply broad context and relationships.
3. Train programming progressively from syntax and small functions through debugging, systems, polyglot composition, and enterprise projects.
4. After every training block, run read-only gates for all earlier stages. Verify that the brain tick does not change during evaluation.
5. Measure exact recall, paraphrase routing, execution, integration/transfer, OOV honesty, and resource growth independently.
6. Reject or revise any training/configuration change that improves the new domain by materially damaging protected knowledge.
7. Checkpoint only after the new block and all protected gates pass. Record the identity, snapshot, curriculum position, and benchmark outputs together.

For very large corpora, a “block” must be bounded rather than an entire corpus phase. The current programming curriculum uses guarded 16,384-row blocks. Every 32-row transaction is WAL-flushed before acknowledgement, and the full snapshot is replaced once at the block boundary. Each block then runs distributed corpus recall (including the first and newest trained rows), complete foundational retention, and the strict enterprise battery before the next block is permitted. This bounds WAL replay to one block while avoiding three redundant rewrites of an increasingly large snapshot. A checkpoint without a retention gate proves durability, not non-interference.

Bulk posting must preserve inference windows without throwing away throughput. Measure every batch transaction, retain the block-wide maximum plus an exponential moving average, and adapt the next block from the maximum observed lock rather than the final batch alone. A last-only measurement can hide an earlier stall when a faster final batch overwrites it. The current live ceiling is eight seconds; batch size `32` remains justified while the measured maximum stays below that ceiling.

Preserve the last accepted snapshot until the candidate chunk passes. The current supervisor creates an NTFS hard link to `brain.bin` before training, so guarding a multi-gigabyte brain requires no duplicate copy. Candidate checkpoint replacement gives `brain.bin` a new file identity while the hard link retains the accepted bytes. Delete the guard only after corpus, foundation, and enterprise gates all pass; retain it and stop training on failure. A recovery still requires coordinated node shutdown and WAL reset—never replace the live server's snapshot underneath it.

Read-only validation must distinguish learned topology from tier residency. In this architecture inference can page evicted neurons and their terminals from SSD into RAM. Consequently `resident_terminals`, `evicted_neurons`, and the current `total_terminals` alias may change during an inference-only battery even though no wiring was created or removed. Protect the tick and stable topology fields (`pool_count`, `total_neurons`, `total_concepts`, `total_binding`, and `binding_pool_id`); record residency changes as operational telemetry rather than treating them as learning. Snapshot/WAL identity remains the strongest durability-level evidence when a byte-level non-mutation claim is required.

“Pre-training” in Wizard Vision should mean accelerated construction of complete atom-grounded episodes and their derived simultaneous features. It must not mean replacing character/byte learning with an external token vocabulary.

## Procedure for configuring a new domain brain

1. Define the raw sensory and action streams.
2. Identify independent, causally useful feature families and assign each to a pool.
3. Define the integration evidence required for a confident action; avoid broad single-feature routing.
4. Add explicit unknown, ambiguity, failure, and success signals.
5. Select a small protected curriculum containing exact, paraphrase, transfer, OOV, and outcome cases.
6. Establish conservative concept, decay, and pruning settings from the closest proven identity.
7. Run ablations before adding complexity: each derived pool must demonstrate value on held-out behavior.
8. Search thresholds/topology genetically only with a multi-objective fitness that protects retention and resource usage.
9. Expand the curriculum in stages, checkpointing only passing configurations.
10. Record each conclusion below with its artifacts and evidence label.

For a new programming brain, use `scripts/train_programming_brain.py` rather than reconstructing the successful training order manually. The exact stage order, repetitions, transaction guards, resume behavior, and gate artifacts are described in `docs/PROGRAMMING_BRAIN_REPRODUCTION.md`.

## Evidence ledger

| Date | Finding | Evidence | Artifacts |
|---|---|---|---|
| 2026-07-13 | Concept threshold `6` narrowly led the tested thresholds. | Experiment-supported | `runtime/debug_topology_ga-20260713/` |
| 2026-07-13 | Instruction-intent pool completed OOV honesty without reducing exact, execution, or transfer. | Experiment-supported, strongly | `runtime/debug_intent_pool-20260713/` |
| 2026-07-13 | Joint threshold `0.90` was too restrictive for the debug curriculum. | Rejected for this workload | `runtime/debug_joint_sweep-20260713/` |
| 2026-07-13 | Long action-stream concept promotion caused extreme terminal growth without recall gain. | Proven live observation | `brains/coding_small.identity.toml` |
| 2026-07-15 | The persisted programming brain passed protected toddler, K-12, OOV, Python execution, debug exact/held-out/structural/OOV gates without changing tick. | Proven live | `runtime/brains/programming-integrated-20260713/` and benchmark reports in `runtime/` |
| 2026-07-15 | Complete `language + behavior` evidence safely composed polyglot manifests: projects `6/6`, components `12/12`, OOV `2/2`, tick unchanged. | Proven live | `runtime/programming-polyglot-composition-after-routing.json` |
| 2026-07-15 | Six canonical multilingual episodes generalized to all six unseen paraphrases after 36 presentations, adding only six bindings. | Proven live | `runtime/benchmarks/multilanguage-after-intent-reinforcement.json` |
| 2026-07-15 | A single ranked complete manifest must outrank raw similarity for one-language requests. An initially green semantic-stress diagnostic was discovered to return success despite missing recalls; after making that gate strict and sweeping equivalent coding verbs and enterprise behavior phrases, all `11/11` suites genuinely passed with no tick mutation. | Proven live | `runtime/benchmarks/enterprise-retention-strict-semantic.json` |
| 2026-07-15 | Post-correction foundation retention remained toddler `32/32`, K-12 `16/16`, OOV `3/3`, Python execution `10/10`, and all debug transfer gates perfect. | Proven live | `runtime/benchmarks/integrated-retention-after-ranked-manifest.json` |
| 2026-07-15 | Semantic feature coverage must be tested across equivalent action verbs and relational phrases, not only canonical keywords. Adding “develop” and behavior-equivalent authorization, replay, rollback, correlation, and redaction evidence raised strict semantic stress from trained `2/4`, held-out `1/4` to `4/4` and `4/4` without training held-out answers. | Proven live | `runtime/benchmarks/semantic-stress-strict-after-feature-sweep.json` |
| 2026-07-15 | Multi-million-row phases require bounded train/checkpoint/gate cycles. The supervisor now limits direct pretraining to 4,096 durable rows, then blocks continuation on distributed corpus recall plus complete foundational and strict enterprise retention. | Operationally enforced | `scripts/programming_curriculum_supervisor.py` |
| 2026-07-15 | The first bounded MathInstruct cycle reached durable row `188720`, recalled `32/32` distributed corpus samples, retained every foundation/debug/Python gate, and passed strict enterprise `11/11` with tick `207532 -> 207532`. | Proven live | `runtime/brains/programming-integrated-20260713/mathinstruct-domain-safe.row-188720.retention-gate.json` |
| 2026-07-15 | A second guarded MathInstruct cycle reached durable row `192816`, recalled `32/32`, retained every foundation gate, and passed strict enterprise `11/11` with tick `211628 -> 211628`. Two consecutive clean calibration cycles justified widening retention blocks to `16384` rows. | Proven live | `runtime/brains/programming-integrated-20260713/mathinstruct-domain-safe.row-192816.retention-gate.json` |
| 2026-07-15 | Retain a hard-linked last-known-good snapshot across each candidate chunk and delete it only after gate acceptance. | Operationally enforced | `runtime/brains/programming-integrated-20260713/brain/brain.last-good.json` |
| 2026-07-15 | A third guarded MathInstruct cycle reached durable row `196912`, recalled `32/32`, retained toddler `32/32`, K-12 `16/16`, OOV `3/3`, Python execution `10/10`, all debug transfer gates, and strict enterprise `11/11` with tick `215724 -> 215724`. | Proven live | `runtime/brains/programming-integrated-20260713/mathinstruct-domain-safe.row-196912.retention-gate.json` |
| 2026-07-15 | Batch size `32` later reached a `10.3935`-second live posting lock, exceeding the configured `8`-second ceiling even though the final batch took only `5.8143` seconds. Tune from the maximum, not the last sample; the responsive controller scales the next block to `24` (`floor(32*8/10.3935)`). | Proven live calibration | `runtime/brains/programming-integrated-20260713/mathinstruct-domain-safe.progress.json` |
| 2026-07-15 | Inference-only enterprise evaluation paged cold neurons from SSD, changing resident terminal and eviction counters while tick and stable topology counts remained fixed. Residency is not a valid structural non-mutation invariant. | Architecture-confirmed live observation | `crates/brain/src/pool.rs`, `crates/brain/src/brain.rs`, and row `196912` runtime stats |
| 2026-07-15 | At row `201008`, 4,096 MathInstruct episodes grew the full snapshot by about `121.6 MB`. With WAL acknowledgement already making every batch durable, snapshotting four times per guarded block creates increasing full-snapshot write amplification without improving retention admission. Align full snapshots with the `16,384`-row guard boundary. | Proven live operational measurement | `runtime/brains/programming-integrated-20260713/brain/brain.bin` and `mathinstruct-domain-safe.progress.json` |
| 2026-07-15 | An unseen strict-TypeScript Multiscale Robot World prompt incorrectly selected recent MathInstruct prose despite detecting `LANGUAGE:TYPESCRIPT` and finding zero compatible feature candidates. Programming-language intent now inhibits ungrounded raw fallback, and capstone safety is a required enterprise suite that accepts only a structurally grounded manifest or honest OOV. | Failure reproduced; correction unit-tested, pending live deployment gate | `runtime/benchmarks/capstone-readiness-before-domain-guard.json`, `scripts/programming_capstone_readiness.py` |
| 2026-07-15 | Experiential coding must be measured as `baseline attempt -> verified failure/success episode -> experienced retry -> held-out structural transfer`, with the held-out function, constants, and wording changed. The live brain initially gave honest OOV for the unseen Nebula rule, learned the verified `- -> +` repair in six atom-grounded episodes, executed the corrected Nebula function, and transferred the relation to an unseen Quasar function with different identifiers, factor, and offset. Foundation and enterprise retention both passed, tick advanced exactly `6`, the admitted state was checkpointed, and the rollback link was released only afterward. | Proven live | `runtime/benchmarks/experiential-generalization-live.json` |
| 2026-07-15 | Cross-disciplinary integration requires a dedicated multi-hop synthesis gate: the final artifact must be absent from training, necessary premises must be distributed across learned domains, and execution, held-out variants, premise ablations, contradictions, and traversal cost must all be measured. | Acceptance protocol defined; benchmark implementation pending | This guide, “Configure for integration, not lookup alone” |
| 2026-07-15 | A language-intent boundary must inspect candidate compatibility rather than suppress all fuzzy/ranked sensory recall. Allow a single-language ranked source only when language+behavior features ground it and its bytes are shaped like that language; continue to reject cross-domain prose and partial polyglot answers. This restored Python paraphrase execution from `4/5` to `5/5` while the unseen TypeScript capstone remained honest OOV. | Proven live | `runtime/benchmarks/capstone-readiness-ranked-source-guard.json` and row `213296` retention artifacts |
| 2026-07-15 | Retention orchestration must validate the authoritative full report, not a compact stdout summary, and the integrated benchmark itself must exit nonzero when any protected foundation, syntax, execution, or transfer count is incomplete. Row `213296` then passed corpus recall `32/32`, integrated retention (including Python `10/10`), and enterprise `12/12` with stable structure and tick. | Proven live and operationally enforced | `runtime/brains/programming-integrated-20260713/mathinstruct-domain-safe.row-213296.retention-gate.json` |
| 2026-07-15 | A 12-discipline synthesis baseline activated 15 intent labels and retrieved 13 ranked candidates but did not produce the required new class; its contradictory no-retry premise also failed to propagate. This separates broad candidate activation from successful causal composition and establishes the next routing/topology bottleneck. | Proven live baseline; capability not yet admitted | `runtime/benchmarks/multidomain-synthesis-baseline.json` |
| 2026-07-15 | Concurrent multi-hop inference exposed a `15.4753`-second training transaction at batch `24`. Preserve maximum timing telemetry across worker restarts, adapt batch size downward within a running worker, and pass the live-lock ceiling into every corpus invocation. The resumed worker selected batch `12` and returned to roughly `1–3` second transactions. | Proven live and operationally enforced | `mathinstruct-domain-safe.progress.json` and `tools/training_standard/drive_corpora_brain.py` |
| 2026-07-15 | A retention block target must be immutable across process failure. Derive it from the hard-linked guard's starting row rather than `current row + gate size`; otherwise each restart silently extends the ungated interval. The active block remains fixed at row `229680` after restarting from row `217796`. | Proven live and operationally enforced | `brain.last-good.json` and `curriculum-supervisor.status.json` |
| 2026-07-15 | Multi-domain synthesis requires structured fragments to survive language+behavior subset recovery; complete-manifest-only filtering left learned fragments unreachable on large prompts. Admit only validated relative-path fragments with nonempty roles, sources, and dependencies. A complete manifest must still outrank an isolated exact fragment when their sparse features collide. | Proven live core routing fix | `crates/node/src/brain_api.rs` |
| 2026-07-15 | The programming brain composed a never-trained `AdaptiveCoordinator` from twelve independently learned disciplines: validation, authorization, migration, recursive redaction, correlated logging, circuit breaking, async retry, idempotency, optimistic concurrency, deduplication, atomic transaction, and transactional outbox. The artifact executed; removing each premise removed its feature and broke execution; a contradictory no-retry premise selected a mutually exclusive single-attempt policy and executed correctly. After resolving partial-fragment shadowing, semantic stress returned to exact/held-out `4/4`, enterprise passed `12/12`, structure and tick stayed stable, and the 102-episode candidate was checkpointed at tick `248636`. | Proven live and admitted | `runtime/benchmarks/multidomain-synthesis-after-precedence.json`, `semantic-stress-after-fragment-precedence.json`, and `multidomain-enterprise-after-precedence.json` |
| 2026-07-15 | The reproducible trainer now runs verified failure-to-repair transfer and causal 12-discipline synthesis after the guarded corpus curriculum, each with transactional rollback, foundation retention, enterprise retention, and checkpoint admission. Corpus transaction-size calibration also carries across phase boundaries so a new subject cannot repeat a lock duration already proven unsafe in an earlier subject. | Implemented and contract-tested | `scripts/train_programming_brain.py`, `scripts/programming_curriculum_supervisor.py`, and `tests/test_programming_runtime_contract.py` |
| 2026-07-15 | A domain-shifted twelve-discipline holdout now requests a never-trained `ResilientFulfillmentService`, replacing the account-transfer class, inventory/state contract, role model, identifiers, constants, and execution harness. Its optional causal sweep removes each abstract discipline independently. This is the next admission gate for distinguishing reusable integration from replay of the admitted coordinator motif. | Implemented; live result pending completion of the guarded corpus block | `scripts/programming_multidomain_holdout.py` |
| 2026-07-15 | The first non-mutating domain-shifted baseline failed in `4.5755s`: it emitted two isolated authorization/logging functions rather than the requested class. Diagnostics showed that valid domain paraphrases activated only part of the twelve-discipline set. The intent encoder now recognizes the missing order-validation, resettable-circuit, order-key-replay, expected-inventory-version, duplicate-SKU-work, and inventory-rollback relations as their existing abstract features. All 30 intent tests pass; the updated binary and full holdout remain to be evaluated after the guarded corpus block. | Baseline proven; feature-path correction implemented but not yet deployed | `runtime/benchmarks/multidomain-holdout-during-mathinstruct.json`, `crates/brain/src/pool.rs`, and `crates/brain/tests/instruction_intent_encoding.rs` |
| 2026-07-15 | After deploying the intent-generalization build, the domain-shifted holdout activated all twelve required abstract disciplines (16 total labels including language/container signals) and retrieved 3,476 characters across four relevant learned components, up from 982 characters and two components. It still failed because `ResilientFulfillmentService` was never constructed; execution now reports that exact missing class rather than a harness-import artifact. This localizes the next bottleneck after successful intent traversal: grounded fragments are concatenated as domain-specific artifacts, but there is no learned parameterized structural motif that binds a novel class/state/method contract across them. | Proven live; composition generalization not yet admitted | `runtime/benchmarks/multidomain-holdout-after-intent-generalization-v2.json` |
| 2026-07-15 | The completed MathInstruct phase passed its authoritative gate on the updated node at tick `264279`: corpus recall accepted `64/64` samples (`59` exact), toddler `32/32`, K–12 `16/16`, Python trained/paraphrase execution `5/5` each, TypeScript trained/paraphrase execution `3/3` each, OOV honesty complete, and enterprise `12/12` with stable tick and topology. MetaMath began from row zero under a new last-good guard using batch `12`, proving that live-lock calibration carried across the phase boundary. | Proven live and admitted | `runtime/brains/programming-integrated-20260713/mathinstruct-domain-safe.completion-gate.json` and `metamathqa-domain-safe.progress.json` |
| 2026-07-15 | The grounded-fragment protocol now supports explicitly declared `python_class_named` and `python_method_named` parameters. A learned source motif containing `{{CLASS_NAME}}` or `{{METHOD_NAME}}` can bind a valid identifier from “class/method named/called …” in the live prompt; missing cues, invalid identifiers, unknown parameter kinds, and unresolved placeholders reject the entire composition. Existing fragments remain byte-identical and all API composition/routing plus 30 runtime-contract tests pass. These are the first structural parameterization primitives required by the domain-shifted holdout; state-contract motifs remain to be learned and generalized. | Implemented and contract-tested; deployment pending next guarded boundary | `crates/node/src/brain_api.rs` |
| 2026-07-15 | The next guarded experience experiment is encoded as thirteen independently trainable fragments spanning the twelve fulfillment disciplines and structural commit closure. No training response contains a complete service. The executable fixture proves that the same atom-grounded motif can instantiate both `ResilientFulfillmentService.fulfill` and the never-trained `DurableWarehouseEngine.allocate_order`; training is transactionally protected and cannot run beside a corpus worker, and admission requires both executions plus foundation and enterprise retention. This tests symbol binding and structural composition without confusing it with whole-artifact recall. | Fixture and transaction gate contract-tested; live training pending MetaMath boundary | `scripts/programming_parameterized_fulfillment.py` and `tests/test_programming_runtime_contract.py` |
| 2026-07-15 | The pre-training two-symbol baseline failed causally: both requested interfaces activated all twelve disciplines and returned the same 3,476-byte set of separate learned components, but neither requested class existed. The read-only run overlapped MetaMath, so its tick delta is explicitly attributed to recorded concurrent training PIDs rather than to inference. This provides the before-state for the guarded parameterized-motif admission. | Proven live baseline | `runtime/benchmarks/parameterized-fulfillment-baseline.json` |

## Maintenance rule

Update this guide whenever a configuration experiment finishes. Add the artifact path, distinguish correlation from causal ablation, and state the benchmark scope. If later evidence overturns a finding, move it to “Findings that failed or were revised” and replace the active recommendation. Never silently retain obsolete configuration folklore.
