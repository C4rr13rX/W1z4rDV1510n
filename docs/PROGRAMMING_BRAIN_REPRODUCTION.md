# Reproducing the programming brain

`scripts/train_programming_brain.py` is the canonical entry point for training a fresh Wizard Vision coding brain with the process proven by the integrated programming run. It preserves atom-level grounding; it does not introduce a tokenizer or bypass the configured neural pools.

## One-command run

Build the current release brain server, then choose a new runtime directory and an unused port:

```powershell
cargo build --release -p w1z4rdv1510n-node --bin w1z4rd_brain_server
python scripts/train_programming_brain.py `
  --runtime runtime/brains/programming-reproduction-001 `
  --port 18601
```

The default corpus root is `D:\w1z4rdv1510n-data\training`. Override it with `--corpus-root` when the generated corpora live elsewhere.

The canonical production defaults use a maximum neural lock scope of `32`, a
two-second supervisor poll, ten process-restart attempts, a `131072`-row
guarded block, and a `16384`-row live canary interval. The lock scope is a
ceiling rather than an unconditional transaction size: persisted slow-batch
evidence selects a smaller starting scope, the live controller increases it
only after repeated sub-threshold measurements, and any breach reduces it
immediately.

The supervisor also keeps a `6 GiB` host-available-memory floor by default.
When active training crosses that floor, it stops the corpus worker only after
three consecutive observations at an exact WAL-durable row, performs the
normal neuron-wise sleep and checkpoint without admitting the partial block,
waits for host memory to recover above the floor, and resumes from that same
row under the existing rollback guard. This treats memory pressure as a
lifecycle yield, not as a learned failure or a reason to quarantine correct
corpus data. Set
`--min-free-memory-gb 0` only when an external resource supervisor provides an
equivalent safeguard.

Use `--resume` with the same runtime after an interruption. The trainer records only accepted stages. Each seed stage is protected using the authoritative brain representation: immutable atomically replaced `brain.bin` checkpoints may use an NTFS hard link, while mutable `brain.wbrain` containers always use an independent copy published by atomic rename. On an owned-node resume, an interrupted transaction is either committed when its durable state record exists or restored before the stage is retried. This avoids silently training a partially completed stage twice and prevents a `.wbrain` candidate from mutating its own rollback guard.

Use `--dry-run` to inspect the complete command plan without creating a runtime or contacting a node. Use `--seed-only` to stop after the curated enterprise curriculum and its strict gate. `--external-node` is available for an intentionally pre-launched node, but the normal owned-node mode provides safer restart and rollback behavior.

In owned-node mode, final cleanup resolves the process currently owning the
runtime endpoint. This matters because automatic quarantine recovery may
replace the PID launched at startup. The trainer verifies that the listener is
a Wizard brain server before stopping it, so successful completion and
exceptions do not leak a replacement node or terminate an unrelated service.
Cleanup is armed only after this trainer launches its initial node; discovering
a pre-existing listener therefore reports the port conflict without attempting
ownership cleanup.

For a controlled deployment or experience-admission window, stop the supervisor near a block target but allow its corpus worker to reach the exact durable boundary. Then run `programming_curriculum_supervisor.py --gate-only-phase <phase>` with the normal endpoint/runtime/gate options. It executes the same authoritative midphase or completion gate, releases the matching last-good guard only on success, and exits without starting another worker. Resume the ordinary supervisor only after the protected maintenance operation is complete.

When a continuous canary rejects a live candidate and a causal correction has
been deployed without rolling that candidate back, use
`--retest-canary-quarantine`. The supervisor requires the exact candidate RAM
and durable offsets, the phase-owned rollback guard, and every disjoint
candidate interval in the unresolved ledger. Its recall gate temporarily
includes only those candidate intervals while continuing to exclude unrelated
deferred ranges. It then runs the normal settled midphase or completion gate.
Failure leaves the candidate, ledger, marker, and guard unresolved. Success
resolves exactly the marker-owned intervals, admits the checkpointed candidate,
removes the quarantine marker, and releases the matching guard:

```powershell
python scripts/programming_curriculum_supervisor.py `
  --endpoint http://127.0.0.1:18095 `
  --runtime runtime/brains/programming-integrated-20260713 `
  --corpus-root D:\w1z4rdv1510n-data\training `
  --retest-canary-quarantine
```

Do not use this mode after restoring the guard or against a different live
candidate. Do not resolve the candidate intervals manually before the retest;
their unresolved identity is part of the admission proof.

The canonical trainer runs the supervisor twice as two restart-safe stages.
The first command uses `--auto-quarantine-recovery` to finish the forward sweep
while excluding exact failed intervals. Only after every forward phase has a
durable completion gate does the second command use `--replay-deferred`. It
orders the still-unresolved intervals by phase and row and replays one exact
half-open span into the final accumulated brain. Each replay has its own
pre-mutation `.wbrain` guard, progress ledger, exact-interval recall, and normal
whole-brain completion gate with only that interval temporarily included. A
failure restores the final accepted brain without rewinding any completed
forward offset and leaves the interval unresolved for the test–fix retry. A
pass records admission, resolves only that interval, and prunes its unreferenced
causal base.

`deferred-replay-active.json` makes this final queue restart-safe. A marker in
`training` state is rolled back before retry; a marker advanced to `admitted`
after the comprehensive gate is committed without rerunning its examples.
The ledger resolution and guard release occur only after that durable admitted
marker, preventing either duplicated replay training or a resolved interval
whose learned state was rolled back.

## Encoded curriculum

The seed curriculum is ordered as follows:

1. toddler, K-12, Python generation, and failure/repair/success debugging;
2. executable multilingual generation;
3. Python enterprise behaviors;
4. executable multi-file Python projects;
5. platform engineering;
6. native-language enterprise behaviors;
7. TypeScript enterprise behaviors;
8. cross-language transfer;
9. semantic-routing reinforcement.

After every seed expansion, the trainer runs the complete foundational/Python/debug retention gate and checkpoints only a passing candidate. After all seed stages it runs strict enterprise retention, including execution, zero-shot composition, semantic stress, OOV honesty, tick immutability, and stable-topology immutability.

The corpus supervisor then trains:

1. TheAlgorithms/Python canonical algorithms, four dense repetitions;
2. GSM8K;
3. MathInstruct;
4. MetaMathQA;
5. CodeSearchNet Python;
6. five-way CodeSearchNet Python paraphrases;
7. scientific Jupyter source;
8. four-way scientific Jupyter paraphrases;
9. partial-context scientific Jupyter examples.

Corpus processing is resumable and WAL-durable. The default production schedule uses a 131,072-row guarded admission block with non-halting 16,384-row canaries. Every small batch is flushed to the WAL before acknowledgement. Interior canaries detect drift while training continues and quarantine only the suspect interval; the canary coincident with the comprehensive block endpoint is deliberately skipped because the worker may still be finishing its final checkpoint. At the exact stopped-worker boundary, the supervisor drains deferred maintenance, serializes all neuron bodies, checkpoints that settled state, verifies zero resident terminals, and only then runs distributed corpus recall, foundational retention, executable transfer, strict enterprise behavior, OOV honesty, and read-only tick/topology invariants. A passing gate releases the independent last-known-good guard. Before the next guard is assigned a corpus row, the live node completes another checkpoint barrier and the guard records the post-barrier topology. Recovery resolves the process that actually owns the configured endpoint, confirms the replacement PID owns it, and requires its reopened topology to match that proof before the ledger may be rewound to the guard row. Historical over-ceiling evidence initializes a conservative server-side neural lock scope. Within a live run, eight consecutive observations below one quarter of the responsiveness ceiling double that scope gradually up to the configured maximum; any ceiling breach immediately scales it downward and records the exact slow rows and stage profile. This sheds obsolete calibration after a causal optimization without weakening the measured responsiveness invariant. During inference, complete raw atom-grounded exact episodes remain authoritative. Exact, composed, or ranked artifacts reconstructed from derived intent pools must pass the same language and behavioral compatibility contract before they can answer; “exact” within a lossy semantic projection is not equivalent to exact sensory experience.

All admission observations use one infrastructure policy. Transient timeout,
connection, empty-output, or truncated-JSON failures are retried and recorded
as `passed: null`; they are never converted into semantic corpus intervals.
If retries are exhausted, the supervisor preserves the guarded candidate and
exits for a restart. Deferred replay restores its pre-replay guard before
pausing. A completed evaluator reporting a behavioral mismatch is the only
path that may reject or quarantine learned rows.

Each deferred interval retains an exact causal base for later bisection. Because the last-good `.wbrain` guard is already an immutable inode independent of the mutable live container, the deferred base uses a hard link to that guard when the filesystem permits it. Rollback always copies the guard into a new live inode before removing the guard name, so the deferred link remains immutable. If rollback regenerates another guard for the same phase, row, storage type, and checkpoint-proven topology, the supervisor links new intervals to the existing causal-base inode instead of retaining another tens-of-gigabytes copy of the same accepted state. This makes quarantine publication effectively constant-space per distinct accepted state; cross-volume or link-restricted filesystems fall back to an independent copy.

Resolved interval bases are not permanent archives. After the append-only ledger records resolution, the supervisor folds the complete ledger and removes only known causal-base directories that no unresolved interval references; unknown experiment directories remain untouched. Before creating an independent `.wbrain` guard, the supervisor also requires free space for both the guard and a later rollback replacement plus reserve. A low-disk host therefore stops before copying rather than accepting a guard it could not safely restore.

After every forward phase is complete, replay the folded deferred ledger with the same supervisor:

```powershell
python scripts/programming_curriculum_supervisor.py `
  --runtime runtime/brains/programming-integrated-20260713 `
  --endpoint http://127.0.0.1:18095 `
  --node-bin target/release/w1z4rd_brain_server.exe `
  --replay-deferred `
  --batch-size 256 `
  --lock-chunk-size 32
```

Replay is deliberately an end-of-corpus transaction. Each interval trains against the final accepted brain under an independent guard, recalls samples from its exact row window, and then passes the complete corpus/foundation/enterprise admission gate with that interval temporarily included. A passing interval is resolved and its unreferenced causal base pruned. A failing interval restores the final brain, remains unresolved with its diagnostic artifacts, and stops the replay loop for a test/family-fix/retry cycle. Use `--replay-interval-id PHASE:START:END` to retry one corrected interval without admitting any other pending range.

Only one curriculum supervisor may own a runtime. Startup scans for an existing Python supervisor with the same resolved runtime and then claims `curriculum-supervisor.pid` using exclusive creation; stale PID files are recovered, while a live owner is rejected. This claim covers guard creation as well as training so two launches cannot race on the same temporary snapshot or progress ledger.

After the corpus curriculum, the trainer admits three experience gates. The
first exercises an unseen environment rule through baseline failure, verified
repair, successful execution, and held-out structural transfer. The second
composes a never-trained class from twelve independently trained disciplines,
executes it, causally ablates every premise, and tests a contradictory no-retry
policy. The third reproduces the production-proven generic state-contract
motif as exactly one presentation of 26 domain-rich/abstract atom-grounded
episodes; no lesson contains a complete service. It must instantiate two
unseen inventory interfaces and transfer to withheld scheduler and quota state
models. All three mutations are transactionally guarded and must preserve
foundation and enterprise retention before their checkpoints are accepted.

The trainer then runs resumable, read-only causal qualification gates for the
multidomain holdout, domain-transfer holdout, third state contract,
cross-project composition, polyglot composition, and composition matrix.
Multidomain and domain-transfer qualification include premise ablations.
The mobile-runtime gate sleeps the brain before each cold request, requires
correct deterministic cold/warm output in under `1.0s`/`0.5s`, limits peak
resident terminals to ten percent of learned terminals, and requires zero
residency between trials. This desktop measurement is a necessary deployment
proxy, not a substitute for the later benchmark on representative phone
hardware.
Each passing gate is recorded independently in
`completed_qualification_gates`, so interruption does not repeat earlier
checks or mistake a partially evaluated suite for final qualification.
The entire read-only sequence is bracketed by tick, pool count, neuron,
concept, binding, and binding-pool identity. Residency counters may change as
neurons page in and out, but any learned-topology change fails final
qualification. The authoritative bracket is written to
`benchmarks/final-qualification.json`.

Finally, the owned live brain completes neuron-wise sleep, checkpoints the
qualified `.wbrain`, and must report zero resident terminals without changing
tick or stable topology. The trainer records the before/sleep/after proof in
`benchmarks/production-finalization.json` and only then marks
`production_brain_finalized` in its durable state.

## Authoritative artifacts

The runtime contains:

- `programming-training.state.json`: accepted seed and corpus milestones;
- `brain.identity.toml` and `brain.deployment.toml`: frozen configuration copies;
- `seed/*.json`: seed-stage and gate reports;
- `logs/*.log`: command-level execution logs;
- `*.progress.json`: RAM and durable corpus offsets plus batch telemetry;
- `*.slow-batches.jsonl`: append-only row ranges and payload/lock evidence for every transaction exceeding the responsiveness ceiling;
- `*.retention-gate.json` and `*.completion-gate.json`: admission evidence;
- `benchmarks/experiential-generalization.json`, `benchmarks/multidomain-synthesis.json`, and `benchmarks/parameterized-fulfillment.json`: post-corpus experience, causal integration, and generalized state-contract admission evidence;
- `benchmarks/final-qualification.json`: the completed holdout list and read-only stable-topology bracket;
- `benchmarks/production-finalization.json`: final neuron-wise sleep, checkpoint, zero-residency, and identity proof;
- `brain/brain.wbrain`: authoritative neuron-addressed container for current runtimes;
- `brain/brain.wal`: durable mutations not yet incorporated into the accepted container state;
- `brain/brain.bin`: optional legacy checkpoint input, not authoritative after `.wbrain` migration;
- `brain/*.last-good.*`: unresolved rollback state, present only while a candidate is under review.

Do not declare a reproduced brain equivalent merely because the process exits successfully. Verify the final state file, every completion gate, the strict enterprise report, execution results, OOV honesty, and the final brain identity together.
