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
then checks logical and physical release separately. If every terminal is
serialized but the allocator still retains enough released pages to remain
below the floor, it recycles the node, reopens the checkpoint, and requires an
identical tick and stable topology before continuing. It waits only when
host-wide memory remains low after that verified recycle, then resumes from
the same row under the existing rollback guard. This treats memory pressure
as a lifecycle yield, not as a learned failure or a reason to quarantine
correct corpus data. Set
`--min-free-memory-gb 0` only when an external resource supervisor provides an
equivalent safeguard.

Use `--resume` with the same runtime after an interruption. The trainer records only accepted stages. Each seed stage is protected using the authoritative brain representation: immutable atomically replaced `brain.bin` checkpoints may use an NTFS hard link, while mutable `brain.wbrain` containers always use an independent copy published by atomic rename. Independent snapshot publication preserves the source uid/gid on POSIX hosts, including when an administrative recovery process restores a brain consumed by an unprivileged node service. On an owned-node resume, an interrupted transaction is either committed when its durable state record exists or restored before the stage is retried. This avoids silently training a partially completed stage twice and prevents a `.wbrain` candidate from mutating its own rollback guard.

Use `--dry-run` to inspect the complete command plan without creating a runtime or contacting a node. Use `--seed-only` to stop after the curated enterprise curriculum and its strict gate. `--external-node` is available for an intentionally pre-launched node, but the normal owned-node mode provides safer restart and rollback behavior.

In owned-node mode, final cleanup resolves the process currently owning the
runtime endpoint. This matters because automatic quarantine recovery may
replace the PID launched at startup. The trainer verifies that the listener is
a Wizard brain server before stopping it, so successful completion and
exceptions do not leak a replacement node or terminate an unrelated service.
Cleanup is armed only after this trainer launches its initial node; discovering
a pre-existing listener therefore reports the port conflict without attempting
ownership cleanup.

A production `.wbrain` can take several minutes to reopen before its HTTP
listener appears. Recovery therefore also identifies a pre-bind node by the
`W1Z4RD_NODE_BRAIN_DIR` environment that names the runtime's brain directory.
An absent listener and stale `node.pid` do not prove that the runtime is
unowned; launching a second process during that window can load the same large
container twice and exhaust host memory.

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
The first command uses `--auto-quarantine-recovery --forward-harvest` to finish
the forward sweep while excluding exact failed intervals. A phase advances only
after either a durable completion gate passes or every row after its restored
accepted guard is covered by durable deferred intervals. Only after every
forward phase is completely accounted does the second command use
`--replay-deferred`. It
orders the still-unresolved intervals by phase and row and replays one exact
half-open span into the final accumulated brain. Each replay has its own
pre-mutation `.wbrain` guard, progress ledger, exact-interval recall, and normal
whole-brain completion gate with only that interval temporarily included. A
failure restores the final accepted brain without rewinding any completed
forward offset and leaves the interval unresolved for the test–fix retry. A
pass records admission, resolves only that interval, and prunes its unreferenced
causal base.

`deferred_intervals_pending` is therefore a successful terminal state for the
forward-harvest process, but not for an ordinary supervisor invocation. The
canonical trainer receives a zero exit status at that handoff and immediately
starts deferred replay; a non-harvest invocation still exits unsuccessfully so
no caller can mistake unresolved intervals for a completed curriculum.

`deferred-replay-active.json` makes this final queue restart-safe. A marker in
`training` state is rolled back before retry; a marker advanced to `admitted`
after the comprehensive gate is committed without rerunning its examples.
The ledger resolution and guard release occur only after that durable admitted
marker, preventing either duplicated replay training or a resolved interval
whose learned state was rolled back.

Automatic same-phase recovery retains the verified immutable rollback guard
after restoring it. The next worker reuses that guard instead of copying the
same multi-gigabyte accepted state a second time. Forward harvest releases the
guard when it accounts the failed phase and advances to a different corpus.

An attached worker is subject to the same three-sample host-memory floor as a
worker launched by the supervisor. On a breach, the supervisor stops the
worker, requires equal RAM and WAL-durable offsets, sleeps and checkpoints the
neuron-scoped state, records the resource settlement, and resumes through the
ordinary guarded loop. Attachment is not an exemption from host protection.

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

Broad language-and-behavior compatibility is not a substitute for the complete
prompt contract. Every derived response exit--ranked feature, character motif,
fragment or manifest composition, and autonomous integration--must apply the
same prompt-specific operation, input, output, edge-condition, and artifact
checks. Only an exact raw sensory episode is authoritative without that derived
route validation. A validator attached to an early candidate-discovery branch
is insufficient when a later fallback can select the same bytes through a
different evidence pool.

Hebbian reinforcement must refresh routing addressability as well as the
binding neuron's `use_count`. Immutable posting generations otherwise leave a
repeatedly successful old binding trapped behind newer equal-feature bindings:
the knowledge still exists, but a bounded readout cannot reach it. Re-advertise
the existing binding identity at early power-of-two recurrence milestones and
at least once every eight recurrences thereafter. This adds no neuron and no
synthetic action while placing a finite upper bound on routing staleness.
Protected route maintenance must therefore present every canonical route at
least eight times, settle the overlay into immutable storage, and pass the full
retention battery before admission. A shorter fixed pass can miss a
logarithmic milestone and is not durable-reachability evidence.

After the conditionally launched or adopted node becomes healthy, the service
wrapper resolves both its listening-socket PID and its
`W1Z4RD_NODE_BRAIN_DIR` runtime identity. They must identify the same process
before the wrapper atomically replaces `node.pid`. This prevents a stale PID
file from contradicting the pre-bind process identity used to block duplicate
large-container loaders.

The curriculum unit itself owns conditional node startup. It adopts a live or
pre-bind runtime owner across wrapper restarts and calls the same guarded node
launcher only when neither a runtime process nor endpoint owner exists. Do not
add an unconditional `Wants=wizard-brain-initial.service` dependency: a
recovered node deliberately outlives the wrapper, so starting that unit again
would create a competing cold loader before address binding rejects it.

Run a maintenance recycle as the same account configured by the curriculum
unit. Stable neural topology alone is not sufficient for adoption: a node
started as `root` can reopen the exact accepted checkpoint and own the correct
socket while remaining unreadable to an `ec2-user` `/proc` identity scan. The
wrapper must continue to reject that ambiguous owner. To recover, stop only a
PID whose executable and endpoint ownership have both been verified, restore
the bounded PID/status/log control files to the service account, and let the
canonical unit launch the node. Reprove tick, pool, neuron, concept, binding,
terminal, and zero-residency invariants before training resumes.

All admission observations use one infrastructure policy. Transient timeout,
connection, empty-output, or truncated-JSON failures are retried and recorded
as `passed: null`; they are never converted into semantic corpus intervals.
An evaluator missing its own corpus, executable, or required deployment
fixture is treated the same way: it produced no observation of brain behavior,
so the candidate is preserved for infrastructure repair instead of being
quarantined as learned regression.

Compiler execution is also an infrastructure observation when the brain
returns the byte-exact trusted fixture but that fixture fails to build. The
language evaluators return exit status `75` only when every execution failure
has that proof; the enterprise wrapper propagates
`infrastructure_only_failure`, and the supervisor retries it without weakening
ordinary semantic failures. Preserve both stdout and stderr because .NET can
put the causal compiler diagnostic on stdout and only the generic build
summary on stderr. Before any settled admission, enforce the configured
host-available-memory floor. If neuron residency is zero but allocator pages
still leave the host below that floor, checkpoint and topology-prove a node
recycle before launching compilers; if capacity is still insufficient, pause
as infrastructure rather than manufacturing a neural regression.
Do not share `DOTNET_CLI_HOME` across operator and service accounts: the CLI
creates a mode-`0600` user `NuGet.Config`. Each C# execution therefore uses its
disposable, account-owned workspace for both .NET home and NuGet packages;
manual root diagnostics cannot alter the service's later admission outcome.

If an older controller already recorded such a false interval, append an
explicit `resolved` event to the immutable deferred ledger with the missing
artifact and checksum restoration as evidence, restart the worker at its equal
RAM/WAL coordinate so it rebuilds its skip set, and rerun the same canary. Do
not erase the original event or accept its rows without a repeated behavioral
gate.
If retries are exhausted, the supervisor preserves the guarded candidate and
exits for a restart. Deferred replay restores its pre-replay guard before
pausing. A completed evaluator reporting a behavioral mismatch is the only
path that may reject or quarantine learned rows.

Each deferred interval retains an exact causal base for later bisection. Because the last-good `.wbrain` guard is already an immutable inode independent of the mutable live container, the deferred base uses a hard link to that guard when the filesystem permits it. Rollback always copies the guard into a new live inode before removing the guard name, so the deferred link remains immutable. When free space cannot hold the rejected live inode and its replacement simultaneously, the stopped-node transaction deletes only that unadmitted live inode and then recreates it from the untouched guard; an interrupted copy remains retryable because the accepted guard and quarantined row ledger are preserved. If rollback regenerates another guard for the same phase, row, storage type, and checkpoint-proven topology, the supervisor links new intervals to the existing causal-base inode instead of retaining another tens-of-gigabytes copy of the same accepted state. This makes quarantine publication effectively constant-space per distinct accepted state; cross-volume or link-restricted filesystems fall back to an independent copy.

Resolved interval bases are not permanent archives. After the append-only ledger records resolution, the supervisor folds the complete ledger and removes only known causal-base directories that no unresolved interval references; unknown experiment directories remain untouched. Before creating an independent `.wbrain` guard, the supervisor requires free space for one complete guard plus reserve. It does not reserve a second simultaneous container: after failed admission, the candidate is derived and disposable while its exact row interval is quarantined, so low-space rollback can remove that rejected inode before copying the accepted guard. A host that cannot hold even the one independent accepted guard still stops before training.

After every forward phase is complete, replay the folded deferred ledger with the same supervisor:

```powershell
python scripts/programming_curriculum_supervisor.py `
  --runtime runtime/brains/programming-integrated-20260713 `
  --endpoint http://127.0.0.1:18095 `
  --node-bin target/release/w1z4rd_brain_server.exe `
  --replay-deferred `
  --batch-size 8 `
  --lock-chunk-size 8 `
  --min-free-memory-gb 8
```

At million-neuron scale, keep the replay request and neural-lock envelopes at
eight episodes as shown. The earlier 32-episode envelope is not a safe default
for a fabric of this size: it can hold the response boundary long enough to
hide memory pressure and block protected inference behind training.

Replay is deliberately an end-of-corpus transaction. Each interval trains against the final accepted brain under an independent guard, recalls samples from its exact row window, and then passes the complete corpus/foundation/enterprise admission gate with that interval temporarily included. A passing interval is resolved and its unreferenced causal base pruned. A failing interval restores the final brain, remains unresolved with its diagnostic artifacts, and stops the replay loop for a test/family-fix/retry cycle. Use `--replay-interval-id PHASE:START:END` to retry one corrected interval without admitting any other pending range.

Corpus growth may change the ranking of individually compatible source
candidates without invalidating an independently grounded project
composition. For a request combining two or more behavior classes, or one
that explicitly asks for a project/modules/multiple files, a ready compatible
manifest composition must precede any single plain-source candidate. Preserve
the narrower source response only when the prompt explicitly asks for a
function, method, or snippet. This precedence is an output-contract invariant,
not a popularity tie-break: otherwise a later source episode can make a valid
multi-component container unreachable even though component retrieval and
composition both succeeded.

Corpus growth can also make a familiar learned word appear inside a novel
single token. Character-motif overlap across that boundary is not grounding:
for example, an observed “arithmetic mean” episode must not answer the unseen
atomic symbol `quasarithmetic`. Exact one-token sensory episodes remain
authoritative, while a non-exact prompt containing only one lexical token must
abstain. Multiword paraphrases continue through the normal raw and independent
feature-pool routes. Always invoke the protected foundation evaluator with
`--details` so a failed OOV gate durably records the exact prompt and reply,
not only an aggregate count.

During forward harvest, a guarded block may be completely covered by unresolved
deferred intervals. Its admission sampler must then contain zero rows. Do not
manufacture a pass and do not append an empty interval: verify gap-free
half-open coverage from the accepted guard row to the block boundary, preserve
the unchanged guard bytes and topology proof, advance only the guard's logical
row, and train the next block. A single uncovered row forbids this transition.
At the corpus boundary, the same proof permits the existing forward-harvest
handoff into deferred replay.

The forward-harvest report is the durable commit record for that handoff.
Publish end-of-phase progress and the report before releasing the immutable
guard. On restart, validate the report against the exact phase, end offsets,
accepted guard row, and the still-unresolved interval IDs covering the entire
tail. If they agree, resume the handoff idempotently; never rerun a completion
sampler over an all-deferred window. If any field or interval differs, refuse
the shortcut and retain the normal admission path. This closes the crash
window between guard release and publication of `deferred_intervals_pending`
without treating quarantine as acceptance or losing a replay obligation.
The `--replay-deferred` process is a separate stage: it first verifies that
all forward progress files are terminal inside `run_deferred_replays`, then
opens one exact interval transaction. It must bypass every ordinary forward
completion gate, because an all-deferred terminal phase has no eligible
forward recall rows by construction.

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
For every failing retention family, preserve row-level evidence before rollback:
the prompt, expected or accepted output, actual reply, grounding decision,
route diagnostics, and assertion result. Aggregate counts locate a broken
contract but cannot support a later causal repair once the candidate state has
been discarded.
Treat output shape and delivery semantics as part of behavioral compatibility.
For example, a word-frequency function must return its mapping; code that
computes the same mapping only to write a framework-managed file is not an
interchangeable response.

## Event-driven Codex supervision

Healthy corpus work does not require a reasoning agent to poll it. Run the
small local watcher with the explicit Codex thread that owns this project:

```powershell
python scripts/aws/watch_programming_brain.py `
  --session-id $env:CODEX_THREAD_ID
```

The watcher queries a compact AWS lifecycle probe every five minutes. A live
supervisor or wrapper remains authoritative even while it reports a canary
failure that automatic quarantine recovery still owns. Codex is resumed
immediately for the durable forward-to-quarantine handoff or an automated-stage
completion. A serious stop, stale heartbeat, or stopped AWS host requires two
identical observations before waking Codex. The event fingerprint is persisted
below the ignored `runtime/` directory, so an identical event cannot repeatedly
consume agent turns within the default thirty-minute repair window. If Codex
returns without changing an actionable condition, the condition is eligible
again after that cooldown; a healthy observation clears it. This prevents both
tight reasoning loops and permanent one-shot abandonment.

The invocation uses `codex exec resume <session-id>` with the saved CLI login,
the repository as its working directory, `approval_policy=never`, and the
existing full-access sandbox setting. It records JSONL and stderr logs below
`runtime/programming-brain-codex-watch/logs/`. Use `--once --dry-run` to inspect
classification without waking Codex. Do not use `--last`: an unrelated newer
Codex task could otherwise receive the programming-brain event.

On Windows, `scripts/aws/show_programming_brain_watch.ps1` is a detachable
viewer. It starts the hidden watcher only when no watcher process exists, shows
the latest AWS phase/row/process snapshot, and follows `activity.log` in real
time. Closing the PowerShell window stops only `Get-Content -Wait`; it does not
signal, kill, or restart the watcher. A desktop shortcut may safely open and
close this viewer independently as often as needed.

The resumed prompt carries the final acceptance contract, not merely the
current corpus alarm. It requires a 1,000-task deterministic enterprise
software obstacle course with capability-family remediation and full
regression, followed by the CoolCryptoUtilities brain selectors and C0D3R V2 /
Brand Dozer Multi-Scale Robot World capstone. The capstone is independently
judged for real 3D robot construction, credible real-world physics, and
fabrication-ready 3D-printing output; the model's own completion claim is never
admission evidence.

The complete, auditable definition of done is
[`PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md`](PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md).
The watcher exits only after the contract's generated completion marker contains
all required passing fields. The marker is a routing guard, not proof by itself;
its referenced reports remain authoritative.
