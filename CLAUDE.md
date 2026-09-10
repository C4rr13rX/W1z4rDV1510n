# W1z4rD V1510n — Claude Code Project Config

## Project Overview
Distributed AI/neural computing node system with cluster, P2P gossip, wallet, and dashboard.
Owner: C4rr13rX (c4rr13rX@gmail.com) | Repo: https://github.com/C4rr13rX/W1z4rDV1510n

## Workspace Structure
- `crates/core` — neural fabric, Hebbian learning, neuro API (port 8080), sensor streams
- `crates/cluster` — P2P cluster ring, OTP join, gossip, heartbeat/election
- `crates/node` — main node binary (`w1z4rdv1510n-node`), node API (port 8090), all HTTP routes
- `crates/dashboard` — egui/eframe desktop GUI (`w1z4rd-dashboard`)
- `crates/experimental-hw` — GPU/hardware experiments

## Build
```bash
export PATH="$PATH:/c/Users/Node/.cargo/bin:/c/Users/Node/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
cargo build --release --workspace
```
Toolchain: `stable-x86_64-pc-windows-gnu` (requires WinLibs MinGW-w64 for `dlltool.exe`).

## Run
```bash
# Node — launch from project root (config is relative to CWD)
cd /d/Projects/W1z4rDV1510n
W1Z4RDV1510N_DATA_DIR="D:\\w1z4rdv1510n-data" ./bin/w1z4rd_node.exe

# Dashboard
./bin/w1z4rd_dashboard.exe
```
Project dir: `D:\Projects\W1z4rDV1510n\` — always launch node from there.
Neuro pool data dir: `D:\w1z4rdv1510n-data\` (set via `W1Z4RDV1510N_DATA_DIR` env var).

## Deploy after build
```bash
# Copy fresh node binary to bin/
cp target/release/w1z4rdv1510n-node.exe bin/w1z4rd_node.exe
```

## Key Ports
| Port  | Service         |
|-------|-----------------|
| 8080  | Neuro API       |
| 8090  | Node API        |
| 51611 | Cluster (SIGIL) |

## Node Modes
- `SENSOR` — local AI/streaming mode, wallet optional (set in `node_config.json`)
- `PRODUCTION` — full Web3 mode, wallet required

## Read before changing the brain

Read the guide for the area first. These document facts that are cheap to
verify and expensive to assume, and each records what has already been tried.

| Area | Guide |
|---|---|
| Recall, routing, similarity scoring, `/brain/chat` answer branches | `docs/RECALL_PATH_FIELD_GUIDE.md` |
| Curricula, corpora, benchmarks, genetic search | `docs/BRAIN_CONFIGURATION_FIELD_GUIDE.md` |
| Fabric internals: atoms, concepts, pools, bindings | `ARCHITECTURE.md` |
| Host operations, deployment, supervisor | `docs/PROGRAMMING_BRAIN_OPERATIONS.md` |

**Never report training as working without checking that it CONVERTS.**
A healthy-looking curriculum that admits nothing has cost real money twice:
the supervisor says `active`, the brain answers `/health`, the tick advances,
and 0 intervals are admitted (measured: 0 admitted / 18 failed over 48 h).
Run `python scripts/aws/admission_watchdog.py` in the background — it exits 2
with the named fault, which re-invokes the agent. The measurement is the
`resolved` count rising, not that a process is alive.

Verify, do not assume:

- **An atom is a byte, not a word.** `ARCHITECTURE.md` lines 27, 190, 281.
  Confirm with `/stats`: `total_neurons` minus `total_concepts` is the atom
  count — measured 879 across 2.55M neurons.
- **The answer branch is an `if/else` chain.** An earlier arm that matches
  and returns `None` ends it. Read `intent_diagnostics.answer_branch` rather
  than inferring which route ran.
- **A live curriculum trains underneath any measurement.** Sample repeatedly;
  one probe is not verification.
- **The heartbeat is whichever writer is freshest, and there are THREE.** The
  replay worker writes `deferred-replay-*.progress.json`; the forward driver
  writes its own `<phase>.progress.json` (its `--progress-path`); the
  supervisor writes `curriculum-supervisor.status.json` only BETWEEN batches,
  so it freezes for minutes during a canary, a settlement or a gate. Both
  fixed choices have now produced a false reading. Measured 2026-09-09: the
  replay file was 100.7 h old reporting `durable_next_row` 201344 beside a
  live status at row 16,416 advancing at 15.3 rows/s. Measured 2026-09-10 mid
  `continuous_canary`: the status file was 718 s stale at row 49,152 while
  `go-systems.progress.json` was 7.6 s old climbing 50192 → 50224, and the
  watchdog published `rows_per_second: 0.0` and woke an agent to diagnose a
  stall on a healthy block. Read `heartbeat.source` and confirm it names the
  file matching the CURRENT phase before believing any rate. A rate of 0 is
  normal: settlement, the admission gate and the continuous canary all freeze
  the row by design.

  **The freshest writer is not always a writer of ROWS.** Selecting on mtime
  alone picked `curriculum-supervisor.status.json`, which during a replay
  carries `resume_row`/`end_row` and never `durable_next_row` — so the probe
  waited the full 120 s bound for a row that file cannot contain and published
  `row: null, rows_per_second: null`. That is worse than a wrong rate:
  `classify_probe` gates its convergence annex on `row is not None`, so BOTH
  branches fell through and the alarm went out as a bare "no interval admitted
  for 111.6h" — against a replay converging at 14.0 rows/s with zero rollback
  exposure, 82,272 rows from its gate. The selector now requires a row and
  publishes `row_source_lag_seconds` (0 means it IS the freshest; large means
  the only file exposing a row is a leftover, so its rate describes the past),
  plus `no_row_writer: true` when nothing exposes one. Verified live: source
  `replay_progress`, row 51,464, 11.99 rows/s, `sample_seconds` 2.0.

- **`curriculum-supervisor.status.json` is not ONE schema — it is whatever
  lifecycle event wrote last.** A forward block publishes `block_target_row`;
  a deferred replay publishes `start_row`/`resume_row`/`end_row`; a
  `resource_node_recycled` record publishes neither, only `trained_rows` and
  a `topology` dict. So any consumer reading a field from that file gets an
  answer that depends on timing. This broke the SAME convergence annex twice
  in one session on 2026-09-10: the fix above made the annex require a row,
  but it still read the TARGET from `status['block_target_row']` — a key only
  the forward stage writes, and a forward stage is exactly where the drought
  branch is already suppressed. In replay, the one stage that admits and the
  only one that reaches that branch, the target was always `None`, so all
  three arms fell through and the alarm went out bare a second time: "no
  interval admitted for 112.6h" against a block at row 86,320 of 131,072
  advancing 12.8 rows/s, ~1 h from its gate, at a 1800 s retry cooldown. Every
  test covering the annex hand-set `block_target_row`, so 47 tests passed
  against a payload no replay ever emits. **Read a block's target from durable
  interval state** — `deferred-replay-active.json`, or the `interval_id`,
  which encodes `phase:start:end` and no recycle can erase — and publish it in
  the heartbeat beside the row it is measured against. Verified live: `row
  91,752 of 131,072, 16.0 rows/s, reaches its gate in about 0.7h`.

- **A failure field truncated from the FRONT deletes the cause.**
  `replay_worker_failure` appends up to 4000 bytes of worker stderr after the
  log path so the reason travels with its address; the probe then cut
  `last_failure` at 180 characters, and the runtime path alone is 135. A
  traceback's one informative line is its LAST. Measured 2026-09-10: this
  wake-up published `deferred replay worker exited 1; stderr=<path>` and
  nothing else; the named file held a `SchemaError` for
  `category='systems_programming_go'` already repaired 3 h before the current
  supervisor started, so the answer was in the payload's own source and still
  cost a round trip to the host. The field now keeps head AND tail and says
  how much it dropped, and `last_failure_suites` extracts the failing suite
  names — the reporting half of the next lesson. **Classify on the whole
  error, never on the display summary**, or trimming for readability silently
  moves the worker-vs-gate split.

  **A ledger entry is not evidence about the process that is running.**
  `curriculum-health.jsonl` is append-only and outlives every writer, so its
  newest record can name a cause repaired generations ago — and an instruction
  to "timestamp `last_failure` against process start" is unusable when the
  payload carries no timestamp. Measured 2026-09-10 on the `quarantine_ready`
  wake-up: `last_failure` was a worker exit **19.6 h** old, quoted against a
  supervisor **0.14 h** old, whose registry `SchemaError` had been redeployed
  **16.6 h** earlier. Two SSM round trips went to proving the text was
  history. Worse, the natural hypothesis was wrong in an instructive way: 65
  worker exits with a 137-character maximum, beside 1116-byte stderr logs,
  reads exactly like the tail append being inert — but the deployed supervisor
  has it (`defines`/`calls`/`marks`/`appends` all true) and simply has not
  failed since. **Check whether a suspect code path has had the OPPORTUNITY to
  run before concluding it is broken**; a count of zero across a window where
  nothing happened is not a defect. The payload now publishes
  `last_failure_age_hours`, `supervisor_age_hours` and
  `last_failure_predates_supervisor`, and the drought alarm appends both ages
  when the named failure predates the supervisor.

- **An 11/12 enterprise gate names no suite in the ledger.** The
  `enterprise_gate_confirmation` record carries only counts, so a drought
  looks causeless from `curriculum-health.jsonl` alone. The per-suite verdict
  is in `<phase>.enterprise-gate.json` next to it, under `results[].name`,
  and the PER-CASE verdict is one level further out in the suite's own
  `polyglot.json` under `results[].executes` — `results[].name` carries only
  a boolean, so a walk looking for `passed` inside it finds nothing and
  reports a vacuous zero. Measured 2026-09-09: 127 gate runs, exactly one
  pass ever, `polyglot` failing every time on a single row —
  `javascript_go_order_workers` canonical composing `ledger.go` where the
  prompt asked for a deduplicator, so `go_deduplication` died on
  `stat dedup.go: no such file`. One row of one suite held every admission
  for over four days.

  **RESOLVED 2026-09-10, and the shape of the fix is the lesson.** The cause
  was not routing, ranking or composition order — every repair attempted on
  those was inert. Go had no grounded corpus at all, so the outbox behaviour
  had only ever been observed in JavaScript. `go_systems_001` (328k
  gofmt-validated CodeSearchNet rows) fixed it: the 8.8 h `polyglot.json`
  shows that case `executes=False` at projects 5/6, components 11/12; the
  1.9 h `_pretest_polyglot.json`, after ~99k go-systems rows had trained,
  shows the same case `executes=True` at 6/6, 12/12, OOV 2/2. **Check corpus
  coverage per requested language before theorising about the router.**

- **A forward stage does not admit intervals.** `hours_since_admission` says
  nothing about health while `service_stage` is `forward`: that stage
  harvests rows and admission belongs to deferred replay. Read the stage
  first. Measured 2026-09-10: an alarm at 102.7 h fired on a block advancing
  normally at row 82,960 of 131,072 with `accepted_episodes` rising in
  lockstep, whose named `last_failure` was a registry SchemaError repaired
  and redeployed 5.8 h earlier. Timestamp `last_failure` against process
  start before re-debugging it.

  **The watchdog has TWO emitters and they drifted apart.**
  `admission_watchdog.faults` learned to see a forward block on 2026-09-10
  (86d5020); `watch_programming_brain.classify_probe` — the one that actually
  publishes `fix_required` — did not, so the same host was simultaneously
  healthy and faulted. Measured hours later: 103.7 h published as a fault
  against `go-systems` at row 111,064 of 131,072 advancing 12.0 rows/s with
  `durable_next_row == ram_next_row` (zero rollback exposure). Because a
  forward phase is 328k rows and `retry_cooldown` is 1800 s, that alarm bills
  an agent wake-up every half hour for days. Both emitters now gate on
  `service_stage`; when changing one, change the other, and assert the
  suppression stays narrow — a FROZEN forward block must still alarm, because
  a forward stage that never reaches its handoff never admits either.

- **The row moves once per COMMITTED BATCH, not continuously.** Between
  commits the progress file is byte-identical, so any sample shorter than the
  commit period reads 0 rows/s on a perfectly healthy block. Measured
  2026-09-10: 32 rows per commit at 0.355 rows/s is one commit every ~90 s,
  and the watchdog's fixed 6 s sample caught it about 7 % of the time. The
  sample is now adaptive up to `HEARTBEAT_SAMPLE_SECONDS` (120 s) with early
  exit. Sampled four hours apart the same block read 0.355 and then 7.99
  rows/s — instantaneous rate is a duty cycle, so never extrapolate an ETA
  from one window.
- **Onboarding a corpus requires a registry `.toml`.** Without it the driver
  exits 2 on `unknown script` and the supervisor retry-loops, stopping ALL
  training. `scripts/onboard_corpus.py` writes it; deploy it with the corpus.
- **A malformed registry `.toml` is worse than a missing one.**
  `load_registry()` is all-or-nothing, so one bad file kills every corpus at
  driver startup. Measured 2026-09-09: one invented `category` rejected 12
  intervals across 4 unrelated corpora as behavioural failures, ended the pass
  `deferred_replay_complete`, and exit-42 latched the service stopped for four
  days. After ANY registry edit run
  `python -m pytest tests/test_training_registry_schema.py` — it loads the
  real directory. Checking the field by eye is what let a second `SchemaError`
  (`must_be_valid`) survive the fix for the first.
- **A replay worker exit is infrastructure, never a semantic verdict.** The
  worker only POSTs rows; every judgement runs in the supervisor after the
  training loop returns. Before blaming an interval's content, read the
  worker's stderr tail — `last_failure` now carries it inline.
- **`check=True` beside `capture_output=True` deletes the evidence that
  classifies a gate failure.** The supervisor decides quarantine-vs-retry by
  scanning the failure text for markers like `timed out`. Measured 2026-09-10:
  `midphase_gate` had 0 `_infrastructure_retry` and 0 `_infrastructure_paused`
  against 45 `_failed`, beside 108/48 for `continuous_canary` and 146 for
  `idle_settlement`. A zero next to healthy neighbours means the branch is
  unreachable, not that the failure never happened. The cause: `debug_eval`
  ran its child captured-and-checked, so the child's `socket.timeout: timed
  out` went into a `CalledProcessError` nobody read, and the classifier saw
  only `returned non-zero exit status 1` — a string with the answer deleted.
  Two go-systems blocks (262,144 rows) were quarantined for a client timeout.
  **Ask what a child is ABLE to say before reading its exit code:**
  `programming_debug_benchmark.py` ends in an unconditional `return 0`, so its
  non-zero exit is never a verdict; `programming_code_eval.py` returns 1 on a
  real failure and prints its report, so its exit is a verdict only when that
  report parses. And when a report is read from a fixed path, **unlink it
  first** — the copy beside both quarantined candidates was a 769.7 h leftover
  reading 6/6, so relaxing the check without deleting would have traded a false
  quarantine for a false admission.
- **Files written over SSM land `root:root`.** The supervisor runs as
  `ec2-user` and dies with `Permission denied` on anything it must write.
  `chown ec2-user:ec2-user` after any host-side write.

## Important Notes
- Always commit and push after any code changes
- Kill old processes before deploying new binary (port conflicts cause silent API thread death)
- `node_config.json` in project root has `data.enabled: false` and `wallet.prompt_on_load: false`
- Neuro pool data lives at `D:\w1z4rdv1510n-data\` — set `W1Z4RDV1510N_DATA_DIR` before launching node
- The GNU toolchain requires WinLibs PATH to be set or dlltool errors occur
- Avira AV may quarantine Rust build artifacts — exclusions are set in Windows Defender
