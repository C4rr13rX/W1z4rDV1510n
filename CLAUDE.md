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
- **A counter that went BACKWARDS is a reset, not a negative rate.** Both
  heartbeat counters — `durable_next_row` and `accepted_episodes` — live in
  the replay worker's progress file and restart with the worker, so any
  sample straddling a `deferred_replay_resource_yield` sees the value fall.
  Measured 2026-09-10: `accepted_episodes` went 712 → 8 and the payload
  published `accepted_per_second: -42.4`, which the drought annex renders as
  "still accepting -42.4 episodes/s, so the block is training". Rates now
  report `None` on a decrease and the heartbeat carries `counter_reset`. A
  genuinely frozen row still reads 0.0 — settlement, the admission gate and
  the continuous canary each freeze it by design — so **do not collapse a
  frozen row into a reset**; they are different facts with different actions.

- **A census of wrapper 0 / supervisor 0 / worker 0 can mean the VOLUME is
  full, not that the stage ended.** Measured 2026-09-10: `/srv/wizard` reached
  20 KB free on 1.0 TB, so the wrapper died writing its 6-byte `node.pid`
  (`OSError: [Errno 28] No space left on device` on `node.pid.<pid>.tmp`),
  systemd restarted it 115 times at `RestartSec=10`, and the probe's census
  landed between restarts. The alarm therefore read "no curriculum supervisor
  or wrapper owns terminal state `deferred_replay_resource_yield`" — true, and
  useless. The unit was `activating (auto-restart)`, never `dead`, and the
  supervisor's own `--min-free-disk-gb 8` guard is DOWNSTREAM of a crash that
  happens before any supervisor is launched. The payload reported memory and
  never disk, so the cause was one `statvfs` away and cost two SSM round trips
  to rediscover from a 57 MB traceback log. `classify_probe` now publishes a
  `disk` block and names the fault; `admission_watchdog.faults` already had
  `disk_low`, which is the two-emitter drift again — **change both.** Read
  `systemctl show -p ActiveState -p SubState -p NRestarts` before believing a
  zero census: `auto-restart` is a crash loop, not an absence.

- **`du` inflates on XFS reflink; only `df` measures a reclaim.** The same
  volume reported 2.48 TB of `st_blocks` inside 1.0 TB, because `reflink=1`
  shares extents and every file counts them in full. Measured 2026-09-10:
  deleting nine deferred directories holding ~560 GB of apparent `st_blocks`
  returned **0.00 GB** — the causal bases are `os.link` hardlinks to the
  last-good guard (`st_nlink` 85, 17, 4, 3, 2 on single inodes) and reflink
  clones of the live brain, so their size is almost entirely shared. **Predict
  reclaim from `df` before and after, never from summing file sizes**, and
  check `st_nlink` before assuming a name owns its bytes.

- **One undeletable directory stopped ALL disk reclaim for five weeks.**
  `prune_resolved_deferred_bases` is the only routine that frees multi-gigabyte
  causal bases, and its `shutil.rmtree` sat bare in the loop. Exactly ONE
  deferred directory was `root:root` — the SSM ownership trap — and unlinking
  needs write permission on the DIRECTORY, not the file, so the supervisor
  (which runs as `ec2-user`) raised `PermissionError` out of the loop before
  reaching any later digest. A single `chown` was the entire repair. The reclaim
  is now per-directory and publishes `deferred_base_prune_blocked` naming what
  it could not remove: **a partial reclaim that reports nothing is
  indistinguishable from a complete one**, which is exactly how this ran for
  five weeks under a policy that was working as designed.

- **The `.wbrain` neuron store is append-only and has NO compactor.** This is
  the standing cause of disk growth, and the code says so: `store/cold.rs`
  lines 7–10 ("no LSM compaction in this first cut: every eviction appends …
  reclaimed by a future compaction pass (Stage 17.4 follow-up)") and
  `store/neuron_store.rs:325` ("Old records become garbage; a future compaction
  pass reclaims"). That follow-up was never built, so every sleep/evict appends
  a fresh record and superseded ones are never returned. Measured 2026-09-10:
  `brain/brain.wbrain` was **1068.79 GB** against a brain of 4.81 M neurons
  whose resident RSS was 11.77 GB. Growth tracks TRAINING ACTIVITY, not brain
  size, so a bigger volume buys time and never a fix. The WAL has compaction
  (`store/wal.rs`); the neuron store does not — do not confuse the two. Burn
  rate measured across four windows: 125.4, 147.2, 73.8, 102.7 GB/h —
  **sustained ~112 GB/h**, not a burst. Dividing the 1068 GB file by five weeks
  to get "1.2 GB/h" is wrong; the checkpoint is rolled back and regrown, so its
  size is not a running total. **The bytes are not new information**: over 301 s
  the file grew 8.59 GB while `total_neurons` rose by 1,040 — 8.26 MB per
  neuron, 8.45 MB per tick. Growth tracks TICKS. The brain sits at 11.57 GB RSS
  on a 15.26 GB host with 3.02 GB free against a 3 GB floor, so it evicts
  continuously and each eviction appends a body that is never reclaimed —
  **memory pressure is converted into permanent disk growth**, which is why more
  disk is the wrong purchase. Size disk alarms from the burn rate, not from the
  supervisor's 8 GB yield guard, which is under four minutes of warning.

- **`worker_count: 0` during a replay is the NORMAL reading, not a stall.**
  The deferred-replay worker is `tools.training_standard.drive_corpora_brain`,
  and `run_deferred_replay_worker` stops and respawns it once per cooperative
  memory yield, so an instantaneous process census lands in a trough most of
  the time. Measured 2026-09-10 on the `quarantine_ready` wake-up: two
  censuses 90 s apart both read `worker 0`, while the same progress file
  advanced 217,024 → 217,192 (1.87 rows/s) and `accepted_episodes` reset
  344 → 168 — a worker demonstrably running and demonstrably restarting.
  **Liveness is the ROW DELTA; the census only distinguishes a yield cycle
  from a host that has lost its supervisor.** The watchdog is right to gate
  on `wrapper_count` (`bash run_programming_curriculum_service.sh`) rather
  than on the worker. And guessed `pgrep` patterns (`deferred_replay_worker`,
  `deferred_replay`) match no process at all, so they report 0 forever —
  copy the patterns from `watch_programming_brain.py`'s `/proc` scan rather
  than inventing them.

- **`deferred-replay-active.json` carries the rejection that SENT an interval
  to quarantine, not the state of the retry now running.** The record for
  `jupyter-scientific-full:201344:262144` quoted an `enterprise regression`
  with `polyglot` at 11/12 — the four-day drought signature — whose inner
  `updated_unix` was **73.4 h** old, three days before the Go corpus closed
  it, beside an outer record 33 days old and a `created_unix` 1.0 h old.
  The same file therefore holds three different clocks. Read `created_unix`
  and `state` for what is running; treat `interval.error` as the reason it
  was queued, and date it before re-debugging it. Same rule as
  `last_failure`: a ledger entry is not evidence about the running process.

- **The `.wbrain` is NOT mostly garbage, and compaction cannot free this
  volume.** The compactor that `cold.rs` promised since Stage 17.4 now exists
  (`store/compaction.rs`, `wbrain_compact` binary, `--inspect` / `--estimate` /
  `--in-place`). Running it here would make things WORSE, and the measurements
  say why. Measured 2026-09-10: `brain.wbrain` 576.67 GB holds **363.34 GB of
  live bodies** — 5,086,800 neurons at a mean body of **71 KB** — so only
  ~213 GB is reclaimable garbage. Meanwhile `brain.last-good.wbrain` is a
  **reflink clone**: `fiemap` returns identical physical blocks for both files
  at 6 of 7 sampled offsets from 4 GB to 412 GB. The tree reports **7,417 GB
  apparent across 5,131 files against 596 GB actually used**. A compacted copy
  writes 363 GB of fresh, unshareable blocks and then reclaims only the
  ~153 GB the original did not share — ending near 292 GB free against 502 GB
  before. **Predict a reclaim from block sharing, never from file size**, and
  run `wbrain_compact --estimate` before assuming a container is mostly
  garbage.

- **The burn is full-body rewrites forced by memory pressure.** A 363 GB brain
  on a 15.26 GB host must evict continuously, and every sleep appends the
  WHOLE 71 KB body — there is no delta encoding — which both breaks reflink
  sharing and is never reclaimed. That is the 112–257 GB/h. No compaction
  schedule outruns it; the fixes are more RAM (fewer evictions), delta-encoded
  terminal updates, or a larger volume. All three are user decisions.
  `--min-free-disk-gb` is now 150, sized from that burn: it buys a clean
  cooperative yield instead of the 115× ENOSPC crash-loop, and buys no
  headroom at all.

- **`prune_resolved_deferred_bases` cannot see a RETIRED interval.** It removes
  `known - active`, where `known` is every `interval_id` in the ledger. Measured
  2026-09-10: 113 deferred directories and 110 base files against 89 ledger-known
  intervals (26 deferred, 63 resolved) — so ~87 directories are "unknown" and
  deliberately preserved, and the pruner returned **0 directories / 0.00 GB**.
  With 70 `unrestorable_quarantine_retired` events, retirement appears to drop an
  interval out of `known`, which makes its base unprunable forever. A pruner that
  reclaims nothing looks identical to one with nothing to reclaim.

  **Re-measured 2026-09-10 against the CORRECT ledger, and the reclaim is
  exhausted, not merely blocked.** The first census read
  `deferred-intervals.jsonl`; the real path is
  `curriculum-deferred-intervals.jsonl`, so every digest fell into "unknown,
  deliberately preserved" and the answer was another vacuous zero — check the
  filename against `deferred_intervals_path()` before believing a population
  count. Against the real ledger (523 lines, 459 deferred / 64 resolved): 113
  directories on disk, 89 ledger-known, 26 protected, **2 prunable**, 85
  unknown-preserved, and an upper bound of **0.01 GB** returnable against
  **399.56 GB pinned by cross-links**. There is no reclaim left on this volume.

- **Waiting on DISK is not a yield — nothing ever frees it.** `--min-free-disk-gb
  150` converted a 115× ENOSPC crash loop into a cooperative yield, and the yield
  landed in `while (memory low) or (disk low): publish("resource_waiting");
  sleep`. Memory leaves that loop on its own — `recycle_settled_runtime_node`
  hands the allocator's arena back, measured 2.99 GB → 14.66 GB. Disk does not:
  the store is append-only, `prune_resolved_deferred_bases` is called at 3686,
  4065 and 4321 and at **none** of the three disk-wait sites, and once the worker
  stops the volume stops falling and never rises. So it was an unbounded wait for
  an impossible event, and a SILENT one — the unit stays `active`, the row parks
  on a durable boundary, and every heartbeat rule here says a frozen row during
  settlement is normal by design. Strictly harder to see than the crash loop it
  replaced. Fixed: reclaim before waiting, three attempts, then publish
  `disk_exhausted_unrecoverable` and keep waiting so a resized volume still
  recovers unattended.

  **Both alarm floors sat BELOW the floor the supervisor halts at.**
  `DISK_ALARM_FLOOR_GB` is 48 and `admission_watchdog`'s trigger is 20, against a
  150 GB stop — so a supervisor parked on its own floor sits at ~149 GB free and
  BOTH emitters call it healthy. Verified by deleting the fix: the classifier
  returns `Decision(kind='healthy', reason='automation owns
  disk_exhausted_unrecoverable')`. **An alarm threshold below the guard it is
  watching can never fire.** The classifier also keys on `resource_waiting` +
  the floor carried in the payload, because a hung host cannot redeploy itself
  and the running generation always predates the fix.

- **The disk guard was on the stage that had FINISHED.** `--min-free-disk-gb`
  is enforced by `disk_floor_breached`, which is called only from the forward
  corpus-phase loop. Once `forward_remaining_rows` reaches 0 that loop is done
  and deferred replay does every remaining row — and `run_deferred_replay_worker`
  polled `replay_memory_floor_breached` and nothing else. Measured 2026-09-11:
  the running supervisor carried `--min-free-disk-gb 150` on its own argv while
  replay trained at 96.39 GB free and 108.66 GB/h, ~40 minutes from the ENOSPC
  crash loop that floor exists to prevent. **Both watchdog emitters called it
  healthy**, because the previous fix keyed on `resource_waiting` — a supervisor
  PARKED on its floor — and this one never reached it. An alarm floor below the
  guard it watches can never fire; neither can a guard attached to a stage that
  has ended. Training below the supervisor's own floor is now its own fault in
  both emitters (`disk_floor_unenforced`), and the disk arm is checked BEFORE
  the drought arm, because a full volume causes droughts: at 96.39 GB this
  classified as `no_admission: 11.1h` and would have sent an agent hunting a
  semantic repair. A disk yield must also HALT (`DISK_EXHAUSTED_EXIT` 90,
  `RestartPreventExitStatus=42 90`) rather than respawn — a memory yield ends
  because the recycle returns the arena, but nothing returns disk, so a
  respawning disk yield is a faster path to ENOSPC than no guard at all.

- **The burn is a handful of HOT ATOMS, and the "71 KB mean body" hid them.**
  363 GB across 5.09 M neurons really is ~71 KB on average, and that average is
  useless here because the neurons being evicted are nothing like the average
  one. Measured 2026-09-11 by walking the container's own records
  (`W1ZNEUR1` + pool:u32 + id:u32 + len:u64 + body): mean body **8.7 MB**,
  median **1.47 MB**, max **86.2 MB**, and the **top 12 records are 85.9 % of
  the bytes**. They are pool 5, neuron ids 25, 28, 35, 43 — low ids, so single-
  byte ATOMS, at 82.2, 81.1, 77.1 and 18.3 MB. The hottest bytes in the corpus
  accumulate millions of terminals, and `evict_neuron` sleeps atoms whenever a
  `.wbrain` store is attached (the never-evict-atoms rule holds only for the
  legacy cold tier). So the whole-brain `/brain/sleep` that
  `settle_brain_for_admission` runs on every memory yield — every two or three
  minutes — rewrites those atoms in full, and they page straight back in
  because they are the hottest neurons in the fabric.

  **Do not derive a body size from `df` divided by `page_outs`.** That is what
  produced "39,971,683 bytes per body": the quotient silently attributes every
  other writer on the volume to eviction. Read the record headers.

  **"Which records are biggest" is not "which records are being written", and
  the 85.9 % above answers the first.** That figure comes from walking the
  container's TAIL, which is an all-time picture. Re-measured 2026-09-11 by
  walking only the region appended DURING a live 240 s window — the bytes the
  burn actually wrote — the distribution is much flatter: 12.855 GB appended
  (192.8 GB/h, against 237.65 GB/h by `df`, so the container is most but not
  all of the volume's drain), and within the 4.269 GB the walk could parse
  before a concurrent append truncated it, **407 distinct neurons, mean body
  10.5 MB, and the twelve largest only 22.6 % of the bytes**. They are ~82 MB
  each and uniform to within 1.4 % (83.11, 82.94, 82.93, 82.69, 82.42 …), low
  ids in pools 1 and 5, so atoms — but there are dozens of them, not a dozen.
  Pinning the hot set is therefore NOT a fix on this host: ~12.8 GB of atom
  bodies per sleep cycle against 15.26 GB of RAM and a 3 GB floor.

  **And nothing was written twice: `max_rewrites_of_one_neuron` was 1 across
  the whole window.** That is the structural reason `clean_skips` reads 0 — not
  that the digests miss, but that a whole-brain sleep touches each neuron once
  per cycle, so there is never a second write to compare against. A suppression
  keyed on repeats cannot fire in a workload with no repeats. The burn is one
  full ~82 MB body per hot atom per memory yield, every ~2.4 minutes, to record
  a few KB of new terminals — so the remaining lever is delta-encoded terminal
  appends, and neither pinning nor compaction touches it.

- **A counter of zero from a path that has not had the opportunity to run is
  not a refutation — including when it is your own fix.** The first reading
  after deploying the clean-skip suppression was `clean_skips: 0` beside an
  unchanged 95.79 GB/h, which reads exactly like an inert fix. But the node had
  restarted 90 s earlier and recorded **235 page-outs against 5.08 M neurons**,
  so almost nothing had been evicted twice and the skip had no case to decide.
  The digest map is per-process by design, so a fresh node always appends
  first. Measure after several sleep cycles, not after the first.

  **Re-measured, and the suppression is genuinely inert here — for a reason
  that names the real fix.** Second reading: `page_outs` 129, `clean_skips` 0,
  burn 95.9 GB/h. Two facts explain it. `evicted_neurons` is already
  5,080,629 of 5,081,059, so a whole-brain `/brain/sleep` re-sleeps almost
  nothing — `evict_neuron` returns early on an already-evicted id — and only
  ~130 neurons are paged in and back out per node lifetime. Those ~130 are the
  hot atoms, so repeats DO occur within one process, and they still never
  match. **The hot atoms are exactly the neurons training mutates every tick**,
  so their bodies differ on every eviction by construction. Suppressing
  redundant writes cannot help a workload whose writes are not redundant. The
  remaining levers are delta-encoded terminal updates (append the new terminals
  rather than an 82 MB body), capping atom fan-out, or enough RAM to keep those
  atoms resident — the first is real work, the last two are user decisions.

  **And the counters reset with the node, which recycles every 2–3 minutes.**
  `page_outs_delta` published **-1008** against a total of 129: the same
  counter-reset trap already documented for `durable_next_row` and
  `accepted_episodes`, now reproduced in instrumentation added to diagnose it.
  Any per-process counter on this host must be read as a reset on a decrease,
  never as a rate.

- **The burn was re-appending bodies that had not changed.** `persist_sleeping`
  → `append_record` → `append_neuron` wrote a full body on every page-out
  unconditionally. A brain whose live bodies total ~363 GB cannot be resident on
  a 15.26 GB host, so it evicts everything it owns (`evicted_neurons` 5,105,285
  of `total_neurons` 5,105,285, `resident_terminals` 0) and pages neurons back
  in to READ them; each read-only round trip re-appended ~71 KB that no
  compaction returns. Stopping the worker took the volume from 108.66 GB/h to
  **-0.0 GB/h**, so the attribution needs no argument. `append_record` now
  serializes first, compares an FNV-1a digest against what is already at the
  offset the slot points to, and skips the write when they match — safe by
  construction rather than by reasoning about dirty flags, since identical bytes
  at the same offset leave the store in the state the write would have produced.
  The digest map is in-memory and deliberately not persisted, so a rollback,
  compaction or reopen falls through to the append. Report `clean_skips` beside
  `page_outs`: their ratio is the only direct readout of how much growth is
  learning rather than churn. **Size the curriculum against the burn before
  believing it can finish**: 2.78 M deferred rows at 1.652 rows/s is 468 h,
  which at the old rate is ~51 TB of appends on a 1 TB volume.

- **The designed rollback reclaims the volume; deleting files by hand does
  not.** `deferred-replay-active.json` carrying `state: training` makes
  `recover_interrupted_deferred_replay` call `restore_rejected_deferred_replay`,
  which reflink-clones `brain.last-good.wbrain` over `brain.wbrain` and
  `os.replace`s it — unlinking the old inode and returning everything unique to
  it. Measured 2026-09-11 by extent subtraction (`filefrag -v`, never file
  size): `brain.wbrain` 982.43 GB with **587.83 GB unique**, against
  `brain.last-good.wbrain` 394.59 GB with **0.0 GB unique** — deleting the guard
  would free nothing, and restarting the supervisor frees 587.83 GB as designed.
  It costs exactly the unadmitted interval the invariant discards anyway.

- **The interval is larger than the disk window, which is a LIVELOCK, not a
  stall.** Durable progress survives a memory yield within one supervisor
  generation (measured 222,104 → 222,520 → 223,240 → 223,456) but a supervisor
  RESTART rolls a `state: training` marker back to its interval start. Measured
  2026-09-10: interval `jupyter-scientific-full:201344:262144` is 60,800 rows at
  ~1.1 rows/s ≈ **15.4 h of training**, against a disk window of ~45 GB of
  headroom at **128–160 GB/h ≈ 0.4 h**, on a volume whose burn is ~100 % neuron
  eviction (`evicted_neurons` 5,098,116 of `total_neurons` 5,098,439 — the brain
  evicts essentially everything it owns because 363 GB of bodies cannot be
  resident on 15.26 GB). 453 `deferred_replay_resource_yield` and 324
  `deferred_replay_failed` against 22 admissions is that livelock's signature.
  **Compare the interval's ETA against the disk window before treating a yield
  as recoverable**; no reclaim, floor or retry setting fixes an interval that
  cannot fit.

- **`wbrain_compact` has no `--estimate`, and `--inspect` cannot measure this
  store.** The deployed binary's usage is `--inspect` / `--in-place` /
  `<src> <dst>`. `--inspect` on the 850 GB brain returned in **0.0 s** with
  `pools_with_offset_vec 0` and `live_in_offset_vecs 0` — it reads metadata only
  and cannot report live bytes for slot-table pools. So compaction cannot be
  estimated on this brain, let alone automated from an estimate. Do not quote a
  live/garbage split for it without a measurement that actually walked the bodies.

- **The SSM transport rewrote its own payload.** `ssm.py` read scripts with
  `read_text()`, whose universal-newline handling turns CRLF into LF. A
  51,846-byte patch arrived as 51,750 — exactly its 96 CRLF pairs — so
  `git apply` rejected `pool.rs` and `wbrain_store.rs` (the two files stored
  with CRLF) while the four stored with LF applied cleanly. That is
  indistinguishable from host source drift, and the md5s disproved drift: all
  six matched the base commit. It reads bytes now. Also: `send_and_wait` passed
  the payload as a command-line argument (Windows caps that at 32 KB — the
  error is `[WinError 206]`, which names the argument, not the caller) and
  raised only stderr on failure, so "patch: command not found" was lost and the
  host has no `patch` binary — use `git apply`, which needs no repository.

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
