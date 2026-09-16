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

- **A lifecycle NOTICE placed above the fault arms outranks all of them once
  its state becomes permanent.** `classify_probe`'s `quarantine_ready` arm
  matched `service_stage == "replay"` with any `deferred_replay_*` state and
  returned unconditionally. That was a one-shot handoff signal until
  `forward_remaining_rows` reached 0 — after which `service_stage` is `replay`
  for all 2.78 M remaining rows and the state is always some
  `deferred_replay_*`, so the classifier could emit exactly one verdict for the
  rest of the curriculum. Measured 2026-09-11 against the live payload: a
  volume at 96.39 GB under the supervisor's own 150 GB floor — the exact case
  `disk_floor_unenforced` had been added for hours earlier — plus near-ENOSPC,
  99 % inode exhaustion, a logged wrapper ENOSPC, control stale 9000 s beside a
  dead heartbeat, a 400 h drought and a gate that had never produced an
  artifact ALL classified `quarantine_ready`. And because `required_polls` is 1
  for that kind, it re-woke a billed agent every 1800 s against a replay that
  was merely training. **Every one of those faults has a passing test**, because
  the `probe()` helper never sets `service_stage` — 67 tests green against a
  payload the host can no longer emit, the same shape as the `block_target_row`
  annex. When a stage becomes the steady state, re-read every arm that ranks
  above it, and build at least one test on the payload the host ACTUALLY emits.

- **An interval that never reaches a verdict is re-selected by every restart.**
  `rejected_this_pass` is a set in memory, so it only protects against
  intervals that LOSE a gate: those return into the loop, join the set, and the
  queue advances. An interval that is killed instead — a disk halt, a reboot —
  is never marked, and `unresolved_deferred_intervals` sorts by
  `(phase, start_row)`, a total order identical on every restart, so the next
  generation selects the same `pending[0]` and dies the same way. Measured
  2026-09-11 on `jupyter-scientific-full:201344:262144`: 361 resource yields
  and 14 gate failures over 437 h, with 131 unresolved intervals behind it
  never getting a turn. Fixed by recording the stall in the health ledger from
  the startup recovery path and sorting the queue on it — a REORDER, not a
  filter, so every obligation stays eligible.

  **Its recorded cause was 86 h stale and already repaired.** All 14 failures
  are at-gate `enterprise regression ... passed_suites: 11`, and the newest
  names `polyglot` — the drought this file records as CLOSED by the Go corpus.
  The freshest gate artifact (`go-systems`, 13.8 h old) passes **12/12**. So
  the interval is not failing a capability; it cannot reach the gate at all.
  Date a gate verdict against the corpus that repaired it, not just against
  process start.

  **And the work unit is NOT sized in rows.** `go-systems:0:131072` and
  `go-systems:131072:262144` — more than twice the span — admitted 16.6 h and
  13.8 h ago with 1 and 4 resource yields. The difference is cost per row: go
  rows run ~12.6 rows/s end-to-end, jupyter-scientific ~0.84–2.4 rows/s, so
  60,800 jupyter rows need ~5.9 h against a disk window of ~1.26 h (300 GB of
  headroom at the measured 237.65 GB/h). **Compare an interval's ETA at ITS
  phase's measured rate against the disk window**; resizing on row count alone
  would have been another inert fix.

- **A halt that forbids restart cannot reach a reclaim that only runs at
  startup.** The disk guard calls `reclaim_disk_for_floor`, which calls
  `prune_resolved_deferred_bases` and nothing else — and that reclaim is
  exhausted here, returning 0.00 GB on three consecutive attempts. The
  supervisor then exits 90, and `RestartPreventExitStatus=42 90` makes 90
  terminal. But the reclaim that DOES return bytes on this volume is the
  rollback any interrupted interval already owes, and it lived only in
  `recover_interrupted_deferred_replay` on the STARTUP path — which an exit
  that forbids restart can never reach. So the halt was terminal by
  construction while holding the bytes that would have cleared its own floor,
  and an operator restarting by hand was the entire recovery mechanism.
  Measured 2026-09-15: the unit sat `failed` at `ExecMainStatus=90` for
  **107.7 h** with 151.16 GB free, and one `systemctl start` returned
  **444,551,897,088 bytes (414.02 GiB)**, taking it to 557.63 GB. Every
  liveness rule in this file was satisfied throughout: the volume was not full,
  no process was crash-looping, and the row was parked on a durable boundary.
  Fixed: the halt rolls back FIRST and only exits 90 if the volume is still
  below its floor afterwards. **When an exit code forbids restart, check what
  runs only at startup** — that is the set of repairs the exit just deleted.

- **Predict a rollback from extent subtraction, and the prediction is exact.**
  `brain.wbrain` held 854.02 GB allocated, **440.00 GB shared** with
  `brain.last-good.wbrain` and **414.02 GB unique**; the guard's own unique
  blocks are **0.00 GB**, so deleting the guard returns nothing. The live `df`
  delta after the rollback was 414.02 GiB — right to the second decimal. Parse
  `filefrag -v` by its documented columns (`ext: logical..logical:
  physical..physical: length: flags`) and subtract sorted INTERVALS: one probe
  summed the wrong column and printed `wbrain_total_gb 5799882.57` for an
  854 GB file, and another tried to build a per-block set of 223M blocks.

- **A rollback does not extend an interval's runway — it DISCARDS the
  interval.** The first version of the window census added the measured
  reclaim to the headroom, reasoning the bytes were available. They are, but
  not to the interval that is running: reaching the floor rolls back and starts
  the next interval from its first row. The dry run against live state caught
  it reporting a 1.16 GB window before any rollback had been measured, and it
  would have reported 829 GB straight after one — both wrong, in opposite
  directions, from the same error. **The runway is what sits above the floor at
  the moment the interval starts, full stop.**

- **Measuring a dead generation's rate to `now` measures the OUTAGE.** The
  halted interval trained 11,584 rows in 2.59 h and then sat on a dead host for
  107.7 h. `time.time() - created_unix` scores that as **103.6 rows/h against a
  true 4,475** — a 43x understatement that would have refused intervals which
  fit. End the clock at the progress file's mtime, which is when the row last
  advanced. Verified against live state before deploying, then confirmed in the
  event the recovery actually wrote: `rows_per_hour: 4475.0`.

- **Ordering by span is inert when nothing fits, and a span is the wrong unit
  anyway.** `order_replay_candidates` sorts by `(stalls, span)` so the queue
  drains what fits first — correct, and useless when the answer is "nothing".
  Measured 2026-09-15 against the live queue: **0 of 22** unresolved intervals
  could reach a gate inside the 3.64 h window, missing by 8x
  (jupyter-scientific-partial, 75,876 rows at 2,585.7 rows/h) to **156x**
  (jupyter-scientific-para4, 131,072 rows at 230.5 rows/h). Cycling all 22
  would have spent ~88 h of billed compute, one rollback and regrow each, and
  admitted nothing — with the unit `active` and the row advancing throughout.
  Row count is a proxy for cost that is wrong by **260x** between the phases
  actually queued (go-systems 59,778 rows/h against para4 230 rows/h).
  `replay_window_census` measures burn, runway and per-phase rate and refuses
  to spend when nothing fits, exiting 91 — deliberately distinct from 90,
  because 90 says resize or compact and 91 says split the work unit or cut the
  burn. It is a REFUSAL TO SPEND, not a retirement: every obligation stays
  `deferred` and eligible, and the census passes again the moment the burn
  falls or the volume grows.

- **A guard keyed on evidence the host cannot produce is inert, and that is
  this repository's most expensive recurring mistake.** The census first
  refused only on a phase rate with two or more admissions — and on the host it
  was built for, NO pending phase has two: `jupyter-scientific-partial` and
  `-full` have none at all and `para4` has one. The guard would have passed all
  22 doomed intervals through. It now measures rates from the STALLS it
  already records (`rows_trained`/`hours` per generation), treats an interval's
  own stall as sufficient evidence about itself, and folds stalls into the
  phase rate — so the queue converges in ~4 measured windows instead of 22.
  **Before shipping a threshold, check the host can actually reach it.**

- **The obstacle course's verdict is not reproducible: it fails on LOAD.**
  Measured 2026-09-15 on the same tree, minutes apart: the full `tests/` run
  (1,437 tests, 62 min) ended 12 failed / 1,425 passed with 10 of those in
  `test_programming_obstacle_course.py`; re-running only the obstacle course
  and its neighbours gave **2 failed / 518 passed** — and the two were
  DIFFERENT tasks (`concurrency_async_distributed-0013`,
  `architecture_multifile_integration-0006`) from the ten. Every failure
  asserts `- passed / + timeout` or `- failed / + timeout`, so the harness is
  timing out its subprocesses under concurrent load rather than judging them.
  One task (`algorithms_data_structures-0009`) fails identically at HEAD, so
  that one is real and pre-existing. **A flaky timeout cannot produce the clean
  1,000/1,000 the acceptance contract requires**, and it fails in whichever
  direction the machine happens to be busy — including scoring a broken
  solution as passing. Size the per-task timeout against a loaded host, or
  serialise the course, before reading any obstacle-course total as a result.

- **The stall record deleted the evidence that its own failure mode produces.**
  `record_replay_stall` published `rows_trained`/`hours`/`rows_per_hour` only
  when `rows_trained > 0` — reasonable on its face, since a rate of 0 rows/h
  folded into `measure_phase_rows_per_hour` would refuse every interval in the
  phase forever. But an interval too large for the window is exactly an
  interval that gets KILLED, and one killed before its first durable commit
  banks nothing, so the more hopeless the interval the less this recorded about
  it. Measured 2026-09-15: both `jupyter-scientific-full` stall records (written
  2026-09-10) carried a `reason` and nothing else, so **115 h later the phase
  still had zero rate samples**, all four of its intervals scored `unknown`,
  and `unknown` counts as eligible — so the census whose entire purpose is to
  refuse a doomed spend selected `:393216:524288` and spent the volume's whole
  1.31 h window on a span needing ~41 h. Fixed: `hours` is recorded
  unconditionally and a barren generation gets its own channel
  (`replay_barren_stalls`), never folded into a phase rate, refusing only the
  interval it measured and only once that generation was at least as long as
  the window — a two-minute reboot banked nothing because it was never given a
  chance. Verified live: the next stall carried `rows_per_hour: 3088.2` where
  its two predecessors carried nothing.

- **A per-interval verdict that selection never reads is inert, and this one
  hid behind an aggregate.** `replay_window_census` computes `fits`/`unknown`/
  `exceeds` per interval; `replay_queue_is_hopeless` collapses that to one
  boolean over the WHOLE queue (true only when nothing fits and nothing is
  unknown); and selection then took `pending[0]` from `order_replay_candidates`,
  sorted on `(stalls, span)`, which had never heard of a verdict. So a single
  `unknown` anywhere kept the queue "not hopeless" while the head was an
  interval the census had just measured to fail. Measured live minutes after
  the barren-stall fix gave the phase its first rate: **21 of 22 intervals
  `exceeds`, and the supervisor was training `jupyter-scientific-full:524288:
  655360` — ETA 42.44 h against a 2.36 h window, an 18x miss, measured and
  selected.** The lone `unknown` holding the gate open was a 2026-09-10 record
  carrying no measurement at all. Fixed by ranking on the verdict
  (`fits` < `unknown` < `exceeds`) — still a REORDER, never a filter, so every
  obligation stays eligible and an `exceeds` is selected the moment nothing
  else remains, which is also exactly when the queue is refused outright.
  **When a decision function publishes a per-item verdict, find the consumer
  that acts on it; an aggregate over the set is not that consumer.**

- **Sample count is a proxy for confidence; the MISS FACTOR is the
  measurement.** `MIN_RATE_SAMPLES_TO_REFUSE = 2` exists so a measurement
  cannot become a self-fulfilling halt, and that is right for a marginal miss.
  It is wrong for `jupyter-scientific-para4`, which missed by **434x** on a
  single sample that had itself observed 7,984 rows over 34.64 h — and still
  had to spend a full window and a rollback to "confirm" what no second
  observation could overturn. One sample now refuses when the miss exceeds
  `REFUSE_ON_ONE_SAMPLE_MISS_FACTOR` (10x), sized well above the ~22x swing of
  an INSTANTANEOUS row rate because these samples are end-to-end over 14–35 h.
  Effect measured live: 18 of 22 intervals refused on evidence already in the
  ledger, each of which would otherwise have cost one window plus a rollback.

- **Splitting the work unit cannot help, because the window is consumed by
  TRAINING HOURS and not by interval boundaries.** The obvious repair for "no
  interval fits" is smaller intervals, and on this volume it is inert: an
  admission returns no disk. The guard is re-cloned from the live brain, so the
  retired guard shares every block with it (measured guard-unique **0.00 GB**),
  and only a rollback returns bytes — which discards the interval. So
  `window_hours` is the volume's TOTAL remaining training capacity however the
  queue is cut up; splitting changes only whether those hours end up admitted
  or discarded, not how many there are. The refusal now publishes a `capacity`
  block so nobody rediscovers this. Measured 2026-09-15 at the halt:
  **2,758,116 pending rows; 2.47 h available (415.18 GB above the floor at
  167.96 GB/h over 120 samples); 46.1 h needed at 59,778 rows/h — the fastest
  rate this host has EVER measured, on a phase with nothing pending — a 18.7x
  deficit; and 9,288.4 h at each interval's own measured phase rate, a 3,760x
  deficit, with 22 of 22 intervals priced.** Publish both bounds: the generous
  one makes the argument unarguable, the measured one is what the queue costs.
  The volume was **565.18 GB free of 1023.5 GB** throughout — this is exit 91,
  not 90, and resizing a disk that is not full would fix nothing.

  The root cause of both halves is one fact already in this file: a ~363 GB
  brain on a 15.26 GB host evicts continuously, and that single pressure
  produces the 167.96 GB/h of append-only full-body rewrites AND the 260x
  spread in per-row cost (go-systems 59,778 rows/h against para4 230.5). More
  RAM is therefore the one purchase that addresses both; a larger volume buys
  only time; delta-encoded terminal updates are the architecture fix. All three
  are user decisions, which is why the halt names the arithmetic rather than
  guessing.

- **A halt BY DESIGN looks identical to a dead host, and both emitters said so.**
  Exit 91 leaves the unit `failed` with no supervisor and no wrapper — exactly
  the census every fault arm treats as catastrophic. `classify_probe` fell
  through to "no curriculum supervisor or wrapper owns terminal state
  `no_interval_fits_disk_window`", and since `cooldown_elapsed` re-triggers an
  unchanged fingerprint on a timer, that would have billed an agent wake-up
  every 1800 s forever against a state no agent can resolve;
  `admission_watchdog.faults` would have reported `supervisor_down`,
  `brain_down` and `no_admission` together. **Change both** — the third time
  they have drifted. The new `awaiting_user_decision` kind fires ONCE per
  distinct situation, with the capacity arithmetic in its fingerprint so a
  grown volume or fallen burn still re-fires; note the fingerprint is built
  inline, because `event_fingerprint` hashes a fixed identity set and would
  have made a fire-once kind permanently silent — worse than the repeating
  alarm. And it sits BELOW the disk arm: a lifecycle notice ranked above the
  fault arms outranks all of them once its state is permanent, which this one
  is until a human acts.

- **`aws ssm send-command` caps parameters plus document at 97 KB, and the
  error is deleted before you see it.** The supervisor is ~284 KB of source;
  gzip+base64 is ~90 KB, and it crossed the limit mid-session as this change
  added lines. `bootstrap_training_host.aws` runs the CLI with `check=True`
  beside `capture_output=True` — the same evidence-deleting pair this file
  already documents for `debug_eval` — so `MaxDocumentSizeExceeded` surfaced
  only as `returned non-zero exit status 254`, which names nothing. Deploys of
  this file must now CHUNK the payload (`_build_capacity_census_deploy.py`
  emits `partN.sh` + `install.sh`, digest-checked on the host before install),
  and it will keep growing, so trimming a fix to fit is not the answer.

- **`event.get("a") or event.get("b")` deletes a legitimate zero.** A
  timestamp, row or count of 0 is falsy, so that idiom silently drops the
  record from the measurement. Caught by a test here, and it is the same class
  as the `row is not None` gate that published a bare drought alarm against a
  converging replay. Use an explicit `isinstance` check (`health_event_unix`).

## Important Notes
- Always commit and push after any code changes
- Kill old processes before deploying new binary (port conflicts cause silent API thread death)
- `node_config.json` in project root has `data.enabled: false` and `wallet.prompt_on_load: false`
- Neuro pool data lives at `D:\w1z4rdv1510n-data\` — set `W1Z4RDV1510N_DATA_DIR` before launching node
- The GNU toolchain requires WinLibs PATH to be set or dlltool errors occur
- Avira AV may quarantine Rust build artifacts — exclusions are set in Windows Defender
