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
