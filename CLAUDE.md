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
  is in `<phase>.enterprise-gate.json` next to it, under `results[].name`.
  Measured 2026-09-09: 127 gate runs, exactly one pass ever, `polyglot`
  failing every time on a single row — `javascript_go_order_workers`
  canonical composing `ledger.go` where the prompt asked for a deduplicator,
  so `go_deduplication` died on `stat dedup.go: no such file`. One row of one
  suite held every admission for over four days.
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
