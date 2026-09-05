# Programming brain local operations bridge

The authoritative senior-software-engineer brain and curriculum remain on the
private AWS host. The node listens only on the host's loopback interface; this
is intentional and must not be weakened to make dashboard integration easier.

On Windows, run `scripts/aws/start_programming_brain_proxy.ps1` to idempotently
start a loopback-only relay at `http://127.0.0.1:18096`. The relay uses the
existing `FountainServer` AWS profile and the already-authorized SSM command
channel. No AWS keys, prompts, or replies are persisted in Git or logged by the
relay. The only supported routes are `/health`, `/brain/chat`, and `/chat`.

Run `scripts/aws/show_programming_brain_watch.ps1` to ensure the deterministic
watchdog exists and tail its durable activity log. Closing the tail does not
stop the watcher or AWS training. The watcher must not reinterpret quarantine
as completion, and the acceptance marker remains governed by
`PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md`.

If IAM later grants `ssm:StartSession`, a standard Session Manager port-forward
can replace the command relay. Until then, do not expose port 18095 publicly.

## The tier orchestrator: read the skip counters, not the eviction rate

`/tier_orchestrator` reports why each scanned neuron was *not* evicted.
Read these before theorising about eviction policy:

| Field | Meaning |
|---|---|
| `skipped_atom` | rejected as a byte-atom (atoms are BYTES, ~879 of them) |
| `skipped_evicted` | already asleep on disk |
| `skipped_newborn` | younger than `min_age_ticks` (suspended under emergency) |
| `skipped_score` | scored at or below `evict_threshold * pressure` |
| `pools_visited` / `pools_no_tier` / `pools_underbudget` | pool-level outcomes |

A low eviction rate has at least three causes with three different fixes:
a filter rejecting everything, a score that never qualifies, and a scan
that never reaches the fabric. The counters distinguish them; the rate
alone does not.

**Measured 2026-09-03:** `neurons_scanned` 131,806 with `skipped_atom`
130,831 (99.26%) and `skipped_score` / `skipped_newborn` /
`skipped_evicted` all **0**. The orchestrator was scanning *logical slot
indices* on a paged `.wbrain` pool, where `neurons_len()` is the logical
count (4.6M, nearly all asleep on disk) but `neuron_at()` resolves only
the RAM-resident map. It never saw a sleeping concept. Fixed by scanning
`Pool::resident_window()` / `resident_len()`.

**Logical length is not resident length.** On a paged pool they differ by
orders of magnitude. Anything that walks neurons for a memory decision
must use the resident accessors.

**`pgrep -f w1z4rd_brain_server` matches the supervisor first** — its
command line contains `--node-bin .../w1z4rd_brain_server`. That probe
reports ~23 MB RSS and looks like a non-hydrating brain. Use
`pgrep -f "release/w1z4rd_brain_server$"`.

## /stats on the brain server is NOT brain_api's h_stats

`crates/node/src/bin/brain_server.rs` includes `brain_api.rs` via
`#[path]`, but mounts only
`brain_api::brain_phase_routes_without_core(...)` and then registers its
own `/stats` from a **typed** `StatsResponse` struct (line ~519).

Consequences, both measured 2026-09-03 at the cost of three rebuild
cycles:

- Fields added to `brain_api::h_stats` never reach the wire. `/stats`
  kept returning its original nine keys through three confirmed clean
  builds and restarts, with the new symbols verifiably present in the
  binary.
- Extra keys in a `json!` there are unreachable anyway — the response is
  a fixed struct.

`/tier_orchestrator` and `/memory_residency` work because they are
**phase** routes, which `without_core` does mount. Add new diagnostics as
phase routes in `brain_api.rs`; only edit `brain_server.rs` when the
field genuinely belongs on core `/stats`, and then edit the struct.

Two probe traps that produce confident wrong answers:

- `strings -a <binary> | grep -c <json_key>` is **not** proof a build
  lacks a field. It reported 0 for `skipped_atom` on a binary that was
  serving `skipped_atom` over HTTP at that moment. Ask the endpoint.
- Cargo hard-links from its cache (link count 2). A rebuild that
  "Finished in 0.30s" and leaves the timestamp unchanged did not relink.
  `touch` the sources to force it, and check `ls -la` for a new mtime.

## A work unit sized to fill the resource window starves its own gate

Measured 2026-09-05 over 16 unattended hours: eight clean yield/recycle
cycles, seven intervals advanced, `accepted_episodes` 18,568,
`durable_next_row` past every interval end, every worker reporting
`131072 xpool pairs posted, 0 failed` — and **zero admissions**, the last
one two weeks earlier.

A 131,072-row replay pass takes ~9,000 s while the brain exhausts its
memory headroom in 2.0–2.5 h. The worker finished the rows, began the
next interval inside the same invocation, and was SIGTERMed by the memory
guard before `interval_recall` and the behavioural gate ran. The interval
was sized to consume exactly the window the gate also needed.

**The diagnostic:** ask *where the transaction died*, not how often it
failed. A starved gate and a rejecting gate both surface as
`deferred_replay_failed`, and they need opposite fixes — resize the work
unit, or repair the capability under test. The error string separates
them: `worker exited -15` died before the gate; anything else means the
gate ran and returned a verdict. The probe reports this split as
`replay_failures_before_gate` / `replay_failures_at_gate`.

**Count a name something actually writes.** This section used to say
"count `*interval_recall*` artifacts; zero means the gate never ran."
Nothing writes that name — `interval_recall` is a health-event *kind* and
a JSON *key* inside `deferred-replay-<digest>.admission.json` — so the
glob returned 0 whether the gate had run a thousand times or never at
all. Measured 2026-09-05, it reported 0 with **45** admission artifacts
and **402** rejection records in the same tree, and that vacuous 0 was
quoted back as evidence the gate had never executed. The real artifacts:

| Outcome | Artifact |
|---|---|
| gate passed | `deferred-replay-<digest>.admission.json` |
| gate ran, rejected | `deferred/<digest>/evidence/<attempt>/failure.json` |

A zero from a glob is only evidence when some non-zero could have
produced it. Check that the pattern matches a real artifact before
reading meaning into its absence.

**None of these mean training is converting:**

| Signal | Was green while nothing admitted |
|---|---|
| `accepted_episodes` rising | 18,568 |
| `durable_next_row` advancing | past every interval end |
| worker `0 failed` | every pass |
| resource yields `passed=True` | 8 of 8 |
| supervisor `state` | `deferred_replay_training` |

Only a rising `deferred_replay_admitted` count and the presence of gate
artifacts mean anything. Size every per-pass work unit to fit **inside**
the resource window with room for the verification that follows it
(`--replay-rows-per-pass`), or the verification silently never happens.

## A silent WAL reader loses the training a crash was supposed to protect

`read_framed_event` read each frame with `Read::read` and treated any short
read as a torn tail: it returned `Ok(None)`, which the caller reads as a
clean end-of-log. Replay stopped there and **reported success**, so nothing
downstream could distinguish "the log ended" from "we stopped reading it".

Measured 2026-09-05: replay stopped mid-body, so the next length prefix came
out of event payload and surfaced as

    WAL replay failed; continuing from brain.bin
    error=invalid value: integer `36`, expected variant index 0 <= i < 8

`WalEvent` has 8 variants; 36 was a label byte read as a discriminant. The
brain fell back to a checkpoint from Aug 19, that failed too, and it came up
empty — 0.08 GB resident against a 15 GB container, answering every query
with "outcome steady" at ~0.01 confidence.

**Compaction was accused and is innocent.** The first fix rewrote
`compact_after_checkpoint`, blaming `set_len` through `get_mut()` against a
seek through the `BufWriter`. Those positions never disagree:
`<BufWriter as Seek>::seek` flushes before seeking, and compaction flushes
first anyway. Both forms pass `compaction_framing_tests`, including at an
offset several 64 KiB buffers past the header. Re-deriving that story costs a
session; the writer change survives only as hardening.

**The prefix had the same defect as the body and outlived the first fix.**
The body was moved to `read_exact`; the four length bytes in front of it kept
bare `read`. The existing test could not see it — its `ChunkedReader` fills a
4-byte request in one call at `chunk` 7, so only bodies ever came back short.
At `chunk` 1 replay recovers **zero** events, including the complete one in
front of the tear. Test short reads at 1, 2 and 3 bytes, not just at a
plausible buffer size.

Verify the fix is in the process that is running, not just on disk:

```bash
PID=$(pgrep -x w1z4rd_brain_se)
grep -qa "WAL replay: torn body at tail; stopping replay" /proc/$PID/exe \
  && echo fixed || echo STALE
```

An independent framing scan is the other half — walk `brain.wal` prefix by
prefix and require it to land exactly on EOF. A file that scans clean to its
last byte has never been written unframed, whatever a comment claims.

## A pass that only checkpoints at the end has nothing to resume from

The deferred-replay resume row was recorded once per **completed** pass, so
anything that ended a pass early discarded every row it had trained and began
the interval again at row 0. Measured 2026-09-05: two supervisor restarts
inside one hour threw away 29,552 and then 1,776 already-WAL-durable
episodes. At ~13 rows/s a 49,152-row pass is an hour of billed compute.

The boundary was always available — the worker publishes `durable_next_row`
continuously under `--wal-durable`, and the supervisor already polls every two
seconds. `checkpoint_replay_resume` now writes it as the pass runs, never
lowering the recorded row and never recording past `end_row`.

This is sound **only because the WAL reader is correct**. `durable_next_row`
advances behind a WAL flush, so an unclean death recovers exactly those rows —
if replay reads them. Do not port mid-pass resume to a binary without the
reader fix above; it would resume past rows the brain no longer holds.

**A yield that won nothing is still not a verdict.** The yield path refuses to
"convert host pressure into a semantic failure", and then the no-progress
check did exactly that: with the floor already breached at pass start, the
worker was stopped before its first durable batch and the interval — untouched
and unjudged — was marked failed. The recycle that follows a yield frees the
window (measured 2.99 GB → 14.66 GB), so the retry gets room this pass never
had. Tolerated up to `MAX_BARREN_REPLAY_YIELDS`, because a recycle that stops
buying a window is a real fault that must surface.

## A retention suite is only as good as the path that trained it

`/brain/observe` + `/brain/tick` cannot form a binding for a long response:
recall via the observe path is exact below ~80 bytes and empty above it. A
suite whose responses exceed that trains rows nothing can retrieve, then fails
its own paraphrase check while reporting `trained` at full marks.

That signature — **`trained` perfect, `paraphrase` empty** — is *consistent
with* a training-path problem, but it does not prove one. It is also what a
degraded brain produces, and telling those apart requires re-measuring, not
reasoning.

Worked example, 2026-09-05. `programming_typescript_enterprise.py` still
trained through the observe path with responses of 379 B, 560 B and 799 B, and
`optimistic_store` recorded `trained 3/3, paraphrase 2/3` on every gate from
2026-09-03 10:13 to 2026-09-04 18:32 — 8 intervals rejected. The training path
was the obvious culprit and the wrong one: run against the recovered brain,
those same observe-path bindings score `trained 3/3, paraphrase 3/3, oov 3/3`
with no re-seed. The rejections sat inside the window when WAL corruption had
left the brain empty, and they stopped when it was restored.

**Re-measure a suite against the current brain before repairing it.** A gate
artifact records what was true of the brain that ran it. Attributing an old
rejection to a code path, then "fixing" that path, produces a change that
cannot be shown to have done anything — and buries the real cause.

Two facts worth keeping anyway: `/brain/pretrain_binding` does not depend on a
response being short enough for the observe path to bind, and the supervisor
only ever invokes these suites with `--no-train`, so any change to how a suite
trains needs a one-off re-seed before a gate can see it.

Measured after the re-seed on 2026-09-05, `--no-train` on the live brain:

| | before | after |
|---|---|---|
| `trained` | 3/3 | 3/3 |
| `paraphrase` | **2/3** | **3/3** |
| `oov_honesty` | 3/3 | 3/3 |
| exit | 1 | 0 |

Cost: +278 neurons, +278 concepts, +275 bindings. OOV honesty held at 3/3,
so this bought paraphrase reach without trading away abstention — check that
every time, because widening a route to fix paraphrase has broken OOV honesty
here before.

**A re-seed is an unadmitted mutation, so a rollback silently undoes it.**
Re-seeding necessarily happens outside any replay transaction, but the
last-good guard was created *before* it, and `restore_rejected_deferred_replay`
restores that guard wholesale. So if the next interval is rejected for any
reason, the re-seeded bindings go with it and the gate returns to failing on
the very case that was just fixed — with nothing in the logs naming the
re-seed, because losing it is not an event. The re-seed only becomes permanent
when an interval is **admitted** and `accept_last_good_guard` releases the
guard. After any `deferred_replay_failed` that follows a re-seed, re-run the
suite with `--no-train` before assuming the repair still holds; if its
paraphrase count dropped back, re-seed again rather than re-diagnosing.

## Deploying a fix is not applying it

Python compiles a module once, at import. A source file written **after** a
process started is not in that process and never will be, no matter how many
times you read it back and confirm the fix is there.

Measured 2026-09-05. The two fixes above — mid-pass resume, and not scoring a
resource yield as a semantic failure — were committed, then copied to the host
byte-identical to `HEAD` (203,558 bytes, matching the 208,237-byte working
copy exactly once its 4,679 CRLFs are stripped). Nothing restarted the unit.
The supervisor kept running the previous module for another 850 s, so on the
host the old branch was still live:

```python
worker = run_deferred_replay_worker(..., stdout, stderr)   # no resume_row
if worker.returncode != 0:                                  # no yield check
    raise RuntimeError(f"deferred replay worker exited {worker.returncode}")
```

The memory guard stops the worker with `SIGTERM`, so **every** yield arrives
here as `-15` and is raised as a semantic rejection. The ledger shows the
signature plainly — three seconds apart, four times in twelve hours:

```
settled_node_memory_recycle      age 24826   available 2.99 GB -> 14.66 GB
deferred_replay_resource_yield   age 24826   para4:786432:917504
deferred_replay_failed           age 24823   "worker exited -15"
```

19 yields, 19 failures, 288 `deferred_replay_failed` in total, 13 passes on
one interval each restarting at row 0, and nothing admitted for 349 hours on a
billed host.

What makes this expensive is that every check pointed the wrong way. Grepping
the deployed file for `def checkpoint_replay_resume` returned true. The unit
was `active`, the brain answered `/health`, the tick advanced, the worker held
a live PID and the progress file was 0 s old. All of it was true and none of
it was the question, which is whether the **process** is the code.

So measure the process, not the artifact. `admission_watchdog.py` now reports
`stale_code_lag` — the supervisor source mtime minus the running process start
time, read from `/proc/<pid>` — and faults above 300 s, wide enough to ignore
a deploy that restarts promptly and far inside the ~4,900 s pass it otherwise
costs. The confirmation after a restart is not that the unit came back up; it
is that `deferred-replay-<digest>.resume.json` **exists and advances**. Before
the restart no such file existed anywhere in the runtime; ninety seconds
after, it read `durable_next_row: 544` against a progress file at 552.

This generalises past this repo. A remote fix has two failure points — did the
bytes land, and did anything reload them — and only the first one leaves an
artifact you can grep.
