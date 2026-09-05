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

`/brain/observe` + `/brain/tick` binds a short response reliably and a long one
unreliably; the often-quoted "exact below ~80 bytes, empty above it" is too
strong, and measurement below contradicts it. A suite whose responses exceed
that size *may* train rows nothing can retrieve, then fail its own paraphrase
check while reporting `trained` at full marks.

That signature — **`trained` perfect, `paraphrase` empty** — is *consistent
with* a training-path problem, but it does not prove one. It is also what a
degraded brain produces, and telling those apart requires re-measuring, not
reasoning.

Worked example, 2026-09-05. `programming_typescript_enterprise.py` still
trained through the observe path with responses of 379 B, 560 B and 799 B, and
`optimistic_store` recorded `trained 3/3, paraphrase 2/3` on every gate from
2026-09-03 10:13 to 2026-09-04 18:32 — 8 intervals rejected. The training path
was the obvious culprit, and the rejections sat inside the window when WAL
corruption had left the brain empty — they stopped when it was restored.

An earlier revision of this section claimed those same observe-path bindings
scored `paraphrase 3/3` "with no re-seed", and drew the causal conclusion from
it. That measurement was contaminated and does not support the claim: the
brain server ran unrestarted from 14:06:12 to 14:33:59 UTC with no rollback and
no health event, so the re-seed recorded in this same section — which took
`paraphrase` from 2/3 to 3/3 — had *already* been applied when the "no re-seed"
reading was taken. Both readings cannot describe one brain state, and the table
below is the one with a mutation behind it.

The exclusion check used to rule this out — "the newest typescript gate
artifact is 2026-09-04, so nothing else ran the suite" — cannot detect what it
claims. A re-seed run and an ad-hoc `--no-train` verification write wherever
`--output` points, not to `<phase>.typescript-gate.json`; two verification runs
at 14:29:09 and 14:29:56 that day left `typescript-reseed-verify.json` and
`typescript-noTrain-verify.json` and were invisible to it. **Prove a suite was
untouched from process and mtime evidence across the whole runtime directory,
never from one artifact name** — and assume a concurrent agent is writing to
the same brain, because on this project one usually is.

The conclusion still stands, on evidence that does not depend on that reading.
Two long-response suites were deliberately *not* re-seeded, so they still carry
their original bindings, and both pass on the recovered brain:
`programming_platform_eval.py` (1042 B responses) at `trained 4/4,
paraphrase 4/4, oov 3/3`, and `programming_cross_language_transfer.py` (903 B)
at `canonical 4/4, heldout 4/4, oov 2/2`. Long responses do retrieve here. The
degraded brain was the cause; the training path was a symptom.

**Re-measure a suite against the current brain before repairing it.** A gate
artifact records what was true of the brain that ran it. Attributing an old
rejection to a code path, then "fixing" that path, produces a change that
cannot be shown to have done anything — and buries the real cause. The
symmetric error is just as expensive: clearing a code path on a measurement
taken *after* you already changed the brain underneath it.

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

Two later sections qualify the remedy rather than the diagnosis. **Restarting
the supervisor mid-replay rolls the interval back** — check
`deferred-replay-active.json` before you restart anything to apply a fix. And
**the watcher is subject to every rule it enforces**: it ran nineteen minutes
behind its own repaired probe and alarmed on the pre-fix reading.

### Confirmed in production, 2026-09-05

With the module actually loaded — supervisor source mtime 3.19 h, process
uptime 3.18 h, so `stale_code_lag` is ~30 s — the same event sequence now
terminates differently:

```
settled_node_memory_recycle      age 1.6 min   available 3.00 GB -> 14.59 GB
deferred_replay_resource_yield   age 1.6 min   jupyter-scientific-full:0:131072
                                 (no deferred_replay_failed follows)
```

One yield, zero failures, against the previous 19-for-19. The interval was not
rolled back: the worker relaunched with `--start-row 120880
--durable-start-row 120880`, the exact row the pre-yield pass had made durable,
and carried on from there.

The discriminator is worth restating because two of the obvious metrics stay
silent here. `deferred_replay_failed` was 288 in the payload and 0 since this
process started — a cumulative counter cannot tell you a fault has stopped, so
bound it by process start. `durable_next_row` rises whether or not the pass
resumed, so it cannot tell you either. **`accepted_episodes` is the tell**: it
read 704 ninety-eight seconds after the recycle and 8,944 thirteen minutes
later, which is a pass converging, not one re-reading rows it had already
trained.

## The named failure is not the failure population

The watchdog reports `last_failure`. It is one row. Repairing it and declaring
the queue fixed assumes the most recent failure is the representative one,
and on 2026-09-05 that assumption was wrong by a factor of fifteen.

`last_failure` read `deferred replay worker exited -15`, the yield
misattribution the section above describes. Classifying all 288
`deferred_replay_failed` events by their error text instead gives:

| Count | Cause |
|---:|---|
| 119 | `gate command failed (1)` — enterprise retention |
| 99 | semantic recall |
| 34 | worker exit, other |
| 19 | `exited -15` — the yield misattribution |
| 8 | `gate command failed (1)` — typescript route |

The SIGTERM story was real, fully diagnosed, and **6.6 %** of the population.
Four fifths of the queue was rejected by the enterprise gate, whose own stdout
names the four suites responsible — `platform`, `cross_project`, `composition`,
`semantic_stress`, all with `infrastructure_failure: false`.

The second half of the lesson is why that table still did not justify repairing
those four suites. Those 119 events accumulated over fourteen days; the gate
artifact written that same afternoon read:

```
jupyter-scientific-full.enterprise-gate.json   passed=True   (12/12 suites)
jupyter-scientific-full.completion-gate.json   passed=True
```

The suites had already been repaired. A count aggregated over a fortnight
describes the brain that produced it, not the brain on disk now, and
`hours_since_admission: 351` is consistent with both "still broken" and "was
broken, fixed at hour 350". Only a fresh artifact separates them.

So: classify the whole population before repairing anything, then re-measure
the dominant cause against the current brain. History says where to look; only
a current measurement says whether to act. This is the same discipline
`verify_before_repairing_a_suite` records, arrived at from the opposite
direction — there the confident story was of a live defect, here of a live
defect that had already been fixed.

One transport note, because it silently produces the wrong table: events in
`curriculum-health.jsonl` carry `updated_unix`, not `unix`. A filter on
`r.get("unix")` matches nothing and reports a quiet, empty window regardless of
what happened in it — `vacuous_zero_signals` again, in a new key.

## A rising row counter does not mean the interval is converging

The two sections above establish that the yield misattribution was real and
that it had already been fixed. Neither establishes that the replay now makes
progress, and the check that looks like it would — "is `durable_next_row`
going up?" — cannot, because it goes up just as steadily on a pass that
restarts the same prefix forever.

That distinction is the whole failure. A 131,072-row interval is trained in
capped passes of `--replay-rows-per-pass` (49,152). Between passes the
supervisor settles the brain and recycles the node process, and the next pass
is told where to start by `deferred_replay_resume_row`. That function returns
the interval's `start_row` — discarding the previous pass's work — whenever
the resume record's `guard_identity` no longer matches the live one:

```python
if record.get("guard_identity") != guard_identity:
    return start
```

`guard_identity` is `phase:created_unix:tick` read from
`brain/brain.last-good.json`. The recycle between passes is exactly the moment
a new last-good guard could be published, and if it were, every pass would
restart at row 0, the interval would never reach its gate, and every external
signal — live PID, advancing tick, fresh progress file, rising
`durable_next_row`, `/health` answering — would look correct while the host
billed indefinitely. Thirteen passes on one interval had already done this.

**The discriminator is `accepted_episodes`, not `durable_next_row`.** The
progress file carries both, and they mean different things: `durable_next_row`
is the absolute row now WAL-durable, while `accepted_episodes` counts only
what *this* pass posted. So

```
pass_start_row = durable_next_row - accepted_episodes
```

and a pass that resumed correctly has a non-zero one. Measured 2026-09-05
across a real boundary:

```
before   durable_next_row 43384   accepted_episodes 43384   -> pass began at 0
after    durable_next_row 60832   accepted_episodes 11680   -> pass began at 49152
```

49,152 is the cap exactly. The `guard_identity` string was byte-identical
either side of the recycle (`jupyter-scientific-full:1788621147.7185087:4148389`)
while the brain's own tick advanced 4,148,389 -> 4,197,541, which is the
combination that has to hold: the guard stable, the brain still learning.

The full boundary, worth recognising because it is what a healthy one looks
like — a pass ending at its **cap** rather than on a yield:

```
t+120s  row=49152  state=deferred_replay_training   memGB=4.63
t+160s  row=49152  state=resource_node_recycled     memGB=14.35   old_pid -> replacement_pid
t+180s  row=49160  state=resource_node_recycled     memGB=6.28
```

`resident_terminals: 0` and `total_neurons` rising across the recycle confirm
the settle serialized every neuron without dropping learned topology.

Two cautions on reading this as success. First, the supervisor writes its
status only at state transitions, so `curriculum-supervisor.status.json` sat
at `resource_node_recycled` and 955 s stale while the replay ran normally
underneath it; the progress file, 0.1 s old, is the heartbeat. Second, and
more important, **a converging interval is not an admitted one**. At the time
of this measurement the ledger still read 28 deferred / 59 resolved and no
interval had resolved during the run. Convergence across a pass boundary is
necessary for admission and is not evidence of it. The count that closes this
out is `resolved` rising — the same rule CLAUDE.md states for the curriculum
as a whole, applied one level down.

**It closed.** `jupyter-scientific-full:0:131072` — the interval converging in
the measurement above — admitted at 14:40 the same day:

```
14:24  deferred=28  resolved=59  last_admission=264627s   (73.5 h)
14:42  deferred=27  resolved=60  last_admission=162s
```

So the whole chain held end to end on one interval: a resource yield paused
the worker and did **not** roll it back, the pass resumed at row 120880 rather
than at 0, it ran to its cap, and the enterprise gate admitted it. That is the
first admission since the yield-accounting repair and it is the evidence the
repair converts, which neither a live PID, an advancing tick, nor a rising
`durable_next_row` could have supplied.

Note what the alarm that prompted the check had said: `hours_since_admission:
353.2`, against a ledger reading 73.5 h at the moment it fired and 0.05 h
eighteen minutes later. Both numbers came from the same host in the same hour.
The rule stands — read the ledger, and treat a drought figure from a watcher
whose payload is missing `hours_since_admission_event`, `gate_rejections`,
`replay_failures_at_gate` and `replay_failures_before_gate` as a statement
about the watcher rather than about the curriculum.

### Sustained, seven yields later: what the repaired loop actually costs

The confirmation above is one interval. Measured later the same day against
supervisor PID 1921673 (started 1788641111), at two points 1.45 h and 1.8 h
into that generation:

```
events since supervisor start, at 1.45 h:  recycle 4  yield 4  failed 0
events since supervisor start, at 1.80 h:  recycle 5  yield 5  failed 0

all yields in a 3 h window (spans the PREVIOUS generation too), ages in s:
  10015  8363  6729 | 4433  2952  1523  161      all passed=True
  seconds between:  1652  1634  2296  1481  1429  1362
```

Read the split carefully, because it is easy to get wrong and it is the whole
strength of the evidence. Seven yields fall in a 3-hour window, but the
generation was only 5,214 s old at that moment, so **three of them belong to
the previous supervisor** — a query windowed on wall-clock time silently
crosses a restart boundary that a query windowed on process start does not.
The claim that holds is: five yields with zero `deferred_replay_failed` inside
one generation, and zero failures anywhere in the 3-hour window across both.
Two intervals resolved inside the window
(`…:0:131072` at 1788633431, `…:131072:201344` at 1788641100), and the running
worker's own argv is the direct evidence that a yield no longer discards a
pass:

```
drive_corpora_brain … --start-row 246576 --limit-rows 15568
                      --durable-start-row 246576
```

The throughput this buys, which is the number to plan with:

| Measure | Value |
|---|---|
| rows/s while the worker runs | 9.72 (60 s sample) |
| rows/s averaged over yields and recycles | 8.4 (46,928 rows / 5,586 s) |
| duty cycle | ~86% |
| yield period | ~27 min |

So the recycle overhead is **14%, not a stall** — worth knowing, because a
yield every 27 minutes reads alarmingly in an event feed and the instinct is
to go looking for headroom. There is none to find: the host is 15.26 GB with
`swap_total_gb 0.0`, the brain settles around 11.45 GB, and the 3.0 GB floor
is what triggers the yield. **Do not add swap to buy a longer period.** That
converts a 14% bounded cost into the `crypto_node_memory_wedge` failure, where
`/health` answers for days while every write route deadlocks on paging.

With 8.4 rows/s sustained, the remaining quarantine is countable rather than
open-ended — 2,782,556 unresolved rows across 26 intervals is ≈3.8 days of
replay plus per-interval gate time:

```
jupyter-scientific-para4     17 intervals  2,105,136 rows
jupyter-scientific-full       4 intervals    454,016 rows
jupyter-scientific-partial    2 intervals    206,948 rows
metamathqa-domain-safe        1 interval      16,384 rows
webstack-units / -projects    2 intervals         72 rows
```

That arithmetic is the useful form of "is it healthy". `quarantine_ready` with
`forward_remaining_rows: 0` is the expected state at this stage and says
nothing about convergence; seven yields with zero failures and a resume row
advancing across them does.

## The failure ledger is older than the process that will be blamed for it

The section on deploying a fix warns that the running process can be older
than the file. The watchdog payload creates the mirror-image trap, and it is
easier to fall into because every field in it is true.

Woken 2026-09-05 16:19 on `quarantine_ready`, the payload led with
`hours_since_admission: 351.2` and

```
last_failure: "deferred replay worker exited -15; stderr=...-408a84ade3bb96e1.stderr.log"
deferred_replay_failed: 288      deferred_replay_resource_yield: 19
```

which is the yield misattribution exactly. The obvious reading — the fix did
not work, go and re-debug it — is wrong, and the two checks that show why cost
about a minute between them.

**Check one: is the fix in the process?** Not in the file, in the process.

```
sha256 (host)  07790e16...9486a  /srv/wizard/project/scripts/programming_curriculum_supervisor.py
sha256 (local) 07790e16...9486a  HEAD
file mtime     14:57      process start  14:58:02      lag  ~ -60 s
```

Byte-identical to `HEAD`, and started *after* it was written, so the guard at
line 3727 (`if worker.returncode != 0 and not yielded:`) is live.

**Check two: when did the failures happen?** Ages, against a process 1.37 h
old at observation:

```
23.51h  21.81h  20.07h  18.07h  16.21h  13.75h  11.26h  8.73h  6.45h  3.64h
```

Every one of the ten most recent failures predates the restart; the newest is
2.3 h older than the process. The ledger is append-only and `last_failure`
carries no notion of which binary produced it, so a repaired fault keeps being
reported as the current one until something new is written over it. The
absence of a *post-restart* failure is the signal, and it is invisible unless
you compare timestamps against process start.

The confirmation is positive, not just an absence: at 16:17 a
`settled_node_memory_recycle` completed with **no** `deferred_replay_failed`
three seconds behind it — the first recycle in 20 h not followed by a
rollback — while `durable_next_row` ran 45,960 -> 50,088 -> 64,000 through it.

So before repairing anything a watchdog names, timestamp it against the
running process. `last_failure` answers "what failed most recently", never
"what is failing now".

## The admission gate is read-only, so measure it before it fires

`run_deferred_replays` admits an interval only if `run_completion_gate`
passes, and a single failing stage rolls back the entire interval — up to
131,072 rows and hours of billed compute. Waiting for the gate to discover a
blocker is therefore the most expensive possible way to find one.

It is also unnecessary. Every stage of that gate is a read-only probe of the
live brain, so the whole chain can be run against the brain *while the replay
is still training*, and it answers "will this admit?" hours early. Measured
2026-09-05 with the interval at row 64,000 of 131,072, ~1.6 h before its gate:

| stage | command | result |
|---|---|---|
| foundation | `programming_brain_eval.py --details` | toddler 32/32, k12 16/16, oov 3/3 |
| code | `programming_code_eval.py --details` | trained 5/5, novel paraphrase 5/5 |
| typescript | `programming_typescript_enterprise.py --no-train` | 3/3, 3/3, oov 3/3, exit 0 |
| enterprise | `<phase>.enterprise-gate.json` | 12/12 suites |

Only `interval_recall` cannot be pre-run, because it samples rows the replay
has not posted yet.

Two things make this safe rather than another mutation of the thing being
measured. The eval scripts take `--no-train`/`--details` and do not observe;
the enterprise report carries `tick_before`, `tick_after` and
`structure_unchanged`, and the run above recorded `tick_delta: 0` with
`structure_unchanged: true`. And the enterprise gate need not even be re-run
if a recent artifact exists — reading the 15:11 report cost nothing where
re-running it is budgeted at four hours.

Worth recording separately: that report is **12/12**. `enterprise_gate_confirmed`
documents 6, 5, 7, 8, 7, 6, 6, 8 of 12 across consecutive runs on one brain,
which is what the confirm-a-failure-once mechanism was built for. One clean
sweep is not proof the flicker is gone, but it is the first 12/12 in the
ledger and it is the reason the next gate is expected to convert.

The prediction this supports is bounded, and the bound is the rule CLAUDE.md
states: every gate stage passing means the gate is *expected* to admit. It is
not an admission. The measurement that closes it out is `hours_since_admission`
falling and the `resolved` count rising.

## The watcher is subject to every rule it enforces

`admission_watchdog.py` reports `stale_code_lag` for the supervisor because
landing bytes and reloading them are different questions. The watcher exempted
itself from that check, and the exemption produced a false alarm whose text was
the exact inverse of the truth.

The `gate_artifacts` probe was repaired at 10:06 on 2026-09-05 — it had counted
`*interval_recall*`, a name nothing writes. The watcher process had started at
09:46. Nineteen minutes older than its own fix, it kept counting with the
vacuous glob and woke a session with:

```
fix_required: admission gate has never produced an artifact across 19 resource
cycles: the gate is not running, so no interval can ever admit
```

Running the repaired probe against the same host in the same minute:

| field | stale watcher | repaired probe |
|---|---:|---:|
| `gate_artifacts` | 0 | **45** |
| `gate_rejections` | *absent* | **402** |
| `replay_failures_at_gate` | *absent* | **235** |
| `replay_failures_before_gate` | *absent* | **53** |

The missing fields are the tell, and they are cheaper to read than the numbers.
A payload lacking keys the current probe unconditionally emits was produced by
older code, whatever the file on disk says. Check the shape of a payload before
you argue with its values.

The watcher now re-execs when its own source — or the SSM transport it imports
— is newer than the process that compiled it. **Re-exec, not exit:** the
`WizardVisionProgrammingBrainCodexWatch` scheduled task is *Disabled*, so
nothing would restart it, and a watcher that exits to be correct supervises
nothing. It runs at the top of the poll loop with no Claude invocation in
flight, so it cannot abandon a running session.

Corollary for a session woken by an alarm: **you cannot restart the watcher
from inside the session it spawned.** Its activity log is streaming your own
tool calls. That is why this is self-healing rather than an operator step.

### Self-healing does not reach backwards

The re-exec landed at 13:03. The watcher process had started at 09:46. A
process cannot run a repair that did not exist when it compiled its own
source, so the fix for stale watchers could not fix the stale watcher — and
the next wake-up, at 14:07, still carried pre-fix numbers:

| field | that wake-up's payload | ledger, same minute |
|---|---:|---:|
| `hours_since_admission` | 353.2 | **73.3** |
| `gate_artifacts` | 0 | populated |

Both were already repaired on disk. `git log -1 --format=%ad <file>` against
the watcher's start time is the check, and it is worth running before
believing any number in an alarm payload.

**Any self-healing mechanism needs exactly one manual application to the
generation that predates it.** Add the guard and the process still running
without it is the last one that will ever need the operator; skip that step
and it is every one after, because the process that would adopt the fix is
the one that cannot.

Restarting it must be deferred past the end of the invoked session, since
`invoke_claude` blocks reading the agent's stdout and killing the parent
breaks that pipe mid-turn. A detached helper that waits for the agent process
to exit, stops the watcher, relaunches it with the same argv and writes the
outcome to `activity.log` does this without a window where nothing supervises
the host. Verify by the `WATCHER START` line, not by the absence of an error.

## The status file is not the heartbeat

The supervisor publishes `curriculum-supervisor.status.json` only at state
transitions. A 49,152-row replay pass at ~13 rows/s is therefore ~3,900 s of
deliberate silence, against a watcher alarm set at 1,800 s.

Measured 2026-09-05: `status_age 2630s` — alarming — on a host whose worker
progress file was **2.3 s old** and advancing at **13.1 rows/s**. The status
file records the last thing that changed; `deferred-replay-<digest>.progress.json`
is rewritten every batch and carries `durable_next_row`. Liveness is the
heartbeat's answer. Whether that liveness *converges* is a different question,
and the admission-drought check already answers it. Both must be stale before
control is actually gone.

## The admitted event was written after the part that can die

`edadb33` made the watcher read the admission drought from the ledger. This is
why the two ever diverged, and the divergence was not small: **60 ledger
resolves against 21 `deferred_replay_admitted` events.**

The commit sequence appended the event *last* — behind `accept_last_good_guard`,
two unlinks and `prune_resolved_deferred_bases`, which deletes multi-gigabyte
`.wbrain` bases and is by far the longest, most interruptible stretch of the
transaction. Anything that ended the supervisor in that window left the
interval resolved in the ledger with a `passed: true` artifact on disk and no
event. `recover_interrupted_deferred_replay` then committed it on restart and
did not write one either — that branch exists *because* the supervisor died
after publishing the artifact, so it is the last place the event can still be
written.

Six intervals were admitted between 235.9 h and 71.8 h ago, each with an
artifact whose own `updated_unix` matches its resolve, and none logged an
event. So `hours_since_admission` read **351.7 h** where the true drought was
**71.8 h**, and a session was billed to repair a curriculum that had admitted
six times inside the window the metric called dead.

The event now sits beside the ledger resolve, before the cleanup, and the
recovery path writes one too. Note the shape of the bug: the metric is
*monotone* once it detaches. It can only ever get staler, so the same defect
that invented a stall would equally have hidden a real one.

**Announce a commit at the commit, never after the cleanup.**

## Restarting the supervisor mid-replay rolls the interval back

"Deploying a fix is not applying it" ends by telling you to restart the unit.
During a deferred replay that instruction is expensive, and nothing above says
so.

`run_deferred_replays` publishes `deferred-replay-active.json` with
`state: "training"` before the first pass. On startup
`recover_interrupted_deferred_replay` reads that marker, and any state other
than `admitted` takes the else branch:

```python
restore_rejected_deferred_replay(
    args, runtime, phase, event,
    "interrupted deferred replay rolled back before retry",
)
```

That discards every row the interval has trained. Measured 2026-09-05 with the
marker 6,762 s old and `durable_next_row` at 78,168 of 131,072: a restart to
pick up an observability fix would have thrown away 78,168 rows — about two
hours of billed compute — to deploy a change that writes one extra log line.

So before restarting the supervisor, read `deferred-replay-active.json`:

| marker | cost of restarting |
|---|---|
| absent | none; restart freely |
| `state: admitted` | none; the recovery path commits it |
| `state: training` | **the whole interval**, back to `start_row` |

A correctness or throughput fix can still be worth that. An observability fix
is not. Deploy it and let the next natural boundary load it — and say plainly
in the handover that the bytes are on the host and the process has *not*
reloaded them, because that is exactly the state the earlier section warns is
invisible to a grep.

## Sample throughput over minutes, or read a transient as a collapse

`durable_next_row` advances in quantised jumps behind the WAL flush, not
smoothly with training. Sampled every 20 s it moved exactly +16 rows a tick,
giving **0.799 rows/s** against a healthy 12.5 — a 15x collapse, with a
`batch_seconds_ema` of 4.9 s agreeing with it.

Both numbers were real and the conclusion was wrong. Twelve consecutive 30 s
windows minutes later:

```
rows/s  13.07 12.81 13.73 13.78 13.02 13.78 14.57 13.25 12.55 12.51 13.29 13.60
ema      0.53  0.68  0.50  0.56  0.62  0.48  0.60  0.55  0.72  0.63  0.55  0.59
```

Steady at **13.1 rows/s**, inside the 0.61-0.76 ema band of every interval that
has ever admitted. The brain tick advanced at 13.1/s alongside it, which is the
independent confirmation: rows and ticks moving together cannot both be a
flushing artifact.

The supporting evidence had also been misread. `cpu_frac 0.888` with
`majflt_s 0.0` and 0.96 MB/s of reads says **CPU-bound, not paging** — on a
brain reporting `resident_terminals: 0` against a 309 GB `.wbrain` the tempting
story is thrashing, and the counters refuse it.

This is the "sample repeatedly; one probe is not verification" rule in
`CLAUDE.md` costing real analysis time. An 80-second window is one probe
wearing a disguise. Before attributing a slowdown to architecture, hold the
measurement for five minutes and check whether an independent counter agrees.

## `pgrep -f w1z4rd_brain_server` matches the supervisor, not only the brain

A probe that measures "the brain" by taking the first PID out of
`pgrep -f w1z4rd_brain_server` measures the **supervisor**. The supervisor's
own command line carries `--node-bin
/srv/wizard/project/target/release/w1z4rd_brain_server`, so `-f` — which
matches the full argument list — matches it too, and it usually sorts first
because it started earlier.

The failure mode is not a missing number, it is a plausible wrong one.
Measured 2026-09-05: the probe reported `VmRSS 30444 kB` for the brain. Thirty
megabytes against a multi-gigabyte store is the exact signature of
`brain_server_not_hydrating` — the documented catastrophic case where the
server comes up holding nothing, every recall returns empty, and no gate can
pass. On that reading the correct response is to stop training and
investigate. The real brain was `1925586`, resident at **11.45 GB**, with
`/stats` reporting 4.79M neurons and 426M terminals and `/health` returning
`ok`.

Two rules, because the first one alone is not enough:

- **Anchor on the binary path, not the binary name.** `pgrep -f
  'target/release/w1z4rd_brain_server'` still matches both, since the
  supervisor's `--node-bin` is that same path. What actually discriminates is
  matching the process's `comm` or argv[0] rather than its whole command line
  — or simply listing `ps -eo pid,rss,args --sort=-rss` and reading which row
  is the server. A one-line probe that cannot distinguish a 30 MB Python
  supervisor from an 11 GB Rust server is not measuring what its variable is
  named.
- **A catastrophic reading is a reason to re-measure, not to act.** This
  repository's expensive mistakes are mostly confident stories built on one
  probe. The cheap confirmation here — one `ps` and one `curl /stats` — cost
  a single SSM round trip and turned a shutdown-grade conclusion into a probe
  bug. Any number that would justify halting training earns that round trip
  first.

The general shape is the same as `vacuous_zero_signals`: a query that cannot
distinguish two cases reports one of them forever, and the danger is highest
when the answer it happens to give is the alarming one.

## Two agents share one index

A watchdog wake-up does not mean you are alone in the repository — on this
project a second session usually is. `git add` writes to a *shared* index, so
staging only your own paths does not protect them: a concurrent `git commit -a`,
or `git commit` with no pathspec, sweeps whatever you have staged into a commit
whose message describes something else. That happened on 2026-09-05; 106 lines
of an admission fix landed inside a commit about obstacle-course duplicate
guards.

### Telling your red from theirs

A concurrent session authoring a family leaves the suite red in a way that
looks like your own breakage. On 2026-09-05 a baseline run showed 32 failures
before a line had been written this session; forty minutes later a different
20 failures, in a different family, on a tree whose HEAD had moved twice.

Both were a second session mid-family: tasks added to a family module, with
their `REFERENCES`/`MUTATIONS` entries not yet written, so both directions of
every new task fail together. The signature is specific and worth recognising
— **failures arrive in contiguous id runs within one family, exactly two per
task id** (`test_reference_solution_passes_its_validator` and
`test_a_broken_solution_fails_its_validator`), and every structural test still
passes. A defect of your own does not distribute itself that way.

What that implies for the working loop:

- Take the baseline *before* editing, and group failures by family. A red
  suite you did not cause is information about who else is working, not a
  thing to fix — repairing it races their writes, and the entries you would
  add are the ones they are about to add.
- Verify your own work family-scoped (`pytest -k <family>`), because the
  whole-suite verdict belongs partly to somebody else.
- Before committing, check that the shared files carry only your additions:
  `git diff --unified=0 tests/obstacle_references.py` filtered to the
  assignment lines names every task id the commit would capture. That is what
  distinguishes "my two files" from "my two files plus half of theirs", and it
  is the check that would have prevented `fc050a8`.

Use `git commit -m ... -- <paths>`, which commits exactly those paths and
ignores the rest of the index. Do not rewrite the shared branch to tidy the
attribution afterwards — the other session has it checked out.

One more, because knowing about a trap does not stop you setting it: a throwaway
probe in this session globbed `*marker*` for the transaction marker, which is
named `deferred-replay-active.json`, and reported `[]`. The conclusion drawn
from that empty list — that restarting the supervisor was safe — was the
opposite of the truth, and the section above exists because the second probe
was done by name. Verify a pattern can match something before believing it
matched nothing, in ad-hoc probes as much as in committed ones.
