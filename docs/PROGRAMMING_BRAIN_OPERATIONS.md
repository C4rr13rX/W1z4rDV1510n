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
