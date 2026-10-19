# Programming brain verification — 2026-10-19

## Verified in this checkout

- `cargo test -p w1z4rd-brain --test teaching_benchmark -- --test-threads=1`
  passed all nine tests (none ignored). This is a local teaching regression
  result, not evidence that the production brain or enterprise course passes.
- The four local commits through `2f944dc` were pushed to `origin/main`.
- The acceptance requirements remain those in
  `PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md`; no completion marker was issued.

## Production verification blocked

A read-only EC2 `describe-instances` request using profile `FountainServer`
and instance `i-0d7a6deeb0ead2dfc` returned AWS `AuthFailure`. Restore that
profile's authorized credentials before attempting deployment or recovery.
Do not infer current AWS liveness from local saved JSON.

The newer saved watcher snapshot under `runtime/programming-brain-watch`
reports `no_interval_fits_disk_window`, 2,758,116 deferred rows, and no
supervisor, wrapper, or worker. This is historical evidence only. The older
`programming-brain-codex-watch` snapshot is also stale. Neither establishes
current accepted totals, available storage, or whether a service is running.

## Next production checks once credentials work

1. Inspect live service/process ownership, progress timestamps, disk and RAM,
   deployed commit, and the exact unresolved quarantine ledger.
2. Reconcile admitted and pending intervals against the authoritative corpus;
   include foundational seed requirements rather than treating forward
   harvesting as qualification.
3. If the disk-window failure persists, measure candidate/checkpoint growth
   and bounded transaction costs before resuming; do not raise resource
   ceilings, discard quarantine, or delete production brain data to hide it.
4. Resume only through the existing transactional admission/replay pipeline.
   Preserve neuron-scoped paging, rollback, and protected retention.
5. Continue the held-out 1,000-task course and independently verified Robot
   World integration. Local nine-test success does not waive any of these.
