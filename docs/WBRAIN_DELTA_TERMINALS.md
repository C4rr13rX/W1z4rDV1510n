# Delta-encoded terminal appends

## What this is for

`append_record` writes a WHOLE neuron body on every eviction. On this host the
hot neurons are single-byte atoms carrying ~4.14 M terminals each, so one
eviction of one atom appends ~82 MB to record a few KB of new connections. The
`.wbrain` store is append-only and its compactor reclaims superseded bodies
only when it runs, so that 82 MB is permanent growth until then.

Measured burn: **167.96 GB/h** over 120 samples, essentially all of it this
path. Stopping the training worker took the volume to **-0.0 GB/h**, so the
attribution needs no argument.

## What it is NOT

**This does not make the curriculum finish.** A byte-weighted census over a
103 GB window split the rewrites in two:

| share | shape | delta helps? |
|---|---|---|
| 41.5 % | append-shaped hub atoms (`identical_fraction` 0.999394, 53 KB delta against an 87 MB body) | yes |
| 58.5 % | perturbed — every terminal's `last_fired_tick` touched | no, not by a naive diff |

Projected reduction **1.71x** against a measured **18.7x** disk deficit and a
**3760x** throughput deficit. It is worth building because it compounds with
more RAM, not because it closes the gap on its own.

## The measurement that sizes the perturbed half

Parsing the disagreeing runs at bincode fixint width gives a stride cycle of
**{4, 12, 5} summing to 21** — exactly
`Terminal{target:(u32,u32), weight:f32, consolidation:u8, last_fired_tick:u64}`.

At that width:

- `target` agreement **1.0** — the connection set is stable
- `weight_unchanged` **0** — these are potentiation updates, ratios 1.0006–1.30
- `last_fired_tick_changed` **20,000 of 20,000**

So the perturbed half is not noise and not decay: a hub atom fires every tick,
and every terminal it owns has its tick and weight updated. A diff against the
previous body finds almost every terminal changed, which is why exact-equality
and common-prefix tests both score it identically to "everything changed".

**A delta that carries changed terminals cannot help the perturbed half.** The
only encodings that would are (a) omitting `last_fired_tick` from the durable
body and reconstructing it, or (b) quantising it — both change recall
semantics and are out of scope here.

## Design

A second record type beside `W1ZNEUR1`:

```
W1ZNDLT1 | pool:u32 | id:u32 | base_offset:u64 | len:u64 | payload
```

`payload` is a bincode `TerminalDelta { appended: Vec<Terminal> }` — terminals
added since `base_offset`'s body. Nothing else may differ: the writer proves
that before choosing a delta, and falls back to a full body otherwise.

**Read**: `read_neuron_at(offset)` follows `base_offset` to the base body, then
applies each delta in chain order. A chain is capped (`MAX_DELTA_CHAIN`) so a
read never degrades without bound; on reaching the cap the next write is a full
body, which also re-bases the chain.

**Safety by construction, not by argument** — the same standard the clean-skip
suppression was held to:

1. The writer serialises the candidate body and compares its non-terminal
   prefix against the base. Any difference outside the appended tail forces a
   full body.
2. The base must still be the offset THIS slot points to. A rollback,
   compaction or reopen moves it, and the delta is then refused.
3. A delta is never written when the terminal vector SHRANK or reordered.
   `terminal_idx` makes reorder possible in principle, so this is checked, not
   assumed.
4. Compaction folds every chain into one body. A compacted container contains
   no delta records, so the format cannot leak into a guard clone.

## What must be measured before believing it works

- `delta_appends` and `delta_bytes_saved` beside `page_outs` and
  `clean_skips`. Their ratio is the only direct readout.
- A counter of zero is not a refutation until the path has had the OPPORTUNITY
  to run: the digest map is per-process and this host recycles the node every
  two to three minutes, so measure after several sleep cycles.
- The burn in GB/h by `df`, before and after. Never by summing file sizes: on
  this XFS reflink volume those differ by orders of magnitude.
