# The RAM goal: a brain that thinks in symbols

Owner's vision, 2026-09-30. This page is the brief for the automated loop.
It overrides older notes where they disagree.

## What the brain must do

The brain reduces what it perceives to **concept atoms**: symbols with
properties. The first time it enters a room it inspects the bed, the chair,
the mirror, the desk, the lamp, the door, the window, the crumpled paper on
the floor, and learns everything about them. The next time, the room is a
handful of symbols. An object's properties, which are its connections to
other neurons, are brought into memory **only when the current goal needs
them**. Otherwise everything rests on the symbols.

Think of a person who both drives cars and repairs them. Driving, they attend
to a few things: the lane, the signs, what the signs mean for what they do
next. Repairing, they think about the whole engine. The brain **zooms**
the same way. What is in RAM follows the task. Knowledge the task needs
is paged in from SSD, and when it is not there at all, the brain trains
itself. Inference is a sequence of frames, like a film: instructions that
drive an agent (C0d3rV2) acting in a 3D environment.

**The RAM promise:** a few hundred MB is the target and 2 GB is the ceiling,
whatever the corpus. The SSD may grow without limit. That is what lets this
run on a phone and still feel like talking to a well-read person.

**Every change must make it a better AI.** Recall of what it was taught
stays at 100%. Integration, meaning correct answers it was never trained on
but which are true in the world, must never get worse.

## Where it stands (measured, not assumed)

`python tools/scorecard.py --stress` trains a scene world of rooms, objects
and properties, then probes it. Numbers from 2026-09-30, after the two
changes below (the first column is where this page started the same day):

| scale | facts | recall | integration | peak RAM was | peak RAM | neuron data | hub fan-out was | hub fan-out |
|---|---|---|---|---|---|---|---|---|
| 1 | 152 | 100% | 0% | 56 MB | 15.0 MB | 0.3 MB | 270 | 152 |
| 4 | 608 | 100% | 0% | 171 MB | 17.6 MB | 1.2 MB | 831 | 512 |
| 16 | 2,432 | 100% | 0% | 626 MB | 24.3 MB | 4.1 MB | 2,683 | 512 |
| 64 | 9,728 | 100% | 0% | **2,471 MB** | **45.6 MB** | 13.9 MB | 10,632 | **512** |

RAM growth scale 1 → 64: **x44.91 → x2.98**. Every number moved for reasons
worth keeping, and every one was found by measuring rather than reasoning:

1. **Answering was the allocator, not learning.** One scorecard run per
   phase at scale 16 (peak measured from outside by `tools/capped.py`): train
   28.3 MB, recall 546.7 MB, infer 628.4 MB. The trained brain is 28 MB;
   answering 2,432 questions allocated the other 518 MB.
   `Pool::check_concept_emergence` inserted one permanent
   `AHashMap<Vec<NeuronId>, u32>` key per run of length
   2..=`max_concept_member_count` ending at every observed atom, so a 17-byte
   question added ~1,071 entries nothing reclaims — ~2.6 M over the probe
   set. The ledger is **empty after training**:
   `pretrain_binding_episode` does not go through emergence at all.
   `Brain::observe_read_only` suppresses emergence for one observe call.
2. **The hub was a single byte.** The census names the top three neurons per
   pool; at scale 16 they were the query-pool atoms `q:cg`, `q:IA` and `q:Pw`
   at fan-out 2,432 each — the fact count exactly — against 21 on the largest
   concept. The site is `Brain::promote_binding_concept`'s bottom-up
   member→binding pass. `PoolConfig::max_atom_fanout` (default 512, 0 =
   unbounded, serde-defaulted so old snapshots keep the old behaviour) caps
   it, and recall stayed at 100% because `binding_sequence_index`,
   `binding_feature_atom_index` and `label_to_id` already reach a binding
   from its members by lookup — the index lookup this page asked for instead
   of a byte firing into millions of terminals.

Three further changes, all in the same structure, took scale-64 peak from
58.3 MB to **45.6 MB** and RAM growth from x3.86 to **x2.98**. Every one was a
duplicate the census had been reporting as two different things:

3. **Five indexes held five copies of one fingerprint.** `moment_history`,
   `binding_recurrences`, `lifetime_recurrences`, `tentative_promoted` and
   `promoted_fingerprints` each cloned the whole `MomentFingerprint`. They
   share one `Arc` now; `Arc<T>: Borrow<T>` left every call site taking a
   plain `&MomentFingerprint`. The census was charging each map the full key,
   which is why `lifetime_recurrences` and `tentative_promoted` both read
   5.13 MB — it charges a map its pointer plus its value and the pointee once,
   under `fingerprint_keys`.
4. **A binding's label spelled its membership.** `"p1n0|p1n0|p1n2|…|ordered:…"`
   — 170 bytes per binding, held in `pool.label_index` and again in the
   neuron. Nothing parses it; it is read back only through `label_to_id`, so
   it is a 35-byte symbol (two 64-bit digests of the fingerprint's own `Hash`).
   A miss falls back to building the legacy form so an old snapshot still
   dedups.
5. **The fingerprint stored its atom stream three times.** `members_per_pool`
   was a clone of `ordered_per_pool` that enrichment extended (nothing is
   enriched on the training path), and `pairs` is that same stream sorted and
   flattened at 8 bytes a pair against 4. Both are derived now
   (`members_per_pool()`, `pairs()`); `members_extra` holds only the
   enrichment suffix. `pairs` could not simply be deleted: legacy bincode
   snapshots persisted the pair signature and NO temporal order, so those
   records restore with an empty `ordered_per_pool` and keep their list in
   `legacy_pairs`.

What is left, measured at scale 64 with
`target/release/examples/scorecard.exe --scale 64 --phase infer --census`:

- **Brain-level index bytes are 12.47 → 4.72 MB** and `pool.label_index`
  1.57 → 0.61 MB. The remainder is still per-fact and still the same shape:
  `fingerprint_keys` 2.79 MB over 9,728 (287 B per fact, now just
  `ordered_per_pool` plus the struct — the census counts Vec CAPACITY here
  now, which is 29 % more than the `len` it counted before), `binding_sequence_index` 1.30 MB over
  9,742 (134 B — the query's atom sequence, a THIRD copy of what the
  fingerprint already holds), `binding_feature_atom_index` 0.64 MB and
  `binding_motif_index` 0.37 MB. Deduplicating the sequence index against the
  fingerprint is the next one of these, and it is worth less than the neuron
  bodies below.
Two more changes, both the same defect: a cache that a MINORITY of neurons
uses was charged to ALL of them. Scale-64 peak 36.6 → **35.3 MB**, growth
x2.47 → **x2.34**, census-accounted 13.43 → **12.11 MB**.

6. **Only a hub allocates a terminal index; every neuron carried its
   header.** Keeping the index BUCKETS off non-hubs did nothing about the
   struct field. An inline `AHashMap` is a hasher plus a `RawTable` —
   measured 64 bytes, **29.6 % of a 216-byte `Neuron` and its largest single
   field** — paid by all 9,776 whether or not a map was ever allocated.
   Boxed, it is a null-pointer-optimised 8, and `size_of::<Neuron>()` is
   **216 → 160**. The field is private now behind `find_terminal`,
   `terminal_index_len/capacity/get` and `release_terminal_index`, so the
   absent case cannot be spelled two ways.
7. **A capped hub cannot repay a hash index over its own terminals.** The
   threshold was 64 because 64 was just above the non-hub maximum; the number
   that matters is the other end. `max_atom_fanout` caps a neuron at 512
   terminals, and a bucket is 17 bytes per 24-byte terminal with 512 entries
   rounded up to 1024 buckets — so the 48 indexed neurons held ~17 KB of
   table each against the ~12 KB their terminals occupy. At 1024 the
   threshold sits above the cap and `terminal_idx` is **0.797 → 0.000 MB**.
   Not a latency trade: a scan over 512 contiguous terminals fits in L1 and
   measured FASTER than the map plus its cache misses (infer 2.46 → 1.97 ms,
   recall 1.31 → 1.26 ms). An uncapped pool still gets an index past 1024.

   Both numbers came from `the_struct_is_priced_field_by_field`, which prints
   the layout field by field — the measurement that picked the change.

- **Neuron bodies are now the largest pot, and the census names the split.**
  `--census` reports `neuron_body_bytes` over 9,776 neurons at scale 64:
  `terminals` 6.11 MB, **`terminal_idx` 5.05 MB**, `struct` 2.11 MB
  (`size_of::<Neuron>()` = 216 B), `members` 1.70 MB (the ~22 refs, 174 B) and
  `label` 0.34 MB (35 B — the symbol above). 15.32 MB total, against a 45.6 MB
  peak with ~15 MB of fixed process, so this is a third of everything
  per-fact.

  **`terminal_idx` is 83 % of what `terminals` itself costs, and it is a
  cache.** It is `#[serde(skip)]`, rebuilt on restore, and exists only to make
  `reinforce_terminal` O(1) over the `terminals` `Vec` it indexes — 17 bytes
  of bucket per terminal against the ~21 the terminal occupies. Its own doc
  comment budgets ~3 GB for it at fabric peak. It is the one component here
  that stores no information the brain does not already hold, so it is the
  next change: a `Vec` kept sorted by target, or a map only on neurons above
  a fan-out floor, buys back most of 5 MB at scale 64. Note the map allocates
  for its CAPACITY, not its length, which is why nothing counting `len` ever
  saw it.

- **The residual is no longer a residual: the allocator itself splits it, and
  most of it is not the brain.** Every census here is a model of what the
  brain *believes* it owns, and `peak_mb` is measured from OUTSIDE by
  `tools/capped.py`, so the gap between them had three possible owners
  needing three different fixes and nothing told them apart. The scorecard now
  installs a counting `GlobalAlloc`, so every byte the process requests is
  booked. Measured at scale 64, infer phase
  (`target/release/examples/scorecard.exe --scale 64 --phase infer --census`):

  | scale | peak_out | process overhead | transient churn | live heap | census-accounted | harness probes | fixed brain construction | uncounted |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 14.9 | 10.84 | 0.01 | 4.04 | 0.42 | 0.02 | 3.27 | 0.33 |
  | 4 | 16.9 | 11.50 | 0.05 | 5.34 | 1.62 | 0.07 | 3.27 | 0.38 |
  | 16 | 22.9 | 13.25 | 0.09 | 9.55 | 5.44 | 0.29 | 3.27 | 0.55 |
  | 64 | **37.2** | **15.13** | **2.93** | **19.14** | **13.43** | **1.17** | **3.27** | **1.27** |

  Read the columns, because they have different futures. **Process overhead
  never passed through the brain's allocator at all** — the Rust runtime, the
  mapped binary, allocator arenas — and no representation change touches it;
  at scale 64 it is 41 % of peak and it is the floor the growth ratio is
  measured against. **`fixed brain construction` is 3.27 MB at every scale**,
  identical to two decimals from 152 facts to 9,728, so it is `Brain::new`
  with two empty pools (the EEM's equation tables, the annealer, the fabric)
  and is NOT a per-fact residual however large it looks at scale 1, where it
  is 81 % of live heap. **The harness's own probe set** — 9,728 facts and
  1,536 integration probes held as `Vec<u8>` by `SceneWorld` — is 1.17 MB that
  belongs to the question, not the answer. What is left genuinely uncounted is
  **1.27 MB**, so the census names **91.4 %** of the per-fact live heap, and
  looking for a large unnamed structure is now a dead end.

  Two of these columns are the next work, and neither is a neuron. Transient
  churn goes 0.01 → 0.09 → 2.93 MB, superlinear in facts: allocation made and
  freed inside the run, which raises peak without raising the brain. And
  process overhead rises 10.84 → 15.13, which for a fixed binary is the
  allocator holding freed arenas — the same churn seen from outside. Those are
  one phenomenon counted twice, and together they are 18.06 MB of a 37.2 MB
  peak.

  How this was reached: 9 MB was `Vec` capacity slack in neuron bodies
  (`footprint()` counted `len`); the Brain-level census and `side_structure_bytes`
  counted `len` for every hash map, which misses the whole control-byte array
  and the ~1/8 of buckets load factor leaves spare (`hash_table_bytes`
  inverts hashbrown's `capacity = buckets - buckets/8`; 9,728 facts allocate
  16,384 buckets, a 68 % undercount per map); `SequenceFingerprint` IS a
  `Vec<NeuronId>`, so the ledger charged a 24-byte header and none of the
  key's heap; and the neuron slot table and the transient firing state were
  allocations no census charged to anything. Per-phase peaks say the brain is
  built while TRAINING, not while answering, so do not look in the recall
  path. The census is unchanged by the fan-out cap, as expected, since a cap
  removes terminals and not fingerprints.
- **Integration is still 0%** at every scale, and that is the second goal.
  The scene world's integration probes chain two trained facts: "r03 lamp on"
  gives "desk" and "r03 desk material" gives "oak", so "r03 lamp on material?"
  should give "oak". Nothing this pass touched it in either direction.

## How to work

- **Measure first.** Find which structure holds the RAM (counts × sizes, or a
  heap profile), then change that structure. A design without a measurement
  behind it does not get built.
- **Choose the simplest representation that makes the number move.** Change
  one thing, rerun the scorecard, and keep the change only if it moved.
- **Change the representation, not the plumbing.** The last ~40 commits on
  main patched symptoms of the representation: compaction, disk alarms,
  capacity halts, write suppression. **Do not add storage-layer machinery.**
  When the same subsystem needs a third fix, stop and question the design
  that keeps needing fixes.
- **Stay backward compatible, or migrate.** Old `brain.bin` snapshots must
  still load, or a tested migration must convert them.
- **Keep the gate green.** `python tools/gate.py` runs the brain tests, a
  compile check of the node, and the scorecard. The scorecard fails if recall
  or integration drops, if RAM rises more than 15% at any scale, or if RAM
  goes over 2 GB. When you improve a number, lock it in with
  `python tools/scorecard.py --stress --save-baseline`.
- **Never lock up the PC.** Run anything that trains or loads a brain through
  `python tools/capped.py --mb <n> -- <command>`. Build with `-j 2`. Never
  start the node, the supervisor, or anything in scripts/aws. Never touch
  `D:\w1z4rdv1510n-data` or the `brain-data*` directories.
