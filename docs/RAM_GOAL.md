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
| 1 | 152 | 100% | 0% | 56 MB | 15.1 MB | 0.3 MB | 270 | 152 |
| 4 | 608 | 100% | 0% | 171 MB | 18.6 MB | 1.1 MB | 831 | 512 |
| 16 | 2,432 | 100% | 0% | 626 MB | 28.0 MB | 3.6 MB | 2,683 | 512 |
| 64 | 9,728 | 100% | 0% | **2,471 MB** | **58.3 MB** | 5.9 MB | 10,632 | **512** |

RAM growth scale 1 → 64: **x44.91 → x3.86**. Both numbers moved for reasons
worth keeping, and both were found by measuring rather than reasoning:

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

What is left, measured at scale 64 with
`target/release/examples/scorecard.exe --scale 64 --phase infer --census`:

- **Every remaining growing structure is per-fact.** Brain-level index bytes
  total 12.47 MB: `lifetime_recurrences` 5.13 MB over 9,728 entries (553 B
  per fact) and `tentative_promoted` 5.13 MB storing the **same**
  `MomentFingerprint` key a second time, plus `binding_sequence_index`
  1.24 MB and `pool.label_index` 1.57 MB. A fingerprint owns three `Vec`s and
  holds every atom id of query and answer twice (`ordered_per_pool` and
  `members_per_pool`), so a ~22-atom fact costs ~1.1 KB across the two maps:
  ~1.1 GB at 1 M facts. That is the next wall. Note before planning it:
  `lifetime_recurrences` has ~30 call sites and is persisted in three formats
  (`persistence::SerializableFingerprint`, `wbrain_metadata`,
  `streaming_migration`), and two sites iterate its keys rather than looking
  one up — so replacing the key with a hash is a migration, not an edit.
  Deduplicating the key SHARED with `tentative_promoted` is the cheaper half
  of the same 10.26 MB.
- **~22 MB at scale 64 is still unaccounted, and it is built while
  TRAINING.** Per-phase peaks at scale 64 after both fixes: train 55.8 MB,
  recall 58.3 MB, infer 58.7 MB. So answering now costs 2.9 MB over the
  trained brain (it cost 518 MB at scale 16 before), and everything left is
  allocated by training: 55.8 MB against 12.47 global + 1.57 pool-side + 5.9
  neurons + ~13.7 MB fixed process. Do not look for it in the recall path.
  9 MB of it was `Vec` capacity slack in neuron bodies: `footprint()` counted
  `len`, and counting `capacity` moved `est_resident_mb` at scale 64 from 5.9
  to 14.9 MB, which narrows the unexplained remainder to ~13 MB. The
  Brain-level census still counts `len` only, and every neuron carries its own
  `terminal_idx` `AHashMap` whose capacity nothing counts — start there.
- **Integration is still 0%** at every scale, and that is the second goal.

Integration at 0% is the second goal. The scene world's integration probes
chain two trained facts. For example, "r03 lamp on" gives "desk" and "r03 desk
material" gives "oak", so "r03 lamp on material?" should give "oak".

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
