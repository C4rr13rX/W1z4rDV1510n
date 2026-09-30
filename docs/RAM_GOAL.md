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
and properties, then probes it. Numbers from 2026-09-30:

| scale | facts | recall | integration | peak RAM | neuron data | hub fan-out |
|---|---|---|---|---|---|---|
| 1 | 152 | 100% | 0% | 56 MB | 1 MB | 270 |
| 4 | 608 | 100% | 0% | 171 MB | 3 MB | 831 |
| 16 | 2,432 | 100% | 0% | 626 MB | 8 MB | 2,683 |
| 64 | 9,728 | 100% | 0% | **2,471 MB** | **24 MB** | 10,632 |

Three facts follow from the table:

1. **RAM grows linearly with the corpus.** It should stay flat.
2. **About 99% of peak RAM is not neuron bodies.** Scale 64 holds 24 MB of
   neurons in a 2.47 GB process. Measure where the rest goes before you
   design anything: indexes, moment history, posting lists, EEM facts,
   caches.
3. **Hub fan-out grows with the corpus.** An atom is a byte, and every
   byte atom holds a terminal to every concept containing it. In production
   those hubs reached about 4 M terminals and 82 MB each, so they could
   never leave RAM. Symbols should hold a bounded number of strong
   connections, with the long tail on SSD. Recognising a concept should be
   an index lookup, not a byte firing into millions of terminals.

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
