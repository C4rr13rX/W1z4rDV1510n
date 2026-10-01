# Teaching benchmark: audio + visual

The owner's spec, 2026-10-01. Everything the owner will do with the brain is
**teaching**, and the brain can handle any data stream. This benchmark checks
two things together:

1. **Teaching works.** Show it things and say what they are, and it learns them
   quickly, keeps them, and builds on them.
2. **RAM stays as low as possible** while every teaching test still passes.

It replaces the text scene world as the **primary** measure. The byte world
and the symbol world stay as regression checks.

## The world (synthetic, deterministic, no downloads)

- **Objects.** N objects per scale (scales 1/4/16/64, at least 16 objects at
  scale 1). Each object has:
  - an **appearance**: a procedurally rendered image of about 32×32 RGB. Shape,
    colour, size and texture come from a seeded RNG, so two objects never look
    the same.
  - a **name**: a synthetic spoken word, which is PCM audio built from a
    seeded sequence of 2–3 "syllables". Each syllable is a short chord of
    frequencies with an amplitude envelope, so each name is a distinct sound.
  - **properties**, also spoken as words: colour, material, and which object it
    rests on.
- **Variation.** Every presentation of an object, in training or testing, is
  jittered: position shifted by up to 4 px, ±10% brightness, pixel noise, and
  ±5% audio pitch and timing. No test frame is byte-identical to a training
  frame. The environment provides identity through the spoken name, but the
  brain still has to recognise *this* view of *that* object.
- **Encoders.** Port `crates/core/src/streaming/image_bits.rs` and
  `audio_bits.rs` into `crates/brain` as `AtomEncoding`s, one visual pool and
  one auditory pool, and keep their label schemes. A label is an atom. No new
  subsystem.

## Teaching episodes

| Episode | What happens |
|---|---|
| **Naming** | Show the object while saying its name: a visual frame and an audio frame in the same moment. |
| **Telling** | Say a fact as a sequence of word sounds, for example "lamp · on · desk" or "desk · material · oak". |
| **Showing a scene** | Show two objects together in one image while saying the relation. |
| **Correcting** | After a wrong fact was taught, teach the right one. |

## Tests

Every test reports a percentage. **Wrong answers and silence are counted
separately:** an invented answer is worse than "I don't know".

| # | Test | Pass means |
|---|---|---|
| T1 | **Name it.** Show an unseen jittered view and get the name. | Visual → audio. The answer decodes to the right name. |
| T2 | **Find it.** Say the name and get the appearance. | Audio → visual. The recalled visual atoms best match the right object. |
| T3 | **Few-shot.** Teach with 1, 3 and 10 naming episodes, then run T1. | Report the curve. Fewer episodes for the same score is better. |
| T4 | **Facts.** Ask a taught fact by sound, e.g. "lamp · on ?" gives "desk". | Recall of what was told. Must stay at 100%. |
| T5 | **Integration.** Ask what was never taught but follows, e.g. show the lamp, ask "material under ?", answer "oak". | Chains across modalities, on held-out relation families written by the lead. |
| T6 | **Keeps it.** Teach batch A, then batch B (same size), then re-test A. | Retention of A. Forgetting is a failure. |
| T7 | **Takes correction.** Teach a wrong fact, then the right fact once, then ask. | The correction wins. |
| T8 | **Rapid sequence.** Feed 100 frames as a stream and act on each. | Milliseconds per inference, and the share of frames answered correctly. |

## T9: crossing disciplines (the owner's definition of generalization)

Generalization is **not** a search procedure over questions, and **not**
connections made for a particular number or subject. It is what happens when
disciplines that were trained **separately** cross paths in the Hebbian fabric
and the relationship between them forms by itself.

The owner's example:

- Teach the brain to **return lists**, on a few categories.
- Teach it **some** lists that involve spots, but not the full list of
  animals with spots.
- Teach it **many single facts** about animals with spots.

Then "list all animals with spots" returns the list, although that list was
never taught.

**The crossing world.** It comes in two forms. The symbol form exists first,
because it is fast. The audio + visual form follows, with spoken words and
shown animals. Each discipline is taught in its **own** episodes and never
combined in training:

| Discipline | Example |
|---|---|
| Lists | "list · red · fruit" gives "apple · cherry · strawberry". Several categories × properties, which teaches the skill of answering with a list. |
| Zoology | "leopard · has · spots", "zebra · has · stripes" and so on. Many single facts. |
| Kinds | "leopard · is · animal", "mushroom · is · fungus", "dice · is · object". Distractors carry spots but are not animals. |

**Probes.** Every probe is a (category, property) list that was never taught
as a list, such as "list · animal · spots". Score each probe by:

- precision against the true set
- recall against the true set
- **wrong items**: listed things that don't belong, for example the mushroom

Report the mean F1 per scale. Scales add more entities, so a list grows from
a handful of members to hundreds.

**It must come from the fabric.** Two checks prove it:

1. **No parser.** Nothing parses the question or special-cases "list".
2. **The ablation check.** Re-run T9 with Hebbian propagation disabled, or
   with the cross-discipline terminals shuffled. T9 must collapse toward 0.

If T9 still scores without the fabric, the score came from a lookup or a
search, not from generalization, and it does not count.

**Speed.** Answering a T9 probe must stay within 3× the time of a plain recall.
"List everything with spots" is where the zoom earns its keep: the "spots"
symbol may hold more connections than RAM keeps, and the goal pages in the
rest from SSD. **A cap that drops connections, or keeps only the first ones
learned, fails T9 by construction.**

## Footprint (same rules as RAM_GOAL.md)

- Peak RAM per scale, measured from outside by `tools/capped.py`.
- RAM growth from scale 1 to 64, pushed toward 1.0.
- The hub fan-out bound.
- The ceiling is 2 GB and the target is a few hundred MB.

Images and audio make far more atoms than text, so this is where the zoom has
to earn its keep: an object's detail atoms should live on SSD once its symbol
exists, and come back only when the goal needs them.

## Rules

- **Fast.** The whole benchmark runs in under 5 minutes, and scale 64 alone in
  under 3.
- **The same path as the product.** The node must accept these audio and
  visual frames on the route the benchmark uses (`/sensor/observe` or its
  successor). See the parity rule in the crew instructions.
- **Gated.** The gate fails when any test's score drops, when RAM rises more
  than 15%, or when RAM goes over the ceiling. A gain is kept only if nothing
  else fell.
- **Honest.** Every test frame is jittered, nothing special-cases a probe, and
  the lead writes the held-out integration families (T5).
