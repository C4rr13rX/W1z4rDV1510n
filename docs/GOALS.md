# W1z4rD V1510n: the four primary goals

Set by the owner on 2026-10-01. All four are pursued **at the same time**, and
no change may buy one by giving back another. Each goal has a measured number,
and the gate (`tools/gate.py`) holds every number at least where it is.

**One absolute rule sits under all four: recall of trained material is always
perfect.** Anything the brain was taught, it answers exactly, 100% of the time,
at every scale and in every benchmark (text, symbol, audio + visual). That is
not "no worse than before". A run below 100% fails the gate, whatever the
baseline says.

## 1. Train to any size, with tightly controlled RAM

- **The rule.** SSD may grow without limit. RAM stays as small as possible:
  a few hundred MB is the target and **2 GB is the ceiling**, whatever the
  brain has learned.
- **Why this is possible.** The symbols active for inference should never come
  near 2 GB. Under normal, human-like operating conditions there isn't that
  much in the environment to look at at one time. Detail lives on SSD and is
  paged in when the goal needs it (the zoom).
- **Measured as** peak RAM per scale and RAM growth from small to large
  (`tools/scorecard.py`, `docs/TEACHING_BENCHMARK.md`). RAM growth must stay
  near 1.0: RAM follows what is active, not what is known.

## 2. Generalization beyond what transformers do

- **The idea.** Transformers ("Attention Is All You Need", Vaswani et al.,
  2017) compute attention over a context window. This architecture builds
  attention into **neural connections and a variable neuron pool system**:
  - what is relevant is what activates
  - the pools that form, and the connections between them, decide what the
    brain attends to

  The aim is generalization that LLMs don't reach, and in time beyond human.
- **What generalization means here.** Disciplines trained separately cross in
  the Hebbian fabric, and the relationships between them form by themselves.
  See T9 in `docs/TEACHING_BENCHMARK.md`.
- **Measured as** T9 F1 on lists and relations never taught. An ablation check
  proves the result comes from the fabric, not from a search or a lookup.

## 3. Correct answers that integrate other correct answers

- **The idea.** The physical world is not arbitrary. Laws and geometry make it
  deterministic; this is Laplace's determinism. Physical phenomena cluster
  together. Train the brain on physics, environmental science, human anatomy
  and more, and it should be able to retrieve the relationships between them,
  including ones science hasn't explored yet, just by being asked.
- **The direction.** The long-run goal is predicting systems like strange
  attractors and outcomes across millions of variables in time and space. On
  the way there, it should get far more right than people know.
- **The bar.** *Perfectly* right. A wrong answer is worse than "I don't know".
- **Measured as** net integration (correct minus wrong) on held-out
  cross-discipline questions, with wrong answers reported separately. This
  goal is reached when that net figure climbs and the wrong rate approaches 0.

## 4. Inference in milliseconds, whatever the brain was trained on

- **The idea.** This is a state-based AI. Inference reads a structured flow of
  logic from the neural state. It does not compute an answer from scratch, so
  it should be as fast as lightning.
- **The target.** About 1 ms per inference, a few ms at most, flat as
  training grows. Loading (deserialization) may take longer; answering may not.
- **Measured as** recall_ms and infer_ms per scale, a T9 probe's time, and
  ms per frame in a rapid sequence (T8). The gate allows no rise above 15%.
  A mechanism that answers by firing hundreds of sub-queries is too slow by
  this goal's terms, whatever its accuracy.

## Keeping the record true

`README.md` describes the architecture **as it actually is**. It is updated in
the same pass as every significant change, and anything a change made untrue
is corrected or removed. Main is pushed to GitHub after every pass the gate
accepts, so the README there is always current.
