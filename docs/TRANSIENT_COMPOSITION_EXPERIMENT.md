# Transient Composition Thought Experiment

## Question

Can independently learned topics produce a deterministic answer to a prompt
that was never trained, without learning during prediction?

## Independent experience classes

The experiment separates four things that need not be learned together:

1. A domain law connects a recognized model to a relation.
2. A sensor or prompt identifies the model present in a new problem.
3. A request identifies the required output representation.
4. A renderer maps a relation into that representation.

Examples of domain laws are:

| Topic | Recognized model | Relation |
|---|---|---|
| Mechanics | mass + acceleration | multiply |
| Density | mass + volume | divide |
| Electricity | voltage + resistance | divide |
| Geometry | width + height | multiply |
| Finance | gross return + fees | subtract |
| Networks | bytes + transfer rate | divide |
| Chemistry | molarity + volume | multiply |
| Music | tempo + duration | multiply |

Each law was crossed with Python, spreadsheet, equation, and SQL renderers,
giving 32 deterministic novel compositions. None of those 32 complete
problem/answer pairs was stored.

## Common pattern

All valid crossings reduce to typed relational joins:

```text
observes(problem, model)
law(topic, model, relation)
    -> applicable(problem, relation)

applicable(problem, relation)
requests(problem, format)
renderer(format, relation, artifact)
    -> solution(problem, artifact)
```

Names are not enough. The shared role must have the same type. Textually equal
values with incompatible types do not join. A conclusion's confidence is the
minimum confidence of its evidence paths, and its provenance is the union of
all experiences and rules used.

## Architectural translation

- Atom and concept neurons provide typed values.
- Binding neurons provide grounded relations between typed values.
- Meta-neuron pools provide learned composition rules.
- Prediction copies activated relations and rules into a transient workspace.
- The workspace performs deterministic variable unification and forward
  chaining without changing neurons, terminals, ticks, or persistence.
- Results carry confidence and complete provenance.
- A result remains transient until an external outcome confirms it through
  consolidation.

This is not unrestricted symbolic logic added beside the brain. It is a
readout and propagation discipline over relations and rules learned by the
brain. The EEM persists confirmed pathways; the workspace is discarded after
each prediction.

## Falsification cases

- Missing renderer: no answer.
- Same text in the wrong semantic type: no join.
- Conflicting evidence: distinct candidate facts remain distinct rather than
  being silently merged.
- Low-confidence premise: conclusion cannot exceed it.
- Prediction without outcome: EEM relation count remains unchanged.

The executable benchmark is `crates/brain/tests/logical_workspace.rs`.

## Runtime surface

- `POST /brain/logic/consolidate` admits a relation or composition rule only
  when `outcome_confirmed` is explicitly `true`.
- `POST /brain/logic/compose` constructs a disposable workspace and returns
  derived facts, confidence, and provenance with `learning: false`.
- `POST /brain/logic/crystallize` admits an outcome-confirmed typed sensor
  frame. Repeated invariant structure becomes a reusable relation template;
  varying positions become typed roles.
- `POST /brain/logic/recognize` matches novel frames against those templates
  and composes the resulting relations without learning.

Structured sensor streams can now crystallize roles automatically from
repeated variation. The remaining upstream problem is sensory typing: raw free
text must identify candidate token roles, while OHLCV and other structured
streams can supply types directly from their schema. The implementation does
not pretend byte overlap is semantic typing.
