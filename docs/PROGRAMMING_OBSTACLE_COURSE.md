# The deterministic 1,000-task obstacle course

`docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md` defines completion for the
senior software-engineer brain as `1000/1000` on a frozen, held-out course.
This document describes the machinery that course runs on, and — more
usefully — the ways a course can report a number nobody should believe, each
of which is now a build-time or run-time error rather than a habit to
remember.

| Component | Path |
|---|---|
| Schema, invariants, freeze/load, leak check | `scripts/programming_obstacle_manifest.py` |
| Authored tasks, one module per family | `scripts/programming_obstacle_tasks/` |
| Bounded isolated runner and scoring | `scripts/programming_obstacle_run.py` |
| Reference solutions and mutations | `tests/obstacle_references.py` |
| Contract tests | `tests/test_programming_obstacle_course.py` |

## Current state

**The course is not built.** Authoring is in progress against a required
1,000, family by family.

Deliberately no count is quoted here. Families land several times an hour
while the course is being written, so a number in this file is stale before
the commit that adds it finishes, and a reader who trusts it is reading
history. Ask the tool instead:

```bash
python scripts/programming_obstacle_manifest.py
```

It prints authored-versus-required for every family, the exact per-family
shortfall, and exits 2 while any of them is unmet. It exits 0 only when a
complete, valid course exists — so the exit code, not a paragraph, is the
answer to "is the course built".

No score has been produced against the brain yet, and none can be: the runner
only accepts a manifest, and a manifest cannot be constructed from an
incomplete task set.

## Why the invariants are where they are

Each of these is a way a course could quietly stop measuring what the
contract asks for. They are enforced in `build_manifest`, so they fail at
build time rather than after a run reports a passing score.

**Exact family counts.** A course of 1,000 tasks skewed toward the families
the brain already passes is not the contract's course. Overshooting a family
is rejected as loudly as undershooting it, because the total would otherwise
absorb the imbalance.

**Distinctness is behavioural, not textual.** A thousand renamings of one
task satisfy a naive uniqueness check while measuring one capability. The
distinctness key is a digest of the *normalized validator plus its fixtures*,
so cosmetic clones collide and the build fails. Comments, whitespace and
Unicode presentation forms are normalized away; anything that could change an
assertion's meaning is preserved. The digest deliberately excludes the family
field — two identical validators filed under different families still measure
one thing.

**Validators must assert.** The contract says identifier or formatting checks
cannot substitute for behaviour. A validator with no assertion passes an
empty candidate, converting an unimplemented capability into a green cell.

**No network, bounded timeouts, pinned toolchains.** A validator that reaches
the network turns an outage into a capability failure. An unbounded validator
turns a hang into an indefinite stall. An unpinned toolchain makes a verdict
irreproducible. All three are build errors.

**A frozen course cannot be edited.** `freeze_manifest` refuses to overwrite a
different course at the same path, and `load_manifest` recomputes the digest
and rejects a manifest whose stored digest no longer matches its own tasks.
Otherwise a failing task could be edited into a passing one between the run
and the audit, with the version string unchanged.

## The two failure directions a validator must survive

A validator can be wrong in two opposite ways, and checking only one is how a
harness silently stops measuring:

- **Unsatisfiable.** A validator nobody has seen pass reports failure forever
  and sends repair effort at curriculum that was never the problem.
- **Undiscriminating.** A validator that accepts anything measures nothing.

So every authored task carries a reference solution *and* a one-line mutation
in `tests/obstacle_references.py`, and the test suite asserts the reference
passes and the mutated version fails. The mutation test additionally asserts
its own find-string is still present in the reference — a stale mutation
would otherwise re-check the unmodified reference and pass vacuously.

Two real defects were found by this check on the day it was written: a
validator asserted the wrong RFC 6901 escape example (`/~00~11` decodes to
`~0/1`, not `~1`; the token that decodes to `~1` is `/~01`), and two mutations
did not actually change behaviour.

### Never pin a verdict to a decimal that looks exact

There is a third way to be unsatisfiable, and the reference test catches it
only after it has already cost the authoring time: a threshold comparison
whose fixture sits *on* the threshold.

`reliability_observability_performance-0005` scores multi-window burn-rate
alerting, where a 1.44% error rate against a 99.9% objective is exactly the
14.4x burn that pages. Written the obvious way, the validator fed an error
rate of `0.0144` and a budget of `0.001` and demanded `severity == "page"`:

```python
>>> 0.0144 / 0.001
14.399999999999999
>>> 0.0144 / 0.001 >= 14.4
False
```

A correct implementation fails that assertion. Nothing in the harness would
have called it a harness bug — it would have been reported as a capability
failure in burn-rate alerting, and the repair loop would have gone looking
for SRE curriculum to fix arithmetic that was already right.

Neither operand is representable in binary, so the quotient lands one ULP
below a threshold that *is* representable. Decimal fixtures that look exact
(`0.0144`, `0.006`, `0.1 + 0.2`) are the common case, not the exotic one.

So: **drive a threshold verdict from a value clear of the boundary, and
assert the boundary arithmetic separately against a tolerance.** The two
questions — "does it compute the burn rate" and "does it apply the rule" —
are separable, and only the first has anything to do with floating point.
Where a task genuinely is about exact boundary behaviour, specify the
comparison in the prompt over integers or `decimal`, so the contract and the
verdict agree on what "equal" means.

## Outcome attribution

The contract admits `1000/1000` "with no skipped, manually waived,
network-dependent, flaky, or validator-error cases", so a non-pass must be
attributed to the right cause:

```
passed          the behaviour contract was executed and held
failed          it was executed and the candidate violated it
timeout         the candidate or validator exceeded the task's bound
no_response     the brain produced nothing to validate
validator_error the validator itself broke — a harness fault, never a verdict
```

`validator_error` is counted separately and blocks admission rather than
being folded into the failure total. Folding it in would make a broken
harness look like a capability gap. This is the same misattribution that made
a SIGTERMed replay worker indistinguishable from a failing admission gate:
"where did the transaction die" and "how often did it fail" have opposite
fixes.

**Attribution follows the traceback, not the exception type.** A candidate
that raises where the contract says it should return is a capability failure,
and that is the *common* shape of the failure — not a clean `False`. So an
exception is charged to the candidate whenever any traceback frame (or a
`SyntaxError` filename) belongs to the candidate file, following `__cause__`
and `__context__`. Only an error raised purely in validator frames is a
harness fault.

## Leakage

Held-out prompts must never become training rows, and a leak is invisible in
the score it produces — a memorised task passes, inflating exactly the number
the contract relies on. `held_out_violations` checks a built manifest against
real corpora.

The fingerprint is **not** a fixed-length prefix. Prompts within a family
share an opening on purpose ("Implement a Python function that ..."), so a
prefix can be common to dozens of tasks and one ordinary training row would
be reported as leaking every one of them. Hundreds of false leaks are not a
conservative failure: they bury the real one. Each task instead gets the
shortest prefix no other task shares, falling back to its whole prompt when
no prefix is unique.

## Running it

```bash
# Audit authored tasks; exits 2 while the course is incomplete.
python scripts/programming_obstacle_manifest.py

# Freeze once complete, checking corpora for leakage first.
python scripts/programming_obstacle_manifest.py --freeze \
    --corpus data/programming/<corpus>.jsonl

# Score the frozen course against the brain (read-only; never observes).
python scripts/programming_obstacle_run.py \
    --endpoint 127.0.0.1:8095 --report runtime/obstacle/report.json
```

The runner exits 0 only on a full-course pass with no validator errors and no
empty answers. A partial run can never report admission however well it
scores, because the contract's threshold is the complete frozen course.

## Authoring the remaining tasks

Add a module per family under `scripts/programming_obstacle_tasks/` exporting
`TASKS`, plus a reference and mutation per task in
`tests/obstacle_references.py`. Validators run under `python -I -S`, so only
the standard library is importable; a task whose verdict depends on a package
installed on one machine and not another is exactly the flaky case the
contract refuses.

Prefer contracts specified by a public standard over contracts specified by
the prompt's own examples — reproducing examples is not evidence of the
capability. Drive the degenerate cases (empty input, single element,
duplicates, cycles, unterminated quotes) explicitly; they are what separate a
memorised textbook body from a working implementation. Where a complexity
claim is part of the contract, assert it structurally — count comparisons or
element reads — never by wall-clock time, which would make the verdict depend
on host load.

### An invisible mutation is a validator that does not probe that axis

The mutation test asserts the mutilated reference **fails**. So a mutation the
validator cannot see does not slip through quietly — it surfaces as that test
failing, with the mutated reference still passing. This is the most useful
signal in the authoring loop, and the temptation is to read it backwards.

Measured while writing the robotics family. The Denavit-Hartenberg link matrix
row is `(ct, -st*ca, st*sa, a*ct)`, and the obvious mutation — dropping the
`alpha` terms to `(ct, -st, 0.0, a*ct)` — changed nothing, because the only
non-planar pose asserted used `theta = 0`, where `st = 0` makes both forms
identical. Three single-joint poses agreed under both.

The fix is almost never to pick a mutation the validator happens to catch.
That trains the mutation on the test and leaves the capability unmeasured. The
fix is to strengthen the **validator** until the honest mutation is visible —
here, adding a pose with two joints bent, where neither link lies on an axis
and any dropped or transposed rotation term diverges.

The rule generalises: a mutation is a probe of the validator, not decoration
for it. Write the bug a competent implementer would actually ship — a
translation negated without being rotated, a quaternion product in the reverse
order, a parallel-axis term added instead of subtracted, a facet record short
by its two attribute bytes — and if the suite still passes, the gap is in the
assertions. Every one of those is correct on the example a prompt would quote
and wrong in an assembly, which is the same reason the contract asks for
standard-specified behaviour rather than reproduced examples.
