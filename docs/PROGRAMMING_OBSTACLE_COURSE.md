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

### An assertion that ends on a resetting event observes nothing before it

The undiscriminating direction has a shape that is easy to write and hard to
see. `frontend_state_ux_accessibility-0009` scores a modal focus trap, where
`Escape` closes the dialog, returns focus to the opener, and makes every
later key a no-op. The obvious way to test the no-op half was one sequence
covering everything:

```python
run_dialog(fields, ['Tab', 'Escape', 'Tab', 'Escape'], 'open_btn')
```

A dialog that keeps handling keys after closing moves focus to `save` on the
third key — and then the fourth key, `Escape`, sets focus back to the opener.
The final state is identical to the correct one. The assertion that existed
specifically to check "later keys are ignored" could not observe a
implementation that ignored none of them.

This is not a mutation-choosing problem; it is the validator measuring less
than it appears to. **A sequence whose last event overwrites the state being
asserted on cannot witness the events before it.** End such a sequence on an
event that *reads* the state rather than one that resets it — here,
`['Escape', 'Tab']`, where a processed `Tab` leaves focus visibly inside a
closed dialog.

### Derive a fixture the rest of the validator already pins

The same family's `-0007` expected a debounced stream with `max_wait=250` to
invoke at `450`. Four assertions earlier, the same validator pins the rule
that decides it: a call arriving exactly when the timer fires opens a new
burst *measured from that call*. Applying that rule to the stream gives `500`
— `450` would require the second burst to begin at a call the first burst had
already absorbed and already delivered as its payload.

Both numbers look equally plausible written down. Only one is implied by the
validator's own other assertions, and an implementation that got the rule
right would have been scored as a debouncing failure. **Where a fixture is
determined by a rule stated elsewhere in the same validator, compute it from
that rule rather than reading it off the scenario** — and if the two
assertions disagree, the validator is inconsistent with itself, which no
candidate can resolve.

### The prompt is the specification, and the table is the second opinion

`-0008` scores CLDR plural selection. Its Arabic fixture expected `101` to be
`one` and `102` to be `two`, which is the `n % 100` rule that genuinely
governs `few` and `many` in that language — but Arabic `one` is `n = 1`
exactly, as the prompt said. The fixture and the prompt disagreed, and the
fixture was wrong.

A fixture table is easy to fill in by pattern from the rows above it. When it
contradicts the prompt, **the prompt wins**: it is what the candidate is
given, so a disagreement is unpassable by construction rather than difficult.
The three cases here were caught only by writing the reference — which is the
argument for writing it in the same change as the tasks, never after.

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

### Borrowing a standard's name obliges you to its whole rule

Preferring standard-specified contracts has a failure mode of its own: a
prompt that names a familiar format and then specifies *most* of it. The gap
is invisible while writing, because the author fills it from memory of the
real standard and the reader cannot.

`cicd_containers_packaging_platform-0006` states gitignore-style matching. The
first draft said a trailing `/` matches directories and left unsaid whether a
pattern without one could match a directory too. In real gitignore it can, so
`docs/*` excludes the directory `docs/img` and everything beneath it. The
validator had been written from the other reading and asserted only
`docs/api.md`. Both readings are defensible from the prompt, which means the
task measured whether the candidate guessed the author's, not whether it can
implement ignore semantics.

The repair is to make the contract self-contained and say so: file patterns
apply to the path, directory patterns to its directory components, stated as
the prompt's own rule rather than as a pointer at git's. Deviating from a
standard is allowed — silently deviating is not. Where the prompt keeps the
standard's name, it owes the standard's behaviour, including the parts nobody
remembers; where it narrows, it must define the narrowed rule outright.

The cheap detection is to re-derive every expected value in a validator from
the prompt's text alone, with the reference implementation closed. The same
pass caught a second defect in that family: a retention fixture asserted an
artifact would be swept when the policy's own per-branch rule protected it —
the expected value had been written from the intent of the case rather than
from the stated rule.

### Claim a family from the working tree, not from HEAD

The write-once `REFERENCES` guard catches two blocks defining the same task.
It does not catch two agents authoring the same *family*, because that
collision happens before either one writes a reference — and by then a family
module is already 30 KB of finished work.

Measured 2026-09-05. `authoring_status()` reported
`testing_debugging_repair_refactoring` at `0/100` and `git log` showed no
commit for it, so it looked like the obvious next family. It was not: another
session had already written all eight of its references into
`tests/obstacle_references.py` and was mid-authoring. The file was `M` in
`git status` with an mtime eighteen seconds old, and the references named
`minimize_failing_input` where the new module named `minimize` — two
incompatible halves of one family. It was caught by scrolling to the end of
the references file for the insertion point, which is luck, not a control.
The other session committed it forty minutes later as `33b836f`.

So the check before authoring is three commands, not one:

```bash
git status --short                     # is the references file already dirty?
git log --oneline -5                   # did a family land since this checkout?
grep -o 'REFERENCES\["[a-z_]*' tests/obstacle_references.py | sort -u
```

The third is the one that matters, because a claim appears in the references
table before it appears anywhere `authoring_status()` can see it: references
live in `tests/`, authored tasks live in `scripts/`, and the status helper
reads only the latter. A family with references and no tasks is not an
inconsistency to repair — it is somebody else's claim, in progress.

That asymmetry is also load-bearing in the other direction, and it is already
guarded: `test_no_reference_without_a_task` fails while such a claim is open.
It is a correct guard reporting a real transient state, so a session that
finds it red should confirm the claim is live before "fixing" it by deleting
another agent's work.

#### The check is only valid at the instant you write

Measured 2026-09-05, the next time a session woke into this repository. All
three commands were run at 21:32 UTC and all three were clean:
`git status --short` printed nothing, `git log` showed no architecture commit,
and no `architecture_multifile_integration` key appeared in the references
table. `authoring_status()` agreed at `0/70`. On that evidence the family was
unclaimed, and it was the obvious pick — the only family nobody had started.

By 21:37 the same three commands reported an untracked
`scripts/programming_obstacle_tasks/architecture_multifile_integration.py`
two minutes old, a references file modified thirty-three seconds earlier, and
six architecture reference blocks. Nothing had changed about the session's
own reasoning; another agent had simply been mid-file at 21:32, with its work
still in an editor buffer rather than on disk. It committed as `2d17d76`
seven minutes later.

So the three commands establish nothing durable. They describe the working
tree at the moment they run, and a family can be claimed in the gap between
surveying and writing — which is exactly the gap that reading the module
conventions, choosing a family, and drafting the first task opens up. **Re-run
them immediately before the first write, not once at the start.** The cheapest
strong signal is the pair of mtimes:

```bash
ls -lt --time-style=+%H:%M:%S tests/obstacle_references.py \
    scripts/programming_obstacle_tasks/
```

A references file modified seconds ago is another agent typing, whatever the
family counts say. A count of zero means nobody has *finished*, never that
nobody has started.

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

### A validator that runs candidate output owns the attribution

`_blames_candidate` decides `failed` versus `validator_error` by walking the
traceback for the candidate's own filename. That is the right rule, and it has
a blind spot the author has to close by hand: **code the candidate *produced*
is not code the candidate *ran*.**

A refactoring task is the natural case. `rename_local` returns source text, so
checking it means compiling and executing that text — and a rewrite that
breaks raises inside `exec`, in a frame belonging to `<module>`, not to
`candidate.py`. No candidate frame appears anywhere in the traceback, so the
runner attributes a genuine capability failure to the harness. The contract
counts `validator_error` separately and lets it **block admission** rather than
score, exactly so a broken harness never reads as a capability gap; the cost
of getting this wrong is therefore not a wrong verdict but no verdict, on a
task that was working correctly as a measurement.

So a validator that executes anything the candidate generated must convert the
outcome itself:

```python
def evaluate(text):
    try:
        exec(compile(text, "<module>", "exec"), namespace)
        return namespace["compute"](0, [Row(3), Row(4)])
    except Exception as error:
        raise AssertionError(f"the rewritten module does not run: {error!r}")
```

The same applies to any task whose subject is emitted code, a generated
config, or a serialized document that the validator then loads.

There is a corollary for the mutation. The mutated reference must trip an
assertion that runs **before** any such `exec`, or the mutation test asserts
`FAILED` and gets `VALIDATOR_ERROR` — a failure that looks like a broken
mutation and is really an ordering problem. For the rename that means the
structural checks (parameters renamed, attribute untouched, literals
untouched) come first and the behaviour check comes last, which is the useful
order anyway: it says *what* is wrong rather than only that something is.

### Commit a family atomically; it is the only claim other sessions can see

Two sessions have now taken the same family at the same time, and the second
occasion was worse than the first. Both wrote
`scripts/programming_obstacle_tasks/<family>.py`, one withdrew, and for a
window the tree held eight references and eight mutations for a module that no
longer existed.

Nothing in that state is silently wrong — the write-once `REFERENCES` guard,
the AST duplicate scan and
`test_every_authored_task_has_exactly_one_reference_and_mutation` all catch it
— but every one of them catches it only when something runs them, and an
untracked file is invisible to a sibling session's `git log`.

So: check `git log --oneline` and `authoring_status()` immediately before
authoring, and commit the family module and its reference/mutation block in
**one** commit as soon as the family's tests are green. Do not leave a family
sitting uncommitted while polishing it. The commit is the claim; the working
tree is not.

That rule was written after two collisions and has now been paid for a third
time, in the other direction: a session left
`frontend_state_ux_accessibility.py` untracked with twelve tasks and no
references at all. The contract suite was red for the whole window, and the
next session's first job was deciding whether the file was a live sibling's
work or an abandoned one — a question the commit log would have answered in a
line. Twelve tasks with no reference is not a partially finished family; it is
an unmeasured one, because neither direction of the satisfiable/discriminating
check has run against any of them.

### Fixture modules: the other side of the seam

`ObstacleTask.fixtures` maps a filename to file contents, written into the
workspace beside the candidate before the validator runs. The harness does
`os.chdir(WORKSPACE)` and `sys.path.insert(0, str(WORKSPACE))`, so a fixture
is importable by name from both the candidate and the validator even though
the child runs under `python -I -S`.

`architecture_multifile_integration` is the first family to use them, and it
needs them: the capability under test is fitting a seam the candidate does not
own, which cannot be posed without supplying the other side. A fixture is
therefore written to be *unhelpful in the way real code is* — the legacy store
there raises one exception type for absence and for malformed input, holds
text where the caller has bytes, and counts its own reads. None of it is a
hint, and a validator can assert against that count to prove the adapter did
not consult the store when the contract said it must not.

Two attribution rules follow from `_blames_candidate`, which recognises only
the candidate's filename:

- **Never let the validator call a fixture directly in a way that can raise.**
  A fixture frame is neither candidate nor validator by name, so an exception
  raised with no candidate frame beneath it scores `validator_error` and
  blocks admission instead of failing the task. Drive fixtures *through* the
  candidate and have the validator inspect the fixture's resulting state.
- An `AssertionError` is always `failed`, whatever raised it, so an ordinary
  assertion on fixture state after the candidate ran is safe and is the
  preferred shape.

### A negative fixture has to reach the rule it violates

The unsatisfiable direction has a shape distinct from the boundary-decimal
one above, and it hides in the list of inputs a validator expects to be
*rejected*. Those lists are written quickly — a handful of malformed strings
per task — and each entry carries an implicit claim that the input actually
reaches the rule it is supposed to break. When it does not, the validator
demands an error no correct implementation can raise.

Measured while writing `validation_parsing_serialization-0009`, the strict
RFC 4648 base32 decoder. Canonical decoding requires that the bits left over
after the last whole byte of a group are zero, and the task asserts a list of
non-canonical strings all raise. One entry was `AAAQEAYB`: eight characters,
no padding. Eight base32 characters are forty bits, which is exactly five
bytes — a full group has *no* leftover bits, so the canonicality rule does not
apply to it and there is nothing for a correct decoder to reject. The
reference caught it immediately, but a course shipped without references
would have reported it as a base32 capability failure forever.

The rule is worth stating in general because it applies to every rejection
list: **a value asserted to be invalid must be in the domain where the
invalidating rule has force.** Padding counts, trailing bits, length limits
and range checks all have preconditions, and a fixture that misses the
precondition tests nothing while looking like a strict test. The check is to
name, for each rejected input, the clause it violates and confirm the input
satisfies that clause's preconditions — here, "spare bits are non-zero"
presupposes a group that has spare bits, which is any group but a full one.

The correction was to swap in `MZXR====` — four characters, twenty bits, two
whole bytes and four spare, with the last of them set.

### Where the standard ships with the language, generate the fixtures from it

Prefer contracts specified by a public standard, and the fixtures for one
still have to come from somewhere. Recalling them is how the wrong RFC 6901
escape example and the wrong CLDR Arabic plural rule reached this repository.

Some standards have a correct implementation already sitting in the standard
library, and where they do it is the fixture generator: RFC 3492 punycode is
`str.encode("punycode")`, Unicode normalization is `unicodedata`, and base64
is `base64`. `validation_parsing_serialization-0014` states ten encode/decode
vectors including `räksmörgås` → `rksmrgs-5wao1o` and `ドメイン名例` →
`eckwd4c7cu47r2wf`, and every one of them was produced by running CPython's
own codec at authoring time rather than transcribed from the RFC's test
section. Punycode is the case that most rewards this: the bias adapts after
each code point, so the encoding of a character depends on everything before
it, and a vector that is one character wrong is indistinguishable by eye from
a correct one.

This does not weaken the task. The prompt forbids the candidate the same
module — `codecs`, `encodings`, and the quoted codec names are checked
against `RESPONSE_TEXT` — so the capability under test is still implementing
the algorithm. The stdlib is used once, by the author, and never at scoring
time; nothing about the verdict depends on it.

Two cautions. The generator has to be genuinely independent of the thing
under test: a fixture produced by the reference solution proves only that the
reference agrees with itself, which is what the mutation test already checks
from the other side. And a stdlib implementation is not always the standard —
`base64.b32decode` accepts the non-canonical trailing bits that `-0009`
requires be rejected, so it is a fixture source for the *encoding* direction
and an example of the defect for the decoding one.
