"""Held-out tasks: testing, debugging, repair, and refactoring.

Every other family asks the candidate to build something. This one asks it to
work on code that already exists and is already wrong, which is where a senior
engineer actually spends the day. The capability under test is therefore not
"can you write an algorithm" but "can you narrow a failure down, attribute it,
and change code without changing what it means".

Two properties recur in the validators here because they are what separate the
real skill from a plausible-looking imitation:

**A repair must be attributable.** Reducing a failing input, bisecting a
regression, bucketing crash reports and ranking suspicious lines all answer the
question "which part is to blame". A routine that returns *a* failing input, or
*a* bad revision, has not answered it. So the validators assert minimality and
probe budgets rather than mere failure-preservation -- returning the whole
input is a correct-but-useless answer that a looser check would admit.

**A refactor must be behaviour-preserving in both directions.** It is easy to
write a rename that catches every occurrence and easy to write one that catches
only the safe ones; the difficulty is doing both at once. The rename task is
checked structurally *before* it is executed, because a token-level
search-and-replace passes an execution test on a source file that happens not
to contain the name in a string or an attribute, and this repository's contract
is that a validator measures the behaviour it names rather than the sample it
was drafted against.

The traceback-grouping task deliberately overlaps nothing in
`reliability_observability_performance`: that family folds a *current* snapshot
of dependency states into a verdict, while this one collapses a *history* of
crash reports whose incidental detail -- line numbers, temporary paths --
differs on every run. Bucketing on raw text is the defect being measured.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "testing_debugging_repair_refactoring"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function minimize_failing_input(items, "
            "is_failing) that reduces a failing input to a 1-minimal one by "
            "delta debugging. items is a sequence and is_failing(subset) "
            "takes a list and returns a bool. Return a list that is a "
            "subsequence of items, preserving their original relative order, "
            "for which is_failing returns True, and which is 1-minimal: "
            "removing any single element from it must make is_failing return "
            "False. Return the empty list if is_failing([]) is already True. "
            "Raise ValueError if is_failing on the whole of items returns "
            "False, since there is then no failure to reduce. Call is_failing "
            "at most len(items) ** 2 + 100 times, which rules out searching "
            "the subsets exhaustively."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("minimize_failing_input") + r'''
def is_subsequence(small, large):
    index = 0
    for value in small:
        while index < len(large) and large[index] != value:
            index += 1
        if index == len(large):
            return False
        index += 1
    return True


class Predicate:
    """Counts calls, so the probe budget is measured rather than assumed."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    def __call__(self, subset):
        self.calls += 1
        return self.rule(list(subset))


# --- an interaction between two elements: the answer is exactly those two ---
# A reducer that only ever tries removing one element at a time never finds
# this, because dropping 7 alone or 42 alone still fails.
items = list(range(1, 101))
pair = Predicate(lambda subset: 7 in subset and 42 in subset)
result = minimize_failing_input(items, pair)
assert result == [7, 42], (
    f"the 1-minimal failing input is [7, 42]; got {result!r}"
)
assert pair.calls <= len(items) ** 2 + 100, (
    f"used {pair.calls} probes for {len(items)} items, over the budget"
)

# --- minimality is the contract, not failure-preservation -------------------
# Any three elements fail here, so many answers are correct and returning the
# whole list is not one of them.
size = Predicate(lambda subset: len(subset) >= 3)
result = minimize_failing_input(items, size)
assert size.rule(result), "the returned input does not fail"
assert is_subsequence(result, items), (
    f"{result!r} is not a subsequence of the input in its original order"
)
assert len(result) == 3, f"expected a 3-element reduction, got {result!r}"
for position in range(len(result)):
    shrunk = result[:position] + result[position + 1:]
    assert not size.rule(shrunk), (
        f"removing {result[position]!r} still fails, so {result!r} is not "
        "1-minimal"
    )

# --- minimality checked generically on a third rule -------------------------
threshold = Predicate(lambda subset: sum(subset) >= 100)
result = minimize_failing_input(items, threshold)
assert threshold.rule(result), "the returned input does not fail"
assert is_subsequence(result, items), "order was not preserved"
for position in range(len(result)):
    shrunk = result[:position] + result[position + 1:]
    assert not threshold.rule(shrunk), (
        f"{result!r} is not 1-minimal: dropping index {position} still fails"
    )

# --- the budget holds on a larger input -------------------------------------
wide = list(range(500))
sparse = Predicate(lambda subset: 11 in subset and 480 in subset)
result = minimize_failing_input(wide, sparse)
assert result == [11, 480], f"got {result!r}"
assert sparse.calls <= len(wide) ** 2 + 100, (
    f"used {sparse.calls} probes for {len(wide)} items, over the budget"
)

# --- degenerate inputs ------------------------------------------------------
assert minimize_failing_input([1, 2, 3], lambda subset: True) == [], (
    "an already-failing empty input must reduce to the empty list"
)
assert minimize_failing_input([42], lambda subset: 42 in subset) == [42]

for bad in (lambda: minimize_failing_input([1, 2, 3],
                                           lambda subset: False),
            lambda: minimize_failing_input([], lambda subset: False)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an input that does not fail was accepted")
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function first_bad_revision(revisions, "
            "is_bad) that finds the commit that introduced a regression. "
            "revisions is a sequence ordered oldest first, and is_bad(rev) is "
            "monotonic: once it returns True for a revision it returns True "
            "for every later one. Return the first revision for which is_bad "
            "is True, or None when no revision is bad. Because is_bad is a "
            "full build-and-test cycle, call it at most "
            "ceil(log2(len(revisions))) + 1 times. Raise ValueError if "
            "revisions is empty."
        ),
        validator=LOAD_CANDIDATE + require("first_bad_revision") + r'''
import math


class Probe:
    def __init__(self, first_bad_index, total):
        self.first_bad_index = first_bad_index
        self.total = total
        self.calls = 0

    def __call__(self, revision):
        self.calls += 1
        index = int(revision.split("-")[1])
        return index >= self.first_bad_index


def revisions(count):
    return [f"rev-{index:06d}" for index in range(count)]


def budget(count):
    return math.ceil(math.log2(count)) + 1


# --- the regression is found, and cheaply -----------------------------------
# A linear scan gets the same answer and would need 2719 builds for this one.
history = revisions(4096)
probe = Probe(2718, 4096)
found = first_bad_revision(history, probe)
assert found == "rev-002718", f"found {found!r}, not rev-002718"
assert probe.calls <= budget(4096), (
    f"used {probe.calls} builds; the budget for 4096 revisions is "
    f"{budget(4096)}"
)

# --- a clean history is not a failure ---------------------------------------
clean = Probe(10**9, 1000)
history = revisions(1000)
assert first_bad_revision(history, clean) is None, (
    "a history with no bad revision must report None"
)
assert clean.calls <= budget(1000), f"used {clean.calls} builds on a clean run"

# --- the boundaries ---------------------------------------------------------
all_bad = Probe(0, 1000)
assert first_bad_revision(history, all_bad) == "rev-000000", (
    "a history that was already broken must blame its first revision"
)
assert all_bad.calls <= budget(1000)

last_only = Probe(999, 1000)
assert first_bad_revision(history, last_only) == "rev-000999", (
    "a regression in the newest revision was missed"
)
assert last_only.calls <= budget(1000)

single_good = Probe(10**9, 1)
assert first_bad_revision(revisions(1), single_good) is None
single_bad = Probe(0, 1)
assert first_bad_revision(revisions(1), single_bad) == "rev-000000"

# --- every position is found, not just the sampled ones ---------------------
history = revisions(64)
for index in range(64):
    probe = Probe(index, 64)
    assert first_bad_revision(history, probe) == f"rev-{index:06d}", (
        f"a regression at position {index} was misattributed"
    )
    assert probe.calls <= budget(64), (
        f"position {index} used {probe.calls} builds"
    )

try:
    first_bad_revision([], lambda revision: True)
except ValueError:
    pass
else:
    raise AssertionError("an empty history was accepted")
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function group_failures(reports) that "
            "collapses repeated crash reports into distinct defects. Each "
            "report is a mapping with keys test and traceback, the latter "
            "being the text of a Python traceback. Compute a signature for a "
            "traceback as follows: take every line of the form '  File "
            "\"<path>\", line <n>, in <function>' in order, reduce each to "
            "'<basename of path>:<function>', and join those with ';'; take "
            "the last non-empty line as the exception line and use the text "
            "before its first ':' as the exception type, or the whole "
            "stripped line when it has no ':'; the signature is the exception "
            "type, then '|', then the joined frames. Line numbers and "
            "directory names must not appear in a signature, because they "
            "differ between runs of one defect. Return a list of mappings "
            "with keys signature, count and tests, where tests holds the "
            "sorted distinct test names, ordered by descending count and then "
            "by ascending signature. Raise ValueError if reports is empty, if "
            "a report is missing either key, or if a traceback has no frame "
            "line."
        ),
        validator=LOAD_CANDIDATE + require("group_failures") + r'''
def trace(directory, line_numbers, functions, files, message):
    lines = ["Traceback (most recent call last):"]
    for number, function, name in zip(line_numbers, functions, files):
        lines.append(
            f'  File "{directory}/{name}", line {number}, in {function}'
        )
        lines.append("    the source line, which is incidental")
    lines.append(message)
    return "\n".join(lines)


# One defect, reported from two runs. Everything incidental differs: the
# temporary directory, every line number, and the source lines. Only the file
# basenames, the call chain and the exception type are the defect.
first = trace("/tmp/build-a91f/src", [140, 62, 17],
              ["handle", "load", "decode"],
              ["service.py", "store.py", "codec.py"],
              "ValueError: invalid literal for int() with base 10: 'x'")
second = trace("/var/ci/run-77/src", [141, 63, 19],
               ["handle", "load", "decode"],
               ["service.py", "store.py", "codec.py"],
               "ValueError: invalid literal for int() with base 10: 'y'")
# A genuinely different defect: same entry point, different failing frame.
other = trace("/tmp/build-a91f/src", [140, 88],
              ["handle", "render"],
              ["service.py", "view.py"],
              "KeyError: 'template'")

groups = group_failures([
    {"test": "test_alpha", "traceback": first},
    {"test": "test_beta", "traceback": second},
    {"test": "test_gamma", "traceback": other},
])
assert len(groups) == 2, (
    f"two defects across three reports, got {len(groups)} groups: "
    f"{[group['signature'] for group in groups]}"
)

top = groups[0]
assert top["count"] == 2, f"the repeated defect has count {top['count']}"
assert top["tests"] == ["test_alpha", "test_beta"], (
    f"the repeated defect lists {top['tests']}"
)
assert top["signature"] == (
    "ValueError|service.py:handle;store.py:load;codec.py:decode"
), f"signature is {top['signature']!r}"
assert groups[1]["signature"] == "KeyError|service.py:handle;view.py:render"
assert groups[1]["count"] == 1 and groups[1]["tests"] == ["test_gamma"]

# --- no digits or directories survive into a signature ----------------------
for group in groups:
    assert "/" not in group["signature"], (
        f"a directory reached the signature: {group['signature']!r}"
    )
    assert not any(character.isdigit() for character in group["signature"]), (
        f"a line number reached the signature: {group['signature']!r}"
    )

# --- one test reported twice counts twice but is named once -----------------
repeated = group_failures([
    {"test": "test_alpha", "traceback": first},
    {"test": "test_alpha", "traceback": second},
])
assert repeated[0]["count"] == 2, "a repeat by one test was not counted"
assert repeated[0]["tests"] == ["test_alpha"], (
    f"a test named twice must appear once: {repeated[0]['tests']}"
)

# --- ordering: count first, then signature ----------------------------------
ordered = group_failures([
    {"test": "t1", "traceback": other},
    {"test": "t2", "traceback": first},
    {"test": "t3", "traceback": second},
])
assert [group["count"] for group in ordered] == [2, 1], (
    "groups are not ordered by descending count"
)

zeta = trace("/tmp/x", [1], ["z"], ["z.py"], "ZeroDivisionError: division")
alpha = trace("/tmp/x", [1], ["a"], ["a.py"], "ArithmeticError: bad")
tied = group_failures([
    {"test": "t1", "traceback": zeta},
    {"test": "t2", "traceback": alpha},
])
assert tied[0]["signature"].startswith("ArithmeticError|"), (
    f"equal counts are not broken by ascending signature: "
    f"{[group['signature'] for group in tied]}"
)

# --- an exception line with no message, and one with no colon ---------------
bare = trace("/tmp/x", [3], ["run"], ["m.py"], "KeyboardInterrupt")
assert group_failures([{"test": "t", "traceback": bare}])[0]["signature"] == (
    "KeyboardInterrupt|m.py:run"
)

for bad in ([],
            [{"test": "t"}],
            [{"traceback": first}],
            [{"test": "t", "traceback": "ValueError: no frames here"}]):
    try:
        group_failures(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"a malformed report set was accepted: {bad!r}")
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function rank_suspicious_lines(coverage, "
            "outcomes) performing spectrum-based fault localisation. coverage "
            "maps a test name to the set of source line numbers that test "
            "executed, and outcomes maps a test name to True when it passed "
            "and False when it failed. For a line, let ef be the number of "
            "failing tests that executed it, ep the number of passing tests "
            "that executed it, and total_failing the number of failing tests "
            "overall. Its Ochiai suspiciousness is "
            "ef / sqrt(total_failing * (ef + ep)). Return a list of "
            "(line, score) pairs ordered by descending score and then "
            "ascending line, containing only lines with ef greater than zero: "
            "a line no failing test ran cannot be the cause, and reporting it "
            "buries the lines that can. Raise ValueError if outcomes is "
            "empty, if no test failed, if coverage and outcomes do not cover "
            "exactly the same test names, or if any line number is not a "
            "positive integer."
        ),
        validator=LOAD_CANDIDATE + require("rank_suspicious_lines") + r'''
import math

# Line 10 runs in both failing tests and neither passing one -- the planted
# defect. Line 20 runs everywhere, so it is common code, not a cause. Line 99
# runs only in passing tests and must not be ranked at all.
coverage = {
    "test_a": {10, 20, 30},
    "test_b": {10, 20, 40},
    "test_c": {20, 99},
    "test_d": {20, 30, 99},
}
outcomes = {"test_a": False, "test_b": False, "test_c": True, "test_d": True}

ranked = rank_suspicious_lines(coverage, outcomes)
lines = [line for line, _ in ranked]
scores = dict(ranked)

assert lines[0] == 10, f"the planted defect did not rank first: {ranked!r}"
assert abs(scores[10] - 1.0) < 1e-9, f"line 10 scored {scores[10]}, not 1.0"

# ef=2, ep=2, total_failing=2 -> 2 / sqrt(2 * 4)
assert abs(scores[20] - 2 / math.sqrt(8)) < 1e-9, (
    f"line 20 scored {scores[20]}"
)
# ef=1, ep=1, total_failing=2 -> 1 / sqrt(2 * 2)
assert abs(scores[30] - 1 / math.sqrt(4)) < 1e-9, f"line 30 scored {scores[30]}"
# ef=1, ep=0, total_failing=2 -> 1 / sqrt(2 * 1)
assert abs(scores[40] - 1 / math.sqrt(2)) < 1e-9, f"line 40 scored {scores[40]}"

assert 99 not in scores, (
    "line 99 is executed only by passing tests and cannot be the cause, but "
    "it was ranked"
)
assert set(lines) == {10, 20, 30, 40}, f"ranked lines are {lines}"

assert all(scores[lines[index]] >= scores[lines[index + 1]] - 1e-12
           for index in range(len(lines) - 1)), (
    f"the ranking is not in descending score order: {ranked!r}"
)

# --- equal scores break by ascending line -----------------------------------
tied = rank_suspicious_lines(
    {"test_a": {7, 3, 5}, "test_b": {7, 3, 5}},
    {"test_a": False, "test_b": True},
)
assert [line for line, _ in tied] == [3, 5, 7], (
    f"equal scores were not broken by ascending line: {tied!r}"
)

# --- a line executed by every failing test and no passing test wins ---------
single = rank_suspicious_lines({"only": {1, 2}}, {"only": False})
assert [line for line, _ in single] == [1, 2]
assert all(abs(score - 1.0) < 1e-9 for _, score in single)

# --- stated error behaviour -------------------------------------------------
missing_outcome = (lambda: rank_suspicious_lines(
    {"test_a": {1}, "test_b": {2}}, {"test_a": False}))
extra_outcome = (lambda: rank_suspicious_lines(
    {"test_a": {1}}, {"test_a": False, "test_b": True}))
for bad in (lambda: rank_suspicious_lines({}, {}),
            lambda: rank_suspicious_lines({"t": {1}}, {"t": True}),
            missing_outcome,
            extra_outcome,
            lambda: rank_suspicious_lines({"t": {0}}, {"t": False}),
            lambda: rank_suspicious_lines({"t": {-3}}, {"t": False}),
            lambda: rank_suspicious_lines({"t": {1.5}}, {"t": False})):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid spectrum was accepted")
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python function rename_local(source, function_name, "
            "old_name, new_name) that renames a local of one top-level "
            "function and returns the rewritten module source. Rename the "
            "function's parameters and its variable references, and nothing "
            "else: an attribute called old_name, a string literal containing "
            "old_name, a keyword argument name in a call, and every use of "
            "old_name in any other function must survive unchanged. The "
            "rewritten module must still parse and behave identically. Raise "
            "ValueError if no top-level function called function_name exists, "
            "if new_name is not a valid Python identifier or is a reserved "
            "keyword, if old_name is not bound as a local or parameter of "
            "that function, if new_name is already used as a name anywhere in "
            "that function, if old_name is declared global or nonlocal there, "
            "or if the function contains a nested function or class "
            "definition."
        ),
        validator=LOAD_CANDIDATE + require("rename_local") + r'''
import ast

SOURCE = (
    "HEADER = 'total'\n"
    "\n"
    "\n"
    "def compute(total, rows):\n"
    "    for row in rows:\n"
    "        total = total + row.total\n"
    "    label = 'total'\n"
    "    return {'total': total, 'label': label, 'via': helper(total=total)}\n"
    "\n"
    "\n"
    "def helper(total):\n"
    "    return total * 2\n"
)


def function_of(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"the rewritten module lost the function {name!r}")


rewritten = rename_local(SOURCE, "compute", "total", "accumulator")
assert isinstance(rewritten, str), "rename_local must return source text"
try:
    tree = ast.parse(rewritten)
except SyntaxError as error:
    raise AssertionError(f"the rewritten module does not parse: {error}")

compute = function_of(tree, "compute")

# --- the rename happened ----------------------------------------------------
parameters = [argument.arg for argument in compute.args.args]
assert parameters == ["accumulator", "rows"], (
    f"the parameter list is {parameters}; the signature was not renamed"
)
loaded = {node.id for node in ast.walk(compute) if isinstance(node, ast.Name)}
assert "accumulator" in loaded, "no renamed variable reference remains"
assert "total" not in loaded, (
    "a variable reference to the old name survived in the function body"
)

# --- and stopped exactly where it should ------------------------------------
attributes = {node.attr for node in ast.walk(compute)
              if isinstance(node, ast.Attribute)}
assert attributes == {"total"}, (
    f"the attribute row.total was rewritten to {attributes}; an attribute is "
    "a different name in a different namespace"
)
constants = [node.value for node in ast.walk(compute)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)]
assert constants.count("total") == 2, (
    f"string literals were rewritten: {constants}; a rename must not edit "
    "data"
)
keywords = {keyword.arg for node in ast.walk(compute)
            if isinstance(node, ast.Call) for keyword in node.keywords}
assert keywords == {"total"}, (
    f"the keyword argument name was rewritten to {keywords}; it names the "
    "callee's parameter, not a local of this function"
)
assert [argument.arg for argument in function_of(tree, "helper").args.args] \
    == ["total"], "another function's parameter was renamed"
assignments = [node.targets[0].id for node in tree.body
               if isinstance(node, ast.Assign)]
assert assignments == ["HEADER"], (
    f"a module-level name was rewritten: {assignments}"
)

# --- behaviour is unchanged -------------------------------------------------
class Row:
    def __init__(self, total):
        self.total = total


def evaluate(text):
    namespace = {}
    try:
        exec(compile(text, "<module>", "exec"), namespace)
        return namespace["compute"](0, [Row(3), Row(4)])
    except Exception as error:
        raise AssertionError(f"the rewritten module does not run: {error!r}")


expected = evaluate(SOURCE)
assert expected == {"total": 7, "label": "total", "via": 14}, (
    f"the validator's own fixture is wrong: {expected!r}"
)
assert evaluate(rewritten) == expected, (
    f"behaviour changed: {evaluate(rewritten)!r} != {expected!r}"
)

# --- renaming a body-only local ---------------------------------------------
relabelled = rename_local(SOURCE, "compute", "label", "caption")
relabelled_tree = ast.parse(relabelled)
names = {node.id for node in ast.walk(function_of(relabelled_tree, "compute"))
         if isinstance(node, ast.Name)}
assert "caption" in names and "label" not in names, (
    f"a body-only local was not renamed: {sorted(names)}"
)
assert evaluate(relabelled) == expected, "renaming a local changed behaviour"

# --- stated error behaviour -------------------------------------------------
NESTED = (
    "def outer(value):\n"
    "    def inner():\n"
    "        return value\n"
    "    return inner()\n"
)
GLOBAL = (
    "counter = 0\n"
    "\n"
    "\n"
    "def bump(step):\n"
    "    global counter\n"
    "    counter = counter + step\n"
    "    return counter\n"
)
for bad in (lambda: rename_local(SOURCE, "absent", "total", "x"),
            lambda: rename_local(SOURCE, "compute", "total", "2bad"),
            lambda: rename_local(SOURCE, "compute", "total", "class"),
            lambda: rename_local(SOURCE, "compute", "total", "rows"),
            lambda: rename_local(SOURCE, "compute", "total", "label"),
            lambda: rename_local(SOURCE, "compute", "missing", "x"),
            lambda: rename_local(SOURCE, "compute", "helper", "x"),
            lambda: rename_local(NESTED, "outer", "value", "amount"),
            lambda: rename_local(GLOBAL, "bump", "counter", "tally")):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an unsafe or impossible rename was accepted")
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function first_difference(expected, actual) "
            "that reports where two nested structures diverge, the way a test "
            "framework explains a failed equality assertion. Values are built "
            "from dicts with string keys, lists, and the scalars str, int, "
            "float, bool and None. Return None when the structures match. "
            "Otherwise return a mapping with keys path, expected and actual, "
            "where path is a list of dict keys and list indices leading to "
            "the first divergence. Two values differ whenever type(expected) "
            "is not type(actual), so True and 1 differ and 1 and 1.0 differ; "
            "comparing them with == alone reports a match and hides a real "
            "bug. Traverse dict keys in sorted order and lists by ascending "
            "index. When a key is present on only one side, or one list is a "
            "prefix of the other, report the missing side as the string "
            "'<missing>' at the path of the absent element. Raise ValueError "
            "if either structure contains a dict with a non-string key or a "
            "value of any other type."
        ),
        validator=LOAD_CANDIDATE + require("first_difference") + r'''
assert first_difference({"a": [1, {"b": "x"}]}, {"a": [1, {"b": "x"}]}) is None
assert first_difference([], []) is None
assert first_difference(None, None) is None

# --- the first difference, by sorted keys then ascending index --------------
report = first_difference(
    {"alpha": {"deep": [1, 2, 3]}, "beta": 1},
    {"alpha": {"deep": [1, 9, 3]}, "beta": 2},
)
assert report is not None, "a difference was not reported"
assert report["path"] == ["alpha", "deep", 1], f"path is {report['path']}"
assert report["expected"] == 2 and report["actual"] == 9, f"got {report!r}"

# 'alpha' sorts before 'beta', so the beta difference must not be the one
# reported even though it is shallower.
report = first_difference({"beta": 1, "alpha": 1}, {"beta": 2, "alpha": 2})
assert report["path"] == ["alpha"], (
    f"dict keys were not traversed in sorted order: {report['path']}"
)

# --- TYPE IS PART OF THE VALUE ---------------------------------------------
# True == 1 and 1 == 1.0 in Python, so an implementation built on == alone
# returns None for both of these and hides the bug the test was written for.
report = first_difference({"flag": True}, {"flag": 1})
assert report is not None, "True and 1 were reported as equal"
assert report["path"] == ["flag"]
assert report["expected"] is True and report["actual"] == 1

report = first_difference([1], [1.0])
assert report is not None, "1 and 1.0 were reported as equal"
assert report["path"] == [0]

report = first_difference({"a": 1}, {"a": "1"})
assert report is not None and report["path"] == ["a"]

# --- a container replaced by a scalar --------------------------------------
report = first_difference({"a": [1, 2]}, {"a": "nope"})
assert report["path"] == ["a"], f"path is {report['path']}"
assert report["expected"] == [1, 2] and report["actual"] == "nope"

# --- missing keys and short lists ------------------------------------------
report = first_difference({"a": 1, "b": 2}, {"a": 1})
assert report["path"] == ["b"], f"path is {report['path']}"
assert report["expected"] == 2 and report["actual"] == "<missing>"

report = first_difference({"a": 1}, {"a": 1, "b": 2})
assert report["path"] == ["b"]
assert report["expected"] == "<missing>" and report["actual"] == 2

report = first_difference([1, 2, 3], [1, 2])
assert report["path"] == [2], f"path is {report['path']}"
assert report["expected"] == 3 and report["actual"] == "<missing>"

report = first_difference([1, 2], [1, 2, 3])
assert report["path"] == [2]
assert report["expected"] == "<missing>" and report["actual"] == 3

# A difference inside the common prefix outranks the length difference.
report = first_difference([1, 5, 3], [1, 6])
assert report["path"] == [1] and report["actual"] == 6, f"got {report!r}"

# --- the root itself can be the difference ----------------------------------
report = first_difference(1, 2)
assert report["path"] == [] and report["expected"] == 1

# --- stated error behaviour -------------------------------------------------
for bad in (lambda: first_difference({1: "a"}, {1: "a"}),
            lambda: first_difference({"a": {2: "b"}}, {"a": {}}),
            lambda: first_difference({"a": (1, 2)}, {"a": (1, 2)}),
            lambda: first_difference([set()], [set()])):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an unsupported structure was accepted")
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function classify_test_history(history, "
            "window) that decides which tests to quarantine. history maps a "
            "test name to a list of the strings 'pass' and 'fail', ordered "
            "oldest first. Consider only the last window runs of each test. "
            "Count the transitions in that slice, meaning positions where an "
            "outcome differs from the one before it. Classify a test as "
            "'flaky' when it has at least two transitions; otherwise as "
            "'failing' when the newest outcome in the slice is 'fail'; "
            "otherwise as 'passing'. A test that fails every run and a test "
            "that passed and then began failing both have fewer than two "
            "transitions and are real failures, not flakes, and quarantining "
            "them would hide a genuine regression. Return a mapping with keys "
            "classes, mapping every test name to its label, and quarantine, "
            "the sorted names of the flaky tests. Raise ValueError if history "
            "is empty, if window is less than 2, if any outcome is not "
            "exactly 'pass' or 'fail', or if any test has fewer than two "
            "recorded runs."
        ),
        validator=LOAD_CANDIDATE + require("classify_test_history") + r'''
def runs(pattern):
    return ["fail" if character == "F" else "pass" for character in pattern]


history = {
    "test_stable":     runs("PPPPPPPPPP"),
    "test_broken":     runs("FFFFFFFFFF"),
    "test_regressed":  runs("PPPPPPFFFF"),
    "test_flaky":      runs("PFPFPFPFPF"),
    "test_fixed":      runs("FFFFFPPPPP"),
    "test_settled":    runs("PFPFPPPPPP"),
}
report = classify_test_history(history, 10)
classes = report["classes"]

assert classes["test_stable"] == "passing"
assert classes["test_flaky"] == "flaky"

# THE DISTINCTION THAT MATTERS. Both of these fail right now, and neither is
# flaky: quarantining a regression is how a real break gets ignored for weeks.
assert classes["test_broken"] == "failing", (
    f"a test that has never passed was classified {classes['test_broken']!r}"
)
assert classes["test_regressed"] == "failing", (
    f"a clean regression was classified {classes['test_regressed']!r}; one "
    "transition is a change, not a flake"
)
assert classes["test_fixed"] == "passing", (
    f"a fixed test was classified {classes['test_fixed']!r}"
)
assert classes["test_settled"] == "flaky", (
    "three transitions inside the window is flaky"
)
assert report["quarantine"] == ["test_flaky", "test_settled"], (
    f"quarantine is {report['quarantine']}"
)

# --- the window is what makes the answer current ----------------------------
# The same history over the last four runs: the old flapping is out of scope
# and only the recent behaviour counts.
recent = classify_test_history(history, 4)
assert recent["classes"]["test_settled"] == "passing", (
    "a test that settled down is still quarantined; the window was ignored"
)
assert recent["classes"]["test_regressed"] == "failing"
assert recent["classes"]["test_flaky"] == "flaky"
assert recent["quarantine"] == ["test_flaky"], (
    f"quarantine over a 4-run window is {recent['quarantine']}"
)

# A window longer than the history uses everything there is.
assert classify_test_history({"t": runs("PFP")}, 100)["classes"]["t"] == "flaky"

# --- boundaries -------------------------------------------------------------
assert classify_test_history({"t": runs("PF")}, 2)["classes"]["t"] == "failing"
assert classify_test_history({"t": runs("FP")}, 2)["classes"]["t"] == "passing"
assert classify_test_history({"t": runs("FF")}, 2)["classes"]["t"] == "failing"
assert set(classify_test_history(history, 10)["classes"]) == set(history), (
    "every test must be classified"
)

for bad in (lambda: classify_test_history({}, 10),
            lambda: classify_test_history({"t": runs("PF")}, 1),
            lambda: classify_test_history({"t": runs("PF")}, 0),
            lambda: classify_test_history({"t": ["pass", "PASS"]}, 4),
            lambda: classify_test_history({"t": ["pass", "error"]}, 4),
            lambda: classify_test_history({"t": ["pass"]}, 4),
            lambda: classify_test_history({"t": []}, 4)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid history was accepted")
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function apply_patch(lines, hunks) that "
            "applies a set of edits to a file. lines is a list of the "
            "original lines without terminators. Each hunk is a mapping with "
            "keys start, a 1-based line number in the ORIGINAL file, remove, "
            "the lines expected to be there, and insert, the lines to put in "
            "their place. Return the patched list, leaving lines unmodified. "
            "Every start refers to the original numbering, so applying one "
            "hunk must not shift where a later hunk lands. Verify each hunk "
            "before applying anything: raise ValueError if the original lines "
            "at a hunk's position are not exactly its remove list, if start "
            "is below 1 or above len(lines) + 1, if a hunk's removal runs "
            "past the end of the file, or if the hunks are not in strictly "
            "increasing, non-overlapping order of position. A hunk with an "
            "empty remove list is an insertion before its start line, and may "
            "target len(lines) + 1 to append. When any hunk is rejected the "
            "file must be left untouched, because a half-applied patch is "
            "worse than a refused one."
        ),
        validator=LOAD_CANDIDATE + require("apply_patch") + r'''
ORIGINAL = ["alpha", "beta", "gamma", "delta", "epsilon"]


def hunk(start, remove, insert):
    return {"start": start, "remove": list(remove), "insert": list(insert)}


# --- positions are original positions, not running ones ---------------------
# Hunk 1 makes the file one line longer. An implementation that edits in place
# and keeps walking forward then reads hunk 2's context at the wrong offset --
# it lands on "gamma" instead of "delta" -- and this is the defect that makes
# a patcher look correct on a single-hunk test and corrupt real files.
source = list(ORIGINAL)
patched = apply_patch(source, [
    hunk(2, ["beta"], ["BETA", "BETA-EXTRA"]),
    hunk(4, ["delta"], []),
    hunk(6, [], ["zeta"]),
])
assert patched == ["alpha", "BETA", "BETA-EXTRA", "gamma", "epsilon", "zeta"], (
    f"patched file is {patched!r}"
)
assert source == ORIGINAL, "apply_patch mutated the caller's list"

# --- single hunks -----------------------------------------------------------
assert apply_patch(ORIGINAL, []) == ORIGINAL, "an empty patch changed the file"
assert apply_patch(ORIGINAL, []) is not ORIGINAL, (
    "an empty patch returned the caller's own list"
)
assert apply_patch(ORIGINAL, [hunk(1, [], ["first"])])[0] == "first"
assert apply_patch(ORIGINAL, [hunk(1, ORIGINAL, [])]) == []
assert apply_patch(ORIGINAL, [hunk(6, [], ["tail"])])[-1] == "tail"
assert apply_patch(ORIGINAL, [hunk(3, ["gamma", "delta"], ["G"])]) == [
    "alpha", "beta", "G", "epsilon"
]

# --- a stale patch is refused, and refused whole ----------------------------
stale = [hunk(2, ["beta"], ["B"]), hunk(4, ["WRONG"], ["D"])]
try:
    apply_patch(ORIGINAL, stale)
except ValueError:
    pass
else:
    raise AssertionError(
        "a hunk whose context does not match the file was applied; the patch "
        "was written against a different version of it"
    )
assert ORIGINAL == ["alpha", "beta", "gamma", "delta", "epsilon"], (
    "a rejected patch left the file half-applied"
)

for bad in ([hunk(0, [], ["x"])],
            [hunk(7, [], ["x"])],
            [hunk(5, ["epsilon", "beyond"], ["x"])],
            [hunk(4, ["delta"], ["D"]), hunk(2, ["beta"], ["B"])],
            [hunk(2, ["beta", "gamma"], ["B"]), hunk(3, ["gamma"], ["G"])],
            [hunk(2, [], ["x"]), hunk(2, [], ["y"])],
            [hunk(2, ["gamma"], ["G"])]):
    try:
        apply_patch(ORIGINAL, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"an invalid patch was accepted: {bad!r}")
''',
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python function extract_function(source, "
            "function_name, start_line, end_line, new_name) that performs the "
            "extract-function refactoring on Python source and returns the "
            "new source as a string. start_line and end_line are 1-based and "
            "inclusive and refer to lines of source. The statements they "
            "cover must be moved out of function_name into a new "
            "module-level function called new_name, defined immediately "
            "before function_name, and replaced in place by a call to it.\n\n"
            "The new function's parameters are exactly the local names the "
            "moved statements read before assigning them, sorted "
            "alphabetically. It returns exactly the local names the moved "
            "statements assign that are still read after the range inside "
            "function_name, sorted alphabetically: no return statement when "
            "there are none, the bare value when there is one, and a tuple "
            "otherwise. The replacement statement rebinds those same names in "
            "that order, and is a bare expression statement when there are "
            "none. Raise ValueError if the range is not a whole run of "
            "consecutive statements at the top level of function_name's body, "
            "or if the moved statements contain return, yield, break, "
            "continue, global, or nonlocal."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("extract_function") + r'''
import ast


def namespace(source):
    scope = {}
    exec(compile(source, "<case>", "exec"), scope)
    return scope


def top_level_function(source, name):
    tree = ast.parse(source)
    found = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(found) == 1, (
        f"expected exactly one module-level def {name}, found {len(found)}"
    )
    return found[0]


COMPUTE = "\n".join([
    "def compute(values, factor):",          # 1
    "    total = 0",                          # 2
    "    for value in values:",               # 3
    "        total += value",                 # 4
    "    scaled = total * factor",            # 5
    "    offset = scaled + len(values)",      # 6
    "    return offset - total",              # 7
])

# --- one returned name, and a name that is assigned before it is read -------
# `scaled` is assigned on line 5 and read on line 6, so it is written before
# it is read inside the range: it is a local of the new function, not a
# parameter. Treating every name the range mentions as a parameter is the
# usual way to get this wrong, and it is caught here.
result = extract_function(COMPUTE, "compute", 5, 6, "derive")
derive = top_level_function(result, "derive")
parameters = [argument.arg for argument in derive.args.args]
assert parameters == ["factor", "total", "values"], (
    f"expected parameters ['factor', 'total', 'values'], got {parameters!r}"
)
tree = ast.parse(result)
order = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
assert order == ["derive", "compute"], (
    f"the extracted function must precede its caller; got {order!r}"
)
original, refactored = namespace(COMPUTE), namespace(result)
for values, factor in (([1, 2, 3], 2), ([], 5), ([7], -1), ([4, 4, 4], 0)):
    expected = original["compute"](values, factor)
    actual = refactored["compute"](values, factor)
    assert actual == expected, (
        f"compute({values!r}, {factor!r}) changed: {expected!r} -> {actual!r}"
    )
body = top_level_function(result, "compute").body
assert not any(
    isinstance(node, ast.Assign)
    and any(getattr(t, "id", None) == "scaled" for t in node.targets)
    for node in body
), "the moved statements are still present in compute"

# --- two returned names, so the call site unpacks a tuple -------------------
ANALYSE = "\n".join([
    "def analyse(data):",                     # 1
    "    low = min(data)",                    # 2
    "    high = max(data)",                   # 3
    "    span = high - low",                  # 4
    "    return span, low, high",             # 5
])
result = extract_function(ANALYSE, "analyse", 2, 3, "bounds")
bounds = top_level_function(result, "bounds")
parameters = [argument.arg for argument in bounds.args.args]
assert parameters == ["data"], f"expected ['data'], got {parameters!r}"
original, refactored = namespace(ANALYSE), namespace(result)
for data in ([1, 2, 3], [5], [-4, 0, 4]):
    expected = original["analyse"](data)
    actual = refactored["analyse"](data)
    assert actual == expected, (
        f"analyse({data!r}) changed: {expected!r} -> {actual!r}"
    )
returned = bounds.body[-1]
assert isinstance(returned, ast.Return), "bounds must return its two names"
assert isinstance(returned.value, ast.Tuple), (
    "two returned names must be returned as a tuple"
)
names = [getattr(element, "id", None) for element in returned.value.elts]
assert names == ["high", "low"], (
    f"returned names must be alphabetical: expected ['high', 'low'], "
    f"got {names!r}"
)

# --- nothing is read afterwards, so the call is a bare statement ------------
LOG_ALL = "\n".join([
    "def log_all(items, sink):",              # 1
    "    count = 0",                          # 2
    "    for item in items:",                 # 3
    "        sink.append(item)",              # 4
    "    return len(items)",                  # 5
])
result = extract_function(LOG_ALL, "log_all", 3, 4, "drain")
drain = top_level_function(result, "drain")
parameters = [argument.arg for argument in drain.args.args]
assert parameters == ["items", "sink"], (
    f"expected ['items', 'sink'], got {parameters!r}"
)
assert not any(isinstance(node, ast.Return) for node in ast.walk(drain)), (
    "nothing the moved statements assign is read afterwards, so drain must "
    "not return anything"
)
original, refactored = namespace(LOG_ALL), namespace(result)
for items in ([1, 2], [], ["a"]):
    expected_sink, actual_sink = [], []
    expected = original["log_all"](items, expected_sink)
    actual = refactored["log_all"](items, actual_sink)
    assert (expected, expected_sink) == (actual, actual_sink), (
        f"log_all({items!r}) changed: {(expected, expected_sink)!r} -> "
        f"{(actual, actual_sink)!r}"
    )

# --- ranges that are not whole top-level statements are refused -------------
for start, end, why in (
        (3, 3, "the for statement spans lines 3-4, so 3-3 is half of it"),
        (4, 4, "line 4 is nested inside the loop, not at the top level"),
        (7, 7, "the range contains a return"),
        (6, 7, "the range ends inside a return"),
        (2, 9, "the range runs past the end of the function"),
):
    try:
        extract_function(COMPUTE, "compute", start, end, "extracted")
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted lines {start}-{end}: {why}")
''',
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python function inline_variable(source, "
            "function_name, variable) that performs the inline-variable "
            "refactoring and returns the new source as a string. Inside "
            "function_name, delete the single assignment to variable and "
            "replace every later read of it with the assignment's "
            "right-hand-side expression, adding parentheses wherever they are "
            "needed so that the value of every enclosing expression is "
            "unchanged.\n\n"
            "Raise ValueError if variable is a parameter of function_name, is "
            "never assigned in it, is assigned more than once, is the target "
            "of an augmented assignment, if the right-hand side contains a "
            "call, await, comprehension, or lambda, or if any name the right-"
            "hand side reads is rebound anywhere between the assignment and a "
            "read of variable -- in each of those cases inlining would not "
            "preserve behaviour."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("inline_variable") + r'''
import ast


def namespace(source):
    scope = {}
    exec(compile(source, "<case>", "exec"), scope)
    return scope


def reads_variable(source, function_name, variable):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return any(
                isinstance(inner, ast.Name) and inner.id == variable
                for inner in ast.walk(node)
            )
    raise AssertionError(f"{function_name} is missing from the result")


# --- precedence: the inlined expression binds looser than its context -------
# Substituting the text of `a + b` into `weight * c - weight` without
# parentheses yields `a + b * c - a + b`, which evaluates differently. The
# check is by value over many operands, so a candidate cannot pass by
# reproducing one sample's arithmetic coincidence.
SCORE = "\n".join([
    "def score(a, b, c):",
    "    weight = a + b",
    "    return weight * c - weight",
])
result = inline_variable(SCORE, "score", "weight")
assert not reads_variable(result, "score", "weight"), (
    "weight is still read after being inlined"
)
original, refactored = namespace(SCORE), namespace(result)
for a in range(-3, 4):
    for b in range(-3, 4):
        for c in range(-3, 4):
            expected = original["score"](a, b, c)
            actual = refactored["score"](a, b, c)
            assert actual == expected, (
                f"score({a}, {b}, {c}) changed: {expected!r} -> {actual!r}"
            )

# --- precedence again, where unary minus and ** disagree about binding ------
# `-base ** 2` is -(base ** 2). Inlining `a - b` for base must produce
# -((a - b) ** 2), not -a - b ** 2.
POWER = "\n".join([
    "def power(a, b):",
    "    base = a - b",
    "    return -base ** 2",
])
result = inline_variable(POWER, "power", "base")
assert not reads_variable(result, "power", "base"), (
    "base is still read after being inlined"
)
original, refactored = namespace(POWER), namespace(result)
for a in range(-4, 5):
    for b in range(-4, 5):
        expected = original["power"](a, b)
        actual = refactored["power"](a, b)
        assert actual == expected, (
            f"power({a}, {b}) changed: {expected!r} -> {actual!r}"
        )

# --- a comparison chain must survive being substituted into ----------------
BETWEEN = "\n".join([
    "def between(a, b, c):",
    "    limit = a or b",
    "    return limit and c",
])
result = inline_variable(BETWEEN, "between", "limit")
original, refactored = namespace(BETWEEN), namespace(result)
for a in (0, 1, "", "x", None):
    for b in (0, 2, "", "y", None):
        for c in (0, 3, "", "z", None):
            expected = original["between"](a, b, c)
            actual = refactored["between"](a, b, c)
            assert actual == expected, (
                f"between({a!r}, {b!r}, {c!r}) changed: {expected!r} -> "
                f"{actual!r}"
            )

# --- the cases where inlining is not behaviour-preserving -------------------
REFUSALS = (
    ("\n".join([
        "def twice(a):",
        "    x = a + 1",
        "    x = x + 1",
        "    return x",
    ]), "twice", "x", "x is assigned twice"),
    ("\n".join([
        "def counted(a):",
        "    x = len(a)",
        "    return x + x",
    ]), "counted", "x", "the right-hand side calls len, so inlining would "
       "evaluate it twice"),
    ("\n".join([
        "def rebound(a):",
        "    x = a + 1",
        "    a = a * 2",
        "    return x + a",
    ]), "rebound", "x", "a is rebound between the assignment and the read"),
    ("\n".join([
        "def parameter(x):",
        "    return x + 1",
    ]), "parameter", "x", "x is a parameter, not a local assignment"),
    ("\n".join([
        "def missing(a):",
        "    return a + 1",
    ]), "missing", "x", "x is never assigned"),
    ("\n".join([
        "def accumulated(a):",
        "    x = a",
        "    x += 1",
        "    return x",
    ]), "accumulated", "x", "x is the target of an augmented assignment"),
)
for source, function_name, variable, why in REFUSALS:
    try:
        inline_variable(source, function_name, variable)
    except ValueError:
        pass
    else:
        raise AssertionError(f"inlined {variable} in {function_name}: {why}")
''',
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement a Python function minimize_test_suite(coverage) that "
            "performs coverage-preserving test-suite minimization. coverage "
            "maps a test name to the collection of requirement ids that test "
            "covers. Return a list of test names whose covered ids union to "
            "exactly the same set as the whole suite, of minimum possible "
            "length. When several subsets of that length cover everything, "
            "return the one whose sorted name list is smallest "
            "lexicographically. The returned list itself must be sorted by "
            "name. Tests that cover nothing are never included. Return the "
            "empty list when the suite covers nothing at all. The suite "
            "contains at most 14 tests."
        ),
        timeout_seconds=90.0,
        validator=LOAD_CANDIDATE + require("minimize_test_suite") + r'''
import itertools


def reference(coverage):
    """Brute force, which is affordable at the 14-test bound the prompt sets."""
    universe = set()
    for covered in coverage.values():
        universe |= set(covered)
    if not universe:
        return []
    names = sorted(coverage)
    for size in range(1, len(names) + 1):
        best = None
        for combination in itertools.combinations(names, size):
            union = set()
            for name in combination:
                union |= set(coverage[name])
            if union == universe:
                candidate = sorted(combination)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            return best
    raise AssertionError("unreachable: the whole suite always covers")


def check(coverage, note):
    expected = reference(coverage)
    actual = minimize_test_suite(coverage)
    assert isinstance(actual, list), f"expected a list, got {type(actual)}"
    assert actual == sorted(actual), f"result is not sorted by name: {actual!r}"
    universe = set()
    for covered in coverage.values():
        universe |= set(covered)
    union = set()
    for name in actual:
        assert name in coverage, f"{name!r} is not a test in the suite"
        union |= set(coverage[name])
    assert union == universe, (
        f"{note}: coverage was lost; missing {sorted(universe - union)}"
    )
    assert len(actual) == len(expected), (
        f"{note}: {len(actual)} tests returned but {len(expected)} suffice "
        f"({expected!r})"
    )
    assert actual == expected, (
        f"{note}: expected the lexicographically smallest minimum "
        f"{expected!r}, got {actual!r}"
    )


# --- greedy picks the big set first and then needs three more --------------
# The largest-first heuristic takes `broad` (4 ids) and must then add three
# singletons, for four tests. The optimum is the two `half` tests. A candidate
# that implements textbook greedy set cover fails exactly here.
check({
    "broad": ["a", "b", "c", "d"],
    "half_one": ["a", "b", "e", "f"],
    "half_two": ["c", "d", "g", "h"],
    "tiny_e": ["e"],
    "tiny_f": ["f"],
    "tiny_g": ["g"],
    "tiny_h": ["h"],
}, "greedy is suboptimal here")

# --- a tie broken by name, not by insertion order --------------------------
check({
    "zulu": ["x", "y"],
    "alpha": ["x", "y"],
    "mike": ["x", "y"],
}, "three equivalent tests: the answer is the alphabetically first")

# --- redundant tests, and one that covers nothing --------------------------
check({
    "empty": [],
    "one": ["p"],
    "two": ["q"],
    "both": ["p", "q"],
    "superset": ["p", "q"],
}, "a test covering nothing must never be selected")

# --- every test is essential -----------------------------------------------
check({
    "t1": ["r1"],
    "t2": ["r2"],
    "t3": ["r3"],
    "t4": ["r4"],
}, "disjoint coverage means the whole suite is minimal")

# --- nothing is covered at all ---------------------------------------------
assert minimize_test_suite({"a": [], "b": []}) == [], (
    "a suite that covers nothing minimizes to no tests"
)
assert minimize_test_suite({}) == [], "an empty suite minimizes to no tests"

# --- duplicate ids inside one test must not change the answer --------------
check({
    "dupes": ["a", "a", "a", "b"],
    "other": ["b", "c"],
    "third": ["c", "a"],
}, "repeated ids within a test are still one requirement")

# --- overlapping mid-size sets where the optimum is three ------------------
check({
    "s1": ["1", "2", "3"],
    "s2": ["3", "4", "5"],
    "s3": ["5", "6", "7"],
    "s4": ["7", "8", "1"],
    "s5": ["2", "4", "6"],
    "s6": ["8", "3", "5"],
}, "a denser instance where the optimum is not the greedy prefix")
''',
    ),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement a Python function mutation_survivors(source, run_tests) "
            "that performs mutation testing. Produce one mutant per mutable "
            "operator occurrence in source, applying exactly these swaps and "
            "no others: binary + becomes -, binary - becomes +, < becomes <=, "
            "<= becomes <, > becomes >=, >= becomes >, == becomes !=, and != "
            "becomes ==. Unary minus is not a binary operator and is never "
            "mutated.\n\n"
            "For each mutant, execute it in a fresh namespace and call "
            "run_tests(namespace). The mutant is killed if run_tests raises "
            "any exception, and it survives if run_tests returns. A mutant "
            "that fails to execute at all is killed. Return the survivors as "
            "a sorted list of (line_number, original_operator, "
            "replacement_operator) tuples, with the operators written as the "
            "source symbols above. Mutants must not leak state into each "
            "other."
        ),
        timeout_seconds=90.0,
        validator=LOAD_CANDIDATE + require("mutation_survivors") + r'''
SOURCE = "\n".join([
    "def total(items, bonus):",               # 1
    "    result = 0",                         # 2
    "    for item in items:",                 # 3
    "        result = result + item",         # 4
    "    return result + bonus",              # 5
])


def weak_tests(namespace):
    """Never passes a non-zero bonus, so the line 5 mutant cannot be killed."""
    assert namespace["total"]([1, 2], 0) == 3
    assert namespace["total"]([], 0) == 0


survivors = mutation_survivors(SOURCE, weak_tests)
assert survivors == [(5, "+", "-")], (
    "the only + this suite cannot kill is the bonus on line 5, because the "
    f"suite always passes bonus=0; got {survivors!r}"
)


def strong_tests(namespace):
    """Passes a non-zero bonus, which kills the line 5 mutant too."""
    assert namespace["total"]([1, 2], 0) == 3
    assert namespace["total"]([1, 2], 10) == 13


assert mutation_survivors(SOURCE, strong_tests) == [], (
    "a suite that exercises a non-zero bonus kills every mutant here"
)

# --- boundary operators, including a genuinely equivalent mutant ------------
CLAMP = "\n".join([
    "def clamp(value, low, high):",           # 1
    "    if value < low:",                    # 2
    "        return low",                     # 3
    "    if value > high:",                   # 4
    "        return high",                    # 5
    "    return value",                       # 6
])


def clamp_tests(namespace):
    clamp = namespace["clamp"]
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


survivors = mutation_survivors(CLAMP, clamp_tests)
assert survivors == [(2, "<", "<="), (4, ">", ">=")], (
    "both boundary mutants survive: at value == low the mutant returns low, "
    "which is the same value the original returns, so no test can kill it. "
    f"got {survivors!r}"
)

# --- a mutant that crashes counts as killed, not as a survivor -------------
DIVIDE = "\n".join([
    "def safe(numerator, denominator):",      # 1
    "    if denominator != 0:",               # 2
    "        return numerator / denominator", # 3
    "    return 0",                           # 4
])


def divide_tests(namespace):
    # The zero case runs first, so the mutant crashes before any assertion
    # can fail: this is the "a mutant that fails to execute is killed" rule
    # rather than the ordinary killed-by-assertion one.
    assert namespace["safe"](1, 0) == 0
    assert namespace["safe"](6, 3) == 2


survivors = mutation_survivors(DIVIDE, divide_tests)
assert survivors == [], (
    "mutating != to == makes safe(1, 0) divide by zero; the ZeroDivisionError "
    f"escapes run_tests, which kills the mutant. got {survivors!r}"
)

# --- unary minus is not a mutation site ------------------------------------
NEGATE = "\n".join([
    "def negate(value):",                     # 1
    "    return -value",                      # 2
])


def negate_tests(namespace):
    assert namespace["negate"](3) == -3


assert mutation_survivors(NEGATE, negate_tests) == [], (
    "unary minus must not be mutated, so this source has no mutants at all "
    "and therefore no survivors"
)

# --- mutants must be independent -------------------------------------------
# If mutants share a namespace, this counter keeps its value across runs and
# the reported survivor set changes. Each call must start from zero.
PAIR = "\n".join([
    "def pair(a, b):",                        # 1
    "    return a + b",                       # 2
])
seen = []


def recording_tests(namespace):
    seen.append(namespace["pair"](2, 2))
    assert namespace["pair"](2, 2) == 4


survivors = mutation_survivors(PAIR, recording_tests)
assert survivors == [], f"the + on line 2 is killed; got {survivors!r}"
assert seen == [0], (
    f"expected exactly one mutant execution reporting 2 - 2 = 0, got {seen!r}"
)
''',
    ),
    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement a Python function unreachable_functions(source, "
            "entry_points) that returns the sorted list of module-level "
            "function names in source that are not reachable from any name in "
            "entry_points through the static call graph.\n\n"
            "A function is reachable if an entry point reaches it through a "
            "chain of references. A reference is any mention of the "
            "function's name inside a reachable function's body, whether it "
            "is called, passed as an argument, assigned, or returned -- a "
            "name that escapes may be called later, so it is live. Names "
            "mentioned only inside unreachable functions stay unreachable, "
            "which means mutually recursive dead code is still dead. Only "
            "module-level def names are considered; methods, nested "
            "functions, and imported names are never reported. Entry points "
            "are themselves reachable. Raise ValueError if an entry point is "
            "not a module-level function in source."
        ),
        timeout_seconds=45.0,
        validator=LOAD_CANDIDATE + require("unreachable_functions") + r'''
SOURCE = "\n".join([
    "import json",
    "",
    "",
    "def main(payload):",
    "    parsed = parse(payload)",
    "    return render(parsed, formatter=as_text)",
    "",
    "",
    "def parse(payload):",
    "    return json.loads(payload)",
    "",
    "",
    "def render(parsed, formatter):",
    "    return formatter(parsed)",
    "",
    "",
    "def as_text(parsed):",
    "    return str(parsed)",
    "",
    "",
    "def as_xml(parsed):",
    "    return wrap(str(parsed))",
    "",
    "",
    "def wrap(text):",
    "    return '<v>' + text + '</v>'",
    "",
    "",
    "def legacy_a(value):",
    "    return legacy_b(value)",
    "",
    "",
    "def legacy_b(value):",
    "    return legacy_a(value)",
    "",
    "",
    "def helper(value):",
    "    return value",
    "",
    "",
    "class Service:",
    "    def parse(self, payload):",
    "        return helper(payload)",
])

result = unreachable_functions(SOURCE, ["main"])
assert result == ["as_xml", "helper", "legacy_a", "legacy_b", "wrap"], (
    "as_text is reachable because main passes it as a keyword argument even "
    "though main never calls it; wrap is dead because only as_xml mentions "
    "it; legacy_a and legacy_b are mutually recursive dead code; helper is "
    "mentioned only by a method, and a method body is not a module-level "
    f"reference. got {result!r}"
)

# --- a method's mentions do not resurrect a function ------------------------
# `helper` is referenced from Service.parse. That is not reachable from main,
# so helper stays dead. A candidate that scans the whole module for the name
# rather than walking the call graph reports helper as live and fails above.

# --- adding an entry point makes its whole cone reachable ------------------
result = unreachable_functions(SOURCE, ["main", "as_xml"])
assert result == ["helper", "legacy_a", "legacy_b"], (
    f"as_xml as an entry point makes wrap reachable; got {result!r}"
)

result = unreachable_functions(SOURCE, ["legacy_a"])
assert result == ["as_text", "as_xml", "helper", "main", "parse", "render",
                  "wrap"], (
    f"only the legacy cycle is reachable from legacy_a; got {result!r}"
)

# --- every function reachable, and none ------------------------------------
assert unreachable_functions(SOURCE, [
    "main", "as_xml", "legacy_a", "helper",
]) == [], "these entry points cover every module-level function"

assert unreachable_functions(SOURCE, []) == [
    "as_text", "as_xml", "helper", "legacy_a", "legacy_b", "main", "parse",
    "render", "wrap",
], "with no entry points every module-level function is unreachable"

# --- self-recursion is not a second reference ------------------------------
RECURSIVE = "\n".join([
    "def countdown(n):",
    "    if n <= 0:",
    "        return 0",
    "    return countdown(n - 1)",
    "",
    "",
    "def orphan(n):",
    "    return n",
])
assert unreachable_functions(RECURSIVE, ["countdown"]) == ["orphan"], (
    "a function that calls itself is reachable from itself"
)

# --- a nested def shadows nothing at module level --------------------------
NESTED = "\n".join([
    "def outer(value):",
    "    def inner(x):",
    "        return x + 1",
    "    return inner(value)",
    "",
    "",
    "def inner(x):",
    "    return x - 1",
])
assert unreachable_functions(NESTED, ["outer"]) == [], (
    "outer's body mentions the name `inner`, which is a live reference to the "
    "module-level function of that name; a nested def does not remove it"
)

# --- an entry point that is not a module-level function is an error --------
for bad in ("Service", "json", "missing", "parse_payload"):
    try:
        unreachable_functions(SOURCE, [bad])
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"{bad!r} is not a module-level function and must be rejected"
        )
''',
    ),
    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement a Python function attribute_failure(frames, "
            "owned_prefixes, helper_prefixes) that attributes a crash to the "
            "code responsible for it. frames is a traceback's frames ordered "
            "outermost first and innermost last; each is a mapping with "
            "'filename', 'lineno', and 'function'. owned_prefixes and "
            "helper_prefixes are collections of path prefixes.\n\n"
            "Return the innermost frame that is owned and is not a helper. A "
            "frame is owned when its filename starts with an owned prefix, "
            "and is a helper when it starts with a helper prefix; the longer "
            "matching prefix wins when a path matches both, so a helper "
            "directory nested inside an owned tree is still a helper. A "
            "filename that is not a real path, meaning one that starts with "
            "'<', is never owned however the prefixes are written. Return "
            "None when no frame qualifies. Return the frame object itself, "
            "not a copy."
        ),
        timeout_seconds=45.0,
        validator=LOAD_CANDIDATE + require("attribute_failure") + r'''
def frame(filename, lineno, function):
    return {"filename": filename, "lineno": lineno, "function": function}


OWNED = ["/srv/app/"]
HELPERS = ["/srv/app/tests/support/"]

# --- blame the deepest owned frame, not the deepest frame ------------------
# The innermost frame is in the standard library. Attributing there produces
# the familiar useless bug report "the crash is in json/decoder.py".
frames = [
    frame("/srv/app/main.py", 10, "handle"),
    frame("/srv/app/parse.py", 42, "load_config"),
    frame("/usr/lib/python3.13/json/__init__.py", 346, "loads"),
    frame("/usr/lib/python3.13/json/decoder.py", 337, "decode"),
]
picked = attribute_failure(frames, OWNED, HELPERS)
assert picked is frames[1], (
    "the deepest owned frame is parse.py:42; the two stdlib frames below it "
    f"are not ours. got {picked!r}"
)

# --- a helper frame is skipped in favour of the code that called it --------
# An assertion helper is always the innermost owned frame, so blaming it
# buckets every failure in the suite under one location.
frames = [
    frame("/srv/app/tests/test_orders.py", 88, "test_total"),
    frame("/srv/app/tests/support/assertions.py", 12, "assert_matches"),
]
picked = attribute_failure(frames, OWNED, HELPERS)
assert picked is frames[0], (
    "the innermost owned frame is the assertion helper; the test that called "
    f"it is the real location. got {picked!r}"
)

# --- generated code is never owned, however the prefixes are written -------
# Code the candidate produced runs in a frame with a filename like <string>.
# A prefix check that only compares strings can match '<string>' against an
# owned prefix of '' or '<'; it must not.
frames = [
    frame("/srv/app/runner.py", 5, "run"),
    frame("<string>", 3, "<module>"),
    frame("<stdin>", 1, "<module>"),
]
picked = attribute_failure(frames, OWNED, HELPERS)
assert picked is frames[0], (
    f"generated frames are never owned; got {picked!r}"
)
picked = attribute_failure(frames, ["<"], HELPERS)
assert picked is None, (
    "even an owned prefix of '<' must not make a generated frame ours; got "
    f"{picked!r}"
)
picked = attribute_failure(frames, [""], HELPERS)
assert picked is frames[0], (
    "an empty prefix owns every real path but still not the generated "
    f"frames; got {picked!r}"
)

# --- the longer matching prefix decides -----------------------------------
frames = [
    frame("/srv/app/tests/support/assertions.py", 12, "assert_matches"),
]
assert attribute_failure(frames, OWNED, HELPERS) is None, (
    "the only frame is a helper, so nothing qualifies"
)
assert attribute_failure(frames, OWNED, []) is frames[0], (
    "with no helper prefixes the same frame is the answer"
)
# Owned prefix is longer than the helper prefix here, so the frame is owned.
picked = attribute_failure(
    frames, ["/srv/app/tests/support/assertions.py"], ["/srv/app/"]
)
assert picked is frames[0], (
    "the longer matching prefix wins, so this frame is owned rather than a "
    f"helper; got {picked!r}"
)
# And the reverse.
picked = attribute_failure(
    frames, ["/srv/"], ["/srv/app/tests/"]
)
assert picked is None, (
    f"the longer helper prefix wins here, so nothing qualifies; got {picked!r}"
)

# --- nothing owned, and nothing at all ------------------------------------
frames = [frame("/usr/lib/python3.13/runpy.py", 198, "_run_module_as_main")]
assert attribute_failure(frames, OWNED, HELPERS) is None, (
    "no frame is ours"
)
assert attribute_failure([], OWNED, HELPERS) is None, (
    "an empty traceback attributes to nothing"
)

# --- several owned frames: the innermost non-helper wins -------------------
frames = [
    frame("/srv/app/a.py", 1, "a"),
    frame("/srv/app/b.py", 2, "b"),
    frame("/srv/app/tests/support/h.py", 3, "h"),
    frame("/srv/app/tests/support/i.py", 4, "i"),
]
picked = attribute_failure(frames, OWNED, HELPERS)
assert picked is frames[1], (
    f"b.py is the innermost owned non-helper frame; got {picked!r}"
)
''',
    ),
    task(
        f"{FAMILY}-0015", FAMILY,
        prompt=(
            "Implement a Python function capture_call_tree(namespace, entry, "
            "args) that instruments a module namespace so the calls between "
            "its functions are recorded, calls namespace[entry](*args), and "
            "returns the pair (result, tree).\n\n"
            "Every value in namespace that is a function must be wrapped so "
            "that calls made through the namespace are recorded, including "
            "calls functions make to each other and to themselves. A node is "
            "a dict with 'name', 'args' as a tuple, 'children' as the list of "
            "calls it made in the order they were made, and exactly one of "
            "'result' or 'error'; 'error' is the exception's class name. The "
            "tree is the node for the entry call. Restore namespace to its "
            "original functions before returning, including when the entry "
            "call raises. If the entry call raises, let the exception "
            "propagate after restoring. Raise KeyError if entry does not name "
            "a function in namespace."
        ),
        timeout_seconds=45.0,
        validator=LOAD_CANDIDATE + require("capture_call_tree") + r'''
def build(source):
    scope = {}
    exec(compile(source, "<case>", "exec"), scope)
    return scope


NESTED = "\n".join([
    "def outer(n):",
    "    return middle(n) + leaf(n)",
    "",
    "def middle(n):",
    "    return leaf(n) * 2",
    "",
    "def leaf(n):",
    "    return n + 1",
])

namespace = build(NESTED)
originals = {
    name: value for name, value in namespace.items()
    if callable(value) and not name.startswith("__")
}
result, tree = capture_call_tree(namespace, "outer", (3,))
assert result == 12, f"outer(3) is middle(3) + leaf(3) = 8 + 4; got {result!r}"
assert tree["name"] == "outer", f"the root is the entry call; got {tree!r}"
assert tree["args"] == (3,), f"args must be a tuple; got {tree['args']!r}"
assert tree["result"] == 12, f"got {tree!r}"
assert "error" not in tree, "a successful call has no 'error' key"
names = [child["name"] for child in tree["children"]]
assert names == ["middle", "leaf"], (
    f"children are recorded in call order; got {names!r}"
)
assert [c["name"] for c in tree["children"][0]["children"]] == ["leaf"], (
    "middle's call to leaf must be nested under middle, not under outer"
)
assert tree["children"][0]["children"][0]["result"] == 4, (
    "the inner leaf(3) returned 4"
)
assert tree["children"][1]["children"] == [], "leaf calls nothing"

# --- the namespace is put back exactly as it was ---------------------------
for name, value in originals.items():
    assert namespace[name] is value, (
        f"{name} was left wrapped in the namespace after the call returned"
    )

# --- recursion nests rather than flattening --------------------------------
RECURSIVE = "\n".join([
    "def countdown(n):",
    "    if n <= 0:",
    "        return 0",
    "    return countdown(n - 1) + 1",
])
namespace = build(RECURSIVE)
result, tree = capture_call_tree(namespace, "countdown", (3,))
assert result == 3, f"countdown(3) is 3; got {result!r}"
depth, node = 0, tree
while node["children"]:
    assert len(node["children"]) == 1, (
        f"each recursive call makes exactly one call; got {node!r}"
    )
    node = node["children"][0]
    depth += 1
assert depth == 3, (
    f"countdown(3) recurses three times, so the tree is four deep; got a "
    f"depth of {depth}"
)
assert node["args"] == (0,), f"the deepest call is countdown(0); got {node!r}"

# --- an exception is recorded and then propagates --------------------------
FAILING = "\n".join([
    "def top(n):",
    "    return bottom(n)",
    "",
    "def bottom(n):",
    "    raise ValueError('no')",
])
namespace = build(FAILING)
originals = {
    name: value for name, value in namespace.items()
    if callable(value) and not name.startswith("__")
}
captured = {}
try:
    capture_call_tree(namespace, "top", (1,))
except ValueError as exc:
    captured["exc"] = exc
else:
    raise AssertionError("the entry call raised, so capture must re-raise it")
for name, value in originals.items():
    assert namespace[name] is value, (
        f"{name} was left wrapped after the entry call raised; restoring must "
        "happen on the failure path too"
    )

# --- a caught exception is recorded on the child, not the parent -----------
CAUGHT = "\n".join([
    "def guarded(n):",
    "    try:",
    "        return risky(n)",
    "    except ValueError:",
    "        return -1",
    "",
    "def risky(n):",
    "    raise ValueError('no')",
])
namespace = build(CAUGHT)
result, tree = capture_call_tree(namespace, "guarded", (1,))
assert result == -1, f"guarded swallows the error and returns -1; got {result!r}"
assert tree["result"] == -1 and "error" not in tree, (
    f"guarded returned normally; got {tree!r}"
)
child = tree["children"][0]
assert child["name"] == "risky", f"got {child!r}"
assert child["error"] == "ValueError", (
    f"the child records the exception class name; got {child!r}"
)
assert "result" not in child, "a failed call has no 'result' key"

# --- an entry that is not a function in the namespace ----------------------
namespace = build(NESTED)
namespace["not_callable"] = 5
for bad in ("missing", "not_callable"):
    try:
        capture_call_tree(namespace, bad, ())
    except KeyError:
        pass
    else:
        raise AssertionError(f"{bad!r} does not name a function in namespace")
''',
    ),
    task(
        f"{FAMILY}-0016", FAMILY,
        prompt=(
            "Implement a Python function close_enough(actual, expected, "
            "rel_tol, abs_tol) returning a bool, for use as a numeric test "
            "assertion over nested data.\n\n"
            "Numbers compare within tolerance: they match when the absolute "
            "difference is at most max(rel_tol * max(abs(actual), "
            "abs(expected)), abs_tol). Two NaNs match each other and a NaN "
            "matches nothing else. Infinities match only the same-signed "
            "infinity, whatever the tolerances. Negative and positive zero "
            "match. bool is not a number here: True matches only True, and "
            "never 1 or 1.0. Lists and tuples match element-wise, and only a "
            "list matches a list. Dicts match when their key sets are equal "
            "and every value matches. Strings, bytes, and None match by "
            "equality. Any other type, or a structure mismatch, is False. "
            "Recursion is by structure, so nesting may be arbitrarily deep."
        ),
        timeout_seconds=45.0,
        validator=LOAD_CANDIDATE + require("close_enough") + r'''
NAN = float("nan")
INF = float("inf")


def yes(actual, expected, note, rel_tol=1e-9, abs_tol=0.0):
    got = close_enough(actual, expected, rel_tol, abs_tol)
    assert got is True or got == True, (  # noqa: E712 - a truthy int is a bug
        f"{note}: expected a match for {actual!r} vs {expected!r}, got {got!r}"
    )


def no(actual, expected, note, rel_tol=1e-9, abs_tol=0.0):
    got = close_enough(actual, expected, rel_tol, abs_tol)
    assert got is False or got == False, (  # noqa: E712
        f"{note}: expected no match for {actual!r} vs {expected!r}, got {got!r}"
    )


# --- the tolerance formula itself ------------------------------------------
yes(1.0, 1.0 + 1e-12, "well inside the relative tolerance")
no(1.0, 1.1, "well outside it")
yes(1e10, 1e10 + 1.0, "relative tolerance scales with magnitude", rel_tol=1e-9)
no(1e-10, 1e-11, "relative tolerance does not rescue tiny magnitudes")
yes(1e-10, 1e-11, "but an absolute tolerance does", abs_tol=1e-9)
yes(0.0, 0.0, "zero matches zero with no tolerance at all")
no(0.0, 1e-12, "a pure relative tolerance can never match against zero")
yes(0.0, 1e-12, "an absolute tolerance can", abs_tol=1e-9)
yes(-5.0, -5.0000000001, "negatives use magnitudes, not signed differences")
no(-5.0, 5.0, "opposite signs are far apart")

# --- the special values ----------------------------------------------------
yes(NAN, NAN, "NaN matches NaN, which plain == does not")
no(NAN, 1.0, "NaN matches nothing else")
no(1.0, NAN, "and not in the other direction either")
no(NAN, INF, "NaN is not an infinity")
yes(INF, INF, "same-signed infinities match")
yes(-INF, -INF, "including negative")
no(INF, -INF, "opposite infinities do not")
no(INF, 1e308, "an infinity does not match a large finite number")
no(INF, 1e308, "not even with a huge tolerance", rel_tol=1.0, abs_tol=1e308)
yes(-0.0, 0.0, "signed zeros match")
yes(0.0, -0.0, "in both directions")

# --- bool is not a number --------------------------------------------------
yes(True, True, "True matches True")
yes(False, False, "False matches False")
no(True, 1, "a bool must not match an int")
no(1, True, "nor an int a bool")
no(True, 1.0, "nor a bool a float")
no(1.0, True, "nor a float a bool")
no(True, False, "and the two bools differ")
no(0, False, "zero is not False here")

# --- integers and floats do compare numerically ---------------------------
yes(1, 1.0, "an int matches an equal float")
yes(2, 2.0000000001, "and within tolerance")
no(2, 3, "but not outside it")

# --- containers ------------------------------------------------------------
yes([1.0, 2.0], [1.0, 2.0 + 1e-12], "lists match element-wise")
no([1.0, 2.0], [1.0, 2.5], "one bad element fails the list")
no([1.0, 2.0], [1.0], "different lengths do not match")
no([1.0, 2.0], (1.0, 2.0), "a list does not match a tuple")
yes((1.0, 2.0), (1.0, 2.0), "a tuple matches a tuple")
yes([], [], "empty lists match")
no([], {}, "an empty list is not an empty dict")

yes({"a": 1.0}, {"a": 1.0 + 1e-12}, "dict values compare with tolerance")
no({"a": 1.0}, {"b": 1.0}, "different keys do not match")
no({"a": 1.0}, {"a": 1.0, "b": 2.0}, "a missing key does not match")
no({"a": 1.0, "b": 2.0}, {"a": 1.0}, "nor an extra one")
yes({}, {}, "empty dicts match")

# --- exact-equality types --------------------------------------------------
yes("x", "x", "strings match exactly")
no("x", "y", "and only exactly")
no("1", 1, "a string does not match a number")
yes(b"x", b"x", "bytes match exactly")
no(b"x", "x", "bytes do not match a str")
yes(None, None, "None matches None")
no(None, 0, "None is not zero")
no(None, "", "nor an empty string")

# --- unsupported types are False, not an exception ------------------------
no(object(), object(), "two unrelated objects do not match")
no({1, 2}, {1, 2}, "a set is not a supported structure")

# --- nesting ---------------------------------------------------------------
yes(
    {"points": [[1.0, 2.0], [3.0, 4.0]], "label": "run", "ok": True},
    {"points": [[1.0, 2.0 + 1e-12], [3.0, 4.0]], "label": "run", "ok": True},
    "nested structures recurse",
)
no(
    {"points": [[1.0, 2.0], [3.0, 4.0]], "ok": True},
    {"points": [[1.0, 2.0], [3.0, 4.5]], "ok": True},
    "a deep mismatch still fails",
)
no(
    {"ok": True},
    {"ok": 1},
    "the bool rule holds at depth too",
)
''',
    ),
    task(
        f"{FAMILY}-0017", FAMILY,
        prompt=(
            "Implement a Python function simplify_control_flow(source, "
            "function_name) that returns new source in which the body of "
            "function_name has been simplified without changing what it "
            "does. Apply all three of these, repeatedly until nothing "
            "changes:\n\n"
            "First, an if whose test is the literal True or False is "
            "replaced by the branch that runs, with the other branch dropped; "
            "when the surviving branch is empty the whole statement goes, and "
            "when a body would become empty it gets a pass. Second, a "
            "statement that follows a return or a raise in the same block is "
            "unreachable and is removed. Third, when every path through an "
            "if's body returns or raises, its else branch is unindented to "
            "follow the if instead of nesting inside it.\n\n"
            "Only literal True and False count as constant tests; a name or a "
            "comparison never does. Behaviour must be identical for every "
            "input."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("simplify_control_flow") + r'''
import ast


def namespace(source):
    scope = {}
    exec(compile(source, "<case>", "exec"), scope)
    return scope


def body_of(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is missing from the result")


def same_behaviour(before, after, name, cases):
    original, simplified = namespace(before), namespace(after)
    for arguments in cases:
        try:
            expected = ("value", original[name](*arguments))
        except Exception as exc:  # noqa: BLE001 - the class is the contract
            expected = ("error", type(exc).__name__)
        try:
            actual = ("value", simplified[name](*arguments))
        except Exception as exc:  # noqa: BLE001
            actual = ("error", type(exc).__name__)
        assert actual == expected, (
            f"{name}{arguments!r} changed: {expected!r} -> {actual!r}"
        )


# --- constant tests, unreachable code, and an else that can be unindented --
SOURCE = "\n".join([
    "def classify(value):",
    "    if False:",
    "        return 'never'",
    "    if True:",
    "        marker = 'always'",
    "    else:",
    "        marker = 'unreachable'",
    "    if value > 0:",
    "        return marker + ':positive'",
    "    else:",
    "        if value == 0:",
    "            raise ValueError('zero')",
    "        else:",
    "            return marker + ':negative'",
    "    return 'dead'",
])
result = simplify_control_flow(SOURCE, "classify")
same_behaviour(SOURCE, result, "classify", [(5,), (-5,), (0,), (1,), (-1,)])

simplified = body_of(result, "classify")
for node in ast.walk(simplified):
    if isinstance(node, ast.If):
        assert not (
            isinstance(node.test, ast.Constant)
            and isinstance(node.test.value, bool)
        ), "a literal-constant if survived the simplification"

assert "'never'" not in result, "the `if False` branch was kept"
assert "'unreachable'" not in result, "the dead else of `if True` was kept"
assert "'dead'" not in result, (
    "the statement after the if/else is unreachable because both branches "
    "return or raise, so it must be removed"
)
assert "'always'" in result, "the live branch of `if True` was lost"

outer = [n for n in simplified.body if isinstance(n, ast.If)]
assert outer and outer[0].orelse == [], (
    "the `value > 0` body returns on every path, so its else must be "
    "unindented to follow the if rather than nesting inside it"
)

# --- a name or comparison is not a constant test ---------------------------
KEEP = "\n".join([
    "def keep(flag, other):",
    "    if flag:",
    "        return 1",
    "    if other == other:",
    "        return 2",
    "    return 3",
])
result = simplify_control_flow(KEEP, "keep")
same_behaviour(KEEP, result, "keep", [
    (True, 1), (False, 1), (0, 2), ("", 3), (None, 0),
])
kept = body_of(result, "keep")
assert len([n for n in ast.walk(kept) if isinstance(n, ast.If)]) == 2, (
    "neither `if flag` nor `if other == other` is a literal constant, so "
    "both must survive"
)

# --- the rules must be applied to a fixed point ----------------------------
# Removing the `if False` exposes a `return` that makes the next statement
# unreachable, which in turn empties a block. One pass is not enough.
CASCADE = "\n".join([
    "def cascade(value):",
    "    if True:",
    "        if False:",
    "            value = value * 100",
    "        return value + 1",
    "    value = value * 2",
    "    return value",
])
result = simplify_control_flow(CASCADE, "cascade")
same_behaviour(CASCADE, result, "cascade", [(0,), (1,), (-3,), (7,)])
cascaded = body_of(result, "cascade")
assert not [n for n in ast.walk(cascaded) if isinstance(n, ast.If)], (
    "every if here is constant, so none should remain"
)
assert "* 100" not in result and "* 2" not in result, (
    "both multiplications are unreachable"
)

# --- a body that would become empty needs a pass ---------------------------
EMPTIED = "\n".join([
    "def emptied(value):",
    "    if value:",
    "        if False:",
    "            value = 1",
    "    return value",
])
result = simplify_control_flow(EMPTIED, "emptied")
same_behaviour(EMPTIED, result, "emptied", [(0,), (1,), ("",), ("x",)])
compile(result, "<result>", "exec")

# --- an unconditional raise also makes what follows unreachable ------------
RAISER = "\n".join([
    "def raiser(value):",
    "    raise ValueError('always')",
    "    return value",
])
result = simplify_control_flow(RAISER, "raiser")
same_behaviour(RAISER, result, "raiser", [(1,), (2,)])
assert "return value" not in result, (
    "the return after an unconditional raise is unreachable"
)
''',
    ),
    task(
        f"{FAMILY}-0018", FAMILY,
        prompt=(
            "Implement a Python function compare_snapshot(actual, stored, "
            "patterns) for golden-file testing. patterns is a sequence of "
            "(regex, replacement) pairs used to erase incidental detail such "
            "as timestamps, temporary paths, and object addresses.\n\n"
            "Normalize both texts by splitting them into lines on '\\n', "
            "dropping a single trailing empty line if the text ended with a "
            "newline, and applying every pattern to every line in the order "
            "given, replacing all occurrences. Return the pair (matched, "
            "differences). matched is True when the normalized line lists are "
            "equal. differences is a list of (index, stored_line, "
            "actual_line) for every position where they differ, using None on "
            "whichever side has no line at that index, ordered by index. "
            "Return the normalized actual lines as a third element only when "
            "asked: the function takes a keyword-only argument update "
            "defaulting to False, and when it is True return (matched, "
            "differences, normalized_actual_lines) instead."
        ),
        timeout_seconds=45.0,
        validator=LOAD_CANDIDATE + require("compare_snapshot") + r'''
TIMESTAMP = (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "<TS>")
ADDRESS = (r"0x[0-9a-f]+", "<ADDR>")
TMPDIR = (r"/tmp/[A-Za-z0-9_]+", "<TMP>")
PATTERNS = [TIMESTAMP, ADDRESS, TMPDIR]

# --- incidental detail must not fail the comparison -----------------------
stored = (
    "run started 2026-01-01T00:00:00\n"
    "object <Widget at 0xdeadbeef>\n"
    "workspace /tmp/aaaaaaa\n"
)
actual = (
    "run started 2026-09-05T18:30:12\n"
    "object <Widget at 0x1234abcd>\n"
    "workspace /tmp/zz99zz\n"
)
matched, differences = compare_snapshot(actual, stored, PATTERNS)
assert matched is True, (
    f"only the normalized detail differs, so this matches; got {differences!r}"
)
assert differences == [], f"a match reports no differences; got {differences!r}"

# --- normalization must not hide a real change ----------------------------
changed = (
    "run started 2026-09-05T18:30:12\n"
    "object <Gadget at 0x1234abcd>\n"
    "workspace /tmp/zz99zz\n"
)
matched, differences = compare_snapshot(changed, stored, PATTERNS)
assert matched is False, "Widget became Gadget, which is a real change"
assert differences == [
    (1, "object <Widget at <ADDR>>", "object <Gadget at <ADDR>>"),
], f"got {differences!r}"

# --- with no patterns the same texts differ everywhere --------------------
matched, differences = compare_snapshot(actual, stored, [])
assert matched is False, "without normalization every line differs"
assert [index for index, _, _ in differences] == [0, 1, 2], (
    f"got {differences!r}"
)

# --- a trailing newline is not a difference, but a blank line is ----------
matched, differences = compare_snapshot("a\nb\n", "a\nb", PATTERNS)
assert matched is True, (
    f"one trailing newline is dropped on both sides; got {differences!r}"
)
matched, differences = compare_snapshot("a\nb\n\n", "a\nb\n", PATTERNS)
assert matched is False, (
    "the second text has a genuine trailing blank line"
)
assert differences == [(2, None, "")], f"got {differences!r}"

# --- length mismatches use None on the short side -------------------------
matched, differences = compare_snapshot("a\nb\nc\n", "a\n", PATTERNS)
assert matched is False, "the actual text has two extra lines"
assert differences == [(1, None, "b"), (2, None, "c")], f"got {differences!r}"
matched, differences = compare_snapshot("a\n", "a\nb\nc\n", PATTERNS)
assert differences == [(1, "b", None), (2, "c", None)], f"got {differences!r}"

# --- patterns apply in order and to every occurrence ----------------------
# The second pattern rewrites what the first produced, so applying them in the
# wrong order, or only to the first match on a line, gives a different result.
ordered = [(r"a", "b"), (r"b", "c")]
matched, differences, normalized = compare_snapshot(
    "aaa", "ccc", ordered, update=True
)
assert normalized == ["ccc"], (
    "every 'a' becomes 'b' and then every 'b' becomes 'c'; got "
    f"{normalized!r}"
)
assert matched is True, f"got {differences!r}"

matched, differences, normalized = compare_snapshot(
    "aaa", "aaa", list(reversed(ordered)), update=True
)
assert normalized == ["bbb"], (
    "with the patterns reversed the 'b' rule runs first and matches nothing, "
    "so only 'a' -> 'b' applies; a candidate that applies them in a fixed or "
    f"sorted order produces 'ccc' here. got {normalized!r}"
)
assert matched is True, f"got {differences!r}"

# --- update returns the lines to store ------------------------------------
result = compare_snapshot(changed, stored, PATTERNS, update=True)
assert len(result) == 3, f"update mode returns a triple; got {len(result)}"
matched, differences, normalized = result
assert matched is False, "update mode still reports the comparison"
assert normalized == [
    "run started <TS>",
    "object <Gadget at <ADDR>>",
    "workspace <TMP>",
], f"the stored form is the normalized actual text; got {normalized!r}"
assert len(compare_snapshot(changed, stored, PATTERNS)) == 2, (
    "without update the result is a pair"
)

# --- empty texts -----------------------------------------------------------
matched, differences = compare_snapshot("", "", PATTERNS)
assert matched is True and differences == [], f"got {differences!r}"
matched, differences = compare_snapshot("", "a", PATTERNS)
assert matched is False and differences == [(0, "a", "")], f"got {differences!r}"
''',
    ),
    task(
        f"{FAMILY}-0201", FAMILY,
        prompt=(
            "Implement a Python function detect_order_dependency(tests, run) "
            "that explains why a test passes alone and fails in the suite. "
            "tests is a list of distinct test names in the order the suite "
            "runs them, and run(order) executes exactly the named tests in "
            "the given order and returns the collection of names that "
            "failed. Let victim be the first test in tests that fails when "
            "the whole suite runs. Find the smallest k for which running "
            "tests[:k] followed by victim still fails victim, and return the "
            "pair (tests[k - 1], victim). You may assume that if some prefix "
            "fails victim then every longer prefix does too. Return None "
            "when the whole suite passes, and also when victim fails on its "
            "own, because a test that fails in isolation is broken rather "
            "than order-dependent. Call run at most "
            "3 * len(tests).bit_length() + 15 times, which rules out trying "
            "each earlier test in turn."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("detect_order_dependency") + r'''
def build(size, polluter_index, victim_index=None, alone_fails=False,
          healthy=False):
    """A suite whose only defect is one test leaking state into another."""
    names = ['t%03d' % index for index in range(size)]
    victim = names[size - 1 if victim_index is None else victim_index]
    polluter = names[polluter_index]
    tally = {'calls': 0}

    def run(order):
        tally['calls'] += 1
        order = list(order)
        for name in order:
            assert name in names, 'run called with an unknown test %r' % name
        assert len(set(order)) == len(order), 'run called with a repeated test'
        if healthy or victim not in order:
            return []
        if alone_fails:
            return [victim]
        if polluter in order and order.index(polluter) < order.index(victim):
            return [victim]
        return []

    return names, run, tally, polluter, victim


def allowance(size):
    return 3 * size.bit_length() + 15


# A polluter near the end of a long suite. Growing a prefix one test at a
# time would need 61 calls against an allowance of 33.
names, run, tally, polluter, victim = build(64, 60)
assert detect_order_dependency(names, run) == (polluter, victim), (
    'the polluter at index 60 was not attributed'
)
assert tally['calls'] <= allowance(64), (
    'used %d calls of an allowed %d' % (tally['calls'], allowance(64))
)

# And one near the front, which shrinking the suite from the back cannot
# afford either. Only a search that halves the prefix satisfies both.
names, run, tally, polluter, victim = build(64, 1)
assert detect_order_dependency(names, run) == (polluter, victim), (
    'the polluter at index 1 was not attributed'
)
assert tally['calls'] <= allowance(64), (
    'used %d calls of an allowed %d' % (tally['calls'], allowance(64))
)

# The victim is not always the last test to run.
names, run, tally, polluter, victim = build(16, 2, victim_index=9)
assert detect_order_dependency(names, run) == (polluter, victim), (
    'a victim in the middle of the suite was not attributed'
)

# The smallest suite that can carry a polluter at all.
names, run, tally, polluter, victim = build(2, 0)
assert detect_order_dependency(names, run) == ('t000', 't001')

# Nothing failed, so there is nothing to attribute.
names, run, tally, polluter, victim = build(8, 3, healthy=True)
assert detect_order_dependency(names, run) is None, (
    'a passing suite was reported as order-dependent'
)

# A test that fails on its own is broken. Naming whichever test precedes it
# sends the repair at innocent code, which is the failure mode this measures.
names, run, tally, polluter, victim = build(8, 3, alone_fails=True)
assert detect_order_dependency(names, run) is None, (
    'a test that fails in isolation was blamed on an earlier test'
)

# An empty suite has no victim.
assert detect_order_dependency([], lambda order: []) is None
''',
    ),
    task(
        f"{FAMILY}-0202", FAMILY,
        prompt=(
            "Implement a Python function merge_three_way(base, left, right) "
            "performing a three-way line merge. Each argument is a list of "
            "lines without terminators. Return a (lines, conflicted) pair. "
            "Where only one side changed a region of base, take that side's "
            "version. Where both sides made the identical change, take it "
            "once. Where both changed one region differently, emit the line "
            "'<<<<<<< left', then the left version of that region, then the "
            "line '=======', then the right version, then the line "
            "'>>>>>>> right'. Two insertions at the same point in base count "
            "as one region. conflicted is True exactly when a conflict "
            "region was emitted, and every region the two sides changed "
            "independently must still be merged in the same result."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("merge_three_way") + r'''
BASE = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta']


def merged(base, left, right):
    result = merge_three_way(list(base), list(left), list(right))
    assert isinstance(result, tuple) and len(result) == 2, (
        'expected a (lines, conflicted) pair; got %r' % (result,)
    )
    lines, conflicted = result
    return list(lines), conflicted


# Nobody changed anything.
assert merged(BASE, BASE, BASE) == (list(BASE), False)

# One side changed and the other did not, from each side in turn.
left_only = ['alpha', 'BETA', 'gamma', 'delta', 'epsilon', 'zeta', 'eta']
assert merged(BASE, left_only, BASE) == (left_only, False), (
    'a change only the left side made was dropped'
)
right_only = ['alpha', 'beta', 'gamma', 'DELTA', 'epsilon', 'zeta', 'eta']
assert merged(BASE, BASE, right_only) == (right_only, False), (
    'a change only the right side made was dropped'
)

# Independent regions merge into one result.
assert merged(BASE, left_only, right_only) == (
    ['alpha', 'BETA', 'gamma', 'DELTA', 'epsilon', 'zeta', 'eta'], False
)

# The same edit from both sides is applied once, not twice.
assert merged(BASE, left_only, left_only) == (left_only, False)

# A deletion is a change like any other.
shortened = ['alpha', 'delta', 'epsilon', 'zeta', 'eta']
assert merged(BASE, shortened, BASE) == (shortened, False)

# Both sides rewrote one line differently. That is the conflict.
lines, conflicted = merged(
    BASE,
    ['alpha', 'beta', 'LEFT', 'delta', 'epsilon', 'zeta', 'eta'],
    ['alpha', 'beta', 'RIGHT', 'delta', 'epsilon', 'zeta', 'eta'],
)
assert conflicted is True, 'a genuine conflict was reported as a clean merge'
assert lines == ['alpha', 'beta', '<<<<<<< left', 'LEFT', '=======', 'RIGHT',
                 '>>>>>>> right', 'delta', 'epsilon', 'zeta', 'eta'], lines

# Two insertions at the same point conflict even though neither side touched
# a line the other touched.
lines, conflicted = merged(
    BASE,
    ['alpha', 'LX', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta'],
    ['alpha', 'RX', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta'],
)
assert conflicted is True and lines == [
    'alpha', '<<<<<<< left', 'LX', '=======', 'RX', '>>>>>>> right',
    'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
], lines

# A conflict in one region must not abandon the merge of the others: the
# left edit before it and the right edit after it both survive.
lines, conflicted = merged(
    BASE,
    ['alpha', 'BETA', 'gamma', 'delta', 'LEFT', 'zeta', 'eta'],
    ['alpha', 'beta', 'gamma', 'delta', 'RIGHT', 'zeta', 'ETA'],
)
assert conflicted is True
assert lines == ['alpha', 'BETA', 'gamma', 'delta', '<<<<<<< left', 'LEFT',
                 '=======', 'RIGHT', '>>>>>>> right', 'zeta', 'ETA'], lines

# An empty base is the degenerate case both sides add to.
assert merged([], ['x', 'y'], ['x', 'y']) == (['x', 'y'], False)
lines, conflicted = merged([], ['x'], ['y'])
assert conflicted is True and lines == [
    '<<<<<<< left', 'x', '=======', 'y', '>>>>>>> right'
], lines
''',
    ),
    task(
        f"{FAMILY}-0204", FAMILY,
        prompt=(
            "Implement a Python function resolve_fixtures(fixtures, "
            "requested) that decides the order test fixtures are set up and "
            "torn down. fixtures maps a fixture name to a mapping with keys "
            "scope, one of 'session', 'module' or 'function', and requires, "
            "a list of fixture names it depends on. requested lists the "
            "fixtures one test asks for, in the order it names them. Return "
            "a (setup, teardown) pair of lists holding every fixture "
            "reachable from requested exactly once. In setup a fixture "
            "appears after everything it requires, broader scopes come "
            "before narrower ones with session before module before "
            "function, and fixtures neither of those rules separates keep "
            "the order a depth-first walk of requested first reached them "
            "in. teardown is setup reversed. Raise ValueError if a named "
            "fixture is not defined, if the dependency graph contains a "
            "cycle, or if a fixture requires one of narrower scope, which "
            "cannot outlive it."
        ),
        validator=LOAD_CANDIDATE + require("resolve_fixtures") + r'''
def fixture(scope, *requires):
    return {'scope': scope, 'requires': list(requires)}


def resolved(fixtures, requested):
    result = resolve_fixtures(fixtures, list(requested))
    assert isinstance(result, tuple) and len(result) == 2, (
        'expected a (setup, teardown) pair; got %r' % (result,)
    )
    setup, teardown = list(result[0]), list(result[1])
    assert len(set(setup)) == len(setup), (
        'a fixture is set up more than once: %r' % (setup,)
    )
    assert teardown == list(reversed(setup)), (
        'teardown is not setup reversed: %r then %r' % (setup, teardown)
    )
    return setup


# A chain is set up from its base outwards.
chain = {
    'database': fixture('session'),
    'schema': fixture('module', 'database'),
    'rows': fixture('function', 'schema'),
}
assert resolved(chain, ['rows']) == ['database', 'schema', 'rows']

# A shared dependency of two requested fixtures is set up once.
diamond = {
    'engine': fixture('session'),
    'reader': fixture('module', 'engine'),
    'writer': fixture('module', 'engine'),
    'case': fixture('function', 'reader', 'writer'),
}
assert resolved(diamond, ['case']) == ['engine', 'reader', 'writer', 'case']

# Scope decides the order of two fixtures that do not depend on each other,
# even when the test names the narrower one first. Setting up a session
# fixture inside a function one would tie its lifetime to the wrong thing.
scoped = {'tmpdir': fixture('function'), 'server': fixture('session')}
assert resolved(scoped, ['tmpdir', 'server']) == ['server', 'tmpdir'], (
    'a session fixture was set up after a function one'
)

# Within one scope the walk order is what remains, so it must be kept.
flat = {'a': fixture('function'), 'b': fixture('function'),
        'c': fixture('function')}
assert resolved(flat, ['c', 'a', 'b']) == ['c', 'a', 'b']
assert resolved(flat, ['b', 'b', 'a']) == ['b', 'a']

# A fixture no test asked for is not set up.
assert resolved(dict(chain, spare=fixture('session')), ['schema']) == \
    ['database', 'schema']

for fixtures, requested, why in (
    ({'a': fixture('function', 'missing')}, ['a'], 'undefined dependency'),
    ({'a': fixture('function')}, ['absent'], 'undefined request'),
    ({'a': fixture('function', 'b'), 'b': fixture('function', 'a')}, ['a'],
     'two-fixture cycle'),
    ({'a': fixture('function', 'a')}, ['a'], 'self-dependency'),
    ({'slow': fixture('session', 'fast'), 'fast': fixture('function')},
     ['slow'], 'session fixture requiring a function one'),
    ({'mid': fixture('module', 'inner'), 'inner': fixture('function')},
     ['mid'], 'module fixture requiring a function one'),
):
    try:
        resolve_fixtures(fixtures, requested)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted a %s' % why)

# Requiring something broader is exactly what scopes are for.
assert resolved({'narrow': fixture('function', 'wide'),
                 'wide': fixture('session')}, ['narrow']) == \
    ['wide', 'narrow']
assert resolve_fixtures({}, []) == ([], []) or \
    list(resolve_fixtures({}, [])[0]) == []
''',
    ),
    task(
        f"{FAMILY}-0205", FAMILY,
        prompt=(
            "Implement a Python function nondeterministic_fields(run, times) "
            "that finds which parts of a result are not reproducible. run() "
            "takes no arguments and returns a structure of dicts, lists and "
            "scalars. Call run exactly times times and return the sorted "
            "list of the paths whose value is not identical in every run. A "
            "path is a string: the empty string is the whole result, a "
            "mapping key extends a path with a dot and the key, and a list "
            "index extends it with the index in square brackets. A key "
            "directly on the result takes no leading dot, so 'a.b[0].c' and "
            "'[2].x' both name nested fields. Descend into two mappings only "
            "when they have the same keys and into two lists only when they "
            "have the same length; when a container's shape differs between "
            "runs, or the values at a path are of different types, report "
            "that container's own path and do not descend into it. "
            "Otherwise report the deepest paths that account for the "
            "difference, never an ancestor of one. Raise ValueError if "
            "times is less than 2."
        ),
        validator=LOAD_CANDIDATE + require("nondeterministic_fields") + r'''
def source(sequence):
    """A run() that returns the next canned result, and counts its calls."""
    state = {'index': 0, 'calls': 0}

    def run():
        state['calls'] += 1
        value = sequence[min(state['index'], len(sequence) - 1)]
        state['index'] += 1
        return value

    return run, state


steady = {'a': {'b': [1, 2]}, 'c': 'fixed'}
run, state = source([steady, steady, steady])
assert nondeterministic_fields(run, 3) == []
assert state['calls'] == 3, 'run was called %d times, not 3' % state['calls']

# One deep field varies; its ancestors must not be reported instead.
run, _ = source([{'a': {'b': [1, 2]}, 'c': 'fixed'},
                 {'a': {'b': [1, 9]}, 'c': 'fixed'}])
assert nondeterministic_fields(run, 2) == ['a.b[1]']

# Two independent fields, reported sorted.
run, _ = source([{'a': 1, 'z': {'q': 'x'}}, {'a': 2, 'z': {'q': 'y'}}])
assert nondeterministic_fields(run, 2) == ['a', 'z.q']

# A field that only moves on the third run is still nondeterministic. A
# check that compares the first two results reports nothing here, which is
# how a flake survives a reproduction attempt that ran twice.
run, _ = source([{'seed': 1}, {'seed': 1}, {'seed': 4}])
assert nondeterministic_fields(run, 3) == ['seed'], (
    'a difference that appears only on a later run was missed'
)

# A container whose shape changes is reported as itself: there is no
# per-element path that survives in both runs.
run, _ = source([{'items': [1, 2], 'ok': True},
                 {'items': [1, 2, 3], 'ok': True}])
assert nondeterministic_fields(run, 2) == ['items']
run, _ = source([{'meta': {'host': 'a'}}, {'meta': {'host': 'a', 'pid': 7}}])
assert nondeterministic_fields(run, 2) == ['meta']

# A type change at a path is a shape change too.
run, _ = source([{'v': 3}, {'v': '3'}])
assert nondeterministic_fields(run, 2) == ['v']

# The whole result is the empty path.
run, _ = source(['first', 'second'])
assert nondeterministic_fields(run, 2) == ['']
run, _ = source([[1, 2], [1, 3]])
assert nondeterministic_fields(run, 2) == ['[1]']
run, _ = source([[{'x': 1}], [{'x': 2}]])
assert nondeterministic_fields(run, 2) == ['[0].x']

for bad in (1, 0, -3):
    try:
        nondeterministic_fields(lambda: 1, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted times=%r, which compares nothing' % bad)
''',
    ),
    task(
        f"{FAMILY}-0206", FAMILY,
        prompt=(
            "Implement a Python function check_equality_contract(values) "
            "that audits value objects for the rules dicts, sets and sorting "
            "rely on. values is a sequence of hashable instances. Return the "
            "sorted list of rule names that are violated, drawn from: "
            "'reflexive' when a value does not equal itself; 'symmetric' "
            "when x == y and y == x disagree for some pair, including a pair "
            "of different types; 'transitive' when x == y and y == z but not "
            "x == z; 'hash' when two values compare equal and hash to "
            "different numbers; 'inequality' when x != y is not the negation "
            "of x == y; and 'ordering' when x < y and x == y are not a "
            "consistent strict order, meaning some pair does not have "
            "exactly one of x < y, x == y and y < x true, or some triple has "
            "x < y and y < z without x < z. Skip the ordering rule entirely "
            "if comparing two values with < raises TypeError. Consider every "
            "pair and triple drawn from values, including a value paired "
            "with itself, and return an empty list when every rule holds."
        ),
        validator=LOAD_CANDIDATE + require("check_equality_contract") + r'''
class Point:
    """The contract kept."""

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x

    def __hash__(self):
        return hash(self.x)

    def __lt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x < other.x


class Loose:
    """Overrides __eq__ and inherits identity hashing, the classic slip."""

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, Loose) and self.x == other.x

    __hash__ = object.__hash__


class Fuzzy:
    """Equality within a tolerance, which cannot be transitive."""

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, Fuzzy) and abs(self.x - other.x) <= 2

    def __hash__(self):
        return 0


class Blurry(Fuzzy):
    """The same tolerance, now hashing by identity as well."""

    __hash__ = object.__hash__


class Wide:
    """Accepts the narrow type; the narrow type does not accept it back."""

    def __eq__(self, other):
        return isinstance(other, (Wide, Narrow))

    def __hash__(self):
        return 7


class Narrow:
    def __eq__(self, other):
        return isinstance(other, Narrow)

    def __hash__(self):
        return 7


class Sloppy:
    """__ne__ written by hand and no longer the negation of __eq__."""

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, Sloppy) and self.x == other.x

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash(self.x)


class Nanish:
    def __eq__(self, other):
        return False

    def __hash__(self):
        return 3


class Weird:
    """Ordering that is true of everything, including a value and itself."""

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, Weird) and self.x == other.x

    def __hash__(self):
        return hash(self.x)

    def __lt__(self, other):
        return True


assert check_equality_contract([]) == []
assert check_equality_contract([Point(1), Point(2), Point(1)]) == [], (
    'a correct value type was reported as broken'
)
assert check_equality_contract([Loose(1), Loose(1)]) == ['hash']
assert check_equality_contract([Fuzzy(0), Fuzzy(2), Fuzzy(4)]) == ['transitive']
assert check_equality_contract([Wide(), Narrow()]) == ['symmetric'], (
    'an asymmetry that only shows across two types was missed'
)
assert check_equality_contract([Sloppy(1), Sloppy(2)]) == ['inequality']
assert check_equality_contract([Nanish(), Nanish()]) == ['reflexive']
assert check_equality_contract([Weird(1), Weird(2)]) == ['ordering']

# Every violated rule is reported, not the first one found.
assert check_equality_contract([Blurry(0), Blurry(2), Blurry(4)]) == \
    ['hash', 'transitive']

# One value on its own still exercises the reflexive and ordering rules.
assert check_equality_contract([Point(5)]) == []
assert check_equality_contract([Nanish()]) == ['reflexive']
assert check_equality_contract([Weird(1)]) == ['ordering']
''',
    ),
    task(
        f"{FAMILY}-0203", FAMILY,
        prompt=(
            "Implement a Python function tidy_imports(source) that removes "
            "the unused imports from a module and returns the new source. "
            "Consider only import statements at module level; leave an "
            "import inside a function or class alone. An imported binding "
            "counts as used when its name appears anywhere else in the "
            "module as a name, including as the leftmost part of an "
            "attribute reference and inside a nested function or class, when "
            "it appears as a string in a module-level __all__ assignment, or "
            "when it appears inside a string annotation. A string anywhere "
            "else, such as an ordinary constant, does not count. 'import "
            "a.b' binds the name a, while 'import a.b as c' binds c. Drop a "
            "'from x import a, b' statement entirely when none of its names "
            "are used, and otherwise rewrite it as one line naming the used "
            "ones in their original order. Never remove a __future__ import "
            "and never remove a star import, whose bindings cannot be known. "
            "Leave every other line of the module exactly as it was."
        ),
        validator=LOAD_CANDIDATE + require("tidy_imports") + r'''
import ast

SOURCE = """from __future__ import annotations

import os
import sys
import json.decoder
import collections.abc as abc
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal
from re import *

__all__ = ["Row", "dump", "Decimal"]

MARKER = "os"


@dataclass
class Row:
    values: Dict[str, int]

    def widen(self) -> "Optional[int]":
        return abc.Sized and sys.maxsize


def dump(payload):
    import string

    return json.decoder.JSONDecoder().decode(payload)
"""


def imports_of(text):
    found = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            found.append(('import',
                          tuple((a.name, a.asname) for a in node.names)))
        elif isinstance(node, ast.ImportFrom):
            found.append(('from', node.module, node.level,
                          tuple((a.name, a.asname) for a in node.names)))
    return found


def body_lines(text):
    return [line for line in text.splitlines()
            if not line.startswith(('import ', 'from '))]


result = tidy_imports(SOURCE)
try:
    ast.parse(result)
except SyntaxError as error:
    raise AssertionError('the tidied module does not parse: %s' % (error,))

assert imports_of(result) == [
    ('from', '__future__', 0, (('annotations', None),)),
    ('import', (('sys', None),)),
    ('import', (('json.decoder', None),)),
    ('import', (('collections.abc', 'abc'),)),
    ('from', 'typing', 0, (('Dict', None), ('Optional', None))),
    ('from', 'dataclasses', 0, (('dataclass', None),)),
    ('from', 'decimal', 0, (('Decimal', None),)),
    ('from', 're', 0, (('*', None),)),
], imports_of(result)

# Nothing outside an import statement may move, including the nested import
# of a name the module never uses.
assert body_lines(result) == body_lines(SOURCE), body_lines(result)
assert '    import string' in result, 'a function-level import was removed'

# A module with nothing to remove comes back byte for byte.
CLEAN = 'import sys\n\nLIMIT = sys.maxsize\n'
assert tidy_imports(CLEAN) == CLEAN

# When every import goes, only the statement lines go with them.
assert tidy_imports('import os\nimport sys\n\nVALUE = 1\n') == '\nVALUE = 1\n'

# A partially used import spanning several lines collapses to one.
WRAPPED = """from typing import (
    Dict,
    List,
)

value: Dict[str, int] = {}
"""
tidied = tidy_imports(WRAPPED)
assert imports_of(tidied) == [('from', 'typing', 0, (('Dict', None),))], \
    imports_of(tidied)
assert 'value: Dict[str, int] = {}' in tidied

# A star import stays even though its bindings are invisible.
assert tidy_imports('from os.path import *\n\nX = 1\n') == \
    'from os.path import *\n\nX = 1\n'

# An empty module is not a special case.
assert tidy_imports('') == ''
''',
    ),
    task(
        f"{FAMILY}-0207", FAMILY,
        prompt=(
            "Implement a Python function explain_assertion(expression, "
            "namespace) that renders the intermediate values behind a failed "
            "assertion the way a test framework does. expression is the "
            "source of one Python expression written on a single line, built "
            "only from names, attribute access, subscripts, calls, "
            "comparisons, boolean and arithmetic operators, conditional "
            "expressions, container displays and literals. namespace maps "
            "names to values. Return a (result, lines) pair. result is the "
            "value of the whole expression, evaluated with those names in "
            "scope, and an exception raised by the whole expression "
            "propagates. lines holds one 'source = repr' string for every "
            "sub-expression of the whole expression that is neither a "
            "literal nor the whole expression itself, where source is "
            "exactly the slice of expression that the sub-expression spans "
            "and repr is the repr of evaluating that slice on its own. "
            "Order the lines by where the sub-expression starts, and for two "
            "that start together put the longer first. A sub-expression that "
            "raises is rendered with the exception class in angle brackets, "
            "such as '<KeyError>', and does not stop the others. Raise "
            "ValueError if expression is not a single expression."
        ),
        validator=LOAD_CANDIDATE + require("explain_assertion") + r'''
def explained(expression, namespace):
    outcome = explain_assertion(expression, dict(namespace))
    assert isinstance(outcome, tuple) and len(outcome) == 2, (
        'expected a (result, lines) pair; got %r' % (outcome,)
    )
    return outcome[0], list(outcome[1])


class Sized:
    """A callable with a stable repr, so the expected text is fixed."""

    def __call__(self, values):
        return len(values)

    def __repr__(self):
        return '<sized>'


result, lines = explained('total == expected', {'total': 3, 'expected': 4})
assert result is False and lines == ['total = 3', 'expected = 4'], lines

# A literal contributes nothing; only the name is worth showing.
result, lines = explained('x + 1', {'x': 2})
assert result == 3 and lines == ['x = 2'], lines

# Two sub-expressions starting at the same column: the longer one first,
# because the reader wants the compound value before its parts.
result, lines = explained("user['name'] == 'bob'", {'user': {'name': 'ann'}})
assert result is False, result
assert lines == ["user['name'] = 'ann'", "user = {'name': 'ann'}"], lines

# The whole expression short-circuits, so the right operand never ran. The
# explanation still has to say what it would have been, without blowing up.
result, lines = explained("flag or data['k']", {'flag': True, 'data': {}})
assert result is True, result
assert lines == ['flag = True', "data['k'] = <KeyError>", 'data = {}'], lines

# The untaken branch of a conditional names something that does not exist.
result, lines = explained('a if ok else b', {'a': 1, 'ok': True})
assert result == 1, result
assert lines == ['a = 1', 'ok = True', 'b = <NameError>'], lines

# A call shows the call, the thing called and each argument.
result, lines = explained('size(items) > limit',
                          {'size': Sized(), 'items': [1, 2], 'limit': 5})
assert result is False, result
assert lines == ['size(items) = 2', 'size = <sized>', 'items = [1, 2]',
                 'limit = 5'], lines

# The same name twice is two occurrences, not one.
result, lines = explained('x + x', {'x': 2})
assert result == 4 and lines == ['x = 2', 'x = 2'], lines

# An exception from the whole expression is the caller's to see.
try:
    explained('missing + 1', {})
except NameError:
    pass
except ValueError:
    raise AssertionError('a valid expression was rejected as invalid')
else:
    raise AssertionError('an expression that raises returned a value')

for bad in ('x = 1', 'import os', 'a\nb', 'for x in y: pass', '('):
    try:
        explain_assertion(bad, {})
    except ValueError:
        pass
    else:
        raise AssertionError('accepted %r, which is not one expression' % bad)
''',
    ),
    task(
        f"{FAMILY}-0208", FAMILY,
        prompt=(
            "Implement a Python function rewrite_module_path(source, old, "
            "new) that updates one module for a package that moved, and "
            "returns the new source. old and new are dotted module paths. "
            "Rewrite an 'import old' or 'import old.sub' statement, and its "
            "'as' form, so the imported module starts with new instead of "
            "old, keeping any alias. Rewrite 'from old import name' and "
            "'from old.sub import name' the same way, leaving relative "
            "imports alone. Where a plain import bound a dotted name, every "
            "attribute reference through it must move too, so with old as "
            "'a' the expression a.sub.call() becomes new's equivalent. A "
            "module path that merely begins with the same characters, such "
            "as 'oldest' when old is 'old', is a different package and must "
            "not change, and neither may a string literal, a comment, or any "
            "other line of the module."
        ),
        validator=LOAD_CANDIDATE + require("rewrite_module_path") + r'''
import ast
import sys
import types


def install(path, **attributes):
    module = types.ModuleType(path)
    module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[path] = module
    if '.' in path:
        parent, _, leaf = path.rpartition('.')
        setattr(sys.modules[parent], leaf, module)
    return module


class Row:
    pass


fresh = install('fresh')
install('fresh.core', call=lambda: 'core')
install('fresh.helpers', assist=lambda: 'assist')
install('fresh.models', Row=Row)
fresh.registry = types.SimpleNamespace(name='registry')
install('staleness')
install('staleness.thing', tag=lambda: 'tag')

# The old package is deliberately absent: a reference the rewrite missed
# raises on import rather than quietly resolving to the code that moved.
SOURCE = """import stale.core
import stale.helpers as helpers
import staleness.thing
from stale.models import Row
from stale import registry

# stale.core moved, and this comment did not.
LABEL = "stale.core"


def run():
    return (stale.core.call(), helpers.assist(), staleness.thing.tag(),
            Row(), registry.name, LABEL)
"""


def imports_of(text):
    found = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            found.append(('import',
                          tuple((a.name, a.asname) for a in node.names)))
        elif isinstance(node, ast.ImportFrom):
            found.append(('from', node.module, node.level,
                          tuple((a.name, a.asname) for a in node.names)))
    return found


rewritten = rewrite_module_path(SOURCE, 'stale', 'fresh')
assert imports_of(rewritten) == [
    ('import', (('fresh.core', None),)),
    ('import', (('fresh.helpers', 'helpers'),)),
    ('import', (('staleness.thing', None),)),
    ('from', 'fresh.models', 0, (('Row', None),)),
    ('from', 'fresh', 0, (('registry', None),)),
], imports_of(rewritten)

namespace = {}
try:
    exec(compile(rewritten, '<module>', 'exec'), namespace)
    outcome = namespace['run']()
except Exception as error:
    raise AssertionError('the rewritten module does not run: %r' % (error,))

assert outcome[0] == 'core', 'the attribute reference was not moved'
assert outcome[1] == 'assist' and outcome[2] == 'tag', outcome
assert isinstance(outcome[3], Row) and outcome[4] == 'registry', outcome
assert outcome[5] == 'stale.core', 'a string literal was rewritten'
assert '# stale.core moved, and this comment did not.' in rewritten, \
    'a comment was rewritten'

# A package that only shares a prefix is a different package.
assert 'staleness.thing' in rewritten and 'freshness' not in rewritten

# Moving a submodule leaves its siblings where they are.
moved = rewrite_module_path(SOURCE, 'stale.core', 'fresh.core')
assert imports_of(moved) == [
    ('import', (('fresh.core', None),)),
    ('import', (('stale.helpers', 'helpers'),)),
    ('import', (('staleness.thing', None),)),
    ('from', 'stale.models', 0, (('Row', None),)),
    ('from', 'stale', 0, (('registry', None),)),
], imports_of(moved)
assert 'fresh.core.call()' in moved, 'the attribute chain did not follow'

# A package that does not appear leaves the module untouched.
assert rewrite_module_path(SOURCE, 'absent', 'other') == SOURCE

# A relative import names no package to move.
RELATIVE = 'from . import stale\nfrom .stale import thing\n'
assert rewrite_module_path(RELATIVE, 'stale', 'fresh') == RELATIVE
''',
    ),
    task(
        f"{FAMILY}-0209", FAMILY,
        prompt=(
            "Implement a Python function audit_resource_lifetimes(events) "
            "that finds resource-handling defects in a trace. events is a "
            "sequence of (action, resource_id) pairs in the order they "
            "happened, where action is 'acquire' or 'release'. Return a "
            "mapping with exactly the keys 'leaked', 'double_released', "
            "'reacquired' and 'out_of_order', each a list of resource ids. "
            "'leaked' holds the ids still held at the end, in the order they "
            "were acquired. 'double_released' holds each release of an id "
            "that is not currently held. 'reacquired' holds each acquire of "
            "an id already held, which does not acquire it a second time. "
            "'out_of_order' holds each release that closes an id other than "
            "the most recently acquired one still held, the pattern that "
            "defeats nesting-based cleanup; such a release still closes its "
            "resource. The last three are in the order the defect happened. "
            "Acquiring an id again after it was released is ordinary and is "
            "not a defect. Raise ValueError if an action is neither "
            "'acquire' nor 'release'."
        ),
        validator=LOAD_CANDIDATE + require("audit_resource_lifetimes") + r'''
EMPTY = {'leaked': [], 'double_released': [], 'reacquired': [],
         'out_of_order': []}


def audited(events):
    report = audit_resource_lifetimes(list(events))
    assert set(report) == set(EMPTY), (
        'expected exactly the four keys; got %r' % (sorted(report),)
    )
    return {key: list(value) for key, value in report.items()}


assert audited([]) == EMPTY

# Properly nested use reports nothing.
assert audited([('acquire', 'a'), ('acquire', 'b'),
                ('release', 'b'), ('release', 'a')]) == EMPTY

# Reusing an id after releasing it is ordinary.
assert audited([('acquire', 'a'), ('release', 'a'),
                ('acquire', 'a'), ('release', 'a')]) == EMPTY

# What was acquired and never released, in acquisition order.
assert audited([('acquire', 'b'), ('acquire', 'a'),
                ('release', 'a')])['leaked'] == ['b']
assert audited([('acquire', 'b'), ('acquire', 'a')])['leaked'] == ['b', 'a']

# Releasing what is not held, both a second time and never at all.
assert audited([('acquire', 'a'), ('release', 'a'),
                ('release', 'a')])['double_released'] == ['a']
assert audited([('release', 'ghost')]) == dict(EMPTY,
                                               double_released=['ghost'])

# Acquiring what is already held does not acquire it twice, so one release
# is enough to close it and there is nothing left over.
report = audited([('acquire', 'a'), ('acquire', 'a'), ('release', 'a')])
assert report == dict(EMPTY, reacquired=['a']), report

# Closing out of nesting order is still a close: nothing leaks here.
report = audited([('acquire', 'a'), ('acquire', 'b'),
                  ('release', 'a'), ('release', 'b')])
assert report == dict(EMPTY, out_of_order=['a']), report

# And several such closes are reported in the order they happened.
report = audited([('acquire', 'a'), ('acquire', 'b'), ('acquire', 'c'),
                  ('release', 'a'), ('release', 'b'), ('release', 'c')])
assert report == dict(EMPTY, out_of_order=['a', 'b']), report

# The innermost close is in order even when an outer one is still held.
report = audited([('acquire', 'a'), ('acquire', 'b'), ('release', 'b')])
assert report == dict(EMPTY, leaked=['a']), report

for bad in ('open', 'close', '', None):
    try:
        audit_resource_lifetimes([('acquire', 'a'), (bad, 'a')])
    except ValueError:
        pass
    else:
        raise AssertionError('accepted the unknown action %r' % (bad,))
''',
    ),
]
