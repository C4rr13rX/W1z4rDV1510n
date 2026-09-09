"""Held-out tasks: algorithms and data structures.

Every validator here executes the candidate against inputs the prompt does
not enumerate, including the degenerate cases (empty input, single element,
duplicate keys, cycles) that separate a memorised textbook body from a
working implementation. Several also assert a complexity-sensitive property
by size rather than by timing, because a wall-clock threshold would make the
verdict depend on host load and the contract admits no flaky cases.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "algorithms_data_structures"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python class LRUCache with a positive integer "
            "capacity. It must expose get(key) returning the stored value or "
            "None when absent, and put(key, value) which inserts or updates. "
            "Reading or writing a key makes it the most recently used. When "
            "an insertion would exceed capacity, evict the least recently "
            "used key. Both operations must run in amortised constant time "
            "regardless of the number of stored keys."
        ),
        validator=LOAD_CANDIDATE + require("LRUCache") + """
cache = LRUCache(2)
cache.put('a', 1)
cache.put('b', 2)
assert cache.get('a') == 1, 'stored key not returned'
cache.put('c', 3)
assert cache.get('b') is None, 'evicted the recently used key, not the LRU'
assert cache.get('a') == 1 and cache.get('c') == 3

# An update must refresh recency without growing the cache.
cache.put('a', 10)
cache.put('d', 4)
assert cache.get('c') is None, 'update did not refresh recency'
assert cache.get('a') == 10 and cache.get('d') == 4

# Capacity one degenerates to "keep only the last write".
single = LRUCache(1)
single.put('x', 1)
single.put('y', 2)
assert single.get('x') is None and single.get('y') == 2

# Constant-time behaviour, asserted structurally: the cache must never hold
# more than capacity entries no matter how many distinct keys pass through.
big = LRUCache(50)
for index in range(5000):
    big.put(index, index)
    assert big.get(index) == index
assert big.get(0) is None
live = sum(1 for index in range(5000) if big.get(index) is not None)
assert live == 50, f'capacity not enforced: {live} live entries'
""",
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function topological_order(nodes, edges) "
            "where nodes is a list of hashable identifiers and edges is a "
            "list of (before, after) pairs. Return a list containing every "
            "node exactly once such that each 'before' precedes its 'after'. "
            "If the constraints cannot all be satisfied because the graph "
            "contains a cycle, raise ValueError."
        ),
        validator=LOAD_CANDIDATE + require("topological_order") + """
def precedes(order, before, after):
    return order.index(before) < order.index(after)

nodes = ['a', 'b', 'c', 'd']
edges = [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')]
order = topological_order(nodes, edges)
assert sorted(order) == sorted(nodes), 'output is not a permutation of nodes'
for before, after in edges:
    assert precedes(order, before, after), f'{before} did not precede {after}'

# Disconnected nodes must still appear exactly once.
order = topological_order(['x', 'y', 'z'], [('x', 'y')])
assert sorted(order) == ['x', 'y', 'z'] and precedes(order, 'x', 'y')

# Empty input is valid and yields an empty order.
assert list(topological_order([], [])) == []

# A self-edge is the smallest cycle and must be rejected.
for bad in ([('a', 'a')], [('a', 'b'), ('b', 'a')],
            [('a', 'b'), ('b', 'c'), ('c', 'a')]):
    try:
        topological_order(['a', 'b', 'c'], bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'cycle {bad} was not rejected')

# A duplicated edge is not a cycle and must still succeed.
order = topological_order(['a', 'b'], [('a', 'b'), ('a', 'b')])
assert order == ['a', 'b']
""",
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function merge_intervals(intervals) taking a "
            "list of half-open [start, end) integer pairs in any order. "
            "Return the minimal sorted list of half-open intervals covering "
            "exactly the same points. Intervals that merely touch, such as "
            "[1, 3) and [3, 5), are contiguous and must be merged; empty "
            "intervals where start equals end cover nothing and must be "
            "dropped. Raise ValueError if any interval has start greater "
            "than end."
        ),
        validator=LOAD_CANDIDATE + require("merge_intervals") + """
assert merge_intervals([]) == []
assert merge_intervals([[1, 4]]) == [[1, 4]] or \\
    merge_intervals([[1, 4]]) == [(1, 4)], 'single interval not preserved'

def as_pairs(result):
    return [tuple(item) for item in result]

# Touching intervals are contiguous under half-open semantics.
assert as_pairs(merge_intervals([[1, 3], [3, 5]])) == [(1, 5)]
# Overlapping and out of order.
assert as_pairs(merge_intervals([[5, 8], [1, 4], [3, 6]])) == [(1, 8)]
# A gap of one point must not be closed.
assert as_pairs(merge_intervals([[1, 3], [4, 6]])) == [(1, 3), (4, 6)]
# Fully contained intervals disappear into their container.
assert as_pairs(merge_intervals([[1, 10], [2, 3], [4, 5]])) == [(1, 10)]
# Empty intervals cover nothing.
assert as_pairs(merge_intervals([[2, 2]])) == []
assert as_pairs(merge_intervals([[1, 3], [4, 4], [5, 7]])) == [(1, 3), (5, 7)]
# Negative coordinates are ordinary.
assert as_pairs(merge_intervals([[-5, -2], [-3, 0]])) == [(-5, 0)]

try:
    merge_intervals([[5, 1]])
except ValueError:
    pass
else:
    raise AssertionError('inverted interval was not rejected')

# Point coverage cross-check on a randomised-but-fixed instance.
import random
rng = random.Random(20260905)
raw = []
covered = set()
for _ in range(200):
    start = rng.randint(-100, 100)
    end = start + rng.randint(0, 12)
    raw.append([start, end])
    covered.update(range(start, end))
merged = as_pairs(merge_intervals(raw))
rebuilt = set()
for start, end in merged:
    rebuilt.update(range(start, end))
assert rebuilt == covered, 'merged set does not cover the same points'
assert merged == sorted(merged), 'result is not sorted'
for left, right in zip(merged, merged[1:]):
    assert left[1] < right[0], 'adjacent intervals were left unmerged'
""",
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python class RunningMedian with add(value) and "
            "median(). median() returns the middle value for an odd count "
            "and the arithmetic mean of the two middle values for an even "
            "count, and raises ValueError when nothing has been added. "
            "add must not re-sort the whole history on each call: adding n "
            "values in total must cost O(n log n), not O(n^2 log n)."
        ),
        validator=LOAD_CANDIDATE + require("RunningMedian") + """
stream = RunningMedian()
try:
    stream.median()
except ValueError:
    pass
else:
    raise AssertionError('median of an empty stream was not rejected')

stream.add(5)
assert stream.median() == 5
stream.add(15)
assert stream.median() == 10
stream.add(1)
assert stream.median() == 5
stream.add(3)
assert stream.median() == 4

# Cross-check against a naive reference over a fixed pseudo-random stream.
import random
rng = random.Random(99991)
reference = []
subject = RunningMedian()
for _ in range(600):
    value = rng.randint(-500, 500)
    reference.append(value)
    subject.add(value)
    ordered = sorted(reference)
    middle = len(ordered) // 2
    expected = (ordered[middle] if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2)
    got = subject.median()
    assert abs(got - expected) < 1e-9, f'median {got} != {expected}'

# Duplicates must not collapse: the median of five equal values is that value.
flat = RunningMedian()
for _ in range(5):
    flat.add(7)
assert flat.median() == 7
""",
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python class DisjointSet supporting find(item) "
            "returning a representative, union(a, b) merging two groups, "
            "connected(a, b), and group_count(). Items are created on first "
            "reference. union must be idempotent and find must keep repeated "
            "queries cheap by compressing paths, so a chain of 20000 unions "
            "still answers find without exceeding the recursion limit."
        ),
        validator=LOAD_CANDIDATE + require("DisjointSet") + """
sets = DisjointSet()
assert sets.group_count() == 0
assert sets.find('a') == sets.find('a'), 'find is not stable'
assert sets.group_count() == 1, 'first reference did not create a group'
assert not sets.connected('a', 'b')
assert sets.group_count() == 2

sets.union('a', 'b')
assert sets.connected('a', 'b') and sets.connected('b', 'a')
assert sets.group_count() == 1
sets.union('a', 'b')
assert sets.group_count() == 1, 'repeated union split or duplicated a group'

sets.union('c', 'd')
assert sets.group_count() == 2 and not sets.connected('a', 'c')
sets.union('b', 'c')
assert sets.group_count() == 1
for left in 'abcd':
    for right in 'abcd':
        assert sets.connected(left, right)

# A long chain must not blow the stack: path compression, not recursion depth.
chain = DisjointSet()
for index in range(20000):
    chain.union(index, index + 1)
assert chain.connected(0, 20000)
assert chain.group_count() == 1
assert chain.find(0) == chain.find(20000)
""",
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function search_rotated(values, target) where "
            "values is a list of distinct integers sorted ascending and then "
            "rotated left by an unknown amount. Return the index of target, "
            "or -1 when it is absent. The list may be rotated by zero. Do not "
            "scan every element: the number of elements examined must grow "
            "logarithmically with the length of the list."
        ),
        validator=LOAD_CANDIDATE + require("search_rotated") + """
assert search_rotated([], 1) == -1
assert search_rotated([3], 3) == 0
assert search_rotated([3], 4) == -1

base = [1, 3, 5, 7, 9, 11, 13]
for rotation in range(len(base)):
    rotated = base[rotation:] + base[:rotation]
    for value in base:
        index = search_rotated(rotated, value)
        assert index != -1 and rotated[index] == value, \\
            f'{value} not found in {rotated}'
    for missing in (0, 2, 14):
        assert search_rotated(rotated, missing) == -1, \\
            f'{missing} wrongly reported present in {rotated}'

# Logarithmic access, measured by counting reads rather than by clock time so
# the verdict does not depend on host load.
class Counting(list):
    reads = 0
    def __getitem__(self, index):
        if isinstance(index, int):
            Counting.reads += 1
        return list.__getitem__(self, index)

size = 1 << 16
values = list(range(1, 2 * size, 2))
rotated = values[size // 3:] + values[:size // 3]
probe = Counting(rotated)
Counting.reads = 0
found = search_rotated(probe, rotated[-1])
assert found == len(rotated) - 1
assert Counting.reads < 200, \\
    f'examined {Counting.reads} elements of {size}; that is a linear scan'
""",
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function window_maxima(values, width) "
            "returning the maximum of every contiguous window of the given "
            "width, left to right. Raise ValueError when width is less than "
            "one, and return an empty list when width exceeds the number of "
            "values. Total work must be linear in the number of values: "
            "re-scanning each window is not acceptable for large inputs."
        ),
        validator=LOAD_CANDIDATE + require("window_maxima") + """
assert list(window_maxima([1, 3, 2, 5, 4], 1)) == [1, 3, 2, 5, 4]
assert list(window_maxima([1, 3, 2, 5, 4], 2)) == [3, 3, 5, 5]
assert list(window_maxima([1, 3, 2, 5, 4], 3)) == [3, 5, 5]
assert list(window_maxima([1, 3, 2, 5, 4], 5)) == [5]
assert list(window_maxima([1, 3, 2, 5, 4], 6)) == []
assert list(window_maxima([], 2)) == []

for bad in (0, -1):
    try:
        window_maxima([1, 2, 3], bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'width {bad} was not rejected')

# Plateaus and negatives are where naive deque handling usually breaks.
assert list(window_maxima([2, 2, 2, 2], 2)) == [2, 2, 2]
assert list(window_maxima([-5, -1, -7, -3], 2)) == [-1, -1, -3]

import random
rng = random.Random(4242)
values = [rng.randint(-1000, 1000) for _ in range(3000)]
for width in (1, 2, 7, 64, 999):
    expected = [max(values[i:i + width])
                for i in range(len(values) - width + 1)]
    assert list(window_maxima(values, width)) == expected, \\
        f'wrong maxima at width {width}'

# Linear work, counted by comparisons the candidate performs on the elements.
class Counted:
    comparisons = 0
    def __init__(self, value):
        self.value = value
    def _cmp(self, other):
        Counted.comparisons += 1
        return other.value if isinstance(other, Counted) else other
    def __lt__(self, other):
        return self.value < self._cmp(other)
    def __le__(self, other):
        return self.value <= self._cmp(other)
    def __gt__(self, other):
        return self.value > self._cmp(other)
    def __ge__(self, other):
        return self.value >= self._cmp(other)
    def __eq__(self, other):
        return self.value == self._cmp(other)

wrapped = [Counted(value) for value in values]
Counted.comparisons = 0
result = list(window_maxima(wrapped, 500))
assert len(result) == len(values) - 499
assert Counted.comparisons < 40 * len(values), (
    f'{Counted.comparisons} comparisons for {len(values)} values is not '
    'linear work'
)
""",
        timeout_seconds=60.0,
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python class PrefixIndex with insert(word), "
            "contains(word), count_with_prefix(prefix) returning how many "
            "stored words start with the prefix, and remove(word) returning "
            "True when a stored word was removed and False otherwise. "
            "Inserting the same word twice must not double its contribution "
            "to prefix counts. The empty string is a valid prefix matching "
            "every stored word."
        ),
        validator=LOAD_CANDIDATE + require("PrefixIndex") + """
index = PrefixIndex()
assert index.count_with_prefix('') == 0
assert not index.contains('anything')
assert index.remove('absent') is False

for word in ('car', 'cart', 'carbon', 'dog'):
    index.insert(word)
assert index.count_with_prefix('') == 4
assert index.count_with_prefix('car') == 3
assert index.count_with_prefix('cart') == 1
assert index.count_with_prefix('d') == 1
assert index.count_with_prefix('z') == 0
assert index.contains('car') and not index.contains('ca')

# Re-inserting must be idempotent for counting purposes.
index.insert('car')
assert index.count_with_prefix('car') == 3, 'duplicate insert double-counted'

assert index.remove('car') is True
assert not index.contains('car')
assert index.count_with_prefix('car') == 2, 'removal did not update counts'
assert index.contains('cart') and index.contains('carbon')
assert index.remove('car') is False

# Removing a word must not damage words that share its path.
assert index.remove('carbon') is True
assert index.count_with_prefix('car') == 1
assert index.contains('cart')

# The empty string is storable and is its own prefix.
index.insert('')
assert index.contains('')
assert index.count_with_prefix('') == 3

import random
rng = random.Random(777)
alphabet = 'abc'
words = {''.join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
         for _ in range(400)}
subject = PrefixIndex()
for word in words:
    subject.insert(word)
for prefix in ('', 'a', 'ab', 'abc', 'cba', 'bb'):
    expected = sum(1 for word in words if word.startswith(prefix))
    assert subject.count_with_prefix(prefix) == expected, \\
        f'prefix {prefix!r}: expected {expected}'
""",
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python function align(source, target) taking two "
            "strings and returning a list of (operation, source_character, "
            "target_character) tuples. operation is one of 'match', "
            "'replace', 'delete' or 'insert'. 'match' and 'replace' each "
            "consume one character from both strings, and 'match' requires "
            "those two characters to be equal. 'delete' consumes one source "
            "character and reports None as its target character; 'insert' "
            "consumes one target character and reports None as its source "
            "character. Reading the non-None source characters of the "
            "returned list in order must reproduce source exactly, and the "
            "non-None target characters must reproduce target. Among all "
            "lists satisfying that, return one whose number of entries that "
            "are not 'match' is as small as possible. Either string may be "
            "empty."
        ),
        validator=LOAD_CANDIDATE + require("align") + """
def cost_of(source, target):
    \"\"\"Validate the alignment structurally and return its non-match count.\"\"\"
    steps = align(source, target)
    assert isinstance(steps, list), 'align must return a list'
    rebuilt_source, rebuilt_target = [], []
    for entry in steps:
        assert isinstance(entry, tuple) and len(entry) == 3, \\
            f'entry is not a 3-tuple: {entry!r}'
        operation, before, after = entry
        assert operation in ('match', 'replace', 'delete', 'insert'), \\
            f'unknown operation {operation!r}'
        if operation in ('match', 'replace'):
            assert before is not None and after is not None, \\
                f'{operation} must consume from both strings: {entry!r}'
            if operation == 'match':
                assert before == after, \\
                    f'match of unequal characters {before!r} and {after!r}'
            rebuilt_source.append(before)
            rebuilt_target.append(after)
        elif operation == 'delete':
            assert before is not None and after is None, \\
                f'delete must consume source only: {entry!r}'
            rebuilt_source.append(before)
        else:
            assert before is None and after is not None, \\
                f'insert must consume target only: {entry!r}'
            rebuilt_target.append(after)
    assert ''.join(rebuilt_source) == source, \\
        f'alignment does not reproduce source {source!r}'
    assert ''.join(rebuilt_target) == target, \\
        f'alignment does not reproduce target {target!r}'
    return sum(1 for operation, _, _ in steps if operation != 'match')

# The oracle enumerates alignments exhaustively with a branch-and-bound
# cutoff. It shares no recurrence with the dynamic program a candidate will
# write, so it cannot inherit that program's mistakes -- in particular it
# cannot agree with a traceback that reconstructs a non-minimal path.
def minimal(source, target):
    best = [len(source) + len(target)]
    def walk(i, j, spent):
        if spent >= best[0]:
            return
        if i == len(source) and j == len(target):
            best[0] = spent
            return
        if i < len(source) and j < len(target):
            walk(i + 1, j + 1, spent + (0 if source[i] == target[j] else 1))
        if i < len(source):
            walk(i + 1, j, spent + 1)
        if j < len(target):
            walk(i, j + 1, spent + 1)
    walk(0, 0, 0)
    return best[0]

for source, target in [
        ('', ''), ('a', ''), ('', 'a'), ('abc', 'abc'), ('abc', 'xyz'),
        ('kitten', 'sitting'), ('flaw', 'lawn'), ('ab', 'ba'),
        ('aaa', 'aa'), ('xabcy', 'abc'), ('abcdef', 'abcdef'),
]:
    expected = minimal(source, target)
    actual = cost_of(source, target)
    assert actual == expected, \\
        f'align({source!r}, {target!r}) used {actual} edits, {expected} suffice'

# A single replacement must be preferred to a delete plus an insert; both
# reproduce the strings, and only one of them is minimal.
steps = align('cat', 'cot')
assert sum(1 for operation, _, _ in steps if operation != 'match') == 1
assert any(operation == 'replace' for operation, _, _ in steps), \\
    'a one-character substitution was not expressed as a replace'

# Long inputs whose answer is fixed by construction rather than by an oracle.
assert cost_of('a' * 40, 'a' * 40) == 0
assert cost_of('a' * 30, 'b' * 30) == 30
assert cost_of('', 'z' * 25) == 25
base = 'abcdefghij' * 3
altered = 'X' + base[1:14] + 'Y' + base[15:28] + 'Z' + base[29:]
assert len(altered) == len(base)
assert cost_of(base, altered) == 3
""",
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python function longest_increasing(values) taking a "
            "list of integers and returning a list that is a subsequence of "
            "values -- obtained by deleting zero or more elements without "
            "reordering the rest -- whose elements are strictly increasing "
            "and whose length is as large as possible. When several "
            "subsequences share that maximum length, return any one of them. "
            "Return an empty list when values is empty."
        ),
        validator=LOAD_CANDIDATE + require("longest_increasing") + """
def check(values):
    result = list(longest_increasing(list(values)))
    # A subsequence, verified by walking the original left to right.
    position = 0
    for item in result:
        while position < len(values) and values[position] != item:
            position += 1
        assert position < len(values), \\
            f'{result} is not a subsequence of {values}'
        position += 1
    for earlier, later in zip(result, result[1:]):
        assert earlier < later, f'{result} is not strictly increasing'
    return len(result)

# Oracle: every subset, largest first. Exhaustive rather than any form of the
# algorithm under test, so it cannot share a blind spot with it.
def longest_by_search(values):
    best = 0
    for mask in range(1 << len(values)):
        picked = [values[i] for i in range(len(values)) if mask >> i & 1]
        if len(picked) > best and all(a < b for a, b in zip(picked, picked[1:])):
            best = len(picked)
    return best

for values in [
        [], [7], [3, 3, 3], [5, 4, 3, 2, 1], [1, 2, 3, 4],
        [10, 9, 2, 5, 3, 7, 101, 18], [0, 1, 0, 3, 2, 3],
        [4, 10, 4, 3, 8, 9], [2, 2, 2, 2, 3], [1, 3, 2, 4, 3, 5],
]:
    expected = longest_by_search(values)
    actual = check(values)
    assert actual == expected, \\
        f'longest_increasing({values}) returned length {actual}, {expected} exists'

# Duplicates must not be chained: 'strictly' is the whole contract here.
assert check([1, 1, 1, 1]) == 1

# A large input whose answer is known by construction rather than by an
# oracle. Each block descends, and every block sits entirely above the one
# before it, so an increasing subsequence takes at most one element per block
# and taking exactly one from each achieves that bound.
blocks, width = 25, 40
values = []
for index in range(blocks):
    base = index * 1000
    values.extend(range(base + width, base, -1))
assert len(values) == blocks * width
assert check(values) == blocks, 'block construction bound not reached'
""",
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement a Python class RangeSum(values) over a sequence of "
            "value objects that support + and - with each other. The class "
            "must assume nothing else about them. Provide update(index, "
            "value) replacing the element at index, and query(low, high) "
            "returning the sum of the half-open range [low, high), which is "
            "a value equal to 0 when low equals high. Constructing the "
            "instance may perform a number of additions proportional to the "
            "element count, but update and query must each perform a number "
            "of additions and subtractions proportional only to the "
            "logarithm of it. Concretely: the caller builds an instance over "
            "4096 elements, then performs 4096 updates and 4096 queries, and "
            "requires fewer than 400000 additions and subtractions on the "
            "stored values in total including construction. An empty "
            "sequence is allowed, and query(0, 0) is then the only legal "
            "query."
        ),
        validator=LOAD_CANDIDATE + require("RangeSum") + """
OPERATIONS = [0]

class Counted:
    \"\"\"A value that reports how often it took part in an addition.

    The complexity claim in the prompt is the contract, so it is measured
    rather than timed -- a wall-clock threshold would make the verdict depend
    on host load, which the acceptance contract refuses as flaky.
    \"\"\"
    __slots__ = ('_value',)

    def __init__(self, value):
        self._value = value

    @staticmethod
    def _raw(other):
        return other._value if isinstance(other, Counted) else other

    def __add__(self, other):
        OPERATIONS[0] += 1
        return Counted(self._value + Counted._raw(other))

    __radd__ = __add__

    def __sub__(self, other):
        OPERATIONS[0] += 1
        return Counted(self._value - Counted._raw(other))

    def __rsub__(self, other):
        OPERATIONS[0] += 1
        return Counted(Counted._raw(other) - self._value)

    def __eq__(self, other):
        return self._value == Counted._raw(other)

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        return f'Counted({self._value})'

# Small cases first: correctness before any claim about how it is achieved.
empty = RangeSum([])
assert empty.query(0, 0) == 0, 'empty range must sum to zero'

single = RangeSum([Counted(7)])
assert single.query(0, 1) == 7
assert single.query(0, 0) == 0 and single.query(1, 1) == 0
single.update(0, Counted(-2))
assert single.query(0, 1) == -2, 'update did not replace the element'

mirror = [3, 1, 4, 1, 5, 9, 2, 6]
subject = RangeSum([Counted(item) for item in mirror])
for low in range(len(mirror) + 1):
    for high in range(low, len(mirror) + 1):
        assert subject.query(low, high) == sum(mirror[low:high]), \\
            f'query({low}, {high}) is wrong'
subject.update(3, Counted(100))
mirror[3] = 100
for low, high in ((0, 8), (3, 4), (2, 6), (4, 8), (0, 3)):
    assert subject.query(low, high) == sum(mirror[low:high]), \\
        f'query({low}, {high}) is wrong after an update'

# The measured claim. A prefix-sum array rebuilds on every update and a plain
# list re-adds the whole range on every query; both exceed this budget by two
# orders of magnitude, while a logarithmic structure stays far beneath it.
size = 4096
mirror = [(index * index) % 97 for index in range(size)]
OPERATIONS[0] = 0
big = RangeSum([Counted(item) for item in mirror])
for step in range(size):
    index = (step * 1237) % size
    replacement = (step * 31) % 89
    big.update(index, Counted(replacement))
    mirror[index] = replacement
    low = (step * 17) % size
    high = low + ((step * 53) % (size - low)) + 1 if low < size else size
    high = min(high, size)
    assert big.query(low, high) == sum(mirror[low:high]), \\
        f'query({low}, {high}) is wrong at step {step}'
used = OPERATIONS[0]
assert used < 400000, \\
    f'{used} value additions and subtractions, budget is 400000'
""",
    ),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement a Python function strong_components(nodes, edges) "
            "where nodes is a list of distinct hashable identifiers and "
            "edges is a list of (source, target) pairs drawn from nodes. Two "
            "nodes belong to the same component when each is reachable from "
            "the other by following edges forwards; every node is reachable "
            "from itself, so a node with no edges forms a component alone. "
            "Return a list of components, each component being the list of "
            "its nodes sorted ascending, ordered so that whenever an edge "
            "joins two different components the component holding that "
            "edge's target appears before the component holding its source. "
            "Return an empty list when nodes is empty."
        ),
        validator=LOAD_CANDIDATE + require("strong_components") + """
def ground_truth(nodes, edges):
    \"\"\"Mutual reachability by brute-force traversal from every node.

    This shares no structure with the single depth-first pass a candidate
    will write, so it cannot inherit that pass's classic defect of closing a
    component at the wrong root.
    \"\"\"
    successors = {node: set() for node in nodes}
    for source, target in edges:
        successors[source].add(target)
    reaches = {}
    for start in nodes:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for following in successors[node]:
                if following not in seen:
                    seen.add(following)
                    stack.append(following)
        reaches[start] = seen
    groups = {}
    for node in nodes:
        key = frozenset(other for other in nodes
                        if other in reaches[node] and node in reaches[other])
        groups[key] = sorted(key)
    return sorted(groups.values())

def check(nodes, edges):
    result = strong_components(list(nodes), list(edges))
    assert isinstance(result, list), 'strong_components must return a list'
    parts = [list(component) for component in result]
    for component in parts:
        assert component == sorted(component), \\
            f'component {component} is not sorted ascending'
    assert sorted(parts) == ground_truth(nodes, edges), \\
        f'components {sorted(parts)} do not match mutual reachability'

    placement = {}
    for index, component in enumerate(parts):
        for node in component:
            placement[node] = index
    for source, target in edges:
        if placement[source] != placement[target]:
            assert placement[target] < placement[source], \\
                (f'edge {source}->{target} runs from component '
                 f'{placement[source]} to {placement[target]}, which is not '
                 'reverse topological order')

check([], [])
check(['a'], [])
check(['a'], [('a', 'a')])
check(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')])
check(['a', 'b', 'c'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
check(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'a'), ('c', 'd'), ('d', 'c')])
check(['a', 'b', 'c', 'd'],
      [('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'd'), ('d', 'c')])
check(['a', 'b', 'c', 'd', 'e'],
      [('a', 'b'), ('b', 'c'), ('c', 'a'), ('c', 'd'), ('d', 'e')])
# Undirected connectivity would merge these two cycles; direction must not.
check(['p', 'q', 'r', 's'],
      [('p', 'q'), ('q', 'p'), ('r', 's'), ('s', 'r'), ('p', 'r'), ('q', 's')])
check(['n1', 'n2', 'n3', 'n4', 'n5', 'n6'],
      [('n1', 'n2'), ('n2', 'n3'), ('n3', 'n1'), ('n4', 'n5'),
       ('n5', 'n6'), ('n6', 'n4'), ('n3', 'n4')])
check([3, 1, 2], [(1, 2), (2, 1), (3, 1)])
""",
    ),
    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement a Python function prefix_code(frequencies) taking a "
            "dict mapping each distinct symbol to a positive integer count. "
            "Return a dict mapping every symbol to a non-empty code string "
            "made only of the characters '0' and '1', such that no symbol's "
            "code is a prefix of another symbol's code and the total, "
            "summed over symbols, of count multiplied by code length is as "
            "small as possible. When only one symbol is supplied its code "
            "must be a single character. Raise ValueError if frequencies is "
            "empty or any count is not a positive integer. Several optimal "
            "codes exist for most inputs; return any of them."
        ),
        validator=LOAD_CANDIDATE + require("prefix_code") + """
_SHAPES = {}

def shapes(leaves):
    \"\"\"Every full binary tree with the given number of leaves.\"\"\"
    if leaves == 1:
        return [None]
    if leaves in _SHAPES:
        return _SHAPES[leaves]
    trees = []
    for left in range(1, leaves):
        for first in shapes(left):
            for second in shapes(leaves - left):
                trees.append((first, second))
    _SHAPES[leaves] = trees
    return trees

def leaf_depths(tree, depth=0):
    if tree is None:
        return [depth]
    return leaf_depths(tree[0], depth + 1) + leaf_depths(tree[1], depth + 1)

def optimal_total(counts):
    \"\"\"Least weighted external path length, by exhaustive tree search.

    Enumerating shapes shares nothing with the greedy merge a candidate will
    write, so an implementation that merges in the wrong order cannot agree
    with this number by construction.
    \"\"\"
    if len(counts) == 1:
        return counts[0]
    heaviest = sorted(counts, reverse=True)
    best = None
    for tree in shapes(len(counts)):
        depths = sorted(leaf_depths(tree))
        total = sum(count * depth for count, depth in zip(heaviest, depths))
        best = total if best is None else min(best, total)
    return best

def check(frequencies):
    codes = prefix_code(dict(frequencies))
    assert set(codes) == set(frequencies), \\
        'the returned codes do not cover exactly the supplied symbols'
    for symbol, code in codes.items():
        assert isinstance(code, str) and code, f'{symbol!r} has no code'
        assert set(code) <= {'0', '1'}, f'{symbol!r} has a non-binary code'
    ordered = sorted(codes.values(), key=len)
    for index, code in enumerate(ordered):
        for longer in ordered[index + 1:]:
            assert not longer.startswith(code), \\
                f'code {code} is a prefix of {longer}'
    total = sum(frequencies[symbol] * len(codes[symbol]) for symbol in codes)
    expected = optimal_total(list(frequencies.values()))
    assert total == expected, \\
        f'encoded length {total}, but {expected} is achievable'

check({'a': 5})
check({'a': 1, 'b': 1})
check({'a': 1, 'b': 1, 'c': 1, 'd': 1})
# Merging the two ones and then pairing the result with the next symbol in
# the ORIGINAL order costs 16 here; re-inserting the merged weight and always
# taking the two smallest costs 14.
check({'a': 1, 'b': 1, 'c': 2, 'd': 4})
check({'a': 1, 'b': 1, 'c': 2, 'd': 4, 'e': 8})
check({'a': 45, 'b': 13, 'c': 12, 'd': 16, 'e': 9, 'f': 5})
check({'x': 7, 'y': 7, 'z': 7, 'w': 1})
check({chr(97 + index): index * index + 1 for index in range(7)})

# A lone symbol still needs a bit; an empty code is a prefix of everything.
assert len(prefix_code({'only': 3})['only']) == 1

for bad in ({}, {'a': 0}, {'a': -1}, {'a': 1.5}, {'a': True, 'b': 2}):
    try:
        prefix_code(dict(bad))
    except ValueError:
        continue
    raise AssertionError(f'{bad!r} was not rejected')
""",
    ),
    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement a Python function select_compatible(intervals) taking "
            "a list of [start, end) half-open integer pairs. Return a "
            "largest possible list of intervals chosen from the input that "
            "do not overlap one another, sorted ascending by start. "
            "Intervals that merely touch, such as [1, 3) and [3, 5), do not "
            "overlap. Two identical intervals do overlap, so at most one of "
            "them may be chosen. Raise ValueError if any interval has start "
            "greater than or equal to end. Return an empty list for empty "
            "input."
        ),
        validator=LOAD_CANDIDATE + require("select_compatible") + """
def largest_by_search(intervals):
    \"\"\"Biggest compatible subset, by enumerating every subset.\"\"\"
    best = 0
    for mask in range(1 << len(intervals)):
        picked = sorted(
            (intervals[i] for i in range(len(intervals)) if mask >> i & 1),
            key=lambda pair: (pair[0], pair[1]),
        )
        if len(picked) <= best:
            continue
        if all(earlier[1] <= later[0]
               for earlier, later in zip(picked, picked[1:])):
            best = len(picked)
    return best

def check(intervals):
    result = select_compatible([list(pair) for pair in intervals])
    chosen = [tuple(pair) for pair in result]
    assert chosen == sorted(chosen, key=lambda pair: pair[0]), \\
        f'{chosen} is not sorted ascending by start'
    available = {}
    for pair in intervals:
        available[tuple(pair)] = available.get(tuple(pair), 0) + 1
    for pair in chosen:
        assert available.get(pair, 0) > 0, \\
            f'{pair} was not available in the input'
        available[pair] -= 1
    for earlier, later in zip(chosen, chosen[1:]):
        assert earlier[1] <= later[0], f'{earlier} overlaps {later}'
    expected = largest_by_search([tuple(pair) for pair in intervals])
    assert len(chosen) == expected, \\
        f'chose {len(chosen)} intervals from {intervals}, {expected} fit'

check([])
check([[0, 1]])
check([[1, 3], [3, 5], [5, 7]])
# Choosing by earliest start takes [0, 10) and blocks both others.
check([[0, 10], [1, 2], [3, 4]])
# Choosing by shortest duration takes [3, 5), which blocks both others.
check([[0, 4], [3, 5], [4, 8]])
check([[2, 4], [2, 4], [2, 4]])
check([[1, 4], [2, 3], [3, 6], [5, 7], [6, 9], [8, 10]])
check([[0, 100], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
check([[5, 6], [1, 2], [3, 4], [0, 7]])
check([[-5, -1], [-2, 3], [0, 4], [3, 9]])

for bad in ([[3, 3]], [[5, 2]], [[0, 1], [4, 4]], [[0, 1], [9, 8]]):
    try:
        select_compatible([list(pair) for pair in bad])
    except ValueError:
        continue
    raise AssertionError(f'{bad} was not rejected')
""",
    ),
    task(
        f"{FAMILY}-0015", FAMILY,
        prompt=(
            "Implement a Python function min_largest_part(weights, parts) "
            "where weights is a non-empty list of non-negative integers and "
            "parts is the number of contiguous, non-empty groups the list "
            "must be divided into while preserving order. Return the "
            "smallest value the largest group sum can take over all such "
            "divisions. Raise ValueError when weights is empty, or when "
            "parts is less than 1 or greater than the number of weights."
        ),
        validator=LOAD_CANDIDATE + require("min_largest_part") + """
import itertools

def smallest_by_search(weights, parts):
    \"\"\"Try every set of cut positions; no search over answers is involved.\"\"\"
    best = None
    for cuts in itertools.combinations(range(1, len(weights)), parts - 1):
        bounds = (0,) + cuts + (len(weights),)
        largest = max(sum(weights[bounds[i]:bounds[i + 1]])
                      for i in range(parts))
        best = largest if best is None else min(best, largest)
    return best

for weights, parts in [
        ([1], 1), ([7, 2, 5, 10, 8], 2), ([1, 2, 3, 4, 5], 2),
        ([1, 4, 4], 3), ([2, 3, 1, 2, 4, 3], 3), ([0, 0, 0, 5], 2),
        ([10, 1, 1, 1, 1, 1, 1, 1, 1, 10], 3),
        ([5, 5, 5, 5, 5, 5], 4), ([1, 1, 1, 1, 1, 1, 1, 9], 2),
        ([9, 1, 1, 1, 1, 1, 1, 1], 2), ([3, 1, 4, 1, 5, 9, 2, 6], 4),
]:
    expected = smallest_by_search(weights, parts)
    actual = min_largest_part(list(weights), parts)
    assert actual == expected, \\
        f'min_largest_part({weights}, {parts}) gave {actual}, not {expected}'

# The two degenerate divisions, where the answer needs no search at all.
sample = [4, 8, 1, 9, 3]
assert min_largest_part(list(sample), 1) == sum(sample)
assert min_largest_part(list(sample), len(sample)) == max(sample)

# Larger inputs whose answers follow from the construction. Equal weights
# divide as evenly as the group count allows, so the largest group holds
# ceil(len / parts) of them.
assert min_largest_part([5] * 35, 7) == 25
assert min_largest_part([1] * 100, 7) == 15
assert min_largest_part([7] * 13, 13) == 7
assert min_largest_part([0] * 50, 3) == 0
# One weight dominates every division it can possibly be placed in.
assert min_largest_part([1] * 20 + [500] + [1] * 20, 3) == 500

for weights, parts in (([], 1), ([1, 2], 0), ([1, 2], 3), ([1, 2], -1),
                       ([], 0)):
    try:
        min_largest_part(list(weights), parts)
    except ValueError:
        continue
    raise AssertionError(f'({weights}, {parts}) was not rejected')
""",
    ),
    task(
        f"{FAMILY}-0016", FAMILY,
        prompt=(
            "Implement a Python function skyline(buildings) taking a list of "
            "(left, right, height) integer triples describing rectangles "
            "standing on the ground and occupying the half-open horizontal "
            "span [left, right). Return the outline as a list of (x, height) "
            "pairs in ascending x, where each pair marks a position at which "
            "the outline's height becomes that height. Consecutive pairs "
            "must never repeat a height, and the final pair must have height "
            "0. Raise ValueError if any building has left greater than or "
            "equal to right, or height less than or equal to 0. Return an "
            "empty list when there are no buildings."
        ),
        validator=LOAD_CANDIDATE + require("skyline") + """
def expected_outline(buildings):
    \"\"\"Sample the height function at every integer position.

    Spans are half-open with integer bounds, so the height is constant on
    each unit interval and sampling is exact. This shares nothing with the
    event sweep a candidate will write, so it cannot inherit that sweep's
    classic defect of emitting a point where the height did not change.
    \"\"\"
    if not buildings:
        return []
    lowest = min(left for left, _, _ in buildings)
    highest = max(right for _, right, _ in buildings)
    points, previous = [], 0
    for x in range(lowest, highest + 1):
        tallest = max(
            [height for left, right, height in buildings if left <= x < right],
            default=0,
        )
        if tallest != previous:
            points.append((x, tallest))
            previous = tallest
    return points

def check(buildings):
    result = [tuple(pair) for pair in skyline([tuple(b) for b in buildings])]
    expected = expected_outline(buildings)
    assert result == expected, \\
        f'skyline({buildings}) gave {result}, expected {expected}'
    for earlier, later in zip(result, result[1:]):
        assert earlier[0] < later[0], f'{result} is not ascending in x'
        assert earlier[1] != later[1], f'{result} repeats a height'
    if result:
        assert result[-1][1] == 0, f'{result} does not return to the ground'

check([])
check([(0, 2, 3)])
check([(0, 5, 4), (1, 2, 9)])
check([(1, 4, 3), (4, 7, 3)])
check([(1, 4, 3), (4, 7, 5)])
check([(0, 4, 2), (2, 6, 2), (4, 8, 2)])
check([(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)])
check([(0, 10, 5), (2, 4, 5), (6, 8, 5)])
check([(1, 3, 4), (1, 3, 4), (1, 3, 4)])
check([(0, 3, 1), (1, 5, 2), (2, 7, 3), (3, 9, 2), (4, 11, 1)])
check([(-5, -2, 6), (-3, 1, 4), (0, 3, 9)])
check([(0, 1, 1), (2, 3, 1), (4, 5, 1)])

for bad in ([(3, 3, 5)], [(5, 2, 1)], [(0, 2, 0)], [(0, 2, -4)],
            [(0, 2, 3), (7, 7, 1)]):
    try:
        skyline([tuple(b) for b in bad])
    except ValueError:
        continue
    raise AssertionError(f'{bad} was not rejected')
""",
    ),
    task(
        f"{FAMILY}-0017", FAMILY,
        prompt=(
            "Implement a Python function min_window(text, needed) taking two "
            "strings. Return the shortest contiguous substring of text that "
            "contains every character of needed at least as many times as "
            "needed itself contains it. When several substrings share the "
            "shortest length, return the one beginning earliest in text. "
            "Characters of text that do not appear in needed may appear "
            "freely inside the window. Return the empty string when no such "
            "substring exists, and also when needed is empty."
        ),
        validator=LOAD_CANDIDATE + require("min_window") + """
def shortest_by_search(text, needed):
    \"\"\"Every substring, shortest and then earliest.\"\"\"
    if not needed:
        return ''
    want = {}
    for character in needed:
        want[character] = want.get(character, 0) + 1
    best = None
    for start in range(len(text)):
        have = {}
        missing = len(want)
        for end in range(start, len(text)):
            character = text[end]
            if character in want:
                have[character] = have.get(character, 0) + 1
                if have[character] == want[character]:
                    missing -= 1
            if missing == 0:
                length = end - start + 1
                if best is None or length < best[0]:
                    best = (length, start)
                break
    return '' if best is None else text[best[1]:best[1] + best[0]]

def check(text, needed):
    expected = shortest_by_search(text, needed)
    actual = min_window(text, needed)
    assert actual == expected, \\
        f'min_window({text!r}, {needed!r}) gave {actual!r}, expected {expected!r}'

check('ADOBECODEBANC', 'ABC')
check('a', 'a')
check('a', 'aa')
check('', 'a')
check('abc', '')
check('', '')
check('aa', 'aa')
check('bba', 'ab')
check('cabwefgewcwaefgcf', 'cae')
check('abcdefghij', 'jih')
# Repeated requirements: a window holding one 'a' does not satisfy 'aa'.
check('aabbaa', 'aab')
check('xxaxxbxxaxxbxx', 'ab')
# Two windows of equal shortest length; the earlier one is the contract.
check('abXXXba', 'ab')
check('zzzz', 'z')
check('the quick brown fox', 'oq')

import random
rng = random.Random(20260909)
for _ in range(12):
    text = ''.join(rng.choice('abcd') for _ in range(rng.randint(0, 120)))
    needed = ''.join(rng.choice('abc') for _ in range(rng.randint(0, 5)))
    check(text, needed)
""",
    ),
    task(
        f"{FAMILY}-0018", FAMILY,
        prompt=(
            "Implement a Python function best_selection(items, capacity) "
            "where items is a list of (weight, value) pairs of non-negative "
            "integers and capacity is a non-negative integer. Return a tuple "
            "(total_value, indices) in which indices is the ascending list "
            "of positions of a subset of items whose combined weight does "
            "not exceed capacity and whose combined value is as large as "
            "possible, and total_value is that subset's combined value. Each "
            "item may be taken at most once. When several subsets reach the "
            "maximum value return any one of them; the empty subset is a "
            "valid answer. Raise ValueError if capacity is negative or any "
            "weight or value is negative."
        ),
        validator=LOAD_CANDIDATE + require("best_selection") + """
def best_by_search(items, capacity):
    \"\"\"Every subset, which cannot share a defect with a table-filling scan.\"\"\"
    best = 0
    for mask in range(1 << len(items)):
        weight = value = 0
        for index in range(len(items)):
            if mask >> index & 1:
                weight += items[index][0]
                value += items[index][1]
        if weight <= capacity and value > best:
            best = value
    return best

def check(items, capacity):
    total, indices = best_selection([tuple(pair) for pair in items], capacity)
    indices = list(indices)
    assert indices == sorted(indices), f'{indices} is not ascending'
    assert len(set(indices)) == len(indices), f'{indices} repeats an item'
    for index in indices:
        assert 0 <= index < len(items), f'{index} is not an item position'
    weight = sum(items[index][0] for index in indices)
    value = sum(items[index][1] for index in indices)
    assert weight <= capacity, \\
        f'selection weighs {weight}, capacity is {capacity}'
    assert value == total, \\
        f'reported total {total} but the indices are worth {value}'
    expected = best_by_search([tuple(pair) for pair in items], capacity)
    assert total == expected, \\
        f'best_selection({items}, {capacity}) got {total}, {expected} is reachable'

check([], 10)
check([(5, 10)], 4)
check([(5, 10)], 5)
check([(0, 0)], 0)
# A weightless item with value must always be taken, even at capacity zero.
check([(0, 7), (3, 1)], 0)
# Taking the best value-per-weight first strands 20 units of capacity here.
check([(10, 60), (20, 100), (30, 120)], 50)
check([(1, 1), (2, 6), (3, 10), (5, 16)], 7)
check([(4, 4), (4, 4), (4, 4), (4, 4)], 9)
check([(2, 3), (3, 4), (4, 5), (5, 6)], 5)
check([(7, 9), (8, 11), (9, 13), (10, 15)], 6)
check([(1, 0), (2, 0), (3, 0)], 6)

import random
rng = random.Random(9092026)
for _ in range(10):
    items = [(rng.randint(0, 12), rng.randint(0, 20))
             for _ in range(rng.randint(0, 14))]
    check(items, rng.randint(0, 30))

for items, capacity in (([(1, 1)], -1), ([(-1, 1)], 5), ([(1, -1)], 5),
                        ([(1, 1), (2, -3)], 4)):
    try:
        best_selection([tuple(pair) for pair in items], capacity)
    except ValueError:
        continue
    raise AssertionError(f'({items}, {capacity}) was not rejected')
""",
    ),
]
