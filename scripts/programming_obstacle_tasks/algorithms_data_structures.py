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
]
