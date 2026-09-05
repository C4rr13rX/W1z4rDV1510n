"""Reference solutions used to prove obstacle validators are not vacuous.

A validator nobody has ever seen pass is worse than no validator: it reports
failure forever and sends repair effort at curriculum that was never the
problem. This repository has already paid for that class of mistake once, when
a watchdog counted a filename nothing writes and read zero unconditionally.

So every authored task carries a reference implementation here, and
`tests/test_programming_obstacle_course.py` asserts both directions:

- the reference **passes**, so the validator is satisfiable at all; and
- a mutilated variant **fails**, so the validator is actually executing the
  behaviour contract rather than accepting anything that parses.

These are harness fixtures, not curriculum. Nothing in this file is ever
admitted as a training row, and the obstacle prompts they answer stay
held-out.
"""

from __future__ import annotations

REFERENCES: dict[str, str] = {}


REFERENCES["algorithms_data_structures-0001"] = r'''
class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._store = {}

    def get(self, key):
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def put(self, key, value):
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.capacity:
            oldest = next(iter(self._store))
            self._store.pop(oldest)
        self._store[key] = value
'''


REFERENCES["algorithms_data_structures-0002"] = r'''
def topological_order(nodes, edges):
    successors = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for before, after in edges:
        successors.setdefault(before, [])
        successors.setdefault(after, [])
        indegree.setdefault(before, 0)
        indegree.setdefault(after, 0)
        successors[before].append(after)
        indegree[after] += 1

    ready = [node for node in successors if indegree[node] == 0]
    order = []
    while ready:
        node = ready.pop()
        order.append(node)
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(successors):
        raise ValueError("constraints contain a cycle")
    return order
'''


REFERENCES["algorithms_data_structures-0003"] = r'''
def merge_intervals(intervals):
    cleaned = []
    for item in intervals:
        start, end = item[0], item[1]
        if start > end:
            raise ValueError("interval start exceeds end")
        if start < end:
            cleaned.append((start, end))
    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = [merged[-1][0], max(merged[-1][1], end)]
        else:
            merged.append([start, end])
    return merged
'''


REFERENCES["algorithms_data_structures-0004"] = r'''
import heapq


class RunningMedian:
    def __init__(self):
        self._low = []
        self._high = []

    def add(self, value):
        heapq.heappush(self._low, -value)
        heapq.heappush(self._high, -heapq.heappop(self._low))
        if len(self._high) > len(self._low):
            heapq.heappush(self._low, -heapq.heappop(self._high))

    def median(self):
        if not self._low:
            raise ValueError("no values have been added")
        if len(self._low) > len(self._high):
            return -self._low[0]
        return (-self._low[0] + self._high[0]) / 2
'''


REFERENCES["algorithms_data_structures-0005"] = r'''
class DisjointSet:
    def __init__(self):
        self._parent = {}
        self._rank = {}
        self._groups = 0

    def _ensure(self, item):
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0
            self._groups += 1

    def find(self, item):
        self._ensure(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a, b):
        left, right = self.find(a), self.find(b)
        if left == right:
            return
        if self._rank[left] < self._rank[right]:
            left, right = right, left
        self._parent[right] = left
        if self._rank[left] == self._rank[right]:
            self._rank[left] += 1
        self._groups -= 1

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def group_count(self):
        return self._groups
'''


REFERENCES["algorithms_data_structures-0006"] = r'''
def search_rotated(values, target):
    low, high = 0, len(values) - 1
    while low <= high:
        mid = (low + high) // 2
        middle = values[mid]
        if middle == target:
            return mid
        left = values[low]
        if left <= middle:
            if left <= target < middle:
                high = mid - 1
            else:
                low = mid + 1
        else:
            right = values[high]
            if middle < target <= right:
                low = mid + 1
            else:
                high = mid - 1
    return -1
'''


REFERENCES["algorithms_data_structures-0007"] = r'''
import collections


def window_maxima(values, width):
    if width < 1:
        raise ValueError("width must be at least one")
    count = len(values)
    if width > count:
        return []
    window = collections.deque()
    result = []
    for index in range(count):
        while window and values[window[-1]] <= values[index]:
            window.pop()
        window.append(index)
        if window[0] <= index - width:
            window.popleft()
        if index >= width - 1:
            result.append(values[window[0]])
    return result
'''


REFERENCES["algorithms_data_structures-0008"] = r'''
class _Node:
    __slots__ = ("children", "count", "terminal")

    def __init__(self):
        self.children = {}
        self.count = 0
        self.terminal = False


class PrefixIndex:
    def __init__(self):
        self._root = _Node()

    def insert(self, word):
        if self.contains(word):
            return
        node = self._root
        node.count += 1
        for character in word:
            node = node.children.setdefault(character, _Node())
            node.count += 1
        node.terminal = True

    def contains(self, word):
        node = self._find(word)
        return bool(node and node.terminal)

    def count_with_prefix(self, prefix):
        node = self._find(prefix)
        return node.count if node else 0

    def remove(self, word):
        if not self.contains(word):
            return False
        node = self._root
        node.count -= 1
        for character in word:
            child = node.children[character]
            child.count -= 1
            if child.count == 0:
                del node.children[character]
                return True
            node = child
        node.terminal = False
        return True

    def _find(self, text):
        node = self._root
        for character in text:
            node = node.children.get(character)
            if node is None:
                return None
        return node
'''


REFERENCES["validation_parsing_serialization-0001"] = r'''
def parse_csv(text):
    rows = []
    row = []
    field = []
    quoted = False
    index = 0
    length = len(text)
    started = False
    while index < length:
        character = text[index]
        if quoted:
            if character == '"':
                if index + 1 < length and text[index + 1] == '"':
                    field.append('"')
                    index += 2
                    continue
                quoted = False
                index += 1
                continue
            field.append(character)
            index += 1
            continue
        if character == '"' and not field:
            quoted = True
            started = True
            index += 1
            continue
        if character == ',':
            row.append("".join(field))
            field = []
            started = True
            index += 1
            continue
        if character in "\r\n":
            if character == '\r' and index + 1 < length and text[index + 1] == '\n':
                index += 1
            row.append("".join(field))
            rows.append(row)
            row = []
            field = []
            started = False
            index += 1
            continue
        field.append(character)
        started = True
        index += 1
    if quoted:
        raise ValueError("unterminated quoted field")
    if field or row or started:
        row.append("".join(field))
        rows.append(row)
    return rows
'''


REFERENCES["validation_parsing_serialization-0002"] = r'''
import re

_CORE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_IDENTIFIER = re.compile(r"^[0-9A-Za-z-]+$")
_NUMERIC = re.compile(r"^(0|[1-9]\d*)$")


def _parse(version):
    if not isinstance(version, str) or not version:
        raise ValueError("not a semantic version")
    core = version.split("+", 1)[0]
    if "-" in core:
        core, _, prerelease = core.partition("-")
    else:
        prerelease = None
    match = _CORE.match(core)
    if not match:
        raise ValueError("not a semantic version")
    numbers = tuple(int(part) for part in match.groups())
    if prerelease is None:
        return numbers, None
    identifiers = prerelease.split(".")
    if not identifiers or any(not _IDENTIFIER.match(item) for item in identifiers):
        raise ValueError("malformed prerelease")
    for item in identifiers:
        if item.isdigit() and not _NUMERIC.match(item):
            raise ValueError("numeric identifier with leading zero")
    return numbers, identifiers


def _compare_identifiers(left, right):
    for a, b in zip(left, right):
        a_numeric, b_numeric = a.isdigit(), b.isdigit()
        if a_numeric and b_numeric:
            if int(a) != int(b):
                return -1 if int(a) < int(b) else 1
        elif a_numeric != b_numeric:
            return -1 if a_numeric else 1
        elif a != b:
            return -1 if a < b else 1
    if len(left) == len(right):
        return 0
    return -1 if len(left) < len(right) else 1


def compare_versions(left, right):
    left_core, left_pre = _parse(left)
    right_core, right_pre = _parse(right)
    if left_core != right_core:
        return -1 if left_core < right_core else 1
    if left_pre is None and right_pre is None:
        return 0
    if left_pre is None:
        return 1
    if right_pre is None:
        return -1
    return _compare_identifiers(left_pre, right_pre)
'''


REFERENCES["validation_parsing_serialization-0003"] = r'''
import re

_INDEX = re.compile(r"^(0|[1-9]\d*)$")


def resolve_pointer(document, pointer):
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("pointer must be empty or start with a slash")
    current = document
    for raw in pointer.split("/")[1:]:
        if "~" in raw:
            checked = raw.replace("~0", "\x00").replace("~1", "/")
            if "~" in checked:
                raise ValueError("invalid escape in reference token")
            token = checked.replace("\x00", "~")
        else:
            token = raw
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(current, list):
            if not _INDEX.match(token):
                raise ValueError("array index must be a decimal with no leading zeros")
            position = int(token)
            if position >= len(current):
                raise IndexError(position)
            current = current[position]
        else:
            raise ValueError("pointer descends into a scalar")
    return current
'''


REFERENCES["validation_parsing_serialization-0004"] = r'''
import re

_PATTERN = re.compile(
    r"^(?P<sign>-)?P"
    r"(?!$)"
    r"(?:(?P<years>\d+(?:\.\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T(?!$)"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$"
)

_WEIGHTS = {
    "years": 365 * 86400.0,
    "months": 30 * 86400.0,
    "weeks": 7 * 86400.0,
    "days": 86400.0,
    "hours": 3600.0,
    "minutes": 60.0,
    "seconds": 1.0,
}


def parse_duration(text):
    if not isinstance(text, str):
        raise ValueError("duration must be a string")
    match = _PATTERN.match(text)
    if not match:
        raise ValueError("not an ISO 8601 duration")
    parts = match.groupdict()
    if not any(parts[name] for name in _WEIGHTS):
        raise ValueError("duration has no components")
    if "T" in text and not any(
        parts[name] for name in ("hours", "minutes", "seconds")
    ):
        raise ValueError("time separator without a time component")
    total = 0.0
    for name, weight in _WEIGHTS.items():
        value = parts[name]
        if value is not None:
            total += float(value) * weight
    return -total if parts["sign"] else total
'''


REFERENCES["validation_parsing_serialization-0005"] = r'''
import math

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _string(value):
    out = ['"']
    for character in value:
        escape = _ESCAPES.get(character)
        if escape is not None:
            out.append(escape)
        elif character < " ":
            out.append("\\u%04x" % ord(character))
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def _number(value):
    if isinstance(value, bool):
        raise TypeError("bool is not a number here")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("NaN and infinity are not JSON")
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    return repr(value)


def _encode(value, seen):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (int, float)):
        return _number(value)
    marker = id(value)
    if marker in seen:
        raise ValueError("structure contains itself")
    seen = seen | {marker}
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda pair: pair[0])
        return "{" + ",".join(
            _string(key) + ":" + _encode(item, seen) for key, item in items
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item, seen) for item in value) + "]"
    raise ValueError("unserializable value")


def canonical_json(value):
    return _encode(value, frozenset()).encode("utf-8")
'''


REFERENCES["validation_parsing_serialization-0006"] = r'''
import re

_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def parse_content_type(header):
    if not isinstance(header, str):
        raise ValueError("header must be a string")
    text = header.strip()
    media, _, remainder = text.partition(";")
    media = media.strip()
    if media.count("/") != 1:
        raise ValueError("malformed media type")
    kind, _, subtype = media.partition("/")
    if not _TOKEN.match(kind) or not _TOKEN.match(subtype):
        raise ValueError("malformed media type")

    params = {}
    index = 0
    remainder = remainder
    length = len(remainder)
    while index < length:
        while index < length and remainder[index] in " \t":
            index += 1
        if index >= length:
            break
        start = index
        while index < length and remainder[index] not in "=;":
            index += 1
        name = remainder[start:index].strip().lower()
        if index >= length or remainder[index] != "=" or not name:
            raise ValueError("parameter without a value")
        if not _TOKEN.match(name):
            raise ValueError("malformed parameter name")
        index += 1
        while index < length and remainder[index] in " \t":
            index += 1
        if index < length and remainder[index] == '"':
            index += 1
            chunks = []
            closed = False
            while index < length:
                character = remainder[index]
                if character == "\\" and index + 1 < length:
                    chunks.append(remainder[index + 1])
                    index += 2
                    continue
                if character == '"':
                    closed = True
                    index += 1
                    break
                chunks.append(character)
                index += 1
            if not closed:
                raise ValueError("unterminated quoted value")
            value = "".join(chunks)
            while index < length and remainder[index] in " \t":
                index += 1
            if index < length and remainder[index] != ";":
                raise ValueError("trailing data after quoted value")
            index += 1
        else:
            start = index
            while index < length and remainder[index] != ";":
                index += 1
            value = remainder[start:index].strip()
            index += 1
        params.setdefault(name, value)
    return media.lower(), params
'''


#: Minimal edits that must turn a passing candidate into a failing one. Each
#: is a (find, replace) pair applied to the reference source; the mutation is
#: chosen to break the behaviour contract rather than the syntax, so a
#: validator that only parses the candidate will not notice it.
MUTATIONS: dict[str, tuple[str, str]] = {
    "algorithms_data_structures-0001": (
        "value = self._store.pop(key)\n        self._store[key] = value\n        return value",
        "return self._store[key]",
    ),
    "algorithms_data_structures-0002": (
        'raise ValueError("constraints contain a cycle")',
        "return order",
    ),
    "algorithms_data_structures-0003": (
        "if merged and start <= merged[-1][1]:",
        "if merged and start < merged[-1][1]:",
    ),
    "algorithms_data_structures-0004": (
        "return (-self._low[0] + self._high[0]) / 2",
        "return -self._low[0]",
    ),
    "algorithms_data_structures-0005": (
        "self._groups -= 1",
        "pass",
    ),
    "algorithms_data_structures-0006": (
        "mid = (low + high) // 2",
        "mid = low",
    ),
    # Keeping an index one step past the window's left edge: the maxima stay
    # plausible but a value that has left the window can still win.
    "algorithms_data_structures-0007": (
        "if window[0] <= index - width:",
        "if window[0] < index - width:",
    ),
    "algorithms_data_structures-0008": (
        "if self.contains(word):\n            return",
        "if False:\n            return",
    ),
    "validation_parsing_serialization-0001": (
        "if index + 1 < length and text[index + 1] == '\"':",
        "if False:",
    ),
    "validation_parsing_serialization-0002": (
        "if a_numeric and b_numeric:",
        "if False:",
    ),
    # The classic RFC 6901 defect: expanding ~0 to a tilde before looking for
    # ~1 turns the token "~01" into "~1" and then into "/".
    "validation_parsing_serialization-0003": (
        'checked = raw.replace("~0", "\\x00").replace("~1", "/")',
        'checked = raw.replace("~0", "~").replace("~1", "/")',
    ),
    "validation_parsing_serialization-0004": (
        '"months": 30 * 86400.0,',
        '"months": 31 * 86400.0,',
    ),
    "validation_parsing_serialization-0005": (
        "items = sorted(value.items(), key=lambda pair: pair[0])",
        "items = list(value.items())",
    ),
    "validation_parsing_serialization-0006": (
        'if character == "\\\\" and index + 1 < length:',
        "if False:",
    ),
}
