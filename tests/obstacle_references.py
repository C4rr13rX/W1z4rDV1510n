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


class _WriteOnce(dict):
    """A dict that refuses to overwrite a key it already holds.

    Both tables below are populated by hundreds of separate
    ``TABLE["task-id"] = ...`` statements, and a plain dict resolves a
    duplicate key silently in favour of whichever assignment appears LATER in
    the file. That is not a stylistic risk, it is a scoring one: the pair of
    tests that keep a validator honest -- the reference must pass, the mutated
    reference must fail -- would then run against a reference solving a
    different problem than the task asks for.

    It has already happened. On 2026-09-05 two sessions authored the
    `scientific_3d_geometry_robotics` family concurrently and both reference
    blocks landed in this file; the second shadowed the first for all eight
    tasks. Both blocks parsed, so nothing complained, and the collision was
    caught by reading the diff rather than by running anything.

    Failing on the second assignment turns that silent shadowing into an
    ImportError naming the duplicated task, at the moment the file is
    imported and before any score exists to be wrong.
    """

    def __setitem__(self, key: str, value: str) -> None:
        if key in self:
            raise KeyError(
                f"{key!r} already has an entry in this table. Two blocks "
                f"define it, and the later one would silently shadow the "
                f"earlier, scoring that task against the wrong solution. "
                f"Keep exactly one and delete the other."
            )
        super().__setitem__(key, value)


REFERENCES: dict[str, str] = _WriteOnce()


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
REFERENCES["requirements_api_contracts-0001"] = r'''
def _parse_ranges(header):
    ranges = []
    for part in header.split(","):
        part = part.strip()
        if not part:
            continue
        bits = [bit.strip() for bit in part.split(";")]
        media = bits[0].lower()
        if "/" not in media:
            continue
        quality = 1.0
        for param in bits[1:]:
            name, _, value = param.partition("=")
            if name.strip().lower() == "q":
                try:
                    quality = float(value.strip())
                except ValueError:
                    quality = 1.0
        ranges.append((media, quality))
    return ranges


def select_media_type(header, available):
    if not header or not header.strip():
        return available[0] if available else None
    ranges = _parse_ranges(header)
    if not ranges:
        return available[0] if available else None

    best = None
    for index, candidate in enumerate(available):
        lowered = candidate.lower()
        ctype, _, _csub = lowered.partition("/")
        chosen = None
        for media, quality in ranges:
            mtype, _, msub = media.partition("/")
            if mtype == "*" and msub == "*":
                specificity = 1
            elif msub == "*" and mtype == ctype:
                specificity = 2
            elif media == lowered:
                specificity = 3
            else:
                continue
            if chosen is None or specificity > chosen[0]:
                chosen = (specificity, quality)
        if chosen is None or chosen[1] <= 0:
            continue
        key = (chosen[1], -index)
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1] if best else None
'''


REFERENCES["requirements_api_contracts-0002"] = r'''
import copy


def merge_patch(target, patch):
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict):
            result[key] = merge_patch(result.get(key), value)
        else:
            result[key] = copy.deepcopy(value)
    return result
'''


REFERENCES["requirements_api_contracts-0003"] = r'''
def _tags(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def _weak(tag):
    return tag[2:] if tag.startswith('W/') else tag


def evaluate_preconditions(method, headers, current_etag, exists):
    lowered = {str(name).lower(): value for name, value in headers.items()}

    if_match = lowered.get("if-match")
    if if_match is not None:
        value = if_match.strip()
        if value == "*":
            if not exists:
                return 412
        elif current_etag is None or current_etag.startswith("W/"):
            return 412
        elif not any(tag == current_etag for tag in _tags(value)):
            return 412

    if_none_match = lowered.get("if-none-match")
    if if_none_match is not None:
        value = if_none_match.strip()
        if value == "*":
            matched = bool(exists)
        else:
            matched = current_etag is not None and any(
                _weak(tag) == _weak(current_etag) for tag in _tags(value)
            )
        if matched:
            return 304 if method in ("GET", "HEAD") else 412

    return 200
'''


REFERENCES["requirements_api_contracts-0004"] = r'''
def parse_byte_range(header, length):
    if not header or "=" not in header:
        return None
    unit, _, spec = header.partition("=")
    if unit.strip().lower() != "bytes" or not spec.strip():
        return None

    satisfiable = []
    for part in spec.split(","):
        part = part.strip()
        if not part or "-" not in part:
            return None
        first_text, _, last_text = part.partition("-")
        first_text, last_text = first_text.strip(), last_text.strip()
        if not first_text and not last_text:
            return None

        if not first_text:
            if not last_text.isdigit():
                return None
            suffix = int(last_text)
            if suffix == 0:
                continue
            satisfiable.append((max(0, length - suffix), length - 1))
            continue

        if not first_text.isdigit():
            return None
        start = int(first_text)
        if not last_text:
            end = length - 1
        else:
            if not last_text.isdigit():
                return None
            end = int(last_text)
            if end < start:
                return None
            end = min(end, length - 1)
        if start > length - 1:
            continue
        satisfiable.append((start, end))

    if not satisfiable:
        raise ValueError("no satisfiable range")
    return satisfiable
'''


REFERENCES["requirements_api_contracts-0005"] = r'''
class CursorPage:
    def __init__(self, rows):
        self._rows = {int(row["id"]): dict(row) for row in rows}

    def insert(self, row):
        self._rows[int(row["id"])] = dict(row)

    def delete(self, row_id):
        self._rows.pop(int(row_id), None)

    def page(self, cursor, limit):
        after = None if cursor is None else int(cursor)
        selected = [
            key for key in sorted(self._rows)
            if after is None or key > after
        ]
        items = [dict(self._rows[key]) for key in selected[:limit]]
        if len(selected) <= limit:
            return items, None
        return items, str(items[-1]["id"])
'''


REFERENCES["requirements_api_contracts-0006"] = r'''
def check_compatibility(old, new):
    findings = []
    for name, before in old.items():
        if name not in new:
            findings.append(f"field {name} was removed")
            continue
        after = new[name]
        if before.get("type") != after.get("type"):
            findings.append(
                f"field {name} changed type from {before.get('type')} "
                f"to {after.get('type')}"
            )
        if not before.get("required", False) and after.get("required", False):
            findings.append(f"field {name} became required")
        old_enum = before.get("enum")
        if old_enum is not None:
            new_enum = after.get("enum")
            for value in old_enum:
                if new_enum is None or value not in new_enum:
                    findings.append(
                        f"field {name} no longer permits {value!r}"
                    )
    for name, after in new.items():
        if name not in old and after.get("required", False):
            findings.append(f"field {name} was added as required")
    return sorted(findings)
'''


REFERENCES["requirements_api_contracts-0007"] = r'''
import json


class IdempotentEndpoint:
    def __init__(self):
        self._entries = {}

    @staticmethod
    def _fingerprint(body):
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    def submit(self, key, body, handler):
        fingerprint = self._fingerprint(body)
        if key in self._entries:
            stored, response = self._entries[key]
            if stored != fingerprint:
                return 422, None
            return 200, response
        response = handler()
        self._entries[key] = (fingerprint, response)
        return 201, response
'''


REFERENCES["requirements_api_contracts-0008"] = r'''
def parse_link_header(header):
    if not header or not header.strip():
        return []

    text = header
    length = len(text)
    index = 0
    links = []

    while index < length:
        while index < length and text[index] in ", \t":
            index += 1
        if index >= length:
            break
        if text[index] != "<":
            raise ValueError("link is missing its angle-bracketed URI")
        end = text.find(">", index)
        if end == -1:
            raise ValueError("unterminated URI reference")
        entry = {"uri": text[index + 1:end]}
        index = end + 1

        while index < length:
            while index < length and text[index] in " \t":
                index += 1
            if index >= length or text[index] == ",":
                break
            if text[index] != ";":
                raise ValueError("expected a parameter separator")
            index += 1
            while index < length and text[index] in " \t":
                index += 1
            name_start = index
            while index < length and text[index] not in "=;,":
                index += 1
            name = text[name_start:index].strip()

            value = ""
            if index < length and text[index] == "=":
                index += 1
                while index < length and text[index] in " \t":
                    index += 1
                if index < length and text[index] == '"':
                    index += 1
                    chunks = []
                    while index < length and text[index] != '"':
                        if text[index] == "\\" and index + 1 < length:
                            chunks.append(text[index + 1])
                            index += 2
                            continue
                        chunks.append(text[index])
                        index += 1
                    value = "".join(chunks)
                    index += 1
                else:
                    value_start = index
                    while index < length and text[index] not in ";,":
                        index += 1
                    value = text[value_start:index].strip()

            if name and name not in entry:
                entry[name] = value

        links.append(entry)

    return links
'''


REFERENCES["concurrency_async_distributed-0001"] = r'''
class TokenBucket:
    def __init__(self, capacity, refill_per_second, clock):
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._updated = clock()

    def _refill(self):
        now = self._clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(
                self.capacity, self._tokens + elapsed * self.refill_per_second
            )
            self._updated = now

    def allow(self, cost=1):
        if cost > self.capacity:
            return False
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False
'''


REFERENCES["concurrency_async_distributed-0002"] = r'''
class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold, recovery_seconds, clock):
        self.failure_threshold = int(failure_threshold)
        self.recovery_seconds = float(recovery_seconds)
        self._clock = clock
        self._failures = 0
        self._opened_at = None

    def call(self, operation):
        if self._opened_at is not None:
            if self._clock() - self._opened_at < self.recovery_seconds:
                raise CircuitOpenError("circuit is open")
        try:
            result = operation()
        except Exception:
            self._failures += 1
            if (self._opened_at is not None
                    or self._failures >= self.failure_threshold):
                self._opened_at = self._clock()
            raise
        self._failures = 0
        self._opened_at = None
        return result
'''


REFERENCES["concurrency_async_distributed-0003"] = r'''
def retry_with_backoff(operation, attempts, base_delay, max_delay, sleep,
                       should_retry):
    last = None
    for index in range(attempts):
        try:
            return operation()
        except Exception as error:
            if not should_retry(error):
                raise
            last = error
            if index == attempts - 1:
                raise
            sleep(min(base_delay * (2 ** index), max_delay))
    raise last
'''


REFERENCES["concurrency_async_distributed-0004"] = r'''
def compare_clocks(left, right):
    nodes = set(left) | set(right)
    less = any(left.get(node, 0) < right.get(node, 0) for node in nodes)
    greater = any(left.get(node, 0) > right.get(node, 0) for node in nodes)
    if less and greater:
        return "concurrent"
    if less:
        return "before"
    if greater:
        return "after"
    return "equal"


def merge_clocks(left, right):
    nodes = set(left) | set(right)
    return {
        node: max(left.get(node, 0), right.get(node, 0)) for node in nodes
    }
'''


REFERENCES["concurrency_async_distributed-0005"] = r'''
class ExactlyOnceInbox:
    def __init__(self, apply):
        self._apply = apply
        self._next = 1
        self._buffer = {}

    @property
    def pending(self):
        return len(self._buffer)

    def deliver(self, sequence, payload):
        sequence = int(sequence)
        if sequence < self._next or sequence in self._buffer:
            return []
        self._buffer[sequence] = payload
        released = []
        while self._next in self._buffer:
            released.append(self._next)
            self._apply(self._buffer.pop(self._next))
            self._next += 1
        return released
'''


REFERENCES["concurrency_async_distributed-0006"] = r'''
def schedule_batches(tasks, dependencies):
    names = list(tasks)
    known = set(names)
    successors = {name: set() for name in names}
    indegree = {name: 0 for name in names}
    for before, after in dependencies:
        if before not in known or after not in known:
            raise ValueError(f"unknown task in dependency {(before, after)}")
        if after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1

    remaining = dict(indegree)
    ready = sorted(name for name in names if remaining[name] == 0)
    batches = []
    scheduled = 0
    while ready:
        batches.append(list(ready))
        scheduled += len(ready)
        following = []
        for name in ready:
            for successor in sorted(successors[name]):
                remaining[successor] -= 1
                if remaining[successor] == 0:
                    following.append(successor)
        ready = sorted(following)
    if scheduled != len(names):
        raise ValueError("dependencies contain a cycle")
    return batches
'''


REFERENCES["concurrency_async_distributed-0007"] = r'''
import threading


class Once:
    def __init__(self):
        self._lock = threading.Lock()
        self._done = False
        self._value = None

    @property
    def done(self):
        return self._done

    def do(self, factory):
        if self._done:
            return self._value
        with self._lock:
            if self._done:
                return self._value
            value = factory()
            self._value = value
            self._done = True
            return value
'''


REFERENCES["concurrency_async_distributed-0008"] = r'''
import threading


class ReadWriteLock:
    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self):
        with self._condition:
            while self._writer or self._waiting_writers > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self):
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._condition.wait()
            finally:
                self._waiting_writers -= 1
            self._writer = True

    def release_write(self):
        with self._condition:
            self._writer = False
            self._condition.notify_all()
'''


REFERENCES["databases_migrations_transactions-0001"] = r'''
import hashlib


def _checksum(sql):
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def apply_migrations(connection, migrations):
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    applied = {
        int(version): checksum
        for version, checksum in connection.execute(
            "SELECT version, checksum FROM schema_migrations"
        )
    }
    highest = max(applied) if applied else None
    pending = []
    for version, name, sql in sorted(migrations, key=lambda item: int(item[0])):
        version = int(version)
        digest = _checksum(sql)
        if version in applied:
            if applied[version] != digest:
                raise ValueError(f"migration {version} changed after being applied")
            continue
        if highest is not None and version <= highest:
            raise ValueError(f"migration {version} is out of order")
        pending.append((version, name, sql, digest))
    done = []
    for version, name, sql, digest in pending:
        connection.execute("BEGIN")
        try:
            connection.execute(sql)
            connection.execute(
                "INSERT INTO schema_migrations (version, name, checksum) "
                "VALUES (?, ?, ?)",
                (version, name, digest),
            )
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        done.append(version)
    return done
'''


REFERENCES["databases_migrations_transactions-0002"] = r'''
ALLOWED = ("title", "body")


def update_document(connection, doc_id, expected_version, fields):
    if not fields:
        raise ValueError("fields must not be empty")
    for key in fields:
        if key not in ALLOWED:
            raise ValueError(f"unknown column {key!r}")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    parameters = list(fields.values()) + [doc_id, expected_version]
    cursor = connection.execute(
        f"UPDATE documents SET {assignments}, version = version + 1 "
        "WHERE id = ? AND version = ?",
        parameters,
    )
    if cursor.rowcount == 0:
        return None
    return int(expected_version) + 1
'''


REFERENCES["databases_migrations_transactions-0003"] = r'''
def record_order(connection, order, event):
    connection.execute("BEGIN")
    try:
        cursor = connection.execute(
            "INSERT INTO outbox (topic, payload) VALUES (?, ?)",
            (event["topic"], event["payload"]),
        )
        event_id = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO orders (id, customer, total_cents) VALUES (?, ?, ?)",
            (order["id"], order["customer"], order["total_cents"]),
        )
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return event_id
'''


REFERENCES["databases_migrations_transactions-0004"] = r'''
def page_after(connection, cursor, limit):
    if cursor is None:
        rows = connection.execute(
            "SELECT id, created_at, title FROM posts "
            "ORDER BY created_at, id LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        created_at, row_id = cursor
        rows = connection.execute(
            "SELECT id, created_at, title FROM posts "
            "WHERE created_at > ? OR (created_at = ? AND id > ?) "
            "ORDER BY created_at, id LIMIT ?",
            (created_at, created_at, row_id, limit),
        ).fetchall()
    rows = [tuple(row) for row in rows]
    if not rows:
        return [], None
    last = rows[-1]
    return rows, (last[1], last[0])
'''


REFERENCES["databases_migrations_transactions-0005"] = r'''
def apply_batch(connection, items):
    applied = []
    rejected = []
    connection.execute("BEGIN")
    try:
        for index, item in enumerate(items):
            connection.execute(f"SAVEPOINT item_{index}")
            try:
                connection.execute(
                    "INSERT INTO accounts (id, owner) VALUES (?, ?)",
                    (item["id"], item["owner"]),
                )
                connection.execute(
                    "INSERT INTO audit (account_id, amount) VALUES (?, ?)",
                    (item["id"], item["amount"]),
                )
            except Exception as exc:
                connection.execute(f"ROLLBACK TO item_{index}")
                connection.execute(f"RELEASE item_{index}")
                rejected.append((item["id"], type(exc).__name__))
            else:
                connection.execute(f"RELEASE item_{index}")
                applied.append(item["id"])
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return applied, rejected
'''


REFERENCES["databases_migrations_transactions-0006"] = r'''
COLUMNS = ("display_name", "email", "locale")


def merge_record(connection, record):
    if "id" not in record:
        raise ValueError("record must carry an id")
    for key in record:
        if key != "id" and key not in COLUMNS:
            raise ValueError(f"unknown column {key!r}")
    row_id = record["id"]
    supplied = {
        key: record[key]
        for key in COLUMNS
        if record.get(key) is not None
    }
    exists = connection.execute(
        "SELECT 1 FROM profiles WHERE id = ?", (row_id,)
    ).fetchone()
    if exists is None:
        columns = ["id"] + list(supplied)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO profiles ({', '.join(columns)}) VALUES ({placeholders})",
            [row_id] + list(supplied.values()),
        )
        return "inserted"
    if supplied:
        assignments = ", ".join(f"{key} = ?" for key in supplied)
        connection.execute(
            f"UPDATE profiles SET {assignments} WHERE id = ?",
            list(supplied.values()) + [row_id],
        )
    return "updated"
'''


REFERENCES["databases_migrations_transactions-0007"] = r'''
import sqlite3


def delete_customer(connection, customer_id):
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("BEGIN")
    try:
        cursor = connection.execute(
            "DELETE FROM customers WHERE id = ?", (customer_id,)
        )
        deleted = cursor.rowcount
    except sqlite3.IntegrityError:
        connection.execute("ROLLBACK")
        return False
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return deleted > 0
'''


REFERENCES["databases_migrations_transactions-0008"] = r'''
def process_payment(connection, idempotency_key, amount_cents):
    if isinstance(amount_cents, bool) or not isinstance(amount_cents, int):
        raise ValueError("amount_cents must be an integer")
    if amount_cents <= 0:
        raise ValueError("amount_cents must be positive")
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT charge_id, amount_cents FROM payments WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is not None:
            connection.execute("COMMIT")
            return {
                "charge_id": int(row[0]),
                "amount_cents": int(row[1]),
                "replayed": True,
            }
        cursor = connection.execute(
            "INSERT INTO payments (idempotency_key, amount_cents) VALUES (?, ?)",
            (idempotency_key, amount_cents),
        )
        charge_id = int(cursor.lastrowid)
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return {
        "charge_id": charge_id,
        "amount_cents": int(amount_cents),
        "replayed": False,
    }
'''


REFERENCES["http_apis_authn_appsec-0001"] = r'''
import base64
import hashlib
import hmac
import json


def _b64decode(segment):
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def verify_token(token, key, now):
    if not isinstance(token, str):
        raise ValueError("token must be a string")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed token")
    head, body, signature_segment = parts
    try:
        header = json.loads(_b64decode(head))
        payload = json.loads(_b64decode(body))
        signature = _b64decode(signature_segment)
    except Exception:
        raise ValueError("malformed token")
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise ValueError("unsupported algorithm")
    if isinstance(key, str):
        key = key.encode("utf-8")
    expected = hmac.new(
        key, f"{head}.{body}".encode("ascii"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, signature):
        raise ValueError("bad signature")
    if not isinstance(payload, dict):
        raise ValueError("malformed payload")
    exp = payload.get("exp")
    if not isinstance(exp, int) or isinstance(exp, bool):
        raise ValueError("missing or invalid exp")
    if now >= exp:
        raise ValueError("expired")
    return payload
'''


REFERENCES["http_apis_authn_appsec-0002"] = r'''
import posixpath


def resolve_within_root(root, user_path):
    if not isinstance(user_path, str) or not user_path:
        raise ValueError("user_path must be a non-empty string")
    if "\x00" in user_path:
        raise ValueError("NUL byte in path")
    if user_path.startswith("/"):
        raise ValueError("absolute paths are not allowed")
    base = posixpath.normpath(root)
    candidate = posixpath.normpath(posixpath.join(base, user_path))
    if candidate != base and not candidate.startswith(base.rstrip("/") + "/"):
        raise ValueError("path escapes the root")
    return candidate
'''


REFERENCES["http_apis_authn_appsec-0003"] = r'''
import base64
import hashlib
import hmac
import secrets

ITERATIONS = 200000


def hash_password(password):
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
    )


def verify_password(password, stored):
    if not isinstance(stored, str) or not isinstance(password, str):
        return False
    parts = stored.split("$")
    if len(parts) != 4:
        return False
    scheme, iterations, salt_b64, key_b64 = parts
    if scheme != "pbkdf2_sha256":
        return False
    try:
        rounds = int(iterations)
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(key_b64, validate=True)
    except Exception:
        return False
    if rounds < 1 or not salt or not expected:
        return False
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, rounds, dklen=len(expected)
    )
    return hmac.compare_digest(derived, expected)
'''


REFERENCES["http_apis_authn_appsec-0004"] = r'''
ACTIONS = ("read", "write", "delete")


def authorize(principal, action, resource):
    if not isinstance(principal, dict) or action not in ACTIONS:
        return False
    roles = principal.get("roles")
    if not isinstance(roles, list):
        return False
    if "admin" in roles:
        return True
    if "auditor" in roles:
        return action == "read"
    if action == "read" and resource.get("visibility") == "public":
        return True
    owner = resource.get("owner_id")
    identity = principal.get("id")
    if owner is None or identity is None:
        return False
    return type(owner) is type(identity) and owner == identity
'''


REFERENCES["http_apis_authn_appsec-0005"] = r'''
import base64
import hashlib
import hmac
import secrets


def _b64(raw):
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(session_id, nonce, key):
    if isinstance(key, str):
        key = key.encode("utf-8")
    return _b64(hmac.new(
        key, f"{session_id}.{nonce}".encode("ascii"), hashlib.sha256
    ).digest())


def issue_csrf(session_id, key):
    nonce = _b64(secrets.token_bytes(16))
    return f"{nonce}.{_sign(session_id, nonce, key)}"


def check_csrf(session_id, token, key):
    if not isinstance(token, str) or not isinstance(session_id, str):
        return False
    parts = token.split(".")
    if len(parts) != 2:
        return False
    nonce, signature = parts
    if not nonce or not signature:
        return False
    try:
        expected = _sign(session_id, nonce, key)
    except Exception:
        return False
    return hmac.compare_digest(expected, signature)
'''


REFERENCES["http_apis_authn_appsec-0006"] = r'''
FORBIDDEN = set(';,"\\ ') | {chr(code) for code in range(0x21)} | {chr(0x7F)}
SAME_SITE = ("Strict", "Lax", "None")


def _check(text, label, extra=""):
    if not isinstance(text, str) or not text:
        raise ValueError(f"{label} must be a non-empty string")
    for character in text:
        if character in FORBIDDEN or character in extra:
            raise ValueError(f"illegal character in {label}")


def build_set_cookie(name, value, *, max_age=None, path="/", secure=True,
                     http_only=True, same_site="Lax"):
    _check(name, "name", extra="=")
    _check(value, "value")
    if same_site not in SAME_SITE:
        raise ValueError("same_site must be Strict, Lax or None")
    if same_site == "None" and not secure:
        raise ValueError("SameSite=None requires Secure")
    if max_age is not None:
        if isinstance(max_age, bool) or not isinstance(max_age, int):
            raise ValueError("max_age must be an integer")
    parts = [f"{name}={value}"]
    if max_age is not None:
        parts.append(f"Max-Age={max_age}")
    if path:
        parts.append(f"Path={path}")
    if secure:
        parts.append("Secure")
    if http_only:
        parts.append("HttpOnly")
    parts.append(f"SameSite={same_site}")
    return "; ".join(parts)
'''


REFERENCES["http_apis_authn_appsec-0007"] = r'''
import urllib.parse


def safe_redirect(target, allowed_hosts):
    if not isinstance(target, str) or not target:
        return "/"
    if any(character in target for character in "\r\n\x00"):
        return "/"
    normalised = target.replace("\\", "/")
    if normalised.startswith("//"):
        return "/"
    if normalised.startswith("/"):
        return target
    try:
        parsed = urllib.parse.urlparse(target)
    except ValueError:
        return "/"
    if parsed.scheme not in ("http", "https"):
        return "/"
    if parsed.username is not None or parsed.password is not None:
        return "/"
    if "@" in parsed.netloc:
        return "/"
    hostname = parsed.hostname
    if not hostname or hostname.lower() not in allowed_hosts:
        return "/"
    return target
'''


REFERENCES["http_apis_authn_appsec-0008"] = r'''
REDACTED = "[REDACTED]"


def _is_secret(key, secret_keys):
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return any(name in lowered for name in secret_keys)


def redact(record, secret_keys):
    if isinstance(record, dict):
        result = {}
        for key, value in record.items():
            if _is_secret(key, secret_keys):
                result[key] = REDACTED
            else:
                result[key] = redact(value, secret_keys)
        return result
    if isinstance(record, tuple):
        return tuple(redact(item, secret_keys) for item in record)
    if isinstance(record, list):
        return [redact(item, secret_keys) for item in record]
    return record
'''


REFERENCES["scientific_3d_geometry_robotics-0001"] = r'''
def compose(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def apply(t, point):
    x, y, z = point
    return tuple(
        t[i][0] * x + t[i][1] * y + t[i][2] * z + t[i][3] for i in range(3)
    )


def invert(t):
    rows = []
    for i in range(3):
        rotation = tuple(t[j][i] for j in range(3))
        offset = -sum(t[j][i] * t[j][3] for j in range(3))
        rows.append(rotation + (offset,))
    rows.append((0.0, 0.0, 0.0, 1.0))
    return tuple(rows)
'''


REFERENCES["scientific_3d_geometry_robotics-0002"] = r'''
import math


def normalize(q):
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("the zero quaternion has no direction")
    return (w / norm, x / norm, y / norm, z / norm)


def multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def to_matrix(q):
    w, x, y, z = normalize(q)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def rotate(q, v):
    m = to_matrix(q)
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))
'''


REFERENCES["scientific_3d_geometry_robotics-0003"] = r'''
def mesh_report(vertices, triangles):
    directed = {}
    undirected = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1
            key = (u, v) if u < v else (v, u)
            undirected[key] = undirected.get(key, 0) + 1
    boundary = sum(1 for count in undirected.values() if count == 1)
    nonmanifold = sum(1 for count in undirected.values() if count > 2)
    return {
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "closed": boundary == 0 and nonmanifold == 0,
        "consistent_winding": all(
            count == 1 for count in directed.values()
        ),
        "euler_characteristic":
            len(vertices) - len(undirected) + len(triangles),
    }
'''


REFERENCES["scientific_3d_geometry_robotics-0004"] = r'''
def _determinant(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def signed_volume(vertices, triangles):
    total = 0.0
    for ia, ib, ic in triangles:
        total += _determinant(vertices[ia], vertices[ib], vertices[ic])
    return total / 6.0


def centroid(vertices, triangles):
    volume = signed_volume(vertices, triangles)
    if volume == 0.0:
        raise ValueError("a mesh enclosing no volume has no centroid")
    moment = [0.0, 0.0, 0.0]
    for ia, ib, ic in triangles:
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        det = _determinant(a, b, c)
        for axis in range(3):
            moment[axis] += det * (a[axis] + b[axis] + c[axis]) / 24.0
    return tuple(value / volume for value in moment)
'''


REFERENCES["scientific_3d_geometry_robotics-0005"] = r'''
def box_inertia(mass, sx, sy, sz):
    if mass <= 0.0 or min(sx, sy, sz) <= 0.0:
        raise ValueError("mass and extents must be positive")
    factor = mass / 12.0
    return (
        (factor * (sy * sy + sz * sz), 0.0, 0.0),
        (0.0, factor * (sx * sx + sz * sz), 0.0),
        (0.0, 0.0, factor * (sx * sx + sy * sy)),
    )


def translate_inertia(inertia, mass, offset):
    squared = sum(component * component for component in offset)
    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            delta = squared if i == j else 0.0
            row.append(
                inertia[i][j] + mass * (delta - offset[i] * offset[j])
            )
        rows.append(tuple(row))
    return tuple(rows)
'''


REFERENCES["scientific_3d_geometry_robotics-0006"] = r'''
def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def intersect(origin, direction, triangle):
    epsilon = 1e-12
    a, b, c = triangle
    edge1 = tuple(b[i] - a[i] for i in range(3))
    edge2 = tuple(c[i] - a[i] for i in range(3))
    pvec = _cross(direction, edge2)
    det = _dot(edge1, pvec)
    if abs(det) < epsilon:
        return None
    inv_det = 1.0 / det
    tvec = tuple(origin[i] - a[i] for i in range(3))
    u = _dot(tvec, pvec) * inv_det
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    qvec = _cross(tvec, edge1)
    v = _dot(direction, qvec) * inv_det
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return None
    t = _dot(edge2, qvec) * inv_det
    if t <= 0.0:
        return None
    return t
'''


REFERENCES["scientific_3d_geometry_robotics-0007"] = r'''
import struct


def _normal(triangle):
    a, b, c = triangle
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
    length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in n)


def write_binary_stl(triangles, header=b""):
    triangles = list(triangles)
    out = bytearray(header[:80].ljust(80, b"\x00"))
    out += struct.pack("<I", len(triangles))
    for triangle in triangles:
        out += struct.pack("<3f", *_normal(triangle))
        for vertex in triangle:
            out += struct.pack("<3f", *vertex)
        out += struct.pack("<H", 0)
    return bytes(out)


def read_binary_stl(data):
    if len(data) < 84:
        raise ValueError("buffer is shorter than a binary STL header")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError("buffer length disagrees with its triangle count")
    triangles = []
    for index in range(count):
        values = struct.unpack_from("<12f", data, 84 + 50 * index)
        triangles.append(tuple(
            tuple(values[3 + 3 * corner + axis] for axis in range(3))
            for corner in range(3)
        ))
    return triangles
'''


REFERENCES["scientific_3d_geometry_robotics-0008"] = r'''
import math


def _link_matrix(a, alpha, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return (
        (ct, -st * ca, st * sa, a * ct),
        (st, ct * ca, -ct * sa, a * st),
        (0.0, sa, ca, d),
        (0.0, 0.0, 0.0, 1.0),
    )


def _multiply(m, n):
    return tuple(
        tuple(sum(m[i][k] * n[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def forward_kinematics(links, joint_values):
    links = list(links)
    values = list(joint_values)
    if len(links) != len(values):
        raise ValueError("one joint value is required per link")
    pose = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(4))
        for row in range(4)
    )
    for (a, alpha, d, offset), value in zip(links, values):
        pose = _multiply(pose, _link_matrix(a, alpha, d, offset + value))
    return pose
'''


REFERENCES["reliability_observability_performance-0001"] = r'''
import math


class LatencyHistogram:
    def __init__(self, relative_error=0.01):
        if not 0 < relative_error < 1:
            raise ValueError("relative_error must be in (0, 1)")
        self.relative_error = relative_error
        self._gamma = (1 + relative_error) / (1 - relative_error)
        self._log_gamma = math.log(self._gamma)
        self._buckets = {}
        self._count = 0

    def record(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be a number")
        if value <= 0:
            raise ValueError("value must be positive")
        index = math.ceil(math.log(value) / self._log_gamma)
        self._buckets[index] = self._buckets.get(index, 0) + 1
        self._count += 1

    def count(self):
        return self._count

    def bucket_count(self):
        return len(self._buckets)

    def quantile(self, q):
        if isinstance(q, bool) or not isinstance(q, (int, float)):
            raise ValueError("q must be a number")
        if q < 0 or q > 1:
            raise ValueError("q must be in [0, 1]")
        if self._count == 0:
            raise ValueError("the histogram is empty")
        target = max(1, math.ceil(q * self._count))
        seen = 0
        for index in sorted(self._buckets):
            seen += self._buckets[index]
            if seen >= target:
                return 2 * self._gamma ** index / (self._gamma + 1)
        raise ValueError("the histogram is empty")
'''


REFERENCES["reliability_observability_performance-0002"] = r'''
import hashlib


def should_sample(trace_id, rate):
    if not isinstance(trace_id, str):
        raise ValueError("trace_id must be a string")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError("rate must be a number")
    if rate < 0 or rate > 1:
        raise ValueError("rate must be in [0, 1]")
    digest = hashlib.sha256(trace_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value < rate * 2 ** 64
'''


REFERENCES["reliability_observability_performance-0003"] = r'''
import math


class SlidingWindowCounter:
    def __init__(self, window_seconds, bucket_count):
        if isinstance(window_seconds, bool) or not isinstance(
                window_seconds, (int, float)) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if isinstance(bucket_count, bool) or not isinstance(
                bucket_count, int) or bucket_count < 1:
            raise ValueError("bucket_count must be at least 1")
        self.window_seconds = float(window_seconds)
        self.bucket_count = bucket_count
        self._width = self.window_seconds / bucket_count
        self._buckets = {}
        self._newest = None

    def _index(self, now):
        return math.floor(now / self._width)

    def _evict(self, reference):
        oldest = self._index(reference) - self.bucket_count
        for index in [key for key in self._buckets if key <= oldest]:
            del self._buckets[index]

    def record(self, now, amount=1):
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError("amount must be a number")
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._newest = now if self._newest is None else max(self._newest, now)
        self._evict(self._newest)
        index = self._index(now)
        if index <= self._index(self._newest) - self.bucket_count:
            return False
        self._buckets[index] = self._buckets.get(index, 0) + amount
        return True

    def total(self, now):
        self._newest = now if self._newest is None else max(self._newest, now)
        self._evict(self._newest)
        return sum(self._buckets.values())

    def resident_buckets(self):
        return len(self._buckets)
'''


REFERENCES["reliability_observability_performance-0004"] = r'''
_VALID_STATUSES = ("up", "degraded", "down")


def aggregate_health(checks):
    if not isinstance(checks, (list, tuple)) or not checks:
        raise ValueError("at least one check is required")
    seen = set()
    down = []
    degraded = []
    critical_down = False
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("each check must be a mapping")
        for key in ("name", "status", "critical"):
            if key not in check:
                raise ValueError("a check is missing " + key)
        name = check["name"]
        status = check["status"]
        critical = check["critical"]
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if status not in _VALID_STATUSES:
            raise ValueError("unknown status " + repr(status))
        if not isinstance(critical, bool):
            raise ValueError("critical must be a bool")
        if name in seen:
            raise ValueError("duplicate name " + repr(name))
        seen.add(name)
        if status == "down":
            down.append(name)
            critical_down = critical_down or critical
        elif status == "degraded":
            degraded.append(name)
    if critical_down:
        status = "unhealthy"
    elif down or degraded:
        status = "degraded"
    else:
        status = "healthy"
    return {"status": status, "down": sorted(down),
            "degraded": sorted(degraded)}
'''


REFERENCES["reliability_observability_performance-0005"] = r'''
_REQUIRED_WINDOWS = ("5m", "1h", "30m", "6h")


def evaluate_error_budget(objective, windows):
    if isinstance(objective, bool) or not isinstance(objective, (int, float)):
        raise ValueError("objective must be a number")
    if not 0 < objective < 1:
        raise ValueError("objective must be in (0, 1)")
    if not isinstance(windows, dict):
        raise ValueError("windows must be a mapping")
    budget = 1.0 - objective
    burn_rates = {}
    for name in _REQUIRED_WINDOWS:
        if name not in windows:
            raise ValueError("missing window " + repr(name))
        entry = windows[name]
        if not isinstance(entry, dict):
            raise ValueError("window " + repr(name) + " must be a mapping")
        if "total" not in entry or "failed" not in entry:
            raise ValueError("window " + repr(name) + " needs total/failed")
        total = entry["total"]
        failed = entry["failed"]
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError("window " + repr(name) + " has a bad total")
        if isinstance(failed, bool) or not isinstance(failed, int):
            raise ValueError("window " + repr(name) + " has a bad failed")
        if failed < 0 or failed > total:
            raise ValueError("window " + repr(name) + " has a bad failed")
        burn_rates[name] = (failed / total) / budget
    if burn_rates["1h"] >= 14.4 and burn_rates["5m"] >= 14.4:
        severity = "page"
    elif burn_rates["6h"] >= 6 and burn_rates["30m"] >= 6:
        severity = "ticket"
    else:
        severity = None
    return {"burn_rates": burn_rates, "severity": severity}
'''


REFERENCES["reliability_observability_performance-0006"] = r'''
import math


class EventThrottle:
    def __init__(self, window_seconds, max_per_window):
        if isinstance(window_seconds, bool) or not isinstance(
                window_seconds, (int, float)) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if isinstance(max_per_window, bool) or not isinstance(
                max_per_window, int) or max_per_window < 1:
            raise ValueError("max_per_window must be at least 1")
        self.window_seconds = float(window_seconds)
        self.max_per_window = max_per_window
        # key -> [window index, emitted in window, suppressed since emission]
        self._state = {}
        self._newest_window = None

    def _forget(self, window):
        stale = [key for key, entry in self._state.items()
                 if entry[0] < window - 1]
        for key in stale:
            del self._state[key]

    def offer(self, now, key, message):
        window = math.floor(now / self.window_seconds)
        if self._newest_window is None:
            self._newest_window = window
        else:
            self._newest_window = max(self._newest_window, window)
        self._forget(self._newest_window)
        entry = self._state.get(key)
        if entry is None:
            entry = [window, 0, 0]
            self._state[key] = entry
        elif entry[0] != window:
            entry = [window, 0, entry[2]]
            self._state[key] = entry
        if entry[1] < self.max_per_window:
            entry[1] += 1
            suppressed = entry[2]
            entry[2] = 0
            return {"key": key, "message": message, "suppressed": suppressed}
        entry[2] += 1
        return None

    def tracked_keys(self):
        return len(self._state)
'''


REFERENCES["reliability_observability_performance-0007"] = r'''
class FrequentItems:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("capacity must be an integer")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._counts = {}

    def offer(self, item):
        if item in self._counts:
            self._counts[item] += 1
        elif len(self._counts) < self.capacity:
            self._counts[item] = 1
        else:
            for key in list(self._counts):
                self._counts[key] -= 1
                if self._counts[key] == 0:
                    del self._counts[key]

    def tracked(self):
        return len(self._counts)

    def top(self, k):
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k < 0:
            raise ValueError("k must be non-negative")
        ordered = sorted(self._counts.items(),
                         key=lambda pair: (-pair[1], pair[0]))
        return ordered[:k]
'''


REFERENCES["reliability_observability_performance-0008"] = r'''
def critical_path(spans):
    if not isinstance(spans, (list, tuple)) or not spans:
        raise ValueError("at least one span is required")
    by_id = {}
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("each span must be a mapping")
        for key in ("span_id", "parent_id", "name", "start_ms", "end_ms"):
            if key not in span:
                raise ValueError("a span is missing " + key)
        span_id = span["span_id"]
        if span_id in by_id:
            raise ValueError("duplicate span_id " + repr(span_id))
        if span["end_ms"] < span["start_ms"]:
            raise ValueError("span " + repr(span_id) + " ends before it starts")
        by_id[span_id] = span

    roots = [span for span in by_id.values() if span["parent_id"] is None]
    if len(roots) != 1:
        raise ValueError("expected exactly one root, found " + str(len(roots)))

    children = {span_id: [] for span_id in by_id}
    for span in by_id.values():
        parent_id = span["parent_id"]
        if parent_id is None:
            continue
        if parent_id not in by_id:
            raise ValueError("parent " + repr(parent_id) + " names no span")
        parent = by_id[parent_id]
        if (span["start_ms"] < parent["start_ms"]
                or span["end_ms"] > parent["end_ms"]):
            raise ValueError("span " + repr(span["span_id"])
                             + " escapes its parent")
        children[parent_id].append(span)

    # Anything the root cannot reach is in a cycle, which no traversal from
    # the root would otherwise visit.
    root = roots[0]
    seen = set()
    stack = [root["span_id"]]
    while stack:
        current = stack.pop()
        if current in seen:
            raise ValueError("the parent links form a cycle")
        seen.add(current)
        stack.extend(child["span_id"] for child in children[current])
    if len(seen) != len(by_id):
        raise ValueError("the parent links form a cycle")

    path = []
    node = root
    while True:
        path.append(node["name"])
        kids = children[node["span_id"]]
        if not kids:
            break
        node = min(kids, key=lambda span: (-(span["end_ms"] - span["start_ms"]),
                                           span["start_ms"], span["span_id"]))
    return {"path": path, "duration_ms": root["end_ms"] - root["start_ms"]}
'''


MUTATIONS: dict[str, tuple[str, str]] = {
    # Sign the payload alone, so the header can be rewritten after signing
    # and the token still verifies -- the alg-confusion family.
    "http_apis_authn_appsec-0001": (
        'key, f"{head}.{body}".encode("ascii"), hashlib.sha256',
        'key, body.encode("ascii"), hashlib.sha256',
    ),
    # Compare the root as a text prefix rather than on a segment boundary,
    # so /srv/data-private passes the containment check for /srv/data.
    "http_apis_authn_appsec-0002": (
        'not candidate.startswith(base.rstrip("/") + "/")',
        "not candidate.startswith(base)",
    ),
    # Derive from a fixed salt: every user who picked the same password now
    # shares one hash, and one precomputed table covers all of them.
    "http_apis_authn_appsec-0003": (
        "salt = secrets.token_bytes(16)",
        'salt = b"static-salt-1234"',
    ),
    # Authenticate without authorising: any signed-in principal reads any
    # resource, which is the object-level access defect exactly.
    "http_apis_authn_appsec-0004": (
        '    if action == "read" and resource.get("visibility") == "public":\n        return True',
        '    if action == "read":\n        return True',
    ),
    # Sign the nonce alone, unbinding the token from its session so one
    # issued anywhere validates everywhere.
    "http_apis_authn_appsec-0005": (
        'key, f"{session_id}.{nonce}".encode("ascii"), hashlib.sha256',
        'key, nonce.encode("ascii"), hashlib.sha256',
    ),
    # Drop CR and LF from the forbidden set, letting a cookie value inject
    # a second response header.
    "http_apis_authn_appsec-0006": (
        "{chr(code) for code in range(0x21)}",
        "{chr(code) for code in range(0x21) if code not in (0x0A, 0x0D)}",
    ),
    # Miss the protocol-relative form, which a browser resolves as an
    # absolute URL to an attacker's host.
    "http_apis_authn_appsec-0007": (
        '    if normalised.startswith("//"):\n        return "/"\n',
        "",
    ),
    # Redact in place, destroying the caller's own data on the way to the
    # log.
    "http_apis_authn_appsec-0008": (
        "        result = {}\n        for key, value in record.items():",
        "        result = record\n        for key, value in list(record.items()):",
    ),
    # Keep the account row a failed item already wrote by releasing its
    # savepoint instead of rolling back to it: the batch then commits half
    # of a row it reported as rejected.
    "databases_migrations_transactions-0005": (
        '                connection.execute(f"ROLLBACK TO item_{index}")\n'
        '                connection.execute(f"RELEASE item_{index}")\n',
        '                connection.execute(f"RELEASE item_{index}")\n',
    ),
    # Trust the recorded version without comparing checksums, so a migration
    # edited after it was applied is silently accepted as already done.
    "databases_migrations_transactions-0001": (
        "if applied[version] != digest:",
        "if False:",
    ),
    # Match the row on id alone. The update still succeeds and still bumps
    # the version, so only a concurrent writer's lost edit reveals it.
    "databases_migrations_transactions-0002": (
        '"WHERE id = ? AND version = ?",',
        '"WHERE id = ? AND version >= ?",',
    ),
    # Commit the work already done instead of unwinding it, publishing an
    # outbox event for an order that never committed.
    "databases_migrations_transactions-0003": (
        '    except Exception:\n        connection.execute("ROLLBACK")\n        raise',
        '    except Exception:\n        connection.execute("COMMIT")\n        raise',
    ),
    # Compare the timestamp alone, which skips every row that shares the
    # boundary row's created_at value.
    "databases_migrations_transactions-0004": (
        '"WHERE created_at > ? OR (created_at = ? AND id > ?) "',
        '"WHERE created_at > ? OR (created_at = ? AND id > ? AND 0) "',
    ),
    # Treat a supplied None as a value to store, erasing the column the
    # caller only meant to leave alone.
    "databases_migrations_transactions-0006": (
        "if record.get(key) is not None",
        "if key in record",
    ),
    # Enable foreign keys after the transaction opens, where the pragma is
    # a silent no-op -- so the delete orphans the orders instead of failing.
    "databases_migrations_transactions-0007": (
        '    connection.execute("PRAGMA foreign_keys = ON")\n    connection.execute("BEGIN")',
        '    connection.execute("BEGIN")\n    connection.execute("PRAGMA foreign_keys = ON")',
    ),
    # Ignore the stored charge and bill again on every retry.
    "databases_migrations_transactions-0008": (
        "if row is not None:",
        "if False:",
    ),
    # Let the bucket accumulate while idle, so a long quiet period buys a
    # burst the rate limit was supposed to forbid.
    "concurrency_async_distributed-0001": (
        "self.capacity, self._tokens + elapsed * self.refill_per_second",
        "float('inf'), self._tokens + elapsed * self.refill_per_second",
    ),
    # Hold the circuit open through the instant the window elapses, so the
    # half-open trial never gets its one call.
    "concurrency_async_distributed-0002": (
        "if self._clock() - self._opened_at < self.recovery_seconds:",
        "if self._clock() - self._opened_at <= self.recovery_seconds:",
    ),
    # Sleep after the final attempt: the caller pays a delay for a retry
    # that never happens.
    "concurrency_async_distributed-0003": (
        "if index == attempts - 1:",
        "if index == attempts:",
    ),
    # Collapse concurrency into ordering, reporting a causal relationship
    # between events that never saw each other.
    "concurrency_async_distributed-0004": (
        'if less and greater:\n        return "concurrent"',
        'if False:\n        return "concurrent"',
    ),
    # Forget which sequence numbers were already applied, so a redelivery
    # replays a side effect that the inbox exists to apply once.
    "concurrency_async_distributed-0005": (
        "if sequence < self._next or sequence in self._buffer:",
        "if sequence in self._buffer:",
    ),
    # Return a partial plan for a cyclic graph instead of rejecting it.
    "concurrency_async_distributed-0006": (
        'if scheduled != len(names):\n        raise ValueError("dependencies contain a cycle")',
        'if False:\n        raise ValueError("dependencies contain a cycle")',
    ),
    # Mark the Once done before the factory returns, so one failed attempt
    # permanently poisons the value nobody ever produced.
    "concurrency_async_distributed-0007": (
        "value = factory()\n            self._value = value\n            self._done = True",
        "self._done = True\n            value = factory()\n            self._value = value",
    ),
    # Drop writer preference: readers keep arriving and the writer waits
    # behind an unbroken queue of them.
    "concurrency_async_distributed-0008": (
        "while self._writer or self._waiting_writers > 0:",
        "while self._writer:",
    ),
    # Take the first range that covers a candidate instead of the most
    # specific one: `*/*;q=0.5, text/html;q=0.1` then reads html at 0.5.
    "requirements_api_contracts-0001": (
        "if chosen is None or specificity > chosen[0]:",
        "if chosen is None:",
    ),
    # Build the result on the caller's dict, so patching edits the target.
    "requirements_api_contracts-0002": (
        "result = copy.deepcopy(target) if isinstance(target, dict) else {}",
        "result = target if isinstance(target, dict) else {}",
    ),
    # Treat a matched If-None-Match as 304 for every method, losing the
    # 412 an unsafe method owes a caller whose copy is already current.
    "requirements_api_contracts-0003": (
        'return 304 if method in ("GET", "HEAD") else 412',
        "return 304",
    ),
    # Trust the client's `last` offset instead of clamping it to the entity.
    "requirements_api_contracts-0004": (
        "end = min(end, length - 1)",
        "end = end",
    ),
    # Resume at the cursor rather than after it, re-emitting one row a page.
    "requirements_api_contracts-0005": (
        "if after is None or key > after",
        "if after is None or key >= after",
    ),
    # Stop reporting a newly added required field as breaking.
    "requirements_api_contracts-0006": (
        'if name not in old and after.get("required", False):',
        "if False:",
    ),
    # Record the key before the side effect succeeds, so a failed attempt
    # burns the key and the caller can never retry the charge.
    "requirements_api_contracts-0007": (
        "response = handler()\n        self._entries[key] = (fingerprint, response)",
        "self._entries[key] = (fingerprint, None)\n        response = handler()",
    ),
    # Let a repeated parameter overwrite, so the last occurrence wins.
    "requirements_api_contracts-0008": (
        "if name and name not in entry:",
        "if name:",
    ),
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
    # Negate the translation without rotating it. Correct for a transform
    # that only translates, wrong the moment a joint also rotates -- so it
    # survives a bench example and inverts an assembly incorrectly.
    "scientific_3d_geometry_robotics-0001": (
        "offset = -sum(t[j][i] * t[j][3] for j in range(3))",
        "offset = -t[i][3]",
    ),
    # Reverse the vector part, turning the Hamilton product into q2*q1.
    # Indistinguishable whenever the two rotations commute.
    "scientific_3d_geometry_robotics-0002": (
        "        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,\n"
        "        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,\n"
        "        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,\n",
        "        w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2,\n"
        "        w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2,\n"
        "        w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2,\n",
    ),
    # Count every directed edge instead of deduplicating to undirected ones,
    # which doubles E and makes the characteristic of a closed solid -4.
    "scientific_3d_geometry_robotics-0003": (
        "            len(vertices) - len(undirected) + len(triangles),",
        "            len(vertices) - len(directed) + len(triangles),",
    ),
    # Weight each tetrahedron's moment by the face centroid (a+b+c)/3
    # instead of the solid centroid (a+b+c+origin)/4. The volume stays
    # exactly right and only the centre of mass moves, which is the shape
    # of the defect that unbalances a simulated part.
    "scientific_3d_geometry_robotics-0004": (
        "            moment[axis] += det * (a[axis] + b[axis] + c[axis]) / 24.0",
        "            moment[axis] += det * (a[axis] + b[axis] + c[axis]) / 18.0",
    ),
    # Add the outer product instead of subtracting it. The diagonal is then
    # wrong in the one direction the parallel-axis theorem leaves alone.
    "scientific_3d_geometry_robotics-0005": (
        "inertia[i][j] + mass * (delta - offset[i] * offset[j])",
        "inertia[i][j] + mass * (delta + offset[i] * offset[j])",
    ),
    # Drop the forward-hit test, so geometry behind the camera or behind a
    # sensor origin reports a hit at a negative parameter.
    "scientific_3d_geometry_robotics-0006": (
        "    if t <= 0.0:\n        return None\n    return t\n",
        "    return t\n",
    ),
    # Omit the attribute byte count, making every facet record 48 bytes
    # rather than the 50 the format fixes. Slicers reject the result.
    "scientific_3d_geometry_robotics-0007": (
        '        out += struct.pack("<H", 0)\n',
        "",
    ),
    # Flip the sign of the link matrix's -sin(theta)cos(alpha) term. Every
    # single-joint pose still lands correctly; only a chain with two bent
    # joints diverges.
    "scientific_3d_geometry_robotics-0008": (
        "        (ct, -st * ca, st * sa, a * ct),",
        "        (ct, st * ca, st * sa, a * ct),",
    ),
    # Floor the bucket index instead of ceiling it. Memory stays bounded and
    # every quantile still looks plausible, but the representative now sits
    # below its bucket's range, so the error reaches 3% against a 1% promise.
    "reliability_observability_performance-0001": (
        "index = math.ceil(math.log(value) / self._log_gamma)",
        "index = math.floor(math.log(value) / self._log_gamma)",
    ),
    # Derive the decision from the built-in hash. Python salts str hashing per
    # process, so every service in the trace samples a different subset and
    # the trace is never complete anywhere.
    "reliability_observability_performance-0002": (
        'digest = hashlib.sha256(trace_id.encode("utf-8")).digest()\n'
        '    value = int.from_bytes(digest[:8], "big")',
        "value = hash(trace_id) % 2 ** 64",
    ),
    # Never evict. Totals are right for a while and residency grows with the
    # stream, which is the leak that survives every short test.
    "reliability_observability_performance-0003": (
        "        for index in [key for key in self._buckets if key <= oldest]:\n"
        "            del self._buckets[index]",
        "        return",
    ),
    # Report healthy while a non-critical dependency is down -- the aggregate
    # that keeps a broken subsystem invisible until a user reports it.
    "reliability_observability_performance-0004": (
        "    elif down or degraded:",
        "    elif degraded:",
    ),
    # Alert on the long window alone. The short-window veto is what stops an
    # incident that already stopped from paging, so removing it pages on
    # history.
    "reliability_observability_performance-0005": (
        'if burn_rates["1h"] >= 14.4 and burn_rates["5m"] >= 14.4:',
        'if burn_rates["1h"] >= 14.4:',
    ),
    # Keep every key forever. Throttling still works; the process now leaks
    # one entry per distinct key it has ever seen.
    "reliability_observability_performance-0006": (
        "        stale = [key for key, entry in self._state.items()\n"
        "                 if entry[0] < window - 1]",
        "        stale = []",
    ),
    # Track a new item instead of decrementing, abandoning the counter bound:
    # the summary becomes an exact tally the size of the stream.
    "reliability_observability_performance-0007": (
        "            for key in list(self._counts):\n"
        "                self._counts[key] -= 1\n"
        "                if self._counts[key] == 0:\n"
        "                    del self._counts[key]",
        "            self._counts[item] = 1",
    ),
    # Descend into the latest-starting child rather than the longest-running
    # one, which reports the last thing that happened as the cause of the
    # trace's duration.
    "reliability_observability_performance-0008": (
        '-(span["end_ms"] - span["start_ms"]),',
        '-span["start_ms"],',
    ),
}
