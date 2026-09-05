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


REFERENCES["validation_parsing_serialization-0007"] = r'''
def _is_number(text):
    return bool(text) and all(character in "0123456789" for character in text)


def parse_range(header, length):
    if not isinstance(header, str):
        raise ValueError("header must be a string")
    if not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative integer")
    unit, separator, spec = header.partition("=")
    if not separator or unit.strip().lower() != "bytes":
        raise ValueError("only the bytes unit is supported")

    ranges = []
    for raw in spec.split(","):
        piece = raw.strip()
        if not piece:
            raise ValueError("empty range spec")
        first, dash, last = piece.partition("-")
        if not dash:
            raise ValueError("a range spec needs a dash")
        first = first.strip()
        last = last.strip()
        if not first:
            if not _is_number(last):
                raise ValueError("malformed suffix range")
            suffix = int(last)
            if suffix == 0:
                continue
            start = max(0, length - suffix)
            end = length - 1
        else:
            if not _is_number(first):
                raise ValueError("malformed first-byte position")
            start = int(first)
            if not last:
                end = length - 1
            else:
                if not _is_number(last):
                    raise ValueError("malformed last-byte position")
                end = int(last)
                if end < start:
                    raise ValueError("last-byte position precedes the first")
            if end > length - 1:
                end = length - 1
        if length == 0 or start > length - 1:
            continue
        ranges.append((start, end))
    if not ranges:
        raise ValueError("no range in the header is satisfiable")
    return ranges
'''


REFERENCES["validation_parsing_serialization-0008"] = r'''
_HEX = "0123456789abcdefABCDEF"


def _unquote(text):
    out = bytearray()
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character == "+":
            out.append(0x20)
            index += 1
            continue
        if character == "%":
            token = text[index + 1:index + 3]
            if len(token) != 2 or token[0] not in _HEX or token[1] not in _HEX:
                raise ValueError("malformed percent escape")
            out.append(int(token, 16))
            index += 3
            continue
        out.extend(character.encode("utf-8"))
        index += 1
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("percent escapes did not form valid UTF-8") from exc


def parse_query(text):
    if not isinstance(text, str):
        raise ValueError("query must be a string")
    if text.startswith("?"):
        text = text[1:]
    pairs = []
    for segment in text.split("&"):
        if not segment:
            continue
        name, _, value = segment.partition("=")
        pairs.append((_unquote(name), _unquote(value)))
    return pairs
'''


REFERENCES["validation_parsing_serialization-0009"] = r'''
def decode_utf8(data):
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("data must be a bytes object")
    out = []
    index = 0
    length = len(data)
    while index < length:
        byte = data[index]
        if byte < 0x80:
            out.append(chr(byte))
            index += 1
            continue
        if byte < 0xc2:
            raise ValueError("continuation byte or overlong lead byte")
        if byte < 0xe0:
            width = 2
            code = byte & 0x1f
        elif byte < 0xf0:
            width = 3
            code = byte & 0x0f
        elif byte < 0xf5:
            width = 4
            code = byte & 0x07
        else:
            raise ValueError("lead byte above the Unicode range")
        if index + width > length:
            raise ValueError("truncated sequence")
        for offset in range(1, width):
            follow = data[index + offset]
            if follow & 0xc0 != 0x80:
                raise ValueError("malformed continuation byte")
            code = (code << 6) | (follow & 0x3f)
        if width == 3 and code < 0x800:
            raise ValueError("overlong encoding")
        if width == 4 and code < 0x10000:
            raise ValueError("overlong encoding")
        if 0xd800 <= code <= 0xdfff:
            raise ValueError("surrogate code point")
        if code > 0x10ffff:
            raise ValueError("code point above U+10FFFF")
        out.append(chr(code))
        index += width
    return "".join(out)
'''


REFERENCES["validation_parsing_serialization-0010"] = r'''
import re

_URI = re.compile(
    r"^(?:(?P<scheme>[^:/?#]+):)?"
    r"(?://(?P<authority>[^/?#]*))?"
    r"(?P<path>[^?#]*)"
    r"(?:\?(?P<query>[^#]*))?"
    r"(?:#(?P<fragment>.*))?$"
)


def _split(text):
    match = _URI.match(text)
    if match is None:
        raise ValueError("not a URI reference")
    return match.groupdict()


def _remove_dot_segments(path):
    out = []
    while path:
        if path.startswith("../"):
            path = path[3:]
        elif path.startswith("./"):
            path = path[2:]
        elif path.startswith("/./"):
            path = "/" + path[3:]
        elif path == "/.":
            path = "/"
        elif path.startswith("/../"):
            path = "/" + path[4:]
            if out:
                out.pop()
        elif path == "/..":
            path = "/"
            if out:
                out.pop()
        elif path in (".", ".."):
            path = ""
        else:
            index = path.find("/", 1) if path.startswith("/") else path.find("/")
            if index < 0:
                out.append(path)
                path = ""
            else:
                out.append(path[:index])
                path = path[index:]
    return "".join(out)


def _merge(base, reference_path):
    if base["authority"] is not None and not base["path"]:
        return "/" + reference_path
    base_path = base["path"]
    index = base_path.rfind("/")
    if index < 0:
        return reference_path
    return base_path[:index + 1] + reference_path


def _recompose(parts):
    text = ""
    if parts["scheme"] is not None:
        text += parts["scheme"] + ":"
    if parts["authority"] is not None:
        text += "//" + parts["authority"]
    text += parts["path"]
    if parts["query"] is not None:
        text += "?" + parts["query"]
    if parts["fragment"] is not None:
        text += "#" + parts["fragment"]
    return text


def resolve_uri(base, reference):
    if not isinstance(base, str) or not isinstance(reference, str):
        raise ValueError("base and reference must be strings")
    base_parts = _split(base)
    if base_parts["scheme"] is None:
        raise ValueError("the base URI must carry a scheme")
    ref = _split(reference)

    target = {
        "scheme": None,
        "authority": None,
        "path": "",
        "query": None,
        "fragment": ref["fragment"],
    }
    if ref["scheme"] is not None:
        target["scheme"] = ref["scheme"]
        target["authority"] = ref["authority"]
        target["path"] = _remove_dot_segments(ref["path"])
        target["query"] = ref["query"]
        return _recompose(target)

    target["scheme"] = base_parts["scheme"]
    if ref["authority"] is not None:
        target["authority"] = ref["authority"]
        target["path"] = _remove_dot_segments(ref["path"])
        target["query"] = ref["query"]
        return _recompose(target)

    target["authority"] = base_parts["authority"]
    if ref["path"] == "":
        target["path"] = base_parts["path"]
        target["query"] = (
            ref["query"] if ref["query"] is not None else base_parts["query"]
        )
        return _recompose(target)

    if ref["path"].startswith("/"):
        target["path"] = _remove_dot_segments(ref["path"])
    else:
        target["path"] = _remove_dot_segments(_merge(base_parts, ref["path"]))
    target["query"] = ref["query"]
    return _recompose(target)
'''


REFERENCES["validation_parsing_serialization-0011"] = r'''
_HEX_DIGITS = b"0123456789abcdefABCDEF"


def decode_chunked(data):
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("data must be a bytes object")
    data = bytes(data)
    body = bytearray()
    index = 0
    while True:
        end = data.find(b"\r\n", index)
        if end < 0:
            raise ValueError("chunk size line is not terminated")
        line = data[index:end]
        index = end + 2
        size_text = line.split(b";", 1)[0].strip()
        if not size_text:
            raise ValueError("missing chunk size")
        if any(byte not in _HEX_DIGITS for byte in size_text):
            raise ValueError("chunk size is not hexadecimal")
        size = int(size_text, 16)
        if size == 0:
            break
        if index + size > len(data):
            raise ValueError("chunk data is truncated")
        body.extend(data[index:index + size])
        index += size
        if data[index:index + 2] != b"\r\n":
            raise ValueError("chunk data is not followed by CRLF")
        index += 2

    trailers = {}
    while True:
        end = data.find(b"\r\n", index)
        if end < 0:
            raise ValueError("trailer section is not terminated")
        line = data[index:end]
        index = end + 2
        if not line:
            break
        name, separator, value = line.partition(b":")
        if not separator:
            raise ValueError("malformed trailer field")
        trailers[name.decode("ascii").strip().lower()] = (
            value.decode("ascii").strip()
        )
    return bytes(body), trailers
'''


REFERENCES["validation_parsing_serialization-0012"] = r'''
_HEX = "0123456789abcdefABCDEF"


def normalize_ipv6(text):
    if not isinstance(text, str):
        raise ValueError("address must be a string")
    work = text.strip()
    if not work:
        raise ValueError("empty address")

    mapped = False
    dotted_text = ""
    if "." in work:
        head, separator, dotted = work.rpartition(":")
        if not separator:
            raise ValueError("a dotted quad must follow a colon")
        octets = dotted.split(".")
        if len(octets) != 4:
            raise ValueError("malformed embedded IPv4 address")
        numbers = []
        for octet in octets:
            if not octet or any(character not in "0123456789" for character in octet):
                raise ValueError("malformed IPv4 octet")
            if len(octet) > 1 and octet[0] == "0":
                raise ValueError("an IPv4 octet may not carry a leading zero")
            number = int(octet)
            if number > 255:
                raise ValueError("IPv4 octet out of range")
            numbers.append(number)
        dotted_text = "%d.%d.%d.%d" % tuple(numbers)
        work = "%s:%x:%x" % (
            head,
            (numbers[0] << 8) | numbers[1],
            (numbers[2] << 8) | numbers[3],
        )
        mapped = True

    if work.count("::") > 1:
        raise ValueError("an address may compress only one run")
    if "::" in work:
        left, _, right = work.partition("::")
        left_parts = left.split(":") if left else []
        right_parts = right.split(":") if right else []
        if any(part == "" for part in left_parts + right_parts):
            raise ValueError("malformed group separator")
        missing = 8 - (len(left_parts) + len(right_parts))
        if missing < 1:
            raise ValueError("a compressed run must cover a zero group")
        groups = left_parts + ["0"] * missing + right_parts
    else:
        groups = work.split(":")
        if len(groups) != 8:
            raise ValueError("an uncompressed address needs eight groups")

    values = []
    for part in groups:
        if not 1 <= len(part) <= 4:
            raise ValueError("a group is one to four hexadecimal digits")
        if any(character not in _HEX for character in part):
            raise ValueError("a group is one to four hexadecimal digits")
        values.append(int(part, 16))

    scan = 6 if mapped else 8
    best_start = -1
    best_length = 0
    index = 0
    while index < scan:
        if values[index]:
            index += 1
            continue
        run = index
        while run < scan and values[run] == 0:
            run += 1
        if run - index > best_length:
            best_start = index
            best_length = run - index
        index = run
    if best_length < 2:
        best_start = -1
        best_length = 0

    if mapped:
        rendered = ["%x" % value for value in values[:6]] + [dotted_text]
    else:
        rendered = ["%x" % value for value in values]
    if best_start < 0:
        return ":".join(rendered)
    return (
        ":".join(rendered[:best_start])
        + "::"
        + ":".join(rendered[best_start + best_length:])
    )
'''


REFERENCES["validation_parsing_serialization-0013"] = r'''
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

#: Padding length -> how many whole bytes the block carries. Every other
#: padding length is unreachable for a base32 encoder.
_KEEP = {0: 5, 1: 4, 3: 3, 4: 2, 6: 1}


def decode_base32(text):
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    if len(text) % 8 != 0:
        raise ValueError("length is not a multiple of eight")

    out = bytearray()
    for offset in range(0, len(text), 8):
        block = text[offset:offset + 8]
        stripped = block.rstrip("=")
        pad = len(block) - len(stripped)
        if "=" in stripped:
            raise ValueError("padding appears before the end of a block")
        if pad and offset + 8 != len(text):
            raise ValueError("padding appears before the final block")
        if pad not in _KEEP:
            raise ValueError("invalid padding length")

        bits = 0
        width = 0
        for character in stripped:
            index = _ALPHABET.find(character)
            if index < 0:
                raise ValueError("character outside the base32 alphabet")
            bits = (bits << 5) | index
            width += 5

        keep = _KEEP[pad]
        leftover = width - keep * 8
        if leftover and bits & ((1 << leftover) - 1):
            raise ValueError("non-canonical trailing bits")
        out.extend((bits >> leftover).to_bytes(keep, "big"))
    return bytes(out)
'''


REFERENCES["validation_parsing_serialization-0014"] = r'''
_WHITESPACE = " \t\n\r"
_DIGITS = "0123456789"
_HEX = "0123456789abcdefABCDEF"
_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/", "b": "\b",
    "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}


class _Scanner:
    def __init__(self, text):
        self.text = text
        self.index = 0

    def fail(self, message):
        raise ValueError("%s at position %d" % (message, self.index))

    def peek(self):
        if self.index >= len(self.text):
            return ""
        return self.text[self.index]

    def at_digit(self):
        character = self.peek()
        return character != "" and character in _DIGITS

    def skip_whitespace(self):
        while self.index < len(self.text) and self.text[self.index] in _WHITESPACE:
            self.index += 1

    def expect(self, character):
        if self.peek() != character:
            self.fail("expected %r" % character)
        self.index += 1

    def parse_value(self):
        character = self.peek()
        if character == "{":
            return self.parse_object()
        if character == "[":
            return self.parse_array()
        if character == '"':
            return self.parse_string()
        if self.text.startswith("true", self.index):
            self.index += 4
            return True
        if self.text.startswith("false", self.index):
            self.index += 5
            return False
        if self.text.startswith("null", self.index):
            self.index += 4
            return None
        if character == "-" or self.at_digit():
            return self.parse_number()
        self.fail("expected a value")

    def parse_object(self):
        self.expect("{")
        result = {}
        self.skip_whitespace()
        if self.peek() == "}":
            self.index += 1
            return result
        while True:
            self.skip_whitespace()
            if self.peek() != '"':
                self.fail("an object key must be a string")
            key = self.parse_string()
            self.skip_whitespace()
            self.expect(":")
            self.skip_whitespace()
            result[key] = self.parse_value()
            self.skip_whitespace()
            character = self.peek()
            if character == ",":
                self.index += 1
                continue
            if character == "}":
                self.index += 1
                return result
            self.fail("expected a comma or a closing brace")

    def parse_array(self):
        self.expect("[")
        result = []
        self.skip_whitespace()
        if self.peek() == "]":
            self.index += 1
            return result
        while True:
            self.skip_whitespace()
            result.append(self.parse_value())
            self.skip_whitespace()
            character = self.peek()
            if character == ",":
                self.index += 1
                continue
            if character == "]":
                self.index += 1
                return result
            self.fail("expected a comma or a closing bracket")

    def read_hex4(self):
        token = self.text[self.index:self.index + 4]
        if len(token) != 4 or any(character not in _HEX for character in token):
            self.fail("malformed unicode escape")
        self.index += 4
        return int(token, 16)

    def parse_string(self):
        self.expect('"')
        chunks = []
        while True:
            if self.index >= len(self.text):
                self.fail("unterminated string")
            character = self.text[self.index]
            if character == '"':
                self.index += 1
                return "".join(chunks)
            if character == "\\":
                self.index += 1
                code = self.peek()
                if code in _ESCAPES:
                    chunks.append(_ESCAPES[code])
                    self.index += 1
                    continue
                if code != "u":
                    self.fail("unknown escape")
                self.index += 1
                first = self.read_hex4()
                if 0xdc00 <= first <= 0xdfff:
                    self.fail("unpaired low surrogate")
                if 0xd800 <= first <= 0xdbff:
                    if not self.text.startswith("\\u", self.index):
                        self.fail("unpaired high surrogate")
                    self.index += 2
                    second = self.read_hex4()
                    if not 0xdc00 <= second <= 0xdfff:
                        self.fail("high surrogate without a low surrogate")
                    first = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                chunks.append(chr(first))
                continue
            if ord(character) < 0x20:
                self.fail("unescaped control character")
            chunks.append(character)
            self.index += 1

    def parse_number(self):
        start = self.index
        if self.peek() == "-":
            self.index += 1
        if not self.at_digit():
            self.fail("a number needs at least one digit")
        if self.peek() == "0":
            self.index += 1
        else:
            while self.at_digit():
                self.index += 1
        is_float = False
        if self.peek() == ".":
            is_float = True
            self.index += 1
            if not self.at_digit():
                self.fail("a fraction needs a digit")
            while self.at_digit():
                self.index += 1
        if self.peek() in ("e", "E"):
            is_float = True
            self.index += 1
            if self.peek() in ("+", "-"):
                self.index += 1
            if not self.at_digit():
                self.fail("an exponent needs a digit")
            while self.at_digit():
                self.index += 1
        token = self.text[start:self.index]
        return float(token) if is_float else int(token)


def parse_json(text):
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    scanner = _Scanner(text)
    scanner.skip_whitespace()
    if scanner.index >= len(text):
        raise ValueError("the document holds no value")
    value = scanner.parse_value()
    scanner.skip_whitespace()
    if scanner.index != len(text):
        raise ValueError("trailing content after the top-level value")
    return value
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


REFERENCES["testing_debugging_repair_refactoring-0001"] = r'''
def minimize_failing_input(items, is_failing):
    items = list(items)
    if is_failing([]):
        return []
    if not is_failing(items):
        raise ValueError("the whole input does not fail; nothing to reduce")

    def split(sequence, count):
        size, extra = divmod(len(sequence), count)
        chunks = []
        start = 0
        for index in range(count):
            width = size + (1 if index < extra else 0)
            if width:
                chunks.append(sequence[start:start + width])
            start += width
        return chunks

    current = items
    granularity = 2
    while len(current) >= 2:
        chunks = split(current, min(granularity, len(current)))
        reduced = False
        for chunk in chunks:
            if is_failing(chunk):
                current = chunk
                granularity = 2
                reduced = True
                break
        if not reduced:
            for index in range(len(chunks)):
                complement = [value
                              for position, chunk in enumerate(chunks)
                              if position != index
                              for value in chunk]
                if complement and is_failing(complement):
                    current = complement
                    granularity = max(len(chunks) - 1, 2)
                    reduced = True
                    break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return current
'''


REFERENCES["testing_debugging_repair_refactoring-0002"] = r'''
def first_bad_revision(revisions, is_bad):
    revisions = list(revisions)
    if not revisions:
        raise ValueError("there are no revisions to bisect")
    low, high = 0, len(revisions)
    while low < high:
        middle = (low + high) // 2
        if is_bad(revisions[middle]):
            high = middle
        else:
            low = middle + 1
    if low == len(revisions):
        return None
    return revisions[low]
'''


REFERENCES["testing_debugging_repair_refactoring-0003"] = r'''
import re

_FRAME = re.compile(r'^\s*File "(?P<path>.+)", line \d+, in (?P<function>.+)$')


def _signature(text):
    frames = []
    for line in text.splitlines():
        match = _FRAME.match(line)
        if match:
            path = match.group("path").replace("\\", "/")
            basename = path.rsplit("/", 1)[-1]
            frames.append(f"{basename}:{match.group('function').strip()}")
    if not frames:
        raise ValueError("a traceback with no frames is not a crash report")
    body = [line.strip() for line in text.splitlines() if line.strip()]
    last = body[-1]
    kind = last.split(":", 1)[0].strip() if ":" in last else last
    return kind + "|" + ";".join(frames)


def group_failures(reports):
    reports = list(reports)
    if not reports:
        raise ValueError("there are no reports to group")
    buckets = {}
    for report in reports:
        if "test" not in report or "traceback" not in report:
            raise ValueError("a report needs both a test and a traceback")
        signature = _signature(report["traceback"])
        entry = buckets.setdefault(signature, {"count": 0, "tests": set()})
        entry["count"] += 1
        entry["tests"].add(report["test"])
    grouped = [{"signature": signature, "count": entry["count"],
                "tests": sorted(entry["tests"])}
               for signature, entry in buckets.items()]
    grouped.sort(key=lambda group: (-group["count"], group["signature"]))
    return grouped
'''


REFERENCES["testing_debugging_repair_refactoring-0004"] = r'''
import math


def rank_suspicious_lines(coverage, outcomes):
    if not outcomes:
        raise ValueError("no test outcomes were supplied")
    if set(coverage) != set(outcomes):
        raise ValueError("coverage and outcomes name different tests")
    total_failing = sum(1 for passed in outcomes.values() if not passed)
    if total_failing == 0:
        raise ValueError("no test failed, so there is nothing to localise")

    failing_hits = {}
    passing_hits = {}
    for test, lines in coverage.items():
        hits = passing_hits if outcomes[test] else failing_hits
        for line in lines:
            if isinstance(line, bool) or not isinstance(line, int) or line < 1:
                raise ValueError(f"{line!r} is not a positive line number")
            hits[line] = hits.get(line, 0) + 1

    ranked = []
    for line in sorted(set(failing_hits) | set(passing_hits)):
        executed_failing = failing_hits.get(line, 0)
        if executed_failing == 0:
            continue
        executed_passing = passing_hits.get(line, 0)
        ranked.append((line, executed_failing / math.sqrt(
            total_failing * (executed_failing + executed_passing))))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked
'''


REFERENCES["testing_debugging_repair_refactoring-0005"] = r'''
import ast
import keyword


def _bound_names(function):
    bound = set()
    arguments = (function.args.posonlyargs + function.args.args
                 + function.args.kwonlyargs)
    for argument in arguments:
        bound.add(argument.arg)
    for extra in (function.args.vararg, function.args.kwarg):
        if extra is not None:
            bound.add(extra.arg)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx,
                                                     (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
    return bound


def rename_local(source, function_name, old_name, new_name):
    tree = ast.parse(source)
    target = None
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name):
            target = node
            break
    if target is None:
        raise ValueError(f"no top-level function named {function_name!r}")
    if not new_name.isidentifier() or keyword.iskeyword(new_name):
        raise ValueError(f"{new_name!r} is not a usable identifier")
    for node in ast.walk(target):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) and node is not target):
            raise ValueError("the function contains a nested definition")
        if (isinstance(node, (ast.Global, ast.Nonlocal))
                and old_name in node.names):
            raise ValueError(f"{old_name!r} is not local to {function_name!r}")

    bound = _bound_names(target)
    if old_name not in bound:
        raise ValueError(f"{old_name!r} is not a local of {function_name!r}")
    used = set(bound)
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            used.add(node.id)
    if new_name in used:
        raise ValueError(f"{new_name!r} is already used in {function_name!r}")

    class _Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == old_name:
                node.id = new_name
            return node

        def visit_arg(self, node):
            if node.arg == old_name:
                node.arg = new_name
            return node

    _Renamer().visit(target)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
'''


REFERENCES["testing_debugging_repair_refactoring-0006"] = r'''
_MISSING = "<missing>"
_SCALARS = (str, int, float, bool, type(None))


def _check(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"dict keys must be strings, got {key!r}")
            _check(item)
    elif isinstance(value, list):
        for item in value:
            _check(item)
    elif not isinstance(value, _SCALARS):
        raise ValueError(f"unsupported value {value!r}")


def _walk(expected, actual, path):
    if type(expected) is not type(actual):
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return {"path": path + [key], "expected": _MISSING,
                        "actual": actual[key]}
            if key not in actual:
                return {"path": path + [key], "expected": expected[key],
                        "actual": _MISSING}
            found = _walk(expected[key], actual[key], path + [key])
            if found is not None:
                return found
        return None
    if isinstance(expected, list):
        shared = min(len(expected), len(actual))
        for index in range(shared):
            found = _walk(expected[index], actual[index], path + [index])
            if found is not None:
                return found
        if len(expected) != len(actual):
            return {
                "path": path + [shared],
                "expected": (expected[shared] if shared < len(expected)
                             else _MISSING),
                "actual": (actual[shared] if shared < len(actual)
                           else _MISSING),
            }
        return None
    if expected != actual:
        return {"path": path, "expected": expected, "actual": actual}
    return None


def first_difference(expected, actual):
    _check(expected)
    _check(actual)
    return _walk(expected, actual, [])
'''


REFERENCES["testing_debugging_repair_refactoring-0007"] = r'''
def classify_test_history(history, window):
    if not history:
        raise ValueError("no test history was supplied")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2")

    classes = {}
    for test, recorded in history.items():
        recorded = list(recorded)
        if len(recorded) < 2:
            raise ValueError(f"{test!r} has fewer than two recorded runs")
        for outcome in recorded:
            if outcome not in ("pass", "fail"):
                raise ValueError(f"{outcome!r} is not a recorded outcome")
        recent = recorded[-window:]
        transitions = sum(1 for index in range(1, len(recent))
                          if recent[index] != recent[index - 1])
        if transitions >= 2:
            classes[test] = "flaky"
        elif recent[-1] == "fail":
            classes[test] = "failing"
        else:
            classes[test] = "passing"
    return {
        "classes": classes,
        "quarantine": sorted(test for test, label in classes.items()
                             if label == "flaky"),
    }
'''


REFERENCES["testing_debugging_repair_refactoring-0008"] = r'''
def apply_patch(lines, hunks):
    original = list(lines)
    ordered = list(hunks)

    boundary = 0
    for index, hunk in enumerate(ordered):
        start = hunk["start"]
        remove = list(hunk["remove"])
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"hunk {index} has a non-integer start")
        if start < 1 or start > len(original) + 1:
            raise ValueError(f"hunk {index} starts outside the file")
        if start <= boundary:
            raise ValueError(f"hunk {index} overlaps the previous hunk")
        if start - 1 + len(remove) > len(original):
            raise ValueError(f"hunk {index} removes past the end of the file")
        if original[start - 1:start - 1 + len(remove)] != remove:
            raise ValueError(f"hunk {index} does not match the file")
        boundary = max(start, start - 1 + len(remove))

    result = []
    cursor = 0
    for hunk in ordered:
        start = hunk["start"]
        result.extend(original[cursor:start - 1])
        result.extend(list(hunk["insert"]))
        cursor = start - 1 + len(list(hunk["remove"]))
    result.extend(original[cursor:])
    return result
'''

REFERENCES["cicd_containers_packaging_platform-0001"] = r'''
import gzip
import io
import tarfile


def build_archive(entries):
    items = []
    seen = set()
    for path, data, mode in entries:
        if not path:
            raise ValueError("an entry has an empty path")
        parts = path.split("/")
        if path.startswith("/") or ".." in parts:
            raise ValueError(f"path {path!r} escapes the context")
        if path in seen:
            raise ValueError(f"duplicate path {path!r}")
        seen.add(path)
        items.append((path, bytes(data), int(mode)))
    items.sort(key=lambda item: item[0])

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path, data, mode in items:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mtime = 0
            info.mode = mode
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            tar.addfile(info, io.BytesIO(data))

    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0) as zipped:
        zipped.write(raw.getvalue())
    return packed.getvalue()
'''


REFERENCES["cicd_containers_packaging_platform-0002"] = r'''
_OPERATORS = ("==", "!=", ">=", "<=", "~=", ">", "<")


def _parse_version(text):
    if not text or not all(part.isdigit() for part in text.split(".")):
        raise ValueError(f"malformed version {text!r}")
    return tuple(int(part) for part in text.split("."))


def _pad(left, right):
    width = max(len(left), len(right))
    return (left + (0,) * (width - len(left)),
            right + (0,) * (width - len(right)))


def _compare(left, right):
    left, right = _pad(left, right)
    return (left > right) - (left < right)


def _satisfies(version, operator, bound):
    order = _compare(version, bound)
    if operator == "==":
        return order == 0
    if operator == "!=":
        return order != 0
    if operator == ">=":
        return order >= 0
    if operator == "<=":
        return order <= 0
    if operator == ">":
        return order > 0
    if operator == "<":
        return order < 0
    if operator == "~=":
        if len(bound) < 2:
            raise ValueError("~= needs at least two version components")
        ceiling = bound[:-1]
        upper = ceiling[:-1] + (ceiling[-1] + 1,)
        return order >= 0 and _compare(version, upper) < 0
    raise ValueError(f"unknown operator {operator!r}")


def _split_requirement(text):
    text = text.strip()
    if not text:
        raise ValueError("empty requirement")
    index = len(text)
    for position, character in enumerate(text):
        if not (character.isalnum() or character in "._-"):
            index = position
            break
    name = text[:index]
    if not name:
        raise ValueError(f"requirement {text!r} names no package")
    clauses = []
    rest = text[index:].strip()
    if rest:
        for piece in rest.split(","):
            piece = piece.strip()
            if not piece:
                raise ValueError(f"empty clause in {text!r}")
            for operator in _OPERATORS:
                if piece.startswith(operator):
                    clauses.append((operator, _parse_version(
                        piece[len(operator):].strip())))
                    break
            else:
                raise ValueError(f"malformed clause {piece!r} in {text!r}")
    return name, clauses


def select_versions(requirements, available):
    constraints = {}
    for requirement in requirements:
        name, clauses = _split_requirement(requirement)
        constraints.setdefault(name, []).extend(clauses)

    locked = {}
    for name, clauses in constraints.items():
        if name not in available:
            raise ValueError(f"{name} is not available")
        best = None
        for text in available[name]:
            version = _parse_version(text)
            if all(_satisfies(version, operator, bound)
                   for operator, bound in clauses):
                if best is None or _compare(version, best[0]) > 0:
                    best = (version, text)
        if best is None:
            raise ValueError(f"no version of {name} satisfies its constraints")
        locked[name] = best[1]
    return locked
'''


REFERENCES["cicd_containers_packaging_platform-0003"] = r'''
import hashlib

_HEX = set("0123456789abcdef")


def _check(digest):
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError(f"malformed digest {digest!r}")
    body = digest[len("sha256:"):]
    if len(body) != 64 or not set(body) <= _HEX:
        raise ValueError(f"malformed digest {digest!r}")
    return digest


def chain_ids(diff_ids):
    chain = []
    previous = None
    for diff_id in diff_ids:
        _check(diff_id)
        if previous is None:
            previous = diff_id
        else:
            previous = "sha256:" + hashlib.sha256(
                f"{previous} {diff_id}".encode("utf-8")
            ).hexdigest()
        chain.append(previous)
    return chain


def image_id(config):
    if not isinstance(config, (bytes, bytearray)):
        raise ValueError("the image config must be bytes")
    return "sha256:" + hashlib.sha256(bytes(config)).hexdigest()
'''


REFERENCES["cicd_containers_packaging_platform-0004"] = r'''
import hashlib


def _selected(source, context):
    if source == ".":
        return sorted(context)
    prefix = source + "/"
    return sorted(path for path in context
                  if path == source or path.startswith(prefix))


def _keys(instructions, context):
    keys = []
    parent = ""
    for command, argument in instructions:
        blob = f"{parent}\n{command}\n{argument}"
        if command in ("COPY", "ADD"):
            fields = argument.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{command} argument {argument!r} needs a source and a "
                    "destination"
                )
            blob += "\n"
            for path in _selected(fields[0], context):
                content = hashlib.sha256(context[path]).hexdigest()
                blob += f"{path} {content}\n"
        parent = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        keys.append(parent)
    return keys


def first_rebuilt(instructions, previous, current):
    before = _keys(instructions, previous)
    after = _keys(instructions, current)
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            return index
    return None
'''


REFERENCES["cicd_containers_packaging_platform-0005"] = r'''
def evaluate_pipeline(jobs, results):
    order = []
    by_name = {}
    for job in jobs:
        name = job["name"]
        if name in by_name:
            raise ValueError(f"duplicate job {name!r}")
        by_name[name] = job
        order.append(name)

    for name, job in by_name.items():
        for need in job.get("needs", []):
            if need not in by_name:
                raise ValueError(f"{name} needs unknown job {need!r}")

    state = {}
    remaining = list(order)
    while remaining:
        progressed = False
        for name in list(remaining):
            job = by_name[name]
            needs = job.get("needs", [])
            if any(need not in state for need in needs):
                continue
            condition = job.get("condition", "on_success")
            if condition not in ("on_success", "on_failure", "always"):
                raise ValueError(f"{name} has unknown condition {condition!r}")
            effective = [state[need] for need in needs]
            if condition == "on_success":
                runs = all(status == "success" for status in effective)
            elif condition == "on_failure":
                runs = any(status == "failed" for status in effective)
            else:
                runs = True
            if runs:
                if name not in results:
                    raise ValueError(f"{name} runs but has no result")
                outcome = results[name]
                if outcome not in ("pass", "fail"):
                    raise ValueError(f"{name} has unknown result {outcome!r}")
                status = "success" if outcome == "pass" else "failed"
            else:
                status = "skipped"
            state[name] = (
                "success"
                if status == "failed" and job.get("continue_on_error", False)
                else status
            )
            by_name[name] = dict(job, _status=status)
            remaining.remove(name)
            progressed = True
        if not progressed:
            raise ValueError("the job graph has a cycle")

    statuses = {name: by_name[name]["_status"] for name in order}
    conclusion = "success"
    for name in order:
        if (statuses[name] == "failed"
                and not by_name[name].get("continue_on_error", False)):
            conclusion = "failed"
    return {"statuses": statuses, "conclusion": conclusion}
'''


REFERENCES["cicd_containers_packaging_platform-0006"] = r'''
import re


def _translate(pattern):
    parts = pattern.split("/")
    out = []
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
            continue
        piece = ""
        for character in part:
            if character == "*":
                piece += "[^/]*"
            elif character == "?":
                piece += "[^/]"
            else:
                piece += re.escape(character)
        out.append(piece)
        if not last:
            out.append("/")
    return "^" + "".join(out) + "$"


def _matches(pattern, text):
    return re.fullmatch(_translate(pattern), text) is not None


def ignored(paths, patterns):
    directory_rules = []
    file_rules = []
    for line in patterns:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        is_directory = line.endswith("/")
        body = line[:-1] if is_directory else line
        anchored = "/" in body
        if body.startswith("/"):
            body = body[1:]
            anchored = True
        if not body:
            continue
        (directory_rules if is_directory else file_rules).append(
            (negated, anchored, body)
        )

    def verdict(rules, candidates):
        excluded = None
        for negated, anchored, body in rules:
            if any(_matches(body, candidate)
                   for candidate in candidates[anchored]):
                excluded = not negated
        return excluded

    result = []
    for path in paths:
        parts = path.split("/")
        prefixes = ["/".join(parts[:depth]) for depth in range(1, len(parts))]
        components = parts[:-1]
        blocked = verdict(directory_rules,
                          {True: prefixes, False: components})
        if blocked:
            result.append(path)
            continue
        if verdict(file_rules, {True: [path], False: [parts[-1]]}):
            result.append(path)
    return sorted(result)
'''


REFERENCES["cicd_containers_packaging_platform-0007"] = r'''
def plan_deletions(artifacts, policy, referenced, now):
    for key in ("keep_per_branch", "min_age_seconds"):
        if policy.get(key, 0) < 0:
            raise ValueError(f"{key} must not be negative")
    keep_per_branch = policy.get("keep_per_branch", 0)
    keep_tagged = policy.get("keep_tagged", False)
    min_age = policy.get("min_age_seconds", 0)

    seen = set()
    records = []
    for item in artifacts:
        for field in ("id", "branch", "created_unix", "tags"):
            if field not in item:
                raise ValueError(f"artifact is missing {field!r}")
        if item["id"] in seen:
            raise ValueError(f"duplicate artifact id {item['id']!r}")
        seen.add(item["id"])
        records.append(item)

    by_branch = {}
    for item in records:
        by_branch.setdefault(item["branch"], []).append(item)
    protected = set()
    for branch, items in by_branch.items():
        items.sort(key=lambda entry: (entry["created_unix"], entry["id"]),
                   reverse=True)
        for item in items[:keep_per_branch]:
            protected.add(item["id"])

    referenced = set(referenced)
    doomed = []
    for item in records:
        if item["id"] in protected:
            continue
        if keep_tagged and item["tags"]:
            continue
        if item["id"] in referenced:
            continue
        if now - item["created_unix"] < min_age:
            continue
        doomed.append(item["id"])
    return sorted(doomed)
'''


REFERENCES["cicd_containers_packaging_platform-0008"] = r'''
def _parse(filename):
    if not filename.endswith(".whl"):
        raise ValueError(f"{filename!r} is not a wheel")
    stem = filename[:-len(".whl")]
    fields = stem.split("-")
    if len(fields) not in (5, 6):
        raise ValueError(f"{filename!r} has {len(fields)} fields")
    if len(fields) == 6:
        build = fields[2]
        if not build or not build[0].isdigit():
            raise ValueError(f"{filename!r} has a malformed build field")
        build_number = int("".join(
            character for character in build
            if character.isdigit()
        ))
        pythons, abis, platforms = fields[3], fields[4], fields[5]
    else:
        build_number = -1
        pythons, abis, platforms = fields[2], fields[3], fields[4]
    tags = set()
    for python in pythons.split("."):
        for abi in abis.split("."):
            for platform in platforms.split("."):
                tags.add(f"{python}-{abi}-{platform}")
    return build_number, tags


def best_wheel(filenames, supported_tags):
    priority = {}
    for index, tag in enumerate(supported_tags):
        priority.setdefault(tag, index)

    best = None
    for filename in filenames:
        build_number, tags = _parse(filename)
        ranks = [priority[tag] for tag in tags if tag in priority]
        if not ranks:
            continue
        candidate = (min(ranks), -build_number, filename)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]
'''


REFERENCES["frontend_state_ux_accessibility-0001"] = r'''
_HEX = "0123456789abcdefABCDEF"

#: The threshold table is keyed on (level, large_text) so an unknown level
#: and an unknown pairing are the same lookup failure.
_THRESHOLDS = {
    ("AA", False): 4.5,
    ("AA", True): 3.0,
    ("AAA", False): 7.0,
    ("AAA", True): 4.5,
}


def _channels(colour):
    if not isinstance(colour, str) or not colour.startswith("#"):
        raise ValueError(f"not a hex colour: {colour!r}")
    digits = colour[1:]
    if len(digits) == 3:
        digits = "".join(digit * 2 for digit in digits)
    if len(digits) != 6 or any(digit not in _HEX for digit in digits):
        raise ValueError(f"not a hex colour: {colour!r}")
    return [int(digits[index:index + 2], 16) / 255 for index in (0, 2, 4)]


def _luminance(colour):
    linear = [
        channel / 12.92 if channel <= 0.03928
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in _channels(colour)
    ]
    red, green, blue = linear
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground, background):
    first = _luminance(foreground)
    second = _luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def meets_wcag(foreground, background, level, large_text):
    try:
        threshold = _THRESHOLDS[(level, bool(large_text))]
    except (KeyError, TypeError):
        raise ValueError(f"unknown conformance level {level!r}") from None
    return contrast_ratio(foreground, background) >= threshold
'''


REFERENCES["frontend_state_ux_accessibility-0002"] = r'''
_ALWAYS_FOCUSABLE = {"button", "select", "textarea"}


def _focusable_by_default(element):
    tag = element.get("tag")
    if tag in _ALWAYS_FOCUSABLE:
        return True
    if tag == "input":
        return element.get("type") != "hidden"
    if tag == "a":
        return bool(element.get("href"))
    return False


def focus_order(elements):
    sequential = []
    document = []
    for position, element in enumerate(elements):
        if element.get("disabled") or element.get("hidden"):
            continue
        tabindex = element.get("tabindex")
        if tabindex is None:
            if _focusable_by_default(element):
                document.append(element["id"])
        elif tabindex > 0:
            sequential.append((tabindex, position, element["id"]))
        elif tabindex == 0:
            document.append(element["id"])
        # A negative tabindex is reachable only programmatically.
    sequential.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in sequential] + document
'''


REFERENCES["frontend_state_ux_accessibility-0003"] = r'''
import re

_CONTENT_TAGS = {
    "button", "a", "label", "legend", "summary", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6",
}


def _collapse(text):
    return re.sub(r"\s+", " ", text).strip()


def _name(tree, element_id, allow_content):
    node = tree[element_id]

    label = node.get("aria_label")
    if label is not None and _collapse(label):
        return _collapse(label)

    # An empty alt names the image as decorative on purpose, so the key being
    # present ends the computation even though the name is "".
    if node.get("tag") == "img" and "alt" in node:
        return _collapse(node["alt"])

    if allow_content or node.get("tag") in _CONTENT_TAGS:
        parts = []
        own = _collapse(node.get("text", ""))
        if own:
            parts.append(own)
        for child_id in node.get("children") or ():
            if child_id not in tree:
                continue
            child = _name(tree, child_id, True)
            if child:
                parts.append(child)
        joined = " ".join(parts)
        if joined:
            return joined

    title = node.get("title")
    if title is not None and _collapse(title):
        return _collapse(title)
    return ""


def accessible_name(tree, element_id):
    node = tree[element_id]
    for reference in node.get("labelledby") or ():
        # Referenced nodes contribute content whatever their tag, and their
        # own labelledby is not followed, so a cycle cannot recur.
        parts = [
            _name(tree, item, True)
            for item in node["labelledby"] if item in tree
        ]
        joined = _collapse(" ".join(part for part in parts if part))
        if joined:
            return joined
        break
    return _name(tree, element_id, False)
'''


REFERENCES["frontend_state_ux_accessibility-0004"] = r'''
def reconcile(events):
    base = {}
    order = []
    writes = {}
    issued = set()

    for event in events:
        kind = event.get("type")
        if kind == "optimistic":
            identifier = event["id"]
            if identifier in issued:
                raise ValueError(f"duplicate optimistic id {identifier!r}")
            issued.add(identifier)
            writes[identifier] = (event["key"], event["value"])
            order.append(identifier)
        elif kind in ("ack", "reject"):
            identifier = event["id"]
            if identifier not in writes:
                raise KeyError(f"no write is pending for {identifier!r}")
            key, value = writes.pop(identifier)
            order.remove(identifier)
            if kind == "ack":
                base[key] = value
        elif kind == "server":
            base[event["key"]] = event["value"]
        else:
            raise ValueError(f"unknown event type {kind!r}")

    # The visible state is always the confirmed base replayed under the
    # still-pending writes, so a rejection falls back to whatever the server
    # last said rather than to a value captured before the write.
    state = dict(base)
    for identifier in order:
        key, value = writes[identifier]
        state[key] = value
    return {"state": state, "pending": list(order)}
'''


REFERENCES["frontend_state_ux_accessibility-0005"] = r'''
import bisect


class VirtualList:
    def __init__(self, heights):
        heights = list(heights)
        if any(height < 0 for height in heights):
            raise ValueError("heights must be non-negative")
        self._heights = heights
        offsets = [0]
        running = 0
        for height in heights:
            running += height
            offsets.append(running)
        self._offsets = offsets

    def total_height(self):
        return self._offsets[-1]

    def offset_of(self, index):
        if index < 0 or index > len(self._heights):
            raise IndexError(f"row {index} is outside the list")
        return self._offsets[index]

    def _visible(self, index, scroll_top, bottom):
        top = self._offsets[index]
        foot = self._offsets[index + 1]
        if foot == top:
            return scroll_top <= top < bottom
        return top < bottom and foot > scroll_top

    def window(self, scroll_top, viewport_height, overscan):
        if scroll_top < 0:
            raise ValueError("scroll_top must not be negative")
        if viewport_height <= 0:
            raise ValueError("viewport_height must be positive")
        if overscan < 0:
            raise ValueError("overscan must not be negative")

        offsets = self._offsets
        count = len(self._heights)
        bottom = scroll_top + viewport_height
        nothing = (count, count, offsets[-1])
        if count == 0:
            return nothing

        # Binary search, not a scan: the straddling row is the last one whose
        # top is at or above scroll_top.
        start = bisect.bisect_right(offsets, scroll_top, 0, count) - 1
        if start < 0:
            start = 0
        if offsets[start] == scroll_top:
            # Zero-height rows share their neighbour's offset; the first of
            # them is the one the viewport actually starts on.
            start = bisect.bisect_left(offsets, scroll_top, 0, start + 1)
        end = min(count, bisect.bisect_left(offsets, bottom, 0, count + 1))

        if start >= end or not self._visible(start, scroll_top, bottom):
            return nothing
        start = max(0, start - overscan)
        end = min(count, end + overscan)
        return (start, end, offsets[start])
'''


REFERENCES["frontend_state_ux_accessibility-0006"] = r'''
COALESCE_MS = 1000


class EditHistory:
    def __init__(self, initial_text, limit):
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._baseline = initial_text
        self._limit = limit
        self._steps = []
        self._index = 0
        # An apply straight after an undo or a redo is a new step: the user
        # moved through history, so the typing burst is over.
        self._moved = False

    def current(self):
        if self._index == 0:
            return self._baseline
        return self._steps[self._index - 1][0]

    def can_undo(self):
        return self._index > 0

    def can_redo(self):
        return self._index < len(self._steps)

    def apply(self, text, timestamp_ms):
        del self._steps[self._index:]
        coalescing = (
            self._index > 0
            and not self._moved
            and timestamp_ms - self._steps[self._index - 1][1] <= COALESCE_MS
        )
        if coalescing:
            self._steps[self._index - 1] = [text, timestamp_ms]
        else:
            self._steps.append([text, timestamp_ms])
            if len(self._steps) > self._limit:
                self._baseline = self._steps.pop(0)[0]
            self._index = len(self._steps)
        self._moved = False
        return self.current()

    def undo(self):
        if self._index > 0:
            self._index -= 1
        self._moved = True
        return self.current()

    def redo(self):
        if self._index < len(self._steps):
            self._index += 1
        self._moved = True
        return self.current()
'''


REFERENCES["frontend_state_ux_accessibility-0007"] = r'''
def debounce(events, wait, max_wait=None):
    if wait <= 0:
        raise ValueError("wait must be positive")
    if max_wait is not None and max_wait < wait:
        raise ValueError("max_wait must not be shorter than wait")

    previous = None
    for time_ms, _payload in events:
        if previous is not None and time_ms < previous:
            raise ValueError("events must be in non-decreasing time order")
        previous = time_ms

    invocations = []
    deadline = None
    cap = None
    payload = None

    for time_ms, item in events:
        if deadline is not None and time_ms >= deadline:
            # The timer fires first; this call opens a new burst.
            invocations.append((deadline, payload))
            deadline = cap = None
        if deadline is None and max_wait is not None:
            cap = time_ms + max_wait
        deadline = time_ms + wait
        if cap is not None and cap < deadline:
            deadline = cap
        payload = item

    if deadline is not None:
        invocations.append((deadline, payload))
    return invocations
'''


REFERENCES["frontend_state_ux_accessibility-0008"] = r'''
import re

_SUPPORTED = {"en", "ru", "pl", "ar"}


def _language(locale):
    if not isinstance(locale, str):
        raise ValueError(f"not a locale: {locale!r}")
    primary = re.split(r"[-_]", locale)[0].lower()
    if primary not in _SUPPORTED:
        raise ValueError(f"unsupported locale {locale!r}")
    return primary


def _whole_number(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    return n


def plural_category(locale, n):
    language = _language(locale)
    n = _whole_number(n)
    tens, hundreds = n % 10, n % 100

    if language == "en":
        return "one" if n == 1 else "other"
    if language == "ru":
        if tens == 1 and hundreds != 11:
            return "one"
        if tens in (2, 3, 4) and hundreds not in (12, 13, 14):
            return "few"
        return "many"
    if language == "pl":
        if n == 1:
            return "one"
        if tens in (2, 3, 4) and hundreds not in (12, 13, 14):
            return "few"
        return "many"
    if n == 0:
        return "zero"
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    if 3 <= hundreds <= 10:
        return "few"
    if 11 <= hundreds <= 99:
        return "many"
    return "other"


def select_message(locale, n, forms):
    category = plural_category(locale, n)
    if category in forms:
        chosen = forms[category]
    elif "other" in forms:
        chosen = forms["other"]
    else:
        raise KeyError(
            f"no {category!r} form and no 'other' fallback for n={n}")
    return chosen.replace("{n}", str(n))
'''


REFERENCES["frontend_state_ux_accessibility-0009"] = r'''
_KEYS = ("Tab", "Shift+Tab", "Escape")


def run_dialog(focusables, keys, opener):
    order = list(focusables)
    if len(set(order)) != len(order):
        raise ValueError("focusable ids must be unique")
    for key in keys:
        if key not in _KEYS:
            raise ValueError(f"unhandled key {key!r}")

    is_open = True
    index = 0
    focus = order[0] if order else "dialog"

    for key in keys:
        if not is_open:
            continue
        if key == "Escape":
            is_open = False
            focus = opener
        elif not order:
            continue
        elif key == "Tab":
            index = (index + 1) % len(order)
            focus = order[index]
        else:
            index = (index - 1) % len(order)
            focus = order[index]

    return {"focus": focus, "open": is_open}
'''


REFERENCES["frontend_state_ux_accessibility-0010"] = r'''
import re


class FormState:
    def __init__(self, fields, initial):
        self._rules = {name: dict(rule) for name, rule in fields.items()}
        self._initial = dict(initial)
        self._values = dict(initial)
        self._touched = set()

    def _known(self, name):
        if name not in self._rules:
            raise KeyError(f"no such field: {name!r}")

    def change(self, name, value):
        self._known(name)
        self._values[name] = value

    def blur(self, name):
        self._known(name)
        self._touched.add(name)

    def submit(self):
        self._touched = set(self._rules)
        return self.errors()

    def reset(self):
        self._values = dict(self._initial)
        self._touched = set()

    def values(self):
        return dict(self._values)

    def touched(self):
        return set(self._touched)

    def dirty(self):
        return {
            name for name, value in self._values.items()
            if value != self._initial.get(name)
        }

    def _message(self, name):
        rule = self._rules[name]
        value = self._values.get(name, "")
        if rule.get("required") and not value.strip():
            return "required"
        minimum = rule.get("min_length")
        if minimum is not None and len(value) < minimum:
            return "min_length"
        pattern = rule.get("pattern")
        if pattern is not None and not re.fullmatch(pattern, value):
            return "pattern"
        return None

    def errors(self):
        found = {}
        for name in self._rules:
            if name not in self._touched:
                continue
            message = self._message(name)
            if message is not None:
                found[name] = message
        return found
'''


REFERENCES["frontend_state_ux_accessibility-0011"] = r'''
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def _quote(text):
    out = []
    for byte in text.encode("utf-8"):
        char = chr(byte)
        out.append(char if char in _UNRESERVED else f"%{byte:02X}")
    return "".join(out)


def _unquote(text):
    raw = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "%":
            pair = text[index + 1:index + 3]
            if len(pair) != 2:
                raise ValueError(f"truncated escape in {text!r}")
            raw.append(int(pair, 16))
            index += 3
        else:
            # '+' is a literal plus here: the form-encoding rule that turns
            # it into a space does not apply to a percent-encoded query.
            raw.extend(char.encode("utf-8"))
            index += 1
    return raw.decode("utf-8")


def _same_keys(state, defaults):
    if set(state) != set(defaults):
        raise KeyError(
            f"state keys {sorted(state)} are not the default keys "
            f"{sorted(defaults)}")


def encode_state(state, defaults):
    _same_keys(state, defaults)
    parts = []
    for key in sorted(state):
        value = state[key]
        default = defaults[key]
        if value == default:
            continue
        name = _quote(key)
        if isinstance(default, list):
            if not value:
                parts.append(name)
            else:
                parts.extend(f"{name}={_quote(item)}" for item in value)
        elif isinstance(default, bool):
            parts.append(f"{name}={'true' if value else 'false'}")
        elif isinstance(default, int):
            parts.append(f"{name}={value}")
        else:
            parts.append(f"{name}={_quote(value)}")
    return "&".join(parts)


def decode_state(query, defaults):
    state = {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in defaults.items()
    }
    replaced = set()
    for entry in query.split("&"):
        if not entry:
            continue
        raw_key, separator, raw_value = entry.partition("=")
        key = _unquote(raw_key)
        if key not in defaults:
            continue
        default = defaults[key]
        value = _unquote(raw_value)
        if isinstance(default, list):
            if key not in replaced:
                replaced.add(key)
                state[key] = []
            if separator:
                state[key].append(value)
        elif isinstance(default, bool):
            if value not in ("true", "false"):
                raise ValueError(f"{key}={value!r} is not a boolean")
            state[key] = value == "true"
        elif isinstance(default, int):
            state[key] = int(value)
        else:
            state[key] = value
    return state
'''


REFERENCES["frontend_state_ux_accessibility-0012"] = r'''
def _bounds(items, indices, to_index):
    for index in indices:
        if index < 0 or index >= len(items):
            raise IndexError(f"row {index} is outside the list")
    if to_index < 0 or to_index > len(items):
        raise IndexError(f"insertion point {to_index} is outside the list")


def move_item(items, from_index, to_index):
    _bounds(items, (from_index,), to_index)
    rest = list(items)
    row = rest.pop(from_index)
    # Removing the row shifts everything after it, so an insertion point past
    # the row lands one place earlier than the original index suggests --
    # which is why dragging a row one place down is a no-op.
    position = to_index - (1 if from_index < to_index else 0)
    rest.insert(position, row)
    return rest


def move_selection(items, selected, to_index):
    _bounds(items, selected, to_index)
    chosen = sorted(set(selected))
    block = [items[index] for index in chosen]
    taken = set(chosen)
    rest = [row for index, row in enumerate(items) if index not in taken]
    position = to_index - sum(1 for index in chosen if index < to_index)
    return rest[:position] + block + rest[position:]
'''


REFERENCES["polyglot_native_interop-0001"] = r'''
#: (size, natural alignment) per C type name.
_TYPES = {
    "int8": (1, 1), "uint8": (1, 1),
    "int16": (2, 2), "uint16": (2, 2),
    "int32": (4, 4), "uint32": (4, 4),
    "int64": (8, 8), "uint64": (8, 8),
    "float": (4, 4), "double": (8, 8),
    "pointer": (8, 8),
}


def _round_up(value, alignment):
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def layout(fields, pack=None):
    if pack is not None:
        if not isinstance(pack, int) or isinstance(pack, bool) or pack < 1 \
                or pack & (pack - 1):
            raise ValueError(f"pack must be a positive power of two: {pack!r}")

    offsets = {}
    cursor = 0
    struct_alignment = 1
    for name, kind in fields:
        if kind not in _TYPES:
            raise ValueError(f"unknown type {kind!r}")
        if name in offsets:
            raise ValueError(f"duplicate field name {name!r}")
        size, alignment = _TYPES[kind]
        if pack is not None:
            alignment = min(alignment, pack)
        cursor = _round_up(cursor, alignment)
        offsets[name] = cursor
        cursor += size
        struct_alignment = max(struct_alignment, alignment)

    # Trailing padding: without it an array of this struct would put every
    # element after the first at a misaligned address.
    return {
        "offsets": offsets,
        "size": _round_up(cursor, struct_alignment),
        "alignment": struct_alignment,
    }
'''


REFERENCES["polyglot_native_interop-0002"] = r'''
def encode_varint(value):
    if value < -(2 ** 63) or value > 2 ** 64 - 1:
        raise ValueError(f"{value} does not fit a 64-bit varint")
    if value < 0:
        # Protobuf sign-extends to 64 bits, which is why a negative varint
        # is always the maximum ten bytes long.
        value += 1 << 64
    out = bytearray()
    while True:
        group = value & 0x7F
        value >>= 7
        if value:
            out.append(group | 0x80)
        else:
            out.append(group)
            return bytes(out)


def decode_varint(data, offset=0):
    value = 0
    shift = 0
    index = offset
    for _ in range(10):
        if index >= len(data):
            raise ValueError("varint ends past the end of the data")
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return (value, index)
        shift += 7
    raise ValueError("varint is longer than ten bytes")


def encode_zigzag(value):
    return (value << 1) ^ (value >> 63) if value < 0 else value << 1


def decode_zigzag(value):
    return (value >> 1) ^ -(value & 1)
'''


REFERENCES["polyglot_native_interop-0003"] = r'''
def _units(text):
    units = []
    for character in text:
        point = ord(character)
        if point < 0x10000:
            units.append(point)
        else:
            rest = point - 0x10000
            units.append(0xD800 + (rest >> 10))
            units.append(0xDC00 + (rest & 0x3FF))
    return units


def utf16_length(text):
    return len(_units(text))


def encode_utf16_units(text):
    return tuple(_units(text))


def decode_utf16_units(units):
    out = []
    pending = None
    for unit in units:
        if not isinstance(unit, int) or unit < 0 or unit > 0xFFFF:
            raise ValueError(f"{unit!r} is not a UTF-16 code unit")
        if pending is not None:
            if not 0xDC00 <= unit <= 0xDFFF:
                raise ValueError("a high surrogate was not followed by a low")
            out.append(chr(0x10000 + ((pending - 0xD800) << 10)
                           + (unit - 0xDC00)))
            pending = None
        elif 0xD800 <= unit <= 0xDBFF:
            pending = unit
        elif 0xDC00 <= unit <= 0xDFFF:
            raise ValueError("a low surrogate appeared without a high one")
        else:
            out.append(chr(unit))
    if pending is not None:
        raise ValueError("the text ends on an unpaired high surrogate")
    return "".join(out)


def code_point_index(text, unit_index):
    total = utf16_length(text)
    if unit_index < 0 or unit_index > total:
        raise IndexError(f"unit offset {unit_index} is outside 0..{total}")
    consumed = 0
    for position, character in enumerate(text):
        if consumed == unit_index:
            return position
        consumed += 2 if ord(character) >= 0x10000 else 1
        if consumed > unit_index:
            raise ValueError(
                f"unit offset {unit_index} falls inside a surrogate pair")
    return len(text)
'''


REFERENCES["polyglot_native_interop-0004"] = r'''
_WIDTHS = (8, 16, 32, 64)


def _check(bits):
    if bits not in _WIDTHS:
        raise ValueError(f"{bits} is not a C integer width")


def wrap(value, bits, signed):
    _check(bits)
    value &= (1 << bits) - 1
    if signed and value >> (bits - 1):
        value -= 1 << bits
    return value


def _range(bits, signed):
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def add(a, b, bits, signed):
    _check(bits)
    exact = a + b
    low, high = _range(bits, signed)
    return (wrap(exact, bits, signed), not low <= exact <= high)


def mul(a, b, bits, signed):
    _check(bits)
    exact = a * b
    low, high = _range(bits, signed)
    return (wrap(exact, bits, signed), not low <= exact <= high)


def shift_right(value, amount, bits, signed):
    _check(bits)
    if amount < 0 or amount >= bits:
        raise ValueError(f"shift of {amount} is undefined for {bits} bits")
    # Wrapping into the type first is the whole of it. Python's >> is
    # arithmetic on an integer of unbounded width, which is exactly right
    # once the value carries the type's own sign: an unsigned value has
    # wrapped to a non-negative one, so the same operator shifts in zeros.
    value = wrap(value, bits, signed)
    return value >> amount
'''


REFERENCES["polyglot_native_interop-0005"] = r'''
import struct


def float_to_bits(value):
    return struct.unpack(">I", struct.pack(">f", value))[0]


def bits_to_float(bits):
    if not isinstance(bits, int) or isinstance(bits, bool) \
            or bits < 0 or bits > 2 ** 32 - 1:
        raise ValueError(f"{bits!r} is not a 32-bit pattern")
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def half_bits_to_float(bits):
    if not isinstance(bits, int) or isinstance(bits, bool) \
            or bits < 0 or bits > 0xFFFF:
        raise ValueError(f"{bits!r} is not a 16-bit pattern")
    sign = -1.0 if bits >> 15 else 1.0
    exponent = (bits >> 10) & 0x1F
    significand = bits & 0x3FF
    if exponent == 0x1F:
        if significand:
            return float("nan")
        return sign * float("inf")
    if exponent == 0:
        # Subnormal: no implicit leading one, and the exponent is the same
        # as the smallest normal rather than one below it.
        return sign * (2.0 ** -14) * (significand / 1024)
    return sign * (2.0 ** (exponent - 15)) * (1 + significand / 1024)
'''


REFERENCES["polyglot_native_interop-0006"] = r'''
def to_c_string(text, capacity):
    if "\x00" in text:
        raise ValueError("the text contains a NUL and would be truncated")
    encoded = text.encode("utf-8")
    if len(encoded) + 1 > capacity:
        raise ValueError(
            f"{len(encoded)} bytes plus a terminator do not fit {capacity}")
    return encoded + b"\x00" * (capacity - len(encoded))


def from_c_string(buffer):
    end = bytes(buffer).find(b"\x00")
    if end < 0:
        raise ValueError("the buffer has no NUL terminator")
    return bytes(buffer[:end]).decode("utf-8")


def truncate_utf8(data, limit):
    if limit < 0:
        raise ValueError("limit must not be negative")
    if limit >= len(data):
        return bytes(data)
    end = limit
    # Walk back off continuation bytes so the cut never lands inside a
    # sequence; the lead byte goes with them.
    while end > 0 and (data[end] & 0xC0) == 0x80:
        end -= 1
    return bytes(data[:end])
'''


REFERENCES["polyglot_native_interop-0007"] = r'''
EPOCH_DIFFERENCE_SECONDS = 11644473600
TICKS_PER_SECOND = 10_000_000

_UNIX_EPOCH_IN_TICKS = EPOCH_DIFFERENCE_SECONDS * TICKS_PER_SECOND


def filetime_to_unix(ticks):
    if not isinstance(ticks, int) or isinstance(ticks, bool) \
            or ticks < 0 or ticks > 2 ** 64 - 1:
        raise ValueError(f"{ticks!r} is not an unsigned 64-bit FILETIME")
    return (ticks - _UNIX_EPOCH_IN_TICKS) / TICKS_PER_SECOND


def unix_to_filetime(seconds):
    ticks = round(seconds * TICKS_PER_SECOND) + _UNIX_EPOCH_IN_TICKS
    if ticks < 0 or ticks > 2 ** 64 - 1:
        raise ValueError(f"{seconds!r} is outside the FILETIME range")
    return ticks
'''


REFERENCES["polyglot_native_interop-0008"] = r'''
_PRIMITIVES = {
    "Z": "boolean", "B": "byte", "C": "char", "S": "short",
    "I": "int", "J": "long", "F": "float", "D": "double",
}


def _read_type(signature, index, allow_void):
    dimensions = 0
    while index < len(signature) and signature[index] == "[":
        dimensions += 1
        index += 1
    if index >= len(signature):
        raise ValueError(f"truncated type in {signature!r}")
    code = signature[index]
    index += 1
    if code == "V":
        if not allow_void or dimensions:
            raise ValueError("void is only legal as an unarrayed return type")
        name = "void"
    elif code == "L":
        end = signature.find(";", index)
        if end < 0 or end == index:
            raise ValueError(f"unterminated object type in {signature!r}")
        name = signature[index:end].replace("/", ".")
        index = end + 1
    elif code in _PRIMITIVES:
        name = _PRIMITIVES[code]
    else:
        raise ValueError(f"unknown type code {code!r}")
    return name + "[]" * dimensions, index


def parse_signature(signature):
    if not signature or signature[0] != "(":
        raise ValueError("a descriptor starts with '('")
    close = signature.find(")")
    if close < 0:
        raise ValueError("a descriptor needs a ')'")

    parameters = []
    index = 1
    while index < close:
        name, index = _read_type(signature, index, False)
        if index > close:
            raise ValueError("a parameter type runs past the ')'")
        parameters.append(name)

    returned, index = _read_type(signature, close + 1, True)
    if index != len(signature):
        raise ValueError("trailing text after the return type")
    return (tuple(parameters), returned)
'''


REFERENCES["polyglot_native_interop-0009"] = r'''
def _prepare(fields, order):
    if order not in ("lsb", "msb"):
        raise ValueError(f"unknown bit order {order!r}")
    seen = set()
    total = 0
    for name, width in fields:
        if width < 1:
            raise ValueError(f"field {name!r} has width {width}")
        if name in seen:
            raise ValueError(f"duplicate field name {name!r}")
        seen.add(name)
        total += width
    if total > 64:
        raise ValueError(f"{total} bits do not fit a 64-bit container")
    return total


def _positions(fields, total, order):
    # Yields (name, width, shift), where shift is the field's low bit.
    if order == "lsb":
        shift = 0
        for name, width in fields:
            yield name, width, shift
            shift += width
    else:
        shift = total
        for name, width in fields:
            shift -= width
            yield name, width, shift


def pack_bits(fields, values, order):
    total = _prepare(fields, order)
    expected = {name for name, _ in fields}
    if set(values) != expected:
        raise KeyError(
            f"values {sorted(values)} are not the fields {sorted(expected)}")
    packed = 0
    for name, width, shift in _positions(fields, total, order):
        value = values[name]
        if value < 0 or value >> width:
            raise ValueError(f"{name}={value} does not fit {width} bits")
        packed |= value << shift
    return packed


def unpack_bits(fields, packed, order):
    total = _prepare(fields, order)
    if packed < 0 or packed >> total:
        raise ValueError(f"{packed} does not fit {total} bits")
    return {
        name: (packed >> shift) & ((1 << width) - 1)
        for name, width, shift in _positions(fields, total, order)
    }
'''


REFERENCES["polyglot_native_interop-0010"] = r'''
#: Types narrower than int are promoted to int; float is promoted to double.
_PROMOTIONS = {
    "bool": "int",
    "char": "int",
    "signed char": "int",
    "unsigned char": "int",
    "short": "int",
    "unsigned short": "int",
    "float": "double",
}

#: Size of every type that can appear AFTER promotion. Alignment equals size.
_SIZES = {
    "int": 4, "unsigned int": 4,
    "long": 8, "unsigned long": 8, "long long": 8,
    "double": 8, "pointer": 8,
}


def varargs_layout(arguments):
    placed = []
    cursor = 0
    widest = 1
    for ctype, value in arguments:
        if ctype not in _PROMOTIONS and ctype not in _SIZES:
            raise ValueError(f"unknown C type {ctype!r}")
        promoted = _PROMOTIONS.get(ctype, ctype)
        size = _SIZES[promoted]
        remainder = cursor % size
        if remainder:
            cursor += size - remainder
        placed.append((promoted, value, cursor))
        cursor += size
        widest = max(widest, size)
    remainder = cursor % widest
    if remainder:
        cursor += widest - remainder
    return {"arguments": placed, "size": cursor}
'''


REFERENCES["architecture_multifile_integration-0001"] = r'''
def _strongly_connected(nodes, edges):
    # Tarjan, iterative: a recursive walk would blow the stack on a deep
    # dependency graph, which is the shape this is used on.
    index_of = {}
    low = {}
    on_stack = set()
    stack = []
    order = [0]
    found = []

    for root in nodes:
        if root in index_of:
            continue
        work = [(root, iter(edges.get(root, ())))]
        index_of[root] = low[root] = order[0]
        order[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index_of:
                    index_of[child] = low[child] = order[0]
                    order[0] += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(edges.get(child, ()))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index_of[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                found.append(component)
    return found


def audit(imports, layers, order):
    if len(set(order)) != len(order):
        raise ValueError(f"order repeats a layer: {order}")
    rank = {name: position for position, name in enumerate(order)}
    for module, layer in layers.items():
        if layer not in rank:
            raise ValueError(f"module {module!r} has unknown layer {layer!r}")

    modules = set(imports)
    for targets in imports.values():
        modules.update(targets)

    violations = []
    for importer, targets in imports.items():
        if importer not in layers:
            continue
        for imported in targets:
            if imported not in layers:
                continue
            if rank[layers[importer]] < rank[layers[imported]]:
                violations.append((importer, imported))

    cycles = []
    for component in _strongly_connected(sorted(modules), imports):
        if len(component) > 1:
            cycles.append(tuple(sorted(component)))
        else:
            only = component[0]
            if only in imports.get(only, ()):
                cycles.append((only,))

    return {
        "violations": sorted(violations),
        "cycles": sorted(cycles),
        "unlayered": sorted(module for module in modules
                            if module not in layers),
    }
'''


REFERENCES["architecture_multifile_integration-0002"] = r'''
import collections


def _graph(migrations):
    edges = collections.defaultdict(list)
    seen = set()
    for source, target, function in migrations:
        if source == target:
            raise ValueError(f"migration {source!r} -> {target!r} is a no-op")
        if (source, target) in seen:
            raise ValueError(f"two migrations define {source!r} -> {target!r}")
        seen.add((source, target))
        edges[source].append((target, function))
    return edges


def plan(migrations, source, target):
    edges = _graph(migrations)
    if source == target:
        return []

    # Breadth first, so the first time a version is reached is by a shortest
    # path -- and a second arrival at the same depth means the upgrade has
    # two answers rather than one.
    depth = {source: 0}
    came_from = {source: None}
    ambiguous = set()
    queue = collections.deque([source])
    while queue:
        version = queue.popleft()
        for nextt, _function in edges.get(version, ()):
            if nextt not in depth:
                depth[nextt] = depth[version] + 1
                came_from[nextt] = (version, nextt)
                queue.append(nextt)
            elif depth[nextt] == depth[version] + 1 and \
                    came_from[nextt] != (version, nextt):
                ambiguous.add(nextt)

    if target not in depth:
        raise ValueError(f"no migration path from {source!r} to {target!r}")

    steps = []
    cursor = target
    while came_from[cursor] is not None:
        step = came_from[cursor]
        if cursor in ambiguous:
            raise ValueError(
                f"two shortest paths reach {cursor!r}; the upgrade is "
                f"ambiguous")
        steps.append(step)
        cursor = step[0]
    steps.reverse()
    return steps


def migrate(migrations, document, source, target):
    functions = {
        (start, finish): function
        for start, finish, function in migrations
    }
    current = dict(document)
    for step in plan(migrations, source, target):
        current = functions[step](current)
    return current
'''


REFERENCES["architecture_multifile_integration-0003"] = r'''
_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def _coerce(kind, value, source, key):
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        # bool("false") is True, so a string is never truthiness-tested.
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"{source}: {key}={value!r} is not a boolean")
    if kind == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{source}: {key}={value!r} is not an integer") from None
    if kind == "float":
        if isinstance(value, float):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{source}: {key}={value!r} is not a number") from None
    if kind == "list":
        if isinstance(value, list):
            return list(value)
        text = str(value).strip()
        if not text:
            return []
        return [item.strip() for item in text.split(",")]
    if kind == "str":
        return value if isinstance(value, str) else str(value)
    raise ValueError(f"{source}: {key} has unknown type {kind!r}")


def compose(schema, sources):
    values = {key: rule["default"] for key, rule in schema.items()}
    origins = {key: "default" for key in schema}
    ignored = []

    for name, mapping in sources:
        for key, raw in mapping.items():
            if key not in schema:
                ignored.append((name, key))
                continue
            values[key] = _coerce(schema[key]["type"], raw, name, key)
            origins[key] = name

    return {"values": values, "origins": origins, "ignored": ignored}
'''


REFERENCES["architecture_multifile_integration-0004"] = r'''
import base64

from cache_api import CacheMiss
from legacy_store import LegacyError


class LegacyBackedCache:
    def __init__(self, store, clock):
        self._store = store
        self._clock = clock

    @staticmethod
    def _key(key):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{key!r} is not a usable cache key")
        return key

    def get(self, key):
        key = self._key(key)
        try:
            blob, expires_at = self._store.fetch(key)
        except LegacyError:
            # The whole point of the adapter: the caller's except clause is
            # written against CacheMiss and would not catch LegacyError.
            raise CacheMiss(key) from None
        if expires_at is not None and expires_at <= self._clock():
            self._store.drop(key)
            raise CacheMiss(key)
        return base64.b64decode(blob.encode("ascii"))

    def set(self, key, value, ttl=None):
        key = self._key(key)
        blob = base64.b64encode(bytes(value)).decode("ascii")
        expires_at = None if ttl is None else self._clock() + ttl
        try:
            self._store.put(key, blob, expires_at)
        except LegacyError as error:
            raise ValueError(str(error)) from None

    def delete(self, key):
        key = self._key(key)
        try:
            self._store.drop(key)
        except LegacyError:
            pass
'''


REFERENCES["architecture_multifile_integration-0005"] = r'''
from stores import StoreError


class PartialCommitError(Exception):
    def __init__(self, committed, failed, cause=None):
        super().__init__(
            f"{failed} refused to commit after {committed} had committed")
        self.committed = list(committed)
        self.failed = failed
        self.cause = cause


class UnitOfWork:
    def __init__(self, stores):
        self._stores = dict(stores)
        self._order = list(stores)
        self._done = False

    def stage(self, name, key, value):
        if name not in self._stores:
            raise KeyError(f"no store named {name!r}")
        self._stores[name].stage(key, value)

    def commit(self):
        if self._done:
            raise RuntimeError("this unit of work has already been committed")
        self._done = True
        committed = []
        for position, name in enumerate(self._order):
            try:
                self._stores[name].commit()
            except StoreError as error:
                # Nothing here can un-commit what already succeeded, so the
                # remaining resources are rolled back and the caller is told
                # exactly how far the write got.
                for pending in self._order[position + 1:]:
                    self._stores[pending].rollback()
                raise PartialCommitError(committed, name, error) from error
            committed.append(name)
        return committed

    def rollback(self):
        for name in self._order:
            self._stores[name].rollback()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.rollback()
            return False
        self.commit()
        return False
'''


REFERENCES["architecture_multifile_integration-0006"] = r'''
from plugin_api import UnknownEvent


def dispatch(registry, event, payload):
    registered = registry.handlers_for(event)
    if not registered:
        raise UnknownEvent(event)

    # Highest priority first; the enumerate index keeps equal priorities in
    # registration order, which a bare sort on priority alone would not
    # guarantee across implementations.
    ordered = sorted(
        enumerate(registered),
        key=lambda entry: (-entry[1][0], entry[0]),
    )

    results = []
    errors = []
    for _position, (_priority, name, function) in ordered:
        try:
            value = function(payload)
        except Exception as error:
            # A plugin the host does not own must not be able to stop the
            # handlers registered after it.
            errors.append((name, str(error)))
            continue
        if value is not None:
            results.append((name, value))
    return {"results": results, "errors": errors}
'''


REFERENCES["validation_parsing_serialization-0007"] = r'''
def _split(uri):
    fragment = None
    if "#" in uri:
        uri, fragment = uri.split("#", 1)
    query = None
    if "?" in uri:
        uri, query = uri.split("?", 1)
    scheme = None
    for index, char in enumerate(uri):
        if char == ":":
            candidate = uri[:index]
            if candidate and candidate[0].isalpha() and all(
                    part.isalnum() or part in "+-." for part in candidate):
                scheme = candidate
                uri = uri[index + 1:]
            break
        if char in "/?#":
            break
    authority = None
    if uri.startswith("//"):
        rest = uri[2:]
        cut = rest.find("/")
        if cut == -1:
            cut = len(rest)
        authority = rest[:cut]
        uri = rest[cut:]
    return scheme, authority, uri, query, fragment


def _remove_dot_segments(path):
    output = []
    while path:
        if path.startswith("../"):
            path = path[3:]
        elif path.startswith("./"):
            path = path[2:]
        elif path.startswith("/./"):
            path = "/" + path[3:]
        elif path == "/.":
            path = "/"
        elif path.startswith("/../"):
            path = "/" + path[4:]
            if output:
                output.pop()
        elif path == "/..":
            path = "/"
            if output:
                output.pop()
        elif path in (".", ".."):
            path = ""
        else:
            end = path.find("/", 1) if path.startswith("/") else path.find("/")
            if end == -1:
                end = len(path)
            output.append(path[:end])
            path = path[end:]
    return "".join(output)


def _merge(authority, base_path, reference_path):
    if authority is not None and not base_path:
        return "/" + reference_path
    cut = base_path.rfind("/")
    if cut == -1:
        return reference_path
    return base_path[:cut + 1] + reference_path


def resolve_uri(base, reference):
    b_scheme, b_authority, b_path, b_query, _ = _split(base)
    if b_scheme is None:
        raise ValueError("the base URI has no scheme")
    r_scheme, r_authority, r_path, r_query, r_fragment = _split(reference)
    if r_scheme is not None:
        scheme, authority = r_scheme, r_authority
        path, query = _remove_dot_segments(r_path), r_query
    elif r_authority is not None:
        scheme, authority = b_scheme, r_authority
        path, query = _remove_dot_segments(r_path), r_query
    elif r_path == "":
        scheme, authority = b_scheme, b_authority
        path = b_path
        query = r_query if r_query is not None else b_query
    elif r_path.startswith("/"):
        scheme, authority = b_scheme, b_authority
        path, query = _remove_dot_segments(r_path), r_query
    else:
        scheme, authority = b_scheme, b_authority
        path = _remove_dot_segments(_merge(b_authority, b_path, r_path))
        query = r_query
    result = scheme + ":"
    if authority is not None:
        result += "//" + authority
    result += path
    if query is not None:
        result += "?" + query
    if r_fragment is not None:
        result += "#" + r_fragment
    return result
'''


REFERENCES["validation_parsing_serialization-0008"] = r'''
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_HEX_DIGITS = "0123456789ABCDEF"


def encode_component(text):
    out = []
    for byte in text.encode("utf-8"):
        char = chr(byte)
        if char in _UNRESERVED:
            out.append(char)
        else:
            out.append("%" + _HEX_DIGITS[byte >> 4] + _HEX_DIGITS[byte & 0xF])
    return "".join(out)


def _unquote(text):
    raw = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "+":
            raw.append(0x20)
            index += 1
        elif char == "%":
            group = text[index + 1:index + 3]
            if len(group) != 2 or any(
                    digit not in "0123456789abcdefABCDEF" for digit in group):
                raise ValueError(f"malformed percent-escape at {index}")
            raw.append(int(group, 16))
            index += 3
        else:
            raw.extend(char.encode("utf-8"))
            index += 1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("the decoded bytes are not UTF-8") from error


def decode_form(query):
    pairs = []
    for chunk in query.split("&"):
        if not chunk:
            continue
        name, _, value = chunk.partition("=")
        pairs.append((_unquote(name), _unquote(value)))
    return pairs
'''


REFERENCES["validation_parsing_serialization-0009"] = r'''
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_VALUES = {char: index for index, char in enumerate(_ALPHABET)}
_CHARS_KEPT = {1: 2, 2: 4, 3: 5, 4: 7, 5: 8}
_REACHABLE_PADDING = {0, 1, 3, 4, 6}


def b32encode(data):
    out = []
    for start in range(0, len(data), 5):
        block = data[start:start + 5]
        bits = int.from_bytes(block + b"\x00" * (5 - len(block)), "big")
        chars = [_ALPHABET[(bits >> shift) & 0x1F]
                 for shift in range(35, -1, -5)]
        keep = _CHARS_KEPT[len(block)]
        out.append("".join(chars[:keep]) + "=" * (8 - keep))
    return "".join(out)


def b32decode(text):
    if len(text) % 8:
        raise ValueError("a base32 string is a multiple of eight characters")
    out = bytearray()
    for start in range(0, len(text), 8):
        block = text[start:start + 8]
        body = block.rstrip("=")
        padding = len(block) - len(body)
        if padding and start + 8 != len(text):
            raise ValueError("padding may only end the final group")
        if padding not in _REACHABLE_PADDING:
            raise ValueError(f"{padding} padding characters cannot occur")
        bits = 0
        for char in body:
            if char not in _VALUES:
                raise ValueError(f"{char!r} is not a base32 character")
            bits = (bits << 5) | _VALUES[char]
        whole = len(body) * 5 // 8
        spare = len(body) * 5 - whole * 8
        if bits & ((1 << spare) - 1):
            raise ValueError("the bits after the last whole byte are not zero")
        out.extend((bits >> spare).to_bytes(whole, "big"))
    return bytes(out)
'''


REFERENCES["validation_parsing_serialization-0010"] = r'''
def _argument(major, value):
    head = major << 5
    if value < 24:
        return bytes([head | value])
    if value < 0x100:
        return bytes([head | 24, value])
    if value < 0x10000:
        return bytes([head | 25]) + value.to_bytes(2, "big")
    if value < 0x100000000:
        return bytes([head | 26]) + value.to_bytes(4, "big")
    if value < 0x10000000000000000:
        return bytes([head | 27]) + value.to_bytes(8, "big")
    raise ValueError("the argument does not fit in eight bytes")


def cbor_encode(value):
    if value is None:
        return b"\xf6"
    if value is True:
        return b"\xf5"
    if value is False:
        return b"\xf4"
    if isinstance(value, int):
        if 0 <= value < 0x10000000000000000:
            return _argument(0, value)
        if -0x10000000000000000 <= value < 0:
            return _argument(1, -1 - value)
        raise ValueError("the integer is outside the 64-bit CBOR range")
    if isinstance(value, bytes):
        return _argument(2, len(value)) + value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return _argument(3, len(encoded)) + encoded
    if isinstance(value, list):
        return _argument(4, len(value)) + b"".join(
            cbor_encode(item) for item in value)
    if isinstance(value, dict):
        entries = sorted(
            ((cbor_encode(key), cbor_encode(item))
             for key, item in value.items()),
            key=lambda pair: pair[0],
        )
        return _argument(5, len(entries)) + b"".join(
            key + item for key, item in entries)
    raise ValueError(f"{type(value).__name__} has no deterministic encoding")
'''


REFERENCES["validation_parsing_serialization-0011"] = r'''
def _escape(token):
    return token.replace("~", "~0").replace("/", "~1")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches(kind, value):
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return _is_number(value)
    if kind == "integer":
        return _is_number(value) and float(value).is_integer()
    raise ValueError(f"unknown type {kind!r}")


def _equal(left, right):
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, bool):
        return left is right
    if _is_number(left) and _is_number(right):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal(one, other) for one, other in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left)
    if type(left) is not type(right):
        return False
    return left == right


def validate(schema, instance, path=""):
    errors = []
    if "type" in schema:
        kinds = schema["type"]
        if isinstance(kinds, str):
            kinds = [kinds]
        if not any(_matches(kind, instance) for kind in kinds):
            return [{"path": path, "keyword": "type"}]
    if "enum" in schema and not any(
            _equal(instance, option) for option in schema["enum"]):
        errors.append({"path": path, "keyword": "enum"})
    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append({"path": path, "keyword": "minimum"})
        if "exclusiveMaximum" in schema \
                and instance >= schema["exclusiveMaximum"]:
            errors.append({"path": path, "keyword": "exclusiveMaximum"})
    if isinstance(instance, str):
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append({"path": path, "keyword": "maxLength"})
    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errors.append({"path": path, "keyword": "required"})
                break
        properties = schema.get("properties", {})
        for name, member in instance.items():
            child = f"{path}/{_escape(name)}"
            if name in properties:
                errors.extend(validate(properties[name], member, child))
            elif schema.get("additionalProperties", True) is False:
                errors.append(
                    {"path": child, "keyword": "additionalProperties"})
    if isinstance(instance, list):
        if schema.get("uniqueItems") is True:
            for outer in range(len(instance)):
                if any(_equal(instance[outer], instance[inner])
                       for inner in range(outer)):
                    errors.append({"path": path, "keyword": "uniqueItems"})
                    break
        if "items" in schema:
            for index, item in enumerate(instance):
                errors.extend(
                    validate(schema["items"], item, f"{path}/{index}"))
    return sorted(errors, key=lambda error: (error["path"], error["keyword"]))
'''


REFERENCES["validation_parsing_serialization-0012"] = r'''
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
_KEY_TAIL = _LOWER + _DIGITS + "_-.*"
_TOKEN_TAIL = _LOWER + _LOWER.upper() + _DIGITS + "!#$%&'*+-.^_`|~:/"


class _Reader:
    def __init__(self, text):
        self.text = text
        self.index = 0

    def peek(self):
        return self.text[self.index] if self.index < len(self.text) else ""

    def take(self):
        char = self.peek()
        if not char:
            raise ValueError("the field ends unexpectedly")
        self.index += 1
        return char

    def skip_spaces(self):
        while self.peek() == " ":
            self.index += 1


def _parse_key(reader):
    char = reader.peek()
    if not char or (char not in _LOWER and char != "*"):
        raise ValueError(f"{char!r} cannot start a key")
    key = ""
    while reader.peek() and reader.peek() in _KEY_TAIL:
        key += reader.take()
    return key


def _parse_string(reader):
    reader.take()
    out = ""
    while True:
        char = reader.take()
        if char == "\\":
            escaped = reader.take()
            if escaped not in ('"', "\\"):
                raise ValueError("only a quote or a backslash may be escaped")
            out += escaped
        elif char == '"':
            return out
        elif not (" " <= char <= "~"):
            raise ValueError("a string holds printable ASCII only")
        else:
            out += char


def _parse_number(reader):
    digits = ""
    if reader.peek() == "-":
        digits += reader.take()
    if not (reader.peek() and reader.peek() in _DIGITS):
        raise ValueError("a number needs at least one digit")
    while reader.peek() and reader.peek() in _DIGITS:
        digits += reader.take()
    if reader.peek() != ".":
        if len(digits.lstrip("-")) > 15:
            raise ValueError("an integer holds at most fifteen digits")
        return int(digits)
    reader.take()
    fraction = ""
    while reader.peek() and reader.peek() in _DIGITS:
        fraction += reader.take()
    if len(digits.lstrip("-")) > 12:
        raise ValueError("a decimal holds at most twelve integer digits")
    if not 1 <= len(fraction) <= 3:
        raise ValueError("a decimal holds one to three fraction digits")
    return float(digits + "." + fraction)


def _parse_bare_item(reader):
    char = reader.peek()
    if char == "?":
        reader.take()
        flag = reader.take()
        if flag == "0":
            return False
        if flag == "1":
            return True
        raise ValueError("a boolean is ?0 or ?1")
    if char == '"':
        return _parse_string(reader)
    if char == "-" or (char and char in _DIGITS):
        return _parse_number(reader)
    if char and (char in _LOWER + _LOWER.upper() or char == "*"):
        token = ""
        while reader.peek() and reader.peek() in _TOKEN_TAIL:
            token += reader.take()
        return {"token": token}
    raise ValueError(f"{char!r} cannot start a bare item")


def _parse_parameters(reader):
    parameters = {}
    while reader.peek() == ";":
        reader.take()
        reader.skip_spaces()
        name = _parse_key(reader)
        if reader.peek() == "=":
            reader.take()
            parameters[name] = _parse_bare_item(reader)
        else:
            parameters[name] = True
    return parameters


def parse_dictionary(text):
    reader = _Reader(text)
    reader.skip_spaces()
    members = {}
    if not reader.peek():
        return members
    while True:
        key = _parse_key(reader)
        if reader.peek() == "=":
            reader.take()
            value = _parse_bare_item(reader)
        else:
            value = True
        members[key] = (value, _parse_parameters(reader))
        reader.skip_spaces()
        if not reader.peek():
            return members
        if reader.take() != ",":
            raise ValueError("members are separated by a comma")
        reader.skip_spaces()
        if not reader.peek():
            raise ValueError("the dictionary ends with a trailing comma")
'''


REFERENCES["validation_parsing_serialization-0013"] = r'''
_HEX = "0123456789abcdefABCDEF"


def _parse_ipv4(text):
    octets = text.split(".")
    if len(octets) != 4:
        raise ValueError("an embedded IPv4 part needs four octets")
    values = []
    for octet in octets:
        if not octet or any(char not in "0123456789" for char in octet):
            raise ValueError(f"{octet!r} is not a decimal octet")
        if len(octet) > 1 and octet[0] == "0":
            raise ValueError(f"{octet!r} has a leading zero")
        value = int(octet)
        if value > 255:
            raise ValueError(f"{octet!r} is above 255")
        values.append(value)
    return [values[0] << 8 | values[1], values[2] << 8 | values[3]]


def _parse_side(text):
    if text == "":
        return []
    groups = []
    pieces = text.split(":")
    for position, piece in enumerate(pieces):
        if "." in piece:
            if position != len(pieces) - 1:
                raise ValueError("an embedded IPv4 part must come last")
            groups.extend(_parse_ipv4(piece))
            continue
        if not 1 <= len(piece) <= 4:
            raise ValueError(f"{piece!r} is not a one to four digit group")
        if any(char not in _HEX for char in piece):
            raise ValueError(f"{piece!r} is not hexadecimal")
        groups.append(int(piece, 16))
    return groups


def _parse(text):
    if not text:
        raise ValueError("an address cannot be empty")
    if text.count("::") > 1:
        raise ValueError("'::' may appear at most once")
    if "::" in text:
        left, right = text.split("::")
        head = _parse_side(left)
        tail = _parse_side(right)
        if len(head) + len(tail) > 7:
            raise ValueError("'::' must stand for at least one group")
        return head + [0] * (8 - len(head) - len(tail)) + tail
    groups = _parse_side(text)
    if len(groups) != 8:
        raise ValueError("an address without '::' needs eight groups")
    return groups


def canonical_ipv6(text):
    groups = _parse(text)
    best_start, best_length = -1, 0
    index = 0
    while index < 8:
        if groups[index] != 0:
            index += 1
            continue
        run = index
        while run < 8 and groups[run] == 0:
            run += 1
        if run - index > best_length:
            best_start, best_length = index, run - index
        index = run
    parts = [format(value, "x") for value in groups]
    if best_length < 2:
        return ":".join(parts)
    head = ":".join(parts[:best_start])
    tail = ":".join(parts[best_start + best_length:])
    return head + "::" + tail
'''


REFERENCES["validation_parsing_serialization-0014"] = r'''
_BASE = 36
_TMIN = 1
_TMAX = 26
_SKEW = 38
_DAMP = 700
_INITIAL_BIAS = 72
_INITIAL_N = 128
_DELIMITER = "-"
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _adapt(delta, numpoints, firsttime):
    delta = delta // _DAMP if firsttime else delta // 2
    delta += delta // numpoints
    shift = 0
    while delta > ((_BASE - _TMIN) * _TMAX) // 2:
        delta //= _BASE - _TMIN
        shift += _BASE
    return shift + (((_BASE - _TMIN + 1) * delta) // (delta + _SKEW))


def _threshold(k, bias):
    return min(max(k - bias, _TMIN), _TMAX)


def punycode_encode(label):
    output = [char for char in label if ord(char) < 128]
    basic_length = len(output)
    handled = basic_length
    if basic_length:
        output.append(_DELIMITER)
    n = _INITIAL_N
    delta = 0
    bias = _INITIAL_BIAS
    while handled < len(label):
        upcoming = min(ord(char) for char in label if ord(char) >= n)
        delta += (upcoming - n) * (handled + 1)
        n = upcoming
        for char in label:
            code = ord(char)
            if code < n:
                delta += 1
            elif code == n:
                q = delta
                k = _BASE
                while True:
                    t = _threshold(k, bias)
                    if q < t:
                        break
                    output.append(_ALPHABET[t + (q - t) % (_BASE - t)])
                    q = (q - t) // (_BASE - t)
                    k += _BASE
                output.append(_ALPHABET[q])
                bias = _adapt(delta, handled + 1, handled == basic_length)
                delta = 0
                handled += 1
        delta += 1
        n += 1
    return "".join(output)


def punycode_decode(text):
    position = text.rfind(_DELIMITER)
    basic = text[:position] if position >= 0 else ""
    extended = text[position + 1:] if position >= 0 else text
    for char in basic:
        if ord(char) >= 128:
            raise ValueError("the basic part must be ASCII")
    output = list(basic)
    n = _INITIAL_N
    i = 0
    bias = _INITIAL_BIAS
    index = 0
    while index < len(extended):
        previous = i
        weight = 1
        k = _BASE
        while True:
            if index >= len(extended):
                raise ValueError("the extended part ends inside a number")
            digit = _ALPHABET.find(extended[index])
            if digit < 0:
                raise ValueError(f"{extended[index]!r} is not a digit")
            index += 1
            i += digit * weight
            t = _threshold(k, bias)
            if digit < t:
                break
            weight *= _BASE - t
            k += _BASE
        bias = _adapt(i - previous, len(output) + 1, previous == 0)
        n += i // (len(output) + 1)
        i %= len(output) + 1
        output.insert(i, chr(n))
        i += 1
    return "".join(output)
'''


REFERENCES["validation_parsing_serialization-0015"] = r'''
def _disposition_parameters(value):
    parameters = {}
    index = 0
    while index < len(value) and value[index] != ";":
        index += 1
    while index < len(value):
        index += 1
        while index < len(value) and value[index] == " ":
            index += 1
        start = index
        while index < len(value) and value[index] not in "=;":
            index += 1
        name = value[start:index].strip().lower()
        text = ""
        if index < len(value) and value[index] == "=":
            index += 1
            if index < len(value) and value[index] == '"':
                index += 1
                pieces = []
                while index < len(value) and value[index] != '"':
                    if value[index] == "\\":
                        index += 1
                        if index >= len(value):
                            raise ValueError("a trailing backslash")
                    pieces.append(value[index])
                    index += 1
                if index >= len(value):
                    raise ValueError("an unterminated quoted parameter")
                index += 1
                text = "".join(pieces)
            else:
                start = index
                while index < len(value) and value[index] != ";":
                    index += 1
                text = value[start:index].strip()
        if name:
            parameters[name] = text
        while index < len(value) and value[index] != ";":
            index += 1
    return parameters


def _parse_part(raw):
    split = raw.find(b"\r\n\r\n")
    if split < 0:
        raise ValueError("a part's headers are not terminated")
    headers = {}
    for line in raw[:split].split(b"\r\n"):
        if not line:
            continue
        if b":" not in line:
            raise ValueError("a header line has no colon")
        name, _, value = line.partition(b":")
        headers[name.decode("utf-8").strip().lower()] = \
            value.decode("utf-8").strip()
    if "content-disposition" not in headers:
        raise ValueError("a part has no Content-Disposition header")
    parameters = _disposition_parameters(headers["content-disposition"])
    if "name" not in parameters:
        raise ValueError("a part has no name parameter")
    return {
        "name": parameters["name"],
        "filename": parameters.get("filename"),
        "headers": headers,
        "content": raw[split + 4:],
    }


def parse_multipart(body, boundary):
    marker = b"--" + boundary.encode("ascii")
    if body.startswith(marker):
        body = b"\r\n" + body
    delimiter = b"\r\n" + marker
    cursor = body.find(delimiter)
    if cursor < 0:
        raise ValueError("the body holds no boundary delimiter")
    parts = []
    while True:
        cursor += len(delimiter)
        if body[cursor:cursor + 2] == b"--":
            return parts
        if body[cursor:cursor + 2] != b"\r\n":
            raise ValueError("a delimiter is followed by CRLF or '--'")
        cursor += 2
        end = body.find(delimiter, cursor)
        if end < 0:
            raise ValueError("the closing delimiter is missing")
        parts.append(_parse_part(body[cursor:end]))
        cursor = end
'''


REFERENCES["validation_parsing_serialization-0016"] = r'''
_WHITESPACE = " \t\n\r"
_DIGITS = "0123456789"
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}


class _Parser:
    def __init__(self, text):
        self.text = text
        self.index = 0

    def peek(self):
        return self.text[self.index] if self.index < len(self.text) else ""

    def skip(self):
        while self.peek() and self.peek() in _WHITESPACE:
            self.index += 1

    def expect(self, char):
        if self.peek() != char:
            raise ValueError(f"expected {char!r} at offset {self.index}")
        self.index += 1

    def value(self):
        char = self.peek()
        if char == "{":
            return self.object()
        if char == "[":
            return self.array()
        if char == '"':
            return self.string()
        if char == "-" or (char and char in _DIGITS):
            return self.number()
        for literal, result in (("true", True), ("false", False),
                                ("null", None)):
            if self.text.startswith(literal, self.index):
                self.index += len(literal)
                return result
        raise ValueError(f"unexpected input at offset {self.index}")

    def object(self):
        self.expect("{")
        result = {}
        self.skip()
        if self.peek() == "}":
            self.index += 1
            return result
        while True:
            self.skip()
            if self.peek() != '"':
                raise ValueError("an object key must be a string")
            key = self.string()
            self.skip()
            self.expect(":")
            self.skip()
            result[key] = self.value()
            self.skip()
            if self.peek() == ",":
                self.index += 1
                continue
            self.expect("}")
            return result

    def array(self):
        self.expect("[")
        items = []
        self.skip()
        if self.peek() == "]":
            self.index += 1
            return items
        while True:
            self.skip()
            items.append(self.value())
            self.skip()
            if self.peek() == ",":
                self.index += 1
                continue
            self.expect("]")
            return items

    def string(self):
        self.expect('"')
        out = []
        while True:
            if self.index >= len(self.text):
                raise ValueError("an unterminated string")
            char = self.text[self.index]
            self.index += 1
            if char == '"':
                return "".join(out)
            if char == "\\":
                out.append(self.escape())
            elif char < " ":
                raise ValueError("a raw control character in a string")
            else:
                out.append(char)

    def escape(self):
        if self.index >= len(self.text):
            raise ValueError("an escape at the end of the document")
        code = self.text[self.index]
        self.index += 1
        if code in _ESCAPES:
            return _ESCAPES[code]
        if code != "u":
            raise ValueError(f"{code!r} is not a JSON escape")
        first = self.hex4()
        if 0xDC00 <= first <= 0xDFFF:
            raise ValueError("a low surrogate with no high surrogate")
        if 0xD800 <= first <= 0xDBFF:
            if not self.text.startswith("\\u", self.index):
                raise ValueError("a high surrogate with no low surrogate")
            self.index += 2
            second = self.hex4()
            if not 0xDC00 <= second <= 0xDFFF:
                raise ValueError("a high surrogate with no low surrogate")
            return chr(0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00))
        return chr(first)

    def hex4(self):
        group = self.text[self.index:self.index + 4]
        if len(group) != 4 or any(
                char not in "0123456789abcdefABCDEF" for char in group):
            raise ValueError("a unicode escape needs four hexadecimal digits")
        self.index += 4
        return int(group, 16)

    def number(self):
        start = self.index
        floating = False
        if self.peek() == "-":
            self.index += 1
        if self.peek() == "0":
            self.index += 1
        elif self.peek() and self.peek() in "123456789":
            while self.peek() and self.peek() in _DIGITS:
                self.index += 1
        else:
            raise ValueError("a number needs an integer part")
        if self.peek() == ".":
            floating = True
            self.index += 1
            if not (self.peek() and self.peek() in _DIGITS):
                raise ValueError("a fraction needs at least one digit")
            while self.peek() and self.peek() in _DIGITS:
                self.index += 1
        if self.peek() in ("e", "E"):
            floating = True
            self.index += 1
            if self.peek() in ("+", "-"):
                self.index += 1
            if not (self.peek() and self.peek() in _DIGITS):
                raise ValueError("an exponent needs at least one digit")
            while self.peek() and self.peek() in _DIGITS:
                self.index += 1
        raw = self.text[start:self.index]
        return float(raw) if floating else int(raw)


def parse_json(text):
    parser = _Parser(text)
    parser.skip()
    value = parser.value()
    parser.skip()
    if parser.index != len(parser.text):
        raise ValueError("trailing content after the JSON value")
    return value
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
    # A suffix range is resolved as length - n, and n may exceed the
    # representation. Without the floor the first byte goes negative and the
    # caller seeks backwards past the start of the entity.
    "validation_parsing_serialization-0007": (
        "start = max(0, length - suffix)",
        "start = length - suffix",
    ),
    "validation_parsing_serialization-0008": (
        "out.append(0x20)",
        "out.append(0x2b)",
    ),
    "validation_parsing_serialization-0009": (
        "if 0xd800 <= code <= 0xdfff:",
        "if False:",
    ),
    # Merge against the base path by cutting at the FIRST slash rather than
    # the last: every relative reference then resolves against the root.
    "validation_parsing_serialization-0010": (
        'index = base_path.rfind("/")',
        'index = base_path.find("/")',
    ),
    "validation_parsing_serialization-0011": (
        'size_text = line.split(b";", 1)[0].strip()',
        "size_text = line.strip()",
    ),
    # RFC 5952 section 4.2.3 gives the leftmost run to an equal-length tie.
    # `>=` hands it to the rightmost, which is legal syntax and not canonical.
    "validation_parsing_serialization-0012": (
        "if run - index > best_length:",
        "if run - index >= best_length:",
    ),
    "validation_parsing_serialization-0013": (
        "if leftover and bits & ((1 << leftover) - 1):",
        "if False:",
    ),
    "validation_parsing_serialization-0014": (
        "first = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)",
        "first = 0x10000 + ((first - 0xd800) << 10) + second",
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
    # Drop the complement phase of delta debugging, keeping only "does this
    # chunk fail on its own". Two elements that only fail together are then
    # never separated from the noise, and the whole input comes back.
    "testing_debugging_repair_refactoring-0001": (
        "            for index in range(len(chunks)):",
        "            for index in range(0):",
    ),
    # Exclude the probed revision from the remaining range instead of keeping
    # it as the current best candidate: the bisect walks past the commit that
    # introduced the break and blames an innocent one.
    "testing_debugging_repair_refactoring-0002": (
        "            high = middle",
        "            high = middle - 1",
    ),
    # Sign the crash on its full path. Two runs of one defect in different
    # working directories become two defects, which is how a triage queue
    # fills with the same bug.
    "testing_debugging_repair_refactoring-0003": (
        '            basename = path.rsplit("/", 1)[-1]',
        "            basename = path",
    ),
    # Rank lines no failing test ever executed. They score zero and sort last,
    # so the list still looks right -- and every line of the file is now a
    # suspect.
    "testing_debugging_repair_refactoring-0004": (
        "        if executed_failing == 0:",
        "        if executed_failing < 0:",
    ),
    # Rename the references but not the parameters, which is the half-rename
    # that still parses: the signature keeps the old name and the body no
    # longer mentions it.
    "testing_debugging_repair_refactoring-0005": (
        "        def visit_arg(self, node):\n"
        "            if node.arg == old_name:\n"
        "                node.arg = new_name\n"
        "            return node",
        "        def visit_arg(self, node):\n"
        "            return node",
    ),
    # Compare on value alone. True == 1 and 1 == 1.0 in Python, so the two
    # mismatches a type-confusion bug actually produces are reported as a
    # match.
    "testing_debugging_repair_refactoring-0006": (
        "    if type(expected) is not type(actual):",
        "    if False:",
    ),
    # Call one transition flaky. A clean regression -- passing, then failing
    # and staying failed -- is then quarantined, which is how a real break
    # gets muted.
    "testing_debugging_repair_refactoring-0007": (
        "        if transitions >= 2:",
        "        if transitions >= 1:",
    ),
    # Apply hunks without checking the context they were written against, so
    # a patch built from an older revision of the file silently edits the
    # wrong lines.
    "testing_debugging_repair_refactoring-0008": (
        "        if original[start - 1:start - 1 + len(remove)] != remove:",
        "        if False:",
    ),
    # Skip the sort, so the archive's member order follows whatever
    # order the caller happened to iterate in. Every build still
    # produces a valid tarball, and no two agree.
    "cicd_containers_packaging_platform-0001": (
        "    items.sort(key=lambda item: item[0])",
        "    pass",
    ),
    # Take the first satisfying version rather than the highest, which
    # locks a stale release that the constraints do permit.
    "cicd_containers_packaging_platform-0002": (
        "                if best is None or _compare(version, best[0]) > 0:",
        "                if best is None:",
    ),
    # Fold the FIRST diff id instead of the previous chain id. The
    # result is a well-formed list of digests no registry agrees with.
    "cicd_containers_packaging_platform-0003": (
        '                f"{previous} {diff_id}".encode("utf-8")',
        '                f"{diff_ids[0]} {diff_id}".encode("utf-8")',
    ),
    # Key copied files by size instead of content, so an edit that keeps
    # the length hits the cache and ships the previous build's code.
    "cicd_containers_packaging_platform-0004": (
        "                content = hashlib.sha256(context[path]).hexdigest()",
        "                content = str(len(context[path]))",
    ),
    # Ignore continue_on_error, so an advisory job's failure blocks the
    # deploy and fails the pipeline.
    "cicd_containers_packaging_platform-0005": (
        '                if status == "failed" and job.get("continue_on_error", False)',
        "                if False",
    ),
    # Consult the file patterns even under an excluded directory, letting
    # a later negation re-include a file the directory rule removed.
    "cicd_containers_packaging_platform-0006": (
        "        if blocked:\n            result.append(path)\n"
        "            continue",
        "        if False:\n            result.append(path)\n"
        "            continue",
    ),
    # Ignore the referenced set, deleting the artifact a live deployment
    # is pinned to -- discovered during the rollback that needed it.
    "cicd_containers_packaging_platform-0007": (
        '        if item["id"] in referenced:\n            continue',
        "        if False:\n            continue",
    ),
    # Order the build number ascending, so a tie picks the oldest build
    # instead of the newest.
    "cicd_containers_packaging_platform-0008": (
        "        candidate = (min(ranks), -build_number, filename)",
        "        candidate = (min(ranks), build_number, filename)",
    ),
    # Give large text at AAA the AA large-text threshold, so a 3.03 ratio
    # reports conformance at the level that actually needs 4.5.
    "frontend_state_ux_accessibility-0001": (
        '    ("AAA", True): 4.5,',
        '    ("AAA", True): 3.0,',
    ),
    # Make a negative tabindex a tab stop, putting a roving-tabindex element
    # that should only be reachable programmatically into the Tab sequence.
    "frontend_state_ux_accessibility-0002": (
        "        elif tabindex == 0:",
        "        else:",
    ),
    # Require alt to be non-empty rather than present, so an intentionally
    # decorative image falls through and announces its title instead.
    "frontend_state_ux_accessibility-0003": (
        '    if node.get("tag") == "img" and "alt" in node:',
        '    if node.get("tag") == "img" and node.get("alt"):',
    ),
    # Replay the pending writes newest-first, so an older optimistic write
    # wins over the one the user made after it.
    "frontend_state_ux_accessibility-0004": (
        "    for identifier in order:",
        "    for identifier in reversed(order):",
    ),
    # Stop backing up over rows that share the straddling row's offset, so a
    # zero-height row sitting exactly at the viewport top is never rendered.
    "frontend_state_ux_accessibility-0005": (
        "        if offsets[start] == scroll_top:",
        "        if False:",
    ),
    # Coalesce across a history move: the edit a user makes after undoing
    # merges into the step it was supposed to follow, losing that step.
    "frontend_state_ux_accessibility-0006": (
        "            and not self._moved\n",
        "",
    ),
    # Measure the cap from the newest call rather than the burst's first, so
    # max_wait never bounds anything and a steady stream never settles.
    "frontend_state_ux_accessibility-0007": (
        "        if deadline is None and max_wait is not None:",
        "        if max_wait is not None:",
    ),
    # Treat Arabic 'one' as an n % 100 test, matching 'few' and 'many' -- the
    # guess that makes 101 announce a singular noun.
    "frontend_state_ux_accessibility-0008": (
        "    if n == 1:\n        return \"one\"\n    if n == 2:",
        "    if hundreds == 1:\n        return \"one\"\n    if n == 2:",
    ),
    # Keep handling keys after the dialog closed, so Escape restores focus to
    # the opener and a later Tab drags it back inside the closed dialog.
    "frontend_state_ux_accessibility-0009": (
        "        if not is_open:\n            continue\n",
        "",
    ),
    # Match the pattern anywhere in the value instead of against the whole of
    # it, so a valid address with junk in front of it validates.
    "frontend_state_ux_accessibility-0010": (
        "        if pattern is not None and not re.fullmatch(pattern, value):",
        "        if pattern is not None and not re.search(pattern, value):",
    ),
    # Decode '+' as a space, the form-encoding rule that does not apply to a
    # percent-encoded query, corrupting every value containing a plus.
    "frontend_state_ux_accessibility-0011": (
        '            raw.extend(char.encode("utf-8"))',
        '            raw.extend(b" " if char == "+" else char.encode("utf-8"))',
    ),
    # Insert at the raw target index without accounting for the row already
    # removed, so dragging a row one place down swaps it instead of doing
    # nothing.
    "frontend_state_ux_accessibility-0012": (
        "    position = to_index - (1 if from_index < to_index else 0)",
        "    position = to_index",
    ),
    # Report the struct's size as the end of its last member, dropping the
    # trailing padding that keeps element one of an array aligned.
    "polyglot_native_interop-0001": (
        '        "size": _round_up(cursor, struct_alignment),',
        '        "size": cursor,',
    ),
    # Encode a negative varint as a magnitude instead of sign-extending it
    # to 64 bits, producing one byte where the wire format needs ten.
    "polyglot_native_interop-0002": (
        "        value += 1 << 64",
        "        value = -value",
    ),
    # Count code points rather than code units, which is Python's length and
    # not the length the other runtime indexes and slices by.
    "polyglot_native_interop-0003": (
        "def utf16_length(text):\n    return len(_units(text))",
        "def utf16_length(text):\n    return len(text)",
    ),
    # Wrap as signed whatever the type asked for, so an unsigned right shift
    # keeps a sign bit the C type does not have and drags ones down from the
    # top instead of zeros.
    "polyglot_native_interop-0004": (
        "    value = wrap(value, bits, signed)\n    return value >> amount",
        "    value = wrap(value, bits, True)\n    return value >> amount",
    ),
    # Bias the binary16 exponent by 16 instead of 15, halving every normal
    # value while leaving zero, the infinities and the NaNs looking right.
    "polyglot_native_interop-0005": (
        "    return sign * (2.0 ** (exponent - 15)) * (1 + significand / 1024)",
        "    return sign * (2.0 ** (exponent - 16)) * (1 + significand / 1024)",
    ),
    # Cut at the byte limit without backing off the continuation bytes, so
    # the receiver is handed a prefix that does not decode.
    "polyglot_native_interop-0006": (
        "    while end > 0 and (data[end] & 0xC0) == 0x80:",
        "    while False:",
    ),
    # Treat a tick as a microsecond rather than 100 nanoseconds: every
    # converted timestamp is off by a factor of ten.
    "polyglot_native_interop-0007": (
        "TICKS_PER_SECOND = 10_000_000",
        "TICKS_PER_SECOND = 1_000_000",
    ),
    # Leave the class name in JNI internal form, so every object type comes
    # back as java/lang/String and no source-level name matches.
    "polyglot_native_interop-0008": (
        'name = signature[index:end].replace("/", ".")',
        "name = signature[index:end]",
    ),
    # Lay the fields out least-significant-first whatever the order asked
    # for, quietly reversing every header packed most-significant-first.
    "polyglot_native_interop-0009": (
        "    if order == \"lsb\":\n        shift = 0",
        "    if True:\n        shift = 0",
    ),
    # Skip the default argument promotions, so a float occupies four bytes
    # and every argument after it is read from the wrong offset.
    "polyglot_native_interop-0010": (
        "        promoted = _PROMOTIONS.get(ctype, ctype)",
        "        promoted = ctype",
    ),
    # Permit a layer to reach one level up, so the adjacent-layer inversion
    # -- the one that actually happens -- stops being reported.
    "architecture_multifile_integration-0001": (
        "            if rank[layers[importer]] < rank[layers[imported]]:",
        "            if rank[layers[importer]] + 1 < rank[layers[imported]]:",
    ),
    # Stop recording that a version was reached twice at the same depth, so
    # an ambiguous upgrade quietly resolves to whichever path was walked
    # first and the document ends up shaped by an arbitrary choice.
    "architecture_multifile_integration-0002": (
        "                ambiguous.add(nextt)",
        "                pass",
    ),
    # Coerce a boolean by truthiness, so the string "false" -- the value an
    # override exists to supply -- turns the flag on.
    "architecture_multifile_integration-0003": (
        "        if text in _TRUE:\n            return True\n"
        "        if text in _FALSE:\n            return False",
        "        return bool(value)",
    ),
    # Expire strictly after the deadline rather than at it, so an entry is
    # served once more at the instant it was supposed to stop being valid.
    "architecture_multifile_integration-0004": (
        "        if expires_at is not None and expires_at <= self._clock():",
        "        if expires_at is not None and expires_at < self._clock():",
    ),
    # Report the partial commit but leave the resources that never committed
    # holding their staged writes, so the next unit of work commits them.
    "architecture_multifile_integration-0005": (
        "                for pending in self._order[position + 1:]:\n"
        "                    self._stores[pending].rollback()",
        "                pass",
    ),
    # Run the handlers lowest priority first, inverting an ordering the host
    # publishes and plugins rely on.
    "architecture_multifile_integration-0006": (
        "        key=lambda entry: (-entry[1][0], entry[0]),",
        "        key=lambda entry: (entry[1][0], entry[0]),",
    ),
    # Merge a relative reference onto the whole base path instead of onto
    # everything up to its last slash, so every sibling reference resolves
    # one level too deep -- correct only when the base path ends in a slash.
    "validation_parsing_serialization-0007": (
        "    return base_path[:cut + 1] + reference_path",
        "    return base_path + reference_path",
    ),
    # Adopt the unreserved set of JavaScript's encodeURIComponent, which
    # leaves the sub-delimiters that a form parser goes on to treat as
    # syntax.
    "validation_parsing_serialization-0008": (
        '"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"',
        '"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!*\'()"',
    ),
    # Drop the spare bits of a partial group instead of requiring them to be
    # zero, which gives every short value several accepted spellings.
    "validation_parsing_serialization-0009": (
        "        if bits & ((1 << spare) - 1):",
        "        if False:",
    ),
    # Order map keys by encoded length first -- the canonical form of the
    # superseded RFC 7049, which agrees with RFC 8949 on same-length keys
    # and disagrees the moment two key types are mixed.
    "validation_parsing_serialization-0010": (
        "            key=lambda pair: pair[0],",
        "            key=lambda pair: (len(pair[0]), pair[0]),",
    ),
    # Let a Python bool answer to isinstance(int), so true validates as an
    # integer, compares equal to 1 in an enum, and stops being unique.
    "validation_parsing_serialization-0011": (
        "    return isinstance(value, (int, float)) "
        "and not isinstance(value, bool)",
        "    return isinstance(value, (int, float))",
    ),
    # Give a member with no '=' the empty string rather than true, which is
    # the whole point of allowing the shorthand.
    "validation_parsing_serialization-0012": (
        "            value = True",
        "            value = \"\"",
    ),
    # Compress the first run of zero groups rather than the longest, which
    # agrees with RFC 5952 whenever there is only one run.
    "validation_parsing_serialization-0013": (
        "        if run - index > best_length:",
        "        if best_length == 0 and run - index > 0:",
    ),
    # Freeze the bias at its initial value. Every single-character label
    # still encodes correctly, because the bias is adapted after the first
    # code point and never consulted again.
    "validation_parsing_serialization-0014": (
        "                bias = _adapt(delta, handled + 1, "
        "handled == basic_length)",
        "                bias = _INITIAL_BIAS",
    ),
    # Trim the trailing CRLF with rstrip, which also eats blank lines the
    # sender meant to transmit and silently corrupts binary content.
    "validation_parsing_serialization-0015": (
        '        "content": raw[split + 4:],',
        '        "content": raw[split + 4:].rstrip(b"\\r\\n"),',
    ),
    # Funnel every number through float(), losing the int/float distinction
    # the document's own syntax carries.
    "validation_parsing_serialization-0016": (
        "        return float(raw) if floating else int(raw)",
        "        return float(raw)",
    ),
}
