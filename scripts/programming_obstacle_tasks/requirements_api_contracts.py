"""Held-out tasks: requirements and API contracts.

These tasks target the gap between an API that returns plausible values and
one that honours the contract a caller was promised. Each validator drives
behaviour specified by a public standard -- content negotiation, conditional
requests, merge-patch, range requests, web linking -- or by an invariant a
caller depends on, such as a cursor that neither skips nor repeats a row when
the underlying table changes mid-scan.

The prompts state the contract in prose. Reproducing the worked examples is
never sufficient, because every validator also drives cases the prompt does
not enumerate.

Boundaries with sibling families, recorded so the next author finds them by
reading rather than by colliding:

- `validation_parsing_serialization` owns URI reference resolution (0007) and
  SemVer precedence (0002). This family took the resolver half of versioning
  instead -- which versions a range admits, and which one is picked -- and
  declines reference resolution outright.
- `http_apis_authn_appsec` owns Cache-Control *parsing* (0009). Task 0103
  here computes RFC 9111 freshness from an already-parsed header, and its
  one quoting case is incidental rather than the thing measured.
- `capability_overlaps` reports neither of those pairs: they share no public
  symbol, and 0009 cites no document. The scan is a backstop, not a substitute
  for reading the neighbouring family first.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "requirements_api_contracts"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function select_media_type(header, available) "
            "performing RFC 7231 proactive content negotiation. `header` is "
            "the value of an Accept header and `available` is a list of "
            "concrete media types the server can emit, in the server's own "
            "order of preference. Each comma-separated media range may carry "
            "a q parameter; a missing q is 1.0 and q=0 makes that range "
            "unacceptable. A candidate is matched by the most specific range "
            "that covers it -- an exact type/subtype beats type/*, which "
            "beats */* -- and that range alone supplies its q value. Return "
            "the acceptable candidate with the highest q, breaking ties by "
            "the order of `available`, or None when every candidate is "
            "unacceptable. An empty or missing header accepts everything."
        ),
        validator=LOAD_CANDIDATE + require("select_media_type") + r'''
# A missing q is 1.0, and an exact match is returned.
assert select_media_type("text/html", ["text/html"]) == "text/html"
assert select_media_type("", ["text/html"]) == "text/html"
assert select_media_type("*/*", ["text/html", "application/json"]) == "text/html"

# Highest q wins regardless of the order it appears in the header.
assert select_media_type(
    "text/html;q=0.3, application/json;q=0.8",
    ["text/html", "application/json"],
) == "application/json"

# Specificity, not q magnitude, decides WHICH range applies to a candidate.
# text/html is covered by the exact range (q=0.1); application/json only by
# */* (q=0.5), so json wins even though html is named explicitly.
assert select_media_type(
    "*/*;q=0.5, text/html;q=0.1",
    ["text/html", "application/json"],
) == "application/json", "a specific range must supply its own q"

# type/* is more specific than */*.
assert select_media_type(
    "*/*;q=0.2, text/*;q=0.9",
    ["application/json", "text/plain"],
) == "text/plain"

# q=0 excludes, even when the range is the most specific one.
assert select_media_type(
    "*/*;q=0.4, text/html;q=0",
    ["text/html", "application/json"],
) == "application/json"
assert select_media_type("text/html;q=0", ["text/html"]) is None
assert select_media_type("application/xml", ["text/html"]) is None

# Ties fall back to the server's order, not the header's.
assert select_media_type(
    "application/json;q=0.7, text/html;q=0.7",
    ["text/html", "application/json"],
) == "text/html"

# Whitespace and parameter noise must not defeat parsing.
assert select_media_type(
    "  text/html ;  q=0.9 ,application/json;q=1.0  ",
    ["text/html", "application/json"],
) == "application/json"
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function merge_patch(target, patch) applying "
            "RFC 7386 JSON Merge Patch. Return the patched document and do "
            "not mutate either argument. When the patch is an object, each "
            "key whose value is None removes that key from the target "
            "(removing an absent key is not an error), and every other key is "
            "merged recursively when both the target's and the patch's values "
            "are objects, or replaced outright otherwise. When the patch is "
            "not an object, it replaces the target entirely. Arrays are "
            "always replaced, never merged element-wise."
        ),
        validator=LOAD_CANDIDATE + require("merge_patch") + r'''
import copy

assert merge_patch({"a": "b"}, {"a": "c"}) == {"a": "c"}
assert merge_patch({"a": "b"}, {"b": "c"}) == {"a": "b", "b": "c"}

# None deletes; deleting an absent key is a no-op, not an error.
assert merge_patch({"a": "b"}, {"a": None}) == {}
assert merge_patch({"a": "b"}, {"c": None}) == {"a": "b"}

# Objects merge recursively.
assert merge_patch({"a": {"b": "c", "d": "e"}}, {"a": {"b": "x"}}) == {
    "a": {"b": "x", "d": "e"}
}
assert merge_patch({"a": {"b": "c"}}, {"a": {"b": None}}) == {"a": {}}

# A non-object patch replaces wholesale.
assert merge_patch({"a": "b"}, "text") == "text"
assert merge_patch({"a": "b"}, None) is None
assert merge_patch({"a": "b"}, [1, 2]) == [1, 2]

# A non-object target is replaced by an object patch rather than merged into.
assert merge_patch("scalar", {"a": "b"}) == {"a": "b"}
assert merge_patch([1, 2], {"a": "b"}) == {"a": "b"}

# Arrays are replaced, never merged element-wise.
assert merge_patch({"a": [1, 2, 3]}, {"a": [4]}) == {"a": [4]}
assert merge_patch({"a": [{"b": 1}]}, {"a": [{"c": 2}]}) == {"a": [{"c": 2}]}

# Neither argument may be mutated, and the result must not alias the target.
target = {"a": {"b": "c"}, "keep": [1, 2]}
patch = {"a": {"b": "x"}, "new": None}
frozen_target = copy.deepcopy(target)
frozen_patch = copy.deepcopy(patch)
result = merge_patch(target, patch)
assert target == frozen_target, "merge_patch mutated its target"
assert patch == frozen_patch, "merge_patch mutated its patch"
result["a"]["b"] = "mutated"
result["keep"].append(99)
assert target == frozen_target, "result aliases the target's nested values"
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function evaluate_preconditions(method, "
            "headers, current_etag, exists) returning the HTTP status a "
            "conditional request should produce, following RFC 9110. "
            "`headers` maps header names (case-insensitive) to their raw "
            "string values and may contain If-Match and If-None-Match, whose "
            "values are `*` or a comma-separated list of entity tags; a tag "
            "written W/\"x\" is weak. Evaluate If-Match first using strong "
            "comparison, then If-None-Match using weak comparison. Return 412 "
            "when If-Match is present and fails, including when the resource "
            "does not exist. When If-None-Match matches, return 304 for GET "
            "and HEAD and 412 for every other method. Return 200 when no "
            "precondition fails."
        ),
        validator=LOAD_CANDIDATE + require("evaluate_preconditions") + r'''
strong = '"v1"'
other = '"v2"'
weak = 'W/"v1"'

# No preconditions: nothing to fail.
assert evaluate_preconditions("GET", {}, strong, True) == 200

# If-Match uses STRONG comparison, so a weak tag never matches.
assert evaluate_preconditions("PUT", {"If-Match": strong}, strong, True) == 200
assert evaluate_preconditions("PUT", {"If-Match": other}, strong, True) == 412
assert evaluate_preconditions("PUT", {"If-Match": weak}, strong, True) == 412, \
    "If-Match must use strong comparison"
assert evaluate_preconditions("PUT", {"If-Match": strong}, weak, True) == 412

# If-Match: * requires only that the resource exist.
assert evaluate_preconditions("PUT", {"If-Match": "*"}, strong, True) == 200
assert evaluate_preconditions("PUT", {"If-Match": "*"}, None, False) == 412
assert evaluate_preconditions("PUT", {"If-Match": strong}, None, False) == 412

# If-None-Match uses WEAK comparison, so W/"v1" does match "v1".
assert evaluate_preconditions("GET", {"If-None-Match": strong}, strong, True) == 304
assert evaluate_preconditions("GET", {"If-None-Match": weak}, strong, True) == 304, \
    "If-None-Match must use weak comparison"
assert evaluate_preconditions("HEAD", {"If-None-Match": strong}, strong, True) == 304
assert evaluate_preconditions("GET", {"If-None-Match": other}, strong, True) == 200

# A matching If-None-Match on an unsafe method is a conflict, not a 304.
assert evaluate_preconditions("PUT", {"If-None-Match": strong}, strong, True) == 412
assert evaluate_preconditions("POST", {"If-None-Match": "*"}, strong, True) == 412
assert evaluate_preconditions("PUT", {"If-None-Match": "*"}, None, False) == 200

# Header names are case-insensitive; lists may carry whitespace.
assert evaluate_preconditions("GET", {"if-none-match": strong}, strong, True) == 304
assert evaluate_preconditions(
    "PUT", {"If-Match": '"a", "v1" ,"b"'}, strong, True) == 200
assert evaluate_preconditions(
    "GET", {"If-None-Match": '"a","b"'}, strong, True) == 200

# If-Match is evaluated BEFORE If-None-Match: a failing If-Match wins.
assert evaluate_preconditions(
    "GET", {"If-Match": other, "If-None-Match": strong}, strong, True) == 412
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function parse_byte_range(header, length) "
            "resolving an RFC 9110 Range header against a representation of "
            "`length` bytes. Return a list of (first, last) inclusive "
            "half-closed byte offsets, or None when the header is malformed "
            "or does not use the bytes unit, in which case the caller ignores "
            "it and serves the whole representation. Raise ValueError when "
            "the header is well formed but every range is unsatisfiable, "
            "which the caller reports as 416. A range `a-b` clamps `b` to the "
            "last byte; `a-` runs to the end; `-n` requests the final n "
            "bytes. A suffix range of zero bytes is unsatisfiable, and so is "
            "any range whose first offset is past the end."
        ),
        validator=LOAD_CANDIDATE + require("parse_byte_range") + r'''
assert parse_byte_range("bytes=0-499", 10000) == [(0, 499)]
assert parse_byte_range("bytes=0-0", 10000) == [(0, 0)]

# The last byte is length-1, and `last` clamps rather than overflowing.
assert parse_byte_range("bytes=9500-", 10000) == [(9500, 9999)]
assert parse_byte_range("bytes=0-99999", 10000) == [(0, 9999)]
assert parse_byte_range("bytes=0-9999", 10000) == [(0, 9999)]

# Suffix ranges count back from the end and clamp to the whole entity.
assert parse_byte_range("bytes=-500", 10000) == [(9500, 9999)]
assert parse_byte_range("bytes=-20000", 10000) == [(0, 9999)]

# Multiple ranges are preserved in order.
assert parse_byte_range("bytes=0-49, 100-149", 10000) == [(0, 49), (100, 149)]
assert parse_byte_range("bytes=0-49,-50", 10000) == [(0, 49), (9950, 9999)]

# Unsatisfiable: first offset past the end, or a zero-length suffix.
for unsatisfiable in ("bytes=10000-", "bytes=20000-30000", "bytes=-0"):
    try:
        parse_byte_range(unsatisfiable, 10000)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{unsatisfiable!r} should be unsatisfiable")

# A satisfiable range alongside an unsatisfiable one still succeeds.
assert parse_byte_range("bytes=0-49, 10000-", 10000) == [(0, 49)]

# Malformed or non-bytes units are ignored, not errors.
assert parse_byte_range("items=0-499", 10000) is None
assert parse_byte_range("bytes=", 10000) is None
assert parse_byte_range("bytes=abc-def", 10000) is None
assert parse_byte_range("bytes=500-499", 10000) is None, \
    "a backwards range is malformed"
assert parse_byte_range("", 10000) is None

# Whitespace around the ranges is tolerated.
assert parse_byte_range("bytes= 0-49 , 100-149 ", 10000) == [(0, 49), (100, 149)]
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python class CursorPage exposing a stable "
            "keyset-pagination contract over an append-and-delete table. "
            "Construct it with CursorPage(rows) where each row is a dict "
            "with an integer 'id' and a 'name'. The method page(cursor, "
            "limit) returns (items, next_cursor): the first `limit` rows in "
            "ascending id order strictly after the opaque `cursor` (None "
            "starts at the beginning), and a next_cursor to pass back, or "
            "None when the page is the last one. The methods insert(row) and "
            "delete(row_id) mutate the table between pages. A full scan must "
            "never return the same id twice and must never skip a row that "
            "was present for the whole scan, however the table changes "
            "mid-scan. Do not return the cursor as a row offset."
        ),
        validator=LOAD_CANDIDATE + require("CursorPage") + r'''
def scan(table, limit):
    seen, cursor = [], None
    while True:
        items, cursor = table.page(cursor, limit)
        seen.extend(item["id"] for item in items)
        if cursor is None:
            return seen
        assert len(seen) < 1000, "pagination did not terminate"

rows = [{"id": i, "name": f"row-{i}"} for i in (1, 2, 3, 4, 5, 6, 7)]

# A plain scan is ordered, complete and duplicate-free.
table = CursorPage(list(rows))
assert scan(table, 3) == [1, 2, 3, 4, 5, 6, 7]
assert scan(table, 1) == [1, 2, 3, 4, 5, 6, 7]
assert scan(table, 100) == [1, 2, 3, 4, 5, 6, 7]

# The last page reports no next cursor.
table = CursorPage(list(rows))
items, cursor = table.page(None, 7)
assert [item["id"] for item in items] == [1, 2, 3, 4, 5, 6, 7]
assert cursor is None
items, cursor = table.page(None, 3)
assert cursor is not None and [item["id"] for item in items] == [1, 2, 3]

# An empty table terminates immediately.
assert CursorPage([]).page(None, 10) == ([], None)

# DELETING an already-returned row must not shift later rows into the gap.
# An offset cursor skips a row here; a keyset cursor does not.
table = CursorPage(list(rows))
items, cursor = table.page(None, 3)
assert [item["id"] for item in items] == [1, 2, 3]
table.delete(1)
rest = []
while cursor is not None:
    items, cursor = table.page(cursor, 3)
    rest.extend(item["id"] for item in items)
assert rest == [4, 5, 6, 7], f"a delete mid-scan skipped rows: {rest}"

# INSERTING behind the cursor must not re-emit, and ahead of it must appear.
table = CursorPage(list(rows))
items, cursor = table.page(None, 3)
table.insert({"id": 0, "name": "behind"})
table.insert({"id": 8, "name": "ahead"})
rest = []
while cursor is not None:
    items, cursor = table.page(cursor, 3)
    rest.extend(item["id"] for item in items)
assert 0 not in rest, "a row inserted behind the cursor was re-emitted"
assert rest == [4, 5, 6, 7, 8], f"unexpected tail: {rest}"

# Deleting the row the cursor points AT must not lose its successors.
table = CursorPage(list(rows))
items, cursor = table.page(None, 2)
assert [item["id"] for item in items] == [1, 2]
table.delete(2)
rest = []
while cursor is not None:
    items, cursor = table.page(cursor, 2)
    rest.extend(item["id"] for item in items)
assert rest == [3, 4, 5, 6, 7], f"deleting the cursor row lost rows: {rest}"
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function check_compatibility(old, new) that "
            "decides whether a request-schema change is backward compatible "
            "for existing clients. A schema is a dict mapping field names to "
            "a dict with 'type' (a string) and 'required' (a bool), and "
            "optionally 'enum' (a list of permitted values). Return a sorted "
            "list of human-readable breaking-change strings, empty when the "
            "change is compatible. Breaking changes are: removing a field, "
            "adding a field that is required, making an optional field "
            "required, changing a field's type, and removing a value from a "
            "field's enum. Adding an optional field, relaxing a required "
            "field to optional, adding an enum value, and adding an enum to a "
            "field that had none are all compatible."
        ),
        validator=LOAD_CANDIDATE + require("check_compatibility") + r'''
def field(type_="string", required=False, enum=None):
    out = {"type": type_, "required": required}
    if enum is not None:
        out["enum"] = list(enum)
    return out

# Identical schemas are compatible.
base = {"a": field(), "b": field("integer", required=True)}
assert check_compatibility(base, dict(base)) == []

# The result is always a sorted list, so callers can compare it directly.
result = check_compatibility(base, {"b": field("integer", required=True)})
assert isinstance(result, list) and result == sorted(result)
assert len(result) == 1, f"removing a field is one breaking change: {result}"

# Compatible directions produce nothing at all.
assert check_compatibility({"a": field()}, {"a": field(), "b": field()}) == []
assert check_compatibility(
    {"a": field(required=True)}, {"a": field(required=False)}) == []
assert check_compatibility(
    {"a": field(enum=["x"])}, {"a": field(enum=["x", "y"])}) == []
assert check_compatibility({"a": field()}, {"a": field(enum=["x"])}) == []

# Breaking directions each produce exactly one finding.
assert len(check_compatibility({"a": field()}, {})) == 1
assert len(check_compatibility(
    {"a": field()}, {"a": field(), "b": field(required=True)})) == 1
assert len(check_compatibility(
    {"a": field(required=False)}, {"a": field(required=True)})) == 1
assert len(check_compatibility(
    {"a": field("string")}, {"a": field("integer")})) == 1
assert len(check_compatibility(
    {"a": field(enum=["x", "y"])}, {"a": field(enum=["x"])})) == 1

# Adding an optional field is compatible even when other fields break.
findings = check_compatibility(
    {"a": field("string"), "b": field()},
    {"a": field("integer"), "b": field(), "c": field()},
)
assert len(findings) == 1, f"only the type change breaks: {findings}"

# Every breaking change is reported, not just the first.
findings = check_compatibility(
    {"a": field("string"), "b": field(), "c": field(enum=["p", "q"])},
    {"a": field("integer"), "c": field(enum=["p"]), "d": field(required=True)},
)
assert len(findings) == 4, f"expected four findings, got {findings}"

# A field name must appear in its own finding, or the report is unusable.
findings = check_compatibility({"secret_token": field()}, {})
assert any("secret_token" in item for item in findings), findings
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python class IdempotentEndpoint enforcing the "
            "idempotency-key contract used by payment APIs. The method "
            "submit(key, body, handler) takes an idempotency key, a JSON-"
            "serializable request body, and a zero-argument callable that "
            "performs the side effect and returns a response. When the key is "
            "new, run the handler exactly once, store the response, and "
            "return (201, response). When the same key is replayed with an "
            "equal body, return (200, stored_response) without calling the "
            "handler again. When the same key is replayed with a different "
            "body, return (422, None) and do not call the handler. If the "
            "handler raises, the key must not be recorded, so a later retry "
            "may run it again; propagate the exception. Bodies that differ "
            "only in dict key order are equal."
        ),
        validator=LOAD_CANDIDATE + require("IdempotentEndpoint") + r'''
calls = []

def handler(tag="ok"):
    def run():
        calls.append(tag)
        return {"charged": tag}
    return run

endpoint = IdempotentEndpoint()

# First submission runs the handler and reports creation.
status, body = endpoint.submit("k1", {"amount": 10}, handler("a"))
assert (status, body) == (201, {"charged": "a"}), (status, body)
assert calls == ["a"]

# Replay with an equal body returns the STORED response and does not re-run.
status, body = endpoint.submit("k1", {"amount": 10}, handler("b"))
assert status == 200, status
assert body == {"charged": "a"}, f"replay returned a fresh response: {body}"
assert calls == ["a"], "replay re-ran the side effect"

# Key order must not make an equal body look different.
status, body = endpoint.submit("k2", {"amount": 1, "to": "x"}, handler("c"))
assert status == 201
status, body = endpoint.submit("k2", {"to": "x", "amount": 1}, handler("d"))
assert status == 200 and body == {"charged": "c"}, (status, body)
assert calls == ["a", "c"]

# A different body under the same key is a conflict and runs nothing.
status, body = endpoint.submit("k1", {"amount": 99}, handler("e"))
assert (status, body) == (422, None), (status, body)
assert calls == ["a", "c"], "a conflicting replay ran the handler"

# Distinct keys are independent.
status, body = endpoint.submit("k3", {"amount": 10}, handler("f"))
assert (status, body) == (201, {"charged": "f"})
assert calls == ["a", "c", "f"]

# A failing handler must NOT burn the key.
def boom():
    calls.append("boom")
    raise RuntimeError("downstream refused")

try:
    endpoint.submit("k4", {"amount": 5}, boom)
except RuntimeError:
    pass
else:
    raise AssertionError("the handler's exception was swallowed")
assert calls == ["a", "c", "f", "boom"]

status, body = endpoint.submit("k4", {"amount": 5}, handler("g"))
assert (status, body) == (201, {"charged": "g"}), \
    f"a failed attempt poisoned its key: {(status, body)}"

# Nested and list-valued bodies compare structurally.
endpoint.submit("k5", {"items": [{"n": 1}]}, handler("h"))
status, body = endpoint.submit("k5", {"items": [{"n": 1}]}, handler("i"))
assert status == 200 and body == {"charged": "h"}
status, body = endpoint.submit("k5", {"items": [{"n": 2}]}, handler("j"))
assert (status, body) == (422, None)
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function parse_link_header(header) for RFC "
            "8288 web linking. Return a list of dicts in header order, each "
            "with 'uri' plus every parameter as a key. The URI is wrapped in "
            "angle brackets and parameters follow as semicolon-separated "
            "name=value pairs whose values may be quoted or bare. A quoted "
            "value may contain commas, semicolons and angle brackets, and a "
            "backslash escapes the next character inside a quoted string. "
            "When a parameter name repeats within one link, the first "
            "occurrence wins. Return an empty list for an empty header, and "
            "raise ValueError when a link is missing its angle-bracketed URI."
        ),
        validator=LOAD_CANDIDATE + require("parse_link_header") + r'''
assert parse_link_header("") == []
assert parse_link_header("   ") == []

assert parse_link_header('<https://api/next>; rel="next"') == [
    {"uri": "https://api/next", "rel": "next"}
]

# Multiple links, and bare (unquoted) parameter values.
parsed = parse_link_header(
    '<https://api/1>; rel="next", <https://api/2>; rel=prev; title=Second'
)
assert parsed == [
    {"uri": "https://api/1", "rel": "next"},
    {"uri": "https://api/2", "rel": "prev", "title": "Second"},
], parsed

# A comma inside a quoted value does NOT split the links.
parsed = parse_link_header('<https://api/1>; title="a, b"; rel="next"')
assert len(parsed) == 1, f"a quoted comma split the header: {parsed}"
assert parsed[0]["title"] == "a, b"

# Semicolons and angle brackets inside quotes are literal too.
parsed = parse_link_header('<https://api/1>; title="x; y <z>"; rel=next')
assert len(parsed) == 1 and parsed[0]["title"] == "x; y <z>", parsed

# A backslash escapes the next character inside a quoted string.
parsed = parse_link_header(r'<https://api/1>; title="say \"hi\""')
assert parsed[0]["title"] == 'say "hi"', parsed

# The first occurrence of a repeated parameter wins.
parsed = parse_link_header('<https://api/1>; rel=first; rel=second')
assert parsed[0]["rel"] == "first", parsed

# A URI may itself contain commas and semicolons.
parsed = parse_link_header('<https://api/x?a=1,2;b=3>; rel=next')
assert parsed == [{"uri": "https://api/x?a=1,2;b=3", "rel": "next"}], parsed

# Whitespace around delimiters is insignificant.
parsed = parse_link_header('  <https://api/1>  ;  rel = "next"  ')
assert parsed == [{"uri": "https://api/1", "rel": "next"}], parsed

# A link with no angle brackets is malformed.
for malformed in ("https://api/1; rel=next", "<https://api/1>, oops"):
    try:
        parse_link_header(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{malformed!r} should be rejected")
''',
    ),
    # ----------------------------------------------------------------------
    # Ids from 0101 upwards. The sequential block below 0100 is reserved for
    # whoever is extending this family from the front; numbering a
    # concurrently authored batch into its own range means two sessions
    # cannot mint the same id, which `audit_manifest` would only report
    # after both had been written.
    # ----------------------------------------------------------------------
    task(
        f"{FAMILY}-0101", FAMILY,
        prompt=(
            "Implement a Python function apply_patch(document, operations) "
            "applying an RFC 6902 JSON Patch. Return the patched document "
            "and never mutate the argument. Each operation is a dict with "
            "'op' and 'path', where the path is an RFC 6901 JSON Pointer: "
            "the empty string addresses the whole document, and inside a "
            "token '~1' means '/' and '~0' means '~', unescaped in that "
            "order so '~01' is the literal '~1'. Support add, remove, "
            "replace, move, copy and test. 'add' inserts before an array "
            "index rather than overwriting it, accepts an index equal to "
            "the array's length, and accepts '-' to append; on an object it "
            "creates or overwrites the member. 'remove' and 'replace' "
            "require the target to already exist. 'move' and 'copy' read "
            "'from', and moving a location into its own child is an error. "
            "'test' compares structurally, and a boolean is never equal to "
            "a number. The patch is atomic: raise ValueError if any "
            "operation fails, apply none of it, and leave the input "
            "document untouched. Values taken from the patch or copied "
            "within the document must not alias their source."
        ),
        validator=LOAD_CANDIDATE + require("apply_patch") + r'''
import copy

# add creates a member; the input is never mutated.
source = {"a": 1}
frozen = copy.deepcopy(source)
assert apply_patch(source, [{"op": "add", "path": "/b", "value": 2}]) == {
    "a": 1, "b": 2
}
assert source == frozen, "apply_patch mutated its input document"

# add INSERTS into an array rather than replacing the element there.
assert apply_patch(
    {"a": [1, 3]}, [{"op": "add", "path": "/a/1", "value": 2}]
) == {"a": [1, 2, 3]}, "add to an array index must insert, not overwrite"
assert apply_patch(
    {"a": [1, 2]}, [{"op": "add", "path": "/a/-", "value": 3}]
) == {"a": [1, 2, 3]}
assert apply_patch(
    {"a": [1, 2]}, [{"op": "add", "path": "/a/2", "value": 3}]
) == {"a": [1, 2, 3]}, "an index equal to the length appends"

# An index past the end is out of range, and add on an object overwrites.
for bad in ("/a/3", "/a/-1"):
    try:
        apply_patch({"a": [1, 2]}, [{"op": "add", "path": bad, "value": 9}])
    except ValueError:
        pass
    else:
        raise AssertionError(f"add to {bad} should be rejected")
assert apply_patch(
    {"a": 1}, [{"op": "add", "path": "/a", "value": 2}]
) == {"a": 2}

# remove and replace require an existing target.
assert apply_patch({"a": 1, "b": 2}, [{"op": "remove", "path": "/b"}]) == {
    "a": 1
}
for operation in ({"op": "remove", "path": "/nope"},
                  {"op": "replace", "path": "/nope", "value": 1}):
    try:
        apply_patch({"a": 1}, [operation])
    except ValueError:
        pass
    else:
        raise AssertionError(f"{operation['op']} on a missing member passed")

# The empty pointer addresses the whole document.
assert apply_patch({"a": 1}, [{"op": "replace", "path": "", "value": [1]}]) == [1]

# Pointer escaping, including the order the two escapes are undone in.
assert apply_patch(
    {"a/b": 1}, [{"op": "replace", "path": "/a~1b", "value": 2}]
) == {"a/b": 2}
assert apply_patch(
    {"m~n": 1}, [{"op": "replace", "path": "/m~0n", "value": 2}]
) == {"m~n": 2}
assert apply_patch(
    {"~1": 1}, [{"op": "replace", "path": "/~01", "value": 2}]
) == {"~1": 2}, "'~01' unescapes to the literal '~1', not to '/'"

# test compares structurally, and true is not 1.
assert apply_patch(
    {"a": [1, {"b": 2}]},
    [{"op": "test", "path": "/a", "value": [1, {"b": 2}]},
     {"op": "add", "path": "/ok", "value": True}],
) == {"a": [1, {"b": 2}], "ok": True}
try:
    apply_patch({"a": True}, [{"op": "test", "path": "/a", "value": 1}])
except ValueError:
    pass
else:
    raise AssertionError("a boolean must not test equal to a number")

# Atomicity: an earlier operation that succeeded is not kept.
document = {"a": 1}
frozen = copy.deepcopy(document)
try:
    apply_patch(document, [
        {"op": "add", "path": "/b", "value": 2},
        {"op": "test", "path": "/a", "value": 99},
    ])
except ValueError:
    pass
else:
    raise AssertionError("a failing test must reject the whole patch")
assert document == frozen, "a rejected patch left the document modified"

# move relocates, and refuses to move a location into its own child.
assert apply_patch(
    {"a": {"b": 1}, "c": 2}, [{"op": "move", "from": "/c", "path": "/a/c"}]
) == {"a": {"b": 1, "c": 2}}
try:
    apply_patch({"a": {"b": 1}}, [{"op": "move", "from": "/a", "path": "/a/b"}])
except ValueError:
    pass
else:
    raise AssertionError("moving a location into its own child passed")

# copy must not alias the value it copied.
result = apply_patch(
    {"a": {"b": [1]}}, [{"op": "copy", "from": "/a", "path": "/z"}]
)
result["z"]["b"].append(2)
assert result["a"]["b"] == [1], "copy aliased the value it duplicated"

# An unknown operation is rejected rather than ignored.
try:
    apply_patch({"a": 1}, [{"op": "increment", "path": "/a", "value": 1}])
except ValueError:
    pass
else:
    raise AssertionError("an unknown op must be rejected")
''',
    ),
    task(
        f"{FAMILY}-0102", FAMILY,
        prompt=(
            "Implement a Python function expand_uri_template(template, "
            "variables) performing RFC 6570 URI Template expansion up to "
            "level 4. Text outside braces is copied verbatim; an "
            "unterminated brace raises ValueError. An expression may open "
            "with one of the operators + # . / ; ? & which selects the "
            "prefix emitted before a non-empty result, the separator "
            "between values, whether each value is emitted as name=value, "
            "and what an empty value produces. The default and . / ; ? & "
            "operators percent-encode everything outside the unreserved set "
            "A-Z a-z 0-9 - . _ ~ using UTF-8; + and # additionally pass "
            "reserved characters and existing percent-encoded triplets "
            "through unchanged. For ? and & an empty value yields 'name=', "
            "for ; it yields bare 'name'. A variable that is absent, None, "
            "an empty list or an empty dict is undefined and contributes "
            "nothing. A ':n' modifier truncates a string to n characters "
            "before encoding; a '*' modifier explodes a list into separate "
            "values and a dict into its own name=value pairs, while an "
            "unexploded dict flattens to comma-joined keys and values. "
            "Dicts keep insertion order."
        ),
        validator=LOAD_CANDIDATE + require("expand_uri_template") + r'''
variables = {
    "var": "value", "hello": "Hello World!", "half": "50%",
    "who": "fred", "base": "http://example.com/home/",
    "path": "/foo/bar", "x": "1024", "y": "768",
    "empty": "", "empty_keys": {}, "undef": None,
    "list": ["red", "green", "blue"],
    "keys": {"semi": ";", "dot": ".", "comma": ","},
}


def check(template, expected):
    actual = expand_uri_template(template, variables)
    assert actual == expected, f"{template} -> {actual!r}, want {expected!r}"


# Simple expansion encodes everything outside the unreserved set.
check("{var}", "value")
check("{hello}", "Hello%20World%21")
check("{half}", "50%25")
check("{base}index", "http%3A%2F%2Fexample.com%2Fhome%2Findex")

# Reserved expansion passes reserved characters and pct-triplets through.
check("{+hello}", "Hello%20World!")
check("{+half}", "50%25")
check("{+base}index", "http://example.com/home/index")
check("{+path}/here", "/foo/bar/here")
check("X{#hello}", "X#Hello%20World!")

# Prefixed operators emit their prefix only for a defined variable.
check("{.who}", ".fred")
check("{/who}", "/fred")
check("{;who}", ";who=fred")
check("{?who}", "?who=fred")
check("{&who}", "&who=fred")

# Named operators differ in what an empty value produces.
check("{?x,y,empty}", "?x=1024&y=768&empty=")
check("{;x,y,empty}", ";x=1024;y=768;empty")
check("O{empty}X", "OX")
check("O{undef}X", "OX")
check("{?x,y,undef}", "?x=1024&y=768")
check("{?empty_keys}", "")

# The prefix modifier truncates before encoding.
check("{var:3}", "val")
check("{+path:6}/here", "/foo/b/here")

# Lists: explode changes the separator, not just the spelling.
check("{list}", "red,green,blue")
check("{list*}", "red,green,blue")
check("{/list}", "/red,green,blue")
check("{/list*}", "/red/green/blue")
check("{.list*}", ".red.green.blue")
check("{?list}", "?list=red,green,blue")
check("{?list*}", "?list=red&list=green&list=blue")

# Dicts flatten to key,value pairs unless exploded.
check("{keys}", "semi,%3B,dot,.,comma,%2C")
check("{keys*}", "semi=%3B,dot=.,comma=%2C")
check("{+keys}", "semi,;,dot,.,comma,,")
check("{?keys}", "?keys=semi,%3B,dot,.,comma,%2C")
check("{?keys*}", "?semi=%3B&dot=.&comma=%2C")
check("{;keys*}", ";semi=%3B;dot=.;comma=%2C")

# Several variables share one expression's prefix and separator.
check("{x,hello,y}", "1024,Hello%20World%21,768")
check("{+x,hello,y}", "1024,Hello%20World!,768")
check("{#x,hello,y}", "#1024,Hello%20World!,768")
check("{/list*,path:4}", "/red/green/blue/%2Ffoo")

# Non-ASCII is encoded as its UTF-8 octets, not as a single byte.
assert expand_uri_template("{n}", {"n": "é"}) == "%C3%A9"
assert expand_uri_template("{+n}", {"n": "é"}) == "%C3%A9"

try:
    expand_uri_template("{var", variables)
except ValueError:
    pass
else:
    raise AssertionError("an unterminated expression must be rejected")
''',
    ),
    task(
        f"{FAMILY}-0103", FAMILY,
        prompt=(
            "Implement a Python function evaluate_cache(response_headers, "
            "request_headers, age_seconds, shared) deciding how a stored HTTP "
            "response may be reused, following RFC 9111. Header names are "
            "case-insensitive and Cache-Control holds comma-separated "
            "directives, each a bare name or name=value whose value may be "
            "quoted. Return 'unusable' when either message carries no-store, "
            "or when the response is private and the cache is shared. "
            "Otherwise compute a freshness lifetime from the first source "
            "that applies: s-maxage, but only for a shared cache; then "
            "max-age; then Expires minus Date, which may be zero or "
            "negative; then a heuristic of a tenth of the interval from "
            "Last-Modified to Date, capped at 86400 seconds; otherwise zero. "
            "Dates are IMF-fixdate. Return 'revalidate' when either message "
            "carries no-cache. The response is fresh when its lifetime "
            "exceeds age_seconds, and additionally, when the request carries "
            "max-age=n, only while age_seconds is strictly below n, and when "
            "the request carries min-fresh=n, only while the remaining "
            "lifetime is at least n. A fresh response returns 'fresh'. A "
            "stale one returns 'revalidate' if the response carries "
            "must-revalidate, or proxy-revalidate in a shared cache; "
            "otherwise 'stale' if the request carries max-stale with no "
            "value, or max-stale=n and the response is stale by no more than "
            "n seconds; otherwise 'revalidate'."
        ),
        validator=LOAD_CANDIDATE + require("evaluate_cache") + r'''
DATE = "Sun, 06 Nov 1994 08:49:37 GMT"
LATER = "Sun, 06 Nov 1994 08:51:17 GMT"      # DATE + 100 s
EARLIER = "Sun, 06 Nov 1994 08:47:57 GMT"    # DATE - 100 s
LONG_AGO = "Sun, 06 Nov 1994 08:33:00 GMT"   # DATE - 997 s


def check(expected, response, request=None, age=0, shared=False):
    actual = evaluate_cache(response, dict(request or {}), age, shared)
    assert actual == expected, (
        f"{response} + {request} at age {age} (shared={shared}) -> "
        f"{actual!r}, want {expected!r}"
    )


# no-store on either side, and private in a shared cache, are unusable.
check("unusable", {"Cache-Control": "no-store"})
check("unusable", {"Cache-Control": "max-age=100"}, {"Cache-Control": "no-store"})
check("unusable", {"Cache-Control": "private, max-age=100"}, shared=True)
check("fresh", {"Cache-Control": "private, max-age=100"}, age=50, shared=False)

# max-age against the stored age.
check("fresh", {"Cache-Control": "max-age=100"}, age=50)
check("revalidate", {"Cache-Control": "max-age=100"}, age=150)

# s-maxage applies only to a shared cache; a private cache must ignore it.
SHARED = {"Cache-Control": "max-age=100, s-maxage=200"}
check("fresh", SHARED, age=150, shared=True)
# A private cache must ignore s-maxage and fall back to max-age.
check("revalidate", SHARED, age=150, shared=False)

# Expires minus Date, including an Expires that has already passed.
check("fresh", {"Date": DATE, "Expires": LATER}, age=50)
check("revalidate", {"Date": DATE, "Expires": LATER}, age=150)
check("revalidate", {"Date": DATE, "Expires": EARLIER}, age=0)

# A directive outranks Expires, and header names are case-insensitive.
check("revalidate", {"date": DATE, "expires": LATER,
                     "cache-control": "max-age=10"}, age=50)

# The heuristic is a tenth of the Last-Modified interval, and it is capped.
check("fresh", {"Date": DATE, "Last-Modified": LONG_AGO}, age=50)
check("revalidate", {"Date": DATE, "Last-Modified": LONG_AGO}, age=150)
check("revalidate", {"Date": DATE}, age=0)
check("revalidate", {}, age=0)

# no-cache on either side forces revalidation however fresh the response is.
check("revalidate", {"Cache-Control": "no-cache, max-age=100"}, age=0)
check("revalidate", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "no-cache"}, age=0)

# Request directives narrow freshness.
check("revalidate", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-age=30"}, age=50)
check("fresh", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-age=60"}, age=50)
# Request max-age bounds the age strictly, so max-age=0 always revalidates.
check("revalidate", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-age=50"}, age=50)
check("revalidate", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-age=0"}, age=0)
check("revalidate", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "min-fresh=60"}, age=50)
check("fresh", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "min-fresh=50"}, age=50)

# max-stale permits serving a stale response, unless the origin forbade it.
check("stale", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-stale=100"}, age=150)
check("revalidate", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-stale=10"}, age=150)
check("stale", {"Cache-Control": "max-age=100"},
      {"Cache-Control": "max-stale"}, age=99999)
check("revalidate", {"Cache-Control": "max-age=100, must-revalidate"},
      {"Cache-Control": "max-stale=100"}, age=150)
check("stale", {"Cache-Control": "max-age=100, proxy-revalidate"},
      {"Cache-Control": "max-stale=100"}, age=150, shared=False)
check("revalidate", {"Cache-Control": "max-age=100, proxy-revalidate"},
      {"Cache-Control": "max-stale=100"}, age=150, shared=True)

# A quoted directive value parses like an unquoted one.
check("fresh", {"Cache-Control": 'max-age="100"'}, age=50)
''',
    ),
    task(
        f"{FAMILY}-0104", FAMILY,
        prompt=(
            "Implement satisfies(version, constraint) and "
            "max_satisfying(versions, constraint), the two operations a "
            "dependency resolver performs against a Semantic Versioning "
            "2.0.0 range. A constraint is a comma-separated conjunction of "
            "comparators using >=, >, <=, < or =, or a single '^' or '~' "
            "range. '^' takes a full major.minor.patch and permits changes "
            "that do not alter the leftmost non-zero field, so ^1.2.3 admits "
            "below 2.0.0, ^0.2.3 below 0.3.0 and ^0.0.3 below 0.0.4. '~' "
            "takes major.minor or major.minor.patch and admits the rest of "
            "that minor series. A version carrying a prerelease satisfies a "
            "constraint only when some comparator in that constraint names a "
            "prerelease of the same major.minor.patch, so a prerelease never "
            "leaks into a range written for released versions. "
            "max_satisfying returns the highest satisfying entry of "
            "`versions`, or None when none qualifies; among entries that "
            "differ only in build metadata it returns the one that appeared "
            "first. Ordering follows the specification, so a numeric "
            "prerelease field compares as a number. Raise ValueError for a "
            "version or comparator that is not well formed, and for a "
            "constraint with no comparators in it."
        ),
        validator=LOAD_CANDIDATE + require("satisfies") + require(
            "max_satisfying") + r'''
# Caret ranges pivot on the leftmost non-zero field.
assert satisfies("1.2.3", "^1.2.3")
assert satisfies("1.9.9", "^1.2.3")
assert not satisfies("2.0.0", "^1.2.3")
assert not satisfies("1.2.2", "^1.2.3")
assert satisfies("0.2.9", "^0.2.3")
assert not satisfies("0.3.0", "^0.2.3")
assert satisfies("0.0.3", "^0.0.3")
assert not satisfies("0.0.4", "^0.0.3"), "^0.0.x admits only that patch"

# Tilde ranges admit the rest of the minor series, from either spelling.
assert satisfies("1.2.9", "~1.2.3")
assert not satisfies("1.3.0", "~1.2.3")
assert satisfies("1.2.0", "~1.2")
assert not satisfies("1.3.0", "~1.2")

# A conjunction must hold on every comparator it contains.
assert satisfies("1.5.0", ">=1.0.0, <2.0.0")
assert not satisfies("2.0.0", ">=1.0.0, <2.0.0")
assert satisfies("1.0.0", "=1.0.0")
assert not satisfies("1.0.1", "=1.0.0")
assert satisfies("1.0.1", ">1.0.0")
assert not satisfies("1.0.0", ">1.0.0")

# Build metadata never changes whether a version qualifies.
assert satisfies("1.2.3+build.9", "^1.2.3")

# A prerelease qualifies only against a range that named a prerelease of its
# own core version.
assert not satisfies("1.2.4-beta", "^1.2.3"), (
    "a prerelease must not leak into a released-version range"
)
assert not satisfies("3.0.0-alpha", ">=1.0.0")
assert satisfies("1.2.3-beta", ">=1.2.3-alpha, <2.0.0")
assert satisfies("1.2.3", "^1.2.3-alpha")
assert not satisfies("1.2.3-alpha", "^1.2.3-beta")

# Resolution picks the highest qualifying entry, not the last or the first.
assert max_satisfying(
    ["1.2.3", "1.9.9", "1.4.0", "2.0.0"], "^1.2.3"
) == "1.9.9"
assert max_satisfying(["1.0.0", "1.0.10", "1.0.9"], "^1.0.0") == "1.0.10", (
    "1.0.10 outranks 1.0.9 numerically"
)
assert max_satisfying(["2.0.0", "0.9.0"], "^1.0.0") is None
assert max_satisfying([], "^1.0.0") is None

# Among entries differing only in build metadata, the first one wins.
assert max_satisfying(["1.0.0+a", "1.0.0+b"], "^1.0.0") == "1.0.0+a"
assert max_satisfying(["1.0.0", "1.0.0+b"], "^1.0.0") == "1.0.0"

# Resolution across prereleases orders numeric fields as numbers.
assert max_satisfying(
    ["1.0.0-beta.2", "1.0.0-beta.11"], ">=1.0.0-beta.1, <1.0.0"
) == "1.0.0-beta.11", "beta.11 is a later prerelease than beta.2"
assert max_satisfying(
    ["1.0.0-rc.1", "1.0.0"], ">=1.0.0-alpha, <2.0.0"
) == "1.0.0"

for bad_version in ("1.0", "1.0.0.0", "01.0.0", "1.0.0-01", "v1.0.0"):
    try:
        satisfies(bad_version, ">=1.0.0")
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad_version!r} should not parse")

for bad_constraint in (">=1.0", "", "  ", "^1.2", "~1"):
    try:
        satisfies("1.2.3", bad_constraint)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad_constraint!r} should be rejected")
''',
    ),
    task(
        f"{FAMILY}-0105", FAMILY,
        prompt=(
            "Implement a Python function classify_signature_change(old, new) "
            "returning (compatible, reasons) for two callables, where "
            "reasons is the sorted list of distinct findings and compatible "
            "is True exactly when that list is empty. Read both signatures "
            "with the inspect module. Ignore *args and **kwargs except that "
            "adding either is never a finding. Pair the parameters "
            "positionally: the positional-only and positional-or-keyword "
            "parameters of each signature line up by index, and each "
            "keyword-only parameter pairs with the identically named "
            "parameter of the other signature wherever it sits. That "
            "pairing decides rename against removal, rather than guessing. "
            "Report 'renamed-parameter' when a pair's names differ and the "
            "old parameter could be passed by keyword; a positional-only "
            "rename is invisible to callers and is not a finding. Report "
            "'removed-default' when the old parameter had a default and its "
            "pair does not. Report 'narrowed-kind' when a "
            "positional-or-keyword parameter becomes positional-only or "
            "keyword-only; the opposite widening is not a finding. Report "
            "'removed-parameter' for an old parameter with no pair, and "
            "'added-required-parameter' for a new parameter with no pair and "
            "no default. Report 'reordered-parameters' when the names common "
            "to both positional lists do not keep their relative order. A "
            "changed default value is not a finding."
        ),
        validator=LOAD_CANDIDATE + require("classify_signature_change") + r'''
def check(old, new, expected):
    compatible, reasons = classify_signature_change(old, new)
    assert list(reasons) == expected, (
        f"{old.__name__} -> {new.__name__}: {list(reasons)!r}, "
        f"want {expected!r}"
    )
    assert compatible is (not expected), (
        f"{old.__name__} -> {new.__name__}: compatible={compatible!r} "
        f"disagrees with {list(reasons)!r}"
    )


def base(a, b=1):
    return a, b


# Compatible changes a caller cannot observe.
def add_optional(a, b=1, c=2):
    return a, b, c


def add_variadic(a, b=1, *args, **kwargs):
    return a, b


def change_default(a, b=2):
    return a, b


check(base, add_optional, [])
check(base, add_variadic, [])
check(base, change_default, [])


# Widening the kind is compatible; narrowing it is not.
def keyword_only(a, *, b=1):
    return a, b


def positional_only(a, b=1, /):
    return a, b


check(keyword_only, base, [])
check(base, keyword_only, ["narrowed-kind"])
check(base, positional_only, ["narrowed-kind"])


# A positional-only rename cannot be seen by a caller.
def slot(a, /, b=1):
    return a, b


def slot_renamed(x, /, b=1):
    return x, b


check(slot, slot_renamed, [])


# A keyword-callable rename can.
def renamed(a, c=1):
    return a, c


check(base, renamed, ["renamed-parameter"])


# Additions and removals.
def added_required_keyword(a, b=1, *, c):
    return a, b, c


def dropped(a):
    return a


def dropped_default(a, b):
    return a, b


check(base, added_required_keyword, ["added-required-parameter"])
check(dropped, base, [])
check(base, dropped, ["removed-parameter"])
check(base, dropped_default, ["removed-default"])


# A keyword-only parameter that disappears is removed, not renamed.
def keyword_only_gone(a):
    return a


check(keyword_only, keyword_only_gone, ["removed-parameter"])


# Swapping two parameters is both a rename at each position and a reorder.
def three(a, b, c=1):
    return a, b, c


def swapped(b, a, c=1):
    return b, a, c


check(three, swapped, ["renamed-parameter", "reordered-parameters"])


# Several findings at once are reported once each, sorted.
def rebuilt(a, *, z):
    return a, z


compatible, reasons = classify_signature_change(base, rebuilt)
assert compatible is False
assert sorted(set(reasons)) == list(reasons), "reasons must be sorted and unique"
assert set(reasons) == {"added-required-parameter", "removed-parameter"}, (
    f"unexpected findings: {reasons!r}"
)
''',
    ),
    task(
        f"{FAMILY}-0106", FAMILY,
        prompt=(
            "Implement to_minor(amount, exponent) and allocate(total, "
            "weights), the two operations an invoice total has to survive. "
            "to_minor converts a decimal amount to an integer number of "
            "minor units at the given currency exponent, rounding half to "
            "even, so at exponent 2 the amount 1.005 becomes 100 and 1.015 "
            "becomes 102. Accept only a str or a decimal.Decimal; a float "
            "raises TypeError because a binary float cannot hold a decimal "
            "amount exactly. Raise ValueError for text that is not a finite "
            "decimal, and for a negative exponent. allocate splits an "
            "integer number of minor units across a sequence of "
            "non-negative int or Decimal weights, returning a list of "
            "integers of the same length whose sum is exactly total. Each "
            "entry is the floor of its exact share of the magnitude, then "
            "the units still unassigned are handed out one each to the "
            "entries with the largest fractional remainder, ties going to "
            "the lower index, and the sign of total is applied to every "
            "entry. Compute the shares exactly rather than in floating "
            "point. Raise ValueError for an empty sequence, a negative "
            "weight, weights that are all zero, or a total that is not an "
            "integer."
        ),
        validator=LOAD_CANDIDATE + require("to_minor") + require("allocate")
        + r'''
import decimal

# Half-even is the rounding, and it is visible only on an exact tie.
assert to_minor("1.005", 2) == 100, "1.005 rounds to even at two places"
assert to_minor("1.015", 2) == 102, "1.015 rounds up because 1 is odd"
assert to_minor("1.004", 2) == 100
assert to_minor("1.006", 2) == 101
assert to_minor("-1.005", 2) == -100
assert to_minor("1", 2) == 100
assert to_minor("1.2345", 3) == 1234
assert to_minor("2.5", 0) == 2
assert to_minor("3.5", 0) == 4
assert to_minor(decimal.Decimal("0.125"), 2) == 12
assert to_minor("0", 2) == 0

# A binary float is refused rather than silently mis-rounded.
try:
    to_minor(1.005, 2)
except TypeError:
    pass
else:
    raise AssertionError("a float amount must raise TypeError")

for bad in ("abc", "", "NaN", "Infinity"):
    try:
        to_minor(bad, 2)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} should be rejected")
try:
    to_minor("1.00", -1)
except ValueError:
    pass
else:
    raise AssertionError("a negative exponent must be rejected")

# The allocation always reconciles, and the leftover unit goes to the
# largest remainder rather than to the first or the last entry.
assert allocate(100, [1, 1, 1]) == [34, 33, 33]
assert allocate(7, [1, 1, 1]) == [3, 2, 2]
assert allocate(5, [3, 1]) == [4, 1], "the larger remainder takes the unit"
assert allocate(100, [1, 2]) == [33, 67]
assert allocate(0, [1, 1]) == [0, 0]
assert allocate(10, [0, 1]) == [0, 10]
assert allocate(3, [1]) == [3]

# The sign of the total reaches every entry.
assert allocate(-100, [1, 1, 1]) == [-34, -33, -33]
assert allocate(-5, [3, 1]) == [-4, -1]

# Decimal weights are exact, not floating point.
assert allocate(10, [decimal.Decimal("0.5"), decimal.Decimal("0.5")]) == [5, 5]
assert allocate(
    100, [decimal.Decimal("0.7"), decimal.Decimal("0.2"),
          decimal.Decimal("0.1")]
) == [70, 20, 10]

# Reconciliation is the invariant, over a range the examples do not cover.
for total in range(-40, 41):
    for weights in ([1, 1, 1], [1, 2, 3], [5, 1, 1, 1], [0, 3, 1], [7]):
        parts = allocate(total, weights)
        assert len(parts) == len(weights)
        assert sum(parts) == total, (
            f"allocate({total}, {weights}) sums to {sum(parts)}"
        )
        if total >= 0:
            assert all(part >= 0 for part in parts)
        else:
            assert all(part <= 0 for part in parts)

for bad_weights in ([], [-1, 1], [0, 0]):
    try:
        allocate(10, bad_weights)
    except ValueError:
        pass
    else:
        raise AssertionError(f"weights {bad_weights!r} should be rejected")
try:
    allocate("10", [1, 1])
except ValueError:
    pass
else:
    raise AssertionError("a non-integer total should be rejected")
''',
    ),
    task(
        f"{FAMILY}-0107", FAMILY,
        prompt=(
            "Implement a Python function project(document, mask) returning "
            "the partial response a caller asked for with a field mask. A "
            "mask is a comma-separated list of selectors. A selector is a "
            "name made of letters, digits, underscore, hyphen or '*', "
            "optionally followed by a sub-selection written either as "
            "'name/child' for a single child or 'name(one,two)' for several. "
            "A name of '*' selects every member at that level. Projecting an "
            "object keeps only the selected members, in the DOCUMENT's key "
            "order rather than the mask's, and a selected name the document "
            "does not carry is simply absent rather than an error or a null. "
            "Projecting a list applies the sub-selection to each element and "
            "drops any element that is not an object. A selected member that "
            "is a scalar with a sub-selection asked of it is dropped, "
            "because a scalar has no members; an object that matches nothing "
            "is kept as an empty object, because it exists. Naming a member "
            "both bare and narrowed, in either order, keeps all of it -- the "
            "wider selection wins. Return new structures: neither the "
            "argument nor anything reachable from it may be mutated or "
            "aliased by the result. Raise ValueError for a malformed mask, "
            "including an empty selector, a trailing or leading comma, and "
            "unbalanced parentheses."
        ),
        validator=LOAD_CANDIDATE + require("project") + r'''
import copy

# Selection at one level, and the document's key order is what survives.
assert project({"a": 1, "b": 2, "c": 3}, "a,c") == {"a": 1, "c": 3}
ordered = project({"b": 1, "a": 2}, "a,b")
assert list(ordered) == ["b", "a"], "the document's key order, not the mask's"

# Both spellings of a sub-selection.
assert project({"a": {"b": 1, "c": 2}}, "a/b") == {"a": {"b": 1}}
assert project({"a": {"b": 1, "c": 2, "d": 3}}, "a(b,d)") == {
    "a": {"b": 1, "d": 3}
}
assert project({"a": {"b": {"c": 1, "d": 2}}}, "a/b/c") == {
    "a": {"b": {"c": 1}}
}
assert project(
    {"a": {"b": 1, "c": 2}, "z": 9}, "a(b),z"
) == {"a": {"b": 1}, "z": 9}

# A list distributes the sub-selection over its elements.
assert project(
    {"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}]}, "a/b"
) == {"a": [{"b": 1}, {"b": 3}]}
assert project({"a": [{"b": 1}, 2, "three"]}, "a/b") == {"a": [{"b": 1}]}

# '*' selects every member at its level.
assert project({"a": 1, "b": 2}, "*") == {"a": 1, "b": 2}
assert project(
    {"m": {"x": {"k": 1, "z": 2}, "y": {"k": 3, "z": 4}}}, "m/*/k"
) == {"m": {"x": {"k": 1}, "y": {"k": 3}}}

# Absent, scalar and empty cases are three different outcomes.
assert project({"a": 1}, "a,zz") == {"a": 1}
assert project({"a": 1}, "a/b") == {}, "a scalar cannot supply a sub-selection"
assert project({"a": {"z": 1}}, "a/b") == {"a": {}}, (
    "an object that matches nothing is still an object"
)

# The wider of two selections of one name wins, whichever came first.
assert project({"a": {"b": 1, "c": 2}}, "a,a/b") == {"a": {"b": 1, "c": 2}}
assert project({"a": {"b": 1, "c": 2}}, "a/b,a") == {"a": {"b": 1, "c": 2}}

# Nothing reachable from the argument is mutated or aliased.
document = {"a": {"b": [1, 2]}, "c": 3}
frozen = copy.deepcopy(document)
result = project(document, "a")
result["a"]["b"].append(99)
result["a"]["new"] = True
assert document == frozen, "the result aliases the document"

for malformed in ("", "   ", "a,", ",a", "a(", "a)", "a()", "a//b", "a/",
                  "a(b", "a(b))"):
    try:
        project({"a": {"b": 1}}, malformed)
    except ValueError:
        pass
    else:
        raise AssertionError(f"mask {malformed!r} should be rejected")
''',
    ),
    task(
        f"{FAMILY}-0108", FAMILY,
        prompt=(
            "Implement a Python class OperationTracker(clock, "
            "retention_seconds) modelling the operation resource a "
            "long-running API hands back, where clock is a callable "
            "returning seconds as a float. start(name) creates an operation "
            "in state 'running' with progress 0; report(name, progress) "
            "records an integer progress between 0 and 100 that must never "
            "decrease; succeed(name, result) and fail(name, code, message) "
            "make it terminal, and succeed sets progress to 100 whatever it "
            "was. States are 'running', 'succeeded', 'failed' and "
            "'cancelled'. Every one of those methods and poll(name) returns "
            "a snapshot dict carrying state, progress, created and updated, "
            "plus result only when it succeeded and error, a dict of code "
            "and message, only when it failed or was cancelled. A snapshot "
            "is a copy: mutating one, including anything nested inside its "
            "result, must not reach the tracker. A terminal operation is "
            "final -- start, report, succeed and fail on it all raise "
            "ValueError, and so does start on a name that is still held -- "
            "but cancel is idempotent, cancelling a running operation and "
            "returning the existing snapshot unchanged for a terminal one. "
            "A terminal operation is retained for retention_seconds after "
            "its updated time and then poll raises KeyError and the name "
            "becomes free again; a running operation never expires however "
            "old it is. poll of an unknown name raises KeyError."
        ),
        validator=LOAD_CANDIDATE + require("OperationTracker") + r'''
now = [1000.0]


def clock():
    return now[0]


tracker = OperationTracker(clock, 60)

snapshot = tracker.start("job")
assert snapshot["state"] == "running"
assert snapshot["progress"] == 0
assert snapshot["created"] == 1000.0 and snapshot["updated"] == 1000.0
assert "result" not in snapshot and "error" not in snapshot

try:
    tracker.start("job")
except ValueError:
    pass
else:
    raise AssertionError("starting a held name must be rejected")

now[0] = 1010.0
assert tracker.report("job", 30)["progress"] == 30
assert tracker.poll("job")["updated"] == 1010.0
assert tracker.report("job", 30)["progress"] == 30, "equal progress is allowed"

for bad in (20, 101, -1, 5.0, True):
    try:
        tracker.report("job", bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"progress {bad!r} should be rejected")

# Succeeding completes the operation whatever the last progress was.
snapshot = tracker.succeed("job", {"rows": [1, 2]})
assert snapshot["state"] == "succeeded"
assert snapshot["progress"] == 100
assert snapshot["result"] == {"rows": [1, 2]}
assert "error" not in snapshot

# A snapshot is a copy all the way down.
snapshot["state"] = "tampered"
snapshot["result"]["rows"].append(99)
fresh = tracker.poll("job")
assert fresh["state"] == "succeeded"
assert fresh["result"] == {"rows": [1, 2]}, "the snapshot aliased the result"

# Terminal is final for every transition except cancel.
for attempt in (lambda: tracker.report("job", 100),
                lambda: tracker.succeed("job", 1),
                lambda: tracker.fail("job", "E", "m"),
                lambda: tracker.start("job")):
    try:
        attempt()
    except ValueError:
        pass
    else:
        raise AssertionError("a terminal operation accepted a transition")

# Cancel is idempotent, and reports what is already there.
assert tracker.cancel("job")["state"] == "succeeded", (
    "cancelling a finished operation is a no-op, not an error"
)

# Failure carries an error and no result.
tracker.start("broken")
snapshot = tracker.fail("broken", "E_IO", "disk full")
assert snapshot["state"] == "failed"
assert snapshot["error"] == {"code": "E_IO", "message": "disk full"}
assert "result" not in snapshot

# Cancelling a running operation is a terminal transition of its own.
tracker.start("doomed")
snapshot = tracker.cancel("doomed")
assert snapshot["state"] == "cancelled"
assert snapshot["error"]["code"] == "cancelled"
assert tracker.cancel("doomed")["state"] == "cancelled"

try:
    tracker.poll("never-started")
except KeyError:
    pass
else:
    raise AssertionError("polling an unknown name must raise KeyError")

# Retention runs from the terminal update, and frees the name.
now[0] = 1010.0 + 60
assert tracker.poll("job")["state"] == "succeeded", "retention is exclusive"
now[0] = 1010.0 + 61
try:
    tracker.poll("job")
except KeyError:
    pass
else:
    raise AssertionError("a retained operation must expire")
assert tracker.start("job")["state"] == "running", (
    "an expired name is free again"
)

# A running operation never expires, however long it sits.
patient = OperationTracker(clock, 10)
patient.start("long")
now[0] += 1_000_000
assert patient.poll("long")["state"] == "running"
assert patient.report("long", 5)["progress"] == 5
''',
    ),
]
