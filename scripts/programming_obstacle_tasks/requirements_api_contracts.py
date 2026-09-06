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
            "Implement a Python function resolve_reference(base, reference) "
            "resolving a URI reference against an absolute base URI using "
            "the strict algorithm of RFC 3986 section 5.2, and returning "
            "the recomposed target. Split each input into scheme, "
            "authority, path, query and fragment, remembering that a "
            "component which is absent differs from one that is present and "
            "empty -- a reference of '?' carries an empty query and must "
            "not inherit the base's. The base's own fragment is never "
            "inherited. A relative path is merged against the base by "
            "replacing everything after the base path's last '/', then the "
            "result has its dot segments removed: '.' and '..' are resolved "
            "against the accumulated output, '..' at the root is discarded "
            "rather than escaping it, and a segment such as '..g' or 'g.' "
            "is an ordinary name. Dot segments inside a query or fragment "
            "are left alone. Raise ValueError when the base has no scheme."
        ),
        validator=LOAD_CANDIDATE + require("resolve_reference") + r'''
BASE = "http://a/b/c/d;p?q"

# The normal examples of RFC 3986 section 5.4.1.
NORMAL = {
    "g:h": "g:h",
    "g": "http://a/b/c/g",
    "./g": "http://a/b/c/g",
    "g/": "http://a/b/c/g/",
    "/g": "http://a/g",
    "//g": "http://g",
    "?y": "http://a/b/c/d;p?y",
    "g?y": "http://a/b/c/g?y",
    "#s": "http://a/b/c/d;p?q#s",
    "g#s": "http://a/b/c/g#s",
    "g?y#s": "http://a/b/c/g?y#s",
    ";x": "http://a/b/c/;x",
    "g;x": "http://a/b/c/g;x",
    "g;x?y#s": "http://a/b/c/g;x?y#s",
    "": "http://a/b/c/d;p?q",
    ".": "http://a/b/c/",
    "./": "http://a/b/c/",
    "..": "http://a/b/",
    "../": "http://a/b/",
    "../g": "http://a/b/g",
    "../..": "http://a/",
    "../../": "http://a/",
    "../../g": "http://a/g",
}

# The abnormal examples of section 5.4.2, which is where a resolver that
# only handles the easy cases diverges.
ABNORMAL = {
    "../../../g": "http://a/g",
    "../../../../g": "http://a/g",
    "/./g": "http://a/g",
    "/../g": "http://a/g",
    "g.": "http://a/b/c/g.",
    ".g": "http://a/b/c/.g",
    "g..": "http://a/b/c/g..",
    "..g": "http://a/b/c/..g",
    "./../g": "http://a/b/g",
    "./g/.": "http://a/b/c/g/",
    "g/./h": "http://a/b/c/g/h",
    "g/../h": "http://a/b/c/h",
    "g;x=1/./y": "http://a/b/c/g;x=1/y",
    "g;x=1/../y": "http://a/b/c/y",
    "g?y/./x": "http://a/b/c/g?y/./x",
    "g?y/../x": "http://a/b/c/g?y/../x",
    "g#s/./x": "http://a/b/c/g#s/./x",
    "g#s/../x": "http://a/b/c/g#s/../x",
}

for cases in (NORMAL, ABNORMAL):
    for reference, expected in cases.items():
        actual = resolve_reference(BASE, reference)
        assert actual == expected, (
            f"{reference!r} -> {actual!r}, want {expected!r}"
        )

# An empty component is not an absent one.
assert resolve_reference(BASE, "?") == "http://a/b/c/d;p?"
assert resolve_reference(BASE, "#") == "http://a/b/c/d;p?q#"
assert resolve_reference("http://a/b/c/d;p?q#f", "g") == "http://a/b/c/g", (
    "the base's fragment must never be inherited"
)

# A base whose path is empty still merges to an absolute path.
assert resolve_reference("http://a", "g") == "http://a/g"

try:
    resolve_reference("/not/absolute", "g")
except ValueError:
    pass
else:
    raise AssertionError("a base without a scheme must be rejected")
''',
    ),
    task(
        f"{FAMILY}-0104", FAMILY,
        prompt=(
            "Implement compare_versions(left, right) and satisfies(version, "
            "constraint) for Semantic Versioning 2.0.0. compare_versions "
            "returns -1, 0 or 1. Build metadata after '+' never affects "
            "precedence. A version with a prerelease ranks below the same "
            "core version without one; prerelease identifiers are compared "
            "field by field, numerically when both fields are all digits, "
            "a numeric field ranking below an alphanumeric one, and a "
            "longer identifier list ranking above a shorter one that is "
            "otherwise identical. Reject anything that is not "
            "major.minor.patch with no leading zeroes -- including a "
            "numeric prerelease field with a leading zero -- by raising "
            "ValueError. satisfies takes a comma-separated conjunction of "
            "comparators using >=, >, <=, < or =, or a single '^' or '~' "
            "range. '^' needs a full major.minor.patch and permits changes "
            "that do not alter the leftmost non-zero field, so ^1.2.3 "
            "allows below 2.0.0, ^0.2.3 below 0.3.0 and ^0.0.3 below 0.0.4. "
            "'~' accepts major.minor or major.minor.patch and allows the "
            "rest of that minor series. A prerelease version satisfies a "
            "constraint only when some comparator in it names a prerelease "
            "of that same major.minor.patch, so a prerelease never leaks "
            "into a range written for released versions."
        ),
        validator=(LOAD_CANDIDATE + require("compare_versions")
                   + require("satisfies") + r'''
# Build metadata is ignored entirely.
assert compare_versions("1.0.0+build.1", "1.0.0+build.9") == 0
assert compare_versions("1.0.0+x", "1.0.0") == 0

# Core precedence.
ORDER = ["1.0.0", "2.0.0", "2.1.0", "2.1.1"]
for index in range(len(ORDER) - 1):
    assert compare_versions(ORDER[index], ORDER[index + 1]) == -1
    assert compare_versions(ORDER[index + 1], ORDER[index]) == 1
    assert compare_versions(ORDER[index], ORDER[index]) == 0

# The prerelease chain from the specification, in order.
CHAIN = [
    "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
    "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
]
for index in range(len(CHAIN) - 1):
    lower, higher = CHAIN[index], CHAIN[index + 1]
    assert compare_versions(lower, higher) == -1, f"{lower} !< {higher}"
    assert compare_versions(higher, lower) == 1

# The two comparisons the chain exists to pin down.
assert compare_versions("1.0.0-beta.2", "1.0.0-beta.11") == -1, (
    "numeric prerelease fields compare numerically, not as text"
)
assert compare_versions("1.0.0-alpha.1", "1.0.0-alpha.beta") == -1, (
    "a numeric field ranks below an alphanumeric one"
)
assert compare_versions("1.0.0-alpha", "1.0.0-alpha.1") == -1

for bad in ("1.0", "1.0.0.0", "01.0.0", "1.01.0", "1.0.0-01", "v1.0.0", ""):
    try:
        compare_versions(bad, "1.0.0")
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} should not parse")

# Caret ranges pivot on the leftmost non-zero field.
assert satisfies("1.2.3", "^1.2.3")
assert satisfies("1.9.9", "^1.2.3")
assert not satisfies("2.0.0", "^1.2.3")
assert not satisfies("1.2.2", "^1.2.3")
assert satisfies("0.2.9", "^0.2.3")
assert not satisfies("0.3.0", "^0.2.3")
assert satisfies("0.0.3", "^0.0.3")
assert not satisfies("0.0.4", "^0.0.3")

# Tilde ranges allow the rest of the minor series.
assert satisfies("1.2.9", "~1.2.3")
assert not satisfies("1.3.0", "~1.2.3")
assert satisfies("1.2.0", "~1.2")
assert not satisfies("1.3.0", "~1.2")

# A conjunction must hold on every comparator.
assert satisfies("1.5.0", ">=1.0.0, <2.0.0")
assert not satisfies("2.0.0", ">=1.0.0, <2.0.0")
assert satisfies("1.0.0", "=1.0.0")
assert not satisfies("1.0.1", "=1.0.0")

# A prerelease only satisfies a range that named a prerelease of its own
# core version.
assert not satisfies("1.2.4-beta", "^1.2.3"), (
    "a prerelease must not leak into a released-version range"
)
assert not satisfies("3.0.0-alpha", ">=1.0.0")
assert satisfies("1.2.3-beta", ">=1.2.3-alpha, <2.0.0")
assert satisfies("1.2.3", "^1.2.3-alpha")
assert not satisfies("1.2.3-alpha", "^1.2.3-beta")
'''),
    ),
]
