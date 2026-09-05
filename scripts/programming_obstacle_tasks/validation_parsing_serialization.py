"""Held-out tasks: validation, parsing, and serialization.

These tasks target the places where a plausible-looking parser is wrong in
production: quoting and escaping, precedence rules that are not
lexicographic, and round-trip stability. Each validator drives the candidate
with inputs whose correct handling is specified by a public standard rather
than by the prompt's examples, so reproducing the examples is not enough.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "validation_parsing_serialization"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function parse_csv(text) that parses RFC 4180 "
            "delimiter-separated data into a list of rows, each a list of "
            "field strings. Fields may be wrapped in double quotes, and a "
            "quoted field may contain commas, CRLF or LF line breaks, and "
            "doubled double-quotes representing one literal quote. Rows may "
            "be separated by LF or CRLF. A trailing newline does not create "
            "an extra empty row. Raise ValueError on a quoted field that is "
            "never closed. Do not use the csv module."
        ),
        validator=LOAD_CANDIDATE + require("parse_csv") + '''
source = RESPONSE_TEXT
assert 'import csv' not in source and 'from csv' not in source, \\
    'the prompt forbids the csv module'

assert parse_csv('a,b,c') == [['a', 'b', 'c']]
assert parse_csv('a,b\\nc,d') == [['a', 'b'], ['c', 'd']]
assert parse_csv('a,b\\r\\nc,d') == [['a', 'b'], ['c', 'd']]
assert parse_csv('a,b\\n') == [['a', 'b']], 'trailing newline made a row'
assert parse_csv('') == []

# Empty fields are fields, not absences.
assert parse_csv('a,,c') == [['a', '', 'c']]
assert parse_csv(',') == [['', '']]

# Quoting.
assert parse_csv('"a,b",c') == [['a,b', 'c']]
assert parse_csv('"line1\\nline2",x') == [['line1\\nline2', 'x']]
assert parse_csv('"line1\\r\\nline2",x') == [['line1\\r\\nline2', 'x']]
assert parse_csv('"say ""hi""",x') == [['say "hi"', 'x']]
assert parse_csv('"",x') == [['', 'x']]
assert parse_csv('"a""""b"') == [['a""b']]

# Quotes only delimit at the start of a field.
assert parse_csv('a"b,c') == [['a"b', 'c']]

for unterminated in ('"abc', 'a,"b', '"a""'):
    try:
        parse_csv(unterminated)
    except ValueError:
        pass
    else:
        raise AssertionError(f'unterminated quote {unterminated!r} accepted')

# Round-trip a table containing every awkward character.
rows = [['plain', 'with,comma'], ['with"quote', 'with\\nnewline'], ['', 'z']]
def encode(rows):
    out = []
    for row in rows:
        fields = []
        for field in row:
            if any(ch in field for ch in ',"\\n\\r'):
                fields.append('"' + field.replace('"', '""') + '"')
            else:
                fields.append(field)
        out.append(','.join(fields))
    return '\\r\\n'.join(out)
assert parse_csv(encode(rows)) == rows, 'round trip lost data'
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function compare_versions(left, right) for "
            "Semantic Versioning 2.0.0 precedence. Return -1, 0 or 1. Compare "
            "major, minor and patch numerically. A version with a prerelease "
            "has lower precedence than the same version without one. "
            "Prerelease identifiers are compared dot-separated, left to "
            "right: numeric identifiers compare numerically and always rank "
            "below alphanumeric ones, and a longer prerelease outranks a "
            "shorter one when all preceding identifiers are equal. Build "
            "metadata after a plus sign is ignored entirely. Raise ValueError "
            "on input that is not a valid semantic version."
        ),
        validator=LOAD_CANDIDATE + require("compare_versions") + '''
def check(left, expected, right):
    got = compare_versions(left, right)
    assert got == expected, f'{left} vs {right}: got {got}, want {expected}'
    mirrored = compare_versions(right, left)
    assert mirrored == -expected, f'{right} vs {left}: not antisymmetric'

check('1.0.0', 0, '1.0.0')
check('2.0.0', 1, '1.9.9')
check('1.10.0', 1, '1.9.0')          # numeric, not lexicographic
check('1.0.10', 1, '1.0.9')

# Build metadata is ignored.
check('1.0.0+build.1', 0, '1.0.0')
check('1.0.0+a', 0, '1.0.0+b')
check('1.0.0-alpha+x', 0, '1.0.0-alpha+y')

# A prerelease ranks below its release.
check('1.0.0-alpha', -1, '1.0.0')

# The precedence chain from the specification.
chain = ['1.0.0-alpha', '1.0.0-alpha.1', '1.0.0-alpha.beta', '1.0.0-beta',
         '1.0.0-beta.2', '1.0.0-beta.11', '1.0.0-rc.1', '1.0.0']
for lower, higher in zip(chain, chain[1:]):
    check(higher, 1, lower)

# Numeric identifiers rank below alphanumeric ones.
check('1.0.0-1', -1, '1.0.0-alpha')
# Numeric identifiers compare numerically, so 11 outranks 2.
check('1.0.0-11', 1, '1.0.0-2')
# A longer prerelease outranks its own prefix.
check('1.0.0-alpha.1', 1, '1.0.0-alpha')

for bad in ('1.0', '1.0.0.0', 'v1.0.0', '1.0.x', '', '01.0.0',
            '1.0.0-', '1.0.0-alpha..1'):
    try:
        compare_versions(bad, '1.0.0')
    except ValueError:
        pass
    else:
        raise AssertionError(f'invalid version {bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function resolve_pointer(document, pointer) "
            "evaluating an RFC 6901 JSON Pointer against a structure of "
            "dicts, lists and scalars. The empty pointer returns the whole "
            "document. Each reference token is preceded by a slash; within a "
            "token, ~1 means a literal slash and ~0 means a literal tilde, "
            "and ~0 must not be unescaped before ~1. Array indices are "
            "decimal with no leading zeros. Raise KeyError for a missing "
            "object member, IndexError for an out-of-range array index, and "
            "ValueError for a malformed pointer."
        ),
        validator=LOAD_CANDIDATE + require("resolve_pointer") + '''
document = {
    'foo': ['bar', 'baz'],
    '': 0,
    'a/b': 1,
    'c%d': 2,
    'e^f': 3,
    'g|h': 4,
    'i\\\\\\\\j': 5,
    'k"l': 6,
    ' ': 7,
    'm~n': 8,
    'nested': {'list': [{'deep': 'value'}]},
}

assert resolve_pointer(document, '') is document
assert resolve_pointer(document, '/foo') == ['bar', 'baz']
assert resolve_pointer(document, '/foo/0') == 'bar'
assert resolve_pointer(document, '/foo/1') == 'baz'
assert resolve_pointer(document, '/') == 0, 'empty key token mishandled'
assert resolve_pointer(document, '/a~1b') == 1, '~1 must decode to a slash'
assert resolve_pointer(document, '/c%d') == 2
assert resolve_pointer(document, '/e^f') == 3
assert resolve_pointer(document, '/g|h') == 4
assert resolve_pointer(document, '/ ') == 7
assert resolve_pointer(document, '/m~0n') == 8, '~0 must decode to a tilde'
assert resolve_pointer(document, '/nested/list/0/deep') == 'value'

# Escape ordering: the token "~01" decodes to the literal "~1", never to a
# slash. An implementation that expands ~0 to a tilde before looking for ~1
# turns "~01" into "~1" and then into "/", which is the classic defect.
ordering = {'~1': 'tilde-one', '/': 'slash'}
assert resolve_pointer(ordering, '/~01') == 'tilde-one', \\
    'unescaped ~0 before ~1 and produced the wrong key'

try:
    resolve_pointer(document, '/missing')
except KeyError:
    pass
else:
    raise AssertionError('missing member did not raise KeyError')

for out_of_range in ('/foo/2', '/foo/99'):
    try:
        resolve_pointer(document, out_of_range)
    except IndexError:
        pass
    else:
        raise AssertionError(f'{out_of_range} did not raise IndexError')

for malformed in ('foo', 'foo/bar', '/foo/01', '/foo/-1', '/foo/x', '/~2'):
    try:
        resolve_pointer(document, malformed)
    except ValueError:
        pass
    else:
        raise AssertionError(f'malformed pointer {malformed!r} accepted')
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function parse_duration(text) converting an "
            "ISO 8601 duration such as P3DT4H5M6S into a total number of "
            "seconds as a float. Support years, months, weeks and days before "
            "the T separator and hours, minutes and seconds after it, a "
            "leading minus sign for a negative duration, and a fractional "
            "seconds component. Treat a year as 365 days and a month as 30 "
            "days. Raise ValueError for input that is not a valid duration, "
            "including a bare P, a missing T before a time component, and "
            "components given out of order."
        ),
        validator=LOAD_CANDIDATE + require("parse_duration") + '''
def close(got, want):
    assert abs(got - want) < 1e-6, f'got {got}, want {want}'

close(parse_duration('PT1S'), 1.0)
close(parse_duration('PT1M'), 60.0)
close(parse_duration('PT1H'), 3600.0)
close(parse_duration('P1D'), 86400.0)
close(parse_duration('P1W'), 7 * 86400.0)
close(parse_duration('P1M'), 30 * 86400.0)
close(parse_duration('P1Y'), 365 * 86400.0)
close(parse_duration('P3DT4H5M6S'), 3 * 86400 + 4 * 3600 + 5 * 60 + 6)
close(parse_duration('PT0S'), 0.0)
close(parse_duration('P0D'), 0.0)

# The same letter M means months before T and minutes after it.
close(parse_duration('P1MT1M'), 30 * 86400 + 60)

close(parse_duration('-PT30S'), -30.0)
close(parse_duration('-P1DT12H'), -(86400 + 12 * 3600))
close(parse_duration('PT1.5S'), 1.5)
close(parse_duration('PT0.001S'), 0.001)
close(parse_duration('P1Y2M3DT4H5M6.5S'),
      365 * 86400 + 2 * 30 * 86400 + 3 * 86400 + 4 * 3600 + 5 * 60 + 6.5)

# Large values must not overflow into an int-only path.
close(parse_duration('PT10000000S'), 10000000.0)

for bad in ('P', '', 'PT', '1D', 'P1H', 'PT1D', 'P1S', 'PT1M1H',
            'P1D1Y', 'PTS', 'P-1D', 'P1.5D2S', 'PT1,5S', 'X1D'):
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'invalid duration {bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python function canonical_json(value) returning a "
            "deterministic UTF-8 encoded bytes serialization suitable for "
            "hashing. Object keys are sorted by their Unicode code points, "
            "there is no insignificant whitespace, strings use the shortest "
            "valid escaping with only the escapes JSON requires, and integral "
            "floats serialize without a trailing point or exponent. Reject "
            "NaN and infinity with ValueError because they are not JSON, and "
            "reject a structure that contains itself with ValueError rather "
            "than recursing until the stack fails."
        ),
        validator=LOAD_CANDIDATE + require("canonical_json") + '''
import json as _json

assert canonical_json({'b': 1, 'a': 2}) == b'{"a":2,"b":1}'
assert canonical_json([1, 2, 3]) == b'[1,2,3]'
assert canonical_json({}) == b'{}'
assert canonical_json([]) == b'[]'
assert canonical_json(None) == b'null'
assert canonical_json(True) == b'true' and canonical_json(False) == b'false'
assert canonical_json('x') == b'"x"'

# Key ordering is by code point, so uppercase sorts before lowercase and
# non-ASCII sorts after both.
result = canonical_json({'b': 0, 'A': 0, 'a': 0, 'A\\u0308': 0})
assert result == '{"A":0,"A\\u0308":0,"a":0,"b":0}'.encode('utf-8'), result

# Integral floats lose their decimal point; non-integral ones keep precision.
assert canonical_json(1.0) == b'1'
assert canonical_json(-0.0) in (b'0', b'-0')
assert canonical_json(2.5) == b'2.5'
assert canonical_json(100) == b'100'

# Only the escapes JSON requires, and no others.
assert canonical_json('a"b') == b'"a\\\\"b"'
assert canonical_json('a\\\\b') == b'"a\\\\\\\\b"'
assert canonical_json('a\\nb') == b'"a\\\\nb"'
assert canonical_json('a\\tb') == b'"a\\\\tb"'
assert canonical_json('\\x00') == b'"\\\\u0000"'
assert canonical_json('/') == b'"/"', 'solidus must not be escaped'
assert canonical_json('\\u00e9') == '"\\u00e9"'.encode('utf-8'), \\
    'non-ASCII must be emitted as UTF-8, not \\\\u escapes'

# The output must be valid JSON that reparses to the same value.
for value in ({'z': [1, {'y': 'x'}], 'a': None}, [[[]]], {'k': 1.25}):
    assert _json.loads(canonical_json(value).decode('utf-8')) == value

# Determinism: equal structures built differently serialize identically.
left = {}
left['a'] = 1
left['b'] = 2
right = {}
right['b'] = 2
right['a'] = 1
assert canonical_json(left) == canonical_json(right)

for bad in (float('nan'), float('inf'), float('-inf')):
    try:
        canonical_json(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad} was serialized')

cycle = {}
cycle['self'] = cycle
try:
    canonical_json(cycle)
except ValueError:
    pass
except RecursionError:
    raise AssertionError('recursed until the stack failed instead of raising')
else:
    raise AssertionError('a self-referencing structure was serialized')
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function parse_content_type(header) returning "
            "a tuple of the lowercased media type and a dict of parameters. "
            "Parameter names are case-insensitive and lowercased; parameter "
            "values keep their case unless quoted. A quoted value may contain "
            "semicolons, spaces and backslash-escaped characters, which must "
            "be unescaped. Whitespace around separators is insignificant. A "
            "repeated parameter name keeps the first occurrence. Raise "
            "ValueError when the media type is missing or malformed, or when "
            "a quoted string is never closed."
        ),
        validator=LOAD_CANDIDATE + require("parse_content_type") + '''
assert parse_content_type('text/plain') == ('text/plain', {})
assert parse_content_type('TEXT/PLAIN') == ('text/plain', {})
assert parse_content_type('  text/plain  ') == ('text/plain', {})

media, params = parse_content_type('text/plain; charset=UTF-8')
assert media == 'text/plain'
assert params == {'charset': 'UTF-8'}, 'unquoted value case was not preserved'

# Parameter names lowercase; whitespace around separators is insignificant.
assert parse_content_type('a/b ;  CharSet = utf-8 ')[1] == {'charset': 'utf-8'}

# Quoted values.
assert parse_content_type('a/b; x="hello world"')[1] == {'x': 'hello world'}
assert parse_content_type('a/b; x="a;b"')[1] == {'x': 'a;b'}
assert parse_content_type('a/b; x="a\\\\"b"')[1] == {'x': 'a"b'}
assert parse_content_type('a/b; x="a\\\\\\\\b"')[1] == {'x': 'a\\\\b'}
assert parse_content_type('a/b; x=""')[1] == {'x': ''}

# Multipart boundaries are the everyday case that breaks naive splitting.
media, params = parse_content_type(
    'multipart/form-data; boundary="--x; y=z"; name=file')
assert media == 'multipart/form-data'
assert params == {'boundary': '--x; y=z', 'name': 'file'}

# First occurrence wins.
assert parse_content_type('a/b; p=1; p=2')[1] == {'p': '1'}

for bad in ('', 'text', 'text/', '/plain', 'text/plain; =v',
            'text/plain; x', 'text/plain; x="unclosed', 'te xt/plain'):
    try:
        parse_content_type(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'malformed header {bad!r} was accepted')
''',
    ),
]
