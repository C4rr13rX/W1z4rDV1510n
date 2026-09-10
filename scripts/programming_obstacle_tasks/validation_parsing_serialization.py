"""Held-out tasks: validation, parsing, and serialization.

These tasks target the places where a plausible-looking parser is wrong in
production: quoting and escaping, precedence rules that are not
lexicographic, and round-trip stability. Each validator drives the candidate
with inputs whose correct handling is specified by a public standard rather
than by the prompt's examples, so reproducing the examples is not enough.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import (
    LOAD_CANDIDATE,
    SHAPE_GUARDS,
    require,
)

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
        validator=LOAD_CANDIDATE + require("parse_duration") + SHAPE_GUARDS + '''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
parse_duration = returning(parse_duration, 'parse_duration(...)')
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
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function resolve_uri(base, reference) "
            "applying RFC 3986 section 5 reference resolution and returning "
            "the target URI as a string. base is an absolute URI carrying a "
            "scheme. Split each argument into scheme, authority, path, query "
            "and fragment, then apply the strict transform: a reference with "
            "its own scheme supplies every component itself; a reference "
            "with an authority takes only the base scheme; a reference whose "
            "path starts with a slash takes the base scheme and authority; "
            "and a reference with a relative path is merged onto the base "
            "path by replacing everything after the base path's last slash, "
            "or onto a single slash when the base has an authority and an "
            "empty path. A reference with an empty path keeps the base path "
            "and takes the base query unless the reference carries a query "
            "of its own. The fragment always comes from the reference. Apply "
            "the remove_dot_segments algorithm to the target path in every "
            "case except the empty-path one, discarding a '..' that would "
            "ascend past the root, and recompose the result. Dot segments in "
            "a query or fragment are left alone. Raise ValueError when base "
            "has no scheme. Do not use urllib or any other URI library."
        ),
        validator=LOAD_CANDIDATE + require("resolve_uri") + r'''
source = RESPONSE_TEXT
assert 'urllib' not in source, 'the prompt forbids urllib'

BASE = 'http://a/b/c/d;p?q'

# RFC 3986 section 5.4.1, the normal examples. These are the contract, not
# illustrations of it: an implementation that reproduces a subset of them by
# string surgery diverges on the abnormal set below.
for reference, expected in [
    ('g:h', 'g:h'),
    ('g', 'http://a/b/c/g'),
    ('./g', 'http://a/b/c/g'),
    ('g/', 'http://a/b/c/g/'),
    ('/g', 'http://a/g'),
    ('//g', 'http://g'),
    ('?y', 'http://a/b/c/d;p?y'),
    ('g?y', 'http://a/b/c/g?y'),
    ('#s', 'http://a/b/c/d;p?q#s'),
    ('g#s', 'http://a/b/c/g#s'),
    ('g?y#s', 'http://a/b/c/g?y#s'),
    (';x', 'http://a/b/c/;x'),
    ('g;x', 'http://a/b/c/g;x'),
    ('g;x?y#s', 'http://a/b/c/g;x?y#s'),
    ('', 'http://a/b/c/d;p?q'),
    ('.', 'http://a/b/c/'),
    ('./', 'http://a/b/c/'),
    ('..', 'http://a/b/'),
    ('../', 'http://a/b/'),
    ('../g', 'http://a/b/g'),
    ('../..', 'http://a/'),
    ('../../', 'http://a/'),
    ('../../g', 'http://a/g'),
]:
    got = resolve_uri(BASE, reference)
    assert got == expected, f'{reference!r} resolved to {got!r}, not {expected!r}'

# Section 5.4.2. Ascending past the root is absorbed rather than escaping,
# and a dot segment inside a query or fragment is ordinary text.
for reference, expected in [
    ('../../../g', 'http://a/g'),
    ('../../../../g', 'http://a/g'),
    ('/./g', 'http://a/g'),
    ('/../g', 'http://a/g'),
    ('g.', 'http://a/b/c/g.'),
    ('.g', 'http://a/b/c/.g'),
    ('g..', 'http://a/b/c/g..'),
    ('..g', 'http://a/b/c/..g'),
    ('./../g', 'http://a/b/g'),
    ('./g/.', 'http://a/b/c/g/'),
    ('g/./h', 'http://a/b/c/g/h'),
    ('g/../h', 'http://a/b/c/h'),
    ('g;x=1/./y', 'http://a/b/c/g;x=1/y'),
    ('g;x=1/../y', 'http://a/b/c/y'),
    ('g?y/./x', 'http://a/b/c/g?y/./x'),
    ('g?y/../x', 'http://a/b/c/g?y/../x'),
    ('g#s/./x', 'http://a/b/c/g#s/./x'),
    ('g#s/../x', 'http://a/b/c/g#s/../x'),
]:
    got = resolve_uri(BASE, reference)
    assert got == expected, f'{reference!r} resolved to {got!r}, not {expected!r}'

# An authority with an empty path merges onto '/' rather than onto nothing.
assert resolve_uri('http://a', 'g') == 'http://a/g'
assert resolve_uri('http://a?q', '') == 'http://a?q'

# A base with no authority still merges on its last slash.
assert resolve_uri('mailto:local/part', 'other') == 'mailto:local/other'

# An empty reference keeps the base query; a reference query replaces it,
# including an empty one.
assert resolve_uri(BASE, '?') == 'http://a/b/c/d;p?'

try:
    resolve_uri('//a/b', 'g')
except ValueError:
    pass
else:
    raise AssertionError('a base without a scheme was accepted')
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement Python functions encode_component(text) and "
            "decode_form(query). encode_component percent-encodes a string "
            "for use as a URI component: the RFC 3986 unreserved characters "
            "-- ASCII letters, digits, '-', '.', '_' and '~' -- are emitted "
            "unchanged, and every other character becomes one '%XX' group "
            "per byte of its UTF-8 encoding, with uppercase hexadecimal "
            "digits. A space encodes as %20, never as '+'. decode_form "
            "parses an application/x-www-form-urlencoded string into a list "
            "of (name, value) pairs in order. Pairs are separated by '&' and "
            "an empty pair is skipped. The first '=' in a pair separates "
            "name from value; a pair with no '=' has an empty value. Within "
            "a name or value '+' decodes to a space and '%XX' decodes to "
            "that byte, accepting either case of hexadecimal, and the "
            "resulting bytes are decoded as UTF-8. Raise ValueError for a "
            "'%' not followed by two hexadecimal digits, and for decoded "
            "bytes that are not valid UTF-8."
        ),
        validator=LOAD_CANDIDATE + require("encode_component")
        + require("decode_form") + r'''
assert encode_component('') == ''
assert encode_component('AZaz09') == 'AZaz09'

# The unreserved set is exactly these four punctuation marks. Two widely
# copied encoders get this wrong in opposite directions: one escapes '~',
# the other leaves '!', '*', "'" and parentheses alone.
assert encode_component('-._~') == '-._~', "'~' is unreserved"
assert encode_component("!*'()") == '%21%2A%27%28%29', \
    'only letters, digits and -._~ survive unencoded'

assert encode_component('a b') == 'a%20b', 'a space is %20, not +'
assert encode_component('/?#&=+;,:@$') == \
    '%2F%3F%23%26%3D%2B%3B%2C%3A%40%24'
assert encode_component('\x0f') == '%0F', 'hexadecimal digits are uppercase'
assert encode_component('\x7f') == '%7F'
assert encode_component('é') == '%C3%A9', 'non-ASCII encodes as UTF-8'
assert encode_component('水') == '%E6%B0%B4'
assert encode_component('\U0001f600') == '%F0%9F%98%80'

assert decode_form('') == []
assert decode_form('a=1&b=2') == [('a', '1'), ('b', '2')]
assert decode_form('a=1&a=2') == [('a', '1'), ('a', '2')], \
    'a repeated name keeps both pairs in order'
assert decode_form('a') == [('a', '')]
assert decode_form('a=') == [('a', '')]
assert decode_form('=v') == [('', 'v')]
assert decode_form('a=1&&b=2') == [('a', '1'), ('b', '2')]
assert decode_form('&') == []
assert decode_form('a=b=c') == [('a', 'b=c')], 'only the first = splits'
assert decode_form('a+b=c+d') == [('a b', 'c d')]
assert decode_form('a=%2B') == [('a', '+')], \
    'an encoded plus is a plus, not a space'
assert decode_form('%C3%A9=%f0%9f%98%80') == [('é', '\U0001f600')], \
    'percent groups accept either case and regroup into UTF-8'

# Encoding then decoding has to be lossless for the characters that are
# special to the form syntax itself.
for text in ('a&b=c', 'x+y', '100%', 'é水\U0001f600', ' ', '~!*'):
    assert decode_form('k=' + encode_component(text)) == [('k', text)], \
        f'{text!r} did not survive a round trip'

for bad in ('a=%', 'a=%2', 'a=%zz', 'a=%2g', '%=1', 'a=%C3', 'a=%80'):
    try:
        decode_form(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement Python functions b32encode(data) and b32decode(text) "
            "for the RFC 4648 section 6 base 32 encoding. b32encode takes "
            "bytes and returns a string: each group of five input bytes "
            "becomes eight characters of the alphabet "
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567', most significant five bits "
            "first, and a final partial group is zero-padded to a whole "
            "number of characters and then padded with '=' to eight "
            "characters. b32decode reverses it and must be strict: raise "
            "ValueError when the input length is not a multiple of eight, "
            "when a group carries a number of '=' characters other than 0, "
            "1, 3, 4 or 6, when '=' appears anywhere but at the end of the "
            "final group, when a character is outside the alphabet -- "
            "lowercase included -- and when the bits left over after the "
            "last whole byte of a partial group are not all zero, which is "
            "what makes an encoding canonical. Do not use the base64, "
            "binascii or codecs modules."
        ),
        validator=LOAD_CANDIDATE + require("b32encode") + require("b32decode")
        + r'''
source = RESPONSE_TEXT
for banned in ('base64', 'binascii', 'codecs'):
    assert banned not in source, f'the prompt forbids {banned}'

# RFC 4648 section 10 test vectors, which pin the padding lengths as well as
# the alphabet.
VECTORS = [
    (b'', ''),
    (b'f', 'MY======'),
    (b'fo', 'MZXQ===='),
    (b'foo', 'MZXW6==='),
    (b'foob', 'MZXW6YQ='),
    (b'fooba', 'MZXW6YTB'),
    (b'foobar', 'MZXW6YTBOI======'),
]
for raw, encoded in VECTORS:
    assert b32encode(raw) == encoded, f'{raw!r} encoded wrongly'
    assert b32decode(encoded) == raw, f'{encoded!r} decoded wrongly'

assert b32encode(bytes(range(5))) == 'AAAQEAYE'
assert b32decode('AAAQEAYE') == bytes(range(5))
assert b32encode(b'\xff' * 5) == '77777777'
assert b32decode('77777777') == b'\xff' * 5

for length in range(0, 26):
    raw = bytes((index * 37 + 11) & 0xFF for index in range(length))
    encoded = b32encode(raw)
    assert len(encoded) % 8 == 0, f'{length} bytes produced {encoded!r}'
    assert b32decode(encoded) == raw, f'{length} bytes did not round trip'

# The canonical-form rule. 'MZXW6===' is 'foo'; the same length with the
# final character carrying a set bit past the last whole byte encodes no
# byte string at all, and an implementation that simply drops the spare bits
# accepts a second spelling of every value.
assert b32decode('MZXW6===') == b'foo'
for non_canonical in ('MZXW7===', 'MZ======', 'MZXW6YR=', 'MZXR===='):
    try:
        b32decode(non_canonical)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{non_canonical!r} is not canonical but decoded')

for bad in (
    'MY=====',        # length 7
    'MY',             # unpadded partial group
    'MZXW6YTB=',      # length 9
    'MZXW6Y==',       # two padding characters is not a reachable count
    'MZX=====',       # five padding characters is not a reachable count
    'M=======',       # seven padding characters is not a reachable count
    'mzxw6===',       # lowercase is outside the alphabet
    'MZX=W6==',       # padding inside the group
    'MZXW6==8',       # padding before data
    'MZXW6=1=',
    'MZXW6!==',
    'MZXW6YTB========',
):
    try:
        b32decode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python function cbor_encode(value) producing the "
            "RFC 8949 core deterministic encoding of a value built from int, "
            "bytes, str, list, dict, bool and None. A data item begins with "
            "a byte holding a three-bit major type and a five-bit argument: "
            "an argument below 24 is that byte's low bits, otherwise the low "
            "bits are 24, 25, 26 or 27 and the argument follows in 1, 2, 4 "
            "or 8 big-endian bytes. Deterministic encoding always uses the "
            "shortest of those forms that holds the argument. Major type 0 "
            "carries a non-negative integer as its own value and major type "
            "1 carries a negative integer as -1 minus the value. Major type "
            "2 is a byte string and 3 a UTF-8 text string, each with its "
            "length as the argument followed by the bytes. Major type 4 is "
            "an array and 5 a map, each with its number of items or pairs as "
            "the argument followed by the encoded elements; a map's pairs "
            "are emitted sorted by the bytewise lexicographic order of the "
            "encoded keys, comparing byte values and treating a prefix as "
            "smaller, not by key length first. false, true and null are the "
            "bytes 0xF4, 0xF5 and 0xF6, and a bool is never encoded as an "
            "integer. Raise ValueError for an integer outside -2**64 to "
            "2**64-1 and for any other unsupported type. Do not use a CBOR "
            "library."
        ),
        validator=LOAD_CANDIDATE + require("cbor_encode") + r'''
# RFC 8949 appendix A. The argument-length boundaries are the point: 23 and
# 24 differ in shape, not just in value.
for value, expected in [
    (0, b'\x00'), (1, b'\x01'), (10, b'\x0a'), (23, b'\x17'),
    (24, b'\x18\x18'), (25, b'\x18\x19'), (100, b'\x18\x64'),
    (255, b'\x18\xff'), (256, b'\x19\x01\x00'), (1000, b'\x19\x03\xe8'),
    (65535, b'\x19\xff\xff'), (65536, b'\x1a\x00\x01\x00\x00'),
    (1000000, b'\x1a\x00\x0f\x42\x40'),
    (4294967295, b'\x1a\xff\xff\xff\xff'),
    (4294967296, b'\x1b\x00\x00\x00\x01\x00\x00\x00\x00'),
    (1000000000000, b'\x1b\x00\x00\x00\xe8\xd4\xa5\x10\x00'),
    (18446744073709551615, b'\x1b\xff\xff\xff\xff\xff\xff\xff\xff'),
    (-1, b'\x20'), (-10, b'\x29'), (-24, b'\x37'), (-25, b'\x38\x18'),
    (-100, b'\x38\x63'), (-1000, b'\x39\x03\xe7'),
    (-18446744073709551616, b'\x3b\xff\xff\xff\xff\xff\xff\xff\xff'),
]:
    got = cbor_encode(value)
    assert got == expected, f'{value} encoded as {got!r}, not {expected!r}'

assert cbor_encode(b'') == b'\x40'
assert cbor_encode(b'\x01\x02\x03\x04') == b'\x44\x01\x02\x03\x04'
assert cbor_encode(bytes(24)) == b'\x58\x18' + bytes(24), \
    'a 24-byte string needs the one-byte argument form'
assert cbor_encode('') == b'\x60'
assert cbor_encode('a') == b'\x61\x61'
assert cbor_encode('IETF') == b'\x64IETF'
assert cbor_encode('"\\') == b'\x62\x22\x5c'
assert cbor_encode('ü') == b'\x62\xc3\xbc', 'the length counts bytes'
assert cbor_encode('水') == b'\x63\xe6\xb0\xb4'
assert cbor_encode('\U00010151') == b'\x64\xf0\x90\x85\x91'

assert cbor_encode([]) == b'\x80'
assert cbor_encode([1, 2, 3]) == b'\x83\x01\x02\x03'
assert cbor_encode([1, [2, 3], [4, 5]]) == \
    b'\x83\x01\x82\x02\x03\x82\x04\x05'
assert cbor_encode(list(range(1, 26))) == \
    b'\x98\x19' + bytes(range(1, 24)) + b'\x18\x18\x18\x19'

assert cbor_encode(False) == b'\xf4'
assert cbor_encode(True) == b'\xf5'
assert cbor_encode(None) == b'\xf6'
assert cbor_encode([True, 1, False, 0, None]) == \
    b'\x85\xf5\x01\xf4\x00\xf6', 'a bool is not an integer'

assert cbor_encode({}) == b'\xa0'
assert cbor_encode({1: 2, 3: 4}) == b'\xa2\x01\x02\x03\x04'
assert cbor_encode({'a': 1, 'b': [2, 3]}) == \
    b'\xa2\x61\x61\x01\x61\x62\x82\x02\x03'

# Bytewise order over the encoded keys, not length-first order. The integer
# key encodes to three bytes beginning 0x19 and the text key to two bytes
# beginning 0x61, so the longer encoding sorts first.
assert cbor_encode({'a': 1, 1000: 2}) == \
    b'\xa2\x19\x03\xe8\x02\x61\x61\x01', \
    'map keys sort by encoded bytes, not by encoded length'
assert cbor_encode({b'\x00': 1, 25: 2}) == b'\xa2\x18\x19\x02\x41\x00\x01'

# A prefix sorts before the string that extends it.
assert cbor_encode({'ab': 1, 'a': 2}) == \
    b'\xa2\x61\x61\x02\x62\x61\x62\x01'
assert cbor_encode({'z': 1, 'aa': 2}) == \
    b'\xa2\x61\x7a\x01\x62\x61\x61\x02'

# Sorting is on the encoding, so it does not depend on insertion order.
assert cbor_encode({'a': 1, 1000: 2}) == cbor_encode({1000: 2, 'a': 1})

assert cbor_encode({'m': {'b': 1, 'a': 2}}) == \
    b'\xa1\x61\x6d\xa2\x61\x61\x02\x61\x62\x01', 'nested maps sort too'

for unsupported in (1.5, 2 ** 64, -(2 ** 64) - 1, set(), (1, 2), object()):
    try:
        cbor_encode(unsupported)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{unsupported!r} was encoded')
''',
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement a Python function validate(schema, instance) checking "
            "a JSON instance against a subset of JSON Schema and returning a "
            "list of {'path': pointer, 'keyword': name} dicts sorted by path "
            "then keyword, where pointer is the RFC 6901 pointer to the "
            "failing location with '~' escaped as '~0' and '/' as '~1'. "
            "Support the keywords type, enum, minimum, exclusiveMaximum, "
            "maxLength, required, properties, additionalProperties, items "
            "and uniqueItems. type is a name or list of names among null, "
            "boolean, object, array, string, number and integer; integer "
            "matches a number whose fractional part is zero, and a boolean "
            "is never a number or an integer. When type fails at a location, "
            "report only that failure there and do not descend. minimum and "
            "exclusiveMaximum apply only to numbers and maxLength only to "
            "strings, counting characters rather than encoded bytes; other "
            "instance types ignore them. required reports at most one "
            "failure per object. properties validates matching members; "
            "additionalProperties, when false, reports every member with no "
            "matching property, at that member's own path. items validates "
            "every element at its index. uniqueItems reports at most one "
            "failure per array. enum and uniqueItems compare by JSON value, "
            "so 1 and 1.0 are equal while true and 1 are not. Do not use a "
            "JSON Schema library."
        ),
        validator=LOAD_CANDIDATE + require("validate") + r'''
def only(schema, instance, keyword, path=''):
    got = validate(schema, instance)
    assert got == [{'path': path, 'keyword': keyword}], \
        f'{instance!r} against {schema!r} gave {got!r}'


def clean(schema, instance):
    got = validate(schema, instance)
    assert got == [], f'{instance!r} against {schema!r} gave {got!r}'


clean({}, {'anything': [1, 2]})
clean({'type': 'integer'}, 1)
clean({'type': 'integer'}, 1.0)
clean({'type': 'number'}, 1)
clean({'type': ['string', 'null']}, None)
clean({'type': 'boolean'}, False)
only({'type': 'integer'}, 1.5, 'type')
only({'type': 'string'}, 1, 'type')
only({'type': 'null'}, 0, 'type')

# A Python bool is an int at the language level and is not an integer at the
# JSON level. Every keyword below has to agree about that.
only({'type': 'integer'}, True, 'type')
only({'type': 'number'}, True, 'type')
only({'type': 'boolean'}, 1, 'type')

clean({'minimum': 3}, 3)
only({'minimum': 3}, 2, 'minimum')
clean({'minimum': 3}, 'x')
clean({'minimum': 3}, True)
only({'exclusiveMaximum': 3}, 3, 'exclusiveMaximum')
clean({'exclusiveMaximum': 3}, 2.5)

clean({'maxLength': 1}, 'é')
clean({'maxLength': 1}, '\U0001f600')
only({'maxLength': 2}, 'abc', 'maxLength')
clean({'maxLength': 0}, [1, 2, 3])

clean({'enum': [1, 2]}, 1.0)
clean({'enum': [{'a': [1]}]}, {'a': [1.0]})
only({'enum': [1, 2]}, True, 'enum')
only({'enum': [1, 2]}, 3, 'enum')
only({'enum': [None]}, 0, 'enum')

only({'uniqueItems': True}, [1, 1.0], 'uniqueItems')
clean({'uniqueItems': True}, [True, 1])
clean({'uniqueItems': True}, [{'a': 1}, {'a': 2}])
only({'uniqueItems': True}, [{'a': 1}, {'a': 1.0}], 'uniqueItems')
assert len(validate({'uniqueItems': True}, [1, 1, 1, 1])) == 1, \
    'uniqueItems reports once per array'

only({'type': 'object', 'required': ['a', 'b']}, {'a': 1}, 'required')
assert validate({'required': ['a', 'b']}, {}) == \
    [{'path': '', 'keyword': 'required'}], 'required reports once per object'
clean({'required': ['a']}, [1])

# The pointer has to be escaped, and the escapes are not interchangeable.
only({'properties': {'a/b': {'type': 'string'}}}, {'a/b': 1},
     'type', '/a~1b')
only({'properties': {'m~n': {'type': 'string'}}}, {'m~n': 1},
     'type', '/m~0n')
only({'properties': {'x': {'properties': {'y': {'type': 'string'}}}}},
     {'x': {'y': 1}}, 'type', '/x/y')

only({'properties': {'a': {}}, 'additionalProperties': False},
     {'a': 1, 'b': 2}, 'additionalProperties', '/b')
clean({'properties': {'a': {}}}, {'a': 1, 'b': 2})
clean({'additionalProperties': False}, [1, 2])

only({'items': {'type': 'integer'}}, [1, 'x', 3], 'type', '/1')
only({'items': {'items': {'type': 'integer'}}}, [[1], ['x']], 'type', '/1/0')
clean({'items': {'type': 'integer'}}, [])

assert validate(
    {'properties': {'b': {'type': 'string'}, 'a': {'minimum': 5}},
     'required': ['z']},
    {'b': 1, 'a': 1},
) == [
    {'path': '', 'keyword': 'required'},
    {'path': '/a', 'keyword': 'minimum'},
    {'path': '/b', 'keyword': 'type'},
], 'errors sort by path then keyword'

assert validate({'minimum': 5, 'exclusiveMaximum': 0, 'enum': [9]}, 3) == [
    {'path': '', 'keyword': 'enum'},
    {'path': '', 'keyword': 'exclusiveMaximum'},
    {'path': '', 'keyword': 'minimum'},
], 'several keywords can fail at one path'

assert validate({'type': 'object', 'minimum': 5}, 3) == \
    [{'path': '', 'keyword': 'type'}], \
    'a failed type suppresses the other keywords at that path'
''',
    ),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement a Python function parse_dictionary(text) parsing an "
            "RFC 8941 structured field Dictionary into a dict mapping each "
            "key to a (value, parameters) tuple, where parameters is a dict. "
            "A key starts with a lowercase letter or '*' and continues with "
            "lowercase letters, digits, '_', '-', '.' or '*'. A member is a "
            "key, optionally '=' and a bare item; a member with no '=' has "
            "the value True. Members are separated by a comma with any "
            "number of spaces around it, and a trailing comma is an error. "
            "A repeated key keeps the last member. Parameters follow the "
            "value as zero or more of ';' then optional spaces then a key, "
            "optionally '=' and a bare item, defaulting to True; no space "
            "may precede the ';'. A bare item is one of: '?0' or '?1' for "
            "False or True; a double-quoted string of printable ASCII where "
            "only '\\\"' and '\\\\' may be escaped; a number with an optional "
            "'-', which is an int of at most 15 digits when it has no '.', "
            "and otherwise a float with at most 12 digits before the '.' and "
            "one to three after it; or a token starting with a letter or '*' "
            "and continuing with letters, digits and any of \"!#$%&'*+-.^_`|"
            "~:/\", returned as the dict {'token': text}. The empty string "
            "is an empty dictionary. Raise ValueError on anything else."
        ),
        validator=LOAD_CANDIDATE + require("parse_dictionary") + r'''
assert parse_dictionary('') == {}
assert parse_dictionary('   ') == {}
assert parse_dictionary('a=1, b=2') == {'a': (1, {}), 'b': (2, {})}
assert parse_dictionary('a=1 ,  b=2') == {'a': (1, {}), 'b': (2, {})}

# A key with no '=' is the boolean true, which is the whole reason the
# syntax allows it. An empty string here would be a different field.
assert parse_dictionary('a') == {'a': (True, {})}
assert parse_dictionary('a, b=1') == {'a': (True, {}), 'b': (1, {})}
assert parse_dictionary('a=?0') == {'a': (False, {})}
assert parse_dictionary('a=?1') == {'a': (True, {})}

value = parse_dictionary('a=1')['a'][0]
assert isinstance(value, int) and not isinstance(value, bool), \
    'an integer stays an integer'
value = parse_dictionary('a=1.0')['a'][0]
assert isinstance(value, float), 'a decimal is a float even when whole'
assert parse_dictionary('a=1.500')['a'][0] == 1.5
assert parse_dictionary('a=-4.25')['a'][0] == -4.25
assert parse_dictionary('a=-0')['a'][0] == 0
assert parse_dictionary('a=999999999999999')['a'][0] == 999999999999999

assert parse_dictionary('a="b, c"') == {'a': ('b, c', {})}, \
    'a comma inside a string does not separate members'
assert parse_dictionary('a=";=?"') == {'a': (';=?', {})}
assert parse_dictionary(r'a="x\"y\\z"') == {'a': ('x"y\\z', {})}
assert parse_dictionary('a=""') == {'a': ('', {})}

assert parse_dictionary('a=foo') == {'a': ({'token': 'foo'}, {})}
assert parse_dictionary('a=*x') == {'a': ({'token': '*x'}, {})}
assert parse_dictionary('a=text/plain') == \
    {'a': ({'token': 'text/plain'}, {})}
assert parse_dictionary('a="foo"') != parse_dictionary('a=foo'), \
    'a token and a string are different types'

assert parse_dictionary('a=1;b=2') == {'a': (1, {'b': 2})}
assert parse_dictionary('a;b=1;c') == {'a': (True, {'b': 1, 'c': True})}
assert parse_dictionary('a=1;  b=2') == {'a': (1, {'b': 2})}
assert parse_dictionary('a=1;b, c=2') == {'a': (1, {'b': True}),
                                          'c': (2, {})}
assert parse_dictionary('a="x";b="y"') == {'a': ('x', {'b': 'y'})}
assert parse_dictionary('a=1, a=2') == {'a': (2, {})}, 'the last member wins'

for bad in (
    'a=1,',
    'a=1, ',
    ',a=1',
    'a=',
    'a=;b=1',
    'A=1',
    '1=a',
    'a=1 ;b=2',
    'a=1.2345',
    'a=1.',
    'a=.5',
    'a=1234567890123456',
    'a=1234567890123.5',
    'a=?2',
    'a=?',
    'a="unterminated',
    r'a="bad\nescape"',
    'a="tab\there"',
    'a=1 b=2',
    'a==1',
    'a=1;;b=2',
):
    try:
        parse_dictionary(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement a Python function canonical_ipv6(text) returning the "
            "canonical text form of an IPv6 address. The input is eight "
            "groups of one to four hexadecimal digits separated by colons, "
            "where at most one run of consecutive all-zero groups may be "
            "written as '::', and where the final two groups may instead be "
            "written as a dotted-quad IPv4 address of four decimal octets "
            "with no leading zeros, contributing the high and low sixteen "
            "bits in that order. The output follows RFC 5952: lowercase "
            "hexadecimal, no leading zeros in a group, never a dotted-quad, "
            "and '::' replacing the longest run of two or more consecutive "
            "zero groups -- the leftmost such run when several are equally "
            "long, and never a run of only one zero group. Raise ValueError "
            "for an empty string, more than one '::', a group that is empty "
            "or longer than four digits or not hexadecimal, a total that is "
            "not eight groups, a '::' standing for no group at all, an IPv4 "
            "part that is not last, and an octet with a leading zero or "
            "above 255. Do not use the ipaddress or socket modules."
        ),
        validator=LOAD_CANDIDATE + require("canonical_ipv6") + r'''
source = RESPONSE_TEXT
assert 'ipaddress' not in source, 'the prompt forbids the ipaddress module'

assert canonical_ipv6('2001:0db8:0000:0000:0000:0000:0000:0001') == \
    '2001:db8::1'
assert canonical_ipv6('2001:0DB8:AbCd::1') == '2001:db8:abcd::1', \
    'hexadecimal is lowercased'
assert canonical_ipv6('2001:db8:0:0:0:0:2:1') == '2001:db8::2:1'
assert canonical_ipv6('1:2:3:4:5:6:7:8') == '1:2:3:4:5:6:7:8'
assert canonical_ipv6('0:0:0:0:0:0:0:0') == '::'
assert canonical_ipv6('::') == '::'
assert canonical_ipv6('::1') == '::1'
assert canonical_ipv6('1::') == '1::'
assert canonical_ipv6('0:1:0:0:0:0:0:0') == '0:1::'

# A single zero group is written out. Compressing it is shorter and wrong,
# and it is the rule an implementation that reaches for the first zero run
# breaks first.
assert canonical_ipv6('2001:db8:0:1:1:1:1:1') == '2001:db8:0:1:1:1:1:1'
assert canonical_ipv6('1:2:3:4:5:6:7::') == '1:2:3:4:5:6:7:0'
assert canonical_ipv6('1:0:2:3:4:5:6:7') == '1:0:2:3:4:5:6:7'

# The longest run wins over an earlier shorter one.
assert canonical_ipv6('2001:db8:0:1:0:0:0:1') == '2001:db8:0:1::1'
assert canonical_ipv6('1:0:0:2:0:0:0:3') == '1:0:0:2::3'

# Equal-length runs are broken leftmost.
assert canonical_ipv6('2001:db8:0:0:1:0:0:1') == '2001:db8::1:0:0:1'
assert canonical_ipv6('1:0:0:2:0:0:3:0') == '1::2:0:0:3:0'

# A dotted-quad is parsed and then written as two hexadecimal groups.
assert canonical_ipv6('::ffff:192.0.2.1') == '::ffff:c000:201'
assert canonical_ipv6('0:0:0:0:0:ffff:192.168.1.1') == '::ffff:c0a8:101'
assert canonical_ipv6('::255.255.255.255') == '::ffff:ffff'
assert canonical_ipv6('1:2:3:4:5:6:0.0.0.0') == '1:2:3:4:5:6::'

for bad in (
    '',
    '1::2::3',
    '1:2:3:4:5:6:7:8:9',
    '1:2:3:4:5:6:7',
    '12345::',
    '1:2:3:4:5:6:7:8:',
    ':1:2:3:4:5:6:7',
    '1:',
    ':',
    'g::1',
    '1::2:',
    '1.2.3.4',
    '::1.2.3.4.5',
    '::1.2.3',
    '::300.1.1.1',
    '::01.2.3.4',
    '::1.2.3.4:5',
    '1:2:3:4:5:6:7:8:1.2.3.4',
    '0x1::',
    '1:: 2',
    '+1::',
):
    try:
        canonical_ipv6(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement Python functions punycode_encode(label) and "
            "punycode_decode(text) for the RFC 3492 bootstring encoding with "
            "the IDNA parameters: base 36, tmin 1, tmax 26, skew 38, damp "
            "700, initial bias 72, initial n 128, delimiter '-', and the "
            "digit alphabet 'abcdefghijklmnopqrstuvwxyz0123456789' where "
            "'a' is 0 and '0' is 26. Encode a single label without any ACE "
            "prefix: emit the code points below 128 in order, then a '-' if "
            "there were any, then the delta encoding of the remaining code "
            "points in ascending order, adapting the bias after each one. "
            "Decode reverses it, taking the part after the last '-' as the "
            "extended part and treating the digit alphabet as lowercase "
            "only. Raise ValueError for a non-ASCII character in the basic "
            "part of a decode input, for a character outside the digit "
            "alphabet in the extended part, and for an extended part that "
            "ends in the middle of a number. Do not use the punycode or "
            "idna codecs, the codecs or encodings modules, or any IDNA "
            "library."
        ),
        validator=LOAD_CANDIDATE + require("punycode_encode")
        + require("punycode_decode") + r'''
source = RESPONSE_TEXT
for banned in ('codecs', 'encodings', '"punycode"', "'punycode'",
               '"idna"', "'idna'"):
    assert banned not in source, f'the prompt forbids {banned}'

# Encoding is not a substitution cipher: the same character encodes
# differently depending on what came before it, because the bias adapts.
VECTORS = [
    ('', ''),
    ('abc', 'abc-'),
    ('a-b', 'a-b-'),
    ('bücher', 'bcher-kva'),
    ('münchen', 'mnchen-3ya'),
    ('räksmörgås', 'rksmrgs-5wao1o'),
    ('例え', 'r8jz45g'),
    ('ドメイン名例', 'eckwd4c7cu47r2wf'),
    ('☃', 'n3h'),
    ('☃☃', 'n3ha'),
]
for label, encoded in VECTORS:
    got = punycode_encode(label)
    assert got == encoded, f'{label!r} encoded as {got!r}, not {encoded!r}'
    back = punycode_decode(encoded)
    assert back == label, f'{encoded!r} decoded as {back!r}, not {label!r}'

# An all-ASCII label still gets the delimiter, so decoding is unambiguous.
assert punycode_encode('abc').endswith('-')
assert punycode_decode('abc-') == 'abc'
assert punycode_decode('-') == ''

# Position matters as much as the code points, so a permutation of the same
# characters is a different encoding.
assert punycode_encode('éa') != punycode_encode('aé')
assert punycode_decode(punycode_encode('éa')) == 'éa'
assert punycode_decode(punycode_encode('aé')) == 'aé'

for label in (
    'a', 'z9', 'hello-world',
    'é', 'éèê', 'café', 'écaf',
    '\U0001f600', 'x\U0001f600y', '中文',
    'aé1é2é3', 'カタカナ', 'på-lys', 'דוגמה',
):
    encoded = punycode_encode(label)
    assert all(ord(char) < 128 for char in encoded), \
        f'{label!r} encoded to non-ASCII'
    assert punycode_decode(encoded) == label, f'{label!r} did not round trip'

for bad in ('bü-x', 'a-é'):
    try:
        punycode_decode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')

for bad in ('bcher-kv!', 'abc-A', 'abc- ', 'n3h_'):
    try:
        punycode_decode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} has a bad digit but was accepted')

for bad in ('kv', 'abc-kv'):
    try:
        punycode_decode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} ends mid-number but was accepted')
''',
    ),
    task(
        f"{FAMILY}-0015", FAMILY,
        prompt=(
            "Implement a Python function parse_multipart(body, boundary) "
            "where body is bytes and boundary is a str, returning a list of "
            "dicts with the keys 'name', 'filename', 'headers' and "
            "'content'. The delimiter is CRLF followed by '--' and the "
            "boundary, except that the first delimiter omits the leading "
            "CRLF when it begins the body. Bytes before the first delimiter "
            "and after the closing delimiter -- the delimiter followed by "
            "'--' -- are discarded. Otherwise a delimiter is followed by "
            "CRLF and then a part. A part is header lines separated by CRLF, "
            "then an empty line, then the content, which runs up to but does "
            "not include the CRLF that begins the next delimiter; only that "
            "one CRLF belongs to the delimiter, so content ending in blank "
            "lines keeps them. 'headers' maps lowercased header names to "
            "values stripped of surrounding spaces, as str. 'name' and "
            "'filename' come from the Content-Disposition parameters, where "
            "a value may be quoted and a backslash inside quotes escapes the "
            "next character; 'filename' is None when absent. Content is "
            "returned as bytes and is never decoded. Raise ValueError when "
            "no delimiter is present, when the closing delimiter is missing, "
            "when a delimiter is followed by neither CRLF nor '--', when a "
            "part's headers are not terminated by an empty line, and when a "
            "part has no Content-Disposition header or no name parameter."
        ),
        validator=LOAD_CANDIDATE + require("parse_multipart") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
parse_multipart = returning(parse_multipart, 'parse_multipart(...)')
BODY = (
    b'preamble is discarded\r\n'
    b'--X\r\n'
    b'Content-Disposition: form-data; name="a"\r\n'
    b'\r\n'
    b'1\r\n'
    b'--X\r\n'
    b'content-disposition: form-data; name="f"; filename="n\\"q.txt"\r\n'
    b'Content-Type:   text/plain  \r\n'
    b'\r\n'
    b'line\r\n\r\n'
    b'\r\n'
    b'--X\r\n'
    b'Content-Disposition: form-data; name=plain\r\n'
    b'\r\n'
    b'a--Xb\r\n'
    b'--X--\r\n'
    b'epilogue is discarded'
)

parts = parse_multipart(BODY, 'X')
assert len(parts) == 3, f'expected three parts, got {len(parts)}'

assert parts[0]['name'] == 'a'
assert parts[0]['filename'] is None
assert parts[0]['content'] == b'1'
assert parts[0]['headers'] == {'content-disposition': 'form-data; name="a"'}

assert parts[1]['name'] == 'f'
assert parts[1]['filename'] == 'n"q.txt', \
    'a backslash inside quotes escapes the next character'
assert parts[1]['headers']['content-type'] == 'text/plain', \
    'header names lowercase and values strip'

# Exactly one CRLF belongs to the delimiter. A part whose content genuinely
# ends in a blank line keeps it, which is what separates stripping one
# delimiter from trimming whatever trailing whitespace happens to be there.
assert parts[1]['content'] == b'line\r\n\r\n', \
    f"content was trimmed to {parts[1]['content']!r}"

# The boundary only delimits when a CRLF precedes it, so this is content.
assert parts[2]['name'] == 'plain', 'an unquoted parameter value is allowed'
assert parts[2]['content'] == b'a--Xb'

# No preamble: the first delimiter starts the body with no leading CRLF.
FLUSH = (
    b'--Y\r\n'
    b'Content-Disposition: form-data; name="k"\r\n'
    b'\r\n'
    b'\r\n'
    b'--Y--\r\n'
)
flush = parse_multipart(FLUSH, 'Y')
assert len(flush) == 1 and flush[0]['name'] == 'k'
assert flush[0]['content'] == b'', 'an empty part has empty content'

BINARY = (
    b'--Z\r\n'
    b'Content-Disposition: form-data; name="b"; filename="raw.bin"\r\n'
    b'\r\n' + bytes(range(256)) + b'\r\n'
    b'--Z--\r\n'
)
binary = parse_multipart(BINARY, 'Z')
assert binary[0]['content'] == bytes(range(256)), \
    'content is returned as raw bytes'
assert binary[0]['filename'] == 'raw.bin'

for bad in (
    b'nothing here at all',
    b'--Q\r\nContent-Disposition: form-data; name="a"\r\n\r\n1\r\n',
    b'--Q\r\nContent-Disposition: form-data; name="a"\r\n1\r\n--Q--\r\n',
    b'--Q\r\nContent-Type: text/plain\r\n\r\n1\r\n--Q--\r\n',
    b'--Q\r\nContent-Disposition: form-data\r\n\r\n1\r\n--Q--\r\n',
    b'--Q\r\nContent-Disposition: form-data; name="a\r\n\r\n1\r\n--Q--\r\n',
    b'--Qtrailing\r\n\r\n1\r\n--Q--\r\n',
):
    try:
        parse_multipart(bad, 'Q')
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0016", FAMILY,
        prompt=(
            "Implement a Python function parse_json(text) parsing one RFC "
            "8259 JSON document strictly and returning the Python value. "
            "Whitespace is only space, tab, LF and CR. A number is an "
            "optional '-', then '0' or a digit 1-9 followed by digits, then "
            "optionally '.' with at least one digit, then optionally 'e' or "
            "'E' with an optional sign and at least one digit; a number with "
            "neither fraction nor exponent is an int and any other number is "
            "a float. A string is double-quoted, allows the escapes "
            "\\\" \\\\ \\/ \\b \\f \\n \\r \\t and \\uXXXX, forbids raw "
            "characters below U+0020, and requires a high surrogate escape "
            "to be followed by a low surrogate escape, which together form "
            "one character. The literals are exactly true, false and null. "
            "An object has string keys and keeps the last of repeated keys. "
            "Arrays and objects forbid a trailing comma. Raise ValueError "
            "for anything else, including a leading zero, a leading '+', a "
            "bare fraction, NaN or Infinity, an unpaired surrogate, an "
            "empty document, and any trailing content after the value. Do "
            "not use the json module."
        ),
        validator=LOAD_CANDIDATE + require("parse_json") + r'''
source = RESPONSE_TEXT
assert 'import json' not in source and 'from json' not in source, \
    'the prompt forbids the json module'

assert parse_json('0') == 0
assert parse_json('-0') == 0
assert parse_json('123') == 123
assert parse_json('-123') == -123

# The int/float split is part of the contract, and a parser that funnels
# every number through float() loses it silently on values that still
# compare equal.
value = parse_json('1')
assert isinstance(value, int) and not isinstance(value, bool)
for text in ('1.0', '1e2', '1E+2', '1.5e-3', '0.0', '-0.0'):
    assert isinstance(parse_json(text), float), f'{text} is a float'
assert parse_json('1e2') == 100.0
assert parse_json('1.5e-3') == 0.0015

assert parse_json('true') is True
assert parse_json('false') is False
assert parse_json('null') is None

assert parse_json('""') == ''
assert parse_json('"abc"') == 'abc'
assert parse_json('"a\\/b"') == 'a/b'
assert parse_json('"\\u0041"') == 'A'
assert parse_json('"\\b\\f\\n\\r\\t"') == '\b\f\n\r\t'
assert parse_json('"\\\\\\""') == '\\"'
assert parse_json('"é"') == 'é', 'a raw non-ASCII character is fine'

# A supplementary character is written as a surrogate pair and has to come
# back as one character, not two.
assert parse_json('"\\ud83d\\ude00"') == '\U0001f600'
assert len(parse_json('"\\ud83d\\ude00"')) == 1

assert parse_json('[]') == []
assert parse_json('{}') == {}
assert parse_json('  {\t"a" : [1, 2, {"b": null}], "c":true}\r\n') == \
    {'a': [1, 2, {'b': None}], 'c': True}
assert parse_json('{"a":1,"a":2}') == {'a': 2}, 'the last key wins'
assert parse_json('[[[[1]]]]') == [[[[1]]]]
assert parse_json('{"":0}') == {'': 0}

for bad in (
    '', '   ', '\n',
    '01', '-01', '00', '1.', '.5', '+1', '1e', '1e+', '1.e2', '--1', '1..2',
    'NaN', 'Infinity', '-Infinity', '0x10', '1_000',
    'True', 'TRUE', 'tru', 'nul', 'nulll',
    '"unterminated', '"a\tb"', '"\\x41"', '"\\u00"', '"\\uZZZZ"', '"\\"',
    '"\\ud800"', '"\\udc00"', '"\\udc00\\ud800"', '"\\ud800a"',
    '"\\ud800\\u0041"',
    '[1,]', '[,1]', '[1 2]', '[1,,2]', '[', ']',
    '{,}', '{"a":1,}', '{"a"}', '{"a" 1}', "{'a':1}", '{a:1}', '{1:2}', '{',
    '1 2', '{} {}', '[1] x', 'nulltrue',
):
    try:
        parse_json(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0017", FAMILY,
        prompt=(
            "Implement a Python function parse_range(header, length) that "
            "resolves an RFC 7233 HTTP Range header against a representation "
            "of `length` bytes. Return a list of (first, last) byte offsets, "
            "both inclusive and both resolved to concrete positions, in the "
            "order the header lists them. Support `bytes=first-last`, an "
            "open-ended `bytes=first-`, and a suffix range `bytes=-n` "
            "meaning the final n bytes. A last-byte position at or beyond "
            "the end of the representation is clamped to the final byte, and "
            "a suffix longer than the representation starts at byte 0. Skip "
            "a range whose first byte lies beyond the representation and a "
            "suffix of zero bytes. Raise ValueError for a unit other than "
            "bytes, for a syntactically malformed spec, for a last position "
            "before the first, and when no range in the header is "
            "satisfiable."
        ),
        validator=LOAD_CANDIDATE + require("parse_range") + r'''
assert parse_range('bytes=0-499', 10000) == [(0, 499)]
assert parse_range('bytes=500-', 10000) == [(500, 9999)]
assert parse_range('bytes=-500', 10000) == [(9500, 9999)]
assert parse_range('bytes=0-0,-1', 10000) == [(0, 0), (9999, 9999)]
assert parse_range('bytes=0-1, 5-6', 10000) == [(0, 1), (5, 6)]
assert parse_range('BYTES=0-1', 10000) == [(0, 1)], 'the unit is case-insensitive'

# A last position past the end is clamped, it is not a syntax error.
assert parse_range('bytes=9500-10500', 10000) == [(9500, 9999)]
assert parse_range('bytes=0-9999', 10000) == [(0, 9999)]

# A suffix longer than the representation is the whole representation. A
# reader that computes length - suffix without a floor returns a negative
# first byte here and the caller seeks backwards past the start.
assert parse_range('bytes=-50000', 10000) == [(0, 9999)]
assert parse_range('bytes=-1', 1) == [(0, 0)]

# An unsatisfiable member is dropped while a satisfiable one survives.
assert parse_range('bytes=20000-,0-1', 10000) == [(0, 1)]
assert parse_range('bytes=-0,0-1', 10000) == [(0, 1)]

for bad in ('items=0-1', 'bytes=abc', 'bytes=5-2', 'bytes=-0', 'bytes=20000-',
            'bytes=', 'bytes=-', '0-1', 'bytes=1-2-3', 'bytes=0-1,'):
    try:
        parse_range(bad, 10000)
    except ValueError:
        pass
    else:
        raise AssertionError(f'malformed range {bad!r} was accepted')

# Every returned span must be orderable and inside the representation.
for header in ('bytes=0-499', 'bytes=-500', 'bytes=500-', 'bytes=0-0,-1'):
    for first, last in parse_range(header, 10000):
        assert 0 <= first <= last <= 9999, f'{header} produced ({first}, {last})'
''',
    ),
    task(
        f"{FAMILY}-0018", FAMILY,
        prompt=(
            "Implement a Python function decode_utf8(data) that decodes a "
            "bytes object as UTF-8 exactly as RFC 3629 defines it and "
            "returns a str. Accept only the shortest encoding of each code "
            "point. Raise ValueError for an overlong encoding, for a "
            "continuation byte where a leading byte is expected, for a "
            "truncated sequence at any position, for the surrogate range "
            "U+D800 to U+DFFF, and for any code point above U+10FFFF. Do not "
            "call bytes.decode, codecs, or str() on the input to do the "
            "work: implement the decoding yourself."
        ),
        validator=LOAD_CANDIDATE + require("decode_utf8") + r'''
source = RESPONSE_TEXT
for forbidden in ('.decode(', 'codecs', 'str(data', 'memoryview'):
    assert forbidden not in source, f'the prompt forbids {forbidden}'

assert decode_utf8(b'') == ''
assert decode_utf8(b'hello') == 'hello'
assert decode_utf8(b'\xc3\xa9') == '\u00e9'
assert decode_utf8(b'\xe2\x82\xac') == '\u20ac'
assert decode_utf8(b'\xf0\x9f\x92\xa9') == '\U0001f4a9'
assert decode_utf8(b'a\xc3\xa9b') == 'a\u00e9b'

# The boundaries either side of the surrogate block stay legal.
assert decode_utf8(b'\xed\x9f\xbf') == '\ud7ff'
assert decode_utf8(b'\xee\x80\x80') == '\ue000'
assert decode_utf8(b'\xf4\x8f\xbf\xbf') == '\U0010ffff'
assert decode_utf8(b'\xc2\x80') == '\u0080'
assert decode_utf8(b'\xe0\xa0\x80') == '\u0800'
assert decode_utf8(b'\xf0\x90\x80\x80') == '\U00010000'

overlong = (b'\xc0\x80', b'\xc1\xbf', b'\xe0\x80\x80', b'\xe0\x9f\xbf',
            b'\xf0\x80\x80\x80', b'\xf0\x8f\xbf\xbf')
surrogate = (b'\xed\xa0\x80', b'\xed\xbf\xbf', b'\xed\xad\xbf')
out_of_range = (b'\xf4\x90\x80\x80', b'\xf5\x80\x80\x80', b'\xf7\xbf\xbf\xbf',
                b'\xf8\x88\x80\x80\x80', b'\xff')
malformed = (b'\x80', b'\xbf', b'\xc3', b'\xe2\x82', b'\xf0\x9f\x92',
             b'\xc3\x28', b'\xe2\x28\xa1', b'a\xc3')
for group, label in ((overlong, 'overlong'), (surrogate, 'surrogate'),
                     (out_of_range, 'out of range'), (malformed, 'malformed')):
    for bad in group:
        try:
            decode_utf8(bad)
        except ValueError:
            continue
        raise AssertionError(f'{label} sequence {bad!r} was accepted')

# Agree with the reference decoder on everything that is actually valid.
for code in list(range(0, 0xd800, 97)) + list(range(0xe000, 0x110000, 1013)):
    text = chr(code)
    assert decode_utf8(text.encode('utf-8')) == text, f'U+{code:04X} mismatch'
''',
    ),
    task(
        f"{FAMILY}-0019", FAMILY,
        prompt=(
            "Implement a Python function decode_chunked(data) that decodes "
            "an RFC 7230 chunked transfer-coding body. `data` is the bytes "
            "of the encoded body. Return a tuple (body, trailers) where "
            "body is the concatenated chunk data as bytes and trailers is a "
            "dict of the trailer field names, lowercased, to their values "
            "with surrounding whitespace stripped. A chunk begins with its "
            "size in hexadecimal, optionally followed by ';' and chunk "
            "extensions, then CRLF, then exactly that many bytes, then "
            "CRLF. A zero-size chunk ends the body and is followed by the "
            "trailer section and a final empty line. Raise ValueError on a "
            "size that is not hexadecimal, on data that is truncated, and "
            "when a chunk is not followed by CRLF."
        ),
        validator=LOAD_CANDIDATE + require("decode_chunked") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
decode_chunked = returning(decode_chunked, 'decode_chunked(...)')
body, trailers = decode_chunked(b'4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n')
assert body == b'Wikipedia', body
assert trailers == {}

assert decode_chunked(b'0\r\n\r\n') == (b'', {})

# Chunk extensions are ignored, not parsed as part of the size.
assert decode_chunked(b'4;name=value\r\nWiki\r\n0\r\n\r\n')[0] == b'Wiki'
assert decode_chunked(b'4 ; a=b ; c\r\nWiki\r\n0\r\n\r\n')[0] == b'Wiki'

# Hexadecimal, in either case.
assert decode_chunked(b'A\r\n0123456789\r\n0\r\n\r\n')[0] == b'0123456789'
assert decode_chunked(b'a\r\n0123456789\r\n0\r\n\r\n')[0] == b'0123456789'
assert decode_chunked(b'1f\r\n' + b'z' * 31 + b'\r\n0\r\n\r\n')[0] == b'z' * 31

# The size is a byte count. A decoder that splits on CRLF instead of
# counting truncates this chunk at the first embedded line break.
assert decode_chunked(b'6\r\na\r\nb\r\n\r\n0\r\n\r\n')[0] == b'a\r\nb\r\n'
assert decode_chunked(b'2\r\n\r\n\r\n0\r\n\r\n')[0] == b'\r\n'

body, trailers = decode_chunked(b'3\r\nabc\r\n0\r\nX-Sum: 42\r\nX-B:  x \r\n\r\n')
assert body == b'abc'
assert trailers == {'x-sum': '42', 'x-b': 'x'}, trailers

many = b''.join(b'%x\r\n%s\r\n' % (len(p), p) for p in (b'a', b'bb', b'ccc'))
assert decode_chunked(many + b'0\r\n\r\n')[0] == b'abbccc'

for bad in (b'zz\r\nab\r\n0\r\n\r\n', b'4\r\nWi\r\n0\r\n\r\n', b'4\r\nWiki',
            b'0x4\r\nWiki\r\n0\r\n\r\n', b'-1\r\n\r\n0\r\n\r\n', b'4\r\nWiki\r\n',
            b';a=b\r\n\r\n0\r\n\r\n', b'4\r\nWikiXX0\r\n\r\n', b'3\r\nabc\r\n0\r\nbad\r\n\r\n'):
    try:
        decode_chunked(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'malformed body {bad!r} was accepted')

# Encode a payload the way the coding specifies and read it back.
payload = bytes(range(256)) * 3
encoded = bytearray()
step = 100
for offset in range(0, len(payload), step):
    piece = payload[offset:offset + step]
    encoded += b'%x\r\n' % len(piece) + piece + b'\r\n'
encoded += b'0\r\n\r\n'
assert decode_chunked(bytes(encoded))[0] == payload, 'round trip lost data'
''',
    ),
    task(
        f"{FAMILY}-0020", FAMILY,
        prompt=(
            "Implement a Python function parse_traceparent(header) that "
            "validates a W3C Trace Context `traceparent` header and returns "
            "a dict with keys 'version', 'trace_id', 'parent_id' and "
            "'sampled', or returns None when the header must be discarded "
            "and a new trace started. The header is hyphen-separated fields: "
            "a 2-digit version, a 32-digit trace-id, a 16-digit parent-id, "
            "and 2-digit trace-flags. Every field is LOWERCASE hexadecimal "
            "and must be exactly its stated length; uppercase hex is invalid, "
            "as is any other spelling of a hex number. Version 'ff' is "
            "reserved and invalid. A trace-id of all zeros is invalid, and so "
            "is a parent-id of all zeros. Version '00' defines exactly four "
            "fields, so anything after trace-flags makes the header invalid. "
            "A higher version may define further hyphen-separated fields: "
            "accept those and ignore them, but the header is still invalid "
            "if any of them is empty, which is what a trailing hyphen "
            "produces. 'sampled' is True when the low bit of trace-flags is "
            "set. Anything that is not a string returns None."
        ),
        validator=LOAD_CANDIDATE + require("parse_traceparent") + r"""
VALID = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
parsed = parse_traceparent(VALID)
assert isinstance(parsed, dict), f'a valid traceparent returned {parsed!r}'
for key in ('version', 'trace_id', 'parent_id', 'sampled'):
    assert key in parsed, f'result has no {key}'
assert parsed['version'] == '00'
assert parsed['trace_id'] == '4bf92f3577b34da6a3ce929d0e0e4736'
assert parsed['parent_id'] == '00f067aa0ba902b7'
assert parsed['sampled'] is True, 'flags 01 is sampled'

# The low bit alone decides sampling; the other seven are not ours to read.
assert parse_traceparent(VALID[:-2] + "00")['sampled'] is False
assert parse_traceparent(VALID[:-2] + "fe")['sampled'] is False
assert parse_traceparent(VALID[:-2] + "ff")['sampled'] is True

# Uppercase is a DIFFERENT header, not a lenient spelling of this one. A
# check built on int(field, 16) accepts it, and two peers that disagree about
# case silently split one trace into two.
assert parse_traceparent(VALID.upper()) is None, 'uppercase hex is invalid'
assert parse_traceparent(
    "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01"
) is None, 'an uppercase trace-id is invalid'
assert parse_traceparent(
    "00-4bf92f3577b34da6a3ce929d0e0e4736-00F067AA0BA902B7-01"
) is None, 'an uppercase parent-id is invalid'

# int(x, 16) also accepts these spellings at the right length.
assert parse_traceparent(
    "00-4bf92f3577b34da6a3ce929d0e0e47_6-00f067aa0ba902b7-01"
) is None, 'an underscore is not a hex digit'
assert parse_traceparent(
    "00-+bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
) is None, 'a sign is not a hex digit'

# All-zero identifiers are the spec's explicit "invalid" case, and they are
# exactly what a caller that forgot to seed its generator emits.
assert parse_traceparent(
    "00-" + "0" * 32 + "-00f067aa0ba902b7-01"
) is None, 'an all-zero trace-id is invalid'
assert parse_traceparent(
    "00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01"
) is None, 'an all-zero parent-id is invalid'
# A zero somewhere in the id is ordinary.
assert parse_traceparent(
    "00-00000000000000000000000000000001-0000000000000001-01"
) is not None

assert parse_traceparent(
    "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
) is None, 'version ff is reserved'

# Version 00 is exactly four fields. A future version may add more.
assert parse_traceparent(VALID + "-cc") is None, 'version 00 takes no extras'
future = parse_traceparent(
    "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-cc-dd"
)
assert isinstance(future, dict), 'a higher version must still parse'
assert future['version'] == '01'
assert future['trace_id'] == '4bf92f3577b34da6a3ce929d0e0e4736'
assert future['sampled'] is True
assert parse_traceparent(
    "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-"
) is None, 'a trailing hyphen leaves an empty field'

# Lengths are fixed, so a short or long field is not merely padded.
assert parse_traceparent(
    "0-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01") is None
assert parse_traceparent(
    "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01") is None
assert parse_traceparent(
    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01") is None
assert parse_traceparent(
    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1") is None
assert parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736") is None
assert parse_traceparent("") is None
assert parse_traceparent(None) is None
assert parse_traceparent(b"00-4bf92f3577b34da6a3ce929d0e0e4736") is None
assert parse_traceparent(1234) is None

# Whitespace is not trimmed: the header value is the field, and a parser that
# strips it disagrees with one that does not about the same trace.
assert parse_traceparent(" " + VALID) is None
assert parse_traceparent(VALID + " ") is None
""",
    ),
    task(
        f"{FAMILY}-0021", FAMILY,
        prompt=(
            "Implement a Python function resolve_json_pointer(document, "
            "pointer) evaluating an RFC 6901 JSON Pointer against a document "
            "built from dicts, lists, strings, numbers, booleans and None. "
            "The pointer is a string. The empty string refers to the whole "
            "document. Otherwise the pointer must begin with '/' and is split "
            "on '/' into reference tokens; within a token '~1' decodes to '/' "
            "and '~0' decodes to '~', and the two must be applied in that "
            "order so that '~01' decodes to '~1'. A token addressing a dict "
            "selects that exact key, including the empty key. A token "
            "addressing a list must be either '0' or a non-zero digit string "
            "with no leading zeros. Raise ValueError if the pointer is "
            "syntactically invalid, including a non-empty pointer that does "
            "not start with '/' and a '~' not followed by '0' or '1'. Raise "
            "LookupError if the pointer is well-formed but does not resolve, "
            "including a list index out of range, the token '-', and any "
            "token applied to a value that is not a dict or list."
        ),
        validator=LOAD_CANDIDATE + require("resolve_json_pointer") + r'''
# The example document from RFC 6901 section 5, verbatim.
DOC = {
    "foo": ["bar", "baz"],
    "": 0,
    "a/b": 1,
    "c%d": 2,
    "e^f": 3,
    "g|h": 4,
    "i\\j": 5,
    "k\"l": 6,
    " ": 7,
    "m~n": 8,
}

assert resolve_json_pointer(DOC, "") == DOC
assert resolve_json_pointer(DOC, "/foo") == ["bar", "baz"]
assert resolve_json_pointer(DOC, "/foo/0") == "bar"
assert resolve_json_pointer(DOC, "/foo/1") == "baz"
assert resolve_json_pointer(DOC, "/") == 0, 'the empty key is a real key'
assert resolve_json_pointer(DOC, "/a~1b") == 1
assert resolve_json_pointer(DOC, "/c%d") == 2
assert resolve_json_pointer(DOC, "/e^f") == 3
assert resolve_json_pointer(DOC, "/g|h") == 4
assert resolve_json_pointer(DOC, "/i\\j") == 5
assert resolve_json_pointer(DOC, "/k\"l") == 6
assert resolve_json_pointer(DOC, "/ ") == 7
assert resolve_json_pointer(DOC, "/m~0n") == 8

# Escape ORDER. Decoding '~0' first would turn '~01' into '~1' and then into
# '/', reaching the wrong member; the RFC fixes the order for this reason.
assert resolve_json_pointer({"~1": "tilde-one"}, "/~01") == "tilde-one"
assert resolve_json_pointer({"/": "slash"}, "/~1") == "slash"
assert resolve_json_pointer({"~": "tilde"}, "/~0") == "tilde"

NESTED = {"a": {"b": [{"c": [10, 20, 30]}]}, "n": None, "t": True, "s": "xy"}
assert resolve_json_pointer(NESTED, "/a/b/0/c/2") == 30
assert resolve_json_pointer(NESTED, "/n") is None
assert resolve_json_pointer(NESTED, "/t") is True


def raises(kind, pointer, why, document=NESTED):
    try:
        resolve_json_pointer(document, pointer)
    except kind:
        return
    except Exception as exc:
        raise AssertionError(
            f'{pointer!r}: raised {type(exc).__name__}, want '
            f'{kind.__name__} ({why})') from None
    raise AssertionError(f'{pointer!r}: accepted, want {kind.__name__} ({why})')


# Syntax is a different fault from a miss, and the two have different fixes.
raises(ValueError, "foo", 'no leading slash')
raises(ValueError, "a/b", 'no leading slash')
raises(ValueError, "/~2", 'a tilde must be followed by 0 or 1')
raises(ValueError, "/~", 'trailing tilde')
raises(ValueError, "/a~", 'trailing tilde in a token')

# Well-formed but unresolvable.
raises(LookupError, "/missing", 'absent key')
raises(LookupError, "/a/b/1", 'list index out of range')
raises(LookupError, "/a/b/-", 'the - token never resolves on read')
raises(LookupError, "/a/b/00", 'leading zero is not a valid index')
raises(LookupError, "/a/b/01", 'leading zero is not a valid index')
raises(LookupError, "/a/b/+1", 'a sign is not a valid index')
raises(LookupError, "/a/b/ 0", 'whitespace is not a valid index')
raises(LookupError, "/a/b/x", 'a non-numeric token on a list')
raises(LookupError, "/n/anything", 'None has no members')
raises(LookupError, "/t/anything", 'a bool has no members')

# A string is not a container, so indexing one must MISS rather than slice.
# A resolver written with plain subscripting returns 'x' here instead.
raises(LookupError, "/s/0", 'a string is not an addressable container')
''',
    ),
    task(
        f"{FAMILY}-0022", FAMILY,
        prompt=(
            "Implement a Python function negotiate_language(header, "
            "supported) implementing HTTP Accept-Language negotiation. header "
            "is the field value or None; supported is a list of language tags "
            "the server can serve, in the server's own order of preference. "
            "Return the best supported tag, preserving the spelling given in "
            "supported, or None if nothing is acceptable. Each element of the "
            "header is a tag or '*' with an optional ';q=' weight between 0 "
            "and 1 with up to three decimal places, defaulting to 1. Matching "
            "is case-insensitive. A header tag matches a supported tag that "
            "equals it, or that extends it at a subtag boundary, so 'en' "
            "matches 'en-US' but not 'english'. A weight of 0 makes a tag "
            "unacceptable, and an explicit 0 for a specific tag overrides a "
            "wildcard that would otherwise allow it. Among acceptable tags "
            "choose the highest weight; break a tie on weight by the more "
            "specific header match, preferring an exact tag over a prefix and "
            "a prefix over '*'; break any remaining tie by the order of "
            "supported. Return None when header is None or empty, since the "
            "caller treats that as no preference, and ignore elements that "
            "are syntactically invalid."
        ),
        validator=LOAD_CANDIDATE + require("negotiate_language") + r'''
def check(header, supported, expected):
    got = negotiate_language(header, supported)
    assert got == expected, (
        f'{header!r} over {supported!r}: got {got!r}, want {expected!r}')


check('en-US', ['en-US', 'de'], 'en-US')
check('en-US,en;q=0.9', ['en-GB', 'en-US'], 'en-US')
check('fr;q=0.9,en;q=0.8', ['en-US', 'de'], 'en-US')
check('de,en;q=0.7', ['en-US', 'de-DE'], 'de-DE')

# Prefix matching happens at a SUBTAG BOUNDARY. A startswith() check accepts
# 'english' for 'en', which is a different language.
check('en', ['en-US'], 'en-US')
check('en', ['english'], None)
check('en', ['en'], 'en')
check('en-US', ['en'], None)
check('zh', ['zh-Hant-TW'], 'zh-Hant-TW')

# Case-insensitive matching, but the SERVER's spelling is returned.
check('EN-us', ['en-US'], 'en-US')
check('en-us', ['EN-US'], 'EN-US')
check('DE', ['de-DE'], 'de-DE')

# Weights.
check('en;q=0.5,de;q=0.8', ['en-US', 'de-DE'], 'de-DE')
check('en;q=0.5, de;q=0.8', ['en-US', 'de-DE'], 'de-DE')
check('en;q=1,de;q=0.999', ['de-DE', 'en-US'], 'en-US')
check('en;q=0.001,de;q=0', ['de-DE', 'en-US'], 'en-US')

# q=0 is a refusal, not a low score.
check('en;q=0', ['en-US'], None)
check('en;q=0,de', ['en-US', 'de-DE'], 'de-DE')
check('en;q=0.000', ['en-US'], None)

# The wildcard, and a specific refusal that overrides it.
check('*', ['de', 'fr'], 'de')
check('*', ['fr', 'de'], 'fr')
check('de,*;q=0.1', ['fr', 'de'], 'de')
check('*;q=0.5,en;q=0.9', ['de', 'en-US'], 'en-US')
check('en;q=0,*', ['en-US', 'de'], 'de')
check('en;q=0,*;q=0.5', ['en-US'], None)

# Tie on weight: exact beats prefix, prefix beats wildcard.
check('en-US;q=0.8,en;q=0.8', ['en-US'], 'en-US')
check('en;q=0.8,*;q=0.8', ['de', 'en-GB'], 'en-GB')

# Tie on weight AND specificity: the server's order decides.
check('en,de', ['de-DE', 'en-US'], 'de-DE')
check('en,de', ['en-US', 'de-DE'], 'en-US')

# No preference expressed.
check(None, ['en', 'de'], None)
check('', ['en', 'de'], None)

# Nothing acceptable.
check('fr', ['en-US', 'de'], None)
check('en', [], None)

# Invalid elements are ignored, and must not take the valid ones with them.
check('en;q=bad,de', ['en-US', 'de-DE'], 'de-DE')
check('en;q=2,de', ['en-US', 'de-DE'], 'de-DE')
check('en;q=-1,de', ['en-US', 'de-DE'], 'de-DE')
check(';q=0.5,de', ['en-US', 'de-DE'], 'de-DE')
check('en;;q=0.5,de', ['en-US', 'de-DE'], 'de-DE')
''',
    ),
    task(
        f"{FAMILY}-0023", FAMILY,
        prompt=(
            "Implement a Python function decode_base32(text) decoding RFC "
            "4648 base32 with the standard alphabet A-Z and 2-7 and '=' "
            "padding, without using the base64 module. text is a str. Return "
            "the decoded bytes. The input length must be a multiple of 8. "
            "Padding may only appear as a suffix, and only in the lengths a "
            "real encoding can produce: 6, 4, 3 or 1 padding characters, "
            "corresponding to 1, 2, 3 and 4 input bytes in the final group. "
            "Raise ValueError on a character outside the alphabet, on a "
            "length that is not a multiple of 8, on a padding count that no "
            "encoding produces, on padding that is not a suffix, and on a "
            "final group whose trailing bits are non-zero, since those bits "
            "carry no data and a decoder that discards them accepts several "
            "distinct strings for the same bytes. Return b'' for empty input."
        ),
        validator=LOAD_CANDIDATE + require("decode_base32") + r'''
source = RESPONSE_TEXT
for banned in ('base64', 'b32decode'):
    assert banned not in source, f'the prompt forbids {banned}'

# The test vectors from RFC 4648 section 10.
assert decode_base32('') == b''
assert decode_base32('MY======') == b'f'
assert decode_base32('MZXQ====') == b'fo'
assert decode_base32('MZXW6===') == b'foo'
assert decode_base32('MZXW6YQ=') == b'foob'
assert decode_base32('MZXW6YTB') == b'fooba'
assert decode_base32('MZXW6YTBOI======') == b'foobar'

# Every byte value round-trips against the reference encoder, at every
# length that exercises a different padding count.
import base64 as _reference
for length in range(0, 17):
    for start in (0, 71, 200):
        raw = bytes((start + i) % 256 for i in range(length))
        encoded = _reference.b32encode(raw).decode('ascii')
        assert decode_base32(encoded) == raw, f'round trip failed for {raw!r}'


def rejects(text, why):
    try:
        decode_base32(text)
    except ValueError:
        return
    raise AssertionError(f'accepted {text!r}: {why}')


# Length must be a multiple of 8.
rejects('MZXW6YT', 'length 7 is not a multiple of 8')
rejects('M', 'length 1 is not a multiple of 8')
rejects('MZXW6YTBO', 'length 9 is not a multiple of 8')

# Alphabet.
rejects('mzxw6ytb', 'lowercase is not the standard alphabet')
rejects('MZXW6YT0', '0 is not in the alphabet')
rejects('MZXW6YT1', '1 is not in the alphabet')
rejects('MZXW6YT8', '8 is not in the alphabet')
rejects('MZXW6YT+', '+ is not in the alphabet')
rejects('MZXW 6YT', 'space is not in the alphabet')

# Padding counts an encoder never emits.
rejects('MZXW6Y==', '2 padding characters are impossible')
rejects('MZX=====', '5 padding characters are impossible')
rejects('M=======', '7 padding characters are impossible')
rejects('========', '8 padding characters are impossible')

# Padding must be a suffix.
rejects('MZ=XW6YT', 'padding in the middle')
rejects('=MZXW6YT', 'leading padding')
rejects('MY======MY======', 'padding inside a multi-group input')

# Non-canonical trailing bits. 'MZXW6YR=' differs from the canonical
# 'MZXW6YQ=' only in bits that carry no data; accepting both means two
# distinct strings decode to b'foob', which breaks any signature or cache
# key computed over the encoded form.
rejects('MZXW6YR=', 'non-zero trailing bits')
rejects('MZXW6YZ=', 'non-zero trailing bits')
rejects('MZ======', 'non-zero trailing bits')
rejects('MZXR====', 'non-zero trailing bits')
# A legal padding count does NOT make a group legal: 'MZXW====' has the four
# data characters a 2-byte group needs, and still carries 4 junk bits.
rejects('MZXW====', 'legal padding count, non-zero trailing bits')
''',
    ),
    task(
        f"{FAMILY}-0024", FAMILY,
        prompt=(
            "Implement a Python function match_glob(pattern, path) matching a "
            "slash-separated path against a shell-style glob, returning a "
            "bool. Do not use fnmatch, glob, pathlib or re. '?' matches "
            "exactly one character other than '/'. '*' matches any run of "
            "characters, including none, other than '/'. '**' as a COMPLETE "
            "path segment matches zero or more whole segments, so 'a/**/b' "
            "matches 'a/b' as well as 'a/x/y/b'. A character class in square "
            "brackets matches one character other than '/', supports ranges "
            "with '-', is negated by a leading '!' or '^', treats a ']' "
            "immediately after the opening bracket or the negation as a "
            "literal, and treats a '-' first or last as a literal. A "
            "backslash escapes the next character so it is matched literally. "
            "Matching is over the whole path, not a prefix. Raise ValueError "
            "on an unterminated class, a trailing backslash, and on '**' "
            "appearing adjacent to other characters within a segment. That "
            "last check is made on the segment's unescaped structure, so an "
            "escaped star is an ordinary literal and does not combine with a "
            "neighbouring star to form a '**': the escapes must be resolved "
            "before the segment is judged."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("match_glob") + r'''
source = RESPONSE_TEXT
for banned in ('fnmatch', 'import re', 'import glob', 'pathlib'):
    assert banned not in source, f'the prompt forbids {banned}'


def yes(pattern, path):
    assert match_glob(pattern, path) is True, f'{pattern!r} should match {path!r}'


def no(pattern, path):
    assert match_glob(pattern, path) is False, (
        f'{pattern!r} should not match {path!r}')


# Literals, and whole-path matching.
yes('a', 'a')
no('a', 'ab')
no('a', 'ba')
no('a/b', 'a/b/c')
yes('a/b/c', 'a/b/c')

# '?' and '*' stop at the separator. This is the property that makes a glob
# a path matcher rather than a string matcher.
yes('a?c', 'abc')
no('a?c', 'ac')
no('a?c', 'abbc')
no('a?c', 'a/c')
yes('*.txt', 'notes.txt')
yes('*', 'anything')
yes('*', '')
no('*', 'a/b')
no('*.txt', 'sub/notes.txt')
yes('a/*/c', 'a/b/c')
no('a/*/c', 'a/b/x/c')
yes('*/*', 'a/b')

# Multiple stars in one segment need real backtracking.
yes('*a*b*', 'xxayybzz')
no('*a*b*', 'xxbyyazz')
yes('a*a*a', 'aaa')
yes('*x*x*x*', 'xxx')

# '**' spans segments, including zero of them.
yes('a/**/b', 'a/b')
yes('a/**/b', 'a/x/b')
yes('a/**/b', 'a/x/y/z/b')
no('a/**/b', 'a/x/y/z/c')
yes('**', 'a')
yes('**', 'a/b/c')
yes('**/b', 'b')
yes('**/b', 'a/b')
yes('a/**', 'a')
yes('a/**', 'a/b/c')
yes('**/*.txt', 'notes.txt')
yes('**/*.txt', 'a/b/notes.txt')
no('**/*.txt', 'a/b/notes.md')
yes('src/**/test/**/*.py', 'src/test/a.py')
yes('src/**/test/**/*.py', 'src/x/test/y/z/a.py')

# Character classes.
yes('[abc]', 'b')
no('[abc]', 'd')
yes('[a-z]', 'q')
no('[a-z]', 'Q')
yes('[a-zA-Z0-9]', 'M')
yes('[!abc]', 'd')
no('[!abc]', 'a')
yes('[^abc]', 'd')
no('[^abc]', 'a')
no('[!abc]', '/')
yes('[]]', ']')
yes('[!]]', 'a')
no('[!]]', ']')
yes('[-a]', '-')
yes('[a-]', '-')
yes('[a-]', 'a')
yes('file[0-9].txt', 'file7.txt')
no('file[0-9].txt', 'filex.txt')

# Escapes make a metacharacter literal.
yes(r'a\*b', 'a*b')
no(r'a\*b', 'axb')
yes(r'a\?b', 'a?b')
yes(r'\[abc\]', '[abc]')
yes(r'a\\b', 'a\\b')
yes(r'\**', '*xyz')

# Errors.
def rejects(pattern, path, why):
    try:
        match_glob(pattern, path)
    except ValueError:
        return
    raise AssertionError(f'accepted {pattern!r}: {why}')


rejects('[abc', 'a', 'unterminated class')
rejects('[!', 'a', 'unterminated class')
rejects('[]', 'a', 'unterminated class: ] after [ is a literal')
rejects('a\\', 'a', 'trailing backslash')
rejects('a**', 'ab', '** adjacent to other characters in a segment')
rejects('**b', 'ab', '** adjacent to other characters in a segment')
rejects('a/**b/c', 'a/b/c', '** adjacent to other characters in a segment')
rejects('a/x**/c', 'a/xy/c', '** adjacent to other characters in a segment')
''',
    ),
    task(
        f"{FAMILY}-0025", FAMILY,
        prompt=(
            "Implement a Python function parse_media_type(value) parsing an "
            "HTTP media type such as a Content-Type field value. Return a "
            "tuple (type, subtype, parameters). type and subtype are "
            "lowercased str. parameters is a dict mapping lowercased "
            "parameter names to their values as str, with quoted-string "
            "values unquoted and their backslash escapes resolved; parameter "
            "values are NOT lowercased. Optional whitespace is permitted "
            "around the semicolons and around the '=' is NOT permitted. When "
            "a parameter name repeats, the first occurrence wins. Raise "
            "ValueError if the type or subtype is missing or contains a "
            "character outside the HTTP token set, if a parameter has no "
            "value, if a quoted string is unterminated, or if a parameter "
            "name is not a valid token."
        ),
        validator=LOAD_CANDIDATE + require("parse_media_type") + r'''
def check(value, expected):
    got = parse_media_type(value)
    assert got == expected, f'{value!r}: got {got!r}, want {expected!r}'


check('text/plain', ('text', 'plain', {}))
check('TEXT/PLAIN', ('text', 'plain', {}))
check('application/json', ('application', 'json', {}))
check('application/vnd.api+json', ('application', 'vnd.api+json', {}))
check('*/*', ('*', '*', {}))

# Parameters. The NAME lowercases, the VALUE does not: a boundary and a
# filename are case-sensitive, and folding them corrupts the message.
check('text/plain;charset=UTF-8', ('text', 'plain', {'charset': 'UTF-8'}))
check('text/plain; charset=UTF-8', ('text', 'plain', {'charset': 'UTF-8'}))
check('text/plain ; charset=UTF-8', ('text', 'plain', {'charset': 'UTF-8'}))
check('text/plain;CHARSET=UTF-8', ('text', 'plain', {'charset': 'UTF-8'}))
check('text/plain;charset=utf-8', ('text', 'plain', {'charset': 'utf-8'}))
check(
    'multipart/form-data; boundary=AaB03x',
    ('multipart', 'form-data', {'boundary': 'AaB03x'}))
check(
    'text/plain;a=1;b=2',
    ('text', 'plain', {'a': '1', 'b': '2'}))

# Quoted strings, including escapes and metacharacters that would otherwise
# terminate the parse.
check('text/plain;charset="UTF-8"', ('text', 'plain', {'charset': 'UTF-8'}))
check('text/plain;x=""', ('text', 'plain', {'x': ''}))
check('text/plain;x="a;b"', ('text', 'plain', {'x': 'a;b'}))
check('text/plain;x="a=b"', ('text', 'plain', {'x': 'a=b'}))
check('text/plain;x="a b"', ('text', 'plain', {'x': 'a b'}))
check(r'text/plain;x="a\"b"', ('text', 'plain', {'x': 'a"b'}))
check(r'text/plain;x="a\\b"', ('text', 'plain', {'x': 'a\\b'}))
check(r'text/plain;x="\a"', ('text', 'plain', {'x': 'a'}))
check(
    'multipart/form-data; boundary="a;b=c"; charset=utf-8',
    ('multipart', 'form-data', {'boundary': 'a;b=c', 'charset': 'utf-8'}))

# First occurrence wins: taking the last lets a request smuggle a second
# charset past a filter that inspected the first.
check('text/plain;charset=utf-8;charset=ascii',
      ('text', 'plain', {'charset': 'utf-8'}))
check('text/plain;A=1;a=2', ('text', 'plain', {'a': '1'}))


def rejects(value, why):
    try:
        parse_media_type(value)
    except ValueError:
        return
    raise AssertionError(f'accepted {value!r}: {why}')


rejects('', 'empty')
rejects('text', 'no subtype')
rejects('text/', 'empty subtype')
rejects('/plain', 'empty type')
rejects('/', 'both empty')
rejects('text/plain/extra', 'a slash is not a token character')
rejects('te xt/plain', 'space is not a token character')
rejects('text/pl ain', 'space is not a token character')
rejects('text@x/plain', '@ is not a token character')
rejects('text/plain;charset', 'parameter with no value')
rejects('text/plain;charset=', 'parameter with an empty unquoted value')
rejects('text/plain;=utf-8', 'parameter with no name')
rejects('text/plain;charset="utf-8', 'unterminated quoted string')
rejects(r'text/plain;charset="utf-8\"', 'unterminated: the quote is escaped')
rejects('text/plain;char set=utf-8', 'space in a parameter name')
rejects('text/plain;charset =utf-8', 'space before = is not permitted')
rejects('text/plain;charset= utf-8', 'space after = is not permitted')
''',
    ),
    task(
        f"{FAMILY}-0026", FAMILY,
        prompt=(
            "Implement a Python function parse_rfc3339(text) parsing an RFC "
            "3339 timestamp and returning the number of seconds since the "
            "Unix epoch as a float, computed without the datetime, time or "
            "calendar modules. The format is a four-digit year, '-', "
            "two-digit month, '-', two-digit day, then 'T' or 't' as the "
            "separator, then two-digit hour, ':', minute, ':', second, an "
            "optional fractional part of a '.' followed by one or more "
            "digits, and finally an offset that is 'Z', 'z', or a sign "
            "followed by two-digit hours, ':' and two-digit minutes. The "
            "result is the instant in UTC, so a positive offset is "
            "subtracted. Validate the calendar: reject a month outside 1 to "
            "12, a day outside the real length of that month in that year "
            "under the proleptic Gregorian leap rule, an hour above 23, and a "
            "minute or second above 59. Reject a leap second value of 60, and "
            "reject an offset whose minutes exceed 59. Raise ValueError on "
            "any input that does not match this grammar exactly, including "
            "missing padding, a missing offset, and trailing characters."
        ),
        validator=LOAD_CANDIDATE + require("parse_rfc3339") + r'''
source = RESPONSE_TEXT
for banned in ('datetime', 'import time', 'import calendar', 'dateutil'):
    assert banned not in source, f'the prompt forbids {banned}'


def check(text, expected):
    got = parse_rfc3339(text)
    # Prove the SHAPE before doing arithmetic on it. A candidate whose
    # parse_rfc3339 falls through and returns None makes `abs(got - expected)`
    # raise TypeError in validator frames alone, which the contract scores
    # validator_error -- a harness fault that blocks admission -- when it is
    # an ordinary wrong answer.
    assert isinstance(got, (int, float)) and not isinstance(got, bool), (
        f'{text!r}: parse_rfc3339 returned {got!r}, want a number')
    assert abs(got - expected) < 1e-6, (
        f'{text!r}: got {got!r}, want {expected!r}')


check('1970-01-01T00:00:00Z', 0.0)
check('1970-01-01t00:00:00z', 0.0)
check('1970-01-02T00:00:00Z', 86400.0)
check('1969-12-31T23:59:59Z', -1.0)
check('2000-01-01T00:00:00Z', 946684800.0)
check('2001-09-09T01:46:40Z', 1000000000.0)
check('2038-01-19T03:14:08Z', 2147483648.0)
check('1900-01-01T00:00:00Z', -2208988800.0)

# Offsets: a positive offset means local time is AHEAD, so the UTC instant
# is EARLIER. Adding instead of subtracting is the classic sign inversion.
check('1970-01-01T01:00:00+01:00', 0.0)
check('1969-12-31T23:00:00-01:00', 0.0)
check('1996-12-19T16:39:57-08:00', 851042397.0)
check('1970-01-01T00:00:00+00:00', 0.0)
check('1970-01-01T00:00:00-00:00', 0.0)
check('1970-01-01T05:30:00+05:30', 0.0)
check('1970-01-01T00:00:00+14:00', -50400.0)

# Fractional seconds, at several precisions.
check('1970-01-01T00:00:00.5Z', 0.5)
check('1970-01-01T00:00:00.25Z', 0.25)
check('1970-01-01T00:00:00.000Z', 0.0)
check('1970-01-01T00:00:00.123456Z', 0.123456)
check('1970-01-01T00:00:00.000000001Z', 0.000000001)

# The leap rule: divisible by 4, except centuries, except those by 400.
check('2000-02-29T00:00:00Z', 951782400.0)
check('2024-02-29T00:00:00Z', 1709164800.0)
check('1996-02-29T00:00:00Z', 825552000.0)


def rejects(text, why):
    try:
        parse_rfc3339(text)
    except ValueError:
        return
    raise AssertionError(f'accepted {text!r}: {why}')


# 1900 and 2100 are NOT leap years; 2000 is. A `year % 4` test accepts the
# first two and a `year % 100` test rejects the third.
rejects('1900-02-29T00:00:00Z', '1900 is not a leap year')
rejects('2100-02-29T00:00:00Z', '2100 is not a leap year')
rejects('2023-02-29T00:00:00Z', '2023 is not a leap year')
rejects('2023-02-30T00:00:00Z', 'February never has 30 days')
rejects('2023-04-31T00:00:00Z', 'April has 30 days')
rejects('2023-06-31T00:00:00Z', 'June has 30 days')
rejects('2023-09-31T00:00:00Z', 'September has 30 days')
rejects('2023-11-31T00:00:00Z', 'November has 30 days')
rejects('2023-01-32T00:00:00Z', 'January has 31 days')
rejects('2023-00-10T00:00:00Z', 'month 0')
rejects('2023-13-10T00:00:00Z', 'month 13')
rejects('2023-01-00T00:00:00Z', 'day 0')
rejects('2023-01-01T24:00:00Z', 'hour 24')
rejects('2023-01-01T00:60:00Z', 'minute 60')
rejects('2023-01-01T00:00:60Z', 'a leap second is rejected by this contract')
rejects('2023-01-01T00:00:61Z', 'second 61')

# Grammar.
rejects('', 'empty')
rejects('2023-01-01', 'no time or offset')
rejects('2023-01-01T00:00:00', 'no offset')
rejects('2023-1-01T00:00:00Z', 'month is not zero-padded')
rejects('2023-01-1T00:00:00Z', 'day is not zero-padded')
rejects('23-01-01T00:00:00Z', 'two-digit year')
rejects('2023-01-01 00:00:00Z', 'space instead of T')
rejects('2023-01-01X00:00:00Z', 'invalid separator')
rejects('2023-01-01T00:00:00.Z', 'a dot with no digits')
rejects('2023-01-01T00:00:00Z ', 'trailing whitespace')
rejects('2023-01-01T00:00:00ZZ', 'trailing character')
rejects('2023-01-01T00:00:00+0100', 'offset needs a colon')
rejects('2023-01-01T00:00:00+01', 'offset needs minutes')
rejects('2023-01-01T00:00:00+01:60', 'offset minutes above 59')
rejects('2023-01-01T00:00:00%01:00', 'invalid offset sign')
rejects('+2023-01-01T00:00:00Z', 'leading sign on the year')
''',
    ),
]
