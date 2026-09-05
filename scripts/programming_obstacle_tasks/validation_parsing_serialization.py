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
        validator=LOAD_CANDIDATE + require("parse_multipart") + r'''
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
]
