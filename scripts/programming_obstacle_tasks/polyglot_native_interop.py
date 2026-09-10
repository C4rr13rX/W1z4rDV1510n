"""Held-out tasks: polyglot, native, and interoperability work.

Interoperability defects share one shape: both sides run, neither crashes,
and they disagree about what the bytes mean. The disagreement is usually
invisible in the case an author tests by hand -- ASCII text, a small positive
integer, a struct whose fields happen to be the same width -- and shows up
only where the two representations diverge. So every validator here drives
the divergent case rather than the agreeing one.

The specific divergences asserted below are all rules a foreign standard
fixes, not rules the prompt's examples define: the trailing padding C adds so
an array of a struct stays aligned, protobuf's ten-byte encoding of a
negative varint, the surrogate pair that makes one emoji two units of a Java
string, the logical shift an unsigned type takes where Python's operator is
always arithmetic, the sign bit that distinguishes -0.0 from 0.0, the
continuation byte a truncating copy must not cut, FILETIME's 1601 epoch and
100-nanosecond tick, JNI's internal class-name form, MSB-first bit-field
order, and the promotion of a float to a double when it crosses into a
variadic call.

Every one of those has a plausible wrong answer that passes an ASCII,
positive, single-field, aligned smoke test. The paired mutations in
`tests/obstacle_references.py` reintroduce exactly those wrong answers.

No task here compiles or links anything. A validator that shelled out to a C
toolchain would make the verdict depend on which compiler a machine has,
which is the irreproducible case the acceptance contract refuses; the
capability under test is agreeing with the foreign representation, and that
is decidable from Python against the standard alone.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import (
    LOAD_CANDIDATE,
    SHAPE_GUARDS,
    require,
)

FAMILY = "polyglot_native_interop"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function layout(fields, pack=None) computing "
            "the memory layout a C compiler gives a struct. fields is a list "
            "of (name, type) pairs where type is one of int8, uint8, int16, "
            "uint16, int32, uint32, int64, uint64, float, double or pointer, "
            "with sizes 1, 1, 2, 2, 4, 4, 8, 8, 4, 8 and 8 and a natural "
            "alignment equal to the size. Each member is placed at the next "
            "offset that is a multiple of its alignment. The struct's own "
            "alignment is the largest member alignment, or 1 when there are "
            "no members, and its size is the end of the last member rounded "
            "up to that alignment, so an array of the struct keeps every "
            "element aligned. When pack is given, every member's alignment "
            "is the smaller of its natural alignment and pack, and so is the "
            "struct's. Return {'offsets': {name: offset}, 'size': int, "
            "'alignment': int}. Raise ValueError for an unknown type, a "
            "repeated field name, or a pack that is not a positive power of "
            "two."
        ),
        validator=LOAD_CANDIDATE + require("layout") + r'''
assert layout([]) == {'offsets': {}, 'size': 0, 'alignment': 1}
assert layout([('only', 'int8')]) == \
    {'offsets': {'only': 0}, 'size': 1, 'alignment': 1}

# Trailing padding is the rule an author who lays fields out by hand almost
# always drops: this struct ends at byte 5 and occupies 8, because an array
# of it has to keep every int32 aligned.
assert layout([('count', 'int32'), ('flag', 'int8')]) == \
    {'offsets': {'count': 0, 'flag': 4}, 'size': 8, 'alignment': 4}, \
    'the struct size must round up to the struct alignment'

assert layout([('a', 'int8'), ('b', 'int32'), ('c', 'int8')]) == \
    {'offsets': {'a': 0, 'b': 4, 'c': 8}, 'size': 12, 'alignment': 4}

# Packing changes both the interior padding and the trailing padding.
assert layout([('a', 'int8'), ('b', 'int32'), ('c', 'int8')], 1) == \
    {'offsets': {'a': 0, 'b': 1, 'c': 5}, 'size': 6, 'alignment': 1}
assert layout([('a', 'int8'), ('b', 'int32'), ('c', 'int8')], 2) == \
    {'offsets': {'a': 0, 'b': 2, 'c': 6}, 'size': 8, 'alignment': 2}, \
    'pack caps each alignment rather than removing alignment'
assert layout([('a', 'int8'), ('b', 'int32'), ('c', 'int8')], 8) == \
    layout([('a', 'int8'), ('b', 'int32'), ('c', 'int8')]), \
    'a pack larger than every natural alignment changes nothing'

assert layout([('handle', 'pointer'), ('tag', 'int8')]) == \
    {'offsets': {'handle': 0, 'tag': 8}, 'size': 16, 'alignment': 8}
assert layout([('x', 'double'), ('y', 'int32'), ('z', 'int32')]) == \
    {'offsets': {'x': 0, 'y': 8, 'z': 12}, 'size': 16, 'alignment': 8}, \
    'two int32 fill the tail of an 8-aligned struct without extra padding'
assert layout([('y', 'int32'), ('x', 'double'), ('z', 'int32')]) == \
    {'offsets': {'y': 0, 'x': 8, 'z': 16}, 'size': 24, 'alignment': 8}, \
    'reordering the same fields changes the size'

for bad in (
    ([('a', 'int24')], None),
    ([('a', 'int8'), ('a', 'int8')], None),
    ([('a', 'int8')], 0),
    ([('a', 'int8')], 3),
    ([('a', 'int8')], -2),
):
    try:
        layout(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'layout{bad} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement Python functions encode_varint(value), "
            "decode_varint(data, offset=0), encode_zigzag(value) and "
            "decode_zigzag(value) implementing protocol buffers' base-128 "
            "wire encoding. encode_varint returns bytes holding the value "
            "seven bits at a time, least significant group first, with the "
            "high bit of every byte except the last set. A negative value is "
            "encoded as its 64-bit two's complement, which always takes ten "
            "bytes. A value outside -2**63 .. 2**64-1 raises ValueError. "
            "decode_varint reads one varint starting at offset and returns "
            "(value, next_offset) with the value read as unsigned; it raises "
            "ValueError when the data ends mid-varint or when the varint "
            "runs past ten bytes. encode_zigzag maps a signed integer onto a "
            "non-negative one so small magnitudes stay short: 0, -1, 1, -2, "
            "2 become 0, 1, 2, 3, 4. decode_zigzag inverts it."
        ),
        validator=LOAD_CANDIDATE + require("encode_varint")
        + require("decode_varint") + require("encode_zigzag")
        + require("decode_zigzag") + r'''
assert encode_varint(0) == b'\x00'
assert encode_varint(1) == b'\x01'
assert encode_varint(127) == b'\x7f'
assert encode_varint(128) == b'\x80\x01', 'the low group comes first'
assert encode_varint(300) == b'\xac\x02'
assert encode_varint(2 ** 32) == b'\x80\x80\x80\x80\x10'

# A negative varint is its 64-bit two's complement, so -1 is the longest
# encoding there is rather than the shortest. Anything that encodes a
# magnitude and a sign is caught here and interoperates with nothing.
assert encode_varint(-1) == b'\xff' * 9 + b'\x01', \
    'a negative varint is the 64-bit two-complement, ten bytes long'
assert encode_varint(-2) == b'\xfe' + b'\xff' * 8 + b'\x01'
assert len(encode_varint(-(2 ** 63))) == 10

assert decode_varint(b'\x00') == (0, 1)
assert decode_varint(b'\xac\x02') == (300, 2)
assert decode_varint(b'\xac\x02\xff', 0) == (300, 2), \
    'decoding stops at the end of the varint, not the end of the buffer'
assert decode_varint(b'\x99\x01\xac\x02', 2) == (300, 4)
assert decode_varint(encode_varint(-1)) == (2 ** 64 - 1, 10), \
    'the decoded value is unsigned'

for value in (0, 1, 127, 128, 300, 2 ** 21, 2 ** 63 - 1, 2 ** 64 - 1):
    assert decode_varint(encode_varint(value)) == (value, len(encode_varint(value)))

assert [encode_zigzag(n) for n in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]
assert encode_zigzag(2147483647) == 4294967294
assert encode_zigzag(-2147483648) == 4294967295
for n in (0, -1, 1, -2, 2, 12345, -12345, 2 ** 40, -(2 ** 40)):
    assert decode_zigzag(encode_zigzag(n)) == n, f'zigzag lost {n}'
    assert encode_zigzag(n) >= 0

for bad in (2 ** 64, -(2 ** 63) - 1):
    try:
        encode_varint(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'encode_varint({bad}) was accepted')

for bad_data in (b'', b'\x80', b'\xac', b'\xff' * 11):
    try:
        decode_varint(bad_data)
    except ValueError:
        pass
    else:
        raise AssertionError(f'decode_varint({bad_data!r}) was accepted')
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement Python functions utf16_length(text), "
            "encode_utf16_units(text), decode_utf16_units(units) and "
            "code_point_index(text, unit_index) for talking to a runtime "
            "whose strings are sequences of UTF-16 code units, such as Java "
            "or JavaScript. utf16_length returns how many code units the "
            "text occupies, which is one per code point below U+10000 and "
            "two above it. encode_utf16_units returns those units as a tuple "
            "of integers, encoding a code point at or above U+10000 as a "
            "high surrogate 0xD800 + ((cp - 0x10000) >> 10) followed by a "
            "low surrogate 0xDC00 + ((cp - 0x10000) & 0x3FF). "
            "decode_utf16_units inverts that and raises ValueError on a "
            "surrogate that is not part of a well-formed pair or a unit "
            "outside 0..0xFFFF. code_point_index returns how many code "
            "points precede a given code-unit offset, accepts an offset "
            "equal to the length, raises ValueError when the offset falls "
            "between the two halves of a surrogate pair, and raises "
            "IndexError when it is outside 0..utf16_length(text)."
        ),
        validator=LOAD_CANDIDATE + require("utf16_length")
        + require("encode_utf16_units") + require("decode_utf16_units")
        + require("code_point_index") + r'''
grin = '\U0001F600'
clef = '\U0001D11E'

assert utf16_length('') == 0
assert utf16_length('abc') == 3
assert utf16_length('é') == 1, 'a two-byte UTF-8 char is still one unit'
assert utf16_length('￿') == 1

# The length that disagrees with Python's: one emoji is two units, which is
# what the other runtime's length, index and substring operations use.
assert utf16_length(grin) == 2, 'an astral code point is two UTF-16 units'
assert utf16_length('a' + grin + 'b') == 4
assert utf16_length(grin * 3) == 6

assert encode_utf16_units('A') == (0x0041,)
assert encode_utf16_units(grin) == (0xD83D, 0xDE00)
assert encode_utf16_units(clef) == (0xD834, 0xDD1E)
assert encode_utf16_units('a' + grin) == (0x0061, 0xD83D, 0xDE00)

assert decode_utf16_units((0xD83D, 0xDE00)) == grin
assert decode_utf16_units(()) == ''
assert decode_utf16_units((0x0061, 0xD834, 0xDD1E, 0x0062)) == 'a' + clef + 'b'
for text in ('', 'abc', 'é', grin, 'a' + grin + 'b', clef + grin):
    assert decode_utf16_units(encode_utf16_units(text)) == text, \
        f'round trip lost {text!r}'

for bad_units in ((0xD83D,), (0xDE00,), (0xD83D, 0x0041),
                  (0xD83D, 0xD83D), (0x0041, 0xDE00), (0x10000,), (-1,)):
    try:
        decode_utf16_units(bad_units)
    except ValueError:
        pass
    else:
        raise AssertionError(f'decode_utf16_units({bad_units!r}) was accepted')

mixed = 'a' + grin + 'b'
assert code_point_index(mixed, 0) == 0
assert code_point_index(mixed, 1) == 1
assert code_point_index(mixed, 3) == 2
assert code_point_index(mixed, 4) == 3, 'the end offset is addressable'
try:
    code_point_index(mixed, 2)
except ValueError:
    pass
else:
    raise AssertionError('an offset inside a surrogate pair was accepted')

for bad_offset in (-1, 5):
    try:
        code_point_index(mixed, bad_offset)
    except IndexError:
        pass
    else:
        raise AssertionError(f'offset {bad_offset} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement Python functions wrap(value, bits, signed), "
            "add(a, b, bits, signed), mul(a, b, bits, signed) and "
            "shift_right(value, amount, bits, signed) giving Python integers "
            "the semantics of a fixed-width C integer type. bits is 8, 16, "
            "32 or 64 and anything else raises ValueError. wrap reduces a "
            "value modulo 2**bits and reinterprets it in the type's range, "
            "so an unsigned type yields 0 .. 2**bits-1 and a signed one "
            "-2**(bits-1) .. 2**(bits-1)-1. add and mul return the tuple "
            "(wrapped_result, overflowed) where overflowed says whether the "
            "exact mathematical result was outside the type's range. "
            "shift_right shifts by amount after wrapping the value: a signed "
            "type shifts arithmetically, keeping its sign bit, and an "
            "unsigned type shifts logically, bringing in zeros. An amount "
            "that is negative or not less than bits raises ValueError."
        ),
        validator=LOAD_CANDIDATE + require("wrap") + require("add")
        + require("mul") + require("shift_right") + r'''
assert wrap(5, 8, True) == 5
assert wrap(255, 8, True) == -1, 'the high bit is the sign bit'
assert wrap(128, 8, True) == -128
assert wrap(-1, 8, False) == 255
assert wrap(256, 8, False) == 0
assert wrap(2 ** 31, 32, True) == -2147483648
assert wrap(-(2 ** 63), 64, True) == -(2 ** 63)
assert wrap(2 ** 64 + 7, 64, False) == 7

assert add(1, 1, 8, True) == (2, False)
assert add(127, 1, 8, True) == (-128, True), 'signed overflow wraps and reports'
assert add(255, 1, 8, False) == (0, True)
assert add(-128, -1, 8, True) == (127, True)
assert add(-1, 1, 8, True) == (0, False)
assert add(2147483647, 1, 32, True) == (-2147483648, True)
assert add(2147483647, 1, 64, True) == (2147483648, False), \
    'the same addition does not overflow a wider type'

assert mul(2, 3, 8, True) == (6, False)
assert mul(1000, 1000, 16, True) == (16960, True)
assert mul(16, 16, 8, False) == (0, True)
assert mul(-1, -1, 8, True) == (1, False)

# The shift is where Python's operator and C's types disagree: Python's >>
# is always arithmetic and its integers never lose high bits, so an unsigned
# right shift written the obvious way keeps the sign it should have dropped.
assert shift_right(-8, 1, 8, True) == -4
assert shift_right(-8, 1, 8, False) == 124, \
    'an unsigned shift is logical: -8 wraps to 248 and 248 >> 1 is 124'
assert shift_right(-1, 3, 32, True) == -1, 'arithmetic shift keeps the sign'
assert shift_right(-1, 3, 32, False) == 536870911
assert shift_right(255, 4, 8, False) == 15
assert shift_right(255, 4, 8, True) == -1
assert shift_right(7, 0, 8, True) == 7

for bad in (
    (5, 7, True), (5, 0, True), (5, 128, False), (5, -8, True),
):
    try:
        wrap(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'wrap{bad} was accepted')

for bad in ((1, -1, 8, True), (1, 8, 8, True), (1, 9, 8, False)):
    try:
        shift_right(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'shift_right{bad} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement Python functions float_to_bits(value), "
            "bits_to_float(bits) and half_bits_to_float(bits) converting "
            "between floats and the IEEE 754 encodings a native ABI passes. "
            "float_to_bits returns the 32-bit unsigned integer holding the "
            "binary32 encoding of value, rounding to nearest even, and "
            "bits_to_float inverts it; a bits argument outside 0 .. 2**32-1 "
            "raises ValueError. half_bits_to_float decodes a 16-bit binary16 "
            "value to a Python float, where the sign is bit 15, the exponent "
            "is bits 14..10 with a bias of 15, and the significand is bits "
            "9..0: an exponent of 0 means a subnormal worth "
            "2**-14 * significand / 1024, an exponent of 31 with a zero "
            "significand means infinity and with a non-zero one means NaN, "
            "and any other exponent means 2**(exponent-15) * "
            "(1 + significand / 1024). A bits argument outside 0 .. 65535 "
            "raises ValueError. Signed zeros, subnormals, infinities and "
            "NaNs must all survive."
        ),
        validator=LOAD_CANDIDATE + require("float_to_bits")
        + require("bits_to_float") + require("half_bits_to_float") + r'''
import math

assert float_to_bits(0.0) == 0x00000000
assert float_to_bits(1.0) == 0x3F800000
assert float_to_bits(-2.0) == 0xC0000000
assert float_to_bits(0.5) == 0x3F000000
assert float_to_bits(float('inf')) == 0x7F800000
assert float_to_bits(float('-inf')) == 0xFF800000

# Negative zero is the value an "is it zero" shortcut destroys, and the C
# side can tell: it changes the sign of 1/x and of every product with it.
assert float_to_bits(-0.0) == 0x80000000, \
    'the sign bit of negative zero must survive'
assert math.copysign(1.0, bits_to_float(0x80000000)) == -1.0

assert bits_to_float(0x3F800000) == 1.0
assert bits_to_float(0xC0000000) == -2.0
assert bits_to_float(0x7F800000) == float('inf')
assert bits_to_float(0xFF800000) == float('-inf')
assert math.isnan(bits_to_float(0x7FC00000))

# The smallest binary32 subnormal, which a naive exponent formula rounds to
# zero because its stored exponent is 0 and its implicit leading bit is not.
assert bits_to_float(0x00000001) == 2.0 ** -149
assert float_to_bits(2.0 ** -149) == 1
assert bits_to_float(0x007FFFFF) == (2.0 ** -126) * (1 - 2.0 ** -23)
assert bits_to_float(0x00800000) == 2.0 ** -126

for value in (0.0, -0.0, 1.0, -1.0, 0.5, 2.0 ** -149, 2.0 ** -126,
              3.4028234663852886e+38, float('inf'), float('-inf')):
    restored = bits_to_float(float_to_bits(value))
    assert restored == value and \
        math.copysign(1.0, restored) == math.copysign(1.0, value), \
        f'round trip lost {value!r}'

assert half_bits_to_float(0x0000) == 0.0
assert math.copysign(1.0, half_bits_to_float(0x8000)) == -1.0
assert half_bits_to_float(0x3C00) == 1.0
assert half_bits_to_float(0xC000) == -2.0
assert half_bits_to_float(0x3D55) == 1.0 + 0x155 / 1024
assert half_bits_to_float(0x3555) == 0.25 * (1.0 + 0x155 / 1024), \
    'the stored exponent is biased by 15, so field 13 scales by 2**-2'
assert half_bits_to_float(0x0400) == 2.0 ** -14
assert half_bits_to_float(0x0001) == 2.0 ** -24, 'the smallest subnormal'
assert half_bits_to_float(0x03FF) == (2.0 ** -14) * (1023 / 1024)
assert half_bits_to_float(0x7BFF) == 65504.0, 'the largest finite binary16'
assert half_bits_to_float(0x7C00) == float('inf')
assert half_bits_to_float(0xFC00) == float('-inf')
assert math.isnan(half_bits_to_float(0x7E00))

for bad in (-1, 2 ** 32):
    try:
        bits_to_float(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'bits_to_float({bad}) was accepted')

for bad in (-1, 0x10000):
    try:
        half_bits_to_float(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'half_bits_to_float({bad}) was accepted')
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement Python functions to_c_string(text, capacity), "
            "from_c_string(buffer) and truncate_utf8(data, limit) for "
            "handing text to a C API across a fixed-size buffer. "
            "to_c_string returns exactly capacity bytes holding the UTF-8 "
            "encoding of text followed by a NUL terminator and then NUL "
            "padding; it raises ValueError when text contains a NUL, because "
            "the C side would stop there and silently receive a prefix, and "
            "when the encoding plus its terminator will not fit. "
            "from_c_string decodes the bytes before the first NUL and raises "
            "ValueError when the buffer contains no NUL at all. "
            "truncate_utf8 returns the longest prefix of data that is at "
            "most limit bytes and does not end in the middle of a multi-byte "
            "sequence, so the result always decodes; a negative limit raises "
            "ValueError."
        ),
        validator=LOAD_CANDIDATE + require("to_c_string")
        + require("from_c_string") + require("truncate_utf8") + r'''
assert to_c_string('hi', 8) == b'hi\x00\x00\x00\x00\x00\x00'
assert to_c_string('', 1) == b'\x00'
assert to_c_string('abc', 4) == b'abc\x00', 'the terminator uses the last byte'
assert to_c_string('é', 4) == b'\xc3\xa9\x00\x00'
assert len(to_c_string('x', 16)) == 16

# An embedded NUL is a truncation attack, not a character: the C side reads
# "safe" from "safe\0../../etc/passwd" and every later check sees the prefix.
try:
    to_c_string('safe\x00evil', 32)
except ValueError:
    pass
else:
    raise AssertionError('a string containing a NUL was accepted')

for bad in (('abc', 3), ('abc', 0), ('é', 2), ('', 0)):
    try:
        to_c_string(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'to_c_string{bad} was accepted')

assert from_c_string(b'hi\x00') == 'hi'
assert from_c_string(b'hi\x00trailing garbage') == 'hi'
assert from_c_string(b'\x00') == ''
assert from_c_string(b'\xc3\xa9\x00\x00') == 'é'
assert from_c_string(to_c_string('round trip', 64)) == 'round trip'
try:
    from_c_string(b'no terminator')
except ValueError:
    pass
else:
    raise AssertionError('an unterminated buffer was accepted')

# Cutting a buffer at a byte count is where a copy corrupts text: the limit
# lands inside a sequence and the receiver gets bytes that do not decode.
data = 'é'.join('hello')  # h e l l o joined by two-byte characters
assert truncate_utf8(b'abc', 10) == b'abc'
assert truncate_utf8(b'abc', 3) == b'abc'
assert truncate_utf8(b'abc', 0) == b''
assert truncate_utf8(b'h\xc3\xa9llo', 2) == b'h', \
    'a limit inside a two-byte sequence drops the whole sequence'
assert truncate_utf8(b'h\xc3\xa9llo', 3) == b'h\xc3\xa9'
assert truncate_utf8('\U0001F600'.encode('utf-8'), 3) == b'', \
    'a four-byte sequence is dropped whole'
assert truncate_utf8('\U0001F600'.encode('utf-8'), 4) == \
    '\U0001F600'.encode('utf-8')
for limit in range(len(data.encode('utf-8')) + 2):
    cut = truncate_utf8(data.encode('utf-8'), limit)
    assert len(cut) <= limit or limit > len(data.encode('utf-8'))
    cut.decode('utf-8')  # must never raise

try:
    truncate_utf8(b'abc', -1)
except ValueError:
    pass
else:
    raise AssertionError('a negative limit was accepted')
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement Python functions filetime_to_unix(ticks) and "
            "unix_to_filetime(seconds) converting between the Windows "
            "FILETIME representation and Unix time. A FILETIME is the number "
            "of 100-nanosecond intervals since 1601-01-01T00:00:00Z, held in "
            "an unsigned 64-bit integer; Unix time counts seconds since "
            "1970-01-01T00:00:00Z. There are exactly 11644473600 seconds "
            "between the two epochs. filetime_to_unix returns a float and "
            "raises ValueError for ticks outside 0 .. 2**64-1. "
            "unix_to_filetime returns an integer number of ticks, rounding "
            "to the nearest tick with halves going to the even tick, and "
            "raises ValueError when the result would fall outside the "
            "unsigned 64-bit range. Also expose the module-level constants "
            "EPOCH_DIFFERENCE_SECONDS and TICKS_PER_SECOND."
        ),
        validator=LOAD_CANDIDATE + require("filetime_to_unix")
        + require("unix_to_filetime") + require("EPOCH_DIFFERENCE_SECONDS")
        + require("TICKS_PER_SECOND") + r'''
assert EPOCH_DIFFERENCE_SECONDS == 11644473600
assert TICKS_PER_SECOND == 10 ** 7, 'a tick is 100 nanoseconds'

UNIX_EPOCH_IN_TICKS = 116444736000000000

# The Unix epoch expressed as a FILETIME. Both the 1601 offset and the tick
# scale have to be right for this to land on zero; getting either wrong
# leaves a plausible-looking timestamp that is centuries or hours out.
assert filetime_to_unix(UNIX_EPOCH_IN_TICKS) == 0.0
assert unix_to_filetime(0) == UNIX_EPOCH_IN_TICKS

assert filetime_to_unix(0) == -11644473600.0, 'tick zero is 1601, not 1970'
assert unix_to_filetime(-11644473600) == 0

assert filetime_to_unix(UNIX_EPOCH_IN_TICKS + 10 ** 7) == 1.0
assert filetime_to_unix(UNIX_EPOCH_IN_TICKS + 10 ** 16) == 1e9
assert unix_to_filetime(1000000000) == UNIX_EPOCH_IN_TICKS + 10 ** 16

assert unix_to_filetime(0.5) == UNIX_EPOCH_IN_TICKS + 5 * 10 ** 6
assert filetime_to_unix(UNIX_EPOCH_IN_TICKS + 5 * 10 ** 6) == 0.5
assert unix_to_filetime(1e-7) == UNIX_EPOCH_IN_TICKS + 1, \
    'a single tick of resolution must survive'

for seconds in (0, 1, -1, 1000000000, 1234567890.5, -11644473600):
    assert filetime_to_unix(unix_to_filetime(seconds)) == float(seconds), \
        f'round trip lost {seconds}'

for bad in (-1, 2 ** 64, 2 ** 70):
    try:
        filetime_to_unix(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'filetime_to_unix({bad}) was accepted')

for bad in (-11644473601, -2e11, 2e12):
    try:
        unix_to_filetime(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'unix_to_filetime({bad}) was accepted')
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function parse_signature(signature) that "
            "reads a JNI method descriptor and returns the tuple "
            "(parameter_types, return_type) where parameter_types is a tuple "
            "of strings and both use Java source syntax. A descriptor is a "
            "parenthesised parameter list followed by a return type. The "
            "field codes are Z boolean, B byte, C char, S short, I int, J "
            "long, F float and D double; V void is legal only as the return "
            "type. L followed by a class name in internal form and then a "
            "semicolon is an object type, and its source name replaces every "
            "'/' with '.'. A '[' prefixes an array of whatever type follows "
            "it, and nests, so its source name is the element type followed "
            "by one '[]' per prefix. Raise ValueError for a descriptor that "
            "does not start with '(', has no ')', has no return type or "
            "anything after it, uses an unknown code, leaves an object type "
            "unterminated, or applies '[' to void."
        ),
        validator=LOAD_CANDIDATE + require("parse_signature") + r'''
assert parse_signature('()V') == ((), 'void')
assert parse_signature('(I)I') == (('int',), 'int')
assert parse_signature('(JIZ)J') == (('long', 'int', 'boolean'), 'long')
assert parse_signature('(BCSFD)V') == \
    (('byte', 'char', 'short', 'float', 'double'), 'void')

# J is long and F is float: the two codes that do not match the initial of
# the type they name, and the pair a hand-written table gets backwards.
assert parse_signature('(J)V')[0] == ('long',)
assert parse_signature('(D)V')[0] == ('double',)

assert parse_signature('(Ljava/lang/String;)V') == \
    (('java.lang.String',), 'void')
assert parse_signature('()Ljava/lang/Object;') == ((), 'java.lang.Object')
assert parse_signature('(Ljava/lang/String;)Ljava/util/List;') == \
    (('java.lang.String',), 'java.util.List')
assert parse_signature('(LFoo;)V') == (('Foo',), 'void'), \
    'a class name is read to its semicolon, not by its first letter'

assert parse_signature('([I)J') == (('int[]',), 'long')
assert parse_signature('([[I)V') == (('int[][]',), 'void')
assert parse_signature('([Ljava/lang/String;)V') == \
    (('java.lang.String[]',), 'void')
assert parse_signature('()[[Ljava/lang/String;') == \
    ((), 'java.lang.String[][]')

assert parse_signature('(Ljava/lang/String;[IJ)V') == \
    (('java.lang.String', 'int[]', 'long'), 'void'), \
    'an object type ends at its semicolon and the next code starts after it'
assert parse_signature('([Ljava/lang/String;[Ljava/lang/String;)V') == \
    (('java.lang.String[]', 'java.lang.String[]'), 'void')

for bad in ('', 'V', 'I)V', '()', '(', '(I', '(IV', '(Q)V', '(I)Q',
            '(Ljava/lang/String)V', '(Ljava/lang/String;', '(V)V',
            '([V)V', '()[V', '(I)IV', '()VV', '([)V'):
    try:
        parse_signature(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'parse_signature({bad!r}) was accepted')
''',
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement Python functions pack_bits(fields, values, order) and "
            "unpack_bits(fields, packed, order) marshalling a C bit-field "
            "struct or a wire header. fields is a list of (name, width) "
            "pairs with widths of at least 1 summing to at most 64; order is "
            "'lsb' when the first field occupies the least significant bits "
            "of the packed integer, or 'msb' when the first field occupies "
            "the most significant bits of the total width. pack_bits returns "
            "the packed unsigned integer and unpack_bits returns "
            "{name: value}. Every value is unsigned and must fit its width. "
            "Raise ValueError for a width below 1, a total width above 64, a "
            "repeated field name, an order that is neither 'lsb' nor 'msb', "
            "a value that does not fit its width or is negative, and a "
            "packed integer that does not fit the total width. Raise "
            "KeyError when values does not have exactly one entry per field."
        ),
        validator=LOAD_CANDIDATE + require("pack_bits")
        + require("unpack_bits") + r'''
# The first byte of an IPv4 header: version then IHL, most significant
# first. Reading it least-significant-first yields 0x54 and a parser that
# reports IHL 4 and version 5 -- both plausible, both wrong.
header = [('version', 4), ('ihl', 4)]
assert pack_bits(header, {'version': 4, 'ihl': 5}, 'msb') == 0x45, \
    "'msb' puts the first field in the high bits of the total width"
assert pack_bits(header, {'version': 4, 'ihl': 5}, 'lsb') == 0x54
assert unpack_bits(header, 0x45, 'msb') == {'version': 4, 'ihl': 5}
assert unpack_bits(header, 0x54, 'lsb') == {'version': 4, 'ihl': 5}

triple = [('a', 1), ('b', 3), ('c', 4)]
assert pack_bits(triple, {'a': 1, 'b': 5, 'c': 9}, 'lsb') == 0x9B
assert pack_bits(triple, {'a': 1, 'b': 5, 'c': 9}, 'msb') == 0xD9
for order in ('lsb', 'msb'):
    assert unpack_bits(triple, pack_bits(
        triple, {'a': 1, 'b': 5, 'c': 9}, order), order) == \
        {'a': 1, 'b': 5, 'c': 9}

single = [('only', 64)]
assert pack_bits(single, {'only': 2 ** 64 - 1}, 'lsb') == 2 ** 64 - 1
assert pack_bits(single, {'only': 2 ** 64 - 1}, 'msb') == 2 ** 64 - 1
assert unpack_bits(single, 0, 'msb') == {'only': 0}

zeros = [('a', 4), ('b', 4)]
assert pack_bits(zeros, {'a': 0, 'b': 0}, 'msb') == 0

for bad in (
    ([('a', 0)], {'a': 0}, 'lsb'),
    ([('a', -1)], {'a': 0}, 'lsb'),
    ([('a', 64), ('b', 1)], {'a': 0, 'b': 0}, 'lsb'),
    ([('a', 4), ('a', 4)], {'a': 1}, 'lsb'),
    (header, {'version': 4, 'ihl': 5}, 'be'),
    (header, {'version': 16, 'ihl': 5}, 'msb'),
    (header, {'version': -1, 'ihl': 5}, 'msb'),
):
    try:
        pack_bits(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'pack_bits{bad} was accepted')

for bad in ((header, {'version': 4}, 'msb'),
            (header, {'version': 4, 'ihl': 5, 'extra': 0}, 'msb')):
    try:
        pack_bits(*bad)
    except KeyError:
        pass
    else:
        raise AssertionError(f'pack_bits{bad} was accepted')

for bad in ((header, 0x100, 'msb'), (header, -1, 'msb')):
    try:
        unpack_bits(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'unpack_bits{bad} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python function varargs_layout(arguments) computing "
            "how C lays out the variadic tail of a call. arguments is a list "
            "of (ctype, value) pairs whose ctype is one of bool, char, "
            "signed char, unsigned char, short, unsigned short, int, "
            "unsigned int, long, unsigned long, long long, float, double or "
            "pointer. Apply the default argument promotions first: every "
            "integer type narrower than int becomes int, and float becomes "
            "double, while the other types are unchanged. The promoted types "
            "have sizes int and unsigned int 4, long, unsigned long, long "
            "long, double and pointer 8, and each is aligned to its own "
            "size. Place the promoted arguments in order, each at the next "
            "offset that is a multiple of its alignment. Return "
            "{'arguments': [(promoted_ctype, value, offset), ...], 'size': "
            "int} where size is the total rounded up to the largest "
            "alignment used, or 0 when there are no arguments. Raise "
            "ValueError for an unknown ctype."
        ),
        validator=LOAD_CANDIDATE + require("varargs_layout") + r'''
assert varargs_layout([]) == {'arguments': [], 'size': 0}

# The promotion that silently corrupts every argument after it: a float
# passed to printf is passed as a double, so a callee reading four bytes
# reads half of it and every later argument comes from the wrong offset.
assert varargs_layout([('float', 1.5)]) == \
    {'arguments': [('double', 1.5, 0)], 'size': 8}, \
    'a float is promoted to double when it crosses into a variadic call'

assert varargs_layout([('char', 65)]) == \
    {'arguments': [('int', 65, 0)], 'size': 4}
assert varargs_layout([('unsigned char', 200)]) == \
    {'arguments': [('int', 200, 0)], 'size': 4}
assert varargs_layout([('short', -3)]) == \
    {'arguments': [('int', -3, 0)], 'size': 4}
assert varargs_layout([('unsigned short', 65535)]) == \
    {'arguments': [('int', 65535, 0)], 'size': 4}, \
    'an unsigned short fits an int, so it promotes to int not unsigned int'
assert varargs_layout([('bool', True)]) == \
    {'arguments': [('int', True, 0)], 'size': 4}

unchanged = [('int', 1), ('unsigned int', 2), ('long', 3),
             ('unsigned long', 4), ('long long', 5), ('double', 6.0),
             ('pointer', 7)]
promoted = varargs_layout(unchanged)['arguments']
assert [entry[0] for entry in promoted] == \
    ['int', 'unsigned int', 'long', 'unsigned long', 'long long', 'double',
     'pointer'], 'these types are already promoted'

assert varargs_layout([('int', 1), ('int', 2)]) == \
    {'arguments': [('int', 1, 0), ('int', 2, 4)], 'size': 8}

# Promotion happens before alignment, so the padding is decided by the
# promoted width rather than the written one.
assert varargs_layout([('char', 65), ('double', 1.0)]) == \
    {'arguments': [('int', 65, 0), ('double', 1.0, 8)], 'size': 16}, \
    'the double aligns to 8 after the promoted int occupies 4'
assert varargs_layout([('int', 1), ('pointer', 0)]) == \
    {'arguments': [('int', 1, 0), ('pointer', 0, 8)], 'size': 16}
assert varargs_layout([('float', 1.0), ('int', 2)]) == \
    {'arguments': [('double', 1.0, 0), ('int', 2, 8)], 'size': 16}, \
    'the trailing int is padded to the largest alignment used'
assert varargs_layout([('int', 1), ('int', 2), ('int', 3)]) == \
    {'arguments': [('int', 1, 0), ('int', 2, 4), ('int', 3, 8)], 'size': 12}

for bad in ([('size_t', 1)], [('void', None)], [('int32_t', 1)]):
    try:
        varargs_layout(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'varargs_layout({bad}) was accepted')
''',
    ),
    task(
        f"{FAMILY}-0201", FAMILY,
        prompt=(
            "Implement two Python functions, cobs_encode(data) and "
            "cobs_decode(frame), converting between arbitrary bytes and a "
            "framing that contains no zero byte, so a zero can delimit "
            "frames on the wire. Both take and return bytes.\n\n"
            "cobs_encode processes the input as consecutive blocks. A block "
            "gathers input bytes until either a zero byte is reached or 254 "
            "non-zero bytes have been gathered, whichever happens first. For "
            "each block emit one code byte equal to the number of gathered "
            "bytes plus one, followed by the gathered bytes themselves. A "
            "zero byte that ended a block is consumed and not emitted. After "
            "a block that ended on a consumed zero byte, another block "
            "always follows, even when the input is exhausted, in which case "
            "that final block gathers nothing and its code byte is 1. After "
            "a block that ended because 254 bytes were gathered, another "
            "block follows only if input remains. Empty input therefore "
            "encodes to the single byte 0x01.\n\n"
            "cobs_decode reverses this: read a code byte c, copy the next "
            "c - 1 bytes to the output, and then append one zero byte unless "
            "c was 255 or the frame is now exhausted. Raise ValueError if "
            "the frame is empty, contains a zero byte, or declares a block "
            "running past its end."
        ),
        validator=LOAD_CANDIDATE + require("cobs_encode") + require("cobs_decode")
        + """
def reference_decode(frame):
    \"\"\"Decode independently of the candidate, so a self-consistent but
    non-standard encoder is caught. This is the peer implementation the
    framing exists to interoperate with; agreeing only with itself is
    exactly the interoperability defect under test.\"\"\"
    out = bytearray()
    index = 0
    while index < len(frame):
        code = frame[index]
        index += 1
        end = index + code - 1
        out += frame[index:end]
        index = end
        if code != 0xFF and index < len(frame):
            out.append(0)
    return bytes(out)

def check(data):
    frame = cobs_encode(data)
    assert isinstance(frame, (bytes, bytearray)), 'cobs_encode must return bytes'
    assert 0 not in frame, f'frame for {data!r} contains a zero byte'
    assert bytes(cobs_decode(bytes(frame))) == data, \\
        f'candidate did not round-trip {data!r}'
    assert reference_decode(bytes(frame)) == data, \\
        f'an independent decoder read {reference_decode(bytes(frame))!r} ' \\
        f'from the frame for {data!r}'
    # One code byte per 254 bytes of payload, and never more.
    assert len(frame) <= len(data) + (len(data) // 254) + 2, \\
        f'frame for {len(data)} bytes is {len(frame)} bytes, too much overhead'
    return bytes(frame)

# The cases the rule pins exactly, each derived from the prompt's own wording
# rather than from a remembered table.
assert check(b'') == b'\\x01'
assert check(b'\\x00') == b'\\x01\\x01'
assert check(b'\\x00\\x00') == b'\\x01\\x01\\x01'
assert check(b'\\x11\\x22\\x00\\x33') == b'\\x03\\x11\\x22\\x02\\x33'
assert check(b'\\x11\\x00\\x00\\x00') == b'\\x02\\x11\\x01\\x01\\x01'
assert check(b'\\x01\\x02\\x03') == b'\\x04\\x01\\x02\\x03'

# The 254-byte boundary, where "another block follows only if input remains"
# is the whole contract. A block gathering exactly 254 bytes at the end of
# the input must not be followed by an empty one.
run = bytes(range(1, 255))
assert len(run) == 254
frame = check(run)
assert frame == b'\\xff' + run, \\
    f'254 non-zero bytes framed as {len(frame)} bytes'
frame = check(run + b'\\x05')
assert frame == b'\\xff' + run + b'\\x02\\x05'
# A block that filled at 254 bytes does not consume the zero that follows,
# so that zero opens a block of its own and the always-present final block
# follows that. Both trailing code bytes are 1 and neither is optional --
# dropping either one loses the payload's last zero on the wire.
frame = check(run + b'\\x00')
assert frame == b'\\xff' + run + b'\\x01\\x01'

for payload in (bytes(255), bytes(range(1, 255)) * 3, b'\\x00' * 700,
                bytes([7]) * 253, bytes([7]) * 254, bytes([7]) * 255,
                bytes([7]) * 508, b'\\x00\\x01' * 400):
    check(payload)

import random
rng = random.Random(20260909)
for _ in range(30):
    length = rng.randint(0, 900)
    check(bytes(rng.randrange(0, 256) for _ in range(length)))

for bad in (b'', b'\\x03\\x11', b'\\x05\\x01\\x02', b'\\x02\\x00\\x01',
            b'\\x01\\x00'):
    try:
        cobs_decode(bad)
    except ValueError:
        continue
    raise AssertionError(f'cobs_decode({bad!r}) was accepted')
""",
    ),
    task(
        f"{FAMILY}-0202", FAMILY,
        prompt=(
            "Implement a Python function crc16(data) returning the CRC-16 of "
            "a bytes object under the parameters commonly labelled "
            "CCITT-FALSE: width 16, generator polynomial 0x1021, initial "
            "register value 0xFFFF, the message processed most significant "
            "bit first with neither the input bytes nor the output register "
            "bit-reversed, and no final exclusive-or. Return an integer in "
            "the range 0 to 65535. The check value for the nine ASCII bytes "
            "'123456789' under these parameters is 0x29B1."
        ),
        validator=LOAD_CANDIDATE + require("crc16") + SHAPE_GUARDS + """
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
crc16 = returning(crc16, 'crc16(...)')
POLYNOMIAL = (1 << 16) | 0x1021

def by_long_division(data):
    \"\"\"The same CRC as polynomial remainder over GF(2).

    A non-reflected CRC with initial value I and no final exclusive-or is the
    remainder of the message -- with I applied to its leading bits -- shifted
    left by the register width. That is ordinary long division, and it shares
    no structure with the byte-at-a-time shift register a candidate will
    write, so it cannot inherit that loop's reflection or seeding mistakes.
    \"\"\"
    seeded = bytearray(data)
    seeded[0] ^= 0xFF
    seeded[1] ^= 0xFF
    value = int.from_bytes(bytes(seeded), 'big') << 16
    for shift in range(value.bit_length() - 17, -1, -1):
        if (value >> (shift + 16)) & 1:
            value ^= POLYNOMIAL << shift
    return value & 0xFFFF

# Anchor the oracle against the parameter set's own published check value
# before letting it judge anything. An oracle nobody checked is just a second
# opinion from the same author.
assert by_long_division(b'123456789') == 0x29B1, 'the oracle is wrong'
assert crc16(b'123456789') == 0x29B1, \\
    f'check value is 0x{crc16(b"123456789"):04X}, not 0x29B1'

samples = [
    b'\\x00\\x00', b'\\xff\\xff', b'A ', b'\\x00\\x01\\x02\\x03',
    bytes(range(32)), b'the quick brown fox', b'\\x80\\x00\\x00\\x00',
    bytes(64), bytes([0xAA, 0x55]) * 9, b'interoperability',
]
import random
rng = random.Random(29177)
for _ in range(40):
    samples.append(bytes(rng.randrange(256)
                         for _ in range(rng.randint(2, 300))))
for data in samples:
    expected = by_long_division(data)
    actual = crc16(data)
    assert actual == expected, \\
        f'crc16({data[:16]!r}...) gave 0x{actual:04X}, expected 0x{expected:04X}'

# Single bytes and the empty message still have to work; with no message bits
# the register is the initial value.
assert crc16(b'') == 0xFFFF, f'empty message gave 0x{crc16(b""):04X}'
for single in (b'\\x00', b'\\x01', b'\\xff', b'A'):
    seeded = bytes([single[0] ^ 0xFF, 0xFF])
    value = int.from_bytes(seeded, 'big') << 16
    for shift in range(value.bit_length() - 17, -1, -1):
        if (value >> (shift + 16)) & 1:
            value ^= POLYNOMIAL << shift
    # Only the first byte is real message, so undo the second seed byte by
    # running the division over one byte directly instead.
    register = 0xFFFF ^ (single[0] << 8)
    for _ in range(8):
        register = (((register << 1) ^ 0x1021) & 0xFFFF
                    if register & 0x8000 else (register << 1) & 0xFFFF)
    assert crc16(single) == register, \\
        f'crc16({single!r}) gave 0x{crc16(single):04X}, expected 0x{register:04X}'

# The register must be 16 bits wide throughout: a candidate that lets it grow
# agrees on short messages and diverges once the overflow would have mattered.
for data in (bytes([0xFF]) * 40, bytes([0x80]) * 40):
    assert 0 <= crc16(data) <= 0xFFFF
    assert crc16(data) == by_long_division(data)
""",
    ),
    task(
        f"{FAMILY}-0203", FAMILY,
        prompt=(
            "Implement a Python function parse_der(data) reading one "
            "distinguished-encoding-rules element from the start of a bytes "
            "object and returning a tuple.\n\n"
            "Every element is an identifier octet, then length octets, then "
            "contents. Assume the identifier is always a single octet; its "
            "bit 0x20 marks the element constructed rather than primitive. "
            "If the first length octet is below 0x80 it is the content "
            "length. Otherwise its low seven bits give how many further "
            "octets hold the length, big-endian.\n\n"
            "For a primitive element return ('primitive', identifier, "
            "contents) with contents as bytes. For a constructed element "
            "return ('constructed', identifier, children), where children is "
            "the list of elements parsed from its contents, each in the same "
            "form, and those children must fill the contents exactly.\n\n"
            "Raise ValueError when the data is empty or ends early, when the "
            "first length octet is 0x80 (the indefinite form, which these "
            "rules forbid) or 0xFF (reserved), when a length is not encoded "
            "in the fewest possible octets -- a length below 128 must use "
            "the single-octet form, and a multi-octet length must not begin "
            "with a zero octet -- or when a constructed element's children "
            "do not end exactly on its content boundary. Trailing bytes "
            "after the first complete element are ignored."
        ),
        validator=LOAD_CANDIDATE + require("parse_der") + """
def encode_length(length):
    \"\"\"Build fixtures with an encoder, so no expected byte is hand-written.\"\"\"
    if length < 0x80:
        return bytes([length])
    body = length.to_bytes((length.bit_length() + 7) // 8, 'big')
    return bytes([0x80 | len(body)]) + body

def primitive(identifier, contents):
    return bytes([identifier]) + encode_length(len(contents)) + contents

def constructed(identifier, *children):
    body = b''.join(children)
    return bytes([identifier | 0x20]) + encode_length(len(body)) + body

assert parse_der(primitive(0x02, b'\\x2a')) == ('primitive', 0x02, b'\\x2a')
assert parse_der(primitive(0x04, b'')) == ('primitive', 0x04, b'')
assert parse_der(primitive(0x05, b'')) == ('primitive', 0x05, b'')

# The short form's boundary in both directions.
short = primitive(0x04, b'x' * 127)
assert short[1] == 127, 'fixture is not using the short form'
assert parse_der(short) == ('primitive', 0x04, b'x' * 127)
longer = primitive(0x04, b'y' * 128)
assert longer[1] == 0x81 and longer[2] == 128
assert parse_der(longer) == ('primitive', 0x04, b'y' * 128)
biggest = primitive(0x04, b'z' * 300)
assert biggest[1] == 0x82
assert parse_der(biggest) == ('primitive', 0x04, b'z' * 300)

nested = constructed(
    0x10,
    primitive(0x02, b'\\x01'),
    constructed(0x10, primitive(0x0c, b'inner')),
    primitive(0x01, b'\\xff'),
)
kind, identifier, children = parse_der(nested)
assert kind == 'constructed' and identifier == 0x30
assert children == [
    ('primitive', 0x02, b'\\x01'),
    ('constructed', 0x30, [('primitive', 0x0c, b'inner')]),
    ('primitive', 0x01, b'\\xff'),
], f'nested parse gave {children}'

# An empty constructed element has no children rather than one empty child.
assert parse_der(constructed(0x10)) == ('constructed', 0x30, [])

# Trailing bytes belong to whatever comes next and are not an error.
assert parse_der(primitive(0x02, b'\\x07') + b'\\x99\\x99') == \\
    ('primitive', 0x02, b'\\x07')

def rejects(data, why):
    try:
        parse_der(data)
    except ValueError:
        return
    raise AssertionError(f'accepted {why}: {data!r}')

rejects(b'', 'empty input')
rejects(b'\\x02', 'identifier with no length')
rejects(b'\\x02\\x03\\x01', 'contents shorter than the declared length')
rejects(b'\\x02\\x80\\x01\\x02\\x00\\x00', 'the indefinite length form')
rejects(b'\\x02\\xff\\x01', 'the reserved 0xFF length octet')
rejects(b'\\x02\\x81\\x01\\x2a', 'a long form encoding a length below 128')
rejects(b'\\x02\\x81\\x7f' + b'x' * 127, 'a long form encoding exactly 127')
rejects(b'\\x02\\x82\\x00\\x80' + b'x' * 128, 'a leading zero length octet')
rejects(b'\\x02\\x82\\x01', 'length octets that end early')
# The children must land exactly on the boundary, not merely inside it.
rejects(bytes([0x30, 0x05]) + primitive(0x02, b'\\x01'),
        'a constructed element whose children stop short')
rejects(bytes([0x30, 0x02]) + primitive(0x02, b'\\x01'),
        'a constructed element whose child overruns it')
""",
    ),
    task(
        f"{FAMILY}-0204", FAMILY,
        prompt=(
            "Implement a Python function read_name(message, offset) that "
            "reads one domain name from a DNS message and returns the tuple "
            "(name, next_offset).\n\n"
            "A name is a sequence of labels. Each label begins with one "
            "length octet. When the octet's top two bits are both zero it "
            "gives the label's length, 1 to 63, and that many bytes follow. "
            "A length octet of zero ends the name. When the top two bits are "
            "both one, that octet and the next form a pointer whose low "
            "fourteen bits are an offset from the start of the message; "
            "reading continues from there and the name ends where that "
            "branch ends.\n\n"
            "Return the labels decoded as ASCII and joined with '.', with "
            "the root name -- a single zero octet -- decoding to the empty "
            "string. next_offset is the position just past the name as it "
            "appears at the offset given, so when a pointer was followed it "
            "is two past that pointer, never a position in the branch.\n\n"
            "Raise ValueError when offset is outside the message, when a "
            "label or pointer runs past the end of the message, when a "
            "length octet's top two bits are 01 or 10 (both reserved), and "
            "when the pointers form a cycle. read_name must always "
            "terminate, however the message is constructed."
        ),
        validator=LOAD_CANDIDATE + require("read_name") + """
def labels(*parts):
    out = bytearray()
    for part in parts:
        out.append(len(part))
        out += part.encode('ascii')
    out.append(0)
    return bytes(out)

def pointer(offset):
    return bytes([0xC0 | (offset >> 8), offset & 0xFF])

# A plain name, and the root.
message = labels('www', 'example', 'com')
assert read_name(message, 0) == ('www.example.com', len(message))
assert read_name(b'\\x00', 0) == ('', 1)

# A pointer to a suffix. next_offset is two past the POINTER, which is the
# assertion a solver that returns the branch's end fails -- and the reason a
# caller can keep walking a message full of compressed names at all.
base = labels('example', 'com')
message = base + labels('www')[:-1] + pointer(0)
name, next_offset = read_name(message, len(base))
assert name == 'www.example.com', f'got {name!r}'
assert next_offset == len(message), \\
    f'next_offset {next_offset} is not two past the pointer'

# A pointer straight to another name, with nothing before it.
message = base + pointer(0)
assert read_name(message, len(base)) == ('example.com', len(base) + 2)

# A chain of pointers is legal as long as it terminates.
first = labels('com')
second = b'\\x07example' + pointer(0)
message = first + second + b'\\x03www' + pointer(len(first))
name, next_offset = read_name(message, len(first) + len(second))
assert name == 'www.example.com', f'chained pointers gave {name!r}'
assert next_offset == len(message)

# Case and hyphens survive; labels are bytes, not identifiers.
message = labels('My-Host', 'EXAMPLE', 'co', 'uk')
assert read_name(message, 0) == ('My-Host.EXAMPLE.co.uk', len(message))

# A label of the maximum length.
longest = 'a' * 63
message = labels(longest, 'net')
assert read_name(message, 0) == (f'{longest}.net', len(message))

def rejects(message, offset, why):
    try:
        read_name(message, offset)
    except ValueError:
        return
    raise AssertionError(f'accepted {why}')

rejects(b'', 0, 'an empty message')
rejects(labels('a'), 99, 'an offset past the end')
rejects(b'\\x05abc', 0, 'a label running past the end')
rejects(b'\\x03www', 0, 'a name with no terminating zero')
rejects(b'\\x40\\x01', 0, 'the reserved 01 length prefix')
rejects(b'\\x80\\x01', 0, 'the reserved 10 length prefix')
rejects(b'\\xc0', 0, 'a pointer with no second octet')
rejects(pointer(500) + b'\\x00', 0, 'a pointer past the end')

# The two cycles that hang a solver which trusts the message. Reaching this
# assertion at all is most of what the task measures; a solver that loops is
# stopped by the task timeout and scored a failure, not a hang.
rejects(pointer(0), 0, 'a pointer to itself')
rejects(pointer(2) + pointer(0), 0, 'a two-step pointer cycle')
rejects(b'\\x01a' + pointer(0), 0, 'a label whose pointer returns to it')
""",
    ),
]
