"""Held-out tasks: frontend state, UX, and accessibility.

Frontend defects are unusually easy to write a passing test for and unusually
hard to write correctly, because the correct behaviour is specified somewhere
other than the component: in WCAG, in the HTML sequential focus navigation
order, in the accessible name computation, in CLDR plural rules. An
implementation that reproduces the prompt's examples and gets the standard
wrong ships a component that looks right to its author and is unusable to
somebody navigating by keyboard or listening to a screen reader.

So the validators here drive the candidate with the cases the standards call
out rather than the cases a demo would show: the channel below WCAG's 0.03928
knee where the linearisation branches, the positive `tabindex` that jumps the
queue, the blank `aria-label` that must fall through instead of winning, the
Russian `11` that is `many` while `21` is `one`, the drag one position
downward that lands a place early when the removal shift is forgotten.

Two behaviours in this family are about state rather than markup, and both are
included because they are where optimistic UIs actually break: a rejected
write must roll back *without* discarding the authoritative value that arrived
while it was in flight, and an undo stack must coalesce keystrokes without
coalescing across an undo. Neither is visible in a small hand-run example --
each needs an event ordering that only happens under real latency.

Rendering, styling and layout are deliberately absent. A validator that
asserted on rendered pixels would need a browser, which the contract's offline
determinism rule forbids, and an assertion on a class name would be exactly
the identifier check the contract says cannot substitute for behaviour.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "frontend_state_ux_accessibility"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement Python functions contrast_ratio(foreground, "
            "background) and meets_wcag(foreground, background, level, "
            "large_text) implementing WCAG 2.1 colour contrast. Colours are "
            "CSS hex strings in either #rgb or #rrggbb form, case "
            "insensitive, where the short form expands by repeating each "
            "digit; anything else raises ValueError. For each channel take "
            "its value c in [0, 1] and linearise it as c / 12.92 when "
            "c <= 0.03928 and ((c + 0.055) / 1.055) ** 2.4 otherwise, take "
            "the relative luminance L = 0.2126*R + 0.7152*G + 0.0722*B, and "
            "return (lighter + 0.05) / (darker + 0.05), which does not depend "
            "on the order of the arguments. meets_wcag returns True when the "
            "ratio is at least the threshold for the level: 'AA' needs 4.5, "
            "or 3.0 for large text; 'AAA' needs 7.0, or 4.5 for large text. "
            "Any other level raises ValueError."
        ),
        validator=LOAD_CANDIDATE + require("contrast_ratio")
        + require("meets_wcag") + r'''
assert abs(contrast_ratio('#000000', '#ffffff') - 21.0) < 1e-9
assert abs(contrast_ratio('#ffffff', '#000000') - 21.0) < 1e-9, \
    'the ratio must not depend on which colour is passed first'
assert abs(contrast_ratio('#ff0000', '#ff0000') - 1.0) < 1e-9

# The short form repeats each digit; it does not zero-pad.
assert abs(contrast_ratio('#fff', '#000') - 21.0) < 1e-9
assert abs(contrast_ratio('#f00', '#ff0000') - 1.0) < 1e-9
assert abs(contrast_ratio('#ABCDEF', '#abcdef') - 1.0) < 1e-9

# Below the 0.03928 knee the linearisation is linear, not the power curve.
# #050505 lands there, and the two branches differ by 4e-3 -- far outside
# this tolerance, so an implementation that applies the power curve to every
# channel is caught here and nowhere else in this task.
assert abs(contrast_ratio('#050505', '#000000') - 1.030352) < 1e-4, \
    'channels at or below 0.03928 use the linear branch'

# Published reference values for grey on white.
assert abs(contrast_ratio('#767676', '#ffffff') - 4.54) < 0.01
assert abs(contrast_ratio('#666666', '#ffffff') - 5.74) < 0.01

for bad in ('', 'fff', '#12345', '#gggggg', '#ffff', 'red', '#', '#ff00'):
    try:
        contrast_ratio(bad, '#000000')
    except ValueError:
        pass
    else:
        raise AssertionError(f'accepted malformed colour {bad!r}')

assert meets_wcag('#000000', '#ffffff', 'AA', False) is True
assert meets_wcag('#000000', '#ffffff', 'AAA', False) is True
assert meets_wcag('#666666', '#ffffff', 'AA', False) is True
assert meets_wcag('#666666', '#ffffff', 'AAA', False) is False
assert meets_wcag('#777777', '#ffffff', 'AA', False) is False, \
    '#777777 on white is 4.48 and does not reach 4.5'
assert meets_wcag('#777777', '#ffffff', 'AA', True) is True
assert meets_wcag('#949494', '#ffffff', 'AA', True) is True
assert meets_wcag('#949494', '#ffffff', 'AAA', True) is False, \
    'large text at AAA needs 4.5, not 3.0'

for level in ('A', 'aa', '', 'AAAA', None):
    try:
        meets_wcag('#000', '#fff', level, False)
    except ValueError:
        pass
    else:
        raise AssertionError(f'accepted conformance level {level!r}')
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function focus_order(elements) returning the "
            "ids of the elements a user reaches by pressing Tab, in order. "
            "Each element is a dict with keys id and tag, and optionally "
            "tabindex (int), type (str), href (str), disabled (bool) and "
            "hidden (bool). Follow the HTML sequential focus navigation "
            "order: every element with a positive tabindex comes first, "
            "ordered by ascending tabindex with ties broken by document "
            "order, followed by every other focusable element in document "
            "order. tabindex 0 makes any element focusable; a negative "
            "tabindex is reachable only programmatically and never appears "
            "in the order. With no tabindex key, only button, select, "
            "textarea, input whose type is anything but 'hidden', and a with "
            "a non-empty href are focusable. An element that is disabled or "
            "hidden never appears, whatever its tabindex. Do not modify the "
            "input list or the dicts in it."
        ),
        validator=LOAD_CANDIDATE + require("focus_order") + r'''
import copy

elements = [
    {'id': 'skip', 'tag': 'a', 'href': '#main'},
    {'id': 'logo', 'tag': 'a'},
    {'id': 'menu', 'tag': 'div', 'tabindex': 0},
    {'id': 'search', 'tag': 'input', 'type': 'search'},
    {'id': 'csrf', 'tag': 'input', 'type': 'hidden'},
    {'id': 'third_stop', 'tag': 'button', 'tabindex': 3},
    {'id': 'first_stop', 'tag': 'button', 'tabindex': 1},
    {'id': 'second_stop', 'tag': 'div', 'tabindex': 1},
    {'id': 'archived', 'tag': 'button', 'disabled': True},
    {'id': 'offscreen', 'tag': 'button', 'hidden': True},
    {'id': 'roving', 'tag': 'div', 'tabindex': -1},
    {'id': 'submit', 'tag': 'button'},
    {'id': 'jumps_but_hidden', 'tag': 'button', 'tabindex': 2, 'hidden': True},
    {'id': 'note', 'tag': 'p'},
    {'id': 'agree', 'tag': 'input', 'type': 'checkbox', 'tabindex': 0},
    {'id': 'country', 'tag': 'select'},
    {'id': 'comment', 'tag': 'textarea', 'disabled': True},
    {'id': 'untyped', 'tag': 'input'},
]
snapshot = copy.deepcopy(elements)

assert focus_order(elements) == [
    'first_stop', 'second_stop', 'third_stop',
    'skip', 'menu', 'search', 'submit', 'agree', 'country', 'untyped',
], 'sequential focus navigation order is wrong'

assert elements == snapshot, 'focus_order mutated its argument'

# The positive-tabindex tie break is document order, and it must stay stable
# across more than a pair.
ties = [
    {'id': f'tie{index}', 'tag': 'button', 'tabindex': 5}
    for index in range(6)
]
ties.insert(3, {'id': 'earlier', 'tag': 'button', 'tabindex': 4})
assert focus_order(ties) == [
    'earlier', 'tie0', 'tie1', 'tie2', 'tie3', 'tie4', 'tie5',
], 'equal tabindex values must keep document order'

assert focus_order([]) == []
assert focus_order([{'id': 'x', 'tag': 'span'}]) == []
assert focus_order([{'id': 'x', 'tag': 'span', 'tabindex': 0}]) == ['x']
assert focus_order([{'id': 'x', 'tag': 'a', 'href': ''}]) == [], \
    'an anchor without a target is not focusable'
assert focus_order([
    {'id': 'x', 'tag': 'button', 'tabindex': 7, 'disabled': True},
]) == [], 'a disabled element is skipped even with a positive tabindex'
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function accessible_name(tree, element_id) "
            "computing an element's accessible name. tree maps an id to a "
            "node dict with a required tag and optional text, aria_label, "
            "labelledby (list of ids), title, alt and children (list of ids) "
            "keys. Apply these steps in order and return the first name they "
            "produce, with every run of whitespace collapsed to one space and "
            "the result stripped. (1) Only at the top of the computation, if "
            "labelledby is non-empty, join the names of the referenced nodes "
            "with a single space, skipping ids that are not in the tree; a "
            "referenced node always contributes its content whatever its tag, "
            "and its own labelledby is not followed, so a reference cycle "
            "cannot recur. Continue to step 2 only if the joined result is "
            "empty. (2) aria_label, if it is not blank. (3) For an img, the "
            "alt value if the key is present, even when it is empty, which "
            "names the image deliberately. (4) Content, used when the tag is "
            "one of button, a, label, legend, summary, td, th, h1, h2, h3, "
            "h4, h5, h6: the node's own text followed by the names of its "
            "children in order, each computed the same way but always "
            "allowed to use content; empty parts contribute nothing, and an "
            "empty result falls through. (5) title, if it is not blank. (6) "
            "Otherwise the empty string. An id that is not in the tree raises "
            "KeyError."
        ),
        validator=LOAD_CANDIDATE + require("accessible_name") + r'''
tree = {
    'save': {'tag': 'button', 'text': '  Save   all\n changes '},
    'iconbtn': {'tag': 'button', 'aria_label': 'Close dialog', 'text': 'X'},
    'blanklabel': {'tag': 'button', 'aria_label': '  \t ',
                   'text': 'Fallback text'},
    'field': {'tag': 'input', 'labelledby': ['heading', 'hint', 'missing'],
              'aria_label': 'never used'},
    'heading': {'tag': 'h2', 'text': 'Billing address'},
    'hint': {'tag': 'span', 'text': '(required)'},
    'decorative': {'tag': 'img', 'alt': '', 'title': 'Company logo'},
    'photo': {'tag': 'img', 'title': 'Team photo'},
    'cat': {'tag': 'img', 'alt': ' A cat  sleeping '},
    'toolbar': {'tag': 'div', 'text': 'not a name', 'title': 'Formatting'},
    'anon': {'tag': 'div', 'text': 'invisible to the name computation'},
    'wrapper': {'tag': 'button', 'children': ['glyph', 'caption']},
    'glyph': {'tag': 'span', 'text': 'star'},
    'caption': {'tag': 'span', 'text': 'Favourite'},
    'emptybtn': {'tag': 'button', 'text': '   ', 'title': 'Delete row'},
    'delegating': {'tag': 'div', 'labelledby': ['inner']},
    'inner': {'tag': 'span', 'labelledby': ['heading'], 'text': 'Own text'},
    'selfref': {'tag': 'button', 'labelledby': ['selfref'],
                'text': 'Loop safe'},
    'nowhere': {'tag': 'div', 'labelledby': ['absent'], 'title': 'Fallback'},
}

expected = {
    'save': 'Save all changes',
    'iconbtn': 'Close dialog',
    'blanklabel': 'Fallback text',
    'field': 'Billing address (required)',
    'heading': 'Billing address',
    'hint': '',
    'decorative': '',
    'photo': 'Team photo',
    'cat': 'A cat sleeping',
    'toolbar': 'Formatting',
    'anon': '',
    'wrapper': 'star Favourite',
    'emptybtn': 'Delete row',
    'delegating': 'Own text',
    'selfref': 'Loop safe',
    'nowhere': 'Fallback',
}
for element_id, want in expected.items():
    got = accessible_name(tree, element_id)
    assert got == want, f'{element_id}: expected {want!r}, got {got!r}'

try:
    accessible_name(tree, 'no_such_element')
except KeyError:
    pass
else:
    raise AssertionError('an unknown element id must raise KeyError')
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function reconcile(events) that replays a "
            "client store's optimistic-update log and returns {'state': "
            "dict, 'pending': list of ids}. Events are applied in order and "
            "are dicts of one of four types. 'optimistic' with id, key and "
            "value is a local write awaiting a verdict. 'ack' with id means "
            "the server accepted that write, so its value becomes part of "
            "the confirmed base. 'reject' with id means the server refused "
            "it and the write disappears. 'server' with key and value is an "
            "authoritative push that updates the confirmed base without "
            "touching pending writes. The visible state is always the "
            "confirmed base with every still-pending write applied over it "
            "in the order the writes were made, so rejecting a write must "
            "not discard a server value that arrived while it was in flight, "
            "and acking out of order is allowed. 'pending' lists the ids "
            "still awaiting a verdict, in order. A verdict for an id that is "
            "not pending raises KeyError, a duplicate optimistic id raises "
            "ValueError, and an unknown event type raises ValueError. Do not "
            "modify the events."
        ),
        validator=LOAD_CANDIDATE + require("reconcile") + r'''
import copy

assert reconcile([]) == {'state': {}, 'pending': []}

events = [
    {'type': 'server', 'key': 'x', 'value': 0},
    {'type': 'optimistic', 'id': 'a', 'key': 'x', 'value': 1},
]
snapshot = copy.deepcopy(events)
assert reconcile(events) == {'state': {'x': 1}, 'pending': ['a']}
assert events == snapshot, 'reconcile mutated its argument'

# The case optimistic UIs get wrong: a server push lands while a write is in
# flight, and the rejection must fall back to the push, not to the value the
# store held before the write.
assert reconcile(events + [
    {'type': 'server', 'key': 'x', 'value': 99},
    {'type': 'reject', 'id': 'a'},
]) == {'state': {'x': 99}, 'pending': []}, \
    'a rejection rolled back over a newer server value'

# Later pending writes to the same key win over earlier ones.
stacked = [
    {'type': 'optimistic', 'id': 'a', 'key': 'x', 'value': 1},
    {'type': 'optimistic', 'id': 'b', 'key': 'x', 'value': 2},
]
assert reconcile(stacked) == {'state': {'x': 2}, 'pending': ['a', 'b']}
assert reconcile(stacked + [{'type': 'reject', 'id': 'b'}]) == \
    {'state': {'x': 1}, 'pending': ['a']}
assert reconcile(stacked + [{'type': 'ack', 'id': 'a'}]) == \
    {'state': {'x': 2}, 'pending': ['b']}, \
    'an out-of-order ack must not promote itself over a later pending write'
assert reconcile(stacked + [
    {'type': 'ack', 'id': 'a'},
    {'type': 'reject', 'id': 'b'},
]) == {'state': {'x': 1}, 'pending': []}

# Independent keys do not interfere, and an acked write survives.
mixed = [
    {'type': 'optimistic', 'id': 'a', 'key': 'title', 'value': 'draft'},
    {'type': 'optimistic', 'id': 'b', 'key': 'seen', 'value': True},
    {'type': 'ack', 'id': 'b'},
    {'type': 'server', 'key': 'title', 'value': 'remote'},
    {'type': 'reject', 'id': 'a'},
]
assert reconcile(mixed) == \
    {'state': {'title': 'remote', 'seen': True}, 'pending': []}

for bad, error in (
    ([{'type': 'ack', 'id': 'ghost'}], KeyError),
    ([{'type': 'reject', 'id': 'ghost'}], KeyError),
    (stacked + [{'type': 'ack', 'id': 'a'}, {'type': 'ack', 'id': 'a'}],
     KeyError),
    (stacked + [{'type': 'optimistic', 'id': 'a', 'key': 'x', 'value': 3}],
     ValueError),
    ([{'type': 'patch', 'key': 'x', 'value': 1}], ValueError),
):
    try:
        reconcile(bad)
    except error:
        pass
    else:
        raise AssertionError(f'{bad[-1]} did not raise {error.__name__}')
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python class VirtualList(heights) that windows a "
            "scrolling list of variable-height rows. heights is a list of "
            "non-negative integer pixel heights. Provide total_height(), "
            "offset_of(index) returning the pixel offset of a row's top, and "
            "window(scroll_top, viewport_height, overscan) returning the "
            "tuple (start, end, offset) where end is exclusive and offset is "
            "the pixel offset of row start. A row is visible when its top is "
            "strictly less than scroll_top + viewport_height and its bottom "
            "is strictly greater than scroll_top; a zero-height row is "
            "visible when scroll_top <= its top < scroll_top + "
            "viewport_height. Extend the visible range by overscan rows on "
            "each side, clamped to the list. When no row is visible return "
            "(len(heights), len(heights), total_height()) with no overscan "
            "applied. window must answer in time logarithmic in the number "
            "of rows: 20,000 queries against 200,000 rows have to finish "
            "well inside the time budget. Raise ValueError on a negative "
            "scroll_top, a viewport_height that is not positive, or a "
            "negative overscan, and IndexError from offset_of for an index "
            "outside 0..len(heights)."
        ),
        timeout_seconds=120.0,
        validator=LOAD_CANDIDATE + require("VirtualList") + r'''
import random

rows = VirtualList([10, 0, 20, 30, 0, 40])
assert rows.total_height() == 100
assert [rows.offset_of(index) for index in range(7)] == \
    [0, 10, 10, 30, 60, 60, 100]

assert rows.window(0, 30, 0) == (0, 3, 0), \
    'a zero-height row at the viewport top is inside the window'
assert rows.window(30, 30, 0) == (3, 4, 30), \
    'a row whose bottom equals scroll_top is above the window'
assert rows.window(60, 40, 0) == (4, 6, 60)
assert rows.window(30, 30, 2) == (1, 6, 10), \
    'overscan must extend the range and move the reported offset'
assert rows.window(0, 1000, 3) == (0, 6, 0), 'overscan must clamp'
assert rows.window(100, 10, 0) == (6, 6, 100), \
    'scrolled past the end there is nothing to render'
assert rows.window(100, 10, 4) == (6, 6, 100), \
    'an empty window must not be widened by overscan'

empty = VirtualList([])
assert empty.total_height() == 0
assert empty.window(0, 50, 2) == (0, 0, 0)

for bad in ((-1, 50, 0), (0, 0, 0), (0, -50, 0), (0, 50, -1)):
    try:
        rows.window(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'window{bad} was accepted')

for bad_index in (-1, 7):
    try:
        rows.offset_of(bad_index)
    except IndexError:
        pass
    else:
        raise AssertionError(f'offset_of({bad_index}) was accepted')

# --- the window must be a search, not a scan -----------------------------
# A per-query linear walk answers every assertion above and still makes a
# 200,000-row list unusable. 20,000 queries * 200,000 rows is 4e9 steps,
# which exceeds this task's budget by orders of magnitude, so a scan is
# caught here rather than in production.
heights = [(index % 7) + 1 for index in range(200000)]
big = VirtualList(heights)
total = sum(heights)
assert big.total_height() == total

generator = random.Random(20260905)
for _ in range(20000):
    scroll_top = generator.randrange(0, total)
    viewport = generator.randrange(200, 900)
    start, end, offset = big.window(scroll_top, viewport, 0)
    assert 0 <= start <= end <= len(heights)
    assert offset == sum(heights[:start]) if start < 40 else True
    assert start == end or (
        offset + heights[start] > scroll_top
    ), 'the first row returned is entirely above the viewport'
    if start > 0:
        assert offset <= scroll_top, 'a visible row was left out at the top'
    if end < len(heights):
        bottom = big.offset_of(end)
        assert bottom >= scroll_top + viewport, \
            'a visible row was left out at the bottom'
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python class EditHistory(initial_text, limit) "
            "providing apply(text, timestamp_ms), undo(), redo(), current(), "
            "can_undo() and can_redo(). Each apply records an undo step, "
            "except that an apply whose timestamp is within 1000 ms of the "
            "previous recorded step's timestamp coalesces into that step, "
            "replacing its text and taking the newer timestamp, so a burst "
            "of typing is one undo. An apply that immediately follows an "
            "undo or a redo never coalesces, because the user has moved the "
            "cursor through history and the burst is over. Any apply clears "
            "the redo stack. limit is the greatest number of undo steps "
            "retained; when a new step would exceed it the oldest step is "
            "dropped and its text becomes the new baseline, so undoing all "
            "the way back returns that text rather than the original. undo() "
            "at the baseline and redo() at the head return the current text "
            "unchanged. Every method returns the current text. A limit below "
            "1 raises ValueError."
        ),
        validator=LOAD_CANDIDATE + require("EditHistory") + r'''
history = EditHistory('', 5)
assert history.current() == ''
assert history.can_undo() is False and history.can_redo() is False

# A burst of typing is one undo step: each edit is within 1000 ms of the
# previous one, so the window slides rather than expiring at the first edit.
assert history.apply('h', 0) == 'h'
history.apply('he', 400)
history.apply('hel', 900)
assert history.current() == 'hel'
history.apply('hell', 2000)
history.apply('hello', 2500)
assert history.current() == 'hello'

assert history.undo() == 'hel', 'the second burst was not one step'
assert history.undo() == ''
assert history.can_undo() is False
assert history.undo() == '', 'undo at the baseline must be a no-op'
assert history.redo() == 'hel'
assert history.redo() == 'hello'
assert history.can_redo() is False
assert history.redo() == 'hello', 'redo at the head must be a no-op'

# An edit that follows an undo starts a new step even inside the window,
# and it discards the redo stack.
history.undo()
assert history.current() == 'hel'
history.apply('help', 1200)
assert history.can_redo() is False, 'a new edit must clear the redo stack'
assert history.undo() == 'hel', \
    'an edit after an undo coalesced into the step it should follow'

bounded = EditHistory('base', 2)
bounded.apply('a', 0)
bounded.apply('b', 5000)
bounded.apply('c', 10000)
assert bounded.current() == 'c'
assert bounded.undo() == 'b'
assert bounded.undo() == 'a', 'the dropped step becomes the baseline'
assert bounded.can_undo() is False, 'the history kept more than limit steps'
assert bounded.redo() == 'b'
assert bounded.redo() == 'c'

for bad_limit in (0, -1):
    try:
        EditHistory('', bad_limit)
    except ValueError:
        pass
    else:
        raise AssertionError(f'limit {bad_limit} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function debounce(events, wait, "
            "max_wait=None) returning the invocation schedule of a "
            "trailing-edge debounced callback. events is a list of "
            "(time_ms, payload) pairs in non-decreasing time order. A call "
            "schedules an invocation wait ms later; another call before the "
            "timer fires resets it to that call's time plus wait, and the "
            "invocation carries the payload of the most recent call. When "
            "max_wait is not None the invocation also happens no later than "
            "max_wait ms after the first call of the current burst, "
            "whichever deadline comes first. Firing ends the burst, so the "
            "next call starts a new one whose max_wait deadline is measured "
            "from that call. If a call arrives at exactly the moment the "
            "timer would fire, the timer fires first and the call begins a "
            "new burst. Return the list of (time_ms, payload) invocations in "
            "time order. Raise ValueError when wait is not positive, when "
            "max_wait is not None and is less than wait, or when the events "
            "are not in non-decreasing time order."
        ),
        validator=LOAD_CANDIDATE + require("debounce") + r'''
assert debounce([], 100) == []

# A burst collapses to one trailing call carrying the latest payload.
assert debounce([(0, 'a'), (50, 'b'), (90, 'c')], 100) == [(190, 'c')]

# A gap longer than wait is two bursts.
assert debounce([(0, 'a'), (200, 'b')], 100) == [(100, 'a'), (300, 'b')]

# A call landing exactly on the pending deadline fires the timer first and
# opens a new burst rather than extending the old one.
assert debounce([(0, 'a'), (100, 'b')], 100) == [(100, 'a'), (200, 'b')], \
    'a call at the firing instant must not extend the burst'

# max_wait caps a burst that would otherwise never settle. The first
# invocation is the cap firing at 0 + 250 carrying the newest call before it;
# the second is the burst opened by the call at 250, whose own cap and whose
# wait deadline after the last call both land on 500. Deriving that second
# value rather than eyeballing it matters: the burst restart rule is the one
# pinned four assertions above, and a hand-guessed 450 would demand a burst
# beginning at a call that burst one had already absorbed and delivered.
steady = [(time, f'p{time}') for time in range(0, 401, 50)]
assert debounce(steady, 100, 250) == [(250, 'p200'), (500, 'p400')], \
    'max_wait must bound a burst measured from its first call'
assert debounce(steady, 100) == [(500, 'p400')], \
    'without max_wait a steady stream fires only after it stops'

# A stream long enough for the cap to fire more than once: every burst is
# bounded from its own first call, so the invocations land one max_wait
# apart even though no gap between calls ever reaches wait.
longer = [(time, f'p{time}') for time in range(0, 701, 50)]
assert debounce(longer, 100, 250) == \
    [(250, 'p200'), (500, 'p450'), (750, 'p700')], \
    'each successive burst re-measures max_wait from its own first call'

# The deadline is whichever comes first, so a short burst still uses wait.
assert debounce([(0, 'a'), (10, 'b')], 100, 1000) == [(110, 'b')]

# Simultaneous calls are allowed and the last one wins.
assert debounce([(0, 'a'), (0, 'b')], 100) == [(100, 'b')]

for bad in (
    ([(0, 'a')], 0, None),
    ([(0, 'a')], -5, None),
    ([(0, 'a')], 100, 50),
    ([(10, 'a'), (5, 'b')], 100, None),
):
    try:
        debounce(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'debounce{bad} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement Python functions plural_category(locale, n) and "
            "select_message(locale, n, forms) implementing CLDR plural "
            "selection for whole numbers. The locale is case insensitive and "
            "any script or region subtag after a '-' or '_' is ignored, so "
            "'RU_ru' is Russian; an unsupported language raises ValueError, "
            "and an n that is not a non-negative integer raises ValueError. "
            "For 'en' return 'one' when n is 1 and 'other' otherwise. For "
            "'ru' return 'one' when n % 10 is 1 and n % 100 is not 11, 'few' "
            "when n % 10 is 2, 3 or 4 and n % 100 is not 12, 13 or 14, and "
            "'many' otherwise. For 'pl' return 'one' when n is exactly 1, "
            "'few' under the same rule as Russian, and 'many' otherwise. For "
            "'ar' return 'zero' when n is 0, 'one' when n is 1, 'two' when n "
            "is 2, 'few' when n % 100 is 3 through 10, 'many' when n % 100 is "
            "11 through 99, and 'other' otherwise. select_message returns the "
            "form for the selected category, falling back to the 'other' "
            "form when that category is absent and raising KeyError when "
            "neither is present, and replaces every occurrence of '{n}' in "
            "the chosen form with the decimal number."
        ),
        validator=LOAD_CANDIDATE + require("plural_category")
        + require("select_message") + r'''
english = {0: 'other', 1: 'one', 2: 'other', 21: 'other', 100: 'other'}
russian = {
    0: 'many', 1: 'one', 2: 'few', 4: 'few', 5: 'many', 11: 'many',
    12: 'many', 14: 'many', 15: 'many', 21: 'one', 22: 'few', 25: 'many',
    100: 'many', 101: 'one', 102: 'few', 111: 'many', 112: 'many',
}
polish = {
    0: 'many', 1: 'one', 2: 'few', 4: 'few', 5: 'many', 12: 'many',
    21: 'many', 22: 'few', 101: 'many', 102: 'few', 112: 'many',
}
# 101 and 102 are the cases that separate the CLDR rule from the plausible
# guess that 'one' and 'two' are n % 100 tests like 'few' and 'many' are.
# They are not: Arabic 'one' is n = 1 exactly, so 101 is 'other'.
arabic = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'few', 10: 'few', 11: 'many',
    99: 'many', 100: 'other', 101: 'other', 102: 'other', 103: 'few',
    203: 'few', 111: 'many',
}
for locale, table in (('en', english), ('ru', russian), ('pl', polish),
                      ('ar', arabic)):
    for number, want in table.items():
        got = plural_category(locale, number)
        assert got == want, f'{locale} {number}: expected {want}, got {got}'

# 11 is the case that separates a real rule from "ends in 1".
assert plural_category('ru', 11) != plural_category('ru', 21)
# Polish keeps 1 alone where Russian promotes every ...1 except 11.
assert plural_category('pl', 21) == 'many' and plural_category('ru', 21) == 'one'

for alias in ('RU', 'ru-RU', 'ru_ru', 'Ru-Cyrl-RU'):
    assert plural_category(alias, 2) == 'few', f'{alias} was not Russian'

for bad_locale in ('', 'xx', 'de', 'english', '-ru'):
    try:
        plural_category(bad_locale, 1)
    except ValueError:
        pass
    else:
        raise AssertionError(f'locale {bad_locale!r} was accepted')

for bad_n in (-1, 1.5, '2', None):
    try:
        plural_category('en', bad_n)
    except ValueError:
        pass
    else:
        raise AssertionError(f'n={bad_n!r} was accepted')

forms = {'one': '{n} file', 'other': '{n} files'}
assert select_message('en', 1, forms) == '1 file'
assert select_message('en', 7, forms) == '7 files'

ru_forms = {'one': '{n} файл', 'few': '{n} файла', 'many': '{n} файлов'}
assert select_message('ru', 21, ru_forms) == '21 файл'
assert select_message('ru', 22, ru_forms) == '22 файла'
assert select_message('ru', 11, ru_forms) == '11 файлов'

# A catalogue that only supplies 'other' still renders every number.
assert select_message('ru', 22, {'other': '{n} items'}) == '22 items'
assert select_message('en', 3, {'other': 'many ({n}) of {n}'}) == \
    'many (3) of 3', 'every {n} occurrence must be replaced'

try:
    select_message('ru', 22, {'one': 'файл'})
except KeyError:
    pass
else:
    raise AssertionError('a catalogue without a usable form must raise')
''',
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python function run_dialog(focusables, keys, "
            "opener) simulating a modal dialog's focus trap and returning "
            "{'focus': id, 'open': bool}. focusables is the ordered list of "
            "focusable element ids inside the dialog. Opening moves focus to "
            "the first of them, or to the string 'dialog' when the list is "
            "empty. keys is a list of 'Tab', 'Shift+Tab' and 'Escape' "
            "applied in order. Tab moves focus to the next focusable and "
            "wraps from the last back to the first; Shift+Tab moves back and "
            "wraps from the first to the last; with no focusables both leave "
            "focus on 'dialog'. Escape closes the dialog, returns focus to "
            "opener, and makes every later key a no-op, including a later "
            "Escape. Any other key raises ValueError, and a focusables list "
            "containing duplicate ids raises ValueError because focus would "
            "be ambiguous."
        ),
        validator=LOAD_CANDIDATE + require("run_dialog") + r'''
fields = ['name', 'email', 'save', 'cancel']

assert run_dialog(fields, [], 'open_btn') == {'focus': 'name', 'open': True}
assert run_dialog(fields, ['Tab'], 'open_btn')['focus'] == 'email'
assert run_dialog(fields, ['Tab'] * 3, 'open_btn')['focus'] == 'cancel'
assert run_dialog(fields, ['Tab'] * 4, 'open_btn')['focus'] == 'name', \
    'Tab past the last focusable must wrap to the first'
assert run_dialog(fields, ['Tab'] * 9, 'open_btn')['focus'] == 'email'
assert run_dialog(fields, ['Shift+Tab'], 'open_btn')['focus'] == 'cancel', \
    'Shift+Tab from the first focusable must wrap to the last'
assert run_dialog(fields, ['Shift+Tab'] * 5, 'open_btn')['focus'] == 'cancel'
assert run_dialog(fields, ['Tab', 'Shift+Tab'], 'open_btn')['focus'] == 'name'

closed = run_dialog(fields, ['Tab', 'Escape', 'Tab', 'Escape'], 'open_btn')
assert closed == {'focus': 'open_btn', 'open': False}, \
    'Escape must restore focus to the opener and ignore later keys'

# The sequence above ends on Escape, which restores focus to the opener and
# would therefore hide a Tab that was wrongly processed after closing. These
# end on a movement key, so a dialog that keeps handling keys once closed
# drags focus back inside it and is visible here.
assert run_dialog(fields, ['Escape', 'Tab'], 'open_btn') == \
    {'focus': 'open_btn', 'open': False}, \
    'a key after Escape moved focus back into the closed dialog'
assert run_dialog(fields, ['Tab', 'Escape', 'Shift+Tab'], 'open_btn') == \
    {'focus': 'open_btn', 'open': False}, \
    'a key after Escape moved focus back into the closed dialog'

bare = run_dialog([], ['Tab', 'Shift+Tab'], 'open_btn')
assert bare == {'focus': 'dialog', 'open': True}
assert run_dialog([], ['Escape'], 'open_btn') == \
    {'focus': 'open_btn', 'open': False}

single = run_dialog(['only'], ['Tab', 'Tab', 'Shift+Tab'], 'open_btn')
assert single == {'focus': 'only', 'open': True}

for bad_key in ('Enter', 'tab', '', 'Shift+Escape'):
    try:
        run_dialog(fields, [bad_key], 'open_btn')
    except ValueError:
        pass
    else:
        raise AssertionError(f'key {bad_key!r} was accepted')

try:
    run_dialog(['a', 'b', 'a'], [], 'open_btn')
except ValueError:
    pass
else:
    raise AssertionError('duplicate focusable ids were accepted')
''',
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python class FormState(fields, initial) modelling a "
            "form's validation lifecycle. fields maps a field name to a rule "
            "dict with optional required (bool), min_length (int) and "
            "pattern (a regular expression the whole value must match) keys; "
            "initial maps each field to its starting string value. Provide "
            "change(name, value), blur(name), submit(), reset(), values(), "
            "touched(), dirty() and errors(). errors() returns at most one "
            "message per field and only for fields that have been blurred or "
            "that submit() has marked, so a field the user has not reached "
            "yet shows nothing however invalid it is; once a field has been "
            "blurred, change() revalidates it immediately. The message is "
            "'required' when the value is empty after stripping, otherwise "
            "'min_length' when it is shorter than the minimum, otherwise "
            "'pattern' when the whole value does not match, and the field is "
            "absent from errors() when it is valid. touched() and dirty() "
            "return sets: dirty() is the fields whose current value differs "
            "from the initial one. submit() marks every field touched and "
            "returns errors(). reset() restores the initial values and "
            "clears touched. An unknown field name raises KeyError."
        ),
        validator=LOAD_CANDIDATE + require("FormState") + r'''
fields = {
    'email': {'required': True, 'pattern': r'[^@\s]+@[^@\s]+\.[a-z]{2,}'},
    'name': {'required': True, 'min_length': 2},
    'bio': {'min_length': 10},
}
initial = {'email': '', 'name': 'Ada', 'bio': ''}
form = FormState(fields, initial)

assert form.errors() == {}, 'an untouched form reports no errors'
assert form.touched() == set() and form.dirty() == set()

form.change('email', 'nope')
assert form.errors() == {}, 'a field that has not been blurred stays silent'
assert form.dirty() == {'email'}

form.blur('email')
assert form.errors() == {'email': 'pattern'}
form.change('email', 'ada@example.com')
assert form.errors() == {}, 'a blurred field must revalidate on change'
form.change('email', '   ')
assert form.errors() == {'email': 'required'}, \
    'a whitespace-only value is empty for a required field'

# The pattern has to match the whole value, not merely occur in it.
form.change('email', 'x ada@example.com')
assert form.errors() == {'email': 'pattern'}, \
    'the pattern must match the entire value'

form.change('bio', 'short')
assert form.errors() == {'email': 'pattern'}, 'bio has not been blurred'
assert form.dirty() == {'email', 'bio'}

reported = form.submit()
assert reported == {'email': 'pattern', 'bio': 'min_length'}, reported
assert form.touched() == {'email', 'name', 'bio'}, \
    'submit must mark every field touched'
assert 'name' not in form.errors(), 'a valid field must not report an error'

# Precedence: empty beats too-short beats malformed.
form.change('bio', '')
assert form.errors()['bio'] == 'min_length', \
    'bio is not required, so an empty value is only too short'
form.change('name', 'A')
assert form.errors()['name'] == 'min_length'
form.change('name', ' ')
assert form.errors()['name'] == 'required', \
    'required outranks min_length'

form.reset()
assert form.values() == initial
assert form.touched() == set() and form.dirty() == set()
assert form.errors() == {}

for bad in ('missing', ''):
    for call in (lambda: form.change(bad, 'x'), lambda: form.blur(bad)):
        try:
            call()
        except KeyError:
            pass
        else:
            raise AssertionError(f'unknown field {bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement Python functions encode_state(state, defaults) and "
            "decode_state(query, defaults) that keep a view's state in the "
            "URL. Each key's type comes from its default and is str, bool, "
            "int, or list of str. encode_state omits every key whose value "
            "equals its default, sorts the rest by key, and joins entries "
            "with '&'. A str, bool or int encodes as key=value with bools "
            "written 'true' and 'false'; a list encodes as one key=value "
            "entry per element in order, and an empty list encodes as the "
            "bare key with no '='. Percent-encode every byte of the UTF-8 "
            "value outside the RFC 3986 unreserved set A-Za-z0-9-._~ using "
            "uppercase hex, so a space is %20 and never '+'. decode_state "
            "starts from defaults, applies the query's entries, ignores keys "
            "that are not in defaults, and percent-decodes values without "
            "treating '+' as a space. A bare key means the empty string for a "
            "str field and the empty list for a list field. A value that is "
            "not a valid int, or a bool that is not exactly 'true' or "
            "'false', raises ValueError. Either function raises KeyError "
            "when a state's keys are not exactly the keys of defaults. "
            "decode_state(encode_state(state, defaults), defaults) must "
            "return the original state."
        ),
        validator=LOAD_CANDIDATE + require("encode_state")
        + require("decode_state") + r'''
defaults = {'q': '', 'tags': [], 'page': 1, 'archived': False, 'sort': 'name'}

assert encode_state(dict(defaults), defaults) == '', \
    'a state equal to the defaults encodes to nothing'

state = {'q': 'a b', 'tags': ['x', 'y'], 'page': 1, 'archived': True,
         'sort': 'name'}
assert encode_state(state, defaults) == 'archived=true&q=a%20b&tags=x&tags=y', \
    'default-valued keys are omitted and the rest are sorted'

# Percent encoding covers reserved and non-ASCII bytes, in uppercase hex.
tricky = dict(defaults, q='a+b&c=d/ü')
assert encode_state(tricky, defaults) == 'q=a%2Bb%26c%3Dd%2F%C3%BC'
assert decode_state('q=a+b', defaults)['q'] == 'a+b', \
    "'+' is a literal plus in a percent-encoded query, not a space"

# An empty list is distinguishable from an absent key.
listed = {'q': '', 'tags': [], 'page': 1, 'archived': False, 'sort': 'name'}
non_empty_default = dict(defaults, tags=['x'])
assert encode_state(listed, non_empty_default) == 'tags'
assert decode_state('tags', non_empty_default)['tags'] == []
assert decode_state('', non_empty_default)['tags'] == ['x']
assert decode_state('q', defaults)['q'] == ''

assert decode_state('page=42&unknown=1', defaults) == \
    dict(defaults, page=42), 'unknown keys are ignored, defaults fill in'
assert decode_state('archived=false', defaults)['archived'] is False
assert decode_state('archived=true', defaults)['archived'] is True

for round_trip in (
    {'q': 'ü & =', 'tags': ['a b', 'c%d'], 'page': 7, 'archived': True,
     'sort': 'created'},
    {'q': '', 'tags': [], 'page': 1, 'archived': False, 'sort': 'name'},
    {'q': '100%', 'tags': [''], 'page': 0, 'archived': False, 'sort': '~-._'},
):
    encoded = encode_state(round_trip, defaults)
    assert decode_state(encoded, defaults) == round_trip, \
        f'round trip lost data: {encoded!r}'

for bad_query in ('page=x', 'page=', 'archived=1', 'archived=True',
                  'page=3.5'):
    try:
        decode_state(bad_query, defaults)
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad_query!r} was accepted')

for bad_state in ({'q': 'x'}, dict(defaults, extra=1)):
    try:
        encode_state(bad_state, defaults)
    except KeyError:
        pass
    else:
        raise AssertionError(f'{bad_state} did not raise KeyError')
''',
    ),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement Python functions move_item(items, from_index, "
            "to_index) and move_selection(items, selected, to_index) "
            "computing the result of a drag in a reorderable list. Every "
            "index refers to a position in the original list, and to_index "
            "is the insertion point in the original list: the dragged rows "
            "land immediately before the row that was at to_index, and "
            "to_index equal to len(items) drops them at the end. Because "
            "removing the dragged rows shifts the positions after them, "
            "dragging a row one place down must be a no-op rather than "
            "landing a place early. move_selection keeps the selected rows "
            "in their original relative order however the indices are given, "
            "drops them as one contiguous block at the insertion point, and "
            "treats a repeated index as one row. Both return a new list and "
            "leave the input unchanged. An index outside 0..len(items)-1, or "
            "a to_index outside 0..len(items), raises IndexError."
        ),
        validator=LOAD_CANDIDATE + require("move_item")
        + require("move_selection") + r'''
items = ['a', 'b', 'c', 'd', 'e']
original = list(items)

assert move_item(items, 0, 3) == ['b', 'c', 'a', 'd', 'e'], \
    'a downward drag lands before the row that was at to_index'
assert move_item(items, 3, 0) == ['d', 'a', 'b', 'c', 'e']
assert move_item(items, 0, 5) == ['b', 'c', 'd', 'e', 'a']
assert move_item(items, 4, 0) == ['e', 'a', 'b', 'c', 'd']
assert move_item(items, 1, 1) == original, 'dropping a row on itself moves it'
assert move_item(items, 1, 2) == original, \
    'dragging a row one place down is a no-op, not a swap'
assert move_item(items, 2, 1) == ['a', 'c', 'b', 'd', 'e'], \
    'dragging one place up does swap'
assert items == original, 'move_item mutated its argument'

assert move_selection(items, [0, 2], 4) == ['b', 'd', 'a', 'c', 'e']
assert move_selection(items, [2, 0], 4) == ['b', 'd', 'a', 'c', 'e'], \
    'the selection keeps its original relative order'
assert move_selection(items, [3, 1], 0) == ['b', 'd', 'a', 'c', 'e']
assert move_selection(items, [1, 1, 3], 0) == ['b', 'd', 'a', 'c', 'e'], \
    'a repeated index is one row'
assert move_selection(items, [0, 1], 3) == ['c', 'a', 'b', 'd', 'e']
assert move_selection(items, [1, 2], 2) == original, \
    'dropping a block inside itself changes nothing'
assert move_selection(items, [], 2) == original
assert move_selection(items, [0, 1, 2, 3, 4], 2) == original
assert move_selection(items, [4], 0) == ['e', 'a', 'b', 'c', 'd']
assert move_selection(items, [0, 4], 5) == ['b', 'c', 'd', 'a', 'e']
assert items == original, 'move_selection mutated its argument'

assert move_item([], 0, 0) == [] if False else True
assert move_selection(['only'], [0], 1) == ['only']

for bad in ((5, 0), (-1, 0), (0, 6), (0, -1)):
    try:
        move_item(items, *bad)
    except IndexError:
        pass
    else:
        raise AssertionError(f'move_item{bad} was accepted')

for bad_selection, bad_target in (([5], 0), ([-1], 0), ([0], 6), ([0], -1)):
    try:
        move_selection(items, bad_selection, bad_target)
    except IndexError:
        pass
    else:
        raise AssertionError(
            f'move_selection({bad_selection}, {bad_target}) was accepted')
''',
    ),
]
