"""Held-out tasks: architecture, multi-file change, and integration.

What separates this family from the rest of the course is that the candidate
does not own the whole problem. Six of the tasks below hand it code it did
not write and cannot change -- a legacy store, a host's extension point, two
transactional resources -- and score whether the new code fits *that* seam
rather than a seam of its own choosing. The others score the reasoning that
spans modules rather than living inside one: which import directions a
layering permits, which upgrade path a set of migrations admits, which source
a configuration value came from.

The failure this family exists to catch is the plausible local decision that
is wrong globally. Translating a legacy error into the caller's exception
type is invisible until a caller's `except` clause does not fire. Committing
two stores in sequence looks atomic until the second one refuses and the
first is already durable. Reading a configuration override with `bool(value)`
works for every string except `"false"`. Each of those passes a test written
against the module in isolation and fails the moment the other side of the
seam is real, so every validator here supplies the other side.

Where a task ships fixture modules they are written to be unhelpful in the
specific way real code is: the legacy store raises one exception type for
absence and for malformed input, holds text where the caller has bytes, and
counts its own reads so a validator can prove the adapter did not consult it
twice. Nothing in a fixture is a hint.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "architecture_multifile_integration"

#: The pre-existing store for -0004. It raises one error type for everything,
#: stores text, and counts fetches so the validator can prove what was read.
_LEGACY_STORE = '''\
"""The store that already exists. It cannot be changed."""


class LegacyError(Exception):
    """Raised for every failure, including a row that is simply absent."""


class LegacyStore:
    def __init__(self):
        self._rows = {}
        self.fetches = 0
        self.puts = 0

    def fetch(self, key):
        self.fetches += 1
        if key not in self._rows:
            raise LegacyError(f"no row named {key!r}")
        return self._rows[key]

    def put(self, key, blob, expires_at):
        self.puts += 1
        if not isinstance(blob, str):
            raise LegacyError("the legacy store holds text, not bytes")
        self._rows[key] = (blob, expires_at)

    def drop(self, key):
        self._rows.pop(key, None)

    def keys(self):
        return sorted(self._rows)
'''

#: The interface the rest of the application is already written against.
_CACHE_API = '''\
"""The interface every caller in the application is written against."""


class CacheMiss(KeyError):
    """The key is absent or has expired."""
'''

#: Two transactional resources for -0005. Neither knows about the other, and
#: neither can un-commit, which is the whole difficulty of the task.
_STORES = '''\
"""Two independent transactional resources. Neither can be changed."""


class StoreError(Exception):
    """A resource refused an operation."""


class Store:
    def __init__(self, name, fail_on_commit=False):
        self.name = name
        self.committed = {}
        self.staged = {}
        self.log = []
        self.fail_on_commit = fail_on_commit

    def stage(self, key, value):
        self.log.append(("stage", key))
        self.staged[key] = value

    def commit(self):
        self.log.append(("commit",))
        if self.fail_on_commit:
            raise StoreError(f"{self.name} refused to commit")
        self.committed.update(self.staged)
        self.staged.clear()

    def rollback(self):
        self.log.append(("rollback",))
        self.staged.clear()
'''

#: The host's extension point for -0006. Handlers arrive with a priority and
#: a registration order, and the registry deliberately preserves both.
_PLUGIN_API = '''\
"""The host's extension point. Plugins register through it; it is fixed."""


class UnknownEvent(Exception):
    """No handler is registered for the event."""


class Registry:
    def __init__(self):
        self._handlers = {}

    def register(self, event, name, priority=0):
        def decorate(function):
            self._handlers.setdefault(event, []).append(
                (priority, name, function))
            return function
        return decorate

    def handlers_for(self, event):
        """Registrations for an event, in the order they were registered."""
        return list(self._handlers.get(event, ()))

    def events(self):
        return sorted(self._handlers)
'''

#: The append-only log for -0007. It records every offset it is asked to read
#: from, which is how the validator tells an incremental catch-up from a
#: projection that quietly rebuilds the world on every call.
_EVENT_LOG = '''\
"""The append-only event log that already exists. It cannot be changed."""


class LogError(Exception):
    """The log refused a read."""


class EventLog:
    def __init__(self, events=()):
        self._events = [dict(event) for event in events]
        self.offsets_requested = []

    def read_from(self, offset):
        """Every record at or after `offset`, as (offset, event) pairs."""
        if not isinstance(offset, int) or isinstance(offset, bool) \\
                or offset < 0:
            raise LogError(f"bad offset {offset!r}")
        self.offsets_requested.append(offset)
        return [(index, dict(event))
                for index, event in enumerate(self._events)
                if index >= offset]

    def append(self, event):
        self._events.append(dict(event))

    def __len__(self):
        return len(self._events)
'''

#: The components for -0008. Construction and teardown both append to a shared
#: trace, so the validator can assert the exact order rather than the fact
#: that a teardown happened at all.
_COMPONENTS = '''\
"""The components that already exist. They cannot be changed."""


class ComponentError(Exception):
    """A component refused to start."""


class Component:
    def __init__(self, name, trace, fails=False):
        self.name = name
        self.trace = trace
        self.closed = False
        if fails:
            trace.append(("failed", name))
            raise ComponentError(f"{name} refused to start")
        trace.append(("started", name))

    def close(self):
        if self.closed:
            raise ComponentError(f"{self.name} was closed twice")
        self.closed = True
        self.trace.append(("closed", self.name))
'''

#: The domain model for -0009. `Money` refuses a float outright, so a
#: candidate that divides in binary floating point cannot hide the loss behind
#: a rounded repr -- it either carries an exact Decimal or it does not.
_DOMAIN = '''\
"""The domain model the application is written against. It is fixed."""

import enum
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


class TranslationError(Exception):
    """The upstream record cannot be represented in this domain."""


class Status(enum.Enum):
    PENDING = "pending"
    SETTLED = "settled"
    FAILED = "failed"


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self):
        if not isinstance(self.amount, Decimal):
            raise TypeError("Money holds a Decimal, never a float")
        if self.currency != self.currency.upper() or len(self.currency) != 3:
            raise ValueError("currency is a three-letter upper-case code")


@dataclass(frozen=True)
class Payment:
    reference: str
    money: Money
    occurred_at: datetime
    status: Status
'''

#: The two pricers for -0010. Both record every order they are asked to
#: price, which is what makes "the shadow call happened exactly once and
#: changed nothing" an assertable property rather than a claim.
_PRICERS = '''\
"""Two pricing implementations that already exist. Neither can change."""


class PricingError(Exception):
    """The pricer could not price the order."""


class LegacyPricer:
    def __init__(self, raise_on=()):
        self.calls = []
        self._raise = set(raise_on)

    def price(self, order):
        self.calls.append(order["id"])
        if order["id"] in self._raise:
            raise PricingError("legacy cannot price this order")
        return order["quantity"] * 100


class ModernPricer:
    def __init__(self, disagree_on=(), raise_on=()):
        self.calls = []
        self._disagree = set(disagree_on)
        self._raise = set(raise_on)

    def price(self, order):
        self.calls.append(order["id"])
        if order["id"] in self._raise:
            raise PricingError("modern is not ready for this order")
        if order["id"] in self._disagree:
            return order["quantity"] * 100 + 1
        return order["quantity"] * 100
'''

#: The gateway for -0011. It replays a key it has already seen and counts the
#: money it actually captured, so a retry that invents a fresh key per attempt
#: is visible as a second capture rather than as a successful retry.
_GATEWAY = '''\
"""The payment gateway that already exists. It cannot be changed."""


class TransientError(Exception):
    """A temporary failure. The identical request may be retried."""


class DeclinedError(Exception):
    """The issuer refused. Retrying will not help."""


class Gateway:
    def __init__(self, transient_failures=0, decline=False):
        self._remaining = transient_failures
        self._decline = decline
        self._by_key = {}
        self.captured_total = 0
        self.attempts = 0

    def charge(self, idempotency_key, amount_cents):
        self.attempts += 1
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("an idempotency key is required")
        if idempotency_key in self._by_key:
            record = dict(self._by_key[idempotency_key])
            record["replayed"] = True
            return record
        if self._remaining > 0:
            self._remaining -= 1
            raise TransientError("the gateway is briefly unavailable")
        if self._decline:
            raise DeclinedError("the issuer declined the card")
        self.captured_total += amount_cents
        record = {"id": f"ch_{len(self._by_key) + 1}",
                  "amount_cents": amount_cents, "replayed": False}
        self._by_key[idempotency_key] = record
        return dict(record)
'''

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function audit(imports, layers, order) "
            "checking a codebase against a layered architecture. imports "
            "maps a module name to the list of modules it imports. layers "
            "maps a module name to its layer. order lists the layer names "
            "from the lowest to the highest, so the highest layer may depend "
            "on everything below it. Return {'violations': [...], 'cycles': "
            "[...], 'unlayered': [...]}. A violation is an (importer, "
            "imported) pair where the importer's layer is strictly lower "
            "than the imported module's, because a lower layer must not know "
            "about a higher one; an import within one layer is allowed. "
            "Report violations sorted. A cycle is a set of two or more "
            "modules that can each reach the others through imports, or a "
            "single module that imports itself; report each as a tuple of "
            "its members sorted, and the list of them sorted. unlayered is "
            "the sorted list of module names that appear in imports as an "
            "importer or an imported module but have no entry in layers; an "
            "import touching one is not checked for a layer violation, "
            "though it still counts towards cycles. Raise ValueError when a "
            "module's layer is not in order, or when order repeats a layer."
        ),
        validator=LOAD_CANDIDATE + require("audit") + r'''
order = ['domain', 'application', 'interface']
clean = {
    'domain.model': [],
    'application.service': ['domain.model'],
    'application.ports': ['domain.model'],
    'interface.http': ['application.service', 'application.ports'],
}
layers = {
    'domain.model': 'domain',
    'application.service': 'application',
    'application.ports': 'application',
    'interface.http': 'interface',
}
assert audit(clean, layers, order) == \
    {'violations': [], 'cycles': [], 'unlayered': []}

# Same-layer imports are allowed; the rule is about direction, not distance.
sideways = dict(clean, **{'application.service': ['application.ports']})
assert audit(sideways, layers, order)['violations'] == []

# The defect a layering exists to prevent: the domain reaching up into the
# delivery mechanism, which makes the domain unusable without a web server.
inverted = dict(clean, **{'domain.model': ['interface.http']})
assert audit(inverted, layers, order)['violations'] == \
    [('domain.model', 'interface.http')], \
    'a lower layer importing a higher one is the violation'

skipping = dict(clean, **{'application.ports': ['interface.http']})
assert audit(skipping, layers, order)['violations'] == \
    [('application.ports', 'interface.http')]

many = dict(clean, **{
    'domain.model': ['interface.http', 'application.service'],
})
assert audit(many, layers, order)['violations'] == [
    ('domain.model', 'application.service'),
    ('domain.model', 'interface.http'),
], 'violations are reported sorted'

# Cycles are found regardless of layer, including within one layer where no
# violation is reported at all.
cyclic = dict(clean, **{
    'application.service': ['domain.model', 'application.ports'],
    'application.ports': ['application.service'],
})
report = audit(cyclic, layers, order)
assert report['violations'] == [], 'a same-layer cycle is not a violation'
assert report['cycles'] == [('application.ports', 'application.service')]

three = {
    'a': ['b'], 'b': ['c'], 'c': ['a'], 'd': ['a'],
}
three_layers = {name: 'domain' for name in 'abcd'}
assert audit(three, three_layers, order)['cycles'] == [('a', 'b', 'c')], \
    'every module that can reach the others is one cycle'

assert audit({'a': ['a']}, {'a': 'domain'}, order)['cycles'] == [('a',)], \
    'a module importing itself is a cycle of one'

separate = {'a': ['b'], 'b': ['a'], 'c': ['d'], 'd': ['c']}
assert audit(separate, {name: 'domain' for name in 'abcd'},
             order)['cycles'] == [('a', 'b'), ('c', 'd')]

# A module nobody assigned a layer cannot be direction-checked, but it can
# still take part in a cycle.
partial = {'domain.model': ['scripts.tool'], 'scripts.tool': ['domain.model']}
outcome = audit(partial, {'domain.model': 'domain'}, order)
assert outcome['unlayered'] == ['scripts.tool']
assert outcome['violations'] == []
assert outcome['cycles'] == [('domain.model', 'scripts.tool')]

assert audit({}, {}, order) == \
    {'violations': [], 'cycles': [], 'unlayered': []}

for bad in (
    ({'a': []}, {'a': 'nope'}, order),
    ({'a': []}, {'a': 'domain'}, ['domain', 'domain']),
):
    try:
        audit(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'audit{bad} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement Python functions plan(migrations, source, target) and "
            "migrate(migrations, document, source, target) upgrading a "
            "document between schema versions. migrations is a list of "
            "(from_version, to_version, function) triples forming a directed "
            "graph over version strings. plan returns the list of "
            "(from_version, to_version) steps on the shortest path from "
            "source to target, or the empty list when they are equal. When "
            "two different shortest paths exist the upgrade is ambiguous and "
            "plan raises ValueError rather than picking one, because the two "
            "would produce different documents. plan raises ValueError when "
            "no path exists, when a migration's two versions are equal, and "
            "when the same (from_version, to_version) pair appears twice. "
            "migrate applies each step's function in order, passing the "
            "document returned by the previous step, and returns the final "
            "document; it must not modify the document it was given. A "
            "migration graph may contain cycles and neither function may "
            "loop forever on one."
        ),
        validator=LOAD_CANDIDATE + require("plan") + require("migrate") + r'''
def rename(old, new):
    def step(document):
        updated = dict(document)
        updated[new] = updated.pop(old)
        return updated
    return step


def add(field, value):
    def step(document):
        return dict(document, **{field: value})
    return step


chain = [
    ('1', '2', add('created', 0)),
    ('2', '3', rename('name', 'title')),
    ('3', '4', add('archived', False)),
]

assert plan(chain, '1', '1') == []
assert plan(chain, '1', '2') == [('1', '2')]
assert plan(chain, '1', '4') == [('1', '2'), ('2', '3'), ('3', '4')]
assert plan(chain, '2', '4') == [('2', '3'), ('3', '4')]

original = {'name': 'report'}
result = migrate(chain, original, '1', '4')
assert result == {'created': 0, 'title': 'report', 'archived': False}, result
assert original == {'name': 'report'}, 'migrate modified its argument'
assert migrate(chain, original, '1', '1') == original
assert migrate(chain, original, '1', '1') is not original, \
    'an empty plan still returns a document the caller can own'

# A shortcut edge makes the shorter path the right one.
with_shortcut = chain + [('1', '4', add('shortcut', True))]
assert plan(with_shortcut, '1', '4') == [('1', '4')]
assert migrate(with_shortcut, original, '1', '4') == \
    {'name': 'report', 'shortcut': True}

# Two shortest paths of equal length produce different documents, so the
# upgrade has no single answer and guessing one silently corrupts data.
diamond = [
    ('1', '2a', add('via', 'a')),
    ('1', '2b', add('via', 'b')),
    ('2a', '3', add('done', True)),
    ('2b', '3', add('done', True)),
]
try:
    plan(diamond, '1', '3')
except ValueError:
    pass
else:
    raise AssertionError('an ambiguous upgrade path was resolved silently')
assert plan(diamond, '1', '2a') == [('1', '2a')], \
    'an unambiguous path in the same graph still resolves'

# A cycle must not hang, and must not lengthen the path it is attached to.
looping = chain + [('4', '2', add('looped', True))]
assert plan(looping, '1', '4') == [('1', '2'), ('2', '3'), ('3', '4')]
assert plan(looping, '4', '3') == [('4', '2'), ('2', '3')]

for bad in (
    (chain, '1', '9'),
    (chain, '9', '1'),
    (chain + [('5', '5', add('x', 1))], '1', '2'),
    (chain + [('1', '2', add('x', 1))], '1', '2'),
):
    try:
        plan(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'plan{bad[1:]} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function compose(schema, sources) merging "
            "configuration from several places. schema maps a key to "
            "{'type': one of 'str', 'int', 'float', 'bool', 'list', "
            "'default': value}. sources is a list of (name, mapping) pairs "
            "ordered from the lowest precedence to the highest, so a later "
            "source overrides an earlier one. Every value may arrive as a "
            "string, because environment variables and command lines carry "
            "no types, and must be coerced to the schema's type: an int or "
            "float by parsing, a list by splitting on commas and stripping "
            "each element with an empty string giving the empty list, and a "
            "bool from exactly 'true', 'false', '1', '0', 'yes' or 'no' "
            "case-insensitively. A value already of the right type is taken "
            "as it is. Return {'values': {key: value}, 'origins': {key: "
            "source_name}, 'ignored': [(source_name, key), ...]} where "
            "origins names the source a value came from or 'default' when no "
            "source set it, and ignored lists in order every key a source "
            "supplied that the schema does not define. Raise ValueError "
            "naming the source and the key when a value cannot be coerced, "
            "and leave every source mapping unmodified."
        ),
        validator=LOAD_CANDIDATE + require("compose") + r'''
import copy

schema = {
    'host': {'type': 'str', 'default': 'localhost'},
    'port': {'type': 'int', 'default': 8080},
    'debug': {'type': 'bool', 'default': False},
    'ratio': {'type': 'float', 'default': 0.5},
    'tags': {'type': 'list', 'default': []},
}

empty = compose(schema, [])
assert empty['values'] == {'host': 'localhost', 'port': 8080,
                           'debug': False, 'ratio': 0.5, 'tags': []}
assert set(empty['origins'].values()) == {'default'}
assert empty['ignored'] == []

sources = [
    ('file', {'host': 'files.example', 'port': '9000', 'debug': 'true'}),
    ('env', {'port': '9100', 'tags': 'a, b ,c'}),
    ('cli', {'debug': 'false', 'ratio': '0.25'}),
]
snapshot = copy.deepcopy(sources)
merged = compose(schema, sources)

assert merged['values'] == {
    'host': 'files.example', 'port': 9100, 'debug': False,
    'ratio': 0.25, 'tags': ['a', 'b', 'c'],
}, merged['values']
assert merged['origins'] == {
    'host': 'file', 'port': 'env', 'debug': 'cli', 'ratio': 'cli',
    'tags': 'env',
}, 'origins must name the source that actually won'
assert sources == snapshot, 'compose modified a source mapping'

# The coercion every configuration layer gets wrong once: 'false' is a
# non-empty string, so bool() reports True and the flag can never be turned
# off from the place that overrides it.
assert compose(schema, [('cli', {'debug': 'false'})])['values']['debug'] \
    is False, "the string 'false' must coerce to False"
for text, expected in (('true', True), ('True', True), ('1', True),
                       ('yes', True), ('false', False), ('FALSE', False),
                       ('0', False), ('no', False)):
    assert compose(schema, [('cli', {'debug': text})])['values']['debug'] \
        is expected, f'{text!r} coerced wrongly'

# A value that already has its type is taken unchanged.
typed = compose(schema, [('code', {'port': 9200, 'tags': ['x'],
                                   'debug': True, 'ratio': 1.5})])
assert typed['values']['port'] == 9200 and typed['values']['tags'] == ['x']
assert typed['values']['debug'] is True and typed['values']['ratio'] == 1.5

assert compose(schema, [('env', {'tags': ''})])['values']['tags'] == []
assert compose(schema, [('env', {'tags': 'solo'})])['values']['tags'] == \
    ['solo']
assert compose(schema, [('env', {'ratio': '2'})])['values']['ratio'] == 2.0

unknown = compose(schema, [('env', {'PATH': '/usr/bin', 'host': 'h'}),
                           ('cli', {'verbose': '1'})])
assert unknown['ignored'] == [('env', 'PATH'), ('cli', 'verbose')], \
    'an unknown key is reported, not an error'
assert unknown['values']['host'] == 'h'

for bad in ({'port': 'nine'}, {'port': '9.5'}, {'ratio': 'half'},
            {'debug': 'maybe'}, {'debug': '2'}):
    try:
        compose(schema, [('env', bad)])
    except ValueError as error:
        assert 'env' in str(error), 'the error must name the source'
        assert next(iter(bad)) in str(error), 'the error must name the key'
    else:
        raise AssertionError(f'{bad} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "The module legacy_store.py already exists and cannot be "
            "changed: LegacyStore.fetch(key) returns the (blob, expires_at) "
            "pair it was given or raises LegacyError, put(key, blob, "
            "expires_at) refuses anything but text, and drop(key) removes a "
            "row. The module cache_api.py defines CacheMiss, which every "
            "caller in the application already catches. Write a class "
            "LegacyBackedCache(store, clock) presenting the modern interface "
            "over that store, where clock is a zero-argument callable "
            "returning the current time as a float. set(key, value, "
            "ttl=None) stores the bytes in value, which are arbitrary and "
            "need not be text, by encoding them with base64 into the ASCII "
            "the legacy store accepts, recording an expiry of clock() + ttl "
            "or None when ttl is None. get(key) returns the original bytes, "
            "and raises cache_api.CacheMiss when the key is absent or its "
            "expiry is at or before clock(); an expired row must also be "
            "dropped from the store so it cannot accumulate. delete(key) "
            "removes a key and does nothing for one that is not there. "
            "LegacyError must never reach a caller, and a key that is not a "
            "non-empty string raises ValueError without touching the store "
            "at all."
        ),
        fixtures={
            "legacy_store.py": _LEGACY_STORE,
            "cache_api.py": _CACHE_API,
        },
        validator=LOAD_CANDIDATE + require("LegacyBackedCache") + r'''
import base64
import cache_api
import legacy_store


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


store = legacy_store.LegacyStore()
clock = Clock()
cache = LegacyBackedCache(store, clock)

# Arbitrary bytes, not text: the legacy store would refuse these directly.
payload = b'\x00\xff\xfe binary \x80'
cache.set('k', payload)
assert cache.get('k') == payload, 'the bytes must survive the text store'
assert store.keys() == ['k']

blob, expires_at = store._rows['k']
assert isinstance(blob, str), 'the legacy store was handed something but text'
assert base64.b64decode(blob.encode('ascii')) == payload
assert expires_at is None, 'no ttl means no expiry'

# The translation the whole adapter exists for: every caller in the
# application writes `except CacheMiss`, and LegacyError would sail past it.
try:
    cache.get('absent')
except cache_api.CacheMiss:
    pass
except legacy_store.LegacyError:
    raise AssertionError('LegacyError reached the caller unchanged')
else:
    raise AssertionError('a missing key did not raise CacheMiss')

cache.set('short', b'value', 10)
assert cache.get('short') == b'value'
clock.now += 9.0
assert cache.get('short') == b'value', 'the entry is not expired yet'
clock.now += 1.0
try:
    cache.get('short')
except cache_api.CacheMiss:
    pass
else:
    raise AssertionError('an entry at exactly its expiry was served')
assert 'short' not in store.keys(), \
    'an expired row must be dropped rather than left to accumulate'

clock.now = 5000.0
assert cache.get('k') == payload, 'a ttl-less entry never expires'

cache.delete('k')
try:
    cache.get('k')
except cache_api.CacheMiss:
    pass
else:
    raise AssertionError('a deleted key was still served')
cache.delete('k')
cache.delete('never existed')

before = (store.fetches, store.puts)
for bad_key in ('', None):
    try:
        cache.get(bad_key)
    except ValueError:
        pass
    except cache_api.CacheMiss:
        raise AssertionError('an invalid key is a caller error, not a miss')
    else:
        raise AssertionError(f'get({bad_key!r}) was accepted')
    try:
        cache.set(bad_key, b'x')
    except ValueError:
        pass
    else:
        raise AssertionError(f'set({bad_key!r}) was accepted')
assert (store.fetches, store.puts) == before, \
    'an invalid key must be rejected without touching the store'
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "The module stores.py already exists and cannot be changed: a "
            "Store has stage(key, value), commit() which may raise "
            "StoreError, and rollback(), and none of them can undo a commit "
            "that already succeeded. Write a class UnitOfWork(stores), where "
            "stores maps a name to a Store, coordinating a write across all "
            "of them. stage(name, key, value) stages into one store and "
            "raises KeyError for a name that is not registered. commit() "
            "commits every store in the order stores was given. If a commit "
            "raises, roll back every store that has not yet been committed "
            "and raise PartialCommitError, a class you define, carrying a "
            "committed attribute listing the names that did commit in order "
            "and a failed attribute naming the one that raised; the stores "
            "already committed stay committed, because no protocol here can "
            "undo them. commit() returns the list of committed names, and "
            "calling it twice raises RuntimeError. UnitOfWork is also a "
            "context manager: leaving its block normally commits, and "
            "leaving it through an exception rolls every store back and lets "
            "the exception propagate."
        ),
        fixtures={"stores.py": _STORES},
        validator=LOAD_CANDIDATE + require("UnitOfWork")
        + require("PartialCommitError") + r'''
import stores as store_module


def build(*failing):
    return {
        name: store_module.Store(name, name in failing)
        for name in ('accounts', 'ledger', 'audit')
    }


registry = build()
work = UnitOfWork(registry)
work.stage('accounts', 'a', 1)
work.stage('ledger', 'l', 2)
assert work.commit() == ['accounts', 'ledger', 'audit']
assert registry['accounts'].committed == {'a': 1}
assert registry['ledger'].committed == {'l': 2}
assert registry['audit'].committed == {}

try:
    work.commit()
except RuntimeError:
    pass
else:
    raise AssertionError('a unit of work was committed twice')

try:
    UnitOfWork(build()).stage('nope', 'k', 1)
except KeyError:
    pass
else:
    raise AssertionError('staging into an unregistered store was accepted')

# The honest hard case: the second resource refuses after the first is
# already durable. There is no undo, so the report has to say what happened
# rather than pretending the whole thing rolled back.
registry = build('ledger')
work = UnitOfWork(registry)
work.stage('accounts', 'a', 1)
work.stage('ledger', 'l', 2)
work.stage('audit', 'x', 3)
try:
    work.commit()
except PartialCommitError as error:
    assert error.committed == ['accounts'], error.committed
    assert error.failed == 'ledger', error.failed
else:
    raise AssertionError('a failed commit did not report a partial commit')

assert registry['accounts'].committed == {'a': 1}, \
    'a store that committed stays committed; nothing here can undo it'
assert registry['ledger'].committed == {}
assert registry['audit'].committed == {}
assert ('rollback',) in registry['audit'].log, \
    'a store that had not committed yet must be rolled back'
assert ('commit',) not in registry['audit'].log, \
    'no store may be committed after an earlier one failed'
assert registry['audit'].staged == {}

registry = build('accounts')
work = UnitOfWork(registry)
work.stage('accounts', 'a', 1)
try:
    work.commit()
except PartialCommitError as error:
    assert error.committed == [] and error.failed == 'accounts'
else:
    raise AssertionError('the first store failing was not reported')

registry = build()
with UnitOfWork(registry) as work:
    work.stage('accounts', 'a', 1)
assert registry['accounts'].committed == {'a': 1}, \
    'leaving the block normally commits'

registry = build()
try:
    with UnitOfWork(registry) as work:
        work.stage('accounts', 'a', 1)
        raise KeyboardInterrupt('the caller gave up')
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('the block exception was swallowed')
assert registry['accounts'].committed == {}, \
    'leaving through an exception must not commit'
assert ('rollback',) in registry['accounts'].log
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "The module plugin_api.py already exists and cannot be changed: "
            "Registry.register(event, name, priority=0) is a decorator that "
            "records a handler, handlers_for(event) returns the recorded "
            "(priority, name, function) triples in registration order, and "
            "UnknownEvent is the host's error. Write a function "
            "dispatch(registry, event, payload) running every handler for an "
            "event. Handlers run from the highest priority to the lowest, "
            "with equal priorities keeping registration order, and each is "
            "called with the payload. Return {'results': [(name, value), "
            "...], 'errors': [(name, message), ...]} in the order the "
            "handlers ran, where results holds the handlers that returned "
            "something other than None and errors holds the name and the "
            "str() of the exception for the handlers that raised. One "
            "handler raising must not stop the rest, because a plugin the "
            "host does not own must not be able to take the host down. Raise "
            "UnknownEvent when no handler is registered for the event, which "
            "is different from every handler failing. Do not modify the "
            "registry, and do not let a handler's exception propagate."
        ),
        fixtures={"plugin_api.py": _PLUGIN_API},
        validator=LOAD_CANDIDATE + require("dispatch") + r'''
import plugin_api

registry = plugin_api.Registry()
ran = []


@registry.register('save', 'audit', priority=10)
def audit(payload):
    ran.append('audit')
    return f"audited {payload['id']}"


@registry.register('save', 'index')
def index(payload):
    ran.append('index')
    return None


@registry.register('save', 'broken', priority=5)
def broken(payload):
    ran.append('broken')
    raise RuntimeError('the plugin is misconfigured')


@registry.register('save', 'notify')
def notify(payload):
    ran.append('notify')
    return 'notified'


@registry.register('save', 'late', priority=-5)
def late(payload):
    ran.append('late')
    return 'late'


before = registry.handlers_for('save')
outcome = dispatch(registry, 'save', {'id': 7})

# Priority first, registration order within a priority: index registered
# before notify and both sit at the default 0.
assert ran == ['audit', 'broken', 'index', 'notify', 'late'], ran
assert outcome['results'] == [
    ('audit', 'audited 7'), ('notify', 'notified'), ('late', 'late'),
], outcome['results']
assert outcome['errors'] == [('broken', 'the plugin is misconfigured')], \
    outcome['errors']
assert registry.handlers_for('save') == before, 'dispatch mutated the registry'

# A plugin that fails must not prevent the ones after it from running: the
# host stays up and reports the failure instead.
assert 'notify' in ran and ran.index('notify') > ran.index('broken')

single = plugin_api.Registry()


@single.register('ping', 'only')
def only(payload):
    raise ValueError('every handler failed')


failed = dispatch(single, 'ping', None)
assert failed == {'results': [], 'errors': [('only', 'every handler failed')]}, \
    'every handler failing is a result, not an UnknownEvent'

try:
    dispatch(single, 'absent', None)
except plugin_api.UnknownEvent:
    pass
else:
    raise AssertionError('an event with no handlers did not raise UnknownEvent')

try:
    dispatch(plugin_api.Registry(), 'save', None)
except plugin_api.UnknownEvent:
    pass
else:
    raise AssertionError('an empty registry did not raise UnknownEvent')

passthrough = plugin_api.Registry()
seen = []


@passthrough.register('echo', 'first')
def first(payload):
    seen.append(payload)
    return payload


assert dispatch(passthrough, 'echo', {'k': 'v'})['results'] == \
    [('first', {'k': 'v'})]
assert seen == [{'k': 'v'}], 'the payload is passed through unchanged'
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "The module event_log.py already exists and cannot be changed: "
            "EventLog.read_from(offset) returns every (offset, event) pair at "
            "or after offset, append(event) adds one, and len() counts them. "
            "Write a class BalanceProjection(log) that folds that log into "
            "account balances. It exposes balances, a dict of account name to "
            "integer balance holding only open accounts; next_offset, the "
            "offset the next read will start from; catch_up(), which applies "
            "every event at or after next_offset and returns how many it "
            "applied; snapshot(), returning a JSON-serialisable dict; and a "
            "classmethod restore(log, snapshot) rebuilding a projection from "
            "one. The event types are opened, deposited, withdrew and closed, "
            "each carrying an account, with deposited and withdrew also "
            "carrying a positive integer amount. Opening an account that is "
            "already open, touching one that is not open, withdrawing more "
            "than the balance, closing an account that still holds funds, a "
            "non-positive or non-integer amount, and an unrecognised event "
            "type are each a ValueError. A rejected event must leave both the "
            "balances and next_offset exactly as they were before it, so that "
            "the same event is reconsidered rather than skipped on the next "
            "call. A catch-up must read only from next_offset."
        ),
        fixtures={"event_log.py": _EVENT_LOG},
        validator=LOAD_CANDIDATE + require("BalanceProjection") + r'''
import event_log

EVENTS = [
    {'type': 'opened', 'account': 'alice'},
    {'type': 'deposited', 'account': 'alice', 'amount': 120},
    {'type': 'opened', 'account': 'bob'},
    {'type': 'deposited', 'account': 'bob', 'amount': 40},
    {'type': 'withdrew', 'account': 'alice', 'amount': 20},
]

log = event_log.EventLog(EVENTS)
projection = BalanceProjection(log)
assert projection.catch_up() == 5, 'not every event was applied'
assert projection.balances == {'alice': 100, 'bob': 40}, projection.balances
assert projection.next_offset == 5

# A second catch-up with nothing new applies nothing and changes nothing.
assert projection.catch_up() == 0, 'events were applied twice'
assert projection.balances == {'alice': 100, 'bob': 40}

# ...and it must not have re-read the whole log to discover that.
assert log.offsets_requested[-1] == 5, (
    'the catch-up read from offset '
    f'{log.offsets_requested[-1]} rather than from next_offset'
)

log.append({'type': 'withdrew', 'account': 'bob', 'amount': 40})
log.append({'type': 'closed', 'account': 'bob'})
assert projection.catch_up() == 2, 'the appended events were not applied'
assert projection.balances == {'alice': 100}, 'a closed account still shows'

# A snapshot must survive a round trip through JSON and resume from it.
import json

carried = json.loads(json.dumps(projection.snapshot()))
resumed = BalanceProjection.restore(log, carried)
assert resumed.balances == {'alice': 100}, resumed.balances
assert resumed.next_offset == projection.next_offset
assert resumed.catch_up() == 0, 'a resumed projection re-applied the log'

# ...and resuming must land exactly where a full rebuild does.
ALL_EVENTS = EVENTS + [{'type': 'withdrew', 'account': 'bob', 'amount': 40},
                       {'type': 'closed', 'account': 'bob'}]
rebuilt = BalanceProjection(event_log.EventLog(ALL_EVENTS))
assert rebuilt.catch_up() == len(ALL_EVENTS)
assert rebuilt.balances == resumed.balances, \
    'restoring from a snapshot diverged from a full rebuild'
assert rebuilt.next_offset == resumed.next_offset

# The seam: a rejected event must be reconsidered, never stepped over.
broken = event_log.EventLog([
    {'type': 'opened', 'account': 'carol'},
    {'type': 'deposited', 'account': 'carol', 'amount': 10},
    {'type': 'withdrew', 'account': 'carol', 'amount': 99},
    {'type': 'deposited', 'account': 'carol', 'amount': 5},
])
stalled = BalanceProjection(broken)
try:
    stalled.catch_up()
except ValueError:
    pass
else:
    raise AssertionError('an overdraft was accepted')
assert stalled.balances == {'carol': 10}, \
    f'the rejected event changed the balances: {stalled.balances}'
assert stalled.next_offset == 2, (
    f'next_offset advanced to {stalled.next_offset}, so the rejected event '
    'would be skipped instead of reconsidered'
)
try:
    stalled.catch_up()
except ValueError:
    pass
else:
    raise AssertionError('the rejected event was skipped on the second call')

for bad in ([{'type': 'deposited', 'account': 'nobody', 'amount': 1}],
            [{'type': 'opened', 'account': 'x'},
             {'type': 'opened', 'account': 'x'}],
            [{'type': 'opened', 'account': 'x'},
             {'type': 'deposited', 'account': 'x', 'amount': 0}],
            [{'type': 'opened', 'account': 'x'},
             {'type': 'deposited', 'account': 'x', 'amount': 1.5}],
            [{'type': 'opened', 'account': 'x'},
             {'type': 'deposited', 'account': 'x', 'amount': 5},
             {'type': 'closed', 'account': 'x'}],
            [{'type': 'transferred', 'account': 'x'}]):
    try:
        BalanceProjection(event_log.EventLog(bad)).catch_up()
    except ValueError:
        pass
    else:
        raise AssertionError(f'{bad!r} was accepted')

assert BalanceProjection(event_log.EventLog([])).catch_up() == 0
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "The module components.py already exists and cannot be changed: "
            "Component(name, trace, fails=False) appends ('started', name) to "
            "the shared trace list, or appends ('failed', name) and raises "
            "ComponentError when fails is true, and close() appends "
            "('closed', name) and raises ComponentError if called twice. "
            "Write a class Container(specs) wiring them together, where specs "
            "maps a component name to a dict with 'requires', a list of names "
            "it depends on, and 'build', a callable taking those dependencies "
            "as keyword arguments. get(name) returns the component, building "
            "each dependency before its dependant and building any given name "
            "at most once for the life of the container. An unknown name is a "
            "KeyError and a dependency cycle is a ValueError. close() closes "
            "every component that was built, in the reverse of the order they "
            "were built, exactly once each, and is a no-op when called again. "
            "If a build raises, every component that this particular get() "
            "call brought into existence must be closed in reverse order and "
            "the original exception must propagate, while components built by "
            "earlier calls stay open and registered."
        ),
        fixtures={"components.py": _COMPONENTS},
        validator=LOAD_CANDIDATE + require("Container") + r'''
import components


def spec(name, trace, requires=(), fails=False):
    def build(**resolved):
        return components.Component(name, trace, fails=fails)
    return {'requires': list(requires), 'build': build}


trace = []
specs = {
    'config': spec('config', trace),
    'pool': spec('pool', trace, ['config']),
    'cache': spec('cache', trace, ['config']),
    'api': spec('api', trace, ['pool', 'cache']),
}
container = Container(specs)
api = container.get('api')
assert api.name == 'api'

started = [name for kind, name in trace if kind == 'started']
assert started[0] == 'config', f'config was not built first: {started}'
assert started.index('pool') < started.index('api')
assert started.index('cache') < started.index('api')
assert sorted(started) == ['api', 'cache', 'config', 'pool']

# A second get must reuse, not rebuild.
assert container.get('api') is api
assert container.get('config') is not None
assert [name for kind, name in trace if kind == 'started'] == started, \
    'a component was built twice'

container.close()
closed = [name for kind, name in trace if kind == 'closed']
assert closed == list(reversed(started)), (
    f'closed in {closed}, which is not the reverse of the build order '
    f'{started}'
)
container.close()
assert [name for kind, name in trace if kind == 'closed'] == closed, \
    'a second close() closed something a second time'

# The seam: a failure part-way through one get() unwinds only what that call
# built, in reverse, and leaves the earlier component alone.
trace = []
specs = {
    'base': spec('base', trace),
    'left': spec('left', trace, ['base']),
    'right': spec('right', trace, ['base'], fails=True),
    'top': spec('top', trace, ['left', 'right']),
}
container = Container(specs)
base = container.get('base')
assert trace == [('started', 'base')]

try:
    container.get('top')
except components.ComponentError:
    pass
else:
    raise AssertionError('the failing build did not propagate')

after = trace[1:]
assert ('started', 'left') in after and ('failed', 'right') in after
assert ('closed', 'left') in after, \
    'the component built by the failing call was left open'
assert ('closed', 'base') not in after, (
    'a component built by an earlier call was closed by an unrelated '
    'failure'
)
assert not base.closed
closed_after = [name for kind, name in after if kind == 'closed']
started_after = [name for kind, name in after if kind == 'started']
assert closed_after == list(reversed(started_after)), (
    f'partial teardown ran {closed_after}, not the reverse of '
    f'{started_after}'
)

# The container is still usable, and did not keep the half-built component.
assert container.get('left') is not None
assert [name for kind, name in trace if kind == 'started'].count('left') == 2

for bad in ('absent', 'also absent'):
    try:
        Container(specs).get(bad)
    except KeyError:
        pass
    else:
        raise AssertionError(f'{bad!r} was resolved')

cyclic = {
    'a': spec('a', [], ['b']),
    'b': spec('b', [], ['c']),
    'c': spec('c', [], ['a']),
}
try:
    Container(cyclic).get('a')
except ValueError:
    pass
else:
    raise AssertionError('a dependency cycle was not reported')

selfish = {'a': spec('a', [], ['a'])}
try:
    Container(selfish).get('a')
except ValueError:
    pass
else:
    raise AssertionError('a self-dependency was not reported')
''',
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "The module domain.py already exists and cannot be changed: it "
            "defines TranslationError, a Status enum of PENDING, SETTLED and "
            "FAILED, a frozen Money(amount, currency) that rejects anything "
            "but a Decimal amount and a three-letter upper-case currency, and "
            "a frozen Payment(reference, money, occurred_at, status). Write a "
            "function translate(record) turning one upstream record into a "
            "Payment. The upstream shape is fixed and unhelpful: 'ref' is the "
            "reference, 'amount_cents' is a non-negative integer number of "
            "minor units, 'ccy' is a lower-case currency code, 'ts' is an "
            "integer count of seconds since the Unix epoch in UTC, and "
            "'state' is 'P', 'S' or 'F'. The Payment's amount must be the "
            "exact decimal value of those minor units, its occurred_at must "
            "be a timezone-aware UTC datetime, and its status must be the "
            "matching Status member. Any missing key, unrecognised state, "
            "malformed currency, or negative or non-integer amount must raise "
            "domain.TranslationError, and no other exception type may reach "
            "the caller."
        ),
        fixtures={"domain.py": _DOMAIN},
        validator=LOAD_CANDIDATE + require("translate") + r'''
import datetime
import domain
from decimal import Decimal

payment = translate({'ref': 'r-1', 'amount_cents': 1250, 'ccy': 'usd',
                     'ts': 1700000000, 'state': 'S'})
assert isinstance(payment, domain.Payment)
assert payment.reference == 'r-1'
assert payment.money.currency == 'USD', 'the currency was not normalised'
assert payment.money.amount == Decimal('12.50')
assert payment.status is domain.Status.SETTLED, 'status is not the enum member'

assert payment.occurred_at.tzinfo is not None, 'occurred_at is naive'
assert payment.occurred_at.utcoffset() == datetime.timedelta(0), \
    'occurred_at is not UTC'
assert payment.occurred_at == datetime.datetime(
    2023, 11, 14, 22, 13, 20, tzinfo=datetime.timezone.utc), \
    payment.occurred_at

assert translate({'ref': 'r', 'amount_cents': 0, 'ccy': 'eur', 'ts': 0,
                  'state': 'P'}).money.amount == Decimal('0')
assert translate({'ref': 'r', 'amount_cents': 7, 'ccy': 'eur', 'ts': 0,
                  'state': 'F'}).status is domain.Status.FAILED

# The seam. A binary-floating-point division survives every small example and
# loses the cents on a large one, and Money refuses a float outright so the
# loss cannot be hidden behind a rounded repr.
exact = translate({'ref': 'big', 'amount_cents': 1234567890123457,
                   'ccy': 'jpy', 'ts': 1, 'state': 'P'})
assert exact.money.amount == Decimal('12345678901234.57'), (
    'the minor units were not converted exactly: got '
    f'{exact.money.amount}'
)

base = {'ref': 'r', 'amount_cents': 100, 'ccy': 'usd', 'ts': 0, 'state': 'P'}
for key in ('ref', 'amount_cents', 'ccy', 'ts', 'state'):
    missing = dict(base)
    del missing[key]
    try:
        translate(missing)
    except domain.TranslationError:
        pass
    except Exception as error:
        raise AssertionError(
            f'a missing {key!r} raised {type(error).__name__}, which no '
            'caller of this layer is written to catch')
    else:
        raise AssertionError(f'a record with no {key!r} was translated')

for field, value in (('state', 'X'), ('state', ''), ('state', None),
                     ('ccy', 'dollars'), ('ccy', 'us'), ('ccy', None),
                     ('amount_cents', -1), ('amount_cents', 1.5),
                     ('amount_cents', '100'), ('ts', 'yesterday')):
    bad = dict(base)
    bad[field] = value
    try:
        translate(bad)
    except domain.TranslationError:
        pass
    except Exception as error:
        raise AssertionError(
            f'{field}={value!r} raised {type(error).__name__} rather than '
            'TranslationError')
    else:
        raise AssertionError(f'{field}={value!r} was translated')
''',
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "The module pricers.py already exists and cannot be changed: "
            "LegacyPricer and ModernPricer both expose price(order) and "
            "record every order id they are asked to price in a calls list, "
            "and either may raise PricingError. Write a class "
            "StranglerPricer(legacy, modern, mode, on_event) migrating "
            "traffic from one to the other. mode is 'legacy', 'shadow' or "
            "'modern', and any other value is a ValueError at construction. "
            "price(order) in legacy mode calls only the legacy pricer and "
            "returns its result. In shadow mode it calls both exactly once "
            "and still returns the legacy result, because legacy is "
            "authoritative during a shadow: if the two disagree it calls "
            "on_event('mismatch', order_id, legacy_result, modern_result) "
            "once, and if the modern pricer raises it calls "
            "on_event('error', order_id, legacy_result, message) with the "
            "exception's string and still returns the legacy result. A "
            "PricingError from the legacy pricer propagates in both of those "
            "modes. In modern mode it calls only the modern pricer, but if "
            "that raises PricingError it falls back to the legacy pricer, "
            "calls on_event('fallback', order_id, None, message) and returns "
            "the legacy result, propagating the legacy error if that fails "
            "too."
        ),
        fixtures={"pricers.py": _PRICERS},
        validator=LOAD_CANDIDATE + require("StranglerPricer") + r'''
import pricers


def order(identifier, quantity=2):
    return {'id': identifier, 'quantity': quantity}


events = []


def record(*args):
    events.append(args)


legacy = pricers.LegacyPricer()
modern = pricers.ModernPricer()
router = StranglerPricer(legacy, modern, 'legacy', record)
assert router.price(order('a')) == 200
assert legacy.calls == ['a'] and modern.calls == [], \
    'legacy mode must not reach the modern pricer'
assert events == []

# Shadow: both run, legacy answers, and agreement is silent.
legacy = pricers.LegacyPricer()
modern = pricers.ModernPricer(disagree_on=['b'])
router = StranglerPricer(legacy, modern, 'shadow', record)
assert router.price(order('a')) == 200
assert legacy.calls == ['a'] and modern.calls == ['a'], \
    'a shadow must call both exactly once'
assert events == [], 'agreement must not be reported'

# ...and disagreement is reported without changing the answer.
assert router.price(order('b')) == 200, \
    'the shadow returned the modern result; legacy is authoritative'
assert events == [('mismatch', 'b', 200, 201)], events
assert legacy.calls == ['a', 'b'] and modern.calls == ['a', 'b']

# A modern failure during a shadow must not become the caller's problem.
events.clear()
legacy = pricers.LegacyPricer()
modern = pricers.ModernPricer(raise_on=['c'])
router = StranglerPricer(legacy, modern, 'shadow', record)
assert router.price(order('c')) == 200, \
    'the modern pricer failing changed what the caller received'
assert len(events) == 1 and events[0][0] == 'error'
assert events[0][1] == 'c' and events[0][2] == 200
assert 'not ready' in events[0][3]

# The legacy pricer is authoritative, so its failure is real.
events.clear()
legacy = pricers.LegacyPricer(raise_on=['d'])
modern = pricers.ModernPricer()
router = StranglerPricer(legacy, modern, 'shadow', record)
try:
    router.price(order('d'))
except pricers.PricingError:
    pass
else:
    raise AssertionError('a legacy failure was swallowed during a shadow')

# Modern mode: only the modern pricer, unless it cannot cope.
events.clear()
legacy = pricers.LegacyPricer()
modern = pricers.ModernPricer(disagree_on=['e'])
router = StranglerPricer(legacy, modern, 'modern', record)
assert router.price(order('e')) == 201, 'modern mode did not return modern'
assert legacy.calls == [], 'modern mode called the legacy pricer anyway'
assert events == []

events.clear()
legacy = pricers.LegacyPricer()
modern = pricers.ModernPricer(raise_on=['f'])
router = StranglerPricer(legacy, modern, 'modern', record)
assert router.price(order('f')) == 200, 'no fallback to the legacy pricer'
assert legacy.calls == ['f'] and modern.calls == ['f']
assert len(events) == 1 and events[0][0] == 'fallback'
assert events[0][1] == 'f' and events[0][2] is None
assert 'not ready' in events[0][3]

events.clear()
legacy = pricers.LegacyPricer(raise_on=['g'])
modern = pricers.ModernPricer(raise_on=['g'])
router = StranglerPricer(legacy, modern, 'modern', record)
try:
    router.price(order('g'))
except pricers.PricingError as error:
    assert 'legacy' in str(error), \
        'the fallback failing must surface the legacy error'
else:
    raise AssertionError('both pricers failed and nothing was raised')

for bad in ('canary', '', None, 'Shadow'):
    try:
        StranglerPricer(pricers.LegacyPricer(), pricers.ModernPricer(),
                        bad, record)
    except ValueError:
        pass
    else:
        raise AssertionError(f'mode {bad!r} was accepted')
''',
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "The module gateway.py already exists and cannot be changed: "
            "Gateway.charge(idempotency_key, amount_cents) captures the "
            "amount and returns a record with an id and replayed set to "
            "False; called again with a key it has already captured it "
            "returns that same record with replayed set to True and takes no "
            "further money; it raises TransientError while it is briefly "
            "unavailable and DeclinedError when the issuer refuses. Write a "
            "function charge_once(gateway, reference, amount_cents, "
            "attempts=3) that captures a payment at most once. It must derive "
            "a single idempotency key from the reference and present that "
            "same key on every attempt, retry only TransientError, make no "
            "more than attempts calls in total, and re-raise the last "
            "TransientError if they are all exhausted. DeclinedError must "
            "propagate immediately without a retry. A reference that is not a "
            "non-empty string, an amount that is not a positive integer, and "
            "an attempts below one are each a ValueError raised before the "
            "gateway is called at all. Two separate calls with the same "
            "reference must never capture the money twice."
        ),
        fixtures={"gateway.py": _GATEWAY},
        validator=LOAD_CANDIDATE + require("charge_once") + r'''
import gateway as gw

# A retry presents the same key, so the money moves exactly once.
service = gw.Gateway(transient_failures=2)
record = charge_once(service, 'order-1', 500)
assert record['amount_cents'] == 500
assert service.attempts == 3, f'{service.attempts} attempts, expected 3'
assert service.captured_total == 500, \
    f'captured {service.captured_total}, so a retry charged again'

# The seam: calling again for the same reference is a replay, not a purchase.
again = charge_once(service, 'order-1', 500)
assert service.captured_total == 500, (
    f'a repeat call captured a second time: total {service.captured_total}'
)
assert again['id'] == record['id'], 'the replay returned a different charge'
assert again['replayed'] is True, 'the gateway did not see a familiar key'

# A different reference is a different payment.
assert charge_once(service, 'order-2', 250)['replayed'] is False
assert service.captured_total == 750

# A decline is final and must not be retried.
declining = gw.Gateway(decline=True)
try:
    charge_once(declining, 'order-3', 100)
except gw.DeclinedError:
    pass
else:
    raise AssertionError('a decline was not propagated')
assert declining.attempts == 1, \
    f'a decline was retried {declining.attempts} times'
assert declining.captured_total == 0

# Exhausting the retries surfaces the transient failure rather than hiding it.
flaky = gw.Gateway(transient_failures=5)
try:
    charge_once(flaky, 'order-4', 100, attempts=3)
except gw.TransientError:
    pass
else:
    raise AssertionError('exhausted retries did not raise')
assert flaky.attempts == 3, f'made {flaky.attempts} attempts, expected 3'
assert flaky.captured_total == 0

single = gw.Gateway(transient_failures=1)
try:
    charge_once(single, 'order-5', 100, attempts=1)
except gw.TransientError:
    pass
else:
    raise AssertionError('attempts=1 retried anyway')
assert single.attempts == 1

# Bad input is rejected before any money can move.
guarded = gw.Gateway()
for reference, amount, attempts in (
        ('', 100, 3), (None, 100, 3), (7, 100, 3),
        ('r', 0, 3), ('r', -5, 3), ('r', 1.5, 3), ('r', '100', 3),
        ('r', True, 3), ('r', 100, 0), ('r', 100, -1)):
    try:
        charge_once(guarded, reference, amount, attempts)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f'charge_once({reference!r}, {amount!r}, {attempts!r}) was '
            'accepted')
assert guarded.attempts == 0, 'the gateway was called with invalid input'
assert guarded.captured_total == 0
''',
    ),
]
