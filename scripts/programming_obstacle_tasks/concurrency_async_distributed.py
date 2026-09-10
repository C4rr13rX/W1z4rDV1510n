"""Held-out tasks: concurrency, asynchronous work, and distributed coordination.

Concurrency tasks are the easiest place in this course to write a validator
that is itself flaky, which the acceptance contract refuses to admit. Two
rules keep these deterministic:

- where the contract is about *time*, the clock and the sleep function are
  injected, so the validator drives the schedule instead of racing it; and
- where the contract is about *mutual exclusion*, the assertion is an
  invariant that must hold under every interleaving -- an exact counter, or
  a witness that two writers were never inside the section together -- rather
  than an observation that happens to occur under one scheduling.

A test that passes only when threads interleave favourably would report a
capability failure on a loaded host, so no assertion here depends on wall
clock duration or on threads actually overlapping.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import (
    LOAD_CANDIDATE,
    SHAPE_GUARDS,
    require,
)

FAMILY = "concurrency_async_distributed"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python class TokenBucket(capacity, refill_per_second, "
            "clock) for rate limiting. `clock` is a zero-argument callable "
            "returning a monotonic float in seconds. The bucket starts full. "
            "The method allow(cost=1) returns True and deducts `cost` when at "
            "least `cost` tokens are available at the current time, otherwise "
            "returns False and deducts nothing. Tokens refill continuously at "
            "refill_per_second, computed from elapsed time rather than on a "
            "timer, and never accumulate beyond `capacity` however long the "
            "bucket idles. A request whose cost exceeds `capacity` can never "
            "succeed. Time never moves backwards."
        ),
        validator=LOAD_CANDIDATE + require("TokenBucket") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_TokenBucket = TokenBucket
def TokenBucket(*args, **kwargs):
    return having(_TokenBucket(*args, **kwargs), 'allow',
                  what='TokenBucket(...)')
now = [0.0]
clock = lambda: now[0]

bucket = TokenBucket(3, 1.0, clock)

# Starts full, and drains exactly.
assert bucket.allow() is True
assert bucket.allow() is True
assert bucket.allow() is True
assert bucket.allow() is False, "bucket allowed a fourth token when empty"

# A rejected request must not deduct, so it cannot deepen the deficit.
now[0] = 1.0
assert bucket.allow() is True, "one second should refill exactly one token"
assert bucket.allow() is False

# Refill is proportional to elapsed time, not a fixed step.
now[0] = 3.5
assert bucket.allow() is True
assert bucket.allow() is True
assert bucket.allow() is False, "2.5s at 1/s must not yield three tokens"

# Idling must not accumulate beyond capacity.
now[0] = 1000.0
assert bucket.allow() is True
assert bucket.allow() is True
assert bucket.allow() is True
assert bucket.allow() is False, "a long idle overfilled the bucket"

# Cost is honoured, and a partial balance rejects a larger cost.
now[0] = 1002.0
bucket2 = TokenBucket(10, 5.0, clock)
assert bucket2.allow(10) is True
assert bucket2.allow(1) is False
now[0] = 1003.0
assert bucket2.allow(6) is False, "5 tokens must not satisfy a cost of 6"
assert bucket2.allow(5) is True

# A cost above capacity is unsatisfiable no matter how long we wait.
now[0] = 99999.0
assert bucket2.allow(11) is False, "cost above capacity must never succeed"
assert bucket2.allow(10) is True

# Fractional rates work.
now[0] = 0.0
slow = TokenBucket(1, 0.5, clock)
assert slow.allow() is True
now[0] = 1.0
assert slow.allow() is False, "0.5/s needs two seconds for one token"
now[0] = 2.0
assert slow.allow() is True
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python class CircuitBreaker(failure_threshold, "
            "recovery_seconds, clock) plus an exception class "
            "CircuitOpenError. `clock` is a zero-argument callable returning "
            "monotonic seconds. The method call(operation) invokes the "
            "zero-argument `operation`. While closed, a returned value resets "
            "the consecutive-failure count and is returned; an exception "
            "increments that count and propagates, and reaching "
            "failure_threshold consecutive failures opens the circuit. While "
            "open, call raises CircuitOpenError without invoking the "
            "operation, until recovery_seconds have elapsed since it opened. "
            "The first call after that elapses is a trial: if it succeeds the "
            "circuit closes and counters reset, and if it fails the circuit "
            "opens again with the recovery window restarted."
        ),
        validator=LOAD_CANDIDATE + require("CircuitBreaker")
        + require("CircuitOpenError") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_CircuitBreaker = CircuitBreaker
def CircuitBreaker(*args, **kwargs):
    return having(_CircuitBreaker(*args, **kwargs), 'call',
                  what='CircuitBreaker(...)')
now = [0.0]
clock = lambda: now[0]
calls = []

def ok(tag="ok"):
    def run():
        calls.append(tag)
        return tag
    return run

def fail(tag="boom"):
    def run():
        calls.append(tag)
        raise RuntimeError(tag)
    return run

breaker = CircuitBreaker(3, 10.0, clock)

# Closed: values pass through, failures propagate as themselves.
assert breaker.call(ok("a")) == "a"
for _ in range(2):
    try:
        breaker.call(fail())
    except RuntimeError:
        pass
    else:
        raise AssertionError("the operation's exception was swallowed")

# A success RESETS the consecutive count, so two more failures must not open.
assert breaker.call(ok("b")) == "b"
for _ in range(2):
    try:
        breaker.call(fail())
    except RuntimeError:
        pass
assert breaker.call(ok("c")) == "c", "consecutive failures must reset on success"

# Three consecutive failures open the circuit.
for _ in range(3):
    try:
        breaker.call(fail())
    except RuntimeError:
        pass

before = len(calls)
try:
    breaker.call(ok("must-not-run"))
except CircuitOpenError:
    pass
else:
    raise AssertionError("an open circuit still ran the operation")
assert len(calls) == before, "an open circuit invoked the operation"

# Still open just before the window elapses.
now[0] = 9.9
try:
    breaker.call(ok("still-open"))
except CircuitOpenError:
    pass
else:
    raise AssertionError("the circuit reopened early")

# The first call after the window is a trial that actually runs.
now[0] = 10.0
before = len(calls)
try:
    breaker.call(fail("trial"))
except RuntimeError:
    pass
else:
    raise AssertionError("the trial call did not run")
assert len(calls) == before + 1

# A failed trial re-opens with the window RESTARTED, not resumed.
now[0] = 10.1
try:
    breaker.call(ok("nope"))
except CircuitOpenError:
    pass
else:
    raise AssertionError("a failed trial did not restart the recovery window")

# A successful trial closes the circuit and resets the counters.
now[0] = 20.1
assert breaker.call(ok("recovered")) == "recovered"
for _ in range(2):
    try:
        breaker.call(fail())
    except RuntimeError:
        pass
assert breaker.call(ok("d")) == "d", "counters did not reset on recovery"
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function retry_with_backoff(operation, "
            "attempts, base_delay, max_delay, sleep, should_retry) that calls "
            "the zero-argument `operation` until it returns, retrying only "
            "exceptions for which should_retry(exception) is true. Return the "
            "operation's value. Sleep between attempts by calling "
            "sleep(seconds) with a full exponential backoff of "
            "base_delay * 2 ** (retry_index) capped at max_delay, where the "
            "first retry uses retry_index 0. Never sleep before the first "
            "attempt and never sleep after the final one. Re-raise the last "
            "exception when attempts are exhausted, and re-raise immediately "
            "and without sleeping when should_retry is false."
        ),
        validator=LOAD_CANDIDATE + require("retry_with_backoff") + r'''
class Transient(Exception):
    pass

class Fatal(Exception):
    pass

retryable = lambda error: isinstance(error, Transient)

# A first-attempt success neither sleeps nor retries.
slept = []
calls = []
def once():
    calls.append(1)
    return "value"
assert retry_with_backoff(once, 5, 1.0, 60.0, slept.append, retryable) == "value"
assert calls == [1] and slept == [], f"slept on a clean call: {slept}"

# Backoff doubles from base and is capped at max_delay.
slept = []
attempts = []
def flaky():
    attempts.append(1)
    if len(attempts) < 6:
        raise Transient("later")
    return "ok"
assert retry_with_backoff(flaky, 8, 1.0, 8.0, slept.append, retryable) == "ok"
assert len(attempts) == 6
assert slept == [1.0, 2.0, 4.0, 8.0, 8.0], f"unexpected backoff: {slept}"

# Exhausting attempts re-raises the LAST exception and sleeps one time fewer.
slept = []
attempts = []
def always():
    attempts.append(1)
    raise Transient(f"attempt-{len(attempts)}")
try:
    retry_with_backoff(always, 4, 0.5, 100.0, slept.append, retryable)
except Transient as error:
    assert str(error) == "attempt-4", f"re-raised the wrong error: {error}"
else:
    raise AssertionError("exhausted retries did not re-raise")
assert len(attempts) == 4, f"expected 4 attempts, got {len(attempts)}"
assert slept == [0.5, 1.0, 2.0], f"slept after the final attempt: {slept}"

# A single attempt never sleeps.
slept = []
attempts = []
try:
    retry_with_backoff(always, 1, 1.0, 10.0, slept.append, retryable)
except Transient:
    pass
assert len(attempts) == 1 and slept == [], f"slept with attempts=1: {slept}"

# A non-retryable error is raised immediately, without sleeping.
slept = []
calls = []
def fatal():
    calls.append(1)
    raise Fatal("do not retry")
try:
    retry_with_backoff(fatal, 5, 1.0, 10.0, slept.append, retryable)
except Fatal:
    pass
else:
    raise AssertionError("a non-retryable error was retried or swallowed")
assert calls == [1], "a non-retryable error was retried"
assert slept == [], "slept before re-raising a non-retryable error"

# max_delay below base_delay clamps every wait.
slept = []
attempts = []
try:
    retry_with_backoff(always, 3, 10.0, 2.0, slept.append, retryable)
except Transient:
    pass
assert slept == [2.0, 2.0], f"max_delay did not clamp: {slept}"
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function compare_clocks(left, right) for "
            "vector clocks, where each argument maps a node id to an integer "
            "counter and a missing node means zero. Return the string "
            "'equal' when the two clocks are identical, 'before' when left "
            "happened before right, 'after' when right happened before left, "
            "and 'concurrent' when neither happened before the other. Left "
            "happens before right when every counter in left is less than or "
            "equal to the matching counter in right and at least one is "
            "strictly less. Also implement merge_clocks(left, right) "
            "returning the element-wise maximum over the union of both node "
            "sets, without mutating either argument."
        ),
        validator=LOAD_CANDIDATE + require("compare_clocks")
        + require("merge_clocks") + r'''
assert compare_clocks({}, {}) == "equal"
assert compare_clocks({"a": 1}, {"a": 1}) == "equal"

# A missing node reads as zero, so these are equal, not concurrent.
assert compare_clocks({"a": 1}, {"a": 1, "b": 0}) == "equal"
assert compare_clocks({"a": 0}, {}) == "equal"

assert compare_clocks({"a": 1}, {"a": 2}) == "before"
assert compare_clocks({"a": 2}, {"a": 1}) == "after"
assert compare_clocks({}, {"a": 1}) == "before"
assert compare_clocks({"a": 1}, {}) == "after"

# Dominating on every axis with at least one strict increase is 'before'.
assert compare_clocks({"a": 1, "b": 2}, {"a": 1, "b": 3}) == "before"
assert compare_clocks({"a": 1, "b": 2}, {"a": 2, "b": 3}) == "before"

# Trading places on different axes is concurrency, not ordering.
assert compare_clocks({"a": 2, "b": 1}, {"a": 1, "b": 2}) == "concurrent"
assert compare_clocks({"a": 1}, {"b": 1}) == "concurrent"
assert compare_clocks({"a": 2, "b": 2}, {"a": 3, "b": 1}) == "concurrent"

# Disjoint node sets where one side is strictly larger everywhere it matters.
assert compare_clocks({"a": 1}, {"a": 1, "b": 1}) == "before"
assert compare_clocks({"a": 1, "b": 1}, {"a": 1}) == "after"

# merge takes the element-wise maximum over the union.
assert merge_clocks({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
assert merge_clocks({"a": 3, "b": 1}, {"a": 1, "b": 5}) == {"a": 3, "b": 5}
assert merge_clocks({}, {}) == {}

# merge must not mutate either argument.
left = {"a": 1, "b": 9}
right = {"a": 7}
merged = merge_clocks(left, right)
assert merged == {"a": 7, "b": 9}, merged
assert left == {"a": 1, "b": 9}, "merge mutated its left argument"
assert right == {"a": 7}, "merge mutated its right argument"

# A merged clock is at or after both of its inputs.
assert compare_clocks(left, merged) == "before"
assert compare_clocks(right, merged) == "before"
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python class ExactlyOnceInbox handling an "
            "at-least-once delivery stream that may duplicate and reorder "
            "messages. Construct it with ExactlyOnceInbox(apply) where "
            "`apply` is a one-argument callable performing the side effect. "
            "The method deliver(sequence, payload) takes a monotonically "
            "assigned integer sequence number starting at 1. Apply each "
            "sequence number exactly once and strictly in ascending order: "
            "buffer a message that arrives before its predecessors and "
            "release the buffered run once the gap is filled. Ignore any "
            "sequence number already applied. Return the list of sequence "
            "numbers applied during that call, in the order applied. Expose a "
            "read-only property `pending` giving the count of buffered "
            "messages awaiting a gap."
        ),
        validator=LOAD_CANDIDATE + require("ExactlyOnceInbox") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_ExactlyOnceInbox = ExactlyOnceInbox
def ExactlyOnceInbox(*args, **kwargs):
    return having(_ExactlyOnceInbox(*args, **kwargs), 'deliver', 'pending',
                  what='ExactlyOnceInbox(...)')
applied = []
inbox = ExactlyOnceInbox(applied.append)

# In-order delivery applies immediately.
assert inbox.deliver(1, "a") == [1]
assert inbox.deliver(2, "b") == [2]
assert applied == ["a", "b"]
assert inbox.pending == 0

# A duplicate is ignored and applies nothing.
assert inbox.deliver(1, "a") == []
assert inbox.deliver(2, "b") == []
assert applied == ["a", "b"], "a duplicate was applied twice"

# Out-of-order messages buffer until the gap is filled.
assert inbox.deliver(5, "e") == []
assert inbox.deliver(4, "d") == []
assert inbox.pending == 2, f"expected two buffered, got {inbox.pending}"
assert applied == ["a", "b"], "a gapped message was applied early"

# Filling the gap releases the whole contiguous run in order.
assert inbox.deliver(3, "c") == [3, 4, 5], "the buffered run was not released"
assert applied == ["a", "b", "c", "d", "e"]
assert inbox.pending == 0

# A duplicate of a buffered message must not double-apply on release.
assert inbox.deliver(8, "h") == []
assert inbox.deliver(8, "h") == []
assert inbox.pending == 1, "a duplicate was buffered twice"
assert inbox.deliver(7, "g") == []
assert inbox.deliver(6, "f") == [6, 7, 8]
assert applied == ["a", "b", "c", "d", "e", "f", "g", "h"]

# A late duplicate of an already-applied message stays ignored.
assert inbox.deliver(4, "d") == []
assert applied == ["a", "b", "c", "d", "e", "f", "g", "h"]

# A fresh inbox must not apply a message that skips sequence 1.
second = []
other = ExactlyOnceInbox(second.append)
assert other.deliver(2, "x") == []
assert second == [], "applied before sequence 1 arrived"
assert other.deliver(1, "w") == [1, 2]
assert second == ["w", "x"]

# Reversed arrival of a long run still applies in ascending order.
third = []
reverse = ExactlyOnceInbox(third.append)
for sequence in range(6, 0, -1):
    reverse.deliver(sequence, f"m{sequence}")
assert third == [f"m{n}" for n in range(1, 7)], third
assert reverse.pending == 0
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function schedule_batches(tasks, "
            "dependencies) that plans maximum-parallelism execution. `tasks` "
            "is a list of task names and `dependencies` is a list of "
            "(before, after) pairs meaning `before` must complete first. "
            "Return a list of batches, each a sorted list of task names, "
            "where every task appears exactly once and appears in the "
            "earliest batch whose predecessors have all completed in strictly "
            "earlier batches. Raise ValueError when the dependencies contain "
            "a cycle, and ValueError when a dependency names a task that is "
            "not in `tasks`."
        ),
        validator=LOAD_CANDIDATE + require("schedule_batches") + r'''
assert schedule_batches([], []) == []
assert schedule_batches(["a"], []) == [["a"]]

# Independent tasks share one batch, sorted.
assert schedule_batches(["c", "a", "b"], []) == [["a", "b", "c"]]

# A chain serializes completely.
assert schedule_batches(
    ["a", "b", "c"], [("a", "b"), ("b", "c")]
) == [["a"], ["b"], ["c"]]

# A diamond runs its middle in parallel.
assert schedule_batches(
    ["a", "b", "c", "d"],
    [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
) == [["a"], ["b", "c"], ["d"]]

# A task waits for its LATEST predecessor, not its first.
# c depends only on a, but d depends on c and on b which is two deep.
plan = schedule_batches(
    ["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
)
assert plan[-1] == ["d"]

# A long chain beside a short one: the join waits for the long one.
plan = schedule_batches(
    ["s", "x1", "x2", "x3", "y", "j"],
    [("s", "x1"), ("x1", "x2"), ("x2", "x3"), ("s", "y"),
     ("x3", "j"), ("y", "j")],
)
assert plan == [["s"], ["x1", "y"], ["x2"], ["x3"], ["j"]], plan

# Every task appears exactly once across the plan.
names = [name for batch in plan for name in batch]
assert sorted(names) == sorted(["s", "x1", "x2", "x3", "y", "j"])
assert len(names) == len(set(names)), "a task was scheduled twice"

# Disconnected components pack into the same batches.
assert schedule_batches(
    ["a", "b", "p", "q"], [("a", "b"), ("p", "q")]
) == [["a", "p"], ["b", "q"]]

# Cycles are an error, including self-dependency.
for cyclic in ([("a", "b"), ("b", "a")], [("a", "a")]):
    try:
        schedule_batches(["a", "b"], cyclic)
    except ValueError:
        pass
    else:
        raise AssertionError(f"cycle {cyclic} was scheduled")

# An unknown task in a dependency is an error, not a silent extra node.
try:
    schedule_batches(["a"], [("a", "ghost")])
except ValueError:
    pass
else:
    raise AssertionError("an unknown dependency was accepted")
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python class Once providing exactly-once "
            "initialization that is safe under threads. The method "
            "do(factory) runs the zero-argument `factory` the first time it "
            "is called and returns its value; every later call returns that "
            "same value without running the factory again, including when "
            "many threads call do concurrently. If the factory raises, "
            "propagate the exception and leave the Once uninitialized so a "
            "later call may try again. Expose a read-only property `done` "
            "that is True only once a value has been stored. Do not hold a "
            "lock while returning an already-computed value."
        ),
        validator=LOAD_CANDIDATE + require("Once") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_Once = Once
def Once(*args, **kwargs):
    return having(_Once(*args, **kwargs), 'do', 'done',
                  what='Once(...)')
import threading

# Single-threaded contract first.
once = Once()
assert once.done is False
runs = []
assert once.do(lambda: runs.append(1) or "value") == "value"
assert once.done is True
assert once.do(lambda: runs.append(2) or "other") == "value"
assert runs == [1], "the factory ran more than once"

# A raising factory leaves it uninitialized and retryable.
guard = Once()
def boom():
    raise RuntimeError("factory failed")
for _ in range(2):
    try:
        guard.do(boom)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the factory's exception was swallowed")
assert guard.done is False, "a failed factory marked the Once as done"
assert guard.do(lambda: "recovered") == "recovered"
assert guard.done is True

# Under contention the factory must still run exactly once, and every
# thread must observe the SAME value. All threads are released together
# to make the race as likely as this harness can arrange; the assertion
# is an invariant, so it holds however they actually interleave.
factory_runs = []
shared = Once()
observed = []
observed_lock = threading.Lock()
start = threading.Barrier(8)

def worker():
    start.wait()
    value = shared.do(lambda: factory_runs.append(1) or object())
    with observed_lock:
        observed.append(value)

threads = [threading.Thread(target=worker) for _ in range(8)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=10)
    assert not thread.is_alive(), "a thread deadlocked in do()"

assert len(factory_runs) == 1, (
    f"the factory ran {len(factory_runs)} times under contention"
)
assert len(observed) == 8
assert all(value is observed[0] for value in observed), \
    "threads observed different values from one Once"
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python class ReadWriteLock allowing many concurrent "
            "readers or one exclusive writer. Provide acquire_read(), "
            "release_read(), acquire_write() and release_write(). A writer "
            "must never run while any reader holds the lock, and two writers "
            "must never run at once. Readers must not starve writers: once a "
            "writer is waiting, a reader arriving afterwards waits for that "
            "writer rather than joining the readers already inside. Assume "
            "each thread releases what it acquired and does not upgrade a "
            "read hold into a write hold."
        ),
        validator=LOAD_CANDIDATE + require("ReadWriteLock") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_ReadWriteLock = ReadWriteLock
def ReadWriteLock(*args, **kwargs):
    return having(_ReadWriteLock(*args, **kwargs), 'acquire_read', 'acquire_write', 'release_read', 'release_write',
                  what='ReadWriteLock(...)')
import threading

lock = ReadWriteLock()

state_lock = threading.Lock()
readers_inside = 0
writers_inside = 0
violations = []
max_concurrent_readers = 0

def note_violation(message):
    with state_lock:
        violations.append(message)

def reader():
    global readers_inside, max_concurrent_readers
    for _ in range(25):
        lock.acquire_read()
        try:
            with state_lock:
                readers_inside += 1
                if writers_inside:
                    violations.append("reader ran while a writer held it")
                max_concurrent_readers = max(
                    max_concurrent_readers, readers_inside
                )
            # Touch shared state to widen the window for any violation.
            sum(range(200))
            with state_lock:
                readers_inside -= 1
        finally:
            lock.release_read()

def writer():
    global writers_inside
    for _ in range(25):
        lock.acquire_write()
        try:
            with state_lock:
                writers_inside += 1
                if writers_inside > 1:
                    violations.append("two writers held the lock")
                if readers_inside:
                    violations.append("a writer ran while readers held it")
            sum(range(200))
            with state_lock:
                writers_inside -= 1
        finally:
            lock.release_write()

threads = [threading.Thread(target=reader) for _ in range(6)]
threads += [threading.Thread(target=writer) for _ in range(3)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=30)
    assert not thread.is_alive(), "a thread deadlocked in the lock"

assert not violations, f"mutual exclusion violated: {sorted(set(violations))}"
assert readers_inside == 0 and writers_inside == 0

# The lock must be reusable afterwards, and readers really are shared:
# two read holds at once must not deadlock.
lock.acquire_read()
lock.acquire_read()
lock.release_read()
lock.release_read()
lock.acquire_write()
lock.release_write()

# A reader arriving after a waiting writer must not overtake it. The writer
# holds the lock; a second reader queues behind the waiting writer.
held = threading.Event()
writer_ready = threading.Event()
order = []
order_lock = threading.Lock()

lock.acquire_read()

def waiting_writer():
    writer_ready.set()
    lock.acquire_write()
    with order_lock:
        order.append("writer")
    lock.release_write()

def late_reader():
    lock.acquire_read()
    with order_lock:
        order.append("reader")
    lock.release_read()

writer_thread = threading.Thread(target=waiting_writer)
writer_thread.start()
writer_ready.wait(timeout=5)
# Give the writer a chance to actually block on acquire_write.
for _ in range(1000):
    sum(range(50))

reader_thread = threading.Thread(target=late_reader)
reader_thread.start()
for _ in range(1000):
    sum(range(50))

lock.release_read()
writer_thread.join(timeout=15)
reader_thread.join(timeout=15)
assert not writer_thread.is_alive() and not reader_thread.is_alive(), \
    "a thread deadlocked waiting for the writer"
assert order == ["writer", "reader"], \
    f"a late reader overtook a waiting writer: {order}"
''',
        timeout_seconds=90.0,
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python class HashRing(nodes=(), replicas=100) that "
            "distributes keys over nodes by consistent hashing. add(node) "
            "and remove(node) change the membership and get(key) returns the "
            "node holding a key. Each node takes `replicas` positions on the "
            "ring so that load stays even. The mapping must depend only on "
            "the current member set, never on the order members were added, "
            "and adding or removing one node must leave every other key "
            "where it was. get raises KeyError on an empty ring and remove "
            "raises KeyError for a node that is not a member."
        ),
        timeout_seconds=90.0,
        validator=LOAD_CANDIDATE + require("HashRing") + SHAPE_GUARDS + '''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_HashRing = HashRing
def HashRing(*args, **kwargs):
    return having(_HashRing(*args, **kwargs), 'add', 'get', 'remove',
                  what='HashRing(...)')
nodes = [f"node-{index}" for index in range(8)]
keys = [f"key-{index}" for index in range(8000)]

ring = HashRing(nodes)
before = {key: ring.get(key) for key in keys}
assert set(before.values()) <= set(nodes), "get returned an unknown node"
assert len(set(before.values())) == 8, "some node was given no keys at all"

# The ring is a function of the member set, not of the build order.
rebuilt = HashRing()
for node in reversed(nodes):
    rebuilt.add(node)
assert all(rebuilt.get(key) == before[key] for key in keys), \\
    "the mapping depends on the order nodes were added"

# Even load is what the replica count buys; one position per node would not
# reach it. The bounds are wide enough that hash noise cannot cross them.
counts = {}
for owner in before.values():
    counts[owner] = counts.get(owner, 0) + 1
mean = len(keys) / 8.0
assert min(counts.values()) > 0.4 * mean, \\
    f"load is badly skewed: {sorted(counts.values())}"
assert max(counts.values()) < 2.0 * mean, \\
    f"load is badly skewed: {sorted(counts.values())}"

# Removing a node may move that node's keys and nothing else. Hashing modulo
# the member count passes every assertion above and fails this one.
victim = nodes[3]
ring.remove(victim)
after = {key: ring.get(key) for key in keys}
assert victim not in set(after.values()), "a removed node still owns keys"
moved = [key for key in keys if after[key] != before[key]]
assert all(before[key] == victim for key in moved), \\
    "removing one node moved keys that did not belong to it"
assert len(moved) == sum(1 for key in keys if before[key] == victim)

# Adding a node may only pull keys onto the newcomer.
grown = HashRing(nodes)
grown.add("node-new")
after_add = {key: grown.get(key) for key in keys}
gained = [key for key in keys if after_add[key] != before[key]]
assert gained, "adding a node moved no keys at all"
assert all(after_add[key] == "node-new" for key in gained), \\
    "adding one node shuffled keys between the existing nodes"

assert all(grown.get(key) == after_add[key] for key in keys), \\
    "get is not deterministic across calls"

solo = HashRing(["only"])
assert all(solo.get(key) == "only" for key in keys[:200])

try:
    HashRing().get("anything")
except KeyError:
    pass
else:
    raise AssertionError("an empty ring returned a node")

try:
    HashRing(["a"]).remove("b")
except KeyError:
    pass
else:
    raise AssertionError("removing a node that is not a member was accepted")
''',
    ),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python class PNCounter(node_id) -- a counter that "
            "several replicas increment and decrement independently and then "
            "reconcile without coordination. increment(amount=1) and "
            "decrement(amount=1) record activity against this replica's own "
            "node_id and raise ValueError for a negative amount. value() "
            "returns the current total. merge(other) returns a NEW PNCounter "
            "holding the join of the two replicas' knowledge and leaves both "
            "operands unchanged. Merging must be idempotent, commutative and "
            "associative, so replicas that exchange states in any order, any "
            "number of times, all settle on the same value."
        ),
        validator=LOAD_CANDIDATE + require("PNCounter") + SHAPE_GUARDS + '''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_PNCounter = PNCounter
def PNCounter(*args, **kwargs):
    return having(_PNCounter(*args, **kwargs), 'increment', 'decrement', 'merge', 'value',
                  what='PNCounter(...)')
a = PNCounter("a")
b = PNCounter("b")
c = PNCounter("c")
a.increment(5)
a.decrement(2)
b.increment(7)
c.decrement(4)
assert a.value() == 3, f"a.value() is {a.value()}, expected 3"
assert b.value() == 7, f"b.value() is {b.value()}, expected 7"
assert c.value() == -4, f"c.value() is {c.value()}, expected -4"

ab = a.merge(b)
assert a.value() == 3 and b.value() == 7, "merge mutated one of its operands"
assert ab.value() == 10, f"a.merge(b) is {ab.value()}, expected 10"

# Re-delivering a state a replica has already seen must change nothing. This
# is what separates joining the two states from adding them: addition is
# right the first time and wrong on every repeat.
assert ab.merge(b).value() == 10, "merging an already-seen state changed the value"
assert ab.merge(b).merge(b).value() == 10, "a third delivery changed the value"
assert ab.merge(ab).value() == 10, "merging a replica with itself changed the value"

assert a.merge(b).value() == b.merge(a).value(), "merge is not commutative"
assert (a.merge(b)).merge(c).value() == a.merge(b.merge(c)).value(), \\
    "merge is not associative"

fresh = PNCounter("fresh")
assert a.merge(fresh).value() == 3, "merging an empty replica changed the value"
assert fresh.merge(a).value() == 3, "merging into an empty replica lost updates"

# A divergent history gossiped in two different orders, with redundant
# deliveries in both. The expected total is accumulated alongside the plan
# rather than written down.
replicas = [PNCounter("r0"), PNCounter("r1"), PNCounter("r2")]
plan = [(0, 1, 3), (1, -1, 1), (2, 1, 10), (0, -1, 4),
        (1, 1, 6), (2, -1, 2), (0, 1, 1), (1, -1, 5)]
total = 0
for index, sign, amount in plan:
    if sign > 0:
        replicas[index].increment(amount)
    else:
        replicas[index].decrement(amount)
    total += sign * amount

left = replicas[0].merge(replicas[1]).merge(replicas[2]).merge(replicas[1])
right = replicas[2].merge(replicas[0]).merge(replicas[1]).merge(replicas[2])
assert left.value() == total, f"one gossip order settled on {left.value()}, not {total}"
assert right.value() == total, f"the other settled on {right.value()}, not {total}"
assert left.merge(right).value() == total, "re-merging two converged replicas drifted"

for bad in (-1, -100):
    try:
        PNCounter("x").increment(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"increment({bad}) was accepted")
    try:
        PNCounter("x").decrement(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"decrement({bad}) was accepted")
''',
    ),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement LeaseManager(lease_seconds, clock) and FencedStore(), "
            "plus exception classes LeaseHeldError and StaleTokenError. "
            "`clock` is a zero-argument callable returning a monotonic float "
            "in seconds. LeaseManager.acquire(holder) grants the lease when "
            "none is active or the active one has expired, and returns a "
            "fencing token: an integer strictly greater than every token the "
            "manager has ever issued. It raises LeaseHeldError while another "
            "holder's lease is unexpired. release(holder) ends the lease and "
            "raises LeaseHeldError if that holder does not hold it. "
            "FencedStore.write(token, key, value) stores the value and "
            "remembers the highest token it has accepted, raising "
            "StaleTokenError for any token below that. read(key) returns the "
            "stored value and raises KeyError when there is none."
        ),
        validator=LOAD_CANDIDATE + require("LeaseManager") + require("FencedStore")
        + require("LeaseHeldError") + require("StaleTokenError") + SHAPE_GUARDS + '''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_LeaseManager = LeaseManager
def LeaseManager(*args, **kwargs):
    return having(_LeaseManager(*args, **kwargs), 'acquire', 'release',
                  what='LeaseManager(...)')
_FencedStore = FencedStore
def FencedStore(*args, **kwargs):
    return having(_FencedStore(*args, **kwargs), 'read', 'write',
                  what='FencedStore(...)')
now = [1000.0]
manager = LeaseManager(30.0, lambda: now[0])
store = FencedStore()

first = manager.acquire("A")
assert isinstance(first, int) and not isinstance(first, bool), \\
    f"the fencing token {first!r} is not an integer"
store.write(first, "config", "from A")
assert store.read("config") == "from A"

now[0] += 10.0
try:
    manager.acquire("B")
except LeaseHeldError:
    pass
else:
    raise AssertionError("two holders were granted the lease at once")

# A stalls -- garbage collection, a network partition -- long enough for its
# lease to lapse, and B takes over.
now[0] += 25.0
second = manager.acquire("B")
assert second > first, "the fencing token did not increase across holders"
store.write(second, "config", "from B")

# A wakes up still believing it holds the lease. Expiry alone cannot stop
# this write; only the token can, and that is the whole point of the task.
try:
    store.write(first, "config", "from A, too late")
except StaleTokenError:
    pass
else:
    raise AssertionError("a write from an expired lease holder was accepted")
assert store.read("config") == "from B", "the stale write survived in the store"

# The current holder keeps writing with the token it already has.
store.write(second, "other", "also from B")
assert store.read("other") == "also from B"

manager.release("B")
third = manager.acquire("A")
assert third > second, "a token was reissued after a clean release"

try:
    manager.release("B")
except LeaseHeldError:
    pass
else:
    raise AssertionError("a holder that does not hold the lease released it")

issued = [first, second, third]
for _ in range(5):
    manager.release("A")
    now[0] += 1.0
    issued.append(manager.acquire("A"))
assert len(set(issued)) == len(issued), f"a token was reused: {issued}"
assert issued == sorted(issued), f"tokens are not strictly increasing: {issued}"

try:
    store.read("missing")
except KeyError:
    pass
else:
    raise AssertionError("reading an absent key did not raise KeyError")
''',
    ),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement quorum helpers for a replicated store, plus exception "
            "classes NotEnoughReplicas and ConflictingVersions. "
            "is_strongly_consistent(n, r, w) reports whether a read quorum of "
            "r and a write quorum of w over n replicas guarantee that a read "
            "sees the latest acknowledged write. tolerated_failures(n, r, w) "
            "returns how many replicas may be unreachable while both quorums "
            "are still satisfiable. Both raise ValueError unless r and w each "
            "lie between 1 and n. resolve(responses, r) takes a list of "
            "(node, version, value) triples: it raises NotEnoughReplicas when "
            "fewer than r responded, raises ConflictingVersions when two "
            "different values share the highest version, and otherwise "
            "returns (value, stale_nodes) where value is the one at the "
            "highest version and stale_nodes is the sorted list of nodes that "
            "answered with an older version."
        ),
        validator=LOAD_CANDIDATE + require("is_strongly_consistent")
        + require("tolerated_failures") + require("resolve")
        + require("NotEnoughReplicas") + require("ConflictingVersions") + '''
# The rule is a strict inequality over integers, so the cases that sit
# exactly on r + w == n are the ones worth asserting, from both sides. The
# asymmetric pairs also separate the overlap rule from the write-quorum rule
# w * 2 > n, which agrees everywhere else.
for n, r, w, expected in (
        (3, 2, 2, True), (3, 2, 1, False), (3, 1, 3, True), (3, 3, 1, True),
        (5, 3, 3, True), (5, 2, 3, False), (5, 1, 1, False), (5, 4, 2, True),
        (5, 5, 1, True), (1, 1, 1, True), (4, 2, 2, False), (4, 3, 2, True)):
    got = is_strongly_consistent(n, r, w)
    assert got is expected, (
        f"is_strongly_consistent(n={n}, r={r}, w={w}) returned {got!r}, "
        f"expected {expected!r}")

for n, r, w, expected in ((3, 2, 2, 1), (5, 2, 2, 3), (5, 1, 5, 0),
                          (3, 1, 1, 2), (5, 4, 1, 1), (1, 1, 1, 0)):
    got = tolerated_failures(n, r, w)
    assert got == expected, (
        f"tolerated_failures(n={n}, r={r}, w={w}) returned {got}, "
        f"expected {expected}")

for n, r, w in ((3, 0, 2), (3, 4, 2), (3, 2, 0), (3, 2, 4), (3, -1, 1)):
    for function in (is_strongly_consistent, tolerated_failures):
        try:
            function(n, r, w)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{function.__name__}({n}, {r}, {w}) was accepted")

value, stale = resolve(
    [("n3", 3, "old"), ("n1", 5, "current"), ("n2", 5, "current")], 2)
assert value == "current", f"resolve chose {value!r}"
assert stale == ["n3"], f"stale nodes {stale}"

value, stale = resolve([("n2", 9, "same"), ("n1", 9, "same")], 2)
assert value == "same" and stale == [], f"unanimous responses reported {stale}"

value, stale = resolve(
    [("nb", 1, "old"), ("nc", 1, "old"), ("na", 4, "new")], 3)
assert value == "new" and stale == ["nb", "nc"], f"stale nodes {stale}"

try:
    resolve([("n1", 5, "only")], 2)
except NotEnoughReplicas:
    pass
else:
    raise AssertionError("resolve answered from fewer than r replicas")

try:
    resolve([], 1)
except NotEnoughReplicas:
    pass
else:
    raise AssertionError("resolve answered from no replicas at all")

try:
    resolve([("n1", 5, "left"), ("n2", 5, "right")], 2)
except ConflictingVersions:
    pass
else:
    raise AssertionError("two values at the same highest version were resolved")

# A conflict below the highest version is not a conflict; it is stale data.
value, stale = resolve(
    [("n1", 5, "winner"), ("n2", 4, "left"), ("n3", 4, "right")], 3)
assert value == "winner" and stale == ["n2", "n3"], f"stale nodes {stale}"
''',
    ),
    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement a Python class TumblingWindows(window_seconds, "
            "allowed_lateness) that groups out-of-order events by event time. "
            "add(event_time, value) files the event into the window starting "
            "at the largest multiple of window_seconds not above event_time; "
            "if that window has already been emitted the event is dropped "
            "instead and counted. advance_watermark(watermark) emits and "
            "forgets every window that the watermark has passed, meaning "
            "window_start + window_seconds + allowed_lateness is at most the "
            "watermark, returning a list of (window_start, values) in "
            "ascending window_start order with values in arrival order. A "
            "watermark below the previous one raises ValueError, as does a "
            "window_seconds that is not positive or a negative "
            "allowed_lateness. dropped_count() returns how many events "
            "arrived too late to be filed."
        ),
        validator=LOAD_CANDIDATE + require("TumblingWindows") + SHAPE_GUARDS + '''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_TumblingWindows = TumblingWindows
def TumblingWindows(*args, **kwargs):
    return having(_TumblingWindows(*args, **kwargs), 'add', 'advance_watermark', 'dropped_count',
                  what='TumblingWindows(...)')
windows = TumblingWindows(10, 5)
windows.add(3, "a")
windows.add(7, "b")
windows.add(12, "c")
assert windows.dropped_count() == 0

# The first window ends at 10 but stays open until 10 + 5 lateness. An
# implementation that forgets allowed_lateness closes it here.
assert windows.advance_watermark(14) == [], \\
    "a window closed before its allowed lateness elapsed"

windows.add(5, "late but welcome")
emitted = windows.advance_watermark(15)
assert emitted == [(0, ["a", "b", "late but welcome"])], f"emitted {emitted}"

windows.add(6, "far too late")
assert windows.dropped_count() == 1, "an event for a closed window was filed"
assert windows.advance_watermark(24) == [], "the second window closed early"

emitted = windows.advance_watermark(25)
assert emitted == [(10, ["c"])], f"emitted {emitted}"

try:
    windows.advance_watermark(20)
except ValueError:
    pass
else:
    raise AssertionError("the watermark was allowed to move backwards")

# Several windows becoming due at once come out in event-time order.
many = TumblingWindows(10, 0)
for event_time, value in ((25, "x"), (4, "y"), (31, "z"), (14, "w"), (6, "v")):
    many.add(event_time, value)
emitted = many.advance_watermark(40)
assert emitted == [(0, ["y", "v"]), (10, ["w"]), (20, ["x"]), (30, ["z"])], \\
    f"emitted {emitted}"
assert many.advance_watermark(41) == [], "a window was emitted twice"

# Zero lateness closes a window exactly at its end, not before it.
edge = TumblingWindows(10, 0)
edge.add(1, "p")
assert edge.advance_watermark(9) == []
assert edge.advance_watermark(10) == [(0, ["p"])]

for bad_window, bad_lateness in ((0, 0), (-10, 0), (10, -1)):
    try:
        TumblingWindows(bad_window, bad_lateness)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"TumblingWindows({bad_window}, {bad_lateness}) was accepted")
''',
    ),
    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement a Python class CyclicBarrier(parties) at which "
            "threads wait for one another. wait() blocks until `parties` "
            "threads have called it, then releases all of them; it returns "
            "the caller's arrival index, a distinct integer in [0, parties) "
            "for each thread in that round. The barrier is reusable: once a "
            "round is released the next wait() starts a fresh round with its "
            "own indices. Raise ValueError unless parties is a positive "
            "integer."
        ),
        timeout_seconds=90.0,
        validator=LOAD_CANDIDATE + require("CyclicBarrier") + '''
import threading

PARTIES = 6
ROUNDS = 5

barrier = CyclicBarrier(PARTIES)
arrived = [0] * ROUNDS
indices = [[] for _ in range(ROUNDS)]
guard = threading.Lock()
faults = []


def worker():
    try:
        for round_number in range(ROUNDS):
            with guard:
                arrived[round_number] += 1
            index = barrier.wait()
            # Every party increments before it calls wait, and wait cannot
            # return until all of them have called it -- so this count must
            # already be complete. The assertion holds under every
            # interleaving rather than under a lucky one.
            with guard:
                seen = arrived[round_number]
                indices[round_number].append(index)
            if seen != PARTIES:
                faults.append(
                    f"round {round_number} released a thread when only "
                    f"{seen} of {PARTIES} had arrived")
    except BaseException as error:
        faults.append(repr(error))


threads = [threading.Thread(target=worker) for _ in range(PARTIES)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=20)

assert not any(thread.is_alive() for thread in threads), \\
    "the barrier deadlocked and never released its parties"
assert not faults, faults[0]

for round_number in range(ROUNDS):
    got = sorted(indices[round_number])
    assert got == list(range(PARTIES)), \\
        f"round {round_number} handed out indices {got}"

alone = CyclicBarrier(1)
assert alone.wait() == 0
assert alone.wait() == 0, "a barrier of one is not reusable"

for bad in (0, -3):
    try:
        CyclicBarrier(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"CyclicBarrier({bad}) was accepted")
''',
    ),

    # Ids from 0301, matching the block this session used in
    # databases_migrations_transactions, so a concurrent author working from
    # 0001.. or 0101.. cannot silently shadow them.
    task(
        f"{FAMILY}-0301", FAMILY,
        prompt=(
            "Implement a Python class SlidingWindowLimiter(limit, "
            "window_seconds, clock) enforcing at most `limit` admissions in "
            "any window of `window_seconds`. `clock` is a zero-argument "
            "callable returning a monotonic float in seconds; call it rather "
            "than reading time yourself. allow() returns True and records an "
            "admission when the number already recorded within the last "
            "window_seconds is below limit, and returns False without "
            "recording otherwise. An admission recorded at time t no longer "
            "counts once the clock reaches t + window_seconds exactly. The "
            "window is continuous, not a calendar bucket: the limit must "
            "hold across every instant, so admissions must not become "
            "available again merely because the clock crossed a multiple of "
            "window_seconds. retry_after() returns 0.0 when allow() would "
            "currently succeed, and otherwise the seconds until the oldest "
            "counted admission expires. Raise ValueError when limit is not a "
            "positive integer or window_seconds is not positive."
        ),
        validator=LOAD_CANDIDATE + require("SlidingWindowLimiter") + SHAPE_GUARDS + r'''
# Guard every call, not just the first: `require` proves a name
# exists, never that it is the right KIND of thing, and an
# AttributeError on the result is raised in validator frames alone.
_SlidingWindowLimiter = SlidingWindowLimiter
def SlidingWindowLimiter(*args, **kwargs):
    return having(_SlidingWindowLimiter(*args, **kwargs), 'allow', 'retry_after',
                  what='SlidingWindowLimiter(...)')
now = [0.0]


def clock():
    return now[0]


limiter = SlidingWindowLimiter(3, 10.0, clock)

# The budget is spent, then refused.
assert [limiter.allow() for _ in range(4)] == [True, True, True, False]
assert limiter.retry_after() == 10.0

# Still refused most of the way through the window.
now[0] = 9.5
assert limiter.allow() is False
assert abs(limiter.retry_after() - 0.5) < 1e-9

# At exactly t + window the oldest admission stops counting.
now[0] = 10.0
assert limiter.retry_after() == 0.0
assert limiter.allow() is True

# THE BOUNDARY BURST. Three admissions at t=8, then the clock crosses 10 --
# a multiple of the window. A limiter that buckets by calendar interval
# starts a fresh count there and admits three more, putting six admissions
# inside a 2.5 second span while every other case above still agrees.
now[0] = 0.0
edge = SlidingWindowLimiter(3, 10.0, clock)
now[0] = 8.0
assert [edge.allow() for _ in range(3)] == [True, True, True]
now[0] = 10.5
assert edge.allow() is False, (
    "crossing a multiple of the window is not a reason to refill the budget"
)
assert abs(edge.retry_after() - 7.5) < 1e-9, edge.retry_after()

# And once the t=8 admissions really do age out, it opens again.
now[0] = 18.0
assert edge.allow() is True

# A limit of one is the degenerate case that must still work.
now[0] = 0.0
single = SlidingWindowLimiter(1, 5.0, clock)
assert single.allow() is True
assert single.allow() is False
now[0] = 5.0
assert single.allow() is True

# Refused calls record nothing, so the window does not creep forward.
now[0] = 0.0
creep = SlidingWindowLimiter(1, 4.0, clock)
assert creep.allow() is True
for tick in (1.0, 2.0, 3.0):
    now[0] = tick
    assert creep.allow() is False
now[0] = 4.0
assert creep.allow() is True, (
    "a refused call must not extend the window by recording itself"
)

for bad_limit in (0, -1, 1.5, True, "2"):
    try:
        SlidingWindowLimiter(bad_limit, 10.0, clock)
    except ValueError:
        pass
    else:
        raise AssertionError("limit %r must raise ValueError" % (bad_limit,))

for bad_window in (0, -3):
    try:
        SlidingWindowLimiter(3, bad_window, clock)
    except ValueError:
        pass
    else:
        raise AssertionError("window %r must raise ValueError" % (bad_window,))
'''),

    task(
        f"{FAMILY}-0302", FAMILY,
        prompt=(
            "Implement a Python function acquire_all(locks) that takes a "
            "list of lock objects and acquires every one of them. Each lock "
            "has a `name` attribute of type str and acquire() and release() "
            "methods taking no arguments. Callers reach these locks by "
            "different routes and pass them in whatever order they happen to "
            "hold them, so acquiring them in the order given lets two "
            "callers take them in opposite orders and wait on each other "
            "forever. Impose a total order instead: acquire in ascending "
            "`name` order, so that the sequence depends only on which locks "
            "are involved and not on the order the caller listed them. "
            "Return the list of names in the order acquired. If an acquire() "
            "raises, release the locks already acquired in the reverse of "
            "the order they were taken and let the exception propagate. "
            "Raise ValueError when two locks share a name, or when any name "
            "is not a str, before acquiring anything."
        ),
        validator=LOAD_CANDIDATE + require("acquire_all") + r'''
class Lock:
    """Records into a shared log instead of blocking.

    Deadlock cannot be demonstrated deterministically by running threads and
    hoping they interleave, and a validator that depended on that would fail
    on a loaded host. The property that actually prevents the deadlock is
    observable without any concurrency at all: the acquisition sequence must
    be a function of the SET of locks, not of the caller's argument order.
    """

    def __init__(self, name, log, fails=False):
        self.name = name
        self.log = log
        self.fails = fails
        self.held = False

    def acquire(self):
        if self.fails:
            raise RuntimeError("cannot acquire %s" % self.name)
        assert not self.held, "%s acquired twice" % self.name
        self.held = True
        self.log.append(("acquire", self.name))

    def release(self):
        assert self.held, "%s released without being held" % self.name
        self.held = False
        self.log.append(("release", self.name))


def acquisitions(log):
    return [name for kind, name in log if kind == "acquire"]


# Two callers, opposite argument orders, one acquisition sequence.
first, second = [], []
a1, b1, c1 = Lock("alpha", first), Lock("beta", first), Lock("gamma", first)
a2, b2, c2 = Lock("alpha", second), Lock("beta", second), Lock("gamma", second)

assert acquire_all([c1, a1, b1]) == ["alpha", "beta", "gamma"]
assert acquire_all([b2, c2, a2]) == ["alpha", "beta", "gamma"]
assert acquisitions(first) == acquisitions(second), (
    "the order two callers acquire in depends on how they listed the locks, "
    "which is exactly the inversion that deadlocks"
)
assert all(lock.held for lock in (a1, b1, c1, a2, b2, c2))

# Names order as strings, not by any incidental numeric reading.
log = []
ordered = acquire_all([Lock("item10", log), Lock("item9", log), Lock("item1", log)])
assert ordered == ["item1", "item10", "item9"], ordered

# A failure partway through unwinds what it took, newest first, and nothing
# stays held.
log = []
good_a = Lock("aaa", log)
good_b = Lock("bbb", log)
broken = Lock("ccc", log, fails=True)
try:
    acquire_all([broken, good_b, good_a])
except RuntimeError:
    pass
else:
    raise AssertionError("a failing acquire must propagate")
assert log == [
    ("acquire", "aaa"), ("acquire", "bbb"),
    ("release", "bbb"), ("release", "aaa"),
], log
assert not good_a.held and not good_b.held

# A lock that fails first leaves nothing to unwind.
log = []
assert_first = Lock("aaa", log, fails=True)
try:
    acquire_all([assert_first, Lock("bbb", log)])
except RuntimeError:
    pass
else:
    raise AssertionError("a failing acquire must propagate")
assert log == [], log

# Duplicate and non-str names are refused before anything is acquired.
log = []
for bad in ([Lock("dup", log), Lock("dup", log)],
            [Lock("ok", log), Lock(7, log)]):
    try:
        acquire_all(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid names must raise ValueError")
    assert log == [], "nothing may be acquired before the names are checked"

# Empty and single-lock inputs are ordinary.
assert acquire_all([]) == []
log = []
assert acquire_all([Lock("only", log)]) == ["only"]
assert acquisitions(log) == ["only"]
'''),
    task(
        f"{FAMILY}-0303", FAMILY,
        prompt=(
            "Implement a Python class HybridLogicalClock(physical) and a "
            "function happens_before(left, right). `physical` is a "
            "zero-argument callable returning the node's wall clock as an "
            "integer number of milliseconds; it may jump forward, stall, or "
            "move BACKWARDS. A timestamp is a tuple (l, c) of two "
            "integers, ordered lexicographically, and the clock starts at "
            "(0, 0). now() returns the current timestamp without changing "
            "it. local() stamps a local event: read the physical time pt, "
            "set l' = max(l, pt), and set c' = c + 1 when l' equals the old "
            "l, otherwise 0. receive(message) merges a timestamp (lm, cm) "
            "received from another node: read pt, set "
            "l' = max(l, lm, pt), then set c' = max(c, cm) + 1 when l' "
            "equals BOTH the old l and lm, c' = c + 1 when it equals only "
            "the old l, c' = cm + 1 when it equals only lm, and 0 "
            "otherwise. local() and receive() both return the new "
            "timestamp. happens_before(left, right) reports whether the "
            "left timestamp is strictly less than the right one."
        ),
        validator=LOAD_CANDIDATE + require("HybridLogicalClock")
        + require("happens_before") + SHAPE_GUARDS + r'''
happens_before = returning(happens_before, 'happens_before(...)')


class Physical:
    """A wall clock the test drives, including backwards."""

    def __init__(self, millis):
        self.millis = millis

    def __call__(self):
        return self.millis


def clock_at(millis):
    physical = Physical(millis)
    return physical, having(
        built(HybridLogicalClock(physical), 'HybridLogicalClock(...)'),
        'now', 'local', 'receive', what='the clock')


# The logical part TRACKS wall time. A Lamport clock -- the neighbouring
# technique, and the one an implementation drifts into -- counts events and
# would report 1 here however large the physical reading is.
physical, clock = clock_at(1_000)
assert built(clock.now(), 'now()') == (0, 0)
assert built(clock.local(), 'local()') == (1_000, 0)

# Two events inside the same millisecond separate on the counter, not on l.
assert clock.local() == (1_000, 1)
assert clock.local() == (1_000, 2)

# Wall time advancing resets the counter rather than continuing it.
physical.millis = 1_005
assert clock.local() == (1_005, 0)

# THE CLOCK MOVES BACKWARDS. This is the whole reason the max is there: an
# implementation that assigns l = pt is correct on every monotonic fixture
# and issues a duplicate-ordered timestamp on this one.
physical.millis = 900
assert clock.local() == (1_005, 1), 'l may never go backwards'
assert clock.local() == (1_005, 2)
physical.millis = 1_005
assert clock.local() == (1_005, 3), \
    'wall time catching up to l is not an advance'

# receive when both sides already sit at the same l keeps the LARGER
# counter. Taking the sender's would move this node backwards from
# (1_005, 3) to (1_005, 2).
assert clock.receive((1_005, 1)) == (1_005, 4)

# receive when only the sender is ahead adopts the sender's counter.
physical.millis = 1_005
assert clock.receive((2_000, 7)) == (2_000, 8)

# receive when the local wall clock is ahead of both resets the counter.
physical.millis = 3_000
assert clock.receive((2_500, 40)) == (3_000, 0)

# A message from the past cannot drag this node back.
assert clock.receive((10, 0)) == (3_000, 1)

# Causality: every stamp a node issues is strictly greater than the last,
# and a receive is strictly greater than the message it merged.
physical, clock = clock_at(50)
issued = []
for step in range(60):
    physical.millis = [50, 51, 51, 49, 52, 40][step % 6]
    issued.append(clock.local() if step % 2 else clock.receive((51, step)))
for earlier, later in zip(issued, issued[1:]):
    assert happens_before(earlier, later) is True, \
        f'{earlier} must precede {later}'
    assert happens_before(later, earlier) is False

assert happens_before((1, 5), (2, 0)) is True
assert happens_before((2, 0), (2, 1)) is True
assert happens_before((2, 1), (2, 1)) is False, 'equal is not before'
assert happens_before((2, 1), (2, 0)) is False
''',
    ),
    task(
        f"{FAMILY}-0304", FAMILY,
        prompt=(
            "Implement a Python class Voter(node_id, log) casting leader "
            "election votes under the Raft rules. `log` is a list of "
            "integers, one per entry, giving the term that entry was "
            "created in; entries are numbered from 1, so the last log "
            "index is len(log) and the last log term is the final element, "
            "or 0 and 0 for an empty log. The voter exposes attributes "
            "current_term (starting at 0) and voted_for (starting None), "
            "and a method request_vote(term, candidate_id, "
            "last_log_index, last_log_term) returning the tuple "
            "(current_term, granted). Reject without any change when term "
            "is below current_term. When term is above current_term, adopt "
            "it and clear voted_for FIRST -- this happens even if the vote "
            "is then refused. Grant only when the voter has not already "
            "voted in this term for a different candidate AND the "
            "candidate's log is at least as up to date as the voter's, "
            "which means the candidate's last log term is higher, or the "
            "terms are equal and the candidate's last log index is at "
            "least the voter's. Record voted_for when granting."
        ),
        validator=LOAD_CANDIDATE + require("Voter") + SHAPE_GUARDS + r'''
def voter(log, node_id='v'):
    return having(built(Voter(node_id, list(log)), 'Voter(...)'),
                  'current_term', 'voted_for', 'request_vote',
                  what='the voter')


# The voter's own log ends at index 3, term 2.
node = voter([1, 1, 2])
assert node.current_term == 0
assert node.voted_for is None

# A LONGER log at an OLDER term loses. This is the case the rule exists for
# and the one a length comparison gets wrong: index 5 beats index 3, so a
# candidate that trails the committed term wins the election and truncates
# entries the cluster already agreed on.
assert built(node.request_vote(1, 'a', 5, 1), 'request_vote(...)') == \
    (1, False), 'a longer log at an older term is not up to date'
assert node.voted_for is None, 'a refused vote is not recorded'
assert node.current_term == 1, 'the term is adopted even when refusing'

# A SHORTER log at a newer term wins, for the same reason.
assert node.request_vote(2, 'b', 2, 3) == (2, True)
assert node.voted_for == 'b'

# One vote per term, and asking twice is idempotent rather than a second
# vote -- a retransmitted RequestVote must not be refused.
assert node.request_vote(2, 'b', 2, 3) == (2, True)
assert node.request_vote(2, 'c', 9, 9) == (2, False), \
    'the voter already voted for b in term 2'
assert node.voted_for == 'b'

# A stale term is refused and changes nothing at all.
assert node.request_vote(1, 'c', 9, 9) == (2, False)
assert node.current_term == 2
assert node.voted_for == 'b'

# A higher term clears the recorded vote, so c can now win.
assert node.request_vote(3, 'c', 9, 9) == (3, True)
assert node.voted_for == 'c'

# Equal last term, equal last index: up to date, so granted.
fresh = voter([1, 1, 2])
assert fresh.request_vote(5, 'd', 3, 2) == (5, True)

# Equal last term, shorter index: refused -- but the term still advances,
# which is what stops a partitioned voter from staying behind forever.
fresh = voter([1, 1, 2])
assert fresh.request_vote(7, 'e', 2, 2) == (7, False)
assert fresh.current_term == 7
assert fresh.voted_for is None

# An empty log is last index 0 at term 0, so anything is at least as good.
empty = voter([])
assert empty.request_vote(1, 'f', 0, 0) == (1, True)
assert voter([]).request_vote(1, 'g', 4, 2) == (1, True)

# ...and a voter WITH a log refuses an empty candidate.
assert voter([1]).request_vote(4, 'h', 0, 0) == (4, False)
''',
    ),
    task(
        f"{FAMILY}-0305", FAMILY,
        prompt=(
            "Implement a Python function run_saga(steps) that executes a "
            "distributed transaction as a compensating saga. `steps` is a "
            "list of (name, action, compensation) triples of a string and "
            "two zero-argument callables. Call each action in order. If "
            "every action returns, the saga succeeded. If an action raises "
            "an exception, stop: do not call any later action, and undo "
            "the work that actually happened by calling the compensation "
            "of each SUCCEEDED step in reverse order. The failed step's "
            "own compensation must not be called, because its action did "
            "not complete. A compensation that itself raises is recorded "
            "and does not stop the remaining compensations from running. "
            "Return an object with attributes succeeded (a bool), "
            "completed (the names whose actions returned, in order), "
            "compensated (the names whose compensations returned, in the "
            "order they were called), failed_step (the name or None), "
            "error (the exception or None), and compensation_errors (a "
            "list of (name, exception) pairs in call order)."
        ),
        validator=LOAD_CANDIDATE + require("run_saga") + SHAPE_GUARDS + r'''
run_saga = returning(run_saga, 'run_saga(...)')

FIELDS = ('succeeded', 'completed', 'compensated', 'failed_step', 'error',
          'compensation_errors')


def step(name, log, fails=False, undo_fails=False):
    def action():
        log.append('do:' + name)
        if fails:
            raise RuntimeError('boom:' + name)

    def compensation():
        log.append('undo:' + name)
        if undo_fails:
            raise RuntimeError('undo-boom:' + name)

    return (name, action, compensation)


def run(steps):
    return having(run_saga(steps), *FIELDS, what='the saga result')


# Everything succeeds: nothing is compensated.
log = []
result = run([step('a', log), step('b', log), step('c', log)])
assert result.succeeded is True
assert list(result.completed) == ['a', 'b', 'c']
assert list(result.compensated) == []
assert result.failed_step is None
assert result.error is None
assert list(result.compensation_errors) == []
assert log == ['do:a', 'do:b', 'do:c']

# The third of four fails. Two facts are asserted here that a saga which
# merely "undoes everything" gets wrong: d's action never runs, and c's
# compensation never runs because c never took effect. Compensating c would
# refund a payment that was never captured.
log = []
result = run([step('a', log), step('b', log), step('c', log, fails=True),
              step('d', log)])
assert result.succeeded is False
assert list(result.completed) == ['a', 'b']
assert list(result.compensated) == ['b', 'a'], 'compensate in reverse order'
assert result.failed_step == 'c'
assert isinstance(result.error, RuntimeError)
assert str(result.error) == 'boom:c'
assert log == ['do:a', 'do:b', 'do:c', 'undo:b', 'undo:a']
assert 'undo:c' not in log, "the failed step's action never completed"
assert 'do:d' not in log

# A compensation that raises does not abandon the ones still owed. Stopping
# there is the plausible wrong move, and it leaves the earliest steps -- the
# ones holding the most state -- permanently uncompensated.
log = []
result = run([step('a', log), step('b', log, undo_fails=True),
              step('c', log, fails=True)])
assert list(result.completed) == ['a', 'b']
assert list(result.compensated) == ['a'], 'b raised, so it did not compensate'
assert log == ['do:a', 'do:b', 'do:c', 'undo:b', 'undo:a'], \
    'a must still be compensated after b fails to'
errors = list(result.compensation_errors)
assert [name for name, _ in errors] == ['b']
assert str(errors[0][1]) == 'undo-boom:b'

# The very first step failing compensates nothing.
log = []
result = run([step('a', log, fails=True), step('b', log)])
assert list(result.completed) == []
assert list(result.compensated) == []
assert result.failed_step == 'a'
assert log == ['do:a']

# An empty saga trivially succeeds.
result = run([])
assert result.succeeded is True
assert list(result.completed) == []
assert result.failed_step is None
''',
    ),
    task(
        f"{FAMILY}-0306", FAMILY,
        prompt=(
            "Implement a Python class Budget(clock, seconds) and an "
            "exception DeadlineExceeded for propagating a request deadline "
            "through nested calls. `clock` is a zero-argument callable "
            "returning a monotonic float in seconds. The budget fixes an "
            "ABSOLUTE deadline of clock() + seconds at construction. "
            "remaining() returns the seconds left, never below zero; "
            "expired() reports whether nothing is left; check() raises "
            "DeadlineExceeded when expired and otherwise returns None. "
            "child(seconds) returns a new Budget on the same clock whose "
            "deadline is the EARLIER of this budget's deadline and "
            "clock() + seconds, so a callee can shorten its own allowance "
            "but can never extend the caller's. Also implement "
            "run_stages(budget, stages), where `stages` is a list of "
            "(name, work) pairs and work is called with the budget: call "
            "budget.check() before each stage, and return the list of "
            "names that ran. If check() raises, let DeadlineExceeded "
            "propagate with an attribute `completed` listing the names "
            "that did run."
        ),
        validator=LOAD_CANDIDATE + require("Budget")
        + require("DeadlineExceeded") + require("run_stages")
        + SHAPE_GUARDS + r'''
class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def budget_of(clock, seconds):
    return having(built(Budget(clock, seconds), 'Budget(...)'),
                  'remaining', 'expired', 'check', 'child',
                  what='the budget')


assert isinstance(DeadlineExceeded, type) and \
    issubclass(DeadlineExceeded, BaseException), \
    'DeadlineExceeded must be an exception class'

clock = Clock(100.0)
parent = budget_of(clock, 10.0)
assert abs(built(parent.remaining(), 'remaining()') - 10.0) < 1e-9
assert parent.expired() is False
assert parent.check() is None

# THE DEADLINE IS ABSOLUTE, not a duration that restarts on each read.
clock.now = 104.0
assert abs(parent.remaining() - 6.0) < 1e-9

# A child may shorten...
short = having(built(parent.child(2.0), 'child(...)'), 'remaining',
               what='the child budget')
assert abs(short.remaining() - 2.0) < 1e-9

# ...and may NEVER lengthen. A callee that asks for a 1000 s timeout on a
# request with 6 s left is the whole point of budget propagation: taking the
# requested value holds the caller's connection open long after it gave up.
patient = parent.child(1000.0)
assert abs(patient.remaining() - 6.0) < 1e-9, \
    'a child cannot outlive its parent'

# A grandchild is bounded by the parent it can no longer see.
assert abs(patient.child(500.0).remaining() - 6.0) < 1e-9

# Time already spent is charged to the child at CREATION, so a child made
# later gets less -- this is what distinguishes an absolute deadline from a
# per-call timeout, which would hand out 5 s here.
clock.now = 108.0
late = parent.child(5.0)
assert abs(late.remaining() - 2.0) < 1e-9

# A child that expires does not expire its parent.
clock.now = 104.0
sibling = parent.child(1.0)
clock.now = 105.5
assert sibling.expired() is True
assert parent.expired() is False
try:
    sibling.check()
except DeadlineExceeded:
    pass
else:
    raise AssertionError('an expired budget must raise from check()')

# remaining() floors at zero rather than going negative.
clock.now = 200.0
assert parent.remaining() == 0
assert parent.expired() is True

# run_stages charges elapsed work against the shared deadline and stops at
# the first stage it cannot afford to start.
clock = Clock(0.0)
budget = budget_of(clock, 5.0)
ran = []


def spend(seconds, name):
    def work(current):
        assert current is not None, 'the stage was not given a budget'
        ran.append(name)
        clock.now += seconds
    return (name, work)


assert list(run_stages(budget, [spend(1.0, 'a'), spend(1.0, 'b')])) == \
    ['a', 'b']
assert ran == ['a', 'b']

clock = Clock(0.0)
budget = budget_of(clock, 4.0)
ran = []
try:
    run_stages(budget, [spend(2.0, 'a'), spend(2.0, 'b'), spend(2.0, 'c'),
                        spend(2.0, 'd')])
except DeadlineExceeded as exceeded:
    assert list(getattr(exceeded, 'completed', None) or []) == ['a', 'b'], \
        'the exception must name the stages that did run'
else:
    raise AssertionError('run_stages ran past its deadline')
assert ran == ['a', 'b'], 'c must not start with a spent budget'

# An empty stage list is fine and does not consult the clock.
assert list(run_stages(budget_of(Clock(0.0), 0.0), [])) == []
''',
    ),
    task(
        f"{FAMILY}-0307", FAMILY,
        prompt=(
            "Implement a Python class BoundedQueue(capacity) plus "
            "exception classes QueueClosed and QueueDrained, providing a "
            "blocking hand-off between producer and consumer threads. "
            "put(item) appends an item, blocking while the queue already "
            "holds `capacity` items, and raises QueueClosed if the queue "
            "is closed -- at call time or while it was blocked. get() "
            "removes and returns the oldest item, blocking while the queue "
            "is empty and still open. close() marks the queue closed, is "
            "idempotent, and must wake every blocked caller. Closing does "
            "NOT discard what is already buffered: get() keeps returning "
            "buffered items in FIFO order after close, and only once the "
            "buffer is empty does it raise QueueDrained. Use "
            "threading primitives; do not busy-wait."
        ),
        validator=LOAD_CANDIDATE + require("BoundedQueue")
        + require("QueueClosed") + require("QueueDrained")
        + SHAPE_GUARDS + r'''
import threading

for name, kind in (('QueueClosed', QueueClosed),
                   ('QueueDrained', QueueDrained)):
    assert isinstance(kind, type) and issubclass(kind, BaseException), \
        f'{name} must be an exception class'


def queue(capacity):
    return having(built(BoundedQueue(capacity), 'BoundedQueue(...)'),
                  'put', 'get', 'close', what='the queue')


def run(target):
    """Run in a thread and re-raise whatever it did, without hanging."""
    box = {}

    def body():
        try:
            box['value'] = target()
        except BaseException as error:  # noqa: BLE001
            box['error'] = error

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread, box


# Ordinary FIFO hand-off inside the capacity.
q = queue(3)
q.put('a')
q.put('b')
assert built(q.get(), 'get()') == 'a'
assert q.get() == 'b'

# CLOSING DOES NOT DISCARD THE BUFFER. Dropping it is the plausible wrong
# move and it loses acknowledged work: the producer was told these were
# accepted.
q = queue(3)
q.put('x')
q.put('y')
q.close()
assert q.get() == 'x', 'buffered items survive close'
assert q.get() == 'y'
try:
    q.get()
except QueueDrained:
    pass
else:
    raise AssertionError('a closed, empty queue must raise QueueDrained')

# close() is idempotent, and put after close is refused.
q.close()
try:
    q.put('z')
except QueueClosed:
    pass
else:
    raise AssertionError('put on a closed queue must raise QueueClosed')

# A consumer blocked on an empty queue is woken by close, not left hanging.
q = queue(2)
thread, box = run(q.get)
thread.join(0.2)
assert thread.is_alive(), 'get must block while the queue is empty and open'
q.close()
thread.join(5.0)
assert not thread.is_alive(), 'close must wake a blocked get'
assert isinstance(box.get('error'), QueueDrained), box

# A producer blocked on a full queue is woken by close too.
q = queue(1)
q.put('full')
thread, box = run(lambda: q.put('blocked'))
thread.join(0.2)
assert thread.is_alive(), 'put must block while the queue is full'
q.close()
thread.join(5.0)
assert not thread.is_alive(), 'close must wake a blocked put'
assert isinstance(box.get('error'), QueueClosed), box
# ...and the item it was holding was never accepted, while the one already
# buffered still is.
assert q.get() == 'full'
try:
    q.get()
except QueueDrained:
    pass
else:
    raise AssertionError('the refused put must not have been buffered')

# Capacity is real: a blocked producer proceeds the moment a slot frees.
q = queue(1)
q.put(0)
thread, box = run(lambda: q.put(1))
thread.join(0.2)
assert thread.is_alive()
assert q.get() == 0
thread.join(5.0)
assert not thread.is_alive(), 'a freed slot must wake a blocked put'
assert 'error' not in box, box
assert q.get() == 1

# Nothing is lost or duplicated across many threads.
q = queue(4)
produced = list(range(200))
consumed = []
lock = threading.Lock()


def produce():
    for item in produced:
        q.put(item)
    q.close()


def consume():
    while True:
        try:
            item = q.get()
        except QueueDrained:
            return
        with lock:
            consumed.append(item)


workers = [threading.Thread(target=produce, daemon=True)]
workers += [threading.Thread(target=consume, daemon=True) for _ in range(3)]
for worker in workers:
    worker.start()
for worker in workers:
    worker.join(20.0)
    assert not worker.is_alive(), 'a worker never finished'
assert sorted(consumed) == produced, 'every item is delivered exactly once'
''',
    ),
    task(
        f"{FAMILY}-0308", FAMILY,
        prompt=(
            "Implement a Python class IdGenerator(node_id, clock) plus an "
            "exception ClockMovedBackwards, generating Snowflake-style "
            "64-bit identifiers. `clock` is a zero-argument callable "
            "returning the wall clock as an integer number of "
            "milliseconds. Expose the class attribute EPOCH_MS = "
            "1700000000000. node_id must be an integer in 0..1023, and "
            "anything else raises ValueError. next_id() returns "
            "((millis - EPOCH_MS) << 22) | (node_id << 12) | sequence, "
            "where sequence starts at 0 for each new millisecond and "
            "increments for each further id inside the same millisecond. "
            "When the sequence would exceed 4095, do not wrap: keep "
            "reading the clock until it reports a later millisecond, then "
            "restart the sequence at 0. If the clock ever reports a "
            "millisecond EARLIER than the last one used, raise "
            "ClockMovedBackwards rather than returning an id."
        ),
        validator=LOAD_CANDIDATE + require("IdGenerator")
        + require("ClockMovedBackwards") + SHAPE_GUARDS + r'''
assert isinstance(ClockMovedBackwards, type) and \
    issubclass(ClockMovedBackwards, BaseException), \
    'ClockMovedBackwards must be an exception class'

EPOCH = getattr(IdGenerator, 'EPOCH_MS', None)
assert EPOCH == 1700000000000, f'EPOCH_MS is {EPOCH!r}'


class Clock:
    """A wall clock the test drives, and which counts its own reads."""

    def __init__(self, millis, advance_after=None, step=1):
        self.millis = millis
        self.reads = 0
        self.advance_after = advance_after
        self.step = step

    def __call__(self):
        self.reads += 1
        if self.advance_after is not None and self.reads > self.advance_after:
            self.millis += self.step
            self.advance_after = None
        return self.millis


def generator(clock, node_id=7):
    return having(built(IdGenerator(node_id, clock), 'IdGenerator(...)'),
                  'next_id', what='the generator')


for bad in (-1, 1024, 5000, 'x', None, 1.5):
    try:
        IdGenerator(bad, Clock(EPOCH))
    except ValueError:
        pass
    else:
        raise AssertionError(f'accepted node_id {bad!r}')

for good in (0, 1023):
    generator(Clock(EPOCH), good)

# The layout is exactly the one the prompt specifies.
clock = Clock(EPOCH + 12345)
gen = generator(clock, 7)
first = built(gen.next_id(), 'next_id()')
assert isinstance(first, int)
assert first >> 22 == 12345, 'the high bits carry millis since EPOCH_MS'
assert (first >> 12) & 0x3FF == 7, 'the middle bits carry the node id'
assert first & 0xFFF == 0, 'the sequence starts at 0 in a new millisecond'

# Inside one millisecond the sequence increments and the id still rises.
second = gen.next_id()
assert second & 0xFFF == 1
assert second > first

# A new millisecond restarts the sequence.
clock.millis = EPOCH + 12346
third = gen.next_id()
assert third >> 22 == 12346
assert third & 0xFFF == 0
assert third > second

# SEQUENCE EXHAUSTION MUST WAIT, NOT WRAP. Wrapping is the plausible wrong
# move: every assertion above still passes, ids still rise for a while, and
# the 4097th id in a millisecond silently duplicates the first.
#
# The threshold is chosen by tracing reads, not estimated. A generator that
# does not spin reads the clock exactly once per id, so it has made 4097
# reads when it issues the 4097th. The clock therefore holds its value until
# read 4098 -- one read PAST the point a non-spinning generator ever reaches.
# Only an implementation that keeps reading gets a fresh millisecond; a
# wrapping one re-issues sequence 0 in the millisecond it already used, and
# the uniqueness assertion below is what catches it.
clock = Clock(EPOCH + 500, advance_after=4098, step=1)
gen = generator(clock)
issued = [gen.next_id() for _ in range(4200)]
assert len(set(issued)) == 4200, 'ids must be unique across the boundary'
assert issued == sorted(issued), 'ids must increase monotonically'
assert [identifier & 0xFFF for identifier in issued[:4096]] == \
    list(range(4096))
assert issued[4096] >> 22 == 501, 'the 4097th id belongs to a later ms'
assert issued[4096] & 0xFFF == 0, 'the sequence restarts at 0'
assert issued[4097] & 0xFFF == 1

# A backwards clock is refused rather than allowed to duplicate ids.
clock = Clock(EPOCH + 9_000)
gen = generator(clock)
before = gen.next_id()
clock.millis = EPOCH + 8_999
try:
    gen.next_id()
except ClockMovedBackwards:
    pass
else:
    raise AssertionError('a backwards clock must raise')

# Recovery: once the clock passes the highest millisecond already used, ids
# resume -- and do not collide with the ones issued before the regression.
clock.millis = EPOCH + 9_001
after = gen.next_id()
assert after > before
assert after >> 22 == 9_001
''',
    ),
    task(
        f"{FAMILY}-0309", FAMILY,
        prompt=(
            "Implement a Python class SingleFlight with a method "
            "do(key, function) that collapses duplicate concurrent work. "
            "`function` is zero-argument. When several threads call do() "
            "with the same key while a call for that key is still "
            "running, the function runs ONCE and every caller receives "
            "that one result; if it raises, every caller receives that "
            "same exception. Different keys never share a call. Once a "
            "call finishes, the key is released: this is de-duplication of "
            "work in flight, not a cache, so a later do() with the same "
            "key runs the function again. Expose the count of function "
            "invocations as the attribute calls. Use threading "
            "primitives; do not busy-wait."
        ),
        validator=LOAD_CANDIDATE + require("SingleFlight") + SHAPE_GUARDS
        + r'''
import threading
import time

group = having(built(SingleFlight(), 'SingleFlight()'), 'do', 'calls',
               what='the single-flight group')
assert group.calls == 0

# A lone call is ordinary, and the key is released afterwards.
assert built(group.do('k', lambda: 'one'), "do('k', ...)") == 'one'
assert group.calls == 1

# THE KEY IS RELEASED, NOT MEMOIZED. Caching the value passes every
# concurrency assertion below and is a different data structure: the second
# request would serve a stale value forever.
assert group.do('k', lambda: 'two') == 'two', \
    'single flight de-duplicates work in flight, it does not cache'
assert group.calls == 2

# Many threads, one key, one execution. The function parks until it is
# released, so every caller is genuinely in flight together.
group = having(built(SingleFlight(), 'SingleFlight()'), 'do', 'calls',
               what='the single-flight group')
entered = threading.Event()
release = threading.Event()
results = {}
lock = threading.Lock()
# The leader parks inside the function, so the key stays in flight until this
# test says otherwise. The only thing that has to be sequenced is that every
# caller has REGISTERED before the leader is allowed to finish -- a straggler
# arriving after the key is released is a second leader, and would look like
# a candidate that failed to collapse the calls. Each caller announces itself
# immediately before calling do(), and the settle covers the few bytecodes
# between that announcement and taking the group's lock.
arrived = [threading.Event() for _ in range(6)]


def slow():
    entered.set()
    assert release.wait(10.0), 'the shared call was never released'
    return 'shared'


def caller(index):
    def body():
        arrived[index].set()
        value = group.do('hot', slow)
        with lock:
            results[index] = value
    return body


threads = [threading.Thread(target=caller(index), daemon=True)
           for index in range(6)]
for thread in threads:
    thread.start()
assert entered.wait(10.0), 'no thread ever entered the function'
for event in arrived:
    assert event.wait(10.0), 'a caller never started'
time.sleep(0.5)
assert group.calls == 1, \
    f'the function ran {group.calls} times while one call was in flight'
release.set()
for thread in threads:
    thread.join(20.0)
    assert not thread.is_alive(), 'a caller never returned'

assert group.calls == 1, f'the function ran {group.calls} times, not once'
assert results == {index: 'shared' for index in range(6)}

# ...and after all of that the key is free again.
assert group.do('hot', lambda: 'later') == 'later'
assert group.calls == 2

# Different keys do not block one another. Each parks until BOTH have
# arrived, so a group that serialises every key deadlocks here rather than
# quietly passing.
group = having(built(SingleFlight(), 'SingleFlight()'), 'do', 'calls',
               what='the single-flight group')
together = threading.Barrier(2, timeout=10.0)
seen = {}


def paired(key):
    def body():
        seen[key] = group.do(key, lambda: (together.wait(), key)[1])
    return body


pair = [threading.Thread(target=paired(key), daemon=True)
        for key in ('left', 'right')]
for thread in pair:
    thread.start()
for thread in pair:
    thread.join(20.0)
    assert not thread.is_alive(), 'distinct keys must run concurrently'
assert seen == {'left': 'left', 'right': 'right'}
assert group.calls == 2

# A failure is shared by every waiter, and does NOT poison the key.
group = having(built(SingleFlight(), 'SingleFlight()'), 'do', 'calls',
               what='the single-flight group')
entered = threading.Event()
release = threading.Event()
errors = {}
arrived = [threading.Event() for _ in range(4)]


def failing():
    entered.set()
    assert release.wait(10.0)
    raise RuntimeError('shared failure')


def catcher(index):
    def body():
        arrived[index].set()
        try:
            group.do('bad', failing)
        except RuntimeError as error:
            errors[index] = str(error)
    return body


threads = [threading.Thread(target=catcher(index), daemon=True)
           for index in range(4)]
for thread in threads:
    thread.start()
assert entered.wait(10.0)
for event in arrived:
    assert event.wait(10.0), 'a caller never started'
time.sleep(0.5)
release.set()
for thread in threads:
    thread.join(20.0)
    assert not thread.is_alive(), 'a caller never saw the failure'
assert errors == {index: 'shared failure' for index in range(4)}, errors
assert group.calls == 1
assert group.do('bad', lambda: 'recovered') == 'recovered', \
    'a failed call must not poison the key'
''',
    ),
    task(
        f"{FAMILY}-0310", FAMILY,
        prompt=(
            "Implement a Python class WorkStealingDeque() with the "
            "Chase-Lev ownership split. The owning worker uses "
            "push_bottom(item) and pop_bottom(); other threads use "
            "steal_top(). push_bottom appends. pop_bottom removes and "
            "returns the item pushed MOST recently, so the owner works "
            "depth-first over the tasks whose data is still warm. "
            "steal_top removes and returns the OLDEST item, so a thief "
            "takes the work furthest from the owner's end. Both return "
            "the module-level sentinel EMPTY when there is nothing to "
            "take. When one item remains, exactly one caller may get it: "
            "an owner and a thief racing for it must not both receive it, "
            "and it must not be lost. Also expose __len__. Use threading "
            "primitives; do not busy-wait."
        ),
        validator=LOAD_CANDIDATE + require("WorkStealingDeque")
        + require("EMPTY") + SHAPE_GUARDS + r'''
import threading

deque = having(built(WorkStealingDeque(), 'WorkStealingDeque()'),
               'push_bottom', 'pop_bottom', 'steal_top', '__len__',
               what='the deque')
assert len(deque) == 0
assert deque.pop_bottom() is EMPTY
assert deque.steal_top() is EMPTY

# THE TWO ENDS ARE DIFFERENT ENDS. A deque whose steal_top pops the same
# side as pop_bottom passes a single-threaded smoke test and destroys the
# locality the structure exists for -- and, worse, makes thieves contend
# with the owner for the very task it is about to run.
for item in ('a', 'b', 'c', 'd'):
    deque.push_bottom(item)
assert len(deque) == 4
assert deque.pop_bottom() == 'd', 'the owner takes the newest item'
assert deque.steal_top() == 'a', 'a thief takes the oldest item'
assert deque.pop_bottom() == 'c'
assert deque.steal_top() == 'b'
assert len(deque) == 0
assert deque.pop_bottom() is EMPTY
assert deque.steal_top() is EMPTY

# Pushing again after draining works from a clean state.
deque.push_bottom(1)
assert deque.steal_top() == 1
deque.push_bottom(2)
assert deque.pop_bottom() == 2

# The single-element case, taken from each side in turn: exactly one caller
# gets it and the other sees EMPTY.
for first, second in (('pop_bottom', 'steal_top'),
                      ('steal_top', 'pop_bottom')):
    solo = WorkStealingDeque()
    solo.push_bottom('only')
    assert getattr(solo, first)() == 'only'
    assert getattr(solo, second)() is EMPTY, \
        f'{second} took an item {first} had already removed'

# Under contention nothing is duplicated and nothing is lost. This is the
# assertion the single-element race actually shows up in.
deque = WorkStealingDeque()
ITEMS = 3000
taken = []
guard = threading.Lock()
stop = threading.Event()


def collect(values):
    with guard:
        taken.extend(values)


def owner():
    mine = []
    for item in range(ITEMS):
        deque.push_bottom(item)
        if item % 3 == 0:
            got = deque.pop_bottom()
            if got is not EMPTY:
                mine.append(got)
    while True:
        got = deque.pop_bottom()
        if got is EMPTY:
            break
        mine.append(got)
    stop.set()
    collect(mine)


def thief():
    mine = []
    while not stop.is_set() or len(deque):
        got = deque.steal_top()
        if got is not EMPTY:
            mine.append(got)
    collect(mine)


workers = [threading.Thread(target=owner, daemon=True)]
workers += [threading.Thread(target=thief, daemon=True) for _ in range(3)]
for worker in workers:
    worker.start()
for worker in workers:
    worker.join(30.0)
    assert not worker.is_alive(), 'a worker never finished'

assert len(deque) == 0, 'the deque must be drained'
assert len(taken) == len(set(taken)), 'an item was taken twice'
assert sorted(taken) == list(range(ITEMS)), 'an item was lost'
''',
    ),
]
