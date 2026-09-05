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
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

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
        validator=LOAD_CANDIDATE + require("TokenBucket") + r'''
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
        + require("CircuitOpenError") + r'''
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
        validator=LOAD_CANDIDATE + require("ExactlyOnceInbox") + r'''
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
        validator=LOAD_CANDIDATE + require("Once") + r'''
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
        validator=LOAD_CANDIDATE + require("ReadWriteLock") + r'''
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
]
