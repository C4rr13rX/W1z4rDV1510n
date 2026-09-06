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
        validator=LOAD_CANDIDATE + require("HashRing") + '''
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
        validator=LOAD_CANDIDATE + require("PNCounter") + '''
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
        + require("LeaseHeldError") + require("StaleTokenError") + '''
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
        validator=LOAD_CANDIDATE + require("TumblingWindows") + '''
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
]
