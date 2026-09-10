"""Run a plausible WRONG solution against a task and report the verdict.

`test_a_broken_solution_fails_its_validator` mutates the reference, which is
the strong form of "is this validator discriminating at all". It is not the
strong form of "is this validator measuring the capability its prompt names".
That question is only answered by the implementation a competent engineer
writes when they miss the point -- the near-miss -- and
`docs/PROGRAMMING_OBSTACLE_COURSE.md` records a task where the reference
passed, the mutation failed, and the near-miss passed too, because the
fixture never separated the named technique from its neighbour.

So this runs named near-misses through the same runner the contract tests
use and prints, per task, whether it was rejected and on which assertion. A
near-miss that PASSES is the finding: the task is scoring something weaker
than it claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.programming_obstacle_run import PASSED, run_task  # noqa: E402
from scripts.programming_obstacle_tasks import load_authored_tasks  # noqa: E402

NEAR_MISSES: dict[str, tuple[str, str]] = {}

# A plain Lamport clock: one counter, incremented per event, merged with max.
# It is the technique HLC is built on top of and orders events perfectly
# well -- it simply carries no relationship to wall time, so nothing can be
# correlated with a log line or a metric.
NEAR_MISSES["concurrency_async_distributed-0303"] = ("plain Lamport clock", r'''
class HybridLogicalClock:
    def __init__(self, physical):
        self.physical = physical
        self._counter = 0

    def now(self):
        return (self._counter, 0)

    def local(self):
        self._counter += 1
        return self.now()

    def receive(self, message):
        self._counter = max(self._counter, message[0]) + 1
        return self.now()


def happens_before(left, right):
    return tuple(left) < tuple(right)
''')

# Refuses the stale-log candidate correctly, but only updates current_term
# when it GRANTS. A voter that refuses without adopting the term keeps
# advertising the old one and re-runs the same election forever.
NEAR_MISSES["concurrency_async_distributed-0304"] = ("term adopted only on grant", r'''
class Voter:
    def __init__(self, node_id, log):
        self.node_id = node_id
        self.log = list(log)
        self.current_term = 0
        self.voted_for = None

    def _last(self):
        return (self.log[-1] if self.log else 0, len(self.log))

    def request_vote(self, term, candidate_id, last_log_index, last_log_term):
        if term < self.current_term:
            return (self.current_term, False)
        up_to_date = (last_log_term, last_log_index) >= self._last()
        if term > self.current_term:
            voted = None
        else:
            voted = self.voted_for
        granted = up_to_date and voted in (None, candidate_id)
        if granted:
            self.current_term = term
            self.voted_for = candidate_id
        return (self.current_term, granted)
''')

# Undoes everything it touched, including the step that raised. Reads as the
# more careful choice and issues a refund for a payment never captured.
NEAR_MISSES["concurrency_async_distributed-0305"] = ("compensates the failed step too", r'''
class SagaResult:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def run_saga(steps):
    attempted = []
    failed_step = None
    error = None
    for name, action, compensation in steps:
        attempted.append((name, compensation))
        try:
            action()
        except Exception as failure:
            failed_step, error = name, failure
            break

    compensated = []
    compensation_errors = []
    if failed_step is not None:
        for name, compensation in reversed(attempted):
            try:
                compensation()
            except Exception as failure:
                compensation_errors.append((name, failure))
            else:
                compensated.append(name)

    return SagaResult(
        succeeded=failed_step is None,
        completed=[name for name, _ in attempted],
        compensated=compensated,
        failed_step=failed_step,
        error=error,
        compensation_errors=compensation_errors,
    )
''')

# Stores a DURATION rather than an absolute deadline, so every read reports
# the full allowance and time already spent is never charged.
NEAR_MISSES["concurrency_async_distributed-0306"] = ("duration, not a deadline", r'''
class DeadlineExceeded(Exception):
    pass


class Budget:
    def __init__(self, clock, seconds):
        self._clock = clock
        self._seconds = seconds
        self._started = clock()

    def remaining(self):
        return max(0.0, self._seconds)

    def expired(self):
        return self.remaining() <= 0

    def check(self):
        if self.expired():
            raise DeadlineExceeded("spent")
        return None

    def child(self, seconds):
        return Budget(self._clock, min(self._seconds, seconds))


def run_stages(budget, stages):
    completed = []
    for name, work in stages:
        try:
            budget.check()
        except DeadlineExceeded as exceeded:
            exceeded.completed = completed
            raise
        work(budget)
        completed.append(name)
    return completed
''')

# Blocks and bounds correctly, but treats close as a hard stop: the buffer is
# dropped and a blocked consumer is told the queue closed rather than being
# allowed to drain it.
NEAR_MISSES["concurrency_async_distributed-0307"] = ("close discards the buffer", r'''
import collections
import threading


class QueueClosed(Exception):
    pass


class QueueDrained(Exception):
    pass


class BoundedQueue:
    def __init__(self, capacity):
        self._capacity = capacity
        self._items = collections.deque()
        self._closed = False
        self._condition = threading.Condition()

    def put(self, item):
        with self._condition:
            while not self._closed and len(self._items) >= self._capacity:
                self._condition.wait()
            if self._closed:
                raise QueueClosed("closed")
            self._items.append(item)
            self._condition.notify_all()

    def get(self):
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if self._closed:
                raise QueueDrained("closed")
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self):
        with self._condition:
            self._closed = True
            self._items.clear()
            self._condition.notify_all()
''')

# Handles regression by clamping to the last millisecond used -- which is
# exactly how the duplicate gets minted, because the sequence for that
# millisecond has already been handed out.
NEAR_MISSES["concurrency_async_distributed-0308"] = ("clamps a backwards clock", r'''
class ClockMovedBackwards(Exception):
    pass


class IdGenerator:
    EPOCH_MS = 1700000000000
    MAX_SEQUENCE = 4095

    def __init__(self, node_id, clock):
        if not isinstance(node_id, int) or not 0 <= node_id <= 1023:
            raise ValueError("node_id")
        self.node_id = node_id
        self._clock = clock
        self._last_millis = -1
        self._sequence = 0

    def next_id(self):
        millis = max(self._clock(), self._last_millis)
        if millis == self._last_millis:
            if self._sequence >= self.MAX_SEQUENCE:
                while millis <= self._last_millis:
                    millis = self._clock()
                self._sequence = 0
            else:
                self._sequence += 1
        else:
            self._sequence = 0
        self._last_millis = millis
        return ((millis - self.EPOCH_MS) << 22) | (self.node_id << 12) \
            | self._sequence
''')

# Collapses duplicate work correctly by holding one global lock for the whole
# call. Every same-key assertion passes; what is lost is that unrelated keys
# now run one at a time.
NEAR_MISSES["concurrency_async_distributed-0309"] = ("one global lock", r'''
import threading


class SingleFlight:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0

    def do(self, key, function):
        with self._lock:
            self.calls += 1
            return function()
''')

# Locks properly and never loses an item, but hands a thief the newest task
# instead of the oldest -- the ownership split inverted.
NEAR_MISSES["concurrency_async_distributed-0310"] = ("thief takes the newest", r'''
import collections
import threading

EMPTY = object()


class WorkStealingDeque:
    def __init__(self):
        self._items = collections.deque()
        self._lock = threading.Lock()

    def push_bottom(self, item):
        with self._lock:
            self._items.append(item)

    def pop_bottom(self):
        with self._lock:
            return self._items.pop() if self._items else EMPTY

    def steal_top(self):
        with self._lock:
            return self._items.pop() if self._items else EMPTY

    def __len__(self):
        with self._lock:
            return len(self._items)
''')


def main() -> int:
    tasks = {item.task_id: item for item in load_authored_tasks()}
    escaped = 0
    for task_id, (label, source) in sorted(NEAR_MISSES.items()):
        item = tasks.get(task_id)
        if item is None:
            print(f"MISSING  {task_id}")
            escaped += 1
            continue
        result = run_task(item, source)
        if result.outcome == PASSED:
            escaped += 1
            print(f"PASSED   {task_id}  <-- {label}: the task is scoring a "
                  "weaker capability than its prompt names")
            continue
        detail = " ".join((result.detail or "").split())
        print(f"rejected {task_id}  ({label})\n"
              f"         {result.outcome}: {detail[-260:]}")
    print(f"\n{len(NEAR_MISSES) - escaped}/{len(NEAR_MISSES)} near-misses "
          "rejected")
    return 1 if escaped else 0


if __name__ == "__main__":
    raise SystemExit(main())
