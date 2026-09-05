"""Held-out tasks: reliability, observability, and performance.

The defects this family targets are the ones that only appear at production
volume: a metric that is correct but unbounded in memory, a sampler that
disagrees with itself across processes, a window that never evicts, an
aggregate that reports healthy while a dependency is down. Each is cheap to
write wrongly and expensive to discover, because the wrong version behaves
perfectly on the small inputs a developer tries by hand.

So the validators here assert two things that a test suite written from the
prompt's examples would not: a **resource ceiling** that holds after hundreds
of thousands of observations, and an **accuracy guarantee** stated as a bound
rather than as a golden value. An implementation that simply retains every
sample answers every accuracy question perfectly and still fails, which is
exactly the trade a production observability component is not allowed to make.

Rate limiting, circuit breaking and retry/backoff are deliberately absent:
`concurrency_async_distributed` already owns those three behaviour contracts,
and a second copy here would measure one capability twice while counting as
two tasks against the contract's family totals.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "reliability_observability_performance"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python class LatencyHistogram(relative_error) that "
            "estimates quantiles of a positive-valued stream in memory "
            "bounded by the value range rather than by the number of "
            "observations. Use logarithmic bucketing: with "
            "gamma = (1 + relative_error) / (1 - relative_error), a value v "
            "belongs to bucket index ceil(log(v) / log(gamma)), and that "
            "bucket's representative value is "
            "2 * gamma**index / (gamma + 1). Provide record(value), which "
            "raises ValueError unless value > 0; count() returning the number "
            "of observations; bucket_count() returning the number of occupied "
            "buckets; and quantile(q) returning the representative value of "
            "the lowest bucket whose cumulative count reaches "
            "max(1, ceil(q * count())). quantile raises ValueError if q is "
            "outside [0, 1] or the histogram is empty. Every returned "
            "quantile must be within relative_error of the true quantile of "
            "the observed values."
        ),
        timeout_seconds=120.0,
        validator=LOAD_CANDIDATE + require("LatencyHistogram") + r'''
import math

# --- accuracy is a bound, not a golden value ------------------------------
histogram = LatencyHistogram(0.01)
observations = list(range(1, 10001))
for value in observations:
    histogram.record(value)
assert histogram.count() == 10000, "count() lost observations"

ordered = sorted(observations)
for q in (0.0, 0.25, 0.5, 0.9, 0.99, 1.0):
    rank = max(1, math.ceil(q * len(ordered)))
    truth = ordered[rank - 1]
    got = histogram.quantile(q)
    assert abs(got - truth) <= 0.01 * truth + 1e-9, (
        f"q={q}: {got} is outside 1% of the true quantile {truth}"
    )

# Quantiles of one sample set never decrease as q rises.
previous = -1.0
for step in range(0, 101):
    current = histogram.quantile(step / 100.0)
    assert current >= previous - 1e-9, "quantile(q) is not monotonic in q"
    previous = current

# --- the ceiling that separates a histogram from a list of samples --------
wide = LatencyHistogram(0.01)
seed = 12345
for index in range(200000):
    # A deterministic spread over six orders of magnitude, so the bucket
    # count is exercised across the whole range without a random module.
    seed = (1103515245 * seed + 12345) % (2 ** 31)
    wide.record(1.0 + (seed % 1000000))
assert wide.count() == 200000, "count() lost observations"
assert wide.bucket_count() <= 2000, (
    f"bucket_count() is {wide.bucket_count()} after 200000 observations; "
    "memory must scale with the value range, not the stream length"
)

# --- stated error behaviour ----------------------------------------------
empty = LatencyHistogram(0.01)
for bad_call in (lambda: empty.quantile(0.5),
                 lambda: histogram.quantile(-0.01),
                 lambda: histogram.quantile(1.01),
                 lambda: histogram.record(0),
                 lambda: histogram.record(-5)):
    try:
        bad_call()
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid call was accepted")
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function should_sample(trace_id, rate) "
            "deciding head-based trace sampling. The decision must be a pure "
            "function of the trace id, identical in every process and every "
            "run, so that all services touching one trace agree without "
            "coordinating. Specification: take the SHA-256 digest of "
            "trace_id encoded as UTF-8, read its first 8 bytes as a "
            "big-endian unsigned integer, and sample when that integer is "
            "less than rate * 2**64. Return a bool. Raise ValueError if rate "
            "is outside [0, 1] or trace_id is not a string. A rate of 0 "
            "samples nothing and a rate of 1 samples everything."
        ),
        validator=LOAD_CANDIDATE + require("should_sample") + r'''
import hashlib

def expected(trace_id, rate):
    digest = hashlib.sha256(trace_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < rate * 2 ** 64

ids = [f"trace-{index:06d}" for index in range(20000)]

# --- the decision is the specified function of the id, not any hash -------
# Python salts str.__hash__ per process, so an implementation built on the
# built-in hash agrees with itself here and disagrees with the next service.
for trace_id in ids[:400]:
    for rate in (0.1, 0.25, 0.5, 0.75):
        got = should_sample(trace_id, rate)
        assert isinstance(got, bool), "should_sample must return a bool"
        assert got == expected(trace_id, rate), (
            f"{trace_id!r} at rate {rate}: decision does not match the "
            "specified SHA-256 rule"
        )

# --- the boundaries ------------------------------------------------------
for trace_id in ids[:200]:
    assert should_sample(trace_id, 1.0) is True, "rate 1 must sample all"
    assert should_sample(trace_id, 0.0) is False, "rate 0 must sample none"

# --- raising the rate never unsamples a trace ----------------------------
for trace_id in ids[:500]:
    sampled_at = None
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        decision = should_sample(trace_id, rate)
        if sampled_at is not None and not decision:
            raise AssertionError(
                f"{trace_id!r} sampled at {sampled_at} but not at {rate}"
            )
        if decision and sampled_at is None:
            sampled_at = rate

# --- the rate is actually honoured in aggregate --------------------------
for rate in (0.05, 0.25, 0.5):
    hits = sum(1 for trace_id in ids if should_sample(trace_id, rate))
    observed = hits / len(ids)
    assert abs(observed - rate) <= 0.02, (
        f"rate {rate} sampled {observed:.4f} of 20000 ids"
    )

for bad in (-0.01, 1.01, 2.0):
    try:
        should_sample("trace-000001", bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"rate {bad} was accepted")

try:
    should_sample(b"trace", 0.5)
except ValueError:
    pass
else:
    raise AssertionError("a non-string trace id was accepted")
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python class SlidingWindowCounter(window_seconds, "
            "bucket_count) that counts events over a trailing time window "
            "using fixed-width buckets. Bucket width is "
            "window_seconds / bucket_count and the bucket index of a "
            "timestamp is floor(now / width). Provide record(now, amount=1) "
            "and total(now), where total(now) sums every retained bucket "
            "whose index is greater than floor(now / width) - bucket_count. "
            "Buckets that leave the window must be discarded, so "
            "resident_buckets() never exceeds bucket_count no matter how "
            "long the counter runs. Events may arrive late: a record whose "
            "bucket has already left the window relative to the newest "
            "timestamp the counter has seen is ignored and returns False, "
            "and an accepted record returns True. A late record must never "
            "resurrect an expired bucket. Raise ValueError for a "
            "non-positive window_seconds, a bucket_count below 1, or a "
            "non-positive amount."
        ),
        timeout_seconds=120.0,
        validator=LOAD_CANDIDATE + require("SlidingWindowCounter") + r'''
counter = SlidingWindowCounter(60.0, 6)   # six ten-second buckets

for second in range(60):
    assert counter.record(float(second)) is True, "a fresh record was refused"
assert counter.total(59.0) == 60, "the full window did not count every event"

# At t=65 the current bucket index is 6, so the window keeps indices 1..6 and
# the ten events in bucket 0 have expired.
assert counter.total(65.0) == 50, (
    f"total(65) is {counter.total(65.0)}; the oldest bucket did not expire"
)
assert counter.total(200.0) == 0, "an emptied window still reported events"

# --- amounts, not just occurrences ---------------------------------------
weighted = SlidingWindowCounter(10.0, 5)
weighted.record(0.0, 3)
weighted.record(1.0, 4)
assert weighted.total(1.0) == 7, "record(now, amount) ignored the amount"

# --- a record older than the window must not resurrect a dropped bucket ---
late = SlidingWindowCounter(10.0, 5)
late.record(100.0)
assert late.record(50.0) is False, "a record outside the window was accepted"
assert late.total(100.0) == 1, "a stale record changed the total"

# --- the ceiling: residency is bounded by bucket_count, forever ----------
long_run = SlidingWindowCounter(60.0, 6)
for second in range(200000):
    long_run.record(float(second))
    if second % 5000 == 0:
        assert long_run.resident_buckets() <= 6, (
            f"resident_buckets() reached {long_run.resident_buckets()} at "
            f"t={second}; expired buckets are never released"
        )
assert long_run.resident_buckets() <= 6, "buckets accumulate without bound"
assert long_run.total(199999.0) == 60, "the trailing window lost its count"

for bad in (lambda: SlidingWindowCounter(0.0, 6),
            lambda: SlidingWindowCounter(-1.0, 6),
            lambda: SlidingWindowCounter(60.0, 0),
            lambda: SlidingWindowCounter(60.0, -3)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid construction was accepted")

try:
    counter.record(300.0, 0)
except ValueError:
    pass
else:
    raise AssertionError("a non-positive amount was accepted")
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function aggregate_health(checks) that turns "
            "per-dependency health into one service verdict. Each check is a "
            "mapping with keys name (string), status (one of 'up', "
            "'degraded', 'down') and critical (bool). Return a mapping with "
            "keys status, down and degraded. The overall status is "
            "'unhealthy' when any critical dependency is down; otherwise "
            "'degraded' when any critical dependency is degraded or any "
            "dependency at all is down or degraded; otherwise 'healthy'. The "
            "down and degraded values are the names of the dependencies in "
            "each of those states, sorted alphabetically. Raise ValueError "
            "on an empty check list, an unknown status, a duplicate name, or "
            "a check missing any required key."
        ),
        validator=LOAD_CANDIDATE + require("aggregate_health") + r'''
def check(name, status, critical):
    return {"name": name, "status": status, "critical": critical}

result = aggregate_health([check("db", "up", True),
                           check("cache", "up", False)])
assert result["status"] == "healthy", "all-up did not report healthy"
assert result["down"] == [] and result["degraded"] == []

# A critical dependency down is the whole service down.
result = aggregate_health([check("db", "down", True),
                           check("cache", "up", False)])
assert result["status"] == "unhealthy", "a critical outage was not unhealthy"
assert result["down"] == ["db"]

# A NON-critical dependency down is still not healthy. Reporting healthy here
# is the defect that lets a broken subsystem stay invisible until a user
# reports it.
result = aggregate_health([check("db", "up", True),
                           check("cache", "down", False)])
assert result["status"] == "degraded", (
    f"a non-critical outage reported {result['status']!r}, not 'degraded'"
)
assert result["down"] == ["cache"]

# A critical dependency degraded does not escalate to unhealthy.
result = aggregate_health([check("db", "degraded", True)])
assert result["status"] == "degraded"
assert result["degraded"] == ["db"] and result["down"] == []

# Critical-down outranks everything reported alongside it.
result = aggregate_health([check("queue", "degraded", True),
                           check("db", "down", True),
                           check("cdn", "down", False)])
assert result["status"] == "unhealthy"
assert result["down"] == ["cdn", "db"], "down names are not sorted"
assert result["degraded"] == ["queue"]

# --- stated error behaviour ----------------------------------------------
for bad in ([],
            [check("db", "UP", True)],
            [check("db", "unknown", True)],
            [check("db", "up", True), check("db", "up", False)],
            [{"name": "db", "status": "up"}],
            [{"status": "up", "critical": True}]):
    try:
        aggregate_health(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid input was accepted: {bad!r}")
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python function evaluate_error_budget(objective, "
            "windows) implementing multi-window burn-rate alerting for a "
            "service level objective. objective is the target success ratio, "
            "strictly between 0 and 1. windows maps each of the names '5m', "
            "'1h', '30m' and '6h' to a mapping with integer keys total and "
            "failed. For each window the error rate is failed / total and "
            "the burn rate is that error rate divided by the error budget "
            "(1 - objective). Return a mapping with keys burn_rates (name to "
            "burn rate) and severity. severity is 'page' when the 1h and 5m "
            "burn rates are both at least 14.4; otherwise 'ticket' when the "
            "6h and 30m burn rates are both at least 6; otherwise None. "
            "Requiring the short window to agree is what stops a single old "
            "spike from paging after it has already stopped. Raise "
            "ValueError if objective is outside the open interval (0, 1), if "
            "any of the four windows is missing, if any total is not "
            "positive, or if failed is negative or exceeds total."
        ),
        validator=LOAD_CANDIDATE + require("evaluate_error_budget") + r'''
def windows(**rates):
    # Each keyword is the error rate for that window over 100000 requests.
    return {
        name: {"total": 100000, "failed": int(round(rate * 100000))}
        for name, rate in rates.items()
    }

# Objective 99.9% => an error budget of 0.001, so a 2% error rate burns at 20x.
#
# The burn rate is asserted against a tolerance and the severities are driven
# by rates clear of the thresholds, deliberately. A validator that demanded
# `severity == "page"` from an error rate of exactly 1.44% would fail a
# CORRECT implementation: 0.0144 / 0.001 is 14.399999999999999 in binary
# floating point, just under its own threshold. Pinning a verdict to an
# exactly-representable-looking decimal boundary tests the platform's rounding
# rather than the candidate's alerting rule.
fast = windows(**{"5m": 0.02, "1h": 0.02, "30m": 0.02, "6h": 0.02})
result = evaluate_error_budget(0.999, fast)
assert abs(result["burn_rates"]["1h"] - 20.0) < 1e-9, (
    f"burn rate is {result['burn_rates']['1h']}, not 20.0"
)
assert result["severity"] == "page", "a 20x burn on both windows did not page"

# The slow pair alone is a ticket, not a page.
slow = windows(**{"5m": 0.007, "1h": 0.007, "30m": 0.007, "6h": 0.007})
assert abs(evaluate_error_budget(0.999, slow)["burn_rates"]["6h"] - 7.0) < 1e-9
assert evaluate_error_budget(0.999, slow)["severity"] == "ticket", (
    "a 7x burn on both slow windows did not raise a ticket"
)

# THE SHORT WINDOW IS A VETO. A long window still carrying an old spike must
# not page once the incident has stopped.
recovered = windows(**{"5m": 0.0, "1h": 0.02, "30m": 0.0, "6h": 0.02})
assert evaluate_error_budget(0.999, recovered)["severity"] is None, (
    "an incident that has already stopped still paged"
)

# A short spike with no long-window support is also not an alert.
spike = windows(**{"5m": 0.05, "1h": 0.0001, "30m": 0.05, "6h": 0.0001})
assert evaluate_error_budget(0.999, spike)["severity"] is None

# Page outranks ticket when both conditions hold: `fast` satisfies the ticket
# rule too, and the more urgent verdict is the one that must be returned.
assert evaluate_error_budget(0.999, fast)["burn_rates"]["30m"] >= 6
assert evaluate_error_budget(0.999, fast)["severity"] == "page"

# Healthy service: no alert, and the burn rates are still reported.
healthy = windows(**{"5m": 0.0, "1h": 0.0, "30m": 0.0, "6h": 0.0})
result = evaluate_error_budget(0.999, healthy)
assert result["severity"] is None
assert set(result["burn_rates"]) == {"5m", "1h", "30m", "6h"}
assert all(rate == 0 for rate in result["burn_rates"].values())

# The objective sets the budget: a looser objective burns more slowly.
loose = evaluate_error_budget(0.99, fast)["burn_rates"]["1h"]
assert abs(loose - 2.0) < 1e-9, f"objective 0.99 gave burn rate {loose}"
assert evaluate_error_budget(0.99, fast)["severity"] is None, (
    "a 2x burn against a 99% objective still alerted"
)

for bad_objective in (0.0, 1.0, -0.5, 1.5):
    try:
        evaluate_error_budget(bad_objective, fast)
    except ValueError:
        pass
    else:
        raise AssertionError(f"objective {bad_objective} was accepted")

incomplete = {name: dict(value) for name, value in fast.items()}
incomplete.pop("5m")
bad_total = {name: dict(value) for name, value in fast.items()}
bad_total["1h"] = {"total": 0, "failed": 0}
bad_failed = {name: dict(value) for name, value in fast.items()}
bad_failed["1h"] = {"total": 100, "failed": 101}
negative = {name: dict(value) for name, value in fast.items()}
negative["1h"] = {"total": 100, "failed": -1}
for bad in (incomplete, bad_total, bad_failed, negative):
    try:
        evaluate_error_budget(0.999, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid windows were accepted")
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python class EventThrottle(window_seconds, "
            "max_per_window) that stops a repeating log event from flooding "
            "a sink while never hiding that it repeated. Provide "
            "offer(now, key, message) returning either a mapping to emit or "
            "None when the event is suppressed. Within a window, identified "
            "by floor(now / window_seconds), the first max_per_window offers "
            "for a key are emitted as {'key': key, 'message': message, "
            "'suppressed': 0} and later offers in that window return None "
            "while counting as suppressed. The first emission for a key in a "
            "later window reports the number suppressed since its last "
            "emission in its 'suppressed' value, after which that counter "
            "resets. Timestamps are non-decreasing. Keys with no activity "
            "for a whole window must be forgotten so tracked_keys() stays "
            "bounded by the number of keys active in the current and "
            "previous window. Raise ValueError for a non-positive "
            "window_seconds or a max_per_window below 1."
        ),
        timeout_seconds=120.0,
        validator=LOAD_CANDIDATE + require("EventThrottle") + r'''
throttle = EventThrottle(60.0, 2)

first = throttle.offer(0.0, "disk-full", "disk 91% full")
assert first == {"key": "disk-full", "message": "disk 91% full",
                 "suppressed": 0}, f"first emission was {first!r}"
second = throttle.offer(1.0, "disk-full", "disk 92% full")
assert second is not None and second["suppressed"] == 0
for moment in range(2, 40):
    assert throttle.offer(float(moment), "disk-full", "again") is None, (
        "the throttle emitted beyond max_per_window"
    )

# A separate key has its own allowance.
other = throttle.offer(3.0, "cert-expiry", "cert expires in 3 days")
assert other is not None and other["suppressed"] == 0

# The next window reports what was dropped, then resets the counter.
rolled = throttle.offer(60.0, "disk-full", "disk 95% full")
assert rolled is not None, "the new window did not emit"
assert rolled["suppressed"] == 38, (
    f"the new window reported {rolled['suppressed']} suppressed, not 38"
)
assert rolled["message"] == "disk 95% full", "an emission carried a stale message"
follow = throttle.offer(61.0, "disk-full", "disk 96% full")
assert follow is not None and follow["suppressed"] == 0, (
    "the suppressed counter did not reset after being reported"
)

# A key that never overflowed reports nothing suppressed.
quiet = EventThrottle(60.0, 2)
assert quiet.offer(0.0, "k", "a")["suppressed"] == 0
assert quiet.offer(60.0, "k", "b")["suppressed"] == 0

# --- the ceiling: idle keys are released ---------------------------------
bounded = EventThrottle(60.0, 1)
for index in range(100000):
    bounded.offer(float(index), f"key-{index}", "one-shot")
assert bounded.tracked_keys() <= 200, (
    f"tracked_keys() is {bounded.tracked_keys()} after 100000 distinct keys; "
    "idle keys are never forgotten"
)

# Forgetting keys must not break a key that is still active.
active = EventThrottle(10.0, 1)
for index in range(500):
    active.offer(float(index), "steady", "tick")
    active.offer(float(index), f"noise-{index}", "tick")
emission = active.offer(500.0, "steady", "tick")
assert emission is not None, "an active key was dropped with the idle ones"
assert active.tracked_keys() <= 60, "residency grew with the stream length"

for bad in (lambda: EventThrottle(0.0, 1), lambda: EventThrottle(-1.0, 1),
            lambda: EventThrottle(60.0, 0), lambda: EventThrottle(60.0, -2)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid construction was accepted")
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python class FrequentItems(capacity) that finds the "
            "most frequent items in a stream using at most capacity counters, "
            "following the Misra-Gries algorithm. offer(item) either "
            "increments the item's counter if it is tracked, or starts it at "
            "1 if fewer than capacity items are tracked, or otherwise "
            "decrements every tracked counter by 1 and drops those that reach "
            "0. Provide tracked() returning the number of tracked items, "
            "which must never exceed capacity, and top(k) returning up to k "
            "(item, estimated_count) pairs ordered by descending estimated "
            "count and then ascending item. The algorithm guarantees that "
            "every item whose true frequency exceeds n / (capacity + 1), for "
            "a stream of n items, is still tracked, and that each estimated "
            "count is at most the true count and at least the true count "
            "minus n / (capacity + 1). Raise ValueError if capacity is below "
            "1 or k is negative."
        ),
        timeout_seconds=120.0,
        validator=LOAD_CANDIDATE + require("FrequentItems") + r'''
import collections

# A stream with three planted heavy hitters buried in unique noise.
stream = []
for index in range(60000):
    stream.append("alpha" if index % 3 == 0 else f"noise-{index}")
for index in range(12000):
    stream.append("beta")
for index in range(6000):
    stream.append("gamma")

capacity = 32
counter = FrequentItems(capacity)
for item in stream:
    counter.offer(item)
    assert counter.tracked() <= capacity, (
        f"tracked() reached {counter.tracked()} with capacity {capacity}"
    )

truth = collections.Counter(stream)
total = len(stream)
threshold = total / (capacity + 1)

reported = dict(counter.top(capacity))
for item, frequency in truth.items():
    if frequency > threshold:
        assert item in reported, (
            f"{item!r} occurs {frequency} times, above the {threshold:.0f} "
            "guarantee threshold, but was not tracked"
        )

# Estimates never overstate, and understate by at most the threshold.
for item, estimate in reported.items():
    actual = truth[item]
    assert estimate <= actual, (
        f"{item!r} estimated {estimate} above its true count {actual}"
    )
    assert estimate >= actual - threshold, (
        f"{item!r} estimated {estimate}, below the guaranteed floor "
        f"{actual - threshold:.0f}"
    )

# --- ordering is part of the contract ------------------------------------
ordered = counter.top(3)
assert len(ordered) == 3
assert [name for name, _ in ordered][0] == "alpha", "top(k) is not ordered"
assert all(ordered[index][1] >= ordered[index + 1][1]
           for index in range(len(ordered) - 1)), "top(k) is not descending"

ties = FrequentItems(8)
for item in ("b", "a", "c"):
    ties.offer(item)
assert ties.top(3) == [("a", 1), ("b", 1), ("c", 1)], (
    f"equal counts are not broken by ascending item: {ties.top(3)}"
)

# --- small cases ---------------------------------------------------------
empty = FrequentItems(4)
assert empty.top(3) == [] and empty.tracked() == 0
assert counter.top(0) == []
assert len(counter.top(1000)) == counter.tracked(), (
    "top(k) beyond the tracked set must return what is tracked"
)

single = FrequentItems(1)
for item in ("x", "y", "y", "y"):
    single.offer(item)
assert single.tracked() <= 1, "capacity 1 tracked more than one item"

for bad in (lambda: FrequentItems(0), lambda: FrequentItems(-1),
            lambda: counter.top(-1)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid call was accepted")
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function critical_path(spans) that reduces a "
            "distributed trace to the chain of work that determined its "
            "duration. Each span is a mapping with keys span_id, parent_id "
            "(None for the root), name, start_ms and end_ms. Starting at the "
            "root, repeatedly descend into the child span with the greatest "
            "duration, breaking ties by the smaller start_ms and then by the "
            "lexicographically smaller span_id, until reaching a span with "
            "no children. Return a mapping with key path, the list of span "
            "names along that chain from the root, and key duration_ms, the "
            "root span's duration. Raise ValueError if the spans are empty, "
            "if there is not exactly one root, if a parent_id names no span, "
            "if a span_id is duplicated, if any end_ms is before its "
            "start_ms, if a child is not fully contained in its parent's "
            "interval, or if the parent links form a cycle."
        ),
        validator=LOAD_CANDIDATE + require("critical_path") + r'''
def span(span_id, parent_id, name, start_ms, end_ms):
    return {"span_id": span_id, "parent_id": parent_id, "name": name,
            "start_ms": start_ms, "end_ms": end_ms}

# The slowest child is the one that mattered, not the first or the last.
trace = [
    span("a", None, "handle-request", 0, 100),
    span("b", "a", "auth", 0, 10),
    span("c", "a", "query", 10, 90),
    span("d", "a", "render", 90, 95),
    span("e", "c", "plan", 10, 20),
    span("f", "c", "scan", 20, 88),
]
result = critical_path(trace)
assert result["path"] == ["handle-request", "query", "scan"], (
    f"critical path is {result['path']}"
)
assert result["duration_ms"] == 100, "duration_ms is not the root duration"

# Span order in the input must not change the verdict.
assert critical_path(list(reversed(trace)))["path"] == result["path"], (
    "the answer depends on input ordering"
)

# A single span is its own critical path.
alone = critical_path([span("a", None, "only", 5, 25)])
assert alone == {"path": ["only"], "duration_ms": 20}, f"got {alone!r}"

# Ties break by start_ms, then by span_id.
tied = [
    span("root", None, "root", 0, 100),
    span("z", "root", "late", 50, 70),
    span("y", "root", "early", 10, 30),
]
assert critical_path(tied)["path"] == ["root", "early"], (
    "equal durations were not broken by the smaller start_ms"
)
same_start = [
    span("root", None, "root", 0, 100),
    span("m", "root", "m-span", 10, 30),
    span("b", "root", "b-span", 10, 30),
]
assert critical_path(same_start)["path"] == ["root", "b-span"], (
    "a full tie was not broken by the smaller span_id"
)

# --- malformed traces are rejected, not silently repaired ----------------
cyclic = [
    span("a", "b", "a", 0, 10),
    span("b", "a", "b", 0, 10),
]
orphan = [span("a", None, "a", 0, 10), span("b", "missing", "b", 0, 5)]
duplicate = [span("a", None, "a", 0, 10), span("a", "a", "dup", 0, 5)]
two_roots = [span("a", None, "a", 0, 10), span("b", None, "b", 0, 10)]
inverted = [span("a", None, "a", 10, 0)]
escaping = [span("a", None, "a", 0, 10), span("b", "a", "b", 5, 20)]
starts_early = [span("a", None, "a", 10, 20), span("b", "a", "b", 5, 15)]
for bad in ([], cyclic, orphan, duplicate, two_roots, inverted, escaping,
            starts_early):
    try:
        critical_path(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"a malformed trace was accepted: {bad!r}")
''',
    ),
    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python function redact_record(record, secret_keys) "
            "that scrubs sensitive values out of a structured log record "
            "before it is shipped. `record` is a JSON-shaped value: nested "
            "dicts, lists, and scalars. `secret_keys` is a collection of "
            "key names. Return a NEW structure in which the value of every "
            "dict key matching a secret key -- compared case-insensitively, "
            "because 'Password' and 'password' arrive from different "
            "services -- is replaced by the string '[redacted]', whatever "
            "that value is. A dict or list under a secret key is replaced "
            "whole rather than descended into, since redacting only its "
            "leaves would still ship its structure. Descend through every "
            "other dict and list at any depth. The input must not be "
            "mutated: the caller still needs the real values. A record "
            "containing a reference cycle must raise ValueError rather than "
            "exhaust the stack, because a logging path that crashes the "
            "process is worse than one that drops a record."
        ),
        validator=LOAD_CANDIDATE + require("redact_record") + r'''
import copy

original = {
    "user": "ada",
    "Password": "hunter2",
    "nested": {"api_key": {"primary": "x", "backup": "y"}, "keep": 1},
    "items": [{"token": "t1"}, {"token": "t2"}, {"ok": "v"}],
}
snapshot = copy.deepcopy(original)
secrets = {"password", "api_key", "token"}

out = redact_record(original, secrets)
assert original == snapshot, "the input record was mutated"

assert out["user"] == "ada"
assert out["Password"] == "[redacted]", "key matching is not case-insensitive"
assert out["nested"]["api_key"] == "[redacted]", (
    "a dict under a secret key must be replaced whole, not descended into"
)
assert out["nested"]["keep"] == 1
assert [item.get("token", item.get("ok")) for item in out["items"]] == [
    "[redacted]", "[redacted]", "v"
]

# Depth is not a limit: a secret nested ten levels down is still a secret.
deep = current = {}
for _ in range(10):
    current["child"] = {}
    current = current["child"]
current["token"] = "deep-secret"
assert "deep-secret" not in repr(redact_record(deep, secrets))

# A cycle is reported, not crashed on.
cyclic = {"a": {}}
cyclic["a"]["self"] = cyclic
try:
    redact_record(cyclic, secrets)
except ValueError:
    pass
else:
    raise AssertionError("a reference cycle must raise ValueError")

# A list that repeats the SAME child twice is not a cycle.
shared = {"x": 1}
assert redact_record([shared, shared], secrets) == [{"x": 1}, {"x": 1}]

# Scalars pass straight through.
assert redact_record(5, secrets) == 5
assert redact_record(None, secrets) is None
assert redact_record([1, "a", None], secrets) == [1, "a", None]

# An empty secret set still returns a copy rather than the original object.
data = {"a": [1, 2]}
copied = redact_record(data, set())
assert copied == data and copied["a"] is not data["a"], (
    "nested containers must be copied, not shared with the input"
)
'''),
    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python class BoundedLabelRegistry(max_series) that "
            "accumulates a labelled counter without unbounded cardinality. "
            "observe(labels, value) takes a mapping of string label names "
            "to string values and adds value to that series. series() "
            "returns a dict mapping a canonical series key to its total, "
            "where the canonical key is ','.join(f'{name}={value}') over "
            "the label names in sorted order -- so the same labels supplied "
            "in a different order are one series, not two. Once max_series "
            "distinct series exist, a NEW label combination must not create "
            "another one: add its value to a single overflow series keyed "
            "'__other__' instead. Series that already exist keep "
            "accumulating normally after the cap is reached. This is the "
            "whole point: a label carrying a user id or a request path "
            "would otherwise grow the registry without limit and exhaust "
            "the process, so series() must never return more than "
            "max_series + 1 entries no matter what it is fed. Raise "
            "ValueError when max_series is not a positive integer."
        ),
        validator=LOAD_CANDIDATE + require("BoundedLabelRegistry") + r'''
registry = BoundedLabelRegistry(3)
registry.observe({"route": "/a"}, 1)
registry.observe({"route": "/b"}, 2)
registry.observe({"route": "/a"}, 3)
assert registry.series() == {"route=/a": 4, "route=/b": 2}, registry.series()

registry.observe({"route": "/c"}, 1)
registry.observe({"route": "/d"}, 5)
series = registry.series()
assert "route=/d" not in series, "a new series was created past the cap"
assert series["__other__"] == 5
assert series["route=/c"] == 1

# Existing series keep working after the cap.
registry.observe({"route": "/a"}, 10)
assert registry.series()["route=/a"] == 14
registry.observe({"route": "/e"}, 1)
assert registry.series()["__other__"] == 6

# Label order is not part of the identity.
ordered = BoundedLabelRegistry(5)
ordered.observe({"a": "1", "b": "2"}, 1)
ordered.observe({"b": "2", "a": "1"}, 1)
assert ordered.series() == {"a=1,b=2": 2}, ordered.series()

# The ceiling holds against a deliberately unbounded label.
flood = BoundedLabelRegistry(10)
for index in range(50000):
    flood.observe({"request_id": str(index)}, 1)
result = flood.series()
assert len(result) <= 11, f"{len(result)} series retained past a cap of 10"
assert result["__other__"] == 49990, result["__other__"]
assert sum(result.values()) == 50000, "observations were lost at the cap"

# An empty label set is a legitimate single series.
empty = BoundedLabelRegistry(2)
empty.observe({}, 7)
assert empty.series() == {"": 7}, empty.series()

for bad in (0, -1, 1.5, "3", None):
    try:
        BoundedLabelRegistry(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"max_series {bad!r} must raise ValueError")
'''),
    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement a Python function reservoir_sample(stream, size, "
            "random_below) drawing a uniform sample from a stream of "
            "unknown length in fixed memory. `stream` is any iterable and "
            "must be consumed exactly once, so its length cannot be "
            "measured up front and it must not be materialised into a "
            "list. `random_below(n)` returns an integer in [0, n) and is "
            "injected so the result is reproducible. Use Algorithm R: keep "
            "the first `size` items; then for the item at zero-based index "
            "i >= size, draw j = random_below(i + 1) and, when j < size, "
            "replace reservoir slot j with that item. Return the reservoir "
            "as a list. A stream shorter than `size` returns all of its "
            "items in arrival order. Raise ValueError when size is not a "
            "positive integer. Draw exactly once per item past the first "
            "`size` and never otherwise -- a caller reasoning about "
            "reproducibility or entropy cost depends on that count."
        ),
        validator=LOAD_CANDIDATE + require("reservoir_sample") + r'''
def source_factory(seed):
    state = {"v": seed}
    calls = {"n": 0}
    def source(n):
        calls["n"] += 1
        state["v"] = (state["v"] * 1103515245 + 12345) % (2 ** 31)
        return state["v"] % n
    source.calls = calls
    return source

items = list(range(20))
source = source_factory(7)
out = reservoir_sample(iter(items), 5, source)
assert len(out) == 5, f"reservoir holds {len(out)} items"
assert len(set(out)) == 5, "an item was sampled twice"
assert set(out) <= set(items)
assert source.calls["n"] == 15, (
    f"expected one draw per item past the first 5, got {source.calls['n']}"
)

# Reproducible: the same injected source gives the same sample.
assert reservoir_sample(iter(items), 5, source_factory(7)) == out

# Short streams come back whole and in order.
assert reservoir_sample(iter([1, 2]), 5, source_factory(1)) == [1, 2]
assert reservoir_sample(iter([]), 5, source_factory(1)) == []
short = source_factory(1)
reservoir_sample(iter([1, 2]), 5, short)
assert short.calls["n"] == 0, "a stream shorter than size must draw nothing"

# The stream is consumed exactly once and lazily.
consumed = {"n": 0}
def big():
    for value in range(200000):
        consumed["n"] += 1
        yield value
sample = reservoir_sample(big(), 10, source_factory(3))
assert consumed["n"] == 200000, "the stream was not consumed exactly once"
assert len(sample) == 10 and len(set(sample)) == 10

# Later items must actually be able to enter the reservoir.
replaced = False
for seed in range(1, 60):
    drawn = reservoir_sample(iter(range(200)), 3, source_factory(seed))
    if any(value >= 3 for value in drawn):
        replaced = True
        break
assert replaced, (
    "no seed ever replaced an initial element, so the first `size` items "
    "are pinned and the sample is not uniform"
)

for bad in (0, -1, 1.5, "3", None):
    try:
        reservoir_sample(iter([1]), bad, source_factory(1))
    except ValueError:
        pass
    else:
        raise AssertionError(f"size {bad!r} must raise ValueError")
'''),
    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement a Python class DecayingCounter(half_life_seconds, "
            "clock) tracking a recent-activity total that fades rather than "
            "accumulating forever. `clock` is a zero-argument callable "
            "returning a float number of seconds, injected so the class is "
            "testable without waiting. add(amount) records an amount at the "
            "current clock reading. value() returns the total of everything "
            "added, each contribution multiplied by 0.5 ** (elapsed / "
            "half_life_seconds) where elapsed is the time since it was "
            "added. Keep O(1) state: the decay is multiplicative, so a "
            "single running value and the timestamp it was last decayed to "
            "are sufficient, and retaining one entry per add is exactly the "
            "unbounded-memory defect this class exists to avoid. A long "
            "idle gap must be handled by one exponentiation, not by "
            "stepping through it. Raise ValueError when half_life_seconds "
            "is not a positive number."
        ),
        validator=LOAD_CANDIDATE + require("DecayingCounter") + r'''
now = {"t": 0.0}
clock = lambda: now["t"]

counter = DecayingCounter(10.0, clock)
counter.add(100)
assert abs(counter.value() - 100.0) < 1e-9, counter.value()

now["t"] = 10.0
assert abs(counter.value() - 50.0) < 1e-9, counter.value()
now["t"] = 20.0
assert abs(counter.value() - 25.0) < 1e-9, counter.value()

# A later add composes with what has already decayed.
counter.add(25)
assert abs(counter.value() - 50.0) < 1e-9, counter.value()
now["t"] = 30.0
assert abs(counter.value() - 25.0) < 1e-9, counter.value()

# value() must not itself advance the decay: reading twice is the same.
assert abs(counter.value() - counter.value()) < 1e-12
now["t"] = 30.0
assert abs(counter.value() - 25.0) < 1e-9, "value() mutated the state"

# Fractional half-lives.
now["t"] = 35.0
assert abs(counter.value() - 25.0 * (0.5 ** 0.5)) < 1e-9, counter.value()

# A very long gap is one exponentiation, and must not hang or overflow.
now["t"] = 1e9
assert counter.value() >= 0.0
assert counter.value() < 1e-9

# O(1) state under a large number of adds.
fast = DecayingCounter(1.0, clock)
now["t"] = 0.0
for index in range(200000):
    now["t"] = index * 0.001
    fast.add(1)
retained = [
    len(item) for item in vars(fast).values()
    if isinstance(item, (list, dict, set, tuple, bytearray))
]
assert all(size <= 8 for size in retained), (
    f"per-add state was retained: container sizes {retained}"
)
assert fast.value() > 0.0

# Zero and negative amounts are legitimate corrections.
signed = DecayingCounter(10.0, clock)
now["t"] = 0.0
signed.add(10)
signed.add(-4)
assert abs(signed.value() - 6.0) < 1e-9, signed.value()

for bad in (0, -1, "10", None):
    try:
        DecayingCounter(bad, clock)
    except ValueError:
        pass
    else:
        raise AssertionError(f"half_life_seconds {bad!r} must raise ValueError")
'''),
    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement a Python function clip_message(text, budget_bytes) "
            "that fits a log message into a byte budget. Log pipelines "
            "reject or silently truncate records over a size limit, and "
            "that limit is counted in BYTES of UTF-8, not characters. "
            "Return text unchanged when its UTF-8 encoding is at most "
            "budget_bytes. Otherwise return the longest prefix of text such "
            "that the prefix plus the three-byte marker '...' still fits "
            "within budget_bytes, followed by that marker. Never split a "
            "multi-byte character: the result must always be valid UTF-8 "
            "that round-trips, since a truncated code point turns into a "
            "replacement character or an outright decode error downstream. "
            "Raise ValueError when budget_bytes is not an integer of at "
            "least 3, because a budget smaller than the marker cannot be "
            "satisfied."
        ),
        validator=LOAD_CANDIDATE + require("clip_message") + r'''
assert clip_message("hello", 10) == "hello"
assert clip_message("hello", 5) == "hello", "an exact fit must not be clipped"
assert clip_message("", 3) == ""

clipped = clip_message("hello world", 8)
assert clipped == "hello..." , repr(clipped)
assert len(clipped.encode("utf-8")) == 8

# Two-byte characters: the budget is bytes, so fewer characters survive.
text = "é" * 10
assert len(text.encode("utf-8")) == 20
out = clip_message(text, 9)
assert len(out.encode("utf-8")) <= 9, f"{len(out.encode('utf-8'))} bytes"
assert out == "ééé...", repr(out)

# Four-byte characters must not be cut in half.
emoji = "\U0001F600" * 5
assert len(emoji.encode("utf-8")) == 20
out = clip_message(emoji, 10)
assert len(out.encode("utf-8")) <= 10
assert out == "\U0001F600...", repr(out)

# Everything returned must round-trip as UTF-8 with no replacement chars.
for budget in range(3, 21):
    for candidate in (text, emoji, "hello world", "aéb\U0001F600c"):
        result = clip_message(candidate, budget)
        raw = result.encode("utf-8")
        assert len(raw) <= budget, (
            f"{candidate!r} at budget {budget} produced {len(raw)} bytes"
        )
        assert raw.decode("utf-8") == result
        assert "�" not in result

# A budget of exactly the marker leaves room for no text at all.
assert clip_message("abcdef", 3) == "..."

# A single character too wide for the remaining room is dropped whole.
# At budget 5 the marker leaves 2 bytes, and the leading emoji needs 4.
assert clip_message("\U0001F600ab", 5) == "..."

for bad in (2, 0, -1, 3.5, "3", None):
    try:
        clip_message("hello world", bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"budget_bytes {bad!r} must raise ValueError")
'''),
    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement a Python function counter_rate(samples) computing "
            "the per-second rate of a monotonically increasing counter that "
            "may reset. `samples` is a list of (timestamp, value) pairs, "
            "timestamps being floats in seconds and values non-negative "
            "numbers. A process restart sets the counter back to zero, so a "
            "sample LOWER than its predecessor is a reset, not a negative "
            "delta: count the new value in full as the increase since the "
            "reset, because the counter climbed from zero to it. Sum every "
            "increase across the samples and divide by the elapsed time "
            "between the first and last timestamp. Raise ValueError when "
            "fewer than two samples are supplied, when the timestamps are "
            "not strictly increasing, when the first and last timestamps "
            "are equal, or when any value is negative. Treating a reset as "
            "a negative delta is the classic version of this bug: it "
            "reports a negative rate, or silently cancels out real traffic."
        ),
        validator=LOAD_CANDIDATE + require("counter_rate") + r'''
assert counter_rate([(0.0, 0.0), (10.0, 100.0)]) == 10.0
assert counter_rate([(0.0, 50.0), (10.0, 150.0)]) == 10.0

# A reset: 100 -> 5 means the counter restarted and climbed to 5.
# Total increase is 100 + 5 = 105 over 20 seconds.
assert counter_rate([(0.0, 0.0), (10.0, 100.0), (20.0, 5.0)]) == 5.25

# Two resets in one window.
value = counter_rate([(0.0, 0.0), (1.0, 10.0), (2.0, 3.0), (3.0, 1.0)])
assert value == (10.0 + 3.0 + 1.0) / 3.0, value

# A flat counter has a rate of zero, not an error.
assert counter_rate([(0.0, 7.0), (5.0, 7.0)]) == 0.0

# A reset to exactly zero contributes nothing but is still not negative.
assert counter_rate([(0.0, 0.0), (10.0, 100.0), (20.0, 0.0)]) == 5.0

# Fractional timestamps.
assert abs(counter_rate([(0.5, 0.0), (1.0, 4.0)]) - 8.0) < 1e-9

# The rate can never come out negative, whatever the reset pattern.
for series in (
    [(0.0, 1000.0), (1.0, 1.0)],
    [(0.0, 5.0), (1.0, 0.0), (2.0, 0.0)],
    [(0.0, 0.0), (1.0, 1e9), (2.0, 1.0)],
):
    assert counter_rate(series) >= 0.0, f"negative rate for {series}"

for bad in (
    [],
    [(0.0, 1.0)],
    [(0.0, 1.0), (0.0, 2.0)],
    [(1.0, 1.0), (0.0, 2.0)],
    [(0.0, 1.0), (1.0, -2.0)],
    [(0.0, -1.0), (1.0, 2.0)],
):
    try:
        counter_rate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"samples {bad!r} must raise ValueError")
'''),
    task(
        f"{FAMILY}-0015", FAMILY,
        prompt=(
            "Implement a Python class LogTail(max_bytes) retaining only the "
            "most recent log lines that fit in a fixed byte budget. "
            "append(line) adds a line; lines() returns the retained lines "
            "oldest first. The budget is the total UTF-8 byte length of the "
            "retained lines, and it is a hard ceiling: after any append, "
            "that total must be at most max_bytes. Make room by dropping "
            "whole lines from the FRONT, never by truncating one, because a "
            "half line in a crash dump is worse than an absent one. A "
            "single line whose own encoding exceeds max_bytes cannot be "
            "stored under that rule, so raise ValueError and leave the "
            "buffer untouched. Raise ValueError when max_bytes is not a "
            "positive integer. Count bytes rather than characters: a buffer "
            "sized in characters overruns its real limit by a factor of "
            "four on non-ASCII text, which is exactly when a crash dump "
            "matters."
        ),
        validator=LOAD_CANDIDATE + require("LogTail") + r'''
tail = LogTail(20)
for line in ("aaaa", "bbbb", "cccc", "dddd", "eeee"):
    tail.append(line)
assert tail.lines() == ["aaaa", "bbbb", "cccc", "dddd", "eeee"]

tail.append("ffff")
assert tail.lines() == ["bbbb", "cccc", "dddd", "eeee", "ffff"], tail.lines()

# Dropping is by whole lines, and enough of them to fit.
tail.append("gggggggggg")
assert tail.lines() == ["eeee", "ffff", "gggggggggg"], tail.lines()
assert sum(len(item.encode("utf-8")) for item in tail.lines()) <= 20

# The budget counts bytes, not characters.
wide = LogTail(10)
wide.append("ééé")
wide.append("ééé")
assert wide.lines() == ["ééé"], wide.lines()
assert sum(len(item.encode("utf-8")) for item in wide.lines()) == 6

# A line that cannot ever fit is refused, and changes nothing.
before = wide.lines()
try:
    wide.append("x" * 11)
except ValueError:
    pass
else:
    raise AssertionError("a line larger than max_bytes must raise ValueError")
assert wide.lines() == before, "a refused line disturbed the buffer"

# Exactly max_bytes is allowed.
exact = LogTail(4)
exact.append("abcd")
assert exact.lines() == ["abcd"]
exact.append("efgh")
assert exact.lines() == ["efgh"]

# Empty lines are real lines and cost nothing.
empties = LogTail(4)
for _ in range(100):
    empties.append("")
assert empties.lines()[-1] == ""

# The ceiling holds over a long stream, keeping the newest lines.
stream = LogTail(1000)
for index in range(100000):
    stream.append(f"line-{index}")
retained = stream.lines()
assert sum(len(item.encode("utf-8")) for item in retained) <= 1000
assert retained[-1] == "line-99999"
assert retained == sorted(retained, key=lambda item: int(item.split("-")[1]))

for bad in (0, -1, 1.5, "10", None):
    try:
        LogTail(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"max_bytes {bad!r} must raise ValueError")
'''),
    task(
        f"{FAMILY}-0016", FAMILY,
        prompt=(
            "Implement a Python class BatchFlusher(max_items, "
            "max_age_seconds, clock, sink) that batches telemetry without "
            "losing or stalling it. `clock` is a zero-argument callable "
            "returning seconds and `sink` is called with a list of items. "
            "add(item) buffers the item and, when the buffer reaches "
            "max_items, immediately flushes. tick() flushes when the OLDEST "
            "buffered item has been waiting at least max_age_seconds, which "
            "is what stops a low-traffic buffer from sitting on data "
            "forever. flush() forces a flush. Every flush passes the "
            "buffered items to sink as one list in arrival order and then "
            "empties the buffer. Never call sink with an empty list: a "
            "downstream that bills per request should not be charged for "
            "nothing. The age is measured from when the oldest item still "
            "in the buffer was added, not from the last flush. Raise "
            "ValueError when max_items is not a positive integer or "
            "max_age_seconds is not a positive number."
        ),
        validator=LOAD_CANDIDATE + require("BatchFlusher") + r'''
now = {"t": 0.0}
clock = lambda: now["t"]
flushed = []

flusher = BatchFlusher(3, 10.0, clock, flushed.append)

flusher.add("a")
flusher.add("b")
assert flushed == [], "flushed before reaching max_items or max_age"

flusher.add("c")
assert flushed == [["a", "b", "c"]], flushed
assert flusher.flush() is None or True
del flushed[:]

# An empty buffer never reaches the sink.
flusher.tick()
flusher.flush()
now["t"] = 1000.0
flusher.tick()
assert flushed == [], "the sink was called with an empty batch"

# Age is measured from the oldest buffered item.
now["t"] = 1000.0
flusher.add("d")
now["t"] = 1005.0
flusher.add("e")
flusher.tick()
assert flushed == [], "flushed before the oldest item was old enough"

now["t"] = 1010.0
flusher.tick()
assert flushed == [["d", "e"]], flushed
del flushed[:]

# After a flush the age restarts from the next item added, not from now.
now["t"] = 1011.0
flusher.add("f")
now["t"] = 1020.0
flusher.tick()
assert flushed == [], "flushed 9 seconds into a 10 second window"
now["t"] = 1021.0
flusher.tick()
assert flushed == [["f"]], flushed
del flushed[:]

# Order is arrival order across a batch boundary.
for item in ("g", "h", "i", "j"):
    flusher.add(item)
assert flushed == [["g", "h", "i", "j"][:3]], flushed
flusher.flush()
assert flushed == [["g", "h", "i"], ["j"]], flushed
del flushed[:]

# Exactly at the threshold counts as old enough.
now["t"] = 2000.0
flusher.add("k")
now["t"] = 2010.0
flusher.tick()
assert flushed == [["k"]], flushed
del flushed[:]

# Nothing is ever lost: every item added comes out exactly once.
sent = []
counting = BatchFlusher(7, 5.0, clock, sent.extend)
for index in range(1000):
    now["t"] = 3000.0 + index * 0.5
    counting.add(index)
    counting.tick()
counting.flush()
assert sent == list(range(1000)), (
    f"{len(sent)} items came out of 1000 added"
)

for bad_items, bad_age in ((0, 1.0), (-1, 1.0), ("3", 1.0), (3, 0), (3, -1)):
    try:
        BatchFlusher(bad_items, bad_age, clock, flushed.append)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"BatchFlusher({bad_items!r}, {bad_age!r}) must raise ValueError"
        )
'''),
]
