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
]
