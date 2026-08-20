"""Regression tests for a canary failure whose window is already deferred.

Context (2026-08-20): the AWS curriculum host logged 12,699 tracebacks of
"failed canary contains no newly trained, non-deferred rows"
(10,114 at phase=jupyter-scientific-full boundary=131072, 2,585 at
boundary=690175). record_deferred_failure() assumed a failing canary always
contains at least one row that is not already an unresolved deferred
obligation. When suspect_intervals() subtracted the existing coverage and
returned nothing, it raised instead of retiring the interval, so the span was
retried forever and the queue never drained.

The midphase path already handled this correctly via
advance_guard_across_deferred_block(); the continuous-canary paths did not.
The fix routes the empty-ranges case to that same handler from inside
record_deferred_failure, so all five call sites are covered at the point of
consumption rather than by per-caller guards.

Two properties matter and are easy to break together:
  * a fully-deferred failure must NOT write a quarantine, because
    assert_training_not_quarantined() fails closed on it; and
  * a window that really does contain fresh rows must still defer AND
    quarantine exactly as before.
"""
import json
import time

from scripts import programming_curriculum_supervisor as sup

PHASE = "jupyter-scientific-full"
SCRIPT_ID = "domain_scientific_python_001"


def build_runtime(tmp_path, guard_row, deferred_end):
    """A runtime whose guard_row..deferred_end span is wholly deferred."""
    runtime = tmp_path / "runtime"
    (runtime / "brain").mkdir(parents=True)
    guard = runtime / "brain" / "brain.last-good.wbrain"
    guard.write_bytes(b"guard-bytes")
    (runtime / "brain" / "brain.last-good.json").write_text(json.dumps({
        "phase": PHASE,
        "row": guard_row,
        "created_unix": time.time() - 3600,
        "state_identity": "deadbeef",
        "guard": str(guard),
    }), encoding="utf-8")
    (runtime / "curriculum-deferred-intervals.jsonl").write_text(json.dumps({
        "interval_id": f"{PHASE}:{guard_row}:{deferred_end}",
        "phase": PHASE,
        "start_row": guard_row,
        "end_row": deferred_end,
        "status": "deferred",
        "base_row": guard_row,
    }) + "\n", encoding="utf-8")
    return runtime


def phase_for(rows=690_175):
    return sup.Phase(PHASE, SCRIPT_ID, tmp_corpus(), rows)


def tmp_corpus():
    from pathlib import Path
    return Path("/nonexistent-corpus.jsonl")


def test_fully_deferred_window_is_absorbed_not_raised(tmp_path):
    runtime = build_runtime(tmp_path, 0, 131072)
    assert sup.suspect_intervals(runtime, PHASE, 131072, 0, 0.0, True) == []

    result = sup.record_deferred_failure(
        runtime, phase_for(), 131072, 131072,
        "admission gate failed", "continuous_canary_failed",
    )

    assert result["kind"] == "fully_deferred_failure_absorbed"
    assert result["suspect_intervals"] == []
    assert result["neural_state_changed"] is False


def test_absorbed_failure_leaves_training_open(tmp_path):
    """A quarantine here would fail training closed against known evidence."""
    runtime = build_runtime(tmp_path, 0, 131072)
    sup.record_deferred_failure(
        runtime, phase_for(), 131072, 131072,
        "admission gate failed", "continuous_canary_failed",
    )
    assert not json.loads(
        (sup.canary_quarantine_path(runtime).read_text(encoding="utf-8")
         if sup.canary_quarantine_path(runtime).exists() else "{}")
    )
    sup.assert_training_not_quarantined(runtime)


def test_absorbed_failure_still_records_evidence(tmp_path):
    runtime = build_runtime(tmp_path, 0, 131072)
    sup.record_deferred_failure(
        runtime, phase_for(), 131072, 131072,
        "admission gate failed", "continuous_canary_failed",
    )
    kinds = [
        json.loads(line)["kind"]
        for line in (runtime / "curriculum-health.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert "fully_deferred_failure_absorbed" in kinds


def test_absorbed_failure_advances_the_logical_cursor(tmp_path):
    """Without this the next supervisor re-benchmarks the same empty sample."""
    runtime = build_runtime(tmp_path, 0, 131072)
    result = sup.record_deferred_failure(
        runtime, phase_for(), 131072, 131072,
        "admission gate failed", "continuous_canary_failed",
    )
    assert result["guard_advanced"] is True
    assert sup.read_json(
        runtime / "brain" / "brain.last-good.json"
    )["row"] == 131072


def test_a_window_with_fresh_rows_still_defers_and_quarantines(tmp_path):
    """The safety property: only the no-fresh-rows case changed."""
    runtime = build_runtime(tmp_path, 0, 65536)
    assert sup.suspect_intervals(
        runtime, PHASE, 131072, 0, 0.0, True
    ) == [(65536, 131072)]

    result = sup.record_deferred_failure(
        runtime, phase_for(), 131072, 131072,
        "admission gate failed", "continuous_canary_failed",
    )

    # The normal path spreads events[0], which carries no "kind" key at all.
    assert result.get("kind") != "fully_deferred_failure_absorbed"
    assert result["interval_id"] == f"{PHASE}:65536:131072"
    assert result["suspect_intervals"] == [
        {"start_row": 65536, "end_row": 131072}
    ]
    assert sup.canary_quarantine_path(runtime).exists()
