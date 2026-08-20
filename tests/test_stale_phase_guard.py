"""Regression tests for a last-good guard left behind by a finished phase.

Context (2026-08-20): adding the webstack-units phase made the supervisor
attempt forward progress for the first time in weeks, and it immediately
died with "unresolved last-good snapshot guard exists" naming
phase=csn-python-full row=421477 — a guard from a phase that had already
completed (durable 421477/421477, forward-harvested).

accept_last_good_guard() retires a guard at the break that ends a phase's
loop, but csn-python-full finished by forward harvest, a path that exits
without reaching that call, so the guard outlived its phase and blocked
every later phase behind it.

ensure_last_good_guard() now retires a guard whose owning phase provably
finished. The safety property is that "provably finished" is judged from
the owning phase's OWN durable progress, so an interrupted phase — the case
the guard exists to protect — still refuses.
"""
import json
import time

from scripts import programming_curriculum_supervisor as sup

OWNER = "csn-python-full"
NEXT = "webstack-units"


def build(tmp_path, guard_row, durable_row, owner=OWNER, write_progress=True):
    runtime = tmp_path / "runtime"
    (runtime / "brain").mkdir(parents=True)
    (runtime / "brain" / "brain.wbrain").write_bytes(b"live")
    (runtime / "brain" / "brain.last-good.wbrain").write_bytes(b"guard")
    (runtime / "brain" / "brain.last-good.json").write_text(json.dumps({
        "phase": owner,
        "row": guard_row,
        "created_unix": time.time() - 3600,
    }), encoding="utf-8")
    if write_progress:
        (runtime / f"{owner}.progress.json").write_text(json.dumps({
            "corpus": f"/srv/wizard/corpora/{owner}.jsonl",
            "durable_next_row": durable_row,
            "ram_next_row": durable_row,
        }), encoding="utf-8")
    return runtime


def next_phase(rows=60):
    from pathlib import Path
    return sup.Phase(NEXT, "programming_webstack_001",
                     Path("/srv/wizard/corpora/webstack_units.jsonl"), rows)


def test_guard_from_a_completed_phase_is_retired(tmp_path):
    runtime = build(tmp_path, guard_row=421477, durable_row=421477)

    guard = sup.ensure_last_good_guard(runtime, next_phase(), 0)

    assert guard.name == "brain.last-good.wbrain"
    # Retired, then recreated for the new phase.
    assert sup.read_json(
        runtime / "brain" / "brain.last-good.json"
    )["phase"] == NEXT


def test_guard_from_an_unfinished_phase_still_refuses(tmp_path):
    """The guard exists precisely to protect an interrupted phase."""
    runtime = build(tmp_path, guard_row=421477, durable_row=300000)

    try:
        sup.ensure_last_good_guard(runtime, next_phase(), 0)
    except RuntimeError as exc:
        assert "unresolved last-good snapshot guard" in str(exc)
    else:
        raise AssertionError("an unfinished phase must keep its guard")


def test_guard_without_progress_evidence_still_refuses(tmp_path):
    """No progress file means no proof of completion, so fail closed."""
    runtime = build(tmp_path, guard_row=421477, durable_row=0,
                    write_progress=False)

    try:
        sup.ensure_last_good_guard(runtime, next_phase(), 0)
    except RuntimeError as exc:
        assert "unresolved last-good snapshot guard" in str(exc)
    else:
        raise AssertionError("missing evidence must not retire a guard")


def test_same_phase_guard_is_reused_not_retired(tmp_path):
    """Unchanged behaviour: a guard for the running phase is returned as-is."""
    runtime = build(tmp_path, guard_row=32, durable_row=32, owner=NEXT)

    guard = sup.ensure_last_good_guard(runtime, next_phase(), 64)

    assert guard.exists()
    metadata = sup.read_json(runtime / "brain" / "brain.last-good.json")
    assert metadata["phase"] == NEXT
    assert metadata["row"] == 32


def test_same_phase_guard_ahead_of_the_row_still_refuses(tmp_path):
    """Unchanged behaviour: a guard ahead of the requested row is a fault."""
    runtime = build(tmp_path, guard_row=128, durable_row=128, owner=NEXT)

    try:
        sup.ensure_last_good_guard(runtime, next_phase(), 64)
    except RuntimeError as exc:
        assert "unresolved last-good snapshot guard" in str(exc)
    else:
        raise AssertionError("a guard ahead of the row must still refuse")


def test_completed_phase_owns_guard_requires_an_integer_row(tmp_path):
    runtime = build(tmp_path, guard_row=421477, durable_row=421477)

    assert sup.completed_phase_owns_guard(
        runtime, {"phase": OWNER, "row": 421477}
    )
    assert not sup.completed_phase_owns_guard(
        runtime, {"phase": OWNER, "row": None}
    )
    assert not sup.completed_phase_owns_guard(runtime, {"row": 421477})
    assert not sup.completed_phase_owns_guard(runtime, {})
