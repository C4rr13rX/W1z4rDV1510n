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


def test_retiring_a_stale_guard_still_takes_the_checkpoint_barrier(tmp_path):
    """The replacement guard must carry a checkpoint proof.

    Retiring the stale guard inside ensure_last_good_guard() left the live
    path on its reuse branch, which skips the barrier and publishes
    checkpoint_proof={}. The restart path then refuses that guard with
    "last-good guard has no checkpoint topology proof", which is exactly how
    the first attempt at this fix stalled the supervisor.
    """
    import argparse

    runtime = build(tmp_path, guard_row=421477, durable_row=421477)
    calls = []

    def fake_post(endpoint, path, payload, timeout=0.0):
        calls.append(path)
        return {"ok": True, "tick": 99, "storage": "wbrain"}

    def fake_json(endpoint, path, timeout=0.0):
        calls.append(path)
        return {"pool_count": 13, "total_neurons": 7}

    original_post = sup.endpoint_post_json
    original_json = sup.endpoint_json
    sup.endpoint_post_json = fake_post
    sup.endpoint_json = fake_json
    try:
        sup.ensure_live_last_good_guard(
            argparse.Namespace(endpoint="http://127.0.0.1:18095"),
            runtime, next_phase(), 0,
        )
    finally:
        sup.endpoint_post_json = original_post
        sup.endpoint_json = original_json

    assert "/brain/checkpoint" in calls, "the barrier must run"
    metadata = sup.read_json(runtime / "brain" / "brain.last-good.json")
    assert metadata["phase"] == NEXT
    assert metadata["checkpoint_proof"], "guard must carry a checkpoint proof"
    assert metadata["checkpoint_proof"]["checkpoint"]["ok"] is True


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


def test_recovery_without_a_quarantine_restarts_instead_of_failing(tmp_path,
                                                                   monkeypatch):
    """No quarantine means nothing to roll back, not an error.

    perform_automatic_recovery() called restore_canary_quarantine()
    unconditionally, which raises "no continuous-canary quarantine exists"
    when the marker is absent. The phase then sat in
    automatic_quarantine_recovery_failed with no brain server running at all.
    """
    import argparse

    runtime = tmp_path / "runtime"
    (runtime / "brain").mkdir(parents=True)
    status = runtime / "status.json"
    calls = []

    monkeypatch.setattr(sup, "stop_runtime_node",
                        lambda *a, **k: calls.append("stop"))
    monkeypatch.setattr(sup, "start_runtime_node",
                        lambda *a, **k: calls.append("start"))

    def explode(*a, **k):
        raise AssertionError("must not try to restore an absent quarantine")

    monkeypatch.setattr(sup, "restore_canary_quarantine", explode)

    args = argparse.Namespace(
        auto_quarantine_recovery=True,
        endpoint="http://127.0.0.1:18095",
        node_bin=tmp_path / "brain_server",
    )

    assert sup.perform_automatic_recovery(args, next_phase(), runtime, status)
    assert calls == ["stop", "start"], "recovery must leave a live brain"


def test_a_quarantine_with_no_guard_is_retired(tmp_path):
    """A rollback target that does not exist cannot protect anything.

    Observed three times on 2026-08-20/21: a quarantine formed with
    last_good={} because no guard file existed, restore_canary_quarantine()
    raised "quarantine lacks valid last-good phase/row", and the supervisor
    sat in deferred_replay_recovery_failed relaunching the brain every few
    seconds without training. Each time it needed the marker deleted by hand.
    """
    runtime = tmp_path / "runtime"
    (runtime / "brain").mkdir(parents=True)
    marker = sup.canary_quarantine_path(runtime)
    marker.write_text(json.dumps({
        "state": "deferred_replay_failed",
        "phase": "csn-python-para5",
        "candidate_row": 2028816,
        "last_good": {},
    }), encoding="utf-8")

    result = sup.restore_canary_quarantine(runtime, finalize=False)

    assert result["retired_unrestorable"] is True
    assert not marker.exists(), "the unsatisfiable marker must be cleared"
    sup.assert_training_not_quarantined(runtime)


def test_a_quarantine_with_a_real_guard_is_never_retired(tmp_path):
    """The safety property: a present guard keeps the normal restore path."""
    runtime = tmp_path / "runtime"
    (runtime / "brain").mkdir(parents=True)
    guard = runtime / "brain" / "brain.last-good.wbrain"
    guard.write_bytes(b"guard")
    (runtime / "brain" / "brain.last-good.json").write_text(json.dumps({
        "phase": "csn-python-para5", "row": 2028816, "guard": str(guard),
    }), encoding="utf-8")

    assert not sup.unrestorable_quarantine(
        runtime, {"phase": "csn-python-para5"}, {"guard": str(guard)}
    )
    # Metadata alone is also enough to keep the guarded path.
    assert not sup.unrestorable_quarantine(
        runtime, {"phase": "csn-python-para5"}, {}
    )


def test_a_retired_quarantine_skips_topology_verification():
    """A retirement rolled nothing back, so there is no barrier to verify.

    Measured 2026-08-21: verify_restored_topology() demanded a checkpoint
    proof from the retirement result, which has none, so every retirement was
    immediately followed by "last-good guard has no checkpoint topology
    proof" -- 26 retirements in five minutes, the exact loop the retirement
    exists to break.
    """
    sup.verify_restored_topology(
        {"phase": "csn-python-para5", "row": 2028816,
         "retired_unrestorable": True},
        {"tick": 1, "pool_count": 13},
    )


def test_a_real_rollback_still_requires_its_barrier():
    """The safety property: only a retirement skips verification."""
    try:
        sup.verify_restored_topology({"phase": "x", "row": 1}, {"tick": 1})
    except RuntimeError as exc:
        assert "checkpoint topology proof" in str(exc)
    else:
        raise AssertionError("a real rollback must still prove its barrier")


def test_a_read_only_sentinel_is_not_blamed_for_the_clock():
    """`tick` is a clock, not learned structure.

    Measured 2026-08-21 with no sentinel running at all: tick moved 1,264 in
    25 s from the concurrent deferred replay while total_neurons,
    total_concepts, total_binding and total_terminals held exactly steady.
    Including tick in topology_delta made the read-only sentinel assertions
    fire on that clock and reject the interval with "read-only sentinel
    mutated topology".
    """
    before = {"tick": 1_300_928, "pool_count": 13, "total_neurons": 1_419_984,
              "total_concepts": 1_419_114, "total_binding": 1_264_324,
              "total_terminals": 116_302_548}
    after = dict(before, tick=1_301_568)

    assert not any(sup.topology_delta(before, after).values())


def test_real_structural_growth_is_still_detected():
    """The safety property: actual mutation must still be caught."""
    before = {"tick": 1, "pool_count": 13, "total_neurons": 10,
              "total_concepts": 9, "total_binding": 8, "total_terminals": 700}
    for field in ("pool_count", "total_neurons", "total_concepts",
                  "total_binding", "total_terminals"):
        after = dict(before)
        after[field] = before[field] + 1
        assert any(sup.topology_delta(before, after).values()), field
