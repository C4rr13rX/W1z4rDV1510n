from __future__ import annotations

from scripts.aws.watch_programming_brain import (
    Decision,
    classify_probe,
    completion_marker_valid,
    cooldown_elapsed,
    observe,
)


def probe(state: str, *, supervisors: int = 1, wrappers: int = 1,
          age: float = 1.0, host: str = "running") -> dict:
    return {
        "host_state": host,
        "status": {"state": state, "phase": "corpus", "interval_id": "i1"},
        "supervisor_count": supervisors,
        "wrapper_count": wrappers,
        "status_age_seconds": age,
    }


def test_live_automation_does_not_wake_codex_for_failures_it_still_owns() -> None:
    decision = classify_probe(
        probe("continuous_canary_failed"), stall_seconds=1800
    )
    assert decision.kind == "healthy"


def test_completed_automation_wakes_codex_for_the_next_stage() -> None:
    decision = classify_probe(
        probe("deferred_replay_complete", supervisors=0, wrappers=0),
        stall_seconds=1800,
    )
    assert decision.kind == "milestone"
    assert decision.fingerprint


def test_quarantine_handoff_wakes_codex_even_while_replay_is_live() -> None:
    ready = probe("deferred_intervals_pending")
    ready["runtime"] = "/runtime"
    replay = probe("deferred_replay_training")
    replay.update({"runtime": "/runtime", "service_stage": "replay"})
    first = classify_probe(ready, stall_seconds=1800)
    second = classify_probe(replay, stall_seconds=1800)
    assert first.kind == "quarantine_ready"
    assert second.kind == "quarantine_ready"
    assert first.fingerprint == second.fingerprint


def test_stopped_control_or_host_requires_repair() -> None:
    stopped = classify_probe(
        probe("running", supervisors=0, wrappers=0), stall_seconds=1800
    )
    host = classify_probe(probe("running", host="stopped"), stall_seconds=1800)
    assert stopped.kind == "fix_required"
    assert host.kind == "fix_required"


def test_stale_live_control_requires_repair() -> None:
    decision = classify_probe(
        probe("running", age=1801), stall_seconds=1800
    )
    assert decision.kind == "fix_required"


def test_action_requires_stability_and_respects_retry_cooldown() -> None:
    decision = Decision("fix_required", "broken", "fix:one")
    state, trigger = observe({}, decision, 2)
    assert not trigger
    state, trigger = observe(state, decision, 2)
    assert trigger
    state.update({
        "last_invoked_fingerprint": decision.fingerprint,
        "last_invoked_unix": 100.0,
    })
    assert not cooldown_elapsed(state, decision, now=199.0, retry_cooldown=100.0)
    assert cooldown_elapsed(state, decision, now=200.0, retry_cooldown=100.0)


def test_healthy_observation_clears_pending_event() -> None:
    state, trigger = observe(
        {"pending_fingerprint": "fix:one", "pending_count": 1},
        Decision("healthy", "running"),
        2,
    )
    assert not trigger
    assert state["pending_fingerprint"] == ""
    assert state["pending_count"] == 0
    assert state["last_invoked_fingerprint"] == ""


def test_completion_marker_requires_every_authoritative_gate() -> None:
    marker = {
        "passed": True,
        "production_brain": {"passed": True, "report": "brain.json"},
        "obstacle_course": {
            "passed": 1000, "total": 1000, "report": "course.json",
        },
        "brain_selectors": {"passed": True, "report": "selectors.json"},
        "capstone": {
            "passed": True, "independently_verified": True,
            "report": "capstone.json",
        },
    }
    assert completion_marker_valid(marker)
    marker["obstacle_course"]["passed"] = 999
    assert not completion_marker_valid(marker)
