from __future__ import annotations

from pathlib import Path

from scripts.programming_curriculum_supervisor import curriculum_phases
from scripts.aws.watch_programming_brain import (
    Decision,
    classify_probe,
    completion_marker_valid,
    cooldown_elapsed,
    format_claude_event,
    observe,
)


def test_curriculum_plan_has_one_authoritative_logical_row_total() -> None:
    base = curriculum_phases(Path("corpora"))
    seeded = curriculum_phases(Path("corpora"), include_seed=True)
    # 2026-08-20: webstack-units adds 60 logical rows at repeats=4, covering
    # the Django/Vue/three.js gap that left the brain unable to answer eight
    # of ten decomposed web-stack tasks.
    assert sum(phase.rows for phase in base) == 6_738_831
    assert sum(phase.rows for phase in seeded) == 6_748_257
    assert sum(phase.rows * phase.repeats for phase in seeded) == 6_754_332
    assert [phase.name for phase in seeded[-9:]] == [phase.name for phase in base]


def probe(state: str, *, supervisors: int = 1, wrappers: int = 1,
          age: float = 1.0, host: str = "running") -> dict:
    return {
        "host_state": host,
        "status": {"state": state, "phase": "corpus", "interval_id": "i1"},
        "supervisor_count": supervisors,
        "wrapper_count": wrappers,
        "status_age_seconds": age,
    }


def test_live_automation_does_not_wake_the_agent_for_failures_it_still_owns() -> None:
    decision = classify_probe(
        probe("continuous_canary_failed"), stall_seconds=1800
    )
    assert decision.kind == "healthy"


def test_completed_automation_wakes_the_agent_for_the_next_stage() -> None:
    decision = classify_probe(
        probe("deferred_replay_complete", supervisors=0, wrappers=0),
        stall_seconds=1800,
    )
    assert decision.kind == "milestone"
    assert decision.fingerprint


def test_quarantine_handoff_wakes_the_agent_even_while_replay_is_live() -> None:
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


def test_probe_process_matching_is_executable_scoped() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts/aws/watch_programming_brain.py"
    ).read_text(encoding="utf-8")
    assert "process_name.startswith('python')" in source
    assert "process_name == 'bash'" in source


def test_tail_reports_full_curriculum_admission_accounting() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts/aws/watch_programming_brain.py"
    ).read_text(encoding="utf-8")
    for field in (
        "total_rows", "durable_processed_rows", "accepted_rows",
        "deferred_rows", "forward_remaining_rows", "minimum_outstanding_rows",
    ):
        assert field in source


def test_claude_stream_json_events_have_a_human_readable_tail() -> None:
    message = format_claude_event({
        "type": "assistant",
        "message": {"content": [
            {"type": "text", "text": "Fixed the replay gate."},
        ]},
    })
    tool = format_claude_event({
        "type": "assistant",
        "message": {"content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "python -m pytest"}},
        ]},
    })
    result = format_claude_event({
        "type": "result", "is_error": False,
        "num_turns": 12, "total_cost_usd": 1.5,
    })
    assert message == "CLAUDE MESSAGE Fixed the replay gate."
    assert tool == "CLAUDE TOOL Bash python -m pytest"
    assert result == "CLAUDE result [ok] turns=12 cost=$1.50"


def test_a_gate_that_never_runs_is_an_alarm_not_a_healthy_brain() -> None:
    """Motion is not progress.

    Measured 2026-09-05: eight clean yield/recycle cycles, seven intervals
    advanced, 18,568 accepted episodes and "0 failed" on every worker pass --
    with zero admissions for two weeks, because the admission gate never
    executed once. A watcher that only checks liveness called that healthy
    and let the budget burn. An absent gate logs neither pass nor failure, so
    the only tell is that it has produced no artifacts.
    """
    live = probe("deferred_replay_training")

    healthy = classify_probe(
        {**live,
         "admissions": {"gate_artifacts": 4, "hours_since_admission": 0.5,
                        "event_counts": {}},
         "memory": {"available_gb": 6.0}},
        stall_seconds=1800.0,
    )
    assert healthy.kind == "healthy"

    never_ran = classify_probe(
        {**live,
         "admissions": {"gate_artifacts": 0, "hours_since_admission": 300.0,
                        "event_counts": {"deferred_replay_resource_yield": 8}},
         "memory": {"available_gb": 6.0}},
        stall_seconds=1800.0,
    )
    assert never_ran.kind == "fix_required"
    assert "never produced an artifact" in never_ran.reason

    # One yield is not yet evidence; two cycles with no artifact is.
    single = classify_probe(
        {**live,
         "admissions": {"gate_artifacts": 0, "hours_since_admission": 1.0,
                        "event_counts": {"deferred_replay_resource_yield": 1}},
         "memory": {"available_gb": 6.0}},
        stall_seconds=1800.0,
    )
    assert single.kind == "healthy"


def test_the_probe_counts_artifact_names_the_supervisor_actually_writes() -> None:
    """A glob that matches nothing reports 0 forever, which reads as an alarm.

    The probe counted `*interval_recall*`. Nothing writes that name:
    `interval_recall` is a health-event *kind* and a JSON *key* inside
    deferred-replay-<digest>.admission.json. Measured 2026-09-05 on the
    training host, the glob returned 0 while 45 admission artifacts and 402
    rejection records sat in the same tree -- and that vacuous 0 was quoted
    as evidence the gate had never executed.

    So pin the counted names to the writer: every pattern the probe globs for
    gate evidence must correspond to something the supervisor publishes.
    """
    root = Path(__file__).parents[1]
    watcher = (root / "scripts" / "aws" / "watch_programming_brain.py").read_text(
        encoding="utf-8"
    )
    supervisor = (
        root / "scripts" / "programming_curriculum_supervisor.py"
    ).read_text(encoding="utf-8")

    assert "runtime.glob('*interval_recall*')" not in watcher, (
        "nothing writes a file named *interval_recall*; counting it is vacuous"
    )
    # A passing gate publishes this; a rejected one leaves failure.json.
    assert "deferred-replay-*.admission.json" in watcher
    assert 'f"deferred-replay-{digest}.admission.json"' in supervisor
    assert "deferred/*/evidence/*/failure.json" in watcher
    assert '"deferred" / digest / "evidence"' in supervisor
    assert 'evidence / "failure.json"' in supervisor


def test_a_starved_gate_is_distinguished_from_a_rejecting_one() -> None:
    """Both arrive as `deferred_replay_failed`; they need opposite fixes.

    A worker SIGTERMed by the memory guard never reached its gate, so the work
    unit is too large for the window. A gate that ran and rejected is a real
    capability failure. Counting failures cannot tell them apart -- only where
    the transaction died can.
    """
    live = probe("deferred_replay_training")
    yields = {"deferred_replay_resource_yield": 8}

    starved = classify_probe(
        {**live,
         "admissions": {"gate_artifacts": 45, "hours_since_admission": 300.0,
                        "replay_failures_before_gate": 53,
                        "replay_failures_at_gate": 0,
                        "event_counts": yields},
         "memory": {"available_gb": 6.0}},
        stall_seconds=1800.0,
    )
    assert starved.kind == "fix_required"
    assert "starving its own gate" in starved.reason

    # The gate is reaching a verdict, so this is a capability failure, not a
    # starved gate -- it must not be reported as the work unit being too big.
    rejecting = classify_probe(
        {**live,
         "admissions": {"gate_artifacts": 45, "hours_since_admission": 0.5,
                        "replay_failures_before_gate": 53,
                        "replay_failures_at_gate": 235,
                        "event_counts": yields},
         "memory": {"available_gb": 6.0}},
        stall_seconds=1800.0,
    )
    assert rejecting.kind == "healthy", rejecting.reason


def test_a_long_admission_drought_wakes_the_agent() -> None:
    stale = classify_probe(
        {**probe("deferred_replay_training"),
         "admissions": {"gate_artifacts": 6, "hours_since_admission": 48.0,
                        "event_counts": {}},
         "memory": {"available_gb": 6.0}},
        stall_seconds=1800.0, admission_stall_hours=6.0,
    )
    assert stale.kind == "fix_required"
    assert "no interval admitted" in stale.reason


def test_host_memory_pressure_wakes_the_agent_before_the_oom() -> None:
    """The kernel killed the brain at anon-rss 15,450,400 kB on a no-swap
    host; an alarm below the floor is the last chance to act first."""
    low = classify_probe(
        {**probe("deferred_replay_training"),
         "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                        "event_counts": {}},
         "memory": {"available_gb": 0.9}},
        stall_seconds=1800.0, memory_floor_gb=1.5,
    )
    assert low.kind == "fix_required"
    assert "below the" in low.reason


def test_watcher_runs_claude_on_opus_at_xhigh_with_permissions_bypassed() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts/aws/watch_programming_brain.py"
    ).read_text(encoding="utf-8")
    # An alarm at 03:00 must be repaired, not queued behind a prompt.
    assert '"--dangerously-skip-permissions"' in source
    assert '"--effort", effort' in source
    assert 'effort: str = "xhigh"' in source
    assert 'model: str = "opus"' in source
    assert '"--output-format", "stream-json"' in source
    # A stale session id must not swallow the alarm.
    assert "retrying as a fresh" in source
