from __future__ import annotations

from pathlib import Path

import scripts.aws.watch_programming_brain as watch
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


def test_a_deployed_fix_that_was_never_loaded_is_a_fault() -> None:
    """Deploying a fix is not applying it.

    Measured 2026-09-05: the fix that stops a memory yield being scored as a
    semantic failure was committed and deployed byte-identical to the host,
    then left unloaded because nothing restarted the unit. Python compiles a
    module at import, so the supervisor kept running the previous one for
    another 850 s. Every check that grepped the FILE reported the fix present
    while the process converted 19 of 19 resource yields into
    `deferred_replay_failed` and admitted nothing for 349 hours.
    """
    from scripts.aws.admission_watchdog import faults

    healthy = {
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 10, "tick_delta": 5, "deferred": 3,
        "disk_free_gb": 200, "mem_free_gb": 9, "progress_age": 4,
        "status_age": 30, "state": "deferred_replay_training",
    }

    # Source newer than the process: the deploy looks finished and changed
    # nothing. This is the only signal that separates the two.
    stale = faults({**healthy, "stale_code_lag": 850}, baseline_deferred=3)
    assert any(f.startswith("stale_code:") for f in stale), stale
    assert "restart the unit" in " ".join(stale)

    # A deploy that restarts promptly writes the source seconds before the
    # new process starts. Alarming there would fire on every correct deploy.
    for lag in (300, 12, 0, -1, -850):
        clean = faults({**healthy, "stale_code_lag": lag}, baseline_deferred=3)
        assert not any(f.startswith("stale_code:") for f in clean), (lag, clean)

    # A host that could not be measured must not manufacture a fault.
    absent = faults({**healthy, "stale_code_lag": None}, baseline_deferred=3)
    assert not any(f.startswith("stale_code:") for f in absent), absent


def test_a_frozen_status_file_is_not_a_wedge_while_the_replay_advances() -> None:
    """`status_stale` matched state NAMES, so it fired on a healthy pass.

    The supervisor writes its status only at state transitions, so it freezes
    in whatever state preceded the pass. `resource_node_recycled` is the usual
    one and was not in the allow-list. Measured 2026-09-05: the fault fired at
    `status_age 3610s` on a host whose progress file was 4 s old and whose
    tick advanced 312-336 per poll.
    """
    from scripts.aws.admission_watchdog import faults

    base = {
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 10, "deferred": 3, "disk_free_gb": 200,
        "mem_free_gb": 9, "state": "resource_node_recycled",
        "status_age": 3610, "supervisor_busy": True,
    }

    converging = faults({**base, "progress_age": 4, "tick_delta": 312},
                        baseline_deferred=3)
    assert not any(f.startswith("status_stale:") for f in converging), converging

    # A moving tick alone is enough, even with no progress file.
    ticking = faults({**base, "progress_age": 10 ** 9, "tick_delta": 312},
                     baseline_deferred=3)
    assert not any(f.startswith("status_stale:") for f in ticking), ticking

    # Nothing alive at all is the wedge the fault exists to catch, and it must
    # still fire -- otherwise this change trades one blind spot for another.
    wedged = faults({**base, "progress_age": 7200, "tick_delta": 0},
                    baseline_deferred=3)
    assert any(f.startswith("status_stale:") for f in wedged), wedged


def test_a_stale_deploy_names_what_restarting_would_cost() -> None:
    """A restart mid-replay is not free; it rolls the interval back.

    `run_deferred_replays` publishes `deferred-replay-active.json` with
    `state: "training"` before its first pass, and on startup
    `recover_interrupted_deferred_replay` rolls back any marker that is not
    `admitted`. Measured 2026-09-05 the marker was 6,762 s old with
    `durable_next_row` at 78,168 of 131,072, so the unconditional advice to
    "restart the unit" would have destroyed ~2 h of billed compute to load an
    observability fix.
    """
    from scripts.aws.admission_watchdog import faults

    base = {
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 10, "tick_delta": 5, "deferred": 3,
        "disk_free_gb": 200, "mem_free_gb": 9, "progress_age": 4,
        "status_age": 30, "state": "deferred_replay_training",
        "stale_code_lag": 850,
    }

    mid = faults({**base, "replay_marker_state": "training",
                  "replay_marker_interval": "jupyter-scientific-full:0:131072"},
                 baseline_deferred=3)
    stale_line = next(f for f in mid if f.startswith("stale_code:"))
    assert "rolls it back to its start row" in stale_line
    assert "jupyter-scientific-full:0:131072" in stale_line
    assert "so restart the unit" not in stale_line

    # An already-committed marker is recovered on startup, not rolled back.
    committed = faults({**base, "replay_marker_state": "admitted"},
                       baseline_deferred=3)
    assert "restart the unit" in next(
        f for f in committed if f.startswith("stale_code:"))

    # No marker at all means no interval is at risk.
    idle = faults(base, baseline_deferred=3)
    assert "restart the unit" in next(
        f for f in idle if f.startswith("stale_code:"))


def test_the_probe_measures_the_process_not_just_the_file() -> None:
    """The fault above is only reachable if the probe actually reports the
    lag; a fault keyed to a field nothing populates is a vacuous zero."""
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts/aws/admission_watchdog.py"
    ).read_text(encoding="utf-8")
    assert 'out["stale_code_lag"]' in source
    # Process start time, compared against the source mtime -- not a grep of
    # the file for the fix, which is what read as healthy on 2026-09-05.
    assert 'os.path.getmtime(f"/proc/{pid}")' in source
    assert "programming_curriculum_supervisor.py" in source
    # Same rule for the marker the remedy is conditioned on: a fault that
    # branches on a field nothing populates always takes one branch.
    assert 'out["replay_marker_state"]' in source
    assert "deferred-replay-active.json" in source


def test_a_drought_is_only_a_fault_when_the_queue_is_not_converging() -> None:
    """Silence and stall were the same thing only while a stalled interval
    was pinned at row 0 forever.

    With mid-pass resume an interval is 131,072 rows at ~4-10 rows/s and
    yields the host several times on the way, so it legitimately outlives the
    200-minute limit while accumulating rows the whole time. Firing there
    wakes the agent every ~3 minutes over an already-fixed fault.
    """
    from scripts.aws.admission_watchdog import faults

    drought = {
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 251282, "tick_delta": 5, "deferred": 3,
        "disk_free_gb": 200, "mem_free_gb": 9, "progress_age": 4,
        "status_age": 30, "state": "deferred_replay_training",
        "stale_code_lag": 0,
    }

    # Converging: 69.8 h of silence, but the resume row moved this poll.
    moving = faults({**drought, "replay_advancing": True}, baseline_deferred=3)
    assert not any(f.startswith("no_admission") for f in moving), moving

    # Not converging: the same silence with a resume row that never moves is
    # the original 13-passes-at-row-0 stall, and must still alarm.
    flat = faults({**drought, "replay_advancing": False}, baseline_deferred=3)
    assert any(f.startswith("no_admission") for f in flat), flat
    assert "resume row is not moving" in " ".join(flat)

    # A settle or admission gate writes no resume row for minutes; that is a
    # legitimate phase, not a stall.
    for excuse in ("gating", "supervisor_busy"):
        busy = faults({**drought, "replay_advancing": False, excuse: True},
                      baseline_deferred=3)
        assert not any(f.startswith("no_admission") for f in busy), (excuse, busy)

    # Inside the limit nothing fires regardless.
    fresh = faults({**drought, "last_admission_age": 60,
                    "replay_advancing": False}, baseline_deferred=3)
    assert not any(f.startswith("no_admission") for f in fresh), fresh


def test_interval_rollover_counts_as_progress_not_a_stall() -> None:
    """A finished interval resets the next one's resume row to a LOWER value.
    Requiring a strictly increasing row would read that rollover -- the one
    event that proves an interval completed -- as a stall."""
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts/aws/admission_watchdog.py"
    ).read_text(encoding="utf-8")
    assert "row != last_resume_row" in source
    # And the probe must actually publish the row the comparison reads.
    assert 'out["replay_resume_row"]' in source
    assert "deferred-replay-*.resume.json" in source


def test_watcher_reexecs_itself_when_its_own_source_changes(monkeypatch,
                                                            tmp_path) -> None:
    """The watcher must apply the rule it enforces on the supervisor.

    It reports `stale_code_lag` for the supervisor and exempted itself. On
    2026-09-05 the `gate_artifacts` probe was repaired at 10:06 and the
    watcher process had started at 09:46, so it kept counting with the vacuous
    `*interval_recall*` glob and woke an agent claiming "the gate is not
    running" against a tree holding 45 admission artifacts.
    """
    activity = tmp_path / "activity.log"
    calls: list[list[str]] = []
    monkeypatch.setattr(
        watch.os, "execv",
        lambda executable, argv: calls.append([executable, *argv[1:]]),
    )

    # Unchanged source must NOT re-exec. Asserted first and separately: a
    # guard that fires unconditionally would pass the positive case below
    # while turning the poll loop into an exec loop.
    monkeypatch.setattr(watch, "source_mtime", lambda: 1000.0)
    watch.reload_stale_watcher(1000.0, activity, dry_run=False)
    watch.reload_stale_watcher(1500.0, activity, dry_run=False)
    assert calls == []
    assert not activity.exists()

    # A dry run never re-execs, so `--once --dry-run` stays a pure probe.
    monkeypatch.setattr(watch, "source_mtime", lambda: 2000.0)
    watch.reload_stale_watcher(1000.0, activity, dry_run=True)
    assert calls == []

    # Source newer than the compiled process re-execs with argv preserved.
    monkeypatch.setattr(watch.sys, "argv", ["watch.py", "--poll-seconds", "300"])
    watch.reload_stale_watcher(1000.0, activity, dry_run=False)
    assert len(calls) == 1
    assert calls[0][1:] == ["watch.py", "--poll-seconds", "300"]
    assert "WATCHER RELOAD" in activity.read_text(encoding="utf-8")


def test_source_mtime_covers_the_transport_module_too() -> None:
    """A stale probe can come from the transport, not just the watcher."""
    assert watch.source_mtime() > 0.0
    watcher = Path(watch.__file__)
    assert watch.source_mtime() >= watcher.stat().st_mtime


def test_a_stale_status_file_alone_is_not_a_stalled_curriculum() -> None:
    """The supervisor is deliberately silent for the length of a replay pass.

    It publishes `curriculum-supervisor.status.json` only at state
    transitions, so a 49,152-row pass at ~13 rows/s leaves it ~3,900 s stale
    against a 1,800 s alarm while training runs perfectly. Measured
    2026-09-05: status_age 2630 s with a progress file 2.3 s old advancing at
    13.1 rows/s across twelve consecutive 30 s windows.
    """
    # Deliberately not a `service_stage: replay` probe: that short-circuits to
    # `quarantine_ready` before the staleness check and would never reach the
    # branch under test.
    converging = probe("midphase_gate_failed", age=2630.0)
    converging["throughput"] = {"age_seconds": 2.3, "durable_next_row": 73944}
    assert classify_probe(converging, stall_seconds=1800).kind == "healthy"

    # Both stale is a genuinely stalled worker and must still alarm.
    dead = probe("deferred_replay_training", age=2630.0)
    dead["throughput"] = {"age_seconds": 4000.0, "durable_next_row": 73944}
    decision = classify_probe(dead, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "heartbeat is 4000s old" in decision.reason

    # No progress file at all cannot be read as liveness.
    missing = probe("midphase_gate_failed", age=2630.0)
    decision = classify_probe(missing, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "no progress file" in decision.reason
