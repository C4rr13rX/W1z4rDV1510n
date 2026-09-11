from __future__ import annotations

import json
import os
import re
import sys
import time
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


def remote_probe_body() -> str:
    """Return the Python the watcher actually ships to the training host.

    Built through the real call path rather than re-read from the file, so the
    f-string interpolation and `{{`/`}}` escaping are exercised exactly as they
    are in production.
    """
    sent: list[str] = []

    class Result:
        stdout = "running"

    def fake_aws(*args, **kwargs):
        return Result()

    def fake_send(profile, instance_id, commands, timeout, comment=""):
        sent.append(commands[0])
        return {"StandardOutputContent": "{}"}

    original_aws, original_send = watch.aws, watch.send_and_wait
    watch.aws, watch.send_and_wait = fake_aws, fake_send
    try:
        watch.remote_probe("p", "i-0", "/srv/wizard/runtime/x")
    finally:
        watch.aws, watch.send_and_wait = original_aws, original_send

    command = sent[0]
    body = command.split("<<'PY'\n", 1)[1]
    return body.rsplit("\nPY", 1)[0]


def test_the_shipped_probe_body_is_valid_python() -> None:
    """A syntax error in the probe would surface only against the live host.

    The body is a heredoc inside an f-string, so nothing in the local import,
    `py_compile`, or any test that greps the file can see a broken one. The
    watcher would raise "probe returned no output" -- indistinguishable from a
    dead host or an SSM fault -- and the only way to tell them apart is a
    round trip to AWS. Compile it here instead.
    """
    body = remote_probe_body()
    # `compile("")` succeeds. If the heredoc markers this extraction depends on
    # are ever renamed, the assertion below is all that stops this test
    # reporting a pass on an empty string forever.
    assert len(body.splitlines()) > 200, "probe body did not extract"
    assert body.rstrip().endswith("separators=(',', ':')))")
    compile(body, "<remote-probe>", "exec")


def test_the_heartbeat_is_the_freshest_writer_not_a_fixed_file() -> None:
    """Two writers advance rows; whichever is live is the heartbeat.

    The replay worker rewrites `deferred-replay-*.progress.json` every batch
    and the forward worker rewrites `curriculum-supervisor.status.json` every
    batch. `throughput` reads only the first, so during a forward block it
    republishes the last replay pass. Measured 2026-09-09 on the fault that
    woke this session: that file was 100.7 h old carrying durable_next_row
    201344, printed beside a live status at row 16416 with no staleness
    marker, while the forward block was advancing at 15.3 rows/s. The payload
    admitted two readings -- "the run went backwards 185k rows" or "throughput
    flatlined for four days" -- and neither was true.
    """
    body = remote_probe_body()
    # The freshest of the two, chosen by age, rather than a hard-coded file.
    assert "'source': source" in body
    assert "min(ages, key=lambda item: item[0])" in body
    assert "'replay_progress'" in body and "'status'" in body
    # A superseded progress file must say so in the payload, not only in a
    # comment no reader of the JSON evidence will ever see.
    assert "throughput['is_live_heartbeat'] = False" in body
    assert "throughput['superseded_by'] = source" in body
    # A rate is the thing that separates a converging block from a live
    # process that is not training, and the payload never carried one.
    assert "'rows_per_second'" in body


def test_the_forward_driver_progress_file_is_a_heartbeat_candidate(tmp_path) -> None:
    """The forward block's own writer must be selectable, not just the two.

    Naming the status file as "the" forward heartbeat fixed the stale-replay
    case and created its mirror image. Measured 2026-09-10 mid
    `continuous_canary`: `go-systems.progress.json` was 7.6 s old climbing
    50192 -> 50224, while the status file the rule selected sat 718 s stale at
    49152. The payload published `rows_per_second: 0.0` and woke an agent to
    diagnose a stall on a block that was training normally.

    So this runs the shipped selection code against a real directory rather
    than grepping for a name: a glob that cannot match reports nothing forever,
    and asserting on the source text would not have caught the original bug
    either.
    """
    body = remote_probe_body()
    start = body.index("status_file = runtime /")
    segment = body[start:body.index("def _file_age(path):")]

    # Oldest to newest, so the forward file is unambiguously the freshest and
    # the stale replay leftover is the trap the old rule fell into.
    (tmp_path / "deferred-replay-abc.progress.json").write_text("{}")
    (tmp_path / "curriculum-supervisor.status.json").write_text("{}")
    (tmp_path / "go-systems.progress.json").write_text("{}")
    for name, age in (("deferred-replay-abc.progress.json", 362_000.0),
                      ("curriculum-supervisor.status.json", 718.0),
                      ("go-systems.progress.json", 7.6)):
        target = tmp_path / name
        os.utime(target, (time.time() - age, time.time() - age))

    namespace: dict = {"runtime": tmp_path}
    exec(compile(segment, "<selection>", "exec"), namespace)

    forward = namespace["forward_file"]
    assert forward is not None, "the forward driver's writer was not selectable"
    assert forward.name == "go-systems.progress.json"
    # The replay leftover must NOT be picked up by the forward glob, or the
    # 100.7 h stale file returns through the door opened for this fix.
    assert not forward.name.startswith("deferred-replay-")
    assert namespace["progress_file"].name == "deferred-replay-abc.progress.json"

    # And it must actually be offered to the freshest-writer rule.
    assert "'forward_progress', forward_file" in body


def test_the_agent_stream_is_decoded_as_utf8_not_the_windows_locale(tmp_path) -> None:
    """An alarm that cannot decode its own agent's output wakes nobody.

    `text=True` without `encoding` uses the locale codec. On this Windows host
    that is cp1252, which has no mapping for 0x9d -- the third byte of U+201D,
    an ordinary curly quote. Measured across 2026-09-05..09: 27 alarms died
    with `AGENT INVOKE FAILED 'charmap' codec can't decode byte 0x9d`, each
    after the agent had already started streaming, so the child was orphaned
    and the fault went unworked. It surfaces as returncode 127, the same code
    used for "no launcher on PATH".

    The child here emits a curly quote deliberately: that single character is
    the whole defect, and a test using only ASCII passes on the broken code.
    """
    stdout_path = tmp_path / "out.jsonl"
    stderr_path = tmp_path / "err.log"
    activity_path = tmp_path / "activity.log"

    child = (
        "import sys;"
        "sys.stdin.read();"
        "sys.stdout.buffer.write("
        "'{\"type\":\"system\",\"subtype\":\"thinking_tokens\"}\\n'"
        ".encode('utf-8'));"
        "sys.stdout.buffer.write('\\u201cbudget\\u201d\\n'.encode('utf-8'));"
        "sys.stdout.buffer.flush()"
    )
    returncode = watch._run_claude(
        [sys.executable, "-c", child],
        Decision("fix_required", "reason", "fp"),
        probe("midphase_gate_failed"),
        stdout_path, stderr_path, activity_path,
    )

    assert returncode == 0
    # The bytes must survive the round trip, not merely fail to raise.
    assert "”" in stdout_path.read_text(encoding="utf-8")


def test_a_new_probe_field_cannot_break_event_deduplication() -> None:
    """`heartbeat` varies every poll; fingerprints must not follow it.

    `event_fingerprint` hashes a fixed identity drawn from `status`, so adding
    a live-varying field is safe -- but only for as long as that stays true.
    If the fingerprint is ever widened to the whole probe, a rate that changes
    each poll would mint a new event id every time, and the cooldown and
    stability-poll machinery that stops the watcher spawning duplicate agents
    would silently stop working.
    """
    base = probe("midphase_gate_failed")
    moving = json.loads(json.dumps(base))
    moving["heartbeat"] = {"source": "status", "rows_per_second": 15.3,
                           "row": 21544, "age_seconds": 1.0}
    still = json.loads(json.dumps(base))
    still["heartbeat"] = {"source": "status", "rows_per_second": 0.0,
                          "row": 9, "age_seconds": 2.0}
    assert (watch.event_fingerprint("fix_required", moving)
            == watch.event_fingerprint("fix_required", still))


def test_the_drought_alarm_is_not_silenced_by_visible_progress() -> None:
    """Motion is not progress, and this rule exists because of that.

    Measured across two weeks: eight clean yield/recycle cycles, seven
    intervals advanced and 18,568 accepted episodes, with zero admissions,
    because the gate never executed. A drought alarm that a positive row rate
    can switch off would have reported all of that as healthy.
    """
    moving = probe("midphase_gate_failed")
    moving["admissions"] = {"hours_since_admission": 100.7, "gate_artifacts": 47}
    moving["heartbeat"] = {"source": "status", "rows_per_second": 18.65,
                           "row": 28384, "age_seconds": 1.4}
    moving["status"]["block_target_row"] = 131072
    decision = classify_probe(moving, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "no interval admitted for 100.7h" in decision.reason


def test_the_drought_alarm_carries_the_convergence_evidence() -> None:
    """The alarm decides between "repair it" and "wait"; it should say which.

    Measured 2026-09-09: a 100.7 h drought whose cause had already been
    repaired, on a block 1.5 h from its gate. The payload carried no row rate
    at all, so separating those two readings cost two SSM round trips -- and
    at a 1800 s retry cooldown that diagnosis would have been repeated about
    three more times before the gate ran.
    """
    converging = probe("midphase_gate_failed")
    converging["admissions"] = {"hours_since_admission": 100.7,
                                "gate_artifacts": 47}
    converging["heartbeat"] = {"source": "status", "rows_per_second": 18.65,
                               "row": 28384, "age_seconds": 1.4}
    converging["status"]["block_target_row"] = 131072
    reason = classify_probe(converging, stall_seconds=1800).reason
    # 18.6, not 18.7: 18.65 is 18.6499... in binary, so `.1f` rounds down.
    assert "18.6 rows/s at row 28384 of 131072" in reason
    assert "reaches its gate in about 1.5h" in reason

    # A block that is NOT advancing must not claim an ETA it cannot support.
    stuck = probe("midphase_gate_failed")
    stuck["admissions"] = {"hours_since_admission": 100.7, "gate_artifacts": 47}
    stuck["heartbeat"] = {"source": "status", "rows_per_second": 0.0,
                          "row": 28384, "age_seconds": 1.4}
    stuck["status"]["block_target_row"] = 131072
    assert "reaches its gate" not in classify_probe(stuck, stall_seconds=1800).reason

    # Nor may one that has already passed its target: a negative remainder
    # would print a negative ETA and read as "overdue by -0.2h".
    past = probe("midphase_gate_failed")
    past["admissions"] = {"hours_since_admission": 100.7, "gate_artifacts": 47}
    past["heartbeat"] = {"source": "status", "rows_per_second": 18.65,
                         "row": 140000, "age_seconds": 1.4}
    past["status"]["block_target_row"] = 131072
    assert "reaches its gate" not in classify_probe(past, stall_seconds=1800).reason

    # And a probe from a watcher that predates the heartbeat field must not
    # crash the classifier -- it simply carries no ETA.
    legacy = probe("midphase_gate_failed")
    legacy["admissions"] = {"hours_since_admission": 100.7, "gate_artifacts": 47}
    assert classify_probe(legacy, stall_seconds=1800).kind == "fix_required"


def _drought(beat: dict, *, state: str = "continuous_canary") -> str:
    """A live curriculum four days into an admission drought."""
    payload = {**probe(state),
               "admissions": {"gate_artifacts": 6,
                              "hours_since_admission": 102.7,
                              "event_counts": {}},
               "memory": {"available_gb": 6.0},
               "heartbeat": beat}
    payload["status"]["block_target_row"] = 131072
    decision = classify_probe(payload, stall_seconds=1800.0,
                              admission_stall_hours=6.0)
    assert decision.kind == "fix_required"
    return decision.reason


def test_a_drought_alarm_still_locates_the_block_when_the_rate_reads_zero() -> None:
    """A zero rate is when the annex matters most, and it used to delete it.

    The row advances once per COMMITTED BATCH, so between commits the progress
    file is byte-identical and any short sample reads zero. Measured 2026-09-10
    on a go-systems forward block: 32 rows per commit at 0.355 rows/s is one
    commit every ~90 s, which a 6 s sample caught roughly 7 % of the time. The
    other 93 % published `rows_per_second: 0.0`, and the annex in `decide()`
    was gated on a positive rate -- so the payload that woke the agent said
    only "no interval admitted for 102.7h while the curriculum reports itself
    active", with no hint that the block sat at row 82,896 of 131,072 and was
    still accepting episodes. Establishing "healthy, wait" cost a full session.

    A zero rate and an unknown rate are different facts. Neither is silence.
    """
    reason = _drought({"source": "forward_progress", "row": 82896,
                       "rows_per_second": 0.0, "sample_seconds": 120.0,
                       "accepted_per_second": 0.36})
    assert "82896" in reason and "131072" in reason, reason
    # The state is named, because settlement, the gate and the canary each
    # freeze the row by design and the right action under those is to wait.
    assert "continuous_canary" in reason, reason
    assert "did not move in 120s" in reason, reason
    # Rows moving is not the same as rows being learned; say which happened.
    assert "accepting 0.4 episodes/s" in reason, reason
    assert "before repairing anything" in reason, reason


def test_a_converging_block_still_reports_its_eta() -> None:
    """The positive-rate annex is the case that already worked: keep it."""
    reason = _drought({"source": "forward_progress", "row": 82896,
                       "rows_per_second": 0.355, "sample_seconds": 120.0})
    assert "0.4 rows/s at row 82896 of 131072" in reason, reason
    assert "reaches its gate in about 37.7h" in reason, reason


def test_a_drought_with_no_heartbeat_at_all_is_still_reported() -> None:
    """No annex is correct only when there is genuinely nothing to say."""
    reason = _drought({})
    assert "no interval admitted for 102.7h" in reason, reason
    assert "row" not in reason.split("--")[0].replace("curriculum", ""), reason


def test_the_heartbeat_sample_outlasts_a_batch_commit() -> None:
    """A fixed 6 s window cannot resolve a row that moves every ~90 s.

    This is the defect one layer below the annex: the consumer could not
    report a rate the producer never measured. The sample must therefore be
    adaptive -- poll until the row changes, bounded -- so that a zero means
    "did not move in two minutes" rather than "did not move in six seconds".
    """
    assert watch.HEARTBEAT_SAMPLE_SECONDS >= 90.0, (
        "the bound must exceed the slowest observed commit period (~90 s), "
        "or a zero rate is meaningless again")
    body = remote_probe_body()
    assert "time.sleep(6.0)" not in body, "the fixed short sample is back"
    # Adaptive: it stops early the moment the row moves, so a fast pass pays
    # ~2 s and only a genuinely frozen row pays the whole bound.
    assert "while time.time() < deadline:" in body
    assert "if last_row is not None and last_row != first_row:" in body
    # The second liveness signal in the same file: rows consumed vs learned.
    assert "'accepted_per_second'" in body


def test_the_transport_bound_clears_the_heartbeat_sample() -> None:
    """A probe that times out publishes nothing, which reads as a dead host."""
    source = Path(watch.__file__).read_text(encoding="utf-8")
    assert "int(HEARTBEAT_SAMPLE_SECONDS) + 300" in source, (
        "the SSM timeout no longer accounts for the heartbeat sample")


def _resume_probe_segment() -> str:
    """The shipped probe's resume-row selection, as executable source."""
    source = Path(watch.__file__).parent.joinpath(
        "admission_watchdog.py").read_text(encoding="utf-8")
    start = source.index('resume = _glob.glob(R + "/deferred-replay-')
    return source[start:source.index("# A gate legitimately freezes the tick")]


def test_the_resume_row_falls_back_to_the_forward_writer(tmp_path) -> None:
    """A forward stage writes no replay resume file, so the old rule froze.

    `replay_advancing` is one of the three conditions guarding the
    `no_admission` alarm, and it was computed only from
    `deferred-replay-*.resume.json`. A forward block writes its own
    `<phase>.progress.json` and leaves the replay resume file untouched, so
    that value is a CONSTANT for the whole block and `replay_advancing` is
    False for days on a run that is training perfectly.

    Measured 2026-09-10: 102.7 h of reported drought while the go-systems
    block advanced 82,960 -> 100,128 with `accepted_episodes` rising in
    lockstep. Same false-positive class as the vacuous glob, one file over.

    This runs the shipped selection against a real directory rather than
    grepping for a filename, because a glob that cannot match reports nothing
    forever and asserting on source text would not have caught the original.
    """
    stale = tmp_path / "deferred-replay-abc.resume.json"
    stale.write_text(json.dumps({"durable_next_row": 201344}))
    live = tmp_path / "go-systems.progress.json"
    live.write_text(json.dumps({"durable_next_row": 100128}))
    old = time.time() - 362_000
    os.utime(stale, (old, old))

    namespace = {"R": str(tmp_path), "out": {}, "os": os, "json": json,
                 "_glob": __import__("glob")}
    exec(compile(_resume_probe_segment(), "<resume>", "exec"), namespace)
    out = namespace["out"]
    assert out["resume_source"] == "forward_progress", out
    assert out["replay_resume_row"] == "forward_progress:100128", out

    # A live replay pass must still win: the forward file is then the leftover.
    os.utime(stale, None)
    os.utime(live, (old, old))
    namespace["out"] = {}
    exec(compile(_resume_probe_segment(), "<resume>", "exec"), namespace)
    assert namespace["out"]["resume_source"] == "replay_resume"
    assert namespace["out"]["replay_resume_row"] == "replay_resume:201344"


def test_a_forward_block_that_is_advancing_is_not_a_drought() -> None:
    """End to end through `faults`, on rows measured from the live host."""
    from scripts.aws.admission_watchdog import faults

    drought = {
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 369_720, "tick_delta": 12, "deferred": 3,
        "disk_free_gb": 200, "mem_free_gb": 9, "progress_age": 2,
        "status_age": 30, "state": "continuous_canary", "stale_code_lag": 0,
    }
    advancing = faults({**drought, "replay_advancing": True},
                       baseline_deferred=3)
    assert not any(f.startswith("no_admission") for f in advancing), advancing

    # And a forward row that genuinely does not move still alarms: the fix
    # must not have made the drought check unfireable.
    frozen = faults({**drought, "replay_advancing": False},
                    baseline_deferred=3)
    assert any(f.startswith("no_admission") for f in frozen), frozen


def _forward_drought_probe() -> dict:
    """The live payload that woke the agent at 103.7 h, verbatim in shape."""
    live = probe("running")
    live["runtime"] = "/runtime"
    live["service_stage"] = "forward"
    live["status"].update({"phase": "go-systems", "block_target_row": 131072,
                           "durable_next_row": 111008, "ram_next_row": 111008})
    live["admissions"] = {"hours_since_admission": 103.7, "gate_artifacts": 47,
                          "event_counts": {"deferred_replay_resource_yield": 191}}
    live["heartbeat"] = {"source": "forward_progress", "rows_per_second": 11.99,
                         "accepted_per_second": 11.99, "row": 111064,
                         "age_seconds": 0.3, "sample_seconds": 2.0,
                         "file": "go-systems.progress.json"}
    live["curriculum"] = {"forward_remaining_rows": 217409}
    live["memory"] = {"available_gb": 2.98}
    return live


def test_a_forward_stage_drought_is_expected_not_a_fault() -> None:
    """A forward stage harvests rows; it is structurally unable to admit.

    Every admission verdict is produced by the replay stage that owns the
    deferred intervals forward creates, so `hours_since_admission` rises for
    the whole length of a forward phase however well it is going -- and
    go-systems alone is 328k rows. At a 1800 s retry cooldown an ungated alarm
    re-fires for days, waking a billed agent whose only finding can be "wait".

    Measured 2026-09-10: 103.7 h published as `fix_required` against a block
    at row 111,064 of 131,072 advancing 12.0 rows/s with zero rollback
    exposure. `admission_watchdog.faults` already knew this; this emitter did
    not, so the two halves of one watchdog disagreed about the same host.
    """
    decision = classify_probe(_forward_drought_probe(), stall_seconds=1800)
    assert decision.kind == "healthy", decision.reason
    assert "go-systems" in decision.reason
    assert "admission belongs to the replay stage" in decision.reason
    # The numbers that decide "wait" must survive into the payload.
    assert "row 111064 of 131072" in decision.reason
    assert "217409 forward rows left" in decision.reason


def test_the_forward_suppression_cannot_silence_a_frozen_block() -> None:
    """The narrowness IS the safety property, so pin each condition.

    A forward stage that never reaches its handoff never admits either, which
    is precisely the fault this check was built for. Suppression therefore
    requires all three of: the stage is forward, the freshest writer is the
    forward driver, and that writer actually moved.
    """
    # Frozen row -- the by-design freezes are the agent's problem to rule out,
    # but silence here would hide a real wedge.
    frozen = _forward_drought_probe()
    frozen["heartbeat"] = {**frozen["heartbeat"], "rows_per_second": 0.0,
                           "accepted_per_second": 0.0}
    assert classify_probe(frozen, stall_seconds=1800).kind == "fix_required"

    # A replay stage is exactly the case the drought alarm exists for: eight
    # clean yield/recycle cycles and 18,568 accepted episodes with zero
    # admissions. Motion must never buy it an exemption.
    replaying = _forward_drought_probe()
    replaying["service_stage"] = "replay"
    replaying["heartbeat"] = {**replaying["heartbeat"], "source": "replay_progress"}
    decision = classify_probe(replaying, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "no interval admitted for 103.7h" in decision.reason

    # A stale forward file that is not the freshest writer proves nothing
    # about the current stage -- the heartbeat lesson, one file over.
    stale_source = _forward_drought_probe()
    stale_source["heartbeat"] = {**stale_source["heartbeat"],
                                 "source": "replay_progress"}
    assert classify_probe(stale_source, stall_seconds=1800).kind == "fix_required"


def test_a_drought_alarm_says_so_when_no_writer_exposes_a_row() -> None:
    """The third case, and the one that fell through to silence.

    Selecting the heartbeat purely on mtime picks the supervisor status file
    whenever it is freshest -- and during a REPLAY that file carries
    `resume_row`/`end_row`, never `durable_next_row`. So `row` is None, both
    annex branches are gated on `row is not None`, and the alarm goes out as a
    bare drought with no convergence evidence at all.

    Measured 2026-09-10: published `row: null, rows_per_second: null` against a
    replay converging at 14.0 rows/s with zero rollback exposure, 82,272 rows
    from its gate. An unknown rate is not a zero rate, and neither is silence.
    """
    blind = probe("deferred_replay_training")
    blind["admissions"] = {"hours_since_admission": 111.6, "gate_artifacts": 47}
    blind["heartbeat"] = {"source": None, "file": "curriculum-supervisor.status.json",
                          "age_seconds": 8.6, "row": None, "rows_per_second": None,
                          "no_row_writer": True, "freshest_writer": "status"}
    blind["status"]["block_target_row"] = 131072
    decision = classify_probe(blind, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "no interval admitted for 111.6h" in decision.reason
    # The annex must name the blindness rather than implying a dead curriculum.
    assert "no writer currently exposes a row" in decision.reason
    assert "carries no convergence evidence" in decision.reason
    assert "status" in decision.reason

    # And it must not fire when a row IS available: that payload has evidence,
    # so it gets the existing frozen/converging annex instead.
    seeing = probe("deferred_replay_training")
    seeing["admissions"] = {"hours_since_admission": 111.6, "gate_artifacts": 47}
    seeing["heartbeat"] = {"source": "replay_progress", "row": 48800,
                           "rows_per_second": 14.0, "age_seconds": 0.4,
                           "row_source_lag_seconds": 0.0}
    seeing["status"]["block_target_row"] = 131072
    reason = classify_probe(seeing, stall_seconds=1800).reason
    assert "no writer currently exposes a row" not in reason
    assert "advancing 14.0 rows/s at row 48800" in reason


def failure_summary_helpers() -> dict:
    """Exec the probe's own failure-rendering helpers out of the shipped body.

    Extracted from the real f-string rather than reimplemented, so the
    `{{`/`}}` escaping in the suite regex is exercised the way production
    renders it -- an unescaped brace there compiles fine and silently changes
    the character class.
    """
    body = remote_probe_body()
    # `failing_suites` resolves `re` from the probe's module scope, so the
    # harness has to supply it -- and then assert the shipped body really
    # imports it, or this test would pass against a probe that raises
    # NameError on the host and reports no failure at all.
    assert "import json, os, pathlib, re, sys, time" in body
    segment = body[body.index("def summarize_failure(error):"):
                   body.index("\ntry:\n", body.index("def failing_suites("))]
    namespace: dict = {"re": re}
    exec(compile(segment, "<failure-helpers>", "exec"), namespace)
    return namespace


def test_a_worker_exit_reports_its_reason_not_only_its_address() -> None:
    """The tail the supervisor deliberately attached must reach the agent.

    `replay_worker_failure` appends up to 4000 bytes of worker stderr after
    the log path, so `last_failure` carries the cause rather than a filename
    on a host nobody is reading. The probe then cut the field at 180
    characters -- and the runtime path alone is 135, so 45 characters
    survived, from the FRONT, which is the least informative end of a
    traceback.

    Measured 2026-09-10: the `quarantine_ready` wake-up published exactly
    `deferred replay worker exited 1; stderr=<path>`. The named file held a
    SchemaError for `category='systems_programming_go'` that had already been
    repaired 3 h before the current supervisor started, so the answer was in
    the payload's own source and still cost a round trip to the host.
    """
    helpers = failure_summary_helpers()
    path = ("/srv/wizard/runtime/programming-integrated-20260713/"
            "deferred-replay-8f4a439a7fc7a772.stderr.log")
    error = (
        f"deferred replay worker exited 1; stderr={path}\n"
        + "Traceback (most recent call last):\n"
        + '  File "/srv/wizard/project/tools/training_standard/schema.py", '
          "line 161, in load_script\n    raise SchemaError(\n" * 6
        + "tools.training_standard.schema.SchemaError: registry/"
          "go_systems_001.toml[script]: category='systems_programming_go' "
          "not in ['agent_planning', 'code_generation']"
    )
    summary = helpers["summarize_failure"](error)

    # The cause, which lives on the LAST line of a traceback.
    assert "SchemaError" in summary
    assert "category='systems_programming_go'" in summary
    # Prove the test fails without the fix: the old rule kept none of it.
    assert "SchemaError" not in error[:180]
    # The head still has to survive, because the worker/gate split keys on it.
    assert summary.startswith("deferred replay worker exited 1")
    assert "worker exited" in summary
    # Say that something was dropped rather than implying the error ended.
    assert "chars]" in summary


def test_a_short_failure_is_passed_through_whole() -> None:
    """Summarising must not mangle the errors that already fit."""
    helpers = failure_summary_helpers()
    error = "deferred replay worker exited -15; stderr=/srv/x.log\nSIGTERM"
    assert helpers["summarize_failure"](error) == error
    assert helpers["summarize_failure"]("") == ""


def test_the_named_failure_says_which_suites_failed() -> None:
    """An 11/12 gate must not arrive as a blob that names no suite.

    CLAUDE.md records four days of admissions held by one suite whose identity
    the ledger never carried. The gate report's repr does carry it; nothing
    pulled it out, and the first 180 characters are spent on tick and
    structure counts, so the names were always past the cut.
    """
    helpers = failure_summary_helpers()
    error = (
        "enterprise regression after go-systems: {'passed': False, "
        "'tick_before': 1108042, 'tick_after': 1108042, 'tick_delta': 0, "
        "'structure_unchanged': True, 'passed_suites': 11, "
        "'total_suites': 12, 'results': ["
        "{'name': 'python_enterprise', 'passed': True, 'timed_out': False}, "
        "{'name': 'polyglot', 'passed': False, 'timed_out': False}]}"
    )
    assert helpers["failing_suites"](error) == ["polyglot"]
    # A passing suite must never be reported as failing just because a later
    # sibling failed -- the character class excluding braces is what stops the
    # match walking out of one result dict and into the next.
    assert "python_enterprise" not in helpers["failing_suites"](error)
    # A failure that is not a suite verdict names no suites, and that must
    # read as "not applicable" rather than "everything passed".
    assert helpers["failing_suites"]("deferred replay worker exited 1") == []


def test_a_replay_block_gets_its_convergence_annex_from_end_row() -> None:
    """The annex must fire on the status a replay actually publishes.

    `block_target_row` is written only by the forward stage, which hands it to
    the driver as `--limit-rows`. Deferred replay publishes the identical
    quantity under `end_row`. The annex read only the former, so it was live
    exclusively during forward blocks -- where the drought branch is already
    suppressed, because a forward stage does not admit -- and silent during
    replay, the one stage that admits and the only one that reaches here.

    Measured 2026-09-10 against the live host: `no interval admitted for
    112.6h while the curriculum reports itself active`, and nothing else,
    while the block sat at row 86,320 of 131,072 advancing 12.8 rows/s about
    an hour from its gate. Every existing test hand-set `block_target_row`, so
    the suite passed against a payload no replay ever emits.
    """
    # The exact status shape the supervisor publishes for a deferred replay:
    # start/resume/end, and no `block_target_row` anywhere.
    live = probe("deferred_replay_training")
    live["admissions"] = {"hours_since_admission": 112.6, "gate_artifacts": 47}
    live["status"].update({
        "phase": "go-systems", "interval_id": "go-systems:0:131072",
        "start_row": 0, "resume_row": 36384, "end_row": 131072,
    })
    live["status"].pop("block_target_row", None)
    live["heartbeat"] = {"source": "replay_progress", "row": 86320,
                         "rows_per_second": 12.8, "age_seconds": 0.5,
                         "row_source_lag_seconds": 0.0, "sample_seconds": 2.0}

    decision = classify_probe(live, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "no interval admitted for 112.6h" in decision.reason
    # The annex, which was the whole point of the drought branch carrying one.
    assert "advancing 12.8 rows/s at row 86320 of 131072" in decision.reason
    assert "confirm convergence before repairing anything" in decision.reason
    # 44752 rows at 12.8 rows/s is ~0.97h. Pin the arithmetic, not just the
    # presence of a sentence.
    assert "reaches its gate in about 1.0h" in decision.reason


def test_a_frozen_replay_block_is_still_named_without_block_target_row() -> None:
    """The freeze arm needs the same fallback, or it reports nothing either.

    A rate of 0 is normal -- settlement, the admission gate and the continuous
    canary each freeze the row by design -- but that is only useful if the
    payload SAYS so. Sharing `target` with the converging arm means fixing one
    without the other would leave a frozen replay indistinguishable from a
    payload that never measured.
    """
    frozen = probe("deferred_replay_training")
    frozen["admissions"] = {"hours_since_admission": 112.6, "gate_artifacts": 47}
    frozen["status"].update({"phase": "go-systems", "end_row": 131072})
    frozen["status"].pop("block_target_row", None)
    frozen["heartbeat"] = {"source": "replay_progress", "row": 131040,
                           "rows_per_second": 0.0, "age_seconds": 0.5,
                           "sample_seconds": 120.0, "accepted_per_second": 4.2,
                           "row_source_lag_seconds": 0.0}

    reason = classify_probe(frozen, stall_seconds=1800).reason
    assert "sits at row 131040 of 131072" in reason
    assert "did not move in 120s" in reason
    # Still accepting episodes means the block is training through the freeze.
    assert "still accepting 4.2 episodes/s" in reason
    assert "freeze the row by design" in reason
    # And it must not be mistaken for the no-writer case, which is a different
    # fact: this payload measured a row, it just did not move.
    assert "no writer currently exposes a row" not in reason


def test_the_probe_resolves_the_block_target_without_the_status_file() -> None:
    """`curriculum-supervisor.status.json` is not one schema.

    Forward blocks publish `block_target_row`; a deferred replay publishes
    `end_row`; and a `resource_node_recycled` record publishes neither --
    only `trained_rows` and a `topology` dict. Measured 2026-09-10 on the
    live host, mid-replay at row 90,616 of 131,072: the status file had just
    been overwritten by a recycle record, so BOTH status keys were absent and
    the drought alarm went out bare for the second time in one session.

    So run the shipped resolver against a real directory rather than grepping
    for a key name -- the interval is durable state and the interval id
    encodes phase:start:end, neither of which a recycle can erase.
    """
    body = remote_probe_body()
    segment = body[body.index("def _block_target(status):"):
                   body.index("\nthroughput = ")]
    return _exercise_block_target(segment)


def _exercise_block_target(segment: str):
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        namespace: dict = {"json": json, "runtime": root}
        exec(compile(segment, "<block-target>", "exec"), namespace)
        resolve = namespace["_block_target"]

        # 1. Forward stage: its own key wins.
        assert resolve({"block_target_row": 131072}) == 131072
        # 2. Replay training: end_row is the same quantity under another name.
        assert resolve({"end_row": 131072, "resume_row": 36384}) == 131072

        # 3. The recycle record that broke it: no target key at all, so the
        #    resolver has to reach the durable interval file.
        recycled = {"state": "resource_node_recycled", "phase": "go-systems",
                    "trained_rows": 131072, "topology": {"tick": 4501934}}
        assert resolve(recycled) is None, "no interval on disk yet"
        (root / "deferred-replay-active.json").write_text(json.dumps({
            "interval_id": "go-systems:0:131072", "phase": "go-systems",
            "state": "training",
            "interval": {"start_row": 0, "end_row": 131072,
                         "interval_id": "go-systems:0:131072"},
        }))
        assert resolve(recycled) == 131072

        # 4. Last resort: the interval id encodes the end row, so a truncated
        #    or unreadable active file still yields a target.
        (root / "deferred-replay-active.json").write_text("{ not json")
        assert resolve({"interval_id": "go-systems:0:131072"}) == 131072
        # 5. And an id that does not encode one must report absence rather
        #    than a fabricated number.
        assert resolve({"interval_id": "go-systems"}) is None
        assert resolve({}) is None


def test_a_drought_alarm_says_when_its_named_failure_is_history() -> None:
    """`last_failure` is read from a ledger that outlives every writer.

    `curriculum-health.jsonl` is append-only, so the newest `deferred_replay_
    failed` record in it can name a cause repaired several supervisor
    generations ago. Quoting that text into a `fix_required` reason without
    its age invites the woken agent to debug a fixed bug -- and CLAUDE.md's
    standing instruction to "timestamp last_failure against process start
    before re-debugging it" was addressed to a reader the payload handed
    nothing to act on.

    Measured 2026-09-10 on the `quarantine_ready` wake-up: `last_failure` was
    a worker exit 19.6 h old, quoted beside a supervisor 0.14 h old, whose
    registry SchemaError had been redeployed 16.6 h before. Two SSM round
    trips went to establishing that the ledger text was history.
    """
    # NOT `service_stage: replay`. The quarantine_ready arm matches a
    # replay stage in a deferred_replay_* state and returns before the
    # drought branch is ever reached -- the if/else chain CLAUDE.md warns
    # about, where an earlier arm that matches ends it.
    live = probe("deferred_replay_training")
    live["admissions"] = {
        "hours_since_admission": 111.6,
        "gate_artifacts": 49,
        "last_failure": (
            "deferred replay worker exited 1; stderr=/srv/wizard/runtime/"
            "programming-integrated-20260713/deferred-replay-8f4a439a.stderr.log"
        ),
        "last_failure_age_hours": 19.62,
        "last_failure_predates_supervisor": True,
        "supervisor_age_hours": 0.14,
    }
    live["status"].update({
        "phase": "jupyter-scientific-full",
        "interval_id": "jupyter-scientific-full:201344:262144",
        "start_row": 201344, "resume_row": 201376, "end_row": 262144,
    })
    live["heartbeat"] = {"source": "replay_progress", "row": 202088,
                         "rows_per_second": 7.99, "age_seconds": 0.5,
                         "row_source_lag_seconds": 0.0, "sample_seconds": 2.0}

    decision = classify_probe(live, stall_seconds=1800)
    assert decision.kind == "fix_required"
    # The convergence annex still fires; this one is additive, not a swap.
    assert "advancing 8.0 rows/s at row 202088 of 262144" in decision.reason
    # The staleness annex itself, with both ages, because the comparison is
    # the point -- "19.6h old" alone does not say it is unreachable.
    assert "19.6h old and predates this supervisor" in decision.reason
    assert "0.1h old" in decision.reason
    assert "confirm it is still reachable before re-debugging it" in decision.reason


def test_a_failure_from_the_running_supervisor_is_not_called_history() -> None:
    """The suppression has to stay narrow or it hides live faults.

    A failure produced BY the process now running is exactly the one worth
    debugging, and an annex that called it history would send the next agent
    away from the only reachable cause on the host. So the annex keys on the
    probe's own comparison rather than on the age alone -- a 19 h-old failure
    under a 40 h-old supervisor is still live.
    """
    live = probe("deferred_replay_training")
    live["admissions"] = {
        "hours_since_admission": 111.6,
        "gate_artifacts": 49,
        "last_failure": "deferred replay worker exited 1; stderr=/srv/x.log",
        "last_failure_age_hours": 19.62,
        "last_failure_predates_supervisor": False,
        "supervisor_age_hours": 40.0,
    }
    live["status"].update({
        "interval_id": "go-systems:0:131072", "end_row": 131072,
    })
    live["heartbeat"] = {"source": "replay_progress", "row": 86320,
                         "rows_per_second": 12.8, "age_seconds": 0.5,
                         "row_source_lag_seconds": 0.0, "sample_seconds": 2.0}

    decision = classify_probe(live, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "predates this supervisor" not in decision.reason
    # And a payload from an older watcher, carrying no verdict at all, must
    # not be read as "live" either -- absence is not False.
    live["admissions"]["last_failure_predates_supervisor"] = None
    assert "predates this supervisor" not in classify_probe(
        live, stall_seconds=1800).reason


def test_the_probe_publishes_the_staleness_fields_it_is_classified_on() -> None:
    """A classifier keyed on a field nothing writes is a vacuous check.

    This repository has already paid for that once: a watchdog globbed
    `*interval_recall*`, which nothing creates, and reported 0 forever. The
    annex above reads `last_failure_predates_supervisor` out of the payload,
    so the shipped probe body has to actually emit it -- and has to capture
    the supervisor start time it is compared against.
    """
    body = remote_probe_body()
    for key in ("'last_failure_unix'", "'last_failure_age_hours'",
                "'last_failure_predates_supervisor'",
                "'supervisor_started_unix'", "'supervisor_age_hours'"):
        assert key in body, f"the shipped probe never publishes {key}"
    # The comparison needs a supervisor start time, and it has to come from
    # the process table rather than from the status file -- the status file
    # is written by whichever lifecycle event ran last, not at startup.
    assert "supervisor_started_unix = max(" in body
    assert "path.parent.stat().st_mtime" in body
    # And the failure's own timestamp has to be captured where the failure is
    # read, not inferred later from file mtimes.
    assert "recent_fail_unix = when" in body


def rate_helper() -> object:
    """Exec the probe's own `_rate` out of the shipped body.

    Extracted rather than reimplemented, so the escaping is exercised the way
    production renders it.
    """
    body = remote_probe_body()
    segment = body[body.index("def _rate(first, last, elapsed):"):
                   body.index("def _row_now(path):")]
    namespace: dict = {}
    exec(compile(segment, "<rate-helper>", "exec"), namespace)
    return namespace["_rate"]


def test_a_counter_that_went_backwards_is_a_reset_not_a_negative_rate() -> None:
    """`accepted_episodes` and `durable_next_row` restart with the worker.

    A `deferred_replay_resource_yield` kills the replay worker and relaunches
    it, so a sample straddling one sees the counter fall rather than rise.
    Measured 2026-09-10 against the live host during a yield: the episode
    count went 712 -> 8 and the payload published
    `accepted_per_second: -42.4` -- which the drought annex renders as "still
    accepting -42.4 episodes/s, so the block is training".

    An unknown rate is not a zero rate, and it is certainly not a negative
    one, so the reset reports absence.
    """
    rate = rate_helper()
    # The measurement that motivated this, to the sample that produced it.
    assert rate(712, 8, 36.0) is None
    assert rate(208648, 201376, 36.0) is None
    # An ordinary forward sample is unaffected.
    assert rate(100, 136, 2.0) == 18.0
    # A genuinely frozen row is still zero, which is a real fact and must
    # NOT be collapsed into the reset case -- settlement and the admission
    # gate both freeze the row by design and still have to read as frozen.
    assert rate(100, 100, 2.0) == 0.0
    # Absence stays absence.
    assert rate(None, 5, 2.0) is None
    assert rate(5, None, 2.0) is None


def test_a_mid_sample_worker_restart_is_named_in_the_drought_alarm() -> None:
    """The freeze branch must not present a reset as a stalled block.

    A frozen row has three by-design causes the alarm already names. A worker
    that restarted mid-sample is a fourth, and the only one where the number
    is not merely uninformative but actively wrong.
    """
    live = probe("deferred_replay_resource_yield")
    live["admissions"] = {"hours_since_admission": 111.6, "gate_artifacts": 49}
    live["status"].update({
        "interval_id": "jupyter-scientific-full:201344:262144",
        "end_row": 262144,
    })
    live["heartbeat"] = {"source": "replay_progress", "row": 208648,
                         "rows_per_second": None, "accepted_per_second": None,
                         "counter_reset": True, "age_seconds": 45.0,
                         "row_source_lag_seconds": 0.6, "sample_seconds": 36.0}

    reason = classify_probe(live, stall_seconds=1800).reason
    assert "restarted mid-window" in reason
    assert "this rate measures nothing" in reason
    # And a plain freeze, with no reset, must not claim a restart happened.
    live["heartbeat"]["counter_reset"] = False
    assert "restarted mid-window" not in classify_probe(
        live, stall_seconds=1800).reason


def test_the_probe_publishes_the_reset_flag_it_is_classified_on() -> None:
    """The annex reads `counter_reset`, so the shipped probe must emit it."""
    body = remote_probe_body()
    assert "'counter_reset'" in body, "the shipped probe never publishes it"
    # And both rates must go through the guard, or one of them still
    # publishes a negative number.
    assert "'rows_per_second': _rate(first_row, last_row, elapsed)" in body
    assert (
        "'accepted_per_second': _rate(first_accepted, last_accepted, elapsed)"
        in body
    )


def test_a_full_volume_is_named_as_the_cause_not_the_missing_owner() -> None:
    """Measured 2026-09-10: /srv/wizard reached 20 KB free on a 1.0 TB volume.

    The wrapper could not write its 6-byte `node.pid`, systemd restarted it 115
    times at RestartSec=10, and the process census landed between restarts. The
    alarm therefore read "no curriculum supervisor or wrapper owns terminal
    state deferred_replay_resource_yield" -- which is true, and useless: it
    names the symptom while the cause was one statvfs call away. Two SSM round
    trips went to rediscovering ENOSPC from a 57 MB traceback log.
    """
    orphaned = {
        **probe("deferred_replay_resource_yield", supervisors=0, wrappers=0),
        "disk": {"free_gb": 0.0, "total_gb": 1099.5, "used_percent": 100.0,
                 "free_inodes": 7105, "inodes_used_percent": 91.0},
    }
    decision = classify_probe(orphaned, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "0.00 GB free" in decision.reason
    # The symptom must survive too -- it is what the operator sees first.
    assert "nothing owns terminal state" in decision.reason

    # Without disk evidence the message must fall back, never invent a cause.
    blind = {**probe("deferred_replay_resource_yield",
                     supervisors=0, wrappers=0)}
    assert "no curriculum supervisor or wrapper owns terminal state" in (
        classify_probe(blind, stall_seconds=1800).reason)


def test_inode_exhaustion_is_enospc_even_though_df_shows_free_space() -> None:
    """Bytes and inodes both surface as ENOSPC; only one shows in `df -h`.

    An inode-exhausted host reads as healthy on free bytes while every write
    fails, so classifying on `free_gb` alone would report a contradiction.
    """
    starved = {
        **probe("deferred_replay_training"),
        "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                       "event_counts": {}},
        "disk": {"free_gb": 400.0, "total_gb": 1099.5, "used_percent": 63.0,
                 "free_inodes": 12, "inodes_used_percent": 99.9},
    }
    decision = classify_probe(starved, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "inodes" in decision.reason
    assert "400.0 GB is free" in decision.reason


def test_a_reclaim_after_the_crash_still_reports_the_failing_unit() -> None:
    """Free bytes beside a crash-looping unit is the post-reclaim reading.

    Space recovered between the failure and the probe would otherwise clear the
    alarm while the service is still down, which is how a disk outage becomes a
    silent one.
    """
    # 599 GB free is the real post-rollback reading on this host: the reclaim
    # more than succeeded, and the unit was still down. Free bytes alone would
    # have called that healthy.
    reclaimed = {
        **probe("deferred_replay_resource_yield", supervisors=0, wrappers=0),
        "disk": {"free_gb": 599.0, "total_gb": 1099.5, "used_percent": 43.0,
                 "free_inodes": 119561, "inodes_used_percent": 37.0,
                 "wrapper_enospc": True},
    }
    decision = classify_probe(reclaimed, stall_seconds=1800)
    assert decision.kind == "fix_required"
    assert "No space left on device" in decision.reason
    assert "needs restarting" in decision.reason


def test_a_healthy_volume_never_manufactures_a_disk_fault() -> None:
    """The supervisor's own floor is 8 GB; alarming at or above it would fire
    on the guard doing its job."""
    healthy = {
        **probe("deferred_replay_training"),
        "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                       "event_counts": {}},
        "disk": {"free_gb": 599.0, "total_gb": 1099.5, "used_percent": 42.0,
                 "free_inodes": 119561, "inodes_used_percent": 37.0,
                 "wrapper_enospc": False},
    }
    assert classify_probe(healthy, stall_seconds=1800).kind == "healthy"
    # A probe that could not stat the volume must not alarm on a missing key.
    errored = {**healthy, "disk": {"error": "OSError: boom"}}
    assert classify_probe(errored, stall_seconds=1800).kind == "healthy"


def test_the_probe_publishes_the_disk_fields_it_is_classified_on() -> None:
    """The vacuous-signal rule: a key the shipped probe never emits classifies
    nothing forever. `admission_watchdog.faults` had `disk_low` while this
    emitter had no disk telemetry at all -- the two-emitter drift CLAUDE.md
    names, recurring on a new axis."""
    body = remote_probe_body()
    assert "os.statvfs" in body, "the shipped probe never measures the volume"
    for field in ("'free_gb'", "'used_percent'", "'free_inodes'",
                  "'inodes_used_percent'", "'wrapper_enospc'"):
        assert field in body, f"the shipped probe never publishes {field}"
    assert "'disk': disk," in body, "the payload never carries the disk block"


def test_both_watchdog_emitters_agree_that_a_full_volume_is_a_fault() -> None:
    """CLAUDE.md: when changing one emitter, change the other. The same host
    must not be simultaneously healthy and faulted."""
    watchdog = (
        Path(__file__).parents[1] / "scripts/aws/admission_watchdog.py"
    ).read_text(encoding="utf-8")
    assert "disk_low" in watchdog
    source = (
        Path(__file__).parents[1] / "scripts/aws/watch_programming_brain.py"
    ).read_text(encoding="utf-8")
    assert "disk_exhaustion_fault" in source
    assert "DISK_ALARM_FLOOR_GB" in source


def test_the_disk_floor_leaves_time_to_act_at_the_measured_burn_rate() -> None:
    """A floor is only useful relative to how fast the volume drains.

    Measured 2026-09-10 over a 420 s window during deferred replay: 125.4 GB/h,
    about 10.8 MB per trained row, because the `.wbrain` neuron store is
    append-only with no compactor. At that rate the supervisor's own 8 GB guard
    is under four minutes of warning and `admission_watchdog`'s 20 GB is under
    ten -- both too late to do anything but watch the wrapper crash-loop.
    """
    burn_gb_per_hour = 125.4
    minutes = watch.DISK_ALARM_FLOOR_GB / burn_gb_per_hour * 60.0
    assert minutes >= 20.0, (
        f"a {watch.DISK_ALARM_FLOOR_GB} GB floor is only {minutes:.1f} minutes "
        f"at the measured {burn_gb_per_hour} GB/h burn rate"
    )
    # And it must sit well clear of the supervisor's own yield floor, or the
    # alarm fires on the guard doing its job rather than on the guard failing.
    assert watch.DISK_ALARM_FLOOR_GB > 8.0 * 2


def test_a_supervisor_stopped_on_its_own_disk_floor_is_not_healthy() -> None:
    """The alarm floors sit BELOW the floor the supervisor halts at.

    `--min-free-disk-gb` is 150, `watch.DISK_ALARM_FLOOR_GB` is 48 and
    `admission_watchdog`'s trigger is 20. A supervisor that gave up on its own
    disk floor therefore parks at ~149 GB free -- three times one alarm floor
    and seven times the other -- so both emitters read a healthy volume beside
    training that is stopped dead. Measured 2026-09-10: burn 159.91 GB/h with
    69.68 GB above the floor gave 0.44 h of runway against a 10.02 h ETA to the
    interval's gate, so this state was about one half-hour away and nothing
    would have reported it.
    """
    parked = {
        **probe("disk_exhausted_unrecoverable"),
        "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                       "event_counts": {}},
        "status": {
            "state": "disk_exhausted_unrecoverable",
            "phase": "jupyter-scientific-full",
            "minimum_free_disk_gb": 150,
            "durable_next_row": 222520,
            "disk_reclaim_attempts": 3,
            "disk_reclaimed_bytes": 0,
        },
        # Far above BOTH alarm floors: this is the whole point.
        "disk": {"free_gb": 149.2, "total_gb": 1023.5, "used_percent": 85.0,
                 "free_inodes": 107322485, "inodes_used_percent": 0.0,
                 "wrapper_enospc": False},
    }
    decision = classify_probe(parked, stall_seconds=1800)
    assert decision.kind == "fix_required", decision
    assert "150" in decision.reason
    assert "222520" in decision.reason
    # It must not be mistaken for the routine quarantine handoff: that arm
    # matches any `deferred_replay_*` state during replay and would have
    # swallowed this one under its old name.
    assert decision.kind != "quarantine_ready"

    # The same evidence must reach the OTHER emitter, or the host is
    # simultaneously healthy and faulted -- the two-emitter drift that has now
    # happened three times in this file's history.
    from scripts.aws.admission_watchdog import faults

    found = faults({
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 600, "tick_delta": 0, "deferred": 3,
        "disk_free_gb": 149, "mem_free_gb": 9, "progress_age": 30,
        "status_age": 30, "state": "disk_exhausted_unrecoverable",
        "min_free_disk_gb": 150, "supervisor_busy": True,
    }, baseline_deferred=3)
    assert any(f.startswith("disk_exhausted_unrecoverable:") for f in found), found

    # A volume above the floor with the supervisor training normally must stay
    # healthy, or this trades a blind spot for a false alarm.
    training = {
        **probe("deferred_replay_training"),
        "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                       "event_counts": {}},
        "disk": {"free_gb": 211.7, "total_gb": 1023.5, "used_percent": 79.0,
                 "free_inodes": 107322485, "inodes_used_percent": 0.0,
                 "wrapper_enospc": False},
    }
    assert classify_probe(training, stall_seconds=1800).kind != "fix_required"


def test_the_halt_is_visible_on_a_supervisor_that_predates_the_fix() -> None:
    """A hung host cannot redeploy itself, so the alarm must not require it.

    The running generation always predates the repair, and it publishes
    `resource_waiting` rather than `disk_exhausted_unrecoverable`. Keying the
    fault only on the new state would leave exactly the hosts that are already
    stuck reporting healthy. The supervisor's own floor travels in the status
    payload, so no assumption about the deployed build is needed.
    """
    old_build = {
        **probe("resource_waiting"),
        "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                       "event_counts": {}},
        "status": {
            "state": "resource_waiting",
            "phase": "jupyter-scientific-full",
            "minimum_free_disk_gb": 150,
            "durable_next_row": 223240,
        },
        "disk": {"free_gb": 149.4, "total_gb": 1023.5, "used_percent": 85.0,
                 "free_inodes": 107322485, "inodes_used_percent": 0.0,
                 "wrapper_enospc": False},
    }
    decision = classify_probe(old_build, stall_seconds=1800)
    assert decision.kind == "fix_required", decision
    assert "resource_waiting" in decision.reason
    assert "cannot end on its own" in decision.reason

    # `resource_waiting` on MEMORY pressure is a real yield that does end --
    # the recycle hands the arena back. Alarming on it would fire on the guard
    # doing its job, so the disk floor must actually be breached.
    memory_yield = {
        **old_build,
        "disk": {**old_build["disk"], "free_gb": 600.0},
    }
    assert classify_probe(memory_yield, stall_seconds=1800).kind != "fix_required"


def test_training_below_the_supervisors_own_floor_is_a_fault() -> None:
    """The exact payload that both emitters called healthy on 2026-09-11.

    The previous fix here keyed on `resource_waiting`, which covers a
    supervisor PARKED on its floor and misses one that never reaches it.
    `--min-free-disk-gb 150` sat on the running supervisor's own argv while
    deferred replay trained at 96.39 GB free and 108.66 GB/h, because
    `disk_floor_breached` is enforced by the forward corpus-phase loop and
    `forward_remaining_rows` was 0. That is not a low volume under a working
    guard; it is a guard attached to a stage that has finished, and the host
    was ~40 minutes from ENOSPC with no alarm anywhere.
    """
    unguarded = {
        **probe("deferred_replay_training"),
        "admissions": {"gate_artifacts": 11, "hours_since_admission": 11.1,
                       "event_counts": {}},
        "status": {
            "state": "deferred_replay_training",
            "phase": "jupyter-scientific-full",
            "interval_id": "jupyter-scientific-full:201344:262144",
            "minimum_free_disk_gb": 150,
            "resume_row": 227056,
        },
        # Above BOTH alarm floors (48 and 20) and below the supervisor's own.
        "disk": {"free_gb": 96.39, "total_gb": 1098.97, "used_percent": 91.2,
                 "free_inodes": 107322484, "inodes_used_percent": 0.0,
                 "wrapper_enospc": False},
    }
    decision = classify_probe(unguarded, stall_seconds=1800)
    assert decision.kind == "fix_required", decision
    assert "96.39" in decision.reason and "150" in decision.reason

    # Same evidence, other emitter. Changing one and not the other is how this
    # host was simultaneously healthy and faulted three times already.
    from scripts.aws.admission_watchdog import faults

    found = faults({
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 39960, "tick_delta": 12, "deferred": 3,
        "disk_free_gb": 96.39, "mem_free_gb": 9, "progress_age": 2,
        "status_age": 32, "state": "deferred_replay_training",
        "min_free_disk_gb": 150, "supervisor_busy": True,
    }, baseline_deferred=3)
    assert any(f.startswith("disk_floor_unenforced:") for f in found), found

    # The suppression must stay narrow: a supervisor legitimately PARKED on its
    # floor already has its own named fault and must not also be reported as an
    # unenforced guard, and a healthy volume must stay healthy.
    parked = faults({
        "unit": "active", "brain_up": True, "failed_since_deploy": 0,
        "last_admission_age": 600, "tick_delta": 0, "deferred": 3,
        "disk_free_gb": 149, "mem_free_gb": 9, "progress_age": 30,
        "status_age": 30, "state": "disk_exhausted_unrecoverable",
        "min_free_disk_gb": 150, "supervisor_busy": True,
    }, baseline_deferred=3)
    assert not any(f.startswith("disk_floor_unenforced:") for f in parked)

    # Hoisting the disk arm above the drought must not let it claim hosts whose
    # volume is fine. The same payload with headroom still reaches the drought
    # arm on its own merits -- 11.1 h with no admission is a real finding -- but
    # it must no longer be reported as a disk fault.
    healthy = {
        **unguarded,
        "status": {**unguarded["status"], "minimum_free_disk_gb": 150},
        "disk": {**unguarded["disk"], "free_gb": 402.0},
    }
    healthy_decision = classify_probe(healthy, stall_seconds=1800)
    assert not healthy_decision.fingerprint.startswith("disk_low:")
    assert "150" not in healthy_decision.reason


def test_the_halt_row_is_read_from_either_stages_schema() -> None:
    """Forward publishes `durable_next_row`; replay publishes `resume_row`.

    `curriculum-supervisor.status.json` is whatever lifecycle event wrote last,
    and reading one key has already produced a vacuous "row None" twice in this
    file's history. Deferred replay is the stage that halts on disk, so the
    replay spelling is the one that must not be dropped.
    """
    parked = {
        **probe("disk_exhausted_unrecoverable"),
        "admissions": {"gate_artifacts": 6, "hours_since_admission": 0.2,
                       "event_counts": {}},
        "status": {
            "state": "disk_exhausted_unrecoverable",
            "phase": "jupyter-scientific-full",
            "minimum_free_disk_gb": 150,
            "resume_row": 229264,
            "disk_reclaim_attempts": 3,
            "disk_reclaimed_bytes": 0,
        },
        "disk": {"free_gb": 149.2, "total_gb": 1023.5, "used_percent": 85.0,
                 "free_inodes": 107322485, "inodes_used_percent": 0.0,
                 "wrapper_enospc": False},
    }
    decision = classify_probe(parked, stall_seconds=1800)
    assert decision.kind == "fix_required", decision
    assert "229264" in decision.reason, decision.reason
    assert "None" not in decision.reason
