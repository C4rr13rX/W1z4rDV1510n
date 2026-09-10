#!/usr/bin/env python3
"""Wake a Claude Code session only when programming-brain work needs it.

The curriculum supervisor, not the agent, owns healthy progress.  This watcher
polls a compact AWS status probe and resumes one explicit Claude Code session
only after the same actionable state is observed repeatedly.  Identical handled
events are deduplicated across watcher restarts.

Runs Claude headless on Opus 5 at xhigh effort with permission checks bypassed,
so an alarm at 03:00 is repaired rather than queued behind a prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.aws.bootstrap_training_host import aws, send_and_wait
except ModuleNotFoundError:
    from bootstrap_training_host import aws, send_and_wait


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE = "i-0d7a6deeb0ead2dfc"
DEFAULT_RUNTIME = "/srv/wizard/runtime/programming-integrated-20260713"
COMPLETE_STATES = {"all_complete", "deferred_replay_complete"}

#: How long the heartbeat probe will wait for the row to move before calling
#: the rate zero. The row advances once per COMMITTED BATCH, not continuously,
#: so this is really "the longest commit period we are willing to mistake for a
#: freeze". Measured 2026-09-10 on a go-systems forward block: 32 rows per
#: commit at 0.355 rows/s is one commit every ~90 s. The previous fixed 6 s
#: sample resolved that ~7 % of the time and published `rows_per_second: 0.0`
#: for the rest, which is what suppressed the convergence annex and woke an
#: agent against a healthy block. Sampling stops early the moment the row
#: moves, so a fast pass pays ~2 s and only a genuinely frozen row pays it all.
HEARTBEAT_SAMPLE_SECONDS = 120.0


@dataclass(frozen=True)
class Decision:
    kind: str
    reason: str
    fingerprint: str = ""


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_activity(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{stamp}] {message.rstrip()}\n")
        stream.flush()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def completion_marker_valid(payload: dict) -> bool:
    brain = payload.get("production_brain") or {}
    course = payload.get("obstacle_course") or {}
    selectors = payload.get("brain_selectors") or {}
    capstone = payload.get("capstone") or {}
    reports = [
        brain.get("report"), course.get("report"),
        selectors.get("report"), capstone.get("report"),
    ]
    return bool(
        payload.get("passed") is True
        and brain.get("passed") is True
        and course.get("passed") == 1000
        and course.get("total") == 1000
        and selectors.get("passed") is True
        and capstone.get("passed") is True
        and capstone.get("independently_verified") is True
        and all(isinstance(path, str) and path.strip() for path in reports)
    )


def source_mtime() -> float:
    """Newest mtime across the modules this process compiled at import."""
    # Resolved through `aws.__module__` rather than a module import, because
    # the transport is imported two different ways depending on whether this
    # runs as a package or from its own directory, and only one of them binds
    # a module name.
    newest = 0.0
    transport = getattr(sys.modules.get(aws.__module__), "__file__", "")
    for module in (__file__, transport):
        try:
            newest = max(newest, Path(module).stat().st_mtime)
        except (OSError, TypeError):
            continue
    return newest


def reload_stale_watcher(loaded_mtime: float, activity_path: Path,
                         dry_run: bool) -> None:
    """Re-exec when this file has changed since the process compiled it.

    THE WATCHER IS SUBJECT TO THE RULE IT ENFORCES. It already reports
    `stale_code_lag` for the supervisor, because "did the bytes land" and "did
    anything reload them" are different questions and only the first leaves an
    artifact you can grep. It exempted itself, and the exemption cost a real
    alarm.

    Measured 2026-09-05. The `gate_artifacts` probe was fixed at 10:06 -- it
    had globbed `*interval_recall*`, a name nothing writes, so it returned 0
    whether the gate had run a thousand times or never. The watcher process
    had started at 09:46. Nineteen minutes older than its own fix, it kept
    counting with the vacuous glob and woke an agent with "the admission gate
    has never produced an artifact ... the gate is not running", while 45
    admission artifacts and 402 rejection records sat in that same tree. The
    diagnosis in the alarm text was exactly backwards, and a session was
    billed to discover the watcher was quoting itself from before the repair.

    Re-exec rather than exit: the scheduled task is disabled, so nothing would
    restart this, and a watcher that exits to be correct supervises nothing.
    `execv` replaces the process in place, keeping the console, the pid file's
    purpose and the poll cadence. It runs only here, at the top of the loop
    with no Claude invocation in flight, so it can never abandon a running
    session -- and the replacement stamps the new mtime, so it settles after
    one hop instead of looping.
    """
    if dry_run:
        return
    current = source_mtime()
    if current <= loaded_mtime:
        return
    append_activity(
        activity_path,
        f"WATCHER RELOAD source changed {current - loaded_mtime:.0f}s after "
        f"this process compiled it; re-exec pid={os.getpid()}",
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


def event_fingerprint(kind: str, probe: dict) -> str:
    status = probe.get("status") or {}
    identity = {
        "kind": kind,
        "host_state": probe.get("host_state"),
        "state": status.get("state"),
        "phase": status.get("phase"),
        "interval_id": status.get("interval_id"),
        "error": status.get("error"),
        "block_target_row": status.get("block_target_row"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"{kind}:{digest}"


#: The supervisor is launched with `--min-free-disk-gb 8`, so it yields rather
#: than trains below that, and the wrapper crash-loops once a 6-byte PID file
#: will not fit. A floor near either number is far too late to act on.
#:
#: The floor is set by the MEASURED burn rate, not by the guard it backs up.
#: Measured 2026-09-10 over a 420 s window during deferred replay: the volume
#: lost 125.4 GB/h while the checkpoint grew 125.4 GB/h -- about 10.8 MB per
#: trained row at 3.2 rows/s, because the `.wbrain` neuron store is append-only
#: with no compactor. At that rate an 8 GB floor is under four minutes of
#: warning and a 20 GB floor is under ten. 48 GB is ~23 minutes at that burn
#: and ~40 days at the 1.2 GB/h five-week average, so it buys time to act
#: without chattering on a healthy host.
DISK_ALARM_FLOOR_GB = 48.0


def disk_exhaustion_fault(probe: dict, *,
                          disk_floor_gb: float = DISK_ALARM_FLOOR_GB) -> str:
    """Name a full volume, or return '' when the volume is not the problem.

    Bytes and inodes both surface as ENOSPC, but only one of them shows up in
    `df -h`, so an inode-exhausted host reads as having free space while every
    write fails. `wrapper_enospc` is carried separately because a reclaim that
    lands between the crash and the probe leaves free bytes beside a service
    that is still failing on the evidence in its own log.
    """
    disk = probe.get("disk") or {}
    if disk.get("error"):
        return ""

    free_gb = disk.get("free_gb")
    if free_gb is not None and float(free_gb) < disk_floor_gb:
        return (
            f"/srv/wizard has {float(free_gb):.2f} GB free, below the "
            f"{disk_floor_gb:.1f} GB alarm floor "
            f"({disk.get('used_percent')}% used)"
        )

    inodes_used = disk.get("inodes_used_percent")
    if inodes_used is not None and float(inodes_used) >= 95.0:
        return (
            f"/srv/wizard has {float(inodes_used):.1f}% of its inodes used "
            f"({disk.get('free_inodes')} free) -- writes fail with ENOSPC even "
            f"though {disk.get('free_gb')} GB is free"
        )

    if disk.get("wrapper_enospc"):
        return (
            "the curriculum wrapper logged 'No space left on device' "
            f"(now {disk.get('free_gb')} GB free) -- reclaim happened after the "
            "failure, so the unit needs restarting"
        )
    return ""


def classify_probe(probe: dict, *, stall_seconds: float,
                   admission_stall_hours: float = 6.0,
                   memory_floor_gb: float = 1.5,
                   disk_floor_gb: float = DISK_ALARM_FLOOR_GB) -> Decision:
    """Classify only deterministic lifecycle evidence, never model quality."""
    host_state = str(probe.get("host_state") or "unknown")
    if host_state != "running":
        return Decision(
            "fix_required", f"AWS host is {host_state}",
            event_fingerprint("fix_required", probe),
        )

    status = probe.get("status") or {}
    state = str(status.get("state") or "missing")
    supervisor_count = int(probe.get("supervisor_count") or 0)
    wrapper_count = int(probe.get("wrapper_count") or 0)
    status_age = float(probe.get("status_age_seconds") or 0.0)
    service_stage = str(probe.get("service_stage") or "")

    if state == "deferred_intervals_pending" or (
        service_stage == "replay"
        and state.startswith("deferred_replay_")
        and state not in COMPLETE_STATES
    ):
        identity = {
            "host_state": host_state,
            "runtime": probe.get("runtime"),
            "status": {"state": "quarantine_ready"},
        }
        return Decision(
            "quarantine_ready",
            "forward harvesting is complete and quarantine replay is ready or active",
            event_fingerprint("quarantine_ready", identity),
        )

    if state in COMPLETE_STATES and supervisor_count == 0:
        return Decision(
            "milestone", f"automated stage reached {state}",
            event_fingerprint("milestone", probe),
        )
    if supervisor_count > 0 or wrapper_count > 0:
        # THE STATUS FILE IS NOT THE HEARTBEAT. The supervisor publishes it
        # only at state transitions, so it goes stale for the whole length of
        # a replay pass while training runs perfectly underneath it -- a
        # 49,152-row pass at ~13 rows/s is ~3,900 s of deliberate silence
        # against a 1,800 s alarm. The worker's progress file is the
        # heartbeat: it carries `durable_next_row` and is rewritten every
        # batch, measured 0.1-2.9 s old throughout.
        #
        # Measured 2026-09-05: `status_age 2630s` on a host whose progress
        # file was 2.3 s old and advancing at 13.1 rows/s across twelve
        # consecutive 30 s windows. Alarming on the status file alone would
        # have woken an agent against a converging interval -- the same
        # false-positive class as the vacuous glob, one file over.
        #
        # Both must be stale to mean control is gone. Liveness comes from the
        # heartbeat; whether that liveness is *converging* is a separate
        # question, already answered by the admission-drought check below.
        throughput = probe.get("throughput") or {}
        heartbeat_age = throughput.get("age_seconds")
        heartbeat_dead = (
            heartbeat_age is None or float(heartbeat_age) > stall_seconds
        )
        if status_age > stall_seconds and heartbeat_dead:
            return Decision(
                "fix_required",
                f"live curriculum control is stale for {status_age:.0f}s and "
                + (
                    "the replay worker has published no progress file"
                    if heartbeat_age is None else
                    f"its progress heartbeat is {float(heartbeat_age):.0f}s old"
                ),
                event_fingerprint("fix_required", probe),
            )
        # Motion is not progress. Measured 2026-09-05: eight clean
        # yield/recycle cycles, seven intervals advanced, 18,568 accepted
        # episodes and "0 failed" on every pass -- with zero admissions for
        # two weeks, because the gate never executed once. A watcher that
        # only checks liveness reports that as healthy and burns the budget.
        admissions = probe.get("admissions") or {}
        gate_artifacts = int(admissions.get("gate_artifacts") or 0)
        since = admissions.get("hours_since_admission")
        counts = admissions.get("event_counts") or {}
        yields = int(counts.get("deferred_replay_resource_yield") or 0)

        before_gate = int(admissions.get("replay_failures_before_gate") or 0)
        at_gate = int(admissions.get("replay_failures_at_gate") or 0)

        # A starved gate is not a failing gate, and the distinction decides
        # the fix: resize the work unit, or repair the capability it tests.
        # Both surface as `deferred_replay_failed`, so read WHERE the
        # transaction died. Every recent failure dying before its gate while
        # the host keeps recycling memory is the work-unit-too-large
        # signature -- the pass consumes exactly the window the gate needs.
        if before_gate >= 2 and at_gate == 0 and yields >= 2:
            return Decision(
                "fix_required",
                f"all {before_gate} replay failures died before reaching "
                f"their admission gate across {yields} resource cycles: the "
                f"work unit is starving its own gate, so nothing can admit",
                event_fingerprint("gate_never_ran", probe),
            )
        if gate_artifacts == 0 and yields >= 2:
            return Decision(
                "fix_required",
                f"admission gate has never produced an artifact across "
                f"{yields} resource cycles: the gate is not running, so no "
                f"interval can ever admit",
                event_fingerprint("gate_never_ran", probe),
            )
        # A FORWARD STAGE CANNOT ADMIT, so the admission clock is the wrong
        # instrument to point at one.
        #
        # Forward harvests rows into deferred intervals; every admission
        # verdict is produced later, by the replay stage that owns those
        # intervals. `hours_since_admission` therefore rises monotonically for
        # the entire length of a forward phase NO MATTER HOW WELL IT IS GOING,
        # and go-systems alone is 328k rows -- days of forward at the measured
        # duty cycle. Left ungated, this alarm re-fires every `retry_cooldown`
        # (1800 s) for that whole span, and each firing wakes a billed agent
        # whose only possible finding is "healthy, wait".
        #
        # That is not a hypothetical. Measured 2026-09-10: 103.7 h of
        # "drought" published as `fix_required` against `go-systems` at row
        # 111,064 of 131,072, advancing 12.0 rows/s, `durable_next_row` equal
        # to `ram_next_row` (zero rollback exposure), 18k rows from its gate.
        # Nothing was wrong. `admission_watchdog.faults` had already been
        # taught this distinction; this emitter had not, so the two halves of
        # the same watchdog disagreed about the same host.
        #
        # The suppression is deliberately narrow. It requires the stage to be
        # `forward` AND the freshest writer to be the forward driver's own
        # progress file AND that file to have actually moved during the
        # sample. A forward block that is genuinely frozen still alarms below,
        # because a forward stage that never reaches its handoff never admits
        # either -- which is the fault this check was built for.
        beat = probe.get("heartbeat") or {}
        forward_rate = beat.get("rows_per_second") or beat.get(
            "accepted_per_second") or 0.0
        forward_advancing = (
            service_stage == "forward"
            and beat.get("source") == "forward_progress"
            and float(forward_rate) > 0.0
        )
        if since is not None and float(since) > admission_stall_hours \
                and forward_advancing:
            curriculum = probe.get("curriculum") or {}
            remaining = curriculum.get("forward_remaining_rows")
            return Decision(
                "healthy",
                f"forward stage {status.get('phase')!r} is harvesting at "
                f"{float(forward_rate):.1f} rows/s (row {beat.get('row')} of "
                f"{status.get('block_target_row')}, "
                f"{remaining} forward rows left); admission belongs to the "
                f"replay stage, so the {float(since):.1f}h since the last one "
                f"is expected, not a fault",
                event_fingerprint("healthy", probe),
            )

        if since is not None and float(since) > admission_stall_hours:
            # Deliberately still a fault, not downgraded by visible progress:
            # the scar behind this rule is eight clean yield/recycle cycles and
            # 18,568 accepted episodes across two weeks with zero admissions,
            # so "rows are moving" is exactly the evidence that fooled a
            # watcher once already. What a converging block does change is the
            # ACTION, and that belongs in the alarm rather than in a probe the
            # woken agent has to run itself. Measured 2026-09-09: the drought
            # was 100.7 h and real, its cause (a registry SchemaError that
            # exit-42 latched the service) had been repaired 40 min earlier,
            # and the block was 1.5 h from its gate -- but the payload carried
            # no rate, so establishing "wait" instead of "repair" cost two SSM
            # round trips. The `retry_cooldown` is 1800 s, so that bill would
            # have been paid roughly three more times before the gate ran.
            # THE ANNEX MUST NOT VANISH WHEN THE RATE IS ZERO. That is exactly
            # the case where the woken agent most needs to know where the block
            # stands, and dropping it publishes an unqualified "no interval
            # admitted for 102.7h" that reads as a dead curriculum. Measured
            # 2026-09-10: the block was at row 82,896 of 131,072 and advancing,
            # but the 6 s sample caught no batch commit, so the payload said
            # only that nothing had admitted for four days -- and establishing
            # "healthy, wait" cost a full diagnostic session. A zero rate and
            # an unknown rate are different facts, and neither one is silence.
            rate = beat.get("rows_per_second")
            row = beat.get("row")
            # THE TARGET HAS TWO NAMES, AND ONLY ONE OF THEM IS EVER SET
            # DURING REPLAY. `block_target_row` is published solely by the
            # forward stage, which passes it to the driver as `--limit-rows`.
            # A deferred replay publishes the same quantity as `end_row`.
            #
            # So the convergence annex below -- added precisely so a drought
            # alarm would carry evidence instead of going out bare -- was
            # live only during forward blocks, the stage where the drought
            # branch is already suppressed because a forward stage does not
            # admit. In replay, the one stage that DOES admit and the only
            # one that reaches this branch, `target` was always None and all
            # three arms fell through to silence.
            #
            # Measured 2026-09-10, a dry-run against the live host published
            # exactly `no interval admitted for 112.6h while the curriculum
            # reports itself active` -- nothing more -- while the replay was
            # at row 86,320 of 131,072 advancing 12.8 rows/s with
            # `durable_next_row == ram_next_row`, about an hour from its
            # gate. At a 1800 s retry cooldown that bills an agent wake-up
            # every half hour against a converging block.
            #
            # Every test covering this annex hand-set `block_target_row`, so
            # the whole suite passed against a payload no replay ever emits.
            #
            # The heartbeat's copy is preferred because the probe resolves it
            # from the durable interval files and the interval id, not from
            # whichever lifecycle event wrote the status file last -- a
            # `resource_node_recycled` record carries no target at all.
            target = (beat.get("block_target_row")
                      or status.get("block_target_row")
                      or status.get("end_row"))
            sampled = beat.get("sample_seconds")
            converging = ""
            if rate and float(rate) > 0 and row is not None and target:
                remaining = int(target) - int(row)
                if remaining > 0:
                    converging = (
                        f"; the live heartbeat ({beat.get('source')}) is "
                        f"advancing {float(rate):.1f} rows/s at row {row} of "
                        f"{target}, so this block reaches its gate in about "
                        f"{remaining / float(rate) / 3600.0:.1f}h -- confirm "
                        f"convergence before repairing anything"
                    )
            elif row is not None and target:
                # Name the state, because settlement, the admission gate and
                # the continuous canary all freeze the row BY DESIGN, and the
                # right action under a by-design freeze is to wait.
                frozen = (
                    f"; the live heartbeat ({beat.get('source')}) sits at row "
                    f"{row} of {target} in state {state!r} and did not move "
                    f"in {float(sampled or 0):.0f}s"
                )
                accepted = beat.get("accepted_per_second")
                if accepted:
                    frozen += (
                        f", though it is still accepting {float(accepted):.1f} "
                        f"episodes/s, so the block is training"
                    )
                converging = frozen + (
                    " -- settlement, the admission gate and the continuous "
                    "canary each freeze the row by design, so confirm the "
                    "freeze is not one of those before repairing anything"
                )
                if beat.get("counter_reset"):
                    # A fourth by-design reason the row reads as frozen, and
                    # the only one where the sample is not merely
                    # uninformative but actively misleading: the worker
                    # restarted mid-sample and its counters began again.
                    converging += (
                        "; the worker's counters went backwards during the "
                        "sample, so it restarted mid-window (a resource yield "
                        "does exactly this) and this rate measures nothing"
                    )
            elif beat.get("no_row_writer"):
                # The third case the two branches above cannot express, and
                # the one that used to fall through to silence: no writer on
                # the host exposes `durable_next_row` at all, so the payload
                # knows nothing about convergence either way. Say that, rather
                # than publishing a bare drought that reads as a dead
                # curriculum -- an unknown rate is not a zero rate.
                converging = (
                    f"; no writer currently exposes a row (freshest is "
                    f"{beat.get('freshest_writer')} at "
                    f"{float(beat.get('age_seconds') or 0):.0f}s), so this "
                    f"payload carries no convergence evidence -- read the "
                    f"replay progress file directly before repairing anything"
                )
            # AND SAY WHETHER THE NAMED FAILURE IS STILL REACHABLE.
            # `last_failure` comes from an append-only ledger that outlives
            # every process that wrote to it, so the newest entry can name a
            # cause repaired generations ago. CLAUDE.md has said "timestamp
            # last_failure against process start before re-debugging it"
            # since 2026-09-10 -- but that instruction was addressed to a
            # reader the payload gave nothing to act on, so obeying it cost a
            # round trip to the host every time. Measured 2026-09-10 on the
            # `quarantine_ready` payload: a worker exit 19.6 h old, quoted
            # against a supervisor 0.14 h old, whose registry SchemaError had
            # been redeployed 16.6 h earlier. Two probes went to prove the
            # ledger text was history.
            stale = admissions.get("last_failure_predates_supervisor")
            if stale and admissions.get("last_failure"):
                converging += (
                    f"; the named last_failure is "
                    f"{float(admissions.get('last_failure_age_hours') or 0):.1f}h "
                    f"old and predates this supervisor "
                    f"({float(admissions.get('supervisor_age_hours') or 0):.1f}h "
                    f"old), so it describes a previous generation -- confirm it "
                    f"is still reachable before re-debugging it"
                )
            return Decision(
                "fix_required",
                f"no interval admitted for {float(since):.1f}h while the "
                f"curriculum reports itself active" + converging,
                event_fingerprint("no_admission", probe),
            )

        memory = probe.get("memory") or {}
        available = float(memory.get("available_gb") or 0.0)
        if available and available < memory_floor_gb:
            return Decision(
                "fix_required",
                f"host memory available is {available:.2f} GB, below the "
                f"{memory_floor_gb:.1f} GB alarm floor",
                event_fingerprint("memory_low", probe),
            )
        disk_fault = disk_exhaustion_fault(probe, disk_floor_gb=disk_floor_gb)
        if disk_fault:
            return Decision(
                "fix_required", disk_fault, event_fingerprint("disk_low", probe),
            )
        return Decision("healthy", f"automation owns {state}")

    # No supervisor and no wrapper. Name the CAUSE if the host can still state
    # it: a full volume stops the wrapper before it ever launches a supervisor,
    # so "nothing owns this state" is the symptom of the fault below, not a
    # separate diagnosis. Checked here as well as on the owned path because the
    # two arms are reached under opposite process censuses.
    disk_fault = disk_exhaustion_fault(probe, disk_floor_gb=disk_floor_gb)
    if disk_fault:
        return Decision(
            "fix_required",
            f"{disk_fault}; nothing owns terminal state {state} because the "
            f"wrapper cannot write its runtime identity files",
            event_fingerprint("disk_low", probe),
        )
    return Decision(
        "fix_required",
        f"no curriculum supervisor or wrapper owns terminal state {state}",
        event_fingerprint("fix_required", probe),
    )


def remote_probe(profile: str, instance_id: str, runtime: str) -> dict:
    instance = aws(
        profile, "ec2", "describe-instances", "--instance-ids", instance_id,
        "--query", "Reservations[0].Instances[0].State.Name", "--output", "text",
    ).stdout.strip()
    if instance != "running":
        return {"host_state": instance, "observed_unix": time.time()}

    heartbeat_sample_seconds = HEARTBEAT_SAMPLE_SECONDS
    remote = f"""python3 - <<'PY'
import json, os, pathlib, re, sys, time
runtime = pathlib.Path({runtime!r})
sys.path.insert(0, '/srv/wizard/project')
from scripts.programming_curriculum_supervisor import (
    curriculum_phases, read_json, unresolved_deferred_intervals,
)
try:
    status = json.loads((runtime / 'curriculum-supervisor.status.json').read_text())
except Exception as exc:
    status = {{'state': 'missing', 'error': str(exc)}}
supervisors = wrappers = workers = 0
# When the CURRENT supervisor generation started. `last_failure` is read from
# an append-only ledger that outlives every process which wrote to it, so the
# age of the newest failure says nothing on its own about whether that failure
# is still reachable. Capturing this here is what lets the payload answer it.
supervisor_started_unix = 0.0
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command = path.read_bytes().replace(b'\\0', b' ').decode(errors='replace')
        process_name = (path.parent / 'comm').read_text().strip().lower()
    except OSError:
        continue
    if (process_name.startswith('python')
            and 'programming_curriculum_supervisor.py' in command
            and {runtime!r} in command):
        supervisors += 1
        try:
            # /proc/<pid> is created when the process is, so its mtime is the
            # start time. Take the NEWEST across generations: a stale entry
            # would understate the age and re-arm the trap this closes.
            supervisor_started_unix = max(
                supervisor_started_unix, path.parent.stat().st_mtime)
        except OSError:
            pass
    if process_name == 'bash' and 'run_programming_curriculum_service.sh' in command:
        wrappers += 1
    if (process_name.startswith('python')
            and 'tools.training_standard.drive_corpora_brain' in command
            and {runtime!r} in command):
        workers += 1
updated = float(status.get('updated_unix') or 0.0)
try:
    service_stage = (runtime / 'curriculum-service-supervisor.stage').read_text().strip()
except OSError:
    service_stage = ''
include_seed = any(
    (runtime / f'{{name}}.progress.json').is_file()
    for name in ('canonical-algorithms', 'gsm8k-domain-safe')
)
phases = curriculum_phases(pathlib.Path('/srv/wizard/corpora'), include_seed)
processed = 0
phase_rows = {{}}
for phase in phases:
    progress = read_json(runtime / f'{{phase.name}}.progress.json')
    durable = min(phase.rows, max(0, int(progress.get('durable_next_row') or 0)))
    phase_rows[phase.name] = durable
    processed += durable
total = sum(phase.rows for phase in phases)
intervals = {{}}
for event in unresolved_deferred_intervals(runtime):
    phase = str(event['phase'])
    start = max(0, int(event['start_row']))
    end = min(phase_rows.get(phase, 0), int(event['end_row']))
    if end > start:
        intervals.setdefault(phase, []).append((start, end))
deferred = 0
for spans in intervals.values():
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    deferred += sum(end - start for start, end in merged)
# --- admission + memory metrics -------------------------------------------
# Every one of these was green for 16 h while nothing admitted, so the probe
# must report the ones that actually distinguish progress from motion.
#
# WHEN an admission happened is the ledger's answer, not the health log's.
# `deferred_replay_admitted` is appended several steps AFTER the interval is
# durably marked resolved -- after `accept_last_good_guard`, which raises --
# so a completed admission can leave a resolved row behind and never emit its
# event. Measured 2026-09-05: three intervals resolved 71.8 h, 89.9 h and
# 112.1 h ago, each reading "final-brain deferred replay passed comprehensive
# admission" and each backed by an enterprise gate artifact reading
# passed=True, while the newest event was 351.7 h old. Reporting the event
# age as the drought overstated it fivefold and woke the agent for a stall
# that had ended three days earlier.
#
# Both are reported. Their divergence is not noise: it means an admission
# committed whose event did not fire, which is worth seeing on its own.
resolved_unix = 0.0
resolved_reason = ''
try:
    with (runtime / 'curriculum-deferred-intervals.jsonl').open(
            encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get('status') != 'resolved':
                continue
            when = float(row.get('updated_unix') or 0.0)
            if when >= resolved_unix:
                resolved_unix = when
                resolved_reason = str(row.get('reason') or '')[:120]
except OSError:
    pass

health = runtime / 'curriculum-health.jsonl'
kinds = {{}}
admitted_unix = 0.0
last_yield = {{}}
recent_fail = ''
recent_fail_suites = []
recent_fail_unix = 0.0
worker_killed = 0
gate_reached = 0

def summarize_failure(error):
    # KEEP THE END OF A TRACEBACK. Truncating head-first is exactly backwards
    # for the two shapes this field actually carries.
    #
    # `replay_worker_failure` appends up to 4000 bytes of the worker's stderr
    # after the log's path, precisely so the reason travels with its address
    # (44e496b, "Carries the reason, not just its address"). But the runtime
    # path alone is 135 characters, so a flat [:180] left 45 characters for
    # that tail -- and a traceback's one informative line, the exception type
    # and message, is its LAST. Measured 2026-09-10: this wake-up reported
    # `deferred replay worker exited 1; stderr=<path>` and nothing more; the
    # file held a SchemaError naming `category='systems_programming_go'`,
    # already repaired 3 h before the current supervisor started. Recovering
    # one line that was sitting in the payload's own source cost a full
    # round-trip to the host.
    #
    # Same defect on the other shape: an `enterprise regression` blob spends
    # its first 180 characters on tick and structure counts, so the failing
    # suite names -- the only part that says WHICH capability regressed --
    # were always past the cut. That is the reporting half of the lesson in
    # CLAUDE.md that an 11/12 gate names no suite in the ledger.
    error = str(error or '')
    if len(error) <= 900:
        return error
    return error[:260] + '\\n...[' + str(len(error) - 860) + ' chars]...\\n' + error[-600:]

def failing_suites(error):
    # The per-suite verdict is embedded in the repr of the gate report. Pull
    # it out rather than making every reader re-derive it from prose.
    return sorted(set(re.findall(
        r"'name':\\s*'([^']+)'[^{{}}]*?'passed':\\s*False", str(error or ''))))[:8]

try:
    for line in health.read_text(errors='replace').splitlines()[-4000:]:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        kind = str(ev.get('kind') or '')
        kinds[kind] = kinds.get(kind, 0) + 1
        when = float(ev.get('updated_unix') or 0.0)
        if kind == 'deferred_replay_admitted':
            admitted_unix = max(admitted_unix, when)
        elif kind == 'deferred_replay_resource_yield':
            last_yield = {{
                'unix': when,
                'before_gb': round(float(ev.get('available_bytes_before') or 0) / 2**30, 2),
                'after_gb': round(float(ev.get('available_bytes_after') or 0) / 2**30, 2),
            }}
        elif kind == 'deferred_replay_failed':
            full_fail = str(ev.get('error') or '')
            recent_fail = summarize_failure(full_fail)
            recent_fail_suites = failing_suites(full_fail)
            recent_fail_unix = when
            # Which half of the transaction died? A worker killed by the
            # memory guard never reached the gate; anything else means the
            # gate ran and returned a verdict. Both arrive as the same
            # `deferred_replay_failed` kind, so the count alone cannot tell
            # a starved gate from a rejecting one -- and they need opposite
            # fixes (resize the work unit vs. repair the capability).
            # Classify on the WHOLE error, never on the display summary --
            # otherwise trimming the field for readability silently moves the
            # worker/gate split, and the two need opposite fixes.
            if 'worker exited' in full_fail:
                worker_killed += 1
            else:
                gate_reached += 1
except OSError:
    pass

# A gate that never RUNS logs neither pass nor failure, so only its artifacts
# can tell "failing" from "never executed".
#
# COUNT A NAME SOMETHING ACTUALLY WRITES. This globbed '*interval_recall*',
# which nothing in the codebase ever creates: `interval_recall` is a health
# event *kind* built as f'{{gate_kind}}_infrastructure_retry' and a JSON *key*
# inside deferred-replay-<digest>.admission.json. The glob therefore matched
# zero files whether the gate had run a thousand times or never at all.
# Measured 2026-09-05 on the training host: it reported gate_artifacts 0 while
# 45 admission artifacts and 402 rejection records sat in the same tree, and
# that vacuous 0 was quoted as evidence the gate had never executed.
#
# A passing gate publishes deferred-replay-<digest>.admission.json; a gate that
# ran and rejected leaves deferred/<digest>/evidence/<attempt>/failure.json.
gate_artifacts = len(list(runtime.glob('deferred-replay-*.admission.json')))
gate_rejections = len(list(runtime.glob('deferred/*/evidence/*/failure.json')))

meminfo = {{}}
try:
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, _, rest = line.partition(':')
        meminfo[key] = int(rest.split()[0])
except Exception:
    pass

# The payload reported memory and never disk, so a FULL VOLUME was invisible to
# the classifier. Measured 2026-09-10: /srv/wizard hit 20 KB free on 1.0 TB, the
# wrapper could not write its 6-byte node.pid, systemd restarted it 115 times at
# RestartSec=10, and the census landed between restarts -- so the alarm read
# "no curriculum supervisor or wrapper owns terminal state
# deferred_replay_resource_yield", naming the symptom while the cause was one
# statvfs call away. `admission_watchdog.faults` already had `disk_low`; this
# emitter did not, which is the two-emitter drift CLAUDE.md names.
disk = {{}}
try:
    stat = os.statvfs(str(runtime))
    disk = {{
        'free_gb': round(stat.f_bavail * stat.f_frsize / 1e9, 2),
        'total_gb': round(stat.f_blocks * stat.f_frsize / 1e9, 2),
        'used_percent': (
            round(100.0 * (1.0 - stat.f_bavail / stat.f_blocks), 1)
            if stat.f_blocks else None),
        # Inode exhaustion presents identically to byte exhaustion (ENOSPC) but
        # df -h shows free space, so report both or the next outage reads as a
        # contradiction.
        'free_inodes': stat.f_favail,
        'inodes_used_percent': (
            round(100.0 * (1.0 - stat.f_favail / stat.f_files), 1)
            if stat.f_files else None),
    }}
except Exception as exc:
    disk = {{'error': f'{{type(exc).__name__}}: {{exc}}'}}

# A crash-looping wrapper writes its reason to the unit's stderr log. Carry the
# ENOSPC verdict itself rather than making the reader infer it from free_gb,
# because a reclaim that lands between the crash and the probe leaves free space
# beside a service that is still failing.
wrapper_enospc = False
try:
    log = runtime / 'curriculum-service.stderr.log'
    with log.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        tail = handle.read().decode('utf-8', 'replace')
    wrapper_enospc = 'No space left on device' in tail
except Exception:
    pass
disk['wrapper_enospc'] = wrapper_enospc
brain_rss_kb = 0
brain_age_s = 0.0
for path in pathlib.Path('/proc').glob('[0-9]*/comm'):
    try:
        if path.read_text().strip() != 'w1z4rd_brain_se':
            continue
        pid = path.parent.name
        for line in (path.parent / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                brain_rss_kb = int(line.split()[1])
                break
        stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
        clk = os.sysconf('SC_CLK_TCK')
        boot = 0.0
        for ln in pathlib.Path('/proc/stat').read_text().splitlines():
            if ln.startswith('btime'):
                boot = float(ln.split()[1]); break
        brain_age_s = max(0.0, time.time() - (boot + float(stat[19]) / clk))
    except Exception:
        continue

status_file = runtime / 'curriculum-supervisor.status.json'
progress_file = max(runtime.glob('deferred-replay-*.progress.json'),
                    key=lambda f: f.stat().st_mtime, default=None)
# The forward driver writes a THIRD file, its own <phase>.progress.json --
# the path is on its command line as --progress-path -- and it matches neither
# candidate above. Measured 2026-09-10 during a go-systems forward block:
# go-systems.progress.json was 7.6 s old and advancing 50192 -> 50224 while
# the status file sat 718 s stale at 49152, so the freshest-writer rule picked
# the status file and published rows_per_second 0.0 on a healthy run.
forward_file = max(
    (f for f in runtime.glob('*.progress.json')
     if not f.name.startswith('deferred-replay-')),
    key=lambda f: f.stat().st_mtime, default=None)


def _file_age(path):
    try:
        return round(time.time() - path.stat().st_mtime, 1)
    except Exception:
        return None


# Per-second change, or None when the counter went BACKWARDS.
#
# Both counters sampled here live in the worker's progress file and restart
# with the worker, so a sample straddling a `deferred_replay_resource_yield`
# sees the value fall rather than rise. That is a reset, not a negative rate,
# and the difference matters because every consumer downstream reads a number
# here as a measurement of throughput.
#
# Comments rather than a docstring: this body is interpolated into a
# triple-quoted f-string, so a `\"\"\"` here would close the probe source.
def _rate(first, last, elapsed):
    if first is None or last is None:
        return None
    if last < first:
        return None
    return round((last - first) / elapsed, 2)


def _row_now(path):
    try:
        row = json.loads(path.read_text()).get('durable_next_row')
        return int(row) if row is not None else None
    except Exception:
        return None


# The second liveness signal in the same file. A row that advances while
# `accepted_episodes` does not is a driver skipping rows rather than training
# on them, and the row count alone cannot tell those two apart.
def _accepted_now(path):
    try:
        value = json.loads(path.read_text()).get('accepted_episodes')
        return int(value) if value is not None else None
    except Exception:
        return None


# WHERE THE BLOCK'S TARGET ROW ACTUALLY LIVES.
#
# `curriculum-supervisor.status.json` is not one schema, it is whatever the
# last lifecycle event published. Forward blocks write `block_target_row`;
# a deferred replay writes `start_row`/`resume_row`/`end_row`; and a
# `resource_node_recycled` record -- measured on this host at 2026-09-10,
# mid-replay -- writes `trained_rows` and a `topology` dict and NO target of
# any kind. So a consumer that reads the target from that file gets an answer
# that depends on which event happened to land last, which is why the
# convergence annex went silent on a converging block.
#
# The interval being replayed is durable state, so read it from the files that
# define it and fall back to the interval id, which encodes phase:start:end.
def _block_target(status):
    for key in ('block_target_row', 'end_row'):
        value = status.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    candidates = [runtime / 'deferred-replay-active.json']
    identifiers = [status.get('interval_id')]
    for path in candidates:
        try:
            body = json.loads(path.read_text())
        except Exception:
            continue
        for scope in (body.get('interval') or {{}}, body):
            if not isinstance(scope, dict):
                continue
            if scope.get('end_row') is not None:
                try:
                    return int(scope['end_row'])
                except (TypeError, ValueError):
                    pass
            identifiers.append(scope.get('interval_id'))
    for identifier in identifiers:
        parts = str(identifier or '').split(':')
        if len(parts) >= 3 and parts[-1].isdigit():
            return int(parts[-1])
    return None


throughput = {{}}
try:
    if progress_file is not None:
        prog = json.loads(progress_file.read_text())
        throughput = {{
            'progress_file': progress_file.name,
            'accepted_episodes': prog.get('accepted_episodes'),
            'durable_next_row': prog.get('durable_next_row'),
            'batch_seconds_ema': prog.get('batch_seconds_ema'),
            'current_batch_size': prog.get('current_batch_size'),
            'age_seconds': _file_age(progress_file),
        }}
except Exception:
    pass

# THE REPLAY PROGRESS FILE IS NOT THE ONLY HEARTBEAT, AND IS OFTEN NOT THE
# LIVE ONE. THREE different writers advance rows: the replay worker rewrites
# deferred-replay-*.progress.json every batch, the forward driver rewrites its
# own <phase>.progress.json every batch, and the supervisor rewrites
# curriculum-supervisor.status.json -- but only between batches, so it FREEZES
# for minutes during a canary, a settlement or a gate. `throughput` above reads
# only the replay file, so during a forward block it publishes whatever the
# last replay pass left behind. Measured 2026-09-09: that file was 100.7 h old
# and carried durable_next_row 201344 / accepted_episodes 5168, printed beside
# a live status at row 16416. Nothing in the payload marked it stale, so the
# only available readings of a forward block advancing at 15.3 rows/s were
# "the run went backwards 185k rows" or "throughput has flatlined for 4 days".
#
# Naming the status file as the forward heartbeat fixed that case and broke
# this one. Measured 2026-09-10, mid `continuous_canary`: status 718 s stale at
# row 49152 while go-systems.progress.json was 7.6 s old and climbing
# 50192 -> 50224. The rule is the freshest writer among ALL THREE, never a
# file chosen by which stage is believed to be running.
#
# So take the freshest writer rather than a fixed one, and sample it twice:
# the payload has never carried a RATE, and a rate is what separates a
# converging block from a live process that is not training. Reporting only --
# a zero rate is deliberately NOT a fault here, because settlement and the
# admission gate both freeze the row for minutes by design, and alarming on
# that is the false-positive class this file already carries two scars from.
#
# THE SAMPLE MUST BE LONGER THAN ONE BATCH COMMIT, or the rate is 0.0 on a
# perfectly healthy block. The row does not advance continuously: the driver
# rewrites the file once per committed batch, so between commits it is
# BYTE-IDENTICAL and a short sample sees nothing. Measured 2026-09-10 on a
# go-systems forward block: 32 rows per commit at 0.355 rows/s is one commit
# every ~90 s, so a fixed 6 s window observed movement roughly 7 % of the
# time. The other 93 % published `rows_per_second: 0.0`, which suppressed the
# convergence annex in `decide()` -- the one thing in the payload that tells a
# woken agent to WAIT rather than repair -- and woke an agent against a block
# that was converging normally, 37 h from its gate.
#
# So sample adaptively: poll until the row actually changes, bounded. A fast
# replay pass returns in one interval and costs nothing; a slow forward block
# pays up to the bound and returns a real rate instead of a misleading zero.
# The bound is what a zero now MEANS -- "did not move in 120 s", not "did not
# move in 6 s" -- so it must stay well above the slowest observed commit
# period rather than being tuned down to save probe time.
#
# ...BUT THE FRESHEST WRITER IS NOT ALWAYS A WRITER OF ROWS. Selecting purely
# on mtime picks `curriculum-supervisor.status.json` whenever the supervisor
# happens to have touched it most recently -- and during a REPLAY that file
# carries `resume_row`/`end_row`, not `durable_next_row`. `_row_now()` returns
# None, so the sample loop waits the full bound for a row that file will never
# contain and publishes `row: null, rows_per_second: null`.
#
# That is worse than a wrong rate. `classify_probe()` builds its convergence
# annex only when `row is not None`, so BOTH branches fall through and the
# alarm goes out as a bare "no interval admitted for 111.6h" -- the exact
# silence the comment above that annex forbids ("A zero rate and an unknown
# rate are different facts, and neither one is silence"). Measured 2026-09-10:
# published against a replay converging at 14.0 rows/s with zero rollback
# exposure, 82,272 rows from its gate.
#
# So require a row. Choose the freshest file that HAS one, and publish how far
# behind the freshest writer overall it is, because that lag is what stops a
# leftover from masquerading as live -- the 100.7 h replay file above still
# loses to a live status whenever the status file carries a row, and when it
# does not, `row_source_lag_seconds` says so out loud instead of implying
# currency by silence.
heartbeat = {{}}
try:
    ages = [(age, name, path) for age, name, path in (
        (_file_age(status_file), 'status', status_file),
        (_file_age(progress_file) if progress_file is not None else None,
         'replay_progress', progress_file),
        (_file_age(forward_file) if forward_file is not None else None,
         'forward_progress', forward_file),
    ) if age is not None]
    rowed = [item for item in ages if _row_now(item[2]) is not None]
    if ages and not rowed:
        freshest = min(ages, key=lambda item: item[0])
        heartbeat = {{
            'source': None, 'file': freshest[2].name,
            'age_seconds': freshest[0], 'row': None,
            'rows_per_second': None, 'accepted_episodes': None,
            'accepted_per_second': None, 'sample_seconds': 0.0,
            # Name the blindness rather than reporting a null that reads as a
            # measured zero: no writer on this host exposes durable_next_row.
            'no_row_writer': True,
            'freshest_writer': freshest[1],
            'block_target_row': _block_target(status),
        }}
    if rowed:
        age, source, path = min(rowed, key=lambda item: item[0])
        freshest_age = min(item[0] for item in ages)
        first_row, first_at = _row_now(path), time.time()
        first_accepted = _accepted_now(path)
        last_row, last_at = first_row, first_at
        last_accepted = first_accepted
        deadline = first_at + {heartbeat_sample_seconds:.1f}
        while time.time() < deadline:
            time.sleep(2.0)
            last_row, last_at = _row_now(path), time.time()
            last_accepted = _accepted_now(path)
            if last_row is not None and last_row != first_row:
                break
        elapsed = max(1e-6, last_at - first_at)
        heartbeat = {{
            'source': source,
            'file': path.name,
            'age_seconds': age,
            'row': last_row,
            # Published beside the row it is measured against, so a consumer
            # never has to guess which of the status file's several schemas
            # happens to be live.
            'block_target_row': _block_target(status),
            'sample_seconds': round(elapsed, 1),
            # A NEGATIVE RATE IS A COUNTER RESET, NOT A MEASUREMENT.
            # Both counters live in the worker's progress file and restart
            # with the worker. A `deferred_replay_resource_yield` kills and
            # relaunches it, so a sample straddling one sees the value fall
            # to near zero. Measured 2026-09-10 against the live host during
            # a yield: `accepted_episodes` went 712 -> 8 and the payload
            # published `accepted_per_second: -42.4`, which the drought annex
            # would have rendered as "still accepting -42.4 episodes/s, so
            # the block is training". Publishing None and saying the counter
            # reset is the honest reading -- an unknown rate is not a zero
            # rate, and it is certainly not a negative one.
            'rows_per_second': _rate(first_row, last_row, elapsed),
            # Whether the rows being consumed are being LEARNED.
            'accepted_episodes': last_accepted,
            'accepted_per_second': _rate(first_accepted, last_accepted, elapsed),
            'counter_reset': (
                _rate(first_row, last_row, elapsed) is None
                and first_row is not None and last_row is not None
            ) or (
                _rate(first_accepted, last_accepted, elapsed) is None
                and first_accepted is not None and last_accepted is not None
            ),
            # How far behind the freshest writer this row-carrying file sits.
            # 0 means it IS the freshest. A large value means the only file
            # exposing a row is a leftover, and its rate describes the past.
            'row_source_lag_seconds': round(age - freshest_age, 1),
        }}
        if throughput and source != 'replay_progress':
            # Say it in the payload, not just in this comment.
            throughput['is_live_heartbeat'] = False
            throughput['superseded_by'] = source
except Exception:
    pass
curriculum = {{
    'total_rows': total,
    'durable_processed_rows': processed,
    'accepted_rows': max(0, processed - deferred),
    'deferred_rows': deferred,
    'forward_remaining_rows': max(0, total - processed),
    'minimum_outstanding_rows': max(0, total - processed) + deferred,
    'include_seed_corpora': include_seed,
}}
print(json.dumps({{
    'host_state': 'running', 'runtime': str(runtime), 'status': status,
    'supervisor_count': supervisors, 'wrapper_count': wrappers,
    'worker_count': workers, 'service_stage': service_stage,
    'curriculum': curriculum,
    'admissions': {{
        'event_counts': kinds,
        'last_admitted_unix': resolved_unix or admitted_unix,
        'hours_since_admission': (
            round((time.time() - resolved_unix) / 3600.0, 1)
            if resolved_unix else None),
        'last_resolved_reason': resolved_reason,
        'hours_since_admission_event': (
            round((time.time() - admitted_unix) / 3600.0, 1)
            if admitted_unix else None),
        'gate_artifacts': gate_artifacts,
        'gate_rejections': gate_rejections,
        'replay_failures_before_gate': worker_killed,
        'replay_failures_at_gate': gate_reached,
        'last_resource_yield': last_yield,
        'last_failure': recent_fail,
        # Which suites the named failure actually names. Empty means the
        # failure was not a suite verdict (a worker exit, say), not that
        # every suite passed.
        'last_failure_suites': recent_fail_suites,
        # WHEN, not just what. `curriculum-health.jsonl` is append-only and
        # outlives every supervisor generation, so the newest failure in it
        # can be arbitrarily old and already repaired. CLAUDE.md has said
        # "timestamp last_failure against process start before re-debugging
        # it" since 2026-09-10, but the payload carried no timestamp, so the
        # only way to act on that instruction was a round trip to the host --
        # and this wake-up spent two of them re-deriving it. Measured
        # 2026-09-10 on THIS payload: `last_failure` was a worker exit 19.6 h
        # old, named against a supervisor 0.14 h old, whose cause (a registry
        # SchemaError) had been redeployed 16.6 h earlier. Publishing the age
        # beside the text is what makes "is this still reachable?" answerable
        # without leaving the payload.
        'last_failure_unix': recent_fail_unix or None,
        'last_failure_age_hours': (
            round((time.time() - recent_fail_unix) / 3600.0, 2)
            if recent_fail_unix else None),
        # True means no process now running could have produced it. That is
        # not proof the cause is fixed -- an unreached code path also never
        # fails -- but it does mean the ledger text is evidence about a
        # PREVIOUS generation, and re-debugging it starts from the wrong end.
        'last_failure_predates_supervisor': (
            bool(recent_fail_unix and supervisor_started_unix
                 and recent_fail_unix < supervisor_started_unix)
            if (recent_fail_unix and supervisor_started_unix) else None),
        'supervisor_started_unix': supervisor_started_unix or None,
        'supervisor_age_hours': (
            round((time.time() - supervisor_started_unix) / 3600.0, 2)
            if supervisor_started_unix else None),
    }},
    'disk': disk,
    'memory': {{
        'available_gb': round(meminfo.get('MemAvailable', 0) / 2**20, 2),
        'total_gb': round(meminfo.get('MemTotal', 0) / 2**20, 2),
        'swap_total_gb': round(meminfo.get('SwapTotal', 0) / 2**20, 2),
        'brain_rss_gb': round(brain_rss_kb / 2**20, 2),
        'brain_age_seconds': round(brain_age_s, 1),
    }},
    'throughput': throughput,
    'heartbeat': heartbeat,
    'status_age_seconds': max(0.0, time.time() - updated) if updated else 1e99,
    'observed_unix': time.time(),
}}, separators=(',', ':')))
PY"""
    invocation = send_and_wait(
        # The heartbeat sample can hold the probe for HEARTBEAT_SAMPLE_SECONDS
        # on a frozen row, so the transport bound has to clear that by enough
        # for the rest of the probe. A probe that times out publishes nothing,
        # which reads exactly like a dead host.
        profile, instance_id, [remote],
        int(HEARTBEAT_SAMPLE_SECONDS) + 300,
        comment="Probe Wizard programming brain watchdog state",
    )
    output = str(invocation.get("StandardOutputContent") or "").strip().splitlines()
    if not output:
        raise RuntimeError("AWS programming-brain probe returned no output")
    return json.loads(output[-1])


def agent_prompt(decision: Decision, probe: dict) -> str:
    return f"""A deterministic Wizard Vision programming-brain watchdog woke this
same session because automation reached an actionable state. This is not a
routine progress poll.

Event: {decision.kind}: {decision.reason}
Evidence:
{json.dumps(probe, indent=2, sort_keys=True)}

Continue the full senior-software-engineer brain objective from authoritative
repository and AWS state. Diagnose and take concrete action until autonomous
training is healthy again, the next required stage is running, or genuinely
new user authority is required. Preserve neuron-scoped serialization and every
accept/quarantine/replay invariant. Run proportionate tests, update the brain
configuration/reproduction documentation with durable lessons, and commit and
push all non-generated work. Do not merely report status and do not reinterpret
an expected quarantine as overall completion.

Completion still requires a deterministic obstacle course of 1,000 distinct,
representative enterprise-software tasks. Group failures by capability, repair
them with appropriately licensed training material or a causal architecture
fix, and rerun affected tasks plus full retention until all 1,000 pass. Then
integrate the independent Wizard brain selectors in CoolCryptoUtilities,
configure C0D3R V2 as Brand Dozer's agent using this brain, and complete the
Multi-Scale Robot World project. Judge that capstone independently and
critically: it is not complete until it is a world-class 3D robot-design system
with credible real-world physics and fabrication-ready designs suitable for 3D
printing. Never accept the brain's own declaration of completion as evidence.
Follow `docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md`. Publish its generated
completion marker only after every referenced authoritative report exists and
passes a fresh requirement-by-requirement audit.
"""



def resolve_claude() -> str | None:
    """Absolute path to the Claude Code launcher, or None if not installed.

    npm installs claude on Windows as claude.cmd/claude.ps1 shims rather than
    a real binary, and subprocess without shell=True only runs actual
    binaries, so a bare "claude" raises WinError 2. Resolve the shim
    explicitly, mirroring how the Codex launcher was located before it.
    """
    direct = shutil.which("claude")
    if direct:
        return direct
    for ext in (".cmd", ".exe", ".ps1", ".bat"):
        found = shutil.which("claude" + ext)
        if found:
            return found
    return None


def format_claude_event(payload: dict) -> str:
    """Render one stream-json line as a single activity log entry.

    Claude Code's stream-json is shaped differently from Codex's event feed:
    assistant/user turns carry a `message` with a `content` list of typed
    blocks, and the run ends with a `result` envelope.
    """
    kind = str(payload.get("type") or "")
    if kind == "system":
        sub = str(payload.get("subtype") or "")
        model = payload.get("model") or ""
        return f"CLAUDE system {sub} {model}".rstrip()
    if kind == "result":
        status = "ERROR" if payload.get("is_error") else "ok"
        turns = payload.get("num_turns")
        cost = payload.get("total_cost_usd")
        parts = [f"CLAUDE result [{status}]"]
        if turns is not None:
            parts.append(f"turns={turns}")
        if isinstance(cost, (int, float)):
            parts.append(f"cost=${cost:.2f}")
        return " ".join(parts)
    if kind not in {"assistant", "user"}:
        return ""
    content = (payload.get("message") or {}).get("content")
    if not isinstance(content, list):
        return ""
    lines = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        if btype == "text":
            text = str(block.get("text") or "").strip().replace("\r", "")
            if text:
                lines.append(f"CLAUDE MESSAGE {text}")
        elif btype == "tool_use":
            name = str(block.get("name") or "tool")
            args = block.get("input") or {}
            detail = ""
            if isinstance(args, dict):
                # Surface the part a human would actually want in a log.
                for key in ("command", "file_path", "pattern", "description"):
                    if args.get(key):
                        detail = str(args[key]).replace("\n", " ")[:200]
                        break
            lines.append(f"CLAUDE TOOL {name} {detail}".rstrip())
        elif btype == "tool_result":
            if block.get("is_error"):
                text = str(block.get("content") or "")[:200].replace("\n", " ")
                lines.append(f"CLAUDE TOOL-ERROR {text}".rstrip())
    return "\n".join(lines)


def invoke_claude(session_id: str, decision: Decision, probe: dict,
                  log_dir: Path, activity_path: Path, *,
                  model: str = "opus", effort: str = "xhigh") -> int:
    """Wake one Claude Code session to repair an actionable training fault.

    Runs headless (`-p`) with stream-json so every tool call and message lands
    in the activity log. `--resume` keeps one continuous session so the agent
    retains what it already learned about this brain across alarms; if that
    session is gone, fall back to a fresh run rather than losing the alarm.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stdout_path = log_dir / f"claude-{stamp}.jsonl"
    stderr_path = log_dir / f"claude-{stamp}.stderr.log"
    executable = resolve_claude()
    if executable is None:
        append_activity(
            activity_path,
            "CLAUDE UNAVAILABLE: no claude launcher on PATH; install with "
            "`npm i -g @anthropic-ai/claude-code`. Supervision continues; "
            "alarms are logged but cannot wake Claude.",
        )
        return 127

    base = [
        executable, "-p",
        "--model", model,
        "--effort", effort,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    command = base + (["--resume", session_id] if session_id else [])
    append_activity(
        activity_path,
        f"ALARM waking Claude ({model}/{effort}): {decision.reason}",
    )
    returncode = _run_claude(command, decision, probe, stdout_path,
                             stderr_path, activity_path)
    # A negative code (POSIX) or 128+n (shell convention) means the child was
    # SIGNALLED, not that it failed. Verified 2026-09-05: stopping the parent
    # task produced 143 (SIGTERM) on a run that had been working correctly,
    # and the watcher reported it beside the 127 it uses for "no launcher on
    # PATH" -- a deliberate stop reading as an install problem. Retrying one
    # would also wake a second agent on a fault that never existed.
    if returncode < 0 or returncode in (130, 143):
        append_activity(
            activity_path,
            f"CLAUDE STOPPED by signal (returncode={returncode}); "
            f"not retrying -- this is a deliberate stop, not a failure.",
        )
        return returncode
    if returncode != 0 and session_id:
        # A stale/absent session id must not swallow the alarm: retry once
        # without --resume so the fault still gets worked.
        stale = stderr_path.read_text(encoding="utf-8", errors="replace")[-400:]
        append_activity(
            activity_path,
            f"CLAUDE resume failed (rc={returncode}); retrying as a fresh "
            f"session. stderr tail: {stale.strip()[-200:]}",
        )
        returncode = _run_claude(base, decision, probe, stdout_path,
                                 stderr_path, activity_path)
    append_activity(activity_path, f"CLAUDE EXIT returncode={returncode}")
    return returncode


def _run_claude(command: list[str], decision: Decision, probe: dict,
                stdout_path: Path, stderr_path: Path,
                activity_path: Path) -> int:
    with stdout_path.open("a", encoding="utf-8") as stdout, \
            stderr_path.open("a", encoding="utf-8") as stderr:
        # `text=True` WITHOUT `encoding` decodes with the Windows locale codec.
        # Claude's stream-json is UTF-8, and cp1252 has no mapping for 0x9d or
        # 0x90 -- the middle bytes of a curly quote (U+201D is E2 80 9D) and of
        # many em-dashes. Measured across 2026-09-05..09: 27 alarms logged
        # `AGENT INVOKE FAILED 'charmap' codec can't decode byte 0x9d`, each
        # one an alarm that fired and produced no agent. The failure lands mid
        # stream, after the agent is already running and emitting
        # thinking_tokens, so the child is orphaned rather than never started,
        # and the broad handler upstream records it as returncode 127 -- the
        # same code used for "no launcher on PATH", which reads as an install
        # problem rather than a decoding one.
        #
        # This also covers stdin: the prompt carries the probe JSON verbatim,
        # so an em-dash in a failure reason would otherwise raise
        # UnicodeEncodeError on the way out.
        process = subprocess.Popen(
            command, cwd=ROOT, text=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr,
            encoding="utf-8", errors="replace",
        )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(agent_prompt(decision, probe))
        process.stdin.close()
        for line in process.stdout:
            stdout.write(line)
            stdout.flush()
            try:
                activity = format_claude_event(json.loads(line))
            except (ValueError, TypeError):
                activity = ""
            if activity:
                append_activity(activity_path, activity)
        process.stdout.close()
        return process.wait()

def observe(state: dict, decision: Decision, stability_polls: int) -> tuple[dict, bool]:
    if decision.kind == "healthy":
        state.update({
            "pending_fingerprint": "", "pending_count": 0,
            "last_invoked_fingerprint": "", "last_invoked_unix": 0.0,
        })
        return state, False
    if state.get("pending_fingerprint") == decision.fingerprint:
        count = int(state.get("pending_count") or 0) + 1
    else:
        count = 1
    state.update({
        "pending_fingerprint": decision.fingerprint,
        "pending_count": count,
    })
    return state, count >= stability_polls


def cooldown_elapsed(state: dict, decision: Decision, *, now: float,
                     retry_cooldown: float) -> bool:
    if state.get("last_invoked_fingerprint") != decision.fingerprint:
        return True
    return now - float(state.get("last_invoked_unix") or 0.0) >= retry_cooldown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--session-id",
        default=(os.environ.get("CLAUDE_SESSION_ID")
                 or os.environ.get("CODEX_THREAD_ID", "")),
        help="Claude Code session to resume so context carries across alarms.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("WIZARD_WATCH_MODEL", "opus"),
        help="Model alias passed to `claude --model`.",
    )
    parser.add_argument(
        "--effort", default=os.environ.get("WIZARD_WATCH_EFFORT", "xhigh"),
        choices=("low", "medium", "high", "xhigh", "max"),
        help="Reasoning effort passed to `claude --effort`.",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stability-polls", type=int, default=2)
    parser.add_argument("--stall-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--admission-stall-hours", type=float, default=6.0,
        help=(
            "Alarm when no interval has admitted for this long while the "
            "curriculum reports itself active. Motion is not progress: "
            "measured 2026-09-05, eight clean resource cycles and seven "
            "intervals advanced with zero admissions for two weeks."
        ),
    )
    parser.add_argument(
        "--memory-floor-gb", type=float, default=1.5,
        help="Alarm when host available memory falls below this.",
    )
    parser.add_argument("--retry-cooldown", type=float, default=1800.0)
    parser.add_argument(
        "--state-path", type=Path,
        default=ROOT / "runtime/programming-brain-watch/state.json",
    )
    parser.add_argument(
        "--completion-marker", type=Path,
        default=(
            ROOT / "runtime/programming-brain-watch/"
            "objective-complete.json"
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # A missing session id is no longer fatal: Claude Code can start a fresh
    # session, and refusing to run would mean no supervision at all.
    if not args.session_id and not args.dry_run:
        print("no --session-id/CLAUDE_SESSION_ID; alarms start fresh sessions",
              file=sys.stderr)
    if args.stability_polls < 1 or args.poll_seconds < 1:
        parser.error("poll and stability values must be positive")

    state = read_json(args.state_path)
    activity_path = args.state_path.parent / "activity.log"
    append_activity(
        activity_path,
        f"WATCHER START pid={os.getpid()} session={args.session_id or 'dry-run'}",
    )
    loaded_mtime = source_mtime()
    while True:
        if completion_marker_valid(read_json(args.completion_marker)):
            state.update({"state": "objective_complete", "updated_unix": time.time()})
            atomic_json(args.state_path, state)
            print(json.dumps({"decision": {"kind": "objective_complete"}}))
            return 0
        reload_stale_watcher(loaded_mtime, activity_path, args.dry_run)
        try:
            probe = remote_probe(args.profile, args.instance_id, args.runtime)
            decision = classify_probe(
                probe, stall_seconds=args.stall_seconds,
                admission_stall_hours=args.admission_stall_hours,
                memory_floor_gb=args.memory_floor_gb,
            )
            required_polls = (
                1 if decision.kind in {"quarantine_ready", "milestone"}
                else args.stability_polls
            )
            state, stable = observe(state, decision, required_polls)
            trigger = stable and cooldown_elapsed(
                state, decision, now=time.time(),
                retry_cooldown=args.retry_cooldown,
            )
            state.update({
                "last_probe": probe,
                "last_decision": decision.__dict__,
                "updated_unix": time.time(),
            })
            atomic_json(args.state_path, state)
            print(json.dumps({"decision": decision.__dict__, "trigger": trigger}))
            status = probe.get("status") or {}
            append_activity(
                activity_path,
                f"{decision.kind.upper()} {decision.reason}; "
                f"phase={status.get('phase', '-')} "
                f"state={status.get('state', probe.get('host_state', '-'))} "
                f"row={status.get('durable_next_row', '-')} "
                f"target={status.get('block_target_row', '-')}; "
                f"curriculum={probe.get('curriculum', {}).get('durable_processed_rows', '-')}"
                f"/{probe.get('curriculum', {}).get('total_rows', '-')} "
                f"accepted={probe.get('curriculum', {}).get('accepted_rows', '-')} "
                f"quarantined={probe.get('curriculum', {}).get('deferred_rows', '-')} "
                f"forward_remaining={probe.get('curriculum', {}).get('forward_remaining_rows', '-')} "
                f"minimum_outstanding={probe.get('curriculum', {}).get('minimum_outstanding_rows', '-')}",
            )
            if trigger and not args.dry_run:
                # Waking the agent is a REMEDIATION, not part of the probe.
                # Keep its failures out of the probe's except block: a missing
                # CLI used to surface as "PROBE ERROR ... cannot find
                # the file specified" on every poll, which read as though the
                # AWS probe itself had broken and hid the numbers it had just
                # fetched successfully.
                try:
                    returncode = invoke_claude(
                        args.session_id, decision, probe,
                        args.state_path.parent / "logs", activity_path,
                        model=args.model, effort=args.effort,
                    )
                except Exception as exc:  # noqa: BLE001 - remediation only
                    returncode = 127
                    append_activity(activity_path, f"AGENT INVOKE FAILED {exc}")
                state["last_agent_returncode"] = returncode
                state["last_agent_unix"] = time.time()
                if returncode == 0:
                    state["last_invoked_fingerprint"] = decision.fingerprint
                    state["last_invoked_unix"] = time.time()
                atomic_json(args.state_path, state)
            # The probe succeeded; make sure a stale error from an earlier
            # cycle does not keep showing up in the header.
            state.pop("probe_error", None)
            atomic_json(args.state_path, state)
        except Exception as exc:
            state.update({"probe_error": str(exc), "updated_unix": time.time()})
            atomic_json(args.state_path, state)
            append_activity(activity_path, f"PROBE ERROR {exc}")
            print(f"watch probe failed: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
