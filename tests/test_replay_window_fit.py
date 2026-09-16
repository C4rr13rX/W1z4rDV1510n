"""A disk halt must reclaim what it can, and must not spend on what cannot fit.

Two defects met on 2026-09-11 and stopped this curriculum for 107.7 hours.

The first is that the halt could not reach its own reclaim. The disk guard
calls `reclaim_disk_for_floor`, which calls `prune_resolved_deferred_bases` and
nothing else, and on this XFS reflink volume that reclaim is exhausted -- three
attempts returned 0.00 GB each. The supervisor then exited 90, and
`RestartPreventExitStatus=42 90` makes 90 terminal, so the unit sat `failed`
until a human restarted it. But the reclaim that DOES work here is the rollback
that any interrupted interval already owes: `brain.wbrain` is a reflink clone of
`brain.last-good.wbrain` plus appends, so replacing it unlinks every block it
does not share. Measured 2026-09-15 by extent subtraction: 854.02 GB allocated,
440.00 GB shared, **414.02 GB unique**. That reclaim lived only on the STARTUP
path, in `recover_interrupted_deferred_replay` -- which an exit that forbids
restart can never reach. The halt was terminal by construction while holding the
bytes that would have cleared its own floor.

The second is that rolling back and retrying, on its own, is a livelock. The
window a rollback buys is 415.18 GB above the floor at a measured 114.12 GB/h --
**3.64 h**. Measured against the live queue the same day, 0 of 22 unresolved
intervals could reach a gate inside it, missing by 8x
(jupyter-scientific-partial, 75,876 rows at 2,585.7 rows/h = 29.34 h) to 156x
(jupyter-scientific-para4, 131,072 rows at 230.5 rows/h = 568.64 h). So a
supervisor that rolled back and tried the next candidate would have spent ~88 h
of billed compute, one rollback and regrow per interval, and admitted nothing --
with the unit `active` and the row advancing the whole time.

`order_replay_candidates` cannot fix that: it orders by `(stalls, span)`, and a
row count is a proxy for cost that is wrong by a factor of 260 between the
phases actually queued (go-systems 59,778 rows/h against para4 230 rows/h). The
census here measures the real thing and refuses to spend when nothing fits --
a refusal, never a retirement: every obligation stays `deferred` and eligible,
`unknown` always counts as eligible, and the same census passes the moment the
burn falls or the volume grows.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import programming_curriculum_supervisor as sup

GIB = 1024 ** 3

#: An epoch-shaped clock. Timestamps here are deliberately realistic rather
#: than starting at 0: `updated_unix` of 0.0 is falsy, and a measurement that
#: reads it with `a or b` drops the record. That is a real trap -- it is how
#: `health_event_unix` came to exist -- but it belongs in its own test, not
#: hidden inside every fixture in the file.
T0 = 1_789_000_000.0


def _ledger(runtime: Path, events: list[dict]) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "curriculum-health.jsonl").open(
            "w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


def _yield_event(unix: float, free_gb: float) -> dict:
    """A resource yield exactly as `run_deferred_replay_worker` writes one."""
    return {
        "kind": "deferred_replay_resource_yield",
        "phase": "jupyter-scientific-partial",
        "interval_id": "jupyter-scientific-partial:131072:206948",
        "disk_free_bytes_before": int(free_gb * GIB),
        "disk_free_bytes_after": int(free_gb * GIB),
        "updated_unix": unix,
    }


def _floor_for_window(runtime: Path, window_gb: float) -> float:
    """A floor that leaves exactly `window_gb` of runway on the real volume.

    `replay_window_census` measures its window as `free - floor` from the
    actual volume, because that is the runway an interval gets before the disk
    guard stops it. Deriving the floor from current free space lets these tests
    assert the arithmetic rather than the test machine's spare capacity.
    """
    runtime.mkdir(parents=True, exist_ok=True)
    return (shutil.disk_usage(runtime).free / GIB) - window_gb


def _interval(interval_id: str, phase: str, start: int, end: int) -> dict:
    return {
        "interval_id": interval_id, "phase": phase,
        "start_row": start, "end_row": end, "status": "deferred",
    }


# ---------------------------------------------------------------------------
# The burn
# ---------------------------------------------------------------------------

def test_burn_is_measured_only_where_the_volume_actually_fell(tmp_path):
    """A rollback is a rise, not negative burn.

    Both heartbeat counters on this host have already produced a negative rate
    by straddling a reset, and a burn averaged across a 414 GB rollback would
    report a drain lower than anything that ever happened -- which would then
    inflate the window and send a doomed interval to train.
    """
    runtime = tmp_path / "runtime"
    _ledger(runtime, [
        _yield_event(T0, 500.0),
        _yield_event(T0 + 3600, 400.0),     # fell 100 GB in 1 h
        _yield_event(T0 + 3700, 814.0),     # a rollback: +414 GB, must be ignored
        _yield_event(T0 + 7300, 714.0),     # fell 100 GB in 1 h
    ])
    burn = sup.measure_disk_burn_gb_per_hour(runtime)
    assert burn["gb_per_hour"] == pytest.approx(100.0, abs=0.5)
    assert burn["hours_measured"] == pytest.approx(2.0, abs=0.05)
    # Averaging the rise in would have produced a *negative* burn here, and a
    # negative burn makes every interval look like it fits.
    assert burn["gb_per_hour"] > 0


def test_burn_with_no_falling_segment_is_unknown_not_zero(tmp_path):
    """A halted host burns nothing, and that must not read as infinite window."""
    runtime = tmp_path / "runtime"
    _ledger(runtime, [_yield_event(T0, 500.0), _yield_event(T0 + 3600, 500.0)])
    burn = sup.measure_disk_burn_gb_per_hour(runtime)
    assert burn["gb_per_hour"] is None
    assert burn["reason"] == "no falling segment"


def test_a_missing_ledger_is_not_an_exception(tmp_path):
    burn = sup.measure_disk_burn_gb_per_hour(tmp_path / "nothing")
    assert burn["gb_per_hour"] is None


# ---------------------------------------------------------------------------
# The reclaim
# ---------------------------------------------------------------------------

def test_an_unmeasured_rollback_reports_unknown_and_never_zero(tmp_path):
    """A reclaim of zero and a reclaim never attempted are different facts.

    Reading the second as the first is precisely what made the halt terminal:
    three prunes returned 0.00 GB, and the guard concluded the volume was
    unrecoverable while a 414 GB rollback sat one call away.
    """
    runtime = tmp_path / "runtime"
    _ledger(runtime, [_yield_event(T0, 500.0)])
    assert sup.rollback_reclaim_bytes(runtime) is None


def test_the_rollback_reclaim_is_read_from_its_own_measured_event(tmp_path):
    runtime = tmp_path / "runtime"
    _ledger(runtime, [
        {"kind": sup.ROLLBACK_RECLAIM_KIND, "reclaimed_bytes": 10 * GIB,
         "updated_unix": T0 + 1},
        {"kind": sup.ROLLBACK_RECLAIM_KIND,
         "reclaimed_bytes": int(414.02 * GIB), "updated_unix": T0 + 2},
    ])
    assert sup.rollback_reclaim_bytes(runtime) == int(414.02 * GIB)


def test_the_rollback_publishes_what_it_returned(tmp_path):
    """`restore_rejected_deferred_replay` must measure itself by `df`.

    Without this event the census has no way to know a rollback returns
    anything, so it would treat the headroom as zero forever and refuse every
    interval -- turning a working reclaim into a permanent halt by measurement.
    """
    source = (sup.ROOT / "scripts"
              / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8")
    body = source.split("def restore_rejected_deferred_replay")[1]
    body = body.split("\ndef ")[0]
    assert "ROLLBACK_RECLAIM_KIND" in body
    assert "free_bytes_before" in body and "free_bytes_after" in body
    # Measured either side of the replace, not summed from file sizes: on this
    # volume those two numbers differ by orders of magnitude.
    assert body.index("free_before_rollback = ") < body.index(
        "finalize_canary_restore(")
    assert body.index("finalize_canary_restore(") < body.index(
        "free_after_rollback = ")


# ---------------------------------------------------------------------------
# The rate, per phase
# ---------------------------------------------------------------------------

def test_phase_rate_is_end_to_end_wall_clock_with_its_sample_count(tmp_path):
    runtime = tmp_path / "runtime"
    _ledger(runtime, [
        {"kind": "deferred_replay_resource_yield", "phase": "go-systems",
         "interval_id": "go-systems:0:131072", "updated_unix": T0},
        {"kind": "deferred_replay_admitted", "phase": "go-systems",
         "interval_id": "go-systems:0:131072", "updated_unix": T0 + 7200},
    ])
    rates = sup.measure_phase_rows_per_hour(runtime)
    assert rates["go-systems"]["samples"] == 1
    assert rates["go-systems"]["rows_per_hour"] == pytest.approx(65536, rel=1e-3)


def test_the_phase_spread_is_what_makes_a_span_key_wrong(tmp_path):
    """Same span, two phases, wildly different cost.

    This is the whole reason the census exists beside `order_replay_candidates`:
    131,072 rows is 2.2 h of go-systems and 568.6 h of para4, and a sort on row
    count cannot tell them apart.
    """
    runtime = tmp_path / "runtime"
    _ledger(runtime, [
        {"kind": "deferred_replay_resource_yield", "phase": "go-systems",
         "interval_id": "go-systems:0:131072", "updated_unix": T0},
        {"kind": "deferred_replay_admitted", "phase": "go-systems",
         "interval_id": "go-systems:0:131072", "updated_unix": T0 + 2.2 * 3600},
        {"kind": "deferred_replay_resource_yield", "phase": "para4",
         "interval_id": "para4:0:131072", "updated_unix": T0},
        {"kind": "deferred_replay_admitted", "phase": "para4",
         "interval_id": "para4:0:131072", "updated_unix": T0 + 568.6 * 3600},
    ])
    rates = sup.measure_phase_rows_per_hour(runtime)
    assert rates["go-systems"]["rows_per_hour"] > 200 * \
        rates["para4"]["rows_per_hour"]


# ---------------------------------------------------------------------------
# The census, and the refusal
# ---------------------------------------------------------------------------

def _census_ledger(runtime: Path, *, burn_gb_per_hour: float,
                   reclaim_gb: float, rates: list[dict]) -> None:
    events: list[dict] = []
    # Two yields an hour apart, falling by exactly the burn.
    events.append(_yield_event(T0, 1000.0))
    events.append(_yield_event(T0 + 3600, 1000.0 - burn_gb_per_hour))
    events.append({"kind": sup.ROLLBACK_RECLAIM_KIND,
                   "reclaimed_bytes": int(reclaim_gb * GIB),
                   "updated_unix": T0 + 3700})
    events.extend(rates)
    _ledger(runtime, events)


def _admissions(phase: str, rows: int, hours: float, count: int) -> list[dict]:
    """`count` admitted intervals of `rows` rows each taking `hours`."""
    events: list[dict] = []
    clock = T0 + 10_000.0
    for index in range(count):
        interval_id = f"{phase}:{index * rows}:{(index + 1) * rows}"
        events.append({"kind": "deferred_replay_resource_yield",
                       "phase": phase, "interval_id": interval_id,
                       "updated_unix": clock})
        events.append({"kind": "deferred_replay_admitted", "phase": phase,
                       "interval_id": interval_id,
                       "updated_unix": clock + hours * 3600})
        clock += hours * 3600 + 10
    return events


def test_an_interval_that_cannot_reach_its_gate_is_named_exceeds(tmp_path):
    runtime = tmp_path / "runtime"
    # 100 GB/h burn, 400 GB returned by a rollback. The census adds current
    # free space, which for a tmp_path is the test machine's own volume, so the
    # window is measured rather than asserted -- only the verdict is asserted.
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0,
                   rates=_admissions("slow", 1000, 100.0, 2))
    census = sup.replay_window_census(
        runtime, [_interval("slow:0:1000000", "slow", 0, 1_000_000)],
        min_free_disk_gb=_floor_for_window(runtime, 400.0),
    )
    assert census["window_hours"] == pytest.approx(4.0, abs=0.1)
    assert census["exceeds"] == ["slow:0:1000000"]
    assert census["intervals"][0]["eta_hours"] == pytest.approx(100_000.0,
                                                                rel=1e-3)
    assert sup.replay_queue_is_hopeless(census) is True


def test_an_interval_that_fits_is_never_refused(tmp_path):
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0,
                   rates=_admissions("fast", 100_000, 1.0, 2))
    census = sup.replay_window_census(
        runtime, [_interval("fast:0:100000", "fast", 0, 100_000)],
        min_free_disk_gb=_floor_for_window(runtime, 400.0),
    )
    assert census["fits"] == ["fast:0:100000"]
    assert sup.replay_queue_is_hopeless(census) is False


def test_one_admission_is_an_anecdote_and_never_refuses(tmp_path):
    """Refusing on a single sample would make the measurement self-fulfilling.

    `jupyter-scientific-para4`'s 230.5 rows/h is n=1 on the live host. It is
    good enough to ORDER by and not good enough to halt on, so a phase under
    `MIN_RATE_SAMPLES_TO_REFUSE` samples stays eligible however bad it looks.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0,
                   rates=_admissions("thin", 1000, 100.0, 1))
    census = sup.replay_window_census(
        runtime, [_interval("thin:0:1000000", "thin", 0, 1_000_000)],
        min_free_disk_gb=_floor_for_window(runtime, 400.0),
    )
    assert census["unknown"] == ["thin:0:1000000"]
    assert census["intervals"][0]["rate_samples"] == 1
    assert sup.replay_queue_is_hopeless(census) is False


def test_a_phase_that_has_never_admitted_is_always_attempted(tmp_path):
    """A count of zero from a path with no opportunity to run is not evidence."""
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0, rates=[])
    census = sup.replay_window_census(
        runtime, [_interval("new:0:1000000", "new", 0, 1_000_000)],
        min_free_disk_gb=_floor_for_window(runtime, 400.0),
    )
    assert census["unknown"] == ["new:0:1000000"]
    assert sup.replay_queue_is_hopeless(census) is False


def test_an_empty_queue_is_not_hopeless(tmp_path):
    """`deferred_replay_complete` is a success; it must not exit 91."""
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0, rates=[])
    census = sup.replay_window_census(runtime, [], min_free_disk_gb=150.0)
    assert sup.replay_queue_is_hopeless(census) is False


def test_the_live_2026_09_11_queue_is_measured_hopeless(tmp_path):
    """Built on the payload the HOST emits, not on a hand-set convenience.

    Every test covering the convergence annex once hand-set `block_target_row`,
    a key only the forward stage writes, so 47 tests passed against a payload no
    replay ever produces. These are the real numbers from the halted host:
    114.12 GB/h burn, a 414.02 GB rollback, a 150 GB floor, and the two phases
    whose rates were actually measured. The queue must come back hopeless, and
    the shortest ETA must still be the 29.34 h that misses the window by 8x.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(
        runtime, burn_gb_per_hour=114.12, reclaim_gb=414.02,
        # 2,585.7 rows/h and 230.5 rows/h, each with enough samples to refuse.
        rates=(_admissions("jupyter-scientific-partial", 11_584, 4.48, 2)
               + _admissions("jupyter-scientific-para4", 7_984, 34.64, 2)),
    )
    pending = [
        _interval("jupyter-scientific-partial:131072:206948",
                  "jupyter-scientific-partial", 131_072, 206_948),
        _interval("jupyter-scientific-partial:0:131072",
                  "jupyter-scientific-partial", 0, 131_072),
        _interval("jupyter-scientific-para4:393216:524288",
                  "jupyter-scientific-para4", 393_216, 524_288),
    ]
    # The window is exactly what the rollback returns, which on the live host
    # was 414.02 GB onto 151.16 GB free against a 150 GB floor -- 415.18 GB, and
    # 3.64 h at 114.12 GB/h. Pinning the floor to this machine's free space
    # reproduces that arithmetic without depending on its spare capacity.
    census = sup.replay_window_census(
        runtime, pending, min_free_disk_gb=_floor_for_window(runtime, 415.18)
    )
    assert census["window_hours"] == pytest.approx(3.63, abs=0.05)
    assert census["fits"] == []
    assert census["unknown"] == []
    assert len(census["exceeds"]) == 3
    etas = {row["interval_id"]: row["eta_hours"] for row in census["intervals"]}
    assert etas["jupyter-scientific-partial:131072:206948"] == pytest.approx(
        29.34, abs=0.1)
    assert etas["jupyter-scientific-para4:393216:524288"] == pytest.approx(
        568.6, abs=1.0)
    assert sup.replay_queue_is_hopeless(census) is True


# ---------------------------------------------------------------------------
# Wiring: a helper nothing calls is this repo's standing inert-fix trap
# ---------------------------------------------------------------------------

def _replay_body() -> str:
    source = (sup.ROOT / "scripts"
              / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8")
    return source.split("def run_deferred_replays")[1].split("\ndef ")[0]


def test_the_disk_halt_rolls_back_before_it_calls_itself_unrecoverable():
    """The reclaim must be ATTEMPTED before the terminal exit is taken.

    Fails against the code committed in 7aabf50, where the `disk_exhausted`
    branch published the state and returned `DISK_EXHAUSTED_EXIT` with no
    rollback anywhere between them.
    """
    body = _replay_body()
    branch = body.split('if getattr(worker, "disk_exhausted", False):')[1]
    branch = branch.split("if worker.returncode != 0")[0]
    assert "restore_rejected_deferred_replay(" in branch, \
        "the halt never attempts the only reclaim that works on this volume"
    assert branch.index("restore_rejected_deferred_replay(") < \
        branch.index("DISK_EXHAUSTED_EXIT"), \
        "the terminal exit must come after the rollback, not instead of it"
    # And the stall is recorded first, so a crash mid-rollback still leaves the
    # evidence that this interval consumed a generation.
    assert branch.index("record_replay_stall(") < \
        branch.index("restore_rejected_deferred_replay(")


def test_a_rollback_that_clears_the_floor_keeps_the_pass_alive():
    """Clearing the floor must continue the queue, not exit 90 anyway."""
    body = _replay_body()
    branch = body.split('if getattr(worker, "disk_exhausted", False):')[1]
    branch = branch.split("if worker.returncode != 0")[0]
    assert "cleared_floor" in branch
    assert "if not cleared:" in branch, \
        "the exit must be conditional on the rollback having failed"
    assert "rejected_this_pass.add(interval_id)" in branch
    assert "disk_rolled_back = True" in branch


def test_a_rolled_back_interval_never_reaches_the_gate():
    """Gating a rolled-back interval would score host pressure as a verdict.

    The rollback discards every row the interval trained, so the interval-recall
    check would miss on rows the brain no longer holds and the miss would be
    recorded as `deferred_replay_failed` -- the exact conversion that produced
    288 semantic failures against 19 yields.
    """
    body = _replay_body()
    assert "if disk_rolled_back:" in body
    assert body.index("if disk_rolled_back:") < body.index(
        "interval_recall = run_admission_json_command("), \
        "the skip must come before the gate it is skipping"


def test_the_census_gates_selection_and_names_its_exit():
    body = _replay_body()
    assert "replay_window_census(" in body
    assert "replay_queue_is_hopeless(" in body
    assert "NO_FITTING_INTERVAL_EXIT" in body
    # After the reorder (so it censuses the queue actually being selected from)
    # and before `pending[0]` is taken (so nothing is spent discovering it).
    assert body.index("order_replay_candidates(") < body.index(
        "replay_window_census(")
    assert body.index("replay_window_census(") < body.index(
        "event = pending[0]")


def test_the_named_exits_are_distinct():
    """A full volume and an oversized work unit need different responses.

    90 means the volume is below its floor right now; 91 means the volume is
    fine and no work unit fits it. Collapsing them would send an operator to
    resize a disk that is not full.
    """
    assert sup.NO_FITTING_INTERVAL_EXIT != sup.DISK_EXHAUSTED_EXIT
    assert sup.NO_FITTING_INTERVAL_EXIT not in {
        sup.RESOURCE_SETTLED_EXIT, sup.RESOURCE_SETTLEMENT_FAILED_EXIT, 42,
    }


def test_the_unit_refuses_to_respawn_into_a_work_unit_that_cannot_fit():
    """An exit systemd restarts is not a halt.

    `Restart=on-failure` with `RestartSec=10` would respawn exit 91 every ten
    seconds forever, each time re-measuring a queue whose answer cannot have
    changed and appending a ledger event to say so. The exit only means
    anything if the unit honours it.
    """
    unit = (sup.ROOT / "scripts" / "aws"
            / "wizard-curriculum-supervisor.service").read_text(
        encoding="utf-8")
    prevented = [
        line.split("=", 1)[1].split()
        for line in unit.splitlines()
        if line.startswith("RestartPreventExitStatus=")
    ]
    assert prevented, "the unit does not prevent any restart"
    assert str(sup.NO_FITTING_INTERVAL_EXIT) in prevented[-1]
    # The two older halts must survive this edit.
    assert str(sup.DISK_EXHAUSTED_EXIT) in prevented[-1]
    assert "42" in prevented[-1]


# ---------------------------------------------------------------------------
# The evidence grade that makes this guard fire at all
# ---------------------------------------------------------------------------

def _stall(interval_id: str, phase: str, rows: int, hours: float,
           unix: float = T0) -> dict:
    """A stall exactly as `record_replay_stall` writes one."""
    return {
        "kind": sup.REPLAY_STALL_KIND,
        "interval_id": interval_id,
        "phase": phase,
        "reason": "disk floor reached before a verdict",
        "rows_trained": rows,
        "hours": hours,
        "rows_per_hour": round(rows / hours, 1),
        "updated_unix": unix,
    }


def test_an_intervals_own_stall_is_enough_to_refuse_it(tmp_path):
    """One observation OF an interval beats two estimates ABOUT its phase.

    This is the difference between a guard that works and a guard that is
    inert. On the host this was built for, not one pending phase has two
    admissions -- `jupyter-scientific-partial` and `jupyter-scientific-full`
    have none at all and `para4` has one -- so a census that could only refuse
    on `MIN_RATE_SAMPLES_TO_REFUSE` phase admissions would have passed all 22
    doomed intervals through and spent ~88 h proving what a single stall
    already measured.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=114.12, reclaim_gb=414.02,
                   rates=[_stall("slow:0:75876", "slow", 11_584, 4.48)])
    census = sup.replay_window_census(
        runtime, [_interval("slow:0:75876", "slow", 0, 75_876)],
        min_free_disk_gb=_floor_for_window(runtime, 400.0),
    )
    row = census["intervals"][0]
    assert row["rate_source"] == "own_stall"
    assert row["rate_samples"] == 1
    assert row["rows_per_hour"] == pytest.approx(2585.7, abs=1.0)
    assert row["eta_hours"] == pytest.approx(29.34, abs=0.1)
    assert census["exceeds"] == ["slow:0:75876"]
    assert sup.replay_queue_is_hopeless(census) is True


def test_a_stall_that_measured_nothing_never_refuses(tmp_path):
    """A generation killed before its first durable batch measured no rate.

    Recording that as 0 rows/h would divide into an infinite ETA and refuse
    every interval in the phase forever -- converting one unlucky reboot into a
    permanent halt.
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    sup.record_replay_stall(runtime, "p:0:1000", "p", "killed",
                            rows_trained=0, hours=2.0)
    sup.record_replay_stall(runtime, "p:0:1000", "p", "killed",
                            rows_trained=500, hours=0.0)
    assert sup.replay_stall_observations(runtime) == {}


def test_the_slowest_stall_describes_the_risk(tmp_path):
    """Two stalls, two rates: the census must take the pessimistic one.

    The question is whether the interval can FINISH inside a window, so the
    generation that got least far is the one that bounds it.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0, rates=[
        _stall("p:0:100000", "p", 90_000, 1.0, T0 + 1),
        _stall("p:0:100000", "p", 10_000, 1.0, T0 + 2),
    ])
    census = sup.replay_window_census(
        runtime, [_interval("p:0:100000", "p", 0, 100_000)],
        min_free_disk_gb=_floor_for_window(runtime, 400.0),
    )
    assert census["intervals"][0]["rows_per_hour"] == pytest.approx(10_000.0)
    assert census["intervals"][0]["rate_samples"] == 2
    assert census["exceeds"] == ["p:0:100000"]


def test_a_stall_records_the_rate_it_measured():
    """Both stall sites must pass their measurement, not just a tally.

    `record_replay_stall` gained `rows_trained`/`hours` precisely so the census
    has evidence on this host. A call site that omits them leaves the guard
    inert for that path, which is the failure this whole test file exists for.
    """
    source = (sup.ROOT / "scripts"
              / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8")
    recovery = source.split(
        "def recover_interrupted_deferred_replay", 1
    )[1].split("\ndef ", 1)[0]
    assert "rows_trained=" in recovery and "hours=" in recovery
    # And it must be read BEFORE the rollback, which unlinks the progress file.
    assert recovery.index("replay_pass_durable_row(") < \
        recovery.index("restore_rejected_deferred_replay(")

    body = _replay_body()
    branch = body.split('if getattr(worker, "disk_exhausted", False):')[1]
    branch = branch.split("if worker.returncode != 0")[0]
    assert "rows_trained=" in branch and "hours=" in branch


# ---------------------------------------------------------------------------
# Two corrections the live dry run forced, each of which had shipped wrong once
# ---------------------------------------------------------------------------

def test_a_measured_rollback_does_not_inflate_the_window(tmp_path):
    """A rollback discards the interval; it does not lengthen its runway.

    The first draft added the measured reclaim to the window. Against the live
    host that produced a 1.16 GB window before any rollback had been measured
    and would have produced an 829 GB one immediately after -- both wrong, in
    opposite directions, from the same error. An interval's runway is what sits
    above the floor when it starts, full stop.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0,
                   rates=[_stall("p:0:100000", "p", 50_000, 1.0)])
    census = sup.replay_window_census(
        runtime, [_interval("p:0:100000", "p", 0, 100_000)],
        min_free_disk_gb=_floor_for_window(runtime, 100.0),
    )
    # 100 GB of runway at 100 GB/h is one hour -- NOT five.
    assert census["window_hours"] == pytest.approx(1.0, abs=0.05)
    assert census["rollback_reclaim_bytes"] == int(400.0 * GIB)
    assert census["expected_free_after_rollback_bytes"] > census["free_bytes"]
    # 100,000 rows at 50,000 rows/h is 2 h, which does not fit 1 h.
    assert census["exceeds"] == ["p:0:100000"]


def test_below_its_floor_the_census_abstains_and_lets_the_disk_guard_own_it(
        tmp_path):
    """A full volume is exit 90's fault to name, not exit 91's.

    Refusing every interval because the disk is full would send an operator to
    split corpus intervals when what they actually need is to reclaim or resize.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0,
                   rates=[_stall("p:0:100000", "p", 10, 1.0)])
    free_gb = shutil.disk_usage(runtime).free / GIB
    census = sup.replay_window_census(
        runtime, [_interval("p:0:100000", "p", 0, 100_000)],
        min_free_disk_gb=free_gb + 50.0,        # floor above current free
    )
    assert census["window_bytes"] < 0
    assert census["window_hours"] is None
    assert census["exceeds"] == []
    assert census["unknown"] == ["p:0:100000"]
    assert sup.replay_queue_is_hopeless(census) is False


def test_the_stall_clock_ends_where_training_ended():
    """Measuring to `now` measures the outage, not the work.

    The halted generation trained 11,584 rows in 2.6 h and then sat dead for
    107.7 h. `time.time() - created_unix` scores that as 107.6 rows/h against a
    true 4,455 -- a 40x understatement that would refuse intervals which fit.
    The progress file's mtime is when the row last advanced.
    """
    source = (sup.ROOT / "scripts"
              / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8")
    recovery = source.split(
        "def recover_interrupted_deferred_replay", 1
    )[1].split("\ndef ", 1)[0]
    assert "progress_path.stat().st_mtime" in recovery, \
        "the stall clock must end at the last durable write, not at now"
    assert recovery.index("progress_path.stat().st_mtime") < \
        recovery.index("record_replay_stall(")


def test_a_stall_measures_its_phase_not_just_its_own_interval(tmp_path):
    """Otherwise every interval needs its own full window to be refused.

    Intervals inside a phase share a corpus and so a cost per row; the 260x
    spread is between phases. Measured on the live queue: 22 intervals at ~3.9 h
    each is ~86 h of billed compute to rediscover, one interval at a time, what
    two stalls in a phase already establish.
    """
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0, rates=[
        _stall("p:0:100000", "p", 10_000, 1.0, T0 + 1),
        _stall("p:100000:200000", "p", 10_000, 1.0, T0 + 2),
    ])
    rates = sup.measure_phase_rows_per_hour(runtime)
    assert rates["p"]["samples"] == 2
    assert rates["p"]["rows_per_hour"] == pytest.approx(10_000.0, rel=1e-3)

    # A THIRD interval of the same phase, which has never stalled itself, is
    # now refusable on the phase's evidence.
    census = sup.replay_window_census(
        runtime, [_interval("p:200000:300000", "p", 200_000, 300_000)],
        min_free_disk_gb=_floor_for_window(runtime, 100.0),
    )
    row = census["intervals"][0]
    assert row["rate_source"] == "phase_admissions"
    assert row["rate_samples"] == 2
    assert census["exceeds"] == ["p:200000:300000"]
    assert sup.replay_queue_is_hopeless(census) is True


def test_one_stall_in_a_phase_still_does_not_refuse_its_siblings(tmp_path):
    """The anecdote guard survives folding stalls in."""
    runtime = tmp_path / "runtime"
    _census_ledger(runtime, burn_gb_per_hour=100.0, reclaim_gb=400.0,
                   rates=[_stall("p:0:100000", "p", 10_000, 1.0)])
    census = sup.replay_window_census(
        runtime, [_interval("p:200000:300000", "p", 200_000, 300_000)],
        min_free_disk_gb=_floor_for_window(runtime, 100.0),
    )
    assert census["unknown"] == ["p:200000:300000"]
    assert sup.replay_queue_is_hopeless(census) is False
