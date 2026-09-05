"""Regression tests for mid-pass deferred-replay resume checkpointing.

Context (2026-09-05): the resume row was recorded once per COMPLETED pass, so
anything that ended a pass early threw away every row it had trained. Two
supervisor restarts inside one hour discarded 29,552 and then 1,776 episodes
that were already WAL-durable, and the interval began again at row 0 both
times. At ~13 rows/s a 49,152-row pass is an hour of billed compute.

The pass now checkpoints its durable boundary while it runs. These tests pin
the parts that are easy to get subtly wrong: the recorded row must never move
backwards, never leave the interval, and must survive the restart path that
deletes a pass's progress file.
"""
import json

import pytest

from scripts import programming_curriculum_supervisor as sup


GUARD = "jupyter-scientific-full:1788614113.93:4148213"


@pytest.fixture
def interval():
    return {
        "interval_id": "jupyter-scientific-full:0:131072",
        "start_row": 0,
        "end_row": 131072,
    }


def digest_of(interval):
    import hashlib

    return hashlib.sha256(
        str(interval["interval_id"]).encode("utf-8")
    ).hexdigest()[:16]


def write_progress(path, durable_next_row):
    path.write_text(
        json.dumps({"durable_next_row": durable_next_row}), encoding="utf-8"
    )


def test_a_running_pass_records_its_durable_boundary(tmp_path, interval):
    digest = digest_of(interval)
    progress = tmp_path / f"deferred-replay-{digest}.progress.json"
    write_progress(progress, 23880)

    recorded = sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 0
    )

    assert recorded == 23880
    assert sup.deferred_replay_resume_row(
        tmp_path, digest, interval, GUARD
    ) == 23880


def test_the_recorded_row_never_moves_backwards(tmp_path, interval):
    digest = digest_of(interval)
    progress = tmp_path / f"deferred-replay-{digest}.progress.json"

    write_progress(progress, 23880)
    recorded = sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 0
    )
    # A worker restarting inside the same pass republishes a lower row.
    write_progress(progress, 272)
    recorded = sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, recorded
    )

    assert recorded == 23880
    assert sup.deferred_replay_resume_row(
        tmp_path, digest, interval, GUARD
    ) == 23880


def test_a_row_past_the_interval_is_never_recorded(tmp_path, interval):
    """The gate covers [start_row, end_row); a resume past end_row would hand
    it rows the replay never posted."""
    digest = digest_of(interval)
    progress = tmp_path / f"deferred-replay-{digest}.progress.json"
    write_progress(progress, 999_999)

    recorded = sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 0
    )

    assert recorded == interval["end_row"]
    assert sup.deferred_replay_resume_row(
        tmp_path, digest, interval, GUARD
    ) == interval["end_row"]


def test_a_missing_or_unreadable_progress_file_is_not_fatal(tmp_path, interval):
    digest = digest_of(interval)
    progress = tmp_path / f"deferred-replay-{digest}.progress.json"

    assert sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 1776
    ) == 1776

    progress.write_text("{not json", encoding="utf-8")
    assert sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 1776
    ) == 1776


def test_a_rollback_invalidates_the_checkpoint(tmp_path, interval):
    """A different guard means the prefix this record describes is gone."""
    digest = digest_of(interval)
    progress = tmp_path / f"deferred-replay-{digest}.progress.json"
    write_progress(progress, 23880)
    sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 0
    )

    assert sup.deferred_replay_resume_row(
        tmp_path, digest, interval, "some-other-guard:0:0"
    ) == interval["start_row"]


def test_an_interrupted_pass_resumes_instead_of_restarting(tmp_path, interval):
    """The whole point: a supervisor restart mid-pass must not rewind to 0.

    Models the startup path, which deletes the progress file whenever the
    resume row is still at the interval's first row.
    """
    digest = digest_of(interval)
    progress = tmp_path / f"deferred-replay-{digest}.progress.json"

    # Pass runs and is killed before it can finish.
    write_progress(progress, 29552)
    sup.checkpoint_replay_resume(
        tmp_path, digest, interval, GUARD, progress, 0
    )

    # Supervisor restarts and recomputes where to begin.
    resume_row = sup.deferred_replay_resume_row(
        tmp_path, digest, interval, GUARD
    )
    if resume_row <= int(interval["start_row"]):
        progress.unlink(missing_ok=True)
        sup.deferred_replay_resume_path(tmp_path, digest).unlink(missing_ok=True)

    assert resume_row == 29552, "the restart rewound the pass to its first row"
    assert progress.exists(), "the durable progress file was deleted"


def test_barren_yield_tolerance_is_bounded():
    """A yield that won nothing may retry, but not forever."""
    assert sup.MAX_BARREN_REPLAY_YIELDS >= 1
    assert sup.MAX_BARREN_REPLAY_YIELDS <= 10
