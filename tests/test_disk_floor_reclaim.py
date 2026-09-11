"""The disk-pressure yield must measure its reclaim, and must be escapable.

Two defects met on 2026-09-10 to make training stop silently.

The first is arithmetic. `--min-free-disk-gb 150` was shipped to convert a 115x
ENOSPC crash loop into a clean cooperative yield, and it does -- but the yield
lands in `while (memory low) or (disk low): publish("resource_waiting"); sleep`.
Memory leaves that loop on its own: `recycle_settled_runtime_node` hands the
allocator's arena back, measured 2.99 GB -> 14.66 GB. Disk does not. The
`.wbrain` neuron store is append-only, its compaction pass is not on this path,
and once the worker is stopped the volume stops falling and nothing raises it.
So the disk arm was an unbounded wait for an event that cannot occur, and a
silent one: the unit stays `active`, the row is parked on a durable boundary,
and every heartbeat rule in CLAUDE.md says a frozen row during settlement is
normal by design.

The second is measurement. The only reclaim on that path is
`prune_resolved_deferred_bases`, and on this XFS reflink volume the bytes it
removes and the bytes it RETURNS are not the same order of magnitude. Measured
2026-09-10: nine deferred directories holding ~560 GB of apparent `st_blocks`
returned 0.00 GB, and re-measured today a single 53.55 GB inode wears **85
names** while another wears 17. Summing file sizes would have reported
terabytes of cleanup for a volume that did not move, and the caller would then
have waited forever on the strength of it.

These tests pin the behaviour that fixes both: a reclaim reported as a `df`
delta, and a floor that names itself when it cannot be cleared.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.programming_curriculum_supervisor import (
    MAX_DISK_RECLAIM_ATTEMPTS,
    deferred_intervals_path,
    reclaim_disk_for_floor,
)


def _deferred_dir(runtime: Path, interval_id: str, payload: bytes) -> Path:
    """Create the causal base directory the pruner keys on."""
    import hashlib

    digest = hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]
    directory = runtime / "deferred" / digest
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "brain.base.wbrain").write_bytes(payload)
    return directory


def _ledger(runtime: Path, rows: list[dict]) -> None:
    path = deferred_intervals_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_a_reclaim_is_a_df_delta_and_never_a_sum_of_file_sizes(
        tmp_path: Path) -> None:
    """Hardlinked bases make the summed size a fiction.

    `st_nlink` 85 on a single 53.55 GB inode means eighty-four of those
    directories cost nothing at all. A reclaim that adds up what it unlinked
    reports success it did not achieve; only the volume's own free-space delta
    is evidence.
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    # One real base, and a second directory whose base is a HARDLINK to it --
    # exactly the shape measured on the host.
    payload = b"x" * (1024 * 1024)
    first = _deferred_dir(runtime, "phase:0:100", payload)
    import hashlib

    second_digest = hashlib.sha256(b"phase:100:200").hexdigest()[:16]
    second = runtime / "deferred" / second_digest
    second.mkdir(parents=True)
    os.link(first / "brain.base.wbrain", second / "brain.base.wbrain")
    assert (second / "brain.base.wbrain").stat().st_nlink == 2

    # Both intervals are KNOWN and RESOLVED, so both are prunable.
    _ledger(runtime, [
        {"interval_id": "phase:0:100", "status": "deferred", "phase": "p",
         "start_row": 0, "end_row": 100},
        {"interval_id": "phase:100:200", "status": "deferred", "phase": "p",
         "start_row": 100, "end_row": 200},
        {"interval_id": "phase:0:100", "status": "resolved"},
        {"interval_id": "phase:100:200", "status": "resolved"},
    ])

    summed_bytes = sum(
        path.stat().st_size
        for path in runtime.glob("deferred/*/brain.base.wbrain")
    )
    assert summed_bytes == 2 * len(payload), "fixture should look like 2 MB"

    report = reclaim_disk_for_floor(runtime, min_free_disk_gb=0.000001)

    assert report["removed_count"] == 2, report
    assert not list(runtime.glob("deferred/*/brain.base.wbrain"))
    # The report must expose the measured delta, not the fiction.
    assert "reclaimed_bytes" in report
    assert report["reclaimed_bytes"] == (
        report["free_bytes_after"] - report["free_bytes_before"]
    )
    # And it must never claim the summed size: the hardlinked copy was free.
    assert report["reclaimed_bytes"] < summed_bytes, report


def test_a_reclaim_that_cannot_clear_the_floor_says_so(tmp_path: Path) -> None:
    """`cleared_floor` is what ends the wait; it must be measured, not assumed.

    With nothing prunable the reclaim returns zero directories and zero bytes.
    If that were reported as success the caller would resume into the same
    pressure; if it were reported as an exception the disk path would lose its
    only diagnostic.
    """
    runtime = tmp_path / "runtime"
    (runtime / "deferred").mkdir(parents=True)
    _ledger(runtime, [])

    impossible = reclaim_disk_for_floor(runtime, min_free_disk_gb=10 ** 9)
    assert impossible["removed_count"] == 0
    assert impossible["reclaimed_bytes"] == 0
    assert impossible["cleared_floor"] is False
    assert "error" not in impossible

    # A floor the volume already clears must report success without deleting.
    satisfied = reclaim_disk_for_floor(runtime, min_free_disk_gb=0.000001)
    assert satisfied["cleared_floor"] is True


def test_a_missing_deferred_tree_is_not_an_exception(tmp_path: Path) -> None:
    """The disk path has no other diagnostic, so a reclaim must not raise.

    A partial reclaim that reports nothing is indistinguishable from a complete
    one -- which is how exactly one root-owned directory stopped all reclaim on
    this volume for five weeks.
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    report = reclaim_disk_for_floor(runtime, min_free_disk_gb=0.000001)
    assert report["removed_count"] == 0
    assert report["reclaimed_bytes"] == 0


def test_the_disk_wait_is_bounded_rather_than_unbounded() -> None:
    """Three attempts, because one blocked inode is not "nothing to reclaim".

    `prune_resolved_deferred_bases` is per-directory and publishes
    `deferred_base_prune_blocked`, so a single failure must not be read as an
    exhausted volume -- but three in a row, measured by `df`, means there is
    genuinely nothing left and the wait needs a name instead of a spin.
    """
    assert MAX_DISK_RECLAIM_ATTEMPTS >= 2
    source = (
        Path(__file__).parents[1]
        / "scripts/programming_curriculum_supervisor.py"
    ).read_text(encoding="utf-8")
    # The old loop could only be left by free space rising. The new one must
    # publish a state a watchdog can key on.
    assert "disk_exhausted_unrecoverable" in source
    assert source.count("reclaim_disk_for_floor") >= 2


def test_the_replay_path_has_its_own_disk_floor() -> None:
    """The guard must be on the stage that RUNS, not the stage that finished.

    `--min-free-disk-gb` was only ever enforced by `disk_floor_breached`, which
    is called from the forward corpus-phase loop. Once `forward_remaining_rows`
    reaches 0 that loop is done and deferred replay does all remaining
    training -- with nothing checking the volume.

    Measured 2026-09-11: the running supervisor carried `--min-free-disk-gb
    150` on its own argv while deferred replay trained at 96.39 GB free and
    108.66 GB/h, roughly 40 minutes from the ENOSPC crash loop that floor
    exists to prevent. Both watchdog emitters called the host healthy. Stopping
    the worker took the burn to -0.0 GB/h, so there was never any doubt about
    which path was writing.
    """
    source = (
        Path(__file__).parents[1]
        / "scripts/programming_curriculum_supervisor.py"
    ).read_text(encoding="utf-8")
    worker = source.split("def run_deferred_replay_worker", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "replay_disk_floor_breached" in worker, (
        "the deferred-replay worker must check the volume, not only memory"
    )
    assert "replay_memory_floor_breached" in worker


def test_replay_disk_floor_reads_the_volume_not_memory(tmp_path: Path) -> None:
    from scripts.programming_curriculum_supervisor import (
        replay_disk_floor_breached,
    )

    # A floor of 0 disables the guard, exactly as the memory floor does.
    assert replay_disk_floor_breached(0.0, tmp_path, free_bytes=1) is False
    one_gb = 1024 ** 3
    assert replay_disk_floor_breached(150.0, tmp_path, free_bytes=96 * one_gb)
    assert not replay_disk_floor_breached(
        150.0, tmp_path, free_bytes=151 * one_gb
    )


def test_an_exhausted_disk_stops_the_pass_instead_of_respawning() -> None:
    """A disk yield that loops is worse than no guard at all.

    Memory and disk leave a yield by different routes: the node recycle hands
    the allocator's arena back (measured 2.98 GB -> 14.65 GB), so respawning
    after a memory yield is correct. Nothing returns disk on an append-only
    store, so respawning after a disk yield burns the remaining headroom and
    stops again -- turning a guard into a faster path to ENOSPC.
    """
    source = (
        Path(__file__).parents[1]
        / "scripts/programming_curriculum_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "DISK_EXHAUSTED_EXIT = 90" in source
    replays = source.split("def run_deferred_replays", 1)[1]
    assert "disk_exhausted" in replays
    assert "return DISK_EXHAUSTED_EXIT" in replays
    # Whatever the pass banked must be recorded before the halt, or the next
    # generation replays the span from its start.
    halt = replays.split("disk_exhausted", 1)[1][:1200]
    assert "record_deferred_replay_resume" in halt

    unit = (
        Path(__file__).parents[1]
        / "scripts/aws/wizard-curriculum-supervisor.service"
    ).read_text(encoding="utf-8")
    assert "RestartPreventExitStatus=42 90" in unit, (
        "systemd must not restart into the same wall"
    )
