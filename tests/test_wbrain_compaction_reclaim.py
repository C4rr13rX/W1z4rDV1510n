"""The capacity halt must compact before it refuses, and compact BOTH names.

Exit 91 halted the curriculum with 2.47 h of disk window against a queue needing
46.1 h at the fastest rate this host has ever measured, and named three
remedies: more RAM, a larger volume, or delta-encoded terminal updates -- all
user decisions. Measured on the live container 2026-09-16, that remedy set is
miscalibrated and the halt was premature.

  * Delta encoding is not the fix it was taken for. A byte-weighted census over
    a 103 GB window of the container found only 41.5 % of rewritten bytes are
    append-shaped (pool 5 hub atoms, identical_fraction 0.999394, a 53 KB delta
    against an 87 MB body). The other 58.5 % rewrite every terminal's weight and
    last_fired_tick with the 8-byte target unchanged, which an append-delta
    cannot compress. Projected reduction: 1.71x, against an 18.7x deficit.

  * The volume is not short of space, it is full of GARBAGE. The `.wbrain`
    store is append-only and has never had a compaction pass on this host, so
    the 472.45 GB container holds a live set measured at ~44.5 GB. The
    operating record's 363.34 GB -- the sole basis for "compaction is
    net-negative" -- came from `compaction::estimate`, which extrapolates
    mean_sampled_body x live_neurons. Run against the same file at four
    strides it returns 848.9 / 119.53 / 49.38 / 44.50 GB: its loosest draw
    exceeds the size of the file it is measuring, because a dozen ~86 MB hub
    atoms sit against a ~1.5 KB median.

So there was a reclaim, it was never attempted, and `RestartPreventExitStatus`
makes exit 91 terminal -- the same shape as the exit-90 halt that sat `failed`
for 107.7 h holding the 414 GB rollback that would have cleared its own floor.

These tests pin the two facts that make the reclaim real rather than reported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.programming_curriculum_supervisor import (
    COMPACTOR_CANDIDATES,
    brain_container_paths,
    locate_compactor,
    reclaim_disk_for_floor,
)


def _runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    (runtime / "brain").mkdir(parents=True)
    return runtime


def test_both_containers_are_compacted_because_they_share_every_extent(
        tmp_path: Path) -> None:
    """Compacting one name alone returns nothing.

    Measured 2026-09-16 straight after a rollback: `brain.wbrain` and
    `brain.last-good.wbrain` each allocate 472.45 GB, SHARE 472.45 GB, and have
    `brain_unique` and `guard_unique` of 0.00 GB -- they are reflink clones of
    each other. Every extent one stops referencing is still pinned by the
    other, which is the same block-sharing trap that made a 560 GB delete
    return 0.00 GB. A reclaim that compacts only the live brain is therefore
    measurable as a no-op.
    """
    runtime = _runtime(tmp_path)
    (runtime / "brain" / "brain.wbrain").write_bytes(b"live")
    (runtime / "brain" / "brain.last-good.wbrain").write_bytes(b"guard")

    names = [path.name for path in brain_container_paths(runtime)]

    assert "brain.wbrain" in names
    assert "brain.last-good.wbrain" in names, (
        "the guard pins every shared extent; compacting only the live brain "
        "reclaims nothing"
    )


def test_a_missing_guard_is_not_an_error(tmp_path: Path) -> None:
    """A brain with no last-good guard still compacts."""
    runtime = _runtime(tmp_path)
    (runtime / "brain" / "brain.wbrain").write_bytes(b"live")

    assert [p.name for p in brain_container_paths(runtime)] == ["brain.wbrain"]


def test_compaction_runs_only_when_pruning_left_the_floor_unclear(
        tmp_path: Path) -> None:
    """Order matters: pruning is cheap, compaction stops the brain.

    Compaction rewrites the container and must not be paid for when a cheap
    directory prune already cleared the floor.
    """
    runtime = _runtime(tmp_path)
    calls: list[str] = []

    def compactor() -> dict:
        calls.append("ran")
        return {"reclaimed_bytes": 0}

    # A floor of ~0 is already cleared by any real volume, so pruning suffices.
    report = reclaim_disk_for_floor(runtime, 0.000001, compactor)

    assert report["cleared_floor"] is True
    assert calls == [], "compaction must not run once the floor is clear"
    assert "compaction" not in report


def test_compaction_runs_when_the_floor_is_unreachable_by_pruning(
        tmp_path: Path) -> None:
    """The halt path's whole purpose: try the reclaim before refusing.

    An impossible floor stands in for the measured case -- pruning exhausted at
    0.00 GB across three consecutive attempts, 2 prunable directories against
    399.56 GB pinned by cross-links.
    """
    runtime = _runtime(tmp_path)
    calls: list[str] = []

    def compactor() -> dict:
        calls.append("ran")
        return {"reclaimed_bytes": 123, "containers": ["brain.wbrain"]}

    report = reclaim_disk_for_floor(runtime, 1_000_000_000.0, compactor)

    assert calls == ["ran"], "an unreachable floor must attempt compaction"
    assert report["compaction"]["reclaimed_bytes"] == 123
    # The reclaim the CALLER sees is still a df delta, never the compactor's
    # own claim: a reclaim reported rather than measured is how this path has
    # been wrong before.
    assert report["reclaimed_bytes"] == (
        report["free_bytes_after"] - report["free_bytes_before"]
    )
    assert report["cleared_floor"] is False


def test_a_compactor_failure_never_raises_into_the_disk_path(
        tmp_path: Path) -> None:
    """The disk-pressure path has no other diagnostic.

    A raising reclaim would propagate out of the yield and be scored as a
    semantic failure of the interval -- the exact conversion that rolled back
    19 of 19 memory yields.
    """
    runtime = _runtime(tmp_path)

    def compactor() -> dict:
        raise RuntimeError("wbrain_compact exited 1: no manifest")

    report = reclaim_disk_for_floor(runtime, 1_000_000_000.0, compactor)

    assert "wbrain_compact exited 1" in report["compaction"]["error"]
    assert report["compaction"]["reclaimed_bytes"] == 0
    assert report["cleared_floor"] is False


def test_reclaim_without_a_compactor_keeps_its_previous_behaviour(
        tmp_path: Path) -> None:
    """`compactor=None` must leave the existing contract untouched."""
    runtime = _runtime(tmp_path)

    report = reclaim_disk_for_floor(runtime, 1_000_000_000.0)

    assert "compaction" not in report
    assert report["reclaimed_bytes"] == (
        report["free_bytes_after"] - report["free_bytes_before"]
    )


def test_locate_compactor_prefers_a_deployed_binary_over_a_build_tree(
        tmp_path: Path) -> None:
    """`bin/` before `target/release/`.

    "Deploy is not load" is already a paid-for lesson here: a fix copied but
    never loaded ran the OLD code for 96 h. Resolving the deployed path first
    keeps the compactor consistent with how every other binary is shipped.
    """
    project = tmp_path / "project"
    (project / "bin").mkdir(parents=True)
    (project / "target" / "release").mkdir(parents=True)

    build = project / "target" / "release" / "wbrain_compact"
    build.write_text("#!/bin/sh\n")
    assert locate_compactor(project) == build

    deployed = project / "bin" / "wbrain_compact"
    deployed.write_text("#!/bin/sh\n")
    assert locate_compactor(project) == deployed


def test_locate_compactor_returns_none_when_nothing_is_built(
        tmp_path: Path) -> None:
    """A missing compactor must be reported, never guessed at.

    `wbrain_compact --estimate` was recorded as not existing on the deployed
    binary, inferred from a usage banner that has never listed it in any
    version of the source. It works. An absence has to be measured too.
    """
    assert locate_compactor(tmp_path) is None
    assert COMPACTOR_CANDIDATES, "the search path must not be empty"
