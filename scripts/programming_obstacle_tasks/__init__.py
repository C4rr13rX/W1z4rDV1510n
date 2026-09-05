"""Authored held-out tasks for the 1,000-task obstacle course.

Each module in this package owns one capability family from
`docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md` and exports `TASKS`. Keeping
one module per family means a family's shortfall is a single file to open, and
it keeps the authored material physically separate from the training corpora:
nothing under this package is ever a training row.

`load_authored_tasks` deliberately reports what exists rather than asserting
the course is finished. Completeness is decided in one place --
`programming_obstacle_manifest.build_manifest` -- so that a family module
cannot accidentally become the thing that declares the contract satisfied.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Sequence

from scripts.programming_obstacle_manifest import (
    FAMILY_TASK_COUNTS,
    ObstacleTask,
    Provenance,
)

#: Tasks written for this repository against its own acceptance contract.
#: Original work under the repository's licence, so redistribution is
#: permitted. Tasks derived from an external source must carry that source's
#: own record instead -- a licence is never inferred from availability.
REPOSITORY_PROVENANCE = Provenance(
    origin="W1z4rDV1510n acceptance contract, authored for this repository",
    spdx_license_id="MIT",
    redistribution_permitted=True,
    notes="Held-out obstacle material; never admitted as a training row.",
)


def task(task_id: str, family: str, *, prompt: str, validator: str,
         language: str = "python", toolchain: str = "cpython-3.13",
         timeout_seconds: float = 30.0,
         fixtures: dict[str, str] | None = None,
         response_filename: str = "candidate.py",
         provenance: Provenance | None = None) -> ObstacleTask:
    """Construct one task with the repository's default provenance."""
    return ObstacleTask(
        task_id=task_id,
        family=family,
        language=language,
        toolchain=toolchain,
        prompt=prompt,
        validator=validator,
        timeout_seconds=timeout_seconds,
        provenance=provenance or REPOSITORY_PROVENANCE,
        fixtures=dict(fixtures or {}),
        response_filename=response_filename,
    )


def load_authored_tasks() -> list[ObstacleTask]:
    """Collect every authored task across the family modules."""
    collected: list[ObstacleTask] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        tasks: Sequence[ObstacleTask] = getattr(module, "TASKS", ())
        collected.extend(tasks)
    return collected


def authoring_status() -> dict[str, dict[str, int]]:
    """Report authored-vs-required counts per family, for planning."""
    counts = {family: 0 for family in FAMILY_TASK_COUNTS}
    for item in load_authored_tasks():
        if item.family in counts:
            counts[item.family] += 1
    return {
        family: {
            "authored": counts[family],
            "required": required,
            "remaining": required - counts[family],
        }
        for family, required in FAMILY_TASK_COUNTS.items()
    }
