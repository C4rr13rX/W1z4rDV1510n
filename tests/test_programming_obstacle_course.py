"""Contract tests for the deterministic 1,000-task obstacle course.

The course is the evidence the acceptance contract turns on, so these tests
are aimed less at the happy path than at the ways a course could report a
number nobody should believe: a family quietly short, a thousand restatements
of one task, a validator that accepts anything, a validator that accepts
nothing, a partial run reporting admission, or a frozen course edited between
the run and the audit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.programming_obstacle_manifest import (
    FAMILY_TASK_COUNTS,
    TOTAL_TASKS,
    ManifestError,
    Provenance,
    audit_manifest,
    build_manifest,
    freeze_manifest,
    held_out_violations,
    load_manifest,
    normalize_source,
    validator_imports,
)
from scripts.programming_obstacle_run import (
    FAILED,
    NO_RESPONSE,
    PASSED,
    VALIDATOR_ERROR,
    extract_code,
    run_task,
    summarize,
)
from scripts.programming_obstacle_tasks import (
    REPOSITORY_PROVENANCE,
    authoring_status,
    load_authored_tasks,
    task,
)
from tests.obstacle_references import MUTATIONS, REFERENCES

AUTHORED = load_authored_tasks()

GOOD_VALIDATOR = "assert RESPONSE_TEXT\n"
GOOD_PROMPT = (
    "Implement a Python function that does something specific enough for a "
    "caller to depend on, with stated error behaviour."
)


def make_task(task_id="algorithms_data_structures-9001", **overrides):
    fields = dict(
        family="algorithms_data_structures",
        prompt=GOOD_PROMPT,
        validator=GOOD_VALIDATOR,
    )
    fields.update(overrides)
    return task(task_id, **fields)


# --------------------------------------------------------------------------
# The counts are the contract, not a preference.
# --------------------------------------------------------------------------

def test_family_counts_match_the_contract_total():
    assert sum(FAMILY_TASK_COUNTS.values()) == TOTAL_TASKS == 1000


def test_build_refuses_an_incomplete_course():
    """A short course must not produce a manifest at any cost.

    This is the invariant that stops an authoring session in progress from
    being mistaken for a passing course: `build_manifest` is the only way to
    obtain a manifest object, and the run harness only accepts one.
    """
    with pytest.raises(ManifestError) as error:
        build_manifest(AUTHORED)
    message = str(error.value)
    assert "incomplete" in message
    assert f"of {TOTAL_TASKS} tasks" in message


def test_audit_reports_the_exact_shortfall_per_family():
    report = audit_manifest(AUTHORED)
    assert report["complete"] is False
    assert report["total"] == len(AUTHORED)
    for family, required in FAMILY_TASK_COUNTS.items():
        authored = report["family_counts"][family]
        if authored != required:
            assert report["family_shortfalls"][family] == required - authored

    status = authoring_status()
    assert sum(item["authored"] for item in status.values()) == len(AUTHORED)
    assert all(item["remaining"] >= 0 for item in status.values())


def test_an_over_full_family_is_rejected_too():
    """Overshooting a family is as wrong as undershooting it.

    1,000 total with the wrong distribution measures a different capability
    mix than the contract specifies, so the audit must name it rather than
    let the total absorb it.
    """
    padded = list(AUTHORED)
    family = "algorithms_data_structures"
    surplus = FAMILY_TASK_COUNTS[family] + 1
    for index in range(surplus):
        padded.append(make_task(
            f"{family}-{9000 + index:04d}",
            validator=f"assert RESPONSE_TEXT\nassert {index} == {index}\n",
        ))
    report = audit_manifest(padded)
    assert report["family_shortfalls"][family] < 0


# --------------------------------------------------------------------------
# Distinctness is behavioural, so restatement cannot inflate the count.
# --------------------------------------------------------------------------

def test_reworded_duplicates_collide_on_behaviour():
    original = make_task("algorithms_data_structures-9001")
    reworded = make_task(
        "algorithms_data_structures-9002",
        prompt=GOOD_PROMPT + " Phrased completely differently for the reader.",
    )
    assert original.behavior_digest() == reworded.behavior_digest()

    report = audit_manifest([original, reworded])
    assert report["duplicate_behaviors"] == [
        ("algorithms_data_structures-9001", "algorithms_data_structures-9002")
    ]
    assert report["complete"] is False


def test_comment_and_whitespace_changes_do_not_create_a_new_task():
    assert normalize_source("a = 1  # note\n\n  b = 2\n") == "a = 1\nb = 2"
    plain = make_task("algorithms_data_structures-9001")
    dressed = make_task(
        "algorithms_data_structures-9002",
        validator="# a comment\n\nassert RESPONSE_TEXT   \n\n",
    )
    assert plain.behavior_digest() == dressed.behavior_digest()


def test_a_real_behaviour_change_is_a_different_task():
    left = make_task("algorithms_data_structures-9001",
                     validator="assert len(RESPONSE_TEXT) > 0\n")
    right = make_task("algorithms_data_structures-9002",
                      validator="assert len(RESPONSE_TEXT) > 1\n")
    assert left.behavior_digest() != right.behavior_digest()


def test_authored_tasks_are_all_behaviourally_distinct():
    report = audit_manifest(AUTHORED)
    assert report["duplicate_behaviors"] == []
    assert report["duplicate_ids"] == []
    assert report["invalid_tasks"] == []


# --------------------------------------------------------------------------
# A task that cannot produce a trustworthy verdict is a build error.
# --------------------------------------------------------------------------

def test_a_validator_that_asserts_nothing_is_rejected():
    """The contract forbids substituting inspection for behaviour.

    A validator with no assertion passes an empty candidate, which would
    convert an unimplemented capability into a green cell.
    """
    with pytest.raises(ManifestError, match="asserts nothing"):
        make_task(validator="print(RESPONSE_TEXT)\n").validate()


@pytest.mark.parametrize("source", [
    "import socket\nassert RESPONSE_TEXT\n",
    "import urllib.request\nassert RESPONSE_TEXT\n",
    "from requests import get\nassert RESPONSE_TEXT\n",
    "import json, httpx\nassert RESPONSE_TEXT\n",
])
def test_network_dependent_validators_are_rejected(source):
    with pytest.raises(ManifestError, match="network-dependent"):
        make_task(validator=source).validate()


def test_import_scanner_sees_grouped_and_aliased_imports():
    found = validator_imports(
        "import json, socket as s\nfrom urllib import parse\nimport os.path\n"
    )
    assert {"json", "socket", "urllib", "os.path"} <= found


def test_offline_stdlib_imports_stay_allowed():
    make_task(
        validator="import json, subprocess, importlib\nassert RESPONSE_TEXT\n"
    ).validate()


@pytest.mark.parametrize("seconds", [0.5, 0.0, -1.0, 901.0, 100000.0])
def test_timeouts_must_be_bounded(seconds):
    with pytest.raises(ManifestError, match="outside"):
        make_task(timeout_seconds=seconds).validate()


def test_an_unpinned_toolchain_is_rejected():
    with pytest.raises(ManifestError, match="pinned toolchain"):
        make_task(toolchain="  ").validate()


def test_task_ids_must_be_prefixed_by_their_family():
    with pytest.raises(ManifestError, match="prefixed with its family"):
        make_task("http_apis_authn_appsec-0001").validate()


def test_a_fixture_cannot_escape_the_workspace():
    with pytest.raises(ManifestError, match="escapes the workspace"):
        make_task(fixtures={"../../etc/passwd": "x"}).validate()


def test_provenance_requires_a_licence_and_permission():
    with pytest.raises(ManifestError, match="SPDX"):
        make_task(provenance=Provenance(
            origin="a website", spdx_license_id="",
            redistribution_permitted=True)).validate()
    with pytest.raises(ManifestError, match="forbids redistribution"):
        make_task(provenance=Provenance(
            origin="a book", spdx_license_id="LicenseRef-Proprietary",
            redistribution_permitted=False)).validate()


def test_every_authored_task_carries_usable_provenance():
    for item in AUTHORED:
        item.provenance.validate(item.task_id)
        assert item.provenance.spdx_license_id
    assert REPOSITORY_PROVENANCE.redistribution_permitted


# --------------------------------------------------------------------------
# Freezing: a scored course cannot be edited afterwards.
# --------------------------------------------------------------------------

def _complete_task_set():
    """A synthetic full course, used to exercise freeze/load/score paths.

    Each validator embeds its own family and ordinal. That is not decoration:
    behavioural distinctness ignores the family field on purpose, so reusing
    one validator body across families would make this fixture collide with
    itself and fail to build -- which is exactly the check working.
    """
    tasks = []
    for family, required in FAMILY_TASK_COUNTS.items():
        for index in range(required):
            tasks.append(task(
                f"{family}-{index:04d}", family,
                prompt=f"{GOOD_PROMPT} Case {family} number {index}.",
                validator=(
                    "assert RESPONSE_TEXT\n"
                    f"assert {index} >= 0 and {family!r}\n"
                ),
            ))
    return tasks


def test_a_complete_set_builds_and_freezes(tmp_path):
    manifest = build_manifest(_complete_task_set())
    assert len(manifest.tasks) == TOTAL_TASKS
    assert {f: len(items) for f, items in manifest.by_family().items()} == \
        FAMILY_TASK_COUNTS

    target = tmp_path / "manifest.json"
    freeze_manifest(manifest, target)
    reloaded = load_manifest(target)
    assert reloaded.digest == manifest.digest

    # Re-freezing the identical course is a no-op, not an error.
    freeze_manifest(manifest, target)


def test_freezing_over_a_different_course_is_refused(tmp_path):
    target = tmp_path / "manifest.json"
    tasks = _complete_task_set()
    freeze_manifest(build_manifest(tasks), target)

    tasks[0] = task(
        tasks[0].task_id, tasks[0].family,
        prompt=tasks[0].prompt,
        validator="assert RESPONSE_TEXT\nassert 12345 >= 0\n",
    )
    with pytest.raises(ManifestError, match="refusing to overwrite"):
        freeze_manifest(build_manifest(tasks), target)


def test_an_edited_frozen_course_fails_to_load(tmp_path):
    """A stored score must be attributable to the exact tasks that produced it."""
    target = tmp_path / "manifest.json"
    freeze_manifest(build_manifest(_complete_task_set()), target)

    record = json.loads(target.read_text(encoding="utf-8"))
    record["tasks"][0]["validator"] = "assert True\n"
    target.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ManifestError, match="frozen course was edited"):
        load_manifest(target)


# --------------------------------------------------------------------------
# Held-out material must not have leaked into training.
# --------------------------------------------------------------------------

def test_a_prompt_that_reached_the_corpus_is_reported(tmp_path):
    manifest = build_manifest(_complete_task_set())
    leaked = manifest.tasks[0]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"prompt": "unrelated row"}) + "\n"
        + json.dumps({"prompt": leaked.prompt.upper(), "answer": "x"}) + "\n",
        encoding="utf-8",
    )
    violations = held_out_violations(manifest, [corpus])
    assert [item["task_id"] for item in violations] == [leaked.task_id]
    assert violations[0]["line"] == 2


def test_a_shared_preamble_does_not_leak_every_task(tmp_path):
    """One row must not be reported as leaking hundreds of tasks.

    Prompts within a family share an opening on purpose, so a prefix-length
    fingerprint reports every task that starts the same way. Hundreds of
    false leaks would bury the single real one.
    """
    manifest = build_manifest(_complete_task_set())
    leaked = manifest.tasks[0]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"prompt": leaked.prompt.upper()}) + "\n", encoding="utf-8"
    )
    assert [item["task_id"] for item in held_out_violations(manifest, [corpus])] \
        == [leaked.task_id]

    # The shared opening on its own identifies nothing.
    shared = tmp_path / "shared.jsonl"
    shared.write_text(json.dumps({"prompt": GOOD_PROMPT}) + "\n",
                      encoding="utf-8")
    assert held_out_violations(manifest, [shared]) == []


def test_a_clean_corpus_reports_nothing(tmp_path):
    manifest = build_manifest(_complete_task_set())
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"prompt": "ordinary training row"}) + "\n",
                      encoding="utf-8")
    assert held_out_violations(manifest, [corpus]) == []
    # A missing corpus is not a silent pass turned into a crash.
    assert held_out_violations(manifest, [tmp_path / "absent.jsonl"]) == []


# --------------------------------------------------------------------------
# The runner attributes every non-pass to the right cause.
# --------------------------------------------------------------------------

def test_extract_code_takes_the_fenced_program():
    assert extract_code("no fence here") == "no fence here"
    assert extract_code("prose\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code("```\nx = 1\n```") == "x = 1\n"
    assert extract_code("a\n```py\nx = 1\n```\nb\n```py\ny = 2\n```") == \
        "x = 1\n\n\ny = 2\n"


def test_an_empty_answer_is_not_a_failure():
    result = run_task(make_task(), "   ")
    assert result.outcome == NO_RESPONSE


def test_a_broken_validator_is_not_a_capability_verdict():
    """A harness fault must never be scored as a failing capability.

    Attributing a broken validator to the brain sends repair effort at
    curriculum that was never the problem -- the same misattribution that
    made a SIGTERMed replay worker look like a failing admission gate.
    """
    result = run_task(
        make_task(validator="assert True\nraise RuntimeError('harness')\n"),
        "x = 1",
    )
    assert result.outcome == VALIDATOR_ERROR
    assert "harness" in result.detail


def test_a_candidate_that_does_not_parse_is_a_capability_failure():
    result = run_task(
        make_task(validator=(
            "import importlib.util as u\n"
            "s = u.spec_from_file_location('c', RESPONSE_PATH)\n"
            "m = u.module_from_spec(s)\n"
            "s.loader.exec_module(m)\n"
            "assert hasattr(m, 'f')\n"
        )),
        "def f(:\n    pass\n",
    )
    assert result.outcome == FAILED


def test_a_hanging_candidate_times_out_rather_than_stalling():
    result = run_task(
        make_task(validator="import time\ntime.sleep(30)\nassert True\n",
                  timeout_seconds=2.0),
        "x = 1",
    )
    assert result.outcome == "timeout"
    assert result.duration_seconds < 20


def test_fixtures_are_materialized_in_the_workspace():
    result = run_task(
        make_task(
            fixtures={"data/input.txt": "seven"},
            validator=(
                "text = (WORKSPACE / 'data' / 'input.txt')"
                ".read_text(encoding='utf-8')\n"
                "assert text == 'seven', text\n"
            ),
        ),
        "x = 1",
    )
    assert result.outcome == PASSED


def test_summarize_never_reports_admission_on_a_partial_run():
    manifest = build_manifest(_complete_task_set())
    from scripts.programming_obstacle_run import TaskResult

    partial = [
        TaskResult(item.task_id, item.family, PASSED, 0.1)
        for item in manifest.tasks[:10]
    ]
    report = summarize(partial, manifest)
    assert report["passed"] == 10
    assert report["full_course"] is False
    assert report["admitted"] is False


def test_summarize_blocks_admission_on_a_validator_error():
    manifest = build_manifest(_complete_task_set())
    from scripts.programming_obstacle_run import TaskResult

    results = [
        TaskResult(item.task_id, item.family, PASSED, 0.1)
        for item in manifest.tasks
    ]
    assert summarize(results, manifest)["admitted"] is True

    results[5] = TaskResult(results[5].task_id, results[5].family,
                            VALIDATOR_ERROR, 0.1, "harness broke")
    report = summarize(results, manifest)
    assert report["validator_errors"] == 1
    assert report["admitted"] is False
    assert results[5].family in report["failing_families"]


# --------------------------------------------------------------------------
# Every authored validator is satisfiable AND discriminating.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("item", AUTHORED, ids=lambda item: item.task_id)
def test_reference_solution_passes_its_validator(item):
    """A validator nobody has seen pass reports failure forever.

    Without this direction the course could be unpassable by construction and
    every run would blame the brain for it.
    """
    reference = REFERENCES.get(item.task_id)
    assert reference, f"{item.task_id} has no reference solution"
    result = run_task(item, reference)
    assert result.outcome == PASSED, (
        f"{item.task_id} rejected its own reference solution:\n"
        f"{result.detail}"
    )


@pytest.mark.parametrize("item", AUTHORED, ids=lambda item: item.task_id)
def test_a_broken_solution_fails_its_validator(item):
    """And a validator that passes everything measures nothing.

    The mutation is asserted to have actually applied, because a stale
    find-string would silently make this test vacuous -- passing while
    checking the unmodified reference a second time.
    """
    reference = REFERENCES[item.task_id]
    find, replace = MUTATIONS[item.task_id]
    assert find in reference, (
        f"{item.task_id}: mutation target is stale, so this test would "
        "re-check the unmodified reference"
    )
    broken = reference.replace(find, replace, 1)
    assert broken != reference

    result = run_task(item, broken)
    assert result.outcome == FAILED, (
        f"{item.task_id} accepted a candidate with broken behaviour "
        f"(outcome={result.outcome}): {result.detail[:400]}"
    )
