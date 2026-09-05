#!/usr/bin/env python3
"""Run the frozen obstacle course in bounded, disposable workspaces.

The acceptance contract admits the brain only on `1000/1000` "with no skipped,
manually waived, network-dependent, flaky, or validator-error cases". That
sentence is the whole design of this runner: a score is only meaningful if
every non-pass is attributed to the right cause, so the outcome vocabulary
distinguishes a candidate that behaved wrongly from a harness or toolchain
that broke.

    passed          the validator executed the behaviour contract and it held
    failed          the contract was executed and the candidate violated it
    timeout         the candidate or validator exceeded the task's bound
    no_response     the brain produced nothing to validate
    validator_error the validator itself broke -- a harness fault, never a
                    capability verdict, and never admissible

That last distinction is the one this repository has paid for before: a replay
worker SIGTERMed *before* its gate ran looked exactly like a gate that ran and
returned a verdict, and the two have opposite fixes. `validator_error` is
counted separately and blocks admission rather than being folded into the
failure total, because folding it in would make a broken harness look like a
capability gap and send repair effort at curriculum that was never the
problem.

Runs are read-only with respect to the brain. Obstacle prompts must never
become training rows, so responses are collected over the conversational
endpoint and this module never posts an observation.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.programming_exec_env import (  # noqa: E402
    evaluation_tool_env,
    tool_output_detail,
)
from scripts.programming_obstacle_manifest import (  # noqa: E402
    FAMILY_TASK_COUNTS,
    FAMILY_TITLES,
    ObstacleManifest,
    ObstacleTask,
    load_manifest,
)

PASSED = "passed"
FAILED = "failed"
TIMEOUT = "timeout"
NO_RESPONSE = "no_response"
VALIDATOR_ERROR = "validator_error"

#: Outcomes that are legitimate verdicts about the candidate. Anything else
#: is a statement about the harness and must not be scored.
VERDICT_OUTCOMES = frozenset({PASSED, FAILED, TIMEOUT})

_EXIT_CONTRACT_VIOLATED = 1
_EXIT_VALIDATOR_BROKE = 2

#: Wraps every validator so the process exit code carries the distinction the
#: contract needs. An `AssertionError` is the validator saying the candidate
#: is wrong; any other exception is the validator saying *it* is wrong, and
#: those must never be scored the same way.
_HARNESS_PREAMBLE = '''\
import json, os, pathlib, sys, traceback

WORKSPACE = pathlib.Path(os.environ["OBSTACLE_WORKSPACE"]).resolve()
RESPONSE_PATH = pathlib.Path(os.environ["OBSTACLE_RESPONSE_PATH"]).resolve()
RESPONSE_TEXT = RESPONSE_PATH.read_text(encoding="utf-8", errors="replace")
TIMEOUT_SECONDS = float(os.environ["OBSTACLE_TIMEOUT_SECONDS"])
_CANDIDATE_FILE = os.path.normcase(str(RESPONSE_PATH))
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))


def _emit(stage, detail):
    sys.stderr.write(json.dumps({"stage": stage, "detail": str(detail)[:4000]}))
    sys.stderr.write("\\n")


def _blames_candidate(error):
    """Decide whether the candidate or the validator raised.

    A candidate that raises where the contract says it should return -- or
    that fails to parse -- is a capability failure, and the most common shape
    of that failure is an ordinary exception rather than a clean False. What
    makes it a *capability* verdict is that the exception passed through the
    candidate's own code, so attribution follows the traceback frames rather
    than the exception type. Only an error raised purely in validator frames
    is a harness fault.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        filename = getattr(error, "filename", None)
        if filename and os.path.normcase(str(filename)) == _CANDIDATE_FILE:
            return True
        frame = error.__traceback__
        while frame is not None:
            origin = frame.tb_frame.f_code.co_filename
            if os.path.normcase(str(origin)) == _CANDIDATE_FILE:
                return True
            frame = frame.tb_next
        error = error.__cause__ or error.__context__
    return False


try:
    exec(compile(VALIDATOR_SOURCE, "<validator>", "exec"), {
        "__name__": "__obstacle_validator__",
        "WORKSPACE": WORKSPACE,
        "RESPONSE_PATH": RESPONSE_PATH,
        "RESPONSE_TEXT": RESPONSE_TEXT,
        "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
    })
except AssertionError:
    _emit("contract_violated", traceback.format_exc())
    sys.exit(%(violated)d)
except Exception as error:
    if _blames_candidate(error):
        _emit("contract_violated", traceback.format_exc())
        sys.exit(%(violated)d)
    _emit("validator_error", traceback.format_exc())
    sys.exit(%(broke)d)
sys.exit(0)
''' % {"violated": _EXIT_CONTRACT_VIOLATED, "broke": _EXIT_VALIDATOR_BROKE}


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    family: str
    outcome: str
    duration_seconds: float
    detail: str = ""

    @property
    def admissible(self) -> bool:
        return self.outcome == PASSED


class BrainClient:
    """Read-only conversational client for the programming brain.

    Deliberately offers no observe/pretrain call. The obstacle prompts are
    held-out material, and the cheapest way to destroy the course would be a
    runner that trained on the very prompts it was scoring.
    """

    def __init__(self, endpoint: str, timeout: float = 300.0) -> None:
        cleaned = endpoint.split("://", 1)[-1].rstrip("/")
        host, _, port = cleaned.partition(":")
        self.host = host
        self.port = int(port or 80)
        self.timeout = timeout

    def chat(self, prompt: str) -> str:
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout
        )
        try:
            body = json.dumps({"message": prompt}).encode("utf-8")
            connection.request(
                "POST", "/brain/chat", body,
                {"Content-Type": "application/json"},
            )
            raw = connection.getresponse().read().decode("utf-8", "replace")
        finally:
            connection.close()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # The brain answers in prose on several routes; a non-JSON body is
            # a valid answer, not a transport failure.
            return raw
        for key in ("answer", "response", "message", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return raw


def extract_code(response: str) -> str:
    """Take the candidate program out of a conversational answer.

    The brain answers with prose around fenced code on most routes. Fence
    stripping is presentation handling, not grading: what the validator then
    executes is the candidate's own behaviour, unmodified.
    """
    text = response.replace("\r\n", "\n")
    if "```" not in text:
        return text
    blocks: list[str] = []
    parts = text.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index]
        newline = block.find("\n")
        if newline != -1 and " " not in block[:newline].strip():
            block = block[newline + 1:]
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else text


def run_task(task: ObstacleTask, response: str, *,
             workspace_root: Path | None = None) -> TaskResult:
    """Execute one task's validator against one candidate response."""
    started = time.monotonic()
    if not response.strip():
        return TaskResult(task.task_id, task.family, NO_RESPONSE, 0.0,
                          "brain returned an empty answer")

    parent = workspace_root or Path(tempfile.gettempdir()) / "obstacle-course"
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"{task.task_id}-", dir=parent))
    try:
        for name, body in task.fixtures.items():
            fixture = workspace / name
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(body, encoding="utf-8")

        response_path = workspace / task.response_filename
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(extract_code(response), encoding="utf-8")

        harness = workspace / "_obstacle_harness.py"
        harness.write_text(
            "VALIDATOR_SOURCE = "
            + repr(task.validator)
            + "\n"
            + _HARNESS_PREAMBLE,
            encoding="utf-8",
        )

        environment = evaluation_tool_env(task.language, workspace)
        environment.update({
            "OBSTACLE_WORKSPACE": str(workspace),
            "OBSTACLE_RESPONSE_PATH": str(response_path),
            "OBSTACLE_TIMEOUT_SECONDS": str(task.timeout_seconds),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        })

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(harness)],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return TaskResult(
                task.task_id, task.family, TIMEOUT,
                time.monotonic() - started,
                f"exceeded {task.timeout_seconds}s",
            )

        duration = time.monotonic() - started
        detail = tool_output_detail(completed)
        if completed.returncode == 0:
            return TaskResult(task.task_id, task.family, PASSED, duration)
        if completed.returncode == _EXIT_CONTRACT_VIOLATED:
            return TaskResult(task.task_id, task.family, FAILED, duration,
                              detail)
        return TaskResult(task.task_id, task.family, VALIDATOR_ERROR, duration,
                          detail)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def run_course(manifest: ObstacleManifest,
               responder: Callable[[ObstacleTask], str],
               *, families: Sequence[str] | None = None,
               workspace_root: Path | None = None,
               on_result: Callable[[TaskResult], None] | None = None,
               ) -> list[TaskResult]:
    selected = [
        task for task in manifest.tasks
        if not families or task.family in families
    ]
    results: list[TaskResult] = []
    for task in selected:
        try:
            response = responder(task)
        except Exception as error:  # transport faults are harness faults
            results.append(TaskResult(task.task_id, task.family,
                                      VALIDATOR_ERROR, 0.0,
                                      f"responder failed: {error}"))
            if on_result:
                on_result(results[-1])
            continue
        result = run_task(task, response, workspace_root=workspace_root)
        results.append(result)
        if on_result:
            on_result(result)
    return results


def summarize(results: Iterable[TaskResult],
              manifest: ObstacleManifest | None = None) -> dict:
    """Score a run, and say plainly whether the score may be trusted.

    `admitted` is only true on a full-course run in which every task passed
    and nothing was attributed to the harness. A partial run can never report
    admission however well it scored, because the contract's threshold is the
    complete frozen course.
    """
    results = list(results)
    outcomes = Counter(result.outcome for result in results)
    by_family: dict[str, Counter] = {}
    for result in results:
        by_family.setdefault(result.family, Counter())[result.outcome] += 1

    passed = outcomes[PASSED]
    total = len(results)
    full_course = (
        manifest is not None
        and total == len(manifest.tasks)
        and all(
            by_family.get(family, Counter()).total() == count
            for family, count in FAMILY_TASK_COUNTS.items()
        )
    )
    return {
        "total": total,
        "passed": passed,
        "outcomes": dict(outcomes),
        "validator_errors": outcomes[VALIDATOR_ERROR],
        "families": {
            family: {
                "title": FAMILY_TITLES.get(family, family),
                "required": FAMILY_TASK_COUNTS.get(family, 0),
                "outcomes": dict(counter),
                "passed": counter[PASSED],
            }
            for family, counter in sorted(by_family.items())
        },
        "failing_families": sorted(
            family for family, counter in by_family.items()
            if counter[PASSED] != counter.total()
        ),
        "full_course": full_course,
        "course_digest": manifest.digest if manifest else "",
        "admitted": bool(
            full_course
            and passed == total
            and outcomes[VALIDATOR_ERROR] == 0
            and outcomes[NO_RESPONSE] == 0
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get(
        "PROGRAMMING_BRAIN_ENDPOINT", "127.0.0.1:8095"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--family", action="append", default=[],
                        choices=sorted(FAMILY_TASK_COUNTS))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--responses", type=Path, default=None,
                        help="re-score a stored {task_id: response} file "
                             "instead of querying the brain")
    parser.add_argument("--workspace-root", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)

    if args.responses:
        stored = json.loads(args.responses.read_text(encoding="utf-8"))
        def responder(task: ObstacleTask) -> str:
            return stored.get(task.task_id, "")
    else:
        client = BrainClient(args.endpoint)
        def responder(task: ObstacleTask) -> str:
            return client.chat(task.prompt)

    def progress(result: TaskResult) -> None:
        print(f"{result.outcome:>15}  {result.task_id}", flush=True)

    results = run_course(manifest, responder, families=args.family,
                         workspace_root=args.workspace_root,
                         on_result=progress)
    report = summarize(results, manifest)
    report["results"] = [asdict(result) for result in results]

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: value for key, value in report.items()
                      if key != "results"}, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
