#!/usr/bin/env python3
"""Find obstacle validators that score the harness instead of the candidate.

`docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md` admits the brain only on
`1000/1000` "with no skipped, manually waived, network-dependent, flaky, or
validator-error cases". `validator_error` is decided by attribution: the run
harness blames the candidate for any exception whose traceback passes through
the candidate's file, and blames itself for one raised purely in validator
frames. A validator that subscripts data the candidate produced therefore
turns the defect it exists to catch into a harness fault -- the failure is
recorded as the course being broken, and it blocks admission.

That is not hypothetical. Authoring `architecture_multifile_integration-0401`
to `-0405` on 2026-09-09, three of five validators did exactly this:
`result['region']` raised KeyError when the candidate dropped a key,
`db.get('outbox:x')['state']` raised TypeError on None when it never wrote the
row, and `next(walk)` raised StopIteration when a generator ended early. All
three are the behaviour under test.

This module audits the whole authored course for the same defect empirically
rather than by reading for subscripts, because whether a subscript is reachable
depends on what the candidate returns. Each task is run against candidates that
are wrong in the most ordinary ways a real answer is wrong:

- `absent`: a module defining none of the required names. Every validator
  begins with the `require()` assertions, so this must be a clean `failed`,
  and a task that manages `validator_error` here is broken before the
  behaviour is even reached.
- `stub`: a module defining every required name as a function whose body is
  `pass`. This is the single most likely wrong answer a language model gives,
  and it is what drives None into every place the validator reads a result.

A task is reported when either candidate produces `validator_error`. The
verdict this module cares about is only ever `validator_error` versus
everything else: `failed` is correct, and `timeout` is a separate concern.

Read-only with respect to the brain. It never contacts an endpoint -- the
candidates are synthesized here, so nothing about this audit can become a
training row.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.programming_obstacle_run import VALIDATOR_ERROR, run_task  # noqa: E402
from scripts.programming_obstacle_tasks import load_authored_tasks  # noqa: E402

#: `_support.require` emits exactly this line, so the required public names
#: can be recovered from the validator source without executing it.
_REQUIRED = re.compile(
    r"assert hasattr\(candidate, '([^']+)'\)")


def required_names(validator: str) -> list[str]:
    return list(dict.fromkeys(_REQUIRED.findall(validator)))


def stub_module(names: list[str]) -> str:
    """A candidate that defines every required name and implements none.

    Written as functions rather than classes on purpose: a model that has
    understood the signature and not the behaviour produces this, and calling
    it yields None wherever the validator expects a result.
    """
    if not names:
        return "# nothing was required\n"
    body = "\n\n".join(
        f"def {name}(*args, **kwargs):\n    pass" for name in names)
    return body + "\n"


def audit(tasks, families=None) -> dict:
    findings = []
    outcomes = Counter()
    for item in tasks:
        if families and item.family not in families:
            continue
        if item.language != "python":
            continue
        names = required_names(item.validator)
        candidates = {
            "absent": "# defines nothing the prompt asked for\n",
            "stub": stub_module(names),
        }
        for label, source in candidates.items():
            result = run_task(item, source)
            outcomes[f"{label}:{result.outcome}"] += 1
            if result.outcome == VALIDATOR_ERROR:
                findings.append({
                    "task_id": item.task_id,
                    "family": item.family,
                    "candidate": label,
                    "required": names,
                    "detail": (result.detail or "")[-600:],
                })
    return {"findings": findings, "outcomes": dict(sorted(outcomes.items()))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    tasks = load_authored_tasks()
    report = audit(tasks, families=set(args.family) or None)
    report["audited_tasks"] = sum(
        1 for t in tasks
        if t.language == "python"
        and (not args.family or t.family in set(args.family))
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True),
                               encoding="utf-8")

    print(json.dumps(report["outcomes"], indent=2, sort_keys=True))
    print(f"\naudited {report['audited_tasks']} python tasks")
    if report["findings"]:
        print(f"{len(report['findings'])} validator_error finding(s):")
        for finding in report["findings"]:
            print(f"  {finding['task_id']} [{finding['candidate']}] "
                  f"required={finding['required']}")
        return 1
    print("no task scored validator_error against a trivially wrong candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
