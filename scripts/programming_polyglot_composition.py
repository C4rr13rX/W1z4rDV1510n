#!/usr/bin/env python3
"""Compile never-trained polyglot repositories composed by the coding brain."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from programming_native_enterprise_eval import CASES as NATIVE_CASES, execute
from programming_project_eval import request


NATIVE = {case.name: case for case in NATIVE_CASES}


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    paraphrase: str
    components: tuple[str, ...]


CASES = [
    Case(
        "javascript_go_order_workers",
        "Build a polyglot project with a JavaScript transactional-outbox order service and a Go concurrency-safe work deduplicator.",
        "Create idempotent Node.js outbox-event ordering code plus Golang synchronization that suppresses duplicate work in one repository.",
        ("javascript_outbox", "go_deduplication"),
    ),
    Case(
        "java_rust_consistency",
        "Create a polyglot project combining a Java optimistic-concurrency store with a Rust atomic ledger transfer.",
        "Build Java expected-version storage and Rust all-or-nothing account transfers together in one codebase.",
        ("java_optimistic_store", "rust_atomic_ledger"),
    ),
    Case(
        "csharp_javascript_resilience",
        "Implement a polyglot project containing a C# bounded asynchronous retry policy and a JavaScript transactional outbox service.",
        "Write C sharp async retry code with maximum attempts alongside Node.js idempotent order creation with an outbox event.",
        ("csharp_async_retry", "javascript_outbox"),
    ),
]

OOV = [
    "Build a Go and Rust integration for two undocumented wire protocols.",
    "Create a JavaScript and Java deployment project whose service contracts have not been provided.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runtime/benchmarks/polyglot_composition.json"),
    )
    args = parser.parse_args()
    rows = []
    for case in CASES:
        for kind, prompt in (("canonical", case.prompt), ("paraphrase", case.paraphrase)):
            response = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(response.get("reply") or "")
            component_results = []
            for component_name in case.components:
                passed, detail = execute(NATIVE[component_name], reply)
                component_results.append({
                    "component": component_name,
                    "executes": passed,
                    "detail": "" if passed else detail,
                })
            try:
                files = json.loads(reply).get("files", {})
            except (json.JSONDecodeError, AttributeError):
                files = {}
            rows.append({
                "name": case.name,
                "kind": kind,
                "executes": all(item["executes"] for item in component_results),
                "components": component_results,
                "files": sorted(files) if isinstance(files, dict) else [],
                "intent_diagnostics": response.get("intent_diagnostics"),
            })
    oov = []
    for prompt in OOV:
        response = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not response.get("reply") and bool(
            (response.get("grounding") or {}).get("outside_grounding")
        )
        oov.append({"prompt": prompt, "honest": honest, "reply": response.get("reply")})
    summary = {
        "polyglot_projects": {"passed": sum(row["executes"] for row in rows), "total": len(rows)},
        "component_executions": {
            "passed": sum(item["executes"] for row in rows for item in row["components"]),
            "total": sum(len(row["components"]) for row in rows),
        },
        "oov_honesty": {"passed": sum(row["honest"] for row in oov), "total": len(oov)},
    }
    report = {"summary": summary, "results": rows, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in rows) and all(row["honest"] for row in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
