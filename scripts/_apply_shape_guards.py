#!/usr/bin/env python3
"""One-shot authoring aid: install SHAPE_GUARDS on the tasks the audit names.

Not part of the course and not imported by it. The repair is mechanical in
shape but not in content -- which names are classes, which are exception types
that must NOT be wrapped, and which members each validator reads all come from
reading the validator -- so those decisions are spelled out in PLAN below and
this only performs the edit. Re-running it is safe: a task that already has
SHAPE_GUARDS is skipped.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = ROOT / "scripts" / "programming_obstacle_tasks"

#: task id -> list of (kind, name, *members)
#:   ("class", Name, *members)  wrap the constructor, assert the members
#:   ("call",  Name)            wrap the callable, assert it returned something
#: Exception types required by a task are deliberately absent: they are used in
#: `except` clauses, and a wrapper function is not catchable.
PLAN: dict[str, list[tuple]] = {
    "cicd_containers_packaging_platform-0001": [("call", "build_archive")],
    "cicd_containers_packaging_platform-0003": [
        ("call", "chain_ids"), ("call", "image_id")],
    "cicd_containers_packaging_platform-0005": [("call", "evaluate_pipeline")],

    "concurrency_async_distributed-0001": [("class", "TokenBucket", "allow")],
    "concurrency_async_distributed-0002": [("class", "CircuitBreaker", "call")],
    "concurrency_async_distributed-0005": [
        ("class", "ExactlyOnceInbox", "deliver", "pending")],
    "concurrency_async_distributed-0007": [("class", "Once", "do", "done")],
    "concurrency_async_distributed-0008": [
        ("class", "ReadWriteLock", "acquire_read", "acquire_write",
         "release_read", "release_write")],
    "concurrency_async_distributed-0009": [
        ("class", "HashRing", "add", "get", "remove")],
    "concurrency_async_distributed-0010": [
        ("class", "PNCounter", "increment", "decrement", "merge", "value")],
    "concurrency_async_distributed-0011": [
        ("class", "LeaseManager", "acquire", "release"),
        ("class", "FencedStore", "read", "write")],
    "concurrency_async_distributed-0013": [
        ("class", "TumblingWindows", "add", "advance_watermark",
         "dropped_count")],
    "concurrency_async_distributed-0301": [
        ("class", "SlidingWindowLimiter", "allow", "retry_after")],

    "databases_migrations_transactions-0004": [("call", "page_after")],
    "databases_migrations_transactions-0005": [("call", "apply_batch")],
    "databases_migrations_transactions-0008": [("call", "process_payment")],
    "databases_migrations_transactions-0009": [("call", "build_filter_query")],
    "databases_migrations_transactions-0011": [("call", "summarize_regions")],

    "frontend_state_ux_accessibility-0001": [
        ("call", "contrast_ratio"), ("call", "meets_wcag")],
    "frontend_state_ux_accessibility-0005": [
        ("class", "VirtualList", "offset_of", "total_height", "window")],
    "frontend_state_ux_accessibility-0006": [
        ("class", "EditHistory", "apply", "can_redo", "can_undo", "current",
         "redo", "undo")],
    "frontend_state_ux_accessibility-0010": [
        ("class", "FormState", "blur", "change", "dirty", "errors", "reset",
         "submit", "touched", "values")],

    "http_apis_authn_appsec-0008": [("call", "redact")],
    "http_apis_authn_appsec-0015": [
        ("class", "SessionStore", "authenticate", "create", "get", "logout",
         "set")],
    "http_apis_authn_appsec-0017": [("class", "IdempotentStore", "execute")],

    "polyglot_native_interop-0202": [("call", "crc16")],

    "reliability_observability_performance-0001": [
        ("class", "LatencyHistogram", "bucket_count", "count", "quantile",
         "record")],
    "reliability_observability_performance-0003": [
        ("class", "SlidingWindowCounter", "record", "resident_buckets",
         "total")],
    "reliability_observability_performance-0004": [("call", "aggregate_health")],
    "reliability_observability_performance-0005": [
        ("call", "evaluate_error_budget")],
    "reliability_observability_performance-0006": [
        ("class", "EventThrottle", "offer", "tracked_keys")],
    "reliability_observability_performance-0007": [
        ("class", "FrequentItems", "offer", "top", "tracked")],
    "reliability_observability_performance-0008": [("call", "critical_path")],
    "reliability_observability_performance-0009": [("call", "redact_record")],
    "reliability_observability_performance-0010": [
        ("class", "BoundedLabelRegistry", "observe", "series")],
    "reliability_observability_performance-0011": [("call", "reservoir_sample")],
    "reliability_observability_performance-0012": [
        ("class", "DecayingCounter", "add", "value")],
    "reliability_observability_performance-0015": [
        ("class", "LogTail", "append", "lines")],
    "reliability_observability_performance-0016": [
        ("class", "BatchFlusher", "add", "flush", "tick")],

    "requirements_api_contracts-0005": [
        ("class", "CursorPage", "delete", "insert", "page")],
    "requirements_api_contracts-0007": [
        ("class", "IdempotentEndpoint", "submit")],
    "requirements_api_contracts-0105": [
        ("call", "classify_signature_change")],
    "requirements_api_contracts-0108": [
        ("class", "OperationTracker", "cancel", "fail", "poll", "report",
         "start", "succeed")],

    "testing_debugging_repair_refactoring-0003": [("call", "group_failures")],
    "testing_debugging_repair_refactoring-0004": [
        ("call", "rank_suspicious_lines")],
    "testing_debugging_repair_refactoring-0007": [
        ("call", "classify_test_history")],
    "testing_debugging_repair_refactoring-0009": [("call", "extract_function")],
    "testing_debugging_repair_refactoring-0010": [("call", "inline_variable")],
    "testing_debugging_repair_refactoring-0015": [("call", "capture_call_tree")],
    "testing_debugging_repair_refactoring-0017": [
        ("call", "simplify_control_flow")],
    "testing_debugging_repair_refactoring-0018": [("call", "compare_snapshot")],
    "testing_debugging_repair_refactoring-0203": [("call", "tidy_imports")],
    "testing_debugging_repair_refactoring-0208": [
        ("call", "rewrite_module_path")],
    "testing_debugging_repair_refactoring-0209": [
        ("call", "audit_resource_lifetimes")],

    "validation_parsing_serialization-0004": [("call", "parse_duration")],
    "validation_parsing_serialization-0015": [("call", "parse_multipart")],
    "validation_parsing_serialization-0019": [("call", "decode_chunked")],
}

_HEADER = """from scripts.programming_obstacle_tasks._support import (
    LOAD_CANDIDATE,
    SHAPE_GUARDS,
    require,
)"""


def preamble(entries: list[tuple]) -> str:
    """The wrapper lines to insert at the top of a validator body."""
    lines = [
        "# Guard every call, not just the first: `require` proves a name",
        "# exists, never that it is the right KIND of thing, and an",
        "# AttributeError on the result is raised in validator frames alone.",
    ]
    for entry in entries:
        kind, name = entry[0], entry[1]
        if kind == "class":
            members = ", ".join(repr(m) for m in entry[2:])
            lines += [
                f"_{name} = {name}",
                f"def {name}(*args, **kwargs):",
                f"    return having(_{name}(*args, **kwargs), {members},",
                f"                  what='{name}(...)')",
            ]
        else:
            lines.append(
                f"{name} = returning({name}, '{name}(...)')")
    return "\n".join(lines) + "\n"


def main() -> int:
    changed, skipped = [], []
    for path in sorted(TASKS.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        text = original = path.read_text(encoding="utf-8")
        family = path.stem
        wanted = {k: v for k, v in PLAN.items() if k.startswith(family + "-")}
        if not wanted:
            continue
        if "SHAPE_GUARDS" not in text:
            old = ("from scripts.programming_obstacle_tasks._support import "
                   "LOAD_CANDIDATE, require")
            if old not in text:
                print(f"!! {family}: unrecognised import block", file=sys.stderr)
                return 1
            text = text.replace(old, _HEADER, 1)

        for task_id, entries in sorted(wanted.items()):
            suffix = task_id.rsplit("-", 1)[1]
            anchor = f'f"{{FAMILY}}-{suffix}", FAMILY,'
            at = text.find(anchor)
            if at < 0:
                print(f"!! {task_id}: not found", file=sys.stderr)
                return 1
            # The validator expression for THIS task: from its `validator=`
            # to the opening quote of the body literal.
            vstart = text.index("validator=", at)
            body = re.compile(r'(\+\s*)?(r?)("""|\'\'\')')
            match = None
            for candidate in body.finditer(text, vstart):
                match = candidate
                break
            if match is None:
                print(f"!! {task_id}: no body literal", file=sys.stderr)
                return 1
            segment = text[vstart:match.start()]
            if "SHAPE_GUARDS" in segment:
                skipped.append(task_id)
                continue
            quote = match.group(3)
            insert_at = match.end()
            head = text[:match.start()] + "+ SHAPE_GUARDS " + match.group(0)
            rest = text[insert_at:]
            # The body always opens with a newline; keep it.
            assert rest.startswith("\n"), task_id
            text = head + "\n" + preamble(entries) + rest[1:]
            changed.append(task_id)

        if text != original:
            path.write_text(text, encoding="utf-8")

    print(f"guarded {len(changed)} task(s); {len(skipped)} already guarded")
    for task_id in changed:
        print("  +", task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
