"""training_standard/runner.py — execute benchmarks + detect regressions.

CLI usage:

  # Run benchmarks for one script (no training side-effects):
  python -m tools.training_standard.runner benchmark <script_id>

  # Mark a phase as freshly trained and run its benchmarks plus every
  # regression_protects reference.  Use this from run_all_training.sh
  # immediately after the phase completes:
  python -m tools.training_standard.runner mark-trained <script_id>

  # List the registry and validate the schema:
  python -m tools.training_standard.runner list

  # Show recent events:
  python -m tools.training_standard.runner report [--tail N]

  # Send everything in the registry through its benchmarks against the
  # currently-running node.  Useful for a cold sanity check.
  python -m tools.training_standard.runner benchmark-all

Event log: data/training_events.jsonl (configurable via
W1Z4RD_TRAINING_EVENTS env var).  Append-only, one JSON object per line.
Each event has at least {ts, kind, script_id} plus kind-specific fields.

The node URL defaults to http://127.0.0.1:8090 and is overridable via
W1Z4RD_NODE_URL.  The runner sends each benchmark prompt to /chat and
scores the response; it does NOT itself perform training.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .schema import TrainingScript, Benchmark, load_registry, SchemaError
from .score import evaluate, BenchmarkResult


# ── Paths ──────────────────────────────────────────────────────────────────

PACKAGE_ROOT = Path(__file__).resolve().parent
REGISTRY_DIR = PACKAGE_ROOT / "registry"

_DEFAULT_EVENTS = Path(
    os.environ.get(
        "W1Z4RD_TRAINING_EVENTS",
        str(Path(os.environ.get("W1Z4RDV1510N_DATA_DIR", "D:/w1z4rdv1510n-data"))
            / "training" / "training_events.jsonl"),
    )
)
NODE_URL = os.environ.get("W1Z4RD_NODE_URL", "http://127.0.0.1:8090")


# ── Event log ──────────────────────────────────────────────────────────────


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_event(event: dict[str, Any], path: Path = _DEFAULT_EVENTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": _ts(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


def read_events(path: Path = _DEFAULT_EVENTS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ── Node probe ─────────────────────────────────────────────────────────────


def _probe_node(prompt: str, timeout: float = 30.0) -> tuple[str, str]:
    """Send `prompt` to the node's /chat endpoint.  Returns (response, error).
    On HTTP failure response is '' and error is set."""
    url = f"{NODE_URL.rstrip('/')}/chat"
    body = json.dumps({"text": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw, ""
        # Common keys produced by the node — accept whichever is present.
        for key in ("answer", "response", "reply", "text", "output"):
            if isinstance(data.get(key), str):
                return data[key], ""
        return raw, ""
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        return "", str(exc)


# ── Benchmark execution ────────────────────────────────────────────────────


def run_benchmarks(script: TrainingScript) -> list[BenchmarkResult]:
    """Probe every benchmark and score the response.  Pure orchestration
    around _probe_node + evaluate; everything verifiable lives in score.py."""
    results: list[BenchmarkResult] = []
    for bench in script.benchmarks:
        response, error = _probe_node(bench.prompt)
        if error:
            results.append(BenchmarkResult(
                benchmark_label=bench.label,
                prompt=bench.prompt,
                response="",
                score=0.0,
                passed=False,
                breakdown={"keyword_score": 0.0, "forbidden_hits": [],
                            "structural_ok": False, "structural_err": ""},
                error=error,
            ))
            continue
        results.append(evaluate(response, bench))
    return results


def _last_pass_for(script_id: str, bench_label: str,
                    events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most recent benchmark event that passed for this
    (script_id, bench_label) pair, or None if there has never been one."""
    for ev in reversed(events):
        if (ev.get("kind") == "benchmark"
                and ev.get("script_id") == script_id
                and ev.get("benchmark_label") == bench_label
                and ev.get("passed")):
            return ev
    return None


def _scripts_run_between(events: list[dict[str, Any]],
                          start_ts: str, end_ts: str) -> list[str]:
    """Order-preserved list of script_ids marked trained between two
    timestamps.  Used to populate regression_alert.trained_between."""
    out: list[str] = []
    for ev in events:
        if ev.get("kind") != "train_end":
            continue
        if start_ts < ev.get("ts", "") <= end_ts:
            sid = ev.get("script_id")
            if sid and sid not in out:
                out.append(sid)
    return out


def detect_regressions(
    script: TrainingScript,
    regression_results: dict[str, list[BenchmarkResult]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Walk every protected script's fresh results; for each benchmark
    that previously passed and now fails, build a regression_alert
    record naming the scripts trained in between."""
    alerts: list[dict[str, Any]] = []
    now = _ts()
    for protected_id, fresh in regression_results.items():
        for r in fresh:
            if r.passed:
                continue
            last_pass = _last_pass_for(protected_id, r.benchmark_label, events)
            if last_pass is None:
                # Was never green — not a regression, just hasn't passed yet.
                continue
            trained_between = _scripts_run_between(
                events, last_pass["ts"], now,
            )
            alerts.append({
                "kind":             "regression_alert",
                "newly_trained":    script.id,
                "broke_script":     protected_id,
                "broke_label":      r.benchmark_label,
                "broke_prompt":     r.prompt,
                "previous_score":   last_pass.get("score"),
                "current_score":    r.score,
                "current_response": r.response[:1000],
                "last_pass_ts":     last_pass.get("ts"),
                "trained_between":  trained_between,
            })
    return alerts


# ── High-level commands ────────────────────────────────────────────────────


def _emit_benchmark_event(script_id: str, r: BenchmarkResult,
                            category: str, phase: int, role: str) -> None:
    """role is 'self' for the script's own benchmarks, 'regression_check'
    for protected scripts' rerun benchmarks."""
    append_event({
        "kind":             "benchmark",
        "role":             role,
        "script_id":        script_id,
        "category":         category,
        "phase":            phase,
        "benchmark_label":  r.benchmark_label,
        "prompt":           r.prompt,
        "response":         r.response[:2000],
        "score":            r.score,
        "passed":           r.passed,
        "breakdown":        r.breakdown,
        "error":            r.error,
    })


def cmd_list(args: argparse.Namespace) -> int:
    scripts = load_registry(REGISTRY_DIR)
    by_phase = sorted(scripts.values(), key=lambda s: (s.phase, s.id))
    print(f"{'phase':>5}  {'category':<30}  id  ({len(scripts)} total)")
    for s in by_phase:
        print(f"{s.phase:>5}  {s.category:<30}  {s.id}  "
                f"benchmarks={len(s.benchmarks)} "
                f"protects={len(s.regression_protects)}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    scripts = load_registry(REGISTRY_DIR)
    if args.script_id not in scripts:
        print(f"unknown script: {args.script_id!r}", file=sys.stderr)
        return 2
    s = scripts[args.script_id]
    print(f"[benchmark] {s.id}  category={s.category}  phase={s.phase}")
    results = run_benchmarks(s)
    passed = sum(1 for r in results if r.passed)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        err  = f"  error={r.error}" if r.error else ""
        print(f"  [{flag}] score={r.score:.3f}  {r.benchmark_label}{err}")
        _emit_benchmark_event(s.id, r, s.category, s.phase, "self")
    print(f"[benchmark] {s.id}: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def cmd_benchmark_all(args: argparse.Namespace) -> int:
    scripts = load_registry(REGISTRY_DIR)
    grand_pass = 0
    grand_total = 0
    for s in sorted(scripts.values(), key=lambda s: (s.phase, s.id)):
        print(f"\n=== {s.id}  category={s.category} ===")
        results = run_benchmarks(s)
        for r in results:
            flag = "PASS" if r.passed else "FAIL"
            err = f"  error={r.error}" if r.error else ""
            print(f"  [{flag}] score={r.score:.3f}  {r.benchmark_label}{err}")
            _emit_benchmark_event(s.id, r, s.category, s.phase, "self")
        sp = sum(1 for r in results if r.passed)
        grand_pass += sp
        grand_total += len(results)
        print(f"  ({sp}/{len(results)} passed)")
    print(f"\nTOTAL: {grand_pass}/{grand_total} benchmarks passed")
    return 0 if grand_pass == grand_total else 1


def cmd_mark_trained(args: argparse.Namespace) -> int:
    scripts = load_registry(REGISTRY_DIR)
    if args.script_id not in scripts:
        print(f"unknown script: {args.script_id!r}", file=sys.stderr)
        return 2
    s = scripts[args.script_id]

    append_event({"kind": "train_start", "script_id": s.id,
                  "category": s.category, "phase": s.phase})
    # No actual training side-effect — operator runs the training
    # separately.  This command exists to bracket the training event
    # and to drive the benchmark + regression sweep afterwards.
    append_event({"kind": "train_end",   "script_id": s.id,
                  "category": s.category, "phase": s.phase})

    print(f"[mark-trained] {s.id} — running own benchmarks")
    own_results = run_benchmarks(s)
    for r in own_results:
        _emit_benchmark_event(s.id, r, s.category, s.phase, "self")
        flag = "PASS" if r.passed else "FAIL"
        print(f"  self  [{flag}] {r.score:.3f}  {r.benchmark_label}")

    regression_results: dict[str, list[BenchmarkResult]] = {}
    for rp in s.regression_protects:
        protected = scripts[rp.script_id]
        print(f"[mark-trained] {s.id} — re-running {protected.id} (regression check)")
        rr = run_benchmarks(protected)
        regression_results[protected.id] = rr
        for r in rr:
            _emit_benchmark_event(protected.id, r, protected.category,
                                    protected.phase, "regression_check")
            flag = "PASS" if r.passed else "FAIL"
            print(f"  reg   [{flag}] {r.score:.3f}  {protected.id}  {r.benchmark_label}")

    # Compare against history for regression alerts.
    events = read_events()
    alerts = detect_regressions(s, regression_results, events)
    for a in alerts:
        append_event(a)
        print(f"  !! REGRESSION  {a['broke_script']}::{a['broke_label']}  "
                f"{a['previous_score']} -> {a['current_score']}  "
                f"trained_between={a['trained_between']}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    events = read_events()
    tail = events[-args.tail:] if args.tail else events
    for ev in tail:
        kind = ev.get("kind", "?")
        if kind == "benchmark":
            print(f"{ev['ts']}  benchmark  {ev['script_id']:<30s}  "
                    f"role={ev.get('role')}  passed={ev.get('passed')}  "
                    f"score={ev.get('score')}  {ev.get('benchmark_label','')}")
        elif kind == "regression_alert":
            print(f"{ev['ts']}  !! REGRESSION  {ev['broke_script']}::"
                    f"{ev['broke_label']}  "
                    f"{ev['previous_score']} -> {ev['current_score']}  "
                    f"trained_between={ev['trained_between']}")
        else:
            print(f"{ev['ts']}  {kind:<10s}  {ev.get('script_id','')}")
    return 0


# ── argparse plumbing ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="training_standard.runner",
                                  description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registry + validate schema")

    pb = sub.add_parser("benchmark", help="Run one script's benchmarks")
    pb.add_argument("script_id")

    sub.add_parser("benchmark-all", help="Run every script's benchmarks")

    pmt = sub.add_parser("mark-trained",
                          help="Record train events + run own + regression benchmarks")
    pmt.add_argument("script_id")

    pr = sub.add_parser("report", help="Print event log tail")
    pr.add_argument("--tail", type=int, default=20)

    args = p.parse_args(argv)
    try:
        if args.cmd == "list":           return cmd_list(args)
        if args.cmd == "benchmark":      return cmd_benchmark(args)
        if args.cmd == "benchmark-all":  return cmd_benchmark_all(args)
        if args.cmd == "mark-trained":   return cmd_mark_trained(args)
        if args.cmd == "report":         return cmd_report(args)
    except SchemaError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
