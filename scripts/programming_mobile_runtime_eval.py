#!/usr/bin/env python3
"""Measure phone-class cold/warm inference and neuron-scoped residency.

Run this only against an idle, fully qualified programming brain. Each cold
trial first completes neuron-wise sleep, then measures one ordinary request,
an immediate warm repeat, the scope paged into RAM, deterministic output, and
the return to zero residency. The benchmark never trains.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


CASES = (
    (
        "categorical-recall",
        "dog",
        lambda reply: reply.strip().casefold() == "animal",
    ),
    (
        "trained-small-code",
        "Python: define square(n) -- return n squared.",
        lambda reply: "def square" in reply and "return n * n" in reply,
    ),
    (
        "unseen-small-code-paraphrase",
        "Create a Python function named square that computes a number times itself.",
        lambda reply: "def square" in reply and "return n * n" in reply,
    ),
)

STABLE_FIELDS = (
    "tick",
    "pool_count",
    "total_neurons",
    "total_concepts",
    "total_binding",
    "binding_pool_id",
)


def request(endpoint: str, path: str, payload: dict | None = None,
            timeout: float = 300.0) -> dict:
    url = endpoint.rstrip("/") + path
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    method = "GET" if body is None else "POST"
    call = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    with urllib.request.urlopen(call, timeout=timeout) as response:
        return json.loads(response.read())


def stats(endpoint: str) -> dict:
    return request(endpoint, "/brain/stats", timeout=120.0)


def sleep_brain(endpoint: str) -> dict:
    result = request(
        endpoint,
        "/brain/sleep",
        {"min_use_count": 2, "stale_ticks": 1000},
        timeout=4 * 3600.0,
    )
    if result.get("error"):
        raise RuntimeError(f"brain sleep failed: {result}")
    return result


def signature(value: dict) -> dict:
    return {field: value.get(field) for field in STABLE_FIELDS}


def timed_chat(endpoint: str, prompt: str) -> tuple[dict, float]:
    started = time.perf_counter()
    response = request(
        endpoint, "/brain/chat", {"text": prompt}, timeout=300.0
    )
    return response, time.perf_counter() - started


def summarize_trials(
        trials: list[dict], initial: dict, final: dict,
        max_cold_seconds: float, max_warm_seconds: float,
        max_resident_fraction: float) -> dict:
    total_terminals = max(
        1,
        int(initial.get("total_terminals") or 0),
        *(int(row["after_warm"].get("total_terminals") or 0)
          for row in trials),
    )
    peak_resident = max(
        (max(
            int(row["after_cold"].get("resident_terminals") or 0),
            int(row["after_warm"].get("resident_terminals") or 0),
        ) for row in trials),
        default=0,
    )
    peak_fraction = peak_resident / total_terminals
    checks = {
        "correct": all(
            row["cold_correct"] and row["warm_correct"] for row in trials
        ),
        "deterministic": all(row["deterministic"] for row in trials),
        "cold_latency": all(
            row["cold_seconds"] <= max_cold_seconds for row in trials
        ),
        "warm_latency": all(
            row["warm_seconds"] <= max_warm_seconds for row in trials
        ),
        "bounded_residency": peak_fraction <= max_resident_fraction,
        "zero_between_trials": all(
            int(row["before_cold"].get("resident_terminals") or 0) == 0
            and int(row["after_resleep"].get("resident_terminals") or 0) == 0
            for row in trials
        ),
        "read_only_identity": signature(initial) == signature(final),
        "final_zero_residency": (
            int(final.get("resident_terminals") or 0) == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "limits": {
            "max_cold_seconds": max_cold_seconds,
            "max_warm_seconds": max_warm_seconds,
            "max_resident_fraction": max_resident_fraction,
        },
        "observed": {
            "max_cold_seconds": max(
                (row["cold_seconds"] for row in trials), default=0.0
            ),
            "max_warm_seconds": max(
                (row["warm_seconds"] for row in trials), default=0.0
            ),
            "peak_resident_terminals": peak_resident,
            "total_terminals": total_terminals,
            "peak_resident_fraction": peak_fraction,
        },
        "initial": initial,
        "final": final,
        "trials": trials,
        "telemetry_limitations": [
            "The current node API does not expose per-request SSD bytes read.",
            "The current node API does not expose focus-chain depth.",
            "Resident-terminal delta is the available neuron-scope proxy.",
        ],
        "updated_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runtime/benchmarks/mobile-runtime.json"),
    )
    parser.add_argument("--max-cold-seconds", type=float, default=1.0)
    parser.add_argument("--max-warm-seconds", type=float, default=0.5)
    parser.add_argument("--max-resident-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.max_cold_seconds <= 0 or args.max_warm_seconds <= 0:
        parser.error("latency limits must be positive")
    if not 0 < args.max_resident_fraction <= 1:
        parser.error("--max-resident-fraction must be in (0, 1]")

    sleep_brain(args.endpoint)
    initial = stats(args.endpoint)
    if int(initial.get("resident_terminals") or 0) != 0:
        raise RuntimeError(
            "mobile benchmark could not establish zero residency"
        )

    trials: list[dict] = []
    for name, prompt, correct in CASES:
        sleep_brain(args.endpoint)
        before_cold = stats(args.endpoint)
        cold, cold_seconds = timed_chat(args.endpoint, prompt)
        after_cold = stats(args.endpoint)
        warm, warm_seconds = timed_chat(args.endpoint, prompt)
        after_warm = stats(args.endpoint)
        sleep_brain(args.endpoint)
        after_resleep = stats(args.endpoint)
        cold_reply = str(cold.get("reply") or "")
        warm_reply = str(warm.get("reply") or "")
        trials.append({
            "name": name,
            "prompt": prompt,
            "cold_seconds": round(cold_seconds, 6),
            "warm_seconds": round(warm_seconds, 6),
            "cold_reply": cold_reply,
            "warm_reply": warm_reply,
            "cold_correct": bool(correct(cold_reply)),
            "warm_correct": bool(correct(warm_reply)),
            "deterministic": cold_reply == warm_reply,
            "before_cold": before_cold,
            "after_cold": after_cold,
            "after_warm": after_warm,
            "after_resleep": after_resleep,
            "intent_diagnostics": cold.get("intent_diagnostics") or {},
        })

    final = stats(args.endpoint)
    report = summarize_trials(
        trials, initial, final,
        args.max_cold_seconds, args.max_warm_seconds,
        args.max_resident_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "passed": report["passed"],
        "checks": report["checks"],
        "observed": report["observed"],
    }))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
