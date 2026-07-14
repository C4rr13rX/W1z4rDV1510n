#!/usr/bin/env python3
"""Sample a JSONL curriculum window and verify atom-grounded code recall."""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from programming_project_eval import request


def read_window(path: Path, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    valid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if valid < start:
                valid += 1
                continue
            rows.append(row)
            valid += 1
            if len(rows) >= count:
                break
    return rows


def sampled(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    indexes = [(i * len(rows)) // count for i in range(count)]
    return [rows[index] for index in indexes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--window-rows", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    rows = read_window(args.corpus, args.start_row, args.window_rows)
    probes = sampled(rows, min(args.samples, len(rows)))
    accepted_by_prompt: dict[str, set[str]] = {}
    for row in rows:
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        expected = (row.get("response") or row.get("answer") or "").strip()
        if prompt and expected:
            accepted_by_prompt.setdefault(prompt, set()).add(expected)
    exact = accepted = syntax = nonempty = 0
    elapsed: list[float] = []
    failures: list[dict] = []
    for row in probes:
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        expected = (row.get("response") or row.get("answer") or "").strip()
        if not prompt or not expected:
            continue
        started = time.perf_counter()
        response = request(args.endpoint, "/brain/chat", {"text": prompt})
        elapsed.append(time.perf_counter() - started)
        actual = (response.get("reply") or response.get("answer") or "").strip()
        nonempty += bool(actual)
        exact += actual == expected
        accepted += actual in accepted_by_prompt.get(prompt, set())
        try:
            ast.parse(actual)
            syntax += 1
        except SyntaxError:
            pass
        if actual not in accepted_by_prompt.get(prompt, set()):
            failures.append({
                "prompt": prompt[:160],
                "decoder": response.get("decoder"),
                "reply_length": len(actual),
                "expected_length": len(expected),
            })

    report = {
        "window": {"start": args.start_row, "rows": len(rows)},
        "sampled": len(probes),
        "nonempty": nonempty,
        "exact": exact,
        "accepted_trained_response": accepted,
        "python_syntax_valid": syntax,
        "latency_seconds": {
            "mean": round(sum(elapsed) / len(elapsed), 4) if elapsed else 0.0,
            "max": round(max(elapsed), 4) if elapsed else 0.0,
        },
        "failures": failures[:5],
    }
    print(json.dumps(report))
    return 0 if probes and accepted == len(probes) and syntax == len(probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
