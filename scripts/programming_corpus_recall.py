#!/usr/bin/env python3
"""Sample a JSONL curriculum window and verify atom-grounded code recall."""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from programming_project_eval import request


def sample_window(path: Path, start: int, count: int,
                  sample_count: int) -> tuple[list[dict], int]:
    """Select evenly distributed logical rows with O(sample_count) memory."""
    if count <= 0 or sample_count <= 0:
        return [], 0
    target_count = min(count, sample_count)
    if target_count == 1:
        targets = [0]
    else:
        # Cover the entire trained interval, including its newest row. Using
        # i*count/target_count leaves the final 1/N segment unprobed and can
        # miss a tail-only durability or interference failure.
        targets = [
            (i * (count - 1)) // (target_count - 1)
            for i in range(target_count)
        ]
    target_cursor = 0
    probes: list[dict] = []
    valid = 0
    window_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if valid < start:
                valid += 1
                continue
            relative = valid - start
            if target_cursor < len(targets) and relative == targets[target_cursor]:
                probes.append(row)
                target_cursor += 1
            valid += 1
            window_rows += 1
            if window_rows >= count:
                break
    return probes, window_rows


def accepted_responses(paths: list[Path], prompts: set[str]) -> dict[str, set[str]]:
    """Collect every supervised answer for only the sampled prompt set."""
    accepted: dict[str, set[str]] = {prompt: set() for prompt in prompts}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (row.get("prompt") or row.get("question") or "").strip()
                if prompt not in accepted:
                    continue
                response = (row.get("response") or row.get("answer") or "").strip()
                if response:
                    accepted[prompt].add(response)
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--window-rows", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--accepted-corpus", action="append", type=Path, default=[],
        help="also accept supervised answers from this durably trained corpus",
    )
    parser.add_argument("--syntax", choices=("python", "none"), default="python")
    args = parser.parse_args()

    probes, window_rows = sample_window(
        args.corpus, args.start_row, args.window_rows, args.samples
    )
    probe_prompts = {
        (row.get("prompt") or row.get("question") or "").strip()
        for row in probes
    }
    probe_prompts.discard("")
    accepted_paths = list(dict.fromkeys([args.corpus, *args.accepted_corpus]))
    accepted_by_prompt = accepted_responses(accepted_paths, probe_prompts)
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
        if args.syntax == "none":
            syntax += 1
        else:
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
        "window": {"start": args.start_row, "rows": window_rows},
        "accepted_corpora": [str(path) for path in accepted_paths],
        "sampled": len(probes),
        "nonempty": nonempty,
        "exact": exact,
        "accepted_trained_response": accepted,
        "syntax_mode": args.syntax,
        "syntax_valid": syntax,
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
