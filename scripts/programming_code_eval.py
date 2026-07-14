#!/usr/bin/env python3
"""Evaluate Python retrieval, syntax, execution, and prompt generalization."""
from __future__ import annotations

import argparse
import ast
import http.client
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/training/code_gen_python_001.jsonl"

CASES = [
    ("def square(", "square", [7], 49,
     "Create a Python function named square that computes a number times itself."),
    ("def is_negative(", "is_negative", [-3], True,
     "Create Python code that tells whether a supplied number is below zero."),
    ("def avg_list(", "avg_list", [[2, 4, 6]], 4.0,
     "Write Python that calculates the arithmetic mean of a list, returning zero for an empty list."),
    ("def filter_odd(", "filter_odd", [[1, 2, 3, 4, 5]], [1, 3, 5],
     "Build a Python function which keeps only odd integers from an input list."),
    ("def word_freq(", "word_freq", ["a b a"], {"a": 2, "b": 1},
     "Produce a Python function mapping every whitespace-separated word to its occurrence count."),
]


class Client:
    def __init__(self, endpoint: str) -> None:
        url = urlparse(endpoint)
        self.prefix = url.path.rstrip("/")
        self.conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=60)

    def chat(self, prompt: str) -> str:
        self.conn.request("POST", f"{self.prefix}/brain/chat", json.dumps({"text": prompt}),
                          {"Content-Type": "application/json"})
        response = self.conn.getresponse()
        payload = response.read()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {payload[:300]!r}")
        return str(json.loads(payload).get("reply") or "")


def load_examples() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def syntax_valid(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def executes(code: str, function: str, args: list, expected: object) -> tuple[bool, str]:
    if not syntax_valid(code):
        return False, "invalid_syntax"
    assertion = (
        f"\nimport json\n_result={function}(*json.loads({json.dumps(args)!r}))"
        f"\n_expected=json.loads({json.dumps(expected)!r})"
        "\nassert _result == _expected, (_result, _expected)\n"
    )
    try:
        run = subprocess.run(
            [sys.executable, "-I", "-c", code + assertion],
            capture_output=True, text=True, timeout=3,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return run.returncode == 0, (run.stderr.strip()[-300:] if run.returncode else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8291")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    rows = load_examples()
    client = Client(args.endpoint)
    results = []
    for response_prefix, function, call_args, expected, novel_prompt in CASES:
        exemplar = next(row for row in rows if str(row.get("response", "")).startswith(response_prefix))
        for kind, prompt in (("trained", exemplar["prompt"]), ("novel_paraphrase", novel_prompt)):
            reply = client.chat(prompt)
            ran, error = executes(reply, function, call_args, expected)
            results.append({
                "kind": kind, "function": function, "prompt": prompt,
                "nonempty": bool(reply), "syntax_valid": syntax_valid(reply),
                "executes": ran, "exact_training_response": reply == exemplar["response"],
                "error": error,
            })
    summary = {}
    for kind in ("trained", "novel_paraphrase"):
        group = [row for row in results if row["kind"] == kind]
        summary[kind] = {
            "count": len(group),
            "nonempty": sum(row["nonempty"] for row in group),
            "syntax_valid": sum(row["syntax_valid"] for row in group),
            "executes": sum(row["executes"] for row in group),
            "exact_training_response": sum(row["exact_training_response"] for row in group),
        }
    report: dict[str, object] = {"summary": summary}
    if args.details:
        report["results"] = results
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
