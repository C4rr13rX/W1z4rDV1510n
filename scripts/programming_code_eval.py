#!/usr/bin/env python3
"""Evaluate Python retrieval, syntax, execution, and prompt generalization."""
from __future__ import annotations

import argparse
import ast
import base64
import http.client
import json
import re
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

    def post(self, path: str, payload: dict) -> dict:
        self.conn.request("POST", f"{self.prefix}{path}", json.dumps(payload),
                          {"Content-Type": "application/json"})
        response = self.conn.getresponse()
        payload = response.read()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {payload[:300]!r}")
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"{path} returned non-object payload: {type(decoded).__name__}"
            )
        return decoded

    def chat_payload(self, prompt: str) -> dict:
        return self.post("/brain/chat", {"text": prompt})

    def chat(self, prompt: str) -> str:
        return str(self.chat_payload(prompt).get("reply") or "")


def load_examples() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def refresh_routes(client: Client, rows: list[dict], repeats: int) -> None:
    """Re-advertise the existing protected bindings without new answers."""
    exemplars = [
        next(row for row in rows if str(row.get("response", "")).startswith(prefix))
        for prefix, *_rest in CASES
    ]
    for _ in range(repeats):
        for exemplar in exemplars:
            prompt = str(exemplar["prompt"])
            response = str(exemplar["response"])
            client.post("/brain/observe", {"pool_id": 1, "frame": b64(prompt)})
            client.post("/brain/observe", {"pool_id": 12, "frame": b64(prompt)})
            client.post("/brain/observe", {"pool_id": 4, "frame": b64(response)})
            client.post("/brain/tick", {})


def syntax_valid(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def prompt_requires_function(prompt: str, function: str) -> bool:
    """Whether the public request constrains the generated identifier."""
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(function)}(?![A-Za-z0-9_])",
        prompt,
        flags=re.IGNORECASE,
    ) is not None


def callable_candidates(code: str, requested: str, prompt: str) -> list[str]:
    if not syntax_valid(code):
        return []
    if prompt_requires_function(prompt, requested):
        return [requested]
    tree = ast.parse(code)
    names = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # Prefer the learned canonical identifier when it is present, but a prompt
    # that did not name it cannot require the model to guess that hidden name.
    return ([requested] if requested in names else []) + [
        name for name in names if name != requested
    ]


def executes(code: str, function: str, args: list, expected: object,
             prompt: str = "") -> tuple[bool, str, str]:
    if not syntax_valid(code):
        return False, "invalid_syntax", ""
    candidates = callable_candidates(code, function, prompt)
    if not candidates:
        return False, "no_top_level_function", ""
    errors = []
    for candidate in candidates:
        assertion = (
            f"\nimport json\n_result={candidate}(*json.loads({json.dumps(args)!r}))"
            f"\n_expected=json.loads({json.dumps(expected)!r})"
            "\nassert _result == _expected, (_result, _expected)\n"
        )
        try:
            run = subprocess.run(
                [sys.executable, "-I", "-c", code + assertion],
                capture_output=True, text=True, timeout=3,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{candidate}: timeout")
            continue
        if run.returncode == 0:
            return True, "", candidate
        errors.append(f"{candidate}: {run.stderr.strip()[-240:]}")
    return False, " | ".join(errors)[-500:], ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8291")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_examples()
    client = Client(args.endpoint)
    if not args.no_train:
        refresh_routes(client, rows, args.repeats)
    results = []
    for response_prefix, function, call_args, expected, novel_prompt in CASES:
        exemplar = next(row for row in rows if str(row.get("response", "")).startswith(response_prefix))
        for kind, prompt in (("trained", exemplar["prompt"]), ("novel_paraphrase", novel_prompt)):
            payload = client.chat_payload(prompt)
            reply = str(payload.get("reply") or "")
            ran, error, invoked_function = executes(
                reply, function, call_args, expected, str(prompt)
            )
            results.append({
                "kind": kind, "function": function, "prompt": prompt,
                "nonempty": bool(reply), "syntax_valid": syntax_valid(reply),
                "executes": ran, "exact_training_response": reply == exemplar["response"],
                "reply": reply,
                "expected_training_response": exemplar["response"],
                "call_args": call_args,
                "expected_result": expected,
                "error": error,
                "invoked_function": invoked_function,
                "route": {
                    "decoder": payload.get("decoder"),
                    "grounding": payload.get("grounding"),
                    "semantic_refinement_score": payload.get("semantic_refinement_score"),
                    "semantic_refinement_margin": payload.get("semantic_refinement_margin"),
                    "intent_diagnostics": payload.get("intent_diagnostics"),
                    "paged_neurons_released": payload.get("paged_neurons_released"),
                },
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
    encoded = json.dumps(report, separators=(",", ":"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(encoded)
    return 0 if all(row["executes"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
