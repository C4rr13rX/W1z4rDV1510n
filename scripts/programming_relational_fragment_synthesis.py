#!/usr/bin/env python3
"""Synthesize never-observed JavaScript from grounded dependency fragments."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from programming_project_eval import b64, request


def fragment(role: str, after: list[str], source: str) -> str:
    return json.dumps({"code_fragment": {"file": "clamp.js", "role": role,
                                          "after": after, "source": source}},
                      sort_keys=True, separators=(",", ":"))


TRAINING = [
    ("Write a JavaScript clamp function signature.",
     fragment("signature", [], "function clamp(value, lower, upper) {\n")),
    ("Implement a JavaScript clamp lower-bound guard.",
     fragment("lower_guard", ["signature"],
              "  if (value < lower) return lower;\n")),
    ("Implement a JavaScript clamp upper-bound guard.",
     fragment("upper_guard", ["signature", "lower_guard"],
              "  if (value > upper) return upper;\n")),
    ("Implement JavaScript clamp logic to otherwise return the input within the bounds.",
     fragment("return", ["signature", "lower_guard", "upper_guard"],
              "  return value;\n}\nmodule.exports = { clamp };\n")),
]

PROMPTS = [
    ("canonical", "Build a JavaScript clamp function signature with a lower-bound guard, an upper-bound guard, and otherwise return the input within the bounds."),
    ("heldout", "Create a Node.js clamp function that floors values at a minimum, caps them at a maximum, and otherwise leaves the value unchanged."),
]

OOV = "Create a JavaScript clamp using an unspecified locale-specific comparison policy."


def train(endpoint: str, repeats: int) -> None:
    for _ in range(repeats):
        for prompt, response in TRAINING:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(response)})
            request(endpoint, "/brain/tick", {})


def execute(response: str) -> tuple[bool, str, str]:
    try:
        source = json.loads(response)["files"]["clamp.js"]
        if not isinstance(source, str):
            return False, "non-string source", ""
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}", ""
    harness = (source + "\nconst assert=require('assert');\n"
               "assert.strictEqual(clamp(-2,0,10),0);\n"
               "assert.strictEqual(clamp(12,0,10),10);\n"
               "assert.strictEqual(clamp(7,0,10),7);\nconsole.log('PASS');\n")
    with tempfile.TemporaryDirectory(prefix="wv-relative-fragment-") as raw:
        path = Path(raw) / "check.js"
        path.write_text(harness, encoding="utf-8")
        run = subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=15)
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-600:], source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/relational_fragment_synthesis.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    observed = {json.loads(value)["code_fragment"]["source"] for _, value in TRAINING}
    rows = []
    for kind, prompt in PROMPTS:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        passed, detail, source = execute(str(result.get("reply") or ""))
        rows.append({"kind": kind, "executes": passed,
                     "novel_whole": bool(source) and source not in observed,
                     "source": source, "detail": "" if passed else detail})
    unknown = request(args.endpoint, "/brain/chat", {"text": OOV})
    honest = not unknown.get("reply") and bool((unknown.get("grounding") or {}).get("outside_grounding"))
    report = {"passed": sum(row["executes"] and row["novel_whole"] for row in rows),
              "total": len(rows), "oov_honest": honest, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "oov_honest": honest}))
    return 0 if report["passed"] == report["total"] and honest else 1


if __name__ == "__main__":
    raise SystemExit(main())
