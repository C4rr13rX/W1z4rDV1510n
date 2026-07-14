#!/usr/bin/env python3
"""Train independent raw-source fragments and execute a never-seen assembly."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from programming_project_eval import b64, request


def fragment(order: int, source: str) -> str:
    return json.dumps({"code_fragment": {"file": "clamp.py", "order": order,
                                          "source": source}},
                      sort_keys=True, separators=(",", ":"))


TRAINING = [
    ("Write a Python clamp function signature.",
     fragment(10, "def clamp(value, lower, upper):\n")),
    ("Implement a Python clamp lower-bound guard.",
     fragment(20, "    if value < lower:\n        return lower\n")),
    ("Implement a Python clamp upper-bound guard.",
     fragment(30, "    if value > upper:\n        return upper\n")),
    ("Implement Python clamp logic to otherwise return the input within the bounds.",
     fragment(40, "    return value\n")),
]

PROMPTS = [
    ("canonical", "Build a Python clamp function signature with a lower-bound guard, an upper-bound guard, and otherwise return the input within the bounds."),
    ("heldout", "Create a Python clamp function that floors values at a minimum, caps them at a maximum, and otherwise leaves the value unchanged."),
]

OOV = "Build a Python clamp using an unspecified custom comparison policy."


def train(endpoint: str, repeats: int) -> None:
    for _ in range(repeats):
        for prompt, response in TRAINING:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(response)})
            request(endpoint, "/brain/tick", {})


def execute(response: str) -> tuple[bool, str, str]:
    try:
        source = json.loads(response)["files"]["clamp.py"]
        if not isinstance(source, str):
            return False, "non-string source", ""
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}", ""
    harness = (source + "\nassert clamp(-2, 0, 10) == 0\n"
               "assert clamp(12, 0, 10) == 10\n"
               "assert clamp(7, 0, 10) == 7\nprint('PASS')\n")
    with tempfile.TemporaryDirectory(prefix="wv-fragment-") as raw:
        path = Path(raw) / "check.py"
        path.write_text(harness, encoding="utf-8")
        run = subprocess.run(["python", "-I", str(path)], capture_output=True,
                             text=True, timeout=15)
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-600:], source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/fragment_synthesis.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    rows = []
    fragment_sources = {json.loads(response)["code_fragment"]["source"]
                        for _, response in TRAINING}
    for kind, prompt in PROMPTS:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        passed, detail, source = execute(str(result.get("reply") or ""))
        rows.append({"kind": kind, "executes": passed,
                     "novel_whole": bool(source) and source not in fragment_sources,
                     "source": source, "detail": "" if passed else detail})
    unknown = request(args.endpoint, "/brain/chat", {"text": OOV})
    honest = not unknown.get("reply") and bool((unknown.get("grounding") or {}).get("outside_grounding"))
    report = {"passed": sum(r["executes"] and r["novel_whole"] for r in rows),
              "total": len(rows), "oov_honest": honest, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "oov_honest": honest}))
    return 0 if report["passed"] == report["total"] and honest else 1


if __name__ == "__main__":
    raise SystemExit(main())
